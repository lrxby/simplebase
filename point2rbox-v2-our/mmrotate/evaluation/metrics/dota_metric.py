# # Copyright (c) OpenMMLab. All rights reserved.
# import copy
# import os
# import os.path as osp
# import re
# import tempfile
# import zipfile
# from collections import OrderedDict, defaultdict
# from typing import List, Optional, Sequence, Union

# import numpy as np
# import torch
# from mmcv.ops import nms_quadri, nms_rotated
# from mmengine.evaluator import BaseMetric
# from mmengine.fileio import dump
# from mmengine.logging import MMLogger

# from mmrotate.evaluation import eval_rbbox_map
# from mmrotate.registry import METRICS
# from mmrotate.structures.bbox import rbox2qbox


# @METRICS.register_module()
# class DOTAMetric(BaseMetric):
#     """DOTA evaluation metric.

#     Note:  In addition to format the output results to JSON like CocoMetric,
#     it can also generate the full image's results by merging patches' results.
#     The premise is that you must use the tool provided by us to crop the DOTA
#     large images, which can be found at: ``tools/data/dota/split``.

#     Args:
#         iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
#         scale_ranges (List[tuple], optional): Scale ranges for evaluating
#             mAP. If not specified, all bounding boxes would be included in
#             evaluation. Defaults to None.
#         metric (str | list[str]): Metrics to be evaluated. Only support
#             'mAP' now. If is list, the first setting in the list will
#              be used to evaluate metric.
#         predict_box_type (str): Box type of model results. If the QuadriBoxes
#             is used, you need to specify 'qbox'. Defaults to 'rbox'.
#         format_only (bool): Format the output results without perform
#             evaluation. It is useful when you want to format the result
#             to a specific format. Defaults to False.
#         outfile_prefix (str, optional): The prefix of json/zip files. It
#             includes the file path and the prefix of filename, e.g.,
#             "a/b/prefix". If not specified, a temp file will be created.
#             Defaults to None.
#         merge_patches (bool): Generate the full image's results by merging
#             patches' results.
#         iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
#             patches. Defaults to 0.1.
#         eval_mode (str): 'area' or '11points', 'area' means calculating the
#             area under precision-recall curve, '11points' means calculating
#             the average precision of recalls at [0, 0.1, ..., 1].
#             The PASCAL VOC2007 defaults to use '11points', while PASCAL
#             VOC2012 defaults to use 'area'. Defaults to '11points'.
#         collect_device (str): Device name used for collecting results from
#             different ranks during distributed training. Must be 'cpu' or
#             'gpu'. Defaults to 'cpu'.
#         prefix (str, optional): The prefix that will be added in the metric
#             names to disambiguate homonymous metrics of different evaluators.
#             If prefix is not provided in the argument, self.default_prefix
#             will be used instead. Defaults to None.
#     """

#     default_prefix: Optional[str] = 'dota'

#     def __init__(self,
#                  iou_thrs: Union[float, List[float]] = 0.5,
#                  scale_ranges: Optional[List[tuple]] = None,
#                  metric: Union[str, List[str]] = 'mAP',
#                  predict_box_type: str = 'rbox',
#                  format_only: bool = False,
#                  outfile_prefix: Optional[str] = None,
#                  merge_patches: bool = False,
#                  iou_thr: float = 0.1,
#                  eval_mode: str = '11points',
#                  collect_device: str = 'cpu',
#                  prefix: Optional[str] = None) -> None:
#         super().__init__(collect_device=collect_device, prefix=prefix)
#         self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
#             else iou_thrs
#         assert isinstance(self.iou_thrs, list)
#         self.scale_ranges = scale_ranges
#         # voc evaluation metrics
#         if not isinstance(metric, str):
#             assert len(metric) == 1
#             metric = metric[0]
#         allowed_metrics = ['mAP']
#         if metric not in allowed_metrics:
#             raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
#         self.metric = metric
#         self.predict_box_type = predict_box_type

#         self.format_only = format_only
#         if self.format_only:
#             assert outfile_prefix is not None, 'outfile_prefix must be not'
#             'None when format_only is True, otherwise the result files will'
#             'be saved to a temp directory which will be cleaned up at the end.'

#         self.outfile_prefix = outfile_prefix
#         self.merge_patches = merge_patches
#         self.iou_thr = iou_thr

#         self.use_07_metric = True if eval_mode == '11points' else False

#     def merge_results(self, results: Sequence[dict],
#                       outfile_prefix: str) -> str:
#         """Merge patches' predictions into full image's results and generate a
#         zip file for DOTA online evaluation.

#         You can submit it at:
#         https://captain-whu.github.io/DOTA/evaluation.html

#         Args:
#             results (Sequence[dict]): Testing results of the
#                 dataset.
#             outfile_prefix (str): The filename prefix of the zip files. If the
#                 prefix is "somepath/xxx", the zip files will be named
#                 "somepath/xxx/xxx.zip".
#         """
#         collector = defaultdict(list)

#         for idx, result in enumerate(results):
#             img_id = result.get('img_id', idx)
#             splitname = img_id.split('__')
#             oriname = splitname[0]
#             pattern1 = re.compile(r'__\d+___\d+')
#             x_y = re.findall(pattern1, img_id)
#             x_y_2 = re.findall(r'\d+', x_y[0])
#             x, y = int(x_y_2[0]), int(x_y_2[1])
#             labels = result['labels']
#             bboxes = result['bboxes']
#             scores = result['scores']
#             ori_bboxes = bboxes.copy()
#             if self.predict_box_type == 'rbox':
#                 ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
#                     [x, y], dtype=np.float32)
#             elif self.predict_box_type == 'qbox':
#                 ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
#                     [x, y, x, y, x, y, x, y], dtype=np.float32)
#             else:
#                 raise NotImplementedError
#             label_dets = np.concatenate(
#                 [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
#                 axis=1)
#             collector[oriname].append(label_dets)

#         id_list, dets_list = [], []
#         for oriname, label_dets_list in collector.items():
#             big_img_results = []
#             label_dets = np.concatenate(label_dets_list, axis=0)
#             labels, dets = label_dets[:, 0], label_dets[:, 1:]
#             for i in range(len(self.dataset_meta['classes'])):
#                 if len(dets[labels == i]) == 0:
#                     big_img_results.append(dets[labels == i])
#                 else:
#                     try:
#                         cls_dets = torch.from_numpy(dets[labels == i]).cuda()
#                     except:  # noqa: E722
#                         cls_dets = torch.from_numpy(dets[labels == i])
#                     if self.predict_box_type == 'rbox':
#                         nms_dets, _ = nms_rotated(cls_dets[:, :5],
#                                                   cls_dets[:,
#                                                            -1], self.iou_thr)
#                     elif self.predict_box_type == 'qbox':
#                         nms_dets, _ = nms_quadri(cls_dets[:, :8],
#                                                  cls_dets[:, -1], self.iou_thr)
#                     else:
#                         raise NotImplementedError
#                     big_img_results.append(nms_dets.cpu().numpy())
#             id_list.append(oriname)
#             dets_list.append(big_img_results)

#         if osp.exists(outfile_prefix):
#             raise ValueError(f'The outfile_prefix should be a non-exist path, '
#                              f'but {outfile_prefix} is existing. '
#                              f'Please delete it firstly.')
#         os.makedirs(outfile_prefix)

#         files = [
#             osp.join(outfile_prefix, 'Task1_' + cls + '.txt')
#             for cls in self.dataset_meta['classes']
#         ]
#         file_objs = [open(f, 'w') for f in files]
#         for img_id, dets_per_cls in zip(id_list, dets_list):
#             for f, dets in zip(file_objs, dets_per_cls):
#                 if dets.size == 0:
#                     continue
#                 th_dets = torch.from_numpy(dets)
#                 if self.predict_box_type == 'rbox':
#                     rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
#                     qboxes = rbox2qbox(rboxes)
#                 elif self.predict_box_type == 'qbox':
#                     qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
#                 else:
#                     raise NotImplementedError
#                 for qbox, score in zip(qboxes, scores):
#                     txt_element = [img_id, str(round(float(score), 2))
#                                    ] + [f'{p:.2f}' for p in qbox]
#                     f.writelines(' '.join(txt_element) + '\n')

#         for f in file_objs:
#             f.close()

#         target_name = osp.split(outfile_prefix)[-1]
#         zip_path = osp.join(outfile_prefix, target_name + '.zip')
#         with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
#             for f in files:
#                 t.write(f, osp.split(f)[-1])

#         return zip_path

#     def results2json(self, results: Sequence[dict],
#                      outfile_prefix: str) -> dict:
#         """Dump the detection results to a COCO style json file.

#         There are 3 types of results: proposals, bbox predictions, mask
#         predictions, and they have different data types. This method will
#         automatically recognize the type, and dump them to json files.

#         Args:
#             results (Sequence[dict]): Testing results of the
#                 dataset.
#             outfile_prefix (str): The filename prefix of the json files. If the
#                 prefix is "somepath/xxx", the json files will be named
#                 "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
#                 "somepath/xxx.proposal.json".

#         Returns:
#             dict: Possible keys are "bbox", "segm", "proposal", and
#             values are corresponding filenames.
#         """
#         bbox_json_results = []
#         for idx, result in enumerate(results):
#             image_id = result.get('img_id', idx)
#             labels = result['labels']
#             bboxes = result['bboxes']
#             scores = result['scores']
#             # bbox results
#             for i, label in enumerate(labels):
#                 data = dict()
#                 data['image_id'] = image_id
#                 data['bbox'] = bboxes[i].tolist()
#                 data['score'] = float(scores[i])
#                 label = int(label) if label.ndim == 0 else label.tolist()
#                 data['category_id'] = label
#                 bbox_json_results.append(data)

#         result_files = dict()
#         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
#         dump(bbox_json_results, result_files['bbox'])

#         return result_files

#     def process(self, data_batch: Sequence[dict],
#                 data_samples: Sequence[dict]) -> None:
#         """Process one batch of data samples and predictions. The processed
#         results should be stored in ``self.results``, which will be used to
#         compute the metrics when all batches have been processed.

#         Args:
#             data_batch (dict): A batch of data from the dataloader.
#             data_samples (Sequence[dict]): A batch of data samples that
#                 contain annotations and predictions.
#         """
#         for data_sample in data_samples:
#             gt = copy.deepcopy(data_sample)
#             gt_instances = gt['gt_instances']
#             gt_ignore_instances = gt['ignored_instances']
#             if gt_instances == {}:
#                 ann = dict()
#             else:
#                 ann = dict(
#                     labels=gt_instances['labels'].cpu().numpy(),
#                     bboxes=gt_instances['bboxes'].cpu().numpy(),
#                     bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
#                     labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
#             result = dict()
#             pred = data_sample['pred_instances']
#             result['img_id'] = data_sample['img_id']
#             result['bboxes'] = pred['bboxes'].cpu().numpy()
#             result['scores'] = pred['scores'].cpu().numpy()
#             result['labels'] = pred['labels'].cpu().numpy()

#             result['pred_bbox_scores'] = []
#             for label in range(len(self.dataset_meta['classes'])):
#                 index = np.where(result['labels'] == label)[0]
#                 pred_bbox_scores = np.hstack([
#                     result['bboxes'][index], result['scores'][index].reshape(
#                         (-1, 1))
#                 ])
#                 result['pred_bbox_scores'].append(pred_bbox_scores)

#             self.results.append((ann, result))

#     def compute_metrics(self, results: list) -> dict:
#         """Compute the metrics from processed results.

#         Args:
#             results (list): The processed results of each batch.
#         Returns:
#             dict: The computed metrics. The keys are the names of the metrics,
#             and the values are corresponding results.
#         """
#         logger: MMLogger = MMLogger.get_current_instance()
#         gts, preds = zip(*results)

#         tmp_dir = None
#         if self.outfile_prefix is None:
#             tmp_dir = tempfile.TemporaryDirectory()
#             outfile_prefix = osp.join(tmp_dir.name, 'results')
#         else:
#             outfile_prefix = self.outfile_prefix

#         eval_results = OrderedDict()
#         if self.merge_patches:
#             # convert predictions to txt format and dump to zip file
#             zip_path = self.merge_results(preds, outfile_prefix)
#             logger.info(f'The submission file save at {zip_path}')
#             return eval_results
#         else:
#             # convert predictions to coco format and dump to json file
#             _ = self.results2json(preds, outfile_prefix)
#             if self.format_only:
#                 logger.info('results are saved in '
#                             f'{osp.dirname(outfile_prefix)}')
#                 return eval_results

#         if self.metric == 'mAP':
#             assert isinstance(self.iou_thrs, list)
#             dataset_name = self.dataset_meta['classes']
#             dets = [pred['pred_bbox_scores'] for pred in preds]

#             mean_aps = []
#             for iou_thr in self.iou_thrs:
#                 logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
#                 mean_ap, _ = eval_rbbox_map(
#                     dets,
#                     gts,
#                     scale_ranges=self.scale_ranges,
#                     iou_thr=iou_thr,
#                     use_07_metric=self.use_07_metric,
#                     box_type=self.predict_box_type,
#                     dataset=dataset_name,
#                     logger=logger)
#                 mean_aps.append(mean_ap)
#                 eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
#             eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
#             eval_results.move_to_end('mAP', last=False)
#         else:
#             raise NotImplementedError
#         return eval_results
# # Copyright (c) OpenMMLab. All rights reserved.
# import copy
# import os
# import os.path as osp
# import re
# import tempfile
# import zipfile
# from collections import OrderedDict, defaultdict
# from typing import List, Optional, Sequence, Union

# import numpy as np
# import torch
# from mmcv.ops import nms_quadri, nms_rotated
# from mmdet.structures.bbox import bbox_overlaps
# from mmengine.evaluator import BaseMetric
# from mmengine.fileio import dump
# from mmengine.logging import MMLogger
# from mmengine.logging import print_log # Added print_log import
# from terminaltables import AsciiTable # Added AsciiTable import

# from mmrotate.evaluation import eval_rbbox_map
# from mmrotate.registry import METRICS
# from mmrotate.structures.bbox import rbox2qbox


# @METRICS.register_module()
# class DOTAMetric(BaseMetric):
#     """DOTA evaluation metric.

#     Note:  In addition to format the output results to JSON like CocoMetric,
#     it can also generate the full image's results by merging patches' results.
#     The premise is that you must use the tool provided by us to crop the DOTA
#     large images, which can be found at: ``tools/data/dota/split``.

#     Args:
#         iou_thrs (float or List[float]): IoU threshold. Defaults to 0.5.
#         scale_ranges (List[tuple], optional): Scale ranges for evaluating
#             mAP. If not specified, all bounding boxes would be included in
#             evaluation. Defaults to None.
#         metric (str | list[str]): Metrics to be evaluated. Only support
#             'mAP' now. If is list, the first setting in the list will
#              be used to evaluate metric.
#         predict_box_type (str): Box type of model results. If the QuadriBoxes
#             is used, you need to specify 'qbox'. Defaults to 'rbox'.
#         format_only (bool): Format the output results without perform
#             evaluation. It is useful when you want to format the result
#             to a specific format. Defaults to False.
#         outfile_prefix (str, optional): The prefix of json/zip files. It
#             includes the file path and the prefix of filename, e.g.,
#             "a/b/prefix". If not specified, a temp file will be created.
#             Defaults to None.
#         merge_patches (bool): Generate the full image's results by merging
#             patches' results.
#         iou_thr (float): IoU threshold of ``nms_rotated`` used in merge
#             patches. Defaults to 0.1.
#         eval_mode (str): 'area' or '11points', 'area' means calculating the
#             area under precision-recall curve, '11points' means calculating
#             the average precision of recalls at [0, 0.1, ..., 1].
#             The PASCAL VOC2007 defaults to use '11points', while PASCAL
#             VOC2012 defaults to use 'area'. Defaults to '11points'.
#         collect_device (str): Device name used for collecting results from
#             different ranks during distributed training. Must be 'cpu' or
#             'gpu'. Defaults to 'cpu'.
#         prefix (str, optional): The prefix that will be added in the metric
#             names to disambiguate homonymous metrics of different evaluators.
#             If prefix is not provided in the argument, self.default_prefix
#             will be used instead. Defaults to None.
#     """

#     default_prefix: Optional[str] = 'dota'

#     def __init__(self,
#                  iou_thrs: Union[float, List[float]] = 0.5,
#                  scale_ranges: Optional[List[tuple]] = None,
#                  metric: Union[str, List[str]] = 'mAP',
#                  predict_box_type: str = 'rbox',
#                  format_only: bool = False,
#                  outfile_prefix: Optional[str] = None,
#                  merge_patches: bool = False,
#                  iou_thr: float = 0.1,
#                  eval_mode: str = '11points',
#                  collect_device: str = 'cpu',
#                  prefix: Optional[str] = None) -> None:
#         super().__init__(collect_device=collect_device, prefix=prefix)
#         self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
#             else iou_thrs
#         assert isinstance(self.iou_thrs, list)
#         self.scale_ranges = scale_ranges
#         # voc evaluation metrics
#         if not isinstance(metric, str):
#             assert len(metric) == 1
#             metric = metric[0]
#         allowed_metrics = ['mAP']
#         if metric not in allowed_metrics:
#             raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
#         self.metric = metric
#         self.predict_box_type = predict_box_type

#         self.format_only = format_only
#         if self.format_only:
#             assert outfile_prefix is not None, 'outfile_prefix must be not'
#             'None when format_only is True, otherwise the result files will'
#             'be saved to a temp directory which will be cleaned up at the end.'

#         self.outfile_prefix = outfile_prefix
#         self.merge_patches = merge_patches
#         self.iou_thr = iou_thr

#         self.use_07_metric = True if eval_mode == '11points' else False

#     def merge_results(self, results: Sequence[dict],
#                       outfile_prefix: str) -> str:
#         """Merge patches' predictions into full image's results and generate a
#         zip file for DOTA online evaluation.

#         You can submit it at:
#         https://captain-whu.github.io/DOTA/evaluation.html

#         Args:
#             results (Sequence[dict]): Testing results of the
#                 dataset.
#             outfile_prefix (str): The filename prefix of the zip files. If the
#                 prefix is "somepath/xxx", the zip files will be named
#                 "somepath/xxx/xxx.zip".
#         """
#         collector = defaultdict(list)

#         for idx, result in enumerate(results):
#             img_id = result.get('img_id', idx)
#             splitname = img_id.split('__')
#             oriname = splitname[0]
#             pattern1 = re.compile(r'__\d+___\d+')
#             x_y = re.findall(pattern1, img_id)
#             x_y_2 = re.findall(r'\d+', x_y[0])
#             x, y = int(x_y_2[0]), int(x_y_2[1])
#             labels = result['labels']
#             bboxes = result['bboxes']
#             scores = result['scores']
#             ori_bboxes = bboxes.copy()
#             if self.predict_box_type == 'rbox':
#                 ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
#                     [x, y], dtype=np.float32)
#             elif self.predict_box_type == 'qbox':
#                 ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
#                     [x, y, x, y, x, y, x, y], dtype=np.float32)
#             else:
#                 raise NotImplementedError
#             label_dets = np.concatenate(
#                 [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
#                 axis=1)
#             collector[oriname].append(label_dets)

#         id_list, dets_list = [], []
#         for oriname, label_dets_list in collector.items():
#             big_img_results = []
#             label_dets = np.concatenate(label_dets_list, axis=0)
#             labels, dets = label_dets[:, 0], label_dets[:, 1:]
#             for i in range(len(self.dataset_meta['classes'])):
#                 if len(dets[labels == i]) == 0:
#                     big_img_results.append(dets[labels == i])
#                 else:
#                     try:
#                         cls_dets = torch.from_numpy(dets[labels == i]).cuda()
#                     except:  # noqa: E722
#                         cls_dets = torch.from_numpy(dets[labels == i])
#                     if self.predict_box_type == 'rbox':
#                         nms_dets, _ = nms_rotated(cls_dets[:, :5],
#                                                   cls_dets[:,
#                                                            -1], self.iou_thr)
#                     elif self.predict_box_type == 'qbox':
#                         nms_dets, _ = nms_quadri(cls_dets[:, :8],
#                                                  cls_dets[:, -1], self.iou_thr)
#                     else:
#                         raise NotImplementedError
#                     big_img_results.append(nms_dets.cpu().numpy())
#             id_list.append(oriname)
#             dets_list.append(big_img_results)

#         if osp.exists(outfile_prefix):
#             raise ValueError(f'The outfile_prefix should be a non-exist path, '
#                              f'but {outfile_prefix} is existing. '
#                              f'Please delete it firstly.')
#         os.makedirs(outfile_prefix)

#         files = [
#             osp.join(outfile_prefix, 'Task1_' + cls + '.txt')
#             for cls in self.dataset_meta['classes']
#         ]
#         file_objs = [open(f, 'w') for f in files]
#         for img_id, dets_per_cls in zip(id_list, dets_list):
#             for f, dets in zip(file_objs, dets_per_cls):
#                 if dets.size == 0:
#                     continue
#                 th_dets = torch.from_numpy(dets)
#                 if self.predict_box_type == 'rbox':
#                     rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
#                     qboxes = rbox2qbox(rboxes)
#                 elif self.predict_box_type == 'qbox':
#                     qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
#                 else:
#                     raise NotImplementedError
#                 for qbox, score in zip(qboxes, scores):
#                     txt_element = [img_id, str(round(float(score), 2))
#                                    ] + [f'{p:.2f}' for p in qbox]
#                     f.writelines(' '.join(txt_element) + '\n')

#         for f in file_objs:
#             f.close()

#         target_name = osp.split(outfile_prefix)[-1]
#         zip_path = osp.join(outfile_prefix, target_name + '.zip')
#         with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
#             for f in files:
#                 t.write(f, osp.split(f)[-1])

#         return zip_path

#     def results2json(self, results: Sequence[dict],
#                      outfile_prefix: str) -> dict:
#         """Dump the detection results to a COCO style json file.

#         There are 3 types of results: proposals, bbox predictions, mask
#         predictions, and they have different data types. This method will
#         automatically recognize the type, and dump them to json files.

#         Args:
#             results (Sequence[dict]): Testing results of the
#                 dataset.
#             outfile_prefix (str): The filename prefix of the json files. If the
#                 prefix is "somepath/xxx", the json files will be named
#                 "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
#                 "somepath/xxx.proposal.json".

#         Returns:
#             dict: Possible keys are "bbox", "segm", "proposal", and
#             values are corresponding filenames.
#         """
#         bbox_json_results = []
#         for idx, result in enumerate(results):
#             image_id = result.get('img_id', idx)
#             labels = result['labels']
#             bboxes = result['bboxes']
#             scores = result['scores']
#             # bbox results
#             for i, label in enumerate(labels):
#                 data = dict()
#                 data['image_id'] = image_id
#                 data['bbox'] = bboxes[i].tolist()
#                 data['score'] = float(scores[i])
#                 label = int(label) if label.ndim == 0 else label.tolist()
#                 data['category_id'] = label
#                 bbox_json_results.append(data)

#         result_files = dict()
#         result_files['bbox'] = f'{outfile_prefix}.bbox.json'
#         dump(bbox_json_results, result_files['bbox'])

#         return result_files

#     def process(self, data_batch: Sequence[dict],
#                 data_samples: Sequence[dict]) -> None:
#         """Process one batch of data samples and predictions. The processed
#         results should be stored in ``self.results``, which will be used to
#         compute the metrics when all batches have been processed.

#         Args:
#             data_batch (dict): A batch of data from the dataloader.
#             data_samples (Sequence[dict]): A batch of data samples that
#                 contain annotations and predictions.
#         """
#         for data_sample in data_samples:
#             gt = copy.deepcopy(data_sample)
#             gt_instances = gt['gt_instances']
#             gt_ignore_instances = gt['ignored_instances']
#             if gt_instances == {}:
#                 ann = dict()
#             else:
#                 ann = dict(
#                     labels=gt_instances['labels'].cpu().numpy(),
#                     bboxes=gt_instances['bboxes'].cpu().numpy(),
#                     bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
#                     labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
#             result = dict()
#             pred = data_sample['pred_instances']
#             result['img_id'] = data_sample['img_id']
#             result['bboxes'] = pred['bboxes'].cpu().numpy()
#             result['scores'] = pred['scores'].cpu().numpy()
#             result['labels'] = pred['labels'].cpu().numpy()

#             result['pred_bbox_scores'] = []
#             for label in range(len(self.dataset_meta['classes'])):
#                 index = np.where(result['labels'] == label)[0]
#                 pred_bbox_scores = np.hstack([
#                     result['bboxes'][index], result['scores'][index].reshape(
#                         (-1, 1))
#                 ])
#                 result['pred_bbox_scores'].append(pred_bbox_scores)

#             self.results.append((ann, result))

#     def _calculate_angle_deviation(self, det_box, gt_box):
#         """Calculate the angle deviation between prediction and ground truth.
#         Args:
#             det_box (ndarray): [cx, cy, w, h, a]
#             gt_box (ndarray): [cx, cy, w, h, a]
#         Returns:
#             float: angle deviation in degrees.
#         """
#         # Angles in radians (usually last element)
#         a_det = det_box[4]
#         a_gt = gt_box[4]
        
#         # Calculate difference
#         diff = a_det - a_gt
        
#         # Handle LE90 periodicity (period is pi)
#         # Normalize diff to [-pi/2, pi/2)
#         diff = (diff + np.pi / 2) % np.pi - np.pi / 2
        
#         return abs(diff * 180 / np.pi)

#     def _calculate_size_iou(self, det_box, gt_box):
#         """Calculate the horizontal IoU of centered boxes to measure size similarity.
#         Args:
#             det_box (ndarray): [cx, cy, w, h, a]
#             gt_box (ndarray): [cx, cy, w, h, a]
#         Returns:
#             float: IoU value.
#         """
#         # Construct horizontal centered boxes: [0, 0, w, h] -> [x1, y1, x2, y2]
#         # x1 = -w/2, y1 = -h/2, x2 = w/2, y2 = h/2
        
#         w_det, h_det = det_box[2], det_box[3]
#         w_gt, h_gt = gt_box[2], gt_box[3]
        
#         det_rect = torch.tensor([[-w_det/2, -h_det/2, w_det/2, h_det/2]])
#         gt_rect = torch.tensor([[-w_gt/2, -h_gt/2, w_gt/2, h_gt/2]])
        
#         iou = bbox_overlaps(det_rect, gt_rect, is_aligned=True)
#         return iou.item()

#     def compute_metrics(self, results: list) -> dict:
#         """Compute the metrics from processed results.

#         Args:
#             results (list): The processed results of each batch.
#         Returns:
#             dict: The computed metrics. The keys are the names of the metrics,
#             and the values are corresponding results.
#         """
#         print(">>>>>>>>>>>>>> DEBUG: DOTAMetric Modified Code is Running! <<<<<<<<<<<<<<")
#         logger: MMLogger = MMLogger.get_current_instance()
#         gts, preds = zip(*results)

#         tmp_dir = None
#         if self.outfile_prefix is None:
#             tmp_dir = tempfile.TemporaryDirectory()
#             outfile_prefix = osp.join(tmp_dir.name, 'results')
#         else:
#             outfile_prefix = self.outfile_prefix

#         eval_results = OrderedDict()
#         if self.merge_patches:
#             # convert predictions to txt format and dump to zip file
#             zip_path = self.merge_results(preds, outfile_prefix)
#             logger.info(f'The submission file save at {zip_path}')
#             return eval_results
#         else:
#             # convert predictions to coco format and dump to json file
#             _ = self.results2json(preds, outfile_prefix)
#             if self.format_only:
#                 logger.info('results are saved in '
#                             f'{osp.dirname(outfile_prefix)}')
#                 return eval_results

#         if self.metric == 'mAP':
#             assert isinstance(self.iou_thrs, list)
#             dataset_name = self.dataset_meta['classes']
#             dets = [pred['pred_bbox_scores'] for pred in preds]

#             mean_aps = []
            
#             # Variables for mAD and mSIoU calculation
#             # We will accumulate them only for the first IoU threshold if multiple are provided
#             # typically, DOTA eval uses single IoU=0.5
#             first_iou_done = False
#             total_tp_count = 0
#             total_angle_dev = 0.0
#             total_size_iou = 0.0

#             for iou_thr in self.iou_thrs:
#                 logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
#                 # Updated eval_rbbox_map call to unpack matched_pairs
#                 mean_ap, _, matched_pairs = eval_rbbox_map(
#                     dets,
#                     gts,
#                     scale_ranges=self.scale_ranges,
#                     iou_thr=iou_thr,
#                     use_07_metric=self.use_07_metric,
#                     box_type=self.predict_box_type,
#                     dataset=dataset_name,
#                     logger=logger)
                
#                 mean_aps.append(mean_ap)
#                 eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                
#                 # Calculate custom metrics only for the first IoU threshold (usually 0.5)
#                 # or you can change logic to average over all thresholds
#                 if not first_iou_done:
#                     for det_box, gt_box in matched_pairs:
#                         total_angle_dev += self._calculate_angle_deviation(det_box, gt_box)
#                         total_size_iou += self._calculate_size_iou(det_box, gt_box)
#                     total_tp_count = len(matched_pairs)
#                     first_iou_done = True

#             eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
#             eval_results.move_to_end('mAP', last=False)
            
#             # Compute final mAD and mSIoU
#             if total_tp_count > 0:
#                 mAD = total_angle_dev / total_tp_count
#                 mSIoU = total_size_iou / total_tp_count
#             else:
#                 mAD = 0.0
#                 mSIoU = 0.0
            
#             eval_results['mAD'] = round(mAD, 3)
#             eval_results['mSIoU'] = round(mSIoU, 3)

#             # ================= [Explicit Print] =================
#             # Create a small table specifically for custom metrics
#             custom_data = [
#                 ['Extra Metric', 'Value'],
#                 ['mAD (Angle Dev)', f'{mAD:.3f} deg'],
#                 ['mSIoU (Size)', f'{mSIoU:.3f}']
#             ]
#             custom_table = AsciiTable(custom_data)
#             # Use logger from start of function
#             print_log('\n' + custom_table.table, logger=logger)
#             # =======================================================
            
#         else:
#             raise NotImplementedError
#         return eval_results

# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os.path as osp
import tempfile
from collections import OrderedDict, defaultdict
from typing import List, Optional, Sequence, Union

import numpy as np
import torch
from mmcv.ops import nms_quadri, nms_rotated
from mmdet.structures.bbox import bbox_overlaps
from mmengine.evaluator import BaseMetric
from mmengine.fileio import dump
from mmengine.logging import MMLogger

# Import print_log and the print function from mean_ap
from mmengine.logging import print_log 
from mmrotate.evaluation.functional.mean_ap import print_map_summary 

from mmrotate.evaluation import eval_rbbox_map
from mmrotate.registry import METRICS
from mmrotate.structures.bbox import rbox2qbox


@METRICS.register_module()
class DOTAMetric(BaseMetric):
    """DOTA evaluation metric."""
    
    default_prefix: Optional[str] = 'dota'

    def __init__(self,
                 iou_thrs: Union[float, List[float]] = 0.5,
                 scale_ranges: Optional[List[tuple]] = None,
                 metric: Union[str, List[str]] = 'mAP',
                 predict_box_type: str = 'rbox',
                 format_only: bool = False,
                 outfile_prefix: Optional[str] = None,
                 merge_patches: bool = False,
                 iou_thr: float = 0.1,
                 eval_mode: str = '11points',
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.iou_thrs = [iou_thrs] if isinstance(iou_thrs, float) \
            else iou_thrs
        assert isinstance(self.iou_thrs, list)
        self.scale_ranges = scale_ranges
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ['mAP']
        if metric not in allowed_metrics:
            raise KeyError(f"metric should be one of 'mAP', but got {metric}.")
        self.metric = metric
        self.predict_box_type = predict_box_type
        self.format_only = format_only
        if self.format_only:
            assert outfile_prefix is not None, 'outfile_prefix must be not None'
        self.outfile_prefix = outfile_prefix
        self.merge_patches = merge_patches
        self.iou_thr = iou_thr
        self.use_07_metric = True if eval_mode == '11points' else False

    def merge_results(self, results: Sequence[dict], outfile_prefix: str) -> str:
        # ... (Keep original implementation of merge_results) ...
        # (Omitting for brevity as it is unchanged from your uploaded file)
        collector = defaultdict(list)
        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            # ... (parsing logic) ...
            x_y_2 = [0, 0] # Simplified for placeholder, please use original code
            # Note: I am pasting the crucial parts. Assuming user keeps the original merge_results logic
            # Since I cannot see the full helper functions content in your context for brevity,
            # I will assume you copy the original merge_results method body here.
            # BUT to be safe, I will paste the core logic if I can.
            pass 
        return "placeholder_path.zip"

    # ... (Please keep results2json, process, etc. unchanged) ...
    # I will provide the FULL compute_metrics method below which is the key.
    
    # Helper functions for calculation
    def _calculate_angle_deviation(self, det_box, gt_box):
        a_det = det_box[4]
        a_gt = gt_box[4]
        diff = a_det - a_gt
        diff = (diff + np.pi / 2) % np.pi - np.pi / 2
        return abs(diff * 180 / np.pi)

    def _calculate_size_iou(self, det_box, gt_box):
        w_det, h_det = det_box[2], det_box[3]
        w_gt, h_gt = gt_box[2], gt_box[3]
        det_rect = torch.tensor([[-w_det/2, -h_det/2, w_det/2, h_det/2]])
        gt_rect = torch.tensor([[-w_gt/2, -h_gt/2, w_gt/2, h_gt/2]])
        iou = bbox_overlaps(det_rect, gt_rect, is_aligned=True)
        return iou.item()

    def merge_results(self, results: Sequence[dict],
                      outfile_prefix: str) -> str:
        # Standard implementation copy to ensure it works
        import re
        import zipfile
        collector = defaultdict(list)
        for idx, result in enumerate(results):
            img_id = result.get('img_id', idx)
            splitname = img_id.split('__')
            oriname = splitname[0]
            pattern1 = re.compile(r'__\d+___\d+')
            x_y = re.findall(pattern1, img_id)
            x_y_2 = re.findall(r'\d+', x_y[0])
            x, y = int(x_y_2[0]), int(x_y_2[1])
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            ori_bboxes = bboxes.copy()
            if self.predict_box_type == 'rbox':
                ori_bboxes[..., :2] = ori_bboxes[..., :2] + np.array(
                    [x, y], dtype=np.float32)
            elif self.predict_box_type == 'qbox':
                ori_bboxes[..., :] = ori_bboxes[..., :] + np.array(
                    [x, y, x, y, x, y, x, y], dtype=np.float32)
            label_dets = np.concatenate(
                [labels[:, np.newaxis], ori_bboxes, scores[:, np.newaxis]],
                axis=1)
            collector[oriname].append(label_dets)

        id_list, dets_list = [], []
        for oriname, label_dets_list in collector.items():
            big_img_results = []
            label_dets = np.concatenate(label_dets_list, axis=0)
            labels, dets = label_dets[:, 0], label_dets[:, 1:]
            for i in range(len(self.dataset_meta['classes'])):
                if len(dets[labels == i]) == 0:
                    big_img_results.append(dets[labels == i])
                else:
                    try:
                        cls_dets = torch.from_numpy(dets[labels == i]).cuda()
                    except:
                        cls_dets = torch.from_numpy(dets[labels == i])
                    if self.predict_box_type == 'rbox':
                        nms_dets, _ = nms_rotated(cls_dets[:, :5], cls_dets[:, -1], self.iou_thr)
                    elif self.predict_box_type == 'qbox':
                        nms_dets, _ = nms_quadri(cls_dets[:, :8], cls_dets[:, -1], self.iou_thr)
                    big_img_results.append(nms_dets.cpu().numpy())
            id_list.append(oriname)
            dets_list.append(big_img_results)

        if osp.exists(outfile_prefix):
            raise ValueError(f'{outfile_prefix} exists.')
        os.makedirs(outfile_prefix)

        files = [osp.join(outfile_prefix, 'Task1_' + cls + '.txt') for cls in self.dataset_meta['classes']]
        file_objs = [open(f, 'w') for f in files]
        for img_id, dets_per_cls in zip(id_list, dets_list):
            for f, dets in zip(file_objs, dets_per_cls):
                if dets.size == 0: continue
                th_dets = torch.from_numpy(dets)
                if self.predict_box_type == 'rbox':
                    rboxes, scores = torch.split(th_dets, (5, 1), dim=-1)
                    qboxes = rbox2qbox(rboxes)
                elif self.predict_box_type == 'qbox':
                    qboxes, scores = torch.split(th_dets, (8, 1), dim=-1)
                for qbox, score in zip(qboxes, scores):
                    txt_element = [img_id, str(round(float(score), 2))] + [f'{p:.2f}' for p in qbox]
                    f.writelines(' '.join(txt_element) + '\n')
        for f in file_objs: f.close()
        target_name = osp.split(outfile_prefix)[-1]
        zip_path = osp.join(outfile_prefix, target_name + '.zip')
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as t:
            for f in files: t.write(f, osp.split(f)[-1])
        return zip_path

    def results2json(self, results: Sequence[dict], outfile_prefix: str) -> dict:
        bbox_json_results = []
        for idx, result in enumerate(results):
            image_id = result.get('img_id', idx)
            labels = result['labels']
            bboxes = result['bboxes']
            scores = result['scores']
            for i, label in enumerate(labels):
                data = dict()
                data['image_id'] = image_id
                data['bbox'] = bboxes[i].tolist()
                data['score'] = float(scores[i])
                data['category_id'] = int(label)
                bbox_json_results.append(data)
        result_files = dict()
        result_files['bbox'] = f'{outfile_prefix}.bbox.json'
        dump(bbox_json_results, result_files['bbox'])
        return result_files

    def process(self, data_batch: Sequence[dict], data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            gt = copy.deepcopy(data_sample)
            gt_instances = gt['gt_instances']
            gt_ignore_instances = gt['ignored_instances']
            if gt_instances == {}:
                ann = dict()
            else:
                ann = dict(
                    labels=gt_instances['labels'].cpu().numpy(),
                    bboxes=gt_instances['bboxes'].cpu().numpy(),
                    bboxes_ignore=gt_ignore_instances['bboxes'].cpu().numpy(),
                    labels_ignore=gt_ignore_instances['labels'].cpu().numpy())
            result = dict()
            pred = data_sample['pred_instances']
            result['img_id'] = data_sample['img_id']
            result['bboxes'] = pred['bboxes'].cpu().numpy()
            result['scores'] = pred['scores'].cpu().numpy()
            result['labels'] = pred['labels'].cpu().numpy()
            result['pred_bbox_scores'] = []
            for label in range(len(self.dataset_meta['classes'])):
                index = np.where(result['labels'] == label)[0]
                pred_bbox_scores = np.hstack([
                    result['bboxes'][index], result['scores'][index].reshape((-1, 1))
                ])
                result['pred_bbox_scores'].append(pred_bbox_scores)
            self.results.append((ann, result))

    def compute_metrics(self, results: list) -> dict:
        """Compute the metrics from processed results."""
        logger: MMLogger = MMLogger.get_current_instance()
        gts, preds = zip(*results)

        tmp_dir = None
        if self.outfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            outfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            outfile_prefix = self.outfile_prefix

        eval_results = OrderedDict()
        if self.merge_patches:
            zip_path = self.merge_results(preds, outfile_prefix)
            logger.info(f'The submission file save at {zip_path}')
            return eval_results
        else:
            _ = self.results2json(preds, outfile_prefix)
            if self.format_only:
                logger.info(f'results are saved in {osp.dirname(outfile_prefix)}')
                return eval_results

        if self.metric == 'mAP':
            assert isinstance(self.iou_thrs, list)
            dataset_name = self.dataset_meta['classes']
            dets = [pred['pred_bbox_scores'] for pred in preds]

            mean_aps = []
            first_iou_done = False
            total_tp_count = 0
            total_angle_dev = 0.0
            total_size_iou = 0.0

            for iou_thr in self.iou_thrs:
                logger.info(f'\n{"-" * 15}iou_thr: {iou_thr}{"-" * 15}')
                
                # [MODIFIED] 1. Call with silent logger to suppress default table
                mean_ap, eval_results_list, cls_matched_pairs = eval_rbbox_map(
                    dets,
                    gts,
                    scale_ranges=self.scale_ranges,
                    iou_thr=iou_thr,
                    use_07_metric=self.use_07_metric,
                    box_type=self.predict_box_type,
                    dataset=dataset_name,
                    logger='silent') # SILENT here
                
                mean_aps.append(mean_ap)
                eval_results[f'AP{int(iou_thr * 100):02d}'] = round(mean_ap, 3)
                
                # [MODIFIED] 2. Calculate Per-Class mAD and mSIoU
                if not first_iou_done:
                    # Iterate over per-class matched pairs
                    for i, class_pairs in enumerate(cls_matched_pairs):
                        if len(class_pairs) > 0:
                            c_angle_dev = 0.0
                            c_size_iou = 0.0
                            for det_box, gt_box in class_pairs:
                                # Calculate deviations
                                c_angle_dev += self._calculate_angle_deviation(det_box, gt_box)
                                c_size_iou += self._calculate_size_iou(det_box, gt_box)
                            
                            # Store per-class average in the results list (which print_map_summary uses)
                            eval_results_list[i]['mAD'] = c_angle_dev / len(class_pairs)
                            eval_results_list[i]['mSIoU'] = c_size_iou / len(class_pairs)
                            
                            # Accumulate for global average
                            total_angle_dev += c_angle_dev
                            total_size_iou += c_size_iou
                            total_tp_count += len(class_pairs)
                        else:
                            # If no TP for this class, set to 0.0
                            eval_results_list[i]['mAD'] = 0.0
                            eval_results_list[i]['mSIoU'] = 0.0
                    
                    first_iou_done = True
                    
                    # [MODIFIED] 3. Print the Big Table explicitly here, now that eval_results_list has extra data
                    print_map_summary(
                        mean_ap, 
                        eval_results_list, 
                        dataset_name, 
                        self.scale_ranges, 
                        logger=logger # Use the real logger here
                    )

            eval_results['mAP'] = sum(mean_aps) / len(mean_aps)
            eval_results.move_to_end('mAP', last=False)
            
            # Global stats
            if total_tp_count > 0:
                mAD = total_angle_dev / total_tp_count
                mSIoU = total_size_iou / total_tp_count
            else:
                mAD = 0.0
                mSIoU = 0.0
            
            eval_results['mAD'] = round(mAD, 3)
            eval_results['mSIoU'] = round(mSIoU, 3)
            
        else:
            raise NotImplementedError
        return eval_results