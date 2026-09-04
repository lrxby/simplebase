# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
import numpy as np
from mmrotate.registry import TRANSFORMS  # 注意：新版使用 TRANSFORMS 注册器

@TRANSFORMS.register_module()
class LoadPseudoAnnotations:
    """加载离线伪标签 (RBoxes) 并注册到 bbox_fields 以同步数据增强"""

    def __init__(self, pkl_path):
        self.pkl_path = pkl_path
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"未找到伪标签文件: {pkl_path}")

        with open(pkl_path, 'rb') as f:
            self.data_dict = pickle.load(f)

        print(f"[LoadPseudoAnnotations] 成功加载离线数据: {pkl_path}")

    def __call__(self, results):
        # 获取文件名 ID (不带路径和后缀)
        file_name = results.get('img_path', '') # MMDetection 3.x 常用 img_path
        if not file_name:
            file_name = results.get('img_info', {}).get('filename', '')

        file_id = os.path.splitext(os.path.basename(file_name))[0]

        if file_id in self.data_dict:
            pseudo_data = np.array(self.data_dict[file_id], dtype=np.float32)
            # 只取前 5 列 [x, y, w, h, angle]
            pseudo_boxes = pseudo_data[:, :5] if pseudo_data.size > 0 else np.zeros((0, 5), dtype=np.float32)
        else:
            # 如果找不到，根据 gt_instances 的数量生成占位符
            num_gts = len(results.get('gt_bboxes', []))
            pseudo_boxes = np.zeros((num_gts, 5), dtype=np.float32)

        results['pseudo_boxes'] = pseudo_boxes

        # 核心：注册到 bbox_fields 确保数据增强同步
        if 'bbox_fields' not in results:
            results['bbox_fields'] = []

        if 'pseudo_boxes' not in results['bbox_fields']:
            results['bbox_fields'].append('pseudo_boxes')

        return results

    def __repr__(self):
        return f'{self.__class__.__name__}(pkl_path={self.pkl_path})'
