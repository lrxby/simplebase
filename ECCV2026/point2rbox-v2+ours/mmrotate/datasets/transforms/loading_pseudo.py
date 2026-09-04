# Copyright (c) OpenMMLab. All rights reserved.
import os
import pickle
import torch
import numpy as np
from mmcv.transforms import BaseTransform
from mmrotate.registry import TRANSFORMS

@TRANSFORMS.register_module()
class LoadPseudoAnnotations(BaseTransform):
    """加载离线生成的伪标签 (Pseudo Boxes)
    
    Args:
        pkl_path (str): 离线伪标签 .pkl 文件的路径
    """
    def __init__(self, pkl_path: str):
        self.pkl_path = pkl_path
        
        # 为了高效读取，我们在初始化时一次性将 pkl 加载到内存中
        print(f"Loading pseudo annotations from {pkl_path}...")
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Pseudo label file not found: {pkl_path}")
            
        with open(self.pkl_path, 'rb') as f:
            self.pseudo_data = pickle.load(f)
        print(f"Successfully loaded {len(self.pseudo_data)} pseudo annotations.")

    def transform(self, results: dict) -> dict:
        """
        MMDetection/MMRotate 数据流 Pipeline 核心函数
        每读取一张图，就会调用一次该函数
        """
        # 获取当前图片的文件名 (例如: 'P0000.png')
        img_path = results['img_path']
        img_name = os.path.basename(img_path)       
        img_stem = os.path.splitext(img_name)[0]    # 'P0000'

        # 从字典中获取当前图片对应的伪标签 (Boxes)
        # 注意：这里根据你 pkl 字典中存的 key (是带后缀还是不带后缀) 自动匹配
        if img_stem in self.pseudo_data:
            pseudo_boxes = self.pseudo_data[img_stem]
        elif img_name in self.pseudo_data:
            pseudo_boxes = self.pseudo_data[img_name]
        else:
            # 如果这幅图中没有物体，或者 pkl 中没有，赋值为空
            pseudo_boxes = None

        # 确保输出的数据格式是 PyTorch Tensor
        if pseudo_boxes is not None:
            if not isinstance(pseudo_boxes, torch.Tensor):
                # 假设离线存的是 numpy 数组
                pseudo_boxes = torch.tensor(pseudo_boxes, dtype=torch.float32)
            
            # 确保 shape 为 [N, 5]，防止维度错乱
            if pseudo_boxes.dim() == 1:
                pseudo_boxes = pseudo_boxes.unsqueeze(0)

        # 把伪标签挂载到 results 字典上，交给下游的 PackDetInputs 打包
        results['pseudo_boxes'] = pseudo_boxes
        
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(pkl_path={self.pkl_path})'
        return repr_str
