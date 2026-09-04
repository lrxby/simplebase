# Copyright (c) OpenMMLab. All rights reserved.
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
from mmdet.models.losses.utils import weighted_loss

from mmrotate.registry import MODELS
from mmrotate.models.losses.gaussian_dist_loss import postprocess


@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    """Gaussian Wasserstein distance loss.
    Modified from gwd_loss. 
    gwd_sigma_loss only involves sigma in Gaussian, with mu ignored.

    Args:
        pred (torch.Tensor): Predicted bboxes.
        target (torch.Tensor): Corresponding gt bboxes.
        fun (str): The function applied to distance. Defaults to 'log1p'.
        tau (float): Defaults to 1.0.
        alpha (float): Defaults to 1.0.
        normalize (bool): Whether to normalize the distance. Defaults to True.

    Returns:
        loss (torch.Tensor)

    """
    Sigma_p = pred
    Sigma_t = target

    whr_distance = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    whr_distance = whr_distance + Sigma_t.diagonal(
        dim1=-2, dim2=-1).sum(dim=-1)

    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    _t_det_sqrt = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr_distance = whr_distance + (-2) * (
        (_t_tr + 2 * _t_det_sqrt).clamp(1e-7).sqrt())

    distance = (alpha * alpha * whr_distance).clamp(1e-7).sqrt()

    if normalize:
        scale = 2 * (
            _t_det_sqrt.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        distance = distance / scale

    return postprocess(distance, fun=fun, tau=tau)


def bhattacharyya_coefficient(pred, target):
    """Calculate bhattacharyya coefficient between 2-D Gaussian distributions.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        target (Tuple): tuple of (xy, sigma).

    Returns:
        coef (Tensor): bhattacharyya coefficient with shape (N,).
    """
    xy_p, Sigma_p = pred
    xy_t, Sigma_t = target

    _shape = xy_p.shape

    xy_p = xy_p.reshape(-1, 2)
    xy_t = xy_t.reshape(-1, 2)
    Sigma_p = Sigma_p.reshape(-1, 2, 2)
    Sigma_t = Sigma_t.reshape(-1, 2, 2)

    Sigma_M = (Sigma_p + Sigma_t) / 2
    dxy = (xy_p - xy_t).unsqueeze(-1)
    t0 = torch.exp(-0.125 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(Sigma_M, dxy)))
    t1 = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    t2 = Sigma_M.det()

    coef = t0 * (t1 / t2).clamp(1e-7).sqrt()[..., None, None]
    coef = coef.reshape(_shape[:-1])
    return coef


@weighted_loss
def gaussian_overlap_loss(pred, target, alpha=0.01, beta=0.6065):
    """Calculate Gaussian overlap loss based on bhattacharyya coefficient.

    Args:
        pred (Tuple): tuple of (xy, sigma).
            xy (torch.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).

    Returns:
        loss (Tensor): overlap loss with shape (N, N).
    """
    mu, sigma = pred
    B = mu.shape[0]
    mu0 = mu[None].expand(B, B, 2)
    sigma0 = sigma[None].expand(B, B, 2, 2)
    mu1 = mu[:, None].expand(B, B, 2)
    sigma1 = sigma[:, None].expand(B, B, 2, 2)
    loss = bhattacharyya_coefficient((mu0, sigma0), (mu1, sigma1))
    loss[torch.eye(B, dtype=bool)] = 0
    loss = F.leaky_relu(loss - beta, negative_slope=alpha) + beta * alpha
    loss = loss.sum(-1)
    return loss


@MODELS.register_module()
class GaussianOverlapLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0,
                 lamb=1e-4):
        super(GaussianOverlapLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.lamb = lamb

    def forward(self,
                pred,
                weight=None,
                avg_factor=None,
                reduction_override=None):
        """Forward function.

        Args:
            pred (Tuple): tuple of (xy, sigma).
                xy (torch.Tensor): center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        assert len(pred[0]) == len(pred[1])

        sigma = pred[1]
        L = torch.linalg.eigh(sigma)[0].clamp(1e-7).sqrt()
        loss_lamb = F.l1_loss(L, torch.zeros_like(L), reduction='none')
        loss_lamb = self.lamb * loss_lamb.log1p().mean()
        
        return self.loss_weight * (loss_lamb + gaussian_overlap_loss(
            pred,
            None,
            weight,
            reduction=reduction,
            avg_factor=avg_factor))


def plot_gaussian_voronoi_watershed(*images):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(len(images) * 4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(images)):
        img = images[i]
        img = (img - img.min()) / (img.max() - img.min())
        if img.dim() == 3:
            img = img.permute(1, 2, 0)
        img = img.detach().cpu().numpy()
        plt.subplot(1, len(images), i + 1)
        if i == 3:
            plt.imshow(img)
            x = np.linspace(0, 1024, 1024)
            y = np.linspace(0, 1024, 1024)
            X, Y = np.meshgrid(x, y)
            plt.contourf(X, Y, img, levels=8, cmap=plt.get_cmap('magma'))
        else:
            plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Gaussian-Voronoi-{fileid}.png')
    plt.close()


def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0


def gaussian_voronoi_watershed_loss(mu, sigma,
                                    label, image, 
                                    pos_thres, neg_thres, 
                                    down_sample=2, topk=0.95, 
                                    default_sigma=4096,
                                    voronoi='gaussian-orientation',
                                    alpha=0.1,
                                    debug=False):
    J = len(sigma)
    if J == 0:
        return sigma.sum()
    
    D = down_sample
    H, W = image.shape[-2:]
    h, w = H // D, W // D
    x = torch.linspace(0, h, h, device=mu.device)
    y = torch.linspace(0, w, w, device=mu.device)
    xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
    vor = mu.new_zeros(J, h, w)
    # Get distribution for each instance
    mm = (mu.detach() / D).round()
    if voronoi == 'standard':
        sg = sigma.new_tensor((default_sigma, 0, 0, default_sigma)).reshape(2, 2)
        sg = sg / D ** 2
        for j, m in enumerate(mm):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
    elif voronoi == 'gaussian-orientation':
        L, V = torch.linalg.eigh(sigma)
        L = L.detach().clone()
        L = L / (L[:, 0:1] * L[:, 1:2]).sqrt() * default_sigma
        sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach()
        sg = sg / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    elif voronoi == 'gaussian-full':
        sg = sigma.detach() / D ** 2
        for j, (m, s) in enumerate(zip(mm, sg)):
            vor[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
    # val: max prob, vor: belong to which instance, cls: belong to which class
    val, vor = torch.max(vor, 0)
    if D > 1:
        vor = vor[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
        val = F.interpolate(
            val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
    cls = label[vor]
    kernel = val.new_ones((1, 1, 3, 3))
    kernel[0, 0, 1, 1] = -8
    ridges = torch.conv2d(vor[None].float(), kernel, padding=1)[0] != 0
    vor += 1
    pos_thres = val.new_tensor(pos_thres)
    neg_thres = val.new_tensor(neg_thres)
    vor[val < pos_thres[cls]] = 0
    vor[val < neg_thres[cls]] = J + 1
    vor[ridges] = J + 1

    cls_bg = torch.where(vor == J + 1, 15, cls)
    cls_bg = torch.where(vor == 0, -1, cls_bg)

    # PyTorch does not support watershed, use cv2
    img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
    img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
    img_uint8 = cv2.medianBlur(img_uint8, 3)
    markers = vor.detach().cpu().numpy().astype(np.int32)
    markers = vor.new_tensor(cv2.watershed(img_uint8, markers))

    if debug:
        plot_gaussian_voronoi_watershed(image, cls_bg, markers)

    L, V = torch.linalg.eigh(sigma)
    L_target = []
    for j in range(J):
        xy = (markers == j + 1).nonzero()[:, (1, 0)].float()
        if len(xy) == 0:
            L_target.append(L[j].detach())
            continue
        xy = xy - mu[j]
        xy = V[j].T.matmul(xy[:, :, None])[:, :, 0]
        max_x = torch.max(torch.abs(xy[:, 0]))
        max_y = torch.max(torch.abs(xy[:, 1]))
        L_target.append(torch.stack((max_x, max_y)) ** 2)
    L_target = torch.stack(L_target)
    L = torch.diag_embed(L)
    L_target = torch.diag_embed(L_target)
    loss = gwd_sigma_loss(L, L_target.detach(), reduction='none')
    loss = torch.topk(loss, int(np.ceil(len(loss) * topk)), largest=False)[0].mean()
    return loss, (vor, markers)


@MODELS.register_module()
class VoronoiWatershedLoss(nn.Module):
    """Gaussian Overlap Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 down_sample=2,
                 reduction='mean',
                 loss_weight=1.0,
                 topk=0.95,
                 alpha=0.1,
                 debug=False):
        super(VoronoiWatershedLoss, self).__init__()
        self.down_sample = down_sample
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.topk = topk
        self.alpha = alpha
        self.debug = debug

    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi='orientation'):
        """Forward function.

        Args:
            pred (Tuple): Tuple of (xy, sigma).
                xy (torch.Tensor): Center point of 2-D Gaussian distribution
                    with shape (N, 2).
                sigma (torch.Tensor): Covariance matrix of 2-D Gaussian distribution
                    with shape (N, 2, 2).
            image (torch.Tensor): The image for watershed with shape (3, H, W).
            standard_voronoi (bool, optional): Use standard or Gaussian voronoi.

        Returns:
            torch.Tensor: The calculated loss
        """
        loss, self.vis = gaussian_voronoi_watershed_loss(*pred, 
                                               label,
                                               image, 
                                               pos_thres, 
                                               neg_thres, 
                                               self.down_sample, 
                                               topk=self.topk,
                                               voronoi=voronoi,
                                               alpha=self.alpha,
                                               debug=self.debug)
        return self.loss_weight * loss


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.

    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.

    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def plot_edge_map(feat, edgex, edgey):
    """Plot figures for debug."""
    import matplotlib.pyplot as plt
    plt.figure(dpi=300, figsize=(4, 4))
    plt.tight_layout()
    fileid = np.random.randint(0, 20)
    for i in range(len(feat)):
        img0 = feat[i, :3]
        img0 = (img0 - img0.min()) / (img0.max() - img0.min())
        img1 = edgex[i, :3]
        img1 = (img1 - img1.min()) / (img1.max() - img1.min())
        img2 = edgey[i, :3]
        img2 = (img2 - img2.min()) / (img2.max() - img2.min())
        img3 = img1 + img2
        img3 = (img3 - img3.min()) / (img3.max() - img3.min())
        img = torch.cat((torch.cat((img0, img2), -1), 
                         torch.cat((img1, img3), -1)), -2
                         ).permute(1, 2, 0).detach().cpu().numpy()
        N = int(np.ceil(np.sqrt(len(feat))))
        plt.subplot(N, N, i + 1)
        plt.imshow(img)
        plt.xticks([])
        plt.yticks([])
    plt.savefig(f'debug/Edge-Map-{fileid}.png')
    plt.close()


@MODELS.register_module()
class EdgeLoss(nn.Module):
    """Edge Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 resolution=24,
                 max_scale=1.6,
                 sigma=6,
                 reduction='mean',
                 loss_weight=1.0,
                 debug=False):
        super(EdgeLoss, self).__init__()
        self.resolution = resolution
        self.max_scale = max_scale
        self.sigma = sigma
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.center_idx = self.resolution / self.max_scale
        self.debug = debug

        self.roi_extractor = MODELS.build(dict(
            type='RotatedSingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlignRotated',
                    out_size=(2 * self.resolution + 1),
                    sample_num=2,
                    clockwise=True),
            out_channels=1,
            featmap_strides=[1],
            finest_scale=1024))

        edge_idx = torch.arange(0, self.resolution + 1)
        edge_distribution = torch.exp(-((edge_idx - self.center_idx) ** 2) / (2 * self.sigma ** 2))
        edge_distribution[0] = edge_distribution[-1] = 0
        self.register_buffer('edge_idx', edge_idx)
        self.register_buffer('edge_distribution', edge_distribution)

    def forward(self, pred, edge):
        """Forward function.

        Args:
            pred (Tuple): Batched predicted rboxes
            edge (torch.Tensor): The edge map with shape (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated loss
        """
        G = self.resolution
        C = self.center_idx
        roi = rbbox2roi(pred)
        roi[:, 3:5] *= self.max_scale
        feat = self.roi_extractor([edge], roi)
        if len(feat) == 0:
            return pred[0].new_tensor(0)
        featx = feat.sum(1).abs().sum(1)
        featy = feat.sum(1).abs().sum(2)
        featx2 = torch.flip(featx[:, :G + 1], (-1,)) + featx[:, G:]
        featy2 = torch.flip(featy[:, :G + 1], (-1,)) + featy[:, G:]  # (N, 25)
        ex = ((featx2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        ey = ((featy2 * self.edge_distribution).softmax(1) * self.edge_idx).sum(1) / C
        exy = torch.stack((ex, ey), -1)
        rbbox_concat = torch.cat(pred, 0)
        
        if self.debug:
            edgex = featx[:, None, None, :].expand(-1, 1, 2 * self.resolution + 1, -1)
            edgey = featy[:, None, :, None].expand(-1, 1, -1, 2 * self.resolution + 1)
            plot_edge_map(feat, edgex, edgey)

        return self.loss_weight * F.smooth_l1_loss(rbbox_concat[:, 2:4], 
                                      (rbbox_concat[:, 2:4] * exy).detach(),
                                      beta=8)


@MODELS.register_module()
class Point2RBoxV2ConsistencyLoss(nn.Module):
    """Consistency Loss.

    Args:
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'. Options are "none", "mean" and
            "sum".
        loss_weight (float, optional): Weight of loss. Defaults to 1.0.

    Returns:
        loss (torch.Tensor)
    """

    def __init__(self,
                 reduction='mean',
                 loss_weight=1.0):
        super(Point2RBoxV2ConsistencyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, ori_pred, trs_pred, square_mask, aug_type, aug_val):
        """Forward function.

        Args:
            ori_pred (Tuple): (Sigma, theta)
            trs_pred (Tuple): (Sigma, theta)
            square_mask: When True, the angle is ignored
            aug_type: 'rot', 'flp', 'sca'
            aug_val: Rotation or scale value

        Returns:
            torch.Tensor: The calculated loss
        """
        ori_gaus, ori_angle = ori_pred
        trs_gaus, trs_angle = trs_pred

        if aug_type == 'rot':
            rot = ori_gaus.new_tensor(aug_val)
            cos_r = torch.cos(rot)
            sin_r = torch.sin(rot)
            R = torch.stack((cos_r, -sin_r, sin_r, cos_r), dim=-1).reshape(-1, 2, 2)
            ori_gaus = R.matmul(ori_gaus).matmul(R.permute(0, 2, 1))
            d_ang = trs_angle - ori_angle - aug_val
        elif aug_type == 'flp':
            ori_gaus = ori_gaus * ori_gaus.new_tensor((1, -1, -1, 1)).reshape(2, 2)
            d_ang = trs_angle + ori_angle
        else:
            sca = ori_gaus.new_tensor(aug_val)
            ori_gaus = ori_gaus * sca
            d_ang = trs_angle - ori_angle
        
        loss_ssg = gwd_sigma_loss(ori_gaus.bmm(ori_gaus), trs_gaus.bmm(trs_gaus))
        d_ang = (d_ang + math.pi / 2) % math.pi - math.pi / 2
        loss_ssa = F.smooth_l1_loss(d_ang, torch.zeros_like(d_ang), reduction='none', beta=0.1)
        loss_ssa = loss_ssa[~square_mask].sum() / max(1, (~square_mask).sum())

        return self.loss_weight * (loss_ssg + loss_ssa)

@MODELS.register_module()
class SizeLoss(nn.Module):  
    """透视感知尺寸一致性损失 。
    实现了对数空间下透视投影的矩阵形式。
    它使用可微分岭回归联合拟合该图片的透视平面（共享斜率）和特定类别的截距。

    公式：log(s) = wx * x + wy * y + b_class

    参数：
        loss_weight (float): 损失权重。默认为 1.0。
        ridge_lambda (float): 岭回归的正则化强度。默认为 1e-4。
        beta (float): SmoothL1Loss 的 beta 参数。默认为 1.0。
        norm_type (str): 坐标归一化类型。可选 'z-score' 或 'image-norm'。默认为 'z-score'。
        target_classes (list[int] | None): 应用损失的类别索引列表。
            如果为 None，则应用于所有类别。
    """

    def __init__(self,
                 loss_weight=1.0,
                 ridge_lambda=1e-4,
                 beta=1.0,
                 norm_type='z-score',
                 target_classes=None):
        super(SizeLoss, self).__init__()  
        self.loss_weight = loss_weight
        self.ridge_lambda = ridge_lambda
        self.beta = beta
        self.norm_type = norm_type
        self.target_classes = target_classes
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none', beta=beta)

    def _solve_single_image(self, pred_bboxes, scores, labels, img_shape=None):
        """
        针对单张图片的求解逻辑。
        """
        # 1. 准备数据
        if pred_bboxes.shape[0] == 0:
            return pred_bboxes.new_tensor(0.0)

        x_c = pred_bboxes[:, 0]
        y_c = pred_bboxes[:, 1]
        w = pred_bboxes[:, 2].clamp(min=1e-2)
        h = pred_bboxes[:, 3].clamp(min=1e-2)

        # 目标向量 Y：对数空间下的 log(sqrt(area))
        s_log = 0.5 * torch.log(w * h)
        Y = s_log.unsqueeze(1)

        # 权重 W
        weights = scores.clamp(min=1e-6)
        sqrt_w = torch.sqrt(weights).unsqueeze(1)

        # 2. 检查约束
        unique_labels, labels_inv = torch.unique(labels, return_inverse=True)
        K = len(unique_labels)
        N = len(pred_bboxes)

        # 样本数必须足以求解 (2个斜率 + K个截距)
        if N < K + 3:
            return pred_bboxes.new_tensor(0.0)

        # 3. 归一化坐标 (Image-wise)
        if self.norm_type == 'z-score':
            # Z-Score: 以当前图片内物体的重心为原点
            x_mean, x_std = x_c.mean().detach(), x_c.std().detach().clamp(min=1e-6)
            y_mean, y_std = y_c.mean().detach(), y_c.std().detach().clamp(min=1e-6)
            x_norm = (x_c - x_mean) / x_std
            y_norm = (y_c - y_mean) / y_std

        elif self.norm_type == 'image-norm':
            # Image-Norm: 以当前图片中心为原点
            if img_shape is None:
                # 如果没拿到 shape，回退到 z-score 或者报错，这里选择回退不中断训练
                x_norm = x_c
                y_norm = y_c
            else:
                H_img, W_img = img_shape[:2]
                x_norm = (x_c - W_img / 2.0) / (W_img / 2.0)
                y_norm = (y_c - H_img / 2.0) / (H_img / 2.0)
        else:
            x_norm = x_c
            y_norm = y_c

        # 4. 构建设计矩阵 A
        A = pred_bboxes.new_zeros((N, 2 + K))
        A[:, 0] = x_norm
        A[:, 1] = y_norm
        # 填充特定类别截距的 One-hot 编码
        A.scatter_(1, 2 + labels_inv.unsqueeze(1), 1.0)

        # 5. 加权岭回归求解
        A_w = A * sqrt_w
        Y_w = Y * sqrt_w

        M = torch.matmul(A_w.t(), A_w)
        I_reg = torch.eye(2 + K, device=pred_bboxes.device) * self.ridge_lambda
        M_reg = M + I_reg

        try:
            RHS = torch.matmul(A_w.t(), Y_w)
            theta = torch.linalg.solve(M_reg, RHS)
        except RuntimeError:
            return pred_bboxes.new_tensor(0.0)

        # 6. 计算 Loss
        Y_hat = torch.matmul(A, theta).squeeze(1)
        diff = s_log - Y_hat

        loss_element = self.smooth_l1(diff, torch.zeros_like(diff))
        loss_weighted = torch.sum(loss_element * weights) / (torch.sum(weights) + 1e-6)

        return loss_weighted

    def forward(self, pred_bboxes, scores, labels, batch_idxs=None, img_metas=None):
        """
        参数：
            pred_bboxes (Tensor): 形状 (N, 5)，所有层级展平后的预测框
            scores (Tensor): 形状 (N, )
            labels (Tensor): 形状 (N, )
            batch_idxs (Tensor): 形状 (N, )，必须提供，用于区分样本属于哪张图
            img_metas (list[dict]): 图片元信息
        """
        # 0. 过滤目标类别
        if self.target_classes is not None:
            mask = torch.zeros_like(labels, dtype=torch.bool)
            for cls_id in self.target_classes:
                mask |= (labels == cls_id)

            if mask.sum() == 0:
                return pred_bboxes.new_tensor(0.0)

            pred_bboxes = pred_bboxes[mask]
            scores = scores[mask]
            labels = labels[mask]
            if batch_idxs is not None:
                batch_idxs = batch_idxs[mask]

        # 1. 基础检查
        if pred_bboxes.shape[-1] < 4:
            return pred_bboxes.new_tensor(0.0)

        # 2. 如果没有 batch_idxs (兼容旧逻辑)，则视为单张大图处理 (Global Fit)
        # 但既然 Head 已经传了 batch_idxs，我们优先走 Loop 逻辑
        if batch_idxs is None:
            return self.loss_weight * self._solve_single_image(pred_bboxes, scores, labels)

        # 3. 逐图计算 Loss (Image-wise Loop)
        unique_batches = torch.unique(batch_idxs)
        total_loss = pred_bboxes.new_tensor(0.0)
        valid_batches = 0.0

        for b_idx in unique_batches:
            # 提取属于当前图片的数据
            mask = (batch_idxs == b_idx)
            b_pred_bboxes = pred_bboxes[mask]
            b_scores = scores[mask]
            b_labels = labels[mask]

            # 获取当前图片的尺寸 (用于 image-norm)
            b_img_shape = None
            if img_metas is not None and len(img_metas) > b_idx:
                b_img_shape = img_metas[int(b_idx.item())]['img_shape']

            # 计算该图的 Loss
            loss_per_img = self._solve_single_image(b_pred_bboxes, b_scores, b_labels, img_shape=b_img_shape)

            # 如果该图有效 (loss != 0)，累加
            if loss_per_img > 0:
                total_loss += loss_per_img
                valid_batches += 1.0

        # 4. 求平均 Loss
        if valid_batches > 0:
            return self.loss_weight * (total_loss / valid_batches)
        else:
            return pred_bboxes.new_tensor(0.0)


@MODELS.register_module()
class AngleLoss(nn.Module):
    """AngleLoss 角度一致性损失

    参数：
        loss_weight (float): 损失总权重。默认为 1.0
        k_radius (float): 高斯核半径缩放系数。默认为 2.0
        score_alpha (float): 置信度权重幂次。默认为 1.0
        target_classes (list[int] | None): 指定参与计算的类别索引列表。
                                           - 如果为 None，则计算所有类别（默认）。
                                           - 如果为 [0, 2]，则只计算类别 0 和 2 的 Loss，忽略其他类别。
        reduction (str): 损失归一化方式。默认为 'mean'
    """

    def __init__(self,
                 loss_weight=1.0,
                 k_radius=2.0,
                 score_alpha=1.0,
                 target_classes=None,
                 reduction='mean'):
        super(AngleLoss, self).__init__()
        self.loss_weight = loss_weight
        self.k_radius = k_radius
        self.score_alpha = score_alpha
        self.target_classes = target_classes
        self.reduction = reduction

    def _forward_single_image(self, bboxes, scores, labels):
        """计算单张图片的 Angle Loss"""
        N = bboxes.shape[0]
        if N < 2:
            return bboxes.sum() * 0.0, 0.0

        # ================= Step 1: 几何解耦 =================
        centers = bboxes[:, :2].detach()
        wh = bboxes[:, 2:4].detach()
        scales = (wh[:, 0] * wh[:, 1]).sqrt().clamp(min=16.0, max=800.0)
        thetas = bboxes[:, 4]

        # ================= Step 2: 矢量化 (4-Theta) =================
        vecs = torch.stack([torch.cos(4 * thetas), torch.sin(4 * thetas)], dim=1)

        # ================= Step 3: 构建亲和矩阵 =================
        # 3.1 空间距离权重
        # 此时 N 仅为当前图片的物体数
        dist_sq = torch.cdist(centers, centers, p=2).pow(2)
        sigmas = scales * self.k_radius
        sigma_mat = sigmas.view(N, 1)
        W_geo = torch.exp(-dist_sq / (2 * sigma_mat.pow(2))).detach()

        # 3.2 置信度加权
        scores_detached = scores.detach().pow(self.score_alpha)
        W_conf = scores_detached.view(1, N)

        # 3.3 逻辑掩码 (仅同类)
        mask_cls = (labels.view(N, 1) == labels.view(1, N)).float()

        W = W_geo * W_conf * mask_cls

        # ================= Step 4: 归一化 =================
        W_sum = W.sum(dim=1, keepdim=True)
        # 防止除零（虽然有自环通常不会，但为了健壮性）
        W_norm = W / (W_sum + 1e-6)

        # ================= Step 5: 能量/混乱度计算 =================
        mean_vecs = torch.mm(W_norm, vecs)
        chaos_score = 1.0 - mean_vecs.norm(dim=1)

        # ================= Step 6: 类别筛选 =================
        if self.target_classes is not None:
            class_mask = torch.zeros_like(labels, dtype=torch.bool)
            for t_cls in self.target_classes:
                class_mask = class_mask | (labels == t_cls)
            class_mask = class_mask.float()
        else:
            class_mask = torch.ones_like(labels, dtype=torch.float)

        final_loss = chaos_score * class_mask

        # 返回 (当前图片的总Loss, 当前图片的有效样本数)
        return final_loss.sum(), class_mask.sum()

    def forward(self,
                pos_bbox_preds,
                pos_scores,
                pos_labels,
                batch_idxs,
                **kwargs):
        """
        遍历 batch_idxs，逐图计算后汇总
        """
        # 初始化统计量
        total_loss = 0.0
        total_valid_samples = 0.0

        # 获取当前 Batch 中包含的所有图片 ID
        unique_batch_ids = torch.unique(batch_idxs)

        for b_id in unique_batch_ids:
            # 1. 提取当前图片的样本
            mask = (batch_idxs == b_id)
            if mask.sum() == 0: continue

            img_bboxes = pos_bbox_preds[mask]
            img_scores = pos_scores[mask]
            img_labels = pos_labels[mask]

            # 2. 计算当前图片的 Loss 和 有效数量
            loss_sum, valid_count = self._forward_single_image(
                img_bboxes, img_scores, img_labels
            )

            total_loss += loss_sum
            total_valid_samples += valid_count

        # ================= Step 7: 最终归一化 =================
        if self.reduction == 'mean':
            if total_valid_samples > 0:
                return self.loss_weight * total_loss / total_valid_samples
            else:
                return pos_bbox_preds.sum() * 0.0
        else:
            return self.loss_weight * total_loss

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from mmrotate.registry import MODELS
from mmdet.models.losses.utils import weighted_loss
from mmrotate.models.losses.gaussian_dist_loss import postprocess

def get_rotated_box_corners(mu, sigma):
    """计算旋转框角点 (用于裁剪和绘图) [-1, 1] Range"""
    device = mu.device
    e, v = torch.linalg.eigh(sigma) 
    e = e.clamp(min=1e-7).sqrt()
    base_corners = torch.tensor([[-1., -1.], [1., -1.], [1., 1.], [-1., 1.]], device=device)
    base_corners = base_corners.unsqueeze(0).expand(mu.shape[0], -1, -1)
    transform_matrix = v * e.unsqueeze(1)
    box_corners = torch.bmm(base_corners, transform_matrix.transpose(1, 2))
    box_corners = box_corners + mu.unsqueeze(1)
    return box_corners

def plot_single_crop(ax, pts_gt, pts_pred, img_crop, box_crop, mask_crop, title, mode='all'):
    """绘制单个裁剪子图"""
    if img_crop is not None:
        ax.imshow(img_crop)
    else:
        ax.set_facecolor('white')
        
    c_gt = 'yellow'
    c_pred = 'blue'
    
    if mode in ['gt', 'all'] and pts_gt is not None:
        ax.scatter(pts_gt[:, 0], pts_gt[:, 1], c=c_gt, s=20, alpha=0.9, edgecolors='k', linewidth=0.3, label='GT')
    if mode in ['pred', 'all'] and pts_pred is not None:
        ax.scatter(pts_pred[:, 0], pts_pred[:, 1], c=c_pred, s=20, alpha=0.7, edgecolors='white', linewidth=0.3, label='Pred')

    if mode == 'box' and mask_crop is not None:
        mask_overlay = np.zeros_like(img_crop)
        if mask_overlay.ndim == 3:
            mask_overlay[mask_crop > 0, 0] = 1.0
        ax.imshow(mask_overlay, alpha=0.4)

    if mode == 'box' and box_crop is not None:
        poly = Polygon(box_crop, closed=True, edgecolor='lime', facecolor='none', linewidth=2, label='Pred Box')
        ax.add_patch(poly)
        
    ax.set_title(title, fontsize=10)
    ax.axis('off')

def save_instance_crop(save_path, img_np, mask_np, gt_pts, pred_pts, mu, sigma, loss_val, prefix, num_objects):
    """
    保存单个实例的 1x4 拼图
    """
    try:
        # 1. 确定裁剪范围
        cx, cy = mu[0].detach().item(), mu[1].detach().item()
        
        e = torch.linalg.eigvalsh(sigma.detach())[0].sqrt() 
        max_side = e.max().item() * 6.0 
        max_side = max(max_side, 80) 
        
        x_min = int(max(0, cx - max_side))
        x_max = int(min(img_np.shape[1], cx + max_side))
        y_min = int(max(0, cy - max_side))
        y_max = int(min(img_np.shape[0], cy + max_side))
        
        # Crop Data
        img_crop = img_np[y_min:y_max, x_min:x_max]
        
        raw_mask_crop = mask_np[y_min:y_max, x_min:x_max]
        foreground_mask_crop = raw_mask_crop.copy()
        bg_mask = (raw_mask_crop > num_objects) | (raw_mask_crop <= 0)
        foreground_mask_crop[bg_mask] = 0
        
        offset = torch.tensor([x_min, y_min], device=mu.device)
        pts_gt_local = (gt_pts - offset).detach().cpu().numpy() if gt_pts is not None else None
        pts_pred_local = (pred_pts - offset).detach().cpu().numpy()
        
        box_global = get_rotated_box_corners(mu.unsqueeze(0), sigma.unsqueeze(0))[0]
        box_local = (box_global - offset).detach().cpu().numpy()

        # 2. Plotting
        plt.ioff()
        fig, axes = plt.subplots(1, 4, figsize=(20, 5), dpi=100)
        
        plot_single_crop(axes[0], pts_gt_local, None, img_crop * 0.5, None, None, "GT Points (Yellow)", mode='gt')
        plot_single_crop(axes[1], None, pts_pred_local, img_crop * 0.5, None, None, "Pred Points (Blue)", mode='pred')
        plot_single_crop(axes[2], pts_gt_local, pts_pred_local, img_crop * 0.5, None, None, f"Overlay (Loss: {loss_val:.2f})", mode='all')
        plot_single_crop(axes[3], None, None, img_crop, box_local, foreground_mask_crop, "Macro: Box(Green) vs Mask(Red)", mode='box')

        plt.suptitle(f"{prefix} Instance Analysis", fontsize=12)
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close(fig)
        
    except Exception as e:
        print(f"[Instance Vis Error] {e}")
        plt.close('all')

def save_global_view(save_dir, img_np, mask_np, gt_points_list, pred_grid_points, mu, sigma, num_objects):
    """
    保存全图可视化
    """
    try:
        all_gt_pts = []
        for pts in gt_points_list:
            if pts is not None: all_gt_pts.append(pts.detach().cpu())
        all_gt_pts = torch.cat(all_gt_pts, dim=0) if len(all_gt_pts) > 0 else None
        
        pred_pts = pred_grid_points.detach().cpu().reshape(-1, 2)
        if torch.isnan(pred_pts).any(): return
        
        pred_boxes = get_rotated_box_corners(mu, sigma).detach().cpu().numpy()
        
        plt.ioff()
        # --- Plot 1: Global Points ---
        fig1 = plt.figure(figsize=(12, 12))
        plt.imshow(img_show_norm(img_np))
        if all_gt_pts is not None:
            plt.scatter(all_gt_pts[:, 0], all_gt_pts[:, 1], c='yellow', s=2, alpha=0.5, label='GT')
        plt.scatter(pred_pts[:, 0], pred_pts[:, 1], c='blue', s=2, alpha=0.3, label='Pred')
        plt.axis('off')
        plt.title('Global Point Sampling')
        plt.legend()
        plt.savefig(os.path.join(save_dir, 'global_points.jpg'))
        plt.close(fig1)
        
        # --- Plot 2: Global Macro ---
        fig2 = plt.figure(figsize=(12, 12))
        plt.imshow(img_show_norm(img_np))
        
        mask_overlay = np.zeros_like(img_np)
        mask_indices = (mask_np >= 1) & (mask_np <= num_objects)
        if mask_indices.any():
            mask_overlay[mask_indices, 0] = 1.0
        plt.imshow(mask_overlay, alpha=0.4)
        
        ax = plt.gca()
        for box in pred_boxes:
            if not np.isnan(box).any():
                poly = Polygon(box, closed=True, edgecolor='lime', facecolor='none', linewidth=1)
                ax.add_patch(poly)
        
        plt.axis('off')
        plt.title('Global Macro: Mask(Red) vs Box(Green)')
        plt.savefig(os.path.join(save_dir, 'global_macro.jpg'))
        plt.close(fig2)
        
    except Exception as e:
        print(f"[Global Vis Error] {e}")
        plt.close('all')

def img_show_norm(img):
    if img.max() > img.min():
        return (img - img.min()) / (img.max() - img.min())
    return img

@weighted_loss
def gwd_sigma_loss(pred, target, fun='log1p', tau=1.0, alpha=1.0, normalize=True):
    Sigma_p, Sigma_t = pred, target
    whr = Sigma_p.diagonal(dim1=-2, dim2=-1).sum(-1) + Sigma_t.diagonal(dim1=-2, dim2=-1).sum(-1)
    _t_tr = (Sigma_p.bmm(Sigma_t)).diagonal(dim1=-2, dim2=-1).sum(-1)
    _t_det = (Sigma_p.det() * Sigma_t.det()).clamp(1e-7).sqrt()
    whr = whr + (-2) * ((_t_tr + 2 * _t_det).clamp(1e-7).sqrt())
    dist = (alpha * alpha * whr).clamp(1e-7).sqrt()
    if normalize:
        scale = 2 * (_t_det.clamp(1e-7).sqrt().clamp(1e-7).sqrt()).clamp(1e-7)
        dist = dist / scale
    return postprocess(dist, fun=fun, tau=tau)

def gaussian_2d(xy, mu, sigma, normalize=False):
    dxy = (xy - mu).unsqueeze(-1)
    t0 = torch.exp(-0.5 * dxy.permute(0, 2, 1).bmm(torch.linalg.solve(sigma, dxy)))
    if normalize:
        t0 = t0 / (2 * np.pi * sigma.det().clamp(1e-7).sqrt())
    return t0

def get_gaussian_grid_points(mu, sigma, grid_size=10):
    B = mu.shape[0]
    device = mu.device
    # Range [-1, 1] 保障推理解码尺寸对齐
    step = torch.linspace(-1.0, 1.0, grid_size, device=device)
    grid_y, grid_x = torch.meshgrid(step, step, indexing='ij')
    base_grid = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=1)
    base_grid = base_grid.unsqueeze(0).expand(B, -1, -1)
    e, v = torch.linalg.eigh(sigma) 
    e = e.clamp(min=1e-7).sqrt()
    transform_matrix = v * e.unsqueeze(1)
    pred_points = torch.bmm(base_grid, transform_matrix.transpose(1, 2))
    pred_points = pred_points + mu.unsqueeze(1)
    return pred_points

def compute_chamfer_distance(pred_points, gt_points_list, img_meta=None, trim_ratio=0.1):
    """
    【鲁棒版】Chamfer Distance 匹配:
    1. L2不加平方 (p=2): 保障旋转不变性，防止远距离梯度爆炸 (恒定导数1)
    2. Trimmed Mean: 截断10%最大距离的离群噪声点
    """
    batch_losses = []
    if img_meta is not None:
        H, W = img_meta 
        scale = pred_points.new_tensor([W, H])
    else:
        scale = pred_points.new_tensor([1024.0, 1024.0])
    scale = scale.unsqueeze(0)
    per_sample_losses = []

    for i in range(len(pred_points)):
        p_pts = pred_points[i] / scale 
        g_pts = gt_points_list[i]
        
        if g_pts is None or len(g_pts) < 1:
            batch_losses.append(pred_points[i].sum() * 0)
            per_sample_losses.append(0.0)
            continue
            
        g_pts = g_pts.float() / scale
        
        # 【修改 1】使用欧氏距离 (无平方)，防梯度爆炸，保留旋转各向同性
        dist_mat = torch.cdist(p_pts, g_pts, p=2)
        
        min_p2g, _ = dist_mat.min(dim=1)
        min_g2p, _ = dist_mat.min(dim=0)
        
        # 【修改 2】引入截断平均去噪 (Trimmed Mean)
        def trimmed_mean(x, ratio):
            if x.numel() == 0: return x.sum() * 0
            # 丢弃最远的 ratio 比例的点
            k = max(1, int(x.numel() * (1.0 - ratio)))
            topk_vals, _ = torch.topk(x, k, largest=False)
            return topk_vals.mean()
            
        loss_p2g = trimmed_mean(min_p2g, trim_ratio)
        loss_g2p = trimmed_mean(min_g2p, trim_ratio)
        
        sample_loss = (loss_p2g + loss_g2p) * 100.0
        batch_losses.append(sample_loss)
        per_sample_losses.append(sample_loss.item())
        
    return torch.stack(batch_losses), per_sample_losses

@MODELS.register_module()
class OurWaterLoss(nn.Module):
    global_vis_counter = 0

    def __init__(self,
                 down_sample=2,
                 loss_weight=1.0,
                 voronoi_type='gaussian-orientation',
                 pos_thres_default=[0.994, 0.005],
                 default_sigma=4096,
                 num_gt_samples=256,  
                 grid_size=16,        
                 topk=0.95, 
                 debug=False,
                 vis_interval=100):
        super(OurWaterLoss, self).__init__()
        self.down_sample = down_sample
        self.loss_weight = loss_weight
        self.voronoi_type = voronoi_type
        self.pos_thres_default = pos_thres_default
        self.default_sigma = default_sigma
        self.num_gt_samples = num_gt_samples
        self.grid_size = grid_size
        self.topk = topk 
        self.debug = debug
        self.vis_interval = vis_interval

    def forward(self, pred, label, image, pos_thres, neg_thres, voronoi=None):
        mu, sigma = pred
        if len(mu) == 0: return mu.sum() * 0
        if voronoi is None: voronoi = self.voronoi_type
            
        J, D = len(mu), self.down_sample
        H, W = image.shape[-2:] 
        h, w = H // D, W // D
        
        x, y = torch.linspace(0, h, h, device=mu.device), torch.linspace(0, w, w, device=mu.device)
        xy = torch.stack(torch.meshgrid(x, y, indexing='xy'), -1)
        vor_map = mu.new_zeros(J, h, w)
        mm = (mu.detach() / D).round()
        
        if voronoi == 'standard':
            sg = sigma.new_tensor((self.default_sigma, 0, 0, self.default_sigma)).reshape(2, 2) / D ** 2
            for j, m in enumerate(mm): vor_map[j] = gaussian_2d(xy.view(-1, 2), m[None], sg[None]).view(h, w)
        elif voronoi == 'gaussian-orientation':
            L, V = torch.linalg.eigh(sigma)
            L = L.detach().clone() / (L[:, 0:1] * L[:, 1:2]).sqrt() * self.default_sigma
            sg = V.matmul(torch.diag_embed(L)).matmul(V.permute(0, 2, 1)).detach() / D ** 2
            for j, (m, s) in enumerate(zip(mm, sg)): vor_map[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)
        elif voronoi == 'gaussian-full':
            sg = sigma.detach() / D ** 2
            for j, (m, s) in enumerate(zip(mm, sg)): vor_map[j] = gaussian_2d(xy.view(-1, 2), m[None], s[None]).view(h, w)

        val, vor_idx = torch.max(vor_map, 0)
        if D > 1:
            vor_idx = vor_idx[:, None, :, None].expand(-1, D, -1, D).reshape(H, W)
            val = F.interpolate(val[None, None], (H, W), mode='bilinear', align_corners=True)[0, 0]
            
        kernel = val.new_ones((1, 1, 3, 3))
        kernel[0, 0, 1, 1] = -8
        ridges = torch.conv2d(vor_idx[None].float(), kernel, padding=1)[0] != 0
        vor_idx += 1 
        pos_t, neg_t = val.new_tensor(pos_thres), val.new_tensor(neg_thres)
        cls = label[vor_idx - 1] 
        vor_idx[val < pos_t[cls]] = 0 
        vor_idx[val < neg_t[cls]] = J + 1 
        vor_idx[ridges] = J + 1 
        
        img_uint8 = (image - image.min()) / (image.max() - image.min()) * 255
        img_uint8 = img_uint8.permute(1, 2, 0).detach().cpu().numpy().astype(np.uint8)
        img_uint8 = cv2.medianBlur(img_uint8, 3)
        markers_np = vor_idx.detach().cpu().numpy().astype(np.int32)
        cv2.watershed(img_uint8, markers_np)
        markers = torch.from_numpy(markers_np).to(mu.device)

        pred_grid_points = get_gaussian_grid_points(mu, sigma, self.grid_size)
        gt_points_list = []
        for j in range(J):
            mask_pts = (markers == (j + 1)).nonzero() 
            if len(mask_pts) < 10: 
                gt_points_list.append(None)
                continue
            
            mask_pts = mask_pts[:, [1, 0]].float()
            num_pts = len(mask_pts)
            
            # 使用完全无序的随机化策略打破扫线偏见
            if num_pts > self.num_gt_samples:
                # 均匀无重复随机采样
                indices = torch.randperm(num_pts, device=mu.device)[:self.num_gt_samples]
                samples = mask_pts[indices]
            else:
                # 针对小目标的 Bootstrap 有放回重采样，补齐权重
                indices = torch.randint(0, num_pts, (self.num_gt_samples,), device=mu.device)
                samples = mask_pts[indices]
                
            gt_points_list.append(samples)

        losses, loss_values = compute_chamfer_distance(pred_grid_points, gt_points_list, img_meta=(H, W), trim_ratio=0.1)
        k = int(np.ceil(len(losses) * self.topk))
        loss_final = torch.topk(losses, k, largest=False)[0].mean() if k > 0 else losses.mean()

        # Vis
        OurWaterLoss.global_vis_counter += 1
        if self.debug and (OurWaterLoss.global_vis_counter % self.vis_interval == 0):
            save_base = 'debug_vis_instances'
            iter_dir = os.path.join(save_base, f'iter_{OurWaterLoss.global_vis_counter:06d}')
            os.makedirs(iter_dir, exist_ok=True)
            try:
                img_np = image.detach().float()
                if img_np.dim() == 3: img_np = img_np.permute(1, 2, 0)
                img_np = img_np.cpu().numpy()
                img_np = img_show_norm(img_np)
                mask_np = markers.detach().cpu().numpy()
                
                # 1. Global
                save_global_view(iter_dir, img_np, mask_np, gt_points_list, pred_grid_points, mu, sigma, J)
                
                # 2. Local Instances
                valid_indices = [i for i, v in enumerate(loss_values) if v > 0]
                if len(valid_indices) > 0:
                    valid_losses = [loss_values[i] for i in valid_indices]
                    max_val = max(valid_losses)
                    max_idx = valid_indices[valid_losses.index(max_val)]
                    save_instance_crop(
                        os.path.join(iter_dir, 'micro_worst.jpg'),
                        img_np, mask_np, gt_points_list[max_idx], pred_grid_points[max_idx],
                        mu[max_idx], sigma[max_idx], max_val, "WORST", J
                    )
                    min_val = min(valid_losses)
                    min_idx = valid_indices[valid_losses.index(min_val)]
                    save_instance_crop(
                        os.path.join(iter_dir, 'micro_best.jpg'),
                        img_np, mask_np, gt_points_list[min_idx], pred_grid_points[min_idx],
                        mu[min_idx], sigma[min_idx], min_val, "BEST", J
                    )
            except Exception as e:
                print(f"[Vis Critical Error] {e}")

        return self.loss_weight * loss_final