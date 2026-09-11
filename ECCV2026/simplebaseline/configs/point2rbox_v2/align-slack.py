# align-slack.py: 对齐实验⑥ §3.6 Slack 丢弃 top-10%
# 论文: omitting the top-10% loss values (丢最大 10%)
# 代码: AngleLoss/SizeLoss topk=0.95 (丢 5%); OurWaterLoss 无 slack
# 仅改三个 loss 的 topk 到 0.90 (统一 slack 一处), 其余与 audit-fixed-ow3.py 完全一致
_base_ = ['./audit-fixed-ow3.py']
model = dict(
    bbox_head=dict(
        loss_angle=dict(type='AngleLoss', topk=0.90),
        loss_size=dict(type='SizeLoss', topk=0.90),
        loss_ourwater=dict(type='OurWaterLoss', topk=0.90)))
