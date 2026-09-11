# audit-fixed-lr25.py: 学习率降到 2.5e-5（G3 动态决策备用，探中间学习率）
# 单变量: lr 5e-5 -> 2.5e-5
_base_ = ['./audit-fixed-baseline-s0.py']
optim_wrapper = dict(
    optimizer=dict(lr=2.5e-5))
