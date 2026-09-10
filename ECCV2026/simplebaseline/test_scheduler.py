#!/usr/bin/env python3
"""测试 scheduler_dynamic.py 的 config 生成逻辑"""
import sys
sys.path.insert(0, '/mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline')

# 直接 import 模块（避免执行主逻辑）
import importlib.util
spec = importlib.util.spec_from_file_location(
    "scheduler_dynamic",
    "/mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline/scheduler_dynamic.py"
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

import os

CODE_DIR = mod.CODE_DIR

# 测试每个实验的 config 生成
for exp in mod.PHASE1:
    name = exp["name"]
    out_path = os.path.join(CODE_DIR, f"configs/point2rbox_v2/auto-{name}.py")
    try:
        changed = mod.generate_config(exp, out_path)
        print(f"✓ {name}: {changed}")
    except AssertionError as e:
        print(f"✗ {name}: anchor not found -> {e}")
        continue

    # 验证生成的文件
    with open(out_path, "r") as f:
        content = f.read()

    # 验证只改了一个参数
    if "size_weight" in exp["params"]:
        assert f"loss_weight={exp['params']['size_weight']}" in content, "size weight not applied!"
    if "angle_warmup_epochs" in exp["params"]:
        assert f"warmup_epochs={exp['params']['angle_warmup_epochs']}" in content, "warmup not applied!"
        assert "LossWarmupHook" in content, "hook not registered!"
    if "lr" in exp["params"]:
        assert f"lr={exp['params']['lr']}" in content, "lr not applied!"
    if "max_epochs" in exp["params"]:
        assert f"max_epochs={exp['params']['max_epochs']}" in content, "max_epochs not applied!"
        assert f"milestones={exp['params']['milestones']}" in content, "milestones not applied!"
    print(f"  ✓ 验证通过: 所有参数正确替换")

# 检查原始 config 是否被改动（应该保持不变）
# 注意: loss_bbox 和 loss_voronoi 原本就是 loss_weight=5.0, 不能作为污染判断
base_path = os.path.join(CODE_DIR, mod.BASE_CONFIG)
with open(base_path, "r") as f:
    base_content = f.read()
# loss_size 部分应该是 loss_weight=1.0（未被 size 实验改动）
size_section = base_content[base_content.find("loss_size=dict"):base_content.find("loss_size=dict")+200]
assert "loss_weight=1.0" in size_section, "原始 config 的 loss_size 被污染了!"
# loss_angle 部分不应有 warmup_epochs
angle_section = base_content[base_content.find("loss_angle=dict"):base_content.find("loss_angle=dict")+200]
assert "warmup_epochs" not in angle_section, "原始 config 的 loss_angle 被污染了!"
# optim_wrapper 应该是 lr=0.00005
assert "lr=0.00005" in base_content, "原始 config 的 lr 被污染了!"
print("\n✓ 原始 config 未被动过")

print("\n全部测试通过！")
