#!/usr/bin/env python3
"""
自动化实验调度脚本
功能：
1. 定义多个实验配置（不同参数组合）
2. 自动基于模板生成 config 文件
3. 检测空闲 GPU，自动启动训练
4. 训练完成后自动解析日志，提取每个 epoch 的 mAP
5. 汇总所有实验结果，生成对比表格

用法：
  python auto_experiments.py --run      # 运行所有实验（自动调度GPU）
  python auto_experiments.py --summary  # 只汇总已有结果
  python auto_experiments.py --status   # 查看当前运行状态
"""

import os
import sys
import json
import time
import argparse
import subprocess
from pathlib import Path
from datetime import datetime

# ============================================================
# 配置区
# ============================================================

CODE_DIR = "/mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline"
PYTHON = "/home/liurunxiang/miniconda3/envs/cuda128/bin/python"
WORK_DIR_BASE = "work_dirs/sb/dt1"
TEMPLATE_CONFIG = "configs/point2rbox_v2/point2rbox_v2-improved-dota.py"

# 实验定义：每个实验是一个字典，包含要修改的参数
# 支持的参数：size_weight, angle_warmup_epochs, lr, max_epochs, milestones
EXPERIMENTS = [
    # --- 综合改进版（已在跑）---
    {
        "name": "improved-full",
        "desc": "Angle warmup2 + Size5.0 + lr1e-5 + 24ep",
        "size_weight": 5.0,
        "angle_warmup_epochs": 2,
        "lr": 0.00001,
        "max_epochs": 24,
        "milestones": [16, 22],
    },
    # --- 消融1：只有 Size 提权 ---
    {
        "name": "ablate-size-only",
        "desc": "Size5.0 only (angle no warmup, lr5e-5, 12ep)",
        "size_weight": 5.0,
        "angle_warmup_epochs": 0,
        "lr": 0.00005,
        "max_epochs": 12,
        "milestones": [8, 11],
    },
    # --- 消融2：只有 Angle warmup ---
    {
        "name": "ablate-angle-warmup-only",
        "desc": "Angle warmup2 only (size1.0, lr5e-5, 12ep)",
        "size_weight": 1.0,
        "angle_warmup_epochs": 2,
        "lr": 0.00005,
        "max_epochs": 12,
        "milestones": [8, 11],
    },
    # --- 消融3：只有 lr 降低 ---
    {
        "name": "ablate-lr-only",
        "desc": "lr1e-5 only (size1.0, angle no warmup, 12ep)",
        "size_weight": 1.0,
        "angle_warmup_epochs": 0,
        "lr": 0.00001,
        "max_epochs": 12,
        "milestones": [8, 11],
    },
    # --- 消融4：Size10.0（更激进）---
    {
        "name": "ablate-size-10",
        "desc": "Size10.0 + angle warmup2 + lr1e-5 + 24ep",
        "size_weight": 10.0,
        "angle_warmup_epochs": 2,
        "lr": 0.00001,
        "max_epochs": 24,
        "milestones": [16, 22],
    },
    # --- 消融5：36 epoch（更长训练）---
    {
        "name": "ablate-36ep",
        "desc": "36ep + Size5.0 + angle warmup2 + lr1e-5",
        "size_weight": 5.0,
        "angle_warmup_epochs": 2,
        "lr": 0.00001,
        "max_epochs": 36,
        "milestones": [24, 33],
    },
]

# ============================================================
# 工具函数
# ============================================================

def run_cmd(cmd, cwd=None, timeout=None):
    """执行 shell 命令"""
    result = subprocess.run(
        cmd, shell=True, cwd=cwd,
        capture_output=True, text=True, timeout=timeout
    )
    return result.stdout.strip(), result.stderr.strip(), result.returncode


def get_free_gpus():
    """获取空闲 GPU 列表（显存占用 < 2GB 视为空闲）"""
    out, _, _ = run_cmd(
        "nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits"
    )
    free = []
    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        gpu_id = int(parts[0].strip())
        mem_used = float(parts[1].strip())
        if mem_used < 2000:  # 小于 2GB 视为空闲
            free.append(gpu_id)
    return free


def generate_config(exp, template_path, output_path):
    """基于模板生成实验 config"""
    with open(template_path, "r") as f:
        content = f.read()

    # 修改 size_weight
    content = content.replace(
        "loss_weight=5.0,\n            beta=1.0,\n            topk=0.95,\n            target_classes=[0,1,4,7,8,9,10,11,14]",
        f"loss_weight={exp['size_weight']},\n            beta=1.0,\n            topk=0.95,\n            target_classes=[0,1,4,7,8,9,10,11,14]"
    )

    # 修改 angle warmup
    content = content.replace(
        "warmup_epochs=2,",
        f"warmup_epochs={exp['angle_warmup_epochs']},"
    )

    # 修改 lr
    content = content.replace(
        "lr=0.00001,",
        f"lr={exp['lr']},"
    )

    # 修改 max_epochs
    content = content.replace(
        "max_epochs=24,",
        f"max_epochs={exp['max_epochs']},"
    )

    # 修改 milestones
    old_ms = f"milestones=[16, 22],"
    new_ms = f"milestones={exp['milestones']},"
    content = content.replace(old_ms, new_ms)

    # 修改 MultiStepLR 的 end
    content = content.replace(
        "end=24,",
        f"end={exp['max_epochs']},"
    )

    with open(output_path, "w") as f:
        f.write(content)

    print(f"  Config generated: {output_path}")


def start_training(exp, gpu_id):
    """启动训练"""
    config_path = f"configs/point2rbox_v2/auto-{exp['name']}.py"
    work_dir = f"{WORK_DIR_BASE}/{exp['name']}"
    log_file = f"{work_dir}-nohup.log"

    # 生成 config
    template = os.path.join(CODE_DIR, TEMPLATE_CONFIG)
    output = os.path.join(CODE_DIR, config_path)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    generate_config(exp, template, output)

    # 创建 work_dir
    os.makedirs(os.path.join(CODE_DIR, work_dir), exist_ok=True)

    # 启动训练（nohup 后台）
    cmd = (
        f"cd {CODE_DIR} && "
        f"CUDA_VISIBLE_DEVICES={gpu_id} nohup {PYTHON} tools/train.py "
        f"{config_path} --work-dir {work_dir} > {log_file} 2>&1 &"
    )
    run_cmd(cmd)

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
          f"Started '{exp['name']}' on GPU{gpu_id}")
    print(f"    Config: {config_path}")
    print(f"    Work dir: {work_dir}")
    print(f"    Log: {log_file}")

    return work_dir, log_file


def is_training_running(exp_name):
    """检查实验是否正在运行"""
    out, _, _ = run_cmd(f"ps aux | grep 'auto-{exp_name}' | grep -v grep")
    return bool(out.strip())


def parse_log(work_dir):
    """解析训练日志，提取每个 epoch 的 mAP"""
    # 找到 vis_data 目录下的 .log 文件
    log_dir = os.path.join(CODE_DIR, work_dir)
    if not os.path.exists(log_dir):
        return None

    # 找最新的 .log 文件
    log_files = []
    for root, dirs, files in os.walk(log_dir):
        for f in files:
            if f.endswith(".log") and "vis_data" in root:
                log_files.append(os.path.join(root, f))

    if not log_files:
        return None

    log_file = max(log_files, key=os.path.getmtime)

    results = []
    with open(log_file, "r") as f:
        for line in f:
            if "dota/mAP:" in line:
                # 解析: Epoch(val) [1316/1316]    dota/mAP: 0.2339  dota/AP50: 0.2340 ...
                parts = line.strip().split()
                epoch_idx = None
                for i, p in enumerate(parts):
                    if p == "Epoch(val)":
                        # 下一个是 [1316/1316]，但 epoch 号不在这行
                        # 需要从上下文推断，或者从 nohup.log 中找
                        pass
                # 直接提取 mAP 值
                for p in parts:
                    if p.startswith("dota/mAP:"):
                        mAP = float(p.split(":")[1])
                        results.append(mAP)

    return results if results else None


def get_epoch_maps(nohup_log):
    """从 nohup.log 中提取每个 epoch 的 mAP（更准确，有 epoch 号）"""
    log_path = os.path.join(CODE_DIR, nohup_log)
    if not os.path.exists(log_path):
        return []

    results = []
    current_epoch = 0
    with open(log_path, "r") as f:
        for line in f:
            # 检测 epoch 切换: Epoch(train)  [6400/6400]
            if "Epoch(train)" in line and "[6400/" in line:
                current_epoch += 1
            # 检测验证结果
            if "dota/mAP:" in line:
                parts = line.strip().split()
                for p in parts:
                    if p.startswith("dota/mAP:"):
                        mAP = float(p.split(":")[1])
                        results.append({"epoch": current_epoch, "mAP": mAP})

    return results


def summarize_results():
    """汇总所有实验结果"""
    print("\n" + "=" * 100)
    print("实验结果汇总")
    print("=" * 100)
    print(f"{'实验名称':<25} {'描述':<45} {'最佳mAP':<10} {'最佳epoch':<10} {'最终mAP':<10} {'状态':<10}")
    print("-" * 100)

    all_results = []
    for exp in EXPERIMENTS:
        work_dir = f"{WORK_DIR_BASE}/{exp['name']}"
        nohup_log = f"{work_dir}-nohup.log"

        running = is_training_running(exp["name"])
        epoch_maps = get_epoch_maps(nohup_log)

        if epoch_maps:
            best = max(epoch_maps, key=lambda x: x["mAP"])
            final = epoch_maps[-1]
            best_map = f"{best['mAP']:.4f}"
            best_epoch = str(best["epoch"])
            final_map = f"{final['mAP']:.4f}"
            status = "运行中" if running else "已完成"
        else:
            best_map = "-"
            best_epoch = "-"
            final_map = "-"
            status = "运行中" if running else "未开始"

        print(f"{exp['name']:<25} {exp['desc']:<45} {best_map:<10} {best_epoch:<10} {final_map:<10} {status:<10}")

        all_results.append({
            "name": exp["name"],
            "desc": exp["desc"],
            "best_map": best_map,
            "best_epoch": best_epoch,
            "final_map": final_map,
            "status": status,
            "epoch_maps": epoch_maps,
        })

    print("=" * 100)

    # 保存为 JSON
    summary_path = os.path.join(CODE_DIR, WORK_DIR_BASE, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)
    print(f"\n结果已保存到: {summary_path}")

    return all_results


def run_experiments():
    """运行所有实验（自动调度 GPU）"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 开始自动化实验调度")
    print(f"共 {len(EXPERIMENTS)} 个实验")

    # 检查哪些实验已经完成或正在运行
    pending = []
    for exp in EXPERIMENTS:
        if is_training_running(exp["name"]):
            print(f"  [跳过] '{exp['name']}' 正在运行")
            continue
        epoch_maps = get_epoch_maps(f"{WORK_DIR_BASE}/{exp['name']}-nohup.log")
        if epoch_maps and len(epoch_maps) >= exp["max_epochs"]:
            print(f"  [跳过] '{exp['name']}' 已完成 ({len(epoch_maps)}/{exp['max_epochs']} epochs)")
            continue
        pending.append(exp)

    print(f"\n待运行实验: {len(pending)} 个")

    # 循环调度，直到所有实验完成
    while pending:
        free_gpus = get_free_gpus()
        if not free_gpus:
            print(f"  [{datetime.now().strftime('%H:%M:%S')}] 无空闲 GPU，等待 60s...")
            time.sleep(60)
            continue

        # 为每个空闲 GPU 分配一个实验
        for gpu_id in free_gpus:
            if not pending:
                break
            exp = pending.pop(0)
            try:
                start_training(exp, gpu_id)
            except Exception as e:
                print(f"  [错误] 启动 '{exp['name']}' 失败: {e}")
                pending.append(exp)

        # 等待一段时间后再检查
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] 等待 5 分钟后检查...")
        time.sleep(300)

    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 所有实验完成！")
    summarize_results()


def show_status():
    """显示当前运行状态"""
    print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 当前状态")
    print(f"空闲 GPU: {get_free_gpus()}")

    for exp in EXPERIMENTS:
        running = is_training_running(exp["name"])
        epoch_maps = get_epoch_maps(f"{WORK_DIR_BASE}/{exp['name']}-nohup.log")
        status = "运行中" if running else ("已完成" if epoch_maps and len(epoch_maps) >= exp["max_epochs"] else "未开始")
        progress = f"{len(epoch_maps)}/{exp['max_epochs']}" if epoch_maps else "0/?"
        best = max([e["mAP"] for e in epoch_maps]) if epoch_maps else "-"
        print(f"  {exp['name']:<25} {status:<8} epoch={progress:<10} best_mAP={best}")


# ============================================================
# 主入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="自动化实验调度")
    parser.add_argument("--run", action="store_true", help="运行所有实验")
    parser.add_argument("--summary", action="store_true", help="汇总已有结果")
    parser.add_argument("--status", action="store_true", help="查看当前状态")
    args = parser.parse_args()

    if args.run:
        run_experiments()
    elif args.summary:
        summarize_results()
    elif args.status:
        show_status()
    else:
        parser.print_help()
