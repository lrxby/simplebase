#!/usr/bin/env python3
"""
动态自适应实验调度器 v3（GPU 并行复用版）
==========================================
设计原则：
  1. 严格单变量消融：每个实验基于【原始 config】只修改一个参数，其余与基线完全一致
  2. 动态决策：Phase 1 单变量消融完成后，只组合【被证明有效】的修改进入 Phase 2
  3. 自动调度：常驻后台，自动检测 GPU 剩余显存、并行启动多个实验、自动解析结果、自动决策
  4. GPU 并行复用：单个实验只占 ~4GB 显存，同一 GPU 按剩余显存可同时跑多个实验（单卡上限 3 个）
  5. 断点续跑：重启脚本后自动识别已完成实验，不重复跑

基线：pca/2-0.403 最佳 mAP = 0.403（DOTA-v1.0，12ep, lr5e-5, size1.0, angle无warmup）
判定标准：单变量实验 best mAP > 基线 + 0.003 视为有效

用法：
  后台常驻：nohup python scheduler_dynamic.py > scheduler.log 2>&1 &
  查看状态：python scheduler_dynamic.py --status
  手动总结：python scheduler_dynamic.py --summary
"""

import os
import re
import sys
import json
import time
import argparse
import subprocess
from datetime import datetime

# ============================================================
# 常量配置
# ============================================================

CODE_DIR = "/mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline"
PYTHON = "/home/liurunxiang/miniconda3/envs/cuda128/bin/python"
WORK_DIR_BASE = "work_dirs/sb/dt1"
BASE_CONFIG = "configs/point2rbox_v2/point2rbox_v2-1x-dota.py"  # 原始基线 config
STATE_FILE = os.path.join(CODE_DIR, WORK_DIR_BASE, "scheduler_state.json")

BASELINE_MAP = 0.413   # 基线最佳 mAP —— 注意: 用 rect 伪标签的基线 rect/2-0.413 (best@ep10)!
                       # 不能用 pca 的 0.403: 所有单变量实验都用 dota1-rect.pkl (与论文一致),
                       # 公平对比必须与同伪标签的 rect 基线比
IMPROVE_THRESHOLD = 0.003  # 超过基线 0.3 mAP 视为有效改进
MEM_PER_EXP = 10000    # 每个实验预留显存 MB（训练实际 ~4GB，留足峰值缓冲防崩）
MAX_EXP_PER_GPU = 2    # 单卡最大并行实验数（硬上限，用户要求每卡最多 2 个，稳一点）
MAX_EPOCHS_DEFAULT = 12

# Phase 1: 严格单变量消融（每个实验只改一个参数）
PHASE1 = [
    {
        "name": "single-size5",
        "desc": "仅改 Size 权重 1.0→5.0",
        "params": {"size_weight": 5.0},
    },
    {
        "name": "single-awarmup2",
        "desc": "仅改 Angle warmup 0→2 epoch",
        "params": {"angle_warmup_epochs": 2},
    },
    {
        "name": "single-lr1e-5",
        "desc": "仅改 lr 5e-5→1e-5",
        "params": {"lr": 0.00001},
    },
    {
        "name": "single-24ep",
        "desc": "仅改 epoch 12→24",
        "params": {"max_epochs": 24, "milestones": [16, 22]},
    },
]


# ============================================================
# 工具函数
# ============================================================

def now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def run_cmd(cmd, timeout=60):
    try:
        result = subprocess.run(
            cmd, shell=True, cwd=CODE_DIR,
            capture_output=True, text=True, timeout=timeout
        )
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    except Exception as e:
        return "", str(e), -1


def load_state():
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r") as f:
            return json.load(f)
    return {"phase1_results": {}, "phase2": None, "started_at": now()}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_ours_per_gpu():
    """统计每个 GPU 上正在运行的【我们的】训练实验数 {gpu_id: count}

    通过 /proc/<pid>/cmdline 识别我们的训练主进程（tools/train.py + auto-*.py/combined-best.py），
    再读 /proc/<pid>/environ 拿 CUDA_VISIBLE_DEVICES。
    注意 mmdet dataloader 的 fork 子进程会继承 cmdline，故按 (config名, gpu) 去重。
    """
    import re
    out, _, _ = run_cmd("ps aux | grep 'tools/train.py' | grep -v grep")
    counts = {}
    seen = set()
    for line in out.split("\n"):
        if not line.strip():
            continue
        parts = line.split()
        if len(parts) < 11:
            continue
        pid = parts[1]
        cmd = " ".join(parts[10:])
        m = re.search(r'(auto-[\w-]+\.py|combined-best\.py)', cmd)
        if not m:
            continue
        config_name = m.group(0)
        # 读该进程的 CUDA_VISIBLE_DEVICES
        gpu = None
        try:
            with open(f"/proc/{pid}/environ", "rb") as f:
                env = f.read().decode("utf-8", "ignore")
            for kv in env.split("\0"):
                if kv.startswith("CUDA_VISIBLE_DEVICES="):
                    gpu = kv.split("=")[1].strip()
                    break
        except Exception:
            continue
        if gpu is None:
            continue
        key = (config_name, gpu)
        if key in seen:
            continue
        seen.add(key)
        counts[gpu] = counts.get(gpu, 0) + 1
    return counts


def get_gpu_slots():
    """返回每个 GPU 还能启动几个实验: {gpu_id: slots}

    双重保险:
      1. 显存维度: 剩余显存 // MEM_PER_EXP（每实验预留充足显存, 防训练峰值崩掉）
      2. 数量维度: MAX_EXP_PER_GPU - 该卡已有实验数（硬上限, 防无限叠加）
    """
    out, _, _ = run_cmd(
        "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits"
    )
    ours = get_ours_per_gpu()
    slots = {}
    for line in out.strip().split("\n"):
        if not line.strip():
            continue
        parts = line.split(",")
        gpu_id = int(parts[0].strip())
        mem_used = float(parts[1].strip())
        mem_total = float(parts[2].strip())
        free = mem_total - mem_used
        by_mem = int(free // MEM_PER_EXP)
        by_count = MAX_EXP_PER_GPU - ours.get(str(gpu_id), 0)
        slots[gpu_id] = max(0, min(by_mem, by_count))
    return slots


def is_process_running(exp_name):
    out, _, _ = run_cmd(f"ps aux | grep 'auto-{exp_name}\\.py' | grep -v grep")
    return bool(out.strip())


def generate_config(exp, out_path):
    """基于【原始基线 config】生成单变量实验 config

    严格单变量原则：只修改 exp['params'] 中指定的参数，
    其他所有内容与原始 config 完全一致。
    """
    base_path = os.path.join(CODE_DIR, BASE_CONFIG)
    with open(base_path, "r") as f:
        content = f.read()

    params = exp["params"]
    changed = []

    # --- 修改 1: Size 权重 ---
    if "size_weight" in params:
        old = "loss_weight=1.0, \n            beta=1.0"
        new = f"loss_weight={params['size_weight']}, \n            beta=1.0"
        assert old in content, "size_weight anchor not found!"
        content = content.replace(old, new)
        changed.append(f"size_weight={params['size_weight']}")

    # --- 修改 2: Angle warmup ---
    if "angle_warmup_epochs" in params:
        # 在 loss_angle 的 topk 后插入 warmup_epochs
        old = "topk=0.95,\n            target_classes=[0,4,5,6,7,8,10,12,14]"
        new = (f"topk=0.95,\n            warmup_epochs={params['angle_warmup_epochs']},\n"
               f"            target_classes=[0,4,5,6,7,8,10,12,14]")
        assert old in content, "angle warmup anchor not found!"
        content = content.replace(old, new)
        # 同时添加 LossWarmupHook 导入和注册
        content = content.replace(
            "imports=['mmrotate.datasets.transforms.loading_pseudo'],",
            "imports=['mmrotate.datasets.transforms.loading_pseudo',\n         'mmrotate.engine.hooks.loss_warmup_hook'],"
        )
        content = content.replace(
            "custom_hooks = [dict(type='mmdet.SetEpochInfoHook')]",
            "custom_hooks = [dict(type='mmdet.SetEpochInfoHook'), dict(type='LossWarmupHook')]"
        )
        changed.append(f"angle_warmup={params['angle_warmup_epochs']}")

    # --- 修改 3: lr ---
    if "lr" in params:
        old = "lr=0.00005,"
        new = f"lr={params['lr']},"
        assert old in content, "lr anchor not found!"
        content = content.replace(old, new)
        changed.append(f"lr={params['lr']}")

    # --- 修改 4: max_epochs + milestones ---
    if "max_epochs" in params:
        old = "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)"
        new = f"train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={params['max_epochs']}, val_interval=1)"
        assert old in content, "max_epochs anchor not found!"
        content = content.replace(old, new)
        # 追加 param_scheduler 覆盖（必须放在文件末尾）
        ms = params.get("milestones", [16, 22])
        scheduler_override = (
            f"\nparam_scheduler = [\n"
            f"    dict(type='LinearLR', start_factor=1.0 / 3, by_epoch=False,\n"
            f"         begin=0, end=500),\n"
            f"    dict(type='MultiStepLR', begin=0, end={params['max_epochs']},\n"
            f"         by_epoch=True, milestones={ms}, gamma=0.1)\n"
            f"]\n"
        )
        content += scheduler_override
        changed.append(f"max_epochs={params['max_epochs']}")

    with open(out_path, "w") as f:
        f.write(content)

    return "; ".join(changed)


def start_training(exp, gpu_id):
    """启动单个实验训练"""
    name = exp["name"]
    config_path = f"configs/point2rbox_v2/auto-{name}.py"
    work_dir = f"{WORK_DIR_BASE}/{name}"
    log_file = os.path.join(CODE_DIR, f"{work_dir}-nohup.log")

    # 生成 config
    out_path = os.path.join(CODE_DIR, config_path)
    changed = generate_config(exp, out_path)

    # 创建 work_dir
    os.makedirs(os.path.join(CODE_DIR, work_dir), exist_ok=True)

    # 启动训练
    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} nohup {PYTHON} tools/train.py "
        f"{config_path} --work-dir {work_dir} > {log_file} 2>&1 &"
    )
    _, err, code = run_cmd(cmd)
    if code != 0:
        return False, f"启动失败: {err}"

    print(f"[{now()}] 启动实验 '{name}' @ GPU{gpu_id}")
    print(f"  修改项: {changed}")
    print(f"  日志: {log_file}")
    return True, f"started on GPU{gpu_id}"


def parse_epoch_maps(exp_name):
    """解析实验日志，返回每个 epoch 的 mAP 列表（正则提取，容错）"""
    log_file = os.path.join(CODE_DIR, f"{WORK_DIR_BASE}/{exp_name}-nohup.log")
    if not os.path.exists(log_file):
        return []

    _re_mAP = re.compile(r"dota/mAP:\s*([0-9.]+)")
    results = []
    current_epoch = 0
    with open(log_file, "r") as f:
        for line in f:
            if "Epoch(train)" in line and "[6400/" in line:
                current_epoch += 1
            m = _re_mAP.search(line)
            if m:
                results.append({"epoch": current_epoch, "mAP": float(m.group(1))})
    return results



def exp_is_finished(exp):
    """判断实验是否完成（达到 max_epochs）"""
    max_ep = exp["params"].get("max_epochs", MAX_EPOCHS_DEFAULT)
    maps = parse_epoch_maps(exp["name"])
    return len(maps) >= max_ep


def analyze_result(exp):
    """分析单个实验的结果，返回 verdict"""
    maps = parse_epoch_maps(exp["name"])
    if not maps:
        return None
    best = max(maps, key=lambda x: x["mAP"])
    final = maps[-1]
    gain = best["mAP"] - BASELINE_MAP
    effective = gain > IMPROVE_THRESHOLD
    return {
        "best_mAP": best["mAP"],
        "best_epoch": best["epoch"],
        "final_mAP": final["mAP"],
        "gain_vs_baseline": gain,
        "effective": effective,
    }


def build_phase2(state):
    """Phase 2: 动态组合所有【有效】的单变量修改"""
    effective_params = {}
    effective_names = []

    for exp in PHASE1:
        name = exp["name"]
        result = state["phase1_results"].get(name)
        if result and result.get("effective"):
            effective_params.update(exp["params"])
            effective_names.append(name)

    if not effective_params:
        return {
            "name": "phase2-none",
            "desc": "无有效修改，不生成组合实验",
            "params": {},
            "effective_sources": [],
        }

    return {
        "name": "combined-best",
        "desc": f"组合 {len(effective_names)} 项有效修改: {effective_names}",
        "params": effective_params,
        "effective_sources": effective_names,
    }


# ============================================================
# 主调度循环
# ============================================================

def scheduler_loop(once=False):
    print(f"[{now()}] 动态调度器启动")
    print(f"  基线 mAP: {BASELINE_MAP}, 有效阈值: +{IMPROVE_THRESHOLD}")
    print(f"  Phase1 实验数: {len(PHASE1)}")

    state = load_state()

    while True:
        state = load_state()
        gpu_slots = get_gpu_slots()

        # ---------- 1. 检查 Phase1 完成情况 ----------
        all_phase1_done = True
        for exp in PHASE1:
            name = exp["name"]
            if name in state["phase1_results"]:
                continue  # 已分析
            if is_process_running(name):
                all_phase1_done = False
                continue  # 正在跑
            if exp_is_finished(exp):
                # 刚完成，分析结果
                result = analyze_result(exp)
                if result:
                    verdict = "有效 ✓" if result["effective"] else "无效 ✗"
                    print(f"[{now()}] 实验 '{name}' 完成: "
                          f"best={result['best_mAP']:.4f}@ep{result['best_epoch']}, "
                          f"final={result['final_mAP']:.4f}, "
                          f"gain={result['gain_vs_baseline']:+.4f} → {verdict}")
                    state["phase1_results"][name] = result
                    save_state(state)
                    all_phase1_done = all_phase1_done and True
                else:
                    print(f"[{now()}] 警告: 实验 '{name}' 进程结束但无有效结果")
                    all_phase1_done = False
            else:
                all_phase1_done = False

        # ---------- 2. 全部 Phase1 完成 → 生成 Phase2 ----------
        if all_phase1_done and state["phase2"] is None:
            state["phase2"] = build_phase2(state)
            p2 = state["phase2"]
            if p2["params"]:
                print(f"[{now()}] Phase1 完成，生成 Phase2: {p2['desc']}")
            else:
                print(f"[{now()}] Phase1 完成，无有效修改可组合，实验终止。")
            save_state(state)

        # ---------- 3. 检查 Phase2 完成情况 ----------
        phase2 = state.get("phase2")
        phase2_done = True
        if phase2 and phase2["params"]:
            if "phase2_results" not in state:
                state["phase2_results"] = {}
            if "combined-best" not in state["phase2_results"]:
                if is_process_running("combined-best"):
                    phase2_done = False
                elif exp_is_finished(phase2):
                    result = analyze_result(phase2)
                    if result:
                        verdict = "有效 ✓" if result["effective"] else "无效 ✗"
                        print(f"[{now()}] Phase2 组合实验完成: "
                              f"best={result['best_mAP']:.4f}@ep{result['best_epoch']}, "
                              f"final={result['final_mAP']:.4f} → {verdict}")
                        state["phase2_results"]["combined-best"] = result
                        save_state(state)
                    else:
                        phase2_done = False
                else:
                    phase2_done = False

        # ---------- 4. 全部完成 → 退出 ----------
        if all_phase1_done and phase2_done:
            print(f"[{now()}] 🎉 所有实验完成！")
            summarize(state)
            if once:
                return
            break

        # ---------- 5. 分配 GPU 槽位 ----------
        # 按剩余显存并行分配: 每张卡可同时跑多个实验
        if any(s > 0 for s in gpu_slots.values()):
            # 收集所有待跑的 Phase1 实验（未完成且未在跑）
            pending = []
            for exp in PHASE1:
                name = exp["name"]
                if name in state["phase1_results"]:
                    continue
                if is_process_running(name):
                    continue
                pending.append(exp)

            # Phase1 没有待跑的, 尝试 Phase2
            phase2 = state.get("phase2")
            if not pending and phase2 and phase2["params"] \
                    and "phase2_results" not in state \
                    and not is_process_running("combined-best"):
                pending.append(phase2)

            # 依次分配: 每个待跑实验找一个还有槽位的 GPU
            for exp in pending:
                for gpu_id in sorted(gpu_slots.keys()):
                    if gpu_slots[gpu_id] > 0:
                        ok, msg = start_training(exp, gpu_id)
                        if ok:
                            gpu_slots[gpu_id] -= 1
                            # 启动后立即刷新进程状态, 避免同一轮重复分配
                            time.sleep(2)
                        break

        # 打印状态
        running = [e["name"] for e in PHASE1 if is_process_running(e["name"])]
        if state.get("phase2") and state["phase2"]["params"] and is_process_running("combined-best"):
            running.append("combined-best")
        done = list(state["phase1_results"].keys())
        slots_now = get_gpu_slots()
        slot_desc = ", ".join(f"GPU{k}:{v}槽" for k, v in sorted(slots_now.items()))
        print(f"[{now()}] 运行中: {running or '无'}, 已完成: {done or '无'}, GPU槽位: {slot_desc}")

        if once:
            return

        time.sleep(180)  # 每 3 分钟检查一次


# ============================================================
# 汇总
# ============================================================

def summarize(state=None):
    if state is None:
        state = load_state()

    print("\n" + "=" * 100)
    print("实验汇总（基线 best mAP = %.3f）" % BASELINE_MAP)
    print("=" * 100)
    print(f"{'实验':<22} {'修改':<38} {'best':<8} {'@ep':<5} {'final':<8} {'gain':<8} {'判定':<6}")
    print("-" * 100)

    for exp in PHASE1:
        name = exp["name"]
        result = state["phase1_results"].get(name)
        mod = "; ".join(f"{k}={v}" for k, v in exp["params"].items())
        if result:
            verdict = "有效" if result["effective"] else "无效"
            print(f"{name:<22} {mod:<38} {result['best_mAP']:.3f} "
                  f"{result['best_epoch']:<5} {result['final_mAP']:.3f} "
                  f"{result['gain_vs_baseline']:+.3f} {verdict}")
        else:
            print(f"{name:<22} {mod:<38} {'-':<8} {'-':<5} {'-':<8} {'-':<8} 未完成")

    p2 = state.get("phase2")
    if p2 and p2["params"]:
        r2 = state.get("phase2_results", {}).get("combined-best")
        if r2:
            print(f"{'combined-best':<22} {p2['desc']:<38} {r2['best_mAP']:.3f} "
                  f"{r2['best_epoch']:<5} {r2['final_mAP']:.3f} "
                  f"{r2['gain_vs_baseline']:+.3f} {'有效' if r2['effective'] else '无效'}")
        else:
            print(f"{'combined-best':<22} {p2['desc']:<38} 运行中/未完成")
    print("=" * 100)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动态自适应实验调度器")
    parser.add_argument("--status", action="store_true", help="查看当前状态")
    parser.add_argument("--summary", action="store_true", help="汇总结果")
    parser.add_argument("--once", action="store_true", help="只跑一轮检查")
    args = parser.parse_args()

    if args.status:
        state = load_state()
        print(f"[{now()}] 调度器状态")
        slots = get_gpu_slots()
        slot_desc = ", ".join(f"GPU{k}:{v}槽" for k, v in sorted(slots.items()))
        print(f"  GPU槽位: {slot_desc}")
        summarize(state)
        # 显示每个实验进度
        for exp in PHASE1:
            name = exp["name"]
            maps = parse_epoch_maps(name)
            max_ep = exp["params"].get("max_epochs", MAX_EPOCHS_DEFAULT)
            progress = f"{len(maps)}/{max_ep}" if maps else "0"
            running = "运行中" if is_process_running(name) else ""
            best = f", best={max(m['mAP'] for m in maps):.3f}" if maps else ""
            print(f"  {name:<22} epoch={progress:<8}{running}{best}")
        sys.exit(0)
    elif args.summary:
        summarize()
        sys.exit(0)
    else:
        scheduler_loop(once=args.once)
