#!/usr/bin/env python3
"""
动态自适应实验调度器 v5（每组 3 seed 统计验证版）
==============================================
升级要点（相对 v4）:
  1. 每组实验固定跑 3 个 seed (seed=0/1/2)，有效性按组内 3 次 best 的平均判定
  2. 兼容已有实验命名: s0 复用旧名（single-size5 等），运行中的实验自动归组不重复启动
  3. Phase2 组合实验同样跑 3 seed
  4. 新增 schedule-12811 组（12ep + 原版 lr 衰减 [8,11]）

基线：rect/2-0.413 best mAP = 0.413（DOTA-v1.0，12ep, lr5e-5, 无 lr schedule）
判定标准：组内 3 seed 平均 best mAP > 基线 + 0.003 视为有效

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

BASELINE_MAP = 0.413   # rect 伪标签基线 (rect/2-0.413, best@ep10)
IMPROVE_THRESHOLD = 0.003  # 组平均超过基线 0.3 mAP 视为有效
MEM_PER_EXP = 10000    # 每个实验预留显存 MB（实际 ~4GB，留足峰值缓冲）
MAX_EXP_PER_GPU = 2    # 单卡最大并行实验数（用户要求每卡最多 2 个）
MAX_EPOCHS_DEFAULT = 12
SEEDS = [0, 1, 2]      # 每组固定跑 3 个 seed

# 修改组定义（严格单变量：每组相对原始 config 只改一处）
GROUPS = [
    {
        "name": "ourwater3",
        "desc": "仅改 OurWaterLoss 权重 1.0→3.0",
        "params": {"ourwater_weight": 3.0},
        "s0_name": "single-ourwater3",
    },
    {
        "name": "size5",
        "desc": "仅改 Size 权重 1.0→5.0",
        "params": {"size_weight": 5.0},
        "s0_name": "single-size5",
    },
    {
        "name": "awarmup2",
        "desc": "仅改 Angle warmup 0→2 epoch",
        "params": {"angle_warmup_epochs": 2},
        "s0_name": "single-awarmup2",
        # seed1 正在手动运行中（rep-awarmup2-s1），调度器识别后不重复启动
        "seed1_override": "rep-awarmup2-s1",
        "seed_prefix": "rep-awarmup2",
    },
    {
        "name": "lr1e5",
        "desc": "仅改 lr 5e-5→1e-5",
        "params": {"lr": 0.00001},
        "s0_name": "single-lr1e-5",
    },
    {
        "name": "24ep",
        "desc": "仅改 epoch 12→24（含对应 lr schedule [16,22]）",
        "params": {"max_epochs": 24, "milestones": [16, 22]},
        "s0_name": "single-24ep",
    },
    {
        "name": "sched12811",
        "desc": "仅加 lr schedule（12ep + [8,11] 衰减，对齐原版）",
        "params": {"max_epochs": 12, "milestones": [8, 11]},
        "s0_name": "schedule-12811",
    },
]

# 展开为 PHASE1 实验列表（5 组 × 3 seed = 15 个）
PHASE1 = []
for g in GROUPS:
    for seed in SEEDS:
        if seed == 0:
            name = g["s0_name"]
        elif seed == 1 and "seed1_override" in g:
            name = g["seed1_override"]
        else:
            prefix = g.get("seed_prefix", g["s0_name"])
            name = f"{prefix}-s{seed}"
        exp = {
            "name": name,
            "group": g["name"],
            "seed": seed,
            "desc": g["desc"],
            "params": {**g["params"], "seed": seed},
        }
        PHASE1.append(exp)

PHASE2_GROUP_NAME = "combined"


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
    return {"phase1_results": {}, "phase2": None, "phase2_results": {}, "started_at": now()}


def save_state(state):
    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2, ensure_ascii=False)


def get_ours_per_gpu():
    """统计每个 GPU 上正在运行的【我们的】训练实验数 {gpu_id: count}

    匹配 tools/train.py configs/point2rbox_v2/ 下的实验 config
    （兼容 auto- 前缀与手动启动名，如 rep-awarmup2-s1.py / schedule-12811.py）
    """
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
        m = re.search(r'configs/point2rbox_v2/([\w.-]+\.py)', cmd)
        if not m:
            continue
        config_name = m.group(1)
        if not (config_name.startswith("auto-") or config_name in (
                "rep-awarmup2-s1.py", "schedule-12811.py")):
            continue
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
    """返回每个 GPU 还能启动几个实验: {gpu_id: slots}（显存 + 数量双重保险）"""
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
    """检查实验进程是否在跑（grep -E 兼容 auto- 前缀与手动启动名）"""
    out, _, _ = run_cmd(f"ps aux | grep -E '(auto-)?{re.escape(exp_name)}\\.py' | grep -v grep")
    return bool(out.strip())


def generate_config(exp, out_path):
    """基于【原始基线 config】生成单变量实验 config（严格单变量 + seed 替换）"""
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
        old = "topk=0.95,\n            target_classes=[0,4,5,6,7,8,10,12,14]"
        new = (f"topk=0.95,\n            warmup_epochs={params['angle_warmup_epochs']},\n"
               f"            target_classes=[0,4,5,6,7,8,10,12,14]")
        assert old in content, "angle warmup anchor not found!"
        content = content.replace(old, new)
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

    # --- 修改 4: max_epochs + lr schedule ---
    if "max_epochs" in params:
        old = "train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=12, val_interval=1)"
        new = f"train_cfg = dict(type='EpochBasedTrainLoop', max_epochs={params['max_epochs']}, val_interval=1)"
        assert old in content, "max_epochs anchor not found!"
        content = content.replace(old, new)
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
        changed.append(f"max_epochs={params['max_epochs']},ms={ms}")

    # --- 修改 6: OurWaterLoss 权重 ---
    if "ourwater_weight" in params:
        old = "type='OurWaterLoss',\n            loss_weight=1.0"
        new = f"type='OurWaterLoss',\n            loss_weight={params['ourwater_weight']}"
        assert old in content, "ourwater_weight anchor not found!"
        content = content.replace(old, new)
        changed.append(f"ourwater_weight={params['ourwater_weight']}")

    # --- 修改 5: seed ---
    if "seed" in params:
        old = "randomness = dict(seed=0, deterministic=False)"
        new = f"randomness = dict(seed={params['seed']}, deterministic=False)"
        assert old in content, "seed anchor not found!"
        content = content.replace(old, new)
        changed.append(f"seed={params['seed']}")

    with open(out_path, "w") as f:
        f.write(content)

    return "; ".join(changed)


def start_training(exp, gpu_id):
    """启动单个实验训练"""
    name = exp["name"]
    config_path = f"configs/point2rbox_v2/auto-{name}.py"
    work_dir = f"{WORK_DIR_BASE}/{name}"
    log_file = os.path.join(CODE_DIR, f"{work_dir}-nohup.log")

    out_path = os.path.join(CODE_DIR, config_path)
    changed = generate_config(exp, out_path)

    os.makedirs(os.path.join(CODE_DIR, work_dir), exist_ok=True)

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_id} nohup {PYTHON} tools/train.py "
        f"{config_path} --work-dir {work_dir} > {log_file} 2>&1 &"
    )
    _, err, code = run_cmd(cmd)
    if code != 0:
        return False, f"启动失败: {err}"

    print(f"[{now()}] 启动实验 '{name}' @ GPU{gpu_id} [组={exp['group']} seed={exp['seed']}]")
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
    """分析单个实验的结果"""
    maps = parse_epoch_maps(exp["name"])
    if not maps:
        return None
    best = max(maps, key=lambda x: x["mAP"])
    final = maps[-1]
    gain = best["mAP"] - BASELINE_MAP
    return {
        "best_mAP": best["mAP"],
        "best_epoch": best["epoch"],
        "final_mAP": final["mAP"],
        "gain_vs_baseline": gain,
    }


def group_results(state):
    """按 group 聚合 3 个 seed 的结果（平均/标准差/有效性）"""
    out = {}
    for g in GROUPS:
        gname = g["name"]
        seeds = []
        seed_names = []
        for exp in PHASE1:
            if exp["group"] == gname and exp["name"] in state["phase1_results"]:
                seeds.append(state["phase1_results"][exp["name"]])
                seed_names.append(exp["name"])
        if len(seeds) == 3:
            bests = [s["best_mAP"] for s in seeds]
            avg = sum(bests) / 3.0
            std = (sum((b - avg) ** 2 for b in bests) / 3.0) ** 0.5
            out[gname] = {
                "avg_best": avg,
                "std": std,
                "bests": bests,
                "seed_names": seed_names,
                "gain": avg - BASELINE_MAP,
                "effective": avg - BASELINE_MAP > IMPROVE_THRESHOLD,
            }
    return out


def build_phase2(state):
    """Phase 2: 组合所有【组平均有效】的修改"""
    gr = group_results(state)
    effective_params = {}
    effective_names = []

    for g in GROUPS:
        gname = g["name"]
        r = gr.get(gname)
        if r and r["effective"]:
            # 组合原始修改参数（不含 seed）
            effective_params.update(g["params"])
            effective_names.append(gname)

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


def phase2_experiments(phase2):
    """展开 Phase2 为 3 个 seed 实验"""
    if not phase2 or not phase2["params"]:
        return []
    exps = []
    for seed in SEEDS:
        name = "combined-best" if seed == 0 else f"combined-best-s{seed}"
        exps.append({
            "name": name,
            "group": PHASE2_GROUP_NAME,
            "seed": seed,
            "desc": phase2["desc"],
            "params": {**phase2["params"], "seed": seed},
        })
    return exps


# ============================================================
# 主调度循环
# ============================================================

def scheduler_loop(once=False):
    print(f"[{now()}] 动态调度器 v5 启动")
    print(f"  基线 mAP: {BASELINE_MAP}, 有效阈值: +{IMPROVE_THRESHOLD}")
    print(f"  修改组: {len(GROUPS)} 组 × {len(SEEDS)} seed = {len(PHASE1)} 个 Phase1 实验")

    state = load_state()

    while True:
        state = load_state()
        gpu_slots = get_gpu_slots()

        # ---------- 1. 检查 Phase1 各实验完成情况 ----------
        for exp in PHASE1:
            name = exp["name"]
            if name in state["phase1_results"]:
                continue  # 已分析
            if is_process_running(name):
                continue  # 正在跑
            if exp_is_finished(exp):
                result = analyze_result(exp)
                if result:
                    print(f"[{now()}] 实验 '{name}' 完成 [组={exp['group']} seed={exp['seed']}]: "
                          f"best={result['best_mAP']:.4f}@ep{result['best_epoch']}, "
                          f"final={result['final_mAP']:.4f}, gain={result['gain_vs_baseline']:+.4f}")
                    state["phase1_results"][name] = result
                    save_state(state)
                else:
                    print(f"[{now()}] 警告: 实验 '{name}' 进程结束但无有效结果")

        # ---------- 2. 组聚合判定 + Phase1 全完成检查 ----------
        all_phase1_done = all(
            exp["name"] in state["phase1_results"] for exp in PHASE1
        )
        if all_phase1_done and state["phase2"] is None:
            gr = group_results(state)
            print(f"[{now()}] Phase1 全部完成，组聚合判定:")
            for g in GROUPS:
                r = gr.get(g["name"])
                if r:
                    verdict = "有效 ✓" if r["effective"] else "无效 ✗"
                    print(f"  组 {g['name']:<12} 3seed best={[f'{b:.3f}' for b in r['bests']]} "
                          f"avg={r['avg_best']:.3f}±{r['std']:.3f} gain={r['gain']:+.3f} → {verdict}")
            state["phase2"] = build_phase2(state)
            p2 = state["phase2"]
            if p2["params"]:
                print(f"[{now()}] 生成 Phase2: {p2['desc']}")
            else:
                print(f"[{now()}] 无有效修改可组合，实验终止。")
            save_state(state)

        # ---------- 3. 检查 Phase2 完成情况 ----------
        phase2 = state.get("phase2")
        phase2_done = True
        if phase2 and phase2["params"]:
            if "phase2_results" not in state:
                state["phase2_results"] = {}
            p2_exps = phase2_experiments(phase2)
            # 逐个检查完成
            for exp in p2_exps:
                name = exp["name"]
                if name in state["phase2_results"]:
                    continue
                if is_process_running(name):
                    phase2_done = False
                    continue
                if exp_is_finished(exp):
                    result = analyze_result(exp)
                    if result:
                        print(f"[{now()}] Phase2 实验 '{name}' 完成: "
                              f"best={result['best_mAP']:.4f}@ep{result['best_epoch']}")
                        state["phase2_results"][name] = result
                        save_state(state)
                    else:
                        phase2_done = False
                else:
                    phase2_done = False
            # 3 seed 全齐 → 组判定
            if phase2_done and all(n in state["phase2_results"] for n in [e["name"] for e in p2_exps]):
                bests = [state["phase2_results"][e["name"]]["best_mAP"] for e in p2_exps]
                avg = sum(bests) / len(bests)
                effective = avg - BASELINE_MAP > IMPROVE_THRESHOLD
                verdict = "有效 ✓" if effective else "无效 ✗"
                print(f"[{now()}] Phase2 组合完成: 3seed best={[f'{b:.3f}' for b in bests]} "
                      f"avg={avg:.3f} gain={avg - BASELINE_MAP:+.3f} → {verdict}")

        # ---------- 4. 全部完成 → 退出 ----------
        if all_phase1_done and phase2_done:
            print(f"[{now()}] 🎉 所有实验完成！")
            summarize(state)
            if once:
                return
            break

        # ---------- 5. 分配 GPU 槽位 ----------
        if any(s > 0 for s in gpu_slots.values()):
            pending = []
            for exp in PHASE1:
                name = exp["name"]
                if name in state["phase1_results"]:
                    continue
                if is_process_running(name):
                    continue
                pending.append(exp)

            phase2 = state.get("phase2")
            if not pending and phase2 and phase2["params"]:
                for exp in phase2_experiments(phase2):
                    if exp["name"] in state.get("phase2_results", {}):
                        continue
                    if is_process_running(exp["name"]):
                        continue
                    pending.append(exp)

            for exp in pending:
                for gpu_id in sorted(gpu_slots.keys()):
                    if gpu_slots[gpu_id] > 0:
                        ok, msg = start_training(exp, gpu_id)
                        if ok:
                            gpu_slots[gpu_id] -= 1
                            time.sleep(2)
                        break

        # 打印状态
        running = [e["name"] for e in PHASE1 if is_process_running(e["name"])]
        p2 = state.get("phase2")
        if p2 and p2["params"]:
            for exp in phase2_experiments(p2):
                if is_process_running(exp["name"]):
                    running.append(exp["name"])
        done = list(state["phase1_results"].keys())
        slots_now = get_gpu_slots()
        slot_desc = ", ".join(f"GPU{k}:{v}槽" for k, v in sorted(slots_now.items()))
        print(f"[{now()}] 运行中: {running or '无'}, 已完成: {len(done)}/{len(PHASE1)}, GPU槽位: {slot_desc}")

        if once:
            return

        time.sleep(180)  # 每 3 分钟检查一次


# ============================================================
# 汇总
# ============================================================

def summarize(state=None):
    if state is None:
        state = load_state()

    print("\n" + "=" * 110)
    print("实验汇总（基线 best mAP = %.3f，每组 3 seed 平均判定）" % BASELINE_MAP)
    print("=" * 110)

    gr = group_results(state)
    for g in GROUPS:
        gname = g["name"]
        print(f"\n▍组 {gname} — {g['desc']}")
        for exp in PHASE1:
            if exp["group"] != gname:
                continue
            result = state["phase1_results"].get(exp["name"])
            if result:
                print(f"    {exp['name']:<24} seed={exp['seed']}  best={result['best_mAP']:.3f}@ep{result['best_epoch']}  final={result['final_mAP']:.3f}")
            else:
                maps = parse_epoch_maps(exp["name"])
                max_ep = exp["params"].get("max_epochs", MAX_EPOCHS_DEFAULT)
                prog = f"{len(maps)}/{max_ep}" if maps else "0"
                rn = "运行中" if is_process_running(exp["name"]) else ""
                print(f"    {exp['name']:<24} seed={exp['seed']}  epoch={prog:<8}{rn}")
        r = gr.get(gname)
        if r:
            verdict = "有效 ✓" if r["effective"] else "无效 ✗"
            print(f"    → 3seed avg={r['avg_best']:.3f}±{r['std']:.3f}  gain={r['gain']:+.3f}  {verdict}")

    p2 = state.get("phase2")
    if p2 and p2["params"]:
        print(f"\n▍Phase2 组合 — {p2['desc']}")
        p2r = state.get("phase2_results", {})
        for seed in SEEDS:
            name = "combined-best" if seed == 0 else f"combined-best-s{seed}"
            result = p2r.get(name)
            if result:
                print(f"    {name:<24} seed={seed}  best={result['best_mAP']:.3f}@ep{result['best_epoch']}")
            else:
                maps = parse_epoch_maps(name)
                prog = f"{len(maps)}/{MAX_EPOCHS_DEFAULT}" if maps else "0"
                rn = "运行中" if is_process_running(name) else ""
                print(f"    {name:<24} seed={seed}  epoch={prog:<8}{rn}")
    print("=" * 110)


# ============================================================
# 入口
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="动态自适应实验调度器 v5")
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
        sys.exit(0)
    elif args.summary:
        summarize()
        sys.exit(0)
    else:
        scheduler_loop(once=args.once)
