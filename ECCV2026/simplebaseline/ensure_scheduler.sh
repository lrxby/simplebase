#!/bin/bash
# ============================================================
# 调度器守护脚本 ensure_scheduler.sh
# 由 crontab 每 5 分钟调用一次：
#   - 若 scheduler_dynamic.py 已在运行 → 什么都不做
#   - 若不在运行（服务器重启 / 调度器崩溃）→ 自动重新拉起
# 防止远程连接断开或服务器重启后实验调度中断。
# ============================================================

CODE_DIR="/mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline"
PYTHON="/home/liurunxiang/miniconda3/envs/cuda128/bin/python"
GUARD_LOG="$CODE_DIR/scheduler_guard.log"

# 检查调度器是否在运行。
# 注意: 必须锚定 python 解释器绝对路径开头, 否则任何命令行里恰好含
# "scheduler_dynamic.py" 字样的进程(如调试命令 bash -c "...")都会造成误判,
# 导致调度器挂掉时守护脚本误以为它还活着而跳过拉起。
if pgrep -f "^/home/liurunxiang/miniconda3/envs/cuda128/bin/python -u scheduler_dynamic\.py" > /dev/null 2>&1; then
    exit 0
fi

# 不在运行 → 拉起调度器（追加日志，保留历史）
cd "$CODE_DIR" || exit 1
nohup "$PYTHON" -u scheduler_dynamic.py >> scheduler.log 2>&1 &

echo "$(date '+%Y-%m-%d %H:%M:%S') [守护] 检测到调度器不在运行，已自动拉起 (PID $!)" >> "$GUARD_LOG"
