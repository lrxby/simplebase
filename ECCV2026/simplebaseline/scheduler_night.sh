#!/bin/bash
# ============================================================
# scheduler_night.sh - 夜间全自动实验调度器
# 规则: 每 GPU 最多 2 实验；单变量；3-seed 验证；动态决策
# G1: ow3 (ourwater=3.0 复验) s0/s1/s2
# G2: 若 ow3 有效(mean>0.420) -> ow5；否则 -> awarmup2
# G3: 根据 G2 结果动态选 ow7 或 lr2.5e-5
# 中断检测: 进程死且未完成 -> 自动 --resume 恢复
# ============================================================
cd /mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline
QUEUE_DIR=work_dirs/night
mkdir -p $QUEUE_DIR
LOG=$QUEUE_DIR/scheduler_night.log
PY=/home/liurunxiang/miniconda3/envs/cuda128/bin/python
MAX_PER_GPU=2
QUEUE=$QUEUE_DIR/queue.txt
RUNNING=$QUEUE_DIR/running.txt
CFG_DIR=configs/point2rbox_v2
BASELINE_MAP=0.4172
FLAG_G2=$QUEUE_DIR/decided_g2.flag
FLAG_G3=$QUEUE_DIR/decided_g3.flag

log(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" >> $LOG; }

gpu_of_pid(){
  tr '\0' '\n' < /proc/$1/environ 2>/dev/null | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -1
}

count_gpu(){
  local g=$1
  # 按实验名去重计数（RUNNING 每实验只应有一行主进程）
  awk -F'|' -v g="$g" '$2==g && $3!="" {print $3}' $RUNNING 2>/dev/null | sort -u | grep -c . 
}

best_map(){
  grep 'Epoch(val)' "$1/console.log" 2>/dev/null | grep '1316/1316' | grep -o 'dota/mAP: [0-9.]*' | awk '{print $2}' | sort -rn | head -1
}

is_done(){
  local c=$(grep -c 'Epoch(val) \[12\]\[1316/1316\]' "$1/console.log" 2>/dev/null)
  [ "$c" -ge 1 ] && echo 1 || echo 0
}

running_names(){ awk -F'|' '{print $3}' $RUNNING 2>/dev/null; }
queue_names(){ awk -F'|' '{print $1}' $QUEUE 2>/dev/null; }

group_all_done(){ # $1 = 前缀 (ow3/ow5/aw2)
  local any=0
  for n in $(running_names) $(queue_names); do
    case $n in $1-*) any=1;; esac
  done
  [ $any -eq 0 ] && echo 1 || echo 0
}

mean_best_of(){ # $1 = 前缀
  local sum=0 n=0 b
  for d in work_dirs/night/${1}_s*; do
    [ -d "$d" ] || continue
    b=$(best_map "$d")
    [ -n "$b" ] && { sum=$(echo "$sum+$b" | bc); n=$((n+1)); }
  done
  if [ $n -gt 0 ]; then echo "scale=4; $sum/$n" | bc; else echo ""; fi
}

launch(){
  local line=$1
  local name cfg seed wd gpu
  name=$(echo "$line" | cut -d'|' -f1)
  cfg=$(echo "$line" | cut -d'|' -f2)
  seed=$(echo "$line" | cut -d'|' -f3)
  wd=$(echo "$line" | cut -d'|' -f4)
  gpu=""
  local g c
  for g in 0 1; do c=$(count_gpu $g); if [ $c -lt $MAX_PER_GPU ]; then gpu=$g; break; fi; done
  [ -z "$gpu" ] && return 1
  mkdir -p "$wd"
  local extra=""
  [ -f "$wd/last_checkpoint" ] && extra="--resume"
  CUDA_VISIBLE_DEVICES=$gpu setsid $PY -u tools/train.py "$cfg" --work-dir "$wd" --cfg-options randomness.seed=$seed $extra </dev/null >"$wd/console.log" 2>&1 &
  local pid=$!
  echo "$pid|$gpu|$name" >> $RUNNING
  sed -i "/^$name|/d" $QUEUE
  log "LAUNCH $name seed=$seed on GPU$gpu pid=$pid (resume=$extra)"
  return 0
}

clean_running(){
  local tmp=$QUEUE_DIR/running.tmp
  : > $tmp
  local pid gpu name
  while IFS='|' read -r pid gpu name; do
    [ -z "$pid" ] && continue
    if kill -0 "$pid" 2>/dev/null; then
      echo "$pid|$gpu|$name" >> $tmp
    else
      local wd=work_dirs/night/${name}
      if [ "$(is_done $wd)" = "1" ]; then
        log "FINISHED $name (best=$(best_map $wd))"
      else
        log "WARN $name process dead (pid=$pid) unfinished -> requeue for resume"
        echo "$name|$CFG_DIR/${name%%-s*}.py|$(echo $name | sed 's/.*-s//')|$wd" >> $QUEUE
      fi
    fi
  done < $RUNNING
  mv $tmp $RUNNING
}

decide_g2(){
  local m=$(mean_best_of ow3)
  log "G1 decision: ow3 3-seed mean best = $m (baseline=$BASELINE_MAP)"
  if [ -n "$m" ] && [ "$(echo "$m > 0.4200" | bc)" = "1" ]; then
    for s in 0 1 2; do echo "ow5-s$s|$CFG_DIR/audit-fixed-ow5.py|$s|work_dirs/night/ow5_s$s" >> $QUEUE; done
    log "G2 = ow5 (ourwater 5.0): ow3 effective"
  else
    for s in 0 1 2; do echo "aw2-s$s|$CFG_DIR/audit-fixed-awarmup2.py|$s|work_dirs/night/aw2_s$s" >> $QUEUE; done
    log "G2 = awarmup2: ow3 not effective"
  fi
  touch $FLAG_G2
}

decide_g3(){
  local g2name="" g2m="" g1m=""
  if [ -d work_dirs/night/ow5_s0 ]; then g2name=ow5; else g2name=aw2; fi
  g2m=$(mean_best_of $g2name)
  g1m=$(mean_best_of ow3)
  log "G2 decision: $g2name mean = $g2m (ow3 mean=$g1m)"
  if [ "$g2name" = "ow5" ] && [ -n "$g2m" ] && [ -n "$g1m" ] && [ "$(echo "$g2m > $g1m" | bc)" = "1" ]; then
    for s in 0 1 2; do echo "ow7-s$s|$CFG_DIR/audit-fixed-ow7.py|$s|work_dirs/night/ow7_s$s" >> $QUEUE; done
    log "G3 = ow7: ow5 improved over ow3"
  else
    for s in 0 1 2; do echo "lr25-s$s|$CFG_DIR/audit-fixed-lr25.py|$s|work_dirs/night/lr25_s$s" >> $QUEUE; done
    log "G3 = lr2.5e-5: ow5 not improving (or G2 was aw2)"
  fi
  touch $FLAG_G3
}

# ============ 初始化: 登记已有训练主进程 (noangle 等, 按实验名去重) ============
: > $RUNNING
declare -A SEEN=()
for p in $(pgrep -f 'tools/train.py'); do
  [ "$p" = "$$" ] && continue
  # 只登记 setsid 主进程 (PPID=1)；worker 子进程 (PPID=主进程) 跳过
  ppid=$(awk '{print $4}' /proc/$p/stat 2>/dev/null)
  [ "$ppid" != "1" ] && continue
  wd=$(tr '\0' '\n' < /proc/$p/cmdline 2>/dev/null | grep -A1 -- '--work-dir' | tail -1)
  [ -z "$wd" ] && continue
  name=$(basename "$wd")
  [ -n "${SEEN[$name]}" ] && continue
  SEEN[$name]=1
  gpu=$(gpu_of_pid $p)
  echo "$p|$gpu|$name" >> $RUNNING
  log "init: registered $name pid=$p gpu=$gpu"
done

# ============ 初始化队列 G1 ============
if [ ! -f $QUEUE ]; then
  : > $QUEUE
  for s in 0 1 2; do
    echo "ow3-s$s|$CFG_DIR/audit-fixed-ow3.py|$s|work_dirs/night/ow3_s$s" >> $QUEUE
  done
  log "G1 queue initialized: ow3 s0/s1/s2"
fi

# ============ 主循环 ============
log "scheduler_night started. Max per GPU=$MAX_PER_GPU"
while true; do
  clean_running
  # 启动所有可启动的 PENDING
  while read -r line; do
    [ -z "$line" ] && continue
    launch "$line" || break
  done < $QUEUE
  # 决策 G2 / G3
  if [ ! -f $FLAG_G2 ] && [ "$(group_all_done ow3)" = "1" ] && [ "$(group_all_done ow5)" = "1" ]; then
    decide_g2
  fi
  if [ ! -f $FLAG_G3 ] && [ "$(group_all_done ow3)" = "1" ] && [ "$(group_all_done ow5)" = "1" ] && [ "$(group_all_done aw2)" = "1" ] && [ -f $FLAG_G2 ]; then
    decide_g3
  fi
  # 状态快照
  log "STATUS: GPU0=$(count_gpu 0) GPU1=$(count_gpu 1) running=$(cat $RUNNING | wc -l) pending=$(wc -l < $QUEUE)"
  sleep 600
done
