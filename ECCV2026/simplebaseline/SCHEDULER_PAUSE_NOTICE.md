# 调度器暂停记录 (2026-09-10 ~17:20)
## 已暂停
- crontab: ensure_scheduler.sh 条目已移除, 备份 /tmp/crontab_backup_20260910.txt
- scheduler_dynamic.py: PID 3052268 (SIGTERM)
- 守护循环 bash: PID 3052266 (SIGTERM)
## 未受影响
- 训练进程: single-24ep (2670508), ourwater3 s0 (3221203), s1, s2 全部继续
## 恢复方式
1. crontab /tmp/crontab_backup_20260910.txt
2. cd /mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline && nohup /home/liurunxiang/miniconda3/envs/cuda128/bin/python -u scheduler_dynamic.py > scheduler.log 2>&1 &
