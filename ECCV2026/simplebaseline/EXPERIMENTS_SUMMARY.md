# ECCV2026 SimpleBaseline — 实验结果与问题总结（供 codex 决策）

> 生成时间：2026-09-11 15:20（服务器时间）
> 分支：`fix/pseudo-geometry-eval`，commit `369f11d`（已 push GitHub）
> 代码根：`/mnt/data/liurunxiang/workplace/simplebase/ECCV2026/simplebaseline/`
> 环境：`/home/liurunxiang/miniconda3/envs/cuda128/bin/python`
> 伪标签：`/mnt/data/liurunxiang/dataset/split_ss_dota/trainval/dota1-rect.pkl`（minAreaRect 法，论文同款，21046 图）

---

## 一、背景

论文《A Simple Baseline with Placement Prior for Point-Supervised Oriented Object Detection》的修复后消融与超参寻优。
此前 codex 审稿发现三大实现级 bug，全部已修复并通过验收（33/33 单测）：

1. **伪标签几何同步缺失**：伪框塞进 `bbox_fields` 但 mmdet Resize/Flip 并不处理该字段 → 约一半翻转样本的伪框方向与图像矛盾。
   修复：新增 `PseudoBoxSync` transform（h/v flip: θ→−θ 模 π；diagonal: θ 不变 + 中心 x/y 都翻；先 Resize 后 Flip；非等比例 Resize 明确报错），加载器加 `pseudo_valid` 掩码，缺失伪标签不再以全零框充当有效监督，增加实例对应关系追踪与统计打印。
2. **mAngle 表示歧义**：原指标未统一长边表示，(60,20,0°) 与 (20,60,90°) 被记为 90° 误差。
   修复：`mAngle_longedge`（长边规范化 + wrap 到 [−π/2, π/2) + 模 π 最小差），保留 `mAngle_raw`；新增轴对齐比例 min(|θ|, π/2−|θ|)<2.5°（同时含水平与垂直）、按长宽比分组统计、TP 召回率统计。
3. **OurWaterLoss 归约**：`gwd_sigma_loss` 被 `@weighted_loss` 默认取均值，逐实例加权不可用；改为显式 `reduction='none'` 后按有效样本归约，全无效时返回连计算图的零损失。

未实施（按用户/审稿锁定）："邻居共识+伪标签绝对锚"方案、SizeLoss/AngleLoss 数学目标修改、优化器/训练长度改动。

---

## 二、实验结果（DOTA trainval，dota/mAP @ epoch）

| 实验组 | 配置 | seed0 | seed1 | seed2 | 均值 | 备注 |
|---|---|---|---|---|---|---|
| 旧基线 | 旧代码，ourwater=1.0 | 0.4133 | — | — | — | 监督错误 |
| 旧 ow3 | 旧代码，ourwater=3.0 | 0.4308 | 0.4377 | 0.4429 | 0.4371 | 官网 test=0.378 |
| 修复后基线 | 修复代码，ourwater=1.0, seed0 | 0.4172 | — | — | — | 12ep 被外部中断 3 次后 resume |
| noangle | 修复代码，关 AngleLoss, seed0 | 0.4631 | — | — | — | 仅 OurWaterLoss 独立学方向 |
| **修复后 ow3** | 修复代码，ourwater=3.0 | **0.4834** | **0.4893** | **0.4870** | **0.4866** | **官网 test=0.4308（当前最好）** |
| **修复后 ow5** | 修复代码，ourwater=5.0 | **0.4873** | **0.4867** | **0.4898** | **0.4879** | 与 ow3 基本持平 |

角度指标（修复后 ow3_s1 / ow5 各 seed，ep12）：
- mAngle_longedge ≈ 7.6–8.0°；长条目标（w/h≥1.5）角度误差 2.5–2.6°；近方形排除后 5.4–5.6°。
- mAngle_raw ≈ 40–47°（仍高，是表示歧义残留，勿直接用 raw 判断）。

### DOTA 官网 test 集结果（VOC task1 mAP）

| 提交权重 | test mAP | 说明 |
|---|---|---|
| 旧基线 rect/2-0.413 | 0.326 | 监督错误时代 |
| 旧 ow3 (ourwater3_s2) | 0.378 | 监督错误时代 |
| **修复后 ow3_s1/ep12（0.4893）** | **0.4308** | **当前最好，用户已提交官网验证** |

修复后 0.4308 各类别：plane 0.740 / baseball-diamond 0.350 / **bridge 0.0002** / ground-track-field 0.201 / small-vehicle 0.702 / **large-vehicle 0.691** / ship 0.772 / tennis-court 0.907 / basketball-court 0.157 / storage-tank 0.756 / soccer-ball-field 0.146 / roundabout 0.238 / harbor 0.173 / swimming-pool 0.268 / helicopter 0.362。
（对比旧提交：large-vehicle 0.238→0.691、ship 0.557→0.772 是最大跃升；弱类仍低。）

---

## 三、关键发现

1. **修复 × 权重协同**：仅修监督（基线 0.4133→0.4172，+0.4 点）几乎无收益；仅提权重（旧代码 0.4133→0.4371，+2.4 点）；**修复+权重3 = 0.4866（+7.3 点）**。修复让 OurWaterLoss 的方向监督真正生效，权重放大才有意义。
2. **AngleLoss 修复后作用存疑**：noangle（关 AngleLoss）= 0.4631 > 修复后基线 0.4172（+4.6 点）。仅靠 OurWaterLoss 就能学好方向（mAngle_longedge 7.77°）。AngleLoss 的一致性监督（4θ 编码，允许整体同向解、不区分长轴方向）在修复后疑似有害或至少无益。**是否关闭/重设计 AngleLoss 是论文层面的重要决策，待定**。
3. **ow5 ≈ ow3**：权重 5 与权重 3 三 seed 均值仅差 +0.0013（0.4879 vs 0.4866），seed 间互有高低，属噪声范围。进一步加权的边际收益已到平台期，**不建议再试 ow7**（G3 候选已暂停）。
4. **角度学习已经到位**：长条目标角度误差 ~2.5°，可视化（100 张随机图）方向贴合目标，用户验收通过。

---

## 四、当前问题 / 待 codex 决策

1. **AngleLoss 去留**：是否在论文消融中纳入"关闭 AngleLoss"？修复后 noangle 反超基线的解释（AngleLoss 引入方向冲突 vs 仅为实验噪声）需要论证或补实验。
2. **弱类提升**：bridge 0.0002 / soccer 0.146 / basketball 0.157 / roundabout 0.238 明显偏低（可能伪标签质量、类别样本量、长宽比分布）。是否值得专项处理（类别权重、伪标签筛选）？注意用户已搁置"对齐官方 bridge"的做法（不调 w=10）。
3. **G3 候选**：原计划 ow7（权重7）或 lr2.5e-5 二选一。基于 ow5≈ow3 结论，**ow7 大概率无增益，建议不再跑**；lr 方案未测。
4. **是否用 ow5 权重再提交一次官网**：ow5_s2=0.4898 是全部实验 trainval 最高单 seed，可以生成提交包再验证一次 test（可选）。
5. **论文数值更新**：修复后 best 0.4898（trainval）/ 0.4308（test）显著高于论文当前报告值，需确认最终版是否更新实验表。
6. **实验体系规范**（codex 已指出，未改）：调度器按名字推断 config、固定 [6400/ 推算 epoch、状态文件未绑定代码版本、Phase2 参数覆盖顺序等——若继续大规模实验需先修。

---

## 五、复现信息

- 训练命令：
  ```
  CUDA_VISIBLE_DEVICES=N $PY tools/train.py configs/point2rbox_v2/audit-fixed-ow3.py --work-dir work_dirs/night/ow3_sX --cfg-options randomness.seed=X
  ```
  （ow5 同理换 audit-fixed-ow5.py；必须 `setsid env CUDA_VISIBLE_DEVICES=N $PY -u ... < /dev/null > log 2>&1 &` 防断连）
- 权重位置：`work_dirs/night/ow3_s{0,1,2}/`、`work_dirs/night/ow5_s{0,1,2}/`（每个含 12 epoch pth，best 在 ep12）
- 提交包生成：`configs/point2rbox_v2/submit-fixed-ow3s1-ep12.py` + `work_dirs/night/ow3_s1/epoch_12.pth` → `work_dirs/night/submit_ow3s1_ep12/Task1/Task1.zip`
- 旧权重诊断：`work_dirs/audit_fixed/old_eval2.log`（旧基线 0.4133 / mAngle_raw 47.70 / mAngle_longedge 18.94）
- 可视化：`work_dirs/audit-eval-old/20260911_131848/work_dirs/night/vis_ow3s1_ep12/`（100 张）；本地 `C:\Users\刘润祥\Desktop\ECCV 2026\audit\vis_samples\`（6 张）

## 六、当前系统状态

- 两卡 GPU 空闲，所有训练已完成，无异常。
- 调度器已停止（15:13:03，日志 `work_dirs/night/scheduler_v2.log` 有记录），`decided_g3.flag` 已生成——不会自动启动任何新实验，等待 codex 决策后人工启动。
- 定时监控任务「训练进度监控」仍在运行（每 30 分钟检查），当前无实验可查属正常。
