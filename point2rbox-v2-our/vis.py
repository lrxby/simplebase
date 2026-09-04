import os
import random
import glob
import torch
import mmcv
from mmengine.utils import ProgressBar
from mmrotate.apis import init_detector, inference_detector
from mmrotate.registry import VISUALIZERS
from mmrotate.utils import register_all_modules

def main():
    # ================= 配置区域 =================
    # 配置文件路径
    CONFIG_FILE = 'configs/point2rbox_v2/point2rbox_v2-1x-dota.py' 
    # 权重文件路径
    CHECKPOINT_FILE = 'work_dirs/roicls-v/1/e2e/epoch_1.pth'
    # 图片文件夹路径
    IMAGE_DIR = '/mnt/data/xiekaikai/split_ss_dota/trainval/images'
    # 结果保存路径
    OUTPUT_DIR = 'work_dirs/roicls-v/1/e2e/visual1'
    
    # [关键] 置信度阈值
    # 训练初期(epoch 3)建议设低一点(如0.2或0.3)，否则可能画不出框。
    # 如果画出来的框全是杂乱的背景，可以适当调高。
    SCORE_THRESHOLD = 0.3
    
    # 随机抽取的图片数量
    NUM_SAMPLES = 20
    
    # 设备
    DEVICE = 'cuda:3' if torch.cuda.is_available() else 'cpu'
    # ===========================================

    # 1. 注册所有模块 (非常重要！否则会报 KeyError: 'Point2RBoxV2' is not in model registry)
    register_all_modules(init_default_scope=True)

    # 2. 初始化模型
    print(f'正在加载模型: {CONFIG_FILE} ...')
    try:
        model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    except Exception as e:
        print(f"模型加载失败，请检查配置文件和权重路径。\n错误信息: {e}")
        return

    # 3. 准备可视化工具
    # 获取配置文件中的 visualizer 配置，如果没有则使用默认的
    visualizer = VISUALIZERS.build(model.cfg.visualizer)
    
    # 手动设置数据集元数据 (类别名称和颜色)
    # DOTA v1.0 类别
    visualizer.dataset_meta = {
        'classes': ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
                    'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                    'basketball-court', 'storage-tank', 'soccer-ball-field', 'roundabout',
                    'harbor', 'swimming-pool', 'helicopter'),
        'palette': [
            (165, 42, 42), (189, 183, 107), (0, 255, 0), (255, 0, 0),
            (138, 43, 226), (255, 128, 0), (255, 0, 255), (0, 255, 255),
            (255, 193, 37), (0, 0, 255), (30, 144, 255), (255, 255, 0),
            (139, 0, 139), (218, 112, 214), (144, 238, 144)
        ]
    }

    # 4. 获取图片列表并随机采样
    if not os.path.exists(IMAGE_DIR):
        print(f"错误: 图片目录不存在 -> {IMAGE_DIR}")
        return

    # 支持 png 和 jpg
    all_imgs = glob.glob(os.path.join(IMAGE_DIR, '*.png')) + \
               glob.glob(os.path.join(IMAGE_DIR, '*.jpg'))
    
    if len(all_imgs) == 0:
        print("未找到任何图片，请检查路径。")
        return

    # 随机采样
    sample_num = min(NUM_SAMPLES, len(all_imgs))
    selected_imgs = random.sample(all_imgs, sample_num)
    print(f'共找到 {len(all_imgs)} 张图片，随机抽取 {sample_num} 张进行推理。')

    # 创建输出目录
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. 循环推理并保存
    progress_bar = ProgressBar(len(selected_imgs))
    for img_path in selected_imgs:
        img_name = os.path.basename(img_path)
        out_path = os.path.join(OUTPUT_DIR, img_name)

        # 读取图片
        img = mmcv.imread(img_path)

        # 推理
        result = inference_detector(model, img)

        # 可视化
        visualizer.add_datasample(
            name=img_name,
            image=img,
            data_sample=result,
            draw_gt=False,      # 我们只画预测框
            draw_pred=True,     # 绘制预测结果
            wait_time=0,
            out_file=out_path,  # 保存路径
            pred_score_thr=SCORE_THRESHOLD # 置信度阈值
        )
        
        progress_bar.update()

    print(f'\n\n可视化完成！结果已保存至: {OUTPUT_DIR}')

if __name__ == '__main__':
    main()