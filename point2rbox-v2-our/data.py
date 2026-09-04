import os
import random
import shutil
from tqdm import tqdm

def make_mini_dataset(src_root, dst_root, ratio=0.01):
    """
    从源数据集随机采样一定比例的数据复制到目标目录。
    
    Args:
        src_root (str): 源数据集根目录 (包含 images 和 labelTxt)
        dst_root (str): 目标数据集根目录
        ratio (float): 采样比例 (0.01 代表 1%)
    """
    
    # 1. 定义路径
    src_img_dir = os.path.join(src_root, 'images')
    src_label_dir = os.path.join(src_root, 'labelTxt')
    
    dst_img_dir = os.path.join(dst_root, 'images')
    dst_label_dir = os.path.join(dst_root, 'labelTxt')
    
    # 2. 检查源目录是否存在
    if not os.path.exists(src_img_dir) or not os.path.exists(src_label_dir):
        print(f"错误：源目录结构不正确，未找到 images 或 labelTxt 文件夹。\n请检查: {src_root}")
        return

    # 3. 创建目标目录 (如果不存在)
    os.makedirs(dst_img_dir, exist_ok=True)
    os.makedirs(dst_label_dir, exist_ok=True)
    print(f"目标目录已准备: {dst_root}")

    # 4. 获取所有图片文件列表
    # 支持常见的图片扩展名
    valid_exts = {'.png', '.jpg', '.jpeg', '.bmp', '.tif'}
    all_images = [f for f in os.listdir(src_img_dir) if os.path.splitext(f)[1].lower() in valid_exts]
    total_imgs = len(all_images)
    
    if total_imgs == 0:
        print("错误：源 images 文件夹为空。")
        return

    # 5. 计算采样数量
    sample_num = int(total_imgs * ratio)
    # 保证至少采一张，或者如果你只想测试极小数据，可以硬编码 sample_num = 50
    sample_num = max(sample_num, 10) 
    
    print(f"源数据集共 {total_imgs} 张图片。")
    print(f"准备抽取 {ratio*100}% -> {sample_num} 张图片及其标签...")

    # 6. 随机采样
    selected_images = random.sample(all_images, sample_num)

    # 7. 开始复制
    success_count = 0
    missing_label_count = 0
    
    print("开始复制文件...")
    for img_file in tqdm(selected_images):
        file_name_no_ext = os.path.splitext(img_file)[0]
        
        # 构建源文件路径
        src_img_path = os.path.join(src_img_dir, img_file)
        src_label_path = os.path.join(src_label_dir, file_name_no_ext + '.txt')
        
        # 构建目标文件路径
        dst_img_path = os.path.join(dst_img_dir, img_file)
        dst_label_path = os.path.join(dst_label_dir, file_name_no_ext + '.txt')
        
        # 复制图片
        shutil.copy2(src_img_path, dst_img_path)
        
        # 复制标签 (如果存在)
        if os.path.exists(src_label_path):
            shutil.copy2(src_label_path, dst_label_path)
            success_count += 1
        else:
            # 有些数据集可能存在只有图没有标签的情况（如背景图），视情况记录
            # print(f"警告：未找到标签文件 {src_label_path}")
            missing_label_count += 1

    print("\n" + "="*30)
    print(f"处理完成！")
    print(f"成功复制: {success_count} 对 (图片+标签)")
    if missing_label_count > 0:
        print(f"缺失标签: {missing_label_count} 张 (仅复制了图片)")
    print(f"新数据集路径: {dst_root}")
    print("="*30)
    print("\n[下一步操作建议]")
    print(f"请修改你的配置文件 (config.py)，将 data_root 指向: {dst_root}")
    print(f"或者将 ann_file 和 img_prefix 修改为上面的 images 和 labelTxt 路径。")

if __name__ == '__main__':
    # ================= 配置区域 =================
    # 源数据集路径 (老师的路径)
    SOURCE_PATH = '/mnt/data/xiekaikai/split_ss_dota/trainval'
    
    # 你的目标路径 (会自动创建 trainval_mini 子文件夹，保持整洁)
    # 建议不要直接散落在 dataset 根目录，而是建一个子文件夹比如 trainval_1pct
    TARGET_PATH = '/mnt/data/liurunxiang/workplace/point2rbox-v2-our/dataset/dota'
    
    # 采样比例 (0.01 = 1%)
    RATIO = 0.1
    # ===========================================
    
    make_mini_dataset(SOURCE_PATH, TARGET_PATH, RATIO)
