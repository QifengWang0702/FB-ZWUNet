import cv2
import os
import shutil

def process_and_save_images(directory, new_directory, filenames):
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):  # 可根据需要添加其他图像格式
            filepath = os.path.join(directory, filename)
            image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)  # 读取为灰度图像
            resized_image = cv2.resize(image, (128, 128))  # 调整大小为128x128
            
            new_filepath = os.path.join(new_directory, filename)
            cv2.imwrite(new_filepath, resized_image)  # 保存图像

def main():
    base_directory = "/root/data1/wqf/FBUSAS/DataSet/CV/"
    save_directory = "/root/data1/wqf/FBUSAS/DataSet/"
    gt_directory = os.path.join(base_directory, "GT")
    img_directory = os.path.join(base_directory, "Img")

    if not os.path.exists(gt_directory) or not os.path.exists(img_directory):
        print("GT或Img目录不存在，请检查路径。")
        return

    # 创建新的目录结构
    new_base_directory = os.path.join(save_directory, "CV-New")
    os.makedirs(new_base_directory, exist_ok=True)

    dataset_splits = ["train", "val", "test"]
    for split in dataset_splits:
        os.makedirs(os.path.join(new_base_directory, split, "GT"), exist_ok=True)
        os.makedirs(os.path.join(new_base_directory, split, "Img"), exist_ok=True)

    # 假设您希望将数据分为80%的训练集，10%的验证集和10%的测试集
    gt_files = sorted(os.listdir(gt_directory))
    img_files = sorted(os.listdir(img_directory))

    total_images = len(gt_files)
    train_count = int(0.8 * total_images)
    val_count = int(0.1 * total_images)

    train_files = gt_files[:train_count]
    val_files = gt_files[train_count:train_count+val_count]
    test_files = gt_files[train_count+val_count:]

    process_and_save_images(gt_directory, os.path.join(new_base_directory, "train", "GT"), train_files)
    process_and_save_images(gt_directory, os.path.join(new_base_directory, "val", "GT"), val_files)
    process_and_save_images(gt_directory, os.path.join(new_base_directory, "test", "GT"), test_files)

    process_and_save_images(img_directory, os.path.join(new_base_directory, "train", "Img"), train_files)
    process_and_save_images(img_directory, os.path.join(new_base_directory, "val", "Img"), val_files)
    process_and_save_images(img_directory, os.path.join(new_base_directory, "test", "Img"), test_files)

    print("图像处理完成。")

if __name__ == "__main__":
    main()
