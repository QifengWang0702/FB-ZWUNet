import os
import numpy as np
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from data import medicalDataLoader
from models.FBUSAS_net import PAM_CAM_stack, FBUSAS_stack, FBUSAS_Model, ZAM_WAM_stack ,ZAM_WAM_Simplified_stack, ZAM_WAM_Unet, Shape_Unet, CC_CV_Unet, CC_SP_Unet
from common.progressBar import printProgressBar
import torchvision
from common.utils import inferenceAllMetrics
import matplotlib.pyplot as plt
import cv2
from common.utils import printProgressBar
                          

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return x

def getSingleImage(pred_y):
    return pred_y.argmax(dim=1, keepdim=True)

#Save Segmentation Result
def saveImages_for2D(net, img_batch, epoch, modelName, savePath):
    # 设置保存路径
    path = os.path.join(savePath) #+ str(epoch)
    if not os.path.exists(path):
        os.makedirs(path)

    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax(dim=1)  # 注意dim=1，这是因为我们希望在类别维度上进行softmax

    for i, data in enumerate(img_batch):
        image, labels, img_names = data

        US = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(US)
        pred_y = softMax(segmentation_prediction)
        
        segmentation = getSingleImage(pred_y)  # 假设这个函数从softmax输出中获取2D分割图像

        # 提取图像名，用于保存
        img_name = img_names[0].split('/')[-1]  # 获取文件名
        segmentation_float = segmentation.float() / (torch.max(segmentation).item() + 1e-5)  # 将其转换为浮点型并归一化
        torchvision.utils.save_image(segmentation_float, os.path.join(path, img_name))
        #torchvision.utils.save_image(segmentation.data, os.path.join(path, img_name))

    print("Test dataset segmentation results saved!")

#Calculate TestModel Metrics
def evaluate_model(net, val_loader, TestModel, args):
    modelName = args.modelName
    directory = args.image_save_dir
    Losses = []  # 假设您已经在某处计算了Losses

    metrics_results = inferenceAllMetrics(net, val_loader)  # 使用修改后的inference_函数

    # 保存损失和评估指标
    if not os.path.exists(directory):
        os.makedirs(directory)

    np.save(os.path.join(directory, 'Losses.npy'), Losses)
    for metric, values in metrics_results.items():
        np.save(os.path.join(directory, f'{metric}Val.npy'), values.cpu().numpy())

    # 打印和保存评估指标
    metrics_str_list = []
    with open(os.path.join(args.statistics_save_dir, ' test_statistics.txt'), 'a') as f:
        for metric, values in metrics_results.items():
            current_value = values.mean().item()
            metrics_str_list.append(f"{metric}: {current_value:.4f}")
        metrics_str = ", ".join(metrics_str_list)
        print(f"[Test][{TestModel}] {metrics_str}")
        f.write(f"[{TestModel}] {metrics_str}\n")

#Metrics Image from .npy/.txt
def plot_metrics_from_files(directory, save_directory, metrics=["Dice", "Jaccard", "Precision", "Recall", "Specificity", "Accuracy", "Hausdorff", "MAD", "SurfaceDistance"]):
    
    """
    Reads the metrics from .npy or .txt files and plots them.

    Parameters:
    - directory: The directory where the .npy or .txt files are saved.
    - save_directory: The directory where the plots will be saved.
    - metrics: List of metrics to plot. Default includes common metrics excluding "Hausdorff".
    """

    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']
    light_colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightcyan', 'lightpink', 'lightyellow', 'lightgray', 'lightsalmon', 'plum']

    for file_name in ['training_statistics.txt', 'valuation_statistics.txt']:
        # Plot each metric in a separate figure
        for metric in metrics:
            metric_values = []
            txt_file_path = os.path.join(directory, file_name)
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # Extract metric value from the line
                        start_index = line.find(metric + ": ") + len(metric) + 2
                        end_index = line.find(",", start_index)
                        value = float(line[start_index:end_index])
                        metric_values.append(value)

                # Plot the metric values and the fitted curve
                plt.figure(figsize=(10, 6))
                color = colors[metrics.index(metric)]
                light_color = light_colors[metrics.index(metric)]
                plt.plot(metric_values, label=f"{metric} Values", color=light_color)
                x = np.linspace(0, len(metric_values)-1, len(metric_values))
                z = np.polyfit(x, metric_values, 3)
                p = np.poly1d(z)
                plt.plot(x, p(x), label=f"{metric} Fit (Max: {max(metric_values):.2f}, Epoch: {metric_values.index(max(metric_values))})", linestyle='--', color=color)

                plt.xlabel('Epochs')
                plt.ylabel('Metric Value')
                plt.title(f'{metric} over Epochs ({file_name.split("_")[0]})')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(save_directory, f'{metric}_{file_name.split("_")[0]}_plot.png'))
                plt.clf()
        
        # Plot all metrics in one figure (only the fitted curves)
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        for metric in metrics:
            metric_values = []
            txt_file_path = os.path.join(directory, file_name)
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # Extract metric value from the line
                        start_index = line.find(metric + ": ") + len(metric) + 2
                        end_index = line.find(",", start_index)
                        value = float(line[start_index:end_index])
                        metric_values.append(value)

                # Plot the fitted curve for each metric
                x = np.linspace(0, len(metric_values)-1, len(metric_values))
                z = np.polyfit(x, metric_values, 3)
                p = np.poly1d(z)
                color = colors[metrics.index(metric)]
                
                if metric in ["Dice", "Jaccard", "Precision", "Recall", "Specificity", "Accuracy"]:
                    ax1.plot(x, p(x), label=f"{metric} Fit (Max: {max(metric_values):.2f}, Epoch: {metric_values.index(max(metric_values))})", color=color)
                else:
                    ax2.plot(x, p(x), label=f"{metric} Fit (Max: {max(metric_values):.2f}, Epoch: {metric_values.index(max(metric_values))})", color=color, linestyle='--')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metric Value (0-1)')
        ax2.set_ylabel('Metric Value (Large)')
        ax1.set_title(f'Metrics over Epochs (Fitted Curves) ({file_name.split("_")[0]})')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'All_Metrics_Fit_{file_name.split("_")[0]}_plot.png'))
        plt.clf()

        # Plot all metrics in one figure (only the values)
        plt.figure(figsize=(10, 6))
        ax1 = plt.gca()
        ax2 = ax1.twinx()

        for metric in metrics:
            metric_values = []
            txt_file_path = os.path.join(directory, file_name)
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # Extract metric value from the line
                        start_index = line.find(metric + ": ") + len(metric) + 2
                        end_index = line.find(",", start_index)
                        value = float(line[start_index:end_index])
                        metric_values.append(value)

                # Plot the metric values
                light_color = light_colors[metrics.index(metric)]
                if metric in ["Dice", "Jaccard", "Precision", "Recall", "Specificity", "Accuracy"]:
                    ax1.plot(metric_values, label=f"{metric} Values (Max: {max(metric_values):.2f}, Epoch: {metric_values.index(max(metric_values))})", color=light_color)
                else:
                    ax2.plot(metric_values, label=f"{metric} Values (Max: {max(metric_values):.2f}, Epoch: {metric_values.index(max(metric_values))})", color=light_color, linestyle='--')

        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Metric Value (0-1)')
        ax2.set_ylabel('Metric Value (Large)')
        ax1.set_title(f'Metrics over Epochs (Values) ({file_name.split("_")[0]})')
        ax1.legend(loc='upper left')
        ax2.legend(loc='upper right')
        ax1.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'All_Metrics_Values_{file_name.split("_")[0]}_plot.png'))
        plt.clf()

    print("Metrics results plotted and saved!")

#Save Image with GT & Pred
def plot_segmentation_results(net, data_loader, save_path):
    total = len(data_loader)
    # 设置保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net.eval()
    softMax = nn.Softmax(dim=1)  # 注意dim=1，这是因为我们希望在类别维度上进行softmax

    for i, data in enumerate(data_loader):
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...\n", length=30)

        image, labels, labels_canny, img_names = data

        US = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(US)
        pred_y = softMax(segmentation_prediction)
        
        segmentation = getSingleImage(pred_y)  # 假设这个函数从softmax输出中获取2D分割图像



        for j in range(inputs.size(0)):
            # 提取图像名，用于保存
            img_name = img_names[j].split('/')[-1].replace('.png', '_overlay.png')

            # Convert predictions and labels to numpy arrays
            predictions_np = segmentation.cpu().detach().numpy()[j, 0, :, :]  # Assuming [batch_size, channels, height, width]
            labels_np = labels[j, 0, :, :].numpy()
            inputs_np = US.cpu().numpy()
            
            if inputs_np.ndim == 3:
                inputs_np = inputs_np.squeeze(0) if inputs_np.shape[0] == 1 else inputs_np.squeeze(2)
            if len(inputs_np.shape) == 2:
                inputs_np = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
            inputs_np = (inputs_np * 255).astype(np.uint8)
            inputs_np_2d = inputs_np[0, 0, :, :]

            original_img = inputs_np_2d
            prediction_img = predictions_np
            label_img = labels_np

            # Convert grayscale image to RGB while keeping the original grayscale values
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

            # Convert images to 8-bit format
            prediction_img = (prediction_img * 255).astype(np.uint8)
            label_img = (label_img * 255).astype(np.uint8)

            # Find contours
            contours_pred, _ = cv2.findContours(prediction_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_label, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Draw contours on the RGB image
            cv2.drawContours(original_img_rgb, contours_pred, -1, (0, 0, 255), 1)  # Red for prediction
            cv2.drawContours(original_img_rgb, contours_label, -1, (0, 255, 0), 1)  # Green for GT

            # Save images
            cv2.imwrite(os.path.join(save_path, img_name), original_img_rgb)

    print("Segmentation results plotted and saved!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', default='CC_CV_Unet_1022', type=str)
    parser.add_argument('--testModelName', default='CC_CV_Unet_1022', type=str)
    parser.add_argument('--batchSize', default=1, type=int)
    
    # Save Root
    base_dir = '/root/data1/wqf/FBUSAS/Results/CC/'
    Model ='CC_CV_Unet_1022'
    TestModel ='Epoch_40_CC_CV_Unet_1022.pth'

    parser.add_argument('--test_dataset_dir', default='/root/data1/wqf/FBUSAS/DataSet/CC-New/', type=str)
    parser.add_argument('--test_model_dir', default=os.path.join(base_dir, Model, 'Model', TestModel), type=str)
    parser.add_argument('--image_save_dir', default=os.path.join(base_dir, Model, 'Image/Seg_Res/', TestModel), type=str)
    parser.add_argument('--statistics_save_dir', default=os.path.join(base_dir, Model, 'Statistics'), type=str)  
    args=parser.parse_args()


    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_set = medicalDataLoader.MedicalImageDataset('val', args.test_dataset_dir, transform=transform, mask_transform=mask_transform)
    test_loader = DataLoader(test_set, batch_size=args.batchSize, shuffle=False)

    #net = FBUSAS_stack()
    #net = ZAM_WAM_Simplified_stack()
    #net = ZAM_WAM_Unet()
    net = CC_CV_Unet()
    net.load_state_dict(torch.load(args.test_model_dir))
    if torch.cuda.is_available():
        net.cuda()

    #saveImages_for2D(net, test_loader, 90, args.modelName, args.image_save_dir)
    #evaluate_model(net, test_loader, args.testModelName, args)
    #plot_metrics_from_files(args.statistics_save_dir, args.statistics_save_dir)
    plot_segmentation_results(net, test_loader, args.image_save_dir)