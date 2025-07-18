import os
import re
import cv2
import argparse
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader

from data import medicalDataLoader
from models.FBUSTS_Net_2 import (Unet,
                                 ResUNet,
                                 AttentionUnet,
                                 R2UNet,
                                 ChannelUNet,
                                 KiUNet,
                                 TransUNet,
                                 MedT,
                                 Shape_Unet,
                                 Shape_ZAM_WAM_Unet,
                                 Shape_Improved_ZAM_WAM_Unet,
                                 Channel_ZAM_WAM_Unet,
                                 Channel_Improved_ZAM_WAM_Unet
                                )
from common.progressBar import printProgressBar
from common.utils import (saveImages_for2D,
                          getSingleImage,
                          getOneHotSegmentation,
                          predToSegmentation,
                          getTargetSegmentation,
                          inferenceAllMetrics,
                          to_var,
                          weights_init,
                          save_model
                        )


# Save Segmentation Result
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

# Calculate TestModel Metrics
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

# Metrics Image from .npy/.txt
def plot_metrics_from_files(directory, save_directory, metrics=["Dice", "Jaccard", "Precision", "Recall", "Specificity", "Accuracy", "Hausdorff"]):
    
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

    for file_name in ['validation_statistics.txt']:
        # Plot each metric in a separate figure
        for metric in metrics:
            metric_values = []
            txt_file_path = os.path.join(directory, file_name)
            
            if os.path.exists(txt_file_path):
                with open(txt_file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        # match = re.search(rf"{metric}: (\d+\.\d+)", line)
                        # if match:
                        #     value = float(match.group(1))
                        #     metric_values.append(value)
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

def extract_metric_values(lines, metric):
    values = []
    for line in lines:
        match = re.search(rf"{metric}: (\d+\.\d+)", line)
        if match:
            value = float(match.group(1))
            values.append(value)
    return values

def plot_metrics_from_txt_files(directory, save_directory):
    
    files = ['validation_statistics.txt']
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple']

    for file_name in files:
        with open(os.path.join(directory, file_name), 'r') as f:
            lines = f.readlines()

        # Extract unique metrics from file
        metrics = list(set([m.split(':')[0].strip() for line in lines for m in line.split(',')]))
        metrics = [m for m in metrics if m not in ["Training", "Epoch"]]

        plt.figure(figsize=(10, 6))
        for metric in metrics:
            metric_values = extract_metric_values(lines, metric)
            plt.plot(metric_values, label=f"{metric} (Max: {max(metric_values):.2f}, Epoch: {metric_values.index(max(metric_values))})", color=colors[metrics.index(metric)])

        plt.xlabel('Epochs')
        plt.ylabel('Metric Value')
        plt.title(f'{file_name.split("_")[0]} Metrics over Epochs')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(save_directory, f'All_Metrics_{file_name.split("_")[0]}_plot.png'))
        plt.clf()

    print("Metrics results plotted and saved!")

# Save Image with GT & Pred
def plot_segmentation_results(net, data_loader, save_path):
    total = len(data_loader)
    # 设置保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net.eval()
    sigmoid = nn.Sigmoid().cuda()

    for i, data in enumerate(data_loader):
        
        printProgressBar(i, total, prefix="[Testing Inference] Getting segmentations...", length=30)
        images, labels, labels_canny, img_names = data

        Image = to_var(images)
        # Region
        region_Mask = to_var(labels)
        region = net(Image)[0]
        region_ = getSingleImage(sigmoid(region))
        # Edge
        edge = to_var(labels_canny)
        edge = net(Image)[1]
        edge_ = getSingleImage(sigmoid(edge))

        for j in range(Image.size(0)):
            # 提取图像名，用于保存
            img_name = img_names[j].split('/')[-1].replace('.png', '_overlay.png')

            # Convert predictions and labels to numpy arrays
            predictions_np = region_.cpu().detach().numpy()[j, 0, :, :]  # Assuming [batch_size, channels, height, width
            predictions_edge_np = edge_.cpu().detach().numpy()[j, 0, :, :]
            
            labels_np = labels[j, 0, :, :].numpy()
            labels_edge_np = labels_canny[j, 0, :, :].numpy()
            
            inputs_np = Image.cpu().numpy()
            
            if inputs_np.ndim == 3:
                inputs_np = inputs_np.squeeze(0) if inputs_np.shape[0] == 1 else inputs_np.squeeze(2)
            if len(inputs_np.shape) == 2:
                inputs_np = cv2.cvtColor(inputs_np, cv2.COLOR_GRAY2BGR)
            inputs_np = (inputs_np * 255).astype(np.uint8)
            inputs_np_2d = inputs_np[0, 0, :, :]

            original_img = inputs_np_2d
            prediction_img = predictions_np
            prediction_edge_img = predictions_edge_np
            label_img = labels_np
            label_edge_img = labels_edge_np

            # Convert grayscale image to RGB while keeping the original grayscale values
            original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_GRAY2RGB)

            # Convert images to 8-bit format
            prediction_img = (prediction_img * 255).astype(np.uint8)
            label_img = (label_img * 255).astype(np.uint8)

            prediction_edge_img = (prediction_edge_img * 255).astype(np.uint8)
            label_edge_img = (label_edge_img * 255).astype(np.uint8)

            # Find contours
            contours_pred, _ = cv2.findContours(prediction_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_label, _ = cv2.findContours(label_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if args.save_type == 'GT&PRED':
                 # Draw contours on the RGB image
                 cv2.drawContours(original_img_rgb, contours_pred, -1, (0, 0, 255), 1)  # Red for prediction
                 cv2.drawContours(original_img_rgb, contours_label, -1, (0, 255, 0), 1)  # Green for GT
                 # Resize the image to 256x256
                 resized_img = cv2.resize(original_img_rgb, (256, 256))
                 # Save images
                 cv2.imwrite(os.path.join(save_path, img_name), original_img_rgb)
                
            elif args.save_type == 'GT':
                 # Draw contours on the RGB image
                 cv2.drawContours(original_img_rgb, contours_label, -1, (0, 255, 0), 1)  # Green for GT
                 # Resize the image to 256x256
                 resized_img = cv2.resize(original_img_rgb, (256, 256))
                 # Save images
                 cv2.imwrite(os.path.join(save_path, img_name), original_img_rgb)
                
            elif args.save_type == 'GT-edge':
                # Save images
                cv2.imwrite(os.path.join(save_path, img_name), label_edge_img)
                
            elif args.save_type == 'GT-region':
                # Save images
                cv2.imwrite(os.path.join(save_path, img_name), label_img)
                
            elif args.save_type == 'Pred':
                 # Draw contours on the RGB image
                 cv2.drawContours(original_img_rgb, contours_pred, -1, (0, 0, 255), 1)  # Red for prediction
                 # Resize the image to 256x256
                 resized_img = cv2.resize(original_img_rgb, (256, 256))
                 # Save images
                 cv2.imwrite(os.path.join(save_path, img_name), original_img_rgb)
            
            elif args.save_type == 'Pred-edge':
                # Save images
                cv2.imwrite(os.path.join(save_path, img_name), prediction_edge_img)
            
            elif args.save_type == 'Pred-region':
                # Save images
                cv2.imwrite(os.path.join(save_path, img_name), prediction_img)

    print("Segmentation results plotted and saved!")

# Save Image with GT & Pred in original size
def plot_segmentation_results_orgsize(net, data_loader, save_path, cc_path):
    total = len(data_loader)
    
    # 设置保存路径
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    net.eval() 
    sigmoid = nn.Sigmoid().cuda()

    for i, data in enumerate(data_loader):

        printProgressBar(i, total, prefix="[Testing Inference] Getting segmentations...", length=30)
        images, labels, labels_canny, img_names = data

        Image = to_var(images)
        region_Mask = to_var(labels)
        # region,_ = net(Image)
        _, _, region  = net(Image)
        region_ = getSingleImage(sigmoid(region))

        for j in range(Image.size(0)):
            # Extract image name for saving results
            img_name = img_names[j].split('/')[-1]
            original_img_path = os.path.join(cc_path, "Img", img_name)
            original_label_path = os.path.join(cc_path, "GT", img_name)

            # Load original image and label
            original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
            original_label = cv2.imread(original_label_path, cv2.IMREAD_GRAYSCALE)

            # Convert image, label, prediction to numpy arrays
            predictions_np = region_.cpu().detach().numpy()[j, 0, :, :]
            
            # Use the loaded image and label directly since they are already numpy arrays
            inputs_np = original_img
            labels_np = original_label
            
            # Adjusting image dimensions if needed
            if inputs_np.ndim == 3:
                if inputs_np.shape[0] == 1:
                    inputs_np = inputs_np.squeeze(0)
                elif inputs_np.shape[2] == 1:
                    inputs_np = inputs_np.squeeze(2)
                inputs_np_2d = inputs_np[0, :, :]
            else:
                inputs_np_2d = inputs_np

            # Convert to BGR if it's a grayscale image
            if inputs_np_2d.ndim == 2:
                inputs_np_2d = cv2.cvtColor(inputs_np_2d, cv2.COLOR_GRAY2BGR)

            # Scale the image values
            inputs_np_2d = (inputs_np_2d * 255).astype(np.uint8)
            predictions_np = (predictions_np * 255).astype(np.uint8)
            
            # Resize prediction to match the original image size
            prediction_img_resized = cv2.resize(predictions_np, (inputs_np_2d.shape[1], inputs_np_2d.shape[0]), interpolation=cv2.INTER_CUBIC)

            # Apply Gaussian Blur
            prediction_img_resized = cv2.GaussianBlur(prediction_img_resized, (5,5), 0)

            # Convert prediction image to binary
            binary_prediction = np.where(prediction_img_resized > 0, 255, 0).astype(np.uint8)

            # Create a 5x5 kernel
            kernel = np.ones((5,5),np.uint8)

            # Apply morphological operations
            opening = cv2.morphologyEx(binary_prediction, cv2.MORPH_OPEN, kernel)
            closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

            # Use the morphologically improved image as the new prediction image
            prediction_img_new = closing

            # Extract contours from the new prediction and the original label
            contours_pred_new, _ = cv2.findContours(prediction_img_new, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours_label_original, _ = cv2.findContours(labels_np.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


            # Convert grayscale original image to RGB
            original_img_rgb = cv2.imread(original_img_path, cv2.IMREAD_COLOR)

            # Draw contours on the RGB image
            cv2.drawContours(original_img_rgb, contours_pred_new, -1, (0, 0, 255), 2)  # Red for prediction
            cv2.drawContours(original_img_rgb, contours_label_original, -1, (0, 255, 0), 2)  # Green for label

            # Save the processed image
            cv2.imwrite(os.path.join(save_path, img_name.replace('.png', '_overlay.png')), original_img_rgb)

    print("Segmentation results plotted and saved!")

if __name__ == '__main__':
 ###  Testing  ###
    #   Model   #
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelName', default='Unet', type=str)
    parser.add_argument('--testModelName', default='Unet', type=str)
    parser.add_argument('--batchSize', default=1, type=int)
    # Save Root #
    base_dir = '/newhome/wangqifeng/FBUSTS-3.0/Results/CV/'
    Model ='UNet'
    TestModel ='Best_UNet_(Epoch-56).pth'
 
    parser.add_argument('--org_dataset_dir', default='/newhome/wangqifeng/FBUSTS-3.0/DataSet/CV/', type=str)
    parser.add_argument('--test_dataset_dir', default='/newhome/wangqifeng/FBUSTS-3.0/DataSet/CV-128/', type=str)
    # parser.add_argument('--test_model_dir', default=os.path.join(base_dir, Model, 'Model', TestModel), type=str)
    parser.add_argument('--test_model_dir', default=os.path.join(base_dir, Model, TestModel), type=str)
    parser.add_argument('--image_save_dir', default=os.path.join(base_dir, Model, 'Image', TestModel), type=str)
    parser.add_argument('--statistics_save_dir', default=os.path.join(base_dir, Model, 'Statistics'), type=str)
    
    parser.add_argument('--save_type', default='Pred', type=str) # GT&Pred, GT, GT-edge, GT-region, Pred, Pred-edge, Pred-region
    args=parser.parse_args()

 ###  Runing  ###
    # Test Data Loading
    transform = transforms.Compose([transforms.ToTensor()])
    test_set = medicalDataLoader.MedicalImageDataset('val', 
                                                    args.test_dataset_dir, 
                                                    transform=transform)
    test_loader = DataLoader(test_set, 
                            batch_size=args.batchSize, 
                            shuffle=False)
    
    # Test Model Setting
    if args.testModelName == 'Unet':
        net = Unet()
    elif args.testModelName == 'ResUNet':
        net = ResUNet()
    elif args.testModelName == 'AttentionUnet':
        net = AttentionUnet()
    elif args.testModelName == 'R2UNet':
        net = R2UNet()
    elif args.testModelName == 'ChannelUNet':
        net = ChannelUNet()
    elif args.testModelName == 'KiUNet':
        net = KiUNet()
    elif args.testModelName == 'TransUNet':
        net = TransUNet()
    elif args.testModelName == 'MedT':
        net = MedT()
    elif args.testModelName == 'Shape_Unet':
        net = Shape_Unet()
    elif args.testModelName == 'Shape_ZAM_WAM_Unet':
        net = Shape_ZAM_WAM_Unet()
    elif args.testModelName == 'Shape_Improved_ZAM_WAM_Unet':
        net = Shape_Improved_ZAM_WAM_Unet()
    elif args.testModelName == 'Channel_ZAM_WAM_Unet':
        net = Channel_ZAM_WAM_Unet()
    elif args.testModelName == 'Channel_Improved_ZAM_WAM_Unet':
        net = Channel_Improved_ZAM_WAM_Unet()
    else:
        print("Invalid model name")
        exit()

    net.load_state_dict(torch.load(args.test_model_dir))
    if torch.cuda.is_available():
        net.cuda()

    # saveImages_for2D(net, test_loader, 90, args.modelName, args.image_save_dir)
    # evaluate_model(net, test_loader, args.testModelName, args)
    # plot_metrics_from_files(args.statistics_save_dir, args.statistics_save_dir)
    plot_segmentation_results(net, test_loader, args.image_save_dir)
    # plot_segmentation_results_orgsize(net, test_loader, args.image_save_dir, args.org_dataset_dir)