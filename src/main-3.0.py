import os
import numpy as np
import dill
import argparse
import time
import glob
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F

from data import medicalDataLoader
from models.FBUSTS_Net_2 import (Shape_Improved_ZAM_WAM_Unet,
                                 Channel_Improved_ZAM_WAM_Unet,
                                 Channel_ZAM_WAM_Unet,
                                 Week_Improved_ZAM_WAM_Unet,
                                 Week_Improved_AttentionUnet,
                                 Week_Feature_AttentionUnet,
                                 Modified_Feature_AttentionUnet,
                                 Modified_Feature_Improved_Unet            
                                )
from common.progressBar import printProgressBar
from common.utils import (saveImages_for2D,
                          getOneHotSegmentation,
                          predToSegmentation,
                          getTargetSegmentation,
                          inferenceAllMetrics,
                          inferenceAllMetrics_week,
                          to_var,
                          weights_init,
                          save_model
                        )
from common.loss import (DiceLoss,
                         FocalLoss,
                         TverskyLoss,
                         Unet_Loss,
                         Shape_Unet_Loss,
                         Channel_Unet_Loss,
                         Week_Unet_Loss
                        )

def runTraining(args):

  ################### Initialize ###################
    
    print('-' * 40)
    print('~~~~~~  Starting the Initialize ~~~~~~')
    print('-' * 40)

    ### Parameter ###
    batch_size = args.batch_size
    lr = args.lr
    epoch = args.epochs
    root_dir = args.root
    week_info_path = args.week_path
    batch_size_val,  batch_size_val_save = 1, 1
    model_dir = 'model'

    data_Name = args.dataName
    print(' Dataset: {} '.format(data_Name))

    ### Data ###
    transform = transforms.Compose([transforms.ToTensor()])

    train_set = medicalDataLoader.MedicalImageDataset_week('train',
                                                      root_dir,
                                                      week_info_path,
                                                      transform=transform,
                                                      augment=True,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset_week('val',
                                                    root_dir,
                                                    week_info_path,
                                                    transform=transform,
                                                    equalize=False) 

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=args.num_workers,
                            shuffle=False)
                                                   
    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        num_workers= args.num_workers,
                                        shuffle=False)

                                                                    
    ### Model ###
    print('-' * 40)
    print("~~~~~~    Creating the Model    ~~~~~~")
    print('-' * 40)


    if args.model == 'Shape_Improved_ZAM_WAM_Unet':
        net = Shape_Improved_ZAM_WAM_Unet()
    elif args.model == 'Channel_Improved_ZAM_WAM_Unet':
        net = Channel_Improved_ZAM_WAM_Unet()
    elif args.model == 'Channel_ZAM_WAM_Unet':
        net = Channel_ZAM_WAM_Unet()
    elif args.model == 'Week_Improved_ZAM_WAM_Unet':
        net = Week_Improved_ZAM_WAM_Unet()
    elif args.model == 'Week_Improved_AttentionUnet':
        net = Week_Improved_AttentionUnet()
    elif args.model == 'Week_Feature_AttentionUnet':
        net = Week_Feature_AttentionUnet()
    elif args.model == 'Modified_Feature_AttentionUnet':
        net = Modified_Feature_AttentionUnet()
    elif args.model == 'Modified_Feature_Improved_Unet':
        net = Modified_Feature_Improved_Unet()
    else:
        print("Invalid model name")
        exit()

    net.apply(weights_init)
    print(" Experiment Name: {}".format(args.modelName))
    print(" Model to create: {}".format(args.model))
    print('-' * 40)

    ### Loss Function ###
    BCE_loss = nn.BCEWithLogitsLoss().cuda()
    CE_loss = nn.CrossEntropyLoss().cuda()
    Mse_Loss = nn.MSELoss().cuda()
    Dice_Loss = DiceLoss().cuda()
    Focal_Loss =  FocalLoss().cuda()
    Tversky_Loss = TverskyLoss().cuda()
    
    ### Optimizer ###
    optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=True)
    print(f" Optimizer: Adam")
    print(f" Scheduler: ReduceLROnPlateau")
    if args.control_lr:
        mode = 'min'
        print(f" Use Train Loss control lr")
    else:
        mode = 'max' 
        print(f" Use Val Dice control lr")
    scheduler = ReduceLROnPlateau(optimizer, mode=mode, factor=0.5, patience=10, verbose=True)
    print('-' * 40)
 
  ###################   Train   ###################
    
    print("~~~~~~   Starting the Training   ~~~~~~")
    print('-' * 40)
    
    total_training_time = 0
    BestDice, BestEpoch = 0, 0
    Losses = []
    for i in range(epoch):
        net.train()
        start_time = time.time()
        metrics_str = ""
        lossVal = []
        
        totalImages = len(train_loader)
        ### Starting Loop ###
        for j, data in enumerate(train_loader):
            
            images, labels, labels_canny, img_names, week = data

            if images.size(0) != batch_size:
                continue
    
            optimizer.zero_grad()
            net.zero_grad()

            # Image, label, Canny_label
            Image = to_var(images)
            region_Mask = to_var(labels)
            edge_Mask = to_var(labels_canny)

            region_Mask_onehot = getOneHotSegmentation(getTargetSegmentation(region_Mask))

            # Forward
            if args.model == 'Shape_Improved_ZAM_WAM_Unet':
                edge_Mask_onehot = getOneHotSegmentation(getTargetSegmentation(edge_Mask))

                net_output = net(Image)
                region, edge, region_512 = net_output
                Loss, loss_region, loss_edge, loss_region_dice = Shape_Unet_Loss(region, edge, region_Mask_onehot, edge_Mask_onehot, BCE_loss, Dice_Loss, net)
            
            elif args.model == 'Channel_ZAM_WAM_Unet' or args.model == 'Channel_Improved_ZAM_WAM_Unet':
                edge_Mask_onehot = getOneHotSegmentation(getTargetSegmentation(edge_Mask))

                net_output = net(Image)
                region, edge, region_512 = net_output
                Loss, loss_region, loss_edge, loss_region_dice = Channel_Unet_Loss(region, edge, region_Mask_onehot, edge_Mask_onehot, BCE_loss, Dice_Loss, Focal_Loss, Tversky_Loss, net)

            elif args.model == 'Week_Improved_ZAM_WAM_Unet' or args.model == 'Week_Improved_AttentionUnet' or args.model == 'Week_Feature_AttentionUnet' or args.model == 'Modified_Feature_AttentionUnet' or args.model == 'Modified_Feature_Improved_Unet':
                week_gt = week

                net_output = net(Image, week)
                region, week = net_output
                Loss, loss_region, loss_week, loss_region_dice = Week_Unet_Loss(region, week, region_Mask_onehot, week_gt, BCE_loss, Dice_Loss, Mse_Loss, net)
            
            else:
                print("Invalid model name")
                exit()            
            
            # Backward
            Loss.backward()
            optimizer.step()
            lossVal.append(Loss.cpu().data.numpy())
            
            # Training Metrics
            if args.model == 'Shape_Improved_ZAM_WAM_Unet' or args.model == 'Channel_ZAM_WAM_Unet' or args.model == 'Channel_Improved_ZAM_WAM_Unet':
                metrics_str = ("[Training] [Epoch-{}] ".format(i) +
                                "Dice: {:.4f}, Loss: {:.4f}, loss_region: {:.4f}, loss_edge: {:.4f}".format(
                                1-loss_region_dice.cpu().data.numpy(),
                                Loss.cpu().data.numpy(),
                                loss_region.cpu().data.numpy(),
                                loss_edge.cpu().data.numpy()
                                ))
            elif args.model == 'Week_Improved_ZAM_WAM_Unet' or args.model == 'Week_Improved_AttentionUnet' or args.model == 'Week_Feature_AttentionUnet' or args.model == 'Modified_Feature_AttentionUnet' or args.model == 'Modified_Feature_Improved_Unet':
                metrics_str = ("[Training] [Epoch-{}] ".format(i) +
                                "Dice: {:.4f}, Loss: {:.4f}, loss_region: {:.4f}, loss_week: {:.4f}".format(
                                1-loss_region_dice.cpu().data.numpy(),
                                Loss.cpu().data.numpy(),
                                loss_region.cpu().data.numpy(),
                                loss_week.cpu().data.numpy()
                                ))
            
            printProgressBar(j + 1, totalImages,
                             prefix="[Training] [Epoch-{}]".format(i),
                             length=15,
                             suffix=metrics_str)
        
        ### Training Time ###
        end_time = time.time() 
        epoch_time = end_time - start_time  
        total_training_time += epoch_time  
        epoch_time_minutes = epoch_time / 60
        total_training_time_minutes = total_training_time / 60
        print(f"[Training] [Epoch-{i}] Time taken: {epoch_time_minutes:.2f} minutes, Total time: {total_training_time_minutes:.2f} minutes\n")

        ### Saving Training Statistics ###
        lr_for_save = optimizer.param_groups[0]['lr']

        if not os.path.exists(args.statistics_save_dir):
            os.makedirs(args.statistics_save_dir)
        
        with open(os.path.join(args.statistics_save_dir, 'training_statistics.txt'), 'a') as f:
            f.write("[Training] [Epoch-{}] ".format(i))
            if args.model == 'Shape_Improved_ZAM_WAM_Unet' or args.model == 'Channel_ZAM_WAM_Unet' or args.model == 'Channel_Improved_ZAM_WAM_Unet':
                f.write("Dice: {:.4f}, Loss: {:.4f}, loss_region: {:.4f}, loss_edge: {:.4f}, learning_rate: {:.1e}\n".format(
                        1-loss_region_dice.cpu().data.numpy(),
                        Loss.cpu().data.numpy(),
                        loss_region.cpu().data.numpy(),
                        loss_edge.cpu().data.numpy(),
                        lr_for_save
                        ))
            elif args.model == 'Week_Improved_ZAM_WAM_Unet' or args.model == 'Week_Improved_AttentionUnet' or args.model == 'Week_Feature_AttentionUnet'or args.model == 'Modified_Feature_AttentionUnet' or args.model == 'Modified_Feature_Improved_Unet':
                f.write("Dice: {:.4f}, Loss: {:.4f}, loss_region: {:.4f}, loss_week: {:.4f}, learning_rate: {:.1e}\n".format(
                        1-loss_region_dice.cpu().data.numpy(),
                        Loss.cpu().data.numpy(),
                        loss_region.cpu().data.numpy(),
                        loss_week.cpu().data.numpy(),
                        lr_for_save
                        ))

        ### Save Training Model ###
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)
        if i % args.save_interval == 0:
            # Save Model
            torch.save(net.state_dict(), os.path.join(args.model_save_dir, "Epoch_{}_{}".format(i, args.modelName) + ".pth"))
            if args.save_optimizer:
                # Save Optimizer
                torch.save(optimizer.state_dict(), os.path.join(args.model_save_dir, "Epoch_{}_{}".format(i, args.modelName) + "_optimizer.pth"))
  
  ###################    Val    ###################

        ### Save Validation Statistics ###
        modelName = args.modelName
        directory = args.statistics_save_dir

        Losses.append(np.mean(lossVal))
        metrics_results = inferenceAllMetrics_week(net, val_loader, args)
        
        # Save Validation Loss
        if not os.path.exists(directory):
            os.makedirs(directory)
        np.save(os.path.join(directory, 'Losses.npy'), Losses)
        
        # Save Validation Metrics
        for metric, values in metrics_results.items():
            np.save(os.path.join(directory, f'{metric}Val.npy'), values.cpu().numpy())
        
        # Save Validation Statistics in txt
        if not os.path.exists(args.statistics_save_dir):
                os.makedirs(args.statistics_save_dir)
        metrics_str_list = []
        with open(os.path.join(args.statistics_save_dir, 'validation_statistics.txt'), 'a') as f:
            for metric, values in metrics_results.items():
                current_value = values.mean().item()
                metrics_str_list.append(f"{metric}: {current_value:.4f}")
            metrics_str = ", ".join(metrics_str_list)
            print(f"[Validation] [Epoch-{i}] {metrics_str}")
            f.write(f"[Epoch-{i}] {metrics_str}\n")
     
        # Save Best Model
        dice_value = metrics_results["Dice"].mean().item()
        currentDice = dice_value
        if currentDice > BestDice :
            BestDice = currentDice
            BestEpoch = i
            if currentDice > 0.60:
                print("~~~~~~   Saving best model   ~~~~~~")
                best_model_path = os.path.join(args.model_save_dir, f"Best_{args.modelName}_*.pth")
                for file in glob.glob(best_model_path):
                    os.remove(file)
                if not os.path.exists(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                save_model(net, os.path.join(args.model_save_dir, f"Best_{args.modelName}_(Epoch-{i}).pth"))

        print("[* Best Dice *][Epoch-{}] {:.4f} ".format(BestEpoch, BestDice))

        ### Learning Rate ###
        if args.control_lr:
            scheduler.step(np.mean(lossVal))
        else:
            scheduler.step(dice_value)
        current_lr = optimizer.param_groups[0]['lr']
        print("[Training] [Epoch-{}] Train Loss: {:.4f}, Val Dice: {:.4f}, Learning Rate: {:.1e}".format(i, np.mean(lossVal), dice_value, current_lr))

        ###  Early stop  ###
        best_score = -np.inf
        counter = 0
        patience_early_stopping = 20  # Early stop after 20 steps
        delta = 0.001
        if dice_value > best_score + delta:
            best_score = dice_value
            counter = 0
        else:
            counter += 1
        if counter > patience_early_stopping:
            print("Early stopping triggered at Epoch-{}!".format(i))
            break
    with open(os.path.join(args.statistics_save_dir, 'validation_statistics.txt'), 'a') as f:
                f.write(f"\n[* Best Dice *][Epoch-{BestEpoch}] {BestDice:.4f}\n")

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
 ### Training ###
    #   Model   #
    parser.add_argument('--model', default='Modified_Feature_Improved_Unet', type=str, help='Name of the model to use')
    parser.add_argument('--modelName', default = 'Modified_Feature_Improved_Unet',type=str)
    #   Data    #
    parser.add_argument('--root', default='/root/data1/wqf/FBUSTS-3.0/DataSet/CV-New/', type=str)
    parser.add_argument('--week_path', default='/root/data1/wqf/FBUSTS-3.0/DataSet/CV-New/CV_name_with_week.txt', type=str)
    parser.add_argument('--dataName', default = 'CV-New',type=str)
    # Parameter #
    parser.add_argument('--num_workers', default = 36, type = int)
    parser.add_argument('--batch_size',default = 16,type = int)
    parser.add_argument('--epochs',default =200,type = int)
    parser.add_argument('--lr',default = 0.001,type = float)
    parser.add_argument('--control_lr', type=bool, default=False, help='Use train loss for LR adjustment if True, else use val Dice')

    #   Save    #
    base_dir = '/root/data1/wqf/FBUSTS-3.0/Results/CV/'
    Model ='Modified_Feature_Improved_Unet'
    parser.add_argument('--statistics_save_dir', default=os.path.join(base_dir, Model, 'Statistics'), type=str)
    parser.add_argument('--image_save_dir', default=os.path.join(base_dir, Model, 'Image'), type=str)
    parser.add_argument('--model_save_dir', default=os.path.join(base_dir, Model, 'Model'), type=str)
    parser.add_argument('--save_interval', type=int, default=20, help='Interval (number of epochs) to save the model')
    parser.add_argument('--save_optimizer', type=bool, default=False, help='Set this flag to save optimizer state')

 ###  Runing  ###
    args=parser.parse_args()
    runTraining(args)