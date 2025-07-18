import os
import numpy as np
import dill
import argparse
import time
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
from models.FBUSYS_Net_2 import (Basic_Unet, 
                                Shape_Unet, 
                                )
from common.progressBar import printProgressBar
from common.utils import (saveImages_for2D,
                          getOneHotSegmentation,
                          predToSegmentation,
                          getTargetSegmentation,
                          computeDiceLoss,
                          computeJaccardLoss,
                          computePrecisionLoss,
                          computeRecallLoss,
                          computeSpecificityLoss,
                          computeAccuracyLoss,
                          computeHausdorffLoss,
                          computeDice,
                          computeJaccard,
                          computePrecision,
                          computeRecall,
                          computeSpecificity,
                          computeAccuracy,
                          computeHausdorff,
                          MetricsToMetric,
                          HausdorffsToHausdorff,
                          inferenceAllMetrics_1,
                          inferenceAllMetrics_2,
                          to_var
                          )
from common.loss import (Shape_Unet_Loss_1,
                         Shape_Unet_Loss_2
                          )


def weights_init(m):
    if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
        nn.init.xavier_normal(m.weight.data)
    elif type(m) == nn.BatchNorm2d:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def save_model(net, path):
    try:
        torch.save(net.state_dict(), path)
    except Exception as e:
        print(f"Error saving model: {e}")

def runTraining(args):
    print('-' * 40)
    print('~~~~~~  Starting the training... ~~~~~~')
    print('-' * 40)

    batch_size = args.batch_size
    batch_size_val = 1
    batch_size_val_save = 1
    lr = args.lr

    epoch = args.epochs
    root_dir = args.root
    model_dir = 'model'

    txt_save_path = os.path.join(args.statistics_save_dir)
    if not os.path.exists(args.statistics_save_dir):
        os.makedirs(args.statistics_save_dir)
    
    print(' Dataset: {} '.format(root_dir))

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    mask_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_set = medicalDataLoader.MedicalImageDataset_2('train',
                                                      root_dir,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=True,
                                                      equalize=False)

    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)

    val_set = medicalDataLoader.MedicalImageDataset_2('val',
                                                    root_dir,
                                                    transform=transform,
                                                    mask_transform=mask_transform,
                                                    equalize=False) 

    val_loader = DataLoader(val_set,
                            batch_size=batch_size_val,
                            num_workers=args.num_workers,
                            shuffle=False)
                                                   
    val_loader_save_images = DataLoader(val_set,
                                        batch_size=batch_size_val_save,
                                        num_workers= args.num_workers,
                                        shuffle=False)

                                                                    
    # Initialize
    print('-' * 40)
    print("~~~~~~ Creating the SATALA model ~~~~~~")

    if args.model == 'Basic_Unet':
        net = Basic_Unet()
    elif args.model == 'Shape_Unet':
        net = Shape_Unet()
    
        # 载入SP训练集
        root_dir_SP = args.root_SP
        train_set_SP = medicalDataLoader.MedicalImageDataset_2('train',
                                                      root_dir_SP,
                                                      transform=transform,
                                                      mask_transform=mask_transform,
                                                      augment=True,
                                                      equalize=False)

        train_loader_SP = DataLoader(train_set_SP,
                                batch_size=batch_size,
                                num_workers=args.num_workers,
                                shuffle=True)

        val_set_SP = medicalDataLoader.MedicalImageDataset_2('val',
                                                        root_dir_SP,
                                                        transform=transform,
                                                        mask_transform=mask_transform,
                                                        equalize=False)

        val_loader_SP = DataLoader(val_set_SP,
                                batch_size=batch_size_val,
                                num_workers=args.num_workers,
                                shuffle=False)
                                                    
        val_loader_save_images_SP = DataLoader(val_set_SP,
                                            batch_size=batch_size_val_save,
                                            num_workers= args.num_workers,
                                            shuffle=False)

    else:
        print("Invalid model name")
        exit()
    
    # Pretrained_Initialize
    if args.pretrained:
        net.load_state_dict(torch.load(args.pretrained))
        print(f" Loaded model from {args.pretrained}")
    else:
        print(f" New model ")

    net.apply(weights_init)

    softMax = nn.Softmax()
    BCE_loss = nn.BCEWithLogitsLoss()
    CE_loss = nn.CrossEntropyLoss()
    MseLoss = nn.MSELoss()
    ### Loss for 1 channel ###
    DiceLoss = computeDiceLoss().cuda()
    JaccardLoss = computeJaccardLoss().cuda()
    PrecisionLoss = computePrecisionLoss().cuda()
    RecallLoss = computeRecallLoss().cuda()
    SpecificityLoss = computeSpecificityLoss().cuda()
    AccuracyLoss = computeAccuracyLoss().cuda()
    HausdorffLoss = computeHausdorffLoss().cuda()
    ### Loss for 2 channels ###
    Dice_loss = computeDice().cuda()
    Jaccard_loss = computeJaccard().cuda()
    Precision_loss = computePrecision().cuda()
    Recall_loss = computeRecall().cuda()
    Specificity_loss = computeSpecificity().cuda()
    Accuracy_loss = computeAccuracy().cuda()
    Hausdorff_loss = computeHausdorff().cuda()
    
    
    if torch.cuda.is_available():
        net.cuda()
        softMax.cuda()
        BCE_loss.cuda()
        Dice_loss.cuda()
    
    # 学习率/优化器设置
    if args.pretrained_optimizer:
        optimizer.load_state_dict(torch.load(args.pretrained_optimizerd))
        print(f" Loaded optimizer from {args.pretrained_optimizerd}")
    else:
        optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.99), amsgrad=True)
        print(f" New optimizer ")
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True)

    print(" Model Name: {}".format(args.modelName))
    print(" Model to create: {}".format(args.model))
    print('-' * 40)

    total_training_time = 0

    BestDice, BestEpoch = 0, 0
    d1Val = []
    Losses = []

    print("~~~~~~  Starting the training  ~~~~~~")
    for i in range(args.start_epoch, args.epochs):
        net.train()
        start_time = time.time()
        metrics_str = ""
        lossVal = []
        
        totalImages = len(train_loader)
        
        for j, data in enumerate(train_loader):
            image, labels, labels_canny, img_names = data
        
            if image.size(0) != batch_size:
                continue

            optimizer.zero_grad()
            US = to_var(image)
            #label
            region_Mask = to_var(labels)
            #Canny_label
            edge_Mask = to_var(labels_canny)

            ################### Train ###################
            net.zero_grad()

            # Network outputs
            net_output = net(US)
            
            ###################  Val  ###################
            ### 1 channel ###
            # region, edge = net_output
            # Loss = Shape_Unet_Loss_1(region, edge, region_Mask, edge_Mask, BCE_loss, DiceLoss, net)
            ### 2 channels ###
            region, edge = net_output
            region_sigmoid = torch.sigmoid(region)
            # edge_sigmoid = torch.sigmoid(edge)
            region_ = predToSegmentation(region_sigmoid)
            region_Mask_ = getOneHotSegmentation(region_Mask)
            region_Mask_onehot = getOneHotSegmentation(getTargetSegmentation(region_Mask))
            edge_Mask_onehot = getOneHotSegmentation(getTargetSegmentation(edge_Mask))
            Loss, loss_region, loss_edge = Shape_Unet_Loss_2(region, edge, region_Mask_onehot, edge_Mask_onehot, BCE_loss, DiceLoss, net)
            

            Loss.backward()
            optimizer.step()
            lossVal.append(Loss.cpu().data.numpy())
            
            
            ### Compute All Metrics ###
            
            ### 1 channel ###
            # Dice_score = DiceLoss(region, region_Mask)
            # Jaccard_score = JaccardLoss(region, region_Mask)
            # Precision_score = PrecisionLoss(region, region_Mask)
            # Recall_score = RecallLoss(region, region_Mask)
            # Specificity_score = SpecificityLoss(region, region_Mask)
            # Accuracy_score = AccuracyLoss(region, region_Mask)
            # Hausdorff_score = HausdorffLoss(region, region_Mask)


            ### 2 channels ###
            _, Dice_score = Dice_loss(region_, region_Mask_)
            Dice_score = MetricsToMetric(Dice_score)

            _, Jaccard_score = Jaccard_loss(region_, region_Mask_)
            Jaccard_score = MetricsToMetric(Jaccard_score)

            _, Precision_score = Precision_loss(region_, region_Mask_)
            Precision_score = MetricsToMetric(Precision_score)

            _, Recall_score = Recall_loss(region_, region_Mask_)
            Recall_score = MetricsToMetric(Recall_score)

            _, Specificity_score = Specificity_loss(region_, region_Mask_)
            Specificity_score = MetricsToMetric(Specificity_score)

            _, Accuracy_score = Accuracy_loss(region_, region_Mask_)
            Accuracy_score = MetricsToMetric(Accuracy_score)

            Hausdorff_score = Hausdorff_loss(region_, region_Mask_)
            Hausdorff_score = HausdorffsToHausdorff(Hausdorff_score)


            # metrics_str = ("[Training] [Epoch-{}] ".format(i) +
            #                 "Dice: {:.4f}, Jaccard: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Accuracy: {:.4f}, Hausdorff: {:.4f}".format(
            #                 Dice_score.cpu().data.numpy(),
            #                 Jaccard_score.cpu().data.numpy(),
            #                 Precision_score.cpu().data.numpy(),
            #                 Recall_score.cpu().data.numpy(),
            #                 Specificity_score.cpu().data.numpy(),
            #                 Accuracy_score.cpu().data.numpy(),
            #                 Hausdorff_score.cpu().data.numpy()
            #                 ))

            metrics_str = ("[Training] [Epoch-{}] ".format(i) +
                            "Dice: {:.4f}, Loss: {:.4f}, loss_region: {:.4f}, loss_edge: {:.4f}".format(
                            Dice_score.cpu().data.numpy(),
                            Loss.cpu().data.numpy(),
                            loss_region.cpu().data.numpy(),
                            loss_edge.cpu().data.numpy()
                            ))
            
            printProgressBar(j + 1, totalImages,
                             prefix="[Training] [Epoch-{}] ".format(i),
                             length=15,
                             suffix=metrics_str)
        
        
        ### Learning Rate ###
        scheduler.step(np.mean(lossVal))
        current_lr = optimizer.param_groups[0]['lr']
        print("[Training] [Epoch-{}] Loss: {:.4f}, Learning Rate: {:.4f}".format(i, np.mean(lossVal), current_lr))        


        ### Print Training Time ###
        end_time = time.time() 
        epoch_time = end_time - start_time  
        total_training_time += epoch_time  
        epoch_time_minutes = epoch_time / 60
        total_training_time_minutes = total_training_time / 60
        print(f"[Training] [Epoch-{i}] Time taken: {epoch_time_minutes:.2f} minutes, Total time: {total_training_time_minutes:.2f} minutes")

        
        ################### Save ###################

        ### Save Training Statistics ###
        if not os.path.exists(args.statistics_save_dir):
            os.makedirs(args.statistics_save_dir)
        
        with open(os.path.join(args.statistics_save_dir, 'training_statistics.txt'), 'a') as f:
            f.write("[Training] [Epoch-{}] ".format(i))
            f.write("Dice: {:.4f}, Jaccard: {:.4f}, Precision: {:.4f}, Recall: {:.4f}, Specificity: {:.4f}, Accuracy: {:.4f}, Hausdorff: {:.4f}\n".format(
                    Dice_score.cpu().data.numpy(),
                    Jaccard_score.cpu().data.numpy(),
                    Precision_score.cpu().data.numpy(),
                    Recall_score.cpu().data.numpy(),
                    Specificity_score.cpu().data.numpy(),
                    Accuracy_score.cpu().data.numpy(),
                    Hausdorff_score.cpu().data.numpy()
                    ))

        ### Save Training Model ###
        if not os.path.exists(args.model_save_dir):
            os.makedirs(args.model_save_dir)

        if i % 5 == 0:
            ### Save Model ###
            torch.save(net.state_dict(), os.path.join(args.model_save_dir, "Epoch_{}_{}".format(i, args.modelName) + ".pth"))# 每5个epoch保存一次模型
            ### Save Optimizer ###
            torch.save(optimizer.state_dict(), os.path.join(args.model_save_dir, "Epoch_{}_{}".format(i, args.modelName) + "_optimizer.pth"))

        
        ### Save Validation Statistics ###
        modelName = args.modelName
        directory = args.statistics_save_dir

        Losses.append(np.mean(lossVal))
        ### 1 channel ###
        # metrics_results = inferenceAllMetrics_1(net, val_loader) 
        ### 2 channels ###
        metrics_results = inferenceAllMetrics_2(net, val_loader) 


        if not os.path.exists(directory):
            os.makedirs(directory)

        np.save(os.path.join(directory, 'Losses.npy'), Losses)
        for metric, values in metrics_results.items():
            ### 1 channel ###
            # np.save(os.path.join(directory, f'{metric}Val.npy'), values.cpu().detach().numpy())
            ### 2 channels ###
            np.save(os.path.join(directory, f'{metric}Val.npy'), values.cpu().numpy())

        metrics_str_list = []
        with open(os.path.join(args.statistics_save_dir, ' validation_statistics.txt'), 'a') as f:
            for metric, values in metrics_results.items():
                current_value = values.mean().item()
                metrics_str_list.append(f"{metric}: {current_value:.4f}")
            metrics_str = ", ".join(metrics_str_list)
            print(f"[Validation] [Epoch-{i}] {metrics_str}")
            f.write(f"[Epoch-{i}] {metrics_str}\n")
     
        dice_value = metrics_results["Dice"].mean().item()
        currentDice = dice_value

        
        if currentDice > BestDice :
            BestDice = currentDice
            BestEpoch = i

            if currentDice > 0.40:
                print("### Saving best model... ###")
                if not os.path.exists(args.model_save_dir):
                    os.makedirs(args.model_save_dir)
                save_model(net, os.path.join(args.model_save_dir, f"Best_{args.modelName}.pth"))


        print("[* Best Dice *][Epoch-{}] {:.4f} ".format(BestEpoch, BestDice))

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    #Change Your Net
    parser.add_argument('--model', default='Shape_Unet', type=str, help='Name of the model to use')
    
    parser.add_argument('--modelName', default = 'Shape_Unet_1101',type=str)
    parser.add_argument('--root', default='/root/data1/wqf/FBUSTS-3.0/DataSet/CC-New/', type=str)
    parser.add_argument('--num_workers', default = 4, type = int)
    parser.add_argument('--batch_size',default = 8,type = int)
    parser.add_argument('--epochs',default =1000,type = int)
    parser.add_argument('--lr',default = 0.0001,type = float)
    
    # For Another Target
    parser.add_argument('--root_SP', default='/root/data1/wqf/FBUSTS-3.0/DataSet/CSP-New/', type=str)
    
    # Pretrained
    parser.add_argument('--pretrained', default=None, type=str, help='Path to the pretrained model to continue training')
    parser.add_argument('--pretrained_optimizer', default=None, type=str, help='Path to the pretrained model to continue training')
    parser.add_argument('--start_epoch', default=0, type=int, help='Epoch to start training from')

    # Save Root
    base_dir = '/root/data1/wqf/FBUSTS-3.0/Results/CC/'
    Model ='Shape_Unet_1101'
    parser.add_argument('--statistics_save_dir', default=os.path.join(base_dir, Model, 'Statistics'), type=str)
    parser.add_argument('--image_save_dir', default=os.path.join(base_dir, Model, 'Image'), type=str)
    parser.add_argument('--model_save_dir', default=os.path.join(base_dir, Model, 'Model'), type=str)

    args=parser.parse_args()
    runTraining(args)