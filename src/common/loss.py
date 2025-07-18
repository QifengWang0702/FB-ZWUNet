import os
import numpy as np

import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn.functional as F
import cv2
from sklearn.cluster import KMeans
from scipy.ndimage import distance_transform_edt

import dill
import argparse
from data import medicalDataLoader

############### Old Loss Functions ################

#Loss 4 ZAM_WAM_Attention ResNet
def compute_loss_ResNet(net_output, Segmentation_onehot, BCE_loss, mseLoss):
    semVector_1_1, semVector_2_1, semVector_1_2, semVector_2_2, \
    semVector_1_3, semVector_2_3, semVector_1_4, semVector_2_4, \
    inp_enc0, inp_enc1, inp_enc2, inp_enc3, inp_enc4, inp_enc5, inp_enc6, inp_enc7, \
    out_enc0, out_enc1, out_enc2, out_enc3, out_enc4, out_enc5, out_enc6, out_enc7, \
    outputs0, outputs1, outputs2, outputs3, outputs0_2, outputs1_2, outputs2_2, outputs3_2 = net_output

    loss0 = BCE_loss(outputs0, Segmentation_onehot)
    loss1 = BCE_loss(outputs1, Segmentation_onehot)
    loss2 = BCE_loss(outputs2, Segmentation_onehot)
    loss3 = BCE_loss(outputs3, Segmentation_onehot)
    loss0_2 = BCE_loss(outputs0_2, Segmentation_onehot)
    loss1_2 = BCE_loss(outputs1_2, Segmentation_onehot)
    loss2_2 = BCE_loss(outputs2_2, Segmentation_onehot)
    loss3_2 = BCE_loss(outputs3_2, Segmentation_onehot)


    lossSemantic1 = mseLoss(semVector_1_1, semVector_2_1)
    lossSemantic2 = mseLoss(semVector_1_2, semVector_2_2)
    lossSemantic3 = mseLoss(semVector_1_3, semVector_2_3)
    lossSemantic4 = mseLoss(semVector_1_4, semVector_2_4)

    lossRec0 = mseLoss(inp_enc0, out_enc0)
    lossRec1 = mseLoss(inp_enc1, out_enc1)
    lossRec2 = mseLoss(inp_enc2, out_enc2)
    lossRec3 = mseLoss(inp_enc3, out_enc3)
    lossRec4 = mseLoss(inp_enc4, out_enc4)
    lossRec5 = mseLoss(inp_enc5, out_enc5)
    lossRec6 = mseLoss(inp_enc6, out_enc6)
    lossRec7 = mseLoss(inp_enc7, out_enc7)

    lossG = (loss0 + loss1 + loss2 + loss3 + loss0_2 + loss1_2 + loss2_2 + loss3_2) \
            + 0.25 * (lossSemantic1 + lossSemantic2 + lossSemantic3 + lossSemantic4) \
            + 0.1 * (lossRec0 + lossRec1 + lossRec2 + lossRec3 + lossRec4 + lossRec5 + lossRec6 + lossRec7)

    return lossG

#Loss 4 ZAM_WAM_Attention UNet
def compute_loss_Unet(net_output, Segmentation_onehot, CE_loss):
    enc1, enc2, enc3, middle, dec3, dec2, dec1, end = net_output
    #end = net_output

    loss_end = CE_loss(end, Segmentation_onehot)

    lossG = loss_end

    return lossG

#Loss 4 ZAM_WAM_Attention UNet_
def get_matching_targets(target, sizes=[(128, 128, 16), (64, 64, 32), (32, 32, 64), (16, 16, 64)]):
    matching_targets = []
    for size in sizes:
        h, w, c = size
        matching_target = F.interpolate(target, size=(h, w), mode='nearest')
        if c != target.size(1):
            matching_target = matching_target.repeat(1, c // target.size(1), 1, 1)
        matching_targets.append(matching_target)
    return matching_targets

def compute_loss_Unet_(net_output, Segmentation_onehot, BCE_loss, mseLoss, DiceLoss, net):
    enc1, enc2, enc3, \
    middle, \
    dec3, dec2, dec1, \
    end = net_output

    # Normalize the weights so that they sum to 1
    W = F.softmax(net.raw_W, dim=0)

    target1, target2, target3, target_middle = get_matching_targets(Segmentation_onehot)

    loss_enc1 = BCE_loss(enc1, target1)
    loss_enc2 = BCE_loss(enc2, target2)
    loss_enc3 = BCE_loss(enc3, target3)
    
    loss_middle = mseLoss(middle, target_middle)  # Assuming middle and target3 have the same size
    
    loss_dec3 = BCE_loss(dec3, target3)
    loss_dec2 = BCE_loss(dec2, target2)
    loss_dec1 = BCE_loss(dec1, target1)
    
    loss_end = DiceLoss(end, Segmentation_onehot)

    # Regularization terms
    l1_reg = torch.tensor(0., requires_grad=True)
    l2_reg = torch.tensor(0., requires_grad=True)
    for param in net.parameters():
        l1_reg = l1_reg + torch.norm(param, 1)
        l2_reg = l2_reg + torch.norm(param, 2)
    
    lambda_l1 = 0.00001
    lambda_l2 = 0.00001
    # print(W[0])
    # print(W[1])
    # print(W[2])
    # print(W[3])

    lossG =   W[0] * (loss_enc1 + loss_enc2 + loss_dec2+ loss_dec1) \
            + W[1] * loss_middle \
            + W[2] * (loss_enc3 + loss_dec3) \
            + W[3] * loss_end \
            + lambda_l1 * l1_reg \
            + lambda_l2 * l2_reg 

    return lossG

#Loss 4 CC&CV UNet
def cluster_regions(tensor):
    """
    Perform K-means clustering on a tensor to identify two regions and
    order them based on their centroids' vertical position.

    Args:
    - tensor (torch.Tensor): A 2D tensor representing the image.

    Returns:
    - torch.Tensor: A tensor with the same shape as the input but with values 0, 1, and 2
                    indicating the background and the two clusters, respectively.
    """
    # Reshape tensor into a 2D array where each row is a pixel and each column is a coordinate (i.e., (x,y)).
    coords = torch.nonzero(tensor).numpy()
    
    # If there aren't enough coordinates for clustering, return the input tensor.
    if len(coords) < 2:
        return tensor

    # Apply K-means clustering with 2 clusters.
    kmeans = KMeans(n_clusters=2, random_state=0).fit(coords)
    
    # Identify centroids of clusters
    centroids = kmeans.cluster_centers_

    # Determine which cluster is on top
    top_cluster = np.argmin(centroids[:, 0])  # index of the cluster with the smaller x-coordinate (higher up in the image)

    # Assign labels from clustering to the pixels in the tensor.
    clustered_tensor = torch.zeros_like(tensor)
    for coord, label in zip(coords, kmeans.labels_):
        # If the cluster is the top cluster, assign it label 1, otherwise assign it label 2
        clustered_tensor[coord[0], coord[1]] = 1 if label == top_cluster else 2

    return clustered_tensor

def compute_centroid(tensor, label):
    """
    Compute the centroid of a labeled region in a tensor.

    Args:
    - tensor (torch.Tensor): A 2D tensor representing the image.
    - label (int): Label of the region to compute the centroid for.

    Returns:
    - tuple: (x, y) coordinates of the centroid.
    """
    coords = torch.nonzero(tensor == label).float()  # Get coordinates of all pixels with the given label
    centroid = coords.mean(dim=0)  # Compute the mean of the coordinates
    return centroid

def position_loss(predicted, target):
    """
    Compute the position loss based on the centroids of two labeled regions.

    Args:
    - predicted (torch.Tensor): Predicted 2D tensor.
    - target (torch.Tensor): Ground truth 2D tensor.

    Returns:
    - torch.Tensor: Scalar tensor representing the position loss.
    """
    # First, perform clustering to label the regions in the tensors
    predicted_clustered = cluster_regions(predicted)
    target_clustered = cluster_regions(target)
    
    # Compute centroids for both regions in both predicted and target tensors
    predicted_centroid1 = compute_centroid(predicted_clustered, 1)
    predicted_centroid2 = compute_centroid(predicted_clustered, 2)
    target_centroid1 = compute_centroid(target_clustered, 1)
    target_centroid2 = compute_centroid(target_clustered, 2)

    # Compute the relative positions
    predicted_relative = predicted_centroid2 - predicted_centroid1
    target_relative = target_centroid2 - target_centroid1

    # Compute the loss as the mean squared error of the relative positions
    loss = F.mse_loss(predicted_relative, target_relative)

    return loss

def morphological_operations(tensor, kernel_size=3):
    """
    Apply morphological operations (dilation and erosion) on a tensor.

    Args:
    - tensor (torch.Tensor): A 2D tensor representing the image.
    - kernel_size (int): Size of the kernel used for morphological operations.

    Returns:
    - tuple: Eroded and dilated tensors.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    tensor_np = tensor.cpu().detach().numpy()
    if len(tensor_np.shape) == 4:  # If it's a batch of 2D images
        eroded = np.array([cv2.erode(img[0], kernel, iterations=1) for img in tensor_np])
        dilated = np.array([cv2.dilate(img[0], kernel, iterations=1) for img in tensor_np])
    else:  # If it's a single 2D image
        eroded = cv2.erode(tensor_np, kernel, iterations=1)
        dilated = cv2.dilate(tensor_np, kernel, iterations=1)
    return torch.from_numpy(eroded), torch.from_numpy(dilated)

def morphological_loss(predicted, target, kernel_size=3):
    """
    Compute morphological loss based on the differences in dilation and erosion.

    Args:
    - predicted (torch.Tensor): Predicted 2D tensor.
    - target (torch.Tensor): Ground truth 2D tensor.
    - kernel_size (int): Size of the kernel used for morphological operations.

    Returns:
    - torch.Tensor: Scalar tensor representing the morphological loss.
    """
    predicted_eroded, predicted_dilated = morphological_operations(predicted, kernel_size)
    target_eroded, target_dilated = morphological_operations(target, kernel_size)

    # Compute the loss as the mean squared error of the differences in dilation and erosion
    erosion_loss = F.mse_loss(predicted_eroded, target_eroded)
    dilation_loss = F.mse_loss(predicted_dilated, target_dilated)

    return erosion_loss + dilation_loss

def compute_distance_map_torch(binary_mask):
    """
    Compute distance map for a binary mask.

    Args:
    - binary_mask (torch.Tensor): A tensor representing the binary image. Can be a single 2D image or a batch of 2D images.

    Returns:
    - torch.Tensor: Distance map tensor.
    """
    binary_mask_np = binary_mask.cpu().detach().numpy()
    
    if len(binary_mask_np.shape) == 4:  # If it's a batch of 2D images
        distance_map = np.array([distance_transform_edt(1 - img[0]) for img in binary_mask_np])
    else:  # If it's a single 2D image
        distance_map = distance_transform_edt(1 - binary_mask_np)
    
    return torch.tensor(distance_map, dtype=torch.float32).to(binary_mask.device)

def distance_map_loss(pred_edge, true_edge):
    # 根据预测和真实的边缘结果计算距离图
    pred_distance_map = compute_distance_map_torch(pred_edge)
    true_distance_map = compute_distance_map_torch(true_edge)

    # 计算两个距离图之间的MSE损失
    loss = F.mse_loss(pred_distance_map, true_distance_map)
    return loss

def compute_loss_CC_CV_Unet(net_output, Segmentation_onehot, Segmentation_onehot_Canny, CE_loss, net):
    segmentation, edge = net_output
    
    loss_segmentation = CE_loss(segmentation, Segmentation_onehot)

    loss_position = position_loss(segmentation, Segmentation_onehot)

    loss_morphological = morphological_loss(segmentation, Segmentation_onehot)

    loss_distance_map = distance_map_loss(edge, Segmentation_onehot_Canny)
    
    lossG = loss_segmentation + loss_position + loss_morphological + loss_distance_map

    return lossG

def compute_loss_CC_SP_Unet(net_output, Segmentation_onehot_CC, Segmentation_onehot_Canny_CC,  Segmentation_onehot_SP, Segmentation_onehot_Canny_SP, CE_loss, BCE_loss, DiceLoss, net):
    
    segmentation_branch1, edge_branch1, segmentation_branch2, edge_branch2, LCCM = net_output
    #CC
    loss_segmentation_CC = CE_loss(segmentation_branch1, Segmentation_onehot_CC)
    
    loss_edge_bce_CC = BCE_loss(edge_branch1, Segmentation_onehot_Canny_CC)
    loss_edge_dice_CC = DiceLoss(edge_branch1, Segmentation_onehot_Canny_CC)
    alpha = 0.5
    beta = 0.5
    loss_edge_CC = alpha * loss_edge_bce_CC + beta * loss_edge_dice_CC
    
    #SP
    loss_segmentation_SP = CE_loss(segmentation_branch2, Segmentation_onehot_SP)

    loss_edge_bce_SP = BCE_loss(edge_branch2, Segmentation_onehot_Canny_SP)
    loss_edge_dice_SP = DiceLoss(edge_branch2, Segmentation_onehot_Canny_SP)
    alpha = 0.5
    beta = 0.5
    loss_edge_SP = alpha * loss_edge_bce_SP + beta * loss_edge_dice_SP

    #CCS&SP
    loss_ccm = LCCM
    
    lossG = loss_segmentation_CC + loss_edge_CC + loss_segmentation_SP + loss_edge_SP + loss_ccm

    return lossG

#Loss 4 ZAM_WAM_Unet_4Layers
def compute_loss_Unet_4Layers(net_output, Segmentation_onehot, CE_loss, DiceLoss):

    segmentation = net_output

    loss_ce = CE_loss(segmentation, Segmentation_onehot)
    loss_dice = DiceLoss(segmentation, Segmentation_onehot)
    
    alpha = 0.5
    beta = 0.5
    loss_end = alpha * loss_ce + beta * loss_dice

    lossG = loss_end

    return lossG


############### New Loss Functions ################
# DiceLoss
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        # Assuming output and target are of shape [batch_size, n_classes, height, width]
        output = torch.sigmoid(output)
        intersection = (output * target).sum(dim=(2, 3))
        union = output.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        dice = 2 * intersection / (union + self.eps)
        
        return (1-dice).mean()

# Loss for UNet
def Unet_Loss(region, region_Mask_onehot, BCE_loss, Dice_Loss, net):
    
    '''region: targer region prediction result
       edge: targer edge prediction result
       region_Mask: label of targer region
       edge_Mask: label of targer edge
    '''
    alpha = 0.5
    beta = 0.5
    loss_region_bce = BCE_loss(region, region_Mask_onehot)
    loss_region_dice = Dice_Loss(region, region_Mask_onehot)
    loss_region = alpha * loss_region_bce + beta * loss_region_dice
    
    loss_edge = loss_region

    Loss = loss_region

    return Loss, loss_region, loss_edge, loss_region_dice

# Loss for Shape UNet
def Shape_Unet_Loss(region, edge, region_Mask_onehot, edge_Mask_onehot, BCE_loss, Dice_Loss, net):

    '''region: targer region prediction result
       edge: targer edge prediction result
       region_Mask: label of targer region
       edge_Mask: label of targer edge
    '''
    alpha = 0.5
    beta = 0.5

    loss_region_bce = BCE_loss(region, region_Mask_onehot)
    loss_region_dice = Dice_Loss(region, region_Mask_onehot)
    loss_region = alpha * loss_region_bce + beta * loss_region_dice

    loss_edge_bce = BCE_loss(edge,  edge_Mask_onehot)
    loss_edge_dice = Dice_Loss(edge,  edge_Mask_onehot)
    loss_edge = alpha * loss_edge_bce + beta * loss_edge_dice
    
    Loss = loss_region + loss_edge

    return Loss, loss_region, loss_edge, loss_region_dice

# FocalLoss
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.8, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return F_loss.mean()

# TverskyLoss
class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, inputs, targets):
        smooth = 1.0

        # 将输入压缩成二值图像
        inputs = torch.sigmoid(inputs)

        # 计算真阳性、假阳性和假阴性
        TP = (inputs * targets).sum()
        FP = ((1 - targets) * inputs).sum()
        FN = (targets * (1 - inputs)).sum()

        Tversky = (TP + smooth) / (TP + self.alpha * FP + self.beta * FN + smooth)
        return 1 - Tversky

# Loss for Channel UNet
def Channel_Unet_Loss(region, edge, region_Mask_onehot, edge_Mask_onehot, BCE_loss, Dice_Loss, Focal_Loss, Tversky_Loss, net):

    '''region: targer region prediction result
       edge: targer edge prediction result
       region_Mask: label of targer region
       edge_Mask: label of targer edge
    '''
    alpha = 0.5
    beta = 0.5

    loss_region_focal = Focal_Loss(region, region_Mask_onehot)
    loss_region_dice = Dice_Loss(region, region_Mask_onehot)
    loss_region = alpha * loss_region_focal + beta * loss_region_dice

    loss_edge_tversky = Tversky_Loss(edge, edge_Mask_onehot)
    loss_edge_bce = BCE_loss(edge, edge_Mask_onehot)
    loss_edge = alpha * loss_edge_tversky + beta * loss_edge_bce

    Loss = loss_region + loss_edge

    return Loss, loss_region, loss_edge, loss_region_dice

# Loss for Week UNet
def Week_Unet_Loss(region, week, region_Mask_onehot, week_gt, BCE_loss, Dice_Loss, Mse_Loss, net):

    '''region: targer region prediction result
       week: week prediction result
       region_Mask: label of targer region
       week_gt: real week number
    '''
    
    alpha = 0.5
    beta = 0.5
    gamma = 0.002

    # Region
    loss_region_bce = BCE_loss(region, region_Mask_onehot)
    loss_region_dice = Dice_Loss(region, region_Mask_onehot)
    loss_region = alpha * loss_region_bce + beta * loss_region_dice

    # Week
    loss_week = gamma * Mse_Loss(week, week_gt.float())

    # Loss
    Loss = loss_region + loss_week

    return Loss, loss_region, loss_week, loss_region_dice
