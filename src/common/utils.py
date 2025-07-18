import os
import numpy as np
import scipy.io as sio
import pdb
import time
from os.path import isfile, join

import nibabel as nib
from PIL import Image
from medpy.metric.binary import dc,hd
import skimage.transform as skiTransf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

from .progressBar import printProgressBar

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

def load_nii(imageFileName, printFileNames):
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()

    return (imageData, img_proxy)

def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)

#### Loss for 1 channel #### 
# Dice Loss
class computeDiceLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(computeDiceLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        # Assuming output and target are of shape [batch_size, n_classes, height, width]
        output = torch.sigmoid(output)

        # Apply binary thresholding
        output = (output > 0.5).float()
        
        intersection = (output * target).sum(dim=(2, 3))
        union = output.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
        
        dice = 2 * intersection / (union + self.eps)
        
        # Average over batch and classes
        return dice.mean()

# Jaccard Loss
class computeJaccardLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(computeJaccardLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        # Apply binary thresholding
        output = (output > 0.5).float()
        intersection = (output * target).sum(dim=(2, 3))
        union = output.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection
        jaccard = intersection / (union + self.eps)
        return jaccard.mean()

#Precision Loss
class computePrecisionLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(computePrecisionLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        # Apply binary thresholding
        output = (output > 0.5).float()
        TP = (output * target).sum(dim=(2, 3))
        FP = (output * (1 - target)).sum(dim=(2, 3))
        precision = TP / (TP + FP + self.eps)
        return precision.mean()

#Recall Loss
class computeRecallLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(computeRecallLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        # Apply binary thresholding
        output = (output > 0.5).float()
        TP = (output * target).sum(dim=(2, 3))
        FN = ((1 - output) * target).sum(dim=(2, 3))
        recall = TP / (TP + FN + self.eps)
        return recall.mean()

#Specificity Loss
class computeSpecificityLoss(nn.Module):
    def __init__(self, eps=1e-7):
        super(computeSpecificityLoss, self).__init__()
        self.eps = eps

    def forward(self, output, target):
        output = torch.sigmoid(output)
        # Apply binary thresholding
        output = (output > 0.5).float()
        TN = ((1 - output) * (1 - target)).sum(dim=(2, 3))
        FP = (output * (1 - target)).sum(dim=(2, 3))
        specificity = TN / (TN + FP + self.eps)
        return specificity.mean()

#Accuracy Loss
class computeAccuracyLoss(nn.Module):
    def __init__(self):
        super(computeAccuracyLoss, self).__init__()

    def forward(self, output, target):
        output = torch.sigmoid(output)
        # Apply binary thresholding
        output = (output > 0.5).float()
        correct = ((output > 0.5) == target).sum(dim=(2, 3))
        total = output.numel()
        accuracy = correct / total
        return accuracy.mean()

#Hausdorff Loss
class computeHausdorffLoss(nn.Module):
    def __init__(self):
        super(computeHausdorffLoss, self).__init__()

    def forward(self, output, target):
        # Assume output and target are binary: 0 or 1
        output = (output > 0.5).float()
        #target = (target > 0.5).float()

        def hausdorff_distance(mask1, mask2):
            if mask1.size(0) == 0 or mask2.size(0) == 0:
                return torch.tensor(float('inf')).to(mask1.device)  # or other value to handle this case
            # Compute the pixel-wise Euclidean distance between mask1 and mask2
            dist1 = torch.nn.functional.pairwise_distance(mask1.unsqueeze(1), mask2.unsqueeze(0), p=2.0)
            dist2 = torch.nn.functional.pairwise_distance(mask2.unsqueeze(1), mask1.unsqueeze(0), p=2.0)
            # Compute the Hausdorff distance
            return max(dist1.max(), dist2.max())

        hausdorff_loss = 0.0
        for i in range(output.shape[0]):
            coords1 = torch.stack((output[i].nonzero(as_tuple=False)[:, 0], output[i].nonzero(as_tuple=False)[:, 1]), dim=1)
            coords2 = torch.stack((target[i].nonzero(as_tuple=False)[:, 0], target[i].nonzero(as_tuple=False)[:, 1]), dim=1)
            hausdorff_loss += hausdorff_distance(coords1, coords2)

        return hausdorff_loss / output.shape[0]


#### Loss for 2 channels #### 
# Dice Similarity Coefficient (DSC) / F1 Score
class computeDice(nn.Module):
    def __init__(self):
        #super(computeDiceOneHot, self).__init__()
        super().__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)
        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def forward(self, pred, GT):
        # Ensure that pred and GT have two channels
        assert pred.size(1) == 2, "Prediction must have 2 channels"
        assert GT.size(1) == 2, "Ground Truth must have 2 channels"

        batchsize = GT.size(0)
        DiceBackground = to_var(torch.zeros(batchsize, 2))
        DiceTarget = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceBackground[i, 0] = self.dice(pred[i, 0], GT[i, 0])
            DiceTarget[i, 0] = self.dice(pred[i, 1], GT[i, 1])


        return DiceBackground, DiceTarget

# Jaccard Index / Intersection over Union (IoU)
class computeJaccard(nn.Module):
    def __init__(self):
        super().__init__()

    def jaccard(self, input, target):
        inter = (input * target).float().sum()
        union = input.sum() + target.sum() - inter
        return inter / (union + 1e-8)

    def forward(self, pred, GT):
        assert pred.size(1) == 2, "Prediction must have 2 channels"
        assert GT.size(1) == 2, "Ground Truth must have 2 channels"

        batchsize = GT.size(0)
        JaccardBackground = to_var(torch.zeros(batchsize, 2))
        JaccardTarget = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            JaccardBackground[i, 0] = self.jaccard(pred[i, 0], GT[i, 0])
            JaccardTarget[i, 0] = self.jaccard(pred[i, 1], GT[i, 1])

        return JaccardBackground, JaccardTarget

# Precision (P)
class computePrecision(nn.Module):
    def __init__(self):
        super().__init__()

    def precision(self, input, target):
        TP = (input * target).float().sum()
        FP = (input * (1 - target)).float().sum()
        return TP / (TP + FP + 1e-8)

    def forward(self, pred, GT):
        assert pred.size(1) == 2, "Prediction must have 2 channels"
        assert GT.size(1) == 2, "Ground Truth must have 2 channels"

        batchsize = GT.size(0)
        PrecisionBackground = to_var(torch.zeros(batchsize, 2))
        PrecisionTarget = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            PrecisionBackground[i, 0] = self.precision(pred[i, 0], GT[i, 0])
            PrecisionTarget[i, 0] = self.precision(pred[i, 1], GT[i, 1])

        return PrecisionBackground, PrecisionTarget

# Recall (R) / Sensitivity / True Positive Rate
class computeRecall(nn.Module):
    def __init__(self):
        super().__init__()

    def recall(self, input, target):
        TP = (input * target).float().sum()
        FN = ((1 - input) * target).float().sum()
        return TP / (TP + FN + 1e-8)

    def forward(self, pred, GT):
        assert pred.size(1) == 2, "Prediction must have 2 channels"
        assert GT.size(1) == 2, "Ground Truth must have 2 channels"

        batchsize = GT.size(0)
        RecallBackground = to_var(torch.zeros(batchsize, 2))
        RecallTarget = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            RecallBackground[i, 0] = self.recall(pred[i, 0], GT[i, 0])
            RecallTarget[i, 0] = self.recall(pred[i, 1], GT[i, 1])

        return RecallBackground, RecallTarget

# Specificity / True Negative Rate
class computeSpecificity(nn.Module):
    def __init__(self):
        super().__init__()

    def specificity(self, input, target):
        TN = ((1 - input) * (1 - target)).float().sum()
        FP = (input * (1 - target)).float().sum()
        return TN / (TN + FP + 1e-8)

    def forward(self, pred, GT):
        assert pred.size(1) == 2, "Prediction must have 2 channels"
        assert GT.size(1) == 2, "Ground Truth must have 2 channels"

        batchsize = GT.size(0)
        SpecificityBackground = to_var(torch.zeros(batchsize, 2))
        SpecificityTarget = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            SpecificityBackground[i, 0] = self.specificity(pred[i, 0], GT[i, 0])
            SpecificityTarget[i, 0] = self.specificity(pred[i, 1], GT[i, 1])

        return SpecificityBackground, SpecificityTarget

# Accuracy (ACC)
class computeAccuracy(nn.Module):
    def __init__(self):
        super().__init__()

    def accuracy(self, input, target):
        TP = (input * target).float().sum()
        TN = ((1 - input) * (1 - target)).float().sum()
        total = input.numel()
        return (TP + TN) / (total + 1e-8)

    def forward(self, pred, GT):
        assert pred.size(1) == 2, "Prediction must have 2 channels"
        assert GT.size(1) == 2, "Ground Truth must have 2 channels"

        batchsize = GT.size(0)
        AccuracyBackground = to_var(torch.zeros(batchsize, 2))
        AccuracyTarget = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            AccuracyBackground[i, 0] = self.accuracy(pred[i, 0], GT[i, 0])
            AccuracyTarget[i, 0] = self.accuracy(pred[i, 1], GT[i, 1])

        return AccuracyBackground, AccuracyTarget

def MetricsToMetric(Metrics):
    sums = Metrics.sum(dim=0)
    return sums[0] / Metrics.size()[0]

# Hausdorff Distance (豪斯多夫距离)
class computeHausdorff(nn.Module):
    def __init__(self):
        super().__init__()

    def hausdorff_distance(self, input, target):
        # Convert tensors to binary masks
        input_binary = (input > 0.5).float()
        target_binary = (target > 0.5).float()

        # Calculate pairwise distances between the two point sets
        distances = self.pairwise_distances(input_binary.view(input_binary.size(0), -1), 
                                            target_binary.view(target_binary.size(0), -1))

        # Calculate directed Hausdorff distance
        h1 = torch.max(distances, dim=1)[0]
        h2 = torch.max(distances, dim=0)[0]
        
        # Return the maximum of the two directed distances
        return torch.max(h1, h2)

    def pairwise_distances(self, x, y):
        # Calculate pairwise distances between two sets of points
        distances = torch.cdist(x, y, p=2)
        return distances

    def forward(self, pred, GT):
        return self.hausdorff_distance(pred, GT)

def HausdorffsToHausdorff(Hausdorffs):
    return Hausdorffs.mean()

# Mean Absolute Distance (MAD)
class computeMAD(nn.Module):
    def __init__(self):
        super().__init__()

    def mean_absolute_distance(self, input, target):
        # Convert tensors to binary masks
        input_binary = (input > 0.5).float()
        target_binary = (target > 0.5).float()

        # Calculate pairwise distances between the two point sets
        distances = self.pairwise_distances(input_binary.view(input_binary.size(0), -1), 
                                            target_binary.view(target_binary.size(0), -1))

        # Calculate mean absolute distance
        mad = distances.mean()
        return mad

    def pairwise_distances(self, x, y):
        # Calculate pairwise distances between two sets of points
        distances = torch.cdist(x, y, p=2)
        return distances

    def forward(self, pred, GT):
        return self.mean_absolute_distance(pred, GT)

# Surface Distance Measures
class computeSurfaceDistance(nn.Module):
    def __init__(self):
        super().__init__()

    def surface_distance(self, input, target):
        # Convert tensors to binary masks
        input_binary = (input > 0.5).float()
        target_binary = (target > 0.5).float()

        # Calculate pairwise distances between the two point sets
        distances = self.pairwise_distances(input_binary.view(input_binary.size(0), -1), 
                                            target_binary.view(target_binary.size(0), -1))

        # Calculate surface distance
        sd = distances.mean(dim=1)
        return sd

    def pairwise_distances(self, x, y):
        # Calculate pairwise distances between two sets of points
        distances = torch.cdist(x, y, p=2)
        return distances

    def forward(self, pred, GT):
        return self.surface_distance(pred, GT)

def SurfaceDistancesToSurfaceDistance(SurfaceDistances):
    return SurfaceDistances.mean()

def getSingleImage(pred_y):
    return pred_y.argmax(dim=1, keepdim=True)

def getSingleImage_(pred):
    # input is a 2-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    num_classes = 2
    Val = to_var(torch.zeros(num_classes))

    # Assuming 0 is for background and 1 is for target region
    Val[1] = 1  # You can set this to any other value if you want a different grayscale value for the target region
    
    x = predToSegmentation(pred)
   
    out = x * Val.view(1, num_classes, 1, 1)

    return out.sum(dim=1, keepdim=True)

def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()

def getOneHotSegmentation(tensor):
    """
    Convert tensor to one-hot format for binary labels.
    tensor: input tensor of shape [batch_size, 1, height, width] or [batch_size, height, width]
    """
    
    if len(tensor.shape) == 4:  # Shape is [batch_size, 1, height, width]
        n, _, h, w = tensor.size()
        tensor = tensor.squeeze(1)
    else:  # Shape is [batch_size, height, width]
        n, h, w = tensor.size()

    one_hot = torch.zeros(n, 2, h, w).to(tensor.device)

    one_hot[:, 0, :, :] = (tensor == 0)  # Background
    one_hot[:, 1, :, :] = (tensor == 1)  # Target

    return one_hot.float()

def getTargetSegmentation(batch):
    # 输入是一个单通道的值介于0和1之间的张量
    # 输出是一个单通道的离散值：0或1

    threshold = 0.5  # 设置一个阈值，大于此值的认为是目标，小于或等于此值的认为是背景

    return (batch > threshold).long().squeeze()

def saveImages_for2D(net, img_batch, epoch, image_save_dir):
    # 设置保存路径
    #path = 'Results/Images/CC/' + modelName + '/Val_Res/' #+ str(epoch)
    path = os.path.join(image_save_dir, '/Epoch_' + str(epoch))
    if not os.path.exists(path):
        os.makedirs(path)

    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax(dim=1)  # 注意dim=1，这是因为我们希望在类别维度上进行softmax

    for i, data in enumerate(img_batch):
        #image, labels, img_names = data
        image, labels, labels_canny, img_names = data

        US = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(US)
        pred_y = softMax(segmentation_prediction)
        
        segmentation = getSingleImage_(pred_y)  # 假设这个函数从softmax输出中获取2D分割图像

        # 提取图像名，用于保存
        img_name = img_names[0].split('/')[-1]  # 获取文件名
        torchvision.utils.save_image(segmentation.data, os.path.join(path, img_name))

    printProgressBar(total, total, done="2D Images saved !")

def inferenceAllMetrics_1(net, img_batch):
    total = len(img_batch)

    metrics_results = {
        "Dice": torch.zeros(total),
        "Jaccard": torch.zeros(total),
        "Precision": torch.zeros(total),
        "Recall": torch.zeros(total),
        "Specificity": torch.zeros(total),
        "Accuracy": torch.zeros(total),
        "Hausdorff": torch.zeros(total),
    }

    net.eval()
    img_names_ALL = []

    dice = computeDiceLoss().cuda()
    jaccard = computeJaccardLoss().cuda()
    precision = computePrecisionLoss().cuda()
    recall = computeRecallLoss().cuda()
    specificity = computeSpecificityLoss().cuda()
    accuracy = computeAccuracyLoss().cuda()
    hausdorff = computeHausdorffLoss().cuda()
    sigmoid = nn.Sigmoid().cuda()

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Validation Inference] Getting segmentations...", length=30)
        #image, labels, img_names = data
        image, labels, labels_canny, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        US = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(US)[0]
        Segmentation_planes = Segmentation
        segmentation_prediction_ones = segmentation_prediction

        dice_results = dice(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Dice"][i] = dice_results
        
        jaccard_results = jaccard(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Jaccard"][i] = jaccard_results

        precision_results = precision(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Precision"][i] = precision_results

        recall_results = recall(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Recall"][i] = recall_results

        specificity_results = specificity(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Specificity"][i] = specificity_results

        accuracy_results = accuracy(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Accuracy"][i] = accuracy_results
        
        hausdorff_results = hausdorff(segmentation_prediction_ones, Segmentation_planes)
        metrics_results["Hausdorff"][i] = hausdorff_results

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    return metrics_results

def inferenceAllMetrics(net, img_batch):
    total = len(img_batch)

    metrics_results = {
        "Dice": torch.zeros(total),
        "Jaccard": torch.zeros(total),
        "Precision": torch.zeros(total),
        "Recall": torch.zeros(total),
        "Specificity": torch.zeros(total),
        "Accuracy": torch.zeros(total),
        "Hausdorff": torch.zeros(total),
    }

    net.eval()
    img_names_ALL = []

    dice = computeDice().cuda()
    jaccard = computeJaccard().cuda()
    precision = computePrecision().cuda()
    recall = computeRecall().cuda()
    specificity = computeSpecificity().cuda()
    accuracy = computeAccuracy().cuda()
    hausdorff = computeHausdorff().cuda()
    sigmoid = nn.Sigmoid().cuda()

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Validation Inference] Getting segmentations...", length=30)
        
        images, labels, labels_canny, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        Image = to_var(images)
        region_Mask = to_var(labels)
        region = net(Image)[0]
        
        region_ = predToSegmentation(sigmoid(region))
        region_Mask_ = getOneHotSegmentation(region_Mask)

        _, dice_results = dice(region_, region_Mask_)
        metrics_results["Dice"][i] = MetricsToMetric(dice_results)
        
        _, jaccard_results = jaccard(region_, region_Mask_)
        metrics_results["Jaccard"][i] = MetricsToMetric(jaccard_results)

        _, precision_results = precision(region_, region_Mask_)
        metrics_results["Precision"][i] = MetricsToMetric(precision_results)

        _, recall_results = recall(region_, region_Mask_)
        metrics_results["Recall"][i] = MetricsToMetric(recall_results)

        _, specificity_results = specificity(region_, region_Mask_)
        metrics_results["Specificity"][i] = MetricsToMetric(specificity_results)

        _, accuracy_results = accuracy(region_, region_Mask_)
        metrics_results["Accuracy"][i] = MetricsToMetric(accuracy_results)
        
        hausdorff_results = hausdorff(region_, region_Mask_)
        metrics_results["Hausdorff"][i] = HausdorffsToHausdorff(hausdorff_results)

    printProgressBar(total, total, done="[Validation Inference] Segmentation Done !")

    return metrics_results

def inferenceAllMetrics_week(net, img_batch, args):
    total = len(img_batch)

    metrics_results = {
        "Dice": torch.zeros(total),
        "Jaccard": torch.zeros(total),
        "Precision": torch.zeros(total),
        "Recall": torch.zeros(total),
        "Specificity": torch.zeros(total),
        "Accuracy": torch.zeros(total),
        "Hausdorff": torch.zeros(total),
    }

    net.eval()
    img_names_ALL = []

    dice = computeDice().cuda()
    jaccard = computeJaccard().cuda()
    precision = computePrecision().cuda()
    recall = computeRecall().cuda()
    specificity = computeSpecificity().cuda()
    accuracy = computeAccuracy().cuda()
    hausdorff = computeHausdorff().cuda()
    sigmoid = nn.Sigmoid().cuda()

    for i, data in enumerate(img_batch):
        printProgressBar(i, total, prefix="[Validation Inference] Getting segmentations...", length=30)
        
        images, labels, labels_canny, img_names, week = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        Image = to_var(images)
        region_Mask = to_var(labels)
        if args.model == 'Week_Improved_ZAM_WAM_Unet'or args.model == 'Week_Improved_AttentionUnet' or args.model == 'Week_Feature_AttentionUnet':
            region = net(Image, week)[0]
        else:
            region = net(Image)[0]
        
        region_ = predToSegmentation(sigmoid(region))
        region_Mask_ = getOneHotSegmentation(region_Mask)

        _, dice_results = dice(region_, region_Mask_)
        metrics_results["Dice"][i] = MetricsToMetric(dice_results)
        
        _, jaccard_results = jaccard(region_, region_Mask_)
        metrics_results["Jaccard"][i] = MetricsToMetric(jaccard_results)

        _, precision_results = precision(region_, region_Mask_)
        metrics_results["Precision"][i] = MetricsToMetric(precision_results)

        _, recall_results = recall(region_, region_Mask_)
        metrics_results["Recall"][i] = MetricsToMetric(recall_results)

        _, specificity_results = specificity(region_, region_Mask_)
        metrics_results["Specificity"][i] = MetricsToMetric(specificity_results)

        _, accuracy_results = accuracy(region_, region_Mask_)
        metrics_results["Accuracy"][i] = MetricsToMetric(accuracy_results)
        
        hausdorff_results = hausdorff(region_, region_Mask_)
        metrics_results["Hausdorff"][i] = HausdorffsToHausdorff(hausdorff_results)

    printProgressBar(total, total, done="[Validation Inference] Segmentation Done !")

    return metrics_results

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()