from functools import reduce

import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
from .attention import (
    PAM_Module,
    CAM_Module,
    PAM_CAM_Layer,
    semanticModule,
    MAM_Module,
    ZAM_Module,
    WAM_Module,
    ZAM_WAM_Layer,
    GaborFeatureExtractor,
    GLCMFeatureExtractor,
    MultiConv,
    SE_Block,
    DropBlock2D,
    CCM_Module
)

from .resnext101_regular import ResNeXt101

class PAM_CAM_stack(nn.Module):
    def __init__(self):
        super(FBUSAS_stack, self).__init__()# 初始化父类nn.Module的构造函数。
        self.resnext = ResNeXt101()
        # 使用ResNeXt101作为backbone，这是一个预训练的深度卷积神经网络，用于提取图像特征。
        # 使用ResNeXt101作为backbone。输入: [8, 1, 128, 128]，输出: [8, 2048, 4, 4]
       
        # 下面的四个downX模块是用于减少特征图的通道数，使其从ResNeXt的输出变为64通道。
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 4, 4]
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 8, 8]
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 16, 16]
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 32, 32]

        inter_channels = 64
        out_channels=64
        
        # 下面的conv6_X和conv7_X模块是一系列的卷积层。
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        # conv8_X模块是另一系列的1x1卷积。
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.conv8_1=nn.Conv2d(64,64,1)
        self.conv8_2=nn.Conv2d(64,64,1)
        self.conv8_3=nn.Conv2d(64,64,1)
        self.conv8_4=nn.Conv2d(64,64,1)
        self.conv8_11=nn.Conv2d(64,64,1)
        self.conv8_12=nn.Conv2d(64,64,1)
        self.conv8_13=nn.Conv2d(64,64,1)
        self.conv8_14=nn.Conv2d(64,64,1)

        self.softmax_1 = nn.Softmax(dim=-1)

        # 下面的pam_attention_X_Y和cam_attention_X_Y模块是并行注意力模块。
        self.pam_attention_1_1= PAM_CAM_Layer(64, True)
        self.cam_attention_1_1= PAM_CAM_Layer(64, False)
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.semanticModule_1_1 = semanticModule(128)
        # 输出: [8, 128, 32, 32] (假设输入是down1和fuse1的concatenation)
        
        
        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # 输出: [8, 64, 32, 32]
    
    #Dual Attention mechanism
        self.pam_attention_1_2 = PAM_CAM_Layer(64)
        self.cam_attention_1_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_3 = PAM_CAM_Layer(64)
        self.cam_attention_1_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_1_4 = PAM_CAM_Layer(64)
        self.cam_attention_1_4 = PAM_CAM_Layer(64, False)
        
        self.pam_attention_2_1 = PAM_CAM_Layer(64)
        self.cam_attention_2_1 = PAM_CAM_Layer(64, False)
        self.semanticModule_2_1 = semanticModule(128)
        
        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.pam_attention_2_2 = PAM_CAM_Layer(64)
        self.cam_attention_2_2 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_3 = PAM_CAM_Layer(64)
        self.cam_attention_2_3 = PAM_CAM_Layer(64, False)
        self.pam_attention_2_4 = PAM_CAM_Layer(64)
        self.cam_attention_2_4 = PAM_CAM_Layer(64, False)
        
        #多重卷积模块
        self.fuse1 = MultiConv(256, 64, False)

        self.attention4 = MultiConv(128, 64)
        self.attention3 = MultiConv(128, 64)
        self.attention2 = MultiConv(128, 64)
        self.attention1 = MultiConv(128, 64)

        # 下面的refineX模块是用于细化特征图的卷积层。
        # 输出: [8, 64, 32, 32] (假设输入是down1和attention1_1的concatenation)
        self.refine4 = MultiConv(128, 64, False)
        self.refine3 = MultiConv(128, 64, False)
        self.refine2 = MultiConv(128, 64, False)
        self.refine1 = MultiConv(128, 64, False)

        # self.predict4 = nn.Conv2d(64, 5, kernel_size=1)
        # self.predict3 = nn.Conv2d(64, 5, kernel_size=1)
        # self.predict2 = nn.Conv2d(64, 5, kernel_size=1)
        # self.predict1 = nn.Conv2d(64, 5, kernel_size=1)

        # self.predict4_2 = nn.Conv2d(64, 5, kernel_size=1)
        # self.predict3_2 = nn.Conv2d(64, 5, kernel_size=1)
        # self.predict2_2 = nn.Conv2d(64, 5, kernel_size=1)
        # self.predict1_2 = nn.Conv2d(64, 5, kernel_size=1)
        
        # predictX和predictX_2模块是最终的预测层。
        # 输出: [8, 2, 32, 32]
        self.predict4 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 2, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        
        #提取语义向量(semVector_1_X)和语义特征图(semanticModule_1_X)
        #pam_attention_1_X和cam_attention_1_X模块是并行的注意力模块,它们分别计算像素注意力(attn_pamX)和通道注意力(attn_camX)。
        semVector_1_1,semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1),1))
        attn_pam4 = self.pam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_cam4 = self.cam_attention_1_4(torch.cat((down4, fuse1), 1))
        attention1_4=self.conv8_1((attn_cam4+attn_pam4)*self.conv_sem_1_1(semanticModule_1_1))

        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_pam3 = self.pam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_cam3 = self.cam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3=self.conv8_2((attn_cam3+attn_pam3)*self.conv_sem_1_2(semanticModule_1_2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        attn_pam2 = self.pam_attention_1_2(torch.cat((down2, fuse1), 1))
        attn_cam2 = self.cam_attention_1_2(torch.cat((down2, fuse1), 1))
        attention1_2=self.conv8_3((attn_cam2+attn_pam2)*self.conv_sem_1_3(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        attn_pam1 = self.pam_attention_1_1(torch.cat((down1, fuse1), 1))
        attn_cam1 = self.cam_attention_1_1(torch.cat((down1, fuse1), 1))
        attention1_1 = self.conv8_4((attn_cam1+attn_pam1) * self.conv_sem_1_4(semanticModule_1_4))
        
        ##new design with stacked attention

        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4_1 = self.pam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4_2 = self.cam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4 = self.conv8_11((refine4_1+refine4_2) * self.conv_sem_2_1(semanticModule_2_1))

        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.pam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3_2 = self.cam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3 = self.conv8_12((refine3_1+refine3_2) * self.conv_sem_2_2(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_1 = self.pam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2_2 = self.cam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2 = self.conv8_13((refine2_1+refine2_2)*self.conv_sem_2_3(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_1 = self.pam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        refine1_2 = self.cam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        refine1=self.conv8_14((refine1_1+refine1_2) * self.conv_sem_2_4(semanticModule_2_4))
        
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        
        if self.training:
            return semVector_1_1,\
                   semVector_2_1, \
                   semVector_1_2, \
                   semVector_2_2, \
                   semVector_1_3, \
                   semVector_2_3, \
                   semVector_1_4, \
                   semVector_2_4, \
                   torch.cat((down1, fuse1), 1),\
                   torch.cat((down2, fuse1), 1),\
                   torch.cat((down3, fuse1), 1),\
                   torch.cat((down4, fuse1), 1), \
                   torch.cat((down1, attention1_1 * fuse1), 1), \
                   torch.cat((down2, attention1_2 * fuse1), 1), \
                   torch.cat((down3, attention1_3 * fuse1), 1), \
                   torch.cat((down4, attention1_4 * fuse1), 1), \
                   semanticModule_1_4, \
                   semanticModule_1_3, \
                   semanticModule_1_2, \
                   semanticModule_1_1, \
                   semanticModule_2_4, \
                   semanticModule_2_3, \
                   semanticModule_2_2, \
                   semanticModule_2_1, \
                   predict1, \
                   predict2, \
                   predict3, \
                   predict4, \
                   predict1_2, \
                   predict2_2, \
                   predict3_2, \
                   predict4_2
        else:
            return ((predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4)

class FBUSAS_stack(nn.Module):
    def __init__(self):
        super(FBUSAS_stack, self).__init__()
        self.resnext = ResNeXt101()

        #self.channel_reducer = nn.Conv2d(64, 1, kernel_size=1)

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )

        # 形态学注意力模块
        self.mam_attention_1 = MAM_Module(64)
        self.mam_attention_2 = MAM_Module(64)

        self.fuse1 = MultiConv(256, 64, False)

        self.predict4 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 2, kernel_size=1)

        #self.upsample = nn.Upsample(size=(128, 128), mode='bilinear', align_corners=True)


    def forward(self, x):
        #x = self.channel_reducer(x)

        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        # 使用形态学注意力模块
        mam_attention1 = self.mam_attention_1(down1)
        mam_attention2 = self.mam_attention_2(down2)

        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))

        # predict1 = self.upsample(self.predict1(down1))
        # predict2 = self.upsample(self.predict2(down2))
        # predict3 = self.upsample(self.predict3(down3))
        # predict4 = self.upsample(self.predict4(down4))

        predict4 = self.predict4(down4)
        predict3 = self.predict3(down3)
        predict2 = self.predict2(down2)
        predict1 = self.predict1(down1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        


        if self.training:
            return mam_attention1, mam_attention2, fuse1, predict1, predict2, predict3, predict4
        else:
            return (predict1 + predict2 + predict3 + predict4) / 4

class FBUSAS_Model(nn.Module):
    def __init__(self, num_classes):
        super(FBUSAS_Model, self).__init__()
        
        resnext101 = models.resnext101_32x8d(pretrained=True)

        # 修改第一层以接受单通道输入
        original_first_layer = resnext101.conv1
        resnext101.conv1 = nn.Conv2d(1, 
                                     original_first_layer.out_channels, 
                                     kernel_size=original_first_layer.kernel_size, 
                                     stride=original_first_layer.stride, 
                                     padding=original_first_layer.padding, 
                                     bias=False)

        # 使用预训练权重初始化新的第一层
        with torch.no_grad():
            resnext101.conv1.weight[:, :] = torch.mean(original_first_layer.weight, dim=1, keepdim=True)

        for param in resnext101.parameters():
            param.requires_grad = False
        
        self.features = nn.Sequential(*list(resnext101.children())[:-2])
        
        # 添加FBUSAS_stack
        self.fbusas = FBUSAS_stack()
        
        # 添加自定义的分类层
        # self.classifier = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Flatten(), 
        #     nn.Linear(2048, num_classes)
        # )
        
    def forward(self, x):
        x = self.features(x)

        # 改变x的通道数到64
        x = nn.Conv2d(2048, 64, kernel_size=1)(x)

        x = self.fbusas(x)
        #x = self.classifier(x)
        return x

class ZAM_WAM_stack(nn.Module):
    def __init__(self):
        super(ZAM_WAM_stack, self).__init__()# 初始化父类nn.Module的构造函数。
        self.resnext = ResNeXt101()
        # 使用ResNeXt101作为backbone，这是一个预训练的深度卷积神经网络，用于提取图像特征。
        # 使用ResNeXt101作为backbone。输入: [8, 1, 128, 128]，输出: [8, 2048, 4, 4]
       
        # 下面的四个downX模块是用于减少特征图的通道数，使其从ResNeXt的输出变为64通道。
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 4, 4]
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 8, 8]
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 16, 16]
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 32, 32]

        self.gabor_feature_extractor4 = GaborFeatureExtractor(64, 64) 
        self.glcm_feature_extractor4 = GLCMFeatureExtractor(64, 64)  
        self.gabor_feature_extractor3 = GaborFeatureExtractor(64, 64) 
        self.glcm_feature_extractor3 = GLCMFeatureExtractor(64, 64)  
        self.gabor_feature_extractor2 = GaborFeatureExtractor(64, 64) 
        self.glcm_feature_extractor2 = GLCMFeatureExtractor(64, 64)  
        self.gabor_feature_extractor1 = GaborFeatureExtractor(64, 64)  
        self.glcm_feature_extractor1 = GLCMFeatureExtractor(64, 64)  

        self.downsize_enhanced4 = nn.Conv2d(192, 64, kernel_size=1)
        self.downsize_enhanced3 = nn.Conv2d(192, 64, kernel_size=1)
        self.downsize_enhanced2 = nn.Conv2d(192, 64, kernel_size=1)
        self.downsize_enhanced1 = nn.Conv2d(192, 64, kernel_size=1)

        inter_channels = 64
        out_channels=64
        
        # 下面的conv6_X和conv7_X模块是一系列的卷积层。
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        # conv8_X模块是另一系列的1x1卷积。
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.conv8_1=nn.Conv2d(64,64,1)
        self.conv8_2=nn.Conv2d(64,64,1)
        self.conv8_3=nn.Conv2d(64,64,1)
        self.conv8_4=nn.Conv2d(64,64,1)
        self.conv8_11=nn.Conv2d(64,64,1)
        self.conv8_12=nn.Conv2d(64,64,1)
        self.conv8_13=nn.Conv2d(64,64,1)
        self.conv8_14=nn.Conv2d(64,64,1)

        self.softmax_1 = nn.Softmax(dim=-1)

        # 使用ZAM_Module和WAM_Module
        self.zam_attention_1_1= ZAM_WAM_Layer(64, True)
        self.wam_attention_1_1= ZAM_WAM_Layer(64, False)
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.semanticModule_1_1 = semanticModule(128)
        
        
        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # 输出: [8, 64, 32, 32]
    
    #Dual Attention mechanism
        self.zam_attention_1_2 = ZAM_WAM_Layer(64)
        self.wam_attention_1_2 = ZAM_WAM_Layer(64, False)
        self.zam_attention_1_3 = ZAM_WAM_Layer(64)
        self.wam_attention_1_3 = ZAM_WAM_Layer(64, False)
        self.zam_attention_1_4 = ZAM_WAM_Layer(64)
        self.wam_attention_1_4 = ZAM_WAM_Layer(64, False)

        self.zam_attention_2_1 = ZAM_WAM_Layer(64)
        self.wam_attention_2_1 = ZAM_WAM_Layer(64, False)
        self.semanticModule_2_1 = semanticModule(128)
        
        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        self.zam_attention_2_2 = ZAM_WAM_Layer(64)
        self.wam_attention_2_2 = ZAM_WAM_Layer(64, False)
        self.zam_attention_2_3 = ZAM_WAM_Layer(64)
        self.wam_attention_2_3 = ZAM_WAM_Layer(64, False)
        self.zam_attention_2_4 = ZAM_WAM_Layer(64)
        self.wam_attention_2_4 = ZAM_WAM_Layer(64, False)

        #多重卷积模块
        self.fuse1 = MultiConv(256, 64, False)

        self.attention4 = MultiConv(128, 64)
        self.attention3 = MultiConv(128, 64)
        self.attention2 = MultiConv(128, 64)
        self.attention1 = MultiConv(128, 64)

        # 下面的refineX模块是用于细化特征图的卷积层。
        # 输出: [8, 64, 32, 32] (假设输入是down1和attention1_1的concatenation)
        self.refine4 = MultiConv(128, 64, False)
        self.refine3 = MultiConv(128, 64, False)
        self.refine2 = MultiConv(128, 64, False)
        self.refine1 = MultiConv(128, 64, False)
        
        # predictX和predictX_2模块是最终的预测层。
        # 输出: [8, 2, 32, 32]
        self.predict4 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 2, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        # Add feature extraction after down-sampling
        gabor4 = self.gabor_feature_extractor4(down4)
        glcm4 = self.glcm_feature_extractor4(down4)
        enhanced4 = torch.cat([down4, gabor4, glcm4], dim=1)
        
        gabor3 = self.gabor_feature_extractor3(down3)
        glcm3 = self.glcm_feature_extractor3(down3)
        enhanced3 = torch.cat([down3, gabor3, glcm3], dim=1)

        gabor2 = self.gabor_feature_extractor2(down2)
        glcm2 = self.glcm_feature_extractor2(down2)
        enhanced2 = torch.cat([down2, gabor2, glcm2], dim=1)

        gabor1 = self.gabor_feature_extractor1(down1)
        glcm1 = self.glcm_feature_extractor1(down1)
        enhanced1 = torch.cat([down1, gabor1, glcm1], dim=1)

        downsize_enhanced4 = self.downsize_enhanced4(enhanced4)
        predict4 = self.predict4(downsize_enhanced4)
        downsize_enhanced3 = self.downsize_enhanced3(enhanced3)
        predict3 = self.predict3(downsize_enhanced3)
        downsize_enhanced2 = self.downsize_enhanced2(enhanced2)
        predict2 = self.predict2(downsize_enhanced2)
        downsize_enhanced1 = self.downsize_enhanced1(enhanced1)
        predict1 = self.predict1(downsize_enhanced1)


        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        
        #提取语义向量(semVector_1_X)和语义特征图(semanticModule_1_X)
        semVector_1_1, semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1), 1))
        attn_zam4 = self.zam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_wam4 = self.wam_attention_1_4(torch.cat((down4, fuse1), 1))
        attention1_4 = self.conv8_1((attn_wam4 + attn_zam4) * self.conv_sem_1_1(semanticModule_1_1))

        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_zam3 = self.zam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_wam3 = self.wam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3 = self.conv8_2((attn_wam3 + attn_zam3) * self.conv_sem_1_2(semanticModule_1_2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        attn_zam2 = self.zam_attention_1_2(torch.cat((down2, fuse1), 1))
        attn_wam2 = self.wam_attention_1_2(torch.cat((down2, fuse1), 1))
        attention1_2 = self.conv8_3((attn_wam2 + attn_zam2) * self.conv_sem_1_3(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        attn_zam1 = self.zam_attention_1_1(torch.cat((down1, fuse1), 1))
        attn_wam1 = self.wam_attention_1_1(torch.cat((down1, fuse1), 1))
        attention1_1 = self.conv8_4((attn_wam1 + attn_zam1) * self.conv_sem_1_4(semanticModule_1_4))

        
        ##new design with stacked attention
        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4_1 = self.zam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4_2 = self.wam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4 = self.conv8_11((refine4_1+refine4_2) * self.conv_sem_2_1(semanticModule_2_1))

        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.zam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3_2 = self.wam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3 = self.conv8_12((refine3_1+refine3_2) * self.conv_sem_2_2(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, attention1_2 * fuse1), 1))
        refine2_1 = self.zam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2_2 = self.wam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        refine2 = self.conv8_13((refine2_1+refine2_2)*self.conv_sem_2_3(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, attention1_1 * fuse1), 1))
        refine1_1 = self.zam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        refine1_2 = self.wam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        refine1 = self.conv8_14((refine1_1+refine1_2) * self.conv_sem_2_4(semanticModule_2_4))

        
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(refine2)
        predict1_2 = self.predict1_2(refine1)

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        
        if self.training:
            return semVector_1_1,\
                   semVector_2_1, \
                   semVector_1_2, \
                   semVector_2_2, \
                   semVector_1_3, \
                   semVector_2_3, \
                   semVector_1_4, \
                   semVector_2_4, \
                   torch.cat((down1, fuse1), 1),\
                   torch.cat((down2, fuse1), 1),\
                   torch.cat((down3, fuse1), 1),\
                   torch.cat((down4, fuse1), 1), \
                   torch.cat((down1, attention1_1 * fuse1), 1), \
                   torch.cat((down2, attention1_2 * fuse1), 1), \
                   torch.cat((down3, attention1_3 * fuse1), 1), \
                   torch.cat((down4, attention1_4 * fuse1), 1), \
                   semanticModule_1_4, \
                   semanticModule_1_3, \
                   semanticModule_1_2, \
                   semanticModule_1_1, \
                   semanticModule_2_4, \
                   semanticModule_2_3, \
                   semanticModule_2_2, \
                   semanticModule_2_1, \
                   predict1, \
                   predict2, \
                   predict3, \
                   predict4, \
                   predict1_2, \
                   predict2_2, \
                   predict3_2, \
                   predict4_2
        else:
            return ((predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4)

class ZAM_WAM_Simplified_stack(nn.Module):

    def __init__(self):
        super(ZAM_WAM_Simplified_stack, self).__init__()# 初始化父类nn.Module的构造函数。
        self.resnext = ResNeXt101()
        # 使用ResNeXt101作为backbone，这是一个预训练的深度卷积神经网络，用于提取图像特征。
        # 使用ResNeXt101作为backbone。输入: [8, 1, 128, 128]，输出: [8, 2048, 4, 4]
       
        # 下面的四个downX模块是用于减少特征图的通道数，使其从ResNeXt的输出变为64通道。
        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 4, 4]
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 8, 8]
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 16, 16]
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 64, kernel_size=1), nn.BatchNorm2d(64), nn.PReLU()
        )# 输出: [8, 64, 32, 32]

        self.gabor_feature_extractor4 = GaborFeatureExtractor(64, 64) 
        self.glcm_feature_extractor4 = GLCMFeatureExtractor(64, 64)  
        self.gabor_feature_extractor3 = GaborFeatureExtractor(64, 64) 
        self.glcm_feature_extractor3 = GLCMFeatureExtractor(64, 64)  
        self.gabor_feature_extractor2 = GaborFeatureExtractor(64, 64) 
        self.glcm_feature_extractor2 = GLCMFeatureExtractor(64, 64)  
        self.gabor_feature_extractor1 = GaborFeatureExtractor(64, 64)  
        self.glcm_feature_extractor1 = GLCMFeatureExtractor(64, 64)  

        self.downsize_enhanced4 = nn.Conv2d(192, 64, kernel_size=1)
        self.downsize_enhanced3 = nn.Conv2d(192, 64, kernel_size=1)
        self.downsize_enhanced2 = nn.Conv2d(192, 64, kernel_size=1)
        self.downsize_enhanced1 = nn.Conv2d(192, 64, kernel_size=1)

        inter_channels = 64
        out_channels=64
        
        # 下面的conv6_X和conv7_X模块是一系列的卷积层。
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.conv6_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv6_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        self.conv7_1 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_2 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_3 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))
        self.conv7_4 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(64, out_channels, 1))

        # conv8_X模块是另一系列的1x1卷积。
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.conv8_1=nn.Conv2d(64,64,1)
        self.conv8_2=nn.Conv2d(64,64,1)
        self.conv8_3=nn.Conv2d(64,64,1)
        self.conv8_4=nn.Conv2d(64,64,1)
        self.conv8_11=nn.Conv2d(64,64,1)
        self.conv8_12=nn.Conv2d(64,64,1)
        self.conv8_13=nn.Conv2d(64,64,1)
        self.conv8_14=nn.Conv2d(64,64,1)

        self.softmax_1 = nn.Softmax(dim=-1)

        # 使用ZAM_Module和WAM_Module
        # self.zam_attention_1_1= ZAM_WAM_Layer(64, True)
        # self.wam_attention_1_1= ZAM_WAM_Layer(64, False)
        # 输出: [8, 64, 32, 32] (假设输入是down1的输出)
        self.semanticModule_1_1 = semanticModule(128)
        
        
        self.conv_sem_1_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_1_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        # 输出: [8, 64, 32, 32]
    
    #Dual Attention mechanism
        # self.zam_attention_1_2 = ZAM_WAM_Layer(64)
        # self.wam_attention_1_2 = ZAM_WAM_Layer(64, False)
        self.zam_attention_1_3 = ZAM_WAM_Layer(64)
        self.wam_attention_1_3 = ZAM_WAM_Layer(64, False)
        self.zam_attention_1_4 = ZAM_WAM_Layer(64)
        self.wam_attention_1_4 = ZAM_WAM_Layer(64, False)

        # self.zam_attention_2_1 = ZAM_WAM_Layer(64)
        # self.wam_attention_2_1 = ZAM_WAM_Layer(64, False)
        self.semanticModule_2_1 = semanticModule(128)
        
        self.conv_sem_2_1 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_2 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_3 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv_sem_2_4 = nn.Conv2d(128, 64, kernel_size=3, padding=1)

        # self.zam_attention_2_2 = ZAM_WAM_Layer(64)
        # self.wam_attention_2_2 = ZAM_WAM_Layer(64, False)
        self.zam_attention_2_3 = ZAM_WAM_Layer(64)
        self.wam_attention_2_3 = ZAM_WAM_Layer(64, False)
        self.zam_attention_2_4 = ZAM_WAM_Layer(64)
        self.wam_attention_2_4 = ZAM_WAM_Layer(64, False)

        #多重卷积模块
        self.fuse1 = MultiConv(256, 64, False)

        self.attention4 = MultiConv(128, 64)
        self.attention3 = MultiConv(128, 64)
        self.attention2 = MultiConv(128, 64)
        self.attention1 = MultiConv(128, 64)

        # 下面的refineX模块是用于细化特征图的卷积层。
        # 输出: [8, 64, 32, 32] (假设输入是down1和attention1_1的concatenation)
        self.refine4 = MultiConv(128, 64, False)
        self.refine3 = MultiConv(128, 64, False)
        self.refine2 = MultiConv(128, 64, False)
        self.refine1 = MultiConv(128, 64, False)
        
        # predictX和predictX_2模块是最终的预测层。
        # 输出: [8, 2, 32, 32]
        self.predict4 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1 = nn.Conv2d(64, 2, kernel_size=1)

        self.predict4_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict3_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict2_2 = nn.Conv2d(64, 2, kernel_size=1)
        self.predict1_2 = nn.Conv2d(64, 2, kernel_size=1)


    def forward(self, x):
        layer0 = self.resnext.layer0(x)
        layer1 = self.resnext.layer1(layer0)
        layer2 = self.resnext.layer2(layer1)
        layer3 = self.resnext.layer3(layer2)
        layer4 = self.resnext.layer4(layer3)

        down4 = F.upsample(self.down4(layer4), size=layer1.size()[2:], mode='bilinear')
        down3 = F.upsample(self.down3(layer3), size=layer1.size()[2:], mode='bilinear')
        down2 = F.upsample(self.down2(layer2), size=layer1.size()[2:], mode='bilinear')
        down1 = self.down1(layer1)

        # Add feature extraction after down-sampling
        gabor4 = self.gabor_feature_extractor4(down4)
        glcm4 = self.glcm_feature_extractor4(down4)
        enhanced4 = torch.cat([down4, gabor4, glcm4], dim=1)
        
        gabor3 = self.gabor_feature_extractor3(down3)
        glcm3 = self.glcm_feature_extractor3(down3)
        enhanced3 = torch.cat([down3, gabor3, glcm3], dim=1)

        gabor2 = self.gabor_feature_extractor2(down2)
        glcm2 = self.glcm_feature_extractor2(down2)
        enhanced2 = torch.cat([down2, gabor2, glcm2], dim=1)

        gabor1 = self.gabor_feature_extractor1(down1)
        glcm1 = self.glcm_feature_extractor1(down1)
        enhanced1 = torch.cat([down1, gabor1, glcm1], dim=1)

        downsize_enhanced4 = self.downsize_enhanced4(enhanced4)
        predict4 = self.predict4(downsize_enhanced4)
        downsize_enhanced3 = self.downsize_enhanced3(enhanced3)
        predict3 = self.predict3(downsize_enhanced3)
        downsize_enhanced2 = self.downsize_enhanced2(enhanced2)
        predict2 = self.predict2(downsize_enhanced2)
        downsize_enhanced1 = self.downsize_enhanced1(enhanced1)
        predict1 = self.predict1(downsize_enhanced1)


        fuse1 = self.fuse1(torch.cat((down4, down3, down2, down1), 1))
        
        #提取语义向量(semVector_1_X)和语义特征图(semanticModule_1_X)
        semVector_1_1, semanticModule_1_1 = self.semanticModule_1_1(torch.cat((down4, fuse1), 1))
        attn_zam4 = self.zam_attention_1_4(torch.cat((down4, fuse1), 1))
        attn_wam4 = self.wam_attention_1_4(torch.cat((down4, fuse1), 1))
        attention1_4 = self.conv8_1((attn_wam4 + attn_zam4) * self.conv_sem_1_1(semanticModule_1_1))

        semVector_1_2, semanticModule_1_2 = self.semanticModule_1_1(torch.cat((down3, fuse1), 1))
        attn_zam3 = self.zam_attention_1_3(torch.cat((down3, fuse1), 1))
        attn_wam3 = self.wam_attention_1_3(torch.cat((down3, fuse1), 1))
        attention1_3 = self.conv8_2((attn_wam3 + attn_zam3) * self.conv_sem_1_2(semanticModule_1_2))

        semVector_1_3, semanticModule_1_3 = self.semanticModule_1_1(torch.cat((down2, fuse1), 1))
        #attn_zam2 = self.zam_attention_1_2(torch.cat((down2, fuse1), 1))
        #attn_wam2 = self.wam_attention_1_2(torch.cat((down2, fuse1), 1))
        #attention1_2 = self.conv8_3((attn_wam2 + attn_zam2) * self.conv_sem_1_3(semanticModule_1_3))

        semVector_1_4, semanticModule_1_4 = self.semanticModule_1_1(torch.cat((down1, fuse1), 1))
        #attn_zam1 = self.zam_attention_1_1(torch.cat((down1, fuse1), 1))
        #attn_wam1 = self.wam_attention_1_1(torch.cat((down1, fuse1), 1))
        #attention1_1 = self.conv8_4((attn_wam1 + attn_zam1) * self.conv_sem_1_4(semanticModule_1_4))

        
        ##new design with stacked attention
        semVector_2_1, semanticModule_2_1 = self.semanticModule_2_1(torch.cat((down4, attention1_4 * fuse1), 1))
        refine4_1 = self.zam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4_2 = self.wam_attention_2_4(torch.cat((down4,attention1_4*fuse1),1))
        refine4 = self.conv8_11((refine4_1+refine4_2) * self.conv_sem_2_1(semanticModule_2_1))

        semVector_2_2, semanticModule_2_2 = self.semanticModule_2_1(torch.cat((down3, attention1_3 * fuse1), 1))
        refine3_1 = self.zam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3_2 = self.wam_attention_2_3(torch.cat((down3,attention1_3*fuse1),1))
        refine3 = self.conv8_12((refine3_1+refine3_2) * self.conv_sem_2_2(semanticModule_2_2))

        semVector_2_3, semanticModule_2_3 = self.semanticModule_2_1(torch.cat((down2, self.conv_sem_1_3(semanticModule_1_3) * fuse1), 1))
        #refine2_1 = self.zam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        #refine2_2 = self.wam_attention_2_2(torch.cat((down2,attention1_2*fuse1),1))
        #refine2 = self.conv8_13((refine2_1+refine2_2)*self.conv_sem_2_3(semanticModule_2_3))

        semVector_2_4, semanticModule_2_4 = self.semanticModule_2_1(torch.cat((down1, self.conv_sem_1_4(semanticModule_1_4) * fuse1), 1))
        #refine1_1 = self.zam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        #refine1_2 = self.wam_attention_2_1(torch.cat((down1,attention1_1 * fuse1),1))
        #refine1 = self.conv8_14((refine1_1+refine1_2) * self.conv_sem_2_4(semanticModule_2_4))

        
        predict4_2 = self.predict4_2(refine4)
        predict3_2 = self.predict3_2(refine3)
        predict2_2 = self.predict2_2(self.conv_sem_2_3(semanticModule_2_3))
        predict1_2 = self.predict1_2(self.conv_sem_2_4(semanticModule_2_4))

        predict1 = F.upsample(predict1, size=x.size()[2:], mode='bilinear')
        predict2 = F.upsample(predict2, size=x.size()[2:], mode='bilinear')
        predict3 = F.upsample(predict3, size=x.size()[2:], mode='bilinear')
        predict4 = F.upsample(predict4, size=x.size()[2:], mode='bilinear')

        predict1_2 = F.upsample(predict1_2, size=x.size()[2:], mode='bilinear')
        predict2_2 = F.upsample(predict2_2, size=x.size()[2:], mode='bilinear')
        predict3_2 = F.upsample(predict3_2, size=x.size()[2:], mode='bilinear')
        predict4_2 = F.upsample(predict4_2, size=x.size()[2:], mode='bilinear')
        
        if self.training:
            return semVector_1_1,\
                   semVector_2_1, \
                   semVector_1_2, \
                   semVector_2_2, \
                   semVector_1_3, \
                   semVector_2_3, \
                   semVector_1_4, \
                   semVector_2_4, \
                   torch.cat((down1, fuse1), 1),\
                   torch.cat((down2, fuse1), 1),\
                   torch.cat((down3, fuse1), 1),\
                   torch.cat((down4, fuse1), 1), \
                   torch.cat((down1, self.conv_sem_1_4(semanticModule_1_4) * fuse1), 1), \
                   torch.cat((down2, self.conv_sem_1_3(semanticModule_1_3) * fuse1), 1), \
                   torch.cat((down3, attention1_3 * fuse1), 1), \
                   torch.cat((down4, attention1_4 * fuse1), 1), \
                   semanticModule_1_4, \
                   semanticModule_1_3, \
                   semanticModule_1_2, \
                   semanticModule_1_1, \
                   semanticModule_2_4, \
                   semanticModule_2_3, \
                   semanticModule_2_2, \
                   semanticModule_2_1, \
                   predict1, \
                   predict2, \
                   predict3, \
                   predict4, \
                   predict1_2, \
                   predict2_2, \
                   predict3_2, \
                   predict4_2
        else:
            return ((predict1_2 + predict2_2 + predict3_2 + predict4_2) / 4)

class ZAM_WAM_Unet(nn.Module):
    def __init__(self):
        super(ZAM_WAM_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(1, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        
        self.middle = self.conv_block(64, 128)
        
        self.decoder3 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = self.conv_block(32, 16)

        self.reduce_dim1 = nn.Conv2d(48, 16, kernel_size=1)
        self.reduce_dim2 = nn.Conv2d(96, 32, kernel_size=1)
        self.reduce_dim3 = nn.Conv2d(64, 64, kernel_size=1)

        self.reduce_dim_middle = nn.Conv2d(128, 64, kernel_size=1)

        self.reduce_dim_dec3 = nn.Conv2d(96, 64, kernel_size=1)
        self.reduce_dim_dec2 = nn.Conv2d(64, 32, kernel_size=1)
        self.reduce_dim_dec1 = nn.Conv2d(32, 16, kernel_size=1)

        self.zam3 = ZAM_Module(64)
        self.wam3 = WAM_Module(64)
        
        self.zam_middle = ZAM_Module(128)
        self.wam_middle = WAM_Module(128)

        self.se3 = SE_Block(64)
        self.se_middle = SE_Block(128)

        self.gabor1 = GaborFeatureExtractor(16, 16)
        self.glcm1 = GLCMFeatureExtractor(16, 16)
        
        self.gabor2 = GaborFeatureExtractor(32, 32)
        self.glcm2 = GLCMFeatureExtractor(32, 32)
        
        #self.final_conv = nn.Conv2d(16, 1, kernel_size=1)
        self.final_conv = nn.Conv2d(16, 2, kernel_size=1) #输出通道数为2，与Segmentation_onehot匹配

        self.raw_W = nn.Parameter(torch.tensor([1.0, 1.0, 1.0, 1.0]), requires_grad=True)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = torch.cat([self.gabor1(enc1), self.glcm1(enc1), enc1], 1)  #gabor & glcm
        enc1 = self.reduce_dim1(enc1)  #torch.Size([8, 16, 128, 128])
        pool1 = F.max_pool2d(enc1, 2)
        
        enc2 = self.encoder2(pool1)
        enc2 = torch.cat([self.gabor2(enc2), self.glcm2(enc2), enc2], 1)  #gabor & glcm
        enc2 = self.reduce_dim2(enc2)  #torch.Size([8, 32, 64, 64])
        pool2 = F.max_pool2d(enc2, 2)
        
        enc3 = self.encoder3(pool2)
        enc3 = enc3 + self.zam3(enc3) + self.wam3(enc3)  # Residual connection
        enc3 = self.se3(enc3)  # SE Block
        enc3 = self.reduce_dim3(enc3)  #torch.Size([8, 64, 32, 32])
        pool3 = F.max_pool2d(enc3, 2)
        
        middle = self.middle(pool3)
        middle = middle + self.zam_middle(middle) + self.wam_middle(middle)  # Residual connection
        middle = self.se_middle(middle)  # SE Block
        middle = self.reduce_dim_middle(middle)  #torch.Size([8, 64, 16, 16])
        
        dec3 = self.decoder2(F.interpolate(middle, scale_factor=2))
        dec3 = torch.cat([dec3, enc3], dim=1)  # Residual connection
        dec3 = self.reduce_dim_dec3(dec3)  #torch.Size([8, 64, 32, 32])
        #dec3 = dec3 + self.zam3(dec3) + self.wam3(dec3)

        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = torch.cat([dec2, enc2], dim=1)  # Residual connection
        dec2 = self.reduce_dim_dec2(dec2)  #torch.Size([8, 32, 64, 64])

        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = torch.cat([dec1, enc1], dim=1)  # Residual connection
        dec1 = self.reduce_dim_dec1(dec1)  #torch.Size([8, 16, 128, 128])

        #end = torch.sigmoid(self.final_conv(dec1))
        end = self.final_conv(dec1)  #torch.Size([8, 2, 128, 128])
        
        if self.training:
            return  enc1,\
                    enc2, \
                    enc3, \
                    middle, \
                    dec3, \
                    dec2, \
                    dec1, \
                    end
        else:
            return end

class Shape_Unet(nn.Module):
    def __init__(self):
        super(Shape_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(1, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        
        self.middle = self.conv_block(64, 128)
        
        self.decoder3 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = self.conv_block(32, 16)

        self.reduce_dim1 = nn.Conv2d(48, 16, kernel_size=1)
        self.reduce_dim2 = nn.Conv2d(96, 32, kernel_size=1)
        self.reduce_dim3 = nn.Conv2d(64, 64, kernel_size=1)

        self.reduce_dim_middle = nn.Conv2d(128, 64, kernel_size=1)

        self.reduce_dim_dec3 = nn.Conv2d(96, 64, kernel_size=1)
        self.reduce_dim_dec2 = nn.Conv2d(64, 32, kernel_size=1)
        self.reduce_dim_dec1 = nn.Conv2d(32, 16, kernel_size=1)

        self.zam3 = ZAM_Module(64)
        self.wam3 = WAM_Module(64)
        
        self.zam_middle = ZAM_Module(128)
        self.wam_middle = WAM_Module(128)

        self.se3 = SE_Block(64)
        self.se_middle = SE_Block(128)

        self.gabor1 = GaborFeatureExtractor(16, 16)
        self.glcm1 = GLCMFeatureExtractor(16, 16)
        
        self.gabor2 = GaborFeatureExtractor(32, 32)
        self.glcm2 = GLCMFeatureExtractor(32, 32)
        
        self.final_conv = nn.Conv2d(16, 2, kernel_size=1) 
        self.edge_conv = nn.Conv2d(16, 2, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        enc1 = self.encoder1(x)
        enc1 = torch.cat([self.gabor1(enc1), self.glcm1(enc1), enc1], 1)  #gabor & glcm
        enc1 = self.reduce_dim1(enc1)  #torch.Size([8, 16, 128, 128])
        pool1 = F.max_pool2d(enc1, 2)
        
        enc2 = self.encoder2(pool1)
        enc2 = torch.cat([self.gabor2(enc2), self.glcm2(enc2), enc2], 1)  #gabor & glcm
        enc2 = self.reduce_dim2(enc2)  #torch.Size([8, 32, 64, 64])
        pool2 = F.max_pool2d(enc2, 2)
        
        enc3 = self.encoder3(pool2)
        enc3 = enc3 + self.zam3(enc3) + self.wam3(enc3)  # Residual connection
        enc3 = self.se3(enc3)  # SE Block
        enc3 = self.reduce_dim3(enc3)  #torch.Size([8, 64, 32, 32])
        pool3 = F.max_pool2d(enc3, 2)
        
        middle = self.middle(pool3)
        middle = middle + self.zam_middle(middle) + self.wam_middle(middle)  # Residual connection
        middle = self.se_middle(middle)  # SE Block
        middle = self.reduce_dim_middle(middle)  #torch.Size([8, 64, 16, 16])
        
        dec3 = self.decoder2(F.interpolate(middle, scale_factor=2))
        dec3 = torch.cat([dec3, enc3], dim=1)  # Residual connection
        dec3 = self.reduce_dim_dec3(dec3)  #torch.Size([8, 64, 32, 32])

        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = torch.cat([dec2, enc2], dim=1)  # Residual connection
        dec2 = self.reduce_dim_dec2(dec2)  #torch.Size([8, 32, 64, 64])

        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = torch.cat([dec1, enc1], dim=1)  # Residual connection
        dec1 = self.reduce_dim_dec1(dec1)  #torch.Size([8, 16, 128, 128])

        segmentation = self.final_conv(dec1)  # 分割结果
        edge = self.edge_conv(dec1)  # 边缘图像
        
        return segmentation, edge

class CC_CV_Unet(nn.Module):
    def __init__(self):
        super(CC_CV_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(3, 16)  # 3通道，考虑坐标映射
        self.encoder2 = self.conv_block(16+2, 32)  # 18通道，16 from enc1 + 2 for coord map
        self.encoder3 = self.conv_block(32+2, 64)  # 34通道，32 from enc2 + 2 for coord map
        
        self.middle = self.conv_block(64+2, 128)  # 66通道, 64 from enc3 + 2 for coord map
        
        self.decoder3 = self.conv_block(64, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = self.conv_block(32, 16)

        self.se3 = SE_Block(64)
        self.se_middle = SE_Block(128)

        self.reduce_dim_middle = nn.Conv2d(128, 64, kernel_size=1)

        self.reduce_dim_dec3 = nn.Conv2d(128, 64, kernel_size=1)  # 修复维度 128->64
        self.reduce_dim_dec2 = nn.Conv2d(64, 32, kernel_size=1)  # 修复维度 96->32
        self.reduce_dim_dec1 = nn.Conv2d(32, 16, kernel_size=1)  # 修复维度 48->16
        
        self.final_conv = nn.Conv2d(16, 2, kernel_size=1)
        self.edge_conv = nn.Conv2d(16, 2, kernel_size=1)

    def coord_map(self, height, width):
            # 生成二维坐标映射
            x = torch.linspace(-1, 1, width).view(1, 1, width).expand(1, height, width)
            y = torch.linspace(-1, 1, height).view(1, height, 1).expand(1, height, width)
            grid = torch.cat([x, y], 0)
            return grid
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Initial coordinate map for input size
        coords = self.coord_map(x.size(2), x.size(3)).to(x.device).unsqueeze(0)
        
        x_enc1 = torch.cat([x, coords.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc1 = self.encoder1(x_enc1)
        
        # Updated coordinate map for pooled size
        coords_pooled_enc1 = self.coord_map(enc1.size(2)//2, enc1.size(3)//2).to(x.device).unsqueeze(0)
        x_enc2 = torch.cat([F.max_pool2d(enc1, 2), coords_pooled_enc1.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc2 = self.encoder2(x_enc2)
        
        # Updated coordinate map for pooled size
        coords_pooled_enc2 = self.coord_map(enc2.size(2)//2, enc2.size(3)//2).to(x.device).unsqueeze(0)
        x_enc3 = torch.cat([F.max_pool2d(enc2, 2), coords_pooled_enc2.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc3 = self.encoder3(x_enc3)
        
        enc3 = self.se3(enc3)
        
        # Updated coordinate map for pooled size
        coords_pooled_enc3 = self.coord_map(enc3.size(2)//2, enc3.size(3)//2).to(x.device).unsqueeze(0)
        x_middle = torch.cat([F.max_pool2d(enc3, 2), coords_pooled_enc3.repeat(x.size(0), 1, 1, 1)], dim=1)
        middle = self.middle(x_middle)
        middle = self.se_middle(middle)
        middle = self.reduce_dim_middle(middle)

        dec3 = self.decoder3(F.interpolate(middle, scale_factor=2))
        dec3 = self.reduce_dim_dec3(torch.cat([dec3, enc3], dim=1))

        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = self.reduce_dim_dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = self.reduce_dim_dec1(torch.cat([dec1, enc1], dim=1))

        segmentation = self.final_conv(dec1)  # 输出分割结果
        edge = self.edge_conv(dec1)  # 边缘图像

        return segmentation, edge

class CC_SP_Unet(nn.Module):
    def __init__(self):
        super(CC_SP_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(3, 16)  # 3通道，考虑坐标映射
        self.encoder2 = self.conv_block(16+2, 32)  # 18通道，16 from enc1 + 2 for coord map
        self.encoder3 = self.conv_block(32+2, 64)  # 34通道，32 from enc2 + 2 for coord map
        
        self.middle = self.conv_block(64+2, 64)
       
        self.ccm = CCM_Module(64, 128) # Add CCM after middle layer

        # Splitting to branches for up-sampling
        self.decoder3_branch1 = self.conv_block(64, 64)
        self.decoder2_branch1 = self.conv_block(64, 32)
        self.decoder1_branch1 = self.conv_block(32, 16)
        
        self.decoder3_branch2 = self.conv_block(64, 64)
        self.decoder2_branch2 = self.conv_block(64, 32)
        self.decoder1_branch2 = self.conv_block(32, 16)

        # Adjust the final convolutions for both branches
        self.final_conv_branch1 = nn.Conv2d(16, 2, kernel_size=1)
        self.edge_conv_branch1 = nn.Conv2d(16, 2, kernel_size=1)

        self.final_conv_branch2 = nn.Conv2d(16, 2, kernel_size=1)
        self.edge_conv_branch2 = nn.Conv2d(16, 2, kernel_size=1)

    def coord_map(self, height, width):
            # 生成二维坐标映射
            x = torch.linspace(-1, 1, width).view(1, 1, width).expand(1, height, width)
            y = torch.linspace(-1, 1, height).view(1, height, 1).expand(1, height, width)
            grid = torch.cat([x, y], 0)
            return grid
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            DropBlock2D(block_size=7, keep_prob=0.9),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
       # Generate coordinate map for input size
        coords = self.coord_map(x.size(2), x.size(3)).to(x.device).unsqueeze(0)
        
        x_enc1 = torch.cat([x, coords.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc1 = self.encoder1(x_enc1)
        
        coords_pooled_enc1 = self.coord_map(enc1.size(2)//2, enc1.size(3)//2).to(x.device).unsqueeze(0)
        x_enc2 = torch.cat([F.max_pool2d(enc1, 2), coords_pooled_enc1.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc2 = self.encoder2(x_enc2)

        coords_pooled_enc2 = self.coord_map(enc2.size(2)//2, enc2.size(3)//2).to(x.device).unsqueeze(0)
        x_enc3 = torch.cat([F.max_pool2d(enc2, 2), coords_pooled_enc2.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc3 = self.encoder3(x_enc3)
        
        coords_pooled_enc3 = self.coord_map(enc3.size(2)//2, enc3.size(3)//2).to(x.device).unsqueeze(0)
        x_middle = torch.cat([F.max_pool2d(enc3, 2), coords_pooled_enc3.repeat(x.size(0), 1, 1, 1)], dim=1)
        middle = self.middle(x_middle)
        LCCM, projected_Fp, projected_Ft = self.ccm(middle)

        # Splitting to two branches
        dec3_branch1 = self.decoder3_branch1(F.interpolate(middle, scale_factor=2))
        dec2_branch1 = self.decoder2_branch1(F.interpolate(dec3_branch1, scale_factor=2))
        dec1_branch1 = self.decoder1_branch1(F.interpolate(dec2_branch1, scale_factor=2))

        dec3_branch2 = self.decoder3_branch2(F.interpolate(middle, scale_factor=2))
        dec2_branch2 = self.decoder2_branch2(F.interpolate(dec3_branch2, scale_factor=2))
        dec1_branch2 = self.decoder1_branch2(F.interpolate(dec2_branch2, scale_factor=2))

        segmentation_branch1 = self.final_conv_branch1(dec1_branch1)
        edge_branch1 = self.edge_conv_branch1(dec1_branch1)

        segmentation_branch2 = self.final_conv_branch2(dec1_branch2)
        edge_branch2 = self.edge_conv_branch2(dec1_branch2)

        return segmentation_branch1, edge_branch1, segmentation_branch2, edge_branch2, LCCM
