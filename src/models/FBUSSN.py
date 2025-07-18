from functools import reduce

import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms.functional
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

# Unet
class Basic_Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Basic_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.middle = self.conv_block(512, 1024)
        
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(pool3)
        pool4 = F.max_pool2d(enc4, 2)

        # Middle
        middle = self.middle(pool4)

        # Decoder
        dec4 = self.decoder4(torch.cat([F.interpolate(middle, scale_factor=2), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        # Final Layer
        out = self.final_conv(dec1)
        
        return out

# Unet return region & edge
class Shape_Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Shape_Unet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.dropout1 = nn.Dropout(0.5)

        # Middle
        self.middle = self.conv_block(512, 1024)
        self.dropout2 = nn.Dropout(0.5)

        # Decoder
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.edge_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, 2)
        
        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, 2)
        
        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, 2)
        
        enc4 = self.encoder4(pool3)
        enc4 = self.dropout1(enc4)
        pool4 = F.max_pool2d(enc4, 2)
        
        # Middle
        middle = self.middle(pool4)
        middle = self.dropout2(middle)

        # Decoder
        dec4 = self.decoder4(torch.cat([F.interpolate(middle, scale_factor=2), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        # Output
        region = self.final_conv(dec1)
        edge = self.edge_conv(dec1)
       
        return region, edge

# Unet with ZAM & WAM return region & edge
class Shape_ZAM_WAM_Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(Shape_ZAM_WAM_Unet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        self.dropout1 = nn.Dropout(0.5)

        # Middle
        self.middle = self.conv_block(512, 1024)
        self.dropout2 = nn.Dropout(0.5)

        # Decoder
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)

        # Attention
        self.ZAM = ZAM_Module(512)
        self.WAM = WAM_Module(1024)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.edge_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, 2)
        
        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, 2)
        
        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, 2)
        
        enc4 = self.encoder4(pool3)
        enc4 = self.ZAM(enc4) # Zernike Attention after the last encoder block
        enc4 = self.dropout1(enc4)
        pool4 = F.max_pool2d(enc4, 2)
        
        # Middle
        middle = self.middle(pool4)
        middle = self.dropout2(middle)

        # Decoder
        dec4 = self.WAM(F.interpolate(middle, scale_factor=2))   # Wavelet Attention before the first decoder block
        dec4 = self.decoder4(torch.cat([F.interpolate(dec4, scale_factor=2), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        # Output
        region = self.final_conv(dec1)
        edge = self.edge_conv(dec1)
       
        return region, edge

# Unet with depthwise conv return region & edge
class Shape_Depthwise_Unet(nn.Module):
    def __init__(self):
        super(Shape_Depthwise_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(1, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        
        self.middle = self.conv_block(512, 1024)
        
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)
        
        self.final_conv = nn.Conv2d(64, 2, kernel_size=1)
        self.edge_conv = nn.Conv2d(64, 2, kernel_size=1)

    def depthwise_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        return nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)

    def pointwise_conv(in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            depthwise_conv(in_channels, out_channels),
            pointwise_conv(in_channels, out_channels),
            nn.ReLU(inplace=True),
            depthwise_conv(out_channels, out_channels),
            pointwise_conv(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(pool3)
        pool4 = F.max_pool2d(enc4, 2)

        # Middle
        middle = self.middle(pool4)

        # Decoder
        dec4 = self.decoder4(torch.cat([F.interpolate(middle, scale_factor=2), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        # Final Layer
        segmentation = self.final_conv(dec1)  # 分割结果
        edge = self.edge_conv(dec1)  # 边缘图像
       
        return segmentation, edge

# Unet with depthwise conv & Dense return region & edge
class DenseBlock(nn.Module):
    def __init__(self, in_channels, growth_rate, num_layers):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            self.layers.append(self.conv_block(in_channels + i*growth_rate, growth_rate))

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    # def depthwise_conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    #     return nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding, groups=in_channels)

    # def pointwise_conv(in_channels, out_channels):
    #     return nn.Conv2d(in_channels, out_channels, kernel_size=1)

    # def conv_block(self, in_channels, out_channels):
    #     return nn.Sequential(
    #         depthwise_conv(in_channels, out_channels),
    #         pointwise_conv(in_channels, out_channels),
    #         nn.ReLU(inplace=True),
    #         depthwise_conv(out_channels, out_channels),
    #         pointwise_conv(out_channels, out_channels),
    #         nn.ReLU(inplace=True)
    #     )

    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, dim=1))
            features.append(new_features)
        return torch.cat(features, dim=1)

class Dense_Unet(nn.Module):
    def __init__(self):
        super(Dense_Unet, self).__init__()
        
        self.encoder1 = DenseBlock(1, 64, 2)  # input_channels, growth_rate, num_layers
        self.encoder2 = DenseBlock(128, 128, 2)
        self.encoder3 = DenseBlock(256, 256, 2)
        self.encoder4 = DenseBlock(512, 512, 2)
        
        self.middle = self.conv_block(1024, 1024)
        
        self.decoder4 = DenseBlock(1024+512, 512, 2)
        self.decoder3 = DenseBlock(512+256, 256, 2)
        self.decoder2 = DenseBlock(256+128, 128, 2)
        self.decoder1 = DenseBlock(128+64, 64, 2)
        
        self.final_conv = nn.Conv2d(128, 2, kernel_size=1)
        self.edge_conv = nn.Conv2d(128, 2, kernel_size=1)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(pool3)
        pool4 = F.max_pool2d(enc4, 2)

        middle = self.middle(pool4)

        dec4 = self.decoder4(torch.cat([F.interpolate(middle, scale_factor=2), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        segmentation = self.final_conv(dec1)
        edge = self.edge_conv(dec1)
        
        return segmentation, edge

# ZAM & WAM Attention Unet
class ZAM_WAM_Unet_4Layers(nn.Module):
    def __init__(self):
        super(ZAM_WAM_Unet_4Layers, self).__init__()
        
        self.encoder1 = self.conv_block(1, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        
        self.middle = self.conv_block(128, 256)
        
        self.decoder4 = self.conv_block(256, 128)
        self.decoder3 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = self.conv_block(32, 16)

        # 1x1 convolutions to reduce dimensions
        self.reduce_dim1 = nn.Conv2d(32, 16, kernel_size=1)
        self.reduce_dim2 = nn.Conv2d(64, 32, kernel_size=1)
        #self.reduce_dim3 = nn.Conv2d(64, 64, kernel_size=1)
        #self.reduce_dim4 = nn.Conv2d(128, 128, kernel_size=1)
        
        self.reduce_dim_dec4 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce_dim_dec3 = nn.Conv2d(128, 64, kernel_size=1)
        self.reduce_dim_dec2 = nn.Conv2d(64, 32, kernel_size=1)
        self.reduce_dim_dec1 = nn.Conv2d(32, 16, kernel_size=1)
        
        # ZAM and WAM modules
        self.zam3 = ZAM_Module(64)
        self.wam3 = WAM_Module(64)
        self.zam4 = ZAM_Module(128)
        self.wam4 = WAM_Module(128)

        # SE Blocks
        self.se3 = SE_Block(64)
        self.se4 = SE_Block(128)

        # Gabor Feature Extractors
        self.gabor1 = GaborFeatureExtractor(16, 16)
        self.gabor2 = GaborFeatureExtractor(32, 32)

        self.final_conv = nn.Conv2d(16, 2, kernel_size=1)  # 输出通道数为2，与Segmentation_onehot匹配
        
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
        # Encoder
        enc1 = self.encoder1(x)
        enc1 = torch.cat([self.gabor1(enc1), enc1], 1)
        enc1 = self.reduce_dim1(enc1)
        pool1 = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(pool1)
        enc2 = torch.cat([self.gabor2(enc2), enc2], 1)
        enc2 = self.reduce_dim2(enc2)
        pool2 = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(pool2)
        enc3 = enc3 + self.zam3(enc3) + self.wam3(enc3)
        enc3 = self.se3(enc3)
        #enc3 = self.reduce_dim3(enc3)
        pool3 = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(pool3)
        enc4 = enc4 + self.zam4(enc4) + self.wam4(enc4)
        enc4 = self.se4(enc4)
        #enc4 = self.reduce_dim4(enc4)
        pool4 = F.max_pool2d(enc4, 2)

        # Middle
        middle = self.middle(pool4)

        # Decoder
        dec4 = self.decoder4(F.interpolate(middle, scale_factor=2))
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.reduce_dim_dec4(dec4)

        dec3 = self.decoder3(F.interpolate(dec4, scale_factor=2))
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.reduce_dim_dec3(dec3)

        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce_dim_dec2(dec2)

        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.reduce_dim_dec1(dec1)

        segmentation = self.final_conv(dec1)

        return segmentation

# ZAM & WAM Attention Unet return region & edge
class Shape_ZAM_WAM_Unet(nn.Module):
    def __init__(self):
        super(Shape_ZAM_WAM_Unet, self).__init__()
        
        self.encoder1 = self.conv_block(1, 16)
        self.encoder2 = self.conv_block(16, 32)
        self.encoder3 = self.conv_block(32, 64)
        self.encoder4 = self.conv_block(64, 128)
        
        self.middle = self.conv_block(128, 256)
        
        self.decoder4 = self.conv_block(256, 128)
        self.decoder3 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = self.conv_block(32, 16)

        # 1x1 convolutions to reduce dimensions
        self.reduce_dim1 = nn.Conv2d(32, 16, kernel_size=1)
        self.reduce_dim2 = nn.Conv2d(64, 32, kernel_size=1)
        #self.reduce_dim3 = nn.Conv2d(64, 64, kernel_size=1)
        #self.reduce_dim4 = nn.Conv2d(128, 128, kernel_size=1)
        
        self.reduce_dim_dec4 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce_dim_dec3 = nn.Conv2d(128, 64, kernel_size=1)
        self.reduce_dim_dec2 = nn.Conv2d(64, 32, kernel_size=1)
        self.reduce_dim_dec1 = nn.Conv2d(32, 16, kernel_size=1)
        
        # ZAM and WAM modules
        self.zam3 = ZAM_Module(64)
        self.wam3 = WAM_Module(64)
        self.zam4 = ZAM_Module(128)
        self.wam4 = WAM_Module(128)

        # SE Blocks
        self.se3 = SE_Block(64)
        self.se4 = SE_Block(128)

        # Gabor Feature Extractors
        self.gabor1 = GaborFeatureExtractor(16, 16)
        self.gabor2 = GaborFeatureExtractor(32, 32)

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
         # Encoder
        enc1 = self.encoder1(x)
        enc1 = torch.cat([self.gabor1(enc1), enc1], 1)
        enc1 = self.reduce_dim1(enc1)
        pool1 = F.max_pool2d(enc1, 2)

        enc2 = self.encoder2(pool1)
        enc2 = torch.cat([self.gabor2(enc2), enc2], 1)
        enc2 = self.reduce_dim2(enc2)
        pool2 = F.max_pool2d(enc2, 2)

        enc3 = self.encoder3(pool2)
        enc3 = enc3 + self.zam3(enc3) + self.wam3(enc3)
        enc3 = self.se3(enc3)
        #enc3 = self.reduce_dim3(enc3)
        pool3 = F.max_pool2d(enc3, 2)

        enc4 = self.encoder4(pool3)
        enc4 = enc4 + self.zam4(enc4) + self.wam4(enc4)
        enc4 = self.se4(enc4)
        #enc4 = self.reduce_dim4(enc4)
        pool4 = F.max_pool2d(enc4, 2)

        # Middle
        middle = self.middle(pool4)

        # Decoder
        dec4 = self.decoder4(F.interpolate(middle, scale_factor=2))
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.reduce_dim_dec4(dec4)

        dec3 = self.decoder3(F.interpolate(dec4, scale_factor=2))
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.reduce_dim_dec3(dec3)

        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.reduce_dim_dec2(dec2)

        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.reduce_dim_dec1(dec1)

        segmentation = self.final_conv(dec1)  # 分割结果
        edge = self.edge_conv(dec1)  # 边缘图像
        
        return segmentation, edge

class CC_CV_Unet_4Layers(nn.Module):
    def __init__(self):
        super(CC_CV_Unet_4Layers, self).__init__()

        self.encoder1 = self.conv_block(1+2, 16)
        self.encoder2 = self.conv_block(16+2, 32)
        self.encoder3 = self.conv_block(32+2, 64)
        self.encoder4 = self.conv_block(64+2, 128)
        
        self.middle = self.conv_block(128+2, 256)
        
        self.decoder4 = self.conv_block(256, 128)
        self.decoder3 = self.conv_block(128, 64)
        self.decoder2 = self.conv_block(64, 32)
        self.decoder1 = self.conv_block(32, 16)

        # 1x1 convolutions to reduce dimensions
        # self.reduce_dim1 = nn.Conv2d(32, 16, kernel_size=1)
        # self.reduce_dim2 = nn.Conv2d(64, 32, kernel_size=1)
        # self.reduce_dim3 = nn.Conv2d(64, 64, kernel_size=1)
        # self.reduce_dim4 = nn.Conv2d(128, 128, kernel_size=1)
        
        self.reduce_dim_dec4 = nn.Conv2d(256, 128, kernel_size=1)
        self.reduce_dim_dec3 = nn.Conv2d(128, 64, kernel_size=1)
        self.reduce_dim_dec2 = nn.Conv2d(64, 32, kernel_size=1)
        self.reduce_dim_dec1 = nn.Conv2d(32, 16, kernel_size=1)
        
        # # ZAM and WAM modules
        # self.zam3 = ZAM_Module(64)
        # self.wam3 = WAM_Module(64)
        # self.zam4 = ZAM_Module(128)
        # self.wam4 = WAM_Module(128)

        # # SE Blocks
        # self.se3 = SE_Block(64)
        # self.se4 = SE_Block(128)

        # # Gabor Feature Extractors
        # self.gabor1 = GaborFeatureExtractor(16, 16)
        # self.gabor2 = GaborFeatureExtractor(32, 32)

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
        coords = self.coord_map(x.size(2), x.size(3)).to(x.device).unsqueeze(0)
        
        x_enc1 = torch.cat([x, coords.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc1 = self.encoder1(x_enc1)
        enc1_pool = F.max_pool2d(enc1, 2)
        
        coords_pooled_enc1 = self.coord_map(enc1_pool.size(2), enc1_pool.size(3)).to(x.device).unsqueeze(0)
        x_enc2 = torch.cat([enc1_pool, coords_pooled_enc1.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc2 = self.encoder2(x_enc2)
        enc2_pool = F.max_pool2d(enc2, 2)
        
        coords_pooled_enc2 = self.coord_map(enc2_pool.size(2), enc2_pool.size(3)).to(x.device).unsqueeze(0)
        x_enc3 = torch.cat([enc2_pool, coords_pooled_enc2.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc3 = self.encoder3(x_enc3)
        enc3 = self.se3(enc3)
        enc3_pool = F.max_pool2d(enc3, 2)
        
        coords_pooled_enc3 = self.coord_map(enc3_pool.size(2), enc3_pool.size(3)).to(x.device).unsqueeze(0)
        x_enc4 = torch.cat([enc3_pool, coords_pooled_enc3.repeat(x.size(0), 1, 1, 1)], dim=1)
        enc4 = self.encoder4(x_enc4)
        enc4 = self.se4(enc4)
        enc4_pool = F.max_pool2d(enc4, 2)

        coords_pooled_enc4 = self.coord_map(enc4_pool.size(2), enc4_pool.size(3)).to(x.device).unsqueeze(0)
        x_middle = torch.cat([enc4_pool, coords_pooled_enc4.repeat(x.size(0), 1, 1, 1)], dim=1)
        middle = self.middle(x_middle)

        dec4 = self.decoder4(F.interpolate(middle, scale_factor=2))
        dec4 = self.reduce_dim_dec4(torch.cat([dec4, enc4], dim=1))
        
        dec3 = self.decoder3(F.interpolate(dec4, scale_factor=2))
        dec3 = self.reduce_dim_dec3(torch.cat([dec3, enc3], dim=1))
        
        dec2 = self.decoder2(F.interpolate(dec3, scale_factor=2))
        dec2 = self.reduce_dim_dec2(torch.cat([dec2, enc2], dim=1))
        
        dec1 = self.decoder1(F.interpolate(dec2, scale_factor=2))
        dec1 = self.reduce_dim_dec1(torch.cat([dec1, enc1], dim=1))
        
        segmentation = self.final_conv(dec1)
        edge = self.edge_conv(dec1)
        
        return segmentation, edge

class CC_SP_Unet_4Layers(nn.Module):
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
