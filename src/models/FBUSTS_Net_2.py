from functools import reduce
import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional
import torchvision.models as models
from torch.autograd import Variable

import numpy as np
import mahotas

from torchvision.models import resnet50
from transformers import BertModel, BertConfig

from .FBUSTS_Net_Model import (
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
                                CCM_Module,
                                ImprovedZAM_Module,
                                ImprovedWAM_Module,
                                Channel_Zernike_Attention_Module,
                                Channel_Wavelet_Attention_Module
                               )


### State of Art Modules ###

# Unet
class Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Unet, self).__init__()
        
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
        
        return out, out

# ResUnet
class ResUNet (nn.Module):
    '''
        ResNet（残差网络）是一种通过残差连接来训练深度神经网络的结构，这使得训练过程更加容易。
        这个ResNet18模型包含了四个残差块层，每层有不同数量的残差块BasicBlock。
        每个残差块包含两个卷积层，这两个卷积层的输出和输入通过一个快捷连接（shortcut connection）相加。
    '''
    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv_block = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
            )
            self.final_relu = nn.ReLU(inplace=True)

        def forward(self, x):
            residual = self.shortcut(x)
            x = self.conv_block(x)
            x += residual
            return self.final_relu(x)
    
    def __init__(self, in_channels=1, out_channels=2):
        super(ResUNet, self).__init__()
        
        self.encoder1 = self.ResBlock(in_channels, 64)
        self.encoder2 = self.ResBlock(64, 128)
        self.encoder3 = self.ResBlock(128, 256)
        self.encoder4 = self.ResBlock(256, 512)
        
        self.middle = self.ResBlock(512, 1024)
        
        self.decoder4 = self.ResBlock(1024 + 512, 512)
        self.decoder3 = self.ResBlock(512 + 256, 256)
        self.decoder2 = self.ResBlock(256 + 128, 128)
        self.decoder1 = self.ResBlock(128 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, kernel_size=2, stride=2)

        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, kernel_size=2, stride=2)

        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, kernel_size=2, stride=2)

        enc4 = self.encoder4(pool3)
        pool4 = F.max_pool2d(enc4, kernel_size=2, stride=2)

        # Middle
        middle = self.middle(pool4)

        # Decoder
        up4 = F.interpolate(middle, scale_factor=2, mode='bilinear', align_corners=True)
        dec4 = self.decoder4(torch.cat([up4, enc4], dim=1))
        
        up3 = F.interpolate(dec4, scale_factor=2, mode='bilinear', align_corners=True)
        dec3 = self.decoder3(torch.cat([up3, enc3], dim=1))
        
        up2 = F.interpolate(dec3, scale_factor=2, mode='bilinear', align_corners=True)
        dec2 = self.decoder2(torch.cat([up2, enc2], dim=1))
        
        up1 = F.interpolate(dec2, scale_factor=2, mode='bilinear', align_corners=True)
        dec1 = self.decoder1(torch.cat([up1, enc1], dim=1))
        
        # Final Convolution
        out = self.final_conv(dec1)
        return out, out

# R2 UNet
class R2UNet(nn.Module):
    '''
        R2U-Net是U-Net的一种变体，它在每个卷积块中增加了递归卷积层。这些递归层可以多次重用同一权重，从而捕获更深层的信息。
    '''
    class RecurrentBlock(nn.Module):
        def __init__(self, in_channels, out_channels, t=2):
            super(R2UNet.RecurrentBlock, self).__init__()
            self.t = t
            self.in_channels = in_channels
            self.out_channels = out_channels
            self.conv = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.initial_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        def forward(self, x):
            # 如果输入通道数不等于输出通道数，需要一个转换层来匹配通道数
            if self.in_channels != self.out_channels:
                x = self.initial_conv(x)
            # 使用非原地操作进行累加
            for _ in range(self.t):
                x1 = self.conv(x)
                x1 = self.bn(x1)
                x1 = self.relu(x1)
                x = x + x1  # 用非原地操作替换原地操作
            return x

    class UpConv(nn.Module):
        def __init__(self, in_channels, out_channels, skip_channels):
            super(R2UNet.UpConv, self).__init__()
            self.up = nn.ConvTranspose2d(in_channels , out_channels, kernel_size=2, stride=2)
            # 注意：这里不再除以2，因为我们想要的通道数是out_channels加上skip_channels
            self.conv = R2UNet.RecurrentBlock(out_channels + skip_channels, out_channels, t=2)
            
        def forward(self, x1, x2):
            x1 = self.up(x1)
            diffY = x2.size()[2] - x1.size()[2]
            diffX = x2.size()[3] - x1.size()[3]
            x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2,
                            diffY // 2, diffY - diffY // 2))
            x = torch.cat([x2, x1], dim=1)  # 此时x将有out_channels + skip_channels个通道
            return self.conv(x)  # RecurrentBlock现在应该能处理这个通道数

    def __init__(self, in_channels=1, out_channels=2, t=2):
        super(R2UNet, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder1 = self.RecurrentBlock(in_channels, 64, t=t)
        self.encoder2 = self.RecurrentBlock(64, 128, t=t)
        self.encoder3 = self.RecurrentBlock(128, 256, t=t)
        self.encoder4 = self.RecurrentBlock(256, 512, t=t)

        self.intermediate = self.RecurrentBlock(512, 1024, t=t)

        # 修改UpConv调用以包括skip_channels参数
        self.upconv4 = self.UpConv(1024, 512, 512)
        self.upconv3 = self.UpConv(512, 256, 256)
        self.upconv2 = self.UpConv(256, 128, 128)
        self.upconv1 = self.UpConv(128, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.max_pool(enc1))
        enc3 = self.encoder3(self.max_pool(enc2))
        enc4 = self.encoder4(self.max_pool(enc3))

        # Intermediate
        inter = self.intermediate(self.max_pool(enc4))

        # Decoder
        dec4 = self.upconv4(inter, enc4)
        dec3 = self.upconv3(dec4, enc3)
        dec2 = self.upconv2(dec3, enc2)
        dec1 = self.upconv1(dec2, enc1)

        # Final Convolution
        out = self.final_conv(dec1)
        return out, out

# KiUNet
class KiUNet(nn.Module):
    '''
        KiU-Net是一种针对医学图像分割设计的网络，它由两个并行的U-Net架构组成，一个针对粗略特征，另一个针对细节特征。
    '''
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
            super(KiUNet.ConvBlock, self).__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
            self.bn = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            return self.relu(self.bn(self.conv(x)))

    class UpConv(nn.Module):
        def __init__(self, in_channels, out_channels):
            super(KiUNet.UpConv, self).__init__()
            self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

        def forward(self, x):
            return self.upconv(x)

    def __init__(self, in_channels=1, out_channels=2):
        super(KiUNet, self).__init__()

        # Encoder for the large features
        self.enc1 = self.ConvBlock(in_channels, 64)
        self.enc2 = self.ConvBlock(64, 128)
        self.enc3 = self.ConvBlock(128, 256)
        self.enc4 = self.ConvBlock(256, 512)

        # Encoder for the fine details
        self.enc1_fine = self.ConvBlock(in_channels, 64, kernel_size=1, padding=0)
        self.enc2_fine = self.ConvBlock(64, 128, kernel_size=1, padding=0)
        self.enc3_fine = self.ConvBlock(128, 256, kernel_size=1, padding=0)
        self.enc4_fine = self.ConvBlock(256, 512, kernel_size=1, padding=0)

        # Bottleneck
        self.bottleneck = self.ConvBlock(512, 1024)
        self.bottleneck_fine = self.ConvBlock(512, 1024, kernel_size=1, padding=0)

        # Decoder for the large features
        self.upconv4 = self.UpConv(1024, 512)
        self.dec4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConv(512, 256)
        self.dec3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConv(256, 128)
        self.dec2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConv(128, 64)
        self.dec1 = self.ConvBlock(128, 64)

        # Decoder for the fine details
        self.upconv4_fine = self.UpConv(1024, 512)
        self.dec4_fine = self.ConvBlock(1024, 512, kernel_size=1, padding=0)
        self.upconv3_fine = self.UpConv(512, 256)
        self.dec3_fine = self.ConvBlock(512, 256, kernel_size=1, padding=0)
        self.upconv2_fine = self.UpConv(256, 128)
        self.dec2_fine = self.ConvBlock(256, 128, kernel_size=1, padding=0)
        self.upconv1_fine = self.UpConv(128, 64)
        self.dec1_fine = self.ConvBlock(128, 64, kernel_size=1, padding=0)

        # Final output layer for the combined features
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path for the large features
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        e3 = self.enc3(F.max_pool2d(e2, 2))
        e4 = self.enc4(F.max_pool2d(e3, 2))

        # Encoder path for the fine details
        e1_fine = self.enc1_fine(x)
        e2_fine = self.enc2_fine(F.max_pool2d(e1_fine, 2))
        e3_fine = self.enc3_fine(F.max_pool2d(e2_fine, 2))
        e4_fine = self.enc4_fine(F.max_pool2d(e3_fine, 2))

        # Bottleneck
        b = self.bottleneck(F.max_pool2d(e4, 2))
        b_fine = self.bottleneck_fine(F.max_pool2d(e4_fine, 2))

        # Decoder path for the large features
        d4 = torch.cat((self.upconv4(b), e4), dim=1)
        d4 = self.dec4(d4)
        d3 = torch.cat((self.upconv3(d4), e3), dim=1)
        d3 = self.dec3(d3)
        d2 = torch.cat((self.upconv2(d3), e2), dim=1)
        d2 = self.dec2(d2)
        d1 = torch.cat((self.upconv1(d2), e1), dim=1)
        d1 = self.dec1(d1)

        # Decoder path for the fine details
        d4_fine = torch.cat((self.upconv4_fine(b_fine), e4_fine), dim=1)
        d4_fine = self.dec4_fine(d4_fine)
        d3_fine = torch.cat((self.upconv3_fine(d4_fine), e3_fine), dim=1)
        d3_fine = self.dec3_fine(d3_fine)
        d2_fine = torch.cat((self.upconv2_fine(d3_fine), e2_fine), dim=1)
        d2_fine = self.dec2_fine(d2_fine)
        d1_fine = torch.cat((self.upconv1_fine(d2_fine), e1_fine), dim=1)
        d1_fine = self.dec1_fine(d1_fine)

        # Combine the large features and fine details
        combined = d1 + d1_fine

        # Final output layer
        out = self.final_conv(combined)
        return out, out

# TransUNet
class TransUNet(nn.Module):
    '''
        TransUNet 结合了 Transformer 和 U-Net 架构，以利用 Transformer 在处理长距离依赖关系方面的优势，并且结合了 U-Net 在捕获局部空间信息方面的能力。
        在这个实现中，我们用 ResNet50 作为特征提取的 backbone，然后将最后一层特征图通过 Transformer 编码器进行处理，最终通过一系列上采样层和卷积层来重建图像。
    '''
    def __init__(self, in_channels=1, out_channels=2, img_dim=224, vit_blocks=12, vit_heads=12, vit_dim_linear_mhsa_block=3072):
        super(TransUNet, self).__init__()

        # ResNet50 backbone
        self.resnet = resnet50(pretrained=True)
        self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Adjust resnet fc layer
        self.resnet.fc = nn.Identity()

        # Transformer blocks
        self.config = BertConfig(
            hidden_size=2048,
            num_hidden_layers=vit_blocks,
            num_attention_heads=vit_heads,
            intermediate_size=vit_dim_linear_mhsa_block,
            hidden_dropout_prob=0.0,
            attention_probs_dropout_prob=0.0,
            image_size=img_dim,
            patch_size=16,
            num_channels=2048
        )
        self.transformer = BertModel(self.config)

        # Decoder part
        self.decoder4 = self.conv_block(2048 + 1024, 512)
        self.decoder3 = self.conv_block(512 + 512, 256)
        self.decoder2 = self.conv_block(256 + 256, 128)
        self.decoder1 = self.conv_block(128 + 64, 64)
        
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # Feature extraction part
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)

        enc1 = self.resnet.layer1(x)
        enc2 = self.resnet.layer2(enc1)
        enc3 = self.resnet.layer3(enc2)
        enc4 = self.resnet.layer4(enc3)

        # Flattening and transformer part
        n, c, h, w = enc4.shape
        x = enc4.view(n, c, -1).transpose(-1, -2)  # Reshape for transformer
        x = self.transformer(inputs_embeds=x)['last_hidden_state']
        x = x.transpose(-1, -2).contiguous().view(n, c, h, w)  # Reshape back to feature map shape

        # Decoder
        dec4 = self.decoder4(torch.cat([F.interpolate(x, scale_factor=2), enc4], dim=1))
        dec3 = self.decoder3(torch.cat([F.interpolate(dec4, scale_factor=2), enc3], dim=1))
        dec2 = self.decoder2(torch.cat([F.interpolate(dec3, scale_factor=2), enc2], dim=1))
        dec1 = self.decoder1(torch.cat([F.interpolate(dec2, scale_factor=2), enc1], dim=1))

        # Final Convolution
        out = self.final_conv(dec1)
        
        return out

# MedT
class MedT(nn.Module):
    '''
        MedT (Medical Transformer) 是一个针对医学图像分割的Transformer模型，通常包含一系列的局部（Local）和全局（Global）Transformer层。
        这个实现中，LocalTransformer 和 GlobalTransformer 分别实现了局部和全局的Transformer层。
        这个模型从图像中创建图像块（patches），然后将这些图像块转换为一系列线性嵌入（embeddings），再通过Transformer层进行处理。最后，解码器通过一系列转置卷积层（transpose convolutions）重建图像。
    '''
    class LocalTransformer(nn.Module):
        def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
            super().__init__()
            layers = []
            for _ in range(depth):
                layers.append(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            x = x.flatten(2).permute(2, 0, 1)  # Convert to seq_len, batch, channels
            x = self.net(x)
            x = x.permute(1, 2, 0).unflatten(2, (int(x.size(1)**0.5), int(x.size(1)**0.5)))  # Convert back to feature map
            return x

    class GlobalTransformer(nn.Module):
        def __init__(self, dim, depth, heads, mlp_dim, dropout = 0.):
            super().__init__()
            self.net = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, dropout=dropout), num_layers=depth)

        def forward(self, x):
            x = x.flatten(2).permute(2, 0, 1)  # Convert to seq_len, batch, channels
            x = self.net(x)
            x = x.permute(1, 2, 0).unflatten(2, (int(x.size(1)**0.5), int(x.size(1)**0.5)))  # Convert back to feature map
            return x

    def __init__(self):
        super().__init__()
        # 固定参数
        img_dim = 128
        in_channels = 1
        out_channels = 2
        patch_size = 16
        dim = 512
        depth = 6
        heads = 8
        mlp_dim = 1024
        channels = 1024
        dropout = 0.1

        self.patch_size = patch_size
        self.dim = dim
        self.channels = channels
        num_patches = (img_dim // patch_size) ** 2

        self.patch_to_embedding = nn.Linear(patch_size*patch_size*in_channels, dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches+1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(dropout)

        self.local_transformer = self.LocalTransformer(dim, depth, heads, mlp_dim, dropout)
        self.global_transformer = self.GlobalTransformer(dim, depth, heads, mlp_dim, dropout)

        self.to_cls_token = nn.Identity()

        self.decoder1 = nn.ConvTranspose2d(dim, channels // 2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder2 = nn.ConvTranspose2d(channels // 2, channels // 4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder3 = nn.ConvTranspose2d(channels // 4, channels // 8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.decoder4 = nn.ConvTranspose2d(channels // 8, channels // 16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.final_conv = nn.Conv2d(channels // 16, out_channels, kernel_size=1)

    def forward(self, x):
        n, c, h, w = x.shape
        x = F.unfold(x, kernel_size=self.patch_size, stride=self.patch_size)
        x = x.transpose(1, 2)
        x = self.patch_to_embedding(x)
        cls_tokens = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.dropout(x)

        x = self.local_transformer(x)
        x = self.global_transformer(x)

        x = x[:, 1:, :]
        x = x.transpose(1, 2)
        x = F.fold(x, output_size=(h, w), kernel_size=self.patch_size, stride=self.patch_size)

        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.decoder3(x)
        x = self.decoder4(x)
        x = self.final_conv(x)

        return x

# Channel UNet
class ChannelUNet(nn.Module):
    '''
        Channel U-Net是U-Net的一个变种，它在U-Net的基础上增加了通道注意力机制。这个机制允许网络在处理输入时对不同的通道给予不同的注意力权重。
    '''
    class ChannelAttention(nn.Module):
        def __init__(self, in_channels, ratio=8):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc1(self.avg_pool(x))
            avg_out = self.relu1(avg_out)
            avg_out = self.fc2(avg_out)
            max_out = self.fc1(self.max_pool(x))
            max_out = self.relu1(max_out)
            max_out = self.fc2(max_out)
            out = avg_out + max_out
            return self.sigmoid(out)

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.ca = ChannelUNet.ChannelAttention(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.ca(x) * x
            return x

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels, skip_channels):
            super().__init__()
            # The input channels for the ConvTranspose2d should be the same as the output
            # channels from the corresponding level in the encoder part of the network.
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            # After concatenation, the number of channels will be `out_channels + skip_channels`.
            # This total becomes the input channels for the next ConvBlock.
            self.conv = ChannelUNet.ConvBlock(out_channels + skip_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            # Before concatenation, make sure the dimensions match with the skip connection.
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            x = torch.cat([x, skip], dim=1)
            return self.conv(x)



    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = self.ConvBlock(512, 1024)

        self.upconv4 = self.UpConvBlock(1024, 512, 512)
        self.upconv3 = self.UpConvBlock(512, 256, 256)
        self.upconv2 = self.UpConvBlock(256, 128, 128)
        self.upconv1 = self.UpConvBlock(128, 64, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        e1 = self.encoder1(x)
        p1 = self.pool1(e1)
        
        e2 = self.encoder2(p1)
        p2 = self.pool2(e2)
        
        e3 = self.encoder3(p2)
        p3 = self.pool3(e3)
        
        e4 = self.encoder4(p3)
        p4 = self.pool4(e4)

        # Intermediate (Bottleneck)
        center = self.center(p4)

        # Decoder
        u4 = self.upconv4(center, e4)
        u3 = self.upconv3(u4, e3)
        u2 = self.upconv2(u3, e2)
        u1 = self.upconv1(u2, e1)

        # Final output
        out = self.final_conv(u1)
        return out, out

# Attention Unet
class AttentionUnet(nn.Module):
    '''
        Attention U-Net 通过加入注意力门（Attention Gates, AGs）来聚焦于特定的图像区域以提高模型性能。
    '''
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = AttentionUnet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            # Input is CHW
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = self.ConvBlock(512, 1024)

        # Decoder
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = self.AttentionGate(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = self.AttentionGate(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = self.AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)

        # Decoder path with attention gates
        g4 = self.upconv4(center, x4)
        x4 = self.attention4(g=g4, x=x4)
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        out = self.final_conv(d1)
        return out, out


### Single Target Shape Attention Modules ###
## Old ##

# Unet return region & edge
class Shape_Unet(nn.Module):
    def __init__(self, in_channels=1, out_channels=2):
        super(Shape_Unet, self).__init__()

        # Encoder
        self.encoder1 = self.conv_block(in_channels, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)
        # self.dropout1 = nn.Dropout(0.5)

        # Middle
        self.middle = self.conv_block(512, 1024)
        # self.dropout2 = nn.Dropout(0.5)

        # Decoder
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)

        # self.final_conv = self.out_layer(64, out_channels)
        # self.edge_conv = self.out_layer(64, out_channels)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)
        self.edge_conv = nn.Conv2d(64, out_channels, kernel_size=1, padding=0)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    
    def out_layer(self, in_channels, out_channels):
        # 这个函数定义了输出层的结构
        layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
        return layers

    def forward(self, x):
        # Encoder
        enc1 = self.encoder1(x)
        pool1 = F.max_pool2d(enc1, 2)
        
        enc2 = self.encoder2(pool1)
        pool2 = F.max_pool2d(enc2, 2)
        
        enc3 = self.encoder3(pool2)
        pool3 = F.max_pool2d(enc3, 2)
        
        enc4 = self.encoder4(pool3)
        # enc4 = self.dropout1(enc4)
        pool4 = F.max_pool2d(enc4, 2)
        
        # Middle
        middle = self.middle(pool4)
        # middle = self.dropout2(middle)

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
        
        # Middle
        self.middle = self.conv_block(512, 1024)

        # Decoder
        self.decoder4 = self.conv_block(1024+512, 512)
        self.decoder3 = self.conv_block(512+256, 256)
        self.decoder2 = self.conv_block(256+128, 128)
        self.decoder1 = self.conv_block(128+64, 64)

        # Attention
        self.ZAM = ZAM_Module(512)
        self.WAM = WAM_Module(512)

        # Dropout
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

        self.final_conv = self.out_layer(64, out_channels)
        self.edge_conv = self.out_layer(64, out_channels)

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
    def out_layer(self, in_channels, out_channels):
        # 这个函数定义了输出层的结构
        layers = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1)
        )
        return layers

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

        # Concatenate all encoder outputs
        concat_enc = torch.cat([enc1, enc2, enc3, enc4], dim=1)
        concat_enc_ZAM = self.ZAM(F.interpolate(concat_enc, scale_factor=2))   # Zernike Attention after concatenation
        concat_enc_WAM = self.WAM(F.interpolate(concat_enc, scale_factor=2))   # Wavelet Attention before the first decoder block

        # Middle
        middle = self.middle(pool4)
        middle = self.dropout1(middle)
        middle_ = torch.cat([middle, concat_enc, concat_enc_ZAM, concat_enc_WAM], dim=1)

        # Decoder
        dec4 = self.decoder4(torch.cat([F.interpolate(middle_, scale_factor=2), enc4], dim=1))
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

## New ##

# Improved ZAM & WAM Unet return region & edge & 512*512 region
class Shape_Improved_ZAM_WAM_Unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Shape_Improved_ZAM_WAM_Unet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = self.ConvBlock(512, 1024)

        # Decoder with improved attention gates
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = ImprovedZAM_Module(F_g=512, F_l=512, F_int=256)  # 使用ImprovedZAM_Module
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = ImprovedWAM_Module(F_g=256, F_l=256, F_int=128)  # 使用ImprovedWAM_Module
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = ImprovedZAM_Module(F_g=128, F_l=128, F_int=64)   # 使用ImprovedZAM_Module
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        # self.attention1 = ImprovedWAM_Module(F_g=64, F_l=64, F_int=32)   # 使用ImprovedZAM_Module
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.edge_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)

        # Decoder path with improved attention gates
        g4 = self.upconv4(center, x4)
        x4 = self.attention4(g=g4, x=x4)  # 使用ImprovedZAM_Module
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)  # 使用ImprovedWAM_Module
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)  # 使用ImprovedZAM_Module
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        # x1 = self.attention1(g=g1, x=x1)  # 使用ImprovedWAM_Module
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region = self.final_conv(d1)
        edge = self.edge_conv(d1)

        region_256 = self.upsample1(region)
        region_512 = self.upsample1(region_256)

        return region, edge, region_512

# Improved ZAM & WAM Unet with Channel Attention return region & edge & 512*512 region
class Channel_Improved_ZAM_WAM_Unet(nn.Module):

    class ChannelAttention(nn.Module):
        def __init__(self, in_channels, ratio=8):
            super().__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.max_pool = nn.AdaptiveMaxPool2d(1)
            self.fc1 = nn.Conv2d(in_channels, in_channels // ratio, 1, bias=False)
            self.relu1 = nn.ReLU()
            self.fc2 = nn.Conv2d(in_channels // ratio, in_channels, 1, bias=False)
            self.sigmoid = nn.Sigmoid()

        def forward(self, x):
            avg_out = self.fc1(self.avg_pool(x))
            avg_out = self.relu1(avg_out)
            avg_out = self.fc2(avg_out)
            max_out = self.fc1(self.max_pool(x))
            max_out = self.relu1(max_out)
            max_out = self.fc2(max_out)
            out = avg_out + max_out
            return self.sigmoid(out)
        
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
            self.ca = Channel_Improved_ZAM_WAM_Unet.ChannelAttention(out_channels)

        def forward(self, x):
            x = self.conv(x)
            x = self.ca(x) * x  # 应用Channel Attention
            return x

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Channel_Improved_ZAM_WAM_Unet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = self.ConvBlock(512, 1024)

        # Decoder with improved attention gates
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = ImprovedZAM_Module(F_g=512, F_l=512, F_int=256)  # 使用ImprovedZAM_Module
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = ImprovedWAM_Module(F_g=256, F_l=256, F_int=128)  # 使用ImprovedWAM_Module
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = ImprovedZAM_Module(F_g=128, F_l=128, F_int=64)   # 使用ImprovedZAM_Module
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.edge_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)

        # Decoder path with improved attention gates
        g4 = self.upconv4(center, x4)
        x4 = self.attention4(g=g4, x=x4)  # 使用ImprovedZAM_Module
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)  # 使用ImprovedWAM_Module
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)  # 使用ImprovedZAM_Module
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region = self.final_conv(d1)
        edge = self.edge_conv(d1)

        region_256 = self.upsample1(region)
        region_512 = self.upsample1(region_256)

        return region, edge, region_512

# Channel ZAM & WAM Unet return region & edge & 512*512 region
class Channel_ZAM_WAM_Unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Channel_ZAM_WAM_Unet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=2):
        super().__init__()
        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.center = self.ConvBlock(512, 1024)

        # Decoder with improved attention gates
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = Channel_Zernike_Attention_Module(F_g=512, F_l=512, F_int=256)  # 使用ImprovedZAM_Module
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = Channel_Wavelet_Attention_Module(F_g=256, F_l=256, F_int=128)  # 使用ImprovedWAM_Module
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = Channel_Zernike_Attention_Module(F_g=128, F_l=128, F_int=64)   # 使用ImprovedZAM_Module
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.edge_conv = nn.Conv2d(64, out_channels, kernel_size=1)

        self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x):
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)

        # Decoder path with channel attention gates
        g4 = self.upconv4(center, x4)
        x4 = self.attention4(g=g4, x=x4)  # 使用Channel_Zernike_Attention_Module
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)  # 使用Channel_Wavelet_Attention_Module
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)  # 使用Channel_Zernike_Attention_Module
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region = self.final_conv(d1)
        edge = self.edge_conv(d1)

        region_256 = self.upsample1(region)
        region_512 = self.upsample1(region_256)

        return region, edge, region_512

# Week Feature + Improved ZAM & WAM Unet return region & edge & 512*512 region
class Week_Improved_ZAM_WAM_Unet(nn.Module):
    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Shape_Improved_ZAM_WAM_Unet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=2, num_weeks=15):
        super().__init__()
        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.week_fc = nn.Linear(num_weeks + 1, 1024)
        # 假设center的通道数为1024
        self.reduce_channel_conv = nn.Conv2d(2048, 1024, kernel_size=1)  # 1x1卷积减半通道数

        self.center = self.ConvBlock(512, 1024)

        # Decoder with improved attention gates
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = ImprovedZAM_Module(F_g=512, F_l=512, F_int=256)  # 使用ImprovedZAM_Module
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = ImprovedWAM_Module(F_g=256, F_l=256, F_int=128)  # 使用ImprovedWAM_Module
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = ImprovedZAM_Module(F_g=128, F_l=128, F_int=64)   # 使用ImprovedZAM_Module
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.week_predictor = nn.Linear(1024, num_weeks + 1) 

        # self.upsample1 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        # self.upsample2 = nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
    
    def forward(self, x, week):
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)

        # 将周数映射到新的编码上
        week_encoded = torch.zeros(x.size(0), 16)
        week_encoded[week == 0, 0] = 1  # 对于周数为0的情况 d
        for i in range(18, 33):
            week_encoded[week == i, i - 17] = 1  # 对于18-32周

        week_feature = self.week_fc(week_encoded.float()) # 通过全连接层得到周数特征
        week_feature = week_feature.view(-1, 1024, 1, 1)
        week_feature = week_feature.expand(-1, -1, center.size(2), center.size(3))

        # 融合周数信息到中间层
        center = torch.cat((center, week_feature), dim=1)  # 将周数特征图和中心特征图在通道维度上拼接
        center = self.reduce_channel_conv(center) # 使用1x1卷积减半通道数，回到1024

        # Decoder path with improved attention gates
        g4 = self.upconv4(center, x4)
        x4 = self.attention4(g=g4, x=x4)  # 使用ImprovedZAM_Module
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)  # 使用ImprovedWAM_Module
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)  # 使用ImprovedZAM_Module
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))
        
        # Final output
        region = self.final_conv(d1)
        # region_256 = self.upsample1(region)
        # region_512 = self.upsample1(region_256)

        # 周数预测
        # 采用AdaptiveAvgPool2d将特征图降维到1x1，然后进行周数预测
        week_pred = F.adaptive_avg_pool2d(center, 1)
        week_pred = week_pred.view(week_pred.size(0), -1)  # Flatten
        week_pred = self.week_predictor(week_pred)  # 周数预测

        return region, week_pred

# Week Feature + Attention Unet + onehot
class Week_Improved_AttentionUnet(nn.Module):

    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Week_Improved_AttentionUnet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            # Input is CHW
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    def __init__(self, in_channels=1, out_channels=2, num_weeks = 15):
        super(Week_Improved_AttentionUnet, self).__init__()
        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Week Encoder
        self.week_fc = nn.Linear(num_weeks+ 1, 1024)
        self.reduce_channel_conv = nn.Conv2d(2048, 1024, kernel_size=1)

        self.center = self.ConvBlock(512, 1024)
        
        # Decoder
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = self.AttentionGate(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = self.AttentionGate(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = self.AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.week_predictor = nn.Linear(1024, num_weeks+1) 

    def forward(self, x, week):
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)
        
        # 将周数映射到新的编码上
        week_encoded = torch.zeros(x.size(0), 16)
        week_encoded[week == 0, 0] = 1  # 对于周数为0的情况 d
        for i in range(18, 33):
            week_encoded[week == i, i - 17] = 1  # 对于18-32周

        week_feature = self.week_fc(week_encoded.float()) # 通过全连接层得到周数特征
        week_feature = week_feature.view(-1, 1024, 1, 1)
        week_feature = week_feature.expand(-1, -1, center.size(2), center.size(3))

        # 融合周数特征
        center = torch.cat((center, week_feature), dim=1)
        center = self.reduce_channel_conv(center)

        # Decoder path with attention gates
        g4 = self.upconv4(center, x4)
        x4 = self.attention4(g=g4, x=x4)
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region_pred = self.final_conv(d1)
        
        # Week output
        # 采用AdaptiveAvgPool2d将特征图降维到1x1，然后进行周数预测
        week_pred = F.adaptive_avg_pool2d(center, 1)
        week_pred = week_pred.view(week_pred.size(0), -1)  # Flatten
        week_pred = self.week_predictor(week_pred)  # 周数预测

        return region_pred, week_pred # 根据你的需求定义最终输出

class Week_Feature_AttentionUnet(nn.Module):
    
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Week_Feature_AttentionUnet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            # Input is CHW
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)
    
    class ResNetFeatureExtractor(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(resnet.fc.in_features, output_size)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    @staticmethod
    def extract_glcm_features(image_tensor, levels=256):
        """
        Extract GLCM texture features from an image using mahotas library.

        :param image_tensor: torch.Tensor, a 2D grayscale image tensor
        :param levels: int, number of gray-levels in the image
        :return: 1D array of texture features
        """
        # 将 torch.FloatTensor 转换为 numpy 数组，并缩放到 0-255 范围
        image = image_tensor.squeeze().cpu().numpy()
        image = (image * (levels - 1)).astype(np.uint8)

        # 使用 mahotas 计算 GLCM 特征
        features = mahotas.features.haralick(image).mean(axis=0)  # 计算每个方向的平均值

        return features
    
    class WeekAttentionModule(nn.Module):
        def __init__(self, channel_center, channel_global):
            super().__init__()
            self.attention_layer = nn.Sequential(
                nn.Conv2d(channel_center + channel_global, channel_center, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channel_center, 1, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, center, combined_features_with_week):
            combined_features_with_week = combined_features_with_week.unsqueeze(2).unsqueeze(3)
            combined_features_with_week = combined_features_with_week.repeat(1, 1, center.shape[2], center.shape[3])
            combined = torch.cat([center, combined_features_with_week], dim=1)
            attention_weights = self.attention_layer(combined)
            attended_center = center * attention_weights
            return attended_center
    
    # 添加新的初始化参数
    def __init__(self, in_channels=1, out_channels=2, num_weeks=15, texture_feature_size=13, deep_feature_size=512, reduced_feature_size=1024):#256
        super(Week_Feature_AttentionUnet, self).__init__()
        
        # 新的特征提取器和处理器
        self.texture_feature_processor = nn.Linear(texture_feature_size, 256)
        self.deep_feature_extractor = self.ResNetFeatureExtractor(deep_feature_size)
        self.feature_fusion = nn.Linear(256 + deep_feature_size, 1024)
        # 特征降维层（模拟PCA效果）
        self.feature_reduction = nn.Linear(texture_feature_size + deep_feature_size, reduced_feature_size)
        # 特征和周数融合层
        self.feature_and_week_fusion = nn.Linear(272, 1024)

        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Center
        self.center = self.ConvBlock(512, 1024)
        self.WAM_center = self.WeekAttentionModule(1024, 1024)
        
        # Decoder
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = self.AttentionGate(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = self.AttentionGate(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = self.AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.week_predictor = nn.Linear(1024, num_weeks+1) 

    def forward(self, x, week):
        
        # 对每个图像提取纹理特征
        batch_texture_features = [self.extract_glcm_features(img) for img in x]
        texture_features = torch.stack([torch.tensor(feat).float().to(x.device) for feat in batch_texture_features])
        
        # 提取深度特征
        x_3 = x.repeat(1, 3, 1, 1)
        deep_features = self.deep_feature_extractor(x_3)

        # 合并特征并降维
        combined_features = torch.cat([texture_features, deep_features], dim=1)
        reduced_features = self.feature_reduction(combined_features)

        # 周数编码（根据您之前的方式）
        week_encoded = torch.zeros(x.size(0), 16)
        week_encoded[week == 0, 0] = 1  # 对于周数为0的情况
        for i in range(18, 33):
            week_encoded[week == i, i - 17] = 1  # 对于18-32周

        # 融合降维特征和周数
        combined_features_with_week = torch.cat([reduced_features, week_encoded], dim=1)
        combined_features_with_week = self.feature_and_week_fusion(combined_features_with_week)

        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)
        
        # 融合周数特征
        attended_center = self.WAM_center(center, combined_features_with_week)
        # attended_center = self.WAM_center(center, reduced_features)
        # center = torch.cat((center, combined_features_with_week), dim=1)
        # attended_center = self.reduce_channel_conv(attended_center)

        # Decoder path with attention gates
        g4 = self.upconv4(attended_center, x4)
        x4 = self.attention4(g=g4, x=x4)
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region_pred = self.final_conv(d1)
        
        # Week output
        week_pred = F.adaptive_avg_pool2d(attended_center, 1)
        week_pred = week_pred.view(week_pred.size(0), -1)  # Flatten
        week_pred = self.week_predictor(week_pred)

        return region_pred, week_pred

class Modified_Feature_AttentionUnet(nn.Module):
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )

            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )

            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Modified_Feature_AttentionUnet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            # Input is CHW
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)

    class SpatialTransformerNetwork(nn.Module):
        def __init__(self):
            super().__init__()
            self.localization = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=7, padding=3),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5, padding=2),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True)
            )

            # 调整第一个全连接层的输入特征数
            self.fc_loc = nn.Sequential(
                nn.Linear(10240, 16),
                nn.ReLU(True),
                nn.Linear(16, 3 * 2)
            )

            # 初始化权重/偏置为恒等变换
            self.fc_loc[2].weight.data.zero_()
            self.fc_loc[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

        def stn(self, x):
            xs = self.localization(x)
            num_features = xs.size(1) * xs.size(2) * xs.size(3)
            xs = xs.view(-1, num_features)
            theta = self.fc_loc(xs)
            theta = theta.view(-1, 2, 3)

            grid = F.affine_grid(theta, x.size())
            x = F.grid_sample(x, grid)

            return x

        def forward(self, x):
            return self.stn(x)

    class ResNetFeatureExtractor(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(resnet.fc.in_features, output_size)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    class MultiScaleAttention(nn.Module):
            # 多尺度注意力机制
            def __init__(self, F_g, F_l, F_int):
                super().__init__()
                self.W_g = nn.Sequential(
                    nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(F_int),
                    nn.ReLU(inplace=True)
                )

                self.W_x = nn.Sequential(
                    nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(F_int),
                    nn.ReLU(inplace=True)
                )

                self.psi = nn.Sequential(
                    nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                    nn.BatchNorm2d(1),
                    nn.Sigmoid()
                )

            def forward(self, g, x):
                g1 = self.W_g(g)
                x1 = self.W_x(x)
                psi = self.psi(g1 + x1)
                return x * psi
    
    class WeekAttentionModule(nn.Module):
        def __init__(self, channel_center, channel_global):
            super().__init__()
            self.attention_layer = nn.Sequential(
                nn.Conv2d(channel_center + channel_global, channel_center, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channel_center, 1, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, center, combined_features_with_week):
            combined_features_with_week = combined_features_with_week.unsqueeze(2).unsqueeze(3)
            combined_features_with_week = combined_features_with_week.repeat(1, 1, center.shape[2], center.shape[3])
            combined = torch.cat([center, combined_features_with_week], dim=1)
            attention_weights = self.attention_layer(combined)
            attended_center = center * attention_weights
            return attended_center
    
    @staticmethod
    def extract_glcm_features(image_tensor, levels=256):
        """
        Extract GLCM texture features from an image using mahotas library.

        :param image_tensor: torch.Tensor, a 2D grayscale image tensor
        :param levels: int, number of gray-levels in the image
        :return: 1D array of texture features
        """
        # 确保不跟踪梯度
        image = image_tensor.squeeze().detach().cpu().numpy()
        image = (image * (levels - 1)).astype(np.uint8)

        # 使用 mahotas 计算 GLCM 特征
        features = mahotas.features.haralick(image).mean(axis=0)

        return features

    def __init__(self, in_channels=1, out_channels=2, num_weeks=15, texture_feature_size=13, deep_feature_size=512, reduced_feature_size=256):
        super(Modified_Feature_AttentionUnet, self).__init__()

        # 初始化空间变换网络
        self.stn = self.SpatialTransformerNetwork()

        # 特征提取器和处理器
        self.texture_feature_processor = nn.Linear(texture_feature_size, 256)
        self.deep_feature_extractor = self.ResNetFeatureExtractor(deep_feature_size)

        # 新增多尺度注意力层
        self.multi_scale_attention = self.MultiScaleAttention(512, 512, 256)

        # 动态特征融合层
        self.dynamic_feature_fusion = nn.Linear(deep_feature_size + reduced_feature_size, reduced_feature_size)

        # 条件特征融合层
        self.conditional_feature_fusion = nn.Linear(reduced_feature_size, 1024)

        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Center
        self.center = self.ConvBlock(512, 1024)
        # 初始化WeekAttentionModule
        self.WAM_center = self.WeekAttentionModule(1024, 1024)

        # Decoder
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = self.AttentionGate(F_g=512, F_l=512, F_int=256)
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = self.AttentionGate(F_g=256, F_l=256, F_int=128)
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = self.AttentionGate(F_g=128, F_l=128, F_int=64)
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        # Final output
        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.week_predictor = nn.Linear(1024, num_weeks+1)

    def forward(self, x, week):
        # 应用空间变换网络
        transformed_x = self.stn(x)

        # 特征提取
        texture_features = torch.stack([self.texture_feature_processor(torch.from_numpy(self.extract_glcm_features(img)).float().to(x.device)) for img in transformed_x])
        deep_features = self.deep_feature_extractor(transformed_x.repeat(1, 3, 1, 1))

        # 动态特征融合
        combined_features = torch.cat([texture_features, deep_features], dim=1)
        combined_features = self.dynamic_feature_fusion(combined_features)

        # 条件特征融合
        # 周数编码
        week_encoded = torch.zeros(x.size(0), 16)
        week_encoded[week == 0, 0] = 1  # 对于周数为0的情况
        for i in range(18, 33):
            week_encoded[week == i, i - 17] = 1  # 对于18-32周
        combined_features_with_week = torch.cat([combined_features, week_encoded], dim=1)
        combined_features_with_week = self.conditional_feature_fusion(combined_features)

        # Encoder path
        x1 = self.encoder1(transformed_x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        # Center
        center = self.center(p4)

        # 应用WeekAttentionModule
        attended_center = self.WAM_center(center, combined_features_with_week)

        # Decoder path with attention gates
        g4 = self.upconv4(attended_center, x4)
        x4 = self.multi_scale_attention(g=g4, x=x4) # 应用多尺度注意力机制
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3)
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2)
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region_pred = self.final_conv(x1)

        # Week output
        week_pred = F.adaptive_avg_pool2d(attended_center, 1)
        week_pred = week_pred.view(week_pred.size(0), -1)  # Flatten
        week_pred = self.week_predictor(week_pred)

        return region_pred, week_pred

# Week Feature(embedding encoding) + Texture feature + Deep feature + Attention Unet + Fusion Loss
class Modified_Feature_Improved_Unet(nn.Module):

    class ConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )

        def forward(self, x):
            return self.conv(x)

    class UpConvBlock(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = Modified_Feature_Improved_Unet.ConvBlock(in_channels, out_channels)

        def forward(self, x, skip):
            x = self.up(x)
            # Input is CHW
            diffY = skip.size()[2] - x.size()[2]
            diffX = skip.size()[3] - x.size()[3]
            x = F.pad(x, [diffX // 2, diffX - diffX // 2,
                          diffY // 2, diffY - diffY // 2])
            x = torch.cat([skip, x], dim=1)
            return self.conv(x)
    
    class ResNetFeatureExtractor(nn.Module):
        def __init__(self, output_size):
            super().__init__()
            resnet = models.resnet18(pretrained=True)
            self.features = nn.Sequential(*list(resnet.children())[:-1])
            self.fc = nn.Linear(resnet.fc.in_features, output_size)

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x
    
    @staticmethod
    def extract_glcm_features(image_tensor, levels=256):
        """
        Extract GLCM texture features from an image using mahotas library.

        :param image_tensor: torch.Tensor, a 2D grayscale image tensor
        :param levels: int, number of gray-levels in the image
        :return: 1D array of texture features
        """
        # 将 torch.FloatTensor 转换为 numpy 数组，并缩放到 0-255 范围
        image = image_tensor.squeeze().cpu().numpy()
        image = (image * (levels - 1)).astype(np.uint8)

        # 使用 mahotas 计算 GLCM 特征
        features = mahotas.features.haralick(image).mean(axis=0)  # 计算每个方向的平均值

        return features
    
    class WeekAttentionModule(nn.Module):
        def __init__(self, channel_center, channel_global):
            super().__init__()
            self.attention_layer = nn.Sequential(
                nn.Conv2d(channel_center + channel_global, channel_center, kernel_size=1),
                nn.ReLU(),
                nn.Conv2d(channel_center, 1, kernel_size=1),
                nn.Sigmoid()
            )

        def forward(self, center, combined_features_with_week):
            combined_features_with_week = combined_features_with_week.unsqueeze(2).unsqueeze(3)
            combined_features_with_week = combined_features_with_week.repeat(1, 1, center.shape[2], center.shape[3])
            combined = torch.cat([center, combined_features_with_week], dim=1)
            attention_weights = self.attention_layer(combined)
            attended_center = center * attention_weights
            return attended_center

    class WeekConditionalAttention(nn.Module):
        def __init__(self, F_g, F_l, F_int, num_weeks, embedding_size):
            super().__init__()
            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)
            # self.week_embedding = nn.Embedding(num_weeks, embedding_size)
            self.transform = nn.Linear(embedding_size, F_int)

        def forward(self, g, x, week):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            # week_embed = self.week_embedding(week)
            week_embed = week
            week_transform = self.transform(week_embed).unsqueeze(2).unsqueeze(3)
            psi = self.relu(g1 + x1 + week_transform)
            psi = self.psi(psi)
            return x * psi
    
    # 添加新的初始化参数
    def __init__(self, in_channels=1, out_channels=2, num_weeks=15, texture_feature_size=13, deep_feature_size=512, week_feature_size=18, reduced_feature_size=1024):
        super(Modified_Feature_Improved_Unet, self).__init__()
        
        # 周数编码
        self.week_embedding = nn.Embedding(num_embeddings=32 - 18 + 2, embedding_dim=week_feature_size)
        
        # 新的特征提取器和处理器
        self.texture_feature_processor = nn.Linear(texture_feature_size, 256)
        self.deep_feature_extractor = self.ResNetFeatureExtractor(deep_feature_size)
        
        # 特征降维层（模拟PCA效果）
        self.feature_reduction = nn.Linear(texture_feature_size + deep_feature_size + week_feature_size, reduced_feature_size)
        
        # 特征和周数融合层
        self.feature_and_week_fusion = nn.Linear(272, 1024)
        

        # Encoder
        self.encoder1 = self.ConvBlock(in_channels, 64)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder2 = self.ConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder3 = self.ConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.encoder4 = self.ConvBlock(256, 512)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Center
        self.center = self.ConvBlock(512, 1024)
        self.WAM_center = self.WeekAttentionModule(1024, 1024)
        
        # Decoder
        self.upconv4 = self.UpConvBlock(1024, 512)
        self.attention4 = self.WeekConditionalAttention(F_g=512, F_l=512, F_int=256, num_weeks=num_weeks, embedding_size=week_feature_size)
        self.decoder4 = self.ConvBlock(1024, 512)
        self.upconv3 = self.UpConvBlock(512, 256)
        self.attention3 = self.WeekConditionalAttention(F_g=256, F_l=256, F_int=128, num_weeks=num_weeks, embedding_size=week_feature_size)
        self.decoder3 = self.ConvBlock(512, 256)
        self.upconv2 = self.UpConvBlock(256, 128)
        self.attention2 = self.WeekConditionalAttention(F_g=128, F_l=128, F_int=64, num_weeks=num_weeks, embedding_size=week_feature_size)
        self.decoder2 = self.ConvBlock(256, 128)
        self.upconv1 = self.UpConvBlock(128, 64)
        self.decoder1 = self.ConvBlock(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.week_predictor = nn.Linear(1024, num_weeks+1) 

    def forward(self, x, week):
        # 编码周数信息
        week_indices = torch.tensor([week - 18 if week >= 18 and week <= 32 else 15 for week in week]).to(x.device)
        week_embedded = self.week_embedding(week_indices)

        # 提取纹理特征
        batch_texture_features = [self.extract_glcm_features(img) for img in x]
        texture_features = torch.stack([torch.tensor(feat).float().to(x.device) for feat in batch_texture_features])
        
        # 提取深度特征
        x_3 = x.repeat(1, 3, 1, 1)
        deep_features = self.deep_feature_extractor(x_3)

        # 融合特征
        combined_features = torch.cat([texture_features, deep_features, week_embedded], dim=1)
        combined_features = self.feature_reduction(combined_features)# 降维
        
        # Encoder path
        x1 = self.encoder1(x)
        p1 = self.pool1(x1)
        x2 = self.encoder2(p1)
        p2 = self.pool2(x2)
        x3 = self.encoder3(p2)
        p3 = self.pool3(x3)
        x4 = self.encoder4(p3)
        p4 = self.pool4(x4)

        center = self.center(p4)
        
        # 融合周数特征
        attended_center = self.WAM_center(center, combined_features)

        # Decoder path with attention gates
        g4 = self.upconv4(attended_center, x4)
        x4 = self.attention4(g=g4, x=x4, week=week_embedded)
        d4 = self.decoder4(torch.cat((x4, g4), dim=1))
        
        g3 = self.upconv3(d4, x3)
        x3 = self.attention3(g=g3, x=x3, week=week_embedded)
        d3 = self.decoder3(torch.cat((x3, g3), dim=1))
        
        g2 = self.upconv2(d3, x2)
        x2 = self.attention2(g=g2, x=x2, week=week_embedded)
        d2 = self.decoder2(torch.cat((x2, g2), dim=1))
        
        g1 = self.upconv1(d2, x1)
        d1 = self.decoder1(torch.cat((g1, x1), dim=1))

        # Final output
        region_pred = self.final_conv(d1)
        
        # Week output
        week_pred = F.adaptive_avg_pool2d(attended_center, 1)
        week_pred = week_pred.view(week_pred.size(0), -1)  # Flatten
        week_pred = self.week_predictor(week_pred)

        return region_pred, week_pred

### Multi Targets Shape Attention Modules ###

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
