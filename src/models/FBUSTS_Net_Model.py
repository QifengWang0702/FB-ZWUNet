import math
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable
import pdb
import cv2
import cmath

#torch_ver = torch.__version__[:3]

__all__ = ['PAM_Module', 'CAM_Module', 'MAM_Module', 'semanticModule', 'ZAM_Module', 'WAM_Module', 'GaborFeatureExtractor', 'GLCMFeatureExtractor', 'SE_Block', 'DropBlock2D', 'CCM_Module', 'ImprovedZAM_Module', 'ImprovedWAM_Module']


#这两个类是语义注意力模块的编码器和解码器部分。编码器逐渐减少特征图的尺寸，而解码器则逐渐增加特征图的尺寸。
class _EncoderBlock(nn.Module):
    """
    Encoder block for Semantic Attention Module
    包含两个卷积层，每个卷积层后面都跟随一个批量归一化层和ReLU激活函数。最后，使用最大池化层减少特征图的尺寸。
    """
    def __init__(self, in_channels, out_channels, dropout=False):
        super(_EncoderBlock, self).__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout())
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.encode = nn.Sequential(*layers)

    def forward(self, x):
        return self.encode(x)

class _DecoderBlock(nn.Module):
    """
    Decoder Block for Semantic Attention Module
    包含两个卷积层和一个转置卷积层，用于增加特征图的尺寸。
    """
    def __init__(self, in_channels, middle_channels, out_channels):
        super(_DecoderBlock, self).__init__()
        self.decode = nn.Sequential(
            nn.Conv2d(in_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(middle_channels, middle_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(middle_channels, out_channels, kernel_size=2, stride=2),
        )

    def forward(self, x):
        return self.decode(x)

class semanticModule(nn.Module):
    """
    Semantic attention module
    这是语义注意力模块的主要部分。它包含两个编码器块和两个解码器块，用于提取和重建特征图。
    """
    def __init__(self, in_dim):
        super(semanticModule, self).__init__()
        self.chanel_in = in_dim

        self.enc1 = _EncoderBlock(in_dim, in_dim*2)
        self.enc2 = _EncoderBlock(in_dim*2, in_dim*4)
        self.dec2 = _DecoderBlock(in_dim * 4, in_dim * 2, in_dim * 2)
        self.dec1 = _DecoderBlock(in_dim * 2, in_dim, in_dim )

    def forward(self,x):

        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)

        dec2 = self.dec2( enc2)
        dec1 = self.dec1( F.upsample(dec2, enc1.size()[2:], mode='bilinear'))

        return enc2.view(-1), dec1

class PAM_Module(nn.Module):
    """ 
    Position attention module
    这是位置注意力模块。它使用查询、键和值的概念来计算注意力权重，并使用这些权重来加权输入特征图。
    """
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)

        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class CAM_Module(nn.Module):
    """ 
    Channel attention module
    这是通道注意力模块。它计算输入特征图的每个通道之间的关系，并使用这些关系来加权输入特征图。
    """
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim
        
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
       
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        return out

class PAM_CAM_Layer(nn.Module):
    """
    Helper Function for PAM and CAM attention
    这是一个辅助函数，用于结合PAM和CAM注意力。它首先使用一个卷积层来融合输入特征图，然后应用PAM或CAM注意力。
    Parameters:
    ----------
    input:
        in_ch : input channels
        use_pam : Boolean value whether to use PAM_Module or CAM_Module
    output:
        returns the attention map
    """
    def __init__(self, in_ch, use_pam = True):
        super(PAM_CAM_Layer, self).__init__()
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
            PAM_Module(in_ch) if use_pam else CAM_Module(in_ch),
			nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.attn(x)

class MAM_Module(nn.Module):
    """ 
    Morphological attention module
    这是形态学注意力模块。它结合了形态学梯度和Canny边缘检测来产生对输入特征图的注意力。
    通过使用可训练的核，该模块可以在训练过程中适应和优化形态学操作，以更好地突出目标的形态学特征。
    
    Parameters:
    ----------
    in_dim : int
        输入特征图的通道数。
    
    Attributes:
    ----------
    chanel_in : int
        输入特征图的通道数。
    alpha : torch.nn.Parameter
        控制形态学梯度在注意力机制中的权重。
    beta : torch.nn.Parameter
        控制Canny边缘检测在注意力机制中的权重。
    gamma : torch.nn.Parameter
        控制如何将形态学注意力应用于输入特征图。
    kernel_size : int
        用于形态学操作的核的大小。
    morph_kernel : torch.nn.Parameter
        用于形态学操作的可训练核。
    
    Methods:
    -------
    morphological_gradient(x, kernel):
        使用给定的核计算形态学梯度。
    canny_edge(x, kernel):
        使用给定的核计算Canny边缘检测的梯度幅度。
    forward(x):
        通过结合形态学梯度和Canny边缘检测来计算形态学注意力，并将其应用于输入特征图。
    """
    def __init__(self, in_dim):
        super(MAM_Module, self).__init__()
        self.chanel_in = in_dim

        # Parameters for controlling the attention mechanism
        self.alpha = nn.Parameter(torch.zeros(1))
        self.beta = nn.Parameter(torch.zeros(1))
        self.gamma = nn.Parameter(torch.zeros(1))

        # Define the kernel size for morphology operations
        self.kernel_size = 3
        #self.morph_kernel = nn.Parameter(torch.ones(1, 1, self.kernel_size, self.kernel_size), requires_grad=True)
        self.morph_kernel = nn.Parameter(torch.ones(self.chanel_in, 1, self.kernel_size, self.kernel_size), requires_grad=True)

    def morphological_gradient(self, x, kernel):
        """计算形态学梯度，它是膨胀和腐蚀之间的差异。"""
        # Dilation
        #dilation = F.conv2d(x, kernel, padding=1)
        dilation = F.conv2d(x, kernel, padding=1, groups=self.chanel_in)
        # Erosion
        #erosion = -F.conv2d(-x, kernel, padding=1)
        erosion = -F.conv2d(-x, kernel, padding=1, groups=self.chanel_in)
        
        return dilation - erosion

    def canny_edge(self, x, kernel):
        """使用Sobel算子计算Canny边缘检测的梯度幅度。"""
        # Sobel operators
        # sobel_x = F.conv2d(x, kernel, padding=1)
        # sobel_y = F.conv2d(x, kernel.transpose(2, 3), padding=1)
        sobel_x = F.conv2d(x, kernel, padding=1, groups=self.chanel_in)
        sobel_y = F.conv2d(x, kernel.transpose(2, 3), padding=1, groups=self.chanel_in)
        
        magnitude = torch.sqrt(sobel_x**2 + sobel_y**2)
        return magnitude

    def forward(self, x):
        """
        Parameters:
        ----------
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X H X W
        """
        m_batchsize, C, height, width = x.size()

        # Morphological Gradient
        morph_gradient = self.morphological_gradient(x, self.morph_kernel)

        # Canny Edge Detection
        #sobel_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_kernel = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).unsqueeze(0).unsqueeze(0).repeat(self.chanel_in, 1, 1, 1)
        sobel_kernel = sobel_kernel.to(x.device)
        edges = self.canny_edge(x, sobel_kernel)

        # Combine the morphological gradient and Canny edges
        combined_attention = self.alpha * morph_gradient + self.beta * edges

        # Apply attention to the input feature map
        out = self.gamma * combined_attention + x

        return out

class ZAM_Module(nn.Module):
    #Zernike Attention Module。这个模块用于计算Zernike矩（Zpq）并将其用作注意力机制。这个模块主要用于捕获图像中的复杂形状或纹理信息，并通过注意力机制进行强调。
    def __init__(self, in_dim):
        super(ZAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.gamma = nn.Parameter(torch.zeros(1))

    def zernike_moment(self, x, p, q):
        N = x.size(2)
        Zpq = torch.zeros_like(x)
        for i in range(N):
            for j in range(N):
                r = np.sqrt(i**2 + j**2) / N
                theta = np.arctan2(j, i)
                Zpq += x * torch.tensor(cmath.exp(-1j * q * theta)).real  # Use real part
        Zpq *= (p + 1) / (np.pi * (N - 1)**2)
        return Zpq

    def forward(self, x):
        m_batchsize, C, height, width = x.size()
        p, q = 2, 1
        Zpq = self.zernike_moment(x, p, q)
        out = self.gamma * Zpq + x
        return out

class WAM_Module(nn.Module):
    '''MultiLayerWaveletAttention'''
    #多层小波注意力模块。使用三种不同尺寸的卷积核（3x3, 5x5, 7x7）来提取特征。这些特征通过可学习的参数（gamma1, gamma2, gamma3）进行加权组合。
    def __init__(self, in_channels):
        super(WAM_Module, self).__init__()
        
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        
        # 第一层小波核
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        
        # 第二层小波核
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=5, padding=2)
        
        # 第三层小波核
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=7, padding=3)

    def forward(self, x):
        wavelet_features1 = self.conv1(x)
        wavelet_features2 = self.conv2(x)
        wavelet_features3 = self.conv3(x)
        
        out = self.gamma1 * wavelet_features1 + self.gamma2 * wavelet_features2 + self.gamma3 * wavelet_features3 + x
        return out

class ZAM_WAM_Layer(nn.Module):
    """
    Helper Function for Zernike and MultiLayerWavelet attention.
    This is a helper function that combines Zernike and MultiLayerWavelet attention.
    It first uses a convolutional layer to fuse the input feature maps, then applies Zernike or MultiLayerWavelet attention.
    Parameters:
    ----------
    input:
        in_ch : input channels
        use_zernike : Boolean value whether to use ZernikeAttention or MultiLayerWaveletAttention
    output:
        returns the attention map
    """
    #Zernike和多层小波注意力的组合层。首先使用一个卷积层来融合输入特征图，然后应用Zernike或多层小波注意力。
    def __init__(self, in_ch, use_zernike=True):
        super(ZAM_WAM_Layer, self).__init__()
        
        self.attn = nn.Sequential(
            nn.Conv2d(in_ch * 2, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU(),
            ZAM_Module(in_ch) if use_zernike else WAM_Module(in_ch),
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_ch),
            nn.PReLU()
        )
    
    def forward(self, x):
        return self.attn(x)

class MultiConv(nn.Module):
    """
    Helper function for Multiple Convolutions for refining.
    这是一个辅助函数，用于多次卷积操作。它包含三个卷积层，每个卷积层后面都跟随一个批量归一化层。最后一个卷积层后面可以选择使用Softmax2d或PReLU激活函数。
    Parameters:
    ----------
    inputs:
        in_ch : input channels
        out_ch : output channels
        attn : Boolean value whether to use Softmax or PReLU
    outputs:
        returns the refined convolution tensor
    """
    def __init__(self, in_ch, out_ch, attn = True):
        super(MultiConv, self).__init__()
        
        self.fuse_attn = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), 
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1), 
            nn.BatchNorm2d(64), 
            nn.PReLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=1), 
            nn.BatchNorm2d(64), 
            nn.Softmax2d() if attn else nn.PReLU()
        )
    
    def forward(self, x):
        return self.fuse_attn(x)

def gabor_kernel(size, sigma, theta, lambd, gamma):
    """生成Gabor滤波器"""
    x, y = np.meshgrid(np.linspace(-size//2, size//2, size), np.linspace(-size//2, size//2, size))
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)
    kernel = np.exp(-0.5 * (x_theta ** 2 + (gamma * y_theta) ** 2) / (sigma ** 2)) * np.cos(2 * np.pi * x_theta / lambd)
    return torch.tensor(kernel, dtype=torch.float32)

class GaborFeatureExtractor(nn.Module):
    #使用Gabor滤波器进行特征提取。Gabor滤波器能捕捉图像的局部空间频率特性。
    def __init__(self, in_channels, out_channels, kernel_size=15):
        super(GaborFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        # 初始化Gabor滤波器
        self.filters = nn.Parameter(torch.zeros((out_channels, in_channels, kernel_size, kernel_size)), requires_grad=False)
        for i in range(out_channels):
            theta = i * np.pi / out_channels  # 方向
            sigma = 2.0  # 标准差
            lambd = 10.0  # 波长
            gamma = 0.5  # 空间纵横比
            self.filters[i, 0] = gabor_kernel(kernel_size, sigma, theta, lambd, gamma)

    def forward(self, x):
        return nn.functional.conv2d(x, self.filters, padding=self.kernel_size//2)

class GLCMFeatureExtractor(nn.Module):
    #灰度共生矩阵（GLCM）特征提取器。使用卷积层模拟GLCM特征提取，用于捕获图像的纹理信息。
    def __init__(self, in_channels, out_channels):
        super(GLCMFeatureExtractor, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 定义卷积层，用于模拟GLCM特征提取
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # 可以添加更多层以捕获更复杂的纹理特征

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        return x

class SE_Block(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SE_Block, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class DropBlock2D(nn.Module):
    def __init__(self, block_size, keep_prob, sync_channels=False):
        super(DropBlock2D, self).__init__()
        self.block_size = block_size
        self.keep_prob = keep_prob
        self.sync_channels = sync_channels

    def _compute_valid_seed_region(self, height, width, device):
        half_block_size = self.block_size // 2
        valid_seed_region = torch.ones((height, width)).to(device)
        valid_seed_region[half_block_size:-half_block_size, half_block_size:-half_block_size] = 0
        return valid_seed_region

    def _compute_drop_mask(self, shape, x):
        height, width = shape[2], shape[3]
        mask = (torch.rand(shape).to(x.device) < self._get_gamma(height, width)).float()
        mask *= self._compute_valid_seed_region(height, width, x.device)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size // 2)
        return 1 - mask

    def _get_gamma(self, height, width):
        return (1.0 - self.keep_prob) * height * width / ((self.block_size ** 2) * (height - self.block_size + 1) * (width - self.block_size + 1))

    def forward(self, x):
        if self.training:
            shape = x.shape
            if self.sync_channels:
                mask = self._compute_drop_mask([shape[0], 1, shape[2], shape[3]], x)
                mask = mask.expand(shape)
            else:
                mask = self._compute_drop_mask(shape, x)
            x *= mask
            x *= (torch.numel(mask) / mask.sum())
        return x

class CCM_Module(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(CCM_Module, self).__init__()
        
        # Two projection heads
        self.proj_CC = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.proj_SP = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        # Splitting the feature map x into dental plaque and teeth representations
        Fp = x[:, :x.size(1)//2, :, :]
        Ft = x[:, x.size(1)//2:, :, :]
        
        # Get projected features
        projected_Fp = self.proj_CC(Fp)
        projected_Ft = self.proj_SP(Ft)
        
        # Compute the contrastive regularization between plaque and teeth features
        LCCM = - torch.sum(projected_Fp * projected_Ft) / (torch.norm(projected_Fp) * torch.norm(projected_Ft))
        
        return LCCM, projected_Fp, projected_Ft


### New Attention Modules ###
class ImprovedZAM_Module(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ImprovedZAM_Module, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def zernike_moment(self, x, p, q, N):
        Zpq = torch.zeros_like(x)
        for i in range(N):
            for j in range(N):
                r = np.sqrt(i**2 + j**2) / N
                theta = np.arctan2(j, i)
                Zpq += x * torch.tensor(cmath.exp(-1j * q * theta)).real
        Zpq *= (p + 1) / (np.pi * (N - 1)**2)
        return Zpq

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Zernike moment parameters
        p, q = 2, 1  # 这里可以根据需要调整 p 和 q 的值
        N = x.size(2)  # 假设图像是正方形

        Zpq = self.zernike_moment(psi, p, q, N)

        return x * Zpq

class ImprovedWAM_Module(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(ImprovedWAM_Module, self).__init__()
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        # 小波卷积核
        self.conv1 = nn.Conv2d(F_int, F_int, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(F_int, F_int, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(F_int, F_int, kernel_size=7, padding=3)

        # 1x1卷积以扩展psi的通道数
        self.expand_channels = nn.Conv2d(1, F_int, kernel_size=1, bias=False)

        # 1x1卷积以调整wavelet_features的通道数与x相匹配
        self.adjust_channels = nn.Conv2d(F_int, F_l, kernel_size=1, bias=False)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

       # 扩展psi的通道数
        psi_expanded = self.expand_channels(psi)

        # 应用小波特征
        wavelet_features = self.conv1(psi_expanded) + self.conv2(psi_expanded) + self.conv3(psi_expanded)

        # 调整wavelet_features的通道数以匹配x
        wavelet_features_adjusted = self.adjust_channels(wavelet_features)
        
        return x * wavelet_features_adjusted

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

class Channel_Zernike_Attention_Module(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Channel_Zernike_Attention_Module, self).__init__()
        self.channel_attention = ChannelAttention(F_l)
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def zernike_moment(self, x, p, q, N):
        Zpq = torch.zeros_like(x)
        if N > 1:  # 确保不会发生除以零的情况
            for i in range(N):
                for j in range(N):
                    r = np.sqrt(i**2 + j**2) / N
                    theta = np.arctan2(j, i)
                    Zpq += x * torch.tensor(cmath.exp(-1j * q * theta)).real
            Zpq *= (p + 1) / (np.pi * (N - 1)**2)
        return Zpq


    def forward(self, g, x):
        x = self.channel_attention(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        # Zernike moment parameters
        p, q = 2, 1  # 这里可以根据需要调整 p 和 q 的值
        N = x.size(2)  # 假设图像是正方形

        Zpq = self.zernike_moment(psi, p, q, N)

        return x * Zpq

class Channel_Wavelet_Attention_Module(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Channel_Wavelet_Attention_Module, self).__init__()
        self.channel_attention = ChannelAttention(F_l)
        self.W_g = nn.Conv2d(F_g, F_int, kernel_size=1, bias=False)
        self.W_x = nn.Conv2d(F_l, F_int, kernel_size=1, bias=False)
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

        # 小波卷积核
        self.conv1 = nn.Conv2d(F_int, F_int, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(F_int, F_int, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(F_int, F_int, kernel_size=7, padding=3)

        # 1x1卷积以扩展psi的通道数
        self.expand_channels = nn.Conv2d(1, F_int, kernel_size=1, bias=False)

        # 1x1卷积以调整wavelet_features的通道数与x相匹配
        self.adjust_channels = nn.Conv2d(F_int, F_l, kernel_size=1, bias=False)

    def forward(self, g, x):
        x = self.channel_attention(x)
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

       # 扩展psi的通道数
        psi_expanded = self.expand_channels(psi)

        # 应用小波特征
        wavelet_features = self.conv1(psi_expanded) + self.conv2(psi_expanded) + self.conv3(psi_expanded)

        # 调整wavelet_features的通道数以匹配x
        wavelet_features_adjusted = self.adjust_channels(wavelet_features)
        
        return x * wavelet_features_adjusted
