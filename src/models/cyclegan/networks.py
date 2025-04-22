# Định nghĩa các mạng Generator, Discriminator
import torch
import torch.nn as nn
import torch.nn.functional as F

# Định nghĩa ReflectionPad vì đây là kỹ thuật padding tốt cho image-to-image translation
def reflect_pad_2d(x, pad):
    return F.pad(x, pad, mode='reflect')

# Lớp ResidualBlock cho Generator
class ResidualBlock(nn.Module):
    def __init__(self, features):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(features, features, 3),
            nn.InstanceNorm2d(features)
        )

    def forward(self, x):
        return x + self.block(x)

# Generator dựa trên kiến trúc ResNet
class Generator(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, n_residual_blocks=6):
        super(Generator, self).__init__()
        
        # Initial convolution
        model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_channels, 64, 7),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True)
        ]
        
        # Downsampling (2 blocks)
        in_features = 64
        out_features = in_features * 2
        for _ in range(2):
            model += [
                nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features * 2
        
        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [ResidualBlock(in_features)]
        
        # Upsampling (2 blocks)
        out_features = in_features // 2
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                nn.InstanceNorm2d(out_features),
                nn.ReLU(inplace=True)
            ]
            in_features = out_features
            out_features = in_features // 2
        
        # Output layer
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, output_channels, 7),
            nn.Tanh()
        ]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, x):
        return self.model(x)

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self, input_channels=3):
        super(Discriminator, self).__init__()
        
        # PatchGAN discriminator - không phân loại toàn bộ ảnh mà chỉ các patch
        # Điều này đặc biệt hiệu quả cho CycleGAN
        
        def discriminator_block(in_filters, out_filters, normalize=True):
            block = [nn.Conv2d(in_filters, out_filters, 4, stride=2, padding=1)]
            if normalize:
                block.append(nn.InstanceNorm2d(out_filters))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block
        
        self.model = nn.Sequential(
            *discriminator_block(input_channels, 32),
            *discriminator_block(32, 64),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(256, 1, 4, padding=1)
        )
    
    def forward(self, x):
        return self.model(x)

# Khởi tạo weights theo phân phối normal
def init_weights(net, init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and m.weight is not None:
            if classname.find('Conv') != -1 or classname.find('Linear') != -1:
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
            elif classname.find('BatchNorm2d') != -1 or classname.find('InstanceNorm2d') != -1:
                torch.nn.init.normal_(m.weight.data, 1.0, init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0.0)
    
    net.apply(init_func)
    return net