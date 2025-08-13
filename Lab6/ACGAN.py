import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class ResBlock_G(nn.Module):
    """Residual Block for the Generator."""
    def __init__(self, in_channels, out_channels):
        super(ResBlock_G, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.upsample = nn.Upsample(scale_factor=2)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        h = self.bn1(x)
        h = nn.ReLU(True)(h)
        h = self.upsample(h)
        h = self.conv1(h)
        h = self.bn2(h)
        h = nn.ReLU(True)(h)
        h = self.conv2(h)
        
        # Shortcut connection
        x_shortcut = self.upsample(x)
        x_shortcut = self.shortcut(x_shortcut)
        
        return h + x_shortcut

class Generator(nn.Module):
    """A ResNet-based Generator."""
    def __init__(self, z_dim=100, c_dim=100, dim=256):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.c_dim = c_dim
        self.dim = dim

        self.label_emb = nn.Linear(24, c_dim)
        
        self.dense = nn.Linear(z_dim + c_dim, 4 * 4 * dim)

        self.res_blocks = nn.Sequential(
            ResBlock_G(dim, dim),        # 8x8
            ResBlock_G(dim, dim),        # 16x16
            ResBlock_G(dim, dim),        # 32x32
        )
        
        self.final_conv = nn.Sequential(
            nn.BatchNorm2d(dim),
            nn.ReLU(True),
            nn.Conv2d(dim, 3, 3, 1, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, c):
        c_emb = self.label_emb(c)
        x = torch.cat([z, c_emb], 1)
        x = self.dense(x)
        x = x.view(-1, self.dim, 4, 4)
        x = self.res_blocks(x)
        x = nn.Upsample(scale_factor=2)(x) # to 64x64
        x = self.final_conv(x)
        return x

class Discriminator(nn.Module):
    def __init__(self, ndf=32):
        super(Discriminator, self).__init__()
        def discriminator_block(in_filters, out_filters, bn=True):
            layers = [spectral_norm(nn.Conv2d(in_filters, out_filters, 4, 2, 1, bias=False))]
            if bn:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.conv_blocks = nn.Sequential(
            *discriminator_block(3, ndf, bn=False),
            *discriminator_block(ndf, ndf * 2),
            *discriminator_block(ndf * 2, ndf * 4),
            *discriminator_block(ndf * 4, ndf * 8),
        )
        final_feature_map_size = ndf * 8 * 4 * 4
        self.adv_layer = nn.Linear(final_feature_map_size, 1)
        self.aux_layer = nn.Sequential(nn.Linear(final_feature_map_size, 24), nn.Sigmoid())

    def forward(self, img):
        features = self.conv_blocks(img)
        features_flat = features.view(features.shape[0], -1)
        validity = self.adv_layer(features_flat)
        label = self.aux_layer(features_flat)
        return validity, label