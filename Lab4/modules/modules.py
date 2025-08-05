import torch.nn as nn
import torch
from .layers import DepthConvBlock, ResidualBlock
from torch.autograd import Variable


__all__ = [
    "Generator",
    "RGB_Encoder",
    "Gaussian_Predictor",
    "Decoder_Fusion",
    "Label_Encoder",
    "Learned_Prior_Predictor"
]

"""
    Receive input from decoder fusion ,and generate ouput RGB image
"""
# In modules.py

# ... (keep existing classes like Generator, RGB_Encoder, etc.)

"""
    # Learned Prior Predictor (New Module based on the paper)
    
    Receives features from the previous frame and the current label 
    to predict the parameters of the prior distribution for the latent variable z_t.
"""
class Learned_Prior_Predictor(nn.Sequential):
    def __init__(self, in_chans=160, out_chans=96): # in_chans = F_dim + L_dim
        super(Learned_Prior_Predictor, self).__init__(
            ResidualBlock(in_chans, out_chans // 4),
            DepthConvBlock(out_chans // 4, out_chans // 4),
            ResidualBlock(out_chans // 4, out_chans // 2),
            nn.LeakyReLU(True),
            # Output mu and logvar, so out_chans * 2 for the two parameters
            nn.Conv2d(out_chans // 2, out_chans * 2, kernel_size=1),
        )

    def forward(self, prev_img_feat, label_feat):
        # Input features from the previous frame and current label
        feature = torch.cat([prev_img_feat, label_feat], dim=1)
        parm = super().forward(feature)
        mu, logvar = torch.chunk(parm, 2, dim=1)
        return mu, logvar

# In __all__ list, add the new module
__all__ = [
    "Generator",
    "RGB_Encoder",
    "Gaussian_Predictor",
    "Decoder_Fusion",
    "Label_Encoder",
    "Learned_Prior_Predictor", # Add this
]

# ... (rest of the file)

class Generator(nn.Sequential):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__(
            DepthConvBlock(input_nc, input_nc),
            ResidualBlock(input_nc, input_nc // 2),
            DepthConvBlock(input_nc // 2, input_nc // 2),
            ResidualBlock(input_nc // 2, input_nc // 4),
            DepthConvBlock(input_nc // 4, input_nc // 4),
            ResidualBlock(input_nc // 4, input_nc // 8),
            DepthConvBlock(input_nc // 8, input_nc // 8),
            nn.Conv2d(input_nc // 8, 3, 1),
        )

    def forward(self, input):
        return super().forward(input)


"""
    Frame Encoder
    Extract the features of input RGB image
"""


class RGB_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans):
        super(RGB_Encoder, self).__init__(
            ResidualBlock(in_chans, out_chans // 8),
            DepthConvBlock(out_chans // 8, out_chans // 8),
            ResidualBlock(out_chans // 8, out_chans // 4),
            DepthConvBlock(out_chans // 4, out_chans // 4),
            ResidualBlock(out_chans // 4, out_chans // 2),
            DepthConvBlock(out_chans // 2, out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, 3, padding=1),
        )

    def forward(self, image):
        return super().forward(image)


# Both RGB_Encoder and Label_Encoder extract features as input for gausssion predictor or decoder diffusion

"""
    Extract features of pose stick figures

"""


class Label_Encoder(nn.Sequential):
    def __init__(self, in_chans, out_chans, norm_layer=nn.BatchNorm2d):
        super(Label_Encoder, self).__init__(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_chans, out_chans // 2, kernel_size=7, padding=0),
            norm_layer(out_chans // 2),
            nn.LeakyReLU(True),
            ResidualBlock(in_ch=out_chans // 2, out_ch=out_chans),
        )

    def forward(self, image):
        return super().forward(image)


"""
    # Posterior Predictor
    
    Output the distribution of image and label
"""


class Gaussian_Predictor(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        super(Gaussian_Predictor, self).__init__(
            ResidualBlock(in_chans, out_chans // 4),
            DepthConvBlock(out_chans // 4, out_chans // 4),
            ResidualBlock(out_chans // 4, out_chans // 2),
            DepthConvBlock(out_chans // 2, out_chans // 2),
            ResidualBlock(out_chans // 2, out_chans),
            nn.LeakyReLU(True),
            nn.Conv2d(out_chans, out_chans * 2, kernel_size=1),
        )

    def reparameterize(self, mu, logvar):

        # mu : mean, logvar : log(var)
        # log(σ ^ 2 ) = 2 log(σ) => log(σ) = 1/2 log(σ^2)
        # σ = exp(1/2 * log(σ^2))
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, img, label):
        # CVAE input
        feature = torch.cat([img, label], dim=1)
        parm = super().forward(feature)
        mu, logvar = torch.chunk(parm, 2, dim=1)
        z = self.reparameterize(mu, logvar)

        return z, mu, logvar


"""
    Receive image , label , param (latent var), produce the intermediate features
"""


class Decoder_Fusion(nn.Sequential):
    def __init__(self, in_chans=48, out_chans=96):
        super().__init__(
            DepthConvBlock(in_chans, in_chans),
            ResidualBlock(in_chans, in_chans // 4),
            DepthConvBlock(in_chans // 4, in_chans // 2),
            ResidualBlock(in_chans // 2, in_chans // 2),
            DepthConvBlock(in_chans // 2, out_chans // 2),
            nn.Conv2d(out_chans // 2, out_chans, 1, 1),
        )

    # --- MODIFIED: Add skip_feat as an argument ---
    def forward(self, img, label, parm, skip_feat):
        # Concatenate skip_feat with other features
        feature = torch.cat([img, label, parm, skip_feat], dim=1)
        return super().forward(feature)


if __name__ == "__main__":
    pass
