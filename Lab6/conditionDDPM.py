import torch
import torch.nn as nn
from diffusers import UNet2DConditionModel

#Reference : https://arxiv.org/pdf/2006.11239 , https://github.com/huggingface/diffusion-models-class

class DiffusionModel(nn.Module):

    def __init__(self,
                 resolution=64,
                 in_channels=3,
                 out_channels=3,
                 num_classes=24,
                 cond_dim=128):
        super().__init__()
        self.unet = UNet2DConditionModel(
            sample_size=resolution,
            in_channels=in_channels,
            out_channels=out_channels,
            layers_per_block=2,
            block_out_channels=(128, 128, 256, 512, 512),
            down_block_types=(
                "DownBlock2D", "DownBlock2D", "DownBlock2D",
                "CrossAttnDownBlock2D", "DownBlock2D"
            ),
            up_block_types=(
                "UpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D",
                "UpBlock2D", "UpBlock2D"
            ),
            cross_attention_dim=cond_dim,
        )
        self.label_embedder = nn.Linear(num_classes, cond_dim)

    def forward(self, noisy_image, timesteps, condition_labels):
        """

        Args:
            noisy_image (torch.Tensor): noise image (B, C, H, W)
            timesteps (torch.Tensor): time step (B,)
            condition_labels (torch.Tensor): multi-hot conditional label (B, 24)

        Returns:
            torch.Tensor: predicted noise (B, C, H, W)
        """
        
        
        condition_embeds = self.label_embedder(condition_labels).unsqueeze(1)

        # Pass the noisy image, timesteps, and condition embeddings through the UNet.
        # The UNet will output the predicted noise.
        noise_pred = self.unet(
            sample=noisy_image,
            timestep=timesteps,
            encoder_hidden_states=condition_embeds
        ).sample

        return noise_pred