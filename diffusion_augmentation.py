"""
Diffusion-based Data Augmentation for Retinal Vessel Segmentation
Implements conditional diffusion models for generating synthetic training data
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional
import math


class SinusoidalPositionEmbeddings(nn.Module):
    """Time step embeddings for diffusion process"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class ResidualBlock(nn.Module):
    """Residual block with time embedding"""
    def __init__(self, in_channels: int, out_channels: int, time_emb_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.residual_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        
    def forward(self, x, t):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.relu(h)
        
        # Add time embedding
        time_emb = self.time_mlp(t)
        h = h + time_emb[:, :, None, None]
        
        h = self.conv2(h)
        h = self.norm2(h)
        h = F.relu(h)
        
        return h + self.residual_conv(x)


class AttentionBlock(nn.Module):
    """Self-attention block"""
    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(8, channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        h = self.norm(x)
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        
        # Compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w).permute(0, 2, 1)
        k = k.reshape(b, c, h * w)
        v = v.reshape(b, c, h * w).permute(0, 2, 1)
        
        attn = torch.bmm(q, k) * (c ** -0.5)
        attn = F.softmax(attn, dim=-1)
        
        h = torch.bmm(attn, v)
        h = h.permute(0, 2, 1).reshape(b, c, h, w)
        h = self.proj_out(h)
        
        return x + h


class ConditionalUNet(nn.Module):
    """
    Conditional U-Net for diffusion model
    Takes noisy image and vessel mask as condition
    """
    def __init__(
        self, 
        in_channels: int = 3,
        condition_channels: int = 1,
        model_channels: int = 64,
        out_channels: int = 3,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_res_blocks: int = 2,
        attention_resolutions: Tuple[int, ...] = (16, 8)
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = out_channels
        self.num_res_blocks = num_res_blocks
        
        time_emb_dim = model_channels * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(model_channels),
            nn.Linear(model_channels, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Initial projection with condition concatenation
        self.init_conv = nn.Conv2d(in_channels + condition_channels, model_channels, 3, padding=1)
        
        # Encoder
        self.encoder_blocks = nn.ModuleList()
        self.encoder_attns = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        ch = model_channels
        input_block_chans = [ch]
        ds = 1
        
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = ResidualBlock(ch, mult * model_channels, time_emb_dim)
                ch = mult * model_channels
                self.encoder_blocks.append(layers)
                
                if ds in attention_resolutions:
                    self.encoder_attns.append(AttentionBlock(ch))
                else:
                    self.encoder_attns.append(nn.Identity())
                    
                input_block_chans.append(ch)
            
            if level != len(channel_mult) - 1:
                self.encoder_downs.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
                input_block_chans.append(ch)
                ds *= 2
            else:
                self.encoder_downs.append(nn.Identity())
        
        # Middle
        self.middle_block1 = ResidualBlock(ch, ch, time_emb_dim)
        self.middle_attn = AttentionBlock(ch)
        self.middle_block2 = ResidualBlock(ch, ch, time_emb_dim)
        
        # Decoder
        self.decoder_blocks = nn.ModuleList()
        self.decoder_attns = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                layers = ResidualBlock(
                    ch + input_block_chans.pop(),
                    model_channels * mult,
                    time_emb_dim
                )
                ch = model_channels * mult
                self.decoder_blocks.append(layers)
                
                if ds in attention_resolutions:
                    self.decoder_attns.append(AttentionBlock(ch))
                else:
                    self.decoder_attns.append(nn.Identity())
                
                if level != 0 and i == num_res_blocks:
                    self.decoder_ups.append(nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1))
                    ds //= 2
                else:
                    self.decoder_ups.append(nn.Identity())
        
        self.out_norm = nn.GroupNorm(8, model_channels)
        self.out_conv = nn.Conv2d(model_channels, out_channels, 3, padding=1)
    
    def forward(self, x, t, condition):
        """
        Args:
            x: Noisy image [B, C, H, W]
            t: Time steps [B]
            condition: Vessel mask [B, 1, H, W]
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Concatenate input with condition
        h = torch.cat([x, condition], dim=1)
        h = self.init_conv(h)
        
        # Encoder
        hs = [h]
        for block, attn, down in zip(self.encoder_blocks, self.encoder_attns, self.encoder_downs):
            h = block(h, t_emb)
            h = attn(h)
            hs.append(h)
            h = down(h)
            if not isinstance(down, nn.Identity):
                hs.append(h)
        
        # Middle
        h = self.middle_block1(h, t_emb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, t_emb)
        
        # Decoder
        for block, attn, up in zip(self.decoder_blocks, self.decoder_attns, self.decoder_ups):
            h = torch.cat([h, hs.pop()], dim=1)
            h = block(h, t_emb)
            h = attn(h)
            h = up(h)
        
        h = self.out_norm(h)
        h = F.relu(h)
        h = self.out_conv(h)
        
        return h


class DiffusionAugmentation:
    """
    Implements DDPM (Denoising Diffusion Probabilistic Models) for image augmentation
    """
    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        device: str = 'cuda'
    ):
        self.model = model
        self.timesteps = timesteps
        self.device = device
        
        # Linear beta schedule
        self.betas = torch.linspace(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        
        # Calculations for diffusion q(x_t | x_{t-1}) and others
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.log_one_minus_alphas_cumprod = torch.log(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )
    
    def q_sample(self, x_start, t, noise=None):
        """
        Forward diffusion process: add noise to image
        """
        if noise is None:
            noise = torch.randn_like(x_start)
        
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise
    
    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise"""
        sqrt_recip_alphas_cumprod_t = self.sqrt_recip_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        sqrt_recipm1_alphas_cumprod_t = self.sqrt_recipm1_alphas_cumprod[t].reshape(-1, 1, 1, 1)
        
        return sqrt_recip_alphas_cumprod_t * x_t - sqrt_recipm1_alphas_cumprod_t * noise
    
    def q_posterior(self, x_start, x_t, t):
        """Compute posterior mean and variance"""
        posterior_mean = (
            self.posterior_mean_coef1[t].reshape(-1, 1, 1, 1) * x_start +
            self.posterior_mean_coef2[t].reshape(-1, 1, 1, 1) * x_t
        )
        posterior_variance = self.posterior_variance[t].reshape(-1, 1, 1, 1)
        posterior_log_variance = self.posterior_log_variance_clipped[t].reshape(-1, 1, 1, 1)
        
        return posterior_mean, posterior_variance, posterior_log_variance
    
    @torch.no_grad()
    def p_sample(self, x_t, t, condition):
        """
        Reverse diffusion: single sampling step
        """
        # Predict noise
        predicted_noise = self.model(x_t, t, condition)
        
        # Predict x_0
        x_start = self.predict_start_from_noise(x_t, t, predicted_noise)
        x_start = torch.clamp(x_start, -1.0, 1.0)
        
        # Get posterior
        posterior_mean, _, posterior_log_variance = self.q_posterior(x_start, x_t, t)
        
        # Sample
        noise = torch.randn_like(x_t)
        # No noise when t == 0
        nonzero_mask = (t != 0).float().reshape(-1, 1, 1, 1)
        
        return posterior_mean + nonzero_mask * torch.exp(0.5 * posterior_log_variance) * noise
    
    @torch.no_grad()
    def p_sample_loop(self, shape, condition, return_all=False):
        """
        Generate images by running reverse diffusion process
        """
        device = self.device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        imgs = [img] if return_all else None
        
        for i in reversed(range(0, self.timesteps)):
            t = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, t, condition)
            if return_all:
                imgs.append(img)
        
        return imgs if return_all else img
    
    @torch.no_grad()
    def generate(self, condition, batch_size=1):
        """
        Generate augmented images conditioned on vessel masks
        
        Args:
            condition: Vessel mask tensor [B, 1, H, W]
            batch_size: Number of images to generate
        """
        self.model.eval()
        shape = (batch_size, 3, condition.shape[2], condition.shape[3])
        return self.p_sample_loop(shape, condition)
    
    def train_loss(self, x_start, condition):
        """
        Compute training loss (simple L2 loss on predicted noise)
        """
        batch_size = x_start.shape[0]
        
        # Sample random timesteps
        t = torch.randint(0, self.timesteps, (batch_size,), device=self.device).long()
        
        # Add noise
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        
        # Predict noise
        predicted_noise = self.model(x_t, t, condition)
        
        # Compute loss
        loss = F.mse_loss(predicted_noise, noise)
        
        return loss


def create_diffusion_augmentor(
    image_size: int = 512,
    model_channels: int = 64,
    timesteps: int = 1000,
    device: str = 'cuda'
) -> DiffusionAugmentation:
    """
    Factory function to create diffusion augmentation model
    """
    model = ConditionalUNet(
        in_channels=3,
        condition_channels=1,
        model_channels=model_channels,
        out_channels=3,
        channel_mult=(1, 2, 4, 8),
        num_res_blocks=2,
        attention_resolutions=(16, 8)
    ).to(device)
    
    diffusion = DiffusionAugmentation(
        model=model,
        timesteps=timesteps,
        device=device
    )
    
    return diffusion
