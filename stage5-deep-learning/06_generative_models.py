"""
生成模型：VAE、GAN和Diffusion
学习目标：掌握主流生成模型的原理和实现
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
import warnings
warnings.filterwarnings('ignore')

print("=== 生成模型：VAE、GAN和Diffusion ===\n")

# 1. 生成模型理论基础
print("1. 生成模型理论基础")
print("生成模型类型：")
print("- 自回归模型：逐步生成序列")
print("- 变分自编码器(VAE)：学习潜在分布")
print("- 生成对抗网络(GAN)：对抗训练生成")
print("- 扩散模型：逐步去噪生成")
print("- 标准化流：可逆变换")

print("\n评估指标：")
print("1. 似然性：对数似然、困惑度")
print("2. 生成质量：FID、IS、LPIPS")
print("3. 多样性：Mode Coverage、多样性距离")
print("4. 条件生成：CLIP Score、人工评估")

# 2. 变分自编码器(VAE)
print("\n2. 变分自编码器(VAE)")

class VAEEncoder(nn.Module):
    """VAE编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAEEncoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class VAEDecoder(nn.Module):
    """VAE解码器"""
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(VAEDecoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, z):
        h = F.relu(self.fc1(z))
        h = F.relu(self.fc2(h))
        return torch.sigmoid(self.fc3(h))

class VAE(nn.Module):
    """变分自编码器"""
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(VAE, self).__init__()
        self.encoder = VAEEncoder(input_dim, hidden_dim, latent_dim)
        self.decoder = VAEDecoder(latent_dim, hidden_dim, input_dim)
        
    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar
    
    def generate(self, num_samples, device):
        """生成新样本"""
        z = torch.randn(num_samples, self.decoder.fc1.in_features).to(device)
        return self.decoder(z)

def vae_loss(x_recon, x, mu, logvar, beta=1.0):
    """VAE损失函数"""
    # 重构损失
    recon_loss = F.binary_cross_entropy(x_recon, x, reduction='sum')
    
    # KL散度损失
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss, recon_loss, kl_loss

# 3. 卷积VAE（用于图像）
print("\n3. 卷积VAE（用于图像）")

class ConvVAEEncoder(nn.Module):
    """卷积VAE编码器"""
    def __init__(self, input_channels, latent_dim):
        super(ConvVAEEncoder, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        
        self.fc_mu = nn.Linear(256 * 2 * 2, latent_dim)
        self.fc_logvar = nn.Linear(256 * 2 * 2, latent_dim)
        
    def forward(self, x):
        h = self.conv_layers(x)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

class ConvVAEDecoder(nn.Module):
    """卷积VAE解码器"""
    def __init__(self, latent_dim, output_channels):
        super(ConvVAEDecoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256 * 2 * 2)
        
        self.deconv_layers = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, output_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        h = self.fc(z)
        h = h.view(h.size(0), 256, 2, 2)
        return self.deconv_layers(h)

class ConvVAE(nn.Module):
    """卷积变分自编码器"""
    def __init__(self, input_channels, latent_dim):
        super(ConvVAE, self).__init__()
        self.encoder = ConvVAEEncoder(input_channels, latent_dim)
        self.decoder = ConvVAEDecoder(latent_dim, input_channels)
        
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return x_recon, mu, logvar

# 4. 生成对抗网络(GAN)
print("\n4. 生成对抗网络(GAN)")

class Generator(nn.Module):
    """GAN生成器"""
    def __init__(self, noise_dim, output_channels, feature_maps=64):
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        
        self.main = nn.Sequential(
            # 输入是噪声向量
            nn.ConvTranspose2d(noise_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            # 状态大小: (feature_maps*8) x 4 x 4
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            # 状态大小: (feature_maps*4) x 8 x 8
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            # 状态大小: (feature_maps*2) x 16 x 16
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            # 状态大小: (feature_maps) x 32 x 32
            nn.ConvTranspose2d(feature_maps, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 状态大小: (output_channels) x 64 x 64
        )
        
    def forward(self, noise):
        return self.main(noise)

class Discriminator(nn.Module):
    """GAN判别器"""
    def __init__(self, input_channels, feature_maps=64):
        super(Discriminator, self).__init__()
        
        self.main = nn.Sequential(
            # 输入大小: (input_channels) x 64 x 64
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (feature_maps) x 32 x 32
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (feature_maps*2) x 16 x 16
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (feature_maps*4) x 8 x 8
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            # 状态大小: (feature_maps*8) x 4 x 4
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# 5. Wasserstein GAN
print("\n5. Wasserstein GAN")

class WGANCritic(nn.Module):
    """WGAN判别器（Critic）"""
    def __init__(self, input_channels, feature_maps=64):
        super(WGANCritic, self).__init__()
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels, feature_maps, 4, 2, 1),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1),
            nn.LayerNorm([feature_maps * 2, 16, 16]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1),
            nn.LayerNorm([feature_maps * 4, 8, 8]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1),
            nn.LayerNorm([feature_maps * 8, 4, 4]),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0)
        )
        
    def forward(self, input):
        return self.main(input).view(-1)

def gradient_penalty(critic, real_samples, fake_samples, device):
    """梯度惩罚项"""
    batch_size = real_samples.size(0)
    alpha = torch.rand(batch_size, 1, 1, 1).to(device)
    
    interpolates = alpha * real_samples + (1 - alpha) * fake_samples
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)
    
    disc_interpolates = critic(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=disc_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(disc_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

# 6. 条件GAN
print("\n6. 条件GAN")

class ConditionalGenerator(nn.Module):
    """条件GAN生成器"""
    def __init__(self, noise_dim, num_classes, embed_dim, output_channels, feature_maps=64):
        super(ConditionalGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.embed_dim = embed_dim
        
        self.label_embedding = nn.Embedding(num_classes, embed_dim)
        
        self.main = nn.Sequential(
            nn.ConvTranspose2d(noise_dim + embed_dim, feature_maps * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 8, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 4, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps * 2, feature_maps, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(feature_maps, output_channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        
    def forward(self, noise, labels):
        # 嵌入标签
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(label_embed.size(0), self.embed_dim, 1, 1)
        
        # 拼接噪声和标签嵌入
        input_tensor = torch.cat([noise, label_embed], dim=1)
        
        return self.main(input_tensor)

class ConditionalDiscriminator(nn.Module):
    """条件GAN判别器"""
    def __init__(self, input_channels, num_classes, feature_maps=64):
        super(ConditionalDiscriminator, self).__init__()
        
        self.label_embedding = nn.Embedding(num_classes, input_channels * 64 * 64)
        
        self.main = nn.Sequential(
            nn.Conv2d(input_channels * 2, feature_maps, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps, feature_maps * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 2, feature_maps * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 4, feature_maps * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_maps * 8),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(feature_maps * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
        
    def forward(self, input, labels):
        # 嵌入标签
        label_embed = self.label_embedding(labels)
        label_embed = label_embed.view(labels.size(0), input.size(1), input.size(2), input.size(3))
        
        # 拼接输入和标签
        input_with_label = torch.cat([input, label_embed], dim=1)
        
        return self.main(input_with_label).view(-1, 1).squeeze(1)

# 7. 简化扩散模型
print("\n7. 简化扩散模型")

class SimpleDiffusion(nn.Module):
    """简化的扩散模型"""
    def __init__(self, input_channels, timesteps=1000):
        super(SimpleDiffusion, self).__init__()
        self.timesteps = timesteps
        
        # 噪声调度
        self.register_buffer('betas', torch.linspace(0.0001, 0.02, timesteps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        
        # U-Net架构的简化版本
        self.down_conv = nn.Sequential(
            nn.Conv2d(input_channels + 1, 64, 3, padding=1),  # +1 for time embedding
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        
        self.up_conv = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, 3, padding=1)
        )
        
        self.time_embed = nn.Embedding(timesteps, 1)
        
    def get_time_embedding(self, timestep, batch_size, height, width):
        """获取时间嵌入"""
        time_embed = self.time_embed(timestep)
        time_embed = time_embed.view(batch_size, 1, 1, 1)
        time_embed = time_embed.expand(-1, -1, height, width)
        return time_embed
    
    def forward(self, x, timestep):
        batch_size, channels, height, width = x.shape
        
        # 时间嵌入
        time_embed = self.get_time_embedding(timestep, batch_size, height, width)
        
        # 拼接输入和时间嵌入
        x_with_time = torch.cat([x, time_embed], dim=1)
        
        # 网络前向传播
        h = self.down_conv(x_with_time)
        noise_pred = self.up_conv(h)
        
        return noise_pred
    
    def add_noise(self, x, noise, timestep):
        """向图像添加噪声"""
        sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod[timestep])
        
        return sqrt_alphas_cumprod * x + sqrt_one_minus_alphas_cumprod * noise
    
    def sample(self, shape, device):
        """从纯噪声开始采样"""
        x = torch.randn(shape).to(device)
        
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((shape[0],), t, dtype=torch.long, device=device)
            
            # 预测噪声
            noise_pred = self(x, t_tensor)
            
            # 去噪步骤（简化版本）
            if t > 0:
                beta = self.betas[t]
                alpha = self.alphas[t]
                alpha_cumprod = self.alphas_cumprod[t]
                alpha_cumprod_prev = self.alphas_cumprod[t-1]
                
                # 计算去噪后的图像
                x = (1.0 / torch.sqrt(alpha)) * (x - beta / torch.sqrt(1.0 - alpha_cumprod) * noise_pred)
                
                # 添加噪声（除了最后一步）
                if t > 1:
                    noise = torch.randn_like(x)
                    x = x + torch.sqrt(beta) * noise
        
        return x

# 8. 生成模型训练函数
print("\n8. 生成模型训练函数")

def train_vae(model, dataloader, epochs=10, lr=1e-3, device='cpu', beta=1.0):
    """训练VAE"""
    optimizer = optim.Adam(model.parameters(), lr=lr)
    model.train()
    
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # 展平
            
            optimizer.zero_grad()
            recon, mu, logvar = model(data)
            loss, recon_loss, kl_loss = vae_loss(recon, data, mu, logvar, beta)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        print(f'Epoch {epoch}, Average Loss: {total_loss/len(dataloader):.4f}')

def train_gan(generator, discriminator, dataloader, epochs=10, lr=2e-4, device='cpu'):
    """训练GAN"""
    criterion = nn.BCELoss()
    opt_g = optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_d = optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    
    for epoch in range(epochs):
        for batch_idx, (real_data, _) in enumerate(dataloader):
            batch_size = real_data.size(0)
            real_data = real_data.to(device)
            
            # 训练判别器
            opt_d.zero_grad()
            
            # 真实数据
            real_label = torch.ones(batch_size).to(device)
            real_output = discriminator(real_data)
            d_loss_real = criterion(real_output, real_label)
            
            # 生成数据
            noise = torch.randn(batch_size, generator.noise_dim, 1, 1).to(device)
            fake_data = generator(noise)
            fake_label = torch.zeros(batch_size).to(device)
            fake_output = discriminator(fake_data.detach())
            d_loss_fake = criterion(fake_output, fake_label)
            
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            opt_d.step()
            
            # 训练生成器
            opt_g.zero_grad()
            fake_output = discriminator(fake_data)
            g_loss = criterion(fake_output, real_label)
            g_loss.backward()
            opt_g.step()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}')

# 9. 生成样本可视化
print("\n9. 生成样本可视化")

def visualize_samples(samples, nrow=8, title="Generated Samples"):
    """可视化生成样本"""
    fig, ax = plt.subplots(figsize=(10, 10))
    
    if samples.dim() == 4:  # 图像数据
        grid = make_grid(samples.cpu(), nrow=nrow, normalize=True, padding=2)
        ax.imshow(grid.permute(1, 2, 0))
    else:  # 其他数据
        # 重塑为图像格式进行可视化
        size = int(np.sqrt(samples.size(-1)))
        samples_reshaped = samples.view(-1, 1, size, size)
        grid = make_grid(samples_reshaped.cpu(), nrow=nrow, normalize=True)
        ax.imshow(grid.permute(1, 2, 0), cmap='gray')
    
    ax.set_title(title)
    ax.axis('off')
    return fig

def compare_models(vae_samples, gan_samples, diffusion_samples):
    """比较不同生成模型的结果"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models = [vae_samples, gan_samples, diffusion_samples]
    titles = ['VAE Samples', 'GAN Samples', 'Diffusion Samples']
    
    for i, (samples, title) in enumerate(zip(models, titles)):
        if samples is not None:
            if samples.dim() == 4:
                grid = make_grid(samples.cpu()[:16], nrow=4, normalize=True)
                axes[i].imshow(grid.permute(1, 2, 0))
            else:
                size = int(np.sqrt(samples.size(-1)))
                samples_reshaped = samples.view(-1, 1, size, size)
                grid = make_grid(samples_reshaped.cpu()[:16], nrow=4, normalize=True)
                axes[i].imshow(grid.permute(1, 2, 0), cmap='gray')
        
        axes[i].set_title(title)
        axes[i].axis('off')
    
    return fig

# 10. 潜在空间插值
print("\n10. 潜在空间插值")

def interpolate_latent(model, z1, z2, steps=10):
    """在潜在空间中插值"""
    alphas = np.linspace(0, 1, steps)
    interpolated_samples = []
    
    model.eval()
    with torch.no_grad():
        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            if hasattr(model, 'decoder'):
                sample = model.decoder(z_interp)
            else:
                sample = model(z_interp)
            interpolated_samples.append(sample)
    
    return torch.stack(interpolated_samples)

def visualize_interpolation(interpolated_samples, title="Latent Space Interpolation"):
    """可视化潜在空间插值"""
    fig, ax = plt.subplots(figsize=(15, 3))
    
    if interpolated_samples.dim() == 5:  # (steps, batch, channels, height, width)
        samples = interpolated_samples.squeeze(1)  # 移除batch维度
        grid = make_grid(samples.cpu(), nrow=len(samples), normalize=True)
        ax.imshow(grid.permute(1, 2, 0))
    else:
        # 处理其他格式
        size = int(np.sqrt(interpolated_samples.size(-1)))
        samples_reshaped = interpolated_samples.view(-1, 1, size, size)
        grid = make_grid(samples_reshaped.cpu(), nrow=len(samples_reshaped), normalize=True)
        ax.imshow(grid.permute(1, 2, 0), cmap='gray')
    
    ax.set_title(title)
    ax.axis('off')
    return fig

# 11. 生成质量评估
print("\n11. 生成质量评估")

class GenerationEvaluator:
    """生成模型评估器"""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_inception_score(self, samples, batch_size=32):
        """计算Inception Score（简化版本）"""
        # 这里是简化实现，实际应该使用预训练的Inception网络
        with torch.no_grad():
            # 计算样本的多样性
            mean_kl = 0
            for i in range(0, len(samples), batch_size):
                batch = samples[i:i+batch_size]
                # 简化的多样性计算
                batch_flat = batch.view(batch.size(0), -1)
                mean_kl += torch.mean(torch.var(batch_flat, dim=0))
            
            is_score = torch.exp(mean_kl / (len(samples) // batch_size))
            return is_score.item()
    
    def calculate_mode_coverage(self, generated_samples, real_samples, threshold=0.1):
        """计算模式覆盖率"""
        gen_flat = generated_samples.view(len(generated_samples), -1)
        real_flat = real_samples.view(len(real_samples), -1)
        
        # 计算生成样本到真实样本的最小距离
        covered_modes = 0
        for real_sample in real_flat:
            distances = torch.norm(gen_flat - real_sample.unsqueeze(0), dim=1)
            if torch.min(distances) < threshold:
                covered_modes += 1
        
        coverage = covered_modes / len(real_samples)
        return coverage
    
    def evaluate_model(self, model, test_data, num_samples=1000):
        """综合评估生成模型"""
        model.eval()
        
        # 生成样本
        with torch.no_grad():
            if hasattr(model, 'generate'):
                generated = model.generate(num_samples, next(model.parameters()).device)
            elif hasattr(model, 'sample'):
                generated = model.sample((num_samples, 1, 28, 28), next(model.parameters()).device)
            else:
                # 对于GAN
                noise = torch.randn(num_samples, model.noise_dim, 1, 1).to(next(model.parameters()).device)
                generated = model(noise)
        
        # 计算评估指标
        is_score = self.calculate_inception_score(generated)
        mode_coverage = self.calculate_mode_coverage(generated, test_data[:100])
        
        self.metrics = {
            'inception_score': is_score,
            'mode_coverage': mode_coverage,
            'num_samples': num_samples
        }
        
        return self.metrics

# 12. 模型比较和分析
print("\n12. 模型比较和分析")

def analyze_model_characteristics():
    """分析不同生成模型的特征"""
    characteristics = {
        'Model': ['VAE', 'GAN', 'WGAN-GP', 'Diffusion', 'Conditional GAN'],
        'Training Stability': ['High', 'Medium', 'High', 'High', 'Medium'],
        'Sample Quality': ['Medium', 'High', 'High', 'Very High', 'High'],
        'Mode Coverage': ['High', 'Low', 'Medium', 'High', 'Medium'],
        'Training Speed': ['Fast', 'Fast', 'Medium', 'Slow', 'Fast'],
        'Memory Usage': ['Low', 'Low', 'Medium', 'High', 'Low'],
        'Controllability': ['High', 'Low', 'Low', 'Medium', 'Very High']
    }
    
    import pandas as pd
    df = pd.DataFrame(characteristics)
    print("生成模型特征比较:")
    print(df.to_string(index=False))
    
    return df

# 13. 可视化分析
print("\n13. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 13.1 损失函数比较
epochs = list(range(1, 11))
vae_losses = np.exp(-np.array(epochs) * 0.1) + np.random.normal(0, 0.1, 10)
gan_d_losses = 0.5 + 0.3 * np.sin(np.array(epochs)) + np.random.normal(0, 0.05, 10)
gan_g_losses = 0.7 + 0.2 * np.cos(np.array(epochs)) + np.random.normal(0, 0.05, 10)

axes[0, 0].plot(epochs, vae_losses, 'b-', label='VAE Loss')
axes[0, 0].plot(epochs, gan_d_losses, 'r-', label='GAN D Loss')
axes[0, 0].plot(epochs, gan_g_losses, 'g-', label='GAN G Loss')
axes[0, 0].set_title('训练损失对比')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 13.2 生成质量指标
models = ['VAE', 'GAN', 'WGAN-GP', 'Diffusion']
is_scores = [2.3, 4.1, 3.8, 5.2]
fid_scores = [45, 25, 28, 15]

ax1 = axes[0, 1]
ax2 = ax1.twinx()

bars1 = ax1.bar([x - 0.2 for x in range(len(models))], is_scores, 0.4, 
               label='IS Score', alpha=0.7, color='blue')
bars2 = ax2.bar([x + 0.2 for x in range(len(models))], fid_scores, 0.4, 
               label='FID Score', alpha=0.7, color='red')

ax1.set_xlabel('Models')
ax1.set_ylabel('IS Score', color='blue')
ax2.set_ylabel('FID Score', color='red')
ax1.set_title('生成质量指标对比')
ax1.set_xticks(range(len(models)))
ax1.set_xticklabels(models)

# 13.3 模式覆盖率
coverage_data = [0.85, 0.45, 0.65, 0.90]
axes[0, 2].bar(models, coverage_data, alpha=0.7, color='green')
axes[0, 2].set_title('模式覆盖率对比')
axes[0, 2].set_ylabel('Coverage Rate')
axes[0, 2].set_ylim(0, 1)

# 13.4 潜在空间可视化（模拟）
np.random.seed(42)
z_points = np.random.multivariate_normal([0, 0], [[1, 0.3], [0.3, 1]], 200)
axes[1, 0].scatter(z_points[:, 0], z_points[:, 1], alpha=0.6, c=np.random.rand(200), cmap='viridis')
axes[1, 0].set_title('VAE潜在空间分布')
axes[1, 0].set_xlabel('Latent Dim 1')
axes[1, 0].set_ylabel('Latent Dim 2')

# 13.5 训练稳定性
training_variance = [0.1, 0.8, 0.3, 0.2, 0.6]
model_names = ['VAE', 'GAN', 'WGAN-GP', 'Diffusion', 'Conditional GAN']
axes[1, 1].bar(model_names, training_variance, alpha=0.7, color='orange')
axes[1, 1].set_title('训练稳定性对比')
axes[1, 1].set_ylabel('Training Variance')
axes[1, 1].tick_params(axis='x', rotation=45)

# 13.6 计算效率
training_times = [1.0, 1.2, 2.0, 5.0, 1.5]  # 相对时间
inference_times = [0.1, 0.05, 0.05, 1.0, 0.05]  # 相对时间

x = np.arange(len(model_names))
width = 0.35

axes[1, 2].bar(x - width/2, training_times, width, label='Training Time', alpha=0.7)
axes[1, 2].bar(x + width/2, inference_times, width, label='Inference Time', alpha=0.7)
axes[1, 2].set_title('计算效率对比')
axes[1, 2].set_ylabel('Relative Time')
axes[1, 2].set_xticks(x)
axes[1, 2].set_xticklabels(model_names, rotation=45)
axes[1, 2].legend()

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/generative_models_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 分析模型特征
model_comparison = analyze_model_characteristics()

print("\n=== 生成模型总结 ===")
print("✅ 理解生成模型的基本原理")
print("✅ 实现VAE变分自编码器")
print("✅ 构建GAN生成对抗网络")
print("✅ 掌握WGAN和条件GAN")
print("✅ 了解扩散模型基础")
print("✅ 学习生成质量评估")
print("✅ 分析不同模型特征")

print("\n关键技术:")
print("1. VAE：变分推断+重参数化技巧")
print("2. GAN：对抗训练+极大极小博弈")
print("3. WGAN：Wasserstein距离+梯度惩罚")
print("4. 条件生成：标签指导的生成")
print("5. 扩散模型：逐步去噪生成")

print("\n模型特点:")
print("1. VAE：训练稳定，生成质量中等，潜在空间平滑")
print("2. GAN：生成质量高，训练不稳定，模式崩塌风险")
print("3. WGAN：改善训练稳定性，更好的收敛保证")
print("4. Diffusion：生成质量最高，计算成本大")
print("5. 条件模型：可控生成，适用于特定任务")

print("\n实际应用:")
print("1. 图像生成：艺术创作、数据增强")
print("2. 文本生成：对话系统、内容创作")
print("3. 药物发现：分子生成优化")
print("4. 语音合成：个性化语音生成")
print("5. 视频生成：动画制作、视频编辑")

print("\n=== 练习任务 ===")
print("1. 实现StyleGAN架构")
print("2. 构建DDPM扩散模型")
print("3. 实现文本到图像生成")
print("4. 尝试NeRF神经辐射场")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现Score-based生成模型")
print("2. 研究Flow-based生成模型")
print("3. 构建多模态生成模型")
print("4. 实现生成模型的可解释性分析")
print("5. 研究生成模型的公平性和偏见问题")