"""
注意力机制和Transformer
学习目标：理解注意力机制原理并实现Transformer架构
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
import warnings
warnings.filterwarnings('ignore')

print("=== 注意力机制和Transformer ===\n")

# 1. 注意力机制理论
print("1. 注意力机制理论")
print("注意力机制核心思想：")
print("- 动态地关注输入的不同部分")
print("- 计算Query、Key、Value之间的相关性")
print("- 通过权重分配实现信息聚合")

print("\n注意力类型：")
print("1. 加性注意力（Additive Attention）")
print("2. 乘性注意力（Multiplicative Attention）")
print("3. 缩放点积注意力（Scaled Dot-Product Attention）")
print("4. 多头注意力（Multi-Head Attention）")

# 2. 基础注意力机制实现
print("\n2. 基础注意力机制实现")

class AdditiveAttention(nn.Module):
    """加性注意力机制"""
    def __init__(self, key_dim, query_dim, hidden_dim):
        super(AdditiveAttention, self).__init__()
        self.W_k = nn.Linear(key_dim, hidden_dim, bias=False)
        self.W_q = nn.Linear(query_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, keys, query, values, mask=None):
        # keys: (batch_size, seq_len, key_dim)
        # query: (batch_size, query_dim)
        # values: (batch_size, seq_len, value_dim)
        
        batch_size, seq_len, _ = keys.size()
        
        # 扩展query维度以匹配keys
        query = query.unsqueeze(1).expand(-1, seq_len, -1)
        
        # 计算注意力得分
        scores = self.v(torch.tanh(self.W_k(keys) + self.W_q(query)))
        scores = scores.squeeze(-1)  # (batch_size, seq_len)
        
        # 应用掩码
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        
        # 计算上下文向量
        context = torch.bmm(attention_weights.unsqueeze(1), values)
        context = context.squeeze(1)
        
        return context, attention_weights

class DotProductAttention(nn.Module):
    """点积注意力机制"""
    def __init__(self, dropout=0.0):
        super(DotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, queries, keys, values, mask=None):
        # queries: (batch_size, num_queries, d_k)
        # keys: (batch_size, num_keys, d_k)
        # values: (batch_size, num_keys, d_v)
        
        d_k = queries.size(-1)
        
        # 计算注意力得分
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(d_k)
        
        # 应用掩码
        if mask is not None:
            scores.masked_fill_(mask == 0, -1e9)
        
        # 计算注意力权重
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 计算输出
        output = torch.matmul(attention_weights, values)
        
        return output, attention_weights

# 3. 多头注意力机制
print("\n3. 多头注意力机制")

class MultiHeadAttention(nn.Module):
    """多头注意力机制"""
    def __init__(self, d_model, num_heads, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        self.attention = DotProductAttention(dropout)
        
    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # 1. 线性变换并分割成多个头
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        
        # 2. 调整掩码维度
        if mask is not None:
            mask = mask.unsqueeze(1).repeat(1, self.num_heads, 1, 1)
        
        # 3. 应用注意力机制
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        
        # 4. 拼接多个头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model)
        
        # 5. 最终线性变换
        output = self.W_o(attn_output)
        
        return output, attn_weights

# 4. 位置编码
print("\n4. 位置编码")

class PositionalEncoding(nn.Module):
    """位置编码"""
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        return x + self.pe[:x.size(0), :]

# 5. Transformer编码器层
print("\n5. Transformer编码器层")

class TransformerEncoderLayer(nn.Module):
    """Transformer编码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 自注意力 + 残差连接 + 层归一化
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 前馈网络 + 残差连接 + 层归一化
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """Transformer编码器"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过编码器层
        for layer in self.layers:
            x = layer(x, mask)
            
        return x

# 6. Transformer解码器层
print("\n6. Transformer解码器层")

class TransformerDecoderLayer(nn.Module):
    """Transformer解码器层"""
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super(TransformerDecoderLayer, self).__init__()
        
        self.self_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, num_heads, dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # 自注意力（带掩码）
        attn_output, _ = self.self_attention(x, x, x, self_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # 交叉注意力
        attn_output, _ = self.cross_attention(x, encoder_output, encoder_output, cross_mask)
        x = self.norm2(x + self.dropout(attn_output))
        
        # 前馈网络
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class TransformerDecoder(nn.Module):
    """Transformer解码器"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        self.layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, self_mask=None, cross_mask=None):
        # 嵌入 + 位置编码
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # 通过解码器层
        for layer in self.layers:
            x = layer(x, encoder_output, self_mask, cross_mask)
            
        return x

# 7. 完整的Transformer模型
print("\n7. 完整的Transformer模型")

class Transformer(nn.Module):
    """完整的Transformer模型"""
    def __init__(self, src_vocab_size, tgt_vocab_size, d_model=512, num_heads=8, 
                 num_encoder_layers=6, num_decoder_layers=6, d_ff=2048, 
                 max_len=5000, dropout=0.1):
        super(Transformer, self).__init__()
        
        self.encoder = TransformerEncoder(
            src_vocab_size, d_model, num_heads, num_encoder_layers, 
            d_ff, max_len, dropout
        )
        
        self.decoder = TransformerDecoder(
            tgt_vocab_size, d_model, num_heads, num_decoder_layers,
            d_ff, max_len, dropout
        )
        
        self.output_projection = nn.Linear(d_model, tgt_vocab_size)
        
    def create_padding_mask(self, seq, pad_idx=0):
        """创建填充掩码"""
        return (seq != pad_idx).unsqueeze(1).unsqueeze(2)
    
    def create_look_ahead_mask(self, size):
        """创建前瞻掩码"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask == 0
    
    def forward(self, src, tgt, src_pad_idx=0, tgt_pad_idx=0):
        # 创建掩码
        src_mask = self.create_padding_mask(src, src_pad_idx)
        tgt_mask = self.create_padding_mask(tgt, tgt_pad_idx)
        
        seq_len = tgt.size(1)
        look_ahead_mask = self.create_look_ahead_mask(seq_len).to(tgt.device)
        tgt_mask = tgt_mask & look_ahead_mask.unsqueeze(0)
        
        # 编码器
        encoder_output = self.encoder(src, src_mask)
        
        # 解码器
        decoder_output = self.decoder(tgt, encoder_output, tgt_mask, src_mask)
        
        # 输出投影
        output = self.output_projection(decoder_output)
        
        return output

# 8. 模拟数据和训练
print("\n8. 模拟数据和训练")

class SimpleTranslationDataset(Dataset):
    """简单的翻译数据集"""
    def __init__(self, size=1000, src_vocab_size=100, tgt_vocab_size=100, 
                 max_len=20, pad_idx=0):
        self.size = size
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.max_len = max_len
        self.pad_idx = pad_idx
        
        # 生成随机数据
        self.data = []
        for _ in range(size):
            src_len = np.random.randint(5, max_len)
            tgt_len = np.random.randint(5, max_len)
            
            src = np.random.randint(1, src_vocab_size, src_len).tolist()
            tgt = np.random.randint(1, tgt_vocab_size, tgt_len).tolist()
            
            # 填充到最大长度
            src += [pad_idx] * (max_len - len(src))
            tgt += [pad_idx] * (max_len - len(tgt))
            
            self.data.append((src[:max_len], tgt[:max_len]))
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        src, tgt = self.data[idx]
        return torch.tensor(src), torch.tensor(tgt)

# 创建数据集和数据加载器
dataset = SimpleTranslationDataset(size=500, src_vocab_size=50, tgt_vocab_size=50)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

# 初始化模型
model = Transformer(
    src_vocab_size=50,
    tgt_vocab_size=50,
    d_model=128,
    num_heads=4,
    num_encoder_layers=2,
    num_decoder_layers=2,
    d_ff=256,
    max_len=20
)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")

# 9. 注意力可视化
print("\n9. 注意力可视化")

def visualize_attention(attention_weights, input_tokens=None, output_tokens=None):
    """可视化注意力权重"""
    # attention_weights: (seq_len, seq_len)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 绘制热力图
    im = ax.imshow(attention_weights.detach().numpy(), cmap='Blues')
    
    # 设置标签
    if input_tokens:
        ax.set_xticks(range(len(input_tokens)))
        ax.set_xticklabels(input_tokens, rotation=45)
    
    if output_tokens:
        ax.set_yticks(range(len(output_tokens)))
        ax.set_yticklabels(output_tokens)
    
    # 添加数值标注
    for i in range(attention_weights.shape[0]):
        for j in range(attention_weights.shape[1]):
            text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                         ha="center", va="center", color="red", fontsize=8)
    
    ax.set_title("Attention Weights")
    ax.set_xlabel("Input Tokens")
    ax.set_ylabel("Output Tokens")
    
    plt.colorbar(im)
    plt.tight_layout()
    return fig

# 生成示例注意力权重
sample_attention = F.softmax(torch.randn(8, 8), dim=-1)
sample_tokens = [f"token_{i}" for i in range(8)]

fig = visualize_attention(sample_attention, sample_tokens, sample_tokens)
fig.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/attention_visualization.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 10. 自注意力分析
print("\n10. 自注意力分析")

class SelfAttentionAnalyzer:
    """自注意力分析器"""
    
    def __init__(self, model):
        self.model = model
        self.attention_weights = {}
        
        # 注册钩子函数
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册钩子函数来捕获注意力权重"""
        def hook_fn(name):
            def hook(module, input, output):
                if isinstance(output, tuple) and len(output) == 2:
                    self.attention_weights[name] = output[1]
            return hook
        
        # 为每个多头注意力层注册钩子
        for name, module in self.model.named_modules():
            if isinstance(module, MultiHeadAttention):
                hook = module.register_forward_hook(hook_fn(name))
                self.hooks.append(hook)
    
    def analyze_attention_patterns(self, input_seq):
        """分析注意力模式"""
        # 前向传播
        with torch.no_grad():
            _ = self.model.encoder(input_seq)
        
        # 分析注意力权重
        analysis = {}
        for name, weights in self.attention_weights.items():
            # weights: (batch_size, num_heads, seq_len, seq_len)
            if weights is not None:
                batch_size, num_heads, seq_len, _ = weights.shape
                
                # 计算平均注意力
                avg_attention = weights.mean(dim=(0, 1))  # (seq_len, seq_len)
                
                # 计算注意力熵（多样性）
                entropy = -(weights * torch.log(weights + 1e-9)).sum(dim=-1).mean()
                
                # 计算自注意力（对角线关注）
                self_attention = torch.diagonal(avg_attention, dim1=-2, dim2=-1).mean()
                
                analysis[name] = {
                    'avg_attention': avg_attention,
                    'entropy': entropy.item(),
                    'self_attention': self_attention.item()
                }
        
        return analysis
    
    def remove_hooks(self):
        """移除钩子函数"""
        for hook in self.hooks:
            hook.remove()

# 分析注意力模式
sample_input = torch.randint(1, 50, (1, 10))
analyzer = SelfAttentionAnalyzer(model)
attention_analysis = analyzer.analyze_attention_patterns(sample_input)

print("注意力分析结果：")
for layer_name, analysis in attention_analysis.items():
    print(f"{layer_name}:")
    print(f"  注意力熵: {analysis['entropy']:.4f}")
    print(f"  自注意力强度: {analysis['self_attention']:.4f}")

analyzer.remove_hooks()

# 11. Transformer变体
print("\n11. Transformer变体")

class LocalAttention(nn.Module):
    """局部注意力机制"""
    def __init__(self, d_model, num_heads, window_size, dropout=0.1):
        super(LocalAttention, self).__init__()
        self.window_size = window_size
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def create_local_mask(self, seq_len):
        """创建局部注意力掩码"""
        mask = torch.zeros(seq_len, seq_len)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = 1
        return mask.bool()
    
    def forward(self, query, key, value, mask=None):
        seq_len = query.size(1)
        local_mask = self.create_local_mask(seq_len).to(query.device)
        
        if mask is not None:
            local_mask = local_mask & mask
        
        return self.attention(query, key, value, local_mask)

class SparseAttention(nn.Module):
    """稀疏注意力机制"""
    def __init__(self, d_model, num_heads, sparsity_ratio=0.1, dropout=0.1):
        super(SparseAttention, self).__init__()
        self.sparsity_ratio = sparsity_ratio
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)
    
    def create_sparse_mask(self, seq_len):
        """创建稀疏注意力掩码"""
        mask = torch.zeros(seq_len, seq_len)
        
        # 保留对角线
        mask.fill_diagonal_(1)
        
        # 随机保留一定比例的位置
        num_keep = int(seq_len * seq_len * self.sparsity_ratio)
        flat_mask = mask.view(-1)
        indices = torch.randperm(seq_len * seq_len)[:num_keep]
        flat_mask[indices] = 1
        
        return mask.bool()
    
    def forward(self, query, key, value, mask=None):
        seq_len = query.size(1)
        sparse_mask = self.create_sparse_mask(seq_len).to(query.device)
        
        if mask is not None:
            sparse_mask = sparse_mask & mask
        
        return self.attention(query, key, value, sparse_mask)

# 12. 性能优化技术
print("\n12. 性能优化技术")

class OptimizedTransformerEncoder(nn.Module):
    """优化的Transformer编码器"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(OptimizedTransformerEncoder, self).__init__()
        
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        # 使用检查点节省内存
        self.layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.gradient_checkpointing = False
        
    def enable_gradient_checkpointing(self):
        """启用梯度检查点"""
        self.gradient_checkpointing = True
    
    def forward(self, x, mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                # 使用梯度检查点
                x = torch.utils.checkpoint.checkpoint(layer, x, mask)
            else:
                x = layer(x, mask)
                
        return x

# 13. 预训练策略
print("\n13. 预训练策略")

class MaskedLanguageModel(nn.Module):
    """掩码语言模型（BERT风格）"""
    def __init__(self, vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout=0.1):
        super(MaskedLanguageModel, self).__init__()
        
        self.encoder = TransformerEncoder(
            vocab_size, d_model, num_heads, num_layers, d_ff, max_len, dropout
        )
        self.mlm_head = nn.Linear(d_model, vocab_size)
        
    def forward(self, x, mask=None):
        # 编码
        hidden_states = self.encoder(x, mask)
        
        # MLM预测
        mlm_logits = self.mlm_head(hidden_states)
        
        return mlm_logits

class NextSentencePrediction(nn.Module):
    """下一句预测（BERT风格）"""
    def __init__(self, d_model):
        super(NextSentencePrediction, self).__init__()
        self.classifier = nn.Linear(d_model, 2)
        
    def forward(self, pooled_output):
        # pooled_output: (batch_size, d_model)
        return self.classifier(pooled_output)

# 14. 可视化分析
print("\n14. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 14.1 位置编码可视化
pos_encoding = PositionalEncoding(128, 100)
pe_sample = pos_encoding.pe[:50, :].squeeze(1)

im1 = axes[0, 0].imshow(pe_sample.T, aspect='auto', cmap='RdBu')
axes[0, 0].set_title('位置编码模式')
axes[0, 0].set_xlabel('位置')
axes[0, 0].set_ylabel('维度')
plt.colorbar(im1, ax=axes[0, 0])

# 14.2 注意力头分析
num_heads = 4
head_patterns = torch.randn(num_heads, 10, 10)
head_patterns = F.softmax(head_patterns, dim=-1)

for i in range(min(4, num_heads)):
    row = i // 2
    col = i % 2 + 1
    if row < 2 and col < 3:
        im = axes[row, col].imshow(head_patterns[i], cmap='Blues')
        axes[row, col].set_title(f'注意力头 {i+1}')
        plt.colorbar(im, ax=axes[row, col])

# 14.3 层间注意力变化
layers = ['Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']
avg_attention = [0.3, 0.5, 0.7, 0.6]
self_attention = [0.8, 0.6, 0.4, 0.5]

axes[1, 2].plot(layers, avg_attention, 'o-', label='平均注意力')
axes[1, 2].plot(layers, self_attention, 's-', label='自注意力')
axes[1, 2].set_title('层间注意力变化')
axes[1, 2].set_ylabel('注意力强度')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/transformer_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 注意力机制和Transformer总结 ===")
print("✅ 理解注意力机制的基本原理")
print("✅ 实现多种注意力机制变体")
print("✅ 构建完整的Transformer架构")
print("✅ 掌握位置编码技术")
print("✅ 分析注意力模式和可视化")
print("✅ 了解Transformer优化技术")
print("✅ 学习预训练策略")

print("\n关键技术:")
print("1. 自注意力机制：Query-Key-Value计算")
print("2. 多头注意力：并行处理不同表示子空间")
print("3. 位置编码：为序列注入位置信息")
print("4. 残差连接：解决深层网络训练问题")
print("5. 层归一化：稳定训练过程")

print("\n架构优势:")
print("1. 并行化：相比RNN可以并行计算")
print("2. 长程依赖：有效捕获远距离关系")
print("3. 可解释性：注意力权重提供解释")
print("4. 迁移学习：预训练模型效果好")
print("5. 灵活性：适用于多种任务")

print("\n实际应用:")
print("1. 机器翻译：seq2seq任务")
print("2. 语言模型：GPT系列模型")
print("3. 文本理解：BERT系列模型")
print("4. 计算机视觉：Vision Transformer")
print("5. 语音识别：Transformer-based ASR")

print("\n=== 练习任务 ===")
print("1. 实现Vision Transformer")
print("2. 构建GPT风格的生成模型")
print("3. 实现稀疏注意力机制")
print("4. 设计层次化Transformer")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现Reformer架构")
print("2. 研究线性注意力机制")
print("3. 构建多模态Transformer")
print("4. 实现知识蒸馏Transformer")
print("5. 研究Transformer的理论分析")