"""
综合深度学习项目：多模态情感分析
学习目标：完成一个端到端的深度学习项目，结合文本和图像进行情感分析
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import os
import re
import warnings
warnings.filterwarnings('ignore')

print("=== 综合深度学习项目：多模态情感分析 ===\n")

# 检查设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 项目背景和目标
print("1. 项目背景和目标")
print("项目背景：社交媒体平台需要分析用户发布的内容（文本+图片）的情感倾向")
print("目标：构建多模态深度学习模型，综合文本和图像信息进行情感分类")
print("类别：正面(Positive), 负面(Negative), 中性(Neutral)")
print("成功标准：准确率 > 80%")

# 2. 数据生成和准备
print("\n2. 数据生成和准备")

# 生成模拟的多模态数据
np.random.seed(42)
torch.manual_seed(42)

# 文本数据模拟
positive_words = ['happy', 'love', 'great', 'amazing', 'wonderful', 'excellent', 
                 'fantastic', 'awesome', 'perfect', 'beautiful', 'good', 'best']
negative_words = ['sad', 'hate', 'terrible', 'awful', 'horrible', 'bad', 
                 'worst', 'disgusting', 'disappointing', 'annoying', 'angry', 'upset']
neutral_words = ['okay', 'normal', 'average', 'fine', 'regular', 'typical', 
                'standard', 'usual', 'common', 'moderate', 'medium', 'general']

def generate_text(sentiment, length=20):
    """生成指定情感的文本"""
    if sentiment == 0:  # positive
        words = np.random.choice(positive_words + neutral_words, 
                               size=length, p=[0.7]*len(positive_words) + [0.3/len(neutral_words)]*len(neutral_words))
    elif sentiment == 1:  # negative  
        words = np.random.choice(negative_words + neutral_words,
                               size=length, p=[0.7]*len(negative_words) + [0.3/len(neutral_words)]*len(neutral_words))
    else:  # neutral
        words = np.random.choice(neutral_words + positive_words + negative_words,
                               size=length, p=[0.6]*len(neutral_words) + [0.2]*len(positive_words) + [0.2]*len(negative_words))
    return ' '.join(words)

# 生成数据集
n_samples = 3000
texts = []
images = []
labels = []

for i in range(n_samples):
    # 随机选择情感标签
    label = np.random.choice([0, 1, 2])  # 0: positive, 1: negative, 2: neutral
    
    # 生成文本
    text = generate_text(label, np.random.randint(10, 30))
    texts.append(text)
    
    # 生成模拟图像特征（实际项目中会是真实图像）
    if label == 0:  # positive - 明亮色彩
        img_features = np.random.normal(0.7, 0.2, (3, 64, 64))
    elif label == 1:  # negative - 暗淡色彩
        img_features = np.random.normal(0.3, 0.2, (3, 64, 64))
    else:  # neutral - 中等色彩
        img_features = np.random.normal(0.5, 0.2, (3, 64, 64))
    
    img_features = np.clip(img_features, 0, 1)
    images.append(img_features)
    labels.append(label)

# 转换为numpy数组
images = np.array(images)
labels = np.array(labels)

print(f"数据集大小: {len(texts)}")
print(f"文本示例: {texts[0]}")
print(f"图像特征形状: {images[0].shape}")
print(f"标签分布: {np.bincount(labels)}")

# 3. 文本预处理
print("\n3. 文本预处理")

# 构建词汇表
all_words = []
for text in texts:
    words = text.lower().split()
    all_words.extend(words)

vocab = list(set(all_words))
vocab_size = len(vocab)
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for word, i in word_to_idx.items()}

print(f"词汇表大小: {vocab_size}")

# 文本转换为序列
max_length = 50

def text_to_sequence(text, max_len=max_length):
    words = text.lower().split()
    sequence = [word_to_idx.get(word, 0) for word in words]
    if len(sequence) > max_len:
        sequence = sequence[:max_len]
    else:
        sequence = sequence + [0] * (max_len - len(sequence))
    return sequence

text_sequences = [text_to_sequence(text) for text in texts]
text_sequences = np.array(text_sequences)

print(f"文本序列形状: {text_sequences.shape}")

# 4. 数据集类定义
print("\n4. 数据集类定义")

class MultiModalDataset(Dataset):
    def __init__(self, texts, images, labels, transform=None):
        self.texts = texts
        self.images = images
        self.labels = labels
        self.transform = transform
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        text = torch.LongTensor(self.texts[idx])
        image = torch.FloatTensor(self.images[idx])
        label = torch.LongTensor([self.labels[idx]])
        
        if self.transform:
            image = self.transform(image)
        
        return text, image, label

# 数据分割
X_text_train, X_text_test, X_img_train, X_img_test, y_train, y_test = train_test_split(
    text_sequences, images, labels, test_size=0.2, random_state=42, stratify=labels)

# 创建数据集和数据加载器
train_dataset = MultiModalDataset(X_text_train, X_img_train, y_train)
test_dataset = MultiModalDataset(X_text_test, X_img_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

print(f"训练集大小: {len(train_dataset)}")
print(f"测试集大小: {len(test_dataset)}")

# 5. 文本编码器（LSTM）
print("\n5. 文本编码器")

class TextEncoder(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, num_layers=2):
        super(TextEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)
        
        # 使用最后一个隐藏状态（双向LSTM的前向和后向）
        # hidden shape: (num_layers*2, batch_size, hidden_dim)
        last_hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # (batch_size, hidden_dim*2)
        
        return self.dropout(last_hidden)

# 6. 图像编码器（CNN）
print("\n6. 图像编码器")

class ImageEncoder(nn.Module):
    def __init__(self, input_channels=3, hidden_dim=256):
        super(ImageEncoder, self).__init__()
        
        self.conv_layers = nn.Sequential(
            # 第一个卷积块
            nn.Conv2d(input_channels, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 64x64 -> 32x32
            
            # 第二个卷积块
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 32x32 -> 16x16
            
            # 第三个卷积块
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 16x16 -> 8x8
            
            # 第四个卷积块
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4))  # -> 4x4
        )
        
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
    
    def forward(self, x):
        # x shape: (batch_size, channels, height, width)
        features = self.conv_layers(x)
        features = features.view(features.size(0), -1)  # 展平
        return self.fc(features)

# 7. 多模态融合模型
print("\n7. 多模态融合模型")

class MultiModalSentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=128, text_hidden_dim=128, 
                 img_hidden_dim=256, fusion_dim=256, num_classes=3):
        super(MultiModalSentimentClassifier, self).__init__()
        
        # 文本和图像编码器
        self.text_encoder = TextEncoder(vocab_size, embed_dim, text_hidden_dim)
        self.image_encoder = ImageEncoder(hidden_dim=img_hidden_dim)
        
        # 融合层
        text_output_dim = text_hidden_dim * 2  # 双向LSTM
        self.fusion_layer = nn.Sequential(
            nn.Linear(text_output_dim + img_hidden_dim, fusion_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # 分类层
        self.classifier = nn.Linear(fusion_dim // 2, num_classes)
        
        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(fusion_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, text, image):
        # 编码文本和图像
        text_features = self.text_encoder(text)  # (batch_size, text_hidden_dim*2)
        image_features = self.image_encoder(image)  # (batch_size, img_hidden_dim)
        
        # 特征融合
        combined_features = torch.cat([text_features, image_features], dim=1)
        fused_features = self.fusion_layer(combined_features)
        
        # 注意力权重
        attention_weights = self.attention(fused_features)
        attended_features = fused_features * attention_weights
        
        # 分类
        logits = self.classifier(attended_features)
        
        return logits, attention_weights

# 8. 模型训练
print("\n8. 模型训练")

# 创建模型
model = MultiModalSentimentClassifier(
    vocab_size=vocab_size,
    embed_dim=128,
    text_hidden_dim=128,
    img_hidden_dim=256,
    fusion_dim=256,
    num_classes=3
).to(device)

print(f"模型参数数量: {sum(p.numel() for p in model.parameters())}")

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

# 训练函数
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (text, image, labels) in enumerate(train_loader):
        text, image, labels = text.to(device), image.to(device), labels.to(device)
        labels = labels.squeeze()
        
        optimizer.zero_grad()
        logits, attention_weights = model(text, image)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(logits.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy

# 评估函数
def evaluate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    all_attention_weights = []
    
    with torch.no_grad():
        for text, image, labels in test_loader:
            text, image, labels = text.to(device), image.to(device), labels.to(device)
            labels = labels.squeeze()
            
            logits, attention_weights = model(text, image)
            loss = criterion(logits, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_attention_weights.extend(attention_weights.cpu().numpy())
    
    avg_loss = total_loss / len(test_loader)
    accuracy = 100 * correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels, all_attention_weights

# 训练模型
num_epochs = 20
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []

print("开始训练...")
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    test_loss, test_acc, _, _, _ = evaluate(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    test_losses.append(test_loss)
    test_accuracies.append(test_acc)
    
    scheduler.step()
    
    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'  训练 - Loss: {train_loss:.4f}, Accuracy: {train_acc:.2f}%')
    print(f'  测试 - Loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%')

# 9. 最终评估
print("\n9. 最终评估")

final_test_loss, final_test_acc, predictions, true_labels, attention_weights = evaluate(
    model, test_loader, criterion, device)

print(f"最终测试准确率: {final_test_acc:.2f}%")
print(f"目标达成: {'是' if final_test_acc > 80 else '否'}")

# 分类报告
class_names = ['Positive', 'Negative', 'Neutral']
print("\n详细分类报告:")
print(classification_report(true_labels, predictions, target_names=class_names))

# 混淆矩阵
cm = confusion_matrix(true_labels, predictions)
print(f"\n混淆矩阵:\n{cm}")

# 10. 可视化分析
print("\n10. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 10.1 训练曲线
axes[0, 0].plot(train_losses, label='训练损失')
axes[0, 0].plot(test_losses, label='测试损失')
axes[0, 0].set_title('损失曲线')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 10.2 准确率曲线
axes[0, 1].plot(train_accuracies, label='训练准确率')
axes[0, 1].plot(test_accuracies, label='测试准确率')
axes[0, 1].set_title('准确率曲线')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 10.3 混淆矩阵
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names, ax=axes[0, 2])
axes[0, 2].set_title('混淆矩阵')
axes[0, 2].set_xlabel('预测标签')
axes[0, 2].set_ylabel('真实标签')

# 10.4 各类别准确率
class_accuracies = []
for i in range(len(class_names)):
    class_mask = np.array(true_labels) == i
    class_correct = np.sum((np.array(predictions)[class_mask] == np.array(true_labels)[class_mask]))
    class_total = np.sum(class_mask)
    class_acc = class_correct / class_total if class_total > 0 else 0
    class_accuracies.append(class_acc * 100)

axes[1, 0].bar(class_names, class_accuracies, alpha=0.7)
axes[1, 0].set_title('各类别准确率')
axes[1, 0].set_ylabel('准确率 (%)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 10.5 注意力权重分布
attention_weights_flat = np.array(attention_weights).flatten()
axes[1, 1].hist(attention_weights_flat, bins=30, alpha=0.7, edgecolor='black')
axes[1, 1].set_title('注意力权重分布')
axes[1, 1].set_xlabel('注意力权重')
axes[1, 1].set_ylabel('频数')
axes[1, 1].grid(True, alpha=0.3)

# 10.6 模型复杂度分析
model_components = ['Text Encoder', 'Image Encoder', 'Fusion Layer', 'Classifier']
param_counts = [
    sum(p.numel() for p in model.text_encoder.parameters()),
    sum(p.numel() for p in model.image_encoder.parameters()),
    sum(p.numel() for p in model.fusion_layer.parameters()),
    sum(p.numel() for p in model.classifier.parameters())
]

axes[1, 2].pie(param_counts, labels=model_components, autopct='%1.1f%%')
axes[1, 2].set_title('模型参数分布')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/multimodal_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 11. 模型保存和部署准备
print("\n11. 模型保存和部署准备")

# 保存模型
torch.save({
    'model_state_dict': model.state_dict(),
    'vocab_size': vocab_size,
    'word_to_idx': word_to_idx,
    'idx_to_word': idx_to_word,
    'max_length': max_length
}, '/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/multimodal_sentiment_model.pth')

print("模型已保存")

# 创建推理函数
def predict_sentiment(text, image_features, model, word_to_idx, max_length, device):
    """
    预测文本和图像的情感
    """
    model.eval()
    
    # 文本预处理
    words = text.lower().split()
    sequence = [word_to_idx.get(word, 0) for word in words]
    if len(sequence) > max_length:
        sequence = sequence[:max_length]
    else:
        sequence = sequence + [0] * (max_length - len(sequence))
    
    # 转换为张量
    text_tensor = torch.LongTensor(sequence).unsqueeze(0).to(device)
    image_tensor = torch.FloatTensor(image_features).unsqueeze(0).to(device)
    
    # 预测
    with torch.no_grad():
        logits, attention_weight = model(text_tensor, image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = torch.argmax(probabilities, dim=1).item()
    
    class_names = ['Positive', 'Negative', 'Neutral']
    confidence = probabilities[0][predicted_class].item()
    
    return class_names[predicted_class], confidence, attention_weight.item()

# 示例预测
sample_text = "this is amazing and wonderful"
sample_image = np.random.normal(0.7, 0.2, (3, 64, 64))  # 明亮图像
sample_image = np.clip(sample_image, 0, 1)

predicted_sentiment, confidence, attention = predict_sentiment(
    sample_text, sample_image, model, word_to_idx, max_length, device)

print(f"\n示例预测:")
print(f"文本: '{sample_text}'")
print(f"预测情感: {predicted_sentiment}")
print(f"置信度: {confidence:.3f}")
print(f"注意力权重: {attention:.3f}")

# 12. 项目总结
print("\n12. 项目总结")

print("=== 多模态情感分析项目总结 ===")
print(f"✅ 数据集大小: {len(texts)} 样本")
print(f"✅ 词汇表大小: {vocab_size}")
print(f"✅ 模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
print(f"✅ 最终测试准确率: {final_test_acc:.2f}%")
print(f"✅ 目标达成情况: {'已达成' if final_test_acc > 80 else '未达成'}")

print("\n模型架构亮点:")
print("1. 文本编码器：双向LSTM处理序列信息")
print("2. 图像编码器：多层CNN提取视觉特征")
print("3. 注意力机制：自适应融合多模态信息")
print("4. 端到端训练：联合优化所有组件")

print("\n技术创新:")
print("1. 多模态特征融合")
print("2. 注意力权重可视化")
print("3. 端到端深度学习流程")
print("4. 模块化架构设计")

print("\n实际应用价值:")
print("1. 社交媒体内容审核")
print("2. 品牌情感监控")
print("3. 用户体验分析")
print("4. 智能推荐系统")

print("\n下一步改进方向:")
print("1. 使用预训练的文本模型（BERT等）")
print("2. 集成更复杂的视觉模型（ResNet、ViT等）")
print("3. 引入更多模态（音频、视频等）")
print("4. 优化注意力机制设计")
print("5. 增加数据规模和多样性")

print("\n=== 深度学习项目完成 ===")
print("这个综合项目展示了现代深度学习的核心技术：")
print("- 多模态数据处理")
print("- 深度神经网络设计")
print("- 注意力机制应用")
print("- 端到端模型训练")
print("- 模型评估和部署")
print("- 实际应用场景分析")