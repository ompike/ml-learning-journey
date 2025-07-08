"""
迁移学习和微调
学习目标：掌握预训练模型的使用、特征提取和微调技术
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
from sklearn.metrics import accuracy_score, classification_report
import os
import warnings
warnings.filterwarnings('ignore')

print("=== 迁移学习和微调 ===\n")

# 设备检查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 迁移学习理论
print("1. 迁移学习理论")
print("迁移学习核心思想：")
print("- 利用在大数据集上预训练的模型")
print("- 将学到的特征迁移到新任务")
print("- 减少训练时间和数据需求")
print("- 提升小数据集上的性能")

print("\n迁移学习策略：")
print("1. 特征提取：冻结预训练层，只训练分类器")
print("2. 微调：解冻部分层，以较小学习率训练")
print("3. 端到端微调：解冻所有层重新训练")

# 2. 数据准备
print("\n2. 数据准备")

# 数据变换
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_test = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载CIFAR-10数据集
print("加载CIFAR-10数据集...")
trainset = torchvision.datasets.CIFAR10(
    root='/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/data',
    train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(
    root='/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/data',
    train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=32, shuffle=False, num_workers=2)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
num_classes = len(classes)

print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")
print(f"类别数: {num_classes}")

# 3. 特征提取方法
print("\n3. 特征提取方法")

class FeatureExtractor(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(FeatureExtractor, self).__init__()
        
        # 使用预训练模型的特征提取部分
        self.features = nn.Sequential(*list(pretrained_model.children())[:-1])
        
        # 冻结特征提取层
        for param in self.features.parameters():
            param.requires_grad = False
        
        # 新的分类器
        # 获取特征维度
        with torch.no_grad():
            dummy_input = torch.randn(1, 3, 224, 224)
            feature_size = self.features(dummy_input).view(1, -1).size(1)
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_size, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)
        return x

# 4. 微调方法
print("\n4. 微调方法")

class FineTuningModel(nn.Module):
    def __init__(self, pretrained_model, num_classes, freeze_layers=None):
        super(FineTuningModel, self).__init__()
        
        self.backbone = pretrained_model
        
        # 替换最后的分类层
        if hasattr(self.backbone, 'fc'):
            # ResNet类型
            in_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(in_features, num_classes)
        elif hasattr(self.backbone, 'classifier'):
            # VGG类型
            if isinstance(self.backbone.classifier, nn.Sequential):
                in_features = self.backbone.classifier[-1].in_features
                self.backbone.classifier[-1] = nn.Linear(in_features, num_classes)
            else:
                in_features = self.backbone.classifier.in_features
                self.backbone.classifier = nn.Linear(in_features, num_classes)
        
        # 冻结指定层
        if freeze_layers is not None:
            self._freeze_layers(freeze_layers)
    
    def _freeze_layers(self, freeze_layers):
        """冻结指定的层"""
        if freeze_layers == 'all_except_classifier':
            # 冻结除了分类器外的所有层
            for name, param in self.backbone.named_parameters():
                if 'fc' not in name and 'classifier' not in name:
                    param.requires_grad = False
        elif isinstance(freeze_layers, int):
            # 冻结前N层
            layer_count = 0
            for param in self.backbone.parameters():
                if layer_count < freeze_layers:
                    param.requires_grad = False
                layer_count += 1
    
    def forward(self, x):
        return self.backbone(x)

# 5. 创建不同的模型
print("\n5. 创建不同的模型")

# 5.1 特征提取模型 (ResNet18)
print("5.1 创建特征提取模型...")
resnet18_pretrained = models.resnet18(pretrained=True)
feature_extractor = FeatureExtractor(resnet18_pretrained, num_classes).to(device)

print(f"特征提取模型参数数量: {sum(p.numel() for p in feature_extractor.parameters())}")
print(f"可训练参数数量: {sum(p.numel() for p in feature_extractor.parameters() if p.requires_grad)}")

# 5.2 微调模型 (ResNet18)
print("\n5.2 创建微调模型...")
resnet18_finetune = models.resnet18(pretrained=True)
finetune_model = FineTuningModel(resnet18_finetune, num_classes, 
                                freeze_layers='all_except_classifier').to(device)

print(f"微调模型参数数量: {sum(p.numel() for p in finetune_model.parameters())}")
print(f"可训练参数数量: {sum(p.numel() for p in finetune_model.parameters() if p.requires_grad)}")

# 5.3 端到端微调模型
print("\n5.3 创建端到端微调模型...")
resnet18_e2e = models.resnet18(pretrained=True)
e2e_model = FineTuningModel(resnet18_e2e, num_classes, freeze_layers=None).to(device)

print(f"端到端模型参数数量: {sum(p.numel() for p in e2e_model.parameters())}")
print(f"可训练参数数量: {sum(p.numel() for p in e2e_model.parameters() if p.requires_grad)}")

# 6. 从零训练的基线模型
print("\n6. 从零训练的基线模型")

class BaselineModel(nn.Module):
    def __init__(self, num_classes):
        super(BaselineModel, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(256, 512, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

baseline_model = BaselineModel(num_classes).to(device)
print(f"基线模型参数数量: {sum(p.numel() for p in baseline_model.parameters())}")

# 7. 训练函数
print("\n7. 训练函数")

def train_model(model, trainloader, testloader, num_epochs=10, learning_rate=0.001, 
                model_name="模型"):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
    print(f"开始训练 {model_name}...")
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for i, (inputs, labels) in enumerate(trainloader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if i % 500 == 499:
                print(f'[{epoch+1}, {i+1:5d}] loss: {running_loss/500:.3f}')
                running_loss = 0.0
        
        train_accuracy = 100 * correct / total
        train_losses.append(running_loss / len(trainloader))
        train_accuracies.append(train_accuracy)
        
        # 测试阶段
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for inputs, labels in testloader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        test_accuracy = 100 * test_correct / test_total
        test_accuracies.append(test_accuracy)
        
        print(f'Epoch [{epoch+1}/{num_epochs}] - Train Acc: {train_accuracy:.2f}%, Test Acc: {test_accuracy:.2f}%')
        
        scheduler.step()
    
    return train_losses, train_accuracies, test_accuracies

# 8. 训练各种模型
print("\n8. 训练各种模型")

results = {}

# 训练特征提取模型
print("\n8.1 训练特征提取模型")
fe_train_losses, fe_train_acc, fe_test_acc = train_model(
    feature_extractor, trainloader, testloader, num_epochs=5, 
    learning_rate=0.001, model_name="特征提取模型")
results['Feature Extraction'] = {
    'train_acc': fe_train_acc,
    'test_acc': fe_test_acc,
    'final_test_acc': fe_test_acc[-1]
}

# 训练微调模型
print("\n8.2 训练微调模型")
ft_train_losses, ft_train_acc, ft_test_acc = train_model(
    finetune_model, trainloader, testloader, num_epochs=5, 
    learning_rate=0.0001, model_name="微调模型")  # 较小的学习率
results['Fine-tuning'] = {
    'train_acc': ft_train_acc,
    'test_acc': ft_test_acc,
    'final_test_acc': ft_test_acc[-1]
}

# 训练基线模型（从零开始）
print("\n8.3 训练基线模型")
bl_train_losses, bl_train_acc, bl_test_acc = train_model(
    baseline_model, trainloader, testloader, num_epochs=5, 
    learning_rate=0.001, model_name="基线模型")
results['From Scratch'] = {
    'train_acc': bl_train_acc,
    'test_acc': bl_test_acc,
    'final_test_acc': bl_test_acc[-1]
}

# 9. 不同预训练模型比较
print("\n9. 不同预训练模型比较")

# 创建不同的预训练模型
pretrained_models = {
    'ResNet18': models.resnet18(pretrained=True),
    'ResNet34': models.resnet34(pretrained=True),
    'VGG16': models.vgg16(pretrained=True),
    'DenseNet121': models.densenet121(pretrained=True)
}

pretrained_results = {}

for model_name, pretrained_model in pretrained_models.items():
    print(f"\n测试 {model_name}...")
    
    # 创建微调模型
    model = FineTuningModel(pretrained_model, num_classes, 
                           freeze_layers='all_except_classifier').to(device)
    
    # 快速训练2个epoch
    _, _, test_acc = train_model(model, trainloader, testloader, num_epochs=2, 
                                learning_rate=0.001, model_name=model_name)
    
    pretrained_results[model_name] = {
        'final_test_acc': test_acc[-1],
        'param_count': sum(p.numel() for p in model.parameters())
    }

# 10. 层级微调实验
print("\n10. 层级微调实验")

def gradual_unfreezing_training(model, trainloader, testloader, num_epochs_per_stage=2):
    """渐进式解冻训练"""
    criterion = nn.CrossEntropyLoss()
    
    # 阶段1：只训练分类器
    print("阶段1：只训练分类器")
    for param in model.parameters():
        param.requires_grad = False
    
    # 解冻分类器
    if hasattr(model.backbone, 'fc'):
        for param in model.backbone.fc.parameters():
            param.requires_grad = True
    elif hasattr(model.backbone, 'classifier'):
        for param in model.backbone.classifier.parameters():
            param.requires_grad = True
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.001)
    
    for epoch in range(num_epochs_per_stage):
        # 简化的训练循环
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            if i > 100:  # 只训练部分batch
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 阶段2：解冻最后几层
    print("阶段2：解冻最后几层")
    layer_count = 0
    for param in reversed(list(model.parameters())):
        if layer_count < 10:  # 解冻最后10层
            param.requires_grad = True
        layer_count += 1
    
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), 
                          lr=0.0001)
    
    for epoch in range(num_epochs_per_stage):
        model.train()
        for i, (inputs, labels) in enumerate(trainloader):
            if i > 100:
                break
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
    # 测试最终性能
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    final_accuracy = 100 * correct / total
    print(f"渐进式解冻最终准确率: {final_accuracy:.2f}%")
    
    return final_accuracy

# 测试渐进式解冻
resnet_gradual = models.resnet18(pretrained=True)
gradual_model = FineTuningModel(resnet_gradual, num_classes, 
                               freeze_layers='all_except_classifier').to(device)
gradual_acc = gradual_unfreezing_training(gradual_model, trainloader, testloader)

# 11. 可视化结果
print("\n11. 可视化结果")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 11.1 训练过程对比
epochs = range(1, len(fe_test_acc) + 1)
axes[0, 0].plot(epochs, fe_test_acc, label='特征提取', marker='o')
axes[0, 0].plot(epochs, ft_test_acc, label='微调', marker='s')
axes[0, 0].plot(epochs, bl_test_acc, label='从零训练', marker='^')
axes[0, 0].set_title('不同方法的测试准确率')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('测试准确率 (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 11.2 最终性能对比
methods = list(results.keys())
final_accs = [results[method]['final_test_acc'] for method in methods]
colors = ['skyblue', 'lightcoral', 'lightgreen']

bars = axes[0, 1].bar(methods, final_accs, color=colors, alpha=0.8)
axes[0, 1].set_title('不同方法最终测试准确率')
axes[0, 1].set_ylabel('测试准确率 (%)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 添加数值标签
for bar, acc in zip(bars, final_accs):
    axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                   f'{acc:.1f}%', ha='center', va='bottom')

# 11.3 不同预训练模型对比
model_names = list(pretrained_results.keys())
model_accs = [pretrained_results[name]['final_test_acc'] for name in model_names]

axes[0, 2].bar(model_names, model_accs, alpha=0.7)
axes[0, 2].set_title('不同预训练模型性能对比')
axes[0, 2].set_ylabel('测试准确率 (%)')
axes[0, 2].tick_params(axis='x', rotation=45)

# 11.4 参数数量对比
param_counts = [pretrained_results[name]['param_count'] for name in model_names]
param_counts_millions = [count / 1e6 for count in param_counts]

axes[1, 0].bar(model_names, param_counts_millions, alpha=0.7, color='orange')
axes[1, 0].set_title('不同模型参数数量对比')
axes[1, 0].set_ylabel('参数数量 (百万)')
axes[1, 0].tick_params(axis='x', rotation=45)

# 11.5 准确率vs参数数量散点图
axes[1, 1].scatter(param_counts_millions, model_accs, s=100, alpha=0.7)
for i, name in enumerate(model_names):
    axes[1, 1].annotate(name, (param_counts_millions[i], model_accs[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[1, 1].set_xlabel('参数数量 (百万)')
axes[1, 1].set_ylabel('测试准确率 (%)')
axes[1, 1].set_title('模型复杂度 vs 性能')
axes[1, 1].grid(True, alpha=0.3)

# 11.6 学习策略对比
strategies = ['从零训练', '特征提取', '微调', '渐进解冻']
strategy_accs = [bl_test_acc[-1], fe_test_acc[-1], ft_test_acc[-1], gradual_acc]

axes[1, 2].bar(strategies, strategy_accs, alpha=0.7, 
               color=['red', 'blue', 'green', 'purple'])
axes[1, 2].set_title('不同学习策略对比')
axes[1, 2].set_ylabel('测试准确率 (%)')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/transfer_learning_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 12. 保存最佳模型
print("\n12. 保存最佳模型")

# 找到最佳模型
best_method = max(results.keys(), key=lambda x: results[x]['final_test_acc'])
print(f"最佳方法: {best_method}, 准确率: {results[best_method]['final_test_acc']:.2f}%")

# 保存模型
if best_method == 'Feature Extraction':
    best_model = feature_extractor
elif best_method == 'Fine-tuning':
    best_model = finetune_model
else:
    best_model = baseline_model

torch.save({
    'model_state_dict': best_model.state_dict(),
    'method': best_method,
    'accuracy': results[best_method]['final_test_acc'],
    'num_classes': num_classes,
    'classes': classes
}, '/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/best_transfer_model.pth')

print("最佳迁移学习模型已保存")

# 13. 特征可视化
print("\n13. 特征可视化")

def extract_features(model, dataloader, num_samples=100):
    """提取模型特征用于可视化"""
    model.eval()
    features = []
    labels = []
    
    with torch.no_grad():
        count = 0
        for inputs, targets in dataloader:
            if count >= num_samples:
                break
            
            inputs = inputs.to(device)
            
            # 获取特征（在最后一层之前）
            if hasattr(model, 'features'):
                feat = model.features(inputs)
            else:
                feat = model.backbone.features(inputs) if hasattr(model.backbone, 'features') else inputs
            
            feat = feat.view(feat.size(0), -1)
            features.append(feat.cpu().numpy())
            labels.append(targets.numpy())
            
            count += len(targets)
    
    return np.vstack(features), np.hstack(labels)

# 提取特征并降维可视化
try:
    from sklearn.manifold import TSNE
    
    # 提取特征
    features, labels = extract_features(best_model, testloader, num_samples=500)
    
    # t-SNE降维
    print("进行t-SNE降维...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    features_2d = tsne.fit_transform(features[:500])  # 限制样本数
    
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=labels[:500], cmap='tab10', alpha=0.7)
    plt.colorbar(scatter, ticks=range(num_classes), 
                label='Classes', format=plt.FuncFormatter(lambda x, p: classes[int(x)]))
    plt.title(f'{best_method} 模型特征可视化 (t-SNE)')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    plt.grid(True, alpha=0.3)
    
    plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/feature_visualization.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print("特征可视化已保存")
    
except ImportError:
    print("sklearn未安装，跳过特征可视化")

print("\n=== 迁移学习总结 ===")
print("✅ 理解迁移学习的基本概念和策略")
print("✅ 实现特征提取和微调方法")
print("✅ 比较不同预训练模型的性能")
print("✅ 掌握渐进式解冻训练技巧")
print("✅ 分析模型复杂度与性能的关系")
print("✅ 实现特征可视化分析")

print("\n关键技术点:")
print("1. 预训练模型的选择和加载")
print("2. 层级冻结和解冻策略")
print("3. 不同学习率的设置")
print("4. 数据增强的重要性")
print("5. 渐进式训练策略")

print("\n实际应用建议:")
print("1. 数据量少时优先使用特征提取")
print("2. 数据量适中时使用微调")
print("3. 任务相似度高时可以解冻更多层")
print("4. 使用较小的学习率进行微调")
print("5. 考虑渐进式解冻以获得更好效果")

print("\n=== 练习任务 ===")
print("1. 尝试不同的预训练模型架构")
print("2. 实现多任务学习")
print("3. 研究领域适应技术")
print("4. 实现知识蒸馏")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现自监督预训练")
print("2. 研究零样本学习")
print("3. 实现元学习算法")
print("4. 构建多模态迁移学习")
print("5. 研究神经架构搜索在迁移学习中的应用")