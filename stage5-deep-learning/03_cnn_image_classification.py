"""
卷积神经网络图像分类
学习目标：掌握CNN的原理和在图像分类任务中的应用
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

print("=== 卷积神经网络图像分类 ===\n")

# 设备检查
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 数据准备和增强
print("1. 数据准备和增强")

# 数据变换
transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# 加载CIFAR-10数据集
trainset = torchvision.datasets.CIFAR10(root='/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/data', 
                                        train=True, download=True, transform=transform_train)
trainloader = DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/data', 
                                       train=False, download=True, transform=transform_test)
testloader = DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

# CIFAR-10类别
classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

print(f"训练集大小: {len(trainset)}")
print(f"测试集大小: {len(testset)}")
print(f"类别数: {len(classes)}")
print(f"图像尺寸: {trainset[0][0].shape}")

# 2. 数据可视化
print("\n2. 数据可视化")

def imshow(img, title=None):
    img = img / 2 + 0.5  # 反归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    if title:
        plt.title(title)

# 显示一些训练图像
dataiter = iter(trainloader)
images, labels = next(dataiter)

# 创建图像网格
fig, axes = plt.subplots(2, 8, figsize=(16, 4))
for i in range(16):
    row, col = i // 8, i % 8
    img = images[i] / 2 + 0.5  # 反归一化
    axes[row, col].imshow(np.transpose(img.numpy(), (1, 2, 0)))
    axes[row, col].set_title(f'{classes[labels[i]]}')
    axes[row, col].axis('off')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/cifar10_samples.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 3. 简单CNN模型
print("\n3. 简单CNN模型")

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块
        x = self.pool(F.relu(self.conv1(x)))  # 32x32 -> 16x16
        
        # 第二个卷积块
        x = self.pool(F.relu(self.conv2(x)))  # 16x16 -> 8x8
        
        # 第三个卷积块
        x = self.pool(F.relu(self.conv3(x)))  # 8x8 -> 4x4
        
        # 展平
        x = x.view(-1, 128 * 4 * 4)
        
        # 全连接层
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# 创建模型
simple_cnn = SimpleCNN().to(device)
print(f"简单CNN参数数量: {sum(p.numel() for p in simple_cnn.parameters())}")

# 4. 改进的CNN模型（类似LeNet）
print("\n4. 改进的CNN模型")

class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=10):
        super(ImprovedCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        
        self.conv4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        
        # 池化层
        self.pool = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
        # Dropout
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        # 第一个卷积块
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)  # 32x32 -> 16x16
        
        # 第二个卷积块
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)  # 16x16 -> 8x8
        
        # 第三个卷积块
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool(x)  # 8x8 -> 4x4
        
        # 展平和全连接
        x = x.view(-1, 256 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        
        return x

# 创建改进模型
improved_cnn = ImprovedCNN().to(device)
print(f"改进CNN参数数量: {sum(p.numel() for p in improved_cnn.parameters())}")

# 5. 残差网络块
print("\n5. 残差网络块")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 如果输入输出维度不同，需要调整
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += residual
        out = F.relu(out)
        
        return out

class SimpleResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleResNet, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # 残差块
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x

# 创建ResNet模型
resnet = SimpleResNet().to(device)
print(f"简单ResNet参数数量: {sum(p.numel() for p in resnet.parameters())}")

# 6. 训练函数
print("\n6. 训练函数")

def train_model(model, trainloader, testloader, num_epochs=10, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
    
    train_losses = []
    train_accuracies = []
    test_accuracies = []
    
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
            
            if i % 100 == 99:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(trainloader)}], '
                      f'Loss: {running_loss/100:.4f}')
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
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Accuracy: {train_accuracy:.2f}%, '
              f'Test Accuracy: {test_accuracy:.2f}%')
        
        scheduler.step()
    
    return train_losses, train_accuracies, test_accuracies

# 7. 训练简单CNN
print("\n7. 训练简单CNN")
print("开始训练简单CNN...")
simple_train_losses, simple_train_acc, simple_test_acc = train_model(
    simple_cnn, trainloader, testloader, num_epochs=5, lr=0.001)

# 8. 模型评估
print("\n8. 模型评估")

def evaluate_model(model, testloader, class_names):
    model.eval()
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # 计算准确率
    accuracy = accuracy_score(all_labels, all_predictions)
    
    # 分类报告
    print(f"总体准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(all_labels, all_predictions, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)
    
    return accuracy, cm, all_predictions, all_labels

simple_accuracy, simple_cm, simple_pred, simple_labels = evaluate_model(
    simple_cnn, testloader, classes)

# 9. 可视化结果
print("\n9. 可视化结果")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 训练曲线
axes[0, 0].plot(simple_train_acc, label='训练准确率')
axes[0, 0].plot(simple_test_acc, label='测试准确率')
axes[0, 0].set_title('简单CNN训练曲线')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('准确率 (%)')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 9.2 损失曲线
axes[0, 1].plot(simple_train_losses)
axes[0, 1].set_title('训练损失曲线')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('损失')
axes[0, 1].grid(True, alpha=0.3)

# 9.3 混淆矩阵
sns.heatmap(simple_cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=classes, yticklabels=classes, ax=axes[0, 2])
axes[0, 2].set_title('混淆矩阵')
axes[0, 2].set_xlabel('预测标签')
axes[0, 2].set_ylabel('真实标签')

# 9.4 每个类别的准确率
class_accuracies = []
for i in range(len(classes)):
    class_mask = np.array(simple_labels) == i
    class_correct = np.sum((np.array(simple_pred)[class_mask] == np.array(simple_labels)[class_mask]))
    class_total = np.sum(class_mask)
    class_acc = class_correct / class_total if class_total > 0 else 0
    class_accuracies.append(class_acc)

axes[1, 0].bar(classes, class_accuracies, alpha=0.7)
axes[1, 0].set_title('各类别准确率')
axes[1, 0].set_ylabel('准确率')
axes[1, 0].tick_params(axis='x', rotation=45)

# 9.5 预测错误样本
def show_misclassified(model, testloader, class_names, num_samples=8):
    model.eval()
    misclassified = []
    
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            
            for i in range(len(labels)):
                if predicted[i] != labels[i] and len(misclassified) < num_samples:
                    misclassified.append((
                        inputs[i].cpu(),
                        labels[i].cpu().item(),
                        predicted[i].cpu().item()
                    ))
            
            if len(misclassified) >= num_samples:
                break
    
    return misclassified

misclassified = show_misclassified(simple_cnn, testloader, classes)

for i, (img, true_label, pred_label) in enumerate(misclassified[:4]):
    row, col = (i // 2) + 1, i % 2
    if col == 0:
        ax = axes[1, 1] if row == 1 else axes[1, 2]
    else:
        ax = axes[1, 2] if row == 1 else None
        if ax is None:
            break
    
    img = img / 2 + 0.5  # 反归一化
    ax.imshow(np.transpose(img.numpy(), (1, 2, 0)))
    ax.set_title(f'True: {classes[true_label]}\nPred: {classes[pred_label]}')
    ax.axis('off')

# 如果有空余的子图，隐藏它们
if len(misclassified) < 4:
    axes[1, 2].set_visible(False)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/cnn_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 10. 保存模型
print("\n10. 保存模型")

torch.save(simple_cnn.state_dict(), 
           '/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/simple_cnn_cifar10.pth')
print("简单CNN模型已保存")

print("\n=== CNN图像分类总结 ===")
print("✅ CIFAR-10数据集加载和预处理")
print("✅ 数据增强技术应用")
print("✅ 简单CNN架构设计")
print("✅ 改进CNN和ResNet实现")
print("✅ 模型训练和优化")
print("✅ 性能评估和可视化")
print("✅ 错误分析和模型保存")

print("\n=== 练习任务 ===")
print("1. 实现更深的ResNet架构")
print("2. 尝试不同的数据增强策略")
print("3. 实现注意力机制")
print("4. 使用预训练模型进行迁移学习")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现DenseNet架构")
print("2. 研究不同的正则化技术")
print("3. 实现GradCAM可视化")
print("4. 尝试对抗训练")
print("5. 实现多尺度特征融合")