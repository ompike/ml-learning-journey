"""
PyTorch/TensorFlow框架基础
学习目标：掌握深度学习框架的核心概念和基本操作
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

print("=== PyTorch/TensorFlow框架基础 ===\n")

# 检查PyTorch版本和GPU可用性
print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

# 1. 张量基础操作
print("\n1. 张量基础操作")

# 创建张量
print("1.1 创建张量")
tensor_from_list = torch.tensor([1, 2, 3, 4])
tensor_zeros = torch.zeros(3, 4)
tensor_ones = torch.ones(2, 3)
tensor_random = torch.randn(2, 3)
tensor_range = torch.arange(0, 10, 2)

print(f"从列表创建: {tensor_from_list}")
print(f"零张量形状: {tensor_zeros.shape}")
print(f"随机张量: {tensor_random}")
print(f"范围张量: {tensor_range}")

# 张量属性
print("\n1.2 张量属性")
x = torch.randn(3, 4)
print(f"张量形状: {x.shape}")
print(f"数据类型: {x.dtype}")
print(f"设备: {x.device}")
print(f"是否需要梯度: {x.requires_grad}")

# 移动到GPU
if torch.cuda.is_available():
    x_gpu = x.to(device)
    print(f"GPU张量设备: {x_gpu.device}")

# 2. 张量运算
print("\n2. 张量运算")

a = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32)
b = torch.tensor([[5, 6], [7, 8]], dtype=torch.float32)

print(f"a = {a}")
print(f"b = {b}")

# 基本运算
print(f"加法: {a + b}")
print(f"乘法: {a * b}")
print(f"矩阵乘法: {torch.mm(a, b)}")
print(f"转置: {a.T}")

# 广播
c = torch.tensor([1, 2])
print(f"广播加法: {a + c}")

# 3. 自动微分
print("\n3. 自动微分")

# 需要梯度的张量
x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# 定义函数
z = x**2 + y**3 + 2*x*y
print(f"z = {z}")

# 反向传播
z.backward()
print(f"dz/dx = {x.grad}")
print(f"dz/dy = {y.grad}")

# 更复杂的例子
print("\n3.1 多变量函数的梯度")
x = torch.randn(3, requires_grad=True)
y = x.sum() ** 2
print(f"x = {x}")
print(f"y = {y}")

y.backward()
print(f"dy/dx = {x.grad}")

# 4. 神经网络模块
print("\n4. 神经网络模块")

# 4.1 简单的线性层
print("4.1 线性层")
linear = nn.Linear(3, 2)  # 输入3维，输出2维
x = torch.randn(5, 3)  # 批量大小5
output = linear(x)
print(f"输入形状: {x.shape}")
print(f"输出形状: {output.shape}")
print(f"权重形状: {linear.weight.shape}")
print(f"偏置形状: {linear.bias.shape}")

# 4.2 激活函数
print("\n4.2 激活函数")
x = torch.randn(5, 3)
print(f"原始数据: {x[0]}")
print(f"ReLU: {F.relu(x[0])}")
print(f"Sigmoid: {torch.sigmoid(x[0])}")
print(f"Tanh: {torch.tanh(x[0])}")

# 4.3 损失函数
print("\n4.3 损失函数")
# 分类损失
predictions = torch.randn(3, 5)  # 3个样本，5个类别
targets = torch.tensor([0, 1, 2])
ce_loss = nn.CrossEntropyLoss()(predictions, targets)
print(f"交叉熵损失: {ce_loss}")

# 回归损失
pred_reg = torch.randn(3, 1)
target_reg = torch.randn(3, 1)
mse_loss = nn.MSELoss()(pred_reg, target_reg)
print(f"均方误差损失: {mse_loss}")

# 5. 构建神经网络
print("\n5. 构建神经网络")

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 创建网络
net = SimpleNN(10, 20, 3)
print(f"网络结构: {net}")

# 打印参数
total_params = sum(p.numel() for p in net.parameters())
trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
print(f"总参数数量: {total_params}")
print(f"可训练参数数量: {trainable_params}")

# 6. 分类任务完整示例
print("\n6. 分类任务完整示例")

# 生成分类数据
X_cls, y_cls = make_classification(n_samples=1000, n_features=10, n_classes=3, 
                                   n_informative=8, random_state=42)

# 数据预处理
scaler = StandardScaler()
X_cls_scaled = scaler.fit_transform(X_cls)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(
    X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls)

# 转换为张量
X_train_tensor = torch.FloatTensor(X_train)
y_train_tensor = torch.LongTensor(y_train)
X_test_tensor = torch.FloatTensor(X_test)
y_test_tensor = torch.LongTensor(y_test)

# 创建数据加载器
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 创建模型
class ClassificationNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(ClassificationNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_classes)
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 初始化模型
model = ClassificationNet(10, 64, 3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练函数
def train_model(model, train_loader, criterion, optimizer, num_epochs=100):
    model.train()
    train_losses = []
    train_accuracies = []
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
        
        avg_loss = epoch_loss / len(train_loader)
        accuracy = 100 * correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(accuracy)
        
        if (epoch + 1) % 20 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%')
    
    return train_losses, train_accuracies

# 评估函数
def evaluate_model(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    test_loss = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            loss = criterion(outputs, target)
            test_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / total
    avg_loss = test_loss / len(test_loader)
    return avg_loss, accuracy

# 训练模型
print("开始训练...")
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, 100)

# 评估模型
test_loss, test_accuracy = evaluate_model(model, test_loader)
print(f'测试损失: {test_loss:.4f}, 测试准确率: {test_accuracy:.2f}%')

# 7. 回归任务示例
print("\n7. 回归任务示例")

# 生成回归数据
X_reg, y_reg = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)

# 数据预处理
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_reg_scaled = scaler_X.fit_transform(X_reg)
y_reg_scaled = scaler_y.fit_transform(y_reg.reshape(-1, 1)).flatten()

# 数据分割
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg_scaled, y_reg_scaled, test_size=0.2, random_state=42)

# 转换为张量
X_train_reg_tensor = torch.FloatTensor(X_train_reg)
y_train_reg_tensor = torch.FloatTensor(y_train_reg)
X_test_reg_tensor = torch.FloatTensor(X_test_reg)
y_test_reg_tensor = torch.FloatTensor(y_test_reg)

# 回归网络
class RegressionNet(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(RegressionNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

# 训练回归模型
reg_model = RegressionNet(10, 64).to(device)
reg_criterion = nn.MSELoss()
reg_optimizer = optim.Adam(reg_model.parameters(), lr=0.001)

# 创建回归数据加载器
reg_train_dataset = TensorDataset(X_train_reg_tensor, y_train_reg_tensor)
reg_train_loader = DataLoader(reg_train_dataset, batch_size=32, shuffle=True)

# 训练回归模型
reg_model.train()
reg_losses = []

for epoch in range(100):
    epoch_loss = 0
    for batch_idx, (data, target) in enumerate(reg_train_loader):
        data, target = data.to(device), target.to(device)
        
        reg_optimizer.zero_grad()
        output = reg_model(data).squeeze()
        loss = reg_criterion(output, target)
        loss.backward()
        reg_optimizer.step()
        
        epoch_loss += loss.item()
    
    avg_loss = epoch_loss / len(reg_train_loader)
    reg_losses.append(avg_loss)
    
    if (epoch + 1) % 20 == 0:
        print(f'Epoch [{epoch+1}/100], Loss: {avg_loss:.4f}')

# 评估回归模型
reg_model.eval()
with torch.no_grad():
    y_pred_reg = reg_model(X_test_reg_tensor.to(device)).cpu().numpy().squeeze()
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    print(f'回归MSE: {mse:.4f}')

# 8. 模型保存和加载
print("\n8. 模型保存和加载")

# 保存模型
torch.save(model.state_dict(), '/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/classification_model.pth')
print("分类模型已保存")

# 加载模型
loaded_model = ClassificationNet(10, 64, 3)
loaded_model.load_state_dict(torch.load('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/classification_model.pth'))
loaded_model.eval()
print("模型已加载")

# 验证加载的模型
with torch.no_grad():
    test_input = X_test_tensor[:5]
    original_output = model(test_input.to(device))
    loaded_output = loaded_model(test_input)
    print(f"原始模型输出: {original_output.cpu()}")
    print(f"加载模型输出: {loaded_output}")
    print(f"输出是否相同: {torch.allclose(original_output.cpu(), loaded_output)}")

# 9. 可视化分析
print("\n9. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 训练损失曲线
axes[0, 0].plot(train_losses)
axes[0, 0].set_title('分类训练损失')
axes[0, 0].set_xlabel('Epoch')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

# 9.2 训练准确率曲线
axes[0, 1].plot(train_accuracies)
axes[0, 1].set_title('分类训练准确率')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Accuracy (%)')
axes[0, 1].grid(True, alpha=0.3)

# 9.3 回归损失曲线
axes[0, 2].plot(reg_losses)
axes[0, 2].set_title('回归训练损失')
axes[0, 2].set_xlabel('Epoch')
axes[0, 2].set_ylabel('Loss')
axes[0, 2].grid(True, alpha=0.3)

# 9.4 激活函数可视化
x = torch.linspace(-5, 5, 100)
axes[1, 0].plot(x, F.relu(x), label='ReLU')
axes[1, 0].plot(x, torch.sigmoid(x), label='Sigmoid')
axes[1, 0].plot(x, torch.tanh(x), label='Tanh')
axes[1, 0].set_title('激活函数对比')
axes[1, 0].set_xlabel('Input')
axes[1, 0].set_ylabel('Output')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 9.5 回归预测结果
axes[1, 1].scatter(y_test_reg, y_pred_reg, alpha=0.6)
axes[1, 1].plot([y_test_reg.min(), y_test_reg.max()], [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
axes[1, 1].set_xlabel('True Values')
axes[1, 1].set_ylabel('Predicted Values')
axes[1, 1].set_title('回归预测结果')
axes[1, 1].grid(True, alpha=0.3)

# 9.6 权重分布
model_weights = []
for param in model.parameters():
    if param.requires_grad:
        model_weights.extend(param.data.cpu().numpy().flatten())

axes[1, 2].hist(model_weights, bins=50, alpha=0.7, edgecolor='black')
axes[1, 2].set_title('模型权重分布')
axes[1, 2].set_xlabel('Weight Value')
axes[1, 2].set_ylabel('Frequency')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/framework_basics_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== PyTorch框架基础总结 ===")
print("✅ 张量基础操作和属性")
print("✅ 自动微分和梯度计算")
print("✅ 神经网络模块构建")
print("✅ 分类任务完整流程")
print("✅ 回归任务实现")
print("✅ 模型保存和加载")
print("✅ 训练过程可视化")

print("\n=== 练习任务 ===")
print("1. 实现自定义损失函数")
print("2. 尝试不同的优化器")
print("3. 实现学习率调度")
print("4. 添加正则化技术")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现批量归一化")
print("2. 构建残差网络")
print("3. 实现自定义数据加载器")
print("4. 研究不同的权重初始化方法")
print("5. 实现模型的分布式训练")

# 10. TensorFlow基础示例 (如果安装了TensorFlow)
try:
    import tensorflow as tf
    print("\n10. TensorFlow基础示例")
    print(f"TensorFlow版本: {tf.__version__}")
    
    # 简单的TensorFlow模型
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(3, activation='softmax')
    ])
    
    tf_model.compile(optimizer='adam',
                     loss='sparse_categorical_crossentropy',
                     metrics=['accuracy'])
    
    print("TensorFlow模型创建成功")
    print(f"模型参数数量: {tf_model.count_params()}")
    
except ImportError:
    print("\n10. TensorFlow未安装，跳过TensorFlow示例")

print("\n=== 框架对比总结 ===")
print("PyTorch特点:")
print("  - 动态计算图，调试友好")
print("  - Pythonic API，易于学习")
print("  - 强大的自动微分系统")
print("  - 适合研究和快速原型开发")
print("\nTensorFlow特点:")
print("  - 静态计算图，部署性能好")
print("  - 完善的生态系统")
print("  - 强大的生产部署工具")
print("  - 适合大规模工业应用")