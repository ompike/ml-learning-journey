"""
神经网络基础实现
学习目标：从零实现多层感知机，理解前向传播、反向传播和梯度下降
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import seaborn as sns

# 设置随机种子
np.random.seed(42)

print("=== 神经网络基础实现 ===\n")

class ActivationFunction:
    """激活函数类"""
    
    @staticmethod
    def sigmoid(x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid导数"""
        s = ActivationFunction.sigmoid(x)
        return s * (1 - s)
    
    @staticmethod
    def tanh(x):
        """Tanh激活函数"""
        return np.tanh(x)
    
    @staticmethod
    def tanh_derivative(x):
        """Tanh导数"""
        return 1 - np.tanh(x) ** 2
    
    @staticmethod
    def relu(x):
        """ReLU激活函数"""
        return np.maximum(0, x)
    
    @staticmethod
    def relu_derivative(x):
        """ReLU导数"""
        return (x > 0).astype(float)
    
    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """Leaky ReLU激活函数"""
        return np.where(x > 0, x, alpha * x)
    
    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Leaky ReLU导数"""
        return np.where(x > 0, 1, alpha)

class LossFunction:
    """损失函数类"""
    
    @staticmethod
    def mse(y_true, y_pred):
        """均方误差"""
        return np.mean((y_true - y_pred) ** 2)
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """均方误差导数"""
        return 2 * (y_pred - y_true) / len(y_true)
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """二元交叉熵"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_crossentropy_derivative(y_true, y_pred):
        """二元交叉熵导数"""
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / len(y_true)

class NeuralNetwork:
    """多层感知机实现"""
    
    def __init__(self, layers, activation='relu', loss='mse', learning_rate=0.01):
        """
        初始化神经网络
        
        Parameters:
        layers: list, 每层的神经元数量 [input_size, hidden1, hidden2, ..., output_size]
        activation: str, 激活函数类型
        loss: str, 损失函数类型
        learning_rate: float, 学习率
        """
        self.layers = layers
        self.activation = activation
        self.loss = loss
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.weights = []
        self.biases = []
        
        for i in range(len(layers) - 1):
            # Xavier初始化
            weight = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            bias = np.zeros((1, layers[i+1]))
            
            self.weights.append(weight)
            self.biases.append(bias)
        
        # 选择激活函数
        if activation == 'sigmoid':
            self.activation_func = ActivationFunction.sigmoid
            self.activation_derivative = ActivationFunction.sigmoid_derivative
        elif activation == 'tanh':
            self.activation_func = ActivationFunction.tanh
            self.activation_derivative = ActivationFunction.tanh_derivative
        elif activation == 'relu':
            self.activation_func = ActivationFunction.relu
            self.activation_derivative = ActivationFunction.relu_derivative
        elif activation == 'leaky_relu':
            self.activation_func = ActivationFunction.leaky_relu
            self.activation_derivative = ActivationFunction.leaky_relu_derivative
        
        # 选择损失函数
        if loss == 'mse':
            self.loss_func = LossFunction.mse
            self.loss_derivative = LossFunction.mse_derivative
        elif loss == 'binary_crossentropy':
            self.loss_func = LossFunction.binary_crossentropy
            self.loss_derivative = LossFunction.binary_crossentropy_derivative
        
        # 记录训练历史
        self.loss_history = []
        self.accuracy_history = []
    
    def forward(self, X):
        """前向传播"""
        self.layer_inputs = [X]  # 保存每层的输入
        self.layer_outputs = [X]  # 保存每层的输出
        
        current_input = X
        
        for i in range(len(self.weights)):
            # 线性变换
            z = np.dot(current_input, self.weights[i]) + self.biases[i]
            self.layer_inputs.append(z)
            
            # 激活函数
            if i == len(self.weights) - 1:  # 输出层
                if self.loss == 'binary_crossentropy':
                    # 二分类使用sigmoid
                    a = ActivationFunction.sigmoid(z)
                else:
                    # 回归任务不使用激活函数
                    a = z
            else:  # 隐藏层
                a = self.activation_func(z)
            
            self.layer_outputs.append(a)
            current_input = a
        
        return current_input
    
    def backward(self, X, y):
        """反向传播"""
        m = X.shape[0]  # 样本数量
        
        # 计算输出层误差
        output = self.layer_outputs[-1]
        
        if self.loss == 'binary_crossentropy':
            # 对于二分类+sigmoid，简化的梯度
            delta = output - y
        else:
            # 其他情况
            loss_grad = self.loss_derivative(y, output)
            if len(self.weights) > 0:  # 如果有隐藏层
                if self.loss == 'mse':
                    delta = loss_grad
                else:
                    delta = loss_grad
            else:
                delta = loss_grad
        
        # 反向传播梯度
        for i in range(len(self.weights) - 1, -1, -1):
            # 计算权重和偏置的梯度
            dW = np.dot(self.layer_outputs[i].T, delta) / m
            db = np.mean(delta, axis=0, keepdims=True)
            
            # 更新权重和偏置
            self.weights[i] -= self.learning_rate * dW
            self.biases[i] -= self.learning_rate * db
            
            # 计算下一层的误差（如果不是输入层）
            if i > 0:
                delta = np.dot(delta, self.weights[i].T) * self.activation_derivative(self.layer_inputs[i])
    
    def train(self, X, y, epochs=1000, verbose=True):
        """训练神经网络"""
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            loss = self.loss_func(y, output)
            self.loss_history.append(loss)
            
            # 计算准确率（对于分类任务）
            if self.loss == 'binary_crossentropy':
                predictions = (output > 0.5).astype(int)
                accuracy = np.mean(predictions == y)
                self.accuracy_history.append(accuracy)
            
            # 反向传播
            self.backward(X, y)
            
            # 打印进度
            if verbose and epoch % (epochs // 10) == 0:
                if self.loss == 'binary_crossentropy':
                    print(f"Epoch {epoch}: Loss = {loss:.4f}, Accuracy = {accuracy:.4f}")
                else:
                    print(f"Epoch {epoch}: Loss = {loss:.4f}")
    
    def predict(self, X):
        """预测"""
        output = self.forward(X)
        if self.loss == 'binary_crossentropy':
            return (output > 0.5).astype(int)
        else:
            return output
    
    def predict_proba(self, X):
        """预测概率"""
        return self.forward(X)

# 1. 测试不同激活函数
print("1. 激活函数可视化")

x = np.linspace(-5, 5, 100)
activations = {
    'Sigmoid': ActivationFunction.sigmoid(x),
    'Tanh': ActivationFunction.tanh(x),
    'ReLU': ActivationFunction.relu(x),
    'Leaky ReLU': ActivationFunction.leaky_relu(x)
}

plt.figure(figsize=(12, 8))
for i, (name, y) in enumerate(activations.items(), 1):
    plt.subplot(2, 2, i)
    plt.plot(x, y, linewidth=2)
    plt.title(f'{name} Activation Function')
    plt.grid(True, alpha=0.3)
    plt.xlabel('Input')
    plt.ylabel('Output')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/activation_functions.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 2. 简单回归任务
print("\n2. 简单回归任务")

# 生成回归数据
np.random.seed(42)
X_reg = np.random.uniform(-2, 2, (200, 1))
y_reg = X_reg**2 + np.random.normal(0, 0.1, (200, 1))

# 数据标准化
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_reg_scaled = scaler_X.fit_transform(X_reg)
y_reg_scaled = scaler_y.fit_transform(y_reg)

# 创建神经网络
nn_reg = NeuralNetwork([1, 10, 10, 1], activation='relu', loss='mse', learning_rate=0.01)

# 训练模型
print("训练回归神经网络...")
nn_reg.train(X_reg_scaled, y_reg_scaled, epochs=1000, verbose=True)

# 预测
y_pred_reg = nn_reg.predict(X_reg_scaled)
y_pred_reg = scaler_y.inverse_transform(y_pred_reg)

# 计算误差
mse = np.mean((y_reg - y_pred_reg) ** 2)
print(f"回归MSE: {mse:.4f}")

# 3. 二分类任务
print("\n3. 二分类任务")

# 生成分类数据
X_cls, y_cls = make_moons(n_samples=500, noise=0.1, random_state=42)
y_cls = y_cls.reshape(-1, 1)

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

# 数据标准化
scaler_cls = StandardScaler()
X_train_scaled = scaler_cls.fit_transform(X_train)
X_test_scaled = scaler_cls.transform(X_test)

# 创建分类神经网络
nn_cls = NeuralNetwork([2, 10, 5, 1], activation='relu', loss='binary_crossentropy', learning_rate=0.1)

# 训练模型
print("训练分类神经网络...")
nn_cls.train(X_train_scaled, y_train, epochs=1000, verbose=True)

# 预测
y_pred_cls = nn_cls.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred_cls)
print(f"分类准确率: {accuracy:.4f}")

# 4. 比较不同架构
print("\n4. 比较不同网络架构")

architectures = {
    'Shallow (2-5-1)': [2, 5, 1],
    'Medium (2-10-5-1)': [2, 10, 5, 1],
    'Deep (2-20-10-5-1)': [2, 20, 10, 5, 1]
}

arch_results = {}

for name, layers in architectures.items():
    nn = NeuralNetwork(layers, activation='relu', loss='binary_crossentropy', learning_rate=0.1)
    nn.train(X_train_scaled, y_train, epochs=500, verbose=False)
    y_pred = nn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    arch_results[name] = accuracy
    print(f"{name} 准确率: {accuracy:.4f}")

# 5. 可视化结果
print("\n5. 可视化结果")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 5.1 回归结果
sorted_indices = np.argsort(X_reg.ravel())
axes[0, 0].scatter(X_reg, y_reg, alpha=0.6, label='真实数据')
axes[0, 0].plot(X_reg[sorted_indices], y_pred_reg[sorted_indices], 'r-', label='神经网络预测', linewidth=2)
axes[0, 0].set_title('神经网络回归结果')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5.2 回归训练曲线
axes[0, 1].plot(nn_reg.loss_history)
axes[0, 1].set_title('回归训练损失曲线')
axes[0, 1].set_xlabel('Epoch')
axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# 5.3 分类决策边界
def plot_decision_boundary(X, y, model, scaler, ax):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    grid_points_scaled = scaler.transform(grid_points)
    Z = model.predict_proba(grid_points_scaled)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='RdYlBu', edgecolors='black')
    ax.set_title('分类决策边界')

plot_decision_boundary(X_test, y_test, nn_cls, scaler_cls, axes[0, 2])

# 5.4 分类训练曲线
axes[1, 0].plot(nn_cls.loss_history, label='Loss')
axes[1, 0].set_title('分类训练损失曲线')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('Loss')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 5.5 分类准确率曲线
axes[1, 1].plot(nn_cls.accuracy_history, 'g-', label='Accuracy')
axes[1, 1].set_title('分类训练准确率曲线')
axes[1, 1].set_xlabel('Epoch')
axes[1, 1].set_ylabel('Accuracy')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 5.6 不同架构对比
arch_names = list(arch_results.keys())
arch_accs = list(arch_results.values())
axes[1, 2].bar(arch_names, arch_accs)
axes[1, 2].set_title('不同网络架构准确率对比')
axes[1, 2].set_ylabel('准确率')
axes[1, 2].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage5-deep-learning/neural_network_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 6. 梯度检查
print("\n6. 梯度检查")

def numerical_gradient(nn, X, y, epsilon=1e-7):
    """数值梯度计算"""
    original_loss = nn.loss_func(y, nn.forward(X))
    numerical_grads = []
    
    for i, weight in enumerate(nn.weights):
        grad = np.zeros_like(weight)
        for row in range(weight.shape[0]):
            for col in range(weight.shape[1]):
                # 正向扰动
                nn.weights[i][row, col] += epsilon
                loss_plus = nn.loss_func(y, nn.forward(X))
                
                # 负向扰动
                nn.weights[i][row, col] -= 2 * epsilon
                loss_minus = nn.loss_func(y, nn.forward(X))
                
                # 恢复原值
                nn.weights[i][row, col] += epsilon
                
                # 计算数值梯度
                grad[row, col] = (loss_plus - loss_minus) / (2 * epsilon)
        
        numerical_grads.append(grad)
    
    return numerical_grads

# 创建简单网络进行梯度检查
X_small = X_train_scaled[:10]  # 使用小批量数据
y_small = y_train[:10]

nn_check = NeuralNetwork([2, 3, 1], activation='relu', loss='binary_crossentropy', learning_rate=0.1)

# 计算解析梯度
output = nn_check.forward(X_small)
nn_check.backward(X_small, y_small)
analytical_grads = [w.copy() for w in nn_check.weights]

# 计算数值梯度
numerical_grads = numerical_gradient(nn_check, X_small, y_small)

# 比较梯度
print("梯度检查结果:")
for i, (analytical, numerical) in enumerate(zip(analytical_grads, numerical_grads)):
    diff = np.abs(analytical - numerical)
    relative_error = np.abs(diff) / (np.abs(analytical) + np.abs(numerical) + 1e-8)
    max_error = np.max(relative_error)
    print(f"层{i+1} 最大相对误差: {max_error:.8f}")
    if max_error < 1e-5:
        print(f"层{i+1} 梯度检查通过 ✅")
    else:
        print(f"层{i+1} 梯度检查失败 ❌")

print("\n=== 神经网络基础总结 ===")
print("✅ 实现了多层感知机")
print("✅ 理解了前向传播和反向传播")
print("✅ 比较了不同激活函数")
print("✅ 完成了回归和分类任务")
print("✅ 进行了梯度检查验证")

print("\n=== 练习任务 ===")
print("1. 实现不同的优化算法（Adam、RMSprop等）")
print("2. 添加正则化技术（Dropout、L2等）")
print("3. 实现批量梯度下降")
print("4. 添加早停机制")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现卷积神经网络层")
print("2. 添加批量归一化")
print("3. 实现残差连接")
print("4. 研究不同的权重初始化方法")
print("5. 实现自适应学习率调度")