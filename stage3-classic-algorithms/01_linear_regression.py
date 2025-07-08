"""
线性回归从零实现
学习目标：理解线性回归的数学原理，实现梯度下降和正规方程两种求解方法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 设置随机种子
np.random.seed(42)

print("=== 线性回归从零实现 ===\n")

class LinearRegressionScratch:
    def __init__(self, learning_rate=0.01, n_iterations=1000, method='gradient_descent'):
        """
        线性回归类
        
        Parameters:
        learning_rate: 学习率
        n_iterations: 迭代次数
        method: 求解方法 ('gradient_descent' 或 'normal_equation')
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.method = method
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X, y):
        """训练模型"""
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        if self.method == 'gradient_descent':
            self._gradient_descent(X, y)
        elif self.method == 'normal_equation':
            self._normal_equation(X, y)
        else:
            raise ValueError("Method must be 'gradient_descent' or 'normal_equation'")
    
    def _gradient_descent(self, X, y):
        """梯度下降法求解"""
        n_samples = X.shape[0]
        
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算损失
            cost = np.mean((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def _normal_equation(self, X, y):
        """正规方程求解"""
        # 添加偏置项
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # 计算参数：θ = (X^T X)^(-1) X^T y
        theta = np.linalg.inv(X_with_bias.T @ X_with_bias) @ X_with_bias.T @ y
        
        self.bias = theta[0]
        self.weights = theta[1:]
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias

# 1. 单变量线性回归
print("1. 单变量线性回归")

# 生成数据
X_single = np.random.randn(100, 1)
y_single = 2 * X_single.ravel() + 3 + np.random.randn(100) * 0.1

# 训练模型
model_single = LinearRegressionScratch(learning_rate=0.1, n_iterations=1000)
model_single.fit(X_single, y_single)

# 预测
y_pred_single = model_single.predict(X_single)

# 可视化
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.scatter(X_single, y_single, alpha=0.6, label='数据点')
plt.plot(X_single, y_pred_single, 'r-', label='拟合直线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('单变量线性回归')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model_single.cost_history)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('损失函数收敛过程')
plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/01_single_variable_regression.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print(f"学习到的参数: w = {model_single.weights[0]:.3f}, b = {model_single.bias:.3f}")
print(f"真实参数: w = 2.000, b = 3.000")

# 2. 多变量线性回归
print("\n2. 多变量线性回归")

# 生成多元数据
X_multi, y_multi = make_regression(n_samples=100, n_features=3, noise=0.1, random_state=42)

# 数据标准化
X_multi_normalized = (X_multi - np.mean(X_multi, axis=0)) / np.std(X_multi, axis=0)

# 训练模型（梯度下降）
model_gd = LinearRegressionScratch(learning_rate=0.01, n_iterations=1000, method='gradient_descent')
model_gd.fit(X_multi_normalized, y_multi)

# 训练模型（正规方程）
model_ne = LinearRegressionScratch(method='normal_equation')
model_ne.fit(X_multi_normalized, y_multi)

# 预测
y_pred_gd = model_gd.predict(X_multi_normalized)
y_pred_ne = model_ne.predict(X_multi_normalized)

# 评估
mse_gd = mean_squared_error(y_multi, y_pred_gd)
mse_ne = mean_squared_error(y_multi, y_pred_ne)
r2_gd = r2_score(y_multi, y_pred_gd)
r2_ne = r2_score(y_multi, y_pred_ne)

print(f"梯度下降法 - MSE: {mse_gd:.3f}, R²: {r2_gd:.3f}")
print(f"正规方程法 - MSE: {mse_ne:.3f}, R²: {r2_ne:.3f}")

# 与scikit-learn对比
sklearn_model = LinearRegression()
sklearn_model.fit(X_multi_normalized, y_multi)
y_pred_sklearn = sklearn_model.predict(X_multi_normalized)
mse_sklearn = mean_squared_error(y_multi, y_pred_sklearn)
r2_sklearn = r2_score(y_multi, y_pred_sklearn)

print(f"Scikit-learn - MSE: {mse_sklearn:.3f}, R²: {r2_sklearn:.3f}")

# 可视化结果
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.scatter(y_multi, y_pred_gd, alpha=0.6)
plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('梯度下降法')
plt.grid(True)

plt.subplot(1, 3, 2)
plt.scatter(y_multi, y_pred_ne, alpha=0.6)
plt.plot([y_multi.min(), y_multi.max()], [y_multi.min(), y_multi.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('正规方程法')
plt.grid(True)

plt.subplot(1, 3, 3)
plt.plot(model_gd.cost_history)
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.title('梯度下降收敛过程')
plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/02_multi_variable_regression.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 3. 正则化线性回归
print("\n3. 正则化线性回归")

class RidgeRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        """
        Ridge回归（L2正则化）
        
        Parameters:
        alpha: 正则化强度
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算损失（包含正则化项）
            cost = np.mean((y_pred - y) ** 2) + self.alpha * np.sum(self.weights ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度（包含正则化项）
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + 2 * self.alpha * self.weights
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias

class LassoRegression:
    def __init__(self, alpha=1.0, learning_rate=0.01, n_iterations=1000):
        """
        Lasso回归（L1正则化）
        
        Parameters:
        alpha: 正则化强度
        """
        self.alpha = alpha
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None
        self.cost_history = []
    
    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iterations):
            # 前向传播
            y_pred = np.dot(X, self.weights) + self.bias
            
            # 计算损失（包含正则化项）
            cost = np.mean((y_pred - y) ** 2) + self.alpha * np.sum(np.abs(self.weights))
            self.cost_history.append(cost)
            
            # 计算梯度（包含正则化项）
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y)) + self.alpha * np.sign(self.weights)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
    def predict(self, X):
        """预测"""
        return np.dot(X, self.weights) + self.bias

# 生成具有噪声的数据
X_reg, y_reg = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)
X_reg_norm = (X_reg - np.mean(X_reg, axis=0)) / np.std(X_reg, axis=0)

# 训练不同的模型
models = {
    'Linear': LinearRegressionScratch(learning_rate=0.01, n_iterations=1000),
    'Ridge': RidgeRegression(alpha=1.0, learning_rate=0.01, n_iterations=1000),
    'Lasso': LassoRegression(alpha=0.1, learning_rate=0.01, n_iterations=1000)
}

# 训练和评估
results = {}
for name, model in models.items():
    model.fit(X_reg_norm, y_reg)
    y_pred = model.predict(X_reg_norm)
    mse = mean_squared_error(y_reg, y_pred)
    r2 = r2_score(y_reg, y_pred)
    results[name] = {'mse': mse, 'r2': r2, 'model': model}
    print(f"{name} - MSE: {mse:.3f}, R²: {r2:.3f}")

# 可视化权重
plt.figure(figsize=(15, 5))

for i, (name, result) in enumerate(results.items()):
    plt.subplot(1, 3, i+1)
    plt.bar(range(len(result['model'].weights)), result['model'].weights)
    plt.title(f'{name} - 权重分布')
    plt.xlabel('特征索引')
    plt.ylabel('权重值')
    plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/03_regularization_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 4. 学习曲线分析
print("\n4. 学习曲线分析")

def plot_learning_curves(X, y, model_class, train_sizes, **kwargs):
    """绘制学习曲线"""
    train_scores = []
    val_scores = []
    
    for train_size in train_sizes:
        # 使用不同大小的训练集
        n_samples = int(train_size * len(X))
        X_subset = X[:n_samples]
        y_subset = y[:n_samples]
        
        # 分割训练集和验证集
        X_train, X_val, y_train, y_val = train_test_split(X_subset, y_subset, 
                                                          test_size=0.2, random_state=42)
        
        # 训练模型
        model = model_class(**kwargs)
        model.fit(X_train, y_train)
        
        # 评估
        train_pred = model.predict(X_train)
        val_pred = model.predict(X_val)
        
        train_score = r2_score(y_train, train_pred)
        val_score = r2_score(y_val, val_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    return train_scores, val_scores

# 生成学习曲线数据
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, val_scores = plot_learning_curves(
    X_reg_norm, y_reg, LinearRegressionScratch, train_sizes,
    learning_rate=0.01, n_iterations=1000
)

# 绘制学习曲线
plt.figure(figsize=(10, 6))
plt.plot(train_sizes, train_scores, 'o-', label='训练集', linewidth=2)
plt.plot(train_sizes, val_scores, 'o-', label='验证集', linewidth=2)
plt.xlabel('训练集大小比例')
plt.ylabel('R² 分数')
plt.title('学习曲线')
plt.legend()
plt.grid(True)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/04_learning_curves.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("学习曲线分析：")
print(f"最终训练集R²: {train_scores[-1]:.3f}")
print(f"最终验证集R²: {val_scores[-1]:.3f}")
print(f"过拟合程度: {train_scores[-1] - val_scores[-1]:.3f}")

print("\n=== 练习任务 ===")
print("1. 实现弹性网络回归（ElasticNet）")
print("2. 添加特征选择功能")
print("3. 实现多项式回归")
print("4. 添加早停机制")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现随机梯度下降（SGD）")
print("2. 添加不同的损失函数")
print("3. 实现贝叶斯线性回归")
print("4. 添加特征交互项")
print("5. 实现在线学习版本")