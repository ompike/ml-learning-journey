"""
机器学习中的数学原理
学习目标：理解机器学习核心算法的数学基础，连接数学理论与实际应用
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 机器学习中的数学原理 ===\n")

# 1. 主成分分析(PCA)的数学原理
print("1. 主成分分析(PCA)的数学原理")

class PCA_FromScratch:
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
    
    def fit(self, X):
        """拟合PCA模型"""
        # 1. 中心化数据
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # 2. 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)
        print(f"协方差矩阵形状: {cov_matrix.shape}")
        
        # 3. 特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # 4. 按特征值降序排列
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # 5. 选择前n_components个主成分
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum(eigenvalues)
        
        print(f"解释方差比: {self.explained_variance_ratio_}")
        
        return self
    
    def transform(self, X):
        """变换数据到主成分空间"""
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_.T)
    
    def fit_transform(self, X):
        """拟合并变换"""
        return self.fit(X).transform(X)

# 生成示例数据
np.random.seed(42)
X_pca = np.random.multivariate_normal([0, 0], [[3, 1.5], [1.5, 1]], 100)

# 应用自制PCA
pca = PCA_FromScratch(n_components=2)
X_pca_transformed = pca.fit_transform(X_pca)

print(f"原始数据形状: {X_pca.shape}")
print(f"主成分: \n{pca.components_}")

# 2. 线性回归的数学推导
print("\n2. 线性回归的数学推导")

class LinearRegression_FromScratch:
    def __init__(self, method='normal_equation'):
        self.method = method
        self.weights = None
        self.bias = None
        
    def fit(self, X, y):
        """拟合线性回归模型"""
        if self.method == 'normal_equation':
            self._fit_normal_equation(X, y)
        elif self.method == 'gradient_descent':
            self._fit_gradient_descent(X, y)
    
    def _fit_normal_equation(self, X, y):
        """正规方程解法：θ = (X^T X)^(-1) X^T y"""
        # 添加偏置项
        X_with_bias = np.c_[np.ones(X.shape[0]), X]
        
        # 计算正规方程
        XtX = X_with_bias.T @ X_with_bias
        Xty = X_with_bias.T @ y
        
        # 检查矩阵是否可逆
        if np.linalg.det(XtX) != 0:
            theta = np.linalg.inv(XtX) @ Xty
        else:
            # 使用伪逆
            theta = np.linalg.pinv(XtX) @ Xty
        
        self.bias = theta[0]
        self.weights = theta[1:]
        
        print(f"权重: {self.weights}")
        print(f"偏置: {self.bias}")
    
    def _fit_gradient_descent(self, X, y, learning_rate=0.01, max_iterations=1000):
        """梯度下降解法"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(max_iterations):
            # 前向传播
            y_pred = X @ self.weights + self.bias
            
            # 计算损失
            cost = np.mean((y_pred - y)**2)
            
            # 计算梯度
            dw = (2/n_samples) * X.T @ (y_pred - y)
            db = (2/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= learning_rate * dw
            self.bias -= learning_rate * db
            
            if i % 100 == 0:
                print(f"迭代 {i}: 损失 = {cost:.6f}")
    
    def predict(self, X):
        """预测"""
        return X @ self.weights + self.bias
    
    def r_squared(self, X, y):
        """计算R²"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

# 生成线性回归数据
X_lr = np.random.randn(100, 2)
true_weights = np.array([3, -2])
true_bias = 1
y_lr = X_lr @ true_weights + true_bias + 0.1 * np.random.randn(100)

print(f"真实权重: {true_weights}, 真实偏置: {true_bias}")

# 正规方程解法
lr_normal = LinearRegression_FromScratch(method='normal_equation')
lr_normal.fit(X_lr, y_lr)
print(f"R² (正规方程): {lr_normal.r_squared(X_lr, y_lr):.4f}")

# 3. 逻辑回归的数学基础
print("\n3. 逻辑回归的数学基础")

class LogisticRegression_FromScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000):
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.weights = None
        self.bias = None
        
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """拟合逻辑回归模型"""
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iterations):
            # 前向传播
            linear_pred = X @ self.weights + self.bias
            y_pred = self.sigmoid(linear_pred)
            
            # 计算损失（交叉熵）
            cost = self.compute_cost(y, y_pred)
            
            # 计算梯度
            dw = (1/n_samples) * X.T @ (y_pred - y)
            db = (1/n_samples) * np.sum(y_pred - y)
            
            # 更新参数
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            if i % 100 == 0:
                print(f"迭代 {i}: 损失 = {cost:.6f}")
    
    def compute_cost(self, y_true, y_pred):
        """计算交叉熵损失"""
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    def predict_proba(self, X):
        """预测概率"""
        linear_pred = X @ self.weights + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        """预测类别"""
        return (self.predict_proba(X) >= 0.5).astype(int)

# 生成逻辑回归数据
X_log, y_log = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                                  n_informative=2, random_state=42)

# 标准化数据
scaler = StandardScaler()
X_log_scaled = scaler.fit_transform(X_log)

# 拟合逻辑回归
lr_log = LogisticRegression_FromScratch(learning_rate=0.1, max_iterations=1000)
lr_log.fit(X_log_scaled, y_log)

accuracy = np.mean(lr_log.predict(X_log_scaled) == y_log)
print(f"逻辑回归准确率: {accuracy:.4f}")

# 4. 神经网络的数学基础
print("\n4. 神经网络的数学基础")

class SimpleNeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        self.learning_rate = learning_rate
        
        # 初始化权重和偏置
        self.W1 = np.random.randn(input_size, hidden_size) * 0.1
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * 0.1
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """Sigmoid激活函数"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
    
    def sigmoid_derivative(self, x):
        """Sigmoid导数"""
        return x * (1 - x)
    
    def forward(self, X):
        """前向传播"""
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)
        return self.a2
    
    def backward(self, X, y, output):
        """反向传播"""
        m = X.shape[0]
        
        # 输出层梯度
        dz2 = output - y
        dW2 = (1/m) * self.a1.T @ dz2
        db2 = (1/m) * np.sum(dz2, axis=0, keepdims=True)
        
        # 隐藏层梯度
        dz1 = (dz2 @ self.W2.T) * self.sigmoid_derivative(self.a1)
        dW1 = (1/m) * X.T @ dz1
        db1 = (1/m) * np.sum(dz1, axis=0, keepdims=True)
        
        # 更新权重
        self.W2 -= self.learning_rate * dW2
        self.b2 -= self.learning_rate * db2
        self.W1 -= self.learning_rate * dW1
        self.b1 -= self.learning_rate * db1
    
    def train(self, X, y, epochs=1000):
        """训练神经网络"""
        costs = []
        for epoch in range(epochs):
            # 前向传播
            output = self.forward(X)
            
            # 计算损失
            cost = np.mean((output - y)**2)
            costs.append(cost)
            
            # 反向传播
            self.backward(X, y, output)
            
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: 损失 = {cost:.6f}")
        
        return costs

# 生成神经网络数据
X_nn = np.random.randn(100, 2)
y_nn = ((X_nn[:, 0]**2 + X_nn[:, 1]**2) > 1).astype(int).reshape(-1, 1)

# 训练神经网络
nn = SimpleNeuralNetwork(input_size=2, hidden_size=5, output_size=1, learning_rate=0.1)
costs = nn.train(X_nn, y_nn, epochs=1000)

# 预测
predictions = nn.forward(X_nn)
accuracy_nn = np.mean((predictions > 0.5) == y_nn)
print(f"神经网络准确率: {accuracy_nn:.4f}")

# 5. 支持向量机的数学原理
print("\n5. 支持向量机的数学原理")

class SVM_FromScratch:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        """拟合SVM模型"""
        # 将标签转换为-1和1
        y_ = np.where(y <= 0, -1, 1)
        
        n_samples, n_features = X.shape
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 训练
        for i in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # 正确分类，只更新正则化项
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w)
                else:
                    # 错误分类或在边界上
                    self.w -= self.learning_rate * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.learning_rate * y_[idx]
    
    def predict(self, X):
        """预测"""
        linear_output = np.dot(X, self.w) - self.b
        return np.sign(linear_output)

# 生成SVM数据
X_svm, y_svm = make_blobs(n_samples=100, centers=2, n_features=2, 
                         random_state=42, cluster_std=1.5)

# 训练SVM
svm = SVM_FromScratch(learning_rate=0.001, lambda_param=0.01, n_iterations=1000)
svm.fit(X_svm, y_svm)

predictions_svm = svm.predict(X_svm)
accuracy_svm = np.mean(predictions_svm == np.where(y_svm == 0, -1, 1))
print(f"SVM准确率: {accuracy_svm:.4f}")

# 6. K-means聚类的数学原理
print("\n6. K-means聚类的数学原理")

class KMeans_FromScratch:
    def __init__(self, k=2, max_iterations=100, random_state=42):
        self.k = k
        self.max_iterations = max_iterations
        self.random_state = random_state
        
    def fit(self, X):
        """拟合K-means模型"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # 随机初始化聚类中心
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        for iteration in range(self.max_iterations):
            # 分配每个点到最近的聚类中心
            distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
            self.labels = np.argmin(distances, axis=0)
            
            # 更新聚类中心
            new_centroids = np.array([X[self.labels == i].mean(axis=0) 
                                    for i in range(self.k)])
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                print(f"K-means在第{iteration + 1}次迭代后收敛")
                break
                
            self.centroids = new_centroids
            
        return self
    
    def predict(self, X):
        """预测新数据点的聚类"""
        distances = np.sqrt(((X - self.centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def inertia(self, X):
        """计算簇内平方和"""
        total_inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels == i]
            if len(cluster_points) > 0:
                total_inertia += np.sum((cluster_points - self.centroids[i])**2)
        return total_inertia

# 生成聚类数据
X_kmeans, _ = make_blobs(n_samples=150, centers=3, n_features=2, 
                        random_state=42, cluster_std=1.5)

# 应用K-means
kmeans = KMeans_FromScratch(k=3, max_iterations=100)
kmeans.fit(X_kmeans)

inertia = kmeans.inertia(X_kmeans)
print(f"K-means簇内平方和: {inertia:.2f}")

# 7. 可视化数学原理
print("\n7. 可视化数学原理")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 7.1 PCA降维可视化
axes[0, 0].scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, label='原始数据')
axes[0, 0].arrow(0, 0, pca.components_[0, 0]*3, pca.components_[0, 1]*3, 
                head_width=0.1, head_length=0.1, fc='red', ec='red', label='PC1')
axes[0, 0].arrow(0, 0, pca.components_[1, 0]*2, pca.components_[1, 1]*2, 
                head_width=0.1, head_length=0.1, fc='blue', ec='blue', label='PC2')
axes[0, 0].set_title('PCA主成分')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 7.2 线性回归拟合
axes[0, 1].scatter(X_lr[:, 0], y_lr, alpha=0.6, label='数据点')
x_line = np.linspace(X_lr[:, 0].min(), X_lr[:, 0].max(), 100)
y_line = lr_normal.weights[0] * x_line + lr_normal.bias
axes[0, 1].plot(x_line, y_line, 'r-', label='拟合直线')
axes[0, 1].set_title('线性回归拟合')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 7.3 逻辑回归决策边界
h = 0.02
x_min, x_max = X_log_scaled[:, 0].min() - 1, X_log_scaled[:, 0].max() + 1
y_min, y_max = X_log_scaled[:, 1].min() - 1, X_log_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = lr_log.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

axes[0, 2].contourf(xx, yy, Z, levels=50, alpha=0.8, cmap='RdYlBu')
scatter = axes[0, 2].scatter(X_log_scaled[:, 0], X_log_scaled[:, 1], c=y_log, cmap='RdYlBu')
axes[0, 2].set_title('逻辑回归决策边界')

# 7.4 神经网络训练过程
axes[1, 0].plot(costs)
axes[1, 0].set_title('神经网络训练损失')
axes[1, 0].set_xlabel('Epoch')
axes[1, 0].set_ylabel('损失')
axes[1, 0].grid(True, alpha=0.3)

# 7.5 神经网络决策边界
x_min, x_max = X_nn[:, 0].min() - 1, X_nn[:, 0].max() + 1
y_min, y_max = X_nn[:, 1].min() - 1, X_nn[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z_nn = nn.forward(np.c_[xx.ravel(), yy.ravel()])
Z_nn = Z_nn.reshape(xx.shape)

axes[1, 1].contourf(xx, yy, Z_nn, levels=50, alpha=0.8, cmap='RdYlBu')
axes[1, 1].scatter(X_nn[:, 0], X_nn[:, 1], c=y_nn.ravel(), cmap='RdYlBu')
axes[1, 1].set_title('神经网络决策边界')

# 7.6 SVM决策边界和支持向量
# 绘制决策边界
x_min, x_max = X_svm[:, 0].min() - 1, X_svm[:, 0].max() + 1
y_min, y_max = X_svm[:, 1].min() - 1, X_svm[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))
Z_svm = svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z_svm = Z_svm.reshape(xx.shape)

axes[1, 2].contourf(xx, yy, Z_svm, alpha=0.8, cmap='RdYlBu')
axes[1, 2].scatter(X_svm[:, 0], X_svm[:, 1], c=y_svm, cmap='RdYlBu')
axes[1, 2].set_title('SVM决策边界')

# 7.7 K-means聚类结果
colors = ['red', 'blue', 'green', 'purple', 'orange']
for i in range(kmeans.k):
    cluster_points = X_kmeans[kmeans.labels == i]
    axes[2, 0].scatter(cluster_points[:, 0], cluster_points[:, 1], 
                      c=colors[i], alpha=0.6, label=f'簇 {i+1}')

axes[2, 0].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                  c='black', marker='x', s=200, linewidths=3, label='聚类中心')
axes[2, 0].set_title('K-means聚类结果')
axes[2, 0].legend()

# 7.8 不同激活函数
x_activation = np.linspace(-5, 5, 100)
sigmoid_y = 1 / (1 + np.exp(-x_activation))
tanh_y = np.tanh(x_activation)
relu_y = np.maximum(0, x_activation)

axes[2, 1].plot(x_activation, sigmoid_y, label='Sigmoid')
axes[2, 1].plot(x_activation, tanh_y, label='Tanh')
axes[2, 1].plot(x_activation, relu_y, label='ReLU')
axes[2, 1].set_title('常用激活函数')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 7.9 损失函数比较
y_true = np.array([0, 0, 1, 1])
y_pred_range = np.linspace(0.01, 0.99, 100)

mse_losses = []
cross_entropy_losses = []

for p in y_pred_range:
    # MSE损失
    mse = np.mean((y_true - p)**2)
    mse_losses.append(mse)
    
    # 交叉熵损失（假设二分类，p是正类概率）
    ce = -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))
    cross_entropy_losses.append(ce)

axes[2, 2].plot(y_pred_range, mse_losses, label='MSE损失')
axes[2, 2].plot(y_pred_range, cross_entropy_losses, label='交叉熵损失')
axes[2, 2].set_title('损失函数比较')
axes[2, 2].set_xlabel('预测概率')
axes[2, 2].set_ylabel('损失值')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage2-math-fundamentals/math_in_ml.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 总结 ===")
print("本练习展示了机器学习核心算法的数学原理：")
print("1. PCA: 特征值分解找到数据的主成分")
print("2. 线性回归: 最小二乘法和正规方程")
print("3. 逻辑回归: 最大似然估计和梯度下降")
print("4. 神经网络: 前向传播和反向传播")
print("5. SVM: 最大间隔分类器")
print("6. K-means: 最小化簇内平方和")

print("\n=== 练习任务 ===")
print("1. 实现其他降维算法（LDA、t-SNE）")
print("2. 推导并实现岭回归和Lasso回归")
print("3. 实现多层神经网络")
print("4. 研究核SVM的数学原理")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现EM算法用于高斯混合模型")
print("2. 推导决策树的信息增益公式")
print("3. 实现朴素贝叶斯的数学推导")
print("4. 研究深度学习中的正则化技术")
print("5. 实现变分自编码器的数学原理")