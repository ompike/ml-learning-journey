"""
支持向量机(SVM)从零实现
学习目标：理解SVM的核心思想，实现线性和非线性SVM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=== 支持向量机(SVM)从零实现 ===\n")

# 1. SVM理论基础
print("1. SVM理论基础")
print("SVM核心思想：")
print("- 找到最优分离超平面，使类间间隔最大")
print("- 支持向量：距离超平面最近的样本点")
print("- 核技巧：将非线性问题映射到高维空间")

# 2. 简单的线性SVM实现
print("\n2. 简单的线性SVM实现")

class LinearSVM:
    def __init__(self, learning_rate=0.001, lambda_param=0.01, n_iters=1000):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # 将标签转换为-1和1
        y_ = np.where(y <= 0, -1, 1)
        
        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0
        
        # 梯度下降
        for i in range(self.n_iters):
            for idx, x_i in enumerate(X):
                condition = y_[idx] * (np.dot(x_i, self.w) - self.b) >= 1
                
                if condition:
                    # 正确分类的点
                    self.w -= self.lr * (2 * self.lambda_param * self.w)
                else:
                    # 错误分类的点
                    self.w -= self.lr * (2 * self.lambda_param * self.w - np.dot(x_i, y_[idx]))
                    self.b -= self.lr * y_[idx]
    
    def predict(self, X):
        approx = np.dot(X, self.w) - self.b
        return np.sign(approx)

# 3. 生成线性可分数据
print("\n3. 线性SVM测试")

# 生成线性可分数据
np.random.seed(42)
X_linear, y_linear = make_classification(n_samples=200, n_features=2, n_redundant=0, 
                                        n_informative=2, n_clusters_per_class=1, 
                                        random_state=42)

# 只使用两个类别
mask = y_linear != 2
X_linear = X_linear[mask]
y_linear = y_linear[mask]

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_linear, y_linear, test_size=0.2, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练自实现的SVM
svm_custom = LinearSVM(learning_rate=0.001, lambda_param=0.01, n_iters=1000)
svm_custom.fit(X_train_scaled, y_train)

# 预测
y_pred_custom = svm_custom.predict(X_test_scaled)
accuracy_custom = accuracy_score(y_test, y_pred_custom)
print(f"自实现线性SVM准确率: {accuracy_custom:.4f}")

# 对比sklearn的SVM
svm_sklearn = SVC(kernel='linear', C=1.0)
svm_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = svm_sklearn.predict(X_test_scaled)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)
print(f"sklearn线性SVM准确率: {accuracy_sklearn:.4f}")

# 4. SMO算法实现 (简化版)
print("\n4. SMO算法实现 (简化版)")

class SimpleSMO:
    def __init__(self, C=1.0, tol=1e-3, max_passes=5):
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        
    def fit(self, X, y):
        self.X = X
        self.y = y.astype(float)
        self.m = X.shape[0]
        
        # 初始化
        self.alphas = np.zeros(self.m)
        self.b = 0
        self.w = np.zeros(X.shape[1])
        
        # 简化的SMO主循环
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            
            for i in range(self.m):
                # 计算Ei = f(xi) - yi
                Ei = self._decision_function(X[i]) - y[i]
                
                # 检查KKT条件
                if ((y[i] * Ei < -self.tol and self.alphas[i] < self.C) or 
                    (y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    
                    # 随机选择j
                    j = np.random.choice([k for k in range(self.m) if k != i])
                    Ej = self._decision_function(X[j]) - y[j]
                    
                    # 保存旧的alpha值
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # 计算边界
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # 计算eta
                    eta = 2 * np.dot(X[i], X[j]) - np.dot(X[i], X[i]) - np.dot(X[j], X[j])
                    
                    if eta >= 0:
                        continue
                    
                    # 计算新的alpha_j
                    self.alphas[j] -= y[j] * (Ei - Ej) / eta
                    
                    # 裁剪alpha_j
                    if self.alphas[j] > H:
                        self.alphas[j] = H
                    elif self.alphas[j] < L:
                        self.alphas[j] = L
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # 更新alpha_i
                    self.alphas[i] += y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # 更新阈值b
                    b1 = (self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * np.dot(X[i], X[i]) - 
                          y[j] * (self.alphas[j] - alpha_j_old) * np.dot(X[i], X[j]))
                    
                    b2 = (self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * np.dot(X[i], X[j]) - 
                          y[j] * (self.alphas[j] - alpha_j_old) * np.dot(X[j], X[j]))
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        # 计算权重向量
        self.w = np.sum((self.alphas * self.y).reshape(-1, 1) * self.X, axis=0)
        
        # 找到支持向量
        self.support_vectors = np.where(self.alphas > 1e-5)[0]
        
    def _decision_function(self, x):
        return np.dot(x, self.w) + self.b
    
    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# 5. 核函数实现
print("\n5. 核函数实现")

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        
    def _kernel_function(self, x1, x2):
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("不支持的核函数")
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y.astype(float)
        n_samples = X.shape[0]
        
        # 计算核矩阵
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])
        
        # 简化的二次规划求解（实际应用中需要更复杂的算法）
        self.alphas = np.random.random(n_samples) * 0.1
        self.b = 0
        
        # 简单的坐标下降优化
        for _ in range(100):
            for i in range(n_samples):
                prediction = np.sum(self.alphas * self.y_train * K[:, i]) + self.b
                error = prediction - self.y_train[i]
                
                if self.y_train[i] * error < 1:
                    old_alpha = self.alphas[i]
                    self.alphas[i] = min(self.C, self.alphas[i] + 0.01)
                    self.b -= 0.01 * self.y_train[i]
    
    def predict(self, X):
        predictions = []
        for x in X:
            prediction = 0
            for i in range(len(self.X_train)):
                prediction += (self.alphas[i] * self.y_train[i] * 
                             self._kernel_function(self.X_train[i], x))
            prediction += self.b
            predictions.append(np.sign(prediction))
        return np.array(predictions)

# 6. 非线性数据测试
print("\n6. 非线性数据测试")

# 生成圆形数据
X_circle, y_circle = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

X_train_circle, X_test_circle, y_train_circle, y_test_circle = train_test_split(
    X_circle, y_circle, test_size=0.2, random_state=42)

# 测试不同核函数
kernels = ['linear', 'poly', 'rbf']
kernel_results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_train_circle, y_train_circle)
    y_pred = svm.predict(X_test_circle)
    accuracy = accuracy_score(y_test_circle, y_pred)
    kernel_results[kernel] = accuracy
    print(f"{kernel.upper()}核SVM准确率: {accuracy:.4f}")

# 7. 可视化决策边界
print("\n7. 可视化决策边界")

def plot_decision_boundary(X, y, model, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    plt.title(title)
    plt.colorbar(scatter)

# 创建可视化
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 线性数据的决策边界
svm_linear = SVC(kernel='linear', C=1.0)
svm_linear.fit(X_linear, y_linear)

plt.subplot(2, 3, 1)
plot_decision_boundary(X_linear, y_linear, svm_linear, '线性SVM - 线性数据')

# 圆形数据的不同核函数
for i, kernel in enumerate(kernels):
    svm = SVC(kernel=kernel, C=1.0, gamma='scale')
    svm.fit(X_circle, y_circle)
    
    plt.subplot(2, 3, i + 2)
    plot_decision_boundary(X_circle, y_circle, svm, f'{kernel.upper()}核SVM - 圆形数据')

# SVM参数对比
plt.subplot(2, 3, 5)
C_values = [0.1, 1.0, 10.0]
C_accuracies = []
for C in C_values:
    svm = SVC(kernel='rbf', C=C, gamma='scale')
    svm.fit(X_train_circle, y_train_circle)
    y_pred = svm.predict(X_test_circle)
    accuracy = accuracy_score(y_test_circle, y_pred)
    C_accuracies.append(accuracy)

plt.bar([str(C) for C in C_values], C_accuracies)
plt.title('不同C值对RBF-SVM性能的影响')
plt.ylabel('准确率')
plt.xlabel('C值')

# Gamma参数对比
plt.subplot(2, 3, 6)
gamma_values = [0.1, 1.0, 10.0]
gamma_accuracies = []
for gamma in gamma_values:
    svm = SVC(kernel='rbf', C=1.0, gamma=gamma)
    svm.fit(X_train_circle, y_train_circle)
    y_pred = svm.predict(X_test_circle)
    accuracy = accuracy_score(y_test_circle, y_pred)
    gamma_accuracies.append(accuracy)

plt.bar([str(gamma) for gamma in gamma_values], gamma_accuracies)
plt.title('不同Gamma值对RBF-SVM性能的影响')
plt.ylabel('准确率')
plt.xlabel('Gamma值')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/svm_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 8. 多类分类SVM
print("\n8. 多类分类SVM")

# 生成多类数据
X_multi, y_multi = make_classification(n_samples=300, n_features=2, n_redundant=0, 
                                      n_informative=2, n_clusters_per_class=1, 
                                      n_classes=3, random_state=42)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# 一对一 (OvO) 策略
svm_ovo = SVC(kernel='rbf', decision_function_shape='ovo')
svm_ovo.fit(X_train_multi, y_train_multi)
y_pred_ovo = svm_ovo.predict(X_test_multi)
accuracy_ovo = accuracy_score(y_test_multi, y_pred_ovo)

# 一对其余 (OvR) 策略
svm_ovr = SVC(kernel='rbf', decision_function_shape='ovr')
svm_ovr.fit(X_train_multi, y_train_multi)
y_pred_ovr = svm_ovr.predict(X_test_multi)
accuracy_ovr = accuracy_score(y_test_multi, y_pred_ovr)

print(f"多类SVM (OvO) 准确率: {accuracy_ovo:.4f}")
print(f"多类SVM (OvR) 准确率: {accuracy_ovr:.4f}")

print("\n=== SVM总结 ===")
print("✅ 理解SVM的核心思想和数学原理")
print("✅ 实现简单的线性SVM")
print("✅ 了解SMO算法的基本思路")
print("✅ 实现和比较不同核函数")
print("✅ 分析超参数对性能的影响")
print("✅ 应用SVM解决多类分类问题")

print("\n=== 练习任务 ===")
print("1. 实现完整的SMO算法")
print("2. 尝试自定义核函数")
print("3. 研究软间隔SVM")
print("4. 实现SVM回归(SVR)")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现在线SVM学习算法")
print("2. 研究SVM的概率输出")
print("3. 实现结构化SVM")
print("4. 比较SVM与其他分类算法")
print("5. 研究大规模数据的SVM优化")