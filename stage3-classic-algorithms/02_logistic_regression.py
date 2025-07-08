"""
逻辑回归从零实现
学习目标：理解逻辑回归的数学原理，实现二元和多元逻辑回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 设置随机种子
np.random.seed(42)

print("=== 逻辑回归从零实现 ===\n")

class LogisticRegressionScratch:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6, 
                 regularization=None, lambda_reg=0.01):
        """
        逻辑回归类
        
        Parameters:
        learning_rate: 学习率
        max_iterations: 最大迭代次数
        tolerance: 收敛容忍度
        regularization: 正则化类型 ('l1', 'l2', None)
        lambda_reg: 正则化强度
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def sigmoid(self, z):
        """Sigmoid激活函数"""
        # 防止数值溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        """训练模型"""
        # 初始化参数
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # 梯度下降
        for i in range(self.max_iterations):
            # 前向传播
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = self.sigmoid(linear_pred)
            
            # 计算损失
            cost = self._compute_cost(y, predictions)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw, db = self._compute_gradients(X, y, predictions)
            
            # 更新参数
            old_weights = self.weights.copy()
            old_bias = self.bias
            
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # 检查收敛
            if (np.linalg.norm(self.weights - old_weights) < self.tolerance and 
                abs(self.bias - old_bias) < self.tolerance):
                print(f"在第{i+1}次迭代后收敛")
                break
                
        return self
    
    def _compute_cost(self, y_true, y_pred):
        """计算损失函数"""
        # 防止log(0)
        y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
        
        # 交叉熵损失
        cost = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        
        # 添加正则化项
        if self.regularization == 'l1':
            cost += self.lambda_reg * np.sum(np.abs(self.weights))
        elif self.regularization == 'l2':
            cost += self.lambda_reg * np.sum(self.weights ** 2)
            
        return cost
    
    def _compute_gradients(self, X, y_true, y_pred):
        """计算梯度"""
        n_samples = X.shape[0]
        
        # 基本梯度
        dw = (1/n_samples) * np.dot(X.T, (y_pred - y_true))
        db = (1/n_samples) * np.sum(y_pred - y_true)
        
        # 添加正则化梯度
        if self.regularization == 'l1':
            dw += self.lambda_reg * np.sign(self.weights)
        elif self.regularization == 'l2':
            dw += 2 * self.lambda_reg * self.weights
            
        return dw, db
    
    def predict_proba(self, X):
        """预测概率"""
        linear_pred = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_pred)
    
    def predict(self, X):
        """预测类别"""
        return (self.predict_proba(X) >= 0.5).astype(int)

class MultiClassLogisticRegression:
    def __init__(self, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
        """多分类逻辑回归（一对多策略）"""
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.tolerance = tolerance
        self.classifiers = {}
        self.classes = None
        
    def fit(self, X, y):
        """训练多分类模型"""
        self.classes = np.unique(y)
        
        # 为每个类别训练一个二元分类器
        for cls in self.classes:
            # 创建二元标签
            binary_y = (y == cls).astype(int)
            
            # 训练二元分类器
            classifier = LogisticRegressionScratch(
                learning_rate=self.learning_rate,
                max_iterations=self.max_iterations,
                tolerance=self.tolerance
            )
            classifier.fit(X, binary_y)
            self.classifiers[cls] = classifier
            
        return self
    
    def predict_proba(self, X):
        """预测各类别概率"""
        probabilities = np.zeros((X.shape[0], len(self.classes)))
        
        for i, cls in enumerate(self.classes):
            probabilities[:, i] = self.classifiers[cls].predict_proba(X)
            
        return probabilities
    
    def predict(self, X):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

# 1. 二元逻辑回归
print("1. 二元逻辑回归")

# 生成二分类数据
X_binary, y_binary = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                                       n_informative=2, random_state=42, n_clusters_per_class=1)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_binary, y_binary, test_size=0.2, random_state=42)

# 训练自制逻辑回归
model_binary = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000)
model_binary.fit(X_train, y_train)

# 预测
y_pred_binary = model_binary.predict(X_test)
y_pred_proba_binary = model_binary.predict_proba(X_test)

# 评估
accuracy_binary = accuracy_score(y_test, y_pred_binary)
print(f"二元逻辑回归准确率: {accuracy_binary:.4f}")

# 与sklearn对比
sklearn_model = LogisticRegression()
sklearn_model.fit(X_train, y_train)
sklearn_pred = sklearn_model.predict(X_test)
sklearn_accuracy = accuracy_score(y_test, sklearn_pred)
print(f"Scikit-learn准确率: {sklearn_accuracy:.4f}")

# 2. 正则化逻辑回归
print("\n2. 正则化逻辑回归")

# 生成高维数据
X_high_dim, y_high_dim = make_classification(n_samples=500, n_features=20, n_informative=10, 
                                           n_redundant=10, random_state=42)

X_train_hd, X_test_hd, y_train_hd, y_test_hd = train_test_split(X_high_dim, y_high_dim, 
                                                               test_size=0.2, random_state=42)

# 标准化数据
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train_hd_scaled = scaler.fit_transform(X_train_hd)
X_test_hd_scaled = scaler.transform(X_test_hd)

# 训练不同正则化的模型
models_reg = {
    'No Regularization': LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000),
    'L1 Regularization': LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000, 
                                                   regularization='l1', lambda_reg=0.01),
    'L2 Regularization': LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000, 
                                                   regularization='l2', lambda_reg=0.01)
}

for name, model in models_reg.items():
    model.fit(X_train_hd_scaled, y_train_hd)
    y_pred = model.predict(X_test_hd_scaled)
    accuracy = accuracy_score(y_test_hd, y_pred)
    print(f"{name} 准确率: {accuracy:.4f}")

# 3. 多分类逻辑回归
print("\n3. 多分类逻辑回归")

# 生成多分类数据
X_multi, y_multi = make_blobs(n_samples=600, centers=3, n_features=2, 
                             random_state=42, cluster_std=2.0)

X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
    X_multi, y_multi, test_size=0.2, random_state=42)

# 训练多分类模型
model_multi = MultiClassLogisticRegression(learning_rate=0.01, max_iterations=1000)
model_multi.fit(X_train_multi, y_train_multi)

# 预测
y_pred_multi = model_multi.predict(X_test_multi)
accuracy_multi = accuracy_score(y_test_multi, y_pred_multi)
print(f"多分类逻辑回归准确率: {accuracy_multi:.4f}")

# 4. 决策边界可视化
print("\n4. 决策边界可视化")

def plot_decision_boundary(X, y, model, title="Decision Boundary", ax=None):
    """绘制决策边界"""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    # 预测网格点
    if hasattr(model, 'predict_proba'):
        Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
        if len(Z.shape) > 1 and Z.shape[1] > 1:  # 多分类
            Z = np.argmax(Z, axis=1)
        else:  # 二分类
            Z = (Z >= 0.5).astype(int)
    else:
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    
    Z = Z.reshape(xx.shape)
    
    # 绘制决策边界
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    
    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_title(title)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    
    return ax

# 5. 学习曲线分析
print("\n5. 学习曲线分析")

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
        X_train_sub, X_val_sub, y_train_sub, y_val_sub = train_test_split(
            X_subset, y_subset, test_size=0.2, random_state=42)
        
        # 训练模型
        model = model_class(**kwargs)
        model.fit(X_train_sub, y_train_sub)
        
        # 评估
        train_pred = model.predict(X_train_sub)
        val_pred = model.predict(X_val_sub)
        
        train_score = accuracy_score(y_train_sub, train_pred)
        val_score = accuracy_score(y_val_sub, val_pred)
        
        train_scores.append(train_score)
        val_scores.append(val_score)
    
    return train_scores, val_scores

# 生成学习曲线数据
train_sizes = np.linspace(0.1, 1.0, 10)
train_scores, val_scores = plot_learning_curves(
    X_binary, y_binary, LogisticRegressionScratch, train_sizes,
    learning_rate=0.01, max_iterations=1000
)

# 6. 特征重要性分析
print("\n6. 特征重要性分析")

# 生成带有明确特征重要性的数据
n_samples = 1000
n_features = 10

# 创建特征，前5个重要，后5个不重要
X_importance = np.random.randn(n_samples, n_features)
important_weights = np.array([2, 1.5, 1, 0.8, 0.5, 0, 0, 0, 0, 0])
y_importance = (np.dot(X_importance, important_weights) + np.random.randn(n_samples) * 0.1 > 0).astype(int)

# 训练模型
model_importance = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000)
model_importance.fit(X_importance, y_importance)

print("学习到的特征权重:")
for i, weight in enumerate(model_importance.weights):
    print(f"特征 {i+1}: {weight:.3f}")

# 7. 概率校准
print("\n7. 概率校准")

def plot_calibration_curve(y_true, y_prob, n_bins=10):
    """绘制概率校准曲线"""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0  # Expected Calibration Error
    bin_centers = []
    bin_accuracies = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # 找到在当前bin内的样本
        in_bin = (y_prob > bin_lower) & (y_prob <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_prob[in_bin].mean()
            
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            
            bin_centers.append(avg_confidence_in_bin)
            bin_accuracies.append(accuracy_in_bin)
    
    return bin_centers, bin_accuracies, ece

# 计算校准曲线
y_prob_test = model_binary.predict_proba(X_test)
bin_centers, bin_accuracies, ece = plot_calibration_curve(y_test, y_prob_test)

print(f"期望校准误差 (ECE): {ece:.4f}")

# 8. 可视化结果
print("\n8. 可视化结果")

fig, axes = plt.subplots(3, 3, figsize=(18, 15))

# 8.1 二元分类决策边界
plot_decision_boundary(X_test, y_test, model_binary, "二元逻辑回归决策边界", axes[0, 0])

# 8.2 多分类决策边界
plot_decision_boundary(X_test_multi, y_test_multi, model_multi, "多分类逻辑回归决策边界", axes[0, 1])

# 8.3 训练损失曲线
axes[0, 2].plot(model_binary.cost_history)
axes[0, 2].set_title('训练损失曲线')
axes[0, 2].set_xlabel('迭代次数')
axes[0, 2].set_ylabel('损失值')
axes[0, 2].grid(True, alpha=0.3)

# 8.4 学习曲线
axes[1, 0].plot(train_sizes, train_scores, 'o-', label='训练集', linewidth=2)
axes[1, 0].plot(train_sizes, val_scores, 'o-', label='验证集', linewidth=2)
axes[1, 0].set_title('学习曲线')
axes[1, 0].set_xlabel('训练集大小比例')
axes[1, 0].set_ylabel('准确率')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 8.5 特征重要性
feature_names = [f'特征{i+1}' for i in range(len(model_importance.weights))]
axes[1, 1].bar(feature_names, np.abs(model_importance.weights))
axes[1, 1].set_title('特征重要性（权重绝对值）')
axes[1, 1].set_xlabel('特征')
axes[1, 1].set_ylabel('|权重|')
axes[1, 1].tick_params(axis='x', rotation=45)

# 8.6 正则化效果比较
regularization_strengths = [0, 0.001, 0.01, 0.1, 1.0]
l1_accuracies = []
l2_accuracies = []

for lambda_reg in regularization_strengths:
    # L1正则化
    model_l1 = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000, 
                                        regularization='l1', lambda_reg=lambda_reg)
    model_l1.fit(X_train_hd_scaled, y_train_hd)
    l1_acc = accuracy_score(y_test_hd, model_l1.predict(X_test_hd_scaled))
    l1_accuracies.append(l1_acc)
    
    # L2正则化
    model_l2 = LogisticRegressionScratch(learning_rate=0.01, max_iterations=1000, 
                                        regularization='l2', lambda_reg=lambda_reg)
    model_l2.fit(X_train_hd_scaled, y_train_hd)
    l2_acc = accuracy_score(y_test_hd, model_l2.predict(X_test_hd_scaled))
    l2_accuracies.append(l2_acc)

axes[1, 2].plot(regularization_strengths, l1_accuracies, 'o-', label='L1正则化', linewidth=2)
axes[1, 2].plot(regularization_strengths, l2_accuracies, 's-', label='L2正则化', linewidth=2)
axes[1, 2].set_title('正则化强度对准确率的影响')
axes[1, 2].set_xlabel('正则化强度')
axes[1, 2].set_ylabel('准确率')
axes[1, 2].set_xscale('log')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# 8.7 混淆矩阵
cm = confusion_matrix(y_test, y_pred_binary)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[2, 0])
axes[2, 0].set_title('混淆矩阵')
axes[2, 0].set_xlabel('预测标签')
axes[2, 0].set_ylabel('真实标签')

# 8.8 ROC曲线（简化版）
from sklearn.metrics import roc_curve, auc
fpr, tpr, _ = roc_curve(y_test, y_pred_proba_binary)
roc_auc = auc(fpr, tpr)

axes[2, 1].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
axes[2, 1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
axes[2, 1].set_xlim([0.0, 1.0])
axes[2, 1].set_ylim([0.0, 1.05])
axes[2, 1].set_xlabel('假阳性率')
axes[2, 1].set_ylabel('真阳性率')
axes[2, 1].set_title('ROC曲线')
axes[2, 1].legend(loc="lower right")
axes[2, 1].grid(True, alpha=0.3)

# 8.9 概率校准曲线
if bin_centers and bin_accuracies:
    axes[2, 2].plot([0, 1], [0, 1], 'k--', label='完美校准')
    axes[2, 2].plot(bin_centers, bin_accuracies, 'o-', label=f'模型校准 (ECE={ece:.3f})')
    axes[2, 2].set_xlabel('平均预测概率')
    axes[2, 2].set_ylabel('准确率')
    axes[2, 2].set_title('概率校准曲线')
    axes[2, 2].legend()
    axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/logistic_regression_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 练习任务 ===")
print("1. 实现Newton-Raphson方法求解逻辑回归")
print("2. 添加早停机制")
print("3. 实现弹性网络正则化")
print("4. 添加类别权重平衡")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现多项逻辑回归（softmax回归）")
print("2. 添加特征选择功能")
print("3. 实现在线学习版本")
print("4. 研究不同优化算法的效果")
print("5. 实现贝叶斯逻辑回归")