"""
K近邻算法从零实现
学习目标：理解KNN的原理、距离度量和参数优化
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from collections import Counter
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== K近邻算法从零实现 ===\n")

# 1. KNN算法理论
print("1. KNN算法理论")
print("KNN核心思想：")
print("- 样本的类别由其K个最近邻居的类别决定")
print("- 懒惰学习：训练时不建模，预测时计算")
print("- 非参数方法：不假设数据分布")
print("- 局部方法：决策基于局部邻域")

# 2. 距离度量函数
print("\n2. 距离度量函数")

class DistanceMetrics:
    @staticmethod
    def euclidean_distance(x1, x2):
        """欧几里得距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    @staticmethod
    def manhattan_distance(x1, x2):
        """曼哈顿距离"""
        return np.sum(np.abs(x1 - x2))
    
    @staticmethod
    def minkowski_distance(x1, x2, p=2):
        """闵可夫斯基距离"""
        return np.sum(np.abs(x1 - x2) ** p) ** (1/p)
    
    @staticmethod
    def cosine_distance(x1, x2):
        """余弦距离"""
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        if norm_x1 == 0 or norm_x2 == 0:
            return 1.0
        return 1 - dot_product / (norm_x1 * norm_x2)
    
    @staticmethod
    def chebyshev_distance(x1, x2):
        """切比雪夫距离"""
        return np.max(np.abs(x1 - x2))

# 3. KNN分类器实现
print("\n3. KNN分类器实现")

class KNNClassifierFromScratch:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
        # 选择距离函数
        self.distance_functions = {
            'euclidean': DistanceMetrics.euclidean_distance,
            'manhattan': DistanceMetrics.manhattan_distance,
            'cosine': DistanceMetrics.cosine_distance,
            'chebyshev': DistanceMetrics.chebyshev_distance
        }
        self.distance_func = self.distance_functions[distance_metric]
    
    def fit(self, X, y):
        """训练（实际上只是存储数据）"""
        self.X_train = X.copy()
        self.y_train = y.copy()
        self.classes = np.unique(y)
    
    def _get_neighbors(self, x):
        """获取K个最近邻居"""
        distances = []
        
        for i, x_train in enumerate(self.X_train):
            dist = self.distance_func(x, x_train)
            distances.append((dist, i))
        
        # 按距离排序并取前K个
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        return neighbors
    
    def _predict_single(self, x):
        """预测单个样本"""
        neighbors = self._get_neighbors(x)
        
        if self.weights == 'uniform':
            # 等权重投票
            neighbor_labels = [self.y_train[i] for _, i in neighbors]
            most_common = Counter(neighbor_labels).most_common(1)
            return most_common[0][0]
        
        elif self.weights == 'distance':
            # 距离加权投票
            class_weights = {}
            for dist, i in neighbors:
                label = self.y_train[i]
                weight = 1 / (dist + 1e-10)  # 避免除零
                
                if label in class_weights:
                    class_weights[label] += weight
                else:
                    class_weights[label] = weight
            
            return max(class_weights.keys(), key=lambda x: class_weights[x])
    
    def predict(self, X):
        """预测多个样本"""
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return np.array(predictions)
    
    def predict_proba(self, X):
        """预测概率"""
        probabilities = []
        
        for x in X:
            neighbors = self._get_neighbors(x)
            class_counts = {cls: 0 for cls in self.classes}
            
            if self.weights == 'uniform':
                for _, i in neighbors:
                    class_counts[self.y_train[i]] += 1
            else:
                for dist, i in neighbors:
                    weight = 1 / (dist + 1e-10)
                    class_counts[self.y_train[i]] += weight
            
            # 归一化为概率
            total = sum(class_counts.values())
            if total > 0:
                probs = [class_counts[cls] / total for cls in self.classes]
            else:
                probs = [1.0 / len(self.classes)] * len(self.classes)
            
            probabilities.append(probs)
        
        return np.array(probabilities)

# 4. KNN回归器实现
print("\n4. KNN回归器实现")

class KNNRegressorFromScratch:
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        self.X_train = None
        self.y_train = None
        
        self.distance_functions = {
            'euclidean': DistanceMetrics.euclidean_distance,
            'manhattan': DistanceMetrics.manhattan_distance,
            'cosine': DistanceMetrics.cosine_distance,
            'chebyshev': DistanceMetrics.chebyshev_distance
        }
        self.distance_func = self.distance_functions[distance_metric]
    
    def fit(self, X, y):
        """训练"""
        self.X_train = X.copy()
        self.y_train = y.copy()
    
    def _get_neighbors(self, x):
        """获取K个最近邻居"""
        distances = []
        
        for i, x_train in enumerate(self.X_train):
            dist = self.distance_func(x, x_train)
            distances.append((dist, i))
        
        distances.sort(key=lambda x: x[0])
        neighbors = distances[:self.k]
        
        return neighbors
    
    def _predict_single(self, x):
        """预测单个样本"""
        neighbors = self._get_neighbors(x)
        
        if self.weights == 'uniform':
            # 简单平均
            neighbor_values = [self.y_train[i] for _, i in neighbors]
            return np.mean(neighbor_values)
        
        elif self.weights == 'distance':
            # 距离加权平均
            weighted_sum = 0
            weight_sum = 0
            
            for dist, i in neighbors:
                weight = 1 / (dist + 1e-10)
                weighted_sum += weight * self.y_train[i]
                weight_sum += weight
            
            return weighted_sum / weight_sum if weight_sum > 0 else 0
    
    def predict(self, X):
        """预测多个样本"""
        predictions = []
        for x in X:
            pred = self._predict_single(x)
            predictions.append(pred)
        return np.array(predictions)

# 5. 生成测试数据
print("\n5. 生成测试数据")

# 分类数据
np.random.seed(42)
X_cls, y_cls = make_classification(n_samples=500, n_features=2, n_redundant=0, 
                                   n_informative=2, n_clusters_per_class=1, 
                                   n_classes=3, random_state=42)

# 回归数据
X_reg, y_reg = make_regression(n_samples=500, n_features=2, noise=0.1, random_state=42)

# 鸢尾花数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

print(f"分类数据形状: {X_cls.shape}")
print(f"回归数据形状: {X_reg.shape}")
print(f"鸢尾花数据形状: {X_iris.shape}")

# 6. 测试KNN分类器
print("\n6. 测试KNN分类器")

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 自实现的KNN
knn_custom = KNNClassifierFromScratch(k=5, distance_metric='euclidean', weights='uniform')
knn_custom.fit(X_train_scaled, y_train)
y_pred_custom = knn_custom.predict(X_test_scaled)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

# sklearn的KNN
knn_sklearn = KNeighborsClassifier(n_neighbors=5, metric='euclidean', weights='uniform')
knn_sklearn.fit(X_train_scaled, y_train)
y_pred_sklearn = knn_sklearn.predict(X_test_scaled)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"自实现KNN分类准确率: {accuracy_custom:.4f}")
print(f"sklearn KNN分类准确率: {accuracy_sklearn:.4f}")

# 7. 测试KNN回归器
print("\n7. 测试KNN回归器")

X_reg_train, X_reg_test, y_reg_train, y_reg_test = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# 标准化回归数据
scaler_reg = StandardScaler()
X_reg_train_scaled = scaler_reg.fit_transform(X_reg_train)
X_reg_test_scaled = scaler_reg.transform(X_reg_test)

# 自实现的KNN回归
knn_reg_custom = KNNRegressorFromScratch(k=5, weights='distance')
knn_reg_custom.fit(X_reg_train_scaled, y_reg_train)
y_reg_pred_custom = knn_reg_custom.predict(X_reg_test_scaled)
mse_custom = mean_squared_error(y_reg_test, y_reg_pred_custom)

# sklearn的KNN回归
knn_reg_sklearn = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg_sklearn.fit(X_reg_train_scaled, y_reg_train)
y_reg_pred_sklearn = knn_reg_sklearn.predict(X_reg_test_scaled)
mse_sklearn = mean_squared_error(y_reg_test, y_reg_pred_sklearn)

print(f"自实现KNN回归MSE: {mse_custom:.4f}")
print(f"sklearn KNN回归MSE: {mse_sklearn:.4f}")

# 8. K值选择实验
print("\n8. K值选择实验")

def find_optimal_k(X_train, y_train, X_test, y_test, max_k=20, task='classification'):
    """寻找最优K值"""
    k_values = range(1, max_k + 1)
    scores = []
    
    for k in k_values:
        if task == 'classification':
            knn = KNNClassifierFromScratch(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            score = accuracy_score(y_test, y_pred)
        else:
            knn = KNNRegressorFromScratch(k=k)
            knn.fit(X_train, y_train)
            y_pred = knn.predict(X_test)
            score = -mean_squared_error(y_test, y_pred)  # 负MSE，越大越好
        
        scores.append(score)
    
    optimal_k = k_values[np.argmax(scores)]
    return k_values, scores, optimal_k

# 分类任务的K值选择
k_values_cls, scores_cls, optimal_k_cls = find_optimal_k(
    X_train_scaled, y_train, X_test_scaled, y_test, task='classification')

print(f"分类任务最优K值: {optimal_k_cls}")

# 回归任务的K值选择
k_values_reg, scores_reg, optimal_k_reg = find_optimal_k(
    X_reg_train_scaled, y_reg_train, X_reg_test_scaled, y_reg_test, task='regression')

print(f"回归任务最优K值: {optimal_k_reg}")

# 9. 不同距离度量比较
print("\n9. 不同距离度量比较")

distance_metrics = ['euclidean', 'manhattan', 'cosine', 'chebyshev']
distance_results = {}

for metric in distance_metrics:
    try:
        knn = KNNClassifierFromScratch(k=5, distance_metric=metric)
        knn.fit(X_train_scaled, y_train)
        y_pred = knn.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        distance_results[metric] = accuracy
        print(f"{metric}距离准确率: {accuracy:.4f}")
    except Exception as e:
        print(f"错误 {metric}: {e}")
        distance_results[metric] = 0.0

# 10. 权重策略比较
print("\n10. 权重策略比较")

weight_strategies = ['uniform', 'distance']
weight_results = {}

for weights in weight_strategies:
    knn = KNNClassifierFromScratch(k=5, weights=weights)
    knn.fit(X_train_scaled, y_train)
    y_pred = knn.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    weight_results[weights] = accuracy
    print(f"{weights}权重准确率: {accuracy:.4f}")

# 11. 数据标准化的影响
print("\n11. 数据标准化的影响")

# 未标准化数据
knn_no_scale = KNNClassifierFromScratch(k=5)
knn_no_scale.fit(X_train, y_train)
y_pred_no_scale = knn_no_scale.predict(X_test)
accuracy_no_scale = accuracy_score(y_test, y_pred_no_scale)

# 标准化数据
knn_scaled = KNNClassifierFromScratch(k=5)
knn_scaled.fit(X_train_scaled, y_train)
y_pred_scaled = knn_scaled.predict(X_test_scaled)
accuracy_scaled = accuracy_score(y_test, y_pred_scaled)

print(f"未标准化数据准确率: {accuracy_no_scale:.4f}")
print(f"标准化数据准确率: {accuracy_scaled:.4f}")

# 12. 决策边界可视化
print("\n12. 决策边界可视化")

def plot_decision_boundary(X, y, model, title, ax, h=0.02):
    """绘制决策边界"""
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    
    mesh_points = np.c_[xx.ravel(), yy.ravel()]
    Z = model.predict(mesh_points)
    Z = Z.reshape(xx.shape)
    
    ax.contourf(xx, yy, Z, alpha=0.8, cmap=plt.cm.RdYlBu)
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
    ax.set_title(title)
    return scatter

# 13. 维数灾难演示
print("\n13. 维数灾难演示")

def curse_of_dimensionality_experiment():
    """演示维数灾难对KNN的影响"""
    dimensions = [2, 5, 10, 20, 50]
    accuracies = []
    
    for dim in dimensions:
        # 生成高维数据
        X_high, y_high = make_classification(n_samples=200, n_features=dim, 
                                           n_informative=min(dim, 10), 
                                           n_redundant=0, random_state=42)
        
        X_tr, X_te, y_tr, y_te = train_test_split(X_high, y_high, test_size=0.3, random_state=42)
        
        # 标准化
        scaler_high = StandardScaler()
        X_tr_scaled = scaler_high.fit_transform(X_tr)
        X_te_scaled = scaler_high.transform(X_te)
        
        # KNN分类
        knn_high = KNNClassifierFromScratch(k=5)
        knn_high.fit(X_tr_scaled, y_tr)
        y_pred_high = knn_high.predict(X_te_scaled)
        accuracy = accuracy_score(y_te, y_pred_high)
        
        accuracies.append(accuracy)
        print(f"维度 {dim}: 准确率 {accuracy:.4f}")
    
    return dimensions, accuracies

dimensions, high_dim_accuracies = curse_of_dimensionality_experiment()

# 14. 可视化分析
print("\n14. 可视化分析")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 14.1 K值选择曲线
axes[0, 0].plot(k_values_cls, scores_cls, 'bo-', label='分类')
axes[0, 0].axvline(x=optimal_k_cls, color='red', linestyle='--', label=f'最优K={optimal_k_cls}')
axes[0, 0].set_title('K值选择（分类任务）')
axes[0, 0].set_xlabel('K值')
axes[0, 0].set_ylabel('准确率')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 14.2 距离度量比较
metrics = list(distance_results.keys())
accuracies = list(distance_results.values())
axes[0, 1].bar(metrics, accuracies, alpha=0.7)
axes[0, 1].set_title('不同距离度量性能对比')
axes[0, 1].set_ylabel('准确率')
axes[0, 1].tick_params(axis='x', rotation=45)

# 14.3 权重策略比较
weights = list(weight_results.keys())
weight_accs = list(weight_results.values())
axes[0, 2].bar(weights, weight_accs, alpha=0.7, color=['orange', 'green'])
axes[0, 2].set_title('权重策略对比')
axes[0, 2].set_ylabel('准确率')

# 14.4 标准化影响
standardization_results = ['未标准化', '标准化']
std_accuracies = [accuracy_no_scale, accuracy_scaled]
axes[1, 0].bar(standardization_results, std_accuracies, alpha=0.7, color=['red', 'blue'])
axes[1, 0].set_title('数据标准化的影响')
axes[1, 0].set_ylabel('准确率')

# 14.5 维数灾难
axes[1, 1].plot(dimensions, high_dim_accuracies, 'ro-')
axes[1, 1].set_title('维数灾难对KNN的影响')
axes[1, 1].set_xlabel('特征维度')
axes[1, 1].set_ylabel('准确率')
axes[1, 1].grid(True, alpha=0.3)

# 14.6 回归结果可视化
if X_reg.shape[1] == 2:
    scatter = axes[1, 2].scatter(X_reg_test[:, 0], X_reg_test[:, 1], 
                                c=y_reg_pred_custom, cmap='viridis', alpha=0.7)
    axes[1, 2].set_title('KNN回归预测结果')
    axes[1, 2].set_xlabel('特征1')
    axes[1, 2].set_ylabel('特征2')
    plt.colorbar(scatter, ax=axes[1, 2])

# 14.7 决策边界（K=3）
knn_k3 = KNNClassifierFromScratch(k=3)
knn_k3.fit(X_train_scaled, y_train)
plot_decision_boundary(X_train_scaled, y_train, knn_k3, 'KNN决策边界 (K=3)', axes[2, 0])

# 14.8 决策边界（K=15）
knn_k15 = KNNClassifierFromScratch(k=15)
knn_k15.fit(X_train_scaled, y_train)
plot_decision_boundary(X_train_scaled, y_train, knn_k15, 'KNN决策边界 (K=15)', axes[2, 1])

# 14.9 预测概率分布
probs = knn_custom.predict_proba(X_test_scaled)
max_probs = np.max(probs, axis=1)
axes[2, 2].hist(max_probs, bins=15, alpha=0.7, edgecolor='black')
axes[2, 2].set_title('预测置信度分布')
axes[2, 2].set_xlabel('最大预测概率')
axes[2, 2].set_ylabel('频数')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/knn_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 15. KNN的优化技术
print("\n15. KNN的优化技术")

class OptimizedKNN:
    """优化的KNN实现（使用KD树思想的简化版本）"""
    
    def __init__(self, k=3):
        self.k = k
        self.X_train = None
        self.y_train = None
    
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
    
    def _fast_distance_batch(self, x, X_batch):
        """批量计算距离"""
        return np.sqrt(np.sum((X_batch - x) ** 2, axis=1))
    
    def predict(self, X):
        """优化的预测方法"""
        predictions = []
        
        for x in X:
            # 批量计算所有距离
            distances = self._fast_distance_batch(x, self.X_train)
            
            # 获取K个最近邻的索引
            k_nearest_indices = np.argpartition(distances, self.k)[:self.k]
            
            # 投票决定类别
            k_nearest_labels = self.y_train[k_nearest_indices]
            prediction = Counter(k_nearest_labels).most_common(1)[0][0]
            predictions.append(prediction)
        
        return np.array(predictions)

# 测试优化版本
optimized_knn = OptimizedKNN(k=5)
optimized_knn.fit(X_train_scaled, y_train)

import time

# 性能对比
start_time = time.time()
y_pred_original = knn_custom.predict(X_test_scaled)
original_time = time.time() - start_time

start_time = time.time()
y_pred_optimized = optimized_knn.predict(X_test_scaled)
optimized_time = time.time() - start_time

print(f"原始KNN预测时间: {original_time:.4f}秒")
print(f"优化KNN预测时间: {optimized_time:.4f}秒")
print(f"加速比: {original_time/optimized_time:.2f}倍")

# 验证结果一致性
print(f"结果一致性: {np.array_equal(y_pred_original, y_pred_optimized)}")

print("\n=== K近邻算法总结 ===")
print("✅ 理解KNN的基本原理和特点")
print("✅ 实现多种距离度量方法")
print("✅ 掌握K值选择和参数优化")
print("✅ 分析权重策略的影响")
print("✅ 理解维数灾难问题")
print("✅ 实现分类和回归任务")
print("✅ 学习KNN的优化技术")

print("\n算法特点:")
print("优点: 简单直观、无需训练、适应性强、可解释")
print("缺点: 计算复杂度高、存储需求大、对噪声敏感、维数灾难")

print("\n关键参数:")
print("1. K值：影响决策边界的平滑程度")
print("2. 距离度量：不同距离适合不同数据类型")
print("3. 权重策略：uniform vs distance")
print("4. 数据预处理：标准化的重要性")

print("\n适用场景:")
print("1. 小到中等规模数据集")
print("2. 非线性决策边界")
print("3. 局部模式识别")
print("4. 推荐系统")

print("\n=== 练习任务 ===")
print("1. 实现KD树加速KNN搜索")
print("2. 尝试局部敏感哈希(LSH)")
print("3. 实现加权KNN变体")
print("4. 研究KNN在异常检测中的应用")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现近似最近邻搜索")
print("2. 研究自适应K值选择")
print("3. 实现分布式KNN算法")
print("4. 结合特征选择优化KNN")
print("5. 实现KNN的在线学习版本")