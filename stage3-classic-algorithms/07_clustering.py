"""
聚类算法从零实现
学习目标：理解K-means、层次聚类、DBSCAN等无监督学习算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_circles, make_moons
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, adjusted_rand_score
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("=== 聚类算法从零实现 ===\n")

# 1. K-means算法实现
print("1. K-means算法实现")

class KMeansFromScratch:
    def __init__(self, k=3, max_iters=100, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state
        
    def fit(self, X):
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        
        # 随机初始化质心
        self.centroids = X[np.random.choice(n_samples, self.k, replace=False)]
        
        # 记录迭代过程
        self.centroids_history = [self.centroids.copy()]
        self.inertia_history = []
        
        for i in range(self.max_iters):
            # 分配样本到最近的质心
            distances = self._calculate_distances(X)
            self.labels_ = np.argmin(distances, axis=1)
            
            # 计算惯性
            inertia = self._calculate_inertia(X)
            self.inertia_history.append(inertia)
            
            # 更新质心
            new_centroids = np.zeros((self.k, n_features))
            for j in range(self.k):
                if np.sum(self.labels_ == j) > 0:
                    new_centroids[j] = X[self.labels_ == j].mean(axis=0)
                else:
                    new_centroids[j] = self.centroids[j]
            
            # 检查收敛
            if np.allclose(self.centroids, new_centroids):
                print(f"K-means在第{i+1}次迭代后收敛")
                break
                
            self.centroids = new_centroids
            self.centroids_history.append(self.centroids.copy())
        
        self.inertia_ = self.inertia_history[-1]
        return self
    
    def _calculate_distances(self, X):
        distances = np.zeros((X.shape[0], self.k))
        for i, centroid in enumerate(self.centroids):
            distances[:, i] = np.linalg.norm(X - centroid, axis=1)
        return distances
    
    def _calculate_inertia(self, X):
        inertia = 0
        for i in range(self.k):
            cluster_points = X[self.labels_ == i]
            if len(cluster_points) > 0:
                inertia += np.sum((cluster_points - self.centroids[i]) ** 2)
        return inertia
    
    def predict(self, X):
        distances = self._calculate_distances(X)
        return np.argmin(distances, axis=1)

# 2. 层次聚类实现
print("\n2. 层次聚类实现")

class AgglomerativeClusteringFromScratch:
    def __init__(self, n_clusters=3, linkage='single'):
        self.n_clusters = n_clusters
        self.linkage = linkage
        
    def fit(self, X):
        n_samples = X.shape[0]
        
        # 初始化每个点为一个簇
        clusters = [[i] for i in range(n_samples)]
        
        # 计算距离矩阵
        self.distance_matrix = self._calculate_distance_matrix(X)
        
        # 记录合并过程
        self.merge_history = []
        
        while len(clusters) > self.n_clusters:
            # 找到最近的两个簇
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._cluster_distance(clusters[i], clusters[j])
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # 合并簇
            self.merge_history.append((merge_i, merge_j, min_dist))
            new_cluster = clusters[merge_i] + clusters[merge_j]
            
            # 移除旧簇，添加新簇
            clusters = [clusters[i] for i in range(len(clusters)) 
                       if i != merge_i and i != merge_j] + [new_cluster]
        
        # 生成标签
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_id in cluster:
                self.labels_[point_id] = cluster_id
        
        return self
    
    def _calculate_distance_matrix(self, X):
        n_samples = X.shape[0]
        dist_matrix = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                dist = np.linalg.norm(X[i] - X[j])
                dist_matrix[i, j] = dist_matrix[j, i] = dist
        return dist_matrix
    
    def _cluster_distance(self, cluster1, cluster2):
        distances = []
        for i in cluster1:
            for j in cluster2:
                distances.append(self.distance_matrix[i, j])
        
        if self.linkage == 'single':
            return min(distances)
        elif self.linkage == 'complete':
            return max(distances)
        elif self.linkage == 'average':
            return np.mean(distances)

# 3. DBSCAN算法实现
print("\n3. DBSCAN算法实现")

class DBSCANFromScratch:
    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        
    def fit(self, X):
        n_samples = X.shape[0]
        self.labels_ = np.full(n_samples, -1)  # -1表示噪声点
        
        cluster_id = 0
        
        for i in range(n_samples):
            # 跳过已处理的点
            if self.labels_[i] != -1:
                continue
            
            # 找到邻居
            neighbors = self._find_neighbors(X, i)
            
            # 如果邻居数量不足，标记为噪声
            if len(neighbors) < self.min_samples:
                self.labels_[i] = -1
            else:
                # 开始新簇
                self._expand_cluster(X, i, neighbors, cluster_id)
                cluster_id += 1
        
        return self
    
    def _find_neighbors(self, X, point_idx):
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0]
    
    def _expand_cluster(self, X, point_idx, neighbors, cluster_id):
        self.labels_[point_idx] = cluster_id
        i = 0
        
        while i < len(neighbors):
            neighbor = neighbors[i]
            
            # 如果是噪声点，改为边界点
            if self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_id
            
            # 如果未处理，处理该点
            elif self.labels_[neighbor] == -1:
                self.labels_[neighbor] = cluster_id
                
                # 找到新邻居
                new_neighbors = self._find_neighbors(X, neighbor)
                
                # 如果是核心点，扩展邻居列表
                if len(new_neighbors) >= self.min_samples:
                    neighbors = np.concatenate([neighbors, new_neighbors])
                    neighbors = np.unique(neighbors)  # 去重
            
            i += 1

# 4. 生成测试数据
print("\n4. 生成测试数据")

# 生成不同类型的数据集
np.random.seed(42)

# 高斯分布的簇
blobs_X, blobs_y = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)

# 同心圆数据
circles_X, circles_y = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

# 月牙形数据
moons_X, moons_y = make_moons(n_samples=300, noise=0.1, random_state=42)

# 各向异性数据
random_state = 170
aniso_X, aniso_y = make_blobs(n_samples=300, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
aniso_X = np.dot(aniso_X, transformation)

print("生成的数据集:")
print(f"高斯簇数据: {blobs_X.shape}")
print(f"同心圆数据: {circles_X.shape}")
print(f"月牙形数据: {moons_X.shape}")
print(f"各向异性数据: {aniso_X.shape}")

# 5. 测试K-means算法
print("\n5. 测试K-means算法")

# 在高斯簇数据上测试
kmeans_custom = KMeansFromScratch(k=4, max_iters=100)
kmeans_custom.fit(blobs_X)

kmeans_sklearn = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans_sklearn.fit(blobs_X)

print(f"自实现K-means惯性: {kmeans_custom.inertia_:.2f}")
print(f"sklearn K-means惯性: {kmeans_sklearn.inertia_:.2f}")

# 计算轮廓系数
silhouette_custom = silhouette_score(blobs_X, kmeans_custom.labels_)
silhouette_sklearn = silhouette_score(blobs_X, kmeans_sklearn.labels_)

print(f"自实现K-means轮廓系数: {silhouette_custom:.4f}")
print(f"sklearn K-means轮廓系数: {silhouette_sklearn:.4f}")

# 6. 肘部法则确定最优K值
print("\n6. 肘部法则确定最优K值")

def elbow_method(X, max_k=10):
    inertias = []
    K_range = range(1, max_k + 1)
    
    for k in K_range:
        kmeans = KMeansFromScratch(k=k, max_iters=100)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)
    
    return K_range, inertias

K_range, inertias = elbow_method(blobs_X, max_k=10)
print(f"不同K值的惯性: {list(zip(K_range, inertias))}")

# 7. 轮廓系数分析
print("\n7. 轮廓系数分析")

def silhouette_analysis(X, max_k=10):
    silhouette_scores = []
    K_range = range(2, max_k + 1)
    
    for k in K_range:
        kmeans = KMeansFromScratch(k=k, max_iters=100)
        kmeans.fit(X)
        score = silhouette_score(X, kmeans.labels_)
        silhouette_scores.append(score)
    
    return K_range, silhouette_scores

K_range_sil, silhouette_scores = silhouette_analysis(blobs_X, max_k=8)
optimal_k = K_range_sil[np.argmax(silhouette_scores)]
print(f"最优K值 (基于轮廓系数): {optimal_k}")

# 8. 比较不同聚类算法
print("\n8. 比较不同聚类算法")

datasets = {
    'Gaussian Blobs': blobs_X,
    'Circles': circles_X,
    'Moons': moons_X,
    'Anisotropic': aniso_X
}

algorithms = {
    'K-Means': lambda X: KMeans(n_clusters=2, random_state=42, n_init=10).fit(X),
    'Agglomerative': lambda X: AgglomerativeClustering(n_clusters=2).fit(X),
    'DBSCAN': lambda X: DBSCAN(eps=0.3, min_samples=10).fit(X)
}

results = {}
for dataset_name, X in datasets.items():
    # 标准化数据
    X_scaled = StandardScaler().fit_transform(X)
    results[dataset_name] = {}
    
    for algo_name, algo_func in algorithms.items():
        try:
            clusterer = algo_func(X_scaled)
            labels = clusterer.labels_
            
            # 计算轮廓系数（忽略DBSCAN的噪声点）
            if len(set(labels)) > 1 and not all(label == -1 for label in labels):
                if algo_name == 'DBSCAN':
                    # 对于DBSCAN，只计算非噪声点的轮廓系数
                    mask = labels != -1
                    if np.sum(mask) > 1 and len(set(labels[mask])) > 1:
                        score = silhouette_score(X_scaled[mask], labels[mask])
                    else:
                        score = -1
                else:
                    score = silhouette_score(X_scaled, labels)
            else:
                score = -1
            
            results[dataset_name][algo_name] = {
                'labels': labels,
                'silhouette_score': score,
                'n_clusters': len(set(labels)) - (1 if -1 in labels else 0)
            }
            
        except Exception as e:
            print(f"错误 {algo_name} on {dataset_name}: {e}")
            results[dataset_name][algo_name] = {
                'labels': np.zeros(len(X_scaled)),
                'silhouette_score': -1,
                'n_clusters': 1
            }

# 打印结果
for dataset_name in datasets.keys():
    print(f"\n{dataset_name} 数据集结果:")
    for algo_name in algorithms.keys():
        result = results[dataset_name][algo_name]
        print(f"  {algo_name}: 簇数={result['n_clusters']}, "
              f"轮廓系数={result['silhouette_score']:.4f}")

# 9. 可视化结果
print("\n9. 可视化结果")

fig, axes = plt.subplots(4, 4, figsize=(20, 20))

dataset_names = list(datasets.keys())
algorithm_names = list(algorithms.keys())

for i, dataset_name in enumerate(dataset_names):
    X = datasets[dataset_name]
    X_scaled = StandardScaler().fit_transform(X)
    
    # 原始数据
    axes[i, 0].scatter(X[:, 0], X[:, 1], c='black', alpha=0.6, s=20)
    axes[i, 0].set_title(f'{dataset_name} - 原始数据')
    
    # 不同算法的结果
    for j, algo_name in enumerate(algorithm_names):
        labels = results[dataset_name][algo_name]['labels']
        
        # 处理DBSCAN的噪声点
        if algo_name == 'DBSCAN':
            # 噪声点用灰色表示
            noise_mask = labels == -1
            if np.any(~noise_mask):
                axes[i, j+1].scatter(X[~noise_mask, 0], X[~noise_mask, 1], 
                                   c=labels[~noise_mask], cmap='viridis', 
                                   alpha=0.6, s=20)
            if np.any(noise_mask):
                axes[i, j+1].scatter(X[noise_mask, 0], X[noise_mask, 1], 
                                   c='gray', alpha=0.3, s=20, marker='x')
        else:
            axes[i, j+1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', 
                               alpha=0.6, s=20)
        
        score = results[dataset_name][algo_name]['silhouette_score']
        axes[i, j+1].set_title(f'{algo_name}\n轮廓系数: {score:.3f}')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/clustering_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 10. K-means收敛过程可视化
print("\n10. K-means收敛过程可视化")

# 重新训练K-means以获取收敛过程
kmeans_viz = KMeansFromScratch(k=4, max_iters=20)
kmeans_viz.fit(blobs_X)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 显示前6次迭代
for i in range(min(6, len(kmeans_viz.centroids_history))):
    row, col = i // 3, i % 3
    
    # 绘制数据点
    axes[row, col].scatter(blobs_X[:, 0], blobs_X[:, 1], 
                          c='lightblue', alpha=0.6, s=20)
    
    # 绘制质心
    centroids = kmeans_viz.centroids_history[i]
    axes[row, col].scatter(centroids[:, 0], centroids[:, 1], 
                          c='red', marker='x', s=200, linewidths=3)
    
    axes[row, col].set_title(f'迭代 {i+1}')
    axes[row, col].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/kmeans_convergence.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 11. 肘部法则和轮廓系数可视化
print("\n11. 优化指标可视化")

fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# 肘部法则
axes[0].plot(K_range, inertias, 'bo-')
axes[0].set_title('肘部法则确定最优K值')
axes[0].set_xlabel('K值')
axes[0].set_ylabel('惯性 (Inertia)')
axes[0].grid(True, alpha=0.3)

# 轮廓系数
axes[1].plot(K_range_sil, silhouette_scores, 'ro-')
axes[1].axvline(x=optimal_k, color='green', linestyle='--', 
                label=f'最优K={optimal_k}')
axes[1].set_title('轮廓系数确定最优K值')
axes[1].set_xlabel('K值')
axes[1].set_ylabel('轮廓系数')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/clustering_metrics.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 聚类算法总结 ===")
print("✅ K-means算法原理和实现")
print("✅ 层次聚类算法实现")
print("✅ DBSCAN密度聚类算法")
print("✅ 肘部法则和轮廓系数评估")
print("✅ 不同算法在各种数据上的表现")
print("✅ 聚类结果的可视化分析")

print("\n=== 练习任务 ===")
print("1. 实现K-means++初始化")
print("2. 尝试其他聚类算法(谱聚类、均值漂移)")
print("3. 实现聚类有效性指标")
print("4. 研究高维数据的聚类")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现在线聚类算法")
print("2. 研究模糊聚类方法")
print("3. 实现大规模数据聚类")
print("4. 研究聚类集成方法")
print("5. 实现层次聚类的可视化树状图")