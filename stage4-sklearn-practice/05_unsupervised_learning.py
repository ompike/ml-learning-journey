"""
无监督学习应用
学习目标：掌握聚类、降维、异常检测等无监督学习方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs, make_circles, load_digits, load_wine
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering, SpectralClustering
from sklearn.decomposition import PCA, FastICA, FactorAnalysis
from sklearn.manifold import TSNE, MDS, LocallyLinearEmbedding
from sklearn.preprocessing import StandardScaler
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
from sklearn.metrics import silhouette_score, adjusted_rand_score, calinski_harabasz_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=== 无监督学习应用 ===\n")

# 1. 数据准备
print("1. 数据准备")

# 生成聚类数据
np.random.seed(42)
X_blobs, y_true_blobs = make_blobs(n_samples=300, centers=4, cluster_std=0.8, random_state=42)
X_circles, y_true_circles = make_circles(n_samples=300, noise=0.1, factor=0.3, random_state=42)

# 加载真实数据集
digits = load_digits()
X_digits, y_digits = digits.data, digits.target

wine = load_wine()
X_wine, y_wine = wine.data, wine.target

print(f"Blobs数据形状: {X_blobs.shape}")
print(f"Circles数据形状: {X_circles.shape}")
print(f"Digits数据形状: {X_digits.shape}")
print(f"Wine数据形状: {X_wine.shape}")

# 2. 聚类算法比较
print("\n2. 聚类算法比较")

# 标准化数据
scaler = StandardScaler()
X_blobs_scaled = scaler.fit_transform(X_blobs)
X_circles_scaled = scaler.fit_transform(X_circles)

# 定义聚类算法
clustering_algorithms = {
    'K-Means': KMeans(n_clusters=4, random_state=42),
    'DBSCAN': DBSCAN(eps=0.3, min_samples=10),
    'Agglomerative': AgglomerativeClustering(n_clusters=4),
    'Spectral': SpectralClustering(n_clusters=4, random_state=42),
    'Gaussian Mixture': GaussianMixture(n_components=4, random_state=42)
}

# 聚类结果
clustering_results = {}

for name, algorithm in clustering_algorithms.items():
    print(f"\n测试 {name}...")
    
    # 在blobs数据上聚类
    if name == 'Gaussian Mixture':
        labels_blobs = algorithm.fit_predict(X_blobs_scaled)
    else:
        labels_blobs = algorithm.fit_predict(X_blobs_scaled)
    
    # 计算评估指标
    if len(set(labels_blobs)) > 1 and not all(label == -1 for label in labels_blobs):
        if name == 'DBSCAN':
            # 对于DBSCAN，只计算非噪声点
            mask = labels_blobs != -1
            if np.sum(mask) > 1 and len(set(labels_blobs[mask])) > 1:
                silhouette = silhouette_score(X_blobs_scaled[mask], labels_blobs[mask])
                ari = adjusted_rand_score(y_true_blobs[mask], labels_blobs[mask])
            else:
                silhouette, ari = -1, -1
        else:
            silhouette = silhouette_score(X_blobs_scaled, labels_blobs)
            ari = adjusted_rand_score(y_true_blobs, labels_blobs)
    else:
        silhouette, ari = -1, -1
    
    clustering_results[name] = {
        'labels': labels_blobs,
        'silhouette_score': silhouette,
        'ari_score': ari,
        'n_clusters': len(set(labels_blobs)) - (1 if -1 in labels_blobs else 0)
    }
    
    print(f"  簇数: {clustering_results[name]['n_clusters']}")
    print(f"  轮廓系数: {silhouette:.4f}")
    print(f"  ARI: {ari:.4f}")

# 3. 降维算法比较
print("\n3. 降维算法比较")

# 标准化digits数据
scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits)

# 定义降维算法
dimensionality_reduction = {
    'PCA': PCA(n_components=2, random_state=42),
    't-SNE': TSNE(n_components=2, random_state=42, perplexity=30),
    'ICA': FastICA(n_components=2, random_state=42),
    'MDS': MDS(n_components=2, random_state=42),
    'LLE': LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
}

# 降维结果
reduction_results = {}

for name, algorithm in dimensionality_reduction.items():
    print(f"执行 {name} 降维...")
    
    try:
        if name == 't-SNE':
            # t-SNE对大数据集较慢，使用子集
            subset_idx = np.random.choice(len(X_digits_scaled), 1000, replace=False)
            X_reduced = algorithm.fit_transform(X_digits_scaled[subset_idx])
            y_subset = y_digits[subset_idx]
        else:
            X_reduced = algorithm.fit_transform(X_digits_scaled)
            y_subset = y_digits
        
        reduction_results[name] = {
            'X_reduced': X_reduced,
            'y': y_subset,
            'algorithm': algorithm
        }
        
        print(f"  {name} 降维完成，输出形状: {X_reduced.shape}")
        
    except Exception as e:
        print(f"  {name} 降维失败: {e}")
        reduction_results[name] = None

# 4. PCA详细分析
print("\n4. PCA详细分析")

# 在wine数据上进行PCA
scaler_wine = StandardScaler()
X_wine_scaled = scaler_wine.fit_transform(X_wine)

pca_wine = PCA()
X_wine_pca = pca_wine.fit_transform(X_wine_scaled)

# 解释方差比
explained_variance_ratio = pca_wine.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print(f"前5个主成分解释方差比: {explained_variance_ratio[:5]}")
print(f"前5个主成分累计解释方差比: {cumulative_variance_ratio[:5]}")

# 确定保留多少主成分（保留95%方差）
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
print(f"保留95%方差需要 {n_components_95} 个主成分")

# 主成分分析
components_df = pd.DataFrame(
    pca_wine.components_[:3].T,  # 前3个主成分
    columns=['PC1', 'PC2', 'PC3'],
    index=wine.feature_names
)

print("\n前3个主成分的特征权重:")
print(components_df.head(10))

# 5. 聚类评估和优化
print("\n5. 聚类评估和优化")

def find_optimal_clusters(X, max_clusters=10):
    """使用肘部法则和轮廓系数寻找最优簇数"""
    inertias = []
    silhouette_scores = []
    calinski_scores = []
    
    K_range = range(2, max_clusters + 1)
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))
        calinski_scores.append(calinski_harabasz_score(X, labels))
    
    return K_range, inertias, silhouette_scores, calinski_scores

# 在wine数据上寻找最优簇数
K_range, inertias, silhouette_scores, calinski_scores = find_optimal_clusters(X_wine_scaled)

optimal_k_silhouette = K_range[np.argmax(silhouette_scores)]
optimal_k_calinski = K_range[np.argmax(calinski_scores)]

print(f"基于轮廓系数的最优簇数: {optimal_k_silhouette}")
print(f"基于Calinski-Harabasz指数的最优簇数: {optimal_k_calinski}")

# 6. 高斯混合模型
print("\n6. 高斯混合模型")

# 使用不同成分数的GMM
n_components_range = range(1, 11)
aic_scores = []
bic_scores = []

for n_components in n_components_range:
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    gmm.fit(X_wine_scaled)
    aic_scores.append(gmm.aic(X_wine_scaled))
    bic_scores.append(gmm.bic(X_wine_scaled))

optimal_n_aic = n_components_range[np.argmin(aic_scores)]
optimal_n_bic = n_components_range[np.argmin(bic_scores)]

print(f"基于AIC的最优成分数: {optimal_n_aic}")
print(f"基于BIC的最优成分数: {optimal_n_bic}")

# 使用最优参数训练GMM
best_gmm = GaussianMixture(n_components=optimal_n_bic, random_state=42)
gmm_labels = best_gmm.fit_predict(X_wine_scaled)
gmm_probs = best_gmm.predict_proba(X_wine_scaled)

print(f"GMM聚类结果 - 簇数: {len(set(gmm_labels))}")
print(f"GMM轮廓系数: {silhouette_score(X_wine_scaled, gmm_labels):.4f}")

# 7. 异常检测
print("\n7. 异常检测")

# 生成带异常值的数据
np.random.seed(42)
X_normal = np.random.multivariate_normal([0, 0], [[1, 0], [0, 1]], 200)
X_outliers = np.random.multivariate_normal([3, 3], [[0.5, 0], [0, 0.5]], 20)
X_anomaly = np.vstack([X_normal, X_outliers])
y_true_anomaly = np.hstack([np.zeros(200), np.ones(20)])  # 1表示异常

# 异常检测算法
anomaly_detectors = {
    'Isolation Forest': IsolationForest(contamination=0.1, random_state=42),
    'Local Outlier Factor': LocalOutlierFactor(contamination=0.1),
    'Elliptic Envelope': EllipticEnvelope(contamination=0.1, random_state=42)
}

anomaly_results = {}

for name, detector in anomaly_detectors.items():
    print(f"测试 {name}...")
    
    if name == 'Local Outlier Factor':
        # LOF的predict方法返回-1和1
        y_pred_anomaly = detector.fit_predict(X_anomaly)
        y_pred_anomaly = (y_pred_anomaly == -1).astype(int)  # 转换为0和1
    else:
        detector.fit(X_anomaly)
        y_pred_anomaly = detector.predict(X_anomaly)
        y_pred_anomaly = (y_pred_anomaly == -1).astype(int)  # 转换为0和1
    
    # 计算准确率
    accuracy = np.mean(y_pred_anomaly == y_true_anomaly)
    precision = np.sum((y_pred_anomaly == 1) & (y_true_anomaly == 1)) / np.sum(y_pred_anomaly == 1) if np.sum(y_pred_anomaly == 1) > 0 else 0
    recall = np.sum((y_pred_anomaly == 1) & (y_true_anomaly == 1)) / np.sum(y_true_anomaly == 1)
    
    anomaly_results[name] = {
        'predictions': y_pred_anomaly,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall
    }
    
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")

# 8. 无监督学习Pipeline
print("\n8. 无监督学习Pipeline")

# 创建完整的无监督学习pipeline
unsupervised_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=0.95)),  # 保留95%方差
    ('kmeans', KMeans(n_clusters=3, random_state=42))
])

# 在wine数据上应用pipeline
pipeline_labels = unsupervised_pipeline.fit_predict(X_wine)

# 评估pipeline结果
pipeline_silhouette = silhouette_score(X_wine, pipeline_labels)
pipeline_ari = adjusted_rand_score(y_wine, pipeline_labels)

print(f"Pipeline轮廓系数: {pipeline_silhouette:.4f}")
print(f"Pipeline ARI: {pipeline_ari:.4f}")

# 获取PCA后的数据
X_wine_pca_pipeline = unsupervised_pipeline.named_steps['pca'].transform(
    unsupervised_pipeline.named_steps['scaler'].transform(X_wine)
)
print(f"PCA降维后形状: {X_wine_pca_pipeline.shape}")

# 9. 可视化分析
print("\n9. 可视化分析")

fig, axes = plt.subplots(3, 4, figsize=(20, 15))

# 9.1 聚类算法比较 (Blobs数据)
for i, (name, result) in enumerate(clustering_results.items()):
    row, col = i // 4, i % 4
    if row < 3 and col < 4:
        labels = result['labels']
        
        if name == 'DBSCAN':
            # DBSCAN的噪声点用不同颜色
            mask = labels != -1
            if np.any(mask):
                axes[0, col].scatter(X_blobs_scaled[mask, 0], X_blobs_scaled[mask, 1], 
                                   c=labels[mask], cmap='viridis', alpha=0.7)
            if np.any(~mask):
                axes[0, col].scatter(X_blobs_scaled[~mask, 0], X_blobs_scaled[~mask, 1], 
                                   c='black', marker='x', alpha=0.7)
        else:
            axes[0, col].scatter(X_blobs_scaled[:, 0], X_blobs_scaled[:, 1], 
                               c=labels, cmap='viridis', alpha=0.7)
        
        axes[0, col].set_title(f'{name}\n轮廓系数: {result["silhouette_score"]:.3f}')

# 9.2 降维结果可视化
for i, (name, result) in enumerate(reduction_results.items()):
    if result is not None and i < 4:
        X_reduced = result['X_reduced']
        y_subset = result['y']
        
        scatter = axes[1, i].scatter(X_reduced[:, 0], X_reduced[:, 1], 
                                   c=y_subset, cmap='tab10', alpha=0.7)
        axes[1, i].set_title(f'{name} 降维结果')
        axes[1, i].set_xlabel('Component 1')
        axes[1, i].set_ylabel('Component 2')

# 9.3 PCA分析
axes[2, 0].plot(range(1, len(explained_variance_ratio) + 1), 
                cumulative_variance_ratio, 'bo-')
axes[2, 0].axhline(y=0.95, color='r', linestyle='--', label='95%方差')
axes[2, 0].set_xlabel('主成分数量')
axes[2, 0].set_ylabel('累计解释方差比')
axes[2, 0].set_title('PCA解释方差')
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3)

# 9.4 聚类评估指标
axes[2, 1].plot(K_range, silhouette_scores, 'bo-', label='轮廓系数')
axes[2, 1].set_xlabel('簇数')
axes[2, 1].set_ylabel('轮廓系数')
axes[2, 1].set_title('最优簇数选择')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 9.5 GMM模型选择
axes[2, 2].plot(n_components_range, aic_scores, 'ro-', label='AIC')
axes[2, 2].plot(n_components_range, bic_scores, 'bo-', label='BIC')
axes[2, 2].set_xlabel('成分数')
axes[2, 2].set_ylabel('信息准则')
axes[2, 2].set_title('GMM模型选择')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

# 9.6 异常检测结果
axes[2, 3].scatter(X_anomaly[y_true_anomaly == 0, 0], X_anomaly[y_true_anomaly == 0, 1], 
                  c='blue', alpha=0.7, label='正常点')
axes[2, 3].scatter(X_anomaly[y_true_anomaly == 1, 0], X_anomaly[y_true_anomaly == 1, 1], 
                  c='red', alpha=0.7, label='异常点')
axes[2, 3].set_title('异常检测数据')
axes[2, 3].legend()

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/unsupervised_learning_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 10. 特征重要性和可解释性
print("\n10. 特征重要性和可解释性")

# PCA特征重要性分析
feature_importance_df = pd.DataFrame(
    np.abs(pca_wine.components_[:3]).T,
    columns=['PC1', 'PC2', 'PC3'],
    index=wine.feature_names
)

print("各特征在前3个主成分中的重要性:")
print(feature_importance_df.round(4))

# 聚类中心分析（K-means）
best_kmeans = KMeans(n_clusters=optimal_k_silhouette, random_state=42)
kmeans_labels = best_kmeans.fit_predict(X_wine_scaled)
cluster_centers = best_kmeans.cluster_centers_

print(f"\nK-means聚类中心 (标准化后):")
centers_df = pd.DataFrame(cluster_centers.T, 
                         index=wine.feature_names,
                         columns=[f'Cluster_{i}' for i in range(optimal_k_silhouette)])
print(centers_df.round(4))

print("\n=== 无监督学习总结 ===")
print("✅ 掌握多种聚类算法的应用")
print("✅ 理解降维技术的原理和应用")
print("✅ 学会聚类评估和参数优化")
print("✅ 掌握高斯混合模型")
print("✅ 实现异常检测方法")
print("✅ 构建无监督学习Pipeline")
print("✅ 分析模型的可解释性")

print("\n关键技术:")
print("1. 聚类算法选择：根据数据特点选择合适算法")
print("2. 降维技术：PCA用于线性降维，t-SNE用于可视化")
print("3. 模型评估：轮廓系数、ARI、信息准则")
print("4. 异常检测：多种算法的组合使用")
print("5. 参数优化：网格搜索和评估指标")

print("\n实际应用:")
print("1. 客户细分和市场分析")
print("2. 图像压缩和特征提取")
print("3. 推荐系统中的协同过滤")
print("4. 网络安全中的异常检测")
print("5. 生物信息学中的基因分析")

print("\n=== 练习任务 ===")
print("1. 尝试更多聚类算法(Mean Shift, Birch)")
print("2. 实现自定义的异常检测方法")
print("3. 研究高维数据的降维技术")
print("4. 应用无监督学习到图像数据")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现深度聚类算法")
print("2. 研究流式数据的无监督学习")
print("3. 实现半监督学习算法")
print("4. 构建自适应异常检测系统")
print("5. 研究无监督学习的可解释性方法")