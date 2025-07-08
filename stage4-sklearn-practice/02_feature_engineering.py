"""
特征工程实践
学习目标：掌握特征选择、特征变换和特征构造的方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler, 
                                 PolynomialFeatures, LabelEncoder, OneHotEncoder)
from sklearn.feature_selection import (SelectKBest, f_classif, f_regression, 
                                     RFE, SelectFromModel, VarianceThreshold)
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=== 特征工程实践 ===\n")

# 1. 数据准备
print("1. 数据准备")

# 生成分类数据
X_cls, y_cls = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                                   n_redundant=5, n_clusters_per_class=1, random_state=42)

# 创建DataFrame
feature_names = [f'feature_{i}' for i in range(X_cls.shape[1])]
df_cls = pd.DataFrame(X_cls, columns=feature_names)
df_cls['target'] = y_cls

print(f"分类数据集形状: {df_cls.shape}")
print(f"特征数量: {X_cls.shape[1]}")
print(f"信息特征数量: 10")
print(f"冗余特征数量: 5")

# 2. 数据探索性分析
print("\n2. 数据探索性分析")

# 基本统计信息
print("基本统计信息:")
print(df_cls.describe())

# 相关性分析
correlation_matrix = df_cls.corr()
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if abs(correlation_matrix.iloc[i, j]) > 0.8:
            high_corr_pairs.append((correlation_matrix.columns[i], 
                                  correlation_matrix.columns[j], 
                                  correlation_matrix.iloc[i, j]))

print(f"\n高相关性特征对 (>0.8): {len(high_corr_pairs)}")
for pair in high_corr_pairs[:5]:  # 只显示前5个
    print(f"  {pair[0]} - {pair[1]}: {pair[2]:.3f}")

# 3. 特征缩放
print("\n3. 特征缩放")

X_train, X_test, y_train, y_test = train_test_split(X_cls, y_cls, test_size=0.2, random_state=42)

scalers = {
    'StandardScaler': StandardScaler(),
    'MinMaxScaler': MinMaxScaler(),
    'RobustScaler': RobustScaler()
}

scaled_data = {}
for name, scaler in scalers.items():
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    scaled_data[name] = (X_train_scaled, X_test_scaled)
    
    print(f"{name}:")
    print(f"  训练集均值: {X_train_scaled.mean():.4f}")
    print(f"  训练集标准差: {X_train_scaled.std():.4f}")
    print(f"  训练集最小值: {X_train_scaled.min():.4f}")
    print(f"  训练集最大值: {X_train_scaled.max():.4f}")

# 4. 特征选择
print("\n4. 特征选择")

# 4.1 方差过滤
print("4.1 方差过滤")
variance_threshold = VarianceThreshold(threshold=0.1)
X_train_var = variance_threshold.fit_transform(X_train)
selected_features_var = variance_threshold.get_support()

print(f"原始特征数: {X_train.shape[1]}")
print(f"方差过滤后特征数: {X_train_var.shape[1]}")
print(f"被移除的特征数: {np.sum(~selected_features_var)}")

# 4.2 单变量特征选择
print("\n4.2 单变量特征选择")
k_best = SelectKBest(score_func=f_classif, k=10)
X_train_kbest = k_best.fit_transform(X_train, y_train)
X_test_kbest = k_best.transform(X_test)

scores = k_best.scores_
selected_features_kbest = k_best.get_support()

print(f"选择的特征数: {X_train_kbest.shape[1]}")
print("特征得分排名:")
feature_scores = list(zip(feature_names, scores, selected_features_kbest))
feature_scores.sort(key=lambda x: x[1], reverse=True)
for i, (name, score, selected) in enumerate(feature_scores[:10]):
    print(f"  {i+1}. {name}: {score:.2f} {'✓' if selected else '✗'}")

# 4.3 递归特征消除
print("\n4.3 递归特征消除")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rfe = RFE(estimator=rf_classifier, n_features_to_select=10, step=1)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)

selected_features_rfe = rfe.get_support()
feature_rankings = rfe.ranking_

print(f"RFE选择的特征数: {X_train_rfe.shape[1]}")
print("特征排名:")
feature_rankings_info = list(zip(feature_names, feature_rankings, selected_features_rfe))
feature_rankings_info.sort(key=lambda x: x[1])
for i, (name, rank, selected) in enumerate(feature_rankings_info[:10]):
    print(f"  {i+1}. {name}: 排名{rank} {'✓' if selected else '✗'}")

# 4.4 基于模型的特征选择
print("\n4.4 基于模型的特征选择")
rf_selector = RandomForestClassifier(n_estimators=100, random_state=42)
model_selector = SelectFromModel(rf_selector, threshold='mean')
X_train_model = model_selector.fit_transform(X_train, y_train)
X_test_model = model_selector.transform(X_test)

selected_features_model = model_selector.get_support()
feature_importances = rf_selector.fit(X_train, y_train).feature_importances_

print(f"模型选择的特征数: {X_train_model.shape[1]}")
print("特征重要性排名:")
feature_importance_info = list(zip(feature_names, feature_importances, selected_features_model))
feature_importance_info.sort(key=lambda x: x[1], reverse=True)
for i, (name, importance, selected) in enumerate(feature_importance_info[:10]):
    print(f"  {i+1}. {name}: {importance:.4f} {'✓' if selected else '✗'}")

# 5. 特征变换
print("\n5. 特征变换")

# 5.1 多项式特征
print("5.1 多项式特征")
poly_features = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)
X_train_poly = poly_features.fit_transform(X_train[:, :5])  # 只用前5个特征避免维度爆炸
X_test_poly = poly_features.transform(X_test[:, :5])

print(f"原始特征数: 5")
print(f"多项式特征数: {X_train_poly.shape[1]}")
print(f"特征名称示例: {poly_features.get_feature_names_out(['f1', 'f2', 'f3', 'f4', 'f5'])[:10]}")

# 5.2 PCA降维
print("\n5.2 PCA降维")
pca = PCA(n_components=0.95)  # 保留95%的方差
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

print(f"原始特征数: {X_train.shape[1]}")
print(f"PCA后特征数: {X_train_pca.shape[1]}")
print(f"解释方差比: {pca.explained_variance_ratio_[:5]}")
print(f"累计解释方差比: {np.cumsum(pca.explained_variance_ratio_)[:5]}")

# 6. 特征选择方法对比
print("\n6. 特征选择方法对比")

feature_selection_methods = {
    'Original': (X_train, X_test),
    'Variance Threshold': (X_train_var, variance_threshold.transform(X_test)),
    'SelectKBest': (X_train_kbest, X_test_kbest),
    'RFE': (X_train_rfe, X_test_rfe),
    'Model Selection': (X_train_model, X_test_model),
    'PCA': (X_train_pca, X_test_pca)
}

results = {}
for method, (X_tr, X_te) in feature_selection_methods.items():
    # 训练逻辑回归
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_tr, y_train)
    y_pred = lr.predict(X_te)
    accuracy = accuracy_score(y_test, y_pred)
    
    results[method] = {
        'accuracy': accuracy,
        'n_features': X_tr.shape[1]
    }
    
    print(f"{method}:")
    print(f"  特征数: {X_tr.shape[1]}")
    print(f"  准确率: {accuracy:.4f}")

# 7. 特征工程Pipeline
print("\n7. 特征工程Pipeline")

# 创建完整的特征工程Pipeline
feature_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('selector', SelectKBest(score_func=f_classif, k=10)),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# 训练Pipeline
feature_pipeline.fit(X_train, y_train)
y_pred_pipeline = feature_pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)

print(f"Pipeline准确率: {accuracy_pipeline:.4f}")

# 获取选择的特征
selected_features_pipeline = feature_pipeline.named_steps['selector'].get_support()
print(f"Pipeline选择的特征数: {np.sum(selected_features_pipeline)}")

# 8. 回归任务的特征工程
print("\n8. 回归任务的特征工程")

# 使用Boston房价数据
boston = load_boston()
X_reg = boston.data
y_reg = boston.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.2, random_state=42)

# 回归特征选择
reg_selector = SelectKBest(score_func=f_regression, k=8)
X_train_reg_selected = reg_selector.fit_transform(X_train_reg, y_train_reg)
X_test_reg_selected = reg_selector.transform(X_test_reg)

# 比较原始特征和选择特征的性能
lr_reg = LinearRegression()

# 原始特征
lr_reg.fit(X_train_reg, y_train_reg)
y_pred_reg_orig = lr_reg.predict(X_test_reg)
mse_orig = mean_squared_error(y_test_reg, y_pred_reg_orig)

# 选择特征
lr_reg.fit(X_train_reg_selected, y_train_reg)
y_pred_reg_selected = lr_reg.predict(X_test_reg_selected)
mse_selected = mean_squared_error(y_test_reg, y_pred_reg_selected)

print(f"原始特征数: {X_train_reg.shape[1]}")
print(f"选择特征数: {X_train_reg_selected.shape[1]}")
print(f"原始特征MSE: {mse_orig:.4f}")
print(f"选择特征MSE: {mse_selected:.4f}")

# 9. 可视化分析
print("\n9. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 特征重要性对比
methods = ['SelectKBest', 'RFE', 'Model Selection']
n_selected = [np.sum(selected_features_kbest), 
              np.sum(selected_features_rfe), 
              np.sum(selected_features_model)]
accuracies = [results['SelectKBest']['accuracy'], 
              results['RFE']['accuracy'], 
              results['Model Selection']['accuracy']]

axes[0, 0].bar(methods, n_selected, alpha=0.7)
axes[0, 0].set_title('不同方法选择的特征数量')
axes[0, 0].set_ylabel('特征数量')
axes[0, 0].tick_params(axis='x', rotation=45)

# 9.2 准确率对比
axes[0, 1].bar(methods, accuracies, alpha=0.7, color='orange')
axes[0, 1].set_title('不同特征选择方法的准确率')
axes[0, 1].set_ylabel('准确率')
axes[0, 1].tick_params(axis='x', rotation=45)

# 9.3 特征数量vs准确率
all_methods = list(results.keys())
all_n_features = [results[method]['n_features'] for method in all_methods]
all_accuracies = [results[method]['accuracy'] for method in all_methods]

axes[0, 2].scatter(all_n_features, all_accuracies, s=100, alpha=0.7)
for i, method in enumerate(all_methods):
    axes[0, 2].annotate(method, (all_n_features[i], all_accuracies[i]), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
axes[0, 2].set_xlabel('特征数量')
axes[0, 2].set_ylabel('准确率')
axes[0, 2].set_title('特征数量 vs 准确率')
axes[0, 2].grid(True, alpha=0.3)

# 9.4 PCA方差解释
axes[1, 0].plot(range(1, len(pca.explained_variance_ratio_) + 1), 
                np.cumsum(pca.explained_variance_ratio_), 'bo-')
axes[1, 0].axhline(y=0.95, color='r', linestyle='--', label='95%方差')
axes[1, 0].set_xlabel('主成分数量')
axes[1, 0].set_ylabel('累计方差解释比')
axes[1, 0].set_title('PCA方差解释')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 9.5 特征得分分布
axes[1, 1].hist(scores, bins=20, alpha=0.7, edgecolor='black')
axes[1, 1].set_xlabel('特征得分')
axes[1, 1].set_ylabel('频数')
axes[1, 1].set_title('特征得分分布')
axes[1, 1].grid(True, alpha=0.3)

# 9.6 相关性矩阵热力图
correlation_subset = correlation_matrix.iloc[:10, :10]  # 只显示前10个特征
im = axes[1, 2].imshow(correlation_subset, cmap='coolwarm', aspect='auto')
axes[1, 2].set_xticks(range(10))
axes[1, 2].set_yticks(range(10))
axes[1, 2].set_xticklabels([f'F{i}' for i in range(10)], rotation=45)
axes[1, 2].set_yticklabels([f'F{i}' for i in range(10)])
axes[1, 2].set_title('特征相关性矩阵')
plt.colorbar(im, ax=axes[1, 2])

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/feature_engineering_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 特征工程总结 ===")
print("✅ 数据探索和相关性分析")
print("✅ 特征缩放方法比较")
print("✅ 多种特征选择方法")
print("✅ 特征变换和降维")
print("✅ 特征工程Pipeline")
print("✅ 回归任务特征工程")

print("\n=== 练习任务 ===")
print("1. 实现自定义特征选择器")
print("2. 尝试不同的特征变换方法")
print("3. 研究特征交互和组合")
print("4. 实现特征工程的自动化流程")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现基于信息增益的特征选择")
print("2. 研究非线性特征变换")
print("3. 实现特征工程的可视化工具")
print("4. 研究时间序列特征工程")
print("5. 实现特征重要性的稳定性分析")