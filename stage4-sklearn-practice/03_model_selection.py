"""
模型选择和调优实践
学习目标：掌握交叉验证、超参数调优和模型比较的方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import (train_test_split, cross_val_score, 
                                   GridSearchCV, RandomizedSearchCV, 
                                   StratifiedKFold, learning_curve, validation_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                           f1_score, classification_report, confusion_matrix,
                           roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from scipy.stats import randint, uniform
import time
import warnings
warnings.filterwarnings('ignore')

print("=== 模型选择和调优实践 ===\n")

# 1. 数据准备
print("1. 数据准备")

# 生成分类数据
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15, 
                          n_redundant=5, n_classes=3, n_clusters_per_class=1, 
                          random_state=42)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                    random_state=42, stratify=y)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"类别分布: {np.bincount(y_train)}")

# 2. 基础模型比较
print("\n2. 基础模型比较")

# 定义多个分类器
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'KNN': KNeighborsClassifier(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# 标准化数据
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 基础模型评估
basic_results = {}
for name, clf in classifiers.items():
    start_time = time.time()
    
    # 训练模型
    clf.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = clf.predict(X_test_scaled)
    
    # 计算指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # 计算AUC（多分类）
    try:
        y_pred_proba = clf.predict_proba(X_test_scaled)
        auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    except:
        auc = np.nan
    
    training_time = time.time() - start_time
    
    basic_results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'time': training_time
    }
    
    print(f"{name}:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  F1分数: {f1:.4f}")
    print(f"  训练时间: {training_time:.4f}s")

# 3. 交叉验证
print("\n3. 交叉验证")

# 使用分层K折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, clf in classifiers.items():
    # 计算交叉验证分数
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    cv_f1_scores = cross_val_score(clf, X_train_scaled, y_train, cv=cv, scoring='f1_weighted')
    
    cv_results[name] = {
        'accuracy_mean': cv_scores.mean(),
        'accuracy_std': cv_scores.std(),
        'f1_mean': cv_f1_scores.mean(),
        'f1_std': cv_f1_scores.std()
    }
    
    print(f"{name}:")
    print(f"  CV准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    print(f"  CV F1分数: {cv_f1_scores.mean():.4f} (+/- {cv_f1_scores.std() * 2:.4f})")

# 4. 网格搜索超参数调优
print("\n4. 网格搜索超参数调优")

# 为随机森林进行网格搜索
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("随机森林网格搜索...")
rf_grid = GridSearchCV(RandomForestClassifier(random_state=42), 
                       rf_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
rf_grid.fit(X_train_scaled, y_train)

print(f"最佳参数: {rf_grid.best_params_}")
print(f"最佳CV分数: {rf_grid.best_score_:.4f}")

# 测试集评估
rf_best_pred = rf_grid.best_estimator_.predict(X_test_scaled)
rf_best_accuracy = accuracy_score(y_test, rf_best_pred)
print(f"测试集准确率: {rf_best_accuracy:.4f}")

# 为SVM进行网格搜索
svm_param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

print("\nSVM网格搜索...")
svm_grid = GridSearchCV(SVC(random_state=42, probability=True), 
                        svm_param_grid, cv=5, scoring='accuracy', n_jobs=-1)
svm_grid.fit(X_train_scaled, y_train)

print(f"最佳参数: {svm_grid.best_params_}")
print(f"最佳CV分数: {svm_grid.best_score_:.4f}")

# 5. 随机搜索
print("\n5. 随机搜索")

# 为梯度提升进行随机搜索
gb_param_dist = {
    'n_estimators': randint(50, 300),
    'max_depth': randint(3, 8),
    'min_samples_split': randint(2, 20),
    'min_samples_leaf': randint(1, 10),
    'learning_rate': uniform(0.01, 0.3)
}

print("梯度提升随机搜索...")
gb_random = RandomizedSearchCV(GradientBoostingClassifier(random_state=42), 
                               gb_param_dist, n_iter=100, cv=5, 
                               scoring='accuracy', n_jobs=-1, random_state=42)
gb_random.fit(X_train_scaled, y_train)

print(f"最佳参数: {gb_random.best_params_}")
print(f"最佳CV分数: {gb_random.best_score_:.4f}")

# 6. 学习曲线分析
print("\n6. 学习曲线分析")

def plot_learning_curve_custom(estimator, X, y, title, cv=5):
    """绘制学习曲线"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 10))
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    val_scores_mean = np.mean(val_scores, axis=1)
    val_scores_std = np.std(val_scores, axis=1)
    
    return train_sizes, train_scores_mean, train_scores_std, val_scores_mean, val_scores_std

# 计算几个模型的学习曲线
learning_curve_models = {
    'Random Forest': rf_grid.best_estimator_,
    'SVM': svm_grid.best_estimator_,
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
}

learning_curves_data = {}
for name, model in learning_curve_models.items():
    print(f"计算{name}学习曲线...")
    lc_data = plot_learning_curve_custom(model, X_train_scaled, y_train, name)
    learning_curves_data[name] = lc_data

# 7. 验证曲线分析
print("\n7. 验证曲线分析")

# 分析随机森林的n_estimators参数
n_estimators_range = [10, 50, 100, 150, 200, 250, 300]
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X_train_scaled, y_train,
    param_name='n_estimators', param_range=n_estimators_range,
    cv=5, scoring='accuracy', n_jobs=-1)

train_scores_mean = np.mean(train_scores, axis=1)
train_scores_std = np.std(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

print("随机森林n_estimators验证曲线:")
for i, n_est in enumerate(n_estimators_range):
    print(f"  {n_est}: 训练{train_scores_mean[i]:.4f}, 验证{val_scores_mean[i]:.4f}")

# 8. 模型融合
print("\n8. 模型融合")

# 简单投票融合
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf_grid.best_estimator_),
        ('svm', svm_grid.best_estimator_),
        ('gb', gb_random.best_estimator_)
    ],
    voting='soft'  # 使用概率投票
)

voting_clf.fit(X_train_scaled, y_train)
voting_pred = voting_clf.predict(X_test_scaled)
voting_accuracy = accuracy_score(y_test, voting_pred)

print(f"投票融合准确率: {voting_accuracy:.4f}")

# 9. 模型性能对比
print("\n9. 模型性能对比")

# 收集所有优化后的模型结果
final_results = {}

models_to_compare = {
    'Random Forest (Tuned)': rf_grid.best_estimator_,
    'SVM (Tuned)': svm_grid.best_estimator_,
    'Gradient Boosting (Tuned)': gb_random.best_estimator_,
    'Voting Classifier': voting_clf
}

for name, model in models_to_compare.items():
    if name != 'Voting Classifier':  # 已经训练过了
        model.fit(X_train_scaled, y_train)
    
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    final_results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
    }

print("最终模型性能对比:")
for name, metrics in final_results.items():
    print(f"{name}:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")

# 10. 可视化分析
print("\n10. 可视化分析")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 10.1 基础模型准确率对比
models = list(basic_results.keys())
accuracies = [basic_results[model]['accuracy'] for model in models]
axes[0, 0].bar(models, accuracies, alpha=0.7)
axes[0, 0].set_title('基础模型准确率对比')
axes[0, 0].set_ylabel('准确率')
axes[0, 0].tick_params(axis='x', rotation=45)

# 10.2 训练时间对比
times = [basic_results[model]['time'] for model in models]
axes[0, 1].bar(models, times, alpha=0.7, color='orange')
axes[0, 1].set_title('模型训练时间对比')
axes[0, 1].set_ylabel('时间(秒)')
axes[0, 1].tick_params(axis='x', rotation=45)

# 10.3 交叉验证结果
cv_means = [cv_results[model]['accuracy_mean'] for model in models]
cv_stds = [cv_results[model]['accuracy_std'] for model in models]
axes[0, 2].bar(models, cv_means, yerr=cv_stds, alpha=0.7, color='green', capsize=5)
axes[0, 2].set_title('交叉验证准确率')
axes[0, 2].set_ylabel('准确率')
axes[0, 2].tick_params(axis='x', rotation=45)

# 10.4 学习曲线 - Random Forest
name = 'Random Forest'
train_sizes, train_mean, train_std, val_mean, val_std = learning_curves_data[name]
axes[1, 0].plot(train_sizes, train_mean, 'o-', label='训练分数')
axes[1, 0].fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
axes[1, 0].plot(train_sizes, val_mean, 'o-', label='验证分数')
axes[1, 0].fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
axes[1, 0].set_title(f'{name} 学习曲线')
axes[1, 0].set_xlabel('训练样本数')
axes[1, 0].set_ylabel('分数')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 10.5 验证曲线
axes[1, 1].plot(n_estimators_range, train_scores_mean, 'o-', label='训练分数')
axes[1, 1].fill_between(n_estimators_range, train_scores_mean - train_scores_std, 
                       train_scores_mean + train_scores_std, alpha=0.1)
axes[1, 1].plot(n_estimators_range, val_scores_mean, 'o-', label='验证分数')
axes[1, 1].fill_between(n_estimators_range, val_scores_mean - val_scores_std, 
                       val_scores_mean + val_scores_std, alpha=0.1)
axes[1, 1].set_title('随机森林 n_estimators 验证曲线')
axes[1, 1].set_xlabel('n_estimators')
axes[1, 1].set_ylabel('分数')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 10.6 最终模型性能雷达图
from math import pi

final_models = list(final_results.keys())
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
angles += angles[:1]

axes[1, 2].set_theta_offset(pi / 2)
axes[1, 2].set_theta_direction(-1)
axes[1, 2].set_thetagrids(np.degrees(angles[:-1]), metrics)

for i, model in enumerate(final_models[:3]):  # 只显示前3个模型
    values = [final_results[model][metric] for metric in metrics]
    values += values[:1]
    axes[1, 2].plot(angles, values, 'o-', linewidth=2, label=model)
    axes[1, 2].fill(angles, values, alpha=0.25)

axes[1, 2].set_ylim(0, 1)
axes[1, 2].set_title('模型性能雷达图')
axes[1, 2].legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))

# 10.7 F1分数对比
f1_scores = [final_results[model]['f1'] for model in final_models]
axes[2, 0].bar(final_models, f1_scores, alpha=0.7, color='purple')
axes[2, 0].set_title('最终模型F1分数对比')
axes[2, 0].set_ylabel('F1分数')
axes[2, 0].tick_params(axis='x', rotation=45)

# 10.8 AUC分数对比
auc_scores = [final_results[model]['auc'] for model in final_models]
axes[2, 1].bar(final_models, auc_scores, alpha=0.7, color='red')
axes[2, 1].set_title('最终模型AUC分数对比')
axes[2, 1].set_ylabel('AUC分数')
axes[2, 1].tick_params(axis='x', rotation=45)

# 10.9 网格搜索热力图（随机森林）
grid_scores = rf_grid.cv_results_['mean_test_score'].reshape(
    len(rf_param_grid['max_depth']), len(rf_param_grid['n_estimators']))
im = axes[2, 2].imshow(grid_scores, cmap='viridis', aspect='auto')
axes[2, 2].set_xticks(range(len(rf_param_grid['n_estimators'])))
axes[2, 2].set_yticks(range(len(rf_param_grid['max_depth'])))
axes[2, 2].set_xticklabels(rf_param_grid['n_estimators'])
axes[2, 2].set_yticklabels(rf_param_grid['max_depth'])
axes[2, 2].set_xlabel('n_estimators')
axes[2, 2].set_ylabel('max_depth')
axes[2, 2].set_title('随机森林网格搜索热力图')
plt.colorbar(im, ax=axes[2, 2])

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/model_selection_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 模型选择和调优总结 ===")
print("✅ 基础模型比较和评估")
print("✅ 交叉验证方法应用")
print("✅ 网格搜索超参数调优")
print("✅ 随机搜索优化")
print("✅ 学习曲线和验证曲线分析")
print("✅ 模型融合技术")
print("✅ 综合性能评估")

print("\n=== 练习任务 ===")
print("1. 实现贝叶斯优化超参数调优")
print("2. 尝试不同的模型融合策略")
print("3. 研究早停和学习率调度")
print("4. 实现自定义评估指标")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现多目标超参数优化")
print("2. 研究模型解释性和可视化")
print("3. 实现增量学习和在线调优")
print("4. 研究模型鲁棒性分析")
print("5. 实现自动机器学习(AutoML)流程")