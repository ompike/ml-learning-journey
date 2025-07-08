"""
集成学习方法实践
学习目标：掌握Bagging、Boosting和Stacking等集成学习技术
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_classification, load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                            AdaBoostClassifier, GradientBoostingClassifier,
                            VotingClassifier, BaggingClassifier)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

print("=== 集成学习方法实践 ===\n")

# 1. 数据准备
print("1. 数据准备")

# 使用红酒质量数据集
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, 
                                                    random_state=42, stratify=y)

# 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"特征数量: {X.shape[1]}")
print(f"类别数量: {len(np.unique(y))}")

# 2. Bagging方法
print("\n2. Bagging方法")

# 2.1 随机森林
print("2.1 随机森林")
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_scaled, y_train)
rf_pred = rf_classifier.predict(X_test_scaled)
rf_accuracy = accuracy_score(y_test, rf_pred)
print(f"随机森林准确率: {rf_accuracy:.4f}")

# 2.2 极端随机树
print("\n2.2 极端随机树")
et_classifier = ExtraTreesClassifier(n_estimators=100, random_state=42)
et_classifier.fit(X_train_scaled, y_train)
et_pred = et_classifier.predict(X_test_scaled)
et_accuracy = accuracy_score(y_test, et_pred)
print(f"极端随机树准确率: {et_accuracy:.4f}")

# 2.3 Bagging with 决策树
print("\n2.3 Bagging + 决策树")
bagging_classifier = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=100,
    random_state=42
)
bagging_classifier.fit(X_train_scaled, y_train)
bagging_pred = bagging_classifier.predict(X_test_scaled)
bagging_accuracy = accuracy_score(y_test, bagging_pred)
print(f"Bagging准确率: {bagging_accuracy:.4f}")

# 3. Boosting方法
print("\n3. Boosting方法")

# 3.1 AdaBoost
print("3.1 AdaBoost")
ada_classifier = AdaBoostClassifier(n_estimators=100, random_state=42)
ada_classifier.fit(X_train_scaled, y_train)
ada_pred = ada_classifier.predict(X_test_scaled)
ada_accuracy = accuracy_score(y_test, ada_pred)
print(f"AdaBoost准确率: {ada_accuracy:.4f}")

# 3.2 梯度提升
print("\n3.2 梯度提升")
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
gb_classifier.fit(X_train_scaled, y_train)
gb_pred = gb_classifier.predict(X_test_scaled)
gb_accuracy = accuracy_score(y_test, gb_pred)
print(f"梯度提升准确率: {gb_accuracy:.4f}")

# 4. 投票集成
print("\n4. 投票集成")

# 基础分类器
base_classifiers = [
    ('lr', LogisticRegression(random_state=42, max_iter=1000)),
    ('rf', RandomForestClassifier(n_estimators=50, random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('knn', KNeighborsClassifier(n_neighbors=5))
]

# 硬投票
hard_voting = VotingClassifier(estimators=base_classifiers, voting='hard')
hard_voting.fit(X_train_scaled, y_train)
hard_pred = hard_voting.predict(X_test_scaled)
hard_accuracy = accuracy_score(y_test, hard_pred)
print(f"硬投票准确率: {hard_accuracy:.4f}")

# 软投票
soft_voting = VotingClassifier(estimators=base_classifiers, voting='soft')
soft_voting.fit(X_train_scaled, y_train)
soft_pred = soft_voting.predict(X_test_scaled)
soft_accuracy = accuracy_score(y_test, soft_pred)
print(f"软投票准确率: {soft_accuracy:.4f}")

# 5. 模型性能对比
print("\n5. 模型性能对比")

ensemble_models = {
    'Random Forest': (rf_classifier, rf_accuracy),
    'Extra Trees': (et_classifier, et_accuracy),
    'Bagging': (bagging_classifier, bagging_accuracy),
    'AdaBoost': (ada_classifier, ada_accuracy),
    'Gradient Boosting': (gb_classifier, gb_accuracy),
    'Hard Voting': (hard_voting, hard_accuracy),
    'Soft Voting': (soft_voting, soft_accuracy)
}

print("集成方法性能排序:")
sorted_results = sorted(ensemble_models.items(), key=lambda x: x[1][1], reverse=True)
for i, (name, (model, accuracy)) in enumerate(sorted_results, 1):
    print(f"{i}. {name}: {accuracy:.4f}")

# 6. 交叉验证评估
print("\n6. 交叉验证评估")

cv_results = {}
for name, (model, _) in ensemble_models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    cv_results[name] = cv_scores
    print(f"{name}: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 7. 特征重要性分析
print("\n7. 特征重要性分析")

# 获取特征重要性（树模型）
tree_models = {
    'Random Forest': rf_classifier,
    'Extra Trees': et_classifier,
    'Gradient Boosting': gb_classifier
}

feature_importance_df = pd.DataFrame()
for name, model in tree_models.items():
    importance = model.feature_importances_
    feature_importance_df[name] = importance

feature_importance_df['Feature'] = wine.feature_names
feature_importance_df = feature_importance_df.set_index('Feature')

print("前10个重要特征:")
mean_importance = feature_importance_df.mean(axis=1).sort_values(ascending=False)
print(mean_importance.head(10))

# 8. 简单Stacking实现
print("\n8. 简单Stacking实现")

class SimpleStacking:
    def __init__(self, base_models, meta_model):
        self.base_models = base_models
        self.meta_model = meta_model
        
    def fit(self, X, y):
        # 训练基础模型
        self.trained_base_models = []
        base_predictions = np.zeros((X.shape[0], len(self.base_models)))
        
        for i, model in enumerate(self.base_models):
            model.fit(X, y)
            self.trained_base_models.append(model)
            base_predictions[:, i] = model.predict(X)
        
        # 训练元模型
        self.meta_model.fit(base_predictions, y)
        
    def predict(self, X):
        # 基础模型预测
        base_predictions = np.zeros((X.shape[0], len(self.trained_base_models)))
        for i, model in enumerate(self.trained_base_models):
            base_predictions[:, i] = model.predict(X)
        
        # 元模型预测
        return self.meta_model.predict(base_predictions)

# 创建Stacking集成
base_models = [
    DecisionTreeClassifier(random_state=42),
    RandomForestClassifier(n_estimators=50, random_state=42),
    SVC(random_state=42)
]
meta_model = LogisticRegression(random_state=42, max_iter=1000)

stacking = SimpleStacking(base_models, meta_model)
stacking.fit(X_train_scaled, y_train)
stacking_pred = stacking.predict(X_test_scaled)
stacking_accuracy = accuracy_score(y_test, stacking_pred)
print(f"简单Stacking准确率: {stacking_accuracy:.4f}")

# 9. 可视化分析
print("\n9. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 9.1 集成方法准确率对比
model_names = list(ensemble_models.keys())
accuracies = [acc for _, acc in ensemble_models.values()]
colors = plt.cm.Set3(np.linspace(0, 1, len(model_names)))

bars = axes[0, 0].bar(model_names, accuracies, color=colors, alpha=0.8)
axes[0, 0].set_title('集成方法准确率对比')
axes[0, 0].set_ylabel('准确率')
axes[0, 0].tick_params(axis='x', rotation=45)
axes[0, 0].set_ylim(0.8, 1.0)

# 添加数值标签
for bar, acc in zip(bars, accuracies):
    axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                   f'{acc:.3f}', ha='center', va='bottom', fontsize=9)

# 9.2 交叉验证结果箱线图
cv_data = [cv_results[name] for name in model_names]
bp = axes[0, 1].boxplot(cv_data, labels=model_names, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
axes[0, 1].set_title('交叉验证分数分布')
axes[0, 1].set_ylabel('CV分数')
axes[0, 1].tick_params(axis='x', rotation=45)

# 9.3 特征重要性热力图
sns.heatmap(feature_importance_df.T, cmap='viridis', ax=axes[0, 2], cbar=True)
axes[0, 2].set_title('特征重要性对比')
axes[0, 2].set_xlabel('特征')
axes[0, 2].set_ylabel('模型')

# 9.4 随机森林树的数量vs性能
n_estimators_range = range(10, 201, 20)
rf_scores = []
for n_est in n_estimators_range:
    rf_temp = RandomForestClassifier(n_estimators=n_est, random_state=42)
    scores = cross_val_score(rf_temp, X_train_scaled, y_train, cv=3)
    rf_scores.append(scores.mean())

axes[1, 0].plot(n_estimators_range, rf_scores, 'o-', color='green')
axes[1, 0].set_title('随机森林：树的数量 vs 性能')
axes[1, 0].set_xlabel('树的数量')
axes[1, 0].set_ylabel('交叉验证分数')
axes[1, 0].grid(True, alpha=0.3)

# 9.5 AdaBoost学习率vs性能
learning_rates = [0.01, 0.1, 0.5, 1.0, 1.5, 2.0]
ada_scores = []
for lr in learning_rates:
    ada_temp = AdaBoostClassifier(learning_rate=lr, n_estimators=50, random_state=42)
    scores = cross_val_score(ada_temp, X_train_scaled, y_train, cv=3)
    ada_scores.append(scores.mean())

axes[1, 1].plot(learning_rates, ada_scores, 'o-', color='red')
axes[1, 1].set_title('AdaBoost：学习率 vs 性能')
axes[1, 1].set_xlabel('学习率')
axes[1, 1].set_ylabel('交叉验证分数')
axes[1, 1].grid(True, alpha=0.3)

# 9.6 特征重要性排序（平均）
top_features = mean_importance.head(10)
axes[1, 2].barh(range(len(top_features)), top_features.values)
axes[1, 2].set_yticks(range(len(top_features)))
axes[1, 2].set_yticklabels(top_features.index)
axes[1, 2].set_title('Top 10 重要特征（平均重要性）')
axes[1, 2].set_xlabel('重要性分数')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/ensemble_methods_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 集成学习总结 ===")
print("✅ Bagging方法（随机森林、极端随机树）")
print("✅ Boosting方法（AdaBoost、梯度提升）")
print("✅ 投票集成（硬投票、软投票）")
print("✅ 简单Stacking实现")
print("✅ 特征重要性分析")
print("✅ 超参数对性能的影响")

print("\n=== 练习任务 ===")
print("1. 实现更复杂的Stacking")
print("2. 尝试XGBoost和LightGBM")
print("3. 研究动态集成方法")
print("4. 实现多样性分析")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现Blending集成方法")
print("2. 研究多级Stacking")
print("3. 实现在线集成学习")
print("4. 研究集成方法的可解释性")
print("5. 实现自适应集成权重调整")