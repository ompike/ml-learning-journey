"""
Scikit-learn基础实践
学习目标：掌握scikit-learn的基本工作流程和核心功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, load_iris, make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=== Scikit-learn基础实践 ===\n")

# 1. 数据加载和探索
print("1. 数据加载和探索")

# 加载鸢尾花数据集
iris = load_iris()
print(f"鸢尾花数据集特征数: {iris.data.shape[1]}")
print(f"样本数: {iris.data.shape[0]}")
print(f"类别数: {len(iris.target_names)}")
print(f"特征名称: {iris.feature_names}")
print(f"类别名称: {iris.target_names}")

# 创建DataFrame方便分析
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
iris_df['species'] = iris_df['target'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

print("\n数据基本统计信息:")
print(iris_df.describe())

# 2. 数据预处理
print("\n2. 数据预处理")

# 检查缺失值
print(f"缺失值情况:\n{iris_df.isnull().sum()}")

# 特征和标签分离
X = iris.data
y = iris.target

# 数据分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 特征标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("标准化前后特征统计:")
print(f"原始数据均值: {np.mean(X_train, axis=0)[:2]}")
print(f"标准化后均值: {np.mean(X_train_scaled, axis=0)[:2]}")
print(f"原始数据标准差: {np.std(X_train, axis=0)[:2]}")
print(f"标准化后标准差: {np.std(X_train_scaled, axis=0)[:2]}")

# 3. 模型训练和预测
print("\n3. 模型训练和预测")

# 训练多个分类器
classifiers = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(random_state=42)
}

results = {}

for name, clf in classifiers.items():
    # 训练模型
    clf.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = clf.predict(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    results[name] = accuracy
    
    print(f"{name} 准确率: {accuracy:.4f}")

# 4. 交叉验证
print("\n4. 交叉验证")

for name, clf in classifiers.items():
    # 5折交叉验证
    cv_scores = cross_val_score(clf, X_train_scaled, y_train, cv=5)
    print(f"{name} CV准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 5. Pipeline使用
print("\n5. Pipeline使用")

# 创建Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 训练Pipeline
pipeline.fit(X_train, y_train)

# 预测
y_pred_pipeline = pipeline.predict(X_test)
accuracy_pipeline = accuracy_score(y_test, y_pred_pipeline)
print(f"Pipeline准确率: {accuracy_pipeline:.4f}")

# Pipeline的优势：可以一次性进行交叉验证
cv_scores_pipeline = cross_val_score(pipeline, X, y, cv=5)
print(f"Pipeline CV准确率: {cv_scores_pipeline.mean():.4f} (+/- {cv_scores_pipeline.std() * 2:.4f})")

# 6. 超参数调优
print("\n6. 超参数调优")

# 定义参数网格
param_grid = {
    'classifier__n_estimators': [50, 100, 200],
    'classifier__max_depth': [3, 5, 7, None],
    'classifier__min_samples_split': [2, 5, 10]
}

# 网格搜索
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳CV分数: {grid_search.best_score_:.4f}")

# 使用最佳模型预测
best_model = grid_search.best_estimator_
y_pred_best = best_model.predict(X_test)
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"最佳模型测试准确率: {accuracy_best:.4f}")

# 7. 回归任务示例
print("\n7. 回归任务示例")

# 生成回归数据
np.random.seed(42)
X_reg = np.random.randn(100, 4)
y_reg = (X_reg[:, 0] * 2 + X_reg[:, 1] * -1 + X_reg[:, 2] * 0.5 + 
         np.random.randn(100) * 0.1)

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg, y_reg, test_size=0.3, random_state=42)

# 回归模型
regressors = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

for name, regressor in regressors.items():
    regressor.fit(X_train_reg, y_train_reg)
    y_pred_reg = regressor.predict(X_test_reg)
    mse = mean_squared_error(y_test_reg, y_pred_reg)
    rmse = np.sqrt(mse)
    print(f"{name} RMSE: {rmse:.4f}")

# 8. 模型评估详细分析
print("\n8. 模型评估详细分析")

# 使用最佳分类模型
y_pred_final = best_model.predict(X_test)

# 分类报告
print("分类报告:")
print(classification_report(y_test, y_pred_final, target_names=iris.target_names))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred_final)
print(f"\n混淆矩阵:\n{cm}")

# 9. 特征重要性分析
print("\n9. 特征重要性分析")

# 获取随机森林的特征重要性
rf_model = best_model.named_steps['classifier']
importances = rf_model.feature_importances_

print("特征重要性:")
for i, importance in enumerate(importances):
    print(f"{iris.feature_names[i]}: {importance:.4f}")

# 10. 可视化分析
print("\n10. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 10.1 数据分布
iris_df.hist(figsize=(10, 8), bins=20, ax=axes[0, 0])
axes[0, 0].set_title('特征分布')

# 10.2 相关性矩阵
correlation_matrix = iris_df.iloc[:, :-2].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=axes[0, 1])
axes[0, 1].set_title('特征相关性')

# 10.3 类别分布
iris_df['species'].value_counts().plot(kind='bar', ax=axes[0, 2])
axes[0, 2].set_title('类别分布')
axes[0, 2].tick_params(axis='x', rotation=45)

# 10.4 模型准确率对比
models = list(results.keys())
accuracies = list(results.values())
axes[1, 0].bar(models, accuracies)
axes[1, 0].set_title('模型准确率对比')
axes[1, 0].set_ylabel('准确率')
axes[1, 0].tick_params(axis='x', rotation=45)

# 10.5 特征重要性
axes[1, 1].bar(iris.feature_names, importances)
axes[1, 1].set_title('特征重要性')
axes[1, 1].set_ylabel('重要性')
axes[1, 1].tick_params(axis='x', rotation=45)

# 10.6 混淆矩阵热力图
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, yticklabels=iris.target_names, ax=axes[1, 2])
axes[1, 2].set_title('混淆矩阵')
axes[1, 2].set_xlabel('预测标签')
axes[1, 2].set_ylabel('真实标签')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/sklearn_basics_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 11. 保存和加载模型
print("\n11. 模型保存和加载")

import joblib

# 保存模型
model_filename = '/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/best_iris_model.pkl'
joblib.dump(best_model, model_filename)
print(f"模型已保存到: {model_filename}")

# 加载模型
loaded_model = joblib.load(model_filename)
y_pred_loaded = loaded_model.predict(X_test)
accuracy_loaded = accuracy_score(y_test, y_pred_loaded)
print(f"加载模型的准确率: {accuracy_loaded:.4f}")

# 12. 实际应用示例
print("\n12. 实际应用示例")

def predict_iris_species(sepal_length, sepal_width, petal_length, petal_width):
    """预测鸢尾花种类的函数"""
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = loaded_model.predict(features)[0]
    species_name = iris.target_names[prediction]
    probability = loaded_model.predict_proba(features)[0]
    
    print(f"预测种类: {species_name}")
    print("各类别概率:")
    for i, prob in enumerate(probability):
        print(f"  {iris.target_names[i]}: {prob:.4f}")
    
    return species_name

# 示例预测
print("示例预测 - 花萼长度:5.1, 花萼宽度:3.5, 花瓣长度:1.4, 花瓣宽度:0.2")
predict_iris_species(5.1, 3.5, 1.4, 0.2)

print("\n=== Scikit-learn基础总结 ===")
print("✅ 数据加载和探索")
print("✅ 数据预处理和特征缩放")
print("✅ 模型训练和预测")
print("✅ 交叉验证")
print("✅ Pipeline构建")
print("✅ 超参数调优")
print("✅ 模型评估和解释")
print("✅ 模型保存和部署")

print("\n=== 练习任务 ===")
print("1. 使用不同的评估指标（F1-score, ROC-AUC等）")
print("2. 尝试不同的交叉验证策略")
print("3. 实现自定义的预处理步骤")
print("4. 比较更多的算法")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 使用多个数据集进行实验")
print("2. 实现自定义的评估指标")
print("3. 创建自定义的转换器")
print("4. 研究不平衡数据集的处理方法")
print("5. 实现模型的在线学习版本")