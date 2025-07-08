"""
模型解释性和可解释性
学习目标：掌握机器学习模型的解释方法和可视化技术
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_boston, load_wine, make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.inspection import permutation_importance, partial_dependence, PartialDependenceDisplay
from sklearn.metrics import accuracy_score, mean_squared_error
import shap
import lime
import lime.lime_tabular
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

print("=== 模型解释性和可解释性 ===\n")

# 1. 模型解释性理论
print("1. 模型解释性理论")
print("模型解释性的重要性：")
print("- 建立信任：理解模型决策过程")
print("- 监管合规：满足法律法规要求")
print("- 错误诊断：发现模型缺陷")
print("- 知识发现：从模型中学习领域知识")

print("\n解释性方法分类：")
print("1. 全局解释：理解整个模型的行为")
print("2. 局部解释：解释单个预测")
print("3. 内在解释：模型本身可解释")
print("4. 后验解释：通过外部方法解释")

# 2. 数据准备
print("\n2. 数据准备")

# 加载数据集
boston = load_boston()
wine = load_wine()

# 回归任务 - 波士顿房价
X_boston, y_boston = boston.data, boston.target
feature_names_boston = boston.feature_names

# 分类任务 - 葡萄酒分类
X_wine, y_wine = wine.data, wine.target
feature_names_wine = wine.feature_names
class_names_wine = wine.target_names

print(f"波士顿房价数据: {X_boston.shape}")
print(f"葡萄酒数据: {X_wine.shape}")

# 数据分割
X_boston_train, X_boston_test, y_boston_train, y_boston_test = train_test_split(
    X_boston, y_boston, test_size=0.2, random_state=42)

X_wine_train, X_wine_test, y_wine_train, y_wine_test = train_test_split(
    X_wine, y_wine, test_size=0.2, random_state=42)

# 数据标准化
scaler_boston = StandardScaler()
X_boston_train_scaled = scaler_boston.fit_transform(X_boston_train)
X_boston_test_scaled = scaler_boston.transform(X_boston_test)

scaler_wine = StandardScaler()
X_wine_train_scaled = scaler_wine.fit_transform(X_wine_train)
X_wine_test_scaled = scaler_wine.transform(X_wine_test)

# 3. 训练不同类型的模型
print("\n3. 训练不同类型的模型")

# 回归模型
models_regression = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

regression_results = {}
for name, model in models_regression.items():
    model.fit(X_boston_train_scaled, y_boston_train)
    y_pred = model.predict(X_boston_test_scaled)
    mse = mean_squared_error(y_boston_test, y_pred)
    regression_results[name] = {'model': model, 'mse': mse}
    print(f"{name} MSE: {mse:.4f}")

# 分类模型
models_classification = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=5)
}

classification_results = {}
for name, model in models_classification.items():
    model.fit(X_wine_train_scaled, y_wine_train)
    y_pred = model.predict(X_wine_test_scaled)
    accuracy = accuracy_score(y_wine_test, y_pred)
    classification_results[name] = {'model': model, 'accuracy': accuracy}
    print(f"{name} Accuracy: {accuracy:.4f}")

# 4. 特征重要性分析
print("\n4. 特征重要性分析")

def analyze_feature_importance(model, feature_names, model_name):
    """分析特征重要性"""
    print(f"\n{model_name} 特征重要性:")
    
    if hasattr(model, 'feature_importances_'):
        # 基于树的模型
        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=False)
        
        print("Top 10 重要特征:")
        print(feature_importance_df.head(10))
        
        return feature_importance_df
    
    elif hasattr(model, 'coef_'):
        # 线性模型
        if len(model.coef_.shape) == 1:
            # 回归或二分类
            coefficients = np.abs(model.coef_)
        else:
            # 多分类
            coefficients = np.mean(np.abs(model.coef_), axis=0)
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': coefficients
        }).sort_values('importance', ascending=False)
        
        print("Top 10 重要特征:")
        print(feature_importance_df.head(10))
        
        return feature_importance_df
    
    return None

# 分析回归模型特征重要性
rf_reg_importance = analyze_feature_importance(
    regression_results['Random Forest']['model'], 
    feature_names_boston, 
    'Random Forest Regression'
)

lr_reg_importance = analyze_feature_importance(
    regression_results['Linear Regression']['model'], 
    feature_names_boston, 
    'Linear Regression'
)

# 分析分类模型特征重要性
rf_cls_importance = analyze_feature_importance(
    classification_results['Random Forest']['model'], 
    feature_names_wine, 
    'Random Forest Classification'
)

# 5. 排列重要性
print("\n5. 排列重要性")

def calculate_permutation_importance(model, X, y, feature_names, scoring='neg_mean_squared_error'):
    """计算排列重要性"""
    perm_importance = permutation_importance(model, X, y, 
                                           scoring=scoring, 
                                           n_repeats=10, 
                                           random_state=42)
    
    perm_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance_mean': perm_importance.importances_mean,
        'importance_std': perm_importance.importances_std
    }).sort_values('importance_mean', ascending=False)
    
    return perm_importance_df

# 回归模型的排列重要性
rf_reg_perm = calculate_permutation_importance(
    regression_results['Random Forest']['model'],
    X_boston_test_scaled, y_boston_test,
    feature_names_boston,
    scoring='neg_mean_squared_error'
)

print("随机森林回归 - 排列重要性 Top 10:")
print(rf_reg_perm.head(10))

# 分类模型的排列重要性
rf_cls_perm = calculate_permutation_importance(
    classification_results['Random Forest']['model'],
    X_wine_test_scaled, y_wine_test,
    feature_names_wine,
    scoring='accuracy'
)

print("\n随机森林分类 - 排列重要性 Top 10:")
print(rf_cls_perm.head(10))

# 6. 部分依赖图
print("\n6. 部分依赖图")

def create_partial_dependence_plots(model, X, feature_names, features_to_plot):
    """创建部分依赖图"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.ravel()
    
    for i, feature_idx in enumerate(features_to_plot[:6]):
        if i < 6:
            try:
                display = PartialDependenceDisplay.from_estimator(
                    model, X, [feature_idx], 
                    feature_names=feature_names,
                    ax=axes[i]
                )
                axes[i].set_title(f'{feature_names[feature_idx]}的部分依赖')
            except Exception as e:
                axes[i].text(0.5, 0.5, f'Error: {str(e)}', 
                           transform=axes[i].transAxes, ha='center')
    
    plt.tight_layout()
    return fig

# 为随机森林回归创建部分依赖图
important_features_reg = rf_reg_importance.head(6).index.tolist()
important_feature_indices_reg = [list(feature_names_boston).index(feat) for feat in rf_reg_importance.head(6)['feature'].tolist()]

print("创建回归模型部分依赖图...")
try:
    fig_pdp_reg = create_partial_dependence_plots(
        regression_results['Random Forest']['model'],
        X_boston_train_scaled,
        feature_names_boston,
        important_feature_indices_reg
    )
    fig_pdp_reg.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/partial_dependence_regression.png', 
                        dpi=300, bbox_inches='tight')
    plt.close()
    print("回归部分依赖图已保存")
except Exception as e:
    print(f"部分依赖图生成失败: {e}")

# 7. SHAP值分析
print("\n7. SHAP值分析")

try:
    # 初始化SHAP解释器
    rf_reg_model = regression_results['Random Forest']['model']
    
    # 使用TreeExplainer for tree-based models
    explainer_reg = shap.TreeExplainer(rf_reg_model)
    shap_values_reg = explainer_reg.shap_values(X_boston_test_scaled[:100])  # 使用子集
    
    print("SHAP值计算完成")
    print(f"SHAP值形状: {shap_values_reg.shape}")
    
    # 全局特征重要性
    shap_importance = np.abs(shap_values_reg).mean(0)
    shap_importance_df = pd.DataFrame({
        'feature': feature_names_boston,
        'shap_importance': shap_importance
    }).sort_values('shap_importance', ascending=False)
    
    print("\nSHAP全局特征重要性 Top 10:")
    print(shap_importance_df.head(10))
    
    # 创建SHAP汇总图
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values_reg, X_boston_test_scaled[:100], 
                     feature_names=feature_names_boston, show=False)
    plt.title('SHAP Summary Plot - Random Forest Regression')
    plt.tight_layout()
    plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/shap_summary.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("SHAP汇总图已保存")
    
except Exception as e:
    print(f"SHAP分析失败: {e}")

# 8. LIME局部解释
print("\n8. LIME局部解释")

try:
    # 初始化LIME解释器
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(
        X_wine_train_scaled,
        feature_names=feature_names_wine,
        class_names=class_names_wine,
        mode='classification'
    )
    
    # 选择一个实例进行解释
    instance_idx = 0
    instance = X_wine_test_scaled[instance_idx]
    
    # 获取解释
    rf_cls_model = classification_results['Random Forest']['model']
    lime_explanation = lime_explainer.explain_instance(
        instance, 
        rf_cls_model.predict_proba, 
        num_features=len(feature_names_wine)
    )
    
    print(f"LIME解释实例 {instance_idx}:")
    print(f"真实标签: {class_names_wine[y_wine_test[instance_idx]]}")
    print(f"预测标签: {class_names_wine[rf_cls_model.predict([instance])[0]]}")
    
    # 获取解释结果
    lime_explanation_list = lime_explanation.as_list()
    print("\nLIME特征贡献 Top 10:")
    for feature, contribution in lime_explanation_list[:10]:
        print(f"  {feature}: {contribution:.4f}")
    
    # 保存LIME解释图
    fig = lime_explanation.as_pyplot_figure()
    fig.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/lime_explanation.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("LIME解释图已保存")
    
except Exception as e:
    print(f"LIME分析失败: {e}")

# 9. 决策树可视化
print("\n9. 决策树可视化")

# 训练一个简单的决策树用于可视化
dt_simple = DecisionTreeClassifier(max_depth=3, random_state=42)
dt_simple.fit(X_wine_train_scaled, y_wine_train)

# 可视化决策树
plt.figure(figsize=(20, 12))
plot_tree(dt_simple, 
          feature_names=feature_names_wine,
          class_names=class_names_wine,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('决策树可视化 - 葡萄酒分类')
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/decision_tree_visualization.png', 
            dpi=300, bbox_inches='tight')
plt.close()
print("决策树可视化已保存")

# 10. 模型比较和解释性分析
print("\n10. 模型比较和解释性分析")

def compare_model_interpretability():
    """比较不同模型的解释性"""
    interpretability_comparison = {
        '模型': ['线性回归', '逻辑回归', '决策树', '随机森林', '神经网络', 'SVM'],
        '内在解释性': ['高', '高', '高', '中', '低', '低'],
        '全局解释': ['容易', '容易', '容易', '中等', '困难', '困难'],
        '局部解释': ['容易', '容易', '容易', '中等', '困难', '困难'],
        '特征重要性': ['系数', '系数', '信息增益', '不纯度减少', '需要外部方法', '需要外部方法'],
        '可视化': ['容易', '容易', '容易', '中等', '困难', '困难']
    }
    
    comparison_df = pd.DataFrame(interpretability_comparison)
    print("模型解释性比较:")
    print(comparison_df.to_string(index=False))
    
    return comparison_df

interpretability_df = compare_model_interpretability()

# 11. 综合可视化分析
print("\n11. 综合可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 11.1 特征重要性对比 (回归)
if rf_reg_importance is not None and lr_reg_importance is not None:
    top_features = rf_reg_importance.head(10)
    axes[0, 0].barh(range(len(top_features)), top_features['importance'])
    axes[0, 0].set_yticks(range(len(top_features)))
    axes[0, 0].set_yticklabels(top_features['feature'])
    axes[0, 0].set_title('随机森林回归 - 特征重要性')
    axes[0, 0].set_xlabel('重要性')

# 11.2 排列重要性 vs 内在重要性
if rf_reg_importance is not None:
    # 合并数据进行比较
    importance_comparison = pd.merge(
        rf_reg_importance[['feature', 'importance']], 
        rf_reg_perm[['feature', 'importance_mean']], 
        on='feature'
    )
    
    axes[0, 1].scatter(importance_comparison['importance'], 
                      importance_comparison['importance_mean'], alpha=0.7)
    axes[0, 1].set_xlabel('内在重要性')
    axes[0, 1].set_ylabel('排列重要性')
    axes[0, 1].set_title('内在重要性 vs 排列重要性')
    
    # 添加对角线
    max_val = max(importance_comparison['importance'].max(), 
                  importance_comparison['importance_mean'].max())
    axes[0, 1].plot([0, max_val], [0, max_val], 'r--', alpha=0.5)

# 11.3 模型性能对比
model_names = list(regression_results.keys())
model_mses = [regression_results[name]['mse'] for name in model_names]

axes[0, 2].bar(model_names, model_mses, alpha=0.7)
axes[0, 2].set_title('回归模型性能对比')
axes[0, 2].set_ylabel('MSE')
axes[0, 2].tick_params(axis='x', rotation=45)

# 11.4 分类模型特征重要性
if rf_cls_importance is not None:
    top_features_cls = rf_cls_importance.head(10)
    axes[1, 0].barh(range(len(top_features_cls)), top_features_cls['importance'])
    axes[1, 0].set_yticks(range(len(top_features_cls)))
    axes[1, 0].set_yticklabels(top_features_cls['feature'])
    axes[1, 0].set_title('随机森林分类 - 特征重要性')
    axes[1, 0].set_xlabel('重要性')

# 11.5 分类模型性能对比
cls_model_names = list(classification_results.keys())
cls_accuracies = [classification_results[name]['accuracy'] for name in cls_model_names]

axes[1, 1].bar(cls_model_names, cls_accuracies, alpha=0.7, color='orange')
axes[1, 1].set_title('分类模型性能对比')
axes[1, 1].set_ylabel('准确率')
axes[1, 1].tick_params(axis='x', rotation=45)

# 11.6 解释性 vs 性能权衡
interpretability_scores = {'线性回归': 4, '随机森林': 3, '决策树': 5}  # 主观评分
performance_scores = {
    '线性回归': 1 - regression_results['Linear Regression']['mse'] / max(model_mses),
    '随机森林': 1 - regression_results['Random Forest']['mse'] / max(model_mses),
    '决策树': classification_results['Decision Tree']['accuracy']
}

models_for_plot = ['线性回归', '随机森林']
interp_vals = [interpretability_scores[m] for m in models_for_plot]
perf_vals = [performance_scores[m] for m in models_for_plot]

axes[1, 2].scatter(interp_vals, perf_vals, s=100, alpha=0.7)
for i, model in enumerate(models_for_plot):
    axes[1, 2].annotate(model, (interp_vals[i], perf_vals[i]), 
                       xytext=(5, 5), textcoords='offset points')
axes[1, 2].set_xlabel('解释性得分')
axes[1, 2].set_ylabel('性能得分')
axes[1, 2].set_title('解释性 vs 性能权衡')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/interpretability_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 12. 实践建议
print("\n12. 实践建议")

recommendations = {
    '场景': ['金融风控', '医疗诊断', '营销推荐', '工业监控'],
    '推荐模型': ['逻辑回归/决策树', '决策树/随机森林', '随机森林/梯度提升', '线性模型/决策树'],
    '解释方法': ['系数分析+LIME', '决策路径+SHAP', 'SHAP+特征重要性', '系数分析+部分依赖'],
    '重点关注': ['合规性', '可信度', '个性化', '稳定性']
}

recommendations_df = pd.DataFrame(recommendations)
print("不同场景的模型解释建议:")
print(recommendations_df.to_string(index=False))

print("\n=== 模型解释性总结 ===")
print("✅ 理解模型解释性的重要性和分类")
print("✅ 掌握特征重要性分析方法")
print("✅ 学会使用排列重要性评估")
print("✅ 应用SHAP进行全局和局部解释")
print("✅ 使用LIME进行实例级解释")
print("✅ 可视化决策树和部分依赖")
print("✅ 比较不同模型的解释性")

print("\n关键技术:")
print("1. 特征重要性：内在重要性 vs 排列重要性")
print("2. SHAP值：基于博弈论的统一解释框架")
print("3. LIME：局部线性近似解释")
print("4. 部分依赖：理解特征与预测的关系")
print("5. 决策树可视化：直观的决策过程")

print("\n最佳实践:")
print("1. 根据应用场景选择合适的解释方法")
print("2. 结合全局和局部解释获得完整理解")
print("3. 验证解释的稳定性和一致性")
print("4. 在模型开发阶段就考虑解释性")
print("5. 定期检查模型解释的有效性")

print("\n=== 练习任务 ===")
print("1. 实现自定义的特征重要性方法")
print("2. 比较SHAP和LIME在不同数据上的表现")
print("3. 研究深度学习模型的解释方法")
print("4. 实现模型解释的自动化报告")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 研究因果解释 vs 关联解释")
print("2. 实现对抗性样本的解释分析")
print("3. 构建可解释AI的评估框架")
print("4. 研究多模态数据的解释方法")
print("5. 实现交互式模型解释界面")