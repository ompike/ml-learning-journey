"""
数据处理工具函数
提供常用的数据处理、可视化和评估功能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.model_selection import learning_curve, validation_curve
import warnings
warnings.filterwarnings('ignore')

def load_sample_datasets():
    """加载示例数据集"""
    datasets = {}
    
    # 房价数据集
    np.random.seed(42)
    n_samples = 1000
    
    # 生成特征
    bedrooms = np.random.randint(1, 6, n_samples)
    bathrooms = np.random.randint(1, 4, n_samples) 
    sqft = np.random.normal(2000, 500, n_samples)
    age = np.random.randint(0, 50, n_samples)
    location_score = np.random.uniform(1, 10, n_samples)
    
    # 生成目标变量（房价）
    price = (bedrooms * 50000 + 
             bathrooms * 30000 + 
             sqft * 150 + 
             (50 - age) * 1000 + 
             location_score * 20000 + 
             np.random.normal(0, 20000, n_samples))
    
    house_data = pd.DataFrame({
        'bedrooms': bedrooms,
        'bathrooms': bathrooms,
        'sqft': sqft,
        'age': age,
        'location_score': location_score,
        'price': price
    })
    
    datasets['house_prices'] = house_data
    
    # 客户流失数据集
    n_customers = 2000
    tenure = np.random.randint(1, 73, n_customers)
    monthly_charges = np.random.uniform(20, 120, n_customers)
    total_charges = tenure * monthly_charges + np.random.normal(0, 100, n_customers)
    contract_type = np.random.choice(['Month-to-month', 'One year', 'Two year'], n_customers)
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n_customers)
    
    # 流失概率计算
    churn_prob = (0.3 * (tenure < 12) + 
                  0.2 * (monthly_charges > 80) + 
                  0.1 * (contract_type == 'Month-to-month') +
                  0.1 * (internet_service == 'Fiber optic') +
                  np.random.uniform(0, 0.3, n_customers))
    
    churn = (churn_prob > 0.5).astype(int)
    
    customer_data = pd.DataFrame({
        'tenure': tenure,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'contract_type': contract_type,
        'internet_service': internet_service,
        'churn': churn
    })
    
    datasets['customer_churn'] = customer_data
    
    return datasets

def plot_data_distribution(data, target_column=None, figsize=(15, 10)):
    """绘制数据分布图"""
    numeric_columns = data.select_dtypes(include=[np.number]).columns
    n_cols = 3
    n_rows = (len(numeric_columns) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_rows > 1 else [axes]
    
    for i, col in enumerate(numeric_columns):
        ax = axes[i] if len(axes) > 1 else axes
        
        if target_column and col != target_column:
            if data[target_column].dtype == 'object' or data[target_column].nunique() < 10:
                # 分类目标变量
                for target_val in data[target_column].unique():
                    subset = data[data[target_column] == target_val][col]
                    ax.hist(subset, alpha=0.7, label=f'{target_column}={target_val}', bins=20)
                ax.legend()
            else:
                # 连续目标变量
                ax.scatter(data[col], data[target_column], alpha=0.6)
                ax.set_ylabel(target_column)
        else:
            ax.hist(data[col], bins=20, alpha=0.7)
        
        ax.set_title(f'{col} Distribution')
        ax.set_xlabel(col)
        ax.grid(True, alpha=0.3)
    
    # 隐藏多余的子图
    for i in range(len(numeric_columns), len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig

def plot_correlation_matrix(data, figsize=(10, 8)):
    """绘制相关性矩阵"""
    numeric_data = data.select_dtypes(include=[np.number])
    correlation_matrix = numeric_data.corr()
    
    plt.figure(figsize=figsize)
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', 
                center=0, square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    return plt.gcf()

def evaluate_classification_model(y_true, y_pred, y_pred_proba=None, class_names=None):
    """评估分类模型"""
    print("=== 分类模型评估结果 ===")
    
    # 准确率
    accuracy = np.mean(y_true == y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n详细分类报告:")
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    print(f"\n混淆矩阵:\n{cm}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2 if y_pred_proba is not None else 1, figsize=(12, 5))
    if y_pred_proba is not None:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
    else:
        axes = [axes]
    
    # 混淆矩阵热力图
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title('Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    # ROC曲线（二分类）
    if y_pred_proba is not None and len(np.unique(y_true)) == 2:
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba[:, 1] if y_pred_proba.ndim > 1 else y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                    label=f'ROC Curve (AUC = {roc_auc:.2f})')
        axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend(loc="lower right")
        axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def evaluate_regression_model(y_true, y_pred):
    """评估回归模型"""
    print("=== 回归模型评估结果 ===")
    
    # 计算指标
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - (np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2))
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"R² 分数: {r2:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 预测 vs 真实值
    axes[0].scatter(y_true, y_pred, alpha=0.6)
    axes[0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    axes[0].set_xlabel('True Values')
    axes[0].set_ylabel('Predicted Values')
    axes[0].set_title('Predicted vs True Values')
    axes[0].grid(True, alpha=0.3)
    
    # 残差图
    residuals = y_true - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.6)
    axes[1].axhline(y=0, color='r', linestyle='--')
    axes[1].set_xlabel('Predicted Values')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residual Plot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_learning_curves_detailed(estimator, X, y, cv=5, n_jobs=-1, 
                                train_sizes=np.linspace(0.1, 1.0, 10)):
    """绘制详细的学习曲线"""
    train_sizes, train_scores, val_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(train_sizes, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel('Training Set Size')
    plt.ylabel('Score')
    plt.title('Learning Curves')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt.gcf()

def plot_validation_curves_detailed(estimator, X, y, param_name, param_range, 
                                  cv=5, scoring='accuracy'):
    """绘制验证曲线"""
    train_scores, val_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range, 
        cv=cv, scoring=scoring)
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    val_mean = np.mean(val_scores, axis=1)
    val_std = np.std(val_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(param_range, train_mean, 'o-', color='blue', label='Training Score')
    plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, 
                     alpha=0.1, color='blue')
    
    plt.plot(param_range, val_mean, 'o-', color='red', label='Validation Score')
    plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, 
                     alpha=0.1, color='red')
    
    plt.xlabel(param_name)
    plt.ylabel('Score')
    plt.title(f'Validation Curves for {param_name}')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    if len(param_range) > 5:
        plt.xticks(param_range[::len(param_range)//5])
    
    plt.tight_layout()
    return plt.gcf()

def create_feature_importance_plot(feature_names, importances, title="Feature Importance"):
    """创建特征重要性图"""
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=45)
    plt.title(title)
    plt.ylabel('Importance')
    plt.tight_layout()
    return plt.gcf()

def compare_models(models_dict, X_train, X_test, y_train, y_test, task_type='classification'):
    """比较多个模型的性能"""
    results = {}
    
    for name, model in models_dict.items():
        # 训练模型
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        if task_type == 'classification':
            score = np.mean(y_test == y_pred)
            results[name] = {'accuracy': score, 'model': model}
        else:  # regression
            mse = np.mean((y_test - y_pred) ** 2)
            rmse = np.sqrt(mse)
            r2 = 1 - (np.sum((y_test - y_pred) ** 2) / np.sum((y_test - np.mean(y_test)) ** 2))
            results[name] = {'mse': mse, 'rmse': rmse, 'r2': r2, 'model': model}
    
    # 可视化比较结果
    if task_type == 'classification':
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        plt.figure(figsize=(10, 6))
        plt.bar(models, accuracies)
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.xticks(rotation=45)
        plt.tight_layout()
    else:
        models = list(results.keys())
        rmses = [results[model]['rmse'] for model in models]
        r2s = [results[model]['r2'] for model in models]
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        axes[0].bar(models, rmses)
        axes[0].set_title('Model RMSE Comparison')
        axes[0].set_ylabel('RMSE')
        axes[0].tick_params(axis='x', rotation=45)
        
        axes[1].bar(models, r2s)
        axes[1].set_title('Model R² Comparison')
        axes[1].set_ylabel('R²')
        axes[1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
    
    return results, plt.gcf()

# 示例用法
if __name__ == "__main__":
    print("数据工具函数测试")
    
    # 加载示例数据
    datasets = load_sample_datasets()
    
    print("可用数据集:")
    for name, data in datasets.items():
        print(f"- {name}: {data.shape}")
        print(f"  列名: {list(data.columns)}")
        print()
    
    # 测试可视化函数
    house_data = datasets['house_prices']
    fig1 = plot_data_distribution(house_data, target_column='price')
    fig1.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/utils/sample_data_distribution.png', 
                 dpi=300, bbox_inches='tight')
    plt.close()
    
    fig2 = plot_correlation_matrix(house_data)
    fig2.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/utils/sample_correlation_matrix.png', 
                 dpi=300, bbox_inches='tight')
    plt.close()
    
    print("数据工具函数测试完成！")