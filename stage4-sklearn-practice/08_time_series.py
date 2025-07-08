"""
时间序列分析
学习目标：掌握时间序列数据的处理、分析和预测方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

print("=== 时间序列分析 ===\n")

# 1. 时间序列理论基础
print("1. 时间序列理论基础")
print("时间序列的特点：")
print("- 时间依赖性：数据点按时间顺序排列")
print("- 趋势性：长期增长或下降的模式")
print("- 季节性：周期性重复的模式")
print("- 随机性：不可预测的波动")

print("\n时间序列分析方法：")
print("1. 经典方法：ARIMA、指数平滑")
print("2. 机器学习：回归、决策树、神经网络")
print("3. 深度学习：LSTM、GRU、Transformer")

# 2. 生成模拟时间序列数据
print("\n2. 生成模拟时间序列数据")

def generate_time_series(n_points=1000, trend=0.02, seasonal_amplitude=10, 
                        seasonal_period=50, noise_std=2):
    """生成包含趋势、季节性和噪声的时间序列"""
    np.random.seed(42)
    
    # 时间轴
    t = np.arange(n_points)
    
    # 趋势分量
    trend_component = trend * t
    
    # 季节性分量
    seasonal_component = seasonal_amplitude * np.sin(2 * np.pi * t / seasonal_period)
    
    # 噪声分量
    noise_component = np.random.normal(0, noise_std, n_points)
    
    # 组合所有分量
    time_series = 100 + trend_component + seasonal_component + noise_component
    
    return time_series, trend_component, seasonal_component, noise_component

# 生成主要时间序列
ts_data, trend, seasonal, noise = generate_time_series()

# 创建日期索引
dates = pd.date_range(start='2020-01-01', periods=len(ts_data), freq='D')
ts_df = pd.DataFrame({
    'date': dates,
    'value': ts_data,
    'trend': trend,
    'seasonal': seasonal,
    'noise': noise
})

print(f"生成时间序列数据点数: {len(ts_data)}")
print(f"数据范围: {dates[0]} 到 {dates[-1]}")

# 3. 时间序列数据探索
print("\n3. 时间序列数据探索")

# 基本统计信息
print("基本统计信息:")
print(ts_df['value'].describe())

# 趋势分析
def detect_trend(data, window=30):
    """检测趋势"""
    # 计算移动平均
    moving_avg = data.rolling(window=window).mean()
    
    # 计算趋势斜率
    trend_slope = np.polyfit(range(len(moving_avg.dropna())), 
                           moving_avg.dropna(), 1)[0]
    
    return moving_avg, trend_slope

moving_avg, trend_slope = detect_trend(ts_df['value'])
print(f"\n趋势斜率: {trend_slope:.6f}")

if trend_slope > 0.01:
    print("检测到上升趋势")
elif trend_slope < -0.01:
    print("检测到下降趋势")
else:
    print("趋势不明显")

# 季节性分析
def detect_seasonality(data, periods_to_test=[7, 30, 50, 365]):
    """检测季节性"""
    correlations = {}
    
    for period in periods_to_test:
        if len(data) > period:
            # 计算滞后相关性
            lag_corr = data.autocorr(lag=period)
            correlations[period] = lag_corr
    
    return correlations

seasonality_corr = detect_seasonality(ts_df['value'])
print("\n季节性分析（滞后相关性）:")
for period, corr in seasonality_corr.items():
    print(f"  周期 {period}: {corr:.4f}")

# 4. 时间序列分解
print("\n4. 时间序列分解")

def decompose_time_series(data, window=50):
    """简单的时间序列分解"""
    # 趋势（移动平均）
    trend = data.rolling(window=window, center=True).mean()
    
    # 去趋势
    detrended = data - trend
    
    # 季节性（周期性移动平均）
    seasonal = detrended.rolling(window=window, center=True).mean()
    
    # 残差
    residual = data - trend - seasonal
    
    return trend, seasonal, residual

decomp_trend, decomp_seasonal, decomp_residual = decompose_time_series(ts_df['value'])

ts_df['decomp_trend'] = decomp_trend
ts_df['decomp_seasonal'] = decomp_seasonal
ts_df['decomp_residual'] = decomp_residual

print("时间序列分解完成")
print(f"趋势分量方差: {decomp_trend.var():.4f}")
print(f"季节性分量方差: {decomp_seasonal.var():.4f}")
print(f"残差方差: {decomp_residual.var():.4f}")

# 5. 特征工程
print("\n5. 特征工程")

def create_time_features(df, date_col='date', value_col='value'):
    """创建时间相关特征"""
    df = df.copy()
    
    # 时间特征
    df['year'] = df[date_col].dt.year
    df['month'] = df[date_col].dt.month
    df['day'] = df[date_col].dt.day
    df['day_of_week'] = df[date_col].dt.dayofweek
    df['day_of_year'] = df[date_col].dt.dayofyear
    df['week_of_year'] = df[date_col].dt.isocalendar().week
    
    # 周期性特征（正弦余弦编码）
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    return df

def create_lag_features(df, value_col='value', lags=[1, 2, 3, 7, 14, 30]):
    """创建滞后特征"""
    df = df.copy()
    
    for lag in lags:
        df[f'{value_col}_lag_{lag}'] = df[value_col].shift(lag)
    
    return df

def create_rolling_features(df, value_col='value', windows=[3, 7, 14, 30]):
    """创建滚动窗口特征"""
    df = df.copy()
    
    for window in windows:
        df[f'{value_col}_mean_{window}'] = df[value_col].rolling(window=window).mean()
        df[f'{value_col}_std_{window}'] = df[value_col].rolling(window=window).std()
        df[f'{value_col}_min_{window}'] = df[value_col].rolling(window=window).min()
        df[f'{value_col}_max_{window}'] = df[value_col].rolling(window=window).max()
    
    return df

# 应用特征工程
ts_df = create_time_features(ts_df)
ts_df = create_lag_features(ts_df)
ts_df = create_rolling_features(ts_df)

print(f"特征工程后的特征数量: {ts_df.shape[1]}")
print("新增特征示例:")
new_features = [col for col in ts_df.columns if col not in ['date', 'value', 'trend', 'seasonal', 'noise']]
print(new_features[:10])

# 6. 时间序列预测模型
print("\n6. 时间序列预测模型")

# 准备建模数据
def prepare_modeling_data(df, target_col='value', test_size=0.2):
    """准备建模数据"""
    # 移除含有NaN的行
    df_clean = df.dropna()
    
    # 分离特征和目标变量
    feature_cols = [col for col in df_clean.columns 
                   if col not in ['date', target_col, 'trend', 'seasonal', 'noise', 
                                 'decomp_trend', 'decomp_seasonal', 'decomp_residual']]
    
    X = df_clean[feature_cols]
    y = df_clean[target_col]
    
    # 时间序列分割（保持时间顺序）
    split_idx = int(len(df_clean) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    return X_train, X_test, y_train, y_test, feature_cols

X_train, X_test, y_train, y_test, feature_cols = prepare_modeling_data(ts_df)

print(f"训练集大小: {X_train.shape}")
print(f"测试集大小: {X_test.shape}")
print(f"特征数量: {len(feature_cols)}")

# 7. 模型训练和评估
print("\n7. 模型训练和评估")

# 定义模型
models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42)
}

# 训练和评估模型
model_results = {}

for name, model in models.items():
    print(f"\n训练 {name}...")
    
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 评估指标
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    train_mae = mean_absolute_error(y_train, y_train_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # MAPE (Mean Absolute Percentage Error)
    train_mape = np.mean(np.abs((y_train - y_train_pred) / y_train)) * 100
    test_mape = np.mean(np.abs((y_test - y_test_pred) / y_test)) * 100
    
    model_results[name] = {
        'model': model,
        'train_mse': train_mse,
        'test_mse': test_mse,
        'train_mae': train_mae,
        'test_mae': test_mae,
        'train_mape': train_mape,
        'test_mape': test_mape,
        'y_test_pred': y_test_pred
    }
    
    print(f"  训练MSE: {train_mse:.4f}")
    print(f"  测试MSE: {test_mse:.4f}")
    print(f"  测试MAE: {test_mae:.4f}")
    print(f"  测试MAPE: {test_mape:.4f}%")

# 8. 时间序列交叉验证
print("\n8. 时间序列交叉验证")

def time_series_cv(X, y, model, n_splits=5):
    """时间序列交叉验证"""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    
    cv_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 训练模型
        model.fit(X_fold_train, y_fold_train)
        
        # 预测
        y_fold_pred = model.predict(X_fold_val)
        
        # 计算MSE
        fold_mse = mean_squared_error(y_fold_val, y_fold_pred)
        cv_scores.append(fold_mse)
        
        print(f"  Fold {fold + 1}: MSE = {fold_mse:.4f}")
    
    return cv_scores

# 对随机森林进行时间序列交叉验证
print("随机森林时间序列交叉验证:")
rf_cv_scores = time_series_cv(X_train, y_train, 
                             RandomForestRegressor(n_estimators=100, random_state=42))

print(f"平均CV MSE: {np.mean(rf_cv_scores):.4f} (+/- {np.std(rf_cv_scores) * 2:.4f})")

# 9. 特征重要性分析
print("\n9. 特征重要性分析")

# 获取随机森林的特征重要性
rf_model = model_results['Random Forest']['model']
feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Top 15 重要特征:")
print(feature_importance.head(15))

# 10. 简单的ARIMA风格模型
print("\n10. 简单的ARIMA风格模型")

class SimpleARModel:
    """简单的自回归模型"""
    
    def __init__(self, p=5):
        self.p = p  # 自回归阶数
        self.coefficients = None
        self.intercept = None
    
    def fit(self, y):
        """训练模型"""
        # 创建滞后特征矩阵
        X_ar = np.zeros((len(y) - self.p, self.p))
        y_ar = y[self.p:]
        
        for i in range(self.p):
            X_ar[:, i] = y[self.p - 1 - i : len(y) - 1 - i]
        
        # 线性回归
        X_ar_with_intercept = np.column_stack([np.ones(X_ar.shape[0]), X_ar])
        
        # 最小二乘法求解
        coeffs = np.linalg.lstsq(X_ar_with_intercept, y_ar, rcond=None)[0]
        self.intercept = coeffs[0]
        self.coefficients = coeffs[1:]
        
        self.last_values = y[-self.p:]
    
    def predict(self, steps=1):
        """预测未来值"""
        predictions = []
        current_values = self.last_values.copy()
        
        for _ in range(steps):
            # 预测下一个值
            next_pred = self.intercept + np.dot(self.coefficients, current_values[::-1])
            predictions.append(next_pred)
            
            # 更新滞后值
            current_values = np.append(current_values[1:], next_pred)
        
        return np.array(predictions)

# 训练简单AR模型
ar_model = SimpleARModel(p=7)
ar_model.fit(y_train.values)

# 预测
ar_predictions = ar_model.predict(steps=len(y_test))
ar_mse = mean_squared_error(y_test, ar_predictions)

print(f"AR模型系数: {ar_model.coefficients}")
print(f"AR模型截距: {ar_model.intercept:.4f}")
print(f"AR模型测试MSE: {ar_mse:.4f}")

# 11. 残差分析
print("\n11. 残差分析")

def analyze_residuals(y_true, y_pred, model_name):
    """分析残差"""
    residuals = y_true - y_pred
    
    print(f"\n{model_name} 残差分析:")
    print(f"  残差均值: {np.mean(residuals):.6f}")
    print(f"  残差标准差: {np.std(residuals):.4f}")
    print(f"  残差偏度: {pd.Series(residuals).skew():.4f}")
    print(f"  残差峰度: {pd.Series(residuals).kurtosis():.4f}")
    
    # Ljung-Box测试（简化版本）
    # 计算残差的自相关
    autocorr_1 = pd.Series(residuals).autocorr(lag=1)
    print(f"  残差滞后1自相关: {autocorr_1:.4f}")
    
    return residuals

# 分析最佳模型的残差
best_model_name = min(model_results.keys(), key=lambda x: model_results[x]['test_mse'])
best_model_residuals = analyze_residuals(
    y_test, 
    model_results[best_model_name]['y_test_pred'], 
    best_model_name
)

# 12. 可视化分析
print("\n12. 可视化分析")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 12.1 原始时间序列
axes[0, 0].plot(ts_df['date'], ts_df['value'], alpha=0.7)
axes[0, 0].plot(ts_df['date'], moving_avg, color='red', linewidth=2, label='移动平均')
axes[0, 0].set_title('原始时间序列')
axes[0, 0].set_xlabel('日期')
axes[0, 0].set_ylabel('值')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 12.2 时间序列分解
axes[0, 1].plot(ts_df['date'], ts_df['trend'], label='真实趋势', alpha=0.7)
axes[0, 1].plot(ts_df['date'], ts_df['seasonal'], label='真实季节性', alpha=0.7)
axes[0, 1].set_title('时间序列分量')
axes[0, 1].set_xlabel('日期')
axes[0, 1].set_ylabel('值')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 12.3 自相关图
max_lags = 30
autocorr_values = [ts_df['value'].autocorr(lag=i) for i in range(1, max_lags + 1)]
axes[0, 2].bar(range(1, max_lags + 1), autocorr_values, alpha=0.7)
axes[0, 2].axhline(y=0, color='black', linestyle='-', alpha=0.5)
axes[0, 2].set_title('自相关函数')
axes[0, 2].set_xlabel('滞后期')
axes[0, 2].set_ylabel('自相关系数')
axes[0, 2].grid(True, alpha=0.3)

# 12.4 预测结果对比
test_dates = ts_df['date'].iloc[-len(y_test):]
axes[1, 0].plot(test_dates, y_test.values, label='真实值', alpha=0.8)
for name, result in model_results.items():
    axes[1, 0].plot(test_dates, result['y_test_pred'], 
                   label=f'{name}预测', alpha=0.8)
axes[1, 0].plot(test_dates, ar_predictions, label='AR模型预测', alpha=0.8)
axes[1, 0].set_title('预测结果对比')
axes[1, 0].set_xlabel('日期')
axes[1, 0].set_ylabel('值')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 12.5 模型性能对比
model_names = list(model_results.keys()) + ['AR Model']
test_mses = [model_results[name]['test_mse'] for name in model_results.keys()] + [ar_mse]

axes[1, 1].bar(model_names, test_mses, alpha=0.7)
axes[1, 1].set_title('模型性能对比 (MSE)')
axes[1, 1].set_ylabel('MSE')
axes[1, 1].tick_params(axis='x', rotation=45)

# 12.6 特征重要性
top_features = feature_importance.head(15)
axes[1, 2].barh(range(len(top_features)), top_features['importance'])
axes[1, 2].set_yticks(range(len(top_features)))
axes[1, 2].set_yticklabels(top_features['feature'])
axes[1, 2].set_title('特征重要性 Top 15')
axes[1, 2].set_xlabel('重要性')

# 12.7 残差分析
axes[2, 0].scatter(model_results[best_model_name]['y_test_pred'], best_model_residuals, alpha=0.6)
axes[2, 0].axhline(y=0, color='red', linestyle='--')
axes[2, 0].set_xlabel('预测值')
axes[2, 0].set_ylabel('残差')
axes[2, 0].set_title(f'{best_model_name} 残差散点图')
axes[2, 0].grid(True, alpha=0.3)

# 12.8 残差分布
axes[2, 1].hist(best_model_residuals, bins=30, alpha=0.7, edgecolor='black')
axes[2, 1].set_xlabel('残差')
axes[2, 1].set_ylabel('频数')
axes[2, 1].set_title('残差分布')
axes[2, 1].grid(True, alpha=0.3)

# 12.9 滚动预测性能
window_size = 30
rolling_mse = []
for i in range(window_size, len(y_test)):
    window_true = y_test.iloc[i-window_size:i]
    window_pred = model_results[best_model_name]['y_test_pred'][i-window_size:i]
    window_mse = mean_squared_error(window_true, window_pred)
    rolling_mse.append(window_mse)

axes[2, 2].plot(rolling_mse)
axes[2, 2].set_title(f'滚动窗口MSE (窗口大小={window_size})')
axes[2, 2].set_xlabel('时间步')
axes[2, 2].set_ylabel('MSE')
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/time_series_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 时间序列分析总结 ===")
print("✅ 理解时间序列的基本概念和特征")
print("✅ 掌握时间序列数据的探索和分解")
print("✅ 学会创建时间相关特征")
print("✅ 实现多种预测模型")
print("✅ 掌握时间序列交叉验证")
print("✅ 进行残差分析和模型诊断")
print("✅ 实现简单的ARIMA风格模型")

print("\n关键技术:")
print("1. 特征工程：滞后特征、滚动窗口特征、周期性特征")
print("2. 模型选择：考虑时间依赖性的验证方法")
print("3. 评估指标：MSE、MAE、MAPE等")
print("4. 残差分析：检验模型假设和预测质量")
print("5. 时间序列分解：趋势、季节性、残差")

print("\n实际应用:")
print("1. 股票价格预测")
print("2. 销售预测")
print("3. 能源需求预测")
print("4. 网站流量预测")
print("5. 设备故障预测")

print("\n=== 练习任务 ===")
print("1. 实现更复杂的ARIMA模型")
print("2. 尝试指数平滑方法")
print("3. 研究多变量时间序列")
print("4. 实现时间序列聚类")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现Facebook Prophet模型")
print("2. 研究深度学习时间序列方法")
print("3. 实现在线时间序列学习")
print("4. 研究时间序列异常检测")
print("5. 实现多尺度时间序列分析")