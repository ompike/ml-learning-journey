"""
统计学基础实现
学习目标：掌握描述性统计、假设检验、置信区间、回归分析等统计学核心概念
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import t, chi2, f
import pandas as pd

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

print("=== 统计学基础实现 ===\n")

# 1. 描述性统计
print("1. 描述性统计")

def descriptive_statistics(data):
    """计算描述性统计量"""
    stats_dict = {
        '均值': np.mean(data),
        '中位数': np.median(data),
        '众数': stats.mode(data).mode[0] if len(stats.mode(data).mode) > 0 else np.nan,
        '标准差': np.std(data, ddof=1),  # 样本标准差
        '方差': np.var(data, ddof=1),    # 样本方差
        '偏度': stats.skew(data),        # 偏度
        '峰度': stats.kurtosis(data),    # 峰度
        '最小值': np.min(data),
        '最大值': np.max(data),
        '范围': np.max(data) - np.min(data),
        '四分位数间距': np.percentile(data, 75) - np.percentile(data, 25)
    }
    return stats_dict

# 生成示例数据
np.random.seed(42)
normal_data = np.random.normal(100, 15, 1000)
skewed_data = np.random.exponential(2, 1000)

print("正态分布数据的描述性统计：")
normal_stats = descriptive_statistics(normal_data)
for key, value in normal_stats.items():
    print(f"{key}: {value:.3f}")

print("\n偏态分布数据的描述性统计：")
skewed_stats = descriptive_statistics(skewed_data)
for key, value in skewed_stats.items():
    print(f"{key}: {value:.3f}")

# 2. 概率分布拟合
print("\n2. 概率分布拟合")

def fit_distribution(data, distribution='normal'):
    """拟合概率分布"""
    if distribution == 'normal':
        mu, sigma = stats.norm.fit(data)
        return mu, sigma
    elif distribution == 'exponential':
        loc, scale = stats.expon.fit(data)
        return loc, scale
    elif distribution == 'uniform':
        loc, scale = stats.uniform.fit(data)
        return loc, scale

# 拟合正态分布
mu_hat, sigma_hat = fit_distribution(normal_data, 'normal')
print(f"拟合正态分布参数: μ = {mu_hat:.3f}, σ = {sigma_hat:.3f}")

# 拟合指数分布
loc_hat, scale_hat = fit_distribution(skewed_data, 'exponential')
print(f"拟合指数分布参数: loc = {loc_hat:.3f}, scale = {scale_hat:.3f}")

# 3. 置信区间
print("\n3. 置信区间")

def confidence_interval_mean(data, confidence=0.95):
    """计算均值的置信区间"""
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)  # 标准误
    
    # 使用t分布
    alpha = 1 - confidence
    t_critical = stats.t.ppf(1 - alpha/2, df=n-1)
    
    margin_of_error = t_critical * std_err
    ci_lower = mean - margin_of_error
    ci_upper = mean + margin_of_error
    
    return ci_lower, ci_upper, mean

def confidence_interval_proportion(x, n, confidence=0.95):
    """计算比例的置信区间"""
    p_hat = x / n
    alpha = 1 - confidence
    z_critical = stats.norm.ppf(1 - alpha/2)
    
    std_err = np.sqrt(p_hat * (1 - p_hat) / n)
    margin_of_error = z_critical * std_err
    
    ci_lower = p_hat - margin_of_error
    ci_upper = p_hat + margin_of_error
    
    return ci_lower, ci_upper, p_hat

# 均值置信区间示例
ci_lower, ci_upper, sample_mean = confidence_interval_mean(normal_data)
print(f"样本均值: {sample_mean:.3f}")
print(f"95%置信区间: [{ci_lower:.3f}, {ci_upper:.3f}]")

# 比例置信区间示例
successes = 85
trials = 100
ci_lower_p, ci_upper_p, p_hat = confidence_interval_proportion(successes, trials)
print(f"样本比例: {p_hat:.3f}")
print(f"95%置信区间: [{ci_lower_p:.3f}, {ci_upper_p:.3f}]")

# 4. 假设检验
print("\n4. 假设检验")

# 4.1 单样本t检验
def one_sample_t_test(data, population_mean, alpha=0.05):
    """单样本t检验"""
    n = len(data)
    sample_mean = np.mean(data)
    sample_std = np.std(data, ddof=1)
    
    # 计算t统计量
    t_statistic = (sample_mean - population_mean) / (sample_std / np.sqrt(n))
    
    # 计算p值（双尾检验）
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_statistic), df))
    
    # 判断是否拒绝原假设
    reject_null = p_value < alpha
    
    return t_statistic, p_value, reject_null

# 4.2 两样本t检验
def two_sample_t_test(data1, data2, equal_var=True, alpha=0.05):
    """两样本t检验"""
    if equal_var:
        t_stat, p_val = stats.ttest_ind(data1, data2)
    else:
        t_stat, p_val = stats.ttest_ind(data1, data2, equal_var=False)
    
    reject_null = p_val < alpha
    return t_stat, p_val, reject_null

# 4.3 配对t检验
def paired_t_test(before, after, alpha=0.05):
    """配对t检验"""
    t_stat, p_val = stats.ttest_rel(before, after)
    reject_null = p_val < alpha
    return t_stat, p_val, reject_null

# 假设检验示例
print("4.1 单样本t检验")
t_stat, p_val, reject = one_sample_t_test(normal_data, 100)
print(f"H0: μ = 100 vs H1: μ ≠ 100")
print(f"t统计量: {t_stat:.3f}, p值: {p_val:.6f}")
print(f"结论: {'拒绝' if reject else '接受'}原假设")

print("\n4.2 两样本t检验")
group1 = np.random.normal(100, 15, 50)
group2 = np.random.normal(105, 15, 50)
t_stat2, p_val2, reject2 = two_sample_t_test(group1, group2)
print(f"H0: μ1 = μ2 vs H1: μ1 ≠ μ2")
print(f"t统计量: {t_stat2:.3f}, p值: {p_val2:.6f}")
print(f"结论: {'拒绝' if reject2 else '接受'}原假设")

# 5. 方差分析(ANOVA)
print("\n5. 方差分析(ANOVA)")

def one_way_anova(*groups):
    """单因素方差分析"""
    f_stat, p_val = stats.f_oneway(*groups)
    return f_stat, p_val

# 生成三组数据
group_a = np.random.normal(100, 10, 30)
group_b = np.random.normal(105, 10, 30)
group_c = np.random.normal(102, 10, 30)

f_stat, p_val = one_way_anova(group_a, group_b, group_c)
print(f"F统计量: {f_stat:.3f}, p值: {p_val:.6f}")
print(f"结论: {'拒绝' if p_val < 0.05 else '接受'}原假设（各组均值相等）")

# 6. 卡方检验
print("\n6. 卡方检验")

def chi_square_goodness_of_fit(observed, expected):
    """卡方拟合优度检验"""
    chi2_stat = np.sum((observed - expected)**2 / expected)
    df = len(observed) - 1
    p_val = 1 - stats.chi2.cdf(chi2_stat, df)
    return chi2_stat, p_val

def chi_square_independence(contingency_table):
    """卡方独立性检验"""
    chi2_stat, p_val, dof, expected = stats.chi2_contingency(contingency_table)
    return chi2_stat, p_val, expected

# 拟合优度检验示例
print("6.1 拟合优度检验（掷骰子）")
observed_dice = [18, 22, 16, 25, 12, 7]  # 观察频数
expected_dice = [100/6] * 6               # 期望频数
chi2_stat, p_val = chi_square_goodness_of_fit(observed_dice, expected_dice)
print(f"卡方统计量: {chi2_stat:.3f}, p值: {p_val:.6f}")

# 独立性检验示例
print("\n6.2 独立性检验")
contingency = np.array([[10, 20, 30],
                       [20, 25, 25]])
chi2_stat, p_val, expected = chi_square_independence(contingency)
print(f"卡方统计量: {chi2_stat:.3f}, p值: {p_val:.6f}")
print("期望频数：")
print(expected)

# 7. 相关性分析
print("\n7. 相关性分析")

def correlation_analysis(x, y):
    """相关性分析"""
    # 皮尔逊相关系数
    pearson_r, pearson_p = stats.pearsonr(x, y)
    
    # 斯皮尔曼等级相关系数
    spearman_r, spearman_p = stats.spearmanr(x, y)
    
    # 肯德尔τ相关系数
    kendall_tau, kendall_p = stats.kendalltau(x, y)
    
    return {
        'pearson': (pearson_r, pearson_p),
        'spearman': (spearman_r, spearman_p),
        'kendall': (kendall_tau, kendall_p)
    }

# 生成相关数据
x = np.random.normal(0, 1, 100)
y_linear = 2 * x + np.random.normal(0, 0.5, 100)  # 线性关系
y_nonlinear = x**2 + np.random.normal(0, 0.5, 100)  # 非线性关系

print("线性关系数据：")
linear_corr = correlation_analysis(x, y_linear)
for method, (r, p) in linear_corr.items():
    print(f"{method}相关系数: r = {r:.3f}, p = {p:.6f}")

print("\n非线性关系数据：")
nonlinear_corr = correlation_analysis(x, y_nonlinear)
for method, (r, p) in nonlinear_corr.items():
    print(f"{method}相关系数: r = {r:.3f}, p = {p:.6f}")

# 8. 回归分析
print("\n8. 回归分析")

def simple_linear_regression(x, y):
    """简单线性回归"""
    n = len(x)
    x_mean, y_mean = np.mean(x), np.mean(y)
    
    # 计算回归系数
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sum((x - x_mean)**2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    
    # 预测值和残差
    y_pred = intercept + slope * x
    residuals = y - y_pred
    
    # R方（决定系数）
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - y_mean)**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # 标准误
    mse = ss_res / (n - 2)
    se_slope = np.sqrt(mse / np.sum((x - x_mean)**2))
    se_intercept = np.sqrt(mse * (1/n + x_mean**2/np.sum((x - x_mean)**2)))
    
    return {
        'slope': slope,
        'intercept': intercept,
        'r_squared': r_squared,
        'se_slope': se_slope,
        'se_intercept': se_intercept,
        'residuals': residuals,
        'y_pred': y_pred
    }

# 回归分析示例
x_reg = np.random.uniform(0, 10, 50)
y_reg = 2 + 3 * x_reg + np.random.normal(0, 1, 50)

regression_results = simple_linear_regression(x_reg, y_reg)
print(f"回归方程: y = {regression_results['intercept']:.3f} + {regression_results['slope']:.3f}x")
print(f"R² = {regression_results['r_squared']:.3f}")
print(f"斜率标准误: {regression_results['se_slope']:.3f}")
print(f"截距标准误: {regression_results['se_intercept']:.3f}")

# 9. 非参数检验
print("\n9. 非参数检验")

# 9.1 Mann-Whitney U检验
def mann_whitney_test(x, y):
    """Mann-Whitney U检验（非参数版本的两样本t检验）"""
    statistic, p_value = stats.mannwhitneyu(x, y, alternative='two-sided')
    return statistic, p_value

# 9.2 Wilcoxon符号秩检验
def wilcoxon_test(x, y):
    """Wilcoxon符号秩检验（非参数版本的配对t检验）"""
    statistic, p_value = stats.wilcoxon(x, y)
    return statistic, p_value

# 9.3 Kruskal-Wallis检验
def kruskal_wallis_test(*groups):
    """Kruskal-Wallis检验（非参数版本的方差分析）"""
    statistic, p_value = stats.kruskal(*groups)
    return statistic, p_value

# 非参数检验示例
print("9.1 Mann-Whitney U检验")
group1_np = np.random.exponential(2, 50)
group2_np = np.random.exponential(3, 50)
u_stat, u_p = mann_whitney_test(group1_np, group2_np)
print(f"U统计量: {u_stat:.3f}, p值: {u_p:.6f}")

# 10. 效应量
print("\n10. 效应量")

def cohens_d(group1, group2):
    """Cohen's d效应量"""
    n1, n2 = len(group1), len(group2)
    pooled_std = np.sqrt(((n1-1)*np.var(group1, ddof=1) + (n2-1)*np.var(group2, ddof=1)) / (n1+n2-2))
    return (np.mean(group1) - np.mean(group2)) / pooled_std

def eta_squared(f_stat, df_between, df_within):
    """η²效应量（方差分析）"""
    return (f_stat * df_between) / (f_stat * df_between + df_within)

# 效应量示例
d = cohens_d(group1, group2)
print(f"Cohen's d: {d:.3f}")

# 解释效应量大小
if abs(d) < 0.2:
    effect_size = "小"
elif abs(d) < 0.5:
    effect_size = "中等"
elif abs(d) < 0.8:
    effect_size = "大"
else:
    effect_size = "非常大"

print(f"效应量大小: {effect_size}")

# 11. 统计图表可视化
print("\n11. 统计图表可视化")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 11.1 直方图和密度图
axes[0, 0].hist(normal_data, bins=30, alpha=0.7, density=True, label='数据')
x_norm = np.linspace(normal_data.min(), normal_data.max(), 100)
axes[0, 0].plot(x_norm, stats.norm.pdf(x_norm, mu_hat, sigma_hat), 'r-', label='拟合正态分布')
axes[0, 0].set_title('直方图与拟合分布')
axes[0, 0].legend()

# 11.2 箱线图
data_groups = [group_a, group_b, group_c]
axes[0, 1].boxplot(data_groups, labels=['组A', '组B', '组C'])
axes[0, 1].set_title('箱线图比较')

# 11.3 Q-Q图
stats.probplot(normal_data, dist="norm", plot=axes[0, 2])
axes[0, 2].set_title('Q-Q图（正态性检验）')

# 11.4 散点图和回归线
axes[1, 0].scatter(x_reg, y_reg, alpha=0.6)
axes[1, 0].plot(x_reg, regression_results['y_pred'], 'r-', linewidth=2)
axes[1, 0].set_title('散点图与回归线')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('y')

# 11.5 残差图
axes[1, 1].scatter(regression_results['y_pred'], regression_results['residuals'], alpha=0.6)
axes[1, 1].axhline(y=0, color='r', linestyle='--')
axes[1, 1].set_title('残差图')
axes[1, 1].set_xlabel('预测值')
axes[1, 1].set_ylabel('残差')

# 11.6 相关性热力图
correlation_data = np.column_stack([x, y_linear, y_nonlinear])
corr_matrix = np.corrcoef(correlation_data.T)
im = axes[1, 2].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
axes[1, 2].set_title('相关性矩阵')
axes[1, 2].set_xticks([0, 1, 2])
axes[1, 2].set_yticks([0, 1, 2])
axes[1, 2].set_xticklabels(['X', 'Y_线性', 'Y_非线性'])
axes[1, 2].set_yticklabels(['X', 'Y_线性', 'Y_非线性'])

# 添加数值标签
for i in range(3):
    for j in range(3):
        axes[1, 2].text(j, i, f'{corr_matrix[i, j]:.2f}', 
                        ha="center", va="center", color="white")

# 11.7 置信区间图
sample_means = []
ci_lowers = []
ci_uppers = []

for i in range(20):
    sample = np.random.normal(100, 15, 30)
    ci_low, ci_up, mean = confidence_interval_mean(sample)
    sample_means.append(mean)
    ci_lowers.append(ci_low)
    ci_uppers.append(ci_up)

x_pos = range(20)
axes[2, 0].errorbar(x_pos, sample_means, 
                   yerr=[np.array(sample_means) - np.array(ci_lowers),
                         np.array(ci_uppers) - np.array(sample_means)],
                   fmt='o', capsize=5)
axes[2, 0].axhline(y=100, color='r', linestyle='--', label='真实均值')
axes[2, 0].set_title('95%置信区间')
axes[2, 0].legend()

# 11.8 功效分析图
effect_sizes = np.linspace(0, 2, 100)
sample_sizes = [10, 20, 30, 50]
powers = []

for n in sample_sizes:
    power_curve = []
    for d in effect_sizes:
        # 计算功效（简化版本）
        ncp = d * np.sqrt(n/2)  # 非中心参数
        power = 1 - stats.t.cdf(stats.t.ppf(0.975, n-1), n-1, ncp) + stats.t.cdf(stats.t.ppf(0.025, n-1), n-1, ncp)
        power_curve.append(power)
    powers.append(power_curve)
    axes[2, 1].plot(effect_sizes, power_curve, label=f'n={n}')

axes[2, 1].set_title('统计功效分析')
axes[2, 1].set_xlabel('效应量')
axes[2, 1].set_ylabel('功效')
axes[2, 1].legend()
axes[2, 1].grid(True, alpha=0.3)

# 11.9 p值分布
# 在原假设为真时的p值分布
p_values_null = []
for _ in range(1000):
    sample1 = np.random.normal(0, 1, 20)
    sample2 = np.random.normal(0, 1, 20)
    _, p = stats.ttest_ind(sample1, sample2)
    p_values_null.append(p)

axes[2, 2].hist(p_values_null, bins=20, alpha=0.7, density=True)
axes[2, 2].axhline(y=1, color='r', linestyle='--', label='理论均匀分布')
axes[2, 2].set_title('原假设为真时的p值分布')
axes[2, 2].set_xlabel('p值')
axes[2, 2].set_ylabel('密度')
axes[2, 2].legend()

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage2-math-fundamentals/statistics_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 练习任务 ===")
print("1. 实现多元线性回归")
print("2. 进行生存分析")
print("3. 实现时间序列分析")
print("4. 进行多重比较校正")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现贝叶斯统计推断")
print("2. 进行元分析(Meta-analysis)")
print("3. 实现引导法(Bootstrap)和刀切法(Jackknife)")
print("4. 进行缺失数据处理")
print("5. 实现混合效应模型")