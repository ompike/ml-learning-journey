"""
概率论基础实现
学习目标：理解概率分布、贝叶斯定理、期望方差等概率论核心概念
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 概率论基础实现 ===\n")

# 1. 基本概率概念
print("1. 基本概率概念")

# 样本空间和事件
def coin_flip_simulation(n_flips=1000):
    """硬币投掷模拟"""
    flips = np.random.choice(['正', '反'], n_flips)
    prob_heads = np.sum(flips == '正') / n_flips
    return flips, prob_heads

flips, prob = coin_flip_simulation(10000)
print(f"投掷10000次硬币，正面概率: {prob:.3f}")

# 2. 概率分布
print("\n2. 概率分布")

# 离散分布
print("2.1 离散分布")

# 伯努利分布
def bernoulli_pmf(x, p):
    """伯努利分布概率质量函数"""
    return p**x * (1-p)**(1-x)

# 二项分布
def binomial_pmf(k, n, p):
    """二项分布概率质量函数"""
    from math import comb
    return comb(n, k) * (p**k) * ((1-p)**(n-k))

# 泊松分布
def poisson_pmf(k, lambda_):
    """泊松分布概率质量函数"""
    from math import exp, factorial
    return (lambda_**k * exp(-lambda_)) / factorial(k)

# 测试离散分布
p = 0.3
n = 10
lambda_ = 3

print(f"伯努利分布 P(X=1|p=0.3) = {bernoulli_pmf(1, p):.3f}")
print(f"二项分布 P(X=3|n=10,p=0.3) = {binomial_pmf(3, n, p):.3f}")
print(f"泊松分布 P(X=2|λ=3) = {poisson_pmf(2, lambda_):.3f}")

# 连续分布
print("\n2.2 连续分布")

# 均匀分布
def uniform_pdf(x, a=0, b=1):
    """均匀分布概率密度函数"""
    return np.where((x >= a) & (x <= b), 1/(b-a), 0)

# 正态分布
def normal_pdf(x, mu=0, sigma=1):
    """正态分布概率密度函数"""
    return (1/(sigma * np.sqrt(2*np.pi))) * np.exp(-0.5*((x-mu)/sigma)**2)

# 指数分布
def exponential_pdf(x, lambda_=1):
    """指数分布概率密度函数"""
    return np.where(x >= 0, lambda_ * np.exp(-lambda_ * x), 0)

# 3. 贝叶斯定理
print("\n3. 贝叶斯定理")

def bayes_theorem(prior, likelihood, evidence):
    """贝叶斯定理：P(H|E) = P(E|H) * P(H) / P(E)"""
    return (likelihood * prior) / evidence

# 经典例子：医疗诊断
print("医疗诊断例子：")
print("疾病患病率（先验概率）: 1%")
print("测试准确率: 99%")
print("假阳性率: 5%")

# P(疾病) = 0.01
prior_disease = 0.01
prior_healthy = 0.99

# P(阳性|疾病) = 0.99
likelihood_positive_given_disease = 0.99

# P(阳性|健康) = 0.05
likelihood_positive_given_healthy = 0.05

# P(阳性) = P(阳性|疾病)*P(疾病) + P(阳性|健康)*P(健康)
evidence_positive = (likelihood_positive_given_disease * prior_disease + 
                    likelihood_positive_given_healthy * prior_healthy)

# P(疾病|阳性)
posterior_disease = bayes_theorem(prior_disease, likelihood_positive_given_disease, evidence_positive)

print(f"测试呈阳性时，实际患病概率: {posterior_disease:.3f}")

# 4. 期望和方差
print("\n4. 期望和方差")

def calculate_expectation(values, probabilities):
    """计算期望"""
    return np.sum(values * probabilities)

def calculate_variance(values, probabilities, expectation=None):
    """计算方差"""
    if expectation is None:
        expectation = calculate_expectation(values, probabilities)
    return np.sum((values - expectation)**2 * probabilities)

# 离散随机变量例子
dice_values = np.array([1, 2, 3, 4, 5, 6])
dice_probs = np.array([1/6] * 6)

dice_expectation = calculate_expectation(dice_values, dice_probs)
dice_variance = calculate_variance(dice_values, dice_probs, dice_expectation)

print(f"掷骰子期望: {dice_expectation:.3f}")
print(f"掷骰子方差: {dice_variance:.3f}")

# 5. 大数定律和中心极限定理
print("\n5. 大数定律和中心极限定理")

def law_of_large_numbers_demo(n_samples_list=[10, 100, 1000, 10000]):
    """大数定律演示"""
    true_mean = 3.5  # 骰子期望值
    sample_means = []
    
    for n in n_samples_list:
        samples = np.random.randint(1, 7, n)
        sample_mean = np.mean(samples)
        sample_means.append(sample_mean)
        print(f"样本数={n}, 样本均值={sample_mean:.3f}, 与真实均值差距={abs(sample_mean-true_mean):.3f}")
    
    return sample_means

sample_means = law_of_large_numbers_demo()

# 中心极限定理演示
def central_limit_theorem_demo(population_dist='uniform', n_samples=1000, sample_size=30):
    """中心极限定理演示"""
    sample_means = []
    
    for _ in range(n_samples):
        if population_dist == 'uniform':
            sample = np.random.uniform(0, 10, sample_size)
        elif population_dist == 'exponential':
            sample = np.random.exponential(2, sample_size)
        else:
            sample = np.random.normal(5, 2, sample_size)
        
        sample_means.append(np.mean(sample))
    
    return np.array(sample_means)

# 6. 概率分布的可视化
print("\n6. 概率分布可视化")

fig, axes = plt.subplots(3, 3, figsize=(15, 12))

# 第一行：离散分布
x_discrete = np.arange(0, 11)

# 二项分布
axes[0, 0].bar(x_discrete, [binomial_pmf(k, 10, 0.3) for k in x_discrete])
axes[0, 0].set_title('二项分布 (n=10, p=0.3)')
axes[0, 0].set_xlabel('k')
axes[0, 0].set_ylabel('P(X=k)')

# 泊松分布
x_poisson = np.arange(0, 15)
axes[0, 1].bar(x_poisson, [poisson_pmf(k, 3) for k in x_poisson])
axes[0, 1].set_title('泊松分布 (λ=3)')
axes[0, 1].set_xlabel('k')
axes[0, 1].set_ylabel('P(X=k)')

# 几何分布
def geometric_pmf(k, p):
    return (1-p)**(k-1) * p

x_geom = np.arange(1, 11)
axes[0, 2].bar(x_geom, [geometric_pmf(k, 0.3) for k in x_geom])
axes[0, 2].set_title('几何分布 (p=0.3)')
axes[0, 2].set_xlabel('k')
axes[0, 2].set_ylabel('P(X=k)')

# 第二行：连续分布
x_continuous = np.linspace(-4, 4, 1000)

# 标准正态分布
axes[1, 0].plot(x_continuous, normal_pdf(x_continuous, 0, 1))
axes[1, 0].set_title('标准正态分布')
axes[1, 0].set_xlabel('x')
axes[1, 0].set_ylabel('f(x)')
axes[1, 0].grid(True, alpha=0.3)

# 不同参数的正态分布
axes[1, 1].plot(x_continuous, normal_pdf(x_continuous, 0, 1), label='μ=0, σ=1')
axes[1, 1].plot(x_continuous, normal_pdf(x_continuous, 1, 1), label='μ=1, σ=1')
axes[1, 1].plot(x_continuous, normal_pdf(x_continuous, 0, 2), label='μ=0, σ=2')
axes[1, 1].set_title('正态分布族')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 指数分布
x_exp = np.linspace(0, 5, 1000)
axes[1, 2].plot(x_exp, exponential_pdf(x_exp, 1), label='λ=1')
axes[1, 2].plot(x_exp, exponential_pdf(x_exp, 2), label='λ=2')
axes[1, 2].set_title('指数分布')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# 第三行：中心极限定理演示
sample_means_uniform = central_limit_theorem_demo('uniform')
sample_means_exp = central_limit_theorem_demo('exponential')

axes[2, 0].hist(sample_means_uniform, bins=50, alpha=0.7, density=True)
axes[2, 0].set_title('样本均值分布 (总体为均匀分布)')
axes[2, 0].set_xlabel('样本均值')
axes[2, 0].set_ylabel('密度')

axes[2, 1].hist(sample_means_exp, bins=50, alpha=0.7, density=True)
axes[2, 1].set_title('样本均值分布 (总体为指数分布)')
axes[2, 1].set_xlabel('样本均值')
axes[2, 1].set_ylabel('密度')

# 大数定律演示
n_values = [10, 100, 1000, 10000]
running_means = []
for n in range(1, 10001):
    if n <= 10000:
        sample = np.random.randint(1, 7, n)
        running_means.append(np.mean(sample))

axes[2, 2].plot(range(1, len(running_means)+1), running_means)
axes[2, 2].axhline(y=3.5, color='r', linestyle='--', label='真实均值=3.5')
axes[2, 2].set_title('大数定律演示')
axes[2, 2].set_xlabel('样本数')
axes[2, 2].set_ylabel('样本均值')
axes[2, 2].legend()
axes[2, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage2-math-fundamentals/probability_distributions.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 7. 协方差和相关系数
print("\n7. 协方差和相关系数")

def calculate_covariance(x, y):
    """计算协方差"""
    mean_x, mean_y = np.mean(x), np.mean(y)
    return np.mean((x - mean_x) * (y - mean_y))

def calculate_correlation(x, y):
    """计算相关系数"""
    cov_xy = calculate_covariance(x, y)
    std_x, std_y = np.std(x), np.std(y)
    return cov_xy / (std_x * std_y)

# 生成相关数据
np.random.seed(42)
n = 1000

# 正相关
x1 = np.random.normal(0, 1, n)
y1 = x1 + np.random.normal(0, 0.5, n)

# 负相关
x2 = np.random.normal(0, 1, n)
y2 = -x2 + np.random.normal(0, 0.5, n)

# 无相关
x3 = np.random.normal(0, 1, n)
y3 = np.random.normal(0, 1, n)

print(f"正相关数据相关系数: {calculate_correlation(x1, y1):.3f}")
print(f"负相关数据相关系数: {calculate_correlation(x2, y2):.3f}")
print(f"无相关数据相关系数: {calculate_correlation(x3, y3):.3f}")

# 8. 条件概率和独立性
print("\n8. 条件概率和独立性")

def conditional_probability(joint_prob, marginal_prob):
    """条件概率 P(A|B) = P(A∩B) / P(B)"""
    return joint_prob / marginal_prob

def check_independence(p_a, p_b, p_ab):
    """检查事件独立性：P(A∩B) = P(A) * P(B)"""
    expected_joint = p_a * p_b
    return abs(p_ab - expected_joint) < 1e-6

# 例子：抛硬币和掷骰子
print("例子：抛硬币(正面)和掷骰子(偶数)")
p_heads = 0.5
p_even = 0.5
p_heads_and_even = 0.25  # 独立事件

print(f"P(正面) = {p_heads}")
print(f"P(偶数) = {p_even}")
print(f"P(正面且偶数) = {p_heads_and_even}")
print(f"事件独立性: {check_independence(p_heads, p_even, p_heads_and_even)}")

# 9. 蒙特卡洛方法
print("\n9. 蒙特卡洛方法")

def monte_carlo_pi(n_samples=100000):
    """用蒙特卡洛方法估计π"""
    # 在单位正方形内随机生成点
    x = np.random.uniform(-1, 1, n_samples)
    y = np.random.uniform(-1, 1, n_samples)
    
    # 计算落在单位圆内的点数
    inside_circle = (x**2 + y**2) <= 1
    pi_estimate = 4 * np.sum(inside_circle) / n_samples
    
    return pi_estimate

pi_est = monte_carlo_pi()
print(f"蒙特卡洛估计π值: {pi_est:.6f}")
print(f"真实π值: {np.pi:.6f}")
print(f"误差: {abs(pi_est - np.pi):.6f}")

# 10. 假设检验基础
print("\n10. 假设检验基础")

def z_test(sample_mean, population_mean, population_std, n):
    """Z检验"""
    z_score = (sample_mean - population_mean) / (population_std / np.sqrt(n))
    return z_score

def p_value_two_tailed(z_score):
    """双尾检验p值"""
    return 2 * (1 - stats.norm.cdf(abs(z_score)))

# 例子：检验样本均值是否等于总体均值
sample_data = np.random.normal(100, 15, 100)  # 样本
population_mean = 100
population_std = 15

sample_mean = np.mean(sample_data)
z_score = z_test(sample_mean, population_mean, population_std, len(sample_data))
p_val = p_value_two_tailed(z_score)

print(f"样本均值: {sample_mean:.2f}")
print(f"Z统计量: {z_score:.3f}")
print(f"p值: {p_val:.6f}")
print(f"在α=0.05水平下{'拒绝' if p_val < 0.05 else '接受'}原假设")

print("\n=== 练习任务 ===")
print("1. 实现其他概率分布（Beta、Gamma等）")
print("2. 编写贝叶斯更新算法")
print("3. 实现马尔可夫链蒙特卡洛(MCMC)")
print("4. 进行A/B测试的统计分析")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现朴素贝叶斯分类器")
print("2. 用EM算法估计混合高斯分布参数")
print("3. 实现重要性采样")
print("4. 进行贝叶斯A/B测试")
print("5. 实现信息论相关概念（熵、互信息等）")