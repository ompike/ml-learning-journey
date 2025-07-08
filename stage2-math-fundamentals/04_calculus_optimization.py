"""
微积分和优化基础实现
学习目标：理解导数、梯度、优化算法，为机器学习中的优化奠定基础
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sympy as sp
from scipy.optimize import minimize, minimize_scalar
from scipy.misc import derivative

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

print("=== 微积分和优化基础实现 ===\n")

# 1. 导数和偏导数
print("1. 导数和偏导数")

def numerical_derivative(f, x, h=1e-7):
    """数值求导（前向差分）"""
    return (f(x + h) - f(x)) / h

def numerical_derivative_central(f, x, h=1e-5):
    """数值求导（中心差分，更精确）"""
    return (f(x + h) - f(x - h)) / (2 * h)

def numerical_partial_derivative(f, x, i, h=1e-7):
    """数值偏导数"""
    x_plus_h = x.copy()
    x_plus_h[i] += h
    x_minus_h = x.copy()
    x_minus_h[i] -= h
    return (f(x_plus_h) - f(x_minus_h)) / (2 * h)

# 示例函数
def f(x):
    return x**2 + 3*x + 2

def f_derivative_analytical(x):
    return 2*x + 3

# 测试数值导数
x_test = 2.0
analytical_deriv = f_derivative_analytical(x_test)
numerical_deriv = numerical_derivative_central(f, x_test)

print(f"在x={x_test}处:")
print(f"解析导数: {analytical_deriv}")
print(f"数值导数: {numerical_deriv:.6f}")
print(f"误差: {abs(analytical_deriv - numerical_deriv):.8f}")

# 多元函数偏导数示例
def g(x):
    """多元函数 g(x,y) = x² + xy + y²"""
    return x[0]**2 + x[0]*x[1] + x[1]**2

def g_partial_x(x):
    """对x的偏导数"""
    return 2*x[0] + x[1]

def g_partial_y(x):
    """对y的偏导数"""
    return x[0] + 2*x[1]

x_test_2d = np.array([1.0, 2.0])
print(f"\n多元函数在点{x_test_2d}处:")
print(f"∂g/∂x 解析值: {g_partial_x(x_test_2d)}")
print(f"∂g/∂x 数值值: {numerical_partial_derivative(g, x_test_2d, 0):.6f}")
print(f"∂g/∂y 解析值: {g_partial_y(x_test_2d)}")
print(f"∂g/∂y 数值值: {numerical_partial_derivative(g, x_test_2d, 1):.6f}")

# 2. 梯度和方向导数
print("\n2. 梯度和方向导数")

def gradient_numerical(f, x, h=1e-7):
    """数值计算梯度"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        grad[i] = numerical_partial_derivative(f, x, i, h)
    return grad

def directional_derivative(f, x, direction, h=1e-7):
    """方向导数"""
    # 归一化方向向量
    direction = direction / np.linalg.norm(direction)
    return (f(x + h * direction) - f(x)) / h

# 梯度示例
grad_analytical = np.array([g_partial_x(x_test_2d), g_partial_y(x_test_2d)])
grad_numerical = gradient_numerical(g, x_test_2d)

print(f"解析梯度: {grad_analytical}")
print(f"数值梯度: {grad_numerical}")

# 方向导数示例
direction = np.array([1.0, 1.0])  # 45度方向
dir_deriv = directional_derivative(g, x_test_2d, direction)
print(f"45度方向的方向导数: {dir_deriv:.6f}")

# 3. 梯度下降算法
print("\n3. 梯度下降算法")

def gradient_descent(f, grad_f, x0, learning_rate=0.01, max_iterations=1000, tolerance=1e-6):
    """梯度下降算法"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iterations):
        grad = grad_f(x)
        x_new = x - learning_rate * grad
        
        # 检查收敛
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"在第{i+1}次迭代后收敛")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

def gradient_descent_with_momentum(f, grad_f, x0, learning_rate=0.01, momentum=0.9, 
                                 max_iterations=1000, tolerance=1e-6):
    """带动量的梯度下降"""
    x = x0.copy()
    velocity = np.zeros_like(x0)
    history = [x.copy()]
    
    for i in range(max_iterations):
        grad = grad_f(x)
        velocity = momentum * velocity - learning_rate * grad
        x_new = x + velocity
        
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"动量梯度下降在第{i+1}次迭代后收敛")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

def adam_optimizer(f, grad_f, x0, learning_rate=0.001, beta1=0.9, beta2=0.999, 
                  epsilon=1e-8, max_iterations=1000, tolerance=1e-6):
    """Adam优化器"""
    x = x0.copy()
    m = np.zeros_like(x0)  # 一阶矩估计
    v = np.zeros_like(x0)  # 二阶矩估计
    history = [x.copy()]
    
    for t in range(1, max_iterations + 1):
        grad = grad_f(x)
        
        # 更新偏置矩估计
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * grad**2
        
        # 偏置校正
        m_hat = m / (1 - beta1**t)
        v_hat = v / (1 - beta2**t)
        
        # 更新参数
        x_new = x - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
        
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"Adam优化器在第{t}次迭代后收敛")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# 测试优化算法
def test_function(x):
    """测试函数：f(x,y) = (x-3)² + (y-2)²"""
    return (x[0] - 3)**2 + (x[1] - 2)**2

def test_function_grad(x):
    """测试函数的梯度"""
    return np.array([2*(x[0] - 3), 2*(x[1] - 2)])

x0 = np.array([0.0, 0.0])
print(f"初始点: {x0}")
print(f"真实最优点: [3, 2]")

# 标准梯度下降
x_opt_gd, history_gd = gradient_descent(test_function, test_function_grad, x0, 
                                       learning_rate=0.1, max_iterations=100)
print(f"梯度下降结果: {x_opt_gd}")

# 带动量的梯度下降
x_opt_momentum, history_momentum = gradient_descent_with_momentum(
    test_function, test_function_grad, x0, learning_rate=0.1, momentum=0.9, max_iterations=100)
print(f"动量梯度下降结果: {x_opt_momentum}")

# Adam优化器
x_opt_adam, history_adam = adam_optimizer(test_function, test_function_grad, x0, 
                                        learning_rate=0.1, max_iterations=100)
print(f"Adam优化器结果: {x_opt_adam}")

# 4. 约束优化
print("\n4. 约束优化")

def lagrange_multiplier_example():
    """拉格朗日乘数法示例：最小化 f(x,y) = x² + y² 约束 g(x,y) = x + y - 1 = 0"""
    # 使用sympy求解
    x, y, lam = sp.symbols('x y lambda')
    
    # 目标函数
    f = x**2 + y**2
    
    # 约束
    g = x + y - 1
    
    # 拉格朗日函数
    L = f + lam * g
    
    # 求偏导并令其为0
    grad_L = [sp.diff(L, var) for var in [x, y, lam]]
    
    # 求解方程组
    solution = sp.solve(grad_L, [x, y, lam])
    
    return solution

lagrange_solution = lagrange_multiplier_example()
print(f"拉格朗日乘数法解: {lagrange_solution}")

# 5. 牛顿法
print("\n5. 牛顿法")

def newton_method_1d(f, df, d2f, x0, max_iterations=100, tolerance=1e-6):
    """一维牛顿法"""
    x = x0
    history = [x]
    
    for i in range(max_iterations):
        fx = f(x)
        dfx = df(x)
        d2fx = d2f(x)
        
        if abs(d2fx) < 1e-12:
            print("二阶导数接近0，牛顿法可能失效")
            break
            
        x_new = x - dfx / d2fx
        
        if abs(x_new - x) < tolerance:
            print(f"牛顿法在第{i+1}次迭代后收敛")
            break
            
        x = x_new
        history.append(x)
    
    return x, history

def newton_method_multivariate(grad_f, hessian_f, x0, max_iterations=100, tolerance=1e-6):
    """多元牛顿法"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iterations):
        grad = grad_f(x)
        hess = hessian_f(x)
        
        # 求解 Hess * delta_x = -grad
        try:
            delta_x = -np.linalg.solve(hess, grad)
        except np.linalg.LinAlgError:
            print("Hessian矩阵奇异，使用伪逆")
            delta_x = -np.linalg.pinv(hess) @ grad
            
        x_new = x + delta_x
        
        if np.linalg.norm(x_new - x) < tolerance:
            print(f"多元牛顿法在第{i+1}次迭代后收敛")
            break
            
        x = x_new
        history.append(x.copy())
    
    return x, np.array(history)

# 一维牛顿法示例
def f_1d(x):
    return x**3 - 2*x - 5

def df_1d(x):
    return 3*x**2 - 2

def d2f_1d(x):
    return 6*x

x_newton, _ = newton_method_1d(f_1d, df_1d, d2f_1d, x0=2.0)
print(f"一维牛顿法求根结果: {x_newton:.6f}")
print(f"验证: f({x_newton:.6f}) = {f_1d(x_newton):.8f}")

# 6. 数值积分
print("\n6. 数值积分")

def trapezoidal_rule(f, a, b, n):
    """梯形法则数值积分"""
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    integral = h * (0.5 * y[0] + np.sum(y[1:-1]) + 0.5 * y[-1])
    return integral

def simpsons_rule(f, a, b, n):
    """辛普森法则数值积分（n必须是偶数）"""
    if n % 2 != 0:
        n += 1  # 确保n是偶数
    
    h = (b - a) / n
    x = np.linspace(a, b, n + 1)
    y = f(x)
    
    integral = h/3 * (y[0] + 4*np.sum(y[1::2]) + 2*np.sum(y[2:-1:2]) + y[-1])
    return integral

def monte_carlo_integration(f, a, b, n_samples=100000):
    """蒙特卡洛积分"""
    x_random = np.random.uniform(a, b, n_samples)
    integral = (b - a) * np.mean(f(x_random))
    return integral

# 积分示例：∫₀¹ x² dx = 1/3
def integrand(x):
    return x**2

true_value = 1/3
trap_result = trapezoidal_rule(integrand, 0, 1, 1000)
simp_result = simpsons_rule(integrand, 0, 1, 1000)
mc_result = monte_carlo_integration(integrand, 0, 1, 100000)

print(f"∫₀¹ x² dx 的真实值: {true_value:.6f}")
print(f"梯形法则结果: {trap_result:.6f}, 误差: {abs(trap_result - true_value):.8f}")
print(f"辛普森法则结果: {simp_result:.6f}, 误差: {abs(simp_result - true_value):.8f}")
print(f"蒙特卡洛结果: {mc_result:.6f}, 误差: {abs(mc_result - true_value):.8f}")

# 7. 可视化优化过程
print("\n7. 可视化优化过程")

fig = plt.figure(figsize=(18, 12))

# 7.1 一维函数和其导数
ax1 = plt.subplot(3, 3, 1)
x_plot = np.linspace(-3, 3, 1000)
y_plot = x_plot**3 - 3*x_plot
dy_plot = 3*x_plot**2 - 3

ax1.plot(x_plot, y_plot, 'b-', label='f(x) = x³ - 3x')
ax1.plot(x_plot, dy_plot, 'r--', label="f'(x) = 3x² - 3")
ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
ax1.set_title('函数及其导数')
ax1.legend()
ax1.grid(True, alpha=0.3)

# 7.2 梯度下降路径对比
ax2 = plt.subplot(3, 3, 2)
x_range = np.linspace(-1, 4, 100)
y_range = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x_range, y_range)
Z = (X - 3)**2 + (Y - 2)**2

contour = ax2.contour(X, Y, Z, levels=20, alpha=0.6)
ax2.plot(history_gd[:, 0], history_gd[:, 1], 'ro-', label='标准梯度下降', markersize=3)
ax2.plot(history_momentum[:, 0], history_momentum[:, 1], 'go-', label='动量梯度下降', markersize=3)
ax2.plot(history_adam[:, 0], history_adam[:, 1], 'bo-', label='Adam', markersize=3)
ax2.plot(3, 2, 'k*', markersize=15, label='最优点')
ax2.set_title('优化算法路径比较')
ax2.legend()

# 7.3 学习率对收敛的影响
ax3 = plt.subplot(3, 3, 3)
learning_rates = [0.01, 0.1, 0.3, 0.9]
for lr in learning_rates:
    _, history = gradient_descent(test_function, test_function_grad, x0, 
                                learning_rate=lr, max_iterations=50)
    losses = [test_function(x) for x in history]
    ax3.plot(losses, label=f'lr={lr}')

ax3.set_title('学习率对收敛的影响')
ax3.set_xlabel('迭代次数')
ax3.set_ylabel('损失值')
ax3.set_yscale('log')
ax3.legend()
ax3.grid(True, alpha=0.3)

# 7.4 数值导数精度比较
ax4 = plt.subplot(3, 3, 4)
h_values = np.logspace(-10, -1, 100)
errors_forward = []
errors_central = []

for h in h_values:
    error_forward = abs(numerical_derivative(f, x_test, h) - analytical_deriv)
    error_central = abs(numerical_derivative_central(f, x_test, h) - analytical_deriv)
    errors_forward.append(error_forward)
    errors_central.append(error_central)

ax4.loglog(h_values, errors_forward, label='前向差分')
ax4.loglog(h_values, errors_central, label='中心差分')
ax4.set_title('数值导数精度比较')
ax4.set_xlabel('步长 h')
ax4.set_ylabel('误差')
ax4.legend()
ax4.grid(True, alpha=0.3)

# 7.5 数值积分收敛性
ax5 = plt.subplot(3, 3, 5)
n_values = [10, 20, 50, 100, 200, 500, 1000]
trap_errors = []
simp_errors = []

for n in n_values:
    trap_error = abs(trapezoidal_rule(integrand, 0, 1, n) - true_value)
    simp_error = abs(simpsons_rule(integrand, 0, 1, n) - true_value)
    trap_errors.append(trap_error)
    simp_errors.append(simp_error)

ax5.loglog(n_values, trap_errors, 'o-', label='梯形法则')
ax5.loglog(n_values, simp_errors, 's-', label='辛普森法则')
ax5.set_title('数值积分收敛性')
ax5.set_xlabel('分割数 n')
ax5.set_ylabel('误差')
ax5.legend()
ax5.grid(True, alpha=0.3)

# 7.6 3D优化曲面
ax6 = plt.subplot(3, 3, 6, projection='3d')
X_3d = np.linspace(-1, 4, 50)
Y_3d = np.linspace(-1, 3, 50)
X_3d, Y_3d = np.meshgrid(X_3d, Y_3d)
Z_3d = (X_3d - 3)**2 + (Y_3d - 2)**2

ax6.plot_surface(X_3d, Y_3d, Z_3d, alpha=0.6, cmap='viridis')
ax6.plot(history_gd[:, 0], history_gd[:, 1], 
         [test_function(x) for x in history_gd], 'ro-', markersize=4)
ax6.set_title('3D优化曲面')

# 7.7 不同优化函数的收敛性
ax7 = plt.subplot(3, 3, 7)

def rosenbrock(x):
    """Rosenbrock函数（香蕉函数）"""
    return (1 - x[0])**2 + 100*(x[1] - x[0]**2)**2

def rosenbrock_grad(x):
    """Rosenbrock函数的梯度"""
    return np.array([-2*(1 - x[0]) - 400*x[0]*(x[1] - x[0]**2),
                     200*(x[1] - x[0]**2)])

x0_rb = np.array([-1.0, 1.0])
_, history_rb = gradient_descent(rosenbrock, rosenbrock_grad, x0_rb, 
                                learning_rate=0.001, max_iterations=1000)

losses_rb = [rosenbrock(x) for x in history_rb]
ax7.plot(losses_rb)
ax7.set_title('Rosenbrock函数优化')
ax7.set_xlabel('迭代次数')
ax7.set_ylabel('函数值')
ax7.set_yscale('log')
ax7.grid(True, alpha=0.3)

# 7.8 方向导数可视化
ax8 = plt.subplot(3, 3, 8)
x_center = np.array([1.0, 1.0])
angles = np.linspace(0, 2*np.pi, 100)
directional_derivs = []

for angle in angles:
    direction = np.array([np.cos(angle), np.sin(angle)])
    dir_deriv = directional_derivative(g, x_center, direction)
    directional_derivs.append(dir_deriv)

ax8.plot(angles, directional_derivs)
ax8.set_title('方向导数随角度变化')
ax8.set_xlabel('角度（弧度）')
ax8.set_ylabel('方向导数')
ax8.grid(True, alpha=0.3)

# 7.9 优化算法性能比较
ax9 = plt.subplot(3, 3, 9)
algorithms = ['标准梯度下降', '动量梯度下降', 'Adam']
iterations = [len(history_gd), len(history_momentum), len(history_adam)]
final_errors = [test_function(history_gd[-1]), 
                test_function(history_momentum[-1]), 
                test_function(history_adam[-1])]

bars = ax9.bar(algorithms, iterations, alpha=0.7)
ax9.set_title('算法收敛迭代次数比较')
ax9.set_ylabel('迭代次数')

# 在柱状图上显示数值
for bar, iter_count in zip(bars, iterations):
    ax9.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             str(iter_count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage2-math-fundamentals/calculus_optimization.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 练习任务 ===")
print("1. 实现二阶优化方法（BFGS、L-BFGS）")
print("2. 实现约束优化算法（内点法、序列二次规划）")
print("3. 研究不同学习率调度策略")
print("4. 实现随机梯度下降及其变种")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现自适应学习率方法（AdaGrad, RMSprop）")
print("2. 研究优化中的鞍点问题")
print("3. 实现分布式优化算法")
print("4. 研究非凸优化的全局优化方法")
print("5. 实现贝叶斯优化算法")