"""
线性代数基础实现
学习目标：掌握向量、矩阵运算，理解特征值分解和奇异值分解
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

print("=== 线性代数基础实现 ===\n")

# 1. 向量基础
print("1. 向量基础")
# 向量创建
v1 = np.array([1, 2, 3])
v2 = np.array([4, 5, 6])
print(f"向量v1: {v1}")
print(f"向量v2: {v2}")

# 向量运算
print(f"向量加法: {v1 + v2}")
print(f"向量减法: {v1 - v2}")
print(f"标量乘法: {3 * v1}")

# 向量内积（点积）
dot_product = np.dot(v1, v2)
print(f"内积: {dot_product}")

# 向量外积（叉积）
cross_product = np.cross(v1, v2)
print(f"外积: {cross_product}")

# 向量模长
norm_v1 = np.linalg.norm(v1)
norm_v2 = np.linalg.norm(v2)
print(f"v1的模长: {norm_v1:.3f}")
print(f"v2的模长: {norm_v2:.3f}")

# 向量夹角
cos_angle = dot_product / (norm_v1 * norm_v2)
angle = np.arccos(cos_angle)
print(f"向量夹角: {angle:.3f} 弧度 ({np.degrees(angle):.1f}度)")

# 2. 矩阵基础
print("\n2. 矩阵基础")
# 矩阵创建
A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
B = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")

# 矩阵运算
print(f"矩阵加法:\n{A + B}")
print(f"矩阵减法:\n{A - B}")
print(f"矩阵乘法:\n{np.dot(A, B)}")
print(f"矩阵转置:\n{A.T}")

# 矩阵的迹
trace_A = np.trace(A)
print(f"矩阵A的迹: {trace_A}")

# 矩阵的行列式
det_A = np.linalg.det(A)
print(f"矩阵A的行列式: {det_A:.3f}")

# 矩阵的秩
rank_A = np.linalg.matrix_rank(A)
print(f"矩阵A的秩: {rank_A}")

# 3. 可逆矩阵和逆矩阵
print("\n3. 可逆矩阵和逆矩阵")
# 创建可逆矩阵
C = np.array([[1, 2], [3, 4]])
print(f"矩阵C:\n{C}")
print(f"矩阵C的行列式: {np.linalg.det(C):.3f}")

# 计算逆矩阵
try:
    C_inv = np.linalg.inv(C)
    print(f"矩阵C的逆矩阵:\n{C_inv}")
    
    # 验证逆矩阵
    identity = np.dot(C, C_inv)
    print(f"CC^(-1) = \n{identity}")
    print(f"是否为单位矩阵: {np.allclose(identity, np.eye(2))}")
except np.linalg.LinAlgError:
    print("矩阵不可逆")

# 4. 线性方程组求解
print("\n4. 线性方程组求解")
# 方程组: 2x + 3y = 7, 4x + y = 6
# 矩阵形式: Ax = b
A_eq = np.array([[2, 3], [4, 1]])
b_eq = np.array([7, 6])
print(f"系数矩阵A:\n{A_eq}")
print(f"常数向量b: {b_eq}")

# 求解
x_solution = np.linalg.solve(A_eq, b_eq)
print(f"解向量x: {x_solution}")

# 验证解
verification = np.dot(A_eq, x_solution)
print(f"验证Ax = {verification}")
print(f"是否等于b: {np.allclose(verification, b_eq)}")

# 5. 特征值和特征向量
print("\n5. 特征值和特征向量")
# 创建对称矩阵
S = np.array([[4, 2], [2, 3]])
print(f"对称矩阵S:\n{S}")

# 计算特征值和特征向量
eigenvalues, eigenvectors = np.linalg.eig(S)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 验证特征值分解
for i in range(len(eigenvalues)):
    lambda_i = eigenvalues[i]
    v_i = eigenvectors[:, i]
    left_side = np.dot(S, v_i)
    right_side = lambda_i * v_i
    print(f"特征值{i+1}: λ = {lambda_i:.3f}")
    print(f"Sv_{i+1} = {left_side}")
    print(f"λ_{i+1}v_{i+1} = {right_side}")
    print(f"验证通过: {np.allclose(left_side, right_side)}")

# 6. 奇异值分解 (SVD)
print("\n6. 奇异值分解 (SVD)")
# 创建矩阵
M = np.array([[1, 2, 3], [4, 5, 6]])
print(f"矩阵M:\n{M}")

# 进行SVD分解
U, s, VT = np.linalg.svd(M)
print(f"U矩阵形状: {U.shape}")
print(f"奇异值: {s}")
print(f"V^T矩阵形状: {VT.shape}")

# 重构原矩阵
S_matrix = np.zeros((U.shape[0], VT.shape[0]))
S_matrix[:len(s), :len(s)] = np.diag(s)
M_reconstructed = np.dot(U, np.dot(S_matrix, VT))
print(f"重构矩阵:\n{M_reconstructed}")
print(f"重构成功: {np.allclose(M, M_reconstructed)}")

# 7. 矩阵的范数
print("\n7. 矩阵的范数")
A_norm = np.array([[1, 2], [3, 4]])
print(f"矩阵A:\n{A_norm}")

# 各种范数
frobenius_norm = np.linalg.norm(A_norm, 'fro')
nuclear_norm = np.linalg.norm(A_norm, 'nuc')
max_norm = np.linalg.norm(A_norm, np.inf)
print(f"Frobenius范数: {frobenius_norm:.3f}")
print(f"核范数: {nuclear_norm:.3f}")
print(f"最大范数: {max_norm:.3f}")

# 8. 向量空间和线性无关性
print("\n8. 向量空间和线性无关性")
# 三个向量
v1 = np.array([1, 0, 0])
v2 = np.array([0, 1, 0])
v3 = np.array([0, 0, 1])
v4 = np.array([1, 1, 1])

# 构成矩阵
V = np.column_stack([v1, v2, v3])
print(f"标准基向量矩阵:\n{V}")
print(f"矩阵的秩: {np.linalg.matrix_rank(V)}")
print("这三个向量线性无关")

# 检查线性相关性
V_dependent = np.column_stack([v1, v2, v1 + v2])
print(f"\n包含线性相关向量的矩阵:\n{V_dependent}")
print(f"矩阵的秩: {np.linalg.matrix_rank(V_dependent)}")
print("第三个向量可以由前两个向量线性表示")

# 9. 投影和正交化
print("\n9. 投影和正交化")
# 向量在另一个向量上的投影
a = np.array([3, 4])
b = np.array([1, 0])

# 计算投影
proj_a_on_b = np.dot(a, b) / np.dot(b, b) * b
print(f"向量a: {a}")
print(f"向量b: {b}")
print(f"a在b上的投影: {proj_a_on_b}")

# Gram-Schmidt正交化
def gram_schmidt(vectors):
    """Gram-Schmidt正交化过程"""
    orthogonal = []
    for v in vectors:
        u = v.copy()
        for prev_u in orthogonal:
            u = u - np.dot(v, prev_u) / np.dot(prev_u, prev_u) * prev_u
        orthogonal.append(u)
    return orthogonal

# 测试正交化
original_vectors = [np.array([1, 1]), np.array([1, 0])]
orthogonal_vectors = gram_schmidt(original_vectors)
print(f"\n原始向量: {original_vectors}")
print(f"正交化后: {orthogonal_vectors}")

# 验证正交性
dot_product = np.dot(orthogonal_vectors[0], orthogonal_vectors[1])
print(f"正交向量内积: {dot_product:.6f}")

# 10. 矩阵分解的应用：主成分分析预览
print("\n10. 矩阵分解的应用：主成分分析预览")
# 生成2D数据
np.random.seed(42)
n_samples = 100
X = np.random.randn(n_samples, 2)
# 添加相关性
X[:, 1] = X[:, 0] + 0.5 * np.random.randn(n_samples)

# 中心化数据
X_centered = X - np.mean(X, axis=0)

# 计算协方差矩阵
cov_matrix = np.cov(X_centered.T)
print(f"协方差矩阵:\n{cov_matrix}")

# 特征值分解
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
print(f"特征值: {eigenvalues}")
print(f"特征向量:\n{eigenvectors}")

# 可视化
plt.figure(figsize=(10, 8))

# 原始数据
plt.subplot(2, 2, 1)
plt.scatter(X[:, 0], X[:, 1], alpha=0.6)
plt.title('原始数据')
plt.grid(True)

# 中心化数据
plt.subplot(2, 2, 2)
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6)
plt.title('中心化数据')
plt.grid(True)

# 主成分方向
plt.subplot(2, 2, 3)
plt.scatter(X_centered[:, 0], X_centered[:, 1], alpha=0.6)
mean_x, mean_y = 0, 0
for i in range(2):
    plt.arrow(mean_x, mean_y, 
              eigenvectors[0, i] * eigenvalues[i], 
              eigenvectors[1, i] * eigenvalues[i],
              head_width=0.1, head_length=0.1, fc=f'C{i}', ec=f'C{i}')
plt.title('主成分方向')
plt.grid(True)

# 变换后的数据
X_transformed = np.dot(X_centered, eigenvectors)
plt.subplot(2, 2, 4)
plt.scatter(X_transformed[:, 0], X_transformed[:, 1], alpha=0.6)
plt.title('主成分变换后的数据')
plt.grid(True)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage2-math-fundamentals/linear_algebra_demo.png', 
            dpi=300, bbox_inches='tight')
plt.close()

print("\n=== 练习任务 ===")
print("1. 实现矩阵的QR分解")
print("2. 计算矩阵的伪逆")
print("3. 实现向量的正交化算法")
print("4. 用SVD进行图像压缩")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现幂方法求最大特征值")
print("2. 用线性代数方法解决最小二乘问题")
print("3. 实现矩阵的谱分解")
print("4. 探索不同矩阵范数的几何意义")
print("5. 实现Householder变换")