"""
NumPy基础练习
学习目标：掌握NumPy数组的创建、操作和基本运算
"""

import numpy as np

print("=== NumPy基础练习 ===\n")

# 1. 数组创建
print("1. 数组创建")
# 从列表创建数组
arr1 = np.array([1, 2, 3, 4, 5])
print(f"从列表创建: {arr1}")

# 创建特殊数组
zeros = np.zeros(5)
ones = np.ones(5)
full = np.full(5, 7)
print(f"零数组: {zeros}")
print(f"全1数组: {ones}")
print(f"全7数组: {full}")

# 创建范围数组
range_arr = np.arange(10)
linspace_arr = np.linspace(0, 10, 5)
print(f"范围数组: {range_arr}")
print(f"线性空间: {linspace_arr}")

# 2. 多维数组
print("\n2. 多维数组")
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print(f"2D数组:\n{matrix}")
print(f"形状: {matrix.shape}")
print(f"维度: {matrix.ndim}")
print(f"大小: {matrix.size}")

# 3. 数组索引和切片
print("\n3. 数组索引和切片")
arr = np.arange(10)
print(f"原数组: {arr}")
print(f"索引[3]: {arr[3]}")
print(f"切片[2:7]: {arr[2:7]}")
print(f"步长切片[::2]: {arr[::2]}")

# 多维数组索引
print(f"矩阵[0,1]: {matrix[0,1]}")
print(f"矩阵第1行: {matrix[0,:]}")
print(f"矩阵第2列: {matrix[:,1]}")

# 4. 数组运算
print("\n4. 数组运算")
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(f"数组a: {a}")
print(f"数组b: {b}")
print(f"加法: {a + b}")
print(f"减法: {a - b}")
print(f"乘法: {a * b}")
print(f"除法: {a / b}")
print(f"幂运算: {a ** 2}")

# 5. 广播机制
print("\n5. 广播机制")
arr_2d = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
print(f"2D数组:\n{arr_2d}")
print(f"标量: {scalar}")
print(f"数组+标量:\n{arr_2d + scalar}")

# 6. 数学函数
print("\n6. 数学函数")
arr = np.array([1, 4, 9, 16])
print(f"数组: {arr}")
print(f"平方根: {np.sqrt(arr)}")
print(f"对数: {np.log(arr)}")
print(f"正弦: {np.sin(arr)}")

# 7. 统计函数
print("\n7. 统计函数")
data = np.random.normal(0, 1, 100)  # 生成100个正态分布随机数
print(f"数据大小: {data.size}")
print(f"均值: {np.mean(data):.3f}")
print(f"标准差: {np.std(data):.3f}")
print(f"最大值: {np.max(data):.3f}")
print(f"最小值: {np.min(data):.3f}")

# 8. 线性代数
print("\n8. 线性代数")
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"矩阵A:\n{A}")
print(f"矩阵B:\n{B}")
print(f"矩阵乘法:\n{np.dot(A, B)}")
print(f"矩阵转置:\n{A.T}")

# 练习任务
print("\n=== 练习任务 ===")
print("1. 创建一个3x3的随机矩阵")
random_matrix = np.random.rand(3, 3)
print(f"随机矩阵:\n{random_matrix}")

print("\n2. 计算矩阵每行的和")
row_sums = np.sum(random_matrix, axis=1)
print(f"每行和: {row_sums}")

print("\n3. 找到矩阵中的最大值和最小值位置")
max_pos = np.unravel_index(np.argmax(random_matrix), random_matrix.shape)
min_pos = np.unravel_index(np.argmin(random_matrix), random_matrix.shape)
print(f"最大值位置: {max_pos}")
print(f"最小值位置: {min_pos}")

# TODO: 尝试修改以下代码，进行更多实验
# 1. 创建不同形状的数组
# 2. 尝试不同的数学运算
# 3. 使用条件索引（布尔索引）
# 4. 数组重塑（reshape）
print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 使用布尔索引筛选数组中大于0的元素")
print("2. 将一维数组重塑为二维数组")
print("3. 计算两个矩阵的特征值和特征向量")
print("4. 使用NumPy生成不同分布的随机数")