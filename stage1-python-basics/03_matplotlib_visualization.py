"""
Matplotlib数据可视化练习
学习目标：掌握基本图表绘制、多子图布局和图表美化
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False   # 用来正常显示负号

print("=== Matplotlib数据可视化练习 ===\n")

# 1. 基本线形图
print("1. 基本线形图")
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', color='blue', linewidth=2)
plt.plot(x, y2, label='cos(x)', color='red', linewidth=2, linestyle='--')
plt.xlabel('x')
plt.ylabel('y')
plt.title('三角函数图像')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/01_line_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. 散点图
print("2. 散点图")
np.random.seed(42)
x = np.random.normal(0, 1, 100)
y = x + np.random.normal(0, 0.5, 100)

plt.figure(figsize=(8, 6))
plt.scatter(x, y, alpha=0.6, color='purple')
plt.xlabel('X值')
plt.ylabel('Y值')
plt.title('散点图示例')
plt.grid(True, alpha=0.3)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/02_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. 柱状图
print("3. 柱状图")
categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(8, 6))
bars = plt.bar(categories, values, color=['red', 'green', 'blue', 'orange', 'purple'])
plt.xlabel('类别')
plt.ylabel('数值')
plt.title('柱状图示例')

# 在柱子上显示数值
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
             str(value), ha='center', va='bottom')

plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/03_bar_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. 直方图
print("4. 直方图")
np.random.seed(42)
data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('数值')
plt.ylabel('频率')
plt.title('正态分布直方图')
plt.axvline(np.mean(data), color='red', linestyle='--', linewidth=2, label=f'均值: {np.mean(data):.2f}')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/04_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. 饼图
print("5. 饼图")
sizes = [30, 25, 20, 15, 10]
labels = ['A类', 'B类', 'C类', 'D类', '其他']
colors = ['gold', 'lightcoral', 'lightskyblue', 'lightgreen', 'plum']
explode = (0.1, 0, 0, 0, 0)  # 突出显示第一个扇形

plt.figure(figsize=(8, 8))
plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', 
        shadow=True, startangle=90)
plt.title('饼图示例')
plt.axis('equal')
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/05_pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. 多子图布局
print("6. 多子图布局")
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 第一个子图：线形图
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('正弦函数')
axes[0, 0].grid(True, alpha=0.3)

# 第二个子图：散点图
x = np.random.randn(50)
y = np.random.randn(50)
axes[0, 1].scatter(x, y, alpha=0.6)
axes[0, 1].set_title('随机散点图')
axes[0, 1].grid(True, alpha=0.3)

# 第三个子图：柱状图
categories = ['A', 'B', 'C', 'D']
values = [1, 3, 2, 4]
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('柱状图')

# 第四个子图：直方图
data = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data, bins=30, alpha=0.7)
axes[1, 1].set_title('正态分布直方图')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/06_subplots.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. 热力图
print("7. 热力图")
# 创建相关矩阵数据
np.random.seed(42)
data = np.random.rand(5, 5)
labels = ['特征1', '特征2', '特征3', '特征4', '特征5']

plt.figure(figsize=(8, 6))
im = plt.imshow(data, cmap='coolwarm', aspect='auto')
plt.colorbar(im)
plt.xticks(range(5), labels)
plt.yticks(range(5), labels)
plt.title('特征相关性热力图')

# 添加数值标签
for i in range(5):
    for j in range(5):
        plt.text(j, i, f'{data[i, j]:.2f}', ha='center', va='center', color='white')

plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/07_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. 箱线图
print("8. 箱线图")
np.random.seed(42)
data_groups = [np.random.normal(0, 1, 100), 
               np.random.normal(1, 1.5, 100), 
               np.random.normal(-1, 0.5, 100)]

plt.figure(figsize=(8, 6))
bp = plt.boxplot(data_groups, labels=['组1', '组2', '组3'], patch_artist=True)
colors = ['lightblue', 'lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)

plt.title('箱线图示例')
plt.ylabel('数值')
plt.grid(True, alpha=0.3)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/08_boxplot.png', dpi=300, bbox_inches='tight')
plt.close()

# 9. 时间序列图
print("9. 时间序列图")
dates = pd.date_range('2023-01-01', periods=100, freq='D')
values = np.cumsum(np.random.randn(100)) + 100

plt.figure(figsize=(12, 6))
plt.plot(dates, values, linewidth=2, color='blue')
plt.xlabel('日期')
plt.ylabel('数值')
plt.title('时间序列图')
plt.xticks(rotation=45)
plt.grid(True, alpha=0.3)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/09_timeseries.png', dpi=300, bbox_inches='tight')
plt.close()

# 10. 3D图
print("10. 3D图")
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# 创建3D数据
x = np.linspace(-5, 5, 50)
y = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x, y)
Z = np.sin(np.sqrt(X**2 + Y**2))

# 绘制3D表面
surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8)
ax.set_xlabel('X轴')
ax.set_ylabel('Y轴')
ax.set_zlabel('Z轴')
ax.set_title('3D表面图')

# 添加颜色条
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/10_3d_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 11. 样式和主题
print("11. 样式和主题")
# 使用不同的样式
plt.style.use('seaborn-v0_8')  # 使用seaborn样式

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, linewidth=3, color='#FF6B6B', label='sin(x)')
plt.fill_between(x, y, alpha=0.3, color='#FF6B6B')
plt.xlabel('X轴', fontsize=12)
plt.ylabel('Y轴', fontsize=12)
plt.title('美化后的图表', fontsize=14, fontweight='bold')
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/11_styled_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 恢复默认样式
plt.style.use('default')

# 综合练习：数据分析可视化
print("\n=== 综合练习：销售数据分析 ===")
np.random.seed(42)
months = ['1月', '2月', '3月', '4月', '5月', '6月']
products = ['产品A', '产品B', '产品C']

# 生成销售数据
sales_data = {}
for product in products:
    sales_data[product] = np.random.randint(50, 200, len(months))

# 创建综合分析图
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 产品销量对比
bottom = np.zeros(len(months))
colors = ['#FF9999', '#66B2FF', '#99FF99']
for i, (product, sales) in enumerate(sales_data.items()):
    axes[0, 0].bar(months, sales, bottom=bottom, label=product, color=colors[i])
    bottom += sales

axes[0, 0].set_title('各产品月销量堆积柱状图')
axes[0, 0].legend()
axes[0, 0].set_ylabel('销量')

# 2. 产品销量趋势
for i, (product, sales) in enumerate(sales_data.items()):
    axes[0, 1].plot(months, sales, marker='o', linewidth=2, 
                    label=product, color=colors[i])

axes[0, 1].set_title('各产品销量趋势')
axes[0, 1].legend()
axes[0, 1].set_ylabel('销量')
axes[0, 1].grid(True, alpha=0.3)

# 3. 总销量饼图
total_sales = {product: sum(sales) for product, sales in sales_data.items()}
axes[1, 0].pie(total_sales.values(), labels=total_sales.keys(), autopct='%1.1f%%', 
               colors=colors, startangle=90)
axes[1, 0].set_title('总销量占比')

# 4. 销量分布箱线图
sales_values = list(sales_data.values())
bp = axes[1, 1].boxplot(sales_values, labels=products, patch_artist=True)
for patch, color in zip(bp['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)

axes[1, 1].set_title('销量分布箱线图')
axes[1, 1].set_ylabel('销量')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/12_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n所有图表已保存到stage1-python-basics目录中！")
print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 创建动态图表（使用matplotlib.animation）")
print("2. 绘制地理数据图表")
print("3. 使用seaborn创建更美观的统计图表")
print("4. 创建交互式图表（使用plotly）")
print("5. 自定义图表样式和颜色主题")