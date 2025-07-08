"""
综合数据分析项目：学生成绩分析系统
学习目标：整合NumPy、Pandas和Matplotlib，完成端到端的数据分析项目
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta

# 设置中文字体和样式
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.style.use('seaborn-v0_8')

print("=== 学生成绩分析系统 ===\n")

# 1. 数据生成
print("1. 生成模拟学生数据...")
np.random.seed(42)

# 学生基本信息
n_students = 500
student_ids = [f'S{i:04d}' for i in range(1, n_students + 1)]
names = [f'学生{i}' for i in range(1, n_students + 1)]
classes = np.random.choice(['高一1班', '高一2班', '高一3班', '高一4班', '高一5班'], n_students)
genders = np.random.choice(['男', '女'], n_students)

# 各科成绩生成（考虑不同科目的难度差异）
subjects = ['语文', '数学', '英语', '物理', '化学', '生物']
scores = {}

# 设置不同科目的均值和标准差
subject_params = {
    '语文': (78, 12),
    '数学': (75, 15),
    '英语': (80, 14),
    '物理': (72, 16),
    '化学': (76, 13),
    '生物': (74, 14)
}

for subject in subjects:
    mean, std = subject_params[subject]
    # 生成成绩，限制在0-100之间
    subject_scores = np.random.normal(mean, std, n_students)
    subject_scores = np.clip(subject_scores, 0, 100).round(1)
    scores[subject] = subject_scores

# 创建DataFrame
data = {
    '学号': student_ids,
    '姓名': names,
    '班级': classes,
    '性别': genders
}
data.update(scores)

df = pd.DataFrame(data)

# 计算总分和平均分
df['总分'] = df[subjects].sum(axis=1)
df['平均分'] = df[subjects].mean(axis=1).round(2)

print(f"成功生成{n_students}名学生的成绩数据")
print(f"数据形状: {df.shape}")
print("\n前5行数据:")
print(df.head())

# 2. 数据探索性分析
print("\n2. 数据探索性分析")
print("\n基本统计信息:")
print(df[subjects + ['总分', '平均分']].describe())

print("\n各班级人数分布:")
print(df['班级'].value_counts())

print("\n性别分布:")
print(df['性别'].value_counts())

# 3. 数据质量检查
print("\n3. 数据质量检查")
print(f"缺失值检查:\n{df.isnull().sum()}")
print(f"\n重复值检查: {df.duplicated().sum()}条重复记录")

# 检查异常值
print("\n异常值检查（成绩超出0-100范围）:")
for subject in subjects:
    outliers = df[(df[subject] < 0) | (df[subject] > 100)]
    if len(outliers) > 0:
        print(f"{subject}: {len(outliers)}个异常值")
    else:
        print(f"{subject}: 无异常值")

# 4. 单科目分析
print("\n4. 单科目分析")

# 创建各科成绩分布图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, subject in enumerate(subjects):
    # 直方图
    axes[i].hist(df[subject], bins=20, alpha=0.7, color=plt.cm.Set3(i))
    axes[i].axvline(df[subject].mean(), color='red', linestyle='--', 
                   label=f'均值: {df[subject].mean():.1f}')
    axes[i].set_title(f'{subject}成绩分布')
    axes[i].set_xlabel('成绩')
    axes[i].set_ylabel('人数')
    axes[i].legend()
    axes[i].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_01_score_distribution.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 5. 班级分析
print("\n5. 班级分析")

# 计算各班级各科平均分
class_avg = df.groupby('班级')[subjects].mean()
print("各班级各科平均分:")
print(class_avg.round(2))

# 班级对比雷达图
fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

angles = np.linspace(0, 2 * np.pi, len(subjects), endpoint=False).tolist()
angles += angles[:1]  # 闭合图形

colors = plt.cm.Set1(np.linspace(0, 1, len(class_avg)))

for i, (class_name, scores) in enumerate(class_avg.iterrows()):
    values = scores.tolist()
    values += values[:1]  # 闭合图形
    
    ax.plot(angles, values, 'o-', linewidth=2, label=class_name, color=colors[i])
    ax.fill(angles, values, alpha=0.25, color=colors[i])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(subjects)
ax.set_ylim(0, 100)
ax.set_title('各班级各科成绩雷达图', y=1.08, fontsize=14)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
ax.grid(True)

plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_02_class_radar.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 6. 性别差异分析
print("\n6. 性别差异分析")

# 计算性别差异
gender_avg = df.groupby('性别')[subjects].mean()
print("不同性别各科平均分:")
print(gender_avg.round(2))

# 性别对比箱线图
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

for i, subject in enumerate(subjects):
    df.boxplot(column=subject, by='性别', ax=axes[i])
    axes[i].set_title(f'{subject}成绩性别差异')
    axes[i].set_xlabel('性别')
    axes[i].set_ylabel('成绩')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_03_gender_comparison.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 7. 相关性分析
print("\n7. 学科相关性分析")

# 计算相关矩阵
correlation_matrix = df[subjects].corr()
print("学科成绩相关矩阵:")
print(correlation_matrix.round(3))

# 绘制相关性热力图
plt.figure(figsize=(10, 8))
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
sns.heatmap(correlation_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
            square=True, fmt='.3f', cbar_kws={"shrink": .8})
plt.title('学科成绩相关性热力图')
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_04_correlation_heatmap.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 8. 成绩等级分析
print("\n8. 成绩等级分析")

# 定义等级
def get_grade(score):
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

# 为每科添加等级
for subject in subjects:
    df[f'{subject}_等级'] = df[subject].apply(get_grade)

# 统计各等级人数
grade_stats = {}
for subject in subjects:
    grade_stats[subject] = df[f'{subject}_等级'].value_counts().sort_index()

# 可视化等级分布
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

grades = ['A', 'B', 'C', 'D', 'F']
colors = ['#2E8B57', '#32CD32', '#FFD700', '#FF8C00', '#DC143C']

for i, subject in enumerate(subjects):
    grade_counts = [grade_stats[subject].get(grade, 0) for grade in grades]
    bars = axes[i].bar(grades, grade_counts, color=colors)
    axes[i].set_title(f'{subject}等级分布')
    axes[i].set_xlabel('等级')
    axes[i].set_ylabel('人数')
    
    # 在柱子上显示数量
    for bar, count in zip(bars, grade_counts):
        if count > 0:
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                        str(count), ha='center', va='bottom')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_05_grade_distribution.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 9. 综合排名分析
print("\n9. 综合排名分析")

# 总分排名
df['总分排名'] = df['总分'].rank(ascending=False, method='min').astype(int)
df['平均分排名'] = df['平均分'].rank(ascending=False, method='min').astype(int)

# Top 10学生
top_10 = df.nsmallest(10, '总分排名')[['姓名', '班级', '总分', '平均分', '总分排名']]
print("总分前10名学生:")
print(top_10)

# 各班级前3名
print("\n各班级前3名学生:")
for class_name in df['班级'].unique():
    class_top3 = df[df['班级'] == class_name].nsmallest(3, '总分排名')[['姓名', '总分', '总分排名']]
    print(f"\n{class_name}:")
    print(class_top3)

# 10. 数据导出
print("\n10. 数据导出")

# 保存完整数据
df.to_csv('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/student_scores_complete.csv', 
          index=False, encoding='utf-8-sig')

# 保存统计摘要
summary_data = {
    '班级平均分': class_avg,
    '性别平均分': gender_avg,
    '学科相关性': correlation_matrix
}

with pd.ExcelWriter('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_summary.xlsx') as writer:
    for sheet_name, data in summary_data.items():
        data.to_excel(writer, sheet_name=sheet_name)

print("完整数据已保存为: student_scores_complete.csv")
print("分析摘要已保存为: analysis_summary.xlsx")

# 11. 生成分析报告
print("\n11. 生成分析报告")

report = f"""
学生成绩分析报告
================

分析时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
数据规模: {n_students}名学生，{len(subjects)}个科目

一、基本情况
------------
1. 学生总数: {n_students}人
2. 班级数量: {df['班级'].nunique()}个
3. 性别分布: 男生{(df['性别']=='男').sum()}人，女生{(df['性别']=='女').sum()}人

二、成绩概况
------------
1. 总分范围: {df['总分'].min():.1f} - {df['总分'].max():.1f}
2. 总分平均: {df['总分'].mean():.1f}
3. 平均分范围: {df['平均分'].min():.1f} - {df['平均分'].max():.1f}

三、各科情况
------------
"""

for subject in subjects:
    mean_score = df[subject].mean()
    std_score = df[subject].std()
    max_score = df[subject].max()
    min_score = df[subject].min()
    
    report += f"""
{subject}:
  平均分: {mean_score:.1f}
  标准差: {std_score:.1f}
  最高分: {max_score:.1f}
  最低分: {min_score:.1f}
  及格率: {(df[subject] >= 60).mean()*100:.1f}%
"""

report += f"""

四、班级表现
------------
总分最高班级: {class_avg.mean(axis=1).idxmax()}
总分最低班级: {class_avg.mean(axis=1).idxmin()}

五、分析结论
------------
1. 整体成绩分布较为正常
2. 各科目之间存在一定相关性
3. 班级间成绩差异不大
4. 性别差异需要进一步关注

详细图表请查看生成的PNG文件。
"""

# 保存报告
with open('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage1-python-basics/analysis_report.txt', 'w', encoding='utf-8') as f:
    f.write(report)

print("分析报告已保存为: analysis_report.txt")

print("\n=== 项目总结 ===")
print("✅ 数据生成和清洗")
print("✅ 探索性数据分析")
print("✅ 多维度统计分析")
print("✅ 数据可视化")
print("✅ 结果导出和报告生成")
print("\n恭喜！你已经完成了一个完整的数据分析项目！")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 添加时间序列分析（模拟多次考试成绩）")
print("2. 实现成绩预测模型")
print("3. 添加更多的统计检验")
print("4. 创建交互式仪表板")
print("5. 进行A/B测试分析")