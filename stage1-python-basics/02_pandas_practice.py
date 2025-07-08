"""
Pandas数据分析练习
学习目标：掌握Pandas的DataFrame操作、数据清洗和分析
"""

import pandas as pd
import numpy as np

print("=== Pandas数据分析练习 ===\n")

# 1. 创建DataFrame
print("1. 创建DataFrame")
# 从字典创建
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [25, 30, 35, 28, 32],
    'city': ['北京', '上海', '广州', '深圳', '杭州'],
    'salary': [8000, 12000, 15000, 9000, 11000]
}
df = pd.DataFrame(data)
print("学生信息DataFrame:")
print(df)
print(f"\n数据形状: {df.shape}")
print(f"数据类型:\n{df.dtypes}")

# 2. 基本信息查看
print("\n2. 基本信息查看")
print("前3行数据:")
print(df.head(3))
print("\n后2行数据:")
print(df.tail(2))
print("\n基本统计信息:")
print(df.describe())

# 3. 数据选择和筛选
print("\n3. 数据选择和筛选")
# 选择列
print("选择name列:")
print(df['name'])
print("\n选择多列:")
print(df[['name', 'age']])

# 条件筛选
print("\n年龄大于30的记录:")
older_than_30 = df[df['age'] > 30]
print(older_than_30)

print("\n薪资在9000-12000之间的记录:")
salary_range = df[(df['salary'] >= 9000) & (df['salary'] <= 12000)]
print(salary_range)

# 4. 数据添加和修改
print("\n4. 数据添加和修改")
# 添加新列
df['experience'] = [2, 5, 8, 3, 6]
df['bonus'] = df['salary'] * 0.1
print("添加经验和奖金列:")
print(df)

# 修改数据
df.loc[0, 'age'] = 26  # 修改第一行的年龄
print("\n修改后的数据:")
print(df)

# 5. 数据分组和聚合
print("\n5. 数据分组和聚合")
# 创建更多数据进行分组演示
extended_data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace'],
    'department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'IT', 'Finance'],
    'age': [25, 30, 35, 28, 32, 29, 31],
    'salary': [8000, 12000, 15000, 9000, 11000, 13000, 10000]
}
df_extended = pd.DataFrame(extended_data)
print("扩展数据:")
print(df_extended)

# 按部门分组
print("\n按部门分组的平均薪资:")
dept_avg_salary = df_extended.groupby('department')['salary'].mean()
print(dept_avg_salary)

print("\n按部门分组的详细统计:")
dept_stats = df_extended.groupby('department').agg({
    'age': 'mean',
    'salary': ['mean', 'max', 'min']
})
print(dept_stats)

# 6. 数据清洗
print("\n6. 数据清洗")
# 创建包含缺失值的数据
dirty_data = {
    'name': ['Alice', 'Bob', None, 'David', 'Eve'],
    'age': [25, None, 35, 28, 32],
    'salary': [8000, 12000, 15000, None, 11000],
    'city': ['北京', '上海', '广州', '深圳', None]
}
df_dirty = pd.DataFrame(dirty_data)
print("包含缺失值的数据:")
print(df_dirty)
print(f"\n缺失值统计:\n{df_dirty.isnull().sum()}")

# 处理缺失值
print("\n删除包含缺失值的行:")
df_clean = df_dirty.dropna()
print(df_clean)

print("\n用均值填充数值列的缺失值:")
df_filled = df_dirty.copy()
df_filled['age'].fillna(df_filled['age'].mean(), inplace=True)
df_filled['salary'].fillna(df_filled['salary'].mean(), inplace=True)
print(df_filled)

# 7. 数据排序
print("\n7. 数据排序")
print("按薪资排序（升序）:")
df_sorted = df_extended.sort_values('salary')
print(df_sorted)

print("\n按部门和薪资排序:")
df_multi_sorted = df_extended.sort_values(['department', 'salary'], ascending=[True, False])
print(df_multi_sorted)

# 8. 数据透视表
print("\n8. 数据透视表")
pivot_table = df_extended.pivot_table(
    values='salary',
    index='department',
    aggfunc=['mean', 'count']
)
print("薪资透视表:")
print(pivot_table)

# 9. 字符串操作
print("\n9. 字符串操作")
df_text = pd.DataFrame({
    'name': ['Alice Smith', 'Bob Johnson', 'Charlie Brown'],
    'email': ['alice@email.com', 'bob@email.com', 'charlie@email.com']
})
print("原始文本数据:")
print(df_text)

# 字符串分割
df_text['first_name'] = df_text['name'].str.split().str[0]
df_text['last_name'] = df_text['name'].str.split().str[1]
print("\n分割姓名后:")
print(df_text)

# 10. 数据合并
print("\n10. 数据合并")
df1 = pd.DataFrame({
    'id': [1, 2, 3],
    'name': ['Alice', 'Bob', 'Charlie']
})
df2 = pd.DataFrame({
    'id': [1, 2, 4],
    'score': [85, 92, 78]
})
print("DataFrame 1:")
print(df1)
print("\nDataFrame 2:")
print(df2)

# 内连接
inner_join = pd.merge(df1, df2, on='id', how='inner')
print("\n内连接结果:")
print(inner_join)

# 外连接
outer_join = pd.merge(df1, df2, on='id', how='outer')
print("\n外连接结果:")
print(outer_join)

# 练习任务
print("\n=== 练习任务 ===")
print("1. 创建一个包含100行学生数据的DataFrame")
np.random.seed(42)
student_data = {
    'student_id': range(1, 101),
    'name': [f'Student_{i}' for i in range(1, 101)],
    'math_score': np.random.randint(60, 100, 100),
    'english_score': np.random.randint(60, 100, 100),
    'class': np.random.choice(['A', 'B', 'C'], 100)
}
students_df = pd.DataFrame(student_data)
print("学生数据样本:")
print(students_df.head())

print("\n2. 计算每个班级的平均分")
class_avg = students_df.groupby('class')[['math_score', 'english_score']].mean()
print(class_avg)

print("\n3. 找出总分最高的前10名学生")
students_df['total_score'] = students_df['math_score'] + students_df['english_score']
top_10 = students_df.nlargest(10, 'total_score')
print(top_10[['name', 'math_score', 'english_score', 'total_score']])

# TODO: 尝试完成以下扩展练习
print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 读取CSV文件并进行数据分析")
print("2. 使用apply()函数进行复杂的数据变换")
print("3. 处理日期时间数据")
print("4. 创建多层索引的DataFrame")
print("5. 使用crosstab()创建交叉表")