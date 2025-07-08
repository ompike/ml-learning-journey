# 📚 如何使用这个机器学习学习项目

## 🎯 最简单的使用方法

### 方法1：使用运行脚本（推荐）
```bash
# 进入项目目录
cd ml-learning-journey

# 运行指定的练习
./run_lesson.sh stage1-python-basics/01_numpy_basics.py
./run_lesson.sh stage1-python-basics/02_pandas_practice.py
./run_lesson.sh stage1-python-basics/03_matplotlib_visualization.py
./run_lesson.sh stage1-python-basics/04_data_analysis_project.py
```

### 方法2：手动激活环境
```bash
# 1. 设置conda环境
export PATH="/opt/anaconda3/bin:$PATH"
source /opt/anaconda3/etc/profile.d/conda.sh

# 2. 激活ml-learning环境
conda activate ml-learning

# 3. 验证环境
python -c "import numpy; print('✅ 环境OK')"

# 4. 运行练习
python stage1-python-basics/01_numpy_basics.py
```

## 📋 学习顺序

### 阶段1：Python基础和数据处理
```bash
./run_lesson.sh stage1-python-basics/01_numpy_basics.py
./run_lesson.sh stage1-python-basics/02_pandas_practice.py
./run_lesson.sh stage1-python-basics/03_matplotlib_visualization.py
./run_lesson.sh stage1-python-basics/04_data_analysis_project.py
```

### 阶段2：数学基础实现
```bash
./run_lesson.sh stage2-math-fundamentals/01_linear_algebra.py
# 更多文件即将创建...
```

### 阶段3：经典算法从零实现
```bash
./run_lesson.sh stage3-classic-algorithms/01_linear_regression.py
# 更多文件即将创建...
```

## ✅ 成功运行的标志

当你运行脚本时，应该看到：
```
🔄 激活ml-learning环境...
📍 当前Python: /opt/anaconda3/envs/ml-learning/bin/python
📊 Python版本: Python 3.9.23
✅ NumPy可用
```

## ❌ 如果遇到问题

### 问题1：bash: ./run_lesson.sh: Permission denied
```bash
chmod +x run_lesson.sh
```

### 问题2：conda: command not found
```bash
# 初始化conda
/opt/anaconda3/bin/conda init
source ~/.zshrc
```

### 问题3：ModuleNotFoundError
- 确保使用 `./run_lesson.sh` 运行
- 或者手动激活环境后再运行

## 🎓 学习建议

1. **按顺序学习**：从stage1开始
2. **理解代码**：不要只是运行，要理解每行代码
3. **做笔记**：记录重要概念
4. **完成练习**：每个文件末尾都有练习任务
5. **动手实验**：修改参数，观察结果

## 🆘 获取帮助

- 查看 `troubleshooting.md` 解决常见问题
- 每个阶段都有详细的 `README.md`
- 代码中有详细注释

---

**现在开始你的机器学习学习之旅吧！🚀**