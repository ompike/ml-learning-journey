# 机器学习学习之旅 - 简易使用指南 🚀

## 🎯 每次学习前只需要两步：

### 1️⃣ 激活环境
```bash
cd ml-learning-journey
./start_learning.sh
```

### 2️⃣ 开始学习
```bash
# 阶段1：Python基础
cd stage1-python-basics
python 01_numpy_basics.py

# 继续其他练习...
python 02_pandas_practice.py
python 03_matplotlib_visualization.py
python 04_data_analysis_project.py
```

## ✅ 成功标志

看到这些信息说明环境正确：
- ✅ 所有依赖包就绪!
- Python路径: `/opt/anaconda3/envs/ml-learning/bin/python`
- Python版本: `Python 3.9.23`

## ❌ 如果还是报错

**问题：** `ModuleNotFoundError: No module named 'numpy'`

**解决方案：**
1. 确保运行了 `./start_learning.sh`
2. 确保看到 "✅ 所有依赖包就绪!"
3. 在同一个终端窗口中运行Python脚本

**手动激活命令：**
```bash
export PATH="/opt/anaconda3/bin:$PATH"
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ml-learning
```

## 📚 学习路径

1. **stage1-python-basics** - Python基础和数据处理
2. **stage2-math-fundamentals** - 数学基础实现
3. **stage3-classic-algorithms** - 经典算法从零实现
4. **stage4-sklearn-practice** - scikit-learn实践
5. **stage5-deep-learning** - 深度学习基础

## 💡 重要提醒

- **每次打开新终端**都需要重新激活环境
- **确保在项目根目录**运行激活脚本
- **按顺序完成**各个阶段的学习

---

现在你可以开始你的机器学习学习之旅了！🎓