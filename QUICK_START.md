# 🚀 快速开始指南

## 重要：每次学习前必须执行这些步骤！

### 1️⃣ 激活环境
```bash
# 进入项目目录
cd ml-learning-journey

# 运行激活脚本（自动激活conda环境）
./activate_env.sh
```

### 2️⃣ 手动激活（如果脚本不工作）
```bash
# 激活conda环境
source /opt/anaconda3/etc/profile.d/conda.sh
conda activate ml-learning

# 验证环境
python -c "import numpy; print('✅ 环境OK')"
```

### 3️⃣ 开始学习
```bash
# 阶段1：Python基础
cd stage1-python-basics
python 01_numpy_basics.py
python 02_pandas_practice.py
python 03_matplotlib_visualization.py
python 04_data_analysis_project.py

# 阶段2：数学基础
cd ../stage2-math-fundamentals
python 01_linear_algebra.py
# ... 更多文件

# 阶段3：算法实现
cd ../stage3-classic-algorithms
python 01_linear_regression.py
# ... 更多文件
```

## 🔧 故障排除

### 问题：ModuleNotFoundError: No module named 'numpy'
**解决方案：**
1. 确保激活了正确的conda环境
2. 运行 `./activate_env.sh` 脚本
3. 检查环境状态：`which python`

### 问题：conda命令不可用
**解决方案：**
```bash
# 初始化conda
/opt/anaconda3/bin/conda init

# 重新加载shell
source ~/.zshrc

# 或者直接使用完整路径
source /opt/anaconda3/etc/profile.d/conda.sh
```

### 问题：权限错误
**解决方案：**
```bash
# 给脚本添加执行权限
chmod +x activate_env.sh
```

## ✅ 验证环境
运行以下命令确保环境正确：
```bash
python -c "
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
print('🎉 所有包都可用！')
print(f'NumPy: {np.__version__}')
print(f'Pandas: {pd.__version__}')
print(f'Matplotlib: {plt.matplotlib.__version__}')
print(f'Scikit-learn: {sklearn.__version__}')
"
```

## 📚 学习建议

1. **按顺序学习**：从stage1开始，不要跳跃
2. **动手实践**：每个例子都要运行并理解
3. **做笔记**：记录重要概念和代码片段
4. **完成练习**：每个文件末尾都有练习任务
5. **提问思考**：理解原理，不要只是运行代码

## 🆘 需要帮助？

如果遇到问题，请：
1. 检查 `troubleshooting.md` 文件
2. 确保按照此快速指南操作
3. 验证环境是否正确激活

---

**记住：每次开始学习前都要激活环境！**