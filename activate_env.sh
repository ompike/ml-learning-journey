#!/bin/bash

echo "=== 激活机器学习环境 ==="

# 确保conda可用
export PATH="/opt/anaconda3/bin:$PATH"
source /opt/anaconda3/etc/profile.d/conda.sh

# 激活conda环境
echo "激活conda环境: ml-learning"
conda activate ml-learning

# 检查环境
echo "当前Python路径: $(which python)"
echo "当前Python版本: $(python --version)"

# 验证包安装
echo "验证依赖包..."
python -c "
try:
    import numpy
    print('✅ NumPy 版本:', numpy.__version__)
except ImportError:
    print('❌ NumPy 未安装')

try:
    import pandas
    print('✅ Pandas 版本:', pandas.__version__)
except ImportError:
    print('❌ Pandas 未安装')

try:
    import matplotlib
    print('✅ Matplotlib 版本:', matplotlib.__version__)
except ImportError:
    print('❌ Matplotlib 未安装')

try:
    import sklearn
    print('✅ Scikit-learn 版本:', sklearn.__version__)
except ImportError:
    print('❌ Scikit-learn 未安装')
"

echo ""
echo "🎉 环境激活完成！现在可以运行学习脚本了："
echo "   cd stage1-python-basics"
echo "   python 01_numpy_basics.py"