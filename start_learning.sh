#!/bin/bash

echo "🚀 开始机器学习之旅"
echo "==================="

# 设置conda路径
export PATH="/opt/anaconda3/bin:$PATH"
source /opt/anaconda3/etc/profile.d/conda.sh

# 激活conda环境
echo "📦 激活 ml-learning 环境..."
conda activate ml-learning

# 检查环境
echo "🔍 检查环境状态:"
echo "   Python路径: $(which python)"
echo "   Python版本: $(python --version)"

# 快速验证
python -c "import numpy, pandas, matplotlib, sklearn; print('✅ 所有依赖包就绪!')" 2>/dev/null || {
    echo "❌ 依赖包检查失败！请检查环境安装。"
    exit 1
}

echo ""
echo "🎓 准备就绪！你可以开始学习了："
echo "   cd stage1-python-basics"
echo "   python 01_numpy_basics.py"
echo ""
echo "💡 提示：每次打开新终端都需要运行此脚本激活环境"