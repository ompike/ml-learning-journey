#!/bin/bash

# 检查参数
if [ $# -eq 0 ]; then
    echo "使用方法: ./run_lesson.sh <python文件名>"
    echo "例如: ./run_lesson.sh stage1-python-basics/01_numpy_basics.py"
    echo ""
    echo "可用的练习文件："
    echo "  stage1-python-basics/01_numpy_basics.py"
    echo "  stage1-python-basics/02_pandas_practice.py"
    echo "  stage1-python-basics/03_matplotlib_visualization.py"
    echo "  stage1-python-basics/04_data_analysis_project.py"
    exit 1
fi

SCRIPT_FILE=$1

# 设置conda环境
export PATH="/opt/anaconda3/bin:$PATH"
source /opt/anaconda3/etc/profile.d/conda.sh

# 激活ml-learning环境
echo "🔄 激活ml-learning环境..."
conda activate ml-learning

# 检查环境
echo "📍 当前Python: $(which python)"
echo "📊 Python版本: $(python --version)"

# 验证numpy可用
if python -c "import numpy" 2>/dev/null; then
    echo "✅ NumPy可用"
else
    echo "❌ NumPy不可用，请检查环境"
    exit 1
fi

echo ""
echo "🚀 运行 $SCRIPT_FILE..."
echo "================================="

# 运行指定的Python文件
python "$SCRIPT_FILE"

echo ""
echo "================================="
echo "✅ 运行完成！"