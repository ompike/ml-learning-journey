# 快速开始脚本

echo '=== 机器学习学习之旅 环境设置 ==='

# 检查Python版本
python --version

# 创建虚拟环境
echo '创建虚拟环境...'
python -m venv ml-env

# 激活虚拟环境
echo '激活虚拟环境...'
source ml-env/bin/activate

# 升级pip
echo '升级pip...'
pip install --upgrade pip

# 安装依赖
echo '安装项目依赖...'
pip install -r requirements.txt

# 验证安装
echo '验证安装...'
python -c "import numpy, pandas, matplotlib, sklearn; print('✅ 环境设置成功！')"

echo ''
echo '🎉 环境设置完成！现在可以开始学习了：'
echo '  cd stage1-python-basics'
echo '  python 01_numpy_basics.py'
