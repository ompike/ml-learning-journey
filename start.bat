@echo off
echo === 机器学习学习之旅 环境设置 ===
echo.

REM 检查Python版本
echo 检查Python版本...
python --version
echo.

REM 创建虚拟环境
echo 创建虚拟环境...
python -m venv ml-env
echo.

REM 激活虚拟环境
echo 激活虚拟环境...
call ml-env\Scripts\activate.bat
echo.

REM 升级pip
echo 升级pip...
python -m pip install --upgrade pip
echo.

REM 安装依赖
echo 安装项目依赖...
pip install -r requirements.txt
echo.

REM 验证安装
echo 验证安装...
python -c "import numpy, pandas, matplotlib, sklearn; print('✅ 环境设置成功！')"
echo.

echo.
echo 🎉 环境设置完成！现在可以开始学习了：
echo   cd stage1-python-basics
echo   python 01_numpy_basics.py
echo.
pause