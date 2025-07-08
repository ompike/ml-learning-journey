# 常见问题解决方案 🔧

## 问题1：ModuleNotFoundError: No module named 'numpy'

### 原因
- 没有激活虚拟环境
- 没有安装依赖包
- 在错误的环境中运行代码

### 解决方案

#### 步骤1：激活虚拟环境
```bash
# 进入项目目录
cd ml-learning-journey

# 激活虚拟环境
source ml-env/bin/activate  # macOS/Linux
# 或
ml-env\Scripts\activate     # Windows

# 验证环境
which python
python --version
```

#### 步骤2：安装依赖包

**方案A：使用完整依赖**
```bash
pip install -r requirements.txt
```

**方案B：网络问题时使用最小依赖**
```bash
pip install -r requirements-minimal.txt
```

**方案C：逐个安装核心包**
```bash
pip install numpy pandas matplotlib seaborn scipy scikit-learn
```

**方案D：使用国内镜像源**
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 步骤3：验证安装
```bash
python -c "import numpy, pandas, matplotlib, sklearn; print('✅ 安装成功！')"
```

## 问题2：网络连接错误

### 症状
- SSL证书错误
- 连接超时
- 下载失败

### 解决方案

#### 方案1：使用国内镜像源
```bash
# 临时使用
pip install numpy -i https://pypi.tuna.tsinghua.edu.cn/simple/

# 永久配置
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple/
```

#### 方案2：使用conda安装（推荐，网络问题时首选）
```bash
# 创建conda环境
conda create -n ml-learning python=3.9 -y

# 激活环境  
conda activate ml-learning

# 安装基础包
conda install numpy pandas matplotlib seaborn scipy scikit-learn -y

# 验证安装
python -c "import numpy, pandas, matplotlib, sklearn; print('✅ 安装成功！')"
```

#### 方案3：离线安装
```bash
# 下载wheel文件后离线安装
pip install --find-links /path/to/wheels numpy
```

## 问题3：权限错误

### 解决方案
```bash
# 使用--user参数
pip install --user numpy pandas matplotlib

# 或修改虚拟环境权限
chmod -R 755 ml-env/
```

## 问题4：虚拟环境损坏

### 解决方案
```bash
# 删除现有虚拟环境
rm -rf ml-env

# 重新创建
python -m venv ml-env
source ml-env/bin/activate
pip install -r requirements-minimal.txt
```

## 问题5：Python版本兼容性

### 解决方案
```bash
# 检查Python版本
python --version

# 如果版本过低，创建指定版本的虚拟环境
python3.9 -m venv ml-env
# 或使用conda
conda create -n ml-learning python=3.9
```

## 快速诊断脚本

创建 `diagnose.py` 文件：

```python
import sys
import subprocess

def check_python():
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")

def check_packages():
    required_packages = ['numpy', 'pandas', 'matplotlib', 'sklearn']
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package} 已安装")
        except ImportError:
            print(f"❌ {package} 未安装")

def check_pip():
    try:
        result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
        print(f"pip可用，已安装 {len(result.stdout.split())} 个包")
    except Exception as e:
        print(f"❌ pip不可用: {e}")

if __name__ == "__main__":
    print("=== 环境诊断 ===")
    check_python()
    print()
    check_packages()
    print()
    check_pip()
```

运行诊断：
```bash
python diagnose.py
```

## 完整重装流程

如果以上方案都不行，完整重装：

```bash
# 1. 删除虚拟环境
rm -rf ml-env

# 2. 重新创建虚拟环境
python -m venv ml-env

# 3. 激活环境
source ml-env/bin/activate

# 4. 升级pip
python -m pip install --upgrade pip

# 5. 安装核心包
pip install numpy pandas matplotlib seaborn scipy scikit-learn

# 6. 验证安装
python -c "import numpy; print('NumPy版本:', numpy.__version__)"
```

## 联系支持

如果问题仍然存在，请提供以下信息：
- 操作系统版本
- Python版本
- 完整的错误信息
- 虚拟环境状态 (`which python`)
- 已安装包列表 (`pip list`)

---

记住：每次开始学习前都要先激活虚拟环境！