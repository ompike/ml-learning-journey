# 项目环境设置指南 🛠️

## 虚拟环境设置（推荐）

### 方法1：使用venv（Python内置）

1. **创建虚拟环境**：
   ```bash
   cd ml-learning-journey
   python -m venv ml-env
   ```

2. **激活虚拟环境**：
   ```bash
   # Windows
   ml-env\Scripts\activate
   
   # macOS/Linux
   source ml-env/bin/activate
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **验证安装**：
   ```bash
   pip list
   ```

5. **退出虚拟环境**：
   ```bash
   deactivate
   ```

### 方法2：使用conda

1. **创建conda环境**：
   ```bash
   conda create -n ml-learning python=3.9
   ```

2. **激活环境**：
   ```bash
   conda activate ml-learning
   ```

3. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

4. **退出环境**：
   ```bash
   conda deactivate
   ```

### 方法3：使用pipenv

1. **安装pipenv**：
   ```bash
   pip install pipenv
   ```

2. **创建环境并安装依赖**：
   ```bash
   cd ml-learning-journey
   pipenv install -r requirements.txt
   ```

3. **激活环境**：
   ```bash
   pipenv shell
   ```

## 环境验证

运行以下命令验证环境是否正确设置：

```bash
python -c "import numpy, pandas, matplotlib, sklearn; print('环境设置成功！')"
```

## 开发工具推荐

### IDE/编辑器
- **VS Code**：推荐安装Python扩展
- **PyCharm**：专业Python IDE
- **Jupyter Notebook**：交互式开发

### Jupyter Notebook设置
```bash
# 在虚拟环境中安装
pip install jupyter

# 启动Jupyter
jupyter notebook
```

### VS Code设置
1. 安装Python扩展
2. 选择Python解释器（Ctrl+Shift+P -> Python: Select Interpreter）
3. 选择虚拟环境中的Python

## 常见问题解决

### 1. 权限问题
```bash
# 如果遇到权限问题，使用--user参数
pip install --user -r requirements.txt
```

### 2. 镜像源设置（国内用户）
```bash
# 使用清华镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple/
```

### 3. 依赖冲突
```bash
# 查看依赖冲突
pip check

# 升级pip
pip install --upgrade pip
```

## 项目结构说明

```
ml-learning-journey/
├── ml-env/                    # 虚拟环境目录（git忽略）
├── stage1-python-basics/      # 阶段1：Python基础
├── stage2-math-fundamentals/  # 阶段2：数学基础
├── stage3-classic-algorithms/ # 阶段3：经典算法
├── stage4-sklearn-practice/   # 阶段4：sklearn实践
├── stage5-deep-learning/      # 阶段5：深度学习
├── datasets/                  # 数据集存放
├── utils/                     # 工具函数
├── requirements.txt           # 依赖包列表
├── setup.md                   # 环境设置指南
└── README.md                  # 项目说明
```

## 学习流程

1. **环境准备**：按照上述步骤设置虚拟环境
2. **开始学习**：从stage1开始，逐步完成每个阶段
3. **记录笔记**：在每个阶段目录下创建自己的笔记文件
4. **实践项目**：完成每个阶段的综合项目

## 提示

- 每次开始学习前都要激活虚拟环境
- 如果添加新的依赖包，记得更新requirements.txt
- 定期备份你的学习笔记和代码修改