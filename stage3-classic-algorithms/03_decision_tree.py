"""
决策树从零实现
学习目标：理解决策树的分裂准则、构建过程和剪枝方法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
import graphviz

print("=== 决策树从零实现 ===\n")

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 分裂特征
        self.threshold = threshold  # 分裂阈值
        self.left = left           # 左子树
        self.right = right         # 右子树
        self.value = value         # 叶节点的值

class DecisionTreeScratch:
    def __init__(self, max_depth=100, min_samples_split=2, min_samples_leaf=1, 
                 criterion='gini', max_features=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.max_features = max_features
        self.root = None
        self.feature_importances_ = None
        
    def fit(self, X, y):
        """训练决策树"""
        self.n_features = X.shape[1]
        self.n_classes = len(np.unique(y))
        self.feature_importances_ = np.zeros(self.n_features)
        
        # 构建决策树
        self.root = self._build_tree(X, y, depth=0)
        
        # 归一化特征重要性
        if np.sum(self.feature_importances_) > 0:
            self.feature_importances_ /= np.sum(self.feature_importances_)
        
        return self
    
    def _build_tree(self, X, y, depth):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_samples < self.min_samples_split or 
            n_classes == 1):
            return Node(value=self._most_common_label(y))
        
        # 选择最佳分裂
        best_feature, best_threshold = self._best_split(X, y)
        
        if best_feature is None:
            return Node(value=self._most_common_label(y))
        
        # 分裂数据
        left_indices = X[:, best_feature] <= best_threshold
        right_indices = ~left_indices
        
        # 检查分裂后样本数量
        if np.sum(left_indices) < self.min_samples_leaf or np.sum(right_indices) < self.min_samples_leaf:
            return Node(value=self._most_common_label(y))
        
        # 递归构建左右子树
        left_subtree = self._build_tree(X[left_indices], y[left_indices], depth + 1)
        right_subtree = self._build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return Node(feature=best_feature, threshold=best_threshold, 
                   left=left_subtree, right=right_subtree)
    
    def _best_split(self, X, y):
        """找到最佳分裂点"""
        best_gain = -1
        best_feature = None
        best_threshold = None
        
        # 选择要考虑的特征
        if self.max_features is None:
            features = range(self.n_features)
        else:
            features = np.random.choice(self.n_features, 
                                      min(self.max_features, self.n_features), 
                                      replace=False)
        
        current_impurity = self._impurity(y)
        
        for feature in features:
            thresholds = np.unique(X[:, feature])
            
            for threshold in thresholds:
                # 分裂数据
                left_indices = X[:, feature] <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                # 计算信息增益
                left_y, right_y = y[left_indices], y[right_indices]
                n_left, n_right = len(left_y), len(right_y)
                n_total = len(y)
                
                # 加权平均杂质
                weighted_impurity = (n_left / n_total) * self._impurity(left_y) + \
                                  (n_right / n_total) * self._impurity(right_y)
                
                # 信息增益
                gain = current_impurity - weighted_impurity
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        # 更新特征重要性
        if best_feature is not None:
            self.feature_importances_[best_feature] += best_gain
            
        return best_feature, best_threshold
    
    def _impurity(self, y):
        """计算杂质"""
        if len(y) == 0:
            return 0
        
        if self.criterion == 'gini':
            return self._gini_impurity(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError("Criterion must be 'gini' or 'entropy'")
    
    def _gini_impurity(self, y):
        """计算基尼杂质"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        return 1 - np.sum(probabilities ** 2)
    
    def _entropy(self, y):
        """计算熵"""
        _, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        # 避免log(0)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _most_common_label(self, y):
        """返回最常见的标签"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def predict(self, X):
        """预测"""
        return np.array([self._predict_sample(sample, self.root) for sample in X])
    
    def _predict_sample(self, sample, node):
        """预测单个样本"""
        if node.value is not None:
            return node.value
        
        if sample[node.feature] <= node.threshold:
            return self._predict_sample(sample, node.left)
        else:
            return self._predict_sample(sample, node.right)
    
    def print_tree(self, node=None, depth=0):
        """打印决策树结构"""
        if node is None:
            node = self.root
        
        if node.value is not None:
            print("  " * depth + f"Predict: {node.value}")
        else:
            print("  " * depth + f"Feature_{node.feature} <= {node.threshold:.3f}")
            print("  " * depth + "├── True:")
            self.print_tree(node.left, depth + 1)
            print("  " * depth + "└── False:")
            self.print_tree(node.right, depth + 1)

class RandomForestScratch:
    def __init__(self, n_estimators=100, max_depth=None, min_samples_split=2,
                 min_samples_leaf=1, max_features='sqrt', bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        
    def fit(self, X, y):
        """训练随机森林"""
        if self.random_state:
            np.random.seed(self.random_state)
        
        n_samples, n_features = X.shape
        
        # 确定每棵树使用的特征数
        if self.max_features == 'sqrt':
            max_features = int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            max_features = int(np.log2(n_features))
        elif isinstance(self.max_features, float):
            max_features = int(self.max_features * n_features)
        else:
            max_features = self.max_features or n_features
            
        self.trees = []
        
        for i in range(self.n_estimators):
            # Bootstrap采样
            if self.bootstrap:
                indices = np.random.choice(n_samples, n_samples, replace=True)
                X_sample, y_sample = X[indices], y[indices]
            else:
                X_sample, y_sample = X, y
            
            # 训练决策树
            tree = DecisionTreeScratch(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=max_features
            )
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)
            
        return self
    
    def predict(self, X):
        """预测（多数投票）"""
        predictions = np.array([tree.predict(X) for tree in self.trees])
        # 对每个样本进行多数投票
        final_predictions = []
        for i in range(X.shape[0]):
            votes = predictions[:, i]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        return np.array(final_predictions)
    
    def feature_importances_(self):
        """计算特征重要性"""
        importances = np.zeros(self.trees[0].n_features)
        for tree in self.trees:
            importances += tree.feature_importances_
        return importances / len(self.trees)

# 1. 测试决策树
print("1. 决策树分类")

# 生成数据
X, y = make_classification(n_samples=1000, n_features=4, n_informative=3, 
                          n_redundant=1, n_classes=3, random_state=42)

feature_names = ['Feature1', 'Feature2', 'Feature3', 'Feature4']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练决策树
dt = DecisionTreeScratch(max_depth=5, min_samples_split=10, criterion='gini')
dt.fit(X_train, y_train)

# 预测
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"决策树准确率: {accuracy:.4f}")

print("\n决策树结构（前几层）:")
dt.print_tree()

print(f"\n特征重要性:")
for i, importance in enumerate(dt.feature_importances_):
    print(f"{feature_names[i]}: {importance:.4f}")

# 2. 比较不同分裂准则
print("\n2. 比较不同分裂准则")

criterions = ['gini', 'entropy']
for criterion in criterions:
    dt_crit = DecisionTreeScratch(max_depth=5, criterion=criterion)
    dt_crit.fit(X_train, y_train)
    y_pred_crit = dt_crit.predict(X_test)
    accuracy_crit = accuracy_score(y_test, y_pred_crit)
    print(f"{criterion.upper()} 准确率: {accuracy_crit:.4f}")

# 3. 测试随机森林
print("\n3. 随机森林分类")

rf = RandomForestScratch(n_estimators=50, max_depth=5, random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
accuracy_rf = accuracy_score(y_test, y_pred_rf)
print(f"随机森林准确率: {accuracy_rf:.4f}")

# 4. 过拟合分析
print("\n4. 过拟合分析")

max_depths = range(1, 21)
train_accuracies = []
test_accuracies = []

for depth in max_depths:
    dt_depth = DecisionTreeScratch(max_depth=depth)
    dt_depth.fit(X_train, y_train)
    
    train_pred = dt_depth.predict(X_train)
    test_pred = dt_depth.predict(X_test)
    
    train_acc = accuracy_score(y_train, train_pred)
    test_acc = accuracy_score(y_test, test_pred)
    
    train_accuracies.append(train_acc)
    test_accuracies.append(test_acc)

print("深度对性能的影响已计算完成")

# 5. 可视化
print("\n5. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 5.1 过拟合曲线
axes[0, 0].plot(max_depths, train_accuracies, 'o-', label='训练集', linewidth=2)
axes[0, 0].plot(max_depths, test_accuracies, 'o-', label='测试集', linewidth=2)
axes[0, 0].set_title('决策树深度对性能的影响')
axes[0, 0].set_xlabel('最大深度')
axes[0, 0].set_ylabel('准确率')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 5.2 特征重要性
axes[0, 1].bar(feature_names, dt.feature_importances_)
axes[0, 1].set_title('特征重要性')
axes[0, 1].set_ylabel('重要性')
axes[0, 1].tick_params(axis='x', rotation=45)

# 5.3 不同准则对比
gini_dt = DecisionTreeScratch(max_depth=10, criterion='gini')
entropy_dt = DecisionTreeScratch(max_depth=10, criterion='entropy')

gini_dt.fit(X_train, y_train)
entropy_dt.fit(X_train, y_train)

methods = ['Gini', 'Entropy']
accuracies = [
    accuracy_score(y_test, gini_dt.predict(X_test)),
    accuracy_score(y_test, entropy_dt.predict(X_test))
]

axes[0, 2].bar(methods, accuracies, color=['blue', 'orange'])
axes[0, 2].set_title('不同分裂准则准确率对比')
axes[0, 2].set_ylabel('准确率')

# 5.4 随机森林 vs 单个决策树
single_tree_acc = accuracy_score(y_test, dt.predict(X_test))
random_forest_acc = accuracy_score(y_test, rf.predict(X_test))

models = ['单个决策树', '随机森林']
model_accuracies = [single_tree_acc, random_forest_acc]

axes[1, 0].bar(models, model_accuracies, color=['red', 'green'])
axes[1, 0].set_title('单个决策树 vs 随机森林')
axes[1, 0].set_ylabel('准确率')

# 5.5 随机森林中树的数量对性能的影响
n_estimators_range = [1, 5, 10, 20, 50, 100]
rf_accuracies = []

for n_est in n_estimators_range:
    rf_temp = RandomForestScratch(n_estimators=n_est, max_depth=5, random_state=42)
    rf_temp.fit(X_train, y_train)
    rf_pred = rf_temp.predict(X_test)
    rf_acc = accuracy_score(y_test, rf_pred)
    rf_accuracies.append(rf_acc)

axes[1, 1].plot(n_estimators_range, rf_accuracies, 'o-', linewidth=2, color='green')
axes[1, 1].set_title('随机森林树数量对性能的影响')
axes[1, 1].set_xlabel('树的数量')
axes[1, 1].set_ylabel('准确率')
axes[1, 1].grid(True, alpha=0.3)

# 5.6 不同数据集大小对性能的影响
data_sizes = [0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
dt_scores = []
rf_scores = []

for size in data_sizes:
    n_samples = int(size * len(X_train))
    X_subset = X_train[:n_samples]
    y_subset = y_train[:n_samples]
    
    # 决策树
    dt_temp = DecisionTreeScratch(max_depth=5)
    dt_temp.fit(X_subset, y_subset)
    dt_score = accuracy_score(y_test, dt_temp.predict(X_test))
    dt_scores.append(dt_score)
    
    # 随机森林
    rf_temp = RandomForestScratch(n_estimators=20, max_depth=5, random_state=42)
    rf_temp.fit(X_subset, y_subset)
    rf_score = accuracy_score(y_test, rf_temp.predict(X_test))
    rf_scores.append(rf_score)

axes[1, 2].plot(data_sizes, dt_scores, 'o-', label='决策树', linewidth=2)
axes[1, 2].plot(data_sizes, rf_scores, 'o-', label='随机森林', linewidth=2)
axes[1, 2].set_title('数据集大小对性能的影响')
axes[1, 2].set_xlabel('训练集大小比例')
axes[1, 2].set_ylabel('准确率')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/decision_tree_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 6. 使用真实数据集测试
print("\n6. 使用鸢尾花数据集测试")

# 加载鸢尾花数据集
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(
    X_iris, y_iris, test_size=0.3, random_state=42)

# 训练决策树
dt_iris = DecisionTreeScratch(max_depth=3)
dt_iris.fit(X_train_iris, y_train_iris)

# 预测
y_pred_iris = dt_iris.predict(X_test_iris)
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print(f"鸢尾花数据集准确率: {accuracy_iris:.4f}")

print("\n决策树在鸢尾花数据集上的特征重要性:")
for i, importance in enumerate(dt_iris.feature_importances_):
    print(f"{iris.feature_names[i]}: {importance:.4f}")

print("\n=== 练习任务 ===")
print("1. 实现决策树剪枝算法")
print("2. 添加回归树功能")
print("3. 实现CART算法")
print("4. 添加缺失值处理")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现Gradient Boosting Trees")
print("2. 添加特征选择功能")
print("3. 实现Extra Trees算法")
print("4. 研究不同bootstrap策略")
print("5. 实现增量学习决策树")