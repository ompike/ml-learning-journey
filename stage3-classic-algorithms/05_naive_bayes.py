"""
朴素贝叶斯分类器从零实现
学习目标：理解贝叶斯定理和朴素贝叶斯分类器的原理
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, fetch_20newsgroups
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

print("=== 朴素贝叶斯分类器从零实现 ===\n")

# 1. 贝叶斯定理基础
print("1. 贝叶斯定理基础")
print("贝叶斯定理: P(A|B) = P(B|A) * P(A) / P(B)")
print("朴素贝叶斯假设: 特征之间相互独立")
print("分类决策: 选择后验概率最大的类别")

# 2. 高斯朴素贝叶斯实现
print("\n2. 高斯朴素贝叶斯实现")

class GaussianNaiveBayesFromScratch:
    def __init__(self):
        self.classes = None
        self.class_priors = {}
        self.feature_params = {}  # 存储每个特征的均值和方差
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # 计算先验概率
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # 计算每个类别每个特征的参数（均值和方差）
        self.feature_params = {}
        for c in self.classes:
            class_data = X[y == c]
            self.feature_params[c] = {
                'mean': np.mean(class_data, axis=0),
                'var': np.var(class_data, axis=0) + 1e-9  # 避免方差为0
            }
    
    def _gaussian_pdf(self, x, mean, var):
        """计算高斯概率密度"""
        return (1 / np.sqrt(2 * np.pi * var)) * np.exp(-0.5 * ((x - mean) ** 2) / var)
    
    def predict_proba(self, X):
        """预测概率"""
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # 先验概率
            prior = self.class_priors[c]
            
            # 似然概率（假设特征独立）
            likelihood = np.ones(n_samples)
            for j in range(X.shape[1]):
                mean = self.feature_params[c]['mean'][j]
                var = self.feature_params[c]['var'][j]
                likelihood *= self._gaussian_pdf(X[:, j], mean, var)
            
            # 后验概率（未归一化）
            probabilities[:, i] = prior * likelihood
        
        # 归一化
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

# 3. 多项式朴素贝叶斯实现
print("\n3. 多项式朴素贝叶斯实现")

class MultinomialNaiveBayesFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # 拉普拉斯平滑参数
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # 计算先验概率
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # 计算特征概率（使用拉普拉斯平滑）
        self.feature_probs = {}
        for c in self.classes:
            class_data = X[y == c]
            # 计算每个特征的条件概率
            feature_counts = np.sum(class_data, axis=0)
            total_count = np.sum(feature_counts)
            
            # 拉普拉斯平滑
            self.feature_probs[c] = (feature_counts + self.alpha) / (total_count + self.alpha * n_features)
    
    def predict_proba(self, X):
        """预测概率"""
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # 先验概率（取对数避免下溢）
            log_prior = np.log(self.class_priors[c])
            
            # 似然概率（取对数）
            log_likelihood = np.sum(X * np.log(self.feature_probs[c]), axis=1)
            
            # 后验概率（对数形式）
            probabilities[:, i] = log_prior + log_likelihood
        
        # 转换回概率形式
        probabilities = np.exp(probabilities)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

# 4. 伯努利朴素贝叶斯实现
print("\n4. 伯努利朴素贝叶斯实现")

class BernoulliNaiveBayesFromScratch:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.class_priors = {}
        self.feature_probs = {}
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.classes = np.unique(y)
        
        # 计算先验概率
        for c in self.classes:
            self.class_priors[c] = np.sum(y == c) / n_samples
        
        # 计算特征概率
        self.feature_probs = {}
        for c in self.classes:
            class_data = X[y == c]
            n_class_samples = class_data.shape[0]
            
            # 对于每个特征，计算P(feature=1|class)
            feature_counts = np.sum(class_data, axis=0)
            
            # 拉普拉斯平滑
            self.feature_probs[c] = (feature_counts + self.alpha) / (n_class_samples + 2 * self.alpha)
    
    def predict_proba(self, X):
        """预测概率"""
        n_samples = X.shape[0]
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        for i, c in enumerate(self.classes):
            # 先验概率（取对数）
            log_prior = np.log(self.class_priors[c])
            
            # 似然概率（取对数）
            p_feature_1 = self.feature_probs[c]
            p_feature_0 = 1 - p_feature_1
            
            log_likelihood = np.sum(
                X * np.log(p_feature_1) + (1 - X) * np.log(p_feature_0), 
                axis=1
            )
            
            # 后验概率（对数形式）
            probabilities[:, i] = log_prior + log_likelihood
        
        # 转换回概率形式
        probabilities = np.exp(probabilities)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        return probabilities
    
    def predict(self, X):
        """预测类别"""
        probabilities = self.predict_proba(X)
        return self.classes[np.argmax(probabilities, axis=1)]

# 5. 生成测试数据
print("\n5. 生成测试数据")

# 5.1 连续特征数据（高斯分布）
np.random.seed(42)
X_continuous, y_continuous = make_classification(
    n_samples=1000, n_features=4, n_redundant=0, n_informative=4,
    n_clusters_per_class=1, n_classes=3, random_state=42
)

print(f"连续特征数据集形状: {X_continuous.shape}")
print(f"类别分布: {np.bincount(y_continuous)}")

# 5.2 离散特征数据（计数数据）
X_discrete = np.random.poisson(3, (1000, 5))  # 泊松分布生成计数数据
y_discrete = np.random.choice([0, 1, 2], 1000, p=[0.4, 0.3, 0.3])

print(f"离散特征数据集形状: {X_discrete.shape}")

# 5.3 二元特征数据
X_binary = np.random.binomial(1, 0.3, (1000, 6))  # 伯努利分布
y_binary = np.random.choice([0, 1], 1000, p=[0.6, 0.4])

print(f"二元特征数据集形状: {X_binary.shape}")

# 6. 测试高斯朴素贝叶斯
print("\n6. 测试高斯朴素贝叶斯")

X_train, X_test, y_train, y_test = train_test_split(
    X_continuous, y_continuous, test_size=0.3, random_state=42)

# 自实现的高斯朴素贝叶斯
gnb_custom = GaussianNaiveBayesFromScratch()
gnb_custom.fit(X_train, y_train)
y_pred_custom = gnb_custom.predict(X_test)
accuracy_custom = accuracy_score(y_test, y_pred_custom)

# sklearn的高斯朴素贝叶斯
gnb_sklearn = GaussianNB()
gnb_sklearn.fit(X_train, y_train)
y_pred_sklearn = gnb_sklearn.predict(X_test)
accuracy_sklearn = accuracy_score(y_test, y_pred_sklearn)

print(f"自实现高斯NB准确率: {accuracy_custom:.4f}")
print(f"sklearn高斯NB准确率: {accuracy_sklearn:.4f}")

# 7. 文本分类示例
print("\n7. 文本分类示例")

# 创建简单的文本数据
texts = [
    "I love this movie", "This film is great", "Amazing story and acting",
    "Best movie ever", "Wonderful cinematography", "Excellent direction",
    "I hate this movie", "This film is terrible", "Boring and bad acting",
    "Worst movie ever", "Awful story", "Poor direction",
    "The movie is okay", "It's an average film", "Not bad not good",
    "Mediocre acting", "Could be better", "Just fine"
]

labels = [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2]  # 1:positive, 0:negative, 2:neutral

# 文本向量化
vectorizer = CountVectorizer()
X_text = vectorizer.fit_transform(texts).toarray()

print(f"文本特征矩阵形状: {X_text.shape}")
print(f"词汇表大小: {len(vectorizer.get_feature_names_out())}")

# 分割数据
X_text_train, X_text_test, y_text_train, y_text_test = train_test_split(
    X_text, labels, test_size=0.3, random_state=42)

# 测试多项式朴素贝叶斯
mnb_custom = MultinomialNaiveBayesFromScratch(alpha=1.0)
mnb_custom.fit(X_text_train, y_text_train)
y_text_pred_custom = mnb_custom.predict(X_text_test)

mnb_sklearn = MultinomialNB(alpha=1.0)
mnb_sklearn.fit(X_text_train, y_text_train)
y_text_pred_sklearn = mnb_sklearn.predict(X_text_test)

text_accuracy_custom = accuracy_score(y_text_test, y_text_pred_custom)
text_accuracy_sklearn = accuracy_score(y_text_test, y_text_pred_sklearn)

print(f"自实现多项式NB准确率: {text_accuracy_custom:.4f}")
print(f"sklearn多项式NB准确率: {text_accuracy_sklearn:.4f}")

# 8. 不同朴素贝叶斯变体比较
print("\n8. 不同朴素贝叶斯变体比较")

# 准备不同类型的数据
datasets = {
    'Continuous Features': (X_continuous, y_continuous),
    'Discrete Features': (X_discrete, y_discrete),
    'Binary Features': (X_binary, y_binary)
}

nb_variants = {
    'Gaussian NB': GaussianNB(),
    'Multinomial NB': MultinomialNB(),
    'Bernoulli NB': BernoulliNB()
}

results = {}

for data_name, (X_data, y_data) in datasets.items():
    X_tr, X_te, y_tr, y_te = train_test_split(X_data, y_data, test_size=0.3, random_state=42)
    results[data_name] = {}
    
    for nb_name, nb_model in nb_variants.items():
        try:
            nb_model.fit(X_tr, y_tr)
            y_pred = nb_model.predict(X_te)
            accuracy = accuracy_score(y_te, y_pred)
            results[data_name][nb_name] = accuracy
            print(f"{data_name} + {nb_name}: {accuracy:.4f}")
        except Exception as e:
            print(f"错误 {nb_name} on {data_name}: {e}")
            results[data_name][nb_name] = 0.0

# 9. 朴素贝叶斯的特征独立性假设验证
print("\n9. 特征独立性假设验证")

def check_feature_independence(X, feature_names=None):
    """检查特征之间的相关性"""
    correlation_matrix = np.corrcoef(X.T)
    
    if feature_names is None:
        feature_names = [f'Feature_{i}' for i in range(X.shape[1])]
    
    print("特征相关性矩阵:")
    print("对角线为1.0，其他值越接近0表示独立性假设越合理")
    
    # 计算非对角线元素的平均绝对值
    n_features = len(feature_names)
    off_diagonal = []
    for i in range(n_features):
        for j in range(i+1, n_features):
            off_diagonal.append(abs(correlation_matrix[i, j]))
    
    avg_correlation = np.mean(off_diagonal)
    print(f"平均绝对相关系数: {avg_correlation:.4f}")
    
    if avg_correlation < 0.3:
        print("✅ 特征相对独立，朴素贝叶斯假设较为合理")
    elif avg_correlation < 0.7:
        print("⚠️ 特征有一定相关性，朴素贝叶斯可能受影响")
    else:
        print("❌ 特征高度相关，朴素贝叶斯假设不太合理")
    
    return correlation_matrix

# 检查连续特征的独立性
corr_matrix = check_feature_independence(X_continuous)

# 10. 拉普拉斯平滑的影响
print("\n10. 拉普拉斯平滑的影响")

alpha_values = [0.1, 0.5, 1.0, 2.0, 5.0]
smoothing_results = {}

for alpha in alpha_values:
    mnb = MultinomialNB(alpha=alpha)
    mnb.fit(X_text_train, y_text_train)
    y_pred = mnb.predict(X_text_test)
    accuracy = accuracy_score(y_text_test, y_pred)
    smoothing_results[alpha] = accuracy
    print(f"Alpha = {alpha}: 准确率 = {accuracy:.4f}")

optimal_alpha = max(smoothing_results.keys(), key=lambda x: smoothing_results[x])
print(f"最优Alpha值: {optimal_alpha}")

# 11. 可视化分析
print("\n11. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 11.1 数据分布可视化（连续特征）
axes[0, 0].hist(X_continuous[y_continuous == 0, 0], alpha=0.7, label='Class 0', bins=20)
axes[0, 0].hist(X_continuous[y_continuous == 1, 0], alpha=0.7, label='Class 1', bins=20)
axes[0, 0].hist(X_continuous[y_continuous == 2, 0], alpha=0.7, label='Class 2', bins=20)
axes[0, 0].set_title('特征分布（第一个特征）')
axes[0, 0].set_xlabel('特征值')
axes[0, 0].set_ylabel('频数')
axes[0, 0].legend()

# 11.2 不同NB变体性能对比
results_df = pd.DataFrame(results).T
sns.heatmap(results_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 1])
axes[0, 1].set_title('不同数据类型上的NB性能')

# 11.3 特征相关性热力图
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, ax=axes[0, 2])
axes[0, 2].set_title('特征相关性矩阵')

# 11.4 拉普拉斯平滑影响
alphas = list(smoothing_results.keys())
accs = list(smoothing_results.values())
axes[1, 0].plot(alphas, accs, 'bo-')
axes[1, 0].set_title('拉普拉斯平滑参数对性能的影响')
axes[1, 0].set_xlabel('Alpha值')
axes[1, 0].set_ylabel('准确率')
axes[1, 0].grid(True, alpha=0.3)

# 11.5 混淆矩阵
cm = confusion_matrix(y_test, y_pred_custom)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('高斯NB混淆矩阵')
axes[1, 1].set_xlabel('预测标签')
axes[1, 1].set_ylabel('真实标签')

# 11.6 概率预测分析
probs = gnb_custom.predict_proba(X_test)
max_probs = np.max(probs, axis=1)
axes[1, 2].hist(max_probs, bins=20, alpha=0.7, edgecolor='black')
axes[1, 2].set_title('预测置信度分布')
axes[1, 2].set_xlabel('最大预测概率')
axes[1, 2].set_ylabel('频数')
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage3-classic-algorithms/naive_bayes_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 12. 实际应用：垃圾邮件检测
print("\n12. 实际应用：垃圾邮件检测")

# 模拟垃圾邮件数据
spam_words = ['free', 'win', 'money', 'offer', 'buy', 'sale', 'discount', 'urgent']
normal_words = ['meeting', 'project', 'team', 'work', 'schedule', 'report', 'update', 'thanks']

def generate_email_text(is_spam, length=10):
    """生成模拟邮件文本"""
    if is_spam:
        words = np.random.choice(spam_words + normal_words, length, 
                               p=[0.7/len(spam_words)]*len(spam_words) + [0.3/len(normal_words)]*len(normal_words))
    else:
        words = np.random.choice(normal_words + spam_words, length,
                               p=[0.8/len(normal_words)]*len(normal_words) + [0.2/len(spam_words)]*len(spam_words))
    return ' '.join(words)

# 生成邮件数据
emails = []
spam_labels = []

for _ in range(500):
    is_spam = np.random.choice([0, 1], p=[0.7, 0.3])
    email_text = generate_email_text(is_spam)
    emails.append(email_text)
    spam_labels.append(is_spam)

# 向量化
email_vectorizer = CountVectorizer()
X_emails = email_vectorizer.fit_transform(emails).toarray()

# 分割数据
X_email_train, X_email_test, y_email_train, y_email_test = train_test_split(
    X_emails, spam_labels, test_size=0.3, random_state=42)

# 训练垃圾邮件分类器
spam_classifier = MultinomialNB(alpha=1.0)
spam_classifier.fit(X_email_train, y_email_train)
y_email_pred = spam_classifier.predict(X_email_test)

spam_accuracy = accuracy_score(y_email_test, y_email_pred)
print(f"垃圾邮件检测准确率: {spam_accuracy:.4f}")

# 分析重要特征
feature_names = email_vectorizer.get_feature_names_out()
feature_log_prob = spam_classifier.feature_log_prob_

print("\n垃圾邮件指示词（对数概率差异最大）:")
prob_diff = feature_log_prob[1] - feature_log_prob[0]  # spam - normal
top_spam_indices = np.argsort(prob_diff)[-10:][::-1]

for idx in top_spam_indices:
    word = feature_names[idx]
    diff = prob_diff[idx]
    print(f"  {word}: {diff:.4f}")

print("\n=== 朴素贝叶斯总结 ===")
print("✅ 理解贝叶斯定理和朴素贝叶斯假设")
print("✅ 实现高斯、多项式、伯努利朴素贝叶斯")
print("✅ 掌握拉普拉斯平滑技术")
print("✅ 分析特征独立性假设的影响")
print("✅ 应用于文本分类和垃圾邮件检测")
print("✅ 比较不同变体的适用场景")

print("\n算法特点:")
print("优点: 简单高效、训练快速、对小数据集表现好、可解释性强")
print("缺点: 特征独立假设过强、对特征缺失敏感")

print("\n适用场景:")
print("1. 文本分类（垃圾邮件、情感分析）")
print("2. 医疗诊断")
print("3. 推荐系统")
print("4. 实时分类任务")

print("\n=== 练习任务 ===")
print("1. 实现补充朴素贝叶斯处理缺失值")
print("2. 尝试核密度估计改进高斯NB")
print("3. 实现在线学习版本")
print("4. 研究特征选择对NB的影响")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现层次贝叶斯模型")
print("2. 研究贝叶斯网络")
print("3. 实现半监督朴素贝叶斯")
print("4. 结合主动学习策略")
print("5. 实现多标签朴素贝叶斯分类")