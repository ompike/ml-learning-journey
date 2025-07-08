"""
文本分析项目
学习目标：掌握文本预处理、特征提取和文本分类的完整流程
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import LatentDirichletAllocation
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud
import warnings
warnings.filterwarnings('ignore')

print("=== 文本分析项目 ===\n")

# 下载必要的NLTK数据
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    print("下载NLTK数据...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

# 1. 数据加载和准备
print("1. 数据加载和准备")

# 加载20新闻组数据集的子集
categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                     shuffle=True, random_state=42)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, 
                                    shuffle=True, random_state=42)

print(f"训练样本数: {len(newsgroups_train.data)}")
print(f"测试样本数: {len(newsgroups_test.data)}")
print(f"类别: {newsgroups_train.target_names}")

# 查看样本数据
print(f"\n样本文本 (前200字符):")
print(newsgroups_train.data[0][:200])
print(f"标签: {newsgroups_train.target_names[newsgroups_train.target[0]]}")

# 2. 文本预处理
print("\n2. 文本预处理")

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
    
    def clean_text(self, text):
        """基础文本清理"""
        # 转换为小写
        text = text.lower()
        
        # 移除邮件头部信息
        text = re.sub(r'^.*?lines:', '', text, flags=re.MULTILINE | re.DOTALL)
        
        # 移除URL
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        
        # 移除邮箱地址
        text = re.sub(r'\S+@\S+', '', text)
        
        # 移除数字和特殊字符，保留字母和空格
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # 移除多余空格
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def remove_stopwords(self, text):
        """移除停用词"""
        words = text.split()
        return ' '.join([word for word in words if word not in self.stop_words])
    
    def stem_text(self, text):
        """词干提取"""
        words = text.split()
        return ' '.join([self.stemmer.stem(word) for word in words])
    
    def lemmatize_text(self, text):
        """词形还原"""
        words = text.split()
        return ' '.join([self.lemmatizer.lemmatize(word) for word in words])
    
    def preprocess(self, text, use_stemming=True, remove_stop=True):
        """完整预处理流程"""
        text = self.clean_text(text)
        
        if remove_stop:
            text = self.remove_stopwords(text)
        
        if use_stemming:
            text = self.stem_text(text)
        else:
            text = self.lemmatize_text(text)
        
        return text

# 预处理文本数据
preprocessor = TextPreprocessor()

print("预处理训练数据...")
train_texts_clean = [preprocessor.preprocess(text) for text in newsgroups_train.data]

print("预处理测试数据...")
test_texts_clean = [preprocessor.preprocess(text) for text in newsgroups_test.data]

# 显示预处理前后的对比
print("\n预处理前:")
print(newsgroups_train.data[0][:200])
print("\n预处理后:")
print(train_texts_clean[0][:200])

# 3. 特征提取
print("\n3. 特征提取")

# 3.1 词袋模型 (Bag of Words)
print("3.1 词袋模型")
vectorizer_bow = CountVectorizer(max_features=5000, min_df=2, max_df=0.8)
X_train_bow = vectorizer_bow.fit_transform(train_texts_clean)
X_test_bow = vectorizer_bow.transform(test_texts_clean)

print(f"BoW特征维度: {X_train_bow.shape[1]}")
print(f"BoW特征矩阵稀疏度: {1 - X_train_bow.nnz / (X_train_bow.shape[0] * X_train_bow.shape[1]):.4f}")

# 3.2 TF-IDF
print("\n3.2 TF-IDF")
vectorizer_tfidf = TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)
X_train_tfidf = vectorizer_tfidf.fit_transform(train_texts_clean)
X_test_tfidf = vectorizer_tfidf.transform(test_texts_clean)

print(f"TF-IDF特征维度: {X_train_tfidf.shape[1]}")
print(f"TF-IDF特征矩阵稀疏度: {1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1]):.4f}")

# 3.3 N-gram特征
print("\n3.3 N-gram特征")
vectorizer_ngram = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), 
                                  min_df=2, max_df=0.8)
X_train_ngram = vectorizer_ngram.fit_transform(train_texts_clean)
X_test_ngram = vectorizer_ngram.transform(test_texts_clean)

print(f"N-gram特征维度: {X_train_ngram.shape[1]}")

# 4. 模型训练和比较
print("\n4. 模型训练和比较")

# 定义分类器
classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='linear', random_state=42)
}

# 定义特征提取方法
feature_methods = {
    'BoW': (X_train_bow, X_test_bow),
    'TF-IDF': (X_train_tfidf, X_test_tfidf),
    'N-gram': (X_train_ngram, X_test_ngram)
}

# 训练和评估所有组合
results = {}
y_train = newsgroups_train.target
y_test = newsgroups_test.target

for feature_name, (X_tr, X_te) in feature_methods.items():
    results[feature_name] = {}
    
    for clf_name, clf in classifiers.items():
        # 训练模型
        clf.fit(X_tr, y_train)
        
        # 预测
        y_pred = clf.predict(X_te)
        
        # 计算准确率
        accuracy = accuracy_score(y_test, y_pred)
        results[feature_name][clf_name] = accuracy
        
        print(f"{feature_name} + {clf_name}: {accuracy:.4f}")

# 5. 特征重要性分析
print("\n5. 特征重要性分析")

# 使用TF-IDF + 逻辑回归分析特征重要性
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_tfidf, y_train)

# 获取特征名称
feature_names = vectorizer_tfidf.get_feature_names_out()

# 每个类别的重要特征
for i, category in enumerate(newsgroups_train.target_names):
    # 获取该类别的系数
    coef = lr_model.coef_[i]
    
    # 获取最重要的正负特征
    top_positive = np.argsort(coef)[-10:][::-1]
    top_negative = np.argsort(coef)[:10]
    
    print(f"\n类别: {category}")
    print("最相关特征:")
    for idx in top_positive:
        print(f"  {feature_names[idx]}: {coef[idx]:.4f}")
    
    print("最不相关特征:")
    for idx in top_negative:
        print(f"  {feature_names[idx]}: {coef[idx]:.4f}")

# 6. 主题模型 (LDA)
print("\n6. 主题模型 (LDA)")

# 使用BoW进行LDA
lda = LatentDirichletAllocation(n_components=4, random_state=42, max_iter=10)
lda.fit(X_train_bow)

# 显示主题
feature_names_bow = vectorizer_bow.get_feature_names_out()

print("发现的主题:")
for topic_idx, topic in enumerate(lda.components_):
    top_words_idx = topic.argsort()[-10:][::-1]
    top_words = [feature_names_bow[i] for i in top_words_idx]
    print(f"主题 {topic_idx + 1}: {', '.join(top_words)}")

# 7. 情感分析示例
print("\n7. 情感分析示例")

# 创建简单的情感词典
positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                 'love', 'like', 'best', 'perfect', 'awesome', 'brilliant']
negative_words = ['bad', 'terrible', 'awful', 'hate', 'worst', 'horrible', 
                 'disgusting', 'annoying', 'stupid', 'useless', 'disappointing']

def simple_sentiment_analysis(text):
    """简单的基于词典的情感分析"""
    words = text.lower().split()
    
    positive_count = sum(1 for word in words if word in positive_words)
    negative_count = sum(1 for word in words if word in negative_words)
    
    if positive_count > negative_count:
        return 'positive'
    elif negative_count > positive_count:
        return 'negative'
    else:
        return 'neutral'

# 对部分测试文本进行情感分析
sample_texts = test_texts_clean[:10]
sentiments = [simple_sentiment_analysis(text) for text in sample_texts]

print("情感分析结果 (前10个样本):")
for i, (text, sentiment) in enumerate(zip(sample_texts, sentiments)):
    print(f"样本 {i+1}: {sentiment} - {text[:100]}...")

# 8. 可视化分析
print("\n8. 可视化分析")

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# 8.1 类别分布
class_counts = np.bincount(y_train)
axes[0, 0].bar(newsgroups_train.target_names, class_counts)
axes[0, 0].set_title('训练集类别分布')
axes[0, 0].set_ylabel('样本数量')
axes[0, 0].tick_params(axis='x', rotation=45)

# 8.2 模型性能对比热力图
results_df = pd.DataFrame(results).T
sns.heatmap(results_df, annot=True, fmt='.3f', cmap='YlOrRd', ax=axes[0, 1])
axes[0, 1].set_title('不同特征提取方法和分类器的性能对比')

# 8.3 文本长度分布
text_lengths = [len(text.split()) for text in train_texts_clean]
axes[0, 2].hist(text_lengths, bins=50, alpha=0.7, edgecolor='black')
axes[0, 2].set_title('文本长度分布')
axes[0, 2].set_xlabel('单词数量')
axes[0, 2].set_ylabel('频数')

# 8.4 最佳模型的混淆矩阵
best_model = LogisticRegression(max_iter=1000, random_state=42)
best_model.fit(X_train_tfidf, y_train)
y_pred_best = best_model.predict(X_test_tfidf)
cm = confusion_matrix(y_test, y_pred_best)

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=newsgroups_train.target_names,
            yticklabels=newsgroups_train.target_names, ax=axes[1, 0])
axes[1, 0].set_title('最佳模型混淆矩阵')
axes[1, 0].set_xlabel('预测标签')
axes[1, 0].set_ylabel('真实标签')

# 8.5 词频分析
all_text = ' '.join(train_texts_clean)
word_freq = {}
for word in all_text.split():
    word_freq[word] = word_freq.get(word, 0) + 1

# 最频繁的词汇
top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:20]
words, freqs = zip(*top_words)

axes[1, 1].barh(range(len(words)), freqs)
axes[1, 1].set_yticks(range(len(words)))
axes[1, 1].set_yticklabels(words)
axes[1, 1].set_title('最频繁的20个词汇')
axes[1, 1].set_xlabel('频数')

# 8.6 主题分布
topic_probs = lda.transform(X_train_bow)
topic_means = topic_probs.mean(axis=0)

axes[1, 2].pie(topic_means, labels=[f'主题 {i+1}' for i in range(len(topic_means))], 
               autopct='%1.1f%%')
axes[1, 2].set_title('主题分布')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/text_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 9. 词云可视化
print("\n9. 生成词云")

try:
    # 为每个类别生成词云
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    for i, category in enumerate(newsgroups_train.target_names):
        # 获取该类别的所有文本
        category_texts = [train_texts_clean[j] for j in range(len(y_train)) 
                         if y_train[j] == i]
        category_text = ' '.join(category_texts)
        
        # 生成词云
        wordcloud = WordCloud(width=400, height=300, background_color='white',
                             max_words=100, random_state=42).generate(category_text)
        
        axes[i].imshow(wordcloud, interpolation='bilinear')
        axes[i].set_title(f'{category} 词云')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/wordclouds.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print("词云已生成")
    
except ImportError:
    print("WordCloud库未安装，跳过词云生成")

# 10. 文本分类Pipeline
print("\n10. 文本分类Pipeline")

# 创建完整的文本分类pipeline
text_pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=5000, min_df=2, max_df=0.8)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# 直接在原始文本上训练
text_pipeline.fit(newsgroups_train.data, y_train)

# 预测
pipeline_pred = text_pipeline.predict(newsgroups_test.data)
pipeline_accuracy = accuracy_score(y_test, pipeline_pred)

print(f"Pipeline准确率: {pipeline_accuracy:.4f}")

# 交叉验证
cv_scores = cross_val_score(text_pipeline, newsgroups_train.data, y_train, cv=5)
print(f"交叉验证准确率: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# 预测新文本
def predict_text_category(text):
    """预测文本类别"""
    prediction = text_pipeline.predict([text])[0]
    probabilities = text_pipeline.predict_proba([text])[0]
    
    print(f"预测类别: {newsgroups_train.target_names[prediction]}")
    print("各类别概率:")
    for i, prob in enumerate(probabilities):
        print(f"  {newsgroups_train.target_names[i]}: {prob:.4f}")
    
    return prediction

# 示例预测
sample_text = "I am having trouble with my computer graphics card. The display is not working properly."
print(f"\n示例文本: {sample_text}")
predict_text_category(sample_text)

print("\n=== 文本分析总结 ===")
print("✅ 文本数据加载和探索")
print("✅ 文本预处理和清理")
print("✅ 多种特征提取方法")
print("✅ 文本分类模型比较")
print("✅ 特征重要性分析")
print("✅ 主题模型应用")
print("✅ 情感分析实现")
print("✅ 可视化和词云生成")

print("\n=== 练习任务 ===")
print("1. 实现更复杂的文本预处理")
print("2. 尝试深度学习方法(Word2Vec, BERT)")
print("3. 实现命名实体识别")
print("4. 构建问答系统")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现文本摘要生成")
print("2. 构建多语言文本分析")
print("3. 实现实时文本流处理")
print("4. 研究偏见检测和公平性")
print("5. 实现文本生成模型")