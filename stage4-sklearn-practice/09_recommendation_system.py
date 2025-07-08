"""
推荐系统实现
学习目标：掌握协同过滤、内容推荐和混合推荐算法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF, PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
import warnings
warnings.filterwarnings('ignore')

print("=== 推荐系统实现 ===\n")

# 1. 推荐系统理论基础
print("1. 推荐系统理论基础")
print("推荐系统类型：")
print("- 协同过滤：基于用户行为相似性")
print("- 内容推荐：基于物品特征相似性") 
print("- 混合推荐：结合多种推荐方法")
print("- 深度学习：神经网络推荐模型")

print("\n评估指标：")
print("1. 准确性：RMSE、MAE")
print("2. 分类：Precision、Recall、F1")
print("3. 排序：NDCG、MAP")
print("4. 多样性、新颖性、覆盖率")

# 2. 生成模拟数据
print("\n2. 生成模拟数据")

np.random.seed(42)

# 用户-物品评分矩阵
n_users = 200
n_items = 100
n_ratings = 3000

# 生成评分数据
users = np.random.randint(0, n_users, n_ratings)
items = np.random.randint(0, n_items, n_ratings)
ratings = np.random.uniform(1, 5, n_ratings)

# 创建评分数据框
ratings_df = pd.DataFrame({
    'user_id': users,
    'item_id': items,
    'rating': ratings
})

# 去重并保留平均评分
ratings_df = ratings_df.groupby(['user_id', 'item_id'])['rating'].mean().reset_index()

print(f"数据集大小: {len(ratings_df)} 条评分")
print(f"用户数: {ratings_df['user_id'].nunique()}")
print(f"物品数: {ratings_df['item_id'].nunique()}")
print(f"稀疏度: {1 - len(ratings_df) / (n_users * n_items):.4f}")

# 创建用户-物品评分矩阵
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print(f"评分矩阵形状: {user_item_matrix.shape}")

# 3. 基于用户的协同过滤
print("\n3. 基于用户的协同过滤")

class UserBasedCollaborativeFiltering:
    def __init__(self, similarity_metric='cosine', k=20):
        self.similarity_metric = similarity_metric
        self.k = k  # 最近邻数量
        self.user_item_matrix = None
        self.user_similarity = None
        
    def fit(self, user_item_matrix):
        """训练模型"""
        self.user_item_matrix = user_item_matrix
        
        # 计算用户相似度矩阵
        if self.similarity_metric == 'cosine':
            self.user_similarity = cosine_similarity(user_item_matrix)
        elif self.similarity_metric == 'euclidean':
            # 转换为相似度
            distances = euclidean_distances(user_item_matrix)
            self.user_similarity = 1 / (1 + distances)
        
        # 将对角线设为0（用户与自己的相似度不考虑）
        np.fill_diagonal(self.user_similarity, 0)
    
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if user_id >= len(self.user_item_matrix) or item_id >= len(self.user_item_matrix.columns):
            return 0.0
        
        # 获取用户相似度
        user_similarities = self.user_similarity[user_id]
        
        # 找到对该物品有评分的用户
        item_ratings = self.user_item_matrix.iloc[:, item_id]
        rated_users = item_ratings[item_ratings > 0].index
        
        if len(rated_users) == 0:
            # 如果没有用户对该物品评分，返回全局平均分
            return self.user_item_matrix[self.user_item_matrix > 0].mean().mean()
        
        # 获取这些用户与目标用户的相似度
        similar_users_sim = user_similarities[rated_users]
        
        # 选择top-k相似用户
        if len(similar_users_sim) > self.k:
            top_k_indices = np.argsort(similar_users_sim)[-self.k:]
            similar_users_sim = similar_users_sim[top_k_indices]
            rated_users = rated_users[top_k_indices]
        
        # 计算加权平均评分
        if np.sum(similar_users_sim) == 0:
            return item_ratings[rated_users].mean()
        
        weighted_ratings = item_ratings[rated_users] * similar_users_sim
        predicted_rating = np.sum(weighted_ratings) / np.sum(similar_users_sim)
        
        return predicted_rating
    
    def recommend(self, user_id, n_recommendations=10):
        """为用户推荐物品"""
        user_ratings = self.user_item_matrix.iloc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        # 按预测评分排序
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# 训练基于用户的协同过滤
user_cf = UserBasedCollaborativeFiltering(k=10)
user_cf.fit(user_item_matrix.values)

print("基于用户的协同过滤训练完成")

# 4. 基于物品的协同过滤
print("\n4. 基于物品的协同过滤")

class ItemBasedCollaborativeFiltering:
    def __init__(self, similarity_metric='cosine', k=20):
        self.similarity_metric = similarity_metric
        self.k = k
        self.user_item_matrix = None
        self.item_similarity = None
        
    def fit(self, user_item_matrix):
        """训练模型"""
        self.user_item_matrix = user_item_matrix
        
        # 计算物品相似度矩阵（基于转置矩阵）
        if self.similarity_metric == 'cosine':
            self.item_similarity = cosine_similarity(user_item_matrix.T)
        elif self.similarity_metric == 'euclidean':
            distances = euclidean_distances(user_item_matrix.T)
            self.item_similarity = 1 / (1 + distances)
        
        np.fill_diagonal(self.item_similarity, 0)
    
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if user_id >= len(self.user_item_matrix) or item_id >= len(self.user_item_matrix.columns):
            return 0.0
        
        # 获取用户的评分历史
        user_ratings = self.user_item_matrix.iloc[user_id]
        rated_items = user_ratings[user_ratings > 0].index
        
        if len(rated_items) == 0:
            return self.user_item_matrix[self.user_item_matrix > 0].mean().mean()
        
        # 获取目标物品与已评分物品的相似度
        item_similarities = self.item_similarity[item_id, rated_items]
        
        # 选择top-k相似物品
        if len(item_similarities) > self.k:
            top_k_indices = np.argsort(item_similarities)[-self.k:]
            item_similarities = item_similarities[top_k_indices]
            rated_items = rated_items[top_k_indices]
        
        # 计算加权平均评分
        if np.sum(item_similarities) == 0:
            return user_ratings[rated_items].mean()
        
        weighted_ratings = user_ratings[rated_items] * item_similarities
        predicted_rating = np.sum(weighted_ratings) / np.sum(item_similarities)
        
        return predicted_rating
    
    def recommend(self, user_id, n_recommendations=10):
        """为用户推荐物品"""
        user_ratings = self.user_item_matrix.iloc[user_id]
        unrated_items = user_ratings[user_ratings == 0].index
        
        predictions = []
        for item_id in unrated_items:
            pred_rating = self.predict(user_id, item_id)
            predictions.append((item_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# 训练基于物品的协同过滤
item_cf = ItemBasedCollaborativeFiltering(k=10)
item_cf.fit(user_item_matrix.values)

print("基于物品的协同过滤训练完成")

# 5. 矩阵分解推荐
print("\n5. 矩阵分解推荐")

class MatrixFactorization:
    def __init__(self, n_factors=50, learning_rate=0.01, regularization=0.01, n_epochs=100):
        self.n_factors = n_factors
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.n_epochs = n_epochs
        self.user_features = None
        self.item_features = None
        self.user_bias = None
        self.item_bias = None
        self.global_mean = None
        
    def fit(self, ratings_df):
        """训练矩阵分解模型"""
        # 初始化参数
        n_users = ratings_df['user_id'].max() + 1
        n_items = ratings_df['item_id'].max() + 1
        
        # 随机初始化特征矩阵
        self.user_features = np.random.normal(0, 0.1, (n_users, self.n_factors))
        self.item_features = np.random.normal(0, 0.1, (n_items, self.n_factors))
        self.user_bias = np.zeros(n_users)
        self.item_bias = np.zeros(n_items)
        self.global_mean = ratings_df['rating'].mean()
        
        # 随机梯度下降训练
        for epoch in range(self.n_epochs):
            for _, row in ratings_df.iterrows():
                user_id = int(row['user_id'])
                item_id = int(row['item_id'])
                rating = row['rating']
                
                # 预测评分
                prediction = self.predict(user_id, item_id)
                error = rating - prediction
                
                # 更新参数
                user_feat = self.user_features[user_id].copy()
                item_feat = self.item_features[item_id].copy()
                
                # 更新特征矩阵
                self.user_features[user_id] += self.learning_rate * (
                    error * item_feat - self.regularization * user_feat
                )
                self.item_features[item_id] += self.learning_rate * (
                    error * user_feat - self.regularization * item_feat
                )
                
                # 更新偏置
                self.user_bias[user_id] += self.learning_rate * (
                    error - self.regularization * self.user_bias[user_id]
                )
                self.item_bias[item_id] += self.learning_rate * (
                    error - self.regularization * self.item_bias[item_id]
                )
            
            if epoch % 20 == 0:
                rmse = self.calculate_rmse(ratings_df)
                print(f"Epoch {epoch}: RMSE = {rmse:.4f}")
    
    def predict(self, user_id, item_id):
        """预测用户对物品的评分"""
        if (user_id >= len(self.user_features) or 
            item_id >= len(self.item_features)):
            return self.global_mean
        
        prediction = (self.global_mean + 
                     self.user_bias[user_id] + 
                     self.item_bias[item_id] + 
                     np.dot(self.user_features[user_id], self.item_features[item_id]))
        
        return np.clip(prediction, 1, 5)  # 限制在评分范围内
    
    def calculate_rmse(self, ratings_df):
        """计算RMSE"""
        predictions = []
        actuals = []
        
        for _, row in ratings_df.iterrows():
            user_id = int(row['user_id'])
            item_id = int(row['item_id'])
            actual = row['rating']
            
            pred = self.predict(user_id, item_id)
            predictions.append(pred)
            actuals.append(actual)
        
        return np.sqrt(mean_squared_error(actuals, predictions))
    
    def recommend(self, user_id, rated_items, n_recommendations=10):
        """为用户推荐物品"""
        n_items = len(self.item_features)
        predictions = []
        
        for item_id in range(n_items):
            if item_id not in rated_items:
                pred_rating = self.predict(user_id, item_id)
                predictions.append((item_id, pred_rating))
        
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:n_recommendations]

# 训练矩阵分解模型
mf = MatrixFactorization(n_factors=20, n_epochs=100)
mf.fit(ratings_df)

print("矩阵分解模型训练完成")

# 6. 基于内容的推荐
print("\n6. 基于内容的推荐")

# 生成物品特征数据
item_features_data = {
    'item_id': range(n_items),
    'genre': np.random.choice(['Action', 'Comedy', 'Drama', 'Thriller', 'Romance'], n_items),
    'year': np.random.randint(1990, 2024, n_items),
    'director': np.random.choice([f'Director_{i}' for i in range(20)], n_items),
    'rating_avg': np.random.uniform(1, 5, n_items)
}

items_df = pd.DataFrame(item_features_data)

class ContentBasedRecommender:
    def __init__(self):
        self.item_features = None
        self.item_similarity = None
        self.tfidf_vectorizer = None
        
    def fit(self, items_df, ratings_df):
        """训练基于内容的推荐模型"""
        self.items_df = items_df
        self.ratings_df = ratings_df
        
        # 处理分类特征
        items_encoded = pd.get_dummies(items_df, columns=['genre'])
        
        # 标准化数值特征
        scaler = StandardScaler()
        numerical_features = ['year', 'rating_avg']
        items_encoded[numerical_features] = scaler.fit_transform(items_encoded[numerical_features])
        
        # 计算物品相似度
        feature_matrix = items_encoded.drop('item_id', axis=1).values
        self.item_similarity = cosine_similarity(feature_matrix)
        
    def get_user_profile(self, user_id):
        """构建用户画像"""
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        
        if len(user_ratings) == 0:
            return np.zeros(len(self.items_df))
        
        # 基于评分加权的物品特征
        user_profile = np.zeros(len(self.items_df))
        
        for _, rating_row in user_ratings.iterrows():
            item_id = rating_row['item_id']
            rating = rating_row['rating']
            
            # 加权累加相似物品
            user_profile += self.item_similarity[item_id] * rating
        
        return user_profile / len(user_ratings)
    
    def recommend(self, user_id, n_recommendations=10):
        """基于内容推荐物品"""
        user_profile = self.get_user_profile(user_id)
        user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
        rated_items = set(user_ratings['item_id'].values)
        
        recommendations = []
        for item_id in range(len(self.items_df)):
            if item_id not in rated_items:
                score = user_profile[item_id]
                recommendations.append((item_id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

# 训练基于内容的推荐
content_recommender = ContentBasedRecommender()
content_recommender.fit(items_df, ratings_df)

print("基于内容的推荐模型训练完成")

# 7. 混合推荐系统
print("\n7. 混合推荐系统")

class HybridRecommender:
    def __init__(self, cf_weight=0.5, content_weight=0.3, mf_weight=0.2):
        self.cf_weight = cf_weight
        self.content_weight = content_weight
        self.mf_weight = mf_weight
        
        self.user_cf = None
        self.content_rec = None
        self.mf_model = None
        
    def fit(self, user_cf, content_rec, mf_model):
        """设置子推荐器"""
        self.user_cf = user_cf
        self.content_rec = content_rec
        self.mf_model = mf_model
    
    def recommend(self, user_id, n_recommendations=10):
        """混合推荐"""
        # 获取各个推荐器的推荐结果
        cf_recs = self.user_cf.recommend(user_id, n_recommendations * 2)
        content_recs = self.content_rec.recommend(user_id, n_recommendations * 2)
        
        # 获取用户已评分物品
        user_ratings = self.content_rec.ratings_df[
            self.content_rec.ratings_df['user_id'] == user_id
        ]
        rated_items = set(user_ratings['item_id'].values)
        
        # 合并推荐结果
        item_scores = {}
        
        # 协同过滤推荐
        for item_id, score in cf_recs:
            if item_id not in rated_items:
                item_scores[item_id] = item_scores.get(item_id, 0) + self.cf_weight * score
        
        # 内容推荐
        for item_id, score in content_recs:
            if item_id not in rated_items:
                item_scores[item_id] = item_scores.get(item_id, 0) + self.content_weight * score
        
        # 矩阵分解推荐
        for item_id in range(len(self.content_rec.items_df)):
            if item_id not in rated_items and item_id in item_scores:
                mf_score = self.mf_model.predict(user_id, item_id)
                item_scores[item_id] += self.mf_weight * mf_score
        
        # 排序并返回top-N
        recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

# 创建混合推荐系统
hybrid_rec = HybridRecommender()
hybrid_rec.fit(user_cf, content_recommender, mf)

print("混合推荐系统构建完成")

# 8. 推荐系统评估
print("\n8. 推荐系统评估")

def evaluate_recommendations(model, ratings_df, test_ratio=0.2):
    """评估推荐系统性能"""
    # 分割数据
    train_df, test_df = train_test_split(ratings_df, test_size=test_ratio, random_state=42)
    
    # 预测测试集评分
    predictions = []
    actuals = []
    
    for _, row in test_df.iterrows():
        user_id = int(row['user_id'])
        item_id = int(row['item_id'])
        actual_rating = row['rating']
        
        if hasattr(model, 'predict'):
            predicted_rating = model.predict(user_id, item_id)
        else:
            predicted_rating = 3.0  # 默认值
        
        predictions.append(predicted_rating)
        actuals.append(actual_rating)
    
    # 计算评估指标
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    
    return rmse, mae, predictions, actuals

# 评估不同推荐算法
print("评估推荐算法性能...")

# 评估矩阵分解
mf_rmse, mf_mae, mf_preds, mf_actuals = evaluate_recommendations(mf, ratings_df)
print(f"矩阵分解 - RMSE: {mf_rmse:.4f}, MAE: {mf_mae:.4f}")

# 9. 推荐多样性分析
print("\n9. 推荐多样性分析")

def calculate_diversity(recommendations, item_similarity_matrix):
    """计算推荐列表的多样性"""
    if len(recommendations) < 2:
        return 0.0
    
    total_similarity = 0
    count = 0
    
    for i in range(len(recommendations)):
        for j in range(i + 1, len(recommendations)):
            item1, item2 = recommendations[i][0], recommendations[j][0]
            if item1 < len(item_similarity_matrix) and item2 < len(item_similarity_matrix):
                similarity = item_similarity_matrix[item1, item2]
                total_similarity += similarity
                count += 1
    
    if count == 0:
        return 0.0
    
    avg_similarity = total_similarity / count
    diversity = 1 - avg_similarity  # 多样性 = 1 - 平均相似度
    
    return diversity

# 计算推荐多样性
sample_user = 0
user_cf_recs = user_cf.recommend(sample_user, 10)
content_recs = content_recommender.recommend(sample_user, 10)
hybrid_recs = hybrid_rec.recommend(sample_user, 10)

# 使用基于内容的物品相似度矩阵
diversity_user_cf = calculate_diversity(user_cf_recs, content_recommender.item_similarity)
diversity_content = calculate_diversity(content_recs, content_recommender.item_similarity)
diversity_hybrid = calculate_diversity(hybrid_recs, content_recommender.item_similarity)

print(f"用户协同过滤多样性: {diversity_user_cf:.4f}")
print(f"基于内容推荐多样性: {diversity_content:.4f}")
print(f"混合推荐多样性: {diversity_hybrid:.4f}")

# 10. 冷启动问题处理
print("\n10. 冷启动问题处理")

class ColdStartHandler:
    def __init__(self, ratings_df, items_df):
        self.ratings_df = ratings_df
        self.items_df = items_df
        self.popular_items = None
        self.item_avg_ratings = None
        
    def fit(self):
        """训练冷启动处理器"""
        # 计算物品流行度
        item_counts = self.ratings_df['item_id'].value_counts()
        self.popular_items = item_counts.head(20).index.tolist()
        
        # 计算物品平均评分
        self.item_avg_ratings = self.ratings_df.groupby('item_id')['rating'].mean()
    
    def recommend_new_user(self, n_recommendations=10):
        """为新用户推荐（基于流行度）"""
        # 结合流行度和平均评分
        popular_with_rating = []
        for item_id in self.popular_items:
            if item_id in self.item_avg_ratings:
                avg_rating = self.item_avg_ratings[item_id]
                popularity_score = len(self.ratings_df[self.ratings_df['item_id'] == item_id])
                combined_score = avg_rating * np.log(1 + popularity_score)
                popular_with_rating.append((item_id, combined_score))
        
        popular_with_rating.sort(key=lambda x: x[1], reverse=True)
        return popular_with_rating[:n_recommendations]
    
    def recommend_new_item(self, item_id, n_recommendations=10):
        """为新物品找到相似用户（基于内容相似度）"""
        if item_id >= len(self.items_df):
            return []
        
        # 基于内容找到相似物品
        item_genre = self.items_df.loc[item_id, 'genre']
        similar_items = self.items_df[self.items_df['genre'] == item_genre]['item_id'].tolist()
        
        # 找到喜欢类似物品的用户
        similar_users = []
        for similar_item in similar_items[:5]:
            users = self.ratings_df[
                (self.ratings_df['item_id'] == similar_item) & 
                (self.ratings_df['rating'] >= 4)
            ]['user_id'].tolist()
            similar_users.extend(users)
        
        # 统计用户频次
        from collections import Counter
        user_counts = Counter(similar_users)
        top_users = user_counts.most_common(n_recommendations)
        
        return [(user_id, count) for user_id, count in top_users]

# 处理冷启动问题
cold_start_handler = ColdStartHandler(ratings_df, items_df)
cold_start_handler.fit()

new_user_recs = cold_start_handler.recommend_new_user(10)
print(f"新用户推荐: {new_user_recs[:5]}")

# 11. 可视化分析
print("\n11. 可视化分析")

fig, axes = plt.subplots(3, 3, figsize=(20, 18))

# 11.1 评分分布
axes[0, 0].hist(ratings_df['rating'], bins=20, alpha=0.7, edgecolor='black')
axes[0, 0].set_title('评分分布')
axes[0, 0].set_xlabel('评分')
axes[0, 0].set_ylabel('频数')
axes[0, 0].grid(True, alpha=0.3)

# 11.2 用户评分数量分布
user_rating_counts = ratings_df['user_id'].value_counts()
axes[0, 1].hist(user_rating_counts.values, bins=30, alpha=0.7, edgecolor='black')
axes[0, 1].set_title('用户评分数量分布')
axes[0, 1].set_xlabel('评分数量')
axes[0, 1].set_ylabel('用户数')
axes[0, 1].grid(True, alpha=0.3)

# 11.3 物品评分数量分布
item_rating_counts = ratings_df['item_id'].value_counts()
axes[0, 2].hist(item_rating_counts.values, bins=30, alpha=0.7, edgecolor='black')
axes[0, 2].set_title('物品评分数量分布')
axes[0, 2].set_xlabel('评分数量')
axes[0, 2].set_ylabel('物品数')
axes[0, 2].grid(True, alpha=0.3)

# 11.4 用户相似度热力图（取样本）
sample_users = 20
user_sim_sample = user_cf.user_similarity[:sample_users, :sample_users]
sns.heatmap(user_sim_sample, cmap='coolwarm', center=0, 
            ax=axes[1, 0], cbar_kws={'label': '相似度'})
axes[1, 0].set_title('用户相似度矩阵（样本）')

# 11.5 物品相似度热力图（取样本）
sample_items = 20
item_sim_sample = item_cf.item_similarity[:sample_items, :sample_items]
sns.heatmap(item_sim_sample, cmap='coolwarm', center=0,
            ax=axes[1, 1], cbar_kws={'label': '相似度'})
axes[1, 1].set_title('物品相似度矩阵（样本）')

# 11.6 预测误差分析
axes[1, 2].scatter(mf_actuals, mf_preds, alpha=0.6)
axes[1, 2].plot([1, 5], [1, 5], 'r--', label='完美预测线')
axes[1, 2].set_xlabel('实际评分')
axes[1, 2].set_ylabel('预测评分')
axes[1, 2].set_title('矩阵分解预测效果')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

# 11.7 不同算法多样性对比
diversity_scores = [diversity_user_cf, diversity_content, diversity_hybrid]
diversity_names = ['用户协同过滤', '基于内容', '混合推荐']
axes[2, 0].bar(diversity_names, diversity_scores, alpha=0.7)
axes[2, 0].set_title('推荐算法多样性对比')
axes[2, 0].set_ylabel('多样性得分')
axes[2, 0].tick_params(axis='x', rotation=45)

# 11.8 物品流行度分析
top_items = item_rating_counts.head(20)
axes[2, 1].bar(range(len(top_items)), top_items.values, alpha=0.7)
axes[2, 1].set_title('Top 20 热门物品')
axes[2, 1].set_xlabel('物品排名')
axes[2, 1].set_ylabel('评分数量')
axes[2, 1].grid(True, alpha=0.3)

# 11.9 评分矩阵稀疏度可视化
sample_matrix = user_item_matrix.iloc[:50, :50].values
axes[2, 2].imshow(sample_matrix, cmap='Blues', aspect='auto')
axes[2, 2].set_title('评分矩阵稀疏度（样本）')
axes[2, 2].set_xlabel('物品ID')
axes[2, 2].set_ylabel('用户ID')

plt.tight_layout()
plt.savefig('/Users/peakom/workstudy/aiproject/ml-learning-journey/stage4-sklearn-practice/recommendation_system_analysis.png', 
            dpi=300, bbox_inches='tight')
plt.close()

# 12. A/B测试框架
print("\n12. A/B测试框架")

class ABTestFramework:
    def __init__(self):
        self.test_results = {}
    
    def run_test(self, algorithm_a, algorithm_b, test_users, name_a="Algorithm A", name_b="Algorithm B"):
        """运行A/B测试"""
        results_a = []
        results_b = []
        
        for user_id in test_users:
            # 算法A的推荐
            recs_a = algorithm_a.recommend(user_id, 10)
            # 算法B的推荐
            recs_b = algorithm_b.recommend(user_id, 10)
            
            # 模拟用户反馈（简化）
            feedback_a = np.random.uniform(0, 1)  # 模拟点击率
            feedback_b = np.random.uniform(0, 1)
            
            results_a.append(feedback_a)
            results_b.append(feedback_b)
        
        # 统计显著性检验（简化）
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(results_a, results_b)
        
        self.test_results[f"{name_a} vs {name_b}"] = {
            'mean_a': np.mean(results_a),
            'mean_b': np.mean(results_b),
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
        
        print(f"A/B测试结果: {name_a} vs {name_b}")
        print(f"  {name_a}平均得分: {np.mean(results_a):.4f}")
        print(f"  {name_b}平均得分: {np.mean(results_b):.4f}")
        print(f"  P值: {p_value:.4f}")
        print(f"  显著性: {'是' if p_value < 0.05 else '否'}")

# 运行A/B测试
ab_test = ABTestFramework()
test_users = list(range(20))  # 选择20个测试用户

ab_test.run_test(user_cf, content_recommender, test_users, 
                "用户协同过滤", "基于内容推荐")

# 13. 实时推荐系统架构
print("\n13. 实时推荐系统架构")

class RealTimeRecommender:
    def __init__(self):
        self.user_profiles = {}
        self.item_features = {}
        self.recent_interactions = {}
        
    def update_user_profile(self, user_id, item_id, rating):
        """实时更新用户画像"""
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = {}
        
        # 更新用户偏好
        if item_id not in self.user_profiles[user_id]:
            self.user_profiles[user_id][item_id] = rating
        else:
            # 加权平均更新
            old_rating = self.user_profiles[user_id][item_id]
            self.user_profiles[user_id][item_id] = 0.7 * old_rating + 0.3 * rating
        
        # 记录最近交互
        if user_id not in self.recent_interactions:
            self.recent_interactions[user_id] = []
        
        self.recent_interactions[user_id].append({
            'item_id': item_id,
            'rating': rating,
            'timestamp': np.datetime64('now')
        })
        
        # 只保留最近的交互
        if len(self.recent_interactions[user_id]) > 50:
            self.recent_interactions[user_id] = self.recent_interactions[user_id][-50:]
    
    def get_real_time_recommendations(self, user_id, n_recommendations=10):
        """获取实时推荐"""
        if user_id not in self.user_profiles:
            return []
        
        # 基于最近交互的快速推荐
        recent_items = [interaction['item_id'] 
                       for interaction in self.recent_interactions.get(user_id, [])]
        
        # 简化的实时推荐逻辑
        recommendations = []
        for item_id in range(50):  # 假设只考虑前50个物品
            if item_id not in recent_items:
                # 基于历史偏好计算得分
                score = np.random.uniform(0, 5)  # 简化计算
                recommendations.append((item_id, score))
        
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations[:n_recommendations]

# 测试实时推荐
real_time_rec = RealTimeRecommender()

# 模拟实时交互
for i in range(10):
    user_id = np.random.randint(0, 5)
    item_id = np.random.randint(0, 20)
    rating = np.random.uniform(3, 5)
    real_time_rec.update_user_profile(user_id, item_id, rating)

# 获取实时推荐
rt_recs = real_time_rec.get_real_time_recommendations(0, 5)
print(f"用户0的实时推荐: {rt_recs}")

print("\n=== 推荐系统总结 ===")
print("✅ 掌握协同过滤算法（用户和物品）")
print("✅ 实现矩阵分解推荐模型")
print("✅ 构建基于内容的推荐系统")
print("✅ 设计混合推荐策略")
print("✅ 处理冷启动问题")
print("✅ 评估推荐系统性能")
print("✅ 分析推荐多样性")
print("✅ 设计A/B测试框架")

print("\n关键技术:")
print("1. 协同过滤：基于用户或物品的相似度")
print("2. 矩阵分解：降维处理稀疏评分矩阵")
print("3. 内容推荐：利用物品特征信息")
print("4. 混合策略：结合多种推荐方法")
print("5. 实时更新：在线学习用户偏好")

print("\n评估指标:")
print("1. 准确性：RMSE、MAE")
print("2. 排序质量：NDCG、MAP")
print("3. 多样性：推荐列表的多样程度")
print("4. 覆盖率：推荐物品的覆盖范围")
print("5. 新颖性：推荐非热门物品的能力")

print("\n实际应用:")
print("1. 电商平台商品推荐")
print("2. 视频/音乐平台内容推荐")
print("3. 社交媒体信息流推荐")
print("4. 新闻资讯个性化推荐")
print("5. 在线教育课程推荐")

print("\n=== 练习任务 ===")
print("1. 实现基于深度学习的推荐算法")
print("2. 处理隐式反馈数据")
print("3. 实现序列推荐算法")
print("4. 添加时间因素的推荐模型")

print("\n=== 扩展练习 ===")
print("请尝试完成以下扩展练习:")
print("1. 实现图神经网络推荐算法")
print("2. 研究多目标推荐优化")
print("3. 构建可解释推荐系统")
print("4. 实现联邦学习推荐系统")
print("5. 研究推荐系统的公平性问题")