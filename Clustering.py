from XGBALG import TermDepositModel
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans, DBSCAN
from sklearn.cluster import AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd

print("start!")

# -----------------------------
# Step 0: 读取原始数据
# -----------------------------
df = pd.read_csv("term-deposit-marketing-2020.csv")
print("reading data finished")


# -----------------------------
# Step 1: 初始化模型（不训练，直接用真实 y）
# -----------------------------
model_runner = TermDepositModel(
    log_cols=['balance', 'duration'])
print("initializing model finished")

# -----------------------------
# Step 2: 筛选真实订阅客户
# -----------------------------
df_proc = model_runner.preprocess_data(df)
subscribers = df_proc[df_proc['y'] == 1].copy()
print(f"Number of actual subscribers: {len(subscribers)}")

'''
# -----------------------------
# Step 1: 初始化并训练模型
# -----------------------------
model_runner = TermDepositModel(log_cols=['balance', 'duration'])
model_runner.train_cv(df)  # 训练 5-fold CV + 最终模型
model_runner.search_global_threshold(min_acc=0.86)  # 可选，找最佳阈值
print("model training finished")

# -----------------------------
# Step 2: 筛选预测为 1 的客户
# -----------------------------
df_proc = model_runner.preprocess_data(df)
y_pred = (model_runner.all_y_proba >= model_runner.best_threshold).astype(int)
subscribers = df_proc[y_pred == 1].copy()
print(f"Number of predicted subscribers: {len(subscribers)}")
'''

# -----------------------------
# Step 3: 特征准备
# -----------------------------

features = subscribers.drop(columns=['y'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(features)

# -----------------------------
# Step 3.1: PCA 降维
# -----------------------------
# 例如保留 10 个主成分
pca = PCA(n_components=10, random_state=42)
X_pca = pca.fit_transform(X_scaled)
print(
    f"Explained variance ratio (10 components): {np.sum(pca.explained_variance_ratio_):.4f}")

# -----------------------------
# Step 4: 寻找最佳 KMeans 簇数量
# -----------------------------
sil_scores = []
K_range = range(3, 8)
for K in K_range:
    kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_pca)  # 用 PCA 数据
    score = silhouette_score(X_pca, labels)
    sil_scores.append(score)
    print(f"K={K}, Silhouette Score={score:.4f}")

best_K = K_range[np.argmax(sil_scores)]
print(f"Best K according to silhouette score: {best_K}")

# -----------------------------
# Step 5: 聚类
# -----------------------------
kmeans = KMeans(n_clusters=best_K, random_state=42, n_init=10)
subscribers['cluster'] = kmeans.fit_predict(X_pca)

# -----------------------------
# Step 6: t-SNE 可视化
# -----------------------------
tsne = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
X_tsne = tsne.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1],
                hue=subscribers['cluster'],
                palette='tab10', s=50)
plt.title("t-SNE Projection of Predicted Subscribers with KMeans Clusters")
plt.xlabel("t-SNE-1")
plt.ylabel("t-SNE-2")
plt.legend(title='Cluster')
plt.show()


# ---------- Step 6.1: 生成 Tableau-friendly CSV ----------
# 1. 拿原始 df 中筛选出的 subscribers（未预处理）
df_clean = df.loc[subscribers.index].copy()  # 保留原始可读数据

# 2. 添加 cluster 和 t-SNE 坐标
df_clean['cluster'] = subscribers['cluster'].values
df_clean['tsne_x'] = X_tsne[:, 0]
df_clean['tsne_y'] = X_tsne[:, 1]

# 3. 导出为 CSV
df_clean.to_csv("tableau_tsne_clustered_customers.csv", index=False)
print("Tableau-friendly CSV 已生成：tableau_tsne_clustered_customers.csv")


# -----------------------------
# Step 7: 简单画像分析（按簇比例排序）
# -----------------------------
# 计算簇的数量
cluster_counts = subscribers['cluster'].value_counts()
# 计算簇比例
cluster_ratios = cluster_counts / len(subscribers)

# 计算每个簇的平均特征
cluster_summary = subscribers.groupby('cluster').mean()

# 合并数量和比例信息
cluster_summary['count'] = cluster_counts
cluster_summary['ratio'] = cluster_ratios

# 按比例从大到小排序
cluster_summary_sorted = cluster_summary.loc[cluster_ratios.sort_values(
    ascending=False).index]

print("Cluster summary sorted by ratio:")
print(cluster_summary_sorted)

# 可选：导出到 CSV
cluster_summary_sorted.to_csv("predicted_subscribers_cluster_summary.csv")

'''
# -----------------------------
# Step 8: 全量客户在现有簇下的预测 y 分布分析
# -----------------------------
# 对完整数据进行相同的预处理
df_proc_all = model_runner.preprocess_data(df)
features_all = df_proc_all.drop(columns=['y'])
X_scaled_all = scaler.transform(features_all)  # 用训练时的 scaler
X_pca_all = pca.transform(X_scaled_all)        # 用训练时的 PCA

# 使用已训练好的 KMeans 预测簇
df_proc_all['cluster'] = kmeans.predict(X_pca_all)

# 生成预测 y_pred 列（0/1）
df_proc_all['y_pred'] = (model_runner.all_y_proba >=
                         model_runner.best_threshold).astype(int)

# 统计每个簇的预测 y=1 和总数
cluster_stats = df_proc_all.groupby('cluster')['y_pred'].agg(['count', 'sum'])
cluster_stats['ratio_pred'] = cluster_stats['sum'] / cluster_stats['count']

# 按预测率排序
cluster_stats_sorted = cluster_stats.sort_values('ratio_pred', ascending=False)

print("Full customer distribution by cluster (sorted by predicted y=1 ratio):")
print(cluster_stats_sorted)
'''

# -----------------------------
# Step 8: 全量客户在现有簇下的实际订阅分布分析（用真实 y）
# -----------------------------

# 对完整数据进行相同的预处理
df_proc_all = model_runner.preprocess_data(df)  # 预处理仅为了保持列一致
df_proc_all['cluster'] = kmeans.predict(pca.transform(
    scaler.transform(df_proc_all.drop(columns=['y']))))

# 统计每个簇的真实订阅数和总数
cluster_stats = df_proc_all.groupby('cluster')['y'].agg(['count', 'sum'])
cluster_stats['ratio_actual'] = cluster_stats['sum'] / cluster_stats['count']

# 按实际订阅比例排序
cluster_stats_sorted = cluster_stats.sort_values(
    'ratio_actual', ascending=False)

print("Full customer distribution by cluster (sorted by actual y=1 ratio):")
print(cluster_stats_sorted)


# -----------------------------
# Step 8.1: 可视化各簇的预测订阅率 + 标注具体人数（按第7步簇比例排序）
# -----------------------------
plt.figure(figsize=(10, 6))

# 按第7步排序的簇顺序
cluster_order = cluster_summary_sorted.index.tolist()
cluster_order_str = [str(c) for c in cluster_order]  # 转成字符串防止连续轴自动排序

# 取 cluster_stats 中对应顺序的数据
cluster_stats_ordered = cluster_stats.loc[cluster_order]

ax = sns.barplot(
    x=[str(c) for c in cluster_stats_ordered.index],  # 强制分类
    y=cluster_stats_ordered['ratio_actual'],
    palette='viridis',
    order=cluster_order_str  # 指定顺序
)

plt.xlabel("Cluster")
plt.ylabel("Subscription Rate")
plt.title("Subscription Rate by Cluster (ordered by cluster ratio)")
plt.ylim(0, cluster_stats_ordered['ratio_actual'].max() * 1.2)

# 在柱子上标注具体人数（预测订阅数 / 簇总人数）
for i, cluster in enumerate(cluster_stats_ordered.index):
    count_yes = int(cluster_stats_ordered.loc[cluster, 'sum'])
    count_total = int(cluster_stats_ordered.loc[cluster, 'count'])
    y = cluster_stats_ordered.loc[cluster, 'ratio_actual']
    ax.text(i, y + 0.002, f"{count_yes}/{count_total}",
            ha='center', va='bottom', fontsize=10)

plt.show()
