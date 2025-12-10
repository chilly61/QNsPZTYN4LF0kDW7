import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from XGBALG import TermDepositModel  # 导入你的类
from xgboost import plot_importance

# 1. 读取原始数据
df = pd.read_csv("term-deposit-marketing-2020.csv")

# 2. 初始化并训练模型
model_runner = TermDepositModel(log_cols=['balance', 'duration'])
model_runner.train_cv(df)
model_runner.search_global_threshold(min_acc=0.81)

# 3. 全量预测概率
X_processed = model_runner.preprocess_data(df).drop(columns=['y']).values
y_proba = model_runner.model.predict_proba(X_processed)[:, 1]

# 4. 根据全局最佳阈值筛选 Top 客户
best_threshold = model_runner.best_threshold
best_threshold = 0.7
top_idx = np.where(y_proba >= best_threshold)[0]
top_customers = df.iloc[top_idx]

print(f"Number of top customers: {len(top_customers)}")
print(top_customers.describe(include='all'))

# 5. Top 客户概率分布
plt.figure(figsize=(8, 5))
plt.hist(y_proba[top_idx], bins=20, color='orange', alpha=0.7)
plt.axvline(best_threshold, color='red', linestyle='--',
            label=f'Threshold={best_threshold}')
plt.xlabel("Predicted Probability")
plt.ylabel("Count")
plt.title("Top Customer Probability Distribution")
plt.legend()
plt.show()

# 6. 类别特征分布示例
for col in ['job', 'marital', 'education']:
    plt.figure(figsize=(6, 4))
    top_customers[col].value_counts().sort_index().plot(kind='bar')
    plt.title(f"Distribution of {col} (Top Customers)")
    plt.xlabel(col)
    plt.ylabel("Count")
    plt.show()


booster = model_runner.model.get_booster()
booster.feature_names = list(
    model_runner.preprocess_data(df).drop(columns=['y']).columns)

# 7. 特征重要性
plt.figure(figsize=(8, 6))
plot_importance(model_runner.model, max_num_features=15,
                importance_type='weight')
plt.title("XGBoost Feature Importance (Top 15)")
plt.show()
