# term-deposit-marketing-2020.csv
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import math

# 1. 读取数据
df = pd.read_csv("term-deposit-marketing-2020.csv")

# 2. 检查重复样本
dup_count = df.duplicated().sum()
print(f"重复样本数量：{dup_count}")

# 3. 区分变量类型
target_col = "y"
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# 如果目标列在categorical中，去掉它
if target_col in categorical_cols:
    categorical_cols.remove(target_col)

print(f"数值型特征: {numeric_cols}")
print(f"类别型特征: {categorical_cols}")

# 4. 合并特征列
all_features = numeric_cols + categorical_cols
n_features = len(all_features)

# 5. 创建统一子图
n_cols = 5  # 每行显示4个图
n_rows = math.ceil(n_features / n_cols)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, n_rows * 5))
axes = axes.flatten()

# 6. 绘制每个特征的分布
for i, col in enumerate(all_features):
    ax = axes[i]
    if col in numeric_cols:
        sns.histplot(df[col], kde=True, bins=30, ax=ax)
        ax.set_title(f"{col} (numeric)", fontsize=11)
    else:
        # 类别型特征
        order = df[col].value_counts().index[:15]  # 只显示前15个类别
        sns.countplot(y=col, data=df, order=order, ax=ax)
        ax.set_title(f"{col} (categorical)", fontsize=11)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', labelsize=9)

# 7. 删除多余空子图
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

# 8. 调整布局
plt.tight_layout()
plt.show()
