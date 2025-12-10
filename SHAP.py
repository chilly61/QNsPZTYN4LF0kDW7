# post_campaign_shap.py
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from XGBALG import TermDepositModel  # 假设你原来的类保存为 term_deposit_model.py

# ---------------------------
# 配置
# ---------------------------
DATA_PATH = "term-deposit-marketing-2020.csv"
LOG_COLS = ['balance', 'duration']  # 和你原模型保持一致

# ---------------------------
# 1. 加载数据
# ---------------------------
df = pd.read_csv(DATA_PATH)

# ---------------------------
# 2. 初始化模型类
# ---------------------------
model_runner = TermDepositModel(log_cols=LOG_COLS)

# ---------------------------
# 3. 数据预处理 + 模型训练
# ---------------------------
model_runner.train_cv(df)
model_runner.search_global_threshold(min_acc=0.81)

# ---------------------------
# 4. 获取训练数据用于 SHAP
# ---------------------------
df_proc = model_runner.preprocess_data(df)
X = df_proc.drop(columns=['y']).values
y = df_proc['y'].values

# ---------------------------
# 5. SHAP 分析
# ---------------------------
explainer = shap.TreeExplainer(model_runner.model)
shap_values = explainer.shap_values(X)

# ---------------------------
# 6. SHAP Summary Plot
# ---------------------------
plt.figure()
shap.summary_plot(shap_values, X, feature_names=df_proc.drop(columns=['y']).columns,
                  plot_type="dot", show=True)

# ---------------------------
# 7. SHAP Bar Plot（平均绝对值）
# ---------------------------
plt.figure()
shap.summary_plot(shap_values, X, feature_names=df_proc.drop(columns=['y']).columns,
                  plot_type="bar", show=True)

# ---------------------------
# 8. 可选：保存 SHAP values 到 CSV
# ---------------------------
shap_df = pd.DataFrame(
    shap_values, columns=df_proc.drop(columns=['y']).columns)
shap_df.to_csv("shap_values.csv", index=False)
print("SHAP analysis complete. Values saved to shap_values.csv")


# ---------------------------
# 8. SHAP Feature Importance（平均绝对值）
# ---------------------------

# 计算每个特征的平均绝对值 SHAP
shap_abs_mean = np.abs(shap_values).mean(axis=0)
feature_importance = pd.DataFrame({
    'feature': df_proc.drop(columns=['y']).columns,
    'importance': shap_abs_mean
})

# 按重要性排序（从大到小）
feature_importance = feature_importance.sort_values(
    by='importance', ascending=False)

# 保存为 CSV
feature_importance.to_csv("shap_feature_importance.csv", index=False)
print("SHAP feature importance saved to shap_feature_importance.csv")
