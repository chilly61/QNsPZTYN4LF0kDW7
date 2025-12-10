import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler, RobustScaler
from xgboost import XGBClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


class TermDepositModel:
    def __init__(self, log_cols=None, scaler_type='robust', n_splits=5, random_state=42):
        """
        log_cols: list of numeric columns to do signed_log transform
        scaler_type: 'robust', 'standard', or None
        n_splits: 5-fold CV
        """
        self.log_cols = log_cols
        self.scaler_type = scaler_type
        self.n_splits = n_splits
        self.random_state = random_state
        self.label_encoders = {}
        self.model = None
        self.best_threshold = 0.5

    @staticmethod
    def signed_log_transform(x):
        x = np.array(x)
        return np.sign(x) * np.log1p(np.abs(x))

    def preprocess_numeric(self, df, numeric_cols):
        df_proc = df.copy()
        for col in numeric_cols:
            df_proc[col] = self.signed_log_transform(df_proc[col])
        if self.scaler_type is not None:
            if self.scaler_type == 'robust':
                scaler = RobustScaler()
            elif self.scaler_type == 'standard':
                scaler = StandardScaler()
            else:
                raise ValueError("scaler_type must be 'robust' or 'standard'")
            df_proc[numeric_cols] = scaler.fit_transform(df_proc[numeric_cols])
        return df_proc

    def preprocess_data(self, df):
        df_proc = df.drop_duplicates().copy()
        # target
        df_proc['y'] = df_proc['y'].map({'yes': 1, 'no': 0})
        # numeric transform
        if self.log_cols is not None:
            df_proc = self.preprocess_numeric(df_proc, self.log_cols)
        # categorical encoding
        for col in df_proc.select_dtypes(include='object').columns:
            if col != 'y':
                le = LabelEncoder()
                df_proc[col] = le.fit_transform(df_proc[col].astype(str))
                self.label_encoders[col] = le
        return df_proc

    def train_cv(self, df):
        df_proc = self.preprocess_data(df)
        X = df_proc.drop(columns=['y']).values
        y = df_proc['y'].values

        kf = StratifiedKFold(n_splits=self.n_splits,
                             shuffle=True, random_state=self.random_state)

        all_y_true = []
        all_y_proba = []

        for fold, (train_idx, test_idx) in enumerate(kf.split(X, y), 1):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]

            pos = sum(y_train == 1)
            neg = sum(y_train == 0)
            scale_pos_weight = neg / pos

            model = XGBClassifier(
                use_label_encoder=False,
                eval_metric='logloss',
                scale_pos_weight=scale_pos_weight,
                random_state=self.random_state
            )
            model.fit(X_train, y_train)
            y_proba = model.predict_proba(X_test)[:, 1]

            all_y_true.append(y_test)
            all_y_proba.append(y_proba)

        # 合并所有折
        self.all_y_true = np.concatenate(all_y_true)
        self.all_y_proba = np.concatenate(all_y_proba)

        # 训练最终模型（全数据）
        pos = sum(y == 1)
        neg = sum(y == 0)
        scale_pos_weight = neg / pos
        self.model = XGBClassifier(
            use_label_encoder=False,
            eval_metric='logloss',
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state
        )
        self.model.fit(X, y)

        print("5-fold CV finished.")

    def search_global_threshold(self, min_acc=0.81):
        thresholds = np.linspace(0, 0.9, 91)
        best_recall = -1
        best_threshold = 0.5
        acc_curve = []
        rec_curve = []
        prec_curve = []

        for t in thresholds:
            y_pred_t = (self.all_y_proba >= t).astype(int)
            acc = accuracy_score(self.all_y_true, y_pred_t)
            rec = recall_score(self.all_y_true, y_pred_t)
            prec = precision_score(self.all_y_true, y_pred_t)
            acc_curve.append(acc)
            rec_curve.append(rec)
            prec_curve.append(prec)
            if acc >= min_acc and rec > best_recall:
                best_recall = rec
                best_threshold = t

        self.best_threshold = best_threshold
        self.acc_curve = acc_curve
        self.rec_curve = rec_curve
        self.prec_curve = prec_curve

        print(f"Global best threshold: {self.best_threshold:.2f}")
        return self.best_threshold

    def plot_threshold_curves(self):
        plt.figure(figsize=(8, 5))
        thresholds = np.linspace(0, 0.9, 91)
        plt.plot(thresholds, self.acc_curve, label='Accuracy')
        plt.plot(thresholds, self.rec_curve, label='Recall (y=1)')
        plt.plot(thresholds, self.prec_curve, label='Precision (y=1)')
        plt.axvline(self.best_threshold, color='red', linestyle='--',
                    label=f'Best Threshold={self.best_threshold:.2f}')
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title("Threshold vs Metrics (Global)")
        plt.legend()
        plt.show()

    def plot_positive_proba(self):
        plt.figure(figsize=(8, 5))
        plt.hist(self.all_y_proba[self.all_y_true == 1], bins=30,
                 alpha=0.7, color='orange', label='Positive class y=1')
        plt.axvline(self.best_threshold, color='red', linestyle='--',
                    label=f'Best Threshold={self.best_threshold:.2f}')
        plt.xlabel("Predicted Probability of y=1")
        plt.ylabel("Count")
        plt.title("Predicted Probability Distribution (Positive Class Only)")
        plt.legend()
        plt.show()

    def plot_confusion_matrix(self):
        y_pred_best = (self.all_y_proba >= self.best_threshold).astype(int)
        cm = confusion_matrix(self.all_y_true, y_pred_best)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=[0, 1], yticklabels=[0, 1])
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title(
            f"Confusion Matrix (Global Best Threshold={self.best_threshold:.2f})")
        plt.show()

    def get_metrics(self):
        y_pred_best = (self.all_y_proba >= self.best_threshold).astype(int)
        acc = accuracy_score(self.all_y_true, y_pred_best)
        rec = recall_score(self.all_y_true, y_pred_best)
        prec = precision_score(self.all_y_true, y_pred_best)
        f1 = f1_score(self.all_y_true, y_pred_best)
        return {"accuracy": acc, "recall": rec, "precision": prec, "f1": f1}


df = pd.read_csv("term-deposit-marketing-2020.csv")
# 初始化类
model_runner = TermDepositModel(log_cols=['balance', 'duration'])

# 训练 5-fold CV + 全数据训练最终模型
model_runner.train_cv(df)

# 搜索全局最佳阈值（Accuracy ≥ 0.81）
model_runner.search_global_threshold(min_acc=0.81)

# 查看最终指标
metrics = model_runner.get_metrics()
print(metrics)

# 绘图
model_runner.plot_threshold_curves()
model_runner.plot_positive_proba()
model_runner.plot_confusion_matrix()
