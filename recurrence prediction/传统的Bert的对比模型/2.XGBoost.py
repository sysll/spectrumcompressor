import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
import xgboost as xgb

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 18
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================== 读取数据 ==================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 文本列
numeric_cols = df.columns[:-3].drop("recurrence")  # 数值列
target_col = 'recurrence'

# 假设人口统计学特征在前10列
demo_cols = numeric_cols[:]

# ================== 文本编码 ==================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2").to(device)
model.eval()

texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
batch_size = 8
text_features = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**encoded)
        # 使用 [CLS] token 的向量作为文本表示
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        text_features.append(cls_emb)

text_features = np.vstack(text_features)  # [样本数, hidden_size]

# ================== 人口统计学特征 ==================
X_demo = df[demo_cols].values
scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(X_demo)

# ================== 融合向量 ==================
X = np.hstack([text_features, X_demo])  # 拼接文本特征和人口统计学特征
y = df[target_col].values

# ================== 划分训练集、验证集和测试集 (6:2:2) ==================
# 首先划分训练集和临时集（60% 训练，40% 临时）
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=seed, stratify=y
)

# 然后将临时集划分为验证集和测试集（各占50%，即总数据的20%）
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp
)

print(f"训练集大小: {len(X_train)}")
print(f"验证集大小: {len(X_val)}")
print(f"测试集大小: {len(X_test)}")

# ================== XGBoost 分类器（使用验证集进行早停） ==================
clf = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=seed
)

# 训练并使用验证集进行早停
clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # 使用验证集而不是测试集
    early_stopping_rounds=50,
    verbose=50
)

# 获取最佳迭代次数
best_iteration = clf.best_iteration
if best_iteration is not None:
    print(f"\n最佳迭代次数: {best_iteration}")
    print(f"最佳验证集AUC: {clf.best_score:.4f}")

# ================== 在测试集上评估 ==================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n===== 测试集评估结果（早停模型） =====")
print(classification_report(y_test, y_pred, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_proba):.4f}")

# ================== 可选：用最佳模型重新训练（使用训练+验证集） ==================
print("\n===== 使用训练+验证集重新训练最佳模型 =====")

# 合并训练集和验证集
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

# 使用找到的最佳迭代次数重新训练
clf_best = xgb.XGBClassifier(
    n_estimators=best_iteration if best_iteration else 500,
    learning_rate=0.05,
    max_depth=6,
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=seed
)

clf_best.fit(
    X_train_full, y_train_full,
    eval_set=[(X_val, y_val)],  # 可选：仍然监控验证集
    verbose=False
)

# 在测试集上评估
y_pred_best = clf_best.predict(X_test)
y_proba_best = clf_best.predict_proba(X_test)[:, 1]

print("\n===== 测试集评估结果（重新训练模型） =====")
print(classification_report(y_test, y_pred_best, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred_best):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba_best):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_proba_best):.4f}")
