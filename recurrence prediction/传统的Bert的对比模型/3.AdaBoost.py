import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

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

# ================== AdaBoost 分类器（带验证集监控） ==================
base_clf = DecisionTreeClassifier(max_depth=1, random_state=seed)  # 弱分类器

# 训练多个AdaBoost模型，逐步增加n_estimators
n_estimators_list = [50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
best_val_auc = 0
best_n_estimators = 50
best_model = None
val_auc_scores = []
train_auc_scores = []

print("\n===== 验证集调参 =====")
for n_est in n_estimators_list:
    # 训练模型
    clf = AdaBoostClassifier(
        estimator=base_clf,
        n_estimators=n_est,
        learning_rate=0.05,
        random_state=seed
    )
    clf.fit(X_train, y_train)

    # 在验证集上评估
    y_val_proba = clf.predict_proba(X_val)[:, 1]
    val_auc = roc_auc_score(y_val, y_val_proba)
    val_auprc = average_precision_score(y_val, y_val_proba)
    val_auc_scores.append(val_auc)

    # 在训练集上评估
    y_train_proba = clf.predict_proba(X_train)[:, 1]
    train_auc = roc_auc_score(y_train, y_train_proba)
    train_auc_scores.append(train_auc)

    print(f"n_estimators={n_est}: 训练集AUC={train_auc:.4f}, 验证集AUC={val_auc:.4f}, 验证集AUPRC={val_auprc:.4f}")

    # 保存最佳模型
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_n_estimators = n_est
        best_model = clf

print(f"\n最佳n_estimators: {best_n_estimators}, 最佳验证集AUC: {best_val_auc:.4f}")

# ================== 绘制学习曲线 ==================
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, train_auc_scores, 'b-', label='训练集AUC', marker='o')
plt.plot(n_estimators_list, val_auc_scores, 'r-', label='验证集AUC', marker='s')
plt.axvline(x=best_n_estimators, color='g', linestyle='--', label=f'最佳n_estimators={best_n_estimators}')
plt.xlabel('n_estimators')
plt.ylabel('AUC')
plt.title('AdaBoost: 验证集调参曲线')
plt.legend()
plt.grid(True)
plt.savefig('adaboost_validation_curve.png')
plt.show()

# ================== 使用最佳模型在测试集上评估 ==================
y_pred = best_model.predict(X_test)
y_proba = best_model.predict_proba(X_test)[:, 1]

print("\n===== 测试集评估结果（最佳模型） =====")
print(classification_report(y_test, y_pred, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_proba):.4f}")

# ================== 可选：使用最佳n_estimators在训练+验证集上重新训练 ==================
print("\n===== 使用训练+验证集重新训练最佳模型 =====")

# 合并训练集和验证集
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

# 使用找到的最佳n_estimators重新训练
clf_best = AdaBoostClassifier(
    estimator=base_clf,
    n_estimators=best_n_estimators,
    learning_rate=0.05,
    random_state=seed
)

clf_best.fit(X_train_full, y_train_full)

# 在测试集上评估
y_pred_best = clf_best.predict(X_test)
y_proba_best = clf_best.predict_proba(X_test)[:, 1]

print("\n===== 测试集评估结果（重新训练模型） =====")
print(classification_report(y_test, y_pred_best, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(y_test, y_pred_best):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_proba_best):.4f}")
print(f"AUPRC: {average_precision_score(y_test, y_proba_best):.4f}")

# ================== 特征重要性 ==================
feature_names = [f"text_feat_{i}" for i in range(text_features.shape[1])] + list(demo_cols)
feature_importance = clf_best.feature_importances_
top_features_idx = np.argsort(feature_importance)[-20:][::-1]  # 取前20个最重要的特征

print("\n===== 特征重要性 (Top 20) =====")
for idx in top_features_idx:
    print(f"{feature_names[idx]}: {feature_importance[idx]:.4f}")

# ================== 错误分析 ==================
print("\n===== 错误分析 =====")
errors = (y_pred_best != y_test)
print(f"错误分类样本数: {np.sum(errors)}/{len(y_test)} ({np.mean(errors) * 100:.2f}%)")

# 分析错误类型
false_positives = (y_pred_best == 1) & (y_test == 0)
false_negatives = (y_pred_best == 0) & (y_test == 1)
print(f"假阳性 (FP): {np.sum(false_positives)}")
print(f"假阴性 (FN): {np.sum(false_negatives)}")

# ================== 学习曲线分析 ==================
# 分析随着弱分类器数量增加的性能变化
staged_scores_train = []
staged_scores_val = []

# 获取阶段性预测结果
for i, y_pred_staged in enumerate(best_model.staged_predict_proba(X_train)):
    if i % 50 == 0:  # 每50个分类器记录一次
        auc = roc_auc_score(y_train, y_pred_staged[:, 1])
        staged_scores_train.append((i + 1, auc))

for i, y_pred_staged in enumerate(best_model.staged_predict_proba(X_val)):
    if i % 50 == 0:
        auc = roc_auc_score(y_val, y_pred_staged[:, 1])
        staged_scores_val.append((i + 1, auc))

print("\n===== 阶段性性能 =====")
print("迭代次数\t训练集AUC\t验证集AUC")
for i in range(min(len(staged_scores_train), len(staged_scores_val))):
    print(f"{staged_scores_train[i][0]}\t\t{staged_scores_train[i][1]:.4f}\t\t{staged_scores_val[i][1]:.4f}")
