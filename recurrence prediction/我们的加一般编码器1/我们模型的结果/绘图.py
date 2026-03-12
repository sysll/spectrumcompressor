import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns

# ================== 读取数据 ==================
y_pred = np.load("best_epoch_probs.npy")   # 模型预测概率（通常是 [neg_prob, pos_prob] 或单列正类概率）
y_true = np.load("best_epoch_labels.npy")  # 真实标签（0 或 1）

# 确保 y_prob 是正类的概率
if y_pred.ndim > 1 and y_pred.shape[1] > 1:
    y_prob = y_pred[:, 1]
else:
    y_prob = y_pred

# ================== 计算所有指标 ==================
# ROC
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# PR & AUPRC (整体)
precision, recall, _ = precision_recall_curve(y_true, y_prob)
auprc = average_precision_score(y_true, y_prob)   # 这就是 AUPRC

# ================== 计算每个类别的 AUPRC ==================
# 对于二分类问题，计算类别0的AUPRC（将类别0视为正类）
y_true_class0 = 1 - y_true  # 将类别0转为正类（1），类别1转为负类（0）
y_prob_class0 = 1 - y_prob  # 类别0的概率 = 1 - 类别1的概率

# 计算类别0的AUPRC
auprc_class0 = average_precision_score(y_true_class0, y_prob_class0)

# 计算类别1的AUPRC（已经计算过了，但为了完整性重新计算）
auprc_class1 = average_precision_score(y_true, y_prob)

# 如果有多个类别（多分类），可以使用以下通用方法
# from sklearn.preprocessing import label_binarize
# classes = np.unique(y_true)
# y_true_bin = label_binarize(y_true, classes=classes)
# auprc_per_class = {}
# for i, class_name in enumerate(['Class 0', 'Class 1']):
#     auprc_per_class[class_name] = average_precision_score(y_true_bin[:, i], y_pred[:, i])

# 预测标签（阈值 0.5）
y_pred_label = (y_prob >= 0.5).astype(int)

# 混淆矩阵 & 列归一化
cm = confusion_matrix(y_true, y_pred_label)
cm_percent = cm.astype('float') / cm.sum(axis=0, keepdims=True) * 100
cm_percent = np.nan_to_num(cm_percent)  # 防止除以 0 出现 nan

# ================== 绘图 ==================
sns.set(style='whitegrid')
plt.rcParams.update({'font.size': 16})

fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor='white')

# 给每个子图加黑色粗边框
for ax in axes:
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(2)

# -------- ROC 曲线 --------
axes[0].plot(fpr, tpr, color='dodgerblue', lw=4, label=f'AUC = {roc_auc:.5f}')
axes[0].fill_between(fpr, 0, tpr, color='dodgerblue', alpha=0.2)
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--', lw=2)
axes[0].set_xlabel('False Positive Rate', fontsize=18)
axes[0].set_ylabel('True Positive Rate', fontsize=18)
axes[0].set_title('ROC Curve', fontsize=20, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=16)
axes[0].tick_params(axis='both', which='major', labelsize=14)

# -------- PR 曲线 --------
axes[1].plot(recall, precision, color='green', lw=4, label=f'AUPRC (Class 1) = {auprc_class1:.5f}')
axes[1].fill_between(recall, 0, precision, color='green', alpha=0.2)
axes[1].set_xlabel('Recall', fontsize=18)
axes[1].set_ylabel('Precision', fontsize=18)
axes[1].set_title('Precision-Recall Curve (Class 1)', fontsize=20, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=14)

# -------- 混淆矩阵（列归一化） --------
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
            ax=axes[2], linewidths=2, linecolor='black', annot_kws={"size":16})
axes[2].set_title('Confusion Matrix (% per column)', fontsize=20, fontweight='bold')
axes[2].set_xlabel('')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()

# ================== classification report + AUPRC ==================
print("\n" + "="*60)
print("Classification Report:")
print("="*60)
report = classification_report(y_true, y_pred_label, target_names=['Class 0', 'Class 1'], digits=4)
print(report)

print("-" * 60)
print("Additional Performance Metrics:")
print(f"  ROC-AUC (overall)   : {roc_auc:.5f}")
print(f"  AUPRC (Class 1)     : {auprc_class1:.5f}")
print(f"  AUPRC (Class 0)     : {auprc_class0:.5f}")
print("-" * 60)

# 可选：计算宏观平均AUPRC和加权平均AUPRC
macro_auprc = (auprc_class0 + auprc_class1) / 2
# 计算加权平均（根据类别样本数）
class_counts = np.bincount(y_true)
weighted_auprc = (auprc_class0 * class_counts[0] + auprc_class1 * class_counts[1]) / len(y_true)

print("\n" + "="*60)
print("Aggregate AUPRC Metrics:")
print("="*60)
print(f"  Macro-average AUPRC   : {macro_auprc:.5f}")
print(f"  Weighted-average AUPRC: {weighted_auprc:.5f}")
print("="*60)