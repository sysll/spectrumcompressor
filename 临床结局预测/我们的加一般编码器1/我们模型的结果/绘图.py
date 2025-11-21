import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import rcParams

# ================== 读取数据 ==================
y_pred = np.load("best_epoch_probs.npy")  # 模型预测概率
y_true = np.load("best_epoch_labels.npy")  # 真实标签

# 如果 y_pred 是多维概率，需要选择正类的概率
if y_pred.ndim > 1 and y_pred.shape[1] > 1:
    y_prob = y_pred[:, 1]
else:
    y_prob = y_pred

# ================== 计算 ROC ==================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

# ================== 计算 PR ==================
precision, recall, _ = precision_recall_curve(y_true, y_prob)
ap_score = average_precision_score(y_true, y_prob)

# ================== 混淆矩阵 ==================
y_pred_label = (y_prob >= 0.5).astype(int)
cm = confusion_matrix(y_true, y_pred_label)

# ✅ 按列（纵向）归一化（每列加和为 100%）
cm_percent = cm.astype('float') / cm.sum(axis=0, keepdims=True) * 100

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
axes[1].plot(recall, precision, color='green', lw=4, label=f'AP = {ap_score:.5f}')
axes[1].fill_between(recall, 0, precision, color='green', alpha=0.2)
axes[1].set_xlabel('Recall', fontsize=18)
axes[1].set_ylabel('Precision', fontsize=18)
axes[1].set_title('Precision-Recall Curve', fontsize=20, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=16)
axes[1].tick_params(axis='both', which='major', labelsize=14)

# -------- 混淆矩阵 --------
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'],
            ax=axes[2], linewidths=2, linecolor='black', annot_kws={"size":16})
axes[2].set_title('Confusion Matrix (% per column)', fontsize=20, fontweight='bold')
axes[2].set_xlabel('')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()

# ================== classification report ==================
from sklearn.metrics import classification_report

report = classification_report(y_true, y_pred_label, target_names=['Class 0', 'Class 1'], digits=4)
print("Classification Report:\n")
print(report)
