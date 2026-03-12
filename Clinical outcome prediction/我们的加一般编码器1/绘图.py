import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

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
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100  # 转百分比

# ================== 绘图 ==================
sns.set(style='whitegrid', font_scale=1.2)
fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# -------- ROC 曲线 --------
axes[0].plot(fpr, tpr, color='dodgerblue', lw=3, label=f'AUC = {roc_auc:.2f}')
axes[0].fill_between(fpr, 0, tpr, color='dodgerblue', alpha=0.2)
axes[0].plot([0, 1], [0, 1], color='gray', linestyle='--')
axes[0].set_xlabel('False Positive Rate', fontsize=14)
axes[0].set_ylabel('True Positive Rate', fontsize=14)
axes[0].set_title('ROC Curve', fontsize=16, fontweight='bold')
axes[0].legend(loc='lower right', fontsize=12)
axes[0].tick_params(axis='both', which='major', labelsize=12)

# -------- PR 曲线 --------
axes[1].plot(recall, precision, color='green', lw=3, label=f'AP = {ap_score:.2f}')
axes[1].fill_between(recall, 0, precision, color='green', alpha=0.2)
axes[1].set_xlabel('Recall', fontsize=14)
axes[1].set_ylabel('Precision', fontsize=14)
axes[1].set_title('Precision-Recall Curve', fontsize=16, fontweight='bold')
axes[1].legend(loc='lower left', fontsize=12)
axes[1].tick_params(axis='both', which='major', labelsize=12)

# -------- 混淆矩阵 --------
sns.heatmap(cm_percent, annot=True, fmt=".1f", cmap="Blues", cbar=False,
            xticklabels=['Pred 0', 'Pred 1'], yticklabels=['True 0', 'True 1'], ax=axes[2])
axes[2].set_title('Confusion Matrix (%)', fontsize=16, fontweight='bold')
axes[2].set_xlabel('')
axes[2].set_ylabel('')

plt.tight_layout()
plt.show()
from sklearn.metrics import classification_report

# ================== 计算 classification report ==================
report = classification_report(y_true, y_pred_label, target_names=['Class 0', 'Class 1'])
print("Classification Report:\n")
print(report)
