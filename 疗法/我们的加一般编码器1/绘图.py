import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay

# ================== 读取 numpy 文件 ==================
y_pred = np.load("best_epoch_probs.npy")      # 模型预测类别
y_true = np.load("best_epoch_labels.npy")     # 真实标签

# 如果是二分类，需要预测概率来画 ROC/PR
# 假设你有 softmax 输出的概率保存在 npy 文件中，这里示例用 0/1 预测直接做 ROC 近似
# 对真实任务，你应该保存 softmax 概率（shape: [num_samples, num_classes]）
# y_prob = np.load("best_epoch_probs.npy")[:, 1]  # 取正类概率
y_prob = y_pred  # 临时用预测类别代替概率（仅示例）

# ================== ROC 曲线 ==================
fpr, tpr, _ = roc_curve(y_true, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC = {roc_auc:.4f})')
plt.plot([0,1], [0,1], color='r', linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.grid(True)
plt.show()

# ================== PR 曲线 ==================
precision, recall, _ = precision_recall_curve(y_true, y_prob)
avg_precision = average_precision_score(y_true, y_prob)

plt.figure(figsize=(6,6))
plt.plot(recall, precision, color='g', label=f'PR curve (AP = {avg_precision:.4f})')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="upper right")
plt.grid(True)
plt.show()

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

# 如果 y_pred 是概率，则需要二值化
if y_pred.dtype != int and y_pred.dtype != np.int64:
    y_pred = (y_pred >= 0.5).astype(int)

# 计算混淆矩阵
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
