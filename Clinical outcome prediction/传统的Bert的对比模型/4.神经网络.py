import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score
import copy
import matplotlib.pyplot as plt

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 16
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================== 读取数据 ==================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 文本列
numeric_cols = df.columns[:-3].drop("survival_status")  # 数值列
target_col = 'survival_status'

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
        batch_texts = texts[i:i + batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        outputs = model(**encoded)
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS] token
        text_features.append(cls_emb)

text_features = np.vstack(text_features)  # [样本数, hidden_size]

# ================== 人口统计学特征 ==================
X_demo = df[demo_cols].values
scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(X_demo)

# ================== 融合向量 ==================
X = np.hstack([text_features, X_demo])
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


# ================== 神经网络分类器 ==================
class FusionNN(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 2))  # 二分类输出
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


input_dim = X_train.shape[1]
model_nn = FusionNN(input_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)
num_epochs = 100  # 增加最大轮数，让早停机制决定最佳
batch_size = 32
patience = 10  # 早停耐心值

# ================== 数据转为 Tensor ==================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# ================== 训练（带验证集监控和早停） ==================
best_val_auc = 0
best_model_state = None
patience_counter = 0
train_losses = []
val_aucs = []

print("\n===== 开始训练（验证集监控） =====")
for epoch in range(num_epochs):
    # 训练阶段
    model_nn.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model_nn(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)

    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)

    # 验证阶段
    model_nn.eval()
    with torch.no_grad():
        val_outputs = model_nn(X_val_tensor)
        val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
        val_auc = roc_auc_score(y_val, val_probs)
        val_auprc = average_precision_score(y_val, val_probs)
        val_aucs.append(val_auc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}")

    # 保存最佳模型
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = copy.deepcopy(model_nn.state_dict())
        patience_counter = 0
        print(f"  -> 新的最佳模型！验证集AUC: {best_val_auc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n早停：验证集AUC连续{patience}轮未提升，停止训练")
            break

# 加载最佳模型
model_nn.load_state_dict(best_model_state)
print(f"\n最佳验证集AUC: {best_val_auc:.4f} (Epoch {epoch + 1 - patience_counter})")

# ================== 绘制训练曲线 ==================
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(train_losses)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('训练损失曲线')
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(val_aucs)
plt.axhline(y=best_val_auc, color='r', linestyle='--', label=f'最佳AUC: {best_val_auc:.4f}')
plt.xlabel('Epoch')
plt.ylabel('AUC')
plt.title('验证集AUC曲线')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig('nn_training_curves.png')
plt.show()

# ================== 在测试集上评估最佳模型 ==================
model_nn.eval()
with torch.no_grad():
    test_outputs = model_nn(X_test_tensor)
    test_probs = torch.softmax(test_outputs, dim=1)[:, 1].cpu().numpy()
    test_preds = torch.argmax(test_outputs, dim=1).cpu().numpy()

print("\n===== 测试集评估结果（最佳模型） =====")
print(classification_report(y_test, test_preds, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(y_test, test_preds):.4f}")
print(f"AUC: {roc_auc_score(y_test, test_probs):.4f}")
print(f"AUPRC: {average_precision_score(y_test, test_probs):.4f}")

# ================== 可选：使用训练+验证集重新训练最佳模型 ==================
print("\n===== 使用训练+验证集重新训练最佳模型 =====")

# 合并训练集和验证集
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

X_train_full_tensor = torch.tensor(X_train_full, dtype=torch.float32).to(device)
y_train_full_tensor = torch.tensor(y_train_full, dtype=torch.long).to(device)

train_full_dataset = torch.utils.data.TensorDataset(X_train_full_tensor, y_train_full_tensor)
train_full_loader = torch.utils.data.DataLoader(train_full_dataset, batch_size=batch_size, shuffle=True)

# 重新初始化模型
model_final = FusionNN(input_dim).to(device)
optimizer_final = torch.optim.Adam(model_final.parameters(), lr=1e-3)

# 使用最佳epoch数重新训练（或者训练直到收敛）
best_epoch = epoch + 1 - patience_counter
print(f"使用最佳epoch数: {best_epoch}")

for epoch in range(best_epoch):
    model_final.train()
    for xb, yb in train_full_loader:
        optimizer_final.zero_grad()
        outputs = model_final(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer_final.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{best_epoch}")

# 在测试集上评估
model_final.eval()
with torch.no_grad():
    test_outputs_final = model_final(X_test_tensor)
    test_probs_final = torch.softmax(test_outputs_final, dim=1)[:, 1].cpu().numpy()
    test_preds_final = torch.argmax(test_outputs_final, dim=1).cpu().numpy()

print("\n===== 测试集评估结果（重新训练模型） =====")
print(classification_report(y_test, test_preds_final, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(y_test, test_preds_final):.4f}")
print(f"AUC: {roc_auc_score(y_test, test_probs_final):.4f}")
print(f"AUPRC: {average_precision_score(y_test, test_probs_final):.4f}")

# ================== 错误分析 ==================
print("\n===== 错误分析 =====")
errors = (test_preds_final != y_test)
print(f"错误分类样本数: {np.sum(errors)}/{len(y_test)} ({np.mean(errors) * 100:.2f}%)")

# 分析错误类型
false_positives = (test_preds_final == 1) & (y_test == 0)
false_negatives = (test_preds_final == 0) & (y_test == 1)
print(f"假阳性 (FP): {np.sum(false_positives)}")
print(f"假阴性 (FN): {np.sum(false_negatives)}")

# 如果有概率预测，可以查看置信度分布
print(f"\n正类预测概率分布:")
print(f"  均值: {test_probs_final.mean():.4f}")
print(f"  标准差: {test_probs_final.std():.4f}")
print(f"  最小值: {test_probs_final.min():.4f}")
print(f"  最大值: {test_probs_final.max():.4f}")