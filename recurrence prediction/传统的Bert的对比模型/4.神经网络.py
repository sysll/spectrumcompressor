import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 19
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

# ================== 构建 PyTorch 数据集 ==================
class FusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FusionDataset(X_train, y_train)
val_dataset = FusionDataset(X_val, y_val)
test_dataset = FusionDataset(X_test, y_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================== 神经网络模型 ==================
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

input_dim = X.shape[1]
model_nn = FusionNN(input_dim).to(device)

# ================== 损失函数和优化器 ==================
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)

# ================== 训练（带验证集监控和早停） ==================
num_epochs = 100  # 增加最大轮数，让早停机制决定最佳
best_val_auc = 0
best_model_state = None
patience = 10  # 早停耐心值
train_losses = []
val_aucs = []
patience_counter = 0

print("\n===== 开始训练（验证集监控） =====")
for epoch in range(num_epochs):
    # 训练阶段
    model_nn.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        outputs = model_nn(X_batch)
        loss = criterion(outputs, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)

    # 验证阶段
    model_nn.eval()
    all_val_preds, all_val_probs, all_val_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model_nn(X_batch)
            probs = torch.softmax(outputs, dim=1)[:, 1]
            preds = torch.argmax(outputs, dim=1)
            all_val_preds.append(preds.cpu())
            all_val_probs.append(probs.cpu())
            all_val_labels.append(y_batch.cpu())

    all_val_preds = torch.cat(all_val_preds).numpy()
    all_val_probs = torch.cat(all_val_probs).numpy()
    all_val_labels = torch.cat(all_val_labels).numpy()

    val_auc = roc_auc_score(all_val_labels, all_val_probs)
    val_auprc = average_precision_score(all_val_labels, all_val_probs)
    val_aucs.append(val_auc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}")

    # 保存最佳模型
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_model_state = model_nn.state_dict()
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

# ================== 在测试集上评估最佳模型 ==================
model_nn.eval()
all_test_preds, all_test_probs, all_test_labels = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model_nn(X_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        all_test_preds.append(preds.cpu())
        all_test_probs.append(probs.cpu())
        all_test_labels.append(y_batch.cpu())

all_test_preds = torch.cat(all_test_preds).numpy()
all_test_probs = torch.cat(all_test_probs).numpy()
all_test_labels = torch.cat(all_test_labels).numpy()

print("\n===== 测试集评估结果（最佳模型） =====")
print(classification_report(all_test_labels, all_test_preds, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(all_test_labels, all_test_preds):.4f}")
print(f"AUC: {roc_auc_score(all_test_labels, all_test_probs):.4f}")
print(f"AUPRC: {average_precision_score(all_test_labels, all_test_probs):.4f}")

# ================== 可选：使用训练+验证集重新训练最佳模型 ==================
print("\n===== 使用训练+验证集重新训练最佳模型 =====")

# 合并训练集和验证集
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

train_full_dataset = FusionDataset(X_train_full, y_train_full)
train_full_loader = DataLoader(train_full_dataset, batch_size=32, shuffle=True)

# 重新初始化模型
model_final = FusionNN(input_dim).to(device)
optimizer_final = torch.optim.Adam(model_final.parameters(), lr=1e-3)

# 使用最佳epoch数重新训练（或者训练直到收敛）
best_epoch = epoch + 1 - patience_counter
print(f"使用最佳epoch数: {best_epoch}")

for epoch in range(best_epoch):
    model_final.train()
    for X_batch, y_batch in train_full_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model_final(X_batch)
        loss = criterion(logits, y_batch)
        optimizer_final.zero_grad()
        loss.backward()
        optimizer_final.step()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{best_epoch}")

# 在测试集上评估
model_final.eval()
all_test_preds_final, all_test_probs_final, all_test_labels_final = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model_final(X_batch)
        probs = torch.softmax(logits, dim=1)[:, 1]
        preds = torch.argmax(logits, dim=1)
        all_test_preds_final.append(preds.cpu())
        all_test_probs_final.append(probs.cpu())
        all_test_labels_final.append(y_batch.cpu())

all_test_preds_final = torch.cat(all_test_preds_final).numpy()
all_test_probs_final = torch.cat(all_test_probs_final).numpy()
all_test_labels_final = torch.cat(all_test_labels_final).numpy()

print("\n===== 测试集评估结果（重新训练模型） =====")
print(classification_report(all_test_labels_final, all_test_preds_final, digits=4))
print(f"准确率 (Accuracy): {accuracy_score(all_test_labels_final, all_test_preds_final):.4f}")
print(f"AUC: {roc_auc_score(all_test_labels_final, all_test_probs_final):.4f}")
print(f"AUPRC: {average_precision_score(all_test_labels_final, all_test_probs_final):.4f}")
