import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 21
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================== 自定义文本编码器 ==================
class SpectralCompressor(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
        super().__init__()
        self.seq2 = seq2
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_dim = self.model.config.hidden_size
        self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.freq_gate = nn.Parameter(torch.randn(seq2, self.hidden_dim))

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs = outputs.hidden_states
        first, middle, last = hs[1], hs[len(hs)//2], hs[-1]
        concat = torch.cat([first, middle, last], dim=-1)
        concat = self.linear_proj(concat)

        # 频谱压缩
        freq = fft.rfft(concat, dim=1)
        seq_len = freq.size(1)
        idx = torch.linspace(0, seq_len - 1, steps=self.seq2, device=freq.device)
        idx_floor = idx.long()
        idx_ceil = torch.clamp(idx_floor + 1, max=seq_len - 1)
        w = (idx - idx_floor).unsqueeze(0).unsqueeze(-1)
        freq_down = (1 - w) * freq[:, idx_floor, :] + w * freq[:, idx_ceil, :]
        gate = torch.sigmoid(self.freq_gate).unsqueeze(0)
        freq_weighted = freq_down * gate
        compressed = fft.irfft(freq_weighted, n=self.seq2, dim=1)
        return compressed.real  # [batch_size, seq2, hidden_dim]

# ================== 读取数据 ==================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]
numeric_cols = df.columns[:-3].drop("recurrence")
target_col = 'recurrence'
demo_cols = numeric_cols[:]

# ================== 文本编码 ==================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
spectral_encoder = SpectralCompressor().to(device)
spectral_encoder.eval()

texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
batch_size = 8
text_features = []

with torch.no_grad():
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        encoded = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        compressed = spectral_encoder(encoded["input_ids"], encoded["attention_mask"])
        cls_emb = compressed.mean(dim=1).cpu().numpy()  # 对序列维度取平均
        text_features.append(cls_emb)

text_features = np.vstack(text_features)

# ================== 人口统计学特征 ==================
X_demo = df[demo_cols].values
scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(X_demo)

# ================== 融合向量 ==================
X = np.hstack([text_features, X_demo])
y = df[target_col].values

# ================== 划分训练集、验证集和测试集（6:2:2） ==================
# 首先将数据划分为训练集+验证集（80%）和测试集（20%）
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# 然后将训练集+验证集划分为训练集（60%）和验证集（20%）
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=seed, stratify=y_train_val
)

# ================== PyTorch 数据集 ==================
class FusionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

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
    def __init__(self, input_dim, hidden_dims=[512, 256], dropout=0.3):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for hdim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hdim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hdim
        layers.append(nn.Linear(prev_dim, 1))  # 输出logit
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

input_dim = X.shape[1]
model_nn = FusionNN(input_dim).to(device)

# ================== 损失函数与优化器 ==================
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_nn.parameters(), lr=1e-3)

# ================== 训练（带验证集监控和早停） ==================
num_epochs = 100  # 增加最大轮数，让早停机制决定最佳
best_val_auc = 0
best_model_state = None
patience = 10  # 早停耐心值
train_losses = []
val_aucs = []

print("\n===== 开始训练（验证集监控） =====")
for epoch in range(num_epochs):
    # 训练阶段
    model_nn.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model_nn(X_batch)
        loss = criterion(logits, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * X_batch.size(0)

    avg_loss = total_loss / len(train_dataset)
    train_losses.append(avg_loss)

    # 验证阶段
    model_nn.eval()
    val_preds, val_probs, val_labels = [], [], []
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            logits = model_nn(X_batch)
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            val_preds.append(preds.cpu())
            val_probs.append(probs.cpu())
            val_labels.append(y_batch.cpu())

    val_preds = torch.cat(val_preds).numpy()
    val_probs = torch.cat(val_probs).numpy()
    val_labels = torch.cat(val_labels).numpy()

    val_auc = roc_auc_score(val_labels, val_probs)
    val_aucs.append(val_auc)

    print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Val AUC: {val_auc:.4f}")

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
best_epoch = epoch + 1 - patience_counter
print(f"\n最佳验证集AUC: {best_val_auc:.4f} (Epoch {best_epoch})")

# ================== 最终测试（仅在测试集上评估一次） ==================
print(f"\n===== Final Test on Test Set (Best Epoch: {best_epoch}) =====")
model_nn.eval()
test_preds, test_probs, test_labels = [], [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        logits = model_nn(X_batch)
        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        test_preds.append(preds.cpu())
        test_probs.append(probs.cpu())
        test_labels.append(y_batch.cpu())

test_preds = torch.cat(test_preds).numpy()
test_probs = torch.cat(test_probs).numpy()
test_labels = torch.cat(test_labels).numpy()

test_acc = accuracy_score(test_labels, test_preds)
test_auc = roc_auc_score(test_labels, test_probs)
test_auprc = average_precision_score(test_labels, test_probs)
print("===== Test Report =====")
print(classification_report(test_labels, test_preds, digits=4))
print("Accuracy:", test_acc)
print("AUC:", test_auc)
print("AUPRC:", test_auprc)
