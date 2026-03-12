import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 59
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
text_cols = df.columns[-3:]  # 文本列
numeric_cols = df.columns[:-3].drop("survival_status")  # 数值列
target_col = 'survival_status'

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
X_demo = df[numeric_cols].values
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
model = FusionNN(input_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
num_epochs = 100
batch_size = 32

# ================== 数据转为 Tensor ==================
X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)
X_val_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
y_val_tensor = torch.tensor(y_val, dtype=torch.long).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
val_dataset = torch.utils.data.TensorDataset(X_val_tensor, y_val_tensor)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# ================== 训练 ==================
best_val_auc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataset):.4f}")

    # ================== 验证 ==================
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val_tensor)
        val_probs = torch.softmax(val_outputs, dim=1)[:, 1].cpu().numpy()
        val_preds = torch.argmax(val_outputs, dim=1).cpu().numpy()

    val_acc = accuracy_score(y_val, val_preds)
    val_auc = roc_auc_score(y_val, val_probs)
    val_auprc = average_precision_score(y_val, val_probs)
    print(f"Validation Accuracy: {val_acc:.4f}, Validation AUC: {val_auc:.4f}, Validation AUPRC: {val_auprc:.4f}")

    # ================== 保存最佳 epoch（基于验证集） ==================
    if val_auc > best_val_auc:
        best_val_auc = val_auc
        best_epoch = epoch + 1
        print(f"-> Best validation epoch updated: {best_epoch}, Validation AUC: {best_val_auc:.4f}")

# ================== 最终测试（仅在测试集上评估一次） ==================
print(f"\n===== Final Test on Test Set (Best Epoch: {best_epoch}) =====")
model.eval()
with torch.no_grad():
    outputs = model(X_test_tensor)
    probs = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
    preds = torch.argmax(outputs, dim=1).cpu().numpy()

print("===== Test Report =====")
print(classification_report(y_test, preds, digits=4))
print("Accuracy:", accuracy_score(y_test, preds))
print("AUC:", roc_auc_score(y_test, probs))
print("AUPRC:", average_precision_score(y_test, probs))
