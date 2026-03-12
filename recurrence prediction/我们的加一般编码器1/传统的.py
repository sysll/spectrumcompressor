import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, average_precision_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1️⃣ 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 12
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================== 2️⃣ 读取数据 ==================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 文本列
numeric_cols = df.columns[:-3].drop("recurrence")  # 数值列
target_col = 'recurrence'

demo_cols = numeric_cols[:5]
serum_cols = numeric_cols[5:]

# ================== 3️⃣ 标准化 ==================
scaler_serum = StandardScaler()
X_serum = scaler_serum.fit_transform(df[serum_cols].values)

scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(df[demo_cols].values)

# ================== 4️⃣ 文本 Tokenizer ==================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = text_inputs['input_ids'].to(device)
attention_mask = text_inputs['attention_mask'].to(device)

# ================== 5️⃣ Dataset ==================
class MultiModalDataset(Dataset):
    def __init__(self, serum_feats, demo_feats, input_ids, attention_mask, labels):
        self.serum_feats = serum_feats
        self.demo_feats = demo_feats
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            self.serum_feats[idx],
            self.demo_feats[idx],
            self.input_ids[idx],
            self.attention_mask[idx],
            self.labels[idx]
        )

# ================== 6️⃣ 划分训练/验证/测试（6:2:2） ==================
y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

# 首先将数据划分为训练集+验证集（80%）和测试集（20%）
train_val_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y.numpy()
)

# 然后将训练集+验证集划分为训练集（60%）和验证集（20%）
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.25, random_state=seed, stratify=y[train_val_idx].numpy()
)

train_dataset = MultiModalDataset(
    torch.tensor(X_serum[train_idx], dtype=torch.float32),
    torch.tensor(X_demo[train_idx], dtype=torch.float32),
    input_ids[train_idx],
    attention_mask[train_idx],
    y[train_idx]
)

val_dataset = MultiModalDataset(
    torch.tensor(X_serum[val_idx], dtype=torch.float32),
    torch.tensor(X_demo[val_idx], dtype=torch.float32),
    input_ids[val_idx],
    attention_mask[val_idx],
    y[val_idx]
)

test_dataset = MultiModalDataset(
    torch.tensor(X_serum[test_idx], dtype=torch.float32),
    torch.tensor(X_demo[test_idx], dtype=torch.float32),
    input_ids[test_idx],
    attention_mask[test_idx],
    y[test_idx]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ================== 7️⃣ 编码器 ==================
# 文本编码器：BERT + 平均池化
class TextEncoder(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.model = AutoModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # [batch, seq_len, hidden_dim]
        pooled = last_hidden.mean(dim=1)        # 平均池化 [batch, hidden_dim]
        return pooled

# 数值特征编码器：MLP
class NumericEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, out_dim=384):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

# ================== 8️⃣ 多模态融合 Transformer ==================
class FusionTransformer(nn.Module):
    def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(dim, class_num)

    def forward(self, x_list):
        # 将每个模态 embedding 扩展为序列长度 1
        seqs = [x.unsqueeze(1) for x in x_list]  # [batch, 1, dim]
        all_seq = torch.cat(seqs, dim=1)         # [batch, num_modalities, dim]
        fused = self.transformer(all_seq)        # [batch, num_modalities, dim]
        pooled = fused.mean(dim=1)               # 平均池化
        logits = self.classifier(pooled)
        return logits

# ================== 9️⃣ 模型实例化 ==================
out_dim = 384
text_encoder = TextEncoder().to(device)
serum_encoder = NumericEncoder(input_dim=X_serum.shape[1], out_dim=out_dim).to(device)
demo_encoder = NumericEncoder(input_dim=X_demo.shape[1], out_dim=out_dim).to(device)
fusion_model = FusionTransformer(dim=out_dim, class_num=2).to(device)

params = list(text_encoder.parameters()) + list(serum_encoder.parameters()) + \
         list(demo_encoder.parameters()) + list(fusion_model.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ================== 10️⃣ 训练 & 验证 ==================
num_epochs = 100  # 增加最大轮数，让早停机制决定最佳
best_val_acc = 0.0
best_epoch = 0
best_model_states = None
patience = 10  # 早停耐心值
patience_counter = 0

print("\n===== 开始训练（验证集监控） =====")
for epoch in range(num_epochs):
    text_encoder.train()
    serum_encoder.train()
    demo_encoder.train()
    fusion_model.train()
    total_loss = 0

    for batch in train_loader:
        serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
        serum_feats = serum_feats.to(device)
        demo_feats = demo_feats.to(device)
        labels = labels.squeeze(1).long().to(device)

        text_emb = text_encoder(input_ids_batch, attn_mask_batch)
        serum_emb = serum_encoder(serum_feats)
        demo_emb = demo_encoder(demo_feats)

        cat = [text_emb, serum_emb, demo_emb]
        outputs = fusion_model(cat)

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)

    avg_loss = total_loss / len(train_dataset)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    # ================== 验证 ==================
    text_encoder.eval()
    serum_encoder.eval()
    demo_encoder.eval()
    fusion_model.eval()
    val_preds, val_labels, val_probs = [], [], []

    with torch.no_grad():
        for batch in val_loader:
            serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
            serum_feats = serum_feats.to(device)
            demo_feats = demo_feats.to(device)
            labels = labels.squeeze(1).long().to(device)

            text_emb = text_encoder(input_ids_batch, attn_mask_batch)
            serum_emb = serum_encoder(serum_feats)
            demo_emb = demo_encoder(demo_feats)

            cat = [text_emb, serum_emb, demo_emb]
            outputs = fusion_model(cat)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            val_preds.append(preds.cpu())
            val_labels.append(labels.cpu())
            val_probs.append(probs.cpu()[:, 1])

    val_preds = torch.cat(val_preds).numpy()
    val_labels = torch.cat(val_labels).numpy()
    val_probs = torch.cat(val_probs).numpy()

    val_acc = accuracy_score(val_labels, val_preds)
    val_auprc = average_precision_score(val_labels, val_probs)
    print("===== Validation Report =====")
    print(classification_report(val_labels, val_preds, digits=4))
    print("Validation Accuracy:", val_acc)
    print("Validation AUPRC:", val_auprc)

    # ================== 保存最佳模型（基于验证集） ==================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_epoch = epoch + 1
        # 保存所有模型的状态
        best_model_states = {
            'text_encoder': text_encoder.state_dict(),
            'serum_encoder': serum_encoder.state_dict(),
            'demo_encoder': demo_encoder.state_dict(),
            'fusion_model': fusion_model.state_dict()
        }
        patience_counter = 0
        print(f"-> 新的最佳模型！验证集Accuracy: {best_val_acc:.4f}")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\n早停：验证集Accuracy连续{patience}轮未提升，停止训练")
            break

# 加载最佳模型
if best_model_states:
    text_encoder.load_state_dict(best_model_states['text_encoder'])
    serum_encoder.load_state_dict(best_model_states['serum_encoder'])
    demo_encoder.load_state_dict(best_model_states['demo_encoder'])
    fusion_model.load_state_dict(best_model_states['fusion_model'])
    print(f"\n加载最佳模型（Epoch {best_epoch}）")

# ================== 11️⃣ 最终测试（仅在测试集上评估一次） ==================
print(f"\n===== Final Test on Test Set (Best Epoch: {best_epoch}) =====")
text_encoder.eval()
serum_encoder.eval()
demo_encoder.eval()
fusion_model.eval()
test_preds, test_labels, test_probs = [], [], []

with torch.no_grad():
    for batch in test_loader:
        serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
        serum_feats = serum_feats.to(device)
        demo_feats = demo_feats.to(device)
        labels = labels.squeeze(1).long().to(device)

        text_emb = text_encoder(input_ids_batch, attn_mask_batch)
        serum_emb = serum_encoder(serum_feats)
        demo_emb = demo_encoder(demo_feats)

        cat = [text_emb, serum_emb, demo_emb]
        outputs = fusion_model(cat)
        probs = torch.softmax(outputs, dim=1)
        preds = torch.argmax(probs, dim=1)

        test_preds.append(preds.cpu())
        test_labels.append(labels.cpu())
        test_probs.append(probs.cpu()[:, 1])

test_preds = torch.cat(test_preds).numpy()
test_labels = torch.cat(test_labels).numpy()
test_probs = torch.cat(test_probs).numpy()

test_acc = accuracy_score(test_labels, test_preds)
test_auprc = average_precision_score(test_labels, test_probs)
print("===== Test Report =====")
print(classification_report(test_labels, test_preds, digits=4))
print("Test Accuracy:", test_acc)
print("Test AUPRC:", test_auprc)

# 保存最终测试结果
np.save("best_epoch_preds.npy", test_preds)
np.save("best_epoch_labels.npy", test_labels)
np.save("best_epoch_probs.npy", test_probs)
print("-> Saved final test predictions, labels and probabilities.")






