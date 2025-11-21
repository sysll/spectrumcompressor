import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ================== 1️⃣ 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 18
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================== 2️⃣ 读取数据 ==================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 文本列
numeric_cols = df.columns[:-3].drop("survival_status")  # 数值列
target_col = 'survival_status'

demo_cols = numeric_cols[:10]
serum_cols = numeric_cols[10:]

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

# ================== 6️⃣ 划分训练/测试 ==================
y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)
train_idx, test_idx = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y.numpy())

train_dataset = MultiModalDataset(
    torch.tensor(X_serum[train_idx], dtype=torch.float32),
    torch.tensor(X_demo[train_idx], dtype=torch.float32),
    input_ids[train_idx],
    attention_mask[train_idx],
    y[train_idx]
)
test_dataset = MultiModalDataset(
    torch.tensor(X_serum[test_idx], dtype=torch.float32),
    torch.tensor(X_demo[test_idx], dtype=torch.float32),
    input_ids[test_idx],
    attention_mask[test_idx],
    y[test_idx]
)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
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
criterion = nn.CrossEntropyLoss()

# ================== 10️⃣ 训练 & 测试 ==================
num_epochs = 10
best_acc = 0.0
best_preds, best_labels, best_probs = None, None, None

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

    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_dataset):.4f}")

    # ================== 测试 ==================
    text_encoder.eval()
    serum_encoder.eval()
    demo_encoder.eval()
    fusion_model.eval()
    all_preds, all_labels, all_probs = [], [], []

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

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu()[:, 1])

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()
    acc = accuracy_score(all_labels, all_preds)
    print("===== Test Report =====")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Accuracy:", acc)

    if acc > best_acc:
        best_acc = acc
        best_preds = all_preds
        best_labels = all_labels
        best_probs = all_probs
        np.save("best_epoch_preds.npy", best_preds)
        np.save("best_epoch_labels.npy", best_labels)
        np.save("best_epoch_probs.npy", best_probs)
        print(f"-> Best epoch updated: {epoch+1}, saved predictions, labels and probabilities.")
