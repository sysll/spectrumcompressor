
import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt

# 设置字体为常用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题
# ==================================================
# 1️⃣ 基础配置
# ==================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 22  #18
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ==================================================
# 2️⃣ 读取数据
# ==================================================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 最后三列文本
numeric_cols = df.columns[:-3].drop("recurrence")  # 血清+人口统计学
target_col = 'recurrence'

# 假设人口统计学特征在前10列
demo_cols = numeric_cols[:10]
serum_cols = numeric_cols[10:]

# ==================================================
# 3️⃣ 数值特征标准化
# ==================================================
scaler_serum = StandardScaler()
X_serum = scaler_serum.fit_transform(df[serum_cols].values)

scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(df[demo_cols].values)

# ==================================================
# 4️⃣ 文本 Tokenizer
# ==================================================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
text_inputs = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt")
input_ids = text_inputs['input_ids'].to(device)
attention_mask = text_inputs['attention_mask'].to(device)

# ==================================================
# 5️⃣ 自定义 Dataset
# ==================================================
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

# ==================================================
# 6️⃣ 划分训练集和测试集
# ==================================================
y = torch.tensor(df[target_col].values, dtype=torch.float32).unsqueeze(1)

train_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y.numpy()
)

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

# ==================================================
# 7️⃣ 模型定义
# ==================================================

# ---- 文本编码器 ----
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
        return compressed.real

# ---- 血清编码器 ----
class SerumMLPEncoder(nn.Module):
    def __init__(self, seq_len_input=33, seq_len_output=37, hidden_dim=128, out_dim=384):
        super().__init__()
        self.seq_len_output = seq_len_output
        # 将输入特征映射到隐藏维度
        self.mlp = nn.Sequential(
            nn.Linear(seq_len_input, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim)
        )
        # 可学习序列向量，用于扩展到 seq_len_output
        self.seq_param = nn.Parameter(torch.randn(seq_len_output, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        # 将输入映射到 out_dim
        feat = self.mlp(x)                 # [batch, out_dim]
        feat = feat.unsqueeze(1)           # [batch, 1, out_dim]
        # 通过可学习参数生成整个序列
        seq_feat = feat + self.seq_param.unsqueeze(0)  # [batch, seq_len_output, out_dim]
        return seq_feat



# ---- 人口统计学编码器 ----
class DemographicEncoder(nn.Module):
    def __init__(self, seq_len_input=10, seq_len_output=10, hidden_dim=64, out_dim=384):
        super().__init__()
        self.seq_len_output = seq_len_output
        # 将输入特征映射到隐藏维度
        self.mlp = nn.Sequential(
            nn.Linear(seq_len_input, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim)
        )
        # 可学习序列向量，用于扩展到 seq_len_output
        self.seq_param = nn.Parameter(torch.randn(seq_len_output, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        # 将输入映射到 out_dim
        feat = self.mlp(x)                        # [batch, out_dim]
        feat = feat.unsqueeze(1)                  # [batch, 1, out_dim]
        # 通过可学习参数生成整个序列
        seq_feat = feat + self.seq_param.unsqueeze(0)  # [batch, seq_len_output, out_dim]
        return seq_feat





import torch
import torch.nn as nn

class FusionTransformer3Modal(nn.Module):
    def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2, topk=3):
        super().__init__()
        self.dim = dim
        self.topk = topk

        # 标准 TransformerEncoderLayer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # 分类器
        self.classifier = nn.Linear(dim, class_num)

        # 可学习的权重向量，用于Top-K加权
        self.topk_weights = nn.Parameter(torch.ones(topk) / topk)

    def weighted_topk_pooling(self, x):
        """
        改进Top-K池化：对Top-K值加权平均
        x: [bs, seq, dim]
        """
        topk_vals, _ = torch.topk(x, self.topk, dim=1)  # [bs, topk, dim]
        # 归一化权重
        weights = torch.softmax(self.topk_weights, dim=0)  # [topk]
        pooled = (topk_vals * weights.view(1, self.topk, 1)).sum(dim=1)  # [bs, dim]
        return pooled

    def forward(self, x_list):
        # 拼接三模态序列
        all_seq = torch.cat(x_list, dim=1)  # [bs, seq_total, dim]

        # Transformer编码
        fused = self.transformer(all_seq)   # [bs, seq_total, dim]

        # 加权Top-K池化
        fused = self.weighted_topk_pooling(fused)  # [bs, dim]

        # 分类器
        logits = self.classifier(fused)           # [bs, class_num]

        return logits






# ==================================================
# 8️⃣ 实例化模型
# ==================================================
out_dim = 384
spectral_model = SpectralCompressor(seq2=20).to(device)
serum_encoder = SerumMLPEncoder(hidden_dim=128, out_dim=out_dim).to(device)
demo_encoder = DemographicEncoder(hidden_dim=64, out_dim=out_dim).to(device)
fusion_model = FusionTransformer3Modal(dim=out_dim, class_num=2).to(device)

params = list(spectral_model.parameters()) + list(serum_encoder.parameters()) + \
         list(demo_encoder.parameters()) + list(fusion_model.parameters())
optimizer = torch.optim.Adam(params, lr=1e-4)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# ==================================================
# 9️⃣ 训练 & 测试循环
# ==================================================
num_epochs = 10
best_acc = 0.0
best_preds, best_labels, best_probs = None, None, None  # 新增 best_probs 保存概率

for epoch in range(num_epochs):
    spectral_model.train()
    serum_encoder.train()
    demo_encoder.train()
    fusion_model.train()
    total_loss = 0

    for batch in train_loader:
        serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
        serum_feats = serum_feats.to(device)
        demo_feats = demo_feats.to(device)
        labels = labels.squeeze(1).long().to(device)

        # 编码
        text_emb = spectral_model(input_ids_batch, attn_mask_batch)
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
    spectral_model.eval()
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

            text_emb = spectral_model(input_ids_batch, attn_mask_batch)
            serum_emb = serum_encoder(serum_feats)
            demo_emb = demo_encoder(demo_feats)

            cat = [text_emb, serum_emb, demo_emb]
            outputs = fusion_model(cat)  # [batch, num_classes]
            probs = torch.softmax(outputs, dim=1)  # 概率
            preds = torch.argmax(probs, dim=1)     # 类别索引

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())
            all_probs.append(probs.cpu()[:, 1])   # 保存正类概率用于 ROC/PR

    all_preds = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    all_probs = torch.cat(all_probs).numpy()

    acc = accuracy_score(all_labels, all_preds)
    print("===== Test Report =====")
    print(classification_report(all_labels, all_preds, digits=4))
    print("Accuracy:", acc)

    # ================== 保存最佳 epoch ==================
    if acc > best_acc:
        best_acc = acc
        best_preds = all_preds
        best_labels = all_labels
        best_probs = all_probs
        np.save("best_epoch_preds.npy", best_preds)
        np.save("best_epoch_labels.npy", best_labels)
        np.save("best_epoch_probs.npy", best_probs)
        print(f"-> Best epoch updated: {epoch+1}, saved predictions, labels and probabilities.")


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# 1️⃣ 获取测试集融合特征
fusion_model.eval()
spectral_model.eval()
serum_encoder.eval()
demo_encoder.eval()

all_features, all_labels = [], []

with torch.no_grad():
    for batch in test_loader:
        serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
        serum_feats = serum_feats.to(device)
        demo_feats = demo_feats.to(device)

        text_emb = spectral_model(input_ids_batch, attn_mask_batch)
        serum_emb = serum_encoder(serum_feats)
        demo_emb = demo_encoder(demo_feats)

        cat = [text_emb, serum_emb, demo_emb]
        fused_feat = fusion_model.transformer(torch.cat(cat, dim=1))  # [bs, seq_total, dim]
        fused_feat = fusion_model.weighted_topk_pooling(fused_feat)    # [bs, dim]

        all_features.append(fused_feat.cpu())
        all_labels.append(labels.squeeze(1))

all_features = torch.cat(all_features).numpy()
all_labels = torch.cat(all_labels).numpy()

# 2️⃣ t-SNE 降维
tsne = TSNE(n_components=2, random_state=seed)
features_2d = tsne.fit_transform(all_features)

# 3️⃣ 绘图
plt.figure(figsize=(8, 6))
for label in np.unique(all_labels):
    idx = all_labels == label
    plt.scatter(features_2d[idx, 0], features_2d[idx, 1], label=f"Class {label}", alpha=0.7)

plt.title("t-SNE Visualization of Fused Features")
plt.xlabel("t-SNE 1")
plt.ylabel("t-SNE 2")
plt.legend()
plt.show()

