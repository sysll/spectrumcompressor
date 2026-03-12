import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, average_precision_score
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import matplotlib.pyplot as plt
import os

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
# 6️⃣ 划分训练集、验证集和测试集（6:2:2）
# ==================================================
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

# ==================================================
# 7️⃣ 血清编码器
# ==================================================
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

# ==================================================
# 8️⃣ 人口统计学编码器
# ==================================================
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

# ==================================================
# 9️⃣ 不同的文本编码器变体
# ==================================================

# 1. 基础模型：多层拼接 + FFT压缩 + 门控
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

# 2. 变体1：仅使用最后一层
class SpectralCompressor_LastLayerOnly(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
        super().__init__()
        self.seq2 = seq2
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_dim = self.model.config.hidden_size
        self.freq_gate = nn.Parameter(torch.randn(seq2, self.hidden_dim))

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        last = outputs.hidden_states[-1]  # 仅使用最后一层

        # 频谱压缩
        freq = fft.rfft(last, dim=1)
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

# 3. 变体2：FFT压缩无门控
class SpectralCompressor_NoGating(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
        super().__init__()
        self.seq2 = seq2
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_dim = self.model.config.hidden_size
        self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs = outputs.hidden_states
        first, middle, last = hs[1], hs[len(hs)//2], hs[-1]
        concat = torch.cat([first, middle, last], dim=-1)
        concat = self.linear_proj(concat)

        # 频谱压缩（无门控）
        freq = fft.rfft(concat, dim=1)
        seq_len = freq.size(1)
        idx = torch.linspace(0, seq_len - 1, steps=self.seq2, device=freq.device)
        idx_floor = idx.long()
        idx_ceil = torch.clamp(idx_floor + 1, max=seq_len - 1)
        w = (idx - idx_floor).unsqueeze(0).unsqueeze(-1)
        freq_down = (1 - w) * freq[:, idx_floor, :] + w * freq[:, idx_ceil, :]
        compressed = fft.irfft(freq_down, n=self.seq2, dim=1)
        return compressed.real

# 4. 变体3：门控无插值
class SpectralCompressor_NoInterpolation(nn.Module):
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

        # 频谱压缩（无插值）
        freq = fft.rfft(concat, dim=1)
        seq_len = freq.size(1)
        # 直接取前seq2个频率分量，无插值
        freq_down = freq[:, :self.seq2, :]
        gate = torch.sigmoid(self.freq_gate).unsqueeze(0)
        freq_weighted = freq_down * gate
        compressed = fft.irfft(freq_weighted, n=self.seq2, dim=1)
        return compressed.real

# 5. 变体4：仅使用低频成分
class SpectralCompressor_LowFreqOnly(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
        super().__init__()
        self.seq2 = seq2
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
        self.hidden_dim = self.model.config.hidden_size
        self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hs = outputs.hidden_states
        first, middle, last = hs[1], hs[len(hs)//2], hs[-1]
        concat = torch.cat([first, middle, last], dim=-1)
        concat = self.linear_proj(concat)

        # 频谱压缩，仅使用低频成分，无门控和插值
        freq = fft.rfft(concat, dim=1)
        # 直接取前seq2个频率分量，无插值和门控
        freq_down = freq[:, :self.seq2, :]
        compressed = fft.irfft(freq_down, n=self.seq2, dim=1)
        return compressed.real

# 6. 变体5：不同的S2值
class SpectralCompressor_DifferentS2(nn.Module):
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=10):
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

# ==================================================
# 10️⃣ 不同的池化方法
# ==================================================

# 基础模型：加权Top-K池化
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

# 变体1：平均池化
class FusionTransformer_MeanPooling(nn.Module):
    def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
        super().__init__()
        self.dim = dim

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

    def forward(self, x_list):
        # 拼接三模态序列
        all_seq = torch.cat(x_list, dim=1)  # [bs, seq_total, dim]

        # Transformer编码
        fused = self.transformer(all_seq)   # [bs, seq_total, dim]

        # 平均池化
        fused = fused.mean(dim=1)  # [bs, dim]

        # 分类器
        logits = self.classifier(fused)           # [bs, class_num]

        return logits

# 变体2：最大池化
class FusionTransformer_MaxPooling(nn.Module):
    def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
        super().__init__()
        self.dim = dim

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

    def forward(self, x_list):
        # 拼接三模态序列
        all_seq = torch.cat(x_list, dim=1)  # [bs, seq_total, dim]

        # Transformer编码
        fused = self.transformer(all_seq)   # [bs, seq_total, dim]

        # 最大池化
        fused, _ = fused.max(dim=1)  # [bs, dim]

        # 分类器
        logits = self.classifier(fused)           # [bs, class_num]

        return logits

# 变体3：CLS池化
class FusionTransformer_CLSPooling(nn.Module):
    def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
        super().__init__()
        self.dim = dim

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

    def forward(self, x_list):
        # 拼接三模态序列
        all_seq = torch.cat(x_list, dim=1)  # [bs, seq_total, dim]

        # Transformer编码
        fused = self.transformer(all_seq)   # [bs, seq_total, dim]

        # CLS池化（取第一个token）
        fused = fused[:, 0, :]  # [bs, dim]

        # 分类器
        logits = self.classifier(fused)           # [bs, class_num]

        return logits

# 变体4：Top-K池化无学习权重
class FusionTransformer_TopKNoWeights(nn.Module):
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

    def topk_pooling(self, x):
        """
        Top-K池化：对Top-K值平均
        x: [bs, seq, dim]
        """
        topk_vals, _ = torch.topk(x, self.topk, dim=1)  # [bs, topk, dim]
        pooled = topk_vals.mean(dim=1)  # [bs, dim]
        return pooled

    def forward(self, x_list):
        # 拼接三模态序列
        all_seq = torch.cat(x_list, dim=1)  # [bs, seq_total, dim]

        # Transformer编码
        fused = self.transformer(all_seq)   # [bs, seq_total, dim]

        # Top-K池化（无学习权重）
        fused = self.topk_pooling(fused)  # [bs, dim]

        # 分类器
        logits = self.classifier(fused)           # [bs, class_num]

        return logits

# ==================================================
# 11️⃣ 训练和评估函数
# ==================================================
def train_and_evaluate(text_encoder, fusion_model, name):
    print(f"\n===== 训练 {name} =====")
    
    # 实例化模型
    out_dim = 384
    serum_encoder = SerumMLPEncoder(hidden_dim=128, out_dim=out_dim).to(device)
    demo_encoder = DemographicEncoder(hidden_dim=64, out_dim=out_dim).to(device)
    
    params = list(text_encoder.parameters()) + list(serum_encoder.parameters()) + \
             list(demo_encoder.parameters()) + list(fusion_model.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # 训练循环
    num_epochs = 10
    best_val_acc = 0.0
    best_epoch = 0
    
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

            # 编码
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

        # 验证
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
        print(f"Validation Accuracy: {val_acc:.4f}, Validation AUPRC: {val_auprc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
    
    # 最终测试
    print(f"\n===== 测试 {name} (Best Epoch: {best_epoch}) =====")
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
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test AUPRC: {test_auprc:.4f}")
    
    return test_acc, test_auprc

# ==================================================
# 12️⃣ 运行所有消融实验
# ==================================================
if __name__ == "__main__":
    results = {}
    
    # # 1. 基础模型
    # print("\n" + "="*80)
    # print("1. 基础模型：多层拼接 + FFT压缩 + 门控 + 加权Top-K池化")
    # print("="*80)
    # text_encoder_base = SpectralCompressor(seq2=20).to(device)
    # fusion_model_base = FusionTransformer3Modal(dim=384, class_num=2).to(device)
    # acc_base, auprc_base = train_and_evaluate(text_encoder_base, fusion_model_base, "基础模型")
    # results["基础模型"] = {"accuracy": acc_base, "auprc": auprc_base}
    #
    # # 2. 变体1：仅使用最后一层
    # print("\n" + "="*80)
    # print("2. 变体1：仅使用最后一层")
    # print("="*80)
    # text_encoder_last = SpectralCompressor_LastLayerOnly(seq2=20).to(device)
    # fusion_model_last = FusionTransformer3Modal(dim=384, class_num=2).to(device)
    # acc_last, auprc_last = train_and_evaluate(text_encoder_last, fusion_model_last, "仅使用最后一层")
    # results["仅使用最后一层"] = {"accuracy": acc_last, "auprc": auprc_last}
    
    # 3. 变体2：FFT压缩无门控
    print("\n" + "="*80)
    print("3. 变体2：FFT压缩无门控")
    print("="*80)
    text_encoder_nogate = SpectralCompressor_NoGating(seq2=20).to(device)
    fusion_model_nogate = FusionTransformer3Modal(dim=384, class_num=2).to(device)
    acc_nogate, auprc_nogate = train_and_evaluate(text_encoder_nogate, fusion_model_nogate, "FFT压缩无门控")
    results["FFT压缩无门控"] = {"accuracy": acc_nogate, "auprc": auprc_nogate}
    
    # 4. 变体3：门控无插值
    print("\n" + "="*80)
    print("4. 变体3：门控无插值")
    print("="*80)
    text_encoder_nointerp = SpectralCompressor_NoInterpolation(seq2=20).to(device)
    fusion_model_nointerp = FusionTransformer3Modal(dim=384, class_num=2).to(device)
    acc_nointerp, auprc_nointerp = train_and_evaluate(text_encoder_nointerp, fusion_model_nointerp, "门控无插值")
    results["门控无插值"] = {"accuracy": acc_nointerp, "auprc": auprc_nointerp}
    
    # 5. 变体4：仅使用低频成分
    print("\n" + "="*80)
    print("5. 变体4：仅使用低频成分")
    print("="*80)
    text_encoder_lowfreq = SpectralCompressor_LowFreqOnly(seq2=20).to(device)
    fusion_model_lowfreq = FusionTransformer3Modal(dim=384, class_num=2).to(device)
    acc_lowfreq, auprc_lowfreq = train_and_evaluate(text_encoder_lowfreq, fusion_model_lowfreq, "仅使用低频成分")
    results["仅使用低频成分"] = {"accuracy": acc_lowfreq, "auprc": auprc_lowfreq}
    
    # 6. 变体5：不同的S2值
    print("\n" + "="*80)
    print("5. 变体4：不同的S2值 (S2=10)")
    print("="*80)
    text_encoder_s2 = SpectralCompressor_DifferentS2(seq2=10).to(device)
    fusion_model_s2 = FusionTransformer3Modal(dim=384, class_num=2).to(device)
    acc_s2, auprc_s2 = train_and_evaluate(text_encoder_s2, fusion_model_s2, "S2=10")
    results["S2=10"] = {"accuracy": acc_s2, "auprc": auprc_s2}
    
    # 6. 变体5：平均池化
    print("\n" + "="*80)
    print("6. 变体5：平均池化")
    print("="*80)
    text_encoder_pool = SpectralCompressor(seq2=20).to(device)
    fusion_model_mean = FusionTransformer_MeanPooling(dim=384, class_num=2).to(device)
    acc_mean, auprc_mean = train_and_evaluate(text_encoder_pool, fusion_model_mean, "平均池化")
    results["平均池化"] = {"accuracy": acc_mean, "auprc": auprc_mean}
    
    # 7. 变体6：最大池化
    print("\n" + "="*80)
    print("7. 变体6：最大池化")
    print("="*80)
    text_encoder_pool = SpectralCompressor(seq2=20).to(device)
    fusion_model_max = FusionTransformer_MaxPooling(dim=384, class_num=2).to(device)
    acc_max, auprc_max = train_and_evaluate(text_encoder_pool, fusion_model_max, "最大池化")
    results["最大池化"] = {"accuracy": acc_max, "auprc": auprc_max}
    
    # 8. 变体7：CLS池化
    print("\n" + "="*80)
    print("8. 变体7：CLS池化")
    print("="*80)
    text_encoder_pool = SpectralCompressor(seq2=20).to(device)
    fusion_model_cls = FusionTransformer_CLSPooling(dim=384, class_num=2).to(device)
    acc_cls, auprc_cls = train_and_evaluate(text_encoder_pool, fusion_model_cls, "CLS池化")
    results["CLS池化"] = {"accuracy": acc_cls, "auprc": auprc_cls}
    
    # 9. 变体8：Top-K池化无学习权重
    print("\n" + "="*80)
    print("9. 变体8：Top-K池化无学习权重")
    print("="*80)
    text_encoder_pool = SpectralCompressor(seq2=20).to(device)
    fusion_model_topk_nw = FusionTransformer_TopKNoWeights(dim=384, class_num=2).to(device)
    acc_topk_nw, auprc_topk_nw = train_and_evaluate(text_encoder_pool, fusion_model_topk_nw, "Top-K池化无学习权重")
    results["Top-K池化无学习权重"] = {"accuracy": acc_topk_nw, "auprc": auprc_topk_nw}
    
    # 保存结果
    print("\n" + "="*80)
    print("消融实验结果汇总")
    print("="*80)
    for model_name, metrics in results.items():
        print(f"{model_name}: Accuracy = {metrics['accuracy']:.4f}, AUPRC = {metrics['auprc']:.4f}")
    
    # 保存到文件
    import json
    with open('ablation_results.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print("\n消融实验结果已保存到 ablation_results.json")
