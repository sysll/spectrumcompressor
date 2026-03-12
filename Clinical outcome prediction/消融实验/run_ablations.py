import torch
import torch.nn as nn
import torch.fft as fft
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, average_precision_score, \
    confusion_matrix
from transformers import AutoTokenizer, AutoModel
import matplotlib.pyplot as plt
import seaborn as sns
from base_model import SpectralCompressor, SerumMLPEncoder, DemographicEncoder, FusionTransformer3Modal

# 设置字体为常用中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# ==================================================
# 1️⃣ 基础配置
# ==================================================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 18
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ==================================================
# 2️⃣ 读取数据
# ==================================================
df = pd.read_excel("../我们的加一般编码器1/patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 最后三列文本
numeric_cols = df.columns[:-3].drop("survival_status")  # 血清+人口统计学
target_col = 'survival_status'

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
# 7️⃣ 消融实验模型变体
# ==================================================

# 变体1: 基础模型（完整模型）
class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "完整模型"
        self.out_dim = 384
        self.spectral_model = SpectralCompressor(seq2=20).to(device)
        self.serum_encoder = SerumMLPEncoder(hidden_dim=128, out_dim=self.out_dim).to(device)
        self.demo_encoder = DemographicEncoder(hidden_dim=64, out_dim=self.out_dim).to(device)
        self.fusion_model = FusionTransformer3Modal(dim=self.out_dim, class_num=2).to(device)

    def forward(self, serum_feats, demo_feats, input_ids, attention_mask):
        text_emb = self.spectral_model(input_ids, attention_mask)
        serum_emb = self.serum_encoder(serum_feats)
        demo_emb = self.demo_encoder(demo_feats)
        cat = [text_emb, serum_emb, demo_emb]
        outputs = self.fusion_model(cat)
        return outputs

    def get_params(self):
        return list(self.spectral_model.parameters()) + list(self.serum_encoder.parameters()) + \
               list(self.demo_encoder.parameters()) + list(self.fusion_model.parameters())


# 变体2: 没有多层聚合（仅使用最后一层）
class NoMultiLayerModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "无多层聚合"

        # 重写SpectralCompressor，仅使用最后一层
        class NoMultiLayerSpectralCompressor(nn.Module):
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
                super().__init__()
                self.seq2 = seq2
                self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
                self.hidden_dim = self.model.config.hidden_size
                self.freq_gate = nn.Parameter(torch.randn(seq2, self.hidden_dim))

            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                # 仅使用最后一层
                last = outputs.last_hidden_state

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

        self.spectral_model = NoMultiLayerSpectralCompressor(seq2=20).to(device)


# 变体3: 没有FFT压缩
class NoFFTModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "无FFT压缩"

        # 重写SpectralCompressor，不使用FFT压缩
        class NoFFTSpectralCompressor(nn.Module):
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
                super().__init__()
                self.seq2 = seq2
                self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
                self.hidden_dim = self.model.config.hidden_size
                self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
                # 直接使用可学习参数作为序列
                self.seq_param = nn.Parameter(torch.randn(seq2, self.hidden_dim))

            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hs = outputs.hidden_states
                first, middle, last = hs[1], hs[len(hs) // 2], hs[-1]
                concat = torch.cat([first, middle, last], dim=-1)
                concat = self.linear_proj(concat)

                # 直接使用可学习参数作为序列
                batch_size = concat.size(0)
                feat = concat.mean(dim=1).unsqueeze(1)
                seq_feat = feat + self.seq_param.unsqueeze(0)
                return seq_feat

        self.spectral_model = NoFFTSpectralCompressor(seq2=20).to(device)


# 变体4: 没有门控机制
class NoGatingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "无门控机制"

        # 重写SpectralCompressor，不使用门控
        class NoGatingSpectralCompressor(nn.Module):
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
                super().__init__()
                self.seq2 = seq2
                self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
                self.hidden_dim = self.model.config.hidden_size
                self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hs = outputs.hidden_states
                first, middle, last = hs[1], hs[len(hs) // 2], hs[-1]
                concat = torch.cat([first, middle, last], dim=-1)
                concat = self.linear_proj(concat)

                # 频谱压缩但不使用门控
                freq = fft.rfft(concat, dim=1)
                seq_len = freq.size(1)
                idx = torch.linspace(0, seq_len - 1, steps=self.seq2, device=freq.device)
                idx_floor = idx.long()
                idx_ceil = torch.clamp(idx_floor + 1, max=seq_len - 1)
                w = (idx - idx_floor).unsqueeze(0).unsqueeze(-1)
                freq_down = (1 - w) * freq[:, idx_floor, :] + w * freq[:, idx_ceil, :]
                # 不使用门控
                compressed = fft.irfft(freq_down, n=self.seq2, dim=1)
                return compressed.real

        self.spectral_model = NoGatingSpectralCompressor(seq2=20).to(device)


# 变体5: 仅使用低频成分
class LowFreqOnlyModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "仅使用低频成分"

        # 重写SpectralCompressor，仅使用低频成分
        class LowFreqSpectralCompressor(nn.Module):
            def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2", seq2=20):
                super().__init__()
                self.seq2 = seq2
                self.model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
                self.hidden_dim = self.model.config.hidden_size
                self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)

            def forward(self, input_ids, attention_mask):
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hs = outputs.hidden_states
                first, middle, last = hs[1], hs[len(hs) // 2], hs[-1]
                concat = torch.cat([first, middle, last], dim=-1)
                concat = self.linear_proj(concat)

                # 频谱压缩，仅使用低频成分，无门控和插值
                freq = fft.rfft(concat, dim=1)
                # 直接取前seq2个频率分量，无插值和门控
                freq_down = freq[:, :self.seq2, :]
                compressed = fft.irfft(freq_down, n=self.seq2, dim=1)
                return compressed.real

        self.spectral_model = LowFreqSpectralCompressor(seq2=20).to(device)


# 变体5: 不同的S2值（S2=10）
class S2_10Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "S2=10"
        self.spectral_model = SpectralCompressor(seq2=10).to(device)


# 变体6: 不同的S2值（S2=30）
class S2_30Model(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "S2=30"
        self.spectral_model = SpectralCompressor(seq2=30).to(device)


# 变体7: 没有加权Top-K池化（使用平均池化）
class NoTopKModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "平均池化"

        # 重写FusionTransformer3Modal，使用平均池化
        class MeanPoolingFusionTransformer(nn.Module):
            def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
                super().__init__()
                self.dim = dim

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                self.classifier = nn.Linear(dim, class_num)

            def forward(self, x_list):
                all_seq = torch.cat(x_list, dim=1)
                fused = self.transformer(all_seq)
                # 使用平均池化
                fused = fused.mean(dim=1)
                logits = self.classifier(fused)
                return logits

        self.fusion_model = MeanPoolingFusionTransformer(dim=self.out_dim, class_num=2).to(device)


# 变体8: 使用最大池化
class MaxPoolingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "最大池化"

        # 重写FusionTransformer3Modal，使用最大池化
        class MaxPoolingFusionTransformer(nn.Module):
            def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
                super().__init__()
                self.dim = dim

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                self.classifier = nn.Linear(dim, class_num)

            def forward(self, x_list):
                all_seq = torch.cat(x_list, dim=1)
                fused = self.transformer(all_seq)
                # 使用最大池化
                fused, _ = fused.max(dim=1)
                logits = self.classifier(fused)
                return logits

        self.fusion_model = MaxPoolingFusionTransformer(dim=self.out_dim, class_num=2).to(device)


# 变体9: 使用CLS池化
class CLSPoolingModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "CLS池化"

        # 重写FusionTransformer3Modal，使用CLS池化
        class CLSPoolingFusionTransformer(nn.Module):
            def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2):
                super().__init__()
                self.dim = dim

                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=dim,
                    nhead=num_heads,
                    dim_feedforward=dim * 4,
                    dropout=dropout,
                    batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

                self.classifier = nn.Linear(dim, class_num)

            def forward(self, x_list):
                all_seq = torch.cat(x_list, dim=1)
                fused = self.transformer(all_seq)
                # 使用CLS池化（取第一个token）
                fused = fused[:, 0, :]
                logits = self.classifier(fused)
                return logits

        self.fusion_model = CLSPoolingFusionTransformer(dim=self.out_dim, class_num=2).to(device)


# 变体10: 门控无插值
class NoInterpolationModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "门控无插值"

        # 重写SpectralCompressor，门控无插值
        class NoInterpolationSpectralCompressor(nn.Module):
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
                first, middle, last = hs[1], hs[len(hs) // 2], hs[-1]
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

        self.spectral_model = NoInterpolationSpectralCompressor(seq2=20).to(device)


# 变体8: 没有血清特征 - 修正版本
class NoSerumModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "无血清特征"
        # 移除血清编码器
        self.serum_encoder = None

    def forward(self, serum_feats, demo_feats, input_ids, attention_mask):
        text_emb = self.spectral_model(input_ids, attention_mask)
        demo_emb = self.demo_encoder(demo_feats)
        # 仅使用文本和人口统计学特征
        cat = [text_emb, demo_emb]
        outputs = self.fusion_model(cat)
        return outputs


# 变体9: 没有人口统计学特征 - 修正版本
class NoDemoModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "无人口统计学特征"
        # 移除人口统计学编码器
        self.demo_encoder = None

    def forward(self, serum_feats, demo_feats, input_ids, attention_mask):
        text_emb = self.spectral_model(input_ids, attention_mask)
        serum_emb = self.serum_encoder(serum_feats)
        # 仅使用文本和血清特征
        cat = [text_emb, serum_emb]
        outputs = self.fusion_model(cat)
        return outputs


# 变体10: 仅使用文本特征
class OnlyTextModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "仅文本特征"
        self.serum_encoder = None
        self.demo_encoder = None

    def forward(self, serum_feats, demo_feats, input_ids, attention_mask):
        text_emb = self.spectral_model(input_ids, attention_mask)
        # 仅使用文本特征
        cat = [text_emb]
        outputs = self.fusion_model(cat)
        return outputs


# 变体11: 仅使用血清特征
class OnlySerumModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "仅血清特征"
        self.spectral_model = None
        self.demo_encoder = None

    def forward(self, serum_feats, demo_feats, input_ids, attention_mask):
        serum_emb = self.serum_encoder(serum_feats)
        # 仅使用血清特征
        cat = [serum_emb]
        outputs = self.fusion_model(cat)
        return outputs


# 变体12: 仅使用人口统计学特征
class OnlyDemoModel(BaseModel):
    def __init__(self):
        super().__init__()
        self.name = "仅人口统计学特征"
        self.spectral_model = None
        self.serum_encoder = None

    def forward(self, serum_feats, demo_feats, input_ids, attention_mask):
        demo_emb = self.demo_encoder(demo_feats)
        # 仅使用人口统计学特征
        cat = [demo_emb]
        outputs = self.fusion_model(cat)
        return outputs


# ==================================================
# 8️⃣ 训练和评估函数
# ==================================================
def train_and_evaluate(model, model_name):
    print(f"\n===== 开始训练: {model_name} =====")

    # 优化器和损失函数
    optimizer = torch.optim.Adam(model.get_params(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 训练参数
    num_epochs = 100
    best_val_auc = 0
    best_model_states = None
    patience = 10
    patience_counter = 0

    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        total_loss = 0

        for batch in train_loader:
            serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
            serum_feats = serum_feats.to(device)
            demo_feats = demo_feats.to(device)
            input_ids_batch = input_ids_batch.to(device)
            attn_mask_batch = attn_mask_batch.to(device)
            labels = labels.squeeze(1).long().to(device)

            # 前向传播
            outputs = model(serum_feats, demo_feats, input_ids_batch, attn_mask_batch)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * labels.size(0)

        avg_loss = total_loss / len(train_dataset)

        # 验证
        model.eval()
        val_preds, val_labels, val_probs = [], [], []

        with torch.no_grad():
            for batch in val_loader:
                serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
                serum_feats = serum_feats.to(device)
                demo_feats = demo_feats.to(device)
                input_ids_batch = input_ids_batch.to(device)
                attn_mask_batch = attn_mask_batch.to(device)
                labels = labels.squeeze(1).long().to(device)

                outputs = model(serum_feats, demo_feats, input_ids_batch, attn_mask_batch)
                probs = torch.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)

                val_preds.append(preds.cpu())
                val_labels.append(labels.cpu())
                val_probs.append(probs.cpu()[:, 1])

        val_preds = torch.cat(val_preds).numpy()
        val_labels = torch.cat(val_labels).numpy()
        val_probs = torch.cat(val_probs).numpy()

        val_acc = accuracy_score(val_labels, val_preds)
        val_auc = roc_auc_score(val_labels, val_probs)
        val_auprc = average_precision_score(val_labels, val_probs)

        print(
            f"Epoch {epoch + 1}/{num_epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}, Val AUC: {val_auc:.4f}, Val AUPRC: {val_auprc:.4f}")

        # 保存最佳模型
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_states = {
                'spectral_model': model.spectral_model.state_dict() if hasattr(model,
                                                                               'spectral_model') and model.spectral_model is not None else None,
                'serum_encoder': model.serum_encoder.state_dict() if hasattr(model,
                                                                             'serum_encoder') and model.serum_encoder is not None else None,
                'demo_encoder': model.demo_encoder.state_dict() if hasattr(model,
                                                                           'demo_encoder') and model.demo_encoder is not None else None,
                'fusion_model': model.fusion_model.state_dict() if hasattr(model, 'fusion_model') else None
            }
            patience_counter = 0
            print(f"  -> 新的最佳模型！验证集AUC: {best_val_auc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n早停：验证集AUC连续{patience}轮未提升，停止训练")
                break

    # 加载最佳模型
    if best_model_states['spectral_model'] is not None:
        model.spectral_model.load_state_dict(best_model_states['spectral_model'])
    if best_model_states['serum_encoder'] is not None:
        model.serum_encoder.load_state_dict(best_model_states['serum_encoder'])
    if best_model_states['demo_encoder'] is not None:
        model.demo_encoder.load_state_dict(best_model_states['demo_encoder'])
    if best_model_states['fusion_model'] is not None:
        model.fusion_model.load_state_dict(best_model_states['fusion_model'])

    # 测试
    model.eval()
    test_preds, test_labels, test_probs = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            serum_feats, demo_feats, input_ids_batch, attn_mask_batch, labels = batch
            serum_feats = serum_feats.to(device)
            demo_feats = demo_feats.to(device)
            input_ids_batch = input_ids_batch.to(device)
            attn_mask_batch = attn_mask_batch.to(device)
            labels = labels.squeeze(1).long().to(device)

            outputs = model(serum_feats, demo_feats, input_ids_batch, attn_mask_batch)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            test_preds.append(preds.cpu())
            test_labels.append(labels.cpu())
            test_probs.append(probs.cpu()[:, 1])

    test_preds = torch.cat(test_preds).numpy()
    test_labels = torch.cat(test_labels).numpy()
    test_probs = torch.cat(test_probs).numpy()

    # 计算混淆矩阵
    cm = confusion_matrix(test_labels, test_preds)

    test_acc = accuracy_score(test_labels, test_preds)
    test_auc = roc_auc_score(test_labels, test_probs)
    test_auprc = average_precision_score(test_labels, test_probs)

    print(f"\n===== 测试集评估结果: {model_name} =====")
    print(f"混淆矩阵:")
    print(f"[[{cm[0][0]}, {cm[0][1]}],")
    print(f" [{cm[1][0]}, {cm[1][1]}]]")
    print(classification_report(test_labels, test_preds, digits=4))
    print(f"Accuracy: {test_acc:.4f}")
    print(f"AUC: {test_auc:.4f}")
    print(f"AUPRC: {test_auprc:.4f}")

    return {
        'name': model_name,
        'val_auc': best_val_auc,
        'test_acc': test_acc,
        'test_auc': test_auc,
        'test_auprc': test_auprc,
        'confusion_matrix': cm.tolist()
    }


# ==================================================
# 9️⃣ 运行所有消融实验
# ==================================================
if __name__ == "__main__":
    # 定义所有模型变体
    models = [
        BaseModel(),
        NoMultiLayerModel(),
        NoFFTModel(),
        NoGatingModel(),
        NoInterpolationModel(),
        LowFreqOnlyModel(),
        S2_10Model(),
        S2_30Model(),
        NoTopKModel(),
        MaxPoolingModel(),
        CLSPoolingModel(),
        NoSerumModel(),
        NoDemoModel(),
        OnlyTextModel(),
        OnlySerumModel(),
        OnlyDemoModel()
    ]

    # 运行所有模型
    results = []
    for model in models:
        try:
            result = train_and_evaluate(model, model.name)
            results.append(result)
        except Exception as e:
            print(f"模型 {model.name} 训练失败: {e}")
            continue

    # 保存结果
    results_df = pd.DataFrame(results)
    results_df.to_csv("ablation_results.csv", index=False, encoding='utf-8-sig')

    # 打印汇总结果
    print("\n===== 消融实验汇总结果 =====")
    print(results_df)

    # 绘制结果对比图
    plt.figure(figsize=(14, 6))
    plt.bar(results_df['name'], results_df['test_auc'], color='skyblue')
    plt.xlabel('模型变体')
    plt.ylabel('测试集AUC')
    plt.title('消融实验AUC对比')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ablation_auc_comparison.png')
    plt.show()

    plt.figure(figsize=(14, 6))
    plt.bar(results_df['name'], results_df['test_auprc'], color='lightgreen')
    plt.xlabel('模型变体')
    plt.ylabel('测试集AUPRC')
    plt.title('消融实验AUPRC对比')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('ablation_auprc_comparison.png')
    plt.show()

    # 打印所有混淆矩阵
    print("\n===== 所有模型的混淆矩阵 =====")
    for result in results:
        cm = result['confusion_matrix']
        print(f"\n{result['name']}:")
        print(f"[[{cm[0][0]}, {cm[0][1]}],")
        print(f" [{cm[1][0]}, {cm[1][1]}]]")