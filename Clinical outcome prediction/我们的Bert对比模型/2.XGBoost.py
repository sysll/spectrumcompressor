import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.fft as fft
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
import xgboost as xgb

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 18
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
        cls_emb = compressed.mean(dim=1).cpu().numpy()
        text_features.append(cls_emb)

text_features = np.vstack(text_features)

# ================== 人口统计学特征 ==================
X_demo = df[demo_cols].values
scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(X_demo)

# ================== 融合向量 ==================
X = np.hstack([text_features, X_demo])
y = df[target_col].values

# ================== 划分训练集、验证集和测试集 (6:2:2) ==================
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=seed, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=seed, stratify=y_temp)

print(f"训练集: {len(X_train)}, 验证集: {len(X_val)}, 测试集: {len(X_test)}")

# ================== XGBoost 分类器（使用验证集早停） ==================
clf = xgb.XGBClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    objective='binary:logistic',
    eval_metric='auc',
    use_label_encoder=False,
    random_state=seed
)

clf.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],  # 使用验证集而不是测试集
    early_stopping_rounds=50,
    verbose=False
)

# ================== 测试集评估 ==================
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

print("\n===== 测试集评估结果 =====")
print(classification_report(y_test, y_pred, digits=4))
print(f"AUC: {roc_auc_score(y_test, y_proba):.4f}")
print(f"AP: {average_precision_score(y_test, y_proba):.4f}")