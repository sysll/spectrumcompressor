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

# 设置随机种子
seed = 32
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ==================================================
# 1. 读取数据
# ==================================================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]           # 最后三列文本
numeric_cols = df.columns[:-3].drop("survival_status")
target_col = 'survival_status'

# 假设人口统计学特征在前10列（根据你的原始代码）
demo_cols = numeric_cols[:10]
serum_cols = numeric_cols[10:]

# ==================================================
# 2. 数值特征标准化
# ==================================================
scaler_serum = StandardScaler()
X_serum = scaler_serum.fit_transform(df[serum_cols].values)

scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(df[demo_cols].values)

# ==================================================
# 3. 定义编码器模型
# ==================================================
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")

# ---- 文本编码器 ---- 
class SpectralCompressor(nn.Module): 
    def __init__(self, model_name= "sentence-transformers/all-MiniLM-L6-v2" , seq2=20): 
        super().__init__() 
        self.seq2 = seq2 
        self.model = AutoModel.from_pretrained(model_name, output_hidden_states= True ) 
        self.hidden_dim = self.model.config.hidden_size 
        self.linear_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim) 
        self.freq_gate = nn.Parameter(torch.randn(seq2, self.hidden_dim)) 

    def forward(self, input_ids, attention_mask): 
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states= True ) 
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
            batch_first= True 
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
# 4. 准备数据
# ==================================================
texts = df[text_cols].fillna("").astype(str).agg(" ".join, axis=1).tolist()
y = df[target_col].values.astype(np.int64)

# ==================================================
# 5. 数据集划分 6:2:2
# ==================================================
train_val_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y
)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.25, random_state=seed, stratify=y[train_val_idx]
)

# ==================================================
# 6. 定义数据集和数据加载器
# ==================================================
class CustomDataset(Dataset):
    def __init__(self, texts, serum, demo, labels):
        self.texts = texts
        self.serum = serum
        self.demo = demo
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.texts[idx], self.serum[idx], self.demo[idx], self.labels[idx]

# 划分数据集
train_val_idx, test_idx = train_test_split(
    np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y
)
train_idx, val_idx = train_test_split(
    train_val_idx, test_size=0.25, random_state=seed, stratify=y[train_val_idx]
)

# 创建数据集
train_dataset = CustomDataset(
    [texts[i] for i in train_idx],
    X_serum[train_idx],
    X_demo[train_idx],
    y[train_idx]
)
val_dataset = CustomDataset(
    [texts[i] for i in val_idx],
    X_serum[val_idx],
    X_demo[val_idx],
    y[val_idx]
)
test_dataset = CustomDataset(
    [texts[i] for i in test_idx],
    X_serum[test_idx],
    X_demo[test_idx],
    y[test_idx]
)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# ==================================================
# 7. 初始化模型
# ==================================================
# 初始化编码器
spectral_encoder = SpectralCompressor().to(device)
serum_encoder = SerumMLPEncoder(seq_len_input=X_serum.shape[1]).to(device)
demo_encoder = DemographicEncoder(seq_len_input=X_demo.shape[1]).to(device)

# 初始化融合模型
model = FusionTransformer3Modal().to(device)

# 优化器
optimizer = torch.optim.Adam(
    list(spectral_encoder.parameters()) +
    list(serum_encoder.parameters()) +
    list(demo_encoder.parameters()) +
    list(model.parameters()),
    lr=1e-3
)

criterion = nn.CrossEntropyLoss()

# ==================================================
# 8. 训练 & 验证循环
# ==================================================
num_epochs = 100
best_val_auprc = 0.0
best_epoch = 0

for epoch in range(num_epochs):
    spectral_encoder.train()
    serum_encoder.train()
    demo_encoder.train()
    model.train()
    
    total_loss = 0
    for batch in train_loader:
        text_batch, serum_batch, demo_batch, label_batch = batch
        
        # 处理文本
        inputs = tokenizer(text_batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # 处理数值特征
        serum_tensor = torch.tensor(serum_batch, dtype=torch.float32, device=device)
        demo_tensor = torch.tensor(demo_batch, dtype=torch.float32, device=device)
        labels = torch.tensor(label_batch, dtype=torch.long, device=device)
        
        # 前向传播
        text_feat = spectral_encoder(input_ids, attention_mask)
        serum_feat = serum_encoder(serum_tensor)
        demo_feat = demo_encoder(demo_tensor)
        
        # 融合预测
        logits = model([text_feat, serum_feat, demo_feat])
        
        # 计算损失
        loss = criterion(logits, labels)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(labels)
    
    train_loss = total_loss / len(train_dataset)
    
    # 验证
    spectral_encoder.eval()
    serum_encoder.eval()
    demo_encoder.eval()
    model.eval()
    
    all_val_labels = []
    all_val_probs = []
    
    with torch.no_grad():
        for batch in val_loader:
            text_batch, serum_batch, demo_batch, label_batch = batch
            
            # 处理文本
            inputs = tokenizer(text_batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs['input_ids'].to(device)
            attention_mask = inputs['attention_mask'].to(device)
            
            # 处理数值特征
            serum_tensor = torch.tensor(serum_batch, dtype=torch.float32).to(device)
            demo_tensor = torch.tensor(demo_batch, dtype=torch.float32).to(device)
            
            # 前向传播
            text_feat = spectral_encoder(input_ids, attention_mask)
            serum_feat = serum_encoder(serum_tensor)
            demo_feat = demo_encoder(demo_tensor)
            
            # 融合预测
            logits = model([text_feat, serum_feat, demo_feat])
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            
            all_val_labels.extend(label_batch)
            all_val_probs.extend(probs)
    
    val_acc = accuracy_score(all_val_labels, np.round(all_val_probs))
    val_auprc = average_precision_score(all_val_labels, all_val_probs)
    
    print(f"Epoch {epoch+1:2d} | Loss: {train_loss:.4f} | "
          f"Val Acc: {val_acc:.4f} | Val AUPRC: {val_auprc:.4f}")
    
    if val_auprc > best_val_auprc:
        best_val_auprc = val_auprc
        best_epoch = epoch + 1
        torch.save({
            'spectral_encoder': spectral_encoder.state_dict(),
            'serum_encoder': serum_encoder.state_dict(),
            'demo_encoder': demo_encoder.state_dict(),
            'model': model.state_dict()
        }, "best_model_fusion.pt")
        print(f"  → New best AUPRC: {best_val_auprc:.4f} @ epoch {best_epoch}")

print(f"\nBest validation epoch: {best_epoch} (AUPRC = {best_val_auprc:.4f})")

# ==================================================
# 9. 最终测试
# ==================================================
print("\n加载最佳模型进行最终测试...")
checkpoint = torch.load("best_model_fusion.pt")
spectral_encoder.load_state_dict(checkpoint['spectral_encoder'])
serum_encoder.load_state_dict(checkpoint['serum_encoder'])
demo_encoder.load_state_dict(checkpoint['demo_encoder'])
model.load_state_dict(checkpoint['model'])

# 测试
spectral_encoder.eval()
serum_encoder.eval()
demo_encoder.eval()
model.eval()

all_test_labels = []
all_test_probs = []
all_test_preds = []

with torch.no_grad():
    for batch in test_loader:
        text_batch, serum_batch, demo_batch, label_batch = batch
        
        # 处理文本
        inputs = tokenizer(text_batch, padding=True, truncation=True, max_length=512, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs['attention_mask'].to(device)
        
        # 处理数值特征
        serum_tensor = torch.tensor(serum_batch, dtype=torch.float32).to(device)
        demo_tensor = torch.tensor(demo_batch, dtype=torch.float32).to(device)
        
        # 前向传播
        text_feat = spectral_encoder(input_ids, attention_mask)
        serum_feat = serum_encoder(serum_tensor)
        demo_feat = demo_encoder(demo_tensor)
        
        # 融合预测
        logits = model([text_feat, serum_feat, demo_feat])
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        
        all_test_labels.extend(label_batch)
        all_test_probs.extend(probs)
        all_test_preds.extend(preds)

print("\n===== Final Test Report (best model) =====")
print(classification_report(all_test_labels, all_test_preds, digits=4))
print(f"Test Accuracy : {accuracy_score(all_test_labels, all_test_preds):.4f}")
print(f"Test AUPRC    : {average_precision_score(all_test_labels, all_test_probs):.4f}")

# 保存用于画ROC和PR曲线的文件
np.save("best_epoch_labels.npy", all_test_labels)
np.save("best_epoch_preds.npy", all_test_preds)
np.save("best_epoch_probs.npy", all_test_probs)

print("\n已保存画图所需文件：")
print("  → best_epoch_labels.npy")
print("  → best_epoch_preds.npy")
print("  → best_epoch_probs.npy")
print("这些文件可直接用于绘制ROC曲线和PR曲线。")