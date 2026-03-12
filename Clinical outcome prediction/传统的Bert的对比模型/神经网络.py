import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# ================== 配置 ==================
device = 'cuda' if torch.cuda.is_available() else 'cpu'
seed = 18
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# ================== 读取数据 ==================
df = pd.read_excel("./patients_all_data_encoded.xlsx")
text_cols = df.columns[-3:]  # 文本列
numeric_cols = df.columns[:-3].drop("survival_status")  # 数值列
target_col = 'survival_status'

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
        cls_emb = outputs.last_hidden_state[:, 0, :].cpu().numpy()  # [CLS]向量
        text_features.append(cls_emb)

text_features = np.vstack(text_features)  # [样本数, hidden_size]

# ================== 数值特征（人口学等） ==================
X_demo = df[numeric_cols].values
scaler_demo = StandardScaler()
X_demo = scaler_demo.fit_transform(X_demo)

# ================== 融合向量 ==================
X = np.hstack([text_features, X_demo])
y = df[target_col].values

# ================== 划分训练/测试集 ==================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=seed, stratify=y
)

# 转成Tensor
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
y_train = torch.tensor(y_train, dtype=torch.long).to(device)
y_test = torch.tensor(y_test, dtype=torch.long).to(device)

# ================== 定义神经网络 ==================
class MLPClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, dropout=0.2):
        super(MLPClassifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 2)  # 二分类输出
        )

    def forward(self, x):
        return self.model(x)

# 初始化网络
input_dim = X_train.shape[1]
model_nn = MLPClassifier(input_dim=input_dim).to(device)

# ================== 训练配置 ==================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_nn.parameters(), lr=1e-4, weight_decay=1e-5)
num_epochs = 30
batch_size = 32

# ================== 训练循环 ==================
for epoch in range(num_epochs):
    model_nn.train()
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0

    for i in range(0, X_train.size(0), batch_size):
        optimizer.zero_grad()
        idx = permutation[i:i+batch_size]
        batch_x, batch_y = X_train[idx], y_train[idx]
        outputs = model_nn(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {epoch_loss:.4f}")

# ================== 测试 ==================
model_nn.eval()
with torch.no_grad():
    logits = model_nn(X_test)
    y_pred = torch.argmax(logits, dim=1).cpu().numpy()
    y_proba = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()

print("\n===== Test Report =====")
print(classification_report(y_test.cpu().numpy(), y_pred, digits=4))
print("Accuracy:", accuracy_score(y_test.cpu().numpy(), y_pred))
print("AUC:", roc_auc_score(y_test.cpu().numpy(), y_proba))
