import torch
import torch.nn as nn
import torch.fft as fft
from transformers import AutoTokenizer, AutoModel

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

class SerumMLPEncoder(nn.Module):
    def __init__(self, seq_len_input=33, seq_len_output=37, hidden_dim=128, out_dim=384):
        super().__init__()
        self.seq_len_output = seq_len_output
        self.mlp = nn.Sequential(
            nn.Linear(seq_len_input, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.seq_param = nn.Parameter(torch.randn(seq_len_output, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        feat = self.mlp(x)
        feat = feat.unsqueeze(1)
        seq_feat = feat + self.seq_param.unsqueeze(0)
        return seq_feat

class DemographicEncoder(nn.Module):
    def __init__(self, seq_len_input=10, seq_len_output=10, hidden_dim=64, out_dim=384):
        super().__init__()
        self.seq_len_output = seq_len_output
        self.mlp = nn.Sequential(
            nn.Linear(seq_len_input, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim)
        )
        self.seq_param = nn.Parameter(torch.randn(seq_len_output, out_dim))

    def forward(self, x):
        batch_size = x.size(0)
        feat = self.mlp(x)
        feat = feat.unsqueeze(1)
        seq_feat = feat + self.seq_param.unsqueeze(0)
        return seq_feat

class FusionTransformer3Modal(nn.Module):
    def __init__(self, dim=384, num_layers=2, num_heads=2, class_num=2, dropout=0.2, topk=3):
        super().__init__()
        self.dim = dim
        self.topk = topk

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=dim*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.classifier = nn.Linear(dim, class_num)

        self.topk_weights = nn.Parameter(torch.ones(topk) / topk)

    def weighted_topk_pooling(self, x):
        topk_vals, _ = torch.topk(x, self.topk, dim=1)
        weights = torch.softmax(self.topk_weights, dim=0)
        pooled = (topk_vals * weights.view(1, self.topk, 1)).sum(dim=1)
        return pooled

    def forward(self, x_list):
        all_seq = torch.cat(x_list, dim=1)
        fused = self.transformer(all_seq)
        fused = self.weighted_topk_pooling(fused)
        logits = self.classifier(fused)
        return logits
