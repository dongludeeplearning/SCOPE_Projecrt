

# inception_relative_model.py
"""
Relative / Pairwise Ranking Model
CNN(InceptionV3) + Transformer + Attention Pooling -> 4 task scalar scores
Compatible with au/va arguments (not fused by default).
"""
import torch
import torch.nn as nn
import timm


class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model_name="inception_v3"):
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # (B,2048)

    def forward(self, x):
        return self.backbone(x)  # (B,2048)


class VideoTransformer(nn.Module):
    def __init__(self, feature_dim=1024, num_layers=6, num_heads=8, ff_dim=2048, use_pos_encoding=True):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_pos_encoding = use_pos_encoding

        enc_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            batch_first=True,  # (B,T,D)
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

    def _sinusoidal_pe(self, T, device):
        position = torch.arange(0, T, dtype=torch.float, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=device)
            * (-(torch.log(torch.tensor(10000.0, device=device)) / self.feature_dim))
        )
        pe = torch.zeros(T, self.feature_dim, device=device)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe  # (T,D)

    def forward(self, x):
        if self.use_pos_encoding:
            B, T, D = x.shape
            pe = self._sinusoidal_pe(T, x.device)
            x = x + pe.unsqueeze(0)
        return self.transformer(x)  # (B,T,D)


class MultiTaskScorer(nn.Module):
    def __init__(self, feature_dim=1024, num_tasks=4):
        super().__init__()
        self.attn = nn.Linear(feature_dim, 1)  # (B,T,1)
        self.softmax = nn.Softmax(dim=1)
        self.heads = nn.ModuleList([nn.Linear(feature_dim, 1) for _ in range(num_tasks)])

    def forward(self, x):
        # x: (B,T,D)
        a = self.softmax(self.attn(x))   # (B,T,1)
        pooled = torch.sum(a * x, dim=1) # (B,D)
        scores = [h(pooled).squeeze(-1) for h in self.heads]  # each (B,)
        return scores


class InceptionRelativeRanker(nn.Module):
    def __init__(self, use_pos_encoding=True):
        super().__init__()
        self.feature_extractor = InceptionFeatureExtractor()
        self.proj = nn.Linear(2048, 1024)
        self.temporal = VideoTransformer(feature_dim=1024, use_pos_encoding=use_pos_encoding)
        self.scorer = MultiTaskScorer(feature_dim=1024, num_tasks=4)

    def forward(self, frames, au=None, va=None):
        """
        frames: (B,T,3,224,224)
        au/va: accepted for compatibility; not fused by default.
        """
        B, T, C, H, W = frames.shape
        x = frames.view(B * T, C, H, W)
        feat = self.feature_extractor(x)         # (B*T,2048)
        feat = feat.view(B, T, -1)               # (B,T,2048)
        feat = self.proj(feat)                   # (B,T,1024)
        feat = self.temporal(feat)               # (B,T,1024)
        return self.scorer(feat)                 # list of 4 scores (B,)