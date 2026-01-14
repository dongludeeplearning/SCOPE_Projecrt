"""
This is the third experiment: MoE-Transformer.
CNN + MoE-Transformer + Attention Pooling
Features:
- Replaces standard FFN in Transformer with Mixture-of-Experts (MoE) Layer.
- 4 Experts with Top-2 Gating.
- Noisy Gating for load balancing during training.
Author: Lu Dong
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  # TIMM provide InceptionNet

# 1. Inception Feature Extractor
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model_name="inception_v3", feature_dim=2048, device="cuda"):
        """
        InceptionNet as feature extractor.
        """
        super().__init__()
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.feature_dim = feature_dim

    def forward(self, x):
        return self.backbone(x.to(self.device))


# 2. MoE Components
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=2, noisy_gating=True):
        """
        MoE Layer with Top-K Gating
        :param input_dim: Input feature dimension (d_model)
        :param hidden_dim: Hidden dimension for experts (d_ff)
        :param num_experts: Number of experts
        :param k: Top-k experts to select
        """
        super().__init__()
        self.k = k
        self.num_experts = num_experts
        self.noisy_gating = noisy_gating
        
        # Experts: 4 independent FFNs
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim)
            ) for _ in range(num_experts)
        ])
        
        # Gating Network (Router)
        self.router = nn.Linear(input_dim, num_experts)
        self.w_noise = nn.Parameter(torch.zeros(input_dim, num_experts), requires_grad=True)
        self.softplus = nn.Softplus()

    def forward(self, x):
        # x shape: (batch_size, seq_len, feature_dim) -> flatten to (batch*seq, dim)
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        
        # Gating logic
        gate_logits = self.router(x_flat)
        
        if self.noisy_gating and self.training:
             # Standard Noisy Top-K Gating for load balancing exploration
            clean_logits = gate_logits
            raw_noise_stddev = x_flat @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = gate_logits

        # Select Top-K experts
        top_k_logits, indices = logits.topk(self.k, dim=1)
        top_k_probs = F.softmax(top_k_logits, dim=1)
        
        # Compute expert outputs
        # Simplified execution for small number of experts
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.k):
            expert_idx = indices[:, i]  # Which expert for each token
            prob = top_k_probs[:, i].unsqueeze(1) # Weight for this choice
            
            # Mask generation and selective execution
            for e in range(self.num_experts):
                 mask = (expert_idx == e)
                 if mask.sum() > 0:
                     selected_input = x_flat[mask]
                     expert_out = self.experts[e](selected_input)
                     final_output[mask] += expert_out * prob[mask]

        return final_output.view(original_shape)


# 3. MoE Transformer Layer
class MoETransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, num_experts=4, k=2):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        
        # Replace standard FFN with MoE
        self.moe = MoELayer(d_model, dim_feedforward, num_experts=num_experts, k=k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        # Self-Attention Block
        # src shape: (batch, seq, dim)
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
        # MoE Block
        src2 = self.moe(src)
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class VideoMoETransformer(nn.Module):
    def __init__(self, feature_dim=1024, num_layers=4, num_heads=8, ff_dim=2048, 
                 use_pos_encoding=True, num_experts=4, k=2, device="cuda"):
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.use_pos_encoding = use_pos_encoding
        
        # Stack MoE Layers
        # Using fewer layers (4) than v1/v2 (6) to keep parameter count manageable with MoE
        self.layers = nn.ModuleList([
            MoETransformerEncoderLayer(
                d_model=feature_dim, 
                nhead=num_heads, 
                dim_feedforward=ff_dim,
                num_experts=num_experts,
                k=k
            ) for _ in range(num_layers)
        ])

    def _get_sinusoidal_pos_encoding(self, seq_len):
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=self.device) * 
                            -(torch.log(torch.tensor(10000.0, device=self.device)) / self.feature_dim))
        pos_encoding = torch.zeros(seq_len, self.feature_dim, device=self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding

    def forward(self, x):
        x = x.to(self.device)
        if self.use_pos_encoding:
            batch_size, seq_len, _ = x.shape
            pos_emb = self._get_sinusoidal_pos_encoding(seq_len)
            x = x + pos_emb.unsqueeze(0)

        for layer in self.layers:
            x = layer(x)
        return x


# 4. Multi-Task Classifier (Standard)
class MultiTaskClassifier(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=4, device="cuda"):
        super().__init__()
        self.device = device
        self.attention_weights = nn.Linear(feature_dim, 1).to(self.device)
        self.softmax = nn.Softmax(dim=1)
        self.task_heads = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(4)])

    def forward(self, x):
        x = x.to(self.device)
        attn_scores = self.attention_weights(x)
        attn_scores = self.softmax(attn_scores)
        x = torch.sum(attn_scores * x, dim=1)
        return [task_head(x) for task_head in self.task_heads]


# 5. Main Model Wrapper
class MultiTaskVideoClassifier(nn.Module):
    def __init__(self, use_pos_encoding=True, device="cuda"):
        """
        MoE-based Video Classifier
        """
        super().__init__()
        self.device = device
        self.feature_extractor = InceptionFeatureExtractor().to(self.device)
        self.projection = nn.Linear(2048, 1024).to(self.device)
        
        # MoE Transformer
        self.transformer = VideoMoETransformer(
            feature_dim=1024,
            num_layers=4,     # 4 Layers of MoE
            num_experts=4,    # 4 Experts
            k=2,              # Top-2 Routing
            use_pos_encoding=use_pos_encoding,
            device=device
        ).to(self.device)
        
        self.classifier = MultiTaskClassifier(feature_dim=1024).to(self.device)

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
            
        batch_size, seq_len, c, h, w = x.shape
        
        # Inception Feature Extraction
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)
        features = features.view(batch_size, seq_len, -1)
        features = self.projection(features)
        
        # MoE Transformer
        transformed_features = self.transformer(features)
        
        # Classification
        return self.classifier(transformed_features)

if __name__ == "__main__":
    # Test initialization
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskVideoClassifier(device=device)
    dummy_input = torch.randn(2, 10, 3, 299, 299).to(device)
    output = model(dummy_input)
    print("V3 Model Initialized Successfully")
    print("Output shapes:", [o.shape for o in output])

