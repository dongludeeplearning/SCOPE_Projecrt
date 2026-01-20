"""
This is the fourth experiment: EfficientNet-B3 + MoE-Transformer.
Backbone: EfficientNet-B3 (Better feature extraction with SE blocks)
Temporal: MoE-Transformer (4 Experts, Top-2 Gating)
ckpt: inception-transformer-v4.pth
Author: Lu Dong
Date: 2026-01-13
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm  

# 1. EfficientNet Feature Extractor
class EfficientNetFeatureExtractor(nn.Module):
    def __init__(self, model_name="efficientnet_b3", device="cuda"):
        """
        EfficientNet as feature extractor.
        """
        super().__init__()
        self.device = device
        # pretrained=True loads ImageNet weights
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.feature_dim = self.backbone.num_features  # 1536 for B3

    def forward(self, x):
        return self.backbone(x.to(self.device))

# 2. Multimodal Fusion Module (Copied from v5 for compatibility)
class MultimodalFusionModule(nn.Module):
    def __init__(self, vis_dim, feat_dim=43, out_dim=1024, device="cuda"):
        super().__init__()
        self.device = device
        self.vis_proj = nn.Linear(vis_dim, 768).to(device)
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Linear(64, 256) 
        ).to(device)
        self.fusion_layer = nn.Sequential(
            nn.Linear(768 + 256, out_dim), nn.LayerNorm(out_dim), nn.Dropout(0.1)
        ).to(device)

    def forward(self, vis_feat, extra_feat):
        B, T, _ = vis_feat.shape
        x_vis = self.vis_proj(vis_feat) 
        x_feat = self.feat_encoder(extra_feat.view(B*T, -1)).view(B, T, -1)
        return self.fusion_layer(torch.cat([x_vis, x_feat], dim=-1))

# 3. MoE Components
class MoELayer(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_experts=4, k=2, noisy_gating=True):
        """
        MoE Layer with Top-K Gating
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
        original_shape = x.shape
        x_flat = x.view(-1, x.shape[-1])
        
        gate_logits = self.router(x_flat)
        
        if self.noisy_gating and self.training:
            clean_logits = gate_logits
            raw_noise_stddev = x_flat @ self.w_noise
            noise_stddev = self.softplus(raw_noise_stddev) + 1e-2
            noisy_logits = clean_logits + (torch.randn_like(clean_logits) * noise_stddev)
            logits = noisy_logits
        else:
            logits = gate_logits

        top_k_logits, indices = logits.topk(self.k, dim=1)
        top_k_probs = F.softmax(top_k_logits, dim=1)
        
        final_output = torch.zeros_like(x_flat)
        
        for i in range(self.k):
            expert_idx = indices[:, i]
            prob = top_k_probs[:, i].unsqueeze(1)
            
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
        self.moe = MoELayer(d_model, dim_feedforward, num_experts=num_experts, k=k)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src):
        src2 = self.self_attn(src, src, src)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        
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


# 4. Multi-Task Classifier
class MultiTaskClassifier(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=4, device="cuda"):
        super().__init__()
        self.device = device
        self.attention_weights = nn.Linear(feature_dim, 1).to(self.device)
        self.softmax = nn.Softmax(dim=1)
        self.task_heads = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(4)])

    def forward(self, x, return_attn=False):
        x = x.to(self.device)
        attn_scores = self.attention_weights(x)
        attn_scores = self.softmax(attn_scores)
        x = torch.sum(attn_scores * x, dim=1)
        
        logits = [task_head(x) for task_head in self.task_heads]
        
        if return_attn:
            return logits, attn_scores
        return logits


# 5. Main Model
class MultiTaskVideoClassifier(nn.Module):
    def __init__(self, use_pos_encoding=True, use_au=False, use_va=False, downsample_ratio=1, device="cuda"):
        """
        EfficientNet-B3 + MoE Transformer (+ Optional Fusion)
        :param downsample_ratio: Downsample input frames by this ratio. e.g. 2 means take every 2nd frame (60->30).
        """
        super().__init__()
        self.device = device
        self.use_au = use_au
        self.use_va = use_va
        self.use_fusion = use_au or use_va
        self.downsample_ratio = downsample_ratio
        
        # Backbone: EfficientNet-B3
        self.feature_extractor = EfficientNetFeatureExtractor(model_name="efficientnet_b3", device=device)
        
        # Determine feature dimension
        vis_dim = self.feature_extractor.feature_dim
        
        if self.use_fusion:
            # Calculate extra feature dimension
            feat_dim = 0
            if use_au: feat_dim += 41
            if use_va: feat_dim += 2
            
            # Use Fusion Module
            self.fusion_module = MultimodalFusionModule(
                vis_dim=vis_dim, 
                feat_dim=feat_dim, 
                out_dim=1024, 
                device=device
            )
            # No separate projection needed as fusion outputs 1024
            self.projection = None 
        else:
            # Standard Projection
            self.projection = nn.Linear(vis_dim, 1024).to(self.device)
            self.fusion_module = None
        
        # Temporal: MoE Transformer
        self.transformer = VideoMoETransformer(
            feature_dim=1024,
            num_layers=4,    
            num_experts=4,  
            k=2,            
            use_pos_encoding=use_pos_encoding,
            device=device
        ).to(self.device)
        
        self.classifier = MultiTaskClassifier(feature_dim=1024).to(self.device)

    def forward(self, x, au=None, va=None, extra_feat=None, return_attn=False):
        if x.device != self.device:
            x = x.to(self.device)
            
        # Apply Downsampling if ratio > 1
        if self.downsample_ratio > 1:
            x = x[:, ::self.downsample_ratio]
            
        # Handle parameter compatibility
        if extra_feat is None:
            feats_to_cat = []
            if au is not None: feats_to_cat.append(au)
            if va is not None: feats_to_cat.append(va)
            
            if len(feats_to_cat) > 0:
                extra_feat = torch.cat(feats_to_cat, dim=-1)
        
        # Apply Downsampling to features as well
        if self.downsample_ratio > 1 and extra_feat is not None:
            extra_feat = extra_feat[:, ::self.downsample_ratio]
            
        batch_size, seq_len, c, h, w = x.shape
        
        # Feature Extraction
        x = x.view(batch_size * seq_len, c, h, w)
        vis_features = self.feature_extractor(x)
        vis_features = vis_features.view(batch_size, seq_len, -1)
        
        if self.use_fusion and extra_feat is not None:
            if extra_feat.device != self.device:
                extra_feat = extra_feat.to(self.device)
            features = self.fusion_module(vis_features, extra_feat)
        else:
            if self.use_fusion:
                # Fallback or error if fusion is expected but no features provided
                # For robustness, we might want to throw error, but let's try to proceed if projection exists
                if self.projection is None:
                     raise ValueError("Model initialized with use_fusion=True but no AU/VA features provided.")
            features = self.projection(vis_features)
        
        # Temporal Modeling
        transformed_features = self.transformer(features)
        
        # Classification
        return self.classifier(transformed_features, return_attn=return_attn)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskVideoClassifier(device=device)
    # EfficientNet input usually 224 or bigger. DAiSEE is usually processed to 224 or 299.
    dummy_input = torch.randn(2, 10, 3, 224, 224).to(device) 
    output = model(dummy_input)
    print("V4 Model (EfficientNet-B3) Initialized Successfully")
    print("Output shapes:", [o.shape for o in output])

