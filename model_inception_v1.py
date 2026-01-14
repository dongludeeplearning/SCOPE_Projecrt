
"""
This is the first experiment of transformer-based model.
CNN + Transformer + Attention Pooling
with positional encoding
ckpt: inception-transformer-v1.pth
result: Test Accuracy for BECF: [0.46149826719578135, 0.5216027989412434, 0.7013937434772166, 0.7851916528225776]
Author: Lu Dong
Date: 2026-01-12
"""
import torch
import torch.nn as nn
import timm  # TIMM provide InceptionNet

# InceptionNet 
class InceptionFeatureExtractor(nn.Module):
    def __init__(self, model_name="inception_v3", feature_dim=2048,device="cuda"):
        """
        InceptionNet as feature extractor.
        :param model_name: Inception model name (default: inception_v3)
        :param feature_dim: Feature dimension to extract (default 2048 for InceptionV3)
        """
        super().__init__()
        self.device = device
        self.backbone = timm.create_model(model_name, pretrained=True, num_classes=0)  # Pretrained model without classification head
        self.feature_dim = feature_dim

    def forward(self, x):
        """
        :param x: Input shape (batch_size * seq_len, 3, H, W)
        :return: Extracted features (batch_size * seq_len, feature_dim)
        """
        return self.backbone(x.to(self.device))  # Output 2048-dimensional features


# Transformer for temporal information processing
class VideoTransformer(nn.Module):
    def __init__(self, feature_dim=1024, num_layers=6, num_heads=8, ff_dim=2048, use_pos_encoding=False, device="cuda"):
        """
        Transformer-based temporal model
        :param feature_dim: Input dimensions (InceptionNet provides 2048-dimensional features)
        :param num_layers: Transformer layers
        :param num_heads: Multi-Head Attention heads
        :param ff_dim: Feedforward network hidden layer size in Transformer
        :param use_pos_encoding: Whether to use sinusoidal positional encoding (default: False)
        :param device: Device to run on
        """
        super().__init__()
        self.device = device
        self.feature_dim = feature_dim
        self.use_pos_encoding = use_pos_encoding
        
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,   
            nhead=num_heads,       
            dim_feedforward=ff_dim 
        )
        self.transformer = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

    def _get_sinusoidal_pos_encoding(self, seq_len):
        """
        Generate sinusoidal positional encoding
        :param seq_len: Sequence length
        :return: Positional encoding of shape (seq_len, feature_dim)
        """
        position = torch.arange(0, seq_len, dtype=torch.float, device=self.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.feature_dim, 2, dtype=torch.float, device=self.device) * 
                            -(torch.log(torch.tensor(10000.0, device=self.device)) / self.feature_dim))
        
        pos_encoding = torch.zeros(seq_len, self.feature_dim, device=self.device)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        
        return pos_encoding

    def forward(self, x):
        """
        :param x: Input shape (batch_size, seq_len, feature_dim)
        :return: Transformer processed features
        """
        x = x.to(self.device)
        
        # Add sinusoidal positional encoding if enabled
        if self.use_pos_encoding:
            batch_size, seq_len, feature_dim = x.shape
            pos_encoding = self._get_sinusoidal_pos_encoding(seq_len)  # (seq_len, feature_dim)
            x = x + pos_encoding.unsqueeze(0)  # Add to batch: (batch_size, seq_len, feature_dim)
        
        return self.transformer(x)


# Multi-task emotion classifier
class MultiTaskClassifier(nn.Module):
    def __init__(self, feature_dim=1024, num_classes=4,device="cuda"):
        """
        Learnable Attention + classification head
        :param feature_dim: Feature dimension from Transformer output
        :param num_classes: Number of classes for each task
        """
        super().__init__()
        self.device = device
        self.attention_weights = nn.Linear(feature_dim, 1).to(self.device)  # Compute attention weights
        self.softmax = nn.Softmax(dim=1)  # Normalization
        self.task_heads = nn.ModuleList([nn.Linear(feature_dim, num_classes) for _ in range(4)])  # 4 classification tasks

    def forward(self, x):
        """
        Forward process
        :param x: Input shape (batch_size, seq_len, feature_dim)
        :return: Classification results for 4 tasks
        """
        x = x.to(self.device)
        attn_scores = self.attention_weights(x)  # Compute attention weights
        attn_scores = self.softmax(attn_scores)  # Normalization
        x = torch.sum(attn_scores * x, dim=1)  # Attention computes weighted sum
        return [task_head(x) for task_head in self.task_heads]  # Output for 4 tasks



class MultiTaskVideoClassifier(nn.Module):
    def __init__(self, use_pos_encoding=False, device="cuda"):
        """
        Build complete video emotion recognition model
        :param use_pos_encoding: Whether to use sinusoidal positional encoding in transformer (default: False)
        :param device: Device to run on
        """
        super().__init__()
        self.device = device
        self.feature_extractor = InceptionFeatureExtractor().to(self.device)
        # Project 2048-d features from Inception to 1024-d for transformer
        self.projection = nn.Linear(2048, 1024).to(self.device)
        self.transformer = VideoTransformer(feature_dim=1024, use_pos_encoding=use_pos_encoding).to(self.device)
        self.classifier = MultiTaskClassifier(feature_dim=1024).to(self.device)

    def forward(self, x):
        """
        :param x: Input shape (batch_size, seq_len, 3, H, W)
        :return: 4 tasks' output
        """
        if x.device != self.device:
            x = x.to(self.device)  # GPU
        batch_size, seq_len, c, h, w = x.shape

        # adapt to InceptionNet
        x = x.view(batch_size * seq_len, c, h, w)
        features = self.feature_extractor(x)  
        features = features.view(batch_size, seq_len, -1)  # (batch_size, seq_len, 2048)
        features = self.projection(features)               # (batch_size, seq_len, 1024)

        # Transformer 
        transformed_features = self.transformer(features)

        # Classifier
        return self.classifier(transformed_features)


# Running case 
if __name__ == "__main__":
    from dataloader import get_dataloader  

    root_folder = "/Work/Datesets/DAiSEE_Process/DataSet"
    label_path = "/mnt/pub/CognitiveDataset/DAiSEE/Labels/AllLabels.csv"
    split = "Test"
    batch_size = 2  

    dataloader = get_dataloader(split, root_folder, label_path, batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiTaskVideoClassifier().to(device)

    # 
    # for name, param in model.named_parameters():
    #     if param.device != device:
    #         print(f"⚠️ Parameter {name} is on {param.device}, expected {device}")

    batch_frames, batch_labels = next(iter(dataloader))

    batch_frames = batch_frames.to(device)
    batch_labels = batch_labels.to(device)

    outputs = model(batch_frames)

    # from pdb import set_trace; set_trace()

    print("Model Outputs:")
    for i, output in enumerate(outputs):
        print(f"Task {i+1} Output Shape: {output.shape}")  

    print("Example Output for First Video:")
    for i, output in enumerate(outputs):
        print(f"Task {i+1} Prediction:", output[0].detach().cpu().numpy())

    print("Ground Truth Labels for First Video:")
    print(batch_labels[0].detach().cpu().numpy())
