"""Siamese network architecture for render-reality matching."""

from typing import Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from .config import ModelConfig


class EmbeddingHead(nn.Module):
    """Embedding head to convert backbone features to compact embeddings."""
    
    def __init__(self, in_features: int, embedding_dim: int):
        """Initialize embedding head.
        
        Args:
            in_features: Number of input features from backbone.
            embedding_dim: Output embedding dimension.
        """
        super().__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.bn2 = nn.BatchNorm1d(embedding_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input features of shape (batch_size, in_features).
            
        Returns:
            L2-normalized embeddings of shape (batch_size, embedding_dim).
        """
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        # L2 normalize embeddings
        x = F.normalize(x, p=2, dim=1)
        return x


class ComparisonHead(nn.Module):
    """Comparison head to compute similarity score from embeddings."""
    
    def __init__(self, embedding_dim: int):
        """Initialize comparison head.
        
        Args:
            embedding_dim: Dimension of input embeddings.
        """
        super().__init__()
        # Input: concatenated embeddings + element-wise difference + product
        input_dim = embedding_dim * 4
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 1)
    
    def forward(
        self, embedding_a: torch.Tensor, embedding_b: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            embedding_a: First embedding (source) of shape (batch_size, embedding_dim).
            embedding_b: Second embedding (render) of shape (batch_size, embedding_dim).
            
        Returns:
            Similarity scores of shape (batch_size, 1) in range [0, 1].
        """
        # Combine embeddings in multiple ways
        diff = torch.abs(embedding_a - embedding_b)
        product = embedding_a * embedding_b
        combined = torch.cat([embedding_a, embedding_b, diff, product], dim=1)
        
        x = F.relu(self.bn1(self.fc1(combined)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = torch.sigmoid(self.fc3(x))
        return x


def get_backbone(name: str, pretrained: bool = True) -> Tuple[nn.Module, int]:
    """Get backbone network.
    
    Args:
        name: Backbone name (resnet18, resnet34, resnet50, efficientnet_b0).
        pretrained: Whether to use pretrained weights.
        
    Returns:
        Tuple of (backbone_module, num_features).
    """
    weights = "IMAGENET1K_V1" if pretrained else None
    
    if name == "resnet18":
        model = models.resnet18(weights=weights)
        num_features = model.fc.in_features
        # Remove the final FC layer
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    elif name == "resnet34":
        model = models.resnet34(weights=weights)
        num_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    elif name == "resnet50":
        model = models.resnet50(weights=weights)
        num_features = model.fc.in_features
        model = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        
    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=weights)
        num_features = model.classifier[1].in_features
        model.classifier = nn.Identity()
        model = nn.Sequential(model, nn.Flatten())
        
    else:
        raise ValueError(f"Unknown backbone: {name}")
    
    return model, num_features


class SiameseNetwork(nn.Module):
    """Siamese network for comparing render and reality images."""
    
    def __init__(self, config: ModelConfig):
        """Initialize Siamese network.
        
        Args:
            config: Model configuration.
        """
        super().__init__()
        self.config = config
        
        # Shared backbone
        self.backbone, num_features = get_backbone(
            config.backbone, config.pretrained
        )
        
        # Optionally freeze backbone
        if config.freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Embedding head (shared)
        self.embedding_head = EmbeddingHead(num_features, config.embedding_dim)
        
        # Comparison head
        self.comparison_head = ComparisonHead(config.embedding_dim)
    
    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get embedding for a single image.
        
        Args:
            x: Input image of shape (batch_size, 3, H, W).
            
        Returns:
            Embedding of shape (batch_size, embedding_dim).
        """
        features = self.backbone(x)
        embedding = self.embedding_head(features)
        return embedding
    
    def forward(
        self,
        source: torch.Tensor,
        render: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.
        
        Args:
            source: Source (reality) images of shape (batch_size, 3, H, W).
            render: Render images of shape (batch_size, 3, H, W).
            
        Returns:
            Similarity scores of shape (batch_size, 1) in range [0, 1].
        """
        # Get embeddings using shared backbone
        embedding_source = self.get_embedding(source)
        embedding_render = self.get_embedding(render)
        
        # Compare embeddings
        score = self.comparison_head(embedding_source, embedding_render)
        return score
    
    def get_embeddings(
        self,
        source: torch.Tensor,
        render: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get embeddings for both images (useful for visualization).
        
        Args:
            source: Source (reality) images of shape (batch_size, 3, H, W).
            render: Render images of shape (batch_size, 3, H, W).
            
        Returns:
            Tuple of (source_embeddings, render_embeddings).
        """
        embedding_source = self.get_embedding(source)
        embedding_render = self.get_embedding(render)
        return embedding_source, embedding_render


def count_parameters(model: nn.Module, trainable_only: bool = True) -> int:
    """Count number of parameters in model.
    
    Args:
        model: PyTorch model.
        trainable_only: If True, count only trainable parameters.
        
    Returns:
        Number of parameters.
    """
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())
