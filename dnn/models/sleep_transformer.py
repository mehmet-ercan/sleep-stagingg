"""
SleepTransformer: CNN + Transformer Hybrid Model for Sleep Stage Classification

Architecture:
    Stage 1 — Intra-Epoch Feature Extraction (CNN Encoder):
        CNN1D_V3 feature extractor (MultiScale + Residual + SE + GAP)
        Extracts a fixed-size feature vector from each 30-second PSG epoch.
        Input:  [B, n_channels, samples_per_epoch]
        Output: [B, d_model]  (d_model = 128 by default)

    Stage 2 — Inter-Epoch Temporal Modeling (Transformer Encoder):
        Positional Encoding + Multi-Head Self-Attention
        Learns temporal dependencies across consecutive epochs.
        Input:  [B, seq_len, d_model]
        Output: [B, seq_len, d_model]

    Stage 3 — Classification Head:
        Per-epoch 5-class sleep stage prediction (W, N1, N2, N3, REM).
        Input:  [B, seq_len, d_model]
        Output: [B, seq_len, n_classes]

Key design decisions:
    - CNN1D_V3 as encoder: proven feature extractor, ~260K params
    - Pre-LN Transformer: more stable training without warmup dependency
    - Sinusoidal or Learnable positional encoding (configurable)
    - Sequence-to-sequence: every epoch in the window gets a prediction
    - ~548K total parameters — fits comfortably in 4GB VRAM
    - Stateless: no hidden state management (simpler than BiLSTM)

Reference:
    SleepTransformer (Phan et al. 2022, IEEE Trans. Biomed. Eng.)
    — our approach uses CNN encoder instead of raw-EEG tokenization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os
import warnings

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Import CNN1D_V3 building blocks
from .cnn1d import MultiScaleStem, ResidualBlock, SEBlock


# ============================================================================
# Positional Encoding
# ============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding (Vaswani et al. 2017).

    Parametresiz — epoch sırası bilgisini sabit sinüs/kosinüs fonksiyonlarıyla
    kodlar. Eğitim sırasında güncellenmez.

    Avantajları:
        - Genelleştirme: eğitimde görülmemiş pozisyonlara da uygulanabilir
        - Parametresiz: overfitting riski yok
    """

    def __init__(self, d_model, max_len=25, dropout=0.1, pe_scale=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe_scale = pe_scale

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer (not a parameter — won't be updated by optimizer)
        self.register_buffer('pe', pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model] with positional information added
        """
        # PE scaling: raw PE norm=8.0, CNN feature norm=~6.3
        # Without scaling, PE dominates and destroys class information
        # pe_scale=0.1 → PE contributes ~0.8 to norm (~13% of feature)
        x = x + self.pe[:, :x.size(1)] * self.pe_scale
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding.

    Her pozisyon için öğrenilebilir bir embedding vektörü saklar. Küçük
    veri setlerinde epoch sırası bilgisini daha iyi yakalayabilir, ancak
    overfitting riski biraz daha yüksektir.

    Avantajları:
        - Uyku geçiş kalıplarına özel pozisyon bilgisi öğrenebilir
        - Daha esnek parametre alanı
    """

    def __init__(self, d_model, max_len=25, dropout=0.1, pe_scale=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.pe_scale = pe_scale
        # Small init to avoid dominating the input at start
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.01)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model] with learned positional information added
        """
        x = x + self.pe[:, :x.size(1)] * self.pe_scale
        return self.dropout(x)


# ============================================================================
# CNN Encoder (Stage 1)
# ============================================================================

class CNNEncoder(nn.Module):
    """
    CNN1D_V3 feature extractor wrapper for SleepTransformer.

    CNN1D_V3'ün classifier kısmı hariç, feature extraction katmanlarını
    kullanır: MultiScaleStem → ResBlock × 2 → SE → GAP → [B, d_model]

    Not: Classifier (Dropout+FC+LN+ReLU+Dropout+FC) hariç tutulur.
         Transformer kendi classification head'ini kullanır.
    """

    def __init__(self, cfg=None, freeze=False):
        super().__init__()

        if cfg is None:
            cfg = config.CNN1D_V3_CONFIG
        self.cfg = cfg

        stem_total = cfg['stem_filters'] * len(cfg['stem_kernels'])

        # Feature extraction layers (CNN1D_V3 ile aynı)
        self.stem = MultiScaleStem(
            cfg['n_channels'], cfg['stem_filters'],
            cfg['stem_kernels'], cfg['stem_pool']
        )

        self.block1 = ResidualBlock(
            stem_total, cfg['block1_filters'],
            cfg['block1_kernel'], cfg['block1_pool']
        )

        self.block2 = ResidualBlock(
            cfg['block1_filters'], cfg['block2_filters'],
            cfg['block2_kernel'], cfg['block2_pool']
        )

        self.se = SEBlock(cfg['block2_filters'], cfg['se_reduction'])

        # Output dimension = block2_filters (128 by default)
        self.output_dim = cfg['block2_filters']

        # Freeze CNN parameters if requested
        if freeze:
            self.freeze()

    def freeze(self):
        """CNN parametrelerini dondur (sadece transformer eğitilir)."""
        for param in self.parameters():
            param.requires_grad = False
        print("  🔒 CNN Encoder parametreleri donduruldu (freeze)")

    def unfreeze(self):
        """CNN parametrelerini aç (fine-tuning için)."""
        for param in self.parameters():
            param.requires_grad = True
        print("  🔓 CNN Encoder parametreleri açıldı (unfreeze)")

    def forward(self, x):
        """
        Extract features from a single epoch.

        Args:
            x: [B, n_channels, samples_per_epoch]
        Returns:
            features: [B, output_dim]  (output_dim = 128)
        """
        x = self.stem(x)       # Multi-scale feature extraction
        x = self.block1(x)     # Residual block 1
        x = self.block2(x)     # Residual block 2
        x = self.se(x)         # Channel attention (SE block)
        x = x.mean(dim=2)      # Global Average Pooling → [B, 128]
        return x


# ============================================================================
# Transformer Encoder (Stage 2)
# ============================================================================

class EpochTransformerEncoder(nn.Module):
    """
    Transformer Encoder for inter-epoch temporal modeling.

    25 ardışık epoch'un CNN feature vektörlerini alır, self-attention ile
    epoch'lar arası temporal bağımlılıkları öğrenir.

    Architecture:
        Positional Encoding → N × TransformerEncoderLayer → LayerNorm

    Parametreler:
        d_model=128, nhead=4, num_layers=2, dim_feedforward=256
        → ~267K parametre (2 layer)

    Pre-LN (norm_first=True): daha stabil training, warmup'a daha az bağımlı.
    """

    def __init__(self, d_model=128, nhead=4, num_layers=2,
                 dim_feedforward=256, dropout=0.3, max_seq_len=25,
                 pos_encoding='sinusoidal', activation='gelu'):
        super().__init__()

        self.d_model = d_model

        # Positional Encoding
        if pos_encoding == 'sinusoidal':
            self.pos_encoder = SinusoidalPositionalEncoding(d_model, max_seq_len, dropout)
        elif pos_encoding == 'learnable':
            self.pos_encoder = LearnablePositionalEncoding(d_model, max_seq_len, dropout)
        else:
            raise ValueError(f"Desteklenmeyen pos_encoding: {pos_encoding}. "
                             f"'sinusoidal' veya 'learnable' olmalı.")

        # Transformer Encoder Layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True,       # [B, seq_len, d_model]
            norm_first=True          # Pre-LN (daha stabil training)
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False  # norm_first=True ile nested tensor desteklenmez
        )

        # Final layer norm (Pre-LN convention: add norm after last layer)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            [B, seq_len, d_model] — contextualized epoch representations
        """
        x = self.pos_encoder(x)     # Add positional information
        x = self.transformer(x)     # Self-attention × num_layers
        x = self.norm(x)            # Final normalization
        return x


# ============================================================================
# Classification Head (Stage 3)
# ============================================================================

class ClassificationHead(nn.Module):
    """
    Per-epoch classification head.

    Sequence-to-sequence: her epoch'un bağlam-zenginleştirilmiş vektörünü
    5 uyku evresinden birine dönüştürür.

    Architecture:
        LayerNorm → Dropout → Linear(d_model → n_classes)
    """

    def __init__(self, d_model=128, n_classes=5, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_classes)
        )

    def forward(self, x):
        """
        Args:
            x: [B, seq_len, d_model]
        Returns:
            logits: [B, seq_len, n_classes]
        """
        return self.head(x)


# ============================================================================
# SleepTransformer (Full Model)
# ============================================================================

class SleepTransformer(nn.Module):
    """
    SleepTransformer: CNN + Transformer Hybrid for Sleep Stage Classification.

    Three-stage architecture:
        1. CNN Encoder (per-epoch): Raw PSG → feature vector
        2. Transformer Encoder (inter-epoch): Self-attention across epochs
        3. Classification Head: Per-epoch 5-class prediction

    Args:
        n_channels: Number of PSG channels (default: 4)
        n_classes: Number of sleep stages (default: 5 for W, N1, N2, N3, REM)
        d_model: Transformer embedding dimension (= CNN output, default: 128)
        nhead: Number of attention heads (default: 4)
        num_layers: Number of transformer encoder layers (default: 2)
        dim_feedforward: FFN hidden dimension (default: 256)
        transformer_dropout: Dropout in transformer layers (default: 0.3)
        head_dropout: Dropout in classification head (default: 0.3)
        pos_encoding: 'sinusoidal' or 'learnable' (default: 'sinusoidal')
        max_seq_len: Maximum sequence length (default: 25)
        activation: Transformer activation function (default: 'gelu')
        freeze_cnn: Whether to freeze CNN encoder parameters (default: False)
    """

    def __init__(self,
                 n_channels=4,
                 n_classes=5,
                 d_model=128,
                 nhead=4,
                 num_layers=2,
                 dim_feedforward=256,
                 transformer_dropout=0.3,
                 head_dropout=0.3,
                 pos_encoding='sinusoidal',
                 max_seq_len=25,
                 activation='gelu',
                 freeze_cnn=False):

        super().__init__()

        self.n_channels = n_channels
        self.n_classes = n_classes
        self.d_model = d_model

        # ============================================================
        # Stage 1: CNN Encoder (per-epoch feature extraction)
        # ============================================================
        self.cnn_encoder = CNNEncoder(freeze=freeze_cnn)

        # Verify CNN output matches d_model
        cnn_out_dim = self.cnn_encoder.output_dim
        if cnn_out_dim != d_model:
            # Add projection layer if dimensions don't match
            self.cnn_projection = nn.Linear(cnn_out_dim, d_model)
            print(f"  ⚠ CNN output ({cnn_out_dim}) ≠ d_model ({d_model}), "
                  f"projection layer eklendi")
        else:
            self.cnn_projection = None

        # ============================================================
        # Stage 2: Transformer Encoder (inter-epoch context)
        # ============================================================
        self.transformer_encoder = EpochTransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dim_feedforward=dim_feedforward,
            dropout=transformer_dropout,
            max_seq_len=max_seq_len,
            pos_encoding=pos_encoding,
            activation=activation
        )

        # ============================================================
        # Stage 3: Classification Head (per-epoch prediction)
        # ============================================================
        self.classification_head = ClassificationHead(
            d_model=d_model,
            n_classes=n_classes,
            dropout=head_dropout
        )

        # Initialize weights
        self._initialize_weights()

        # Print model summary
        self._print_model_info(pos_encoding, num_layers, nhead,
                               dim_feedforward, max_seq_len, freeze_cnn)

    def _initialize_weights(self):
        """Initialize transformer and head weights.
        
        IMPORTANT: nn.TransformerEncoder uses deepcopy to create layers,
        so all layers start with identical weights. We must re-initialize
        to break this symmetry. Also, MultiheadAttention.in_proj_weight
        is NOT an nn.Linear submodule — it needs explicit init.
        """
        for name, module in self.named_modules():
            # Skip CNN encoder — it has its own initialization
            if name.startswith('cnn_encoder'):
                continue
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.MultiheadAttention):
                # in_proj_weight is a Parameter, not a Linear submodule
                # Must init explicitly to break deepcopy symmetry
                if module.in_proj_weight is not None:
                    nn.init.xavier_normal_(module.in_proj_weight)
                if module.in_proj_bias is not None:
                    nn.init.zeros_(module.in_proj_bias)
                # out_proj is handled by nn.Linear branch above
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor — supports two formats:
               - Sequence mode: [B, seq_len, n_channels, samples_per_epoch]
               - Single epoch mode: [B, n_channels, samples_per_epoch]

        Returns:
            Sequence mode:     [B, seq_len, n_classes]
            Single epoch mode: [B, n_classes]
        """
        if x.dim() == 4:
            # Sequence mode: [B, seq_len, n_channels, samples]
            return self._forward_sequence(x)
        elif x.dim() == 3:
            # Single epoch mode: [B, n_channels, samples]
            return self._forward_single(x)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D. "
                             f"Shape: {x.shape}")

    def _forward_sequence(self, x):
        """
        Process a sequence of epochs through CNN + Transformer.

        Args:
            x: [B, seq_len, n_channels, samples_per_epoch]
        Returns:
            logits: [B, seq_len, n_classes]
        """
        batch_size, seq_len, n_channels, samples = x.shape

        # Stage 1: CNN encoding (process each epoch independently)
        # Reshape: [B * seq_len, n_channels, samples]
        x_flat = x.view(batch_size * seq_len, n_channels, samples)
        features_flat = self.cnn_encoder(x_flat)  # [B * seq_len, d_model]

        # Optional projection if CNN output ≠ d_model
        if self.cnn_projection is not None:
            features_flat = self.cnn_projection(features_flat)

        # Reshape back: [B, seq_len, d_model]
        features = features_flat.view(batch_size, seq_len, -1)

        # Stage 2: Transformer encoding (inter-epoch context)
        contextualized = self.transformer_encoder(features)  # [B, seq_len, d_model]

        # Stage 3: Classification (per-epoch prediction)
        logits = self.classification_head(contextualized)  # [B, seq_len, n_classes]

        return logits

    def _forward_single(self, x):
        """
        Process a single epoch (no temporal context).
        Uses CNN encoder only, bypasses transformer.

        Args:
            x: [B, n_channels, samples_per_epoch]
        Returns:
            logits: [B, n_classes]
        """
        # Stage 1: CNN encoding
        features = self.cnn_encoder(x)  # [B, d_model]

        # Optional projection
        if self.cnn_projection is not None:
            features = self.cnn_projection(features)

        # Stage 2: Skip transformer (no sequence to attend to)
        # Stage 3: Classify — add fake seq dim, classify, remove
        features = features.unsqueeze(1)  # [B, 1, d_model]
        logits = self.classification_head(features)  # [B, 1, n_classes]
        logits = logits.squeeze(1)  # [B, n_classes]

        return logits

    def get_attention_weights(self, x):
        """
        Extract attention weights for visualization.

        Args:
            x: [B, seq_len, n_channels, samples_per_epoch]
        Returns:
            attention_weights: list of [B, nhead, seq_len, seq_len] per layer
        """
        batch_size, seq_len, n_channels, samples = x.shape

        # CNN encoding
        x_flat = x.view(batch_size * seq_len, n_channels, samples)
        features_flat = self.cnn_encoder(x_flat)
        if self.cnn_projection is not None:
            features_flat = self.cnn_projection(features_flat)
        features = features_flat.view(batch_size, seq_len, -1)

        # Positional encoding
        features = self.transformer_encoder.pos_encoder(features)

        # Extract attention from each layer
        attention_weights = []
        x = features
        for layer in self.transformer_encoder.transformer.layers:
            # Pre-LN: normalize first
            x_norm = layer.norm1(x)
            # Get attention weights
            _, attn_weight = layer.self_attn(
                x_norm, x_norm, x_norm,
                need_weights=True,
                average_attn_weights=False  # Get per-head weights
            )
            attention_weights.append(attn_weight.detach())
            # Complete the forward pass for this layer
            x = x + layer.dropout1(layer.self_attn(x_norm, x_norm, x_norm)[0])
            x = x + layer._ff_block(layer.norm2(x))

        return attention_weights

    def _print_model_info(self, pos_encoding, num_layers, nhead,
                          dim_feedforward, max_seq_len, freeze_cnn):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        cnn_params = sum(p.numel() for p in self.cnn_encoder.parameters())
        transformer_params = sum(p.numel() for p in self.transformer_encoder.parameters())
        head_params = sum(p.numel() for p in self.classification_head.parameters())

        # Memory estimate
        mem_mb = (total_params * 4 * 3) / (1024 * 1024)  # params + grads + optimizer

        print(f"\n{'='*70}")
        print(f"SLEEPTRANSFORMER MODEL (CNN + Transformer Hybrid)")
        print(f"{'='*70}")

        print(f"\n📊 Configuration:")
        print(f"  Input channels: {self.n_channels}")
        print(f"  Output classes: {self.n_classes}")
        print(f"  d_model: {self.d_model}")

        print(f"\n🏗️  Architecture:")
        print(f"  [Stage 1] CNN Encoder (CNN1D_V3 feature extractor):")
        print(f"    MultiScaleStem → ResBlock×2 → SE → GAP → [{self.d_model}]")
        print(f"    Parameters: {cnn_params:,}")
        print(f"    Frozen: {'Yes 🔒' if freeze_cnn else 'No (trainable)'}")

        print(f"\n  [Stage 2] Transformer Encoder:")
        print(f"    Positional Encoding: {pos_encoding}")
        print(f"    Layers: {num_layers}")
        print(f"    Attention heads: {nhead} (head_dim={self.d_model // nhead})")
        print(f"    FFN dim: {dim_feedforward}")
        print(f"    Max sequence length: {max_seq_len}")
        print(f"    Norm: Pre-LN (norm_first=True)")
        print(f"    Parameters: {transformer_params:,}")

        print(f"\n  [Stage 3] Classification Head:")
        print(f"    LayerNorm → Dropout → Linear({self.d_model}→{self.n_classes})")
        print(f"    Parameters: {head_params:,}")

        print(f"\n📈 Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Est. memory (FP32): ~{mem_mb:.0f} MB")
        print(f"{'='*70}\n")


# ============================================================================
# Configuration Presets
# ============================================================================

SLEEPTRANSFORMER_CONFIG_DEFAULT = {
    'n_channels': config.N_CHANNELS,
    'n_classes': config.N_CLASSES,
    'd_model': 128,
    'nhead': 4,
    'num_layers': 2,
    'dim_feedforward': 256,
    'transformer_dropout': 0.3,
    'head_dropout': 0.3,
    'pos_encoding': 'sinusoidal',
    'max_seq_len': 25,
    'activation': 'gelu',
    'freeze_cnn': False,
}


def get_sleep_transformer(config_dict=None, **kwargs):
    """
    Create SleepTransformer model with configuration.

    Args:
        config_dict: Configuration dictionary (uses default if None)
        **kwargs: Override specific config values

    Returns:
        model: SleepTransformer instance

    Example:
        model = get_sleep_transformer()
        model = get_sleep_transformer(pos_encoding='learnable')
        model = get_sleep_transformer(freeze_cnn=True)
    """
    if config_dict is None:
        config_dict = SLEEPTRANSFORMER_CONFIG_DEFAULT.copy()
    else:
        config_dict = config_dict.copy()

    # Override with kwargs
    config_dict.update(kwargs)

    model = SleepTransformer(**config_dict)
    return model


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    print("Testing SleepTransformer...")

    # Test 1: Sequence mode
    print(f"\n{'='*70}")
    print(f"Test 1: Sequence mode (25 epochs)")
    print(f"{'='*70}")

    model = get_sleep_transformer()

    batch_size = 4
    seq_len = 25
    x_seq = torch.randn(batch_size, seq_len, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
    print(f"  Input shape: {x_seq.shape}")

    with torch.no_grad():
        out = model(x_seq)
    print(f"  Output shape: {out.shape}")
    print(f"  Expected: [{batch_size}, {seq_len}, {config.N_CLASSES}]")
    assert out.shape == (batch_size, seq_len, config.N_CLASSES), "Shape mismatch!"
    print(f"  ✓ PASSED")

    # Test 2: Single epoch mode
    print(f"\n{'='*70}")
    print(f"Test 2: Single epoch mode")
    print(f"{'='*70}")

    x_single = torch.randn(batch_size, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
    print(f"  Input shape: {x_single.shape}")

    with torch.no_grad():
        out_single = model(x_single)
    print(f"  Output shape: {out_single.shape}")
    print(f"  Expected: [{batch_size}, {config.N_CLASSES}]")
    assert out_single.shape == (batch_size, config.N_CLASSES), "Shape mismatch!"
    print(f"  ✓ PASSED")

    # Test 3: Learnable PE
    print(f"\n{'='*70}")
    print(f"Test 3: Learnable positional encoding")
    print(f"{'='*70}")

    model_learnable = get_sleep_transformer(pos_encoding='learnable')
    with torch.no_grad():
        out_learn = model_learnable(x_seq)
    print(f"  Output shape: {out_learn.shape}")
    assert out_learn.shape == (batch_size, seq_len, config.N_CLASSES), "Shape mismatch!"
    print(f"  ✓ PASSED")

    # Test 4: Frozen CNN
    print(f"\n{'='*70}")
    print(f"Test 4: Frozen CNN encoder")
    print(f"{'='*70}")

    model_frozen = get_sleep_transformer(freeze_cnn=True)
    cnn_trainable = sum(p.numel() for p in model_frozen.cnn_encoder.parameters() if p.requires_grad)
    total_trainable = sum(p.numel() for p in model_frozen.parameters() if p.requires_grad)
    print(f"  CNN trainable params: {cnn_trainable:,} (should be 0)")
    print(f"  Total trainable params: {total_trainable:,}")
    assert cnn_trainable == 0, "CNN should be frozen!"
    print(f"  ✓ PASSED")

    # Test 5: Attention weights extraction
    print(f"\n{'='*70}")
    print(f"Test 5: Attention weights extraction")
    print(f"{'='*70}")

    model.eval()
    with torch.no_grad():
        attn_weights = model.get_attention_weights(x_seq)
    print(f"  Number of layers: {len(attn_weights)}")
    for i, aw in enumerate(attn_weights):
        print(f"  Layer {i}: {aw.shape}  (B, nhead, seq, seq)")
    print(f"  ✓ PASSED")

    # Memory summary
    print(f"\n{'='*70}")
    print(f"SUMMARY")
    print(f"{'='*70}")
    total = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total:,}")
    print(f"  Model size (FP32): {total * 4 / (1024**2):.1f} MB")
    print(f"  Est. training memory: {total * 4 * 3 / (1024**2):.1f} MB")
    print(f"\n✓ All tests passed!")
