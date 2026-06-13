"""
Sleep Stage Classification - 1D CNN Models
Ham EEG sinyallerinden öğrenen CNN mimarileri

Models:
  - CNN1D: Basit 3-layer CNN (1.68M params)
  - CNN1D_V3: Multi-scale + Residual + SE + GAP (~280K params)
  - CNN1DPaper: Paper replication CNN
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import config


# ============================================================================
# Building Blocks (CNN1D_V3 için)
# ============================================================================

class MultiScaleStem(nn.Module):
    """
    Multi-scale konvolüsyon stem bloğu.
    Farklı kernel boyutlarıyla paralel feature extraction.
    Kernel boyutları config'de SAMPLE_RATE'e göre otomatik ölçeklenir:
      - Küçük kernel (~150ms): Sleep spindles, K-complex peaks
      - Orta kernel (~510ms): Theta/alpha rhythms
      - Büyük kernel (~1.25s): Delta/slow oscillations
    """
    
    def __init__(self, in_channels, filters_per_branch, kernels, pool_size):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Conv1d(in_channels, filters_per_branch, k, padding=k // 2)
            for k in kernels
        ])
        total_filters = filters_per_branch * len(kernels)
        self.bn = nn.BatchNorm1d(total_filters)
        self.pool = nn.MaxPool1d(pool_size)
    
    def forward(self, x):
        outputs = [branch(x) for branch in self.branches]
        x = torch.cat(outputs, dim=1)
        x = F.relu(self.bn(x))
        x = self.pool(x)
        return x


class ResidualBlock(nn.Module):
    """
    1D Residual Block with BatchNorm.
    Conv → BN → ReLU → Conv → BN → (+skip) → ReLU → MaxPool
    
    Skip connection: 1x1 conv if channel mismatch, else identity.
    """
    
    def __init__(self, in_channels, out_channels, kernel_size, pool_size):
        super().__init__()
        pad = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=pad)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.skip = nn.Identity()
        
        self.pool = nn.MaxPool1d(pool_size)
    
    def forward(self, x):
        residual = self.skip(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x = F.relu(x + residual)
        x = self.pool(x)
        return x


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) channel attention.
    Global context → channel importance weights.
    Hangi kanalların (feature map) daha önemli olduğunu öğrenir.
    """
    
    def __init__(self, channels, reduction=16):
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)
    
    def forward(self, x):
        # x: [B, C, L]
        w = x.mean(dim=2)             # GAP → [B, C]
        w = F.relu(self.fc1(w))       # Squeeze → [B, C/r]
        w = torch.sigmoid(self.fc2(w))  # Excitation → [B, C]
        return x * w.unsqueeze(2)     # Scale channels


class CNN1D(nn.Module):
    """
    1D Convolutional Neural Network for sleep stage classification.
    
    Architecture:
        Input: [batch, n_channels, samples]
        
        Conv1D(32) → ReLU → MaxPool
        Conv1D(64) → ReLU → MaxPool
        Conv1D(128) → ReLU → MaxPool
        Flatten → Dropout
        FC(256) → ReLU → Dropout
        FC(n_classes) → Softmax
        
        Output: [batch, n_classes]
    """
    
    def __init__(self, cfg=config.CNN1D_CONFIG):
        super(CNN1D, self).__init__()
        
        self.cfg = cfg
        
        # Convolutional layers
        self.conv1 = nn.Conv1d(
            in_channels=cfg['n_channels'],
            out_channels=cfg['conv1_filters'],
            kernel_size=cfg['conv1_kernel'],
            padding=cfg['conv1_kernel'] // 2  # Same padding
        )
        self.pool1 = nn.MaxPool1d(kernel_size=cfg['conv1_pool'])
        
        self.conv2 = nn.Conv1d(
            in_channels=cfg['conv1_filters'],
            out_channels=cfg['conv2_filters'],
            kernel_size=cfg['conv2_kernel'],
            padding=cfg['conv2_kernel'] // 2
        )
        self.pool2 = nn.MaxPool1d(kernel_size=cfg['conv2_pool'])
        
        self.conv3 = nn.Conv1d(
            in_channels=cfg['conv2_filters'],
            out_channels=cfg['conv3_filters'],
            kernel_size=cfg['conv3_kernel'],
            padding=cfg['conv3_kernel'] // 2
        )
        self.pool3 = nn.MaxPool1d(kernel_size=cfg['conv3_pool'])
        
        # Flatten sonrası boyutu hesapla (dinamik - gerçek forward pass ile)
        # Basit bölme işlemi same padding kullanıldığında yanlış sonuç verebilir
        flatten_size = self._calculate_flatten_size(cfg)
        
        # Fully connected layers
        self.dropout1 = nn.Dropout(cfg['dropout'])
        
        self.fc1 = nn.Linear(flatten_size, cfg['fc1_units'])
        self.dropout2 = nn.Dropout(cfg['dropout'])
        
        self.fc2 = nn.Linear(cfg['fc1_units'], cfg['n_classes'])
        
        # Model bilgilerini yazdır
        self._print_model_info(flatten_size)
    
    def _calculate_flatten_size(self, cfg):
        """
        Konvolüsyon ve pooling katmanları sonrası flatten boyutunu dinamik olarak hesapla.
        Dummy input kullanarak gerçek boyutu belirler.
        """
        # Dummy input oluştur (batch=1)
        dummy_input = torch.zeros(1, cfg['n_channels'], config.SAMPLES_PER_EPOCH)
        
        # Conv + Pool katmanlarından geçir
        x = self.conv1(dummy_input)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        # Flatten sonrası boyut
        flatten_size = x.view(1, -1).size(1)
        return flatten_size
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor [batch, n_channels, samples]
        
        Returns:
            logits: [batch, n_classes]
        """
        # Conv block 1
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        # Conv block 2
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv block 3
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        x = self.dropout1(x)
        
        # FC layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        x = self.fc2(x)
        
        return x
    
    def _print_model_info(self, flatten_size):
        """
        Model bilgilerini yazdır
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*70}")
        print(f"1D-CNN MODEL")
        print(f"{'='*70}")
        print(f"Architecture:")
        print(f"  Input: [{config.N_CHANNELS}, {config.SAMPLES_PER_EPOCH}]")
        print(f"  Conv1: {self.cfg['conv1_filters']} filters, kernel={self.cfg['conv1_kernel']}")
        print(f"  Conv2: {self.cfg['conv2_filters']} filters, kernel={self.cfg['conv2_kernel']}")
        print(f"  Conv3: {self.cfg['conv3_filters']} filters, kernel={self.cfg['conv3_kernel']}")
        print(f"  Flatten: {flatten_size} features")
        print(f"  FC1: {self.cfg['fc1_units']} units")
        print(f"  FC2 (output): {self.cfg['n_classes']} classes")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"{'='*70}\n")


def get_model(model_type='cnn1d'):
    """
    Model oluştur ve döndür.
    
    Args:
        model_type: Model tipi ('cnn1d', 'cnn1d_v3', 'cnn1d_paper', 'deepsleepnet', 'transformer')
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'cnn1d':
        model = CNN1D()
    elif model_type == 'cnn1d_v3':
        model = CNN1D_V3()
    elif model_type == 'cnn1d_paper':
        model = CNN1DPaper()
    elif model_type == 'deepsleepnet':
        # Import here to avoid circular imports
        from .deepsleepnet import get_deepsleepnet
        # Use config from config.py
        model = get_deepsleepnet(config.DEEPSLEEPNET_CONFIG)
        # Already on device from deepsleepnet, but ensure consistency
        model = model.to(config.DEVICE)
        return model
    elif model_type == 'transformer':
        # Import here to avoid circular imports
        from .sleep_transformer import get_sleep_transformer
        import torch
        # Use TRANSFORMER_CONFIG from config.py
        transformer_cfg = getattr(config, 'TRANSFORMER_CONFIG', {})
        model = get_sleep_transformer(
            n_channels=config.N_CHANNELS,
            n_classes=config.N_CLASSES,
            d_model=transformer_cfg.get('d_model', 128),
            nhead=transformer_cfg.get('nhead', 4),
            num_layers=transformer_cfg.get('num_layers', 2),
            dim_feedforward=transformer_cfg.get('dim_feedforward', 256),
            transformer_dropout=transformer_cfg.get('transformer_dropout', 0.3),
            head_dropout=transformer_cfg.get('head_dropout', 0.3),
            pos_encoding=transformer_cfg.get('pos_encoding', 'sinusoidal'),
            max_seq_len=transformer_cfg.get('max_seq_len', 25),
            activation=transformer_cfg.get('activation', 'gelu'),
            freeze_cnn=transformer_cfg.get('freeze_cnn', False),
        )
        
        # Pre-trained CNN ağırlıklarını yükle (varsa)
        pretrained_path = transformer_cfg.get('pretrained_cnn_checkpoint', None)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"\n{'='*70}")
            print(f"PRE-TRAINED CNN AĞIRLIKLARI YÜKLENİYOR")
            print(f"{'='*70}")
            print(f"  Kaynak: {pretrained_path}")
            
            ckpt = torch.load(pretrained_path, map_location='cpu', weights_only=False)
            baseline_sd = ckpt['model_state_dict']
            
            # CNN ağırlıklarını eşle: baseline key → cnn_encoder.key
            cnn_sd = {}
            skipped = []
            for k, v in baseline_sd.items():
                if 'classifier' in k:
                    skipped.append(k)
                    continue
                new_key = f'cnn_encoder.{k}'
                if new_key in model.state_dict():
                    cnn_sd[new_key] = v
            
            # Yükle
            model.load_state_dict(cnn_sd, strict=False)
            print(f"  ✓ {len(cnn_sd)} CNN parametresi yüklendi")
            print(f"  ✗ {len(skipped)} classifier parametresi atlandı")
            if 'best_val_acc' in ckpt:
                print(f"  📊 Baseline acc: {ckpt['best_val_acc']:.2f}%")
            print(f"{'='*70}\n")
        
        model = model.to(config.DEVICE)
        return model
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
    
    # Device'a taşı
    model = model.to(config.DEVICE)
    
    return model


# ============================================================================
# CNN1D V3: Multi-Scale + Residual + SE Attention + GAP
# ============================================================================

class CNN1D_V3(nn.Module):
    """
    Enhanced 1D CNN for sleep stage classification.
    
    Improvements over CNN1D:
      1. Multi-scale stem: Parallel convolutions (k=15/51/125) for
         capturing different EEG temporal patterns simultaneously
      2. Residual connections: Skip connections for better gradient flow
      3. BatchNorm: After every conv layer for stable training
      4. SE attention: Channel-wise attention (which features matter)
      5. Global Average Pooling: Replaces flatten → massive param reduction
    
    Architecture (kernel boyutları SAMPLE_RATE'e göre ölçeklenir):
        Input: [B, n_channels, samples_per_epoch]

        MultiScaleStem(→48, k=scaled) → MaxPool(4)
        ResBlock(48→96, k=scaled)     → MaxPool(4)
        ResBlock(96→128, k=scaled)    → MaxPool(4)
        SE(128, r=16) → channel attention
        GAP → [B, 128]
        Dropout → FC(128→64) → LayerNorm → ReLU → Dropout → FC(64→5)

    Parameters: ~280K+ (SAMPLE_RATE arttıkça kernel boyutları büyür)
    """
    
    def __init__(self, cfg=None):
        super().__init__()
        
        if cfg is None:
            cfg = config.CNN1D_V3_CONFIG
        self.cfg = cfg
        
        stem_total = cfg['stem_filters'] * len(cfg['stem_kernels'])
        
        # Multi-scale stem
        self.stem = MultiScaleStem(
            cfg['n_channels'], cfg['stem_filters'],
            cfg['stem_kernels'], cfg['stem_pool']
        )
        
        # Residual blocks
        self.block1 = ResidualBlock(
            stem_total, cfg['block1_filters'],
            cfg['block1_kernel'], cfg['block1_pool']
        )
        
        self.block2 = ResidualBlock(
            cfg['block1_filters'], cfg['block2_filters'],
            cfg['block2_kernel'], cfg['block2_pool']
        )
        
        # Channel attention
        self.se = SEBlock(cfg['block2_filters'], cfg['se_reduction'])
        
        # Classifier (GAP output → FC)
        self.classifier = nn.Sequential(
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['block2_filters'], cfg['fc1_units']),
            nn.LayerNorm(cfg['fc1_units']),
            nn.ReLU(),
            nn.Dropout(cfg['dropout']),
            nn.Linear(cfg['fc1_units'], cfg['n_classes'])
        )
        
        self._print_model_info()
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, n_channels, samples]
        Returns:
            logits: [batch, n_classes]
        """
        x = self.stem(x)       # Multi-scale feature extraction
        x = self.block1(x)     # Residual block 1
        x = self.block2(x)     # Residual block 2
        x = self.se(x)         # Channel attention
        x = x.mean(dim=2)      # Global Average Pooling
        x = self.classifier(x) # Classification
        return x
    
    def _print_model_info(self):
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        cfg = self.cfg
        stem_total = cfg['stem_filters'] * len(cfg['stem_kernels'])
        
        print(f"\n{'='*70}")
        print(f"CNN1D V3 MODEL (Multi-Scale + Residual + SE + GAP)")
        print(f"{'='*70}")
        print(f"Architecture:")
        print(f"  Input: [{config.N_CHANNELS}, {config.SAMPLES_PER_EPOCH}]")
        print(f"  Stem: {len(cfg['stem_kernels'])}x Conv1D({cfg['stem_filters']}) "
              f"k={cfg['stem_kernels']} → concat {stem_total}ch")
        print(f"  ResBlock1: {stem_total}→{cfg['block1_filters']}, k={cfg['block1_kernel']}")
        print(f"  ResBlock2: {cfg['block1_filters']}→{cfg['block2_filters']}, k={cfg['block2_kernel']}")
        print(f"  SE: {cfg['block2_filters']}ch, reduction={cfg['se_reduction']}")
        print(f"  GAP → {cfg['block2_filters']}")
        print(f"  FC1: {cfg['fc1_units']} units + LayerNorm")
        print(f"  FC2 (output): {cfg['n_classes']} classes")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Reduction vs CNN1D: {1679781/total_params:.1f}x fewer params")
        print(f"{'='*70}\n")


class CNN1DPaper(nn.Module):
    """
    Paper replication: CNN architecture from "A CNN-LSTM model for sleep stage classification"
    
    Architecture (bottom to top in paper figure):
        Input: [batch, n_channels, samples] (3 channels, 2800 samples @ 93.3Hz or 3000 @ 100Hz)
        
        BN → Dropout(0.2) → Conv1D(128, k=50, s=5) 
        BN → Dropout(0.2) → Conv1D(256, k=5, s=1) → MaxPool(2)
        BN → Dropout(0.2) → Conv1D(300, k=5, s=2) → MaxPool(2)
        
        Flatten
        Dense(1500) → BN → Dropout(0.5)
        Dense(1500) → BN → Dropout(0.5)
        Dense(5) → Softmax
        
        Output: [batch, n_classes]
    """
    
    def __init__(self, cfg=None):
        super(CNN1DPaper, self).__init__()
        
        if cfg is None:
            cfg = config.CNN1D_CONFIG_PAPER
        self.cfg = cfg
        
        # ============================================================
        # Conv Block 1: 128 filters, kernel=50, stride=5
        # ============================================================
        self.bn1 = nn.BatchNorm1d(cfg['n_channels'])
        self.dropout1 = nn.Dropout(cfg['dropout_conv'])
        self.conv1 = nn.Conv1d(
            in_channels=cfg['n_channels'],
            out_channels=cfg['conv1_filters'],  # 128
            kernel_size=cfg['conv1_kernel'],     # 50
            stride=cfg['conv1_stride'],          # 5
            padding=0  # Paper uses no padding, stride reduces size
        )
        # No pooling after first conv in paper
        
        # ============================================================
        # Conv Block 2: 256 filters, kernel=5, stride=1 + MaxPool(2)
        # ============================================================
        self.bn2 = nn.BatchNorm1d(cfg['conv1_filters'])
        self.dropout2 = nn.Dropout(cfg['dropout_conv'])
        self.conv2 = nn.Conv1d(
            in_channels=cfg['conv1_filters'],
            out_channels=cfg['conv2_filters'],  # 256
            kernel_size=cfg['conv2_kernel'],     # 5
            stride=cfg['conv2_stride'],          # 1
            padding=2  # Same padding for stride=1
        )
        self.pool2 = nn.MaxPool1d(kernel_size=cfg['conv2_pool'])  # 2
        
        # ============================================================
        # Conv Block 3: 300 filters, kernel=5, stride=2 + MaxPool(2)
        # ============================================================
        self.bn3 = nn.BatchNorm1d(cfg['conv2_filters'])
        self.dropout3 = nn.Dropout(cfg['dropout_conv'])
        self.conv3 = nn.Conv1d(
            in_channels=cfg['conv2_filters'],
            out_channels=cfg['conv3_filters'],  # 300
            kernel_size=cfg['conv3_kernel'],     # 5
            stride=cfg['conv3_stride'],          # 2
            padding=2
        )
        self.pool3 = nn.MaxPool1d(kernel_size=cfg['conv3_pool'])  # 2
        
        # ============================================================
        # Calculate flatten size dynamically
        # ============================================================
        flatten_size = self._calculate_flatten_size(cfg)
        
        # ============================================================
        # Dense Block 1: 1500 units + BN + Dropout(0.5)
        # ============================================================
        self.fc1 = nn.Linear(flatten_size, cfg['fc1_units'])  # 1500
        self.bn_fc1 = nn.BatchNorm1d(cfg['fc1_units'])
        self.dropout_fc1 = nn.Dropout(cfg['dropout_dense'])
        
        # ============================================================
        # Dense Block 2: 1500 units + BN + Dropout(0.5)
        # ============================================================
        self.fc2 = nn.Linear(cfg['fc1_units'], cfg['fc2_units'])  # 1500
        self.bn_fc2 = nn.BatchNorm1d(cfg['fc2_units'])
        self.dropout_fc2 = nn.Dropout(cfg['dropout_dense'])
        
        # ============================================================
        # Output: 5 classes
        # ============================================================
        self.fc_out = nn.Linear(cfg['fc2_units'], cfg['n_classes'])
        
        # Print model info
        self._print_model_info(flatten_size)
    
    def _calculate_flatten_size(self, cfg):
        """Dynamically calculate flatten size using dummy forward pass."""
        dummy_input = torch.zeros(1, cfg['n_channels'], config.SAMPLES_PER_EPOCH)
        
        # Conv Block 1
        x = self.bn1(dummy_input)
        x = self.conv1(x)
        
        # Conv Block 2
        x = self.bn2(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        # Conv Block 3
        x = self.bn3(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        return x.view(1, -1).size(1)
    
    def forward(self, x):
        """Forward pass."""
        # Conv Block 1
        x = self.bn1(x)
        x = self.dropout1(x)
        x = self.conv1(x)
        x = F.relu(x)
        
        # Conv Block 2 + Pool
        x = self.bn2(x)
        x = self.dropout2(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        # Conv Block 3 + Pool
        x = self.bn3(x)
        x = self.dropout3(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool3(x)
        
        # Flatten
        x = torch.flatten(x, start_dim=1)
        
        # Dense Block 1
        x = self.fc1(x)
        x = F.relu(x)
        x = self.bn_fc1(x)
        x = self.dropout_fc1(x)
        
        # Dense Block 2
        x = self.fc2(x)
        x = F.relu(x)
        x = self.bn_fc2(x)
        x = self.dropout_fc2(x)
        
        # Output
        x = self.fc_out(x)
        
        return x
    
    def _print_model_info(self, flatten_size):
        """Print model summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        print(f"\n{'='*70}")
        print(f"CNN1D PAPER MODEL")
        print(f"{'='*70}")
        print(f"Architecture (Paper replication):")
        print(f"  Input: [{config.N_CHANNELS}, {config.SAMPLES_PER_EPOCH}]")
        print(f"  Conv1: {self.cfg['conv1_filters']} filters, k={self.cfg['conv1_kernel']}, s={self.cfg['conv1_stride']}")
        print(f"  Conv2: {self.cfg['conv2_filters']} filters, k={self.cfg['conv2_kernel']}, s={self.cfg['conv2_stride']} + Pool(2)")
        print(f"  Conv3: {self.cfg['conv3_filters']} filters, k={self.cfg['conv3_kernel']}, s={self.cfg['conv3_stride']} + Pool(2)")
        print(f"  Flatten: {flatten_size} features")
        print(f"  FC1: {self.cfg['fc1_units']} units + BN + Dropout({self.cfg['dropout_dense']})")
        print(f"  FC2: {self.cfg['fc2_units']} units + BN + Dropout({self.cfg['dropout_dense']})")
        print(f"  Output: {self.cfg['n_classes']} classes")
        print(f"\nParameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"{'='*70}\n")





def count_parameters(model):
    """
    Model parametrelerini say
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total, trainable


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Model oluştur
    model = get_model('cnn1d')
    
    # Test input
    batch_size = 8
    test_input = torch.randn(batch_size, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
    test_input = test_input.to(config.DEVICE)
    
    print(f"Test input shape: {test_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output = model(test_input)
    
    print(f"Output shape: {output.shape}")
    print(f"Output (logits): {output[0]}")
    
    # Softmax uygula
    probs = F.softmax(output, dim=1)
    print(f"\nProbabilities: {probs[0]}")
    print(f"Sum: {probs[0].sum()}")
    
    # Predicted class
    predicted = torch.argmax(probs, dim=1)
    print(f"\nPredicted classes: {predicted}")
    
    # Model özeti
    total, trainable = count_parameters(model)
    print(f"\n✓ Model test başarılı!")
    print(f"  Total parameters: {total:,}")
    print(f"  Trainable parameters: {trainable:,}")