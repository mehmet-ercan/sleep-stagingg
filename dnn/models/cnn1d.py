"""
Sleep Stage Classification - 1D CNN Model
Ham EEG sinyallerinden öğrenen basit CNN mimarisi
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import config


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
        model_type: Model tipi ('cnn1d', 'cnn1d_paper', 'deepsleepnet')
    
    Returns:
        model: PyTorch model
    """
    if model_type == 'cnn1d':
        model = CNN1D()
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
    else:
        raise ValueError(f"Desteklenmeyen model tipi: {model_type}")
    
    # Device'a taşı
    model = model.to(config.DEVICE)
    
    return model


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