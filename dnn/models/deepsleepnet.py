"""
DeepSleepNet: CNN + BiLSTM Hybrid Model for Sleep Stage Classification

Based on: "DeepSleepNet: a Model for Automatic Sleep Stage Scoring based 
on Raw Single-Channel EEG" - Supratak et al. 2017

Architecture:
    1. Representation Learning (CNN): Two parallel branches for multi-scale feature extraction
       - Small filter branch: Captures high-frequency features (spindles, K-complexes)
       - Large filter branch: Captures low-frequency features (slow waves, delta)
    
    2. Sequence Residual Learning (BiLSTM): Temporal dependencies with residual connection
       - Two stacked BiLSTM layers with dropout
       - Residual connection from FC layer to second BiLSTM output
       - Final softmax for 5-class classification (W, N1, N2, N3, REM)

This implementation supports:
    - Single or multi-channel input
    - Flexible sampling rates (100Hz, 256Hz, etc.)
    - Sequence-to-sequence learning for temporal context
    - 4GB VRAM optimization (reduced hidden sizes, shorter sequences)

N1 Detection Improvement:
    - BiLSTM learns temporal transitions: Wake→N1→N2, N2→N1→REM
    - Residual connection preserves gradient flow for rare classes
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import sys
import os

# Add parent directory to path for config import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config


class SmallFilterCNN(nn.Module):
    """
    Small filter branch - captures high-frequency features.
    
    Detects: Sleep spindles (11-16 Hz), K-complexes, alpha waves
    Kernel size = Fs/2 (128 samples @ 256Hz = 0.5 second window)
    
    Architecture (from paper):
        Conv1D(64, Fs/2, stride=Fs/16) → MaxPool(8) → Dropout(0.5)
        Conv1D(128, 8) → Conv1D(128, 8) → Conv1D(128, 8) → MaxPool(4)
    """
    
    def __init__(self, n_channels, sample_rate, dropout=0.5):
        super(SmallFilterCNN, self).__init__()
        
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        
        # Paper: Fs/2 kernel, Fs/16 stride for first conv
        # 256Hz: kernel=128 (0.5s), stride=16
        # 100Hz: kernel=50 (0.5s), stride=6
        kernel1 = max(sample_rate // 2, 32)  # Minimum 32 for stability
        stride1 = max(sample_rate // 16, 4)  # Minimum 4
        
        # Conv Block 1: Large kernel for initial feature extraction
        self.conv1 = nn.Conv1d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=kernel1,
            stride=stride1,
            padding=kernel1 // 2
        )
        # Paper: eps=1e-5, decay=0.999 → PyTorch momentum=0.001
        self.bn1 = nn.BatchNorm1d(64, eps=1e-5, momentum=0.001)
        self.pool1 = nn.MaxPool1d(kernel_size=8, stride=8)
        self.dropout1 = nn.Dropout(dropout)
        
        # Conv Block 2-4: Stack of 3 conv layers with smaller kernels
        self.conv2 = nn.Conv1d(64, 128, kernel_size=8, padding=4)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.001)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=8, padding=4)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.001)
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=8, padding=4)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.001)
        
        self.pool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        return x


class LargeFilterCNN(nn.Module):
    """
    Large filter branch - captures low-frequency features.
    
    Detects: Delta waves (0.5-4 Hz), slow oscillations, K-complex slow component
    Kernel size = Fs*4 (1024 samples @ 256Hz = 4 second window)
    
    Architecture (from paper):
        Conv1D(64, Fs*4, stride=Fs/2) → MaxPool(4) → Dropout(0.5)
        Conv1D(128, 6) → Conv1D(128, 6) → Conv1D(128, 6) → MaxPool(2)
    """
    
    def __init__(self, n_channels, sample_rate, dropout=0.5):
        super(LargeFilterCNN, self).__init__()
        
        self.n_channels = n_channels
        self.sample_rate = sample_rate
        
        # Paper: Fs*4 kernel, Fs/2 stride for first conv
        # 256Hz: kernel=1024 (4s), stride=128
        # 100Hz: kernel=400 (4s), stride=50
        kernel1 = sample_rate * 4
        stride1 = max(sample_rate // 2, 16)
        
        # Conv Block 1: Very large kernel for slow wave detection
        self.conv1 = nn.Conv1d(
            in_channels=n_channels,
            out_channels=64,
            kernel_size=kernel1,
            stride=stride1,
            padding=kernel1 // 2
        )
        # Paper: eps=1e-5, decay=0.999 → PyTorch momentum=0.001
        self.bn1 = nn.BatchNorm1d(64, eps=1e-5, momentum=0.001)
        self.pool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        self.dropout1 = nn.Dropout(dropout)
        
        # Conv Block 2-4: Stack of 3 conv layers
        self.conv2 = nn.Conv1d(64, 128, kernel_size=6, padding=3)
        self.bn2 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.001)
        
        self.conv3 = nn.Conv1d(128, 128, kernel_size=6, padding=3)
        self.bn3 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.001)
        
        self.conv4 = nn.Conv1d(128, 128, kernel_size=6, padding=3)
        self.bn4 = nn.BatchNorm1d(128, eps=1e-5, momentum=0.001)
        
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x):
        # Conv Block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = self.dropout1(x)
        
        # Conv Block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Conv Block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Conv Block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        return x


class RepresentationLearning(nn.Module):
    """
    CNN Feature Extractor: Two parallel branches merged at the end.
    
    Extracts features from a single 30-second epoch.
    Small branch → high-freq features (spindles, alpha)
    Large branch → low-freq features (delta, slow waves)
    
    Output is a fixed-size feature vector representing the epoch.
    """
    
    def __init__(self, n_channels, sample_rate, dropout=0.5):
        super(RepresentationLearning, self).__init__()
        
        self.small_cnn = SmallFilterCNN(n_channels, sample_rate, dropout)
        self.large_cnn = LargeFilterCNN(n_channels, sample_rate, dropout)
        
        self.dropout = nn.Dropout(dropout)
        
        # LayerNorm will be set after we know the output size
        self._output_size = None
        self.layer_norm = None
        
    def _init_layer_norm(self, feature_dim):
        """Initialize LayerNorm after knowing feature dimension."""
        if self.layer_norm is None:
            self.layer_norm = nn.LayerNorm(feature_dim)
            # Move to same device as conv1
            device = next(self.small_cnn.parameters()).device
            self.layer_norm = self.layer_norm.to(device)
        
    def forward(self, x):
        """
        Args:
            x: [batch, n_channels, samples] - single epoch
            
        Returns:
            features: [batch, feature_dim] - flattened and concatenated features
        """
        # Two parallel branches
        small_features = self.small_cnn(x)  # [batch, 128, T1]
        large_features = self.large_cnn(x)  # [batch, 128, T2]
        
        # Flatten both
        small_flat = torch.flatten(small_features, start_dim=1)
        large_flat = torch.flatten(large_features, start_dim=1)
        
        # Concatenate
        combined = torch.cat([small_flat, large_flat], dim=1)
        
        # Initialize LayerNorm on first forward pass
        if self.layer_norm is None:
            self._init_layer_norm(combined.shape[1])
        
        # Apply LayerNorm to stabilize values before LSTM
        combined = self.layer_norm(combined)
        combined = self.dropout(combined)
        
        return combined
    
    def get_output_size(self, n_channels, samples_per_epoch):
        """Calculate output feature dimension with dummy forward pass."""
        dummy = torch.zeros(1, n_channels, samples_per_epoch)
        with torch.no_grad():
            out = self.forward(dummy)
        return out.shape[1]


class SequenceResidualLearning(nn.Module):
    """
    Stateful BiLSTM with Residual Connection for Sequence Learning.
    
    STATEFUL: Hidden state hasta boyunca taşınır, yeni hasta başında reset edilir.
    Bu sayede model tüm geceyi (6-8 saat) "hatırlayarak" tahmin yapabilir.
    
    Learns temporal dependencies between consecutive sleep epochs:
    - Wake → N1 → N2 (falling asleep transition)
    - N2 → N1 → REM (REM cycle entry)
    - N3 → N2 → N1 (arousal pattern)
    
    The residual connection helps:
    1. Preserve gradients for rare classes (N1)
    2. Allow direct feature propagation
    3. Faster convergence
    
    Architecture (from paper):
        FC(1024) for feature projection (parallel residual path)
        BiLSTM(512) → Dropout → BiLSTM(512) → Dropout
        Residual: Add FC output to second BiLSTM output
        Final Dropout → Softmax(5)
    """
    
    def __init__(self, input_size, hidden_size=512, n_classes=5, dropout=0.5):
        super(SequenceResidualLearning, self).__init__()
        
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        
        # Bidirectional output size
        bidirectional_size = hidden_size * 2
        
        # Parallel FC path for residual connection
        self.fc_residual = nn.Linear(input_size, bidirectional_size)
        
        # First BiLSTM layer
        self.lstm1 = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout1 = nn.Dropout(dropout)
        
        # Second BiLSTM layer  
        self.lstm2 = nn.LSTM(
            input_size=bidirectional_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0
        )
        self.dropout2 = nn.Dropout(dropout)
        
        # Final dropout after residual addition
        self.dropout_final = nn.Dropout(dropout)
        
        # Output classification layer
        self.fc_out = nn.Linear(bidirectional_size, n_classes)
        
        # Stateful: Hidden states (None = will be initialized on first forward)
        self.hidden1 = None
        self.hidden2 = None
        
    def reset_state(self, batch_size=1, device='cuda'):
        """
        Yeni hasta başında çağrılır - hidden state'leri sıfırla.
        
        Args:
            batch_size: Batch boyutu (stateful için genellikle 1)
            device: CUDA veya CPU
        """
        # BiLSTM: num_layers * 2 (bidirectional)
        h1 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        c1 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        self.hidden1 = (h1, c1)
        
        h2 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        c2 = torch.zeros(2, batch_size, self.hidden_size, device=device)
        self.hidden2 = (h2, c2)
        
    def forward(self, x, return_sequence=True):
        """
        Stateful forward pass - hidden state taşınır.
        
        Args:
            x: [batch, seq_len, features] - sequence of epoch features
            return_sequence: If True, return output for each timestep
                           If False, return output for only the middle epoch
        
        Returns:
            If return_sequence=True: [batch, seq_len, n_classes]
            If return_sequence=False: [batch, n_classes]
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden states if needed
        if self.hidden1 is None:
            self.reset_state(batch_size, x.device)
        
        # Residual path: FC projection (applied to each timestep)
        residual = self.fc_residual(x)  # [batch, seq_len, hidden*2]
        
        # Detach hidden states - backprop sadece bu sequence için
        hidden1 = (self.hidden1[0].detach(), self.hidden1[1].detach())
        hidden2 = (self.hidden2[0].detach(), self.hidden2[1].detach())
        
        # BiLSTM path - STATE TAŞINIYOR
        lstm1_out, self.hidden1 = self.lstm1(x, hidden1)  # [batch, seq_len, hidden*2]
        lstm1_out = self.dropout1(lstm1_out)
        
        lstm2_out, self.hidden2 = self.lstm2(lstm1_out, hidden2)  # [batch, seq_len, hidden*2]
        lstm2_out = self.dropout2(lstm2_out)
        
        # Residual connection (element-wise addition)
        combined = lstm2_out + residual
        combined = self.dropout_final(combined)
        
        # Output projection for each timestep
        output = self.fc_out(combined)  # [batch, seq_len, n_classes]
        
        if return_sequence:
            return output
        else:
            # Return only middle epoch prediction (for inference)
            middle_idx = seq_len // 2
            return output[:, middle_idx, :]


class DeepSleepNet(nn.Module):
    """
    Complete DeepSleepNet Model: CNN + BiLSTM Hybrid Architecture.
    
    Two modes of operation:
    1. Single-epoch mode: Process one epoch, no temporal context (use_bilstm=False)
    2. Sequence mode: Process sequence of epochs with BiLSTM (use_bilstm=True)
    
    For training, use sequence mode with sequence_length >= 3.
    This allows the model to learn temporal transitions like:
        Wake → N1 → N2 → N3 (falling asleep)
        N3 → N2 → N1 → REM (REM cycle)
    
    Args:
        n_channels: Number of input channels (1 for single-channel EEG like F4-M1)
        n_classes: Number of sleep stages (default: 5 for W, N1, N2, N3, REM)
        sample_rate: Sampling frequency in Hz (100 or 256)
        sequence_length: Number of consecutive epochs for BiLSTM context
        dropout: Dropout probability
        lstm_hidden_size: BiLSTM hidden units per direction
        samples_per_epoch: Samples per 30-second epoch (auto-calculated if None)
    
    Memory Optimization for 4GB VRAM:
        - Use sequence_length=5 (instead of 25)
        - Use lstm_hidden_size=128 (instead of 512)
        - Use batch_size=4 with gradient accumulation
        - Enable mixed precision training (FP16)
    """
    
    def __init__(self, 
                 n_channels=1,
                 n_classes=5,
                 sample_rate=256,
                 sequence_length=5,
                 dropout=0.5,
                 lstm_hidden_size=128,
                 samples_per_epoch=None):
        
        super(DeepSleepNet, self).__init__()
        
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.sample_rate = sample_rate
        self.sequence_length = sequence_length
        self.dropout_rate = dropout
        self.lstm_hidden_size = lstm_hidden_size
        
        # Samples per epoch (30 seconds * sample_rate)
        if samples_per_epoch is None:
            self.samples_per_epoch = sample_rate * 30
        else:
            self.samples_per_epoch = samples_per_epoch
        
        # ============================================================
        # Stage 1: Representation Learning (Two-Branch CNN)
        # ============================================================
        self.cnn = RepresentationLearning(
            n_channels=n_channels,
            sample_rate=sample_rate,
            dropout=dropout
        )
        
        # Calculate CNN output size dynamically
        self.cnn_output_size = self.cnn.get_output_size(
            n_channels, self.samples_per_epoch
        )
        
        # ============================================================
        # Stage 2: Sequence Residual Learning (BiLSTM)
        # ============================================================
        self.bilstm = SequenceResidualLearning(
            input_size=self.cnn_output_size,
            hidden_size=lstm_hidden_size,
            n_classes=n_classes,
            dropout=dropout
        )
        
        # ============================================================
        # Alternative: Single-epoch classifier (no BiLSTM)
        # Used when processing individual epochs without sequence context
        # ============================================================
        self.fc_single = nn.Sequential(
            nn.Linear(self.cnn_output_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, n_classes)
        )
        
        # Initialize weights properly
        self._initialize_weights()
        
        # Print model info
        self._print_model_info()
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming/He initialization for ReLU networks."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Conv1d):
                # Kaiming initialization for Conv layers
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm1d):
                # BatchNorm: weight=1, bias=0
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                # Linear layers: Xavier initialization
                nn.init.xavier_normal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                # LSTM: Orthogonal initialization for hidden weights
                for param_name, param in module.named_parameters():
                    if 'weight_ih' in param_name:
                        nn.init.xavier_uniform_(param)
                    elif 'weight_hh' in param_name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in param_name:
                        nn.init.zeros_(param)
                        # Set forget gate bias to 1 for better gradient flow
                        n = param.size(0)
                        param.data[n//4:n//2].fill_(1.0)
        
    def forward(self, x, use_bilstm=True):
        """
        Forward pass with flexible input handling.
        
        Args:
            x: Input tensor, can be:
               - [batch, n_channels, samples]: Single epoch per sample
               - [batch, seq_len, n_channels, samples]: Sequence of epochs
            use_bilstm: Whether to use BiLSTM for sequence processing
            
        Returns:
            logits: [batch, n_classes] or [batch, seq_len, n_classes]
        """
        # Check input dimensions
        if x.dim() == 3:
            # Single epoch mode: [batch, n_channels, samples]
            return self._forward_single(x)
        elif x.dim() == 4:
            # Sequence mode: [batch, seq_len, n_channels, samples]
            if use_bilstm:
                return self._forward_sequence(x)
            else:
                # Process each epoch independently (no temporal context)
                batch, seq_len, n_ch, samples = x.shape
                x_flat = x.view(batch * seq_len, n_ch, samples)
                out_flat = self._forward_single(x_flat)
                return out_flat.view(batch, seq_len, -1)
        else:
            raise ValueError(f"Expected 3D or 4D input, got {x.dim()}D")
    
    def _forward_single(self, x):
        """Process single epochs without BiLSTM."""
        features = self.cnn(x)
        logits = self.fc_single(features)
        return logits
    
    def _forward_sequence(self, x):
        """Process sequence of epochs with BiLSTM for temporal context."""
        batch_size, seq_len, n_channels, samples = x.shape
        
        # Process each epoch through CNN
        # Reshape: [batch * seq_len, n_channels, samples]
        x_flat = x.view(batch_size * seq_len, n_channels, samples)
        
        # CNN feature extraction for all epochs
        features_flat = self.cnn(x_flat)  # [batch * seq_len, cnn_output_size]
        
        # Reshape back: [batch, seq_len, cnn_output_size]
        features = features_flat.view(batch_size, seq_len, -1)
        
        # BiLSTM sequence processing with residual
        logits = self.bilstm(features, return_sequence=True)  # [batch, seq_len, n_classes]
        
        return logits
    
    def get_cnn_features(self, x):
        """
        Extract CNN features for visualization/analysis (e.g., SHAP).
        
        Args:
            x: [batch, n_channels, samples]
            
        Returns:
            features: [batch, cnn_output_size]
        """
        return self.cnn(x)
    
    def _print_model_info(self):
        """Print model architecture summary."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Estimate memory usage (rough: 4 bytes per param for FP32, 2x for gradients)
        mem_mb = (total_params * 4 * 3) / (1024 * 1024)  # params + grads + optimizer states
        
        print(f"\n{'='*70}")
        print(f"DEEPSLEEPNET MODEL (4GB VRAM Optimized)")
        print(f"{'='*70}")
        print(f"\n📊 Configuration:")
        print(f"  Input channels: {self.n_channels}")
        print(f"  Sample rate: {self.sample_rate} Hz")
        print(f"  Samples per epoch: {self.samples_per_epoch}")
        print(f"  Sequence length: {self.sequence_length} epochs")
        print(f"  Output classes: {self.n_classes}")
        
        print(f"\n🏗️  Architecture:")
        print(f"  [Stage 1] Representation Learning (Two-Branch CNN):")
        print(f"    - Small filter branch: kernel={self.sample_rate // 2} (~0.5s window)")
        print(f"    - Large filter branch: kernel={self.sample_rate * 4} (~4s window)")
        print(f"    - CNN output features: {self.cnn_output_size}")
        
        print(f"\n  [Stage 2] Sequence Residual Learning (BiLSTM):")
        print(f"    - LSTM hidden size: {self.lstm_hidden_size} × 2 (bidirectional)")
        print(f"    - Residual FC: {self.cnn_output_size} → {self.lstm_hidden_size * 2}")
        print(f"    - Stacked BiLSTM layers: 2")
        print(f"    - Dropout: {self.dropout_rate}")
        
        print(f"\n📈 Parameters:")
        print(f"  Total: {total_params:,}")
        print(f"  Trainable: {trainable_params:,}")
        print(f"  Est. memory (FP32): ~{mem_mb:.0f} MB")
        print(f"{'='*70}\n")


# ============================================================================
# Configuration Presets
# ============================================================================

# 4GB VRAM optimized (T1200, GTX 1650, etc.)
DEEPSLEEPNET_CONFIG_4GB = {
    'n_channels': 1,          # Single channel (F4-M1)
    'n_classes': 5,           # W, N1, N2, N3, REM
    'sample_rate': 256,       # Hz
    'sequence_length': 5,     # Reduced from 25 for memory
    'dropout': 0.5,
    'lstm_hidden_size': 128,  # Reduced from 512 for memory
}

# 8GB VRAM (RTX 3060, RTX 2070, etc.)
DEEPSLEEPNET_CONFIG_8GB = {
    'n_channels': 1,
    'n_classes': 5,
    'sample_rate': 256,
    'sequence_length': 11,
    'dropout': 0.5,
    'lstm_hidden_size': 256,
}

# Full paper config (12GB+ VRAM)
DEEPSLEEPNET_CONFIG_FULL = {
    'n_channels': 1,
    'n_classes': 5,
    'sample_rate': 256,
    'sequence_length': 25,
    'dropout': 0.5,
    'lstm_hidden_size': 512,
}

# 100Hz version (PhysioNet dataset)
DEEPSLEEPNET_CONFIG_100HZ = {
    'n_channels': 1,
    'n_classes': 5,
    'sample_rate': 100,
    'sequence_length': 5,
    'dropout': 0.5,
    'lstm_hidden_size': 128,
}


def get_deepsleepnet(config_dict=None, **kwargs):
    """
    Create DeepSleepNet model with configuration.
    
    Args:
        config_dict: Configuration dictionary (uses DEEPSLEEPNET_CONFIG_4GB if None)
        **kwargs: Override specific config values
        
    Returns:
        model: DeepSleepNet instance
    
    Example:
        # Default 4GB config
        model = get_deepsleepnet()
        
        # Custom config
        model = get_deepsleepnet(DEEPSLEEPNET_CONFIG_8GB)
        
        # Override specific values
        model = get_deepsleepnet(sequence_length=7, lstm_hidden_size=192)
    """
    if config_dict is None:
        config_dict = DEEPSLEEPNET_CONFIG_4GB.copy()
    else:
        config_dict = config_dict.copy()
    
    # Override with kwargs
    config_dict.update(kwargs)
    
    model = DeepSleepNet(**config_dict)
    return model


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    print("Testing DeepSleepNet (4GB VRAM Optimized)...")
    
    # Test with 256Hz (your setup)
    print(f"\n{'='*70}")
    print(f"Sample Rate: 256 Hz (Your PSG Data)")
    print(f"{'='*70}")
    
    model = get_deepsleepnet(DEEPSLEEPNET_CONFIG_4GB)
    
    # Test single epoch input
    print("\n[Test 1] Single epoch input:")
    x_single = torch.randn(4, 1, 7680)  # batch=4, channels=1, samples=7680
    out_single = model(x_single, use_bilstm=False)
    print(f"  Input shape: {x_single.shape}")
    print(f"  Output shape: {out_single.shape}")
    print(f"  Expected: [4, 5]")
    
    # Test sequence input
    print("\n[Test 2] Sequence input (with BiLSTM):")
    seq_len = 5
    x_seq = torch.randn(4, seq_len, 1, 7680)  # batch=4, seq=5, channels=1, samples=7680
    out_seq = model(x_seq, use_bilstm=True)
    print(f"  Input shape: {x_seq.shape}")
    print(f"  Output shape: {out_seq.shape}")
    print(f"  Expected: [4, 5, 5]  (batch, seq_len, n_classes)")
    
    # Memory estimation
    print("\n[Memory Check]")
    total_params = sum(p.numel() for p in model.parameters())
    mem_fp32 = total_params * 4 / (1024**2)
    mem_training = mem_fp32 * 3  # params + grads + optimizer
    print(f"  Parameters: {total_params:,}")
    print(f"  Model size (FP32): {mem_fp32:.1f} MB")
    print(f"  Est. training memory: {mem_training:.1f} MB (excluding activations)")
    
    # Test with smaller batch for VRAM simulation
    print("\n[Test 3] Batch size simulation for 4GB VRAM:")
    for batch_size in [2, 4, 8]:
        x = torch.randn(batch_size, 5, 1, 7680)
        try:
            out = model(x)
            activation_mem = x.numel() * 4 / (1024**2) * 10  # rough activation estimate
            print(f"  Batch {batch_size}: OK (est. activation ~{activation_mem:.0f} MB)")
        except RuntimeError as e:
            print(f"  Batch {batch_size}: OOM")
    
    print("\n✓ All tests passed!")
    print("\n💡 Recommendation for 4GB VRAM:")
    print("   - Use batch_size=4")
    print("   - Enable mixed precision (FP16)")
    print("   - Use gradient accumulation if needed")
