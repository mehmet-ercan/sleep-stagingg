"""
Sleep Stage Classification - Configuration File
Tüm hyperparameter'lar ve ayarlar burada
"""

import torch
import os

# ============================================================================
# Dataset Selection
# ============================================================================
USE_PHYSIONET = True  # True: PhysioNet dataset, False: Original dataset

# ============================================================================
# MongoDB Settings
# ============================================================================
MONGO_URI = "mongodb://localhost:27017/"

# Veritabanı isimleri
DB_NAME_RAW = "psg_data"              # Raw kanallar (45 kanal, filtresiz)
DB_NAME_FILTERED = "psg_data_filtered" # Differential kanallar (9 kanal, filtreli)

# Eğitim için kullanılacak veritabanı (genellikle filtered)
DB_NAME = DB_NAME_FILTERED

if USE_PHYSIONET:
    DB_NAME = "physionet_sleep"
    DB_NAME_RAW = "physionet_sleep"  # PhysioNet için aynı DB

# ============================================================================
# Data Quality - Epoch Trimming
# ============================================================================
# Baştan ve sondan kırpılacak epoch sayısı (uyanıklık geçiş bölgelerini temizler)
TRIM_EPOCHS_START = 5   # İlk 5 epoch kırpılır (150s)
TRIM_EPOCHS_END = 5     # Son 5 epoch kırpılır (150s)

# ============================================================================
# Data Settings
# ============================================================================
# Kullanılacak kanallar - DIFFERENTIAL CHANNELS (AASM Standard)
# Contralateral mastoid referencing for EEG/EOG, bipolar for EMG
# Added frontal channels (F3-M2, F4-M1) for better N1/REM detection
SELECTED_CHANNELS = ['C3-M2', 'C4-M1', 'F3-M2', 'F4-M1', 'E1-M2', 'CHIN1-CHIN2']
N_CHANNELS = len(SELECTED_CHANNELS)
SAMPLE_RATE = 256  # Hz (NATIVE frequency - no downsampling for better resolution)

# ============================================================================
# Single Channel Mode (for DeepSleepNet)
# ============================================================================
# True: Tek kanal kullan (F3-M2), False: Tüm kanalları kullan
SINGLE_CHANNEL_MODE = False
SINGLE_CHANNEL_NAME = 'F3-M2'  # Frontal EEG - good for N1/REM detection

# Aktif kanal ayarları (MODEL_TYPE'a göre otomatik belirlenir)
if SINGLE_CHANNEL_MODE:
    ACTIVE_CHANNELS = [SINGLE_CHANNEL_NAME]
    N_CHANNELS_ACTIVE = 1
else:
    ACTIVE_CHANNELS = SELECTED_CHANNELS
    N_CHANNELS_ACTIVE = N_CHANNELS

# Epoch parametreleri (USE_PHYSIONET'ten önce tanımlanmalı)
EPOCH_DURATION = 30  # saniye

if USE_PHYSIONET:
    # SELECTED_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
    SELECTED_CHANNELS = ['EEG Pz-Oz']
    N_CHANNELS = 1
    N_CHANNELS_ACTIVE = 1  # PhysioNet için aktif kanal sayısı da güncellenmeli
    ACTIVE_CHANNELS = SELECTED_CHANNELS
    SAMPLE_RATE = 100  # Hz

# SAMPLES_PER_EPOCH hesapla (USE_PHYSIONET ayarından sonra)
SAMPLES_PER_EPOCH = SAMPLE_RATE * EPOCH_DURATION

# Balanced dataset (overfitting'i azaltmak için)
# BALANCE_STRATEGY = 'none' ile class_weights birlikte kullanılır
BALANCE_STRATEGY = 'none'  # 'undersample', 'oversample', 'none'

# Sınıf isimleri
CLASS_NAMES = ['W', 'N1', 'N2', 'N3', 'REM']
N_CLASSES = len(CLASS_NAMES)

# Sınıfları sayısal değerlere map'le
CLASS_TO_IDX = {
    'W': 0,
    'N1': 1,
    'N2': 2,
    'N3': 3,
    'REM': 4
}

IDX_TO_CLASS = {v: k for k, v in CLASS_TO_IDX.items()}

# ============================================================================
# Train/Val/Test Split (Patient-level)
# ============================================================================
N_TRAIN_PATIENTS = 27  # ~74%
N_VAL_PATIENTS = 5     # ~13%
N_TEST_PATIENTS = 5    # ~13%p

if USE_PHYSIONET:
    # PhysioNet: 44 hasta
    N_TRAIN_PATIENTS = 34  # ~75%
    N_VAL_PATIENTS = 5     # ~12%
    N_TEST_PATIENTS = 5    # ~12%

# K-Fold Cross Validation
USE_KFOLD = False           # True: K-Fold CV, False: Normal train/val/test
N_FOLDS = 5                # Fold sayısı
N_TEST_PATIENTS_KFOLD = 5  # K-Fold modunda test için ayrılacak hasta sayısı

# Random seed (reproducibility için)
RANDOM_SEED = 42

# ============================================================================
# Model Settings
# ============================================================================
# Model tipi: 'cnn1d', 'cnn1d_paper', 'deepsleepnet'
# deepsleepnet: Two-branch CNN + BiLSTM for temporal learning (better N1 detection)
MODEL_TYPE = 'cnn1d'  # CNN+BiLSTM hybrid for sequence learning

# ============================================================================
# DeepSleepNet Settings (CNN + BiLSTM Hybrid)
# ============================================================================
# Sequence length: Kaç ardışık epoch BiLSTM'e verilecek
# Daha uzun = daha iyi temporal context, ama daha fazla VRAM
# Stateful LSTM: 15 epoch = 7.5 dakika context
SEQUENCE_LENGTH = 25

# Sequence stride: Kaç epoch atlayarak yeni sequence oluştur
# stride=1: Her epoch'tan başla (maksimum data, 80% overlap)
# stride=2-4: Daha az data, daha az overlap → underfitting riski
# NOT: stride>1 deneyleri underfitting gösterdi, stride=1 + güçlü regularization tercih edildi
SEQUENCE_STRIDE = 1

# BiLSTM hidden size (per direction, bidirectional = 2x)
# Stateful LSTM: 512 per direction (toplam 1024)
LSTM_HIDDEN_SIZE = 512

# Sequence training: True ise ardışık epoch'lar ile eğitim (BiLSTM için gerekli)
USE_SEQUENCE_TRAINING = False

# DeepSleepNet Configuration (4GB VRAM optimized - T1200)
DEEPSLEEPNET_CONFIG = {
    'n_channels': N_CHANNELS_ACTIVE if SINGLE_CHANNEL_MODE else N_CHANNELS,
    'n_classes': N_CLASSES,
    'sample_rate': SAMPLE_RATE,
    'sequence_length': SEQUENCE_LENGTH,
    'dropout': 0.65,  # Artırıldı: stride=1 ile overfitting önlemek için (0.5 → 0.65)
    'lstm_hidden_size': LSTM_HIDDEN_SIZE,
    'samples_per_epoch': SAMPLES_PER_EPOCH,
}

# Alternative configs for different VRAM sizes
DEEPSLEEPNET_CONFIG_8GB = {
    'n_channels': N_CHANNELS_ACTIVE if SINGLE_CHANNEL_MODE else N_CHANNELS,
    'n_classes': N_CLASSES,
    'sample_rate': SAMPLE_RATE,
    'sequence_length': 11,
    'dropout': 0.5,
    'lstm_hidden_size': 256,
    'samples_per_epoch': SAMPLES_PER_EPOCH,
}

# 1D-CNN Architecture
CNN1D_CONFIG = {
    'n_channels': N_CHANNELS,
    'n_classes': N_CLASSES,
    'dropout': 0.5,  # Increased from 0.4 to reduce overfitting
    
    # Convolutional layers (scaled for 256Hz: kernels × 2.56)
    'conv1_filters': 32,
    'conv1_kernel': 128,  # 50 × 2.56 = 128 (for sleep spindles, K-complexes)
    'conv1_pool': 8,      # Best 256Hz config: [8,8,8] achieved 64.70% val (10% gap)
    
    'conv2_filters': 64,
    'conv2_kernel': 64,   # 25 × 2.56 = 64
    'conv2_pool': 8,      # Aggressive but works for 256Hz
    
    'conv3_filters': 128,
    'conv3_kernel': 26,   # 10 × 2.56 = 26
    'conv3_pool': 8,      # Maintains consistency
    
    # Fully connected layers
    'fc1_units': 256,
}


# Paper: "A Deep Learning Approach to Automated Sleep Stages Classification Using Multi-Modal Signals"
CNN1D_CONFIG_PAPER = {
    'n_channels': N_CHANNELS,
    'n_classes': N_CLASSES,
    'dropout_conv': 0.2,   # Paper: 0.2 for conv layers
    'dropout_dense': 0.5,  # Paper: 0.5 for dense layers
    
    # Conv Layer 1: 128 filters, kernel=50, stride=5
    'conv1_filters': 128,
    'conv1_kernel': 50,
    'conv1_stride': 5,
    'conv1_pool': 2,       # Maxpool 1x2
    
    # Conv Layer 2: 256 filters, kernel=5, stride=1
    'conv2_filters': 256,
    'conv2_kernel': 5,
    'conv2_stride': 1,
    'conv2_pool': 2,       # Maxpool 1x2
    
    # Conv Layer 3: 300 filters, kernel=5, stride=2
    'conv3_filters': 300,
    'conv3_kernel': 5,
    'conv3_stride': 2,
    'conv3_pool': 2,       # Maxpool 1x2
    
    # Fully connected layers: 2x1500 units
    'fc1_units': 1500,
    'fc2_units': 1500,
    
    # Paper uses BatchNorm before each conv layer
    'use_batchnorm': True,
}
# ============================================================================
# Training Settings
# ============================================================================
# Batch size: 
# - DeepSleepNet (stateful LSTM): batch_size=1 ZORUNLU
# - CNN modelleri (cnn1d, cnn1d_paper): batch_size=32 veya daha yüksek
if MODEL_TYPE == 'deepsleepnet':
    BATCH_SIZE = 1  # Stateful LSTM için
else:
    BATCH_SIZE = 32  # CNN modelleri için

LEARNING_RATE = 1e-3
N_EPOCHS = 50

# Dropout rate (top-level for easy access)
DROPOUT = 0.5

# ============================================================================
# Mixed Precision Training (FP16) - 4GB VRAM için önemli
# ============================================================================
USE_MIXED_PRECISION = False  # Debug için kapatıldı - stabil olunca True yap

# Gradient Accumulation: Effective batch size artırmak için
# effective_batch = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
# 4 * 1 = 4 effective batch size (debug için azaltıldı)
GRADIENT_ACCUMULATION_STEPS = 1

# Gradient Clipping (LSTM için önemli - exploding gradient önleme)
USE_GRADIENT_CLIPPING = False
GRADIENT_CLIP_VALUE = 5.0  # Stateful LSTM için artırıldı: 0.5 → 5.0

# Optimizer
OPTIMIZER = 'AdamW'  # AdamW is better for LSTM/Transformer models
WEIGHT_DECAY = 0.0001  # Artırıldı: stride=1 ile overfitting önlemek için (0.01 → 0.02)

# Loss function
LOSS_FUNCTION = 'CrossEntropyLoss'
USE_CLASS_WEIGHTS = True  # KRITIK: Class imbalance için (N2 %51 dominant)

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'  # 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'
LR_SCHEDULER_PATIENCE = 5  # Kaç epoch validation loss düşmezse LR'yi azalt
LR_SCHEDULER_FACTOR = 0.5  # LR'yi ne kadar azalt

# Learning rate warmup
# İlk N epoch boyunca LR'yi kademeli artır (training stabilitesi için)
USE_WARMUP = True
WARMUP_EPOCHS = 3  # İlk 3 epoch warmup (LR: 0 → LEARNING_RATE)
WARMUP_START_LR = 1e-4  # Warmup başlangıç LR'si

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 7  # Paper: 15 epochs patience

# ============================================================================
# Data Preprocessing
# ============================================================================
# Amplitude Clipping (Artifact removal - reduces kurtosis)
# PSG data has high kurtosis (~2000) due to movement/electrode artifacts
# Clipping at p1-p99 reduces kurtosis to ~2 without data loss
# NOTE: Clipping is now applied in edf_to_mongo.py during DB import
#       Set to False to avoid double clipping
USE_AMPLITUDE_CLIPPING = False  # Already applied in edf_to_mongo.py
CLIP_PERCENTILE_LOW = 1    # Lower percentile (e.g., 1 = p1)
CLIP_PERCENTILE_HIGH = 99  # Upper percentile (e.g., 99 = p99)

# Normalization method
NORMALIZATION = 'z-score'  # 'z-score', 'min-max', 'none'

# Z-score normalization parametreleri
NORM_EPSILON = 1e-8  # Sıfıra bölme hatası için

# Filtering (şimdilik kapalı)
USE_FILTERING = False
LOWCUT = 0.5   # Hz
HIGHCUT = 30.0  # Hz
FILTER_ORDER = 5

# ============================================================================
# Data Augmentation (şimdilik kapalı)
# ============================================================================
USE_AUGMENTATION = False
AUG_NOISE_STD = 0.01
AUG_TIME_SHIFT_MAX = 100  # samples
AUG_AMPLITUDE_SCALE_RANGE = (0.9, 1.1)

# ============================================================================
# DataLoader Settings
# ============================================================================
NUM_WORKERS = 4  # Multi-process data loading
PIN_MEMORY = True  # GPU için hızlandırma
PERSISTENT_WORKERS = True  # Worker'ları tekrar başlatma

# ============================================================================
# Model Export Settings (Netron)
# ============================================================================
EXPORT_TO_ONNX = False  # True ise model ONNX'e export edilir (opset v18)
ONNX_EXPORT_AFTER_TRAINING = False   # Training bitince export yap

# ============================================================================
# Checkpoint & Logging
# ============================================================================
# Dizinler
OUTPUT_DIR = 'outputs'
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, 'checkpoints')
LOG_DIR = os.path.join(OUTPUT_DIR, 'logs')
HYPNOGRAM_DIR = os.path.join(OUTPUT_DIR, 'hypnograms' + DB_NAME)

# Checkpoint kaydetme
SAVE_BEST_ONLY = True  # Sadece en iyi modeli kaydet
CHECKPOINT_METRIC = 'val_loss'  # 'val_loss', 'val_accuracy', 'val_f1'

# Logging
LOG_INTERVAL = 10  # Kaç batch'te bir log yazdır
TENSORBOARD = False  # TensorBoard kullan (opsiyonel)

# ============================================================================
# Evaluation Settings
# ============================================================================
# Confusion matrix
SAVE_CONFUSION_MATRIX = True

# Hypnogram comparison
SAVE_HYPNOGRAMS = True
HYPNOGRAM_FIGSIZE = (16, 6)
HYPNOGRAM_DPI = 150

# Metrics
CALCULATE_KAPPA = True  # Cohen's Kappa
CALCULATE_PER_CLASS_METRICS = True  # Her sınıf için precision/recall/f1

# ============================================================================
# Device Settings
# ============================================================================
# CUDA var mı kontrol et
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.device('cuda' if USE_CUDA else 'cpu')

# Multi-GPU (gelecekte)
USE_MULTI_GPU = False

# ============================================================================
# Reproducibility
# ============================================================================
# Random seed'leri sabitle
def set_seed(seed=RANDOM_SEED):
    """
    Reproducibility için tüm random seed'leri sabitle
    """
    import random
    import numpy as np
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if USE_CUDA:
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # Deterministic behavior (biraz yavaşlatabilir)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ============================================================================
# Dizinleri oluştur
# ============================================================================
def create_directories():
    """
    Gerekli dizinleri oluştur
    """
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(HYPNOGRAM_DIR, exist_ok=True)
    print(f"✓ Dizinler oluşturuldu: {OUTPUT_DIR}/")

# ============================================================================
# Config özeti yazdır
# ============================================================================
def print_config():
    """
    Configuration özetini yazdır
    """
    print(f"\n{'='*70}")
    print(f"SLEEP STAGE CLASSIFICATION - CONFIGURATION")
    print(f"{'='*70}")
    print(f"\n📊 DATA:")
    print(f"  Channels: {SELECTED_CHANNELS}")
    print(f"  Sample rate: {SAMPLE_RATE} Hz")
    print(f"  Samples per epoch: {SAMPLES_PER_EPOCH}")
    print(f"  Classes: {CLASS_NAMES}")
    
    print(f"\n🔀 SPLIT:")
    print(f"  Train: {N_TRAIN_PATIENTS} patients")
    print(f"  Val: {N_VAL_PATIENTS} patients")
    print(f"  Test: {N_TEST_PATIENTS} patients")
    
    print(f"\n🏗️  MODEL:")
    print(f"  Type: {MODEL_TYPE}")
    print(f"  Architecture: 1D-CNN")
    
    print(f"\n⚙️  TRAINING:")
    print(f"  Database: {DB_NAME}")
    print(f"  Batch size: {BATCH_SIZE}")
    print(f"  Learning rate: {LEARNING_RATE}")
    print(f"  Epochs: {N_EPOCHS}")
    print(f"  Optimizer: {OPTIMIZER}")
    print(f"  Weight decay: {WEIGHT_DECAY:.4f}")
    print(f"  Dropout: {CNN1D_CONFIG['dropout']}")
    print(f"  Loss function: {LOSS_FUNCTION}")
    print(f"  Balance strategy: {BALANCE_STRATEGY}")
    print(f"  Early stopping: {USE_EARLY_STOPPING}")
    
    print(f"\n🔧 PREPROCESSING:")
    print(f"  Normalization: {NORMALIZATION}")
    print(f"  Filtering: {USE_FILTERING}")
    print(f"  Augmentation: {USE_AUGMENTATION}")
    
    print(f"\n💻 DEVICE:")
    print(f"  CUDA available: {USE_CUDA}")
    print(f"  Device: {DEVICE}")
    
    print(f"\n📁 OUTPUT:")
    print(f"  Directory: {OUTPUT_DIR}")
    
    print(f"\n{'='*70}\n")


# ============================================================================
# Main (test için)
# ============================================================================
if __name__ == "__main__":
    set_seed()
    create_directories()
    print_config()
    
    print("Config test başarılı! ✓")