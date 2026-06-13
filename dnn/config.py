"""
Sleep Stage Classification - Configuration File
Tüm hyperparameter'lar ve ayarlar burada
"""

import torch
import os

# ============================================================================
# Dataset Selection
# ============================================================================
USE_PHYSIONET = False  # True: PhysioNet dataset, False: Original dataset
PHYSIONET_DATASET = "ST"  # "ST" (Sleep Telemetry) veya "SC" (Sleep Cassette)

# ============================================================================
# Offline Mode (Kaggle/Colab - MongoDB olmadan çalıştırmak için)
# ============================================================================
# True: Veriler .npz dosyalarından yüklenir (MongoDB gerekmez)
# False: Veriler MongoDB'den yüklenir (varsayılan)
# Kullanım: edf_to_mongo.py / physionet_import.py çalıştırıldığında .npz otomatik oluşur.
#           Bu dosyaları Kaggle/Colab'a yükle → USE_OFFLINE_DATA = True yap
USE_OFFLINE_DATA = False

# NPZ dosyalarının saklanacağı kök dizin.
# Raw data ile aynı disk/klasör yapısında tutmak için özelleştirilebilir.
# None ise proje dizini (dnn/) altına düşer.
#OFFLINE_DATA_BASE_DIR = "/media/mehmet/40BC0D26BC0D17D41/sleep-staging-data"

# ============================================================================
# MongoDB Settings (USE_OFFLINE_DATA=False ise kullanılır)
# ============================================================================
MONGO_URI = "mongodb://localhost:27017/"

# Hastane veritabanları
DB_NAME_RAW = "psg_data"              # Raw kanallar (45 kanal, filtresiz)
DB_NAME_FILTERED = "psg_data_filtered" # Differential kanallar (9 kanal, filtreli)

# Varsayılan: filtered (differential montaj, AASM standard)
DB_NAME = DB_NAME_FILTERED
OFFLINE_DATA_BASE_DIR = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/psg_data_phase3/data/raw_data"

if USE_PHYSIONET:
    if PHYSIONET_DATASET == "SC":
        DB_NAME = "physionet_sleep_sc"
        OFFLINE_DATA_BASE_DIR = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/physionet_sleep/sleep-edf-database-expanded-1.0.0/sleep-cassette"
        SC_RAW_DATA_DIR = OFFLINE_DATA_BASE_DIR  # SC EDF dosyalarının bulunduğu klasör
    else:
        DB_NAME = "physionet_sleep"
        OFFLINE_DATA_BASE_DIR = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/physionet_sleep/sleep-edf-database-expanded-1.0.0/sleep-telemetry"

# Offline data dizini: OFFLINE_DATA_BASE_DIR / offline_data / DB_NAME
# Örn: .../sleep-staging-data/offline_data/physionet_sleep/*.npz
#      .../sleep-staging-data/offline_data/psg_data/*.npz
_base = OFFLINE_DATA_BASE_DIR if OFFLINE_DATA_BASE_DIR else os.path.dirname(__file__)
OFFLINE_DATA_DIR = os.path.join(_base, 'offline_data', DB_NAME)

# ============================================================================
# Data Settings
# ============================================================================
# Hastane verisi: Differential kanallar (AASM Standard)
# psg_data_filtered'da saklanan kanal adları: C3-M2, C4-M1, F3-M2, F4-M1, O1-M2, O2-M1, E1-M2, E2-M1, CHIN1-CHIN2
# F4-M1: Frontal EEG — N1 evresindeki theta aktivitesi ve vertex dalgaları için kritik
SELECTED_CHANNELS = ['C3-M2', 'F4-M1', 'E1-M2', 'O1-M2']  # 4 kanal (C3, F4, EOG, O1) - AASM standardına uygun
N_CHANNELS = 4
SAMPLE_RATE = 256  # Hz

if USE_PHYSIONET:
    if PHYSIONET_DATASET == "SC":
        # SC: EMG submental 1Hz → kullanılmaz, sadece 100Hz kanallar
        SELECTED_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal']
        N_CHANNELS = 3
    else:
        # ST: Tüm kanallar 100Hz
        SELECTED_CHANNELS = ['EEG Fpz-Cz','EOG horizontal']
        #SELECTED_CHANNELS = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']
        N_CHANNELS = 2
    SAMPLE_RATE = 100  # Hz

# Balanced dataset (overfitting'i azaltmak için)
BALANCE_STRATEGY = 'none'  # 'undersample', 'oversample', 'none'

# Epoch parametreleri
EPOCH_DURATION = 30  # saniye
SAMPLES_PER_EPOCH = SAMPLE_RATE * EPOCH_DURATION

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
N_TRAIN_PATIENTS = 25  # ~70%
N_VAL_PATIENTS = 6     # ~15%
N_TEST_PATIENTS = 6    # ~15%p

if USE_PHYSIONET:
    if PHYSIONET_DATASET == "SC":
        # PhysioNet SC: 78 subject (153 kayıt, subject-level split)
        # Sayılar SUBJECT bazında (recording değil!)
        N_TRAIN_PATIENTS = 58  # ~74% of 78 subjects
        N_VAL_PATIENTS = 10    # ~13%
        N_TEST_PATIENTS = 10   # ~13%
    else:
        # PhysioNet ST: 44 hasta (her hasta = 1 kayıt)
        N_TRAIN_PATIENTS = 34  # ~75%
        N_VAL_PATIENTS = 5     # ~12%
        N_TEST_PATIENTS = 5    # ~12%

# K-Fold Cross Validation
USE_KFOLD = True           # True: K-Fold CV, False: Normal train/val/test
N_FOLDS = 5                # Fold sayısı
N_TEST_PATIENTS_KFOLD = 8  # K-Fold: test subject sayısı (SC: 8 subject ≈ 16 recording ≈ 10K epoch)

# Random seed (reproducibility için)
RANDOM_SEED = 42

# ============================================================================
# Model Settings
# ============================================================================
# Model tipi ('cnn1d', 'cnn1d_v3', 'cnn1d_paper', 'deepsleepnet', 'transformer')
MODEL_TYPE = 'transformer'

# 1D-CNN Architecture
CNN1D_CONFIG = {
    'n_channels': N_CHANNELS,
    'n_classes': N_CLASSES,
    'dropout': 0.3,  # 0.6 → 0.7 → 0.3 → 0.5
    
    # Convolutional layers
    'conv1_filters': 32,
    'conv1_kernel': 50,
    'conv1_pool': 4,
    
    'conv2_filters': 64,
    'conv2_kernel': 25,
    'conv2_pool': 4,
    
    'conv3_filters': 128,
    'conv3_kernel': 10,
    'conv3_pool': 4,
    
    # Fully connected layers
    'fc1_units': 256,
}


CNN1D_CONFIG_V2 = {
    'n_channels': N_CHANNELS,
    'n_classes': N_CLASSES,
    'dropout': 0.6,  # 0.6 → 0.7 → 0.3 → 0.5
    
    # Convolutional layers (2x capacity increase)
    'conv1_filters': 64,   # 32 → 64 (daha fazla low level feature)
    'conv1_kernel': 50,
    'conv1_pool': 4,
    
    'conv2_filters': 128,  # 64 → 128 (daha fazla mid level feature)
    'conv2_kernel': 25,
    'conv2_pool': 4,
    
    'conv3_filters': 256,  # 128 → 256 (daha fazla high level feature)
    'conv3_kernel': 10,
    'conv3_pool': 4,
    
    # Fully connected layers (2x capacity increase)
    'fc1_units': 512,      # 256 → 512 (daha güçlü classification)
}

# CNN1D V3: Multi-Scale + Residual + SE Attention + GAP
# ~280K params (vs 1.68M for CNN1D) - daha verimli parametre kullanımı
# Kernel boyutları 100 Hz referans alınarak tanımlanır, SAMPLE_RATE'e göre otomatik ölçeklenir.
# Böylece 100 Hz (PhysioNet) ve 256 Hz (hastane verisi) arasında temporal coverage korunur.
_sr_scale = SAMPLE_RATE / 100.0  # 100 Hz → 1.0, 256 Hz → 2.56
def _scale_kernel(base_k):
    """Kernel boyutunu sample rate'e göre ölçekle, tek sayı yap (symmetric padding için)."""
    k = int(round(base_k * _sr_scale))
    return k if k % 2 == 1 else k + 1  # Tek sayı garanti

def _scale_pool(base_p):
    """Pool boyutunu sample rate'e göre ölçekle (en az 2).
    3 pool katmanı kaskad olduğundan küp kök kullanılır:
      100 Hz: 3000 → /4=750 → /4=187 → /4=46
      256 Hz: 7680 → /5=1536 → /5=307 → /5=61
    """
    return max(2, int(round(base_p * (_sr_scale ** (1/3)))))

CNN1D_V3_CONFIG = {
    'n_channels': N_CHANNELS,
    'n_classes': N_CLASSES,
    'dropout': 0.3,

    # Multi-scale stem: farklı temporal ölçekleri paralel yakala
    # Referans (100 Hz): fine=15(~150ms), medium=51(~510ms), broad=125(~1.25s)
    'stem_filters': 16,                # Her branch'te 16 filter (toplam=48)
    'stem_kernels': [_scale_kernel(15), _scale_kernel(51), _scale_kernel(125)],
    'stem_pool': _scale_pool(4),

    # Residual block 1
    # Referans (100 Hz): k=7 (~70ms)
    'block1_filters': 96,
    'block1_kernel': _scale_kernel(7),
    'block1_pool': _scale_pool(4),

    # Residual block 2
    # Referans (100 Hz): k=5 (~50ms)
    'block2_filters': 128,
    'block2_kernel': _scale_kernel(5),
    'block2_pool': _scale_pool(4),

    # SE (Squeeze-and-Excitation) channel attention
    'se_reduction': 16,

    # Classifier
    'fc1_units': 64,
}

# SleepTransformer: CNN1D_V3 Encoder + Transformer Encoder
# ~548K params — CNN encoder (~260K) + Transformer (~267K) + Head (~1K)
TRANSFORMER_CONFIG = {
    # CNN Encoder (Stage 1) — CNN1D_V3 feature extractor
    'encoder_type': 'cnn1d_v3',
    'd_model': 128,                     # Transformer embedding = CNN GAP output

    # Transformer Encoder (Stage 2)
    'nhead': 4,                         # Multi-head attention head sayısı
    'num_layers': 2,                    # Transformer encoder layer sayısı
    'dim_feedforward': 256,             # Feed-forward network boyutu (2×d_model)
    'transformer_dropout': 0.1,         # Transformer dropout (0.3→0.1: daha fazla bilgi akışı)
    'activation': 'gelu',              # Activation function
    'norm_first': True,                 # Pre-LN (daha stabil training)

    # Positional Encoding
    'pos_encoding': 'sinusoidal',        # 'sinusoidal' veya 'learnable'
    'max_seq_len': 25,                  # Maksimum sequence uzunluğu

    # Classification Head (Stage 3)
    'head_dropout': 0.3,

    # CNN Encoder Freeze & Pre-trained
    'freeze_cnn': False,                # True: CNN parametrelerini baştan dondur
    'freeze_cnn_epochs': 10,            # İlk N epoch CNN freeze, sonra unfreeze (0=kapalı)
    'pretrained_cnn_checkpoint': 'outputs/runs/260325-1530/checkpoints/best_model.pth',
    # None veya '' ise sıfırdan eğitilir. Checkpoint path verilirse CNN ağırlıkları yüklenir.
}

# ============================================================================
# Training Settings
# ============================================================================
BATCH_SIZE = 32
LEARNING_RATE = 0.001
N_EPOCHS = 50

# Optimizer
OPTIMIZER = 'Adam'  # 'Adam', 'SGD', 'AdamW'
WEIGHT_DECAY = 1e-4  # 1e-5 → 1e-4 → 5e-4

# Loss function
LOSS_FUNCTION = 'CrossEntropyLoss'  # 'CrossEntropyLoss', 'FocalLoss'
FOCAL_LOSS_GAMMA = 2.0       # Focal Loss gamma parametresi (2.0 standart değer)
LABEL_SMOOTHING = 0.1        # Label smoothing (0→kapalı, 0.1→önerilen). Tek sınıfa collapse'ı önler.
USE_CLASS_WEIGHTS = True  # Balanced dataset kullanıldığında False olmalı
CLASS_WEIGHT_MODE = 'sqrt'  # 'inverse': tam inverse freq (N3 5.4x boost)
                            # 'sqrt': yumuşatılmış (√inverse, N3 2.3x boost)
                            # 'log': logaritmik yumuşatma

# Gradient Clipping (transformer training stabilitesi için)
GRADIENT_CLIP_VALUE = 1.0  # 0: kapalı, >0: max gradient norm

# Learning rate scheduler
USE_LR_SCHEDULER = True
LR_SCHEDULER_TYPE = 'ReduceLROnPlateau'  # 'StepLR', 'ReduceLROnPlateau', 'CosineAnnealingWarmRestarts'
LR_SCHEDULER_PATIENCE = 5  # Kaç epoch validation loss düşmezse LR'yi azalt
LR_SCHEDULER_FACTOR = 0.5  # LR'yi ne kadar azalt

# Learning Rate Warmup (Transformer için önerilen)
USE_WARMUP = False           # True: ilk N epoch lineer warmup uygula
WARMUP_EPOCHS = 5            # Warmup epoch sayısı
WARMUP_START_LR = 1e-6       # Warmup başlangıç LR

# CosineAnnealingWarmRestarts parametreleri
COSINE_T_0 = 20              # İlk restart periyodu (epoch)
COSINE_T_MULT = 2            # Her restart'ta periyodu 2× uzat
COSINE_ETA_MIN = 1e-6        # Minimum LR

# Early stopping
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 10  # Kaç epoch iyileşme yoksa dur

# ============================================================================
# Data Preprocessing
# ============================================================================
# Normalization method
NORMALIZATION = 'z-score'  # 'z-score', 'min-max', 'none'

# Z-score normalization parametreleri
NORM_EPSILON = 1e-8  # Sıfıra bölme hatası için

# Amplitude Clipping (Artifact removal)
USE_AMPLITUDE_CLIPPING = False  # True: percentile clipping uygula
CLIP_PERCENTILE_LOW = 1    # Lower percentile (p1)
CLIP_PERCENTILE_HIGH = 99  # Upper percentile (p99)

# Bandpass Filtering (Butterworth)
# 0.3 Hz: DC drift + çok düşük frekans artifact temizleme
# 35 Hz: Yüksek frekans gürültü temizleme (sleep spindles 11-16 Hz korunur)
USE_FILTERING = False
LOWCUT = 0.3   # Hz (slow oscillations korunur)
HIGHCUT = 35.0  # Hz (gamma altı, EMG artifact azaltma)
FILTER_ORDER = 5

# ============================================================================
# Data Augmentation (şimdilik kapalı)
# ============================================================================
USE_AUGMENTATION = False
AUG_NOISE_STD = 0.05 # Sinyal genliğinin %5'i kadar Gaussian gürültü ekle (0.01 → 0.05)
AUG_TIME_SHIFT_MAX = 50 # ms cinsinden maksimum zaman kaydırma (0.1s = 100ms → 50ms) (100 → 50)
AUG_AMPLITUDE_SCALE_RANGE = (0.8, 1.2) # Sinyal genliğini 0.8x - 1.2x arasında rastgele ölçeklendir (0.8-1.2 → 0.9-1.1)

# ============================================================================
# Sequence Settings (DeepSleepNet / BiLSTM için)
# ============================================================================
SEQUENCE_LENGTH = 25  # Ardışık epoch sayısı
SEQUENCE_STRIDE = 1   # Sequence stride

# ============================================================================
# DataLoader Settings
# ============================================================================
NUM_WORKERS = 4  # Multi-process data loading
PIN_MEMORY = True  # GPU için hızlandırma
PERSISTENT_WORKERS = True  # Worker'ları tekrar başlatma

# ============================================================================
# Model Export Settings (Netron)
# ============================================================================
EXPORT_TO_ONNX = False  # Geçici: ONNX bağımlılık uyumsuzluğunda export'u atla
ONNX_EXPORT_AFTER_TRAINING = False   # Training bitince export yapma

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
# Model-Specific Auto Configuration
# ============================================================================
# MODEL_TYPE'a göre training parametrelerini otomatik ayarla.
# Kullanıcı sadece MODEL_TYPE'ı değiştirmeli, geri kalan parametreler
# otomatik olarak optimize edilmiş değerlere set edilir.

if MODEL_TYPE == 'transformer':
    # Transformer-optimal training settings
    BATCH_SIZE = 16               # Sequence mode, VRAM dostu
    LEARNING_RATE = 5e-4          # Transformer standardı (1e-3 çok agresif)
    OPTIMIZER = 'AdamW'           # Weight decay decouple
    WEIGHT_DECAY = 1e-2           # Transformer regularization
    N_EPOCHS = 80                 # Transformer daha yavaş converge
    EARLY_STOPPING_PATIENCE = 10  # Overfitting kontrolü (15→10)
    GRADIENT_CLIP_VALUE = 1.0     # Gradient norm clipping
    
    # LR Scheduler: Warmup + CosineAnnealing
    USE_LR_SCHEDULER = True
    LR_SCHEDULER_TYPE = 'CosineAnnealingWarmRestarts'
    USE_WARMUP = True
    WARMUP_EPOCHS = 5
    WARMUP_START_LR = 1e-6


# ============================================================================
# Main (test için)
# ============================================================================
if __name__ == "__main__":
    set_seed()
    create_directories()
    print_config()
    
    print("Config test başarılı! ✓")