# DeepSleepNet Mimarisi Detaylı Açıklama

> Bu döküman DeepSleepNet (CNN + BiLSTM) modelinin nasıl çalıştığını açıklar.
> 
> Referans: "DeepSleepNet: a Model for Automatic Sleep Stage Scoring based on Raw Single-Channel EEG" - Supratak et al. 2017

---

## 1. VERİ AKIŞI - Sequence Oluşturma

```
Veritabanı (psg_data_filtered)
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│  SleepSequenceDataset                                           │
│  ─────────────────────                                          │
│  Her hasta için ardışık 5 epoch'u bir "sequence" olarak al      │
│                                                                 │
│  Örnek: Hasta A, Epoch 10-14                                    │
│  ┌─────┬─────┬─────┬─────┬─────┐                                │
│  │ E10 │ E11 │ E12 │ E13 │ E14 │  ← 5 ardışık 30-sn epoch       │
│  │ N2  │ N2  │ N1  │ REM │ REM │  ← Her epoch'un gerçek labeli  │
│  └─────┴─────┴─────┴─────┴─────┘                                │
│                                                                 │
│  Stride=1: Sonraki sequence → Epoch 11-15                       │
│  Bu sayede temporal geçişler öğrenilir:                         │
│    - N2→N1→REM (REM'e giriş)                                    │
│    - W→N1→N2 (uykuya dalış)                                     │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
    Output Shape: [batch, 5, 4, 7680]
                   │     │  │   │
                   │     │  │   └── 7680 sample (30sn × 256Hz)
                   │     │  └────── 4 kanal (C3-M2, F4-M1, E1-M2, CHIN1-CHIN2)
                   │     └───────── 5 epoch sequence
                   └─────────────── batch size
```

---

## 2. CNN - ÇİFT BRANCH FEATURE EXTRACTION

Her epoch **ayrı ayrı** CNN'den geçirilir. İki paralel branch farklı frekans bantlarını yakalar:

```
Bir Epoch: [4 kanal, 7680 sample]
                │
    ┌───────────┴───────────┐
    ▼                       ▼
┌─────────────────┐  ┌─────────────────┐
│ SMALL FILTER    │  │ LARGE FILTER    │
│ BRANCH          │  │ BRANCH          │
├─────────────────┤  ├─────────────────┤
│ Kernel = Fs/2   │  │ Kernel = Fs×4   │
│ = 128 sample    │  │ = 1024 sample   │
│ = 0.5 saniye    │  │ = 4 saniye      │
├─────────────────┤  ├─────────────────┤
│ YAKALAR:        │  │ YAKALAR:        │
│ • Sleep spindle │  │ • Delta dalgası │
│   (11-16 Hz)    │  │   (0.5-4 Hz)    │
│ • K-complex     │  │ • Slow waves    │
│ • Alpha (8-12)  │  │ • Theta (4-8)   │
│ • Beta (12-30)  │  │                 │
├─────────────────┤  ├─────────────────┤
│ Conv1: 64 filtr │  │ Conv1: 64 filtr │
│ ↓ MaxPool(8)    │  │ ↓ MaxPool(4)    │
│ Conv2-4: 128    │  │ Conv2-4: 128    │
│ ↓ MaxPool(4)    │  │ ↓ MaxPool(2)    │
└────────┬────────┘  └────────┬────────┘
         │                    │
         └────────┬───────────┘
                  ▼
         ┌────────────────┐
         │  CONCATENATE   │
         │  + LayerNorm   │
         │  + Dropout     │
         └───────┬────────┘
                 ▼
         Feature Vector: ~3000 dim
         (Her epoch için sabit boyutlu özellik)
```

### Small Filter Branch Detayı

| Layer | Output Shape | Açıklama |
|-------|-------------|----------|
| Input | [4, 7680] | 4 kanal × 7680 sample |
| Conv1(64, k=128, s=16) | [64, 481] | İlk özellik çıkarımı |
| BatchNorm + ReLU | [64, 481] | Normalizasyon |
| MaxPool(8) | [64, 60] | Boyut küçültme |
| Dropout(0.5) | [64, 60] | Regularization |
| Conv2(128, k=8) | [128, 60] | Daha derin özellikler |
| Conv3(128, k=8) | [128, 60] | |
| Conv4(128, k=8) | [128, 60] | |
| MaxPool(4) | [128, 15] | Final pooling |
| **Flatten** | [1920] | Feature vector |

### Large Filter Branch Detayı

| Layer | Output Shape | Açıklama |
|-------|-------------|----------|
| Input | [4, 7680] | 4 kanal × 7680 sample |
| Conv1(64, k=1024, s=128) | [64, 61] | Düşük frekans özellikleri |
| BatchNorm + ReLU | [64, 61] | Normalizasyon |
| MaxPool(4) | [64, 15] | Boyut küçültme |
| Dropout(0.5) | [64, 15] | Regularization |
| Conv2(128, k=6) | [128, 15] | |
| Conv3(128, k=6) | [128, 15] | |
| Conv4(128, k=6) | [128, 15] | |
| MaxPool(2) | [128, 7] | Final pooling |
| **Flatten** | [896] | Feature vector |

**Toplam CNN Output: 1920 + 896 = ~2816 features**

---

## 3. SEQUENCE İŞLEME - CNN'den BiLSTM'e

```
5 Epoch Sequence
┌─────┬─────┬─────┬─────┬─────┐
│ E10 │ E11 │ E12 │ E13 │ E14 │
└──┬──┴──┬──┴──┬──┴──┬──┴──┬──┘
   │     │     │     │     │
   ▼     ▼     ▼     ▼     ▼
┌─────────────────────────────────────────┐
│            CNN (shared weights)          │
│  Her epoch AYNI CNN'den geçer           │
└──┬──────┬──────┬──────┬──────┬──────────┘
   │      │      │      │      │
   ▼      ▼      ▼      ▼      ▼
┌─────┬─────┬─────┬─────┬─────┐
│ F10 │ F11 │ F12 │ F13 │ F14 │  Feature vectors (~3000 dim each)
└─────┴─────┴─────┴─────┴─────┘
   │      │      │      │      │
   └──────┴──────┴──────┴──────┘
                 │
                 ▼
    [batch, 5, ~3000] → BiLSTM'e giriş
```

**Önemli:** CNN ağırlıkları tüm epoch'lar için **paylaşımlıdır** (shared weights). Bu sayede:
- Model boyutu küçük kalır
- Epoch'lar arası tutarlı özellik çıkarımı sağlanır
- Transfer learning kolaylaşır

---

## 4. BiLSTM - TEMPORAL LEARNING + RESIDUAL

```
                    Feature Sequence: [batch, 5, 3000]
                              │
            ┌─────────────────┴─────────────────┐
            │                                   │
            ▼                                   ▼
    ┌───────────────┐                  ┌───────────────┐
    │   FC Layer    │                  │   BiLSTM-1    │
    │   (Residual)  │                  │  hidden=128   │
    │   3000→256    │                  │  bidirectional│
    └───────┬───────┘                  └───────┬───────┘
            │                                   │
            │                                   ▼
            │                          ┌───────────────┐
            │                          │   Dropout     │
            │                          └───────┬───────┘
            │                                   │
            │                                   ▼
            │                          ┌───────────────┐
            │                          │   BiLSTM-2    │
            │                          │  hidden=128   │
            │                          └───────┬───────┘
            │                                   │
            │                                   ▼
            │                          ┌───────────────┐
            │                          │   Dropout     │
            │                          └───────┬───────┘
            │                                   │
            └───────────────┬───────────────────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   RESIDUAL    │  ← FC output + BiLSTM output
                    │   CONNECTION  │
                    └───────┬───────┘
                            │
                            ▼
                    ┌───────────────┐
                    │   FC Output   │
                    │   256 → 5     │
                    └───────┬───────┘
                            │
                            ▼
         Output: [batch, 5, 5] (her epoch için 5 sınıf logit)
                               │
                               ▼
         Softmax → Tahmin: [batch, 5] (W, N1, N2, N3, REM)
```

### Residual Connection Neden Önemli?

1. **Gradient Flow**: Derin ağlarda gradient kaybolmasını önler
2. **N1 Sınıfı**: Nadir sınıflar için daha iyi gradient akışı
3. **Hızlı Yakınsama**: Training daha hızlı converge eder
4. **Feature Preservation**: Orijinal CNN özellikleri korunur

---

## 5. BiLSTM NEDEN ÖNEMLİ?

### Temporal Pattern Learning

```
BiLSTM'in Gördüğü Temporal Pattern:
                              
    Time ───────────────────────────────────►
    
    ┌─────┬─────┬─────┬─────┬─────┐
    │  W  │  W  │ N1  │ N2  │ N2  │  ← Uykuya dalış paterni
    └─────┴─────┴─────┴─────┴─────┘
    
    ┌─────┬─────┬─────┬─────┬─────┐
    │ N2  │ N2  │ N1  │ REM │ REM │  ← REM'e giriş paterni
    └─────┴─────┴─────┴─────┴─────┘
    
    ┌─────┬─────┬─────┬─────┬─────┐
    │ N3  │ N3  │ N2  │ N2  │ N1  │  ← Uyanma paterni
    └─────┴─────┴─────┴─────┴─────┘
```

### Bidirectional Avantajı

```
BiLSTM BIDIRECTIONAL olduğu için:
    - Forward LSTM: Sol→Sağ (geçmiş bilgisi)
    - Backward LSTM: Sağ→Sol (gelecek bilgisi)
    
Bu sayede E12'yi sınıflarken hem E10-E11'i hem E13-E14'ü görür!
```

### N1 İçin Önemi

N1 tek başına sınıflandırması zor çünkü:
- Wake'e benzer (düşük amplitüd)
- N2'ye benzer (bazı özellikler örtüşür)

**Ama temporal context ile:**
- `W → ? → N2` ise `?` muhtemelen **N1**
- `N2 → ? → REM` ise `?` muhtemelen **N1**
- `N3 → N2 → ?` ise `?` muhtemelen **N1** (uyanma)

---

## 6. ÖZET - SAYILARLA

### Konfigürasyon

| Parametre | Değer |
|-----------|-------|
| Sample Rate | 256 Hz |
| Epoch Duration | 30 saniye |
| Samples per Epoch | 7680 |
| Channels | 4 (C3-M2, F4-M1, E1-M2, CHIN1-CHIN2) |
| Sequence Length | 5 epoch |
| Classes | 5 (W, N1, N2, N3, REM) |

### Model Boyutları

| Katman | Shape |
|--------|-------|
| CNN Input | [batch, 5, 4, 7680] |
| CNN Output (per epoch) | ~3000 features |
| BiLSTM Input | [batch, 5, 3000] |
| BiLSTM Hidden | 128 (×2 bidirectional = 256) |
| BiLSTM Output | [batch, 5, 256] |
| Final Output | [batch, 5, 5] |

### Kaynak Kullanımı

| Metrik | Değer |
|--------|-------|
| Parametre Sayısı | ~6.1M |
| VRAM Kullanımı | ~3.5 GB (batch=4) |
| Training Süresi | ~23 dakika (14 epoch) |

---

## 7. TRAINING SONUÇLARI (260127-1855)

| Epoch | Train Acc | Val Acc |
|-------|-----------|---------|
| 1 | 36.71% | 22.24% |
| 3 | 58.60% | 39.26% |
| 7 | 66.42% | **57.25%** (Best) |
| 14 | 80.59% | 58.97% |

**Best Validation Accuracy: 61.88%**

### Gözlemler

1. ✅ Model öğreniyor (accuracy artıyor)
2. ⚠️ Overfitting var (Train-Val gap: 21.62%)
3. 🔄 Early stopping aktif (7 epoch iyileşme olmadı)

### İyileştirme Önerileri

1. **Regularization artır**: Dropout 0.5 → 0.6-0.7
2. **Data Augmentation**: Time shift, noise injection
3. **Focal Loss**: N1 gibi nadir sınıflar için
4. **Daha fazla veri**: Cross-validation veya daha fazla hasta

---

## 8. DOSYA REFERANSLARI

| Dosya | Açıklama |
|-------|----------|
| `dnn/models/deepsleepnet.py` | Model implementasyonu |
| `dnn/dataset.py` | SleepSequenceDataset sınıfı |
| `dnn/train.py` | Training loop |
| `dnn/config.py` | Hyperparametreler |
| `dnn/edf_to_mongo.py` | Veri çekme (differential channel support) |

---

