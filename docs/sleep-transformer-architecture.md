# SleepTransformer: Mimari Kılavuz

> **Amaç:** Bu doküman standart Transformer mimarisini açıklar, ardından projede kullandığımız SleepTransformer'ın standart mimariden nasıl ve neden farklılaştığını anlatır.
>
> **Hedef Okuyucu:** Temel DL bilgisine sahip biri. CNN ve RNN kavramlarına aşina olmak yeterli.
>
> **Son Güncelleme:** 2026-04-28

---

## İçindekiler

1. [Standart Transformer — Büyük Resim](#1-standart-transformer--büyük-resim)
2. [Self-Attention Mekanizması](#2-self-attention-mekanizması)
3. [Transformer Encoder Bloğu](#3-transformer-encoder-bloğu)
4. [SleepTransformer — Bizim Adaptasyonumuz](#4-sleeptransformer--bizim-adaptasyonumuz)
5. [Standart vs SleepTransformer — Farklar Tablosu](#5-standart-vs-sleeptransformer--farklar-tablosu)
6. [Neden LSTM Yerine Transformer?](#6-neden-lstm-yerine-transformer)
7. [Sık Sorulan Sorular](#7-sık-sorulan-sorular)

---

## 1. Standart Transformer — Büyük Resim

Transformer, 2017'de Google'ın "Attention Is All You Need" makalesinde tanıtıldı.
Başlangıçta dil çevirisi (İngilizce→Fransızca) için tasarlandı.

### Orijinal Mimari (NLP)

```
               ┌─────────────────┐
  Kaynak dil   │    ENCODER      │   Hedef dil
  "The cat"    │  (6× blok)      │   "Le chat"
       │       └────────┬────────┘        ▲
       │                │                 │
       ▼                ▼                 │
  ┌──────────┐    ┌─────────────┐    ┌───────────┐
  │ Input    │    │  Encoder    │    │ DECODER   │
  │ Embedding│───▶│  Output     │───▶│ (6× blok) │──▶ Çıkış
  │ + PosEnc │    │             │    │           │
  └──────────┘    └─────────────┘    └───────────┘
```

**İki ana parça:**

| Parça | Görevi | Örnek |
|-------|--------|-------|
| **Encoder** | Giriş verisini anla, zengin temsil çıkar | "The cat sat" → anlam vektörleri |
| **Decoder** | Encoder'ın çıktısını kullanarak hedef diziyi üret | Anlam vektörleri → "Le chat assis" |

> **Önemli:** Bizim SleepTransformer'da **sadece Encoder** kullanıyoruz.
> Decoder'a ihtiyacımız yok, çünkü "yeni bir dizi üretmiyoruz" — her epoch'u sınıflandırıyoruz.
> Bu yapıya **Encoder-only Transformer** denir (BERT gibi).

---

## 2. Self-Attention Mekanizması

Self-attention, Transformer'ın kalbidir. Amacı:
**"Bu dizideki her eleman, diğer elemanlara ne kadar dikkat etmeli?"**

### 2.1 Günlük Hayattan Analoji

Bir doktoru düşünün: Bir hastanın uyku kaydının 15. epoch'unu (30 saniyelik dilimini) skorlarken, sadece o 30 saniyeye değil, önceki ve sonraki epoch'lara da bakar:

```
Epoch:  11    12    13    14   [15]   16    17    18    19
Stage:   W     W     W    N1   [??]   N2    N2    N2    N3
                          ▲
                   "Burası N1→N2 geçişi,
                    muhtemelen N1 veya N2"
```

Self-attention tam olarak bunu yapar: **her epoch, diğer tüm epoch'lara bakarak "hangisi benim için önemli?" diye sorar.**

### 2.2 Matematiksel Mekanizma: Query, Key, Value

Her epoch'un vektörü üç farklı role dönüştürülür:

```
Epoch vektörü (x) ───┬── W_Q ──▶ Query  (Q)     : "Ben ne arıyorum?"
                     ├── W_K ──▶ Key    (K)     : "Ben ne sunuyorum?"
                     └── W_V ──▶ Value  (V)     : "Benim içeriğim nedir?"
```

**Query (Sorgu):** "Ben N1 geçişi arıyorum, yakınımda Wake var mı?"

**Key (Anahtar):** "Ben Wake epoch'uyum, alfa dalgaları içeriyorum."

**Value (Değer):** "İşte benim gerçek feature vektörüm."

### 2.3 Attention Skoru Hesaplama

```
                    Q × K^T
Attention(Q,K,V) = softmax( ──────── ) × V
                    √d_k

Adım adım:

1. Her epoch çifti için benzerlik skoru:
   skor(epoch_15, epoch_13) = Q₁₅ · K₁₃ᵀ

2. √d_k'ya böl (skoru normalize et, d_k = head boyutu):
   skor_norm = skor / √32 = skor / 5.66

3. Softmax uygula (tüm skorları 0-1 arası olasılığa çevir):
   ağırlık(epoch_15 → epoch_13) = softmax(skor_norm) = 0.31

4. Value'larla ağırlıklı toplama yap:
   çıkış₁₅ = 0.05·V₁₁ + 0.08·V₁₂ + 0.31·V₁₃ + 0.25·V₁₄ + 0.10·V₁₅
            + 0.12·V₁₆ + 0.05·V₁₇ + 0.03·V₁₈ + 0.01·V₁₉
```

**Sonuç:** Epoch 15'in yeni temsili artık sadece kendi bilgisini değil, komşularının (özellikle Epoch 13 ve 14'ün) bilgisini de taşır.

### 2.4 Multi-Head Attention

Tek bir attention yerine, birden çok attention "başlığı" (head) paralel çalışır:

```
                 ┌── Head 1: "Temporal yakınlık" (komşu epoch'lar)
                 │
Input ───────────┼── Head 2: "Frekans benzerliği" (aynı EEG paterni)
                 │
                 ├── Head 3: "Geçiş paterni" (W→N1 gibi)
                 │
                 └── Head 4: "Simetri" (önceki/sonraki denge)
                 
                 ↓ concat + linear
                 
                Output
```

Her head farklı bir "bakış açısı" öğrenir. Biz **4 head** kullanıyoruz.

> **Parametre:** `nhead=4`, `d_model=128` → her head `d_k = 128/4 = 32` boyutlu

---

## 3. Transformer Encoder Bloğu

Bir Transformer Encoder bloğu iki alt katmandan oluşur:

```
         ┌────────────────────────────────────────┐
         │        Transformer Encoder Block       │
         │                                        │
Input ──▶│  ┌──────────────────────────────────┐  │
    x    │  │  Layer Norm (Pre-LN)             │  │
         │  │  Multi-Head Self-Attention       │  │
         │  │  Dropout                         │  │
         │  │  + Residual Connection (x + ...) │  │
         │  └──────────────┬───────────────────┘  │
         │                 ↓                      │
         │  ┌──────────────────────────────────┐  │
         │  │  Layer Norm (Pre-LN)             │  │
         │  │  Feed-Forward Network (FFN)      │  │
         │  │  Linear(128→256) → GELU          │  │
         │  │  Linear(256→128)                 │  │
         │  │  Dropout                         │  │
         │  │  + Residual Connection           │  │
         │  └──────────────┬───────────────────┘  │
         │                 ↓                      │
         └────────────── Output ──────────────────┘
```

### 3.1 Alt Katmanlar

| Katman | Ne Yapar | Uyku Skorlama Karşılığı |
|--------|----------|------------------------|
| **Self-Attention** | Her epoch, diğer tüm epoch'lara bakarak bağlam toplar | "Bu epoch'un komşuları ne diyor?" |
| **Feed-Forward (FFN)** | Her epoch'un topladığı bağlamı işler, dönüştürür | "Bu bağlam bilgisinden ne çıkarabilirim?" |
| **Residual Connection** | Giriş ile çıkışı toplar (`x + sublayer(x)`) | Orijinal bilginin kaybolmasını önler |
| **Layer Normalization** | Değerleri normalize eder | Eğitim stabilitesi |

### 3.2 Pre-LN vs Post-LN

İki farklı normalizasyon sırası var:

```
Post-LN (orijinal paper):          Pre-LN (modern, bizim):
  x → Attention → Add → LN           x → LN → Attention → Add
  
```

**Biz Pre-LN kullanıyoruz** (`norm_first=True`):
- Eğitim daha stabil
- Learning rate warmup'a daha az bağımlı
- Gradient exploding riski düşük

---

## 4. SleepTransformer — Bizim Adaptasyonumuz

### 4.1 Genel Bakış

Standart Transformer, NLP için tasarlandı: **kelime dizileri** işler.
Biz uyku skorlama için uyarlıyoruz: **epoch dizileri** işleriz.

```
NLP Transformer:                    SleepTransformer:
─────────────────                   ──────────────────
  Kelime → Embedding                  Epoch → CNN Encoder
  (word2vec, 512-dim)                 (CNN1D_V3, 128-dim)
  
  Cümle = kelime dizisi               Gece = epoch dizisi
  [I, love, cats] (3 token)           [E₁, E₂, ..., E₂₅] (25 epoch)
  
  Encoder + Decoder                    SADECE Encoder
  (çeviri üret)                        (sınıflandır)
  
  Çıkış: yeni cümle                   Çıkış: her epoch için sınıf
  "J'aime les chats"                  [W, W, N1, N2, N2, ...]
```

### 4.2 Tam Mimari — 3 Aşama

```
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 1: CNN ENCODER (per-epoch, bağımsız)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Her 30-saniyelik PSG epoch'u → sabit boyutlu vektör

  Epoch verisi                         Feature vektörü
  [4 kanal × 7680 örnek]               [128 boyut]
  
  ┌─────────────────────┐    ┌───────────────────────────┐
  │ C3-M2  ~~~~~~~~~~~~ │    │                           │
  │ F4-M1  ~~~~~~~~~~~~ │──▶ │  CNN1D_V3 Feature Extract │──▶ [128]
  │ E1-M2  ~~~~~~~~~~~~ │    │  (MultiScale+Res+SE+GAP)  │
  │ O1-M2  ~~~~~~~~~~~~ │    │                           │
  └─────────────────────┘    └───────────────────────────┘
        30 saniye                    ~260K parametre

  Bu aşama 25 epoch için BAĞIMSIZ çalışır:
  Epoch  1 → CNN → [128]  ┐
  Epoch  2 → CNN → [128]  │
  Epoch  3 → CNN → [128]  │
  ...                     ├──▶ [25 × 128] matris
  Epoch 24 → CNN → [128]  │
  Epoch 25 → CNN → [128]  ┘

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 2: TRANSFORMER ENCODER (epoch'lar arası bağlam)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

25 epoch'un birbirine göre bağlamını öğren

         [25 × 128]
             │
             ▼
  ┌──────────────────────┐
  │  + Positional Enc.   │  ← Epoch sıra bilgisi ekle
  │    (sinusoidal)      │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Transformer Block 1 │  ← Self-Attention + FFN
  │  (4 head, 128→256)   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Transformer Block 2 │  ← Self-Attention + FFN
  │  (4 head, 128→256)   │
  └──────────┬───────────┘
             │
             ▼
  ┌──────────────────────┐
  │  Final LayerNorm     │
  └──────────┬───────────┘
             │
             ▼
  [25 × 128]  ← Her epoch artık bağlam bilgisi taşıyor

  Attention örneği (Epoch 13, N1 → tahmini):
  
  Epoch:   9   10   11   12  [13]  14   15   16   17
  Stage:   W    W    W   N1  [N1]  N2   N2   N2   N3
  Weight: .02  .04  .12  .28 [.20] .18  .10  .04  .02
                         ▲▲▲      ▲▲▲
                    "N1 komşum"  "N2'ye geçiş"
  
  → Model, 13. epoch'u skorlarken en çok 12 ve 14'e bakıyor

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STAGE 3: CLASSIFICATION HEAD (epoch başına tahmin)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Her epoch'un zenginleştirilmiş vektörünü 5 sınıfa dönüştür

         [25 × 128]
             │
             ▼
  ┌──────────────────────┐
  │  LayerNorm           │
  │  Dropout(0.3)        │
  │  Linear(128 → 5)     │  ← W, N1, N2, N3, REM
  └──────────┬───────────┘
             │
             ▼
  [25 × 5]  ← Her epoch için 5 sınıf logit

  Epoch  1 → [0.8, 0.1, 0.05, 0.03, 0.02] → W      ✓
  Epoch  2 → [0.7, 0.2, 0.05, 0.03, 0.02] → W      ✓
  ...
  Epoch 13 → [0.1, 0.4, 0.35, 0.05, 0.10] → N1     ✓ (bağlam sayesinde!)
  ...
  Epoch 25 → [0.0, 0.0, 0.10, 0.85, 0.05] → N3     ✓

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

### 4.3 Neden CNN + Transformer?

Saf Transformer (SleepTransformer, Phan 2022) ham EEG'yi doğrudan tokenize eder.
Biz bunun yerine CNN + Transformer hybrid kullanıyoruz:

```
Yaklaşım A: Saf Transformer (Phan 2022)
──────────────────────────────────────────
  Ham EEG [7680 sample] → 30 token'a böl (256 sample/token)
  → Her token'a linear embedding uygula
  → Intra-epoch Transformer (epoch içi attention)
  → Inter-epoch Transformer (epoch'lar arası attention)
  
  ⚠ Epoch içi 30 token: yüksek hesaplama maliyeti
  ⚠ Linear embedding: CNN kadar güçlü feature extraction değil

Yaklaşım B: CNN + Transformer (Bizim — SleepTransformer)
──────────────────────────────────────────────────────────
  Ham EEG [4ch × 7680] → CNN1D_V3 → [128] tek vektör
  → 25 epoch → [25 × 128]
  → Inter-epoch Transformer (epoch'lar arası attention)
  
  ✓ CNN1D_V3: kanıtlanmış feature extractor (multi-scale, residual, SE)
  ✓ Daha az token (25 vs 30×25=750): daha hafif, daha hızlı
  ✓ 4GB VRAM'de rahat çalışır
```

**Neden bu tercih:**
1. CNN1D_V3 zaten PSG sinyallerinden iyi feature çıkarıyor (tek epoch %67 acc)
2. Asıl eksik olan temporal context — transformer bunu ekliyor
3. 37 hasta ile saf transformer overfitting riski çok yüksek
4. Hybrid yaklaşım daha az parametre (~548K) ile daha verimli

---

## 5. Standart vs SleepTransformer — Farklar Tablosu

| Özellik | Standart Transformer (NLP) | SleepTransformer (Bizim) |
|---------|---------------------------|--------------------------|
| **Giriş tipi** | Kelime dizisi (metin) | PSG epoch dizisi (sinyal) |
| **Token** | Kelime (veya alt-kelime) | 30 saniyelik EEG epoch'u |
| **Embedding** | Word embedding (Word2Vec, BPE) | **CNN1D_V3** ile öğrenilen feature | 
| **Embedding boyutu** | 512 (base) / 768 (BERT) | **128** (VRAM optimize) |
| **Encoder-Decoder** | Her ikisi de var | **Sadece Encoder** |
| **Encoder katman sayısı** | 6 (base) / 12 (BERT) | **2** (küçük dataset, overfitting) |
| **Attention head sayısı** | 8 (base) | **4** |
| **FFN boyutu** | 2048 (base) | **256** (2×d_model) |
| **Sequence uzunluğu** | 512 (BERT) / 4096+ (GPT) | **25** (epoch sayısı) |
| **Positional Encoding** | Sinusoidal (orijinal), Learnable (BERT) | **Sinusoidal** (sonra learnable) |
| **Normalization** | Post-LN (orijinal) | **Pre-LN** (daha stabil training) |
| **Activation** | ReLU (orijinal) | **GELU** (modern standart) |
| **Çıkış** | Sequence-to-sequence (çeviri) | **Sequence-to-sequence (sınıflandırma)** |
| **Parametre sayısı** | 65M (base) / 110M (BERT) | **~548K** |
| **Training verisi** | Milyarlarca kelime | **~150K epoch (37 hasta)** |

### Neden bu kadar küçük?

```
Standart:  d_model=512, nhead=8, num_layers=6, d_ff=2048 → ~65M param
Bizim:     d_model=128, nhead=4, num_layers=2, d_ff=256  → ~267K param (sadece transformer)

Küçültme oranı: ~250×

Sebep: 37 hasta ≈ 150K epoch. 65M parametre bu veriyle eğitilemez.
       548K parametre bile dikkatli regularization gerektirir.
```

---

## 6. Neden LSTM Yerine Transformer?

Projede daha önce **DeepSleepNet (CNN+BiLSTM)** kullanıldı. Transformer'a geçiş nedenleri:

### 6.1 LSTM'in Sınırlılıkları

```
LSTM: Sıralı İşleme
─────────────────────

Epoch 1 → LSTM → h₁ → Epoch 2 → LSTM → h₂ → ... → Epoch 25 → LSTM → h₂₅
              ╰──────────╯              ╰─────╯                    ╰───▶ çıkış

Problem 1: Epoch 1'in bilgisi h₁ → h₂ → ... → h₂₅ boyunca "taşınmalı"
           Her adımda bilgi kaybı → uzun mesafede gradient vanishing
           
Problem 2: Sıralı hesaplama → paralelleştirilemez → yavaş

Problem 3: Bidirectional bile olsa, iki yönlü bilgi ayrı ayrı işlenir
```

### 6.2 Transformer'ın Avantajları

```
Transformer: Paralel İşleme (Self-Attention)
──────────────────────────────────────────────

Epoch  1 ←───attention──→ Epoch 13    (direkt bağlantı!)
Epoch  2 ←───attention──→ Epoch 13
           ...
Epoch 12 ←───attention──→ Epoch 13    (hepsi aynı anda!)
Epoch 14 ←───attention──→ Epoch 13
           ...
Epoch 25 ←───attention──→ Epoch 13

✓ Epoch 1 ve Epoch 25 arasında bile DOĞRUDAN bağlantı
✓ Tüm çiftler paralel hesaplanır → GPU friendly
✓ Gradient her epoch'a doğrudan ulaşır → vanishing yok
✓ Attention score'lar → hangi epoch'a dikkat edildiği GÖRÜNEBİLİR
```

### 6.3 Uyku Skorlama İçin Somut Avantaj

N1 evresi uyku tıbbında en zor sınıftır:

```
Tipik N1 Senaryosu:
   ... W  W  W  N1  N1  N2  N2  N2 ...
                ▲▲▲
         "Uyanıklıktan uykuya geçiş"

CNN (tek epoch): N1'i sadece kendi 30-saniyesine bakarak tahmin eder
                 → N1'in EEG paterni W ve N2'ye çok benzer → BAŞARISIZ

LSTM: N1'den önceki W dizisini "hatırlar", ama uzaklaştıkça unutur
      Bidirectional olsa bile iki yön bilgisi ayrı işlenir

Transformer: "Bu epoch'un hemen öncesinde 3 tane Wake var,
              hemen sonrasında N2 başlıyor.
              Geçiş paterni: büyük ihtimalle N1."
              
              → Tüm bağlamı EŞ ZAMANLI görür
              → W→N1→N2 geçiş kalıbını doğrudan öğrenebilir
```

### 6.4 Karşılaştırma Tablosu

| Özellik | CNN (tek epoch) | BiLSTM (DeepSleepNet) | Transformer (SleepTransformer) |
|---------|----------------|----------------------|-------------------------------|
| Temporal context | ❌ Yok | ✅ Sıralı (sequential) | ✅ **Global (tüm epoch'lar)** |
| Uzak epoch erişimi | ❌ | ⚠️ Gradient kaybı | ✅ **Direkt attention** |
| Paralelleştirme | ✅ Tam | ❌ Sıralı zorunlu | ✅ **Tam paralel** |
| Interpretability | ❌ Kara kutu | ❌ Hidden state opak | ✅ **Attention heatmap** |
| Stateful yönetim | ❌ Gerekmiyor | ⚠️ Hidden state reset | ✅ **Stateless (basit)** |
| N1 potansiyeli | Düşük | Orta | **Yüksek** |

---

## 7. Sık Sorulan Sorular

### "Neden Decoder yok?"

Decoder, **yeni bir dizi üretmek** için gereklidir (çeviri, metin üretme gibi).
Biz yeni bir dizi üretmiyoruz — her epoch'u **sınıflandırıyoruz**.
Bu bir classification görevi, generation değil. BERT gibi encoder-only yeterli.

### "Neden 2 katman? Daha fazla olsa daha iyi olmaz mı?"

37 hasta ≈ 150K epoch. Bu, GPT-3'ün eğitildiği verinin ~0.00001'i.
Fazla katman = fazla parametre = overfitting. Zaten CNN1D_V3 ile tek epoch'ta
%17 train-val gap gördük. Transformer parametreleri minimize tutuyoruz.

### "Positional Encoding neden gerekli?"

Self-attention sıra-agnostiktir — epoch'ların sırasını bilmez.
"Epoch 3'ten sonra Epoch 4 gelir" bilgisini açıkça vermemiz gerekir.
Sinusoidal PE matematiksel bir formülle sıra bilgisi kodlar.

### "CNN kısmı neden var? Doğrudan ham EEG verse olmaz mı?"

Olur (Phan 2022 bunu yapıyor), ama:
1. Ham EEG'den feature çıkarmak için çok daha fazla parametre gerekir
2. CNN1D_V3 zaten kanıtlanmış bir feature extractor
3. 37 hasta ile sıfırdan EEG embedding öğrenmek riskli
4. Hybrid yaklaşım daha az parametre ile daha verimli

### "Bu model PhysioNet verisiyle de çalışır mı?"

Evet. Tek fark:
- PSG data: 4 kanal, 256Hz → epoch = [4, 7680]
- PhysioNet ST: 2 kanal, 100Hz → epoch = [2, 3000]

CNN encoder zaten `config.py`'daki `SAMPLE_RATE` ve `N_CHANNELS`'a göre
otomatik ölçekleniyor. Transformer kısmı `d_model=128` ile aynı kalır.

### "Attention score'ları nasıl göreceğiz?"

Eğitim tamamlandıktan sonra test verisinde:
1. Model'den attention weight'lerini çıkar
2. Her epoch için hangi epoch'lara dikkat edildiğini göster
3. N1 epoch'larında attention pattern'ini analiz et

Bu, modelin "neden N1 dediğini" anlamamızı sağlar — klinik kabul edilebilirlik
için çok önemli bir özellik.

---

## Özet Şeması

```
┌───────────────────────────────────────────────────────────────┐
│                                                               │
│                    SleepTransformer                           │
│                                                               │
│    Ham PSG                CNN Encoder           Transformer   │
│   ┌───────┐         ┌──────────────────┐    ┌──────────────┐  │
│   │ 4 ch  │  epoch  │  MultiScaleStem  │    │  PosEnc      │  │
│   │ ×     │ ──×25──▶│  ResBlock ×2     │──▶ │  Attention   │  │
│   │ 7680  │  ayrı   │  SE + GAP        │    │  ×2 layer    │  │
│   │ sample│  CNN    │  → [128]         │    │  → [128]     │  │
│   └───────┘         └──────────────────┘    └──────┬───────┘  │
│                                                    │          │
│                                              ┌─────▼─────┐    │
│                                              │ LN + FC   │    │
│                                              │ → 5 class │    │
│                                              └─────┬─────┘    │
│                                                    │          │
│                                              ┌─────▼─────┐    │
│                                              │ W N1 N2   │    │
│                                              │ N3 REM    │    │
│                                              └───────────┘    │
│                                                               │
│       ~548K param │ 4 kanal │ 256Hz │ 25 epoch bağlam         │
└───────────────────────────────────────────────────────────────┘
```
