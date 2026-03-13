"""
Sleep Stage Classification - PyTorch Dataset (RAM Cache Version)
Tüm veriyi başlangıçta RAM'e yükler, sonra oradan okur.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pymongo import MongoClient
import gridfs
import random
import sys
import os
from tqdm import tqdm

# Kendi modüllerimizi import et
import config
from edf_to_mongo import get_cached_stages, _STAGES_CACHE, extract_epoch_range_data_v2


class SleepEpochDataset(Dataset):
    """
    Uyku epoch'larını MongoDB'den çeken PyTorch Dataset.
    RAM Cache version: Tüm veriyi başlangıçta yükler.
    
    Args:
        patient_ids: Hasta ID listesi
        mongo_uri: MongoDB URI
        db_name: Database adı
        channels: Kullanılacak kanal isimleri
        normalize: Normalizasyon yöntemi ('z-score', 'min-max', None)
        augment: Data augmentation kullan mı
        preload: True ise tüm veriyi RAM'e yükle
    """
    
    def __init__(self, patient_ids, mongo_uri=config.MONGO_URI,
                 db_name=config.DB_NAME, channels=config.SELECTED_CHANNELS,
                 normalize=config.NORMALIZATION, augment=False, preload=True,
                 balance_strategy=config.BALANCE_STRATEGY):
        
        self.patient_ids = patient_ids
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        self.channels = channels
        self.normalize = normalize
        self.augment = augment
        self.preload = preload
        self.balance_strategy = balance_strategy
        
        # Epoch metadata'sını oluştur
        self.epoch_metadata = []
        self._build_epoch_metadata()
        
        # Balanced sampling (stratejiye göre)
        if self.balance_strategy == 'undersample':
            self._undersample_epoch_metadata()
        elif self.balance_strategy == 'oversample':
            self._oversample_epoch_metadata()
        elif self.balance_strategy == 'none':
            print("\n⚠️ Balanced sampling kapalı - orijinal dağılım kullanılıyor\n")
        else:
            print(f"\n⚠️ Bilinmeyen balance_strategy: {self.balance_strategy}\n")
        
        # RAM Cache (preload=True ise)
        self.signal_cache = {}
        if self.preload:
            self._preload_all_signals()
        
        print(f"✓ Dataset oluşturuldu: {len(self.epoch_metadata)} epoch, "
              f"{len(patient_ids)} hasta")
    
    def _build_epoch_metadata(self):
        """
        Tüm hastaların epoch bilgilerini topla.
        Her epoch için: (patient_id, epoch_idx, stage_label) sakla.
        """
        print(f"\n{'='*70}")
        print(f"EPOCH METADATA OLUŞTURULUYOR")
        print(f"{'='*70}")
        
        for patient_id in self.patient_ids:
            # Cache'den stage verilerini al
            stages_data = get_cached_stages(patient_id, self.mongo_uri, self.db_name)
            
            if stages_data is None:
                print(f"⚠ {patient_id}: Stage verisi bulunamadı, atlanıyor")
                continue
            
            # Her epoch için metadata ekle
            for epoch_data in stages_data:
                epoch_idx = epoch_data['epoch']
                stage = epoch_data['stage']
                
                # Sadece bilinen stage'leri al
                if stage in config.CLASS_TO_IDX:
                    self.epoch_metadata.append({
                        'patient_id': patient_id,
                        'epoch_idx': epoch_idx,
                        'stage': stage,
                        'label': config.CLASS_TO_IDX[stage]
                    })
            
            print(f"  {patient_id}: {len(stages_data)} epoch eklendi")
        
        print(f"\n{'='*70}")
        print(f"TOPLAM: {len(self.epoch_metadata)} epoch")
        
        # Sınıf dağılımını yazdır
        self._print_class_distribution()
        print(f"{'='*70}\n")
    
    def _print_class_distribution(self):
        """
        Sınıf dağılımını yazdır
        """
        class_counts = {}
        for meta in self.epoch_metadata:
            stage = meta['stage']
            class_counts[stage] = class_counts.get(stage, 0) + 1
        
        total = len(self.epoch_metadata)
        print(f"\nSınıf dağılımı:")
        for stage in config.CLASS_NAMES:
            count = class_counts.get(stage, 0)
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  {stage}: {count} epoch ({percentage:.1f}%)")
    
    def _undersample_epoch_metadata(self):
        """
        Her sınıftan eşit sayıda epoch seç (undersampling).
        Minimum sınıf sayısı kadar her sınıftan random sampling yapar.
        """
        import random
        
        print(f"\n{'='*70}")
        print(f"UNDERSAMPLING UYGULANLIYOR")
        print(f"{'='*70}")
        
        # Sınıflara göre epoch'ları grupla
        class_epochs = {class_name: [] for class_name in config.CLASS_NAMES}
        
        for meta in self.epoch_metadata:
            stage = meta['stage']
            class_epochs[stage].append(meta)
        
        # Her sınıftaki epoch sayısını yazdır
        print("\nMevcut dağılım:")
        for stage in config.CLASS_NAMES:
            print(f"  {stage}: {len(class_epochs[stage])} epoch")
        
        # Minimum sınıf sayısını bul
        min_count = min(len(epochs) for epochs in class_epochs.values())
        
        if min_count == 0:
            print("\n⚠️ Uyarı: En az bir sınıfta epoch yok! Balanced sampling uygulanamıyor.")
            return
        
        print(f"\nMinimum sınıf sayısı: {min_count} epoch")
        print(f"Her sınıftan {min_count} epoch seçilecek...")
        
        # Her sınıftan min_count kadar random seç
        balanced_metadata = []
        
        for stage in config.CLASS_NAMES:
            # Random sampling
            selected = random.sample(class_epochs[stage], min_count)
            balanced_metadata.extend(selected)
        
        # Metadata'yı güncelle
        original_count = len(self.epoch_metadata)
        self.epoch_metadata = balanced_metadata
        
        # Karıştır (sınıflar karışık olsun)
        random.shuffle(self.epoch_metadata)
        
        print(f"\n✓ Undersampling tamamlandı!")
        print(f"  Önceki toplam: {original_count} epoch")
        print(f"  Yeni toplam: {len(self.epoch_metadata)} epoch")
        print(f"  Her sınıf: {min_count} epoch")
        
        # Yeni dağılımı yazdır
        self._print_class_distribution()
        print(f"{'='*70}\n")
    
    def _oversample_epoch_metadata(self):
        """
        Her sınıftan maksimum sınıf sayısı kadar epoch seç (oversampling).
        Azınlık sınıflarını random sampling ile çoğaltır.
        """
        import random
        
        print(f"\n{'='*70}")
        print(f"OVERSAMPLING UYGULANLIYOR")
        print(f"{'='*70}")
        
        # Sınıflara göre epoch'ları grupla
        class_epochs = {class_name: [] for class_name in config.CLASS_NAMES}
        
        for meta in self.epoch_metadata:
            stage = meta['stage']
            class_epochs[stage].append(meta)
        
        # Her sınıftaki epoch sayısını yazdır
        print("\nMevcut dağılım:")
        for stage in config.CLASS_NAMES:
            print(f"  {stage}: {len(class_epochs[stage])} epoch")
        
        # Maksimum sınıf sayısını bul
        max_count = max(len(epochs) for epochs in class_epochs.values())
        
        if max_count == 0:
            print("\n⚠️ Uyarı: Hiç epoch yok! Oversampling uygulanamıyor.")
            return
        
        print(f"\nMaksimum sınıf sayısı: {max_count} epoch")
        print(f"Her sınıftan {max_count} epoch olacak şekilde çoğaltılıyor...")
        
        # Her sınıftan max_count kadar seç (tekrarlı sampling)
        oversampled_metadata = []
        
        for stage in config.CLASS_NAMES:
            current_epochs = class_epochs[stage]
            current_count = len(current_epochs)
            
            if current_count == 0:
                print(f"\n⚠️ Uyarı: {stage} sınıfında epoch yok!")
                continue
            
            # Random sampling with replacement (tekrarlı)
            selected = random.choices(current_epochs, k=max_count)
            oversampled_metadata.extend(selected)
            
            # Kaç kez tekrarlandığını göster
            duplication_factor = max_count / current_count
            print(f"  {stage}: {current_count} → {max_count} epoch (x{duplication_factor:.2f})")
        
        # Metadata'yı güncelle
        original_count = len(self.epoch_metadata)
        self.epoch_metadata = oversampled_metadata
        
        # Karıştır (sınıflar karışık olsun)
        random.shuffle(self.epoch_metadata)
        
        print(f"\n✓ Oversampling tamamlandı!")
        print(f"  Önceki toplam: {original_count} epoch")
        print(f"  Yeni toplam: {len(self.epoch_metadata)} epoch")
        print(f"  Her sınıf: {max_count} epoch")
        print(f"  Artış oranı: x{len(self.epoch_metadata) / original_count:.2f}")
        
        # Yeni dağılımı yazdır
        self._print_class_distribution()
        print(f"{'='*70}\n")
    
    def _preload_all_signals(self):
        """
        Tüm epoch'ları MongoDB'den çekip RAM'e yükle.
        Cache format: {(patient_id, epoch_idx, channel_name): signal_array}
        """
        print(f"\n{'='*70}")
        print(f"RAM CACHE OLUŞTURULUYOR - TÜM VERİLER YÜKLENİYOR")
        print(f"{'='*70}")
        print(f"Toplam epoch: {len(self.epoch_metadata)}")
        print(f"Kanal sayısı: {len(self.channels)}")
        print(f"Tahmini boyut: ~{len(self.epoch_metadata) * len(self.channels) * 7680 * 4 / (1024**3):.2f} GB")
        print(f"{'='*70}\n")
        
        # STDOUT'u kapat (print bastırma)
        original_stdout = sys.stdout
        
        # Progress bar ile yükleme
        total_loads = len(self.epoch_metadata) * len(self.channels)
        
        with tqdm(total=total_loads, desc="Veri yükleniyor", unit="epoch-kanal") as pbar:
            for meta in self.epoch_metadata:
                patient_id = meta['patient_id']
                epoch_idx = meta['epoch_idx']
                
                for channel_name in self.channels:
                    cache_key = (patient_id, epoch_idx, channel_name)
                    
                    # Zaten yüklenmişse atla
                    if cache_key in self.signal_cache:
                        pbar.update(1)
                        continue
                    
                    # Print'leri bastır
                    sys.stdout = open(os.devnull, 'w')
                    
                    try:
                        # MongoDB'den çek
                        result = extract_epoch_range_data_v2(
                            patient_id=patient_id,
                            channel_name=channel_name,
                            start_epoch=epoch_idx,
                            end_epoch=epoch_idx,
                            mongo_uri=self.mongo_uri,
                            db_name=self.db_name,
                            save_to_file=False
                        )
                        
                        # Stdout'u geri yükle
                        sys.stdout = original_stdout
                        
                        if result is None:
                            signal = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
                        else:
                            signal = result['signal_data'].astype(np.float32)
                            
                            # Boyut kontrolü
                            if len(signal) != config.SAMPLES_PER_EPOCH:
                                if len(signal) < config.SAMPLES_PER_EPOCH:
                                    signal = np.pad(signal, 
                                                   (0, config.SAMPLES_PER_EPOCH - len(signal)),
                                                   mode='constant')
                                else:
                                    signal = signal[:config.SAMPLES_PER_EPOCH]
                        
                        # Cache'e ekle
                        self.signal_cache[cache_key] = signal
                        
                    except Exception as e:
                        # Hata durumunda stdout'u geri yükle
                        sys.stdout = original_stdout
                        print(f"\n⚠ Hata: {patient_id}, epoch {epoch_idx}, {channel_name}: {str(e)}")
                        # Sıfırlarla doldur
                        self.signal_cache[cache_key] = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
                    
                    pbar.update(1)
        
        # Bellek kullanımını hesapla
        total_bytes = sum(sig.nbytes for sig in self.signal_cache.values())
        total_gb = total_bytes / (1024**3)
        
        print(f"\n{'='*70}")
        print(f"✓ RAM CACHE HAZIR!")
        print(f"{'='*70}")
        print(f"Cache'deki veri sayısı: {len(self.signal_cache):,}")
        print(f"Toplam bellek kullanımı: {total_gb:.2f} GB")
        print(f"{'='*70}\n")
    
    def __len__(self):
        """
        Dataset boyutu
        """
        return len(self.epoch_metadata)
    
    def __getitem__(self, idx):
        """
        Bir epoch'u cache'den al ve döndür.
        
        Returns:
            signal: Tensor [n_channels, samples_per_epoch]
            label: int (0-4 arası sınıf indeksi)
        """
        # Metadata'dan bilgileri al
        meta = self.epoch_metadata[idx]
        patient_id = meta['patient_id']
        epoch_idx = meta['epoch_idx']
        label = meta['label']
        
        # Kanalları cache'den al
        channel_signals = []
        
        for channel_name in self.channels:
            cache_key = (patient_id, epoch_idx, channel_name)
            
            if self.preload:
                # Cache'den al
                signal = self.signal_cache.get(cache_key)
                if signal is None:
                    # Yoksa sıfırlarla doldur
                    signal = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
            else:
                # Preload kapalıysa, on-the-fly yükle
                signal = self._load_signal_on_the_fly(patient_id, epoch_idx, channel_name)
            
            channel_signals.append(signal)
        
        # Tüm kanalları stack'le: [n_channels, samples]
        signal = np.stack(channel_signals, axis=0)
        
        # Bandpass filtering (DC drift ve gürültüyü kaldır)
        if config.USE_FILTERING:
            signal = self._bandpass_filter(signal)
        
        # Amplitude clipping (artifact removal - kurtosis'i düşürür)
        if config.USE_AMPLITUDE_CLIPPING:
            signal = self._amplitude_clip(signal)
        
        # Normalizasyon
        if self.normalize == 'z-score':
            signal = self._normalize_zscore(signal)
        elif self.normalize == 'min-max':
            signal = self._normalize_minmax(signal)
        
        # Augmentation (eğer training modundaysa)
        if self.augment:
            signal = self._augment_signal(signal)
        
        # Tensor'a çevir
        signal = torch.from_numpy(signal).float()
        label = torch.tensor(label, dtype=torch.long)
        
        return signal, label
    
    def _load_signal_on_the_fly(self, patient_id, epoch_idx, channel_name):
        """
        Cache kapalıysa MongoDB'den direkt yükle
        """
        # Print'leri bastır
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            result = extract_epoch_range_data_v2(
                patient_id=patient_id,
                channel_name=channel_name,
                start_epoch=epoch_idx,
                end_epoch=epoch_idx,
                mongo_uri=self.mongo_uri,
                db_name=self.db_name,
                save_to_file=False
            )
            
            sys.stdout = original_stdout
            
            if result is None:
                return np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
            
            signal = result['signal_data'].astype(np.float32)
            
            if len(signal) != config.SAMPLES_PER_EPOCH:
                if len(signal) < config.SAMPLES_PER_EPOCH:
                    signal = np.pad(signal, 
                                   (0, config.SAMPLES_PER_EPOCH - len(signal)),
                                   mode='constant')
                else:
                    signal = signal[:config.SAMPLES_PER_EPOCH]
            
            return signal
            
        except Exception as e:
            sys.stdout = original_stdout
            return np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
    
    def _normalize_zscore(self, signal):
        """
        Z-score normalization (per-channel, per-epoch)
        
        signal: [n_channels, samples]
        """
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        
        # Sıfıra bölme hatası için epsilon ekle
        std = np.where(std < config.NORM_EPSILON, config.NORM_EPSILON, std)
        
        signal = (signal - mean) / std
        return signal
    
    def _normalize_minmax(self, signal):
        """
        Min-Max normalization [0, 1]
        
        signal: [n_channels, samples]
        """
        min_val = signal.min(axis=1, keepdims=True)
        max_val = signal.max(axis=1, keepdims=True)
        
        # Sıfıra bölme hatası için
        range_val = max_val - min_val
        range_val = np.where(range_val < config.NORM_EPSILON, 
                             config.NORM_EPSILON, range_val)
        
        signal = (signal - min_val) / range_val
        return signal
    
    def _amplitude_clip(self, signal):
        """
        Amplitude clipping to remove extreme artifacts.
        Clips signal to percentile range (e.g., p1-p99) per channel.
        
        This significantly reduces kurtosis (2000 → 2) by removing
        extreme spikes from movement/electrode artifacts while
        preserving 98% of the signal.
        
        signal: [n_channels, samples]
        """
        for i in range(signal.shape[0]):
            low = np.percentile(signal[i], config.CLIP_PERCENTILE_LOW)
            high = np.percentile(signal[i], config.CLIP_PERCENTILE_HIGH)
            signal[i] = np.clip(signal[i], low, high)
        return signal
    
    def _bandpass_filter(self, signal, lowcut=None, highcut=None, fs=None, order=None):
        """
        Butterworth bandpass filter
        DC drift ve yüksek frekanslı gürültüyü kaldırır.
        
        Args:
            signal: 1D veya 2D numpy array [samples] veya [channels, samples]
            lowcut: Alt kesim frekansı (Hz)
            highcut: Üst kesim frekansı (Hz)
            fs: Sampling frequency (Hz)
            order: Filtre derecesi
        
        Returns:
            Filtered signal (same shape as input)
        """
        if lowcut is None:
            lowcut = config.LOWCUT
        if highcut is None:
            highcut = config.HIGHCUT
        if fs is None:
            fs = config.SAMPLE_RATE
        if order is None:
            order = config.FILTER_ORDER
        
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        
        # Butterworth bandpass filter katsayıları
        b, a = butter(order, [low, high], btype='band')
        
        # 2D ise her kanal için ayrı filtrele
        if signal.ndim == 2:
            filtered = np.zeros_like(signal)
            for i in range(signal.shape[0]):
                filtered[i] = filtfilt(b, a, signal[i])
            return filtered
        else:
            return filtfilt(b, a, signal)
    
    def _augment_signal(self, signal):
        """
        Data augmentation (training için)
        
        - Gaussian noise ekleme
        - Time shift
        - Amplitude scaling
        """
        if not config.USE_AUGMENTATION:
            return signal
        
        # Gaussian noise
        if random.random() > 0.5:
            noise = np.random.normal(0, config.AUG_NOISE_STD, signal.shape)
            signal = signal + noise
        
        # Time shift
        if random.random() > 0.5:
            shift = random.randint(-config.AUG_TIME_SHIFT_MAX, 
                                  config.AUG_TIME_SHIFT_MAX)
            signal = np.roll(signal, shift, axis=1)
        
        # Amplitude scaling
        if random.random() > 0.5:
            scale = random.uniform(*config.AUG_AMPLITUDE_SCALE_RANGE)
            signal = signal * scale
        
        return signal
    
    def get_class_weights(self):
        """
        Dengesiz sınıflar için class weight'leri hesapla.
        
        Returns:
            torch.Tensor: [n_classes] boyutunda weight'ler
        """
        class_counts = np.zeros(config.N_CLASSES)
        
        for meta in self.epoch_metadata:
            label = meta['label']
            class_counts[label] += 1
        
        # Inverse frequency
        total = len(self.epoch_metadata)
        class_weights = total / (config.N_CLASSES * class_counts + 1e-6)
        
        return torch.from_numpy(class_weights).float()
    
    def clear_cache(self):
        """
        RAM cache'i temizle (bellek boşaltma)
        """
        self.signal_cache.clear()
        print("✓ RAM cache temizlendi")

def split_patients(all_patient_ids, n_train, n_val, n_test, seed=None, run_dir=None):
    """
    Hastaları train/val/test setlerine böl (RANDOM - her run için farklı)
    
    Args:
        all_patient_ids: Tüm hasta ID'leri
        n_train: Train set hasta sayısı
        n_val: Validation set hasta sayısı
        n_test: Test set hasta sayısı
        seed: Random seed (None ise her run'da farklı split)
        run_dir: Split'in kaydedileceği run directory
    
    Returns:
        train_ids, val_ids, test_ids
    """
    import random
    import json
    import os
    
    # Random seed ayarla (None ise timestamp-based random)
    if seed is not None:
        random.seed(seed)
    else:
        # Her run için farklı seed (timestamp-based)
        import time
        random.seed(int(time.time() * 1000) % (2**32))
    
    # Hasta listesini karıştır
    patient_list = list(all_patient_ids)
    random.shuffle(patient_list)
    
    # Toplam hasta sayısını kontrol et
    total_needed = n_train + n_val + n_test
    if len(patient_list) < total_needed:
        raise ValueError(
            f"Yetersiz hasta sayısı! "
            f"Gerekli: {total_needed} (train:{n_train} + val:{n_val} + test:{n_test}), "
            f"Mevcut: {len(patient_list)}"
        )
    
    # Split yap
    train_ids = patient_list[:n_train]
    val_ids = patient_list[n_train:n_train + n_val]
    test_ids = patient_list[n_train + n_val:n_train + n_val + n_test]
    
    print(f"\n{'='*70}")
    print(f"PATIENT SPLIT (RANDOM)")
    print(f"{'='*70}")
    print(f"Toplam hasta: {len(patient_list)}")
    print(f"Train: {len(train_ids)} hasta")
    print(f"Val:   {len(val_ids)} hasta")
    print(f"Test:  {len(test_ids)} hasta")
    print(f"Seed:  {'Random (timestamp-based)' if seed is None else seed}")
    print(f"{'='*70}\n")
    
    # Split'i kaydet (run directory'ye)
    if run_dir:
        split_data = {
            'train': train_ids,
            'val': val_ids,
            'test': test_ids,
            'seed': seed,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        split_file = os.path.join(run_dir, 'patient_split.json')
        with open(split_file, 'w') as f:
            json.dump(split_data, f, indent=2)
        
        print(f"✓ Split kaydedildi: {split_file}\n")
    
    return train_ids, val_ids, test_ids

def create_dataloaders(train_ids, val_ids, test_ids, preload=True):
    """
    Train/Val/Test DataLoader'larını oluştur.
    
    Args:
        train_ids: Train hasta ID'leri
        val_ids: Validation hasta ID'leri
        test_ids: Test hasta ID'leri
        preload: True ise RAM cache kullan
    
    Returns:
        train_loader, val_loader, test_loader
    """
    print(f"\n{'='*70}")
    print(f"DATALOADER'LAR OLUŞTURULUYOR")
    print(f"{'='*70}")
    print(f"RAM Cache: {'Aktif ✓' if preload else 'Pasif'}\n")
    
    print("train_dataset başlıyor")

    # Dataset'leri oluştur
    train_dataset = SleepEpochDataset(
        patient_ids=train_ids,
        normalize=config.NORMALIZATION,
        augment=config.USE_AUGMENTATION,
        preload=preload
    )

    print("train_dataset bitti")
    
    val_dataset = SleepEpochDataset(
        patient_ids=val_ids,
        normalize=config.NORMALIZATION,
        augment=False,
        preload=preload,
        balance_strategy='none'  # Val set için oversampling kapalı (gerçek dağılım)
    )

    print("val_dataset bitti")
    
    test_dataset = SleepEpochDataset(
        patient_ids=test_ids,
        normalize=config.NORMALIZATION,
        augment=False,
        preload=preload,
        balance_strategy='none'  # Test set için oversampling kapalı (gerçek dağılım)
    )

    print("test_dataset bitti")
    
    
    # DataLoader'ları oluştur
    # Preload aktifse num_workers=0 yapabiliriz (zaten bellekte)
    num_workers = 0 if preload else config.NUM_WORKERS
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if not preload else False,
        persistent_workers=False
    )

    print("train_loader bitti")
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if not preload else False,
        persistent_workers=False
    )

    print("val_loader bitti")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=config.PIN_MEMORY if not preload else False,
        persistent_workers=False
    )

    print("test_loader bitti")
    
    print(f"✓ DataLoader'lar hazır!")
    print(f"  Train: {len(train_loader)} batch")
    print(f"  Val:   {len(val_loader)} batch")
    print(f"  Test:  {len(test_loader)} batch")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Sequence Dataset for DeepSleepNet (BiLSTM)
# ============================================================================
class SleepSequenceDataset(Dataset):
    """
    Sequence-based dataset for DeepSleepNet with BiLSTM.
    Returns consecutive epochs for temporal learning.
    
    BiLSTM sayesinde model şu geçişleri öğrenir:
    - Wake → N1 → N2 (uykuya dalış)
    - N2 → N1 → REM (REM döngüsü)
    - N3 → N2 → N1 (uyanma)
    
    Her sample: sequence_length kadar ardışık epoch içerir.
    Label: Her epoch için ayrı label (sequence-to-sequence learning)
    
    Args:
        patient_ids: Hasta ID listesi
        sequence_length: Ardışık epoch sayısı (default: 5 for 4GB VRAM)
        stride: Kaç epoch atlayarak ilerle (default: 1)
        channels: Kullanılacak kanal isimleri (None = config'den al)
        normalize: Normalization method ('z-score', 'min-max', None)
        augment: Data augmentation uygula mı
        preload: Tüm veriyi RAM'e yükle
    """
    
    def __init__(self, patient_ids, sequence_length=None, stride=None,
                 mongo_uri=config.MONGO_URI, db_name=config.DB_NAME,
                 channels=None, normalize=config.NORMALIZATION,
                 augment=False, preload=True):
        
        self.patient_ids = patient_ids
        self.sequence_length = sequence_length if sequence_length else config.SEQUENCE_LENGTH
        self.stride = stride if stride is not None else config.SEQUENCE_STRIDE
        self.mongo_uri = mongo_uri
        self.db_name = db_name
        
        # Kanal seçimi: Single channel mode için ACTIVE_CHANNELS kullan
        if channels is None:
            # config.ACTIVE_CHANNELS tanımlı mı kontrol et
            if hasattr(config, 'ACTIVE_CHANNELS'):
                self.channels = config.ACTIVE_CHANNELS
            else:
                self.channels = config.SELECTED_CHANNELS
        else:
            self.channels = channels
            
        self.normalize = normalize
        self.augment = augment
        self.preload = preload
        
        # Hasta bazlı epoch verileri
        self.patient_epochs = {}  # {patient_id: [epoch_metadata_list]}
        self._build_patient_epochs()
        
        # Sequence metadata (hangi hasta, başlangıç indeksi)
        self.sequence_metadata = []
        self._build_sequence_metadata()
        
        # RAM Cache
        self.signal_cache = {}
        if self.preload:
            self._preload_all_signals()
        
        print(f"\n{'='*70}")
        print(f"✓ SEQUENCE DATASET OLUŞTURULDU")
        print(f"{'='*70}")
        print(f"  Toplam sequence: {len(self.sequence_metadata)}")
        print(f"  Sequence length: {self.sequence_length} epoch")
        print(f"  Stride: {self.stride}")
        print(f"  Kanallar: {self.channels}")
        print(f"  Hasta sayısı: {len(patient_ids)}")
        print(f"{'='*70}\n")
    
    def _build_patient_epochs(self):
        """Her hasta için epoch listesi oluştur (sıralı)."""
        print(f"\n{'='*70}")
        print(f"SEQUENCE DATASET: EPOCH VERİLERİ YÜKLENİYOR")
        print(f"{'='*70}")
        
        for patient_id in self.patient_ids:
            stages_data = get_cached_stages(patient_id, self.mongo_uri, self.db_name)
            
            if stages_data is None:
                print(f"⚠ {patient_id}: Stage verisi bulunamadı, atlanıyor")
                continue
            
            # Epoch'ları epoch numarasına göre sırala
            sorted_epochs = sorted(stages_data, key=lambda x: x['epoch'])
            
            # Sadece geçerli stage'leri al
            valid_epochs = []
            for epoch_data in sorted_epochs:
                stage = epoch_data['stage']
                if stage in config.CLASS_TO_IDX:
                    valid_epochs.append({
                        'patient_id': patient_id,
                        'epoch_idx': epoch_data['epoch'],
                        'stage': stage,
                        'label': config.CLASS_TO_IDX[stage]
                    })
            
            if len(valid_epochs) > 0:
                self.patient_epochs[patient_id] = valid_epochs
                print(f"  {patient_id}: {len(valid_epochs)} epoch")
        
        total_epochs = sum(len(epochs) for epochs in self.patient_epochs.values())
        print(f"\nToplam: {total_epochs} epoch, {len(self.patient_epochs)} hasta")
        print(f"{'='*70}\n")
    
    def _build_sequence_metadata(self):
        """Geçerli sequence başlangıç noktalarını belirle."""
        print(f"\n{'='*70}")
        print(f"SEQUENCE METADATA OLUŞTURULUYOR")
        print(f"{'='*70}")
        print(f"Sequence length: {self.sequence_length}, Stride: {self.stride}")
        
        for patient_id, epochs in self.patient_epochs.items():
            n_epochs = len(epochs)
            
            # Yeterli epoch yoksa atla
            if n_epochs < self.sequence_length:
                print(f"⚠ {patient_id}: Yetersiz epoch ({n_epochs} < {self.sequence_length}), atlanıyor")
                continue
            
            # Stride ile sequence başlangıç noktaları
            n_sequences = (n_epochs - self.sequence_length) // self.stride + 1
            
            for i in range(0, n_epochs - self.sequence_length + 1, self.stride):
                self.sequence_metadata.append({
                    'patient_id': patient_id,
                    'start_idx': i,  # patient_epochs listesindeki başlangıç indeksi
                })
            
            print(f"  {patient_id}: {n_sequences} sequence")
        
        # STATEFUL LSTM: Shuffle YOK - hastalar sıralı işlenir
        # Hastalar zaten train başında random seçildi
        # random.shuffle(self.sequence_metadata)  # KALDIRILDI
        
        print(f"\nToplam: {len(self.sequence_metadata)} sequence")
        print(f"(Stateful LSTM: Shuffle kapalı - hastalar sıralı işlenecek)")
        print(f"{'='*70}\n")
    
    def _preload_all_signals(self):
        """Tüm epoch sinyallerini RAM'e yükle."""
        print(f"\n{'='*70}")
        print(f"RAM CACHE OLUŞTURULUYOR (SEQUENCE MODE)")
        print(f"{'='*70}")
        
        # Tüm unique (patient_id, epoch_idx, channel) kombinasyonlarını bul
        all_keys = set()
        for patient_id, epochs in self.patient_epochs.items():
            for epoch_data in epochs:
                for channel in self.channels:
                    all_keys.add((patient_id, epoch_data['epoch_idx'], channel))
        
        print(f"Yüklenecek toplam: {len(all_keys):,} sinyal")
        
        # Tahmini boyut
        est_size_gb = len(all_keys) * config.SAMPLES_PER_EPOCH * 4 / (1024**3)
        print(f"Tahmini boyut: ~{est_size_gb:.2f} GB")
        
        original_stdout = sys.stdout
        
        with tqdm(total=len(all_keys), desc="Veri yükleniyor", unit="sinyal") as pbar:
            for patient_id, epoch_idx, channel in all_keys:
                cache_key = (patient_id, epoch_idx, channel)
                
                if cache_key in self.signal_cache:
                    pbar.update(1)
                    continue
                
                sys.stdout = open(os.devnull, 'w')
                
                try:
                    result = extract_epoch_range_data_v2(
                        patient_id=patient_id,
                        channel_name=channel,
                        start_epoch=epoch_idx,
                        end_epoch=epoch_idx,
                        mongo_uri=self.mongo_uri,
                        db_name=self.db_name,
                        save_to_file=False
                    )
                    
                    sys.stdout = original_stdout
                    
                    if result is None:
                        signal = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
                    else:
                        signal = result['signal_data'].astype(np.float32)
                        
                        # Boyut kontrolü ve düzeltme
                        if len(signal) != config.SAMPLES_PER_EPOCH:
                            if len(signal) < config.SAMPLES_PER_EPOCH:
                                signal = np.pad(signal, (0, config.SAMPLES_PER_EPOCH - len(signal)))
                            else:
                                signal = signal[:config.SAMPLES_PER_EPOCH]
                    
                    self.signal_cache[cache_key] = signal
                    
                except Exception as e:
                    sys.stdout = original_stdout
                    self.signal_cache[cache_key] = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
                
                pbar.update(1)
        
        # Gerçek bellek kullanımı
        total_gb = sum(s.nbytes for s in self.signal_cache.values()) / (1024**3)
        print(f"\n✓ RAM Cache hazır: {total_gb:.2f} GB ({len(self.signal_cache):,} sinyal)")
        print(f"{'='*70}\n")
    
    def __len__(self):
        return len(self.sequence_metadata)
    
    def __getitem__(self, idx):
        """
        Returns:
            signals: Tensor [sequence_length, n_channels, samples_per_epoch]
            labels: Tensor [sequence_length] - her epoch için label
        """
        meta = self.sequence_metadata[idx]
        patient_id = meta['patient_id']
        start_idx = meta['start_idx']
        
        epochs = self.patient_epochs[patient_id]
        
        signals_list = []
        labels_list = []
        
        for i in range(self.sequence_length):
            epoch_data = epochs[start_idx + i]
            epoch_idx = epoch_data['epoch_idx']
            label = epoch_data['label']
            
            # Her kanal için sinyal al
            channel_signals = []
            for channel in self.channels:
                cache_key = (patient_id, epoch_idx, channel)
                
                if self.preload:
                    signal = self.signal_cache.get(cache_key)
                    if signal is None:
                        signal = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
                else:
                    signal = self._load_signal_on_the_fly(patient_id, epoch_idx, channel)
                
                channel_signals.append(signal)
            
            # Stack channels: [n_channels, samples]
            epoch_signal = np.stack(channel_signals, axis=0)
            
            # Normalize per-epoch
            if self.normalize == 'z-score':
                epoch_signal = self._normalize_zscore(epoch_signal)
            elif self.normalize == 'min-max':
                epoch_signal = self._normalize_minmax(epoch_signal)
            
            signals_list.append(epoch_signal)
            labels_list.append(label)
        
        # Stack epochs: [sequence_length, n_channels, samples]
        signals = np.stack(signals_list, axis=0)
        labels = np.array(labels_list, dtype=np.int64)
        
        return torch.from_numpy(signals).float(), torch.from_numpy(labels).long()
    
    def _load_signal_on_the_fly(self, patient_id, epoch_idx, channel_name):
        """Cache kapalıysa MongoDB'den direkt yükle."""
        original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        
        try:
            result = extract_epoch_range_data_v2(
                patient_id=patient_id,
                channel_name=channel_name,
                start_epoch=epoch_idx,
                end_epoch=epoch_idx,
                mongo_uri=self.mongo_uri,
                db_name=self.db_name,
                save_to_file=False
            )
            
            sys.stdout = original_stdout
            
            if result is None:
                return np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
            
            signal = result['signal_data'].astype(np.float32)
            
            if len(signal) != config.SAMPLES_PER_EPOCH:
                if len(signal) < config.SAMPLES_PER_EPOCH:
                    signal = np.pad(signal, (0, config.SAMPLES_PER_EPOCH - len(signal)))
                else:
                    signal = signal[:config.SAMPLES_PER_EPOCH]
            
            return signal
            
        except Exception:
            sys.stdout = original_stdout
            return np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
    
    def _normalize_zscore(self, signal):
        """Z-score normalization (per-channel)."""
        mean = signal.mean(axis=1, keepdims=True)
        std = signal.std(axis=1, keepdims=True)
        std = np.where(std < config.NORM_EPSILON, config.NORM_EPSILON, std)
        return (signal - mean) / std
    
    def _normalize_minmax(self, signal):
        """Min-Max normalization [0, 1] (per-channel)."""
        min_val = signal.min(axis=1, keepdims=True)
        max_val = signal.max(axis=1, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val < config.NORM_EPSILON, config.NORM_EPSILON, range_val)
        return (signal - min_val) / range_val
    
    def get_class_distribution(self):
        """Sınıf dağılımını hesapla ve döndür."""
        class_counts = {name: 0 for name in config.CLASS_NAMES}
        
        for patient_id, epochs in self.patient_epochs.items():
            for epoch in epochs:
                class_counts[epoch['stage']] += 1
        
        total = sum(class_counts.values())
        distribution = {}
        for stage, count in class_counts.items():
            pct = (count / total * 100) if total > 0 else 0
            distribution[stage] = {'count': count, 'percentage': pct}
        
        return distribution
    
    def get_class_weights(self):
        """
        Class imbalance için ağırlıklar hesapla.
        Inverse frequency weighting kullanır.
        
        Returns:
            torch.Tensor: [n_classes] boyutunda class weights
        """
        # Tüm epoch'ların label'larını say
        class_counts = np.zeros(config.N_CLASSES)
        
        for patient_id, epochs in self.patient_epochs.items():
            for epoch in epochs:
                label = epoch['label']
                class_counts[label] += 1
        
        # Total samples
        total = sum(class_counts)
        
        # Inverse frequency weighting: weight = total / (n_classes * count)
        # Adds epsilon to prevent division by zero
        class_weights = total / (config.N_CLASSES * class_counts + 1e-6)
        
        return torch.from_numpy(class_weights).float()
    
    def iterate_by_patient(self):
        """
        Stateful LSTM için hasta-bazlı iterasyon.
        
        Her hasta için sıralı sequence'ler döndürür.
        Yeni hastaya geçişte 'is_new_patient' flag'i True olur.
        
        Yields:
            dict: {
                'patient_id': str,
                'is_new_patient': bool,  # True ise LSTM state reset edilmeli
                'signals': Tensor [1, seq_len, n_channels, samples],
                'labels': Tensor [1, seq_len]
            }
        """
        for patient_id in self.patient_ids:
            if patient_id not in self.patient_epochs:
                continue
            
            epochs = self.patient_epochs[patient_id]
            n_epochs = len(epochs)
            
            if n_epochs < self.sequence_length:
                continue
            
            is_first_sequence = True
            
            # Stride ile sequence'leri oluştur
            for start_idx in range(0, n_epochs - self.sequence_length + 1, self.stride):
                signals_list = []
                labels_list = []
                
                for i in range(self.sequence_length):
                    epoch_data = epochs[start_idx + i]
                    epoch_idx = epoch_data['epoch_idx']
                    label = epoch_data['label']
                    
                    # Her kanal için sinyal al
                    channel_signals = []
                    for channel in self.channels:
                        cache_key = (patient_id, epoch_idx, channel)
                        
                        if self.preload:
                            signal = self.signal_cache.get(cache_key)
                            if signal is None:
                                signal = np.zeros(config.SAMPLES_PER_EPOCH, dtype=np.float32)
                        else:
                            signal = self._load_signal_on_the_fly(patient_id, epoch_idx, channel)
                        
                        channel_signals.append(signal)
                    
                    # Stack channels: [n_channels, samples]
                    epoch_signal = np.stack(channel_signals, axis=0)
                    
                    # Normalize per-epoch
                    if self.normalize == 'z-score':
                        epoch_signal = self._normalize_zscore(epoch_signal)
                    elif self.normalize == 'min-max':
                        epoch_signal = self._normalize_minmax(epoch_signal)
                    
                    signals_list.append(epoch_signal)
                    labels_list.append(label)
                
                # Stack: [seq_len, n_channels, samples]
                signals = np.stack(signals_list, axis=0)
                labels = np.array(labels_list, dtype=np.int64)
                
                # Tensor'a çevir ve batch dimension ekle: [1, seq_len, n_channels, samples]
                signals_tensor = torch.from_numpy(signals).float().unsqueeze(0)
                labels_tensor = torch.from_numpy(labels).long().unsqueeze(0)
                
                yield {
                    'patient_id': patient_id,
                    'is_new_patient': is_first_sequence,
                    'signals': signals_tensor,
                    'labels': labels_tensor
                }
                
                is_first_sequence = False
    
    def get_total_sequences_per_patient(self):
        """Her hasta için toplam sequence sayısını döndür."""
        result = {}
        for patient_id in self.patient_ids:
            if patient_id not in self.patient_epochs:
                continue
            n_epochs = len(self.patient_epochs[patient_id])
            if n_epochs >= self.sequence_length:
                n_seq = (n_epochs - self.sequence_length) // self.stride + 1
                result[patient_id] = n_seq
        return result


def create_sequence_dataloaders(train_ids, val_ids, test_ids, 
                                 sequence_length=None, stride=None, preload=True):
    """
    DeepSleepNet için sequence-based DataLoader'lar oluştur.
    
    Args:
        train_ids: Training hasta ID'leri
        val_ids: Validation hasta ID'leri
        test_ids: Test hasta ID'leri
        sequence_length: Ardışık epoch sayısı (None = config'den al)
        stride: Sequence stride (None = config.SEQUENCE_STRIDE'dan al)
        preload: Tüm veriyi RAM'e yükle
        
    Returns:
        train_loader, val_loader, test_loader
    """
    if sequence_length is None:
        sequence_length = config.SEQUENCE_LENGTH
    
    if stride is None:
        stride = config.SEQUENCE_STRIDE
    
    print(f"\n{'='*70}")
    print(f"SEQUENCE DATALOADERS OLUŞTURULUYOR")
    print(f"{'='*70}")
    print(f"Sequence length: {sequence_length}")
    print(f"Stride: {stride}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"{'='*70}\n")
    
    # Datasets
    print("📦 Train dataset oluşturuluyor...")
    train_dataset = SleepSequenceDataset(
        patient_ids=train_ids,
        sequence_length=sequence_length,
        stride=stride,
        preload=preload,
        augment=config.USE_AUGMENTATION
    )
    
    print("📦 Validation dataset oluşturuluyor...")
    val_dataset = SleepSequenceDataset(
        patient_ids=val_ids,
        sequence_length=sequence_length,
        stride=stride,
        preload=preload,
        augment=False
    )
    
    print("📦 Test dataset oluşturuluyor...")
    test_dataset = SleepSequenceDataset(
        patient_ids=test_ids,
        sequence_length=sequence_length,
        stride=stride,
        preload=preload,
        augment=False
    )
    
    # DataLoaders
    # Sequence mode'da num_workers=0 daha stabil (büyük tensor'lar)
    num_workers = 0 if preload else min(config.NUM_WORKERS, 2)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False,
        drop_last=True  # Son eksik batch'i atla (BiLSTM için)
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
        persistent_workers=False
    )
    
    print(f"\n{'='*70}")
    print(f"✓ SEQUENCE DATALOADERS HAZIR!")
    print(f"{'='*70}")
    print(f"  Train: {len(train_loader)} batch ({len(train_dataset)} sequence)")
    print(f"  Val:   {len(val_loader)} batch ({len(val_dataset)} sequence)")
    print(f"  Test:  {len(test_loader)} batch ({len(test_dataset)} sequence)")
    print(f"\n  Batch shape: [{config.BATCH_SIZE}, {sequence_length}, n_channels, {config.SAMPLES_PER_EPOCH}]")
    print(f"{'='*70}\n")
    
    return train_loader, val_loader, test_loader


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Config'i yükle
    config.set_seed()
    config.create_directories()
    
    # MongoDB'den hasta listesini al
    from pymongo import MongoClient
    import gridfs
    
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    fs = gridfs.GridFS(db)
    
    all_patient_ids = set()
    for file in fs.find({"metadata.data_type": "sleep_stages"}):
        all_patient_ids.add(file.metadata['patient_id'])
    
    all_patient_ids = sorted(all_patient_ids)
    client.close()
    
    print(f"Toplam hasta sayısı: {len(all_patient_ids)}")
    
    # Split yap
    train_ids, val_ids, test_ids = split_patients(
        all_patient_ids,
        config.N_TRAIN_PATIENTS,
        config.N_VAL_PATIENTS,
        config.N_TEST_PATIENTS
    )
    
    # DataLoader'ları oluştur (preload=True)
    train_loader, val_loader, test_loader = create_dataloaders(
        train_ids, val_ids, test_ids, preload=True
    )
    
    # İlk batch'i test et
    print("\nİlk batch test ediliyor...")
    signals, labels = next(iter(train_loader))
    
    print(f"Signal shape: {signals.shape}")
    print(f"Label shape: {labels.shape}")
    print(f"Signal dtype: {signals.dtype}")
    print(f"Label dtype: {labels.dtype}")
    print(f"Signal range: [{signals.min():.3f}, {signals.max():.3f}]")
    print(f"Labels: {labels[:5]}")
    
    print("\n✓ Dataset test başarılı!")