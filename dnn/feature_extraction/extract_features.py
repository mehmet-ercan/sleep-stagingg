"""
Main Feature Extraction Module
Tüm kanallar için time + frequency domain feature'ları çıkarır
"""

import numpy as np
import sys
import os
from tqdm import tqdm

# Local imports
from feature_extraction.time_domain import extract_time_domain_features
from feature_extraction.frequency_domain import extract_frequency_domain_features

# Project imports
import config
from edf_to_mongo import extract_epoch_range_data_v2


def extract_single_channel_features(signal, channel_name, sampling_rate=256):
    """
    Tek bir kanal için tüm feature'ları çıkar.
    
    Args:
        signal: 1D numpy array
        channel_name: Kanal ismi (prefix için)
        sampling_rate: Örnekleme frekansı
    
    Returns:
        dict: Prefixed feature'lar (örn: C3_mean, C3_delta_power)
    """
    # Time domain features
    time_features = extract_time_domain_features(signal)
    
    # Frequency domain features
    freq_features = extract_frequency_domain_features(signal, sampling_rate)
    
    # Tüm feature'ları birleştir
    all_features = {**time_features, **freq_features}
    
    # Kanal ismi ile prefix ekle
    prefixed_features = {
        f"{channel_name}_{key}": value
        for key, value in all_features.items()
    }
    
    return prefixed_features


def extract_epoch_features(patient_id, epoch_idx, channels, 
                          mongo_uri=config.MONGO_URI,
                          db_name=config.DB_NAME,
                          verbose=False):
    """
    Tek bir epoch için tüm kanalların feature'larını çıkar.
    
    Args:
        patient_id: Hasta ID
        epoch_idx: Epoch indeksi
        channels: Kanal listesi
        mongo_uri: MongoDB URI
        db_name: Database adı
        verbose: Print'leri göster
    
    Returns:
        dict: Tüm kanalların feature'ları
    """
    all_features = {}
    
    # STDOUT'u kapat (MongoDB print'lerini sustur)
    original_stdout = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, 'w')
    
    for channel_name in channels:
        try:
            # MongoDB'den sinyali çek
            result = extract_epoch_range_data_v2(
                patient_id=patient_id,
                channel_name=channel_name,
                start_epoch=epoch_idx,
                end_epoch=epoch_idx,
                mongo_uri=mongo_uri,
                db_name=db_name,
                save_to_file=False
            )
            
            if result is None or result['signal_data'] is None:
                # Hata durumunda NaN feature'lar
                channel_features = create_nan_features(channel_name)
            else:
                signal = result['signal_data']
                sampling_rate = result['sample_frequency']
                
                # Sinyal kontrolü
                if len(signal) == 0 or np.std(signal) < 1e-10:
                    # Flat veya boş sinyal
                    channel_features = create_nan_features(channel_name)
                else:
                    # Feature extraction
                    channel_features = extract_single_channel_features(
                        signal, channel_name, sampling_rate
                    )
            
            # Birleştir
            all_features.update(channel_features)
            
        except Exception as e:
            if verbose:
                sys.stdout = original_stdout
                print(f"⚠ Hata: {patient_id}, epoch {epoch_idx}, {channel_name}: {str(e)}")
                if "index 0 is out of bounds" in str(e):
                    print(f"   → Muhtemelen flat signal (sıfır varyans)")
                sys.stdout = open(os.devnull, 'w')
            
            # Hata durumunda NaN
            channel_features = create_nan_features(channel_name)
            all_features.update(channel_features)
    
    # STDOUT'u geri yükle
    sys.stdout = original_stdout
    
    return all_features


def create_nan_features(channel_name):
    """
    Hata durumunda NaN feature'lar oluştur.
    
    Args:
        channel_name: Kanal ismi
    
    Returns:
        dict: NaN değerlerle dolu feature dictionary
    """
    # Dummy signal ile feature isimlerini al
    dummy_signal = np.zeros(100)
    time_features = extract_time_domain_features(dummy_signal)
    freq_features = extract_frequency_domain_features(dummy_signal)
    
    all_features = {**time_features, **freq_features}
    
    # NaN'larla doldur
    nan_features = {
        f"{channel_name}_{key}": np.nan
        for key in all_features.keys()
    }
    
    return nan_features


def extract_dataset_features(patient_ids, channels, 
                             mongo_uri=config.MONGO_URI,
                             db_name=config.DB_NAME,
                             max_epochs_per_patient=None):
    """
    Tüm hasta ve epoch'lar için feature'ları çıkar.
    
    Args:
        patient_ids: Hasta ID listesi
        channels: Kanal listesi
        mongo_uri: MongoDB URI
        db_name: Database adı
        max_epochs_per_patient: Hasta başına max epoch (None = hepsi)
    
    Returns:
        X: Feature matrix [n_samples, n_features]
        y: Label array [n_samples]
        metadata: Her sample'ın bilgileri
    """
    from edf_to_mongo import get_cached_stages
    
    print(f"\n{'='*70}")
    print(f"FEATURE EXTRACTION - DATASET OLUŞTURMA")
    print(f"{'='*70}")
    print(f"Hastalar: {len(patient_ids)}")
    print(f"Kanallar: {len(channels)}")
    print(f"{'='*70}\n")
    
    X = []  # Feature matrix
    y = []  # Labels
    metadata = []  # Patient ID, epoch idx, stage
    
    for patient_id in patient_ids:
        print(f"\nİşleniyor: {patient_id}")
        
        # Stage verilerini al
        stages_data = get_cached_stages(patient_id, mongo_uri, db_name)
        
        if stages_data is None:
            print(f"  ⚠ Stage verisi bulunamadı, atlanıyor")
            continue
        
        # Epoch'ları filtrele (sadece bilinen stage'ler)
        valid_epochs = [
            epoch_data for epoch_data in stages_data
            if epoch_data['stage'] in config.CLASS_TO_IDX
        ]
        
        # Max epoch limiti
        if max_epochs_per_patient:
            valid_epochs = valid_epochs[:max_epochs_per_patient]
        
        print(f"  Toplam epoch: {len(valid_epochs)}")
        
        # Her epoch için feature extraction
        for epoch_data in tqdm(valid_epochs, desc=f"  {patient_id}", leave=False):
            epoch_idx = epoch_data['epoch']
            stage = epoch_data['stage']
            label = config.CLASS_TO_IDX[stage]
            
            # Feature'ları çıkar
            features = extract_epoch_features(
                patient_id, epoch_idx, channels,
                mongo_uri, db_name, verbose=False
            )
            
            # Feature değerlerini array'e çevir (sıralı)
            feature_values = [features[key] for key in sorted(features.keys())]
            
            X.append(feature_values)
            y.append(label)
            metadata.append({
                'patient_id': patient_id,
                'epoch_idx': epoch_idx,
                'stage': stage,
                'label': label
            })
        
        print(f"  ✓ {len(valid_epochs)} epoch işlendi")
    
    # NumPy array'e çevir
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    
    print(f"\n{'='*70}")
    print(f"DATASET HAZIR!")
    print(f"{'='*70}")
    print(f"Feature matrix: {X.shape}")
    print(f"Label array: {y.shape}")
    print(f"Toplam sample: {len(y)}")
    
    # NaN kontrolü
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        print(f"⚠ Uyarı: {nan_count} NaN değer bulundu")
        print(f"  NaN'ları median ile dolduruyorum...")
        
        # Sütun bazında median
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = imputer.fit_transform(X)
        
        print(f"  ✓ NaN'lar dolduruldu")
    
    print(f"{'='*70}\n")
    
    # Feature isimleri
    dummy_features = extract_epoch_features(
        patient_ids[0], 0, channels[:1], mongo_uri, db_name
    )
    feature_names = sorted(dummy_features.keys())
    
    return X, y, metadata, feature_names


def get_feature_names(channels):
    """
    Tüm feature isimlerini al (kanal × feature_type)
    
    Args:
        channels: Kanal listesi
    
    Returns:
        list: Sıralı feature isimleri
    """
    # Dummy signal ile feature tiplerini al
    dummy_signal = np.zeros(100)
    time_features = extract_time_domain_features(dummy_signal)
    freq_features = extract_frequency_domain_features(dummy_signal)
    
    base_features = list(time_features.keys()) + list(freq_features.keys())
    
    # Her kanal için prefix ekle
    all_feature_names = []
    for channel in channels:
        for feature in base_features:
            all_feature_names.append(f"{channel}_{feature}")
    
    return sorted(all_feature_names)


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Test: Tek bir epoch
    print("Feature Extraction Test")
    print("=" * 70)
    
    test_signal = np.random.randn(7680)
    
    features = extract_single_channel_features(test_signal, "C3", 256)
    
    print(f"\nToplam feature sayısı (tek kanal): {len(features)}")
    print("\nÖrnek feature'lar:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name:30s}: {value:10.4f}")
    
    # Feature isimleri
    test_channels = ['C3', 'C4', 'E1']
    feature_names = get_feature_names(test_channels)
    
    print(f"\n3 kanal için toplam feature: {len(feature_names)}")
    print(f"İlk 5 feature:")
    for name in feature_names[:5]:
        print(f"  {name}")
    
    print("\n✓ Feature extraction test başarılı!")