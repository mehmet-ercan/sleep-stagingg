#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import config

# Offline modda MongoDB gerekmez
try:
    import pyedflib
    from pymongo import MongoClient
    import gridfs
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

try:
    from datetime import datetime
    import io
    import xml.etree.ElementTree as ET
    import glob
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from scipy.signal import butter, filtfilt, iirnotch
except ImportError:
    pass  # Bazıları offline modda gerekmeyebilir


# ============================================================================
# Differential Channel Configuration (AASM Standard)
# ============================================================================
DIFFERENTIAL_CHANNELS = {
    # EEG: Contralateral mastoid referencing
    'C3-M2': ('C3', 'M2'),
    'C4-M1': ('C4', 'M1'),
    'F3-M2': ('F3', 'M2'),
    'F4-M1': ('F4', 'M1'),
    'O1-M2': ('O1', 'M2'),
    'O2-M1': ('O2', 'M1'),
    # EOG: Contralateral referencing
    'E1-M2': ('E1', 'M2'),
    'E2-M1': ('E2', 'M1'),
    # EMG: Bipolar
    'CHIN1-CHIN2': ('CHIN1', 'CHIN2'),
}

# ============================================================================
# Signal-Specific Filter Settings (AASM Standard)
# ============================================================================
FILTER_SETTINGS = {
    'EEG': {'lowcut': 0.3, 'highcut': 35.0, 'order': 5, 'notch': 50.0},
    'EOG': {'lowcut': 0.3, 'highcut': 35.0, 'order': 5, 'notch': 50.0},
    'EMG': {'lowcut': 10.0, 'highcut': 100.0, 'order': 5, 'notch': 50.0},
}

# Amplitude clipping DEVRE DIŞI - sinyal dinamik aralığını bozuyor (özellikle EMG)
# Z-score normalizasyonu uç değerleri training sırasında halleder
USE_AMPLITUDE_CLIPPING = False

# Channel name to signal type mapping
CHANNEL_SIGNAL_TYPE = {
    'C3-M2': 'EEG', 'C4-M1': 'EEG',
    'F3-M2': 'EEG', 'F4-M1': 'EEG',
    'O1-M2': 'EEG', 'O2-M1': 'EEG',
    'E1-M2': 'EOG', 'E2-M1': 'EOG',
    'CHIN1-CHIN2': 'EMG',
}


def get_signal_type(channel_name):
    """Get signal type (EEG, EOG, EMG) for a channel name."""
    return CHANNEL_SIGNAL_TYPE.get(channel_name, 'EEG')


# ============================================================================
# Filter Functions
# ============================================================================
def apply_notch_filter(signal_data, notch_freq=50.0, quality=30.0, fs=256):
    """Notch filter (50 Hz powerline interference removal)."""
    nyq = fs / 2.0
    w0 = notch_freq / nyq
    if not (0 < w0 < 1):
        return signal_data
    b, a = iirnotch(w0, quality)
    return filtfilt(b, a, signal_data)


def apply_bandpass_filter(signal_data, lowcut=0.3, highcut=35.0, fs=256, order=5):
    """Butterworth bandpass filter."""
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    if low >= high:
        return signal_data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, signal_data)


def apply_all_filters(signal_data, fs=256, channel_name=None):
    """
    Notch + bandpass filtreleri uygula (AASM standard).
    EEG/EOG: 0.3-35 Hz, EMG: 10-100 Hz
    """
    signal_type = get_signal_type(channel_name) if channel_name else 'EEG'
    settings = FILTER_SETTINGS.get(signal_type, FILTER_SETTINGS['EEG'])

    lowcut = settings['lowcut']
    highcut = settings['highcut']
    order = settings['order']
    notch_freq = settings['notch']

    nyq = fs / 2.0
    if highcut >= nyq:
        highcut = nyq * 0.95

    if notch_freq < nyq:
        signal_filtered = apply_notch_filter(signal_data, notch_freq=notch_freq, fs=fs)
    else:
        signal_filtered = signal_data

    signal_filtered = apply_bandpass_filter(signal_filtered, lowcut=lowcut, highcut=highcut, fs=fs, order=order)

    filter_params = {
        'signal_type': signal_type,
        'lowcut': lowcut,
        'highcut': highcut,
        'order': order,
        'notch_freq': notch_freq,
        'applied_at_fs': fs
    }

    return signal_filtered, filter_params


def apply_amplitude_clipping(signal_data, low_pct=1, high_pct=99):
    """Amplitude clipping (percentile tabanlı)."""
    low_val = np.percentile(signal_data, low_pct)
    high_val = np.percentile(signal_data, high_pct)
    clipped_signal = np.clip(signal_data, low_val, high_val)
    clip_params = {
        'low_percentile': low_pct,
        'high_percentile': high_pct,
        'low_value': float(low_val),
        'high_value': float(high_val),
        'clipped_samples_pct': float(np.sum((signal_data < low_val) | (signal_data > high_val)) / len(signal_data) * 100)
    }
    return clipped_signal, clip_params


def create_differential_channel(signal_active, signal_reference):
    """Differential kanal oluştur: Active - Reference."""
    min_len = min(len(signal_active), len(signal_reference))
    return signal_active[:min_len] - signal_reference[:min_len]


# GLOBAL CACHE - Tüm program boyunca geçerli
_STAGES_CACHE = {}
_SIGNALS_CACHE = {}  # Offline mod: {(patient_id, channel_safe): ndarray(n_epochs, samples)}
_EPOCH_INDICES_CACHE = {}  # Offline mod: {patient_id: ndarray(epoch_indices)}


def clear_stages_cache():
    """
    Cache'i temizle (ihtiyaç olursa)
    """
    global _STAGES_CACHE, _SIGNALS_CACHE, _EPOCH_INDICES_CACHE
    _STAGES_CACHE.clear()
    _SIGNALS_CACHE.clear()
    _EPOCH_INDICES_CACHE.clear()
    print("✓ Stage cache temizlendi")


# ============================================================================
# Offline Data Functions (.npz dosyalarından yükleme)
# ============================================================================

def get_offline_patient_ids():
    """
    Offline data dizininden hasta listesini al.
    
    Returns:
        list: Sıralı hasta ID listesi
    """
    import json
    info_path = os.path.join(config.OFFLINE_DATA_DIR, 'dataset_info.json')
    
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
        return sorted(info['patient_ids'])
    
    # dataset_info.json yoksa dosya isimlerinden çıkar
    patient_ids = []
    for fname in os.listdir(config.OFFLINE_DATA_DIR):
        if fname.startswith('patient_') and fname.endswith('.npz'):
            pid = fname[len('patient_'):-len('.npz')]
            patient_ids.append(pid)
    
    return sorted(patient_ids)


def _load_offline_patient(patient_id, load_signals=True):
    """
    Bir hastanın .npz dosyasını yükle ve cache'lere ekle.
    
    Args:
        patient_id: Hasta ID
        load_signals: True ise sinyal verilerini de yükle (büyük bellek kullanımı).
                      False ise sadece stages ve epoch_indices yükle (düşük bellek).
    
    Returns:
        bool: Başarılı ise True
    """
    global _STAGES_CACHE, _SIGNALS_CACHE, _EPOCH_INDICES_CACHE
    import json
    
    npz_path = os.path.join(config.OFFLINE_DATA_DIR, f'patient_{patient_id}.npz')
    
    if not os.path.exists(npz_path):
        print(f"⚠ Offline veri bulunamadı: {npz_path}")
        return False
    
    data = np.load(npz_path, allow_pickle=True)
    
    # Stages
    if patient_id not in _STAGES_CACHE:
        stages_bytes = bytes(data['stages_json'])
        stages_data = json.loads(stages_bytes.decode('utf-8'))
        _STAGES_CACHE[patient_id] = stages_data
    
    # Epoch indices
    if patient_id not in _EPOCH_INDICES_CACHE:
        _EPOCH_INDICES_CACHE[patient_id] = data['epoch_indices']
    
    # Sinyal verileri (isteğe bağlı - bellek tasarrufu için atlanabilir)
    if load_signals:
        for key in data.files:
            if key.startswith('signal_'):
                channel_safe = key[len('signal_'):]
                cache_key = (patient_id, channel_safe)
                if cache_key not in _SIGNALS_CACHE:
                    _SIGNALS_CACHE[cache_key] = data[key]
    
    return True


def preload_offline_patients(patient_ids=None, load_signals=True):
    """
    Offline modda tüm hastaların verilerini cache'e yükle.
    
    Args:
        patient_ids: Hasta listesi (None ise tüm hastalar)
        load_signals: True ise sinyal verilerini de yükle (büyük bellek).
                      False ise sadece stages/epoch_indices yükle (düşük bellek).
                      Sinyaller dataset _preload_all_signals() tarafından
                      hasta bazında yüklenip serbest bırakılacak.
    """
    if patient_ids is None:
        patient_ids = get_offline_patient_ids()
    
    import time as _time
    mode_str = "STAGES + SİNYALLER" if load_signals else "SADECE STAGES (düşük bellek modu)"
    
    print(f"\n{'='*70}", flush=True)
    print(f"OFFLINE VERİLER CACHE'E YÜKLENİYOR (.npz)", flush=True)
    print(f"{'='*70}", flush=True)
    print(f"Kaynak: {config.OFFLINE_DATA_DIR}", flush=True)
    print(f"Hasta sayısı: {len(patient_ids)}", flush=True)
    print(f"Yükleme modu: {mode_str}", flush=True)
    print(f"{'-'*70}", flush=True)
    
    loaded = 0
    t_start = _time.time()
    for idx, pid in enumerate(patient_ids, 1):
        t0 = _time.time()
        if _load_offline_patient(pid, load_signals=load_signals):
            n_epochs = len(_STAGES_CACHE.get(pid, []))
            elapsed = _time.time() - t0
            print(f"[{idx}/{len(patient_ids)}] {pid}: {n_epochs} epoch ✓ ({elapsed:.1f}s)", flush=True)
            loaded += 1
        else:
            print(f"[{idx}/{len(patient_ids)}] {pid}: Bulunamadı ✗", flush=True)
    
    # Bellek kullanımı
    signal_bytes = sum(arr.nbytes for arr in _SIGNALS_CACHE.values())
    signal_gb = signal_bytes / (1024**3)
    stage_bytes = sum(len(str(s).encode()) for s in _STAGES_CACHE.values())
    stage_mb = stage_bytes / (1024**2)
    
    total_time = _time.time() - t_start
    print(f"\n{'='*70}", flush=True)
    print(f"OFFLINE CACHE HAZIR!", flush=True)
    print(f"  Yüklenen hasta: {loaded}/{len(patient_ids)} ({total_time:.1f}s)", flush=True)
    print(f"  Stage cache: {len(_STAGES_CACHE)} hasta (~{stage_mb:.1f} MB)", flush=True)
    print(f"  Sinyal cache: {len(_SIGNALS_CACHE)} (patient, channel) çifti", flush=True)
    print(f"  Sinyal bellek: {signal_gb:.2f} GB", flush=True)
    print(f"{'='*70}\n", flush=True)


def get_offline_signal(patient_id, channel_name, epoch_idx):
    """
    Offline cache'den bir epoch'un sinyalini al.
    
    Args:
        patient_id: Hasta ID
        channel_name: Kanal adı (orijinal, ör: 'EEG Fpz-Cz')
        epoch_idx: Epoch indeksi
    
    Returns:
        np.ndarray: Sinyal dizisi (samples_per_epoch,) veya None
    """
    channel_safe = channel_name.replace(' ', '_').replace('/', '-')
    cache_key = (patient_id, channel_safe)
    
    # Cache'de yoksa yüklemeyi dene
    if cache_key not in _SIGNALS_CACHE:
        _load_offline_patient(patient_id)
    
    if cache_key not in _SIGNALS_CACHE:
        return None
    
    signals = _SIGNALS_CACHE[cache_key]
    epoch_indices = _EPOCH_INDICES_CACHE.get(patient_id)
    
    if epoch_indices is None:
        return None
    
    # epoch_idx'in dizideki pozisyonunu bul
    positions = np.where(epoch_indices == epoch_idx)[0]
    if len(positions) == 0:
        return None
    
    pos = positions[0]
    if pos >= len(signals):
        return None
    
    return signals[pos].copy()


def get_cached_stages(patient_id, mongo_uri=None, db_name=None):
    """
    Hastanın stage verilerini cache'den getir, yoksa kaynaktan yükle.
    Offline modda .npz dosyalarından, normal modda MongoDB'den yükler.
    
    Args:
        patient_id: Hasta ID'si
        mongo_uri: MongoDB bağlantı URI'si (offline modda kullanılmaz)
        db_name: Veritabanı adı (offline modda kullanılmaz)
    
    Returns:
        list: Stage verileri veya None
    """
    global _STAGES_CACHE
    
    # Cache'de var mı kontrol et
    if patient_id in _STAGES_CACHE:
        return _STAGES_CACHE[patient_id]
    
    # Offline mod: .npz dosyasından yükle
    if config.USE_OFFLINE_DATA:
        _load_offline_patient(patient_id)
        return _STAGES_CACHE.get(patient_id)
    
    # Normal mod: MongoDB'den yükle
    if mongo_uri is None:
        mongo_uri = config.MONGO_URI
    if db_name is None:
        db_name = config.DB_NAME
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    stages_file = fs.find_one({
        "metadata.patient_id": patient_id,
        "metadata.data_type": "sleep_stages"
    })
    
    if not stages_file:
        client.close()
        return None
    
    import json
    stages_data = json.loads(stages_file.read().decode('utf-8'))
    
    # Cache'e ekle
    _STAGES_CACHE[patient_id] = stages_data
    
    client.close()
    return stages_data


def preload_all_stages(patient_ids=None, mongo_uri=config.MONGO_URI,
                       db_name=config.DB_NAME, load_signals=False):
    """
    Tüm hastaların stage verilerini önceden cache'e yükle.
    Offline modda .npz dosyalarından, normal modda MongoDB'den yükler.
    
    Args:
        patient_ids: Hasta ID listesi (None ise tüm hastalar)
        mongo_uri: MongoDB bağlantı URI'si (offline modda kullanılmaz)
        db_name: Veritabanı adı (offline modda kullanılmaz)
        load_signals: Offline modda sinyalleri de yükle (default: False).
                      False ise sadece stages yüklenir, sinyaller dataset
                      tarafından hasta bazında yüklenir (bellek tasarrufu).
    
    Returns:
        dict: Yükleme özeti
    """
    global _STAGES_CACHE
    
    # Offline mod: .npz dosyalarından yükle
    if config.USE_OFFLINE_DATA:
        if patient_ids is None:
            patient_ids = get_offline_patient_ids()
        preload_offline_patients(patient_ids, load_signals=load_signals)
        return {
            'loaded': len(_STAGES_CACHE),
            'total_patients': len(patient_ids),
            'total_epochs': sum(len(s) for s in _STAGES_CACHE.values()),
            'total_size_mb': 0,
            'cache_size': len(_STAGES_CACHE)
        }
    
    # Normal mod: MongoDB'den yükle
    print(f"\n{'='*70}")
    print(f"TÜM STAGE VERİLERİ CACHE'E YÜKLENİYOR")
    print(f"{'='*70}")
    
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Hasta listesi verilmemişse tüm hastaları bul
    if patient_ids is None:
        patient_ids = set()
        for file in fs.find({"metadata.data_type": "sleep_stages"}):
            patient_ids.add(file.metadata['patient_id'])
        patient_ids = sorted(patient_ids)
    
    print(f"Toplam {len(patient_ids)} hasta bulundu")
    print(f"{'-'*70}")
    
    loaded_count = 0
    total_epochs = 0
    total_size_mb = 0
    
    import json
    
    for idx, patient_id in enumerate(patient_ids, 1):
        # Zaten cache'de varsa atla
        if patient_id in _STAGES_CACHE:
            print(f"[{idx}/{len(patient_ids)}] {patient_id}: Cache'de zaten mevcut ⊘")
            continue
        
        # MongoDB'den yükle
        stages_file = fs.find_one({
            "metadata.patient_id": patient_id,
            "metadata.data_type": "sleep_stages"
        })
        
        if not stages_file:
            print(f"[{idx}/{len(patient_ids)}] {patient_id}: Stage verisi bulunamadı ✗")
            continue
        
        stages_data = json.loads(stages_file.read().decode('utf-8'))
        
        # Cache'e ekle
        _STAGES_CACHE[patient_id] = stages_data
        
        # İstatistikler
        epoch_count = len(stages_data)
        size_bytes = len(json.dumps(stages_data))
        size_mb = size_bytes / (1024 * 1024)
        
        loaded_count += 1
        total_epochs += epoch_count
        total_size_mb += size_mb
        
        print(f"[{idx}/{len(patient_ids)}] {patient_id}: {epoch_count} epoch, "
              f"{size_mb:.2f} MB ✓")
    
    client.close()
    
    print(f"\n{'='*70}")
    print(f"CACHE YÜKLEME TAMAMLANDI")
    print(f"{'='*70}")
    print(f"Yüklenen hasta: {loaded_count}/{len(patient_ids)}")
    print(f"Toplam epoch: {total_epochs:,}")
    print(f"Toplam boyut: {total_size_mb:.2f} MB")
    print(f"Cache boyutu: {len(_STAGES_CACHE)} hasta")
    print(f"{'='*70}\n")
    
    return {
        'loaded': loaded_count,
        'total_patients': len(patient_ids),
        'total_epochs': total_epochs,
        'total_size_mb': total_size_mb,
        'cache_size': len(_STAGES_CACHE)
    }


def delete_patient_data(patient_id, mongo_uri=config.MONGO_URI, 
                        db_name=config.DB_NAME, confirm=True):
    """
    Hastanın TÜM verilerini MongoDB'den siler (kanallar + uyku evreleri).
    
    Args:
        patient_id: Hasta ID'si
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
        confirm: True ise kullanıcıdan onay ister
    
    Returns:
        dict: Silme işlemi özeti
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Hastaya ait tüm dosyaları bul
    patient_files = list(fs.find({"metadata.patient_id": patient_id}))
    
    if not patient_files:
        print(f"✗ Hasta bulunamadı: {patient_id}")
        client.close()
        return {
            'patient_id': patient_id,
            'status': 'not_found',
            'deleted_files': 0
        }
    
    # Dosya sayılarını hesapla
    channel_files = [f for f in patient_files if f.metadata.get('data_type') == 'raw_signal']
    stages_files = [f for f in patient_files if f.metadata.get('data_type') == 'sleep_stages']
    
    print(f"\n{'='*70}")
    print(f"Hasta ID: {patient_id}")
    print(f"{'='*70}")
    print(f"Kanal dosyaları: {len(channel_files)}")
    print(f"Uyku evresi dosyaları: {len(stages_files)}")
    print(f"Toplam silinecek dosya: {len(patient_files)}")
    
    # Kanal detaylarını göster
    if channel_files:
        print(f"\nKanallar:")
        for ch_file in channel_files:
            ch_meta = ch_file.metadata
            size_mb = ch_file.length / (1024 * 1024)
            print(f"  - {ch_meta['channel_name']} ({ch_meta['sample_frequency']} Hz, {size_mb:.2f} MB)")
    
    # Onay iste
    if confirm:
        print(f"\n⚠ UYARI: Bu işlem geri alınamaz!")
        response = input(f"\n'{patient_id}' hastasının TÜM verilerini silmek istediğinize emin misiniz? (evet/hayır): ")
        
        if response.lower() not in ['evet', 'yes', 'e', 'y']:
            print("İşlem iptal edildi.")
            client.close()
            return {
                'patient_id': patient_id,
                'status': 'cancelled',
                'deleted_files': 0
            }
    
    # Tüm dosyaları sil
    print(f"\nDosyalar siliniyor...")
    deleted_count = 0
    
    for file in patient_files:
        try:
            fs.delete(file._id)
            deleted_count += 1
            print(f"  ✓ Silindi: {file.filename}")
        except Exception as e:
            print(f"  ✗ Silinemedi: {file.filename} - {str(e)}")
    
    client.close()
    
    print(f"\n{'='*70}")
    print(f"✓ İşlem tamamlandı")
    print(f"Silinen dosya sayısı: {deleted_count}/{len(patient_files)}")
    print(f"{'='*70}")
    
    return {
        'patient_id': patient_id,
        'status': 'deleted',
        'deleted_files': deleted_count,
        'total_files': len(patient_files)
    }


def delete_patient_datas(patient_ids, mongo_uri=config.MONGO_URI, 
                        db_name=config.DB_NAME, confirm=True):
    """
    Hastaların TÜM verilerini MongoDB'den siler (kanallar + uyku evreleri).
    
    Args:
        patient_ids: Hasta ID'leri
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
        confirm: True ise kullanıcıdan onay ister
    
    Returns:
        dict: Silme işlemleri özeti
    """

    for patient_id in patient_ids:
        delete_patient_data(patient_id, mongo_uri, db_name, confirm)


def patient_exists_in_mongodb(patient_id, mongo_uri=config.MONGO_URI, 
                               db_name=config.DB_NAME):
    """
    Hastanın MongoDB'de olup olmadığını kontrol eder.
    
    Args:
        patient_id: Hasta ID'si
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
    
    Returns:
        bool: Hasta varsa True, yoksa False
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Hasta ID'sine ait herhangi bir dosya var mı kontrol et
    exists = fs.find_one({"metadata.patient_id": patient_id}) is not None
    
    client.close()
    return exists


def regenerate_filtered_from_raw(patient_id, mongo_uri=config.MONGO_URI,
                                  db_name_raw=config.DB_NAME_RAW,
                                  db_name_filtered=config.DB_NAME_FILTERED):
    """
    Raw veritabanından filtered veritabanını yeniden oluşturur.
    EDF dosyasını tekrar okumadan, sadece raw sinyalleri kullanır.
    Differential montaj + notch + bandpass uygular (amplitude clipping KAPALI).

    Args:
        patient_id: Hasta ID'si
        mongo_uri: MongoDB bağlantı URI'si
        db_name_raw: Raw data veritabanı adı
        db_name_filtered: Filtered data veritabanı adı

    Returns:
        bool: Başarılı ise True
    """
    client = MongoClient(mongo_uri)
    db_raw = client[db_name_raw]
    db_filtered = client[db_name_filtered]
    fs_raw = gridfs.GridFS(db_raw)
    fs_filtered = gridfs.GridFS(db_filtered)

    try:
        # Raw kanalları oku
        raw_signals = {}
        for f in fs_raw.find({'metadata.patient_id': patient_id}):
            if f.metadata and f.metadata.get('data_type') == 'raw_signal':
                ch_name = f.metadata.get('channel_name')
                if ch_name:
                    data = f.read()
                    if data[:6] == b'\x93NUMPY':
                        signal = np.load(io.BytesIO(data))
                    else:
                        signal = np.frombuffer(data, dtype=np.float64)

                    raw_signals[ch_name] = {
                        'signal': signal,
                        'sample_freq': f.metadata.get('sample_frequency', 256),
                        'metadata': f.metadata
                    }

        if not raw_signals:
            print(f"  ✗ Raw sinyal bulunamadı: {patient_id}")
            client.close()
            return False

        # Sleep stages'i kopyala
        stages_data = None
        stages_metadata = None
        for f in fs_raw.find({'metadata.patient_id': patient_id}):
            if f.metadata and f.metadata.get('data_type') == 'sleep_stages':
                stages_data = f.read()
                stages_metadata = dict(f.metadata)
                break

        if stages_data:
            stages_metadata['source_db'] = db_name_raw
            stages_buffer = io.BytesIO(stages_data)
            stages_id = fs_filtered.put(
                stages_buffer,
                filename=f"{patient_id}_sleep_stages.json",
                metadata=stages_metadata
            )
            print(f"  ✓ Sleep stages kopyalandı")
        else:
            stages_id = None
            print(f"  ⚠ Sleep stages bulunamadı")

        # Differential kanalları oluştur
        diff_count = 0
        for diff_name, (active_ch, ref_ch) in DIFFERENTIAL_CHANNELS.items():
            if active_ch in raw_signals and ref_ch in raw_signals:
                signal_active = raw_signals[active_ch]['signal']
                signal_ref = raw_signals[ref_ch]['signal']
                fs_hz = raw_signals[active_ch]['sample_freq']

                if raw_signals[ref_ch]['sample_freq'] != fs_hz:
                    continue

                # Differential sinyal oluştur
                diff_signal = create_differential_channel(signal_active, signal_ref)

                # Notch + bandpass filtrele
                filtered_signal, filter_params = apply_all_filters(
                    diff_signal, fs=fs_hz, channel_name=diff_name)

                # Amplitude clipping (varsayılan: KAPALI)
                clip_params = None
                if USE_AMPLITUDE_CLIPPING:
                    filtered_signal, clip_params = apply_amplitude_clipping(filtered_signal)

                # Metadata
                orig_metadata = raw_signals[active_ch].get('metadata', {})

                diff_metadata = {
                    "patient_id": patient_id,
                    "channel_name": diff_name,
                    "differential_channel": True,
                    "active_electrode": active_ch,
                    "reference_electrode": ref_ch,
                    "sample_frequency": float(fs_hz),
                    "n_samples": int(len(filtered_signal)),
                    "duration_hours": float(len(filtered_signal) / fs_hz / 3600),
                    "data_type": "raw_signal",
                    "dtype": str(filtered_signal.dtype),
                    "source_db": db_name_raw,
                    "preprocessing_applied": True,
                    "preprocessing_params": {
                        "differential": f"{active_ch} - {ref_ch}",
                        "signal_type": filter_params['signal_type'],
                        "notch_filter": {"freq": filter_params['notch_freq'], "Q": 30.0},
                        "bandpass_filter": {
                            "lowcut": filter_params['lowcut'],
                            "highcut": filter_params['highcut'],
                            "order": filter_params['order']
                        },
                        "amplitude_clipping": clip_params if clip_params else {"applied": False}
                    },
                    "upload_date": datetime.now(),
                    "regenerated_from_raw": True,
                    "stages_id": stages_id,
                    "has_sleep_stages": stages_id is not None
                }

                # Trim bilgisi varsa ekle
                if hasattr(orig_metadata, 'get'):
                    if orig_metadata.get('trim_start_time') is not None:
                        diff_metadata['trim_start_time'] = orig_metadata['trim_start_time']
                    if orig_metadata.get('trim_end_time') is not None:
                        diff_metadata['trim_end_time'] = orig_metadata['trim_end_time']

                buffer = io.BytesIO()
                np.save(buffer, filtered_signal)
                buffer.seek(0)

                file_id = fs_filtered.put(
                    buffer,
                    filename=f"{patient_id}_{diff_name}_filtered.npy",
                    metadata=diff_metadata,
                    chunk_size=255*1024
                )

                diff_count += 1
                print(f"  ✓ {diff_name} → kayıt edildi")

        client.close()
        print(f"  ✓ {diff_count} differential kanal oluşturuldu")
        return diff_count > 0

    except Exception as e:
        print(f"  ✗ Hata: {e}")
        client.close()
        return False


def process_patient_folders(root_folder, mongo_uri=config.MONGO_URI,
                            db_name=None, db_name_filtered=None,
                            skip_existing=True, save_filtered=True,
                            save_raw=True):
    """
    Klasörleri dolaşarak her hastanın verilerini MongoDB'ye aktarır.

    Her hasta için iki veritabanına kayıt yapar:
    1. psg_data (raw): Tüm kanallar (filtresiz)
    2. psg_data_filtered: Differential kanallar + notch + bandpass filtreli

    Args:
        root_folder: Ana klasör yolu
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Raw data veritabanı adı (default: config.DB_NAME_RAW)
        db_name_filtered: Filtered data veritabanı adı (default: config.DB_NAME_FILTERED)
        skip_existing: True ise daha önce aktarılmış hastaları atlar
        save_filtered: True ise differential+filtered verileri de kaydet

    Returns:
        dict: İşlem özeti
    """
    if db_name is None:
        db_name = config.DB_NAME_RAW
    if db_name_filtered is None:
        db_name_filtered = config.DB_NAME_FILTERED

    print(f"\n{'='*80}")
    print(f"TOPLU HASTA VERİSİ AKTARIMI")
    print(f"{'='*80}")
    print(f"Ana klasör: {root_folder}")
    print(f"Raw veritabanı: {db_name}")
    print(f"Filtered veritabanı: {db_name_filtered}")
    print(f"Amplitude clipping: {'AÇIK' if USE_AMPLITUDE_CLIPPING else 'KAPALI'}")
    print(f"Mevcut hastaları atla: {skip_existing}")
    print(f"{'='*80}\n")
    
    # Ana klasördeki tüm alt klasörleri bul
    patient_folders = [f for f in os.listdir(root_folder) 
                      if os.path.isdir(os.path.join(root_folder, f))]
    
    print(f"Toplam {len(patient_folders)} hasta klasörü bulundu.\n")
    
    results = {
        'total': len(patient_folders),
        'processed': 0,
        'skipped': 0,
        'failed': 0,
        'details': []
    }
    
    for idx, patient_id in enumerate(sorted(patient_folders), 1):
        patient_folder = os.path.join(root_folder, patient_id)
        
        print(f"\n{'='*80}")
        print(f"[{idx}/{len(patient_folders)}] Hasta: {patient_id}")
        print(f"{'='*80}")
        
        # Hasta daha önce işlendi mi kontrol et (filtered DB'ye bakılır)
        check_db = db_name_filtered if db_name_filtered else db_name
        if skip_existing and patient_exists_in_mongodb(patient_id, mongo_uri, check_db):
            print(f"⊘ ATLANDI - Bu hasta zaten MongoDB'de mevcut ({check_db})")
            results['skipped'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'status': 'skipped',
                'reason': 'already_exists'
            })
            continue
        
        # EDF dosyasını bul
        edf_files = glob.glob(os.path.join(patient_folder, "*.edf"))
        if not edf_files:
            print(f"✗ HATA - EDF dosyası bulunamadı: {patient_folder}")
            results['failed'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'status': 'failed',
                'reason': 'no_edf_file'
            })
            continue
        
        if len(edf_files) > 1:
            print(f"⚠ UYARI - Birden fazla EDF dosyası bulundu, ilki kullanılacak:")
            for ef in edf_files:
                print(f"    {os.path.basename(ef)}")
        
        edf_path = edf_files[0]
        print(f"EDF dosyası: {os.path.basename(edf_path)}")
        
        # XML dosyasını bul
        xml_files = glob.glob(os.path.join(patient_folder, "*.xml"))
        if not xml_files:
            print(f"✗ HATA - XML dosyası bulunamadı: {patient_folder}")
            results['failed'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'status': 'failed',
                'reason': 'no_xml_file'
            })
            continue
        
        if len(xml_files) > 1:
            print(f"⚠ UYARI - Birden fazla XML dosyası bulundu, ilki kullanılacak:")
            for xf in xml_files:
                print(f"    {os.path.basename(xf)}")
        
        xml_path = xml_files[0]
        print(f"XML dosyası: {os.path.basename(xml_path)}")
        
        # MongoDB'ye aktar
        try:
            result = save_all_channels_to_mongodb(
                edf_path=edf_path,
                stages_path=xml_path,
                stages_format='xml',
                mongo_uri=mongo_uri,
                db_name=db_name,
                db_name_filtered=db_name_filtered,
                patient_id=patient_id,
                save_filtered=save_filtered,
                save_raw=save_raw
            )
            
            results['processed'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'status': 'success',
                'channels': len(result['channels']),
                'epochs': len(result.get('sleep_stages', [])) if 'sleep_stages' in result else 'N/A'
            })
            
            print(f"\n✓ BAŞARILI - {patient_id} aktarıldı")
            
        except Exception as e:
            print(f"\n✗ HATA - {patient_id} aktarılırken hata oluştu:")
            print(f"    {str(e)}")
            results['failed'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'status': 'failed',
                'reason': str(e)
            })
    
    # Özet rapor
    print(f"\n\n{'='*80}")
    print(f"AKTARIM ÖZETI")
    print(f"{'='*80}")
    print(f"Toplam klasör: {results['total']}")
    print(f"Başarılı: {results['processed']} ✓")
    print(f"Atlanan: {results['skipped']} ⊘")
    print(f"Başarısız: {results['failed']} ✗")
    print(f"{'='*80}\n")
    
    # Detaylı rapor
    if results['failed'] > 0:
        print("Başarısız aktarımlar:")
        for detail in results['details']:
            if detail['status'] == 'failed':
                print(f"  - {detail['patient_id']}: {detail['reason']}")
    
    if results['skipped'] > 0:
        print(f"\nAtlanan hastalar: {results['skipped']} hasta (zaten MongoDB'de mevcut)")
    
    return results


def parse_sleep_stages_xml(xml_path):
    """
    XML formatındaki uyku evresi etiketlerini parse eder.
    Sadece belirtilen uyku evrelerini (SLEEP-REM, SLEEP-S0, SLEEP-S1, SLEEP-S2, SLEEP-S3) alır.
    
    Args:
        xml_path: XML dosyasının yolu
    
    Returns:
        list: [
            {
                'epoch': int,
                'stage': str,
                'start_time': float,
                'duration': float,
                'onset': str
            },
            ...
        ]
    """
    print(f"XML dosyası parse ediliyor: {xml_path}")
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # İstenen uyku evresi tipleri
    valid_event_types = {
        'SLEEP-REM': 'REM',
        'SLEEP-S0': 'W',
        'SLEEP-S1': 'N1',
        'SLEEP-S2': 'N2',
        'SLEEP-S3': 'N3'
    }
    
    stages = []
    epoch_counter = 0
    
    # Events elementini bul
    events_element = root.find('Events')
    if events_element is None:
        print("⚠ Events elementi bulunamadı")
        return stages
    
    # İlk eventi bul (referans zamanı için)
    first_event_time = None
    
    # Tüm Event elementlerini dolaş
    for event in events_element.findall('Event'):
        # Event tipini al
        type_elem = event.find('Type')
        if type_elem is None:
            continue
        
        event_type = type_elem.text.strip() if type_elem.text else None
        
        # Sadece istenen tipleri işle
        if event_type not in valid_event_types:
            continue
        
        # StartTime ve StopTime bilgilerini al
        start_time_elem = event.find('StartTime')
        stop_time_elem = event.find('StopTime')
        
        if start_time_elem is None or stop_time_elem is None:
            continue
        
        start_time_str = start_time_elem.text.strip() if start_time_elem.text else None
        stop_time_str = stop_time_elem.text.strip() if stop_time_elem.text else None
        
        if not start_time_str or not stop_time_str:
            continue
        
        try:
            # ISO format zamanlarını datetime'a çevir
            start_dt = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
            stop_dt = datetime.fromisoformat(stop_time_str.replace('Z', '+00:00'))
            
            # İlk event zamanını referans olarak kaydet
            if first_event_time is None:
                first_event_time = start_dt
            
            # Başlangıçtan itibaren geçen süreyi hesapla (saniye)
            start_time = (start_dt - first_event_time).total_seconds()
            duration = (stop_dt - start_dt).total_seconds()
            
            # Uyku evresi etiketini normalize et
            stage = valid_event_types[event_type]
            
            stages.append({
                'epoch': epoch_counter,
                'stage': stage,
                'start_time': start_time,
                'duration': duration,
                'onset': start_time_str
            })
            
            epoch_counter += 1
            
        except (ValueError, AttributeError) as e:
            print(f"⚠ Event parse hatası: {str(e)}")
            continue
    
    # İstatistikleri yazdır
    if stages:
        stage_counts = {}
        total_duration = 0
        
        for stage_info in stages:
            stage = stage_info['stage']
            duration = stage_info['duration']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
            total_duration += duration
        
        print(f"✓ {len(stages)} epoch parse edildi")
        print(f"  Toplam süre: {total_duration:.2f} saniye ({total_duration/3600:.2f} saat)")
        print(f"  Uyku evresi dağılımı:")
        
        for stage, count in sorted(stage_counts.items()):
            percentage = (count / len(stages)) * 100
            print(f"    {stage}: {count} epoch ({percentage:.1f}%)")
    else:
        print(f"⚠ İstenen uyku evresi tiplerinde data bulunamadı")
    
    return stages


def parse_sleep_stages_edf(edf_path, annotation_channel=None):
    """
    EDF+ formatındaki annotation kanalından uyku evrelerini okur.
    
    Args:
        edf_path: EDF dosyasının yolu
        annotation_channel: Annotation kanalının indeksi (None ise otomatik bulur)
    
    Returns:
        list: [{epoch: int, stage: str, start_time: float, duration: float}, ...]
    """
    print(f"EDF dosyasından annotationlar okunuyor: {edf_path}")
    
    f = pyedflib.EdfReader(edf_path)
    
    try:
        # Annotation varsa oku
        annotations = f.readAnnotations()
        f.close()
        
        stages = []
        
        for i, (onset, duration, description) in enumerate(zip(
            annotations[0], annotations[1], annotations[2]
        )):
            # Uyku evresi annotationlarını filtrele
            stage = description.strip()
            
            # Yaygın uyku evresi etiketlerini normalize et
            stage_mapping = {
                'Sleep stage W': 'W',
                'Sleep stage R': 'REM',
                'Sleep stage N1': 'N1',
                'Sleep stage N2': 'N2',
                'Sleep stage N3': 'N3',
                'Sleep stage 1': 'N1',
                'Sleep stage 2': 'N2',
                'Sleep stage 3': 'N3',
                'Sleep stage 4': 'N3',
                'Wake': 'W',
                'REM sleep': 'REM',
            }
            
            stage = stage_mapping.get(stage, stage)
            
            # Sadece uyku evresi etiketlerini al
            if stage in ['W', 'REM', 'N1', 'N2', 'N3', 'N4']:
                stages.append({
                    'epoch': len(stages),
                    'stage': stage,
                    'start_time': float(onset),
                    'duration': float(duration)
                })
        
        print(f"✓ {len(stages)} epoch parse edildi")
        return stages
        
    except Exception as e:
        f.close()
        print(f"✗ EDF annotation okuma hatası: {str(e)}")
        return []


def _update_dataset_info(patient_id):
    """
    dataset_info.json dosyasını güncelle veya oluştur.
    Her yeni hasta eklendiğinde hasta listesine eklenir.
    Offline modda get_offline_patient_ids() bu dosyayı okur.
    """
    import json
    import time
    
    info_path = os.path.join(config.OFFLINE_DATA_DIR, 'dataset_info.json')
    
    # Mevcut dosya varsa oku, yoksa boş oluştur
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
    else:
        info = {
            'db_name': config.DB_NAME,
            'use_physionet': config.USE_PHYSIONET,
            'channels': config.SELECTED_CHANNELS,
            'n_channels': config.N_CHANNELS,
            'sample_rate': config.SAMPLE_RATE,
            'epoch_duration': config.EPOCH_DURATION,
            'samples_per_epoch': config.SAMPLES_PER_EPOCH,
            'class_names': config.CLASS_NAMES,
            'class_to_idx': config.CLASS_TO_IDX,
            'patient_ids': [],
            'n_patients': 0,
        }
    
    # Hasta listesine ekle (tekrar eklenmesini önle)
    if patient_id not in info['patient_ids']:
        info['patient_ids'].append(patient_id)
        info['patient_ids'] = sorted(info['patient_ids'])
    
    info['n_patients'] = len(info['patient_ids'])
    info['last_updated'] = time.strftime('%Y-%m-%d %H:%M:%S')
    
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)


def _export_patient_npz(patient_id, sleep_stages, raw_signals):
    """
    Bir hastanın verisini .npz dosyasına export eder.
    save_all_channels_to_mongodb() ve physionet_import.py tarafından çağrılır.
    Hasta NPZ zaten mevcutsa atlar (hasta bazlı skip).
    
    NPZ içeriği:
        - stages_json: Stage verilerinin JSON string'i (bytes)
        - epoch_indices: Geçerli epoch indeks dizisi (int32)
        - signal_{channel_safe}: Her kanal için (n_epochs, samples_per_epoch) float32 array
    
    Args:
        patient_id: Hasta ID
        sleep_stages: parse_sleep_stages_xml/edf çıktısı (list of dicts)
        raw_signals: {channel_label: (signal_data_array, sample_freq)} — EDF'den okunan ham sinyaller
    """
    import json
    
    # Hasta NPZ dosyası zaten varsa atla
    os.makedirs(config.OFFLINE_DATA_DIR, exist_ok=True)
    output_path = os.path.join(config.OFFLINE_DATA_DIR, f'patient_{patient_id}.npz')
    if os.path.exists(output_path):
        print(f"  ⊘ NPZ export: {patient_id} zaten mevcut, atlanıyor")
        # dataset_info.json'da da kayıtlı olduğundan emin ol
        _update_dataset_info(patient_id)
        return
    
    # Geçerli epoch'ları filtrele
    valid_epochs = []
    for epoch_data in sleep_stages:
        if epoch_data['stage'] in config.CLASS_TO_IDX:
            valid_epochs.append(epoch_data)
    
    if len(valid_epochs) == 0:
        print(f"  ⚠ NPZ export: {patient_id} için geçerli epoch yok, atlanıyor")
        return
    
    n_epochs = len(valid_epochs)
    epoch_indices = np.array([e['epoch'] for e in valid_epochs], dtype=np.int32)
    
    # Seçili kanallar için sinyal verilerini epoch'lara böl
    channel_data = {}
    missing_channels = []
    
    for channel_name in config.SELECTED_CHANNELS:
        if channel_name not in raw_signals:
            missing_channels.append(channel_name)
            continue
        
        signal_data, sample_freq = raw_signals[channel_name]
        samples_per_epoch = int(sample_freq * config.EPOCH_DURATION)
        
        signals = np.zeros((n_epochs, config.SAMPLES_PER_EPOCH), dtype=np.float32)
        
        for i, epoch_data in enumerate(valid_epochs):
            start_time = epoch_data['start_time']
            start_sample = int(start_time * sample_freq)
            end_sample = start_sample + samples_per_epoch
            
            if end_sample <= len(signal_data):
                epoch_signal = signal_data[start_sample:end_sample].astype(np.float32)
                # sample_freq != config.SAMPLE_RATE durumu genelde olmaz ama güvenlik
                if len(epoch_signal) >= config.SAMPLES_PER_EPOCH:
                    signals[i] = epoch_signal[:config.SAMPLES_PER_EPOCH]
                else:
                    signals[i, :len(epoch_signal)] = epoch_signal
        
        channel_safe = channel_name.replace(' ', '_').replace('/', '-')
        channel_data[f'signal_{channel_safe}'] = signals
    
    if missing_channels:
        print(f"  ⚠ NPZ export: Seçili kanallar EDF'de bulunamadı: {missing_channels}")
    
    if not channel_data:
        print(f"  ⚠ NPZ export: Hiçbir seçili kanal bulunamadı, .npz oluşturulmadı")
        return
    
    # NPZ dosyasına kaydet
    save_data = {
        'stages_json': np.void(json.dumps(sleep_stages).encode('utf-8')),
        'epoch_indices': epoch_indices,
        **channel_data
    }
    
    np.savez_compressed(output_path, **save_data)
    
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"  ✓ NPZ export: {output_path} ({n_epochs} epoch, {len(channel_data)} kanal, {file_size_mb:.1f} MB)")
    
    # dataset_info.json güncelle (hasta listesi + metadata)
    _update_dataset_info(patient_id)


def save_all_channels_to_mongodb(edf_path, stages_path, stages_format='xml',
                                  mongo_uri=config.MONGO_URI,
                                  db_name=None,
                                  db_name_filtered=None,
                                  patient_id=None,
                                  save_filtered=True,
                                  save_raw=True):
    """
    EDF dosyasındaki TÜM kanalları ve uyku evresi etiketlerini GridFS'e kaydeder.

    İki veritabanına kayıt yapar:
    1. psg_data (raw): Tüm kanallar (filtresiz)
    2. psg_data_filtered: Differential kanallar (C3-M2, F4-M1, vb.) + Notch + Bandpass filtreli

    Aynı zamanda offline kullanım için .npz dosyası da oluşturur.

    Args:
        edf_path: EDF dosyasının yolu
        stages_path: Uyku evresi etiketleri dosyasının yolu (XML veya EDF)
        stages_format: 'xml' veya 'edf'
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Raw data veritabanı adı (default: config.DB_NAME_RAW)
        db_name_filtered: Filtered data veritabanı adı (default: config.DB_NAME_FILTERED)
        patient_id: Hasta ID'si (opsiyonel, dosya isimlerini gruplamak için)
        save_filtered: True ise differential+filtered verileri de kaydet

    Returns:
        dict: {
            'patient_id': str,
            'stages_id': ObjectId,
            'channels': [...]
        }
    """
    # Config'den default değerleri al
    if db_name is None:
        db_name = config.DB_NAME_RAW
    if db_name_filtered is None:
        db_name_filtered = config.DB_NAME_FILTERED

    # Hasta ID'si yoksa dosya isminden oluştur
    if patient_id is None:
        patient_id = edf_path.split('/')[-1].split('.')[0]

    print(f"{'='*70}")
    print(f"Hasta ID: {patient_id}")
    print(f"{'='*70}")

    # ================================================================
    # EPOCH TRIMMING AYARLARINI AL
    # ================================================================
    trim_start = getattr(config, 'TRIM_EPOCHS_START', 0)
    trim_end = getattr(config, 'TRIM_EPOCHS_END', 0)

    # ================================================================
    # SİNYAL SÜRESİNİ BELİRLE (EDF'den)
    # ================================================================
    print("\n[0/4] Sinyal süresi kontrol ediliyor...")
    f_check = pyedflib.EdfReader(edf_path)
    signal_duration = min(f_check.getNSamples()[i] / f_check.getSampleFrequency(i)
                         for i in range(f_check.signals_in_file))
    f_check.close()
    print(f"  Sinyal süresi: {signal_duration:.1f}s ({signal_duration/3600:.2f} saat)")

    # Uyku evrelerini parse et
    print("\n[1/4] Uyku evreleri parse ediliyor...")
    if stages_format.lower() == 'xml':
        sleep_stages = parse_sleep_stages_xml(stages_path)
    elif stages_format.lower() == 'edf':
        sleep_stages = parse_sleep_stages_edf(stages_path)
    else:
        raise ValueError(f"Desteklenmeyen format: {stages_format}. 'xml' veya 'edf' olmalı.")

    original_stage_count = len(sleep_stages)
    print(f"  Orijinal epoch sayısı: {original_stage_count}")

    # ================================================================
    # STAGE'LERİ SİNYAL SÜRESİNE GÖRE FİLTRELE VE KIRP
    # ================================================================
    valid_stages = [s for s in sleep_stages
                    if s['start_time'] + s['duration'] <= signal_duration]

    if len(valid_stages) < original_stage_count:
        removed = original_stage_count - len(valid_stages)
        print(f"  ⚠ Sinyal sınırı dışı epoch çıkarıldı: {removed}")

    if len(valid_stages) > trim_start + trim_end:
        if trim_end > 0:
            trimmed_stages = valid_stages[trim_start:-trim_end]
        else:
            trimmed_stages = valid_stages[trim_start:]

        trim_start_time = trimmed_stages[0]['start_time']
        trim_end_time = trimmed_stages[-1]['start_time'] + trimmed_stages[-1]['duration']

        print(f"  ✓ Epoch kırpma: baştan {trim_start}, sondan {trim_end}")
        print(f"    Zaman aralığı: {trim_start_time:.1f}s - {trim_end_time:.1f}s")
        print(f"    Epoch sayısı: {original_stage_count} → {len(trimmed_stages)}")
    else:
        trimmed_stages = valid_stages
        trim_start_time = 0
        trim_end_time = signal_duration
        print(f"  ⚠ Yeterli epoch yok, kırpma atlandı")

    # Stage start_time'larını sıfırdan başlayacak şekilde güncelle
    sleep_stages = []
    for i, stage in enumerate(trimmed_stages):
        sleep_stages.append({
            'epoch': i,
            'stage': stage['stage'],
            'start_time': float(i * 30),
            'duration': 30.0
        })

    # Uyku evresi istatistikleri
    stage_counts = {}
    for stage_info in sleep_stages:
        stage = stage_info['stage']
        stage_counts[stage] = stage_counts.get(stage, 0) + 1

    print(f"\nUyku evresi istatistikleri (kırpma sonrası):")
    print(f"  Toplam epoch sayısı: {len(sleep_stages)}")
    for stage, count in sorted(stage_counts.items()):
        percentage = (count / len(sleep_stages)) * 100
        print(f"  {stage}: {count} epoch ({percentage:.1f}%)")

    # MongoDB'ye bağlan
    print(f"\n[2/4] MongoDB'ye bağlanılıyor...")
    import json
    stages_json = json.dumps(sleep_stages, indent=2)
    stages_id = None

    # stages_metadata her durumda lazım (filtered DB de kullanıyor)
    stages_metadata = {
        "patient_id": patient_id,
        "data_type": "sleep_stages",
        "source_file": stages_path,
        "stages_format": stages_format,
        "total_epochs": len(sleep_stages),
        "epoch_duration": 30,
        "stage_counts": stage_counts,
        "upload_date": datetime.now()
    }

    if save_raw:
        client = MongoClient(mongo_uri)
        db = client[db_name]
        fs = gridfs.GridFS(db)

        stages_buffer = io.BytesIO()
        stages_buffer.write(stages_json.encode('utf-8'))
        stages_buffer.seek(0)

        stages_id = fs.put(
            stages_buffer,
            filename=f"{patient_id}_sleep_stages.json",
            metadata=stages_metadata
        )

        print(f"✓ Uyku evreleri kaydedildi (ID: {stages_id})")
    else:
        print(f"⊘ Raw DB yazma atlandı (save_raw=False)")

    # EDF dosyasını aç
    print(f"\n[3/4] EDF kanalları okunuyor ve kaydediliyor...")
    f = pyedflib.EdfReader(edf_path)

    channel_results = []
    raw_signals = {}  # NPZ export + filtered DB için

    try:
        n_channels = f.signals_in_file
        print(f"\nToplam kanal sayısı: {n_channels}")
        print(f"{'-'*70}")

        for channel_idx in range(n_channels):
            signal_label = f.getLabel(channel_idx)
            sample_freq = f.getSampleFrequency(channel_idx)
            n_samples_original = f.getNSamples()[channel_idx]
            physical_min = f.getPhysicalMinimum(channel_idx)
            physical_max = f.getPhysicalMaximum(channel_idx)
            digital_min = f.getDigitalMinimum(channel_idx)
            digital_max = f.getDigitalMaximum(channel_idx)

            print(f"\nKanal {channel_idx + 1}/{n_channels}: {signal_label}")
            print(f"  Frekans: {sample_freq} Hz")
            print(f"  Orijinal örnek sayısı: {n_samples_original:,}")

            # Sinyali oku (tam)
            signal_data_full = f.readSignal(channel_idx)

            # Sinyali kırpılmış epoch aralığına kes
            start_sample = int(trim_start_time * sample_freq)
            end_sample = int(trim_end_time * sample_freq)
            signal_data = signal_data_full[start_sample:end_sample]
            n_samples = len(signal_data)

            data_size_mb = signal_data.nbytes / (1024 * 1024)
            print(f"  Kırpılmış örnek sayısı: {n_samples:,}")
            print(f"  Boyut: {data_size_mb:.2f} MB")

            # Filtered DB ve NPZ export için memory'de tut
            raw_signals[signal_label] = (signal_data, sample_freq)

            file_id = None
            if save_raw:
                # Metadata hazırla
                channel_metadata = {
                    "patient_id": patient_id,
                    "channel_name": signal_label,
                    "channel_index": int(channel_idx),
                    "sample_frequency": float(sample_freq),
                    "n_samples": int(n_samples),
                    "n_samples_original": int(n_samples_original),
                    "trim_start_time": float(trim_start_time),
                    "trim_end_time": float(trim_end_time),
                    "duration_hours": float(n_samples/sample_freq/3600),
                    "physical_min": float(physical_min),
                    "physical_max": float(physical_max),
                    "digital_min": float(digital_min),
                    "digital_max": float(digital_max),
                    "data_type": "raw_signal",
                    "dtype": str(signal_data.dtype),
                    "source_file": edf_path,
                    "upload_date": datetime.now(),
                    "stages_id": stages_id,
                    "has_sleep_stages": True
                }

                buffer = io.BytesIO()
                np.save(buffer, signal_data)
                buffer.seek(0)

                file_id = fs.put(
                    buffer,
                    filename=f"{patient_id}_ch{channel_idx}_{signal_label}.npy",
                    metadata=channel_metadata,
                    chunk_size=255*1024
                )

                print(f"  ✓ Kaydedildi (ID: {file_id})")

            channel_results.append({
                'channel_idx': channel_idx,
                'channel_name': signal_label,
                'file_id': file_id,
                'sample_freq': sample_freq,
                'n_samples': n_samples,
                'data_size_mb': data_size_mb,
                'signal_data': signal_data
            })

        f.close()
        if save_raw:
            client.close()

        # ============================================================
        # NPZ Export: Offline mod için .npz oluştur (mevcutsa atlar)
        # ============================================================
        _export_patient_npz(patient_id, sleep_stages, raw_signals)

        # Özet
        print(f"\n{'='*70}")
        print(f"✓ RAW VERİLER BAŞARIYLA KAYDEDİLDİ!")
        print(f"{'='*70}")
        print(f"Hasta ID: {patient_id}")
        print(f"Uyku evreleri ID: {stages_id}")
        print(f"Toplam kanal sayısı: {len(channel_results)}")
        total_size = sum(ch['data_size_mb'] for ch in channel_results)
        print(f"Toplam veri boyutu: {total_size:.2f} MB")
        print(f"Veritabanı: {db_name}")

        # ================================================================
        # STEP 4: Differential + Filtered verileri psg_data_filtered'a kaydet
        # ================================================================
        if save_filtered:
            print(f"\n{'='*70}")
            print(f"[4/4] DIFFERENTIAL + FILTERED VERİLER KAYDEDİLİYOR")
            print(f"{'='*70}")
            print(f"Hedef veritabanı: {db_name_filtered}")

            client_filtered = MongoClient(mongo_uri)
            db_filtered = client_filtered[db_name_filtered]
            fs_filtered = gridfs.GridFS(db_filtered)

            # Sleep stages'i filtered DB'ye de kaydet
            stages_metadata_filtered = stages_metadata.copy()
            stages_metadata_filtered['source_db'] = db_name

            stages_buffer2 = io.BytesIO()
            stages_buffer2.write(stages_json.encode('utf-8'))
            stages_buffer2.seek(0)

            stages_id_filtered = fs_filtered.put(
                stages_buffer2,
                filename=f"{patient_id}_sleep_stages.json",
                metadata=stages_metadata_filtered
            )
            print(f"✓ Sleep stages kopyalandı (ID: {stages_id_filtered})")

            # Differential kanalları oluştur ve kaydet
            diff_count = 0
            for diff_name, (active_ch, ref_ch) in DIFFERENTIAL_CHANNELS.items():
                if active_ch in raw_signals and ref_ch in raw_signals:
                    signal_active, fs_active = raw_signals[active_ch]
                    signal_ref, fs_ref = raw_signals[ref_ch]

                    if fs_active != fs_ref:
                        print(f"  ⚠ {diff_name}: Farklı örnekleme hızları, atlandı")
                        continue

                    # Differential sinyal oluştur
                    diff_signal = create_differential_channel(signal_active, signal_ref)

                    # Notch + bandpass filtrele (AASM standard)
                    filtered_signal, filter_params = apply_all_filters(
                        diff_signal, fs=fs_active, channel_name=diff_name)

                    # Amplitude clipping (varsayılan: KAPALI)
                    clip_params = None
                    if USE_AMPLITUDE_CLIPPING:
                        filtered_signal, clip_params = apply_amplitude_clipping(filtered_signal)

                    # Metadata
                    diff_metadata = {
                        "patient_id": patient_id,
                        "channel_name": diff_name,
                        "differential_channel": True,
                        "active_electrode": active_ch,
                        "reference_electrode": ref_ch,
                        "sample_frequency": float(fs_active),
                        "n_samples": int(len(filtered_signal)),
                        "trim_start_time": float(trim_start_time),
                        "trim_end_time": float(trim_end_time),
                        "duration_hours": float(len(filtered_signal) / fs_active / 3600),
                        "data_type": "raw_signal",
                        "dtype": str(filtered_signal.dtype),
                        "source_file": edf_path,
                        "source_db": db_name,
                        "preprocessing_applied": True,
                        "preprocessing_params": {
                            "differential": f"{active_ch} - {ref_ch}",
                            "signal_type": filter_params['signal_type'],
                            "notch_filter": {"freq": filter_params['notch_freq'], "Q": 30.0},
                            "bandpass_filter": {
                                "lowcut": filter_params['lowcut'],
                                "highcut": filter_params['highcut'],
                                "order": filter_params['order']
                            },
                            "amplitude_clipping": clip_params if clip_params else {"applied": False}
                        },
                        "upload_date": datetime.now(),
                        "stages_id": stages_id_filtered,
                        "has_sleep_stages": True
                    }

                    buffer = io.BytesIO()
                    np.save(buffer, filtered_signal)
                    buffer.seek(0)

                    file_id = fs_filtered.put(
                        buffer,
                        filename=f"{patient_id}_{diff_name}_filtered.npy",
                        metadata=diff_metadata,
                        chunk_size=255*1024
                    )

                    diff_count += 1
                    print(f"  ✓ {diff_name} ({fs_active} Hz) → ID: {file_id}")
                else:
                    missing = []
                    if active_ch not in raw_signals:
                        missing.append(active_ch)
                    if ref_ch not in raw_signals:
                        missing.append(ref_ch)
                    print(f"  ✗ {diff_name}: Eksik kanal: {', '.join(missing)}")

            client_filtered.close()

            print(f"\n{'='*70}")
            print(f"✓ FILTERED VERİLER KAYDEDİLDİ!")
            print(f"{'='*70}")
            print(f"Differential kanal sayısı: {diff_count}")
            print(f"Veritabanı: {db_name_filtered}")

        return {
            'patient_id': patient_id,
            'stages_id': stages_id,
            'channels': channel_results
        }

    except Exception as e:
        f.close()
        client.close()
        print(f"\n✗ Hata oluştu: {str(e)}")
        raise


def read_patient_data(patient_id, mongo_uri=config.MONGO_URI, 
                      db_name=config.DB_NAME, channel_name=None):
    """
    Hasta verilerini MongoDB'den okur.
    
    Args:
        patient_id: Hasta ID'si
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
        channel_name: Belirli bir kanal okumak için (None ise tüm kanalları listeler)
    
    Returns:
        dict: Hasta verisi
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Uyku evrelerini bul
    stages_file = fs.find_one({
        "metadata.patient_id": patient_id,
        "metadata.data_type": "sleep_stages"
    })
    
    if not stages_file:
        print(f"✗ Hasta {patient_id} için uyku evreleri bulunamadı")
        client.close()
        return None
    
    # Uyku evrelerini oku
    import json
    stages_data = json.loads(stages_file.read().decode('utf-8'))
    
    # Kanalları bul
    channel_files = list(fs.find({
        "metadata.patient_id": patient_id,
        "metadata.data_type": "raw_signal"
    }))
    
    print(f"\nHasta ID: {patient_id}")
    print(f"Uyku evreleri: {len(stages_data)} epoch")
    print(f"Toplam kanal sayısı: {len(channel_files)}")
    print(f"\nMevcut kanallar:")
    
    for ch_file in channel_files:
        ch_meta = ch_file.metadata
        print(f"  - {ch_meta['channel_name']} ({ch_meta['sample_frequency']} Hz)")
    
    # Belirli bir kanal istendiyse
    if channel_name:
        print(f"\nKanal okunuyor: {channel_name}")
        for ch_file in channel_files:
            if ch_file.metadata['channel_name'] == channel_name:
                buffer = io.BytesIO(ch_file.read())
                signal_data = np.load(buffer)
                
                print(f"✓ Kanal yüklendi: {len(signal_data):,} örnek")
                
                client.close()
                return {
                    'patient_id': patient_id,
                    'channel_name': channel_name,
                    'signal_data': signal_data,
                    'metadata': ch_file.metadata,
                    'sleep_stages': stages_data
                }
        
        print(f"✗ Kanal '{channel_name}' bulunamadı")
    
    client.close()
    return {
        'patient_id': patient_id,
        'available_channels': [ch.metadata['channel_name'] for ch in channel_files],
        'sleep_stages': stages_data
    }


def list_all_patients(mongo_uri=config.MONGO_URI, 
                      db_name=config.DB_NAME,
                      show_channels_info=False):
    """
    Veritabanındaki tüm hastaları listeler.
    """
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Tüm hasta ID'lerini bul
    patient_ids = set()
    for file in fs.find({"metadata.patient_id": {"$exists": True}}):
        patient_ids.add(file.metadata['patient_id'])
    
    print(f"\nVeritabanında {len(patient_ids)} hasta bulundu:")
    print(f"{'='*70}")
    
    for patient_id in sorted(patient_ids):
        # Hasta bilgilerini al
        channel_files = list(fs.find({
            "metadata.patient_id": patient_id,
            "metadata.data_type": "raw_signal"
        }))
        channel_count = len(channel_files)

        if channel_files and show_channels_info:
            print(f"\nKanallar:")
            for ch_file in channel_files:
                ch_meta = ch_file.metadata
                size_mb = ch_file.length / (1024 * 1024)
                print(f"  - {ch_meta['channel_name']} ({ch_meta['sample_frequency']} Hz, {size_mb:.2f} MB)")
        
        stages_file = fs.find_one({
            "metadata.patient_id": patient_id,
            "metadata.data_type": "sleep_stages"
        })
        
        # Epoch sayısını oku (PhysioNet: n_epochs, Orijinal: total_epochs)
        epoch_count = 0
        if stages_file:
            epoch_count = stages_file.metadata.get('n_epochs', 
                          stages_file.metadata.get('total_epochs', 0))
        
        print(f"\nHasta ID: {patient_id}")
        print(f"  Kanal sayısı: {channel_count}")
        print(f"  Epoch sayısı: {epoch_count}")
        
        if stages_file:
            stage_counts = stages_file.metadata.get('stage_counts', {})
            print(f"  Uyku evreleri: {', '.join([f'{k}={v}' for k, v in stage_counts.items()])}")
    
    client.close()


def extract_epoch_range_data_v2(patient_id, channel_name, start_epoch, end_epoch,
                             mongo_uri=config.MONGO_URI,
                             db_name=config.DB_NAME,
                             save_to_file=False,
                             output_file=None,
                             output_format='npy',
                             verbose=True):
    """
    Belirtilen hasta, kanal ve epoch aralığına ait ham veriyi çeker.
    OPTIMIZE: 
    - Signal verisi: Partial read (GridFS chunks)
    - Stage verisi: Session cache (ilk yüklemeden sonra bellekte)
    
    
    Args:
        patient_id: Hasta ID'si
        channel_name: Kanal adı (örn: "E1", "C3-A2")
        start_epoch: Başlangıç epoch numarası (dahil)
        end_epoch: Bitiş epoch numarası (dahil)
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
        save_to_file: True ise dosyaya kaydeder
        output_file: Çıktı dosyasının yolu (None ise otomatik oluşturulur)
        output_format: Dosya formatı ('npy', 'csv', 'txt')
    
    Returns:
        dict: Hasta, kanal ve sinyal bilgilerini içeren sözlük
    """
    
    print(f"\n{'='*70}")
    print(f"VERİ ÇEKME İŞLEMİ (OPTİMİZE - PARTIAL READ)")
    print(f"{'='*70}")
    print(f"Hasta ID: {patient_id}")
    print(f"Kanal: {channel_name}")
    print(f"Epoch Aralığı: {start_epoch} - {end_epoch} (toplam {end_epoch - start_epoch + 1} epoch)")
    
    # MongoDB'ye bağlan
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    import sys
    original_stdout = sys.stdout
    if not verbose:
        sys.stdout = open(os.devnull, 'w')

    try:
        # 1. Uyku evrelerini cache'den al
        print(f"\n[1/4] Uyku evreleri okunuyor...")
        
        all_stages = get_cached_stages(patient_id, mongo_uri, db_name)
        
        if all_stages is None:
            print(f"✗ Hasta {patient_id} için uyku evreleri bulunamadı")
            client.close()
            return None
        
        # Cache'den mi geldi kontrol et
        if patient_id in _STAGES_CACHE and _STAGES_CACHE[patient_id] is all_stages:
            print(f"✓ {len(all_stages)} epoch bulundu (cache'den alındı - MongoDB'ye gidilmedi)")
        else:
            print(f"✓ {len(all_stages)} epoch bulundu (MongoDB'den yüklendi ve cache'lendi)")
        
        total_epochs = len(all_stages)
        
        # Epoch aralığını kontrol et
        if start_epoch < 0 or end_epoch >= total_epochs:
            print(f"✗ Geçersiz epoch aralığı! (Mevcut: 0-{total_epochs-1})")
            client.close()
            return None
        
        if start_epoch > end_epoch:
            print(f"✗ Başlangıç epoch'u bitiş epoch'undan büyük olamaz!")
            client.close()
            return None
        
        # İstenen epoch'ların zaman bilgilerini al (bellekte filtreleme - çok hızlı)
        stages_data = all_stages[start_epoch:end_epoch+1]
        start_time = stages_data[0]['start_time']
        end_time = stages_data[-1]['start_time'] + stages_data[-1]['duration']
        duration = end_time - start_time
        
        print(f"  Zaman aralığı: {start_time:.2f}s - {end_time:.2f}s (süre: {duration:.2f}s)")
        
        # 2. Kanal dosyasını bul (metadata'yı al)
        print(f"\n[2/4] Kanal metadata'sı alınıyor...")
        channel_file = fs.find_one({
            "metadata.patient_id": patient_id,
            "metadata.channel_name": channel_name,
            "metadata.data_type": "raw_signal"
        })
        
        if not channel_file:
            print(f"✗ Kanal '{channel_name}' bulunamadı")
            
            # Mevcut kanalları göster
            available_channels = []
            for ch_file in fs.find({
                "metadata.patient_id": patient_id,
                "metadata.data_type": "raw_signal"
            }):
                available_channels.append(ch_file.metadata['channel_name'])
            
            if available_channels:
                print(f"\nMevcut kanallar:")
                for ch in sorted(available_channels):
                    print(f"  - {ch}")
            
            client.close()
            return None
        
        # Kanal metadata'sını al
        ch_metadata = channel_file.metadata
        sample_freq = ch_metadata['sample_frequency']
        n_samples = ch_metadata.get('n_samples', 0)
        
        print(f"✓ Kanal bulundu")
        print(f"  Örnekleme frekansı: {sample_freq} Hz")
        print(f"  Toplam örnek sayısı: {n_samples:,}")
        
        # 3. Okunacak byte aralığını hesapla
        print(f"\n[3/4] Okunacak veri aralığı hesaplanıyor...")
        
        # Zaman bilgisini örnek indeksine çevir
        start_sample = int(start_time * sample_freq)
        end_sample = int(end_time * sample_freq)
        
        # Sınırları kontrol et
        start_sample = max(0, start_sample)
        end_sample = min(n_samples, end_sample)
        
        # NumPy array float64 = 8 byte per sample
        bytes_per_sample = 8
        
        # NPY dosya formatı: 128 byte header + veri
        npy_header_size = 128
        
        # Okunacak byte pozisyonları
        start_byte = npy_header_size + (start_sample * bytes_per_sample)
        end_byte = npy_header_size + (end_sample * bytes_per_sample)
        bytes_to_read = end_byte - start_byte
        
        print(f"✓ Hesaplama tamamlandı")
        print(f"  Örnek aralığı: {start_sample:,} - {end_sample:,}")
        print(f"  Byte aralığı: {start_byte:,} - {end_byte:,}")
        print(f"  Okunacak veri: {bytes_to_read / (1024*1024):.2f} MB")
        
        # 4. GridFS'den sadece gerekli bölümü oku (PARTIAL READ)
        print(f"\n[4/4] Veri kısmi okuma ile çekiliyor...")
        
        # GridFS'den sadece belirtilen byte aralığını oku
        file_id = channel_file._id
        chunks_collection = db['fs.chunks']
        
        # Chunk boyutunu al (varsayılan 255KB)
        chunk_size = channel_file.chunk_size
        
        # Hangi chunk'ları okuyacağımızı hesapla
        start_chunk_n = start_byte // chunk_size
        end_chunk_n = (end_byte - 1) // chunk_size
        
        print(f"  Chunk boyutu: {chunk_size / 1024:.0f} KB")
        print(f"  Okunacak chunk'lar: {start_chunk_n} - {end_chunk_n} (toplam {end_chunk_n - start_chunk_n + 1} chunk)")
        
        # İlgili chunk'ları çek
        chunks = chunks_collection.find({
            'files_id': file_id,
            'n': {'$gte': start_chunk_n, '$lte': end_chunk_n}
        }).sort('n', 1)
        
        # Chunk'ları birleştir
        combined_data = b''
        for chunk in chunks:
            combined_data += chunk['data']
        
        # Chunk içindeki offset'i hesapla
        offset_in_first_chunk = start_byte % chunk_size
        offset_in_last_chunk = end_byte % chunk_size
        
        # Gereksiz baş ve son kısımları çıkar
        if start_chunk_n == end_chunk_n:
            # Aynı chunk içindeyse
            combined_data = combined_data[offset_in_first_chunk:offset_in_first_chunk + bytes_to_read]
        else:
            # Farklı chunk'lardaysa
            combined_data = combined_data[offset_in_first_chunk:]
            if offset_in_last_chunk > 0:
                total_length = len(combined_data)
                excess = total_length - bytes_to_read
                if excess > 0:
                    combined_data = combined_data[:-excess]
        
        # Byte array'i numpy array'e çevir
        extracted_signal = np.frombuffer(combined_data, dtype=np.float64)
        
        print(f"✓ Veri çıkarıldı")
        print(f"  Çıkarılan örnek sayısı: {len(extracted_signal):,}")
        print(f"  Veri boyutu: {extracted_signal.nbytes / (1024*1024):.2f} MB")
        print(f"  Bellek tasarrufu: %{(1 - bytes_to_read / (n_samples * bytes_per_sample)) * 100:.1f}")
        
        # Sonuç dictionary'si
        result = {
            'patient_id': patient_id,
            'channel_name': channel_name,
            'start_epoch': start_epoch,
            'end_epoch': end_epoch,
            'epoch_count': end_epoch - start_epoch + 1,
            'signal_data': extracted_signal,
            'sample_frequency': sample_freq,
            'start_time': start_time,
            'end_time': end_time,
            'duration': duration,
            'start_sample': start_sample,
            'end_sample': end_sample
        }
        
        # 5. Dosyaya kaydet (istenirse)
        if save_to_file:
            print(f"\n[5/5] Dosyaya kaydediliyor...")
            
            # Dosya adı oluştur
            if output_file is None:
                output_file = f"{patient_id}_{channel_name}_epoch{start_epoch}-{end_epoch}.{output_format}"
            
            if output_format == 'npy':
                np.save(output_file, extracted_signal)
                print(f"✓ NumPy formatında kaydedildi: {output_file}")
            
            elif output_format == 'csv':
                np.savetxt(output_file, extracted_signal, delimiter=',', fmt='%.6f')
                print(f"✓ CSV formatında kaydedildi: {output_file}")
            
            elif output_format == 'txt':
                np.savetxt(output_file, extracted_signal, fmt='%.6f')
                print(f"✓ TXT formatında kaydedildi: {output_file}")
            
            else:
                print(f"✗ Desteklenmeyen format: {output_format}")
                output_file = None
            
            result['output_file'] = output_file
        
        client.close()
        
        print(f"\n{'='*70}")
        print(f"✓ İŞLEM TAMAMLANDI")
        print(f"{'='*70}\n")
        
        return result
        
    except Exception as e:
        print(f"\n✗ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        client.close()
        return None

    finally:
        # Her durumda stdout'u geri yükle
        sys.stdout = original_stdout


def extract_epoch_range_whole_data_for_validation(): 
    # DONT TOUCH THIS SECTION BECAUSE THIS WILL USE FOR VALIDATION
    print("")
    # def extract_epoch_range_data(patient_id, channel_name, start_epoch, end_epoch,
    #                              mongo_uri=config.MONGO_URI,
    #                              db_name=config.DB_NAME,
    #                              save_to_file=False,
    #                              output_file=None,
    #                              output_format='npy'):
    #     """
    #     Belirtilen hasta, kanal ve epoch aralığına ait ham veriyi çeker.
    #     Önce tüm kanal çekiliyor, sonra slice yaılıyor. Validation için bırakıldı.
        
    #     Args:
    #         patient_id: Hasta ID'si
    #         channel_name: Kanal adı (örn: "E1", "C3-A2")
    #         start_epoch: Başlangıç epoch numarası (dahil)
    #         end_epoch: Bitiş epoch numarası (dahil)
    #         mongo_uri: MongoDB bağlantı URI'si
    #         db_name: Veritabanı adı
    #         save_to_file: True ise dosyaya kaydeder
    #         output_file: Çıktı dosyasının yolu (None ise otomatik oluşturulur)
    #         output_format: Dosya formatı ('npy', 'csv', 'txt')
        
    #     Returns:
    #         dict: {
    #             'patient_id': str,
    #             'channel_name': str,
    #             'start_epoch': int,
    #             'end_epoch': int,
    #             'epoch_count': int,
    #             'signal_data': np.array,
    #             'sample_frequency': float,
    #             'start_time': float,
    #             'end_time': float,
    #             'duration': float,
    #             'output_file': str (eğer save_to_file=True ise)
    #         }
    #     """
        
    #     print(f"\n{'='*70}")
    #     print(f"VERİ ÇEKME İŞLEMİ")
    #     print(f"{'='*70}")
    #     print(f"Hasta ID: {patient_id}")
    #     print(f"Kanal: {channel_name}")
    #     print(f"Epoch Aralığı: {start_epoch} - {end_epoch} (toplam {end_epoch - start_epoch + 1} epoch)")
        
    #     # MongoDB'ye bağlan
    #     client = MongoClient(mongo_uri)
    #     db = client[db_name]
    #     fs = gridfs.GridFS(db)
        
    #     try:
    #         # 1. Uyku evrelerini çek
    #         print(f"\n[1/3] Uyku evreleri okunuyor...")
    #         stages_file = fs.find_one({
    #             "metadata.patient_id": patient_id,
    #             "metadata.data_type": "sleep_stages"
    #         })
            
    #         if not stages_file:
    #             print(f"✗ Hasta {patient_id} için uyku evreleri bulunamadı")
    #             client.close()
    #             return None
            
    #         import json
    #         stages_data = json.loads(stages_file.read().decode('utf-8'))
    #         total_epochs = len(stages_data)
            
    #         print(f"✓ Toplam {total_epochs} epoch bulundu")
            
    #         # Epoch aralığını kontrol et
    #         if start_epoch < 0 or end_epoch >= total_epochs:
    #             print(f"✗ Geçersiz epoch aralığı! (Mevcut: 0-{total_epochs-1})")
    #             client.close()
    #             return None
            
    #         if start_epoch > end_epoch:
    #             print(f"✗ Başlangıç epoch'u bitiş epoch'undan büyük olamaz!")
    #             client.close()
    #             return None
            
    #         # İstenen epoch'ların zaman bilgilerini al
    #         start_time = stages_data[start_epoch]['start_time']
    #         end_time = stages_data[end_epoch]['start_time'] + stages_data[end_epoch]['duration']
    #         duration = end_time - start_time
            
    #         print(f"  Zaman aralığı: {start_time:.2f}s - {end_time:.2f}s (süre: {duration:.2f}s)")
            
    #         # 2. Kanal verisini çek
    #         print(f"\n[2/3] Kanal verisi okunuyor...")
    #         channel_file = fs.find_one({
    #             "metadata.patient_id": patient_id,
    #             "metadata.channel_name": channel_name,
    #             "metadata.data_type": "raw_signal"
    #         })
            
    #         if not channel_file:
    #             print(f"✗ Kanal '{channel_name}' bulunamadı")
                
    #             # Mevcut kanalları göster
    #             available_channels = []
    #             for ch_file in fs.find({
    #                 "metadata.patient_id": patient_id,
    #                 "metadata.data_type": "raw_signal"
    #             }):
    #                 available_channels.append(ch_file.metadata['channel_name'])
                
    #             if available_channels:
    #                 print(f"\nMevcut kanallar:")
    #                 for ch in sorted(available_channels):
    #                     print(f"  - {ch}")
                
    #             client.close()
    #             return None
            
    #         # Kanal metadata'sını al
    #         ch_metadata = channel_file.metadata
    #         sample_freq = ch_metadata['sample_frequency']
            
    #         print(f"✓ Kanal bulundu")
    #         print(f"  Örnekleme frekansı: {sample_freq} Hz")
            
    #         # Tam sinyali oku
    #         buffer = io.BytesIO(channel_file.read())
    #         full_signal = np.load(buffer)
            
    #         print(f"  Toplam örnek sayısı: {len(full_signal):,}")
            
    #         # 3. Epoch aralığına karşılık gelen örnekleri çıkar
    #         print(f"\n[3/3] Epoch aralığı kesiliyor...")
            
    #         # Zaman bilgisini örnek indeksine çevir
    #         start_sample = int(start_time * sample_freq)
    #         end_sample = int(end_time * sample_freq)
            
    #         # Sınırları kontrol et
    #         start_sample = max(0, start_sample)
    #         end_sample = min(len(full_signal), end_sample)
            
    #         # İlgili bölümü kes
    #         extracted_signal = full_signal[start_sample:end_sample]
            
    #         print(f"✓ Veri çıkarıldı")
    #         print(f"  Örnek aralığı: {start_sample:,} - {end_sample:,}")
    #         print(f"  Çıkarılan örnek sayısı: {len(extracted_signal):,}")
    #         print(f"  Veri boyutu: {extracted_signal.nbytes / (1024*1024):.2f} MB")
            
    #         # Sonuç dictionary'si
    #         result = {
    #             'patient_id': patient_id,
    #             'channel_name': channel_name,
    #             'start_epoch': start_epoch,
    #             'end_epoch': end_epoch,
    #             'epoch_count': end_epoch - start_epoch + 1,
    #             'signal_data': extracted_signal,
    #             'sample_frequency': sample_freq,
    #             'start_time': start_time,
    #             'end_time': end_time,
    #             'duration': duration,
    #             'start_sample': start_sample,
    #             'end_sample': end_sample
    #         }
            
    #         # 4. Dosyaya kaydet (istenirse)
    #         if save_to_file:
    #             print(f"\n[4/4] Dosyaya kaydediliyor...")
                
    #             # Dosya adı oluştur
    #             if output_file is None:
    #                 output_file = f"{patient_id}_{channel_name}_epoch{start_epoch}-{end_epoch}.{output_format}"
                
    #             if output_format == 'npy':
    #                 np.save(output_file, extracted_signal)
    #                 print(f"✓ NumPy formatında kaydedildi: {output_file}")
                
    #             elif output_format == 'csv':
    #                 np.savetxt(output_file, extracted_signal, delimiter=',', fmt='%.6f')
    #                 print(f"✓ CSV formatında kaydedildi: {output_file}")
                
    #             elif output_format == 'txt':
    #                 np.savetxt(output_file, extracted_signal, fmt='%.6f')
    #                 print(f"✓ TXT formatında kaydedildi: {output_file}")
                
    #             else:
    #                 print(f"✗ Desteklenmeyen format: {output_format}")
    #                 output_file = None
                
    #             result['output_file'] = output_file
            
    #         client.close()
            
    #         print(f"\n{'='*70}")
    #         print(f"✓ İŞLEM TAMAMLANDI")
    #         print(f"{'='*70}\n")
            
    #         return result
            
    #     except Exception as e:
    #         print(f"\n✗ Hata oluştu: {str(e)}")
    #         import traceback
    #         traceback.print_exc()
    #         client.close()
    #         return None


def create_hypnogram(patient_id, mongo_uri=config.MONGO_URI,
                     db_name=config.DB_NAME, output_folder="hypnograms",
                     figsize=(16, 6), dpi=150, _stages_cache=None):
    """
    Belirtilen hasta için hipnogram grafiği oluşturur ve kaydeder.
    Cache sistemini kullanarak MongoDB'ye gereksiz bağlantıları önler.
    
    Args:
        patient_id: Hasta ID'si
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
        output_folder: Grafiklerin kaydedileceği klasör
        figsize: Grafik boyutu (genişlik, yükseklik)
        dpi: Grafik çözünürlüğü
        _stages_cache: Stage cache referansı (opsiyonel, dışarıdan inject edilebilir)
    
    Returns:
        str: Kaydedilen dosyanın yolu veya None
    """
    
    print(f"\n{'='*70}")
    print(f"HİPNOGRAM OLUŞTURULUYOR: {patient_id}")
    print(f"{'='*70}")
    
    try:
        # Cache'den veya get_cached_stages ile al
        if _stages_cache is not None and patient_id in _stages_cache:
            stages_data = _stages_cache[patient_id]
            print(f"✓ {len(stages_data)} epoch yüklendi (cache'den - MongoDB'ye gidilmedi)")
        else:
            # get_cached_stages fonksiyonunu kullan (otomatik cache yönetimi)
            stages_data = get_cached_stages(patient_id, mongo_uri, db_name)
            
            if stages_data is None:
                print(f"✗ Hasta {patient_id} için uyku evreleri bulunamadı")
                return None
            
            print(f"✓ {len(stages_data)} epoch yüklendi (get_cached_stages ile)")
        
        # Uyku evrelerini sayısal değerlere dönüştür
        stage_map = {
            'W': 0,      # Uyanık
            'REM': -1,   # REM uykusu
            'N1': -2,    # N1 (hafif uyku)
            'N2': -3,    # N2 (orta uyku)
            'N3': -4     # N3 (derin uyku)
        }
        
        stage_labels = {
            0: 'Wake',
            -1: 'REM',
            -2: 'N1',
            -3: 'N2',
            -4: 'N3'
        }
        
        # Renk paleti
        stage_colors = {
            0: '#FF6B6B',    # Kırmızı - Uyanık
            -1: '#4ECDC4',   # Turkuaz - REM
            -2: '#95E1D3',   # Açık yeşil - N1
            -3: '#45B7D1',   # Mavi - N2
            -4: '#2C3E50'    # Koyu mavi - N3
        }
        
        # Veriyi hazırla
        times = []
        stages = []
        
        for epoch_data in stages_data:
            stage_name = epoch_data['stage']
            start_time = epoch_data['start_time']
            duration = epoch_data['duration']
            
            if stage_name in stage_map:
                times.append(start_time / 3600)  # Saate çevir
                stages.append(stage_map[stage_name])
        
        times = np.array(times)
        stages = np.array(stages)
        
        # İstatistikleri hesapla
        total_time_hours = times[-1] if len(times) > 0 else 0
        stage_stats = {}
        
        for stage_name, stage_value in stage_map.items():
            count = np.sum(stages == stage_value)
            duration_hours = count * 30 / 3600  # 30 saniyelik epoch'lar
            percentage = (count / len(stages)) * 100 if len(stages) > 0 else 0
            stage_stats[stage_name] = {
                'count': count,
                'duration_hours': duration_hours,
                'percentage': percentage
            }
        
        # Grafik oluştur
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Her epoch için dikdörtgen çiz
        epoch_duration = 30 / 3600  # 30 saniye = saat cinsinden
        
        for i in range(len(times)):
            color = stage_colors[stages[i]]
            rect = mpatches.Rectangle(
                (times[i], stages[i] - 0.4),
                epoch_duration,
                0.8,
                facecolor=color,
                edgecolor='none',
                alpha=0.9
            )
            ax.add_patch(rect)
        
        # Eksen ayarları
        # Son epoch'u tamamen göstermek için epoch_duration ekle
        max_time = times[-1] + epoch_duration if len(times) > 0 else 8
        ax.set_xlim(0, max(max_time, 8))  # 0'dan başla, en az 8 saat göster
        ax.set_ylim(-4.5, 0.5)
        
        # Y ekseni etiketleri
        ax.set_yticks([0, -1, -2, -3, -4])
        ax.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3'])
        
        # X ekseni (saat formatında)
        ax.set_xlabel('Zaman (Saat)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Uyku Evresi', fontsize=12, fontweight='bold')
        
        # Grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.grid(True, axis='y', alpha=0.2)
        
        # Başlık
        title = f'Hipnogram - Hasta: {patient_id}\n'
        title += f'Toplam Kayıt Süresi: {total_time_hours:.1f} saat | '
        title += f'Toplam Epoch: {len(stages_data)}'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # İstatistik kutusu
        stats_text = 'Uyku Evresi Dağılımı:\n'
        stats_text += '-' * 30 + '\n'
        
        for stage_name in ['W', 'REM', 'N1', 'N2', 'N3']:
            stats = stage_stats[stage_name]
            stats_text += f'{stage_labels[stage_map[stage_name]]}: '
            stats_text += f'{stats["duration_hours"]:.1f}h ({stats["percentage"]:.1f}%)\n'
        
        # İstatistik kutusunu sağ üst köşeye ekle
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
        ax.text(0.98, 0.97, stats_text,
                transform=ax.transAxes,
                fontsize=10,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=props,
                family='monospace')
        
        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=stage_colors[0], label='Wake (Uyanık)', alpha=0.9),
            mpatches.Patch(facecolor=stage_colors[-1], label='REM Sleep', alpha=0.9),
            mpatches.Patch(facecolor=stage_colors[-2], label='N1 (Hafif Uyku)', alpha=0.9),
            mpatches.Patch(facecolor=stage_colors[-3], label='N2 (Orta Uyku)', alpha=0.9),
            mpatches.Patch(facecolor=stage_colors[-4], label='N3 (Derin Uyku)', alpha=0.9)
        ]
        ax.legend(handles=legend_elements, loc='upper left', 
                 framealpha=0.9, fontsize=9)
        
        plt.tight_layout()
        
        # Klasörü oluştur
        os.makedirs(output_folder, exist_ok=True)
        
        # Dosya adı ve yolu
        output_path = os.path.join(output_folder, f"{patient_id}_hypnogram.png")
        
        # Kaydet
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"✓ Hipnogram kaydedildi: {output_path}")
        print(f"  Boyut: {figsize[0]}x{figsize[1]} inch, {dpi} DPI")
        print(f"{'='*70}\n")
        
        return output_path
        
    except Exception as e:
        print(f"✗ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        return None


def create_all_hypnograms(mongo_uri=config.MONGO_URI,
                          db_name=config.DB_NAME,
                          output_folder="hypnograms",
                          patient_ids=None,
                          figsize=(16, 6),
                          dpi=150,
                          use_cache=True):
    """
    Veritabanındaki tüm hastalar için hipnogram oluşturur.
    Cache sistemini kullanarak performanslı çalışır.
    
    Args:
        mongo_uri: MongoDB bağlantı URI'si
        db_name: Veritabanı adı
        output_folder: Grafiklerin kaydedileceği klasör
        patient_ids: İşlenecek hasta ID listesi (None ise tümü)
        figsize: Grafik boyutu
        dpi: Grafik çözünürlük
        use_cache: True ise global cache'i kullanır, False ise her seferinde DB'den okur
    
    Returns:
        dict: İşlem özeti
    """
    
    print(f"\n{'='*70}")
    print(f"TOPLU HİPNOGRAM OLUŞTURMA")
    print(f"{'='*70}")
    print(f"Çıktı klasörü: {output_folder}")
    print(f"Veritabanı: {db_name}")
    print(f"Cache kullanımı: {'Aktif ✓' if use_cache else 'Pasif'}")
    print(f"{'='*70}\n")
    
    # Global cache'i import et
    if use_cache:
        # Hasta listesini belirle
        if patient_ids is None:
            # Cache'de varsa oradan al
            if _STAGES_CACHE:
                patient_ids = sorted(_STAGES_CACHE.keys())
                print(f"Hasta listesi cache'den alındı: {len(patient_ids)} hasta\n")
            else:
                # Cache boşsa MongoDB'den hasta listesini al
                client = MongoClient(mongo_uri)
                db = client[db_name]
                fs = gridfs.GridFS(db)
                
                patient_ids = set()
                for file in fs.find({"metadata.data_type": "sleep_stages"}):
                    patient_ids.add(file.metadata['patient_id'])
                patient_ids = sorted(patient_ids)
                
                client.close()
                
                print(f"Toplam {len(patient_ids)} hasta bulundu (MongoDB'den)\n")
                
                # Tüm stage verilerini önceden yükle
                print("Performans için tüm stage verileri cache'e yükleniyor...")
                preload_all_stages(patient_ids, mongo_uri, db_name)
        else:
            # Belirtilen hastalar için cache'i kontrol et
            missing_patients = [pid for pid in patient_ids if pid not in _STAGES_CACHE]
            
            if missing_patients:
                print(f"Cache'de olmayan {len(missing_patients)} hasta için veri yükleniyor...")
                preload_all_stages(missing_patients, mongo_uri, db_name)
        
        # Cache referansını kullan
        cache_ref = _STAGES_CACHE
    else:
        # Cache kullanılmayacak
        cache_ref = None
        
        # Hasta listesini belirle
        if patient_ids is None:
            client = MongoClient(mongo_uri)
            db = client[db_name]
            fs = gridfs.GridFS(db)
            
            patient_ids = set()
            for file in fs.find({"metadata.data_type": "sleep_stages"}):
                patient_ids.add(file.metadata['patient_id'])
            patient_ids = sorted(patient_ids)
            
            client.close()
            print(f"Toplam {len(patient_ids)} hasta bulundu.\n")
    
    results = {
        'total': len(patient_ids),
        'success': 0,
        'failed': 0,
        'details': []
    }
    
    # Her hasta için hipnogram oluştur
    for idx, patient_id in enumerate(patient_ids, 1):
        print(f"[{idx}/{len(patient_ids)}] İşleniyor: {patient_id}")
        
        try:
            output_path = create_hypnogram(
                patient_id=patient_id,
                mongo_uri=mongo_uri,
                db_name=db_name,
                output_folder=output_folder,
                figsize=figsize,
                dpi=dpi,
                _stages_cache=cache_ref
            )
            
            if output_path:
                results['success'] += 1
                results['details'].append({
                    'patient_id': patient_id,
                    'status': 'success',
                    'output_path': output_path
                })
            else:
                results['failed'] += 1
                results['details'].append({
                    'patient_id': patient_id,
                    'status': 'failed',
                    'reason': 'unknown'
                })
                
        except Exception as e:
            print(f"✗ Hata: {patient_id} - {str(e)}\n")
            results['failed'] += 1
            results['details'].append({
                'patient_id': patient_id,
                'status': 'failed',
                'reason': str(e)
            })
    
    # Özet rapor
    print(f"\n{'='*70}")
    print(f"İŞLEM ÖZETİ")
    print(f"{'='*70}")
    print(f"Toplam hasta: {results['total']}")
    print(f"Başarılı: {results['success']} ✓")
    print(f"Başarısız: {results['failed']} ✗")
    print(f"Çıktı klasörü: {os.path.abspath(output_folder)}")
    
    if use_cache:
        print(f"Cache kullanımı: {len(_STAGES_CACHE)} hasta bellekte tutuldu")
    
    print(f"{'='*70}\n")
    
    if results['failed'] > 0:
        print("Başarısız işlemler:")
        for detail in results['details']:
            if detail['status'] == 'failed':
                print(f"  - {detail['patient_id']}: {detail.get('reason', 'unknown')}")
    
    return results


# Kullanım örneği
if __name__ == "__main__":
    
    # ====================================================================
    # TOPLU AKTARIM - USE_PHYSIONET flag'ine göre doğru import yapılır
    # ====================================================================
    
    if config.USE_PHYSIONET:
        # PhysioNet format: PSG.edf + Hypnogram.edf
        from tool.physionet_import import import_all_physionet_patients
        
        # root_folder = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/physionet_sleep/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
        # Not: sleep-telemetry-raw klasöründe 7 Hypnogram EDF bozuk (EDF+ Recordingfield hatası)
        #      sleep-telemetry-edited klasöründe düzeltilmiş versiyonları var
        #root_folder = "/media/mehmet/40BC0D26BC0D17D41/sleep-staging-data/physionet_data/sleep-telemetry-edited"
        root_folder = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/physionet_sleep/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
        
        print(f"\n{'='*70}")
        print(f"PHYSIONET IMPORT MODU")
        print(f"{'='*70}")
        print(f"Kaynak: {root_folder}")
        print(f"Hedef DB: {config.DB_NAME}")
        print(f"{'='*70}\n")
        
        results = import_all_physionet_patients(
            data_dir=root_folder,
            mongo_uri=config.MONGO_URI,
            db_name=config.DB_NAME
        )
    else:
        # Orijinal format: EDF + XML labels
        root_folder = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/psg_data_phase3/data/raw_data"

        results = process_patient_folders(
            root_folder=root_folder,
            mongo_uri=config.MONGO_URI,
            db_name=config.DB_NAME_RAW,           # psg_data (raw)
            db_name_filtered=config.DB_NAME_FILTERED,  # psg_data_filtered
            skip_existing=True,  # filtered DB'de olan hastaları atla
            save_raw=False       # psg_data (raw) zaten mevcut, tekrar yazma
        )

    # Stage verilerini cache'e yükle
    
    preload_all_stages(
        mongo_uri=config.MONGO_URI,
        db_name=config.DB_NAME
    )

    # Cache'i temizler (gerekirse)
    """ clear_stages_cache() """

        # Örnek 1: Tek bir hasta için hipnogram oluştur
    """ 
    create_hypnogram(
        patient_id="patient_001",
        output_folder="hypnograms"
    )
    """
    
    # Örnek 2: Tüm hastalar için hipnogram oluştur
    
    """     
    results = create_all_hypnograms(
        output_folder="hypnograms",
        figsize=(16, 6),
        dpi=150
    ) """
    
    # Örnek 3: Belirli hastalar için oluştur
    
    """ patient_list = ["patient_001", "patient_002", "patient_003"]
    results = create_all_hypnograms(
        patient_ids=patient_list,
        output_folder="my_hypnograms"
    )
    
    print("Just Now Any Info did not write") """

    # ====================================================================
    # VERİ LİSTELEME/OKUMA/SİLME
    # ====================================================================

    # Tüm hastaları listele
    list_all_patients(
        mongo_uri=config.MONGO_URI,
        db_name=config.DB_NAME,
        show_channels_info=True
    )

    # Belirli hastanın/hastaların verilerini sil
    """ delete_patient_data(
        patient_id="53ab11aa-3f94-41c1-9d1e-48767cce3c26",
        mongo_uri=config.MONGO_URI,
        db_name=config.DB_NAME
    ) """

    # %15 den fazla uyanıklığı olan hastalar çıkarıldı
    """ delete_patient_datas(
        patient_ids=["6b19b899-e4b2-4465-93d7-ab2c329b092d",
                    "255fc64b-7e43-4525-b6f7-109e6c7a0f0b",
                    "596ce269-a11b-4dce-859e-9b4f25428033",
                    "66d496f0-3092-48db-801e-ae81fb623131",
                    "1319073d-138e-44da-8536-1c3c35752347",
                    "f94c4f40-d9e5-4624-b4f9-b02d35285646"
                    ],
        mongo_uri=config.MONGO_URI,
        db_name=config.DB_NAME,
        confirm=False
    ) """


    # Belirli bir hastanın verilerini oku, dosyaya yaz
    
    """ patient_data = read_patient_data(
        patient_id="4f680835-27cb-422f-b254-1d2994ff548a",
        mongo_uri=config.MONGO_URI,
        db_name=config.DB_NAME,
        channel_name="O1"
    ) """

    # np.savetxt("6b19b899-e4b2-4465-93d7-ab2c329b092d_C3.txt", patient_data["signal_data"]) # Tüm veriyi bir dosyaya yazdırmak için


    # # KANAL İŞLEMLERİ

    # ===================================================================
    # ÖRNEK: Tek kanal, tek epoch aralığı - RAW DATA
    # ===================================================================
    """ print("\n\n")
    print("=" * 80)
    print("ÖRNEK 5: Tek kanal, tek epoch aralığı - RAW DATA")
    print("=" * 80) """

    """ result_whole_data = extract_epoch_range_whole_data_for_validation(
        patient_id="3da2681a-5991-449c-80a4-52cf3257db02",
        channel_name="Activity",
        start_epoch=17,
        end_epoch=23,
        save_to_file=True,
        output_format='txt'
    ) """

    """ result_with_partial_read = extract_epoch_range_data_v2(
        patient_id="3da2681a-5991-449c-80a4-52cf3257db02",
        channel_name="Activity",
        start_epoch=17,
        end_epoch=23,
        save_to_file=True,
        output_format='txt'
    ) """

    """ if result_with_partial_read:
        print(f"\nVeri çekildi:")
        print(f"  Signal shape: {result_with_partial_read['signal_data'].shape}")
        print(f"  Sample frequency: {result_with_partial_read['sample_frequency']} Hz")
        print(f"  Duration: {result_with_partial_read['duration']:.2f} seconds") """


    """ # Örnek: Aynı hasta için tekrar çekme (cache'den alır - çok hızlı)
    result1 = extract_epoch_range_data_v2(
        patient_id='patient_001',  # Aynı hasta
        channel_name='01',       # Farklı kanal
        start_epoch=200,
        end_epoch=249,
        mongo_uri=config.MONGO_URI,
        db_name=config.DB_NAME,
        output_format="txt"
    ) """
