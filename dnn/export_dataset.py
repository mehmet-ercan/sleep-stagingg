"""
Sleep Stage Classification - Dataset Export Script
MongoDB'deki verileri .npz dosyalarına export eder.

Kaggle/Colab gibi MongoDB olmayan ortamlarda kullanmak için:
1. Bu scripti MongoDB'nin çalıştığı lokal makinede çalıştır
2. Oluşan offline_data/ klasörünü zip'le
3. Kaggle Dataset veya Colab'a yükle
4. config.py'de USE_OFFLINE_DATA = True yap

Kullanım:
    cd dnn
    python export_dataset.py

Çıktı yapısı:
    offline_data/
        dataset_info.json       → metadata (hasta listesi, kanallar, sample rate)
        patient_SC4001E0.npz    → hasta bazlı sinyal + evre verileri
        patient_SC4002E0.npz    → ...
"""

import numpy as np
import json
import os
import sys
import time
from tqdm import tqdm

import config

# MongoDB sadece export sırasında gerekli
from pymongo import MongoClient
import gridfs
from edf_to_mongo import (preload_all_stages, get_cached_stages, 
                           extract_epoch_range_data_v2, _STAGES_CACHE)


def export_patient(patient_id, fs, output_dir, channels, mongo_uri, db_name):
    """
    Tek bir hastanın tüm verilerini .npz dosyasına export et.
    
    NPZ içeriği:
        - stages_json: Stage verilerinin JSON string'i (bytes)
        - signal_{channel_safe}: Her kanal için (n_epochs, samples_per_epoch) array
        - epoch_indices: Epoch indeks dizisi
    
    Args:
        patient_id: Hasta ID
        fs: GridFS instance
        output_dir: Çıktı dizini
        channels: Kanal listesi
        mongo_uri: MongoDB URI
        db_name: DB adı
    
    Returns:
        dict: Export istatistikleri
    """
    # Stage verilerini al
    stages_data = get_cached_stages(patient_id, mongo_uri, db_name)
    if stages_data is None:
        print(f"  ⚠ {patient_id}: Stage verisi bulunamadı, atlanıyor")
        return None
    
    # Geçerli epoch'ları filtrele
    valid_epochs = []
    for epoch_data in stages_data:
        if epoch_data['stage'] in config.CLASS_TO_IDX:
            valid_epochs.append(epoch_data)
    
    if len(valid_epochs) == 0:
        print(f"  ⚠ {patient_id}: Geçerli epoch bulunamadı")
        return None
    
    n_epochs = len(valid_epochs)
    epoch_indices = [e['epoch'] for e in valid_epochs]
    
    # Her kanal için sinyal verilerini topla
    channel_signals = {}
    
    for channel_name in channels:
        # Kanal adını dosya-güvenli yap (boşluk → _, tire korunur)
        channel_safe = channel_name.replace(' ', '_').replace('/', '-')
        
        signals = np.zeros((n_epochs, config.SAMPLES_PER_EPOCH), dtype=np.float32)
        
        for i, epoch_data in enumerate(valid_epochs):
            epoch_idx = epoch_data['epoch']
            
            # Stdout'u bastır (extract_epoch_range_data_v2 çok konuşkan)
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            
            try:
                result = extract_epoch_range_data_v2(
                    patient_id=patient_id,
                    channel_name=channel_name,
                    start_epoch=epoch_idx,
                    end_epoch=epoch_idx,
                    mongo_uri=mongo_uri,
                    db_name=db_name,
                    save_to_file=False
                )
                sys.stdout = old_stdout
                
                if result is not None:
                    signal = result['signal_data'].astype(np.float32)
                    # Boyut uyumu
                    if len(signal) >= config.SAMPLES_PER_EPOCH:
                        signals[i] = signal[:config.SAMPLES_PER_EPOCH]
                    else:
                        signals[i, :len(signal)] = signal
                        
            except Exception as e:
                sys.stdout = old_stdout
                # Hata durumunda sıfır bırak
                pass
        
        channel_signals[f'signal_{channel_safe}'] = signals
    
    # NPZ dosyasına kaydet
    output_path = os.path.join(output_dir, f'patient_{patient_id}.npz')
    
    # Kaydedilecek veriler
    save_data = {
        'stages_json': np.void(json.dumps(stages_data).encode('utf-8')),
        'epoch_indices': np.array(epoch_indices, dtype=np.int32),
        **channel_signals
    }
    
    np.savez_compressed(output_path, **save_data)
    
    # Dosya boyutu
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    
    return {
        'patient_id': patient_id,
        'n_epochs': n_epochs,
        'n_channels': len(channels),
        'file_size_mb': file_size_mb
    }


def main():
    """
    MongoDB'deki tüm veriyi offline_data/ klasörüne export et.
    """
    print(f"\n{'='*70}")
    print(f"DATASET EXPORT - MongoDB → NPZ")
    print(f"{'='*70}")
    print(f"Database: {config.DB_NAME}")
    print(f"Kanallar: {config.SELECTED_CHANNELS}")
    print(f"Sample rate: {config.SAMPLE_RATE} Hz")
    print(f"Samples per epoch: {config.SAMPLES_PER_EPOCH}")
    print(f"Çıktı dizini: {config.OFFLINE_DATA_DIR}")
    print(f"{'='*70}\n")
    
    # Çıktı dizinini oluştur
    os.makedirs(config.OFFLINE_DATA_DIR, exist_ok=True)
    
    # MongoDB'ye bağlan
    print("MongoDB'ye bağlanılıyor...")
    client = MongoClient(config.MONGO_URI)
    db = client[config.DB_NAME]
    fs = gridfs.GridFS(db)
    
    # Hasta listesini al
    all_patient_ids = set()
    for file in fs.find({"metadata.data_type": "sleep_stages"}):
        all_patient_ids.add(file.metadata['patient_id'])
    all_patient_ids = sorted(all_patient_ids)
    
    print(f"✓ {len(all_patient_ids)} hasta bulundu\n")
    
    # Stage verilerini cache'e yükle (hız için)
    print("Stage verileri cache'e yükleniyor...")
    preload_all_stages(all_patient_ids, config.MONGO_URI, config.DB_NAME)
    print(f"✓ {len(_STAGES_CACHE)} hasta cache'de\n")
    
    # Her hasta için export
    print(f"\n{'='*70}")
    print(f"HASTALAR EXPORT EDİLİYOR")
    print(f"{'='*70}\n")
    
    export_stats = []
    start_time = time.time()
    
    for idx, patient_id in enumerate(tqdm(all_patient_ids, desc="Export")):
        print(f"\n[{idx+1}/{len(all_patient_ids)}] {patient_id} export ediliyor...")
        
        stats = export_patient(
            patient_id=patient_id,
            fs=fs,
            output_dir=config.OFFLINE_DATA_DIR,
            channels=config.SELECTED_CHANNELS,
            mongo_uri=config.MONGO_URI,
            db_name=config.DB_NAME
        )
        
        if stats:
            export_stats.append(stats)
            print(f"  ✓ {stats['n_epochs']} epoch, {stats['file_size_mb']:.1f} MB")
    
    client.close()
    
    # Dataset bilgilerini kaydet
    dataset_info = {
        'db_name': config.DB_NAME,
        'use_physionet': config.USE_PHYSIONET,
        'channels': config.SELECTED_CHANNELS,
        'n_channels': config.N_CHANNELS,
        'sample_rate': config.SAMPLE_RATE,
        'epoch_duration': config.EPOCH_DURATION,
        'samples_per_epoch': config.SAMPLES_PER_EPOCH,
        'class_names': config.CLASS_NAMES,
        'class_to_idx': config.CLASS_TO_IDX,
        'patient_ids': all_patient_ids,
        'n_patients': len(all_patient_ids),
        'export_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'export_stats': export_stats
    }
    
    info_path = os.path.join(config.OFFLINE_DATA_DIR, 'dataset_info.json')
    with open(info_path, 'w') as f:
        json.dump(dataset_info, f, indent=2)
    
    # Özet
    elapsed = time.time() - start_time
    total_size_mb = sum(s['file_size_mb'] for s in export_stats)
    total_epochs = sum(s['n_epochs'] for s in export_stats)
    
    print(f"\n{'='*70}")
    print(f"EXPORT TAMAMLANDI!")
    print(f"{'='*70}")
    print(f"  Toplam hasta: {len(export_stats)}")
    print(f"  Toplam epoch: {total_epochs:,}")
    print(f"  Toplam boyut: {total_size_mb:.1f} MB ({total_size_mb/1024:.2f} GB)")
    print(f"  Süre: {elapsed:.0f} saniye ({elapsed/60:.1f} dakika)")
    print(f"  Çıktı: {config.OFFLINE_DATA_DIR}/")
    print(f"\n📋 SONRAKİ ADIMLAR:")
    print(f"  1. cd {config.OFFLINE_DATA_DIR} && zip -r offline_data.zip .")
    print(f"  2. Zip'i Kaggle Dataset veya Colab'a yükle")
    print(f"  3. config.py'de USE_OFFLINE_DATA = True yap")
    print(f"  4. Gerekirse OFFLINE_DATA_DIR yolunu güncelle")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
