#!/usr/bin/env python
# coding: utf-8

"""
PhysioNet Importer
PhysioNet Sleep Telemetry verilerini MongoDB'ye ve NPZ'ye aktarır.
MongoDB yoksa sadece NPZ üretir (Kaggle/Colab desteği).
"""

import numpy as np
import json

try:
    import gridfs
    from pymongo import MongoClient
    HAS_PYMONGO = True
except ImportError:
    HAS_PYMONGO = False

from tool.physionet_parser import (parse_physionet_psg, parse_physionet_hypnogram, 
                                   find_patient_files, trim_wake_epochs)
from edf_to_mongo import _export_patient_npz, _update_dataset_info
import config


def import_physionet_patient(psg_path, hypno_path, 
                             mongo_uri="mongodb://localhost:27017/",
                             db_name="physionet_sleep",
                             dataset_type="ST"):
    """
    PhysioNet formatındaki bir hastayı MongoDB'ye ve NPZ'ye aktar.
    SC dataset'inde wake trimming otomatik uygulanır.
    
    Args:
        psg_path: PSG.edf dosya yolu
        hypno_path: Hypnogram.edf dosya yolu
        mongo_uri: MongoDB URI
        db_name: Database adı
        dataset_type: "ST" veya "SC"
    
    Returns:
        bool: Başarılı ise True
    """
    source_label = f"physionet_sleep_{'cassette' if dataset_type == 'SC' else 'telemetry'}"
    try:
        # 1. PSG parse et
        psg_data = parse_physionet_psg(psg_path)
        patient_id = psg_data['patient_id']
        
        # 2. Hypnogram parse et
        hypno_data = parse_physionet_hypnogram(hypno_path)
        
        # 2.5 SC dataset ise wake trimming uygula
        if dataset_type == "SC":
            original_count = len(hypno_data)
            hypno_data = trim_wake_epochs(hypno_data)
            if not hypno_data:
                print(f"  ⚠ {patient_id}: Wake trimming sonrası epoch kalmadı, atlanıyor")
                return False
            print(f"  ✂ Wake trimming: {original_count} → {len(hypno_data)} epoch")
        
        # 2.6 Sadece seçili kanalları filtrele (SC: 100Hz kanallar)
        filtered_channels = {}
        for ch_name, ch_signal in psg_data['channels'].items():
            if ch_name in config.SELECTED_CHANNELS:
                filtered_channels[ch_name] = ch_signal
            else:
                print(f"  ⊘ Kanal atlandı (config'de yok): {ch_name}")
        psg_data['channels'] = filtered_channels
        
        # 3. MongoDB'ye kaydet (varsa ve bağlanılabiliyorsa)
        mongo_saved = False
        if HAS_PYMONGO:
            try:
                client = MongoClient(mongo_uri, serverSelectionTimeoutMS=5000)
                # Bağlantıyı test et (hızlı timeout)
                client.server_info()
                db = client[db_name]
                fs = gridfs.GridFS(db)
                
                # 4. Mevcut veriyi kontrol et (skip existing)
                existing = fs.find_one({
                    "metadata.patient_id": patient_id,
                    "metadata.data_type": "sleep_stages"
                })
                
                if existing:
                    print(f"  ⊘ {patient_id}: MongoDB'de zaten var, atlanıyor")
                    client.close()
                    # MongoDB'de var ama NPZ yoksa üret
                    raw_signals = {
                        ch_name: (ch_signal, psg_data['sample_rate'])
                        for ch_name, ch_signal in psg_data['channels'].items()
                    }
                    _export_patient_npz(patient_id, hypno_data, raw_signals)
                    return False
                
                print(f"\n{'='*70}")
                print(f"MONGODB'YE AKTARILIYOR: {patient_id}")
                print(f"{'='*70}")
                
                # 5. Sleep stages'i kaydet
                stages_json = json.dumps(hypno_data, indent=2)
                stages_bytes = stages_json.encode('utf-8')
                
                stages_metadata = {
                    'patient_id': patient_id,
                    'data_type': 'sleep_stages',
                    'n_epochs': len(hypno_data),
                    'source': source_label
                }
                
                fs.put(stages_bytes, 
                       filename=f"{patient_id}_sleep_stages.json",
                       metadata=stages_metadata)
                
                print(f"  ✓ Sleep stages kaydedildi: {len(hypno_data)} epoch")
                
                # 6. Her kanalı ayrı ayrı kaydet
                for channel_name, signal_data in psg_data['channels'].items():
                    # NumPy array'i binary'ye çevir
                    signal_bytes = signal_data.astype(np.float64).tobytes()
                    
                    channel_metadata = {
                        'patient_id': patient_id,
                        'data_type': 'raw_signal',
                        'channel_name': channel_name,
                        'sample_frequency': psg_data['sample_rate'],
                        'n_samples': len(signal_data),
                        'duration': psg_data['duration'],
                        'source': source_label
                    }
                    
                    fs.put(signal_bytes,
                           filename=f"{patient_id}_{channel_name}.npy",
                           metadata=channel_metadata)
                    
                    print(f"  ✓ {channel_name}: {len(signal_data):,} samples")
                
                client.close()
                mongo_saved = True
            except Exception as mongo_err:
                print(f"\n  ℹ MongoDB bağlantısı başarısız ({type(mongo_err).__name__}), sadece NPZ export yapılacak: {patient_id}")
        else:
            print(f"\n  ℹ MongoDB mevcut değil, sadece NPZ export yapılacak: {patient_id}")
        
        # 7. NPZ export — offline mod için (MongoDB olsun olmasın her zaman çalışır)
        raw_signals = {
            ch_name: (ch_signal, psg_data['sample_rate'])
            for ch_name, ch_signal in psg_data['channels'].items()
        }
        _export_patient_npz(patient_id, hypno_data, raw_signals)
        
        print(f"✓ {patient_id} başarıyla kaydedildi!")
        print(f"{'='*70}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ HATA: {patient_id}")
        print(f"  {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def import_all_physionet_patients(data_dir, 
                                  mongo_uri="mongodb://localhost:27017/",
                                  db_name="physionet_sleep",
                                  dataset_type="ST"):
    """
    Tüm PhysioNet hastalarını MongoDB'ye ve NPZ'ye aktar
    
    Args:
        data_dir: Veri klasörü
        mongo_uri: MongoDB URI
        db_name: Database adı
        dataset_type: "ST" veya "SC"
    
    Returns:
        dict: İşlem özeti
    """
    print(f"\n{'='*70}")
    print(f"TOPLU PHYSIONET IMPORT ({dataset_type})")
    print(f"{'='*70}")
    print(f"Kaynak: {data_dir}")
    print(f"Hedef DB: {db_name}")
    if dataset_type == "SC":
        print(f"Wake trimming: AKTİF")
        print(f"Kanallar: {config.SELECTED_CHANNELS}")
    print(f"{'='*70}\n")
    
    # Hasta dosyalarını bul
    patient_pairs = find_patient_files(data_dir, dataset_type=dataset_type)
    
    if not patient_pairs:
        print("✗ Hasta dosyası bulunamadı!")
        return {'total': 0, 'success': 0, 'failed': 0, 'skipped': 0}
    
    # İstatistikler
    stats = {
        'total': len(patient_pairs),
        'success': 0,
        'failed': 0,
        'skipped': 0
    }
    
    # Her hastayı import et
    for idx, (psg_path, hypno_path) in enumerate(patient_pairs, 1):
        print(f"\n[{idx}/{len(patient_pairs)}] Import ediliyor...")
        
        result = import_physionet_patient(psg_path, hypno_path, mongo_uri, db_name, 
                                           dataset_type=dataset_type)
        
        if result:
            stats['success'] += 1
        elif result is False:  # Skipped
            stats['skipped'] += 1
        else:
            stats['failed'] += 1
    
    # Özet rapor
    print(f"\n{'='*70}")
    print(f"IMPORT ÖZETİ")
    print(f"{'='*70}")
    print(f"Toplam hasta: {stats['total']}")
    print(f"Başarılı: {stats['success']} ✓")
    print(f"Atlandı: {stats['skipped']} ⊘")
    print(f"Başarısız: {stats['failed']} ✗")
    print(f"{'='*70}\n")
    
    return stats


if __name__ == "__main__":
    dataset_type = config.PHYSIONET_DATASET  # config.py'den oku ("SC" veya "ST")
    
    if dataset_type == "SC":
        data_dir = config.SC_RAW_DATA_DIR
        db_name = config.DB_NAME  # "physionet_sleep_sc"
    else:
        data_dir = "/mnt/ssd2/2.SLEEP STAGING/1.DATA/physionet_sleep/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
        db_name = config.DB_NAME  # "physionet_sleep"
    
    mongo_uri = config.MONGO_URI
    
    # Tüm hastaları import et
    stats = import_all_physionet_patients(data_dir, mongo_uri, db_name, 
                                          dataset_type=dataset_type)
    
    print("\n✓ Import işlemi tamamlandı!")
    print(f"  Başarılı: {stats['success']}/{stats['total']}")
