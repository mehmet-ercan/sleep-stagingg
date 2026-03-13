#!/usr/bin/env python
# coding: utf-8

"""
PhysioNet to MongoDB Importer
Imports PhysioNet Sleep Telemetry data to MongoDB
"""

import numpy as np
import gridfs
from pymongo import MongoClient
import json
from tool.physionet_parser import parse_physionet_psg, parse_physionet_hypnogram, find_patient_files


def import_physionet_patient(psg_path, hypno_path, 
                             mongo_uri="mongodb://localhost:27017/",
                             db_name="physionet_sleep"):
    """
    PhysioNet formatındaki bir hastayı MongoDB'ye aktar
    
    Args:
        psg_path: PSG.edf dosya yolu
        hypno_path: Hypnogram.edf dosya yolu
        mongo_uri: MongoDB URI
        db_name: Database adı
    
    Returns:
        bool: Başarılı ise True
    """
    try:
        # 1. PSG parse et
        psg_data = parse_physionet_psg(psg_path)
        patient_id = psg_data['patient_id']
        
        # 2. Hypnogram parse et
        hypno_data = parse_physionet_hypnogram(hypno_path)
        
        # 3. MongoDB'ye bağlan
        client = MongoClient(mongo_uri)
        db = client[db_name]
        fs = gridfs.GridFS(db)
        
        # 4. Mevcut veriyi kontrol et (skip existing)
        existing = fs.find_one({
            "metadata.patient_id": patient_id,
            "metadata.data_type": "sleep_stages"
        })
        
        if existing:
            print(f"  ⊘ {patient_id}: Zaten var, atlanıyor")
            client.close()
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
            'source': 'physionet_sleep_telemetry'
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
                'source': 'physionet_sleep_telemetry'
            }
            
            fs.put(signal_bytes,
                   filename=f"{patient_id}_{channel_name}.npy",
                   metadata=channel_metadata)
            
            print(f"  ✓ {channel_name}: {len(signal_data):,} samples")
        
        client.close()
        
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
                                  db_name="physionet_sleep"):
    """
    Tüm PhysioNet hastalarını MongoDB'ye aktar
    
    Args:
        data_dir: Sleep telemetry klasörü
        mongo_uri: MongoDB URI
        db_name: Database adı
    
    Returns:
        dict: İşlem özeti
    """
    print(f"\n{'='*70}")
    print(f"TOPLU PHYSIONET IMPORT")
    print(f"{'='*70}")
    print(f"Kaynak: {data_dir}")
    print(f"Hedef DB: {db_name}")
    print(f"{'='*70}\n")
    
    # Hasta dosyalarını bul
    patient_pairs = find_patient_files(data_dir)
    
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
        
        result = import_physionet_patient(psg_path, hypno_path, mongo_uri, db_name)
        
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
    # PhysioNet Sleep Telemetry klasörü
    data_dir = "/media/mehmet-ercan/1e3f69e1-d33e-4f85-814c-6075e302e896/Downloads/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
    
    # MongoDB ayarları
    mongo_uri = "mongodb://localhost:27017/"
    db_name = "physionet_sleep"
    
    # Tüm hastaları import et
    stats = import_all_physionet_patients(data_dir, mongo_uri, db_name)
    
    print("\n✓ Import işlemi tamamlandı!")
    print(f"  Başarılı: {stats['success']}/{stats['total']}")
