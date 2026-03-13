#!/usr/bin/env python3
"""
PSG Data Filtered Yeniden Yükleme Script

Bu script:
1. psg_data_filtered veritabanını DROP eder
2. Tüm hasta verilerini psg_data'dan okur
3. Differential channel + Filter + Clipping uygulayarak yeniden kaydeder

Kullanım:
    python reload_psg_data_filtered.py
"""

import os
import sys
from pymongo import MongoClient
import gridfs
from datetime import datetime
from tqdm import tqdm

# Aynı klasördeki modülleri import et
from edf_to_mongo import (
    DIFFERENTIAL_CHANNELS, 
    apply_all_filters, 
    apply_amplitude_clipping,
    create_differential_channel,
    USE_AMPLITUDE_CLIPPING,
    CLIP_PERCENTILE_LOW,
    CLIP_PERCENTILE_HIGH
)
import numpy as np
import io
import json

# MongoDB ayarları
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME_RAW = "psg_data"
DB_NAME_FILTERED = "psg_data_filtered"


def get_all_patients(mongo_uri, db_name):
    """Tüm hasta ID'lerini getir"""
    client = MongoClient(mongo_uri)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    patients = set()
    for f in fs.find():
        if f.metadata and 'patient_id' in f.metadata:
            patients.add(f.metadata['patient_id'])
    
    client.close()
    return list(patients)


def get_raw_channels(fs, patient_id):
    """Bir hastanın tüm raw kanallarını getir"""
    channels = {}
    
    for f in fs.find({'metadata.patient_id': patient_id}):
        if f.metadata and f.metadata.get('data_type') == 'raw_signal':
            ch_name = f.metadata.get('channel_name')
            if ch_name:
                data = f.read()
                if data[:6] == b'\x93NUMPY':
                    signal = np.load(io.BytesIO(data))
                else:
                    signal = np.frombuffer(data, dtype=np.float64)
                
                channels[ch_name] = {
                    'signal': signal,
                    'sample_freq': f.metadata.get('sample_frequency', 256),
                    'metadata': f.metadata
                }
    
    return channels


def get_sleep_stages(fs, patient_id):
    """Bir hastanın sleep stages verisini getir"""
    for f in fs.find({'metadata.patient_id': patient_id}):
        if f.metadata and f.metadata.get('data_type') == 'sleep_stages':
            data = f.read()
            stages = json.loads(data.decode('utf-8'))
            return stages, f.metadata
    return None, None


def process_patient(fs_raw, fs_filtered, patient_id, verbose=False):
    """Bir hastayı işle ve filtered DB'ye kaydet"""
    
    # Raw kanalları al
    raw_channels = get_raw_channels(fs_raw, patient_id)
    if not raw_channels:
        return False, "No raw channels found"
    
    # Sleep stages al
    stages, stages_metadata = get_sleep_stages(fs_raw, patient_id)
    if stages is None:
        return False, "No sleep stages found"
    
    # Sleep stages'i filtered DB'ye kaydet
    stages_json = json.dumps(stages, indent=2)
    stages_buffer = io.BytesIO()
    stages_buffer.write(stages_json.encode('utf-8'))
    stages_buffer.seek(0)
    
    stages_metadata_new = dict(stages_metadata) if stages_metadata else {}
    stages_metadata_new['source_db'] = DB_NAME_RAW
    stages_metadata_new['reprocessed'] = True
    stages_metadata_new['reprocess_date'] = datetime.now()
    
    stages_id = fs_filtered.put(
        stages_buffer,
        filename=f"{patient_id}_sleep_stages.json",
        metadata=stages_metadata_new
    )
    
    # Differential kanalları oluştur ve kaydet
    diff_count = 0
    for diff_name, (active_ch, ref_ch) in DIFFERENTIAL_CHANNELS.items():
        if active_ch in raw_channels and ref_ch in raw_channels:
            signal_active = raw_channels[active_ch]['signal']
            signal_ref = raw_channels[ref_ch]['signal']
            fs_hz = raw_channels[active_ch]['sample_freq']
            
            # Referans kanalı farklı sample rate'e sahipse atla
            if raw_channels[ref_ch]['sample_freq'] != fs_hz:
                continue
            
            # Differential sinyal oluştur
            diff_signal = create_differential_channel(signal_active, signal_ref)
            
            # Filtreleme uygula
            filtered_signal, filter_params = apply_all_filters(diff_signal, fs=fs_hz, channel_name=diff_name)
            
            # Amplitude clipping uygula
            clip_params = None
            if USE_AMPLITUDE_CLIPPING:
                filtered_signal, clip_params = apply_amplitude_clipping(filtered_signal)
            
            # Metadata hazırla
            orig_metadata = raw_channels[active_ch].get('metadata', {})
            
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
                "source_db": DB_NAME_RAW,
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
                "reprocessed": True,
                "reprocess_date": datetime.now(),
                "stages_id": stages_id,
                "has_sleep_stages": True
            }
            
            # Trim bilgisi varsa ekle
            if 'trim_start_time' in orig_metadata:
                diff_metadata['trim_start_time'] = orig_metadata['trim_start_time']
            if 'trim_end_time' in orig_metadata:
                diff_metadata['trim_end_time'] = orig_metadata['trim_end_time']
            
            # GridFS'e kaydet
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
            if verbose:
                clipped_pct = clip_params.get('clipped_samples_pct', 0) if clip_params else 0
                print(f"    ✓ {diff_name} → clipped {clipped_pct:.2f}%")
    
    return True, f"{diff_count} channels"


def main():
    print("=" * 70)
    print("PSG_DATA_FILTERED YENİDEN YÜKLEME")
    print("=" * 70)
    print(f"\nClipping ayarları:")
    print(f"  USE_AMPLITUDE_CLIPPING: {USE_AMPLITUDE_CLIPPING}")
    print(f"  CLIP_PERCENTILE_LOW:    {CLIP_PERCENTILE_LOW}")
    print(f"  CLIP_PERCENTILE_HIGH:   {CLIP_PERCENTILE_HIGH}")
    
    # Onay al
    print(f"\n⚠️  UYARI: Bu işlem {DB_NAME_FILTERED} veritabanını SİLECEK!")
    confirm = input("Devam etmek istiyor musunuz? (evet/hayır): ")
    
    if confirm.lower() not in ['evet', 'e', 'yes', 'y']:
        print("İşlem iptal edildi.")
        return
    
    # MongoDB bağlantısı
    client = MongoClient(MONGO_URI)
    
    # Mevcut veritabanını drop et
    print(f"\n[1/3] {DB_NAME_FILTERED} veritabanı siliniyor...")
    client.drop_database(DB_NAME_FILTERED)
    print(f"✓ {DB_NAME_FILTERED} silindi")
    
    # GridFS bağlantıları
    db_raw = client[DB_NAME_RAW]
    db_filtered = client[DB_NAME_FILTERED]
    fs_raw = gridfs.GridFS(db_raw)
    fs_filtered = gridfs.GridFS(db_filtered)
    
    # Hastaları al
    print(f"\n[2/3] Hasta listesi alınıyor...")
    patients = get_all_patients(MONGO_URI, DB_NAME_RAW)
    print(f"✓ {len(patients)} hasta bulundu")
    
    # Her hastayı işle
    print(f"\n[3/3] Veriler işleniyor (Filter + Clipping)...")
    print("-" * 70)
    
    success_count = 0
    error_count = 0
    
    for patient_id in tqdm(patients, desc="İşleniyor"):
        success, msg = process_patient(fs_raw, fs_filtered, patient_id, verbose=False)
        if success:
            success_count += 1
        else:
            error_count += 1
            print(f"\n  ✗ {patient_id[:8]}...: {msg}")
    
    client.close()
    
    # Sonuç
    print("\n" + "=" * 70)
    print("✅ İŞLEM TAMAMLANDI!")
    print("=" * 70)
    print(f"Başarılı:  {success_count} hasta")
    print(f"Hatalı:    {error_count} hasta")
    print(f"\nVeritabanı: {DB_NAME_FILTERED}")
    print(f"Preprocessing: Differential + Filter + Clipping (p{CLIP_PERCENTILE_LOW}-p{CLIP_PERCENTILE_HIGH})")


if __name__ == "__main__":
    main()
