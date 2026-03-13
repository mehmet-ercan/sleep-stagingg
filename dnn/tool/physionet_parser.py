#!/usr/bin/env python
# coding: utf-8

"""
PhysioNet Sleep-EDF Database Parser
Parses Sleep Telemetry PSG and Hypnogram files
"""

import pyedflib
import numpy as np
import os
from typing import Dict, List, Tuple, Optional


# Sleep stage mapping: PhysioNet → Standard format
PHYSIONET_STAGE_MAP = {
    'Sleep stage W': 'W',       # Wake
    'Sleep stage R': 'REM',     # REM
    'Sleep stage 1': 'N1',      # N1
    'Sleep stage 2': 'N2',      # N2
    'Sleep stage 3': 'N3',      # N3
    'Sleep stage 4': 'N3',      # N4 → N3 (birleştir)
    'Sleep stage ?': None,      # Not scored → Skip
    'Movement time': None       # Movement → Skip
}


def parse_physionet_psg(psg_file_path: str) -> Dict:
    """
    PhysioNet PSG.edf dosyasını parse et
    
    Args:
        psg_file_path: *PSG.edf dosya yolu
    
    Returns:
        {
            'patient_id': str,
            'channels': {
                'EEG Fpz-Cz': np.array,
                'EEG Pz-Oz': np.array,
                'EOG horizontal': np.array,
                'EMG submental': np.array
            },
            'sample_rate': 100,
            'duration': float (seconds),
            'n_samples': int
        }
    """
    print(f"\n{'='*70}")
    print(f"PSG DOSYASI PARSE EDİLİYOR")
    print(f"{'='*70}")
    print(f"Dosya: {os.path.basename(psg_file_path)}")
    
    # Patient ID'yi dosya adından çıkar (ST7011J0-PSG.edf → ST7011J0)
    patient_id = os.path.basename(psg_file_path).replace('-PSG.edf', '')
    
    # EDF dosyasını aç
    f = pyedflib.EdfReader(psg_file_path)
    
    try:
        n_signals = f.signals_in_file
        duration = f.file_duration
        
        print(f"Patient ID: {patient_id}")
        print(f"Sinyal sayısı: {n_signals}")
        print(f"Süre: {duration / 3600:.2f} saat")
        
        # Kanalları oku
        channels = {}
        sample_rate = None
        
        for i in range(n_signals):
            label = f.getLabel(i)
            freq = f.getSampleFrequency(i)
            
            # Sadece ihtiyacımız olan kanalları al
            if 'EEG' in label or 'EOG' in label or 'EMG' in label:
                signal = f.readSignal(i)
                channels[label] = signal
                
                if sample_rate is None:
                    sample_rate = int(freq)
                
                print(f"  ✓ {label}: {len(signal):,} samples @ {freq} Hz")
        
        result = {
            'patient_id': patient_id,
            'channels': channels,
            'sample_rate': sample_rate,
            'duration': duration,
            'n_samples': len(next(iter(channels.values())))
        }
        
        print(f"\n✓ PSG parse tamamlandı")
        print(f"  Hasta ID: {patient_id}")
        print(f"  Kanal sayısı: {len(channels)}")
        print(f"  Sampling rate: {sample_rate} Hz")
        print(f"{'='*70}\n")
        
        return result
        
    finally:
        f.close()


def parse_physionet_hypnogram(hypno_file_path: str, epoch_duration: float = 30.0) -> List[Dict]:
    """
    PhysioNet Hypnogram.edf dosyasını parse et
    
    PhysioNet formatında annotation'lar interval bazlıdır (örn: 5:00-10:00 arası N1).
    Bu fonksiyon interval'ları 30 saniyelik epoch'lara genişletir.
    
    Args:
        hypno_file_path: *Hypnogram.edf dosya yolu
        epoch_duration: Epoch süresi (varsayılan 30 saniye)
    
    Returns:
        [
            {'epoch': 0, 'stage': 'W', 'start_time': 0.0, 'duration': 30.0},
            {'epoch': 1, 'stage': 'W', 'start_time': 30.0, 'duration': 30.0},
            {'epoch': 2, 'stage': 'N1', 'start_time': 60.0, 'duration': 30.0},
            ...
        ]
    """
    print(f"\n{'='*70}")
    print(f"HYPNOGRAM DOSYASI PARSE EDİLİYOR")
    print(f"{'='*70}")
    print(f"Dosya: {os.path.basename(hypno_file_path)}")
    
    # EDF dosyasını aç
    f = pyedflib.EdfReader(hypno_file_path)
    
    try:
        # Annotations oku (sleep stages burada)
        annotations = f.readAnnotations()
        onsets = annotations[0]      # Başlangıç zamanları
        durations = annotations[1]   # Süreler (interval süreleri)
        descriptions = annotations[2] # Stage açıklamaları
        
        print(f"Annotation sayısı (interval): {len(onsets)}")
        
        # Sleep stages'leri parse et - HER 30 SANİYELİK EPOCH İÇİN AYRI KAYIT
        sleep_stages = []
        epoch_idx = 0
        
        for i in range(len(onsets)):
            onset = onsets[i]
            annotation_duration = durations[i]
            description = descriptions[i]
            
            # Sleep stage mi kontrol et
            if description in PHYSIONET_STAGE_MAP:
                mapped_stage = PHYSIONET_STAGE_MAP[description]
                
                # None ise (Movement veya Not scored) atla
                if mapped_stage is None:
                    continue
                
                # Bu annotation kaç epoch kapsıyor?
                n_epochs_in_interval = int(annotation_duration / epoch_duration)
                
                # Her 30 saniyelik epoch için ayrı kayıt oluştur
                for j in range(n_epochs_in_interval):
                    epoch_start_time = onset + (j * epoch_duration)
                    
                    sleep_stages.append({
                        'epoch': epoch_idx,
                        'stage': mapped_stage,
                        'start_time': epoch_start_time,
                        'duration': epoch_duration
                    })
                    
                    epoch_idx += 1
        
        print(f"\n✓ Hypnogram parse tamamlandı")
        print(f"  Interval sayısı: {len(onsets)}")
        print(f"  Toplam epoch (30sn): {len(sleep_stages)}")
        
        # Stage dağılımını göster
        stage_counts = {}
        for stage_data in sleep_stages:
            stage = stage_data['stage']
            stage_counts[stage] = stage_counts.get(stage, 0) + 1
        
        print(f"\n  Stage dağılımı:")
        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            count = stage_counts.get(stage, 0)
            percentage = (count / len(sleep_stages)) * 100 if sleep_stages else 0
            print(f"    {stage}: {count} epoch ({percentage:.1f}%)")
        
        print(f"{'='*70}\n")
        
        return sleep_stages
        
    finally:
        f.close()


def find_patient_files(data_dir: str) -> List[Tuple[str, str]]:
    """
    Veri klasöründeki PSG ve Hypnogram dosya çiftlerini bul
    
    Args:
        data_dir: Sleep telemetry klasörü
    
    Returns:
        [(psg_path, hypno_path), ...]
    """
    print(f"\n{'='*70}")
    print(f"HASTA DOSYALARI TARANIYOR")
    print(f"{'='*70}")
    print(f"Klasör: {data_dir}\n")
    
    # Tüm PSG dosyalarını bul
    psg_files = sorted([f for f in os.listdir(data_dir) if f.endswith('-PSG.edf')])
    
    patient_pairs = []
    
    for psg_file in psg_files:
        # Patient ID'yi çıkar (ST7011J0-PSG.edf → ST7011)
        # Son 2 karakteri (J0) çıkar, sadece sayı kısmını al
        patient_base = psg_file.split('J')[0]  # ST7011J0-PSG.edf → ST7011
        
        # Karşılık gelen Hypnogram dosyasını bul
        # Hypnogram dosyaları ST7011J* formatında (JP, JM, JO, JA, JR, JE, etc.)
        hypno_files = [f for f in os.listdir(data_dir) 
                      if f.startswith(patient_base) and 'Hypnogram.edf' in f]
        
        if hypno_files:
            psg_path = os.path.join(data_dir, psg_file)
            hypno_path = os.path.join(data_dir, hypno_files[0])
            patient_pairs.append((psg_path, hypno_path))
            print(f"  ✓ {patient_base}: {psg_file} + {hypno_files[0]}")
        else:
            print(f"  ✗ {patient_base}: Hypnogram bulunamadı!")
    
    print(f"\n{'='*70}")
    print(f"✓ {len(patient_pairs)} hasta dosya çifti bulundu")
    print(f"{'='*70}\n")
    
    return patient_pairs


# Test fonksiyonu
if __name__ == "__main__":
    # Test için örnek dosyalar
    data_dir = "/media/mehmet-ercan/1e3f69e1-d33e-4f85-814c-6075e302e896/Downloads/sleep-edf-database-expanded-1.0.0/sleep-telemetry"
    
    # Hasta dosyalarını bul
    patient_pairs = find_patient_files(data_dir)
    
    if patient_pairs:
        # İlk hastayı test et
        psg_path, hypno_path = patient_pairs[0]
        
        print("\n" + "="*70)
        print("TEST: İlk hasta parse ediliyor")
        print("="*70)
        
        # PSG parse et
        psg_data = parse_physionet_psg(psg_path)
        
        # Hypnogram parse et
        hypno_data = parse_physionet_hypnogram(hypno_path)
        
        print("\n" + "="*70)
        print("TEST BAŞARILI!")
        print("="*70)
        print(f"Patient ID: {psg_data['patient_id']}")
        print(f"Channels: {list(psg_data['channels'].keys())}")
        print(f"Sample rate: {psg_data['sample_rate']} Hz")
        print(f"Total epochs: {len(hypno_data)}")
        print("="*70 + "\n")
