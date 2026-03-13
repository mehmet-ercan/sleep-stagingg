#!/usr/bin/env python
# coding: utf-8
"""
Data Quality Analyzer
PhysioNet vs PSG Data karşılaştırması

Kullanım:
    python -m tool.data_quality_analyzer --max-patients 10
    python -m tool.data_quality_analyzer --full-analysis
"""

import numpy as np
from pymongo import MongoClient
import gridfs
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew
from collections import defaultdict
import json
from tqdm import tqdm
import sys
import os

# Parent path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ============================================================================
# CONFIGURATION
# ============================================================================

PHYSIONET_DB = "physionet_sleep"
PSG_DB = "psg_data_filtered"
MONGO_URI = config.MONGO_URI

# PhysioNet kanalları (100 Hz)
CHANNELS_PHYSIONET = ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental']

# PSG kanalları (256 Hz) - differential
CHANNELS_PSG = ['C3-M2', 'F3-M2', 'E1-M2', 'CHIN1-CHIN2']

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_db_info(db_name):
    """Veritabanı hakkında genel bilgi al."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    info = {
        'total_files': fs.find().count() if hasattr(fs.find(), 'count') else len(list(fs.find())),
        'collections': db.list_collection_names()
    }
    
    client.close()
    return info


def get_all_patients(db_name):
    """Veritabanındaki tüm hastaları getir."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    patients = set()
    
    # sleep_stages data_type ile dene
    for file in fs.find({"metadata.data_type": "sleep_stages"}):
        if 'patient_id' in file.metadata:
            patients.add(file.metadata['patient_id'])
    
    # Eğer boşsa, tüm dosyaları tara
    if not patients:
        for file in fs.find():
            if file.metadata and 'patient_id' in file.metadata:
                patients.add(file.metadata['patient_id'])
    
    client.close()
    return sorted(patients)


def get_patient_stages(patient_id, db_name):
    """Hastanın tüm sleep stage'lerini getir."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Farklı query'ler dene
    queries = [
        {"metadata.patient_id": patient_id, "metadata.data_type": "sleep_stages"},
        {"metadata.patient_id": patient_id, "metadata.channel_name": "sleep_stages"},
        {"metadata.patient_id": patient_id}
    ]
    
    stages_file = None
    for query in queries:
        stages_file = fs.find_one(query)
        if stages_file and 'sleep_stages' in str(stages_file.metadata):
            break
        # JSON dosyası olabilir
        if stages_file:
            try:
                content = stages_file.read()
                stages_file.seek(0)
                data = json.loads(content.decode('utf-8'))
                if isinstance(data, list) and len(data) > 0:
                    if isinstance(data[0], dict) and 'stage' in data[0]:
                        client.close()
                        return data
            except:
                pass
    
    # Sleep stages için özel arama
    for file in fs.find({"metadata.patient_id": patient_id}):
        if file.metadata.get('data_type') == 'sleep_stages' or \
           file.metadata.get('channel_name') == 'sleep_stages':
            try:
                content = file.read()
                data = json.loads(content.decode('utf-8'))
                client.close()
                return data
            except:
                pass
    
    client.close()
    return None


def get_patient_signal(patient_id, channel_name, db_name):
    """Hastanın belirli bir kanalının sinyalini getir."""
    import io
    
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    # Farklı query'ler dene
    queries = [
        {"metadata.patient_id": patient_id, "metadata.channel_name": channel_name, "metadata.data_type": "raw_signal"},
        {"metadata.patient_id": patient_id, "metadata.channel_name": channel_name},
    ]
    
    signal_file = None
    for query in queries:
        signal_file = fs.find_one(query)
        if signal_file:
            break
    
    if signal_file is None:
        client.close()
        return None, None
    
    try:
        raw_data = signal_file.read()
        
        # NPY formatı mı kontrol et (np.save ile kaydedilmiş)
        if raw_data[:6] == b'\x93NUMPY':
            # np.load ile oku
            buffer = io.BytesIO(raw_data)
            signal_data = np.load(buffer)
        else:
            # Raw binary olarak oku
            dtype = signal_file.metadata.get('dtype', 'float64')
            if 'float64' in str(dtype):
                np_dtype = np.float64
            elif 'float32' in str(dtype):
                np_dtype = np.float32
            else:
                np_dtype = np.float64
            signal_data = np.frombuffer(raw_data, dtype=np_dtype)
    except Exception as e:
        client.close()
        return None, None
    
    sample_freq = signal_file.metadata.get('sample_frequency', 
                   signal_file.metadata.get('sample_rate', 256))
    
    client.close()
    return signal_data, sample_freq


def get_available_channels(patient_id, db_name):
    """Hastada mevcut kanalları listele."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)
    
    channels = []
    for file in fs.find({"metadata.patient_id": patient_id}):
        ch = file.metadata.get('channel_name', None)
        if ch and ch not in ['sleep_stages']:
            channels.append(ch)
    
    client.close()
    return list(set(channels))


# ============================================================================
# SIGNAL QUALITY METRICS
# ============================================================================

def calculate_snr(signal, fs=256):
    """Basit SNR tahmini (sinyal gücü / gürültü gücü)."""
    if len(signal) < 1024:
        return 0.0
    
    try:
        # Band-pass filter ile "temiz" sinyal
        nyq = fs / 2
        low = 0.5 / nyq
        high = min(35 / nyq, 0.99)
        
        if low >= high:
            return 0.0
        
        b, a = scipy_signal.butter(4, [low, high], btype='band')
        filtered = scipy_signal.filtfilt(b, a, signal)
        
        noise = signal - filtered
        signal_power = np.mean(filtered ** 2)
        noise_power = np.mean(noise ** 2)
        
        if noise_power == 0 or signal_power == 0:
            return 0.0
        
        return 10 * np.log10(signal_power / noise_power)
    except Exception as e:
        return 0.0


def detect_flatlines(signal, threshold=1e-6, min_duration=256):
    """Düz çizgi (flatline) segmentlerini tespit et."""
    if len(signal) < 2:
        return 0
    
    diff = np.abs(np.diff(signal))
    flatline = diff < threshold
    
    flatline_count = 0
    current_run = 0
    
    for is_flat in flatline:
        if is_flat:
            current_run += 1
        else:
            if current_run >= min_duration:
                flatline_count += 1
            current_run = 0
    
    return flatline_count


def detect_clipping(signal, percentile=99.9):
    """Clipping (sinyal saturasyonu) tespit et."""
    if len(signal) == 0:
        return 0.0
    
    max_val = np.percentile(np.abs(signal), percentile)
    if max_val == 0:
        return 0.0
    
    clipped = np.abs(signal) >= max_val * 0.99
    return np.sum(clipped) / len(signal) * 100


def calculate_psd_features(signal, fs=256):
    """Power Spectral Density özelliklerini hesapla."""
    if len(signal) < 256:
        return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'total': 0}
    
    try:
        freqs, psd = scipy_signal.welch(signal, fs=fs, nperseg=min(len(signal), 1024))
        
        # Band güçleri
        delta_mask = (freqs >= 0.5) & (freqs < 4)
        theta_mask = (freqs >= 4) & (freqs < 8)
        alpha_mask = (freqs >= 8) & (freqs < 13)
        beta_mask = (freqs >= 13) & (freqs < 30)
        
        total_power = np.sum(psd)
        if total_power == 0:
            return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'total': 0}
        
        return {
            'delta': float(np.sum(psd[delta_mask]) / total_power),
            'theta': float(np.sum(psd[theta_mask]) / total_power),
            'alpha': float(np.sum(psd[alpha_mask]) / total_power),
            'beta': float(np.sum(psd[beta_mask]) / total_power),
            'total': float(total_power)
        }
    except:
        return {'delta': 0, 'theta': 0, 'alpha': 0, 'beta': 0, 'total': 0}


def detect_artifacts(signal, fs=256, epoch_length=30):
    """Epoch bazında artifact tespiti."""
    samples_per_epoch = int(fs * epoch_length)
    n_epochs = len(signal) // samples_per_epoch
    
    artifact_epochs = 0
    signal_std = np.std(signal)
    signal_p99 = np.percentile(np.abs(signal), 99.9)
    
    for i in range(n_epochs):
        start = i * samples_per_epoch
        end = start + samples_per_epoch
        epoch = signal[start:end]
        
        # Artifact kriterleri
        epoch_std = np.std(epoch)
        if signal_std > 0 and epoch_std > signal_std * 5:  # Çok yüksek varyans
            artifact_epochs += 1
        elif signal_p99 > 0 and np.max(np.abs(epoch)) > signal_p99 * 2:  # Spike
            artifact_epochs += 1
        elif epoch_std < 1e-6:  # Flatline
            artifact_epochs += 1
    
    return artifact_epochs, n_epochs


def analyze_signal_quality(signal, fs=256):
    """Tek bir sinyal için tüm kalite metriklerini hesapla."""
    if signal is None or len(signal) == 0:
        return None
    
    # NaN/Inf kontrolü
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    
    try:
        artifact_count, total_epochs = detect_artifacts(signal, fs)
        
        return {
            'length': int(len(signal)),
            'duration_hours': float(len(signal) / fs / 3600),
            'mean': float(np.mean(signal)),
            'std': float(np.std(signal)),
            'min': float(np.min(signal)),
            'max': float(np.max(signal)),
            'range': float(np.max(signal) - np.min(signal)),
            'kurtosis': float(kurtosis(signal)) if len(signal) > 4 else 0.0,
            'skewness': float(skew(signal)) if len(signal) > 4 else 0.0,
            'snr_db': float(calculate_snr(signal, fs)),
            'flatline_count': int(detect_flatlines(signal, min_duration=fs)),
            'clipping_percent': float(detect_clipping(signal)),
            'zero_crossing_rate': float(np.sum(np.diff(np.sign(signal)) != 0) / len(signal)) if len(signal) > 1 else 0.0,
            'artifact_epochs': artifact_count,
            'artifact_ratio': float(artifact_count / total_epochs) if total_epochs > 0 else 0.0,
            'psd': calculate_psd_features(signal, fs)
        }
    except Exception as e:
        print(f"    Warning: Signal analysis error: {e}")
        return None


# ============================================================================
# STAGE QUALITY METRICS
# ============================================================================

def analyze_stage_quality(stages):
    """Stage annotation kalitesini analiz et."""
    if stages is None or len(stages) == 0:
        return None
    
    # Stage listesini çıkar
    if isinstance(stages[0], dict):
        stage_list = [s.get('stage', s.get('label', 'Unknown')) for s in stages]
    else:
        stage_list = list(stages)
    
    # Stage normalizasyonu
    stage_map = {
        'Wake': 'W', 'W': 'W', 'WAKE': 'W', '0': 'W',
        'Stage 1': 'N1', 'N1': 'N1', 'NREM1': 'N1', '1': 'N1',
        'Stage 2': 'N2', 'N2': 'N2', 'NREM2': 'N2', '2': 'N2',
        'Stage 3': 'N3', 'N3': 'N3', 'NREM3': 'N3', '3': 'N3',
        'Stage 4': 'N3', 'N4': 'N3', 'NREM4': 'N3', '4': 'N3',
        'REM': 'REM', 'R': 'REM', '5': 'REM',
        'Movement': 'W', 'Unknown': 'W', '?': 'W'
    }
    
    stage_list = [stage_map.get(str(s), 'W') for s in stage_list]
    
    if len(stage_list) == 0:
        return None
    
    # Dağılım
    stage_counts = defaultdict(int)
    for s in stage_list:
        stage_counts[s] += 1
    
    # Geçiş matrisi
    transitions = defaultdict(lambda: defaultdict(int))
    for i in range(len(stage_list) - 1):
        from_stage = stage_list[i]
        to_stage = stage_list[i + 1]
        transitions[from_stage][to_stage] += 1
    
    # Şüpheli geçişler (fizyolojik olmayan)
    suspicious_transitions = []
    
    # W → N3 (atlamalı - çok nadir)
    if transitions['W']['N3'] > 0:
        suspicious_transitions.append(('W→N3', transitions['W']['N3']))
    
    # REM → N3 (çok nadir)
    if transitions['REM']['N3'] > 0:
        suspicious_transitions.append(('REM→N3', transitions['REM']['N3']))
    
    # N3 → W (ani uyanma - olabilir ama dikkat)
    if transitions['N3']['W'] > 2:
        suspicious_transitions.append(('N3→W', transitions['N3']['W']))
    
    # N1 → N3 (N2 atlaması)
    if transitions['N1']['N3'] > 0:
        suspicious_transitions.append(('N1→N3', transitions['N1']['N3']))
    
    # Çok kısa evreler (1-2 epoch = 30-60s)
    short_runs = 0
    very_short_runs = 0  # Sadece 1 epoch
    current_stage = None
    current_run = 0
    
    for s in stage_list:
        if s == current_stage:
            current_run += 1
        else:
            if current_run == 1:
                very_short_runs += 1
            if current_run > 0 and current_run <= 2:
                short_runs += 1
            current_stage = s
            current_run = 1
    
    total_epochs = len(stage_list)
    
    # Stage fragment sayısı (toplam geçiş sayısı)
    total_transitions = sum(transitions[from_s][to_s] 
                           for from_s in transitions 
                           for to_s in transitions[from_s] 
                           if from_s != to_s)
    
    return {
        'total_epochs': total_epochs,
        'duration_hours': total_epochs * 30 / 3600,
        'stage_distribution': dict(stage_counts),
        'w_ratio': stage_counts.get('W', 0) / total_epochs if total_epochs > 0 else 0,
        'n1_ratio': stage_counts.get('N1', 0) / total_epochs if total_epochs > 0 else 0,
        'n2_ratio': stage_counts.get('N2', 0) / total_epochs if total_epochs > 0 else 0,
        'n3_ratio': stage_counts.get('N3', 0) / total_epochs if total_epochs > 0 else 0,
        'rem_ratio': stage_counts.get('REM', 0) / total_epochs if total_epochs > 0 else 0,
        'suspicious_transitions': suspicious_transitions,
        'suspicious_count': len(suspicious_transitions),
        'short_run_count': short_runs,
        'very_short_run_count': very_short_runs,
        'total_transitions': total_transitions,
        'fragmentation_index': total_transitions / total_epochs if total_epochs > 0 else 0,
        'transition_matrix': {k: dict(v) for k, v in transitions.items()}
    }


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def analyze_dataset(db_name, channels, max_patients=None):
    """Bir dataset'i tamamen analiz et."""
    print(f"\n{'='*70}")
    print(f"ANALYZING: {db_name}")
    print(f"{'='*70}\n")
    
    # DB bilgisi
    try:
        db_info = get_db_info(db_name)
        print(f"Database collections: {db_info['collections']}")
    except Exception as e:
        print(f"Warning: Could not get DB info: {e}")
    
    patients = get_all_patients(db_name)
    if max_patients:
        patients = patients[:max_patients]
    
    print(f"Total patients found: {len(patients)}")
    
    if len(patients) == 0:
        print("No patients found!")
        return {'db_name': db_name, 'patient_count': 0, 'patients': {}}
    
    results = {
        'db_name': db_name,
        'patient_count': len(patients),
        'patients': {}
    }
    
    for patient_id in tqdm(patients, desc="Analyzing patients"):
        patient_result = {
            'stages': None,
            'channels': {},
            'available_channels': []
        }
        
        # Mevcut kanalları listele
        available = get_available_channels(patient_id, db_name)
        patient_result['available_channels'] = available
        
        # Stage analizi
        stages = get_patient_stages(patient_id, db_name)
        patient_result['stages'] = analyze_stage_quality(stages)
        
        # Kanal analizleri - verilen kanalları dene, yoksa mevcut olanları kullan
        channels_to_try = channels if channels else available
        
        for channel in channels_to_try:
            signal, fs = get_patient_signal(patient_id, channel, db_name)
            if signal is not None and len(signal) > 0:
                patient_result['channels'][channel] = analyze_signal_quality(signal, fs)
        
        # Eğer belirtilen kanallar bulunamazsa, mevcut kanalları dene
        if not patient_result['channels'] and available:
            for channel in available[:4]:  # İlk 4 kanal
                signal, fs = get_patient_signal(patient_id, channel, db_name)
                if signal is not None and len(signal) > 0:
                    patient_result['channels'][channel] = analyze_signal_quality(signal, fs)
        
        results['patients'][patient_id] = patient_result
    
    return results


def aggregate_metrics(results):
    """Tüm hastaların metriklerini aggregate et."""
    aggregated = defaultdict(list)
    
    for patient_id, patient_data in results['patients'].items():
        # Stage metrics
        if patient_data['stages']:
            for key in ['n1_ratio', 'w_ratio', 'n2_ratio', 'n3_ratio', 'rem_ratio', 
                       'total_epochs', 'short_run_count', 'very_short_run_count',
                       'fragmentation_index', 'suspicious_count']:
                if key in patient_data['stages']:
                    aggregated[key].append(patient_data['stages'][key])
        
        # Signal metrics (tüm kanallardan ortalama)
        channel_metrics = defaultdict(list)
        for channel, channel_data in patient_data['channels'].items():
            if channel_data:
                for key, value in channel_data.items():
                    if isinstance(value, (int, float)) and key not in ['length', 'psd']:
                        channel_metrics[key].append(value)
        
        # Hasta ortalaması
        for key, values in channel_metrics.items():
            if values:
                aggregated[f'signal_{key}'].append(np.mean(values))
    
    # Calculate mean, std for each metric
    final = {}
    for key, values in aggregated.items():
        if len(values) > 0:
            arr = np.array(values)
            arr = arr[np.isfinite(arr)]  # NaN/Inf temizle
            if len(arr) > 0:
                final[key] = {
                    'mean': float(np.mean(arr)),
                    'std': float(np.std(arr)),
                    'min': float(np.min(arr)),
                    'max': float(np.max(arr)),
                    'median': float(np.median(arr)),
                    'count': len(arr)
                }
    
    return final


def compare_datasets(physionet_results, psg_results):
    """İki dataset'i karşılaştır ve farkları raporla."""
    print(f"\n{'='*70}")
    print("📊 DATASET COMPARISON REPORT")
    print(f"{'='*70}\n")
    
    # Aggregate metrics
    physionet_metrics = aggregate_metrics(physionet_results) if physionet_results else {}
    psg_metrics = aggregate_metrics(psg_results) if psg_results else {}
    
    # Genel bilgiler
    print("📋 GENERAL INFO")
    print("-" * 50)
    p_count = physionet_results['patient_count'] if physionet_results else 0
    c_count = psg_results['patient_count'] if psg_results else 0
    print(f"  PhysioNet patients: {p_count}")
    print(f"  PSG patients: {c_count}")
    
    if 'total_epochs' in physionet_metrics and 'total_epochs' in psg_metrics:
        print(f"  PhysioNet avg epochs/patient: {physionet_metrics['total_epochs']['mean']:.0f}")
        print(f"  PSG avg epochs/patient: {psg_metrics['total_epochs']['mean']:.0f}")
    
    # Stage dağılımı karşılaştırması
    print(f"\n📈 STAGE DISTRIBUTION (mean %)")
    print("-" * 50)
    stage_metrics = ['w_ratio', 'n1_ratio', 'n2_ratio', 'n3_ratio', 'rem_ratio']
    
    print(f"  {'Stage':<6} {'PhysioNet':>12} {'PSG':>12} {'Diff':>10}")
    print(f"  {'-'*6} {'-'*12} {'-'*12} {'-'*10}")
    
    for metric in stage_metrics:
        p_val = physionet_metrics.get(metric, {}).get('mean', 0)
        c_val = psg_metrics.get(metric, {}).get('mean', 0)
        diff = (c_val - p_val) * 100
        stage_name = metric.replace('_ratio', '').upper()
        print(f"  {stage_name:<6} {p_val*100:>11.1f}% {c_val*100:>11.1f}% {diff:>+9.1f}%")
    
    # Annotation kalitesi
    print(f"\n📝 ANNOTATION QUALITY")
    print("-" * 50)
    
    annotation_metrics = [
        ('fragmentation_index', 'Fragmentation Index'),
        ('short_run_count', 'Short Runs (<3 epochs)'),
        ('very_short_run_count', 'Very Short Runs (1 epoch)'),
        ('suspicious_count', 'Suspicious Transitions'),
    ]
    
    for metric_key, metric_name in annotation_metrics:
        p_val = physionet_metrics.get(metric_key, {}).get('mean', 0)
        c_val = psg_metrics.get(metric_key, {}).get('mean', 0)
        
        status = "⚠️" if c_val > p_val * 1.5 else "✓"
        print(f"  {status} {metric_name:30s}: P={p_val:8.2f}, PSG={c_val:8.2f}")
    
    # Sinyal kalitesi karşılaştırması
    print(f"\n🔊 SIGNAL QUALITY")
    print("-" * 50)
    
    signal_metrics = [
        ('signal_std', 'Amplitude StdDev', False),
        ('signal_snr_db', 'SNR (dB)', True),  # Higher is better
        ('signal_kurtosis', 'Kurtosis (artifact)', False),
        ('signal_clipping_percent', 'Clipping %', False),
        ('signal_flatline_count', 'Flatlines', False),
        ('signal_artifact_ratio', 'Artifact Ratio', False),
    ]
    
    for metric_key, metric_name, higher_better in signal_metrics:
        p_val = physionet_metrics.get(metric_key, {}).get('mean', None)
        c_val = psg_metrics.get(metric_key, {}).get('mean', None)
        
        if p_val is not None and c_val is not None:
            if higher_better:
                status = "⚠️" if c_val < p_val * 0.8 else "✓"
            else:
                status = "⚠️" if c_val > p_val * 1.5 else "✓"
            print(f"  {status} {metric_name:20s}: P={p_val:10.4f}, PSG={c_val:10.4f}")
        elif c_val is not None:
            print(f"  ? {metric_name:20s}: P=N/A, PSG={c_val:10.4f}")
    
    return physionet_metrics, psg_metrics


def identify_problematic_patients(results, threshold_kurtosis=10, threshold_snr=5):
    """Problemli hastaları tespit et."""
    problematic = []
    
    for patient_id, patient_data in results['patients'].items():
        issues = []
        severity = 0
        
        # Stage issues
        if patient_data['stages']:
            stages = patient_data['stages']
            
            # Çok yüksek N1 oranı (>25%)
            if stages.get('n1_ratio', 0) > 0.25:
                issues.append(f"⚠️ High N1 ratio: {stages['n1_ratio']*100:.1f}%")
                severity += 2
            
            # Çok düşük uyku verimi (çok fazla W)
            if stages.get('w_ratio', 0) > 0.5:
                issues.append(f"⚠️ High Wake ratio: {stages['w_ratio']*100:.1f}%")
                severity += 1
            
            # Şüpheli geçişler
            if stages.get('suspicious_count', 0) > 5:
                issues.append(f"⚠️ Many suspicious transitions: {stages['suspicious_count']}")
                severity += 2
            
            # Çok kısa run'lar
            if stages.get('very_short_run_count', 0) > 30:
                issues.append(f"⚠️ Many 1-epoch runs: {stages['very_short_run_count']}")
                severity += 1
            
            # Yüksek fragmantasyon
            if stages.get('fragmentation_index', 0) > 0.3:
                issues.append(f"⚠️ High fragmentation: {stages['fragmentation_index']:.2f}")
                severity += 1
            
            # Çok kısa kayıt
            if stages.get('total_epochs', 0) < 100:
                issues.append(f"⚠️ Very short recording: {stages['total_epochs']} epochs")
                severity += 2
        
        # Signal issues
        for channel, channel_data in patient_data['channels'].items():
            if channel_data:
                # Yüksek kurtosis (artifact)
                if abs(channel_data.get('kurtosis', 0)) > threshold_kurtosis:
                    issues.append(f"🔊 {channel}: High kurtosis ({channel_data['kurtosis']:.1f})")
                    severity += 1
                
                # Düşük SNR
                if channel_data.get('snr_db', 100) < threshold_snr:
                    issues.append(f"🔊 {channel}: Low SNR ({channel_data['snr_db']:.1f} dB)")
                    severity += 2
                
                # Flatline
                if channel_data.get('flatline_count', 0) > 0:
                    issues.append(f"🔊 {channel}: {channel_data['flatline_count']} flatlines")
                    severity += 2
                
                # Clipping
                if channel_data.get('clipping_percent', 0) > 1:
                    issues.append(f"🔊 {channel}: {channel_data['clipping_percent']:.1f}% clipping")
                    severity += 1
                
                # Yüksek artifact oranı
                if channel_data.get('artifact_ratio', 0) > 0.1:
                    issues.append(f"🔊 {channel}: {channel_data['artifact_ratio']*100:.1f}% artifacts")
                    severity += 2
                
                # Çok düşük varyans (ölü kanal)
                if channel_data.get('std', 1) < 1e-6:
                    issues.append(f"🔊 {channel}: Dead channel (zero variance)")
                    severity += 3
        
        if issues:
            problematic.append({
                'patient_id': patient_id,
                'issues': issues,
                'severity': severity
            })
    
    # Severity'ye göre sırala
    problematic.sort(key=lambda x: x['severity'], reverse=True)
    
    return problematic


def print_problematic_patients(problematic, dataset_name="", max_show=15):
    """Problemli hastaları yazdır."""
    print(f"\n{'='*70}")
    print(f"🚨 PROBLEMATIC PATIENTS - {dataset_name} (Top {min(len(problematic), max_show)})")
    print(f"{'='*70}")
    
    if not problematic:
        print("  ✓ No significant issues found!")
        return
    
    for i, p in enumerate(problematic[:max_show]):
        print(f"\n[{i+1}] {p['patient_id']} (severity: {p['severity']})")
        for issue in p['issues']:
            print(f"    {issue}")
    
    # Özet
    severe = len([p for p in problematic if p['severity'] >= 5])
    moderate = len([p for p in problematic if 3 <= p['severity'] < 5])
    mild = len([p for p in problematic if p['severity'] < 3])
    
    print(f"\n  📊 Summary: {len(problematic)} problematic patients")
    print(f"     🔴 Severe (≥5): {severe}")
    print(f"     🟡 Moderate (3-4): {moderate}")
    print(f"     🟢 Mild (<3): {mild}")


def generate_recommendations(physionet_metrics, psg_metrics, psg_problematic):
    """Analiz sonuçlarına göre öneriler üret."""
    print(f"\n{'='*70}")
    print("💡 RECOMMENDATIONS")
    print(f"{'='*70}\n")
    
    recommendations = []
    
    if psg_metrics:
        # SNR kontrolü
        psg_snr = psg_metrics.get('signal_snr_db', {}).get('mean', 100)
        physio_snr = physionet_metrics.get('signal_snr_db', {}).get('mean', 100) if physionet_metrics else 100
        
        if psg_snr < 10:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'Low SNR in PSG data ({psg_snr:.1f} dB vs {physio_snr:.1f} dB)',
                'action': 'Apply stronger bandpass filtering (0.3-35 Hz) and consider artifact rejection'
            })
        
        # Fragmantasyon kontrolü
        psg_frag = psg_metrics.get('fragmentation_index', {}).get('mean', 0)
        if psg_frag > 0.2:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'High stage fragmentation ({psg_frag:.2f})',
                'action': 'Apply temporal smoothing to predictions or review annotations for consistency'
            })
        
        # Şüpheli geçişler
        psg_sus = psg_metrics.get('suspicious_count', {}).get('mean', 0)
        if psg_sus > 3:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f'Many suspicious transitions ({psg_sus:.1f} per patient)',
                'action': 'Review and clean annotations, especially W↔N3 and REM↔N3 transitions'
            })
        
        # N1 oranı
        psg_n1 = psg_metrics.get('n1_ratio', {}).get('mean', 0)
        physio_n1 = physionet_metrics.get('n1_ratio', {}).get('mean', 0) if physionet_metrics else 0
        
        if psg_n1 > physio_n1 * 1.5:
            recommendations.append({
                'priority': 'MEDIUM',
                'issue': f'Higher N1 ratio in PSG ({psg_n1*100:.1f}% vs {physio_n1*100:.1f}%)',
                'action': 'N1 is often confused with W or N2. Consider using class weights or focal loss'
            })
        
        # Artifact oranı
        psg_artifact = psg_metrics.get('signal_artifact_ratio', {}).get('mean', 0)
        if psg_artifact > 0.05:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'High artifact ratio ({psg_artifact*100:.1f}%)',
                'action': 'Implement epoch-level artifact rejection before training'
            })
    
    # Problemli hastalar
    if psg_problematic:
        severe_patients = [p for p in psg_problematic if p['severity'] >= 5]
        if len(severe_patients) > 3:
            recommendations.append({
                'priority': 'HIGH',
                'issue': f'{len(severe_patients)} severely problematic patients',
                'action': f'Consider excluding: {", ".join([p["patient_id"] for p in severe_patients[:5]])}'
            })
    
    # Yazdır
    if recommendations:
        for i, rec in enumerate(recommendations, 1):
            priority_emoji = {'HIGH': '🔴', 'MEDIUM': '🟡', 'LOW': '🟢'}[rec['priority']]
            print(f"{i}. {priority_emoji} [{rec['priority']}] {rec['issue']}")
            print(f"   → {rec['action']}\n")
    else:
        print("  ✓ No critical issues found. Data quality appears acceptable.")
    
    return recommendations


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze data quality of sleep datasets')
    parser.add_argument('--max-patients', type=int, default=None,
                        help='Max patients per dataset (None=all)')
    parser.add_argument('--output', type=str, default='data_quality_report.json',
                        help='Output JSON file')
    parser.add_argument('--physionet-only', action='store_true',
                        help='Only analyze PhysioNet')
    parser.add_argument('--psg-only', action='store_true',
                        help='Only analyze PSG')
    args = parser.parse_args()
    
    physionet_results = None
    psg_results = None
    
    # PhysioNet analizi
    if not args.psg_only:
        try:
            physionet_results = analyze_dataset(
                PHYSIONET_DB, 
                CHANNELS_PHYSIONET,
                max_patients=args.max_patients
            )
        except Exception as e:
            print(f"⚠️ PhysioNet analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # PSG analizi
    if not args.physionet_only:
        try:
            psg_results = analyze_dataset(
                PSG_DB,
                CHANNELS_PSG,
                max_patients=args.max_patients
            )
        except Exception as e:
            print(f"⚠️ PSG analysis failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Karşılaştırma
    physionet_metrics = None
    psg_metrics = None
    
    if physionet_results and psg_results:
        physionet_metrics, psg_metrics = compare_datasets(physionet_results, psg_results)
    elif physionet_results:
        physionet_metrics = aggregate_metrics(physionet_results)
        print(f"\n📊 PhysioNet Metrics Summary:")
        for k, v in list(physionet_metrics.items())[:10]:
            print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
    elif psg_results:
        psg_metrics = aggregate_metrics(psg_results)
        print(f"\n📊 PSG Metrics Summary:")
        for k, v in list(psg_metrics.items())[:10]:
            print(f"  {k}: mean={v['mean']:.4f}, std={v['std']:.4f}")
    
    # Problemli hastaları tespit et
    psg_problematic = []
    physio_problematic = []
    
    if psg_results:
        psg_problematic = identify_problematic_patients(psg_results)
        print_problematic_patients(psg_problematic, "PSG", max_show=15)
    
    if physionet_results:
        physio_problematic = identify_problematic_patients(physionet_results)
        if physio_problematic:
            print_problematic_patients(physio_problematic, "PhysioNet", max_show=5)
    
    # Öneriler
    generate_recommendations(physionet_metrics, psg_metrics, psg_problematic)
    
    # Sonuçları kaydet
    report = {
        'analysis_date': str(np.datetime64('now')),
        'physionet': physionet_results,
        'psg': psg_results,
        'comparison': {
            'physionet_metrics': physionet_metrics,
            'psg_metrics': psg_metrics
        },
        'problematic_psg': psg_problematic,
        'problematic_physionet': physio_problematic
    }
    
    # JSON'a kaydet
    output_path = args.output
    with open(output_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print(f"✅ Analysis complete! Report saved to: {output_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
