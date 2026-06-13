"""
Hastane vs PhysioNet ST veri karşılaştırma analizi.
Sinyal kalitesi, sınıf dağılımı, hasta heterojenliği, kanal mevcudiyeti.
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pymongo import MongoClient
import gridfs
import json
from collections import defaultdict

MONGO_URI = "mongodb://localhost:27017/"

# ============================================================================
# 1) Hasta bazlı sınıf dağılımı analizi
# ============================================================================
def analyze_class_distribution(db_name, label="Dataset"):
    """Her hasta için sınıf dağılımını analiz et."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    # Hasta listesi
    patient_ids = set()
    for f in fs.find({"metadata.data_type": "sleep_stages"}):
        patient_ids.add(f.metadata['patient_id'])
    patient_ids = sorted(patient_ids)

    print(f"\n{'='*70}")
    print(f"SINIF DAGILIMI ANALİZİ: {label} ({db_name})")
    print(f"{'='*70}")
    print(f"Toplam hasta: {len(patient_ids)}")

    # Global ve hasta bazlı dağılım
    global_counts = defaultdict(int)
    patient_data = {}

    for pid in patient_ids:
        stages_file = fs.find_one({
            "metadata.patient_id": pid,
            "metadata.data_type": "sleep_stages"
        })
        if stages_file is None:
            continue

        stages = json.loads(stages_file.read().decode('utf-8'))
        counts = defaultdict(int)
        for s in stages:
            stage = s['stage']
            if stage in ['W', 'N1', 'N2', 'N3', 'REM']:
                counts[stage] += 1
                global_counts[stage] += 1

        total = sum(counts.values())
        patient_data[pid] = {
            'counts': dict(counts),
            'total': total,
            'percentages': {k: v/total*100 for k, v in counts.items()} if total > 0 else {}
        }

    # Global dağılım
    total_epochs = sum(global_counts.values())
    print(f"\nGlobal dağılım ({total_epochs} epoch):")
    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        c = global_counts.get(stage, 0)
        print(f"  {stage}: {c:5d} ({c/total_epochs*100:5.1f}%)")

    # Hasta bazlı N1 oranı (en sorunlu sınıf)
    n1_rates = []
    print(f"\nHasta bazlı N1 oranları:")
    for pid in patient_ids:
        if pid not in patient_data:
            continue
        pct = patient_data[pid]['percentages']
        n1_pct = pct.get('N1', 0)
        n1_rates.append(n1_pct)
        total = patient_data[pid]['total']
        n1_count = patient_data[pid]['counts'].get('N1', 0)
        print(f"  {pid[:8]}: N1={n1_count:3d}/{total:4d} ({n1_pct:5.1f}%) | "
              f"W={pct.get('W',0):5.1f}% N2={pct.get('N2',0):5.1f}% "
              f"N3={pct.get('N3',0):5.1f}% REM={pct.get('REM',0):5.1f}%")

    if n1_rates:
        print(f"\n  N1 ortalaması: {np.mean(n1_rates):.1f}% ± {np.std(n1_rates):.1f}%")
        print(f"  N1 min/max: {np.min(n1_rates):.1f}% / {np.max(n1_rates):.1f}%")
        zero_n1 = sum(1 for r in n1_rates if r < 1.0)
        print(f"  N1 < %1 olan hasta: {zero_n1}/{len(n1_rates)}")

    # Hasta bazlı toplam epoch varyansı
    totals = [patient_data[p]['total'] for p in patient_data]
    print(f"\n  Hasta başına epoch: {np.mean(totals):.0f} ± {np.std(totals):.0f} "
          f"(min={np.min(totals)}, max={np.max(totals)})")

    # Hasta heterojenliği: en çok sınıfı baskın olan hastalar
    print(f"\nBaskın sınıf analizi (en yüksek tek-sınıf oranı olan hastalar):")
    dominant_data = []
    for pid in patient_ids:
        if pid not in patient_data or not patient_data[pid]['percentages']:
            continue
        pcts = patient_data[pid]['percentages']
        max_stage = max(pcts, key=pcts.get)
        dominant_data.append((pid, max_stage, pcts[max_stage]))

    dominant_data.sort(key=lambda x: x[2], reverse=True)
    for pid, stage, pct in dominant_data[:10]:
        print(f"  {pid[:8]}: {stage} = {pct:.1f}%")

    client.close()
    return patient_data, global_counts

# ============================================================================
# 2) Kanal mevcudiyeti kontrolü
# ============================================================================
def check_channel_availability(db_name, channels, label="Dataset"):
    """Her hasta için istenen kanalların var olup olmadığını kontrol et."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    print(f"\n{'='*70}")
    print(f"KANAL MEVCUDİYETİ: {label}")
    print(f"İstenen kanallar: {channels}")
    print(f"{'='*70}")

    # Hasta listesi
    patient_ids = set()
    for f in fs.find({"metadata.data_type": "sleep_stages"}):
        patient_ids.add(f.metadata['patient_id'])
    patient_ids = sorted(patient_ids)

    # Her hasta için mevcut kanalları bul
    missing_report = []
    for pid in patient_ids:
        available = set()
        for f in fs.find({"metadata.patient_id": pid, "metadata.data_type": "signal"}):
            available.add(f.metadata.get('channel_name', ''))

        missing = [ch for ch in channels if ch not in available]
        if missing:
            missing_report.append((pid, missing, available))
            print(f"  ⚠ {pid[:8]}: EKSİK kanallar: {missing}")
            print(f"    Mevcut: {sorted(available)}")
        else:
            print(f"  ✓ {pid[:8]}: Tüm kanallar mevcut")

    if missing_report:
        print(f"\n  ⚠ UYARI: {len(missing_report)}/{len(patient_ids)} hastada eksik kanal var!")
    else:
        print(f"\n  ✓ Tüm hastalarda tüm kanallar mevcut")

    client.close()
    return missing_report

# ============================================================================
# 3) Sinyal istatistikleri (amplitüd, SNR, frekans)
# ============================================================================
def analyze_signal_quality(db_name, channels, label="Dataset", max_patients=10, epochs_per_patient=20):
    """Sinyal kalitesi analizi: amplitüd, düzlük, clipping, frekans içeriği."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    print(f"\n{'='*70}")
    print(f"SİNYAL KALİTESİ ANALİZİ: {label}")
    print(f"{'='*70}")

    # Hasta listesi
    patient_ids = set()
    for f in fs.find({"metadata.data_type": "sleep_stages"}):
        patient_ids.add(f.metadata['patient_id'])
    patient_ids = sorted(patient_ids)[:max_patients]

    # Sample rate
    if 'physionet' in db_name:
        sr = 100
    else:
        sr = 256
    samples_per_epoch = sr * 30

    channel_stats = {ch: {
        'amplitudes': [], 'stds': [], 'flat_count': 0, 'clip_count': 0,
        'total_epochs': 0, 'kurtosis': [], 'zero_crossing': [],
        'band_powers': {'delta': [], 'theta': [], 'alpha': [], 'beta': []}
    } for ch in channels}

    for pid in patient_ids:
        # Stage verileri (hangi epoch'lar geçerli)
        stages_file = fs.find_one({
            "metadata.patient_id": pid,
            "metadata.data_type": "sleep_stages"
        })
        if stages_file is None:
            continue
        stages = json.loads(stages_file.read().decode('utf-8'))
        valid_epochs = [s['epoch'] for s in stages if s['stage'] in ['W', 'N1', 'N2', 'N3', 'REM']]
        sample_epochs = valid_epochs[:epochs_per_patient]

        for ch in channels:
            signal_file = fs.find_one({
                "metadata.patient_id": pid,
                "metadata.data_type": "signal",
                "metadata.channel_name": ch
            })
            if signal_file is None:
                continue

            raw_data = np.frombuffer(signal_file.read(), dtype=np.float64)

            for ep in sample_epochs:
                start = ep * samples_per_epoch
                end = start + samples_per_epoch
                if end > len(raw_data):
                    continue

                segment = raw_data[start:end]
                stats = channel_stats[ch]
                stats['total_epochs'] += 1

                # Temel istatistikler
                amp = np.max(np.abs(segment))
                std = np.std(segment)
                stats['amplitudes'].append(amp)
                stats['stds'].append(std)

                # Düz sinyal tespiti (std < 0.1 µV)
                if std < 0.1:
                    stats['flat_count'] += 1

                # Clipping tespiti (>500 µV)
                if amp > 500:
                    stats['clip_count'] += 1

                # Kurtosis (sivrilik - artifact göstergesi)
                from scipy.stats import kurtosis as calc_kurtosis
                stats['kurtosis'].append(calc_kurtosis(segment))

                # Zero-crossing rate
                zc = np.sum(np.diff(np.sign(segment)) != 0) / len(segment)
                stats['zero_crossing'].append(zc)

                # Frekans bant güçleri
                from scipy.fft import rfft, rfftfreq
                freqs = rfftfreq(len(segment), 1.0/sr)
                fft_vals = np.abs(rfft(segment))**2

                def band_power(f_low, f_high):
                    mask = (freqs >= f_low) & (freqs < f_high)
                    return np.mean(fft_vals[mask]) if np.any(mask) else 0

                stats['band_powers']['delta'].append(band_power(0.5, 4))
                stats['band_powers']['theta'].append(band_power(4, 8))
                stats['band_powers']['alpha'].append(band_power(8, 13))
                stats['band_powers']['beta'].append(band_power(13, 30))

    # Sonuçları yazdır
    for ch in channels:
        stats = channel_stats[ch]
        if stats['total_epochs'] == 0:
            print(f"\n  {ch}: VERİ YOK!")
            continue

        n = stats['total_epochs']
        print(f"\n  {ch} ({n} epoch):")
        print(f"    Amplitüd (max|segment|): {np.mean(stats['amplitudes']):.1f} ± {np.std(stats['amplitudes']):.1f} µV")
        print(f"    Std: {np.mean(stats['stds']):.2f} ± {np.std(stats['stds']):.2f} µV")
        print(f"    Kurtosis: {np.mean(stats['kurtosis']):.1f} ± {np.std(stats['kurtosis']):.1f}")
        print(f"    Zero-crossing rate: {np.mean(stats['zero_crossing']):.4f}")
        print(f"    Düz sinyal (std<0.1): {stats['flat_count']}/{n} ({stats['flat_count']/n*100:.1f}%)")
        print(f"    Clipping (>500µV): {stats['clip_count']}/{n} ({stats['clip_count']/n*100:.1f}%)")

        bp = stats['band_powers']
        total_power = np.mean(bp['delta']) + np.mean(bp['theta']) + np.mean(bp['alpha']) + np.mean(bp['beta'])
        if total_power > 0:
            print(f"    Bant güçleri (normalize):")
            print(f"      Delta (0.5-4Hz): {np.mean(bp['delta'])/total_power*100:.1f}%")
            print(f"      Theta (4-8Hz):   {np.mean(bp['theta'])/total_power*100:.1f}%")
            print(f"      Alpha (8-13Hz):  {np.mean(bp['alpha'])/total_power*100:.1f}%")
            print(f"      Beta (13-30Hz):  {np.mean(bp['beta'])/total_power*100:.1f}%")

    client.close()
    return channel_stats

# ============================================================================
# 4) Split varyansı analizi
# ============================================================================
def analyze_split_variance(db_name, n_splits=10, n_train=25, n_val=6, n_test=6):
    """Farklı random split'lerin val/test sınıf dağılımı üzerindeki etkisi."""
    client = MongoClient(MONGO_URI)
    db = client[db_name]
    fs = gridfs.GridFS(db)

    print(f"\n{'='*70}")
    print(f"SPLIT VARYANS ANALİZİ: {db_name}")
    print(f"{'='*70}")

    # Hasta listesi ve stage verileri
    patient_ids = set()
    for f in fs.find({"metadata.data_type": "sleep_stages"}):
        patient_ids.add(f.metadata['patient_id'])
    patient_ids = sorted(patient_ids)

    patient_stages = {}
    for pid in patient_ids:
        stages_file = fs.find_one({
            "metadata.patient_id": pid,
            "metadata.data_type": "sleep_stages"
        })
        if stages_file is None:
            continue
        stages = json.loads(stages_file.read().decode('utf-8'))
        counts = defaultdict(int)
        for s in stages:
            if s['stage'] in ['W', 'N1', 'N2', 'N3', 'REM']:
                counts[s['stage']] += 1
        patient_stages[pid] = dict(counts)

    import random
    val_accs_proxy = []
    val_n1_pcts = []
    test_n1_pcts = []

    print(f"\n{n_splits} random split simülasyonu:")
    for i in range(n_splits):
        random.seed(i * 1000 + 42)
        pids = list(patient_stages.keys())
        random.shuffle(pids)

        val_pids = pids[n_train:n_train+n_val]
        test_pids = pids[n_train+n_val:n_train+n_val+n_test]

        # Val sınıf dağılımı
        val_counts = defaultdict(int)
        for pid in val_pids:
            for stage, count in patient_stages[pid].items():
                val_counts[stage] += count

        test_counts = defaultdict(int)
        for pid in test_pids:
            for stage, count in patient_stages[pid].items():
                test_counts[stage] += count

        val_total = sum(val_counts.values())
        test_total = sum(test_counts.values())
        val_n1_pct = val_counts.get('N1', 0) / val_total * 100 if val_total > 0 else 0
        test_n1_pct = test_counts.get('N1', 0) / test_total * 100 if test_total > 0 else 0

        val_n1_pcts.append(val_n1_pct)
        test_n1_pcts.append(test_n1_pct)

        # N2 baskınlığı (accuracy proxy - N2 doğru tahmin = yüksek acc)
        val_n2_pct = val_counts.get('N2', 0) / val_total * 100 if val_total > 0 else 0
        val_accs_proxy.append(val_n2_pct)

        print(f"  Split {i}: Val N1={val_n1_pct:.1f}% N2={val_n2_pct:.1f}% | "
              f"Test N1={test_n1_pct:.1f}%  (val={val_total}, test={test_total} epoch)")

    print(f"\n  Val N1 varyansı: {np.mean(val_n1_pcts):.1f}% ± {np.std(val_n1_pcts):.1f}% "
          f"(min={np.min(val_n1_pcts):.1f}%, max={np.max(val_n1_pcts):.1f}%)")
    print(f"  Test N1 varyansı: {np.mean(test_n1_pcts):.1f}% ± {np.std(test_n1_pcts):.1f}% "
          f"(min={np.min(test_n1_pcts):.1f}%, max={np.max(test_n1_pcts):.1f}%)")
    print(f"  Val N2 varyansı: {np.mean(val_accs_proxy):.1f}% ± {np.std(val_accs_proxy):.1f}%")

    client.close()

# ============================================================================
# 5) PhysioNet ST ile kıyaslama
# ============================================================================
def compare_datasets():
    """Hastane ve PhysioNet ST veri setlerini karşılaştır."""
    print(f"\n{'#'*70}")
    print(f"HASTANE vs PhysioNet ST KARŞILAŞTIRMA ANALİZİ")
    print(f"{'#'*70}")

    # 1) Sınıf dağılımı
    hosp_data, hosp_counts = analyze_class_distribution("psg_data_filtered", "Hastane")
    st_data, st_counts = analyze_class_distribution("physionet_sleep", "PhysioNet ST")

    # 2) Kanal mevcudiyeti (hastane)
    check_channel_availability("psg_data_filtered",
                               ['C3-M2', 'F4-M1', 'E1-M2', 'CHIN1-CHIN2'],
                               "Hastane (4 kanal)")

    # 3) Sinyal kalitesi
    print(f"\n{'#'*70}")
    print(f"SİNYAL KALİTESİ KARŞILAŞTIRMASI")
    print(f"{'#'*70}")
    analyze_signal_quality("psg_data_filtered",
                          ['C3-M2', 'F4-M1', 'E1-M2', 'CHIN1-CHIN2'],
                          "Hastane", max_patients=15, epochs_per_patient=30)
    analyze_signal_quality("physionet_sleep",
                          ['EEG Fpz-Cz', 'EEG Pz-Oz', 'EOG horizontal', 'EMG submental'],
                          "PhysioNet ST", max_patients=15, epochs_per_patient=30)

    # 4) Split varyansı
    analyze_split_variance("psg_data_filtered", n_splits=20)

    # 5) Özet karşılaştırma
    print(f"\n{'#'*70}")
    print(f"ÖZET KARŞILAŞTIRMA")
    print(f"{'#'*70}")

    hosp_total = sum(hosp_counts.values())
    st_total = sum(st_counts.values())

    print(f"\n{'Metrik':<30} {'Hastane':>15} {'PhysioNet ST':>15}")
    print(f"{'-'*60}")
    print(f"{'Hasta sayısı':<30} {len(hosp_data):>15} {len(st_data):>15}")
    print(f"{'Toplam epoch':<30} {hosp_total:>15} {st_total:>15}")
    print(f"{'Epoch/hasta (ort)':<30} {hosp_total/len(hosp_data):>15.0f} {st_total/len(st_data):>15.0f}")

    for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
        h_pct = hosp_counts.get(stage, 0) / hosp_total * 100
        s_pct = st_counts.get(stage, 0) / st_total * 100
        print(f"{stage + ' oranı':<30} {h_pct:>14.1f}% {s_pct:>14.1f}%")

    print(f"\n{'Kanal sayısı':<30} {'4 (C3,F4,E1,CHIN)':>15} {'4 (Fpz,Pz,EOG,EMG)':>15}")
    print(f"{'Sample rate':<30} {'256 Hz':>15} {'100 Hz':>15}")
    print(f"{'Train/Val/Test':<30} {'25/6/6':>15} {'34/5/5':>15}")
    print(f"{'Seed':<30} {'null (random!)':>15} {'null (random!)':>15}")


if __name__ == "__main__":
    compare_datasets()
