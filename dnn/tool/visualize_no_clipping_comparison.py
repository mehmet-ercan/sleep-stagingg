#!/usr/bin/env python3
# coding: utf-8
"""
Amplitude clipping OLMADAN aktarılan psg_data_filtered verilerinin
sinyal kalitesi görselleştirmesi.

Çıktı: dnn/outputs/signal_comparison_no_clipping/
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pymongo import MongoClient
import gridfs
import io
import json
from scipy.stats import pearsonr

# ============================================================================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "psg_data_filtered"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'signal_comparison_no_clipping')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_RATE = 256
EPOCH_SAMPLES = 30 * SAMPLE_RATE

STAGE_COLORS = {'W': '#e74c3c', 'N1': '#f39c12', 'N2': '#3498db', 'N3': '#2ecc71', 'REM': '#9b59b6'}
STAGE_ORDER = ['W', 'N1', 'N2', 'N3', 'REM']


def get_all_patients(fs):
    patients = set()
    for f in fs.find({'metadata.data_type': 'sleep_stages'}):
        patients.add(f.metadata['patient_id'])
    return sorted(patients)


def read_signal(fs, patient_id, channel_name):
    sig_file = fs.find_one({
        'metadata.patient_id': patient_id,
        'metadata.channel_name': channel_name,
        'metadata.data_type': 'raw_signal'
    })
    if not sig_file:
        return None
    data = sig_file.read()
    return np.load(io.BytesIO(data), allow_pickle=True)


def read_stages(fs, patient_id):
    stages_file = fs.find_one({
        'metadata.patient_id': patient_id,
        'metadata.data_type': 'sleep_stages'
    })
    if not stages_file:
        return None
    data = stages_file.read()
    return json.loads(data.decode('utf-8'))


def get_epoch_stds(signal, stages):
    results = {s: [] for s in STAGE_ORDER}
    for stage_info in stages:
        epoch_idx = stage_info['epoch']
        stage = stage_info['stage']
        if stage not in STAGE_ORDER:
            continue
        start = epoch_idx * EPOCH_SAMPLES
        end = start + EPOCH_SAMPLES
        if end > len(signal):
            break
        results[stage].append(np.std(signal[start:end]))
    return results


# ============================================================================
# PLOT 1: Tüm kanallar - stage bazlı std dağılımı
# ============================================================================
def plot_all_channels_by_stage(fs, patients):
    print("Plot 1: all_channels_by_stage.png oluşturuluyor...")
    channels = ['C3-M2', 'F4-M1', 'E1-M2', 'CHIN1-CHIN2']
    sample = patients[:10]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Kanal Bazlı Epoch STD Dağılımı (Amplitude Clipping YOK)', fontsize=14, fontweight='bold')

    for ax, ch in zip(axes.flat, channels):
        all_stds = {s: [] for s in STAGE_ORDER}
        for pid in sample:
            signal = read_signal(fs, pid, ch)
            stages = read_stages(fs, pid)
            if signal is None or stages is None:
                continue
            epoch_stds = get_epoch_stds(signal, stages)
            for s in STAGE_ORDER:
                all_stds[s].extend(epoch_stds[s])

        data = [all_stds[s] for s in STAGE_ORDER]
        bp = ax.boxplot(data, labels=STAGE_ORDER, patch_artist=True, showfliers=False)
        for patch, stage in zip(bp['boxes'], STAGE_ORDER):
            patch.set_facecolor(STAGE_COLORS[stage])
            patch.set_alpha(0.7)

        means = [np.mean(all_stds[s]) if all_stds[s] else 0 for s in STAGE_ORDER]
        valid_means = [m for m in means if m > 0]
        if valid_means:
            ratio = max(valid_means) / min(valid_means)
            ax.set_title(f'{ch} (dinamik oran: {ratio:.2f}x)', fontsize=12)
        else:
            ax.set_title(ch, fontsize=12)
        ax.set_ylabel('Epoch STD (µV)')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'all_channels_by_stage.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Kaydedildi")


# ============================================================================
# PLOT 2: CHIN1-CHIN2 dinamik aralık
# ============================================================================
def plot_chin_dynamic_range(fs, patients):
    print("Plot 2: chin_dynamic_range.png oluşturuluyor...")
    sample = patients[:15]

    all_stds = {s: [] for s in STAGE_ORDER}
    for pid in sample:
        signal = read_signal(fs, pid, 'CHIN1-CHIN2')
        stages = read_stages(fs, pid)
        if signal is None or stages is None:
            continue
        epoch_stds = get_epoch_stds(signal, stages)
        for s in STAGE_ORDER:
            all_stds[s].extend(epoch_stds[s])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle('CHIN1-CHIN2 (EMG) Dinamik Aralık Analizi - Clipping YOK', fontsize=13, fontweight='bold')

    for stage in STAGE_ORDER:
        if all_stds[stage]:
            ax1.hist(all_stds[stage], bins=50, alpha=0.5,
                     label=f'{stage} (n={len(all_stds[stage])})',
                     color=STAGE_COLORS[stage])
    ax1.set_xlabel('Epoch STD (µV)')
    ax1.set_ylabel('Epoch Sayısı')
    ax1.set_title('Stage Bazlı STD Dağılımı')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    means = [np.mean(all_stds[s]) if all_stds[s] else 0 for s in STAGE_ORDER]
    bars = ax2.bar(STAGE_ORDER, means, color=[STAGE_COLORS[s] for s in STAGE_ORDER], alpha=0.8)
    for bar, m in zip(bars, means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f'{m:.1f}', ha='center', fontsize=10)
    ax2.set_ylabel('Ortalama Epoch STD (µV)')
    ax2.set_title('Stage Bazlı Ortalama STD')
    ax2.grid(True, alpha=0.3, axis='y')

    valid = [m for m in means if m > 0]
    if valid:
        ratio = max(valid) / min(valid)
        ax2.text(0.95, 0.95, f'Dinamik Oran: {ratio:.2f}x',
                 transform=ax2.transAxes, ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8), fontsize=11)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'chin_dynamic_range.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Kaydedildi")


# ============================================================================
# PLOT 3: Stage bazlı raw sinyal segmentleri
# ============================================================================
def plot_signal_segments(fs, patients):
    print("Plot 3: signal_segments_no_clipping.png oluşturuluyor...")
    pid = patients[0]
    signal = read_signal(fs, pid, 'CHIN1-CHIN2')
    stages = read_stages(fs, pid)

    target_stages = ['W', 'N2', 'N3', 'REM']
    stage_epochs = {}
    for s_info in stages:
        stage = s_info['stage']
        if stage in target_stages and stage not in stage_epochs:
            epoch_idx = s_info['epoch']
            start = epoch_idx * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            if end <= len(signal):
                stage_epochs[stage] = signal[start:end]
        if len(stage_epochs) == len(target_stages):
            break

    fig, axes = plt.subplots(len(target_stages), 1, figsize=(16, 3 * len(target_stages)))
    fig.suptitle(f'CHIN1-CHIN2 Sinyal Segmentleri - Clipping YOK\nHasta: {pid[:12]}...',
                 fontsize=13, fontweight='bold')

    t = np.arange(EPOCH_SAMPLES) / SAMPLE_RATE
    for ax, stage in zip(axes, target_stages):
        if stage in stage_epochs:
            d = stage_epochs[stage]
            ax.plot(t, d, color=STAGE_COLORS[stage], linewidth=0.5, alpha=0.8)
            ax.set_title(f'{stage} (std={np.std(d):.2f}, min={np.min(d):.2f}, max={np.max(d):.2f})')
            ax.set_ylabel('µV')
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Zaman (s)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'signal_segments_no_clipping.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Kaydedildi")


# ============================================================================
# PLOT 4: C3-M2 vs F4-M1 korelasyon
# ============================================================================
def plot_c3_vs_f4(fs, patients):
    print("Plot 4: c3_vs_f4_correlation_new.png oluşturuluyor...")
    sample = patients[:5]
    stage_data = {s: {'c3': [], 'f4': []} for s in STAGE_ORDER}

    for pid in sample:
        c3 = read_signal(fs, pid, 'C3-M2')
        f4 = read_signal(fs, pid, 'F4-M1')
        stages = read_stages(fs, pid)
        if c3 is None or f4 is None or stages is None:
            continue
        for s_info in stages:
            stage = s_info['stage']
            if stage not in STAGE_ORDER:
                continue
            idx = s_info['epoch']
            start = idx * EPOCH_SAMPLES
            end = start + EPOCH_SAMPLES
            if end > len(c3) or end > len(f4):
                break
            stage_data[stage]['c3'].append(np.std(c3[start:end]))
            stage_data[stage]['f4'].append(np.std(f4[start:end]))

    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle('C3-M2 vs F4-M1 Epoch STD Korelasyonu (Clipping YOK)', fontsize=13, fontweight='bold')

    for ax, stage in zip(axes, STAGE_ORDER):
        x = stage_data[stage]['c3']
        y = stage_data[stage]['f4']
        if len(x) > 2:
            r, _ = pearsonr(x, y)
            ax.scatter(x, y, c=STAGE_COLORS[stage], alpha=0.3, s=10)
            ax.set_title(f'{stage} (r={r:.3f}, n={len(x)})')
        else:
            ax.set_title(f'{stage} (yetersiz veri)')
        ax.set_xlabel('C3-M2 STD')
        ax.set_ylabel('F4-M1 STD')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'c3_vs_f4_correlation_new.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Kaydedildi")


# ============================================================================
# PLOT 5: Özet heatmap
# ============================================================================
def plot_summary_heatmap(fs, patients):
    print("Plot 5: signal_stats_summary.png oluşturuluyor...")
    channels = ['C3-M2', 'C4-M1', 'F3-M2', 'F4-M1', 'O1-M2', 'O2-M1', 'E1-M2', 'E2-M1', 'CHIN1-CHIN2']

    all_data = {ch: {s: [] for s in STAGE_ORDER} for ch in channels}
    for pid in patients:
        stages = read_stages(fs, pid)
        if stages is None:
            continue
        for ch in channels:
            signal = read_signal(fs, pid, ch)
            if signal is None:
                continue
            epoch_stds = get_epoch_stds(signal, stages)
            for s in STAGE_ORDER:
                all_data[ch][s].extend(epoch_stds[s])

    mean_matrix = np.zeros((len(channels), len(STAGE_ORDER)))
    for i, ch in enumerate(channels):
        for j, s in enumerate(STAGE_ORDER):
            if all_data[ch][s]:
                mean_matrix[i, j] = np.mean(all_data[ch][s])

    ratio_per_channel = []
    for i in range(len(channels)):
        row = mean_matrix[i]
        valid = row[row > 0]
        ratio_per_channel.append(max(valid) / min(valid) if len(valid) >= 2 else 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8), gridspec_kw={'width_ratios': [3, 1]})
    fig.suptitle('Tüm Kanallar - Stage Bazlı Ort. Epoch STD (37 Hasta, Clipping YOK)',
                 fontsize=13, fontweight='bold')

    im = ax1.imshow(mean_matrix, aspect='auto', cmap='YlOrRd')
    ax1.set_xticks(range(len(STAGE_ORDER)))
    ax1.set_xticklabels(STAGE_ORDER)
    ax1.set_yticks(range(len(channels)))
    ax1.set_yticklabels(channels)
    ax1.set_xlabel('Uyku Evresi')
    ax1.set_ylabel('Kanal')

    for i in range(len(channels)):
        for j in range(len(STAGE_ORDER)):
            val = mean_matrix[i, j]
            color = 'white' if val > mean_matrix.max() * 0.6 else 'black'
            ax1.text(j, i, f'{val:.1f}', ha='center', va='center', color=color, fontsize=8)

    plt.colorbar(im, ax=ax1, label='Ortalama STD (µV)')

    colors = ['#e74c3c' if r > 2 else '#f39c12' if r > 1.5 else '#95a5a6' for r in ratio_per_channel]
    bars = ax2.barh(range(len(channels)), ratio_per_channel, color=colors, alpha=0.8)
    ax2.set_yticks(range(len(channels)))
    ax2.set_yticklabels(channels)
    ax2.set_xlabel('Dinamik Oran (max/min STD)')
    ax2.set_title('Ayırt Edicilik')
    ax2.axvline(x=1.5, color='orange', linestyle='--', alpha=0.5, label='1.5x')
    ax2.axvline(x=2.0, color='red', linestyle='--', alpha=0.5, label='2.0x')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3, axis='x')

    for bar, r in zip(bars, ratio_per_channel):
        ax2.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                 f'{r:.2f}x', va='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'signal_stats_summary.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  ✓ Kaydedildi")


# ============================================================================
if __name__ == '__main__':
    print("=" * 70)
    print("SİNYAL KALİTE GÖRSELLEŞTİRME - AMPLITUDE CLİPPİNG YOK")
    print("=" * 70)

    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)

    patients = get_all_patients(fs)
    print(f"Toplam hasta: {len(patients)}")
    print(f"Çıktı: {OUTPUT_DIR}\n")

    plot_all_channels_by_stage(fs, patients)
    plot_chin_dynamic_range(fs, patients)
    plot_signal_segments(fs, patients)
    plot_c3_vs_f4(fs, patients)
    plot_summary_heatmap(fs, patients)

    client.close()
    print(f"\n{'=' * 70}")
    print("TAMAMLANDI")
    print(f"{'=' * 70}")
