"""
CHIN1-CHIN2 (Hastane) vs EMG submental (PhysioNet ST)
Evre bazlı sinyal görselleştirmesi — preprocessing etkisi analizi.

Ayrıca RAW (psg_data) vs FILTERED (psg_data_filtered) karşılaştırması.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
from pymongo import MongoClient
import gridfs
import json
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MONGO_URI = "mongodb://localhost:27017/"
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'outputs', 'signal_comparison')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def read_signal(fs, patient_id, channel_name):
    """GridFS'ten sinyal oku (NPY header'lı veya raw)."""
    f = fs.find_one({
        'metadata.patient_id': patient_id,
        'metadata.channel_name': channel_name,
        'metadata.data_type': 'raw_signal'
    })
    if f is None:
        return None
    data_bytes = f.read()
    buf = io.BytesIO(data_bytes)
    try:
        return np.load(buf, allow_pickle=True)
    except:
        return np.frombuffer(data_bytes[128:], dtype='float64')

def get_stage_epochs(fs, patient_id, target_stage, max_count=3):
    """Belirli bir evrenin epoch indekslerini döndür."""
    f = fs.find_one({'metadata.patient_id': patient_id, 'metadata.data_type': 'sleep_stages'})
    if f is None:
        return []
    stages = json.loads(f.read().decode('utf-8'))
    result = []
    for s in stages:
        if s['stage'] == target_stage and len(result) < max_count:
            result.append(s['epoch'])
    return result

def plot_comparison():
    client = MongoClient(MONGO_URI)

    # ================ HASTANE: FILTERED vs RAW ================
    db_filt = client['psg_data_filtered']
    fs_filt = gridfs.GridFS(db_filt)

    # psg_data (raw) kontrol et
    db_raw = client['psg_data']
    fs_raw = gridfs.GridFS(db_raw)

    # PhysioNet ST
    db_st = client['physionet_sleep']
    fs_st = gridfs.GridFS(db_st)

    # İlk hastane hastası
    hosp_patients = sorted(set(
        f.metadata['patient_id'] for f in fs_filt.find({'metadata.data_type': 'sleep_stages'})
    ))
    pid_hosp = hosp_patients[0]

    # İlk ST hastası
    st_patients = sorted(set(
        f.metadata['patient_id'] for f in fs_st.find({'metadata.data_type': 'sleep_stages'})
    ))
    pid_st = st_patients[0]

    stages_to_show = ['W', 'N1', 'N2', 'N3', 'REM']

    # ====================
    # PLOT 1: CHIN1-CHIN2 (filtered) vs EMG submental (ST) — evre bazlı
    # ====================
    fig, axes = plt.subplots(5, 2, figsize=(20, 15))
    fig.suptitle('CHIN1-CHIN2 (Hastane FILTERED) vs EMG submental (PhysioNet ST)\nEvre bazlı sinyal karşılaştırması', fontsize=14)

    chin_sig = read_signal(fs_filt, pid_hosp, 'CHIN1-CHIN2')
    emg_sig = read_signal(fs_st, pid_st, 'EMG submental')

    sr_hosp, sr_st = 256, 100
    spe_hosp, spe_st = sr_hosp * 30, sr_st * 30

    for i, stage in enumerate(stages_to_show):
        # Hastane
        epochs_hosp = get_stage_epochs(fs_filt, pid_hosp, stage, max_count=1)
        if epochs_hosp and chin_sig is not None:
            ep = epochs_hosp[0]
            seg = chin_sig[ep*spe_hosp:(ep+1)*spe_hosp]
            t = np.arange(len(seg)) / sr_hosp
            axes[i, 0].plot(t, seg, 'b-', linewidth=0.3)
            axes[i, 0].set_ylabel(f'{stage}')
            axes[i, 0].set_ylim(-150, 150)
            amp_range = np.max(seg) - np.min(seg)
            std = np.std(seg)
            axes[i, 0].set_title(f'std={std:.1f}µV, range={amp_range:.0f}µV' if i == 0 else f'std={std:.1f}µV')
            if i == 0:
                axes[i, 0].set_title(f'CHIN1-CHIN2 (Hastane, filtered)\nstd={std:.1f}µV, range={amp_range:.0f}µV')

        # PhysioNet ST
        epochs_st = get_stage_epochs(fs_st, pid_st, stage, max_count=1)
        if epochs_st and emg_sig is not None:
            ep = epochs_st[0]
            seg = emg_sig[ep*spe_st:(ep+1)*spe_st]
            t = np.arange(len(seg)) / sr_st
            axes[i, 1].plot(t, seg, 'r-', linewidth=0.3)
            axes[i, 1].set_ylim(-150, 150)
            amp_range = np.max(seg) - np.min(seg)
            std = np.std(seg)
            if i == 0:
                axes[i, 1].set_title(f'EMG submental (PhysioNet ST, RAW)\nstd={std:.1f}µV, range={amp_range:.0f}µV')
            else:
                axes[i, 1].set_title(f'std={std:.1f}µV')

        if i == 4:
            axes[i, 0].set_xlabel('Zaman (s)')
            axes[i, 1].set_xlabel('Zaman (s)')

    plt.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, 'chin_vs_emg_by_stage.png')
    plt.savefig(path1, dpi=150)
    print(f'✓ Kaydedildi: {path1}')
    plt.close()

    # ====================
    # PLOT 2: Tüm kanallar — FILTERED (hastane) evre bazlı
    # ====================
    channels_hosp = ['C3-M2', 'F4-M1', 'E1-M2', 'CHIN1-CHIN2']
    fig, axes = plt.subplots(5, 4, figsize=(24, 15))
    fig.suptitle(f'Hastane Filtered — Tüm kanallar, evre bazlı ({pid_hosp[:8]})', fontsize=14)

    for j, ch in enumerate(channels_hosp):
        sig = read_signal(fs_filt, pid_hosp, ch)
        if sig is None:
            continue
        for i, stage in enumerate(stages_to_show):
            epochs = get_stage_epochs(fs_filt, pid_hosp, stage, max_count=1)
            if epochs:
                ep = epochs[0]
                seg = sig[ep*spe_hosp:(ep+1)*spe_hosp]
                t = np.arange(len(seg)) / sr_hosp
                axes[i, j].plot(t, seg, linewidth=0.3)
                std = np.std(seg)
                if i == 0:
                    axes[i, j].set_title(f'{ch}\nstd={std:.1f}µV')
                else:
                    axes[i, j].set_title(f'std={std:.1f}µV', fontsize=9)
                axes[i, j].set_ylabel(stage)
                axes[i, j].set_ylim(-150, 150)

    plt.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, 'hospital_all_channels_by_stage.png')
    plt.savefig(path2, dpi=150)
    print(f'✓ Kaydedildi: {path2}')
    plt.close()

    # ====================
    # PLOT 3: C3-M2 vs F4-M1 korelasyon
    # ====================
    c3_sig = read_signal(fs_filt, pid_hosp, 'C3-M2')
    f4_sig = read_signal(fs_filt, pid_hosp, 'F4-M1')

    if c3_sig is not None and f4_sig is not None:
        fig, axes = plt.subplots(2, 3, figsize=(18, 8))
        fig.suptitle('C3-M2 vs F4-M1 Korelasyon (Hastane Filtered)', fontsize=14)

        for i, stage in enumerate(['N2', 'N3', 'REM']):
            epochs = get_stage_epochs(fs_filt, pid_hosp, stage, max_count=3)
            c3_segs, f4_segs = [], []
            for ep in epochs:
                c3_seg = c3_sig[ep*spe_hosp:(ep+1)*spe_hosp]
                f4_seg = f4_sig[ep*spe_hosp:(ep+1)*spe_hosp]
                if len(c3_seg) == spe_hosp and len(f4_seg) == spe_hosp:
                    c3_segs.append(c3_seg)
                    f4_segs.append(f4_seg)

            if c3_segs:
                c3_cat = np.concatenate(c3_segs)
                f4_cat = np.concatenate(f4_segs)
                corr = np.corrcoef(c3_cat, f4_cat)[0, 1]

                # Sinyal overlay
                t = np.arange(spe_hosp) / sr_hosp
                axes[0, i].plot(t, c3_segs[0], 'b-', linewidth=0.3, label='C3-M2')
                axes[0, i].plot(t, f4_segs[0], 'r-', linewidth=0.3, alpha=0.7, label='F4-M1')
                axes[0, i].set_title(f'{stage}: corr={corr:.3f}')
                axes[0, i].legend(fontsize=8)
                axes[0, i].set_ylim(-100, 100)

                # Scatter
                axes[1, i].scatter(c3_cat[::10], f4_cat[::10], s=1, alpha=0.3)
                axes[1, i].set_xlabel('C3-M2')
                axes[1, i].set_ylabel('F4-M1')
                axes[1, i].set_title(f'Pearson r = {corr:.3f}')

        plt.tight_layout()
        path3 = os.path.join(OUTPUT_DIR, 'c3_vs_f4_correlation.png')
        plt.savefig(path3, dpi=150)
        print(f'✓ Kaydedildi: {path3}')
        plt.close()

    # ====================
    # PLOT 4: RAW (psg_data) vs FILTERED (psg_data_filtered) CHIN1-CHIN2
    # ====================
    # Raw DB'de CHIN1 ve CHIN2 ayrı ayrı olabilir, kontrol et
    raw_channels = set()
    for f in fs_raw.find({'metadata.patient_id': pid_hosp, 'metadata.data_type': 'raw_signal'}):
        raw_channels.add(f.metadata.get('channel_name', ''))

    print(f'\nRaw DB ({pid_hosp[:8]}) mevcut kanallar: {sorted(raw_channels)[:20]}')

    # CHIN1 ve CHIN2 raw sinyallerini bul
    chin1_raw = read_signal(fs_raw, pid_hosp, 'CHIN1')
    chin2_raw = read_signal(fs_raw, pid_hosp, 'CHIN2')

    if chin1_raw is not None and chin2_raw is not None:
        # Differential hesapla
        min_len = min(len(chin1_raw), len(chin2_raw))
        chin_raw_diff = chin1_raw[:min_len] - chin2_raw[:min_len]

        chin_filt = read_signal(fs_filt, pid_hosp, 'CHIN1-CHIN2')

        fig, axes = plt.subplots(5, 2, figsize=(20, 15))
        fig.suptitle(f'CHIN1-CHIN2: RAW (sol) vs FILTERED (sağ)\nHasta: {pid_hosp[:8]}', fontsize=14)

        for i, stage in enumerate(stages_to_show):
            epochs = get_stage_epochs(fs_filt, pid_hosp, stage, max_count=1)
            if not epochs:
                continue
            ep = epochs[0]

            # Raw
            start_r = ep * spe_hosp
            end_r = start_r + spe_hosp
            if end_r <= len(chin_raw_diff):
                seg_raw = chin_raw_diff[start_r:end_r]
                t = np.arange(len(seg_raw)) / sr_hosp
                axes[i, 0].plot(t, seg_raw, 'g-', linewidth=0.3)
                axes[i, 0].set_ylabel(stage)
                std_r = np.std(seg_raw)
                amp_r = np.max(np.abs(seg_raw))
                if i == 0:
                    axes[i, 0].set_title(f'CHIN RAW (differential)\nstd={std_r:.1f}µV, max={amp_r:.0f}µV')
                else:
                    axes[i, 0].set_title(f'std={std_r:.1f}µV, max={amp_r:.0f}µV', fontsize=9)

            # Filtered
            if chin_filt is not None:
                start_f = ep * spe_hosp
                end_f = start_f + spe_hosp
                if end_f <= len(chin_filt):
                    seg_filt = chin_filt[start_f:end_f]
                    t = np.arange(len(seg_filt)) / sr_hosp
                    axes[i, 1].plot(t, seg_filt, 'b-', linewidth=0.3)
                    std_f = np.std(seg_filt)
                    amp_f = np.max(np.abs(seg_filt))
                    if i == 0:
                        axes[i, 1].set_title(f'CHIN FILTERED (bandpass+clip)\nstd={std_f:.1f}µV, max={amp_f:.0f}µV')
                    else:
                        axes[i, 1].set_title(f'std={std_f:.1f}µV, max={amp_f:.0f}µV', fontsize=9)

        for ax in axes.flat:
            ax.set_ylim(-300, 300)
        plt.tight_layout()
        path4 = os.path.join(OUTPUT_DIR, 'chin_raw_vs_filtered.png')
        plt.savefig(path4, dpi=150)
        print(f'✓ Kaydedildi: {path4}')
        plt.close()
    else:
        print(f'⚠ Raw CHIN1/CHIN2 bulunamadı')

    # ====================
    # PLOT 5: E1-M2 (Hastane) vs EOG horizontal (ST) — evre bazlı
    # ====================
    fig, axes = plt.subplots(5, 2, figsize=(20, 15))
    fig.suptitle('E1-M2 (Hastane FILTERED) vs EOG horizontal (PhysioNet ST RAW)', fontsize=14)

    eog_hosp = read_signal(fs_filt, pid_hosp, 'E1-M2')
    eog_st = read_signal(fs_st, pid_st, 'EOG horizontal')

    for i, stage in enumerate(stages_to_show):
        # Hastane
        epochs_h = get_stage_epochs(fs_filt, pid_hosp, stage, max_count=1)
        if epochs_h and eog_hosp is not None:
            ep = epochs_h[0]
            seg = eog_hosp[ep*spe_hosp:(ep+1)*spe_hosp]
            t = np.arange(len(seg)) / sr_hosp
            axes[i, 0].plot(t, seg, 'b-', linewidth=0.3)
            axes[i, 0].set_ylabel(stage)
            std = np.std(seg)
            if i == 0:
                axes[i, 0].set_title(f'E1-M2 (Hastane, filtered)\nstd={std:.1f}µV')
            else:
                axes[i, 0].set_title(f'std={std:.1f}µV', fontsize=9)
            axes[i, 0].set_ylim(-200, 200)

        # PhysioNet ST
        epochs_s = get_stage_epochs(fs_st, pid_st, stage, max_count=1)
        if epochs_s and eog_st is not None:
            ep = epochs_s[0]
            seg = eog_st[ep*spe_st:(ep+1)*spe_st]
            t = np.arange(len(seg)) / sr_st
            axes[i, 1].plot(t, seg, 'r-', linewidth=0.3)
            std = np.std(seg)
            if i == 0:
                axes[i, 1].set_title(f'EOG horizontal (ST, RAW)\nstd={std:.1f}µV')
            else:
                axes[i, 1].set_title(f'std={std:.1f}µV', fontsize=9)
            axes[i, 1].set_ylim(-500, 500)

        if i == 4:
            axes[i, 0].set_xlabel('Zaman (s)')
            axes[i, 1].set_xlabel('Zaman (s)')

    plt.tight_layout()
    path5 = os.path.join(OUTPUT_DIR, 'eog_hospital_vs_st.png')
    plt.savefig(path5, dpi=150)
    print(f'✓ Kaydedildi: {path5}')
    plt.close()

    client.close()
    print('\n✓ Tüm görselleştirmeler tamamlandı!')


if __name__ == '__main__':
    plot_comparison()
