"""
Generate publication-quality PSG epoch figure for thesis.

Reads real signals from MongoDB psg_data_filtered database.
Shows:
  - EEG (C3-M2): top panel
  - EOG (E1-M2): middle panel
  - EMG (CHIN1-CHIN2): bottom panel

Outputs a high-resolution PDF + PNG for LaTeX inclusion.
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
from matplotlib.ticker import MultipleLocator

# ============================================================================
# Configuration
# ============================================================================
MONGO_URI = "mongodb://localhost:27017/"
DB_NAME = "psg_data_filtered"
SAMPLE_RATE = 256
EPOCH_DURATION = 30  # seconds
SAMPLES_PER_EPOCH = SAMPLE_RATE * EPOCH_DURATION

# Channels to plot
CHANNELS = {
    'EEG': 'C3-M2',
    'EOG': 'E1-M2',
    'EMG': 'CHIN1-CHIN2',
}

# Target sleep stage for the example epoch
TARGET_STAGE = 'N2'

# Output path
OUTPUT_DIR = "/mnt/ssd2/2.SLEEP STAGING/2.REPO/sleep-staging/latex_gtü_tez/figures"


def read_signal(fs, patient_id, channel_name):
    """Read signal from GridFS (NPY-header or raw float64)."""
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


def get_stage_epochs(fs, patient_id, target_stage, max_count=50):
    """Return epoch indices for a given stage."""
    f = fs.find_one({
        'metadata.patient_id': patient_id,
        'metadata.data_type': 'sleep_stages'
    })
    if f is None:
        return []
    stages = json.loads(f.read().decode('utf-8'))
    result = []
    for s in stages:
        if s['stage'] == target_stage and len(result) < max_count:
            result.append(s['epoch'])
    return result


def score_epoch(signals, epoch_idx, spe, sr):
    """Score an epoch for visual quality. Lower is better."""
    score = 0
    
    for label, sig in signals.items():
        start = epoch_idx * spe
        end = start + spe
        if end > len(sig):
            return float('inf')
        seg = sig[start:end]
        std = np.std(seg)
        max_abs = np.max(np.abs(seg))
        p99 = np.percentile(np.abs(seg), 99)
        
        if label == 'EEG':
            # Want moderate EEG with visible K-complexes/spindles
            # std ~20-50 µV, no extreme spikes
            if std < 5 or std > 80:
                score += 1000
            score += abs(std - 25)
            # Penalize extreme artifacts
            if max_abs > 150:
                score += (max_abs - 150) * 2
            # Check for K-complex-like features (sharp transients)
            diff = np.abs(np.diff(seg))
            n_sharp = np.sum(diff > 3 * np.std(diff))
            # Some sharpness is good (K-complexes), too much is noise
            if n_sharp < 10:
                score += 20  # too flat
            elif n_sharp > 500:
                score += 50  # too noisy
                
        elif label == 'EOG':
            # Want calm EOG (N2 has no rapid eye movements)
            # std ~10-40 µV, slow rolling movements are fine
            if std < 3 or std > 60:
                score += 1000
            score += abs(std - 15)
            if max_abs > 200:
                score += (max_abs - 200)
                
        else:  # EMG
            # Want low-amplitude EMG with visible high-freq activity
            # std ~5-25 µV, no extreme bursts
            if std < 2:
                score += 1000  # too flat
            if std > 40:
                score += (std - 40) * 5  # too noisy
            score += abs(std - 12)
            # Penalize large spikes
            if max_abs > 80:
                score += (max_abs - 80) * 3
            # Penalize if p99 is much larger than std (artifact bursts)
            if p99 > 4 * std:
                score += (p99 / std - 4) * 20
    
    return score


def generate_figure():
    """Generate the PSG epoch figure."""
    client = MongoClient(MONGO_URI)
    db = client[DB_NAME]
    fs = gridfs.GridFS(db)
    
    # Get all patients
    patients = sorted(set(
        f.metadata['patient_id']
        for f in fs.find({'metadata.data_type': 'sleep_stages'})
    ))
    
    if not patients:
        print("❌ No patients found in database!")
        return
    
    print(f"Available patients: {len(patients)}")
    
    # Search across multiple patients for the best epoch
    best_overall = None
    best_overall_score = float('inf')
    
    for pid in patients[:10]:  # Try first 10 patients
        print(f"\n--- Patient: {pid[:12]}... ---")
        
        # Read all three channels
        signals = {}
        all_found = True
        for label, ch_name in CHANNELS.items():
            sig = read_signal(fs, pid, ch_name)
            if sig is None:
                print(f"  ⚠ Channel {ch_name} not found")
                all_found = False
                break
            signals[label] = sig
        
        if not all_found:
            continue
        
        # Get N2 epochs
        epochs = get_stage_epochs(fs, pid, TARGET_STAGE, max_count=50)
        if not epochs:
            print(f"  ⚠ No {TARGET_STAGE} epochs")
            continue
        
        print(f"  {len(epochs)} {TARGET_STAGE} epochs found")
        
        # Score each epoch
        for ep in epochs:
            sc = score_epoch(signals, ep, SAMPLES_PER_EPOCH, SAMPLE_RATE)
            if sc < best_overall_score:
                best_overall_score = sc
                best_overall = (pid, ep, {k: v.copy() for k, v in signals.items()})
                print(f"  ★ New best: epoch {ep}, score={sc:.1f}")
    
    if best_overall is None:
        print("❌ Could not find suitable data!")
        client.close()
        return
    
    pid, epoch_idx, signals = best_overall
    print(f"\n{'='*50}")
    print(f"SELECTED: Patient {pid[:12]}, Epoch {epoch_idx}, Score {best_overall_score:.1f}")
    print(f"{'='*50}")
    
    # ========================================================================
    # Create the figure
    # ========================================================================
    plt.rcParams.update({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'font.size': 11,
        'axes.labelsize': 12,
        'axes.titlesize': 12,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'lines.linewidth': 0.4,
        'axes.linewidth': 0.8,
        'xtick.major.width': 0.6,
        'ytick.major.width': 0.6,
    })
    
    fig, axes = plt.subplots(3, 1, figsize=(7.5, 5.0), 
                              gridspec_kw={'hspace': 0.45, 'height_ratios': [1, 1, 0.7]})
    
    # Color palette - publication friendly
    colors = {
        'EEG': '#1a5276',   # Dark blue
        'EOG': '#196f3d',   # Dark green
        'EMG': '#922b21',   # Dark red
    }
    
    channel_labels = {
        'EEG': 'EEG (C3−M2)',
        'EOG': 'EOG (E1−M2)',
        'EMG': 'EMG (Chin)',
    }
    
    for i, (label, ax) in enumerate(zip(['EEG', 'EOG', 'EMG'], axes)):
        sig = signals[label]
        start = epoch_idx * SAMPLES_PER_EPOCH
        end = start + SAMPLES_PER_EPOCH
        segment = sig[start:end]
        
        # Time axis
        t = np.arange(len(segment)) / SAMPLE_RATE
        
        ax.plot(t, segment, color=colors[label], linewidth=0.4, rasterized=True)
        
        # Set y-limits based on actual signal amplitude (dynamic)
        p99 = np.percentile(np.abs(segment), 99)
        ylim = max(p99 * 1.5, 20)  # at least ±20 µV
        # Round ylim to a nice number
        if ylim < 50:
            ylim = int(np.ceil(ylim / 10) * 10)
        elif ylim < 200:
            ylim = int(np.ceil(ylim / 25) * 25)
        else:
            ylim = int(np.ceil(ylim / 50) * 50)
        
        ax.set_ylim(-ylim, ylim)
        
        # Labels
        ax.set_ylabel(f'{channel_labels[label]}\nAmplitude (µV)', fontsize=10)
        
        # Y tick interval
        if ylim <= 30:
            ytick = 10
        elif ylim <= 60:
            ytick = 25
        elif ylim <= 120:
            ytick = 50
        else:
            ytick = 50
        
        ax.yaxis.set_major_locator(MultipleLocator(ytick))
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.grid(True, which='major', alpha=0.3, linewidth=0.5)
        ax.grid(True, which='minor', alpha=0.1, linewidth=0.3, axis='x')
        
        # Clean spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # X-axis label only on bottom
        if i == 2:
            ax.set_xlabel('Time (seconds)', fontsize=11)
        else:
            ax.set_xticklabels([])
        
        ax.set_xlim(0, 30)
    
    # Save
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    pdf_path = os.path.join(OUTPUT_DIR, 'psg_epoch_example.pdf')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.1)
    print(f'\n✓ PDF saved: {pdf_path}')
    
    png_path = os.path.join(OUTPUT_DIR, 'psg_epoch_example.png')
    fig.savefig(png_path, format='png', bbox_inches='tight', pad_inches=0.1)
    print(f'✓ PNG saved: {png_path}')
    
    plt.close()
    
    # Print signal stats
    print(f'\n--- Figure Info ---')
    print(f'Patient: {pid}')
    print(f'Epoch: {epoch_idx} (Stage: {TARGET_STAGE})')
    print(f'Time range: {epoch_idx * 30}s - {(epoch_idx + 1) * 30}s')
    for label in ['EEG', 'EOG', 'EMG']:
        seg = signals[label][epoch_idx * SAMPLES_PER_EPOCH:(epoch_idx + 1) * SAMPLES_PER_EPOCH]
        print(f'  {label} ({CHANNELS[label]}): std={np.std(seg):.1f}µV, '
              f'range=[{np.min(seg):.1f}, {np.max(seg):.1f}]µV')
    
    client.close()
    print('\n✓ Done!')


if __name__ == '__main__':
    generate_figure()
