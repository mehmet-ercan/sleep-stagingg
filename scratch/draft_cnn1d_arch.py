import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

def draw_layer(ax, x, y, w, h, text, color, edgecolor='#4A5568'):
    # Draw a rounded box representing a single layer
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.03,rounding_size=0.04",
        linewidth=1.2,
        edgecolor=edgecolor,
        facecolor=color
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=9, color='#1A202C', multialignment='center')

def draw_block_container(ax, x, y, w, h, title, color='#F7FAFC', edgecolor='#CBD5E0'):
    # Draw a larger outer box to group layers into a block
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.08",
        linewidth=1.5,
        edgecolor=edgecolor,
        facecolor=color,
        linestyle='--'
    )
    ax.add_patch(box)
    ax.text(x, y + h/2 + 0.15, title, ha='center', va='bottom', fontsize=11, fontweight='bold', color='#2D3748')

def draw_arrow(ax, x1, y1, x2, y2, color='#4A5568'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        mutation_scale=12,
        linewidth=1.5,
        color=color
    )
    ax.add_patch(arrow)

def generate_draft():
    # Figure size and limits
    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7)
    ax.axis('off')
    
    # Draw white background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # 1. Input Section Title
    ax.text(1.3, 6.3, "Input PSG Signal\n(30s Epoch)", ha='center', va='center', fontsize=12, fontweight='bold', color='#1A202C')
    
    # Generate realistic signals
    np.random.seed(42)
    t = np.linspace(0, 10, 500)
    
    # EEG: spindle-like + high-frequency noise
    eeg = 0.15 * np.sin(2 * np.pi * t) * np.sin(0.2 * np.pi * t) + 0.05 * np.random.normal(0, 0.5, 500)
    # EOG: slow rolling eye movement
    eog = 0.4 * np.sin(0.5 * np.pi * t) + 0.03 * np.random.normal(0, 0.5, 500)
    # EMG: high frequency muscle tone
    emg = 0.12 * np.random.normal(0, 0.7, 500)
    
    # Plot waveforms in custom bounding boxes
    wave_x = 0.8
    wave_w = 1.2
    wave_h = 0.6
    
    channels = [
        ("EEG (Fz-Cz)", eeg, 4.8),
        ("EOG (ROC-LOC)", eog, 3.6),
        ("EMG (Submental)", emg, 2.4)
    ]
    
    for label, signal, y_center in channels:
        # Draw channel name
        ax.text(wave_x - 0.1, y_center, label, ha='right', va='center', fontsize=9, fontweight='bold', color='#4A5568')
        
        # Plot signal inside a white rectangular box with thin gray outline
        box = FancyBboxPatch(
            (wave_x, y_center - wave_h/2), wave_w, wave_h,
            boxstyle="round,pad=0.01,rounding_size=0.02",
            linewidth=1.0,
            edgecolor='#CBD5E0',
            facecolor='white'
        )
        ax.add_patch(box)
        
        # Superimpose the signal wave inside the box
        # Scale and shift time and signal values to map into matplotlib coordinates
        t_scaled = wave_x + 0.05 + (t / 10.0) * (wave_w - 0.1)
        sig_scaled = y_center + (signal / (1.5 * np.max(np.abs(signal)))) * (wave_h/2 - 0.05)
        ax.plot(t_scaled, sig_scaled, color='#2B6CB0', linewidth=0.8)
        
        # Draw merge arrows from each channel towards the first block input
        draw_arrow(ax, wave_x + wave_w, y_center, 2.5, 3.6, color='#718096')
        
    # Dimension label under inputs
    ax.text(wave_x + wave_w/2, 1.7, "Dimension:\n3 x 3000", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#718096')
    
    # 2. Convolutional Block 1
    # Center x=3.6, y=3.6, width=1.4, height=3.8
    draw_block_container(ax, 3.6, 3.6, 1.4, 3.8, "Convolutional Block 1")
    draw_layer(ax, 3.6, 5.0, 1.2, 0.6, "Conv 1D\n32 filters, k=50", "#EBF8FF")
    draw_layer(ax, 3.6, 4.2, 1.2, 0.5, "Batch Norm", "#E6FFFA")
    draw_layer(ax, 3.6, 3.5, 1.2, 0.5, "ReLU", "#FEFCBF")
    draw_layer(ax, 3.6, 2.8, 1.2, 0.5, "Max Pool 1D\npool=4", "#FED7D7")
    
    # Connection 1 -> 2
    draw_arrow(ax, 2.5, 3.6, 2.8, 3.6)
    draw_arrow(ax, 4.35, 3.6, 4.65, 3.6)
    
    # 3. Convolutional Block 2
    # Center x=5.4, y=3.6, width=1.4, height=3.8
    draw_block_container(ax, 5.4, 3.6, 1.4, 3.8, "Convolutional Block 2")
    draw_layer(ax, 5.4, 5.0, 1.2, 0.6, "Conv 1D\n64 filters, k=25", "#BEE3F8")
    draw_layer(ax, 5.4, 4.2, 1.2, 0.5, "Batch Norm", "#B2F5EA")
    draw_layer(ax, 5.4, 3.5, 1.2, 0.5, "ReLU", "#FEEBC8")
    draw_layer(ax, 5.4, 2.8, 1.2, 0.5, "Max Pool 1D\npool=4", "#FEB2B2")
    
    # Connection 2 -> 3
    draw_arrow(ax, 6.15, 3.6, 6.45, 3.6)
    
    # 4. Convolutional Block 3
    # Center x=7.2, y=3.6, width=1.4, height=3.8
    draw_block_container(ax, 7.2, 3.6, 1.4, 3.8, "Convolutional Block 3")
    draw_layer(ax, 7.2, 5.0, 1.2, 0.6, "Conv 1D\n128 filters, k=10", "#90CDF4")
    draw_layer(ax, 7.2, 4.2, 1.2, 0.5, "Batch Norm", "#81E6D9")
    draw_layer(ax, 7.2, 3.5, 1.2, 0.5, "ReLU", "#FBD38D")
    draw_layer(ax, 7.2, 2.8, 1.2, 0.5, "Max Pool 1D\npool=4", "#FC8181")
    
    # Connection 3 -> Prep
    draw_arrow(ax, 7.95, 3.6, 8.25, 3.6)
    
    # 5. Flatten & Fully Connected Block
    # Center x=9.0, y=3.6, width=1.4, height=3.8
    draw_block_container(ax, 9.0, 3.6, 1.4, 3.8, "Flatten & FC")
    draw_layer(ax, 9.0, 4.8, 1.2, 0.6, "Flatten\n[128x46] -> 5888", "#E6FFFA")
    draw_layer(ax, 9.0, 4.0, 1.2, 0.5, "Dropout (p=0.3)", "#EDF2F7")
    draw_layer(ax, 9.0, 3.2, 1.2, 0.6, "Dense (FC)\n256 units + ReLU", "#FEEBC8")
    draw_layer(ax, 9.0, 2.4, 1.2, 0.5, "Dropout (p=0.3)", "#EDF2F7")
    
    # Connection Prep -> Output
    draw_arrow(ax, 9.75, 3.6, 10.05, 3.6)
    
    # 6. Classification Output Block
    # Center x=10.8, y=3.6, width=1.4, height=3.8
    draw_block_container(ax, 10.8, 3.6, 1.4, 3.8, "Output Block")
    draw_layer(ax, 10.8, 4.6, 1.2, 0.6, "Dense Layer\n5 units", "#FEFCBF")
    draw_layer(ax, 10.8, 3.6, 1.2, 0.6, "Softmax\nActivation", "#EBF8FF")
    draw_layer(ax, 10.8, 2.6, 1.2, 0.6, "Sleep Stages\nW, N1, N2, N3, REM", "#EDF2F7")
    
    # Final output arrow pointing right
    draw_arrow(ax, 11.55, 3.6, 11.85, 3.6)
    
    # Save the draft image
    plt.tight_layout()
    plt.savefig('scratch/cnn1d_draft.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Draft generated and saved to scratch/cnn1d_draft.png")

if __name__ == '__main__':
    generate_draft()
