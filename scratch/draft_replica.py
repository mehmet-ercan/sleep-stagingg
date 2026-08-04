import os
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

# Set clean sans-serif font configuration globally
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'Liberation Sans']

def draw_layer_box(ax, x, y, w, h, text, fontsize=8):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.01,rounding_size=0.03",
        linewidth=1.2,
        edgecolor='#2D3748',
        facecolor='white'
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize, color='black', multialignment='center')

def draw_outer_container(ax, x, y, w, h):
    box = FancyBboxPatch(
        (x - w/2, y - h/2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.06",
        linewidth=1.2,
        edgecolor='#4A5568',
        facecolor='white'
    )
    ax.add_patch(box)

def draw_arrow_h(ax, x1, y1, x2, y2, color='#2D3748'):
    arrow = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle='-|>',
        mutation_scale=10,
        linewidth=1.2,
        color=color
    )
    ax.add_patch(arrow)

def draw_arrow_v(ax, x, y1, y2, color='#2D3748'):
    arrow = FancyArrowPatch(
        (x, y1), (x, y2),
        arrowstyle='-|>',
        mutation_scale=8,
        linewidth=1.0,
        color=color
    )
    ax.add_patch(arrow)

def generate_replica():
    fig, ax = plt.subplots(figsize=(15, 7.5))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0.5, 9.5)
    ax.axis('off')
    
    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')
    
    # --- Column 0: Input PSG Signals ---
    ax.text(1.25, 8.2, "Input\nMulti-Channel\nPSG Signal", ha='center', va='center', fontsize=11, fontweight='bold')
    
    # Generate realistic signals
    np.random.seed(42)
    t = np.linspace(0, 10, 300)
    # EEG: spindle + noise
    eeg = 0.15 * np.sin(3.5 * np.pi * t) * np.sin(0.4 * np.pi * t) + 0.04 * np.random.normal(0, 0.5, 300)
    # EOG: slow rolling eye movement
    eog = 0.3 * np.sin(0.4 * np.pi * t) + 0.01 * np.random.normal(0, 0.5, 300)
    # EMG: muscle activation burst
    emg = 0.1 * np.random.normal(0, 0.7, 300)
    emg[100:200] *= 2.5
    
    wave_x = 0.6
    wave_w = 1.3
    wave_h = 0.65
    
    channels = [
        ("EEG\n(Fz-Cz)", eeg, 6.2),
        ("EOG\n(ROC-LOC)", eog, 5.0),
        ("EMG\n(Submental)", emg, 3.8)
    ]
    
    for label, signal, y_center in channels:
        ax.text(wave_x - 0.05, y_center, label, ha='right', va='center', fontsize=9, fontweight='bold')
        
        box = FancyBboxPatch(
            (wave_x, y_center - wave_h/2), wave_w, wave_h,
            boxstyle="round,pad=0.005,rounding_size=0.01",
            linewidth=1.0,
            edgecolor='#4A5568',
            facecolor='white'
        )
        ax.add_patch(box)
        
        t_scaled = wave_x + 0.03 + (t / 10.0) * (wave_w - 0.06)
        sig_scaled = y_center + (signal / (1.5 * np.max(np.abs(signal)))) * (wave_h/2 - 0.04)
        ax.plot(t_scaled, sig_scaled, color='black', linewidth=0.8)
        
    # Dimension label at the bottom of inputs
    ax.text(1.25, 2.5, "(Channels x Length,\ne.g., 3 x 3000)", ha='center', va='center', fontsize=9.5)
    
    # Arrow to Block 1
    draw_arrow_h(ax, 2.0, 5.0, 2.4, 5.0)
    
    # --- Columns 1, 2, 3: Convolutional Blocks ---
    blocks_info = [
        ("Convolutional\nBlock 1", 3.3, "1D Convolution\n32 Filters\nKernel Size 50\nStride 1\nPadding", "ReLU Activation", "Dropout (0.2)", "Max-Pooling 1D\nPool Size 4", "(e.g., L/4 x 32)"),
        ("Convolutional\nBlock 2", 5.5, "1D Convolution\n64 Filters\nKernel Size 25\nStride 1\nPadding", "ReLU Activation", "Dropout (0.2)", "Max-Pooling 1D\nPool Size 4", "(e.g., L/16 x 64)"),
        ("Convolutional\nBlock 3", 7.7, "1D Convolution\n128 Filters\nKernel Size 10\nStride 1\nPadding", "ReLU Activation", "Dropout (0.3)", "Max-Pooling 1D\nPool Size 4", "(e.g., L/64 x 128)")
    ]
    
    for title, x_c, l1_text, l2_text, l3_text, l4_text, b_label in blocks_info:
        ax.text(x_c, 8.2, title, ha='center', va='center', fontsize=11, fontweight='bold')
        
        draw_outer_container(ax, x_c, 5.0, 1.6, 5.2)
        
        draw_layer_box(ax, x_c, 6.9, 1.4, 0.9, l1_text, fontsize=8)
        draw_layer_box(ax, x_c, 5.8, 1.4, 0.45, l2_text, fontsize=8.5)
        draw_layer_box(ax, x_c, 4.9, 1.4, 0.45, l3_text, fontsize=8.5)
        draw_layer_box(ax, x_c, 3.9, 1.4, 0.6, l4_text, fontsize=8)
        
        draw_arrow_v(ax, x_c, 6.4, 6.1)
        draw_arrow_v(ax, x_c, 5.5, 5.2)
        draw_arrow_v(ax, x_c, 4.6, 4.3)
        
        ax.text(x_c, 2.1, b_label, ha='center', va='center', fontsize=9.5)
        
    draw_arrow_h(ax, 4.2, 5.0, 4.6, 5.0)
    draw_arrow_h(ax, 6.4, 5.0, 6.8, 5.0)
    draw_arrow_h(ax, 8.6, 5.0, 9.0, 5.0)
    
    # --- Column 4: Flattening Layer ---
    # Center y is 5.0, total height = 12 * 0.32 = 3.84. Start center at 5.0 - 1.92 + 0.16 = 3.24
    ax.text(9.4, 8.2, "Flattening\nLayer", ha='center', va='center', fontsize=11, fontweight='bold')
    
    flat_y_start = 3.24
    flat_cell_h = 0.32
    flat_cells = 12
    flat_x = 9.3
    flat_w = 0.2
    
    flat_y_centers = []
    for i in range(flat_cells):
        y_c = flat_y_start + i * flat_cell_h
        flat_y_centers.append(y_c)
        cell = Rectangle((flat_x, y_c - flat_cell_h/2), flat_w, flat_cell_h, edgecolor='#2D3748', facecolor='white', linewidth=1.0)
        ax.add_patch(cell)
        
    ax.text(9.4, 1.0, "Flattening\nLayer", ha='center', va='center', fontsize=10)
    
    # --- Column 5: Fully Connected Layer ---
    # Centered at y=5.0
    ax.text(11.2, 8.2, "Fully\nConnected\nLayer\n(256 units)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    fc_y_centers = [6.8, 5.9, 5.0, 4.1, 3.2]
    fc_w = 0.35
    fc_h = 0.35
    fc_x = 11.2
    
    for i, y_c in enumerate(fc_y_centers):
        if i == 2:  # Draw vertical ellipsis for the middle position
            ax.text(fc_x, y_c, "...", ha='center', va='center', fontsize=14, fontweight='bold')
        else:
            cell = Rectangle((fc_x - fc_w/2, y_c - fc_h/2), fc_w, fc_h, edgecolor='#2D3748', facecolor='white', linewidth=1.2)
            ax.add_patch(cell)
            
    ax.text(11.2, 1.0, "Fully\nConnected\nLayer\n(ReLU,\nDropout 0.5)", ha='center', va='center', fontsize=9)
    
    # Draw Dense Web between Flatten and FC
    for fy in flat_y_centers:
        for fi, fcy in enumerate(fc_y_centers):
            if fi != 2:
                ax.plot([flat_x + flat_w, fc_x - fc_w/2], [fy, fcy], color='#718096', linewidth=0.5, alpha=0.35)
                
    # --- Column 6: Output Layer (Softmax) ---
    # Centered at y=5.0
    ax.text(13.4, 8.2, "Output Layer\n(Softmax)\n(1x5)", ha='center', va='center', fontsize=10, fontweight='bold')
    
    out_y_centers = [6.8, 5.9, 5.0, 4.1, 3.2]
    out_w = 0.45
    out_h = 0.45
    out_x = 13.4
    
    stages = [
        ("W (Wake)", [0.85, 0.05, 0.04, 0.03, 0.03]),
        ("N1 (NREM 1)", [0.05, 0.75, 0.10, 0.05, 0.05]),
        ("N2 (NREM 2)", [0.02, 0.08, 0.80, 0.08, 0.02]),
        ("N3 (NREM 3)", [0.01, 0.04, 0.05, 0.88, 0.02]),
        ("REM", [0.05, 0.10, 0.05, 0.02, 0.78])
    ]
    
    for i, y_c in enumerate(out_y_centers):
        # Rectangle representing the Softmax output cell
        cell = Rectangle((out_x - out_w/2, y_c - out_h/2), out_w, out_h, edgecolor='#2D3748', facecolor='white', linewidth=1.2)
        ax.add_patch(cell)
        
        # Stage label next to it (with clear margin)
        label, probs = stages[i]
        ax.text(out_x + out_w/2 + 0.12, y_c, label, ha='left', va='center', fontsize=9, fontweight='bold')
        
        # Separated Probability Bar Chart: starts at x=14.85 to avoid overlapping the label
        bar_x = 14.85
        bar_y = y_c - 0.15
        
        # Draw base axis line for the mini-chart
        ax.plot([bar_x - 0.02, bar_x + 5*0.08 + 4*0.04 + 0.02], [bar_y, bar_y], color='#CBD5E0', linewidth=0.8)
        
        # Draw 5 vertical bars representing the 5 classes
        for j, p in enumerate(probs):
            r_bar = Rectangle((bar_x + j * 0.12, bar_y), 0.08, p * 0.38, facecolor='#4A5568', edgecolor='none')
            ax.add_patch(r_bar)
            
    ax.text(13.4, 1.0, "Output Layer\n(Softmax)", ha='center', va='center', fontsize=10)
    
    # Draw connection lines between FC units and Output boxes
    for fi, fcy in enumerate(fc_y_centers):
        if fi != 2:
            for oy in out_y_centers:
                ax.plot([fc_x + fc_w/2, out_x - out_w/2], [fcy, oy], color='#718096', linewidth=0.5, alpha=0.35)
                
    # Labels (1x5) underneath
    ax.text(13.4, 2.3, "(1x5)", ha='center', va='center', fontsize=9.5)
    
    # Save the figure
    plt.tight_layout()
    plt.savefig('scratch/cnn1d_replica.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Updated replica figure saved to scratch/cnn1d_replica.png")

if __name__ == '__main__':
    generate_replica()
