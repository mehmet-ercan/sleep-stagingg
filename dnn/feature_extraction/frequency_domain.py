"""
Frequency Domain Feature Extraction
Frekans domaininde özellikler çıkarır (FFT, power spektrum, EEG bantları)
"""

import numpy as np
from scipy import signal
from scipy.fft import fft, fftfreq


def create_zero_frequency_features():
    """
    Flat signal (sıfır varyans) için tüm feature'ları sıfır döndür.
    
    Returns:
        dict: Sıfır feature'lar
    """
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    features = {}
    
    # Bant güçleri
    for band in bands:
        features[f'{band}_power'] = 0.0
        features[f'{band}_relative_power'] = 0.0
        features[f'{band}_welch_power'] = 0.0
    
    # Bant oranları
    features['delta_alpha_ratio'] = 0.0
    features['theta_beta_ratio'] = 0.0
    features['slow_fast_ratio'] = 0.0
    features['alpha_beta_ratio'] = 0.0
    
    # Spektral özellikler
    features['dominant_frequency'] = 0.0
    features['dominant_frequency_power'] = 0.0
    features['spectral_centroid'] = 0.0
    features['spectral_spread'] = 0.0
    features['spectral_skewness'] = 0.0
    features['spectral_kurtosis'] = 0.0
    features['spectral_entropy'] = 0.0
    features['edge_frequency_95'] = 0.0
    
    return features


def extract_frequency_domain_features(signal_data, sampling_rate=256):
    """
    Frekans domaininde feature'ları çıkar.
    
    Args:
        signal_data: 1D numpy array
        sampling_rate: Örnekleme frekansı (Hz)
    
    Returns:
        dict: Feature isimleri ve değerleri
    """
    # Sıfır varyans kontrolü (flat signal)
    if np.std(signal_data) < 1e-10:
        return create_zero_frequency_features()
    
    features = {}
    
    # ============================================================================
    # FFT Hesapla
    # ============================================================================
    N = len(signal_data)
    
    # FFT
    fft_vals = fft(signal_data)
    fft_freq = fftfreq(N, 1/sampling_rate)
    
    # Pozitif frekansları al
    positive_freq_idx = fft_freq > 0
    freqs = fft_freq[positive_freq_idx]
    power = np.abs(fft_vals[positive_freq_idx]) ** 2
    
    # Normalize et
    total_power = np.sum(power)
    
    # Sıfır güç kontrolü (flat signal)
    if total_power < 1e-10:
        return create_zero_frequency_features()
    
    power = power / total_power
    
    # ============================================================================
    # EEG Bantları (Uyku analizi için standart)
    # ============================================================================
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 50)
    }
    
    band_powers = {}
    
    for band_name, (low, high) in bands.items():
        band_idx = (freqs >= low) & (freqs <= high)
        band_power = np.sum(power[band_idx])
        band_powers[band_name] = float(band_power)
        features[f'{band_name}_power'] = float(band_power)
    
    # ============================================================================
    # Göreceli Güç (Relative Power)
    # ============================================================================
    total_power = sum(band_powers.values())
    
    for band_name, band_power in band_powers.items():
        relative_power = band_power / (total_power + 1e-8)
        features[f'{band_name}_relative_power'] = float(relative_power)
    
    # ============================================================================
    # Bant Oranları (Band Ratios - EEG için çok önemli!)
    # ============================================================================
    features['delta_alpha_ratio'] = float(
        band_powers['delta'] / (band_powers['alpha'] + 1e-8)
    )
    
    features['theta_beta_ratio'] = float(
        band_powers['theta'] / (band_powers['beta'] + 1e-8)
    )
    
    features['slow_fast_ratio'] = float(
        (band_powers['delta'] + band_powers['theta']) /
        (band_powers['alpha'] + band_powers['beta'] + 1e-8)
    )
    
    features['alpha_beta_ratio'] = float(
        band_powers['alpha'] / (band_powers['beta'] + 1e-8)
    )
    
    # ============================================================================
    # Spektral Özellikler
    # ============================================================================
    # Dominant frekans
    dominant_freq_idx = np.argmax(power)
    features['dominant_frequency'] = float(freqs[dominant_freq_idx])
    features['dominant_frequency_power'] = float(power[dominant_freq_idx])
    
    # Spektral merkez
    features['spectral_centroid'] = float(np.sum(freqs * power) / (np.sum(power) + 1e-8))
    
    # Spektral spread
    spectral_spread = np.sqrt(
        np.sum(((freqs - features['spectral_centroid']) ** 2) * power) /
        (np.sum(power) + 1e-8)
    )
    features['spectral_spread'] = float(spectral_spread)
    
    # Spektral skewness
    features['spectral_skewness'] = float(
        np.sum(((freqs - features['spectral_centroid']) ** 3) * power) /
        ((spectral_spread ** 3) * np.sum(power) + 1e-8)
    )
    
    # Spektral kurtosis
    features['spectral_kurtosis'] = float(
        np.sum(((freqs - features['spectral_centroid']) ** 4) * power) /
        ((spectral_spread ** 4) * np.sum(power) + 1e-8)
    )
    
    # Spektral entropy
    power_norm = power / (np.sum(power) + 1e-10)
    power_norm_safe = np.maximum(power_norm, 1e-10)
    spectral_entropy = -np.sum(power_norm_safe * np.log(power_norm_safe))
    features['spectral_entropy'] = float(spectral_entropy)
    
    # ============================================================================
    # Edge Frequency
    # ============================================================================
    cumsum_power = np.cumsum(power)
    
    if len(cumsum_power) == 0 or cumsum_power[-1] < 1e-10:
        features['edge_frequency_95'] = 0.0
    else:
        edge_95_indices = np.where(cumsum_power >= 0.95 * cumsum_power[-1])[0]
        if len(edge_95_indices) > 0:
            edge_95_idx = edge_95_indices[0]
            features['edge_frequency_95'] = float(freqs[edge_95_idx])
        else:
            features['edge_frequency_95'] = 0.0
    
    # ============================================================================
    # Welch Yöntemi ile Power Spectral Density
    # ============================================================================
    try:
        freqs_welch, psd = signal.welch(
            signal_data,
            fs=sampling_rate,
            nperseg=min(256, len(signal_data))
        )
        
        # Welch yöntemi ile bant güçleri
        for band_name, (low, high) in bands.items():
            band_idx = (freqs_welch >= low) & (freqs_welch <= high)
            if np.sum(band_idx) > 0:
                welch_band_power = np.trapz(psd[band_idx], freqs_welch[band_idx])
                features[f'{band_name}_welch_power'] = float(welch_band_power)
            else:
                features[f'{band_name}_welch_power'] = 0.0
    except Exception:
        # Welch başarısız olursa sıfırla doldur
        for band_name in bands.keys():
            features[f'{band_name}_welch_power'] = 0.0
    
    return features


def calculate_band_power(signal_data, sampling_rate, low_freq, high_freq):
    """
    Belirli bir frekans bandındaki gücü hesapla.
    
    Args:
        signal_data: 1D numpy array
        sampling_rate: Örnekleme frekansı
        low_freq: Alt frekans sınırı
        high_freq: Üst frekans sınırı
    
    Returns:
        float: Bant gücü
    """
    freqs, psd = signal.welch(
        signal_data,
        fs=sampling_rate,
        nperseg=min(256, len(signal_data))
    )
    
    band_idx = (freqs >= low_freq) & (freqs <= high_freq)
    band_power = np.trapz(psd[band_idx], freqs[band_idx])
    
    return float(band_power)


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Test sinyali (simulated EEG)
    np.random.seed(42)
    
    sampling_rate = 256
    duration = 30  # saniye
    t = np.linspace(0, duration, sampling_rate * duration)
    
    # Delta (2 Hz) + Alpha (10 Hz) + noise
    test_signal = (
        2.0 * np.sin(2 * np.pi * 2 * t) +
        1.0 * np.sin(2 * np.pi * 10 * t) +
        0.5 * np.random.randn(len(t))
    )
    
    print("Frequency Domain Feature Extraction Test")
    print("=" * 70)
    
    features = extract_frequency_domain_features(test_signal, sampling_rate)
    
    print(f"\nToplam feature sayısı: {len(features)}")
    
    print("\nEEG Bant Güçleri:")
    for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
        print(f"  {band:6s}: {features[f'{band}_power']:10.6f}")
    
    print("\nBant Oranları:")
    print(f"  Delta/Alpha: {features['delta_alpha_ratio']:10.6f}")
    print(f"  Theta/Beta:  {features['theta_beta_ratio']:10.6f}")
    print(f"  Slow/Fast:   {features['slow_fast_ratio']:10.6f}")
    
    print("\nSpektral Özellikler:")
    print(f"  Dominant Freq: {features['dominant_frequency']:10.2f} Hz")
    print(f"  Spectral Centroid: {features['spectral_centroid']:10.2f} Hz")
    
    # Flat signal testi
    print("\n" + "=" * 70)
    print("Flat Signal Test")
    print("=" * 70)
    
    flat_signal = np.zeros(1000)
    flat_features = extract_frequency_domain_features(flat_signal, sampling_rate)
    
    print(f"Toplam feature: {len(flat_features)}")
    print(f"Delta power: {flat_features['delta_power']}")
    print(f"Alpha power: {flat_features['alpha_power']}")
    
    print("\n✓ Frequency domain feature extraction test başarılı!")