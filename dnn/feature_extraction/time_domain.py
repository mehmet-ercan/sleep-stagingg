"""
Time Domain Feature Extraction
Zaman domaininde istatistiksel özellikler çıkarır
"""

import numpy as np
from scipy import stats
from scipy.signal import find_peaks


def extract_time_domain_features(signal):
    """
    Zaman domaininde feature'ları çıkar.
    
    Args:
        signal: 1D numpy array (örnek: 7680 samples)
    
    Returns:
        dict: Feature isimleri ve değerleri
    """
    features = {}
    
    # ============================================================================
    # Temel İstatistikler
    # ============================================================================
    features['mean'] = float(np.mean(signal))
    features['std'] = float(np.std(signal))
    features['var'] = float(np.var(signal))
    features['min'] = float(np.min(signal))
    features['max'] = float(np.max(signal))
    features['range'] = float(np.ptp(signal))  # peak-to-peak
    features['median'] = float(np.median(signal))
    
    # ============================================================================
    # Dağılım İstatistikleri
    # ============================================================================
    features['skewness'] = float(stats.skew(signal))  # Asimetri
    features['kurtosis'] = float(stats.kurtosis(signal))  # Sivrilik
    
    # ============================================================================
    # Enerji ve Güç
    # ============================================================================
    features['energy'] = float(np.sum(signal ** 2))  # Toplam enerji
    features['power'] = float(np.mean(signal ** 2))  # Ortalama güç
    features['rms'] = float(np.sqrt(np.mean(signal ** 2)))  # Root mean square
    
    # ============================================================================
    # Değişim Özellikleri
    # ============================================================================
    # Zero crossing rate (sıfır geçiş oranı)
    zero_crossings = np.where(np.diff(np.sign(signal)))[0]
    features['zero_crossing_rate'] = float(len(zero_crossings) / len(signal))
    
    # Mean crossing rate
    signal_centered = signal - np.mean(signal)
    mean_crossings = np.where(np.diff(np.sign(signal_centered)))[0]
    features['mean_crossing_rate'] = float(len(mean_crossings) / len(signal))
    
    # ============================================================================
    # Türev Özellikleri (Değişim hızı)
    # ============================================================================
    signal_diff = np.diff(signal)
    features['diff_mean'] = float(np.mean(signal_diff))
    features['diff_std'] = float(np.std(signal_diff))
    features['diff_max'] = float(np.max(np.abs(signal_diff)))
    
    # ============================================================================
    # Percentile Özellikleri
    # ============================================================================
    features['percentile_25'] = float(np.percentile(signal, 25))
    features['percentile_75'] = float(np.percentile(signal, 75))
    features['iqr'] = float(np.percentile(signal, 75) - np.percentile(signal, 25))  # Interquartile range
    
    # ============================================================================
    # Peak Özellikleri
    # ============================================================================
    # Pozitif ve negatif peak sayısı
    peaks_pos, _ = find_peaks(signal)
    peaks_neg, _ = find_peaks(-signal)
    
    features['n_peaks_pos'] = float(len(peaks_pos))
    features['n_peaks_neg'] = float(len(peaks_neg))
    features['peak_rate'] = float((len(peaks_pos) + len(peaks_neg)) / len(signal))
    
    # ============================================================================
    # Entropi (Düzensizlik ölçüsü)
    # ============================================================================
    # Sample entropy (basitleştirilmiş)
    features['sample_entropy'] = float(calculate_sample_entropy(signal))
    
    # ============================================================================
    # Hjorth Parameters (EEG için önemli)
    # ============================================================================
    hjorth = calculate_hjorth_parameters(signal)
    features['hjorth_activity'] = hjorth['activity']
    features['hjorth_mobility'] = hjorth['mobility']
    features['hjorth_complexity'] = hjorth['complexity']
    
    return features


def calculate_sample_entropy(signal, m=2, r=None):
    """
    Sample Entropy hesapla (sinyalin düzensizlik ölçüsü)
    
    Args:
        signal: 1D array
        m: Pattern uzunluğu
        r: Tolerans (None ise std * 0.2)
    
    Returns:
        float: Sample entropy değeri
    """
    if r is None:
        r = 0.2 * np.std(signal)
    
    N = len(signal)
    
    # Basitleştirilmiş versiyon (hız için)
    # Tam implementasyon çok yavaş
    
    # Normalize et
    signal_norm = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)
    
    # Ardışık farkların standart sapması (proxy)
    diff_std = np.std(np.diff(signal_norm))
    
    # Entropi proxy
    entropy = -np.log(diff_std + 1e-8)
    
    return float(entropy)


def calculate_hjorth_parameters(signal):
    """
    Hjorth Parameters (EEG analizi için standart)
    
    Activity: Sinyal varyansı
    Mobility: Standart sapmanın değişim hızı
    Complexity: Mobility'nin değişim hızı
    
    Args:
        signal: 1D array
    
    Returns:
        dict: Hjorth parametreleri
    """
    # İlk türev
    d1 = np.diff(signal)
    
    # İkinci türev
    d2 = np.diff(d1)
    
    # Varyanslar
    var_signal = np.var(signal)
    var_d1 = np.var(d1)
    var_d2 = np.var(d2)
    
    # Activity (güç)
    activity = var_signal
    
    # Mobility (frekans içeriği)
    mobility = np.sqrt(var_d1 / (var_signal + 1e-8))
    
    # Complexity (dalga formu kompleksitesi)
    complexity = np.sqrt(var_d2 / (var_d1 + 1e-8)) / (mobility + 1e-8)
    
    return {
        'activity': float(activity),
        'mobility': float(mobility),
        'complexity': float(complexity)
    }


# ============================================================================
# Test
# ============================================================================
if __name__ == "__main__":
    # Test sinyali
    np.random.seed(42)
    test_signal = np.random.randn(7680)  # 30 saniye @ 256 Hz
    
    print("Time Domain Feature Extraction Test")
    print("=" * 70)
    
    features = extract_time_domain_features(test_signal)
    
    print(f"\nToplam feature sayısı: {len(features)}")
    print("\nÖrnek feature'lar:")
    for i, (name, value) in enumerate(list(features.items())[:10]):
        print(f"  {name:20s}: {value:10.4f}")
    
    print("\n✓ Time domain feature extraction test başarılı!")