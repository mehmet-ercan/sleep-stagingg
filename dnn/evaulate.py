"""
Sleep Stage Classification - Evaluation Script
Test set değerlendirme, confusion matrix, hipnogram karşılaştırma
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend (tkinter crash önlemi)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, 
                             cohen_kappa_score, accuracy_score,
                             precision_recall_fscore_support)
import os
from collections import defaultdict

import config
from models.cnn1d import get_model
from dataset import (split_patients, create_dataloaders, 
                     create_test_dataloader, create_test_dataloader_from_cache,
                     preload_all_signals_to_cache,
                     SleepSequenceDataset, SleepEpochDataset)
from edf_to_mongo import preload_all_stages, _STAGES_CACHE, get_offline_patient_ids
from torch.utils.data import DataLoader


class Evaluator:
    """
    Model değerlendirme sınıfı
    """
    
    def __init__(self, model, test_loader, device=config.DEVICE, run_dir=None, 
                 sequence_mode=False):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.run_dir = run_dir
        self.sequence_mode = sequence_mode
        
        # Sonuçları sakla
        self.all_predictions = []
        self.all_labels = []
        self.all_probs = []
        
        # Hasta bazlı sonuçlar
        self.patient_predictions = defaultdict(list)
        self.patient_labels = defaultdict(list)
    
    def evaluate(self):
        """
        Test set üzerinde değerlendirme yap.
        Sequence mode destekler: 3D output'u flatten eder.
        """
        print(f"\n{'='*70}")
        print(f"TEST SET DEĞERLENDİRMESİ")
        if self.sequence_mode:
            print(f"Mode: SEQUENCE (seq_len={config.SEQUENCE_LENGTH})")
        else:
            print(f"Mode: SINGLE EPOCH")
        print(f"{'='*70}\n")
        
        self.model.eval()
        
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch_idx, (signals, labels) in enumerate(self.test_loader):
                # Device'a taşı
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                outputs = self.model(signals)
                
                # Sequence mode: outputs [B, seq_len, C], labels [B, seq_len]
                if self.sequence_mode and outputs.dim() == 3:
                    B, S, C = outputs.shape
                    outputs_flat = outputs.reshape(B * S, C)
                    labels_flat = labels.reshape(B * S)
                else:
                    outputs_flat = outputs
                    labels_flat = labels
                
                probs = F.softmax(outputs_flat, dim=1)
                _, predicted = torch.max(outputs_flat, 1)
                
                # İstatistikler
                total += labels_flat.size(0)
                correct += (predicted == labels_flat).sum().item()
                
                # Sonuçları sakla
                self.all_predictions.extend(predicted.cpu().numpy())
                self.all_labels.extend(labels_flat.cpu().numpy())
                self.all_probs.extend(probs.cpu().numpy())
                
                # Progress
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Batch [{batch_idx + 1}/{len(self.test_loader)}] işlendi")
        
        # Genel istatistikler
        accuracy = 100. * correct / total
        
        print(f"\n{'='*70}")
        print(f"GENEL SONUÇLAR")
        print(f"{'='*70}")
        print(f"Toplam test epoch'u: {total}")
        print(f"Doğru tahmin: {correct}")
        print(f"Overall Accuracy: {accuracy:.2f}%")
        print(f"{'='*70}\n")
        
        return accuracy
    
    def calculate_metrics(self):
        """
        Detaylı metrikleri hesapla
        """
        print(f"\n{'='*70}")
        print(f"DETAYLI METRİKLER")
        print(f"{'='*70}\n")
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        
        # Accuracy
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Overall Accuracy: {accuracy*100:.2f}%")
        
        # Cohen's Kappa
        kappa = cohen_kappa_score(y_true, y_pred)
        print(f"Cohen's Kappa: {kappa:.4f}")
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, labels=range(config.N_CLASSES),
            zero_division=0  # Tahmin edilmeyen sınıflar için 0 döndür (warning bastır)
        )
        
        print(f"\n{'='*70}")
        print(f"PER-CLASS METRİKLER")
        print(f"{'='*70}")
        print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12} {'Support':<10}")
        print(f"{'-'*70}")
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            print(f"{class_name:<10} {precision[i]:<12.4f} {recall[i]:<12.4f} "
                  f"{f1[i]:<12.4f} {support[i]:<10}")
        
        # Macro average
        macro_precision = precision.mean()
        macro_recall = recall.mean()
        macro_f1 = f1.mean()
        
        print(f"{'-'*70}")
        print(f"{'Macro Avg':<10} {macro_precision:<12.4f} {macro_recall:<12.4f} "
              f"{macro_f1:<12.4f} {support.sum():<10}")
        print(f"{'='*70}\n")
        
        # Classification report
        print("\nScikit-learn Classification Report:")
        print("="*70)
        report = classification_report(
            y_true, y_pred, 
            target_names=config.CLASS_NAMES,
            digits=4,
            zero_division=0  # Tahmin edilmeyen sınıflar için 0 döndür
        )
        print(report)
        
        metrics = {
            'accuracy': accuracy,
            'kappa': kappa,
            'precision': precision.tolist(),
            'recall': recall.tolist(),
            'f1': f1.tolist(),
            'support': support.tolist(),
            'macro_precision': macro_precision,
            'macro_recall': macro_recall,
            'macro_f1': macro_f1
        }
        
        return metrics
    
    def plot_confusion_matrix(self, save_path=None, normalize=True):
        """
        Confusion matrix çizdir
        
        Args:
            save_path: Kaydedilecek dosya yolu
            normalize: True ise normalize et (yüzde olarak)
        """
        print(f"\n{'='*70}")
        print(f"CONFUSION MATRIX OLUŞTURULUYOR")
        print(f"{'='*70}\n")
        
        y_true = np.array(self.all_labels)
        y_pred = np.array(self.all_predictions)
        
        # Confusion matrix hesapla
        cm = confusion_matrix(y_true, y_pred, labels=range(config.N_CLASSES))
        
        if normalize:
            # Satır bazında normalize et (her sınıfın toplamı 1 olsun)
            cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            cm_normalized = np.nan_to_num(cm_normalized)  # NaN'ları 0 yap
        else:
            cm_normalized = cm
        
        # Plot
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Heatmap
        sns.heatmap(
            cm_normalized if normalize else cm,
            annot=True,
            fmt='.2f' if normalize else 'd',
            cmap='Blues',
            xticklabels=config.CLASS_NAMES,
            yticklabels=config.CLASS_NAMES,
            cbar_kws={'label': 'Proportion' if normalize else 'Count'},
            ax=ax
        )
        
        ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
        ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
        
        title = 'Confusion Matrix'
        if normalize:
            title += ' (Normalized)'
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Confusion matrix kaydedildi: {save_path}")
        
        plt.close()
        
        # Raw confusion matrix'i de yazdır
        print("\nRaw Confusion Matrix:")
        print("="*70)
        print(f"{'':>10}", end='')
        for class_name in config.CLASS_NAMES:
            print(f"{class_name:>10}", end='')
        print()
        print("-"*70)
        
        for i, class_name in enumerate(config.CLASS_NAMES):
            print(f"{class_name:>10}", end='')
            for j in range(config.N_CLASSES):
                print(f"{cm[i, j]:>10}", end='')
            print()
        print("="*70)
    
    def plot_per_class_metrics(self, metrics, save_path=None):
        """
        Her sınıf için precision, recall, f1 bar chart
        """
        print(f"\n{'='*70}")
        print(f"PER-CLASS METRİK GRAFİĞİ OLUŞTURULUYOR")
        print(f"{'='*70}\n")
        
        # Veriyi hazırla
        classes = config.CLASS_NAMES
        precision = metrics['precision']
        recall = metrics['recall']
        f1 = metrics['f1']
        
        x = np.arange(len(classes))
        width = 0.25
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Bars
        bars1 = ax.bar(x - width, precision, width, label='Precision', color='#3498db')
        bars2 = ax.bar(x, recall, width, label='Recall', color='#e74c3c')
        bars3 = ax.bar(x + width, f1, width, label='F1-Score', color='#2ecc71')
        
        # Etiketler
        ax.set_xlabel('Sleep Stage', fontsize=12, fontweight='bold')
        ax.set_ylabel('Score', fontsize=12, fontweight='bold')
        ax.set_title('Per-Class Metrics', fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(classes)
        ax.legend()
        ax.set_ylim([0, 1.0])
        ax.grid(axis='y', alpha=0.3)
        
        # Bar üzerine değerleri yaz
        def autolabel(bars):
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=8)
        
        autolabel(bars1)
        autolabel(bars2)
        autolabel(bars3)
        
        plt.tight_layout()
        
        # Kaydet
        if save_path is None:
            save_path = os.path.join(config.OUTPUT_DIR, 'per_class_metrics.png')
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Per-class metrics grafiği kaydedildi: {save_path}")
        
        plt.close()
    
    def generate_hypnograms_for_test_patients(self):
        """
        Test set'teki her hasta için hipnogram oluştur (predicted vs ground truth).
        Hem single-epoch hem sequence mode destekler.
        """
        print(f"\n{'='*70}")
        print(f"HİPNOGRAMLAR OLUŞTURULUYOR (TEST HASTALARI)")
        print(f"{'='*70}\n")
        
        test_dataset = self.test_loader.dataset
        
        # Her hasta için epoch'ları grupla
        patient_epochs = defaultdict(lambda: {'predictions': [], 'labels': [], 'epoch_indices': []})
        
        if self.sequence_mode:
            # Sequence mode: tahminler zaten flatten edildiş (evaluate()'da)
            # Sequence metadata'dan hasta ve epoch bilgilerini çıkar
            pred_idx = 0
            for seq_idx in range(len(test_dataset)):
                meta = test_dataset.sequence_metadata[seq_idx]
                patient_id = meta['patient_id']
                start_idx = meta['start_idx']
                seq_len = meta.get('length', config.SEQUENCE_LENGTH)
                
                for offset in range(seq_len):
                    if pred_idx >= len(self.all_predictions):
                        break
                    epoch_idx = start_idx + offset
                    pred = self.all_predictions[pred_idx]
                    label = self.all_labels[pred_idx]
                    
                    # Aynı epoch'un tekrar eklenmesini önle
                    # (overlapping sequences'da aynı epoch birden fazla kez tahmin edilebilir)
                    key = (patient_id, epoch_idx)
                    existing_indices = patient_epochs[patient_id]['epoch_indices']
                    if epoch_idx not in existing_indices:
                        patient_epochs[patient_id]['predictions'].append(pred)
                        patient_epochs[patient_id]['labels'].append(label)
                        patient_epochs[patient_id]['epoch_indices'].append(epoch_idx)
                    
                    pred_idx += 1
        else:
            # Single epoch mode: epoch_metadata'dan direkt oku
            for idx in range(len(test_dataset)):
                meta = test_dataset.epoch_metadata[idx]
                patient_id = meta['patient_id']
                epoch_idx = meta['epoch_idx']
                
                pred = self.all_predictions[idx]
                label = self.all_labels[idx]
                
                patient_epochs[patient_id]['predictions'].append(pred)
                patient_epochs[patient_id]['labels'].append(label)
                patient_epochs[patient_id]['epoch_indices'].append(epoch_idx)
        
        # Her hasta için hipnogram çiz
        for patient_id, data in patient_epochs.items():
            self._plot_hypnogram_comparison(
                patient_id=patient_id,
                predictions=data['predictions'],
                labels=data['labels'],
                epoch_indices=data['epoch_indices']
            )
    
    def _plot_hypnogram_comparison(self, patient_id, predictions, labels, epoch_indices):
        """
        Tek bir hasta için predicted vs ground truth hipnogram
        Tüm uyku verisi için görselleştirme yapar
        """
        # Epoch'ları sırala
        sorted_data = sorted(zip(epoch_indices, predictions, labels))
        epoch_indices, predictions, labels = zip(*sorted_data)
        
        # Toplam epoch sayısı ve süre bilgisi
        n_epochs = len(epoch_indices)
        total_hours = n_epochs * 30 / 3600  # 30 saniye per epoch → saat
        
        print(f"  {patient_id}: {n_epochs} epoch ({total_hours:.1f} saat)")
        
        # Stage mapping (display için)
        stage_map = {
            0: 0,    # W
            1: -2,   # N1
            2: -3,   # N2
            3: -4,   # N3
            4: -1    # REM
        }
        
        # Sayısal değerlere çevir
        pred_stages = [stage_map[p] for p in predictions]
        true_stages = [stage_map[l] for l in labels]
        
        # Zaman ekseni (saat cinsinden) - sıralı epoch'lar için 0'dan başlat
        # epoch_indices mutlak değer olabilir, bu yüzden relatif zaman kullan
        times = np.arange(n_epochs) * 30 / 3600  # Her epoch 30 saniye
        
        # Renk paleti
        stage_colors = {
            0: '#FF6B6B',    # W
            -1: '#4ECDC4',   # REM
            -2: '#95E1D3',   # N1
            -3: '#45B7D1',   # N2
            -4: '#2C3E50'    # N3
        }
        
        # Dinamik figsize (uyku süresine göre)
        # Her saat için ~2 inch genişlik, minimum 16, maksimum 40
        fig_width = max(16, min(40, total_hours * 2))
        fig_height = 8
        
        # Plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(fig_width, fig_height), sharex=True)
        
        epoch_duration = 30 / 3600  # 30 saniye = saat cinsinden
        
        # Ground Truth
        for i in range(len(times)):
            color = stage_colors[true_stages[i]]
            rect = plt.Rectangle(
                (times[i], true_stages[i] - 0.4),
                epoch_duration,
                0.8,
                facecolor=color,
                edgecolor='none',
                alpha=0.9
            )
            ax1.add_patch(rect)
        
        # X-axis limitleri ayarla (0'dan başla, tüm veriyi göster)
        max_time = times[-1] + epoch_duration if len(times) > 0 else 8
        ax1.set_xlim(0, max(max_time, 8))  # En az 8 saat göster
        ax1.set_ylim(-4.5, 0.5)
        ax1.set_yticks([0, -1, -2, -3, -4])
        ax1.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3'])
        ax1.set_ylabel('Sleep Stage', fontsize=11, fontweight='bold')
        ax1.set_title(f'Ground Truth - {patient_id} ({n_epochs} epochs, {total_hours:.1f}h)', 
                     fontsize=12, fontweight='bold')
        ax1.grid(True, axis='x', alpha=0.3)
        
        # Predicted
        for i in range(len(times)):
            color = stage_colors[pred_stages[i]]
            rect = plt.Rectangle(
                (times[i], pred_stages[i] - 0.4),
                epoch_duration,
                0.8,
                facecolor=color,
                edgecolor='none',
                alpha=0.9
            )
            ax2.add_patch(rect)
        
        ax2.set_xlim(0, max(max_time, 8))  # En az 8 saat göster
        ax2.set_ylim(-4.5, 0.5)
        ax2.set_yticks([0, -1, -2, -3, -4])
        ax2.set_yticklabels(['Wake', 'REM', 'N1', 'N2', 'N3'])
        ax2.set_ylabel('Sleep Stage', fontsize=11, fontweight='bold')
        ax2.set_xlabel('Time (hours)', fontsize=11, fontweight='bold')
        ax2.set_title(f'Predicted - {patient_id}', fontsize=12, fontweight='bold')
        ax2.grid(True, axis='x', alpha=0.3)
        
        # Genel başlık
        fig.suptitle(f'Hypnogram Comparison - {patient_id}', 
                    fontsize=14, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        
        # Kaydet (run directory'ye)
        if self.run_dir:
            hypnogram_dir = os.path.join(self.run_dir, 'hypnograms')
            os.makedirs(hypnogram_dir, exist_ok=True)
            save_path = os.path.join(hypnogram_dir, f'{patient_id}_comparison.png')
        else:
            save_path = os.path.join(config.HYPNOGRAM_DIR, f'{patient_id}_comparison.png')
        
        plt.savefig(save_path, dpi=config.HYPNOGRAM_DPI, bbox_inches='tight')
        print(f"  ✓ {patient_id} hipnogram kaydedildi: {fig_width}x{fig_height} inch")
        
        plt.close()
    
    def save_results(self, metrics, save_path=None):
        """
        Sonuçları JSON olarak kaydet
        """
        import json
        
        if save_path is None:
            save_path = os.path.join(config.LOG_DIR, 'test_results.json')
        
        results = {
            'metrics': metrics,
            'class_names': config.CLASS_NAMES,
            'predictions': [int(p) for p in self.all_predictions],
            'labels': [int(l) for l in self.all_labels]
        }
        
        with open(save_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✓ Test sonuçları kaydedildi: {save_path}")


def evaluate_single_run(run_dir, all_patient_ids, shared_signal_cache=None):
    """
    Tek bir run (veya fold) için evaluation yap.
    Transformer/DeepSleepNet sequence mode otomatik desteklenir.
    
    Args:
        run_dir: Run veya fold dizini
        all_patient_ids: Tüm hasta ID'leri
        shared_signal_cache: K-Fold modunda paylaşılan sinyal cache'i.
                            None ise sadece test verisi yüklenir.
    """
    # Sequence mode kontrolü
    sequence_models = ['transformer', 'deepsleepnet']
    sequence_mode = config.MODEL_TYPE in sequence_models
    
    # Patient split (run directory'den yükle)
    split_file = os.path.join(run_dir, 'patient_split.json')
    
    if not os.path.exists(split_file):
        print(f"✗ Split dosyası bulunamadı: {split_file}")
        print(f"Lütfen doğru run directory'yi belirtin.")
        return None
    
    import json
    with open(split_file, 'r') as f:
        split_data = json.load(f)
    
    train_ids = split_data['train']
    val_ids = split_data['val']
    test_ids = split_data['test']
    
    print(f"\n{'='*70}")
    print(f"SPLIT YÜKLENDİ (RUN DIRECTORY'DEN)")
    print(f"{'='*70}")
    print(f"Train: {len(train_ids)} hasta (sadece bilgi - yüklenmeyecek)")
    print(f"Val:   {len(val_ids)} hasta (sadece bilgi - yüklenmeyecek)")
    print(f"Test:  {len(test_ids)} hasta (değerlendirme için yüklenecek)")
    print(f"Mode:  {'SEQUENCE' if sequence_mode else 'SINGLE EPOCH'}")
    print(f"{'='*70}\n")
    
    # Test DataLoader oluştur
    if sequence_mode:
        # Sequence mode: SleepSequenceDataset kullan
        print(f"  Sequence DataLoader oluşturuluyor (seq_len={config.SEQUENCE_LENGTH})...")
        test_dataset = SleepSequenceDataset(
            patient_ids=test_ids,
            sequence_length=config.SEQUENCE_LENGTH,
            stride=config.SEQUENCE_LENGTH,  # Test'te non-overlapping (her epoch 1 kez)
            preload=True,
            augment=False
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        print(f"  ✓ Test: {len(test_loader)} batch ({len(test_dataset)} sequence)")
    elif shared_signal_cache is not None:
        # K-Fold: Shared cache'den test DataLoader oluştur
        test_loader = create_test_dataloader_from_cache(
            test_ids, shared_signal_cache, preload=True
        )
    else:
        # Normal run: Sadece test verisi yükle
        test_loader = create_test_dataloader(test_ids, preload=True)
    
    # Model yükle
    print(f"\n{'='*70}")
    print(f"MODEL YÜKLENİYOR")
    print(f"{'='*70}\n")
    
    model = get_model(config.MODEL_TYPE)
    
    # Best checkpoint'i yükle (run directory'den)
    checkpoint_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')
    
    if not os.path.exists(checkpoint_path):
        print(f"✗ Checkpoint bulunamadı: {checkpoint_path}")
        print("Önce train.py çalıştırın!")
        return None
    
    checkpoint = torch.load(checkpoint_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"✓ Checkpoint yüklendi: {checkpoint_path}")
    print(f"  Epoch: {checkpoint['epoch']}")
    print(f"  Best val loss: {checkpoint['best_val_loss']:.4f}")
    print(f"  Best val acc: {checkpoint['best_val_acc']:.2f}%")
    
    
    # Evaluator (run_dir ile, sequence mode destekli)
    evaluator = Evaluator(model, test_loader, config.DEVICE, 
                          run_dir=run_dir, sequence_mode=sequence_mode)
    
    # 1. Temel değerlendirme
    accuracy = evaluator.evaluate()
    
    # 2. Detaylı metrikler
    metrics = evaluator.calculate_metrics()
    
    # 3. Confusion matrix (run directory'ye kaydet)
    cm_path = os.path.join(run_dir, 'confusion_matrix.png')
    cm_raw_path = os.path.join(run_dir, 'confusion_matrix_raw.png')
    evaluator.plot_confusion_matrix(normalize=True, save_path=cm_path)
    evaluator.plot_confusion_matrix(normalize=False, save_path=cm_raw_path)
    
    # 4. Per-class metrics grafiği (run directory'ye kaydet)
    metrics_plot_path = os.path.join(run_dir, 'per_class_metrics.png')
    evaluator.plot_per_class_metrics(metrics, save_path=metrics_plot_path)
    
    # 5. Hipnogramlar
    evaluator.generate_hypnograms_for_test_patients()
    
    # 6. Sonuçları kaydet (run directory'ye)
    results_path = os.path.join(run_dir, 'test_results.json')
    evaluator.save_results(metrics, save_path=results_path)
    
    # Tüm figürleri kapat (fold'lar arası bellek sızıntısı önlemi)
    plt.close('all')
    
    # RAM temizle
    del evaluator, model, test_loader
    import gc
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    return {
        'accuracy': accuracy,
        'metrics': metrics
    }


def main():
    """
    Ana evaluation fonksiyonu - normal run ve kfold run destekler.
    """
    import argparse
    
    # Command-line arguments
    parser = argparse.ArgumentParser(description='Sleep Stage Classification - Evaluation')
    parser.add_argument('--run_dir', type=str, required=True,
                       help='Training run directory (e.g., outputs/runs/251220-1730)')
    args = parser.parse_args()
    
    run_dir = args.run_dir
    
    # Run directory kontrolü
    if not os.path.exists(run_dir):
        print(f"✗ Run directory bulunamadı: {run_dir}")
        print(f"Lütfen geçerli bir run directory belirtin.")
        return
    
    print(f"\n{'='*70}")
    print(f"EVALUATION - RUN: {os.path.basename(run_dir)}")
    print(f"{'='*70}")
    print(f"Run directory: {run_dir}")
    print(f"{'='*70}\n")
    
    # Config
    config.set_seed()
    config.print_config()
    
    # Hasta listesini al (offline veya MongoDB'den)
    if config.USE_OFFLINE_DATA:
        print("Offline modda hasta listesi alınıyor...")
        all_patient_ids = get_offline_patient_ids()
    else:
        from pymongo import MongoClient
        import gridfs
        print("MongoDB'den hasta listesi alınıyor...")
        client = MongoClient(config.MONGO_URI)
        db = client[config.DB_NAME]
        fs = gridfs.GridFS(db)
        all_patient_ids = set()
        for file in fs.find({"metadata.data_type": "sleep_stages"}):
            all_patient_ids.add(file.metadata['patient_id'])
        all_patient_ids = sorted(all_patient_ids)
        client.close()
    
    print(f"✓ Toplam {len(all_patient_ids)} hasta bulundu\n")
    
    # Stage verilerini cache'e yükle
    print("Stage verileri cache'e yükleniyor...")
    preload_all_stages(all_patient_ids, config.MONGO_URI, config.DB_NAME)
    print(f"✓ {len(_STAGES_CACHE)} hasta cache'de\n")
    
    # K-Fold mu normal run mı kontrol et
    kfold_summary = os.path.join(run_dir, 'kfold_summary.json')
    is_kfold = os.path.exists(kfold_summary)
    
    if is_kfold:
        # ============================================================
        # K-FOLD EVALUATION
        # ============================================================
        import json
        with open(kfold_summary, 'r') as f:
            summary = json.load(f)
        
        n_folds = summary['n_folds']
        print(f"\n{'='*70}")
        print(f"K-FOLD EVALUATION MODU ({n_folds} fold)")
        print(f"{'='*70}\n")
        
        # Tüm fold'larda kullanılacak hastaları belirle
        all_fold_test_ids = set()
        for fold_idx in range(n_folds):
            fold_dir = os.path.join(run_dir, f'fold_{fold_idx}')
            split_file = os.path.join(fold_dir, 'patient_split.json')
            if os.path.exists(split_file):
                with open(split_file, 'r') as f:
                    split_data = json.load(f)
                all_fold_test_ids.update(split_data.get('test', []))
        
        # Tüm sinyal verilerini bir kere yükle (tüm fold'lar paylaşacak)
        print(f"\n{'='*70}")
        print(f"TÜM FOLD'LAR İÇİN SİNYAL VERİLERİ TEK SEFERDE YÜKLENİYOR")
        print(f"Toplam benzersiz test hastası: {len(all_fold_test_ids)}")
        print(f"{'='*70}\n")
        
        shared_signal_cache = preload_all_signals_to_cache(sorted(all_fold_test_ids))
        
        fold_accuracies = []
        
        # Her fold için ayrı evaluation (veri tekrar yüklenmez!)
        for fold_idx in range(n_folds):
            fold_dir = os.path.join(run_dir, f'fold_{fold_idx}')
            
            if not os.path.exists(fold_dir):
                print(f"⚠ Fold {fold_idx} dizini bulunamadı, atlanıyor...")
                continue
            
            print(f"\n{'#'*70}")
            print(f"# FOLD {fold_idx}/{n_folds - 1} EVALUATION")
            print(f"{'#'*70}\n")
            
            result = evaluate_single_run(fold_dir, all_patient_ids, 
                                         shared_signal_cache=shared_signal_cache)
            
            if result:
                fold_accuracies.append(result['accuracy'])
                print(f"\n✓ Fold {fold_idx} Test Accuracy: {result['accuracy']:.2f}%")
        
        # K-Fold özet
        if fold_accuracies:
            import numpy as np
            mean_acc = np.mean(fold_accuracies)
            std_acc = np.std(fold_accuracies)
            
            print(f"\n{'='*70}")
            print(f"K-FOLD EVALUATION TAMAMLANDI!")
            print(f"{'='*70}")
            print(f"\n📊 TEST SONUÇLARI ({len(fold_accuracies)} fold):")
            for i, acc in enumerate(fold_accuracies):
                print(f"  Fold {i}: Test Acc = {acc:.2f}%")
            print(f"\n  Ortalama Test Acc: {mean_acc:.2f}% ± {std_acc:.2f}%")
            print(f"{'='*70}\n")
            
            # Özet güncelle
            summary['test_mean_acc'] = float(mean_acc)
            summary['test_std_acc'] = float(std_acc)
            summary['test_fold_accuracies'] = fold_accuracies
            
            with open(kfold_summary, 'w') as f:
                json.dump(summary, f, indent=2)
            print(f"✓ kfold_summary.json güncellendi (test sonuçları eklendi)")
        
        # Shared cache'i temizle (bellek boşalt)
        del shared_signal_cache
        import gc
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        print("✓ Shared signal cache temizlendi")
    
    else:
        # ============================================================
        # NORMAL EVALUATION
        # ============================================================
        result = evaluate_single_run(run_dir, all_patient_ids)
        
        if result:
            print(f"\n{'='*70}")
            print(f"DEĞERLENDİRME TAMAMLANDI!")
            print(f"{'='*70}")
            print(f"Tüm sonuçlar '{run_dir}' klasöründe")
            print(f"{'='*70}\n")


if __name__ == "__main__":
    main()