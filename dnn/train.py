"""
Sleep Stage Classification - Training Script
Model eğitimi, validation, checkpoint kaydetme
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR
from torch.cuda.amp import autocast, GradScaler  # Mixed Precision
import numpy as np
import time
import os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Backend for non-GUI environments

import config
from dataset import split_patients, create_dataloaders, SleepEpochDataset
from dataset import SleepSequenceDataset, create_sequence_dataloaders  # For DeepSleepNet
from models.cnn1d import get_model
from edf_to_mongo import preload_all_stages, _STAGES_CACHE
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import json


def export_model_for_netron(model, save_path, input_shape):
    """
    Model'i Netron için export et (ONNX formatında).
    
    Args:
        model: PyTorch model
        save_path: Kaydedilecek dosya yolu (.onnx)
        input_shape: Input shape (batch_size dahil)
    """
    import torch.onnx
    
    print(f"\n{'='*70}")
    print(f"MODEL NETRON İÇİN EXPORT EDİLİYOR")
    print(f"{'='*70}")
    
    # Model'i eval mode'a al
    model.eval()
    
    # Dummy input oluştur
    dummy_input = torch.randn(*input_shape).to(config.DEVICE)
    
    print(f"Input shape: {input_shape}")
    print(f"Export path: {save_path}")
    
    # ONNX'e export et
    torch.onnx.export(
        model,
        dummy_input,
        save_path,
        export_params=True,
        opset_version=18,  # 11 → 18 (PyTorch'un güncel versiyonu)
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Model export edildi!")
    print(f"\nNetron ile görselleştirmek için:")
    print(f"  1. https://netron.app/ adresine git")
    print(f"  2. '{save_path}' dosyasını yükle")
    print(f"  VEYA")
    print(f"  1. pip install netron")
    print(f"  2. netron {save_path}")
    print(f"{'='*70}\n")


def plot_training_history(history, save_path=None):
    """
    Training history'yi görselleştir (accuracy ve loss grafikleri).
    Overfitting'i tespit etmek için train vs validation metriklerini gösterir.
    
    Args:
        history: Dictionary with keys: 'train_loss', 'train_acc', 'val_loss', 'val_acc', 'lr'
        save_path: Kaydedilecek dosya yolu (.png). None ise otomatik oluşturulur.
    """
    if save_path is None:
        save_path = os.path.join(config.OUTPUT_DIR, 'training_history.png')
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Figure oluştur (2 subplot: loss ve accuracy)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss grafiği
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss', linewidth=2, markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2, markersize=4)
    ax1.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss', fontsize=12, fontweight='bold')
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # Overfitting bölgesini vurgula (val_loss artarken train_loss azalıyorsa)
    if len(history['val_loss']) > 5:
        min_val_loss_idx = np.argmin(history['val_loss'])
        if min_val_loss_idx < len(epochs) - 1:
            ax1.axvline(x=min_val_loss_idx + 1, color='green', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Best Val Loss (Epoch {min_val_loss_idx + 1})')
            ax1.legend(fontsize=10)
    
    # Accuracy grafiği
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Accuracy', linewidth=2, markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Validation Accuracy', linewidth=2, markersize=4)
    ax2.set_xlabel('Epoch', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax2.set_title('Training vs Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # Overfitting bölgesini vurgula
    if len(history['val_acc']) > 5:
        max_val_acc_idx = np.argmax(history['val_acc'])
        if max_val_acc_idx < len(epochs) - 1:
            ax2.axvline(x=max_val_acc_idx + 1, color='green', linestyle='--', 
                       linewidth=2, alpha=0.7, label=f'Best Val Acc (Epoch {max_val_acc_idx + 1})')
            ax2.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\n{'='*70}")
    print(f"TRAINING HISTORY GRAFİĞİ KAYDEDİLDİ")
    print(f"{'='*70}")
    print(f"Dosya: {save_path}")
    print(f"{'='*70}\n")
    
    # Overfitting analizi
    if len(history['train_acc']) > 5:
        final_train_acc = history['train_acc'][-1]
        final_val_acc = history['val_acc'][-1]
        max_val_acc = max(history['val_acc'])
        
        print(f"📊 OVERFITTING ANALİZİ:")
        print(f"  Final Train Accuracy: {final_train_acc:.2f}%")
        print(f"  Final Val Accuracy: {final_val_acc:.2f}%")
        print(f"  Best Val Accuracy: {max_val_acc:.2f}%")
        print(f"  Train-Val Gap: {final_train_acc - final_val_acc:.2f}%")
        
        if final_train_acc - final_val_acc > 10:
            print(f"\n  ⚠️  OVERFITTING TESPİT EDİLDİ!")
            print(f"  Öneriler:")
            print(f"    - Daha fazla veri ekleyin")
            print(f"    - Dropout oranını artırın (şu an: {config.CNN1D_CONFIG['dropout']})")
            print(f"    - Weight decay artırın (şu an: {config.WEIGHT_DECAY})")
            print(f"    - Data augmentation kullanın")
            print(f"    - Early stopping patience azaltın")
        else:
            print(f"\n  ✓ Model dengeli görünüyor")
        
        print(f"{'='*70}\n")


class Trainer:
    """
    Model eğitim sınıfı.
    
    MODEL_TYPE'a göre farklı training stratejileri:
    - cnn1d, cnn1d_paper: Standard DataLoader training
    - deepsleepnet: Stateful LSTM training (hasta-bazlı iterate)
    """
    
    def __init__(self, model, train_loader, val_loader, 
                 criterion, optimizer, scheduler=None, device=config.DEVICE, run_dir=None,
                 model_type='cnn1d', use_mixed_precision=False, gradient_accumulation_steps=1,
                 use_gradient_clipping=False, gradient_clip_value=1.0,
                 train_dataset=None, val_dataset=None):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.train_dataset = train_dataset  # Stateful training için (deepsleepnet)
        self.val_dataset = val_dataset      # Stateful validation için (deepsleepnet)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.run_dir = run_dir if run_dir else config.CHECKPOINT_DIR
        
        # Model type: 'cnn1d', 'cnn1d_paper', 'deepsleepnet'
        self.model_type = model_type
        
        # Mixed precision (for 4GB VRAM)
        self.use_mixed_precision = use_mixed_precision
        self.scaler = GradScaler() if use_mixed_precision else None
        
        # Gradient accumulation (effective batch size = batch_size * gradient_accumulation_steps)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Gradient clipping (important for LSTM to prevent exploding gradients)
        self.use_gradient_clipping = use_gradient_clipping
        self.gradient_clip_value = gradient_clip_value
        
        # Training history
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        
        # Best model tracking
        self.best_val_loss = float('inf')
        self.best_val_acc = 0.0
        self.epochs_without_improvement = 0
        
        # Timing
        self.start_time = None
        
        # Print training mode info
        print(f"\n{'='*70}")
        print(f"TRAINER INITIALIZED")
        print(f"{'='*70}")
        print(f"  Model type: {model_type}")
        print(f"  Training mode: {'Stateful LSTM (patient-wise)' if model_type == 'deepsleepnet' else 'Standard DataLoader'}")
        print(f"  Mixed precision (FP16): {use_mixed_precision}")
        print(f"  Gradient accumulation: {gradient_accumulation_steps} steps")
        print(f"  Gradient clipping: {use_gradient_clipping} (max_norm={gradient_clip_value})")
        print(f"  Effective batch size: {config.BATCH_SIZE * gradient_accumulation_steps}")
        print(f"{'='*70}\n")
        
    def train_epoch(self, epoch):
        """
        Bir epoch eğitim.
        
        Sequence mode (DeepSleepNet + BiLSTM):
        - Her hasta sıralı işlenir, LSTM state hasta boyunca taşınır
        - Yeni hastada state reset edilir
        
        Standard mode (CNN):
        - Normal DataLoader ile batch training
        """
        self.model.train()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # ============================================================
        # CNN MODELS (cnn1d, cnn1d_paper) - Standard DataLoader training
        # ============================================================
        if self.model_type in ['cnn1d', 'cnn1d_paper']:
            for batch_idx, (signals, labels) in enumerate(self.train_loader):
                signals = signals.to(self.device)
                labels = labels.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(signals)
                        loss = self.criterion(outputs, labels)
                    
                    self.scaler.scale(loss).backward()
                    
                    if self.use_gradient_clipping:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(signals)
                    loss = self.criterion(outputs, labels)
                    
                    loss.backward()
                    
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Log
                if (batch_idx + 1) % config.LOG_INTERVAL == 0:
                    batch_loss = running_loss / (batch_idx + 1)
                    batch_acc = 100. * correct / total
                    print(f"  Batch [{batch_idx+1}/{len(self.train_loader)}] | "
                          f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%")
            
            # Epoch istatistikleri
            epoch_loss = running_loss / len(self.train_loader) if len(self.train_loader) > 0 else 0
            epoch_acc = 100. * correct / total if total > 0 else 0
            
            return epoch_loss, epoch_acc
        
        # ============================================================
        # DEEPSLEEPNET - Stateful LSTM training (patient-wise iterate)
        # ============================================================
        elif self.model_type == 'deepsleepnet':
            n_sequences = 0
            current_patient = None
            
            # Dataset'ten hasta-bazlı iterate et
            if self.train_dataset is None:
                raise ValueError("train_dataset must be provided for stateful training")
            
            for batch_data in self.train_dataset.iterate_by_patient():
                patient_id = batch_data['patient_id']
                is_new_patient = batch_data['is_new_patient']
                signals = batch_data['signals'].to(self.device)  # [1, seq_len, n_ch, samples]
                labels = batch_data['labels'].to(self.device)     # [1, seq_len]
                
                # Yeni hasta - LSTM state reset
                if is_new_patient:
                    self.model.bilstm.reset_state(batch_size=1, device=self.device)
                    if current_patient is not None:
                        pass  # Progress log için kullanılabilir
                    current_patient = patient_id
                
                # Flatten labels for loss
                labels_flat = labels.view(-1)  # [seq_len]
                
                # Forward pass
                self.optimizer.zero_grad()
                
                if self.use_mixed_precision:
                    with autocast():
                        outputs = self.model(signals)  # [1, seq_len, n_classes]
                        outputs_flat = outputs.view(-1, outputs.size(-1))  # [seq_len, n_classes]
                        loss = self.criterion(outputs_flat, labels_flat)
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    outputs = self.model(signals)
                    outputs_flat = outputs.view(-1, outputs.size(-1))
                    loss = self.criterion(outputs_flat, labels_flat)
                    
                    loss.backward()
                    
                    if self.use_gradient_clipping:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.gradient_clip_value)
                    
                    self.optimizer.step()
                
                # Statistics
                running_loss += loss.item()
                n_sequences += 1
                
                _, predicted = torch.max(outputs_flat.data, 1)
                total += labels_flat.size(0)
                correct += (predicted == labels_flat).sum().item()
                
                # Log
                if n_sequences % config.LOG_INTERVAL == 0:
                    batch_loss = running_loss / n_sequences
                    batch_acc = 100. * correct / total
                    print(f"  Seq [{n_sequences}] Patient: {patient_id} | "
                          f"Loss: {batch_loss:.4f} | Acc: {batch_acc:.2f}%")
            
            # Epoch istatistikleri
            epoch_loss = running_loss / n_sequences if n_sequences > 0 else 0
            epoch_acc = 100. * correct / total if total > 0 else 0
            
            return epoch_loss, epoch_acc
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def validate(self):
        """
        Validation - model tipine göre farklı loop.
        """
        self.model.eval()
        
        running_loss = 0.0
        correct = 0
        total = 0
        
        # ============================================================
        # CNN MODELS (cnn1d, cnn1d_paper) - Standard DataLoader validation
        # ============================================================
        if self.model_type in ['cnn1d', 'cnn1d_paper']:
            with torch.no_grad():
                for signals, labels in self.val_loader:
                    signals = signals.to(self.device)
                    labels = labels.to(self.device)
                    
                    if self.use_mixed_precision:
                        with autocast():
                            outputs = self.model(signals)
                            loss = self.criterion(outputs, labels)
                    else:
                        outputs = self.model(signals)
                        loss = self.criterion(outputs, labels)
                    
                    running_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()
            
            val_loss = running_loss / len(self.val_loader) if len(self.val_loader) > 0 else 0
            val_acc = 100. * correct / total if total > 0 else 0
            
            return val_loss, val_acc
        
        # ============================================================
        # DEEPSLEEPNET - Stateful LSTM validation (patient-wise iterate)
        # ============================================================
        elif self.model_type == 'deepsleepnet':
            n_sequences = 0
            
            if self.val_dataset is None:
                raise ValueError("val_dataset must be provided for stateful validation")
            
            with torch.no_grad():
                for batch_data in self.val_dataset.iterate_by_patient():
                    is_new_patient = batch_data['is_new_patient']
                    signals = batch_data['signals'].to(self.device)
                    labels = batch_data['labels'].to(self.device)
                    
                    # Yeni hasta - LSTM state reset
                    if is_new_patient:
                        self.model.bilstm.reset_state(batch_size=1, device=self.device)
                    
                    labels_flat = labels.view(-1)
                    
                    if self.use_mixed_precision:
                        with autocast():
                            outputs = self.model(signals)
                            outputs_flat = outputs.view(-1, outputs.size(-1))
                            loss = self.criterion(outputs_flat, labels_flat)
                    else:
                        outputs = self.model(signals)
                        outputs_flat = outputs.view(-1, outputs.size(-1))
                        loss = self.criterion(outputs_flat, labels_flat)
                    
                    running_loss += loss.item()
                    n_sequences += 1
                    
                    _, predicted = torch.max(outputs_flat.data, 1)
                    total += labels_flat.size(0)
                    correct += (predicted == labels_flat).sum().item()
            
            val_loss = running_loss / n_sequences if n_sequences > 0 else 0
            val_acc = 100. * correct / total if total > 0 else 0
            
            return val_loss, val_acc
        
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")
    
    def train(self, n_epochs):
        """
        Tam training loop
        """
        print(f"\n{'='*70}")
        print(f"TRAINING BAŞLIYOR")
        print(f"{'='*70}")
        print(f"Epochs: {n_epochs}")
        print(f"Batch size: {config.BATCH_SIZE}")
        print(f"Learning rate: {config.LEARNING_RATE}")
        if config.USE_WARMUP:
            print(f"Warmup: {config.WARMUP_EPOCHS} epochs (LR: {config.WARMUP_START_LR:.1e} → {config.LEARNING_RATE:.1e})")
        print(f"Device: {self.device}")
        print(f"{'='*70}\n")
        
        self.start_time = time.time()
        
        for epoch in range(1, n_epochs + 1):
            epoch_start = time.time()
            
            # === WARMUP: İlk N epoch'ta LR'yi kademeli artır ===
            if config.USE_WARMUP and epoch <= config.WARMUP_EPOCHS:
                # Linear warmup: start_lr → target_lr
                warmup_progress = epoch / config.WARMUP_EPOCHS
                warmup_lr = config.WARMUP_START_LR + (config.LEARNING_RATE - config.WARMUP_START_LR) * warmup_progress
                
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = warmup_lr
                
                print(f"\n🔥 Warmup [{epoch}/{config.WARMUP_EPOCHS}] - LR: {warmup_lr:.6f}")
            
            print(f"\nEpoch [{epoch}/{n_epochs}]")
            print(f"-" * 70)
            
            # Training
            train_loss, train_acc = self.train_epoch(epoch)
            
            # Validation
            val_loss, val_acc = self.validate()
            
            # Learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # History'ye ekle
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['lr'].append(current_lr)
            
            # Epoch süresi
            epoch_time = time.time() - epoch_start
            
            # Özet yazdır
            print(f"\n{'='*70}")
            print(f"Epoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
            print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
            print(f"  Learning Rate: {current_lr:.6f}")
            print(f"  Time: {epoch_time:.1f}s")
            
            # Learning rate scheduler (warmup bittikten sonra aktif)
            if self.scheduler is not None:
                # Warmup döneminde scheduler'ı çalıştırma
                if config.USE_WARMUP and epoch <= config.WARMUP_EPOCHS:
                    pass  # Warmup döneminde scheduler pasif
                else:
                    if config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
                        self.scheduler.step(val_loss)
                    else:
                        self.scheduler.step()
            
            
            # Best metrics'leri her zaman güncelle (hangi metrik kullanılırsa kullanılsın)
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
            
            # Best model kaydet (seçilen metriğe göre)
            is_best = False
            if config.CHECKPOINT_METRIC == 'val_loss':
                if val_loss == self.best_val_loss:  # Bu epoch'ta best loss elde edildi
                    is_best = True
            elif config.CHECKPOINT_METRIC == 'val_accuracy':
                if val_acc == self.best_val_acc:  # Bu epoch'ta best acc elde edildi
                    is_best = True
            
            if is_best:
                print(f"  ✓ Best model! (Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%)")
                self.save_checkpoint(epoch, is_best=True)
                self.epochs_without_improvement = 0
            else:
                self.epochs_without_improvement += 1
                if not config.SAVE_BEST_ONLY:
                    self.save_checkpoint(epoch, is_best=False)
            
            print(f"{'='*70}")
            
            # Early stopping
            if config.USE_EARLY_STOPPING:
                if self.epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                    print(f"\n⚠ Early stopping! {config.EARLY_STOPPING_PATIENCE} "
                          f"epoch boyunca iyileşme yok.")
                    break
        
        # Training tamamlandı
        total_time = time.time() - self.start_time
        
        print(f"\n{'='*70}")
        print(f"TRAINING TAMAMLANDI!")
        print(f"{'='*70}")
        print(f"Toplam süre: {total_time/60:.1f} dakika")
        print(f"Best val loss: {self.best_val_loss:.4f}")
        print(f"Best val acc: {self.best_val_acc:.2f}%")
        print(f"{'='*70}\n")
    
    def save_checkpoint(self, epoch, is_best=False):
        """
        Model checkpoint'i kaydet
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_acc': self.best_val_acc,
            'config': {
                'model_type': config.MODEL_TYPE,
                'channels': config.SELECTED_CHANNELS,
                'n_classes': config.N_CLASSES,
                'class_names': config.CLASS_NAMES
            }
        }
        
        if is_best:
            filename = 'best_model.pth'
        else:
            filename = f'checkpoint_epoch_{epoch}.pth'
        
        filepath = os.path.join(self.run_dir, 'checkpoints', filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filepath):
        """
        Checkpoint'ten model yükle
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint['history']
        self.best_val_loss = checkpoint['best_val_loss']
        self.best_val_acc = checkpoint['best_val_acc']
        
        print(f"✓ Checkpoint yüklendi: {filepath}")
        print(f"  Epoch: {checkpoint['epoch']}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"  Best val acc: {self.best_val_acc:.2f}%")
        
        return checkpoint['epoch']


def run_normal_training():
    """
    Normal training modu (mevcut train/val/test split)
    """
    # Timestamped run directory oluştur
    from datetime import datetime
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    run_dir = os.path.join(config.OUTPUT_DIR, 'runs', timestamp)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(os.path.join(run_dir, 'checkpoints'), exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"RUN DIRECTORY OLUŞTURULDU")
    print(f"{'='*70}")
    print(f"Run ID: {timestamp}")
    print(f"Directory: {run_dir}")
    print(f"{'='*70}\n")
    
    # Config
    config.set_seed()
    config.create_directories()
    config.print_config()
    
    # MongoDB'den hasta listesini al
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
    
    # Stage verilerini cache'e yükle (performans için)
    print("Stage verileri cache'e yükleniyor...")
    preload_all_stages(all_patient_ids, config.MONGO_URI, config.DB_NAME)
    print(f"✓ {len(_STAGES_CACHE)} hasta cache'de\n")
    
    # Patient split (run-specific)
    train_ids, val_ids, test_ids = split_patients(
        all_patient_ids,
        config.N_TRAIN_PATIENTS,
        config.N_VAL_PATIENTS,
        config.N_TEST_PATIENTS,
        run_dir=run_dir
    )
    
    # DataLoader'lar - Model tipine göre seç
    use_sequence_mode = (config.MODEL_TYPE == 'deepsleepnet' and 
                         hasattr(config, 'USE_SEQUENCE_TRAINING') and 
                         config.USE_SEQUENCE_TRAINING)
    
    if use_sequence_mode:
        print(f"\n{'='*70}")
        print(f"SEQUENCE MODE AKTİF (DeepSleepNet + BiLSTM)")
        print(f"{'='*70}")
        print(f"Sequence length: {config.SEQUENCE_LENGTH}")
        print(f"{'='*70}\n")
        
        train_loader, val_loader, test_loader = create_sequence_dataloaders(
            train_ids, val_ids, test_ids, 
            sequence_length=config.SEQUENCE_LENGTH,
            preload=True
        )
    else:
        # Standard single-epoch training
        train_loader, val_loader, test_loader = create_dataloaders(
            train_ids, val_ids, test_ids, preload=True  # RAM Cache aktif
        )
    
    # Model
    model = get_model(config.MODEL_TYPE)
    
    # Model'i Netron için export et (opsiyonel)
    # Not: DeepSleepNet için sequence input gerekli
    if config.EXPORT_TO_ONNX:
        onnx_path = os.path.join(run_dir, 'model_architecture.onnx')
        if use_sequence_mode:
            # Sequence mode: [batch, seq_len, channels, samples]
            n_channels = config.N_CHANNELS_ACTIVE if hasattr(config, 'N_CHANNELS_ACTIVE') else config.N_CHANNELS
            input_shape = (1, config.SEQUENCE_LENGTH, n_channels, config.SAMPLES_PER_EPOCH)
        else:
            input_shape = (1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
        
        export_model_for_netron(
            model=model,
            save_path=onnx_path,
            input_shape=input_shape
        )
    
    # Loss function
    if config.USE_CLASS_WEIGHTS:
        # Class weights kullan (dengesiz veri için)
        print("\n" + "="*70)
        print("CLASS WEIGHTS HESAPLANIYOR")
        print("="*70)
        
        train_dataset = train_loader.dataset
        class_weights = train_dataset.get_class_weights().to(config.DEVICE)
        
        print("Class weights:")
        for i, class_name in enumerate(config.CLASS_NAMES):
            print(f"  {class_name}: {class_weights[i]:.4f}")
        print("="*70 + "\n")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:
        # Balanced dataset kullanıldığında class weights'e gerek yok
        print("\n" + "="*70)
        print("BALANCED DATASET KULLANILIYOR - CLASS WEIGHTS YOK")
        print("="*70 + "\n")
        
        criterion = nn.CrossEntropyLoss()
    
    # Optimizer
    if config.OPTIMIZER == 'Adam':
        optimizer = optim.Adam(model.parameters(), 
                              lr=config.LEARNING_RATE,
                              weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'SGD':
        optimizer = optim.SGD(model.parameters(),
                             lr=config.LEARNING_RATE,
                             momentum=0.9,
                             weight_decay=config.WEIGHT_DECAY)
    elif config.OPTIMIZER == 'AdamW':
        optimizer = optim.AdamW(model.parameters(),
                               lr=config.LEARNING_RATE,
                               weight_decay=config.WEIGHT_DECAY)
    else:
        raise ValueError(f"Desteklenmeyen optimizer: {config.OPTIMIZER}")
    
    # Learning rate scheduler
    scheduler = None
    if config.USE_LR_SCHEDULER:
        if config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
            scheduler = ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=config.LR_SCHEDULER_FACTOR,
                patience=config.LR_SCHEDULER_PATIENCE
            )
        elif config.LR_SCHEDULER_TYPE == 'StepLR':
            scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
    
    # Mixed precision and gradient accumulation settings
    use_mixed_precision = hasattr(config, 'USE_MIXED_PRECISION') and config.USE_MIXED_PRECISION
    gradient_accumulation_steps = getattr(config, 'GRADIENT_ACCUMULATION_STEPS', 1)
    
    # Gradient clipping settings (important for LSTM)
    use_gradient_clipping = getattr(config, 'USE_GRADIENT_CLIPPING', False)
    gradient_clip_value = getattr(config, 'GRADIENT_CLIP_VALUE', 1.0)
    
    # Dataset referanslarını al (stateful training için)
    train_dataset = train_loader.dataset
    val_dataset = val_loader.dataset
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=config.DEVICE,
        run_dir=run_dir,
        model_type=config.MODEL_TYPE,
        use_mixed_precision=use_mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
        use_gradient_clipping=use_gradient_clipping,
        gradient_clip_value=gradient_clip_value,
        train_dataset=train_dataset,
        val_dataset=val_dataset
    )
    
    # Training başlat
    trainer.train(config.N_EPOCHS)
    
    # Training history grafiğini çiz (run directory'ye kaydet)
    plot_path = os.path.join(run_dir, 'training_history.png')
    plot_training_history(trainer.history, save_path=plot_path)
    
    # Eğitilmiş model'i de export et
    if config.ONNX_EXPORT_AFTER_TRAINING:
        trained_onnx_path = os.path.join(run_dir, 'trained_model.onnx')
        export_model_for_netron(
            model=model,
            save_path=trained_onnx_path,
            input_shape=(1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
        )
    
    # Training history'yi kaydet (run directory'ye)
    import json
    history_path = os.path.join(run_dir, 'training_history.json')
    with open(history_path, 'w') as f:
        json.dump(trainer.history, f, indent=2)
    print(f"✓ Training history kaydedildi: {history_path}")


def run_kfold_training():
    """
    K-Fold Cross Validation training modu.
    5 hasta test için ayrılır, kalan hastalar üzerinde K-Fold CV yapılır.
    """
    from datetime import datetime
    from pymongo import MongoClient
    import gridfs
    
    # Timestamped run directory oluştur
    timestamp = datetime.now().strftime("%y%m%d-%H%M")
    run_dir = os.path.join(config.OUTPUT_DIR, 'runs', f'{timestamp}_kfold')
    os.makedirs(run_dir, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS VALIDATION BAŞLIYOR")
    print(f"{'='*70}")
    print(f"Run ID: kfold_{timestamp}")
    print(f"Directory: {run_dir}")
    print(f"Fold sayısı: {config.N_FOLDS}")
    print(f"Test hasta sayısı: {config.N_TEST_PATIENTS_KFOLD}")
    print(f"{'='*70}\n")
    
    # Config
    config.set_seed()
    config.create_directories()
    config.print_config()
    
    # MongoDB'den hasta listesini al
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
    
    # Test hastalarını ayır (sabit)
    import random
    random.seed(config.RANDOM_SEED)
    shuffled_patients = list(all_patient_ids)
    random.shuffle(shuffled_patients)
    
    test_patients = shuffled_patients[:config.N_TEST_PATIENTS_KFOLD]
    cv_patients = shuffled_patients[config.N_TEST_PATIENTS_KFOLD:]
    
    # Test hastalarını kaydet
    test_patients_file = os.path.join(run_dir, 'test_patients.json')
    with open(test_patients_file, 'w') as f:
        json.dump({'test_patients': test_patients, 'cv_patients': cv_patients}, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"HASTA AYIRIMI")
    print(f"{'='*70}")
    print(f"Test hastalar ({len(test_patients)}): {test_patients}")
    print(f"CV hastalar ({len(cv_patients)}): {len(cv_patients)} hasta")
    print(f"{'='*70}\n")
    
    # K-Fold split
    kf = KFold(n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    
    # Fold sonuçlarını sakla
    fold_results = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(cv_patients)):
        print(f"\n{'#'*70}")
        print(f"# FOLD {fold_idx + 1}/{config.N_FOLDS}")
        print(f"{'#'*70}\n")
        
        # Fold directory
        fold_dir = os.path.join(run_dir, f'fold_{fold_idx}')
        os.makedirs(fold_dir, exist_ok=True)
        os.makedirs(os.path.join(fold_dir, 'checkpoints'), exist_ok=True)
        
        # Train/Val hastaları
        train_patients = [cv_patients[i] for i in train_idx]
        val_patients = [cv_patients[i] for i in val_idx]
        
        print(f"Train hastalar: {len(train_patients)}")
        print(f"Val hastalar: {len(val_patients)}")
        
        # Split'i kaydet (evaulate.py uyumluluğu için test de ekliyoruz)
        split_data = {
            'fold': fold_idx,
            'train': train_patients,
            'val': val_patients,
            'test': test_patients,  # Ana test hastalarını da ekle
            'train_patients': train_patients,
            'val_patients': val_patients
        }
        with open(os.path.join(fold_dir, 'patient_split.json'), 'w') as f:
            json.dump(split_data, f, indent=2)
        
        # Dataset'leri oluştur
        print("\nDataset'ler oluşturuluyor...")
        train_dataset = SleepEpochDataset(
            patient_ids=train_patients,
            normalize=config.NORMALIZATION,
            augment=config.USE_AUGMENTATION,
            preload=True
        )
        
        val_dataset = SleepEpochDataset(
            patient_ids=val_patients,
            normalize=config.NORMALIZATION,
            augment=False,
            preload=True,
            balance_strategy='none'
        )
        
        # DataLoader'lar
        train_loader = DataLoader(
            train_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=True,
            num_workers=0,
            pin_memory=False
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config.BATCH_SIZE,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        print(f"✓ Train: {len(train_loader)} batch, Val: {len(val_loader)} batch")
        
        # Model (her fold için yeni model)
        model = get_model(config.MODEL_TYPE)
        
        # Loss function
        criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        if config.OPTIMIZER == 'Adam':
            optimizer = optim.Adam(model.parameters(), 
                                  lr=config.LEARNING_RATE,
                                  weight_decay=config.WEIGHT_DECAY)
        elif config.OPTIMIZER == 'AdamW':
            optimizer = optim.AdamW(model.parameters(),
                                   lr=config.LEARNING_RATE,
                                   weight_decay=config.WEIGHT_DECAY)
        else:
            optimizer = optim.SGD(model.parameters(),
                                 lr=config.LEARNING_RATE,
                                 momentum=0.9,
                                 weight_decay=config.WEIGHT_DECAY)
        
        # Scheduler
        scheduler = None
        if config.USE_LR_SCHEDULER:
            if config.LR_SCHEDULER_TYPE == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(
                    optimizer,
                    mode='min',
                    factor=config.LR_SCHEDULER_FACTOR,
                    patience=config.LR_SCHEDULER_PATIENCE
                )
            elif config.LR_SCHEDULER_TYPE == 'StepLR':
                scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
        
        # Trainer
        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=config.DEVICE,
            run_dir=fold_dir
        )
        
        # Training
        trainer.train(config.N_EPOCHS)
        
        # Fold sonuçlarını kaydet
        fold_result = {
            'fold': fold_idx,
            'best_val_loss': trainer.best_val_loss,
            'best_val_acc': trainer.best_val_acc,
            'final_train_acc': trainer.history['train_acc'][-1] if trainer.history['train_acc'] else 0,
            'final_val_acc': trainer.history['val_acc'][-1] if trainer.history['val_acc'] else 0
        }
        fold_results.append(fold_result)
        
        # Fold history grafiği
        plot_path = os.path.join(fold_dir, 'training_history.png')
        plot_training_history(trainer.history, save_path=plot_path)
        
        # History kaydet
        history_path = os.path.join(fold_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(trainer.history, f, indent=2)
        
        # RAM temizle
        train_dataset.clear_cache()
        val_dataset.clear_cache()
        del model, trainer, train_loader, val_loader
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        print(f"\n✓ Fold {fold_idx + 1} tamamlandı!")
        print(f"  Best Val Acc: {fold_result['best_val_acc']:.2f}%")
        print(f"  Best Val Loss: {fold_result['best_val_loss']:.4f}")
        
        # Her fold sonrasında kfold_summary.json güncelle (incremental)
        val_accs = [r['best_val_acc'] for r in fold_results]
        val_losses = [r['best_val_loss'] for r in fold_results]
        
        interim_summary = {
            'status': 'in_progress' if fold_idx < config.N_FOLDS - 1 else 'completed',
            'completed_folds': fold_idx + 1,
            'n_folds': config.N_FOLDS,
            'n_test_patients': len(test_patients),
            'n_cv_patients': len(cv_patients),
            'mean_val_acc': float(np.mean(val_accs)),
            'std_val_acc': float(np.std(val_accs)) if len(val_accs) > 1 else 0.0,
            'mean_val_loss': float(np.mean(val_losses)),
            'std_val_loss': float(np.std(val_losses)) if len(val_losses) > 1 else 0.0,
            'fold_results': fold_results
        }
        
        summary_path = os.path.join(run_dir, 'kfold_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(interim_summary, f, indent=2)
        
        print(f"  ✓ kfold_summary.json güncellendi ({fold_idx + 1}/{config.N_FOLDS} fold)")
    
    # K-Fold özet
    print(f"\n{'='*70}")
    print(f"K-FOLD CROSS VALIDATION TAMAMLANDI!")
    print(f"{'='*70}")
    
    # İstatistikler
    val_accs = [r['best_val_acc'] for r in fold_results]
    val_losses = [r['best_val_loss'] for r in fold_results]
    
    mean_acc = np.mean(val_accs)
    std_acc = np.std(val_accs)
    mean_loss = np.mean(val_losses)
    std_loss = np.std(val_losses)
    
    print(f"\n📊 SONUÇLAR:")
    print(f"  Validation Accuracy: {mean_acc:.2f}% ± {std_acc:.2f}%")
    print(f"  Validation Loss: {mean_loss:.4f} ± {std_loss:.4f}")
    print(f"\nFold detayları:")
    for r in fold_results:
        print(f"  Fold {r['fold']}: Val Acc = {r['best_val_acc']:.2f}%, Val Loss = {r['best_val_loss']:.4f}")
    
    print(f"{'='*70}\n")
    
    # Özet kaydet
    summary = {
        'n_folds': config.N_FOLDS,
        'n_test_patients': len(test_patients),
        'n_cv_patients': len(cv_patients),
        'mean_val_acc': mean_acc,
        'std_val_acc': std_acc,
        'mean_val_loss': mean_loss,
        'std_val_loss': std_loss,
        'fold_results': fold_results
    }
    
    summary_path = os.path.join(run_dir, 'kfold_summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ K-Fold özeti kaydedildi: {summary_path}")


def main():
    """
    Ana training fonksiyonu - USE_KFOLD flag'ına göre mod seç
    """
    if config.USE_KFOLD:
        run_kfold_training()
    else:
        run_normal_training()


if __name__ == "__main__":
    main()