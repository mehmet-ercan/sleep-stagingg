"""
SleepTransformer Derin Diagnostik Analizi
==========================================
CNN feature kalitesi, transformer attention ağırlıkları, 
weight istatistikleri, prediction dağılımı ve feature separability.

Kullanım:
    python tool/transformer_diagnostics.py --fold_dir outputs/runs/260428-1450_kfold/fold_0
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import argparse
from collections import defaultdict, Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn1d import get_model
from dataset import SleepSequenceDataset
from torch.utils.data import DataLoader


def load_model_and_data(fold_dir):
    """Model ve test verisini yükle."""
    # Patient split
    with open(os.path.join(fold_dir, 'patient_split.json')) as f:
        split = json.load(f)
    
    test_ids = split['test']
    
    # Model
    model = get_model(config.MODEL_TYPE)
    ckpt_path = os.path.join(fold_dir, 'checkpoints', 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    
    print(f"✓ Model yüklendi: epoch {ckpt['epoch']}")
    print(f"  Val loss: {ckpt['best_val_loss']:.4f}, Val acc: {ckpt['best_val_acc']:.2f}%")
    
    # Test dataset (küçük bir subset yeterli)
    test_dataset = SleepSequenceDataset(
        patient_ids=test_ids,
        sequence_length=config.SEQUENCE_LENGTH,
        stride=config.SEQUENCE_LENGTH,  # non-overlapping
        preload=True,
        augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    
    return model, test_loader, test_dataset


def analyze_weights(model):
    """Model ağırlık istatistikleri — dead neurons, gradient sorunları."""
    print(f"\n{'='*70}")
    print("1. WEIGHT İSTATİSTİKLERİ")
    print(f"{'='*70}\n")
    
    for name, param in model.named_parameters():
        if param.dim() < 2:
            continue  # bias, LayerNorm scale atla
        
        data = param.data.cpu()
        mean = data.mean().item()
        std = data.std().item()
        abs_mean = data.abs().mean().item()
        max_val = data.abs().max().item()
        zero_pct = (data.abs() < 1e-7).float().mean().item() * 100
        
        # Sorun tespiti
        issues = []
        if std < 1e-5:
            issues.append("⚠ STD≈0 (dead)")
        if zero_pct > 50:
            issues.append(f"⚠ {zero_pct:.0f}% sıfır")
        if max_val > 10:
            issues.append(f"⚠ max={max_val:.1f} (büyük)")
        if std > 5:
            issues.append("⚠ yüksek varyans")
        
        issue_str = " ".join(issues) if issues else ""
        
        # Sadece sorunlu veya önemli katmanları göster
        short_name = name.replace('transformer_encoder.', 'trfm.')
        short_name = short_name.replace('classification_head.', 'head.')
        short_name = short_name.replace('cnn_encoder.', 'cnn.')
        
        if issues or any(k in name for k in ['head', 'self_attn', 'linear', 'norm']):
            print(f"  {short_name:50s} shape={str(list(param.shape)):15s} "
                  f"mean={mean:+.4f} std={std:.4f} |max|={max_val:.3f} {issue_str}")


def analyze_cnn_features(model, test_loader):
    """CNN encoder çıktılarının kalitesini analiz et."""
    print(f"\n{'='*70}")
    print("2. CNN FEATURE KALİTESİ ANALİZİ")
    print(f"{'='*70}\n")
    
    all_features = []
    all_labels = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (signals, labels) in enumerate(test_loader):
            if batch_idx >= 10:  # İlk 10 batch yeterli
                break
            signals = signals.to(config.DEVICE)
            B, S, C, T = signals.shape
            
            # CNN encoder'dan feature çıkar
            x_flat = signals.view(B * S, C, T)
            features = model.cnn_encoder(x_flat)  # [B*S, d_model]
            
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.reshape(-1).numpy())
    
    features = np.concatenate(all_features, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    
    print(f"Toplam sample: {len(features)}")
    print(f"Feature dim: {features.shape[1]}")
    
    # Genel feature istatistikleri
    print(f"\nGenel Feature İstatistikleri:")
    print(f"  Mean: {features.mean():.4f}")
    print(f"  Std:  {features.std():.4f}")
    print(f"  Min:  {features.min():.4f}")
    print(f"  Max:  {features.max():.4f}")
    
    # Dead feature oranı (her zaman 0 olan dimension'lar)
    dead_dims = (features.std(axis=0) < 1e-6).sum()
    print(f"  Dead dimensions: {dead_dims}/{features.shape[1]} ({dead_dims/features.shape[1]*100:.1f}%)")
    
    # Aktif range
    active_dims = features.shape[1] - dead_dims
    print(f"  Active dimensions: {active_dims}")
    
    # Sınıf bazlı feature analizi
    print(f"\nSınıf Bazlı Feature Ortalamaları:")
    class_names = config.CLASS_NAMES
    class_means = []
    for c in range(config.N_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            print(f"  {class_names[c]}: Yok")
            continue
        c_features = features[mask]
        c_mean = c_features.mean(axis=0)
        c_std = c_features.std(axis=0).mean()
        class_means.append(c_mean)
        print(f"  {class_names[c]:4s}: n={mask.sum():5d}  mean={c_mean.mean():.4f}  "
              f"std={c_std:.4f}  norm={np.linalg.norm(c_mean):.3f}")
    
    # Sınıflar arası cosine similarity
    if len(class_means) >= 2:
        print(f"\nSınıflar Arası Cosine Similarity (1.0 = aynı, 0.0 = farklı):")
        class_means = np.array(class_means)
        norms = np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8
        normalized = class_means / norms
        cos_sim = normalized @ normalized.T
        
        present_classes = [class_names[c] for c in range(config.N_CLASSES) if (labels == c).sum() > 0]
        
        print(f"  {'':6s}", end="")
        for cn in present_classes:
            print(f"{cn:>8s}", end="")
        print()
        for i, cn1 in enumerate(present_classes):
            print(f"  {cn1:6s}", end="")
            for j, cn2 in enumerate(present_classes):
                sim = cos_sim[i, j]
                marker = " ⚠" if i != j and sim > 0.95 else ""
                print(f"{sim:8.3f}", end="")
            print()
        
        # Ortalama inter-class similarity
        n = len(present_classes)
        inter_sim = (cos_sim.sum() - np.trace(cos_sim)) / (n * (n-1)) if n > 1 else 0
        print(f"\n  Ortalama inter-class similarity: {inter_sim:.4f}")
        if inter_sim > 0.9:
            print(f"  ⚠ ÇOK YÜKSEK! CNN feature'lar sınıfları ayırt edemiyor!")
        elif inter_sim > 0.8:
            print(f"  ⚠ Yüksek — feature'lar yeterince ayırt edici değil")
        else:
            print(f"  ✓ Makul düzeyde")
    
    return features, labels


def analyze_transformer_attention(model, test_loader):
    """Transformer attention pattern analizi."""
    print(f"\n{'='*70}")
    print("3. TRANSFORMER ATTENTION ANALİZİ")
    print(f"{'='*70}\n")
    
    all_attn = []
    
    model.eval()
    with torch.no_grad():
        for batch_idx, (signals, labels) in enumerate(test_loader):
            if batch_idx >= 5:
                break
            signals = signals.to(config.DEVICE)
            
            try:
                attn_weights = model.get_attention_weights(signals)
                for layer_idx, aw in enumerate(attn_weights):
                    if batch_idx == 0:
                        all_attn.append([])
                    all_attn[layer_idx].append(aw.cpu().numpy())
            except Exception as e:
                print(f"  Attention extraction hatası: {e}")
                return
    
    for layer_idx in range(len(all_attn)):
        attn = np.concatenate(all_attn[layer_idx], axis=0)  # [N, nhead, S, S]
        
        print(f"\nLayer {layer_idx}:")
        print(f"  Shape: {attn.shape}")
        
        # Her head için entropy (uniform attention = yüksek entropy)
        for head in range(attn.shape[1]):
            head_attn = attn[:, head, :, :]  # [N, S, S]
            
            # Entropy hesapla
            eps = 1e-10
            entropy = -(head_attn * np.log(head_attn + eps)).sum(axis=-1).mean()
            max_entropy = np.log(attn.shape[-1])
            entropy_ratio = entropy / max_entropy
            
            # Diagonal dominance (her epoch kendine mi bakıyor?)
            diag_mean = np.array([head_attn[:, i, i] for i in range(head_attn.shape[1])]).mean()
            
            # Max attention position
            max_positions = head_attn.mean(axis=0).argmax(axis=-1)
            
            marker = ""
            if entropy_ratio > 0.95:
                marker = "⚠ UNIFORM (bilgi yok)"
            elif diag_mean > 0.5:
                marker = "⚠ SELF-ONLY (komşulara bakmıyor)"
            
            print(f"  Head {head}: entropy_ratio={entropy_ratio:.3f}  "
                  f"diag={diag_mean:.3f}  {marker}")


def analyze_predictions(model, test_loader):
    """Prediction dağılımı ve class collapse analizi."""
    print(f"\n{'='*70}")
    print("4. PREDICTION DAĞILIMI ANALİZİ")
    print(f"{'='*70}\n")
    
    all_preds = []
    all_labels = []
    all_logits = []
    
    model.eval()
    with torch.no_grad():
        for signals, labels in test_loader:
            signals = signals.to(config.DEVICE)
            outputs = model(signals)
            
            if outputs.dim() == 3:
                B, S, C = outputs.shape
                outputs = outputs.reshape(B * S, C)
                labels = labels.reshape(B * S)
            
            all_logits.append(outputs.cpu().numpy())
            all_preds.extend(outputs.argmax(dim=1).cpu().numpy())
            all_labels.extend(labels.numpy())
    
    preds = np.array(all_preds)
    labels = np.array(all_labels)
    logits = np.concatenate(all_logits, axis=0)
    
    # Prediction distribution
    pred_counts = Counter(preds)
    label_counts = Counter(labels)
    
    print(f"{'Class':8s} {'True':>8s} {'True%':>8s} {'Pred':>8s} {'Pred%':>8s} {'Durum':>10s}")
    print("-" * 55)
    for c in range(config.N_CLASSES):
        tc = label_counts.get(c, 0)
        pc = pred_counts.get(c, 0)
        tp = tc / len(labels) * 100
        pp = pc / len(preds) * 100
        
        status = ""
        if pc == 0:
            status = "❌ COLLAPSE"
        elif pp > tp * 2:
            status = "⚠ DOMINANT"
        elif pp < tp * 0.3:
            status = "⚠ AZ TAHMİN"
        
        print(f"{config.CLASS_NAMES[c]:8s} {tc:8d} {tp:7.1f}% {pc:8d} {pp:7.1f}% {status}")
    
    # Logit istatistikleri
    print(f"\nLogit İstatistikleri:")
    probs = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    
    print(f"  Max prob (ortalama): {probs.max(axis=1).mean():.4f}")
    print(f"  Entropy (ortalama): {-(probs * np.log(probs + 1e-10)).sum(axis=1).mean():.4f}")
    print(f"  Max entropy (5 class): {np.log(5):.4f}")
    
    confidence = probs.max(axis=1).mean()
    if confidence < 0.4:
        print(f"  ⚠ Model çok düşük güvenle tahmin yapıyor — kararsız!")
    elif confidence > 0.9:
        print(f"  ⚠ Model aşırı güvenli — overfit olmuş olabilir")
    
    # Sınıf bazlı logit ortalamaları
    print(f"\nSınıf Bazlı Ortalama Logitler (her true class için model ne çıktı veriyor):")
    for c in range(config.N_CLASSES):
        mask = labels == c
        if mask.sum() == 0:
            continue
        c_logits = logits[mask].mean(axis=0)
        pred_class = c_logits.argmax()
        vals = " ".join([f"{v:+.2f}" for v in c_logits])
        correct = "✓" if pred_class == c else f"→ {config.CLASS_NAMES[pred_class]}"
        print(f"  True {config.CLASS_NAMES[c]:4s}: [{vals}] {correct}")


def analyze_gradient_flow(model, test_loader):
    """Gradient akışı analizi — hangi katmanlar öğrenebiliyor?"""
    print(f"\n{'='*70}")
    print("5. GRADIENT AKIŞI ANALİZİ")
    print(f"{'='*70}\n")
    
    model.train()  # Gradient için train mode
    
    # Tek bir batch ile gradient hesapla
    signals, labels = next(iter(test_loader))
    signals = signals.to(config.DEVICE).requires_grad_(True)
    labels = labels.to(config.DEVICE)
    
    outputs = model(signals)
    if outputs.dim() == 3:
        B, S, C = outputs.shape
        outputs = outputs.reshape(B * S, C)
        labels = labels.reshape(B * S)
    
    loss = F.cross_entropy(outputs, labels)
    loss.backward()
    
    print(f"Loss: {loss.item():.4f}")
    print()
    
    # Her katmanın gradient büyüklüğü
    print(f"{'Katman':50s} {'Grad Mean':>12s} {'Grad Std':>12s} {'Durum':>15s}")
    print("-" * 92)
    
    for name, param in model.named_parameters():
        if param.grad is None or param.dim() < 2:
            continue
        
        grad = param.grad.data.cpu()
        grad_mean = grad.abs().mean().item()
        grad_std = grad.std().item()
        
        short = name.replace('transformer_encoder.', 'trfm.')
        short = short.replace('classification_head.', 'head.')
        short = short.replace('cnn_encoder.', 'cnn.')
        
        status = ""
        if grad_mean < 1e-7:
            status = "❌ VANISHING"
        elif grad_mean > 1.0:
            status = "⚠ EXPLODING"
        elif grad_mean < 1e-5:
            status = "⚠ çok küçük"
        
        print(f"  {short:48s} {grad_mean:12.2e} {grad_std:12.2e} {status}")
    
    model.eval()


def main():
    parser = argparse.ArgumentParser(description='SleepTransformer Diagnostik Analizi')
    parser.add_argument('--fold_dir', type=str, required=True,
                       help='Fold dizini (ör: outputs/runs/260428-1450_kfold/fold_0)')
    args = parser.parse_args()
    
    from edf_to_mongo import preload_all_stages, get_offline_patient_ids
    
    print(f"\n{'#'*70}")
    print(f"# SLEEPTRANSFORMER DERİN DİAGNOSTİK ANALİZİ")
    print(f"# Fold: {args.fold_dir}")
    print(f"{'#'*70}\n")
    
    # Stage cache yükle — patient ID'leri split dosyasından al
    split_file = os.path.join(args.fold_dir, 'patient_split.json')
    with open(split_file) as f:
        split = json.load(f)
    all_ids = sorted(set(split['train'] + split['val'] + split['test']))
    
    preload_all_stages(all_ids, config.MONGO_URI, config.DB_NAME)
    
    # Model ve veri yükle
    model, test_loader, test_dataset = load_model_and_data(args.fold_dir)
    
    # Analizler
    analyze_weights(model)
    features, labels = analyze_cnn_features(model, test_loader)
    analyze_transformer_attention(model, test_loader)
    analyze_predictions(model, test_loader)
    analyze_gradient_flow(model, test_loader)
    
    print(f"\n{'#'*70}")
    print(f"# DİAGNOSTİK TAMAMLANDI")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
