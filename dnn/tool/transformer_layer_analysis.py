"""
Transformer Katman-Katman Derin Diagnostik
==========================================
Dıştan içe doğru sistematik analiz:
  Level 1: Transformer bypass testi (CNN feature → Head)
  Level 2: Residual bağlantı analizi (transformer ne kadar değiştiriyor?)
  Level 3: Positional encoding etkisi
  Level 4: Attention Q, K, V detaylı analiz
  Level 5: LayerNorm ve FFN etkisi

Kullanım:
    python tool/transformer_layer_analysis.py --fold_dir outputs/runs/260511-1656_kfold/fold_3
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import argparse
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from models.cnn1d import get_model
from dataset import SleepSequenceDataset
from torch.utils.data import DataLoader


def load_model_and_data(fold_dir):
    """Model ve test verisini yükle."""
    with open(os.path.join(fold_dir, 'patient_split.json')) as f:
        split = json.load(f)
    
    from edf_to_mongo import preload_all_stages
    all_ids = sorted(set(split['train'] + split['val'] + split['test']))
    preload_all_stages(all_ids, config.MONGO_URI, config.DB_NAME)
    
    model = get_model(config.MODEL_TYPE)
    ckpt_path = os.path.join(fold_dir, 'checkpoints', 'best_model.pth')
    ckpt = torch.load(ckpt_path, map_location=config.DEVICE, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'])
    model.eval()
    print(f"✓ Model yüklendi: epoch {ckpt['epoch']}, val acc: {ckpt['best_val_acc']:.2f}%")
    
    test_dataset = SleepSequenceDataset(
        patient_ids=split['test'],
        sequence_length=config.SEQUENCE_LENGTH,
        stride=config.SEQUENCE_LENGTH,
        preload=True, augment=False
    )
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)
    return model, test_loader


def get_batches(test_loader, n=5):
    """İlk N batch'i al."""
    batches = []
    for i, (signals, labels) in enumerate(test_loader):
        if i >= n:
            break
        batches.append((signals.to(config.DEVICE), labels.to(config.DEVICE)))
    return batches


# ========================================================================
# LEVEL 1: Transformer bypass — CNN feature'lar doğrudan head'e verilse?
# ========================================================================
def level1_bypass_test(model, batches):
    """CNN → Head (transformer bypass). Transformer eklemenin etkisini ölç."""
    print(f"\n{'#'*70}")
    print(f"# LEVEL 1: TRANSFORMER BYPASS TESTİ")
    print(f"# CNN feature → Classification Head (transformer atlanıyor)")
    print(f"{'#'*70}\n")
    
    correct_with_trfm = 0
    correct_without_trfm = 0
    total = 0
    
    model.eval()
    with torch.no_grad():
        for signals, labels in batches:
            B, S, C, T = signals.shape
            labels_flat = labels.reshape(-1)
            
            # A) Normal forward (CNN → Transformer → Head)
            logits_full = model(signals)  # [B, S, 5]
            preds_full = logits_full.reshape(-1, 5).argmax(dim=1)
            
            # B) Bypass forward (CNN → Head, transformer atlanıyor)
            x_flat = signals.view(B * S, C, T)
            features = model.cnn_encoder(x_flat)  # [B*S, 128]
            if model.cnn_projection is not None:
                features = model.cnn_projection(features)
            features_seq = features.view(B, S, -1)  # [B, S, 128]
            logits_bypass = model.classification_head(features_seq)  # [B, S, 5]
            preds_bypass = logits_bypass.reshape(-1, 5).argmax(dim=1)
            
            correct_with_trfm += (preds_full == labels_flat).sum().item()
            correct_without_trfm += (preds_bypass == labels_flat).sum().item()
            total += labels_flat.numel()
    
    acc_full = correct_with_trfm / total * 100
    acc_bypass = correct_without_trfm / total * 100
    diff = acc_full - acc_bypass
    
    print(f"  CNN → Transformer → Head:  {acc_full:.1f}% accuracy")
    print(f"  CNN → Head (bypass):       {acc_bypass:.1f}% accuracy")
    print(f"  Fark:                      {diff:+.1f}%")
    
    if diff < -2:
        print(f"\n  ❌ TRANSFORMER ZARAR VERİYOR! Bypass daha iyi ({-diff:.1f}% fark)")
    elif diff < 2:
        print(f"\n  ⚠ Transformer neredeyse hiç etki etmiyor (fark < 2%)")
    else:
        print(f"\n  ✓ Transformer iyileştirme sağlıyor (+{diff:.1f}%)")
    
    return acc_full, acc_bypass


# ========================================================================
# LEVEL 2: Residual connection — Transformer feature'ları ne kadar değiştiriyor?
# ========================================================================
def level2_residual_analysis(model, batches):
    """Transformer giriş vs çıkış karşılaştırması."""
    print(f"\n{'#'*70}")
    print(f"# LEVEL 2: RESIDUAL CONNECTION ANALİZİ")
    print(f"# CNN feature → Transformer → ne kadar değişti?")
    print(f"{'#'*70}\n")
    
    all_input_norms = []
    all_output_norms = []
    all_delta_norms = []
    all_cosine_sims = []
    
    model.eval()
    with torch.no_grad():
        for signals, labels in batches:
            B, S, C, T = signals.shape
            
            # CNN encoding
            x_flat = signals.view(B * S, C, T)
            features = model.cnn_encoder(x_flat)
            if model.cnn_projection is not None:
                features = model.cnn_projection(features)
            features_in = features.view(B, S, -1)  # [B, S, 128]
            
            # Transformer encoding
            features_out = model.transformer_encoder(features_in)  # [B, S, 128]
            
            # Delta analizi
            delta = features_out - features_in
            
            in_norm = features_in.norm(dim=-1).mean().item()
            out_norm = features_out.norm(dim=-1).mean().item()
            delta_norm = delta.norm(dim=-1).mean().item()
            
            # Cosine similarity (giriş vs çıkış)
            cos = F.cosine_similarity(
                features_in.reshape(-1, features_in.shape[-1]),
                features_out.reshape(-1, features_out.shape[-1]),
                dim=-1
            ).mean().item()
            
            all_input_norms.append(in_norm)
            all_output_norms.append(out_norm)
            all_delta_norms.append(delta_norm)
            all_cosine_sims.append(cos)
    
    in_norm = np.mean(all_input_norms)
    out_norm = np.mean(all_output_norms)
    delta_norm = np.mean(all_delta_norms)
    cos_sim = np.mean(all_cosine_sims)
    ratio = delta_norm / (in_norm + 1e-8) * 100
    
    print(f"  CNN feature norm (giriş):     {in_norm:.4f}")
    print(f"  Transformer çıkış norm:       {out_norm:.4f}")
    print(f"  Delta norm (değişim):         {delta_norm:.4f}")
    print(f"  Delta / Input oranı:          {ratio:.2f}%")
    print(f"  Cosine sim (giriş↔çıkış):    {cos_sim:.4f}")
    
    if cos_sim > 0.99:
        print(f"\n  ❌ TRANSFORMER NE GİRDİYSE ONU VERİYOR! (cos_sim={cos_sim:.4f})")
        print(f"     Residual connection dominant — transformer çıkışı yok sayılıyor")
    elif cos_sim > 0.95:
        print(f"\n  ⚠ Transformer çok az değişiklik yapıyor (cos_sim={cos_sim:.4f})")
    else:
        print(f"\n  ✓ Transformer anlamlı dönüşüm yapıyor")


# ========================================================================
# LEVEL 3: Positional Encoding etkisi
# ========================================================================
def level3_positional_encoding(model, batches):
    """PE eklenmeden önce ve sonra feature'lar nasıl değişiyor?"""
    print(f"\n{'#'*70}")
    print(f"# LEVEL 3: POSITIONAL ENCODING ETKİSİ")
    print(f"{'#'*70}\n")
    
    pe_module = model.transformer_encoder.pos_encoder
    
    # PE değerlerini incele
    if hasattr(pe_module, 'pe'):
        pe_data = pe_module.pe.data
        if pe_data.dim() == 3:
            pe_data = pe_data[0]  # [max_len, d_model]
        
        pe_norm = pe_data.norm(dim=-1)
        print(f"  PE shape: {list(pe_data.shape)}")
        print(f"  PE norm (pozisyon başı): min={pe_norm.min():.4f} max={pe_norm.max():.4f} mean={pe_norm.mean():.4f}")
    
    model.eval()
    with torch.no_grad():
        signals, labels = batches[0]
        B, S, C, T = signals.shape
        
        x_flat = signals.view(B * S, C, T)
        features = model.cnn_encoder(x_flat)
        if model.cnn_projection is not None:
            features = model.cnn_projection(features)
        features_in = features.view(B, S, -1)
        
        feature_norm = features_in.norm(dim=-1).mean().item()
        
        # PE eklenmiş hali
        features_with_pe = pe_module(features_in)
        
        # PE sinyalinin feature'a oranı
        pe_signal = features_with_pe - features_in  # dropout olmasa exact PE olurdu
        pe_signal_norm = pe_signal.norm(dim=-1).mean().item()
        
        ratio = pe_signal_norm / feature_norm * 100
        
        print(f"\n  Feature norm (PE öncesi):     {feature_norm:.4f}")
        print(f"  PE signal norm:               {pe_signal_norm:.4f}")
        print(f"  PE / Feature oranı:           {ratio:.2f}%")
        
        if ratio < 1:
            print(f"\n  ❌ PE ÇOK ZAYIF! Feature'ın %{ratio:.2f}'si kadar — duyulamıyor")
        elif ratio < 5:
            print(f"\n  ⚠ PE düşük — pozisyon bilgisi zayıf")
        elif ratio > 50:
            print(f"\n  ⚠ PE çok güçlü — feature bilgisini bozabilir")
        else:
            print(f"\n  ✓ PE makul düzeyde")


# ========================================================================
# LEVEL 4: Q, K, V detaylı analiz
# ========================================================================
def level4_qkv_analysis(model, batches):
    """Attention Q, K, V projektionlarını detaylı incele."""
    print(f"\n{'#'*70}")
    print(f"# LEVEL 4: Q, K, V DETAYLI ANALİZİ")
    print(f"{'#'*70}\n")
    
    model.eval()
    
    # Transformer layer'lara eriş
    trfm = model.transformer_encoder.transformer
    
    for layer_idx, layer in enumerate(trfm.layers):
        print(f"\n--- Layer {layer_idx} ---")
        
        self_attn = layer.self_attn
        d_model = self_attn.embed_dim
        nhead = self_attn.num_heads
        head_dim = d_model // nhead
        
        # in_proj_weight: [3*d_model, d_model] = Q, K, V weight'leri birleşik
        W = self_attn.in_proj_weight.data  # [384, 128]
        b = self_attn.in_proj_bias.data if self_attn.in_proj_bias is not None else None
        
        Wq = W[:d_model]       # [128, 128]
        Wk = W[d_model:2*d_model]  # [128, 128]
        Wv = W[2*d_model:]     # [128, 128]
        
        # Weight istatistikleri
        print(f"  Wq: mean={Wq.mean():.5f} std={Wq.std():.5f} |max|={Wq.abs().max():.4f}")
        print(f"  Wk: mean={Wk.mean():.5f} std={Wk.std():.5f} |max|={Wk.abs().max():.4f}")
        print(f"  Wv: mean={Wv.mean():.5f} std={Wv.std():.5f} |max|={Wv.abs().max():.4f}")
        
        # Q, K, V benzerliği — eğer Q≈K ise attention uniform olur
        qk_cos = F.cosine_similarity(Wq.flatten().unsqueeze(0), 
                                      Wk.flatten().unsqueeze(0)).item()
        qv_cos = F.cosine_similarity(Wq.flatten().unsqueeze(0), 
                                      Wv.flatten().unsqueeze(0)).item()
        kv_cos = F.cosine_similarity(Wk.flatten().unsqueeze(0), 
                                      Wv.flatten().unsqueeze(0)).item()
        
        print(f"\n  Weight cosine similarity:")
        print(f"    Wq↔Wk: {qk_cos:.4f}", end="")
        if abs(qk_cos) > 0.9:
            print(f"  ⚠ Q≈K → attention score'lar çok benzer → uniform attention!")
        else:
            print()
        print(f"    Wq↔Wv: {qv_cos:.4f}")
        print(f"    Wk↔Wv: {kv_cos:.4f}")
        
        # Gerçek Q, K, V değerlerini hesapla
        with torch.no_grad():
            signals, labels = batches[0]
            B, S, C, T = signals.shape
            
            x_flat = signals.view(B * S, C, T)
            features = model.cnn_encoder(x_flat)
            if model.cnn_projection is not None:
                features = model.cnn_projection(features)
            features_seq = features.view(B, S, -1)
            
            # PE ekle
            features_pe = model.transformer_encoder.pos_encoder(features_seq)
            
            # Pre-LN: norm_first=True ise önce LayerNorm uygulanır
            if layer_idx == 0:
                x_in = features_pe
            else:
                # İlk layer'ın çıkışını al
                x_in = trfm.layers[0](features_pe)
            
            # Self-attention norm
            norm1 = layer.norm1
            x_normed = norm1(x_in)  # [B, S, d_model]
            
            # Q, K, V hesapla
            Q = F.linear(x_normed, Wq, b[:d_model] if b is not None else None)
            K = F.linear(x_normed, Wk, b[d_model:2*d_model] if b is not None else None)
            V = F.linear(x_normed, Wv, b[2*d_model:] if b is not None else None)
            
            print(f"\n  Gerçek Q, K, V değerleri (batch 0):")
            print(f"    Q: mean={Q.mean():.4f} std={Q.std():.4f} norm={Q.norm(dim=-1).mean():.4f}")
            print(f"    K: mean={K.mean():.4f} std={K.std():.4f} norm={K.norm(dim=-1).mean():.4f}")
            print(f"    V: mean={V.mean():.4f} std={V.std():.4f} norm={V.norm(dim=-1).mean():.4f}")
            
            # Attention score analizi (softmax öncesi)
            # [B, nhead, S, head_dim]
            Q_heads = Q.view(B, S, nhead, head_dim).transpose(1, 2)
            K_heads = K.view(B, S, nhead, head_dim).transpose(1, 2)
            
            # Raw attention scores
            scale = head_dim ** 0.5
            raw_scores = torch.matmul(Q_heads, K_heads.transpose(-2, -1)) / scale
            
            print(f"\n  Raw attention scores (softmax öncesi):")
            print(f"    Mean: {raw_scores.mean():.4f}")
            print(f"    Std:  {raw_scores.std():.4f}")
            print(f"    Min:  {raw_scores.min():.4f}")
            print(f"    Max:  {raw_scores.max():.4f}")
            print(f"    Range: {raw_scores.max() - raw_scores.min():.4f}")
            
            score_range = (raw_scores.max() - raw_scores.min()).item()
            if score_range < 1.0:
                print(f"    ❌ RANGE ÇOK DAR ({score_range:.3f})! Softmax sonucu ~uniform olur")
                print(f"       Softmax(x) ≈ 1/S when max(x)-min(x) << 1")
            elif score_range < 3.0:
                print(f"    ⚠ Range düşük — softmax kısmen uniform")
            else:
                print(f"    ✓ Range yeterli — softmax selective olabilir")
            
            # Softmax sonrası
            attn_probs = F.softmax(raw_scores, dim=-1)
            
            # Her head için entropy
            for h in range(nhead):
                h_probs = attn_probs[:, h]  # [B, S, S]
                eps = 1e-10
                entropy = -(h_probs * torch.log(h_probs + eps)).sum(dim=-1).mean()
                max_ent = np.log(S)
                print(f"    Head {h}: entropy={entropy:.4f}/{max_ent:.4f} ratio={entropy/max_ent:.4f}")


# ========================================================================
# LEVEL 5: Layer-by-layer feature transformation
# ========================================================================
def level5_layer_tracking(model, batches):
    """Her transformer layer'dan geçerken feature'ın nasıl değiştiğini izle."""
    print(f"\n{'#'*70}")
    print(f"# LEVEL 5: LAYER-BY-LAYER FEATURE İZLEME")
    print(f"# Her katmandan geçerken feature nasıl değişiyor?")
    print(f"{'#'*70}\n")
    
    model.eval()
    trfm = model.transformer_encoder
    
    with torch.no_grad():
        signals, labels = batches[0]
        B, S, C, T = signals.shape
        labels_flat = labels.reshape(-1)
        
        # CNN encoding
        x_flat = signals.view(B * S, C, T)
        features = model.cnn_encoder(x_flat)
        if model.cnn_projection is not None:
            features = model.cnn_projection(features)
        x = features.view(B, S, -1)
        
        stages = ["CNN output"]
        feature_snapshots = [x.clone()]
        
        # PE
        x = trfm.pos_encoder(x)
        stages.append("+ PE")
        feature_snapshots.append(x.clone())
        
        # Her transformer layer
        for i, layer in enumerate(trfm.transformer.layers):
            x = layer(x)
            stages.append(f"Layer {i} output")
            feature_snapshots.append(x.clone())
        
        # Final norm
        x = trfm.norm(x)
        stages.append("Final LayerNorm")
        feature_snapshots.append(x.clone())
        
        # Head
        logits = model.classification_head(x)
        preds = logits.reshape(-1, 5).argmax(dim=1)
        acc = (preds == labels_flat).float().mean().item() * 100
        
        # Her aşamayı karşılaştır
        print(f"{'Stage':25s} {'Norm':>10s} {'Δ from prev':>12s} {'cos(prev)':>12s} {'cos(CNN)':>12s}")
        print("-" * 75)
        
        for i, (stage, snapshot) in enumerate(zip(stages, feature_snapshots)):
            norm = snapshot.norm(dim=-1).mean().item()
            
            if i > 0:
                delta = (snapshot - feature_snapshots[i-1]).norm(dim=-1).mean().item()
                cos_prev = F.cosine_similarity(
                    snapshot.reshape(-1, snapshot.shape[-1]),
                    feature_snapshots[i-1].reshape(-1, feature_snapshots[i-1].shape[-1]),
                    dim=-1
                ).mean().item()
            else:
                delta = 0
                cos_prev = 1.0
            
            # CNN output ile karşılaştır
            cos_cnn = F.cosine_similarity(
                snapshot.reshape(-1, snapshot.shape[-1]),
                feature_snapshots[0].reshape(-1, feature_snapshots[0].shape[-1]),
                dim=-1
            ).mean().item()
            
            print(f"  {stage:23s} {norm:10.4f} {delta:12.4f} {cos_prev:12.4f} {cos_cnn:12.4f}")
        
        print(f"\n  Final accuracy (batch 0): {acc:.1f}%")
        
        # CNN feature → her layer sonrası sınıf ayrımı
        print(f"\n  Sınıf ayrımı her aşamada nasıl değişiyor?")
        for i, (stage, snapshot) in enumerate(zip(stages, feature_snapshots)):
            flat = snapshot.reshape(-1, snapshot.shape[-1]).cpu().numpy()
            labs = labels_flat.cpu().numpy()
            
            # Inter-class cosine similarity
            class_means = []
            for c in range(5):
                mask = labs == c
                if mask.sum() > 0:
                    class_means.append(flat[mask].mean(axis=0))
            
            if len(class_means) >= 2:
                class_means = np.array(class_means)
                norms = np.linalg.norm(class_means, axis=1, keepdims=True) + 1e-8
                normed = class_means / norms
                cos_sim = normed @ normed.T
                n = len(class_means)
                inter = (cos_sim.sum() - np.trace(cos_sim)) / (n * (n-1))
                marker = " ⚠" if inter > 0.85 else (" ✓" if inter < 0.7 else "")
                print(f"    {stage:23s}: inter-class sim = {inter:.4f}{marker}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fold_dir', type=str, required=True)
    args = parser.parse_args()
    
    print(f"\n{'#'*70}")
    print(f"# TRANSFORMER KATMAN-KATMAN DERİN DİAGNOSTİK")
    print(f"# Fold: {args.fold_dir}")
    print(f"{'#'*70}\n")
    
    model, test_loader = load_model_and_data(args.fold_dir)
    batches = get_batches(test_loader, n=5)
    
    # Dıştan içe sistematik analiz
    level1_bypass_test(model, batches)
    level2_residual_analysis(model, batches)
    level3_positional_encoding(model, batches)
    level4_qkv_analysis(model, batches)
    level5_layer_tracking(model, batches)
    
    print(f"\n{'#'*70}")
    print(f"# DİAGNOSTİK TAMAMLANDI")
    print(f"{'#'*70}\n")


if __name__ == "__main__":
    main()
