"""
Model Diagnostik Scripti
Eğitilmiş modeli yükleyip katman katman inceleyerek
neden hep aynı sınıfı tahmin ettiğini araştırır.
"""

import sys
import os
import torch
import torch.nn.functional as F
import numpy as np
import json

# Proje path'ini ayarla
sys.path.insert(0, os.path.dirname(__file__))
import config

# ============================================================================
# 1. Checkpoint Yükleme ve Temel Kontroller
# ============================================================================
def load_checkpoint(run_id):
    """Checkpoint yükle ve model durumunu kontrol et."""
    run_dir = os.path.join('outputs/runs', run_id)
    ckpt_path = os.path.join(run_dir, 'checkpoints', 'best_model.pth')

    if not os.path.exists(ckpt_path):
        print(f"HATA: Checkpoint bulunamadı: {ckpt_path}")
        return None, None

    print(f"\n{'='*70}")
    print(f"CHECKPOINT YÜKLENİYOR: {run_id}")
    print(f"{'='*70}")

    checkpoint = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    print(f"Checkpoint keys: {list(checkpoint.keys())}")
    if 'epoch' in checkpoint:
        print(f"Best epoch: {checkpoint['epoch']}")
    if 'val_loss' in checkpoint:
        print(f"Best val_loss: {checkpoint['val_loss']:.4f}")
    if 'val_accuracy' in checkpoint:
        print(f"Best val_accuracy: {checkpoint['val_accuracy']:.2f}%")

    return checkpoint, run_dir


def analyze_model_weights(state_dict):
    """Model ağırlıklarının istatistiklerini detaylı incele."""
    print(f"\n{'='*70}")
    print("MODEL AĞIRLIKLARI ANALİZİ")
    print(f"{'='*70}")

    issues = []

    for name, param in state_dict.items():
        p = param.float()
        mean_val = p.mean().item()
        std_val = p.std().item()
        min_val = p.min().item()
        max_val = p.max().item()
        abs_max = p.abs().max().item()
        zero_pct = (p == 0).float().mean().item() * 100
        nan_count = torch.isnan(p).sum().item()
        inf_count = torch.isinf(p).sum().item()

        # Sorun tespiti
        flag = ""
        if nan_count > 0:
            flag = " ⚠️ NaN TESPİT EDİLDİ!"
            issues.append(f"NaN: {name}")
        elif inf_count > 0:
            flag = " ⚠️ Inf TESPİT EDİLDİ!"
            issues.append(f"Inf: {name}")
        elif std_val < 1e-7:
            flag = " ⚠️ ÇOK KÜÇÜK STD (dead weights?)"
            issues.append(f"Dead weights: {name}")
        elif abs_max > 100:
            flag = " ⚠️ ÇOK BÜYÜK DEĞERLER"
            issues.append(f"Exploding: {name}")
        elif zero_pct > 90:
            flag = " ⚠️ %90+ SIFIR (dead neurons)"
            issues.append(f"Dead neurons: {name}")

        print(f"\n  {name}: shape={list(p.shape)}{flag}")
        print(f"    mean={mean_val:+.6f}  std={std_val:.6f}  "
              f"min={min_val:+.6f}  max={max_val:+.6f}  "
              f"zero%={zero_pct:.1f}%")

    return issues


def analyze_final_layer(state_dict):
    """Son katman (classifier) ağırlıklarını özel olarak incele."""
    print(f"\n{'='*70}")
    print("SON KATMAN (CLASSIFIER) DETAYLI ANALİZ")
    print(f"{'='*70}")

    # Son FC katmanın weight ve bias'ını bul
    fc_weight_key = None
    fc_bias_key = None

    for name in state_dict:
        if 'fc2.weight' in name or 'fc_out.weight' in name or 'classifier.5.weight' in name:
            fc_weight_key = name
        if 'fc2.bias' in name or 'fc_out.bias' in name or 'classifier.5.bias' in name:
            fc_bias_key = name

    if fc_weight_key is None:
        # CNN1D_V3 classifier Sequential
        for name in state_dict:
            if name.endswith('.weight') and 'classifier' in name:
                fc_weight_key = name  # son eşleşme
            if name.endswith('.bias') and 'classifier' in name:
                fc_bias_key = name

    print(f"  Weight key: {fc_weight_key}")
    print(f"  Bias key: {fc_bias_key}")

    if fc_weight_key:
        w = state_dict[fc_weight_key].float()
        print(f"\n  Output layer weight shape: {list(w.shape)}")
        print(f"  Her sınıf için weight norm:")
        for i, cls_name in enumerate(config.CLASS_NAMES):
            row = w[i]
            print(f"    {cls_name} (idx={i}): norm={row.norm():.4f}  "
                  f"mean={row.mean():.6f}  std={row.std():.6f}  "
                  f"min={row.min():.6f}  max={row.max():.6f}")

    if fc_bias_key:
        b = state_dict[fc_bias_key].float()
        print(f"\n  Output layer bias:")
        for i, cls_name in enumerate(config.CLASS_NAMES):
            print(f"    {cls_name} (idx={i}): bias={b[i]:.6f}")

        # Bias farkı - en büyük bias farkı ne kadar?
        max_bias = b.max().item()
        min_bias = b.min().item()
        bias_range = max_bias - min_bias
        dominant_class = config.CLASS_NAMES[b.argmax().item()]
        print(f"\n  Bias aralığı: {bias_range:.4f}")
        print(f"  En yüksek bias: {dominant_class} ({max_bias:.6f})")

        if bias_range > 2.0:
            print(f"  ⚠️ BÜYÜK BİAS FARKI! Model {dominant_class} sınıfına doğru güçlü bias var!")
            print(f"     Bu, modelin input'tan bağımsız olarak {dominant_class} tahmin etmesine yol açabilir.")


def forward_pass_analysis(model, run_dir):
    """
    Modelden gerçek veri geçirip her katmandaki aktivasyonları analiz et.
    """
    print(f"\n{'='*70}")
    print("FORWARD PASS ANALİZİ (Rastgele ve Gerçek Veri)")
    print(f"{'='*70}")

    model.eval()

    # --- Test 1: Sıfır input ---
    print("\n--- Test 1: SIFIR INPUT ---")
    zero_input = torch.zeros(1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
    with torch.no_grad():
        logits = model(zero_input)
    probs = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    print(f"  Logits: {logits[0].numpy()}")
    print(f"  Probs:  {probs[0].numpy()}")
    print(f"  Tahmin: {config.CLASS_NAMES[pred]} (idx={pred})")

    # --- Test 2: Rastgele normal dağılım input ---
    print("\n--- Test 2: RASTGELE NORMAL INPUT ---")
    rand_input = torch.randn(1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
    with torch.no_grad():
        logits = model(rand_input)
    probs = F.softmax(logits, dim=1)
    pred = logits.argmax(dim=1).item()
    print(f"  Logits: {logits[0].numpy()}")
    print(f"  Probs:  {probs[0].numpy()}")
    print(f"  Tahmin: {config.CLASS_NAMES[pred]} (idx={pred})")

    # --- Test 3: Farklı büyüklüklerde rastgele input ---
    print("\n--- Test 3: FARKLI BÜYÜKLÜKLERDE INPUT ---")
    for scale in [0.001, 0.1, 1.0, 10.0, 100.0]:
        rand_input = torch.randn(1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH) * scale
        with torch.no_grad():
            logits = model(rand_input)
        pred = logits.argmax(dim=1).item()
        logit_range = logits.max().item() - logits.min().item()
        print(f"  Scale={scale:6.3f}: tahmin={config.CLASS_NAMES[pred]}, "
              f"logit_range={logit_range:.4f}, logits={logits[0].numpy().round(3)}")

    # --- Test 4: Batch'te çeşitli input ---
    print("\n--- Test 4: BATCH ANALİZİ (100 rastgele sample) ---")
    batch_input = torch.randn(100, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
    with torch.no_grad():
        logits = model(batch_input)
    preds = logits.argmax(dim=1).numpy()
    unique, counts = np.unique(preds, return_counts=True)
    print(f"  Tahmin dağılımı:")
    for u, c in zip(unique, counts):
        print(f"    {config.CLASS_NAMES[u]}: {c}/100 ({c}%)")

    logit_means = logits.mean(dim=0).numpy()
    logit_stds = logits.std(dim=0).numpy()
    print(f"  Ortalama logitler: {logit_means.round(4)}")
    print(f"  Logit std'ler:     {logit_stds.round(4)}")

    # Her sınıf arası logit farkı
    print(f"\n  Sınıflar arası ortalama logit farkları:")
    for i in range(5):
        for j in range(i+1, 5):
            diff = abs(logit_means[i] - logit_means[j])
            print(f"    {config.CLASS_NAMES[i]} vs {config.CLASS_NAMES[j]}: {diff:.4f}")


def layer_by_layer_analysis(model, model_type='cnn1d_v3'):
    """
    Modeli katman katman geçirip her aşamadaki aktivasyonları kaydet.
    """
    print(f"\n{'='*70}")
    print("KATMAN KATMAN AKTİVASYON ANALİZİ")
    print(f"{'='*70}")

    model.eval()
    x = torch.randn(8, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)

    def print_stats(name, tensor):
        t = tensor.detach()
        dead_pct = (t.abs() < 1e-6).float().mean().item() * 100
        print(f"  {name:30s}: shape={list(t.shape)}  "
              f"mean={t.mean():.4f}  std={t.std():.4f}  "
              f"min={t.min():.4f}  max={t.max():.4f}  "
              f"dead%={dead_pct:.1f}")

    print_stats("INPUT", x)

    with torch.no_grad():
        if model_type == 'cnn1d_v3':
            # Stem - her branch
            branch_outputs = []
            for i, branch in enumerate(model.stem.branches):
                out = branch(x)
                print_stats(f"  stem.branch[{i}] (k={config.CNN1D_V3_CONFIG['stem_kernels'][i]})", out)
                branch_outputs.append(out)

            x = torch.cat(branch_outputs, dim=1)
            print_stats("  stem.concat", x)

            x = F.relu(model.stem.bn(x))
            print_stats("  stem.bn+relu", x)

            x = model.stem.pool(x)
            print_stats("  stem.pool", x)

            # Block 1
            residual = model.block1.skip(x)
            print_stats("  block1.skip", residual)

            x1 = F.relu(model.block1.bn1(model.block1.conv1(x)))
            print_stats("  block1.conv1+bn+relu", x1)

            x1 = model.block1.bn2(model.block1.conv2(x1))
            print_stats("  block1.conv2+bn", x1)

            x = F.relu(x1 + residual)
            print_stats("  block1.residual+relu", x)

            x = model.block1.pool(x)
            print_stats("  block1.pool", x)

            # Block 2
            residual = model.block2.skip(x)
            print_stats("  block2.skip", residual)

            x1 = F.relu(model.block2.bn1(model.block2.conv1(x)))
            print_stats("  block2.conv1+bn+relu", x1)

            x1 = model.block2.bn2(model.block2.conv2(x1))
            print_stats("  block2.conv2+bn", x1)

            x = F.relu(x1 + residual)
            print_stats("  block2.residual+relu", x)

            x = model.block2.pool(x)
            print_stats("  block2.pool", x)

            # SE
            w = x.mean(dim=2)
            print_stats("  se.gap", w)

            w = F.relu(model.se.fc1(w))
            print_stats("  se.fc1+relu", w)

            w = torch.sigmoid(model.se.fc2(w))
            print_stats("  se.sigmoid", w)

            x = x * w.unsqueeze(2)
            print_stats("  se.output", x)

            # GAP
            x = x.mean(dim=2)
            print_stats("  GAP", x)

            # Classifier
            layers = list(model.classifier.children())
            for i, layer in enumerate(layers):
                x = layer(x)
                print_stats(f"  classifier[{i}] ({type(layer).__name__})", x)

            print(f"\n  FINAL LOGITS (batch[0]):")
            for i, cls_name in enumerate(config.CLASS_NAMES):
                print(f"    {cls_name}: {x[0][i]:.6f}")

        elif model_type == 'cnn1d':
            x = model.conv1(x)
            print_stats("conv1", x)
            x = F.relu(x)
            print_stats("relu1", x)
            x = model.pool1(x)
            print_stats("pool1", x)

            x = model.conv2(x)
            print_stats("conv2", x)
            x = F.relu(x)
            print_stats("relu2", x)
            x = model.pool2(x)
            print_stats("pool2", x)

            x = model.conv3(x)
            print_stats("conv3", x)
            x = F.relu(x)
            print_stats("relu3", x)
            x = model.pool3(x)
            print_stats("pool3", x)

            x = torch.flatten(x, start_dim=1)
            print_stats("flatten", x)

            x = model.fc1(x)
            print_stats("fc1", x)
            x = F.relu(x)
            print_stats("fc1_relu", x)

            x = model.fc2(x)
            print_stats("fc2 (logits)", x)

            print(f"\n  FINAL LOGITS (batch[0]):")
            for i, cls_name in enumerate(config.CLASS_NAMES):
                print(f"    {cls_name}: {x[0][i]:.6f}")


def analyze_real_data(model, run_dir):
    """
    Gerçek test verisinden birkaç sample yükleyip analiz et.
    """
    print(f"\n{'='*70}")
    print("GERÇEK VERİ ANALİZİ")
    print(f"{'='*70}")

    # Patient split yükle
    split_path = os.path.join(run_dir, 'patient_split.json')
    if not os.path.exists(split_path):
        print("  patient_split.json bulunamadı, atlaniyor.")
        return

    with open(split_path) as f:
        split = json.load(f)

    test_ids = split.get('test', [])
    print(f"  Test hastaları: {len(test_ids)}")

    # Test sonuçlarından label dağılımına bak
    results_path = os.path.join(run_dir, 'test_results.json')
    if os.path.exists(results_path):
        with open(results_path) as f:
            results = json.load(f)

        labels = np.array(results.get('labels', results.get('true_labels', [])))
        preds = np.array(results['predictions'])

        print(f"\n  TEST SONUÇLARI:")
        print(f"  Toplam sample: {len(labels)}")
        print(f"  Gerçek etiket dağılımı:")
        for i, cls in enumerate(config.CLASS_NAMES):
            count = (labels == i).sum()
            print(f"    {cls}: {count} ({count/len(labels)*100:.1f}%)")

        print(f"  Tahmin dağılımı:")
        for i, cls in enumerate(config.CLASS_NAMES):
            count = (preds == i).sum()
            print(f"    {cls}: {count} ({count/len(preds)*100:.1f}%)")

        # Logits varsa analiz et
        if 'logits' in results:
            logits = np.array(results['logits'])
            print(f"\n  LOGİT ANALİZİ:")
            print(f"  Ortalama logitler: {logits.mean(axis=0).round(4)}")
            print(f"  Std logitler:      {logits.std(axis=0).round(4)}")
            print(f"  Max logitler:      {logits.max(axis=0).round(4)}")
            print(f"  Min logitler:      {logits.min(axis=0).round(4)}")


def check_batchnorm_stats(state_dict):
    """BatchNorm running_mean ve running_var istatistiklerini kontrol et."""
    print(f"\n{'='*70}")
    print("BATCHNORM RUNNING STATS KONTROLÜ")
    print(f"{'='*70}")

    bn_keys = [k for k in state_dict if 'running_mean' in k or 'running_var' in k or 'num_batches_tracked' in k]

    if not bn_keys:
        print("  BatchNorm bulunamadı.")
        return

    for key in sorted(bn_keys):
        val = state_dict[key].float()
        if 'num_batches' in key:
            print(f"  {key}: {val.item()}")
        else:
            print(f"  {key}: mean={val.mean():.6f}  std={val.std():.6f}  "
                  f"min={val.min():.6f}  max={val.max():.6f}  "
                  f"zero%={(val.abs() < 1e-8).float().mean().item()*100:.1f}%")

            if 'running_var' in key and (val < 0).any():
                print(f"    ⚠️ NEGATİF VARYANS TESPİT EDİLDİ!")
            if 'running_var' in key and (val < 1e-10).any():
                count = (val < 1e-10).sum().item()
                print(f"    ⚠️ {count} kanal neredeyse sıfır varyansa sahip (dead channel)")


def gradient_check(model):
    """
    Modele veri geçirip gradient akışını kontrol et.
    Gradientler çok küçükse (vanishing) veya çok büyükse (exploding) tespit eder.
    """
    print(f"\n{'='*70}")
    print("GRADİENT AKIŞ KONTROLÜ")
    print(f"{'='*70}")

    model.train()
    x = torch.randn(4, config.N_CHANNELS, config.SAMPLES_PER_EPOCH, requires_grad=False)
    target = torch.tensor([0, 1, 2, 3])  # Farklı sınıflar

    logits = model(x)
    loss = F.cross_entropy(logits, target)
    loss.backward()

    print(f"  Loss: {loss.item():.4f}")

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            grad_norm = grad.norm().item()
            grad_mean = grad.abs().mean().item()
            grad_max = grad.abs().max().item()

            flag = ""
            if grad_norm < 1e-8:
                flag = " ⚠️ VANISHING GRADIENT!"
            elif grad_norm > 100:
                flag = " ⚠️ EXPLODING GRADIENT!"

            print(f"  {name:40s}: grad_norm={grad_norm:.6f}  "
                  f"mean_abs={grad_mean:.6f}  max_abs={grad_max:.6f}{flag}")

    model.zero_grad()
    model.eval()


def check_data_signal_stats(run_dir):
    """
    MongoDB'den veya cache'den birkaç örnek yükleyip sinyal istatistiklerini kontrol et.
    """
    print(f"\n{'='*70}")
    print("SİNYAL VERİSİ KALİTE KONTROLÜ")
    print(f"{'='*70}")

    split_path = os.path.join(run_dir, 'patient_split.json')
    if not os.path.exists(split_path):
        print("  patient_split.json bulunamadı, atlaniyor.")
        return

    with open(split_path) as f:
        split = json.load(f)

    # Tüm hastalardan birkaçını kontrol et
    all_ids = split.get('train', [])[:3] + split.get('test', [])[:3]

    try:
        from dataset import SleepEpochDataset

        print(f"  {len(all_ids)} hasta kontrol ediliyor...")

        dataset = SleepEpochDataset(
            patient_ids=all_ids,
            channels=config.SELECTED_CHANNELS,
            normalize=None,  # Ham veriyi al
            augment=False,
            preload=True
        )

        print(f"  Dataset boyutu: {len(dataset)} epoch")

        # İlk 50 sample'ın ham istatistiklerini kontrol et
        all_signals = []
        all_labels = []
        n_samples = min(50, len(dataset))

        for i in range(n_samples):
            signal, label = dataset[i]
            all_signals.append(signal.numpy())
            all_labels.append(label.item())

        signals = np.array(all_signals)  # [N, C, T]
        labels = np.array(all_labels)

        print(f"\n  HAM SİNYAL İSTATİSTİKLERİ ({n_samples} sample, normalize=None):")
        for ch_idx, ch_name in enumerate(config.SELECTED_CHANNELS):
            ch_data = signals[:, ch_idx, :]
            print(f"    {ch_name}: mean={ch_data.mean():.4f}  std={ch_data.std():.4f}  "
                  f"min={ch_data.min():.4f}  max={ch_data.max():.4f}")

            # Tüm sıfır mı?
            if ch_data.std() < 1e-8:
                print(f"    ⚠️ {ch_name} KANALI TAMAMEN SIFIR VEYA SABİT!")

            # Çok büyük değerler?
            if ch_data.std() > 1000:
                print(f"    ⚠️ {ch_name} KANALI ÇOK BÜYÜK DEĞERLER İÇERİYOR (artifact?)")

        # Etiket dağılımı
        print(f"\n  Bu sample'lardaki etiket dağılımı:")
        for i, cls in enumerate(config.CLASS_NAMES):
            count = (labels == i).sum()
            if count > 0:
                print(f"    {cls}: {count}")

        # Farklı etiketli sample'ların sinyallerini karşılaştır
        print(f"\n  SINIFLAR ARASI SİNYAL FARKI:")
        for cls_idx, cls_name in enumerate(config.CLASS_NAMES):
            mask = labels == cls_idx
            if mask.sum() > 0:
                cls_signals = signals[mask]
                print(f"    {cls_name} ({mask.sum()} sample): "
                      f"mean={cls_signals.mean():.4f}  std={cls_signals.std():.4f}")

        # Z-score sonrası kontrol
        print(f"\n  Z-SCORE SONRASI SİNYAL İSTATİSTİKLERİ:")
        dataset2 = SleepEpochDataset(
            patient_ids=all_ids,
            channels=config.SELECTED_CHANNELS,
            normalize='z-score',
            augment=False,
            preload=True
        )

        norm_signals = []
        for i in range(n_samples):
            signal, label = dataset2[i]
            norm_signals.append(signal.numpy())

        norm_signals = np.array(norm_signals)
        for ch_idx, ch_name in enumerate(config.SELECTED_CHANNELS):
            ch_data = norm_signals[:, ch_idx, :]
            print(f"    {ch_name}: mean={ch_data.mean():.4f}  std={ch_data.std():.4f}  "
                  f"min={ch_data.min():.4f}  max={ch_data.max():.4f}")

            # NaN veya Inf kontrol
            nan_count = np.isnan(ch_data).sum()
            inf_count = np.isinf(ch_data).sum()
            if nan_count > 0:
                print(f"    ⚠️ {ch_name} NaN TESPİT EDİLDİ! ({nan_count} değer)")
            if inf_count > 0:
                print(f"    ⚠️ {ch_name} Inf TESPİT EDİLDİ! ({inf_count} değer)")

        dataset.clear_cache()
        dataset2.clear_cache()

    except Exception as e:
        print(f"  Veri yükleme hatası: {e}")
        import traceback
        traceback.print_exc()


def logit_sensitivity_test(model):
    """
    Aynı base input'a küçük pertürbasyonlar ekleyerek
    modelin output'unun ne kadar değiştiğini ölç.
    Eğer logitler neredeyse hiç değişmiyorsa → model input'u ignore ediyor.
    """
    print(f"\n{'='*70}")
    print("LOGİT SENSİTİVİTE TESTİ")
    print(f"{'='*70}")

    model.eval()

    base_input = torch.randn(1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)

    with torch.no_grad():
        base_logits = model(base_input).numpy()[0]

    print(f"  Base logits: {base_logits.round(4)}")
    print(f"  Base tahmin: {config.CLASS_NAMES[base_logits.argmax()]}")

    # 10 farklı pertürbasyon dene
    logit_diffs = []
    pred_changes = 0

    for i in range(10):
        perturbed = base_input + torch.randn_like(base_input) * 0.1
        with torch.no_grad():
            new_logits = model(perturbed).numpy()[0]

        diff = np.abs(new_logits - base_logits).mean()
        logit_diffs.append(diff)

        if new_logits.argmax() != base_logits.argmax():
            pred_changes += 1

    mean_diff = np.mean(logit_diffs)
    print(f"\n  Küçük pertürbasyon (std=0.1) sonucu:")
    print(f"    Ortalama logit değişimi: {mean_diff:.6f}")
    print(f"    Tahmin değişme sayısı:   {pred_changes}/10")

    if mean_diff < 0.01:
        print(f"    ⚠️ Model input'a ÇOK AZ duyarlı! Logitler neredeyse sabit.")
        print(f"       Bu, modelin input'u görmezden geldiğini gösterir.")
    elif mean_diff < 0.1:
        print(f"    ⚠️ Model input'a DÜŞÜK duyarlılık gösteriyor.")
    else:
        print(f"    ✓ Model input'a duyarlı.")

    # Tamamen farklı inputlar
    print(f"\n  Tamamen farklı inputlar (20 sample):")
    all_logits = []
    all_preds = []
    for i in range(20):
        rand_input = torch.randn(1, config.N_CHANNELS, config.SAMPLES_PER_EPOCH)
        with torch.no_grad():
            logits = model(rand_input).numpy()[0]
        all_logits.append(logits)
        all_preds.append(logits.argmax())

    all_logits = np.array(all_logits)
    unique_preds = np.unique(all_preds)
    print(f"    Farklı tahmin sayısı: {len(unique_preds)}/5 sınıf")
    print(f"    Tahmin edilen sınıflar: {[config.CLASS_NAMES[p] for p in unique_preds]}")
    print(f"    Logit std (sınıflar arası varyasyon): {all_logits.std(axis=0).round(4)}")
    print(f"    Logit std (samplelar arası varyasyon): {all_logits.std(axis=1).mean():.4f}")

    if len(unique_preds) == 1:
        print(f"    ⚠️⚠️ TÜM TAHMİNLER AYNI SINIF: {config.CLASS_NAMES[unique_preds[0]]}")
        print(f"       Model TAMAMEN ÇÖKMÜŞ (collapsed)!")


# ============================================================================
# MAIN
# ============================================================================
if __name__ == "__main__":
    # Her iki run'ı da analiz et
    run_ids = ['260322-1635', '260322-1855']

    for run_id in run_ids:
        print(f"\n\n{'#'*70}")
        print(f"# RUN: {run_id}")
        print(f"{'#'*70}")

        checkpoint, run_dir = load_checkpoint(run_id)
        if checkpoint is None:
            continue

        state_dict = checkpoint.get('model_state_dict', checkpoint)

        # 1. Ağırlık analizi
        issues = analyze_model_weights(state_dict)

        # 2. Son katman analizi
        analyze_final_layer(state_dict)

        # 3. BatchNorm kontrolü
        check_batchnorm_stats(state_dict)

        # 4. Modeli yükle
        from models.cnn1d import CNN1D_V3, get_model
        model = CNN1D_V3()
        model.load_state_dict(state_dict)
        model.eval()

        # 5. Forward pass analizi
        forward_pass_analysis(model, run_dir)

        # 6. Katman katman analiz
        layer_by_layer_analysis(model, model_type='cnn1d_v3')

        # 7. Sensitivite testi
        logit_sensitivity_test(model)

        # 8. Gradient kontrolü
        gradient_check(model)

        # 9. Gerçek veri analizi
        analyze_real_data(model, run_dir)

        print(f"\n{'='*70}")
        print(f"SORUN ÖZETİ ({run_id})")
        print(f"{'='*70}")
        if issues:
            for issue in issues:
                print(f"  ⚠️ {issue}")
        else:
            print("  Ağırlıklarda belirgin anomali tespit edilmedi.")
        print()

    # Son olarak: sinyal kalite kontrolü (sadece ilk run için)
    print(f"\n\n{'#'*70}")
    print(f"# SİNYAL KALİTE KONTROLÜ")
    print(f"{'#'*70}")
    check_data_signal_stats('outputs/runs/260322-1635')
