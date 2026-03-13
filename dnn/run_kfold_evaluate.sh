#!/bin/bash
# K-Fold Cross Validation Evaluation Wrapper Script
# Evaluates a K-Fold CV training run using the best fold model

# Kullanım kontrolü
if [ $# -eq 0 ]; then
    echo "========================================================================"
    echo "KULLANIM"
    echo "========================================================================"
    echo "  ./run_kfold_evaluate.sh <run_id>"
    echo ""
    echo "Örnek:"
    echo "  ./run_kfold_evaluate.sh 251220-1730_kfold"
    echo ""
    echo "Mevcut K-Fold run'lar:"
    echo "========================================================================"
    
    # Mevcut K-Fold run'ları listele
    if [ -d "outputs/runs" ]; then
        ls -1t outputs/runs/ | grep "_kfold$" | head -10
    else
        echo "  (Henüz run yok)"
    fi
    
    echo "========================================================================"
    exit 1
fi

RUN_ID=$1
RUN_DIR="outputs/runs/$RUN_ID"

# Run directory kontrolü
if [ ! -d "$RUN_DIR" ]; then
    echo "✗ Run directory bulunamadı: $RUN_DIR"
    echo ""
    echo "Mevcut K-Fold run'lar:"
    ls -1t outputs/runs/ 2>/dev/null | grep "_kfold$" || echo "  (Henüz run yok)"
    exit 1
fi

# kfold_summary.json kontrolü
if [ ! -f "$RUN_DIR/kfold_summary.json" ]; then
    echo "✗ Bu bir K-Fold run değil: $RUN_DIR"
    echo "  (kfold_summary.json bulunamadı)"
    exit 1
fi

echo "========================================================================"
echo "K-FOLD EVALUATION BAŞLIYOR - RUN ID: $RUN_ID"
echo "========================================================================"
echo "Run Directory: $RUN_DIR"
echo ""

# K-Fold özet bilgilerini göster
echo "K-Fold Özeti:"
python3 -c "
import json
with open('$RUN_DIR/kfold_summary.json') as f:
    s = json.load(f)
print(f\"  CV Hastalar: {s['n_cv_patients']}\")
print(f\"  Test Hastalar: {s['n_test_patients']}\")
print(f\"  Fold Sayısı: {s['n_folds']}\")
print(f\"  Mean Val Acc: {s['mean_val_acc']:.2f}% ± {s['std_val_acc']:.2f}%\")
print(f\"  Mean Val Loss: {s['mean_val_loss']:.4f} ± {s['std_val_loss']:.4f}\")
"
echo ""

# En iyi fold'u bul
BEST_FOLD=$(python3 -c "
import json
with open('$RUN_DIR/kfold_summary.json') as f:
    s = json.load(f)
best = max(s['fold_results'], key=lambda x: x['best_val_acc'])
print(best['fold'])
")

echo "En iyi fold: fold_$BEST_FOLD"
BEST_FOLD_DIR="$RUN_DIR/fold_$BEST_FOLD"

# Test hastalarıyla evaluate et (patient_split.json içinde test var)
echo ""
echo "Test seti üzerinde evaluation yapılıyor (fold_$BEST_FOLD modeli)..."
echo "========================================================================"

python3 evaulate.py --run_dir "$BEST_FOLD_DIR" 2>&1 | tee "$RUN_DIR/evaluate.log"

EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ K-FOLD EVALUATION TAMAMLANDI!"
    echo "========================================================================"
    echo "Run ID: $RUN_ID"
    echo "Best Model: fold_$BEST_FOLD"
    echo ""
    echo "Sonuç dosyaları:"
    echo "  - evaluate.log"
    echo "  - fold_$BEST_FOLD/test_results.json"
    echo "  - fold_$BEST_FOLD/confusion_matrix.png"
    echo "  - fold_$BEST_FOLD/hypnograms/*.png"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ K-FOLD EVALUATION BAŞARISIZ (Exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "Log dosyası: $RUN_DIR/evaluate.log"
    echo "========================================================================"
fi

exit $EXIT_CODE
