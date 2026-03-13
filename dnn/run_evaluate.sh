#!/bin/bash
# Evaluation Wrapper Script
# Evaluates a specific training run

# Kullanım kontrolü
if [ $# -eq 0 ]; then
    echo "========================================================================"
    echo "KULLANIM"
    echo "========================================================================"
    echo "  ./run_evaluate.sh <run_id>"
    echo ""
    echo "Örnek:"
    echo "  ./run_evaluate.sh 251220-1730"
    echo ""
    echo "Mevcut run'lar:"
    echo "========================================================================"
    
    # Mevcut run'ları listele
    if [ -d "outputs/runs" ]; then
        ls -1t outputs/runs/ | head -10
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
    echo "Mevcut run'lar:"
    ls -1t outputs/runs/ 2>/dev/null || echo "  (Henüz run yok)"
    exit 1
fi

echo "========================================================================"
echo "EVALUATION BAŞLIYOR - RUN ID: $RUN_ID"
echo "========================================================================"
echo "Run Directory: $RUN_DIR"
echo ""

# Python script'i çalıştır ve log'u hem ekrana hem dosyaya yaz
python3 evaulate.py --run_dir "$RUN_DIR" 2>&1 | tee "$RUN_DIR/evaluate.log"

# Exit code'u kontrol et
EXIT_CODE=${PIPESTATUS[0]}

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ EVALUATION TAMAMLANDI!"
    echo "========================================================================"
    echo "Run ID: $RUN_ID"
    echo "Run Directory: $RUN_DIR"
    echo ""
    echo "Sonuç dosyaları:"
    echo "  - evaluate.log"
    echo "  - test_results.json"
    echo "  - confusion_matrix.png"
    echo "  - per_class_metrics.png"
    echo "  - hypnograms/*.png"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ EVALUATION BAŞARISIZ (Exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "Log dosyası: $RUN_DIR/evaluate.log"
    echo "========================================================================"
fi

exit $EXIT_CODE
