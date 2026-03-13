#!/bin/bash
# K-Fold Cross Validation Training Wrapper Script
# Automatically creates timestamped run directory and logs output

# Timestamp oluştur (YYMMDD-HHMM formatında)
TIMESTAMP=$(date +"%y%m%d-%H%M")
RUN_DIR="outputs/runs/${TIMESTAMP}_kfold"

# Geçici log için tmp klasörünü kullan
mkdir -p tmp
TEMP_LOG="tmp/train_kfold_${TIMESTAMP}.log"

echo "========================================================================"
echo "K-FOLD CROSS VALIDATION BAŞLIYOR - RUN ID: ${TIMESTAMP}_kfold"
echo "========================================================================"
echo ""

# Python script'i çalıştır ve log'u geçici dosyaya yaz
python3 train.py 2>&1 | tee "$TEMP_LOG"

# Exit code'u kontrol et
EXIT_CODE=${PIPESTATUS[0]}

# Log dosyasını run directory'ye taşı
if [ -d "$RUN_DIR" ]; then
    mv "$TEMP_LOG" "$RUN_DIR/train.log"
    LOG_LOCATION="$RUN_DIR/train.log"
else
    LOG_LOCATION="$TEMP_LOG"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ K-FOLD TRAINING TAMAMLANDI!"
    echo "========================================================================"
    echo "Run ID: ${TIMESTAMP}_kfold"
    echo "Run Directory: $RUN_DIR"
    echo ""
    echo "Dosyalar:"
    echo "  - train.log"
    echo "  - kfold_summary.json"
    echo "  - test_patients.json"
    echo "  - fold_0/ ... fold_4/"
    echo "    - training_history.png"
    echo "    - checkpoints/best_model.pth"
    echo ""
    echo "Evaluate için:"
    echo "  ./run_kfold_evaluate.sh ${TIMESTAMP}_kfold"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ K-FOLD TRAINING BAŞARISIZ (Exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "Log dosyası: $LOG_LOCATION"
    echo "========================================================================"
fi

exit $EXIT_CODE
