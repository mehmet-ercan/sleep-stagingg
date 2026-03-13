#!/bin/bash
# Training Wrapper Script
# Automatically creates timestamped run directory and logs output

# Timestamp oluştur (YYMMDD-HHMM formatında)
TIMESTAMP=$(date +"%y%m%d-%H%M")
RUN_DIR="outputs/runs/$TIMESTAMP"

# Geçici log için tmp klasörünü kullan
mkdir -p tmp
TEMP_LOG="tmp/train_${TIMESTAMP}.log"

echo "========================================================================"
echo "TRAINING BAŞLIYOR - RUN ID: $TIMESTAMP"
echo "========================================================================"
echo ""

# Python script'i çalıştır ve log'u geçici dosyaya yaz
# Run directory train.py tarafından oluşturulacak
python3 train.py 2>&1 | tee "$TEMP_LOG"

# Exit code'u kontrol et
EXIT_CODE=${PIPESTATUS[0]}

# Log dosyasını run directory'ye taşı
if [ -d "$RUN_DIR" ]; then
    mv "$TEMP_LOG" "$RUN_DIR/train.log"
    LOG_LOCATION="$RUN_DIR/train.log"
else
    # Run directory oluşmadıysa (hata durumu), temp'te bırak
    LOG_LOCATION="$TEMP_LOG"
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ TRAINING TAMAMLANDI!"
    echo "========================================================================"
    echo "Run ID: $TIMESTAMP"
    echo "Run Directory: $RUN_DIR"
    echo ""
    echo "Dosyalar:"
    echo "  - train.log"
    echo "  - training_history.png"
    echo "  - training_history.json"
    echo "  - patient_split.json"
    echo "  - checkpoints/best_model.pth"
    echo ""
    echo "Evaluate için:"
    echo "  ./run_evaluate.sh $TIMESTAMP"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ TRAINING BAŞARISIZ (Exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "Log dosyası: $LOG_LOCATION"
    echo "========================================================================"
fi

exit $EXIT_CODE
