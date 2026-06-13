#!/bin/bash
# Training Wrapper Script
# Automatically creates timestamped run directory and logs output
#
# Kullanım:
#   ./run_training.sh                           # Yeni eğitim başlat
#   ./run_training.sh --resume 260314-2125_kfold  # Kalan fold'lardan devam et

# --resume parametresini kontrol et
RESUME_ID=""
if [ "$1" = "--resume" ]; then
    if [ -z "$2" ]; then
        echo "========================================================================"
        echo "HATA: --resume parametresi için run ID gerekli!"
        echo "========================================================================"
        echo ""
        echo "Kullanım:"
        echo "  ./run_training.sh --resume <run_id>"
        echo ""
        echo "Örnek:"
        echo "  ./run_training.sh --resume 260314-2125_kfold"
        echo ""
        echo "Mevcut kfold run'lar:"
        if [ -d "outputs/runs" ]; then
            ls -1dt outputs/runs/*_kfold 2>/dev/null | while read dir; do
                basename "$dir"
            done
        else
            echo "  (Henüz run yok)"
        fi
        echo "========================================================================"
        exit 1
    fi

    RESUME_ID=$2
    RESUME_DIR="outputs/runs/$RESUME_ID"

    if [ ! -d "$RESUME_DIR" ]; then
        echo "========================================================================"
        echo "HATA: Run dizini bulunamadı: $RESUME_DIR"
        echo "========================================================================"
        echo ""
        echo "Mevcut kfold run'lar:"
        ls -1dt outputs/runs/*_kfold 2>/dev/null | while read dir; do
            basename "$dir"
        done
        echo "========================================================================"
        exit 1
    fi
fi

# Başlangıç zamanı
START_TIME=$(date +%s)
START_LABEL=$(date +"%Y-%m-%d %H:%M:%S")

# Timestamp oluştur (YYMMDD-HHMM formatında)
TIMESTAMP=$(date +"%y%m%d-%H%M")

# Geçici log için tmp klasörünü kullan
mkdir -p tmp
TEMP_LOG="tmp/train_${TIMESTAMP}.log"

if [ -n "$RESUME_ID" ]; then
    echo "========================================================================"
    echo "K-FOLD RESUME BAŞLIYOR - RUN ID: $RESUME_ID"
    echo "========================================================================"
    echo ""

    # Python script'i --resume ile çalıştır
    python3 -u train.py --resume "$RESUME_ID" 2>&1 | tee "$TEMP_LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    # Log dosyasını resume dizinine taşı
    RUN_DIR="$RESUME_DIR"
    mv "$TEMP_LOG" "$RUN_DIR/train_resume_${TIMESTAMP}.log"
    LOG_LOCATION="$RUN_DIR/train_resume_${TIMESTAMP}.log"
else
    RUN_DIR="outputs/runs/$TIMESTAMP"

    echo "========================================================================"
    echo "TRAINING BAŞLIYOR - RUN ID: $TIMESTAMP"
    echo "========================================================================"
    echo ""

    # Python script'i çalıştır ve log'u geçici dosyaya yaz
    # Run directory train.py tarafından oluşturulacak
    python3 -u train.py 2>&1 | tee "$TEMP_LOG"
    EXIT_CODE=${PIPESTATUS[0]}

    # Log dosyasını run directory'ye taşı
    # train.py kfold modunda {TIMESTAMP}_kfold dizini oluşturur, normal modda {TIMESTAMP}
    KFOLD_DIR="outputs/runs/${TIMESTAMP}_kfold"
    if [ -d "$KFOLD_DIR" ]; then
        mv "$TEMP_LOG" "$KFOLD_DIR/train.log"
        LOG_LOCATION="$KFOLD_DIR/train.log"
        RUN_DIR="$KFOLD_DIR"
    elif [ -d "$RUN_DIR" ]; then
        mv "$TEMP_LOG" "$RUN_DIR/train.log"
        LOG_LOCATION="$RUN_DIR/train.log"
    else
        # Run directory oluşmadıysa (hata durumu), temp'te bırak
        LOG_LOCATION="$TEMP_LOG"
    fi
fi

# Geçen süreyi hesapla
END_TIME=$(date +%s)
END_LABEL=$(date +"%Y-%m-%d %H:%M:%S")
ELAPSED=$((END_TIME - START_TIME))
ELAPSED_H=$((ELAPSED / 3600))
ELAPSED_M=$(((ELAPSED % 3600) / 60))
ELAPSED_S=$((ELAPSED % 60))
ELAPSED_STR=$(printf "%02d:%02d:%02d" $ELAPSED_H $ELAPSED_M $ELAPSED_S)

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "========================================================================"
    echo "✓ TRAINING TAMAMLANDI!"
    echo "========================================================================"
    # RUN_DIR kfold modunda güncellendi, basename'i al
    RUN_NAME=$(basename "$RUN_DIR")
    echo "Run ID: $RUN_NAME"
    echo "Run Directory: $RUN_DIR"
    echo "Başlangıç : $START_LABEL"
    echo "Bitiş     : $END_LABEL"
    echo "Toplam süre: $ELAPSED_STR (saat:dakika:saniye)"
    echo ""
    echo "Dosyalar:"
    echo "  - train.log"
    echo "  - training_history.png"
    echo "  - training_history.json"
    echo "  - patient_split.json"
    echo "  - checkpoints/best_model.pth"
    echo ""
    echo "Evaluate için:"
    echo "  ./run_evaluate.sh $RUN_NAME"
    echo "========================================================================"
else
    echo ""
    echo "========================================================================"
    echo "✗ TRAINING BAŞARISIZ (Exit code: $EXIT_CODE)"
    echo "========================================================================"
    echo "TÜM PIPELINE SÜRESİ:"
    echo "Başlangıç : $START_LABEL"
    echo "Bitiş     : $END_LABEL"
    echo "Geçen süre: $ELAPSED_STR (saat:dakika:saniye)"
    echo "Log dosyası: $LOG_LOCATION"
    echo "========================================================================"
fi

exit $EXIT_CODE
