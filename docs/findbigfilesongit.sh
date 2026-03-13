IDX=".git/objects/pack/pack-8f7973aaf44eab5edc2ada4b56be06d130cd1c52.idx"

# 1) Pack içindeki en büyük 50 blob'un SHA'sını al
git verify-pack -v "$IDX" \
  | awk '$2=="blob" {print $1, $3}' \
  | sort -k2 -n \
  | tail -n 50 > /tmp/big-blobs.txt

# 2) Bu blob SHA'larını dosya yollarına map et (ulaşılabilir olanları)
cut -d' ' -f1 /tmp/big-blobs.txt > /tmp/big-shas.txt
git rev-list --objects --all > /tmp/all-objects.txt

grep -Ff /tmp/big-shas.txt /tmp/all-objects.txt | head -n 50