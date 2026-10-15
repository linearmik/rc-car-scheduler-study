#!/bin/bash
# ============================================================
# run_overnight.sh — 스케줄러 실험 80회 야간 실행
# ============================================================
# 사용법:
#     cd ~/Desktop/aicar        (또는 ~/aicar)
#     chmod +x run_overnight.sh   ← 최초 1회만
#     ./run_overnight.sh
#
# 중단:  Ctrl+C  (다시 실행하면 남은 것부터 이어서 진행)
# 진행:  다른 터미널에서  tail -f run.log
# ============================================================

set -u
cd "$(dirname "$0")"

DATA_DIR="linear_rc"
OUT_DIR="results"
LOG="run.log"

echo "============================================================"
echo " 스케줄러 실험 80회 (4 스케줄러 x 4 데이터 x 5 시드)"
echo "============================================================"

# ---------- 사전 점검 ----------
fail=0

# 1) 필수 파일
for f in config.py schedulers.py train_experiment.py run_all_experiments.py; do
    if [ ! -f "$f" ]; then
        echo "  [X] $f 없음"
        fail=1
    fi
done

# 2) 데이터 폴더
if [ ! -d "$DATA_DIR" ]; then
    echo "  [X] $DATA_DIR 폴더 없음 → unzip -q merged_0827.zip 먼저 실행"
    fail=1
else
    n=$(ls "$DATA_DIR"/*.png 2>/dev/null | wc -l | tr -d ' ')
    if [ "$n" -lt 1000 ]; then
        echo "  [X] $DATA_DIR 안에 이미지가 $n 장뿐 (22890장이어야 정상)"
        echo "      압축이 merged/merged/ 처럼 한 겹 더 풀렸을 수 있음"
        fail=1
    else
        echo "  [O] 이미지 $n 장"
    fi
fi

# 3) 파이썬 패키지
python3 - <<'PY' || fail=1
import sys
try:
    import torch, cv2, numpy
except ImportError as e:
    print(f"  [X] 패키지 없음: {e.name}")
    sys.exit(1)
dev = "MPS(GPU)" if torch.backends.mps.is_available() else (
      "CUDA" if torch.cuda.is_available() else "CPU(느림!)")
print(f"  [O] torch {torch.__version__} / opencv {cv2.__version__} / 장치 {dev}")
PY

# 4) 전원 연결 (배터리로 돌리면 중간에 죽음)
if pmset -g ps 2>/dev/null | grep -q "Battery Power"; then
    echo "  [!] 배터리로 실행 중 — 전원 어댑터를 연결하세요"
    fail=1
else
    echo "  [O] 전원 연결됨"
fi

# 5) 디스크 여유
avail=$(df -g . | tail -1 | awk '{print $4}')
if [ "$avail" -lt 5 ]; then
    echo "  [!] 디스크 여유 ${avail}GB — 5GB 이상 권장"
else
    echo "  [O] 디스크 여유 ${avail}GB"
fi

echo "------------------------------------------------------------"
if [ "$fail" -ne 0 ]; then
    echo " 점검 실패. 위 [X] 항목을 해결한 뒤 다시 실행하세요."
    exit 1
fi

# ---------- 남은 실험 수 ----------
done_n=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo " 완료 $done_n / 80 회  →  남은 $((80 - done_n)) 회"
echo " 예상 소요: 약 14시간 (early stopping 반영)"
echo "------------------------------------------------------------"
echo " 시작하면:"
echo "   - 화면만 끄려면  Ctrl+Shift+전원버튼"
echo "   - 뚜껑은 닫지 마세요 (닫으면 잠들어서 중단됩니다)"
echo "   - 진행 확인: 다른 터미널에서  tail -f $LOG"
echo "------------------------------------------------------------"
printf " 시작할까요? [y/N] "
read -r ans
case "$ans" in
    [yY]) ;;
    *) echo " 취소했습니다."; exit 0 ;;
esac

# ---------- 실행 ----------
echo ""
echo "시작: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"

caffeinate -is python3 run_all_experiments.py \
    --data-dir "$DATA_DIR" \
    --out-dir  "$OUT_DIR" \
    --save-model \
    2>&1 | tee -a "$LOG"

echo "종료: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG"

final=$(ls "$OUT_DIR"/*.json 2>/dev/null | wc -l | tr -d ' ')
echo ""
echo "============================================================"
echo " 결과 JSON: $final / 80 개"
echo " 위치: $(pwd)/$OUT_DIR"
if [ "$final" -lt 80 ]; then
    echo " 덜 끝났습니다. ./run_overnight.sh 를 다시 실행하면 이어서 진행됩니다."
fi
echo "============================================================"