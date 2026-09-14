"""
차선 검출기 순수 성능 평가
==========================
fallback / 스무딩을 끄고 프레임별로 독립 평가하여 실제 검출률을 측정한다.
조향각 구간별로 나눠 어느 구간이 취약한지 확인한다.

사용법:
    cd ~/Desktop/aicar
    python3 eval_detector.py                # 구간별 200장씩
    python3 eval_detector.py 500            # 구간별 500장씩
"""
import glob, re, sys, random
import cv2, numpy as np
from lane_detector import LaneDetector, LaneDetectorConfig

N = int(sys.argv[1]) if len(sys.argv) > 1 else 200
random.seed(0)

# fallback / 스무딩 OFF → 순수 검출 성능만 측정
cfg = LaneDetectorConfig(fallback_to_last=False, smoothing_window=1)

BUCKETS = [("급좌회전", -99, -46), ("좌회전", -45, -6), ("직진", -5, 5),
           ("우회전", 6, 45), ("급우회전", 46, 99)]

files = glob.glob("merged/*.png")
groups = {b[0]: [] for b in BUCKETS}
for f in files:
    m = re.search(r'_(\d{3})_([+-]\d+)\.png$', f)
    if not m:
        continue
    a = int(m.group(2))
    for name, lo, hi in BUCKETS:
        if lo <= a <= hi:
            groups[name].append((f, a)); break

print("=" * 68)
print(f"차선 검출 순수 성능 (fallback OFF, 스무딩 OFF, 구간당 최대 {N}장)")
print("=" * 68)
print(f"{'구간':<10}{'표본':>7}{'검출':>7}{'검출률':>9}{'평균|편차|':>11}{'편차 std':>10}")
print("-" * 68)

tot_n = tot_ok = 0
all_dev = []
for name, lo, hi in BUCKETS:
    sample = random.sample(groups[name], min(N, len(groups[name])))
    devs = []
    for path, ang in sample:
        det = LaneDetector(cfg)          # 프레임마다 새 인스턴스 → 상태 격리
        img = cv2.imread(path)
        if img is None:
            continue
        d = det.compute_deviation(img)
        if d is not None:
            devs.append(d)
    n, ok = len(sample), len(devs)
    tot_n += n; tot_ok += ok; all_dev += devs
    rate = ok / n * 100 if n else 0
    ma = np.mean(np.abs(devs)) if devs else float('nan')
    sd = np.std(devs) if devs else float('nan')
    print(f"{name:<10}{n:>7}{ok:>7}{rate:>8.1f}%{ma:>11.2f}{sd:>10.2f}")

print("-" * 68)
print(f"{'전체':<10}{tot_n:>7}{tot_ok:>7}{tot_ok/tot_n*100:>8.1f}%"
      f"{np.mean(np.abs(all_dev)):>11.2f}{np.std(all_dev):>10.2f}")

print("\n" + "=" * 68)
print("판정")
print("=" * 68)
r = tot_ok / tot_n * 100
if r >= 90:
    print(f"  검출률 {r:.1f}% — 실차 검증 진행 가능")
elif r >= 75:
    print(f"  검출률 {r:.1f}% — 사용 가능하나 취약 구간 파라미터 조정 권장")
else:
    print(f"  검출률 {r:.1f}% — 파라미터 재조정 필요")
print("\n  구간별 검출률이 크게 다르면, 낮은 구간에서 편차가 체계적으로")
print("  누락되어 실차 비교가 왜곡될 수 있으므로 논문에 명시해야 합니다.")