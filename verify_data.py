"""
데이터 분포 검증 스크립트
---------------------------
수집된 실차 주행 데이터의 속도/조향각 분포와 편향 여부를 즉시 확인한다.

사용법:
    python3 verify_data.py <데이터_폴더_경로>

예:
    python3 verify_data.py ~/aicarA/data/merged
    python3 verify_data.py /Users/linear/Downloads/dataset

파일명 규격 (기존 gi.py 규격):
    P0001_045_+15.png  → 프로젝트=P0001, 속도=45, 조향각=+15
"""
import os
import sys
import re
from pathlib import Path
from collections import Counter
import argparse
from lap_split import time_order, blocked_split, check

order = time_order(files)          # files = 파일명 리스트 변수명 맞춰서
print(np.array(files)[order][:5])  # 순서 확인 ★ 눈으로 봐
print(np.array(files)[order][-5:])

steer_t = np.asarray(steer)[order] # 조향각도 같은 순서로
tr, va = blocked_split(len(order), 10, (2,7), 60)
check(steer_t, tr, va)

try:
    import matplotlib
    matplotlib.use('Agg')          # GUI 없이 이미지 저장만
    import matplotlib.pyplot as plt
    import numpy as np
except ImportError:
    print("matplotlib/numpy 필요: pip install matplotlib numpy")
    sys.exit(1)


PATTERN = re.compile(
    r'^(?P<prefix>.*?)'
    r'(?P<index>\d{4})_'
    r'(?P<speed>\d{2,3})_'
    r'(?P<aug>[LR])?'
    r'(?P<angle>[+-]?\d{1,3})'
    r'(?:_S\d+)?'
    r'(?:_F\d+)?'
    r'\.(png|jpg|jpeg)$'
)


def parse_filename(name):
    m = PATTERN.match(name)
    if not m:
        return None
    try:
        speed = int(m.group('speed'))
        angle = int(m.group('angle'))
        is_aug = bool(m.group('aug'))
        return speed, angle, is_aug
    except (ValueError, TypeError):
        return None


def compute_uniformity(counter):
    values = list(counter.values())
    if not values or min(values) == 0:
        return None
    return max(values) / min(values)


def bin_speeds(speeds, bin_size=5):
    return Counter([int(s // bin_size * bin_size) for s in speeds])


def bin_angles(angles, bin_size=15):
    return Counter([int(a // bin_size * bin_size) for a in angles])


def draw_distribution(speeds, angles, output_path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].hist(speeds, bins=range(min(speeds)-2, max(speeds)+3, 5),
                 edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Speed')
    axes[0].set_ylabel('Count')
    axes[0].set_title(f'Speed Distribution (Total {len(speeds):,})')
    axes[0].grid(alpha=0.3)

    axes[1].hist(angles, bins=range(-100, 105, 10),
                 edgecolor='black', alpha=0.7, color='orange')
    axes[1].set_xlabel('Steering Angle (deg)')
    axes[1].set_ylabel('Count')
    axes[1].set_title(f'Steering Angle Distribution (Total {len(angles):,})')
    axes[1].axvline(0, color='red', linestyle='--', alpha=0.5, label='Straight (0)')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close()
    print(f"[OK] 분포 그래프 저장: {output_path}")


def verify(folder, save_plot=True):
    folder = Path(folder).expanduser().resolve()
    if not folder.is_dir():
        print(f"[ERROR] 폴더가 없거나 폴더가 아님: {folder}")
        return

    print("=" * 60)
    print(f"데이터 폴더: {folder}")
    print("=" * 60)

    speeds = []
    angles = []
    skipped = []
    aug_count = 0

    all_files = list(folder.rglob('*.png')) + list(folder.rglob('*.jpg'))

    for path in all_files:
        parsed = parse_filename(path.name)
        if parsed is None:
            skipped.append(path.name)
            continue
        speed, angle, is_aug = parsed
        speeds.append(speed)
        angles.append(angle)
        if is_aug:
            aug_count += 1

    total = len(speeds)
    print(f"\n전체 이미지: {total:,}장")
    if aug_count:
        print(f"  원본: {total - aug_count:,}장")
        print(f"  증강: {aug_count:,}장")
    if skipped:
        print(f"[WARN] 파일명 파싱 실패: {len(skipped)}개")
        for name in skipped[:5]:
            print(f"     - {name}")
        if len(skipped) > 5:
            print(f"     ... 외 {len(skipped)-5}개")

    if total == 0:
        print("\n[ERROR] 파싱된 이미지가 없습니다. 파일명 규격을 확인하세요.")
        return

    # 속도 분포
    print("\n" + "-" * 60)
    print("속도 분포 (5 단위)")
    print("-" * 60)
    speed_bins = bin_speeds(speeds, bin_size=5)
    for sp in sorted(speed_bins.keys()):
        cnt = speed_bins[sp]
        ratio = cnt / total * 100
        bar = '#' * int(ratio / 2)
        print(f"  {sp:3d}~{sp+4:3d}: {cnt:6,d}장 ({ratio:5.1f}%) {bar}")

    print(f"\n  min: {min(speeds)}, max: {max(speeds)}, mean: {sum(speeds)/total:.1f}")

    speed_uniformity = compute_uniformity(speed_bins)
    if speed_uniformity is not None:
        print(f"  균등성 (max/min): {speed_uniformity:.2f}")
        if speed_uniformity < 2:
            print(f"  [OK] 균등한 편입니다 (편향 낮음)")
        elif speed_uniformity < 5:
            print(f"  [WARN] 약간 편향됨 (허용 범위)")
        else:
            print(f"  [FAIL] 편향이 큽니다 (논문 결과 신뢰도 우려)")

    # 조향각 분포
    print("\n" + "-" * 60)
    print("조향각 분포 (15° 단위)")
    print("-" * 60)
    angle_bins = bin_angles(angles, bin_size=15)
    for ang in sorted(angle_bins.keys()):
        cnt = angle_bins[ang]
        ratio = cnt / total * 100
        bar = '#' * int(ratio / 2)
        print(f"  {ang:+4d}~{ang+14:+4d}: {cnt:6,d}장 ({ratio:5.1f}%) {bar}")

    print(f"\n  min: {min(angles):+d}, max: {max(angles):+d}, mean: {sum(angles)/total:+.1f}")

    left = sum(1 for a in angles if a < -5)
    right = sum(1 for a in angles if a > 5)
    straight = total - left - right
    print(f"\n  직진 (|angle| <= 5): {straight:6,d}장 ({straight/total*100:5.1f}%)")
    print(f"  좌회전 (angle < -5): {left:6,d}장 ({left/total*100:5.1f}%)")
    print(f"  우회전 (angle > +5): {right:6,d}장 ({right/total*100:5.1f}%)")

    if left > 0 and right > 0:
        lr_balance = max(left, right) / min(left, right)
        print(f"  좌/우 균형 (max/min): {lr_balance:.2f}")
        if lr_balance < 1.5:
            print(f"  [OK] 좌우 균형 좋음")
        elif lr_balance < 3:
            print(f"  [WARN] 좌우 다소 불균형")
        else:
            print(f"  [FAIL] 좌우 심한 불균형 (증강으로 보완 권장)")

    # 서브셋 안내
    print("\n" + "-" * 60)
    print("논문 실험용 서브셋 분할 가능성")
    print("-" * 60)
    print(f"  전체:  {total:,}장")

    for subset_size in [1000, 3000, 5000, 10000]:
        if total >= subset_size:
            print(f"  {subset_size:5,d} 서브셋: [OK] 가능 ({total - subset_size:,}장 여유)")
        else:
            print(f"  {subset_size:5,d} 서브셋: [FAIL] 부족 ({subset_size - total:,}장 부족)")

    # 시각화
    if save_plot:
        print("\n" + "-" * 60)
        output_path = folder / 'data_distribution.png'
        draw_distribution(speeds, angles, output_path)

    # 최종 진단
    print("\n" + "=" * 60)
    print("최종 진단")
    print("=" * 60)

    issues = []
    if total < 5000:
        issues.append(f"[FAIL] 데이터 규모가 작음 ({total:,}장 < 5,000장)")
    if speed_uniformity is not None and speed_uniformity > 5:
        issues.append(f"[FAIL] 속도 편향이 큼 (max/min = {speed_uniformity:.1f})")
    if left > 0 and right > 0 and (max(left, right) / min(left, right)) > 3:
        issues.append(f"[FAIL] 좌우 조향각 불균형")

    if not issues:
        print("[OK] 논문 실험 진행 가능! 데이터 품질 양호.")
    else:
        print("[WARN] 아래 이슈 확인 후 진행:")
        for issue in issues:
            print(f"   {issue}")

    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='실차 데이터 분포 검증')
    parser.add_argument('folder', help='데이터 폴더 경로')
    parser.add_argument('--no-plot', action='store_true',
                        help='히스토그램 저장 생략')
    args = parser.parse_args()
    verify(args.folder, save_plot=not args.no_plot)
