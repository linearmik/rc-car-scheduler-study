"""
급커브 구간 조향각 예측 분포 비교
==================================
두 모델(22K 학습 / 14K 학습)을 동일 테스트셋에 적용하여
Ground Truth 대비 예측값이 얼마나 수축하는지 정량 비교한다.

실차에서 관측된 조향 부족(22K 모델 최대 34.7° vs 14K 모델 59°)이
오프라인 예측 분포에서도 확인되는지 검증하는 것이 목적이다.

사용법:
    cd ~/Desktop/aicar
    python3 compare_prediction.py

    # 경로를 직접 지정하려면
    python3 compare_prediction.py --data linear_rc \
        --model-a results_22k/warmup_cosine_0_42.pt \
        --model-b results/warmup_cosine_0_42.pt

출력:
    - 구간별 예측 통계 표
    - prediction_distribution.png (논문 그림용)
"""
import argparse, glob, random, re, sys
from pathlib import Path

import numpy as np
import cv2
import torch
import torch.nn as nn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent))
from config import AppConfig, DLConfig, parse_image_filename, get_angle_group
from train_experiment import NvidiaModel, stratified_split, load_items

APP, DLC = AppConfig(), DLConfig()


def preprocess(path):
    """train_experiment.py 와 동일한 전처리."""
    img = cv2.imread(str(path))
    if img is None:
        return None
    h, w = img.shape[:2]
    if (h, w) != (APP.MODEL_INPUT_HEIGHT, APP.MODEL_INPUT_WIDTH):
        if h > APP.ROI_YD:
            img = img[APP.ROI_YU:APP.ROI_YD, APP.ROI_XL:APP.ROI_XR]
        img = cv2.resize(img, (APP.MODEL_INPUT_WIDTH, APP.MODEL_INPUT_HEIGHT))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    img = cv2.GaussianBlur(img, APP.GAUSSIAN_BLUR_KERNEL, 0)
    img = img.astype(np.float32) / APP.NORMALIZE_DIVISOR
    return np.transpose(img, (2, 0, 1))


def load_model(path):
    sd = torch.load(path, map_location="cpu")
    sd = sd.get("state_dict", sd)
    if any(k.startswith("layer") for k in sd):        # ar_research 규격이면 되돌림
        rev = {"layer1.0":"features.0","layer1.2":"features.2","layer1.4":"features.4",
               "layer1.6":"features.6","layer1.9":"features.8","layer2.1":"regressor.1",
               "layer2.3":"regressor.3","layer2.5":"regressor.5","layer3.0":"regressor.7"}
        sd = {f"{rev[k.rpartition('.')[0]]}.{k.rpartition('.')[2]}": v
              for k, v in sd.items()}
    m = NvidiaModel(DLC.MODEL_OUTPUT_DIM)
    m.load_state_dict(sd)
    m.eval()
    return m


def predict(model, items, device="cpu", batch=64):
    preds = []
    buf = []
    for path, *_ in items:
        x = preprocess(path)
        if x is None:
            preds.append(np.nan); continue
        buf.append(x)
        if len(buf) == batch:
            with torch.inference_mode():
                out = model(torch.from_numpy(np.stack(buf)))
            preds.extend((out[:, 0] * DLC.LABEL_SCALE).tolist()); buf = []
    if buf:
        with torch.inference_mode():
            out = model(torch.from_numpy(np.stack(buf)))
        preds.extend((out[:, 0] * DLC.LABEL_SCALE).tolist())
    return np.array(preds)


BANDS = [("직진 |θ|≤10", 0, 10), ("완만 10<|θ|≤25", 10, 25),
         ("중간 25<|θ|≤40", 25, 40), ("급커브 40<|θ|≤60", 40, 60),
         ("초급커브 |θ|>60", 60, 999)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="linear_rc", help="테스트셋으로 쓸 데이터 폴더")
    ap.add_argument("--model-a", required=True, help="비교 모델 A (예: 22K 학습)")
    ap.add_argument("--model-b", required=True, help="비교 모델 B (예: 14K 학습)")
    ap.add_argument("--label-a", default="22K 학습")
    ap.add_argument("--label-b", default="14K 학습")
    ap.add_argument("--max-samples", type=int, default=2000)
    args = ap.parse_args()

    items = load_items(Path(args.data).expanduser())
    if not items:
        raise SystemExit(f"이미지를 찾지 못했습니다: {args.data}")

    # 학습에 쓰인 분할과 동일한 시드로 테스트셋만 추출 (누수 방지)
    _, _, test = stratified_split(items, seed=DLC.RANDOM_SEED)
    if len(test) > args.max_samples:
        test = random.Random(0).sample(test, args.max_samples)
    gt = np.array([a for _, a, _, _ in test])
    print(f"테스트셋 {len(test):,}장  (전체 {len(items):,}장의 15% 홀드아웃)")
    print(f"정답 조향각 범위 {gt.min():+d} ~ {gt.max():+d}\n")

    results = {}
    for label, path in [(args.label_a, args.model_a), (args.label_b, args.model_b)]:
        print(f"추론 중: {label}  ({path})")
        results[label] = predict(load_model(path), test)

    # ---------- 구간별 통계 ----------
    print("\n" + "=" * 78)
    print("정답 조향각 구간별 예측 통계")
    print("=" * 78)
    header = f"{'구간':<18}{'표본':>6}{'정답평균':>10}"
    for lb in results:
        header += f"{lb+' 예측':>14}{'수축률':>9}"
    print(header)
    print("-" * 78)

    for name, lo, hi in BANDS:
        mask = (np.abs(gt) > lo) & (np.abs(gt) <= hi)
        n = mask.sum()
        if n < 5:
            continue
        gm = np.abs(gt[mask]).mean()
        row = f"{name:<18}{n:>6}{gm:>10.1f}"
        for lb, pr in results.items():
            pm = np.abs(pr[mask]).mean()
            row += f"{pm:>14.1f}{(1-pm/gm)*100:>8.1f}%"
        print(row)

    # ---------- 극단값 재현력 ----------
    print("\n" + "=" * 78)
    print("극단 조향 재현력 (전체 테스트셋)")
    print("=" * 78)
    print(f"{'지표':<24}{'정답':>10}", end="")
    for lb in results: print(f"{lb:>14}", end="")
    print()
    print("-" * 78)
    for stat, fn in [("최대 |θ|", lambda v: np.abs(v).max()),
                     ("99 백분위 |θ|", lambda v: np.percentile(np.abs(v), 99)),
                     ("95 백분위 |θ|", lambda v: np.percentile(np.abs(v), 95)),
                     ("표준편차", lambda v: v.std()),
                     ("|θ|>40 비율(%)", lambda v: (np.abs(v) > 40).mean()*100)]:
        print(f"{stat:<24}{fn(gt):>10.1f}", end="")
        for lb, pr in results.items(): print(f"{fn(pr):>14.1f}", end="")
        print()

    # ---------- 급커브 RMSE ----------
    print("\n" + "=" * 78)
    print("급커브(|정답|>40°) 구간 오차")
    print("=" * 78)
    sharp = np.abs(gt) > 40
    print(f"  표본 {sharp.sum()}장")
    for lb, pr in results.items():
        e = pr[sharp] - gt[sharp]
        print(f"  {lb:<12} RMSE {np.sqrt((e**2).mean()):6.2f}°   "
              f"평균 편향 {e.mean():+6.2f}°  (음수=덜 꺾음)")

    # ---------- 그림 ----------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
    bins = np.arange(-100, 105, 5)

    axes[0].hist(gt, bins=bins, alpha=.75, color="gray", label="Ground Truth")
    for lb, pr in results.items():
        axes[0].hist(pr, bins=bins, histtype="step", lw=2, label=lb)
    axes[0].set_title("Steering angle distribution (all)")
    axes[0].set_xlabel("Steering angle (deg)"); axes[0].legend(fontsize=8)

    axes[1].hist(gt[sharp], bins=bins, alpha=.75, color="gray", label="Ground Truth")
    for lb, pr in results.items():
        axes[1].hist(pr[sharp], bins=bins, histtype="step", lw=2, label=lb)
    axes[1].set_title("Sharp curves only (|GT| > 40°)")
    axes[1].set_xlabel("Steering angle (deg)"); axes[1].legend(fontsize=8)

    for lb, pr in results.items():
        axes[2].scatter(gt, pr, s=3, alpha=.25, label=lb)
    lim = max(abs(gt).max(), 100)
    axes[2].plot([-lim, lim], [-lim, lim], "k--", lw=1, label="ideal")
    axes[2].set_xlabel("Ground truth (deg)"); axes[2].set_ylabel("Prediction (deg)")
    axes[2].set_title("Prediction vs Ground truth")
    axes[2].legend(fontsize=8); axes[2].grid(alpha=.3)

    plt.tight_layout()
    plt.savefig("prediction_distribution.png", dpi=140)
    print("\n[OK] prediction_distribution.png 저장")


if __name__ == "__main__":
    main()
