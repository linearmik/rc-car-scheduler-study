"""
train_experiment.py — 스케줄러 실험 1회 실행
=============================================
스케줄러 / 데이터 규모 / 시드를 인자로 받아 학습 1회를 수행하고
결과를 JSON + CSV로 저장한다.

사용법:
    python3 train_experiment.py \
        --data-dir  ~/aicar/merged \
        --scheduler warmup_cosine \
        --data-size 3000 \
        --seed      42 \
        --out-dir   ~/aicar/results

OneCycle max_lr 통제 실험:
    python3 train_experiment.py ... --scheduler onecycle \
        --onecycle-max-lr-ratio 1.0 --out-dir ~/aicar/results_oc1x

필요 파일 (같은 폴더에):
    config.py        (프로젝트 공통 설정)
    schedulers.py    (스케줄러 4종 모듈)

출력:
    <out-dir>/{scheduler}_{size}_{seed}.json     최종 지표
    <out-dir>/{scheduler}_{size}_{seed}_hist.csv 에폭별 이력
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path
import re
import os
from collections import Counter
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent))
from config import AppConfig, DLConfig, parse_image_filename, get_angle_group
from schedulers import build_scheduler, GradientNormTracker

APP = AppConfig()
DLC = DLConfig()


# ============================================================
# 장치 선택 (CUDA > MPS > CPU)
# ============================================================
def pick_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0"), True, True
    if torch.backends.mps.is_available():
        return torch.device("mps"), False, False   # MPS는 pin_memory 미지원
    return torch.device("cpu"), False, False


# ============================================================
# 재현성
# ============================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 모델 — NVIDIA PilotNet 다중 출력 회귀
#   입력 66x200x3 (YUV) → 출력 2 [steering, speed] (모두 /99 정규화)
# ============================================================
class NvidiaModel(nn.Module):
    def __init__(self, output_dim: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Dropout(0.2),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 1 * 18, 100), nn.ELU(),
            nn.Linear(100, 50), nn.ELU(),
            nn.Linear(50, 10), nn.ELU(),
            nn.Linear(10, output_dim),
        )

    def forward(self, x):
        return self.regressor(self.features(x))


# ============================================================
# 손실 — 조향 0.9 / 속도 0.1 가중 MSE
# ============================================================
def multitask_regression_loss(output, target,
                              w_angle=DLC.STEERING_LOSS_WEIGHT,
                              w_speed=DLC.SPEED_LOSS_WEIGHT):
    angle_loss = nn.functional.mse_loss(output[:, 0], target[:, 0])
    speed_loss = nn.functional.mse_loss(output[:, 1], target[:, 1])
    return w_angle * angle_loss + w_speed * speed_loss, angle_loss, speed_loss


def constant_predictor_loss(train_items, valid_items):
    """학습셋 평균만 출력하는 상수 예측기의 검증 손실.

    발산 판정의 기준선. 발산한 시행의 손실이 이 값 근처에서 평평해지면
    '모델이 붕괴해 평균만 출력하는 상태'임을 수치로 보일 수 있다.
    """
    tr = np.array([[a / DLC.LABEL_SCALE, s / DLC.LABEL_SCALE]
                   for _, a, s, _ in train_items], dtype=np.float64)
    va = np.array([[a / DLC.LABEL_SCALE, s / DLC.LABEL_SCALE]
                   for _, a, s, _ in valid_items], dtype=np.float64)
    if len(tr) == 0 or len(va) == 0:
        return None
    mu = tr.mean(axis=0)
    ang = float(np.mean((va[:, 0] - mu[0]) ** 2))
    spd = float(np.mean((va[:, 1] - mu[1]) ** 2))
    return DLC.STEERING_LOSS_WEIGHT * ang + DLC.SPEED_LOSS_WEIGHT * spd


# ============================================================
# 데이터셋 — 파일 경로 리스트를 받아 lazy 로딩
# ============================================================
class DrivingDataset(Dataset):
    def __init__(self, items):
        # items: [(path, angle, speed), ...]
        self.items = items

    def __len__(self):
        return len(self.items)

    def _preprocess(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(path)
        h, w = img.shape[:2]
        # gi.py가 이미 ROI 크롭 + 리사이즈해서 저장한 경우(66x200)는 그대로 사용.
        # 원본 카메라 프레임(640x480 등)이면 ROI를 잘라낸 뒤 리사이즈한다.
        if (h, w) != (APP.MODEL_INPUT_HEIGHT, APP.MODEL_INPUT_WIDTH):
            if h > APP.ROI_YD:
                img = img[APP.ROI_YU:APP.ROI_YD, APP.ROI_XL:APP.ROI_XR]
            img = cv2.resize(img, (APP.MODEL_INPUT_WIDTH, APP.MODEL_INPUT_HEIGHT))
        # [수정] gi.py 저장 이미지는 이미 YUV다. 여기서 BGR2YUV를 다시 적용하면 색공간이
        # 뒤틀려(BGR2YUV(YUV)), 배포(ar_linear: 카메라 BGR→BGR2YUV=정상 YUV)와 학습 입력이
        # 불일치한다. dl_research.py는 변환 없이 YUV를 그대로 학습하므로, 여기서도 추가 변환
        # 없이 저장 YUV를 그대로 쓴다. 이러면 학습=추론 색공간이 일치해 배포 모델이 정상 주행.
        #   (이전 버그: img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV))
        img = cv2.GaussianBlur(img, APP.GAUSSIAN_BLUR_KERNEL, 0)
        img = img.astype(np.float32) / APP.NORMALIZE_DIVISOR
        return np.transpose(img, (2, 0, 1))          # HWC → CHW

    def __getitem__(self, idx):
        path, angle, speed = self.items[idx]
        x = torch.from_numpy(self._preprocess(path))
        y = torch.tensor([angle / DLC.LABEL_SCALE,
                          speed / DLC.LABEL_SCALE], dtype=torch.float32)
        return x, y


# ============================================================
# 데이터 로딩 & 층화 분할
# ============================================================
def load_items(data_dir: Path):
    """폴더의 모든 이미지를 (path, angle, speed, group)으로 파싱."""
    items = []
    for p in sorted(data_dir.rglob("*.png")):
        lb = parse_image_filename(p)
        if lb is None:
            continue
        items.append((p, lb.angle, lb.speed, get_angle_group(lb.angle)))
    return items


def _frame_num(path):
    """P0123_070_+03.png -> 123"""
    m = re.match(r'P(\d+)_', os.path.basename(str(path)))
    return int(m.group(1)) if m else -1


def stratified_split(items, seed, valid_rate=0.15, test_rate=0.15, block_size=100):
    """블록 단위 층화 분할 — 인접 프레임 누수 방지.

    프레임 번호순으로 block_size장씩 묶은 뒤, 각 블록의 대표 조향각 그룹
    (최빈값) 기준으로 그룹별 70/15/15 배분한다. 블록은 통째로 한 split에만
    속하므로 연속 프레임이 train과 valid로 쪼개지지 않는다.

    주의: _frame_num이 -1을 반환하는 파일(P<숫자>_ 패턴 불일치)은 정렬 앞쪽에
    임의 순서로 몰려 블록 구조가 깨진다. 개수를 세어 경고한다.
    """
    rng = random.Random(seed)

    n_unparsed = sum(1 for it in items if _frame_num(it[0]) < 0)
    if n_unparsed:
        print(f"  [WARN] 프레임 번호 파싱 실패 {n_unparsed:,}장 "
              f"({n_unparsed / len(items) * 100:.1f}%) — 블록 분할 무결성 확인 필요")

    ordered = sorted(items, key=lambda it: _frame_num(it[0]))
    blocks = [ordered[i:i + block_size] for i in range(0, len(ordered), block_size)]

    by_group = {}
    for blk in blocks:
        rep = Counter(it[3] for it in blk).most_common(1)[0][0]
        by_group.setdefault(rep, []).append(blk)

    train, valid, test = [], [], []
    for g, blks in by_group.items():
        blks = blks[:]
        rng.shuffle(blks)
        n = len(blks)
        if n >= 3:
            n_test = max(1, int(round(n * test_rate)))
            n_valid = max(1, int(round(n * valid_rate)))
        else:
            n_test = n_valid = 0          # 블록이 너무 적은 그룹은 전부 train
        for blk in blks[:n_test]:
            test.extend(blk)
        for blk in blks[n_test:n_test + n_valid]:
            valid.extend(blk)
        for blk in blks[n_test + n_valid:]:
            train.extend(blk)
    rng.shuffle(train)
    return train, valid, test


def subsample(train, size, seed):
    """학습 세트에서 size장을 그룹 비율 유지하며 뽑는다. size<=0이면 전체."""
    if size <= 0 or size >= len(train):
        return train
    rng = random.Random(seed)
    by_group = {}
    for it in train:
        by_group.setdefault(it[3], []).append(it)

    total = len(train)
    picked = []
    for g, lst in by_group.items():
        lst = lst[:]
        rng.shuffle(lst)
        k = max(1, round(size * len(lst) / total))
        picked.extend(lst[:k])
    rng.shuffle(picked)
    return picked[:size]


# ============================================================
# EarlyStopping
# ============================================================
class EarlyStopping:
    def __init__(self, patience, min_delta=1e-7):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.counter = 0
        self.best_state = None
        self.stop = False
        self.best_epoch = 0

    def step(self, valid_loss, model, epoch):
        if valid_loss < self.best - self.min_delta:
            self.best = valid_loss
            self.counter = 0
            self.best_epoch = epoch
            self.best_state = {k: v.detach().cpu().clone()
                               for k, v in model.state_dict().items()}
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


# ============================================================
# 학습 1회
# ============================================================
def run(args):
    device, pin_mem, non_block = pick_device()
    set_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser()
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.scheduler}_{args.data_size}_{args.seed}"
    json_path = out_dir / f"{tag}.json"
    csv_path = out_dir / f"{tag}_hist.csv"

    if json_path.exists() and not args.force:
        print(f"[SKIP] 이미 완료: {tag}")
        return

    # --- 데이터 준비 ---
    all_items = load_items(data_dir)
    if not all_items:
        raise SystemExit(f"이미지를 찾지 못했습니다: {data_dir}")

    # 분할 시드는 고정 → 모든 실험이 동일한 valid/test를 공유(공정 비교)
    train_all, valid_set, test_set = stratified_split(all_items, seed=DLC.RANDOM_SEED)
    train_set = subsample(train_all, args.data_size, seed=args.seed)

    baseline_loss = constant_predictor_loss(train_set, valid_set)

    print(f"[{tag}] device={device} | train={len(train_set):,} "
          f"valid={len(valid_set):,} test={len(test_set):,}")
    if baseline_loss is not None:
        print(f"  상수 예측기(평균) 검증 손실 = {baseline_loss:.6f}  ← 붕괴 기준선")

    def loader(items, bs, shuffle):
        g = torch.Generator(); g.manual_seed(args.seed)
        ds = DrivingDataset([(p, a, s) for p, a, s, _ in items])
        return DataLoader(ds, batch_size=min(bs, len(ds)), shuffle=shuffle,
                          num_workers=args.workers, pin_memory=pin_mem,
                          generator=g if shuffle else None)

    train_loader = loader(train_set, args.batch_size, True)
    valid_loader = loader(valid_set, args.batch_size, False)

    steps_per_epoch = len(train_loader)

    # --- 모델 / 옵티마이저 / 스케줄러 ---
    model = NvidiaModel(DLC.MODEL_OUTPUT_DIM).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, amsgrad=True)
    scheduler = build_scheduler(
        name=args.scheduler, optimizer=optimizer,
        total_epochs=args.epochs, steps_per_epoch=steps_per_epoch,
        base_lr=args.lr,
        onecycle_max_lr_ratio=args.onecycle_max_lr_ratio,
    )
    stopper = EarlyStopping(args.patience)
    grad_tracker = GradientNormTracker()

    # OneCycle의 실효 max_lr을 명시적으로 기록 (논문 실험설정 절에 필요)
    effective_max_lr = (args.lr * args.onecycle_max_lr_ratio
                        if args.scheduler == "onecycle" else args.lr)
    print(f"  scheduler={args.scheduler} base_lr={args.lr:.2e} "
          f"max_lr={effective_max_lr:.2e} steps/epoch={steps_per_epoch}")

    # --- 발산 추적 ---
    # NaN이 아닌 발산은 기존 isfinite 검사로 잡히지 않는다. 검증 손실이
    # 그때까지의 best 대비 급등하면 그 에폭을 기록한다.
    diverged_epoch = None
    diverged_lr = None
    peak_valid_loss = 0.0

    history = []
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        # ---- train ----
        model.train()
        sums = np.zeros(3); n_seen = 0
        lr_now = float(optimizer.param_groups[0]["lr"])

        for x, y in train_loader:
            x = x.to(device, non_blocking=non_block)
            y = y.to(device, non_blocking=non_block)
            optimizer.zero_grad(set_to_none=True)
            out = model(x)
            total, ang, spd = multitask_regression_loss(out, y)
            if not torch.isfinite(total):
                raise RuntimeError(f"{tag} epoch {epoch}: non-finite loss")
            total.backward()
            grad_tracker.record_step(model)     # 클리핑 이전 노름 (발산 진단용)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            if scheduler.step_per_batch:
                scheduler.step()
            bn = x.shape[0]; n_seen += bn
            sums += np.array([total.item(), ang.item(), spd.item()]) * bn

        tr_total, tr_ang, tr_spd = sums / n_seen
        grad_tracker.end_epoch()

        # ---- valid ----
        model.eval()
        sums = np.zeros(3); n_seen = 0
        with torch.inference_mode():
            for x, y in valid_loader:
                x = x.to(device, non_blocking=non_block)
                y = y.to(device, non_blocking=non_block)
                out = model(x)
                total, ang, spd = multitask_regression_loss(out, y)
                bn = x.shape[0]; n_seen += bn
                sums += np.array([total.item(), ang.item(), spd.item()]) * bn

        va_total, va_ang, va_spd = sums / n_seen
        peak_valid_loss = max(peak_valid_loss, va_total)

        # ---- 발산 판정 (기록만; 기본적으로 학습은 계속) ----
        if (diverged_epoch is None
                and np.isfinite(stopper.best)
                and va_total > args.diverge_factor * stopper.best):
            diverged_epoch = epoch
            diverged_lr = lr_now
            print(f"  [DIVERGE] epoch {epoch}: valid {stopper.best:.6f} "
                  f"→ {va_total:.6f} ({va_total / stopper.best:.1f}x) at lr={lr_now:.2e}")

        if not scheduler.step_per_batch:
            scheduler.step(va_total if scheduler.needs_metric else None)

        history.append(dict(epoch=epoch, lr=lr_now,
                            train_total=tr_total, valid_total=va_total,
                            train_angle=tr_ang, valid_angle=va_ang,
                            train_speed=tr_spd, valid_speed=va_spd,
                            grad_norm=grad_tracker.epoch_avg_norms[-1]))

        stopper.step(va_total, model, epoch)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  [{epoch:3d}/{args.epochs}] train={tr_total:.6f} "
                  f"valid={va_total:.6f} lr={lr_now:.2e}")

        if diverged_epoch is not None and args.abort_on_diverge:
            print(f"  발산으로 중단 @ epoch {epoch} (best={stopper.best:.6f})")
            break
        if stopper.stop:
            print(f"  early stop @ epoch {epoch} (best={stopper.best:.6f})")
            break

    elapsed = time.time() - t0

    # --- best 모델로 최종 지표 (라벨 단위 RMSE) ---
    if stopper.best_state:
        model.load_state_dict(stopper.best_state)
    model.eval()
    P, T = [], []
    with torch.inference_mode():
        for x, y in valid_loader:
            P.append(model(x.to(device)).cpu().numpy())
            T.append(y.numpy())
    P = np.concatenate(P) * DLC.LABEL_SCALE
    T = np.concatenate(T) * DLC.LABEL_SCALE
    steer_rmse = float(np.sqrt(np.mean((P[:, 0] - T[:, 0]) ** 2)))
    speed_rmse = float(np.sqrt(np.mean((P[:, 1] - T[:, 1]) ** 2)))

    # 종료 시점 손실이 상수 예측기 수준이면 '붕괴 상태로 종료'로 표시
    final_valid = history[-1]["valid_total"] if history else float("nan")
    collapsed = bool(baseline_loss is not None
                     and np.isfinite(final_valid)
                     and abs(final_valid - baseline_loss) <= 0.25 * baseline_loss)

    result = dict(
        scheduler=args.scheduler, data_size=args.data_size, seed=args.seed,
        train_images=len(train_set), valid_images=len(valid_set),
        best_valid_loss=stopper.best, best_epoch=stopper.best_epoch,
        epochs_run=len(history),
        # 주의: LABEL_SCALE(=99) 단위. gi.py가 기록한 조향 명령값이며
        # 물리적 각도(°)로 환산된 값이 아니다. 논문 표기 시 단위 확인 필요.
        steering_rmse_label=steer_rmse,
        steering_rmse_deg=steer_rmse,          # 기존 분석 스크립트 호환용 별칭
        label_unit=f"steering command, |v| <= {int(DLC.LABEL_SCALE)} (not degrees)",
        speed_rmse=speed_rmse, elapsed_sec=round(elapsed, 1),
        device=str(device), lr=args.lr, batch_size=args.batch_size,
        # --- 아래부터 추가 필드 ---
        steps_per_epoch=steps_per_epoch,
        onecycle_max_lr_ratio=args.onecycle_max_lr_ratio,
        effective_max_lr=effective_max_lr,
        patience=args.patience,
        grad_clip=args.grad_clip,
        baseline_constant_loss=baseline_loss,
        diverged=diverged_epoch is not None,
        diverged_epoch=diverged_epoch,
        diverged_lr=diverged_lr,
        peak_valid_loss=peak_valid_loss,
        final_valid_loss=final_valid,
        collapsed_at_end=collapsed,
        early_epoch_grad_norms=grad_tracker.epoch_avg_norms[:10],
    )

    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader(); w.writerows(history)

    if args.save_model:
        torch.save(stopper.best_state, out_dir / f"{tag}.pt")

    flag = ""
    if diverged_epoch is not None:
        flag = f" | DIVERGED@e{diverged_epoch}" + (" COLLAPSED" if collapsed else " (회복)")
    print(f"[DONE] {tag} | valid={stopper.best:.6f} "
          f"steer_rmse={steer_rmse:.2f} speed_rmse={speed_rmse:.2f} "
          f"| {elapsed/60:.1f}분{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--scheduler", required=True,
                    choices=["plateau", "cosine", "warmup_cosine", "onecycle"])
    ap.add_argument("--data-size", type=int, required=True,
                    help="학습 서브셋 크기 (0 또는 -1이면 전체)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--save-model", action="store_true")
    ap.add_argument("--force", action="store_true", help="이미 있어도 재실행")
    # --- 추가 인자 ---
    ap.add_argument("--onecycle-max-lr-ratio", type=float, default=10.0,
                    help="OneCycle max_lr = lr x ratio. 기존 실험 재현은 10.0, "
                         "동일 최대 LR 통제 실험은 1.0")
    ap.add_argument("--diverge-factor", type=float, default=5.0,
                    help="검증 손실이 best 대비 이 배수를 넘으면 발산으로 기록")
    ap.add_argument("--abort-on-diverge", action="store_true",
                    help="발산 감지 시 즉시 중단 (기본: 기록만 하고 계속)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()