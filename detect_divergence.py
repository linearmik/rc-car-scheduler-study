"""
발산(학습 붕괴) 판정 — [문제 A]
========================================================================
OneCycleLR은 onecycle_max_lr_ratio=10 때문에 max_lr = 5e-4 x 10 = 5e-3 로
학습된다(다른 3종은 5e-4). 이 고LR 구간에서 모델이 상수 예측기로 붕괴하며
검증 손실이 튄 뒤 회복하지 못하고 고정된다.

문제점: JSON의 best_valid_loss는 '붕괴 직전' 최저값이라, JSON만 보면 정상으로
보인다. 기존 torch.isfinite 검사는 NaN만 잡아 이 붕괴를 통과시켰다.

판정 기준(데이터 비의존, hist.csv만으로):
    발산 = (최종 검증손실 > best 검증손실 x DIVERGE_RATIO)
    → best 이후 손실이 튄 뒤 마지막까지 회복하지 못한 런.
    초기 에폭의 일시적 고손실은 회복하므로 final 기준이면 오탐하지 않는다.

사용법:
    python3 detect_divergence.py                 # results_22k, results_14k 둘 다
    python3 detect_divergence.py results_22k     # 지정 폴더만
"""
import csv, glob, os, sys
from collections import defaultdict

SCH = ["plateau", "cosine", "warmup_cosine", "onecycle"]
LB = {"plateau": "Plateau", "cosine": "Cosine",
      "warmup_cosine": "Warmup+Cosine", "onecycle": "OneCycleLR"}
SEEDS = [42, 123, 456, 789, 1024]

DIVERGE_RATIO = 3.0   # 최종손실이 best의 3배를 넘으면 회복 실패(붕괴)로 본다


def load_run(path):
    """hist.csv → (best, best_epoch, final, max, lr_max, n_epochs)."""
    ep, lr, vt = [], [], []
    with open(path) as f:
        for r in csv.DictReader(f):
            try:
                ep.append(int(float(r["epoch"])))
                lr.append(float(r["lr"]))
                vt.append(float(r["valid_total"]))
            except (KeyError, ValueError):
                pass
    if not vt:
        return None
    best = min(vt)
    bi = vt.index(best)
    return dict(best=best, best_epoch=ep[bi], final=vt[-1],
               vmax=max(vt), lr_max=max(lr), n=len(vt))


def load_dir(dirname):
    """{(scheduler, size, seed): run_dict}."""
    out = {}
    for path in glob.glob(os.path.join(dirname, "*_hist.csv")):
        stem = os.path.basename(path)[:-len("_hist.csv")]
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        sch, size, seed = parts[0], int(parts[1]), int(parts[2])
        run = load_run(path)
        if run:
            out[(sch, size, seed)] = run
    return out


def diverged(run):
    return run["final"] > run["best"] * DIVERGE_RATIO


def report(dirname):
    H = load_dir(dirname)
    if not H:
        print(f"[{dirname}] *_hist.csv 를 찾지 못했습니다.\n")
        return
    sizes = sorted({k[1] for k in H}, key=lambda s: (s == 0, s))
    size_lb = {s: ("전체" if s == 0 else f"{s // 1000}K" if s >= 1000 else str(s))
               for s in sizes}

    print("=" * 78)
    print(f"[{dirname}] 발산 판정  (기준: final > best x {DIVERGE_RATIO})")
    print("=" * 78)

    # 1) OneCycle 런별 상세
    print("\nOneCycleLR 런별 (best는 JSON에 기록된 값 = 붕괴 직전 최저)")
    print(f"{'규모':<6}{'seed':>6}{'best':>10}{'final':>10}{'배율':>8}{'lr_max':>10}  판정")
    print("-" * 78)
    for s in sizes:
        for sd in SEEDS:
            run = H.get(("onecycle", s, sd))
            if not run:
                continue
            ratio = run["final"] / run["best"]
            verdict = "발산" if diverged(run) else "정상"
            print(f"{size_lb[s]:<6}{sd:>6}{run['best']:>10.4f}{run['final']:>10.4f}"
                  f"{ratio:>7.1f}x{run['lr_max']:>10.1e}  {verdict}")
        print("-" * 78)

    # 2) 스케줄러 x 규모 발산률
    print("\n발산률 (발산 시드 수 / 전체 시드 수)")
    print(f"{'규모':<6}" + "".join(f"{LB[s]:>16}" for s in SCH))
    print("-" * 78)
    grid = defaultdict(lambda: [0, 0])   # (sch,size) -> [diverged, total]
    for (sch, size, sd), run in H.items():
        grid[(sch, size)][1] += 1
        if diverged(run):
            grid[(sch, size)][0] += 1
    for s in sizes:
        row = f"{size_lb[s]:<6}"
        for sch in SCH:
            d, t = grid[(sch, s)]
            row += f"{(f'{d}/{t}' if t else '-'):>16}"
        print(row)

    # 3) 논지 1/2 보호 확인: 전통 3종은 발산 0이어야 함
    trad_div = sum(grid[(sch, s)][0] for sch in SCH[:3] for s in sizes)
    oc_div = sum(grid[("onecycle", s)][0] for s in sizes)
    oc_tot = sum(grid[("onecycle", s)][1] for s in sizes)
    print("-" * 78)
    print(f"전통 3종 발산 합계: {trad_div}건  (0이어야 논지 1·2가 발산의 영향 없음)")
    print(f"OneCycle 발산 합계: {oc_div}/{oc_tot}건")
    print()


if __name__ == "__main__":
    dirs = sys.argv[1:] or ["results_22k", "results_14k"]
    for d in dirs:
        report(d)
