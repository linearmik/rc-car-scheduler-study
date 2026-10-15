"""
그래디언트 동역학 분석
======================
results/*_hist.csv 의 grad_norm 열을 읽어 스케줄러별 학습 동역학을 비교한다.

사용법:
    cd ~/Desktop/aicar
    python3 analyze_gradients.py

출력:
    - 스케줄러별 초기/최대/평균 그래디언트 노름
    - 학습률과 그래디언트 노름의 상관
    - grad_dynamics.png (논문 그림용)
"""
import csv, glob, json, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCH   = ["plateau", "cosine", "warmup_cosine", "onecycle"]
LABEL = {"plateau":"Plateau","cosine":"Cosine",
         "warmup_cosine":"Warmup+Cosine","onecycle":"OneCycleLR"}
SIZES = [(1000,"1K"),(3000,"3K"),(10000,"10K"),(0,"22K")]

def load(dirname="results"):
    """{(scheduler, size, seed): [{epoch, lr, grad_norm, valid_total}, ...]}"""
    out = {}
    for path in glob.glob(os.path.join(dirname, "*_hist.csv")):
        stem = os.path.basename(path)[:-len("_hist.csv")]
        parts = stem.rsplit("_", 2)
        if len(parts) != 3:
            continue
        sch, size, seed = parts[0], int(parts[1]), int(parts[2])
        rows = []
        with open(path) as f:
            for r in csv.DictReader(f):
                try:
                    rows.append(dict(epoch=int(float(r["epoch"])),
                                     lr=float(r["lr"]),
                                     grad=float(r["grad_norm"]),
                                     valid=float(r["valid_total"])))
                except (KeyError, ValueError):
                    pass
        if rows:
            out[(sch, size, seed)] = rows
    return out


def summarize(H):
    print("=" * 78)
    print("그래디언트 L2-Norm 요약 (5-Seed 평균)")
    print("=" * 78)
    print(f"{'데이터':<6}{'스케줄러':<16}{'1에폭':>10}{'초기10평균':>12}"
          f"{'최대':>10}{'최대시점':>10}{'전체평균':>10}")
    print("-" * 78)
    for size, lb in SIZES:
        for sch in SCH:
            runs = [v for k, v in H.items() if k[0] == sch and k[1] == size]
            if not runs:
                continue
            first  = np.mean([r[0]["grad"] for r in runs])
            early  = np.mean([np.mean([x["grad"] for x in r[:10]]) for r in runs])
            peaks  = [max(x["grad"] for x in r) for r in runs]
            peak_e = [int(np.argmax([x["grad"] for x in r])) + 1 for r in runs]
            allm   = np.mean([np.mean([x["grad"] for x in r]) for r in runs])
            print(f"{lb:<6}{LABEL[sch]:<16}{first:>10.4f}{early:>12.4f}"
                  f"{np.mean(peaks):>10.4f}{np.mean(peak_e):>9.1f}ep{allm:>10.4f}")
        print()


def warmup_effect(H):
    """Warmup이 초기 그래디언트 스파이크를 실제로 억제했는가."""
    print("=" * 78)
    print("Warmup의 초기 안정화 효과 검증")
    print("=" * 78)
    print(f"{'데이터':<6}{'Warmup 초기5':>14}{'Cosine 초기5':>14}{'차이':>10}")
    print("-" * 78)
    for size, lb in SIZES:
        w = [np.mean([x["grad"] for x in r[:5]])
             for k, r in H.items() if k[0] == "warmup_cosine" and k[1] == size]
        c = [np.mean([x["grad"] for x in r[:5]])
             for k, r in H.items() if k[0] == "cosine" and k[1] == size]
        if not w or not c:
            continue
        wm, cm = np.mean(w), np.mean(c)
        print(f"{lb:<6}{wm:>14.4f}{cm:>14.4f}{(wm-cm)/cm*100:>+9.1f}%")
    print("\n  음수면 Warmup이 초기 그래디언트를 억제했다는 뜻")


def lr_grad_corr(H):
    """학습률과 그래디언트 노름의 관계 — OneCycle 고LR 구간 확인."""
    print("\n" + "=" * 78)
    print("학습률 대비 그래디언트 노름 (OneCycleLR, 22K)")
    print("=" * 78)
    runs = [v for k, v in H.items() if k[0] == "onecycle" and k[1] == 0]
    if not runs:
        print("  데이터 없음")
        return
    n = min(len(r) for r in runs)
    lrs  = np.mean([[x["lr"]   for x in r[:n]] for r in runs], axis=0)
    grds = np.mean([[x["grad"] for x in r[:n]] for r in runs], axis=0)
    print(f"{'에폭':>6}{'학습률':>12}{'grad_norm':>12}")
    print("-" * 78)
    for e in range(0, n, max(1, n // 12)):
        print(f"{e+1:>6}{lrs[e]:>12.2e}{grds[e]:>12.4f}")
    if n > 3:
        r = np.corrcoef(lrs, grds)[0, 1]
        print(f"\n  학습률-그래디언트 상관계수 r = {r:+.3f}")


def draw(H, out="grad_dynamics.png"):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    for ax, (size, lb) in zip(axes.ravel(), SIZES):
        for sch in SCH:
            runs = [v for k, v in H.items() if k[0] == sch and k[1] == size]
            if not runs:
                continue
            n = min(len(r) for r in runs)
            if n < 2:
                continue
            g = np.mean([[x["grad"] for x in r[:n]] for r in runs], axis=0)
            ax.plot(range(1, n + 1), g, label=LABEL[sch], lw=1.6)
        ax.set_title(f"{lb} dataset")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient L2-Norm")
        ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8)
    plt.suptitle("Gradient Norm Dynamics by Scheduler (5-seed mean)")
    plt.tight_layout()
    plt.savefig(out, dpi=140)
    print(f"\n[OK] 그래프 저장: {out}")


if __name__ == "__main__":
    H = load("results_22k")
    Hf = load("results_full")
    for k, v in Hf.items():                       # OneCycle 완주본으로 교체
        if k[0] == "onecycle":
            H[k] = v
    print(f"불러온 학습 이력: {len(H)}건\n")
    if not H:
        raise SystemExit("results_22k/*_hist.csv 를 찾지 못했습니다.")
    summarize(H)
    warmup_effect(H)
    lr_grad_corr(H)
    draw(H)
