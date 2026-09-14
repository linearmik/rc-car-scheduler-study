"""
스케줄러 통계 재분석 — 논지 1·2 (정상 전처리 데이터)
========================================================================
전통 3종(plateau/cosine/warmup_cosine)의 통계적 비구별성과 등가성,
데이터 규모 효과를 재산출한다. OneCycle은 발산(문제 A)이 있어 등가성
비교에서 제외하고 별도로 다룬다.

검정:
  - 대응표본 t-검정(paired, 시드로 페어링). Welch 아님. (§12 규칙)
  - TOST 등가성 검정, ±5% 마진.
  - Cohen's dz (대응 효과크기).
  - 규모 효과: 최소규모→전체 손실 감소율.

사용법: python3 stats_reanalysis.py
"""
import json, glob, itertools
import numpy as np
from scipy import stats

TRAD = ["plateau", "cosine", "warmup_cosine"]
LB = {"plateau": "Plateau", "cosine": "Cosine", "warmup_cosine": "Warmup"}
TOST_MARGIN = 0.05      # ±5%


def load(d):
    """{(sch,size): {seed: best_valid_loss}}"""
    R = {}
    for f in glob.glob(d + "/*.json"):
        j = json.load(open(f))
        R.setdefault((j["scheduler"], j["data_size"]), {})[j["seed"]] = j["best_valid_loss"]
    return R


def paired(a_map, b_map):
    """공통 시드로 대응표본 배열 반환."""
    seeds = sorted(set(a_map) & set(b_map))
    a = np.array([a_map[s] for s in seeds])
    b = np.array([b_map[s] for s in seeds])
    return a, b


def tost_paired(a, b, margin_rel):
    """대응표본 TOST. margin은 두 평균 평균의 상대비율. 등가면 p<0.05."""
    d = a - b
    n = len(d)
    md, sd = d.mean(), d.std(ddof=1)
    se = sd / np.sqrt(n)
    df = n - 1
    delta = margin_rel * np.mean([a.mean(), b.mean()])
    if se == 0:
        return 0.0 if abs(md) < delta else 1.0, delta
    t_lower = (md + delta) / se          # H1: md > -delta
    p_lower = 1 - stats.t.cdf(t_lower, df)
    t_upper = (md - delta) / se          # H1: md < +delta
    p_upper = stats.t.cdf(t_upper, df)
    return max(p_lower, p_upper), delta


def report(name, d):
    R = load(d)
    sizes = sorted({k[1] for k in R}, key=lambda s: (s == 0, s))
    slb = {s: ("전체" if s == 0 else f"{s//1000}K") for s in sizes}

    print("=" * 76)
    print(f"[{name}]  전통 3종 대응표본 비교 (시드 5개)")
    print("=" * 76)
    print(f"{'규모':<6}{'쌍':<20}{'평균A':>9}{'평균B':>9}{'차이%':>8}{'paired p':>10}{'dz':>7}{'TOST p':>9}  등가")
    print("-" * 76)

    n_pairs = n_indist = n_equiv = 0
    dz_list = []
    for sz in sizes:
        for A, B in itertools.combinations(TRAD, 2):
            if (A, sz) not in R or (B, sz) not in R:
                continue
            a, b = paired(R[(A, sz)], R[(B, sz)])
            if len(a) < 2:
                continue
            t, p = stats.ttest_rel(a, b)
            dz = (a - b).mean() / (a - b).std(ddof=1) if (a - b).std(ddof=1) else 0.0
            ptost, delta = tost_paired(a, b, TOST_MARGIN)
            diff_pct = (a.mean() - b.mean()) / b.mean() * 100
            n_pairs += 1
            n_indist += (p > 0.05)
            n_equiv += (ptost < 0.05)
            dz_list.append(abs(dz))
            eq = "O" if ptost < 0.05 else "-"
            print(f"{slb[sz]:<6}{LB[A]+'-'+LB[B]:<20}{a.mean():>9.4f}{b.mean():>9.4f}"
                  f"{diff_pct:>+7.1f}%{p:>10.3f}{dz:>+7.2f}{ptost:>9.3f}   {eq}")
        print("-" * 76)

    print(f"쌍 총 {n_pairs}개 | paired p>0.05(구별안됨): {n_indist}/{n_pairs} | "
          f"TOST 등가(±5%): {n_equiv}/{n_pairs} | |dz| 중앙값 {np.median(dz_list):.3f}")

    # 규모 효과 (전통3종 평균)
    small = min(s for s in sizes if s > 0)
    tmean = lambda sz: np.mean([np.mean(list(R[(s, sz)].values())) for s in TRAD])
    red = (tmean(small) - tmean(0)) / tmean(small) * 100
    full_means = [np.mean(list(R[(s, 0)].values())) for s in TRAD]
    sched_spread = (max(full_means) - min(full_means)) / min(full_means) * 100
    print(f"규모 효과: {slb[small]}→전체 손실 {red:.1f}% 감소  vs  전체서 스케줄러 최대차 {sched_spread:.1f}%")
    print()


if __name__ == "__main__":
    for name, d in [("22K", "results_22k"), ("14K", "results_14k")]:
        report(name, d)
    print("주: OneCycle은 규모 의존적 발산(detect_divergence.py 참고)으로 등가 비교 제외.")
    print("   p값은 대응표본 t-검정. TOST 등가는 p<0.05일 때 ±5% 마진 내 동등 입증.")
