"""
그래디언트 스파이크 시드별 검증 + 공정 비교 그래프
====================================================
1) OneCycle 스파이크가 특정 시드의 우연인지 5시드 공통 현상인지 확인
2) 조기종료로 선 길이가 다른 문제를 보정한 논문용 그래프 생성

사용법: cd ~/Desktop/aicar && python3 check_spike.py
"""
import csv, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCH=["plateau","cosine","warmup_cosine","onecycle"]
LB={"plateau":"Plateau","cosine":"Cosine","warmup_cosine":"Warmup+Cosine","onecycle":"OneCycleLR"}
def discover_sizes(H):
    sizes = sorted({k[1] for k in H}, key=lambda s: (s == 0, s))
    return [(s, "전체" if s == 0 else f"{s//1000}K") for s in sizes]

SEEDS=[42,123,456,789,1024]

def load(d):
    out={}
    for p in glob.glob(os.path.join(d,"*_hist.csv")):
        s=os.path.basename(p)[:-len("_hist.csv")].rsplit("_",2)
        if len(s)!=3: continue
        rows=[]
        with open(p) as f:
            for r in csv.DictReader(f):
                try: rows.append((int(float(r["epoch"])),float(r["lr"]),float(r["grad_norm"])))
                except Exception: pass
        if rows: out[(s[0],int(s[1]),int(s[2]))]=rows
    return out

H=load("results_22k")
for k,v in load("results_full").items():
    if k[0]=="onecycle": H[k]=v

SIZES = discover_sizes(H)   # H를 채운 뒤에 규모 목록을 만든다

print("="*70)
print("OneCycleLR 최대 그래디언트 — 시드별 (평균이 아닌 개별값)")
print("="*70)
print(f"{'데이터':<6}" + "".join(f"{s:>10}" for s in SEEDS) + f"{'중앙값':>10}{'판정':>16}")
print("-"*70)
for sz,lb in SIZES:
    peaks=[]
    for sd in SEEDS:
        r=H.get(("onecycle",sz,sd))
        peaks.append(max(x[2] for x in r) if r else float('nan'))
    med=np.nanmedian(peaks); mx=np.nanmax(peaks); mn=np.nanmin(peaks)
    verdict = "5시드 공통" if mn > med*0.3 else "특정시드 편중"
    print(f"{lb:<6}" + "".join(f"{p:>10.2f}" for p in peaks) + f"{med:>10.2f}{verdict:>16}")

print("\n  최소값이 중앙값의 30% 이상이면 '공통 현상'으로 판정")

print("\n"+"="*70)
print("3종 최대 그래디언트 중앙값 (비교 기준)")
print("="*70)
for sz,lb in SIZES:
    vals=[]
    for s in SCH[:3]:
        for sd in SEEDS:
            r=H.get((s,sz,sd))
            if r: vals.append(max(x[2] for x in r))
    oc=[max(x[2] for x in H[("onecycle",sz,sd)]) for sd in SEEDS if ("onecycle",sz,sd) in H]
    print(f"  {lb:<5} 3종 중앙값 {np.median(vals):>7.3f} | OneCycle 중앙값 {np.median(oc):>7.3f}"
          f" | 배율 {np.median(oc)/np.median(vals):>5.1f}x")

# ---- 공정 비교 그래프 ----
fig,axes=plt.subplots(2,2,figsize=(13,9))
for ax,(sz,lb) in zip(axes.ravel(),SIZES):
    lens=[]
    for s in SCH:
        runs=[H[(s,sz,sd)] for sd in SEEDS if (s,sz,sd) in H]
        if runs: lens.append(min(len(r) for r in runs))
    cut=min(lens) if lens else 100          # 가장 짧은 조건에 맞춰 자름
    for s in SCH:
        runs=[H[(s,sz,sd)] for sd in SEEDS if (s,sz,sd) in H]
        if not runs: continue
        g=np.median([[x[2] for x in r[:cut]] for r in runs],axis=0)   # 평균 대신 중앙값
        ax.plot(range(1,cut+1),g,label=LB[s],lw=1.6)
    ax.set_title(f"{lb} dataset  (first {cut} epochs, all schedulers)")
    ax.set_xlabel("Epoch"); ax.set_ylabel("Gradient L2-Norm")
    ax.set_yscale("log"); ax.grid(alpha=.3); ax.legend(fontsize=8)
plt.suptitle("Gradient Norm Dynamics (5-seed median, equal epoch range)")
plt.tight_layout(); plt.savefig("grad_dynamics_fair.png",dpi=140)
print("\n[OK] grad_dynamics_fair.png 저장 (중앙값 + 동일 에폭 구간)")
