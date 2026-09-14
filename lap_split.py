# lap_split.py
import re, numpy as np

def time_order(files):
    """파일명 숫자 기준 시간순 인덱스"""
    def key(f):
        m = re.findall(r'\d+', f)
        return int(m[-1]) if m else -1     # 보통 마지막 숫자가 순번
    order = sorted(range(len(files)), key=lambda i: key(files[i]))
    return np.array(order)

def blocked_split(n, n_blocks=10, val_blocks=(2, 7), embargo=60):
    edges = np.linspace(0, n, n_blocks + 1).astype(int)
    val = np.zeros(n, bool); drop = np.zeros(n, bool)
    for b in val_blocks:
        s, e = edges[b], edges[b+1]
        val[s:e] = True
        drop[max(0,s-embargo):s] = True
        drop[e:min(n,e+embargo)] = True
    return np.where(~val & ~drop)[0], np.where(val)[0]

def check(steer, tr, va):
    for name, idx in [("train", tr), ("val", va)]:
        th = np.asarray(steer)[idx]
        print(f"{name:5s} n={len(idx):6d} "
              f"급회전%={np.mean(np.abs(th)>40)*100:5.1f} "
              f"좌우비={(th>0).sum()/max((th<0).sum(),1):.2f}")