"""
저장 이미지 색공간 진단
=======================
merged/ 의 이미지가 BGR인지 YUV인지 판별하고,
각 해석에서 차선 검출이 가능한지 비교한다.

사용법: cd ~/Desktop/aicar && python3 diagnose_image.py
출력:   diagnose_color.png  (4가지 해석 비교)
"""
import glob, cv2, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

files = sorted(glob.glob("merged/*.png"))[:4]
if not files:
    raise SystemExit("merged/ 에 이미지가 없습니다")

print("="*66)
print("채널별 통계 (BGR로 읽었을 때)")
print("="*66)
for f in files[:2]:
    img = cv2.imread(f)
    print(f"\n{f.split('/')[-1]}  shape={img.shape}")
    for i, c in enumerate("BGR"):
        ch = img[:, :, i]
        print(f"  {c}: min={ch.min():3d} max={ch.max():3d} "
              f"mean={ch.mean():6.1f} std={ch.std():5.1f}")
    print(f"  200 이상 픽셀 비율: {(img > 200).mean()*100:.2f}%")

print("\n" + "="*66)
print("판별 기준")
print("="*66)
img = cv2.imread(files[0])
b, g, r = [img[:, :, i].mean() for i in range(3)]
bright = (img > 200).mean() * 100
if bright < 0.5:
    print("  밝은 픽셀이 거의 없음 → 흰색 차선이 없거나 YUV 저장 의심")
if abs(b - g) > 25 or abs(g - r) > 25:
    print(f"  채널 평균 편차 큼 (B={b:.0f} G={g:.0f} R={r:.0f}) → YUV 저장 유력")
else:
    print(f"  채널 균형 정상 (B={b:.0f} G={g:.0f} R={r:.0f}) → BGR 저장으로 보임")

# 4가지 해석 시각화
fig, axes = plt.subplots(len(files), 4, figsize=(15, 3.2*len(files)))
if len(files) == 1:
    axes = axes.reshape(1, -1)
for row, f in enumerate(files):
    img = cv2.imread(f)
    views = [
        ("as BGR (원본 해석)", cv2.cvtColor(img, cv2.COLOR_BGR2RGB)),
        ("YUV→BGR 복원", cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_YUV2BGR), cv2.COLOR_BGR2RGB)),
        ("Y채널 (밝기)", img[:, :, 0]),
        ("이진화 thresh=150", (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) > 150).astype(np.uint8)*255),
    ]
    for col, (title, v) in enumerate(views):
        ax = axes[row, col]
        ax.imshow(v, cmap="gray" if v.ndim == 2 else None)
        if row == 0:
            ax.set_title(title, fontsize=10)
        ax.axis("off")
plt.tight_layout()
plt.savefig("diagnose_color.png", dpi=130)
print("\n[OK] diagnose_color.png 저장 — 어느 해석에서 차선이 보이는지 확인하세요")
