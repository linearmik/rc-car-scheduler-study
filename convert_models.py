"""
학습 모델 → ar_research 호환 형식 변환 + 배포 파일명 부여
=========================================================================
train_experiment.py 가 저장한 state_dict(features/regressor) 키를
ar_research.py 의 NvidiaModel(layer1/layer2/layer3) 규격으로 변환하고,
ar_linear.py 가 찾을 수 있는 배포 파일명 `_{project}_{원본이름}` 으로 저장한다.

ar_linear.py 는 실행 시 `{project}/_{project}_*.pt` (또는 ~/Downloads/_{project}_*.pt)
를 찾으므로, 출력물을 그 규칙에 맞춰 낸다.

두 모델은 레이어 구성이 동일하며 Dropout 위치만 다르다(추론엔 무영향).

사용법:
    # 22K(merged) 배포모델: 4스케줄러 × 전체 × seed42
    python3 convert_models.py --src-dir results_22k --project linear0827
    # 14K(linear_rc)
    python3 convert_models.py --src-dir results_14k --project linear_rc
    # 특정 모델만
    python3 convert_models.py --src-dir results_22k --project linear0827 \
        --schedulers warmup_cosine --sizes 0 --seeds 42
    # 변환본 검증
    python3 convert_models.py --check FILE
"""
import argparse
import sys
from pathlib import Path

import torch

KEY_MAP = {
    "features.0":  "layer1.0", "features.2":  "layer1.2",
    "features.4":  "layer1.4", "features.6":  "layer1.6",
    "features.8":  "layer1.9",
    "regressor.1": "layer2.1", "regressor.3": "layer2.3",
    "regressor.5": "layer2.5", "regressor.7": "layer3.0",
}


def convert_state_dict(sd):
    out, unknown = {}, []
    for k, v in sd.items():
        prefix, _, suffix = k.rpartition(".")
        if prefix in KEY_MAP:
            out[f"{KEY_MAP[prefix]}.{suffix}"] = v
        else:
            unknown.append(k)
    return out, unknown


def convert_file(src: Path, dst: Path):
    sd = torch.load(src, map_location="cpu")
    if not isinstance(sd, dict):
        return None, "dict 형식이 아님"
    sd = sd.get("state_dict", sd)
    if any(k.startswith("layer") for k in sd):
        return None, "이미 변환된 파일"
    new_sd, unknown = convert_state_dict(sd)
    if unknown:
        return None, f"매핑 불가 키: {unknown[:3]}"
    if "layer3.0.weight" not in new_sd:
        return None, "출력층(layer3.0) 생성 실패"
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(new_sd, dst)
    return dst, f"출력 차원 {new_sd['layer3.0.weight'].shape[0]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", help="학습 결과 폴더 (results_22k / results_14k)")
    ap.add_argument("--project", help="배포 프로젝트명 = ar_linear 인자 (linear0827 / linear_rc)")
    ap.add_argument("--out", default=None, help="출력 폴더 (기본: --project 이름)")
    ap.add_argument("--schedulers", default="plateau,cosine,warmup_cosine,onecycle")
    ap.add_argument("--sizes", default="0", help="쉼표구분 규모 (기본 0=전체)")
    ap.add_argument("--seeds", default="42", help="쉼표구분 시드 (기본 42)")
    ap.add_argument("--all", action="store_true", help="src-dir의 모든 .pt 변환")
    ap.add_argument("--check", help="변환본 키 검증만")
    args = ap.parse_args()

    if args.check:
        sd = torch.load(args.check, map_location="cpu")
        for k, v in sd.items():
            print(f"  {k:<22} {tuple(v.shape)}")
        print(f"\nar_research 호환: {'예' if 'layer3.0.weight' in sd else '아니오'}")
        return

    if not args.src_dir or not args.project:
        ap.error("--src-dir 와 --project 는 필수입니다.")

    src_dir = Path(args.src_dir)
    project = args.project
    out_dir = Path(args.out) if args.out else Path(project)

    if args.all:
        names = [p.name for p in sorted(src_dir.glob("*.pt"))]
    else:
        scheds = args.schedulers.split(",")
        sizes = args.sizes.split(",")
        seeds = args.seeds.split(",")
        names = [f"{s}_{sz}_{sd}.pt" for s in scheds for sz in sizes for sd in seeds]

    print("=" * 68)
    print(f"변환: {src_dir}/ → {out_dir}/  (배포명 _{project}_*.pt)")
    print("=" * 68)
    ok = skip = 0
    for name in names:
        src = src_dir / name
        if not src.exists():
            print(f"  [없음]   {name}")
            skip += 1
            continue
        dst = out_dir / f"_{project}_{name}"       # 배포 파일명 규칙
        res, msg = convert_file(src, dst)
        if res:
            ok += 1
            print(f"  [OK]     {dst.name:<40} {msg}")
        else:
            skip += 1
            print(f"  [건너뜀] {name:<32} {msg}")

    print("-" * 68)
    print(f"변환 {ok}개 / 건너뜀 {skip}개")
    if ok:
        print(f"\n라즈베리파이로 복사(프로젝트 폴더 안에 넣기):")
        print(f"  scp {out_dir}/_{project}_*.pt pi@192.168.137.96:~/aicar/{project}/")
        print(f"실행: cd ~/aicar && python3 ar_linear.py {project}")


if __name__ == "__main__":
    main()
