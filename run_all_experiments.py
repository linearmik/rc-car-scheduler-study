"""
run_all_experiments.py — 48회 실험 자동 실행
=============================================
스케줄러 4종 x 데이터 4규모 x 시드 5개 = 80회를 순차 실행한다.
이미 결과 JSON이 있으면 건너뛰므로, 중간에 끊겨도 다시 돌리면 이어서 진행된다.

맥북에서 밤새 돌리기 (잠자기 방지):
    caffeinate -is python3 run_all_experiments.py \
        --data-dir ~/aicar/merged --out-dir ~/aicar/results 2>&1 | tee run.log

진행 상황만 보기:
    tail -f run.log

중단:
    Ctrl+C  (다시 실행하면 남은 것부터 이어서)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

SCHEDULERS = ["plateau", "cosine", "warmup_cosine", "onecycle"]
DATA_SIZES = [1000, 2000, 3000, 6000, 0]    # 0 = 전체(10,077장)
SEEDS = [42, 123, 456, 789, 1024]

SIZE_LABEL = {1000: "1K", 2000: "2K", 3000: "3K", 6000: "6K", 10000: "10K", 0: "전체"}

def size_label(size):
    return SIZE_LABEL.get(size, f"{size//1000}K" if size >= 1000 else str(size))

def log(msg):
    print(f"[{datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--out-dir", default="./results")
    ap.add_argument("--sizes", default=None,
                    help="쉼표구분 규모(0=전체). 예: 22K는 1000,3000,10000,0 / "
                         "14K는 1000,2000,3000,6000,0. 미지정시 기본 그리드")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--workers", type=int, default=0,
                    help="DataLoader 워커 수 (디스크 I/O 병렬화)")
    ap.add_argument("--cooldown", type=int, default=30,
                    help="실험 간 대기 초 (발열 관리)")
    ap.add_argument("--save-model", action="store_true",
                    help="실차 검증용 .pt 저장 (용량 주의)")
    ap.add_argument("--dry-run", action="store_true",
                    help="실행 계획만 출력")
    args = ap.parse_args()

    script = Path(__file__).parent / "train_experiment.py"
    out_dir = Path(args.out_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = ([int(s) for s in args.sizes.split(",")] if args.sizes else DATA_SIZES)

    # 작은 데이터부터 → 빨리 결과가 쌓이고, 문제를 일찍 발견할 수 있다
    jobs = [(sch, size, seed)
            for size in sizes
            for sch in SCHEDULERS
            for seed in SEEDS]

    todo = [j for j in jobs
            if not (out_dir / f"{j[0]}_{j[1]}_{j[2]}.json").exists()]

    log(f"전체 {len(jobs)}회 중 {len(jobs)-len(todo)}회 완료됨, {len(todo)}회 남음")

    if args.dry_run:
        for sch, size, seed in todo:
            print(f"  {sch:14s} {size_label(size):10s} seed={seed}")
        return

    if not todo:
        log("모든 실험이 이미 완료되었습니다.")
        return

    done = failed = 0
    t_start = time.time()
    durations = []

    for i, (sch, size, seed) in enumerate(todo, 1):
        label = f"{sch} | {size_label(size)} | seed {seed}"
        log(f"({i}/{len(todo)}) 시작: {label}")

        cmd = [sys.executable, str(script),
               "--data-dir", args.data_dir, "--out-dir", str(out_dir),
               "--scheduler", sch, "--data-size", str(size), "--seed", str(seed),
               "--epochs", str(args.epochs), "--batch-size", str(args.batch_size),
               "--lr", str(args.lr), "--patience", str(args.patience),
               "--workers", str(args.workers)]
        if args.save_model:
            cmd.append("--save-model")

        t0 = time.time()
        try:
            r = subprocess.run(cmd, capture_output=True, text=True)
        except KeyboardInterrupt:
            log("사용자 중단. 다시 실행하면 이어서 진행됩니다.")
            return
        dt = time.time() - t0

        if r.returncode == 0:
            done += 1
            durations.append(dt)
            tail = [l for l in r.stdout.strip().split("\n") if l.startswith("[DONE]")]
            log(f"  완료 ({dt/60:.1f}분) {tail[-1] if tail else ''}")
        else:
            failed += 1
            log(f"  실패 (returncode={r.returncode})")
            err = r.stderr.strip().split("\n")
            for line in err[-4:]:
                log(f"    {line}")

        # 남은 시간 추정
        if durations:
            avg = sum(durations) / len(durations)
            eta = timedelta(seconds=int(avg * (len(todo) - i)))
            log(f"  진행 {i}/{len(todo)} | 성공 {done} 실패 {failed} | 예상 잔여 {eta}")

        if i < len(todo) and args.cooldown:
            time.sleep(args.cooldown)

    total = timedelta(seconds=int(time.time() - t_start))
    log("=" * 55)
    log(f"종료 | 성공 {done} · 실패 {failed} | 총 소요 {total}")
    log(f"결과 위치: {out_dir}")


if __name__ == "__main__":
    main()