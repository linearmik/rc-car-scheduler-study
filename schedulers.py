"""
학습률 스케줄러 4종 통합 모듈
--------------------------------
논문 실험용: Plateau / Cosine / Warmup+Cosine / OneCycleLR 을 동일 인터페이스로 사용

사용법:
    from schedulers import build_scheduler, SchedulerType

    scheduler = build_scheduler(
        name='warmup_cosine',
        optimizer=optimizer,
        total_epochs=100,
        steps_per_epoch=len(train_loader),
        base_lr=1e-3,
    )

    # 매 스텝(OneCycle) 또는 매 에폭(그 외) 호출:
    if scheduler.step_per_batch:
        for batch in train_loader:
            ...
            scheduler.step()
    else:
        for epoch in range(...):
            ...
            scheduler.step(valid_loss)  # Plateau만 valid_loss 필요
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch.optim import Optimizer
from torch.optim.lr_scheduler import (
    ReduceLROnPlateau,
    CosineAnnealingLR,
    LinearLR,
    SequentialLR,
    OneCycleLR as TorchOneCycleLR,
)


# ============================================================
# 스케줄러 이름 상수 (오타 방지)
# ============================================================
class SchedulerType:
    PLATEAU = 'plateau'
    COSINE = 'cosine'
    WARMUP_COSINE = 'warmup_cosine'
    ONECYCLE = 'onecycle'

    ALL = ['plateau', 'cosine', 'warmup_cosine', 'onecycle']


# ============================================================
# 통합 래퍼 클래스
# ============================================================
class SchedulerWrapper:
    """
    4종 스케줄러를 동일 인터페이스로 감싸는 클래스.
    - step_per_batch: True면 매 배치마다 step(), False면 매 에폭마다 step()
    - needs_metric: True면 step(valid_loss) 필요 (Plateau만)
    """
    def __init__(self, scheduler, name: str, step_per_batch: bool, needs_metric: bool):
        self.scheduler = scheduler
        self.name = name
        self.step_per_batch = step_per_batch
        self.needs_metric = needs_metric

    def step(self, metric: Optional[float] = None):
        """스케줄러 진행. Plateau만 metric(valid_loss) 필요."""
        if self.needs_metric:
            if metric is None:
                raise ValueError(f"{self.name} scheduler requires 'metric' (valid_loss)")
            self.scheduler.step(metric)
        else:
            self.scheduler.step()

    def get_last_lr(self):
        """현재 학습률 반환."""
        if hasattr(self.scheduler, 'get_last_lr'):
            return self.scheduler.get_last_lr()
        # ReduceLROnPlateau는 get_last_lr 없음 → optimizer에서 직접
        return [pg['lr'] for pg in self.scheduler.optimizer.param_groups]

    def state_dict(self):
        return self.scheduler.state_dict()

    def load_state_dict(self, state_dict):
        self.scheduler.load_state_dict(state_dict)


# ============================================================
# 팩토리 함수
# ============================================================
def build_scheduler(
    name: str,
    optimizer: Optimizer,
    total_epochs: int,
    steps_per_epoch: int,
    base_lr: float = 1e-3,
    # Plateau 전용
    plateau_factor: float = 0.5,
    plateau_patience: int = 10,
    # Warmup+Cosine 전용
    warmup_epochs: int = 5,
    warmup_start_factor: float = 0.01,
    # OneCycleLR 전용
    onecycle_max_lr_ratio: float = 10.0,   # base_lr × 10 = max_lr (일반적 관례)
    onecycle_pct_start: float = 0.3,
    onecycle_final_div_factor: float = 1e4,
) -> SchedulerWrapper:
    """
    스케줄러 이름을 받아 SchedulerWrapper를 반환한다.

    Args:
        name: 'plateau' | 'cosine' | 'warmup_cosine' | 'onecycle'
        optimizer: PyTorch optimizer
        total_epochs: 전체 학습 에폭 수 (Cosine/Warmup/OneCycle에 필요)
        steps_per_epoch: 에폭당 배치 수 (OneCycle에 필요)
        base_lr: 기본 학습률 (Plateau/Cosine/Warmup 기준값)

    Returns:
        SchedulerWrapper: 통합 인터페이스 스케줄러
    """
    name = name.lower()

    if name == SchedulerType.PLATEAU:
        # 검증 손실 정체 시 학습률 절반으로
        sched = ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=plateau_factor,
            patience=plateau_patience,
        )
        return SchedulerWrapper(
            sched, name='plateau',
            step_per_batch=False,
            needs_metric=True,
        )

    elif name == SchedulerType.COSINE:
        # 전체 에폭 동안 코사인 곡선으로 감소
        sched = CosineAnnealingLR(optimizer, T_max=max(1, total_epochs))
        return SchedulerWrapper(
            sched, name='cosine',
            step_per_batch=False,
            needs_metric=False,
        )

    elif name == SchedulerType.WARMUP_COSINE:
        # 초반 warmup_epochs 동안 선형 증가 → 이후 코사인 감소
        # 짧은 학습(예: 테스트용 5 epoch)에서도 코사인 구간이 남도록 warmup을
        # 전체의 1/4 이내로 제한한다. 제한이 없으면 T_max=0으로 0 나누기 발생.
        warmup_epochs = max(1, min(warmup_epochs, total_epochs // 4))
        t_max = max(1, total_epochs - warmup_epochs)
        warmup = LinearLR(
            optimizer,
            start_factor=warmup_start_factor,
            end_factor=1.0,
            total_iters=warmup_epochs,
        )
        cosine = CosineAnnealingLR(
            optimizer,
            T_max=t_max,
        )
        sched = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine],
            milestones=[warmup_epochs],
        )
        return SchedulerWrapper(
            sched, name='warmup_cosine',
            step_per_batch=False,
            needs_metric=False,
        )

    elif name == SchedulerType.ONECYCLE:
        # 한 번의 사이클(상승 → 감소 → 극저 학습률)
        # 매 배치마다 step 필요!
        total_steps = max(1, total_epochs * steps_per_epoch)
        max_lr = base_lr * onecycle_max_lr_ratio
        sched = TorchOneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=onecycle_pct_start,
            anneal_strategy='cos',
            final_div_factor=onecycle_final_div_factor,
        )
        return SchedulerWrapper(
            sched, name='onecycle',
            step_per_batch=True,   # ← 중요: 매 배치마다 step()
            needs_metric=False,
        )

    else:
        raise ValueError(
            f"Unknown scheduler: {name}. "
            f"Choose from {SchedulerType.ALL}"
        )


# ============================================================
# 학습률 이력 추적 헬퍼 (논문 그래프용)
# ============================================================
class LRHistoryTracker:
    """에폭별/스텝별 학습률 이력을 기록한다."""
    def __init__(self):
        self.epoch_lrs = []       # 에폭 종료 시점의 학습률
        self.step_lrs = []        # 매 스텝의 학습률 (OneCycle용)

    def record_epoch(self, scheduler: SchedulerWrapper):
        lrs = scheduler.get_last_lr()
        # param_group이 여러 개면 첫 번째 것만 (일반적 관례)
        self.epoch_lrs.append(lrs[0] if lrs else 0.0)

    def record_step(self, scheduler: SchedulerWrapper):
        lrs = scheduler.get_last_lr()
        self.step_lrs.append(lrs[0] if lrs else 0.0)

    def to_dict(self):
        return {
            'epoch_lrs': self.epoch_lrs,
            'step_lrs': self.step_lrs,
        }


# ============================================================
# 그래디언트 노름 추적 헬퍼 (논문 그래프용)
# ============================================================
class GradientNormTracker:
    """
    학습 중 그래디언트 L2-Norm을 추적한다.
    논문의 "그래디언트 동역학" 섹션에 사용.
    """
    def __init__(self):
        self.step_norms = []      # 매 스텝의 grad norm
        self.epoch_avg_norms = [] # 에폭 평균 grad norm
        self._current_epoch_norms = []

    def record_step(self, model: torch.nn.Module):
        """
        backward() 이후, optimizer.step() 이전에 호출.
        모델의 모든 파라미터 그래디언트 L2 norm을 계산.
        """
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.detach().data.norm(2)
                total_norm_sq += param_norm.item() ** 2
        total_norm = math.sqrt(total_norm_sq)
        self.step_norms.append(total_norm)
        self._current_epoch_norms.append(total_norm)
        return total_norm

    def end_epoch(self):
        """에폭 종료 시 호출. 에폭 평균 norm 저장."""
        if self._current_epoch_norms:
            avg = sum(self._current_epoch_norms) / len(self._current_epoch_norms)
            self.epoch_avg_norms.append(avg)
            self._current_epoch_norms = []

    def to_dict(self):
        return {
            'step_norms': self.step_norms,
            'epoch_avg_norms': self.epoch_avg_norms,
        }


# ============================================================
# 자체 테스트
# ============================================================
if __name__ == '__main__':
    # 스케줄러 정상 생성 테스트
    import torch.nn as nn

    model = nn.Linear(10, 2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    print("=" * 60)
    print("스케줄러 4종 생성 테스트")
    print("=" * 60)

    for name in SchedulerType.ALL:
        # 매 테스트마다 새 optimizer (state 오염 방지)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        sched = build_scheduler(
            name=name,
            optimizer=opt,
            total_epochs=100,
            steps_per_epoch=50,
            base_lr=1e-3,
        )
        print(f"\n[{name}]")
        print(f"  step_per_batch: {sched.step_per_batch}")
        print(f"  needs_metric:   {sched.needs_metric}")
        print(f"  초기 LR:        {sched.get_last_lr()[0]:.6f}")

        # 1 스텝 진행해보기
        if sched.needs_metric:
            sched.step(metric=0.5)
        else:
            sched.step()
        print(f"  1스텝 후 LR:   {sched.get_last_lr()[0]:.6f}")

    print("\n✅ 모든 스케줄러 정상 생성")