# 소규모 자율주행 회귀 CNN: 데이터 규모와 학습률 스케줄러의 영향

**Effects of Data Scale and Learning-Rate Scheduling in a Small-Scale Autonomous-Driving Regression CNN: Offline and Closed-Loop Evaluation**

라즈베리파이 기반 소형 RC 자율주행차를 대상으로, **학습률 스케줄러 선택**과 **데이터 규모**가
회귀 CNN 성능에 미치는 영향을 **오프라인 검증**과 **실차 폐루프 주행**에서 비교한 실증 연구의 코드입니다.

> 📄 한국정보처리학회 추계학술발표대회(ACK 2026) 학부생 논문 · 저널 확장판 준비 중
> 저자: 김선형, 박영민 (숙명여자대학교)

---

## 핵심 결과

1. **데이터 규모가 스케줄러 선택을 압도한다.** 전통 스케줄러 3종(Plateau·Cosine·Warmup+Cosine)은
   대응표본 t-검정에서 통계적 차이가 관찰되지 않았고(TOST 등가성 다수 통과), 데이터 규모 증가에 따른
   검증 손실 감소(41.9~45.6%)가 스케줄러 간 최대 차이(0.4~3.5%)보다 훨씬 컸다.
2. **OneCycle은 데이터 규모가 커질수록 학습 붕괴 빈도가 증가**했다(설정한 최대 학습률 조건에서).
3. **오프라인 손실이 거의 같아도 실차 완주 결과는 갈렸다.** 즉 오프라인 검증 손실만으로는 실제
   폐루프 주행 성능을 충분히 설명하지 못했다 → **오프라인·폐루프 병행 검증**의 필요성.

---

## 저장소 구조

```
aicar/
├─ 학습 · 실험
│  ├─ train_experiment.py     학습 1회 실행 (스케줄러/규모/시드 인자)
│  ├─ run_all_experiments.py  격자 실험 자동 실행 (이어하기 지원)
│  ├─ schedulers.py           스케줄러 4종 통합 모듈
│  ├─ config.py               프로젝트 공통 설정 (모델·하드웨어·배포)
│  └─ run_overnight.sh        밤샘 실행 스크립트
├─ 분석
│  ├─ stats_reanalysis.py     통계 재분석 (t-검정·TOST·Cohen's dz·규모효과)
│  ├─ detect_divergence.py    OneCycle 발산(학습 붕괴) 판정 (final > best×3)
│  ├─ analyze_gradients.py    그래디언트 동역학 분석
│  ├─ check_spike.py          손실 급등 점검
│  └─ verify_data.py          데이터 분포 검증
├─ 실차 배포 · 주행 (라즈베리파이 전용)
│  ├─ convert_models.py       학습 모델 → Pi 배포 형식 변환 + 파일명 부여
│  ├─ ar_linear.py            실차 자율주행 메인 (카메라→CNN→조향/속도)
│  ├─ lane_detector.py        차선 검출 및 경로 편차 측정
│  └─ compare_prediction.py   예측 분포 비교
└─ *.png                      논문 그림 (데이터 분포·그래디언트 동역학·예측 분포)
```

> ⚠️ **데이터셋·학습 결과·모델 가중치·논문 원고는 저장소에 포함하지 않습니다**
> (용량 및 개인정보/미공개 원고 사유 — `.gitignore` 참고). 코드만 공개합니다.

---

## 환경

- Python 3.9+
- 학습·분석: `torch`, `numpy`, `opencv-python`, `scipy`, `matplotlib`
- 실차 주행(Pi 전용): `RPi.GPIO`, `VL53L0X`(ToF 거리센서), 그리고 하드웨어 모듈 `ts`, `NEOPIXEL`

```bash
pip install torch numpy opencv-python scipy matplotlib
```

---

## 실행 방법

### 1) 학습 (오프라인)

단일 실험:
```bash
python3 train_experiment.py \
    --data-dir  ~/aicar/merged \
    --scheduler warmup_cosine \
    --data-size 3000 \
    --seed      42 \
    --out-dir   ~/aicar/results
```

격자 전체 실험(스케줄러 4 × 규모 × 시드 5, 이어하기 지원):
```bash
caffeinate -is python3 run_all_experiments.py \
    --data-dir ~/aicar/merged --out-dir ~/aicar/results 2>&1 | tee run.log
```

데이터 규모는 학습 세트만 하위표집하여 조절하며, 검증·평가 세트는 시드 고정으로 모든 규모에서 동일합니다.
결과는 실행마다 `*_hist.csv`(에폭별 손실/lr)와 `*.json`(요약)으로 저장됩니다.

### 2) 통계 분석

```bash
python3 stats_reanalysis.py       # t-검정·TOST(±5%)·Cohen's dz·규모 효과
python3 detect_divergence.py      # OneCycle 발산 판정 (final > best×3)
```

### 3) 실차 배포 · 주행 (라즈베리파이)

**① 맥에서 모델 변환** — 학습 state_dict를 Pi 코드(NvidiaModel) 규격으로 변환하고 배포 파일명 부여:
```bash
python3 convert_models.py --src-dir results_14k --project linear_rc     # 14K 데이터셋
python3 convert_models.py --src-dir results_22k --project linear0827    # 22K 데이터셋
```
→ `{project}/_{project}_*.pt` 생성.

**② Pi로 전송** — 변환된 `_{project}_*.pt` 파일을 Pi의 `~/Downloads/` 또는 `ar_linear.py` 옆
`{project}/` 폴더로 복사. 트랙 ROI 파일 `_{project}_track.pickle`도 `{project}/`에 함께 둡니다.

**③ Pi에서 주행:**
```bash
python3 ar_linear.py linear_rc      # 인자 = 프로젝트명 (모델·트랙 파일 지정)
```
- `.pt`가 여러 개면 목록에서 **↑/↓ 선택 · Enter 로드** (1개면 자동).
- 주행 조작키:

  | 키 | 동작 |
  |----|------|
  | `g` | 자율주행 시작 |
  | `↑` / `↓` | 수동 개입 (개입 카운트 +1) |
  | `n` | 새 랩 시작 |
  | `a` | 신호등 AUTO 토글 |
  | `x` / ESC | 종료 |

- 주행 로그(`_{project}_drive_log_*.csv`)와 주행 프레임(`_frames_*/`)이 자동 저장되어,
  검출기 파라미터를 바꿔도 재주행 없이 **오프라인으로 경로 편차를 재계산**할 수 있습니다.

---

## 모델

NVIDIA PilotNet 계열 회귀 CNN (합성곱 5층 + 완전연결 4층, 약 252k 파라미터).
입력 200×66 YUV 영상 → 출력 조향값·속도값 2개. 손실은 조향 0.9 / 속도 0.1 가중 MSE.

## 인용

```
김선형, 박영민, "소규모 자율주행 회귀 CNN에서 데이터 규모와 학습률 스케줄러 선택의 효과:
오프라인 및 폐루프 평가," 한국정보처리학회 추계학술발표대회(ACK 2026).
```
