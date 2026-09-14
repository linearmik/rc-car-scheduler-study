"""
config.py
==========
Raspberry Pi 기반 자율주행 RC카 연구 프로젝트의 공통 설정 파일입니다.

- gi_research.py : 주행 이미지·조향각·속도 수집
- dl_research.py : 조향각·속도 다중 출력 모델 학습
- ar_research.py : 학습 모델 기반 자율주행 및 주행 로그 기록

하드웨어 배선, 카메라/ROI, 이미지 전처리, 파일명 규약, 학습 설정을 이 파일에서
일관되게 관리합니다. 실행 중 변경되는 상태(현재 lap, 파일 번호, 모터 PWM 등)는
각 프로그램에 두고, 재사용 가능한 기본값만 이 파일에 정의합니다.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union


# =============================================================================
# OpenCV BGR 색상 상수
# =============================================================================
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GRAY = (128, 128, 128)
DARK_GRAY = (32, 32, 32)
RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
CYAN = (255, 255, 0)
MAGENTA = (255, 0, 255)


# =============================================================================
# 7개 조향각 그룹: (그룹명, 최소 조향각, 최대 조향각)
# gi/dl의 분포 분석, train/valid/test 분할, metadata.csv에 공통 사용
# =============================================================================
ANGLE_GROUPS = (
    ("A", -99, -47),
    ("B", -46, -22),
    ("C", -21, -6),
    ("D", -5, 5),
    ("E", 6, 21),
    ("F", 22, 46),
    ("G", 47, 99),
)


class AppConfig:
    """세 프로그램이 공유하는 화면·카메라·이미지·ROI 설정."""

    # LCD 및 카메라
    LCD_WIDTH = 800
    LCD_HEIGHT = 480
    SIDE_PANEL_WIDTH = 80
    CAMERA_INDEX = 0
    CAMERA_BACKEND = "CAP_V4L"
    CAMERA_WIDTH = 640
    CAMERA_HEIGHT = 480
    CAMERA_FPS = 30

    # NVIDIA PilotNet 계열 모델 입력
    MODEL_INPUT_WIDTH = 200
    MODEL_INPUT_HEIGHT = 66
    MODEL_INPUT_SHAPE = (MODEL_INPUT_HEIGHT, MODEL_INPUT_WIDTH, 3)
    COLOR_SPACE = "YUV"
    GAUSSIAN_BLUR_KERNEL = (3, 3)
    NORMALIZE_DIVISOR = 255.0

    # 기본 차선/도로 ROI. GI에서 사용자 조정 후 track.pickle로 프로젝트별 저장 가능
    ROI_XL = 0
    ROI_XR = CAMERA_WIDTH - 1
    ROI_YU = 246
    ROI_YD = 459
    ROI_MIN_WIDTH = 200
    ROI_MIN_HEIGHT = 20

    # 데이터/라벨
    IMAGE_PREFIX = "P"
    IMAGE_EXTENSION = ".png"
    METADATA_FILENAME = "metadata.csv"
    TRACK_PICKLE_SUFFIX = "_track.pickle"
    TRAFFIC_SIGNAL_PICKLE_SUFFIX = "_TS.pickle"

    # 조향각 및 속도
    STEERING_MIN = -99
    STEERING_MAX = 99
    STEERING_NEUTRAL = 0
    SPEED_MIN = 0
    SPEED_MAX = 100
    DEFAULT_FORWARD_SPEED = 60


class HardwareConfig:
    """Raspberry Pi BCM GPIO 배선과 모터 제어 공통 설정."""

    # 2륜 DC 모터
    MOTOR_L_PWM = 12
    MOTOR_L_DIR = 5
    MOTOR_R_PWM = 13
    MOTOR_R_DIR = 6
    MOTOR_PWM_FREQUENCY = 500
    MOTOR_PWM_MIN = -100
    MOTOR_PWM_MAX = 100

    # 공통 센서·등화 장치
    LIGHT_SENSOR = 25
    BUZZER = 23
    MUSIC = 24
    LAMP_R_YELLOW = 20
    LAMP_L_YELLOW = 26
    LAMP_BRAKE = 21


class GIConfig:
    """gi_research.py: 데이터 수집 및 수동 주행 기본값."""

    FILE_PREFIX = AppConfig.IMAGE_PREFIX
    AUTO_LIGHT_DEFAULT = True
    MANUAL_LIGHT_DEFAULT = False
    VARIABLE_MODE_DEFAULT = True
    MENU_VIEW_DEFAULT = True

    DEFAULT_FORWARD_SPEED = AppConfig.DEFAULT_FORWARD_SPEED
    DEFAULT_MOTOR_ANGLE_GAIN = 0.15
    DEFAULT_RECORD_INTERVAL_NS = 500_000_000
    RECORD_INTERVAL_UNIT_NS = 100_000_000
    VARIABLE_MIN_FORWARD_SPEED = 40
    VARIABLE_SPEED_STEP = 5
    VARIABLE_INTERVAL_STEPS = 11

    # 수집 통계
    METADATA_ON = True
    INITIAL_LAP = 1
    EXTREME_ANGLE_LIMIT = 60

    # 수동 조작 PWM
    MANUAL_FORWARD_PWM = 80
    MANUAL_REVERSE_PWM = -70
    MANUAL_TURN_PWM = 70
    PANEL_GRAY = DARK_GRAY

    @classmethod
    def record_interval_list(cls) -> list[int]:
        return [cls.RECORD_INTERVAL_UNIT_NS * i for i in range(1, cls.VARIABLE_INTERVAL_STEPS + 1)]


class DLConfig:
    """dl_research.py: 데이터 분할, 증강, 모델 학습의 공통 설정."""

    # 기본 학습 파라미터
    EPOCHS = 300
    LEARN_RATE = 1e-3
    VALID_RATE = 0.20
    EVT = "V"                         # E: Example 제외, V: Validation 포함, T: Train 포함
    RANDOM_SEED = 42

    # Augmentation
    AUG_SHIFT = 10
    AUG_ANGLE = 10
    AUG_MODE_DEFAULT = False
    YUV_MODE_DEFAULT = True

    # 데이터 표시 화면
    CELL_H_AUG = 37
    CELL_H_STD = 70
    ROW_N_AUG = 11
    ROW_N_STD = 6
    RED_DN = (0, 0, 96)
    BLUE_DN = (96, 0, 0)
    GRAY_DN = DARK_GRAY

    # Lap 기반 분할은 실험에서 필요할 때 True로 변경합니다.
    # 기본값 False는 기존의 그룹 내 무작위 분할 방식을 유지합니다.
    LAP_SPLIT_ENABLED = False
    VALID_LAPS = (5,)
    TEST_LAPS = (6,)

    # 다중 출력 [steering_angle, forward_speed]
    MODEL_OUTPUT_DIM = 2
    LABEL_SCALE = 99.0
    STEERING_LOSS_WEIGHT = 0.9
    SPEED_LOSS_WEIGHT = 0.1

    # Raspberry Pi 학습 시 메모리 사용량을 고려한 배치 크기
    TRAIN_BATCH_SIZE = 16
    VALID_BATCH_SIZE = 16
    TEST_BATCH_SIZE = 16
    EARLY_STOPPING_PATIENCE = 30

    # 파일명 suffix
    DEEPLEARN_PICKLE_SUFFIX = "_deep_learn.pickle"
    COLAB_PICKLE_SUFFIX = "_colab_split.pickle"
    DATASET_MANIFEST_SUFFIX = "_dataset_manifest.csv"
    MODEL_CHECK_SUFFIX = "_model_check.pt"
    RESULT_IMAGE_SUFFIX = "_result.png"
    ZIP_COMPRESSION_LEVEL = 9


class ARConfig:
    """ar_research.py: 자율주행, 거리 센서, 주행 로그 설정."""

    DEFAULT_RUN_SPEED = 70
    ANGLE_GAIN = 1.0
    HORN_DISTANCE_MM = 350
    STOP_DISTANCE_MM = 110
    RED_SIGNAL_RESTART_FRAMES = 33
    MOTOR_BAR_CENTER_Y = 240

    RESEARCH_LOG_ON = True
    INITIAL_LAP = 1

    # 다중 출력 모델의 속도 예측을 실제 PWM에 반영할지 여부
    # False이면 기존처럼 DEFAULT_RUN_SPEED를 사용하되 예측 속도는 로그에 남깁니다.
    USE_PREDICTED_SPEED = False
    PREDICTED_SPEED_MIN = GIConfig.VARIABLE_MIN_FORWARD_SPEED
    PREDICTED_SPEED_MAX = 95

    # 이전 단일 출력 모델(.pt) 호환용
    LEGACY_SINGLE_OUTPUT_IS_NORMALIZED = False
    MODEL_FILE_GLOB = "*.pt"
    DOWNLOADS_DIRNAME = "Downloads"


class TrafficSignalConfig:
    """ar_research.py에서 사용하는 신호등 검출 기본값."""

    WIN_XL = 0
    WIN_XR = AppConfig.CAMERA_WIDTH - 1
    WIN_YU = 30
    WIN_YD = 180
    MAX_RADIUS = 9
    MIN_RADIUS = 8

    LINE_LEAD_ANGLE_GREEN = 200
    LINE_TAIL_ANGLE_GREEN = 90
    LINE_LEAD_ANGLE_RED = 40
    LINE_TAIL_ANGLE_RED = 300
    SATURATION_HALF_GREEN = 50
    SATURATION_HALF_RED = 50
    HSV_VALUE_GREEN = 50
    HSV_VALUE_RED = 50
    THRESH_GREEN = 50
    THRESH_RED = 50


@dataclass(frozen=True)
class ImageLabel:
    """파일명에서 복원한 학습 라벨과 파일 정보."""

    file_name: str
    index: int
    speed: int
    angle: int
    aug_tag: str = ""
    prefix: str = AppConfig.IMAGE_PREFIX
    path: Optional[Path] = None

    @property
    def filename(self) -> str:
        return self.file_name

    @property
    def angle_raw(self) -> int:
        return self.angle

    @property
    def is_augmented(self) -> bool:
        return bool(self.aug_tag)

    @property
    def angle_group(self) -> str:
        return get_angle_group(self.angle)

    @property
    def group(self) -> str:
        return self.angle_group

    @property
    def image_path(self) -> Optional[Path]:
        return self.path


# 신규 표준: P0001_060_+15.png / P0001_060_L-05.png
_NEW_FILENAME_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<index>\d{4})_(?P<speed>\d{3})_(?P<aug>[LR])?(?P<angle>[+-]\d{2})$",
    re.IGNORECASE,
)

# 기존 형식: P0001__-45.png / P0001_R-24.png
_LEGACY_FILENAME_RE = re.compile(
    r"^(?P<prefix>.*?)(?P<index>\d{4})_{1,2}(?P<aug>[LR])?(?P<angle>[+-]\d{2})$",
    re.IGNORECASE,
)


def clamp_int(
    value: Union[int, float, str],
    min_value: int = AppConfig.STEERING_MIN,
    max_value: int = AppConfig.STEERING_MAX,
) -> int:
    """값을 반올림 정수로 바꾼 뒤 지정 범위로 제한합니다."""

    try:
        result = int(round(float(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"정수로 변환할 수 없는 값입니다: {value!r}") from exc
    return max(min_value, min(max_value, result))


def get_angle_group(angle: Union[int, float, str]) -> str:
    """조향각에 대응하는 A~G 그룹을 반환합니다."""

    value = clamp_int(angle)
    for group_name, lower, upper in ANGLE_GROUPS:
        if lower <= value <= upper:
            return group_name
    raise ValueError(f"조향각 범위를 벗어났습니다: {value}")


def get_angle_group_bounds(group_name: str) -> tuple[int, int]:
    """A~G 그룹의 조향각 하한·상한을 반환합니다."""

    target = str(group_name).upper()
    for name, lower, upper in ANGLE_GROUPS:
        if name == target:
            return lower, upper
    raise KeyError(f"알 수 없는 조향각 그룹입니다: {group_name}")


def build_image_filename(
    source: Optional[Union[ImageLabel, int]] = None,
    *,
    index: Optional[int] = None,
    speed: Optional[Union[int, float, str]] = None,
    angle: Optional[Union[int, float, str]] = None,
    aug_tag: str = "",
    prefix: Optional[str] = None,
    extension: str = AppConfig.IMAGE_EXTENSION,
) -> str:
    """표준 학습 이미지 파일명을 생성합니다.

    표준 형식
        P0001_060_+15.png
        P0001_060_L-05.png

    ``source``에 ImageLabel을 주면 index/speed/prefix를 이어받아 증강 파일명을
    만들 수 있습니다. 예: ``build_image_filename(label, angle=-24, aug_tag='L')``.
    """

    inherited: Optional[ImageLabel] = source if isinstance(source, ImageLabel) else None
    if inherited is not None:
        if index is None:
            index = inherited.index
        if speed is None:
            speed = inherited.speed
        if angle is None:
            angle = inherited.angle
        if prefix is None:
            prefix = inherited.prefix
    elif isinstance(source, int) and index is None:
        index = source

    if index is None:
        raise ValueError("이미지 파일명에는 index가 필요합니다.")
    if angle is None:
        raise ValueError("이미지 파일명에는 angle이 필요합니다.")

    file_index = max(0, int(index))
    file_speed = clamp_int(
        AppConfig.DEFAULT_FORWARD_SPEED if speed is None else speed,
        AppConfig.SPEED_MIN,
        AppConfig.SPEED_MAX,
    )
    steering = clamp_int(angle, AppConfig.STEERING_MIN, AppConfig.STEERING_MAX)
    file_prefix = AppConfig.IMAGE_PREFIX if prefix is None else str(prefix)
    tag = str(aug_tag).upper().strip()
    if tag not in ("", "L", "R"):
        raise ValueError("aug_tag은 '', 'L', 'R' 중 하나여야 합니다.")

    suffix = extension if str(extension).startswith(".") else f".{extension}"
    return f"{file_prefix}{file_index:04d}_{file_speed:03d}_{tag}{steering:+03d}{suffix}"


def parse_image_filename(file_path: Union[str, Path]) -> Optional[ImageLabel]:
    """신규·기존 수집 파일명에서 index, speed, angle, augmentation 정보를 읽습니다.

    지원 형식
        P0001_060_+15.png       신규 표준 파일
        P0001_060_L-05.png      신규 증강 파일
        P0001__-45.png          이전 gi.py 파일(속도는 기본값 60으로 보완)
        P0001_R-24.png          이전 증강 파일
    """

    path = Path(file_path)
    stem = path.stem

    match = _NEW_FILENAME_RE.match(stem)
    if match:
        data = match.groupdict()
        return ImageLabel(
            file_name=path.name,
            index=int(data["index"]),
            speed=clamp_int(data["speed"], AppConfig.SPEED_MIN, AppConfig.SPEED_MAX),
            angle=clamp_int(data["angle"]),
            aug_tag=(data.get("aug") or "").upper(),
            prefix=data.get("prefix") or AppConfig.IMAGE_PREFIX,
            path=path,
        )

    match = _LEGACY_FILENAME_RE.match(stem)
    if match:
        data = match.groupdict()
        return ImageLabel(
            file_name=path.name,
            index=int(data["index"]),
            speed=AppConfig.DEFAULT_FORWARD_SPEED,
            angle=clamp_int(data["angle"]),
            aug_tag=(data.get("aug") or "").upper(),
            prefix=data.get("prefix") or AppConfig.IMAGE_PREFIX,
            path=path,
        )

    return None
