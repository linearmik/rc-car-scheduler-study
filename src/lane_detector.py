"""
차선 검출 및 경로 편차(Path Deviation) 계산
------------------------------------------------
저희 트랙(흰색 차선)에서 좌우 차선을 검출하고,
차선 중앙과 카메라 광축(이미지 중앙) 간의 픽셀 오차를 계산한다.

논문 실차 검증용 지표:
    - 평균 편차 (Mean Deviation)
    - 최대 편차 (Max Deviation)
    - 편차 표준편차 (Std)
    - 급조향 빈도

사용법 (단일 이미지):
    from lane_detector import LaneDetector
    detector = LaneDetector()
    deviation = detector.compute_deviation(frame)

사용법 (ar.py 통합):
    detector = LaneDetector()
    frame = capture_frame()
    dev = detector.compute_deviation(frame)
    log.append({'frame': i, 'deviation_px': dev, ...})

주요 파라미터 (트랙 특성에 맞춰 튜닝 필요):
    - white_thresh: 흰색 이진화 임계값 (기본 200)
    - roi_top_ratio: ROI 상단 비율 (기본 0.5 = 이미지 하단 절반)
"""
import cv2
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional


@dataclass
class LaneDetectorConfig:
    # 입력 색공간: 저장된 학습 이미지(YUV)면 True, 카메라 원본(BGR)이면 False
    input_is_yuv: bool = True

    # 이진화
    white_thresh: int = 150      # Y채널 기준. 진단 결과 차선이 뚜렷해 150이면 충분
    blur_ksize: int = 3          # 66x200 저해상도이므로 커널 축소

    # ROI: 저장 이미지는 이미 ROI 크롭이 끝난 상태라 전체를 사용
    roi_top_ratio: float = 0.0

    # 차선 스캔 (여러 y위치에서 평균). 5개 위치를 훑어 급회전 검출률을 높인다.
    scan_y_ratios: tuple = (0.3, 0.45, 0.6, 0.75, 0.9)
    min_lane_width: int = 15
    max_lane_gap: int = 150
    # 직진 구간 실측 차선 간격 중앙값(px). 쌍 선택의 기준으로 사용.
    expected_lane_width: float = 55.0

    # 노이즈 필터링
    smoothing_window: int = 5
    fallback_to_last: bool = True

    # 급조향 판단
    sudden_change_threshold: float = 30.0


class LaneDetector:
    """실시간 차선 검출 + 경로 편차 계산."""

    def __init__(self, config: Optional[LaneDetectorConfig] = None):
        self.cfg = config or LaneDetectorConfig()

        self._last_left = None
        self._last_right = None
        self._last_deviation = None
        self._deviation_buffer = deque(maxlen=self.cfg.smoothing_window)

        self.total_frames = 0
        self.detected_frames = 0
        self.fallback_frames = 0
        self.last_used_fallback = False
        self.deviations = []
        self.sudden_changes = 0

    def _binarize(self, frame):
        """
        밝기 채널을 뽑아 흰색 차선만 남긴다.

        gi.py가 저장한 학습 이미지는 YUV로 기록되어 있으므로 0번 채널이
        곧 밝기(Y)다. 반면 ar.py 주행 중 넘어오는 원본 카메라 프레임은
        BGR이므로 회색조 변환이 필요하다. input_is_yuv 로 구분한다.
        """
        if frame.ndim == 3 and self.cfg.input_is_yuv:
            gray = frame[:, :, 0]                       # Y 채널
        elif frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        blurred = cv2.GaussianBlur(gray, (self.cfg.blur_ksize, self.cfg.blur_ksize), 0)
        _, binary = cv2.threshold(blurred, self.cfg.white_thresh, 255, cv2.THRESH_BINARY)
        return binary

    def _crop_roi(self, binary):
        h = binary.shape[0]
        top = int(h * self.cfg.roi_top_ratio)
        return binary[top:, :]

    def _find_lanes_at_y(self, roi, y):
        """
        ROI의 y좌표에서 주행 차로의 좌/우 경계 x좌표를 찾는다.

        노면에는 차선 외에 화살표·정지선 등 흰색 표시가 함께 잡히므로,
        검출된 흰색 그룹들 중 '기대 차선 간격에 가장 가깝고 차량에 가까운'
        인접 쌍을 선택한다. 이미지 중앙을 기준으로 좌우를 강제 분리하면
        차량이 차로를 벗어난 급회전 구간에서 검출이 실패하므로 그 제약은
        두지 않는다.
        """
        h, w = roi.shape
        if y >= h:
            return None, None

        white_xs = np.where(roi[y, :] > 200)[0]
        if len(white_xs) < 2:
            return None, None

        # 연속된 흰색 픽셀을 그룹화
        groups = []
        current = [white_xs[0]]
        for x in white_xs[1:]:
            if x - current[-1] <= 3:
                current.append(x)
            else:
                if len(current) >= 2:
                    groups.append(current)
                current = [x]
        if len(current) >= 2:
            groups.append(current)

        if len(groups) < 2:
            return None, None

        centers = sorted((min(g) + max(g)) / 2 for g in groups)

        best, best_cost = None, float("inf")
        for i in range(len(centers) - 1):
            l, r = centers[i], centers[i + 1]
            gap = r - l
            if not (self.cfg.min_lane_width <= gap <= self.cfg.max_lane_gap):
                continue
            # 기대 폭과의 상대 오차 + 차량(이미지 중앙)에서 멀어진 정도
            cost = (abs(gap - self.cfg.expected_lane_width)
                    / self.cfg.expected_lane_width
                    + abs((l + r) / 2 - w / 2) / w)
            if cost < best_cost:
                best_cost, best = cost, (l, r)

        return best if best else (None, None)

    def _detect_lanes_robust(self, roi):
        """여러 y위치에서 검출한 결과를 평균."""
        h = roi.shape[0]
        left_candidates = []
        right_candidates = []

        for ratio in self.cfg.scan_y_ratios:
            y = int(h * ratio)
            l, r = self._find_lanes_at_y(roi, y)
            if l is not None:
                left_candidates.append(l)
            if r is not None:
                right_candidates.append(r)

        left = float(np.mean(left_candidates)) if left_candidates else None
        right = float(np.mean(right_candidates)) if right_candidates else None

        return left, right

    def compute_deviation(self, frame):
        """
        한 프레임의 경로 편차 계산.
        양수: 차량이 오른쪽으로 치우침
        음수: 차량이 왼쪽으로 치우침
        None: 검출 실패
        """
        self.total_frames += 1

        binary = self._binarize(frame)
        roi = self._crop_roi(binary)
        left, right = self._detect_lanes_robust(roi)

        # fallback 대입 전에 '진짜 검출' 여부를 판정한다.
        # 좌우가 모두 이번 프레임에서 검출됐을 때만 genuine.
        genuine = left is not None and right is not None

        if left is None and self.cfg.fallback_to_last:
            left = self._last_left
        if right is None and self.cfg.fallback_to_last:
            right = self._last_right

        if left is None or right is None:
            return None

        # genuine일 때만 검출 카운트/직전값 갱신, 아니면 fallback으로 집계.
        self.last_used_fallback = not genuine
        if genuine:
            self._last_left = left
            self._last_right = right
            self.detected_frames += 1
        else:
            self.fallback_frames += 1

        lane_center = (left + right) / 2
        image_center = frame.shape[1] / 2
        raw_deviation = image_center - lane_center

        self._deviation_buffer.append(raw_deviation)
        smoothed = float(np.mean(self._deviation_buffer))

        if self._last_deviation is not None:
            change = abs(smoothed - self._last_deviation)
            if change > self.cfg.sudden_change_threshold:
                self.sudden_changes += 1

        self._last_deviation = smoothed

        self.deviations.append(smoothed)

        return smoothed

    def visualize(self, frame):
        """검출 결과를 시각화한 이미지 반환 (디버깅용)."""
        binary = self._binarize(frame)
        roi = self._crop_roi(binary)
        left, right = self._detect_lanes_robust(roi)

        vis = (cv2.cvtColor(frame, cv2.COLOR_YUV2BGR)
               if (frame.ndim == 3 and self.cfg.input_is_yuv) else frame.copy())
        h, w = frame.shape[:2]
        roi_top = int(h * self.cfg.roi_top_ratio)

        # 이미지 중앙선 (빨강)
        cv2.line(vis, (w // 2, roi_top), (w // 2, h), (0, 0, 255), 2)

        if left is not None and right is not None:
            for ratio in self.cfg.scan_y_ratios:
                y_in_roi = int((h - roi_top) * ratio)
                y_orig = roi_top + y_in_roi
                cv2.circle(vis, (int(left), y_orig), 5, (0, 255, 0), -1)
                cv2.circle(vis, (int(right), y_orig), 5, (255, 0, 0), -1)

            lane_center = int((left + right) / 2)
            cv2.line(vis, (lane_center, roi_top), (lane_center, h),
                     (0, 255, 255), 2)

            dev = w / 2 - lane_center
            cv2.putText(vis, f'Deviation: {dev:+.1f} px',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
        else:
            cv2.putText(vis, 'Lane detection FAILED',
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 0, 255), 2)

        if roi_top > 0:
            cv2.line(vis, (0, roi_top), (w, roi_top), (128, 128, 128), 1)
        return vis

    def summary(self):
        """전체 주행 통계 (논문 표에 넣을 지표)."""
        if not self.deviations:
            return {
                'total_frames': self.total_frames,
                'detected_frames': 0,
                'detection_rate': 0.0,
                'fallback_frames': self.fallback_frames,
                'fallback_rate': (
                    self.fallback_frames / max(self.total_frames, 1) * 100
                ),
                'mean_abs_deviation': None,
                'max_abs_deviation': None,
                'std_deviation': None,
                'sudden_changes': self.sudden_changes,
                'sudden_change_rate_per_1000f': 0.0,
            }

        devs = np.array(self.deviations)
        return {
            'total_frames': self.total_frames,
            'detected_frames': self.detected_frames,
            'detection_rate': self.detected_frames / max(self.total_frames, 1) * 100,
            'fallback_frames': self.fallback_frames,
            'fallback_rate': self.fallback_frames / max(self.total_frames, 1) * 100,
            'mean_abs_deviation': float(np.mean(np.abs(devs))),
            'max_abs_deviation': float(np.max(np.abs(devs))),
            'std_deviation': float(np.std(devs)),
            'sudden_changes': self.sudden_changes,
            'sudden_change_rate_per_1000f': (
                self.sudden_changes / max(self.total_frames, 1) * 1000
            ),
        }

    def reset(self):
        """새 주행 시작 시 통계 초기화."""
        self._last_left = None
        self._last_right = None
        self._last_deviation = None
        self._deviation_buffer.clear()
        self.total_frames = 0
        self.detected_frames = 0
        self.fallback_frames = 0
        self.last_used_fallback = False
        self.deviations = []
        self.sudden_changes = 0


if __name__ == '__main__':
    """
    실사용 전 테스트:
        python3 lane_detector.py <이미지_파일>
        python3 lane_detector.py <이미지_폴더>
    """
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("사용법: python3 lane_detector.py <이미지_파일_또는_폴더>")
        print()
        print("예:")
        print("  python3 lane_detector.py sample.png")
        print("  python3 lane_detector.py ~/data/track/")
        sys.exit(0)

    input_path = Path(sys.argv[1])
    detector = LaneDetector()

    if input_path.is_file():
        frame = cv2.imread(str(input_path))
        if frame is None:
            print(f"[ERROR] 이미지를 읽을 수 없습니다: {input_path}")
            sys.exit(1)

        deviation = detector.compute_deviation(frame)
        print(f"편차: {deviation}")

        vis = detector.visualize(frame)
        output = input_path.parent / f'lane_vis_{input_path.name}'
        cv2.imwrite(str(output), vis)
        print(f"시각화 저장: {output}")

    elif input_path.is_dir():
        images = sorted(list(input_path.glob('*.png')) + list(input_path.glob('*.jpg')))[:10]
        if not images:
            print(f"[ERROR] 이미지가 없습니다: {input_path}")
            sys.exit(1)

        output_dir = input_path.parent / 'lane_vis'
        output_dir.mkdir(exist_ok=True)

        print(f"총 {len(images)}장 처리 중...")
        for img_path in images:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            deviation = detector.compute_deviation(frame)
            vis = detector.visualize(frame)
            output = output_dir / f'vis_{img_path.name}'
            cv2.imwrite(str(output), vis)
            print(f"  {img_path.name}: deviation={deviation}")

        print(f"\n시각화 저장 위치: {output_dir}")

        print("\n" + "=" * 50)
        print("검출 요약:")
        for k, v in detector.summary().items():
            print(f"  {k}: {v}")
    else:
        print(f"[ERROR] 경로가 유효하지 않습니다: {input_path}")