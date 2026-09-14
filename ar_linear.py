#############################################################################################
# Auto Drive Car  V1.0                                                          <ar.py>
# PyTorch 딥러닝 프레임 웍에서 학습된 모델파일에 의하여 자율주행 합니다.
# $ python ar.py <이미지 디렉터리>
#############################################################################################
import os
import sys
import time
import glob
import pickle
import csv
import datetime
import cv2 as cv
import numpy as np
import RPi.GPIO as GPIO
import torch
import torch.nn as nn

from config import (
    AppConfig,
    HardwareConfig,
    DLConfig,
    ARConfig,
    TrafficSignalConfig,
    RED, GREEN, BLUE, MAGENTA, CYAN, YELLOW, WHITE, GRAY, BLACK,
    clamp_int,
)

import VL53L0X
import ts
import NEOPIXEL as nx
from lane_detector import LaneDetector, LaneDetectorConfig

# 설정 인스턴스
APP_CFG = AppConfig()
HW_CFG = HardwareConfig()
DL_CFG = DLConfig()
AR_CFG = ARConfig()
TS_CFG = TrafficSignalConfig()

HORN_DISTANCE = AR_CFG.HORN_DISTANCE_MM
STOP_DISTANCE = AR_CFG.STOP_DISTANCE_MM
RUN_SPEED = AR_CFG.DEFAULT_RUN_SPEED
ANGLE_GAIN = AR_CFG.ANGLE_GAIN
downLoads = AR_CFG.DOWNLOADS_DIRNAME

# 전역 변수
imageDir = ''
model = None
modelFile = None
MODEL_FILE = False
lightMode = 'MANU'
lightCar = ' '

# 주행 로그 관련
RESEARCH_LOG_ON = AR_CFG.RESEARCH_LOG_ON
# 주행 프레임(200x66 BGR) 저장 여부. autoRun일 때만 _frames_{sessionId}/에 저장.
# 검출기 파라미터를 바꿔도 재주행 없이 오프라인으로 편차를 다시 뽑기 위한 원본.
SAVE_DRIVE_FRAMES = True
driveFramesDir = None
driveLogPath = None
driveLogFile = None
driveLogWriter = None
driveSessionId = ''
driveStartTime = 0.0
currentLap = AR_CFG.INITIAL_LAP
frameId = 0
autoFrameCount = 0
manualFrameCount = 0
interventionCount = 0
cameraFailCount = 0
redStopFrameCount = 0
tofStopFrameCount = 0
modelInferCount = 0
modelInferTotalMs = 0.0
maxAbsPredAngle = 0
MODEL_OUTPUT_DIM = DL_CFG.MODEL_OUTPUT_DIM

# 신호등 파라미터
TSWIN_XL = TS_CFG.WIN_XL
TSWIN_XR = TS_CFG.WIN_XR
TSWIN_YU = TS_CFG.WIN_YU
TSWIN_YD = TS_CFG.WIN_YD
maxRadius = TS_CFG.MAX_RADIUS
minRadius = TS_CFG.MIN_RADIUS
lineLeadAngGREEN = TS_CFG.LINE_LEAD_ANGLE_GREEN
lineTailAngGREEN = TS_CFG.LINE_TAIL_ANGLE_GREEN
lineLeadAngRED = TS_CFG.LINE_LEAD_ANGLE_RED
lineTailAngRED = TS_CFG.LINE_TAIL_ANGLE_RED
saturationHalfGREEN = TS_CFG.SATURATION_HALF_GREEN
saturationHalfRED = TS_CFG.SATURATION_HALF_RED
HSV_valueGREEN = TS_CFG.HSV_VALUE_GREEN
HSV_valueRED = TS_CFG.HSV_VALUE_RED
THRESH_GREEN = TS_CFG.THRESH_GREEN
THRESH_RED = TS_CFG.THRESH_RED

HSV_hueLeadGREEN = int(lineLeadAngGREEN / 2)
HSV_hueTailGREEN = int(lineTailAngGREEN / 2)
HSV_saturationGREEN = saturationHalfGREEN * 2

HSV_hueLeadRED = int(lineLeadAngRED / 2)
HSV_hueTailRED = int(lineTailAngRED / 2)
HSV_saturationRED = saturationHalfRED * 2

# 차선 감지 ROI
WIN_XL = APP_CFG.ROI_XL
WIN_XR = APP_CFG.ROI_XR
WIN_YU = APP_CFG.ROI_YU
WIN_YD = APP_CFG.ROI_YD

# GPIO Pin Number
MOTOR_L_PWM = HW_CFG.MOTOR_L_PWM
MOTOR_L_DIR = HW_CFG.MOTOR_L_DIR
MOTOR_R_PWM = HW_CFG.MOTOR_R_PWM
MOTOR_R_DIR = HW_CFG.MOTOR_R_DIR
BUZZER = HW_CFG.BUZZER
MUSIC = HW_CFG.MUSIC
LAMP_R_YELLOW = HW_CFG.LAMP_R_YELLOW
LAMP_L_YELLOW = HW_CFG.LAMP_L_YELLOW
LAMP_BRAKE = HW_CFG.LAMP_BRAKE
LIGHT_SENSOR = HW_CFG.LIGHT_SENSOR

GPIO.setwarnings(False)
GPIO.setmode(GPIO.BCM)
GPIO.setup(MOTOR_L_PWM, GPIO.OUT)
GPIO.setup(MOTOR_L_DIR, GPIO.OUT)
GPIO.setup(MOTOR_R_PWM, GPIO.OUT)
GPIO.setup(MOTOR_R_DIR, GPIO.OUT)
GPIO.setup(BUZZER, GPIO.OUT)
GPIO.setup(MUSIC, GPIO.OUT)
GPIO.setup(LAMP_R_YELLOW, GPIO.OUT)
GPIO.setup(LAMP_L_YELLOW, GPIO.OUT)
GPIO.setup(LAMP_BRAKE, GPIO.OUT)
GPIO.setup(LIGHT_SENSOR, GPIO.IN, pull_up_down=GPIO.PUD_UP)

MOTOR_L = GPIO.PWM(MOTOR_L_PWM, HW_CFG.MOTOR_PWM_FREQUENCY)
MOTOR_R = GPIO.PWM(MOTOR_R_PWM, HW_CFG.MOTOR_PWM_FREQUENCY)
MOTOR_L.start(0)
MOTOR_R.start(0)

tof = VL53L0X.VL53L0X()
viewWin = np.zeros((APP_CFG.LCD_HEIGHT, APP_CFG.LCD_WIDTH, 3), np.uint8)
msgBoxL = cv.imread('./_IMAGE/armL.png', cv.IMREAD_COLOR)
msgBoxR = cv.imread('./_IMAGE/armR.png', cv.IMREAD_COLOR)
if msgBoxL is None:
    msgBoxL = np.zeros((APP_CFG.LCD_HEIGHT, APP_CFG.SIDE_PANEL_WIDTH, 3), np.uint8)
if msgBoxR is None:
    msgBoxR = np.zeros((APP_CFG.LCD_HEIGHT, APP_CFG.SIDE_PANEL_WIDTH, 3), np.uint8)

cv.namedWindow('Out', cv.WND_PROP_FULLSCREEN)
cv.setWindowProperty('Out', cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

class NvidiaModel(nn.Module):
    def __init__(self, output_dim=DL_CFG.MODEL_OUTPUT_DIM):
        super(NvidiaModel, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=(5, 5), stride=(2, 2)),
            nn.ELU(inplace=True),
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(inplace=True),
            nn.Dropout(0.2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
            nn.ELU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features=18 * 64, out_features=100),
            nn.ELU(inplace=True),
            nn.Linear(in_features=100, out_features=50),
            nn.ELU(inplace=True),
            nn.Linear(in_features=50, out_features=10),
            nn.ELU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Linear(in_features=10, out_features=output_dim)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = x.reshape(x.shape[0], -1)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

def load_nvidia_model(model_path):
    global MODEL_OUTPUT_DIM
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    state_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError('지원하지 않는 PyTorch checkpoint 형식입니다.')

    output_weight = state_dict.get('layer3.0.weight')
    if output_weight is None:
        raise KeyError('layer3.0.weight를 찾을 수 없습니다.')

    MODEL_OUTPUT_DIM = int(output_weight.shape[0])
    restored_model = NvidiaModel(output_dim=MODEL_OUTPUT_DIM)
    restored_model.load_state_dict(state_dict)
    restored_model.eval()
    return restored_model

PATH_DEVIATION_ON = True
laneDetector = None
lastDeviation = None
lastDeviationRaw = None

def initLaneDetector():
    global laneDetector
    if not PATH_DEVIATION_ON:
        return
    laneDetector = LaneDetector(LaneDetectorConfig(
        input_is_yuv=False,
        roi_top_ratio=0.0,
    ))
    print('경로 편차 측정 활성화 (차선 검출 기반)')

def measureDeviation(trackImage):
    global lastDeviation, lastDeviationRaw
    if not PATH_DEVIATION_ON or laneDetector is None:
        return None, False

    dev = laneDetector.compute_deviation(trackImage)
    usedFallback = (dev is not None and laneDetector.last_used_fallback)

    lastDeviation = dev
    lastDeviationRaw = None if usedFallback else dev
    return dev, usedFallback

def printDeviationSummary():
    if not PATH_DEVIATION_ON or laneDetector is None:
        return
    s = laneDetector.summary()
    print('\n=== 실차 경로 편차 요약 ===')
    print('총 프레임:', s['total_frames'])
    print(f"차선 검출률: {s['detection_rate']:.1f}%")
    if s['mean_abs_deviation'] is not None:
        print(f"평균 절대 편차: {s['mean_abs_deviation']:.2f} px")
        print(f"최대 절대 편차: {s['max_abs_deviation']:.2f} px")
        print(f"편차 표준편차:   {s['std_deviation']:.2f} px")
    print(f"급조향 발생:    {s['sudden_changes']}회 ({s['sudden_change_rate_per_1000f']:.1f}/1000프레임)")
    print('==========================\n')

def initDriveLog():
    global driveLogPath, driveLogFile, driveLogWriter, driveSessionId, driveStartTime, driveFramesDir
    if not RESEARCH_LOG_ON:
        return
    driveSessionId = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    driveStartTime = time.time()

    # 주행 프레임 저장 폴더(세션별). 검출기 파라미터 변경 후 오프라인 재계산용.
    if SAVE_DRIVE_FRAMES:
        driveFramesDir = f'_frames_{driveSessionId}'
        os.makedirs(driveFramesDir, exist_ok=True)
        print(f'주행 프레임 저장 활성화 → {driveFramesDir}/')
    logDir = imageDir if imageDir and os.path.exists(imageDir) else '.'
    projectName = imageDir if imageDir else 'ManualMode'
    driveLogPath = os.path.join(logDir, f'_{projectName}_drive_log_{driveSessionId}.csv')

    driveLogFile = open(driveLogPath, 'w', newline='', encoding='utf-8')
    driveLogWriter = csv.writer(driveLogFile)
    driveLogWriter.writerow([
        'session_id', 'frame_id', 'timestamp', 'elapsed_sec', 'lap',
        'mode', 'model_available', 'model_file',
        'key_code', 'key_event', 'intervention_count',
        'traffic_signal', 'sign_red_delay', 'tof_flag', 'distance_mm',
        'cds', 'light_mode', 'light_car',
        'pred_angle_raw', 'pred_angle_gain', 'pred_speed_raw', 'run_speed', 'angle_gain',
        'left_motor_pwm', 'right_motor_pwm',
        'cycle_ms', 'inference_ms',
        'win_xl', 'win_xr', 'win_yu', 'win_yd',
        'tswin_xl', 'tswin_xr', 'tswin_yu', 'tswin_yd',
        'auto_frame_count', 'manual_frame_count', 'red_stop_frame_count', 'tof_stop_frame_count',
        'path_deviation_px', 'deviation_is_fallback'
    ])
    print('논문 실험용 주행 로그 파일:', driveLogPath)

def writeDriveLog(keyCode, keyEvent, autoRun, trafficSignal, signRedDelay, tofFlag, distance,
                  cdsState, predRaw, predGain, predSpeedRaw, commandSpeed,
                  leftPWM, rightPWM, cycleMs, inferenceMs,
                  pathDeviation=None, deviationFallback=False):
    if not RESEARCH_LOG_ON or driveLogWriter is None:
        return
    elapsed = time.time() - driveStartTime if driveStartTime else 0.0
    mode = 'AUTO' if autoRun else 'MANUAL'
    driveLogWriter.writerow([
        driveSessionId, frameId, datetime.datetime.now().isoformat(timespec='milliseconds'),
        f'{elapsed:.3f}', currentLap,
        mode, MODEL_FILE, modelFile,
        keyCode, keyEvent, interventionCount,
        trafficSignal, signRedDelay, tofFlag, distance,
        cdsState, lightMode, lightCar,
        '' if predRaw is None else f'{predRaw:.4f}', predGain,
        '' if predSpeedRaw is None else f'{predSpeedRaw:.4f}', commandSpeed, ANGLE_GAIN,
        leftPWM, rightPWM,
        cycleMs, f'{inferenceMs:.3f}',
        WIN_XL, WIN_XR, WIN_YU, WIN_YD,
        TSWIN_XL, TSWIN_XR, TSWIN_YU, TSWIN_YD,
        autoFrameCount, manualFrameCount, redStopFrameCount, tofStopFrameCount,
        '' if pathDeviation is None else f'{pathDeviation:.2f}',
        1 if deviationFallback else 0
    ])

def closeDriveLog():
    global driveLogFile
    if driveLogFile is not None:
        driveLogFile.flush()
        driveLogFile.close()
        driveLogFile = None
    print('\n=== 논문 실험용 자율주행 요약 ===')
    print('주행 로그 파일:', driveLogPath)
    print('총 프레임:', frameId)
    print('자율주행 프레임:', autoFrameCount)
    print('수동/정지 프레임:', manualFrameCount)
    print('수동 개입 횟수:', interventionCount)
    print('카메라 프레임 실패:', cameraFailCount)
    print('신호/정지 지연 프레임:', redStopFrameCount)
    print('ToF 정지 프레임:', tofStopFrameCount)
    print('마지막 Lap 번호:', currentLap)
    if modelInferCount:
        print(f'평균 모델 추론 시간: {modelInferTotalMs/modelInferCount:.2f} ms')
    print('최대 절대 예측 조향각:', maxAbsPredAngle)
    print('================================\n')
    printDeviationSummary()

def keyName(keyCode):
    if keyCode == -1 or keyCode == 255: return ''
    if keyCode == 80: return 'HOME/G_AUTO_START'
    if keyCode == 81: return 'LEFT'
    if keyCode == 82: return 'UP'
    if keyCode == 83: return 'RIGHT'
    if keyCode == 84: return 'DOWN'
    if keyCode == 87: return 'END'
    if keyCode == 32: return 'SPACE_STOP'
    if 32 <= keyCode <= 126: return chr(keyCode)
    return str(keyCode)

def motorRun(leftMotor, rightMotor):
    amotorL = min(abs(leftMotor), HW_CFG.MOTOR_PWM_MAX)
    amotorR = min(abs(rightMotor), HW_CFG.MOTOR_PWM_MAX)
    
    GPIO.output(MOTOR_L_DIR, GPIO.HIGH if leftMotor >= 0 else GPIO.LOW)
    MOTOR_L.ChangeDutyCycle(amotorL)

    GPIO.output(MOTOR_R_DIR, GPIO.HIGH if rightMotor >= 0 else GPIO.LOW)
    MOTOR_R.ChangeDutyCycle(amotorR)

def main():
    global lightMode, lightCar
    global currentLap, frameId, autoFrameCount, manualFrameCount, interventionCount
    global cameraFailCount, redStopFrameCount, tofStopFrameCount
    global modelInferCount, modelInferTotalMs, maxAbsPredAngle

    RESTART_NUM = AR_CFG.RED_SIGNAL_RESTART_FRAMES
    YM = AR_CFG.MOTOR_BAR_CENTER_Y
    nx.lamp(0, 0, 0)
    
    camera = cv.VideoCapture(APP_CFG.CAMERA_INDEX, cv.CAP_V4L)
    camera.set(cv.CAP_PROP_FRAME_WIDTH, APP_CFG.CAMERA_WIDTH)
    camera.set(cv.CAP_PROP_FRAME_HEIGHT, APP_CFG.CAMERA_HEIGHT)

    VIEW_FLAG = True
    autoRun = False
    BLINK_LEFT = False
    BLINK_RIGHT = False
    TOF_FLAG = False
    YUV = False
    CDS = 'NIGHT'
    preKey = ord(' ')
    PWM_COLOR = GREEN
    mL = 0
    mR = 0
    signRedDelay = 0
    angle = 0
    timeCycle = 0
    horn = 0
    blinkCount = 0

    initDriveLog()
    initLaneDetector()

    while(camera.isOpened()):
        startCycle = time.time()
        distance = -1
        trafficSignal = ''
        predAngleRaw = None
        predAngleGain = angle
        predSpeedRaw = None
        commandSpeed = RUN_SPEED
        inferenceMs = 0.0
        keyEvent = ''

        if GPIO.input(LIGHT_SENSOR): CDS = 'NIGHT'
        else: CDS = 'DAY'

        if lightMode == 'AUTO' and CDS == 'NIGHT': nx.lamp(255, 255, 255)
        elif lightCar == 'ON': nx.lamp(255, 255, 255)
        elif lightCar == 'POLICE 1': nx.policeCar1()
        elif lightCar == 'POLICE 2': nx.policeCar2()
        elif lightCar == 'POLICE 3': nx.policeCar3()
        elif lightCar == 'AMBULANCE': nx.ambulance()
        elif lightCar == 'FIRE TRUCK': nx.fireTruck()
        elif lightCar == 'PATROL': nx.patrol()
        elif lightCar == 'OFF' or lightCar == ' ': nx.lamp(0, 0, 0)

        if horn:
            GPIO.output(BUZZER, GPIO.HIGH)
            horn -= 1
        else:
            GPIO.output(BUZZER, GPIO.LOW)

        ret, frame = camera.read()
        if not ret or frame is None:
            cameraFailCount += 1
            print('[오류] 카메라 프레임을 읽지 못했습니다. 자율주행을 종료합니다.')
            break

        viewWin[0:480, 80:720] = frame[0:480, 0:640]
        viewWin[0:480, 0:80] = msgBoxL
        viewWin[0:480, 720:800] = msgBoxR

        # 1) 트랙 ROI 원본 잘라내기 (BGR)
        track_roi_raw = viewWin[WIN_YU:WIN_YD, WIN_XL+80:WIN_XR+80]

        # 2) 모델 입력 해상도로 리사이즈 (학습과 100% 동일한 전처리의 첫 단계)
        image_resized = cv.resize(track_roi_raw, (APP_CFG.MODEL_INPUT_WIDTH, APP_CFG.MODEL_INPUT_HEIGHT))

        # 3) 차선 편차 측정은 학습 해상도(200x66) BGR에서 수행한다.
        #    LaneDetectorConfig의 폭/간격 파라미터가 200x66 기준으로 튜닝돼 있어,
        #    오프라인 lane_detector.py 실행과 픽셀 단위가 일치한다. (input_is_yuv=False 유지)
        pathDeviation, deviationFallback = measureDeviation(image_resized)

        # 주행 프레임(200x66 BGR) 저장 — autoRun일 때만. 오프라인 편차 재계산용 원본.
        if SAVE_DRIVE_FRAMES and autoRun and driveFramesDir is not None:
            cv.imwrite(os.path.join(driveFramesDir, f'{frameId:06d}.png'), image_resized)

        # 4) 모델 입력 전처리 파이프라인 (학습과 100% 동일하게 일치)
        imageYUV = cv.cvtColor(image_resized, cv.COLOR_BGR2YUV)
        imageYUVG = cv.GaussianBlur(imageYUV, APP_CFG.GAUSSIAN_BLUR_KERNEL, 0)
        procImg = imageYUVG / APP_CFG.NORMALIZE_DIVISOR

        # 신호등 인식
        v = ts.trafficSign(viewWin, VIEW_FLAG,
                           TSWIN_XL, TSWIN_YU, TSWIN_XR, TSWIN_YD,
                           minRadius, maxRadius,
                           HSV_hueLeadGREEN, HSV_hueTailGREEN,
                           HSV_hueLeadRED, HSV_hueTailRED,
                           HSV_saturationGREEN, HSV_saturationRED,
                           HSV_valueGREEN, HSV_valueRED,
                           THRESH_GREEN*10, THRESH_RED*10)
        trafficSignal = v

        if v == 'G': signRedDelay = 0
        if v == 'R' and preKey == 80: signRedDelay = RESTART_NUM
        if signRedDelay > 0: signRedDelay -= 1
        if signRedDelay > 0:
            cv.putText(viewWin, f'{signRedDelay:2d}', (750, 110), cv.FONT_HERSHEY_PLAIN, 1, YELLOW)

        # 깜빡이
        blinkCount = (blinkCount + 1) % 10
        if blinkCount > 5:
            if BLINK_LEFT: GPIO.output(LAMP_L_YELLOW, GPIO.HIGH); viewWin[350:380, 10:70] = YELLOW
            if BLINK_RIGHT: GPIO.output(LAMP_R_YELLOW, GPIO.HIGH); viewWin[350:380, 730:790] = YELLOW
        else:
            GPIO.output(LAMP_R_YELLOW, GPIO.LOW); GPIO.output(LAMP_L_YELLOW, GPIO.LOW)

        # ToF 거리 센서
        if TOF_FLAG:
            distance = tof.get_distance()
            if VIEW_FLAG:
                txt = f'{distance:3d} MM' if distance < 2000 else '--- MM'
                cv.putText(viewWin, txt, (320, 165), cv.FONT_HERSHEY_COMPLEX_SMALL, 2, WHITE)
            if distance < HORN_DISTANCE:
                cv.circle(viewWin, (400, 160), 100, RED, 25)
                cv.line(viewWin, (400+70, 160-70), (400-70, 160+70), RED, 25)
                GPIO.output(LAMP_BRAKE, GPIO.HIGH)
                if preKey in (80, 82): horn = True
                if distance < STOP_DISTANCE:
                    horn = False
                    if preKey in (80, 82):
                        signRedDelay = RESTART_NUM
                        tofStopFrameCount += 1
                        BLINK_LEFT = True; BLINK_RIGHT = True
            else:
                GPIO.output(LAMP_BRAKE, GPIO.LOW)

        # UI 오버레이
        if VIEW_FLAG:
            if MODEL_FILE:
                cv.putText(viewWin, f'{imageDir}', (90, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, MAGENTA)
            cv.putText(viewWin, f'{timeCycle:4d} mS', (340, 20), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, WHITE)
            cv.putText(viewWin, f'{lightMode} {CDS} {lightCar}', (510, 475), cv.FONT_HERSHEY_PLAIN, 1, GREEN)
            cv.putText(viewWin, 'Lane Detect Area', (WIN_XL+90, WIN_YU+18), cv.FONT_HERSHEY_PLAIN, 1, YELLOW)
            cv.rectangle(viewWin, (WIN_XL+80, WIN_YU), (WIN_XR+80, WIN_YD), YELLOW, 1)
            cv.putText(viewWin, 'Traffic Siginal Area', (TSWIN_XL+90, TSWIN_YU+18), cv.FONT_HERSHEY_PLAIN, 1, CYAN)
            cv.rectangle(viewWin, (TSWIN_XL+80, TSWIN_YU), (TSWIN_XR+80, TSWIN_YD), CYAN, 1)

            # 디스플레이 썸네일
            viewWin[407:407+66, 300:300+200] = image_resized
            if (lightMode == 'AUTO' and CDS == 'NIGHT') or lightCar == 'ON':
                cv.rectangle(viewWin, (300-2, 407-2), (300+200+2, 407+66+2), WHITE, 3)

        # 자율주행 모델 추론
        if MODEL_FILE and autoRun:
            X = torch.as_tensor(np.asarray([procImg]), dtype=torch.float32).permute(0, 3, 1, 2)
            inferStart = time.time()
            with torch.no_grad():
                prediction = model(X)[0].detach().cpu().numpy()

            if MODEL_OUTPUT_DIM >= 2:
                predAngleRaw = float(prediction[0]) * DL_CFG.LABEL_SCALE
                predSpeedRaw = float(prediction[1]) * DL_CFG.LABEL_SCALE
            else:
                predAngleRaw = float(prediction[0])
                if AR_CFG.LEGACY_SINGLE_OUTPUT_IS_NORMALIZED:
                    predAngleRaw *= DL_CFG.LABEL_SCALE

            angle = clamp_int(predAngleRaw * ANGLE_GAIN, APP_CFG.STEERING_MIN, APP_CFG.STEERING_MAX)
            
            if AR_CFG.USE_PREDICTED_SPEED and predSpeedRaw is not None:
                commandSpeed = clamp_int(predSpeedRaw, AR_CFG.PREDICTED_SPEED_MIN, AR_CFG.PREDICTED_SPEED_MAX)
            else:
                commandSpeed = RUN_SPEED

            inferenceMs = (time.time() - inferStart) * 1000.0
            predAngleGain = angle
            modelInferCount += 1
            modelInferTotalMs += inferenceMs
            if abs(angle) > maxAbsPredAngle:
                maxAbsPredAngle = abs(angle)

            if VIEW_FLAG:
                cv.putText(viewWin, f'{angle:3d}', (260, 380), cv.FONT_HERSHEY_COMPLEX_SMALL, 6, WHITE)

            mL = clamp_int(commandSpeed + angle, HW_CFG.MOTOR_PWM_MIN, HW_CFG.MOTOR_PWM_MAX)
            mR = clamp_int(commandSpeed - angle, HW_CFG.MOTOR_PWM_MIN, HW_CFG.MOTOR_PWM_MAX)

        PWM_COLOR = RED if autoRun else GREEN
        if mL >= 0: ys, ye = YM - mL, YM
        else: ys, ye = YM, YM - mL
        cv.rectangle(viewWin, (30, ys), (50, ye), PWM_COLOR, -1)

        if mR >= 0: ys, ye = YM - mR, YM
        else: ys, ye = YM, YM - mR
        cv.rectangle(viewWin, (750, ys), (770, ye), PWM_COLOR, -1)

        cv.putText(viewWin, f'{mL:3d}', (15, 340), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, WHITE)
        cv.putText(viewWin, f'{mR:3d}', (735, 340), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, WHITE)

        if signRedDelay > 0:
            redStopFrameCount += 1
            if autoRun or preKey == 82:
                mL = 0; mR = 0

        motorRun(mL, mR)

        if preKey != 84:
            if abs(mL - mR) > 9:
                BLINK_LEFT = (mL - mR) <= 0
                BLINK_RIGHT = (mL - mR) > 0
            else:
                BLINK_LEFT = False; BLINK_RIGHT = False

        cv.imshow('Out', viewWin)

        keyBoard = cv.waitKey(1)
        keyEvent = keyName(keyBoard)

        if keyBoard in (0x1B, 0x09, ord('x'), ord('X')):
            keyEvent = 'EXIT'
            break
        elif keyBoard in (80, ord('g'), ord('G')):
            keyEvent = 'AUTO_START'
            autoRun = True; signRedDelay = 0; preKey = 80
        elif keyBoard in (ord('c'), ord('C')):
            keyEvent = 'VIEW_TOGGLE'
            VIEW_FLAG = not VIEW_FLAG
        elif keyBoard in (ord('h'), ord('H')):
            keyEvent = 'HORN'
            horn = 5
        elif keyBoard in (ord('l'), ord('L')):
            keyEvent = 'LIGHT_MODE_CHANGE'
            modes = [' ', 'ON', 'OFF', 'POLICE 1', 'POLICE 2', 'POLICE 3', 'AMBULANCE', 'FIRE TRUCK', 'PATROL']
            idx = (modes.index(lightCar) + 1) % len(modes) if lightCar in modes else 0
            lightCar = modes[idx]
        elif keyBoard in (ord('a'), ord('A')):
            keyEvent = 'AUTO_LIGHT_TOGGLE'
            lightMode = 'AUTO' if lightMode == 'MANU' else 'MANU'
        elif keyBoard in (ord('d'), ord('D')):
            keyEvent = 'TOF_TOGGLE'
            TOF_FLAG = not TOF_FLAG
            if TOF_FLAG: tof.start_ranging(VL53L0X.VL53L0X_HIGH_SPEED_MODE)
            else: tof.stop_ranging()
        elif keyBoard in (ord('y'), ord('Y')):
            keyEvent = 'YUV_TOGGLE'
            YUV = not YUV
        elif keyBoard in (ord('n'), ord('N')):
            currentLap += 1
            keyEvent = f'LAP_{currentLap}_START'
            print(f'Lap {currentLap} 시작')
        elif keyBoard == 82:  # UP
            if autoRun: interventionCount += 1
            keyEvent = 'MANUAL_UP'
            autoRun = False
            if preKey == ord(' '): mL = 80; mR = 80; signRedDelay = 0; preKey = keyBoard
            else: preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC, GPIO.LOW)
        elif keyBoard == 84:  # DOWN
            if autoRun: interventionCount += 1
            keyEvent = 'MANUAL_DOWN'
            autoRun = False
            if preKey == ord(' '):
                mL = -70; mR = -70; signRedDelay = 0; preKey = keyBoard
                GPIO.output(MUSIC, GPIO.HIGH); BLINK_LEFT = True; BLINK_RIGHT = True
            else: preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC, GPIO.LOW)
        elif keyBoard == 81:  # LEFT
            if autoRun: interventionCount += 1
            keyEvent = 'MANUAL_LEFT'
            autoRun = False
            if preKey == ord(' '): mL = -70; mR = 70; signRedDelay = 0; preKey = keyBoard
            else: preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC, GPIO.LOW)
        elif keyBoard == 83:  # RIGHT
            if autoRun: interventionCount += 1
            keyEvent = 'MANUAL_RIGHT'
            autoRun = False
            if preKey == ord(' '): mL = 70; mR = -70; signRedDelay = 0; preKey = keyBoard
            else: preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC, GPIO.LOW)
        elif keyBoard in (ord(' '), 87):  # STOP
            if autoRun: interventionCount += 1
            keyEvent = 'STOP'
            autoRun = False; preKey = ord(' '); mL = 0; mR = 0; GPIO.output(MUSIC, GPIO.LOW)

        if autoRun: autoFrameCount += 1
        else: manualFrameCount += 1

        endCycle = time.time()
        timeCycle = int((endCycle - startCycle) * 1000)
        writeDriveLog(keyBoard, keyEvent, autoRun, trafficSignal, signRedDelay, TOF_FLAG, distance,
                      CDS, predAngleRaw, predAngleGain, predSpeedRaw, commandSpeed,
                      mL, mR, timeCycle, inferenceMs,
                      pathDeviation, deviationFallback)
        frameId += 1

    if TOF_FLAG: tof.stop_ranging()
    motorRun(0, 0)
    nx.lamp(0, 0, 0)
    camera.release()
    closeDriveLog()
    GPIO.cleanup()
    cv.destroyAllWindows()

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        imageDir = sys.argv[1]
    
    username = os.getlogin()
    file_listD = glob.glob(f'/home/{username}/{downLoads}/_{imageDir}_*.pt')
    file_listC = glob.glob(os.path.join(os.getcwd(), f'{imageDir}/_{imageDir}_*.pt'))
    file_list = file_listD + file_listC
    sorted_files = sorted([(f, os.path.getmtime(f)) for f in file_list], key=lambda x: x[1])

    def listingFile(files, startN, count, cline):
        viewWin[:, :] = BLACK
        for i in range(count):
            color = YELLOW if i == cline else GRAY
            cv.putText(viewWin, f'{startN+i+1:2d}: {files[startN+i][0]}', (10, 20+20*i), cv.FONT_HERSHEY_COMPLEX_SMALL, 1, color)
        cv.imshow('Out', viewWin)

    c = len(sorted_files)
    if c:
        maxRows = 15
        startN = max(0, c - maxRows)
        count = min(c, maxRows)
        cline = count - 1

        while True:
            if c == 1:
                break
            listingFile(sorted_files, startN, count, cline)
            keyBoard = cv.waitKey(0) & 0xFF
            if keyBoard == 13: break
            elif keyBoard == 82 and cline > 0: cline -= 1
            elif keyBoard == 84 and cline < count - 1: cline += 1

        modelFile = sorted_files[startN + cline][0]
        try:
            model = load_nvidia_model(modelFile)
            MODEL_FILE = True
            print(f'\n학습 모델 파일 {modelFile} 로드 완료 (출력 차원: {MODEL_OUTPUT_DIM})\n')
        except Exception as e:
            MODEL_FILE = False
            print(f'\n[오류] 모델 로드 실패: {e}\n')
            modelFile = 'Manual Mode'
    else:
        print('\n학습 모델 파일이 없습니다.\n')
        modelFile = 'Manual Mode'

    # 트랙 및 신호등 pickle 설정 로드
    t = f'./{imageDir}/_{imageDir}_track.pickle'
    if os.path.exists(t):
        try:
            with open(t, 'rb') as f:
                d = pickle.load(f)
                if len(d) >= 6:
                    WIN_XL, WIN_XR, WIN_YU, WIN_YD = d[0], d[1], d[2], d[3]
                    if d[4]: lightMode = 'AUTO'
                    elif d[5]: lightCar = 'ON'
            print('트랙 ROI 설정 완료:', WIN_XL, WIN_XR, WIN_YU, WIN_YD)
        except Exception as e:
            print('[경고] 트랙 파일 읽기 실패:', e)

    fn = f'./{imageDir}/_{imageDir}_TS.pickle'
    if os.path.exists(fn):
        try:
            with open(fn, 'rb') as fr:
                d = pickle.load(fr)
            if len(d) >= 16:
                TSWIN_XL, TSWIN_YU, TSWIN_XR, TSWIN_YD = d[0], d[1], d[2], d[3]
                minRadius, maxRadius = d[4], d[5]
                HSV_hueLeadGREEN, HSV_hueTailGREEN = int(d[6]/2), int(d[7]/2)
                HSV_hueLeadRED, HSV_hueTailRED = int(d[8]/2), int(d[9]/2)
                HSV_saturationGREEN, HSV_saturationRED = d[10]*2, d[11]*2
                HSV_valueGREEN, HSV_valueRED = d[12], d[13]
                THRESH_GREEN, THRESH_RED = d[14], d[15]
            print('신호등 파라미터 로드 완료')
        except Exception as e:
            print('[경고] 신호등 파일 읽기 실패:', e)

    main()