# 설정 파일: 모델 선택, 프레임 간격 등 전역 설정

# 기본 모델 설정
# "base": 가볍고 빠른 BLIP 기본 모델 (CPU 환경 권장)
# "blip2": 고성능 BLIP-2 모델 (GPU 권장)
DEFAULT_MODEL = "base"

# 모델 ID 매핑
MODEL_IDS = {
    "base": "Salesforce/blip-image-captioning-base",
    "blip2": "Salesforce/blip2-opt-2.7b",
}

# 프레임 추출 기본 간격 (초)
DEFAULT_FRAME_INTERVAL = 2

# 영상 파일 지원 확장자
SUPPORTED_FORMATS = [".mp4", ".avi", ".mov", ".mkv"]

# 디바이스 설정: "cpu" 또는 "cuda" (GPU)
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 최대 분석 프레임 수 (너무 많으면 느려짐)
MAX_FRAMES = 30

# 캡셔닝 생성 시 최대 토큰 수
MAX_CAPTION_TOKENS = 50

# 질의응답(VQA) 생성 시 최대 토큰 수
MAX_QA_TOKENS = 30

# 요약 및 UI에 표시할 상위 객체 수
TOP_OBJECTS_COUNT = 5

# 장면 전환 감지 기본 임계값 (0~1, 클수록 둔감)
SCENE_CHANGE_THRESHOLD = 0.3

# 장면 전환 감지 프레임 샘플링 간격 (초)
SCENE_SAMPLE_INTERVAL = 0.5

# 웹 UI 포트
WEB_PORT = 7860
