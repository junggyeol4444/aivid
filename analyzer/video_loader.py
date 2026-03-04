# 영상 로드 및 프레임 추출 모듈
# OpenCV를 사용하여 영상 파일을 열고 일정 간격으로 프레임을 추출합니다.

import os
import cv2
from PIL import Image
from config import SUPPORTED_FORMATS


def load_video(path: str) -> cv2.VideoCapture:
    """
    영상 파일을 로드합니다.

    Args:
        path: 영상 파일 경로

    Returns:
        cv2.VideoCapture 객체

    Raises:
        FileNotFoundError: 파일이 존재하지 않을 경우
        ValueError: 지원하지 않는 파일 형식이거나 영상을 열 수 없을 경우
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"영상 파일을 찾을 수 없습니다: {path}")

    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {ext}\n"
            f"지원 형식: {', '.join(SUPPORTED_FORMATS)}"
        )

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError(f"영상 파일을 열 수 없습니다: {path}")

    return cap


def extract_frames(video_path: str, interval_sec: float = 2) -> list:
    """
    영상에서 일정 간격으로 프레임을 추출합니다.

    Args:
        video_path: 영상 파일 경로
        interval_sec: 프레임 추출 간격 (초)

    Returns:
        [{"timestamp": float, "frame": PIL.Image}, ...] 형태의 리스트
    """
    cap = load_video(video_path)

    # 영상 기본 정보 가져오기
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = total_frames / fps if fps > 0 else 0

    print(f"  영상 정보: {duration_sec:.1f}초, {fps:.1f}fps, 총 {total_frames}프레임")

    # 추출할 프레임 인덱스 계산
    frame_interval = max(1, int(fps * interval_sec))
    frames = []

    frame_idx = 0
    while True:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # BGR → RGB 변환 후 PIL Image로 변환
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(frame_rgb)

        timestamp = frame_idx / fps if fps > 0 else 0
        frames.append({
            "timestamp": round(timestamp, 2),
            "frame": pil_image,
        })

        frame_idx += frame_interval

    cap.release()
    print(f"  총 {len(frames)}개의 프레임을 추출했습니다.")
    return frames


def get_video_metadata(path: str) -> dict:
    """
    영상 파일의 메타데이터(해상도, fps, 길이 등)를 반환합니다.

    Args:
        path: 영상 파일 경로

    Returns:
        메타데이터 딕셔너리
    """
    cap = load_video(path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_sec = total_frames / fps if fps > 0 else 0

    cap.release()

    # 파일 크기
    file_size = os.path.getsize(path)
    if file_size >= 1024 * 1024:
        size_str = f"{file_size / (1024 * 1024):.1f} MB"
    else:
        size_str = f"{file_size / 1024:.1f} KB"

    # 재생 시간 포맷
    minutes = int(duration_sec // 60)
    seconds = int(duration_sec % 60)
    duration_str = f"{minutes}분 {seconds}초" if minutes > 0 else f"{seconds}초"

    return {
        "파일 경로": path,
        "해상도": f"{width} x {height}",
        "FPS": f"{fps:.1f}",
        "총 프레임 수": str(total_frames),
        "재생 시간": duration_str,
        "파일 크기": size_str,
    }
