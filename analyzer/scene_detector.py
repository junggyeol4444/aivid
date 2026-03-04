# 장면 전환 감지 모듈
# 프레임 간 차이를 분석하여 영상 내 장면 전환(scene change)을 감지합니다.

import cv2
import numpy as np
from PIL import Image
from analyzer.video_loader import load_video


def compute_frame_difference(frame_a: np.ndarray, frame_b: np.ndarray) -> float:
    """
    두 프레임 간의 차이를 0~1 사이 값으로 계산합니다.
    히스토그램 비교 방식을 사용합니다.

    Args:
        frame_a: 첫 번째 프레임 (BGR numpy 배열)
        frame_b: 두 번째 프레임 (BGR numpy 배열)

    Returns:
        0~1 사이의 차이 값 (1에 가까울수록 큰 변화)
    """
    # 그레이스케일로 변환
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)

    # 히스토그램 계산
    hist_a = cv2.calcHist([gray_a], [0], None, [256], [0, 256])
    hist_b = cv2.calcHist([gray_b], [0], None, [256], [0, 256])

    # 히스토그램 정규화
    cv2.normalize(hist_a, hist_a)
    cv2.normalize(hist_b, hist_b)

    # 상관관계 비교 (1 = 동일, -1 = 완전 반대)
    correlation = cv2.compareHist(hist_a, hist_b, cv2.HISTCMP_CORREL)

    # 차이 값으로 변환 (0 = 동일, 1 = 완전히 다름)
    difference = max(0.0, min(1.0, 1.0 - correlation))
    return round(difference, 4)


def detect_scene_changes(
    video_path: str,
    threshold: float = 0.3,
    sample_interval: float = 0.5,
) -> list:
    """
    영상에서 장면 전환을 감지합니다.

    일정 간격으로 프레임을 비교하여, 차이가 임계값(threshold)을 초과하면
    장면 전환으로 판정합니다.

    Args:
        video_path: 영상 파일 경로
        threshold: 장면 전환 판정 임계값 (0~1, 기본값: 0.3)
        sample_interval: 프레임 샘플링 간격 (초, 기본값: 0.5)

    Returns:
        장면 전환 정보 리스트:
        [{"timestamp": float, "difference": float, "scene_index": int}, ...]
    """
    cap = load_video(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, int(fps * sample_interval))

    print(f"  장면 전환 감지 중... (임계값: {threshold}, 간격: {sample_interval}초)")

    scene_changes = []
    prev_frame = None
    scene_index = 0
    frame_idx = 0

    # 첫 번째 프레임은 항상 첫 장면의 시작
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    ret, first_frame = cap.read()
    if ret:
        scene_changes.append({
            "timestamp": 0.0,
            "difference": 0.0,
            "scene_index": scene_index,
        })
        prev_frame = first_frame

    frame_idx = frame_interval
    while frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, current_frame = cap.read()
        if not ret:
            break

        if prev_frame is not None:
            diff = compute_frame_difference(prev_frame, current_frame)

            if diff >= threshold:
                scene_index += 1
                timestamp = frame_idx / fps if fps > 0 else 0
                scene_changes.append({
                    "timestamp": round(timestamp, 2),
                    "difference": diff,
                    "scene_index": scene_index,
                })

        prev_frame = current_frame
        frame_idx += frame_interval

    cap.release()
    print(f"  총 {len(scene_changes)}개의 장면을 감지했습니다.")
    return scene_changes


def get_scene_thumbnails(
    video_path: str,
    scene_changes: list,
) -> list:
    """
    각 장면 전환 시점의 프레임을 PIL Image로 추출합니다.

    Args:
        video_path: 영상 파일 경로
        scene_changes: detect_scene_changes()의 결과

    Returns:
        [{"timestamp": float, "scene_index": int, "thumbnail": PIL.Image}, ...]
    """
    cap = load_video(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    thumbnails = []
    for scene in scene_changes:
        timestamp = scene["timestamp"]
        frame_idx = int(timestamp * fps) if fps > 0 else 0

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(frame_rgb)
            thumbnails.append({
                "timestamp": timestamp,
                "scene_index": scene["scene_index"],
                "thumbnail": pil_image,
            })

    cap.release()
    return thumbnails
