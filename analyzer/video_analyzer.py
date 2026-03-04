# 전체 영상 분석 통합 모듈
# 프레임 추출과 프레임 분석을 결합하여 영상 전체를 분석하고 요약합니다.

import json
from analyzer.video_loader import extract_frames
from analyzer.frame_analyzer import analyze_frame, detect_objects
from config import DEFAULT_MODEL, MAX_FRAMES, TOP_OBJECTS_COUNT


def analyze_video(video_path: str, interval_sec: float = 2, model_key: str = DEFAULT_MODEL) -> dict:
    """
    영상 파일 전체를 분석합니다.

    1. 일정 간격으로 프레임을 추출합니다.
    2. 각 프레임의 장면을 설명합니다.
    3. 전체 영상의 요약을 생성합니다.

    Args:
        video_path: 영상 파일 경로
        interval_sec: 프레임 추출 간격 (초)
        model_key: 사용할 모델 키

    Returns:
        분석 결과 딕셔너리:
        {
            "video_path": str,
            "total_frames_analyzed": int,
            "frames": [{"timestamp": float, "description": str, "objects": list}, ...],
            "summary": str,
        }
    """
    print(f"\n[영상 분석 시작] {video_path}")
    print(f"  설정: {interval_sec}초 간격, 모델: {model_key}")

    # 1단계: 프레임 추출
    print("\n[1단계] 프레임 추출 중...")
    all_frames = extract_frames(video_path, interval_sec)

    # 최대 프레임 수 제한 (균등 샘플링)
    if len(all_frames) > MAX_FRAMES:
        step = len(all_frames) // MAX_FRAMES
        all_frames = all_frames[::step][:MAX_FRAMES]
        print(f"  최대 프레임 수({MAX_FRAMES})로 제한: {len(all_frames)}개 선택")

    # 2단계: 각 프레임 분석
    print(f"\n[2단계] 프레임 분석 중 (총 {len(all_frames)}개)...")
    frame_results = []

    for i, frame_info in enumerate(all_frames):
        timestamp = frame_info["timestamp"]
        image = frame_info["frame"]

        print(f"  [{i + 1}/{len(all_frames)}] {timestamp:.1f}초 프레임 분석 중...", end=" ")
        description = analyze_frame(image, model_key)
        objects = detect_objects(image, model_key)
        print(f"완료 - {description[:50]}...")

        frame_results.append({
            "timestamp": timestamp,
            "description": description,
            "objects": objects,
        })

    # 3단계: 전체 요약 생성
    print("\n[3단계] 전체 영상 요약 생성 중...")
    summary = _generate_summary(frame_results)
    print(f"  요약 완료: {summary[:80]}...")

    result = {
        "video_path": video_path,
        "total_frames_analyzed": len(frame_results),
        "frames": frame_results,
        "summary": summary,
    }

    print("\n[영상 분석 완료]")
    return result


def _generate_summary(frame_results: list) -> str:
    """
    프레임별 분석 결과로부터 전체 영상 요약을 생성합니다.
    모든 프레임의 설명을 취합하여 주요 내용을 정리합니다.

    Args:
        frame_results: 프레임 분석 결과 리스트

    Returns:
        요약 문자열
    """
    if not frame_results:
        return "분석된 프레임이 없습니다."

    # 모든 객체 수집 및 빈도 계산
    object_count: dict = {}
    descriptions = []

    for frame in frame_results:
        descriptions.append(frame["description"])
        for obj in frame["objects"]:
            object_count[obj] = object_count.get(obj, 0) + 1

    # 자주 등장한 상위 5개 객체
    top_objects = sorted(object_count.items(), key=lambda x: x[1], reverse=True)[:TOP_OBJECTS_COUNT]
    top_object_names = [obj for obj, _ in top_objects]

    # 첫 번째, 중간, 마지막 장면 설명 활용
    n = len(descriptions)
    key_descriptions = []
    if n >= 1:
        key_descriptions.append(f"시작: {descriptions[0]}")
    if n >= 3:
        key_descriptions.append(f"중간: {descriptions[n // 2]}")
    if n >= 2:
        key_descriptions.append(f"마지막: {descriptions[-1]}")

    summary_parts = [
        f"총 {len(frame_results)}개의 프레임을 분석했습니다.",
    ]

    if top_object_names:
        summary_parts.append(f"주요 등장 요소: {', '.join(top_object_names)}")

    summary_parts.extend(key_descriptions)

    return " | ".join(summary_parts)


def save_results(result: dict, output_path: str):
    """
    분석 결과를 JSON 파일로 저장합니다.

    Args:
        result: 분석 결과 딕셔너리
        output_path: 저장할 JSON 파일 경로
    """
    # PIL Image 객체는 JSON 직렬화 불가 → frames에서 frame 키 제외
    serializable = {
        "video_path": result["video_path"],
        "total_frames_analyzed": result["total_frames_analyzed"],
        "frames": result["frames"],
        "summary": result["summary"],
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, ensure_ascii=False, indent=2)

    print(f"\n결과가 저장되었습니다: {output_path}")
