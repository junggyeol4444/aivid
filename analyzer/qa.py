# 영상 내용 질의응답 모듈
# BLIP 모델의 Visual Question Answering(VQA) 기능을 활용하여
# 영상에 대한 자연어 질문에 답변합니다.

import torch
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from analyzer.video_loader import extract_frames
from config import DEFAULT_MODEL, MODEL_IDS, DEVICE, MAX_FRAMES, MAX_QA_TOKENS


# VQA 모델 캐시
_vqa_processor = None
_vqa_model = None
_vqa_model_key = None


def _load_vqa_model(model_key: str = DEFAULT_MODEL):
    """
    VQA 전용 BLIP 모델을 로드합니다.

    Args:
        model_key: 사용할 모델 키
    """
    global _vqa_processor, _vqa_model, _vqa_model_key

    if _vqa_model is not None and _vqa_model_key == model_key:
        return _vqa_processor, _vqa_model

    # VQA용 모델 ID (base 모델만 VQA 지원, blip2는 별도 처리)
    if model_key == "blip2":
        # blip2는 VQA 전용 체크포인트 사용
        vqa_model_id = "Salesforce/blip2-opt-2.7b"
        # blip2는 BlipForConditionalGeneration 사용 (VQA도 가능)
        from transformers import BlipForConditionalGeneration
        print(f"  VQA 모델 로드 중: {vqa_model_id} (디바이스: {DEVICE})")
        _vqa_processor = BlipProcessor.from_pretrained(vqa_model_id)
        _vqa_model = BlipForConditionalGeneration.from_pretrained(vqa_model_id)
    else:
        # base 모델: VQA 전용 체크포인트 사용
        vqa_model_id = "Salesforce/blip-vqa-base"
        print(f"  VQA 모델 로드 중: {vqa_model_id} (디바이스: {DEVICE})")
        _vqa_processor = BlipProcessor.from_pretrained(vqa_model_id)
        _vqa_model = BlipForQuestionAnswering.from_pretrained(vqa_model_id)

    _vqa_model.to(DEVICE)
    _vqa_model.eval()
    _vqa_model_key = model_key

    print("  VQA 모델 로드 완료!")
    return _vqa_processor, _vqa_model


def ask_frame(image: Image.Image, question: str, model_key: str = DEFAULT_MODEL) -> str:
    """
    단일 프레임에 대해 질문에 답변합니다.

    Args:
        image: PIL.Image 객체
        question: 질문 문자열 (영어)
        model_key: 사용할 모델 키

    Returns:
        답변 문자열
    """
    processor, model = _load_vqa_model(model_key)

    inputs = processor(images=image, text=question, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_QA_TOKENS)

    answer = processor.decode(output[0], skip_special_tokens=True)
    return answer


def ask_about_video(
    video_path: str,
    question: str,
    interval_sec: float = 2,
    model_key: str = DEFAULT_MODEL,
) -> str:
    """
    영상 전체에 대해 질문에 답변합니다.
    여러 프레임에서 답변을 수집하고 가장 많이 나온 답변을 반환합니다.

    Args:
        video_path: 영상 파일 경로
        question: 한국어 또는 영어 질문
        interval_sec: 프레임 추출 간격 (초)
        model_key: 사용할 모델 키

    Returns:
        종합된 답변 문자열
    """
    print(f"\n[질의응답] 질문: {question}")
    print("  프레임 추출 중...")

    frames = extract_frames(video_path, interval_sec)

    # 최대 프레임 수 제한
    if len(frames) > MAX_FRAMES:
        step = len(frames) // MAX_FRAMES
        frames = frames[::step][:MAX_FRAMES]

    print(f"  {len(frames)}개 프레임에서 답변 수집 중...")

    # 모델은 영어 질문에 최적화되어 있으므로 영어로 변환 안내
    answers = []
    for i, frame_info in enumerate(frames):
        answer = ask_frame(frame_info["frame"], question, model_key)
        if answer:
            answers.append(answer.lower().strip())
        print(f"  [{i + 1}/{len(frames)}] {frame_info['timestamp']:.1f}초: {answer}")

    if not answers:
        return "답변을 찾을 수 없습니다."

    # 가장 많이 등장한 답변 반환
    answer_count: dict = {}
    for ans in answers:
        answer_count[ans] = answer_count.get(ans, 0) + 1

    best_answer = max(answer_count, key=lambda x: answer_count[x])

    # 결과 요약
    unique_answers = list(dict.fromkeys(answers))  # 순서 유지하며 중복 제거
    summary = (
        f"가장 많은 프레임에서의 답변: '{best_answer}'\n"
        f"(수집된 고유 답변들: {', '.join(unique_answers[:5])})"
    )

    return summary
