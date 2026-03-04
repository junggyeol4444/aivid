# 프레임 분석 모듈
# Hugging Face의 BLIP 또는 BLIP-2 모델을 사용하여 이미지 프레임을 분석합니다.
# GPU 없이 CPU에서도 동작 가능합니다.

from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from config import DEFAULT_MODEL, MODEL_IDS, DEVICE, MAX_CAPTION_TOKENS


# 모델 및 프로세서 전역 캐시 (처음 한 번만 로드)
_processor = None
_model = None
_current_model_key = None


def _load_model(model_key: str = DEFAULT_MODEL):
    """
    BLIP 모델과 프로세서를 로드합니다. 이미 로드된 경우 캐시를 반환합니다.

    Args:
        model_key: 사용할 모델 키 ("base" 또는 "blip2")
    """
    global _processor, _model, _current_model_key

    if _model is not None and _current_model_key == model_key:
        return _processor, _model

    model_id = MODEL_IDS.get(model_key, MODEL_IDS[DEFAULT_MODEL])
    print(f"  모델 로드 중: {model_id} (디바이스: {DEVICE})")

    _processor = BlipProcessor.from_pretrained(model_id)
    _model = BlipForConditionalGeneration.from_pretrained(model_id)
    _model.to(DEVICE)
    _model.eval()
    _current_model_key = model_key

    print("  모델 로드 완료!")
    return _processor, _model


def analyze_frame(image: Image.Image, model_key: str = DEFAULT_MODEL) -> str:
    """
    단일 프레임을 분석하여 장면 설명 텍스트를 반환합니다.

    Args:
        image: PIL.Image 객체
        model_key: 사용할 모델 키

    Returns:
        장면 설명 문자열 (영어)
    """
    processor, model = _load_model(model_key)

    # 프로세서로 입력 준비
    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    # 캡션 생성
    import torch
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=MAX_CAPTION_TOKENS)

    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption


def detect_objects(image: Image.Image, model_key: str = DEFAULT_MODEL) -> list:
    """
    프레임 내 주요 객체를 감지합니다.
    BLIP 모델의 캡션을 파싱하여 명사(객체)를 추출하는 방식으로 동작합니다.

    Args:
        image: PIL.Image 객체
        model_key: 사용할 모델 키

    Returns:
        감지된 객체 이름 리스트 (영어 명사)
    """
    caption = analyze_frame(image, model_key)

    # 간단한 명사 추출: 캡션에서 단어를 분리하고 불용어 제거
    stopwords = {
        "a", "an", "the", "is", "are", "in", "on", "at", "of", "and",
        "with", "there", "this", "that", "it", "to", "be", "have",
        "has", "was", "were", "some", "many", "few", "two", "three",
    }
    words = caption.lower().split()
    objects = [w.strip(".,!?") for w in words if w.strip(".,!?") not in stopwords and len(w) > 2]

    # 중복 제거
    seen = set()
    unique_objects = []
    for obj in objects:
        if obj not in seen:
            seen.add(obj)
            unique_objects.append(obj)

    return unique_objects
