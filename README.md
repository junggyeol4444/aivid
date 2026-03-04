# 🎬 aivid - AI 영상 시각 분석기

인간이 영상을 직접 시청하고 분석하는 것처럼, AI가 영상의 시각적 내용을 이해하고 분석하는 프로그램입니다.
메타데이터(파일 크기, 코덱 등) 분석이 아닌, **영상 속 장면, 행동, 객체, 감정 등을 AI가 직접 이해**합니다.

## ✨ 주요 기능

- **영상 프레임 추출**: OpenCV를 사용하여 영상에서 일정 간격으로 프레임 자동 추출
- **장면 분석**: 각 프레임에서 객체 감지, 행동 인식, 장면 설명 생성
- **전체 요약**: 영상 전체의 스토리/내용 자동 요약
- **질의응답 (VQA)**: 영상 내용에 대해 자연어 질문 → 답변
- **장면 전환 감지**: 프레임 간 차이 분석으로 장면 변화 시점 자동 감지
- **HTML 리포트**: 분석 결과를 시각적인 HTML 리포트로 출력
- **영상 메타데이터**: 해상도, FPS, 재생 시간 등 영상 정보 조회
- **일괄 분석**: 여러 영상 파일을 한 번에 분석
- **CLI 인터페이스**: 터미널에서 간편하게 사용
- **웹 인터페이스**: Gradio 기반의 직관적인 웹 UI

## 🛠 기술 스택

| 구성 요소 | 라이브러리 |
|---|---|
| 프레임 추출 | OpenCV (`opencv-python`) |
| AI 모델 | Hugging Face Transformers (BLIP / BLIP-2) |
| 웹 UI | Gradio |
| 딥러닝 | PyTorch |

## 📋 시스템 요구사항

- **Python**: 3.9 이상
- **RAM**: 최소 4GB (BLIP base 모델), 16GB 권장 (BLIP-2 모델)
- **GPU**: 선택 사항. GPU 없이 CPU로도 동작하지만 속도가 느릴 수 있습니다.
  - GPU 사용 시: CUDA 11.7 이상, VRAM 8GB 이상 권장

## 📦 설치 방법

```bash
# 저장소 클론
git clone https://github.com/junggyeol4444/aivid.git
cd aivid

# 의존성 설치
pip install -r requirements.txt
```

## 🚀 사용 방법

### CLI (터미널)

```bash
# 영상 분석 (기본 2초 간격)
python main.py analyze --video my_video.mp4

# 영상 분석 (3초 간격, 결과를 JSON으로 저장)
python main.py analyze --video my_video.mp4 --interval 3 --output result.json

# 고성능 BLIP-2 모델로 분석
python main.py analyze --video my_video.mp4 --model blip2

# 영상 내용에 대해 질문하기
python main.py ask --video my_video.mp4 --question "What is the person doing?"

# 장면 전환 감지
python main.py scenes --video my_video.mp4
python main.py scenes --video my_video.mp4 --threshold 0.2 --sample-interval 1.0

# HTML 리포트 생성
python main.py report --video my_video.mp4 --output report.html

# 영상 메타데이터 조회
python main.py info --video my_video.mp4

# 여러 영상 일괄 분석
python main.py batch --videos v1.mp4 v2.mp4 v3.mp4 --output-dir results/

# 웹 UI 실행
python main.py web
```

### 옵션 설명

| 옵션 | 설명 | 기본값 |
|---|---|---|
| `--video`, `-v` | 분석할 영상 파일 경로 | (필수) |
| `--interval`, `-i` | 프레임 추출 간격 (초) | 2 |
| `--model`, `-m` | 사용할 모델 (`base` 또는 `blip2`) | `base` |
| `--output`, `-o` | 결과 저장 파일 경로 | 없음 |
| `--question`, `-q` | 영상에 대한 질문 | (ask 명령 필수) |
| `--threshold`, `-t` | 장면 전환 감지 임계값 (0~1) | 0.3 |
| `--sample-interval`, `-s` | 장면 감지 샘플링 간격 (초) | 0.5 |
| `--videos` | 일괄 분석할 영상 파일 경로들 | (batch 명령 필수) |
| `--output-dir` | 일괄 분석 결과 저장 디렉토리 | 없음 |

### 웹 UI

```bash
python main.py web
```

브라우저에서 `http://localhost:7860` 접속 후:
1. 영상 파일 업로드
2. 프레임 간격 및 모델 선택
3. **영상 분석 시작** 클릭 → 요약 및 프레임별 분석 결과 확인
4. **장면 전환 감지** 탭에서 장면 변화 시점 감지
5. **질의응답** 탭에서 영상에 대해 자유롭게 질문
6. **영상 정보** 탭에서 해상도, FPS, 재생 시간 등 확인

### Python API

```python
from analyzer.video_analyzer import analyze_video
from analyzer.qa import ask_about_video

# 영상 분석
result = analyze_video("my_video.mp4", interval_sec=2)
print(result["summary"])

# 질의응답
answer = ask_about_video("my_video.mp4", question="What is happening?")
print(answer)
```

## 🤖 지원 모델

### 1. BLIP Base (기본값, 권장)
- **모델 ID**: `Salesforce/blip-image-captioning-base`
- **특징**: 가볍고 빠름, CPU에서도 적절한 속도
- **용도**: 이미지 캡셔닝 (장면 설명)
- **메모리**: 약 1~2GB RAM

### 2. BLIP-2 (고성능)
- **모델 ID**: `Salesforce/blip2-opt-2.7b`
- **특징**: 높은 정확도, GPU 권장
- **용도**: 이미지 캡셔닝 + 고급 시각적 이해
- **메모리**: 약 8~16GB RAM

모든 모델은 Hugging Face에서 **무료로 자동 다운로드**됩니다.

## 📁 프로젝트 구조

```
aivid/
├── README.md                # 프로젝트 설명 (한국어)
├── requirements.txt         # 의존성 패키지
├── setup.py                 # 패키지 설정
├── config.py                # 설정 파일 (모델 선택, 프레임 간격 등)
├── main.py                  # CLI 진입점
├── analyzer/
│   ├── __init__.py
│   ├── video_loader.py      # 영상 로드 및 프레임 추출 (OpenCV)
│   ├── frame_analyzer.py    # 프레임 분석 (BLIP 모델 사용)
│   ├── video_analyzer.py    # 전체 영상 분석 통합
│   ├── qa.py                # 영상 내용 질의응답 (VQA)
│   ├── scene_detector.py    # 장면 전환 감지
│   └── report.py            # HTML 리포트 생성
├── ui/
│   ├── __init__.py
│   └── web_app.py           # Gradio 웹 UI
└── tests/
    ├── __init__.py
    └── test_analyzer.py     # 단위 테스트
```

## 🧪 테스트 실행

```bash
python -m pytest tests/ -v
# 또는
python -m unittest discover tests/
```

## 📝 출력 예시

### 영상 분석 결과
```
[영상 분석 시작] sample.mp4
  설정: 2초 간격, 모델: base

[1단계] 프레임 추출 중...
  영상 정보: 30.0초, 30.0fps, 총 900프레임
  총 15개의 프레임을 추출했습니다.

[2단계] 프레임 분석 중 (총 15개)...
  [1/15] 0.0초 프레임 분석 중... 완료 - a person walking in a park...
  [2/15] 2.0초 프레임 분석 중... 완료 - a dog running on the grass...

[전체 요약]
총 15개의 프레임을 분석했습니다. | 주요 등장 요소: person, dog, park, grass, tree
```

## 📄 라이선스

MIT License

---

> **참고**: BLIP/BLIP-2 모델은 Salesforce Research에서 개발한 무료 오픈소스 모델입니다.
> 모든 모델은 Hugging Face Hub에서 자동으로 다운로드됩니다.
