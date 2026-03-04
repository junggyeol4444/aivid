# Gradio 웹 UI 모듈
# 영상 업로드, 분석 결과 표시, 질의응답 인터페이스를 제공합니다.

import gradio as gr
from analyzer.video_analyzer import analyze_video
from analyzer.qa import ask_about_video
from config import DEFAULT_MODEL, WEB_PORT, TOP_OBJECTS_COUNT


def _format_frame_results(frames: list) -> str:
    """
    프레임 분석 결과를 보기 좋은 텍스트로 포맷합니다.
    """
    lines = []
    for f in frames:
        ts = f["timestamp"]
        desc = f["description"]
        objs = ", ".join(f["objects"][:TOP_OBJECTS_COUNT]) if f["objects"] else "없음"
        lines.append(f"⏱ {ts:.1f}초 | 장면: {desc} | 주요 객체: {objs}")
    return "\n".join(lines)


def run_analysis(video_file, interval_sec: float, model_key: str):
    """
    영상 분석을 실행하고 결과를 반환합니다.

    Args:
        video_file: Gradio에서 업로드된 영상 파일 경로
        interval_sec: 프레임 추출 간격 (초)
        model_key: 사용할 모델 키

    Returns:
        (요약 텍스트, 프레임별 분석 텍스트)
    """
    if video_file is None:
        return "영상 파일을 업로드해주세요.", ""

    try:
        result = analyze_video(video_file, interval_sec=interval_sec, model_key=model_key)
        summary = result["summary"]
        frame_text = _format_frame_results(result["frames"])
        return summary, frame_text
    except Exception as e:
        return f"오류 발생: {str(e)}", ""


def run_qa(video_file, question: str, interval_sec: float, model_key: str):
    """
    영상에 대한 질의응답을 실행합니다.

    Args:
        video_file: 영상 파일 경로
        question: 사용자 질문
        interval_sec: 프레임 추출 간격 (초)
        model_key: 사용할 모델 키

    Returns:
        답변 텍스트
    """
    if video_file is None:
        return "영상 파일을 업로드해주세요."
    if not question.strip():
        return "질문을 입력해주세요."

    try:
        answer = ask_about_video(
            video_file,
            question=question,
            interval_sec=interval_sec,
            model_key=model_key,
        )
        return answer
    except Exception as e:
        return f"오류 발생: {str(e)}"


def create_app() -> gr.Blocks:
    """
    Gradio 웹 앱을 생성하고 반환합니다.
    """
    with gr.Blocks(title="AI 영상 분석기") as app:
        gr.Markdown(
            """
            # 🎬 AI 영상 분석기
            AI가 영상의 시각적 내용을 분석합니다. 영상을 업로드하고 분석을 시작하세요!
            """
        )

        with gr.Row():
            with gr.Column():
                video_input = gr.Video(label="영상 업로드")
                interval_slider = gr.Slider(
                    minimum=1,
                    maximum=10,
                    value=2,
                    step=1,
                    label="프레임 추출 간격 (초)",
                )
                model_dropdown = gr.Dropdown(
                    choices=["base", "blip2"],
                    value=DEFAULT_MODEL,
                    label="모델 선택 (base: 빠름, blip2: 고성능)",
                )
                analyze_btn = gr.Button("🔍 영상 분석 시작", variant="primary")

        with gr.Tab("📊 분석 결과"):
            summary_output = gr.Textbox(label="전체 요약", lines=4)
            frames_output = gr.Textbox(label="프레임별 분석", lines=15)

        with gr.Tab("💬 질의응답"):
            question_input = gr.Textbox(
                label="영상에 대한 질문을 입력하세요 (영어 권장)",
                placeholder="What is happening in the video?",
            )
            ask_btn = gr.Button("❓ 질문하기", variant="secondary")
            answer_output = gr.Textbox(label="답변", lines=5)

        # 이벤트 연결
        analyze_btn.click(
            fn=run_analysis,
            inputs=[video_input, interval_slider, model_dropdown],
            outputs=[summary_output, frames_output],
        )

        ask_btn.click(
            fn=run_qa,
            inputs=[video_input, question_input, interval_slider, model_dropdown],
            outputs=[answer_output],
        )

    return app


def launch():
    """
    웹 앱을 시작합니다.
    """
    app = create_app()
    print(f"\n웹 UI를 시작합니다. 브라우저에서 http://localhost:{WEB_PORT} 을 열어주세요.")
    app.launch(server_port=WEB_PORT, share=False)


if __name__ == "__main__":
    launch()
