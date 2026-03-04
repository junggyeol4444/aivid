# Gradio 웹 UI 모듈
# 영상 업로드, 분석 결과 표시, 장면 전환 감지, 질의응답 인터페이스를 제공합니다.

import gradio as gr
from analyzer.video_analyzer import analyze_video
from analyzer.qa import ask_about_video
from analyzer.scene_detector import detect_scene_changes
from analyzer.video_loader import get_video_metadata
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


def _format_metadata(metadata: dict) -> str:
    """
    영상 메타데이터를 보기 좋은 텍스트로 포맷합니다.
    """
    lines = []
    for key, value in metadata.items():
        lines.append(f"📌 {key}: {value}")
    return "\n".join(lines)


def _format_scene_changes(scene_changes: list) -> str:
    """
    장면 전환 결과를 보기 좋은 텍스트로 포맷합니다.
    """
    if not scene_changes:
        return "장면 전환이 감지되지 않았습니다."

    lines = [f"총 {len(scene_changes)}개의 장면이 감지되었습니다.\n"]
    for sc in scene_changes:
        diff_str = f" (변화율: {sc['difference']:.2%})" if sc["difference"] > 0 else " (시작)"
        lines.append(f"🎬 장면 {sc['scene_index'] + 1}: {sc['timestamp']:.1f}초{diff_str}")
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


def run_scene_detection(video_file, threshold: float, sample_interval: float):
    """
    장면 전환 감지를 실행합니다.

    Args:
        video_file: 영상 파일 경로
        threshold: 장면 전환 임계값
        sample_interval: 샘플링 간격

    Returns:
        장면 전환 결과 텍스트
    """
    if video_file is None:
        return "영상 파일을 업로드해주세요."

    try:
        scenes = detect_scene_changes(
            video_file,
            threshold=threshold,
            sample_interval=sample_interval,
        )
        return _format_scene_changes(scenes)
    except Exception as e:
        return f"오류 발생: {str(e)}"


def run_video_info(video_file):
    """
    영상 메타데이터를 조회합니다.

    Args:
        video_file: 영상 파일 경로

    Returns:
        메타데이터 텍스트
    """
    if video_file is None:
        return "영상 파일을 업로드해주세요."

    try:
        metadata = get_video_metadata(video_file)
        return _format_metadata(metadata)
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

        with gr.Tab("🎬 장면 전환 감지"):
            gr.Markdown("영상에서 장면이 바뀌는 시점을 자동으로 감지합니다.")
            with gr.Row():
                threshold_slider = gr.Slider(
                    minimum=0.1,
                    maximum=0.8,
                    value=0.3,
                    step=0.05,
                    label="장면 전환 감지 임계값 (낮을수록 민감)",
                )
                sample_slider = gr.Slider(
                    minimum=0.1,
                    maximum=2.0,
                    value=0.5,
                    step=0.1,
                    label="샘플링 간격 (초)",
                )
            scene_btn = gr.Button("🎬 장면 전환 감지", variant="secondary")
            scene_output = gr.Textbox(label="장면 전환 결과", lines=10)

        with gr.Tab("💬 질의응답"):
            question_input = gr.Textbox(
                label="영상에 대한 질문을 입력하세요 (영어 권장)",
                placeholder="What is happening in the video?",
            )
            ask_btn = gr.Button("❓ 질문하기", variant="secondary")
            answer_output = gr.Textbox(label="답변", lines=5)

        with gr.Tab("ℹ️ 영상 정보"):
            info_btn = gr.Button("📋 영상 정보 조회", variant="secondary")
            info_output = gr.Textbox(label="영상 메타데이터", lines=8)

        # 이벤트 연결
        analyze_btn.click(
            fn=run_analysis,
            inputs=[video_input, interval_slider, model_dropdown],
            outputs=[summary_output, frames_output],
        )

        scene_btn.click(
            fn=run_scene_detection,
            inputs=[video_input, threshold_slider, sample_slider],
            outputs=[scene_output],
        )

        ask_btn.click(
            fn=run_qa,
            inputs=[video_input, question_input, interval_slider, model_dropdown],
            outputs=[answer_output],
        )

        info_btn.click(
            fn=run_video_info,
            inputs=[video_input],
            outputs=[info_output],
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
