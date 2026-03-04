#!/usr/bin/env python3
# CLI 진입점
# 터미널에서 영상 분석, 질의응답, 장면 전환 감지, 리포트 생성, 웹 UI 실행을 제공합니다.
#
# 사용법:
#   python main.py analyze --video my_video.mp4 --interval 3
#   python main.py ask --video my_video.mp4 --question "What is happening?"
#   python main.py scenes --video my_video.mp4
#   python main.py report --video my_video.mp4 --output report.html
#   python main.py info --video my_video.mp4
#   python main.py batch --videos v1.mp4 v2.mp4 --output-dir results/
#   python main.py web

import argparse
import sys
import os


def cmd_analyze(args):
    """영상 분석 커맨드를 실행합니다."""
    from analyzer.video_analyzer import analyze_video, save_results

    result = analyze_video(
        video_path=args.video,
        interval_sec=args.interval,
        model_key=args.model,
    )

    # 결과 출력
    print("\n" + "=" * 60)
    print("[전체 요약]")
    print(result["summary"])
    print("\n[프레임별 분석]")
    for frame in result["frames"]:
        print(f"  {frame['timestamp']:.1f}초: {frame['description']}")
        if frame["objects"]:
            print(f"    주요 객체: {', '.join(frame['objects'][:5])}")

    # JSON 저장 (선택)
    if args.output:
        save_results(result, args.output)


def cmd_ask(args):
    """질의응답 커맨드를 실행합니다."""
    from analyzer.qa import ask_about_video

    answer = ask_about_video(
        video_path=args.video,
        question=args.question,
        interval_sec=args.interval,
        model_key=args.model,
    )

    print("\n" + "=" * 60)
    print(f"[질문] {args.question}")
    print(f"[답변] {answer}")


def cmd_scenes(args):
    """장면 전환 감지 커맨드를 실행합니다."""
    from analyzer.scene_detector import detect_scene_changes

    scenes = detect_scene_changes(
        video_path=args.video,
        threshold=args.threshold,
        sample_interval=args.sample_interval,
    )

    print("\n" + "=" * 60)
    print(f"[장면 전환 감지 결과] 총 {len(scenes)}개 장면")
    for sc in scenes:
        diff_str = f" (변화율: {sc['difference']:.2%})" if sc["difference"] > 0 else " (시작)"
        print(f"  장면 {sc['scene_index'] + 1}: {sc['timestamp']:.1f}초{diff_str}")


def cmd_report(args):
    """HTML 리포트 생성 커맨드를 실행합니다."""
    from analyzer.video_analyzer import analyze_video
    from analyzer.scene_detector import detect_scene_changes
    from analyzer.video_loader import get_video_metadata
    from analyzer.report import generate_html_report

    # 영상 메타데이터
    print("\n[영상 정보 수집 중...]")
    metadata = get_video_metadata(args.video)

    # 장면 전환 감지
    print("\n[장면 전환 감지 중...]")
    scenes = detect_scene_changes(args.video, threshold=0.3)

    # 영상 분석
    result = analyze_video(
        video_path=args.video,
        interval_sec=args.interval,
        model_key=args.model,
    )

    # HTML 리포트 생성
    print("\n[리포트 생성 중...]")
    output_path = args.output or "report.html"
    generate_html_report(
        analysis_result=result,
        video_metadata=metadata,
        scene_changes=scenes,
        output_path=output_path,
    )

    print(f"\n리포트가 생성되었습니다: {output_path}")


def cmd_info(args):
    """영상 메타데이터 출력 커맨드를 실행합니다."""
    from analyzer.video_loader import get_video_metadata

    metadata = get_video_metadata(args.video)

    print("\n" + "=" * 60)
    print("[영상 정보]")
    for key, value in metadata.items():
        print(f"  {key}: {value}")


def cmd_batch(args):
    """여러 영상을 일괄 분석하는 커맨드를 실행합니다."""
    from analyzer.video_analyzer import analyze_video, save_results

    # 출력 디렉토리 생성
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)

    results = []
    for i, video_path in enumerate(args.videos, 1):
        print(f"\n{'=' * 60}")
        print(f"[{i}/{len(args.videos)}] {video_path}")
        print("=" * 60)

        try:
            result = analyze_video(
                video_path=video_path,
                interval_sec=args.interval,
                model_key=args.model,
            )

            # 결과 요약 출력
            print(f"\n  요약: {result['summary']}")
            results.append({"video": video_path, "status": "성공", "result": result})

            # JSON 저장
            if args.output_dir:
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                json_path = os.path.join(args.output_dir, f"{base_name}_result.json")
                save_results(result, json_path)

        except Exception as e:
            print(f"\n  오류: {str(e)}")
            results.append({"video": video_path, "status": "실패", "error": str(e)})

    # 전체 요약
    print(f"\n{'=' * 60}")
    print("[일괄 분석 완료]")
    success = sum(1 for r in results if r["status"] == "성공")
    print(f"  성공: {success}/{len(results)}")
    if success < len(results):
        print(f"  실패: {len(results) - success}/{len(results)}")


def cmd_web(args):
    """웹 UI를 실행합니다."""
    from ui.web_app import launch
    launch()


def main():
    """CLI 메인 함수."""
    parser = argparse.ArgumentParser(
        description="AI 영상 분석기 - 영상의 시각적 내용을 AI로 분석합니다.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
사용 예시:
  python main.py analyze --video my_video.mp4 --interval 3
  python main.py analyze --video my_video.mp4 --output result.json
  python main.py ask --video my_video.mp4 --question "What is the person doing?"
  python main.py scenes --video my_video.mp4 --threshold 0.3
  python main.py report --video my_video.mp4 --output report.html
  python main.py info --video my_video.mp4
  python main.py batch --videos v1.mp4 v2.mp4 --output-dir results/
  python main.py web
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="실행할 명령어")

    # 공통 인수
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--video", "-v", required=True, help="분석할 영상 파일 경로"
    )
    common.add_argument(
        "--interval", "-i", type=float, default=2.0,
        help="프레임 추출 간격(초), 기본값: 2"
    )
    common.add_argument(
        "--model", "-m", choices=["base", "blip2"], default="base",
        help="사용할 모델 (base: 빠름, blip2: 고성능), 기본값: base"
    )

    # analyze 서브커맨드
    analyze_parser = subparsers.add_parser(
        "analyze", parents=[common], help="영상을 분석하고 결과를 출력합니다."
    )
    analyze_parser.add_argument(
        "--output", "-o", help="결과를 저장할 JSON 파일 경로 (선택)"
    )
    analyze_parser.set_defaults(func=cmd_analyze)

    # ask 서브커맨드
    ask_parser = subparsers.add_parser(
        "ask", parents=[common], help="영상 내용에 대해 질문합니다."
    )
    ask_parser.add_argument(
        "--question", "-q", required=True, help="영상에 대한 질문"
    )
    ask_parser.set_defaults(func=cmd_ask)

    # scenes 서브커맨드
    scenes_parser = subparsers.add_parser(
        "scenes", help="영상에서 장면 전환을 감지합니다."
    )
    scenes_parser.add_argument(
        "--video", "-v", required=True, help="분석할 영상 파일 경로"
    )
    scenes_parser.add_argument(
        "--threshold", "-t", type=float, default=0.3,
        help="장면 전환 감지 임계값 (0~1), 기본값: 0.3"
    )
    scenes_parser.add_argument(
        "--sample-interval", "-s", type=float, default=0.5,
        help="프레임 샘플링 간격 (초), 기본값: 0.5"
    )
    scenes_parser.set_defaults(func=cmd_scenes)

    # report 서브커맨드
    report_parser = subparsers.add_parser(
        "report", parents=[common], help="영상 분석 HTML 리포트를 생성합니다."
    )
    report_parser.add_argument(
        "--output", "-o", default="report.html",
        help="HTML 리포트 저장 경로 (기본값: report.html)"
    )
    report_parser.set_defaults(func=cmd_report)

    # info 서브커맨드
    info_parser = subparsers.add_parser(
        "info", help="영상 파일의 메타데이터를 출력합니다."
    )
    info_parser.add_argument(
        "--video", "-v", required=True, help="영상 파일 경로"
    )
    info_parser.set_defaults(func=cmd_info)

    # batch 서브커맨드
    batch_parser = subparsers.add_parser(
        "batch", help="여러 영상을 일괄 분석합니다."
    )
    batch_parser.add_argument(
        "--videos", nargs="+", required=True, help="분석할 영상 파일 경로들"
    )
    batch_parser.add_argument(
        "--interval", "-i", type=float, default=2.0,
        help="프레임 추출 간격(초), 기본값: 2"
    )
    batch_parser.add_argument(
        "--model", "-m", choices=["base", "blip2"], default="base",
        help="사용할 모델 (base: 빠름, blip2: 고성능), 기본값: base"
    )
    batch_parser.add_argument(
        "--output-dir", "-o", help="결과를 저장할 디렉토리 경로 (선택)"
    )
    batch_parser.set_defaults(func=cmd_batch)

    # web 서브커맨드
    web_parser = subparsers.add_parser(
        "web", help="웹 UI를 실행합니다."
    )
    web_parser.set_defaults(func=cmd_web)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
