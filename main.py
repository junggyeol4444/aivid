#!/usr/bin/env python3
# CLI 진입점
# 터미널에서 영상 분석, 질의응답, 웹 UI 실행을 제공합니다.
#
# 사용법:
#   python main.py analyze --video my_video.mp4 --interval 3
#   python main.py ask --video my_video.mp4 --question "What is happening?"
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
