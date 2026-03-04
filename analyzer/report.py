# HTML 리포트 생성 모듈
# 영상 분석 결과를 시각적인 HTML 리포트로 출력합니다.

import os
import json
import base64
from io import BytesIO
from PIL import Image


def _image_to_base64(image: Image.Image, max_width: int = 320) -> str:
    """
    PIL Image를 base64 인코딩된 문자열로 변환합니다.

    Args:
        image: PIL.Image 객체
        max_width: 최대 가로 크기 (리사이즈)

    Returns:
        base64 인코딩된 JPEG 문자열
    """
    # 리사이즈
    if image.width > max_width:
        ratio = max_width / image.width
        new_size = (max_width, int(image.height * ratio))
        image = image.resize(new_size, Image.LANCZOS)

    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=80)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return encoded


def generate_html_report(
    analysis_result: dict,
    video_metadata: dict = None,
    scene_changes: list = None,
    output_path: str = "report.html",
) -> str:
    """
    분석 결과를 HTML 리포트로 생성합니다.

    Args:
        analysis_result: analyze_video()의 결과 딕셔너리
        video_metadata: get_video_metadata()의 결과 (선택)
        scene_changes: detect_scene_changes()의 결과 (선택)
        output_path: 저장할 HTML 파일 경로

    Returns:
        생성된 HTML 파일 경로
    """
    video_path = analysis_result.get("video_path", "알 수 없음")
    summary = analysis_result.get("summary", "")
    frames = analysis_result.get("frames", [])
    total_frames = analysis_result.get("total_frames_analyzed", 0)

    # HTML 생성
    html_parts = [
        "<!DOCTYPE html>",
        '<html lang="ko">',
        "<head>",
        '<meta charset="UTF-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1.0">',
        f"<title>영상 분석 리포트 - {os.path.basename(video_path)}</title>",
        "<style>",
        _get_report_css(),
        "</style>",
        "</head>",
        "<body>",
        '<div class="container">',
    ]

    # 헤더
    html_parts.append('<div class="header">')
    html_parts.append("<h1>🎬 영상 분석 리포트</h1>")
    html_parts.append(f"<p>파일: <strong>{os.path.basename(video_path)}</strong></p>")
    html_parts.append("</div>")

    # 영상 메타데이터
    if video_metadata:
        html_parts.append('<div class="section">')
        html_parts.append("<h2>📋 영상 정보</h2>")
        html_parts.append('<table class="metadata-table">')
        for key, value in video_metadata.items():
            html_parts.append(f"<tr><td><strong>{key}</strong></td><td>{value}</td></tr>")
        html_parts.append("</table>")
        html_parts.append("</div>")

    # 전체 요약
    html_parts.append('<div class="section">')
    html_parts.append("<h2>📊 전체 요약</h2>")
    html_parts.append(f'<div class="summary-box">{summary}</div>')
    html_parts.append(f"<p>분석된 프레임 수: <strong>{total_frames}</strong>개</p>")
    html_parts.append("</div>")

    # 장면 전환 정보
    if scene_changes and len(scene_changes) > 1:
        html_parts.append('<div class="section">')
        html_parts.append(f"<h2>🎬 장면 전환 ({len(scene_changes)}개 장면)</h2>")
        html_parts.append('<div class="scene-list">')
        for sc in scene_changes:
            html_parts.append('<div class="scene-item">')
            html_parts.append(
                f'<span class="scene-badge">장면 {sc["scene_index"] + 1}</span> '
                f'{sc["timestamp"]:.1f}초'
            )
            if sc["difference"] > 0:
                html_parts.append(f' <span class="diff-value">(변화율: {sc["difference"]:.2%})</span>')
            html_parts.append("</div>")
        html_parts.append("</div>")
        html_parts.append("</div>")

    # 프레임별 분석
    html_parts.append('<div class="section">')
    html_parts.append("<h2>🖼 프레임별 분석</h2>")

    for frame in frames:
        html_parts.append('<div class="frame-card">')
        html_parts.append(f'<div class="frame-time">⏱ {frame["timestamp"]:.1f}초</div>')
        html_parts.append(f'<div class="frame-desc">{frame["description"]}</div>')
        if frame.get("objects"):
            objs = ", ".join(frame["objects"][:5])
            html_parts.append(f'<div class="frame-objects">주요 객체: {objs}</div>')
        html_parts.append("</div>")

    html_parts.append("</div>")

    # 푸터
    html_parts.append('<div class="footer">')
    html_parts.append("<p>AI 영상 분석기 (aivid)로 생성됨</p>")
    html_parts.append("</div>")

    html_parts.append("</div>")
    html_parts.append("</body>")
    html_parts.append("</html>")

    html_content = "\n".join(html_parts)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"  HTML 리포트가 저장되었습니다: {output_path}")
    return output_path


def _get_report_css() -> str:
    """리포트 CSS 스타일을 반환합니다."""
    return """
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background-color: #f5f7fa;
            color: #333;
            line-height: 1.6;
        }
        .container {
            max-width: 900px;
            margin: 0 auto;
            padding: 20px;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 12px;
            margin-bottom: 24px;
            text-align: center;
        }
        .header h1 { font-size: 28px; margin-bottom: 8px; }
        .header p { opacity: 0.9; }
        .section {
            background: white;
            padding: 24px;
            border-radius: 12px;
            margin-bottom: 20px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        }
        .section h2 {
            font-size: 20px;
            margin-bottom: 16px;
            color: #2c3e50;
            border-bottom: 2px solid #eee;
            padding-bottom: 8px;
        }
        .summary-box {
            background: #f0f4ff;
            padding: 16px;
            border-radius: 8px;
            border-left: 4px solid #667eea;
            margin-bottom: 12px;
        }
        .metadata-table {
            width: 100%;
            border-collapse: collapse;
        }
        .metadata-table td {
            padding: 8px 12px;
            border-bottom: 1px solid #eee;
        }
        .metadata-table td:first-child { width: 40%; color: #555; }
        .scene-list { display: flex; flex-wrap: wrap; gap: 8px; }
        .scene-item {
            background: #e8f4fd;
            padding: 8px 16px;
            border-radius: 20px;
            font-size: 14px;
        }
        .scene-badge {
            background: #3498db;
            color: white;
            padding: 2px 8px;
            border-radius: 10px;
            font-size: 12px;
            font-weight: bold;
        }
        .diff-value { color: #888; font-size: 12px; }
        .frame-card {
            border: 1px solid #eee;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 12px;
            transition: box-shadow 0.2s;
        }
        .frame-card:hover { box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
        .frame-time {
            font-weight: bold;
            color: #667eea;
            margin-bottom: 4px;
        }
        .frame-desc { margin-bottom: 4px; }
        .frame-objects {
            font-size: 13px;
            color: #777;
            font-style: italic;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #aaa;
            font-size: 13px;
        }
    """
