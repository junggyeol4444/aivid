# 기본 테스트 모듈
# analyzer 모듈의 핵심 기능을 테스트합니다.
# 실제 모델 로드 없이 동작하도록 mock을 사용합니다.

import os
import sys
import json
import unittest
import cv2
from unittest.mock import MagicMock, patch
from PIL import Image

# 프로젝트 루트를 경로에 추가
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestVideoLoader(unittest.TestCase):
    """video_loader.py 테스트"""

    def test_load_video_file_not_found(self):
        """존재하지 않는 파일에 대해 FileNotFoundError를 발생시켜야 합니다."""
        from analyzer.video_loader import load_video

        with self.assertRaises(FileNotFoundError):
            load_video("/tmp/nonexistent_video.mp4")

    def test_load_video_unsupported_format(self):
        """지원하지 않는 형식에 대해 ValueError를 발생시켜야 합니다."""
        from analyzer.video_loader import load_video

        # 임시 파일 생성
        tmp_path = "/tmp/test_video.xyz"
        with open(tmp_path, "w") as f:
            f.write("dummy")

        try:
            with self.assertRaises(ValueError):
                load_video(tmp_path)
        finally:
            os.remove(tmp_path)

    @patch("analyzer.video_loader.cv2.VideoCapture")
    def test_extract_frames_returns_list(self, mock_cap_cls):
        """extract_frames가 프레임 리스트를 반환해야 합니다."""
        import numpy as np
        from analyzer.video_loader import extract_frames

        # 임시 영상 파일 생성 (실제 영상 내용 없이 경로만)
        tmp_path = "/tmp/test_video.mp4"
        with open(tmp_path, "w") as f:
            f.write("dummy")

        # VideoCapture mock 설정
        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_POS_MSEC: 0,
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90,
        }.get(prop, 0)

        # 3프레임 반환 후 종료
        dummy_frame = np.zeros((100, 100, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, dummy_frame),
            (True, dummy_frame),
            (True, dummy_frame),
            (False, None),
        ]
        mock_cap_cls.return_value = mock_cap

        try:
            frames = extract_frames(tmp_path, interval_sec=1)
            self.assertIsInstance(frames, list)
            self.assertGreater(len(frames), 0)
            # 각 프레임에 timestamp와 frame 키가 있어야 함
            for frame_info in frames:
                self.assertIn("timestamp", frame_info)
                self.assertIn("frame", frame_info)
                self.assertIsInstance(frame_info["frame"], Image.Image)
        finally:
            os.remove(tmp_path)

    @patch("analyzer.video_loader.cv2.VideoCapture")
    def test_get_video_metadata_returns_dict(self, mock_cap_cls):
        """get_video_metadata가 올바른 딕셔너리를 반환해야 합니다."""
        from analyzer.video_loader import get_video_metadata

        tmp_path = "/tmp/test_meta_video.mp4"
        with open(tmp_path, "w") as f:
            f.write("dummy content for size")

        mock_cap = MagicMock()
        mock_cap.isOpened.return_value = True
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 900,
            cv2.CAP_PROP_FRAME_WIDTH: 1920,
            cv2.CAP_PROP_FRAME_HEIGHT: 1080,
        }.get(prop, 0)
        mock_cap_cls.return_value = mock_cap

        try:
            metadata = get_video_metadata(tmp_path)
            self.assertIsInstance(metadata, dict)
            self.assertIn("해상도", metadata)
            self.assertIn("FPS", metadata)
            self.assertIn("재생 시간", metadata)
            self.assertIn("파일 크기", metadata)
            self.assertEqual(metadata["해상도"], "1920 x 1080")
            self.assertEqual(metadata["FPS"], "30.0")
        finally:
            os.remove(tmp_path)


class TestFrameAnalyzer(unittest.TestCase):
    """frame_analyzer.py 테스트"""

    @patch("analyzer.frame_analyzer._load_model")
    def test_analyze_frame_returns_string(self, mock_load):
        """analyze_frame이 문자열을 반환해야 합니다."""
        from analyzer.frame_analyzer import analyze_frame

        # 모델 mock 설정
        mock_processor = MagicMock()
        mock_model = MagicMock()
        mock_load.return_value = (mock_processor, mock_model)

        mock_processor.return_value = {"pixel_values": MagicMock()}
        mock_model.generate.return_value = [[1, 2, 3]]
        mock_processor.decode.return_value = "a person walking in a park"

        image = Image.new("RGB", (100, 100))
        result = analyze_frame(image)

        self.assertIsInstance(result, str)

    @patch("analyzer.frame_analyzer.analyze_frame")
    def test_detect_objects_returns_list(self, mock_analyze):
        """detect_objects가 리스트를 반환해야 합니다."""
        from analyzer.frame_analyzer import detect_objects

        mock_analyze.return_value = "a person walking with a dog in the park"

        image = Image.new("RGB", (100, 100))
        result = detect_objects(image)

        self.assertIsInstance(result, list)
        # 불용어가 포함되지 않아야 함
        stopwords = {"a", "an", "the", "is", "in", "with"}
        for obj in result:
            self.assertNotIn(obj, stopwords)


class TestVideoAnalyzer(unittest.TestCase):
    """video_analyzer.py 테스트"""

    def test_generate_summary_empty(self):
        """빈 프레임 리스트에 대해 적절한 메시지를 반환해야 합니다."""
        from analyzer.video_analyzer import _generate_summary

        result = _generate_summary([])
        self.assertIn("없습니다", result)

    def test_generate_summary_with_frames(self):
        """프레임 결과로부터 요약을 생성해야 합니다."""
        from analyzer.video_analyzer import _generate_summary

        frames = [
            {"timestamp": 0.0, "description": "a person sitting", "objects": ["person", "chair"]},
            {"timestamp": 2.0, "description": "a dog running", "objects": ["dog", "grass"]},
            {"timestamp": 4.0, "description": "a cat sleeping", "objects": ["cat", "sofa"]},
        ]
        result = _generate_summary(frames)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # 프레임 수 정보가 포함되어야 함
        self.assertIn("3", result)

    def test_save_results_creates_file(self):
        """save_results가 JSON 파일을 생성해야 합니다."""
        from analyzer.video_analyzer import save_results

        result = {
            "video_path": "/tmp/test.mp4",
            "total_frames_analyzed": 2,
            "frames": [
                {"timestamp": 0.0, "description": "test scene", "objects": ["person"]},
            ],
            "summary": "테스트 요약",
        }

        output_path = "/tmp/test_result.json"
        try:
            save_results(result, output_path)
            self.assertTrue(os.path.exists(output_path))

            with open(output_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)

            self.assertEqual(loaded["video_path"], result["video_path"])
            self.assertEqual(loaded["summary"], result["summary"])
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)


class TestSceneDetector(unittest.TestCase):
    """scene_detector.py 테스트"""

    def test_compute_frame_difference_identical(self):
        """동일한 프레임 간 차이는 0이어야 합니다."""
        import numpy as np
        from analyzer.scene_detector import compute_frame_difference

        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        diff = compute_frame_difference(frame, frame)
        self.assertAlmostEqual(diff, 0.0, places=2)

    def test_compute_frame_difference_different(self):
        """다른 프레임 간 차이는 0보다 커야 합니다."""
        import numpy as np
        from analyzer.scene_detector import compute_frame_difference

        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 255, dtype=np.uint8)
        diff = compute_frame_difference(frame_a, frame_b)
        self.assertGreater(diff, 0.0)
        self.assertLessEqual(diff, 1.0)

    @patch("analyzer.scene_detector.load_video")
    def test_detect_scene_changes_returns_list(self, mock_load):
        """detect_scene_changes가 장면 리스트를 반환해야 합니다."""
        import numpy as np
        from analyzer.scene_detector import detect_scene_changes

        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 30.0,
            cv2.CAP_PROP_FRAME_COUNT: 90,
        }.get(prop, 0)

        # 3프레임: 첫 두 프레임은 동일, 세 번째는 다름
        frame_same = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_diff = np.full((100, 100, 3), 255, dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, frame_same),   # 첫 번째 프레임 (시작)
            (True, frame_same),   # 두 번째 프레임 (변화 없음)
            (True, frame_diff),   # 세 번째 프레임 (장면 전환)
            (False, None),
        ]
        mock_cap.set = MagicMock()
        mock_load.return_value = mock_cap

        scenes = detect_scene_changes("/tmp/test.mp4", threshold=0.3, sample_interval=0.5)

        self.assertIsInstance(scenes, list)
        self.assertGreater(len(scenes), 0)
        # 첫 번째 장면은 항상 timestamp 0
        self.assertEqual(scenes[0]["timestamp"], 0.0)
        self.assertEqual(scenes[0]["scene_index"], 0)

    @patch("analyzer.scene_detector.load_video")
    def test_detect_scene_changes_detects_transition(self, mock_load):
        """큰 변화가 있을 때 장면 전환을 감지해야 합니다."""
        import numpy as np
        from analyzer.scene_detector import detect_scene_changes

        mock_cap = MagicMock()
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FPS: 10.0,
            cv2.CAP_PROP_FRAME_COUNT: 30,
        }.get(prop, 0)

        frame_a = np.zeros((100, 100, 3), dtype=np.uint8)
        frame_b = np.full((100, 100, 3), 255, dtype=np.uint8)
        mock_cap.read.side_effect = [
            (True, frame_a),   # 시작
            (True, frame_b),   # 변화
            (True, frame_b),   # 유지
            (False, None),
        ]
        mock_cap.set = MagicMock()
        mock_load.return_value = mock_cap

        scenes = detect_scene_changes("/tmp/test.mp4", threshold=0.1, sample_interval=0.5)

        # 최소 2개 장면이 감지되어야 함 (시작 + 전환)
        self.assertGreaterEqual(len(scenes), 2)


class TestReport(unittest.TestCase):
    """report.py 테스트"""

    def test_generate_html_report_creates_file(self):
        """HTML 리포트 파일이 생성되어야 합니다."""
        from analyzer.report import generate_html_report

        analysis_result = {
            "video_path": "/tmp/test.mp4",
            "total_frames_analyzed": 3,
            "frames": [
                {"timestamp": 0.0, "description": "a person sitting", "objects": ["person"]},
                {"timestamp": 2.0, "description": "a dog running", "objects": ["dog"]},
                {"timestamp": 4.0, "description": "a cat sleeping", "objects": ["cat"]},
            ],
            "summary": "테스트 요약입니다.",
        }

        output_path = "/tmp/test_report.html"
        try:
            result_path = generate_html_report(analysis_result, output_path=output_path)
            self.assertTrue(os.path.exists(output_path))
            self.assertEqual(result_path, output_path)

            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            self.assertIn("영상 분석 리포트", html_content)
            self.assertIn("테스트 요약입니다.", html_content)
            self.assertIn("a person sitting", html_content)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_generate_html_report_with_metadata(self):
        """메타데이터가 포함된 HTML 리포트가 올바르게 생성되어야 합니다."""
        from analyzer.report import generate_html_report

        analysis_result = {
            "video_path": "/tmp/test.mp4",
            "total_frames_analyzed": 1,
            "frames": [
                {"timestamp": 0.0, "description": "test", "objects": []},
            ],
            "summary": "test summary",
        }

        metadata = {
            "해상도": "1920 x 1080",
            "FPS": "30.0",
            "재생 시간": "30초",
        }

        output_path = "/tmp/test_report_meta.html"
        try:
            generate_html_report(
                analysis_result,
                video_metadata=metadata,
                output_path=output_path,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            self.assertIn("1920 x 1080", html_content)
            self.assertIn("영상 정보", html_content)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_generate_html_report_with_scene_changes(self):
        """장면 전환 정보가 포함된 HTML 리포트가 올바르게 생성되어야 합니다."""
        from analyzer.report import generate_html_report

        analysis_result = {
            "video_path": "/tmp/test.mp4",
            "total_frames_analyzed": 1,
            "frames": [
                {"timestamp": 0.0, "description": "test", "objects": []},
            ],
            "summary": "test summary",
        }

        scene_changes = [
            {"timestamp": 0.0, "difference": 0.0, "scene_index": 0},
            {"timestamp": 5.0, "difference": 0.45, "scene_index": 1},
        ]

        output_path = "/tmp/test_report_scenes.html"
        try:
            generate_html_report(
                analysis_result,
                scene_changes=scene_changes,
                output_path=output_path,
            )

            with open(output_path, "r", encoding="utf-8") as f:
                html_content = f.read()

            self.assertIn("장면 전환", html_content)
            self.assertIn("장면 1", html_content)
            self.assertIn("장면 2", html_content)
        finally:
            if os.path.exists(output_path):
                os.remove(output_path)

    def test_image_to_base64(self):
        """PIL Image가 base64 문자열로 변환되어야 합니다."""
        from analyzer.report import _image_to_base64

        image = Image.new("RGB", (100, 100), color=(255, 0, 0))
        result = _image_to_base64(image)

        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)


class TestConfig(unittest.TestCase):
    """config.py 테스트"""

    def test_default_model_exists_in_model_ids(self):
        """기본 모델이 MODEL_IDS에 존재해야 합니다."""
        from config import DEFAULT_MODEL, MODEL_IDS

        self.assertIn(DEFAULT_MODEL, MODEL_IDS)

    def test_device_is_valid(self):
        """DEVICE가 'cpu' 또는 'cuda'여야 합니다."""
        from config import DEVICE

        self.assertIn(DEVICE, ["cpu", "cuda"])

    def test_supported_formats_contains_common_extensions(self):
        """SUPPORTED_FORMATS에 주요 형식이 포함되어야 합니다."""
        from config import SUPPORTED_FORMATS

        for fmt in [".mp4", ".avi", ".mov", ".mkv"]:
            self.assertIn(fmt, SUPPORTED_FORMATS)

    def test_scene_change_config_values(self):
        """장면 전환 감지 설정값이 올바른 범위여야 합니다."""
        from config import SCENE_CHANGE_THRESHOLD, SCENE_SAMPLE_INTERVAL

        self.assertGreater(SCENE_CHANGE_THRESHOLD, 0)
        self.assertLess(SCENE_CHANGE_THRESHOLD, 1)
        self.assertGreater(SCENE_SAMPLE_INTERVAL, 0)


class TestWebApp(unittest.TestCase):
    """web_app.py 테스트"""

    def test_format_metadata(self):
        """영상 메타데이터가 올바르게 포맷되어야 합니다."""
        from ui.web_app import _format_metadata

        metadata = {"해상도": "1920 x 1080", "FPS": "30.0"}
        result = _format_metadata(metadata)
        self.assertIn("1920 x 1080", result)
        self.assertIn("30.0", result)

    def test_format_scene_changes(self):
        """장면 전환 결과가 올바르게 포맷되어야 합니다."""
        from ui.web_app import _format_scene_changes

        scenes = [
            {"timestamp": 0.0, "difference": 0.0, "scene_index": 0},
            {"timestamp": 5.0, "difference": 0.5, "scene_index": 1},
        ]
        result = _format_scene_changes(scenes)
        self.assertIn("2개", result)
        self.assertIn("장면 1", result)
        self.assertIn("장면 2", result)

    def test_format_scene_changes_empty(self):
        """빈 장면 전환 리스트에 대해 적절한 메시지를 반환해야 합니다."""
        from ui.web_app import _format_scene_changes

        result = _format_scene_changes([])
        self.assertIn("감지되지 않았습니다", result)


if __name__ == "__main__":
    unittest.main()
