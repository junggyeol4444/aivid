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


if __name__ == "__main__":
    unittest.main()
