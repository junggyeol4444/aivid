from setuptools import setup, find_packages

setup(
    name="aivid",
    version="0.1.0",
    description="AI 기반 영상 시각 분석 프로그램",
    author="aivid",
    python_requires=">=3.9",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.8.0",
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "transformers>=4.35.0",
        "Pillow>=9.0.0",
        "gradio>=4.0.0",
        "numpy>=1.24.0",
    ],
    entry_points={
        "console_scripts": [
            "aivid=main:main",
        ],
    },
)
