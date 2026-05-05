from setuptools import setup, find_packages

setup(
    name="neuralhive",
    version="0.1.0",
    description="Run massive AI models on weak hardware. Free. Forever. Offline.",
    author="NeuralHive Community",
    packages=find_packages(),
    install_requires=[
        "psutil>=5.9.0",
        "requests>=2.31.0",
        "rich>=13.0.0",
        "click>=8.1.0",
        "llama-cpp-python>=0.2.0",
        "huggingface-hub>=0.20.0",
        "numpy>=1.24.0",
        "tqdm>=4.65.0",
    ],
    entry_points={
        "console_scripts": [
            "neuralhive=cli.main:main",
            "nh=cli.main:nh_command",
        ],
    },
    python_requires=">=3.9",
)