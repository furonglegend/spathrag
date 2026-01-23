# setup.py
from setuptools import setup, find_packages
from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="s_path_rag",
    version="0.1.0",
    description="S-Path-RAG: semantic-aware shortest-path RAG prototype for KGQA",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    license="MIT",
    packages=find_packages(where="src") or find_packages(),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.13.0",
        "transformers>=4.30.0",
        "networkx>=2.8.0",
        "numpy>=1.24.0",
        "PyYAML>=6.0",
        "tqdm>=4.64.0",
    ],
    extras_require={
        "dev": ["pytest>=7.0.0", "matplotlib>=3.6.0", "scikit-learn>=1.2.0"],
        "dgl": ["dgl"],  # placeholder
    },
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "s-path-rag=src.main:main",
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
