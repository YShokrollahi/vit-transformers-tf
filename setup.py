# setup.py

from setuptools import setup, find_packages

setup(
    name="vision_transformer",
    version="0.1.0",
    description="Vision Transformer implementation in TensorFlow",
    author="Yasin Shokrollahi",
    author_email="",
    packages=find_packages(),
    install_requires=[
        "tensorflow>=2.0.0",
        "numpy",
        "matplotlib",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
