from setuptools import find_packages, setup

setup(
    name="faceforensics-plus-plus",
    version="0.1.0",
    description="FaceForensics++ reproduction toolkit for deepfake detection",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "timm>=0.9.0",
        "opencv-python>=4.8.0",
        "facenet-pytorch>=2.5.3",
        "numpy>=1.24.0",
        "PyYAML>=6.0.0",
        "tqdm>=4.66.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "pandas>=2.0.0",
        "Pillow>=9.5.0",
    ],
    python_requires=">=3.10",
)
