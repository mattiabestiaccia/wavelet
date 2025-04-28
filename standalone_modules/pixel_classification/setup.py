from setuptools import setup, find_packages

setup(
    name="pixel_classification",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=1.8.0",
        "torchvision>=0.9.0",
        "kymatio>=0.3.0",
        "numpy>=1.19.0",
        "matplotlib>=3.3.0",
        "pillow>=8.0.0",
        "tqdm>=4.50.0",
        "albumentations>=1.0.0",
        "scikit-learn>=0.24.0",
        "opencv-python>=4.5.0",
    ],
    author="Mattia Bruscia",
    author_email="bruscia95@gmail.com",
    description="Modulo autonomo per la classificazione pixel-wise con Wavelet Scattering Transform",
    keywords="wavelet, scattering, pixel, classification, deep learning",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "pixel-train=pixel_classification.train:main",
            "pixel-predict=pixel_classification.predict:main",
            "pixel-test=pixel_classification.test:main",
        ],
    },
)
