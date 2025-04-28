from setuptools import setup, find_packages

setup(
    name="tile_classification",
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
    ],
    author="Mattia Bruscia",
    author_email="bruscia95@gmail.com",
    description="Modulo autonomo per la classificazione di immagini con Wavelet Scattering Transform",
    keywords="wavelet, scattering, classification, deep learning",
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "tile-train=tile_classification.train:main",
            "tile-predict=tile_classification.predict:main",
        ],
    },
)
