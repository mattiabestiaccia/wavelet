from setuptools import setup, find_packages

setup(
    name="wavelet_lib",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "kymatio",
        "matplotlib",
        "scikit-learn",
        "pillow",
        "tqdm",
    ],
    description="A modular framework for image classification using Wavelet Scattering Transform representations",
    author="brus",
    author_email="brus@example.com",
    url="https://github.com/mattiabestiaccia/wavelet",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)