from setuptools import setup, find_packages

setup(
    name="art_style_explorer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "matplotlib",
        "opencv-python",
        "pillow",
        "scikit-image",
        "scikit-learn",
        "torch",
        "torchvision",
        "pandas",
        "tqdm",
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="Application that identifies artists and artworks with similar art styles",
    keywords="art, style, computer vision, neural networks",
    python_requires=">=3.8",
) 