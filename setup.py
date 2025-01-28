from setuptools import setup, find_packages

setup(
    name="metamnist",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29.1",
        "numpy>=1.24.0",
        "mnist>=0.2.2",  # Lightweight MNIST data loader
        "scipy>=1.10.0",  # For image transformations
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="A meta-learning environment for MNIST using Gymnasium",
    python_requires=">=3.8",
)