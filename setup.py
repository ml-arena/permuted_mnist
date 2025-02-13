from setuptools import setup, find_packages

setup(
    name="metamnist",
    version="0.8",
    packages=find_packages(),
    include_package_data=True,  # Important for including data files
    package_data={
        'metamnist': ['data/*.npy'],  # Include MNIST data files
    },
    install_requires=[
        "scipy>=1.10.0",
    ],
)