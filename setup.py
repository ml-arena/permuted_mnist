from setuptools import setup, find_packages

setup(
    name="permuted_mnist",
    version="0.13",
    packages=find_packages(),
    include_package_data=True,  # Important for including data files
    package_data={
        'permuted_mnist': ['data/*.npy'],  # Include MNIST data files
    },
    install_requires=[
        "scipy>=1.10.0",
    ],
)