# setup.py
from setuptools import setup, find_packages

setup(
    name="trading-bot",
    version="1.0",
    packages=find_packages(),
    package_dir={'': 'src'},
)