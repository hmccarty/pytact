from setuptools import setup, find_packages
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "marker_util",
        ["pytact/tasks/flow/matching.cpp"],
    ),
]

setup(name='Pytact',
    version='0.1',
    description='Visuo-tactile sensor interface',
    author='Harrison McCarty',
    author_email='hmccarty@pm.me',
    packages=find_packages(),
    ext_modules=ext_modules)