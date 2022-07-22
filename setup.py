from glob import glob
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension

ext_modules = [
    Pybind11Extension(
        "pytact_pybind",
        sorted(glob("pytact_pybind/*.cpp")),
    ),
]

setup(name='Pytact',
    version='0.1',
    description='Visuo-tactile sensor interface',
    author='Harrison McCarty',
    author_email='hmccarty@pm.me',
    packages=['pytact'],
    ext_modules=ext_modules)