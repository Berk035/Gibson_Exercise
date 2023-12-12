from setuptools import setup, find_packages
import sys

if sys.version_info.major != 3:
    print("This Python is only compatible with Python 3, but you are running "
          "Python {}. The installation will likely fail.".format(sys.version_info.major))


setup(name='tensorflowlib',
      packages=[package for package in find_packages()
                if package.startswith('tensorflowlib')],
      install_requires=[
          'tensorflow',
      ],
      description="This local packages are related with ODE functions",
      author="tf",
      url='https://github.com/tensorflow/tensorflow/tree/r2.0',
      version="0.0.1")