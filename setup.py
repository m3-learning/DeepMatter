# python setup.py sdist bdist_wheel
#twine upload dist/*

from setuptools import setup, find_packages
import os
with open('src/DeepMatter/requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='DeepMatter',
    version='0.0.59',
    packages=find_packages(where="src"),
    url='https://github.com/m3-learning/DeepMatter.git',
    install_requires=requirements,
    license=' BSD-3-Clause',
    author='Joshua C. Agar, Shuyu Qin',
    author_email='jca318@lehigh.edu, shq219@lehigh.edu',
    description='Tool for machine learning in materials Science',
    classifiers = [
                  "Programming Language :: Python :: 3",
                  "License :: OSI Approved :: BSD License",
                  "Operating System :: OS Independent",
              ],
              package_dir = {"": "src"},
                            python_requires = ">=3.6",
)
