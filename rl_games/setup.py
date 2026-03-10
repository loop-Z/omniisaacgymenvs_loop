"""Setup script for rl_games"""

import sys
import os
import pathlib

from setuptools import setup, find_packages
# The directory containing this file
HERE = pathlib.Path(__file__).parent

# The text of the README file
README = (HERE / "README.md").read_text()
print(find_packages())

setup(name='rl-games',
      long_description=README,
      long_description_content_type="text/markdown",
      url="https://github.com/Denys88/rl_games",
      #packages=[package for package in find_packages() if package.startswith('rl_games')],
      packages = ['.','rl_games','docs'],
      package_data={'rl_games':['*','*/*','*/*/*'],'docs':['*','*/*','*/*/*'],},
      version='1.6.1',
      author='Denys Makoviichuk, Viktor Makoviichuk',
      author_email='trrrrr97@gmail.com, victor.makoviychuk@gmail.com',
      license="MIT",
      classifiers=[
            "License :: OSI Approved :: MIT License",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.7",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10"
      ],
      #packages=["rlg"],
      include_package_data=True,
      install_requires=[
            # this setup is only for pytorch
            # 
            'gym==0.23.1',
            'torch==2.0.1',
            'numpy==1.23.5',
            'ray==2.47.0',
            'tensorboard==2.19.0',
            'tensorboardX==2.6.4',
            'setproctitle==1.3.6',
            'psutil==5.9.8',
            'pyyaml==6.0.1',
            'opencv-python==4.11.0.86',
            'wandb==0.12.21'
            # Optional dependencies
            # 'ray>=1.1.0',
      ],
      )
