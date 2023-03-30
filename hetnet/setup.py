# -*- coding: utf-8 -*-
"""
Setup python file for Cython.
Last update: 2018/11/18
"""

from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

src_fls = ['calc_capacity_rapper.pyx', 'calc_capacity_rap.cpp']

ext = Extension('calc_capacity_rapper', sources=src_fls, language="c++")

setup(
    name='calc_capacity_rapper',
    cmdclass={'build_ext': build_ext},
    ext_modules=cythonize([ext]),
    include_dirs=[numpy.get_include()]
)
