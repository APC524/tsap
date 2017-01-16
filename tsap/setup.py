#!/usr/bin/env python

from distutils.core import setup
from distutils.extension import Extension

from Cython.Build import cythonize

import numpy
sourcefiles = ['ts_gen.pyx', 'c_ts_gen.c']
extensions = [Extension("ts_gen", sourcefiles, include_dirs = [numpy.get_include()] ) ]

setup(
    name = "ts_gen",
    ext_modules = cythonize(extensions )
)
