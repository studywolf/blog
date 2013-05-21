from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
  name = 'SimpleRNG',
  ext_modules=[ 
    Extension("SimpleRNG", 
              sources=["pySimpleRNG.pyx", "SimpleRNG.cpp"], # Note, you can link against a c++ library instead of including the source
              language="c++"),
    ],
  cmdclass = {'build_ext': build_ext},

)
