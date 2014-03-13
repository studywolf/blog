from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
  
setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules=[Extension("py3LinkArm", 
               sources=["py3LinkArm.pyx"],
               language="c++"),],
)

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules=[Extension("py3LinkArm_damping", 
               sources=["py3LinkArm_damping.pyx"],
               language="c++"),],
)

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules=[Extension("py3LinkArm_gravity", 
               sources=["py3LinkArm_gravity.pyx"],
               language="c++"),],
)

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules=[Extension("py3LinkArm_gravity_damping", 
               sources=["py3LinkArm_gravity_damping.pyx"],
               language="c++"),],
)

setup(
  cmdclass = {'build_ext': build_ext},
  ext_modules=[Extension("py3LinkArm_smallmass", 
               sources=["py3LinkArm_smallmass.pyx"],
               language="c++"),],
)

