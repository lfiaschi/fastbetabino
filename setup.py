# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules = cythonize(["./fastbetabino/_betactr.pyx"],
#                              extra_compile_args=['-fopenmp'],
#                              extra_link_args=['-fopenmp']
#                             ),
#
# )

from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Distutils import build_ext

ext_module = Extension(
    "fastbetabino",
    ["fastbetabino.pyx"],
    extra_compile_args=['-fopenmp'],
    extra_link_args=['-fopenmp'],
)

setup(
    install_requires=['scipy'],
    name = 'fastbetabino',
    cmdclass = {'build_ext': build_ext},
    ext_modules = [ext_module],
    packages = find_packages()
)