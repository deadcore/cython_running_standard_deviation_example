import numpy as np
from Cython.Build import cythonize
from Cython.Distutils import build_ext
from setuptools import setup
from setuptools.extension import Extension


def create_moment_extension(name):
    return Extension(
        f'cython_example.native.{name}',
        sources=[f'cython_example/native/{name}.pyx'],
        include_dirs=[np.get_include()],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_compile_args=["-Wno-unused-function", "-O3", "-march=native"],
        extra_link_args=["-O3", "-march=native"],
        language="c++"
    )


moments_extension = [
    create_moment_extension('standard_deviation_1'),
    create_moment_extension('standard_deviation_2'),
    create_moment_extension('standard_deviation_3'),
    create_moment_extension('standard_deviation_4'),
    create_moment_extension('standard_deviation_5'),
]

extensions = [
    *moments_extension
]

setup(
    cmdclass={'build_ext': build_ext},
    name="Cython Example",
    ext_modules=cythonize(extensions, annotate=True, build_dir='build', compiler_directives={'language_level': "3"})
)





