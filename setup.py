from distutils.core import setup
from Cython.Build import cythonize
import numpy

setup(name='evoltree',
      version='0.1',
      description='Evolutionary Decision Trees Package',
      url='http://github.com/vgarciasc/evoltree',
      author='Vinicius Garcia',
      author_email='',
      license='MIT',
      packages=['evoltree'],
      zip_safe=False,
      ext_modules = cythonize("evoltree/tree_evaluation.pyx",
                              compiler_directives={'language_level': "3"},
                              annotate=True),
      include_dirs = [numpy.get_include()])