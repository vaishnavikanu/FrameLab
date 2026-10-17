"""Optional native build:  python setup.py build_ext --inplace
Produces framelab/_cpp*.so (or .pyd). FrameLab works without it (pure-Python fallbacks)."""
from setuptools import setup

try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
    ext_modules = [Pybind11Extension("framelab._cpp", ["cpp/image_ops.cpp"], cxx_std=17)]
    cmdclass = {"build_ext": build_ext}
except ImportError:  # pybind11 missing: install it with `pip install pybind11`
    ext_modules, cmdclass = [], {}

setup(name="framelab", version="2.0.0", packages=["framelab"],
      ext_modules=ext_modules, cmdclass=cmdclass)
