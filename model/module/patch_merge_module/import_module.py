import sys
import os
import importlib.util

current_dir = os.path.dirname(os.path.abspath(__file__))

py_ver = f"{sys.version_info.major}{sys.version_info.minor}"

assert py_ver in ["37", "38", "39", "310", "312"], "Python version must be 3.7, 3.8, 3.9, 3.10, or 3.12"

print('trying to import PatchMergeModule')
if py_ver == "37":
    from .py37_module.PatchMergeModule import PatchMergeModule
elif py_ver == "38":
    from .py38_module.PatchMergeModule import PatchMergeModule
elif py_ver == "39":
    from .py39_module.PatchMergeModule import PatchMergeModule
elif py_ver == "310":
    from .py310_module.PatchMergeModule import PatchMergeModule
elif py_ver == "312":
    from .py312_module.PatchMergeModule import PatchMergeModule
    
else:
    raise NotImplementedError(f"Python version {py_ver} is not supported.")