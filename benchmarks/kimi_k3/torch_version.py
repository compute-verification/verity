"""`python torch_version.py`: the running torch, its CUDA and Triton, on one line; `--site`: the site-packages directory."""
import sys
import sysconfig

if sys.argv[1:] == ["--site"]:
    print(sysconfig.get_paths()["purelib"])
    sys.exit(0)
import torch

try:
    import triton

    tv = triton.__version__
except ImportError:
    tv = "absent"
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}), triton {tv}, python {sys.version.split()[0]}")
