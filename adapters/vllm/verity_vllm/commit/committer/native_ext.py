"""The native collector's extension: ``native_collect.cpp`` and the ``.cu`` kernels beside it (`native_jit.SOURCES`), JIT-built once
per process.  The collector (`native_collect`) and the host committer's rows-sha512 weights registration (`weights_rows`) both load it
here, so neither imports the other."""
from __future__ import annotations

import os

from verity_vllm import settings
from verity_vllm.log import init_logger

logger = init_logger(__name__)

_EXT = None
HERE = os.path.dirname(os.path.abspath(__file__))


def ext(build_dir: str | None = None, verbose: bool = False):
    """JIT-build the C++ collector (pybind11 + cudart + .cu kernels) once per process.  `NATIVE_COLLECT_BUILD` (or `build_dir`) names the
    PARENT of the build directory; the source-set digest picks the child (`<parent>/<digest12>`, native_jit) -- torch's
    mtime test is never the arbiter of staleness (a ship that preserves source mtimes would otherwise adopt a pod-wide dir warmed by
    another tree's build).  Without a parent, torch's default cache dir is used, keyed the same way.  Concurrent jobs
    on one host serialise on the dir's build lock (`native_jit.jit_load`)."""
    global _EXT
    if _EXT is None:
        # Read here, not at import: `native_jit` takes the kernels' headers from the core `verity`, which only a build needs.
        from verity_vllm.commit.committer import native_jit

        _EXT, bd, digest = native_jit.jit_load("verity_native_collect", native_jit.source_paths(HERE),
                                               parent=build_dir or settings.NATIVE_COLLECT_BUILD or None,
                                               cflags=native_jit.CFLAGS, cuda_cflags=native_jit.CUDA_CFLAGS, verbose=verbose)
        missing = native_jit.missing_bindings(_EXT, os.path.join(HERE, "native_collect.cpp"))
        if missing:
            _EXT = None
            raise RuntimeError(f"native collector .so in {bd} does not expose {missing} bound by {HERE}/native_collect.cpp -- a stale build "
                               f"(source digest {digest[:12]}); remove the directory and rebuild")
        logger.info("[jit] native collector source digest %s -> %s", digest[:12], bd)
    return _EXT
