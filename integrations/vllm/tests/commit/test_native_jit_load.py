"""`native_jit.jit_load`: the torch JIT extensions of the Commit, safe for several jobs on one host.  The build dir is keyed by the source
set, one builder at a time holds the dir's fcntl build lock (released by the kernel however its holder exits), and a torch `lock` left in
the dir by a killed build is removed under it.  Torch-free: a stand-in loader, FIFOs and process exits instead of clocks.

Every test that loads, forks, signals or locks runs in a fresh interpreter (`_isolated`): run in the pytest worker, they leave it in a
state that breaks later tests in the same worker."""
from __future__ import annotations

import errno
import fcntl
import functools
import multiprocessing
import os
import signal
import subprocess
import sys

import pytest

from verity_vllm.commit.committer import native_jit as nj

_RUN = ("import pathlib, sys, types\n"
        "path, name, tmp = sys.argv[1:4]\n"
        "cases = sys.modules['native_jit_cases'] = types.ModuleType('native_jit_cases')\n"
        "cases.__file__ = path\n"
        "exec(compile(open(path).read(), path, 'exec'), cases.__dict__)\n"
        "getattr(cases, name).__wrapped__(pathlib.Path(tmp))\n")


def _isolated(case):
    """The test, run as `case(tmp_path)` in a fresh interpreter of this module; its failure is reported with the child's output.  The
    child compiles this file from source, never from a cached `.pyc`, whose mtime check a same-size edit within a second passes."""
    @functools.wraps(case)
    def test(tmp_path):
        r = subprocess.run([sys.executable, "-c", _RUN, __file__, case.__name__, str(tmp_path)], capture_output=True, text=True)
        assert r.returncode == 0, f"{case.__name__} in a fresh interpreter:\n{r.stdout}{r.stderr}"
    return test

VERSIONS = {"torch_version": "2.8.0", "cuda_version": "12.9"}


def _sources(tmp_path, body: bytes = b"__global__ void k(){}") -> list[str]:
    p = tmp_path / "src" / "ext.cu"
    p.parent.mkdir(exist_ok=True)
    p.write_bytes(body)
    return [str(p)]


def _held(build_dir: str) -> bool:
    """Whether another process holds the build lock of `build_dir` (a non-blocking try, released at once when it succeeds)."""
    fd = os.open(build_dir + ".build-lock", os.O_RDWR | os.O_CREAT, 0o644)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError as e:
        assert e.errno in (errno.EWOULDBLOCK, errno.EAGAIN)
        return True
    finally:
        os.close(fd)
    return False


def _loader(**kw):
    return {"built_in": kw["build_directory"], "name": kw["name"], "cuda_cflags": kw["extra_cuda_cflags"]}


def _blocking_load(sources, parent, started, go):
    """A child's jit_load whose loader says it started (the `started` FIFO), then waits for the test (the `go` FIFO)."""
    def loader(**kw):
        with open(started, "w") as f:
            f.write(kw["build_directory"] + "\n")
        with open(go) as f:
            f.read()
        return None
    nj.jit_load("ext", sources, parent=parent, loader=loader, **VERSIONS)


def _hold_lock_until_killed(build_dir, started):
    with nj.build_lock(build_dir):
        with open(started, "w") as f:
            f.write("held\n")
        signal.pause()


@_isolated
def test_the_build_dir_is_the_source_set_s_child_of_the_parent(tmp_path):
    mod, bd, digest = nj.jit_load("ext", _sources(tmp_path), parent=str(tmp_path / "jit"), cuda_cflags=("-O3",), loader=_loader, **VERSIONS)
    assert bd == os.path.join(str(tmp_path / "jit"), digest[:12]) and os.path.isdir(bd)
    assert mod == {"built_in": bd, "name": "ext", "cuda_cflags": ["-O3"]}
    assert digest == nj.source_set_digest(_sources(tmp_path), cflags=(), cuda_cflags=("-O3",), **VERSIONS)
    other = nj.jit_load("ext", _sources(tmp_path, b"__global__ void k2(){}"), parent=str(tmp_path / "jit"), loader=_loader, **VERSIONS)[1]
    assert other != bd, "another source set never shares the dir"


@_isolated
def test_two_trees_with_one_source_set_compile_the_same_staged_files(tmp_path):
    parent, seen = str(tmp_path / "jit"), []

    def loader(**kw):
        seen.append([(p, os.stat(p).st_mtime_ns, open(p, "rb").read()) for p in kw["sources"]])

    for tree in ("tree-a", "tree-b", "tree-a"):
        (tmp_path / tree).mkdir(exist_ok=True)
        _, bd, _ = nj.jit_load("ext", _sources(tmp_path / tree), parent=parent, loader=loader, **VERSIONS)
    assert seen[0] == seen[1] == seen[2], "same paths, bytes and mtimes: build.ninja and ninja's view are the same for every tree"
    assert seen[0][0][0] == os.path.join(bd, "src", "ext.cu") and seen[0][0][2] == b"__global__ void k(){}"


@_isolated
def test_a_staged_source_that_differs_from_the_source_is_rewritten(tmp_path):
    src, parent = _sources(tmp_path), str(tmp_path / "jit")
    bd = nj.keyed_build_dir(parent, nj.source_set_digest(src, cflags=(), cuda_cflags=(), **VERSIONS))
    os.makedirs(os.path.join(bd, "src"))
    (tmp_path / "jit" / os.path.basename(bd) / "src" / "ext.cu").write_bytes(b"partial")
    nj.jit_load("ext", src, parent=parent, loader=_loader, **VERSIONS)
    assert open(os.path.join(bd, "src", "ext.cu"), "rb").read() == b"__global__ void k(){}"


@_isolated
def test_sources_with_one_basename_twice_are_refused(tmp_path):
    a, b = tmp_path / "a" / "ext.cu", tmp_path / "b" / "ext.cu"
    for p in (a, b):
        p.parent.mkdir()
        p.write_bytes(b"__global__ void k(){}")
    with pytest.raises(ValueError, match="basenames"):
        nj.jit_load("ext", [str(a), str(b)], parent=str(tmp_path / "jit"), loader=_loader, **VERSIONS)


@_isolated
def test_local_headers_are_staged_beside_the_sources_and_key_the_digest(tmp_path):
    src = tmp_path / "src"
    (src / "sub").mkdir(parents=True)
    (src / "ext.cu").write_bytes(b'#include <cuda.h>\n#include "sub/a.h"\n#include "absent.h"\n__global__ void k(){}')
    (src / "sub" / "a.h").write_bytes(b'#include "b.h"\n')
    (src / "sub" / "b.h").write_bytes(b"#define B 1\n")
    paths = [str(src / "ext.cu")]
    assert nj.local_headers(paths) == [("sub/a.h", str(src / "sub" / "a.h")), ("sub/b.h", str(src / "sub" / "b.h"))]
    seen = {}
    _, bd, digest = nj.jit_load("ext", paths, parent=str(tmp_path / "jit"), loader=lambda **kw: seen.update(kw), **VERSIONS)
    assert seen["sources"] == [os.path.join(bd, "src", "ext.cu")]
    assert open(os.path.join(bd, "src", "sub", "b.h"), "rb").read() == b"#define B 1\n"
    (src / "sub" / "b.h").write_bytes(b"#define B 2\n")
    assert nj.source_set_digest(paths, cflags=(), cuda_cflags=(), **VERSIONS) != digest, "a header byte moves the digest"


def test_a_header_free_source_set_keeps_its_digest(tmp_path):
    import hashlib

    paths = _sources(tmp_path)
    h = hashlib.sha256(b"ext.cu\0__global__ void k(){}\0cflags=|cuda_cflags=|torch=2.8.0|cuda=12.9")
    assert nj.source_set_digest(paths, cflags=(), cuda_cflags=(), **VERSIONS) == h.hexdigest()


def test_the_committer_stages_the_gpus_sha512_header_and_its_bytes_key_the_digest(tmp_path, monkeypatch):
    from verity.ml.kernels import SHA512

    paths = nj.source_paths()
    assert ("sha512.cuh", str(SHA512)) in nj.local_headers(paths)
    digest = nj.source_set_digest(paths, **VERSIONS)
    moved = tmp_path / "sha512.cuh"
    moved.write_bytes(SHA512.read_bytes() + b"\n")
    monkeypatch.setitem(nj.SHARED, "sha512.cuh", str(moved))
    assert nj.source_set_digest(paths, **VERSIONS) != digest, "a byte of the shared header moves the digest"


def test_a_header_outside_the_sources_directory_is_refused(tmp_path):
    (tmp_path / "src").mkdir()
    (tmp_path / "up.h").write_bytes(b"")
    (tmp_path / "src" / "ext.cu").write_bytes(b'#include "../up.h"\n')
    with pytest.raises(ValueError, match="outside"):
        nj.local_headers([str(tmp_path / "src" / "ext.cu")])


@_isolated
def test_a_torch_lock_left_by_a_killed_build_is_removed_under_the_build_lock(tmp_path):
    src, parent = _sources(tmp_path), str(tmp_path / "jit")
    bd = nj.keyed_build_dir(parent, nj.source_set_digest(src, cflags=(), cuda_cflags=(), **VERSIONS))
    os.makedirs(bd)
    (tmp_path / "jit" / os.path.basename(bd) / "lock").write_text("")
    seen = {}

    def loader(**kw):
        seen["lock_present"] = os.path.exists(os.path.join(kw["build_directory"], "lock"))
        seen["held"] = _held(kw["build_directory"])      # flock is per open file: a second descriptor sees the loader's hold
        return None

    nj.jit_load("ext", src, parent=parent, loader=loader, **VERSIONS)
    assert seen == {"lock_present": False, "held": True}


@_isolated
def test_a_second_job_cannot_build_while_the_first_holds_the_lock(tmp_path):
    src, parent = _sources(tmp_path), str(tmp_path / "jit")
    started, go = str(tmp_path / "started"), str(tmp_path / "go")
    os.mkfifo(started)
    os.mkfifo(go)
    ctx = multiprocessing.get_context("fork")
    p = ctx.Process(target=_blocking_load, args=(src, parent, started, go))
    p.start()
    try:
        with open(started) as f:
            bd = f.readline().strip()                  # the child is inside its loader, under the lock
        assert _held(bd), "a builder holds the dir's lock while it builds"
        with open(go, "w") as f:
            f.write("go\n")
        p.join()
    finally:
        if p.is_alive():
            p.kill()
            p.join()
    assert p.exitcode == 0 and not _held(bd)


@_isolated
def test_a_killed_holder_leaves_no_lock_behind(tmp_path):
    bd = str(tmp_path / "jit" / "abcdef012345")
    os.makedirs(bd)
    started = str(tmp_path / "started")
    os.mkfifo(started)
    p = multiprocessing.get_context("fork").Process(target=_hold_lock_until_killed, args=(bd, started))
    p.start()
    with open(started) as f:
        assert f.readline().strip() == "held"
    assert _held(bd)
    os.kill(p.pid, signal.SIGKILL)
    p.join()
    assert p.exitcode == -signal.SIGKILL and not _held(bd), "the kernel drops the lock with its holder"


@pytest.mark.parametrize("module", ["native_ext", "leafhash", "hidden_gpu"])
def test_every_torch_jit_extension_of_the_commit_loads_through_jit_load(module):
    here = os.path.dirname(os.path.abspath(nj.__file__))
    path = {"native_ext": "native_ext.py", "leafhash": "leafhash.py", "hidden_gpu": os.path.join("hidden_gpu_src", "hidden_gpu.py")}[module]
    src = open(os.path.join(here, path)).read()
    assert "native_jit.jit_load(" in src and "cpp_extension import load" not in src
