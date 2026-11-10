"""hash_prefill.cuh's launches on the host (pearl_c_sm90/hash_prefill_host.cpp: g++ builds them beside hash_decode_host.cpp's
hash_h3.cuh and hash_decode.cuh kernels), as pearl_c_sm90/run.py's `Pipeline` launches them past decode under its chain "pl":
A's level 0 (`hash_rows_a`, pl_leaves), its levels (`tree_a`, pl_levels), seed_A with line_a (`seed_a`, pl_seed_a) and each
real row's seed and line key (`seeds_a`, pl_seeds).  At real row counts on both sides of 256 (ranks and indices of one byte
and of two), with partial CTAs, a tree of one leaf and trees past one pl_levels launch's PL_LEVELS, every output is poisoned
(0xA5) before each chain ("pl", and "h3": hash_h3.cuh's launches), and each chain's bytes are held to Pearl-C -h3's Python
(`pearl_c.row_digest`, A's `audit.HM96` tree under the unit's index, `pearl_c.row_seed` with `pearl_c.line_key`,
`pearl_c.unit_seed_a`), so to each other; the no-write control, the poison with no launch, fails every comparison."""

from __future__ import annotations

import ctypes
import hashlib
import importlib.util
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from verity.core.protocols.pouw import audit
from verity.core.protocols.pouw.audit import ROW_SCHEMA, activation_binding, rows_domain
from verity.ml.protocols.pouw.schemes import pearl_c as C

HERE = Path(__file__).resolve().parent
KERNEL = HERE.parents[0] / "pearl_c_sm90"


def load(name: str, path: Path):
    """The module at `path` under its own name: other trees in the suite have a run.py too."""
    if name not in sys.modules:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = sys.modules[name] = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    return sys.modules[name]


R = load("pearl_c_sm90_run", KERNEL / "run.py")
#: the host `Dev` (`Host`) and A's tree in Python (`levels`): hash_decode.cuh's host test's
HD = load("pearl_c_hash_decode_test", HERE / "test_pearl_c_hash_decode.py")
SOURCE = (KERNEL / "pearl_c_sm90.cu").read_text()
#: pearl_c_sm90.cu's PD_VARIANT_LIST, which hash_decode_host.cpp builds beside hash_prefill.cuh's PL_INSTANCE(851)
PD_LIST = re.search(r"#define PD_VARIANT_LIST\(X\) (.*)", SOURCE).group(1).strip()
PL = {f"pl_{kind}_{R.PL_VARIANT}" for kind in ("leaves", "levels", "seed_a", "seeds")}
H3 = {"hm_leaves", "s512_frame_leaves", "s512_nodes", "s512_tree_small", "s512_seeds"}


@pytest.fixture(scope="module")
def host(tmp_path_factory):
    gxx = shutil.which("g++")
    if not gxx:
        pytest.skip("no g++ for the host build of hash_prefill.cuh")
    lib = tmp_path_factory.mktemp("hash-prefill-host") / "hash_prefill_host.so"
    subprocess.run([gxx, "-O2", "-std=c++20", "-pthread", "-shared", "-fPIC", "-Wno-unknown-pragmas", f"-DPD_VARIANT_LIST(X)={PD_LIST}",
                    "-o", str(lib), str(KERNEL / "hash_prefill_host.cpp")], check=True)
    cdll = ctypes.CDLL(str(lib))
    cdll.pd_run.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
    return HD.Host(cdll)


def test_the_cubin_builds_the_variant_run_py_launches():
    assert re.findall(r"^PL_INSTANCE\((\d+)\)", SOURCE, re.M) == [str(R.PL_VARIANT)] and str(R.PL_VARIANT) in PD_LIST
    text = (KERNEL / "hash_prefill.cuh").read_text()
    for name, value in (("PL_ROWS", R.PL_ROWS), ("PL_LEVELS", R.PL_LEVELS)):
        assert re.search(rf"\b{name} = (\d+)", text).group(1) == str(value)
    assert "PL_MAX_ROWS = 1 << 16" in text and R.PL_MAX_ROWS == 1 << 16 and "pl_seeds_t" in text and R.PL_SEEDS == 32
    host_src = (KERNEL / "hash_prefill_host.cpp").read_text()
    assert f"#define PD_HOST_EXTRA_INSTANCES PL_INSTANCE({R.PL_VARIANT})\n" in host_src
    assert f"#define PD_HOST_EXTRA_KERNELS PL_KERNELS({R.PL_VARIANT})\n" in host_src


@pytest.mark.parametrize("mr", [1, 40, 128, 300, 600, 2100])
def test_both_chains_write_pearl_c_h3s_bytes_and_the_no_write_control_fails(host, mr):
    k, n, idx = 256, 64, 9
    m = R.cdiv(mr, R.BM_D) * R.BM_D
    rng = np.random.default_rng(mr)
    a_rows = [rng.integers(0, 256, 4 * k, dtype=np.uint8).tobytes() for _ in range(mr)]
    salt, salt_key, root_b = bytes(range(32)), bytes(range(100, 132)), hashlib.sha512(b"root_b").digest()
    dom_a = rows_domain(activation_binding(salt, idx), idx, mr, audit.HM96)
    keys = {"salt_key": salt_key, "hm": audit.hm96_kernel_constants(), "unit": salt + root_b + idx.to_bytes(8, "little"),
            "dom_a": dom_a.domain_id}
    sizes = R.Pipeline.sizes(m, n, k, mr)
    bufs = {x: host.alloc(sizes[x]) for x in (*R.CHAIN_OUTPUTS, "segdig_a")}
    keys = {name: host.put(v) for name, v in keys.items()}
    xa = host.put(b"".join(a_rows) + bytes(4 * k * (m - mr)))

    digests = [C.row_digest(r) for r in a_rows]
    tree_a = HD.levels(dom_a, a_rows, ROW_SCHEMA, salt_key, idx)
    seeds = [C.row_seed(salt, d, root_b, idx, r, "h3") for r, d in enumerate(digests)]
    seed_a = C.unit_seed_a(salt, tree_a[-64:], root_b, idx, "h3")
    want = {"rowdig_a": b"".join(digests), "tree_a": tree_a, "seed_a": bytes(seed_a), "line_a": C.line_key(seed_a),
            "seeds_a": b"".join(seeds), "lkeys_a": b"".join(C.line_key(s) for s in seeds)}
    assert set(want) == set(R.CHAIN_OUTPUTS) and all(len(want[x]) == sizes[x] for x in want)

    depth = R.depth(mr)
    for chain in (None, *R.CHAINS):
        for x in want:
            host.array(bufs[x])[:] = R.POISON
        host.log.clear()
        if chain is not None:
            pipe = R.Pipeline(host, m, n, k, "h100", keys, xa, 0, bufs=bufs, m_real=mr, chain=chain)
            assert pipe.pl == (chain == "pl") and pipe.pd is None
            assert [nm for nm, _ in pipe.commit_steps()] == ["rowdig_a", "hash_rows_a", "tree_a", "seed_a", "seeds_a"]
            host.log.clear()
            R.run_commit(pipe)
            pipe.free()
        got = {x: host.read(bufs[x], len(want[x])) == want[x] for x in want}
        if chain is None:
            assert got == dict.fromkeys(want, False), "the no-write control passed"
            continue
        assert got == dict.fromkeys(want, True), (chain, mr, got)
        names = [name for name, *_ in host.log]
        if chain == "h3":
            assert not PL & set(names) and {"hm_leaves", "s512_frame_leaves", "s512_seeds"} <= set(names)
            continue
        assert not H3 & set(names) and {nm for nm in PL if depth or "levels" not in nm} <= set(names)
        levels = [(grid, i) for name, grid, _, _, i in host.log if name.startswith("pl_levels_")]
        assert len(levels) == -(-depth // R.PL_LEVELS) and sum(i[2] for _, i in levels) == depth
        assert all(grid == i[0] >> i[2] for grid, i in levels)
