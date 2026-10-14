"""The DeepSeek-V4-Pro Marlin capture without a GPU: its jobs over launches made here (the kernel's words stood in for by the twin at
a known config), the kernel's config read off its profiled name and grid, the probe telling summation orders apart where ordinary
activations cannot, a flipped word found and located, the reference against the twin, the sweep and its model, the assignment and
statics checks, the records of a case's launches, the JIT'd kernels read back from a compute cache made here and the driver held
to the map's, the verdict, and `main` end to end (the GPU run is `tests/properties/dsv4_marlin_capture_gpu.py --out ...` on an
H200)."""
from __future__ import annotations

import copy
import hashlib
import struct

import numpy as np
import pytest
import torch

from tests.properties import dsv4_marlin_capture_gpu as X
from verity_vllm.program.kernels import marlin_moe_rows as MR
from verity_vllm.properties.record import REGISTRY, digest_of

E, K, N, SMS = 8, 256, 128, 3
#: the config the stand-in kernel runs: TK 64 (two warp rows), TN 128, 4 blocks per SM, so GRID 12 splits K
TRUE = (64, 128, 4)
TYPES = [5, 1095, 5, 2]


def _bf16_rows(rng: np.random.Generator, M: int, K_: int) -> np.ndarray:
    x = rng.standard_normal((M, K_)).astype(np.float32) * np.exp2(rng.uniform(-3, 3, (M, 1))).astype(np.float32)
    return (x.view(np.uint32) >> 16).astype(np.uint16)


def _align(ids_flat: np.ndarray, block: int = 8, experts: int = E) -> tuple[np.ndarray, np.ndarray, int]:
    """`moe_align_block_size` as the kernel reads it: each expert's pairs in order, padded to a moe block with the pair count."""
    st, ex = [], []
    for e in range(experts):
        p = np.flatnonzero(ids_flat == e).tolist()
        for b in range(0, len(p), block):
            blk = p[b:b + block]
            st += blk + [ids_flat.size] * (block - len(blk))
            ex.append(e)
    return np.array(st), np.array(ex), len(st)


def _weights(tmp_path) -> dict[str, str]:
    rng = np.random.default_rng(7)
    wq = rng.integers(0, 16, (E, N, K)).astype(np.uint8)
    ws = rng.integers(118, 128, (E, N, K // X.GROUP)).astype(np.uint8)
    edge = rng.random((N, K // X.GROUP)) < 0.3
    ws[0] = np.where(edge, rng.choice(X.EDGE_BYTES, (N, K // X.GROUP)), ws[0])
    return X._save(tmp_path / "weights", wq=wq, ws=ws)


def _launch(tmp_path, kind: str, at=TRUE, *, M: int = 2, flip: tuple[int, int] | None = None, phase: str = "auto",
            probe: bool = False) -> dict:
    """A w13-like launch (M tokens x 6 slots, rows x[p // 6]) or a w2-like one (6M rows, one slot each, the routed weight), its
    words the twin's at config ``at``, on ordinary rows or the probe's; expert 0 carries edge scale bytes, so some words are NaN or
    inf."""
    rng = np.random.default_rng(11 + M)
    ids = np.stack([rng.permutation(E)[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex, ntpp = _align(ids)
    ok, blk, e, par = X.assignment(st, ex, ntpp, ids, block=8)
    assert ok
    top_k = X.TOP_K if kind == "w13" else 1
    P = M * X.TOP_K
    w = X._load(paths := _weights(tmp_path))
    if probe:
        x = X.probe(np.random.default_rng(3), [e[r * top_k:(r + 1) * top_k] for r in range(P // top_k)], w["wq"], w["ws"])[0]
    else:
        x = _bf16_rows(rng, P // top_k, K)
    g = (rng.uniform(0.05, 1, P).astype(np.float32)).view(np.uint32) if kind == "w2" else None
    C = MR.launch(x[np.arange(P) // top_k], blk, par, e, w["wq"], w["ws"], np.arange(N), TK=at[0], TN=at[1], GRID=SMS * at[2], g=g)
    if flip:
        C[flip] ^= 0x0001
    arrays = {"x": x, "C": C, "blk": blk, "e": e, **({"g": g} if g is not None else {})}
    spec = {"phase": phase, "case": "decode2", "launch": kind, "config": None if phase == "auto" else list(at), "M": M, "K": K, "N": N,
            "E": E, "top_k": top_k, "pairs": P, "par": par, "weights": paths,
            "paths": X._save(tmp_path / f"{phase}_{kind}_{'_'.join(map(str, at))}_{int(probe)}", **arrays)}
    if phase == "forced":
        spec["statics"] = [at[0], at[1], SMS * at[2]]
    return spec


def _total(results: list[dict]) -> dict:
    assert not [r for r in results if "error" in r], results
    return {k: sum(r.get(k, 0) for r in results) for k in X.TOTALS}


def _name(threads: int, n_blocks: int, k_blocks: int, *, m8: str = "true", stages: int = 4, group_blocks: int = 2,
          m_blocks: int = 1) -> str:
    return (f"void marlin_moe_wna16::Marlin<(vllm::ScalarTypeId)5, 1095l, 5l, 2l, {threads}, {m_blocks}, {n_blocks}, {k_blocks}, {m8}, "
            f"{stages}, {group_blocks}, false>(int4 const*, int4 const*, int4*, int4*, int4 const*, float const*, int4 const*, "
            "float const*, int4 const*, int const*, int const*, int const*, int const*, float const*, int, bool, int, int, int, int, "
            "int*, bool, bool, bool)")


def _kernel_event(ts: int, name: str, grid, block) -> dict:
    return {"ph": "X", "cat": "kernel", "name": name, "ts": ts, "dur": 5,
            "args": {"correlation": ts, "grid": list(grid), "block": list(block), "registers per thread": 168, "shared memory": 0}}


def test_the_kernel_name_and_grid_read_the_launch_config() -> None:
    trace = {"traceEvents": [_kernel_event(30, _name(256, 8, 8), [SMS, 1, 1], [256, 1, 1]),
                             _kernel_event(20, "void vllm::moe::moe_align_block_size_kernel<int>(...)", [1, 1, 1], [1024, 1, 1]),
                             _kernel_event(10, _name(128, 8, 4), [4 * SMS, 1, 1], [128, 1, 1]),
                             {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "ts": 9}]}
    got = X.marlin_kernels(trace)
    assert [k["template"]["thread_k_blocks"] for k in got] == [4, 8], "launch order, Marlin kernels only"
    assert got[0]["template"] == {"a_type": 5, "b_type": 1095, "c_type": 5, "s_type": 2, "threads": 128, "thread_m_blocks": 1,
                                  "thread_n_blocks": 8, "thread_k_blocks": 4, "m_block_size_8": True, "stages": 4, "group_blocks": 2,
                                  "is_zp_float": False}
    assert X.kernel_config(got[0], SMS, TYPES, block=8) == ([64, 128, 4], True)
    assert X.kernel_config(got[1], SMS, TYPES, block=8) == ([128, 128, 1], True)
    assert X.kernel_config(got[1], SMS, None, block=8) == ([128, 128, 1], True)
    assert not X.kernel_config(got[1], SMS, [5, 1095, 5, 3], block=8)[1], "the scale type"
    bad = [({"grid": [4 * SMS + 1, 1, 1]}, None), ({"grid": [5 * SMS, 1, 1]}, None), ({"grid": [SMS, 2, 1]}, None),
           ({"block": [256, 1, 1]}, None), (None, _name(128, 8, 4, stages=2)), (None, _name(128, 8, 4, group_blocks=4)),
           (None, _name(128, 8, 4, m8="false")), (None, _name(256, 8, 4)), (None, _name(128, 4, 4)),
           (None, _name(128, 8, 4, m_blocks=2))]
    for args, name in bad:
        k = copy.deepcopy(got[0])
        k.update(args or {})
        if name:
            k["template"] = X.kernel_template(name)
        assert not X.kernel_config(k, SMS, TYPES, block=8)[1], (args, name)
    assert not X.kernel_config(got[0], SMS, TYPES, block=16)[1], "a block of 16 runs no m_block_size_8 kernel"


@pytest.mark.parametrize("block,name,b,ok", [
    (16, _name(128, 8, 4, m8="false"), 4, True), (16, _name(256, 8, 8, m8="false"), 1, True),
    (16, _name(128, 8, 4, m8="false", m_blocks=2), 2, False), (16, _name(128, 8, 4, m8="false"), 5, False),
    (32, _name(256, 16, 4, m8="false", m_blocks=2), 2, True), (32, _name(128, 8, 4, m8="false", m_blocks=2), 1, True),
    (32, _name(256, 16, 4, m8="false", m_blocks=2), 3, False), (32, _name(256, 8, 8, m8="false", m_blocks=2), 2, False),
    (32, _name(256, 16, 4, m8="false"), 2, False), (32, _name(256, 16, 4, m_blocks=2), 2, False)])
def test_a_larger_moe_block_s_kernel_is_its_own_instantiation(block: int, name: str, b: int, ok: bool) -> None:
    """At a moe block of 16 the small-batch configs at up to 4 blocks per SM without `m_block_size_8`; at 32 two m blocks over the
    large-batch configs at up to 2 (ops.cu:285-292, 325-328, 362-363)."""
    t = X.kernel_template(name)
    k = {"template": t, "grid": [b * SMS, 1, 1], "block": [t["threads"], 1, 1]}
    config, got = X.kernel_config(k, SMS, TYPES, block=block)
    assert config == [16 * t["thread_k_blocks"], 16 * t["thread_n_blocks"], b] and got is ok


def test_a_kernel_name_that_does_not_parse_is_no_config() -> None:
    assert X.kernel_template("void other::Marlin<1, 2>(int)") is None
    assert X.kernel_template(_name(128, 8, 4).replace("1095l", "x")) is None
    assert X.kernel_template(_name(128, 8, 4).replace(", false>", ", false, 7>")) is None
    assert X.kernel_config(None, SMS, TYPES, block=8) == (None, False)
    assert X.kernel_config({"template": None}, SMS, TYPES, block=8) == (None, False)


def _symbol(threads: int, n_blocks: int, k_blocks: int, *, group_blocks: str = "2") -> str:
    """`_name`'s kernel mangled, as its cubin names it."""
    return (f"_ZN16marlin_moe_wna166MarlinILl5ELl1095ELl5ELl2ELi{threads}ELi1ELi{n_blocks}ELi{k_blocks}ELb1ELi4ELi{group_blocks}ELb0EEEv"
            "PK4int4S3_PS1_S4_S3_PKfS3_S6_S3_PKiS8_S8_S8_S6_ibiiiiPibbb")


def _cubin(functions: dict[str, bytes]) -> bytes:
    """A little-endian ELF64 CUDA image: a null section, a `.text.<symbol>` section per function, a NOBITS `.nv.shared` section
    larger than the file, and the section-name table, with the section headers last."""
    names = b"\0" + b"".join(b".text." + s.encode() + b"\0" for s in functions) + b".nv.shared\0.shstrtab\0"
    body, sections, name_at = b"", [], 1
    for s, text in functions.items():
        sections.append((name_at, 1, 64 + len(body), len(text)))
        name_at += len(".text.") + len(s) + 1
        body += text
    sections.append((name_at, 8, 64 + len(body), 1 << 20))
    sections.append((name_at + len(".nv.shared") + 1, 3, 64 + len(body), len(names)))
    body += names
    shdrs = bytes(64) + b"".join(struct.pack("<IIQQQQIIQQ", n, t, 0, 0, o, z, 0, 0, 1, 0) for n, t, o, z in sections)
    ehdr = b"\x7fELF\x02\x01\x01\x33\x07" + bytes(7) + struct.pack("<HHIQQQIHHHHHH", 2, X.EM_CUDA, 1, 0, 0, 64 + len(body), 0x50055A, 64,
                                                                  0, 0, 64, len(sections) + 1, len(sections))
    return ehdr + body + shdrs


def _listing(sym: str, ops: tuple[str, ...] = ("HMMA.16816.F32.BF16", "FADD", "HMUL2.BF16_V2"), arch: str = "sm_90") -> str:
    lines = "".join(f"        /*{16 * i:04x}*/              {'@!P0 ' if i % 2 else ''}{op} R1, R2, R3 ;   /* 0x000fe20000000a00 */\n"
                    for i, op in enumerate(ops))
    return f"\n\tcode for {arch}\n\t\tFunction : {sym}\n\t.headerflags\t@\"EF_CUDA_SM90\"\n{lines}"


def test_a_mangled_marlin_symbol_reads_as_its_profiled_name() -> None:
    for threads, n, k in ((256, 8, 8), (128, 8, 4), (128, 4, 8)):
        assert X.mangled_template(_symbol(threads, n, k)) == X.kernel_template(_name(threads, n, k)) is not None
    assert X.mangled_template(_symbol(128, 8, 4, group_blocks="n1"))["group_blocks"] == -1
    assert X.mangled_template("_ZN7vllm3moe25moe_align_block_size_kernelIiEEvPKT_") is None
    assert X.mangled_template(_symbol(128, 8, 4).replace("ELb0EEEv", "EEEv")) is None, "eleven arguments"


def test_a_cuda_image_is_found_behind_a_cache_header() -> None:
    img = _cubin({_symbol(128, 8, 4): b"\x01" * 48, "_Z5otherv": b"\x02" * 16})
    blob = b"\x7fELF-not-an-image" + b"\0" * 23 + img + b"\x7fELF" + b"trailer"
    assert X.cuda_elfs(blob) == [(40, img)], "the NOBITS section takes no bytes; a truncated magic is no image"
    assert X.elf_functions(img) == {_symbol(128, 8, 4): b"\x01" * 48, "_Z5otherv": b"\x02" * 16}
    assert X.cuda_elfs(img[:-1]) == [] and X.cuda_elfs(img[:18] + b"\x3e\x00" + img[20:]) == [], "truncated; an x86-64 image"


def test_the_jit_phase_reads_each_kernel_back_from_the_driver_cache(tmp_path, monkeypatch) -> None:
    sym, text = _symbol(128, 8, 4), b"\x07" * 64
    served = [{"template": X.kernel_template(_name(128, 8, 4)), "registers": [120]}]
    listings = {"-sass": _listing(sym), "-res-usage": f"Resource usage:\n Common:\n  GLOBAL:0\n Function {sym}:\n  REG:120 STACK:0\n"}
    monkeypatch.setattr(X, "_cuobjdump", lambda *a: listings[a[0]])
    monkeypatch.setattr(X, "driver_libraries", lambda: {"libcuda.so.580.173.02": {"sha256": "ab"}})
    cache = tmp_path / "cache"
    (cache / "1" / "f").mkdir(parents=True)
    (cache / "index").write_bytes(bytes(32))
    img = _cubin({sym: text, "_Z5otherv": b"\x02" * 16})
    (cache / "1" / "f" / "entry").write_bytes(b"H" * 24 + img)

    def run(served=served) -> dict:
        return X.jit(cache, tmp_path / "out" / "jit", served, "sm_90")

    rec = run()
    k = rec["kernels"][0]
    assert rec["ok"] and rec["files"] == 2 and rec["cubins"] == [{"file": "1/f/entry", "offset": 24, "bytes": len(img),
                                                                  "sha256": hashlib.sha256(img).hexdigest(), "e_flags": "0x50055a",
                                                                  "functions": 2, "marlin": 1}]
    assert k["symbol"] == sym and k["text_sha256"] == [hashlib.sha256(text).hexdigest()] and k["registers"] == 120 and k["arch"] == "sm_90"
    assert k["fp_ops"] == {"FADD": 1, "HMMA.16816.F32.BF16": 1, "HMUL2.BF16_V2": 1}
    assert (tmp_path / "out" / k["kept"]).read_bytes() == img
    assert not run([{**served[0], "registers": [121]}])["ok"], "the profiled kernel's registers"
    assert not run([{**served[0], "registers": [120, 121]}])["ok"]
    for listing in (_listing(sym, ("HMMA.16816.F32.BF16", "FFMA")), _listing(sym, ("HMMA.16816.F32.TF32",)), _listing(sym, ("FADD",)),
                    _listing(sym, arch="sm_80"), _listing("_Z5otherv")):
        listings["-sass"] = listing
        assert not run()["ok"], listing
    listings["-sass"] = _listing(sym)
    other = {**served[0], "template": X.kernel_template(_name(256, 8, 8))}
    assert run([other])["kernels"][0]["found"] is False and not run([other])["ok"] and not run([])["ok"]
    (cache / "1" / "f" / "entry2").write_bytes(b"H" * 8 + _cubin({sym: b"\x08" * 64}))
    rec = run()
    assert not rec["ok"] and len(rec["kernels"][0]["text_sha256"]) == 2, "one template, two SASS"


def test_the_gate_holds_the_driver_to_the_map_s(monkeypatch) -> None:
    import json

    assert X.pinned_driver() == json.loads(X.MAP.read_text())["target"]["driver"]["version"] == "580.173.02"
    monkeypatch.setattr(X.C, "gate", lambda *a: {"driver": "580.173.02", "want": {"min_driver": 575}, "checks": {"driver": True},
                                                 "ok": True})
    assert X.gate(575, "9.0", 132, "580.173.02")["ok"]
    g = X.gate(575, "9.0", 132, "580.126.09")
    assert not g["ok"] and g["checks"]["driver_pinned"] is False and g["want"]["driver"] == "580.126.09"


def test_the_probe_tells_summation_orders_apart(tmp_path) -> None:
    """Rounded once to bf16, fp32 sums in different orders give the same words on ordinary rows; on the probe's they do not."""
    w = X._load(_weights(tmp_path))
    rng = np.random.default_rng(13)
    M = 2
    ids = np.stack([rng.permutation(np.arange(1, E))[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex, ntpp = _align(ids)
    _, blk, e, par = X.assignment(st, ex, ntpp, ids, block=8)
    rows = [e[r * X.TOP_K:(r + 1) * X.TOP_K] for r in range(M)]
    xp, shifts = X.probe(np.random.default_rng(1), rows, w["wq"], w["ws"])
    assert xp.shape == (M, K) and xp.dtype == np.uint16 and np.all((xp & 0x7F80) != 0x7F80) and np.all((xp & 0x7F80) != 0)
    vals = (xp.astype(np.uint32) << 16).view(np.float32).astype(np.float64)
    for r, es in enumerate(rows):
        sums = np.abs(np.concatenate([X.dequant64(w["wq"][q], w["ws"][q]) @ vals[r] for q in es]))
        assert abs(np.log2(np.median(sums)) - X.PROBE_LOG2) <= 0.5 + 1e-9, (r, shifts)
    plain = _bf16_rows(rng, M, K)
    classes = {}
    for name, x in (("probe", xp), ("plain", plain)):
        got: dict[bytes, list] = {}
        for c in X.candidates(8):
            C = MR.launch(x[np.arange(M * X.TOP_K) // X.TOP_K], blk, par, e, w["wq"], w["ws"], np.arange(N), TK=c[0], TN=c[1],
                          GRID=SMS * c[2])
            got.setdefault(C.tobytes(), []).append(c)
        classes[name] = list(got.values())
    assert len(classes["plain"]) == 1 and len(classes["probe"]) >= 6, classes["probe"]
    assert [TRUE] in classes["probe"], "the stand-in's config has words of its own"


def test_dequant64_reads_e8m0_as_the_bf16_view() -> None:
    q = np.array([[1, 7, 9, 0] * 8])
    s = np.array([[127], [0], [255]])
    got = X.dequant64(np.repeat(q, 3, axis=0), s)
    assert got[0, :4].tolist() == [0.5, 6.0, -0.5, 0.0] and np.all(got[1] == 0)
    assert np.isinf(got[2, 0]) and np.isnan(got[2, 3])


@pytest.mark.parametrize("kind", ["w13", "w2"])
def test_word_and_reference_jobs_take_the_kernel_words(tmp_path, kind: str) -> None:
    spec = {**_launch(tmp_path, kind), "config": list(TRUE)}
    statics = [TRUE[0], TRUE[1], SMS * TRUE[2]]
    t = _total([X.compare(j) for j in X.word_jobs(spec, statics) + [X.reference_job(spec, statics)]])
    assert t["words"] == spec["pairs"] * N and t["mismatch"] == t["twin_mismatch"] == 0
    P = spec["pairs"]
    assert t["reference_schedules"] > 0 and t["reference_coords"] == sum(len({0, N - 1, p * 7919 % N}) for p in {0, P - 1})
    assert t["reference_expert_coords"] == (1 if kind == "w2" else 0)
    flipped = {**_launch(tmp_path, kind, flip=(7, 100)), "config": list(TRUE)}
    res = [X.compare(j) for j in X.word_jobs(flipped, statics)]
    assert _total(res)["mismatch"] == 1 and next(r["first_mismatch"] for r in res if r["first_mismatch"])["pair"] == 7
    probed = {**_launch(tmp_path, kind, probe=True), "config": list(TRUE)}
    assert _total([X.compare(j) for j in X.word_jobs(probed, statics)])["mismatch"] == 0
    other = [X.compare(j) for j in X.word_jobs(probed, [128, 128, SMS])]
    assert _total(other)["mismatch"] > 0, "on the probe the words read the config they were made at"


def test_the_jobs_split_a_large_launch() -> None:
    jobs = X.word_jobs({"pairs": 96, "N": X.HIDDEN, "K": X.INTER}, [128, 64, 264])
    assert jobs[0]["pair_range"][0] == 0 and jobs[-1]["pair_range"][1] == 96 and len(jobs) > 1
    assert all(a["pair_range"][1] == b["pair_range"][0] for a, b in zip(jobs, jobs[1:]))


#: the per-config limits of the large-batch kernels a block of 32 runs: 2 blocks per SM each
LARGE = {(64, 256): 2, (64, 128): 2, (128, 64): 2}


def _served(allow: dict, sms: int = 132, large: dict = LARGE) -> dict:
    """The config at each swept M under ``allow`` at the blocks of 8 and 16 and ``large`` at 32."""
    return {kind: {str(M): list(X._auto(M, kind, sms, large if X.block_of(M) == 32 else allow)) for M in X.SERVED_M}
            for kind in ("w13", "w2")}


@pytest.mark.parametrize("allow,large", [({(128, 128): 2, (64, 128): 4, (128, 64): 4}, LARGE),
                                         ({(128, 128): 1, (64, 128): 1, (128, 64): 3}, {(64, 256): 1, (64, 128): 2}),
                                         ({(64, 128): 2, (128, 64): 2}, {(128, 64): 1})])
def test_the_model_finds_the_limits_behind_the_sweep(allow: dict, large: dict) -> None:
    got = X.model(_served(allow, large=large), 132)
    blocks = {8: 18, 16: 2, 32: 2}
    assert set(got) == {f"{kind}/{b}" for kind in ("w13", "w2") for b in blocks}
    for kind in ("w13", "w2"):
        for b, n in blocks.items():
            assert got[f"{kind}/{b}"]["observed"] == n and got[f"{kind}/{b}"]["consistent"] >= 1, (kind, b, got[f"{kind}/{b}"])
    if large == LARGE:
        assert got["w13/32"]["consistent"] == got["w2/32"]["consistent"] == 9, "(64, 256) at 2 wins whatever the other two allow"
    bad = _served(allow, large=large)
    bad["w2"]["1"] = [128, 64, 1] if bad["w2"]["1"] != [128, 64, 1] else [64, 128, 1]
    assert X.model(bad, 132)["w2/8"]["consistent"] < got["w2/8"]["consistent"] or X.model(bad, 132)["w2/8"]["consistent"] == 0
    assert X.model(bad, 132)["w2/32"] == got["w2/32"], "a block's limits read only its own batches"
    bad = _served(allow, large=large)
    bad["w13"]["1024"] = [128, 128, 2]
    assert X.model(bad, 132)["w13/32"]["consistent"] == 0, "(128, 128) is no large-batch config"


def _served_launch(kind: str, M: int) -> dict:
    K_, N_ = X.shape(kind)
    return {"args": {"moe_block_size": X.block_of(M), "top_k": X.launch_top_k(kind), "mul_topk_weights": kind == "w2",
                     "size_m": X.prob_m(kind, M), "size_n": N_, "size_k": K_, "is_k_full": True, "use_atomic_add": False, "use_fp32_reduce": True, "is_zp_float": False,
                     "b_q_type": "float4_e2m1f", "thread_k": -1, "thread_n": -1, "blocks_per_sm": -1},
            "absent": sorted(X.ABSENT), "dtypes": {"input": "torch.bfloat16", "b_scales": "torch.float8_e8m0fnu",
                                                   "topk_weights": "torch.float32", "b_qweight": "torch.int32", "output": "torch.bfloat16"}}


def _kernel(config, sms: int = 132, block: int = 8) -> dict:
    tk, tn, b = config
    threads = next(t for k, n, t in MR.thread_configs(block) if (k, n) == (tk, tn))
    name = _name(threads, tn // 16, tk // 16, m8="true" if block == 8 else "false", m_blocks=MR.thread_m_blocks(block))
    return {"template": X.kernel_template(name), "grid": [sms * b, 1, 1], "block": [threads, 1, 1], "registers": 168, "shared": 0,
            "name": "Marlin"}


def test_the_sweep_names_a_served_config_at_every_batch() -> None:
    allow = {(128, 128): 2, (64, 128): 4, (128, 64): 4}
    served = _served(allow)
    launches = [{**_served_launch(kind, M), "label": f"sweep/{M}", "kernel": _kernel(served[kind][str(M)], block=X.block_of(M))}
                for M in X.SERVED_M for kind in ("w13", "w2")]
    got = X.sweep_record(launches, 132, TYPES)
    assert got["ok"] and got["statics_ok"] and got["served"] == served and got["model"]["w2/8"]["consistent"] >= 1
    assert served["w13"]["1024"] == served["w2"]["922"] == [64, 256, 2] and served["w13"]["461"] == [64, 128, 4]
    assert X.served_config(got, "w13", 16) == served["w13"]["16"] and X.served_config({}, "w13", 1) is None
    assert not X.sweep_record(launches[:-1], 132, TYPES)["ok"], "a batch the sweep did not reach"
    broken = copy.deepcopy(launches)
    broken[3]["kernel"]["block"] = [64, 1, 1]
    assert not X.sweep_record(broken, 132, TYPES)["ok"] and X.sweep_record(broken, 132, TYPES)["served"]["w2"]["2"] is None
    broken = copy.deepcopy(launches)
    broken[0]["args"]["use_atomic_add"] = True
    assert not X.sweep_record(broken, 132, TYPES)["ok"]
    for M, block in ((461, 8), (922, 16), (460, 16)):
        broken = copy.deepcopy(launches)
        at = next(i for i, L in enumerate(broken) if L["label"] == f"sweep/{M}" and L["args"]["size_n"] == 768)
        broken[at]["kernel"] = _kernel([64, 128, 2], block=block)
        assert X.sweep_record(broken, 132, TYPES)["served"]["w13"][str(M)] is None, (M, block)
        broken[at]["args"]["moe_block_size"] = block
        assert not X.sweep_record(broken, 132, TYPES)["statics_ok"], "M's moe block is fused_marlin_moe's"


def _tensor(a, dtype=torch.int32) -> torch.Tensor:
    return torch.tensor(np.asarray(a), dtype=dtype)


@pytest.mark.parametrize("M", [2, 461, 922])
def test_a_case_s_records_read_its_two_runs(tmp_path, M: int) -> None:
    """`case_launches` over intercepted launches as the Interceptor leaves them (CPU tensors here), at a moe block of 8, 16 and 32:
    the assignment, the statics, the kernel's config in both runs, determinism, and the arrays the jobs read."""
    block = X.block_of(M)
    rng = np.random.default_rng(5)
    ids = np.stack([rng.permutation(np.arange(8, 40))[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex, ntpp = _align(ids, block, experts=40)
    w13_at = (64, 256, 2) if block == 32 else (128, 64, 2)
    launches = []
    for run in ("run1", "run2"):
        for kind in ("w13", "w2"):
            K_, N_ = X.shape(kind)
            rows = M if kind == "w13" else M * X.TOP_K
            out = torch.zeros(M * X.TOP_K, N_, dtype=torch.bfloat16)
            if run == "run2" and kind == "w2":
                out[0, 0] = 1.0
            launches.append({**_served_launch(kind, M), "label": f"decode2/{run}", "output": out,
                             "input": torch.zeros(rows, K_, dtype=torch.bfloat16), "sorted_token_ids": _tensor(st),
                             "expert_ids": _tensor(ex), "num_tokens_past_padded": _tensor([ntpp]),
                             "topk_weights": torch.full((M, X.TOP_K), 0.5), "workspace": 0,
                             "kernel": _kernel(w13_at if kind == "w13" else (64, 128, 1 if run == "run1" else 2), block=block)})
    W = {"paths": {"w13": {"wq": "-", "ws": "-"}, "w2": {"wq": "-", "ws": "-"}}}
    records, specs, captured = X.case_launches(tmp_path, "decode2", M, "uniform", _tensor(ids.reshape(M, X.TOP_K)), launches, W, 132,
                                               TYPES)
    w13, w2 = records
    assert w13["assignment_ok"] and w13["statics_ok"] and w13["unmoved_equal"] and w13["kernel_ok"] and w13["config_deterministic"]
    assert w13["moved_pairs"] == 0 and w2["moved_pairs"] == 0, "both runs read one _align here"
    assert w13["config"] == list(w13_at) and w13["moe_block"] == block and w13["par"] == len(ex) == ntpp // block
    assert w13["experts_used"] == len(set(ex.tolist())) and (len(ex) > w13["experts_used"]) is (M > 2)
    assert not w2["unmoved_equal"] and not w2["config_deterministic"] and w2["config"] == [64, 128, 1]
    assert [s["launch"] for s in specs] == ["w13", "w2"] and [c["launch"] for c in captured] == ["w13", "w2"]
    arrays = X._load(specs[1]["paths"])
    assert arrays["x"].shape == (M * X.TOP_K, X.INTER) and arrays["g"].tolist() == [0x3F000000] * (M * X.TOP_K)
    assert arrays["blk"].tolist() == w2_blocks(st, ids.size, block)
    with pytest.raises(RuntimeError):
        X.case_launches(tmp_path, "decode2", M, "uniform", _tensor(ids), launches[:3], W, 132, TYPES)


def test_two_align_draws_split_a_case_into_unmoved_and_moved_pairs(tmp_path) -> None:
    """Run r20261014-065141-a1ec: decode460 and hot1024 were NO-GO on the v2 whole-output comparison at zero word mismatches
    against the twin, because `moe_align_block_size` places a multi-block expert's pairs across its blocks in atomic arrival order
    and a pair's words read its block's k-slice schedule (the capture's module docstring, Replay determinism).  Two runs whose
    aligns placed one expert's pairs differently must still check out: the pairs both placed alike are held to each other
    (`unmoved_equal`, which run2's differing moved rows do not break), and run2's moved pairs get a spec of their own, at run2's
    recorded blocks."""
    ids = np.repeat(np.arange(1, 5), 9)                      # 6 tokens x top-6: experts 1..4, 9 pairs each, two moe blocks each
    M = ids.size // X.TOP_K
    st1, ex, ntpp = _align(ids)
    st2 = st1.copy()
    st2[[0, 8]] = st2[[8, 0]]                                # expert 1's first pair swaps moe blocks with its ninth
    _, blk1, e, par = X.assignment(st1, ex, ntpp, ids, block=8)
    ok2, blk2, _, _ = X.assignment(st2, ex, ntpp, ids, block=8)
    assert ok2 and par == 8 > len(set(ids.tolist()))
    moved = np.flatnonzero(blk1 != blk2)
    assert sorted(moved.tolist()) == sorted(int(v) for v in (st1[0], st1[8]))
    launches = []
    for run, st in (("run1", st1), ("run2", st2)):
        for kind in ("w13", "w2"):
            K_, N_ = X.shape(kind)
            out = torch.zeros(ids.size, N_, dtype=torch.bfloat16)
            if run == "run2":
                out[torch.from_numpy(moved)] = 1.0           # a moved pair's words may change; an unmoved pair's may not
            launches.append({**_served_launch(kind, M), "label": f"decode8/{run}", "output": out,
                             "input": torch.zeros(M if kind == "w13" else ids.size, K_, dtype=torch.bfloat16),
                             "sorted_token_ids": _tensor(st), "expert_ids": _tensor(ex), "num_tokens_past_padded": _tensor([ntpp]),
                             "topk_weights": torch.full((M, X.TOP_K), 0.5), "workspace": 528, "kernel": _kernel((64, 128, 4))})
    W = {"paths": {"w13": {"wq": "-", "ws": "-"}, "w2": {"wq": "-", "ws": "-"}}}
    records, specs, _ = X.case_launches(tmp_path, "decode8", M, "uniform", _tensor(ids.reshape(M, X.TOP_K)), launches, W, 132, TYPES)
    for r in records:
        assert r["assignment_ok"] and r["unmoved_equal"] and r["moved_pairs"] == moved.size, r["launch"]
    assert [(s["launch"], s.get("run")) for s in specs] == [("w13", None), ("w13", 2), ("w2", None), ("w2", 2)]
    for s in specs:
        arrays = X._load(s["paths"])
        want = blk2 if s.get("run") == 2 else blk1
        assert arrays["blk"].tolist() == want.tolist() and s.get("moved", moved.tolist()) == moved.tolist()
    bumped = copy.deepcopy(launches)
    bumped[2]["output"][int(np.setdiff1d(np.arange(ids.size), moved)[0]), 0] = 1.0
    records, _, _ = X.case_launches(tmp_path / "b", "decode8", M, "uniform", _tensor(ids.reshape(M, X.TOP_K)), bumped, W, 132, TYPES)
    assert not records[0]["unmoved_equal"], "an unmoved pair whose words changed still fails"


def test_a_moved_pair_s_words_read_its_own_block(tmp_path) -> None:
    """Run2's moved pairs are compared against the twin at run2's recorded blocks, where they match; at run1's they do not, since
    the two blocks' k-slice cut patterns differ (here par 8 over GRID 12: iters 3, and NT * KT = 4 is no multiple)."""
    ids = np.repeat(np.arange(1, 5), 9)
    M = ids.size // X.TOP_K
    st1, ex, ntpp = _align(ids)
    st2 = st1.copy()
    st2[[0, 8]] = st2[[8, 0]]
    _, blk1, e, par = X.assignment(st1, ex, ntpp, ids, block=8)
    _, blk2, _, _ = X.assignment(st2, ex, ntpp, ids, block=8)
    moved = np.flatnonzero(blk1 != blk2)
    w = X._load(paths := _weights(tmp_path))
    xp = X.probe(np.random.default_rng(2), [e[r * X.TOP_K:(r + 1) * X.TOP_K] for r in range(M)], w["wq"], w["ws"])[0]
    x_pairs = xp[np.arange(ids.size) // X.TOP_K]
    statics = [TRUE[0], TRUE[1], SMS * TRUE[2]]
    C2 = MR.launch(x_pairs, blk2, par, e, w["wq"], w["ws"], np.arange(N), TK=TRUE[0], TN=TRUE[1], GRID=statics[2])

    def spec(blk: np.ndarray) -> dict:
        return {"phase": "auto", "run": 2, "case": "decode8", "launch": "w13", "config": list(TRUE), "M": M, "K": K, "N": N, "E": E,
                "top_k": X.TOP_K, "pairs": int(ids.size), "par": par, "weights": paths, "moved": [int(v) for v in moved],
                "paths": X._save(tmp_path / f"run2_{blk[int(moved[0])]}", x=xp, C=C2, blk=blk, e=e)}

    jobs = X.word_jobs(spec(blk2), statics)
    t = _total([X.compare(j) for j in jobs])
    assert t["words"] == moved.size * N and t["mismatch"] == 0, "run2's words match the twin at run2's blocks"
    res = [X.compare(j) for j in X.word_jobs(spec(blk1), statics)]
    bad = {k: sum(r.get(k, 0) for r in res) for k in X.TOTALS}
    assert bad["mismatch"] > 0, "and not at run1's: the moved pair's block picks its k-slice fold"
    first = next(r["first_mismatch"] for r in res if r["first_mismatch"])
    assert first["pair"] in moved.tolist()


def test_the_replay_relaunches_the_recorded_call(monkeypatch) -> None:
    """`replay` re-launches run1's exact call into a SENTINEL-filled output with a fresh workspace and holds the words to run1's,
    as bits (the vLLM op stood in for here; on the GPU the kernel's slice fold is lock-serialized, so the real one must agree)."""
    import sys
    import types

    recorded = torch.tensor(np.array([[17, 0x7FA6], [3, 4]], dtype=np.int16)).view(torch.bfloat16)
    state = {"words": recorded.view(torch.int16).clone()}

    def gemm(**kw):
        assert kw["workspace"] == "fresh" and kw["output"] is not recorded and kw["input"] == "x"
        assert bool((kw["output"].view(torch.int16) == X.SENTINEL).all()), "the output starts as SENTINEL"
        kw["output"].view(torch.int16).copy_(state["words"])

    ops = types.ModuleType("vllm._custom_ops")
    ops.moe_wna16_marlin_gemm = gemm
    chain = ("vllm", "vllm.model_executor", "vllm.model_executor.layers", "vllm.model_executor.layers.quantization",
             "vllm.model_executor.layers.quantization.utils")
    mods = {name: types.ModuleType(name) for name in chain}
    mu = types.ModuleType("vllm.model_executor.layers.quantization.utils.marlin_utils")
    mu.marlin_make_workspace_new = lambda device, n: "fresh"
    mods["vllm"]._custom_ops = ops
    for name, mod in {**mods, "vllm._custom_ops": ops, "vllm.model_executor.layers.quantization.utils.marlin_utils": mu}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    L = {"output": recorded, "call": {"input": "x", "output": "stale", "workspace": "stale"}}
    assert X.replay(L)
    state["words"][0, 1] ^= 1
    assert not X.replay(L), "a word one bit off, a NaN payload among them, is not a replay"


def test_a_case_over_the_edge_experts_is_deterministic_by_its_words(tmp_path) -> None:
    """Run r20261013-205619-ce5e's edge8 (8 tokens routed over the edge experts 0..7 only, both launches at 64/128/4 on 132 SMs) was
    NO-GO on determinism alone, every word equal to the twin's: an edge expert's +inf scale byte on a +-0 code writes NaN words, and
    `torch.equal` on the bf16 outputs is false wherever one stands, so it called the two runs different whatever they wrote.
    `case_launches` compares the runs' words as bits (`same_words`): equal words are deterministic, and a word one bit off, a NaN's
    payload among them, is not."""
    C = X._load(_launch(tmp_path, "w13")["paths"])["C"]
    nan = ((C & 0x7F80) == 0x7F80) & ((C & 0x7F) != 0)
    assert nan.any(), "the twin writes NaN words on the edge expert"
    M = 8
    rng = np.random.default_rng(8)
    ids = np.stack([rng.permutation(X.EDGE_EXPERTS)[:X.TOP_K] for _ in range(M)]).reshape(-1)
    st, ex, ntpp = _align(ids)

    def output(kind: str, payload: bool) -> torch.Tensor:
        words = np.resize(C, (M * X.TOP_K, X.shape(kind)[1])).astype(np.uint16)
        if payload:
            words[np.unravel_index(np.flatnonzero(((words & 0x7F80) == 0x7F80) & ((words & 0x7F) != 0))[0], words.shape)] ^= 0x0001
        return torch.from_numpy(words.view(np.int16).copy()).view(torch.bfloat16)

    def records(bumped: str | None) -> list[dict]:
        launches = []
        for run in ("run1", "run2"):
            for kind in ("w13", "w2"):
                K_, _N = X.shape(kind)
                launches.append({**_served_launch(kind, M), "label": f"edge8/{run}", "output": output(kind, run == "run2" and kind == bumped),
                                 "input": torch.zeros(M if kind == "w13" else M * X.TOP_K, K_, dtype=torch.bfloat16),
                                 "sorted_token_ids": _tensor(st), "expert_ids": _tensor(ex), "num_tokens_past_padded": _tensor([ntpp]),
                                 "topk_weights": torch.full((M, X.TOP_K), 0.5), "workspace": 528, "kernel": _kernel((64, 128, 4))})
        W = {"paths": {"w13": {"wq": "-", "ws": "-"}, "w2": {"wq": "-", "ws": "-"}}}
        return X.case_launches(tmp_path, "edge8", M, "edge", _tensor(ids.reshape(M, X.TOP_K)), launches, W, 132, TYPES)[0]

    a, b = output("w2", False), output("w2", False)
    assert not torch.equal(a, b) and X.same_words(a, b) and not X.same_words(a, output("w2", True))
    assert X.unmoved_rows_equal(a, b, np.zeros(0, dtype=np.int64)) and not X.unmoved_rows_equal(a, output("w2", True), np.zeros(0, dtype=np.int64))
    for r in records(None):
        assert r["unmoved_equal"] and X._launch_ok({**r, "identified": True, "replay_ok": True}), r["launch"]
        assert r["config"] == [64, 128, 4] and r["par"] == r["experts_used"] == X.EDGE_EXPERTS and r["pairs"] == M * X.TOP_K
    w13, w2 = records("w2")
    assert w13["unmoved_equal"] and not w2["unmoved_equal"] and not X._launch_ok({**w2, "identified": True, "replay_ok": True})


def w2_blocks(st: list[int], n: int, block: int) -> list[int]:
    blk = [0] * n
    for i, p in enumerate(st):
        if p < n:
            blk[p] = i // block
    return blk


def test_the_assignment_reads_the_moe_blocks() -> None:
    ids = np.array([5, 2, 5, 7, 1, 0])
    st, ex, ntpp = np.array([5, 6, 6, 6, 6, 6, 6, 6, 4, 6, 6, 6, 6, 6, 6, 6, 1, 6, 6, 6, 6, 6, 6, 6, 0, 2, 6, 6, 6, 6, 6, 6,
                             3, 6, 6, 6, 6, 6, 6, 6]), np.array([0, 1, 2, 5, 7]), 40
    ok, blk, e, par = X.assignment(st, ex, ntpp, ids, block=8)
    assert ok and par == 5 and blk.tolist() == [3, 2, 3, 4, 1, 0] and e.tolist() == ids.tolist()
    assert not X.assignment(st, np.array([0, 1, 2, 7, 5]), ntpp, ids, block=8)[0]
    lost = st.copy()
    lost[24] = 6
    assert not X.assignment(lost, ex, ntpp, ids, block=8)[0]
    assert not X.assignment(st, ex, 32, ids, block=8)[0], "a block past num_tokens_post_padded is not run"
    assert not X.assignment(st, ex, ntpp, ids, block=16)[0], "read at 16, blocks of 8 put two experts' pairs in one block"
    ids = np.repeat(np.array([3, 1]), 20)
    st, ex, ntpp = _align(ids, 16)
    ok, blk, e, par = X.assignment(st, ex, ntpp, ids, block=16)
    assert ok and par == len(ex) == 4 and ex.tolist() == [1, 1, 3, 3] and blk.tolist() == [2] * 16 + [3] * 4 + [0] * 16 + [1] * 4
    assert not X.assignment(st, ex, ntpp, ids, block=8)[0] and not X.assignment(st, ex, ntpp - 16, ids, block=16)[0]


@pytest.mark.parametrize("kind", ["w13", "w2"])
def test_the_statics_are_the_served_paths(kind: str) -> None:
    assert X.statics_ok(_served_launch(kind, 3), kind, 3)
    assert not X.statics_ok(_served_launch(kind, 3), kind, 4)
    for k, v in (("use_atomic_add", True), ("use_fp32_reduce", False), ("moe_block_size", 16), ("thread_k", 128),
                 ("mul_topk_weights", kind == "w13"), ("b_q_type", "float8_e4m3fn")):
        L = _served_launch(kind, 3)
        L["args"][k] = v
        assert not X.statics_ok(L, kind, 3), k
    L = _served_launch(kind, 3)
    L["absent"].remove("global_scale")
    assert not X.statics_ok(L, kind, 3)
    L = _served_launch(kind, 3)
    L["dtypes"]["b_scales"] = "torch.float8_e4m3fn"
    assert not X.statics_ok(L, kind, 3)
    for M, block in ((460, 8), (461, 16), (921, 16), (922, 32), (X.ENVELOPE, 32)):
        L = _served_launch(kind, M)
        assert L["args"]["moe_block_size"] == block and X.statics_ok(L, kind, M), M
        L["args"]["moe_block_size"] = 8 if block > 8 else 16
        assert not X.statics_ok(L, kind, M), M


def test_the_programs_bind_at_the_served_configs() -> None:
    got = X.bind_check({("w13", 128, 64, 264), ("w2", 64, 128, 528), ("w13", 64, 256, 264)})
    assert got["ok"] and len(got["bindings"]) == 3
    assert got["bindings"]["MarlinMoeExpertGemm_v1{E=384,K=7168,N=768,TK=128,TN=64,GRID=264,DOT={\"fn\":\"HopperBF16WgmmaDot16_v1\"}}"] == {
        "ok": True, "boolean": "MarlinMoeExpertGemm_v2{E=384,K=7168,N=768,TK=128,TN=64,GRID=264,DOT={\"fn\":\"HopperBF16WgmmaDot16_v2\"}}"}
    assert not X.bind_check(set())["ok"]


TOTALS = {"words": 10, "mismatch": 0, "reference_schedules": 2, "reference_coords": 6, "reference_expert_coords": 1, "twin_mismatch": 0}


def _doc() -> dict:
    auto = {"phase": "auto", "case": "decode1", "unmoved_equal": True, "replay_ok": True, "moved_pairs": 0, "statics_ok": True,
            "assignment_ok": True, "kernel_ok": True,
            "config_deterministic": True, "identified": True, "config": [128, 64, 2], "served": [128, 64, 2]}
    forced = {"phase": "forced", "case": "hot16", "config": [64, 128, 1]}
    probe = {"case": "hot16", "accepted": 2, "classes": [[[64, 128, 1]], [[128, 64, 1]]], "ok": True}
    return {"schema": X.SCHEMA, "gate": {"ok": True}, "errors": {}, "selection": {"ok": True}, "bind": {"ok": True}, "forced_phase": True,
            "sweep": {"ok": True}, "jit": {"ok": True},
            "probe": {"hot16/w13": {**probe, "launch": "w13"}, "hot16/w2": {**probe, "launch": "w2"}},
            "launches": [{**auto, "launch": "w13"}, {**auto, "launch": "w2"}, {**forced, "launch": "w13", "accepted": True},
                         {**forced, "launch": "w2", "accepted": True}, {**forced, "launch": "w2", "accepted": False}],
            "comparisons": [{"phase": "auto", "case": "decode1", "launch": "w13", "config": [128, 64, 2], **TOTALS},
                            {"phase": "forced", "case": "hot16", "launch": "w2", "config": [64, 128, 1], **TOTALS}],
            "totals": TOTALS}


@pytest.mark.parametrize("path,value", [(("gate", "ok"), False), (("errors", "case/decode1"), "boom"), (("selection", "ok"), False),
                                        (("bind", "ok"), False), (("sweep", "ok"), False), (("jit", "ok"), False),
                                        (("totals", "mismatch"), 1),
                                        (("totals", "words"), 0), (("totals", "twin_mismatch"), 1), (("totals", "reference_coords"), 0),
                                        (("totals", "reference_schedules"), 0), (("totals", "reference_expert_coords"), 0),
                                        (("launches", 0, "unmoved_equal"), False), (("launches", 1, "replay_ok"), False),
                                        (("launches", 1, "statics_ok"), False),
                                        (("launches", 0, "assignment_ok"), False), (("launches", 1, "identified"), False),
                                        (("launches", 0, "kernel_ok"), False), (("launches", 1, "config_deterministic"), False),
                                        (("launches", 3, "accepted"), False), (("probe", "hot16/w2", "ok"), False)])
def test_the_verdict_needs_every_check(path: tuple, value) -> None:
    assert X._verdict(_doc())
    doc = copy.deepcopy(_doc())
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    assert not X._verdict(doc)
    assert not X._verdict({**_doc(), "sweep": {}})
    assert not X._verdict({k: v for k, v in _doc().items() if k != "jit"}), "no jit phase"
    assert not X._verdict({**_doc(), "launches": _doc()["launches"][:1] + _doc()["launches"][2:]}), "both launch kinds"
    assert not X._verdict({**_doc(), "probe": {"hot16/w13": _doc()["probe"]["hot16/w13"]}}), "a forced launch without its probe"
    assert X._verdict({**_doc(), "forced_phase": False, "launches": _doc()["launches"][:2], "probe": {}})


def test_each_case_has_its_own_verdict() -> None:
    doc = _doc()
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"decode1": True, "hot16": False}, "hot16 ran no auto launch here"
    assert got["decode1"]["configs"] == ["w13:128/64/2", "w2:128/64/2"] and got["hot16"]["refused"] == 1
    doc["comparisons"][0]["mismatch"] = 1
    doc["errors"]["case/edge8"] = "boom"
    got = X.cases(doc)
    assert {k: c["ok"] for k, c in got.items()} == {"decode1": False, "hot16": False, "edge8": False}
    doc = _doc()
    doc["errors"]["compare/decode1/w2/reference/[128, 64, 2]"] = "boom"
    assert X.cases(doc)["decode1"]["ok"] is False
    doc = _doc()
    doc["probe"]["hot16/w2"]["ok"] = False
    assert X.cases(doc)["hot16"]["launches_ok"] is False


@pytest.mark.parametrize("flip", [False, True])
def test_main_end_to_end_on_stubbed_launches(tmp_path, monkeypatch, capsys, flip: bool) -> None:
    """`main` with the GPU phases stood in for (the gate, a sweep serving TRUE, launches whose words the twin made at TRUE, a forced
    w2 launch at another config with one word flipped or not, the jit phase) and vLLM's config module stubbed: the environment and
    the compute cache it points the driver at, the driver pinned to the map's, the comparisons through the spawned worker pool,
    identification against the sweep, the bindings, the per-case verdicts, the sealed record and the exit status, so a slip in the
    orchestration shows here and not in the H200's first hour."""
    import contextlib
    import json
    import os
    import sys
    import types

    vc = types.ModuleType("vllm.config")
    vc.CompilationConfig = vc.VllmConfig = lambda **kw: kw
    vc.set_current_vllm_config = lambda cfg: contextlib.nullcontext()
    monkeypatch.setitem(sys.modules, "vllm", types.ModuleType("vllm"))
    monkeypatch.setitem(sys.modules, "vllm.config", vc)
    for k in ("VLLM_BATCH_INVARIANT", "VLLM_ENABLE_V1_MULTIPROCESSING", "CUDA_CACHE_MAXSIZE"):
        monkeypatch.setenv(k, "")
    for k in X.JIT_ENV[:2]:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("VLLM_MARLIN_INPUT_DTYPE", "fp8")
    monkeypatch.setenv("CUDA_FORCE_PTX_JIT", "1")
    monkeypatch.setenv("CUDA_CACHE_PATH", "/host/cache/cuda/k-0")
    monkeypatch.setattr(X.C, "gate", lambda *a: {"ok": True, "sms": SMS, "driver": "580.173.02", "want": {}, "checks": {}})
    monkeypatch.setattr(X.G, "_environment", dict)
    forced_at = (128, 64, 2)

    def jit(cache, keep, served, arch):
        assert os.environ["CUDA_CACHE_PATH"] == str(cache) and cache.is_dir() and not any(cache.iterdir())
        assert keep == out / "jit" and served == [] and arch == "sm_90"
        return {"ok": True, "kernels": []}

    monkeypatch.setattr(X, "jit", jit)

    def gpu(doc, work, sms, quick, forced, only=None):
        assert sms == SMS and forced and not quick and only is None
        doc["selection"] = {"ok": True}
        doc["sweep"] = {"ok": True, "served": {k: {str(M): list(TRUE) for M in X.SERVED_M} for k in ("w13", "w2")}}
        specs = [{**_launch(tmp_path / "a", kind, M=1), "case": "decode1"} for kind in ("w13", "w2")]
        doc["launches"] += [{"phase": "auto", "case": "decode1", "launch": s["launch"], "M": 1, "pairs": s["pairs"], "par": s["par"],
                             "unmoved_equal": True, "replay_ok": True, "moved_pairs": 2, "statics_ok": True, "assignment_ok": True,
                             "kernel_ok": True, "config_deterministic": True, "config": list(TRUE)} for s in specs]
        specs.append({**specs[0], "run": 2, "moved": [0, 5]})
        fs = {**_launch(tmp_path / "f", "w2", forced_at, M=1, flip=(3, 5) if flip else None, phase="forced", probe=True),
              "case": "decode1"}
        doc["launches"] += [{"phase": "forced", "case": "decode1", "launch": k, "config": list(forced_at), "accepted": True}
                            for k in ("w13", "w2")]
        doc["probe"] = {f"decode1/{k}": {"case": "decode1", "launch": k, "accepted": 1, "classes": [[list(forced_at)]], "ok": True}
                        for k in ("w13", "w2")}
        return specs, [fs]

    monkeypatch.setattr(X, "gpu_phases", gpu)
    out = tmp_path / "out"
    assert X.main(["--out", str(out), "--scratch", str(tmp_path), "--workers", "1"]) == (1 if flip else 0)
    doc = json.loads((out / X.RECORD).read_text())
    assert doc["digest"] == digest_of(doc) and doc["ok"] is (not flip) and not doc["errors"]
    assert doc["env"]["VLLM_BATCH_INVARIANT"] == "0" and doc["env"]["unset"] == {
        "VLLM_MARLIN_INPUT_DTYPE": "fp8", "CUDA_CACHE_DISABLE": None, "CUDA_DISABLE_PTX_JIT": None, "CUDA_FORCE_PTX_JIT": "1"}
    assert doc["env"]["replaced"] == {"CUDA_CACHE_PATH": "/host/cache/cuda/k-0"} and doc["env"]["CUDA_CACHE_MAXSIZE"] == str(4 << 30)
    assert doc["gate"]["want"]["driver"] == "580.173.02" and doc["gate"]["checks"]["driver_pinned"] and doc["jit"]["ok"]
    assert all(x["identified"] and x["served"] == list(TRUE) for x in doc["launches"][:2])
    assert doc["bind"]["ok"] and len(doc["bind"]["bindings"]) == 2
    assert doc["totals"]["mismatch"] == (1 if flip else 0) and doc["totals"]["reference_expert_coords"] == 1
    assert doc["totals"]["words"] == 3 * 6 * N + 2 * N, "the auto and forced launches whole, and run2's two moved pairs"
    assert sum(r.get("run") == 2 for r in doc["comparisons"]) == 1, "run2's moved pairs are their own comparison row"
    assert doc["cases"]["decode1"]["ok"] is (not flip)
    printed = capsys.readouterr().out
    assert f"DSV4-MARLIN {'NO-GO' if flip else 'GO'}" in printed and "case decode1" in printed
    assert f"served@M1={json.dumps({'w13': list(TRUE), 'w2': list(TRUE)})}" in printed


def test_the_record_is_a_registered_property() -> None:
    from verity_vllm.properties.record import seal

    doc = seal(_doc(), X._verdict(_doc()))
    assert REGISTRY["dsv4_marlin_capture"].schema == X.SCHEMA and doc["ok"] and doc["digest"] == digest_of(doc)


def test_the_cases_cover_the_decode_batches_and_the_edge_experts() -> None:
    Ms = {M for _, M, _ in X.CASES}
    assert Ms >= {1, 2, 3, 4} and max(Ms) == max(X.SERVED_M) == X.ENVELOPE == 1024 and Ms <= set(X.SERVED_M)
    assert set(X.QUICK) <= {c[0] for c in X.CASES} and set(X.FORCED) | set(X.QUICK_FORCED) <= {c[0] for c in X.CASES}
    assert [c for c in X.CASES if c[2] == "edge"] and 0 in X.EDGE_BYTES and 255 in X.EDGE_BYTES
    assert len(X.candidates(8)) == len(X.candidates(16)) == 12 and len(X.candidates(32)) == 6 and X.SENTINEL != 0x7FFF
    assert all(c[2] != "edge" for c in X.CASES if c[0] in X.FORCED + X.QUICK_FORCED), "the probe's scale reads finite weights"
    assert {X.block_of(M) for M in range(1, X.ENVELOPE + 1)} == {X.block_of(M) for M in Ms} == {8, 16, 32}
    assert {X.block_of(M) for M, pool in ((c[1], c[2]) for c in X.CASES) if pool == "edge"} == {8, 32}
    assert all(X.block_of(c[1]) == 8 for c in X.CASES if c[0] in X.FORCED + X.QUICK_FORCED)
    edges = {m for M in range(2, X.ENVELOPE + 1) if X.block_of(M) != X.block_of(M - 1) for m in (M - 1, M)}
    assert edges == {460, 461, 921, 922} and set(X.SERVED_M) == set(range(1, 18)) | edges | {X.ENVELOPE}
