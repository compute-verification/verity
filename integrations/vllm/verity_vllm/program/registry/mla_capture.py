"""Captures of vLLM d9105ea80's Triton MLA decode (`TritonMLAImpl.forward_mqa` -> `decode_attention_fwd`, grouped kernel,
`is_mla=True`) at GLM-4.7-Flash's shapes, for `mla.TritonMlaDecode`:

    # node, torch venv:  verity-vllm mla-capture produce --out DIR [--suite NAME] [--triton-cache DIR]
    # verity venv:       verity-vllm mla-capture check --dir DIR [--out REPORT.json]
    #                    verity-vllm mla-capture check-bits --dir DIR [--tmin T] [--tmax T]
    #                        [--worker I --workers N] [--skip EARLIER.json ...] [--fresh] [--out R.json]

`check` holds the numpy models (`mla_check`) to every captured word; `check-bits` evaluates the Boolean Definitions
themselves: `mla.TritonMlaDecodeHead` on the one-split decode rows, `mla.MlaBmm` and `mla.MoeSharedCombine` on every case.

The inputs are counter-based (`words`: splitmix64 of the case seed and the word index), so the check rebuilds them from the
manifest and only the kernel's outputs are stored: `o` (bf16 words `[B, H, 512]`), `lse` (bf16 `[B, H]`) and the stage-1
scratch `attn_logits` (f32 words `[B, H, SPLITS, 513]`: each split's `acc / e_sum` and `e_max + log(e_sum)`). `produce` calls
`decode_attention_fwd` exactly as forward_mqa does: q `[B, H, 576]`, the paged cache `[pages, PAGE, 1, 576]` and its first 512
columns as the value buffer, a shuffled block table, `k_scale = v_scale = 1.0`. With `TRITON_CACHE_DIR` set the compiled
kernels' TTGIR, LLIR, PTX and cubin land there; `produce` writes each cubin's SASS beside it (`--triton-cache`, which defaults
to that variable).
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from verity_vllm.config import option

DL, DR = 512, 64
D = DL + DR
H = 20
SCALE = 0.0625  # GLM-4.7-Flash: (qk_nope 192 + qk_rope 64) ** -0.5, no yarn

_M = np.uint64(0xFFFFFFFFFFFFFFFF)


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


#: exponent ranges [lo, hi] of the drawn bf16 words, per kind and tensor
KINDS = {
    "unit": {"q": (121, 128), "kv": (121, 128)},          # |x| in [2^-6, 4): ordinary scores
    "wide": {"q": (117, 131), "kv": (119, 129)},          # scores spread over hundreds: deep exp underflow
    "tiny": {"q": (1, 40), "kv": (1, 40)},                # products underflow: zero and subnormal scores
    "narrow": {"q": (110, 115), "kv": (110, 115)},        # near-uniform softmax
}
#: special words mixed into the "special" kind's cache rows (rate 1/64 each class)
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x807F, 0x7F7F, 0xFF7F, 0x0080, 0x3F80], np.uint16)


def words(seed: int, stream: int, n: int, kind: str, tensor: str) -> np.ndarray:
    """n bf16 words: sign, a uniform exponent in the kind's range, a uniform 7-bit mantissa; "special" adds SPECIALS."""
    idx = np.arange(n, dtype=np.uint64) + (np.uint64(stream) << np.uint64(40)) + (np.uint64(seed) << np.uint64(52))
    r = _splitmix(idx)
    base = "unit" if kind == "special" else kind
    lo, hi = KINDS[base][tensor]
    e = (lo + (r >> np.uint64(8)) % np.uint64(hi - lo + 1)).astype(np.uint16)
    w = ((r & np.uint64(1)).astype(np.uint16) << 15) | (e << 7) | ((r >> np.uint64(1)) & np.uint64(0x7F)).astype(np.uint16)
    if kind == "special" and tensor == "kv":
        pick = ((r >> np.uint64(24)) & np.uint64(63)) == 0
        w[pick] = SPECIALS[((r >> np.uint64(32)) % np.uint64(len(SPECIALS))).astype(np.int64)[pick]]
    return w


def case_inputs(case: dict) -> tuple[np.ndarray, list[np.ndarray]]:
    """(q words `[B, H, 576]`, per request its kv words `[T_b, 576]`)."""
    B, seed, kind = len(case["seqlens"]), int(case["seed"]), case["kind"]
    q = words(seed, 0, B * H * D, kind, "q").reshape(B, H, D)
    kv = [words(seed, 1 + b, T * D, kind, "kv").reshape(T, D) for b, T in enumerate(case["seqlens"])]
    return q, kv


def _lens(seed: int, B: int, hi: int) -> list[int]:
    r = _splitmix(np.arange(B, dtype=np.uint64) + (np.uint64(seed) << np.uint64(52)) + (np.uint64(0xFFFF) << np.uint64(40)))
    return [int(1 + v % np.uint64(hi)) for v in r]


EDGE_LENS = [1, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 2049,
             4095, 4096]


def suite(name: str) -> list[dict]:
    """The cases of a capture suite: batch sizes 1..32, lengths 1..4096, the batch-invariant split (1) and the served ones."""
    cases: list[dict] = []
    seed = 1

    def add(lens, splits, kind, page):
        nonlocal seed
        cases.append({"seqlens": [int(t) for t in lens], "splits": splits, "kind": kind, "page": page, "seed": seed})
        seed += 1

    if name == "smoke":
        add([1, 33, 100], 1, "unit", 16)
        add([64, 513], 2, "unit", 64)
        return cases
    if name == "decode-v1":
        for i in range(0, len(EDGE_LENS), 5):
            add(EDGE_LENS[i:i + 5], 1, "unit", 64)
        for kind in KINDS:
            for B, hi in ((1, 4096), (4, 4096), (8, 2048), (16, 1024), (32, 512), (32, 2048), (32, 4096)):
                add(_lens(seed, B, hi), 1, kind, 16 if B % 8 else 64)
        add(_lens(seed, 16, 2048), 1, "special", 64)
        add(_lens(seed, 32, 1024), 1, "special", 16)
        for splits in (2, 4, 8):
            for kind in ("unit", "wide"):
                add(_lens(seed, 8, 4096), splits, kind, 64)
        return cases
    if name == "bmm-v1":
        for NH, M in ((20, 1), (20, 4), (5, 2), (20, 32), (1, 7)):
            for K, N in ((192, 512), (512, 256)):
                for kind in ("act", "special"):
                    cases.append({"op": "bmm", "NH": NH, "M": M, "K": K, "N": N, "kind": kind, "seed": seed})
                    seed += 1
        return cases
    if name == "combine-v1":
        for T in (1, 7, 32):
            for kind in ("act", "special"):
                cases.append({"op": "combine", "T": T, "HD": 2048, "rsf": 1.8, "kind": kind, "seed": seed})
                seed += 1
        cases.append({"op": "combine", "T": 4, "HD": 2048, "rsf": 2.5, "kind": "act", "seed": seed})
        return cases
    raise KeyError(name)


#: exponent ranges of the GEMM and combine operands: activations around 1, weights around 2^-6
OPERANDS = {"act": {"x": (118, 129), "w": (112, 122)}}


def operand_words(case: dict, stream: int, n: int, tensor: str) -> np.ndarray:
    """The words of a bmm or combine operand: as `words`, with the operand's exponent range; "special" mixes in SPECIALS and
    a few infinities and NaNs (1/64 each class)."""
    idx = np.arange(n, dtype=np.uint64) + (np.uint64(stream) << np.uint64(40)) + (np.uint64(case["seed"]) << np.uint64(52))
    r = _splitmix(idx)
    lo, hi = OPERANDS["act"][tensor]
    e = (lo + (r >> np.uint64(8)) % np.uint64(hi - lo + 1)).astype(np.uint16)
    w = ((r & np.uint64(1)).astype(np.uint16) << 15) | (e << 7) | ((r >> np.uint64(1)) & np.uint64(0x7F)).astype(np.uint16)
    if case["kind"] == "special":
        sp = np.concatenate([SPECIALS, np.array([0x7F80, 0xFF80, 0x7FC1], np.uint16)])
        pick = ((r >> np.uint64(24)) & np.uint64(63)) == 0
        w[pick] = sp[((r >> np.uint64(32)) % np.uint64(len(sp))).astype(np.int64)[pick]]
    return w


def bmm_inputs(case: dict) -> tuple[np.ndarray, np.ndarray]:
    NH, M, K, N = case["NH"], case["M"], case["K"], case["N"]
    return (operand_words(case, 0, NH * M * K, "x").reshape(NH, M, K), operand_words(case, 1, NH * K * N, "w").reshape(NH, K, N))


def combine_inputs(case: dict) -> tuple[np.ndarray, np.ndarray]:
    T, HD = case["T"], case["HD"]
    return operand_words(case, 0, T * HD, "x").reshape(T, HD), operand_words(case, 1, T * HD, "x").reshape(T, HD)


#: op -> its case's inputs, rebuilt from the manifest (`mla_check.check`)
INPUTS = {"decode": case_inputs, "bmm": bmm_inputs, "combine": combine_inputs}


# --- produce (node, torch venv) -----------------------------------------------------------------------------------------

def _bf(a: np.ndarray, device: str):
    import torch
    return torch.from_numpy(a.view(np.int16).copy()).view(torch.bfloat16).to(device)


def _w16(t) -> np.ndarray:
    import torch
    return t.contiguous().cpu().view(torch.int16).numpy().view(np.uint16)


def _run_bmm(case: dict, device: str) -> dict:
    """`torch.bmm` under VLLM_BATCH_INVARIANT: vLLM's `bmm_batch_invariant` (the Triton `bmm_kernel`)."""
    from vllm.model_executor.determinism.batch_invariant import bmm_batch_invariant

    a, b = bmm_inputs(case)
    c = bmm_batch_invariant(_bf(a, device), _bf(b, device))
    return {"c": _w16(c)}


def _run_combine(case: dict, device: str) -> dict:
    """`MoERunner._maybe_apply_routed_scale_to_output` (bf16: ``fused_output *= rsf``) then ``shared_output + fused_output``."""
    routed, shared = combine_inputs(case)
    fused, sh = _bf(routed, device), _bf(shared, device)
    fused *= case["rsf"]
    return {"out": _w16(sh + fused)}


_MUFU_SRC = {"ex2_ftz": "ex2.approx.ftz.f32 $0, $1;", "rcp_ftz": "rcp.approx.ftz.f32 $0, $1;",
             "lg2_ftz": "lg2.approx.ftz.f32 $0, $1;", "ex2": "ex2.approx.f32 $0, $1;", "lg2": "lg2.approx.f32 $0, $1;"}


def _mufu(op: str, x):
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def k(X, Y, n, BLOCK: tl.constexpr, ASM: tl.constexpr):
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        v = tl.load(X + i, mask=i < n)
        y = tl.inline_asm_elementwise(ASM, "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
        tl.store(Y + i, y, mask=i < n)

    y = torch.empty_like(x)
    k[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), BLOCK=1024, ASM=_MUFU_SRC[op])
    return y


def mufu_probe(out: Path, device: str) -> dict:
    """MUFU.EX2 and MUFU.RCP on this GPU against core's sm_89 tables (`verity.ml.circuits.ops.mufu`): ex2 on [0, 1) at every 2^-23 step and
    rcp on every mantissa of [1, 2), plus ex2 / lg2 (ftz and not) on a sweep of 2^20 words; the words are stored."""
    import torch

    from verity.ml.circuits.ops import mufu as MU

    rep = {}
    j = np.arange(1 << 23, dtype=np.uint32)
    ex2_in = (j.astype(np.float64) / (1 << 23)).astype(np.float32)
    rcp_in = (np.uint32(0x3F800000) | j).view(np.float32)
    for op, xs, name in (("ex2_ftz", ex2_in, "ex2"), ("rcp_ftz", rcp_in, "rcp")):
        y = _mufu(op, torch.from_numpy(xs.copy()).to(device)).cpu().numpy().view(np.uint32)
        ref = MU.table(name)
        bad = np.flatnonzero(y != ref)
        rep[op] = {"words": int(y.size), "mismatches": int(bad.size), "first": [[int(i), int(y[i]), int(ref[i])] for i in bad[:8]]}
        if bad.size:
            np.savez_compressed(out / f"mufu_{op}.npz", y=y)
    sweep = _splitmix(np.arange(1 << 20, dtype=np.uint64) + (np.uint64(0xABC) << np.uint64(40))).astype(np.uint32)
    np.savez_compressed(out / "mufu_sweep_in.npz", x=sweep)
    for op in ("ex2", "lg2", "lg2_ftz", "ex2_ftz"):
        y = _mufu(op, torch.from_numpy(sweep.view(np.float32).copy()).to(device)).cpu().numpy().view(np.uint32)
        np.savez_compressed(out / f"mufu_sweep_{op}.npz", y=y)
    lg2_in = (np.uint32(0x3F800000) | j).view(np.float32)
    y = _mufu("lg2_ftz", torch.from_numpy(lg2_in.copy()).to(device)).cpu().numpy().view(np.uint32)
    np.savez_compressed(out / "mufu_lg2_ftz_mantissas.npz", y=y)
    rep["lg2_ftz_mantissas"] = {"words": int(y.size)}
    return rep


def _run_case(case: dict, device: str):
    import torch
    from vllm.v1.attention.ops.triton_decode_attention import decode_attention_fwd

    q_w, kv_w = case_inputs(case)
    B, page, S = len(case["seqlens"]), int(case["page"]), int(case["splits"])
    need = [-(-T // page) for T in case["seqlens"]]
    npages = sum(need) + 3
    order = np.argsort(_splitmix(np.arange(npages, dtype=np.uint64) + (np.uint64(case["seed"]) << np.uint64(52))
                                 + (np.uint64(0xFFFE) << np.uint64(40))), kind="stable")
    cache = np.zeros((npages, page, D), np.uint16)
    table = np.zeros((B, max(need)), np.int32)
    nxt = 0
    for b, T in enumerate(case["seqlens"]):
        for j in range(need[b]):
            p = int(order[nxt]); nxt += 1
            table[b, j] = p
            rows = kv_w[b][j * page:(j + 1) * page]
            cache[p, :rows.shape[0]] = rows
    q = _bf(q_w, device)
    kv = _bf(cache, device).unsqueeze(2)
    kv_c = kv[..., :DL]
    o = torch.zeros(B, H, DL, dtype=torch.bfloat16, device=device)
    lse = torch.zeros(B, H, dtype=torch.bfloat16, device=device)
    logits = torch.empty((B, H, S, DL + 1), dtype=torch.float32, device=device)
    one = torch.ones((), dtype=torch.float32, device=device)
    decode_attention_fwd(q, kv, kv_c, o, lse, torch.from_numpy(table).to(device),
                         torch.tensor(case["seqlens"], dtype=torch.int32, device=device), logits, S, SCALE, page,
                         k_scale=one, v_scale=one, is_mla=True)
    torch.cuda.synchronize()
    return {"o": _w16(o), "lse": _w16(lse), "logits": logits.contiguous().cpu().view(torch.int32).numpy().view(np.uint32)}


def _sass(cache_dir: Path) -> list[str]:
    import shutil
    import subprocess

    import triton

    tool = Path(triton.__file__).parent / "backends" / "nvidia" / "bin" / "cuobjdump"
    exe = str(tool) if tool.exists() else shutil.which("cuobjdump")
    done = []
    for cubin in sorted(cache_dir.rglob("*.cubin")):
        if exe is None:
            break
        out = subprocess.run([exe, "-sass", str(cubin)], capture_output=True, text=True)
        cubin.with_suffix(".sass").write_text(out.stdout + out.stderr)
        done.append(str(cubin.relative_to(cache_dir)))
    return done


RUN = {"decode": _run_case, "bmm": _run_bmm, "combine": _run_combine}


def produce(out: Path, suite_name: str, device: str = "cuda", triton_cache: Path | None = None) -> dict:
    import torch
    import triton
    import vllm

    out.mkdir(parents=True, exist_ok=True)
    man = {"schema": "verity-vllm/mla-capture/v1", "suite": suite_name, "torch": torch.__version__, "triton": triton.__version__,
           "vllm": vllm.__version__, "device": torch.cuda.get_device_name(0), "cc": list(torch.cuda.get_device_capability(0)),
           "scale": SCALE, "H": H}
    if suite_name == "mufu":
        man["mufu"] = mufu_probe(out, device)
        cases = []
    else:
        cases = suite(suite_name)
    man["cases"] = cases
    for i, case in enumerate(cases):
        np.savez_compressed(out / f"case_{i}.npz", **RUN[case.get("op", "decode")](case, device))
        print(f"[mla-capture] case {i}: {json.dumps(case)[:160]}", file=sys.stderr, flush=True)
    if triton_cache is not None:
        man["sass"] = _sass(triton_cache)
    (out / "manifest.json").write_text(json.dumps(man, indent=1) + "\n")
    return man


def _bits_groups(d: Path, man: dict, tmin: int | None, tmax: int | None) -> dict:
    """{key: (cost, run)} of a capture directory: `run()` evaluates one Boolean Definition on bits over its instances and
    returns (got, want, where), word arrays `[rows, words]` and `where(row)` naming a row. A decode key is one T (its one-split
    rows, one head of one request each, with tmin <= T <= tmax); a bmm key one case (a row is one head of one token, an
    instance one token); a combine key one case (a row and an instance are one token)."""
    from collections import defaultdict

    from verity.ml.circuits.evaluation import bits as BS
    from verity.ml.circuits.defs import bind
    from verity_vllm.program.registry import mla as M

    G = M.GLM47_FLASH
    groups: dict = {}
    by_t: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for i, case in enumerate(man["cases"]):
        op = case.get("op", "decode")
        if op == "decode" and case["splits"] == 1:
            for b, T in enumerate(case["seqlens"]):
                if (tmin is None or T >= tmin) and (tmax is None or T <= tmax):
                    by_t[T].append((i, b))
        elif op == "bmm":
            def run(i=i, case=case):
                x, w = bmm_inputs(case)
                NH, Mt, K, N = x.shape[0], x.shape[1], x.shape[2], w.shape[2]
                f = bind(M.MlaBmm, NHEADS=NH, K=K, N=N, DOT=G["DOT"])
                xs = x.transpose(1, 0, 2)
                ws = np.broadcast_to(w.transpose(0, 2, 1), (Mt, NH, N, K))
                got = np.asarray(BS.evaluate_words(f, xs, ws)).reshape(Mt * NH, N)
                want = np.load(d / f"case_{i}.npz")["c"].transpose(1, 0, 2).reshape(Mt * NH, N)
                return got, want, lambda r: {"case": i, "token": r // NH, "head": r % NH}
            groups[f"case={i}"] = (case["NH"] * case["K"] * case["N"], run)
        elif op == "combine":
            def run(i=i, case=case):
                routed, shared = combine_inputs(case)
                f = bind(M.MoeSharedCombine, H=routed.shape[1], RSF=int(np.float32(case["rsf"]).view(np.uint32)))
                got = np.asarray(BS.evaluate_words(f, routed, shared))
                return got, np.load(d / f"case_{i}.npz")["out"], lambda r: {"case": i, "token": r}
            groups[f"case={i}"] = (case["HD"], run)
    for T, rows in by_t.items():
        def run(T=T, rows=rows):
            qs, kvs, want = [], [], []
            for i, b in rows:
                case = man["cases"][i]
                seed, kind = int(case["seed"]), case["kind"]
                qs.append(words(seed, 0, len(case["seqlens"]) * H * D, kind, "q").reshape(-1, H, D)[b])
                kvs.append(np.broadcast_to(words(seed, 1 + b, T * D, kind, "kv").reshape(T, D), (H, T, D)))
                want.append(np.load(d / f"case_{i}.npz")["o"][b])
            head = bind(M.TritonMlaDecodeHead, T=T, DL=G["DL"], DR=G["DR"], BN=G["BN"], SPLITS=1, SCALE=G["SCALE"],
                        DOT=G["DOT"])
            got = np.asarray(BS.evaluate_words(head, np.concatenate(qs), np.concatenate(kvs)))
            where = lambda r: {"case": rows[r // H][0], "request": rows[r // H][1], "T": T, "head": r % H}  # noqa: E731
            return got, np.concatenate(want), where
        groups[f"T={T}"] = (T + 64, run)
    return groups


def done_keys(reports: list[Path]) -> set[str]:
    """The keys earlier `check-bits` reports finished (`by_key`, or a decode-only report's `by_T`)."""
    keys: set[str] = set()
    for p in reports:
        r = json.loads(p.read_text())
        keys |= set(r.get("by_key", {})) | {f"T={t}" for t in r.get("by_T", {})}
    return keys


def _check_key(d: Path, key: str, tmin: int | None, tmax: int | None) -> tuple[int, int, int, dict | None]:
    """One key of `check_bits`: (rows, mismatched rows, differing words, the first differing word or None)."""
    man = json.loads((d / "manifest.json").read_text())
    got, want, where = _bits_groups(d, man, tmin, tmax)[key][1]()
    got, want = got.astype(np.uint16), want.astype(np.uint16)
    bad = got != want
    first = None
    if bad.any():
        r, c = (int(v) for v in np.argwhere(bad)[0])
        first = {**where(r), "column": c, "got": int(got[r, c]), "want": int(want[r, c])}
    return bad.shape[0], int(bad.any(axis=1).sum()), int(bad.sum()), first


def check_bits(d: Path, tmax: int | None = None, worker: int = 0, workers: int = 1, out: Path | None = None,
               tmin: int | None = None, skip: set[str] = frozenset(), fresh: bool = False) -> dict:
    """The Boolean Definitions on bits (`verity.ml.circuits.evaluation.bits`) at GLM-4.7-Flash's statics against the captured words:
    `mla.TritonMlaDecodeHead` against `o` of every one-split decode row with tmin <= T <= tmax (one evaluation per distinct
    T), `mla.MlaBmm` against `c` and `mla.MoeSharedCombine` against `out` (one evaluation per case). The keys not in `skip`
    are dealt to `workers` shares by cost (largest first) and this call takes share `worker`; each finished key is written
    to `out` at once, so a stopped worker keeps what it checked and a rerun skips it (`done_keys`). With `fresh` each key
    runs in a new process: a worker's memory is then one key's, where in one process it grows with every T it has bound."""
    import time
    from concurrent.futures import ProcessPoolExecutor

    man = json.loads((d / "manifest.json").read_text())
    groups = {k: g for k, g in _bits_groups(d, man, tmin, tmax).items() if k not in skip}
    load, mine = [0] * workers, []
    for key in sorted(groups, key=lambda k: (-groups[k][0], k)):
        w = load.index(min(load))
        load[w] += groups[key][0]
        if w == worker:
            mine.append(key)
    rep: dict = {"dir": str(d), "suite": man["suite"], "worker": worker, "workers": workers, "tmin": tmin, "tmax": tmax,
                 "rows": 0, "mismatched_rows": 0, "words_bad": 0, "first_mismatch": None, "by_key": {}}
    for key in sorted(mine, key=lambda k: (groups[k][0], k)):
        t0 = time.time()
        if fresh:
            with ProcessPoolExecutor(1, max_tasks_per_child=1) as pool:
                n, nb, nw, first = pool.submit(_check_key, d, key, tmin, tmax).result()
        else:
            n, nb, nw, first = _check_key(d, key, tmin, tmax)
        if nb and rep["first_mismatch"] is None:
            rep["first_mismatch"] = first
        rep["rows"] += n
        rep["mismatched_rows"] += nb
        rep["words_bad"] += nw
        rep["by_key"][key] = {"rows": n, "mismatched_rows": nb, "seconds": round(time.time() - t0, 1)}
        print(f"{key} rows={n} bad_rows={nb} {time.time() - t0:.1f}s", flush=True)
        if out:
            out.write_text(json.dumps(rep, indent=1) + "\n")
    return rep


@dataclass(frozen=True, kw_only=True)
class ProduceOptions:
    """Capture a suite's cases on the GPU (node, torch venv)."""

    out: str = option("--out")
    suite: str = option("--suite", default="decode-v1")
    device: str = option("--device", default="cuda")
    triton_cache: str | None = option("--triton-cache", env="TRITON_CACHE_DIR", default=None,
                                      help="the Triton cache the run compiles into, whose cubins' SASS the manifest records")


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    """The numpy models (`mla_check`) against every captured word."""

    dir: str = option("--dir")
    out: str | None = option("--out", default=None)


@dataclass(frozen=True, kw_only=True)
class CheckBitsOptions:
    """The Boolean Definitions on bits against the captures."""

    dir: str = option("--dir")
    out: str | None = option("--out", default=None)
    tmin: int | None = option("--tmin", type=int, default=None)
    tmax: int | None = option("--tmax", type=int, default=None)
    worker: int = option("--worker", type=int, default=0)
    workers: int = option("--workers", type=int, default=1)
    skip: list[str] = option("--skip", nargs="*", default=[], help="earlier check-bits reports: their finished keys are not rerun")
    fresh: bool = option("--fresh", action="store_true", default=False,
                         help="each key in a new process (a worker's memory is one key's)")


COMMANDS = {"produce": ProduceOptions, "check": CheckOptions, "check-bits": CheckBitsOptions}


def main(a: ProduceOptions | CheckOptions | CheckBitsOptions) -> int:
    if isinstance(a, ProduceOptions):
        produce(Path(a.out), a.suite, a.device, Path(a.triton_cache) if a.triton_cache else None)
        return 0
    if isinstance(a, CheckBitsOptions):
        rep = check_bits(Path(a.dir), a.tmax, a.worker, a.workers, Path(a.out) if a.out else None, a.tmin,
                         done_keys([Path(p) for p in a.skip]), a.fresh)
        print(json.dumps({k: rep[k] for k in ("rows", "mismatched_rows", "words_bad", "first_mismatch")}), file=sys.stderr)
        return 0 if rep["mismatched_rows"] == 0 else 1
    from verity_vllm.program.registry import mla_check

    rep = mla_check.check(Path(a.dir), INPUTS)
    if a.out:
        Path(a.out).write_text(json.dumps(rep, indent=1) + "\n")
    print(json.dumps(rep["summary"]), file=sys.stderr)
    return 0 if rep["summary"]["mismatched_rows"] == 0 else 1
