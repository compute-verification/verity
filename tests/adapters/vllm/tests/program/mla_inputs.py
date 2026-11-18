"""The inputs of the Triton MLA decode captures and the check of their words against the Boolean Definitions, for
`tests/program/test_mla.py` and `tests/program/test_sparse_mla_split.py`.

The captures are of vLLM d9105ea80's Triton MLA decode (`TritonMLAImpl.forward_mqa` -> `decode_attention_fwd`, grouped kernel,
`is_mla=True`) at GLM-4.7-Flash's shapes. The inputs are counter-based (`words`: splitmix64 of the case seed and the word
index), so a check rebuilds them from a capture's manifest and only the kernel's outputs are stored: `o` (bf16 words
`[B, H, 512]`), `lse` (bf16 `[B, H]`) and the per-split scratch `attn_logits` (f32 words `[B, H, SPLITS, 513]`).
`check_bits` evaluates the Boolean Definitions themselves: `mla.TritonMlaDecodeHead` on the one-split decode rows,
`mla.MlaBmm` and `mla.MoeSharedCombine` on every case.
"""
from __future__ import annotations

import json
from collections.abc import Collection
from pathlib import Path

import numpy as np

DL, DR = 512, 64
D = DL + DR
H = 20


def _splitmix(x: np.ndarray) -> np.ndarray:
    with np.errstate(over="ignore"):
        z = x + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


#: exponent ranges [lo, hi] of the sampled bf16 words, per kind and tensor
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


def _bits_groups(d: Path, man: dict, tmin: int | None, tmax: int | None) -> dict:
    """{key: (cost, run)} of a capture directory: `run()` evaluates one Boolean Definition on bits over its instances and
    returns (got, want, where), word arrays `[rows, words]` and `where(row)` naming a row. A decode key is one T (its one-split
    rows, one head of one request each, with tmin <= T <= tmax); a bmm key one case (a row is one head of one token, an
    instance one token); a combine key one case (a row and an instance are one token)."""
    from collections import defaultdict

    from verity.circuits.defs import bind
    from verity.circuits.evaluation import bits as BS

    from verity_vllm.definitions.registry import mla as M

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
               tmin: int | None = None, skip: Collection[str] = frozenset(), fresh: bool = False) -> dict:
    """The Boolean Definitions on bits (`verity.circuits.evaluation.bits`) at GLM-4.7-Flash's statics against the captured words:
    `mla.TritonMlaDecodeHead` against `o` of every one-split decode row with tmin <= T <= tmax (one evaluation per distinct
    T), `mla.MlaBmm` against `c` and `mla.MoeSharedCombine` against `out` (one evaluation per case). The keys not in `skip`
    are dealt to `workers` shares by cost (largest first) and this call takes share `worker`; each finished key is written
    to `out` at once, so a stopped worker keeps what it checked and a rerun skips it (`done_keys`). With `fresh` each key
    runs in a new process: a worker's memory is then one key's, where in one process it grows with every T it has bound."""
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
        rep["by_key"][key] = {"rows": n, "mismatched_rows": nb}
        print(f"{key} rows={n} bad_rows={nb}", flush=True)
        if out:
            out.write_text(json.dumps(rep, indent=1) + "\n")
    return rep
