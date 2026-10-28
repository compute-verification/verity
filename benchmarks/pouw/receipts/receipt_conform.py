"""Conformance to PoUW's receipt folds: the published vectors (`vectors.json`, made from receipt_folds.py by `--write`) and
the checks an implementation calls with its own functions. Each check raises AssertionError naming the first vector that
differs.

    check_steps({"xin13": fn, ...})        a fold's step, fn(y, d0, d1, d2, d3, *keys) as receipt_folds' (duo: (y, z) in, out)
    check_key_rules({"key_ok_r9": fn, ...})
    check_header(path)                     a C++ header defining receipt_folds.cuh's rf_ functions, built with g++ (or nvcc)
    p2ff_r9(mod), pfill_ref(mod), egemm_ref(mod), e2e_segment(mod), e2e_definitions(mod), scorer_folds(FOLDS, RECEIPT_TAGS)
                                           the existing implementations, each against the vectors and the schedule it runs
                                           (e2e_definitions evaluates E2E's Boolean Definitions on bits)
    pfill_cuh(path)                        pfill.cuh itself, through its host build, in each fold build

    python receipt_conform.py                       check receipt_folds.py and receipt_folds.cuh against vectors.json
    python receipt_conform.py --write               rewrite vectors.json from receipt_folds.py
    python receipt_conform.py --adopter NAME PATH   check an implementation's module (p2ff_r9, pfill_ref, egemm_ref,
                                                    e2e_segment, e2e_definitions), scoring.py (scorer) or pfill.cuh
                                                    (pfill_cuh) at PATH
"""
from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
VECTORS = HERE / "vectors.json"
_spec = importlib.util.spec_from_file_location("receipt_folds", HERE / "receipt_folds.py")
F = sys.modules.setdefault("receipt_folds", importlib.util.module_from_spec(_spec))
if not hasattr(F, "FOLDS"):
    _spec.loader.exec_module(F)

M32 = F.M32
SCHEMA = "pouw-receipts/v1"
#: the round's keys of the XOR-in-13 drop-in check (pfill's default a0, a1; SHA-256's first IV words as k0, k1)
R5_KEYS = {"a0": 0x9E3779B9, "a1": 0x7F4A7C15, "k0": 0x6A09E667, "k1": 0xBB67AE85}
R5_WORDS = [(1, 2, 3, 4), (-1, -2, -3, -4), (100, -100, 2048, -2048)]
EDGE_WORDS = [(0, 0, 0, 0), (M32, M32, M32, M32), (0x80000000, 1, 0x80000000, 1), (1, 0x80000000, 1, 0x80000000)]
#: the tile rule the whole-tile checks use: T_s[r][c] = mix32(mix32(TILE_TAG + s) ^ (128 r + c))
TILE_TAG = 0x7E570000


def h(x: int) -> str:
    return f"{x & M32:08x}"


def u(s: str) -> int:
    return int(s, 16)


def stream(tag: int, n: int, start: int = 0) -> list[int]:
    """The vectors' public inputs: mix32(mix32(tag) ^ i) for i = start .. start + n - 1."""
    k = F.mix32(tag)
    return [F.mix32(k ^ i) for i in range(start, start + n)]


def _first(tag: int, rule, n: int) -> list[int]:
    out, i = [], 0
    while len(out) < n:
        a = stream(tag, 1, i)[0]
        i += 1
        if rule(a):
            out.append(a)
    return out


def _state(fold: str, s) -> list[str]:
    return [h(x) for x in s] if fold == "duo" else [h(s)]


def _steps() -> list[dict]:
    a0, a1 = _first(0x5EED0001, F.key_ok_r11, 2)
    k0, k1 = stream(0x5EED0002, 2)
    long_keys = {"a0": a0, "a1": a1, "k0": k0, "k1": k1}
    cases = [(f"r5-y{y0}", R5_KEYS, y0, [tuple(w & M32 for w in d) for d in R5_WORDS]) for y0 in (0, 5, 1023)]
    cases.append(("edge", long_keys, M32, EDGE_WORDS))
    lw = stream(0x5EED0003, 4 * 32)
    cases.append(("long", long_keys, stream(0x5EED0004, 1)[0], [tuple(lw[4 * i:4 * i + 4]) for i in range(32)]))
    out = []
    for name, keys, y0, words in cases:
        for fold in F.FOLDS:
            start = F.pfill_start(y0) if fold == "duo" else y0
            out.append({"name": name, "fold": fold, "keys": {k: h(v) for k, v in keys.items()},
                        "start": _state(fold, start), "words": [[h(w) for w in d] for d in words],
                        "chain": [_state(fold, s) for s in F.chain(fold, start, words, keys)]})
    return out


def _duo_words() -> list[dict]:
    w = stream(0x5EED0005, 5 * 6)
    out = []
    for i, r in enumerate((1, 5, 16, 29, 31)):
        y, z, d, k, a = w[5 * i:5 * i + 5]
        a |= 1
        out.append({"y": h(y), "z": h(z), "d": h(d), "k": h(k), "a": h(a), "r": r, "out": [h(x) for x in F.duo_word(y, z, d, k, a, r)]})
    return out


def _kj19_only(j: int, sign: int) -> int:
    """The first stream word with bits [19 - j, 30 - j] all `sign` that passes key_ok_r9: a key KJ19 alone rejects."""
    mask = ((1 << 12) - 1) << (19 - j)
    for a in stream(0x5EED0006 + j, 1 << 16):
        a = (a & ~mask) | (mask if sign else 0)
        if F.key_ok_r9(a):
            return a
    raise AssertionError(f"no key_ok_r9 word with bits {19 - j}..{30 - j} all {sign}")


def _key_cases() -> list[tuple[int, str]]:
    xa0, xa1 = F.e2e_xin_keys()
    cases = [(F.PFILL_KEY_DEFAULTS[0], "pfill's default a0"), (F.PFILL_KEY_DEFAULTS[1], "pfill's default a1"),
             (0x2AAAAAAB, "3 a = 1 mod 2^30: an int8 ratio"), (1, "1"), (M32, "-1"), (3, "3"), (0x80000001, "1 + 2^31"),
             (0x20000001, "±1 + 2^29 (Part IX.2.1's j = 2 family)"), (0x10000001, "±1 + 2^28 (j = 3)"),
             (0x9E3779B8, "even")]
    cases += [(0x5A0000FD, "m = 253: under key_ok's low bound"), (0x5A000103, "m = 259: the least m over it with r allowed"),
              (0x5AFFFEFD, "m = 2^24 - 259: the greatest m under key_ok's high bound with r allowed"),
              (0x5AFFFF03, "m = 2^24 - 253: over it"),
              (0x5A123401, "r = 1"), (0x5A123403, "r = 3"), (0x5A12341D, "r = 29"), (0x5A12341F, "r = 31"),
              (0x4000007F, "a = 127 mod 2^30: in lambda8's window at b = 1"), (0x40000081, "a = 129 mod 2^30: out of it at b = 1"),
              (0x5A7FFFFF, "a = -1 mod 2^23: 128 a = -128 mod 2^30, in lambda8's window at b = 128")]
    cases += [(_kj19_only(j, s), f"KJ19 alone rejects it (bits {19 - j}..{30 - j} all {s})") for j in (0, 7, 14) for s in (0, 1)]
    for seed in (0, 0x10483):
        for rule in ("key_ok_r9", "key_ok_r11"):
            a0, a1, _ = F.egemm_xin_keys(seed, F.KEY_RULES[rule])
            cases += [(a0, f"EGEMM a0 at seed {seed:#x} under {rule}"), (a1, f"EGEMM a1 at seed {seed:#x} under {rule}")]
    cases += [(a, f"E2E XA{c}[{t}]") for c, xs in ((0, xa0), (1, xa1)) for t, a in enumerate(xs)]
    cases += [(a, "stream") for a in stream(0x5EED0007, 32)]
    return cases


def _served_case(seed: int, tile: int, ctid: int, w: int, nw: int, segs: list[list[int]]) -> dict:
    y, z = F.served_start(seed, tile, ctid, w, nw)
    out = {"seed": h(seed), "tile": tile, "ctid": ctid, "w": w, "nw": nw, "start": {"y": [h(v) for v in y], "z": [h(v) for v in z]},
           "segments": [[h(v) for v in p] for p in segs], "after": []}
    for p in segs:
        y, z = F.served_duo_absorb(y, z, p)
        out["after"].append({"y": [h(v) for v in y], "z": [h(v) for v in z]})
    fin = [F.duo_final(a, b) for a, b in zip(y, z)]
    out["final"] = [h(v) for v in fin]
    wd = 0
    for v in fin:
        wd ^= v
    out["word"] = h(wd)
    return out


def _egemm_case(seed: int, tile: int, ctid: int, segs: list[list[int]]) -> dict:
    a0, a1, skipped = F.egemm_xin_keys(seed)
    y = F.egemm_xin13_start(tile, ctid)
    out = {"seed": h(seed), "tile": tile, "ctid": ctid, "keys": [h(a0), h(a1)], "skipped": skipped,
           "keys_r11": [h(v) for v in F.egemm_xin_keys(seed, F.key_ok_r11)[:2]], "start": [h(v) for v in y],
           "segments": [[h(v) for v in p] for p in segs], "after": []}
    for p in segs:
        y = F.egemm_xin13_absorb(y, p, a0, a1)
        out["after"].append([h(v) for v in y])
    return out


def _e2e_case(seed: int, tile: int, ctid: int, w: int, nw: int, segs: list[list[int]]) -> dict:
    xa0, xa1 = F.e2e_xin_keys()
    y = F.served_start(seed, tile, ctid, w, nw)[0]
    out = {"seed": h(seed), "tile": tile, "ctid": ctid, "w": w, "nw": nw, "keys": {"a0": [h(v) for v in xa0], "a1": [h(v) for v in xa1]},
           "start": [h(v) for v in y], "segments": [[h(v) for v in p] for p in segs], "after": []}
    for p in segs:
        y = F.e2e_xin_absorb(y, p, xa0, xa1)
        out["after"].append([h(v) for v in y])
    wd = 0
    for v in y:
        wd ^= v
    out["word"] = h(wd)
    return out


def _pfill_cases() -> list[dict]:
    """Lane 17 of warp 3, chain 2: its keys from a key block's rows, its start, and six steps of each fold its builds run
    (an r9 build: duo, xin13 and li under key_ok_r9; else ram and duo under key_ok)."""
    one = [2 * x for x in range(16)]
    one[5] = 0x12345679
    out = []
    for name, rows, r9 in (("hashed", stream(0x5EED0008, 16), False), ("hashed-r9", stream(0x5EED0008, 16), True),
                           ("one-passes", one, False), ("none-pass", [2 * x for x in range(16)], True)):
        keys = dict(zip(("a0", "a1", "k0", "k1"), F.pfill_keys(rows, r9)))
        pos = 4 * (32 * 3 + 17) + 2
        start = F.pfill_start(pos)
        words = [tuple(v) for v in zip(*[iter(stream(0x5EED0009, 4 * 6))] * 4)]
        folds = ("duo", "xin13", "li") if r9 else ("ram", "duo")
        out.append({"name": name, "rows": [h(v) for v in rows], "r9": r9, "keys": {k: h(v) for k, v in keys.items()},
                    "pos": pos, "start": [h(v) for v in start], "words": [[h(w) for w in d] for d in words],
                    "chains": {f: [_state(f, s) for s in F.chain(f, start if f == "duo" else start[0], words, keys)] for f in folds}})
    return out


def tile_words(s: int) -> list[list[int]]:
    """Segment s's 128 x 128 words under TILE_TAG's rule."""
    k = F.mix32(TILE_TAG + s)
    return [[F.mix32(k ^ (128 * r + c)) for c in range(128)] for r in range(128)]


def thread_words(T: list[list[int]], ctid: int) -> list[int]:
    return [T[r][c] for r, c in (F.served_entry(ctid, q) for q in range(32))]


def build() -> dict:
    segs = [stream(0x5EED000A + s, 32) for s in range(2)]
    T = [tile_words(s) for s in range(2)]
    return {
        "schema": SCHEMA,
        "made_by": "receipt_conform.py --write, from receipt_folds.py; words are uint32 as 8 hex digits",
        "steps": _steps(),
        "duo_words": _duo_words(),
        "key_rules": [{"a": h(a), "why": why, **{r: bool(fn(a)) for r, fn in F.KEY_RULES.items()}} for a, why in _key_cases()],
        "schedules": {
            "pfill-lane": _pfill_cases(),
            "served-duo": [_served_case(0x0123ABCD, 3, 37, 0, 1, segs), _served_case(0x0123ABCD, 3, 37, 2, 4, segs)],
            "egemm-xin13": [_egemm_case(seed, 3, 37, segs) for seed in (0, 0x10483)],
            "e2e-xin": [_e2e_case(0x0123ABCD, 3, 37, 2, 4, segs)],
        },
        "served_constants": {"ka": [h(v) for v in F.SERVED_KA], "kk": [h(v) for v in F.SERVED_KK], "rot": list(F.SERVED_ROT)},
        "tile": {"rule": "T_s[r][c] = mix32(mix32(0x7E570000 + s) ^ (128 r + c)), s = 0, 1; seed 0x0123abcd, tile 3",
                 "served-duo": {str(c): _served_case(0x0123ABCD, 3, c, 0, 1, [thread_words(t, c) for t in T])["final"]
                                for c in (0, 1, 37, 511)},
                 "egemm-xin13": {str(c): _egemm_case(0x0123ABCD, 3, c, [thread_words(t, c) for t in T])["after"][-1]
                                 for c in (0, 1, 37, 511)}},
        "key_rates": {"how": "exhaustive over the 2^31 odd 32-bit keys (C, 4 threads)", "odd": 1 << 31, "key_ok": 1878990848,
                      "key_ok_r9": 1878957312, "kj19_fails": 10481280, "key_ok_r9_and_kj19_fails": 8394208,
                      "key_ok_r11": 1870563104},
    }


def load() -> dict:
    v = json.loads(VECTORS.read_text())
    assert v["schema"] == SCHEMA, f"{VECTORS}: schema {v['schema']!r}, not {SCHEMA!r}"
    return v


# ---- checks -------------------------------------------------------------------------------------------------------------

def _keyargs(fold: str, keys: dict) -> list[int]:
    return [u(keys[k]) for k in F.FOLDS[fold]["keys"]]


def check_steps(impl: dict, v: dict | None = None, cast=int) -> int:
    """impl: fold -> step with receipt_folds' signature; `cast` wraps each argument (np.uint32 for numpy steps). Returns
    the steps checked."""
    v = v or load()
    n = 0
    for case in v["steps"]:
        fold = case["fold"]
        if fold not in impl:
            continue
        s = [u(x) for x in case["start"]]
        ka = [cast(x) for x in _keyargs(fold, case["keys"])]
        for i, (d, want) in enumerate(zip(case["words"], case["chain"])):
            d = [cast(u(x)) for x in d]
            got = impl[fold](*[cast(x) for x in s], *d, *ka) if fold == "duo" else impl[fold](cast(s[0]), *d, *ka)
            got = [int(x) & M32 for x in (got if fold == "duo" else (got,))]
            assert got == [u(x) for x in want], f"{fold} {case['name']} step {i}: got {[h(x) for x in got]}, want {want}"
            s, n = got, n + 1
    for case in v["duo_words"] if "duo_word" in impl else ():
        got = [int(x) & M32 for x in impl["duo_word"](*(cast(u(case[k])) for k in "yzdka"), case["r"])]
        assert got == [u(x) for x in case["out"]], f"duo_word {case}: got {[h(x) for x in got]}"
        n += 1
    return n


def check_key_rules(impl: dict, v: dict | None = None) -> int:
    """impl: rule name (receipt_folds.KEY_RULES) -> predicate on a word."""
    v = v or load()
    n = 0
    for case in v["key_rules"]:
        for rule, fn in impl.items():
            assert bool(fn(u(case["a"]))) == case[rule], f"{rule}({case['a']}) ({case['why']}): want {case[rule]}"
            n += 1
    return n


def _header_ops(v: dict) -> tuple[list[str], list[str]]:
    ops, want = [], []
    for case in v["steps"]:
        fold, s = case["fold"], case["start"]
        ka = list(case["keys"][k] for k in F.FOLDS[fold]["keys"])
        for d, nxt in zip(case["words"], case["chain"]):
            ops.append(" ".join([fold, *s, *d, *ka]))
            want.append(" ".join(nxt))
            s = nxt
    for c in v["duo_words"]:
        ops.append(f"duow {c['y']} {c['z']} {c['d']} {c['k']} {c['a']} {c['r']:x}")
        want.append(" ".join(c["out"]))
    for c in v["key_rules"]:
        ops.append(f"keys {c['a']}")
        want.append(" ".join(str(int(c[r])) for r in ("key_ok", "lambda8", "key_ok_r9", "kj19", "key_ok_r11")))
    sc = v["served_constants"]
    for t in range(8):
        ops.append(f"served {t:x}")
        want.append(f"{sc['ka'][t]} {sc['kk'][t]} {sc['rot'][t]}")
    for c in v["schedules"]["served-duo"]:
        for j in range(8):
            idx = (((c["tile"] * F.NCT + c["ctid"]) * c["nw"] + c["w"]) * F.NCH + j) & M32
            ops.append(f"served0 {c['seed']} {idx:x}")
            want.append(f"{c['start']['y'][j]} {c['start']['z'][j]}")
        for a, b, w in zip(c["after"][-1]["y"], c["after"][-1]["z"], c["final"]):
            ops.append(f"final {a} {b}")
            want.append(w)
    for c in v["schedules"]["pfill-lane"]:
        ops.append(f"pfill0 {c['pos']:x}")
        want.append(" ".join(c["start"]))
    for c in v["schedules"]["egemm-xin13"]:
        for i in range(4):
            ops.append(f"egemm0 {c['tile']:x} {c['ctid']:x} {i:x}")
            want.append(c["start"][i])
        cand = [F.mix32((u(c["seed"]) ^ (F.EGEMM_KEY_BASE + i)) & M32) for i in range(8)]
        for i, a in enumerate(cand):
            ops.append(f"egemmk {c['seed']} {i:x}")
            want.append(h(a))
    for i in range(8):
        ops.append(f"e2ek {i:x}")
        want.append(h(F.mix32(F.E2E_KEY_BASE + i) | 1))
    for x in stream(0x5EED000C, 8):
        ops.append(f"mix32 {h(x)}")
        want.append(h(F.mix32(x)))
    return ops, want


def check_header(header: Path | str = HERE / "receipt_folds.cuh", compiler: str = "g++", v: dict | None = None) -> int:
    """Build conform_host.cpp against `header` with `compiler` (g++, clang++ or nvcc) and run every vector through it."""
    v = v or load()
    header = Path(header).resolve()
    with tempfile.TemporaryDirectory() as tmp:
        exe = Path(tmp) / "conform_host"
        cmd = [compiler, "-O2", "-std=c++17", f"-DRF_HEADER=\"{header}\"", f"-I{HERE}", "-o", str(exe)]
        cmd += ["-x", "cu", str(HERE / "conform_host.cpp")] if Path(compiler).name == "nvcc" else [str(HERE / "conform_host.cpp")]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        ops, want = _header_ops(v)
        r = subprocess.run([str(exe)], input="\n".join(ops) + "\n", check=True, capture_output=True, text=True)
    got = r.stdout.splitlines()
    assert len(got) == len(want), f"{header}: {len(got)} answers to {len(want)} operations"
    for op, g, w in zip(ops, got, want):
        assert g == w, f"{header} ({compiler}): {op} gave {g}, want {w}"
    return len(ops)


def check_reference(v: dict | None = None) -> int:
    """receipt_folds.py against the vectors, schedules included."""
    v = v or load()
    n = check_steps({f: d["step"] for f, d in F.FOLDS.items()} | {"duo_word": F.duo_word}, v)
    n += check_key_rules(F.KEY_RULES, v)
    for c in v["schedules"]["pfill-lane"]:
        assert [h(x) for x in F.pfill_keys([u(x) for x in c["rows"]], c["r9"])] == [c["keys"][k] for k in ("a0", "a1", "k0", "k1")], c["name"]
        n += 1
    for c in v["schedules"]["served-duo"]:
        y, z = F.served_start(u(c["seed"]), c["tile"], c["ctid"], c["w"], c["nw"])
        for p, a in zip(c["segments"], c["after"]):
            y, z = F.served_duo_absorb(y, z, [u(x) for x in p])
            assert [h(x) for x in y] == a["y"] and [h(x) for x in z] == a["z"], f"served-duo {c['ctid']}"
            n += 1
    for c in v["schedules"]["egemm-xin13"]:
        a0, a1, sk = F.egemm_xin_keys(u(c["seed"]))
        assert [h(a0), h(a1)] == c["keys"] and sk == c["skipped"], f"egemm keys at {c['seed']}"
        y = F.egemm_xin13_start(c["tile"], c["ctid"])
        for p, a in zip(c["segments"], c["after"]):
            y = F.egemm_xin13_absorb(y, [u(x) for x in p], a0, a1)
            assert [h(x) for x in y] == a, f"egemm-xin13 at {c['seed']}"
            n += 1
    for c in v["schedules"]["e2e-xin"]:
        xa0, xa1 = F.e2e_xin_keys()
        assert [h(x) for x in xa0] == c["keys"]["a0"] and [h(x) for x in xa1] == c["keys"]["a1"], "e2e keys"
        y = F.served_start(u(c["seed"]), c["tile"], c["ctid"], c["w"], c["nw"])[0]
        for p, a in zip(c["segments"], c["after"]):
            y = F.e2e_xin_absorb(y, [u(x) for x in p], xa0, xa1)
            assert [h(x) for x in y] == a, "e2e-xin"
            n += 1
    return n


# ---- the existing implementations ---------------------------------------------------------------------------------------

def p2ff_r9(rf, v: dict | None = None) -> int:
    """The r9 twin (receipt_folds_r9.py): its key rules and its two steps (numpy)."""
    import numpy as np

    v = v or load()
    n = check_key_rules({"key_ok": rf.key_ok, "key_ok_r9": rf.key_ok_r9}, v)
    with np.errstate(over="ignore"):
        n += check_steps({"xin13": rf.xorin_step, "li": rf.lop3imad_step}, v, cast=np.uint32)
    return n


def pfill_ref(pr, v: dict | None = None) -> int:
    """pfill_ref.py (pfill.cuh's twin): its key rules, `keys` and `absorb` at each NC its builds run. An older copy is
    checked on what it has: no `absorb` (steps inline in its `fold`), no r9 rule, XIN (rotl 7) in place of XIN13."""
    import numpy as np

    v = v or load()
    n = check_key_rules({"key_ok": pr.key_ok} | ({"key_ok_r9": pr.key_ok_r9} if hasattr(pr, "key_ok_r9") else {}), v)
    nc = {f: getattr(pr, a) for f, a in (("ram", "RAM"), ("duo", "DUO"), ("xin13", "XIN13"), ("li", "LI"), ("xin7", "XIN"))
          if hasattr(pr, a) and hasattr(pr, "absorb")}

    def at(fold):
        def fn(*a):
            k = list(a[-len(F.FOLDS[fold]["keys"]):])
            k = (k + [np.uint32(0)] * 2) if len(k) == 2 else k
            if fold == "duo":
                return pr.absorb(nc[fold], a[0], a[1], *a[2:6], *k)
            return pr.absorb(nc[fold], a[0], np.uint32(0), *a[1:5], *k)[0]
        return fn
    with np.errstate(over="ignore"):
        n += check_steps({f: at(f) for f in nc}, v, cast=np.uint32)
    r9 = hasattr(pr, "key_ok_r9")
    for c in v["schedules"]["pfill-lane"]:
        if c["r9"] and not r9:
            continue
        rows = np.array([u(x) for x in c["rows"]], np.uint32)
        got = pr.keys(rows, c["r9"]) if r9 else pr.keys(rows)
        assert [h(int(x)) for x in got] == [c["keys"][k] for k in ("a0", "a1", "k0", "k1")], f"pfill keys {c['name']}"
        n += 1
    return n


_PFILL_HARNESS = r"""
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#define __host__
#define __device__
#define __forceinline__ inline
#define __constant__ static const
inline uint64_t smix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ull;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ull;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBull;
  return x ^ (x >> 31);
}
inline uint64_t dseed(uint64_t c0, uint64_t c1, uint64_t tile) { return smix(smix(c0 ^ smix(tile)) ^ c1); }
#include PFILL_CUH
int main() {
  char op[16];
  unsigned w[16];
  while (std::scanf("%15s", op) == 1) {
    if (!std::strcmp(op, "absorb")) {
      int nc;
      if (std::scanf("%d", &nc) != 1) return 2;
      for (int i = 0; i < 10; i++) if (std::scanf("%x", &w[i]) != 1) return 2;
      uint32_t y = w[0], z = w[1];
      const PfKeys k{w[6], w[7], w[8], w[9]};
      if (nc == 2) pf_absorb<2>(y, z, w[2], w[3], w[4], w[5], k);
      else pf_absorb<4>(y, z, w[2], w[3], w[4], w[5], k);
      std::printf("%08x %08x\n", y, z);
    } else if (!std::strcmp(op, "rule")) {
      if (std::scanf("%x", &w[0]) != 1) return 2;
      std::printf("%d\n", (int)(PF_R9 ? pf_key_ok_r9(w[0]) : pf_key_ok(w[0])));
    } else if (!std::strcmp(op, "keys")) {
      for (int i = 0; i < 16; i++) if (std::scanf("%x", &w[i]) != 1) return 2;
      uint64_t st[8];
      for (int i = 0; i < 8; i++) st[i] = (uint64_t)pf_bswap(w[2 * i]) << 32 | pf_bswap(w[2 * i + 1]);
      const PfKeys k = pf_keys(st);
      std::printf("%08x %08x %08x %08x\n", k.a0, k.a1, k.k0, k.k1);
    } else if (!std::strcmp(op, "start")) {
      if (std::scanf("%x", &w[0]) != 1) return 2;
      std::printf("%08x %08x\n", pf_y0(w[0]), pf_z0(w[0]));
    } else {
      return 2;
    }
  }
}
"""


def _pfill_build_ops(v: dict, flag: str | None) -> tuple[list[str], list[str]]:
    builds = F.SCHEDULES["pfill-lane"]["build"]
    folds = [f for f in F.SCHEDULES["pfill-lane"]["folds"] if builds[f][1] == flag or (f == "duo" and flag)]
    r9 = flag is not None
    ops, want = [], []

    def chain(fold, start, words, keys, out):
        nc, s = builds[fold][0], list(start) if fold == "duo" else [start[0], "00000000"]
        for d, nxt in zip(words, out):
            ops.append(" ".join(["absorb", str(nc), *s, *d, *keys]))
            s = list(nxt) if fold == "duo" else [nxt[0], s[1]]
            want.append(" ".join(s))

    for case in v["steps"]:
        if case["fold"] in folds:
            k = case["keys"]
            chain(case["fold"], case["start"], case["words"], [k["a0"], k["a1"], k.get("k0", "0"), k.get("k1", "0")],
                  case["chain"])
    for c in v["key_rules"]:
        ops.append(f"rule {c['a']}")
        want.append(str(int(c["key_ok_r9" if r9 else "key_ok"])))
    for c in v["schedules"]["pfill-lane"]:
        if c["r9"] != r9:
            continue
        ops.append("keys " + " ".join(c["rows"]))
        want.append(" ".join(c["keys"][k] for k in ("a0", "a1", "k0", "k1")))
        ops.append(f"start {c['pos']:x}")
        want.append(" ".join(c["start"]))
        for fold in folds:
            if fold in c["chains"]:
                chain(fold, c["start"], c["words"], [c["keys"][k] for k in ("a0", "a1", "k0", "k1")], c["chains"][fold])
    return ops, want


def pfill_cuh(path: Path | str, compiler: str = "g++", v: dict | None = None) -> int:
    """pfill.cuh itself, built for the host with `compiler` as pfill_host.cpp builds it (dcap.cu's smix and dseed), once a
    fold build: pf_absorb at each NC on the vectors' steps, the build's key rule (pf_key_ok, or pf_key_ok_r9 in an r9
    build), pf_keys on the pfill-lane cases' rows and pf_y0, pf_z0 at their positions, and their chains. Returns the
    operations checked."""
    v = v or load()
    cuh = Path(path).resolve()
    flags = sorted({f for _, f in F.SCHEDULES["pfill-lane"]["build"].values() if f != "-DPFILL_XIN"}, key=str)
    n = 0
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "pfill_conform.cpp"
        src.write_text(_PFILL_HARNESS)
        runs = []
        for flag in [None] + [f for f in flags if f]:
            exe = Path(tmp) / f"pfill_conform{flag or ''}"
            cmd = [compiler, "-O1", "-std=c++17", "-Wno-unknown-pragmas", f"-DPFILL_CUH=\"{cuh}\"", *([flag] if flag else []),
                   "-o", str(exe), str(src)]
            runs.append((flag, exe, subprocess.Popen(cmd, stderr=subprocess.PIPE, text=True)))
        for flag, exe, p in runs:
            err = p.communicate()[1]
            assert p.returncode == 0, f"{cuh} ({flag or 'default'} build) didn't compile:\n{err[-2000:]}"
            ops, want = _pfill_build_ops(v, flag)
            r = subprocess.run([str(exe)], input="\n".join(ops) + "\n", capture_output=True, text=True, check=True)
            got = r.stdout.splitlines()
            assert len(got) == len(want), f"{path} ({flag or 'default'}): {len(got)} answers to {len(want)} operations"
            for op, g, w in zip(ops, got, want):
                assert g == w, f"{path} ({flag or 'default'} build): {op} gave {g}, want {w}"
            n += len(ops)
    return n


def egemm_ref(er, v: dict | None = None) -> int:
    """egemm_ref.py: xin_keys, xin13_receipts and the duo tile_receipts on one whole tile (TILE_TAG's rule), every thread
    against receipt_folds' egemm-xin13 and served-duo schedules."""
    import numpy as np

    v = v or load()
    n = 0
    for c in v["schedules"]["egemm-xin13"]:
        assert [h(x) for x in er.xin_keys(u(c["seed"]), r9=True)[:2]] == c["keys"], f"xin_keys at {c['seed']}"
        n += 1
    seed, tile, T = 0x0123ABCD, 3, [tile_words(s) for s in range(2)]
    Ts = [np.array(t, np.uint32) for t in T]
    x13 = er.xin13_receipts(seed, Ts, tile)
    duo = er.tile_receipts(seed, Ts, tile)
    a0, a1, _ = F.egemm_xin_keys(seed)
    for ctid in range(F.NCT):
        p = [thread_words(t, ctid) for t in T]
        y = F.egemm_xin13_start(tile, ctid)
        yd, zd = F.served_start(seed, tile, ctid)
        for q in p:
            y = F.egemm_xin13_absorb(y, q, a0, a1)
            yd, zd = F.served_duo_absorb(yd, zd, q)
        assert [int(x) for x in x13[ctid, :4]] == y and not x13[ctid, 4:].any(), f"xin13_receipts thread {ctid}"
        assert [int(x) for x in duo[ctid]] == [F.duo_final(a, b) for a, b in zip(yd, zd)], f"tile_receipts thread {ctid}"
        n += 2
    for c, want in v["tile"]["egemm-xin13"].items():
        assert [h(int(x)) for x in x13[int(c), :4]] == want, f"tile vectors, thread {c}"
    return n


def e2e_segment(sg, v: dict | None = None) -> int:
    """E2E's segment.py (PouwSegCheck's word twins): its constants, key rule, init and both folds' absorb and final."""
    v = v or load()
    sc = v["served_constants"]
    assert [h(x) for x in sg.KA] == sc["ka"] and [h(x) for x in sg.KK] == sc["kk"] and list(sg.ROT) == sc["rot"], "duo keys"
    assert tuple(sg.XROT) == F.FOLDS["xin13"]["xor_rotations"], "xin rotations"
    n = 1 + check_key_rules({"lambda8": sg.key_ok}, v)
    for x in stream(0x5EED000C, 8):
        assert sg.mix32(x) == F.mix32(x)
    for c in v["schedules"]["served-duo"]:
        y, z = sg.init(u(c["seed"]), c["tile"], c["ctid"], c["w"], c["nw"])
        assert [h(x) for x in y] == c["start"]["y"] and [h(x) for x in z] == c["start"]["z"], "init"
        st = y + z
        for p, a in zip(c["segments"], c["after"]):
            st = sg.fold_absorb("duo", st, [u(x) for x in p])
            assert [h(x) for x in st] == a["y"] + a["z"], "duo absorb"
            n += 1
        assert h(sg.fold_final("duo", st)) == c["word"], "duo final"
    for c in v["schedules"]["e2e-xin"]:
        assert [h(x) for x in sg.XA0] == c["keys"]["a0"] and [h(x) for x in sg.XA1] == c["keys"]["a1"], "XA0, XA1"
        st = sg.fold_keys("xin", u(c["seed"]), c["tile"], c["ctid"], c["w"], c["nw"])
        assert [h(x) for x in st] == c["start"], "xin start"
        for p, a in zip(c["segments"], c["after"]):
            st = sg.fold_absorb("xin", st, [u(x) for x in p])
            assert [h(x) for x in st] == a, "xin absorb"
            n += 1
        assert h(sg.fold_final("xin", st)) == c["word"], "xin final"
    return n


def e2e_definitions(sg, v: dict | None = None) -> int:
    """E2E's Boolean Definitions themselves (PouwDuoAbsorb_v2, PouwXinAbsorb_v2 and their finals, `segment.FOLDS`),
    evaluated on bits over the served-duo and e2e-xin vectors."""
    import numpy as np
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation.bits import evaluate_bits

    v = v or load()

    def bits(ws: list[int]) -> np.ndarray:
        return np.array([[(w >> i) & 1 for w in ws for i in range(32)]], np.uint8)

    def words(b) -> list[int]:
        b = [int(x) for x in np.asarray(b).reshape(-1)]
        return [sum(x << i for i, x in enumerate(b[32 * j:32 * j + 32])) for j in range(len(b) // 32)]

    n = 0
    for fold, cases in (("duo", v["schedules"]["served-duo"]), ("xin", v["schedules"]["e2e-xin"])):
        absorb, final, _ = sg.FOLDS[fold]
        for c in cases:
            st = [u(x) for x in (c["start"]["y"] + c["start"]["z"] if fold == "duo" else c["start"])]
            for p, a in zip(c["segments"], c["after"]):
                pb = np.array([[[(u(w) >> i) & 1 for i in range(32)] for w in p]], np.uint8)
                st = words(evaluate_bits(bind(absorb), bits(st), pb))
                assert [h(x) for x in st] == (a["y"] + a["z"] if fold == "duo" else a), f"{absorb.id} {c['ctid']}"
                n += 1
            assert h(words(evaluate_bits(bind(final), bits(st)))[0]) == c["word"], f"{final.id}"
            n += 1
    return n


#: SCORER's fold names (scoring.FOLDS) as receipt_folds names them
SCORER_NAMES = {"ram": "ram", "duo": "duo", "xor-in": "xin7", "xor-in-13": "xin13", "lop3-imad": "li"}


def scorer_folds(folds: dict, tags: dict) -> int:
    """filler-scorer's fold table (scoring.FOLDS, RECEIPT_TAGS): each fold's NC and build flag are pfill's, and an
    XOR-family fold's rotations into the XORs are receipt_folds'."""
    build = F.SCHEDULES["pfill-lane"]["build"]
    assert set(folds) == set(SCORER_NAMES), f"scorer folds {sorted(folds)}, reference {sorted(SCORER_NAMES)}"
    for name, f in folds.items():
        ref = SCORER_NAMES[name]
        nc, flag = build[ref]
        assert tuple(f["nc"]) == (nc,) and f["flag"] == flag, f"{name}: nc {f['nc']} flag {f['flag']}, pfill builds {nc} {flag}"
        if "rotations" in f:
            assert tuple(f["rotations"]) == F.FOLDS[ref]["xor_rotations"], f"{name}: rotations {f['rotations']}"
    assert set(tags.values()) <= set(folds), f"receipt tags {tags} name folds outside {sorted(folds)}"
    return len(folds)


def _load(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main(argv: list[str]) -> int:
    if argv[:1] == ["--write"]:
        VECTORS.write_text(json.dumps(build(), indent=1) + "\n")
        print(f"wrote {VECTORS}")
        return 0
    if argv[:1] == ["--adopter"]:
        name, path = argv[1], argv[2]
        if name == "scorer":
            mod = _load(path, "scoring")
            print(f"scorer: {scorer_folds(mod.FOLDS, mod.RECEIPT_TAGS)} folds agree")
        elif name == "pfill_cuh":
            print(f"pfill_cuh: {pfill_cuh(path)} operations agree")
        else:
            mod = _load(path, Path(path).stem)
            print(f"{name}: {globals()[name](mod)} checks agree")
        return 0
    v = load()
    print(f"receipt_folds.py: {check_reference(v)} checks agree")
    for cxx in ("g++", "nvcc"):
        if shutil.which(cxx):
            print(f"receipt_folds.cuh ({cxx}): {check_header(compiler=cxx, v=v)} operations agree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
