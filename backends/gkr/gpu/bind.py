"""Operand binding ("A-GKR, operands bound"): the frozen operands as public inputs of the statement, and the linear
constraints tying the committed operand columns of every unit to them (PROTOCOL.md §16).

Statement files (beside the four Params-only files and ``public.bin``):

  bind.txt       Params-only, one line per bound unit column:
                     bind <col> <arr>:<stride>:<off>:<coef> [<arr>:<stride>:<off>:<coef> ...]
                 meaning ``X[(v,i), col] = Σ coef · ARR[v, stride·i + off] (mod p)`` for every VU ``v`` and step ``i``,
                 ``ARR`` one of ``x`` / ``w`` (the VU's operand words, VU-major)
  instances.txt  which frozen set: ``relation R+bound``, ``lo L``, ``vus N``, ``x <u8|u16> <words per VU>``, ``w ...``,
                 ``y <u16|u32>`` (the width the public words are hashed at)
  x.bin, w.bin   the operand words, little-endian, ``vus`` rows of ``words per VU`` (the vu-k1536 tier's are its own
                 ``vu-k1536.x.u16`` / ``.w.u16`` files byte for byte)

A bound statement is claimed as ``R+bound`` (``verify --relation R+bound``) for a relation ``R`` of
``verifier/pins.txt``: its circuit files are a pinned set of ``R``; ``R`` itself never names a bound statement.
The verifier reads ``x.bin`` / ``w.bin`` itself, evaluates the bound values, and checks the SHA-256 of both files and of
the public words against a pinned table (``verifier/src/instances.rs``).  The transcript absorbs SHA-256 of bind.txt,
instances.txt, x.bin, w.bin after chain.txt's; after the chain challenge ``geo`` it draws ``rho_bind, geo_bind`` and the
functional gains ``rho_bind · geo_bind^(u·nb + j)`` on unit ``u``'s bound column ``j`` (``b`` gains the same times the
value).  The prover never reads the values: the coefficients are statement-independent powers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARRAYS = ("x", "w")
DTYPES = {"u8": "<u1", "u16": "<u2", "u32": "<u4"}
NVF4_K, NVF4_G, NVF4_BYTES = 64, 4, 68


@dataclass
class Spec:
    cols: list[int]
    names: list[str]
    terms: list[list[tuple[str, int, int, int]]]
    text: str


def derive(circ, man: dict) -> list[tuple[str, list[tuple[str, int, int, int]]]]:
    """The bound columns of an export's unit circuit: checker v2 (BF16 / E4M3: ``k`` operand pairs per unit, columns
    ``5 .. 5+2k``) or fp4-nvf4 (``x.p{i} = 16·a + b`` per E2M1 code pair, ``x.s{g} = 128·a + b`` per UE4M3 scale pair)."""
    names = circ.col_names
    if man.get("relation") == "fp4-nvf4":
        W = NVF4_BYTES
        out = [(f"x.p{i}", [("x", W, i, 16), ("w", W, i, 1)]) for i in range(NVF4_K)]
        out += [(f"x.s{g}", [("x", W, NVF4_K + g, 128), ("w", W, NVF4_K + g, 1)]) for g in range(NVF4_G)]
    else:
        if names[:5] != ["one", "x.s", "x.e", "x.M", "x.z"] or "y.s" not in names:
            raise ValueError("bind: not a checker-v2 unit column layout")
        k = (names.index("y.s") - 5) // 2
        out = [(names[5 + j], [("x", k, j, 1)]) for j in range(k)]
        out += [(names[5 + k + j], [("w", k, j, 1)]) for j in range(k)]
    for n, _ in out:
        circ.col_index(n)
    return out


def spec_text(lines) -> str:
    return "".join(f"bind {n} " + " ".join(f"{a}:{s}:{o}:{c}" for a, s, o, c in ts) + "\n" for n, ts in lines)


def read_spec(path: Path, circ) -> Spec:
    text = path.read_text()
    cols, names, terms = [], [], []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] != "bind" or len(t) < 3:
            raise ValueError(f"bind.txt: unknown line {line!r}")
        ts = []
        for tok in t[2:]:
            a, s, o, c = tok.split(":")
            if a not in ARRAYS or int(s) <= 0 or int(o) < 0:
                raise ValueError(f"bind.txt: bad term {tok!r}")
            ts.append((a, int(s), int(o), int(c)))
        cols.append(circ.col_index(t[1]))
        names.append(t[1])
        terms.append(ts)
    if len(set(cols)) != len(cols) or not cols:
        raise ValueError("bind.txt: the bound columns must be distinct and non-empty")
    return Spec(cols, names, terms, text)


def _words(a: np.ndarray, dt: str) -> np.ndarray:
    return np.ascontiguousarray(a.reshape(a.shape[0], -1)).astype(DTYPES[dt], copy=False)


def write_instances(out: Path, relation: str, lo: int, x: np.ndarray, w: np.ndarray, y: np.ndarray,
                    xdt: str, ydt: str) -> dict:
    """``instances.txt``, ``x.bin``, ``w.bin`` into the statement directory ``out``; the digests."""
    xs, ws = _words(x, xdt), _words(w, xdt)
    if xs.shape != ws.shape:
        raise ValueError("bind: x and w differ in shape")
    (out / "x.bin").write_bytes(xs.tobytes())
    (out / "w.bin").write_bytes(ws.tobytes())
    ys = np.ascontiguousarray(y).reshape(-1).astype(DTYPES[ydt])
    txt = (f"relation {relation}\nlo {lo}\nvus {xs.shape[0]}\nx {xdt} {xs.shape[1]}\nw {xdt} {ws.shape[1]}\ny {ydt}\n")
    (out / "instances.txt").write_text(txt)
    return {"relation": relation + BOUND, "lo": lo, "vus": xs.shape[0], "words_per_vu": xs.shape[1], "word": xdt,
            "x_sha256": hashlib.sha256(xs.tobytes()).hexdigest(), "w_sha256": hashlib.sha256(ws.tobytes()).hexdigest(),
            "y_sha256": hashlib.sha256(ys.tobytes()).hexdigest(), "y_word": ydt,
            "instances_txt_sha256": hashlib.sha256(txt.encode()).hexdigest()}


def read_instances(d: Path) -> tuple[dict, np.ndarray, np.ndarray]:
    meta: dict = {}
    for line in (d / "instances.txt").read_text().splitlines():
        t = line.split()
        if t:
            meta[t[0]] = t[1:]
    vus = int(meta["vus"][0])
    xdt, xn = meta["x"][0], int(meta["x"][1])
    arrs = [np.frombuffer((d / f"{a}.bin").read_bytes(), dtype=DTYPES[xdt]).reshape(vus, xn) for a in ARRAYS]
    return meta, arrs[0], arrs[1]


def digests(d: Path) -> list[bytes]:
    """The four SHA-256 digests the transcript absorbs, in order."""
    return [hashlib.sha256((d / n).read_bytes()).digest() for n in ("bind.txt", "instances.txt", "x.bin", "w.bin")]


def values(spec: Spec, x: np.ndarray, w: np.ndarray, steps: int, P: int) -> np.ndarray:
    """``(vus·steps, nb)`` int64: the bound value of every unit's column ``j``."""
    arrs = {"x": x.reshape(x.shape[0], -1).astype(np.int64), "w": w.reshape(w.shape[0], -1).astype(np.int64)}
    vus, n = arrs["x"].shape
    out = np.zeros((vus * steps, len(spec.cols)), dtype=np.int64)
    for j, ts in enumerate(spec.terms):
        for a, s, o, c in ts:
            if s * (steps - 1) + o >= n:
                raise ValueError(f"bind.txt: {spec.names[j]} reads past the VU's {n} words")
            out[:, j] += (c % P) * arrs[a][:, o: o + s * (steps - 1) + 1: s].reshape(-1)
    return out % P


BOUND = "+bound"


def pin(relation: str, lo: int, vus: int, instances: Path | None, procs: int) -> dict:
    """The pinned-table entry (``verifier/src/instances.rs``) of the frozen set of ``relation`` (a ``pins.txt``
    relation; the entry is named ``relation + BOUND``), from its own generator."""
    import bench_result as br

    src = Path(br.__file__).resolve().parents[2]
    if relation == "fp4-nvf4":
        import torch

        from .nvf4.circuit import MODEL as model
        from .nvf4.witness import public_words

        A, B, Y, _ref, chk = br.load_nvf4(lo, vus, procs)
        x, w, xdt = A.reshape(vus, -1), B.reshape(vus, -1), "u8"
        y, ydt = public_words(torch.from_numpy(Y)).numpy().reshape(-1), "u32"
        source = f"{chk['generator']} seed {chk['seed']} (python -m gpu.bind pin fp4-nvf4)"
    elif relation == br.AMPERE_RELATION:
        if instances is None:
            raise SystemExit(f"gpu.bind pin {relation}: --instances (the {br.TIER} tier's data directory)")
        x, w, y, _ = br.load_frozen(instances, src / "fixtures" / "bench-instances" / "v1" / "manifest.json", br.TIER, lo, vus)
        xdt, ydt, model = "u16", "u16", br.AMPERE_MODEL
        source = "fixtures/bench-instances/v1/manifest.json tier vu-k1536 (.x.u16, .w.u16, .y.u16)"
    else:
        x, w, y, rel = br.load_relation(relation, src, lo, vus, procs)
        xdt, ydt, model = "u16", ("u32" if y.dtype == np.uint32 else "u16"), rel.model.name
        source = f"backends/direct/ligero/relchain.py instances() seed {rel.instance_seed} (python -m gpu.bind pin {relation})"
    h = {a: hashlib.sha256(_words(v, xdt).tobytes()).hexdigest() for a, v in (("x", x), ("w", w))}
    h["y"] = hashlib.sha256(np.ascontiguousarray(y).reshape(-1).astype(DTYPES[ydt]).tobytes()).hexdigest()
    return {"relation": relation + BOUND, "lo": lo, "vus": vus, "model": model, **h, "source": source}


def main(argv: list[str] | None = None) -> int:
    from .circuit import load_circuit

    ap = argparse.ArgumentParser(prog="python -m gpu.bind")
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("spec", help="write DIR/bind.txt from DIR's unit circuit")
    s.add_argument("dir", type=Path)
    pn = sub.add_parser("pin", help="a frozen set's pinned-table entry (verifier/src/instances.rs), as a Rust literal")
    pn.add_argument("relation", help="bf16-ampere | bf16-hopper | fp8-ada | fp8-hopper | fp4-nvf4")
    pn.add_argument("--lo", type=int, default=0)
    pn.add_argument("--vus", type=int, default=4096)
    pn.add_argument("--instances", type=Path, default=None)
    pn.add_argument("--procs", type=int, default=16)
    args = ap.parse_args(argv)
    if args.cmd == "pin":
        e = pin(args.relation, args.lo, args.vus, args.instances, args.procs)
        print(json.dumps(e))
        print(f'    Pin {{\n        relation: "{e["relation"]}", lo: {e["lo"]}, vus: {e["vus"]}, model: "{e["model"]}",\n'
              f'        x: "{e["x"]}",\n        w: "{e["w"]}",\n        y: "{e["y"]}",\n        source: "{e["source"]}",\n    }},')
        return 0
    d = args.dir
    man = json.loads((d / "manifest.json").read_text())
    text = spec_text(derive(load_circuit(d / "circuit.txt"), man))
    (d / "bind.txt").write_text(text)
    print(json.dumps({"bind_sha256": hashlib.sha256(text.encode()).hexdigest(), "columns": text.count("\n")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
