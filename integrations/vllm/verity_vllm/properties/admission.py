"""verity_vllm/properties/admission.py — PROTECTED (gate G4, quarantine admission).  Randomized differential tests of a registered
body against the pinned production operation, bit-exact, in two halves so the torch venv and the 3.12 verity venv never meet:

    # on the pod, in the vLLM venv (torch + the production kernel):
    verity-vllm properties-admission produce --adapter verity_vllm.program.registry.quarantine.<lane>.<def>_difftest --n 32 --seed 7 --out DIR
    # on the pod in /workspace/venv-py312 (or on the laptop after copying DIR):
    verity-vllm properties-admission check --adapter <same> --dir DIR --out quarantine.json [--append]

The ADAPTER (lane-written, next to the quarantined definition) is a plain module with three functions and no import of
torch at module level:

    SEMANTIC_ID = "<lane>/<Definition>@v1"                        # the versioned identity being admitted
    def cases(rng: random.Random, i: int) -> dict:                 # statics + input specs for case i; MUST include the
        return {"statics": {...}, "inputs": [{"shape": [...], "dtype": "bfloat16", "kind": "random|zeros|max|subnormal|alternating"}], "edge": bool}
    def production(statics: dict, inputs: list["torch.Tensor"]) -> list["torch.Tensor"]:   # runs the pinned production op (imports torch inside)
    def spec(statics: dict) -> "SpecializedDefinition":            # the Verity specialization under test (torch-free)

`produce` draws the inputs (uint16/uint32 words -> device tensors), runs `production`, and stores every case's input and
output words (`case_<i>/in_<j>.npy`, `out_<k>.npy`, `case.json`).  `check` rebuilds `spec(statics)`, evaluates its body on
the SAME input words through the Verity evaluator (core `verity.ml.circuits.evaluation.evaluate`) and compares the output encodings
word for word.  Gate G4 reads the resulting `quarantine.json`: `differential.passed == differential.total > 0` for every
quarantined definition the run used.  Edge-kind inputs (`zeros`, `max`, `subnormal`, `alternating`) are required: at least
one case per kind; `check` fails the admission otherwise.

Torch-free except inside `produce`.
"""
from __future__ import annotations

import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from dataclasses import dataclass, field

from verity_vllm.config import option

from verity_vllm.commit import challenge as CH
from verity_vllm.properties.record import seal

REQUIRED_EDGE_KINDS = ("zeros", "max", "subnormal", "alternating")
DTYPE_BITS = {"bfloat16": 16, "float16": 16, "float32": 32, "int32": 32, "int64": 64, "uint8": 8, "int8": 8, "bool": 8}


# --- word generation (shared by both halves; deterministic in (seed, case)) --------------------------------------------

def words_for(kind: str, shape: list[int], dtype: str, rng: np.random.Generator) -> np.ndarray:
    n = int(np.prod(shape)) if shape else 1
    if dtype == "bfloat16":
        if kind == "random":
            f = (rng.standard_normal(n) * float(rng.choice([0.05, 1.0, 8.0]))).astype(np.float32)
            return (f.view(np.uint32) >> 16).astype(np.uint16).reshape(shape)
        if kind == "zeros":
            w = np.zeros(n, np.uint16); w[::7] = 0x8000                       # +0 and -0
            return w.reshape(shape)
        if kind == "max":
            w = rng.choice(np.array([0x7F7F, 0xFF7F, 0x4300, 0xC300], np.uint16), n)   # ±max finite, ±128
            return w.astype(np.uint16).reshape(shape)
        if kind == "subnormal":
            w = rng.integers(1, 0x0080, n).astype(np.uint16) | rng.choice(np.array([0, 0x8000], np.uint16), n)
            return w.reshape(shape)
        if kind == "alternating":
            f = (np.arange(n) % 2 * 2 - 1).astype(np.float32) * (1.0 + rng.random(n).astype(np.float32))
            return (f.view(np.uint32) >> 16).astype(np.uint16).reshape(shape)
    if dtype == "float32":
        if kind == "random":
            return (rng.standard_normal(n).astype(np.float32)).view(np.uint32).reshape(shape)
        if kind == "zeros":
            return np.zeros(n, np.uint32).reshape(shape)
        if kind == "max":
            return rng.choice(np.array([0x7F7FFFFF, 0xFF7FFFFF, 0x43000000], np.uint32), n).astype(np.uint32).reshape(shape)
        if kind == "subnormal":
            return rng.integers(1, 0x00800000, n).astype(np.uint32).reshape(shape)
        if kind == "alternating":
            f = (np.arange(n) % 2 * 2 - 1).astype(np.float32) * (1.0 + rng.random(n).astype(np.float32))
            return f.view(np.uint32).reshape(shape)
    if dtype in ("int32", "int64"):
        hi = 2**31 - 1 if dtype == "int32" else 2**62
        v = rng.integers(0, hi if kind != "zeros" else 1, n)
        return v.astype(np.uint32 if dtype == "int32" else np.uint64).reshape(shape)
    raise ValueError(f"no generator for kind={kind} dtype={dtype}")


def _np_dtype(dtype: str):
    return {"bfloat16": np.uint16, "float16": np.uint16, "float32": np.uint32, "int32": np.uint32, "int64": np.uint64,
            "uint8": np.uint8, "int8": np.uint8, "bool": np.uint8}[dtype]


# --- produce (torch venv) -----------------------------------------------------------------------------------------------

def produce(adapter: str, n: int, seed: int, out: Path, device: str = "cuda") -> dict[str, Any]:
    import torch

    mod = importlib.import_module(adapter)
    rng_py = CH.stream(seed)
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {"schema": "verity-gen/difftest-produce/v1", "adapter": adapter, "semantic_id": getattr(mod, "SEMANTIC_ID", None),
                                "seed": seed, "n": n, "torch": torch.__version__, "device": torch.cuda.get_device_name(0) if device.startswith("cuda") else device,
                                "cases": []}
    tmap = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "int32": torch.int32, "int64": torch.int64,
            "uint8": torch.uint8, "int8": torch.int8, "bool": torch.bool}
    for i in range(n):
        case = mod.cases(rng_py, i)
        rng = CH.generator(CH.case_seed(seed, i))
        cdir = out / f"case_{i}"
        cdir.mkdir(exist_ok=True)
        ins = []
        for j, spec in enumerate(case["inputs"]):
            if "words" in spec:                                   # [X-06] explicit words (bound inputs such as seeds / temperatures)
                w = np.asarray(spec["words"], dtype=_np_dtype(spec["dtype"])).reshape(list(spec["shape"]))
            elif "npy" in spec:                                   # [X-06] real captured words (e.g. a model's logits row), kind "real"
                w = np.load(spec["npy"])
                w = (w[int(spec["row"])] if "row" in spec else w).view(_np_dtype(spec["dtype"])).reshape(list(spec["shape"]))
            else:
                w = words_for(spec.get("kind", "random"), list(spec["shape"]), spec["dtype"], rng)
            np.save(cdir / f"in_{j}.npy", w)
            t = torch.from_numpy(w.view(_np_dtype(spec["dtype"])).copy()).view(tmap[spec["dtype"]]).to(device)
            ins.append(t)
        t0 = time.time()
        outs = mod.production(case["statics"], ins)
        torch.cuda.synchronize() if device.startswith("cuda") else None
        rec = {"i": i, "statics": case["statics"], "inputs": case["inputs"], "outputs": [], "seconds": round(time.time() - t0, 4),
               "edge": bool(case.get("edge", any(s.get("kind", "random") != "random" for s in case["inputs"])))}
        for k, o in enumerate(outs):
            dt = str(o.dtype).replace("torch.", "")
            o = o.detach().contiguous().cpu()
            # numpy has no bfloat16: reinterpret the words as int16 before crossing (a bf16 production output is
            # otherwise unrepresentable here -- `Tensor.numpy()` raises "Got unsupported ScalarType BFloat16")
            words = o.view(torch.int16).numpy() if dt == "bfloat16" else o.view(tmap[dt]).numpy()
            words = words.view(_np_dtype(dt)) if dt != "bool" else words.astype(np.uint8)
            np.save(cdir / f"out_{k}.npy", words)
            rec["outputs"].append({"shape": list(o.shape), "dtype": dt})
        (cdir / "case.json").write_text(json.dumps(rec, indent=1, default=str))
        manifest["cases"].append(rec)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str))
    return manifest


# --- check (verity venv) ------------------------------------------------------------------------------------------------

def evaluate_spec(spec, operand_words: list[np.ndarray]) -> list[int]:
    """Encodings returned by one call of `spec` on operands given as word arrays: core `verity.ml.circuits.evaluation.evaluate`."""
    from verity.ml.circuits.evaluation import evaluate

    params, _ = spec.signature()
    if len(params) != len(operand_words):
        raise ValueError(f"{spec.id}: {len(params)} parameters, {len(operand_words)} operands")
    for (pname, pt), w in zip(params, operand_words):
        if pt.leaves != w.size:
            raise ValueError(f"{spec.id}: parameter {pname} has {pt.leaves} leaves, operand has {w.size} words")
    return list(evaluate(spec, *(w.reshape(-1).tolist() for w in operand_words)))


def check(adapter: str, d: Path) -> dict[str, Any]:
    mod = importlib.import_module(adapter)
    man = json.loads((d / "manifest.json").read_text())
    rep: dict[str, Any] = {"schema": "verity-gen/difftest/v1", "adapter": adapter, "semantic_id": getattr(mod, "SEMANTIC_ID", None),
                           "produce": {k: man.get(k) for k in ("seed", "n", "torch", "device")}, "cases": [], "status": "fail"}
    kinds_seen: set[str] = set()
    passed = 0
    for rec in man["cases"]:
        cdir = d / f"case_{rec['i']}"
        ent = {"i": rec["i"], "statics": rec["statics"], "kinds": [s.get("kind", "random") for s in rec["inputs"]]}
        kinds_seen |= set(ent["kinds"])
        try:
            t0 = time.time()
            sp = mod.spec(rec["statics"])
            ins = [np.load(cdir / f"in_{j}.npy") for j in range(len(rec["inputs"]))]
            got = evaluate_spec(sp, ins)
            exp: list[int] = []
            for k in range(len(rec["outputs"])):
                exp += [int(x) for x in np.load(cdir / f"out_{k}.npy").reshape(-1).tolist()]
            if len(got) != len(exp):
                raise ValueError(f"{len(got)} return encodings vs {len(exp)} production words")
            diff = [i for i in range(len(got)) if got[i] != exp[i]]
            ent.update(spec=sp.id, status="pass" if not diff else "fail", n_words=len(exp), n_diff=len(diff), first_diff=diff[:4],
                       seconds=round(time.time() - t0, 3))
            passed += not diff
        except Exception as e:  # noqa: BLE001
            ent.update(status="error", reason=f"{type(e).__name__}: {e}")
        rep["cases"].append(ent)
    missing = [k for k in REQUIRED_EDGE_KINDS if k not in kinds_seen]
    rep["differential"] = {"passed": passed, "total": len(man["cases"])}
    rep["edge_kinds_missing"] = missing
    rep["status"] = "pass" if passed == len(man["cases"]) > 0 and not missing else "fail"
    if missing:
        rep["reason"] = f"edge-kind inputs missing: {missing}"
    return seal(rep, rep["status"] == "pass")


@dataclass(frozen=True, kw_only=True)
class ProduceOptions:
    adapter: str = option("--adapter")
    n: int = option("--n", type=int, default=32)
    seed: int = option("--seed", type=int, default=7)
    out: str = option("--out")
    device: str = option("--device", default="cuda")
    cmd: str = field(default="produce", init=False)


@dataclass(frozen=True, kw_only=True)
class CheckOptions:
    adapter: str = option("--adapter")
    dir: str = option("--dir")
    out: str = option("--out")
    append: bool = option("--append", default=False, action="store_true", help="merge into an existing quarantine.json")
    cmd: str = field(default="check", init=False)


COMMANDS = {"produce": ProduceOptions, "check": CheckOptions}


def main(a: ProduceOptions | CheckOptions) -> int:
    if a.cmd == "produce":
        m = produce(a.adapter, a.n, a.seed, Path(a.out), a.device)
        print(f"[difftest] produced {len(m['cases'])} cases for {m['semantic_id']} -> {a.out}", file=sys.stderr)
        return 0
    rep = check(a.adapter, Path(a.dir))
    out = Path(a.out)
    doc: dict[str, Any] = {"schema": "verity-gen/quarantine/v1", "definitions": {}}
    if a.append and out.exists():
        doc = json.loads(out.read_text())
    doc["definitions"][rep["semantic_id"] or a.adapter] = rep
    tot = sum(v["differential"]["total"] for v in doc["definitions"].values())
    ok = sum(v["differential"]["passed"] for v in doc["definitions"].values())
    doc["differential"] = {"passed": ok, "total": tot}
    doc["status"] = "pass" if all(v["status"] == "pass" for v in doc["definitions"].values()) else "fail"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"[difftest] {rep['status']}: {rep['differential']['passed']}/{rep['differential']['total']} cases bit-exact for {rep['semantic_id']}"
          + (f"; {rep['reason']}" if rep.get("reason") else ""), file=sys.stderr)
    return 0 if rep["status"] == "pass" else 1
