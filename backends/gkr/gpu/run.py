"""Load a VU batch directory (``bb_export``) and prove / verify it on a device; the measurement driver.

    python -m gpu.run prove DIR [--vus N] [--device cuda] [--reps 3] [--warmup 1] [--clear] [--json OUT]
    python -m gpu.run negatives DIR             # every DIR/neg/<id>: prove + verify, all must reject
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch

from . import prover
from .bb_export import read_rows
from .circuit import layers, load_circuit
from .field import P
from .gkr import VerifyError


def read_chain(path: Path, ucirc, ecirc, steps: int, public: list[int]) -> prover.Chain:
    """``chain.txt`` (``main.rs::read_chain``, PROTOCOL.md 14.3): ``link <in|-> <out> <epi> <init> <zero_nonlast>``."""
    text = path.read_text()
    links = []
    for line in text.splitlines():
        t = line.split()
        if not t:
            continue
        if t[0] != "link":
            raise ValueError(f"chain.txt: unknown line {t[0]}")
        init = int(t[4])
        if init >= P:
            raise ValueError("chain.txt: init is not a canonical BabyBear element (a Goldilocks chain file?)")
        links.append(prover.Link(None if t[1] == "-" else ucirc.col_index(t[1]), ucirc.col_index(t[2]), ecirc.col_index(t[3]), init, t[5] == "1"))
    return prover.Chain(steps, [], [], [], ecirc.col_index("y16"), public, links=links,
                        spec_hash=hashlib.sha256(text.encode()).digest())


def load_instance(d: Path, device, vus: int | None = None, tile: bool = False) -> prover.Instance:
    root = d if (d / "manifest.json").is_file() else d.parent.parent      # DIR/neg/<id> shares DIR's manifest/circuits
    man = json.loads((root / "manifest.json").read_text())
    ucirc = load_circuit(root / "circuit.txt")
    ecirc = load_circuit(root / "epilogue.txt")
    steps = man["steps"]
    units = read_rows(d / "units.bin").astype("int64")
    epi = read_rows(d / "epilogue.bin").astype("int64")
    pub = read_rows(d / "public.bin").astype("int64")[:, 0]
    have = epi.shape[0]
    if vus is None:
        vus = have
    if vus > have:
        if not tile:
            raise ValueError(f"{have} VUs available, {vus} requested (pass tile=True for timing runs)")
        reps = -(-vus // have)
        units = torch.from_numpy(units).repeat(reps, 1).numpy()
        epi = torch.from_numpy(epi).repeat(reps, 1).numpy()
        pub = torch.from_numpy(pub).repeat(reps).numpy()
    units, epi, pub = units[: vus * steps], epi[:vus], pub[:vus]
    if man.get("variant") == "v2":
        chain = read_chain(root / man.get("chain_file", "chain.txt"), ucirc, ecirc, steps, [int(v) for v in pub])
    else:
        ch = man["chain"]
        chain = prover.Chain(steps, ch["col_c"], ch["col_y"], ch["epi_y32"], ch["epi_y16"], [int(v) for v in pub])
    uh, eh = ucirc.hash, ecirc.hash
    segs = [
        prover.Segment("unit", ucirc, layers(ucirc), torch.from_numpy(units).to(device) % P, uh),
        prover.Segment("epilogue", ecirc, layers(ecirc), torch.from_numpy(epi).to(device) % P, eh),
    ]
    return prover.Instance(segs, chain)


def _dev(name: str):
    if name == "cuda" and not torch.cuda.is_available():
        print("cuda not available, using cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(name)


def prove_once(inst: prover.Instance, pcs: bool):
    dev = inst.device
    t0 = time.perf_counter()
    proof, st = prover.prove(inst, pcs)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_prove = time.perf_counter() - t0
    t1 = time.perf_counter()
    err = None
    try:
        prover.verify(inst, proof, pcs)
    except VerifyError as e:
        err = str(e)
    if dev.type == "cuda":
        torch.cuda.synchronize(dev)
    t_verify = time.perf_counter() - t1
    return proof, st, t_prove, t_verify, err


def cmd_prove(args) -> int:
    dev = _dev(args.device)
    inst = load_instance(Path(args.dir), dev, args.vus, tile=args.tile)
    pcs = not args.clear
    runs = []
    for _ in range(args.warmup):
        # the LogUp CUDA graphs (one per round and shape) compile and capture on the first proof of a process (~1 min);
        # a warm-up proof is excluded from the recorded repetitions and reported as such
        proof, st, tp, tv, err = prove_once(inst, pcs)
        print(json.dumps({"warmup": True, "prover_s": tp, "verified": err is None, "error": err}), flush=True)
    for rep in range(args.reps):
        proof, st, tp, tv, err = prove_once(inst, pcs)
        row = {
            "rep": rep, "vus": inst.vus(), "units": inst.segs[0].units, "prover_s": tp, "verifier_s": tv,
            "verified": err is None, "error": err, "proof_bytes": proof.nbytes(), "proof_elems": st.proof_elems,
            "t_commit": st.t_commit, "t_lookup": st.t_lookup, "t_witness_wires": st.t_witness_wires, "t_arith": st.t_arith,
            "t_open": st.t_open, "slots": st.slots, "sequential_depth": st.sequential_depth(pcs),
            "depth_circuit": st.depth_circuit, "depth_lookup_max": st.depth_lookup_max, "lookup_dims": st.lookup_dims,
            "committed_elements": st.committed_elements, "ligero_rows": st.ligero_rows,
            "linkage_constraints": st.linkage_constraints, "peak_mem_bytes": st.peak_mem, "warmup_reps": args.warmup,
        }
        runs.append(row)
        print(json.dumps(row), flush=True)
        if err is not None:
            print(f"VERIFY FAILED: {err}", file=sys.stderr)
        if args.proof_out and rep == 0:
            data = proof.to_bytes()
            Path(args.proof_out).write_bytes(data)
            # round trip through the file format, then the Python verifier again on the parsed copy
            back = prover.Proof.from_bytes(data)
            prover.verify(inst, back, pcs)
            print(f"proof written: {args.proof_out} ({len(data)} bytes, round trip verified)", flush=True)
    if args.json:
        Path(args.json).write_text(json.dumps({"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                                               "pcs": pcs, "runs": runs}, indent=1) + "\n")
    return 0 if all(r["verified"] for r in runs) else 1


def _prove_verify_expect_reject(inst: prover.Instance, pcs: bool, name: str, proof_out: Path | None = None) -> dict:
    try:
        proof, _st = prover.prove(inst, pcs)
    except Exception as e:  # noqa: BLE001 -- the prover itself notices (a lookup miss, a nonzero fractional sum)
        print(name, "rejected at prover:", e, flush=True)
        return {"rejected": True, "where": "prover", "why": f"{type(e).__name__}: {e}"}
    if proof_out is not None:
        proof_out.parent.mkdir(parents=True, exist_ok=True)
        proof_out.write_bytes(proof.to_bytes())
    try:
        prover.verify(inst, proof, pcs)
    except VerifyError as e:
        print(name, "rejected at verifier:", e, flush=True)
        return {"rejected": True, "where": "verifier", "why": str(e)}
    print(name, "ACCEPTED (soundness bug)", flush=True)
    return {"rejected": False, "where": "verifier", "why": "ACCEPTED"}


def cmd_unit_negatives(args) -> int:
    """``DIR/circuit.txt`` + ``DIR/negatives/<name>.bin`` (one mutated unit row each, ``gpu.v2.export negatives``):
    each is proved as a one-segment, one-unit instance without a chain and must be rejected; ``DIR/honest1.bin``
    must be accepted."""
    dev = _dev(args.device)
    root = Path(args.dir)
    circ = load_circuit(root / "circuit.txt")
    lay = layers(circ)

    def inst_of(path: Path) -> prover.Instance:
        rows = read_rows(path).astype("int64")
        return prover.Instance([prover.Segment("unit", circ, lay, torch.from_numpy(rows).to(dev) % P, circ.hash)])

    results = {}
    honest = root / "honest1.bin"
    if honest.is_file():
        proof, _ = prover.prove(inst_of(honest), not args.clear)
        prover.verify(inst_of(honest), proof, not args.clear)
        results["honest1"] = {"rejected": False, "where": "verifier", "why": "accepted (control)"}
        print("honest1 accepted (control)", flush=True)
    for f in sorted((root / "negatives").glob("*.bin")):
        results[f.stem] = _prove_verify_expect_reject(inst_of(f), not args.clear, f.stem)
    negs = {k: v for k, v in results.items() if k != "honest1"}
    n_rej = sum(1 for r in negs.values() if r["rejected"])
    print(f"{n_rej}/{len(negs)} unit negatives rejected")
    if args.json:
        Path(args.json).write_text(json.dumps({"device": str(dev), "results": results, "rejected": n_rej, "total": len(negs)}, indent=1) + "\n")
    return 0 if n_rej == len(negs) else 1


def cmd_epilogue_negatives(args) -> int:
    """From an honest ``DIR`` (first ``--vus`` VUs): the honest instance must be accepted; then, for every base column
    of the epilogue circuit (state, flags, the word, every hint), the column of the last VU's epilogue row bumped by
    +1 and the derived columns recomputed must be rejected.  Exercises the limb-form Pack/cast (``gpu/v2/epilogue_limbs``)
    inside the GPU prover -- a bumped hint pair would be exactly the BabyBear wrap attack on the word form."""
    from verity_numerical.gkr_export import v2 as V2

    from .v2 import export as X

    dev = _dev(args.device)
    root = Path(args.dir)
    pcs = not args.clear
    inst = load_instance(root, dev, args.vus)
    proof, _ = prover.prove(inst, pcs)
    prover.verify(inst, proof, pcs)
    print("honest accepted (control)", flush=True)
    ectx, _paths = X.API.build_epilogue()
    seg = inst.segs[1]
    ecirc = seg.circ
    derived = list(ectx.derived) + [None] * (len(ectx.cols) - len(ectx.derived))
    base_cols = [c for c in range(1, len(ectx.cols)) if derived[c] is None]
    honest_rows = seg.cols.cpu()
    results = {}
    for col in base_cols:
        name = ectx.cols[col]
        base = [None if derived[c] is not None else int(v) for c, v in enumerate(honest_rows[-1].tolist())]
        base[col] = (base[col] + 1) % P
        row = V2.derive(ectx, base)
        rows = honest_rows.clone()
        rows[-1] = torch.tensor([v % P for v in row], dtype=rows.dtype)
        bad = prover.Instance([inst.segs[0], prover.Segment("epilogue", ecirc, seg.layers, rows.to(dev), seg.hash)], inst.chain)
        results[name] = _prove_verify_expect_reject(bad, pcs, f"epilogue[{name}]+1")
    n_rej = sum(1 for r in results.values() if r["rejected"])
    print(f"{n_rej}/{len(results)} epilogue negatives rejected")
    if args.json:
        Path(args.json).write_text(json.dumps({"device": str(dev), "results": results, "rejected": n_rej, "total": len(results)}, indent=1) + "\n")
    return 0 if n_rej == len(results) else 1


def cmd_negatives(args) -> int:
    dev = _dev(args.device)
    root = Path(args.dir)
    man = json.loads((root / "manifest.json").read_text())
    results = {}
    for vid, why in man.get("rejected_at_witness", {}).items():
        results[int(vid)] = {"rejected": True, "where": "witness", "why": why}
    def key(q: Path):
        return (0, int(q.name), "") if q.name.isdigit() else (1, 0, q.name)

    for d in sorted((root / "neg").iterdir(), key=key):
        name = int(d.name) if d.name.isdigit() else d.name
        inst = load_instance(d, dev)
        # --proof-dir: the proof the prover produced for this negative (an independent verifier must reject it too)
        proof_out = (Path(args.proof_dir) / f"{d.name}.bin") if args.proof_dir else None
        results[name] = _prove_verify_expect_reject(inst, not args.clear, str(name), proof_out)
    n_rej = sum(1 for r in results.values() if r["rejected"])
    print(f"{n_rej}/{len(results)} negatives rejected")
    if args.json:
        Path(args.json).write_text(json.dumps({"device": str(dev), "results": results, "rejected": n_rej, "total": len(results)}, indent=1) + "\n")
    return 0 if n_rej == len(results) else 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prove")
    p.add_argument("dir")
    p.add_argument("--vus", type=int, default=None)
    p.add_argument("--tile", action="store_true", help="tile the available VUs to --vus (timing runs only)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--reps", type=int, default=1)
    p.add_argument("--warmup", type=int, default=0, help="unrecorded proofs first (graph build / kernel compile)")
    p.add_argument("--clear", action="store_true")
    p.add_argument("--json", default=None)
    p.add_argument("--proof-out", default=None, help="write the first rep's proof as proof.bin (README 'Proof file')")
    n = sub.add_parser("negatives")
    n.add_argument("dir")
    n.add_argument("--device", default="cuda")
    n.add_argument("--clear", action="store_true")
    n.add_argument("--json", default=None)
    n.add_argument("--proof-dir", default=None, help="write every negative's proof as <id>.bin here")
    v = sub.add_parser("verify-file", help="the Python verifier on a proof.bin (cross-check for the Rust verifier)")
    v.add_argument("dir")
    v.add_argument("proof")
    v.add_argument("--vus", type=int, default=None)
    v.add_argument("--device", default="cpu")
    u = sub.add_parser("unit-negatives")
    u.add_argument("dir")
    u.add_argument("--device", default="cuda")
    u.add_argument("--clear", action="store_true")
    u.add_argument("--json", default=None)
    e = sub.add_parser("epilogue-negatives")
    e.add_argument("dir")
    e.add_argument("--vus", type=int, default=4)
    e.add_argument("--device", default="cuda")
    e.add_argument("--clear", action="store_true")
    e.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return {"prove": cmd_prove, "negatives": cmd_negatives, "verify-file": cmd_verify_file, "unit-negatives": cmd_unit_negatives,
            "epilogue-negatives": cmd_epilogue_negatives}[args.cmd](args)


def cmd_verify_file(args) -> int:
    dev = _dev(args.device)
    inst = load_instance(Path(args.dir), dev, args.vus)
    proof = prover.Proof.from_bytes(Path(args.proof).read_bytes())
    t0 = time.perf_counter()
    try:
        prover.verify(inst, proof, True)
        err = None
    except VerifyError as e:
        err = str(e)
    dt = time.perf_counter() - t0
    print(json.dumps({"verified": err is None, "error": err, "verifier_s": dt, "proof_bytes": len(proof.to_bytes())}))
    return 0 if err is None else 1


if __name__ == "__main__":
    sys.exit(main())
