"""Load a VU batch directory (``bb_export``) and prove / verify it on a device; the measurement driver.

    python -m gpu.run prove DIR [--vus N] [--device cuda] [--reps 3] [--clear] [--json OUT]
    python -m gpu.run negatives DIR             # every DIR/neg/<id>: prove + verify, all must reject
"""
from __future__ import annotations

import argparse
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
    for rep in range(args.reps):
        proof, st, tp, tv, err = prove_once(inst, pcs)
        row = {
            "rep": rep, "vus": inst.vus(), "units": inst.segs[0].units, "prover_s": tp, "verifier_s": tv,
            "verified": err is None, "error": err, "proof_bytes": proof.nbytes(), "proof_elems": st.proof_elems,
            "t_commit": st.t_commit, "t_lookup": st.t_lookup, "t_witness_wires": st.t_witness_wires, "t_arith": st.t_arith,
            "t_open": st.t_open, "slots": st.slots, "sequential_depth": st.sequential_depth(pcs),
            "depth_circuit": st.depth_circuit, "depth_lookup_max": st.depth_lookup_max, "lookup_dims": st.lookup_dims,
            "committed_elements": st.committed_elements, "ligero_rows": st.ligero_rows,
            "linkage_constraints": st.linkage_constraints, "peak_mem_bytes": st.peak_mem,
        }
        runs.append(row)
        print(json.dumps(row), flush=True)
        if err is not None:
            print(f"VERIFY FAILED: {err}", file=sys.stderr)
    if args.json:
        Path(args.json).write_text(json.dumps({"device": str(dev), "gpu": torch.cuda.get_device_name(dev) if dev.type == "cuda" else "cpu",
                                               "pcs": pcs, "runs": runs}, indent=1) + "\n")
    return 0 if all(r["verified"] for r in runs) else 1


def cmd_negatives(args) -> int:
    dev = _dev(args.device)
    root = Path(args.dir)
    man = json.loads((root / "manifest.json").read_text())
    results = {}
    for vid, why in man.get("rejected_at_witness", {}).items():
        results[int(vid)] = {"rejected": True, "where": "witness", "why": why}
    for d in sorted((root / "neg").iterdir(), key=lambda q: int(q.name)):
        inst = load_instance(d, dev)
        try:
            proof, _st = prover.prove(inst, not args.clear)
        except Exception as e:  # noqa: BLE001 -- the prover itself notices (a lookup miss, a nonzero fractional sum)
            results[int(d.name)] = {"rejected": True, "where": "prover", "why": f"{type(e).__name__}: {e}"}
            print(d.name, "rejected at prover:", e, flush=True)
            continue
        try:
            prover.verify(inst, proof, not args.clear)
            results[int(d.name)] = {"rejected": False, "where": "verifier", "why": "ACCEPTED"}
            print(d.name, "ACCEPTED (soundness bug)", flush=True)
        except VerifyError as e:
            results[int(d.name)] = {"rejected": True, "where": "verifier", "why": str(e)}
            print(d.name, "rejected at verifier:", e, flush=True)
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
    p.add_argument("--clear", action="store_true")
    p.add_argument("--json", default=None)
    n = sub.add_parser("negatives")
    n.add_argument("dir")
    n.add_argument("--device", default="cuda")
    n.add_argument("--clear", action="store_true")
    n.add_argument("--json", default=None)
    args = ap.parse_args(argv)
    return cmd_prove(args) if args.cmd == "prove" else cmd_negatives(args)


if __name__ == "__main__":
    sys.exit(main())
