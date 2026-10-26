"""Llama-3.1-70B int8's TP4 all-reduces (4 RTX PRO 6000s, vLLM's batch invariance, ``NCCL_P2P_DISABLE=1``) against the route
`tp_allreduce.CAPTURED` names for them, NCCL's tree all-reduce `AllReduceNcclTree`, and that tree probed on the GPUs.

``replay`` reads a TP4 capture (the capture worker's ``coll_meta.rank<r>.json``, ``coll.rank<r>.jsonl`` and
``coll.rank<r>.step<S>.pt``: each collective's partial and received words on each rank; the driver's ``topology.json`` and
``versions.json``). It takes the route facts from the capture (each rank's world, ``disable_custom_all_reduce`` and NCCL
environment, whose batch-invariant pins it checks; the engine's flags; the GPU's capability and SMs), checks every tensor against the
sha256 its row records, and holds every recorded all-reduce's received words on every rank to the route `tp_allreduce.covered` names
for the call at those facts, three ways: the Boolean Definition (`AllReduceNcclTreeCoordinate_v2`, bit-sliced, on every element;
`AllReduceNcclTree_v2` is one of it per element), the word Definition `AllReduceNcclTree_v1` at the call's SIZE (``--no-word`` skips
it, and the record is then NO-GO), and `tp_allreduce_check.nccl_tree_rows`. Two other orders are counted beside it, the chain reversed and the f32 sum of the four
rounded once, so the record shows the capture tells them apart. It writes ``<out>/calls.jsonl`` (one line per call) and the property
record ``<out>/properties/tp4_nccl_tree_capture.json`` (`properties.record`), and prints ``TP4-NCCL-TREE GO`` or ``NO-GO``:

    python tests/properties/tp4_nccl_tree_replay.py replay --capture DIR --out DIR [--jobs N] [--no-word]

``probe`` (4 GPUs, vLLM installed) runs one process per rank with vLLM's `PyNcclCommunicator` under `override_envs_for_invariance`'s
NCCL pins and ``NCCL_P2P_DISABLE=1``, NCCL logging its graph: the all-reduce of every 4-tuple of `SPECIALS` (NaNs, infinities,
subnormals, signed zeros, the finite maxima), of random words, of words three binades apart (rounding ties, cancellations) and of
hidden-state-like words at sizes from 1 word to 512 x 8192 (past NCCL's 4 MiB buffer), each rank's received words against the model
at CHAIN, the reversed chain and the f32 sum. It writes ``<out>/probe.json`` with the tree, channel and transport lines NCCL logged:

    CUDA_VISIBLE_DEVICES=2,3,6,7 python tests/properties/tp4_nccl_tree_replay.py probe --out DIR
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import socket
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from verity.ml.definitions.numerics import derived_rows as DR
from verity.ml.definitions.registry import tp_allreduce as T
from verity.ml.definitions.registry import tp_allreduce_check as K

SCHEMA = "verity-vllm/tp4-nccl-tree-capture/v1"
RECORD = "tp4_nccl_tree_capture.json"
WORLD = 4
DEVICE_NAME = "RTX PRO 6000 Blackwell Server Edition"
#: the NCCL switches vLLM's `override_envs_for_invariance` sets (vLLM d9105ea80) that decide the algorithm, protocol and channels
BI_PINS = {"NCCL_ALGO": "allreduce:tree", "NCCL_PROTO": "Simple", "NCCL_MIN_NCHANNELS": "1", "NCCL_MAX_NCHANNELS": "1",
           "NCCL_NVLS_ENABLE": "0", "NCCL_COLLNET_ENABLE": "0"}
#: one Boolean evaluation's rows, so a 256 x 8192 call's bits stay a few hundred MB
BOOL_ROWS = 1 << 20
#: signed zeros, the smallest and largest subnormals, the smallest normals, +-1, the finite maxima, infinities, NaNs
SPECIALS = np.array([0x0000, 0x8000, 0x0001, 0x8001, 0x007F, 0x807F, 0x0080, 0x8080, 0x3F80, 0xBF80, 0x7F7F, 0xFF7F, 0x7F80,
                     0xFF80, 0x7FC0, 0x7F81, 0xFFFF], np.uint16)
ORDERS = ("boolean", "word", "model", "reversed", "f32_once")


def f32_once(parts: np.ndarray, chain) -> np.ndarray:
    """The f32 sum of the ranks' words down `chain` from its leaf, rounded once to bf16 (F2FP)."""
    acc = DR.bf16_to_f32_words(parts[chain[-1]])
    for r in reversed(tuple(chain)[:-1]):
        acc = DR.add_rn(DR.bf16_to_f32_words(parts[r]), acc)
    return DR.f2fp_bf16(acc)


def _words(t) -> np.ndarray:
    import torch

    return t.contiguous().view(torch.int16).numpy().view(np.uint16).reshape(-1)


def _sha(t) -> str:
    import torch

    return hashlib.sha256(t.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


# --- replay ---------------------------------------------------------------------------------------------------------------------


def capture_facts(capture: Path) -> dict[str, Any]:
    """The route facts the capture recorded: per rank and agreed, world, the custom all-reduce switch, the NCCL environment, the
    GPU's name and the visible devices; the engine's flags (``topology.json``) and the GPU (``versions.json``), as `device`
    (`tp_allreduce.device_of`, None when not recorded)."""
    per = {}
    for r in range(WORLD):
        d = json.loads((capture / f"coll_meta.rank{r}.json").read_text())
        f = d.get("facts") or {}
        per[r] = {"world": int(d.get("world_size") or 0), "disable_custom_all_reduce": (f.get("parallel_config") or {}).get("disable_custom_all_reduce"),
                  "nccl_env": dict(f.get("nccl_env") or {}), "gpu_name": (f.get("device") or {}).get("name"),
                  "cuda_visible_devices": f.get("cuda_visible_devices")}
    first = per[0]
    agreed = all(per[r] == first for r in per)
    flags = dict(json.loads((capture / "topology.json").read_text()).get("env_flags") or {})
    gpu = json.loads((capture / "versions.json").read_text()).get("gpu") or {}
    device = T.device_of(gpu["cc"], gpu["sms"]) if gpu.get("cc") and gpu.get("sms") else None
    return {"ranks": per, "agreed": agreed, **first, "env_flags": flags, "gpu": gpu, "device": device}


def calls_of(capture: Path) -> tuple[dict[tuple[int, int], list[dict]], list[str]]:
    """{(step, k): [rank 0's row .. rank 3's]} of every all-reduce every rank recorded, and what disagrees across ranks."""
    rows: dict[tuple[int, int], dict[int, dict]] = {}
    for r in range(WORLD):
        for line in (capture / f"coll.rank{r}.jsonl").read_text().splitlines():
            if line.strip():
                x = json.loads(line)
                if x.get("kind") == "all_reduce":
                    rows.setdefault((int(x["step"]), int(x["k"])), {})[r] = x
    problems = []
    out = {}
    for key, by_rank in sorted(rows.items()):
        if sorted(by_rank) != list(range(WORLD)):
            problems.append(f"step {key[0]} k {key[1]}: recorded on ranks {sorted(by_rank)} only")
            continue
        shapes = {tuple(by_rank[r]["shape"]) for r in by_rank} | {tuple(by_rank[r]["out_shape"]) for r in by_rank}
        if len(shapes) != 1 or len(next(iter(shapes))) != 2 or {by_rank[r]["dtype"] for r in by_rank} != {"bfloat16"}:
            problems.append(f"step {key[0]} k {key[1]}: shapes {sorted(shapes)}, dtypes {sorted({by_rank[r]['dtype'] for r in by_rank})}")
            continue
        out[key] = [by_rank[r] for r in range(WORLD)]
    return out, problems


def route_for(facts: dict[str, Any], M: int, N: int):
    """The `CAPTURED` entry and the bound statics of one [M, N] call at the capture's facts, or (None, why)."""
    settings = T.settings_of(bool(facts["disable_custom_all_reduce"]), {**facts["nccl_env"], **facts["env_flags"]})
    s, route = T.served_route(settings, M, N)
    device = facts["device"]
    c = T.covered(s, N, route, M, world=facts["world"], device=device, transport=settings["nccl_transport"]) if s else None
    if c is None:
        return None, f"no CAPTURED entry for setting={s!r} N={N} route={route!r} M={M} world={facts['world']} device={device!r} " \
                     f"transport={settings['nccl_transport']!r}"
    return c, T.route_binding(c, facts["world"], M, N)[1]


def _replay(capture: str, step: int, items: list, word: bool) -> list[dict]:
    """One step's calls `items` ([(k, rows, statics)]): every rank's words against each order."""
    import torch

    from verity.ml.circuits import evaluation as E
    from verity.ml.circuits.defs import bind
    from verity.ml.circuits.evaluation import bits as BS
    from verity.ml.definitions.registry import boolean_tp_allreduce as BA

    dumps = [torch.load(Path(capture) / f"coll.rank{r}.step{step}.pt", map_location="cpu", mmap=True, weights_only=True)
             for r in range(WORLD)]
    res = []
    for k, rows, statics in items:
        part_t = [dumps[r]["part"][k] for r in range(WORLD)]
        out_t = [dumps[r]["out"][k] for r in range(WORLD)]
        hashes = sum((_sha(part_t[r]) != rows[r]["part_sha256"]) + (_sha(out_t[r]) != rows[r]["out_sha256"]) for r in range(WORLD))
        parts = np.stack([_words(t) for t in part_t])
        outs = np.stack([_words(t) for t in out_t])
        chain = tuple(statics["CHAIN"])
        size = parts.shape[1]
        coord = bind(BA.AllReduceNcclTreeCoordinate, WORLD=WORLD, CHAIN=chain)
        got = {"boolean": np.concatenate([np.asarray(BS.evaluate_words(coord, parts[:, lo:lo + BOOL_ROWS].T), np.uint64).reshape(-1)
                                          for lo in range(0, size, BOOL_ROWS)]).astype(np.uint16),
               "model": K.nccl_tree_rows(parts, chain), "reversed": K.nccl_tree_rows(parts, chain[::-1]), "f32_once": f32_once(parts, chain)}
        if word:
            spec = bind(T.AllReduceNcclTree, **statics)
            got["word"] = np.asarray(E.evaluate_batch(spec, parts[None]), np.uint64).reshape(-1).astype(np.uint16)
        differ = {name: int((outs != g[None, :]).sum()) for name, g in got.items()}
        res.append({"step": step, "k": k, "module": rows[0].get("module"), "shape": list(rows[0]["shape"]), "elements": int(size),
                    "spec": bind(T.AllReduceNcclTree, **statics).id, "hash_mismatch": int(hashes),
                    "ranks_equal": bool((outs == outs[0]).all()), "differ": differ})
    return res


def replay(a: argparse.Namespace) -> dict[str, Any]:
    from verity_vllm.properties.record import DIR, seal

    capture, outdir = Path(a.capture), Path(a.out)
    (outdir / DIR).mkdir(parents=True, exist_ok=True)
    facts = capture_facts(capture)
    calls, problems = calls_of(capture)
    pins = {k: facts["nccl_env"].get(k) for k in BI_PINS}
    if pins != BI_PINS:
        problems.append(f"NCCL environment {pins} is not batch invariance's {BI_PINS}")
    if not facts["agreed"]:
        problems.append("the ranks' recorded facts differ")
    if facts["world"] != WORLD or DEVICE_NAME not in str(facts["gpu_name"]) or facts["device"] != T.RTX_PRO_6000:
        problems.append(f"world {facts['world']} on {facts['gpu_name']!r} {facts['device']!r}, not {WORLD} {DEVICE_NAME} {T.RTX_PRO_6000}")
    if a.no_word:
        problems.append("the word Definition was not evaluated (--no-word)")
    steps: dict[int, list] = {}
    uncovered = []
    for (step, k), rows in calls.items():
        M, N = rows[0]["shape"]
        c, statics = route_for(facts, int(M), int(N))
        if c is None:
            uncovered.append({"step": step, "k": k, "why": statics})
            continue
        steps.setdefault(step, []).append((k, rows, statics))
    units = []
    for step, items in sorted(steps.items()):
        big = [it for it in items if it[1][0]["shape"][0] * it[1][0]["shape"][1] >= 1 << 20]
        units += [(str(capture), step, [it], not a.no_word) for it in big]
        small = [it for it in items if it not in big]
        if small:
            units.append((str(capture), step, small, not a.no_word))
    results: list[dict] = []
    if a.jobs <= 1:
        for u in units:
            results += _replay(*u)
    else:
        with ProcessPoolExecutor(max_workers=a.jobs) as ex:
            for part in ex.map(_replay, *zip(*units)):
                results += part
    results.sort(key=lambda r: (r["step"], r["k"]))
    with (outdir / "calls.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    orders = [o for o in ORDERS if a.no_word is False or o != "word"]
    shapes: dict[str, int] = {}
    for r in results:
        shapes[str(r["shape"])] = shapes.get(str(r["shape"]), 0) + 1
    doc = {"schema": SCHEMA, "capture": str(capture), "facts": facts,
           "route": next((dict(c._asdict()) for c in T.CAPTURED if c.definition == "AllReduceNcclTree_v1" and c.world == WORLD), None),
           "specs": sorted({r["spec"] for r in results}), "steps": sorted({r["step"] for r in results}), "calls": len(results),
           "shapes": shapes, "elements": sum(r["elements"] for r in results), "rank_words": WORLD * sum(r["elements"] for r in results),
           "hash_mismatch": sum(r["hash_mismatch"] for r in results), "ranks_differ": sum(not r["ranks_equal"] for r in results),
           "uncovered": uncovered, "problems": problems,
           "differ": {o: sum(r["differ"][o] for r in results) for o in orders},
           "calls_equal": {o: sum(r["differ"][o] == 0 for r in results) for o in orders}}
    ok = (not problems and not uncovered and doc["calls"] > 0 and doc["calls"] == len(calls) and doc["hash_mismatch"] == 0
          and doc["ranks_differ"] == 0 and all(doc["differ"][o] == 0 for o in orders if o not in ("reversed", "f32_once")))
    doc = seal(doc, ok)
    (outdir / DIR / RECORD).write_text(json.dumps(doc, indent=1, default=str) + "\n")
    print(f"TP4-NCCL-TREE {'GO' if ok else 'NO-GO'} {doc['digest']} calls={doc['calls']} steps={len(doc['steps'])} "
          f"rank_words={doc['rank_words']} shapes={json.dumps(shapes)} differ={json.dumps(doc['differ'])} "
          f"calls_equal={json.dumps(doc['calls_equal'])} hash_mismatch={doc['hash_mismatch']} uncovered={len(uncovered)} "
          f"problems={problems}", flush=True)
    return doc


# --- probe ----------------------------------------------------------------------------------------------------------------------


def _hidden(rng, shape) -> np.ndarray:
    return ((rng.standard_normal(shape) * 0.05).astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def probe_cases() -> list[tuple[str, np.ndarray]]:
    """(name, (WORLD, n) bf16 words), the same on every rank."""
    rng = np.random.default_rng(20261018)
    tuples = np.array(list(itertools.product(SPECIALS, repeat=WORLD)), np.uint16).T.copy()
    close = (rng.integers(124, 128, (WORLD, 1 << 20)) << 7 | rng.integers(0, 128, (WORLD, 1 << 20))
             | rng.integers(0, 2, (WORLD, 1 << 20)) << 15).astype(np.uint16)
    cases = [("specials", tuples), ("random", rng.integers(0, 1 << 16, (WORLD, 1 << 20)).astype(np.uint16)), ("close", close)]
    cases += [(f"size{n}", _hidden(rng, (WORLD, n))) for n in (1, 2, 3, 7, 1000, 65537)]
    cases += [(f"m{m}x8192", _hidden(rng, (WORLD, m * 8192))) for m in (1, 2, 3, 16, 31, 128, 255, 256, 257, 512)]
    return cases


def _probe_rank(rank: int, port: int, out: str, chain: tuple[int, ...]) -> None:
    os.environ.update(NCCL_P2P_DISABLE="1", NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,GRAPH,TUNING,ENV",
                      NCCL_DEBUG_FILE=str(Path(out) / f"nccl.rank{rank}.log"))
    from vllm.model_executor.determinism.batch_invariant import override_envs_for_invariance

    override_envs_for_invariance()
    import torch
    import torch.distributed as dist
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    torch.cuda.set_device(rank)
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=WORLD)
    comm = PyNcclCommunicator(dist.group.WORLD, device=torch.device(f"cuda:{rank}"))
    res = []
    for name, parts in probe_cases():
        x = torch.from_numpy(parts[rank].view(np.int16).copy()).view(torch.bfloat16).cuda()
        y = comm.all_reduce(x)
        torch.cuda.synchronize()
        got = _words(y.cpu())
        nan = (got & 0x7F80) == 0x7F80
        nan &= (got & 0x007F) != 0
        want = {"model": K.nccl_tree_rows(parts, chain), "reversed": K.nccl_tree_rows(parts, chain[::-1]), "f32_once": f32_once(parts, chain)}
        res.append({"case": name, "elements": int(parts.shape[1]), "differ": {o: int((got != w).sum()) for o, w in want.items()},
                    "nan_words": sorted({int(v) for v in got[nan]}), "nan": int(nan.sum())})
    (Path(out) / f"probe.rank{rank}.json").write_text(json.dumps({"rank": rank, "nccl": comm.nccl.ncclGetVersion(),
                                                                   "device": torch.cuda.get_device_name(rank), "cases": res}))
    dist.barrier()
    dist.destroy_process_group()


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def probe(a: argparse.Namespace) -> dict[str, Any]:
    import torch.multiprocessing as mp

    outdir = Path(a.out)
    outdir.mkdir(parents=True, exist_ok=True)
    chain = T.nccl_chain(next(c.statics["CHAIN"] for c in T.CAPTURED if c.definition == "AllReduceNcclTree_v1" and c.world == WORLD), WORLD)
    mp.spawn(_probe_rank, args=(_free_port(), str(outdir), chain), nprocs=WORLD, join=True)
    ranks = [json.loads((outdir / f"probe.rank{r}.json").read_text()) for r in range(WORLD)]
    log = [ln.rstrip() for r in range(WORLD) for p in sorted(outdir.glob(f"nccl.rank{r}.log*")) for ln in p.read_text(errors="replace").splitlines()
           if any(s in ln for s in ("Trees", " via ", "NCCL version", "Channel 00", "Pattern", "Algo", "NCCL_P2P_DISABLE", "NCCL_ALGO"))]
    cases = {c["case"]: {"elements": c["elements"], "differ": {o: sum(r["cases"][i]["differ"][o] for r in ranks) for o in c["differ"]},
                         "nan": sum(r["cases"][i]["nan"] for r in ranks), "nan_words": sorted({w for r in ranks for w in r["cases"][i]["nan_words"]})}
             for i, c in enumerate(ranks[0]["cases"])}
    ok = all(c["differ"]["model"] == 0 for c in cases.values()) and all(set(c["nan_words"]) <= {0x7FFF} for c in cases.values())
    doc = {"chain": list(chain), "visible": os.environ.get("CUDA_VISIBLE_DEVICES"), "nccl": ranks[0]["nccl"], "device": ranks[0]["device"],
           "cases": cases, "nccl_log": log, "ok": ok}
    (outdir / "probe.json").write_text(json.dumps(doc, indent=1) + "\n")
    for name, c in cases.items():
        print(f"[tp4-nccl-probe] {name:12s} n={c['elements']:8d} differ={json.dumps(c['differ'])} nan={c['nan']} nan_words={[hex(w) for w in c['nan_words']]}")
    for ln in log:
        if "Trees" in ln or " via " in ln:
            print(f"[tp4-nccl-probe] {ln}")
    print(f"TP4-NCCL-PROBE {'GO' if ok else 'NO-GO'} chain={list(chain)} nccl={doc['nccl']} visible={doc['visible']}", flush=True)
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="the TP4 capture's all-reduces against AllReduceNcclTree, and NCCL's tree probed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--capture", required=True, help="the capture directory (coll_meta.rank<r>.json, coll.rank<r>.jsonl, coll.rank<r>.step<S>.pt)")
    r.add_argument("--out", required=True)
    r.add_argument("--jobs", type=int, default=1)
    r.add_argument("--no-word", action="store_true", help="skip the word Definition (the Boolean one and the model still run); never GO")
    p = sub.add_parser("probe")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return 0 if (replay(a) if a.cmd == "replay" else probe(a))["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
