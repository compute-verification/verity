"""M3's live draw over the H200 block's step: commit a lockstep run's steps under salt keys only the prover
can rebuild, register their roots, and after the auditor's live coin re-run the same steps, hold every slot root and step root
to the registered ones, and write the drawn units' values and openings.

    python examples/h200/m3draw.py commit --out OUT --secret FILE --new-secret [run flags]
    python examples/h200/m3draw.py open --out OUT --secret FILE --registered ROOTS.json --draw DRAW.json [run flags]

* **The salt keys.** Step t's salt key is ``SHA-512("e2e-h200/m3draw/salt-key/v1" 0 || secret || t)[:32]``, the secret 64 bytes
  from the operating system's generator, written once by ``commit --new-secret`` to ``--secret`` (mode 0600, outside the run
  dir, never published). They are as hidden as `slots.fresh_salt_key`'s, and the re-run rebuilds the same leaves.
* **commit** writes ``roots.json`` (`ROOTS`): the run's public commitments, every step's root and every slot's root, the
  layout's digest and the step partition's size, and the requests' and outputs' digests. Nothing private.
* **open** re-runs the same steps under the same salt keys and checks every slot root and step root against
  ``--registered`` (``root_check.json``); one that differs refuses the bundle (exit 5) before any unit is opened. Then each
  drawn unit (``--draw``: ``{"step": S, "units": [i, ...]}``, i a global index of step S's partition, slot ``i // U`` and
  commit unit ``i % U``, U the layout's units a slot) is replayed from its slot's request alone (`opening.replay_slot`), its
  slot's replayed roots compared with the committed ones, its inputs and outputs replayed against the Definitions
  (`units.Replayer`) and its one commit unit opened against the step's root (`opening.Openings`), and ``drawn.json`` with
  ``drawn.bin`` (mode 0600: the rows, salts and inputs) is written.

Exit 0 when every root is the registered one and every drawn unit replays and opens; 3 when a unit doesn't; 5 when a root
differs (the bundle is refused).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as RUN
from e2e_h200.commit import Committer, fixture_secrets
from e2e_h200.driver import Driver, Params
from e2e_h200.opening import Openings, replay_slot
from e2e_h200.plan import Map
from e2e_h200.shape import SHAPES
from e2e_h200.units import Replayer
from e2e_h200.weights import Weights
from e2e_h200 import served_weights as SW

ROOTS = "verity/e2e-h200/m3-roots/v1"
DRAWN = "verity/e2e-h200/m3-drawn/v1"
SALT_TAG = b"e2e-h200/m3draw/salt-key/v1\0"
SECRET_BYTES = 64


def salt_from(secret: bytes) -> Callable[[int], bytes]:
    if len(secret) != SECRET_BYTES:
        raise ValueError(f"the salt secret is {SECRET_BYTES} bytes, not {len(secret)}")
    return lambda step: hashlib.sha512(SALT_TAG + secret + step.to_bytes(4, "big")).digest()[:32]


def new_secret(path: Path) -> bytes:
    """A fresh secret at ``path`` (0600), refused if one is there: a second secret would orphan the first run's roots."""
    path.parent.mkdir(parents=True, exist_ok=True)
    secret = os.urandom(SECRET_BYTES)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        f.write(secret)
    return secret


def read_secret(path: Path) -> bytes:
    if path.stat().st_mode & 0o077:
        raise SystemExit(f"{path} is readable by others: the salt secret is the prover's alone (chmod 600)")
    return path.read_bytes()


def families(committer: Committer, shape: Any) -> list[tuple[int, str, int]]:
    """``(first commit unit, family, units)`` in commit-unit order, covering the layout's units exactly."""
    units = Map(shape).units()
    out = sorted((committer.bases[n], n, units[n].count) for n in committer.bases)
    at = 0
    for base, name, count in out:
        if base != at:
            raise ValueError(f"family {name} starts at commit unit {base}, not {at}")
        at += count
    if at != committer.layout.unit_count:
        raise ValueError(f"the families cover {at} commit units of {committer.layout.unit_count}")
    return out


def locate(fams: list[tuple[int, str, int]], u: int) -> tuple[str, int]:
    """Commit unit ``u``'s family and index in it."""
    lo, hi = 0, len(fams)
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if fams[mid][0] <= u else (lo, mid)
    base, name, count = fams[lo]
    if not base <= u < base + count:
        raise ValueError(f"commit unit {u} is in no family")
    return name, u - base


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("commit", "open"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--secret", required=True, help="the salt secret's file (0600, outside the run dir)")
    ap.add_argument("--new-secret", action="store_true", help="commit: write a fresh secret (refused if the file exists)")
    ap.add_argument("--registered", default="", help="open: the registered roots.json")
    ap.add_argument("--draw", default="", help='open: {"step": S, "units": [global index, ...]}')
    ap.add_argument("--private", default="", help="open: where drawn.bin goes (default --out); keep it outside the run dir")
    ap.add_argument("--shape", default="small", choices=sorted(SHAPES))
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--prompt-bytes", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--micro-batch", type=int, default=0)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--attention", default=None, choices=("attn_ref", "numpy"))
    ap.add_argument("--run-id", default="e2e-h200-fixture")
    ap.add_argument("--seed", default="test")
    return ap


def roots_doc(a: argparse.Namespace, committer: Committer, results: list, prompts: np.ndarray, shape: Any) -> dict:
    outputs = np.stack([r.tokens_out for r in results], 1).astype(np.uint8)
    d = committer.describe()
    return {"format": ROOTS, "run_id": a.run_id, "shape": shape.name, "seed": a.seed, "batch": a.batch, "steps": len(results),
            "prompt_bytes": a.prompt_bytes, "context": committer.context.hex(), "layout": d["layout"],
            "layout_digest": d["layout_digest"], "units_a_slot": d["units_a_slot"], "rows_a_slot": d["rows_a_slot"],
            "row_schema": d["row_schema"], "salt_keys": "SHA-512(tag || prover's secret || step)[:32] (m3draw.salt_from)",
            "prompts_sha512": hashlib.sha512(np.ascontiguousarray(prompts.astype(np.uint8)).tobytes()).hexdigest(),
            "outputs_sha512": hashlib.sha512(np.ascontiguousarray(outputs).tobytes()).hexdigest(),
            "step_roots": [{"step": r.step, "root": r.root.hex(), "slots": r.slots.tolist(), "positions": r.positions.tolist(),
                            "slot_roots": [s.hex() for s in r.slot_roots]} for r in results]}


def root_check(mine: dict, registered: dict) -> dict:
    """Every step root and slot root of ``mine`` against ``registered``'s, and the public fields they're under."""
    fields = ("run_id", "shape", "batch", "steps", "context", "layout", "layout_digest", "units_a_slot", "prompts_sha512")
    unequal_fields = [k for k in fields if mine.get(k) != registered.get(k)]
    steps, slots_unequal = [], 0
    for m, r in zip(mine["step_roots"], registered["step_roots"], strict=True):
        bad = [b for b, (x, y) in enumerate(zip(m["slot_roots"], r["slot_roots"], strict=True)) if x != y]
        slots_unequal += len(bad)
        steps.append({"step": m["step"], "root_equal": m["root"] == r["root"], "slots_unequal": bad[:8]})
    ok = not unequal_fields and slots_unequal == 0 and all(s["root_equal"] for s in steps) \
        and mine["outputs_sha512"] == registered["outputs_sha512"]
    return {"ok": ok, "fields_unequal": unequal_fields, "outputs_equal": mine["outputs_sha512"] == registered["outputs_sha512"],
            "steps": len(steps), "step_roots_unequal": [s["step"] for s in steps if not s["root_equal"]],
            "slot_roots_checked": sum(len(m["slot_roots"]) for m in mine["step_roots"]), "slot_roots_unequal": slots_unequal,
            "first_unequal": next((s for s in steps if not s["root_equal"] or s["slots_unequal"]), None)}


class Payload:
    """``drawn.bin``: the private bytes, each named by ``[offset, length]``."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []
        self.n = 0

    def add(self, b: bytes) -> list[int]:
        span = [self.n, len(b)]
        self.parts.append(b)
        self.n += len(b)
        return span

    def write(self, path: Path) -> str:
        data = b"".join(self.parts)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        return hashlib.sha512(data).hexdigest()


def array_doc(p: Payload, a: np.ndarray) -> dict:
    a = np.ascontiguousarray(a)
    return {"dtype": str(a.dtype), "shape": list(a.shape), "bytes": p.add(a.tobytes())}


def open_drawn(a, shape, weights, params, committer, results, prompts, registered: dict, draw: dict, out: Path,
               private: Path) -> tuple[dict, int]:
    S, c, lay = committer.S, committer, committer.layout
    step, U, batch = int(draw["step"]), lay.unit_count, a.batch
    fams = families(committer, shape)
    targets = []
    for i in draw["units"]:
        b, u = divmod(int(i), U)
        if not 0 <= b < batch:
            raise ValueError(f"unit {i} is outside step {step}'s partition of {batch * U}")
        targets.append((int(i), b, u, *locate(fams, u)))
    opn, pay, docs, replays = Openings(committer, results, a.workers), Payload(), [], {}
    r = results[step]
    for i, b, u, fam, idx in targets:
        slot = int(r.slots[b])
        if slot not in replays:
            op = replay_slot(shape, params, committer, results, prompts, slot, step, a.device, a.attention)
            rp = Replayer(shape, weights, op.driver.steps, op.tokens)
            replays[slot] = (op, rp, {f.name: f for f in rp.model_families(step)})
        op, rp, by = replays[slot]
        unit = rp.unit(by[fam], step, 0, idx)
        unit.slot = slot
        opened = opn.open(op, step, unit)
        sc, rows = opn.slot(op, step)
        base, n = lay.row_base[u], len(lay.units[u])
        salts = S.row_salts(r.salt_key, b, base, n)
        opening = S.UnitOpening(b, u, sc.strings[base:base + n], sc.tree.open([u]), S.step_tree(c.context, lay, step, r.slot_roots).open([b]))
        docs.append({
            "index": i, "step": step, "slot": slot, "row": b, "position": int(r.positions[b]), "commit_unit": u,
            "family": fam, "unit": idx, "template": unit.template, "replay_ok": unit.ok, "opened": opened,
            "slot_roots_equal": all(op.roots_equal), "reference_slot_root_equal": opn.reference_equal(op, step),
            "inputs": [array_doc(pay, x) for x in unit.inputs],
            "outputs": array_doc(pay, unit.got),
            "opening": {**opening.to_json(), "rows": [{"row": pay.add(rows[base + k]), "salt": pay.add(salts[S.SALT_BYTES * k:S.SALT_BYTES * (k + 1)]),
                                                       "bits": 8 * len(rows[base + k])} for k in range(n)]},
        })
    sha = pay.write(private / "drawn.bin")
    doc = {"format": DRAWN, "run_id": a.run_id, "registered": registered.get("registration"), "step": step,
           "step_root": r.root.hex(), "slots": batch, "units_a_slot": U, "population": batch * U, "context": c.context.hex(),
           "layout": lay.name, "layout_digest": S.layout_digest(lay).hex(), "draw": draw,
           "payload": {"file": "drawn.bin", "bytes": pay.n, "sha512": sha}, "units": docs}
    (out / "drawn.json").write_text(json.dumps(doc, indent=1) + "\n")
    bad = sum(not (d["replay_ok"] and d["opened"] and d["opened"]["verified"] and d["slot_roots_equal"]
                   and d["reference_slot_root_equal"]) for d in docs)
    return doc, bad


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    secret_path = Path(a.secret)
    if a.phase == "commit" and a.new_secret:
        secret = new_secret(secret_path)
    else:
        secret = read_secret(secret_path)
    shape = SHAPES[a.shape]
    seed = {"test": SW.TEST_SEED, "demo": SW.DEMO_SEED}.get(a.seed, a.seed.encode())
    weights = Weights(shape, seed)
    a.store, a.salt, a.stage_lib, a.weights_file, a.weights_manifest = "generator", "", "", "", ""
    store = RUN.make_store(a, weights, a.device)
    ctx, key = fixture_secrets(a.run_id)
    committer = Committer("slots", ctx, key, shape, a.device, salt_from(secret))
    prompts = RUN.prompts_for(a.run_id, a.batch, a.prompt_bytes)
    params = Params(shape, store, a.device, True)
    print(json.dumps({"phase": a.phase, "params": params.describe()}), flush=True)
    t0 = time.perf_counter()
    driver = Driver(shape, params, a.batch, committer, a.device, keep=False, micro_batch=a.micro_batch or None,
                    attention=a.attention)
    results = driver.run(prompts, a.steps)
    for r in results:
        print(json.dumps({"step": r.step, "root": r.root.hex()[:32], "seconds": {k: round(v, 3) for k, v in r.seconds.items()}}),
              flush=True)
    mine = roots_doc(a, committer, results, prompts, shape)
    mine["seconds"] = time.perf_counter() - t0
    (out / "roots.json").write_text(json.dumps(mine, indent=1) + "\n")
    if a.phase == "commit":
        print(json.dumps({"roots": "roots.json", "steps": mine["steps"], "last_step_root": mine["step_roots"][-1]["root"],
                          "population_a_step": a.batch * mine["units_a_slot"]}), flush=True)
        return 0
    registered = json.loads(Path(a.registered).read_text())
    check = root_check(mine, registered)
    (out / "root_check.json").write_text(json.dumps(check, indent=1) + "\n")
    print("ROOTCHECK " + ("EQUAL" if check["ok"] else "UNEQUAL") + " " + json.dumps(
        {k: check[k] for k in ("steps", "step_roots_unequal", "slot_roots_checked", "slot_roots_unequal", "fields_unequal",
                               "outputs_equal")}), flush=True)
    if not check["ok"]:
        print("REFUSED: a re-run root differs from the registered one; no unit is opened", flush=True)
        return 5
    draw = json.loads(Path(a.draw).read_text())
    t0 = time.perf_counter()
    private = Path(a.private) if a.private else out
    private.mkdir(mode=0o700, parents=True, exist_ok=True)
    doc, bad = open_drawn(a, shape, weights, params, committer, results, prompts, registered, draw, out, private)
    for d in doc["units"]:
        print("DRAWN " + json.dumps({k: d[k] for k in ("index", "slot", "commit_unit", "family", "unit", "template", "replay_ok",
                                                       "opened", "slot_roots_equal", "reference_slot_root_equal")}), flush=True)
    print(json.dumps({"opened": len(doc["units"]), "bad": bad, "seconds": time.perf_counter() - t0}), flush=True)
    store.close()
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
