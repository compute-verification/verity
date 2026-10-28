"""H1 + fx1 receipts beside vLLM's GEMM, not useful-work credit: a padded H1 + fx1 window (`eps1-sm120-h1fx1`) on vLLM's
int8-exact row, one process on one GPU, in the static form: the window's tiles, rows and positions are a
fixed function of public parameters alone.

The receipts bind extra work beside cuBLASLt's served y, not the served kernel's.  vllm's PoUW option
(``pouw:eps1-sm120-h1fx1``, `verity_vllm.protocols.pouw.h1`) hands over each int8-exact linear call's s8 rows and scales
and its layer, and holds its own stand-in root(B) per layer (`pouw_h1.weight_root`), their weights' root and the epoch's
64-byte salt from the challenger's salt coin (`pouw_h1.program_salt`).  Its own per-call positions, padding and stand-in
roots follow the step's real rows, so this window doesn't use them (the seeded mode keeps the option's salt).  Instead:

* **Public parameters**: the model, the batch slots (vLLM's ``max_num_seqs``), ``T`` rows a step (``max_num_batched_tokens``:
  the scheduler never puts more of a step's tokens in a linear), ``S`` steps and the max tokens.
* **A side**: each (step s, linear w) call's A table holds all T slot rows, as AROW's row table (`commit.s8_table`,
  `verity.ml.commitments.row_table`: K s8 bytes, then the f32 scale).  The step's real rows fill slots 0 .. m - 1, and every
  other slot is an empty row, committed like the rest: zero bytes with scale 0.  Since a served row's scale is never 0,
  "empty" is read from the committed row.  Steps past the last served one are all empty.  A row count never comes from the
  step's tokens, its batch or EOS.  Step s's A root is the tree over its A tables' roots (`step_roots`); AROW's step root
  replaces it once AROW computes it at serve time.  The window's A table is the tree over the steps' A roots.
* **Units and positions**: the call's units are its T/32 row blocks by N/32 column blocks, at positions
  s·U + off(w) + u.  Here U = Σ_w (T/32)(N_w/32), off(w) is the sum over the linears before w, and u runs row block by row block.
  The units' count, S·U, is fixed by the public parameters.
* **B side**: each linear's B table holds its N weight rows in the same row-table format (K s8 bytes, then the row's f32
  weight scale), and the weights' root is the tree over the linears' B roots in linear order (`b_tables`).  It is registered
  before every salt: the challenger's salt coin opens against the registration root over it and the queue's root.
* **Request queue** (`credit.queue_of`): the scheduler's record of a reference pass over the same prompts (`schedule_tap`:
  each engine step's requests and their rows in batch order) as the slot map (per step and slot, the request's index, 0
  for none) and each request's live steps [start, stop), committed (`commit_queue`) before the first coin.  The served
  pass's record must be the same queue.
* **Selection** (`credit`): slot i of step t is selected iff the queue assigns it a request live at t, never a function of
  the row, and admitted iff it is selected and its row's scale is nonzero (`pouw.circuit.admit.PouwQueueAdmit`).  A call's
  T admit bits are T/16 16-bit words, each its own hiding leaf of the call's admit tree, written by an always-proved admit
  unit that reads the 16 slots' queue entries and scales, and refuses a slot it does not admit unless its row is the
  canonical empty row (`credit.junk`) and a selected slot whose row is (`credit.hollow`).
* **Receipts**: `h1tile`'s unit (`PouwH1Tile_v4`) on the GPU (`receipts.gated_receipts`) on every live unit, one with an
  admitted slot among its 32 rows, whatever their bytes: the tile's `admit` input is its rows' admit bits, bound by
  commitment equality to the admit-tree leaves the tally reads, so it is live exactly when the tally credits it
  (RT4-LOGIC-4).  Seeds come from the salt, the position and the call's (root(A), root(B)), inside the unit.  A dead unit skips
  the noise and the receipt: its leaf is the canonical empty receipt (zero words), so every receipt tree keeps its shape.
  There is one receipt tree per call.
* **Coins** (``--coins``): seeded, the option's one salt, opened at install before serving; or live, a round a step from
  the proof system's challenger (`coins.py`), whose salt coin opens first, against the registration root over the
  weights' root and the queue's root.  Step s's round root, over its A root and its admit root, is recorded before round
  s's coin opens (`sampled_proofs.challenger.LiveRecord`), and the step's units take the round's salt (`h1tile.live_salt`:
  `window.round_key(...).stream(0, 64)`), so every candidate and every selection a round's noise reads is committed before
  that noise's coin.  The gate's units come from the draw coin, opened after the window's receipts root (`receipts_root`).
  This window has no gated session of its own (compute accounting, 21 Oct): the gated served window of record is vllm's
  H1 path, `integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py` (#3053), which holds the salt, round-0 request
  table, open root R, session S and requests_root rules (#2950, PRF-132), so ``--gate USER@HOST`` (or
  ``$POUW_CHALLENGER_GATE``) is refused.  ``--challenger HOST:PORT`` and ``--session ID`` (or ``$POUW_CHALLENGER`` and
  ``$POUW_CHALLENGER_SESSION``, with ``$POUW_CHALLENGER_SECRET_FILE`` for a session that has a secret) take a session
  opened before the run, for development and timing only: its salt coin opens after the registration root, so a gate
  session's challenger, which opens it only after the session's root, refuses it.  The layout is the window's
  (`coins.layout`) and the manifest records the session.  Without a session the challenger is `window.live_challenger`
  in this process, and the run is labelled ``coins=in-process (not certified)``, as is every seeded run.
* **Credit**: only the admitted slots, a tile's iff `PouwH1Tile_v4`'s live bit (the OR of its admit bits) is set, an
  admitted tile of zero-byte rows included (RT4-LOGIC-20) (`credit.tally`, in 16-bit limbs, committed and not proved until
  a tally Definition exists): Cr = Σ admitted rows · N · K, and the published credit point(level(floor_point(Cr))) <= Cr.
  The padded compute (the non-admitted rows of live units, noised) is shown and not credited; dead units cost only their
  emptiness check.
* **Not covered**: attention (QKᵀ, P·V) and lm_head.

In order, into OUT (`record.json` throughout, `manifest.json` at the end, the option's `pouw-h1-calls.json`):

  timing    `noise.cu` over a call's per-unit noise against `torch._int_mm` of the same call, CUDA events, K = 4096 and 14336
  engine    the int8 pin and `engine.build.build_engine(target=from_device() + int8_exact)`, eager, prefix caching off, T and slots
  wrap      the fraction of weights that W' = B + c + U mod 2^8 wraps, per layer and site (`wrap_counts`)
  reference the prompts generated with no option installed, the scheduler's record taken (`schedule_tap`)
  queue     the request queue from that record, committed
  install   the option (its own stand-in weight roots and salt, unused under live coins)
  weights   every linear's B table and root(B), the weights' root, and the registration root over it and the queue's root,
            which the live challenger's salt coin opens against
  serve     the prompts again with the option; inside its `record`, each call's A table, admit words and receipts; then the
            empty steps; the token ids must be the reference's
  receipts  one receipt tree per call (one leaf per unit, `commit`)
  credit    the tally's limbs and tree, the credited (admitted) and padded split, the dead units skipped
  prover    the prover's data (the table key, each call's real rows, scales and receipts) as SCRATCH/prover.tar, before the
            gate so a gate failure keeps it; `serve.sh` puts it in the store once the GPU is released, else moves it into
            OUT, which the run's custody uploads
  gate      units from the draw coin (live; the prover's entropy when seeded), live and dead, of real and empty steps,
            against `h1tile.reference` gated by the tile's admit
            words opened from the call's admit tree (receipt) and the
            served integer product (z); controls: the unit at the next position, a flipped receipt bit, root(A) rebuilt (and
            a tampered row, real or empty), the tile's 32 A rows opened against root(A) and the step's A root and A' = -b
            (the tile's B rows negated) refused there, its re-commitment moving both roots (CMP-38), root(B) rebuilt, the
            tile's 32 B rows opened against it and a tampered weight row refused, the admit root rebuilt from the queue, a
            dead unit's rows empty and its skipped noise real work, a junk row in a slot the queue leaves unassigned refused
            and not admitted, the canonical empty row in a slot the queue selects refused and not admitted, the tile's admit
            words equal to the tally's bits and a tampered one refused against the admit root, the tile's selected rows
            zeroed under a nonzero scale still admitted, a live unit's receipt the ungated one

    cd integrations/vllm; PYTHONPATH=.:../..:../../tools/research/src:../../tools/config:../../benchmarks/pouw/served_h1
    $PY ../../benchmarks/pouw/served_h1/serve.py --out OUT --cubin noise.cubin --at 2026-10-18T00:00:00Z --by compute-accounting \
      --coins live --challenger 127.0.0.1:PORT --session ID
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import coins as CO  # noqa: E402
import commit as CM  # noqa: E402
import credit as CR  # noqa: E402
from verity_config.settings import of  # noqa: E402

settings = of(__file__)
SCHEMA = "verity/pouw-h1-pass/v1"
RECORD = "verity/pouw-served-h1/record/v1"
SCHEME = "eps1-sm120-h1fx1"
FRAMING = "the receipts bind extra work beside cuBLASLt's served y, not the served kernel's"
NOT_COVERED = ["attention (QKᵀ, P·V)", "lm_head"]
SITES = ("qkv_proj", "o_proj", "gate_up_proj", "down_proj")


def site_of(name: str) -> tuple[int, str]:
    return int(name.split(".layers.")[1].split(".")[0]), name.rsplit(".", 1)[1]


class RequestRow:
    def __init__(self, out: Path):
        self.out, self.doc = out, {"schema": RECORD, "argv": sys.argv, "steps": {}}

    def step(self, name: str, value: Any) -> None:
        self.doc["steps"][name] = value
        self.save()
        print(f"[served-h1] {name}: {json.dumps(value, default=str)[:600]}", flush=True)

    def save(self) -> None:
        (self.out / "record.json").write_text(json.dumps(self.doc, indent=1, default=str) + "\n")


# -- timing -------------------------------------------------------------------------------------------------------------

def timing(cubin, reps: int = 10) -> dict:
    """Per-unit noise on the GPU against the int8 GEMM of the same M x N x K call, both from CUDA events."""
    import torch
    import receipts as RC

    def clock(fn, n: int) -> float:
        fn()
        torch.cuda.synchronize()
        s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        s.record()
        for _ in range(n):
            fn()
        e.record()
        torch.cuda.synchronize()
        return s.elapsed_time(e) / n

    out = []
    for m, n, k in ((4096, 4096, 4096), (4096, 4096, 14336)):
        a = torch.randint(-127, 128, (m, k), dtype=torch.int8, device="cuda")
        b = torch.randint(-127, 128, (n, k), dtype=torch.int8, device="cuda")
        gemm = clock(lambda: torch._int_mm(a, b.t()), reps)
        units = (m // 32) * (n // 32)
        lay = RC.layout(k)
        blocks = lay.blocks + RC.LANES
        seeds = torch.randint(-2 ** 62, 2 ** 62, (units, 8), dtype=torch.int64, device="cuda")
        sink = torch.empty(units, blocks, dtype=torch.int32, device="cuda")
        hashed = clock(lambda: cubin.launch(seeds, blocks, sink, sink=True), reps)
        full = torch.empty(units, blocks, 16, dtype=torch.int32, device="cuda")
        written = clock(lambda: cubin.launch(seeds, blocks, full), reps)
        full = sink = None
        torch.cuda.empty_cache()
        out.append({"m": m, "n": n, "k": k, "units": units, "blocks_per_unit": blocks, "noise_blocks": lay.blocks,
                    "key_blocks": RC.LANES, "gemm_ms": round(gemm, 4), "noise_hash_ms": round(hashed, 4),
                    "noise_written_ms": round(written, 4), "ratio_hash": round(hashed / gemm, 3),
                    "ratio_written": round(written / gemm, 3), "ns_per_unit_hash": round(1e6 * hashed / units, 2),
                    "gemm_tops": round(2 * m * n * k / gemm / 1e9, 1),
                    "compressions_per_s": round(units * blocks / hashed * 1e3 / 1e9, 2)})
    return {"device": torch.cuda.get_device_name(0), "reps": reps, "shapes": out,
            "what": "noise_hash: SHA-512 of every noise and key block of every 32 x 32 unit, each block's halves XORed to one "
                    "word (no stream written); noise_written: the same, writing the stream receipts.py reads; gemm: "
                    "torch._int_mm (cuBLASLt s8, s32 accumulate) of the call; the seeds' 4 compressions a unit and the fold are "
                    "not in either"}


# -- weights ------------------------------------------------------------------------------------------------------------

def wrap_prob() -> np.ndarray:
    """P(B + c + U leaves [-128, 127]) for each B in [-128, 127]: c a fair bit, U = -64 + (byte mod 129), byte uniform."""
    pu = np.zeros(129)
    for byte in range(256):
        pu[byte % 129] += 1 / 256
    p = np.zeros(256)
    for v in range(-128, 128):
        for c in (0, 1):
            for r in range(129):
                if not -128 <= v + c + r - 64 <= 127:
                    p[v + 128] += 0.5 * pu[r]
    return p


def wrap_counts(mods: dict) -> dict:
    import torch
    p = torch.tensor(wrap_prob(), dtype=torch.float64)
    per_layer: dict[int, dict] = {}
    for name, m in mods.items():
        layer, site = site_of(name)
        h = torch.bincount((m.weight.to(torch.int64) + 128).flatten(), minlength=256).cpu().to(torch.float64)
        tot = float(h.sum())
        big = float(h[:128 - 63].sum() + h[128 + 64:].sum())
        d = per_layer.setdefault(layer, {"weights": 0, "abs_gt_63": 0, "expected_wrapped": 0.0, "sites": {}})
        d["weights"] += int(tot)
        d["abs_gt_63"] += int(big)
        d["expected_wrapped"] += float((h * p).sum())
        d["sites"][site] = {"weights": int(tot), "frac_abs_gt_63": round(big / tot, 6),
                            "frac_expected_wrapped": round(float((h * p).sum()) / tot, 6), "abs_max": int(m.weight.to(torch.int16).abs().max())}
    for d in per_layer.values():
        d["frac_abs_gt_63"] = round(d["abs_gt_63"] / d["weights"], 6)
        d["frac_expected_wrapped"] = round(d["expected_wrapped"] / d["weights"], 6)
    tot = sum(d["weights"] for d in per_layer.values())
    return {"rule": "W' = B + c + U mod 2^8 wraps where B + c + U is outside [-128, 127]: possible only for |B| > 63; "
                    "expected_wrapped weights each B by P(wrap | B) over c (a fair bit) and U (-64 + byte mod 129)",
            "frac_abs_gt_63": round(sum(d["abs_gt_63"] for d in per_layer.values()) / tot, 6),
            "frac_expected_wrapped": round(sum(d["expected_wrapped"] for d in per_layer.values()) / tot, 6),
            "layers": {str(k): v for k, v in sorted(per_layer.items())}}


def padded(t, rows: int):
    """`t` [m, k] with zero rows appended up to `rows`."""
    import torch
    if t.shape[0] == rows:
        return t
    out = torch.zeros(rows, t.shape[1], dtype=t.dtype, device=t.device)
    out[:t.shape[0]] = t
    return out


# -- the run ------------------------------------------------------------------------------------------------------------

def static_layout(ns: list[int], t: int) -> tuple[list[int], int]:
    """off(w) of each linear and U, the units of one step: linear w's call is (t / 32) row blocks by (N_w / 32) column blocks."""
    offs, at = [], 0
    for n in ns:
        if n % 32 or t % 32:
            raise ValueError(f"N = {n} or T = {t} is not a multiple of 32")
        offs.append(at)
        at += (t // 32) * (n // 32)
    return offs, at


def chunk_of(k: int) -> int:
    return max(16, (1 << 22) // k)


def step_roots(window: str, s: int, cs: list[dict]) -> dict:
    """Step s's roots from its calls (in linear order): `a_step`, the tree over its A tables' roots (`CM.step_root`, domain
    'a-step'), which AROW's step root replaces once AROW computes it at serve time; `admit_step`, over its admit roots
    ('admit-step'); and `root`, the round root over the two ('round'), which `LiveRecord` records before round s's coin."""
    fields = {"window": window, "step": s, "linears": len(cs)}
    a_step = CM.step_root(CM.domain(fields, len(cs), "a-step"), [bytes.fromhex(c["root_a"]) for c in cs])
    admit_step = CM.step_root(CM.domain(fields, len(cs), "admit-step"), [bytes.fromhex(c["root_admit"]) for c in cs])
    r_fields = {"window": window, "step": s, "leaves": "a-step root, admit-step root"}
    root = CM.step_root(CM.domain(r_fields, 2, "round"), [a_step, admit_step])
    return {"fields": r_fields, "a_step": a_step, "admit_step": admit_step, "root": root}


#: the gate's draw key: the challenger's draw coin, opened once after the window's receipts root
GATE_DRAW_DOMAIN = "verity/pouw/served-h1-gate-draw/v1"


def receipts_root(window: str, calls: list[dict]) -> bytes:
    """The window's receipts root, the tree over its calls' receipt roots in call order ('receipts-window'): what the draw
    coin opens after, so no unit is drawn before every receipt is committed."""
    return CM.step_root(CM.domain({"window": window, "calls": len(calls)}, len(calls), "receipts-window"),
                        [bytes.fromhex(c["receipt_root"]) for c in calls])


def gate_rng(commitment, coin: bytes, after: bytes) -> random.Random:
    """The gate's picks from the draw coin `coin` opened after `after` (`receipts_root`): the harness's self-check of the
    receipts, not the window's draw of record."""
    from verity.ml.randomness import derive
    key = derive(coin, GATE_DRAW_DOMAIN, {"coins": commitment.root, "epoch": commitment.epoch, "receipts": after})
    return random.Random(key.stream("gate", 64))


def coins_expect(manifest: dict) -> dict:
    """What the verifier recomputes from a live window's manifest for `coins.check`: the round roots from its calls' A
    and admit roots (`step_roots`), each round's unit salt as its calls carry it, and the draw coin's root, the receipts
    root over its calls' receipt roots, when the window opened one."""
    calls, window = manifest["calls"], manifest["window"]
    by_step: dict[int, list[dict]] = {}
    for c in calls:
        by_step.setdefault(c["step"], []).append(c)
    salts = {}
    for s, cs in by_step.items():
        got = {c.get("salt") for c in cs}
        salts[s] = got.pop() if len(got) == 1 else "the step's calls read more than one salt"
    return {"rounds": [step_roots(window, s, by_step[s])["root"] for s in sorted(by_step)], "salts": salts,
            "draw_after": receipts_root(window, calls) if manifest.get("coins", {}).get("draw_coin") else None}


def schedule_tap(runner) -> tuple[list, dict, Any]:
    """(spans, info, restore): each engine step's (request id, rows) in batch order, the order its token rows take in every
    linear, read once the runner's `execute_model` returns, from the step's batch (the new runner's `gather_batch_req_state`
    rows, else `execute_model_state.input_batch`, else the persistent `input_batch`) and its scheduled tokens.  ``info``
    counts the steps whose rows don't sum to the step's scheduled tokens (``short``)."""
    spans: list[list[tuple[str, int]]] = []
    info: dict[str, Any] = {"src": None, "short": 0}
    seen: dict[str, Any] = {}
    orig_exec, orig_gather = runner.execute_model, getattr(runner, "gather_batch_req_state", None)

    def gather_batch_req_state(*args, **kw):
        res = orig_gather(*args, **kw)
        bs = res[0] if isinstance(res, tuple) else res
        if getattr(bs, "req_ids", None) is not None:
            seen["batch"] = bs
        return res

    def execute_model(so, *args, **kw):
        out = orig_exec(so, *args, **kw)
        n_tok = int(getattr(so, "total_num_scheduled_tokens", 0) or 0)
        if not n_tok:
            return out
        nst = dict(getattr(so, "num_scheduled_tokens", None) or {})
        ib, src = seen.pop("batch", None), "gather_batch_req_state"
        if ib is None:
            ib, src = getattr(getattr(runner, "execute_model_state", None), "input_batch", None), "execute_model_state.input_batch"
        if ib is None:
            ib, src = getattr(runner, "input_batch", None), "input_batch"
        ids = list(getattr(ib, "req_ids", None) or [])
        n = int(getattr(ib, "num_reqs", len(ids)) or len(ids))
        per_row = getattr(ib, "num_scheduled_tokens", None)
        per_row = None if per_row is None or isinstance(per_row, dict) else [int(x) for x in per_row]
        step = []
        for i, rid in enumerate(ids[:n]):
            k = per_row[i] if per_row is not None and i < len(per_row) else int(nst.get(rid, 0))
            if rid is not None and k > 0:
                step.append((str(rid), k))
        info["src"] = src
        info["short"] += int(sum(k for _, k in step) != n_tok)
        spans.append(step)
        return out

    runner.execute_model = execute_model
    if orig_gather is not None:
        runner.gather_batch_req_state = gather_batch_req_state

    def restore() -> None:
        runner.execute_model = orig_exec
        if orig_gather is not None:
            runner.gather_batch_req_state = orig_gather

    return spans, info, restore


def commit_queue(window: str, key: bytes, queue: dict) -> dict:
    """The queue's two hiding tables (`credit.queue_leaves`: 'queue-slots', one leaf per step and 16 slots; 'queue-requests',
    padded to the public bound S·T + 1) and its root over their roots ('queue')."""
    slots, reqs = CR.queue_leaves(queue)
    sm = queue["slot_map"]
    s_dom = CM.domain({"window": window, "steps": int(sm.shape[0]), "slots": int(sm.shape[1])}, len(slots), "queue-slots")
    r_dom = CM.domain({"window": window, "requests": len(reqs)}, len(reqs), "queue-requests")
    s_lv = CM.levels(s_dom, [CM.leaf(s_dom, key, "q", i, x)[0] for i, x in enumerate(slots)])
    r_lv = CM.levels(r_dom, [CM.leaf(r_dom, key, "qr", i, x)[0] for i, x in enumerate(reqs)])
    fields = {"window": window, "leaves": "queue-slots root, queue-requests root"}
    root = CM.step_root(CM.domain(fields, 2, "queue"), [s_lv[-1][0], r_lv[-1][0]])
    return {"root": root, "slots_root": s_lv[-1][0], "requests_root": r_lv[-1][0], "fields": fields,
            "slot_leaves": len(slots), "request_leaves": len(reqs), "s_levels": s_lv, "r_levels": r_lv}


def b_rows(layer) -> np.ndarray:
    """A linear's B table rows (`commit.s8_rows`): each weight row's N s8 bytes, then its f32 weight scale."""
    w = layer.weight.detach().cpu().numpy()
    s = layer.weight_scale.detach().float().reshape(-1).cpu().numpy()
    if s.size == 1:
        s = np.full(w.shape[0], s[0], dtype=np.float32)
    if s.size != w.shape[0]:
        raise ValueError(f"a weight of {w.shape[0]} rows with {s.size} scales")
    return CM.s8_rows(w, s)


def b_fields(model: dict, w: int, module: str, n: int, k: int) -> dict:
    return {"model": model, "layer": w, "module": module, "rows": n, "k": k, "schema": CM.SCHEMA}


def b_tables(pool, model: dict, layers: list, names: dict, key: bytes, chunk: int = 2048) -> dict:
    """Each linear's B table root (AROW's row-table format over `b_rows`, domain 'b-table', labels b/<w>), built in pool
    tasks, and the weights' root, the tree over them in linear order ('weights')."""
    roots = []
    for w, m in enumerate(layers):
        rows = b_rows(m)
        n, k = int(m.weight.shape[0]), int(m.weight.shape[1])
        dom = CM.domain(b_fields(model, w, names[w], n, k), n, "b-table")
        got = pool.map(CM.leaves, [(dom.binding, dom.count, key, f"b/{w}", at, rows[at:at + chunk])
                                   for at in range(0, n, chunk)], chunksize=1)
        roots.append(CM.levels(dom, [x for g in got for x in CM.split(g)])[-1][0])
    fields = {"model": model, "layers": len(layers)}
    return {"roots": roots, "root": CM.step_root(CM.domain(fields, len(layers), "weights"), roots), "fields": fields}


def run(a: argparse.Namespace) -> dict:
    import multiprocessing as mp
    import torch

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = RequestRow(out)
    if a.gate:
        raise ValueError(f"--gate {a.gate}: served_h1's own gated window is superseded; the gated served window of "
                         "record is vllm's H1 path, integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py (#3053), "
                         "which holds the salt, round-0 request table, R, S and requests_root rules (#2950, PRF-132); "
                         "here a window's coins are in-process, or a development session's (--challenger, --session)")
    at = CO.session(a.challenger, a.session)
    a.coins = a.coins or ("live" if at is not None else "seeded")
    live = a.coins == "live"
    if at is not None and not live:
        raise ValueError(f"challenger session {at.id} serves live coins, not --coins {a.coins}")
    n_rounds = a.steps or a.max_tokens + 2
    cs = CO.open_coins(at, a.epoch, n_rounds) if live else None
    rec.doc["coins"] = cs.label if live else CO.IN_PROCESS
    rec.step("coin_source", {"mode": a.coins, **(cs.record() if live else {"source": "in-process", "label": CO.IN_PROCESS}),
                             **({"commitment": cs.commitment.to_json()} if live else {})})
    workers = a.workers or max(2, min(48, len(os.sched_getaffinity(0)) - 2))
    pool = mp.get_context("spawn").Pool(workers)
    import receipts as RC

    cubin = RC.Cubin(a.cubin) if a.cubin else None
    torch.backends.cuda.matmul.allow_tf32 = False
    if cubin is not None and not a.skip_timing:
        rec.step("timing", timing(cubin))
    if cubin is not None:
        probe = [os.urandom(64) for _ in range(3)]
        rec.step("noise_kernel_vs_hashlib", {"equal": bool(torch.equal(RC.noise_blocks(probe, 40, "cuda", cubin).cpu(),
                                                                       RC.noise_blocks_cpu(probe, 40)))})

    from tests.engine import int8_exact_gpu as G
    from verity.core.protocols.pouw import verifier
    from verity.core.protocols.pouw.coins import Challenger
    from verity_vllm import protocols as PO
    from verity_vllm.engine import int8_exact as E
    from verity_vllm.engine import int8_ops
    from verity_vllm.engine.build import build_engine, target_mismatches
    from verity_vllm import TargetProfile
    from verity_vllm.protocols import pouw
    from verity_vllm.protocols.pouw import h1 as PH

    t_max, s_max = a.rows, a.steps or a.max_tokens + 2
    target = TargetProfile.from_device().with_overrides(int8_exact=a.int8_exact)
    prompts = list(G.PROMPTS)
    with tempfile.TemporaryDirectory() as scratch:
        entry, how = G.pin(a)
        checkpoints = G.scratch_checkpoints(a.checkpoints, entry, Path(scratch))
        eng = build_engine(checkpoints, case=a.role, target=target, max_num_seqs=len(prompts),
                           engine_args={"gpu_memory_utilization": a.gpu_mem, "max_num_batched_tokens": t_max,
                                        "enable_chunked_prefill": True})
    public = {"model": {k: entry.get(k) for k in ("repo", "revision", "recipe")}, "slots": len(prompts), "rows": t_max,
              "steps": s_max, "max_tokens": a.max_tokens}
    rec.step("engine", {"pin": {k: entry.get(k) for k in ("repo", "revision", "recipe", "local_path")}, "pin_source": how,
                        "build_s": round(eng.build_s, 1), "target_mismatches": target_mismatches(eng),
                        "enforce_eager": eng.kwargs.get("enforce_eager"), "prefix_caching": eng.kwargs.get("enable_prefix_caching"),
                        "int8_exact": a.int8_exact, "public": public, "target": target.to_json(), "effective": eng.effective})

    layers = PH.layers(eng.model, pouw.linear_types())
    ids = {id(m): w for w, m in enumerate(layers)}
    names = {ids[id(m)]: n for n, m in eng.model.named_modules() if id(m) in ids}
    offs, u_step = static_layout([int(m.weight.shape[0]) for m in layers], t_max)
    rec.step("wrap", wrap_counts({names[w]: m for w, m in enumerate(layers)}))

    ref_spans, ref_tap, restore = schedule_tap(eng.runner)
    try:
        reference = G.generate(eng.llm, prompts, a.max_tokens)
    finally:
        restore()
    rec.step("reference", {"requests": len(reference), "tokens": sum(len(r["token_ids"]) for r in reference),
                           "schedule": {"steps": len(ref_spans), "rows": [sum(k for _, k in x) for x in ref_spans], **ref_tap}})

    window = settings.RESEARCH_RUN_ID or "local"
    key = os.urandom(64)
    queue = CR.queue_of(ref_spans, s_max, t_max)
    q_commit = commit_queue(window, key, queue)
    rec.step("queue", {"root": q_commit["root"].hex(), "requests": len(queue["requests"]) - 1,
                       "lifetimes": queue["requests"][1:], "assigned_per_step": [int((r > 0).sum()) for r in queue["slot_map"]],
                       "slot_leaves": q_commit["slot_leaves"], "request_leaves": q_commit["request_leaves"]})

    option = f"pouw:{PH.SCHEME}"
    ctx = PO.Context(stage="serve", enabled=(option,), settings={"pouw": {"coins": Challenger(a.epoch, verifier.coins), "retain_steps": 0,
                                                                         "transcript_dir": str(out)}})
    t0 = time.perf_counter()
    comp = PO.install(eng.model, ctx)
    h1 = pouw.h1_run(option)
    rec.step("install", {"s": round(time.perf_counter() - t0, 1), "option": option, "salt": h1.salt.hex(),
                         "salt_bytes": len(h1.salt), "weights_root": h1.commits.get("weights_root"),
                         "coins": h1.commits.get("coins"), "salt_opening": h1.commits.get("salt_opening"),
                         "weight_roots": {names[w]: r.hex() for w, r in sorted(h1.weight_roots.items())}, "roots": PH.ROOTS})

    t0 = time.perf_counter()
    weights = b_tables(pool, public["model"], layers, names, key)
    reg_fields = {"window": window, "leaves": "weights root, queue root"}
    registration = CM.step_root(CM.domain(reg_fields, 2, "registration"), [weights["root"], q_commit["root"]])
    rec.step("weights", {"s": round(time.perf_counter() - t0, 1), "weights_root": weights["root"].hex(),
                         "b_roots": {names[w]: r.hex() for w, r in enumerate(weights["roots"])},
                         "registration": registration.hex(), "registration_fields": reg_fields})
    dev = layers[0].weight.device
    st = {"on": True, "step": -1, "last": None, "order_ok": True, "refused": []}
    stash: dict[int, tuple] = {}
    calls: list[dict] = []
    data: list[dict] = []
    a_check = {"equal": 0, "mismatch": 0, "untapped": 0}
    orig_linear, orig_record = int8_ops.linear, h1.record

    def sync() -> None:
        if dev.type == "cuda":
            torch.cuda.synchronize()

    pending: list[tuple] = []
    round_log: list[dict] = []
    if live:
        from verity.core.protocols.pouw import window as Wd
        from verity.core.protocols.pouw.circuit import h1tile as HT
        from verity.core.protocols.sampled_proofs import challenger as Ch
        live_salt = cs.salt(registration)
        rounds = cs.rounds

    def receipts(i: int, slots, salt: bytes) -> None:
        c = calls[i]
        sync()
        t = time.perf_counter()
        rc, lu = RC.gated_receipts(RC.Call(slots, layers[c["linear"]].weight), data[i]["admit"], salt, c["base"],
                                   bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"]), cubin, chunk=chunk_of(c["k"]))
        sync()
        c.update(units=int(rc.shape[0]), live_units=len(lu), receipt_words=int(rc.shape[1]),
                 receipts_s=round(time.perf_counter() - t, 3), salt=salt.hex())
        data[i]["receipts"], data[i]["live"] = rc, lu

    def close_round(s: int) -> None:
        """Round s (step s) under live coins: its root over the step's A root and admit root, recorded before its coin
        opens; then the step's receipts under the round's 64-byte salt (`h1tile.live_salt`)."""
        st = step_roots(window, s, [calls[i] for i, _ in pending])
        opening = rounds.record(st["root"])
        salt = HT.live_salt(cs.commitment, opening.coin, s, st["root"])
        round_log.append({"round": s, "root": st["root"].hex(), "fields": st["fields"], "a_step": st["a_step"].hex(),
                          "admit_step": st["admit_step"].hex(), "opening": opening.to_json()})
        for i, slots in pending:
            receipts(i, slots, salt)
        pending.clear()

    def table(s: int, w: int, xq, sa: np.ndarray) -> None:
        """Call (s, w): its T-slot A table from the step's m real rows `xq` [m, K] (on the device) and scales, its admit
        words; its receipts now (seeded) or once the step's round closes (live)."""
        m, k = int(xq.shape[0]), int(layers[w].weight.shape[1])
        n = int(layers[w].weight.shape[0])
        name = names[w]
        layer, site = site_of(name)
        slots = torch.zeros(t_max, k, dtype=torch.int8, device=dev)
        slots[:m] = xq
        scales = np.zeros(t_max, dtype=np.float32)
        scales[:m] = sa
        fields = {"window": window, "step": s, "linear": w, "module": name, "slots": t_max, "k": k, "schema": CM.SCHEMA}
        dom = CM.domain(fields, t_max, "a-table")
        q = slots.cpu().numpy()
        rows = CM.s8_rows(q, scales)
        root_a = CM.s8_table(dom, key, f"a/{s}/{w}", q, scales).root
        bits = CR.queue_admit(queue, s, rows)
        ws = CR.words(bits)
        s_fields = {"window": window, "step": s, "linear": w, "module": name, "slots": t_max, "schema": CM.SCHEMA}
        s_dom = CM.domain(s_fields, len(ws), "admit")
        s_lv = CM.levels(s_dom, [CM.leaf(s_dom, key, f"s/{s}/{w}", q, x)[0] for q, x in enumerate(CR.word_rows(ws))])
        calls.append({"step": s, "linear": w, "module": name, "layer": layer, "site": site, "k": k, "n": n, "slots": t_max,
                      "base": s * u_step + offs[w], "units": (t_max // 32) * (n // 32), "root_a": root_a.hex(),
                      "root_b": weights["roots"][w].hex(), "a_domain": dom.domain_id.hex(), "a_fields": fields,
                      "root_admit": s_lv[-1][0].hex(), "admit_fields": s_fields, "selected": int(bits.sum()),
                      "scale_only": len(CR.scale_only(rows)), "junk": len(CR.junk(rows, bits)),
                      "hollow": len(CR.hollow(rows, CR.queue_select(queue, s)))})
        data.append({"m": m, "xq": slots[:m].cpu().numpy(), "sa": scales[:m].copy(), "receipts": None, "admit": bits,
                     "words": ws, "s_levels": s_lv, "prefix": bool(np.array_equal(bits, np.arange(t_max) < m))})
        if not live:
            receipts(len(calls) - 1, slots, h1.salt)
            return
        pending.append((len(calls) - 1, slots))
        if w == len(layers) - 1:
            close_round(s)

    def tap(x2, w, s_w):
        xq, s_a, y = orig_linear(x2, w, s_w)
        if st["on"]:
            stash[w.data_ptr()] = (xq, s_a)
        return xq, s_a, y

    def record(w: int, xq, sa, n: int, y=None):
        call = orig_record(w, xq, sa, n, y)
        served = stash.pop(layers[w].weight.data_ptr(), None)
        sa = np.asarray(sa, dtype=np.float32).reshape(-1)
        if served is None:
            a_check["untapped"] += 1
            rows = torch.from_numpy(np.asarray(xq)).to(dev)
        else:
            same = (np.array_equal(served[0].cpu().numpy(), xq)
                    and np.array_equal(served[1].reshape(-1).cpu().numpy(), sa))
            a_check["equal" if same else "mismatch"] += 1
            rows = served[0]
        if w == 0:
            st["step"] += 1
        elif st["last"] is None or w != st["last"] + 1:
            st["order_ok"] = False
        st["last"] = w
        if st["step"] >= s_max or call.m > t_max:
            st["refused"].append({"step": st["step"], "linear": w, "m": call.m})
            return call
        table(st["step"], w, rows, sa)
        return call

    int8_ops.linear, h1.record = tap, record
    srv_spans, srv_tap, restore = schedule_tap(eng.runner)
    t0 = time.perf_counter()
    try:
        served = G.generate(eng.llm, prompts, a.max_tokens)
    finally:
        int8_ops.linear = orig_linear
        restore()
        st["on"] = False
    srv_queue = CR.queue_of(srv_spans, s_max, t_max) if len(srv_spans) <= s_max else None
    queue_check = {"served_steps": len(srv_spans), **{f"served_{k}": v for k, v in srv_tap.items()},
                   "served_is_committed": srv_queue is not None
                   and np.array_equal(srv_queue["slot_map"], queue["slot_map"])
                   and srv_queue["requests"] == queue["requests"]}
    serve_s = time.perf_counter() - t0
    done = st["step"] + 1
    complete = st["last"] == len(layers) - 1
    t0 = time.perf_counter()
    if pending:
        close_round(calls[pending[0][0]]["step"])
    for s in range(done, s_max):
        for w in range(len(layers)):
            table(s, w, torch.zeros(0, int(layers[w].weight.shape[1]), dtype=torch.int8, device=dev), np.zeros(0, np.float32))
    static_ok = ([(c["step"], c["linear"], c["base"], c["units"]) for c in calls]
                 == [(s, w, s * u_step + offs[w], (t_max // 32) * (int(layers[w].weight.shape[0]) // 32))
                     for s in range(s_max) for w in range(len(layers))])
    padded_superset = {"units": sum(c["units"] for c in calls), "rows": len(calls) * t_max,
                       "macs": sum(t_max * c["n"] * c["k"] for c in calls)}
    tl = CR.tally(calls, [c["selected"] for c in calls], s_max, SITES)
    live_u = sum(c["live_units"] for c in calls)
    noised = {"units": live_u, "rows": sum(32 * len(CR.live_blocks(d["admit"])) for d in data),
              "macs": sum(c["live_units"] * 32 * 32 * c["k"] for c in calls)}
    noise_blocks = sum(c["units"] * RC.layout(c["k"]).blocks for c in calls)
    noise_live = sum(c["live_units"] * RC.layout(c["k"]).blocks for c in calls)
    noise_masked = sum((c["n"] // 32) * sum(int(d["admit"][32 * ri:32 * ri + 32].sum()) * RC.layout(c["k"]).rb
                                            + 32 * RC.layout(c["k"]).cb for ri in CR.live_blocks(d["admit"]))
                       for c, d in zip(calls, data))
    credit = {"credited": {"rows": tl["rows"], "macs": tl["macs"], **CR.published(tl["macs"]),
                           "frac_of_padded_macs": round(tl["macs"] / max(1, padded_superset["macs"]), 6)},
              "padded": {"rows": noised["rows"] - tl["rows"], "macs": noised["macs"] - tl["macs"],
                         "what": "the non-admitted rows of live units: noised and receipted, shown, never credited"},
              "dead": {"units": padded_superset["units"] - live_u,
                       "frac_of_units": round(1 - live_u / max(1, padded_superset["units"]), 6),
                       "macs": padded_superset["macs"] - noised["macs"],
                       "what": "units whose 32 slots are all the canonical empty row: no noise, no receipt, an emptiness "
                               "check (their admit words, always proved) and the canonical empty receipt leaf"},
              "noise": {"blocks_all_units": noise_blocks, "blocks_live_units": noise_live,
                        "frac_live": round(noise_live / max(1, noise_blocks), 6),
                        "blocks_if_rows_masked": noise_masked, "frac_if_rows_masked": round(noise_masked / max(1, noise_blocks), 6),
                        "rows_masked": "an estimate: a template that skips a non-admitted row's noise blocks inside a live "
                                       "unit still derives every column's (c (K + 256) bits), so it costs "
                                       "(admitted rows RB + 32 CB) / (32 RB + 32 CB) of a unit's noise"},
              "padded_superset": padded_superset, "noised": noised, "tally": tl}
    rec.step("serve", {"steps_served": done, "steps": s_max, "last_step_complete": complete, "order_ok": st["order_ok"],
                       "refused": st["refused"], "calls": len(calls), "units": sum(c["units"] for c in calls),
                       "units_static": s_max * u_step, "static": static_ok, "u_step": u_step, "s": round(serve_s, 1),
                       "empty_steps_s": round(time.perf_counter() - t0, 1), "a_served_vs_recorded": a_check,
                       "receipts_s": round(sum(c["receipts_s"] for c in calls), 1), "live_units": live_u,
                       "admit_is_served_prefix": all(d["prefix"] for d in data),
                       "scale_only_rows": sum(c["scale_only"] for c in calls),
                       "queue": {**queue_check, "junk_rows": sum(c["junk"] for c in calls),
                                 "hollow_rows": sum(c["hollow"] for c in calls)},
                       "real_rows_per_step": [max((d["m"] for c, d in zip(calls, data) if c["step"] == s), default=0)
                                              for s in range(s_max)]})
    equal = [s_["token_ids"] == r["token_ids"] for s_, r in zip(served, reference)]
    rec.step("tokens", {"equal": all(equal), "per_prompt": equal, "served": served, "reference": reference})

    t0 = time.perf_counter()
    rtasks = []
    for c, d in zip(calls, data):
        fields = {"window": window, "step": c["step"], "linear": c["linear"], "module": c["module"], "base": c["base"],
                  "units": c["units"], "words": c["receipt_words"], "schema": CM.SCHEMA}
        dom = CM.domain(fields, c["units"])
        c.update(receipt_domain=dom.domain_id.hex(), receipt_fields=fields)
        rtasks.append((dom, CM.receipt_rows(d["receipts"])))
    rl = pool.map(CM.leaves, [(dom.binding, dom.count, key, f"r/{c['step']}/{c['linear']}", 0, rows)
                              for c, (dom, rows) in zip(calls, rtasks)], chunksize=1)
    levels = []
    for c, (dom, _), leaves in zip(calls, rtasks, rl):
        lv = CM.levels(dom, CM.split(leaves))
        levels.append(lv)
        c["receipt_root"] = lv[-1][0].hex()
    del rtasks, rl
    rec.step("receipt_trees", {"calls": len(calls), "leaves": sum(c["units"] for c in calls),
                               "s": round(time.perf_counter() - t0, 1)})
    comp.close()

    limbs = [v for row in tl["step_site"] for v in row] + [tl["site"][x] for x in SITES]
    limbs.append(credit["credited"]["level"] or 0)
    t_fields = {"window": window, "steps": s_max, "sites": list(SITES), "limbs": "per (step, site), per site, level",
                "schema": CM.SCHEMA}
    t_dom = CM.domain(t_fields, len(limbs), "tally")
    t_root = CM.levels(t_dom, [CM.leaf(t_dom, key, "t", i, int(v).to_bytes(2, "little"))[0]
                               for i, v in enumerate(limbs)])[-1][0]
    credit["tally_tree"] = {"root": t_root.hex(), "fields": t_fields, "limbs": len(limbs),
                            "max_limb": max(limbs), "limbs_16_bit": max(limbs) < CR.LIMB}
    credit["proof_plan"] = CR.proof_plan(len(calls), t_max, s_max, SITES, s_max * u_step)
    rec.step("credit", credit)

    by_step: dict[int, list[dict]] = {}
    for c in calls:
        by_step.setdefault(c["step"], []).append(c)
    steps = [step_roots(window, s, by_step[s]) for s in sorted(by_step)]
    w_fields = {"window": window, "steps": len(steps)}
    a_window = CM.step_root(CM.domain(w_fields, len(steps), "a-window"), [st["a_step"] for st in steps])
    a_roots = {"a_step": [st["a_step"].hex() for st in steps], "admit_step": [st["admit_step"].hex() for st in steps],
               "a_window": a_window.hex(), "a_window_fields": w_fields}
    coins = {"mode": a.coins, "ok": True, **(cs.record() if live else {"source": "in-process", "label": CO.IN_PROCESS})}
    gate_picks = {"from": "the prover's entropy (os.urandom): seeded, not certified"}
    rng = random.Random(os.urandom(16))
    if live:
        roots = [bytes.fromhex(r["root"]) for r in round_log]
        openings = [Ch.LiveOpening.from_json(r["opening"]) for r in round_log]
        commitment = Ch.LiveCommitment.from_json(cs.commitment.to_json())
        draw_after = receipts_root(window, calls)
        draw = cs.draw(draw_after)
        cs.close()
        rng = gate_rng(commitment, draw.coin, draw_after)
        gate_picks = {"from": f"the draw coin ({cs.label}), opened after the receipts root", "domain": GATE_DRAW_DOMAIN,
                      "after": draw_after.hex(), "what": "the harness's self-check of the receipts, not the window's draw"}
        recomputed = [steps[r["round"]]["root"] == root and r["a_step"] == steps[r["round"]]["a_step"].hex()
                      and {c["salt"] for c in by_step[r["round"]]}
                      == {Wd.round_key(commitment, o.coin, r["round"], root).stream(0, 64).hex()}
                      for r, root, o in zip(round_log, roots, openings)]
        coins.update(scheme=Wd.LIVE_SCHEME, commitment=cs.commitment.to_json(), salt_coin=live_salt.to_json(),
                     draw_coin=draw.to_json(), draw_after=draw_after.hex(),
                     draw_reason=Ch.live_reason(commitment, draw, Wd.DRAW, 0, draw_after),
                     rounds=round_log, rounds_reason=Wd.rounds_reason(commitment, roots, openings),
                     rounds_cover_steps=[r["round"] for r in round_log] == list(range(s_max)),
                     roots_and_keys_recomputed=all(recomputed), coin_bytes=len(openings[0].coin) if openings else None,
                     release_log=list(getattr(rounds, "log", ())),
                     rule="round s = step s: its root is the frame-v3-sha512 tree over [the step's A root, its admit "
                          "root] (domain identity_digest_sha512('verity/pouw/served-h1/round/v0', {window, step, leaves})); "
                          "the A root (roots.a_step) is the tree over the step's A-table roots, AROW's row tables, in "
                          "linear order ('a-step', {window, step, linears}), the admit root likewise ('admit-step'); it "
                          "is recorded through sampled_proofs.challenger.LiveRecord before round s's pouw/round coin "
                          "opens; the step's 64-byte H1 salt is h1tile.live_salt = "
                          "window.round_key(commitment, coin_s, s, root_s).stream(0, 64), the tile's coin port; each "
                          "tile derives its own noise from it, its public position and its roots, inside the tile")
        coins["ok"] = bool(coins["rounds_reason"] is None and coins["draw_reason"] is None and coins["rounds_cover_steps"]
                           and all(recomputed))
        coins["verify"] = ("the verifier's check: coins.py check --challenger-out <the session as the gate publishes it> "
                           "--ledger <each head of the auditor's ledger infra published> --pins <the host's pins log "
                           "infra published> --main <the verifier's git rev-list --first-parent origin/main> "
                           "--manifest manifest.json "
                           "--key-record <the challenger's "
                           "challenger-key/v1 record> --period <the window's period> --root <the window's root>, from the "
                           "challenger's own record, never this one" if cs.source == "off-node"
                           else "none: the challenger ran in the prover's process")
        rec.step("coins", {k: v for k, v in coins.items() if k not in ("rounds", "release_log")})

    manifest = {"schema": SCHEMA, "scheme": SCHEME, "tile": [RC.R, RC.C_], "framing": FRAMING, "not_covered": NOT_COVERED,
                "window": window, "public": public, "epoch": a.epoch, "option": option, "commitment": h1.commits,
                "coins": coins, "salt": h1.salt.hex(),
                "salt_rule": ("live: each call's salt is its round's (coins.rule); manifest.salt is the option's, unused"
                              if live else
                              "pouw_h1.program_salt: SHA-512(b'verity/pouw/h1/program-salt/v1\\0' || the epoch's 32-byte "
                              "salt), the epoch's salt audit.Epoch.start's from the challenger's salt coin opened after the "
                              "weights' root (pouw.weights_root over every layer's pouw_h1.weight_root)"),
                "roots": {"a": "the call's a-table root: AROW's row table (verity.ml.commitments.row_table) over its T slot "
                               "rows, under this window's per-call domain and salts",
                          "a_step_rule": "per step, the tree over its calls' a-table roots in linear order (domain "
                                         "'a-step'), committed before the step's coin; AROW's step root replaces it once "
                                         "AROW computes it at serve time",
                          "a_window_rule": "the window's A table: the tree over the steps' A roots (domain 'a-window')",
                          **a_roots,
                          "b": "the linear's B table root: AROW's row-table format over its N weight rows (K s8 bytes, "
                               "then the row's f32 weight scale), domain 'b-table' ({model, layer, module, rows, k, "
                               "schema}), labels b/<w>",
                          "b_roots": [r.hex() for r in weights["roots"]], "weights_root": weights["root"].hex(),
                          "weights_rule": "the tree over the linears' B roots in linear order ('weights', {model, layers})",
                          "queue": {k: (v.hex() if isinstance(v, bytes) else v) for k, v in q_commit.items()
                                    if k not in ("s_levels", "r_levels")},
                          "queue_rule": "the tree over [the slot map's root ('queue-slots', {window, steps, slots}: one "
                                        "leaf per step and 16 slots, 16 u16 LE request indices, 0 for none), the requests' "
                                        "root ('queue-requests', {window, requests}: (start, stop) u16 LE per index, the "
                                        "null request first, padded to S T + 1)] ('queue', {window, leaves})",
                          "registration": registration.hex(), "registration_fields": reg_fields,
                          "registration_rule": "the tree over [weights_root, the queue's root] ('registration'): live, "
                                               "the challenger's salt coin opens against it before every other coin"},
                "positions": "call (s, w) at base s U + off(w), U = sum_w (T / 32)(N_w / 32), off(w) the sum over linears "
                             "before w; unit u = slot rows 32 (u // (N_w / 32)) .., weight rows 32 (u % (N_w / 32)) ..",
                "u_step": u_step, "offsets": offs, "units": s_max * u_step,
                "leaves": {"a": "slot row i: K s8 bytes then its f32 scale LE; an empty slot (the canonical empty row) "
                                "zero bytes and scale 0",
                           "b": "weight row j: K s8 bytes then its f32 weight scale LE",
                           "s": "admit word q: slots 16 q .. 16 q + 15's admit bits (bit b slot 16 q + b), u16 LE, "
                                "PouwQueueAdmit's word on the step, the slots' queue entries and their rows' scales",
                           "r": "unit u: its receipt's 2 (KS + m) u32 words LE (per set y then z); a dead unit's the "
                                "canonical empty receipt, zero words",
                           "t": "tally limb i: u16 LE, the per (step, site) limbs in step-major order, the per-site limbs, "
                                "the level",
                           "schema": CM.SCHEMA, "tree": "frame-v3-sha512, one per call and kind (one tally tree)",
                           "domain": "identity_digest_sha512('verity/pouw/served-h1/<a-table|admit|receipts|tally|a-step|"
                                     "admit-step|round|a-window|b-table|weights|queue-slots|queue-requests|queue|"
                                     "registration>/v0', "
                                     "fields), owner -1, ranks 0 .. count - 1",
                           "salts": "SHA-512(SALT_TAG || key || u32be len(label) || label || u64be i || u8 c), c = 0..2, "
                                    "labels a/<s>/<w>, s/<s>/<w>, r/<s>/<w>, t, b/<w>, q and qr; the key is the prover's"},
                "credit": {**{k: v for k, v in credit.items() if k != "tally"},
                           "rule": "PADCREDIT round 2 with RT10's fixes: only admitted slots are credited, a tile's iff "
                                   "PouwH1Tile_v4's live bit (the OR of its 32 admit bits) is set; slot i of step "
                                   "t is admitted iff the request queue (committed before the first coin) assigns it a "
                                   "request live at t and its row's scale is nonzero (pouw.circuit.admit.PouwQueueAdmit: "
                                   "the row can withhold admission, never grant it; RT-ALGO-4), written by an "
                                   "always-proved admit unit into its own leaf, which refuses a slot not admitted whose row "
                                   "is not the canonical empty row and a slot the queue selects whose row is; the admit "
                                   "leaves are committed (live) in the round "
                                   "root before the round's coin; the tally is 16-bit limbs, committed and not proved "
                                   "until a tally Definition exists; Pub = point(level(floor_point(Cr)))"},
                "calls": calls, "tokens_equal": all(equal)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1, default=str) + "\n")
    rec.step("prover", prover(a, out, key, calls, data))
    try:
        rec.step("gate", {**gate(a, calls, data, levels, layers, public["model"], names, queue, weights, key, E, t_max, done,
                                 rng), "picks": gate_picks})
    except Exception as e:  # noqa: BLE001
        rec.step("gate", {"ok": False, "error": f"{type(e).__name__}: {e}"[:4000]})
    s = rec.doc["steps"]["serve"]
    rec.doc["ok"] = bool(all(equal) and calls and not a_check["mismatch"] and not a_check["untapped"] and not s["refused"]
                         and s["order_ok"] and s["last_step_complete"] and s["static"] and rec.doc["steps"]["gate"]["ok"]
                         and coins["ok"] and s["admit_is_served_prefix"] and not s["scale_only_rows"]
                         and s["queue"]["served_is_committed"] and not s["queue"]["junk_rows"]
                         and not s["queue"]["hollow_rows"]
                         and credit["tally_tree"]["limbs_16_bit"]
                         and credit["credited"]["pub_macs"] <= credit["credited"]["macs"])
    rec.save()
    pool.close()
    print(f"SERVED-H1 {'OK' if rec.doc['ok'] else 'FAIL'} calls={len(calls)} units={s['units']} steps={done}/{s_max} "
          f"tokens_equal={all(equal)} static={s['static']} gate={rec.doc['steps']['gate']['ok']} "
          f"coins={rec.doc['coins']}", flush=True)
    return rec.doc


def tile_admit(c: dict, d: dict, key: bytes, ri: int, words: list[int] | None = None) -> np.ndarray | None:
    """Row block ri's `PouwH1Tile_v4` admit input: its words (`credit.tile_words`; the committed ones by default) opened
    from call c's admit tree against its root, as the tile's 32 bits; None if one doesn't open."""
    ws = d["words"] if words is None else words
    dom = CM.domain(c["admit_fields"], len(d["words"]), "admit")
    label, root = f"s/{c['step']}/{c['linear']}", bytes.fromhex(c["root_admit"])
    for q in CR.tile_words(ri):
        if not CM.verify(dom, root, q, CM.leaf(dom, key, label, q, CR.word_rows([ws[q]])[0])[0], CM.path(d["s_levels"], q)):
            return None
    return CR.bits_of([ws[q] for q in CR.tile_words(ri)])


def gate(a, calls: list[dict], data: list[dict], levels: list, layers: list, model: dict, names: dict, queue: dict,
         weights: dict, key: bytes, E, t_max: int, done: int, rng: random.Random) -> dict:
    """Units picked on `rng` (`gate_rng`'s, from the draw coin, when live) against the gated `h1tile.reference`, and the
    controls: per site a live and a dead unit of real steps' calls, then units of empty steps (dead)."""
    import torch
    import receipts as RC

    picks = []
    for site in SITES:
        real = [i for i, c in enumerate(calls) if c["site"] == site and c["step"] < done]
        for _ in range(a.gate_per_site if site != "down_proj" else max(1, a.gate_per_site // 2)):
            if real:
                i = rng.choice(real)
                picks.append((i, rng.choice(data[i]["live"])))
        partly = [i for i in real if 0 < len(data[i]["live"]) < calls[i]["units"]]
        if partly:
            i = rng.choice(partly)
            picks.append((i, rng.choice(sorted(set(range(calls[i]["units"])) - set(data[i]["live"])))))
    empty = [i for i, c in enumerate(calls) if c["step"] >= done]
    picks += [(i, rng.randrange(calls[i]["units"])) for i in rng.sample(empty, min(2, len(empty)))]
    units, ok, b_tabs = [], True, {}
    w_dom = CM.domain(weights["fields"], len(weights["roots"]), "weights")
    w_lv = CM.levels(w_dom, weights["roots"])
    for i, u in picks:
        c, d = calls[i], data[i]
        weight = layers[c["linear"]].weight
        slots = padded(torch.from_numpy(d["xq"]), t_max)
        scales = np.zeros(t_max, dtype=np.float32)
        scales[:d["m"]] = d["sa"]
        call = RC.Call(slots, weight.cpu())
        root_a, root_b = bytes.fromhex(c["root_a"]), bytes.fromhex(c["root_b"])
        salt = bytes.fromhex(c["salt"])
        ri, cj = call.units[u]
        a_rows = CM.s8_rows(slots.numpy(), scales)
        bits = CR.queue_admit(queue, c["step"], a_rows)
        chosen = CR.queue_select(queue, c["step"])
        live = bool(bits[32 * ri:32 * ri + 32].any())
        s_dom = CM.domain(c["admit_fields"], t_max // CR.WORD, "admit")
        s_label = f"s/{c['step']}/{c['linear']}"
        tile_bits = tile_admit(c, d, key, ri)
        if tile_bits is None:
            raise ValueError(f"call {i}: row block {ri}'s admit words don't open against its admit root")
        forged = list(d["words"])
        forged[rng.choice(list(CR.tile_words(ri)))] ^= 1 << rng.randrange(CR.WORD)
        z, ungated = RC.reference(call, salt, c["base"], root_a, root_b, u)
        want = RC.reference(call, salt, c["base"], root_a, root_b, u, gated=tile_bits)[1]
        got = [int(v) for v in d["receipts"][u]]
        y = E.int_mm(slots[32 * ri:32 * ri + 32].to(weight.device), weight[32 * cj:32 * cj + 32]).cpu().to(torch.int64)
        z_ok = [v & 0xFFFFFFFF for v in y.flatten().tolist()] == z
        nxt = RC.reference(call, salt, c["base"] + 1, root_a, root_b, u)[1] if live else None

        admit_root = CM.levels(s_dom, [CM.leaf(s_dom, key, s_label, q, x)[0]
                                       for q, x in enumerate(CR.word_rows(CR.words(bits)))])[-1][0]
        sel = {"live": live, "admit_root_rebuilt": admit_root == bytes.fromhex(c["root_admit"]),
               "admit_words_committed": CR.words(bits) == d["words"], "no_junk_row": not CR.junk(a_rows, bits),
               "no_hollow_row": not CR.hollow(a_rows, chosen),
               "tile_admit_is_tally_bits": bool(np.array_equal(tile_bits, bits[32 * ri:32 * ri + 32])),
               "tampered_tile_admit_refused": tile_admit(c, d, key, ri, forged) is None}
        free = [r_ for r_ in range(32 * ri, 32 * ri + 32) if not bits[r_]]
        if free:
            jr = rng.choice(free)
            junk_rows = a_rows.copy()
            junk_rows[jr, :-4] = np.frombuffer(os.urandom(junk_rows.shape[1] - 4), dtype=np.uint8) | 1
            junk_rows[jr, -4:] = np.frombuffer(np.float32(0.5).tobytes(), dtype=np.uint8)
            sel.update(junk_row_refused=CR.junk(junk_rows, bits) == [jr],
                       junk_row_not_admitted=not CR.queue_admit(queue, c["step"], junk_rows)[jr]
                       and int(bits.sum()) == c["selected"])
        taken = [r_ for r_ in range(32 * ri, 32 * ri + 32) if chosen[r_]]
        if taken:
            hr = rng.choice(taken)
            hollow_rows = a_rows.copy()
            hollow_rows[hr] = 0
            sel.update(hollow_row_refused=CR.hollow(hollow_rows, chosen) == [hr],
                       hollow_row_not_admitted=not CR.queue_admit(queue, c["step"], hollow_rows)[hr]
                       and int(CR.queue_admit(queue, c["step"], hollow_rows).sum()) == c["selected"] - 1)
            zero_rows = a_rows.copy()
            zero_rows[taken, :-4] = 0
            zero_rows[taken, -4:] = np.frombuffer(np.float32(0.5).tobytes(), dtype=np.uint8)
            zb = CR.queue_admit(queue, c["step"], zero_rows)
            sel.update(zero_byte_rows_admitted=bool(zb[taken].all()) and set(taken) <= set(CR.scale_only(zero_rows)))
        if live:
            sel.update(next_position_differs=nxt != got, live_receipt_is_ungated=got == ungated)
        else:
            sel.update(dead_receipt_is_empty_receipt=not any(got),
                       dead_rows_canonical_empty=not a_rows[32 * ri:32 * ri + 32].any(),
                       skipped_noise_was_work=any(ungated) and ungated != got)
        dom = CM.domain(c["receipt_fields"], c["units"])
        root_r = bytes.fromhex(c["receipt_root"])
        rows = CM.receipt_rows(d["receipts"][u:u + 1])[0]
        label = f"r/{c['step']}/{c['linear']}"
        leaf = CM.leaf(dom, key, label, u, bytes(rows))[0]
        sibs = CM.path(levels[i], u)
        flipped = bytearray(rows)
        flipped[rng.randrange(len(flipped))] ^= 1 << rng.randrange(8)
        leaf_f = CM.leaf(dom, key, label, u, bytes(flipped))[0]
        a_dom = CM.domain(c["a_fields"], t_max, "a-table")
        a_label = f"a/{c['step']}/{c['linear']}"
        q = slots.numpy()
        a_tab = CM.s8_table(a_dom, key, a_label, q, scales)
        a_root = a_tab.root
        ranks = range(32 * ri, 32 * ri + 32)
        opening = a_tab.open(ranks)
        neg_b = (-weight.cpu()[32 * cj:32 * cj + 32].to(torch.int16)).clamp(-128, 127).to(torch.int8).numpy()
        q_forged = q.copy()
        q_forged[32 * ri:32 * ri + 32] = neg_b
        forged_rows = CM.s8_rows(q_forged, scales)

        def witness(rows_) -> dict:
            return {r_: (bytes(rows_[r_]), 8 * rows_.shape[1], CM.salt(key, a_label, r_)) for r_ in ranks}

        step_cs = [cc for cc in calls if cc["step"] == c["step"]]
        at = step_cs.index(c)
        s_dom = CM.domain({"window": c["a_fields"]["window"], "step": c["step"], "linears": len(step_cs)}, len(step_cs), "a-step")
        s_lv = CM.levels(s_dom, [bytes.fromhex(cc["root_a"]) for cc in step_cs])
        forged_root = CM.s8_table(a_dom, key, a_label, q_forged, scales).root
        forged_step = CM.step_root(s_dom, [forged_root if cc is c else bytes.fromhex(cc["root_a"]) for cc in step_cs])
        cmp38 = {"a_opens_against_step_root": CM.RT.verify_rows(root_a, a_dom.domain_id, t_max, opening, witness(a_rows))
                 and CM.verify(s_dom, s_lv[-1][0], at, root_a, CM.path(s_lv, at)),
                 "a_neg_b_refused": not CM.RT.verify_rows(root_a, a_dom.domain_id, t_max, opening, witness(forged_rows)),
                 "a_neg_b_moves_roots": forged_root != root_a and forged_step != s_lv[-1][0]}
        row = 32 * ri + rng.randrange(32)
        t_rows = a_rows.copy()
        t_rows[row, rng.randrange(t_rows.shape[1])] ^= 1
        a_tampered = CM.levels(a_dom, [CM.leaf(a_dom, key, a_label, r, bytes(x))[0] for r, x in enumerate(t_rows)])[-1][0]
        committed = CM.s8_rows(slots.numpy(), scales)
        useful = sum(int(np.frombuffer(bytes(x[-4:]), "<f4")[0] != 0) for x in committed)
        w = c["linear"]
        if w not in b_tabs:
            rows_b = b_rows(layers[w])
            b_dom = CM.domain(b_fields(model, w, names[w], c["n"], c["k"]), c["n"], "b-table")
            b_tabs[w] = (b_dom, rows_b, CM.s8_table(b_dom, key, f"b/{w}", rows_b[:, :-4].view(np.int8),
                                                    rows_b[:, -4:].copy().view("<f4").reshape(-1)))
        b_dom, rows_b, b_tab = b_tabs[w]
        b_ranks = range(32 * cj, 32 * cj + 32)
        b_open = b_tab.open(b_ranks)
        t_b = rows_b.copy()
        t_b[32 * cj + rng.randrange(32), rng.randrange(c["k"])] ^= 1

        def witness_b(rows_) -> dict:
            return {r_: (bytes(rows_[r_]), 8 * rows_.shape[1], CM.salt(key, f"b/{w}", r_)) for r_ in b_ranks}

        b_checks = {"b_root_rebuilt": b_tab.root == root_b,
                    "b_rows_open": CM.RT.verify_rows(root_b, b_dom.domain_id, c["n"], b_open, witness_b(rows_b)),
                    "tampered_b_row_refused": not CM.RT.verify_rows(root_b, b_dom.domain_id, c["n"], b_open, witness_b(t_b)),
                    "b_root_in_weights_root": CM.verify(w_dom, weights["root"], w, root_b, CM.path(w_lv, w))}
        r = {"step": c["step"], "linear": c["linear"], "module": c["module"], "k": c["k"], "unit": u,
             "position": c["base"] + u, "real_rows_in_tile": max(0, min(32, d["m"] - 32 * ri)),
             "receipt_equal_reference": got == want,
             "z_equal_served_product": z_ok, **sel,
             "receipt_opens": CM.verify(dom, root_r, u, leaf, sibs),
             "flipped_receipt_refused": not CM.verify(dom, root_r, u, leaf_f, sibs),
             "a_root_rebuilt": a_root == root_a, **cmp38, "tampered_a_row_changes_root": a_tampered != root_a,
             "tampered_row_was_empty": row >= d["m"], "served_rows_from_scales": useful == d["m"], **b_checks}
        r["ok"] = all(v for k_, v in r.items() if isinstance(v, bool) and k_ not in ("tampered_row_was_empty", "live"))
        ok &= r["ok"]
        units.append(r)
        print(f"[served-h1] gate {json.dumps(r)}", flush=True)
    return {"ok": bool(ok and units), "units": units}


def prover(a, out: Path, key: bytes, calls: list[dict], data: list[dict]) -> dict:
    """The prover's data: the table key, each call's real rows (the rest of its slots are empty), scales, admit words, live
    units and their receipts (a dead unit's is zero words), as SCRATCH/prover.tar for `serve.sh` to put in the store after
    the lease (OUT/prover with no --scratch)."""
    dest = Path(a.scratch) if a.scratch else out / "prover"
    dest.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=dest) as td:
        p = Path(td) / "prover"
        p.mkdir()
        (p / "prover.json").write_text(json.dumps({"key": key.hex(), "calls": [[c["step"], c["linear"], d["m"]]
                                                                              for c, d in zip(calls, data)],
                                                   "npz": "xq, sa: the real rows; words: admit words; live: live units; "
                                                          "receipts: the live units' rows, in that order"}) + "\n")
        for c, d in zip(calls, data):
            np.savez(p / f"s{c['step']:03d}w{c['linear']:03d}.npz", xq=d["xq"], sa=d["sa"],
                     words=np.array(d["words"], dtype=np.uint16), live=np.array(d["live"], dtype=np.int32),
                     receipts=d["receipts"][d["live"]])
        tar = dest / "prover.tar"
        with tarfile.open(tar, "w") as t:
            t.add(p, arcname="prover")
    h = hashlib.sha256()
    with tar.open("rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return {"bytes": tar.stat().st_size, "sha256": h.hexdigest(), "tar": str(tar),
            "put": "serve.sh, after the lease: OUT/prover-put.txt, else OUT/prover.tar"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--cubin", help="noise.cu built for this device (else hashlib, and no timing)")
    ap.add_argument("--scratch", default=None, help="where the prover's tar is built")
    ap.add_argument("--checkpoints", default="manifests/checkpoints.json")
    ap.add_argument("--base-role", default="LLAMA31_8B")
    ap.add_argument("--role", default="LLAMA31_8B_INT8")
    ap.add_argument("--repo", default="verity-int8/Llama-3.1-8B-INT8")
    ap.add_argument("--hf-home", default=settings.HF_HOME)
    ap.add_argument("--at", required=True, help="resolved_at of a new int8 pin entry (UTC, ISO 8601)")
    ap.add_argument("--by", required=True, help="resolved_by of a new int8 pin entry")
    ap.add_argument("--int8-exact", default="all", choices=["linears", "attention", "all"])
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--rows", type=int, default=64, help="T, the public rows a step: vLLM's max_num_batched_tokens")
    ap.add_argument("--steps", type=int, default=0, help="S, the public steps (default max tokens + 2)")
    ap.add_argument("--gpu-mem", type=float, default=0.6, help="vLLM's gpu_memory_utilization: the rest is the side computation's")
    ap.add_argument("--epoch", type=int, default=1)
    ap.add_argument("--gate-per-site", type=int, default=2)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--skip-timing", action="store_true")
    ap.add_argument("--coins", default=None, choices=["seeded", "live"],
                    help="live: a round a step, its coin opened after its A tables' root (the proof system's challenger); "
                         "default live with a challenger address, else seeded")
    ap.add_argument("--challenger", default=None, metavar="HOST:PORT",
                    help=f"live: the challenger off the node (default ${CO.CHALLENGER_ENV}); none: in-process, not certified")
    ap.add_argument("--session", default=None, metavar="ID",
                    help=f"the challenger session's id (default ${CO.SESSION_ENV}, its secret ${CO.SECRET_ENV}); "
                         "development only, since its salt coin opens after the registration root, not a window's root")
    ap.add_argument("--gate", default=settings.get(CO.GATE_ENV), metavar="USER@HOST",
                    help=f"refused (default ${CO.GATE_ENV}): the gated served window of record is vllm's H1 path, "
                         "integrations/vllm/verity_vllm/protocols/pouw/h1/__init__.py (#3053)")
    return 0 if run(ap.parse_args(argv))["ok"] else 1


if __name__ == "__main__":
    try:
        rc = main()
    except BaseException:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        rc = 2
    sys.stdout.flush()
    sys.stderr.flush()
    # the in-process engine and the spawn pool can keep an exiting interpreter alive past the lease
    os._exit(rc)
