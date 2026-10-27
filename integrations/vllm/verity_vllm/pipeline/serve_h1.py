"""serve-openai's PoUW H1 arm (``--protocols pouw:eps1-sm120-h1fx1``): one H1 window a Commit window, tied to the server's
own Commit.

* **One Commit.** H1 builds no committer: its tap goes on the server's rows-sha512 collector (`acquire.h1_commit.tap`),
  which already acquires each linear's x_q and s_a (the bench row's int8 source) and each step's input ids (its step-input
  port), so ROOTS1 holds H1's A tables and input ids to the values the served window commits.
* **One window a Commit window.** `serve_commit.CommitWindows` opens the arm once the committer is installed, so H1's
  hooks sit on top of the committer's, and closes it first thing at the window's close, before the committer comes off.
  The H1 window's T is the engine's token budget (``--max-num-batched-tokens``) and its S the Commit window's steps
  (``--window-steps``); a window that closes early (idle, ``--window-gb``, ``--window-requests``) closes H1's with its
  remaining steps empty (`pouw_h1.H1Run.close`).
* **Once a server.** The period's declared Program over the served linears at (T, S) and its digest
  (`pouw_h1_served.declared`), and the weights' ``b/<K>`` registered under it (`pouw_h1_served.register_weights`), with a
  key of this process's: no auditor's registration (``model-key``) is read yet. The Program holds a node for each tile
  unit (`units`), S × slot(T)/32 × Σn/32, and declaring it and its digest take time linear in them before the server
  listens: about 7 minutes for Llama-3.2-1B at T 512 and S 16, about 3 minutes a step at T 2048. The server logs both.
* **Coins.** The epoch's in-process coins, one epoch a window (`pouw_h1_settings.from_engine`): such a window credits
  nothing (`pouw_h1.IN_PROCESS`). The auditor host's challenger needs one gate session a window, which isn't wired here
  (``challenger`` is refused).

Each window's outcome (`H1Windows.close`) is its ``windows.jsonl`` line's ``pouw_h1``: ROOTS1's verdict, the requests'
admission and gaps, the credit gate, the coins and the commitment's calls root; a window whose H1 failed to open or close
carries its error there instead, and the Commit serves on. ``serve.json``'s ``protocols`` is `H1Windows.describe`,
``summary.json``'s ``pouw_h1`` `H1Windows.summary` with the windows that failed. ``--protocols-config`` is the pouw
settings the server doesn't set itself, ``{"pouw": {...}}`` (`config_of`)."""
from __future__ import annotations

import dataclasses
import json
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.protocols.pouw.h1 import SCHEME

ENTRY = f"pouw:{SCHEME}"
#: the pouw settings the server sets for each window itself
OWN = ("rows", "steps", "runner", "served", "transcript_dir", "coins", "rounds")
#: the pouw settings that need a gate session a window, not wired on serve-openai yet
SESSIONED = ("challenger", "session", "gate", "period")


def config_of(value: str | None) -> dict[str, Any]:
    """``--protocols-config``: a JSON object, or a JSON file, ``{"pouw": {...}}``; empty for None. `UsageError` on any key
    the server sets itself (`OWN`), one it can't serve yet (`SESSIONED`), or a value `pouw.Settings` refuses."""
    from verity_vllm.config import UsageError
    from verity_vllm.protocols import load_settings
    from verity_vllm.protocols import pouw as PW

    if not value:
        return {}
    try:
        doc = load_settings(json.loads(value) if value.lstrip().startswith("{") else value)
    except (OSError, ValueError) as e:
        raise UsageError(f"--protocols-config {value[:80]}: {e}") from None
    if stray := sorted(set(doc) - {"pouw"}):
        raise UsageError(f"--protocols-config: {stray} are not the enabled protocols' (pouw)")
    pouw = doc.get("pouw", {})
    if own := sorted(set(pouw) & set(OWN)):
        raise UsageError(f"--protocols-config: the pouw settings {own} are the server's own (T, S, its runner, the window's "
                         "served tables and coins, one epoch a window)")
    if sessioned := sorted(set(pouw) & set(SESSIONED)):
        raise UsageError(f"--protocols-config: the pouw settings {sessioned} need one gate session a window, which serve-openai "
                         "doesn't reserve yet: its windows draw in-process coins and credit nothing")
    try:
        PW.Settings.of(pouw)
    except ValueError as e:
        raise UsageError(f"--protocols-config: {e}") from None
    return doc


def refusal(opts: Any) -> str | None:
    """Why ``opts`` can't serve ``--protocols`` (None when they can, or none is asked for)."""
    if not opts.protocols:
        return None
    if opts.protocols != ENTRY:
        return f"--protocols {opts.protocols}: serve-openai serves {ENTRY} (PoUW H1) only"
    if not opts.commit or opts.scheme != "rows-sha512" or not opts.required_manifest:
        return (f"--protocols {ENTRY} taps the server's rows-sha512 Commit (its int8 source and step-input port): it needs "
                "--commit, --scheme rows-sha512 and the bench row's --required-manifest")
    if opts.max_num_batched_tokens % 32:
        return f"--protocols {ENTRY}: H1's T, --max-num-batched-tokens {opts.max_num_batched_tokens}, is a multiple of 32"
    return None


def units(shapes: Any, rows: int, steps: int) -> int:
    """The tile units of the window Program `served.declared` builds over ``shapes`` (each ``(n, K)``) at T ``rows`` and S
    ``steps``: one node of its body each, so declaring it and its digest cost time linear in them."""
    from verity.core.protocols.pouw.circuit import h1window as HW

    from verity_vllm.protocols.pouw import h1 as PH

    r, c = PH.TILE
    return int(steps) * (HW.slot(int(rows), PH.TILE) // r) * sum(int(n) // c for n, _k in shapes)


class H1Windows:
    """The arm over ``model`` (the served `verity_vllm.LLM`'s) and the server's committer ``com``: T ``rows`` and S
    ``steps`` a window, ``config`` the pouw settings of ``--protocols-config``, ``engine`` the built ``vllm.LLM`` the
    runner and each window's coins come from, ``out`` where each window's transcript goes (``h1/w<k>/``). ``key`` is the
    weights' registration key (a fresh one when None); ``unit`` the Program's tile unit at each K (`h1window.tile_unit`
    when None)."""

    def __init__(self, model: Any, com: Any, rows: int, steps: int, *, engine: Any, out: Path,
                 config: Mapping[str, Any] | None = None, key: bytes | None = None, unit: Any = None) -> None:
        from verity.core.protocols.sampled_proofs.one_stage import partition as P

        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import hooks as PHK
        from verity_vllm.protocols.pouw.h1 import served as SV

        self.model, self.com, self.rows, self.steps, self.engine = model, com, int(rows), int(steps), engine
        self.out, self.config, self.unit = Path(out), dict(config or {}), unit
        self.layers = PHK.layers(model, pouw.linear_types())
        if not self.layers:
            raise ValueError(f"{ENTRY}: the served model has no int8 linear H1 records (serve an int8 precision)")
        shapes = [tuple(int(d) for d in m.weight.shape) for m in self.layers]
        modules = {w: f"linear {w}" for w in range(len(self.layers))} | pouw._named(model, self.layers)
        print(f"[serve-h1] declaring the window Program: {self.steps} steps of {len(shapes)} linears at T {self.rows}, "
              f"{units(shapes, self.rows, self.steps)} tile units", flush=True)
        t0 = time.perf_counter()
        self.program = SV.declared(shapes, modules, self.rows, self.steps, unit=unit)
        t1 = time.perf_counter()
        self.digest = P.program_digest(self.program)
        print(f"[serve-h1] declared in {t1 - t0:.1f} s, its digest {self.digest[:16]} in {time.perf_counter() - t1:.1f} s",
              flush=True)
        self.weights = SV.register_weights(self.digest, shapes, SV.weights_of(self.layers), key or os.urandom(32))
        self.tap: Any = None
        self.comp: Any = None
        self.run: Any = None
        self.window: int | None = None
        self.outcomes: list[dict[str, Any]] = []

    def describe(self) -> dict[str, Any]:
        """serve.json's ``protocols``: the entry, T and S, the period's Program digest and the weights' root."""
        return {"entry": ENTRY, "rows": self.rows, "steps": self.steps, "program": self.digest,
                "weights_root": self.weights.root().hex(), "weights_key": "local", "coins": "in-process",
                "linears": len(self.layers), "config": dict(self.config)}

    def open(self, window: int) -> None:
        """Window ``window``'s H1, on the open Commit window: the tap on the committer, a fresh `Served`, and the option
        installed with the epoch ``window``'s coins."""
        from verity_vllm import protocols as PO
        from verity_vllm.acquire import h1_commit as HC
        from verity_vllm.protocols import pouw
        from verity_vllm.protocols.pouw.h1 import served as SV

        if self.comp is not None:
            raise RuntimeError(f"{ENTRY}: window {window} opens while window {self.window}'s H1 is open")
        self.tap = HC.tap(self.com, self.model, self.layers, range(self.steps))
        served = SV.Served(unit=self.unit, weights=self.weights, tap=self.tap, window=os.urandom(64),
                           program=self.program, program_digest=self.digest)
        settings = {**self.config.get("pouw", {}), "rows": self.rows, "steps": self.steps, "served": served,
                    "transcript_dir": str(self.out / "h1" / f"w{window}")}
        ctx = PO.Context(stage="serve", enabled=(ENTRY,), settings={"pouw": settings})
        try:
            ctx = PO.from_engine(dataclasses.replace(ctx, filled=PO.engine_fills(ctx)), self.engine, epoch=window)
            self.comp = PO.install(self.model, ctx)
        except BaseException:
            self.tap.untap()
            self.tap = None
            raise
        self.run, self.window = pouw.h1_run(ENTRY), window

    def close(self, window: int) -> dict[str, Any] | None:
        """Window ``window``'s H1 closed and released (its hooks off, its manifest committed, the draw coin opened), the tap
        off the committer; its outcome. None when no H1 window is open."""
        if self.comp is None:
            return None
        run = self.run
        try:
            self.comp.close()
        finally:
            self.comp = self.run = None
            self.tap.untap()
        d, c = run.details, run.commits
        roots1, gate = d.get("roots1"), d.get("gate", {})
        out = {"entry": ENTRY, "window": window, "rows": self.rows, "steps": self.steps,
               "steps_served": d.get("steps_served"), "roots1": roots1, "requests": d.get("requests"),
               "static": d.get("static"), "u": d.get("u"), "refused": run.refused[:8],
               "gate": {k: gate.get(k) for k in ("coins", "gate_passed", "refused", "padded", "units")},
               "coin_source": c.get("coin_source"), "calls_root": c.get("calls_root"), "weights_root": c.get("weights_root"),
               "served_refused": list(run.served.refused[:8]) if run.served is not None else None,
               "tied": bool(roots1["tied"]) if roots1 is not None and d.get("steps_served") else None}
        self.outcomes.append(out)
        return out

    def summary(self) -> dict[str, Any]:
        """summary.json's ``pouw_h1``: windows closed, tied and refused at ROOTS1 (a window that served no step, whose
        tables H1 never registers, is neither), and the rows the gate passed."""
        return {"windows": len(self.outcomes), "tied": sum(o["tied"] is True for o in self.outcomes),
                "untied": [o["window"] for o in self.outcomes if o["tied"] is False][:20],
                "gate_passed_rows": sum(int(o["gate"]["gate_passed"]["rows"]) for o in self.outcomes
                                        if o["gate"]["gate_passed"] is not None)}


__all__ = ["ENTRY", "H1Windows", "OWN", "SCHEME", "SESSIONED", "config_of", "refusal", "units"]
