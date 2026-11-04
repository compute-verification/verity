"""serve-openai's Pearl+ arm (``--protocols pouw:pearl-c-<device>-v1-h3``): Pearl-C's PoUW installed on the served engine
itself, so each Commit window's linears are the Pearl+ matmuls whose passes the option commits, a pass an engine step
(`pouw.pearl_c.passes`).

* **One engine.** The option is installed as the engine is built (`Serving.engine`), under the pouw settings
  ``--protocols-config`` gives (the kernel and its cubin, `pouw.Settings`) and the server's own (`OWN`): ``transcript_dir``
  ``OUT/private/pearlplus-serve/passes`` (each pass retained in ``pass-NNNNN``, complete for Pearl+'s verdict,
  owner-only), and the epoch's coins. The committer is built after it, so its hooks see each linear's x and the y Pearl+
  computed.
* **Coins.** With ``--challenger-gate`` and ``--challenger-peer``, a salt-only session of the auditor host's challenger,
  reserved under ``--pearl-lane``'s gate key before the engine is built (`pouw.h1.window.reserve_salt`, which refuses
  while the lane's open period is H1's), which the install opens on its salt epoch's root; else an in-process challenger
  of epoch 0 (`pouw.challenged`), which credits nothing. Either way ``passes/coins.json`` is the coins' commitment.
* **A window's passes.** `PearlWindows` records which passes each Commit window's steps ended (its outcome, the window
  line's `KEY`: ``passes`` [first, end), those with calls and the window's wall) and, at the stop, once the option is closed
  and every retained pass written, ``private/pearlplus-serve/served.json`` in ``served_passes.py serve``'s form, each pass
  with the window that ended it (none for a step outside every window, such as the learn warm-up's). ``served_passes.py
  window`` cuts it to one window's passes, for ``retain --window-record W``, ``verify --tie W`` and the window commit tied to W
  (``prove_path.py window-commit --window-registration W --pearl-passes DIR``), whose circuit is cut at the linears
  (`window_class.PearlTables`): round s, the window's s-th pass with calls, is the window's s-th engine step.
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, option

PREFIX = "pouw:pearl-c-"
#: the window line's (and the summary's) key for this arm's outcome
KEY = "pouw_pearl"
#: where the passes and their document go under the serve dir: they hold the run's activations and the prover's salt key
PRIVATE = Path("private") / "pearlplus-serve"
#: the pouw settings the server sets itself: the retained passes' directory and the epoch's coins
OWN = ("transcript_dir", "coins", "challenger", "session", "gate", "period")
#: what ``served.json`` keeps of each pass, `served_passes.KEEP`'s fields
KEEP = ("units", "base", "tiles", "screened_rows", "failing_rows", "unsure_rows", "excluded_rows", "transcript_root",
        "activation_roots", "schedule", "manifest")
SCHEMA = "verity-pouw/pearl-c-served-passes/v1"
#: the epoch the salt-only session is reserved at, and the in-process challenger's
EPOCH = 0


@dataclass(frozen=True, kw_only=True)
class PearlOptions:
    """serve-openai's ``--pearl-lane`` (its ``pearl`` group)."""
    pearl_lane: str | None = option("--pearl-lane", default=None, metavar="LANE",
                                    help=f"with --protocols {PREFIX}* and --challenger-gate: the lane whose gate key reserves "
                                         "the epoch's salt-only session (pouw.h1.window.reserve_salt), the lane a period of "
                                         "record names")


def is_pearl(opts: Any) -> bool:
    return str(getattr(opts, "protocols", None) or "").startswith(PREFIX)


def config_of(value: str | None) -> dict[str, Any]:
    """``--protocols-config`` under this arm: a JSON object, or a JSON file, ``{"pouw": {...}}``; ``{"pouw": {}}`` for None.
    `UsageError` on another protocol's settings, a pouw setting the server sets itself (`OWN`), or one `pouw.Settings`
    refuses."""
    from verity_vllm.protocols import load_settings
    from verity_vllm.protocols import pouw as PW

    doc: dict[str, Any] = {}
    if value:
        try:
            doc = load_settings(json.loads(value) if value.lstrip().startswith("{") else value)
        except (OSError, ValueError) as e:
            raise UsageError(f"--protocols-config {value[:80]}: {e}") from None
    if stray := sorted(set(doc) - {"pouw"}):
        raise UsageError(f"--protocols-config: {stray} are not the enabled protocols' (pouw)")
    pouw = dict(doc.get("pouw", {}))
    if own := sorted(set(pouw) & set(OWN)):
        raise UsageError(f"--protocols-config: the pouw settings {own} are the server's own (the retained passes' directory "
                         "and the epoch's coins: give --challenger-gate, --challenger-peer and --pearl-lane for a live "
                         "session's)")
    try:
        PW.Settings.of(pouw)
    except ValueError as e:
        raise UsageError(f"--protocols-config: {e}") from None
    return {"pouw": pouw}


def refusal(opts: Any) -> str | None:
    """Why ``opts`` can't serve Pearl+ (None when they can); `config_of`'s `UsageError` when it refuses
    ``--protocols-config``."""
    p, g = str(opts.protocols), opts.gate
    if not is_pearl(opts):
        return f"--pearl-lane is Pearl+'s: it needs --protocols {PREFIX}*" if opts.pearl.pearl_lane else None
    if "," in p:
        return f"--protocols {p}: serve-openai serves one {PREFIX}* entry alone"
    if g.h1_prover or g.h1_receipts or g.h1_window_commit or g.h1_model or g.h1_weights_root:
        return f"--h1-prover, --h1-receipts, --h1-window-commit, --h1-model and --h1-weights-root are PoUW H1's, not {p}'s"
    from verity_vllm.pipeline.serve_h1 import DRAWER

    after = getattr(getattr(opts, "retention", None), "release_after", "") or ""
    if DRAWER in (x.strip() for x in after.split(",")):
        return f"--release-after {DRAWER} waits for PoUW H1's draw of each window, which {p} doesn't put"
    if bool(g.challenger_gate) != bool(g.challenger_peer):
        return ("--challenger-gate and --challenger-peer go together: the gate's command and this machine's name in the "
                "auditor host's peers, where its challenger reaches the server (ssh -R)")
    if bool(g.challenger_gate) != bool(opts.pearl.pearl_lane):
        return ("--pearl-lane names the lane whose gate key reserves the epoch's salt session at --challenger-gate: give "
                "both or neither")
    if not opts.commit or opts.scheme != "rows-sha512" or not opts.required_manifest:
        return (f"--protocols {p} ties Pearl+'s passes to the window the server's rows-sha512 Commit commits: it needs "
                "--commit, --scheme rows-sha512 and the bench row's --required-manifest")
    config_of(opts.protocols_config)
    return None


def coins_of(option: Any, reserved: Mapping[str, Any] | None) -> dict[str, Any]:
    """Where the option took its salt coin (`served_step.coins_of`'s form): its ``coin_source``, the gate's answer it
    carries (no secret), its salt epoch's number and the root its salt coin follows, its salt, how its session closed and
    the session head a retained pass's manifest names; ``reserved`` the gate's reserve answer."""
    from verity.core.protocols.pouw import salt as PS

    from verity_vllm.protocols.pouw.pearl_c.manifest import session_head

    c = option.commitment()
    k, w = c.get("pass_index"), c.get("weights_root")
    after = w if k is None or w is None else PS.epoch_open_root(bytes.fromhex(w), k).hex()
    return {"source": c.get("coin_source", "in-process"), "reserved": None if reserved is None else dict(reserved),
            "session": c.get("session"), "pass_index": k, "weights_root": w, "salt_after": after, "salt": c.get("salt"),
            "closed": option.info().get("challenger"), "manifest_head": session_head(c)}


def macs(manifest: str | None) -> int | None:
    """A retained pass's linear MACs, its calls' real rows × n × k from its manifest; None without one."""
    if not manifest:
        return None
    return sum(u["real_rows"] * u["n"] * u["k"] for u in json.loads(Path(manifest).read_text())["matmuls"])


def _private(d: Path) -> Path:
    d.mkdir(mode=0o700, parents=True, exist_ok=True)
    d.chmod(0o700)
    return d


class Serving:
    """The arm before its engine: the pouw settings the engine is built under (`engine`), the passes under ``out``'s
    `PRIVATE`, and the epoch's coins (a live session reserved now, through ``gate``, a `commit_coins.ChallengerGate` of
    ``--challenger-gate`` by default, or an in-process challenger whose commitment is written now)."""

    def __init__(self, opts: Any, out: Path, gate: Any = None) -> None:
        from verity_vllm.protocols import pouw as PW

        self.entry, self.out, self.run_id = str(opts.protocols), Path(out), getattr(opts, "run_id", None)
        self.dir = _private(self.out / PRIVATE)
        self.passes = _private(self.dir / "passes")
        pouw = {**config_of(opts.protocols_config)["pouw"], "transcript_dir": str(self.passes)}
        self.reserved = self.period = None
        self.peer, self.lane = opts.gate.challenger_peer, opts.pearl.pearl_lane
        if opts.gate.challenger_gate:
            from verity_vllm.pipeline.commit_coins import ChallengerGate
            from verity_vllm.protocols.pouw.h1 import window as PHW

            gate = gate or ChallengerGate(opts.gate.challenger_gate)
            self.reserved, self.period = PHW.reserve_salt(gate.call, gate.ledger, self.lane, self.peer, EPOCH)
            pouw.update(challenger="127.0.0.1:0", session=self.reserved, gate=PHW.gate_of(gate.call), period=self.period)
        else:
            pouw = PW.challenged({"pouw": pouw}, [self.entry], EPOCH)["pouw"]
            self._coins(pouw["coins"].commitment.to_json())
        self.config = {"pouw": pouw}

    def _coins(self, doc: Mapping[str, Any]) -> None:
        (self.passes / "coins.json").write_text(json.dumps(doc) + "\n")

    def engine(self) -> dict[str, Any]:
        """`verity_vllm.LLM`'s ``protocols`` and ``protocols_config``."""
        return {"protocols": self.entry, "protocols_config": self.config}

    def arm(self, llm: Any) -> PearlWindows:
        """The arm over the built ``llm``, its live session's commitment written once the install has opened it."""
        arm = PearlWindows(llm, self)
        if self.reserved is not None:
            self._coins(arm.run.commits["coins"])
        return arm


class PearlWindows:
    """The arm over ``llm`` (the served `verity_vllm.LLM`, its pouw option Pearl-C's) for `serve_commit.CommitWindows`:
    each window's passes, from the first its steps ended to the last (`close`), and at the stop the passes' document
    (`drain`). ``clock`` times each window."""

    KEY = KEY

    def __init__(self, llm: Any, serving: Serving, clock: Callable[[], float] = time.perf_counter) -> None:
        self.llm, self.serving, self.clock = llm, serving, clock
        self.option = llm.protocols.get("pouw") if llm.protocols is not None else None
        if self.option is None or not callable(getattr(self.option, "end_pass", None)):
            raise ValueError(f"{serving.entry}: the engine's pouw option ends no pass, so it is no Pearl-C install")
        self.run = self.option.end_pass.__self__
        self.windows: dict[int, dict[str, Any]] = {}
        self.opened: tuple[int, int, float] | None = None
        self.written: dict[str, Any] | None = None

    def _passes(self) -> list[dict[str, Any]]:
        return self.run.details["passes"]

    def open(self, window: int) -> None:
        """Window ``window`` opens before its first step: the passes so far are none of its."""
        if self.opened is not None:
            raise RuntimeError(f"{self.serving.entry}: window {window} opens while window {self.opened[0]}'s is open")
        self.opened = (window, len(self._passes()), self.clock())

    def close(self, window: int) -> dict[str, Any] | None:
        """Window ``window``'s outcome once its last step returned: the passes its steps ended, [first, end), and those with
        calls, its wall from its open; None when it never opened."""
        if self.opened is None or self.opened[0] != window:
            return None
        _, first, t0 = self.opened
        self.opened = None
        passes = self._passes()
        end = len(passes)
        out = {"entry": self.serving.entry, "window": window, "passes": [first, end],
               "with_calls": sum(1 for p in passes[first:end] if p.get("units")), "wall_s": round(self.clock() - t0, 6)}
        self.windows[window] = out
        return out

    def drawing(self, draws: Any) -> None:
        """A Pearl+ window puts no draw into serve-openai's draw-gated release."""

    def window_of(self, i: int) -> int | None:
        return next((k for k, w in self.windows.items() if w["passes"][0] <= i < w["passes"][1]), None)

    def describe(self) -> dict[str, Any]:
        """serve.json's ``protocols``."""
        s = self.serving
        return {"entry": s.entry, "key": KEY, "passes": str(PRIVATE / "passes"), "served": str(PRIVATE / "served.json"),
                "coins": "challenger" if s.reserved is not None else "in-process", "peer": s.peer, "lane": s.lane,
                "session": None if s.reserved is None else s.reserved.get("session"),
                "max_step_tokens": self.run.details.get("max_step_tokens"), "pass_ends": self.run.details.get("pass_ends")}

    def summary(self) -> dict[str, Any]:
        """summary.json's `KEY`: windows closed, passes ended and those with calls, and where their document went."""
        passes = self._passes()
        return {"windows": len(self.windows), "passes": len(passes), "with_calls": sum(1 for p in passes if p.get("units")),
                "served": self.written}

    def document(self) -> dict[str, Any]:
        """``served.json``: every pass the option ended, with its window (``timed``: one of a window's), `served_passes.py
        serve`'s caps and coins."""
        from verity_vllm.protocols.pouw import pearl_c as PPD

        run = self.run
        caps = {"tiles": run.capacity, "units": PPD.CALL_CAPACITY, "screened_rows": len(run.flag_buf)}
        got = []
        for i, p in enumerate(self._passes()):
            k = self.window_of(i)
            got.append({"pass": i, "timed": k is not None, "window": k, **{f: p.get(f) for f in KEEP if f in p},
                        "macs": macs(p.get("manifest"))})
        over = [p["pass"] for p in got if any((p.get(f) or 0) > cap for f, cap in caps.items())]
        return {"schema": SCHEMA, "served_by": "serve-openai", "arm": self.serving.entry, "run": self.serving.run_id,
                "retained": True, "passes_dir": str(self.serving.passes), "windows": {str(k): w for k, w in self.windows.items()},
                "pass_ends": run.details.get("pass_ends"), "max_step_tokens": run.details.get("max_step_tokens"),
                "caps": caps, "passes": got, "passes_with_calls": sum(1 for p in got if p["units"]), "over_caps": over,
                "coins": coins_of(self.option, self.serving.reserved)}

    def drain(self) -> None:
        """At the stop: the option closed (every retained pass written, its session closed), then ``served.json``."""
        self.llm.close()
        doc = self.document()
        path = self.serving.dir / "served.json"
        path.write_text(json.dumps(doc, indent=1) + "\n")
        self.written = {"path": str(PRIVATE / "served.json"), "passes": len(doc["passes"]),
                        "with_calls": doc["passes_with_calls"], "over_caps": len(doc["over_caps"])}


def serving_of(opts: Any, out: Path) -> Serving | None:
    """The Pearl+ arm's `Serving` under ``--protocols pouw:pearl-c-*``, None otherwise."""
    return Serving(opts, out) if is_pearl(opts) else None


__all__ = ["EPOCH", "KEEP", "KEY", "OWN", "PREFIX", "PRIVATE", "PearlOptions", "PearlWindows", "SCHEMA", "Serving",
           "coins_of", "config_of", "is_pearl", "macs", "refusal", "serving_of"]
