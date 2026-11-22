"""A served window's record: at its close (`ClosePlan`) the card's part, the strings of the units the serve acquired made
where it committed them under the window's key (`Record.handoff`, left in ``windows/w<k>/handoff/``); then, at the close
or after the card is released (`close_plan.finish`), its circuit's units recorded from its served values, its registered values,
closed with those strings and the rest from the host (`window_strings`), and written by `window_record.commit` under
``windows/w<k>/record/``."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from verity_vllm.pipeline import advice as ADV

__all__ = ["DEPLOYMENT", "REGISTRATION", "Record"]

REGISTRATION = "weights_registration.json"
DEPLOYMENT = "deployment-registration.json"


@dataclass
class Record:
    """What a window's record reads besides its close: the rate ``rule`` its design is (`window_record.window_design`; the
    serve states it, where the prover is to propose it in the committed advice), the checkpoint of record whose weights it
    registers (``checkpoint_dir``, pinned by ``checkpoints``), the stated ``cap`` (units a member), the close's ``workers``,
    where the salt libraries build (``build``) and how many ports' strings the host twin recomputes (``sample``). A
    ``development`` rule is a development sampling design, which the record says (`development_of`)."""

    rule: str
    checkpoint_dir: str
    build: Path
    checkpoints: str | None = None
    cap: int = 64
    workers: int = 8
    sample: int = 8
    development: bool = False
    _words: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def roots(self, registered: Any, out: Path) -> dict[str, str]:
        """The serve's weights registration at ``out``'s `REGISTRATION`, made once from the checkpoint before the first
        window's advice is written (`window_weights.weights_registration`, its salts beside it, its deployment record at
        `DEPLOYMENT`), as each window's advice registration records it (`advice.registered_weights`)."""
        from verity.commitments import salt_file as SF

        from . import window_weights as WW

        path = Path(out) / REGISTRATION
        if not path.is_file():
            doc = WW.weights_registration(registered, SF.beside(path), self.checkpoint_dir, checkpoints_manifest=self.checkpoints,
                                          tables=served_tables(out))
            (Path(out) / DEPLOYMENT).write_text(json.dumps(WW.deployment_record(doc), indent=1) + "\n")
            path.write_text(json.dumps(doc, indent=1) + "\n")
        serve = json.loads((Path(out) / "serve.json").read_text())
        return ADV.registered_weights(json.loads(path.read_text()), serve)

    def words(self, registered: Any, out: Path) -> dict[str, Any]:
        """The weights record's fields as the checkpoint of record composes them (`window_weights.record_words`), and its
        analytic tables as the serve under ``out`` served them (`served_tables`), once a serve."""
        if self._words is None:
            from . import window_weights as WW

            self._words = WW.record_words(registered, self.checkpoint_dir, self.checkpoints, tables=served_tables(out))
        return self._words

    def hashers(self, com: Any) -> tuple[Any, Any, Any]:
        """The strings' hasher where ``com`` holds its rows (a card: `window_strings.Device`; else the host twin), the host twin,
        and the reader of ``com``'s committed blocks (`window_strings.block_of`; None where it holds none)."""
        from verity.commitments.cuda.salt_stream.engine import device_lib, host_lib

        from . import window_strings as WS

        twin = WS.HostTwin(host_lib(self.build / "host"))
        blocks = getattr(com, "_gpu_blocks", None) or {}
        first = next(iter(blocks.values()), None)
        if first is None:
            return twin, twin, None
        blk = first[0]
        if not getattr(blk, "is_cuda", False):
            return twin, twin, WS.block_of(com)
        return WS.Device(device_lib(self.build / "device", arch(), nvcc()), blk.device), twin, WS.block_of(com)

    def handoff(self, plan: Any, window: int, low: Any, units: Any, binding: Any) -> dict[str, Any]:
        """The card's part of window ``window``'s close: a key drawn fresh for the window (`window_record.key_of`) and the
        strings of every unit ``units`` (`window_plan.Plan`) feeds, made where ``binding.com`` committed it
        (`window_strings.card_strings`), written as `window_strings.HANDOFF` under ``windows/w<k>/handoff/`` (private: it
        holds the key) for `make` to close with. -> where, and the card's cost."""
        from . import window_record as WREC
        from . import window_strings as WS

        run = bytes(binding.rc.run_root).hex()
        hasher, _twin, block = self.hashers(binding.com)
        if block is None:
            raise ValueError("handoff: the committer retains no blocks, so no unit's row is where the serve committed it")
        key = WREC.key_of(run)
        card, cost = WS.card_strings(units, block, low.circuit, key, hasher)
        to = plan.out / "windows" / f"w{window}" / "handoff"
        WS.save_card(to, card, key, {"window": window, "run": run, "card": cost})
        return {"out": str(to), "files": list(WS.HANDOFF), **cost}

    def make(self, plan: Any, window: int, compiled: Any, low: Any, units: Any, served: list[Any], binding: Any,
             handoff: Path | None = None) -> dict[str, Any]:
        """Window ``window``'s record, ``plan``'s (`ClosePlan`) circuit ``low`` compiled as ``compiled``, its slots' served
        values ``served`` read from the close's committer (``binding.com`` under ``binding.rc``), its units ``units``
        (`window_plan.Plan`) staged where committed. Its run is the step rows' run root; its scan state the stand-in its
        slots' rows give (`scan_state_rows.standin_rows`: no attester scan is read at the close). Given ``handoff`` (`handoff`'s
        directory), closed under its key with its strings (`window_strings.handed_strings`), else with strings made here
        (`window_strings.strings_of`). -> what it wrote, its strings' cost and the seconds each part took."""
        from verity.circuits.codec import program_digest
        from verity.protocols.run import schemes

        from verity_vllm.pipeline import advice_windows as AW
        from verity_vllm.pipeline import window_advice as WA

        from . import scan_state as SS
        from . import window_record as WREC
        from . import window_rows as WRS
        from . import window_strings as WS
        from . import window_weights as WW

        SR = schemes.catalog("network-attester", "scan_state_rows")

        seconds: dict[str, float] = {}
        t = [time.perf_counter()]

        def lap(what: str) -> None:
            now = time.perf_counter()
            seconds[what], t[0] = round(now - t[0], 3), now

        cls, out, registered = plan.cls, plan.out, plan._registered()
        run = bytes(binding.rc.run_root).hex()
        opening = json.loads((out / "advice" / f"w{window}" / "private" / "opening.json").read_text())
        bound = AW.opened(plan.compiler, opening, WA)[1]
        reqs = [u for u in compiled.requests if u is not None]
        placed = WREC.place_slots(cls, [u.id for u in reqs], ADV.bodies(out, window))
        state = WRS.state_value(program_digest(low.circuit.program()[0]), cls,
                                SR.standin_rows(cls, [(row, m) for _u, row, m in placed]), window=run)
        words = self.words(registered, out)
        lap("weights_read")
        weights = weight_elements(words)
        slot_of = {(cls.anchor(k), cls.lane(k)): k for k in range(cls.slots)}
        slots = [WRS.Slot(P.fn, cls, placed[slot_of[(P.anchor, P.lane)]][1], weights, s) for P, s in zip(low.programs, served)]
        values = {**WW.weight_values(registered, words, out / REGISTRATION, keep_rows=True), SS.SCAN_STATE: state}
        lap("values")
        cost: dict[str, Any] = {}
        key = None
        if handoff is not None:
            from verity.commitments.cuda.salt_stream.engine import host_lib

            key, card, _meta = WS.load_card(handoff)
            twin = WS.HostTwin(host_lib(self.build / "host"))

            def strings(win: Any, key: Any) -> Any:
                return WS.handed_strings(win, key, twin, card, sample=self.sample, cost=cost)
        else:
            hasher, twin, block = self.hashers(binding.com)

            def strings(win: Any, key: Any) -> Any:
                return WS.strings_of(win, key, hasher, units, block, twin=twin, sample=self.sample, cost=cost)

        to = out / "windows" / f"w{window}" / "record"
        to.mkdir(parents=True, exist_ok=True)
        design = WREC.window_design(self.rule, low.circuit)
        filled, win = WREC.commit(low, cls, slots, values, run, design, self.cap, bound, to,
                                  window, strings=strings, workers=self.workers, lap=lap, key=key)
        population = sum(filled.n.values())
        said = {"development": development_of(design, population)} if self.development else {}
        return {"run": run, "out": str(to), "members": len(win.window.members), "population": population,
                "design": {"rule": self.rule, "from": "the serve's --sampling-design", **said}, "cap": self.cap,
                "scan_state": "stand-in", "weights": "checkpoint of record", "strings": cost, "seconds": seconds}


def development_of(design: dict[str, Any], n: int) -> dict[str, Any]:
    """A development sampling design's statement: its draw of the window's ``n`` proof blocks, so no accept under it reads
    as one at the rate of record (what an accept establishes, ε at δ, the verifier reports). Refused for a design that is
    not a uniform draw."""
    if design.get("law") != "subset":
        raise ValueError(f"a development sampling design is a uniform draw (rate:NUM/DEN), not {design.get('law')!r}")
    return {"says": "DEVELOPMENT: a development sampling design, not the rate of record", "draws": int(design["k"]), "of": n}


def served_tables(out: Path) -> dict[str, Any] | None:
    """The analytic tables the serve under ``out`` served, by the weights record's table name: its rotary table, which
    serve-openai writes to ``private/cos_sin_cache.npy``; None where it wrote none."""
    import numpy as np

    f = Path(out) / "private" / "cos_sin_cache.npy"
    return {"cos_sin": np.load(f)} if f.is_file() else None


def weight_elements(words: dict[str, Any]) -> Any:
    """`window_rows.Weights` over ``words`` (`window_weights.record_words`, each field in its checkpoint shape): a field's
    elements [lo, hi) of its row-major flattening."""

    def weights(label: str, lo: int, hi: int) -> Any:
        if label not in words:
            raise ValueError(f"weight {label}: the checkpoint of record composes no such field")
        return words[label].reshape(-1)[lo:hi]

    return weights


def arch() -> str:
    """This host's card's arch as the salt stream's device library is built for it (``90a``)."""
    from verity_vllm import settings
    from verity_vllm.config import card_arch

    return card_arch(settings.card_sku()).removeprefix("sm_")


def nvcc() -> str | None:
    """``$CUDA_HOME/bin/nvcc`` where it is, else None (the one on the path)."""
    from verity_vllm import settings

    tool = Path(settings.get("CUDA_HOME") or "/nonexistent") / "bin" / "nvcc"
    return str(tool) if tool.is_file() else None
