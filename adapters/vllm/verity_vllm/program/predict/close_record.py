"""A served window's record made at its close (`ClosePlan`): its circuit's units recorded from the window's committer, its
registered values, and its leaf strings from where the serve committed its rows (`window_strings`), written by
`window_record.commit` under ``windows/w<k>/record/``."""
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
    where the salt libraries build (``build``) and how many ports' strings the host twin recomputes (``sample``)."""

    rule: str
    checkpoint_dir: str
    build: Path
    checkpoints: str | None = None
    cap: int = 64
    workers: int = 8
    sample: int = 8
    _words: dict[str, Any] | None = field(default=None, init=False, repr=False)

    def roots(self, registered: Any, out: Path) -> dict[str, str]:
        """The serve's weights registration at ``out``'s `REGISTRATION`, made once from the checkpoint before the first
        window's advice is written (`window_weights.weights_registration`, its salts beside it, its deployment record at
        `DEPLOYMENT`), as each window's advice registration records it (`advice.registered_weights`)."""
        from verity.commitments import salt_file as SF

        from . import window_weights as WW

        path = Path(out) / REGISTRATION
        if not path.is_file():
            doc = WW.weights_registration(registered, SF.beside(path), self.checkpoint_dir, checkpoints_manifest=self.checkpoints)
            (Path(out) / DEPLOYMENT).write_text(json.dumps(WW.deployment_record(doc), indent=1) + "\n")
            path.write_text(json.dumps(doc, indent=1) + "\n")
        serve = json.loads((Path(out) / "serve.json").read_text())
        return ADV.registered_weights(json.loads(path.read_text()), serve)

    def words(self, registered: Any) -> dict[str, Any]:
        """The weights record's fields as the checkpoint of record composes them (`window_weights.record_words`), once a serve."""
        if self._words is None:
            from . import window_weights as WW

            self._words = WW.record_words(registered, self.checkpoint_dir, self.checkpoints)
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

    def make(self, plan: Any, window: int, compiled: Any, low: Any, units: Any, served: list[Any], binding: Any) -> dict[str, Any]:
        """Window ``window``'s record, ``plan``'s (`ClosePlan`) circuit ``low`` compiled as ``compiled``, its slots' served
        values ``served`` read from the close's committer (``binding.com`` under ``binding.rc``), its units ``units``
        (`window_plan.Plan`) staged where committed. Its run is the step rows' run root; its scan state the stand-in its
        slots' rows give (`scan_state_rows.standin_rows`: no attester scan is read at the close). -> what it wrote, its
        strings' cost (`window_strings.strings_of`) and the seconds each part took."""
        from verity.circuits.codec import program_digest
        from verity.protocols.resource_accounting.network.prover import (
            scan_state_rows as SR,
        )

        from verity_vllm.pipeline import advice_windows as AW
        from verity_vllm.pipeline import window_advice as WA

        from . import scan_state as SS
        from . import window_record as WREC
        from . import window_rows as WRS
        from . import window_strings as WS
        from . import window_weights as WW

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
        words = self.words(registered)
        lap("weights_read")

        def weights(label: str, lo: int, hi: int) -> Any:
            if label not in words:
                raise ValueError(f"weight {label}: the checkpoint of record composes no such field")
            return words[label][lo:hi]

        slot_of = {(cls.anchor(k), cls.lane(k)): k for k in range(cls.slots)}
        slots = [WRS.Slot(P.fn, cls, placed[slot_of[(P.anchor, P.lane)]][1], weights, s) for P, s in zip(low.programs, served)]
        values = {**WW.weight_values(registered, words, out / REGISTRATION, keep_rows=True), SS.SCAN_STATE: state}
        lap("values")
        hasher, twin, block = self.hashers(binding.com)
        cost: dict[str, Any] = {}

        def strings(win: Any, key: Any) -> Any:
            return WS.strings_of(win, key, hasher, units, block, twin=twin, sample=self.sample, cost=cost)

        to = out / "windows" / f"w{window}" / "record"
        to.mkdir(parents=True, exist_ok=True)
        filled, win = WREC.commit(low, cls, slots, values, run, WREC.window_design(self.rule, low.circuit), self.cap, bound, to,
                                  window, strings=strings, workers=self.workers, lap=lap)
        return {"run": run, "out": str(to), "members": len(win.window.members), "population": sum(filled.n.values()),
                "design": {"rule": self.rule, "from": "the serve's --sampling-design"}, "cap": self.cap,
                "scan_state": "stand-in", "weights": "checkpoint of record", "strings": cost, "seconds": seconds}


def arch() -> str:
    """This host's card's arch as the salt stream's device library is built for it (``90a``)."""
    from verity_vllm import settings
    from verity_vllm.config import card_arch

    return card_arch(settings.card_sku()).removeprefix("sm_")


def nvcc() -> str | None:
    """``$CUDA_HOME/bin/nvcc`` where it is, else None (the one on the path)."""
    import os

    tool = Path(os.environ.get("CUDA_HOME") or "/nonexistent") / "bin" / "nvcc"
    return str(tool) if tool.is_file() else None
