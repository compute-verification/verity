"""`serve-openai`'s entry: the serve (`pipeline.serve_openai`) with each committed window's circuit planned at its close, and
its record made there given a sampling design (``--sampling-design``)."""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from verity_vllm.config import UsageError, group, option
from verity_vllm.pipeline import serve_openai as SO
from verity_vllm.pipeline.serve_openai import main as serve

__all__ = ["Options", "RecordOptions", "main", "plan_of", "preflight"]

#: the preflight's random rows, under a key drawn from the OS for the one check
PREFLIGHT_ROWS_DOMAIN = "verity/vllm/preflight-rows/v1"


@dataclass(frozen=True, kw_only=True)
class RecordOptions:
    """serve-openai's ``record`` group: each sealed window's record at its close (`close_record.Record`)."""
    sampling_design: str | None = option("--sampling-design", default=None, metavar="RATE",
                                         help="the rate each window's record registers its design under (window_record.window_design); "
                                              "none: windows are planned, and no record is made")
    checkpoint_dir: str | None = option("--checkpoint-dir", default=None,
                                        help="the checkpoint of record whose weights the records register (--sampling-design needs it)")
    checkpoints: str | None = option("--checkpoints", default=None, help="manifests/checkpoints.json (shard sha256 pins)")
    record_cap: int = option("--record-cap", type=int, default=64, help="units a member, the record's stated cap")
    record_workers: int = option("--record-workers", type=int, default=8, help="the record's close's workers")
    twin_ports: int = option("--twin-ports", type=int, default=8,
                             help="ports a window's strings are recomputed on the host and compared (0: none)")
    development_design: bool = option("--development-design", action="boolean_optional", default=False,
                                      help="--sampling-design is a development sampling design, which each record says")
    record_in_lease: bool = option("--record-in-lease", action="boolean_optional", default=False,
                                   help="make each window's record at its close, after the card's strings; default: the close "
                                        "leaves the card's strings (close_record.Record.handoff) and close_record's finish "
                                        "records the window once the card is released")


@dataclass(frozen=True, kw_only=True)
class Options(SO.Options):
    record: RecordOptions = group(RecordOptions)


def record_of(out: Path, opts: Any) -> Any:
    """The serve's `close_record.Record`, or None without ``--sampling-design``; refused (UsageError) with one and no
    ``--seal``, whose binding map the record reads, or no ``--checkpoint-dir``."""
    r = getattr(opts, "record", None)
    if r is None or r.sampling_design is None:
        return None
    if not opts.seal:
        raise UsageError("--sampling-design: a window's record reads its seal's binding map, so it needs --seal")
    if not r.checkpoint_dir:
        raise UsageError("--sampling-design: a window's record registers the weights of --checkpoint-dir, which is not given")
    from .close_record import Record

    return Record(r.sampling_design, r.checkpoint_dir, Path(out) / "private" / "salt_stream", checkpoints=r.checkpoints,
                  cap=r.record_cap, workers=r.record_workers, sample=r.twin_ports, development=r.development_design)


def plan_of(out: Path, opts: Any, flush: Callable[[], Any]) -> Any:
    """The serve's `close_plan.ClosePlan`, at the window class ``--window-class`` declares (none: each window says so)."""
    from .close_plan import ClosePlan
    from verity_vllm.inputs.serves import window_class_parts

    from .window_class import WindowClass

    cls = WindowClass(*window_class_parts(opts.window_class)) if opts.window_class else None
    r = getattr(opts, "record", None)
    return ClosePlan(out, cls, flush=flush, record=record_of(out, opts), in_lease=bool(r is not None and r.record_in_lease))


def preflight(opts: Any, *, card: bool) -> dict[str, Any]:
    """What the close builds that can be built before the serve: its plan at the declared class, refused without one; its
    record's design rule (`service.SamplingDesign`) and checkpoint; the host salt library, its twin's strings byte for byte
    the period's (`rows.row_strings`) on random rows; and on a ``card`` host the device salt library. The class circuit is
    lowered only at a window's close: φ reads the serving card's target and engine record from ``serve.json``."""
    import os
    import tempfile

    import numpy as np
    from verity.commitments import randomness as RND
    from verity.commitments.cuda.salt_stream.engine import host_lib
    from verity.commitments.periods import rows as RW
    from verity.commitments.rowleaf import SCHEMA_HM96_SHA512_ROW_V2

    from . import window_strings as WS
    from .window_record import SALTS_DOMAIN

    out = Path(tempfile.mkdtemp(prefix="plan-preflight-"))
    plan = plan_of(out, opts, lambda: None)
    if plan.cls is None:
        raise UsageError("the close plans no circuit without --window-class S,P,O")
    got: dict[str, Any] = {"class": [plan.cls.slots, plan.cls.prompt, plan.cls.outputs], "record": plan.record is not None}
    if (rec := plan.record) is not None:
        from verity.protocols.proof.spot.verifier import service as S

        S.SamplingDesign(rec.rule)
        ck = Path(rec.checkpoint_dir)
        if card and not ((ck / "config.json").is_file() and any(ck.glob("*.safetensors"))):
            raise UsageError(f"--checkpoint-dir {ck} holds no config.json and safetensors shards")
    twin = WS.HostTwin(host_lib(Path(rec.build if rec else out / "salt_stream") / "host"))
    rows_of = RND.derive(os.urandom(64), PREFLIGHT_ROWS_DOMAIN, {})
    key = RND.derive(os.urandom(64), SALTS_DOMAIN, {"window": "preflight"})
    for words in (1, 7, 64):
        rows = np.frombuffer(rows_of.stream(words, 10 * words), dtype="<u2").astype(np.uint16).reshape(5, words)
        lr = _Leaf(3, 2, rows, SCHEMA_HM96_SHA512_ROW_V2)
        strings, _ = WS.leaf_strings({("t", "p"): lr}, key, twin, {})
        ys = strings[("t", "p")][0]
        if strings[("t", "p")] != (ys, RW.row_strings(rows, ys, SCHEMA_HM96_SHA512_ROW_V2)):
            raise ValueError(f"the host twin's strings of {words}-word rows are not the period's")
    got["twin"] = "equal on 1-, 7- and 64-word rows"
    if card:
        from verity.commitments.cuda.salt_stream.engine import device_lib

        from .close_record import arch, nvcc

        device_lib(Path(rec.build if rec else out / "salt_stream") / "device", arch(), nvcc())
        got["device_lib"] = arch()
    return got


@dataclass
class _Leaf:
    template: int
    port: int
    rows: Any
    schema: str

    @property
    def units(self) -> list[int]:
        return list(range(len(self.rows)))


def main(opts: Options) -> int:
    return serve(opts, plan_of=plan_of)
