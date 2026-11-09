"""The served window's commitment data for its prover (`pouw_h1_served`'s, which re-exports these): its committed
trees (`trees`), each call's verifier ``--registered`` file (`reads_file`), and the entries a window commitment record
lists, held to the prover's earlier records (`register`, `WindowCommitmentEntries`), each file named `file_name`. It
imports nothing of `pouw_h1_served`: a run's ``served`` is its `Served`, read once the window is closed."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from verity.ml.protocols.pouw.circuit import h1window as HW
from verity.core.protocols.sampled_proofs.one_stage import partition as P, registration as R
from verity.core.protocols.sampled_proofs.one_stage.served import ServedTree
from verity.ml.circuits.codec import canonical_json

#: a statement's ``--registered`` file, one a call
FILE = "pouw-h1-registered-c{call}.json"


def file_name(call: int) -> str:
    return FILE.format(call=int(call))


def _served(run: Any) -> Any:
    served = getattr(run, "served", None)
    if getattr(served, "registered", None) is None:
        raise ValueError(f"{run.entry}: the window's rows aren't registered values (the run has no `Served`)")
    if not run.closed:
        raise ValueError(f"{run.entry}: the window isn't closed, so its tables aren't all committed")
    if served.refused or not served.registered:
        raise ValueError(f"{run.entry}: the window's trees were refused: {'; '.join(served.refused) or 'none committed'}")
    return served


def trees(run: Any) -> list[ServedTree]:
    """The window's registered trees, in registration order: refused unless `run` committed them as `Served` values."""
    return [ServedTree.of(r) for r in _served(run).registered.values()]


def reads_file(run: Any, call: int, ports: Mapping[str, str] | None = None, tp: Any = None) -> dict[str, Any]:
    """The verifier's ``--registered`` file of a statement over call ``call``'s units: `h1window.table_reads_file` over the
    window's Program (``tp`` its template partition, `partition.evaluate`'s, when the caller holds it) and its trees'
    roots, under the circuit ports ``ports`` (default `Served.ports`), each call table's domain bound to the window's open
    root and the weights' to none, as their trees' are. With ``receipts``, the file reads its units' receipts at the
    call's receipt table as its output row o0 (`h1window.RECEIPT_PORT`), unit u0 + t at row t. Its public row
    (`h1window.PUB_PORT`) is the call's public table at its known content, under the window's Program salt."""
    served = _served(run)
    prog = served.program
    tp = tp if tp is not None else P.evaluate(prog, served.query())
    return HW.table_reads_file(prog, tp, int(call), served.roots(), served.ports() if ports is None else dict(ports),
                               prog=served.program_digest, window=served.root, steps=served.steps,
                               salt=run._salt.hex())


@dataclass(frozen=True)
class WindowCommitmentEntries:
    """What `register` returns: the entries (every tree of the window, the record's ``registered`` field) and the open
    root that the window's commitment record lists, and each call's ``--registered`` file. It is the prover's input to
    that record and to its verifier calls, not the record: the record, its receipt and its digest are
    `service.WindowCommitmentRecord`'s."""

    registered: list[dict[str, Any]]
    window_root: str
    files: dict[int, dict[str, Any]]


def register(run: Any, calls: Sequence[int] | None = None, log: Sequence[Mapping[str, Any]] = (),
             directory: str | Path | None = None, ports: Mapping[str, str] | None = None) -> WindowCommitmentEntries:
    """The window's trees as one record's ``registered`` entries, held to this prover's earlier records ``log`` (R7-held:
    a value an earlier record registered under the same domain with another root is refused), and the ``--registered``
    file of each of ``calls`` (default every call; `reads_file`), written as `file_name` in ``directory`` (default the
    run's ``transcript_dir``)."""
    got, served = trees(run), run.served
    held = R.held_roots(log)
    for t in got:
        if held.get((t.value, t.domain), t.root) != t.root:
            raise ValueError(f"served tree {t.value!r}: an earlier record of the log registered it under this domain with "
                             f"root {held[(t.value, t.domain)][:16]}..., and a value is committed once a domain (R7-held)")
    tp = P.evaluate(served.program, served.query())
    files = {int(c): reads_file(run, c, ports, tp) for c in (range(len(served.calls)) if calls is None else calls)}
    d = directory if directory is not None else run.transcript_dir
    if d is not None:
        Path(d).mkdir(parents=True, exist_ok=True)
        for c, doc in files.items():
            (Path(d) / file_name(c)).write_bytes(canonical_json(doc))
    return WindowCommitmentEntries([t.entry() for t in got], served.root, files)


__all__ = ["FILE", "WindowCommitmentEntries", "file_name", "reads_file", "register", "trees"]
