"""A two-stage window's select inputs (`boundary_rows.Layout.logits`): the logits each served id was selected from, as
the sampler took them, committed at serve beside the class's ids so that round 2 reads its select's inputs at rows fixed
before any coin.

**Rows.** On rank 0, `SamplerHook` wraps the model runner's sampler (vLLM's Model Runner V2, as `engine.sampler_capture`
wraps it in a replay): at each call it keeps, for every request that emits a token, its row of logits as bf16 bits
(`sampler_capture.emitting`, `sampler_capture.kept_rows`), in `Logits`, in the order the request's ids were emitted.

**The value.** A class's span r holds O·K rows of N words (`Layout.logit_rows`): token j's row of V logits as its K
units read it, unit k's lanes [k·N, (k + 1)·N) at row j·K + k, so a span's rows are its (O, V) logits reshaped to
(O·K, N); past its n ids served, token n−1's rows again (`span_rows`), as `boundary_rows.id_rows` repeats its id. The
rows are `hm96-sha512/row/v2` rows under the class's window domain (`boundary_rows.window_domain`), each salt fresh from
the OS.

**Commitment.** The class's slot of each span is fixed only when its window closes, so the rows are committed then
(`Builder.build`): the window's boundary commitment sends rank 0 the class's `assignment` (the domain's items, and each
span's engine id, ids served and finish, in slot order) through `Channel` (files in a private directory on the node) or,
in one process, `LocalChannel`; rank 0 commits the class's rows on its host, keeps them (`Kept`) and answers with the
value's entry, whose root the class lists (`root`) before any coin.

**What round 2 opens** (`load`): the class's rows and salts as rank 0 kept them, in files on the node readable by their
owner only, never under a run's directory or the serve's records, recommitted and held to the class's root, as
`boundary_rows.Committed.port` opens the return. A class's files are under ``logits/w<window>-<run>-p<P>-o<O>[-nostop]``:
under a stop rule a stop class and its sibling without a stop block share (P, O), and the suffix tells them apart.
"""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from verity.commitments.windows import links as LK, rows as CS
from verity_vllm.engine import sampler_capture as SC
from verity_vllm.pipeline import boundary_rows as BR, boundary_tables as BT

__all__ = ["Builder", "Channel", "Kept", "LocalChannel", "Logits", "SamplerHook", "assignment", "load", "root",
           "span_rows", "stem"]

SCHEMA = BR.SCHEMA
#: the files a class's kept value is in
ROWS, SALTS, HEAD = "logits.rows", "logits.salts", "logits.json"


def stem(window: int, run: str, prompt: int, outputs: int, stopped: bool | None) -> str:
    """A class's name in rank 0's files: its window, run, (P, O), and ``-nostop`` for a class under a stop rule that is
    not its stop class (``stopped`` False)."""
    return f"w{int(window)}-{run[:16]}-p{int(prompt)}-o{int(outputs)}{'-nostop' if stopped is False else ''}"


def span_rows(rows: np.ndarray, outputs: int, words: int) -> np.ndarray:
    """A span's rows of the value in a class of ``outputs`` ids, rows of ``words`` words, from its ``rows`` (``(n, V)``
    u16, one an id served, in order): each token's row in V / ``words`` rows, then past its n ids token n−1's again, as
    the replay block's instances past them are copies of its last; zero for a span served no id."""
    r = np.asarray(rows, dtype=np.uint16)
    n, V = r.shape
    if V % words or n > outputs:
        raise ValueError(f"a span's {n} rows of {V} logits are not whole rows of {words} words for at most {outputs} ids")
    pad = r[-1:] if n else np.zeros((1, V), dtype=np.uint16)
    full = np.concatenate([r, np.repeat(pad, outputs - n, axis=0)])
    return full.reshape(outputs * (V // words), words)


def assignment(window: int, layout: Any, spans: Sequence[Any]) -> dict[str, Any]:
    """A class's request for its value (`boundary_rows.Layout` ``layout``, its `Span`\\ s ``spans`` in slot order): what
    the value's domain binds, its rows a span and words a row, and each span's engine id, ids served and finish.
    Private: it names the engine's ids."""
    rows, words = layout.logit_rows
    return {"window": int(window), "run": layout.run, "prompt": layout.prompt, "outputs": layout.outputs,
            "stopped": layout.held if layout.stop is not None else None, "program": layout.partition["program"],
            "partition": BR.P.digest(layout.partition), "value": layout.logits_port, "rows": rows, "words": words,
            "spans": [[s.engine_id, len(s.served), s.finish] for s in spans]}


def root(answer: Mapping[str, Any], layout: Any) -> bytes:
    """Rank 0's answer to the class's `assignment`: the value's root, refused unless it is the layout's value at its own
    leaves, words and domain, or saying why rank 0 committed none."""
    if answer.get("error") is not None:
        raise ValueError(f"rank 0 committed no logits: {answer['error']}")
    e, port = answer.get("entry") or {}, layout.logits_port
    want = {"value": port, "leaves": layout.leaves(port), "words": layout.words(port),
            "domain": layout.domain(port).domain_id.hex()}
    if {k: e.get(k) for k in want} != want or len(bytes.fromhex(str(e.get("root", "")))) != 64:
        raise ValueError(f"rank 0's logits {e.get('value')!r} aren't the class's {port} at its leaves, words and domain")
    return bytes.fromhex(e["root"])


class Logits:
    """Rank 0's logits of each request, one row an id emitted, in order: `add` appends a call's, `take` hands a finished
    request's over. Thread-safe: the sampler adds, the builder takes. A hook failure (`fail`) is kept and refuses every
    later build."""

    def __init__(self) -> None:
        self._rows: dict[str, list[np.ndarray]] = {}
        self._lock = threading.Lock()
        self.failed: str | None = None

    def add(self, request: str, row: np.ndarray) -> None:
        with self._lock:
            self._rows.setdefault(str(request), []).append(np.array(row, dtype=np.uint16, copy=True))

    def take(self, request: str) -> np.ndarray | None:
        """Request ``request``'s rows ``(n, V)``, no longer held here; None where it emitted none."""
        with self._lock:
            mine = self._rows.pop(str(request), None)
        return np.stack(mine) if mine else None

    def requests(self) -> list[str]:
        with self._lock:
            return sorted(self._rows)

    def fail(self, why: str) -> None:
        self.failed = self.failed or why


class SamplerHook:
    """The model runner's sampler (``inner``), keeping in ``logits`` each emitting request's row before each call: rows
    taken while a CUDA graph is captured, or from a batch the sampler's rows can't be read from
    (`sampler_capture.kept_rows`), fail ``logits`` (`Logits.fail`), and sampling goes on. Every other attribute is the
    sampler's."""

    def __init__(self, inner: Any, logits: Logits | None = None,
                 capturing: Callable[[], bool] | None = None) -> None:
        self.inner, self.logits, self.capturing = inner, logits if logits is not None else Logits(), capturing

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def __call__(self, logits: Any, input_batch: Any, *a: Any, **kw: Any) -> Any:
        rows, keep = None, []
        if not self.logits.failed and not (self.capturing is not None and self.capturing()):
            try:
                keep = SC.emitting(input_batch, lambda _r: True)
                rows = SC.kept_rows(logits, input_batch, keep) if keep else None
            except Exception as e:  # noqa: BLE001 - the builder refuses every class after it, and says why
                self.logits.fail(f"{type(e).__name__}: {e}")
                keep = []
        out = self.inner(logits, input_batch, *a, **kw)
        for k, i in enumerate(keep):
            self.logits.add(input_batch.req_ids[i], rows[k])
        return out

    @classmethod
    def install(cls, runner: Any, capturing: Callable[[], bool] | None = None) -> SamplerHook:
        """Wrap ``runner``'s sampler. Refused for a runner that shards sampling across ranks: each rank's logits are
        then an all-to-all's, not the gathered row."""
        if getattr(runner, "batch_sharder", None) is not None:
            raise TypeError("the runner shards sampling across ranks: each rank's logits are an all-to-all's, not the "
                            "gathered row")
        if capturing is None:
            import torch

            capturing = torch.cuda.is_current_stream_capturing
        runner.sampler = cls(runner.sampler, capturing=capturing)
        return runner.sampler


class Kept:
    """Each class's value as rank 0 committed it, under ``directory`` on the node: in `stem`/, its rows (`ROWS`, u16
    little-endian), their salts (`SALTS`) and its entry with what its domain binds (`HEAD`), written last. Every directory
    and file is readable by its owner only."""

    def __init__(self, directory: Path) -> None:
        self.dir = Path(directory)

    @classmethod
    def of(cls, cfg: Mapping[str, Any], *public: Path | None) -> Kept:
        """From a serve's ``logits`` configuration: under its ``private`` directory (`boundary_tables.private_root` of
        ``public``)."""
        return cls(BT.private(BT.private_root(cfg["private"], *public)) / "logits")

    def write(self, name: str, head: Mapping[str, Any], rows: np.ndarray, salts: bytes) -> None:
        d = self.dir / name
        for q in (self.dir, d):
            BT.private(q)
        BT._bytes(d / ROWS, np.ascontiguousarray(rows, dtype="<u2").tobytes())
        BT._bytes(d / SALTS, bytes(salts))
        BT._bytes(d / HEAD, (json.dumps(dict(head)) + "\n").encode())

    def load(self, name: str) -> LK.Port:
        """Class ``name``'s value (`stem`) as a later statement reads its rows (`links.Port`), recommitted from its files
        and refused unless it has the root it was kept with."""
        d = self.dir / name
        h = json.loads((d / HEAD).read_text())
        n, w = int(h["leaves"]), int(h["words"])
        raw, ys = (d / ROWS).read_bytes(), (d / SALTS).read_bytes()
        if len(raw) != 2 * n * w or len(ys) != n * CS.SALT_BYTES:
            raise ValueError(f"{h['value']}: its files hold {len(raw)} row bytes and {len(ys)} salt bytes, not {n} rows' "
                             f"{2 * n * w} and {n * CS.SALT_BYTES}")
        rows = np.frombuffer(raw, dtype="<u2").reshape(n, w).astype(np.uint16)
        dom = BR.window_domain(h["program"], h["partition"], h["value"], n, h["run"])
        if dom.domain_id.hex() != h["domain"]:
            raise ValueError(f"{h['value']}: its kept domain is not the one its program, partition and run bind")
        roots, bc = CS.commit_ports({h["value"]: dom}, {h["value"]: rows}, {}, {h["value"]: ys},
                                    schemas={h["value"]: SCHEMA})
        if roots[h["value"]].hex() != h["root"]:
            raise ValueError(f"{h['value']}: its files recommit to another root than the one kept")
        return LK.Port(h["value"], SCHEMA, w, dom, bytes(bc[h["value"]]), rows, ys)


def load(directory: str | Path, window: int, run: str, prompt: int, outputs: int, stopped: bool | None) -> LK.Port:
    """The class's value as rank 0 kept it under a serve's ``logits`` configuration's ``private`` directory
    ``directory``: what the interior opens a sampled span's select inputs from, at round 1's root. On the node only."""
    return Kept(Path(directory).resolve() / "logits").load(stem(window, run, prompt, outputs, stopped))


class Builder:
    """Rank 0's commitment of each class's value: `build` answers a class's `assignment` with the value's entry, its rows
    those `span_rows` lays out of each span's in ``logits`` in slot order, each salt fresh from the OS, and kept in
    ``keep`` before the entry is given."""

    def __init__(self, logits: Logits, keep: Kept | None = None) -> None:
        self.logits, self.keep = logits, keep

    def build(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """The value's entry, or why there is none."""
        head = {k: doc.get(k) for k in ("window", "run", "prompt", "outputs", "stopped")}
        try:
            return {**head, "entry": self._commit(doc), "error": None}
        except (ValueError, KeyError, TypeError, OSError) as e:
            for s in doc.get("spans") or ():
                self.logits.take(str(s[0]))
            return {**head, "entry": None, "error": f"{type(e).__name__}: {e}"}

    def _commit(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        if self.logits.failed:
            raise ValueError(f"rank 0's logits are incomplete: {self.logits.failed}")
        O, per, words, port = int(doc["outputs"]), int(doc["rows"]), int(doc["words"]), str(doc["value"])
        out = []
        for r, (engine_id, served, finish) in enumerate(doc["spans"]):
            rows = self.logits.take(str(engine_id))
            n = 0 if rows is None else len(rows)
            if n != served and not (finish in ("abort", "error", None) and n > served):
                raise ValueError(f"slot {r}: rank 0 holds {n} rows of logits for its {served} ids served ({finish})")
            got = span_rows(rows[:served] if rows is not None else np.zeros((0, per // O * words), np.uint16), O, words)
            if got.shape != (per, words):
                raise ValueError(f"slot {r}: {got.shape[0]} rows of {got.shape[1]} words, not the class's {per} of {words}")
            out.append(got)
        rows = np.concatenate(out)
        n = len(rows)
        dom = BR.window_domain(str(doc["program"]), str(doc["partition"]), port, n, str(doc["run"]))
        salts = os.urandom(CS.SALT_BYTES * n)
        roots, _bc = CS.commit_ports({port: dom}, {port: rows}, {}, {port: salts}, schemas={port: SCHEMA})
        entry = {"value": port, "root": roots[port].hex(), "leaves": n, "words": words, "domain": dom.domain_id.hex()}
        if self.keep is not None:
            name = stem(doc["window"], doc["run"], doc["prompt"], doc["outputs"], doc.get("stopped"))
            self.keep.write(name, {**entry, "program": doc["program"], "partition": doc["partition"], "run": doc["run"]},
                            rows, salts)
        return entry


def _name(doc: Mapping[str, Any]) -> str:
    return stem(doc["window"], doc["run"], doc["prompt"], doc["outputs"], doc.get("stopped")) + ".json"


class Channel(BT.Channel):
    """`boundary_tables.Channel` for the value: a class's assignment and rank 0's answer under its `stem`, so a stop class
    and its sibling at one (P, O) are two."""

    @classmethod
    def of(cls, cfg: Mapping[str, Any], *public: Path | None) -> Channel:
        """From a serve's ``logits`` configuration: ``private``, its directory (`boundary_tables.private_root` of
        ``public``), and ``timeout_s``."""
        return cls(BT.private(BT.private_root(cfg["private"], *public)) / "logits-channel",
                   timeout_s=float(cfg.get("timeout_s", 900.0)))

    def assign(self, doc: Mapping[str, Any]) -> None:
        BT._write(self.dir / "assign" / _name(doc), doc)

    def answer(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        """Rank 0's answer to assignment ``doc``, waited for."""
        path, t0 = self.dir / "roots" / _name(doc), self.clock()
        while not path.exists():
            if self.clock() - t0 > self.timeout_s:
                raise TimeoutError(f"rank 0 answered no logits for {_name(doc)} in {self.timeout_s:g} s")
            self.sleep(self.poll_s)
        return json.loads(path.read_text())


class LocalChannel:
    """`Channel` in one process: `assign` builds at once."""

    def __init__(self, builder: Builder) -> None:
        self.builder, self.done = builder, {}

    def assign(self, doc: Mapping[str, Any]) -> None:
        self.done[_name(doc)] = self.builder.build(doc)

    def answer(self, doc: Mapping[str, Any]) -> dict[str, Any]:
        return self.done.pop(_name(doc))

