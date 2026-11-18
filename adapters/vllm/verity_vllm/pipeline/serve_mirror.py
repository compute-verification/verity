"""The serve dir's records that may leave it: `PUBLIC` names each mirrored file's public fields with their kinds, and `Mirror`
copies those fields, and nothing else, into the run's ``$OUT`` (serve-openai's ``--copy-records-to``) after each close and
each seal."""
from __future__ import annotations

import json
import os
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

#: the kinds of field that may leave the node (AGENTS.md's public kinds): protocol structure, the protocol's outputs, the
#: verifier's coins and draws, and hiding commitments
KINDS = ("structure", "output", "coin", "commitment")

#: The two lists below are written out, not read from their sources, so that a field added at a source stays in the serve
#: dir until it is listed here; a test holds each name to its source, and `LEFT_OUT` states what each source has that its
#: list leaves out.
#: `TargetProfile.to_json`'s fields but ``card`` (a file's name)
_TARGET = {k: "structure" for k in (
    "compute_capability", "num_sms", "batch_invariant", "attention_backend", "vllm_revision", "numerics_profile",
    "flash_attn_version", "compilation", "fp8_block_gemm", "int8_exact", "moe_backend", "async_scheduling", "moe_construction",
    "fa3_construction", "weight_only_calls", "sampler_construction", "fp8_scale_construction", "protocols", "ir",
    "registered_values")}
#: `served_statics`' names (`compile_slice.statics`)
_STATICS = {k: "structure" for k in (
    "R", "W", "A", "M", "T", "L", "B", "H", "CH", "WM", "FULL", "PC", "NB", "IW", "AS", "EW", "Q", "D", "P", "K", "CN", "CTW",
    "CU", "G")}
#: the names `TargetProfile.to_json` and `compile_slice.STATICS` write that `_TARGET` and `_STATICS` leave out
LEFT_OUT = {"target": ("card",), "served_rows_statics": ()}

#: Every field of the serve dir's records that may leave it for the run's ``$OUT`` (`Mirror`), by file, each with its kind
#: (`KINDS`); a mapping names a record's nested fields. A field not named, and a file not here, stays in the serve dir: the
#: requests and their users, counts, sizes, timings, paths, hosts, error text and whole dumped dictionaries among them.
#: ``seed_coin`` is out because it is flipped in-process, and ``binding_map_digest`` because it is unsalted. Each of these is
#: none of the four kinds: a window's ``step_roots``, whose number is the window's steps; ``passed``, the serving node's own
#: check of its Commit and readback, not the protocol's output; ``run_id``, a launcher's free string the mirror holds to no
#: fixed form; and ``served_rows``' G and n_bits, which the statics fix (a row is G = P steps, each step's bits at R, W and
#: A, idle steps zero), so a reader has them once, from ``serve.json``'s ``served_rows_statics``.
PUBLIC: dict[str, dict[str, Any]] = {
    "serve.json": {"precision": "structure", "protocol": "structure", "default_seed": "structure", "scheme": "structure",
                   "window_steps": "structure", "window_requests": "structure", "window_gb": "structure",
                   "served_rows_statics": _STATICS, "seal_off_loop": "structure", "seal_hold_gb": "structure",
                   "target": _TARGET, "model": "structure", "revision": "structure", "weights": {"root": "commitment"}},
    "windows.jsonl": {"window": "structure", "run_root": "commitment",
                      "served_rows": {"binding": "commitment", "root": "commitment"}},
    "seals.jsonl": {"window": "structure", "run_root": "commitment", "commit_scheme": "structure"},
}



def check_kinds(table: Mapping[str, Any]) -> None:
    """Refuse a table any of whose fields has a kind not in `KINDS`, naming the file, the field and the kind."""
    def walk(name: str, path: str, spec: Any) -> None:
        if isinstance(spec, Mapping):
            for k, s in spec.items():
                walk(name, f"{path}.{k}" if path else k, s)
        elif spec not in KINDS:
            raise ValueError(f"{name}: {path}: kind {spec!r} is not one of KINDS")
    for name, spec in table.items():
        walk(name, "", spec)


check_kinds(PUBLIC)

#: the records mirrored into the run's $OUT, each as `PUBLIC` projects it
MIRRORED = tuple(PUBLIC)

_SCALARS = (str, int, float, bool, type(None))
_DROP = object()


def project(value: Any, spec: Any) -> Any:
    """``value`` as `PUBLIC`'s ``spec`` lets it leave: under a kind, a JSON scalar or a list of them, anything else dropped;
    under a mapping of names, those fields of a mapping (or of each mapping in a list), each projected in turn."""
    if isinstance(spec, str):
        if isinstance(value, _SCALARS) or (isinstance(value, list) and all(isinstance(x, _SCALARS) for x in value)):
            return value
        return _DROP
    if isinstance(value, list):
        got = [project(x, spec) for x in value]
        return [x for x in got if x is not _DROP]
    if not isinstance(value, Mapping):
        return _DROP
    got = {k: project(value[k], s) for k, s in spec.items() if k in value}
    return {k: v for k, v in got.items() if v is not _DROP}


class Mirror:
    """The serve dir's records (`MIRRORED`), each record projected through `PUBLIC`, written into ``dst`` (the run's ``$OUT``,
    which custody uploads), so their public fields outlive the serve dir: a JSON-lines file up to its last whole line, appended
    from where the last call read; any other file whole. A line that isn't a JSON object is left out (``errors``). Any thread."""

    def __init__(self, src: str | Path, dst: str | Path, names: Sequence[str] = MIRRORED) -> None:
        if unlisted := [n for n in names if n not in PUBLIC]:
            raise ValueError(f"{unlisted}: a record mirrors only as PUBLIC projects it, and PUBLIC lists none of its fields")
        self.src, self.dst, self.names = Path(src), Path(dst), tuple(names)
        self.errors: list[str] = []
        self._read: dict[str, int] = {}
        self._lock = threading.Lock()

    def __call__(self) -> None:
        with self._lock:
            self.dst.mkdir(parents=True, exist_ok=True)
            for name in self.names:
                try:
                    self._one(name)
                except OSError as e:
                    self.errors.append(f"{name}: {type(e).__name__}: {e}"[:300])

    def _record(self, name: str, raw: bytes) -> bytes | None:
        try:
            rec = json.loads(raw)
        except ValueError:
            rec = None
        if not isinstance(rec, Mapping):
            self.errors.append(f"{name}: a record that isn't a JSON object, left out")
            return None
        return json.dumps(project(rec, PUBLIC[name])).encode()

    def _one(self, name: str) -> None:
        s, d = self.src / name, self.dst / name
        if not s.is_file():
            return
        if s.suffix != ".jsonl":
            if (got := self._record(name, s.read_bytes())) is not None:
                part = d.with_name(d.name + ".part")
                part.write_bytes(got + b"\n")
                os.replace(part, d)
            return
        have = self._read.get(name, 0)
        with open(s, "rb") as f:
            size = os.fstat(f.fileno()).st_size
            have = have if have <= size else 0          # the serve dir's file is not the one read so far: project it again whole
            f.seek(have)
            data = f.read(size - have)
        data = data[:data.rfind(b"\n") + 1]
        lines = [got for x in data.splitlines() if x.strip() and (got := self._record(name, x)) is not None]
        if data or not have:
            with open(d, "ab" if have else "wb") as g:
                g.write(b"".join(x + b"\n" for x in lines))
        self._read[name] = have + len(data)


def records_mirror(out: Path, dst: str | None) -> Mirror | None:
    """`Mirror` of ``out``'s records into ``--copy-records-to`` (None without one, or when it is ``out``)."""
    if not dst or Path(dst).resolve() == Path(out).resolve():
        return None
    return Mirror(out, dst)
