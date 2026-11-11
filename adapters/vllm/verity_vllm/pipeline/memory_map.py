"""Each request Build's census and KV rows, beside its ``descriptor.json.gz``: the Program of record's census/v1 and its KV
rows (`commit.memory_map.census_of`, `rows`), written once by the Build (`census_entry`, `cache`) and read, or computed and
cached, by whoever lays out its KV (`of_build`): the Commit's opt-in memory map (PoUB's, ``commit --memory-map``).  The
census decides no Build.
"""
from __future__ import annotations

import gzip
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from verity_vllm.check.replay.cached_prefix import layout_of
from verity_vllm.commit import memory_map as MM


#: beside a request Build's ``descriptor.json.gz``: its Program's census (`MM.census_of`) and KV rows (`MM.rows_doc`), each naming
#: the Program by digest; KV rows of another schema are recomputed (v1 refused MLA's latent rows, v2 a warm Program's cached prefix)
CENSUS, KV_ROWS = "census.json.gz", "kv_rows.json.gz"
KV_ROWS_SCHEMA = "verity-vllm/kv-rows/v3"


def _read(path: Path, digest: str, schema: str | None = None) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    with gzip.open(path, "rt") as f:
        doc = json.load(f)
    return doc if doc.get("program") == digest and schema in (None, doc.get("schema")) else None


def _put(path: Path, doc: Mapping[str, Any]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with gzip.open(tmp, "wt", compresslevel=6) as f:
        json.dump(doc, f, separators=(",", ":"))
    tmp.replace(path)


def cache(d: str | Path, program: Any, digest: str, build: str | Path | None = None) -> dict[str, Any]:
    """Write Build dir ``d``'s `CENSUS` and `KV_ROWS` for its request Program ``program`` (``digest``), the census read back when
    one beside it already names the Program, a warm Program's cached prefix as the Build (``build``, else ``d``) laid it out
    (``prefix_cache.json``); a Program with no KV a memory map lays out gets the reason in place of its rows."""
    d = Path(d)
    c = _read(d / CENSUS, digest)
    if c is None:
        c = MM.census_of(program, digest)
        _put(d / CENSUS, c)
    doc: dict[str, Any] = {"schema": KV_ROWS_SCHEMA, "program": digest, "census": c["digest"]}
    try:
        doc.update(MM.rows_doc(MM.rows(program, c, layout_of(str(d if build is None else build)))))
    except ValueError as e:
        doc["refused"] = str(e)
    _put(d / KV_ROWS, doc)
    return {"census": c["digest"], "kv_rows": "refused" not in doc}


def census_entry(d: str | Path, program: Any, digest: str) -> dict[str, Any]:
    """`cache`'s result for the Build's `result.json`, or its error in its place: the census serves the Commit's opt-in
    memory map, so it decides no Build."""
    try:
        return cache(d, program, digest)
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:300]}


def of_build(d: str | Path, digest: str, at: str | Path | None = None) -> MM.Rows:
    """The KV rows of Build dir ``d``'s request Program ``digest``: its `KV_ROWS` when they name the Program, else computed from its
    descriptor (refused unless that is Program ``digest``) and cached beside it (`cache`), once per Program digest; read and
    cached at ``at`` in place of ``d`` when given, ``d`` only read."""
    from verity.definitions.boolean import load_program

    d = Path(d)
    at = d if at is None else Path(at)
    doc = _read(at / KV_ROWS, digest, KV_ROWS_SCHEMA)
    if doc is None:
        program, got = load_program(d / "descriptor.json.gz")
        if got != digest:
            raise ValueError(f"{d}: descriptor.json.gz is Program {got[:16]}, its instance rows {str(digest)[:16]}")
        at.mkdir(parents=True, exist_ok=True)
        cache(at, program, digest, d)
        doc = _read(at / KV_ROWS, digest, KV_ROWS_SCHEMA)
    if "refused" in doc:
        raise ValueError(f"{d}: {doc['refused']}")
    return MM.rows_of(doc)
