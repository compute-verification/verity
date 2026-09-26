"""One record for every property, and the citation a run carries.

A property harness writes one document under its own `schema`: the evidence, `ok` (the property held over that evidence)
and `digest`, the sha256 of the canonical JSON of every other field.  `seal` adds the last two.  `REGISTRY` names every
property by the schema of its document; a run cites a record by `(name, digest)`, and a citation is trusted only after
`PropertyRecord.problems` recomputes the digest.  Records a run cites live in `<run>/properties/*.json`.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from verity_vllm.correspondence.runtime import canonical_json

DIR = "properties"


@dataclass(frozen=True)
class Property:
    name: str
    schema: str
    claim: str


REGISTRY: dict[str, Property] = {p.name: p for p in (
    Property("fa_tap_exactness", "verity-vllm/fa-tap-exactness/v1",
             "The FA tap build computes exactly the installed FA kernel's output, writes exactly the declared stream words, "
             "and skips a launch whose declared tile is not the kernel's."),
    Property("norm_tap_exactness", "verity-vllm/norm-tap-exactness/v1",
             "The tapped norm kernels (fused CUDA op, Triton copy) compute exactly the installed kernels' outputs, write every row's "
             "scale, and each scale is the IR model's scale bit for bit; the Triton copy's compiled arithmetic is the installed kernel's."),
    Property("noninterference", "verity-vllm/noninterference/v2",
             "At the record's tensor-parallel world, the observed engine generated the bare engine's tokens for every request (and, "
             "where boundary hashes were taken, every compared boundary value is bit-identical)."),
    Property("census", "verity-capture/census-reconciliation/v1",
             "Every device kernel the bare engine launched is claimed by a kernel the Match resolution declares or by the frozen "
             "kernel allowlist, at every step."),
    Property("golden", "verity-gen/golden-check/v1",
             "Every golden log still resolves to its recorded attribution under the current patterns."),
    Property("holdout", "verity-gen/holdouts/v1",
             "Every holdout run passed each of its invariants."),
    Property("difftest", "verity-gen/difftest/v1",
             "The quarantined Definition evaluates every produced case to the adapter's bytes."),
    Property("quarantine_lint", "verity-gen/quarantine-lint/v1",
             "Every quarantined Definition module keeps to the quarantine rules, with no duplicate names."),
    Property("protected", "verity-gen/provenance/v1",
             "The candidate diff touches no protected path."),
)}
BY_SCHEMA = {p.schema: p for p in REGISTRY.values()}


def digest_of(doc: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json({k: v for k, v in doc.items() if k != "digest"})).hexdigest()


def seal(doc: Mapping[str, Any], ok: bool) -> dict[str, Any]:
    """`doc` with `ok` and its `digest`; `doc["schema"]` must be a registered property's."""
    if doc.get("schema") not in BY_SCHEMA:
        raise ValueError(f"schema {doc.get('schema')!r} is not a registered property's")
    body = {k: v for k, v in doc.items() if k != "digest"}
    body["ok"] = bool(ok)
    return {**body, "digest": digest_of(body)}


@dataclass(frozen=True)
class PropertyRecord:
    doc: Mapping[str, Any]
    source: str | None = None

    @property
    def name(self) -> str | None:
        p = BY_SCHEMA.get(self.doc.get("schema"))
        return p.name if p else None

    @property
    def ok(self) -> bool:
        return self.doc.get("ok") is True

    @property
    def digest(self) -> str | None:
        return self.doc.get("digest")

    def problems(self) -> list[str]:
        out = [] if self.name else [f"schema {self.doc.get('schema')!r} is not a registered property's"]
        if not isinstance(self.doc.get("ok"), bool):
            out.append("no boolean ok")
        if self.digest != digest_of(self.doc):
            out.append("digest does not match the record")
        return out

    def citation(self) -> dict[str, Any]:
        return {"name": self.name, "schema": self.doc.get("schema"), "digest": self.digest, "ok": self.ok, "source": self.source,
                "problems": self.problems()}

    def write(self, run_dir: str | Path) -> Path:
        p = Path(run_dir) / DIR / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.doc, indent=1) + "\n")
        return p

    @classmethod
    def load(cls, path: str | Path, root: str | Path | None = None) -> "PropertyRecord":
        """An unreadable file loads as an empty document, so its citation names the problems instead of dropping the file."""
        p = Path(path)
        try:
            doc = json.loads(p.read_text())
        except (OSError, ValueError):
            doc = {}
        return cls(doc if isinstance(doc, dict) else {}, str(p.relative_to(root)) if root else str(p))


def records_of(run_dir: str | Path) -> list[PropertyRecord]:
    """The property records a run cites: every `<run_dir>/properties/*.json`, by file name."""
    d = Path(run_dir)
    return [PropertyRecord.load(p, d) for p in sorted((d / DIR).glob("*.json"))]


def citations(run_dir: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    """The citation of every record of the run, and a note for each record that does not hold or whose digest does not recompute."""
    cites = [r.citation() for r in records_of(run_dir)]
    notes = [f"property {c['name'] or c['schema']} ({c['source']}): {'; '.join(c['problems']) or 'does not hold'}"
             for c in cites if c["problems"] or not c["ok"]]
    return cites, notes
