"""Pearl+'s window record W: a ``verity/registration/v2`` record of every call's A rows and y that the window's passes
committed, written at the window's close, before the one-stage audit's sample, so a sampled proof block reads them as
output values of W (`registration`'s reused values) under the roots Pearl+'s verdict credits.

Round s is the s-th pass with calls of the run, in the order its ended passes are recorded (``details["passes"]``, the
order ``served.json`` lists them and the verdict reads them), a warm-up's included; call c is the c-th call of the pass's
manifest. Each call registers its A rows ``pearl-a/<s>/<c>``: its m₁₆ committed rows of K FP32 words, at its ``root_a``,
under `audit.activation_binding`'s domain; and its y ``pearl-y/<s>/<c>``: its real rows of N FP32 words, at its y root,
under `served_y.y_domain`'s (`entries`). Both are hm96 rows of 32-bit words, the entries `verify_run.tie` holds to each
call's root and recomputed domain.

The pouw setting ``window_record`` (W's path) turns it on and needs ``transcript_dir``, whose manifests W reads: every pass
of the window is retained. Unset, nothing is written and the run is unchanged (`at_close`).

The audited window's registration, chained right after W, registers each value as W's (`reuse`): the rows the audited
circuit's proof blocks make of it, committed as Pearl+ committed them, so its entry is W's, and one-stage's R7 takes it
as an output value of W (records_audit's `tie_of`)."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from verity.core.protocols.pouw.audit import HM96, activation_binding, hm96_salt, rows_domain
from verity.core.protocols.sampled_proofs.one_stage import registered as G, registration as REG
from verity.ml.commitments import CommitmentDomain
from verity.ml.commitments.merkle import tree_levels

from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY

#: the pouw setting naming W's path
KEY = "window_record"
#: the width of a word of A's and y's committed rows: each BF16 word widened to FP32
WORD_BITS = 32


def a_value(s: int, c: int) -> str:
    """Round ``s``'s call ``c``'s A rows."""
    return f"pearl-a/{s}/{c}"


def y_value(s: int, c: int) -> str:
    """Round ``s``'s call ``c``'s y."""
    return f"pearl-y/{s}/{c}"


def entries(man: Mapping[str, Any], s: int) -> list[dict[str, Any]]:
    """Round ``s``'s entries, from its pass's manifest ``man``: each call's A rows, then its y. ValueError for a pass whose
    rows aren't hm96 rows, or a call that commits no y."""
    hashing = PPD.scheme_of(man["scheme"]).commitment_hash
    if hashing != HM96:
        raise ValueError(f"round {s}'s pass commits {hashing} rows, and W registers {HM96} rows")
    salt, out = bytes.fromhex(man["salt"]), []
    for c, u in enumerate(man["matmuls"]):
        if "y" not in u:
            raise ValueError(f"round {s}'s call {c} (index {u['index']}) commits no y for W to register")
        a = rows_domain(activation_binding(salt, u["index"]), u["index"], u["m"], HM96)
        y = SY.y_domain(salt, u["index"], u["real_rows"])
        out += [{"value": a_value(s, c), "root": u["root_a"], "leaves": u["m"], "words": u["k"],
                 "domain": a.domain_id.hex(), G.WIDTH_KEY: WORD_BITS},
                {"value": y_value(s, c), "root": u["y"]["root"], "leaves": u["real_rows"], "words": u["n"],
                 "domain": y.domain_id.hex(), G.WIDTH_KEY: WORD_BITS}]
    return out


def record(manifests: Sequence[Mapping[str, Any]], previous: str | None = None) -> dict[str, Any]:
    """W over a run's passes with calls, their manifests in order, chained after ``previous`` (the digest of the prover's
    last one-stage record, `registration.digest`; None for its first). ValueError for no pass, or passes of two salts."""
    if not manifests:
        raise ValueError("W registers the calls of a run's passes, and the run has no pass with calls")
    head = manifests[0]
    if other := next((m["salt"] for m in manifests if m["salt"] != head["salt"]), None):
        raise ValueError(f"W is one epoch's: its passes are under salts {head['salt'][:16]}… and {other[:16]}…")
    return {"format": REG.FORMAT_V2, "protocol": REG.PROTOCOL,
            "window": {"kind": "served", "entry": head["entry"], "scheme": head["scheme"], "salt": head["salt"]},
            "registered": [e for s, m in enumerate(manifests) for e in entries(m, s)], "previous": previous}


def write(path: str | Path, passes: Sequence[Mapping[str, Any]], previous: str | None = None) -> dict[str, Any]:
    """W over a run's ended passes (``details["passes"]``, each pass with calls read from its manifest) written to
    ``path``, and returned. ValueError for a pass with calls that retained no manifest."""
    manifests = []
    for i, p in enumerate(passes):
        if not p.get("units"):
            continue
        if not p.get("manifest"):
            raise ValueError(f"pass {i} has calls and retained no manifest, which W reads: set the pouw setting "
                             "'transcript_dir'")
        manifests.append(json.loads(Path(p["manifest"]).read_text()))
    w = record(manifests, previous)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(w, indent=1) + "\n")
    return w


def at_close(release: Callable[[], None], run: Any, path: str | None) -> Callable[[], None]:
    """``release``, then with ``path`` W over ``run``'s passes written there (`write`), its path, digest and value count
    in ``run.details["window_record"]``; ``release`` itself without ``path``."""
    if path is None:
        return release

    def close() -> None:
        release()
        w = write(path, run.details["passes"])
        run.details[KEY] = {"path": str(path), "digest": REG.digest(w), "values": len(w["registered"])}
    return close


@dataclass(frozen=True)
class Committed(G.Registered):
    """A call's A rows or y as a registered value of the audited circuit (`reuse`): its FP32 words in ``row-seg/v1`` rows,
    salted as Pearl+ salts them, in a tree under ``call_domain``, the ``frame-v3-sha512`` domain of the call's binding,
    index and rows (W's entry's domain id), so its entry is W's."""

    call_domain: CommitmentDomain | None = None

    @property
    def domain(self) -> CommitmentDomain:
        return self.call_domain


def reuse(w: Mapping[str, Any], manifests: Sequence[Mapping[str, Any]],
          rows_of: Callable[[str, int, np.ndarray], np.ndarray]) -> dict[str, Committed]:
    """The values W registers (``w``, the window record of ``manifests``: `record`), each as the audited circuit's
    registered value: the rows ``rows_of(name, rows, empty)`` gives (`window_rows.Rows.value`: u16 words, an FP32 word's
    two low first, every row no proof block makes ``empty``, zero), each salted with `audit.hm96_salt` under its pass's
    ``hm96_salt_key`` at the call's index for A and `served_y.HM96_Y` past it for y. ValueError unless ``w`` is the record
    of ``manifests`` and each value's entry is W's: the proof blocks don't make the rows Pearl+ committed, or the key is
    another."""
    if dict(w) != record(manifests, w.get("previous")):
        raise ValueError("W is not the window record of these passes (`record`), whose calls' values the audited window reads")
    out: dict[str, Committed] = {}
    for s, man in enumerate(manifests):
        salt, key, want = bytes.fromhex(man["salt"]), bytes.fromhex(man["hm96_salt_key"]), iter(entries(man, s))
        for u in man["matmuls"]:
            i = u["index"]
            for binding, nonce in ((activation_binding(salt, i), i), (SY.y_binding(salt, i), SY.HM96_Y + i)):
                e = next(want)
                n, k = e["leaves"], e["words"]
                words = np.ascontiguousarray(rows_of(e["value"], n, np.zeros(2 * k, np.uint16)), "<u2").view("<u4")
                base = G.Registered.commit(man["scheme"], e["value"], words.tolist(), [hm96_salt(key, nonce, j) for j in range(n)],
                                           word_bits=WORD_BITS, schema=G.SCHEMA_SEG)
                dom = rows_domain(binding, i, n, "sha512")
                levels = tree_levels(dom, [G.leaf(dom, j, bc, G.SCHEMA_SEG) for j, bc in enumerate(base.bcs)])
                got = out[e["value"]] = Committed(**{**{f.name: getattr(base, f.name) for f in fields(base)}, "levels": levels},
                                                  call_domain=dom)
                if got.entry() != e:
                    raise ValueError(f"{e['value']}: the rows the audited circuit's proof blocks make of it are not the rows "
                                     f"Pearl+ committed (root {got.root.hex()[:16]}… for W's {e['root'][:16]}…): the model "
                                     "didn't make them, or the salt key is another")
    return out


def setting(raw: Mapping[str, Any]) -> str | None:
    """The pouw setting ``window_record``: None, or W's path, which needs ``transcript_dir``. ValueError otherwise."""
    path = raw.get(KEY)
    if path is None:
        return None
    if not isinstance(path, str) or not path:
        raise ValueError(f"pouw setting '{KEY}' is a path string, where W goes at the window's close")
    if raw.get("transcript_dir") is None:
        raise ValueError(f"pouw setting '{KEY}' needs 'transcript_dir': W reads each pass's retained manifest")
    return path


__all__ = ["KEY", "WORD_BITS", "Committed", "a_value", "at_close", "entries", "record", "reuse", "setting", "write", "y_value"]
