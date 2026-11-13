"""Pearl+'s window record W: a ``verity/registration/v2`` record of every call's A rows and y of the rounds the prover
retains and every weight's B rows they read, written after the retention and before the verifier's coin and the
one-stage audit's sample, so a sampled proof block reads them as output values of W (`registration`'s reused values)
under the roots Pearl+'s verdict credits.

Round s is a pass's place among ``served.json``'s passes with calls (the verdict's numbering, `verify_run.tie`), so the
warm-up's passes are rounds 0 and 1 and a dropped pass keeps its number; call c is the c-th call of the pass's manifest.
W lists only the retained rounds the timed call made (`write`: a retained warm-up round is refused, since the verdict
ties every retained round to W), each under its own number. Each call registers its A rows ``pearl-a/<s>/<c>``: its m₁₆
committed rows of K FP32 words, at its ``root_a``, under `audit.activation_binding`'s domain; and its y
``pearl-y/<s>/<c>``: its real rows of N FP32 words, at its y root, under `served_y.y_domain`'s (`entries`). Each weight w
the calls read registers its B rows once, ``pearl-b/<w>``: its n rows of K FP32 words, at the weight's root, under
`audit.weight_binding`'s domain (`b_entries`), the tree PoUW reads B from and the epoch's weights' root folds, so the
weights PoUW credits are a registered value of the one log. All are hm96 rows of 32-bit words, the entries
`verify_run.tie` holds to each call's and weight's root and recomputed domain.

W states its version in its ``window`` header (`VERSION`, `version_of`). Version 2 registers the B rows. A W written
before W registered them states none: it is version 1, its values the calls' alone, which `record` and `reuse` still
build at that version, so it stays readable, and `verify_run.tie` reads it as a tie of the calls with the weights
untied, by name (`verify_run.WEIGHTS_UNTIED`).

``served_passes.py retain --window-record W`` writes it: it drops the warm-up's rounds, writes the retention, then W over
the rounds it retains. It reads the served run's ``served.json`` (`SERVED_SCHEMA`: its passes as `served_passes` records
them, and its coins, `coins_of`), which ``served_passes.py serve`` and serve-openai's Pearl+ arm (`serve_pearl`) write.

The audited window's registration, chained right after W, registers each call's value as W's (`reuse`): the rows the
audited circuit's proof blocks make of it, committed as Pearl+ committed them, so its entry is W's, and one-stage's R7
takes it as an output value of W (records_audit's `tie_of`). The circuit is cut at its linears and reads no B row, so
W's B rows are no value of it."""
from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np
from verity.protocols.pouw.verifier.audit import HM96, activation_binding, hm96_salt, rows_domain, weight_binding
from verity.protocols.spot.verifier.one_stage import registered as G, registration as REG
from verity.commitments import CommitmentDomain
from verity.commitments.merkle import tree_levels

from verity_vllm.protocols.pouw import pearl_c as PPD
from verity_vllm.protocols.pouw.pearl_c import served_y as SY
from verity_vllm.records.pearl_served import SCHEMA as SERVED_SCHEMA

#: the width of a word of A's and y's committed rows: each BF16 word widened to FP32
WORD_BITS = 32
#: W's version, its ``window`` header's ``version``: 2 registers each weight's B rows; a W that states none is version 1,
#: written before W registered them, whose values are its calls' alone
VERSION = 2
VERSIONS = (1, VERSION)
#: why `version_of` refuses a W
UNKNOWN_VERSION = (f"W's window states a version other than {VERSION}: a W states version {VERSION}, or none (version 1, "
                   "written before W registered the weights)")
#: what a served pass's record keeps of its device run's pass (`PearlCDeviceRun.details` ``passes``)
KEEP = ("units", "base", "tiles", "screened_rows", "failing_rows", "unsure_rows", "excluded_rows", "transcript_root",
        "activation_roots", "schedule", "manifest")


def macs(manifest: str | None) -> int | None:
    """A retained pass's linear MACs, its calls' real rows × n × k from its manifest; None without one."""
    if not manifest:
        return None
    return sum(u["real_rows"] * u["n"] * u["k"] for u in json.loads(Path(manifest).read_text())["matmuls"])


def served_passes(passes: Sequence[Mapping[str, Any]], warm: int = 0) -> list[dict[str, Any]]:
    """A served run's passes (its device run's ``details["passes"]``, in order) as ``served.json`` records them: each one's
    number (its ``pass-NNNNN``), whether the timed call made it (all but the first ``warm``, a warm-up's), what `KEEP`
    keeps of it and its `macs`."""
    return [{"pass": i, "timed": i >= warm, **{k: p[k] for k in KEEP if k in p}, "macs": macs(p.get("manifest"))}
            for i, p in enumerate(passes)]


def coins_of(option: Any, reserved: Any = None) -> dict[str, Any]:
    """Where the PoUW option ``option`` took its salt coin: its ``coin_source``, the gate's answer it carries (``session``,
    no secret), its salt epoch's number (``pass_index``) and the root its salt coin follows (that epoch's root over the
    weights' root, `salt.epoch_open_root`; the weights' root for a coin held in-process), its salt, how its session closed
    (``details.challenger``, once the engine has closed) and the session head a retained pass's manifest names
    (``manifest_head``, `pearl_c.manifest.session_head`); ``reserved`` the gate's reserve answer, when one was taken."""
    from verity.protocols.pouw.verifier import salt as PS

    from verity_vllm.protocols.pouw.pearl_c.manifest import session_head

    c = option.commitment()
    k, w = c.get("pass_index"), c.get("weights_root")
    after = w if k is None or w is None else PS.epoch_open_root(bytes.fromhex(w), k).hex()
    return {"source": c.get("coin_source", "in-process"), "reserved": reserved, "session": c.get("session"),
            "pass_index": k, "weights_root": w, "salt_after": after, "salt": c.get("salt"),
            "closed": option.info().get("challenger"), "manifest_head": session_head(c)}


def a_value(s: int, c: int) -> str:
    """Round ``s``'s call ``c``'s A rows."""
    return f"pearl-a/{s}/{c}"


def y_value(s: int, c: int) -> str:
    """Round ``s``'s call ``c``'s y."""
    return f"pearl-y/{s}/{c}"


def b_value(w: int) -> str:
    """Weight ``w``'s B rows."""
    return f"pearl-b/{w}"


def _hm96(man: Mapping[str, Any], s: int) -> None:
    hashing = PPD.scheme_of(man["scheme"]).commitment_hash
    if hashing != HM96:
        raise ValueError(f"round {s}'s pass commits {hashing} rows, and W registers {HM96} rows")


def entries(man: Mapping[str, Any], s: int) -> list[dict[str, Any]]:
    """Round ``s``'s entries, from its pass's manifest ``man``: each call's A rows, then its y. ValueError for a pass whose
    rows aren't hm96 rows, or a call that commits no y."""
    _hm96(man, s)
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


def b_entries(man: Mapping[str, Any], s: int) -> list[dict[str, Any]]:
    """The B rows of the weights round ``s``'s pass reads, from its manifest ``man``, by ascending weight id: each weight's
    n rows of K FP32 words (its calls' n and k), at its root, under `audit.weight_binding`'s domain. ValueError for a pass
    whose rows aren't hm96 rows, a weight two of its calls read at other shapes, or a weight it commits no root of."""
    _hm96(man, s)
    shapes: dict[int, tuple[int, int]] = {}
    for c, u in enumerate(man["matmuls"]):
        if (had := shapes.setdefault(u["weight"], (u["n"], u["k"]))) != (u["n"], u["k"]):
            raise ValueError(f"round {s}'s call {c} (index {u['index']}) reads weight {u['weight']} as {u['n']} rows of "
                             f"{u['k']} words, and an earlier call as {had[0]} of {had[1]}: W registers one shape of each "
                             "weight's B rows")
    out = []
    for w in sorted(shapes):
        n, k = shapes[w]
        if "root" not in man.get("weights", {}).get(str(w), {}):
            raise ValueError(f"round {s}'s pass commits no root of weight {w}, whose B rows W registers")
        out.append({"value": b_value(w), "root": bytes.fromhex(man["weights"][str(w)]["root"]).hex(), "leaves": n,
                    "words": k, "domain": rows_domain(weight_binding(w), -2, n, HM96).domain_id.hex(), G.WIDTH_KEY: WORD_BITS})
    return out


def _weights(manifests: Sequence[Mapping[str, Any]], rounds: Sequence[int]) -> list[dict[str, Any]]:
    """Every weight's B rows over the passes (`b_entries`), each once, in the order the passes first read them. ValueError
    for a weight two passes commit as other rows."""
    out: dict[str, dict[str, Any]] = {}
    for s, m in zip(rounds, manifests):
        for e in b_entries(m, s):
            if out.setdefault(e["value"], e) != e:
                raise ValueError(f"round {s}'s pass commits {e['value']} as other rows than an earlier round's: W registers "
                                 "one install's weights")
    return list(out.values())


def _rounds(manifests: Sequence[Mapping[str, Any]], rounds: Sequence[int] | None) -> list[int]:
    got = list(range(len(manifests))) if rounds is None else list(rounds)
    if len(got) != len(manifests) or not all(type(s) is int and s >= 0 for s in got) or got != sorted(set(got)):
        raise ValueError(f"W's rounds {got} are not distinct rounds in order, one a pass ({len(manifests)} passes)")
    return got


def version_of(w: Mapping[str, Any]) -> int:
    """W's version (`VERSION`): its ``window`` header's ``version``, or 1 where it states none. ValueError for any other
    (`UNKNOWN_VERSION`), a stated 1 included, so each version has one form."""
    head = w.get("window")
    if not isinstance(head, Mapping) or "version" not in head:
        return 1
    if type(head["version"]) is not int or head["version"] != VERSION:
        raise ValueError(UNKNOWN_VERSION)
    return VERSION


def record(manifests: Sequence[Mapping[str, Any]], previous: str | None = None,
           rounds: Sequence[int] | None = None, version: int = VERSION) -> dict[str, Any]:
    """W at ``version`` over passes' manifests in order, pass i round ``rounds[i]`` (default i), chained after
    ``previous`` (the digest of the prover's last one-stage record, `registration.digest`; None for its first): every
    call's entries, round by round, then at version 2 every weight's B rows, its header stating the version (version 1
    states none and registers no weight). ValueError for no pass, rounds that aren't one a pass in order, passes of two
    salts, a weight two passes commit as other rows, or a version that isn't one of `VERSIONS`."""
    if version not in VERSIONS:
        raise ValueError(f"W is version {' or '.join(map(str, VERSIONS))}, not {version!r}")
    if not manifests:
        raise ValueError("W registers the calls of a run's retained rounds, and it retains none")
    head = manifests[0]
    if other := next((m["salt"] for m in manifests if m["salt"] != head["salt"]), None):
        raise ValueError(f"W is one epoch's: its passes are under salts {head['salt'][:16]}… and {other[:16]}…")
    at = _rounds(manifests, rounds)
    tied = version == VERSION
    return {"format": REG.FORMAT_V2, "protocol": REG.PROTOCOL,
            "window": {"kind": "served", "entry": head["entry"], "scheme": head["scheme"], "salt": head["salt"],
                       **({"version": VERSION} if tied else {})},
            "registered": [*(e for s, m in zip(at, manifests) for e in entries(m, s)),
                           *(_weights(manifests, at) if tied else [])],
            "previous": previous}


def write(path: str | Path, passes: Sequence[Mapping[str, Any]], rounds: Sequence[int],
          previous: str | None = None) -> dict[str, Any]:
    """W over the retained ``rounds`` (places among ``passes``, ``served.json``'s, with calls: the retention's) written to
    ``path``, and returned. ValueError for a round that is no pass with calls, a warm-up's round (``timed`` false: W
    lists only the timed call's rounds, so a tied run drops the warm-up's), or a round that retained no manifest."""
    held = [p for p in passes if p.get("units")]
    manifests = []
    for s in rounds:
        if not 0 <= s < len(held):
            raise ValueError(f"no round {s} among the run's {len(held)} passes with calls")
        if not held[s].get("timed", True):
            raise ValueError(f"round {s} is the warm-up's (timed false) and retained: W lists only the timed call's rounds, "
                             f"and the verdict ties every retained round to W, so drop it before the coin "
                             f"(served_passes.py retain --drop {s}, or retain --window-record W)")
        if not held[s].get("manifest"):
            raise ValueError(f"round {s} retained no manifest, which W reads: serve with the pouw setting 'transcript_dir'")
        manifests.append(json.loads(Path(held[s]["manifest"]).read_text()))
    w = record(manifests, previous, rounds)
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(w, indent=1) + "\n")
    return w


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
          rows_of: Callable[[str, int, np.ndarray], np.ndarray], rounds: Sequence[int] | None = None) -> dict[str, Committed]:
    """The calls' values W registers (``w``, the window record of ``manifests`` at ``rounds`` and at the version it states:
    `record`, `version_of`; its B rows are no value of the audited circuit), each as the audited circuit's registered
    value: the rows ``rows_of(name, rows,
    empty)`` gives (`window_rows.Rows.value`: u16 words, an FP32 word's two low first, every row no proof block makes
    ``empty``, zero), each salted with `audit.hm96_salt` under its pass's ``hm96_salt_key`` at the call's index for A and
    `served_y.HM96_Y` past it for y. ValueError unless ``w`` is the record of ``manifests`` and each value's entry is W's:
    the proof blocks don't make the rows Pearl+ committed, or the key is another."""
    if dict(w) != record(manifests, w.get("previous"), rounds, version_of(w)):
        raise ValueError("W is not the window record of these passes (`record`), whose calls' values the audited window reads")
    out: dict[str, Committed] = {}
    for s, man in zip(_rounds(manifests, rounds), manifests):
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


__all__ = ["KEEP", "SERVED_SCHEMA", "UNKNOWN_VERSION", "VERSION", "VERSIONS", "WORD_BITS", "Committed", "a_value", "b_entries",
           "b_value", "coins_of", "entries", "macs", "record", "reuse", "served_passes", "version_of", "write", "y_value"]
