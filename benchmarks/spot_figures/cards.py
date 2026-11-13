"""The SPOT figures' input cards (`spot-card/v1`): every number the figures read is a card, marked measured or modeled
with its source, published to the evidence store and kept here as `cards/SLOT.MODEL.json` with the art id it came from.

A card is one JSON object:

    {"schema": "spot-card/v1", "slot": SLOT, "model": "GLM-5.3" | "DeepSeek-V4-Pro" | null,
     "status": "measured" | "modeled", "provisional": bool, "source": "...", "runs": ["r..."], "owner": "@lane",
     "value": {...}}

`SLOTS` gives each slot's `value` fields, every number in the unit its field names; `model` is null only in the slots
that hold for every model (`MODEL_FREE`).  A proof block's Definition is named as circuit-check names the
specialization, the key `workload` and `proof_overhead` join on.  A `workload` card's strata are its proof blocks a
token over its ranks; a stratum whose Definition has no cut on main, sized at the grain it would take, is `modeled`, and
`served_macs_per_token` is the whole circuit's MACs a token where the strata leave some out; `params`, the model's
total parameters, scales the references that go as the model's size.  The `references` cards
of a model add up: each is kept as `cards/references.MODEL.ART.json`, under the first twelve hex digits of its art id,
and the figures plot every card's lines, with its `caption` appended to the caption of each panel it has lines on; a
line at 0 is drawn along its panel's floor and named in its caption.  A `partition` card is another cut of the model's circuit: its
strata take the place of the `workload` card's strata of the kinds it `replaces`, each with the k_log it is staged at,
which prices it until the `proof_overhead` card measures its type; they add up as `references` cards do, each Figure
1's line of its own.  A `verifier_cost` card gives the verifier's core-seconds a statement by block prefix, a family's
other types on the line through its members (or through the fixed seconds at size 0 and its one member), any other type
at the measured types' median, and the core-seconds that cost one GPU-second, which Figure 1 adds to each statement's
fixed seconds.  A `headline` card holds the post's headline rows, each with its cells (S, verifier step, scope) and its
`basis` (`BASES`): `served-measured`, `served-card-priced` or `design`, with what sets it; a `measured` headline card
holds only `served-measured` rows, and Figure 5 names each row's basis.  A `serving_rate` card is the engine's sustained
computed tokens, prompt and output, a GPU-second on the mix its `plan` names (the art id of the plan file or of the
record that holds it, or the id of the run that does), measured; `saturated` only where a rate above it was served and
computed no more, so a lower bound is `saturated` false on a `provisional` card, and `bound` says what held the rate
(`BOUNDS`).  No figure reads it unless asked to (`figure1.SERVING`).  A card reaches the figures by art id, a later one
naming the earlier card it replaces:

    uv run python benchmarks/spot_figures/cards.py check CARD.json ...       # refuses unknown and missing fields
    research data put --kind spot-card/v1 --file CARD.json --preserve --json
    uv run python benchmarks/spot_figures/cards.py fetch art:... [--replaces art:OLD]   # checks it, keeps it in cards/
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

HERE = Path(__file__).resolve().parent
CARDS = HERE / "cards"
SCHEMA = "spot-card/v1"
MODELS = ("GLM-5.3", "DeepSeek-V4-Pro")
MODEL_FREE = ("design", "commitment", "references", "commit_hillclimb")
#: the slots whose cards add up rather than replace one another: each is kept under its art id
MANY = ("references", "partition")
PANELS = ("exfiltration", "infiltration", "skipped_work")
CUTS = ("tokens", "kv", "layer_of_token", "every_block")
#: the server configurations that pay a serve tax of their own (`total_cost.CONFIGURATIONS`)
CONFIGURATIONS = ("tokens", "tokens_bi", "layer_of_token", "every_block")
RUN = re.compile(r"r\d{8}-\d{6}-[0-9a-f]{4}")
ART = re.compile(r"art:[0-9a-f]{64}")
#: what holds a serving rate where it is: the GPUs' compute; the engine's step, where requests waited with room in the
#: KV pool and the record does not tell the GPUs' arithmetic from a step's token budget; the KV pool's size; or not known
BOUNDS = ("compute", "engine-step", "kv-pool", "unknown")
NUM = (int, float)


@dataclass(frozen=True)
class F:
    """A field: its JSON types, whether it may be absent or null, the fields of each entry of a list of objects, and
    the values it may take."""

    kind: type | tuple
    required: bool = True
    nullable: bool = False
    each: dict | None = None
    choices: tuple | None = None


def _opt(kind: type | tuple, **kw) -> F:
    return F(kind, required=False, **kw)


ENVELOPE = {"schema": F(str, choices=(SCHEMA,)), "slot": F(str), "model": F(str, nullable=True, choices=MODELS),
            "status": F(str, choices=("measured", "modeled")), "provisional": F(bool), "source": F(str),
            "runs": F(list), "owner": F(str), "value": F(dict), "art": _opt(str)}
STRATUM = {"block": F(str), "kind": _opt(str), "per_token": F(NUM), "macs": F(NUM), "out_bits": F(NUM),
           "in_bits": _opt(NUM), "ands": _opt(NUM), "k_log": _opt(int), "commit_bytes": _opt(NUM),
           "modeled": _opt(bool)}
OVERHEAD_BLOCK = {"block": F(str), "stands_in": _opt(bool), "macs": F(NUM), "out_bits": _opt(NUM), "k_log": _opt(int),
                  "seconds_per_block": F(NUM), "fixed_seconds": F(NUM), "statement_blocks": F(int),
                  "overhead": F(NUM, nullable=True), "runs": _opt(list)}
CUT = {"cut": F(str, choices=CUTS), "bytes_per_token": F(NUM), "serve_share": F(NUM, nullable=True), "runs": _opt(list)}
LINE = {"panel": F(str, choices=PANELS), "label": F(str), "unit": F(str, choices=("bit", "FLOP", "AND")),
        "value": _opt(NUM), "by_scope": _opt(dict), "source": F(str)}
SERIES = {"scope": F(str), "budget": F(NUM), "x": F(list), "y": F(list)}
AXIS = {"label": F(str), "unit": F(str)}
POINT = {"x": F(NUM), "y": F(NUM), "runs": _opt(list), "note": _opt(str)}
CURVE = {"label": F(str), "model": _opt(str, nullable=True, choices=MODELS), "points": F(list, each=POINT)}
#: how a serve tax's part got its share: priced from the code and a measured rate, simulated on served traces, or
#: measured on the served engine
PART_STATUS = ("modeled", "simulated", "measured")
TAX_PART = {"part": F(str), "status": F(str, choices=PART_STATUS), "share": F(NUM), "low": F(NUM), "high": F(NUM),
            "source": F(str)}
TAX = {"config": F(str, choices=CONFIGURATIONS), "status": F(str, choices=("priced", "measured")), "tax": F(NUM),
       "low": F(NUM), "high": F(NUM), "lower_bound": F(bool), "parts": F(list, each=TAX_PART), "runs": _opt(list)}
VERIFY_TYPE = {"block": F(str), "core_seconds": F(NUM), "peak_gb": F(NUM, nullable=True), "runs": _opt(list)}
VERIFY_FAMILY = {"types": F(list), "size": F(str, nullable=True), "members": F(list)}
#: what a headline row or a Figure 5 step rests on: the served cut with every type measured on served values and every
#: lever measured; the served cut at the proof-overhead card's prices; a cut no serving path commits, or an estimate
BASES = ("served-measured", "served-card-priced", "design")
HEADLINE_ROW = {"row": F(str), "label": F(str), "pricing": F(str), "basis": F(str, choices=BASES),
                "basis_from": F(list), "rests_on": F(list), "cells": F(dict), "runs": F(list)}
SLOTS = {
    "workload": {"ranks": F(int), "device": F(str), "request": F(dict), "tokens": F(int), "weights_bytes": F(NUM),
                 "strata": F(list, each=STRATUM), "served_macs_per_token": _opt(NUM, nullable=True),
                 "params": _opt(NUM, nullable=True), "boundaries": _opt(dict)},
    "proof_overhead": {"definition": F(str), "device": F(str), "format": F(str, choices=("fp8", "s8", "bf16", "nvfp4")),
                       "peak_mac_s": F(NUM), "arm": _opt(str), "blocks": F(list, each=OVERHEAD_BLOCK),
                       "model_overhead": F(NUM, nullable=True), "label": F(str)},
    "commitment": {"hash_rate": F(NUM), "device": F(str), "scheme": F(str), "cuts": _opt(list, each=CUT)},
    "replay": {"factor": F(NUM), "batch_layout": F(str, choices=("served", "alone")),
               "deterministic": F(bool, nullable=True), "gpu_seconds_per_request": _opt(NUM, nullable=True)},
    "design": {"delta": F(NUM), "budgets": F(list), "overheads": F(list), "token_bytes": F(NUM),
               "main_lines": _opt(dict), "notes": _opt(dict)},
    "references": {"lines": F(list, each=LINE), "caption": _opt(str)},
    "infiltration": {"design": F(str, choices=("two-stage", "layer_of_token", "one-stage", "request")),
                     "form": F(str, choices=("write", "information")), "unit": F(str), "caption": F(str),
                     "consumer": F(str, nullable=True), "cap_bits_per_block": _opt(NUM),
                     "replay_blocks_per_token": _opt(NUM), "boundary_bytes_per_token": _opt(NUM),
                     "vector_bits": _opt(NUM), "series": _opt(list, each=SERIES)},
    "commit_hillclimb": {"x": F(dict, each=AXIS), "y": F(dict, each=AXIS), "workload": F(str),
                         "series": F(list, each=CURVE)},
    "serve_tax": {"baseline": F(str), "configurations": F(list, each=TAX)},
    "partition": {"name": F(str), "label": F(str), "replaces": F(list), "strata": F(list, each=STRATUM)},
    "verifier_cost": {"verifier": F(str), "device": F(str), "core_seconds_per_gpu_second": F(NUM),
                      "ratio_source": F(str), "fixed_seconds": F(NUM), "types": F(list, each=VERIFY_TYPE),
                      "families": F(list, each=VERIFY_FAMILY)},
    "headline": {"bases": F(list), "rows": F(list, each=HEADLINE_ROW), "quote": F(str)},
    "serving_rate": {"tokens_per_gpu_second": F(NUM), "plan": F(str), "saturated": F(bool), "bound": F(str, choices=BOUNDS)},
}


def _cells(cells: dict, where: str) -> list[str]:
    """A headline row's cells: each S, then each verifier step, then each scope's share of the compute it covers."""
    errors = []
    for s, by_step in cells.items():
        if not isinstance(by_step, dict):
            errors.append(f"{where}[{s!r}]: {type(by_step).__name__}, but it is an object of verifier steps")
            continue
        for m, by_scope in by_step.items():
            if not isinstance(by_scope, dict):
                errors.append(f"{where}[{s!r}][{m!r}]: {type(by_scope).__name__}, but it is an object of scopes")
                continue
            errors += [f"{where}[{s!r}][{m!r}][{scope!r}]: {v!r}, but it is a share, a number at least 0"
                       for scope, v in by_scope.items() if not isinstance(v, NUM) or isinstance(v, bool) or v < 0]
    return errors


def _fields(obj: dict, spec: dict, where: str) -> list[str]:
    errors = [f"{where}: unknown field {k!r} (the fields are {', '.join(spec)}; ask @proofs to add one)"
              for k in obj if k not in spec]
    for name, f in spec.items():
        at = f"{where}.{name}"
        if name not in obj:
            if f.required:
                errors.append(f"{at}: missing")
            continue
        v = obj[name]
        if v is None:
            if not f.nullable:
                errors.append(f"{at}: null, but it takes a value")
            continue
        if not isinstance(v, f.kind) or (isinstance(v, bool) and f.kind is not bool):
            errors.append(f"{at}: {type(v).__name__}, but it is {'a number' if f.kind is NUM else f.kind.__name__}")
            continue
        if f.choices is not None and v not in f.choices:
            errors.append(f"{at}: {v!r}, but it is one of {', '.join(map(repr, f.choices))}")
        if f.each is not None:
            for i, entry in enumerate(v if isinstance(v, list) else [v]):
                sub = f"{at}[{i}]" if isinstance(v, list) else at
                errors += (_fields(entry, f.each, sub) if isinstance(entry, dict)
                           else [f"{sub}: {type(entry).__name__}, but it is an object"])
    return errors


def check(card: dict) -> dict:
    """The card, or ValueError naming every field that breaks the schema."""
    errors = _fields(card, ENVELOPE, "card")
    slot, value = card.get("slot"), card.get("value")
    if slot not in SLOTS:
        errors.append(f"card.slot: {slot!r}, but it is one of {', '.join(SLOTS)}")
    elif isinstance(value, dict):
        errors += _fields(value, SLOTS[slot], f"{slot}.value")
    if card.get("model") is None and slot not in MODEL_FREE:
        errors.append(f"card.model: null, but a {slot} card names its model ({', '.join(MODELS)})")
    errors += [f"card.runs: {r!r} is not a run id" for r in card.get("runs") or [] if not RUN.fullmatch(str(r))]
    if not str(card.get("source", "")).strip():
        errors.append("card.source: empty; say what produced the card and from what")
    if not str(card.get("owner", "@")).startswith("@"):
        errors.append(f"card.owner: {card['owner']!r}, but it is the maintaining lane's handle, e.g. '@proofs'")
    if slot == "references" and isinstance(value, dict):
        errors += [f"references.value.lines[{i}]: give exactly one of value and by_scope"
                   for i, line in enumerate(value.get("lines") or [])
                   if isinstance(line, dict) and ("value" in line) == ("by_scope" in line)]
    if slot == "serve_tax" and not errors:
        names = [t["config"] for t in value["configurations"]]
        errors += [f"serve_tax.value.configurations: {n!r} given {names.count(n)} times" for n in sorted(set(names))
                   if names.count(n) > 1]
        errors += [f"serve_tax.value.configurations[{i}]: low <= tax <= high, but {t['low']}, {t['tax']}, {t['high']}"
                   for i, t in enumerate(value["configurations"]) if not 0 <= t["low"] <= t["tax"] <= t["high"]]
        for i, t in enumerate(value["configurations"]):
            parts = [p["part"] for p in t["parts"]]
            errors += [f"serve_tax.value.configurations[{i}].parts: {n!r} given {parts.count(n)} times"
                       for n in sorted(set(parts)) if parts.count(n) > 1]
            errors += [f"serve_tax.value.configurations[{i}].{k}: {t[k]}, but its parts' {key}s add to {total}: a "
                       "priced configuration is the sum of its named parts (regenerate it with total_cost.py "
                       "priced-card)"
                       for k, key in (("tax", "share"), ("low", "low"), ("high", "high")) if t["status"] == "priced"
                       for total in [sum(p[key] for p in t["parts"])] if abs(t[k] - total) > 1e-12 * max(1.0, total)]
    if slot == "verifier_cost" and not errors:
        measured = {t["block"] for t in value["types"]}
        errors += [f"verifier_cost.value.families[{i}]: member {m!r} is none of the measured types' blocks"
                   for i, fam in enumerate(value["families"]) for m in fam["members"] if m not in measured]
        errors += [f"verifier_cost.value.families[{i}]: {len(fam['members'])} members, but a family scales from one "
                   "(through the fixed seconds at size 0) or two" for i, fam in enumerate(value["families"])
                   if not 1 <= len(fam["members"]) <= 2]
    if slot == "partition" and not errors:
        errors += [f"partition.value.strata[{i}]: no k_log, so no staged witness bits to price it by until its type is "
                   "measured; give the k_log circuit-check stages it at"
                   for i, x in enumerate(value["strata"]) if "k_log" not in x]
    if slot == "headline" and not errors:
        rows = value["rows"]
        if card["status"] == "measured":
            errors += [f"headline.value.rows[{i}]: basis {r['basis']!r}, but a measured headline card holds only "
                       "served-measured rows; mark the card modeled" for i, r in enumerate(rows)
                       if r["basis"] != "served-measured"]
        errors += [f"headline.value.rows[{i}]: basis {r['basis']!r} with an empty basis_from; name what sets it"
                   for i, r in enumerate(rows) if r["basis"] != "served-measured" and not r["basis_from"]]
        errors += [e for i, r in enumerate(rows) for e in _cells(r["cells"], f"headline.value.rows[{i}].cells")]
    if slot == "serving_rate" and not errors:
        if card["status"] != "measured":
            errors.append(f"card.status: {card['status']!r}, but a serving rate is measured on a served run")
        if value["tokens_per_gpu_second"] <= 0:
            errors.append(f"serving_rate.value.tokens_per_gpu_second: {value['tokens_per_gpu_second']!r}, but it is a rate "
                          "above 0")
        if not (RUN.fullmatch(value["plan"]) or ART.fullmatch(value["plan"])):
            errors.append(f"serving_rate.value.plan: {value['plan']!r}, but it is an art id (the plan file's, or the "
                          "record's that holds it) or the id of the run that holds it")
        if not value["saturated"] and not card["provisional"]:
            errors.append("serving_rate.value.saturated: false on a card that is not provisional; a lower bound is "
                          "provisional, so that a floor is never read as the rate")
    if slot == "workload" and not errors and value.get("served_macs_per_token") is not None:
        in_blocks = sum(s["per_token"] * s["macs"] for s in value["strata"])
        if value["served_macs_per_token"] < in_blocks * (1 - 1e-9):
            errors.append(f"workload.value.served_macs_per_token: {value['served_macs_per_token']:.4g}, below the "
                          f"strata's {in_blocks:.4g} MACs a token; it is the whole circuit's")
    if errors:
        raise ValueError("not a spot-card/v1 card:\n  " + "\n  ".join(errors))
    return card


def path(slot: str, model: str | None, art: str | None = None) -> Path:
    tail = f".{art.removeprefix('art:')[:12]}" if slot in MANY and art else ""
    return CARDS / f"{slot}.{(model or 'all').lower()}{tail}.json"


def load(slot: str, model: str | None, required: bool = True) -> dict | None:
    """The model's card for the slot, else the slot's card for every model."""
    for p in (path(slot, model), path(slot, None)):
        if p.exists():
            return check(json.loads(p.read_text()))
    if required:
        raise FileNotFoundError(f"no {slot} card for {model}: its owner publishes one (`research data put --kind "
                                f"{SCHEMA} --file CARD.json --preserve`) and `cards.py fetch ART` keeps it in {CARDS}")
    return None


def load_all(slot: str, model: str | None) -> list[dict]:
    """Every card of a slot in `MANY` for the model and for every model, the model's first."""
    found = []
    for m in (model, None):
        p = path(slot, m)
        found += [check(json.loads(q.read_text())) for q in [p, *sorted(CARDS.glob(f"{p.stem}.*.json"))] if q.exists()]
    return found


def write(card: dict, force: bool = False, replaces: tuple[str, ...] = ()) -> Path:
    """Keep `card` under `cards/`, removing the kept cards whose art ids it `replaces`."""
    check(card)
    gone = [q for q in CARDS.glob("*.json") if replaces and json.loads(q.read_text()).get("art") in replaces]
    if len(gone) != len(set(replaces)):
        raise FileNotFoundError(f"replaces {sorted(replaces)}, but {len(gone)} of them are kept in {CARDS}")
    for q in gone:
        q.unlink()
    p = path(card["slot"], card["model"], card.get("art"))
    if p.exists() and not force:
        old = json.loads(p.read_text()).get("art")
        if old == card.get("art"):
            return p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(card, indent=1) + "\n")
    return p


def fetch(art: str, force: bool = False, replaces: tuple[str, ...] = ()) -> Path:
    """The published card `art`, checked and kept under `cards/` with its art id, in place of `replaces`."""
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["research", "data", "fetch", art, "--to", tmp], check=True, stdout=subprocess.DEVNULL)
        found = [p for p in Path(tmp).rglob("*.json")]
        if len(found) != 1:
            raise ValueError(f"{art}: a card is one JSON file, found {[p.name for p in found]}")
        card = json.loads(found[0].read_text())
    return write(card | {"art": art}, force, replaces)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("check").add_argument("cards", type=Path, nargs="+")
    f = sub.add_parser("fetch")
    f.add_argument("arts", nargs="+")
    f.add_argument("--force", action="store_true", help="rewrite a card already kept under the same art id")
    f.add_argument("--replaces", action="append", default=[], metavar="ART",
                   help="a kept card's art id the fetched card replaces, whose file is removed (repeatable)")
    sub.add_parser("list")
    args = ap.parse_args(argv)
    if args.cmd == "check":
        bad = 0
        for p in args.cards:
            try:
                c = check(json.loads(p.read_text()))
                print(f"{p}: ok ({c['slot']}, {c['model'] or 'every model'}, {c['status']})")
            except ValueError as e:
                bad += 1
                print(f"{p}: {e}")
        sys.exit(1 if bad else 0)
    if args.cmd == "fetch":
        if args.replaces and len(args.arts) != 1:
            ap.error("--replaces names what one fetched card replaces: fetch that card alone")
        for art in args.arts:
            print(fetch(art, args.force, tuple(args.replaces)))
    if args.cmd == "list":
        for p in sorted(CARDS.glob("*.json")):
            c = json.loads(p.read_text())
            print(f"{p.name}: {c['status']}{' provisional' if c['provisional'] else ''}, {c.get('art') or 'in tree'}, "
                  f"{c['owner']}")


if __name__ == "__main__":
    main()
