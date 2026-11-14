"""cover_count.py (COVER | art:ID) [--card CARD] [--form boolean|listed] [--check]: how many of a request's block types
have no statement the Lean verifier accepted, recounted from a request cover record (`cover.json`, schema
`circuits/request-cover/v7`, or v8: v7 with `card`, the card it is of, by its `source`, `path` and `tree`).

`--card CARD` holds the record to a sampling plan's card (`benchmarks/sampling_plan/glm53_blocks.json`): the record's
on-card types must be exactly the card's `definitions`, else it is refused (exit 2), naming up to five types on each
side.

COVER is a path to the record; an `art:` id is fetched with `research data fetch ART --to DIR` into a temporary
directory. The count comes from the record's entries alone: `blocks` (a block type of the request each, counted when
`on_card`, with its `bound_outcome` and, where it was run, its tag in `bound_runs.types`) and `beside_extras`
(`v2t<NNN>`, the Boolean version of listed type `t<NNN>`, which `earlier_form` names). A run counts as accepted only
when `B.verify.held` has exit 0 and accepted true and both `B.verify.no-program` and `B.verify.other-def` have exit 1
and accepted false; a run whose recorded outcome says otherwise than its verdicts is an error. A listed type with no
bound run of its own (`accepted-earlier`) counts as accepted only with an accepted statement in its `statements`.

The recount is then held to the record's stored totals (`bound_totals`, `bound_shares`, `fifth_pass.by_family`), and
any disagreement, an unknown schema or an extra naming no listed type on the card is an error: the message goes to
stderr and the exit status is 2.

`--form listed` counts a type as accepted only in its listed form; the default, `--form boolean`, also counts a type
whose listed form has none when its Boolean version was accepted. It prints the types and blocks on the card, those
accepted in the listed form, those accepted on the Boolean version only, those with none by family (each with the
record's reason and its blocks), and last a line that starts with the count. `--check` exits 1 when the count is not 0.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

KNOWN_SCHEMAS = ("circuits/request-cover/v7", "circuits/request-cover/v8")
ROOT = Path(__file__).resolve().parents[2]
#: a block type's family, as `fifth_pass.by_family` names it: its Definition's name without the version
FAMILY = re.compile(r"(\w+?)_v\d+\{")
#: what `bound_shares` rounds a share to
SHARE_DIGITS = 6
#: how many types `--card` names on each side of a mismatch
NAMED = 5


class RecordError(Exception):
    pass


def fetch(art: str, into: Path) -> Path:
    """The record an `art:` id names, fetched into ``into``."""
    research = shutil.which("research")
    cmd = [research] if research else ["uv", "run", "-q", "--project", str(ROOT / "tools" / "research"), "research"]
    done = subprocess.run([*cmd, "data", "fetch", art, "--to", str(into)], capture_output=True, text=True)
    if done.returncode != 0:
        raise RecordError(f"research data fetch {art} exited {done.returncode}: {done.stderr.strip()[-2000:]}")
    found = sorted(into.rglob("cover.json"))
    if len(found) != 1:
        raise RecordError(f"{art} holds {len(found)} cover.json files, not one")
    return found[0]


def label(r: dict) -> str:
    return r["outcome"] + (f"/{r['why']}" if r.get("why") else "")


def accepted(r: dict, where: str) -> bool:
    """Whether a run's verdicts are an acceptance with both controls refused, held to its recorded outcome."""
    v = r.get("verdicts") or {}
    held, no_program, other_def = (v.get(f"B.verify.{k}") for k in ("held", "no-program", "other-def"))
    took = bool(held and held.get("exit") == 0 and held.get("accepted") is True)
    refused = all(x and x.get("exit") == 1 and x.get("accepted") is False for x in (no_program, other_def))
    if (r.get("outcome") == "accepted") != (took and refused):
        why = ("its verdicts are an acceptance with both controls refused" if took and refused else
               "B.verify.held did not accept" if not took else
               "a control (B.verify.no-program, B.verify.other-def) did not refuse")
        raise RecordError(f"{where}: the record says {r.get('outcome')!r} but {why}")
    if (r.get("outcome") == "held-control-off") != (took and not refused):
        raise RecordError(f"{where}: the record says {r.get('outcome')!r}, which its verdicts contradict")
    return took and refused


def held_to_card(c: dict, card_path: Path) -> None:
    """The record's on-card types against the card's `definitions`, named up to `NAMED` on each side."""
    card = set(json.loads(card_path.read_text())["definitions"])
    record = {b["block"] for b in c["blocks"] if b["on_card"]}
    if record == card:
        return

    def names(side: set) -> str:
        shown = sorted(side)[:NAMED]
        return ", ".join(shown) + (f" and {len(side) - len(shown)} more" if len(side) > len(shown) else "")

    sides = [f"{len(s)} {what}: {names(s)}" for s, what in ((card - record, "on the card and not in the record"),
                                                           (record - card, "in the record and not on the card")) if s]
    raise RecordError(f"the record's {len(record)} on-card types are not {card_path}'s {len(card)} definitions; "
                      + "; ".join(sides))


def family(block: str) -> str:
    m = FAMILY.match(block)
    return m[1] if m else block


def count(c: dict, form: str) -> dict:
    """The recount: every on-card type with whether it is accepted in each form, held to the record's stored totals."""
    if c.get("schema") not in KNOWN_SCHEMAS:
        raise RecordError(f"schema {c.get('schema')!r} is not one this command knows ({', '.join(KNOWN_SCHEMAS)})")
    if c["schema"] != KNOWN_SCHEMAS[0] and not (c.get("card") or {}).get("source"):
        raise RecordError(f"a {c['schema']} record names its card's source under card.source")
    on = [b for b in c["blocks"] if b["on_card"]]
    runs = c["bound_runs"]["types"]
    total = sum(b["blocks"] for b in on)

    types = []
    for b in on:
        tag = (b.get("cir272") or {}).get("tag")
        if tag is not None:
            if tag not in runs:
                raise RecordError(f"{b['block']}: its tag {tag} is not in bound_runs.types")
            listed = accepted(runs[tag], f"bound run {tag}")
            if listed != b["bound_outcome"].startswith("accepted"):
                raise RecordError(f"{tag}: bound_outcome {b['bound_outcome']!r} "
                                  f"but its bound run says {label(runs[tag])!r}")
        else:
            listed = b["bound_outcome"].startswith("accepted")
            if listed and not any(s.get("verdict") == "accepted" for s in b.get("statements") or ()):
                raise RecordError(f"{b['block']}: bound_outcome {b['bound_outcome']!r} with no accepted statement")
        types.append({"tag": tag, "block": b["block"], "blocks": b["blocks"], "family": family(b["block"]),
                      "outcome": b["bound_outcome"], "listed": listed, "extra": None})

    by_tag = {t["tag"]: t for t in types if t["tag"] is not None}
    for x, r in (c.get("beside_extras") or {}).items():
        ef = r["earlier_form"]
        t = by_tag.get(ef["tag"])
        if t is None or t["block"] != ef["block"] or t["blocks"] != ef["blocks"]:
            raise RecordError(f"beside extra {x}: its earlier form {ef['tag']} is not that block type on the card")
        if t["extra"] is not None:
            raise RecordError(f"beside extras {t['extra']['tag']} and {x} are both the Boolean version of {ef['tag']}")
        t["extra"] = {"tag": x, "outcome": label(r), "accepted": accepted(r, f"beside extra {x}")}

    held_to_stored(c, types, total)

    for t in types:
        boolean = bool(t["extra"] and t["extra"]["accepted"])
        t["accepted"] = t["listed"] or (form == "boolean" and boolean)
        t["boolean_only"] = not t["listed"] and boolean
        if t["listed"]:
            t["reason"] = None
        elif form == "listed" or t["extra"] is None:
            t["reason"] = t["outcome"] + ("" if form == "listed" else "; no Boolean version in the record")
        else:
            t["reason"] = f"{t['extra']['outcome']} ({t['extra']['tag']}; listed form {t['outcome']})"
    return {"total": total, "types": types}


def held_to_stored(c: dict, types: list, total: int) -> None:
    by_outcome: dict = {}
    for t in types:
        n = by_outcome.setdefault(t["outcome"], [0, 0])
        n[0] += 1
        n[1] += t["blocks"]
    if by_outcome != c["bound_totals"]:
        raise RecordError(f"bound_totals {c['bound_totals']} but the blocks recount {by_outcome}")

    shares = {g: [0, 0] for g in c["bound_shares"]}
    for outcome, (n, k) in by_outcome.items():
        g = next((g for g in shares if outcome == g or outcome.startswith(g + "-")), None)
        if g is None:
            raise RecordError(f"bound_outcome {outcome!r} falls in no group of bound_shares ({', '.join(shares)})")
        shares[g][0] += n
        shares[g][1] += k
    for g, (n, k) in shares.items():
        s = c["bound_shares"][g]
        if [s["types"], s["blocks"]] != [n, k] or round(k / total, SHARE_DIGITS) != round(s["share"], SHARE_DIGITS):
            raise RecordError(f"bound_shares[{g!r}] {s} but the blocks recount {n} types, {k} blocks, "
                              f"share {round(k / total, SHARE_DIGITS)}")

    fifth = (c.get("fifth_pass") or {}).get("by_family")
    if fifth is None:
        raise RecordError("the record has no fifth_pass.by_family to hold the beside extras to")
    fam: dict = {}
    for t in types:
        if t["extra"] is not None:
            f = fam.setdefault(t["family"], {"types": 0, "blocks": 0, "outcomes": Counter()})
            f["types"] += 1
            f["blocks"] += t["blocks"]
            f["outcomes"][t["extra"]["outcome"]] += 1
    got = {k: {"types": v["types"], "blocks": v["blocks"], "outcomes": dict(v["outcomes"])} for k, v in fam.items()}
    stored = {k: {"types": v["types"], "blocks": v["blocks"], "outcomes": v["outcomes"]} for k, v in fifth.items()}
    if got != stored:
        raise RecordError(f"fifth_pass.by_family {stored} but the beside extras recount {got}")


def report(c: dict, recount: dict, form: str) -> tuple[list[str], int]:
    types, total = recount["types"], recount["total"]

    def line(what: str, ts: list) -> str:
        return f"{what}: {len(ts)} types, {sum(t['blocks'] for t in ts):,} blocks"

    listed = [t for t in types if t["listed"]]
    boolean = [t for t in types if t["boolean_only"]]
    none = [t for t in types if not t["accepted"]]
    card = c.get("card")
    out = [f"record: {c['schema']}, tree {c.get('tree')}" + (f", of the card {card['source']}" if card else ""),
           f"on the card: {len(types)} types, {total:,} blocks",
           line("accepted in the listed form", listed),
           line("accepted on the Boolean version only", boolean)
           + (" (not counted: --form listed)" if form == "listed" else ""),
           line(f"with none ({'listed form only' if form == 'listed' else 'the Boolean version counted'})", none)]
    blocks_of = Counter()
    for t in none:
        blocks_of[t["family"]] += t["blocks"]
    for f in sorted(blocks_of, key=lambda f: (-blocks_of[f], f)):
        ts = sorted((t for t in none if t["family"] == f), key=lambda t: t["tag"] or "")
        out.append("  " + line(f, ts))
        out += [f"    {t['tag'] or '-'}  {t['blocks']:,} blocks  {t['reason']}  {t['block']}" for t in ts]
    out.append(f"{len(none)} of {len(types)} block types have no statement the Lean verifier accepted (--form {form})")
    return out, len(none)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cover", help="a cover.json, or an art: id to fetch")
    ap.add_argument("--card", type=Path, help="a sampling plan's card the record's on-card types must be exactly")
    ap.add_argument("--form", choices=("boolean", "listed"), default="boolean",
                    help="listed: count only a type's listed form; boolean (default): its Boolean version too")
    ap.add_argument("--check", action="store_true", help="exit 1 when the count is not 0")
    a = ap.parse_args(argv)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            path = fetch(a.cover, Path(tmp)) if a.cover.startswith("art:") else Path(a.cover)
            c = json.loads(path.read_text())
        if a.card:
            held_to_card(c, a.card)
        out, n = report(c, count(c, a.form), a.form)
    except RecordError as e:
        print(f"cover_count: {e}", file=sys.stderr)
        return 2
    print("\n".join(out))
    return 1 if a.check and n else 0


if __name__ == "__main__":
    sys.exit(main())
