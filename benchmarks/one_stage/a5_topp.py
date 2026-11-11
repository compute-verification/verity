"""One-stage audit, run A5, on a top-p stand-in: a window of ``GumbelTopPTokenSelectSharedGreedy_v1{V}`` rows cut into their
stages, the fixture's (`adapters/vllm/tests/pipeline/test_serving_rows_stages.py`: its ``Served`` rows, wired by its ``members``
and read by its ``read``, each carried row wired to the output row of the unit that wrote it), under a partition naming each
stage on bits (the fixture's ``boolean``, the ``_v2`` Definitions of `boolean_topp_stages`), its run ``<ROOT>/rank0``, committed
as serving commits it and audited as `a5_sparse_mla.py` audits its own window (`a5_sparse_mla.audited`: `a5.audit` as rank 0 of a
one-rank step on a local challenger's coins under ``stratified-rate:1/64``, a 64th of each template's units and at least one).
``--stages slots`` names the stages as words instead (``_v1``), whose table reads go in lookup slots (`chunk_rows.reads`), the
form Lean's ``verify --zk`` refuses (`one_stage.proofs.SCOPE_WHY`).

The window holds the stages a5 can stage. A proof block is composed only when one of its rows is past one SHA-512 block
(`interface.check_proof_block`), and a stats block's longest row is its logits, two bytes a lane, so the stats of split count S
are composed only from 56 lanes a slice (V of at least 56 S), and every other stage is wired back to them. ``--splits`` names
the split counts whose stats, probs and step blocks the window holds, and the select chunks, which read every split count's
last step, are held only when it names them all (V of at least 1,792). The default is V = 64, where the fixture stages its
stats blocks, and split count 1 alone: one unit a row of each of 8 templates of four families (``TopPStatsHead``,
``TopPStatsExp``, ``TopPProbs``, ``TopPStep``), and no ``GumbelTopPSelect`` chunk. A window with a stage that is not composed
is refused before anything is committed.

It stands in for the served row's stages (V = 154,880: 8,159 units of 85 templates), whose units it does not prove. On bits
the run exits 0 only when the outcome is accepted and every negative refused; with ``--stages slots`` only when the outcome is
not accepted and Lean refused each sampled proof block's statement by that reason. ``--out`` holds the window serving wrote
(``window/``, its private files too) and a5's records (``rank0/``, ``challenger/``, ``summary.json``, which names each sampled
proof block's template, family and verdicts, and each family's units held and sampled). The arguments but ``--stages``,
``--v``, ``--splits`` and ``--rows`` are `a5_sparse_mla.parser`'s, and the PYTHONPATH is that script's.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "adapters" / "vllm"))

import a5  # noqa: E402
import a5_sparse_mla as SM  # noqa: E402
from tests.pipeline import test_serving_rows_stages as TS  # noqa: E402
from verity.circuits.codec import descriptor_id  # noqa: E402
from verity.circuits.defs import bind  # noqa: E402
from verity.commitments.windows import interface as IF  # noqa: E402
from verity.commitments.windows import rows as CS  # noqa: E402
from verity.definitions.registry import sampling as SA  # noqa: E402
from verity.definitions.registry import topp_keep_builder as K  # noqa: E402
from verity.definitions.registry import topp_stages as T  # noqa: E402
from verity.protocols.spot.verifier.one_stage import proofs as PF  # noqa: E402
from verity_vllm.pipeline import serving_rows as SRW  # noqa: E402

#: the stand-in's session root, as a TP Commit's run root is its ranks' session root
ROOT = hashlib.sha512(b"verity/a5/topp-stages-stand-in/v1").hexdigest()
FIXTURE = "adapters/vllm/tests/pipeline/test_serving_rows_stages.py"
#: the most units a member holds, the fixture's
MEMBER_UNITS = 7


def split_counts(s: str) -> tuple[int, ...]:
    """``--splits``: split counts of `topp_keep_builder.SPLIT_COUNTS`, comma-separated, ascending."""
    got = tuple(sorted({int(x) for x in s.split(",")}))
    if not got or not set(got) <= set(K.SPLIT_COUNTS):
        raise ValueError(f"{s}: split counts of {K.SPLIT_COUNTS}, comma-separated")
    return got


def templates(v: int, splits: tuple[int, ...]) -> tuple[str, list[str]]:
    """The row at V = ``v`` and the stage templates of it the window holds, as words in the row's order: the stats, probs and
    step blocks of the split counts ``splits``, and the select chunks when those are every split count."""
    spec = descriptor_id(bind(SA.GumbelTopPTokenSelectSharedGreedy, V=v))
    every = set(splits) == set(K.SPLIT_COUNTS)
    held = {descriptor_id(u.defn) for u in T.plan(v) if (every if u.key[0] == "select" else u.key[1] in splits)}
    return spec, [t for t, _n in SRW.unit_templates(spec)[1:] if t in held]


def commit(out: Path, v: int, splits: tuple[int, ...], rows: int, stages: str) -> tuple[CS.Binding, dict]:
    """The window of ``rows`` rows at V = ``v`` over the stages of ``splits`` (`templates`), on bits or, with ``stages``
    ``slots``, as words, under ``<ROOT>/rank0``, committed into ``out`` (`a5_sparse_mla.committed`), and what it is a stand-in
    of. Refused (SystemExit) when Flock's composer would refuse one of its templates' proof blocks."""
    spec, words = templates(v, splits)
    tmpls = words if stages == "slots" else [TS.boolean(t) for t in words]
    for t in tmpls:
        try:
            IF.check_proof_block(SRW.template(t).fn)
        except ValueError as e:
            raise SystemExit(f"a5_topp: V = {v} with split counts {list(splits)} holds a stage whose proof block Flock's "
                             f"composer refuses, so a5 stages no window of it (a stats block is composed from 56 lanes a "
                             f"slice: --v of at least {56 * max(splits)}): {e}") from e
    s = TS.Served(spec, n=rows)
    ms = TS.members(s, tmpls, dict.fromkeys(tmpls, MEMBER_UNITS))
    defs = [(SRW.template(t).fn, sum(m.n for m in ms if m.template == t)) for t in tmpls]
    w = a5.rated(CS.Binding(*CS.population_program(defs, several=True), a5.D.parse_design("stratified:1"), f"{ROOT}/rank0",
                            tuple(ms)), SM.RULE)
    of = {"fixture": FIXTURE, "row": spec, "rows": rows, "splits": list(splits), "stages": stages}
    return SM.committed(w, TS.read(s, w.members, tmpls), out, of, "a5_topp.py"), of


def families(held: dict) -> dict:
    """A summary's ``window.templates`` (`a5_sparse_mla.audited`) by stage family: its templates, and their units held and
    sampled."""
    out: dict = {}
    for t, n in held.items():
        f = out.setdefault(SRW.template(t).fn.name, {"templates": 0, "units": 0, "sampled": 0})
        f.update(templates=f["templates"] + 1, units=f["units"] + n["units"], sampled=f["sampled"] + n["sampled"])
    return out


def sessions(mine: CS.Binding, rank: dict) -> list[dict]:
    """Each member with a sampled proof block, as the rank's records hold its session (`a5.audit`'s summary): its template and
    family, Lean's ``verify --zk`` verdict of it with Lean's reason when it is refused, M0's verifier's verdict, Lean's and
    the session's seconds and the circuit's ``k_log``."""
    audit = rank.get("audit") or {}
    verdicts, why = audit.get("verdicts") or {}, audit.get("why") or {}
    timing, circuits = rank.get("timing") or {}, rank.get("circuits") or {}
    out = []
    for k, m in enumerate(mine.members):
        if f"lean-verify/m{k}" not in verdicts:
            continue
        t = timing.get(m.key) or {}
        out.append({"member": k, "template": m.template, "family": SRW.template(m.template).fn.name,
                    "lean": verdicts[f"lean-verify/m{k}"], "why": why.get(f"lean-verify/m{k}"),
                    "m0": verdicts.get(f"m0-verifier/m{k}"), "lean_verify_s": t.get("lean_verify_s"),
                    "session_s": t.get("session_s"), "k_log": (circuits.get(m.template) or {}).get("k_log")})
    return out


def expected(stages: str, accepted: bool, proved: list[dict]) -> bool:
    """On bits: accepted, with every negative refused (`a5.audit`). As words: not accepted, Lean's ``verify --zk`` having
    refused the statement of each member with a sampled proof block as outside the form it takes (`proofs.SCOPE_WHY`)."""
    if stages == "bits":
        return accepted
    return not accepted and bool(proved) and all(p["lean"] is False and PF.SCOPE_WHY in (p["why"] or "") for p in proved)


def main() -> int:
    ap = SM.parser()
    ap.add_argument("--stages", choices=("bits", "slots"), default="bits", help="the partition names each stage on bits "
                    "(_v2, laid out flat) or as words (slots: _v1, its table reads in lookup slots, to be refused by Lean)")
    ap.add_argument("--v", type=int, default=64, help="the rows' V")
    ap.add_argument("--splits", type=split_counts, default=(1,), help="the split counts whose stages the window holds, "
                    "comma-separated; the select chunks only with all of them")
    ap.add_argument("--rows", type=int, default=2, help="the window's rows: each template's units are its units a row "
                    "times this, so that a sample of one of each leaves units unsampled for the negatives")
    a = ap.parse_args()
    out = Path(a.out)
    mine, of = commit(out / "window", a.v, a.splits, a.rows, a.stages)
    summary, rank = SM.audited(a, mine, of, ROOT)
    proved = sessions(mine, rank)
    summary.update(stages=a.stages, families=families(summary["window"]["templates"]), sessions=proved,
                   as_expected=expected(a.stages, summary["accepted"], proved))
    (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True, default=str))
    print("A5-TOPP " + json.dumps({x: summary[x] for x in ("accepted", "as_expected", "stages", "families", "sessions")}
                                  | {x: rank.get(x) for x in SM.SHOWN}, sort_keys=True, default=str))
    return 0 if summary["as_expected"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
