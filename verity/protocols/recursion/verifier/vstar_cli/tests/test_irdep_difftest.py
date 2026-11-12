"""`RecDeploy.VStarBuild.IrDep`, the deployed reference inner prover `HonestFixture` names, against the deployed builder
at a deployed run's inner session, so that the fixture's witness run, checked at the builder's own inner prover, is a
witness at `IrDep`.

``VERITY_IRDEP_TREE`` is the run's tree (`85-rec-reprice.sh`'s ``$DD``: ``inner/s0``, ``v/fold.json``, ``v/points.json``,
``v/acc-salts.bin``) and ``VERITY_IRDEP_LEAN`` the output of `IrDepSession.lean` at it, whose ``IRDEP`` line is `IrDep`'s
rows at every inner round and its rest after the last. The builder's side, from the session's own files:

* **rows**: round 0's, ``root_B``'s nodes (`rec_live.top_rows`); then each rep's rounds', rep by rep, each round's new
  tops' nodes then its message row (`rec_live.Proxy`'s private rounds), as u16 words. A round's tops are its node rows
  (their salts are the tape's), so the rows hold the tops too.
* **openings** (`rec_vstar.queries`): rep, level, query, position, row and siblings.
* **κ, eq(α), v** (`rec_vstage.verifier` at the run's fold and points): each rep's `verifier_rows`.
* **running sums' salts** (`rec_vstage.acc_salts`, as many as the openings and one).

Not vacuous: rows at rep 0's proof with one byte of its first cap node flipped (Lean's ``planted_rows``) differ in that one
word, and a one-byte change to any compared value of Lean's is found. Skips unless both variables name files."""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest
from verity_flock import rec_live as RL
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V

sys.path.insert(0, str(Path(__file__).resolve().parent))
import settings

TREE = settings.irdep_tree()
LEAN_OUT = settings.irdep_lean()
pytestmark = pytest.mark.skipif(TREE is None or LEAN_OUT is None,
                                reason="VERITY_IRDEP_TREE and VERITY_IRDEP_LEAN name no deployed run and IrDep output")
TABLE = "circuit"


def h128(x: int) -> str:
    return f"{x:032x}"


def node_rows(nodes: list[str]) -> list[list[int]]:
    return RL.top_rows(bytes.fromhex("".join(nodes)))


def expected(tree: Path) -> dict:
    """The deployed builder's rows, openings, κ, eq(α), v and running sums' salts at the run, in `IrDepSession`'s form."""
    public, private, proofs = V.load(tree / "inner" / "s0")
    prv = {(r["stream"], r["k"]): r for r in private["rounds"]}
    rows = [node_rows(private["roots"][TABLE]["nodes"])]
    for rep in range(2):
        stream, k = f"{TABLE}/rep{rep}", 0
        while (stream, k) in prv:
            r = prv[(stream, k)]
            rows.append([w for t in r["tops"] for w in node_rows(t["nodes"])] + r["rows"])
            k += 1
    qs = V.queries(public, private, proofs)
    vf = S.verifier(public, json.loads((tree / "v" / "fold.json").read_text()),
                    json.loads((tree / "v" / "points.json").read_text()))
    nlev = len(vf.reps[0].info["kappa"])
    return {"rows": rows,
            "openings": [{"rep": q.rep, "level": q.level, "j": q.j, "pos": q.pos, "row": [h128(x) for x in q.row],
                          "sibs": [x.hex() for x in q.sibs]} for q in qs],
            "acc_salts": [x.hex() for x in S.acc_salts(tree / "v" / "acc-salts.bin", len(qs) + 1)],
            "kappa": [[[[h128(x) for x in row] for row in k] for k in rv.info["kappa"]] for rv in vf.reps],
            "eq_a": [[[h128(x) for x in rv.info["queries"][lv][1]] for lv in range(nlev)] for rv in vf.reps],
            "vs": [[h128(x) for x in rv.v] for rv in vf.reps]}


def lean_out(path: Path) -> dict:
    """The ``IRDEP`` line's JSON, after whatever position Lean prints an `#eval`'s output under."""
    lines = [ln for ln in path.read_text().splitlines() if "IRDEP {" in ln]
    assert len(lines) == 1, f"{path}: {len(lines)} IRDEP lines"
    return json.loads(lines[0][lines[0].index("IRDEP {") + len("IRDEP "):])


def first_diff(a, b, at: str = "") -> str | None:
    """Where ``a`` and ``b`` first differ (a JSON path), or None."""
    if isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            if (d := first_diff(x, y, f"{at}[{i}]")) is not None:
                return d
        return None if len(a) == len(b) else f"{at}: {len(a)} entries, {len(b)}"
    if isinstance(a, dict) and isinstance(b, dict):
        if sorted(a) != sorted(b):
            return f"{at}: keys {sorted(a)}, {sorted(b)}"
        for k in a:
            if (d := first_diff(a[k], b[k], f"{at}.{k}")) is not None:
                return d
        return None
    return None if a == b else f"{at}: {a!r} != {b!r}"


def compare(want: dict, got: dict) -> dict[str, str]:
    """Each compared field where Lean's ``got`` is not the builder's ``want``, with where it first differs. Lean's rows run
    one round past the last, an empty one (`IrDep.rows` past the schedule)."""
    out = {}
    rows = got["rows"]
    if rows[len(want["rows"]):] != [[]]:
        out["rows"] = f"{len(rows)} rounds, past the builder's {len(want['rows'])} not one empty one"
    elif (d := first_diff(want["rows"], rows[:len(want["rows"])], "rows")) is not None:
        out["rows"] = d
    for k in ("openings", "acc_salts", "kappa", "eq_a", "vs"):
        if (d := first_diff(want[k], got[k], k)) is not None:
            out[k] = d
    return out


def flip_hex(s: str) -> str:
    """``s`` with its first byte's low bit flipped."""
    return f"{int(s[:2], 16) ^ 1:02x}{s[2:]}"


@pytest.fixture(scope="module")
def sides() -> tuple[dict, dict]:
    return expected(TREE), lean_out(LEAN_OUT)


def test_irdep_is_the_deployed_builder(sides):
    """`IrDep`'s rows at every inner round, and its openings, running sums' salts, κ, eq(α) and v after the last, are the
    deployed builder's at the run."""
    want, got = sides
    print(json.dumps({"irdep": {"rounds": len(want["rows"]), "rows": sum(map(len, want["rows"])),
                                "openings": len(want["openings"]), "acc_salts": len(want["acc_salts"]),
                                "m": got["m"], "k_log": got["k_log"], "levels": got["levels"],
                                "rep_lens": got["rep_lens"], "T": got["T"]}}), flush=True)
    assert want["openings"] and want["kappa"] and all(want["vs"]), "an empty side"
    assert compare(want, got) == {}


def test_a_planted_byte_is_found(sides):
    """Rep 0's proof with one byte of its first cap node flipped gives rows that differ from the builder's in round 0's
    first row alone, in one word; and a one-byte change to any compared value of Lean's fails its field."""
    want, got = sides
    planted = dict(got, rows=got["planted_rows"])
    assert set(compare(want, planted)) == {"rows"}
    diffs = [(i, j, w) for i, (a, b) in enumerate(zip(want["rows"], planted["rows"])) for j, (x, y) in enumerate(zip(a, b))
             for w, (u, v) in enumerate(zip(x, y)) if u != v]
    assert diffs == [(0, 0, 0)], diffs
    later = next(i for i, r in enumerate(got["rows"]) if i > len(want["rows"]) // 2 and r)

    def flip(a: list, i: int) -> None:
        a[i] = a[i] ^ 1 if isinstance(a[i], int) else flip_hex(a[i])

    edits = {"rows": lambda o: flip(o["rows"][later][-1], 0),
             "openings": lambda o: flip(o["openings"][-1]["sibs"], 0),
             "acc_salts": lambda o: flip(o["acc_salts"], -1),
             "kappa": lambda o: flip(o["kappa"][1][0][0], 0),
             "eq_a": lambda o: flip(o["eq_a"][0][-1], 0),
             "vs": lambda o: flip(o["vs"][1], 0)}
    for field, edit in [*edits.items(), ("openings", lambda o: flip(o["openings"][0]["row"], 0))]:
        bad = copy.deepcopy(got)
        edit(bad)
        assert set(compare(want, bad)) == {field}, field
