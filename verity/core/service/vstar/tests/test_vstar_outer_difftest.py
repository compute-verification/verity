"""`RecDeploy`'s honest-run hypotheses checked at one deployed recursive run's tree (``VERITY_VSTAR_OUTER``, the tree
`85-rec-reprice.sh` stages and proves), so that each conjunct of the honest fixture is witnessed at the run whose sessions
the compiled verifier accepted: `IrShaped` and `honest_half`'s `hTops` by `vstar-difftest`'s ``ir_shaped`` on the
stager's inputs at the run (`test_vstar_in_difftest`'s ``src``, read from the run's inner session), and
`HonestWitSatisfies` by ``vstar-trace --sat`` on every V* part's verifier files at its session's draw
(`test_vstar_trace_difftest`'s check). Skips unless ``VERITY_VSTAR_OUTER`` names such a tree."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest
import verity_difftest as D

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_vstar_in_difftest as IN
from test_vstar_trace_difftest import MASK_SEED, sat, satisfied
from verity_flock import rec_vstage as S
from verity_flock import rec_vstar as V

LEAN = IN.LEAN

#: a deployed recursive run's tree (`85-rec-reprice.sh`'s ``$DD``: ``inner/s0``; ``v/`` with each part's verifier files and
#: ``inst.bin``, ``fold.json``, ``points.json``, ``acc-salts.bin``; ``v-sess/<part>/session.json`` and ``<part>.stmt``) and
#: the ``rec-inner.json`` its Lean step read (``VERITY_VSTAR_REC_INNER``, else the tree's own)
OUTER = Path(os.environ["VERITY_VSTAR_OUTER"]) if os.environ.get("VERITY_VSTAR_OUTER") else None
pytestmark = [IN.pytestmark, pytest.mark.skipif(OUTER is None, reason="VERITY_VSTAR_OUTER names no deployed run's tree")]
server = IN.server


@pytest.fixture(scope="module")
def lean() -> Path:
    return D.build(LEAN, "vstar-trace")


def outer_parts(outer: Path) -> dict[str, Path]:
    """Each V* part's session (``v-sess/<name>``) and its statement's directory under ``v/`` (``<name>.stmt``)."""
    return {f.stem: outer / "v" / f.read_text().strip().rsplit("/v/", 1)[1]
            for f in sorted((outer / "v-sess").glob("*.stmt"))}


def outer_src(outer: Path) -> dict:
    """The stager's inputs at the run, as `part_in` reads them: the inner session's records (``inner/s0``) at the run's
    fold and points, its running sums' salts, and each level's fresh salts read back from its ``inst.bin``."""
    v = outer / "v"
    assert (v / "acc-salts.bin").exists(), "acc-salts.bin: the prover's salts, which `acc_salts` would otherwise draw"
    public, private, proofs = V.load(outer / "inner" / "s0")
    vf = S.verifier(public, json.loads((v / "fold.json").read_text()), json.loads((v / "points.json").read_text()))
    qs = V.queries(public, private, proofs)
    ts = {rep: V.tops(public, private, f"circuit/rep{rep}", "circuit") for rep in sorted(proofs)}
    mrs = S.message_rounds(public, private, vf.sh)
    nlev = len(vf.reps[0].info["kappa"])
    levels = sorted(p.name for p in v.glob("L*") if (p / "inst.bin").exists())
    h = IN.h128
    return {"msgs": [[[mr["rows"][r], mr["salts"][r].hex()] for r in range(2)] for mr in mrs],
            "openings": [{"rep": q.rep, "level": q.level, "j": q.j, "pos": q.pos, "row": [h(x) for x in q.row],
                          "sibs": [x.hex() for x in q.sibs]} for q in qs],
            "tops": [[[[n.hex(), t.hex()] for n, t in zip(top.nodes, top.salts)] for top in ts[r]] for r in sorted(ts)],
            "acc_salts": [x.hex() for x in S.acc_salts(v / "acc-salts.bin", len(qs) + 1)],
            "kappa": [[[[h(x) for x in row] for row in k] for k in vf.reps[r].info["kappa"]] for r in range(2)],
            "eq_a": [[[h(x) for x in vf.reps[r].info["queries"][lv][1]] for lv in range(nlev)] for r in range(2)],
            "vs": [[h(x) for x in rv.v] for rv in vf.reps],
            "fresh": {nm: {p: [x.hex() for x in IN._salts_of(v / nm, p)] for p in ("row", "aux")} for nm in levels}}


def ir_shaped(server, src: dict, rec_inner: dict) -> dict:
    kind, text = server.call("ir_shaped", {"src": src, "rec_inner": rec_inner})
    assert kind == "output", text
    return json.loads(text)


@pytest.mark.slow
def test_the_deployed_runs_openings_are_shaped(server):
    """`RecDeploy.IrShaped` at a deployed run (``VERITY_VSTAR_OUTER``): both reps open, and at each of V*'s level
    statements at the run's inner shape the level's openings are as many as the statement's instances, of its lanes and
    siblings, each below its level's committed tops; and `honest_half`'s `hTops` (each level above 0's reps' tops differ).
    Not vacuous: one opening fewer, or one past the tops, is refused."""
    src = outer_src(OUTER)
    rec_inner = json.loads(Path(os.environ.get("VERITY_VSTAR_REC_INNER", OUTER / "rec-inner.json")).read_text())
    r = ir_shaped(server, src, rec_inner)
    print(json.dumps({"ir_shaped": r}), flush=True)
    assert r["ok"] is True and r["tops_at"] is True and len(r["levels"]) > 0, r
    assert all("tops_refused" not in lv for lv in r["levels"]), r
    q = next(i for i, o in enumerate(src["openings"]) if o["level"] == r["levels"][-1]["level"])
    fewer = dict(src, openings=src["openings"][:q] + src["openings"][q + 1:])
    assert ir_shaped(server, fewer, rec_inner)["ok"] is False
    lv = r["levels"][-1]
    past = json.loads(json.dumps(src))
    past["openings"][q]["pos"] += lv["tops_rows"] << lv["H"]
    moved = ir_shaped(server, past, rec_inner)
    assert moved["ok"] is False and not moved["levels"][-1]["in_tops"], moved


@pytest.mark.slow
def test_the_deployed_runs_parts_satisfy(lean):
    """`RecDeploy.HonestWitSatisfies` at a deployed run (``VERITY_VSTAR_OUTER``): every V* part's trace satisfies the table
    ``verify --zk`` sets up on its verifier's files at its session's draw, under the zero mask and a random one."""
    parts = outer_parts(OUTER)
    assert parts, OUTER
    checked = {}
    for nm, d in parts.items():
        for mask in ("zero", MASK_SEED):
            r = sat(lean, d, "--session", str(OUTER / "v-sess" / nm / "session.json"), "--mask", mask)
            print(json.dumps({"sat": nm, "mask": mask, **r}), flush=True)
            assert satisfied(r) and r["drawn"] is True, (nm, r)
        checked[nm] = {k: r[k] for k in ("m", "k_log", "blocks", "regions", "region_bits")}
    print(json.dumps({"satisfied": checked}))
    assert all(any(c.startswith(k) for c in checked) for k in ("L", "cap", "alg-")), checked
