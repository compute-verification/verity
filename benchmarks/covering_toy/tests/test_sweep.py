import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sweep as SW  # noqa: E402

NAME = "mlp222-b1-t2"


def _point(G, k, mode, n, lb, cap_inner):
    return {"instance": NAME, "k": k, "mode": mode, "s": SW.gate_budget(G, k, mode), "gates": G.N,
            "inner": len(G.inner()), "full": {"n": n, "lb": lb, "optimal": lb >= n, "cover": [], "blocks": []},
            "M": None, "cap": {"inner": {"n": cap_inner, "ub": cap_inner}},
            "lean_cw1": {"D": 3, "M_star": 0, "per_block": 1.0, "n_lb": 0.0},
            "production_beta1": {"n_lb": 1.0}, "trivial": 0.0}


def test_merge_keeps_only_sound_brackets(tmp_path):
    G = SW.build(NAME)
    a, b, out = tmp_path / "a", tmp_path / "b", tmp_path / "out"
    a.mkdir()
    b.mkdir()
    (a / f"{NAME}.json").write_text(json.dumps([_point(G, 4, "eta1", 10, 3, 90), _point(G, 4, "inf", 9, 8, 90)]))
    (b / f"{NAME}.json").write_text(json.dumps([_point(G, 4, "eta1", 12, 5, 90), _point(G, 6, "inf", 11, 2, 90)]))
    SW.merge([a, b], out, [NAME])
    pts = {(r["k"], r["mode"]): r["full"] for r in json.loads((out / f"{NAME}.json").read_text())}
    # the smallest covering found, and a larger point's bound carried down
    assert (pts[4, "eta1"]["n"], pts[4, "eta1"]["lb"]) == (10, 8)
    # a smaller point's covering carried up, and ceil(inner / cap) as a bound
    assert pts[6, "inf"]["n"] == 9 and pts[6, "inf"]["via"] == [4, "inf"]
    assert pts[6, "inf"]["lb"] == 2
    assert pts[4, "inf"]["optimal"] is False and (out / "table.md").exists()
