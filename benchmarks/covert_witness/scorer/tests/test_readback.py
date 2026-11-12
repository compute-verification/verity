"""readback.py: a put's read-back compares the served panel with the body as numbers, so the route's writing of a float
with no fractional part as an integer is the body, and any other change is not."""
import decimal
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import readback


def route(x):
    """`x` as the route serves it: a float with no fractional part written as an integer, in its shortest digits."""
    if isinstance(x, float) and x.is_integer():
        return int(decimal.Decimal(repr(x)))
    if isinstance(x, list):
        return [route(v) for v in x]
    if isinstance(x, dict):
        return {k: route(v) for k, v in x.items()}
    return x


BODY = {"id": "training/frontier-fit", "note": "n", "rows": [["fit", 6.52837e+20, 0.5, None, "v3", 4.0]]}


def test_the_routes_integer_for_a_whole_float_is_the_same_number_where_pythons_equality_says_not():
    back = json.loads(json.dumps(route(BODY)))
    assert back["rows"][0][1] == 652837000000000000000
    assert back != BODY and sorted(k for k in {*BODY, *back} if back.get(k) != BODY.get(k)) == ["rows"]
    assert readback.same(back, BODY) and readback.differ(BODY, back) == []


def test_a_changed_cell_a_type_or_a_missing_key_differs():
    assert readback.differ(BODY, BODY | {"rows": [["fit", 6.52838e+20, 0.5, None, "v3", 4.0]]}) == ["rows"]
    assert readback.differ(BODY, BODY | {"rows": [["fit", 6.52837e+20, 0.5, 0, "v3", 4.0]]}) == ["rows"]
    assert readback.differ(BODY, BODY | {"rows": [["fit", 6.52837e+20, 0.5, None, "v3", "4"]]}) == ["rows"]
    assert readback.differ(BODY, BODY | {"rows": [["fit", 6.52837e+20, 0.5, None, "v3"]]}) == ["rows"]
    assert readback.differ(BODY, {k: v for k, v in BODY.items() if k != "note"}) == ["note"]
    assert not readback.same(True, 1) and not readback.same(1, True) and readback.same(False, False)
    assert readback.same(None, None) and not readback.same({}, None)
