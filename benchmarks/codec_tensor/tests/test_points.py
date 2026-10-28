"""`points`' rates rows: each field from its fastest k = 1 encode, at its best config."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import points as P  # noqa: E402


def test_rates_rows_take_the_fastest_encode_and_the_best_config():
    def lib(n, sq, enc):
        return {"constants": {"n": n}, "analysis": {"read_gbps": 1300.0, "words_per_plain_byte": 0.01,
                                                     "k": {"1": {"x_cuda": 1.0}}},
                "rates": {"square_cuda": {"a": {"per_s": sq}, "b": {"per_s": 2 * sq}}, "square_tc": {},
                          "decode": {"1": {"per_s": 5.0}}, "encode": {"1": {"per_s": enc}, "2": {"per_s": 9.0}}}}
    r1 = {"device": {"name": "gpu"}, "libs": {"p768": lib(768, 1.0, 0.5), "p512": {"constants": {"n": 512}}}}
    r2 = {"device": {"name": "gpu"}, "libs": {"p768": lib(768, 3.0, 0.7)}}
    r3 = {"device": {"name": "gpu"}, "libs": {"p768": lib(768, 2.0, 0.6)}}
    rows = P.rates_rows(P.merge_rates([("r1", r1), ("r2", r2), ("r3", r3)]), 2092.0)
    assert [(r["field"], r["run"], r["squarings_s_cuda"], r["squarings_s_tc"]) for r in rows] == [("p768", "r2", 6.0, None)]
    assert rows[0]["schema"] == P.RATES_SCHEMA and rows[0]["encode_words_s"] == {"1": 0.7, "2": 9.0}
