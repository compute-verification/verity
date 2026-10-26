"""The pipeline's seeded draws, pinned by digest: the coverage workloads' lengths and prompts, the top-p split fixture's
cases and manifest, and compiled_merge's bootstrap interval.  Every committed workload and fixture was drawn from these
streams, so a change to how a generator is built or seeded fails here first."""
from __future__ import annotations

import contextlib
import hashlib
import io
import json

import pytest

from verity_vllm.pipeline import compiled_merge as cm
from verity_vllm.pipeline import coverage_workloads as gw
from verity_vllm.pipeline import topp_split_probe as tp


class _CharTok:
    def encode(self, text: str) -> list[int]:
        return [ord(c) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(map(chr, ids))


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def test_coverage_length_draws_are_pinned():
    draws = {f"{p}/{c}/{s}": gw.draw_lengths(p, c, s) for p in gw.PROFILES for c in sorted(gw.CTX) for s in (0, 3)}
    assert _digest(draws) == "35bc3641918bdcfc"


def test_coverage_prompt_draws_are_pinned():
    reqs = {f"{p}/{c}/{ps}": gw.build_requests(_CharTok(), gw.draw_lengths(p, c, 3), case="B0", profile=p, ctx=c, seed=3, prompt_seed=ps)
            for p in gw.PROFILES for c in sorted(gw.CTX) for ps in (None, 1003)}
    assert _digest(reqs) == "636ca9201d792690"


def test_topp_case_draws_are_pinned():
    rows = b"".join(tp.gen_random(s, V).tobytes() for s in (1, tp.MANIFEST_SEED) for V in (1024, 3000))
    assert hashlib.sha256(rows).hexdigest()[:16] == "f5267c3302f8f553"
    ties = [tp.gen_ties(s, 1024, S, k, kf, group_bias=gb) for s, S, k, kf, gb in ((5, 4, 12, 0.25, None), (9, 1, 3, 0.5, 0), (11, 32, 24, 0.95, 1))]
    assert hashlib.sha256(b"".join(x.tobytes() for x, _ in ties)).hexdigest()[:16] == "bc7c710066a79ac0"
    assert [p for _, p in ties] == [0.9839564561843872, 0.9970954060554504, 0.997718095779419]


def test_topp_manifest_is_pinned():
    assert tp.MANIFEST_SEED == 20260920
    assert tp.ProduceOptions.__dataclass_fields__["seed"].default == tp.MANIFEST_SEED
    assert [_digest(m) for m in (tp.build_manifest(tp.MANIFEST_SEED), tp.build_manifest(tp.MANIFEST_SEED, with_record=True),
                                 tp.build_manifest(7))] == ["2d7e5964a47bf969", "efdd1af16ebc4625", "021964f345c7a7a7"]


@pytest.mark.parametrize(("record", "digest", "lines"), [(False, "f8749841368067f4", 163), (True, "cbbaac9ce386bca3", 181)])
def test_topp_manifest_command_prints_the_pinned_manifest(record: bool, digest: str, lines: int):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tp.main(tp.ManifestOptions(record=record))
    assert hashlib.sha256(buf.getvalue().encode()).hexdigest()[:16] == digest and buf.getvalue().count("\n") == lines


def test_compiled_merge_bootstrap_interval_is_pinned():
    xs = [0.5, -1.25, 3.0, 2.0, 0.0, 1.5, -0.75]
    assert cm._bootstrap_ci(xs, seed=0) == (-0.32142857142857145, 1.7857142857142858)
    assert cm._bootstrap_ci(xs, seed=1) == (-0.32142857142857145, 1.75)
    assert cm._bootstrap_ci(xs, 500, seed=2) == (-0.35714285714285715, 1.75)
    lo, hi = cm._bootstrap_ci([1.0], seed=0)
    assert lo != lo and hi != hi
