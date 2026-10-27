"""Test-only witnesses beside the sampling family's Definitions (`registry.sampling`), not evaluators.

`add_rz_ftz_v1_model` is the round-to-nearest model of `add.rz.ftz.f32` that `nv_log1pf` used before its add was made a true
toward-zero one, and `nv_log1pf_v1_model` is `nv_log1pf` under it. The tests compare both with the registered evaluators on every
word; no Definition calls them. Their vectorised forms are `sampling_rows`'s."""
from __future__ import annotations

import math

from verity.ml.definitions.registry.sampling import M32, ONE_BITS, _f, _ftz, _log1pf_from_sum, _u


def add_rz_ftz_v1_model(a: int, b: int) -> int:
    """The `add.rz.ftz.f32` model that `nv_log1pf` used before, kept word for word so the finding stays reproducible: it is
    round-to-NEAREST, not toward zero -- both toward-zero branches step the fp32 value `rf` with `math.nextafter(rf, 0.0)`
    (one DOUBLE ulp) and `_u` rounds that straight back to `rf`'s word, so
    add_rz_ftz_v1_model(a, b) == ftz(RN32(a + b)) on every input (0 differences on 300,000 random pairs; a true RZ differs on 21.5 %).

    Its only caller was `nv_log1pf` (`__fadd_rz(a, 1.0f)`), where the sum feeds ONLY the binade selector `e`; the exhaustive scan of all
    2^32 words (test_sampling_rows, opt-in) shows nv_log1pf under this model and under the true RZ (`add_rz_ftz`) differ on exactly ONE
    word, a = 0x3EFFFFFF (+0.49999997, a positive input the Gumbel lane never forms: it feeds -u, u in [2^-31', 1)).  GumbelNoiseLane_v1's
    word map is therefore the same function under either add; the registered evaluator now uses the true RZ (`nv_log1pf`) and this
    model is referenced by the tests alone."""
    fa, fb = _f(_ftz(a)), _f(_ftz(b))
    s = fa + fb
    if s == 0.0 or math.isinf(s) or math.isnan(s):
        return _ftz(_u(s))
    bb = s - fa
    e = (fa - (s - bb)) + (fb - bb)
    r = _u(s)
    rf = _f(r)
    if abs(rf) > abs(s):
        r = _u(math.nextafter(rf, 0.0))          # dead: a double ulp below rf rounds back to rf (F-stoch-r17-01)
    elif rf == s and e != 0.0 and (e < 0) == (s > 0):
        r = _u(math.nextafter(rf, 0.0))          # dead, the same way
    return _ftz(r)


def nv_log1pf_v1_model(a: int) -> int:
    """`nv_log1pf` as it evaluated before: `__fadd_rz` modelled RN (`add_rz_ftz_v1_model`).  Test-only:
    the exhaustive comparison with `nv_log1pf` is the proof that the registered lane did not move."""
    return _log1pf_from_sum(a & M32, add_rz_ftz_v1_model(a & M32, ONE_BITS))


__all__ = ["add_rz_ftz_v1_model", "nv_log1pf_v1_model"]
