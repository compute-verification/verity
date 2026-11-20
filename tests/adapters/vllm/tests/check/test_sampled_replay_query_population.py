"""The population of record is Q(P) -- the required manifest: Q_module(P), the module-boundary
population over the Program whose families the canonical VU query selects (the record states the unit and names the
implemented Q) -- never the collector's expected population and never the replay's own enumeration of the Program rows.  `sampled_replay` reconciles its reach against
Q(P) IDENTITY BY IDENTITY and carries the result on the record:

  * `population.query_total`      |Q(P)| and its families (with the manifest / Program / query digests)
  * `population.sampled_strata`   the identities the exact replay REACHES (>= 1 evaluable VU) and EVALUATED, as a share of Q(P)
  * `population.uncovered_by_name` every required identity the exact replay has no evaluable VU for, grouped (family, spec family, class)
                                  with the mechanism of record that covers it -- NOT_YET (state-linked Attention / no-evaluator RoPE),
                                  plane-fed producers, interior-composed VUs, and the query families with no VU row (weights, ids, streams)
  * `population.query_checks`     the fail-closed reconciliation (a VU outside the query, a required identity no row produces, a block
                                  height that disagrees, a rank set that disagrees, a family with no stated mechanism) -> `ok`

and `commit_verdict` states the scope on EVERY PASS and FAILs by name when the reconciliation fails."""
from __future__ import annotations



from tests.check import test_sampled_replay as T
from tests.check.test_sampled_replay import moe_world, world  # noqa: F401  -- fixtures
from verity_vllm.check.replay import driver as SR


def _programs(w):
    return {rid: w.inst for rid in w.prompts}


def _weights(w):
    def read(field, lo, hi):
        flo, _fhi = w.woff[field]
        return w.weights[flo + lo: flo + hi]
    return read


def _manifest_for(w) -> dict:
    """The world's Q(P): every request shares one Program shape (requests arrive together at engine step 0)."""
    return T.v2_manifest(w.inst, tuple(sorted(w.prompts)))


def test_the_cublas_version_is_the_mapped_librarys_or_unavailable_by_name(tmp_path):
    """The version is read from the one libcublas the process maps (never the linker's search order): none mapped, two mapped or one
    that does not load is 'unavailable: <why>'."""
    line = "7f0000000000-7f0000001000 r-xp 00000000 08:01 42 {}\n"
    for files, why in (([], "0 libcublas.so* mapped []"),
                       (["/a/libcublas.so.12", "/b/libcublas.so.13", "/a/libcublasLt.so.12"], "2 libcublas.so* mapped ['/a/libcublas.so.12', '/b/libcublas.so.13']")):
        maps = tmp_path / "maps"
        maps.write_text("7f0000002000-7f0000003000 rw-p 00000000 00:00 0\n" + "".join(line.format(f) for f in files))
        got = SR.cublas_versions(str(maps))
        assert got["cublas_version"] == got["cublas_library"] == f"unavailable: LookupError: {why}"
        assert got["cublaslt_version"].startswith("unavailable: ")
    maps.write_text(line.format(tmp_path / "libcublas.so.12"))
    assert SR.cublas_versions(str(maps))["cublas_version"].startswith("unavailable: OSError: ")


# ---------------------------------------------------------------------------------------------------------------------------------------
# ONE mechanism: the not-executed class DERIVED FROM r_t (Continuation_v1's recurrence over the served list),
# the arithmetic min(a_r + LAG, cap) demoted to a CHECK; which mechanism produced the class is on the record BY NAME
# ---------------------------------------------------------------------------------------------------------------------------------------


def _rule_like_pad(served, *, cap, A, eos):
    """A stand-in with `executed_prefix_by_rule`'s semantics (check/replay/continuation.py): the FIRST EOS stops, the lag slot(s)
    after it are executed rows without a served token, served past the first EOS / an early stop without EOS / over the cap are problems."""
    eos = set(eos)
    a = len(served)
    problems = []
    first = next((i for i, t in enumerate(served) if t in eos), None)
    past = list(range(first + 1, a)) if first is not None else []
    if a > cap:
        problems.append(f"served {a} tokens over the declared cap {cap}")
    if past:
        problems.append(f"served past an EOS at index {first}: positions {past} (the rule stops at the FIRST EOS)")
    if a < cap and first is None:
        problems.append(f"served {a} of its cap {cap} but no served token is a declared EOS id")
    n = cap if first is None else min(first + 1 + A, cap)
    return {"ok": not problems, "executed_prefix_len": n, "eos_step": first if not problems else None, "lag_steps": list(range(first + 1, n)) if first is not None else [],
            "first_eos_index": first, "served_past_first_eos": past, "cap_exhausted": (a == cap and first is None), "eos_stop": first is not None and not past,
            "problems": problems}


class _NotYetPi:
    """Two NOT_YET rows whose address rule raises on the second: the smallest index `population` reads."""
    n_steps = 1
    rows = [{"i": 0, "spec": "AttentionHead_v1{H=2}"}, {"i": 1, "spec": "AttentionHead_v1{H=2}"}]

    def describe(self, r):
        if r["i"] == 1:
            raise KeyError("op_path")
        return {"op_path": "model.layers.0.attn", "step": 0, "members": [("out",)]}


