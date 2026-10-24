"""REDTEAM phase 13, target 1 (REDTEAM-ADVICE, note:network-accounting/20261024T0558Z-finding-redteam-v3) against #3778,
NET-E2E's adapter that adopts `prove_path.py audit --serve-run` audits as these runs' prove path, as in G-A's report of
record r20261024-163242-7e60. The attack is on the *assembly*, not the pieces.

The report of record reads OK with w212's advice registered **after serving** (the lane, 14.2 min before the first
draw). The run's own `out/advice-check.json` is the content tie for that window: `the_audited_root_is_the_lanes` and
`the_lanes_opening_opens_its_root_to_the_reports_w212_advice`, both True — the lane's opening opens 7679's committed root
to exactly the report's w212 advice (`chain.opens`, which really binds content: the opening's bytes must equal the v0
encoding of the report's advice, re-registered to the registered root). So in the run of record the content is pinned.

The finding is that this content tie is **not in the report's `ok`**. For a window registered apart (its own salt, as
w212's), the report's prove-path binding (`served_commit`) holds the audited arm by:
- its **public identity** — the endpoint's own `windows.jsonl` run id and root, which the arm self-declares and which are
  public, so trivially matched; and
- `the_reports_bits`, which is `adv.bits == row["v0_bits"]`, a v0 **bit count** (`chain.COUNTS`), not the content.

`chain.opens` — the only content check — is called only for the endpoint's own advice inside `served_advice`, never for
the adopted `--serve-run` arms, and `e2e.report`'s `ok` never references it or the advice-check. So a consumer trusting
report.json's `ok` accepts an audit of advice whose content differs from what served, as long as the v0 bit count matches
and the public identity is named. `--prove-path-registered`'s `at_serve_time` is unverified job-stated metadata that gates
nothing in `ok`; a prover can claim serve-time for late advice and only the disclosure (`stubs`, `_registered`) changes —
hiding exactly the late registration under which the gap matters.

One-line fix: fold the advice-check into the report's `ok` — for a window held by v0 bits only, require `chain.opens`
(the registered root opens to the report's advice) and `audited_root == the registered root`, not just the bit count; and
derive `at_serve_time` from the registration's recorded time against the serve window rather than trusting the job's
claim. Owner: NET-E2E.
"""
from __future__ import annotations

import inspect
import json

import pytest

from test_chain import EP, chain, I1, e2e


def _served():
    """`served_advice` on the endpoint fixture with the run's link-windows shifted so only Commit windows 1 and 2 fall
    in them (as w212 is one of the honest run's windows, registered apart): window 1 is of the honest run."""
    io = json.loads((EP / "io.json").read_text())["requests"]
    first = chain._first_steps(chain._jsonl(EP / "advice_trace.jsonl"))
    info, b = chain.served_advice(EP, EP / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  io, 60, [128001])
    return info, b


def _serve_arm(d, name, w, bits, advice_root, *, forged=False):
    """A `--serve-run` arm of the endpoint's Commit window ``w`` whose committed advice has root ``advice_root`` and v0
    bit count ``bits``. Its serve detail self-declares the endpoint's own (public) run id and root for ``w``."""
    c = next(x for x in chain._jsonl(EP / "windows.jsonl") if int(x["window"]) == w)
    adv = {"root": advice_root, "bits": bits}
    rank = {"ok": True, "audit": {"accepted": not forged, "wrong_units": None if forged else {"bound": 3}},
            "population": 10, "draw": [1, 2], "seconds": 1.5, "source": "stand-in", "negatives": {"a": True}}
    if forged:
        rank["forged"] = {"ok": True}
    s = {"ok": True, "forged": forged, "world": 1, "advice": adv, "ranks": {"0": rank}, "run_root": c["run_root"],
         "serve": {"ok": True, "codes": [], "window": w,
                   "detail": {"window": w, "run_id": c["run_id"], "published_root": c["run_root"], "advice": adv}}}
    (d / name).mkdir(parents=True)
    (d / name / "summary.json").write_text(json.dumps(s))


_LATE = {"1": {"by": "the lane", "at_serve_time": False, "at": "2026-10-24T15:33:57.704Z",
               "first_draw": "2026-10-24T15:48:10.322Z", "ref": "r-reg"}}


def test_the_serve_run_binding_is_the_v0_bit_count_not_the_content(tmp_path):
    """The report's prove-path binding for a window registered apart is the v0 bit count, invariant to the committed
    advice root: the same count binds True under any root (the report never sees the content), and only an off-by-one
    count is refused."""
    info, b = _served()
    cnt = {x["window"]: x for x in b["commit_windows"]}[1]["v0_bits"]

    honest = tmp_path / "honest"
    _serve_arm(honest, "served", 1, cnt, "a" * 128)
    g1 = chain.prove_path([honest], info, b["bound"], b, EP, _LATE)
    assert g1["bound"] is True and g1["ok"] is True
    c1 = g1["arms"]["w1 served"]["commit"]
    assert c1["the_reports_root"] is False and c1["the_reports_bits"] is True  # bound by the count, not the root

    other_root = tmp_path / "other_root"
    _serve_arm(other_root, "served", 1, cnt, "dead" * 32)
    g2 = chain.prove_path([other_root], info, b["bound"], b, EP, _LATE)
    assert g2["bound"] is True and g2["arms"]["w1 served"]["commit"]["advice_root"] == "dead" * 32

    short = tmp_path / "short"
    _serve_arm(short, "served", 1, cnt - 1, "a" * 128)
    g3 = chain.prove_path([short], info, b["bound"], b, EP, _LATE)
    assert g3["bound"] is False and "isn't held to the report's registration" in g3["bound_why"]


def test_the_content_tie_is_opens_and_the_report_ok_never_calls_it(tmp_path):
    """`chain.opens` is a real content binding — it opens the registered root to exactly one advice. It is the only
    content check, and it lives in `served_advice` for the endpoint's own advice; the report's `ok` (and the whole
    prove-path adoption path) never calls it, so the content of an adopted `--serve-run` arm is bound only out of band in
    advice-check.json."""
    reg = json.loads((EP / "advice/w1/registration.json").read_text())
    opening = json.loads((EP / "advice/w1/private/opening.json").read_text())
    adv = I1.from_json(json.loads((EP / "advice/w1/private/advice.json").read_text()))
    assert chain.opens(reg, opening, adv) is True

    import dataclasses
    other = dataclasses.replace(adv, release=tuple(r + 1 for r in adv.release))
    assert I1.encode(other) != I1.encode(adv) and chain.opens(reg, opening, other) is False  # content is bound by opens

    assert "opens(" not in inspect.getsource(e2e.report)
    assert "opens(" not in inspect.getsource(chain.prove_path)
    assert "opens(" not in inspect.getsource(chain.served_commit)
    # the only caller of opens is served_advice (the endpoint's own advice), not any adopted-arm path
    callers = [n for n in dir(chain) if inspect.isfunction(getattr(chain, n)) and n != "opens"
               and getattr(chain, n).__module__ == chain.__name__ and "opens(" in inspect.getsource(getattr(chain, n))]
    assert callers == ["served_advice"]


def test_at_serve_time_is_unverified_and_only_changes_the_disclosure(tmp_path):
    """`--prove-path-registered`'s `at_serve_time` gates nothing: a lie claiming serve-time for the same late window
    leaves `bound`/`ok` unchanged, and only the disclosure (`stubs`, `_registered`) changes — removing the "after
    serving" flag that is the report's one signal of the late registration."""
    info, b = _served()
    cnt = {x["window"]: x for x in b["commit_windows"]}[1]["v0_bits"]
    d = tmp_path / "arm"
    _serve_arm(d, "served", 1, cnt, "f" * 128)

    lie = {"1": {"by": "the lane", "at_serve_time": True}}
    g_late = chain.prove_path([d], info, b["bound"], b, EP, _LATE)
    g_lie = chain.prove_path([d], info, b["bound"], b, EP, lie)
    assert g_late["bound"] is True and g_lie["bound"] is True and g_late["ok"] == g_lie["ok"] is True

    def pp_stub(g):
        s = [x for x in chain.stubs({"chain": {"advice": info, "bits": b, "prove_path": g}}, True)
             if x.startswith("Prove path")]
        return s[0] if s else ""

    assert "after serving" in pp_stub(g_late) and "after serving" not in pp_stub(g_lie)
    assert "after serving" in chain._registered(_LATE["1"]) and "serve time" in chain._registered(lie["1"])
