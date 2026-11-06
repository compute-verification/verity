"""An adversarial case against the adapter that adopts `prove_path.py audit --serve-run` audits as these runs' prove
path. The attack is on the *assembly*, not the pieces.

A report could read OK with w212's advice registered **after serving**, before the first draw. A content tie for such
a Commit window opens the audited arm's committed root to exactly the report's advice of that Commit window
(`chain.opens`, which binds content: the opening's bytes must equal the v0 encoding of the report's advice,
re-registered to the registered root).

The finding was that this content tie was **not in the report's `ok`**. For a window registered apart (its own salt, as
w212's), the report's prove-path binding (`served_commit`) held the audited arm by:
- its **public identity** — the endpoint's own `windows.jsonl` run id and root, which the arm self-declares and which are
  public, so trivially matched; and
- `the_reports_bits`, which was `adv.bits == row["v0_bits"]`, a v0 **bit count** (`chain.COUNTS`), not the content.

`chain.opens` — the only content check — was called only for the endpoint's own advice inside `served_advice`, never for
the adopted `--serve-run` arms, and `e2e.report`'s `ok` never referenced it or the advice-check. `--prove-path-registered`'s
`at_serve_time` was unverified job-stated metadata that gated nothing in `ok`; a prover could claim serve-time for late
advice and only the disclosure (`stubs`, `_registered`) changed.

The fix, which these cases now hold as regressions: every adopted arm's advice is held by
content (`chain.held_to`): its root is the registered root, and the registration's opening opens it to the report's
advice of the window (`chain.opens`), which the report's `ok` requires; the bit count never passes. `at_serve_time` is
read off the registration's own time, before the arm's sample (its `advice_before_draw` receipt) and no later than the next
window's close in the endpoint's trace (`chain.closes`), never the job's flag.
"""
from __future__ import annotations

import dataclasses
import json
import os
import shutil

from test_chain import EP, I1, _apart, chain


def _served(ep=EP):
    """`served_advice` on the endpoint fixture (or a copy ``ep``) with the run's link-windows shifted so only Commit
    windows 1 and 2 fall in them (as w212 is one of the honest run's windows, registered apart): window 1 is of the
    honest run."""
    io = json.loads((ep / "io.json").read_text())["requests"]
    first = chain._first_steps(chain._jsonl(ep / "advice_trace.jsonl"))
    info, b = chain.served_advice(ep, ep / "advice", {"origin_unix": first[0] + 1.2, "windows": 2, "window_s": 6.0},
                                  io, 60, [128001])
    return info, b


def _serve_arm(d, name, w, bits, advice_root, *, forged=False, before_draw=None):
    """A `--serve-run` arm of the endpoint's Commit window ``w`` whose committed advice has root ``advice_root`` and v0
    bit count ``bits``, and whose rank's `advice_before_draw` is ``before_draw`` (absent when None). Its serve detail
    self-declares the endpoint's own (public) run id and root for ``w``."""
    c = next(x for x in chain._jsonl(EP / "windows.jsonl") if int(x["window"]) == w)
    adv = {"root": advice_root, "bits": bits}
    rank = {"ok": True, "audit": {"accepted": not forged, "wrong_units": None if forged else {"bound": 3}},
            "population": 10, "draw": [1, 2], "seconds": 1.5, "source": "stand-in", "negatives": {"a": True}}
    if forged:
        rank["forged"] = {"ok": True}
    if before_draw is not None:
        rank["advice_before_draw"] = {"ok": before_draw}
    s = {"ok": True, "forged": forged, "world": 1, "advice": adv, "ranks": {"0": rank}, "run_root": c["run_root"],
         "serve": {"ok": True, "codes": [], "window": w,
                   "detail": {"window": w, "run_id": c["run_id"], "published_root": c["run_root"], "advice": adv}}}
    (d / name).mkdir(parents=True)
    (d / name / "summary.json").write_text(json.dumps(s))


_LATE = {"1": {"by": "the lane", "at_serve_time": False, "at": "2026-10-24T15:33:57.704Z",
               "first_draw": "2026-10-24T15:48:10.322Z", "ref": "r-reg"}}


def test_a_serve_run_arm_is_bound_by_its_content_never_its_v0_bit_count(tmp_path):
    """The report's prove-path binding is the content: an arm whose root isn't a registration of its window's advice is
    refused whatever its v0 bit count, and one whose root is is bound whatever count it declares."""
    info, b = _served()
    cnt = {x["window"]: x for x in b["commit_windows"]}[1]["v0_bits"]
    own = json.loads((EP / "advice/w1/registration.json").read_text())["binding"]["advice"]["root"]

    for name, root in (("honest", "a" * 128), ("other_root", "dead" * 32)):
        _serve_arm(tmp_path / name, "served", 1, cnt, root)
        g = chain.prove_path([tmp_path / name], info, b["bound"], b, EP, _LATE)
        c = g["arms"]["w1 served"]["commit"]
        assert g["bound"] is False and g["tied"] is False and c["the_registered_root"] is False
        assert g["tied_why"] == f"w1 served: its advice root isn't the one registered at {EP / 'advice/w1'}"
        assert "the_reports_bits" not in c

    _serve_arm(tmp_path / "short", "served", 1, cnt - 1, own)
    g = chain.prove_path([tmp_path / "short"], info, b["bound"], b, EP, _LATE)
    assert g["bound"] is True and g["tied"] is True and g["ok"] is True


def test_the_content_tie_is_opens_and_every_adopted_arm_takes_it(tmp_path):
    """`chain.opens` is a real content binding — it opens the registered root to exactly one advice — and every adopted
    `--serve-run` arm takes it (`chain.held_to`): a registration apart of other advice, its root the arm's and naming
    the window's Commit, is refused because its opening doesn't open to the report's advice; one of the same advice
    under salts of its own is bound."""
    reg = json.loads((EP / "advice/w1/registration.json").read_text())
    opening = json.loads((EP / "advice/w1/private/opening.json").read_text())
    adv = I1.from_json(json.loads((EP / "advice/w1/private/advice.json").read_text()))
    assert chain.opens(reg, opening, adv) is True
    other = dataclasses.replace(adv, release=tuple(r + 1 for r in adv.release))
    assert I1.encode(other) != I1.encode(adv) and chain.opens(reg, opening, other) is False

    info, b = _served()
    cnt = {x["window"]: x for x in b["commit_windows"]}[1]["v0_bits"]
    forged, apart = _apart(tmp_path / "reg-forged", 1, other), _apart(tmp_path / "reg-apart", 1)
    for name, root in (("forged", forged), ("apart", apart)):
        _serve_arm(tmp_path / name, "served", 1, cnt, root)
        _serve_arm(tmp_path / name, "served-forged", 1, cnt, root, forged=True)
    g = chain.prove_path([tmp_path / "forged"], info, b["bound"], b, EP,
                         {"1": {"by": "the lane", "registration": str(tmp_path / "reg-forged")}})
    c = g["arms"]["w1 served"]["commit"]
    assert g["bound"] is False and g["tied"] is False and (c["the_registered_root"], c["opens"]) == (True, False)
    assert g["tied_why"].startswith(f"w1 served: the registration's opening at {tmp_path / 'reg-forged'} doesn't "
                                    "open its root to the report's advice of Commit window 1;")
    g = chain.prove_path([tmp_path / "apart"], info, b["bound"], b, EP,
                         {"1": {"by": "the lane", "registration": str(tmp_path / "reg-apart")}})
    assert g["bound"] is True and g["tied"] is True and g["arms"]["w1 served"]["commit"]["opens"] is True


def test_at_serve_time_is_read_off_the_registration_never_the_job(tmp_path):
    """`--prove-path-registered`'s `at_serve_time` is no say: the same registration reads the same whatever the job
    claims, after serving when its time is after the next window's close in the endpoint's trace and at serve time when
    it is before its sample (the arm's own receipt) and not after that close."""
    ep = tmp_path / "ep"
    shutil.copytree(EP, ep)
    info, b = _served(ep)
    cnt = {x["window"]: x for x in b["commit_windows"]}[1]["v0_bits"]
    own = json.loads((ep / "advice/w1/registration.json").read_text())["binding"]["advice"]["root"]
    d = tmp_path / "arm"
    _serve_arm(d, "served", 1, cnt, own, before_draw=True)
    closed = chain.closes(chain._jsonl(ep / "advice_trace.jsonl"))[1]
    lie = {"1": {"by": "the lane", "at_serve_time": True}}

    def pp_stub(g):
        s = [x for x in chain.stubs({"chain": {"advice": info, "bits": b, "prove_path": g}}, True)
             if x.startswith("Prove path")]
        return s[0] if s else ""

    os.utime(ep / "advice/w1/registration.json", (closed + 852.0,) * 2)
    g_late = chain.prove_path([d], info, b["bound"], b, ep, _LATE)
    g_lie = chain.prove_path([d], info, b["bound"], b, ep, lie)
    assert g_late["bound"] is True and g_lie["bound"] is True and g_late["ok"] == g_lie["ok"] is True
    late, lied = (g["arms"]["w1 served"]["commit"]["registered"] for g in (g_late, g_lie))
    assert late["at_serve_time"] is lied["at_serve_time"] is False and late["at"] == lied["at"] != _LATE["1"]["at"]
    late_text = ("after serving, by the lane, 852.0 s after its Commit window closed, 851.3 s after the next window "
                 "closed")
    assert late_text in pp_stub(g_lie)
    assert chain._registered(lied) == late_text

    os.utime(ep / "advice/w1/registration.json", (closed - 1.0,) * 2)
    g = chain.prove_path([d], info, b["bound"], b, ep, _LATE)
    assert g["arms"]["w1 served"]["commit"]["registered"]["at_serve_time"] is True
    assert pp_stub(g) and "w1's advice was registered" not in pp_stub(g)
    assert chain._registered(g["arms"]["w1 served"]["commit"]["registered"]) == (
        "by the lane at serve time (r-reg), 1.0 s before its Commit window closed, before its draw and 1.7 s before "
        "the next window closed")
