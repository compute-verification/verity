"""The agreement corpus's coverage: every form the Lean verifier distinguishes is exercised by some entry of
`verifier/vectors.json` (which `verifier/ci.py` runs through `agree.py` on both verifiers), or is named in `UNCOVERED`,
which only shrinks. The forms, each read from where the verifier or the agreement check defines it:

- each statement version (`Flock.Tags.all` and `Tags.templated`), covered by a replayable or live set whose
  `lean_statement` it is;
- each record check (the `S<n>` reasons the verifier writes), covered by an expectation whose `why` names it;
- each coin mode of a plain session (`--coins`: `seed`, the default, and `os`), covered by a set staged with it;
- each outcome `agree.py` compares: an accepted and a rejected expectation, an intended divergence (upstream accepts what
  the Lean verifier refuses), each of its `--lean-refuses…` markers on a set, and a set without `--program`
  (`agree.NO_PROGRAM`).

`--zk` sessions are outside the corpus: no pinned upstream build has them."""
from __future__ import annotations

import json
import re
from pathlib import Path

LEAN = Path(__file__).resolve().parents[3] / "core" / "service" / "flock"
VERIFIER = Path(__file__).resolve().parents[1] / "verifier"

#: forms no entry of vectors.json exercises yet; an entry leaves this list in the change that adds its vector
UNCOVERED = frozenset({
    "statement verity/flock-circuit@eb90718f",
    "statement verity/flock-circuit@e51e2b86",
    "statement verity/flock-circuit@967b8d06",
    "statement verity/flock-circuit/types@210d32e1",
    "statement verity/flock-circuit/types@210d32e1+seed-injection",
    "statement verity/flock-circuit@f5df3bbc",
    "statement verity/flock-circuit/types@f5df3bbc",
    "statement verity/flock-circuit@b3727bd9",
    "statement verity/flock-circuit/types@b3727bd9",
    "statement verity/flock-circuit/types@b3727bd9+seed-injection",
    "statement verity/flock-circuit@cec3e347",
    "statement verity/flock-circuit/types@cec3e347",
    "statement verity/flock-circuit/types@cec3e347+seed-injection",
    "statement verity/flock-circuit+seed-injection",
    "statement verity/flock-circuit/types",
    "statement verity/flock-circuit/types+seed-injection",
    "statement verity/flock-circuit+plain-leaves",
    "statement verity/flock-circuit/tmpl",
    "check S0",
    "check S1",
    "check S4",
    "check S5",
    "check S8",
    "check S9",
    "check S10",
    "check S12",
    "check S16",
    "check S18",
})


def statements() -> list[str]:
    """The names of `Flock.Tags.all` and `Tags.templated`, each list's definitions read from `Flock/Tags.lean`."""
    src = (LEAN / "Flock" / "Tags.lean").read_text()
    bodies = dict(re.findall(r"^def (\w+) : Tags :=(.*?)(?=^(?:def |/--|end ))", src, re.S | re.M))
    names = []
    for listed in ("all", "templated"):
        m = re.search(rf"^def {listed} : List Tags := \[(.*?)\]", src, re.S | re.M)
        assert m, f"Flock/Tags.lean has no `def {listed} : List Tags := [...]`: update statements() to read the versions"
        for d in re.findall(r"\w+", m.group(1)):
            name = re.search(r'name := "([^"]+)"', bodies.get(d, ""))
            assert name, f"Tags.{d}: no `name := \"…\"` in its definition: update statements() to read its name"
            names.append(name.group(1))
    return names


def checks() -> set[str]:
    """The record checks: every `S<n>` a reason of the verifier's sources starts with."""
    return {f"S{n}" for p in LEAN.rglob("*.lean") if ".lake" not in p.parts
            for n in re.findall(r'"S(\d+)[:/]', p.read_text())}


def refusal_markers() -> list[str]:
    """`agree.py`'s `--lean-refuses…` options, as the set keys `ci.py` passes them from."""
    return [o[2:].replace("-", "_") for o in re.findall(r'"(--lean-refuses[\w-]*)"', (VERIFIER / "agree.py").read_text())]


def expectations(x):
    """Every expectation object in `x`: a dict with an `expect` or an `accepted` (a session's `expect` object too)."""
    if isinstance(x, dict):
        if "expect" in x or "accepted" in x:
            yield x
        for y in x.values():
            yield from expectations(y)
    elif isinstance(x, list):
        for y in x:
            yield from expectations(y)


def accepted(e) -> bool | None:
    x = e.get("expect", e)
    if isinstance(x, dict):
        return x.get("accepted")
    return {"accept": True, "reject": False}.get(x)


def forms() -> set[str]:
    return ({f"statement {s}" for s in statements()} | {f"check {c}" for c in checks()}
            | {"coins seed", "coins os"}
            | {"outcome accepted", "outcome rejected", "outcome upstream_accepted", "outcome no_program"}
            | {f"outcome {k}" for k in refusal_markers()})


def covered() -> set[str]:
    v = json.loads((VERIFIER / "vectors.json").read_text())
    sets = v["replayable"]["sets"] + v["live"]["sets"]
    es = list(expectations(v))
    out = {f"statement {s['lean_statement']}" for s in sets if "lean_statement" in s}
    out |= {f"check S{n}" for e in es for n in re.findall(r"\bS(\d+)\b", str(e.get("why", "")))}
    out |= {f"coins {s['statement'].get('coins', 'seed')}" for s in sets}
    out |= {f"outcome {k}" for s in sets for k in refusal_markers() if k in s}
    out |= {"outcome no_program" for s in sets if "program" not in s["statement"]}
    out |= {{True: "outcome accepted", False: "outcome rejected"}[a] for a in map(accepted, es) if a is not None}
    if any("upstream_accepted" in e or "upstream_expect" in e for e in es):
        out.add("outcome upstream_accepted")
    return out


def test_every_form_of_the_verifier_has_a_vector():
    have, cov = forms(), covered()
    missing = sorted(have - cov - UNCOVERED)
    assert not missing, (
        "vectors.json exercises no vector of these forms the Lean verifier distinguishes: add a replayable or live set "
        "(or an expectation) that exercises each to verity/ml/flock/verifier/vectors.json, or, only if no pinned build "
        "can record one, name it in UNCOVERED:\n" + "\n".join(missing))


def test_uncovered_only_shrinks():
    have, cov = forms(), covered()
    stale = sorted((UNCOVERED & cov) | (UNCOVERED - have))
    assert not stale, ("remove these from UNCOVERED: vectors.json now exercises each, or the verifier no longer has it:\n"
                       + "\n".join(stale))
