"""The Lean verifier's stratified sample against core's (PR #164's `verity.core.protocols.sampled_proofs.one_stage.draw`, PR #161's `Stratified`).

For each program (A4 P6's six-template population, as #164's test builds it, and small and random programs):

- the law: `flock-verify draw --program P --partition Q --stratified K` against `draw.derive`, and the population against
  `partition.units`;
- the units on a given stream: against `reference_draw`, the sampler of record written out here (each stratum's
  `subset k_s n_s` on its own segment of `2 k_s · width(n_s) + 64` bytes, local index j to the stratum's j-th unit);
- the sample: `draw.check_draw` accepts it, from a given stream and from the operating system's randomness;
- U2: `flock-verify draw-test` with the stratified flags against `draw.check_draw` on mutations of an accepted draw;
- one sample over J = 2 and 3 tables: `flock-verify draw --tables J` against `draw.tables_problem`, the same draw where both
  take it and the same refusal where both refuse;
- on a small program, each unit's inclusion frequency over OS draws against `k_s / n_s`;
- A4's law as core's `Stratified` (`audit.core_law`): the count bound and a 287-unit stratum's own bound at 2^-20.

Needs core with #164 on PYTHONPATH (the repository root). Prints one JSON line; exit 0 iff
everything agrees.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

A4_SIZES = (287, 6_171_648, 11_480, 287, 287, 587_776)


def width(n: int) -> int:
    return (n.bit_length() + 64 + 7) // 8


def _uniform(n: int, src: bytes, pos: int):
    w = width(n)
    space = 2 ** (8 * w)
    while pos + w <= len(src):
        x = int.from_bytes(src[pos:pos + w], "big")
        pos += w
        if x < space - space % n:
            return x % n, pos
    return None


def _subset(k: int, n: int, src: bytes):
    sw: dict[int, int] = {}
    pos = 0
    for i in range(k):
        r = _uniform(n - i, src, pos)
        if r is None:
            return None
        j, pos = i + r[0], r[1]
        sw[i], sw[j] = sw.get(j, j), sw.get(i, i)
    return sorted(sw.get(i, i) for i in range(k))


def _nth(ranges, j: int) -> int:
    for lo, hi in ranges:
        if j < hi - lo:
            return lo + j
        j -= hi - lo
    raise IndexError(j)


def reference_draw(law: dict, src: bytes) -> list[int] | None:
    """The stratified sampler of record on the given bytes: stratum s's stream is the next `2 k_s · width(n_s) + 64` bytes."""
    out, at = [], 0
    for s in law["strata"]:
        n_s = sum(hi - lo for lo, hi in s["units"])
        need = 2 * s["k"] * width(n_s) + 64
        local = _subset(s["k"], n_s, src[at:at + need])
        if local is None:
            return None
        out += [_nth(s["units"], j) for j in local]
        at += need
    return sorted(out)


def _templates(m: int):
    from verity.ml.circuits.defs import CompositeDefinition, PrimitiveDefinition, bind
    from verity.ml.circuits.types import Array, Value
    xor = PrimitiveDefinition("StratAgreeXor8", 1, (("a", Value(8)), ("b", Value(8))), Value(8), lambda a, b: a ^ b,
                              register=False)
    return [bind(CompositeDefinition(f"StratAgreeT{i}", 0, (), lambda S: ((("x", Array(2, Value(8))),), Value(8)),
                                     lambda B, S, x, i=i: B.call(xor, x[i % 2], x[(i + 1) % 2]), register=False))
            for i in range(m)]


def _lines(p: subprocess.CompletedProcess, tag: str) -> list:
    return [json.loads(line[len(tag) + 1:]) for line in p.stdout.splitlines() if line.startswith(tag + " ")]


def _mutations(good: dict, rng: random.Random) -> list[tuple[str, dict]]:
    """Mutations of an accepted stratified sample, each a change U2 or the law check should see (or not)."""
    def m(f):
        d = json.loads(json.dumps(good))
        f(d)
        return d
    S, U = good["strata"], good["units"]
    out = [("as drawn", m(lambda d: None))]
    out += [(f"without {k}", m(lambda d, k=k: d.pop(k))) for k in ("law", "population", "k", "strata", "units")]
    out += [("an extra key", m(lambda d: d.update(p="1/2"))), ("law subset", m(lambda d: d.update(law="subset"))),
            ("k + 1", m(lambda d: d.update(k=d["k"] + 1))), ("k - 1", m(lambda d: d.update(k=d["k"] - 1))),
            ("k 0", m(lambda d: d.update(k=0))), ("population + 1", m(lambda d: d.update(population=d["population"] + 1))),
            ("no strata", m(lambda d: d.update(strata=[]))), ("strata reversed", m(lambda d: d["strata"].reverse())),
            ("a unit dropped", m(lambda d: d["units"].pop(rng.randrange(len(U))))),
            ("units unsorted", m(lambda d: d["units"].reverse()) if len(U) > 1 else m(lambda d: d["units"].append(-1))),
            ("a unit twice", m(lambda d: d["units"].insert(0, d["units"][0]))),
            ("a unit past the population", m(lambda d: d["units"].append(d["population"])))]
    for i, s in enumerate(S):
        out += [(f"stratum {i} k + 1", m(lambda d, i=i: d["strata"][i].update(k=d["strata"][i]["k"] + 1))),
                (f"stratum {i} renamed", m(lambda d, i=i: d["strata"][i].update(template=d["strata"][i]["template"] + "x"))),
                (f"stratum {i} extra key", m(lambda d, i=i: d["strata"][i].update(n=1))),
                (f"stratum {i} without k", m(lambda d, i=i: d["strata"][i].pop("k"))),
                (f"stratum {i} no ranges", m(lambda d, i=i: d["strata"][i].update(units=[]))),
                (f"stratum {i} last range grown", m(lambda d, i=i: d["strata"][i]["units"][-1].__setitem__(1, d["strata"][i]["units"][-1][1] + 1))),
                (f"stratum {i} first range shrunk", m(lambda d, i=i: d["strata"][i]["units"][0].__setitem__(0, d["strata"][i]["units"][0][0] + 1)))]
        lo, hi = s["units"][0]
        if hi - lo >= 2:
            out.append((f"stratum {i} first range split", m(lambda d, i=i, lo=lo, hi=hi: d["strata"][i]["units"].__setitem__(
                slice(0, 1), [[lo, lo + 1], [lo + 1, hi]]))))
        out.append((f"stratum {i} empty range", m(lambda d, i=i, lo=lo: d["strata"][i]["units"].insert(0, [lo, lo]))))
        inside = [u for u in U if any(a <= u < b for a, b in s["units"])]
        outside = [u for u in range(good["population"]) if u not in U and not any(a <= u < b for a, b in s["units"])][:1] \
            if good["population"] <= 100_000 else []
        if inside and outside:
            out.append((f"stratum {i}: a unit moved out", m(lambda d, a=inside[0], b=outside[0]: d.update(
                units=sorted([u for u in d["units"] if u != a] + [b])))))
    if len(S) > 1:
        out.append(("templates equal", m(lambda d: d["strata"][1].update(template=d["strata"][0]["template"]))))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lean", required=True)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--random", type=int, default=40, help="random programs")
    ap.add_argument("--os-draws", type=int, default=400, help="OS draws of the small program, for the frequency check")
    ap.add_argument("--seed", type=int, default=20260927)
    a = ap.parse_args()
    from verity.ml.circuits.codec import canonical_json, encode_program
    from verity.core.protocols.sampled_proofs.one_stage import audit, draw as D, partition as P

    rng = random.Random(a.seed)
    out = a.out or Path(tempfile.mkdtemp(prefix="stratified-agree-"))
    out.mkdir(parents=True, exist_ok=True)
    T = _templates(6)
    cases = [("a4", list(zip(T, A4_SIZES)), [1024, 1, 6, 10_000]),
             ("split", [(T[0], 2), (T[1], 3), (T[0], 1)], [1, 2, 3, 4, 5, 6]),
             ("merged", [(T[1], 2), (T[1], 3), (T[0], 1)], [1, 3, 6]),
             ("single", [(T[2], 5)], [1, 3, 5])]
    for r in range(a.random):
        members = [(T[rng.randrange(4)], rng.randint(1, 40)) for _ in range(rng.randint(1, 6))]
        n = sum(c for _, c in members)
        cases.append((f"random{r}", members, sorted({rng.randint(1, n) for _ in range(3)})))
    bad, checked, mutations, refusals, tables_checked = [], 0, 0, 0, 0
    res: dict = {}
    for label, members, ks in cases:
        prog = P.mixed_population_program(members)
        tids = list(dict.fromkeys(P.template_id(t) for t, _ in members))
        q = P.template_instance_query(tids[0]) if len(tids) == 1 else P.template_instances_query(tids)
        desc, part = out / f"{label}.desc", out / f"{label}.partition.json"
        desc.write_bytes(canonical_json(encode_program(prog)))
        part.write_text(json.dumps(P.partition(prog, q)))
        n = P.units(prog, q)
        flags = ["--program", str(desc), "--partition", str(part)]
        for k in ks:
            law = D.derive(prog, q, D.parse_law(f"stratified:{k}"))
            need = sum(2 * s["k"] * width(sum(h - l for l, h in s["units"])) + 64 for s in law["strata"])
            src = os.urandom(need)
            (out / "stream.bin").write_bytes(src)
            p = subprocess.run([a.lean, "draw", *flags, "--stratified", str(k), "--stream-file", str(out / "stream.bin")],
                               capture_output=True, text=True)
            got = _lines(p, "DRAW")
            checked += 1
            if not got:
                bad.append((label, k, "no draw", p.stderr.strip()[-300:]))
                continue
            d = got[0]
            if D.law_of(d) != law or d["population"] != n:
                bad.append((label, k, "law or population"))
            ref = reference_draw(law, src)
            if ref is not None and d["units"] != ref:
                bad.append((label, k, "units against the reference sampler"))
            v = D.check_draw(d, {"population": n}, law)
            if not v.ok:
                bad.append((label, k, "check_draw refuses the stream draw", v.codes))
            p = subprocess.run([a.lean, "draw", *flags, "--stratified", str(k)], capture_output=True, text=True)
            got = _lines(p, "DRAW")
            if not got or not D.check_draw(got[0], {"population": n}, law).ok:
                bad.append((label, k, "the OS draw", p.stderr.strip()[-300:]))
            for tables in (2, 3):
                p = subprocess.run([a.lean, "draw", *flags, "--stratified", str(k), "--tables", str(tables),
                                    "--stream-file", str(out / "stream.bin")], capture_output=True, text=True)
                tables_checked += 1
                why = D.tables_problem(law, tables)
                got = _lines(p, "DRAW")
                if why is None and (not got or got[0] != d or not D.check_draw(got[0], {"population": n}, law, tables).ok):
                    bad.append((label, k, tables, "tables: Lean refuses or draws otherwise", p.stderr.strip()[-300:]))
                if why is not None and (got or why not in p.stderr):
                    bad.append((label, k, tables, "tables: Lean takes what core refuses", why, p.stderr.strip()[-300:]))
            if label == "a4" and k == 1024:
                core = audit.core_law(D.law_of(d), n)
                res["a4"] = {"k_s": [s["k"] for s in d["strata"]], "drawn": len(d["units"]),
                             "units": [s["units"] for s in d["strata"]],
                             "count_bound_2^-20": core.count_bound(2.0 ** -20),
                             "stratum_bound_2^-20": core.stratum_bound(d["strata"][0]["template"], 2.0 ** -20),
                             "escape_whole_template": float(core.escape({d["strata"][0]["template"]: A4_SIZES[0]}))}
            if label in ("a4", "split", "merged", "single") or label.startswith("random") and int(label[6:]) < 10:
                muts = _mutations(d, rng)
                f = out / f"{label}.{k}.cases.json"
                f.write_text(json.dumps([{"record": x, "statement": x, "instances": len(x.get("units", []))
                                          if isinstance(x.get("units"), list) else 0} for _, x in muts]))
                p = subprocess.run([a.lean, "draw-test", str(f), *flags, "--stratified", str(k)], capture_output=True, text=True)
                lean_v = _lines(p, "DRAWCHECK")
                for (why, x), lv in zip(muts, lean_v):
                    mutations += 1
                    try:
                        cv = D.check_draw(x, {"population": n}, law).ok
                    except Exception as e:
                        cv = False
                    if cv != lv["ok"]:
                        bad.append((label, k, "U2", why, {"core": cv, "lean": lv}))
                if len(lean_v) != len(muts):
                    bad.append((label, k, "draw-test lines", len(lean_v), len(muts)))
        for k in (0, n + 1):
            p = subprocess.run([a.lean, "draw", *flags, "--stratified", str(k)], capture_output=True, text=True)
            refusals += 1
            try:
                D.derive(prog, q, {"law": "stratified", "k": k})
                core_refuses = False
            except ValueError:
                core_refuses = True
            if p.returncode == 0 or not core_refuses:
                bad.append((label, k, "k out of range not refused"))
    # the frequency check: each unit's inclusion over OS draws against k_s / n_s
    prog = P.mixed_population_program([(T[0], 3), (T[1], 5), (T[0], 2)])
    q = P.template_instances_query([P.template_id(T[0]), P.template_id(T[1])])
    law = D.derive(prog, q, D.parse_law("stratified:4"))
    (out / "freq.desc").write_bytes(canonical_json(encode_program(prog)))
    (out / "freq.partition.json").write_text(json.dumps(P.partition(prog, q)))
    counts = [0] * 10
    for _ in range(a.os_draws):
        p = subprocess.run([a.lean, "draw", "--program", str(out / "freq.desc"), "--partition", str(out / "freq.partition.json"),
                            "--stratified", "4"], capture_output=True, text=True)
        for u in _lines(p, "DRAW")[0]["units"]:
            counts[u] += 1
    worst = 0.0
    for s in law["strata"]:
        n_s = sum(h - l for l, h in s["units"])
        pr = s["k"] / n_s
        sd = (a.os_draws * pr * (1 - pr)) ** 0.5 or 1.0
        for lo, hi in s["units"]:
            for u in range(lo, hi):
                worst = max(worst, abs(counts[u] - a.os_draws * pr) / sd)
    if worst > 5:
        bad.append(("frequency", counts, worst))
    # a query that defines no strata
    wq = {"format": "verity/partition/v1", "program": P.program_digest(prog), "query": {"name": "Q_word", "version": 1,
                                                                                      "params": {"X": 16, "W": 32}}}
    (out / "qword.partition.json").write_text(json.dumps(wq))
    p = subprocess.run([a.lean, "draw", "--program", str(out / "freq.desc"), "--partition", str(out / "qword.partition.json"),
                        "--stratified", "4"], capture_output=True, text=True)
    refusals += 1
    if p.returncode == 0 or "defines no strata" not in p.stderr:
        bad.append(("Q_word stratified not refused", p.stderr.strip()))
    # core's check_draw takes a stratum's `"k": true` for k_s = 1 (Python's True == 1); U2 refuses it
    a4 = json.loads((out / "a4.1024.cases.json").read_text())[0]["statement"]
    lax = json.loads(json.dumps(a4))
    lax["strata"][0]["k"] = True
    res["core_takes_a_boolean_stratum_k"] = D.check_draw(lax, {"population": a4["population"]}, D.law_of(a4)).ok
    res.update({"programs": len(cases), "draws": checked, "mutations": mutations, "refusals": refusals,
                "tables": tables_checked,
                "os_draws": a.os_draws, "frequency_worst_sd": round(worst, 2), "disagree": bad[:20]})
    print(json.dumps(res))
    return 0 if not bad else 1


if __name__ == "__main__":
    sys.exit(main())
