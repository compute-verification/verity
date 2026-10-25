"""REDTEAM phase 19 target 2: #4000 (ADV-SPEC @406c08a9f6), the compile slice's served rows in blocks of G < P steps.

CompileSliceBlocks_v1{…, G}: ``srv`` is P / G rows (`served_rows.block_bits`), step u is checked against row u // G, and
the verifier holds row j of window w to leaf w·(P / G) + j (`served_rows.leaf`). Probed: steps at block boundaries, a G
that doesn't divide P, leaf reuse across windows, and two blocks swapped (in the slice's rows and in the registration),
each judged by the reference and, at the demonstration classes, by the circuit; the statics tie
(`ServedStaticsTie_v1{…, G}`) judges the registration's swapped blocks too."""
from __future__ import annotations

import itertools
import sys
from dataclasses import replace
from types import SimpleNamespace as N

sys.path.insert(0, "verity/core/protocols/certifier/tests")

import test_network_certifier_compile_slice as TCS  # noqa: E402

from verity.core.protocols.certifier import compile_slice as CS, served_rows as SR, served_statics as SS  # noqa: E402

BOOK = TCS.BOOK
ROOT = bytes(range(64))
ROWS_FLAG = "rows"


def divisors(P):
    return [g for g in range(1, P + 1) if P % g == 0]


def at(name):
    st, sl = getattr(CS, name)()
    return N(name=name, st=st, S=N(**st), sl=sl)


def refused(c):
    return {f for f, ok in c.flags.items() if not ok}


# ------------------------------------------------------------------------------------------------------------------
# (A) steps at block boundaries
# ------------------------------------------------------------------------------------------------------------------
print("== (A) steps at block boundaries ==")
for x in (at("demo"), at("demo_async")):
    P = x.S.P
    for G in divisors(P):
        st = CS.with_block(x.st, G)
        S = N(**st)
        honest = CS.case(S, x.sl, BOOK)
        assert honest.verdict == CS.ACCEPT, (x.name, G)
        steps = list(honest.steps)
        U = P // G
        bounds = sorted({0, P - 1} | {b for j in range(1, U) for b in (j * G - 1, j * G)})
        cases, labels, unpackable = [honest], ["honest"], 0
        for u in bounds:
            if steps[u]:
                for k, row in enumerate(steps[u]):
                    for f in range(1, 5):
                        bad = list(row)
                        bad[f] ^= 1
                        served = steps[:u] + [steps[u][:k] + (tuple(bad),) + steps[u][k + 1:]] + steps[u + 1:]
                        try:
                            cases.append(CS.case(S, x.sl, BOOK, served=served))
                        except ValueError:
                            unpackable += 1
                            continue
                        labels.append(f"step {u} slot {k} field {f}")
                served = steps[:u] + [None] + steps[u + 1:]
                cases.append(CS.case(S, x.sl, BOOK, served=served))
                labels.append(f"step {u} emptied")
            else:
                donor = next(s for s in steps if s)
                served = steps[:u] + [donor] + steps[u + 1:]
                cases.append(CS.case(S, x.sl, BOOK, served=served))
                labels.append(f"idle step {u} given a row")
        for j in range(1, U):
            a, b = j * G - 1, j * G
            if steps[a] != steps[b]:
                served = list(steps)
                served[a], served[b] = served[b], served[a]
                cases.append(CS.case(S, x.sl, BOOK, served=served))
                labels.append(f"steps {a} and {b} swapped across the boundary")
            served = steps[:a] + [steps[b]] + steps[b:]
            served = served[:P]
            cases.append(CS.case(S, x.sl, BOOK, served=served))
            labels.append(f"a shift by one step at the boundary {b}")
        circ = CS.evaluate(st, cases)
        ref = [c.verdict for c in cases]
        bad_acc = [lab for lab, c in zip(labels[1:], cases[1:]) if ROWS_FLAG not in refused(c)]
        print(f"{x.name} P={P} G={G} ({'CompileSliceBlocks_v1' if G < P else 'CompileSlice_v1'}): boundary steps "
              f"{bounds}; {len(cases) - 1} tampers ({unpackable} unpackable), refused at rows: "
              f"{len(cases) - 1 - len(bad_acc)}; circuit == reference: {circ == ref}")
        assert circ == ref and not bad_acc, bad_acc

# ------------------------------------------------------------------------------------------------------------------
# (B) a G that doesn't divide P
# ------------------------------------------------------------------------------------------------------------------
print("\n== (B) a G that doesn't divide P, at each entry point ==")


def outcome(f):
    try:
        return f"ok {f()!r}"
    except Exception as e:  # noqa: BLE001
        return f"{type(e).__name__}"


x = at("demo_async")
P = x.S.P
window = list(CS.case(x.S, x.sl, BOOK).steps)
for g in [0, -2, 4, 5, 7, 12, P, True]:
    with_block = outcome(lambda: CS.with_block(x.st, g).get("G"))
    unit = outcome(lambda: CS.unit(**x.st, G=g).id[-6:])
    reg = outcome(lambda: SR.register(x.S, [window], ROOT, G=g).G)
    leaf = outcome(lambda: [SR.leaf(x.S, {"G": g}, w, 0) for w in range(3)])
    tie = outcome(lambda: SS.widths(N(**SS.statics(x.S, G=g))).SB)
    print(f"P={P} G={g!r:>5}: with_block {with_block:<14} unit {unit:<16} register {reg:<14} "
          f"served_rows.leaf(w=0,1,2; j=0) {leaf:<16} ServedStaticsTie statics {tie}")
print("  G = True:", outcome(lambda: CS.unit(**CS.with_block(x.st, True)).id[-8:]), "vs G = 1:",
      outcome(lambda: CS.unit(**CS.with_block(x.st, 1)).id[-8:]))

# ------------------------------------------------------------------------------------------------------------------
# (C) leaf reuse across windows
# ------------------------------------------------------------------------------------------------------------------
print("\n== (C) leaf reuse across windows ==")
x = at("demo_async")
P = x.S.P
honest = CS.case(x.S, x.sl, BOOK)
w0 = list(honest.steps)
session = [w0, w0[::-1], [None] * P, w0[2:] + w0[:2]]
for G in divisors(P):
    served = SR.register(x.S, session, ROOT, G=G)
    rec, reg = served.record(), served.registration
    U = P // G
    idx = [SR.leaf(x.S, rec, w, j) for w in range(len(session)) for j in range(U)]
    assert sorted(idx) == list(range(rec["count"])), (G, idx)
    cross = sum(SR.leaf_ok(rec, ROOT, k2, served.bc(k1), reg.open(k1).path) for k1 in idx for k2 in idx if k1 != k2)
    same_rows = sum(reg.rows[k1] == reg.rows[k2] for k1 in idx for k2 in idx if k1 != k2)
    own = all(SR.leaf_ok(rec, ROOT, k, served.bc(k), reg.open(k).path) for k in idx)
    print(f"G={G}: {len(session)} windows x {U} blocks -> leaves {idx[:8]}{'…' if len(idx) > 8 else ''}, injective "
          f"and onto the {rec['count']} leaves; each b||c opens its own leaf: {own}, another's: {cross} of "
          f"{len(idx) * (len(idx) - 1)} ({same_rows // 2} pairs of equal rows, held apart by their salts)")
    assert own and cross == 0
# the record's G against the unit's: a slice at G = 2 (3 rows) indexed by a record that says G = 1
st2 = CS.with_block(x.st, 2)
S2 = N(**st2)
rec1 = SR.register(x.S, session, ROOT, G=1).record()
print(f"a record of G = 1 for a slice at G = 2: unit w's srv rows at leaves "
      f"{[[SR.leaf(S2, rec1, w, j) for j in range(3)] for w in range(2)]} (the record's w·6 + j, not the slice's "
      f"w·3 + j); widths {CS.row_bits(S2)['srv0']} vs the record's {rec1['n_bits']} bits, so no b||c of one opens a leaf of "
      f"the other")
s2 = SR.register(x.S, session, ROOT, G=2)
c2 = CS.case(S2, x.sl, BOOK)
print("  the slice's honest rows under that record:",
      [SR.leaf_ok(rec1, ROOT, SR.leaf(S2, rec1, 0, j), s2.bc(j), s2.registration.open(j).path) for j in range(3)])

# ------------------------------------------------------------------------------------------------------------------
# (D) two blocks swapped
# ------------------------------------------------------------------------------------------------------------------
print("\n== (D) two blocks swapped ==")
for x in (at("demo"), at("demo_async")):
    P = x.S.P
    for G in [g for g in divisors(P) if g < P]:
        st = CS.with_block(x.st, G)
        S = N(**st)
        U = P // G
        Sst = N(**SS.statics(x.S, G=G))
        assert SS.widths(Sst).SB == CS.step_row_bits(x.S)
        honest = CS.case(S, x.sl, BOOK)
        steps = list(honest.steps)
        progs = SS.program_of(steps + [None] * (P - len(steps)))
        served = SR.register(x.S, [steps], ROOT, G=G)
        rec, reg = served.record(), served.registration
        names = CS.srv_names(S)
        proof_cases, reg_cases, tie_cases, leaf_acc, pairs, equal = [], [], [], 0, 0, 0
        for j, j2 in itertools.combinations(range(U), 2):
            pairs += 1
            if steps[j * G:(j + 1) * G] == steps[j2 * G:(j2 + 1) * G]:
                equal += 1
                continue
            # (D1) the proof's srv{j} and srv{j2} swapped: each b||c is the other block's registered leaf
            leaf_acc += SR.leaf_ok(rec, ROOT, SR.leaf(S, rec, 0, j), served.bc(SR.leaf(S, rec, 0, j2)),
                                   reg.open(SR.leaf(S, rec, 0, j2)).path)
            rows = dict(honest.rows)
            rows[names[j]], rows[names[j2]] = rows[names[j2]], rows[names[j]]
            flags = CS.check(S, rows)
            proof_cases.append(replace(honest, rows=rows, verdict=CS.verdict(flags), flags=flags))
            # (D2) the registration's blocks swapped: the slice reads them in its rows, the tie against the Programs
            sw = list(steps)
            sw[j * G:(j + 1) * G], sw[j2 * G:(j2 + 1) * G] = steps[j2 * G:(j2 + 1) * G], steps[j * G:(j + 1) * G]
            reg_cases.append(CS.case(S, x.sl, BOOK, served=sw))
            for b in (j, j2):
                tie_cases.append(SS.case(Sst, SR.block_bits(x.S, sw, G, b), progs[b * G:(b + 1) * G]))
        circ = CS.evaluate(st, proof_cases + reg_cases) if proof_cases else []
        ref = [c.verdict for c in proof_cases + reg_cases]
        tie_circ = SS.evaluate(SS.statics(x.S, G=G), tie_cases) if tie_cases else []
        slice_acc = sum(ROWS_FLAG not in refused(c) for c in proof_cases + reg_cases)
        tie_acc = sum(c.verdict == SS.ACCEPT for c in tie_cases)
        print(f"{x.name} P={P} G={G}: {pairs} block pairs ({equal} equal, skipped); the proof's rows swapped: leaf_ok "
              f"accepts {leaf_acc}, the slice accepts at rows {sum(ROWS_FLAG not in refused(c) for c in proof_cases)}; "
              f"the registration's swapped: slice accepts {sum(ROWS_FLAG not in refused(c) for c in reg_cases)}, "
              f"statics tie accepts {tie_acc} of {len(tie_cases)} blocks; circuit == reference: slice "
              f"{circ == ref}, tie {tie_circ == [c.verdict for c in tie_cases]}")
        assert leaf_acc == 0 and slice_acc == 0 and circ == ref and tie_circ == [c.verdict for c in tie_cases]

print("\n== the cost ==")
for x in (at("demo"), at("demo_async")):
    print(x.name, {G: CS.size(**CS.with_block(x.st, G))["ands"] for G in divisors(x.S.P)})
print("\nall assertions hold")
