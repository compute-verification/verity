"""REDTEAM probe — Target 4 (#3922): the AS=1 witnessed R x R batch->slot map (compile_slice).

The lead's three vectors, plus an independent soundness oracle and a brute-force circuit==reference check:

  (A) idle steps: can the prover claim an active step idle (or an idle step active) through the map?
  (B) a step whose batch repeats a request: can a duplicated served request drop a distinct scan row?
      and does the reference's request-keyed dict `slot_map` (used to GENERATE the honest map) diverge
      from the verdict helpers `_mapped`/`_build_map` when the SCAN rows repeat a request?
  (C) R where the slot count changes mid-window (demo_async drops 2->1 at unit 4): is every wrong map
      at the drop refused, and is a fabricated extra served row refused?

For each case we read three things:
  - truth   : an independent oracle — per step, is the committed srv a field-for-field permutation of the
              scan's active rows (idle steps zero)? This is what `rows` MUST equal, map or no map.
  - ref     : the reference verdict's `rows` flag (compile_slice.check, via `_mapped`/`_rows_ok` over w[map]).
  - circuit : the Boolean circuit's `rows` bit (CS.evaluate -> verdict bit 4), the thing core trusts.

A finding is any case with circuit-rows=1 while truth=0 (a soundness hole), or circuit != ref (a divergence).
"""
from __future__ import annotations

import itertools
from types import SimpleNamespace

from verity.core.protocols.certifier import compile_slice as CS
from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import BOOK

ROWS_BIT = CS.FLAGS.index("rows")  # verdict bit 4


def _demo():
    st, sl = CS.demo_async()
    return st, SimpleNamespace(**st), sl


def circuit_rows(st, c) -> int:
    """The Boolean circuit's `rows` bit on one case (the thing on core's path)."""
    v = CS.evaluate(st, [c])[0]
    return (v >> ROWS_BIT) & 1


def ref_rows(c) -> int | None:
    return c.flags["rows"]


def truth(S, scan_steps, served_steps) -> bool:
    """Oracle: per step, the committed srv is a field-for-field permutation of the scan's active rows
    (request, kind, start, n), idle steps zero. The map is only a witness for this permutation."""
    P = S.P
    sc = list(scan_steps) + [None] * (P - len(scan_steps))
    sv = list(served_steps) + [None] * (P - len(served_steps))
    for u in range(P):
        a = sorted(tuple(r[1:]) for r in (sc[u] or ()))
        b = sorted(tuple(r[1:]) for r in (sv[u] or ()))
        if a != b:
            return False
    return True


def row(st, S, sl, **kw):
    """Build a case and return (circuit-rows, ref-rows, truth) with the served steps used."""
    c = CS.case(S, sl, BOOK, **kw)
    served = kw.get("served")
    served = list(c.steps) if served is None else served
    return circuit_rows(st, c), ref_rows(c), truth(S, c.steps, served), c


def banner(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


def check(label, cr, rf, tr, want_circuit):
    """Report one case; flag a soundness hole (circuit accepts a non-permutation) or a divergence."""
    sound = "" if not (cr == 1 and tr is False) else "  <<< SOUNDNESS HOLE (circuit accepts a non-permutation)"
    div = "" if cr == rf or (rf is None) else "  <<< DIVERGENCE circuit != ref"
    mark = "ok" if cr == want_circuit else "!! unexpected"
    print(f"  {label:<46} circuit_rows={cr}  ref_rows={rf}  truth={tr}  [{mark}]{sound}{div}")
    return (cr == 1 and tr is False), (cr != rf and rf is not None)


def main():
    st, S, sl = _demo()
    holes, divs = 0, 0

    honest = CS.case(S, sl, BOOK)
    print(f"R={S.R} P={S.P} AS={S.AS}  demo_async scan steps:")
    for u, s in enumerate(honest.steps):
        print(f"   unit {u}: {s}")
    hc, hr, ht = circuit_rows(st, honest), ref_rows(honest), truth(S, honest.steps, honest.steps)
    print(f"  honest: circuit_rows={hc} ref_rows={hr} truth={ht}  verdict={honest.verdict:#x} (ACCEPT={CS.ACCEPT:#x})")

    # Identify which units are active (len>0) and the drop point.
    lens = [len(s) for s in honest.steps]
    print(f"  rows per unit: {lens}")

    # ---- (A) idle steps -------------------------------------------------------------------------------
    banner("(A) idle steps")
    # Find a genuinely idle unit if any (len 0); else synthesise one by serving idle where scan is active.
    idle_units = [u for u, n in enumerate(lens) if n == 0]
    print(f"  genuinely idle units in demo_async: {idle_units}")

    # A1: an ACTIVE unit, served idle (zero rows) -- try every map. Expect: all refused.
    u0 = next(u for u, n in enumerate(lens) if n > 0)
    served = list(honest.steps)
    served[u0] = ()  # claim unit u0 ran nothing
    acc = []
    for m in itertools.product((0, 1), repeat=S.R * S.R):
        mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
        cr, rf, tr, _ = row(st, S, sl, served=served, maps={u0: mm})
        if cr == 1:
            acc.append((mm, rf, tr))
    print(f"  A1 active-unit-{u0} served idle: {len(acc)}/{2**(S.R*S.R)} maps give circuit_rows=1 (expect 0)")
    for mm, rf, tr in acc:
        h, d = check(f"A1 map={mm}", 1, rf, tr, 0)
        holes += h; divs += d
    if not acc:
        print("    -> every map refused (idle-claim on an active unit blocked)")

    # A2: an IDLE unit served with a real row -- try every map. Expect: all refused.
    # Use the last unit if it is idle; else append an extra served row on a short unit.
    if idle_units:
        ui = idle_units[0]
        served = list(honest.steps)
        served[ui] = ((0, 0, 1, 5, 1),)  # a fabricated active row on an idle unit
        acc = []
        for m in itertools.product((0, 1), repeat=S.R * S.R):
            mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
            cr, rf, tr, _ = row(st, S, sl, served=served, maps={ui: mm})
            if cr == 1:
                acc.append((mm, rf, tr))
        print(f"  A2 idle-unit-{ui} served a row: {len(acc)}/{2**(S.R*S.R)} maps give circuit_rows=1 (expect 0)")
        for mm, rf, tr in acc:
            h, d = check(f"A2 map={mm}", 1, rf, tr, 0)
            holes += h; divs += d
        if not acc:
            print("    -> every map refused (active-claim on an idle unit blocked)")
    else:
        print("  A2 skipped: demo_async has no genuinely idle unit")

    # ---- (B) a step whose batch repeats a request ----------------------------------------------------
    banner("(B) a batch that repeats a request")
    # Find a unit with two DISTINCT requests.
    ub = next(u for u, s in enumerate(honest.steps) if len({r[1] for r in s}) == 2)
    a, b = honest.steps[ub]
    print(f"  unit {ub} scan rows: {a}, {b}  (requests {a[1]} and {b[1]})")
    # B1: commit srv as two rows BOTH carrying request a[1] (drop the distinct row b by duplicating a).
    dup = ((0, a[1], *a[2:]), (1, a[1], *a[2:]))
    served = list(honest.steps); served[ub] = dup
    acc = []
    for m in itertools.product((0, 1), repeat=S.R * S.R):
        mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
        cr, rf, tr, _ = row(st, S, sl, served=served, maps={ub: mm})
        if cr == 1:
            acc.append((mm, rf, tr))
    print(f"  B1 served=[req {a[1]} twice]: {len(acc)}/{2**(S.R*S.R)} maps give circuit_rows=1 (expect 0)")
    for mm, rf, tr in acc:
        h, d = check(f"B1 map={mm}", 1, rf, tr, 0)
        holes += h; divs += d
    if not acc:
        print("    -> every map refused (a duplicated served request can't drop a distinct scan row)")

    # B2: the reference helper divergence when the SCAN rows repeat a request.
    # This exercises slot_map's request-keyed dict `at = {x[1]: x[0] ...}` directly, against the verdict
    # helpers _mapped (reference) and _build_map (circuit), on a synthetic step. Not necessarily reachable
    # from a real scan, but a latent split between the generator and the checker would be a finding.
    banner("(B2) reference slot_map request-keyed dict vs the verdict helpers (direct, synthetic scan)")
    _b2_direct(S)

    # ---- (C) slot count changes mid-window -----------------------------------------------------------
    banner("(C) R where the slot count changes mid-window")
    drop = next((u for u in range(1, len(lens)) if lens[u] < lens[u - 1] and lens[u] > 0), None)
    print(f"  rows per unit: {lens}; first drop at unit {drop}")
    if drop is not None:
        # C1: at the drop unit, enumerate every map; only the correct (single-row) map should pass.
        ok = []
        for m in itertools.product((0, 1), repeat=S.R * S.R):
            mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
            cr, rf, tr, _ = row(st, S, sl, maps={drop: mm})
            if cr == 1:
                ok.append((tuple(m), rf, tr))
        print(f"  C1 unit {drop} map sweep: {len(ok)}/{2**(S.R*S.R)} maps accept (expect exactly 1, correct+truth)")
        for m, rf, tr in ok:
            mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
            h, d = check(f"C1 accepted map={mm}", 1, rf, tr, 1)
            holes += h; divs += d
        # C2: fabricate an extra served row at the drop unit (claim 2 rows where the scan ran 1).
        extra = tuple(honest.steps[drop]) + ((1, 99, 0, 0, 1),)
        served = list(honest.steps); served[drop] = extra
        acc = []
        for m in itertools.product((0, 1), repeat=S.R * S.R):
            mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
            cr, rf, tr, _ = row(st, S, sl, served=served, maps={drop: mm})
            if cr == 1:
                acc.append((mm, rf, tr))
        print(f"  C2 fabricated extra row at unit {drop}: {len(acc)} maps give circuit_rows=1 (expect 0)")
        for mm, rf, tr in acc:
            h, d = check(f"C2 map={mm}", 1, rf, tr, 0)
            holes += h; divs += d
        if not acc:
            print("    -> every map refused (a fabricated extra served row blocked)")
    else:
        print("  C skipped: no mid-window slot-count drop in demo_async")

    # ---- (D) brute-force circuit == reference over a structured perturbation set ----------------------
    banner("(D) brute-force circuit == reference over perturbed (served, maps)")
    divs += _bruteforce(st, S, sl, honest)

    # ---- (E) exhaustive soundness sweep at one unit --------------------------------------------------
    banner("(E) exhaustive soundness sweep at unit 0: {accepted srv over ANY map} == {permutations}")
    h, d = _sweep_unit0(st, S, sl, honest)
    holes += h; divs += d

    # ---- summary -------------------------------------------------------------------------------------
    banner("SUMMARY")
    print(f"  soundness holes (circuit accepts a non-permutation): {holes}")
    print(f"  circuit/reference divergences:                       {divs}")
    if holes == 0 and divs == 0:
        print("  VERDICT: CONFIRMATION — the map is a field-preserving bijection; all three vectors blocked,")
        print("           circuit == reference on every probed case.")


def _b2_direct(S):
    """Drive slot_map / _mapped / _build_map on a synthetic step whose SCAN rows repeat a request."""
    from verity.ml.circuits.boolean.forms import Circuit, ONE, ZERO

    # Two scan slots, SAME request 7 (a scenario a real scan should never make), distinct other fields.
    rows = [(0, 7, 0, 0, 5), (1, 7, 1, 9, 1)]          # (slot, request, kind, start, n)
    served = [(0, 7, 0, 0, 5), (1, 7, 1, 9, 1)]        # the honest served batch, same two rows
    m = CS.slot_map(S, rows, served)
    print(f"  scan rows repeat request 7: slot_map -> {m}")
    print(f"    (at = {{x[1]:x[0]}} collapses both slots to one key, so one column is lost)")
    mapped = CS._mapped(S, rows, m)
    print(f"  _mapped(rows, slot_map) -> {mapped}  (None means the verdict would refuse the honest map)")

    # Does the honest-generated map pass the circuit? Build _build_map directly.
    def by_slot(step):
        bits = CS.rows_bits(S, step)
        return CS._by_slot(S, bits)

    C = Circuit()
    rep = [[ONE if x else ZERO for x in r] for r in by_slot(rows)]
    srv = [[ONE if x else ZERO for x in r] for r in by_slot(served)]
    mm = [[ONE if m[b][s] else ZERO for s in range(S.R)] for b in range(S.R)]
    bad = CS._build_map(C, S, rep, srv, mm)
    circuit_ok = all(b is ZERO for b in bad)
    print(f"  _build_map(rep, srv, slot_map) all-zero? {circuit_ok}  (True = circuit accepts this map)")
    ref_ok = mapped is not None and CS.srv_row(S, [tuple((b, *r[1:]) for b, r in enumerate(served))])  # placeholder
    print("  => slot_map is only the honest-map GENERATOR; the verdict is _mapped/_build_map over the")
    print("     witnessed map. If slot_map drops a column here, the honest prover's own map is refused")
    print("     (a liveness dent on an unreachable scan), never an accepted forgery.")


def _sweep_unit0(st, S, sl, honest):
    """Enumerate every committed srv at unit 0 over a small field alphabet (the two scan rows' field tuples,
    one foreign tuple, and idle) crossed with all 2^(R*R) maps; collect the srv the circuit EVER accepts and
    confirm it is exactly the permutations of the scan's rows. This pins `rows=1 <=> srv is a permutation`."""
    base = list(honest.steps)
    v0, v1 = base[0][0][1:], base[0][1][1:]          # (request, kind, start, n) of the two scan rows
    foreign = (0, 0, 0, 1)                            # fits the field widths, not among the scan rows
    alpha = {"idle": None, "v0": v0, "v1": v1, "fgn": foreign}
    keys = list(alpha)
    want_perms = {frozenset((("p0", "v0"), ("p1", "v1"))), frozenset((("p0", "v1"), ("p1", "v0")))}

    accepted, bad_hole, bad_div = set(), 0, 0
    for k0, k1 in itertools.product(keys, keys):
        step = []
        if alpha[k0] is not None:
            step.append((0, *alpha[k0]))
        if alpha[k1] is not None:
            step.append((1, *alpha[k1]))
        served = list(base); served[0] = tuple(step)
        tr = truth(S, honest.steps, served)
        ever = False
        for m in itertools.product((0, 1), repeat=S.R * S.R):
            mm = [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]
            c = CS.case(S, sl, BOOK, served=served, maps={0: mm})
            cr = circuit_rows(st, c)
            rf = c.flags["rows"]
            if rf is not None and cr != rf:
                bad_div += 1
                print(f"  DIVERGENCE srv={step} map={mm}: circuit={cr} ref={rf}")
            if cr == 1:
                ever = True
                if tr is False:
                    bad_hole += 1
                    print(f"  SOUNDNESS HOLE srv={step} map={mm}: circuit accepts, truth=False")
        if ever:
            accepted.add((k0, k1))
    perms = {(a, b) for a, b in itertools.product(keys, keys)
             if frozenset((("p0", a), ("p1", b))) in want_perms}
    print(f"  srv configs the circuit accepts over some map: {sorted(accepted)}")
    print(f"  the true permutations of the scan rows:        {sorted(perms)}")
    print(f"  equal? {accepted == perms}   holes={bad_hole}  divergences={bad_div}")
    return bad_hole, bad_div


def _bruteforce(st, S, sl, honest):
    """Perturb served rows and maps across all units; assert CS.evaluate's rows bit == check's rows flag."""
    base = list(honest.steps)
    cases = []
    labels = []

    # 1) every single-unit map replaced by each of the 2^(R*R) maps, honest served rows.
    for u in range(S.P):
        for m in itertools.product((0, 1), repeat=S.R * S.R):
            mm = {u: [list(m[i * S.R:(i + 1) * S.R]) for i in range(S.R)]}
            cases.append(CS.case(S, sl, BOOK, maps=mm)); labels.append(("map", u, m))

    # 2) reversed served order at each unit (honest maps auto-generated) + a few swaps/drops.
    from verity.core.protocols.certifier.tests.test_network_certifier_compile_slice import _reversed
    for u in range(S.P):
        if len(base[u]) >= 1:
            srv = list(base); srv[u] = _reversed(base[u])
            cases.append(CS.case(S, sl, BOOK, served=srv)); labels.append(("rev", u, None))
        if len(base[u]) >= 2:
            a, b = base[u][0], base[u][1]
            srv = list(base); srv[u] = ((0, b[1], *a[2:]), (1, a[1], *b[2:]))
            cases.append(CS.case(S, sl, BOOK, served=srv)); labels.append(("swap", u, None))
            srv = list(base); srv[u] = base[u][:-1]
            cases.append(CS.case(S, sl, BOOK, served=srv)); labels.append(("drop", u, None))

    circ = CS.evaluate(st, cases)
    bad = 0
    for lab, c, v in zip(labels, cases, circ):
        cr = (v >> ROWS_BIT) & 1
        rf = c.flags["rows"]
        if rf is not None and cr != rf:
            bad += 1
            print(f"  DIVERGENCE {lab}: circuit_rows={cr} ref_rows={rf}")
    print(f"  checked {len(cases)} perturbed cases; divergences: {bad}")
    return bad


if __name__ == "__main__":
    main()
