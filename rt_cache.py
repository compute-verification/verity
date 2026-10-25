"""REDTEAM probe -- Target 5 (#3945): the CacheStep_v2 eviction threshold hint `tau`.

Lead's questions:
  (1) Can a prover choose `tau` to pick WHICH entries get evicted?
  (2) Can a prover choose `tau` to hide an eviction from the charge?

`tau` is meant to be fully constrained by two counts: with e = max(new - hf, 0) > 0, exactly e queued copies have
stamp <= tau and fewer than e have stamp < tau, so `tau` is the e-th oldest queued stamp and the removed set is the
e oldest. This probe builds a pool directly and, for both the Python reference (`C2.reference`) and the Boolean circuit
(`C2.evaluate`), sweeps every `tau` to see which the unit accepts and what each removes.
"""
from __future__ import annotations

from types import SimpleNamespace

from verity.core.protocols.certifier import cache_step_v2 as C2

ST = C2.statics(8, 4, 3, 4, 24, 3, 2, 7, 6)   # the circuit-check small class: N8 U4 TW3 RW4 L24 R3 D2 NB7 SW6
S = SimpleNamespace(**ST)
ROOT = S.N


def pool_with_queued(stamps, held_slot0=True):
    """A flat trie (every entry at level 1, parent ROOT) with one queued copy per entry at the given stamps, plus one
    entry held by slot 0. hf = 0, so any allocation must evict."""
    table = []
    for i, st in enumerate(stamps):
        toks = tuple(((i + 1) >> (3 * k)) & 7 for k in range(S.U))  # distinct unit tokens, within TW=3 bits
        table.append(C2.Entry(parent=ROOT, tokens=toks, owner=i % (1 << S.RW), level=1, copies=(C2.Copy(0, st),)))
    held = []
    spfx = [ROOT] * S.R
    if held_slot0:
        table.append(C2.Entry(parent=ROOT, tokens=(7, 7, 7, 7), owner=0, level=1, copies=(C2.Copy(0b001, 0),)))
        spfx[0] = len(table) - 1
    table += [None] * (S.N - len(table))
    return C2.Pool(tuple(table), tuple(spfx), 0, max(stamps) + 1)


def removed_entries(before: C2.Pool, after: C2.Out):
    """Which entry indices lost all copies (became invalid) across the call."""
    if not after.ok:
        return None
    return [i for i in range(S.N) if before.table[i] is not None and after.pool.table[i] is None]


def run(pool, op):
    """(reference Out, circuit Out) for one (pool, op)."""
    ref = C2.reference(S, pool, op)
    cir = C2.evaluate(ST, [(pool, op)])[0]
    return ref, cir


def banner(t):
    print("\n" + "=" * 96 + f"\n{t}\n" + "=" * 96)


def sweep_tau(pool, slot, new, label=""):
    """For a compute of `new` blocks, sweep every tau; report which reference/circuit accept and what they remove."""
    e = max(new - pool.hf, 0)
    q = sorted(c.stamp for x in pool.table if x for c in x.copies if not c.mask)
    print(f"  {label}: hf={pool.hf} new={new} -> e={e}; queued stamps (oldest first) = {q}")
    acc_ref, acc_cir, divergent = [], [], []
    for tau in range(1 << S.SW):
        op = C2.Op(slot=slot, r=1, new=new, tau=tau)
        ref, cir = run(pool, op)
        if ref.ok != cir.ok:
            divergent.append((tau, ref.ok, cir.ok))
        if ref.ok:
            acc_ref.append((tau, tuple(removed_entries(pool, ref))))
        if cir.ok:
            acc_cir.append((tau, tuple(removed_entries(pool, cir))))
    print(f"    reference accepts tau in {[t for t, _ in acc_ref]}; removes {[(t, r) for t, r in acc_ref]}")
    print(f"    circuit   accepts tau in {[t for t, _ in acc_cir]}")
    print(f"    circuit==reference on every tau? {not divergent}" + (f"  DIVERGENCES {divergent}" if divergent else ""))
    want_tau = q[e - 1] if e else 0
    want_removed = tuple(i for i in range(S.N) if pool.table[i] and not pool.table[i].copies[0].mask
                         and pool.table[i].copies[0].stamp <= want_tau) if e else ()
    forced = acc_ref == [(want_tau, want_removed)]
    print(f"    expected: the e-th oldest stamp tau={want_tau} alone, removing the e oldest {want_removed}")
    print(f"    tau is FULLY FORCED (exactly one accepted, the e oldest)? {forced}")
    return forced, not divergent


def main():
    ok_all = True

    banner("(1) with distinct stamps, can tau pick which entries get evicted?")
    pool = pool_with_queued([5, 8, 11])     # entries 0,1,2 queued at 5,8,11; entry 3 held by slot 0
    for new in (1, 2, 3):
        forced, agree = sweep_tau(pool, slot=1, new=new, label=f"evict {new} oldest")
        ok_all &= forced and agree
    print("\n  => eviction is monotone in stamp (oldest first) and tau is the e-th oldest; the prover cannot keep an")
    print("     older queued block while evicting a newer one, nor choose any non-oldest set.")

    banner("(1b) a prover tries to EVICT THE NEWER (stamp 11) and KEEP THE OLDER (stamps 5,8)")
    # To remove only entry 2 (stamp 11) the prover would need a tau that removes {11} but not {5,8}. Impossible:
    # any tau >= 11 also has 5,8 <= tau. Show no (e, tau) achieves it.
    achievable = []
    for new in range(1, 5):
        e = max(new - pool.hf, 0)
        for tau in range(1 << S.SW):
            op = C2.Op(slot=1, r=1, new=new, tau=tau)
            ref = C2.reference(S, pool, op)
            if ref.ok and removed_entries(pool, ref) == [2]:
                achievable.append((new, tau))
    print(f"  (new, tau) that remove ONLY entry 2 (the newest queued): {achievable or 'NONE'}")
    ok_all &= not achievable

    banner("(2) hide an eviction: with hf=0, can an allocation take a block WITHOUT evicting?")
    # e = new - hf. With hf=0, any new>0 forces e=new evictions; e=0 (tau=0) is only valid when new<=hf.
    hid = []
    for new in range(1, 5):
        op0 = C2.Op(slot=1, r=1, new=new, tau=0)       # claim no eviction (e would be 0)
        ref = C2.reference(S, pool, op0)
        if ref.ok and removed_entries(pool, ref) == []:
            hid.append(new)
    print(f"  new>0 accepted with tau=0 and nothing evicted (hf=0): {hid or 'NONE'}")
    print("  => an allocation at an empty hashless queue must evict; e is forced by new and hf, not chosen.")
    ok_all &= not hid

    banner("(3) e past the queued copies is refused")
    r1 = C2.reference(S, pool, C2.Op(slot=1, r=1, new=4, tau=11))   # e=4 > 3 queued
    c1 = C2.evaluate(ST, [(pool, C2.Op(slot=1, r=1, new=4, tau=11))])[0]
    print(f"  e=4 (new=4, hf=0) with only 3 queued: reference ok={r1.ok} circuit ok={c1.ok} (expect both False)")
    ok_all &= (not r1.ok and not c1.ok)

    banner("(4) duplicate stamps (a dishonest pool) give no selective freedom")
    dup = pool_with_queued([8, 8, 11])      # entries 0,1 tied at stamp 8
    for e_new in (1, 2, 3):
        acc = []
        for tau in range(1 << S.SW):
            op = C2.Op(slot=1, r=1, new=e_new, tau=tau)
            ref = C2.reference(S, dup, op)
            if ref.ok:
                acc.append((tau, tuple(removed_entries(dup, ref))))
        print(f"  e={e_new}: accepted (tau, removed) = {acc}")
    print("  => removal is by stamp value (all copies with the same stamp go together); e that splits a tie has no")
    print("     valid tau and is refused -- never a way to evict one of a tied pair selectively.")

    banner("(5) the counts behind e: a free that overstates `held` hides later evictions (standalone unit)")
    # Slot 0 holds one copy (entry 3), so vLLM's free returns 1 block (its copy, now queued) and hf stays 0. A free
    # claiming held = 1 + k adds k phantom hashless blocks to hf (ret = 1 + k agrees), so the next k allocations
    # take hf instead of evicting. The unit bounds hf only by NB - 1; ReplayStep_v1 derives a request's blocks as
    # cdiv(comp, B), so the wiring #3945 lists ("each request's blocks to the frees' held") is what refuses this.
    copies = lambda p: sum(len(x.copies) for x in p.table if x)
    hidden = []
    for k in range(0, S.NB):
        fr = C2.Op(slot=0, free=True, held=1 + k, ret=1 + k)
        rf, cf = run(pool, fr)
        if not (rf.ok and cf.ok):
            print(f"  held=1+{k}: reference ok={rf.ok} circuit ok={cf.ok}  (refused: hf would pass NB-1={S.NB - 1})")
            continue
        p1 = rf.pool
        same = cf.pool == p1
        rc, cc = run(p1, C2.Op(slot=1, r=1, new=k, tau=0))
        gone = removed_entries(p1, rc) if rc.ok else None
        honest = C2.reference(S, rf.pool if k == 0 else run(pool, C2.Op(slot=0, free=True, held=1, ret=1))[0].pool,
                              C2.Op(slot=1, r=1, new=k, tau=0))
        tight = p1.hf + copies(p1) <= S.NB - 1
        print(f"  held=1+{k}: free ok (ref,cir)=({rf.ok},{cf.ok}) pool equal={same}; hf -> {p1.hf}; then new={k} with "
              f"tau=0: ref ok={rc.ok} cir ok={cc.ok} evicted={gone}; honest (held=1) same compute with tau=0 ok="
              f"{honest.ok}; hf + copies = {p1.hf + copies(p1)} <= NB-1={S.NB - 1}? {tight}")
        if k and rc.ok and cc.ok and gone == []:
            hidden.append((k, tight))
    print(f"  evictions hidden by an overstated held, (k, would a hf + copies <= NB-1 bound miss it): {hidden}")
    print("  => tau stays forced; what can move is e itself, through `new` and a free's `held`, which the unit takes as")
    print("     witnesses and only ReplayStep_v1's wiring binds (blocks per request = cdiv(comp, B)).")

    banner("(6) which binding closes (5): ReplayStep's `ret <= held(r)` and its free count are not enough")
    # Slot 0 holds X (level 1, shared with slot 1) and its child Y (level 2, only slot 0): comp = 8 tokens, so
    # ReplayStep's held(r) = cdiv(8, U) = 2. vLLM's free returns Y only: ret = 1. A free claiming held = 3 gives ret = 2,
    # which still passes ReplayStep's `ret <= held(r)`; and ReplayStep's free count (free += ret) rises by the same 1 as
    # the unit's hf + queued, so binding the free count to hf + queued doesn't see it either. Only held == held(r) does.
    B = S.U
    tabl = [C2.Entry(ROOT, (1, 1, 1, 1), 0, 1, (C2.Copy(0b011, 0),)),
            C2.Entry(0, (2, 2, 2, 2), 0, 2, (C2.Copy(0b001, 0),)),
            C2.Entry(ROOT, (3, 3, 3, 3), 1, 1, (C2.Copy(0, 5),))]
    shared = C2.Pool(tuple(tabl + [None] * (S.N - len(tabl))), (1, 0) + (ROOT,) * (S.R - 2), 0, 6)
    comp = 2 * B
    held_r = -(-comp // B)
    queued = lambda p: sum(1 for x in p.table if x for c in x.copies if not c.mask)
    rows = []
    for held in (2, 3):
        ret = 1 + held - 2
        rf, cf = run(shared, C2.Op(slot=0, free=True, held=held, ret=ret))
        p1 = rf.pool
        rc, cc = run(p1, C2.Op(slot=2, r=2, new=1, tau=0))
        rows.append((held, ret, rf.ok and cf.ok, ret <= held_r, p1.hf + queued(p1) - (shared.hf + queued(shared)),
                     rc.ok and cc.ok and removed_entries(p1, rc) == []))
    for held, ret, fok, rp, dfree, hid in rows:
        print(f"  held={held} ret={ret}: unit free ok={fok}; ReplayStep ret <= held(r)={held_r}? {rp}; unit hf+queued "
              f"moved by {dfree} (= ReplayStep's free += ret? {dfree == ret}); next new=1 needs no eviction? {hid}")
    leak6 = rows[1][2] and rows[1][3] and rows[1][4] == rows[1][1] and rows[1][5]
    print(f"  => with only those two bindings an inflated held (3 vs held(r) = {held_r}) still hides an eviction: {leak6}")
    print("     The wiring must bind the free's held to ReplayStep's cdiv(comp, B) by equality.")

    banner("SUMMARY")
    if ok_all:
        print("  CONFIRMATION: tau is fully constrained by (e, the queued stamps); the removed set is the e oldest,")
        print("  with no freedom to pick entries and no way to take a block without the forced eviction. Circuit ==")
        print("  reference on every swept tau.")
        print(f"  COMPOSITION DEPENDENCY: e's inputs (new, held) are free in the standalone unit; {len(hidden)} hidden-")
        print("  eviction amounts above are accepted by both reference and circuit until ReplayStep_v1 binds them.")
    else:
        print("  ATTENTION: a vector above did not behave as a full constraint -- see the sections.")


if __name__ == "__main__":
    main()
