module

public import Proofs.Properties.Nci.LeafWidths
public import Proofs.Properties.Nci.ResolveWithin
public import Mathlib.Tactic

/-!
# Statements only

The statements of the lemmas `LeafWide` waits on, each with no proof, to see that they elaborate. This file does not
land.
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.LeafWidths
open Flock.Program
open NciCheck.Properties.SecurityProofs.Resolve (Good RefOk BodyOk)

/-! ## The closed forms are sound -/

theorem wAt_mem_rangeWidths (t : Ty) (lo hi j : ℕ) (hlo : lo ≤ j) (hhi : j < hi) (hj : j < t.leaves) :
    Widths.wAt t j ∈ Widths.rangeWidths t lo hi := by
  sorry

theorem homo_wAt (t : Ty) (w : ℕ) (h : Widths.homo t = some w) (j : ℕ) (hj : j < t.leaves) :
    Widths.wAt t j = w := by
  sorry

theorem cover_spec (t : Ty) (lo hi : ℕ) (h : lo < hi) (hh : hi ≤ t.leaves) :
    (Widths.cover t lo hi).2 ≤ lo ∧ hi ≤ (Widths.cover t lo hi).2 + (Widths.cover t lo hi).1.leaves ∧
      (Widths.cover t lo hi).2 + (Widths.cover t lo hi).1.leaves ≤ t.leaves ∧
      ∀ j, (Widths.cover t lo hi).2 ≤ j → j < (Widths.cover t lo hi).2 + (Widths.cover t lo hi).1.leaves →
        Widths.wAt t j = Widths.wAt (Widths.cover t lo hi).1 (j - (Widths.cover t lo hi).2) := by
  sorry

theorem residues_mem (stride : ℤ) (count L cap : ℕ) (A : List ℕ) (hL : 0 < L)
    (h : Widths.residues stride count L cap = some A) (k : ℕ) (hk : k < count) :
    ((k : ℤ) * stride % (L : ℤ)).toNat ∈ A := by
  sorry

theorem eqRange_sound (fuel : ℕ) (t : Ty) (a : ℕ) (S : Ty) (b n w w' : ℕ) (ha : a + n ≤ t.leaves)
    (hb : b + n ≤ S.leaves) (h : (Widths.eqRange fuel t a S b n).run w = .ok true w') :
    ∀ k < n, Widths.wAt t (a + k) = Widths.wAt S (b + k) := by
  sorry

theorem partOk_sound (B : Body) (below : ℕ) (t : Ty) (a : ℕ) (q : Part) (w w' : ℕ) (hq : BodiesOk.PartNonneg q)
    (ha : a + q.count ≤ t.leaves) (hin : ∀ k < q.count, BodiesOk.RefIn B (q.get k))
    (h : (Widths.partOk (srcOf B below) t (Widths.homo t) a q).run w = .ok true w') :
    ∀ k < q.count, ∀ S, (srcOf B below).ty? (q.get k).p (q.get k).i = some S →
      Widths.wAt S (q.get k).j = Widths.wAt t (a + k) := by
  sorry

theorem seqBad_sound (B : Body) (below : ℕ) (t : Ty) (ps : Array Part) (w w' : ℕ)
    (hq : ∀ q ∈ ps, BodiesOk.PartNonneg q) (hc : (Seq.ofParts ps).count = t.leaves)
    (hin : ∀ j < (Seq.ofParts ps).count, BodiesOk.RefIn B ((Seq.ofParts ps).get j))
    (h : (Widths.seqBad (srcOf B below) t (Seq.ofParts ps)).run w = .ok none w') :
    SeqWide (srcOf B below) t (Seq.ofParts ps) := by
  sorry

/-- **A body the width check passes reads every leaf at its width.** -/
theorem bodyBad_sound (defs defs' : Array Defn) (i : ℕ) (key : String) (B : Body) (w w' : ℕ)
    (hB : BodiesOk.BodyFacts defs i B) (hret : B.ret.count = B.retType.leaves)
    (h : (Widths.bodyBad defs' key B).run w = .ok none w') : BodyWide B := by
  sorry

/-! ## The decoder ran the check -/

theorem decoded_checked (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧
        ∃ defs' key w w', (Widths.bodyBad defs' key (P.defs[d]!).body).run w = .ok none w' := by
  sorry

/-! ## A leaf's width is its type's -/

theorem wAt_widths (t : Ty) (j : ℕ) (hj : j < t.leaves) : Widths.wAt t j = (Ty.widths t)[j]! := by
  sorry

/-! ## Resolution keeps the width -/

/-- **Where resolution ends**: at a member of a node of a primitive, in an activation resolution reaches, whose
primitive returns one value as wide as the leaf the reference names. -/
theorem resolve_node (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (sc : Scope) (hsc : Good P sc) (r : Ref) (hr : RefOk P sc.d (P.defs[sc.d]!).body.nodes.size r) (g : ℕ)
    (h : resolve P sc r = .ok g) :
    ∃ sc' k m, Good P sc' ∧ k < (P.defs[sc'.d]!).body.nodes.size ∧
      (P.defs[((P.defs[sc'.d]!).body.nodes[k]!).fn]!).prim = true ∧
      m < ((P.defs[sc'.d]!).body.nodes[k]!).members ∧ g = sc'.off + (P.defs[sc'.d]!).body.offsets[k]! + m ∧
      (P.defs[((P.defs[sc'.d]!).body.nodes[k]!).fn]!).ret = .value (Widths.wAt (tyOf P sc r) r.j) := by
  sorry

/-- **The walk arrives where the layout says**: the gate of member `m` of a primitive's node `k`, in an activation
resolution reaches, is located at that node, with its callee for primitive. -/
theorem gate_of_node (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d) (sc : Scope)
    (hsc : Good P sc) (k m : ℕ) (hk : k < (P.defs[sc.d]!).body.nodes.size)
    (hp : (P.defs[((P.defs[sc.d]!).body.nodes[k]!).fn]!).prim = true)
    (hm : m < ((P.defs[sc.d]!).body.nodes[k]!).members) (p : ℕ) (ops : Array ℕ) (sc' : Scope)
    (h : P.gate (sc.off + (P.defs[sc.d]!).body.offsets[k]! + m) = .ok (p, ops, sc')) :
    p = ((P.defs[sc.d]!).body.nodes[k]!).fn ∧ sc' = sc := by
  sorry

/-- **A gate a reference resolves to is as wide as the leaf it names**, wherever the gate is located. -/
theorem resolve_wide (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
    (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
    (hD : ∀ d < P.defs.size, BodiesOk.DefOk P.defs d P.defs[d]!)
    (hW : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
      (P.defs[d]!).body.retType = (P.defs[d]!).ret ∧ BodyWide (P.defs[d]!).body)
    (sc : Scope) (hsc : Good P sc) (r : Ref) (hr : RefOk P sc.d (P.defs[sc.d]!).body.nodes.size r) (g : ℕ)
    (h : resolve P sc r = .ok g) (p : ℕ) (ops : Array ℕ) (sc' : Scope) (hg : P.gate g = .ok (p, ops, sc')) :
    (P.defs[p]!).ret = .value (Widths.wAt (tyOf P sc r) r.j) := by
  obtain ⟨sc₁, k, m, hsc₁, hk, hp, hm, rfl, hw⟩ := resolve_node P hroot hB hL hD hW sc hsc r hr g h
  rw [(gate_of_node P hroot hB hL sc₁ hsc₁ k m hk hp hm p ops sc' hg).1]
  exact hw

/-! ## A Definition has its own key -/

/-- **The decoder's index is its Definitions' keys**: an indexed key is the key of the Definition at its index, and
every Definition's key is indexed at the Definition. So no two Definitions have one key. -/
theorem decode_keys (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
    (∀ k v, P.index[k]? = some v → v < P.defs.size ∧ (P.defs[v]!).key = k) ∧
      ∀ i < P.defs.size, P.index[(P.defs[i]!).key]? = some i := by
  sorry

end NciCheck.Properties.SecurityProofs.LeafWidths
