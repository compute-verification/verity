import Proofs.Flock.Soundness.TmplUnit
import Proofs.Flock.Soundness.TmplUnrollAt

/-!
# The templated unit's share is the flat slot's

`Stmt.partialRangesT` folds a range whose net is a read template's unrolled net (`Stmt.unitShare`) with `Tmpl.foldUnit`
instead of projecting the slot's `2^sl` base. Its entries are the projection's (`unitShare_get`, by `foldUnit_eq_project`
on the template's own facts, `TmplUnit`), and the slot-type cache only spares work, so `partialRangesT` is `partialRanges`
on any two caches of `slotBase`'s values (`partialRangesT_eq`).
-/

open Flock Flock.Tmpl FlockLevel3 GF128

namespace FlockSoundness.TmplFold

/-- A slot-type cache of `slotBase`'s values. -/
def CacheOk (st : Stmt) (alpha : F128) (lw rho : Array F128) (b : Array ((Kind × ℕ) × Array F128)) : Prop :=
  ∀ kb ∈ b, st.slotBase alpha lw rho kb.1 = .ok kb.2

theorem cacheOk_push {st : Stmt} {alpha : F128} {lw rho : Array F128} {b : Array ((Kind × ℕ) × Array F128)}
    {key : Kind × ℕ} {x : Array F128} (hb : CacheOk st alpha lw rho b) (hx : st.slotBase alpha lw rho key = .ok x) :
    CacheOk st alpha lw rho (b.push (key, x)) := by
  intro kb hkb
  rcases Array.mem_push.mp hkb with h1 | h1
  · exact hb kb h1
  · rw [h1]; exact hx

theorem slotBase_net (st : Stmt) (alpha : F128) (lw rho : Array F128) {i sl : ℕ}
    (h : (st.c.nets.getD i default).2.unitLog = sl) :
    st.slotBase alpha lw rho (.net i, sl) = .ok (Stmt.baseOf alpha (Stmt.tensor lw (rho.extract 0 (sl - 6)))
      (st.c.nets.getD i default).2.a (st.c.nets.getD i default).2.b (st.c.nets.getD i default).2.lookup sl) := by
  subst h; simp [Stmt.slotBase, pure, Except.pure]

/-- **A templated unit's share is the projection of its slot's base**: what `unitShare` returns is the range's net's
slot base, projected, entry by entry. -/
theorem unitShare_get {st : Stmt} {alpha : F128} {lw rho rr p : Array F128} {r : Range}
    (h : st.unitShare alpha lw rho rr r = some p) :
    ∃ i, r.kind = .net i ∧ (st.c.nets.getD i default).2.unitLog = r.slotLog ∧ ∀ s < 64,
      p[s]! = (Stmt.project (Stmt.baseOf alpha (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6)))
        (st.c.nets.getD i default).2.a (st.c.nets.getD i default).2.b (st.c.nets.getD i default).2.lookup r.slotLog)
        (eqTable (rr.extract 0 (r.slotLog - 6))))[s]! := by
  unfold Stmt.unitShare at h
  split at h
  · rename_i tu i htu hk
    dsimp only at h
    split at h
    · rename_i hc
      simp only [Bool.and_eq_true, beq_iff_eq, Option.isNone_iff_eq_none, withPtrEq, decide_eq_true_eq] at hc
      obtain ⟨⟨hsl, hlk⟩, hnet⟩ := hc
      have hF : Tmpl.foldUnit tu.t tu.lays r.slotLog alpha lw (rho.extract 0 (r.slotLog - 6))
          (rr.extract 0 (r.slotLog - 6)) = .ok p := by
        revert h
        cases Tmpl.foldUnit tu.t tu.lays r.slotLog alpha lw (rho.extract 0 (r.slotLog - 6))
          (rr.extract 0 (r.slotLog - 6)) <;> simp [Except.toOption]
      refine ⟨i, hk, hsl, fun s hs => ?_⟩
      have hU : Flock.Tmpl.unrollAt tu.t tu.lays = .ok (st.c.nets.getD i default).2 := by rw [hnet]; exact tu.unrolled
      rw [hlk]
      exact foldUnit_eq_project tu.wf tu.laid hU hsl hF hs
    · simp at h
  · simp at h

/-- The slot-type lookup `partialRanges` and `partialRangesT` share. -/
def slotLook (st : Stmt) (alpha : F128) (lw rho : Array F128) (key : Kind × ℕ)
    (b : Array ((Kind × ℕ) × Array F128)) :
    V (Array F128 × Array ((Kind × ℕ) × Array F128)) :=
  match b.find? (fun kb => decide (kb.1 = key)) with
  | some (_, y) => pure (y, b)
  | none => do
    let y ← st.slotBase alpha lw rho key
    pure (y, b.push (key, y))

/-- A range's step on the 64 entries, its share `p`. -/
def rangeAcc (r : Range) (rho rr out p : Array F128) : Array F128 :=
  Array.ofFn (n := 64) fun s => out[s.val]! + Stmt.sumUpTo r.count (fun k =>
    (eqTable (rho.extract (r.slotLog - 6) rho.size))[r.pos0 + k]! *
      (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) * p[s.val]!

theorem partialRanges_cons (st : Stmt) (alpha : F128) (lw rho rr : Array F128) (r : Range) (rs : List Range)
    (b : Array ((Kind × ℕ) × Array F128)) (out : Array F128) :
    st.partialRanges alpha lw rho rr (r :: rs) b out = (slotLook st alpha lw rho (r.kind, r.slotLog) b >>= fun x =>
      st.partialRanges alpha lw rho rr rs x.2
        (rangeAcc r rho rr out (Stmt.project x.1 (eqTable (rr.extract 0 (r.slotLog - 6)))))) := by
  rw [Stmt.partialRanges]; unfold slotLook
  cases Array.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) b with
  | some kb => rfl
  | none => cases st.slotBase alpha lw rho (r.kind, r.slotLog) <;> rfl

theorem partialRangesT_some (st : Stmt) (alpha : F128) (lw rho rr : Array F128) {r : Range} (rs : List Range)
    (b : Array ((Kind × ℕ) × Array F128)) (out : Array F128) {p : Array F128}
    (h : st.unitShare alpha lw rho rr r = some p) :
    st.partialRangesT alpha lw rho rr (r :: rs) b out =
      st.partialRangesT alpha lw rho rr rs b (rangeAcc r rho rr out p) := by
  rw [Stmt.partialRangesT, h]; rfl

theorem partialRangesT_none (st : Stmt) (alpha : F128) (lw rho rr : Array F128) {r : Range} (rs : List Range)
    (b : Array ((Kind × ℕ) × Array F128)) (out : Array F128) (h : st.unitShare alpha lw rho rr r = none) :
    st.partialRangesT alpha lw rho rr (r :: rs) b out = (slotLook st alpha lw rho (r.kind, r.slotLog) b >>= fun x =>
      st.partialRangesT alpha lw rho rr rs x.2
        (rangeAcc r rho rr out (Stmt.project x.1 (eqTable (rr.extract 0 (r.slotLog - 6)))))) := by
  rw [Stmt.partialRangesT, h]; dsimp only; unfold slotLook
  cases Array.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) b with
  | some kb => rfl
  | none => cases st.slotBase alpha lw rho (r.kind, r.slotLog) <;> rfl

theorem slotLook_ok {st : Stmt} {alpha : F128} {lw rho : Array F128} {key : Kind × ℕ} {y : Array F128}
    {b : Array ((Kind × ℕ) × Array F128)} (hb : CacheOk st alpha lw rho b)
    (hy : st.slotBase alpha lw rho key = .ok y) :
    ∃ b', CacheOk st alpha lw rho b' ∧ slotLook st alpha lw rho key b = .ok (y, b') := by
  unfold slotLook
  cases hf : b.find? (fun kb => decide (kb.1 = key)) with
  | some kb =>
    have hk := Array.find?_some hf
    have hkey : kb.1 = key := of_decide_eq_true hk
    have h := hb kb (Array.mem_of_find?_eq_some hf)
    rw [hkey, hy, Except.ok.injEq] at h
    subst h
    exact ⟨b, hb, rfl⟩
  | none =>
    refine ⟨_, cacheOk_push hb hy, ?_⟩
    simp only [bind, Except.bind, hy, pure, Except.pure]

theorem slotLook_error {st : Stmt} {alpha : F128} {lw rho : Array F128} {key : Kind × ℕ} {e : String}
    {b : Array ((Kind × ℕ) × Array F128)} (hb : CacheOk st alpha lw rho b)
    (he : st.slotBase alpha lw rho key = .error e) :
    slotLook st alpha lw rho key b = .error e := by
  unfold slotLook
  cases hf : b.find? (fun kb => decide (kb.1 = key)) with
  | some kb =>
    have hk := Array.find?_some hf
    have hkey : kb.1 = key := of_decide_eq_true hk
    have h := hb kb (Array.mem_of_find?_eq_some hf)
    rw [hkey, he] at h
    cases h
  | none =>
    simp only [bind, Except.bind, he]

/-- **The templated fold is the flat one**: `partialRangesT` is `partialRanges`, on any two caches of `slotBase`'s
values. -/
theorem partialRangesT_eq (st : Stmt) (alpha : F128) (lw rho rr : Array F128) : ∀ (rs : List Range)
    (b1 b2 : Array ((Kind × ℕ) × Array F128)) (out : Array F128),
    CacheOk st alpha lw rho b1 → CacheOk st alpha lw rho b2 →
    st.partialRangesT alpha lw rho rr rs b1 out = st.partialRanges alpha lw rho rr rs b2 out := by
  intro rs
  induction rs with
  | nil => intro b1 b2 out _ _; rfl
  | cons r rs ih =>
    intro b1 b2 out h1 h2
    cases hu : st.unitShare alpha lw rho rr r with
    | some p =>
      obtain ⟨i, hk, hsl, hp⟩ := unitShare_get hu
      have hsb : st.slotBase alpha lw rho (r.kind, r.slotLog) = .ok (Stmt.baseOf alpha
          (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) (st.c.nets.getD i default).2.a
          (st.c.nets.getD i default).2.b (st.c.nets.getD i default).2.lookup r.slotLog) := by
        rw [hk]; exact slotBase_net st alpha lw rho hsl
      obtain ⟨b2', hb2', he2⟩ := slotLook_ok h2 hsb
      rw [partialRangesT_some st alpha lw rho rr rs b1 out hu, partialRanges_cons, he2]
      have ha : rangeAcc r rho rr out p = rangeAcc r rho rr out (Stmt.project (Stmt.baseOf alpha
          (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) (st.c.nets.getD i default).2.a
          (st.c.nets.getD i default).2.b (st.c.nets.getD i default).2.lookup r.slotLog)
          (eqTable (rr.extract 0 (r.slotLog - 6)))) :=
        congrArg (fun f => Array.ofFn (n := 64) f) (funext fun s => by rw [hp s.val s.isLt])
      rw [ha]
      exact ih b1 b2' _ h1 hb2'
    | none =>
      rw [partialRangesT_none st alpha lw rho rr rs b1 out hu, partialRanges_cons]
      cases hsb : st.slotBase alpha lw rho (r.kind, r.slotLog) with
      | error e =>
        rw [slotLook_error h1 hsb, slotLook_error h2 hsb]; rfl
      | ok y =>
        obtain ⟨b1', hb1', he1⟩ := slotLook_ok h1 hsb
        obtain ⟨b2', hb2', he2⟩ := slotLook_ok h2 hsb
        rw [he1, he2]
        exact ih b1' b2' _ hb1' hb2'

end FlockSoundness.TmplFold
