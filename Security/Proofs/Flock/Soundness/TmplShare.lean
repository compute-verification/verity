import Proofs.Flock.Soundness.TmplUnit

/-!
# The templated unit's share is the flat slot's (PRF-133, part 3)

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
  subst h; simp [Stmt.slotBase]

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
    simp only [Stmt.partialRangesT, Stmt.partialRanges]
    cases hu : st.unitShare alpha lw rho rr r with
    | some p =>
      obtain ⟨i, hk, hsl, hp⟩ := unitShare_get hu
      have hsb : st.slotBase alpha lw rho (r.kind, r.slotLog) = .ok (Stmt.baseOf alpha
          (Stmt.tensor lw (rho.extract 0 (r.slotLog - 6))) (st.c.nets.getD i default).2.a
          (st.c.nets.getD i default).2.b (st.c.nets.getD i default).2.lookup r.slotLog) := by
        rw [hk]; exact slotBase_net st alpha lw rho hsl
      have hout : ∀ (b : Array F128), st.slotBase alpha lw rho (r.kind, r.slotLog) = .ok b →
          (Array.ofFn (n := 64) fun s => out[s.val]! + Stmt.sumUpTo r.count (fun k =>
            (eqTable (rho.extract (r.slotLog - 6) rho.size))[r.pos0 + k]! *
              (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) * p[s.val]!) =
          (Array.ofFn (n := 64) fun s => out[s.val]! + Stmt.sumUpTo r.count (fun k =>
            (eqTable (rho.extract (r.slotLog - 6) rho.size))[r.pos0 + k]! *
              (eqTable (rr.extract (r.slotLog - 6) rr.size))[r.pos0 + k]!) *
              (Stmt.project b (eqTable (rr.extract 0 (r.slotLog - 6))))[s.val]!) := by
        intro b hb
        rw [hsb, Except.ok.injEq] at hb
        subst hb
        exact congrArg (fun f => Array.ofFn (n := 64) f) (funext fun s => by rw [hp s.val s.isLt])
      cases hf : b2.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
      | some kb =>
        have hkey : kb.1 = (r.kind, r.slotLog) := of_decide_eq_true (Array.find?_some hf)
        have hkb : st.slotBase alpha lw rho (r.kind, r.slotLog) = .ok kb.2 := by
          rw [← hkey]; exact h2 kb (Array.mem_of_find?_eq_some hf)
        simp only [bind, Except.bind, pure, Except.pure]
        rw [hout kb.2 hkb]
        exact ih b1 b2 _ h1 h2
      | none =>
        simp only [bind, Except.bind, pure, Except.pure, hsb]
        rw [hout _ hsb]
        exact ih b1 _ _ h1 (cacheOk_push h2 hsb)
    | none =>
      simp only [bind, Except.bind, pure, Except.pure]
      cases hsb : st.slotBase alpha lw rho (r.kind, r.slotLog) with
      | error e =>
        have hn : ∀ b, CacheOk st alpha lw rho b → b.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) = none := by
          intro b hb
          cases hf : b.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
          | none => rfl
          | some kb =>
            have hkey : kb.1 = (r.kind, r.slotLog) := of_decide_eq_true (Array.find?_some hf)
            have := hb kb (Array.mem_of_find?_eq_some hf)
            rw [hkey, hsb] at this
            cases this
        rw [hn b1 h1, hn b2 h2]
      | ok x =>
        have hv : ∀ b, CacheOk st alpha lw rho b → ∃ b', CacheOk st alpha lw rho b' ∧
            (match b.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
              | some (_, y) => (Except.ok (y, b) : V (Array F128 × Array ((Kind × ℕ) × Array F128)))
              | none => match st.slotBase alpha lw rho (r.kind, r.slotLog) with
                | .error e => .error e
                | .ok y => .ok (y, b.push ((r.kind, r.slotLog), y))) = .ok (x, b') := by
          intro b hb
          cases hf : b.find? (fun kb => decide (kb.1 = (r.kind, r.slotLog))) with
          | some kb =>
            have hkey : kb.1 = (r.kind, r.slotLog) := of_decide_eq_true (Array.find?_some hf)
            have := hb kb (Array.mem_of_find?_eq_some hf)
            rw [hkey, hsb] at this
            cases this
            exact ⟨b, hb, rfl⟩
          | none => exact ⟨_, cacheOk_push hb hsb, by rw [hsb]⟩
        obtain ⟨b1', hb1', he1⟩ := hv b1 h1
        obtain ⟨b2', hb2', he2⟩ := hv b2 h2
        rw [he1, he2]
        exact ih b1' b2' _ hb1' hb2'

end FlockSoundness.TmplFold
