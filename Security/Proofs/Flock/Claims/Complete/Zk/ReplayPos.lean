import Proofs.Flock.Claims.Complete.Zk.Transfer

/-!
# Where the masked replay stops, and the coins its claims read

`replayWith_ret`: a masked replay (`Zk.replayWith`) that returns from round `k₀` of rounds `R` stops at round
`k₀ + 3 + (m − 6) + 3 + (kLog − 6)` with nothing pending, and its claims are `abcClaims` at the coins it read there
(`zcAt`, `lcAt`), whatever the proof. It reads only rounds' coins and positions, so no prover is involved.

* **Runs that return** (`RetV x t P`): every run of `x` from `t` that returns ends in `P`; composed by `RetV.bind`,
  through loops by `RetV.forIn_range`.
-/

namespace FlockComplete

open Flock Transcript FlockComplete.Replay

/-- **Every run of `x` from `t` that returns ends in `P`.** -/
def RetV {α : Type} (x : TMW V α) (t : Transcript) (P : α → Transcript → Prop) : Prop :=
  ∀ a t', x t = .ok (a, t') → P a t'

namespace RetV

theorem bind {α β : Type} {x : TMW V α} {f : α → TMW V β} {t : Transcript} {P : β → Transcript → Prop}
    (h : RetV x t fun a t₁ => RetV (f a) t₁ P) : RetV (x >>= f) t P := fun b t' e => by
  obtain ⟨a, t₁, e1, e2⟩ := stateT_bind_ok e
  exact h a t₁ e1 b t' e2

theorem pure {α : Type} {a : α} {t : Transcript} {P : α → Transcript → Prop} (h : P a t) :
    RetV (Pure.pure a : TMW V α) t P := fun a' t' e => by
  change Except.ok (a, t) = _ at e
  cases e
  exact h

theorem throw {α : Type} {e : String} {t : Transcript} {P : α → Transcript → Prop} :
    RetV (MonadExcept.throw e : TMW V α) t P := fun _ _ h => by cases h

/-- A step that returns `a` at `t₁`, then `f a`. -/
theorem ok_then {α β : Type} {x : TMW V α} {f : α → TMW V β} {t t₁ : Transcript} {a : α}
    {P : β → Transcript → Prop} (hx : x t = .ok (a, t₁)) (h : RetV (f a) t₁ P) : RetV (x >>= f) t P :=
  bind fun a' t' e => by
    rw [hx] at e
    cases e
    exact h

theorem mono {α : Type} {x : TMW V α} {t : Transcript} {P Q : α → Transcript → Prop} (h : RetV x t P)
    (hPQ : ∀ a t', P a t' → Q a t') : RetV x t Q := fun a t' e => hPQ a t' (h a t' e)

/-- A gate on a returned `Bool` that returns `v` returned it. -/
theorem gate_ok {β : Type} {ob : V Bool} {why : String} {v w : β}
    (e : (ob >>= fun ok => if (!ok) = true then MonadExcept.throw why else Pure.pure v : V β) = .ok w) : v = w := by
  rcases ob with _ | b
  · cases e
  · cases b
    · cases e
    · change Except.ok v = Except.ok w at e
      cases e
      rfl

/-- **A squeeze that returns** read round `t.k`'s coins and moved to the next round with nothing pending. -/
theorem squeeze (hs : Hashes V) (kind : UInt8) {n : ℕ} (hn : n ≠ 0) {t : Transcript}
    {P : Array F128 → Transcript → Prop} (h : P (t.rounds[t.k]!).coins ⟨t.rounds, t.k + 1, ByteArray.empty⟩) :
    RetV (squeezeWith hs kind n) t P := by
  intro a t' e
  unfold squeezeWith at e
  simp only [hn, ite_false] at e
  rcases hr : t.rounds[t.k]? with _ | r
  · rw [hr] at e; cases e
  rw [hr] at e
  obtain ⟨hlt, rfl⟩ := Array.getElem?_eq_some_iff.1 hr
  dsimp only at e
  by_cases hN : t.rounds[t.k].n ≠ n
  · rw [ite_eq_left hN] at e; cases e
  rw [ite_eq_right hN] at e
  cases gate_ok e
  rw [getElem!_pos t.rounds t.k hlt] at h
  exact h

theorem check_t {c : Bool} {why : String} {t t' : Transcript} {a : Unit} (e : Flock.check c why t = .ok (a, t')) :
    t' = t := by
  cases c
  · cases e
  · rw [Flock.RecAlgebra.check_eq rfl] at e
    cases e
    rfl

theorem sample (hs : Hashes V) {t : Transcript} {P : F128 → Transcript → Prop}
    (h : P ((t.rounds[t.k]!).coins.getD 0 0) ⟨t.rounds, t.k + 1, ByteArray.empty⟩) :
    RetV (sampleWith hs) t P := by
  unfold sampleWith
  exact bind (squeeze hs 1 one_ne_zero (pure h))

/-- **A loop whose every returning step yields** keeps its invariant. -/
theorem forIn_range {σ : Type} (n : ℕ) (f : ℕ → σ → TMW V (ForInStep σ)) (inv : ℕ → σ → Transcript → Prop)
    (hf : ∀ i b t, i < n → inv i b t → RetV (f i b) t fun r t' => ∃ b', r = .yield b' ∧ inv (i + 1) b' t')
    (b : σ) (t : Transcript) (h0 : inv 0 b t) : RetV (forIn [:n] b f) t fun b' t' => inv n b' t' := by
  rw [FlockSoundness.Refine.forIn_range_eq]
  suffices h : ∀ k a b t, a + k = n → inv a b t → RetV (forIn (List.range' a k) b f) t fun b' t' => inv n b' t' from
    h n 0 b t (Nat.zero_add n) h0
  intro k
  induction k with
  | zero =>
    intro a b t hk hi
    rw [Nat.add_zero] at hk
    subst hk
    exact pure hi
  | succ k ih =>
    intro a b t hk hi
    rw [List.range'_succ, List.forIn_cons]
    refine bind fun r t₁ e => ?_
    obtain ⟨b', rfl, hi'⟩ := hf a b t (by omega) hi r t₁ e
    exact ih (a + 1) b' t₁ (by omega) hi'

end RetV

/-- The verifier's coin at round `j` of `R`: its first. -/
def coinAt (R : Array Round) (j : ℕ) : F128 := (R[j]!).coins.getD 0 0

/-- **The masked replay's zerocheck output** at the coins of rounds `R` from `k₀` (its values masked, so `0`). -/
def zcAt (m : ℕ) (R : Array Round) (k₀ : ℕ) : ZcOut :=
  { z := coinAt R (k₀ + 2), rho := pushes (fun i => coinAt R (k₀ + 3 + i)) (m - 6),
    rRest := pinned ++ (R[k₀ + 1]!).coins, vA := 0, vB := 0, vC := 0 }

/-- **The masked replay's lincheck output** at the coins of rounds `R` from `k₀` (its value masked, so `0`). -/
def lcAt (m kLog : ℕ) (R : Array Round) (k₀ : ℕ) : LcOut :=
  { w := 0, rPrime := coinAt R (k₀ + 3 + (m - 6) + 2 + (kLog - 6)),
    rIn := (pushes (fun j => coinAt R (k₀ + 3 + (m - 6) + 2 + j)) (kLog - 6)).reverse }

/-- **A masked replay that returns** from round `k₀` of `R` stops `3 + (m − 6) + 3 + (kLog − 6)` rounds on with
nothing pending, and its claims are `abcClaims` at the coins it read. -/
theorem replayWith_ret (hs : Hashes V) (st : Setup) (hm : 14 ≤ st.m) (lay : Zk.PadLayout) (p : Proof)
    (root : ByteArray) (R : Array Round) (k₀ : ℕ) (pend : ByteArray) :
    RetV (Zk.replayWith hs st lay p root) ⟨R, k₀, pend⟩ fun rp t' =>
      t' = ⟨R, k₀ + 3 + (st.m - 6) + 3 + (st.kLog - 6), ByteArray.empty⟩ ∧
        rp.claims = abcClaims st.m st.kLog (zcAt st.m R k₀) (lcAt st.m st.kLog R k₀) := by
  unfold Zk.replayWith
  simp only [Flock.With.liftTM_V]
  refine RetV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.absorbBytes_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.absorbBytes_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.absorbBytes_eq _ _) ?_
  refine RetV.bind (RetV.squeeze hs 2 (by decide) ?_)
  refine RetV.bind (RetV.squeeze hs 2 (by omega) ?_)
  refine RetV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
  refine RetV.bind (RetV.sample hs ?_)
  dsimp only
  refine RetV.bind (RetV.mono (RetV.forIn_range _ _
    (fun j b t => t.rounds = R ∧ t.k = k₀ + 3 + j ∧ t.pending = ByteArray.empty ∧
      b.2 = pushes (fun i => coinAt R (k₀ + 3 + i)) j) ?_ _ _ ⟨rfl, rfl, rfl, rfl⟩) fun b₁ t₁ h₁ => ?_)
  · intro i b t _ hi
    obtain ⟨hR, hk, hp, hb⟩ := hi
    refine RetV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine RetV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine RetV.bind (RetV.sample hs (RetV.pure ⟨_, rfl, hR, by simp only [hk]; omega, rfl, ?_⟩))
    simp only [hb, hR, hk, pushes, coinAt]
  obtain ⟨hR, hk, hp, hb⟩ := h₁
  refine RetV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
  refine RetV.ok_then (Flock.RecAlgebra.label_eq _ _) ?_
  refine RetV.bind (RetV.sample hs ?_)
  refine RetV.bind (RetV.sample hs ?_)
  dsimp only
  refine RetV.bind (RetV.mono (RetV.forIn_range _ _
    (fun j b t => t.rounds = R ∧ t.k = k₀ + 3 + (st.m - 6) + 2 + j ∧ t.pending = ByteArray.empty ∧
      b.2 = pushes (fun i => coinAt R (k₀ + 3 + (st.m - 6) + 2 + i)) j) ?_ _ _
      ⟨hR, by simp only [hk], rfl, rfl⟩) fun b₂ t₂ h₂ => ?_)
  · intro i b t _ hi
    obtain ⟨hR, hk, hp, hb⟩ := hi
    refine RetV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine RetV.ok_then (Flock.RecAlgebra.absorbF_eq _ _) ?_
    refine RetV.bind (RetV.sample hs (RetV.pure ⟨_, rfl, hR, by simp only [hk]; omega, rfl, ?_⟩))
    simp only [hb, hR, hk, pushes, coinAt]
  obtain ⟨hR₂, hk₂, hp₂, hb₂⟩ := h₂
  split
  rotate_left
  · exact RetV.throw
  refine RetV.bind (RetV.pure ?_)
  refine RetV.bind (fun _ t' e => ?_)
  cases RetV.check_t e
  refine RetV.ok_then (Flock.RecAlgebra.absorbSlice_eq _ _) ?_
  refine RetV.bind (RetV.sample hs (RetV.pure ⟨?_, ?_⟩))
  · simp only [hR₂, hk₂, Transcript.mk.injEq, true_and, and_true]
    omega
  · simp only [zcAt, lcAt, coinAt, hb, hb₂, hR₂, hk₂]

end FlockComplete
