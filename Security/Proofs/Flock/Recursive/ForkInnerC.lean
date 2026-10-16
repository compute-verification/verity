import Proofs.Flock.Recursive.ForkLast

/-!
# Recursion: gap 2 at the committed rounds

`zk_sessions_recursive_fork_inner` for an inner protocol whose last round is not committed (candidate 3's): V*'s bridge
as `Flock.Assumptions.VBridgeC`, which decodes the rounds before the last as `VBridge` does and at the last round holds
some message, and the inner term `InnerAtLeavesL`'s, whose last stage term is the message the recursive prover's own
leaf decodes. The conclusion is `zk_sessions_recursive_fork_inner`'s.

* `exists_of_rel`, `zip_eq_relabelL`: a decoded transcript that holds a last message, and whose earlier messages are the
  reference openings', is `relabelL` at that last message.
* `InnerAtLeavesL.toZ`: `InnerAtLeavesL` as `InnerAtLeavesZ` at that reading.
* `zk_sessions_recursive_fork_innerL`: gap 2 at `VBridgeC`, the inner term a hypothesis.
* `zk_sessions_recursive_fork_innerC`: it for an inner verifier and a pick that ignore the last coin (`Inner.LastFree`,
  `PickLastFree`), the inner term from `innerAtLeavesL_of_forkOn`.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section Last

variable {M C : Type}

theorem exists_of_rel {α β : Type} {R : α → β → Prop} {o : Option α} {o' : Option β} (h : Option.Rel R o o')
    (h' : o' ≠ none) : ∃ a, o = some a := by
  cases h with
  | some _ => exact ⟨_, rfl⟩
  | none => exact absurd rfl h'

/-- **A decoded transcript is `relabelL` at its last message** when its messages before the last are the reference
openings'. -/
theorem zip_eq_relabelL {N : Type} : ∀ (ref : List (M × C) → M → N) (m : N) (t : List (M × C)) (w : List N),
    w[t.length - 1]? = some m → (∀ k, k + 1 < t.length → w[k]? = (t[k]?).map fun e => ref (t.take k) e.1) →
    w.zip (t.map Prod.snd) = relabelL ref m t
  | _, _, [], _, _, _ => by simp [relabelL]
  | _, m, [(c, ρ)], w, hm, _ => by
      obtain ⟨w', rfl⟩ : ∃ w', w = m :: w' := by
        cases w with
        | nil => simp at hm
        | cons a w' => simp at hm; exact ⟨w', by rw [hm]⟩
      simp [relabelL]
  | ref, m, (c, ρ) :: e :: t, w, hm, h => by
      obtain ⟨a, w', rfl⟩ : ∃ a w', w = a :: w' := by
        cases w with
        | nil => simpa using h 0 (by simp only [List.length_cons] <;> omega)
        | cons a w' => exact ⟨a, w', rfl⟩
      have h0 : a = ref [] c := by simpa using h 0 (by simp only [List.length_cons] <;> omega)
      have hlen : ((c, ρ) :: e :: t).length - 1 = (e :: t).length - 1 + 1 := by
        simp only [List.length_cons] <;> omega
      have hm' : w'[(e :: t).length - 1]? = some m := by
        rw [hlen, List.getElem?_cons_succ] at hm
        exact hm
      have ht : ∀ k, k + 1 < (e :: t).length →
          w'[k]? = ((e :: t)[k]?).map fun e' => (fun p c' => ref ((c, ρ) :: p) c') ((e :: t).take k) e'.1 :=
        fun k hk => by simpa using h (k + 1) (by simp only [List.length_cons] at hk ⊢; omega)
      rw [List.map_cons, List.zip_cons_cons, h0,
        zip_eq_relabelL (fun p c' => ref ((c, ρ) :: p) c') m (e :: t) w' hm' ht]
      rfl

end Last

section Outer

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

theorem InnerAtLeavesL.toZ {σ : Strategy (recGame L Reg I Cm outer)}
    {Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop} {x : Reg → L.Ω → I.Stmt} {W : Type}
    {dec : ∀ R ω t, Strategy (outer R ω t) → β → W} {wmsg : W → List I.Msg}
    {ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg} {εin : ℝ≥0∞} (h : InnerAtLeavesL σ Acc x dec wmsg ref εin) :
    InnerAtLeavesZ σ Acc x dec wmsg (fun ω t u => ∃ m, u = relabelL (ref ω) m t) εin := by
  intro ω
  refine (leafAvg_mono ?_ _).trans (h ω)
  intro t s
  refine ite_one_zero_le ?_
  rintro ⟨hx, b, hb, ha, ⟨m, hz⟩, hq⟩
  exact ⟨hx, m, b, hb, ha, hz, by rw [← hz]; exact hq⟩

end Outer

section ChainL

open Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL Composed
  ZkReg

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg Cm : Type} {I : Inner}

/-- **Gap 2 at the committed rounds, the inner term given**: `zk_sessions_recursive_fork_on` at the stages' commit
strings before the last round and messages equal there, reading the decoded transcript as `relabelL` at its own last
message, with the inner term `InnerAtLeavesL` a hypothesis wherever the recursive prover has no inner fork, in place of
the inner protocol's soundness. -/
theorem zk_sessions_recursive_fork_innerL
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hbr : Flock.Assumptions.VBridgeC zs dg N k Rw x wmsg Sk cmt msg)
    {D : Type} (H : List UInt8 → D) (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair)
    {εin : ℝ≥0∞} (hr0 : 0 < I.r)
    (hleaf : ¬ InnerForkRec zs dg N k Rw σ x wmsg H pick →
      ∀ ref, InnerAtLeavesL σ (fun R ω t o => AcceptsV (zs R ω t) o) x (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt)
        (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω)) (fun w => wmsg w.1) ref εin) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg H pick ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin := by
  by_cases hf : InnerForkRec zs dg N k Rw σ x wmsg H pick
  · exact Or.inr (Or.inr (Or.inr (Or.inl hf)))
  have hrd : ∀ (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) ω t (l : List I.Msg), t.length = I.r →
      (∀ k < I.r, Option.Rel (fun a b => k + 1 < I.r → a = b) l[k]? ((t[k]?).map fun e => ref ω (t.take k) e.1)) →
      ∃ m, l.zip (t.map Prod.snd) = relabelL (ref ω) m t := by
    intro ref ω t l ht h
    obtain ⟨m, hm⟩ := exists_of_rel (h (t.length - 1) (by omega)) (by
      simp only [ne_eq, Option.map_eq_none_iff, List.getElem?_eq_none_iff, Nat.not_le]; omega)
    exact ⟨m, zip_eq_relabelL (ref ω) m t l hm fun k hk =>
      option_eq_of_rel (R := fun a b => k + 1 < I.r → a = b) (fun _ _ hab => hab (by omega)) (h k (by omega))⟩
  rcases zk_sessions_recursive_fork_on hA3 zs dg N hk hRw hM hρ hr σ x wmsg hbr.1 (fun i _ a b => i + 1 < I.r → a = b)
      (fun ref ω t u => ∃ m, u = relabelL (ref ω) m t) hrd (fun i j => if i + 1 < I.r then Sk i j else ∅) cmt msg
      hbr.2.1 hbr.2.2.1 hbr.2.2.2 (fun ref => InnerAtLeavesL.toZ (hleaf hf ref))
      with h | h | h | h
  · exact Or.inl h
  · exact Or.inr (Or.inl h)
  · exact Or.inr (Or.inr (Or.inl h))
  · exact Or.inr (Or.inr (Or.inr (Or.inr h)))

end ChainL

section Chain

open Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed DecodeZL Composed
  ZkReg

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}

/-- **Gap 2 at the committed rounds.** `zk_sessions_recursive_fork_inner` with V*'s bridge as
`Flock.Assumptions.VBridgeC`, for an inner verifier and a pick that ignore the last round's coin: a registrant fork, a
round fork, a session fork, an inner fork of `σ`'s own runs, or the same bound: `zk_sessions_recursive_fork_innerL`
with the inner term from `innerAtLeavesL_of_forkOn`. -/
theorem zk_sessions_recursive_fork_innerC
    (hA3 : ∀ j, Assumptions.UniformRandomBytes (Vs j).law.Lb (Vs j).law.src)
    (zs : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N : ℕ)
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ R ω t R₂ j S' R' i,
      rateZ (ZkHidden.planZHJ (Reg := Reg') (zs R ω t R₂ j).dj (zs R ω t R₂ j).y₀ S' R') i k ≤ ρ)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N))
    (x : Reg → L.Ω → I.Stmt) (wmsg : ((j : Fin S) → (Vs j).Bits) → List I.Msg)
    (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))
    (cmt : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg)
    (msg : ℕ → ((j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) → I.Msg)
    (hbr : Flock.Assumptions.VBridgeC zs dg N k Rw x wmsg Sk cmt msg)
    {D : Type} (H : List UInt8 → D) (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair)
    {εin : ℝ≥0∞} (hin : InnerSoundForkOn H I pick εin) (hr0 : 0 < I.r) (hlast : I.LastFree)
    (hpick : ∀ x, PickLastFree I.r (pick x)) :
    (regFinder zs dg N σ).Finds H512 ∨ RoundFork zs dg N k Rw σ cmt ∨ SessionFork zs dg N k Rw M σ ∨
      InnerForkRec zs dg N k Rw σ x wmsg H pick ∨
      prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
          (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
        expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).boundCR dg N (Vs j).law.model k Rw M ρ τ) s) +
          (I.r : ℝ≥0∞) * expAt σ (fun ω t s => sumV (zs σ.1 ω t) dg N
            (fun R₂ j τ => (zs σ.1 ω t R₂ j).slackCR dg N k Rw M ρ τ) s) + εin :=
  zk_sessions_recursive_fork_innerL hA3 zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr H pick hr0 fun hf ref =>
    innerAtLeavesL_of_forkOn σ (fun R ω t o => AcceptsV (zs R ω t) o) x
      (W := ((j : Fin S) → (Vs j).Bits) × I.Stmt) (fun R ω t _ o => (layers (zs R ω t o.1) dg N k Rw o.2, x R ω))
      (fun w => wmsg w.1) H pick hin hlast hpick hf ref

end Chain

end FlockSoundness.Discharge.Recursion
