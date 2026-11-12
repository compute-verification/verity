module

public import Proofs.Pouw.PearlC.TTOutW1
public import Definitions.Assumptions.PearlC

/-!
# TT_OUT along a simulation, and W1's unpriced resources at the adversary's best case

* **A simulation carries TT_OUT down** (`Pouw.ttOutTile_of_simulatedBy`): if `CM'` simulates `CM`
  (`CostModel.SimulatedBy`), every adversary of `CM` is one of `CM'` with the same transcript within the same bounds,
  so TT_OUT per tile at `CM'` is TT_OUT per tile at `CM`.
* **A compression and a first fetch at 0 is the best case** (`W1.w1Priced_s8H100_simulatedBy_best`): the run's memory
  and its fetched words never read a price, and its cost only grows with each, so at any `sha, fetch ≥ 0`
  `W1Device.s8H100 sha fetch` is simulated by `W1Device.s8H100Best`, each program by itself.
  `ttOutTilePearlCDevRev1W1_s8H100_of_best` is the consequence for TT_OUT: `TTOutTilePearlCH100Rev1S8` holds at every
  such price.
-/

@[expose] public section

namespace Pouw

open Pouw.PearlC

variable {Q R S : Type}

/-- **TT_OUT per tile moves along a simulation**: at `CM`, given `CM.Queries` and a simulation of `CM` by `CM'`, from
TT_OUT per tile at `CM'`. -/
theorem ttOutTile_of_simulatedBy [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM CM' : CostModel Q R S}
    {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop} {g₀ : ℝ} {δ : ℕ → ℕ → ℝ}
    (hQ : CM.Queries) (hS : CM.SimulatedBy CM') (h : Assumptions.TTOutTile CM' P TR D g₀ δ) :
    Assumptions.TTOutTile CM P TR D g₀ δ := by
  refine ⟨hQ, fun U hU T hT q hq A hA => ?_⟩
  obtain ⟨p, hp⟩ := hS A.online
  let A' : Adversary CM' := ⟨A.pre, p, A.act⟩
  have hrun : ∀ (H : Q → R) (s : S), A'.transcript U H s = A.transcript U H s :=
    fun H s => (hp H U A.act (A.pre.eval H) s).1
  have hB : A'.Bounded U T q := fun H s =>
    ⟨(hp H U A.act (A.pre.eval H) s).2.1.trans (hA H s).1,
      (Nat.add_le_add_left (hp H U A.act (A.pre.eval H) s).2.2 _).trans (hA H s).2⟩
  have key := h.2 U hU T hT q hq A' hB
  simp only [hrun] at key
  exact key

end Pouw

namespace Pouw.PearlC.W1

variable {a b a' b' : ℚ}

theorem exec_s8H100 (M : Mem) (i : Instr) :
    i.exec (W1Device.s8H100 a' b') M = i.exec (W1Device.s8H100 a b) M := by
  cases i <;> rfl

theorem reads_s8H100 (M : Mem) (i : Instr) :
    i.reads (W1Device.s8H100 a' b') M = i.reads (W1Device.s8H100 a b) M := by
  cases i <;> rfl

theorem price_s8H100_le (h : a' ≤ a) (t : Bool) (i : Instr) :
    i.price (W1Device.s8H100 a' b') t ≤ i.price (W1Device.s8H100 a b) t := by
  cases i <;> first | exact le_rfl | exact add_le_add h le_rfl

/-- A run on `s8H100 a' b'` and one on `s8H100 a b` from states with one memory and one fetch list keep them equal,
and the first's cost stays at most the second's when `a' ≤ a`. -/
theorem exec_s8H100_le (h : a' ≤ a) (c : Code) : ∀ (t : Bool) (st st' : St), st'.mem = st.mem →
    st'.fetched = st.fetched → st'.cost ≤ st.cost →
    (exec (W1Device.s8H100 a' b') t c st').mem = (exec (W1Device.s8H100 a b) t c st).mem ∧
      (exec (W1Device.s8H100 a' b') t c st').fetched = (exec (W1Device.s8H100 a b) t c st).fetched ∧
      (exec (W1Device.s8H100 a' b') t c st').cost ≤ (exec (W1Device.s8H100 a b) t c st).cost := by
  induction c with
  | done => intro t st st' hm hf hc; exact ⟨hm, hf, hc⟩
  | op i k ih =>
    intro t st st' hm hf hc
    simp only [exec]
    rw [hm, hf, exec_s8H100 (a' := a') (b' := b') (a := a) (b := b),
      reads_s8H100 (a' := a') (b' := b') (a := a) (b := b)]
    refine ih t _ _ ?_ ?_ ?_
    · rfl
    · rfl
    · exact add_le_add hc (price_s8H100_le h t i)
  | ite p ct ce k iht ihe ihk =>
    intro t st st' hm hf hc
    simp only [exec]
    rw [hm, hf]
    split_ifs
    · have e := ihe true ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + (W1Device.s8H100 a b).prices.int32⟩
        ⟨st.mem, st.fetched ++ [p].filter preSalt, st'.cost + (W1Device.s8H100 a' b').prices.int32⟩ rfl rfl
        (add_le_add hc le_rfl)
      exact ihk t _ _ e.1 e.2.1 e.2.2
    · have e := iht true ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + (W1Device.s8H100 a b).prices.int32⟩
        ⟨st.mem, st.fetched ++ [p].filter preSalt, st'.cost + (W1Device.s8H100 a' b').prices.int32⟩ rfl rfl
        (add_le_add hc le_rfl)
      exact ihk t _ _ e.1 e.2.1 e.2.2

theorem final_s8H100 (h : a' ≤ a) (salt : List ℕ) (pr : CodeImage) (U : Workload) (act : ℕ → Codes)
    (σ : State) :
    (final (W1Device.s8H100 a' b') salt pr U act σ).mem = (final (W1Device.s8H100 a b) salt pr U act σ).mem ∧
      (final (W1Device.s8H100 a' b') salt pr U act σ).fetched =
        (final (W1Device.s8H100 a b) salt pr U act σ).fetched ∧
      (final (W1Device.s8H100 a' b') salt pr U act σ).cost ≤ (final (W1Device.s8H100 a b) salt pr U act σ).cost :=
  exec_s8H100_le h _ false _ _ rfl rfl le_rfl

/-- **A compression and a first fetch at 0 is the adversary's best case**: at `sha, fetch ≥ 0`, W1 priced on
`s8H100 sha fetch` is simulated by W1 priced on `s8H100Best`, each program by itself. -/
theorem w1Priced_s8H100_simulatedBy_best {Q R S : Type} {sha fetch : ℚ} (hs : 0 ≤ sha) (hf : 0 ≤ fetch)
    (salt : S → List ℕ) :
    (w1Priced (W1Device.s8H100 sha fetch) salt : CostModel Q R S).SimulatedBy
      (w1Priced W1Device.s8H100Best salt) := by
  intro p
  refine ⟨p, fun H U act σ s => ?_⟩
  obtain ⟨e1, e2, e3⟩ := final_s8H100 (a := sha) (b := fetch) (a' := 0) (b' := 0) hs (salt s) p U act σ
  refine ⟨?_, ?_, le_rfl⟩
  · show readout U act (final (W1Device.s8H100 0 0) (salt s) p U act σ).mem =
      readout U act (final (W1Device.s8H100 sha fetch) (salt s) p U act σ).mem
    rw [e1]
  · show (((final (W1Device.s8H100 0 0) (salt s) p U act σ).cost +
        (0 : ℚ) * w1Read (W1Device.s8H100 0 0) (salt s) p U act σ : ℚ) : ℝ) ≤
      (((final (W1Device.s8H100 sha fetch) (salt s) p U act σ).cost +
        fetch * w1Read (W1Device.s8H100 sha fetch) (salt s) p U act σ : ℚ) : ℝ)
    have hr : w1Read (W1Device.s8H100 0 0) (salt s) p U act σ =
        w1Read (W1Device.s8H100 sha fetch) (salt s) p U act σ := by
      unfold w1Read
      rw [e2]
    rw [hr, Rat.cast_le]
    have hn := mul_nonneg hf (Nat.cast_nonneg (α := ℚ) (w1Read (W1Device.s8H100 sha fetch) (salt s) p U act σ))
    linarith

end Pouw.PearlC.W1

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions Pouw.PearlC.TTOutW1

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **`TTOutTilePearlCH100Rev1S8` holds at every price of a compression and a first fetch**: TT_OUT rev1 per tile at
the H100 record on `W1Device.s8H100 sha fetch`, at any `sha, fetch ≥ 0`. -/
theorem ttOutTilePearlCDevRev1W1_s8H100_of_best {sha fetch : ℚ} (hs : 0 ≤ sha) (hf : 0 ≤ fetch)
    {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutTilePearlCH100Rev1S8 salt sem ρ) :
    TTOutTilePearlCDevRev1W1 (W1.W1Device.s8H100 sha fetch) salt devH100 sem ρ :=
  (ttOutTileW1_iff _ _ _ _ _ _ _).2 (ttOutTile_of_simulatedBy (w1Priced_queries _ salt)
    (W1.w1Priced_s8H100_simulatedBy_best hs hf salt) ((ttOutTileW1_iff _ _ _ _ _ _ _).1 h))

end Pouw.PearlC
