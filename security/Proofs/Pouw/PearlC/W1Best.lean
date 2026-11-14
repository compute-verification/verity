module

public import Proofs.Pouw.Prob
public import Proofs.Pouw.PearlC.TTOutW1
public import Definitions.Pouw.Assumptions.PearlC.TTOutS8

/-!
# TT_OUT along a domination within every bound, and W1's unpriced resources at the adversary's best case

* **A domination within every bound carries TT_OUT down** (`Pouw.PearlC.ttOutTile_of_dominatedWithin`): if `CM'`
  dominates `CM` within every bound on the tiles (`CostModel.DominatedWithin`), every adversary of `CM` within `T` and
  `q` has one of `CM'` within `T` and `q` whose good tiles beat TT_OUT's threshold at least as often, so TT_OUT per tile
  at `CM'` is TT_OUT per tile at `CM`.
* **The program-by-program forms imply it.** `CostModel.DominatedBy` (for each adversary, a program of `CM'` on its
  inputs at no more cost and calls on every oracle and salt, whose good tiles earn no less there) gives a domination
  within every bound (`CostModel.dominatedWithin_of_dominatedBy`), and a simulation (`CostModel.SimulatedBy`, the same
  run at no more cost and calls) is such a domination (`CostModel.dominatedBy_of_simulatedBy`).
  `ttOutTile_of_dominatedBy` and `ttOutTile_of_simulatedBy` are the transfers along them.
* **A compression and a first fetch at 0 is the best case** (`W1.w1Priced_s8H100Tensor_simulatedBy_best`): the run's
  memory and its fetched words never read a price, and its cost only grows with each, so at any `sha, fetch ≥ 0`
  `W1Device.s8H100Tensor sha fetch` is simulated by `W1Device.s8H100Best`, each program by itself.
  `ttOutTilePearlCDevRev1W1_s8H100Tensor_of_best` is the consequence for TT_OUT: `TTOutTilePearlCH100Rev1S8` holds at
  every such price.
-/

@[expose] public section

namespace Pouw.PearlC

variable {Q R S : Type}

/-- **TT_OUT per tile moves along a domination within every bound**: at `CM`, given `CM.Queries` and a domination
of `CM` by `CM'` within every bound on the tiles, from TT_OUT per tile at `CM'`. -/
theorem ttOutTile_of_dominatedWithin [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM CM' : CostModel Q R S}
    {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop} {g₀ : ℝ} {δ : ℕ → ℕ → ℝ}
    (hQ : CM.Queries) (hD : CM.DominatedWithin P TR D CM') (h : Assumptions.TTOutTile CM' P TR D g₀ δ) :
    Assumptions.TTOutTile CM P TR D g₀ δ := by
  refine ⟨hQ, fun U hU T hT q hq A hA => ?_⟩
  obtain ⟨A', hB, hpr⟩ := hD U hU T q A hA
  exact (hpr _).trans (h.2 U hU T hT q hq A' hB)

/-- **`CM'` dominates `CM` program by program on the tiles `TR`**: for every in-domain workload and every adversary of
`CM`, some program of `CM'`, run on the adversary's activations and its preprocessed state, costs no more and makes no
more oracle calls on every oracle and salt, and the credit of the good tiles it commits is at least that of the
adversary's there. -/
def CostModel.DominatedBy (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop) (CM CM' : CostModel Q R S) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ A : Adversary CM, ∃ p' : CM'.Prog, ∀ (H : Q → R) (s : S),
    CM'.cost p' H U A.act (A.pre.eval H) s ≤ CM.cost A.online H U A.act (A.pre.eval H) s ∧
      CM'.calls p' H U A.act (A.pre.eval H) s ≤ CM.calls A.online H U A.act (A.pre.eval H) s ∧
      ∑ g ∈ goodTiles P TR U H s (A.transcript U H s),
          TR.credit H s U g (actOf (A.transcript U H s) ((TR.tiling U.layout).unit g)) ≤
        ∑ g ∈ goodTiles P TR U H s (CM'.run p' H U A.act (A.pre.eval H) s),
          TR.credit H s U g (actOf (CM'.run p' H U A.act (A.pre.eval H) s) ((TR.tiling U.layout).unit g))

/-- A program-by-program domination is a domination within every bound: the dominating program, with the adversary's
preprocessing and activations, stays within its bounds and earns no less on every oracle and salt. -/
theorem CostModel.dominatedWithin_of_dominatedBy [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]
    {CM CM' : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop}
    (hD : CM.DominatedBy P TR D CM') : CM.DominatedWithin P TR D CM' := by
  intro U hU T q A hA
  obtain ⟨p, hp⟩ := hD U hU A
  refine ⟨⟨A.pre, p, A.act⟩, fun H s =>
    ⟨(hp H s).1.trans (hA H s).1, (Nat.add_le_add_left (hp H s).2.1 _).trans (hA H s).2⟩, fun c => ?_⟩
  exact pr_mono fun x hx => hx.trans_le (hp x.1 x.2).2.2

/-- **TT_OUT per tile moves along a program-by-program domination**: `ttOutTile_of_dominatedWithin` at
`dominatedWithin_of_dominatedBy`. -/
theorem ttOutTile_of_dominatedBy [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM CM' : CostModel Q R S}
    {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop} {g₀ : ℝ} {δ : ℕ → ℕ → ℝ}
    (hQ : CM.Queries) (hD : CM.DominatedBy P TR D CM') (h : Assumptions.TTOutTile CM' P TR D g₀ δ) :
    Assumptions.TTOutTile CM P TR D g₀ δ :=
  ttOutTile_of_dominatedWithin hQ (CostModel.dominatedWithin_of_dominatedBy hD) h

/-- **`CM'` simulates `CM`**: every program of `CM` has a program of `CM'`, fixed before the oracle and the salt, with the
same run at no more cost and no more oracle calls on every oracle, workload, activation input, state and salt. -/
def CostModel.SimulatedBy (CM CM' : CostModel Q R S) : Prop :=
  ∀ p : CM.Prog, ∃ p' : CM'.Prog, ∀ (H : Q → R) (U : Workload) (act : ℕ → Codes) (σ : State) (s : S),
    CM'.run p' H U act σ s = CM.run p H U act σ s ∧ CM'.cost p' H U act σ s ≤ CM.cost p H U act σ s ∧
      CM'.calls p' H U act σ s ≤ CM.calls p H U act σ s

/-- A simulation is a domination on any tiles: the simulating program commits the same transcript. -/
theorem CostModel.dominatedBy_of_simulatedBy {CM CM' : CostModel Q R S} (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) (h : CM.SimulatedBy CM') : CM.DominatedBy P TR D CM' := by
  intro U _ A
  obtain ⟨p, hp⟩ := h A.online
  refine ⟨p, fun H s => ⟨(hp H U A.act (A.pre.eval H) s).2.1, (hp H U A.act (A.pre.eval H) s).2.2, ?_⟩⟩
  rw [(hp H U A.act (A.pre.eval H) s).1]
  exact le_rfl

/-- **TT_OUT per tile moves along a simulation**: `ttOutTile_of_dominatedBy` at `dominatedBy_of_simulatedBy`. -/
theorem ttOutTile_of_simulatedBy [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM CM' : CostModel Q R S}
    {P : Protocol Q R S} {TR : TileRules Q R S} {D : Layout → Prop} {g₀ : ℝ} {δ : ℕ → ℕ → ℝ}
    (hQ : CM.Queries) (hS : CM.SimulatedBy CM') (h : Assumptions.TTOutTile CM' P TR D g₀ δ) :
    Assumptions.TTOutTile CM P TR D g₀ δ :=
  ttOutTile_of_dominatedBy hQ (CostModel.dominatedBy_of_simulatedBy P TR D hS) h

end Pouw.PearlC

namespace Pouw.PearlC.W1

variable {a b a' b' : ℚ}

theorem exec_s8H100Tensor (M : Mem) (i : Instr) :
    i.exec (W1Device.s8H100Tensor a' b') M = i.exec (W1Device.s8H100Tensor a b) M := by
  cases i <;> rfl

theorem reads_s8H100Tensor (M : Mem) (i : Instr) :
    i.reads (W1Device.s8H100Tensor a' b') M = i.reads (W1Device.s8H100Tensor a b) M := by
  cases i <;> rfl

theorem price_s8H100Tensor_le (h : a' ≤ a) (t : Bool) (i : Instr) :
    i.price (W1Device.s8H100Tensor a' b') t ≤ i.price (W1Device.s8H100Tensor a b) t := by
  cases i <;> first | exact le_rfl | exact add_le_add h le_rfl

/-- A run on `s8H100Tensor a' b'` and one on `s8H100Tensor a b` from states with one memory and one fetch list keep them equal,
and the first's cost stays at most the second's when `a' ≤ a`. -/
theorem exec_s8H100Tensor_le (h : a' ≤ a) (c : Code) : ∀ (t : Bool) (st st' : St), st'.mem = st.mem →
    st'.fetched = st.fetched → st'.cost ≤ st.cost →
    (exec (W1Device.s8H100Tensor a' b') t c st').mem = (exec (W1Device.s8H100Tensor a b) t c st).mem ∧
      (exec (W1Device.s8H100Tensor a' b') t c st').fetched = (exec (W1Device.s8H100Tensor a b) t c st).fetched ∧
      (exec (W1Device.s8H100Tensor a' b') t c st').cost ≤ (exec (W1Device.s8H100Tensor a b) t c st).cost := by
  induction c with
  | done => intro t st st' hm hf hc; exact ⟨hm, hf, hc⟩
  | op i k ih =>
    intro t st st' hm hf hc
    simp only [exec]
    rw [hm, hf, exec_s8H100Tensor (a' := a') (b' := b') (a := a) (b := b),
      reads_s8H100Tensor (a' := a') (b' := b') (a := a) (b := b)]
    refine ih t _ _ ?_ ?_ ?_
    · rfl
    · rfl
    · exact add_le_add hc (price_s8H100Tensor_le h t i)
  | ite p ct ce k iht ihe ihk =>
    intro t st st' hm hf hc
    simp only [exec]
    rw [hm, hf]
    split_ifs
    · have e := ihe true ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + (W1Device.s8H100Tensor a b).prices.int32⟩
        ⟨st.mem, st.fetched ++ [p].filter preSalt, st'.cost + (W1Device.s8H100Tensor a' b').prices.int32⟩ rfl rfl
        (add_le_add hc le_rfl)
      exact ihk t _ _ e.1 e.2.1 e.2.2
    · have e := iht true ⟨st.mem, st.fetched ++ [p].filter preSalt, st.cost + (W1Device.s8H100Tensor a b).prices.int32⟩
        ⟨st.mem, st.fetched ++ [p].filter preSalt, st'.cost + (W1Device.s8H100Tensor a' b').prices.int32⟩ rfl rfl
        (add_le_add hc le_rfl)
      exact ihk t _ _ e.1 e.2.1 e.2.2

theorem final_s8H100Tensor (h : a' ≤ a) (salt : List ℕ) (pr : CodeImage) (U : Workload) (act : ℕ → Codes)
    (σ : State) :
    (final (W1Device.s8H100Tensor a' b') salt pr U act σ).mem = (final (W1Device.s8H100Tensor a b) salt pr U act σ).mem ∧
      (final (W1Device.s8H100Tensor a' b') salt pr U act σ).fetched =
        (final (W1Device.s8H100Tensor a b) salt pr U act σ).fetched ∧
      (final (W1Device.s8H100Tensor a' b') salt pr U act σ).cost ≤ (final (W1Device.s8H100Tensor a b) salt pr U act σ).cost :=
  exec_s8H100Tensor_le h _ false _ _ rfl rfl le_rfl

/-- **A compression and a first fetch at 0 is the adversary's best case**: at `sha, fetch ≥ 0`, W1 priced on
`s8H100Tensor sha fetch` is simulated by W1 priced on `s8H100Best`, each program by itself. -/
theorem w1Priced_s8H100Tensor_simulatedBy_best {Q R S : Type} {sha fetch : ℚ} (hs : 0 ≤ sha) (hf : 0 ≤ fetch)
    (salt : S → List ℕ) :
    (w1Priced (W1Device.s8H100Tensor sha fetch) salt : CostModel Q R S).SimulatedBy
      (w1Priced W1Device.s8H100Best salt) := by
  intro p
  refine ⟨p, fun H U act σ s => ?_⟩
  obtain ⟨e1, e2, e3⟩ := final_s8H100Tensor (a := sha) (b := fetch) (a' := 0) (b' := 0) hs (salt s) p U act σ
  refine ⟨?_, ?_, le_rfl⟩
  · show readout U act (final (W1Device.s8H100Tensor 0 0) (salt s) p U act σ).mem =
      readout U act (final (W1Device.s8H100Tensor sha fetch) (salt s) p U act σ).mem
    rw [e1]
  · show (((final (W1Device.s8H100Tensor 0 0) (salt s) p U act σ).cost +
        (0 : ℚ) * w1Read (W1Device.s8H100Tensor 0 0) (salt s) p U act σ : ℚ) : ℝ) ≤
      (((final (W1Device.s8H100Tensor sha fetch) (salt s) p U act σ).cost +
        fetch * w1Read (W1Device.s8H100Tensor sha fetch) (salt s) p U act σ : ℚ) : ℝ)
    have hr : w1Read (W1Device.s8H100Tensor 0 0) (salt s) p U act σ =
        w1Read (W1Device.s8H100Tensor sha fetch) (salt s) p U act σ := by
      unfold w1Read
      rw [e2]
    rw [hr, Rat.cast_le]
    have hn := mul_nonneg hf (Nat.cast_nonneg (α := ℚ) (w1Read (W1Device.s8H100Tensor sha fetch) (salt s) p U act σ))
    linarith

end Pouw.PearlC.W1

namespace Pouw.PearlC

open Pouw.PearlC.Assumptions Pouw.PearlC.TTOutW1

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **`TTOutTilePearlCH100Rev1S8` holds at every price of a compression and a first fetch**: TT_OUT rev1 per tile at
the H100 record on `W1Device.s8H100Tensor sha fetch`, at any `sha, fetch ≥ 0`. -/
theorem ttOutTilePearlCDevRev1W1_s8H100Tensor_of_best {sha fetch : ℚ} (hs : 0 ≤ sha) (hf : 0 ≤ fetch)
    {salt : S → List ℕ} {sem : PearlCSem Q R S} {ρ : ℚ} (h : TTOutTilePearlCH100Rev1S8 salt sem ρ) :
    TTOutTilePearlCDevRev1W1 (W1.W1Device.s8H100Tensor sha fetch) salt devH100Tensor sem ρ :=
  (ttOutTileW1_iff _ _ _ _ _ _ _).2 (ttOutTile_of_simulatedBy (w1Priced_queries _ salt)
    (W1.w1Priced_s8H100Tensor_simulatedBy_best hs hf salt) ((ttOutTileW1_iff _ _ _ _ _ _ _).1 h))

end Pouw.PearlC
