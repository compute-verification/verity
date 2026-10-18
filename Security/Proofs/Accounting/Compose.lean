import Models.Accounting.Guarantees.Compose

/-!
# PoUW's credit on an arbitrary declared circuit: proofs (open)

Every guarantee's proof here is open (note:compute-accounting/20261018T1758Z-draft-compose-r6 gives the order), and
this file doesn't land until none is a `sorry`. What the window's proofs' layer meets is proved: its static half is
defined from the IR (`Declared.layer`), so `StageCuts` (under `LayerCommitted`, which `Declared.atIR` meets),
`StageCoins` (under `Declared.Wf`), `StageSlots`, `WriterReads` and `StageComplete` (under `Declared.Wf` and
`units ≤ N_p`) are lemmas below, not premises, as are the draw cut of `Declared.atDraw`'s layer and `serviceGame`,
`tileMode` as `jointGame`, `jointMode` at a split-stage window's joint run.

1. `SplitRateClosed`: the folded branch at `W_0 = 0` (`ν = expm1(W_F·a/Λ)/((1 − p)·W_F)`, the exponent concave in
   `W_F` so its endpoints suffice) and the whole branch at `W_F = 0` the same at `b`, `p0` (`ν₀`), then the mixed
   point by convexity of the left side in `(W_F, W_0)` at the larger `Λ`, and `L_F, L_0 > 0` by convexity in them.
2. `SplitRateExp` (RT-COMPOSE r4's route): `ν, ν₀ ≥ 0` as `Λ > 0`; the left side is convex in `(L_F, L_0)` for
   `ρ·U ≤ 1` and the right affine, so `L_F ∈ {0, W_F}`, `L_0 ∈ {0, W_0}` suffice; `(1 − t)^K ≤ e^{−Kt}`; at each
   `L`-vertex the bound is the exponential of an affine function of `(W_F, W_0)`, so the simplex's vertices suffice;
   at `(w_max, 0)` with `L_F = 0` it is `e^{a·w_max/Λ} ≤ 1/(p + (1 − p)·e^{−Kρw_max})`, `Λ` at least the folded
   branch, and with `L_F = w_max` an identity by `ν`'s definition; the same at `(0, w_max)` with `b`, `p0`, `ν₀`.
3. `SplitRateMerged`: one class at cost `b` and luck `q`, the folded branch of step 1, then monotonicity in `a ≤ b`,
   `p, p0 ≤ q`.
4. `ServiceBudget`: given the stages' coins, a uniform `K`-subset of `N` misses the tiles whose checked values aren't
   honest with probability at most `(1 − ρ·(their share))^K`, which factors over the stages; `SplitRate` per stage at
   its share `≤ w_max`, its luck in expectation from `LuckAt` (`E[L_F] ≤ p·E[W_F]`, `E[L_0] ≤ p0·E[W_0]`), then
   backward induction over the stages, as `Accounting.SecurityProofs.TS.StageBudget`.
5. `ServiceCovers`: `Accounting.SecurityProofs.TS.DrawnLayer.covers_of_cone` at the tiles' words (`≥ 0`), at any joint
   run (the live statements use it at `SplitStages.joint`, by `serviceGame_eq` and `tileMode_eq`). The game's
   deficiency is the not-paid tiles' `tileCone` (the three modes, at `1` and `1`), so it is enough that a tile's layer
   cone is at most `N_w` times its `tileCone`: a layer edge `DrawnLayer.Late u v` at slot `k` has its position a value
   the prover writes and a gate of `u`'s unit (`layer_stageSlots`), and either one of `v`'s own gates (`u = v`) or an
   operand of one, read by a gate `j` of `v`'s, whose round is `v`'s (`Declared.Wf`, `layer_stageCoins`) and before the
   position's cut (`layer_stageCuts`, under `LayerCommitted`), so `Declared.Late` and a `Declared.UnitLate` edge; along
   `ReflTransGen` the layer's cone lies in `Declared.Reaches`, and a tile's words are `N_w` times its weight (`0` when
   `N_w = 0`: a tile's words are among the circuit's).
6. `CircuitWork`: the luck first. At each stage, each credited tile's `tileGame … .folded` is `Pinned` (its check is
   `same`), so `Accounting.SecurityProofs.TS.Headline.operandLuck_of_pointMass` (Lemma B at fibre 1) on `OperandHMu` and
   `LuckOfReceipt` with `ReceiptLuckAt` give `Pr[folded ∧ same] ≤ (μ + (1 − μ)·r_B)·Pr[folded]` (`r_B ≤ 1` by
   `Scheme.LuckProved`), and `GuessLuckAt` on `tileGame … .whole` gives `Pr[whole ∧ same] ≤ guess·Pr[whole]`; a tile of
   no credited kind is held to `1`; so `SplitStages.LuckAt (tileLuck …) (tileLuck0 …)`. Each credited tile's round is
   live (`Scope`'s `Declared.Live`), so its stage's coin is the verifier's, independent of the past, as `LuckAt` and the
   per-stage games read it; under `LayerCommitted` every value's layer cut is its IR cut (`layer_stageCuts`), so a run
   input is bound before its first reader's coin. Then `Accounting.SecurityProofs.TS.DrawnRightOff` at `S = 1` (its draw
   the layer's, `drawOf`) gives one event of probability at most `δ_bind + δ_ks` off which every accepted drawn tile is
   right at its committed slots, so `Of` gives the share game's pass (RT-I8's `compose_sampled` shape); `ServiceBudget`
   on `serviceGame` at `a`, `b` and `shareCone` (the same coins and pass) and `budget_tail` at `t = ln(1/δ_s)` bound
   `a·S_F + b·S_0 ≥ Λ·ln(1/δ_s)` by `δ_s`, `S_F`, `S_0` the folded and whole shares at `shareCone`. Off both,
   `UnitFixed` (its `WriterReads` by `layer_writerReads`) at `tileMode` and step 5's `Covers` give the good tiles' words
   at least `(1 − (c_F + c₀))·N_w` and the words not skipped whole at least `(1 − c₀)·N_w`, `c_F`, `c₀` the folded and
   whole shares at `tileCone` (`Σ_t tileWords = N_w`, by `Declared.Wf` and `units ≤ N_p`; a tile's cone holds itself,
   `ReflTransGen.refl`). `tileCone` and `shareCone` differ only on tiles of no words, so `c_F ≤ S_F + D_F` and
   `c₀ ≤ S_0 + D_0'` with `D_F + D_0' ≤ D₀`. Take the zero-word excess in the cheaper share: for `a ≤ b`, `d₀ = S_0` and
   `d_F = c_F + c₀ − S_0`, which keeps `d_F + d₀ = c_F + c₀`, `d₀ ≤ c₀` and `a·d_F + b·d₀ ≤ a·S_F + b·S_0 + a·D₀`; for
   `b < a`, `d_F = S_F` and `d₀ = c_F + c₀ − S_F`, at `b·D₀`. So `WorkAt` at
   `a·d_F + b·d₀ < Λ·ln(1/δ_s) + min(a, b)·D₀`.
7. `CreditOfWork`: `SheetDIdealFloor`'s reduction of (I1)–(I5) to A1–A7 on the one run, then `floor_of_price`'s
   identity with FOLD-OPT at the folded words (`FoldOptFolded`), which prices the deficit at `a·d_F + b·d₀ < r`, and
   `T_cred > 0` (`Model.Wf`) to clear the division.
8. `CircuitCredit`: step 6 at `a = aOf`, `b = bOf` (`≥ 0` by `stars_nonneg` and `Model.Wf`), and step 7 on each run
   off its event at `r = rOf`; the certificate is issued at `.live`.
9. `CircuitWorkSeeded`: `SeedExtractedAt` gives one event `Off` of probability at most `ε_id(T, |J|)` off which
   every round's `sq a r` is a query. Per round `r` and mode, `SeedLuckAt` gives the queries' gains `g` and `Break`;
   off `Off` and `Break` round `r`'s gain is at most `g (sq a r)`, so at most `max_q g q`. Where `θ_r > 0`
   (`seedTheta r` folded, `wholeTheta r` whole), GRIND's `Accounting.SecurityProofs.TS.GrindPremiumAt` (at `Ω₀` the
   window's other answers, `redraw`'s one answer the query's, `M = 1`, its own break event empty) bounds
   `∃ q, ln(T_q/δ_g)/θ_r ≤ g q` by `δ_g`, so round `r`'s gain `L − κ·U` is below `π*_r` except with probability
   `δ_g + δ_b`, off `Off`; where `θ_r = 0`, `roundPremium` is the round's weight (`roundCone`), which bounds `L` and so
   the gain (`κ ≥ 0`). A union over the `rounds` rounds and the two modes gives `L_F ≤ 4·U_F + Π_F` and
   `L_0 ≤ κ₀·U_0 + Π_0` (`premiumOf` at `seedTheta` and at `wholeTheta`) off `ε_id + 2·rounds·(δ_g + δ_b)`, with no
   stage order: each round's moment fixes every other answer of the window, so
   a mode that reads a later round's answers is in it. Given the rounds' coins, the service's live draw of a uniform
   `K_d`-subset misses the tiles whose checked values aren't honest with probability at most
   `(1 − ρ·U)^{K_d} ≤ e^{−ρ·K_d·U}`, so an accepted run with `ρ·K_d·U ≥ ln(1/δ_s)` has probability at most `δ_s`.
   Off these, `DrawnRightOff` on `Declared.atDraw`'s layer (its `MovedIn`, `JointSound`, `Pins`) and `Of` give the
   share game's pass; `UnitFixed` and step 5's `Covers` on `Declared.atIR`'s layer (`LayerCommitted` by
   `layerCommitted_atIR`) at `jointMode` give the good tiles' words as in step 6. The two layers differ only in their
   cuts, and `UnitFixed`, `Covers` and the game read none of a layer's committed values, so they combine: `WorkAt` at
   `a·S_F + b·S_0 ≤ max(5a, (1 + κ₀)·b)·U + a·Π_F + b·Π_0`, strictly below `rOfSeeded`'s first three terms
   (`a > 0`), with `D₀` as in step 6. `SeedNamed` (in `ScopeSeeded`) is read by no step: it is the protocol fact under
   which the `UnitFixed` premise on `Declared.atIR`'s layer can hold, as `LayerCommitted` is for the live pair.
10. `CircuitCreditSeeded`: step 9 at `a = aOf`, `b = bOf`, and step 7 on each run off its event at `r = rOfSeeded`;
   the certificate is issued at `.seeded`.
11. `BindingOfECR`: `Accounting.SecurityProofs.TS.DrawnBindingOfFork` gives `Pr[MovedIn] ≤ |L|·√Pr[fork]`; AM–GM,
   `|L|·√x ≤ δ' + |L|²·x/(4δ')`; then `Pr[fork] ≤ F.Pr H + κ` and, where `Claims.ExpectedCR H F` holds,
   `F.Pr H ≤ E[cost]/2^256`, through `ENNReal.ofReal`'s monotonicity and `ofReal_add_le`; where it fails, the right side
   is `⊤` unless `L` is empty, where `MovedIn` can't hold (no slot to move).
12. `WholeLuckOfEntropy`: per round take `WholeEntropyAt`'s `Wh`, `Lk` and `Break`, and
   `g q a = Σ_{v ∈ Wh q a} w_v·((1 + κ₀)·[v ∈ Lk q a] − κ₀)`. Off `Break` at `sq a r = some q` the round's whole
   tiles are `Wh q a` and `same` lands in `Lk q a`, so `roundGain ≤ g q a` (`w ≥ 0`, `1 + κ₀ ≥ 0`). `Wh q` doesn't
   read `q`'s answer, so on `redraw a r q b` the sum is over one set `W`, and with `x_v = θ_r·w_v`
   `exp(θ_r·g) = Π_{v ∈ W} (e^{−κ₀x_v} + [v ∈ Lk]·c_v)`, `c_v = e^{x_v} − e^{−κ₀x_v} ≥ 0`
   (`Finset.prod_add`), `Σ_{S ⊆ W} Π_{v ∈ S} c_v·[S ⊆ Lk]·Π_{v ∉ S} e^{−κ₀x_v}`. Summing over `b`, the entropy
   conjunct bounds each `Σ_b [S ⊆ Lk]` by `|B|·e^{−Σ_S h}`, and the sum folds back to
   `|B|·Π_v (e^{−h_v}·e^{x_v} + (1 − e^{−h_v})·e^{−κ₀x_v})`, each factor in `[0, 1]`.
-/

namespace Accounting.SecurityProofs.Compose

open Accounting.Compose

/-! ## The window's proofs' layer, from the IR -/

/-- A gate of the circuit is one of its own unit's gates. -/
theorem mem_unitGates {ι : Type} (Cd : Declared ι) {i : ℕ} (h : i < Cd.dag.length) :
    i ∈ unitGates Cd (Cd.unit i) :=
  List.mem_filter.mpr ⟨List.mem_range.mpr h, by simp⟩

/-- Under `Declared.Wf` every unit's round is one of the circuit's. -/
theorem roundOf_lt {ι : Type} {Cd : Declared ι} (hw : Cd.Wf) (hr : 0 < Cd.rounds) (u : ℕ) :
    Cd.roundOf u < Cd.rounds := by
  unfold Declared.roundOf
  rcases hl : unitGates Cd u with _ | ⟨i, l⟩
  · exact hr
  · have hi : i ∈ unitGates Cd u := by rw [hl]; exact List.mem_cons_self
    exact hw.2.1 _ (hw.1 i (List.mem_range.mp (List.mem_filter.mp hi).1)).1

section Layer

variable {ι C Ξ V : Type} (Cd : Declared ι) {Np Kd : ℕ} (hr : 0 < Cd.rounds) (hK : Kd ≤ Np)
  (pr : ProofRun C (Equiv.Perm (Fin Np) × Ξ) Cd.rounds Kd Np V)

theorem layer_draw : (Cd.layer hr hK pr).draw = drawOf hK := rfl

theorem layer_src (ℓ : ℕ) : (Cd.layer hr hK pr).src ℓ =
    if h : ℓ < Cd.dag.length ∧ Cd.binding ℓ ≠ some .coin ∧ Cd.unit ℓ < Np then some (0, ⟨Cd.unit ℓ, h.2.2⟩)
    else none := rfl

/-- Every one of tile `t`'s IR slots is one of the layer's. -/
theorem layer_pos_of_mem (w : Fin 1) (t : Fin Np) (ℓ : ℕ) (h : ℓ ∈ Cd.slots t.val) :
    ∃ k, (Cd.layer hr hK pr).pos w t k = ℓ := by
  obtain ⟨k, hk⟩ := List.mem_iff_getElem?.mp h
  refine ⟨k, ?_⟩
  show (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) = ℓ
  rw [hk, Option.getD_some]

/-- Every one of the layer's slots of tile `t` is one of its IR slots, or no position when it has none. -/
theorem layer_pos_mem (w : Fin 1) (t : Fin Np) (k : ℕ) :
    (Cd.layer hr hK pr).pos w t k ∈ Cd.slots t.val ∨
      (Cd.slots t.val = [] ∧ (Cd.layer hr hK pr).pos w t k = Cd.dag.length) := by
  show (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) ∈ Cd.slots t.val ∨
    (Cd.slots t.val = [] ∧
      (Cd.slots t.val)[k]?.getD ((Cd.slots t.val).headD Cd.dag.length) = Cd.dag.length)
  cases hl : Cd.slots t.val with
  | nil => exact Or.inr ⟨rfl, rfl⟩
  | cons a l =>
    refine Or.inl ?_
    cases hk : (a :: l)[k]? with
    | none => exact List.mem_cons_self
    | some b => exact List.mem_iff_getElem?.mpr ⟨k, hk⟩

theorem layer_stageCuts (hc : LayerCommitted Cd) : StageCuts Cd (Cd.layer hr hK pr) := fun ℓ hℓ => by
  show min (max (Cd.cutOf ℓ) (Cd.commitRound ℓ)) Cd.rounds = min (Cd.cutOf ℓ) Cd.rounds
  rw [Nat.max_eq_left (hc ℓ hℓ)]

/-- The circuit cut at the IR commits by the IR's cuts. -/
theorem layerCommitted_atIR : LayerCommitted Cd.atIR := fun _ _ => Nat.zero_le _

/-- The layer of the circuit committed at the draw cuts every position at the draw. -/
theorem layer_atDraw_cut (ℓ : ℕ) : (((Cd.atDraw.layer hr hK pr).cut ℓ).2 : ℕ) = Cd.rounds :=
  Nat.min_eq_right (Nat.le_max_right _ _)

theorem layer_stageCoins (hw : Cd.Wf) : StageCoins Cd (Cd.layer hr hK pr) :=
  fun t => Nat.min_eq_left (Nat.le_sub_one_of_lt (roundOf_lt hw hr t.val))

theorem layer_stageSlots : StageSlots Cd (Cd.layer hr hK pr) := by
  refine And.intro ?_ fun t k => layer_pos_mem Cd hr hK pr 0 t k
  intro ℓ u hu
  rw [layer_src] at hu
  split at hu
  · rename_i h
    obtain rfl := Option.some.inj hu
    exact And.intro (And.intro h.1 h.2.1) rfl
  · cases hu

theorem layer_writerReads : (Cd.layer hr hK pr).WriterReads := by
  intro ℓ u hu
  rw [layer_src] at hu
  split at hu
  · rename_i h
    obtain rfl := Option.some.inj hu
    have hm : ℓ ∈ Cd.slots (Cd.unit ℓ) := List.mem_append_left _ (mem_unitGates Cd h.1)
    exact layer_pos_of_mem Cd hr hK pr 0 ⟨Cd.unit ℓ, h.2.2⟩ ℓ hm
  · cases hu

theorem layer_stageComplete (hw : Cd.Wf) (hN : Cd.units ≤ Np) : StageComplete Cd (Cd.layer hr hK pr) := by
  refine And.intro ?_ fun t ℓ h => layer_pos_of_mem Cd hr hK pr 0 t ℓ h
  intro ℓ hℓ
  have hu : Cd.unit ℓ < Np := Nat.lt_of_lt_of_le (hw.1 ℓ hℓ.1).2 hN
  exact ⟨(0, ⟨Cd.unit ℓ, hu⟩),
    (layer_src Cd hr hK pr ℓ).trans (dite_eq_left (And.intro hℓ.1 (And.intro hℓ.2 hu))), rfl⟩

end Layer

/-- A split-stage window's service game is the joint game at its joint run. -/
theorem serviceGame_eq {C Ξ : Type} {N M : ℕ} (stg : Fin N → ℕ) (W : SplitStages C (Fin N) M) (w : Fin N → ℝ)
    (a b : ℝ) (K : ℕ) : serviceGame (Ξ := Ξ) stg W w a b K = jointGame (W.joint stg) w a b K := rfl

/-- A split-stage window's modes are its joint run's. -/
theorem tileMode_eq {ι C B : Type} {Np : ℕ} (Cd : Declared ι) (W : SplitStages C (Fin Np) Cd.rounds) :
    tileMode (B := B) Cd W = jointMode (W.joint (tileStage Cd Np)) := rfl

/-! ## The guarantees -/

theorem SplitRateClosed : Accounting.Guarantees.Compose.SplitRateClosed := by
  sorry

theorem SplitRateExp : Accounting.Guarantees.Compose.SplitRateExp := by
  sorry

theorem SplitRateMerged : Accounting.Guarantees.Compose.SplitRateMerged := by
  sorry

theorem ServiceBudget : Accounting.Guarantees.Compose.ServiceBudget := by
  sorry

theorem ServiceCovers : Accounting.Guarantees.Compose.ServiceCovers := by
  sorry

theorem BindingOfECR : Accounting.Guarantees.Compose.BindingOfECR := by
  sorry

theorem WholeLuckOfEntropy : Accounting.Guarantees.Compose.WholeLuckOfEntropy := by
  sorry

theorem CircuitWork : Accounting.Guarantees.Compose.CircuitWork := by
  sorry

theorem CreditOfWork : Accounting.Guarantees.Compose.CreditOfWork := by
  sorry

theorem CircuitCredit : Accounting.Guarantees.Compose.CircuitCredit := by
  sorry

theorem CircuitWorkSeeded : Accounting.Guarantees.Compose.CircuitWorkSeeded := by
  sorry

theorem CircuitCreditSeeded : Accounting.Guarantees.Compose.CircuitCreditSeeded := by
  sorry

end Accounting.SecurityProofs.Compose
