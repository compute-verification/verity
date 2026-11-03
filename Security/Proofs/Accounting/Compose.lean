import Proofs.Accounting.Compose.ServiceCovers
import Definitions.Accounting.Guarantees.Compose
import Proofs.Accounting.TS
import Proofs.Accounting.Compose.ServiceBudget
import Proofs.Accounting.Compose.SplitRateMerged
import Proofs.Accounting.Compose.BindingOfECR
import Proofs.Accounting.Compose.CircuitWork
import Proofs.Accounting.Compose.WholeLuckOfEntropy
import Proofs.Accounting.Compose.CircuitWorkSeeded
import Proofs.Pouw.RelativeSecurity

/-!
# PoUW's credit on an arbitrary declared circuit: proofs

Every guarantee here is proved: the three rates, `ServiceBudget`, `ServiceCovers`, `CircuitWork`, `CircuitWorkSeeded`,
`BindingOfECR` and `WholeLuckOfEntropy` (steps 1–6, 9, 11 and 12, each in its own file under
`Proofs/Accounting/Compose/`), and `CreditOfWork` (step 7), `CircuitCredit` (step 8) and `CircuitCreditSeeded`
(step 10) below. What the window's proofs' layer meets is proved: its static half is defined from the IR
(`Declared.layer`), so `StageCuts` (at every value committed by its IR cut, which every one is at `Declared.atIR`),
`StageCoins` (under `Declared.Wf`), `StageSlots`, `WriterReads` and `StageComplete` (under `Declared.Wf` and
`units ≤ N_p`) are lemmas below, not premises, as are the draw cut of `Declared.atDraw`'s layer and `serviceGame`,
`tileMode` as `jointGame`, `jointMode` at a split-stage window's joint run.

1. `SplitRateClosed` (proved, `Compose/SplitRateClosed.lean`), from step 2. A larger rate keeps `SplitRateAt` at the
   same `ν, ν₀`, and each closed branch is at least the sharp one: at `c = Kρw_max` and `q = p + (1 − p)e^{−c}`,
   `−ln q ≥ 1 − q` and `(1 + c)e^{−c} ≤ 1`. At `w_max = 0`, `ν = ν₀ = 0` do.
2. `SplitRateExp` (proved, `Compose/SplitRateExp.lean`). `(1 − t)^K ≤ e^{−Kt}` bounds the left side by the exponential
   of a linear form on the simplex of `(W_F − L_F, L_F, W_0 − L_0, L_0)`, and Jensen for `exp` at its vertices bounds
   that by the right side's interpolation. At the unlucky folded vertex this is `Λ` at least the folded branch, and at
   `L_F = w_max` an identity by `ν`'s definition; the whole vertices are the same at `b`, `p0`, `ν₀`.
3. `SplitRateMerged` (proved, `Compose/SplitRateMerged.lean`): step 1 at `p = p0 = q`, where the whole branch is the
   larger (`a ≤ b`), and lower lucks keep `SplitRateAt`.
4. `ServiceBudget` (proved, `Compose/ServiceBudget.lean`): given the stages' coins, a uniform `K`-subset of `N`
   misses the tiles whose checked values aren't honest with probability at most `(1 − ρ·(their share))^K`, which
   factors over the stages; `SplitRate` per stage at its share `≤ w_max`, its luck in expectation from `LuckAt`
   (`E[L_F] ≤ p·E[W_F]`, `E[L_0] ≤ p0·E[W_0]`), then backward induction over the stages, as
   `Accounting.SecurityProofs.TS.StageBudget`.
5. `ServiceCovers` (proved, `Compose/ServiceCovers.lean`): `Accounting.SecurityProofs.TS.DrawnLayer.covers_of_cone` at
   the tiles' words (`≥ 0`), at any joint run (the live statements use it at `SplitStages.joint`, by `serviceGame_eq`
   and `tileMode_eq`). The game's
   deficiency is the not-paid tiles' `tileCone` (the three modes, at `1` and `1`), so it is enough that a tile's layer
   cone is at most `N_w` times its `tileCone`: a layer edge `DrawnLayer.Late u v` at slot `k` has its position a value
   the prover writes and a gate of `u`'s unit (`layer_stageSlots`), and either one of `v`'s own gates (`u = v`) or an
   operand of one, read by a gate `j` of `v`'s, whose round is `v`'s (`Declared.Wf`, `layer_stageCoins`). Under
   `LayerCommitted` the position is committed by its IR cut, so `j`'s round is before that cut (`layer_stageCuts`),
   `Declared.Late` and a `Declared.UnitLate` edge, or it is a computed value only its own unit reads, so `j`'s unit,
   `v`, is `u` (a value committed late adds no edge out of its unit); along
   `ReflTransGen` the layer's cone lies in `Declared.Reaches`, and a tile's words are `N_w` times its weight (`0` when
   `N_w = 0`: a tile's words are among the circuit's).
6. `CircuitWork` (proved, `Compose/CircuitWork.lean`): the luck first. At each stage, each credited tile's
   `tileGame … .folded` is `Pinned` (its check is `same`), so
   `Accounting.SecurityProofs.TS.Headline.operandLuck_of_pointMass` (Lemma B at fibre 1) on `OperandHMu` and
   `LuckOfReceipt` with `ReceiptLuckAt` give `Pr[folded ∧ same] ≤ (μ + (1 − μ)·r_B)·Pr[folded]` (`r_B ≤ 1` by
   `Scheme.LuckProved`), and `GuessLuckAt` on `tileGame … .whole` gives `Pr[whole ∧ same] ≤ guess·Pr[whole]`; a tile of
   no credited kind is held to `1`; so `SplitStages.LuckAt (tileLuck …) (tileLuck0 …)`. `LuckAt` and the per-stage
   games read the stages' coins as `SplitStages`' uniform ones; `Scope`'s `Declared.Live` is read by no step: it is the
   protocol fact under which those are the service's coins, as `SeedNamed` is in step 9. Under `LayerCommitted` an
   input's layer cut is its IR cut (`layer_stageCuts`: an input has no second case), so a run input is bound before its
   first reader's coin. Then
   `Accounting.SecurityProofs.TS.DrawnRightOff` at `S = 1` (its draw the layer's, `drawOf`) gives one event of
   probability at most `δ_bind + δ_ks` off which every accepted drawn tile is right at its committed slots, so `Of`
   gives the share game's pass; `ServiceBudget` on `serviceGame` at `a`, `b` and `shareCone` (the same coins and pass)
   and `budget_tail` at `t = ln(1/δ_s)` bound `a·S_F + b·S_0 ≥ Λ·ln(1/δ_s)` by `δ_s`, `S_F`, `S_0` the folded and whole
   shares at `shareCone`. Off both,
   `UnitFixed` (its `WriterReads` by `layer_writerReads`) at `tileMode` and step 5's `Covers` give the good tiles' words
   at least `(1 − (c_F + c₀))·N_w`, `c_F`, `c₀` the folded and whole shares at `tileCone`, and the words not skipped
   whole at least `(1 − S_0)·N_w`, since a tile's words are at most `N_w` times its `shareCone` (a tile's cone holds
   itself, `ReflTransGen.refl`); both read `Σ_t tileWords ≥ N_w` (each credited word is in its unit's tile, by
   `Declared.Wf` and `units ≤ N_p`). `tileCone` and `shareCone` differ only on tiles of no words, so
   `c_F + c₀ = S_F + S_0 + Z_F + Z_0` with `Z_F + Z_0 ≤ D₀`, the folded and whole cones of those tiles. Take them in the
   cheaper share: for `a ≤ b`, `d_F = S_F + Z_F + Z_0` and `d₀ = S_0`; for `b < a`, `d_F = S_F` and
   `d₀ = S_0 + Z_F + Z_0`. Either keeps `d_F + d₀ = c_F + c₀` and `d₀ ≥ S_0`, so `WorkAt` at
   `a·d_F + b·d₀ < Λ·ln(1/δ_s) + min(a, b)·D₀`.
7. `CreditOfWork` (proved): `SheetDIdealFloor`'s reduction of (I1)–(I5) to A1–A7 on the one run
   (`facetCeilings_of_sheet`, `scalarRoute_half`), `price_of_parts` with FOLD-OPT at the folded words and `lam`
   (`price_of_parts_folded`, from `FoldOptFoldedAt`), `price_le_time`, and `identity` at `d̄ = ā = 0`; the deficit,
   `(d_F + d₀)·N_w·w* + d₀·(1 − μ_f)·N_w·φ*`, is `(a·d_F + b·d₀)·B_♭ < r·B_♭`, and `f_max > 0` (`Model.Wf`) clears
   the division.
8. `CircuitCredit` (proved): step 6 at `a = aOf`, `b = bOf` (`≥ 0` by `costs_nonneg`,
   from `stars_nonneg` and `Model.Wf`), and step 7 on each run off its event at `r = rOf`; the certificate is issued at
   `.live`.
9. `CircuitWorkSeeded` (proved, `Compose/CircuitWorkSeeded.lean`): `SeedExtractedAt` gives one event `Off` of
   probability at most `ε_id(T, |J|)` off which every round's `sq a r` is a query. Per round `r` and mode,
   `SeedLuckAt` gives the queries' gains `g` and `Break`; off `Off` and `Break` round `r`'s gain is at most
   `g (sq a r)`, so at most `max_q g q`. Where `θ_r > 0` (`seedTheta r` folded, `wholeTheta r` whole), GRIND's
   premium (`grind_le`) bounds `∃ q, ln(T_q/δ_g)/θ_r ≤ g q` by `δ_g`: redrawing the query's one answer is an
   involution of the window (`redraw_redraw`), so the window's mean of `e^{θ_r·g q}` is the mean of its moment over
   that answer, at most `1` (`sum_redraw`), and Markov and a union over the `T_q` queries give `δ_g`. So round `r`'s
   gain `L − κ·U` is below `π*_r` except with probability `δ_g + δ_b`, off `Off` (`gain_le_premium`); where
   `θ_r = 0`, `roundPremium` is the round's weight (`roundCone`), which bounds `L` and so the gain (`κ ≥ 0`). A union
   over the `rounds` rounds and the two modes gives `L_F ≤ 4·U_F + Π_F` and `L_0 ≤ κ₀·U_0 + Π_0` (`gainSum_le`,
   `premiumOf` at `seedTheta` and at `wholeTheta`) off `ε_id + 2·rounds·(δ_g + δ_b)`, with no stage order: each
   round's moment fixes every other answer of the window, so a mode that reads a later round's answers is in it.
   Given the rounds' coins, the service's live draw of a uniform `K_d`-subset misses the tiles whose checked values
   aren't honest with probability at most `(1 − ρ·U)^{K_d} ≤ e^{−ρ·K_d·U}`, so an accepted run with
   `ρ·K_d·U ≥ ln(1/δ_s)` has probability at most `δ_s` (`live_le`). Off these, `DrawnRightOff` on
   `Declared.atDraw`'s layer (its `MovedIn`, `JointSound`, `Pins`) and `Of` give the share game's pass; `UnitFixed`
   and step 5's `Covers` on `Declared.atIR`'s layer (`LayerCommitted` by `layerCommitted_atIR`) at `jointMode` give
   the good tiles' words as in step 6 (`WorkProof.work_of_bounds`). The two layers differ only in their cuts, and
   `UnitFixed`, `Covers` and the game read none of a layer's committed values, so they combine: each mode's share is
   its `L − κ·U` plus `(1 + κ)·U` (`share_split`), so `a·S_F + b·S_0 ≤ max(5a, (1 + κ₀)·b)·U + a·Π_F + b·Π_0`,
   strictly below `rOfSeeded`'s first three terms (`a > 0`), with `D₀` as in step 6. `SeedNamed` (in `ScopeSeeded`)
   is read by no step: it is the protocol fact under which the `UnitFixed` premise on `Declared.atIR`'s layer can
   hold, as `LayerCommitted` is for the live pair.
10. `CircuitCreditSeeded` (proved): step 9 at `a = aOf`, `b = bOf`, and step 7 on each run off its event at
   `r = rOfSeeded`; the certificate is issued at `.seeded`.
11. `BindingOfECR` (proved, `Compose/BindingOfECR.lean`): `Accounting.SecurityProofs.TS.DrawnBindingOfFork` gives
   `Pr[MovedIn] ≤ |L|·√Pr[fork]`; AM–GM, `|L|·√x ≤ δ' + |L|²·x/(4δ')`; then `Pr[fork] ≤ F.Pr H + κ` and, where
   `Claims.ExpectedCR H F` holds, `F.Pr H ≤ E[cost]/2^256`, through `ENNReal.ofReal`'s monotonicity and
   `ofReal_add_le`; where it fails, the right side is `⊤` unless `L` is empty, where `MovedIn` can't hold (no slot to
   move).
12. `WholeLuckOfEntropy` (proved, `Compose/WholeLuckOfEntropy.lean`): per round take `WholeEntropyAt`'s `Wh`, `Lk`
   and `Break`, and `g q a = Σ_{v ∈ Wh q a} w_v·((1 + κ₀)·[v ∈ Lk q a] − κ₀)`. Off `Break` at `sq a r = some q` the
   round's whole tiles are `Wh q a` and `same` lands in `Lk q a`, so `roundGain ≤ g q a` (`w ≥ 0`, `1 + κ₀ ≥ 0`).
   `Wh q` doesn't read `q`'s answer, so on `redraw a r q b` the sum is over one set `W`, and with `x_v = θ_r·w_v`
   (`moment_le`) `exp(θ_r·g) = Π_{v ∈ W} ([v ∈ Lk]·c_v + e^{−κ₀x_v})`, `c_v = e^{x_v} − e^{−κ₀x_v} ≥ 0`, which
   `Finset.prod_add` expands to `Σ_{S ⊆ W} [S ⊆ Lk]·Π_{v ∈ S} c_v·Π_{v ∈ W \ S} e^{−κ₀x_v}`. Summing over `b`, the
   entropy conjunct bounds each `Σ_b [S ⊆ Lk]` by `|B|·e^{−Σ_S h}`, and the sum folds back to
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

theorem layer_stageCuts : StageCuts Cd (Cd.layer hr hK pr) := fun ℓ _ hc => by
  show min (max (Cd.cutOf ℓ) (Cd.commitRound ℓ)) Cd.rounds = min (Cd.cutOf ℓ) Cd.rounds
  rw [Nat.max_eq_left hc]

/-- The circuit cut at the IR commits by the IR's cuts. -/
theorem layerCommitted_atIR : LayerCommitted Cd.atIR := fun _ _ => Or.inl (Nat.zero_le _)

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
  exact Rates.splitRateClosed

theorem SplitRateExp : Accounting.Guarantees.Compose.SplitRateExp := by
  exact Rates.splitRateExp

theorem SplitRateMerged : Accounting.Guarantees.Compose.SplitRateMerged := by
  exact Rates.splitRateMerged

theorem ServiceBudget : Accounting.Guarantees.Compose.ServiceBudget := by
  exact Budget.serviceBudget

theorem ServiceCovers : Accounting.Guarantees.Compose.ServiceCovers := by
  exact CoversProof.serviceCovers

theorem BindingOfECR : Accounting.Guarantees.Compose.BindingOfECR := by
  exact Binding.bindingOfECR

theorem WholeLuckOfEntropy : Accounting.Guarantees.Compose.WholeLuckOfEntropy := by
  exact Whole.wholeLuckOfEntropy

theorem CircuitWork : Accounting.Guarantees.Compose.CircuitWork := by
  exact WorkProof.circuitWork

theorem CreditOfWork : Accounting.Guarantees.Compose.CreditOfWork := by
  intro ι K P T L U F X _ _ _ _ _ _ _ _ _ _ Sc Cd Sh Z A folded lam r hFits hMS hA hClock hDisp hTen hPipe hSmem
    hWords hSem hFold hFill hwork
  obtain ⟨h, hZ, hq, hqT, hb⟩ := hMS
  obtain ⟨-, hf, -, hB, -, -, -, -, -, -, -, -, hmu1, -⟩ := id hZ
  obtain ⟨-, hw0, hphi0, -⟩ := Accounting.SecurityProofs.TS.stars_nonneg Z (floorOf Sc Cd) lam hZ h
  obtain ⟨dF, d0, -, -, hfix, hfold, hlt⟩ := hwork
  rw [← hFits.2] at hfix hfold
  have hM : Accounting.Assumptions.TS.FacetCeilings Z.G A.run := by
    rw [hFits.1]
    exact Accounting.SecurityProofs.TS.facetCeilings_of_sheet Sh _
      (Accounting.SecurityProofs.TS.run_nonneg_of_split hA) hClock hDisp hTen hPipe hSmem
  have hΦ := (Accounting.SecurityProofs.TS.price_of_parts_folded Z (floorOf Sc Cd) A lam folded h hZ hA hWords.1
    (Accounting.SecurityProofs.TS.scalarRoute_half Z A hA hWords hq hqT hb) hSem.2 hSem.1 hFold hFill).trans
    (Accounting.SecurityProofs.TS.price_le_time Z.G lam A.run h hf hM)
  have hid := (div_eq_iff hB.ne').mp (Accounting.SecurityProofs.TS.identity Z (floorOf Sc Cd) lam 0 0 hZ)
  simp only [Accounting.TS.Model.epsLuck, Accounting.TS.Model.epsAdapt, zero_mul, sub_zero] at hid
  have h1 := mul_le_mul_of_nonneg_right hfix hw0
  have h2 := mul_le_mul_of_nonneg_right (mul_le_mul_of_nonneg_left hfold (sub_nonneg.mpr hmu1)) hphi0
  have hNW : Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd) * Z.bnd (floorOf Sc Cd) = Z.Nw * Z.wStar lam :=
    div_mul_cancel₀ _ hB.ne'
  have hc : (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd) * Z.bnd (floorOf Sc Cd) =
      (1 - Z.muF) * Z.Nw * Z.phiStar lam := div_mul_cancel₀ _ hB.ne'
  have hab : (aOf Sc Cd Z lam * dF + bOf Sc Cd Z lam * d0) * Z.bnd (floorOf Sc Cd) =
      (dF + d0) * (Z.Nw * Z.wStar lam) + d0 * ((1 - Z.muF) * Z.Nw * Z.phiStar lam) := by
    show (Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd) * dF + (Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd) +
      (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd)) * d0) * Z.bnd (floorOf Sc Cd) = _
    linear_combination (dF + d0) * hNW + d0 * hc
  have hrB := mul_lt_mul_of_pos_right hlt hB
  show (1 - (Z.epsSkip (floorOf Sc Cd) lam + Z.epsCert (floorOf Sc Cd) lam + Z.epsRoute (floorOf Sc Cd) lam +
    Z.epsMand (floorOf Sc Cd) lam + r)) * (Z.bnd (floorOf Sc Cd) / Z.G.fmax) ≤ A.run.time
  rw [← mul_div_assoc, div_le_iff₀ hf]
  linarith

/-- The circuit's costs `aOf` and `bOf` are `≥ 0` in the model's ranges. -/
theorem costs_nonneg {ι K R T L U F X : Type} [Fintype K] [Fintype R] [Fintype T] [Fintype L] [Fintype U]
    [Fintype F] [Fintype X] [Nonempty F] [Nonempty X] (Sc : Scheme K) (Cd : Declared ι)
    (Z : Accounting.TS.Model K R T L U F X) (lam : R → ℝ) (h : ModelScope Sc Cd Z lam) :
    0 ≤ aOf Sc Cd Z lam ∧ 0 ≤ bOf Sc Cd Z lam := by
  have hst := Accounting.SecurityProofs.TS.stars_nonneg Z (floorOf Sc Cd) lam h.2.1 h.1
  obtain ⟨-, -, hNw, hbnd, -, -, -, -, -, -, -, -, hmu, -⟩ := h.2.1
  have ha : 0 ≤ aOf Sc Cd Z lam := by
    show 0 ≤ Z.Nw * Z.wStar lam / Z.bnd (floorOf Sc Cd)
    exact div_nonneg (mul_nonneg hNw.le hst.2.1) hbnd.le
  refine ⟨ha, ?_⟩
  show 0 ≤ aOf Sc Cd Z lam + (1 - Z.muF) * Z.Nw * Z.phiStar lam / Z.bnd (floorOf Sc Cd)
  exact add_nonneg ha (div_nonneg (mul_nonneg (mul_nonneg (sub_nonneg.mpr hmu) hNw.le) hst.2.2.1) hbnd.le)

theorem CircuitCredit : Accounting.Guarantees.Compose.CircuitCredit := by
  intro ι K P T L U F X C Ξ V _ _ _ _ _ _ _ _ _ _ _ _ _ Sc Cd Sh Z H Np Kd hN hr hK W Pl pr folded lam
    ρ p p0 wmax Λ δs δbind δks win hS hρ hρN hw hlk hΛ hSR hgame hOf hMoved hJS hPins hMu hRB hG hUF hFits hMS
    hSplit hClock hDisp hTen hPipe hSmem hWords hSem hFold hFill
  obtain ⟨ha, hb⟩ := costs_nonneg Sc Cd Z lam hMS
  unfold CreditCertificate.Holds
  refine le_trans (Pouw.RelativeSecurity.prob_mono ?_)
    (CircuitWork Sc Cd hN hr hK W Pl pr folded ρ p p0 _ _ wmax Λ δs δbind δks hS hρ hρN hw hlk ha hb hΛ hSR
      hgame hOf hMoved hJS hPins hMu hRB hG hUF)
  rintro x ⟨hacc, htime⟩
  exact ⟨hacc, fun hwork => (not_le.mpr htime) (CreditOfWork Sc Cd Sh Z (Pl.run x) (folded x) lam _ hFits hMS
    (hSplit x) (hClock x) (hDisp x) (hTen x) (hPipe x) (hSmem x) (hWords x) (hSem x) (hFold x) (hFill x) hwork)⟩

theorem CircuitWorkSeeded : Accounting.Guarantees.Compose.CircuitWorkSeeded := by
  exact Seeded.circuitWorkSeeded

theorem CircuitCreditSeeded : Accounting.Guarantees.Compose.CircuitCreditSeeded := by
  intro ι K P T L U F X B Ξ V _ _ _ _ _ _ _ _ _ _ _ _ _ Tq Sc Cd Sh Z H Np Kd hN hr hK R Pl pr folded lam sq εid
    T J ρ κ0 δs δg δb δbind δks win hS hδg hKd hρ hρN ha hκ hgame hOf hMoved hJS hPins hSE hSLF hSLW hUF hFits hMS
    hSplit hClock hDisp hTen hPipe hSmem hWords hSem hFold hFill
  have hb := (costs_nonneg Sc Cd Z lam hMS).2
  unfold CreditCertificate.Holds
  refine le_trans (Pouw.RelativeSecurity.prob_mono ?_)
    (CircuitWorkSeeded Sc Cd hN hr hK R Pl pr folded sq εid T J ρ _ _ κ0 δs δg δb δbind δks hS hδg hKd hρ hρN ha hb
      hκ hgame hOf hMoved hJS hPins hSE hSLF hSLW hUF)
  rintro x ⟨hacc, htime⟩
  exact ⟨hacc, fun hwork => (not_le.mpr htime) (CreditOfWork Sc Cd Sh Z (Pl.run x) (folded x) lam _ hFits hMS
    (hSplit x) (hClock x) (hDisp x) (hTen x) (hPipe x) (hSmem x) (hWords x) (hSem x) (hFold x) (hFill x) hwork)⟩

end Accounting.SecurityProofs.Compose
