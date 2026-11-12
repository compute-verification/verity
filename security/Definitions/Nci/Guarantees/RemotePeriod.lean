module

public import Definitions.Nci.Guarantees.Remote

/-!
# The compute dampener, remote protocol, over a monitoring period: the statement

`DampenerRemote` over a monitoring period of `T` windows (results are per monitoring period), with its
premises on the protocols, `PouwSound`, `SampledDetection` and one declaration's count verdict, replaced by what each
protocol's own guarantee gives on live coins and by each day's verdict. A window is one round of the verifier's live coins: window `t`'s
coins `Ω t` are uniform and independent of the other windows', and a run of the period is `ξ : (t : Fin T) → Ω t`. Each
leg is stated as `Proofs.Flock.Soundness.Audit.Periods` states a period's bound: at any value of the other windows'
coins, over window `t`'s own (`∀ ξ, P_a[E_t(ξ[t ↦ a])] ≤ p`).

* **PoUW** (in place of `PouwSound`): window `t`'s certificate is wrong (`wrong t`: accepted with the window's time
  below `(1 − ε)·T_cred`) with probability at most `p` over its coins. That is
  `Accounting.SecurityProofs.Compose.CircuitCredit`'s conclusion, `p = δ_s + δ_bind + δ_ks`, at window `t`'s play, for
  a run whose window `t` at each value of the other windows' coins is a play meeting `CircuitCredit`'s premises; that
  bridge is not stated, and `δ_bind` and `δ_ks` have no value yet (`PouwSound`), so `p` is a parameter with none.
  `ComposeMonitoringPeriod.CircuitCreditPeriod` credits live words over a period's accepted prefix, not time, so it
  does not give `wrong`'s form.
* **Sampled proofs** (in place of `SampledDetection`): a window's faulty blocks are fixed before its coins (its incorrect
  proof blocks at registration, `wrongRegZ`), and its sample and verdict read its coins and the past. Two per-window bounds,
  each of which a window of `Flock.Guarantees.IntegrityPeriod` has at the strategy that plays it: the sample misses all
  `m` of the window's faulty blocks with probability at most `(1 − q)^m` (`Flock.Guarantees.HiddenWindow.drawMiss_le`,
  under A3: a uniform `kd`-subset of `n` units misses `m` fixed units with probability
  `C(n − m, kd)/C(n, kd) ≤ (1 − kd/n)^m`), and the window is accepted with a sampled faulty block with probability at most
  `ε_ks,t + δ_link,t` (`HiddenWindow.accepted_drawnWrong_le`, from `Flock.SecurityProofs.EndToEndHiddenDrawn`). That
  the deployment's window `t`, at each value of the other windows' coins, is such a window with its faulty blocks as
  its incorrect proof blocks is a bridge not stated either: `IntegrityPeriod` plays the windows as one sequential game, not over a
  product of window coins. No guarantee implies `SampledDetection`'s per-block form: it conditions each block on its
  own window's other blocks, and it folds the knowledge term into `q`, where the protocol's is additive per window. The
  statement does not need it: a block's time is its window's, and each identity's escaped blocks are bounded at the
  predictable time its faulty blocks reach `K + 1`, the stopped, per-identity form of `IntegrityHorizon`'s own
  argument, which `FlockSoundness.Audit.Periods.prCoin_forall_sum_le_pow` bounds by `(1 − q)^(K+1)`. That is a tail
  bound, so the escape term needs no Markov's inequality: `J (1 − q)^(K+1)` at an escape share `J K B/C`, against
  `DampenerRemote`'s `J ((1 − q)/q) B/(θC)`.
* **The halt**: `TerminalDetection` stays a named premise. The halt is code (`Fleet`'s R5-all in
  `verity/protocols/spot/verifier/service.py`, its verifier `Spot.Fleet`), and
  `Properties.Spot` proves what it does (`FleetHaltIsFinal`, `NoAcceptAfterHalt`), but not this form: it halts
  an operator at its `budget`-th failure, not its first (`budget ≥ 1`); it ends the operator's registrations, not its
  hardware's use; and nothing states the verdict's time `Δ` after the detected block.
* **The inference-only policy**: each day `i` of the period declares `d_i`, and the statement takes the day's verdict
  exactly, `CompliesAtMost c k n_i`, at most `n_i` proof blocks of a `(c, k)`-partition at the day's public block count
  `n_i`: what a published `complies` of the count of `(c, k)`-inference blocks (`NciCheck.Count.check`) gives the Boolean
  declaration its Program lowers to (`NciCheck.Properties.DescCountsCK`, stated in `security/Properties` and not yet
  proved). Reading the check itself needs the code (package `security_properties`), so the statement takes its verdict.
  The day's charged work meets (H) and (E) (`Nci.Cut.Assumptions.Hardness`, `Exports`), and `Nci.Cut.Guarantees.CountHE`
  bounds its MACs by `n_i (k + α) + (k/2) √(n_i O_i / (β_w β_x))` (`Nci.Cut.workHE`). No rate form, padding or other
  premise on the circuit enters. What is open on this leg is `DescCountsCK`'s proof and the bridge from the deployment's
  days to the declarations the count reads: it is not stated.

Every other premise is `DampenerRemote`'s: the model's accounting of the four sources of counted multiplications, the
capacities `C` and `U` (`s = U/(C + U)`), and the hardware's rate `Φ` per identity. The developer's identities `I`
are fixed over the period: nothing here models or rests on re-registration.

`Nci.SecurityProofs.DampenerRemotePeriod` proves it. `DampenerRemotePeriod_honest` and `DampenerRemotePeriod_nontrivial`
are its witnesses: its premises hold at a constructed honest developer with one declared day, and at `AcceptAll`, a
PoUW verifier that accepts every certificate, every other premise holds while PoUW's per-window bound and the conclusion
fail.
-/

@[expose] public section

open Verity.Game renaming prPMFReal → prob

namespace Nci.Guarantees

open Verity Nci.Dampener

/-- **The compute dampener, remote protocol, over a monitoring period.** A monitoring period of `T` windows, window `t`
with its own uniform coins `Ω t`, independent of the others' (a window is a round of live coins: the deployment's
window, or a finer cut of it where its PoUW and sampled-proofs coins are flipped at different times). Every bound on a
window holds at any value `ξ` of the other windows' coins, over its own (`ξ[t ↦ a]`, `a` uniform on `Ω t`).

* PoUW certifies window `t` at capacity `cap_t ≥ 0`, `∑ cap_t ≤ C`, its certificate wrong (`wrong t`) with probability
  at most `p ≥ 0`: `CircuitCredit`'s conclusion at window `t`'s play.
* Sampled proofs: the developer's incorrect proof blocks `bad ξ` (block `b` in window `win b`, skipping `0 ≤ w_b ≤ B` of
  declared work) are fixed before their window's coins; the units the window samples (`drawn`) and the faulty blocks it
  detects (`detected`, those of a window flock-verify rejects) read only its coins and the earlier windows'. The sample
  misses the window's `m` faulty blocks with probability at most `(1 − q)^m`, `0 ≤ q ≤ 1` (A3, `q ≤ kd/n`), and the
  window detects none of them while it samples one with probability at most `ks t` (`EndToEndHiddenDrawn`,
  `ks t = ε_ks + δ_link`).
* Window `t` is run by identity `idW t`, one of the `J = |I|` the developer controls, at time `τw t`; a block runs at
  its window's time, and each identity's windows run in their order. Detection is terminal at a lag `Δ ≥ 0`
  (`TerminalDetection`), and one identity's faulty blocks within any span of `Δ` skip at most `Φ Δ`, `Φ H ≤ C` over the
  period's length `H > 0`, as in `VerdictLag`.
* The inference-only policy: day `i` of the period's days `R` declares `d_i`, whose verdict is `CompliesAtMost c k n_i`
  at its public block count `n_i`. Its charged work `W_i` (the matmuls `γ` of its circuit, `N_i` MACs and `O_i` output
  wires) meets (H) at the costs `β_w, β_x > 0` and the read cap `k`, and (E) at `b > 0`, the values `val_i` its words
  carry one-to-one on each matmul's words, no two matmuls sharing a pair. So the period's declared MACs are at most
  `X = ∑_i (n_i (k + α) + (k/2) √(n_i O_i / (β_w β_x)))`, `α = alphaHE k (β_w β_x) b` (`CountHE` at each day).
* The counted multiplications `M` are as in `DampenerRemote`, with the declared days' `∑ N_i`: unmonitored compute at
  `MFU₀ ∈ (0, 1]` (`M_U ≤ MFU₀ κ U/2`), the residual outside the work certified correctly and run
  (`M_r ≤ κ (C − (1 − ε_W) W_cert + wrongCapacity + skipped)/2`, `W_cert` the period's certified work, `ε_W` PoUW's
  deficit), and incorrect proof blocks within the cheating share `χ`.

Then, for any `ε' > 0` and `K`, with probability at least `1 − δ` over the period's coins,
`δ = p/ε' + J (1 − q)^(K+1) + ∑_t ks t`, the compute efficiency is at most
`s + (1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + J K B/C) + ν + χ`, and the hardware multiple the developer needs is at
least `ρ = max(1, 1/(s + ((1 − s)(r̂ + ε_W W_cert/C + ε' + JΔ/H + J K B/C) + ν + χ)/MFU₀))`, with
`r̂ = 1 − W_cert/C`, `s = U/(C + U)` and `ν = (2/κ) X/(C + U)` (`efficiency κ X C U`), the declared days' pooled bound
as a share of all compute at peak. At a pooled `δ` (1% and 10⁻⁶), `ε' = 2p/δ` and `K ≥ ln(2J/δ)/q` give
`δ + ∑_t ks t`. The pooled `δ` covers the first two terms only: `∑_t ks t`, the windows' knowledge errors, grows with
the number of windows and has no value yet. The per-day figure is the statement at one day (`R` a singleton, `T` the
day's windows). With every proof block proved (`q = 1`), `K = 0` and the escape share is 0. -/
def DampenerRemotePeriod : Prop :=
  ∀ (T : ℕ) (Ω : Fin T → Type) [∀ t, Fintype (Ω t)] [∀ t, Nonempty (Ω t)]
    (cap : Fin T → ℝ) (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p ε : ℝ)
    (β ι : Type) [DecidableEq ι] (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T)
    (I : Finset ι) (idW : Fin T → ι) (τw : Fin T → ℝ) (w : β → ℝ) (ks : Fin T → ℝ) (K : ℕ)
    (Φ Δ H q B εW κ C U χ mfu : ℝ) (W : ((t : Fin T) → Ω t) → ℝ) (M Mu Mr Mx : ((t : Fin T) → Ω t) → ℕ)
    (R : Type) [Fintype R] (d : R → RegisteredCircuit) (n : R → ℕ) (c k : ℕ) (γ V : Type) [Fintype γ] [DecidableEq γ]
    [DecidableEq V] (Wk : (i : R) → Cut.Work (d i).D γ) (val : (i : R) → Finset (Fin (d i).D.N) → V) (βw βx b : ℝ),
    -- PoUW, per window
    0 ≤ p → 0 < ε → (∀ t, 0 ≤ cap t) → Assumptions.WithinCompute cap C → Assumptions.PouwPerWindow wrong p →
    -- sampled proofs, per window: faulty blocks fixed before the window's coins, its sample and verdict read them
    Assumptions.SampledPerWindow bad drawn detected win q ks → 0 ≤ q → q ≤ 1 →
    -- the identities and the halt
    Assumptions.IdentitySchedule bad win I idW τw w Φ Δ H C B → 0 < H → 0 ≤ Δ →
    (∀ ξ, Assumptions.TerminalDetection (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b)) Δ) →
    (∀ b, 0 ≤ w b) → 0 ≤ B →
    -- the inference-only policy: each day's verdict, and (H) and (E) on its charged work
    0 < κ → 0 < C → 0 ≤ U → 0 < mfu → mfu ≤ 1 → 0 < βw → 0 < βx → 0 < b →
    (∀ i, CompliesAtMost c k (n i) (d i)) → (∀ i, Cut.ValuesInjective (Wk i) (val i)) →
    (∀ i, Cut.NoSharedPair (Wk i) (val i)) →
    (∀ i, Cut.Assumptions.Hardness (d i) ![(Wk i).allW, (Wk i).allX] (val i) ![βw, βx] k) →
    (∀ i, Cut.Assumptions.Exports (Wk i) b) →
    -- the counted multiplications
    (∀ ξ, 0 < M ξ) → (∀ ξ, Assumptions.SourcesCover (M ξ) (Mu ξ) (Mr ξ) (∑ i, (Wk i).N) (Mx ξ)) →
    (∀ ξ, Assumptions.UnmonitoredAtMost (Mu ξ) mfu κ U) →
    (∀ ξ, Assumptions.ResidualAtMost (Mr ξ) κ C εW (W ξ) (wrongCapacity cap (fun t => wrong t ξ)) (∑ b ∈ bad ξ, w b)) →
    (∀ ξ, Assumptions.CheatingAtMost (Mx ξ) κ χ C U) →
      1 - (p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t) ≤
        prob (PMF.uniformOfFintype ((t : Fin T) → Ω t)) (fun ξ =>
          efficiency κ (M ξ) C U ≤
              unregisteredShare C U +
                  (1 - unregisteredShare C U) *
                    (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
                efficiency κ (∑ i, Cut.workHE (n i) k (βw * βx) b (Wk i).O) C U + χ ∧
            remoteDampener mfu (unregisteredShare C U)
                ((1 - unregisteredShare C U) *
                    (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
                  efficiency κ (∑ i, Cut.workHE (n i) k (βw * βx) b (Wk i).O) C U + χ) ≤
              hardwareMultiple mfu κ (M ξ) C U)

/-- **`DampenerRemotePeriod`'s premises hold where its conclusion says something**: an instance meets every premise with
the failure probability `δ = p/ε' + J (1 − q)^(K+1) + ∑_t ks t` at most `10⁻⁶`, and on every run the overhead bound on
the hardware multiple (`remoteDampener`'s value) exceeds 1 while the run's own hardware multiple meets it. The proof's
instance is an honest developer over one window, run by its one identity: its PoUW certificate right, no incorrect proof
block, its monitored compute all certified work, and one declared day, the toy `Nci.Cut.Toy` complying with
`CompliesAtMost 1 2 2` under (H) at unit costs and (E) at `b = 2`, whose two MACs are the run's counted multiplications,
at `MFU₀ = 1/2`, `p = 3 · 2⁻⁴⁰` and `ε' = 2p/δ` at the pooled `δ = 10⁻⁶`, with every proof block proved (`q = 1`,
`K = 0`) and no knowledge error (`ks = 0`). The instance's certificate is never wrong, so it meets `PouwPerWindow` at
any `p ≥ 0`, and its `δ` is `1/(2 · 10⁶)` at any `p > 0`: it shows the premises hold together where the bound exceeds 1,
and gives no deployment's `p`, `ks` or `δ`. -/
def DampenerRemotePeriod_honest : Prop :=
  ∃ (T : ℕ) (Ω : Fin T → Type) (_ : ∀ t, Fintype (Ω t)) (_ : ∀ t, Nonempty (Ω t))
    (cap : Fin T → ℝ) (wrong : Fin T → ((t : Fin T) → Ω t) → Prop) (p ε : ℝ)
    (β ι : Type) (_ : DecidableEq ι) (bad drawn detected : ((t : Fin T) → Ω t) → Finset β) (win : β → Fin T)
    (I : Finset ι) (idW : Fin T → ι) (τw : Fin T → ℝ) (w : β → ℝ) (ks : Fin T → ℝ) (K : ℕ)
    (Φ Δ H q B εW κ C U χ mfu : ℝ) (W : ((t : Fin T) → Ω t) → ℝ) (M Mu Mr Mx : ((t : Fin T) → Ω t) → ℕ)
    (R : Type) (_ : Fintype R) (d : R → RegisteredCircuit) (n : R → ℕ) (c k : ℕ) (γ V : Type) (_ : Fintype γ)
    (_ : DecidableEq γ) (_ : DecidableEq V) (Wk : (i : R) → Cut.Work (d i).D γ)
    (val : (i : R) → Finset (Fin (d i).D.N) → V) (βw βx b : ℝ),
    -- PoUW, per window
    0 ≤ p ∧ 0 < ε ∧ (∀ t, 0 ≤ cap t) ∧ Assumptions.WithinCompute cap C ∧ Assumptions.PouwPerWindow wrong p ∧
    -- sampled proofs, per window: faulty blocks fixed before the window's coins, its sample and verdict read them
    Assumptions.SampledPerWindow bad drawn detected win q ks ∧ 0 ≤ q ∧ q ≤ 1 ∧
    -- the identities and the halt
    Assumptions.IdentitySchedule bad win I idW τw w Φ Δ H C B ∧ 0 < H ∧ 0 ≤ Δ ∧
    (∀ ξ, Assumptions.TerminalDetection (bad ξ) (detected ξ) (fun b => idW (win b)) (fun b => τw (win b)) Δ) ∧
    (∀ b, 0 ≤ w b) ∧ 0 ≤ B ∧
    -- the inference-only policy: each day's verdict, and (H) and (E) on its charged work
    0 < κ ∧ 0 < C ∧ 0 ≤ U ∧ 0 < mfu ∧ mfu ≤ 1 ∧ 0 < βw ∧ 0 < βx ∧ 0 < b ∧
    (∀ i, CompliesAtMost c k (n i) (d i)) ∧ (∀ i, Cut.ValuesInjective (Wk i) (val i)) ∧
    (∀ i, Cut.NoSharedPair (Wk i) (val i)) ∧
    (∀ i, Cut.Assumptions.Hardness (d i) ![(Wk i).allW, (Wk i).allX] (val i) ![βw, βx] k) ∧
    (∀ i, Cut.Assumptions.Exports (Wk i) b) ∧
    -- the counted multiplications
    (∀ ξ, 0 < M ξ) ∧ (∀ ξ, Assumptions.SourcesCover (M ξ) (Mu ξ) (Mr ξ) (∑ i, (Wk i).N) (Mx ξ)) ∧
    (∀ ξ, Assumptions.UnmonitoredAtMost (Mu ξ) mfu κ U) ∧
    (∀ ξ, Assumptions.ResidualAtMost (Mr ξ) κ C εW (W ξ) (wrongCapacity cap (fun t => wrong t ξ)) (∑ b ∈ bad ξ, w b)) ∧
    (∀ ξ, Assumptions.CheatingAtMost (Mx ξ) κ χ C U) ∧
    p / ε + I.card * (1 - q) ^ (K + 1) + ∑ t, ks t ≤ 1 / 10 ^ 6 ∧
    ∀ ξ, 1 < remoteDampener mfu (unregisteredShare C U)
        ((1 - unregisteredShare C U) *
            (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
          efficiency κ (∑ i, Cut.workHE (n i) k (βw * βx) b (Wk i).O) C U + χ) ∧
      remoteDampener mfu (unregisteredShare C U)
          ((1 - unregisteredShare C U) *
              (residualShare (W ξ) C + εW * W ξ / C + ε + lagShare I.card Δ H + I.card * K * B / C) +
            efficiency κ (∑ i, Cut.workHE (n i) k (βw * βx) b (Wk i).O) C U + χ) ≤
        hardwareMultiple mfu κ (M ξ) C U

noncomputable section

/-! **`AcceptAll`, a PoUW verifier that accepts every certificate**: one window, its coins a bit, run by the one
identity at time 0, whose certificate, for all of the monitored compute, is wrong and accepted on every run. It has no
incorrect proof block and no declared day, all of the monitored compute is certified work, and the run's one counted
multiplication is in the residual. `DampenerRemotePeriod_nontrivial` is stated at it, its numbers written in place. -/
namespace AcceptAll

/-- The window certifies all of the monitored compute, `C = 1`. -/
abbrev cap : Fin 1 → ℝ := fun _ => 1

/-- The window's certificate is wrong on every run. -/
abbrev wrong : Fin 1 → (Fin 1 → Bool) → Prop := fun _ _ => True

/-- No incorrect proof block: none sampled, none detected, and the knowledge-soundness bound 0. -/
abbrev bad : (Fin 1 → Bool) → Finset Empty := fun _ => ∅
abbrev drawn : (Fin 1 → Bool) → Finset Empty := fun _ => ∅
abbrev detected : (Fin 1 → Bool) → Finset Empty := fun _ => ∅
abbrev win : Empty → Fin 1 := fun b => b.elim
abbrev w : Empty → ℝ := fun _ => 0
abbrev ks : Fin 1 → ℝ := fun _ => 0

/-- The one identity runs the window at time 0. -/
abbrev idW : Fin 1 → Unit := fun _ => ()
abbrev τw : Fin 1 → ℝ := fun _ => 0

/-- All of the monitored compute is certified work, `W_cert = 1`, and the run's one counted multiplication is in the
residual. -/
abbrev W : (Fin 1 → Bool) → ℝ := fun _ => 1
abbrev M : (Fin 1 → Bool) → ℕ := fun _ => 1
abbrev Mu : (Fin 1 → Bool) → ℕ := fun _ => 0
abbrev Mr : (Fin 1 → Bool) → ℕ := fun _ => 1
abbrev Mx : (Fin 1 → Bool) → ℕ := fun _ => 0

/-- No declared day. -/
abbrev d : Empty → RegisteredCircuit := fun i => i.elim
abbrev n : Empty → ℕ := fun i => i.elim
abbrev Wk : (i : Empty) → Cut.Work (d i).D Unit := fun i => i.elim
abbrev val : (i : Empty) → Finset (Fin (d i).D.N) → Unit := fun i => i.elim

end AcceptAll

end

/-- **`DampenerRemotePeriod` is false without PoUW's per-window bound**: at `AcceptAll`, with `p = 0`, `ε' = 1/2`,
`C = 1`, `U = 0`, `κ = 2`, `MFU₀ = 1`, `q = 1`, `K = 0`, `c = k = 1` and `β_w = β_x = b = 1`, every other premise
holds, PoUW's per-window bound fails (the certificate is wrong with probability 1, above `p`), and the conclusion fails:
at `δ = 0`, the run's compute efficiency is 1, above the bound `1/2`, and its hardware multiple is 1, below the overhead
bound `2`, on every run. -/
def DampenerRemotePeriod_nontrivial : Prop :=
  -- every other premise: the window's coins
  (∀ _ : Fin 1, Nonempty Bool) ∧
  -- PoUW's capacities
  (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 / 2 ∧ (∀ t, 0 ≤ AcceptAll.cap t) ∧ Assumptions.WithinCompute AcceptAll.cap 1 ∧
  -- sampled proofs, per window
  Assumptions.SampledPerWindow (Ω := fun _ => Bool) AcceptAll.bad AcceptAll.drawn AcceptAll.detected AcceptAll.win 1
    AcceptAll.ks ∧
  (0 : ℝ) ≤ 1 ∧ (1 : ℝ) ≤ 1 ∧
  -- the identities and the halt
  Assumptions.IdentitySchedule AcceptAll.bad AcceptAll.win (∅ : Finset Unit) AcceptAll.idW AcceptAll.τw AcceptAll.w 0 0 1
    1 0 ∧
  (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧
  (∀ ξ, Assumptions.TerminalDetection (AcceptAll.bad ξ) (AcceptAll.detected ξ)
    (fun b => AcceptAll.idW (AcceptAll.win b)) (fun b => AcceptAll.τw (AcceptAll.win b)) 0) ∧
  (∀ b, 0 ≤ AcceptAll.w b) ∧ (0 : ℝ) ≤ 0 ∧
  -- the inference-only policy
  (0 : ℝ) < 2 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) ≤ 0 ∧ (0 : ℝ) < 1 ∧ (1 : ℝ) ≤ 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧ (0 : ℝ) < 1 ∧
  (∀ i, CompliesAtMost 1 1 (AcceptAll.n i) (AcceptAll.d i)) ∧
  (∀ i, Cut.ValuesInjective (AcceptAll.Wk i) (AcceptAll.val i)) ∧ (∀ i, Cut.NoSharedPair (AcceptAll.Wk i) (AcceptAll.val i)) ∧
  (∀ i, Cut.Assumptions.Hardness (AcceptAll.d i) ![(AcceptAll.Wk i).allW, (AcceptAll.Wk i).allX] (AcceptAll.val i)
    ![1, 1] 1) ∧
  (∀ i, Cut.Assumptions.Exports (AcceptAll.Wk i) 1) ∧
  -- the counted multiplications
  (∀ ξ, 0 < AcceptAll.M ξ) ∧
  (∀ ξ, Assumptions.SourcesCover (AcceptAll.M ξ) (AcceptAll.Mu ξ) (AcceptAll.Mr ξ) (∑ i, (AcceptAll.Wk i).N)
    (AcceptAll.Mx ξ)) ∧
  (∀ ξ, Assumptions.UnmonitoredAtMost (AcceptAll.Mu ξ) 1 2 0) ∧
  (∀ ξ, Assumptions.ResidualAtMost (AcceptAll.Mr ξ) 2 1 0 (AcceptAll.W ξ)
    (wrongCapacity AcceptAll.cap (fun t => AcceptAll.wrong t ξ)) (∑ b ∈ AcceptAll.bad ξ, AcceptAll.w b)) ∧
  (∀ ξ, Assumptions.CheatingAtMost (AcceptAll.Mx ξ) 2 0 1 0) ∧
  -- PoUW's per-window bound fails
  ¬ Assumptions.PouwPerWindow (Ω := fun _ => Bool) AcceptAll.wrong 0 ∧
  -- and so does the conclusion
  ¬ (1 - (0 / (1 / 2) + (∅ : Finset Unit).card * (1 - 1) ^ ((0 : ℕ) + 1) + ∑ t, AcceptAll.ks t) ≤
      prob (PMF.uniformOfFintype (Fin 1 → Bool)) (fun ξ =>
        efficiency 2 (AcceptAll.M ξ) 1 0 ≤
            unregisteredShare 1 0 +
                (1 - unregisteredShare 1 0) *
                  (residualShare (AcceptAll.W ξ) 1 + 0 * AcceptAll.W ξ / 1 + 1 / 2 +
                    lagShare (∅ : Finset Unit).card 0 1 + (∅ : Finset Unit).card * (0 : ℕ) * 0 / 1) +
              efficiency 2 (∑ i, Cut.workHE (AcceptAll.n i) 1 (1 * 1) 1 (AcceptAll.Wk i).O) 1 0 + 0 ∧
          remoteDampener 1 (unregisteredShare 1 0)
              ((1 - unregisteredShare 1 0) *
                  (residualShare (AcceptAll.W ξ) 1 + 0 * AcceptAll.W ξ / 1 + 1 / 2 +
                    lagShare (∅ : Finset Unit).card 0 1 + (∅ : Finset Unit).card * (0 : ℕ) * 0 / 1) +
                efficiency 2 (∑ i, Cut.workHE (AcceptAll.n i) 1 (1 * 1) 1 (AcceptAll.Wk i).O) 1 0 + 0) ≤
            hardwareMultiple 1 2 (AcceptAll.M ξ) 1 0))

end Nci.Guarantees
