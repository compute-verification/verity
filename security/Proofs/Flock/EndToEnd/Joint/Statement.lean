module

public import Proofs.Flock.EndToEnd.Period.Run
public import Proofs.Flock.Soundness.Assumptions.Zk
public import Proofs.Flock.Soundness.Discharge.ZkHidden.SoundSampleOk
public import Flock.Desc

/-!
# The joint stratified bound: one detection budget over a sampling block's whole sample, and over a period (trusted)

A sampling block is the proof blocks one sample covers, committed and registered before the sample. Its sample is stratified
by subcircuit: subcircuit `s` holds `n_s` of the sampling block's units, the verifier samples `k_s` of them uniformly
without replacement, and an incorrect set `B` holds `x_s` of them. `B` escapes the sample with probability
`∏_s C(n_s − x_s, k_s)/C(n_s, k_s) ≤ exp(−Σ_s k_s·x_s/n_s)`. So the prover has one detection budget,
`Σ_s k_s·x_s/n_s ≤ ln(1/δ)`, spent across every subcircuit of the sampling block at once: there is no `δ` split by
subcircuit and no sum of per-subcircuit worst cases.

`StratifiedBlock` is such a sampling block: one hidden-output `--zk` session of one table (a stratified sample is a
one-table session: verity-verify-proof takes only the subset sample at more than one table, `RecordCustodyZKJ`) whose units the
challenger's coins sample by the live stratified sample (`Flock.Draw.drawWith` at `.stratified`, `SamplingDesign.execOS`), with every
binder of `zk_session_soundH_custody` but the prover. Its incorrect proof blocks are `incorrectRegZ` in the committed transcript fixed
at registration (`XplurZC`), as `SamplingBlock`'s are.

* `JointStratified`, per sampling block: except with probability `δ + ε_ks + ε_bind`, if verity-verify-proof accepts, the
  sampling block's incorrect proof blocks `B` lie in the confidence set (`SamplingDesign.stratified`'s missProb of `B` is at least `δ`), so they
  spend at most the one budget, `Σ_s k_s·x_s/n_s ≤ ln(1/δ)`, and their ANDs and their output bits are each at most the
  most any set in the confidence set holds (`StratifiedBlock.maxHarm`). Both harm bounds are read from the same sample,
  at the same `δ`.
* `JointStratifiedPooled`, pooled: the same over a period of `T` sampling blocks played by one adaptive prover, with
  one budget for the whole period: except with probability `δ + Σ_t (ε_ks,t + ε_bind,t)`, if verity-verify-proof accepts every
  sampling block, the product of their escapes is at least `δ`, the incorrect proof blocks of all of them spend at most
  `Σ_t Σ_s k_{t,s}·x_{t,s}/n_{t,s} ≤ ln(1/δ)`, and their ANDs and output bits in all are each at most the most any
  choice of incorrect sets the period's samples miss together with probability at least `δ` holds. The bound is absolute: it
  does not grow with `T` but through the proof system's terms.

ANDs and output bits are per-unit weights (`ands u`, the unit's own `Candidate.ands`; `bits u`, its output bits); the
statements hold at every pair of weights. `ε_ks` is the sampling block's knowledge term at the worst registration
(`StratifiedBlock.epsKs`) and `ε_bind` its binding term, `ε_bind` (`StratifiedBlock.link`). The named assumptions are
`zk_session_soundH_custody`'s at every sampling block: A3 (`uniform/io-getrandombytes`) for its sample bytes,
`live-verifier` (`StratifiedBlock.hCust`), and `cr/sha-512` (`TableCRZ`) and `ecr/sha-512` (`LinkCRZC`) at the
strategy each sampling block is played with (in a period, at every strategy it is played with on some run,
`Stratified.Plays`), stated flat. Their proofs are `Flock.SecurityProofs.JointStratified` and
`Flock.SecurityProofs.JointStratifiedPooled` (`Joint/Bound.lean`).

Flock's definitions are still in its proofs package, so, like `EndToEndHidden`, this module imports them.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.HiddenDec
open scoped ENNReal

/-- **A sampling block whose sample is stratified by subcircuit**: one hidden-output `--zk` session of one table over `n`
proof blocks, with every binder of `zk_session_soundH_custody` but the prover. Its statement (circuit descriptor, public
file); its strata, unit `u` in subcircuit `sub u`, `ks s` of subcircuit `s`'s units sampled, the executable strata `St`
describing them (`SamplingDesign.ExecStrata`); its record (sample files, session writer, reference setup); the verifier's byte
sources, one per subcircuit; the verify command's tags; `live-verifier` (`hCust`); the count curve's parameters; and the
knowledge finders' budgets `qF`, `qS`. -/
structure StratifiedBlock where
  n : ℕ
  Reg : Type
  I : Inputs
  ht : I.tags.typed = false
  D : Flock.Desc
  hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D
  hU : Built D.unroll.unit
  pub : Public
  hpub : HmOut.loadPublic (zkTags I) D.unroll I.publicFile = .ok pub
  hpt : pub.tables = none
  hscope : Discharge.Layout.scopeOkH D.unroll = true
  m : ℕ
  sub : Fin n → Fin m
  ks : Fin m → ℕ
  hks : ∀ s, ks s ≤ (SamplingDesign.stratum sub s).card
  St : Fin m → Flock.Draw.Stratum
  hS : SamplingDesign.ExecStrata sub ks St
  Kd : ℕ
  pop : ℕ
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  y₀ : SampleSetupZKH I (mPtsOf D.unroll.g (∑ s, ks s)) S₀ (dj S₀)
  wr : Finset (Fin n) → Reg → LiveOut (mPtsOf D.unroll.g (∑ s, ks s)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin m → ℕ
  src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256
  hTags : TagsOkH I
  hCust : Assumptions.Zk.RecordCustodyZK I dj wr (.stratified Kd (Array.ofFn St)) pop
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planAtZKH (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  qF : ℝ
  qS : ℝ

attribute [instance] StratifiedBlock.fintypeΩ StratifiedBlock.nonemptyΩ

namespace StratifiedBlock

open Classical

variable (w : StratifiedBlock)

/-- The sampling block's sample: `Flock.Draw.drawWith`'s stratified sample on the challenger's coins. -/
def L : SamplingDesign w.n := SamplingDesign.execOS (n := w.n) w.src (.stratified w.Kd (Array.ofFn w.St)) w.pop

/-- The sampling block's audit as it runs: `zk_session_soundH_custody`'s game. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript) :=
  auditLive w.L w.Reg (liveOf w.dg (mPtsOf w.D.unroll.g (∑ s, w.ks s)) w.N w.wr)

/-- The key program's partition into proof blocks. -/
noncomputable def P := (keyProg w.D.unroll w.hU w.n).partition (Discharge.Layout.outsOf w.D.unroll w.hU w.n)

/-- The public file's commitments. -/
noncomputable def vb :=
  (rsZH w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg)).bindingZ
    (rsZH_computes w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg))

/-- The model audit's strategy that a live strategy `τ` refines to (`zk_session_soundH_custody`'s). -/
noncomputable def sim (τ : Strategy w.game) :=
  simZ execArith hs512 _ w.I w.wr _ w.dg _ _ _ _ w.N
    (decodesOnZK_of_runsOnZC execArith hs512 _ w.I w.wr _ _ _ _
      (runsZC_liveH w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj w.y₀
        w.wr _ tablesAtZKH_live w.hTags w.hCust.1
        (fun ω => sampleOkZKH_of_exec w.I (Flock.Desc.parse_flat w.hD) w.sub w.ks w.hks w.St w.hS w.Kd w.pop w.Lb w.src
          w.dj w.hCust.2 ω))) τ

/-- **The sampling block's incorrect proof blocks** at the strategy `τ`: `incorrectRegZ` in the committed transcript fixed at
registration (`XplurZC`). -/
noncomputable def incorrectSet (τ : Strategy w.game) : Finset (Fin w.n) :=
  incorrectRegZ execArith w.P (planAtZKH w.dj w.y₀) (tabAtZKH w.dj w.y₀) w.vb (reg (w.sim τ))
    (XplurZC execArith hs512 w.P (planAtZKH w.dj w.y₀) (tabAtZKH w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k
      w.Rw)

end StratifiedBlock

/-- What a period's run records of sampling block `idx`: whether verity-verify-proof accepted it, its incorrect proof blocks
(`StratifiedBlock.incorrectSet`), and whether a sampled proof block is one of them. -/
structure BlockRec (W : ℕ → StratifiedBlock) where
  idx : ℕ
  accepted : Prop
  incorrect : Finset (Fin (W idx).n)
  sampledWrong : Prop

namespace BlockRec

variable {W : ℕ → StratifiedBlock}

/-- The record's incorrect proof blocks' total weight `Σ_{u ∈ B} h u`, with `h t` sampling block `t`'s per-unit weights. -/
def harm (h : (t : ℕ) → Fin (W t).n → ℕ) (r : BlockRec W) : ℕ := ∑ u ∈ r.incorrect, h r.idx u

end BlockRec

namespace StratifiedBlock

open Classical

end StratifiedBlock

namespace Stratified

end Stratified

end Flock.Guarantees
