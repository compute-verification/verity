module

public import Proofs.Flock.EndToEnd.Joint.MT.Draw
public import Proofs.Flock.EndToEnd.Period.Run
public import Proofs.Flock.Soundness.Discharge.ZkHidden.PublicJ
public import Proofs.Flock.Soundness.Assumptions.Zk
public import Flock.Desc

/-!
# The joint stratified bound over sampling blocks of several tables (trusted)

`JointStratified` and `JointStratifiedPooled` (`Joint/Statement.lean`) at sampling blocks proved in `J` tables in
parallel. A sampling block is one hidden-output `--zk` session of `J` tables over `n` proof blocks, whose units the
challenger's coins sample by one live stratified sample (`Flock.Draw.drawWith` at `.stratified`, `SamplingDesign.execOS`): stratum
`s` holds `n_s` of the sampling block's units, `k_s` of them are sampled uniformly without replacement, independently
across strata, and the strata run over every table's units, so a table's subcircuits are strata of the one sample.
`verify --zk --session-tables J` proves the sampled proof blocks in `J` equal consecutive parts, table `j` part `j` (`drawJ`,
`partS`); when the units are laid out table by table and every table's strata sample the same number of units, part `j` is
table `j`'s own stratified sample. The record's custody is `RecordCustodyZKJS` (`live-verifier` at the stratified sampling design);
that the parts are proved and set up with the session's link points is verity-verify-proof's own arithmetic
(`ZkHidden.sampleOkZKHJ_of_accepts_stratified`).

`StratifiedBlockMT` is such a sampling block, with every binder of `EndToEndHidden` but the prover and its sampling design, as
`Period`'s `SamplingBlock` is. Its incorrect proof blocks are `incorrectRegZ` in the committed transcript fixed at registration
(`XplurZC`). The bound is `JointStratified`'s, with the same budget and the same confidence set:

* `JointStratifiedMT`, per sampling block: except with probability `δ + ε_ks + ε_bind`, if verity-verify-proof accepts, the
  sampling block's incorrect proof blocks `B` lie in the confidence set (`SamplingDesign.stratified`'s missProb of `B` is at least `δ`), so they
  spend at most the one budget, `Σ_s k_s·x_s/n_s ≤ ln(1/δ)` over every stratum of every table, and their ANDs and their
  output bits are each at most the most any set in the confidence set holds (`StratifiedBlockMT.maxHarm`).
* `JointStratifiedPooledMT`, pooled: the same over a period of `T` such sampling blocks played by one adaptive prover,
  with one budget for the whole period: except with probability `δ + Σ_t (ε_ks,t + ε_bind,t)`.

`ε_ks` is the sampling block's knowledge term at the worst registration (`StratifiedBlockMT.epsKs`) and `ε_bind` its
binding term `ε_bind` (`StratifiedBlockMT.link`). The named assumptions are A3 (`uniform/io-getrandombytes`) for the
sample's bytes, `live-verifier` (`StratifiedBlockMT.hCust`, `RecordCustodyZKJS`), and `cr/sha-512` (`TableCRZ`) and
`ecr/sha-512` (`LinkCRZC`) at the strategy each sampling block is played with (in a period, at every strategy it is
played with on some run, `StratifiedMT.Plays`), stated flat. Their proofs are `Flock.SecurityProofs.JointStratifiedMT`
and `Flock.SecurityProofs.JointStratifiedPooledMT` (`Joint/MT/Bound.lean`).

Flock's definitions are still in its proofs package, so, like `EndToEndHidden`, this module imports them.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **A sampling block of `J` tables whose sample is stratified**: one hidden-output `--zk` session of `nTab I` tables over
`n` proof blocks, with every binder of `EndToEndHidden` but the prover and its sampling design. Its statement (circuit descriptor,
public file); its strata over every table's units, unit `u` in stratum `sub u`, `ks s` of stratum `s`'s units sampled, the
executable strata `St` describing them (`SamplingDesign.ExecStrata`); its record (sample files, session writer, reference setup at
table 0, `∑ s, ks s / nTab I` sampled proof blocks a table); the verifier's byte sources, one per stratum; the verify command's
tags; `live-verifier` at the stratified sampling design (`hCust`); the count curve's parameters; and the knowledge finders' budgets
`qF`, `qS`. -/
structure StratifiedBlockMT where
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
  d₀ : Lean.Json
  y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g ((∑ s, ks s) / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g ((∑ s, ks s) / nTab I)) →
      Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin m → ℕ
  src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJS I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr Kd
    (Array.ofFn St) pop
  k : ℕ
  Rw : ℕ
  M : ℕ
  hk : 1 ≤ k
  hRw : 1 ≤ Rw
  hM : 1 ≤ M
  ρ : ℝ
  hρ : ρ < 1
  hr : ∀ S R j, rateZ (planZHJ (Reg := Reg) dj y₀ S R) j k ≤ ρ
  t' : ℝ
  ht' : 0 ≤ t'
  qF : ℝ
  qS : ℝ

attribute [instance] StratifiedBlockMT.fintypeΩ StratifiedBlockMT.nonemptyΩ

namespace StratifiedBlockMT

open Classical

variable (w : StratifiedBlockMT)

/-- The sampling block's sample: `Flock.Draw.drawWith`'s stratified sample on the challenger's coins, over every table's
units. -/
def L : SamplingDesign w.n := SamplingDesign.execOS (n := w.n) w.src (.stratified w.Kd (Array.ofFn w.St)) w.pop

/-- The sampling block's audit as it runs: `EndToEndHidden`'s game at the stratified sample. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript) :=
  auditLive w.L w.Reg (liveOfZJ w.dg (planZHJ w.dj w.y₀) w.N w.wr)

/-- The key program's partition into proof blocks. -/
noncomputable def P := (keyProg w.D.unroll w.hU w.n).partition (Discharge.Layout.outsOf w.D.unroll w.hU w.n)

/-- The public file's commitments. -/
noncomputable def vb :=
  (rsZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg)).bindingZ
  (rsZHJ_computes w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg))

/-- The model audit's strategy that a live strategy `τ` refines to: `EndToEndHidden`'s `σ`, with the sample at each
accepted outcome from custody at the stratified sampling design (`sampleOkZKHJ_of_accepts_stratified`). -/
noncomputable def sim (τ : Strategy w.game) :=
  simZJ execArith hs512 (planZHJ w.dj w.y₀) w.I w.wr (opnAtZHJ (planZHJ w.dj w.y₀) w.I w.dj) w.dg
    (decAtZJ (planZHJ w.dj w.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) w.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ w.I w.wr _ _ _ _ _
      (runsZCJH_liveA w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj
        w.y₀ w.wr w.L w.hTags w.hCust.1
        fun ω R o hacc => sampleOkZKHJ_of_accepts_stratified w.I w.ht (Flock.Desc.parse_flat w.hD) w.sub w.ks w.hks
          w.St w.hS w.Kd w.pop w.Lb w.src w.dj w.hCust.2 w.hCust.1 ω R o hacc)) τ

/-- **The sampling block's incorrect proof blocks** at the strategy `τ`: `incorrectRegZ` in the committed transcript fixed at
registration (`XplurZC`), as in `EndToEndHidden`. -/
noncomputable def incorrectSet (τ : Strategy w.game) : Finset (Fin w.n) :=
  incorrectRegZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ))
    (XplurZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k w.Rw)

end StratifiedBlockMT

/-- What a period's run records of sampling block `idx`: whether verity-verify-proof accepted it, its incorrect proof blocks
(`StratifiedBlockMT.incorrectSet`), and whether a sampled proof block is one of them. -/
structure BlockRecMT (W : ℕ → StratifiedBlockMT) where
  idx : ℕ
  accepted : Prop
  incorrect : Finset (Fin (W idx).n)
  sampledWrong : Prop

namespace BlockRecMT

variable {W : ℕ → StratifiedBlockMT}

/-- The record's incorrect proof blocks' total weight `Σ_{u ∈ B} h u`, with `h t` sampling block `t`'s per-unit weights. -/
def harm (h : (t : ℕ) → Fin (W t).n → ℕ) (r : BlockRecMT W) : ℕ := ∑ u ∈ r.incorrect, h r.idx u

end BlockRecMT

namespace StratifiedBlockMT

open Classical

end StratifiedBlockMT

namespace StratifiedMT

end StratifiedMT

end Flock.Guarantees
