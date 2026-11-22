module

public import Proofs.Flock.EndToEndHidden.Statement
public import Proofs.Flock.EndToEnd.Period.Run

/-!
# A period of sampling blocks (trusted)

A period's integrity audit is a sequence of sampling blocks, each one `--zk` session under hidden outputs with a fresh
live subset sample (a `SamplingBlock`: every binder of `EndToEndHidden` but the prover). One prover strategy plays the
whole period (`Period.game`), adaptively: the strategy each sampling block is played with may depend on everything
before it, and `Period.Plays` says which strategies a sampling block is played with on some run. `SamplingBlock.ks` is
the sampling block's knowledge term at the worst registration, since the registration is the prover's, and
`SamplingBlock.link` its binding term; `SamplingBlockRec` is what a run records of a sampling block: whether
verity-verify-proof accepted it, its incorrect proof blocks and whether a sampled proof block is one of them.

Flock's definitions are still in its proofs package, so, like `EndToEndHidden`, this module imports them.
-/

@[expose] public section

namespace Flock.Guarantees

open FlockSoundness FlockSoundness.Game Spot.Sampling FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkHidden
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **One sampling block of a period**: one `--zk` session under hidden outputs over `n` proof blocks, with every binder
of `EndToEndHidden` but the prover: its statement (circuit descriptor, public file), its sampling design (a uniform `kd`-subset of
the `n` units, split evenly over the session's tables), its record (sample files, session writer, reference setup), the
verifier's byte source, the verify command's tags, `live-verifier` (`hCust`), the count curve's parameters, the
threshold `K₀` and the knowledge finders' budgets `qF`, `qS`. -/
structure SamplingBlock where
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
  kd : ℕ
  hkd : kd ≤ n
  hdiv : kd % nTab I = 0
  dj : Finset (Fin n) → Lean.Json
  S₀ : Finset (Fin n)
  d₀ : Lean.Json
  y₀ : SampleSetupZHJ (inputsJ I 0) (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀
  wr : (S : Finset (Fin n)) → (R : Reg) →
    LiveOutJ (planZHJ (Reg := Reg) dj y₀ S R).length (mPtsOf D.unroll.g (kd / nTab I)) → Discharge.Exec.Transcript
  dg : ByteArray → ByteArray
  N : ℕ
  Ω : Type
  [fintypeΩ : Fintype Ω]
  [nonemptyΩ : Nonempty Ω]
  Lb : Fin 1 → ℕ
  src : Ω → (i : Fin 1) → Fin (Lb i) → Fin 256
  hTags : TagsOkZHJ I
  hCust : Assumptions.Zk.RecordCustodyZKJ I dj (fun S R => (planZHJ (Reg := Reg) dj y₀ S R).length) wr kd
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
  K₀ : ℕ
  hK₀ : K₀ ≤ n
  qF : ℝ
  qS : ℝ

attribute [instance] SamplingBlock.fintypeΩ SamplingBlock.nonemptyΩ

namespace SamplingBlock

open Classical

variable (w : SamplingBlock)

/-- The sampling block's sample: `Flock.Draw.drawWith`'s uniform `kd`-subset on the challenger's coins. -/
def L : SamplingDesign w.n := SamplingDesign.execOS (n := w.n) w.src (.subset w.kd) w.n

/-- The sampling block's audit as it runs: `EndToEndHidden`'s game. -/
noncomputable def game : Game (w.Reg × Finset (Fin w.n) × Discharge.Exec.Transcript) :=
  auditLive w.L w.Reg (liveOfZJ w.dg (planZHJ w.dj w.y₀) w.N w.wr)

/-- The key program's partition into proof blocks. -/
noncomputable def P := (keyProg w.D.unroll w.hU w.n).partition (Discharge.Layout.outsOf w.D.unroll w.hU w.n)

/-- The public file's commitments. -/
noncomputable def vb :=
  (rsZHJ w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg)).bindingZ
  (rsZHJ_computes w.dj w.y₀ w.ht (Flock.Desc.parse_flat w.hD) w.hscope w.hU w.hpub w.hpt (Reg := w.Reg))

/-- The model audit's strategy that a live strategy `τ` refines to (`EndToEndHidden`'s `σ`). -/
noncomputable def sim (τ : Strategy w.game) :=
  simZJ execArith hs512 (planZHJ w.dj w.y₀) w.I w.wr (opnAtZHJ (planZHJ w.dj w.y₀) w.I w.dj) w.dg
    (decAtZJ (planZHJ w.dj w.y₀)) (fun _ _ _ _ => true) (fun _ _ _ _ _ => []) (fun _ _ rs j => capOfRoot rs[j.val]!) w.N
    (decodesOnZKJ_of_runsOnZCJ execArith hs512 _ w.I w.wr _ _ _ _ _
      (runsZCJH_live w.I w.ht (Flock.Desc.parse_flat w.hD) (Discharge.Layout.scopeOkH_spec w.hU w.hscope).2.1 w.dj w.y₀
        w.wr _ w.hTags w.hCust.1
        fun _ => sampleOkZKHJ_of_file w.I w.ht (Flock.Desc.parse_flat w.hD) w.kd w.n w.hdiv w.dj w.hCust.2 _)) τ

/-- **`cr/sha-512` and `ecr/sha-512` at the strategy `τ`**: `EndToEndHidden`'s hypotheses `TableCRZ`, for the knowledge
finders of each table the prover's session runs, and `LinkCRZC`, for its link finders. -/
def CR (τ : Strategy w.game) : Prop :=
  (∀ ω j, TableCRZ execArith hs512 (planZHJ w.dj w.y₀ (w.L.draw ω) (reg (w.sim τ))) j
      (zcStrat execArith hs512 (planZHJ w.dj w.y₀) (cont (w.sim τ) ω)) w.qF w.qS) ∧
    LinkCRZC execArith hs512 w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb (reg (w.sim τ)) (cont (w.sim τ)) w.k w.Rw
      w.M w.t'

/-- The knowledge term `ε_ks`, `ksAvgStrictZ` at the worst registration. -/
noncomputable def ks : ℝ≥0∞ := ⨆ R : w.Reg, ksAvgStrictZ (planZHJ w.dj w.y₀) w.L w.k w.qF w.qS R

/-- The binding error `ε_bind = Q_s/(1 − ρ)·(2t'(1 + k)/2^256 + 1/(eM) + k/(eR_w))`, as in `EndToEndHidden`. -/
noncomputable def link : ℝ≥0∞ :=
  ENNReal.ofReal ((∑ p : w.vb.Pos,
      (Fintype.card (ΩcZ execArith w.P (planZHJ w.dj w.y₀) (tabZHJ w.dj w.y₀) w.vb w.L p) : ℝ) / Fintype.card w.L.Ω) *
    (2 * (w.t' * (1 + w.k)) / 2 ^ (256 : ℝ) + 1 / (Real.exp 1 * w.M) + w.k / (Real.exp 1 * w.Rw)) / (1 - w.ρ))

end SamplingBlock

/-- What a period's run records of sampling block `idx`: whether verity-verify-proof accepted it, how many of its units are
incorrect, and whether a sampled proof block is one of them. -/
structure SamplingBlockRec where
  idx : ℕ
  accepted : Prop
  incorrect : ℕ
  sampledWrong : Prop

namespace SamplingBlock

open Classical

end SamplingBlock

namespace Period

/-- **A period of `T` sampling blocks**, sampling block `t`'s audit after sampling block `t − 1`'s, against one
prover. -/
noncomputable def game (W : ℕ → SamplingBlock) (T : ℕ) : Game Unit := monitoringPeriodG (fun t => (W t).game) 0 T

/-- **The strategies the period plays sampling block `t` with**: `τ` plays sampling block `t` on some run of the period
at `s`. -/
def Plays (W : ℕ → SamplingBlock) (T : ℕ) (s : Strategy (game W T)) (t : ℕ) (τ : Strategy (W t).game) : Prop :=
  MonitoringPeriodPlays (fun t => (W t).game) 0 T s t τ

end Period

end Flock.Guarantees
