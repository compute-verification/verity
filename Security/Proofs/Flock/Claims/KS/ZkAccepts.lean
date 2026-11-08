module

public import Proofs.Flock.Claims.Sampling.Deployed
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.KS.ConflictRun
import Proofs.Flock.Claims.KS.Width

/-!
# An accepted run of the deployed `--zk` verifier is flock-verify `--zk`'s acceptance

The claims form's Flock verifier is `V true` (`deployZk`'s and `deployZkTrim`'s `V`, and so the sessions' verifier of
the sampling record `SamplingDeploy.deploy`). Where its run accepts against SHA-512 (`zkAccepts_of_run`), it decided
at the statement's inputs `s` (`insOf true x = some s`, what flock-verify reads of the statement), on the last message's
record and proof files `f`, and:

* the record's sample is the statement's as a value (`Custody.recordDraw_of_checks`: the verifier rebuilds both,
  `Flock.norm`);
* flock-verify `--zk` accepts that transcript at the statement's inputs (`AcceptsZK s.read.I ⟨f.record, f.proofs⟩`, and
  so `LiveAcceptsZK` at every outcome carrying it).

This is the `--zk` side of `plainVerdict_of_run`. Equality of the sample as a value is what the setup reads: `Exec.setupOf`
sets the record up at the record's own sample (`Custody.setupOf_record`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody
open Lean (Json)

theorem deployZkTrim_V (rel : Readable.Parsed → Readable.Relation) : (FlockDeploy.deployZkTrim rel).V = V true := rfl

theorem deploy_flock_V (rel : Readable.Parsed → Readable.Relation) (c : SamplingDeploy.Code) :
    (SamplingDeploy.deploy rel c).flock.V = V true := rfl

/-- **An accepted run of `V true` is flock-verify `--zk`'s acceptance**: against SHA-512, the run decided at the
statement's inputs `s` on its last message's record and proof files `f`; the record's sample is the statement's, and
`verify --zk` accepts `⟨f.record, f.proofs⟩` at `s.read.I`. -/
theorem zkAccepts_of_run {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ s spec sts link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s spec sts link c₁ mids f ∧
      insOf true x = some s ∧
      Exec.recordDraw f.record = s.read.draw ∧ ZkExec.AcceptsZK s.read.I ⟨f.record, f.proofs⟩ := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, -⟩ := acc_decide h
  have hacc := run_acc_next sha512 P (V true) x ω r h
  replace hacc : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc
  obtain ⟨hv, hc⟩ := (next_decide_sha512 hact).1 hacc
  simp only [ite_true] at hv
  obtain ⟨hins, hs⟩ := setupOf_some (act_decide hact)
  have hzk : s.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hdraw := Claims.Bridge.Custody.recordDraw_of_checks hins hc
  refine ⟨s, spec, sts, link, c₁, mids, f, hact, hins, hdraw, ?_⟩
  exact ⟨spec, sts, Claims.Bridge.Custody.setupOf_zkInputs (Ins.setup_read hs) hzk
    (Claims.Bridge.Custody.publicInputs_none_of_checks hc) hdraw, Claims.Bridge.Custody.setup_shapeOk hs hzk, hv⟩

/-- **`zkAccepts_of_run` as the live event**: `LiveAcceptsZK` at the statement's inputs, at every outcome whose
transcript is the last message's record and proof files. -/
theorem liveAcceptsZK_of_run {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : FlockDeploy.Ins) (f : FlockDeploy.Final), insOf true x = some s ∧
      Exec.recordDraw f.record = s.read.draw ∧ ∀ (R : Reg) (S : Finset (Fin n)),
        ZkExec.LiveAcceptsZK s.read.I (R, S, (⟨f.record, f.proofs⟩ : Exec.Transcript)) := by
  obtain ⟨s, -, -, -, -, -, f, -, hins, hdraw, hacc⟩ := zkAccepts_of_run h
  exact ⟨s, f, hins, hdraw, fun _ _ => hacc⟩

end FlockDeploy.KS
