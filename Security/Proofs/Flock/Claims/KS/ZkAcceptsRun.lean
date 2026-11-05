module

public import Proofs.Flock.Claims.Deploy
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
import Proofs.Flock.Claims.KS.PlainWrP
import Proofs.Flock.Claims.KS.Width

/-!
# An accepted run of the deployed `--zk` verifier is flock-verify `--zk`'s acceptance

`liveAcceptsZK_of_run_rebuilt`: against SHA-512, an accepted run of `Deploy.V true` decided at the statement's inputs
`s`, and `verify --zk` accepts the last message's record and proof files at `s.read.I` (`AcceptsZK`), so
`LiveAcceptsZK` holds at every outcome carrying that transcript. No premise on the draw: the verifier rebuilds both the
record's draw and the statement's (`Flock.norm`), so the record's draw is the statement's as a value
(`Custody.recordDraw_of_decide`, inside `Custody.accepts_of_decide`).
-/

@[expose] public section

namespace FlockDeploy.KS

open Flock FlockSoundness FlockSoundness.Discharge
open Lean (Json)

/-- **An accepted run of `V true` is flock-verify `--zk`'s acceptance, with no premise on the draw**: against SHA-512,
the run decided at the statement's inputs `s`, and `verify --zk` accepts the last message's record and proof files `f`
at `s.read.I` (`AcceptsZK`), so `LiveAcceptsZK` holds at every outcome carrying that transcript. -/
theorem liveAcceptsZK_of_run_rebuilt {n : ℕ} {Reg : Type} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} (h : (Claims.run sha512 P (V true) x ω r).out = .acc) :
    ∃ (s : Ins) (f : Final), insOf true x = some s ∧
      ZkExec.AcceptsZK s.read.I (⟨f.record, f.proofs⟩ : Exec.Transcript) ∧
      ∀ (R : Reg) (S : Finset (Fin n)),
        ZkExec.LiveAcceptsZK s.read.I (R, S, (⟨f.record, f.proofs⟩ : Exec.Transcript)) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, -⟩ := acc_decide h
  have hacc : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := run_acc_next sha512 P (V true) x ω r h
  obtain ⟨-, hc⟩ := (next_decide_sha512 hact).1 hacc
  obtain ⟨d, hd⟩ := checks_draw hc
  have hZ := (Claims.Bridge.Custody.accepts_of_decide (J := 0) (mPts := 0) hact hacc hd (Nat.zero_le _)).1
  exact ⟨s, f, (setupOf_some (act_decide hact)).1, hZ, fun _ _ => hZ⟩

end FlockDeploy.KS
