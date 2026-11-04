import Proofs.Properties.Verity.Recursive.ScheduleSession

/-!
# The recursive verifier's schedule: the run accepts

F1 and F2 assembled: a run of the compiled recursive verifier `RecDeploy.V IV st` accepts where its statement has a V*
part and its last V* session passes F2's hypotheses (`run_acc_last`). That session's acceptance at full length is a
fact about the run's own messages, so it already holds the earlier sessions: the run reached its last round, so it
passed every check before it. `run_acc` is the form the split states, every session's hypotheses; `run_acc_sha512`
is it at SHA-512, the oracle `Verity.Recursive.acc_of_accepts` runs at, where `(recordAt …).V` is `RecDeploy.V` at
`(boundAt wE w).IVStar` and `stageAt …` by definition.
-/

namespace Verity.Recursive.Schedule

open Claims (Msg Out Prover run)
open FlockDeploy (Coin)
open RecDeploy

section Run

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- **F3 at the last session.** Where the statement has a V* part, and its last V* session's staged numbers are its
numbers and the outer verifier, at an oracle where it accepts only at its last round, accepts that session's view of
the run, the run accepts. -/
theorem run_acc_last (hO : OuterAtEnd H) (hn : 0 < nS x)
    (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) (nS x - 1)) =
      nums st x (nS x - 1))
    (hacc : ((nextO (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) (nS x - 1))
        ((session IV st x (nS x - 1) (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x (nS x - 1) (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 =
      .acc) :
    (run H P (V IV st) x ω r).out = .acc := by
  obtain ⟨-, ⟨a, ha⟩, -⟩ := session_end H IV st x P ω r hO (s := nS x - 1) (by omega) hnum hacc
  rw [show (nS x - 1 + 1 == nS x) = true from beq_iff_eq.2 (by omega)] at ha
  exact run_acc_out H IV st x P ω r ha

/-- **F3: the run accepts** where its statement has a V* part and every V* session passes F2's hypotheses. -/
theorem run_acc (hO : OuterAtEnd H) (hn : 0 < nS x)
    (hnum : ∀ s < nS x,
      numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ∀ s < nS x, ((nextO (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 = .acc) :
    (run H P (V IV st) x ω r).out = .acc :=
  run_acc_last H IV st x P ω r hO hn (hnum _ (by omega)) (hacc _ (by omega))

end Run

/-- **F3 at SHA-512**, the oracle the recursive record runs at. -/
theorem run_acc_sha512 (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin ((V IV st).k x) → Coin) (hn : 0 < nS x)
    (hnum : ∀ s < nS x, numsOf (staged st x ((run FlockDeploy.sha512 P (V IV st) x ω r).msgs.take
      (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ∀ s < nS x, ((nextO (vstarAt st x ((run FlockDeploy.sha512 P (V IV st) x ω r).msgs.take
        (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (run FlockDeploy.sha512 P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run FlockDeploy.sha512 P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval
          FlockDeploy.sha512).1 = .acc) :
    (run FlockDeploy.sha512 P (V IV st) x ω r).out = .acc :=
  run_acc FlockDeploy.sha512 IV st x P ω r outerAtEnd_sha512 hn hnum hacc

end Verity.Recursive.Schedule
