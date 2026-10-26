import Proofs.Properties.Verity.Recursive.ScheduleInner
import Proofs.Flock.Claims.Recursion.RecRecord

/-!
# The recursive verifier's schedule: one V* session (PRF-202, piece F2)

The converse of `RecDeploy.session_core` and `RecDeploy.sim`, on a run of the compiled recursive verifier
`RecDeploy.V IV st` that need not accept. Take V* session `s` of the run, at its V* statement `vstarAt` read off the
run's inner and staging rounds, and suppose:

* part `s`'s staged numbers are its numbers (`numsOf (staged …) = nums …`, the check the compiled verifier makes at each
  of the session's rounds);
* the outer verifier `VO` accepts the session's view as the run shows it (`session`, with each answer's coin
  `outCoin`, as `LiveCoins` reads it), at an oracle where it accepts only at its last round (`OuterAtEnd`, which
  SHA-512 is).

Then the run reaches the session's last round, and the compiled verifier answers it as the outer verifier's
acceptance (`sessOut … .acc`: `.acc` at the last session, else `.msg []`), and before the last session the run goes on
to session `s + 1` (`session_end`). The session's earlier rounds are answered by their coins (`session_mid`), so the
coins its view shows are the run's own, but for the last one, which no decision reads (`nextO_last_coin`).
-/

namespace Verity.Recursive.Schedule

open Claims (Msg Out Prover Trace stepWith run coinSeq)
open FlockDeploy (Coin)
open RecDeploy

/-- **Session `s`'s first round** in a run whose inner and staging rounds are `ms`: past them and the earlier V*
sessions' rounds, at the V* statements read off `ms` (`session`'s `off`). -/
noncomputable def sessOff (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) (ms : List (Msg × Out)) (s : ℕ) : ℕ :=
  IV.k (xIOf x) + kS x + (((vstar st x ms).take s).map VO.k).sum

section Session

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- **F2, the session's rounds before its last**: under F2's hypotheses, each is answered by its coin's bytes. -/
theorem session_mid (hO : OuterAtEnd H) {s : ℕ} (hs : s < nS x)
    (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ((nextO (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 = .acc)
    {j : ℕ} (hj : j + 1 < VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)) :
    ∃ a, (run H P (V IV st) x ω r).msgs[sessOff IV st x
        ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + j]? =
      some (a, .msg (coinSeq r (sessOff IV st x
        ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s + j)).bytes) := by
  sorry

/-- **F2: one V\* session.** Where part `s`'s staged numbers are its numbers and the outer verifier, at an oracle where
it accepts only at its last round, accepts session `s`'s view of the run, the run reaches the session's last round, the
compiled verifier answers it as the outer verifier's acceptance (`.acc` at the last session, else `.msg []`), and before
the last session the run goes on to session `s + 1`. -/
theorem session_end (hO : OuterAtEnd H) {s : ℕ} (hs : s < nS x)
    (hnum : numsOf (staged st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) = nums st x s)
    (hacc : ((nextO (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map Prod.fst)
        ((session IV st x s (run H P (V IV st) x ω r).seen).msgs.map fun m => outCoin m.2)).eval H).1 = .acc) :
    sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s +
        VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) ≤
      (run H P (V IV st) x ω r).msgs.length ∧
    (∃ a, (run H P (V IV st) x ω r).msgs[sessOff IV st x
        ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s +
        VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) - 1]? =
      some (a, sessOut (s + 1 == nS x) true .acc)) ∧
    (s + 1 < nS x → sessOff IV st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s +
        VO.k (vstarAt st x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s) <
      (run H P (V IV st) x ω r).msgs.length) := by
  sorry

end Session

end Verity.Recursive.Schedule
