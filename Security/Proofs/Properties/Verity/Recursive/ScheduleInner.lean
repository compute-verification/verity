import Proofs.Flock.Claims.Recursion.DeploySessions

/-!
# The recursive verifier's schedule: the inner and staging rounds (PRF-202, piece F1)

The converse of `RecDeploy.ract_inner`, on a run of the compiled recursive verifier `RecDeploy.V IV st` that need not
accept. Where the statement reads (`readX`), and so wherever it has a V* part (`0 < nS x`):

* every inner and staging round the run reaches is answered by its coin's bytes (`RecDeploy.run_inner`), so the run
  reaches session 0's first round (`reach_sessions`);
* every later round reads the run's own inner and staging rounds (`innerPairs_reach`), whose last message is the
  staging message that `Stage.ofMsg` reads each part's public file and the inner session's record off, with the coin
  server's transcript of the inner rounds before it (`stage_reads`).

The run lemmas under them hold for any verifier: a round before a run's last is answered by a message
(`stepWith_mid`), a run whose round is answered by a message runs on or has used its rounds up (`stepWith_full`), and
a run that accepts a round accepts (`stepWith_acc_out`).
-/

namespace Verity.Recursive.Schedule

open Claims (Msg Out Prover Trace stepWith run coinSeq)
open FlockDeploy (Coin)
open RecDeploy

/-! ## Runs -/

section Steps

variable {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vn : List Msg → Claims.OC Out)

/-- **A round before a run's last is answered by a message**: a run stops at the first round it accepts or refuses. -/
theorem stepWith_mid :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace),
      t.msgs.length + i + 1 < (stepWith H P x ω vn f hist as t).msgs.length →
      ∃ a m, (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .msg m) := by
  sorry

/-- **A run whose round is answered by a message runs on, or has used its rounds up.** -/
theorem stepWith_full :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a m : Msg),
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .msg m) →
      t.msgs.length + i + 1 < (stepWith H P x ω vn f hist as t).msgs.length ∨
        (stepWith H P x ω vn f hist as t).msgs.length = t.msgs.length + f := by
  sorry

/-- **A run that accepts a round accepts.** -/
theorem stepWith_acc_out :
    ∀ (f i : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace) (a : Msg),
      (stepWith H P x ω vn f hist as t).msgs[t.msgs.length + i]? = some (a, .acc) →
      (stepWith H P x ω vn f hist as t).out = .acc := by
  sorry

end Steps

section Runs

variable (H : Msg → Msg) (IV : Claims.Verifier Coin) (st : Stage) (x : Msg) {Ω : Type} (P : Prover Ω) (ω : Ω)
  (r : Fin ((V IV st).k x) → Coin)

/-- A round before the run's last is answered by a message. -/
theorem run_mid {i : ℕ} (h : i + 1 < (run H P (V IV st) x ω r).msgs.length) :
    ∃ a m, (run H P (V IV st) x ω r).msgs[i]? = some (a, .msg m) := by
  sorry

/-- A round answered by a message is followed by another, or is the verifier's last. -/
theorem run_full {i : ℕ} {a m : Msg} (h : (run H P (V IV st) x ω r).msgs[i]? = some (a, .msg m)) :
    i + 1 < (run H P (V IV st) x ω r).msgs.length ∨ (run H P (V IV st) x ω r).msgs.length = (V IV st).k x := by
  sorry

/-- **A run that accepts a round accepts.** -/
theorem run_acc_out {i : ℕ} {a : Msg} (h : (run H P (V IV st) x ω r).msgs[i]? = some (a, .acc)) :
    (run H P (V IV st) x ω r).out = .acc := by
  sorry

/-- A statement with a V* part reads. -/
theorem readX_of_nS (hn : 0 < nS x) : ∃ xr, readX x = some xr := by
  sorry

/-- **The compiled verifier's rounds**: the inner and staging rounds, then each V* session's, at any inner rounds. -/
theorem k_eq (ms : List (Msg × Out)) :
    (V IV st).k x = IV.k (xIOf x) + kS x + ((vstar st x ms).map VO.k).sum := by
  sorry

/-- **F1: the run reaches session 0.** Where the statement has a V* part, every inner and staging round is answered
by its coin's bytes, so the run goes past them: its message at session 0's first round is there. -/
theorem reach_sessions (hn : 0 < nS x) :
    IV.k (xIOf x) + kS x < (run H P (V IV st) x ω r).msgs.length := by
  sorry

/-- **F1: every later round reads the run's own inner and staging rounds**: messages that extend the run's first
`m ≥ k_I + k_S` and coins that extend its first `k_I + k_S` show them as `innerPairs`. `RecDeploy.innerPairs_run`
without the run's acceptance. -/
theorem innerPairs_reach {xr : XR} (hx : readX x = some xr) {m : ℕ} (hk : IV.k (xIOf x) + kS x ≤ m)
    (hm : m ≤ (run H P (V IV st) x ω r).msgs.length) (bs : List Msg) {cs : List Coin}
    (hc : cs.take (IV.k (xIOf x) + kS x) = (List.range (IV.k (xIOf x) + kS x)).map (coinSeq r)) :
    innerPairs (IV.k (xIOf x) + kS x) (((run H P (V IV st) x ω r).msgs.take m).map Prod.fst ++ bs) cs =
      (run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x) := by
  sorry

/-- **F1: the staging round as `Stage.ofMsg` reads it.** Where the statement has a V* part, the run's staging round
(round `k_I`) is its message `a`, answered by its coin's bytes, and `Stage.ofMsg tr` reads each part's public file and
the inner session's record off `a`, and the coin server's transcript `tr` of the inner rounds before it. -/
theorem stage_reads (tr : Msg → List (Msg × Out) → ByteArray) (hn : 0 < nS x) :
    ∃ a, (run H P (V IV st) x ω r).msgs[IV.k (xIOf x)]? = some (a, .msg (coinSeq r (IV.k (xIOf x))).bytes) ∧
      (∀ s, (Stage.ofMsg tr).pub x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) s =
        ((decodeStage a).bind fun p => p.1[s]?).getD ByteArray.empty) ∧
      (Stage.ofMsg tr).record x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) =
        ((decodeStage a).map Prod.snd).getD ByteArray.empty ∧
      (Stage.ofMsg tr).transcript x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x) + kS x)) =
        tr x ((run H P (V IV st) x ω r).msgs.take (IV.k (xIOf x))) := by
  sorry

end Runs

end Verity.Recursive.Schedule
