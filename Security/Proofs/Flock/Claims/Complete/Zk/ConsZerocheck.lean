import Proofs.Flock.Claims.Complete.Zk.PiopGo
import Proofs.Flock.Claims.Complete.Replay.ZerocheckAlgebra
import Proofs.Flock.Soundness.Discharge.Composed.Inner.ZerocheckMasked

/-!
# The zerocheck's two constraints vanish at the masked prover's pads

The masked prover's zerocheck proof (`ZcZ.proof`) is the plain prover's (`Zc.proof`) with each value plus its pad, which
is the soundness analysis's `Masks` (`zcZ_masks`). Its evaluations of the replay's zerocheck forms
(`ZerocheckMasked.c_eval_eval`, `final_eval`) then give:

* `cEval_zero`: `zerocheck c_eval` vanishes at any pads, at every coin (`final_c` is `Pc(z)`);
* `finalAB_zero`: `zerocheck final a·b` vanishes at pads with `h_ab = h_a·h_b`, for a trace satisfying the block R1CS
  bitwise and outer coins other than `1` (`Zc.fold_final`).
-/

namespace FlockComplete.Replay

open Flock FlockSoundness.Discharge.Composed

/-- **The masked zerocheck proof is the plain one masked by the pads**, on all `m − 6` residual rounds. -/
theorem zcZ_masks (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) :
    ZerocheckMasked.Masks lay h (Zc.proof m cs tr) (ZcZ.proof lay h m cs tr) (m - 6) where
  round1ab i hi := by
    simp only [ZcZ.proof, ZcZ.r1ab, Zc.proof]
    rw [getElem!_pos _ i (by simpa using hi)]
    simp
  round1c i hi := by
    simp only [ZcZ.proof, ZcZ.r1c, Zc.proof]
    rw [getElem!_pos _ i (by simpa using hi)]
    simp
  rounds j hj := by
    simp only [ZcZ.proof, Zc.proof]
    rw [pushes_get _ _ j hj, pushes_get _ _ j hj]
    rfl
  finalA := rfl
  finalB := rfl
  finalC := rfl

/-- **`zerocheck c_eval` vanishes at the masked prover's pads**, at any pads and every coin. -/
theorem cEval_zero (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (cs : Zc.Challenges) (tr : Trace) :
    ZerocheckMasked.evalL ((ZerocheckMasked.computedC lay (ZcZ.proof lay h m cs tr) cs.z).add
      (ZerocheckMasked.fcL lay (ZcZ.proof lay h m cs tr))) h = 0 := by
  rw [ZerocheckMasked.c_eval_eval lay h _ _ (m - 6) (zcZ_masks lay h m cs tr) cs.z]
  exact ZerocheckMasked.add_self' _

/-- **`zerocheck final a·b` vanishes at the masked prover's pads** where `h_ab = h_a·h_b`, for a trace satisfying the
block R1CS bitwise and outer coins other than `1`. -/
theorem finalAB_zero (lay : Zk.PadLayout) (h : Array F128) {m : ℕ} (hm : 14 ≤ m) (hn : m - 6 ≤ lay.nMlv)
    (cs : Zc.Challenges) (tr : Trace) (hR1CS : ∀ i < 2 ^ m, (tr.az[i]! && tr.bz[i]!) = tr.z[i]!)
    (hc1 : ∀ k < m - 13, cs.c1 k ≠ 1) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) :
    ZerocheckMasked.evalL ((ZerocheckMasked.running lay (ZcZ.proof lay h m cs tr) (cs.r m) (pushes cs.ρ (m - 6)) cs.z
      (m - 6)).add (ZerocheckMasked.prodL lay (ZcZ.proof lay h m cs tr))) h = 0 := by
  have hf := Zc.fold_final hm cs tr hR1CS hc1
  rw [Zc.zcFold_fst (cs.r m) (Zc.proof m cs tr) (fun i => #[cs.ρ i]) (pushes cs.ρ (m - 6)) (m - 6)
    (fun k hk => (pushes_get cs.ρ (m - 6) k hk).symm) cs.z (m - 6) le_rfl] at hf
  rw [ZerocheckMasked.final_eval lay h _ _ (m - 6) hn (zcZ_masks lay h m cs tr) hab, hf]
  exact ZerocheckMasked.add_self' _

end FlockComplete.Replay
