module

public import Proofs.Certifier.Witnesses
public import Proofs.Certifier.SampledLeak
public import Proofs.Certifier.Counting
public import Proofs.Certifier.RecordRows
public import Definitions.Certifier.Witness
public import Definitions.Certifier.PooledLeak
public import Definitions.Certifier.TapRows
public import Definitions.Certifier.Schedule
public import Definitions.Certifier.Ingress
public import Definitions.Certifier.Assumptions

/-!
# What holds at the instances the network attester's count and period witnesses are stated at

The instances are definitions of `Definitions.Certifier.Witness`, which the witnesses' statements read; this
module proves what `CountWitnesses` and `MonitoringPeriodWitnesses` conclude from. Each is built by hand, at a positive
instance where the guarantee allows one: several steps, a non-empty incorrect set at a positive `δ`. `Witnesses` has the
accepting verifiers at the negation `notStmt` (`leakL`, `certL`), which these extend.

* `fxLeakI`, a sender whose secret is its every step's frame length: `fxLeakDec` reads it back from the last step of
  the record, so at `acceptAll`, the verifier that accepts everything, every secret is decoded (`fxLeakObs_not_le`).
* `inI`, a certifier recording one frame a bucket: every status is complete, and it meets W2, W4 at one frame a bucket
  and J (`statusComplete_inI`, `oneRow_packed`, `oneRow_count`).
* `pooled_bound_fails`: `PooledLeakAccepted`'s bound fails at an audit that always accepts, at one window of one step
  whose incorrect set holds the one unit `Fmax = 1` allows none of, at `δ = 1/2`.
* `fxRows`, a rows format of the Python network attester's fixture `fxQueue` (`fixtureAgrees` checks its egress
  record): it is well formed (`fxRows_wf`), and over one window of four slots `fxQueueSched` fits it (`fxQueue_fits`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.Witness.SamplingGeneric (notStmt notPart)

/-! ## The verifier that accepts everything -/

theorem fxLeak_obs (z k : ℕ) : fxLeakDec (fxLeakI.obsRun z (k + 1)) = z := by
  first
    | simp [fxLeakDec, Interaction.obsRun, Interaction.trace_succ, fxLeakI]
    | simp [fxLeakDec, Interaction.obsRun, Interaction.trace_succ, fxLeakI, List.getLast?_concat]

/-- **At the verifier that accepts everything, every secret of `fxLeakI` is accepted and decoded from the records**,
over any positive number of steps. -/
theorem fxLeak_obs_univ {n : ℕ} (hn : 0 < n) :
    {z : ℕ | acceptAll (fxLeakI.obsRun z n) ∧ fxLeakDec (fxLeakI.obsRun z n) = z} = Set.univ := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  exact Set.eq_univ_of_forall fun z => ⟨trivial, fxLeak_obs z k⟩

/-- No finite count bounds the secrets of `fxLeakI` that the verifier that accepts everything passes and a decoder of
the records recovers. -/
theorem fxLeakObs_not_le {n : ℕ} (hn : 0 < n) (N : ℕ∞) (hN : N ≠ ⊤) :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z n) ∧ fxLeakDec (fxLeakI.obsRun z n) = z}.encard ≤ N := by
  rw [fxLeak_obs_univ hn, Set.encard_univ, ENat.card_eq_top_of_infinite, top_le_iff]
  exact hN

theorem two_mul_ne_top (m : ℕ) : (2 : ℕ∞) * (m : ℕ∞) ≠ ⊤ := by
  have h : (2 : ℕ∞) * (m : ℕ∞) = ((2 * m : ℕ) : ℕ∞) := by first | norm_cast | simp | (push_cast; rfl)
  rw [h]
  exact ENat.coe_ne_top _

/-- Two content assignments of the one link: the frame `[true]`, or the frame `[false]`. -/
theorem twoContents_encard : ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard = 2 :=
  Set.encard_pair fun h => by
    have := congrFun h 0
    simp at this

/-! ## Ingress -/

theorem oneRow_packed : RowLeftPacked fun s => (oneRow s).isSome := by
  first | decide | (unfold RowLeftPacked oneRow; decide)

theorem oneRow_count : (rowCount oneRow : ℕ) ≤ 1 := by
  first | decide | rfl | (simp [rowCount, oneRow]; decide)

theorem statusComplete_inI (n : ℕ) : StatusComplete n (inI.obsRun () n) := by
  intro t ht
  refine ⟨by rw [Interaction.length_obsRun]; exact ht, fun κ => ?_⟩
  rw [inI.obsRun_getElem () ht]
  rfl

/-! ## The monitoring period, pooled -/

/-- **`PooledLeakAccepted`'s bound fails at an audit that always accepts**, at `δ = 1/2` and `e = e' = 0`: one window
of one step whose incorrect set holds its one unit, which `Fmax = 1` allows none of, so the good event fails on every
coin. -/
theorem pooled_bound_fails :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ PooledGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) := by
  simp [Pr, Claims.prUniform, PooledGood] <;> norm_num

/-! ## The Python network attester's fixture -/

theorem fxRows_wf : fxRows.Wf := by
  refine ⟨List.nodup_finRange 3, List.mem_finRange, ?_⟩
  rintro ⟨c, ⟨⟨⟩⟩ | _ | _⟩ <;> fin_cases c <;> rfl

theorem fxQueue_bucket (c : Fin 3) (j : ℕ) (hj : j < fxQueueSched.cells c) : fxQueueSched.bucket c j < 10 := by
  fin_cases c <;> simp [fxQueueSched, CertifierSchedule.bucket] at hj ⊢ <;> omega

/-- **`fxQueue`'s schedule fits the format** over one window of four slots: its cells fall in buckets `4 … 9`, and no
bucket holds more than four frames. -/
theorem fxQueue_fits : (RowsFormat.recordFormat fxRows fxRows_wf 4 1).fits fxQueueSched := by
  refine ⟨fun c j hj => by simpa [fxRows] using fxQueue_bucket c j hj, fun b => ?_⟩
  rcases Nat.lt_or_ge b 10 with hb | hb
  · interval_cases b <;> decide
  · have h0 : ∀ c, laneLoad fxQueueSched c b = 0 := by
      intro c
      have hadm : ¬ (fxQueueSched.adm c ≤ b ∧ b < fxQueueSched.adm c + fxQueueSched.D) := by
        fin_cases c <;> simp [fxQueueSched] <;> omega
      simp only [laneLoad, hadm, ↓reduceIte, zero_add, List.length_eq_zero_iff, List.filter_eq_nil_iff,
        List.mem_range, beq_iff_eq]
      intro j hj
      have := fxQueue_bucket c j hj
      omega
    simp [fxRows, h0]

end NetTiming.Witness
