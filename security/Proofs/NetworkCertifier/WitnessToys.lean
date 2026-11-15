module

public import Proofs.NetworkCertifier.Witnesses
public import Proofs.NetworkCertifier.SampledLeak
public import Proofs.NetworkCertifier.Counting
public import Proofs.NetworkCertifier.RecordRows
public import Definitions.NetworkCertifier.Witness
public import Definitions.NetworkCertifier.PooledLeak
public import Definitions.NetworkCertifier.TapRows
public import Definitions.NetworkCertifier.Schedule
public import Definitions.NetworkCertifier.Ingress
public import Definitions.NetworkCertifier.Assumptions

/-!
# What holds at the instances the network certifier's count and period witnesses are stated at

The instances are definitions of `Definitions.NetworkCertifier.Witness`, which the witnesses' statements read; this
module proves what `CountWitnesses` and `PeriodWitnesses` conclude from. Each is built by hand, at a positive instance
where the guarantee allows one: several steps, two execution blocks of one advice bit each, a non-empty incorrect set at
a positive `δ`. `Witnesses` has the accepting verifiers at the negation `notStmt` (`leakL`, `certL`), which these
extend.

* `honestI P a`, the honest certifier at the Program `P` under the advice `a`: `P` accepts its run over every number of
  steps (`honestI_accepted`, `honestI_mem`).
* `acceptAll`, the verifier that accepts everything: `AccSeqsAt` at the Program's acceptance is `AccSeqs`
  (`accSeqs_eq_at`), and over any positive number of steps `AccSeqsAt acceptAll`'s sequences are infinitely many
  (`accSeqsAt_acceptAll_not_le`).
* `fxLeakI`, a sender whose secret is its every step's frame length: `fxLeakDec` reads it back from the last step of
  the wire or the record, so at `acceptAll` every secret is decoded (`fxLeak_not_le`, `fxLeakObs_not_le`).
* `inI`, a certifier recording one frame a bucket: every status is complete, and it meets W2, W4 at one frame a bucket
  and J (`statusComplete_inI`, `oneRow_packed`, `oneRow_count`).
* `sampProg`, sampled Bernoulli at rate `1/2` (`sampDesign`), its first input the ingress (`notLay`). At `δ = 1/4`
  both `∅` and `{0}` are possible incorrect sets (`single_mem_sampFaulty`), each committing at most its three gates'
  bits (`sampBits`), and one input is free (`sampFree`). The step `obs1`, the empty frame the negation of `false`
  holds, is accepted (`obs1_sampAcc`).
* `pooled_bound_fails`: `PooledLeakAccepted`'s bound fails at an audit that always accepts, at one window of one step
  whose incorrect set holds the one unit `Fmax = 1` allows none of, at `δ = 1/2`.
* `fxRows`, a rows format of the Python network certifier's fixture `fxQueue` (`fixtureAgrees` checks its egress
  record): it is well formed (`fxRows_wf`), and over one window of four slots `fxQueueSched` fits it (`fxQueue_fits`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.Witness.SamplingGeneric (notStmt notPart)

/-! ## The honest certifier -/

/-- **`P` accepts the honest certifier's run under `a`**, over every number of steps. -/
theorem honestI_accepted {L K S : ℕ} {Adv : Type} (P : EgressPrediction L S Unit Adv) (a : Adv) (n : ℕ) :
    Accepted P unitOp n ((honestI (K := K) P a).obsRun () n) a := by
  intro t ht
  have hlen : t < ((honestI (K := K) P a).obsRun () n).length := by
    rw [Interaction.length_obsRun]
    exact ht
  refine ⟨hlen, ?_, fun κ => ?_⟩
  · rw [(honestI (K := K) P a).obsRun_getElem () ht]
    have hxs : ((honestI (K := K) P a).trace () (t + 1)).2.1 =
        ingressUpTo unitOp ((honestI (K := K) P a).obsRun () n) t :=
      List.ext_getElem (by rw [((honestI (K := K) P a).trace_length () (t + 1)).2.1]; simp [ingressUpTo])
        fun _ _ _ => Subsingleton.elim _ _
    exact congrArg (fun ins => P.rows t ins a) hxs
  · rw [(honestI (K := K) P a).obsRun_getElem () ht]
    rfl

/-- The honest certifier's run is among `P`'s accepted sequences at `D = 1`. -/
theorem honestI_mem {L K S : ℕ} {Adv : Type} (P : EgressPrediction L S Unit Adv) (a : Adv) (n : ℕ) :
    (honestI (K := K) P a).obsRun () n ∈ AccSeqs P (unitOp (K := K)) n 1 := by
  refine ⟨Interaction.length_obsRun _ () n, fun x hx => ?_, a, honestI_accepted P a n⟩
  obtain ⟨k, xs, as, rfl⟩ := Interaction.mem_obsRun _ () n x hx
  exact Nat.one_pos

/-! ## The verifier that accepts everything -/

theorem accSeqs_eq_at {L K S : ℕ} {In Adv : Type} (P : EgressPrediction L S In Adv)
    (op : ℕ → List (Obs L K S) → In) (n D : ℕ) : AccSeqs P op n D = AccSeqsAt (AcceptedUpTo P op n) n D :=
  rfl

/-- **At the verifier that accepts everything, the sequences of a positive number of steps are infinitely many**: one
link of `S > 0` slots carries any payload at every step. -/
theorem accSeqsAt_acceptAll_infinite {S : ℕ} (hS : 0 < S) (k : ℕ) :
    (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := S)) (k + 1) 1).Infinite := by
  refine Set.infinite_of_injective_forall_mem
    (f := fun m : ℕ => List.replicate (k + 1)
      (⟨fun _ _ => some (List.replicate m true), 0, fun _ => .complete⟩ : Obs 1 1 S)) ?_ ?_
  · intro a b hab
    have h := congrArg (fun o : List (Obs 1 1 S) => o.head?.map fun x => (x.rows 0 ⟨0, hS⟩).map List.length) hab
    have h' : some (some (List.replicate a true).length) = some (some (List.replicate b true).length) := h
    simpa using h'
  · intro m
    simp [AccSeqsAt, acceptAll]

/-- No finite count bounds `AccSeqsAt acceptAll` over a positive number of steps. -/
theorem accSeqsAt_acceptAll_not_le {S n : ℕ} (hS : 0 < S) (hn : 0 < n) (N : ℕ) :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := S)) n 1).encard ≤ N := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  rw [(accSeqsAt_acceptAll_infinite hS k).encard_eq]
  simp

theorem fxLeak_wire (z k : ℕ) : fxLeakDec (fxLeakI.wireRun z (k + 1)) = z := by
  first
    | simp [fxLeakDec, Interaction.wireRun, Interaction.trace_succ, fxLeakI]
    | simp [fxLeakDec, Interaction.wireRun, Interaction.trace_succ, fxLeakI, List.getLast?_concat]

theorem fxLeak_obs (z k : ℕ) : fxLeakDec (fxLeakI.obsRun z (k + 1)) = z := by
  first
    | simp [fxLeakDec, Interaction.obsRun, Interaction.trace_succ, fxLeakI]
    | simp [fxLeakDec, Interaction.obsRun, Interaction.trace_succ, fxLeakI, List.getLast?_concat]

/-- **At the verifier that accepts everything, every secret of `fxLeakI` is accepted and decoded from the wire**, over
any positive number of steps. -/
theorem fxLeak_wire_univ {n : ℕ} (hn : 0 < n) :
    {z : ℕ | acceptAll (fxLeakI.obsRun z n) ∧ fxLeakDec (fxLeakI.wireRun z n) = z} = Set.univ := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  exact Set.eq_univ_of_forall fun z => ⟨trivial, fxLeak_wire z k⟩

/-- As `fxLeak_wire_univ`, decoded from the records. -/
theorem fxLeak_obs_univ {n : ℕ} (hn : 0 < n) :
    {z : ℕ | acceptAll (fxLeakI.obsRun z n) ∧ fxLeakDec (fxLeakI.obsRun z n) = z} = Set.univ := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  exact Set.eq_univ_of_forall fun z => ⟨trivial, fxLeak_obs z k⟩

/-- No finite count bounds the secrets of `fxLeakI` that the verifier that accepts everything passes and a decoder of
the wire recovers. -/
theorem fxLeak_not_le {n : ℕ} (hn : 0 < n) (N : ℕ∞) (hN : N ≠ ⊤) :
    ¬ {z : ℕ | acceptAll (fxLeakI.obsRun z n) ∧ fxLeakDec (fxLeakI.wireRun z n) = z}.encard ≤ N := by
  rw [fxLeak_wire_univ hn, Set.encard_univ, ENat.card_eq_top_of_infinite, top_le_iff]
  exact hN

/-- As `fxLeak_not_le`, with the decoder reading the records. -/
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

/-! ## A sampled audit with a possible incorrect proof block -/

theorem sampLay_fits : notLay.Fits sampProg 1 := by
  refine ⟨fun t _ ℓ σ g hg => ?_, fun t _ g hg => ?_, fun t _ t' _ htt' => absurd htt' (by omega)⟩
  · simp [notLay, slotGates] at hg
    subst hg
    decide
  · simp [notLay] at hg
    subst hg
    rfl

/-- With nothing held and the first input `false`, the circuit computes its negation `true`. -/
theorem sampProg_val (inp : ℕ → Bool) (h0 : inp 0 = false) : valOver sampProg inp (fun _ => none) 1 = true := by
  have hg0 : sampProg.gates[0]? = some .input := rfl
  have hg1 : sampProg.gates[1]? = some (.not 0) := rfl
  have hv0 : valOver sampProg inp (fun _ => none) 0 = false := by
    rw [valOver]
    simp [hg0, BGate.val, h0]
  rw [valOver]
  simp [hg1, BGate.val, hv0]

theorem sampPa (B : Finset ℕ) :
    sampDesign.Pa sampPart B =
      (((2 - 1 : ℕ) : ℝ) / ((2 : ℕ) : ℝ)) ^ (Claims.Sampling.window 0 1 B).card * 1 := by
  first
    | rfl
    | simp [Design.Pa, classesPa, OneStage.Pa, sampDesign, sampPart, strataSizes]

theorem sampAllSlots : allSlots sampDesign sampPart = ∅ := by
  first | rfl | decide | simp [allSlots, allSlotsFrom, sampDesign, sampPart]

theorem sampFaulty_mem {F : Finset ℕ} (hsub : F ⊆ Finset.range 1) (hPa : (1 / 4 : ℝ) < sampDesign.Pa sampPart F) :
    F ∈ faultySets sampDesign sampPart (1 / 4) := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset]
  exact ⟨hsub, hPa, by rw [sampAllSlots]; exact Finset.disjoint_empty_right _⟩

theorem empty_mem_sampFaulty : (∅ : Finset ℕ) ∈ faultySets sampDesign sampPart (1 / 4) :=
  sampFaulty_mem (Finset.empty_subset _) (by rw [sampPa]; norm_num [Claims.Sampling.window])

/-- **The one proof block is a possible incorrect set** at `δ = 1/4`: the design misses it with probability `1/2`. -/
theorem single_mem_sampFaulty : ({0} : Finset ℕ) ∈ faultySets sampDesign sampPart (1 / 4) :=
  sampFaulty_mem (by decide) (by
    rw [sampPa, show (Claims.Sampling.window 0 1 ({0} : Finset ℕ)).card = 1 by
      first | decide | rfl | simp [Claims.Sampling.window]]
    norm_num)

/-- Each possible incorrect set at `δ = 1/4` commits at most the proof block's three gates' bits. -/
theorem sampBits : ∀ F ∈ faultySets sampDesign sampPart (1 / 4), faultyBits sampProg sampPart F ≤ 3 := by
  intro F hF
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  have h : faultyBits sampProg sampPart F ≤ faultyBits sampProg sampPart (Finset.range 1) := by
    unfold faultyBits
    exact Finset.sum_le_sum_of_subset hF.1
  have h3 : faultyBits sampProg sampPart (Finset.range 1) = 3 := by
    first
      | decide
      | rfl
      | simp [faultyBits, Claims.Sampling.committed, Prog.Boundary, sampProg, sampPart, Partition.blocks]
  omega

/-- The second input is the one free input over one step. -/
theorem sampFree : (freeInputs sampProg notLay 1 \ ∅).card = 1 := by
  first | decide | rfl | simp [freeInputs, sampProg, notLay]

theorem sampAs_encard : (AsReg sampProg notLay 1 ∅ fun _ => false).encard ≤ ((2 ^ 1 : ℕ) : ℕ∞) := by
  have h := encard_asReg_le sampProg notLay 1 ∅ fun _ => false
  rwa [sampFree] at h

/-- **`obs1` is accepted with the empty incorrect set**, its free input `false`, at one step at `δ = 1/4`. -/
theorem obs1_sampAcc :
    [obs1] ∈ AccSeqsSampled sampProg sampPart sampDesign notLay 1 (AsReg sampProg notLay 1 ∅ fun _ => false) (1 / 4)
      (fun _ _ => [false]) 1 := by
  refine ⟨rfl, by simp [obs1], ∅, empty_mem_sampFaulty, fun _ => false,
    ⟨fun g hg => absurd hg (Finset.notMem_empty g), fun _ _ => rfl⟩, fun _ => none,
    fun _ _ h => by simp at h, fun t ht => ?_⟩
  obtain rfl : t = 0 := by omega
  refine ⟨Nat.one_pos, ?_, fun _ => rfl⟩
  funext ℓ σ
  have hin : notLay.inputOf 1 [[false]] (fun _ => false) 0 = false := rfl
  have hrow : rowOf (valOver sampProg (notLay.inputOf 1 [[false]] (fun _ => false)) (fun _ => none))
      (notLay.egress 0 ℓ) σ = if valOver sampProg (notLay.inputOf 1 [[false]] (fun _ => false)) (fun _ => none) 1 then
        some (([] : List ℕ).map (valOver sampProg (notLay.inputOf 1 [[false]] (fun _ => false)) (fun _ => none)))
      else none := rfl
  show some [] = rowOf (valOver sampProg (notLay.inputOf 1 [[false]] (fun _ => false)) (fun _ => none))
    (notLay.egress 0 ℓ) σ
  rw [hrow, sampProg_val _ hin]
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

/-! ## The Python network certifier's fixture -/

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
