import Proofs.NetworkCertifier.Witnesses
import Proofs.NetworkCertifier.SampledLeak
import Proofs.NetworkCertifier.Counting
import Proofs.NetworkCertifier.RecordRows
import Definitions.NetworkCertifier.HorizonLeak
import Definitions.NetworkCertifier.TapRows
import Definitions.NetworkCertifier.Schedule
import Definitions.NetworkCertifier.Ingress
import Definitions.NetworkCertifier.Assumptions

/-!
# The instances the network certifier's count and period witnesses are stated at

Each built by hand, at a positive instance where the guarantee allows one: several steps, two values of content advice,
two clock-sync values, a non-empty faulty set at a positive `δ`. `CountWitnesses` and `PeriodWitnesses` state the
witnesses; `Witnesses` has the negation `notStmt` and its accepting verifier `leakD`, which these extend.

* `honestI P a`: the honest certifier at the Program `P` under the advice `a`, which records and puts on the wire `P`'s
  rows, so `P` accepts its run at every horizon (`honestI_accepted`, `honestI_mem`).
* `fxRate`: the constant-rate Program over two windows of two buckets, one link of two slots, the sync set `syncTwo`
  (two offsets, one rate) and the content advice `Bool`, whose two frames carry the advice bit and its negation.
  `fxRate1` is it at the one-value sync set `syncOne`, and `fxFixed` it with its clock fixed.
* `acceptAll`, the verifier that accepts everything: `AccSeqsAt acceptAll` is `AccSeqs` with it in place of the
  Program's acceptance, and over any positive horizon its sequences are infinitely many
  (`accSeqsAt_acceptAll_not_le`).
* `fxLeakI`: a sender whose secret is its every step's frame length, which `fxLeakDec` reads back from the last step of
  the wire or the record, so at `acceptAll` every secret is decoded (`fxLeak_not_le`, `fxLeakObs_not_le`).
* `inI`: a certifier recording one frame a bucket in the first of two slots, every status complete, meeting W2, W4 at
  one frame a bucket and J (`statusComplete_inI`, `oneRow_packed`, `oneRow_count`).
* `sampProg`: two inputs and the negation of the first, the first the ingress (`notLay`), one replay unit `{0, 1, 2}`
  drawn Bernoulli at rate `1/2` (`sampDesign`). At `δ = 1/4` both `∅` and `{0}` are possible faulty sets
  (`single_mem_sampFaulty`), each committing at most its three gates' bits (`sampBits`), and one input is free
  (`sampFree`). The step `obs1`, the empty frame the negation of `false` holds, is accepted (`obs1_sampAcc`).
* `tapD`: `leakD` with every opening passing, where the network certifier's openings `tapOps` register `obs0`
  (`tapOps_opens`), so at every `δ` no witness is in `RLeakTap` (`tapD_noRLeakTap`), and at `δ = 1/8` the per-window
  leak bound of `PeriodLeakAccepted` fails at its silent prover (`tapD_bound_fails`).
* `horizon_bound_fails`: `HorizonLeakAccepted`'s bound fails at an audit that always accepts, at one window of one step
  whose faulty set holds the one unit `Fmax = 1` allows none of, at `δ = 1/2`.
* `fxRows`: a rows format of the Python network certifier's fixture `fxQueue` (three connections, `fixtureAgrees`
  checks its egress record), one window of ten buckets of four slots, which `fxQueueSched` fits (`fxQueue_fits`).
-/

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling

/-! ## The honest certifier -/

/-- The operator, which sends nothing. -/
def unitOp {L K S : ℕ} : ℕ → List (Obs L K S) → Unit := fun _ _ => ()

/-- **The honest certifier at `P` under the advice `a`**: at each step it records and puts on the wire `P`'s row on
the ingress so far, jitter class 0 and every status complete, whatever the sender does. -/
def honestI {L K S : ℕ} {Adv : Type} (P : Program L S Unit Adv) (a : Adv) :
    Interaction L K S Unit Unit Unit (Obs L K S) where
  certifier :=
    { obs := fun k xs _ => ⟨P.rows k xs a, 0, fun _ => .complete⟩
      wire := fun k xs _ => ⟨P.rows k xs a, 0, fun _ => .complete⟩
      fault := fun _ => False }
  op := unitOp
  snd := fun _ _ _ => ()

/-- **`P` accepts the honest certifier's run under `a`**, at every horizon. -/
theorem honestI_accepted {L K S : ℕ} {Adv : Type} (P : Program L S Unit Adv) (a : Adv) (n : ℕ) :
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
theorem honestI_mem {L K S : ℕ} {Adv : Type} (P : Program L S Unit Adv) (a : Adv) (n : ℕ) :
    (honestI (K := K) P a).obsRun () n ∈ AccSeqs P (unitOp (K := K)) n 1 := by
  refine ⟨Interaction.length_obsRun _ () n, fun x hx => ?_, a, honestI_accepted P a n⟩
  obtain ⟨k, xs, as, rfl⟩ := Interaction.mem_obsRun _ () n x hx
  exact Nat.one_pos

/-! ## The shipped schedule -/

/-- Two clock-sync values: offsets `0` and `1`, rate `1`. -/
def syncTwo : SyncSet where
  deltas := {0, 1}
  rhos := {1}
  deltas_nonempty := ⟨0, Finset.mem_insert_self 0 {1}⟩
  rhos_nonempty := ⟨1, Finset.mem_singleton_self 1⟩
  rhos_pos := fun ρ h => by rw [Finset.mem_singleton.1 h]; norm_num

/-- `syncTwo`'s value at offset `0`. -/
def syncTwo0 : syncTwo.Elem := (⟨0, Finset.mem_insert_self 0 {1}⟩, ⟨1, Finset.mem_singleton_self 1⟩)

/-- One clock-sync value, offset `0` and rate `1`. -/
def syncOne : SyncSet where
  deltas := {0}
  rhos := {1}
  deltas_nonempty := ⟨0, Finset.mem_singleton_self 0⟩
  rhos_nonempty := ⟨1, Finset.mem_singleton_self 1⟩
  rhos_pos := fun ρ h => by rw [Finset.mem_singleton.1 h]; norm_num

/-- `syncOne`'s one value. -/
def syncOne0 : syncOne.Elem := (⟨0, Finset.mem_singleton_self 0⟩, ⟨1, Finset.mem_singleton_self 1⟩)

/-- Two frames on the one link, anchored at bucket 0: the content advice bit, then its negation. -/
def fxFrames : List Unit → Bool → Fin 1 → List LogicalFrame := fun _ c _ => [⟨[c], 0, 0⟩, ⟨[!c], 0, 1⟩]

/-- **The constant-rate Program** over two windows of two buckets, one link of two slots, the sync set `syncTwo`. -/
def fxRate : Program 1 2 Unit (Bool × (Fin 2 → syncTwo.Elem)) :=
  constantRate 2 2 syncTwo fxFrames

/-- `fxRate`'s advice: the bit `true`, offset `0` in both windows. -/
def fxAdv : Bool × (Fin 2 → syncTwo.Elem) := (true, fun _ => syncTwo0)

/-- `fxRate` at the one-value sync set `syncOne`. -/
def fxRate1 : Program 1 2 Unit (Bool × (Fin 2 → syncOne.Elem)) :=
  constantRate 2 2 syncOne fxFrames

/-- `fxRate1`'s advice. -/
def fxAdv1 : Bool × (Fin 2 → syncOne.Elem) := (true, fun _ => syncOne0)

/-- `fxRate` with its clock fixed at `syncTwo0`: a Program whose clock-sync advice is `Unit`. -/
def fxFixed : Program 1 2 Unit (Bool × (Fin 2 → Unit)) where
  rows t ins a ℓ := fxRate.rows t ins (a.1, fun _ => syncTwo0) ℓ

/-- `fxFixed`'s advice. -/
def fxAdvU : Bool × (Fin 2 → Unit) := (true, fun _ => ())

/-- The honest certifier at `fxRate` under `fxAdv`. -/
def fxHonestI : Interaction 1 1 2 Unit Unit Unit (Obs 1 1 2) := honestI (K := 1) fxRate fxAdv

/-! ## The verifier that accepts everything -/

/-- The observation sequences of length `n` with jitter below `D` that the acceptance `acc` passes: `AccSeqs` with
`acc` in place of the Program's acceptance (`accSeqs_eq_at`). -/
def AccSeqsAt {L K S : ℕ} (acc : List (Obs L K S) → Prop) (n D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ acc o}

/-- The verifier that accepts everything. -/
def acceptAll {L K S : ℕ} (_ : List (Obs L K S)) : Prop := True

theorem accSeqs_eq_at {L K S : ℕ} {In Adv : Type} (P : Program L S In Adv) (op : ℕ → List (Obs L K S) → In)
    (n D : ℕ) : AccSeqs P op n D = AccSeqsAt (AcceptedUpTo P op n) n D :=
  rfl

/-- **At the verifier that accepts everything, a positive horizon's sequences are infinitely many**: one link of
`S > 0` slots carries any payload at every step. -/
theorem accSeqsAt_acceptAll_infinite {S : ℕ} (hS : 0 < S) (k : ℕ) :
    (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := S)) (k + 1) 1).Infinite := by
  refine Set.infinite_of_injective_forall_mem
    (f := fun m : ℕ => List.replicate (k + 1)
      (⟨fun _ _ => some (List.replicate m true), 0, fun _ => .complete⟩ : Obs 1 1 S)) ?_ ?_
  · intro a b hab
    have := congrArg (fun o : List (Obs 1 1 S) => o.head?.map fun x => (x.rows 0 ⟨0, hS⟩).map List.length) hab
    simpa using this
  · intro m
    simp [AccSeqsAt, acceptAll]

/-- No finite count bounds `AccSeqsAt acceptAll` over a positive horizon. -/
theorem accSeqsAt_acceptAll_not_le {S n : ℕ} (hS : 0 < S) (hn : 0 < n) (N : ℕ) :
    ¬ (AccSeqsAt (acceptAll (L := 1) (K := 1) (S := S)) n 1).encard ≤ N := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  rw [(accSeqsAt_acceptAll_infinite hS k).encard_eq]
  simp

/-- **A sender whose secret shows**: its secret `z` is its every action, and the certifier records and puts on the wire
one frame of `z` bits in the first slot, jitter class 0, its one status complete. -/
def fxLeakI : Interaction 1 1 2 ℕ Unit ℕ (Obs 1 1 2) where
  certifier :=
    { obs := fun _ _ as => ⟨fun _ σ => if σ = 0 then some (List.replicate (as.getLast?.getD 0) true) else none, 0,
        fun _ => .complete⟩
      wire := fun _ _ as => ⟨fun _ σ => if σ = 0 then some (List.replicate (as.getLast?.getD 0) true) else none, 0,
        fun _ => .complete⟩
      fault := fun _ => False }
  op := unitOp
  snd := fun z _ _ => z

/-- `fxLeakI`'s decoder: the length of the last step's first frame. -/
def fxLeakDec (w : List (Obs 1 1 2)) : ℕ := ((w.getLast?.bind fun o => o.rows 0 0).map List.length).getD 0

theorem fxLeak_wire (z k : ℕ) : fxLeakDec (fxLeakI.wireRun z (k + 1)) = z := by
  first
    | simp [fxLeakDec, Interaction.wireRun, Interaction.trace_succ, fxLeakI]
    | simp [fxLeakDec, Interaction.wireRun, Interaction.trace_succ, fxLeakI, List.getLast?_concat]

theorem fxLeak_obs (z k : ℕ) : fxLeakDec (fxLeakI.obsRun z (k + 1)) = z := by
  first
    | simp [fxLeakDec, Interaction.obsRun, Interaction.trace_succ, fxLeakI]
    | simp [fxLeakDec, Interaction.obsRun, Interaction.trace_succ, fxLeakI, List.getLast?_concat]

/-- **At the verifier that accepts everything, every secret of `fxLeakI` is accepted and decoded from the wire**, over
any positive horizon. -/
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

theorem two_mul_ne_top (X : ℕ) : (2 : ℕ∞) * (X : ℕ∞) ≠ ⊤ := by
  have h : (2 : ℕ∞) * (X : ℕ∞) = ((2 * X : ℕ) : ℕ∞) := by first | norm_cast | simp
  rw [h]
  exact ENat.coe_ne_top _

/-- Two content assignments of the one link: the frame `[true]`, or the frame `[false]`. -/
theorem twoContents_encard : ({fun _ => [[true]], fun _ => [[false]]} : Set (Fin 1 → List (List Bool))).encard = 2 :=
  Set.encard_pair fun h => by
    have := congrFun h 0
    simp at this

/-! ## Ingress -/

/-- One frame `[true]` in the first of two slots. -/
def oneRow : Row 2 (List Bool) := fun σ => if σ = 0 then some [true] else none

theorem oneRow_packed : RowLeftPacked fun s => (oneRow s).isSome := by
  first | decide | (unfold RowLeftPacked oneRow; decide)

theorem oneRow_count : (rowCount oneRow : ℕ) ≤ 1 := by
  first | decide | rfl | (simp [rowCount, oneRow]; decide)

/-- **A certifier at one frame a bucket**: at every step it records and puts on the wire `oneRow` on its one link,
jitter class 0, its one status complete. -/
def inI : Interaction 1 1 2 Unit Unit Unit (Obs 1 1 2) where
  certifier :=
    { obs := fun _ _ _ => ⟨fun _ => oneRow, 0, fun _ => .complete⟩
      wire := fun _ _ _ => ⟨fun _ => oneRow, 0, fun _ => .complete⟩
      fault := fun _ => False }
  op := unitOp
  snd := fun _ _ _ => ()

theorem statusComplete_inI (n : ℕ) : StatusComplete n (inI.obsRun () n) := by
  intro t ht
  refine ⟨by rw [Interaction.length_obsRun]; exact ht, fun κ => ?_⟩
  rw [inI.obsRun_getElem () ht]
  rfl

/-! ## A sampled audit with a possible faulty unit -/

/-- Two inputs and the negation of the first, the negation returned. -/
def sampProg : Claims.Sampling.Prog := ⟨[.input, .not 0, .input], [1]⟩

/-- One class of one stratum of one replay unit, the proof unit `{0, 1, 2}`. -/
def sampPart : Claims.Sampling.Partition := ⟨[[[[[0, 1, 2]]]]]⟩

/-- The one class drawn Bernoulli at rate `1/2`, every proof unit of a drawn replay unit checked. -/
def sampDesign : Claims.Sampling.Design := ⟨[⟨.bernoulli 1 2, none⟩]⟩

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
    | simp [Design.Pa, classesPa, Rule.Pa, OneStage.Pa, sampDesign, sampPart, strataSizes, ruSizes]

theorem sampAllSlots : allSlots sampDesign sampPart = ∅ := by
  first | rfl | decide | simp [allSlots, allSlotsFrom, sampDesign, sampPart]

theorem sampFaulty_mem {F : Finset ℕ} (hsub : F ⊆ Finset.range 1) (hPa : (1 / 4 : ℝ) < sampDesign.Pa sampPart F) :
    F ∈ faultySets sampDesign sampPart (1 / 4) := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset]
  exact ⟨hsub, hPa, by rw [sampAllSlots]; exact Finset.disjoint_empty_right _⟩

theorem empty_mem_sampFaulty : (∅ : Finset ℕ) ∈ faultySets sampDesign sampPart (1 / 4) :=
  sampFaulty_mem (Finset.empty_subset _) (by rw [sampPa]; norm_num [Claims.Sampling.window])

/-- **The one replay unit is a possible faulty set** at `δ = 1/4`: the design misses it with probability `1/2`. -/
theorem single_mem_sampFaulty : ({0} : Finset ℕ) ∈ faultySets sampDesign sampPart (1 / 4) :=
  sampFaulty_mem (by decide) (by
    rw [sampPa, show (Claims.Sampling.window 0 1 ({0} : Finset ℕ)).card = 1 by
      first | decide | rfl | simp [Claims.Sampling.window]]
    norm_num)

/-- Each possible faulty set at `δ = 1/4` commits at most the replay unit's three gates' bits. -/
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
      | simp [faultyBits, Claims.Sampling.committed, Prog.Boundary, sampProg, sampPart, Partition.rus, RU.gates]
  omega

/-- The second input is the one free input over one step. -/
theorem sampFree : (freeInputs sampProg notLay 1 \ ∅).card = 1 := by
  first | decide | rfl | simp [freeInputs, sampProg, notLay]

theorem sampAs_encard : (AsReg sampProg notLay 1 ∅ fun _ => false).encard ≤ ((2 ^ 1 : ℕ) : ℕ∞) := by
  have h := encard_asReg_le sampProg notLay 1 ∅ fun _ => false
  rwa [sampFree] at h

/-- One step: an empty frame on the link, every status complete, what the negation of the ingress `false` holds. -/
def obs1 : Obs 1 1 1 := ⟨fun _ _ => some [], 0, fun _ => .complete⟩

/-- **`obs1` is accepted with the empty faulty set**, its free input `false`, at one step at `δ = 1/4`. -/
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

/-! ## The network certifier's openings at the negation -/

theorem obs0_covered : Covered 1 [obs0] := by
  intro t ht _ κ
  obtain rfl : t = 0 := by omega
  rfl

/-- At any `δ`, the only possible faulty set of `notStmt` is empty: its one unit is drawn `all`. -/
theorem notFaulty_empty {δ : ℝ} {F : Finset ℕ} (hF : F ∈ faultySets notStmt.design notPart δ) : F = ∅ := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  obtain ⟨hsub, -, hdis⟩ := hF
  refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
  have hv1 : v < 1 := Finset.mem_range.1 (hsub hv)
  obtain rfl : v = 0 := by omega
  exact Finset.disjoint_left.1 hdis hv (by decide)

/-- **`obs0` is accepted under no possible faulty set**, at every `δ` and whatever the inputs `As` admits: the only
possible faulty set is empty, so no gate is held, and the circuit, the negation of the ingress `false`, holds a frame
where `obs0` has none. -/
theorem obs0_notAcc (δ : ℝ) (As : Set (ℕ → Bool)) :
    [obs0] ∉ AccSeqsSampled notProg notPart notStmt.design notLay 1 As δ (fun _ _ => [false]) 1 := by
  intro h
  obtain ⟨-, -, F, hF, a, -, free, hfree, hacc⟩ := h
  obtain rfl := notFaulty_empty hF
  have hnone : free = fun _ => none := funext fun g => by
    cases hg : free g with
    | none => rfl
    | some b =>
      obtain ⟨v, hv, -⟩ := hfree g b hg
      simp at hv
  subst hnone
  obtain ⟨_, hrows, -⟩ := hacc 0 Nat.one_pos
  have hin : notLay.inputOf 1 [[false]] a 0 = false := rfl
  have h1 : (none : Option (List Bool)) =
      rowOf (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)) (notLay.egress 0 0) 0 :=
    congrFun (congrFun hrows 0) 0
  have hrow : rowOf (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)) (notLay.egress 0 0) 0 =
      if valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none) 1 then
        some (([] : List ℕ).map (valOver notProg (notLay.inputOf 1 [[false]] a) (fun _ => none)))
      else none := rfl
  rw [hrow, notProg_val _ hin] at h1
  simp at h1

/-- The network certifier's openings at `notStmt`: the input `false` and the negation's presence gate `false`. -/
def tapOps : List (Opening Bool) := [⟨0, false, []⟩, ⟨1, false, []⟩]

section Toy

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- `leakD` with every opening passing. -/
noncomputable def tapD (Coin : Type) : Claims.Sampling.Deploy Coin Unit Unit :=
  { leakD Coin with commit := accCm }

/-- **The network certifier's openings register `obs0`**, under every opening passing: they open the input and the
egress slot's presence gate, both `false`, so the step's row holds no frame and its ingress is `false`. -/
theorem tapOps_opens (H : Msg → Msg) :
    TapOpens H accCm notLay 1 (fun _ _ => [false]) notStmt tapOps [obs0] := by
  refine ⟨fun q _ => rfl, fun t ht => ?_⟩
  obtain rfl : t = 0 := by omega
  refine ⟨fun ℓ σ g hg => ?_, fun g hg => ?_, Nat.one_pos, fun ℓ => ?_, rfl⟩
  · simp [notLay, slotGates] at hg
    subst hg
    exact ⟨fun b hb => by simp [notStmt, notProg] at hb, ⟨1, false, []⟩, by simp [tapOps], rfl⟩
  · simp [notLay] at hg
    subst hg
    exact ⟨⟨0, false, []⟩, by simp [tapOps], rfl⟩
  · funext σ
    rfl

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **No witness is in `RLeakTap`** at `tapD`, at one step at every `δ`, whatever the inputs `As` admits: the network
certifier registers `obs0` (`tapOps_opens`), which no possible faulty set accepts (`obs0_notAcc`). -/
theorem tapD_noRLeakTap (H : Msg → Msg) (δ : ℝ) (As : Claims.Sampling.AuditStatement → (ℕ → Bool) → Set (ℕ → Bool))
    (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ RLeakTap H (tapD Coin).commit (tapD Coin).parse (tapD Coin).partition 1 1 1 (fun _ => notLay) (fun _ => 1) δ
      As 1 (fun _ => tapOps) x w := by
  rintro ⟨s, P, hs, hP, h⟩
  obtain rfl : notStmt = s := Option.some.inj hs
  obtain rfl : notPart = P := Option.some.inj hP
  exact obs0_notAcc δ _ (h (fun _ _ => [false]) [obs0] (tapOps_opens H) obs0_covered rfl (by simp [obs0]))

theorem tapD_bext (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Bext H (tapD Coin).V (Claims.Sampling.E (tapD Coin) decode) Unit silentP x).Pr H = 0 := by
  simp only [Bext]
  rw [dite_eq_left (show 0 < (tapD Coin).V.k x from Nat.one_pos)]
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp [ownLog_noHash _ _ _ (noHash_E (tapD Coin) (fun _ _ _ => rfl) _ _ _ _), findColl])]
  exact Pr_false

/-- With every opening passing at no call, `B_tap` checks the network certifier's openings at no call either. -/
theorem tapD_btap (H : Msg → Msg) (decode : List (ℕ × Unit) → List Unit) (x : Msg) :
    (Btap H (tapD Coin).V (Claims.Sampling.E (tapD Coin) decode) (tapD Coin).commit (tapD Coin).parse
      (fun _ => tapOps) Unit silentP x).Pr H = 0 := by
  unfold Finder.Pr
  rw [Pr_congr fun s => iff_false_intro (by
    simp only [Btap, ownLog_noHash _ _ _ (noHash_E (tapD Coin) (fun _ _ _ => rfl) _ _ _ _)]
    simp [tapLog, tapD, leakD, accCm, OC.eval, tapOps, findColl])]
  exact Pr_false

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- At `δ = 1/8` the accepting verifier's knowledge error is below `1`: at `δ = 0` it is below `1/3` (`sksD_κ`), and
`δ` adds `c²·δ ≤ 4δ`. -/
theorem tapD_κ_eighth (x : Msg) : Claims.Sampling.κ (tapD Coin) (1 / 8) x < 1 := by
  have h0 : Claims.Sampling.κ (tapD Coin) 0 x < 1 / 3 := sksD_κ (Coin := Coin) x
  have hs := Claims.Witness.Sampling.slack_le_two
  have hp := Claims.Witness.Sampling.slack_pos
  have hsq : Claims.Sampling.slack ^ 2 ≤ 4 := by nlinarith
  have hκ : Claims.Sampling.κ (tapD Coin) (1 / 8) x =
      Claims.Sampling.κ (tapD Coin) 0 x + Claims.Sampling.slack ^ 2 * (1 / 8) := by
    unfold Claims.Sampling.κ
    ring
  rw [hκ]
  linarith

end Toy

/-- **`PeriodLeakAccepted`'s per-window bound fails at the accepting verifier** `tapD`, at every hash, at one step at
`δ = 1/8` with no input registered and the statement `[]` in scope (`leakD_scopeAcc`): no witness is in `RLeakTap`
(`tapD_noRLeakTap`), so the silent prover's run fails it with probability `1`, while `κ < 1` and the finders find
nothing. -/
theorem tapD_bound_fails (H : Msg → Msg) :
    ¬ PrBad H (tapD Unit).V
        (RLeakTap H (tapD Unit).commit (tapD Unit).parse (tapD Unit).partition 1 1 1 (fun _ => notLay) (fun _ => 1)
          (1 / 8) (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => tapOps))
        (Claims.Sampling.E (tapD Unit) fun _ => [()]) [] silentP ≤
      Claims.Sampling.κ (tapD Unit) (1 / 8) [] +
          Real.sqrt (Claims.Sampling.γ (tapD Unit) [] * PrB H (tapD Unit).V silentP []) +
        (Bext H (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) Unit silentP []).Pr H +
        (Btap H (tapD Unit).V (Claims.Sampling.E (tapD Unit) fun _ => [()]) (tapD Unit).commit (tapD Unit).parse
          (fun _ => tapOps) Unit silentP []).Pr H := by
  intro h
  have hp : PrBad H (tapD Unit).V
      (RLeakTap H (tapD Unit).commit (tapD Unit).parse (tapD Unit).partition 1 1 1 (fun _ => notLay) (fun _ => 1)
        (1 / 8) (fun s => AsReg s.prog notLay 1 ∅) 1 (fun _ => tapOps))
      (Claims.Sampling.E (tapD Unit) fun _ => [()]) [] silentP = 1 :=
    prBad_accV H _ _ [] silentP (tapD_noRLeakTap H _ _ [])
  have hb : PrB H (tapD Unit).V silentP [] = 0 := prB_accV H silentP []
  simp only [hp, hb, mul_zero, Real.sqrt_zero, add_zero, tapD_bext, tapD_btap] at h
  linarith [tapD_κ_eighth (Coin := Unit) []]

/-! ## The horizon -/

/-- **`HorizonLeakAccepted`'s bound fails at an audit that always accepts**, at `δ = 1/2` and `e = e' = 0`: one window
of one step whose faulty set holds its one unit, which `Fmax = 1` allows none of, so the good event fails on every
coin. -/
theorem horizon_bound_fails :
    ¬ Pr (fun _ : Fin 1 → Unit => (∀ _ : Fin 1, True) ∧
        ¬ HorizonGood (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ => ([] : List Bool)) 1 (fun _ => {0}) (fun _ => [obs0])) ≤
      (1 / 2 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) + ∑ _ : Fin 1, (0 : ℝ) := by
  simp [Pr, HorizonGood] <;> norm_num

/-! ## The Python network certifier's fixture -/

/-- A frame of the fixture's three connections, as a number below `9`: its connection's ordinal times three, plus its
kind. -/
def fxCode : Fin 3 × CellView Unit → ℕ
  | (c, .data _) => 3 * c.val
  | (c, .pad) => 3 * c.val + 1
  | (c, .fail) => 3 * c.val + 2

/-- The frame a number below `9` is. -/
def fxFrame (n : ℕ) : Option (Fin 3 × CellView Unit) :=
  if h : n < 9 then
    some (⟨n / 3, by omega⟩, if n % 3 = 0 then .data () else if n % 3 = 1 then .pad else .fail)
  else none

/-- **A rows format of the fixture `fxQueue`**: its three connections in order, a frame encoded as that many `true`
bits, one egress link and one status, ten buckets a window. -/
def fxRows : RowsFormat (Fin 3) Unit 1 1 where
  conns := List.finRange 3
  enc x := List.replicate (fxCode x) true
  dec l := fxFrame l.length
  link := 0
  T := 10
  jitter _ _ _ := 0
  status _ _ _ _ := .complete

theorem fxRows_wf : fxRows.Wf := by
  refine ⟨List.nodup_finRange 3, List.mem_finRange, ?_⟩
  rintro ⟨c, ⟨⟨⟩⟩ | _ | _⟩ <;> fin_cases c <;> rfl

theorem fxQueue_bucket (c : Fin 3) (j : ℕ) (hj : j < fxQueueSched.cells c) : fxQueueSched.bucket c j < 10 := by
  fin_cases c <;> simp [fxQueueSched, WardenSchedule.bucket] at hj ⊢ <;> omega

/-- **`fxQueue`'s schedule fits the format** over one window of four slots: its cells fall in buckets `4 … 9`, and no
bucket holds more than four frames. -/
theorem fxQueue_fits : (RowsFormat.tapRecord fxRows fxRows_wf 4 1).fits fxQueueSched := by
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
