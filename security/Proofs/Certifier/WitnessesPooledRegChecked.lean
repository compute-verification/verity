module

public import Definitions.Certifier.Guarantees.PooledRegChecked
public import Proofs.Certifier.WitnessesPooledReg
public import Proofs.Certifier.PooledRegChecked
import Proofs.Flock.SamplingGeneric.Efficient
import Proofs.Flock.Claims.Recursion.BitsLin2
import Proofs.Flock.Claims.Sampling.Runs
import Proofs.Flock.Claims.Witnesses.SamplingGenericCost

/-!
# The witnesses of the checked pooled bound at the trace with the registered openings

`PooledLeakRegisteredChecked`'s honest lemma (`PooledLeakRegisteredChecked_honest`) is its premises at one instance
that accepts: `PooledLeakRegistered_honest`'s toy instance (one execution phase audited by `twoL` against the prover
that sends nothing, the program of two inputs sampled whole, the layout `twoLay` and the registered openings
`certOps`), with no input registered and the free input, the gate `1`, checked. The check (`checkedValid₂`) pins the run at
the gate `1` to the value the run at the trace with the registered openings holds there (`regRunR`), so it pins the
checked input (`checkedUnique_pin`) and holds on that run, and its miss is `0` (`PhaseChecks` at `εchk = 0`). The
network attester has no connection; its record is the one step's `twoObs` (`checkedFormat₂`). Its last conjunct is the
guarantee's event there, which is not empty: at every history the phase is accepted, registered as the period's and
its record complete.

Its nontrivial lemma (`PooledLeakRegisteredChecked_nontrivial`): at that instance the bound's right side is `0`, below
`1`, the phase's own error being `0` (`twoL` makes no SHA-512 call, `twoL_εregX` and `twoL_bopsX`), and the checked
count is strictly below `PooledLeakRegistered`'s, the checked input being a free input outside the registered ones
(`pooledCount_checked_lt`).
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Witness.SamplingGeneric (twoA twoL twoStmt twoPart twoProg twoSessions_meets twoL_parse twoL_partition)

local notation "shaD" => FlockDeploy.sha512

local notation "twoLay" => (BoundaryGates.mk (fun _ ℓ _ => Fin.elim0 ℓ) (fun _ => [0]) : BoundaryGates 0 1)
local notation "twoObs" => (Obs.mk (fun ℓ => Fin.elim0 ℓ) 0 (fun _ => Status.complete) : Obs 0 1 1)
local notation "Ω₂" => (Fin 1 → Unit × (Fin 2 → Bool))
local notation "D₂" => (fun _ => twoL Bool : Msg → Compiled Bool (twoA Bool))
local notation "x₂" => (fun _ _ _ => [] : Unit → Ω₂ → Fin 1 → Msg)
local notation "P₂" => (fun _ _ _ => silentP : Unit → Ω₂ → Fin 1 → Prover Unit)
local notation "ops₂" => (fun _ _ _ _ => certOps : Unit → Ω₂ → Fin 1 → AuditStatement → List (Opening Bool))
local notation "obs₂" => (fun _ _ _ => [twoObs] : Unit → Ω₂ → Fin 1 → List (Obs 0 1 1))
local notation "e₂" =>
  (fun _ => Compiled.εregX shaD (twoL Bool) [] silentP +
    Finder.Pr shaD (BopsX shaD (twoL Bool) (fun _ => certOps) Unit silentP []) : Fin 1 → ℝ)

/-! The check: the run holds the gate `1` at the value the run at the trace with the registered openings holds there. -/

/-- The run at the trace with the registered openings at the honest instance's statement. -/
noncomputable abbrev checkedRun₂ : ℕ → Bool :=
  regRunR shaD (twoL Bool) silentP [] (fun _ => certOps) (fun _ => twoLay) (fun _ => 1) twoStmt twoPart

/-- The honest instance's check: the run holds the gate `1` at `checkedRun₂`'s value there. -/
noncomputable abbrev checkedValid₂ : AuditStatement → (ℕ → Bool) → Prop :=
  fun _ ρ => ∀ g ∈ ({1} : Finset ℕ), ρ g = checkedRun₂ g

/-! The network attester: no connection, no server behavior to speak of, one device. -/

/-- The record format of a network attester with no connection over one execution phase: it holds every schedule, its
record is the one step's record `twoObs`, and no cell reads back from it, as the code sends none. -/
def checkedFormat₂ : RecordFormat Empty Unit (Fin 1 → List (Obs 0 1 1)) :=
  ⟨fun _ _ _ _ => [twoObs], fun _ _ c => c.elim, fun _ => True, fun _ _ _ => funext fun c => c.elim⟩

/-- No connection, so no arrivals. -/
abbrev checkedArr₂ : Unit → Empty → Arrivals Unit := fun _ c => c.elim

/-- The network attester's device runs its recording code at `checkedFormat₂`. -/
abbrev checkedTrace₂ :
    Unit → CertifierSchedule Empty × Unit → (Empty → List (ℕ × CellView Unit)) × (Fin 1 → List (Obs 0 1 1)) :=
  fun _ p => recordingCode checkedFormat₂ p.1 (checkedArr₂ p.2)

/-- The read-back under the one parameter: no connection, so no cell. -/
abbrev checkedRead₂ : Unit → (Fin 1 → List (Obs 0 1 1)) → Empty → List (ℕ × CellView Unit) := fun _ _ c => c.elim

/-- The one schedule, of no connection. -/
abbrev checkedSched₂ : Unit → Ω₂ → CertifierSchedule Empty := fun _ _ => ⟨0, 0, Empty.elim, Empty.elim⟩

/-- The gate `1` of the program of two inputs is a free input at `twoLay` over one step: an input, and no step's
ingress, which is the gate `0`. -/
theorem one_mem_freeInputs : 1 ∈ freeInputs twoStmt.prog twoLay 1 := by
  decide

/-- At the honest instance the phase is registered as the period's with no input registered: its statement parses to
`twoStmt` with the partition `twoPart`. -/
theorem twoPhase_registered_none (ω : Ω₂) (i : Fin 1) :
    PhaseRegistered D₂ (fun _ => twoLay) (fun _ => 1) (fun _ => ∅) (fun _ => twoStmt) (fun _ => twoPart)
      (fun _ _ => true) x₂ ops₂ () ω i :=
  ⟨twoStmt, rfl, ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, fun g hg => absurd hg (Finset.notMem_empty g)⟩

/-- **`PooledLeakRegisteredChecked`'s premises at an instance that accepts, with a checked input.** Toy code: one
execution phase audited by `twoL` against the prover that sends nothing, the program of two inputs sampled whole (rate
`1`), the gate `0` the ingress, no input registered, the gate `1` checked, the registered openings `certOps`, the record
`twoObs`, the phase's error `e₂` (`twoL`'s committed form's, with the collision finder against `certOps`). The check
pins the run's gate `1` to the value of the run at the trace with the registered openings there: it pins the checked
input (`CheckedUnique`, by `checkedUnique_pin`) and its miss is `0` (`PhaseChecks`), since that run passes it by
construction; it is not a check any deployed system runs. The network attester has no connection, so no cell reaches the
client, and one secret: the read-back and the channel hold trivially. At every history the phase is accepted, registered
as the period's and its record complete, which is the guarantee's event. The prover is the one that sends nothing, so
the witness shows the premises jointly satisfiable at an accepting instance, not that an honest prover of a real program
passes. -/
theorem PooledLeakRegisteredChecked_honest :
    (twoA Bool).KS shaD ∧ (twoL Bool).commit.Binds shaD ∧ (twoL Bool).Sessions shaD ∧ PhaseRounds D₂ 2 x₂ ∧
      PhasesAdaptive x₂ P₂ ops₂ ∧
      (∀ s, (twoL Bool).parse [] = some s → RegOpens shaD (twoL Bool).commit s ∅ certOps) ∧
      (∀ s Pt, (twoL Bool).parse [] = some s → (twoL Bool).partition s = some Pt → RateAt s Pt 1) ∧
      PhaseErrors shaD D₂ x₂ P₂ ops₂ e₂ ∧
      (∀ _ : Fin 1, CheckedUnique twoStmt.prog twoPart twoStmt.design twoLay 1 {1} (checkedValid₂ twoStmt)) ∧
      PhaseChecks shaD D₂ (fun _ => twoLay) (fun _ => 1) checkedValid₂ (fun _ => twoStmt) x₂ P₂ ops₂ (fun _ => 0) ∧
      (twoLay).Fits twoStmt.prog 1 ∧
      Verity.Assumptions.TrustedDevice () checkedTrace₂ (fun p => recordingCode checkedFormat₂ p.1 (checkedArr₂ p.2)) ∧
      checkedFormat₂.ReadsBy checkedRead₂ (fun _ _ => True) ∧
      (∀ ω : Ω₂, MonitoringPeriodChannelReg checkedFormat₂ (fun (_ : Unit) _ => True) (Set.univ : Set Unit)
        (Set.univ : Set Unit)
        (fun z => checkedSched₂ z ω) (fun _ => ()) (fun _ => ()) fun p => (checkedTrace₂ () p).1) ∧
      (∀ s, (twoL Bool).parse [] = some s →
        CertifierRegisters shaD (twoL Bool).commit twoLay 1 (fun _ _ => [true]) s certOps [twoObs]) ∧
      ∀ (ω : Ω₂) (i : Fin 1), phaseAcc shaD D₂ x₂ P₂ (fun _ _ _ => le_rfl) () i ω (ω i) ∧
        PhaseRegistered D₂ (fun _ => twoLay) (fun _ => 1) (fun _ => ∅) (fun _ => twoStmt) (fun _ => twoPart)
          (fun _ _ => true) x₂ ops₂ () ω i ∧
        RecordComplete (fun _ => 1) 1 (fun _ => twoStmt) obs₂ () ω i := by
  obtain ⟨hA, hb, hS, -, hacc⟩ := twoSessions_meets (Coin := Bool) shaD
  refine ⟨hA, hb, hS, fun _ _ _ => le_rfl, ⟨fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl⟩,
    fun s hs => ⟨fun o ho => ?_, fun g hg => absurd hg (Finset.notMem_empty g)⟩,
    fun s Pt hs hPt => ?_, fun _ _ _ => le_rfl,
    fun _ => checkedUnique_pin twoPart _ twoLay_fits (Finset.singleton_subset_iff.2 one_mem_freeInputs) checkedRun₂,
    fun _ _ _ => ?_, twoLay_fits, fun _ => rfl, fun _ _ _ _ _ => funext fun (c : Empty) => c.elim,
    fun _ => ⟨fun _ _ => ⟨trivial, (), Set.mem_univ _, trivial⟩, fun _ _ => rfl⟩, fun s hs => ?_,
    fun ω i => ⟨(hacc _).2, twoPhase_registered_none ω i, twoPhase_record ω i⟩⟩
  · simp only [certOps, List.mem_cons, List.not_mem_nil, or_false] at ho
    rcases ho with rfl | rfl <;> rfl
  · obtain rfl := twoL_parse hs
    obtain rfl := twoL_partition hPt
    exact rateAt_all rfl rfl
  · unfold PrCheckMissR
    refine (Claims.Pr_mono (q := fun _ => False) fun p hp => ?_).trans Claims.Pr_false.le
    obtain ⟨-, s, Pt, hs, hPt, -, hv⟩ := hp
    obtain rfl := twoL_parse hs
    obtain rfl := twoL_partition hPt
    exact hv fun _ _ => rfl
  · obtain rfl := twoL_parse hs
    exact twoLay_registers

/-- At that instance the guarantee's hypotheses are those premises and its event is every phase accepted, so its bound
reads there: at `q = 1`, `Fmax = 1` and `εchk = 0`, the phase is accepted outside its checked good event with
probability at most its own error. -/
example (h : Guarantees.PooledLeakRegisteredChecked) :
    Pr (fun ω : Ω₂ => (∀ i, phaseAcc shaD D₂ x₂ P₂ (fun _ _ _ => le_rfl) () i ω (ω i)) ∧
        ¬PooledGoodChecked (fun _ => twoLay) (fun _ => 1) (fun _ => ∅) checkedValid₂ 1 (fun _ => twoStmt)
          (fun _ => twoPart) (fun _ _ => true) (fun _ _ _ => [true]) 1 (WregR shaD D₂ x₂ P₂ ops₂ () ω) (obs₂ () ω)) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ i, e₂ i + ∑ _ : Fin 1, (0 : ℝ) := by
  obtain ⟨hA, hb, hS, -, had, hops, hrate, he, hchk, hck, hfit, htrust, hread, hch, hcert, hev⟩ :=
    PooledLeakRegisteredChecked_honest
  refine le_trans (Claims.Pr_mono fun ω hω => ?_)
    ((h shaD D₂ hA (fun _ => hb) (fun _ => hS) (fun _ => twoLay) (fun _ => 1) (fun _ => ∅) (fun _ => {1})
      checkedValid₂ 1 (fun _ => twoStmt) (fun _ => twoPart) (fun _ _ => true) hchk (fun _ _ _ _ => [true]) x₂ P₂
      (fun _ _ _ => le_rfl) ops₂ had (fun _ _ _ s hs => hops s hs) 1 zero_le_one le_rfl
      (fun _ _ _ s Pt hs hPt => hrate s Pt hs hPt) e₂ he (fun _ => 0) hck (fun _ => hfit) checkedArr₂ checkedFormat₂ ()
      checkedTrace₂ htrust checkedRead₂ (fun _ _ => True) hread Set.univ Set.univ checkedSched₂ (fun _ _ => ()) (fun _ _ => ()) hch
      (fun _ _ _ s hs _ => hcert s hs) 1).1 () (Set.mem_univ _))
  exact ⟨fun i => ⟨hω.1 i, (hev ω i).2⟩, hω.2⟩

/-! ## The instance's error is `0`: `twoL` makes no SHA-512 call -/

/-- A session's witness makes no SHA-512 call of its own, at any compiled verifier: the proof system's extractor's waits
are reruns (`noHash_ext`), served through the session's view. -/
theorem noHash_sessWit {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}
    (L : Compiled Coin A) (x : Msg) (u : ℕ) (r : Fin (L.V.k x) → Coin) (v : Seen) (c : A.EC (L.sh x u)) :
    Claims.Sampling.NoHash (L.sessWit x u r v c) := by
  unfold Compiled.sessWit
  split
  · exact .ret _
  · split
    · exact Claims.Recursion.noHash_serve (fun _ => Claims.Recursion.noHash_viewing _ _)
        (Claims.SamplingGeneric.noHash_ext _ _ _ _)
    · exact .ret _

/-- `twoL`'s extractions make no SHA-512 call of their own: each session's witness makes none (`noHash_sessWit`), and
`trueCm` checks an opening with no call. -/
theorem twoL_ownLog_runExt (H : Msg → Msg) (x : Msg) (s : AuditStatement) (r : Fin ((twoL Bool).V.k x) → Bool)
    (v : Seen) (c : (twoL Bool).SlotEC x) (run' : (Fin ((twoL Bool).V.k x) → Bool) → Transcript) :
    ownLog H run' ((twoL Bool).runExt x s r v c) = [] := by
  refine List.eq_nil_of_length_eq_zero ?_
  unfold Compiled.runExt
  rw [Claims.Sampling.ownLog_flatMapRP]
  refine List.sum_eq_zero fun n hn => ?_
  obtain ⟨iu, -, rfl⟩ := List.mem_map.1 hn
  split
  · unfold Compiled.sessExt
    rw [Claims.Sampling.ownLog_bind, List.length_append,
      Claims.Sampling.ownLog_of_noHash H run' (noHash_sessWit _ _ _ _ _ _), List.length_nil, zero_add]
    split
    · rfl
    · rw [Claims.Sampling.ownLog_flatMapRP]
      refine List.sum_eq_zero fun n' hn' => ?_
      obtain ⟨u', -, rfl⟩ := List.mem_map.1 hn'
      rw [Claims.Sampling.ownLog_checkRevealed_length]
      refine List.sum_eq_zero fun m hm => ?_
      obtain ⟨p, -, rfl⟩ := List.mem_map.1 hm
      rfl
  · rfl

/-- At `twoL` the collision finder between two runs' extractions finds nothing. -/
theorem twoL_bconfX (H : Msg → Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (x : Msg) :
    (Compiled.BconfX H (twoL Bool) Ω P x).Pr H = 0 := by
  simp only [Compiled.BconfX]
  rcases (twoL Bool).parse x with _ | s
  · exact Claims.Witness.SamplingGeneric.finder_pr_none H _ fun _ => rfl
  · exact Claims.Witness.SamplingGeneric.finder_pr_none H _ fun q => by simp [twoL_ownLog_runExt, findColl]

/-- At `twoL` the collision finder against the registered openings `certOps` finds nothing: the extractions make no
call, and `trueCm` checks the openings with none. -/
theorem twoL_bopsX (H : Msg → Msg) (x : Msg) : (BopsX H (twoL Bool) (fun _ => certOps) Unit silentP x).Pr H = 0 := by
  simp only [BopsX]
  rcases (twoL Bool).parse x with _ | s
  · exact Claims.Witness.SamplingGeneric.finder_pr_none H _ fun _ => rfl
  · refine Claims.Witness.SamplingGeneric.finder_pr_none H _ fun q => ?_
    have hc : certifierLog H (twoL Bool).commit s.roots certOps = [] := rfl
    simp [twoL_ownLog_runExt, hc, findColl]

/-- `twoL`'s committed form's error at the trace read at the statement is `0` at the prover that sends nothing: its
proof blocks' bounds `κb` and `γb` are `0`, and `B_conf,X` finds nothing (`twoL_bconfX`). -/
theorem twoL_εregX (H : Msg → Msg) (x : Msg) : (twoL Bool).εregX H x silentP = 0 := by
  have hκ : (twoL Bool).κs x = 0 := Claims.Witness.SamplingGeneric.maxBelow_zero _
  have hγ : (twoL Bool).γs x = 0 := by
    unfold Compiled.γs
    rw [show (fun u => (twoL Bool).γb x u / (twoA Bool).rounds ((twoL Bool).sh x u)) = fun _ => (0 : ℝ) from
      funext fun _ => zero_div _]
    exact Claims.Witness.SamplingGeneric.maxBelow_zero _
  unfold Compiled.εregX
  rw [hκ, hγ, twoL_bconfX]
  simp

/-- **`PooledLeakRegisteredChecked`'s bound is not vacuous at the honest instance**: there, at `q = 1`, `Fmax = 1` and
`εchk = 0`, its right side is `0`, below `1`, the phase's own error being `0` (`twoL_εregX`, `twoL_bopsX`); and the factor
its count takes for the phase's inputs, with no input registered and the gate `1` checked, is strictly below the one
`PooledLeakRegistered`'s count takes at the same registration, the gate `1` being a free input outside the registered
ones (`pooledCount_checked_lt`). Toy code, one phase: it compares the two bounds at one instance and says nothing of a
deployed check or of how many secrets a deployed record singles out. -/
theorem PooledLeakRegisteredChecked_nontrivial :
    (1 - 1 : ℝ) ^ 1 + ∑ i, e₂ i + ∑ _ : Fin 1, (0 : ℝ) < 1 ∧
    ∏ i : Fin 1, (2 ^ (freeInputs ((fun _ => twoStmt : Fin 1 → AuditStatement) i).prog twoLay 1 \ (∅ ∪ {1})).card *
        1 ^ 1) <
      ∏ i : Fin 1, (2 ^ (freeInputs ((fun _ => twoStmt : Fin 1 → AuditStatement) i).prog twoLay 1 \ ∅).card * 1 ^ 1) := by
  refine ⟨?_, pooledCount_checked_lt (fun _ => twoLay) (fun _ => 1) (fun _ => ∅) (fun _ => {1}) Nat.one_pos
    (fun _ => twoStmt) 0 (Finset.mem_singleton_self 1) one_mem_freeInputs (Finset.notMem_empty 1)⟩
  simp only [twoL_εregX, twoL_bopsX]
  norm_num

/-! ## A check missed by an invalid run at a phase that accepts on half its tapes -/

/-- A prover on a tape of one bit: both messages empty on `true`, so `twoL` accepts, and `[0]` on `false`, so it
rejects. -/
def coinP₂ : Prover Bool := ⟨fun _ _ b => .ret (if b then [] else [0])⟩

/-- `twoL` accepts `coinP₂` exactly on the tape `true`, at any coins. -/
theorem coinP_acc (H : Msg → Msg) (x : Msg) (b : Bool) (r : Fin ((twoL Bool).V.k x) → Bool) :
    (run H coinP₂ (twoL Bool).V x b r).out = .acc ↔ b = true := by
  have h : (run H coinP₂ (twoL Bool).V x b r).out =
      Claims.Witness.SamplingGeneric.twoOut (Claims.Witness.SamplingGeneric.twoMsg H coinP₂ x b 0)
        (Claims.Witness.SamplingGeneric.twoMsg H coinP₂ x b 1) :=
    congrArg Seen.out (Claims.Witness.SamplingGeneric.twoL_run H coinP₂ x b r)
  rw [h]
  cases b <;> simp [Claims.Witness.SamplingGeneric.twoOut, Claims.Witness.SamplingGeneric.twoMsg,
    Claims.Witness.SamplingGeneric.twoHist, coinP₂, OC.eval]

/-- A tape bit is `true` with probability `1/2`, beside any finite coins. -/
theorem pr_fst_true {β : Type} [Fintype β] [Nonempty β] : Pr (fun p : Bool × β => p.1 = true) = 1 / 2 := by
  classical
  have hβ : (Fintype.card β : ℝ) ≠ 0 := Nat.cast_ne_zero.2 Fintype.card_ne_zero
  unfold Pr Claims.prUniform
  rw [Finset.card_filter, Fintype.sum_prod_type, Fintype.sum_bool, Fintype.card_prod, Fintype.card_bool]
  simp only [if_true, Bool.false_eq_true, if_false, Finset.sum_const, Finset.card_univ, smul_eq_mul, mul_one,
    Finset.sum_const_zero, add_zero]
  push_cast
  field_simp
  ring

/-- At the registered openings `certOps`, which open both gates of `twoStmt`, `twoL`'s incorrect set at the trace
with the registered openings is empty, at every prover. -/
theorem twoL_FregR_certOps (H : Msg → Msg) {Ω : Type} [Fintype Ω] (P : Prover Ω) (x : Msg) :
    FregR H (twoL Bool) P x (fun _ => certOps) = ∅ := by
  have hp : (twoL Bool).parse x = some twoStmt := rfl
  have hb : ((twoL Bool).part twoStmt).blocks = [[0], [1]] := rfl
  simp only [FregR, hp, hb]
  refine Finset.filter_eq_empty_iff.2 fun v hv => ?_
  rw [not_not]
  have hv2 : v < 2 := by simpa using hv
  refine ⟨[v], ?_, regValR H (twoL Bool) P x certOps, fun _ => true, fun _ _ => rfl, ?_⟩
  · interval_cases v <;> rfl
  · interval_cases v <;>
      simp [Prog.Correct, Prog.refIn, Prog.Boundary, Prog.runIn, Prog.read, BGate.val, BGate.reads, regValR,
        certOps, certifierVal, twoStmt, twoProg]

/-- The run at the trace with the registered openings at `coinP₂`. -/
noncomputable abbrev coinRun₂ : ℕ → Bool :=
  regRunR shaD (twoL Bool) coinP₂ [] (fun _ => certOps) (fun _ => twoLay) (fun _ => 1) twoStmt twoPart

/-- The check that pins the gate `1` to the negation of `coinRun₂`'s value there: `coinRun₂` fails it. -/
noncomputable abbrev coinValid₂ : AuditStatement → (ℕ → Bool) → Prop :=
  fun _ ρ => ∀ g ∈ ({1} : Finset ℕ), ρ g = !coinRun₂ g

/-- **`PhaseChecks` holds below `1` at a run that fails the check.** Toy code: one execution phase audited by `twoL`,
the program of two inputs sampled whole, the registered openings `certOps`, the prover `coinP₂`, whose tape is one bit
and which `twoL` accepts on the tape `true` only. The check pins the gate `1` to the negation of the run at the trace
with the registered openings (`coinValid₂`): a check that pins a checked input to a constant (`CheckedUnique`, by
`checkedUnique_pin`), which that run fails. That run reads no tape, so it fails the check on every tape. The incorrect
set `FregR` is empty, so the miss is not charged to the phase's error: it is the acceptance probability, `1/2`, and
`PhaseChecks` holds at `εchk = 1/2` and at no smaller value. So the premise is more than "the run passes the check": it
bounds how often a phase accepts while its run fails. Toy code, one phase, no secret: the check is not one a deployed
system runs, and the prover is not an honest prover of a real program. -/
theorem PhaseChecks_invalid :
    (∀ _ : Fin 1, CheckedUnique twoStmt.prog twoPart twoStmt.design twoLay 1 {1} (coinValid₂ twoStmt)) ∧
      ¬coinValid₂ twoStmt coinRun₂ ∧
      PrCheckMissR shaD (twoL Bool) (fun _ => twoLay) (fun _ => 1) (fun _ => coinValid₂ twoStmt) [] coinP₂
        (fun _ => certOps) = 1 / 2 ∧
      PhaseChecks (kV := 2) (Sec := Unit) shaD (fun _ => twoL Bool) (fun _ => twoLay) (fun _ => 1) coinValid₂
        (fun _ : Fin 1 => twoStmt) (fun _ _ _ => []) (fun _ _ _ => coinP₂) (fun _ _ _ _ => certOps) fun _ => 1 / 2 := by
  have hnot : ¬coinValid₂ twoStmt coinRun₂ := fun h => by simpa using h 1 (Finset.mem_singleton_self 1)
  have hmiss : PrCheckMissR shaD (twoL Bool) (fun _ => twoLay) (fun _ => 1) (fun _ => coinValid₂ twoStmt) []
      coinP₂ (fun _ => certOps) = 1 / 2 := by
    unfold PrCheckMissR
    rw [← pr_fst_true (β := Fin ((twoL Bool).V.k []) → Bool)]
    congr 1
    funext p
    refine propext ⟨fun h => (coinP_acc shaD [] p.1 p.2).1 h.1, fun h => ⟨(coinP_acc shaD [] p.1 p.2).2 h,
      twoStmt, twoPart, rfl, rfl, ?_, hnot⟩⟩
    rw [twoL_FregR_certOps]
    exact Finset.disjoint_empty_left _
  exact ⟨fun _ => checkedUnique_pin twoPart _ twoLay_fits (Finset.singleton_subset_iff.2 one_mem_freeInputs)
    fun g => !coinRun₂ g, hnot, hmiss, fun _ _ _ => hmiss.le⟩

end NetTiming.Witness
