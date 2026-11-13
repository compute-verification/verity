module

public import Definitions.NetworkCertifier.Guarantees.MonitoringPeriodChecked
public import Proofs.NetworkCertifier.Witnesses
public import Proofs.Flock.Claims.CompleteFails
public import Proofs.Flock.Claims.Witnesses.SamplingGenericPlaced
public import Proofs.Flock.SamplingGeneric.Fails
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
import Mathlib.Tactic.Polynomial.Basic
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Prob

/-!
# The read-back is needed: `MonitoringPeriodLeakChecked`'s count fails without `RecordFormat.ReadsBy`

`MonitoringPeriodLeakChecked` counts the secrets by the registered parameters' number times the windows' count, and takes the
read-back of the record under a registered parameter (`RecordFormat.ReadsBy`). This file builds a monitoring period of
one placed window at which every other premise holds, no reader reads the record back, and two secrets are counted
against a count of 1 (`MonitoringPeriodLeakChecked_readsBy_exclusion`).

**The code** (`Proofs.Flock.Claims.Witnesses.SamplingGenericPlaced`). `posL` is a compiled verifier that rejects every
run, so `Compiled.KS` holds at it, over the statement whose program negates its input (`notProg`), under a commitment
`posCm` whose roots hold the input's bit: an opening of the input passes at that bit and one of the negation at its
complement, so it binds with no call. The network certifier's openings (`posTops`) register one step whose ingress is
the bit and whose egress frame is there when the bit is unset. The window meets `ExecutionPhaseGoodChecked` (`pos_good`) at the
check that always holds, with no checked input (`pos_checkedUnique`): the leak relation holds at every witness, since
the circuit computes the frame the openings hold.

**The network certifier** (`Definitions.NetworkCertifier.Witness`) is one connection of one cell with no data
(`Cell := Empty`): its end is late for its bucket or not (`bitArr`), so the client receives the fail cell or padding,
and the record format `bitFmt` writes that bit as the window's frame and reads it back under the whole schedule. Its
host runs `recordingCode bitFmt`.

**The exclusion.** The two secrets are the connection's admission bucket (`admSched`), which `bitFmt`'s record does
not hold, at a server that is never late: the client reads the secret off the cell's bucket (`adm_channel`), both runs
write the same record (`adm_obs`), so no reader takes the record to both views under a parameter that allows both
schedules (`adm_not_readsBy`), and both secrets are counted against `#Θ · windowsCount = 1` (`pos_count`).

**The premises together.** At the same window, under the one schedule `oneSched` and a server that is not late, the
record reads back under the registered parameter (`RecordFormat.readsBy_comp`), the network certifier registers its
observations at the operator whose move is the unset bit, the window meets `ExecutionPhaseGoodChecked`, and
`MonitoringPeriodLeakChecked`'s premises hold together (`MonitoringPeriodLeakChecked_placed`). It is at one secret: the toy circuit
has no free input, so its count is 1, and an operator fixed before the secrets registers one ingress. `posL` rejects
every run, so the lemma shows that the premises and the good event are consistent at a placed window. It is no honest
lemma of the statement: no honest run is accepted there. The toy has no checked input, so neither lemma shows
anything about a checked input.
-/

@[expose] public section

namespace NetTiming

/-- Where the registered parameter gives the schedule (`s = g θ`), the record reads back under it: the format's own
read-back at that schedule. -/
theorem RecordFormat.readsBy_comp {κ Cell Ob P : Type} (f : RecordFormat κ Cell Ob) (g : P → CertifierSchedule κ) :
    f.ReadsBy (fun θ => f.read (g θ)) fun θ s => s = g θ := by
  rintro θ s rfl h a
  exact f.read_record _ h a

end NetTiming

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness Claims.Witness.Sampling
open Claims.SamplingGeneric (Arg Compiled maxBelow_nonneg)
open Claims.CompleteFails (run_rejV)
open Claims.Witness.SamplingGeneric (toyA bext_pr notProg notStmt notPart rootBit posOpens posCm posStmt posX posTops
  posL)
open scoped NNRat

set_option linter.unusedVariables false

/-! ## The code -/

theorem posCm_binds (H : Msg → Msg) : posCm.Binds H := by
  intro roots o o' hpos hne ho ho'
  have h1 : posOpens roots o = true := ho
  have h2 : posOpens roots o' = true := ho'
  refine absurd ?_ hne
  unfold posOpens at h1 h2
  rw [← hpos] at h2
  clear ho ho' hne hpos
  revert h1 h2
  generalize o.val = a
  generalize o'.val = a'
  generalize rootBit roots = r
  generalize o.pos = p
  intro h1 h2
  cases a <;> cases a' <;> cases r <;> simp_all

theorem certVal0 (b : Bool) : certifierVal (posTops (posStmt b)) 0 = b := by cases b <;> rfl

theorem certVal1 (b : Bool) : certifierVal (posTops (posStmt b)) 1 = !b := by cases b <;> rfl

theorem pos_parse {Coin : Type} (z : Bool) : (posL Coin).parse (posX z) = some (posStmt z) := by cases z <;> rfl

/-- With nothing held, the circuit computes the negation of its input. -/
theorem pos_notVal (inp : ℕ → Bool) (b : Bool) (h0 : inp 0 = b) : valOver notProg inp (fun _ => none) 1 = !b := by
  cases b
  · exact notProg_val inp h0
  · exact notProg_val_true inp h0

theorem pos_scopeAcc {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (x : Msg) :
    ScopeAcc (posL Coin) (fun _ => notLay) (fun _ => 1) x :=
  ⟨⟨posStmt (!x.isEmpty), notPart, rfl, rfl, inScope_not⟩, posStmt (!x.isEmpty), rfl, notLay_fits⟩

section KS

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **`Compiled.KS` at `posL`**: no run is accepted, and its errors are not negative. -/
theorem posL_ks (H : Msg → Msg) : (posL Coin).KS H := by
  intro δ hδ x _ Ω _ _ P
  have hs : 0 < slack := by unfold slack; positivity
  have hκs : 0 ≤ (posL Coin).κs x := maxBelow_nonneg _ _
  have hκ : 0 ≤ (posL Coin).κ δ x := by
    unfold Compiled.κ
    exact add_nonneg (add_nonneg (mul_nonneg (pow_nonneg hs.le 2) (add_nonneg hδ hκs)) (by positivity))
      (mul_nonneg hs.le (by positivity))
  have hb : (Claims.Sampling.Bext H (posL Coin).V (posL Coin).E Ω P x).Pr H = 0 :=
    bext_pr H (posL Coin) (fun _ _ _ => rfl) (fun _ => Nat.one_pos) Ω P x
  have hfp : (0 : ℝ) ≤ 1 * (Claims.Sampling.Bext H (posL Coin).V (posL Coin).E Ω P x).Pr H := by
    rw [hb]
    simp
  refine le_trans (le_of_eq ?_) (add_nonneg (add_nonneg hκ (Real.sqrt_nonneg _)) hfp)
  unfold PrBad
  refine (Pr_congr (q := fun _ => False) fun s => ⟨fun h => ?_, False.elim⟩).trans Pr_false
  exact absurd ((run_rejV H P x s.1 s.2.1).symm.trans h.1) (by simp)

end KS

/-! ## The window -/

theorem pos_opens (H : Msg → Msg) (s : AuditStatement) : ∀ q ∈ posTops s, posCm.Opens H s.roots q = true := by
  intro q hq
  simp only [posTops, List.mem_cons, List.not_mem_nil, or_false] at hq
  rcases hq with rfl | rfl
  · show posOpens s.roots ⟨0, rootBit s.roots, []⟩ = true
    simp [posOpens]
  · show posOpens s.roots ⟨1, !rootBit s.roots, []⟩ = true
    simp [posOpens]

/-- **The network certifier registers the step** `posObs b` at `posStmt b`, at an operator whose first move is the
bit. -/
theorem pos_registers (H : Msg → Msg) (b : Bool) (op : ℕ → List (Obs 1 1 1) → List Bool) (hop : op 0 [] = [b]) :
    CertifierRegisters H posCm notLay 1 op (posStmt b) (posTops (posStmt b)) [posObs b] := by
  refine ⟨pos_opens H (posStmt b), fun t ht => ?_⟩
  obtain rfl : t = 0 := by omega
  refine ⟨fun ℓ σ g hg => ?_, fun g hg => ?_, Nat.one_pos, fun ℓ => ?_, ?_⟩
  · simp [notLay, slotGates] at hg
    subst hg
    exact ⟨fun c hc => by simp [posStmt, notProg] at hc, ⟨1, !rootBit (posStmt b).roots, []⟩, by simp [posTops], rfl⟩
  · simp [notLay] at hg
    subst hg
    exact ⟨⟨0, rootBit (posStmt b).roots, []⟩, by simp [posTops], rfl⟩
  · funext σ
    have h : rowOf (certifierVal (posTops (posStmt b))) (notLay.egress 0 ℓ) σ =
        if certifierVal (posTops (posStmt b)) 1 then some [] else none := rfl
    rw [h, certVal1]
    cases b <;> rfl
  · have h : (notLay.ingress 0).map (certifierVal (posTops (posStmt b))) =
        [certifierVal (posTops (posStmt b)) 0] := rfl
    rw [h, certVal0]
    exact hop

theorem pos_empty_mem : (∅ : Finset ℕ) ∈ faultySets (⟨[.all]⟩ : Design) notPart 0 := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset]
  refine ⟨Finset.empty_subset _, ?_, Finset.disjoint_empty_left _⟩
  show (0 : ℝ) < Design.Pa ⟨[.all]⟩ notPart ∅
  simp [Design.Pa, notPart, classesPa, OneStage.Pa, window]

theorem pos_faulty_empty {F : Finset ℕ} (hF : F ∈ faultySets (⟨[.all]⟩ : Design) notPart 0) : F = ∅ := by
  classical
  simp only [faultySets, Finset.mem_filter, Finset.mem_powerset] at hF
  obtain ⟨hsub, -, hdis⟩ := hF
  refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
  have hv1 : v < 1 := Finset.mem_range.1 (hsub hv)
  obtain rfl : v = 0 := by omega
  exact Finset.disjoint_left.1 hdis hv (by decide)

theorem pos_faultySets : faultySets (⟨[.all]⟩ : Design) notPart 0 = {∅} :=
  Finset.eq_singleton_iff_unique_mem.2 ⟨pos_empty_mem, fun _ h => pos_faulty_empty h⟩

/-- **What the network certifier registers at `posStmt b` is accepted**, at every operator: the openings hold the
ingress `b` and the frame's presence `!b`, which the circuit computes from that ingress, with no incorrect proof
block. -/
theorem pos_acc (H : Msg → Msg) (b : Bool) (ρ : ℕ → Bool) (op : ℕ → List (Obs 1 1 1) → List Bool)
    (o : List (Obs 1 1 1))
    (hreg : CertifierRegisters H posCm notLay 1 op (posStmt b) (posTops (posStmt b)) o) (hcov : Covered 1 o)
    (hlen : o.length = 1) (hj : ∀ y ∈ o, y.jitter < 1) :
    o ∈ AccSeqsChecked notProg notPart (⟨[.all]⟩ : Design) notLay 1 (AsReg notProg notLay 1 ∅ ρ)
      (fun _ => True ∧ True) 0 op 1 := by
  obtain ⟨-, -, h0, hrows, hop⟩ := hreg.2 0 Nat.one_pos
  have hop' : op 0 [] = [b] := by
    have h : op 0 (o.take 0) = [certifierVal (posTops (posStmt b)) 0] := hop
    rw [certVal0] at h
    exact h
  have hin : ingressUpTo op o 0 = [[b]] := by
    have h : ingressUpTo op o 0 = [op 0 []] := rfl
    rw [h, hop']
  show o.length = 1 ∧ (∀ y ∈ o, y.jitter < 1) ∧ ∃ F ∈ faultySets (⟨[.all]⟩ : Design) notPart 0,
    AcceptedChecked notProg notPart notLay 1 (AsReg notProg notLay 1 ∅ ρ) (fun _ => True ∧ True) F op o
  refine ⟨hlen, hj, ∅, pos_empty_mem, fun _ => false,
    ⟨fun g hg _ => absurd hg (Finset.notMem_empty g), fun _ _ => rfl⟩, fun _ => none, fun g c hg => by simp at hg,
    ⟨trivial, trivial⟩, fun t ht => ?_⟩
  obtain rfl : t = 0 := by omega
  refine ⟨h0, ?_, fun κ => hcov 0 Nat.one_pos h0 κ⟩
  funext ℓ σ
  have hl : o[0].rows ℓ σ = if certifierVal (posTops (posStmt b)) 1 then some [] else none :=
    congrFun (hrows ℓ) σ
  have hr : (circuitProgram notProg notLay 1).rows 0 (ingressUpTo op o 0) (fun _ => false, fun _ => none) ℓ σ =
      if valOver notProg (notLay.inputOf 1 [[b]] fun _ => false) (fun _ => none) 1 then some [] else none := by
    rw [hin]
    rfl
  refine hl.trans (Eq.trans ?_ hr.symm)
  rw [certVal1]
  cases b
  · rw [pos_notVal (notLay.inputOf 1 [[false]] fun _ => false) false rfl]
  · rw [pos_notVal (notLay.inputOf 1 [[true]] fun _ => false) true rfl]

/-- **The window is good at each bit**: at the statement `posX z`, any witness and the step `posObs z`. -/
theorem pos_good {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) (z : Bool)
    (ρ : ℕ → Bool) :
    ExecutionPhaseGoodChecked H (posL Coin) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅) (fun _ _ => True ∧ True) 1
      (posStmt false) ρ posTops (posX z) ⟨[]⟩ [posObs z] := by
  refine ⟨⟨posStmt z, notPart, pos_parse z, rfl, fun op o hreg hcov hlen hj => pos_acc H z _ op o hreg hcov hlen hj⟩,
    pos_scopeAcc (posX z), posStmt z, pos_parse z, ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl,
    fun g hg => absurd hg (Finset.notMem_empty g), ?_, rfl, fun y hy => ?_⟩
  · intro t ht h κ
    have ht' : t < 1 := ht
    obtain rfl : t = 0 := by omega
    rfl
  · obtain rfl := List.mem_singleton.1 hy
    exact Nat.one_pos

theorem pos_placed {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] :
    WindowsPlaced (posL Coin).partition 0 (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) :=
  ⟨fun _ => rfl, fun _ F hF => by
    rw [pos_faulty_empty hF]
    simp [faultyBits]⟩

/-- **No checked input, so the check pins them**: over the horizon of one step, two runs whose ingress agrees at step 0
and whose inputs agree everywhere are one run. -/
theorem pos_checkedUnique :
    CheckedUnique notProg notPart (⟨[.all]⟩ : Design) notLay 1 ∅ (fun _ => True ∧ True) := by
  intro t ht ins ins' a a' free hlen hlen' hins ha _ _ _ ℓ
  obtain rfl : a = a' := funext fun g => ha g (Finset.notMem_empty g)
  obtain ⟨u, rfl⟩ := List.length_eq_one_iff.1 hlen
  obtain ⟨u', rfl⟩ := List.length_eq_one_iff.1 hlen'
  obtain rfl : u = u' := by simpa using hins 0 (Nat.zero_le t)
  rfl

theorem pos_windowsCount :
    windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1 (fun _ : Fin 1 => posStmt false)
      (fun _ => notPart) (fun _ => 0) = 1 := by
  have hfree : (freeInputs notProg notLay 1 \ (∅ ∪ ∅)).card = 0 := by decide
  unfold windowsCount
  rw [Fin.prod_univ_one]
  show 2 ^ (freeInputs notProg notLay 1 \ (∅ ∪ ∅)).card * (faultySets (⟨[.all]⟩ : Design) notPart 0).card * 2 ^ 0 *
    1 ^ 1 = 1
  rw [hfree, pos_faultySets]
  simp

theorem posDec_obs (z : Bool) : posDec (fun _ => [posObs z]) = z := by cases z <;> rfl

/-! ## One connection of one cell with no data -/

theorem two_not_le_one : ¬ (2 : ℕ∞) ≤ 1 := by simp

theorem pos_count :
    ({()} : Set Unit).encard * ((windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1
      (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) : ℕ) : ℕ∞) = 1 := by
  rw [pos_windowsCount]
  simp

/-- A lane of one cell with no data: the fail cell where its end is late for the bucket, else padding. -/
theorem lane_one (a : Arrivals Empty) (t : ℕ → ℕ) :
    certifierLane a t 0 1 = [(t 0, if t 0 < a.ready 0 then CellView.fail else CellView.pad)] := by
  have hs : a.stream = [] := by
    cases h : a.stream with
    | nil => rfl
    | cons x _ => exact x.elim
  by_cases h : t 0 < a.ready 0
  · simp [certifierLane, hs, h]
  · simp [certifierLane, streamCell, hs, h]

/-- The cells the client receives at the bit: the fail cell or padding, in bucket 0. -/
theorem bit_code (z : Bool) :
    certifierCode oneSched (bitArr z) () = [(0, if z then CellView.fail else CellView.pad)] := by
  have h : certifierCode oneSched (bitArr z) () = certifierLane (bitArr z ()) (oneSched.bucket ()) 0 1 := rfl
  rw [h, lane_one]
  cases z <;> rfl

theorem bit_dec (z : Bool) : failedAt0 (certifierCode oneSched (bitArr z) ()) = z := by
  rw [bit_code]
  cases z <;> rfl

/-- The window's record at the bit: the step `posObs z`. -/
theorem bit_obs (z : Bool) : (recordingCode bitFmt oneSched (bitArr z)).2 = fun _ => [posObs z] := by
  funext i
  show [posObs (failedAt0 (certifierCode oneSched (bitArr z) ()))] = [posObs z]
  rw [bit_dec]

/-! ## The record must read back under the registered parameter -/

/-- The cells the client receives at the admission `z` from a server that is never late: padding, in the admission's
bucket. -/
theorem adm_code (z : Bool) :
    certifierCode (admSched z) (bitArr false) () = [(if z then 1 else 0, CellView.pad)] := by
  have h : certifierCode (admSched z) (bitArr false) () =
      certifierLane (bitArr false ()) ((admSched z).bucket ()) 0 1 := rfl
  rw [h, lane_one]
  cases z <;> rfl

theorem adm_dec (z : Bool) : lateAt0 (certifierCode (admSched z) (bitArr false) ()) = z := by
  rw [adm_code]
  cases z <;> rfl

/-- The window's record is the same step at both admissions: `bitFmt` writes whether the cell failed and nothing
of its bucket. -/
theorem adm_obs (z : Bool) : (recordingCode bitFmt (admSched z) (bitArr false)).2 = fun _ => [posObs false] := by
  funext i
  show [posObs (failedAt0 (certifierCode (admSched z) (bitArr false) ()))] = [posObs false]
  rw [adm_code]
  rfl

theorem adm_channel :
    MonitoringPeriodChannelReg bitFmt (fun (_ : Unit) (_ : CertifierSchedule Unit) => True) {()} (Set.univ : Set Bool) admSched
      (fun _ => ()) (fun v => lateAt0 (v ())) fun p => (recordingCode bitFmt p.1 (bitArr false)).1 :=
  ⟨fun _ _ => ⟨rfl, (), Set.mem_singleton (), trivial⟩, fun z _ => adm_dec z⟩

/-- **No reader takes `bitFmt`'s record to the client's cells at both admissions.** Under a parameter that allows both
schedules, the two runs write the same record (`adm_obs`) and send the cell in different buckets (`adm_code`). -/
theorem adm_not_readsBy (rd : Unit → (Fin 1 → List (Obs 1 1 1)) → Unit → List (ℕ × CellView Empty)) :
    ¬ bitFmt.ReadsBy rd fun _ _ => True := fun h => by
  have key : ∀ z : Bool, rd () (fun _ => [posObs false]) () = [(if z then 1 else 0, CellView.pad)] := fun z =>
    ((congrArg (fun ob => rd () ob ()) (adm_obs z)).symm.trans
      (congrFun (h () (admSched z) trivial rfl (bitArr false)) ())).trans (adm_code z)
  have h10 := (key true).symm.trans (key false)
  simp at h10

/-! ## The exclusion -/

/-- **Without the read-back, `MonitoringPeriodLeakChecked`'s count fails.** At `posL`, one window, a server that is never late and
the format `bitFmt`, with the secret the connection's admission bucket: the coins are a nonempty type and `δ = 0` is
nonnegative; `Compiled.KS`, the commitment's binding, the window's placement, its check's pinning, `TrustedHost` at a
host that runs the network certifier's code, and the channel all hold; the network certifier registers its
observations at both secrets, at an operator that reads nothing; no reader reads the record back under the one
registered parameter; and the two secrets, each of whose windows meets its good event, are counted against
`#Θ · windowsCount = 1`. -/
theorem MonitoringPeriodLeakChecked_readsBy_exclusion :
    Nonempty Bool ∧ (0 : ℝ) ≤ 0 ∧ (posL Bool).KS id ∧ (posL Bool).commit.Binds id ∧
      WindowsPlaced (posL Bool).partition 0 (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) ∧
      CheckedUnique notProg notPart (⟨[.all]⟩ : Design) notLay 1 ∅ (fun _ => True ∧ True) ∧
      Verity.Assumptions.TrustedHost ()
        (fun (_ : Unit) (p : CertifierSchedule Unit × Unit) => recordingCode bitFmt p.1 (bitArr false))
        (fun p => recordingCode bitFmt p.1 (bitArr false)) ∧
      MonitoringPeriodChannelReg bitFmt (fun (_ : Unit) (_ : CertifierSchedule Unit) => True) {()} (Set.univ : Set Bool) admSched
        (fun _ => ()) (fun v => lateAt0 (v ())) (fun p => (recordingCode bitFmt p.1 (bitArr false)).1) ∧
      (∀ (z : Bool) (i : Fin 1) (s : AuditStatement), (posL Bool).parse (posX false) = some s →
        CertifierRegisters id (posL Bool).commit notLay 1 (fun _ _ => [false]) s (posTops s)
          ((recordingCode bitFmt (admSched z) (bitArr false)).2 i)) ∧
      (∀ rd : Unit → (Fin 1 → List (Obs 1 1 1)) → Unit → List (ℕ × CellView Empty),
        ¬ bitFmt.ReadsBy rd fun _ _ => True) ∧
      ¬ {z : Bool | z ∈ (Set.univ : Set Bool) ∧ ∀ i : Fin 1,
            ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅)
              (fun _ _ => True ∧ True) 1 (posStmt false) (fun _ => false) posTops (posX false) ⟨[]⟩
              ((recordingCode bitFmt (admSched z) (bitArr false)).2 i)}.encard ≤
          ({()} : Set Unit).encard * ((windowsCount (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅ ∪ ∅) 1
            (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) : ℕ) : ℕ∞) := by
  refine ⟨⟨true⟩, le_rfl, posL_ks id, posCm_binds id, pos_placed, pos_checkedUnique, fun _ => rfl, adm_channel,
    fun z i s hs => ?_, adm_not_readsBy, fun hc => ?_⟩
  · obtain rfl : posStmt false = s := Option.some.inj ((pos_parse (Coin := Bool) false).symm.trans hs)
    have ho : (recordingCode bitFmt (admSched z) (bitArr false)).2 i = [posObs false] := congrFun (adm_obs z) i
    exact ho ▸ pos_registers id false (fun _ _ => [false]) rfl
  · have hgood : ∀ z : Bool, ∀ i : Fin 1,
        ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅) (fun _ _ => True ∧ True) 1
          (posStmt false) (fun _ => false) posTops (posX false) ⟨[]⟩
          ((recordingCode bitFmt (admSched z) (bitArr false)).2 i) :=
      fun z i => (congrFun (adm_obs z) i) ▸ pos_good id false _
    have h2 := (Set.encard_pair (show true ≠ false by decide)).symm.le.trans
      ((Set.encard_le_encard (s := ({true, false} : Set Bool)) fun z _ =>
        Set.mem_ofPred.2 ⟨Set.mem_univ z, fun i => hgood z i⟩).trans hc)
    exact absurd (h2.trans (le_of_eq pos_count)) two_not_le_one

/-! ## The premises at a placed window -/

/-- **`MonitoringPeriodLeakChecked`'s premises hold together at a monitoring period of one placed window that meets its good
event**, at a verifier that rejects every run: they are consistent there, and no honest run passes. At `posL`, the format `bitFmt`, the one schedule `oneSched` and the one secret of a server that is not late:
`Compiled.KS`, the commitment's binding, the window's placement (`WindowsPlaced`), its check's pinning at no checked
input (`CheckedUnique`), `TrustedHost` at a host that runs the network certifier's code, the read-back under the one
registered parameter (`RecordFormat.ReadsBy`) and the channel of the secret (`MonitoringPeriodChannelReg`); the network certifier
registers the window's observations at the operator whose move is the unset bit (`CertifierRegisters`); and the window
meets its good event (`ExecutionPhaseGoodChecked`), at the check every run passes. -/
theorem MonitoringPeriodLeakChecked_placed :
    (posL Bool).KS id ∧ (posL Bool).commit.Binds id ∧
      WindowsPlaced (posL Bool).partition 0 (fun _ : Fin 1 => posStmt false) (fun _ => notPart) (fun _ => 0) ∧
      CheckedUnique notProg notPart (⟨[.all]⟩ : Design) notLay 1 ∅ (fun _ => True ∧ True) ∧
      Verity.Assumptions.TrustedHost ()
        (fun (_ : Unit) (p : CertifierSchedule Unit × Bool) => recordingCode bitFmt p.1 (bitArr p.2))
        (fun p => recordingCode bitFmt p.1 (bitArr p.2)) ∧
      bitFmt.ReadsBy (fun (_ : Unit) => bitFmt.read oneSched) (fun _ s => s = oneSched) ∧
      MonitoringPeriodChannelReg bitFmt (fun (_ : Unit) s => s = oneSched) {()} ({false} : Set Bool) (fun _ => oneSched) id
        (fun v => failedAt0 (v ())) (fun p => (recordingCode bitFmt p.1 (bitArr p.2)).1) ∧
      (∀ (i : Fin 1) (s : AuditStatement), (posL Bool).parse (posX false) = some s →
        CertifierRegisters id (posL Bool).commit notLay 1 (fun _ _ => [false]) s (posTops s)
          ((recordingCode bitFmt oneSched (bitArr false)).2 i)) ∧
      ∀ i : Fin 1, ExecutionPhaseGoodChecked id (posL Bool) 1 1 1 (fun _ => notLay) (fun _ => 1) 0 (fun _ => ∅)
        (fun _ _ => True ∧ True) 1 (posStmt false) (fun _ => false) posTops (posX false) ⟨[]⟩
        ((recordingCode bitFmt oneSched (bitArr false)).2 i) := by
  have ho : ∀ i : Fin 1, (recordingCode bitFmt oneSched (bitArr false)).2 i = [posObs false] :=
    fun i => congrFun (bit_obs false) i
  refine ⟨posL_ks id, posCm_binds id, pos_placed, pos_checkedUnique, fun _ => rfl,
    bitFmt.readsBy_comp fun _ => oneSched, ⟨fun _ _ => ⟨rfl, (), Set.mem_singleton (), rfl⟩, fun z _ => bit_dec z⟩,
    fun i s hs => ?_, fun i => (ho i) ▸ pos_good id false _⟩
  obtain rfl : posStmt false = s := Option.some.inj ((pos_parse (Coin := Bool) false).symm.trans hs)
  exact (ho i) ▸ pos_registers id false (fun _ _ => [false]) rfl

end NetTiming.Witness
