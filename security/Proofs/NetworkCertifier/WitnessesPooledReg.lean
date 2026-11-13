module

public import Definitions.NetworkCertifier.Guarantees.PooledReg
public import Definitions.NetworkCertifier.Witness
public import Proofs.Flock.Claims.Deploy
public import Proofs.Flock.Claims.Witnesses.Shared
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.NetworkCertifier.PooledRegSample

/-!
# The witnesses of the pooled bound at the trace with the registered openings

`PooledLeakRegistered`'s honest lemma (`PooledLeakRegistered_honest`) is its premises at one instance that accepts: a
monitoring period of one execution phase, audited by the compiled verifier of two sampled sessions (`twoL`, over
`twoA`, at SHA-512) against the prover that sends nothing, which it accepts. The program is two inputs sampled whole,
so the rate is `1` (`rateAt_all`); the layout has no egress link and the input gate `0` as its ingress (`twoLay`), the
registered input is the gate `1`, and the registered openings open both at `true` (`certOps`), as the one step's record
does (`twoObs`). The phase's error is the instance's own. Its last conjunct is the guarantee's event there, which is
not empty: at every history the phase is accepted, registered as the period's and its record complete
(`twoPhase_registered`, `twoPhase_record`). So an `example` reads the guarantee as a bound on the phase accepted
outside its good event.

Its nontrivial lemma (`PooledLeakRegistered_nontrivial`): the bound fails at a verifier that always accepts
(`leakL`), at `q = 1`, `Fmax = 1` and no error, where the phase is registered as the period's with a complete record
of one step and the layout does not fit the registered program (`trivLay`'s egress gate is no output of it), so its
good event fails at every history.
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Witness.SamplingGeneric (twoA twoL twoStmt twoPart leakL notStmt notPart twoSessions_meets twoL_parse
  twoL_partition)

local notation "shaD" => FlockDeploy.sha512

/-! The honest instance: a layout of no egress link whose ingress at every step is the input gate `0` (`twoLay`); one
step's record over no egress link, jitter class `0`, every status complete (`twoObs`); one execution phase, its coins
a tape of no bit and two coins (`Ω₂`); `twoL` at every statement (`D₂`); the empty bytes as its statement (`x₂`); the
prover that sends nothing (`P₂`); both input gates registered open at `true` (`ops₂`); that one record (`obs₂`); and
the instance's own error (`e₂`). -/

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

/-- `twoLay` fits the program of two inputs over one step: it has no egress gate, and its ingress is an input. -/
theorem twoLay_fits : (twoLay).Fits twoStmt.prog 1 := by
  refine ⟨fun t _ ℓ => ℓ.elim0, fun t _ g hg => ?_, fun t _ t' _ htt' => absurd htt' (by omega)⟩
  simp at hg
  subst hg
  rfl

/-- The network certifier registers the one step's record `twoObs` at `twoLay` among `certOps`: both openings pass
under `twoL`'s commitment, the ingress gate has one, and the operator's move is the ingress it holds. -/
theorem twoLay_registers :
    CertifierRegisters shaD (twoL Bool).commit twoLay 1 (fun _ _ => [true]) twoStmt certOps [twoObs] := by
  refine ⟨fun q hq => ?_, fun t ht => ?_⟩
  · simp only [certOps, List.mem_cons, List.not_mem_nil, or_false] at hq
    rcases hq with rfl | rfl <;> rfl
  · obtain rfl : t = 0 := by omega
    refine ⟨fun ℓ => ℓ.elim0, fun g hg => ?_, Nat.one_pos, fun ℓ => ℓ.elim0, ?_⟩
    · simp at hg
      subst hg
      exact ⟨⟨0, true, []⟩, by simp [certOps], rfl⟩
    · simp [certifierVal, certOps]

/-- At the honest instance the phase is registered as the period's: its statement parses to `twoStmt` with the
partition `twoPart`, and the registered openings hold `true` at the registered input, the gate `1`. -/
theorem twoPhase_registered (ω : Ω₂) (i : Fin 1) :
    PhaseRegistered D₂ (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) (fun _ => twoStmt) (fun _ => twoPart)
      (fun _ _ => true) x₂ ops₂ () ω i :=
  ⟨twoStmt, rfl, ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, fun g hg => by
    obtain rfl := Finset.mem_singleton.1 hg
    rfl⟩

/-- At the honest instance the phase's record is complete: one observation for the one step, every status complete,
its jitter class `0` below `1`. -/
theorem twoPhase_record (ω : Ω₂) (i : Fin 1) : RecordComplete (fun _ => 1) 1 (fun _ => twoStmt) obs₂ () ω i := by
  refine ⟨fun t ht _ κ => ?_, rfl, fun y hy => ?_⟩
  · obtain rfl : t = 0 := Nat.lt_one_iff.1 ht
    rfl
  · simp only [List.mem_singleton] at hy
    subst hy
    exact Nat.one_pos

/-- **`PooledLeakRegistered`'s premises at an instance that accepts**: one execution phase audited by `twoL` against
the prover that sends nothing, at the rate `1` of a program sampled whole, the registered input the gate `1`, the
registered openings `certOps`, the record `twoObs` at the layout `twoLay`, and the phase's own error. The proof
system's knowledge soundness, the commitment's binding and the sessions' contract hold there (`twoSessions_meets`);
the verifier has two rounds; nothing reads a coin; the registered openings pass and open the registered input; the
layout fits; the network certifier registers the record; and at every history the phase is accepted, registered as
the period's and its record complete, which is the guarantee's event. -/
theorem PooledLeakRegistered_honest :
    (twoA Bool).KS shaD ∧ (twoL Bool).commit.Binds shaD ∧ (twoL Bool).Sessions shaD ∧ PhaseRounds D₂ 2 x₂ ∧
      PhasesAdaptive x₂ P₂ ops₂ ∧
      (∀ s, (twoL Bool).parse [] = some s → RegOpens shaD (twoL Bool).commit s {1} certOps) ∧
      (∀ s Pt, (twoL Bool).parse [] = some s → (twoL Bool).partition s = some Pt → RateAt s Pt 1) ∧
      PhaseErrors shaD D₂ x₂ P₂ ops₂ e₂ ∧ (twoLay).Fits twoStmt.prog 1 ∧
      (∀ s, (twoL Bool).parse [] = some s →
        CertifierRegisters shaD (twoL Bool).commit twoLay 1 (fun _ _ => [true]) s certOps [twoObs]) ∧
      ∀ (ω : Ω₂) (i : Fin 1), phaseAcc shaD D₂ x₂ P₂ (fun _ _ _ => le_rfl) () i ω (ω i) ∧
        PhaseRegistered D₂ (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) (fun _ => twoStmt) (fun _ => twoPart)
          (fun _ _ => true) x₂ ops₂ () ω i ∧
        RecordComplete (fun _ => 1) 1 (fun _ => twoStmt) obs₂ () ω i := by
  obtain ⟨hA, hb, hS, -, hacc⟩ := twoSessions_meets (Coin := Bool) shaD
  refine ⟨hA, hb, hS, fun _ _ _ => le_rfl, ⟨fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl⟩,
    fun s hs => ⟨fun o ho => ?_, fun g hg => ⟨⟨1, true, []⟩, by simp [certOps], (Finset.mem_singleton.1 hg).symm⟩⟩,
    fun s Pt hs hPt => ?_, fun _ _ _ => le_rfl, twoLay_fits, fun s hs => ?_,
    fun ω i => ⟨(hacc _).2, twoPhase_registered ω i, twoPhase_record ω i⟩⟩
  · simp only [certOps, List.mem_cons, List.not_mem_nil, or_false] at ho
    rcases ho with rfl | rfl <;> rfl
  · obtain rfl := twoL_parse hs
    obtain rfl := twoL_partition hPt
    exact rateAt_all rfl rfl
  · obtain rfl := twoL_parse hs
    exact twoLay_registers

/-- At that instance the guarantee's hypotheses are those premises and its event is every phase accepted, so its
bound reads there: at `q = 1` and `Fmax = 1`, the phase is accepted outside its good event with probability at most
its own error. -/
example (h : Guarantees.PooledLeakRegistered) :
    Pr (fun ω : Ω₂ => (∀ i, phaseAcc shaD D₂ x₂ P₂ (fun _ _ _ => le_rfl) () i ω (ω i)) ∧
        ¬PooledGood (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) 1 (fun _ => twoStmt) (fun _ => twoPart)
          (fun _ _ => true) (fun _ _ _ _ => [true]) 1 (WregR shaD D₂ x₂ P₂ ops₂ () ω) (obs₂ () ω)) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ i, e₂ i := by
  obtain ⟨hA, hb, hS, -, had, hops, hrate, he, hfit, hcert, hev⟩ := PooledLeakRegistered_honest
  refine le_trans (Claims.Pr_mono fun ω hω => ?_)
    (h shaD D₂ hA (fun _ => hb) (fun _ => hS) (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) 1 (fun _ => twoStmt)
      (fun _ => twoPart) (fun _ _ => true) (fun _ _ _ _ => [true]) x₂ P₂ (fun _ _ _ => le_rfl) ops₂ obs₂ had
      (fun _ _ _ s hs => hops s hs) 1 zero_le_one le_rfl (fun _ _ _ s Pt hs hPt => hrate s Pt hs hPt) e₂ he
      (fun _ => hfit) (fun _ _ _ s hs _ => hcert s hs) 1 ())
  exact ⟨fun i => ⟨hω.1 i, (hev ω i).2⟩, hω.2⟩

/-! The instance the bound fails at: one step's record over one egress link that holds no frame, jitter class `0`,
every status complete (`obs₁`). -/

local notation "obs₁" => ([Obs.mk (fun _ _ => none) 0 (fun _ => Status.complete)] : List (Obs 1 1 1))

/-- **`PooledLeakRegistered`'s bound fails at a verifier that always accepts**: one execution phase audited by `leakL`
with no registered opening, at `q = 1`, `Fmax = 1` and no error. The phase is registered as the period's and its
record of one step is complete, and the layout `trivLay` does not fit the registered program, its egress gate being no
output of it, so the phase's good event fails at every history. -/
theorem PooledLeakRegistered_nontrivial :
    ¬ Pr (fun ω : Fin 1 → Unit × (Fin 1 → Bool) =>
        (∀ i, phaseAcc (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
            (fun _ _ _ => le_rfl) () i ω (ω i) ∧
          PhaseRegistered (m := 1) (kV := 1) (fun _ => leakL Bool) (fun _ => trivLay) (fun _ => 1) (fun _ => ∅)
            (fun _ => notStmt) (fun _ => notPart) (fun _ _ => false) (fun _ _ _ => []) (fun _ _ _ _ => []) () ω i ∧
          RecordComplete (m := 1) (kV := 1) (fun _ => 1) 1 (fun _ => notStmt) (fun _ _ _ => obs₁) () ω i) ∧
        ¬PooledGood (fun _ => trivLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
          (fun _ _ => false) (fun _ _ _ _ => [false]) 1
          (WregR (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
            (fun _ _ _ _ => []) () ω) (fun _ => obs₁)) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) := by
  have hp : (fun ω : Fin 1 → Unit × (Fin 1 → Bool) =>
      (∀ i, phaseAcc (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
          (fun _ _ _ => le_rfl) () i ω (ω i) ∧
        PhaseRegistered (m := 1) (kV := 1) (fun _ => leakL Bool) (fun _ => trivLay) (fun _ => 1) (fun _ => ∅)
          (fun _ => notStmt) (fun _ => notPart) (fun _ _ => false) (fun _ _ _ => []) (fun _ _ _ _ => []) () ω i ∧
        RecordComplete (m := 1) (kV := 1) (fun _ => 1) 1 (fun _ => notStmt) (fun _ _ _ => obs₁) () ω i) ∧
      ¬PooledGood (fun _ => trivLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt) (fun _ => notPart)
        (fun _ _ => false) (fun _ _ _ _ => [false]) 1
        (WregR (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
          (fun _ _ _ _ => []) () ω) (fun _ => obs₁)) = fun _ => True := by
    funext ω
    refine propext (iff_true_intro ⟨fun i => ⟨run_accV_out shaD silentP [] (ω i).1 _, ?_, ?_⟩, fun h => ?_⟩)
    · exact ⟨notStmt, rfl, ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, fun g hg => by simp at hg⟩
    · refine ⟨fun t ht _ κ => ?_, rfl, fun y hy => ?_⟩
      · obtain rfl : t = 0 := Nat.lt_one_iff.1 ht
        rfl
      · simp only [List.mem_singleton] at hy
        subst hy
        exact Nat.one_pos
    · exact absurd ((h.2 0).1.1 0 Nat.one_pos 0 0 0 (by simp [trivLay, slotGates])).1 (by decide)
  rw [hp, Claims.Sampling.Pr_true]
  norm_num

end NetTiming.Witness
