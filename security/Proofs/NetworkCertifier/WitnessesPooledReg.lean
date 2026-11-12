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
does (`twoObs`). The phase's error is the instance's own. An `example` reads the guarantee there.

Its nontrivial lemma (`PooledLeakRegistered_nontrivial`): the bound fails at a verifier that always accepts
(`leakL`), at `q = 1`, `Fmax = 1` and no error, where the phase's record is empty and its statement has one step, so
its good event fails at every history.
-/

@[expose] public section

namespace NetTiming.Witness

open Claims Claims.Sampling Claims.Witness
open Claims.SamplingGeneric (Arg Compiled)
open Claims.Witness.SamplingGeneric (twoA twoL twoStmt twoPart leakL notStmt notPart twoSessions_meets twoL_parse
  twoL_partition)

local notation "shaD" => FlockDeploy.sha512

/-! The honest instance: one execution phase, its coins a tape of no bit and two coins (`Ω₂`); `twoL` at every
statement (`D₂`); the empty bytes as its statement (`x₂`); the prover that sends nothing (`P₂`); both input gates
registered open at `true` (`ops₂`); one step's record over no egress link (`obs₂`); and the instance's own error
(`e₂`). -/

local notation "Ω₂" => (Fin 1 → Unit × (Fin 2 → Bool))
local notation "D₂" => (fun _ => twoL Bool : Msg → Compiled Bool (twoA Bool))
local notation "x₂" => (fun _ _ _ => [] : Unit → Ω₂ → Fin 1 → Msg)
local notation "P₂" => (fun _ _ _ => silentP : Unit → Ω₂ → Fin 1 → Prover Unit)
local notation "ops₂" => (fun _ _ _ _ => certOps : Unit → Ω₂ → Fin 1 → AuditStatement → List (Opening Bool))
local notation "obs₂" => (fun _ _ _ => [twoObs] : Unit → Ω₂ → Fin 1 → List (Obs 0 1 1))
local notation "e₂" =>
  (fun _ => (twoL Bool).εregX shaD [] silentP +
    (BopsX shaD (twoL Bool) (fun _ => certOps) Unit silentP []).Pr shaD : Fin 1 → ℝ)

/-- `twoLay` fits the program of two inputs over one step: it has no egress gate, and its ingress is an input. -/
theorem twoLay_fits : twoLay.Fits twoStmt.prog 1 := by
  refine ⟨fun t _ ℓ => ℓ.elim0, fun t _ g hg => ?_, fun t _ t' _ htt' => absurd htt' (by omega)⟩
  simp [twoLay] at hg
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
    · simp [twoLay] at hg
      subst hg
      exact ⟨⟨0, true, []⟩, by simp [certOps], rfl⟩
    · simp [twoLay, certifierVal, certOps]

/-- **`PooledLeakRegistered`'s premises at an instance that accepts**: one execution phase audited by `twoL` against
the prover that sends nothing, at the rate `1` of a program sampled whole, the registered input the gate `1`, the
registered openings `certOps`, the record `twoObs` at the layout `twoLay`, and the phase's own error. The proof
system's knowledge soundness, the commitment's binding and the sessions' contract hold there (`twoSessions_meets`);
the verifier has two rounds; nothing reads a coin; the registered openings pass and open the registered input; the
layout fits; the network certifier registers the record; and the accepted phase's statement is registered as the
period's, with a complete record. -/
theorem PooledLeakRegistered_honest :
    (twoA Bool).KS shaD ∧ (twoL Bool).commit.Binds shaD ∧ (twoL Bool).Sessions shaD ∧ PhaseRounds D₂ 2 x₂ ∧
      PhasesAdaptive x₂ P₂ ops₂ ∧
      (∀ s, (twoL Bool).parse [] = some s → RegOpens shaD (twoL Bool).commit s {1} certOps) ∧
      (∀ s Pt, (twoL Bool).parse [] = some s → (twoL Bool).partition s = some Pt → RateAt s Pt 1) ∧
      PhaseErrors shaD D₂ Set.univ x₂ P₂ ops₂ e₂ ∧ twoLay.Fits twoStmt.prog 1 ∧
      (∀ s, (twoL Bool).parse [] = some s →
        CertifierRegisters shaD (twoL Bool).commit twoLay 1 (fun _ _ => [true]) s certOps [twoObs]) ∧
      PhasesRegistered shaD D₂ (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) (fun _ => twoStmt) (fun _ => twoPart)
        (fun _ _ => true) Set.univ x₂ P₂ (fun _ _ _ => le_rfl) ops₂ ∧
      RecordsComplete shaD D₂ (fun _ => 1) 1 Set.univ x₂ P₂ (fun _ _ _ => le_rfl) obs₂ := by
  obtain ⟨hA, hb, hS, -⟩ := twoSessions_meets (Coin := Bool) shaD
  refine ⟨hA, hb, hS, fun _ _ _ => le_rfl, ⟨fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl, fun _ _ _ _ _ => rfl⟩,
    fun s hs => ⟨fun o ho => ?_, fun g hg => ⟨⟨1, true, []⟩, by simp [certOps], (Finset.mem_singleton.1 hg).symm⟩⟩,
    fun s Pt hs hPt => ?_, fun _ _ _ _ => le_rfl, twoLay_fits, fun s hs => ?_, fun z _ ω i s hs _ => ?_,
    fun z _ ω i s _ _ => ⟨fun t ht _ κ => ?_, rfl, fun y hy => ?_⟩⟩
  · simp only [certOps, List.mem_cons, List.not_mem_nil, or_false] at ho
    rcases ho with rfl | rfl <;> rfl
  · obtain rfl := twoL_parse hs
    obtain rfl := twoL_partition hPt
    exact rateAt_all rfl rfl
  · obtain rfl := twoL_parse hs
    exact twoLay_registers
  · obtain rfl := twoL_parse hs
    exact ⟨⟨rfl, rfl, rfl, rfl, rfl, rfl⟩, rfl, fun g hg => by
      obtain rfl := Finset.mem_singleton.1 hg
      rfl⟩
  · obtain rfl : t = 0 := Nat.lt_one_iff.1 ht
    rfl
  · simp only [List.mem_singleton] at hy
    subst hy
    exact Nat.one_pos

/-- At that instance the guarantee's hypotheses are those premises, so its bound reads there: at `q = 1` and
`Fmax = 1`, the phase is accepted outside its good event with probability at most its own error. -/
example (h : Guarantees.PooledLeakRegistered) :
    Pr (fun ω : Ω₂ => (∀ i, phaseAcc shaD D₂ x₂ P₂ (fun _ _ _ => le_rfl) () i ω (ω i)) ∧
        ¬PooledGood (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) 1 (fun _ => twoStmt) (fun _ => twoPart)
          (fun _ _ => true) (fun _ _ _ _ => [true]) 1 (WregR shaD D₂ x₂ P₂ ops₂ () ω) (obs₂ () ω)) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ i, e₂ i := by
  obtain ⟨hA, hb, hS, -, had, hops, hrate, he, hfit, hcert, hreg, hrec⟩ := PooledLeakRegistered_honest
  exact h shaD D₂ hA (fun _ => hb) (fun _ => hS) (fun _ => twoLay) (fun _ => 1) (fun _ => {1}) 1 (fun _ => twoStmt)
    (fun _ => twoPart) (fun _ _ => true) (fun _ _ _ _ => [true]) Set.univ x₂ P₂ (fun _ _ _ => le_rfl) ops₂ obs₂ had
    (fun _ _ _ _ s hs => hops s hs) 1 zero_le_one le_rfl (fun _ _ _ _ s Pt hs hPt => hrate s Pt hs hPt) e₂ he
    (fun _ => hfit) (fun _ _ _ _ s hs _ => hcert s hs) hreg hrec 1 () (Set.mem_univ _)

/-- **`PooledLeakRegistered`'s bound fails at a verifier that always accepts**: one execution phase audited by `leakL`
with no registered opening, at `q = 1`, `Fmax = 1` and no error, the phase's record empty where its statement has one
step, so its good event fails at every history. -/
theorem PooledLeakRegistered_nontrivial :
    ¬ Pr (fun ω : Fin 1 → Unit × (Fin 1 → Bool) =>
        (∀ i, phaseAcc (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
          (fun _ _ _ => le_rfl) () i ω (ω i)) ∧
        ¬PooledGood (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt)
          (fun _ => notPart) (fun _ _ => false) (fun _ _ _ _ => [false]) 1
          (WregR (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
            (fun _ _ _ _ => []) () ω) (fun _ => [])) ≤
      (1 - 1 : ℝ) ^ 1 + ∑ _ : Fin 1, (0 : ℝ) := by
  have hp : (fun ω : Fin 1 → Unit × (Fin 1 → Bool) =>
      (∀ i, phaseAcc (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
        (fun _ _ _ => le_rfl) () i ω (ω i)) ∧
      ¬PooledGood (K := 1) (fun _ => notLay) (fun _ => 1) (fun _ => ∅) 1 (fun _ : Fin 1 => notStmt)
        (fun _ => notPart) (fun _ _ => false) (fun _ _ _ _ => [false]) 1
        (WregR (m := 1) (kV := 1) shaD (fun _ => leakL Bool) (fun _ _ _ => []) (fun _ _ _ => silentP)
          (fun _ _ _ _ => []) () ω) (fun _ => [])) = fun _ => True :=
    funext fun ω => propext (iff_true_intro
      ⟨fun i => run_accV_out shaD silentP [] (ω i).1 _, fun h => absurd (h.2 0).2.1 (by simp)⟩)
  rw [hp, Claims.Sampling.Pr_true]
  norm_num

end NetTiming.Witness
