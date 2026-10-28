import Definitions.Accounting.Guarantees.JointLive
import Proofs.Core.Game.Prob
import Proofs.Core.Circuit
import Proofs.Pouw.PearlC.WholeCircuit

/-!
# Proof of the joint guarantee over a causal game

Each subprotocol's per-period guarantee over the causal game (`PeriodLive.Sound`) bounds its failure at every strategy
of the whole game, so at a strategy the union bound over the subprotocols, the periods, the wipes and the memory terms
sums the errors (`prob_exists_le`, `Verity.Game.prob_or_le`). Nothing here reads how the coins interleave: that is in
each subprotocol's own guarantee. With `ecr/sha-512` at each finder, each loss is read at the finder's expected calls
(`JointPeriodLiveECR`). A subprotocol whose own guarantee holds outside a break event gets its `Sound` with the break
charged at its finder (`Live.sound_of_break`), not folded into its acceptance.

The honest prover's extractor reads an evaluation, in which every unit is correct (`correct_eval`), and the commitment
opens every gate to it, so no unit is unbound and the integrity claim holds at every nonnegative share
(`JointPeriodLive_honest`). A one-gate circuit whose extractor reads the wrong bit fails the integrity claim on its one
run (`JointPeriodLive_nontrivial`).
-/

namespace Accounting.SecurityProofs

open Accounting Verity Verity.Game
open scoped ENNReal

/-- **The union bound at a strategy**, over a finite family of events. -/
theorem Live.prob_exists_le {O ι : Type} (t : Finset ι) (p : ι → O → Prop) (G : Game O) (s : Strategy G) :
    prob (fun o => ∃ i ∈ t, p i o) G s ≤ ∑ i ∈ t, prob (p i) G s := by
  classical
  induction t using Finset.induction_on with
  | empty => simp [prob_false]
  | insert j t hj ih =>
      rw [Finset.sum_insert hj]
      calc prob (fun o => ∃ i ∈ insert j t, p i o) G s
          ≤ prob (fun o => p j o ∨ ∃ i ∈ t, p i o) G s :=
            prob_mono (fun o ⟨i, hi, hpi⟩ => by
              rcases Finset.mem_insert.1 hi with rfl | hi
              · exact Or.inl hpi
              · exact Or.inr ⟨i, hi, hpi⟩) G s
        _ ≤ prob (p j) G s + prob (fun o => ∃ i ∈ t, p i o) G s := prob_or_le _ _ G s
        _ ≤ prob (p j) G s + ∑ i ∈ t, prob (p i) G s := add_le_add le_rfl ih

/-- The union bound at a strategy, over a finite type. -/
theorem Live.prob_exists_univ_le {O ι : Type} [Fintype ι] (p : ι → O → Prop) (G : Game O) (s : Strategy G) :
    prob (fun o => ∃ i, p i o) G s ≤ ∑ i, prob (p i) G s :=
  (prob_mono (fun _ ⟨i, hi⟩ => ⟨i, Finset.mem_univ i, hi⟩) G s).trans (Live.prob_exists_le Finset.univ p G s)

/-- **The joint guarantee over a period played as a causal game** (`Accounting.Guarantees.JointPeriodLive`). -/
theorem JointPeriodLive : Accounting.Guarantees.JointPeriodLive := by
  intro ι α _ d C O G r c err h s
  exact (Live.prob_exists_univ_le _ G s).trans (Finset.sum_le_sum fun a _ => h a s)

/-- A finder meeting `ecr/sha-512` within `q` expected calls outputs a collision with probability at most
`q / 2^256`. -/
theorem Live.pr_le_of_expectedCR {H : Claims.Msg → Claims.Msg} {F : Claims.Finder} {q : ℝ}
    (hF : Claims.ExpectedCR H F) (hq : Claims.Ex (fun x => (F.cost x : ℝ)) ≤ q) :
    F.Pr H ≤ q / 2 ^ (256 : ℝ) :=
  hF.trans (div_le_div_of_nonneg_right hq (by positivity))

/-- **The joint guarantee over a period, with `ecr/sha-512` asked of the finders only**
(`Accounting.Guarantees.JointPeriodLiveECR`). -/
theorem JointPeriodLiveECR : Accounting.Guarantees.JointPeriodLiveECR := by
  intro ι α _ d C O G r H c δ loss q h hmono hcr hq s
  refine (JointPeriodLive ι α d C O G r c (fun a s => r.error H a (δ a s) (loss a) s) h s).trans
    (Finset.sum_le_sum fun a _ => ?_)
  show δ a s + loss a (ENNReal.ofReal ((r.finder a s).Pr H)) ≤ _
  exact add_le_add le_rfl
    (hmono a (ENNReal.ofReal_le_ofReal (Live.pr_le_of_expectedCR (hcr a s) (hq a s))))

/-- **A subprotocol's break, charged at its finder.** A subprotocol whose own guarantee bounds its failure outside an
event `Break s` of the run (a root opened two ways, a forged release; PoUS's `RealizesSpaceOfPeriodHeld` reads its
acceptance as `Accepts ∧ ¬ Break`) has its per-period guarantee at `PeriodLive.error`, once the break's probability at
every strategy is at most its loss at the probability that its finder, built from that strategy, outputs a collision:
the break is charged as its own term rather than folded into the acceptance. -/
theorem Live.sound_of_break {ι : Type} {d : Device ι} {α : Type} [Fintype α] {C : Verity.Circuit} {O : Type}
    {G : Game O} (r : PeriodLive d α C G) (H : Claims.Msg → Claims.Msg) (c : Certificates) (a : Audit)
    (δ : Strategy G → ℝ≥0∞) (loss : ℝ≥0∞ → ℝ≥0∞) (Break : Strategy G → O → Prop)
    (hoff : ∀ s, prob (fun o => r.accepts a o ∧ ¬ Break s o ∧ ¬ r.Holds c s a o) G s ≤ δ s)
    (hbreak : ∀ s, prob (Break s) G s ≤ loss (r.collides H a s)) :
    r.Sound c a (fun s => r.error H a (δ s) loss s) := by
  intro s
  show _ ≤ δ s + loss (r.collides H a s)
  calc prob (fun o => r.accepts a o ∧ ¬ r.Holds c s a o) G s
      ≤ prob (fun o => (r.accepts a o ∧ ¬ Break s o ∧ ¬ r.Holds c s a o) ∨ Break s o) G s :=
        prob_mono (fun o ⟨ha, hn⟩ => (Classical.em (Break s o)).elim Or.inr fun hb => Or.inl ⟨ha, hb, hn⟩) G s
    _ ≤ prob (fun o => r.accepts a o ∧ ¬ Break s o ∧ ¬ r.Holds c s a o) G s + prob (Break s) G s :=
        prob_or_le _ _ G s
    _ ≤ δ s + loss (r.collides H a s) := add_le_add (hoff s) (hbreak s)

/-- **No unit is unbound** in a transcript that is an evaluation and that the openings open at every gate. -/
theorem Live.unbound_eval {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n)
    (opens : (g : Fin C.N) → BitVec (C.w g) → Prop) (x : C.Transcript) (ho : ∀ g, opens g (C.eval x g)) :
    unbound P opens (C.eval x) = ∅ := by
  refine Finset.eq_empty_of_forall_notMem fun u hu => ?_
  simp only [unbound, Finset.mem_filter, Finset.mem_univ, true_and] at hu
  rcases hu with hw | ⟨g, -, hg⟩
  · exact hw (Pouw.PearlC.WholeCircuit.correct_eval P x u)
  · exact hg (ho g)

/-- **The honest prover meets every claim** (`Accounting.Guarantees.JointPeriodLive_honest`). -/
theorem JointPeriodLive_honest : Accounting.Guarantees.JointPeriodLive_honest := by
  intro ι α _ d C O G r c s hε hX hO hW hS hQ hK
  have hall : ∀ a o, r.Holds c s a o := by
    intro a o
    cases a with
    | sampled =>
        obtain ⟨x, hx⟩ := hX o
        have hu : unbound r.parts r.opens (r.extract s o) = ∅ := by
          rw [hx]
          exact Live.unbound_eval r.parts r.opens x fun g => by rw [← hx]; exact hO o g
        show IntegrityHolds r.parts r.opens (r.extract s o) c.ε
        rw [IntegrityHolds, hu, Finset.card_empty, Nat.cast_zero]
        exact mul_nonneg hε (Nat.cast_nonneg _)
    | pouw => exact hW o
    | pous => exact hS o
    | poub => exact hQ o
    | network => exact hK o
  refine ⟨hall, le_antisymm ?_ zero_le⟩
  calc prob (fun o => ∃ a, r.accepts a o ∧ ¬ r.Holds c s a o) G s
      ≤ prob (fun _ => False) G s := prob_mono (fun o ⟨a, _, hn⟩ => hn (hall a o)) G s
    _ = 0 := prob_false G s

/-! ## A period whose integrity claim fails -/

/-- One computed gate, reading nothing, holding the bit `1`, and the circuit's output. -/
def Live.toyCircuit : Verity.Circuit where
  N := 1
  w := fun _ => 1
  isInput := fun _ => false
  reads := fun _ => ∅
  reads_lt := fun _ _ h => by simp at h
  input_reads := fun _ h => by simp at h
  op := fun _ _ => 1
  op_congr := fun _ _ _ _ => rfl
  outputs := {0}

/-- The one-unit partition of `toyCircuit`. -/
def Live.toyParts : Verity.Partition Live.toyCircuit 1 where
  unit := fun _ => some 0
  input_iff := fun _ => by simp [Live.toyCircuit]

/-- `toyCircuit`'s one gate. -/
def Live.toyGate : Fin Live.toyCircuit.N := ⟨0, Nat.one_pos⟩

/-- The transcript holding `0` at the gate is not correct at the one unit, whose gate computes `1`. -/
theorem Live.toy_not_correct : ¬ Live.toyParts.Correct (fun _ => 0) 0 := by
  intro hc
  have hg : Live.toyGate ∈ Live.toyParts.committed :=
    Live.toyParts.mem_committed_of_output (Finset.mem_singleton.2 rfl)
  have h := hc Live.toyGate hg rfl
  rw [Verity.Partition.localEval_of_mem Live.toyParts _ (rfl : Live.toyParts.unit Live.toyGate = some 0)] at h
  have hne : (0 : BitVec 1) ≠ 1 := by decide
  exact hne h

/-- A device with no side channels and no resources. -/
def Live.toyDevice : Device Empty := ⟨0, 0, fun _ => 0, 0, fun e => e.elim⟩

/-- The world in which the one account does nothing. -/
def Live.toyWorld : PeriodWorld Live.toyDevice Unit where
  acct := fun _ => ()
  win := ⟨0, 0⟩
  schedule := fun _ => ⟨fun _ _ => 0, fun _ => 0, fun _ e => e.elim, fun _ => 0, fun _ => 0⟩
  fits := fun _ => by simp [Step.Fits]
  commWithin := fun _ => True

/-- The period over `toyCircuit` in the game with no rounds, whose extractor reads `0` at the gate and whose
subprotocols all accept. -/
def Live.toyPeriod : PeriodLive Live.toyDevice Unit Live.toyCircuit (.ret ()) where
  units := 1
  parts := Live.toyParts
  opens := fun _ _ => True
  extract := fun _ _ _ => 0
  finder := fun _ _ => ⟨Unit, fun _ => none, fun _ => 0⟩
  computed := fun _ _ => ∅
  world := fun _ _ => Live.toyWorld
  accepts := fun _ _ => True

/-- The one strategy of the game with no rounds. -/
def Live.toyStrategy : Strategy (Game.ret () : Game Unit) := PUnit.unit

/-- **The bound is not vacuous** (`Accounting.Guarantees.JointPeriodLive_nontrivial`): on the one-gate circuit, a
period whose extractor reads `0` at the gate and whose sampled proofs accept is wrong at its one unit with
probability `1`, at integrity share `0`. -/
theorem JointPeriodLive_nontrivial : Accounting.Guarantees.JointPeriodLive_nontrivial := by
  refine ⟨Empty, Unit, inferInstance, Live.toyDevice, Live.toyCircuit, Unit, .ret (), Live.toyPeriod,
    ⟨0, 0, 0, 0, 0⟩, Live.toyStrategy, ?_⟩
  have hmem : (0 : Fin 1) ∈ unbound Live.toyParts (fun _ _ => True) (fun _ => 0) := by
    simp only [unbound, Finset.mem_filter, Finset.mem_univ, true_and]
    exact Or.inl Live.toy_not_correct
  have hfail : ¬ Live.toyPeriod.Holds ⟨0, 0, 0, 0, 0⟩ Live.toyStrategy .sampled () := by
    show ¬ IntegrityHolds Live.toyParts (fun _ _ => True) (fun _ => 0) 0
    rw [IntegrityHolds, zero_mul, not_le]
    exact_mod_cast Finset.card_pos.2 ⟨0, hmem⟩
  calc (0 : ℝ≥0∞) < 1 := one_pos
    _ = prob (fun _ => True) (Game.ret ()) Live.toyStrategy := (prob_true _ _).symm
    _ ≤ _ := prob_mono (fun _ _ => ⟨.sampled, trivial, hfail⟩) _ _

/-- **The joint guarantee over a horizon of periods played as one causal game**
(`Accounting.Guarantees.JointHorizonLive`). -/
theorem JointHorizonLive : Accounting.Guarantees.JointHorizonLive := by
  intro ι α _ d m C O G r c err h s
  calc prob (fun o => ∃ j a, (r j).accepts a o ∧ ¬ (r j).Holds (c j) s a o) G s
      ≤ ∑ j, prob (fun o => ∃ a, (r j).accepts a o ∧ ¬ (r j).Holds (c j) s a o) G s :=
        Live.prob_exists_univ_le _ G s
    _ ≤ ∑ j, ∑ a, prob (fun o => (r j).accepts a o ∧ ¬ (r j).Holds (c j) s a o) G s :=
        Finset.sum_le_sum fun j _ => Live.prob_exists_univ_le _ G s
    _ ≤ _ := Finset.sum_le_sum fun j _ => Finset.sum_le_sum fun a _ => h j a s

/-- **The joint guarantee over a horizon of server-time blocks and their wipes, played as one causal game**
(`Accounting.Guarantees.JointHorizonBlocksLive`). -/
theorem JointHorizonBlocksLive : Accounting.Guarantees.JointHorizonBlocksLive := by
  intro ι α _ _ d B C O G S c fw σ err εWipe δPous h hW hM s
  have hA := JointHorizonLive ι α d B C O G S.block c err h s
  have hW' : prob (fun o => ∃ b, (S.wipe b).accepts o ∧ ¬ (S.wipe b).Survives fw o) G s ≤ ∑ b, εWipe b s :=
    (Live.prob_exists_univ_le _ G s).trans (Finset.sum_le_sum fun b _ => hW b s)
  have hM' : prob (fun o => ∃ b, (S.block b).accepts .pous o ∧ ¬ S.MemoryTerm (σ b) s b o) G s ≤ ∑ b, δPous b s :=
    (Live.prob_exists_univ_le _ G s).trans (Finset.sum_le_sum fun b _ => hM b s)
  calc _ ≤ _ := prob_or_le _ _ G s
    _ ≤ _ := add_le_add le_rfl (prob_or_le _ _ G s)
    _ ≤ _ := add_le_add hA (add_le_add hW' hM')
    _ = _ := (add_assoc _ _ _).symm

end Accounting.SecurityProofs
