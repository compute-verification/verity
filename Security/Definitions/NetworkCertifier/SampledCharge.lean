import Definitions.NetworkCertifier.SampledAudit

/-!
# The sampled audit's charge: what the faulty units add to the leak

Definitions only. An accepted sampled audit gives `Accepted` by the circuit's Program at `δ`, with the declared advice
holding values at the faulty units' committed gates (`RAcc`, `Definitions.NetworkCertifier.SampledAudit`). Those values
are the prover's to choose, and so is which units are faulty, so the leak bound of a sampled audit charges, beyond the
inputs the circuit reads besides the ingress, the faulty units' committed bits (`faultyBits`) and the choice of faulty set
among the ones `RAcc δ` allows (`faultySets`). The sequences it counts are `AccSeqsSampled`.
-/

namespace NetTiming

open Claims Claims.Sampling

/-- **The faulty units' committed bits**: over the slots `F` of the partition `P`, each replay unit's gates that its
boundary commits (`committed`). -/
def faultyBits (C : Prog) (P : Partition) (F : Finset ℕ) : ℕ :=
  ∑ v ∈ F, committed C (RU.gates (P.rus.getD v []))

open Classical in
/-- **The possible faulty sets at `δ`**: the sets of `P`'s slots that are in `M_δ` and hold no slot drawn `all`, the
faulty sets `RAcc δ` allows. At `δ ≥ miss(1)` none is non-empty. -/
noncomputable def faultySets (d : Design) (P : Partition) (δ : ℝ) : Finset (Finset ℕ) :=
  (Finset.range P.rus.length).powerset.filter fun F => F ∈ d.M P δ ∧ Disjoint F (allSlots d P)

/-- **Accepted with the faulty set `F`**: some declared advice `(a, free)` makes the first `k` steps `Accepted` by the
circuit's Program, with `a`, the inputs other than the ingress, among `As`, and `free` holding values only at the
committed gates of `F`'s units (`FaultyGate`). -/
def AcceptedFaulty {L K S : ℕ} (C : Prog) (P : Partition) (lay : Layout L S) (k : ℕ) (As : Set (ℕ → Bool))
    (F : Finset ℕ) (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)) : Prop :=
  ∃ a ∈ As, ∃ free : ℕ → Option Bool, (∀ g b, free g = some b → FaultyGate C P F g) ∧
    Accepted (circuitProgram C lay k) op k o (a, free)

/-- **The sequences a sampled audit accepts at `δ`**: the observation sequences of length `n`, with jitter below `D`,
accepted with some possible faulty set at `δ` (`AcceptedFaulty`). -/
def AccSeqsSampled {L K S : ℕ} (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (n : ℕ)
    (As : Set (ℕ → Bool)) (δ : ℝ) (op : ℕ → List (Obs L K S) → List Bool) (D : ℕ) : Set (List (Obs L K S)) :=
  {o | o.length = n ∧ (∀ x ∈ o, x.jitter < D) ∧ ∃ F ∈ faultySets d P δ, AcceptedFaulty C P lay n As F op o}

/-! ## The leak relation

`RAcc` gives `Accepted` with some inputs besides the ingress, and bounds none of them. A count over `As` (`AccSeqsSampled`)
then needs the circuit to accept with inputs among `As` whatever it accepts with any (`AdviceIn`). That holds at the
inputs that are `false` off the free inputs (`AsFree`), whatever the circuit, since the Program reads no other input. -/

/-- **Inputs among `As` suffice** at `δ`: a sequence accepted with a possible faulty set and any inputs besides the
ingress is accepted with that set and inputs among `As`. -/
def AdviceIn {L S : ℕ} (K : ℕ) (C : Prog) (P : Partition) (d : Design) (lay : Layout L S) (k : ℕ)
    (As : Set (ℕ → Bool)) (δ : ℝ) : Prop :=
  ∀ F ∈ faultySets d P δ, ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
    AcceptedFaulty C P lay k Set.univ F op o → AcceptedFaulty C P lay k As F op o

/-- **The free inputs** of `C` over the horizon `k`: its input gates that are no step's ingress before `k`, the inputs
the circuit's Program reads from the advice (`Layout.inputOf`). -/
def freeInputs {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) : Finset ℕ :=
  (Finset.range C.gates.length).filter fun g =>
    C.gates[g]? = some .input ∧ (List.range k).any (fun i => (lay.ingress i).contains g) = false

/-- **The inputs `false` off the free inputs**: `2^#freeInputs` of them. -/
def AsFree {L S : ℕ} (C : Prog) (lay : Layout L S) (k : ℕ) : Set (ℕ → Bool) :=
  {a | ∀ g, g ∉ freeInputs C lay k → a g = false}

/-- **The leak relation** at `δ`, on the statement's bytes: the statement parses and has a partition, and every
observation sequence of length `k` with jitter below `Dj` that the trace's transfer gates carry (`Carries`) with every
status complete (`Covered`) is among the sequences `AccSeqsSampled` counts at the inputs besides the ingress `As`: a set
that the statement, `δ`, `As`, the operator and `Dj` fix, whatever the witness. -/
def RLeak (H : Msg → Msg) (Cm : Commitment Bool) (parse : Msg → Option Stmt) (partition : Stmt → Option Partition)
    (L K S : ℕ) (lay : Stmt → Layout L S) (k : Stmt → ℕ) (δ : ℝ) (As : Stmt → Set (ℕ → Bool)) (Dj : ℕ) : Rel Wit :=
  fun x w => ∃ s P, parse x = some s ∧ partition s = some P ∧
    ∀ (op : ℕ → List (Obs L K S) → List Bool) (o : List (Obs L K S)),
      Carries (lay s) s.prog (w.val H Cm s) op (k s) o → Covered (k s) o → o.length = k s → (∀ y ∈ o, y.jitter < Dj) →
        o ∈ AccSeqsSampled s.prog P s.design (lay s) (k s) (As s) δ op Dj

end NetTiming
