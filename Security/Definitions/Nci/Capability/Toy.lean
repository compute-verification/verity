import Definitions.Nci.Capability
import Mathlib.Probability.ProbabilityMassFunction.Constructions
import Mathlib.Probability.Distributions.Uniform

/-!
# Local bounds: the toy instance their witnesses are stated at

One small Boolean circuit, at which the local bounds' witnesses (`Definitions.Nci.Guarantees.Capability.Witnesses`) show
each bound's premises holding together, and its conclusion failing where a premise is dropped. Four one-bit gates:

* `0`, a registered input: the weight, registered as `1`;
* `1`, an unregistered input: the token's operand;
* `2` and `3`, each `0 ∧ 1`: the product, and a second gate computing it again.

The registration `R1` puts the computed gates in one proof block, a `(2, 1)`-partition (two gates, the token its one
unregistered input bit); `R2` puts each in its own. One counted product (`B`) multiplies the weight (`W`, its slot) by the
token (`O`, its operand) at gate `2`; its use is the one consumer of `K`, whose region and required gate are the token. The
item `Y p` is the product's value priced `p`. The decoder `M` reads only programs of two bits: a tag, then one bit it
outputs when the tag is set; every description it gives is two bits long. The run distribution `μ` draws the token
uniformly.

The builder `Ra` takes a one-bit advice `b` to `Rb b`, the toy circuit with its computed gates gated by `b` (`Cb b`),
registered as `R1`; `ya` is what its evaluator extracts. The trace `Xf` holds a wrong product at gate `2`, so the first
proof block of `P2` is faulty.
-/

namespace Nci.Capability.Toy

open Verity Finset

/-- The toy circuit: two inputs, the weight `0` and the token `1`, and two gates `2` and `3` computing `0 ∧ 1`. -/
@[reducible] def C : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := fun g => if g.val < 2 then ∅ else {0, 1}
  reads_lt := by decide
  input_reads := by decide
  op := fun g v => if g.val < 2 then 0 else BitVec.ofBool ((v 0).getLsbD 0 && (v 1).getLsbD 0)
  op_congr := by
    intro g v v' h
    by_cases hg : g.val < 2
    · simp only [hg, ↓reduceIte]
    · simp only [hg, ↓reduceIte]
      rw [h 0 (by simp [hg]), h 1 (by simp [hg])]
  outputs := {2, 3}

/-- The toy declaration: the weight is registered. -/
@[reducible] def D : Declaration where
  D := C
  θ := {0}
  θ_input := by decide

/-- One proof block holding both computed gates. -/
def P1 : Partition C 1 where
  unit := fun g => if g.val < 2 then none else some 0
  input_iff := by decide

/-- Two proof blocks, gate `2` in the first and gate `3` in the second. -/
def P2 : Partition C 2 where
  unit := fun g => if g.val < 2 then none else if g.val = 2 then some 0 else some 1
  input_iff := by decide

/-- The registration with one proof block, the weight registered as `1`. -/
@[reducible] def R1 : Registration where
  d := D
  val := fun _ => (1 : BitVec 1)
  B := 1
  P := P1

/-- The registration with two proof blocks. -/
@[reducible] def R2 : Registration where
  d := D
  val := fun _ => (1 : BitVec 1)
  B := 2
  P := P2

/-- The one counted product: weight `0` times token `0`, at gate `2`, the token's computation being gate `2`. -/
@[reducible] def B : Nci.Covering.MulBlocks D where
  nw := 1
  nt := 1
  P := {(0, 0)}
  m := fun _ => {2}
  o := fun _ => 2
  T := fun _ => {2}
  m_disjoint := by
    intro p hp q hq hpq
    exact absurd ((mem_singleton.1 hp).trans (mem_singleton.1 hq).symm) hpq
  m_computed := by decide
  o_mem := by simp
  reach := by
    intro p _ g hg
    rw [mem_singleton.1 hg]
    exact .refl (mem_singleton_self _)
  T_disjoint := fun j j' h => absurd (Subsingleton.elim j j') h
  T_θ := by decide

/-- The slot items: the weight, priced one bit. -/
def W : Items C (Fin 1) where
  cells := fun _ => {{0}}
  price := fun _ => 1
  price_nonneg := fun _ => zero_le_one

/-- The operand items: the token, priced one bit. -/
def O : Items C (Fin 1) where
  cells := fun _ => {{1}}
  price := fun _ => 1
  price_nonneg := fun _ => zero_le_one

/-- The product's value, priced `p` bits. -/
def Y (p : ℕ) : Items C (Fin 1) where
  cells := fun _ => {{2}}
  price := fun _ => p
  price_nonneg := fun _ => Nat.cast_nonneg p

/-- The two computed gates' values, priced one bit each: item `i` is gate `2 + i`. -/
def W2 : Items C (Fin 2) where
  cells := fun i => if i = 0 then {{2}} else {{3}}
  price := fun _ => 1
  price_nonneg := fun _ => zero_le_one

/-- The product reads the weight and the token. -/
def I : Incidence B W O where
  occ := fun _ => 0
  slots := fun _ _ => ⟨{0}, mem_singleton_self _, show ({0} : Finset (Fin 4)) ⊆ holds C {2} by decide⟩
  operands := fun _ _ => ⟨{1}, mem_singleton_self _, show ({1} : Finset (Fin 4)) ⊆ holds C {2} by decide⟩

/-- The product's use as a consumer: its region and required gate are the token. -/
def K : Consumers D (Fin 1) :=
  Consumers.ofUses B (fun _ => {1}) (fun _ => {1}) (fun _ => subset_refl _) (fun _ => by decide)

/-- A consumer that requires both computed gates. -/
def K2 : Consumers D (Fin 1) where
  fins := fun _ => ∅
  A := fun _ => {2, 3}
  G := fun _ => {2, 3}
  G_sub := fun _ => subset_refl _
  A_θ := fun _ => by decide

/-- The decoder: on a program of two bits it outputs the second when the first is set, and nothing otherwise; it reads
no other program, so every description it gives is two bits long. -/
def M : Decoder Unit where
  run := fun _ p _ => if p.length = 2 then some (if p.head? = some true then p.tail else []) else none
  mono := fun _ _ _ _ _ _ h => h
  prefixFree := by
    intro r t p p' y y' h h' hp
    have h1 : p.length = 2 := by
      by_contra hc
      simp [hc] at h
    have h2 : p'.length = 2 := by
      by_contra hc
      simp [hc] at h'
    exact hp.eq_of_length (h1.trans h2.symm)

/-- The inputs with the token `b` and every other gate `1`. -/
def tok (b : Bool) : C.Transcript := fun g => if g = 1 then BitVec.ofBool b else 1

/-- The run distribution: the token drawn uniformly. -/
noncomputable def μ : PMF C.Transcript := (PMF.uniformOfFintype Bool).map tok

/-- The token's bit, the source the toy recovers. -/
def z (T : C.Transcript) : List Bool := [(T 1).getLsbD 0]

/-- The toy circuit with its computed gates gated by an advice bit `b`: gates `2` and `3` compute `b ∧ 0 ∧ 1`. -/
@[reducible] def Cb (b : Bool) : Circuit where
  N := 4
  w := fun _ => 1
  isInput := fun g => decide (g.val < 2)
  reads := fun g => if g.val < 2 then ∅ else {0, 1}
  reads_lt := by decide
  input_reads := by decide
  op := fun g v => if g.val < 2 then 0 else BitVec.ofBool (b && (v 0).getLsbD 0 && (v 1).getLsbD 0)
  op_congr := by
    intro g v v' h
    by_cases hg : g.val < 2
    · simp only [hg, ↓reduceIte]
    · simp only [hg, ↓reduceIte]
      rw [h 0 (by simp [hg]), h 1 (by simp [hg])]
  outputs := {2, 3}

/-- `Cb b`, registered as `R1` registers the toy circuit: the weight as `1`, both computed gates in one proof block. -/
@[reducible] def Rb (b : Bool) : Registration where
  d := { D := Cb b, θ := {0}, θ_input := by decide +revert }
  val := fun _ => (1 : BitVec 1)
  B := 1
  P := { unit := fun g => if g.val < 2 then none else some 0, input_iff := by decide +revert }

/-- The toy builder: the advice's first bit gates the computed gates, so the block's structure depends on the advice. -/
@[reducible] def Ra (a : List Bool) : Registration := Rb (a.headD false)

/-- What the toy evaluator extracts under the advice `a`: the product's value when the advice is set, and nothing
otherwise. -/
def ya (a : List Bool) (_u : Fin (Ra a).B) (T : (Ra a).d.D.Transcript) : List Bool :=
  if a.headD false then [(T 2).getLsbD 0] else []

/-- A trace with a faulty product: the weight `1` and the token `0`, with gate `2` holding `1` where it computes `0`, and
gate `3` holding `0`. -/
def Xf : C.Transcript := fun g => if g.val = 0 ∨ g.val = 2 then 1 else 0

end Nci.Capability.Toy
