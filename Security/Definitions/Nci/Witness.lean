import Definitions.Nci.Capability.Toy
import Definitions.Nci.Threads
import Definitions.Nci.Dark
import Definitions.Nci.Links
import Definitions.Nci.Fit
import Definitions.Nci.Layered
import Definitions.Nci.Packed
import Definitions.Nci.BitView

/-!
# The (c, k) guarantees: the instances their witnesses are stated at

The witnesses of the inference-only policy's guarantees in `(c, k)` (`Definitions.Nci.Guarantees.Witnesses`) are stated
at the local bounds' toy circuit (`Nci.Capability.Toy.C`): a registered weight `0`, an unregistered token `1`, and two
gates `2` and `3` computing `0 ∧ 1`. Its declaration `Toy.D` registers the weight; `Dθ` registers the token too, so
that nothing a proof block reads is fresh and the policy holds at read cap `0`.

* `idC`: a circuit contained in itself, and `M1` and `Mθ`, the toy in `Toy.D` and in `Dθ` through it; `ones`, the toy's
  run with every input `1`;
* `ch1`: the one-product chain, `W[0, 0] x[0]` at gate `2`, its entry complete there; `ch0`, the chain of no
  multiplication;
* `T1`: the toy's threading, one counted multiplication (gate `2`) of the weight by the token, the token the one
  thread's value; `T0`, the threading of no counted multiplication; `σ1` and `σ0`, their one parameter slot;
  `noLinks`, links with no sources; `layered0` and `layered1`, `T0` and `T1` keyed with no roots and no lookups;
* `E0`, the declaration of no gates, which implements itself (`implE0`);
* `mb0`, the cover theorem's multiplication blocks of no product on a declaration's bit view;
* `OneWindow`: one PoUW window of capacity `2` over the outcomes `Bool`, its certificate `right` or `wrong` at every
  outcome, and no faulty proof block (`bad`).
-/

namespace Nci.Witness

open Verity Finset Nci.Capability.Toy

/-- The toy declaration with the token registered too: both inputs are registered information. -/
@[reducible] def Dθ : Declaration where
  D := C
  θ := {0, 1}
  θ_input := by decide

/-- A circuit contains itself. -/
def idC (C' : Circuit) : Containment C' C' where
  emb := id
  emb_inj := Function.injective_id
  emb_w _ := rfl
  isInput_emb _ := rfl
  reads_emb _ _ := (map_refl).symm
  op_emb _ _ _ := rfl
  outputs_emb _ h := h

/-- The toy contained in itself: a partial containment with no dark gate. -/
abbrev M1 : PartialContainment D D := PartialContainment.ofContainment (idC C)

/-- The toy contained in itself with the token registered (`Dθ`). -/
abbrev Mθ : PartialContainment D Dθ := PartialContainment.ofContainment (idC C)

/-- The run of the toy with every input `1`: the weight at its registered value. -/
def ones : C.Transcript := fun _ => 1

/-- The one-product chain: one multiplication of inner and output dimension `1` at one position, its product
`W[0, 0] x[0]` at gate `2`, which holds the complete entry. -/
def ch1 : Chain C where
  L := 1
  N := 1
  d _ := 1
  e _ := 1
  prod _ _ _ _ := 2
  prod_inj := by
    rintro ⟨ℓ, t, i, j⟩ ⟨ℓ', t', i', j'⟩ -
    obtain rfl : ℓ = ℓ' := Subsingleton.elim _ _
    obtain rfl : t = t' := Subsingleton.elim _ _
    obtain rfl : i = i' := Subsingleton.elim (α := Fin 1) _ _
    obtain rfl : j = j' := Subsingleton.elim (α := Fin 1) _ _
    rfl
  prod_computed _ _ _ _ := rfl
  y _ _ _ := 2
  sums _ _ _ _ := .refl 2
  π ℓ h := absurd h (by omega)
  π_inj ℓ h := absurd h (by omega)
  links ℓ h := absurd h (by omega)

/-- The chain of no multiplication. -/
def ch0 (D : Circuit) : Chain D where
  L := 0
  N := 0
  d := Fin.elim0
  e := Fin.elim0
  prod ℓ := ℓ.elim0
  prod_inj p := p.1.elim0
  prod_computed ℓ := ℓ.elim0
  y ℓ := ℓ.elim0
  sums ℓ := ℓ.elim0
  π ℓ := ℓ.elim0
  π_inj ℓ := ℓ.elim0
  links ℓ := ℓ.elim0

/-- The threading of no counted multiplication. -/
def T0 (r : Declaration) : Threading r where
  m := 0
  W := ∅
  thread _ := none
  thread_W _ _ := rfl
  M := ∅
  M_computed := empty_subset _
  weight := id
  weight_W _ h := absurd h (notMem_empty _)
  weight_reads _ h := absurd h (notMem_empty _)
  act := id
  act_reads _ h := absurd h (notMem_empty _)
  act_thread _ h := absurd h (notMem_empty _)
  reuse := 0
  reuse_le _ _ := by simp

/-- The toy's threading: gate `2` is one counted multiplication of the weight `0` by the token `1`, the one value of the
one thread, used once. -/
def T1 : Threading D where
  m := 1
  W := {0}
  thread g := if g = 1 then some 0 else none
  thread_W := by decide
  M := {2}
  M_computed := by decide
  weight _ := 0
  weight_W := by decide
  weight_reads := by decide
  act _ := 1
  act_reads := by decide
  act_thread := by decide
  reuse := 1
  reuse_le := by decide

/-- `T0`'s one parameter slot. -/
def σ0 (r : Declaration) : (T0 r).Slots where
  p := 1
  slot _ := 0
  reuse_le _ _ := by simp [T0]

/-- `T1`'s one parameter slot, the weight's. -/
def σ1 : T1.Slots where
  p := 1
  slot _ := 0
  reuse_le := by decide

/-- The circuit of no gates. -/
@[reducible] def C0 : Circuit where
  N := 0
  w := Fin.elim0
  isInput := Fin.elim0
  reads := Fin.elim0
  reads_lt g := g.elim0
  input_reads g := g.elim0
  op g := g.elim0
  op_congr g := g.elim0
  outputs := ∅

/-- The declaration of no gates. -/
@[reducible] def E0 : Declaration where
  D := C0
  θ := ∅
  θ_input _ h := absurd h (notMem_empty _)

/-- The declaration of no gates implements itself. -/
def implE0 : Implementation C0 C0 where
  inp := id
  inp_input g := g.elim0
  inp_inj g := g.elim0
  inp_w g := g.elim0
  out := id
  out_mem _ h := absurd h (notMem_empty _)
  out_w _ h := absurd h (notMem_empty _)
  correct _ _ h := absurd h (notMem_empty _)

/-- Links with no sources: every activation is computed from no counted multiplication. -/
def noLinks {r : Declaration} (T : Threading r) : T.Links where
  src _ := ∅
  src_M _ := empty_subset _
  src_feeds _ _ h := absurd h (notMem_empty _)

/-- `T0` keyed with no roots and no lookups. -/
def layered0 (r : Declaration) (d e eLk b : ℕ) : Layered (T0 r) d e eLk b where
  key _ := none
  key_reads _ _ h := absurd h (by simp)
  key_thread _ _ h := absurd h (by simp)
  roots := ∅
  lookups := ∅
  keyed _ _ _ h := absurd h (by simp [T0])
  thread_θ _ _ h := absurd h (by simp [T0])
  legs _ := ∅
  legs_reads _ h := absurd h (notMem_empty _)
  legs_thread _ h := absurd h (notMem_empty _)
  legs_card _ h := absurd h (notMem_empty _)
  legs_disjoint _ h := absurd h (notMem_empty _)
  reach _ X hX _ := by simp [subset_empty.1 (show X ⊆ ∅ from hX)]
  reach_lookups _ h := absurd h (notMem_empty _)
  lookup_input _ h := absurd h (notMem_empty _)
  col _ := ∅
  col_reads _ h := absurd h (notMem_empty _)
  col_input _ h := absurd h (notMem_empty _)
  col_bits _ h := absurd h (notMem_empty _)
  col_disjoint _ h := absurd h (notMem_empty _)

/-- `T1` keyed with no roots and no lookups: the token has no key, and its one counted multiplication reaches it. -/
def layered1 (d b : ℕ) : Layered T1 d 1 0 b where
  key _ := none
  key_reads _ _ h := absurd h (by simp)
  key_thread _ _ h := absurd h (by simp)
  roots := ∅
  lookups := ∅
  keyed := by decide
  thread_θ := by decide
  legs _ := ∅
  legs_reads _ h := absurd h (notMem_empty _)
  legs_thread _ h := absurd h (notMem_empty _)
  legs_card _ h := absurd h (notMem_empty _)
  legs_disjoint _ h := absurd h (notMem_empty _)
  reach _ X hX _ := (card_le_card hX).trans (by decide)
  reach_lookups _ h := absurd h (notMem_empty _)
  lookup_input _ h := absurd h (notMem_empty _)
  col _ := ∅
  col_reads _ h := absurd h (notMem_empty _)
  col_input _ h := absurd h (notMem_empty _)
  col_bits _ h := absurd h (notMem_empty _)
  col_disjoint _ h := absurd h (notMem_empty _)

/-- The multiplication blocks of no product on a declaration's bit view: no counted product has a use, so the upstream
certificate holds at every depth. -/
def mb0 (r : Declaration) : Cover.MulBlocks (Cover.Circuit.ofDecl r) Empty Unit where
  Pall := ∅
  P := ∅
  P_subset := subset_rfl
  m _ := ∅
  m_nonempty p := p.1.elim
  m_computed p := p.1.elim
  m_disjoint p := p.1.elim
  o p := p.1.elim
  m_outputs p := p.1.elim

/-! One PoUW window of capacity `2` over the outcomes `Bool`, and no faulty proof block. -/

namespace OneWindow

/-- The window's capacity. -/
abbrev cap : Fin 1 → ℝ := fun _ => 2

/-- No certificate is wrong. -/
abbrev right : Bool → Fin 1 → Prop := fun _ _ => False

/-- Every certificate is wrong. -/
abbrev wrong : Bool → Fin 1 → Prop := fun _ _ => True

/-- No faulty proof block. -/
abbrev bad : Bool → Finset Empty := fun _ => ∅

/-- A faulty proof block's identity, of none. -/
abbrev ident : Empty → Unit := fun b => b.elim

/-- A faulty proof block's time and the declared work it skips, of none. -/
abbrev time : Empty → ℝ := fun b => b.elim

end OneWindow

end Nci.Witness
