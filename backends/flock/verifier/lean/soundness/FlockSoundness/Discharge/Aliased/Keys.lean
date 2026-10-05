import FlockSoundness.Discharge.Aliased.Copies
import FlockSoundness.Protocol.Discharge.Aliased.Keys

/-!
# Δ's source as a key the same at every VU

Δ copies into unit copy `s₀`'s input port bit a column that depends on `s₀` only through its copy index `s₀ % upv` and
its VU `s₀ / upv` (`flatSrc`). `srcKey c x` names that column for VU-stacked input `x` without the VU: a forced-zero row,
a message bit of the VU's rows, an output bit of a net's slot in the VU, a fixed column, or (at an input padding row) the
row itself. `realize` turns a key back into a column at VU `g` (`flatSrc_key`).

So `Copies` holds for an instance that puts inputs on one gate only when they have one key (`copies_of_key`), at every
VU at once: equal keys are one column at VU `g`, or forced-zero rows. `keyIdx` encodes the keys of `n` inputs in
`Option (Fin n)`, `none` exactly at the zero (`keyIdx_eq_none`), and equal codes are equal keys (`keyIdx_eq`): a
program can index its input gates by it.

A typed flat class's slot is one unit copy, and its input `i` is the port bit `inCols[i]` (`flatKey`). So the class's
`TableClass.Copies` at slot `g` holds for sources that only join inputs with one key at `g`'s copy index
(`flatCopies_of_key`).
-/

open Flock FlockLevel3

namespace FlockSoundness.Discharge.Aliased

/-- **A unit with only leaf inputs**: every stacked input at a port bit has the zero key or a message bit's. -/
theorem srcKey_leaf (c : Circuit) (h1 : c.unit.inGroups.size = 1) (hnd : (portBits c.unit.inGroups).Nodup) {x : ℕ}
    (hm : x % (c.unit.inWords * WORD) ∈ portBits c.unit.inGroups) :
    srcKey c x = .zero ∨ ∃ q byte bit, srcKey c x = .msg q byte bit ∧ bit < 8 := by
  unfold srcKey
  rw [ite_eq_left hm]
  obtain ⟨p, t, hp, ht, hpt⟩ := mem_portBits hm
  rw [hpt, portOf_eq hnd hp ht]
  refine srcAt_leaf c _ _ ?_
  obtain ⟨g, hg⟩ := List.length_eq_one_iff.1 (show c.unit.inGroups.toList.length = 1 by simpa using h1)
  have hgs : c.unit.inGroups = #[g] := Array.toList_inj.1 (by simpa using hg)
  have hsz : (portCols #[g]).size = g.bits.size := by
    simpa using congrArg List.length (portCols_single g)
  rw [hgs] at hp ⊢
  simpa [hsz] using hp

section keys

variable {tags : Tags} {st : Stmt} (L : Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
  (F : UnitFacts tags st)

end keys

/-! ## The typed flat class -/

/-- A flat class's unit input `i` at copy index `k`: the key of the port bit it is (`inCols[i]`). -/
def flatKey (c : Circuit) (inCols : Array ℕ) (k i : ℕ) : Src :=
  srcAt c k (portOf (portCols c.unit.inGroups) (inCols.getD i 0))
    (inCols.getD i 0 - (portCols c.unit.inGroups)[portOf (portCols c.unit.inGroups) (inCols.getD i 0)]!.1)

section flat

variable {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
  {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = true)
  (h : Stmt.setupH tags cf pf coins tables draw partition program = .ok st) {k : Typed.Class} {net : Net}
  (hr : Typed.read tags cf tables = .ok (k, .inl net)) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))

theorem flatCopySrc_key (g : Fin st.c.unitRange.count) (i : ℕ) :
    (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i = .zero ∧
      (flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) = st.c.unitRange.slot g + st.c.unit.useful ∨
        flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) = st.c.unitRange.slot 0 + st.c.unit.useful)) ∨
    (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i ≠ .zero ∧
      flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) =
        realize st.c (g.val / st.c.upv) (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i)) :=
  flatSrc_key st.c g.val _ _

/-- **A flat class's input with the zero key copies a forced-zero row.** -/
theorem flatCopyPos_zero (g : Fin st.c.unitRange.count) {i : ℕ}
    (hz : flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i = .zero) :
    flatCopyPos ht h hr regs g i ∈ Finset.univ.image (flatZeroPos ht h hr regs) := by
  unfold flatCopyPos
  split
  · rcases flatCopySrc_key (k := k) g i with ⟨-, he | he⟩ | ⟨hne, -⟩
    · exact Finset.mem_image.2 ⟨g, Finset.mem_univ _, Fin.ext he.symm⟩
    · exact Finset.mem_image.2 ⟨⟨0, Nat.lt_of_le_of_lt (Nat.zero_le _) g.isLt⟩, Finset.mem_univ _, Fin.ext he.symm⟩
    · exact absurd hz hne
  · exact Finset.mem_image.2 ⟨g, Finset.mem_univ _, rfl⟩

/-- **`TableClass.Copies` at a typed flat class's slot, from keys**: slot `g`'s inputs that read one source of the unit
placed there copy one position, or forced-zero rows, when inputs that read one source have one key at the slot's copy
index. -/
theorem flatCopies_of_key (g : Fin st.c.unitRange.count) (src : ℕ → FlockSoundness.Src)
    (hk : ∀ i < (setupH_flatTableClass ht h hr regs).rows.nIn, ∀ j < (setupH_flatTableClass ht h hr regs).rows.nIn,
      src i = src j → flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i =
        flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) j) :
    (setupH_flatTableClass ht h hr regs).Copies g src := by
  intro i hi j hj hs
  have hkey := hk i hi j hj hs
  show flatCopyPos ht h hr regs g i = flatCopyPos ht h hr regs g j ∨
    flatCopyPos ht h hr regs g i ∈ Finset.univ.image (flatZeroPos ht h hr regs) ∧
      flatCopyPos ht h hr regs g j ∈ Finset.univ.image (flatZeroPos ht h hr regs)
  by_cases hz : flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i = .zero
  · exact Or.inr ⟨flatCopyPos_zero ht h hr regs g hz, flatCopyPos_zero ht h hr regs g (hkey ▸ hz)⟩
  · have e : ∀ i', flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i' ≠ .zero →
        flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i' 0) =
          realize st.c (g.val / st.c.upv) (flatKey st.c (k.done.getD k.unit default).inCols (g.val % st.c.upv) i') :=
      fun i' hne => by
        rcases flatCopySrc_key (k := k) g i' with ⟨hz', -⟩ | ⟨-, he⟩
        · exact absurd hz' hne
        · exact he
    have he : flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD i 0) =
        flatSrcAt st.c g ((k.done.getD k.unit default).inCols.getD j 0) := by
      rw [e i hz, e j (hkey ▸ hz), hkey]
    left
    unfold flatCopyPos
    simp only [he]

end flat

end FlockSoundness.Discharge.Aliased
