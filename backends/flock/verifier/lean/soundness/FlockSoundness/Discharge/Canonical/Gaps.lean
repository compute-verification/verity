import FlockSoundness.Discharge.Canonical.Statements
import FlockProofs.Registered

set_option Elab.async false

/-!
# The three gaps between the canonical headline and every canonical workload

`zk_session_sound_canonical` (`Headline.lean`) is the one theorem. Three kinds of canonical circuit are outside what it
says today. For each, here is the statement it still needs, frozen, with what the verifier and the other lanes must
change.

**Tails (the biggest).** When a tail's last stage returns the outputs (`outNet ≠ unitNet`), `HmRow.check` makes every
`leavesOut` empty, so the headline's program `keyProg` (the unit net's rows, `Placed.circuitRows`) computes no output
and its count of wrong units is vacuous. Two things are missing.

* Nothing makes a tail's outputs a function of its input rows. Main's verifier checks only that each wired input port
  has one source of its width. A wiring cycle among stages and lookups is accepted (under `formatOk` a unit reads no
  wire), and so is a stage row that reads an input row past the net's ports. `Net.ofRows` makes that row a self row, and
  no wire copies into it, so its bit is free. `Computed` is the missing condition on the block's own matrices. It ranks
  every column an output depends on, and each such column is its row's product over pinned columns (the pin, region
  bits), input row bits and columns of lower rank. `OutputsComputedC` is the verifier obligation: after verifier change
  V2, every VU's output columns are `Computed` from its input rows. V2 is a walk of each output's cone over `A₀`, `B₀`
  and Δ that refuses a cycle or a free column; Δ's copies cancel a self row's entry over GF(2), so a wired input reads
  only its source. For a circuit whose unit returns the outputs, `OutputsComputedC` should already hold at main
  (unproved): a unit reads no wire, and `checkOrder` orders its rows. For tails it is false. `outputs_determinedC`
  (proved from `block_determined`) is what `Computed` gives: two satisfying witnesses that agree on a VU's input rows
  agree on its output rows.
* The headline's program must become the VU's. The rows of a VU's output cone, in rank order, compose
  (`Compose.Rows.compose`) into the `Rows` `keyProg` stacks in place of the unit net's (`Layout`, outside `Canonical`).
  A tail's stages combine a VU's units, so the partition's unit becomes the VU. That changes `keyProg`, `Layout.outsOf`
  and the session's `layShC`/`rsShC` (`Session.lean`), and the headline's statement with them.

This gap is the biggest. It needs a verifier change in Lean and Rust, a new program whose unit is the VU, and the
session's pieces re-proved at that program. The other two gaps reuse binding machinery that is already proved.

`scope-g`'s finding (a stage's input row past its ports reads itself) is a case of the first bullet: such a row has no
rank, so V2 refuses any output cone through it. It is fixed in the statement (`OutputsComputedC`), not in `formatOk`.
`formatOk`'s statements read no stage column (outputs are bound to their rows, `OutputsBindingC`, but not to the inputs).
The unit's own case is `formatOk`'s `portReadsB c.unit`.

**Public inputs.** `formatOk` admits them: a public leaf is a zero leaf of `leaves_in`, so `readsRows` sees `.zero`. The
headline's `Inputs` has no `publicInputs`, so `Stmt.setupHidden` runs with `none`, and `HmIn.checkOwn` refuses every
circuit that names public input ports. Three changes are needed. `Inputs` gains `publicInputs`, which the session passes
to `setupHidden`. `Aliased.srcKey` gains a key for a public leaf, the verifier's own word, so that `keyProg` stops
reading the leaf as zero (`Layout`). And `PublicPinnedC` holds: in every satisfying witness, a public leaf's bits are the
verifier's own recomputation of that word, through the `in` regions (`HmIn.regions`) and `checkOwn`. No verifier change
is needed: main's checks are the ones `PublicPinnedC` reads.

**Registered reads.** `Flock.Registered.checkPort` refuses every `v2` row (registered rows are `hm96-sha512/row/v1`,
padded to `Port.fits`' words), so under `formatOk` a statement that names registered reads is refused. The soundness
side is mostly ready: `Registered.Reads` and `registered_weights` take any `HmRows` encoding, and `check_ok` reads
`b ‖ c` without a schema. Three changes are needed.

* Verifier change V3: `checkPort` accepts an unsegmented `v2` row of the registered width in bits, and `rowLeaf` tags
  the leaf with the row's schema. Registration commits `hm96-sha512/row/v2` leaves, so registered values re-register.
* The `--zk` session needs the registered reads in its event, with their openings' term
  (`flock_e2e_drawn_hm96_reads`' `rd.Off` and `qR² / 2^513`, `ZkHidden`).
* `RegisteredRowC` must hold: a VU's read of a registered row holds the registrant's committed bits, or SHA-512
  collides, explicitly. It composes `rows_committedC` with `check_ok`. At main it is vacuous under `formatOk`, since
  `checkPort` refuses the `v2` row; after V3 it says something.

**Proof work.** `block_determined` and `outputs_determinedC` are proved. `parse_outputsComputed` waits for V2, and
`public_pinnedC` and `registered_rowC` are frozen targets: `sorry`. Like `Verifier.lean`, this file is outside the root
import.
-/

namespace FlockSoundness.Discharge.Canonical

open Flock FlockLevel3

/-! ## Tails: outputs computed from the input rows -/

/-- **A block column the statement pins**: the pin, or a bit some region fixes to the statement's value. -/
def Pinned (S : Model.Statement) (x : Fin (2 ^ S.kLog)) : Prop :=
  x = S.pin ∨ ∃ R ∈ S.regions, ∃ i < 2 ^ R.free.length, R.inBlock i = x.val

/-- **Block columns `D` computed from `ins`**: a rank on block columns under which every column of `D` that is neither in
`ins` nor pinned is its row's product (`A₀`, `B₀`) over pinned columns, columns of `ins`, and columns of `D` of lower
rank. A self row outside `ins` reads itself, so it is in no such `D`, and neither is a cycle. -/
def Computed (S : Model.Statement) (ins D : ℕ → Prop) : Prop :=
  ∃ rank : ℕ → ℕ, ∀ x : Fin (2 ^ S.kLog), D x.val → ¬ ins x.val → ¬ Pinned S x →
    ∀ y, (S.A₀ x y ≠ 0 ∨ S.B₀ x y ≠ 0) → Pinned S y ∨ ins y.val ∨ (D y.val ∧ rank y.val < rank x.val)

theorem pinned_eq {S : Model.Statement} {z z' : Model.Witness S.m} (hz : S.Satisfies z) (hz' : S.Satisfies z')
    (o : Fin (2 ^ S.nbl)) {x : Fin (2 ^ S.kLog)} (hx : Pinned S x) : S.block z o x = S.block z' o x := by
  rcases hx with rfl | ⟨R, hR, i, hi, hx⟩
  · rw [hz.2.1 o, hz'.2.1 o]
  · have hb : R.inBlock i < 2 ^ S.kLog := hx ▸ x.isLt
    have e : x = ⟨R.inBlock i, hb⟩ := Fin.ext hx.symm
    rw [e, hz.2.2 R hR o i hi hb, hz'.2.2 R hR o i hi hb]

/-- **Computed columns are determined**: two satisfying witnesses that agree on `ins` in block `o` agree there on every
column of `D` (strong induction on the rank; a pinned column holds the statement's value in both). -/
theorem block_determined {S : Model.Statement} {ins D : ℕ → Prop} (h : Computed S ins D) {z z' : Model.Witness S.m}
    (hz : S.Satisfies z) (hz' : S.Satisfies z') (o : Fin (2 ^ S.nbl))
    (hin : ∀ x : Fin (2 ^ S.kLog), ins x.val → S.block z o x = S.block z' o x) :
    ∀ x : Fin (2 ^ S.kLog), D x.val → S.block z o x = S.block z' o x := by
  obtain ⟨rank, hr⟩ := h
  have key : ∀ r, ∀ x : Fin (2 ^ S.kLog), rank x.val = r → D x.val → S.block z o x = S.block z' o x := by
    intro r
    induction r using Nat.strong_induction_on with
    | _ r ih =>
      intro x hxr hx
      by_cases hi : ins x.val
      · exact hin x hi
      by_cases hp : Pinned S x
      · exact pinned_eq hz hz' o hp
      have hy : ∀ y, S.A₀ x y ≠ 0 ∨ S.B₀ x y ≠ 0 → S.block z o y = S.block z' o y := fun y hy => by
        rcases hr x hx hi hp y hy with h | h | ⟨hD, hlt⟩
        · exact pinned_eq hz hz' o h
        · exact hin y h
        · exact ih _ (hxr ▸ hlt) y rfl hD
      have hmv : ∀ M : Matrix (Fin (2 ^ S.kLog)) (Fin (2 ^ S.kLog)) (ZMod 2), (∀ y, M x y ≠ 0 → S.A₀ x y ≠ 0 ∨
          S.B₀ x y ≠ 0) → M.mulVec (S.block z o) x = M.mulVec (S.block z' o) x := fun M hM => by
        simp only [Matrix.mulVec, dotProduct]
        refine Finset.sum_congr rfl fun y _ => ?_
        by_cases hm : M x y = 0
        · rw [hm, zero_mul, zero_mul]
        · rw [hy y (hM y hm)]
      rw [← hz.1 o x, ← hz'.1 o x, hmv S.A₀ fun y h => Or.inl h, hmv S.B₀ fun y h => Or.inr h]
  exact fun x hx => key _ x rfl hx

/-- **VU `g`'s input rows' data bits**: block column `x` is bit `i < ℓ_p` of input row `p`'s message at VU `g`. -/
def inColsC (c : Circuit) (g x : ℕ) : Prop :=
  ∃ p < HmOut.nIn c, ∃ i < c.ports[p]!.bits, x = Typed.msgCol c g p (i / 8) (i % 8)

/-- **VU `g`'s output bits**: block column `x` is bit `i` of output row `j` at VU `g`, as `outBitsC` reads it. -/
def outColsC (c : Circuit) (g x : ℕ) : Prop :=
  ∃ j < c.outRowPorts.size, ∃ i < 16 * (HmOut.outGroup c j).2.2.2,
    x = HmOut.outCol c g ((HmOut.outGroup c j).2.2.1 + i / 16) + i % 16

/-- **`OutputsComputedC`** (the verifier's obligation after V2): in a drawn statement `Stmt.setupHidden` accepts on a
canonical circuit, every VU's output bits are `Computed` from its input rows' data bits, at every layout and regions. -/
def OutputsComputedC : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    formatOk st.c = true →
    ∀ (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))) {g : ℕ}, g < st.c.g →
    ∃ D : ℕ → Prop, Computed (L.model regs) (inColsC st.c g) D ∧ ∀ x, outColsC st.c g x → D x

/-- **Every VU's outputs are computed from its input rows** (after V2; false at main for tails, whose wiring main does
not order). -/
theorem parse_outputsComputed : OutputsComputedC := by
  sorry

theorem bit_inj {x y : ZMod 2} (h : FlockSoundness.bit x = FlockSoundness.bit y) : x = y := by
  revert x y; decide

/-- **A tail's outputs are its input rows' function**: where VU `g`'s output bits are `Computed` from its input rows,
two satisfying witnesses whose input rows at VU `g` of block `o` hold the same bits compute the same output rows there. -/
theorem outputs_determinedC {S : Model.Statement} {c : Circuit} {g : ℕ} {D : ℕ → Prop}
    (hD : Computed S (inColsC c g) D) (hout : ∀ x, outColsC c g x → D x) {z z' : Model.Witness S.m}
    (hz : S.Satisfies z) (hz' : S.Satisfies z') (o : Fin (2 ^ S.nbl))
    (hin : ∀ p < HmOut.nIn c, rowBits (Hm.blockBit S z o) c g p = rowBits (Hm.blockBit S z' o) c g p) :
    ∀ j < c.outRowPorts.size, outBitsC (Hm.blockBit S z o) c g j = outBitsC (Hm.blockBit S z' o) c g j := by
  have hin' : ∀ x : Fin (2 ^ S.kLog), inColsC c g x.val → S.block z o x = S.block z' o x := by
    rintro x ⟨p, hp, i, hi, hx⟩
    have e := congrArg (fun l => l.getD i false) (hin p hp)
    simp only [rowBits_getD _ c g p hi, ← hx, Hm.blockBit, x.isLt, ↓reduceDIte] at e
    exact bit_inj e
  have hall := block_determined hD hz hz' o hin'
  intro j hj
  simp only [outBitsC]
  congr 1
  funext i
  simp only [Hm.blockBit]
  split
  · rename_i hlt
    rw [hall ⟨_, hlt⟩ (hout _ ⟨j, hj, i.val, i.isLt, rfl⟩)]
  · rfl

/-! ## Public inputs: a public leaf is the verifier's own word -/

/-- **Word `q` of population member `i`'s public words in the verifier's own recomputation** (`--public-inputs`, as
`HmIn.checkOwn` reads it): u16 little-endian after the header line, instance-major, `pw` words per instance. -/
def ownWord (own : ByteArray) (pw i q : ℕ) : ℕ :=
  let body := own.extract ((Flock.findByte own 10).getD 0 + 1) own.size
  (body.get! (2 * (i * pw + q))).toNat + 256 * (body.get! (2 * (i * pw + q) + 1)).toNat

/-- **`PublicPinnedC`**: in a drawn statement `Stmt.setupHidden` accepts with the verifier's own public input words
`own`, on a canonical circuit, in every satisfying witness of its block R1CS at regions containing the executable's, at
block `o` and VU `g`, bit `t` of unit `u`'s leaf `j`, a public leaf naming public word `q`, is bit `t` of word `q` of the
VU's population member in `own`. -/
def PublicPinnedC : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {own : ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program (some own) = .ok st →
    ∀ {pi : HmIn.Ports}, HmIn.parse st.c = .ok pi →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    formatOk st.c = true →
    ∀ (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g u j q t : ℕ}, g < st.c.g → u < st.c.upv →
    g * st.c.upv + u < st.c.unitRange.count → (pi.leaves.getD u #[]).getD j (-1) = (q : ℤ) → t < 16 →
    ∀ (hr : o.val * st.c.g + g < D.proved.size),
    Hm.blockBit (L.model regs) z o (st.c.unitRange.slot (g * st.c.upv + u) +
        (st.c.unit.inGroups.getD 0 default).col * 128 + 16 * j + t) =
      (ownWord own pi.words D.proved[o.val * st.c.g + g] q).testBit t

/-- **A public leaf holds the verifier's own word** (frozen; main's `in` regions and `checkOwn` are what it reads). -/
theorem public_pinnedC : PublicPinnedC := by
  sorry

/-! ## Registered reads: a read is the registrant's row -/

/-- **A registrant's leaf for row `p`**: hm96's tree leaf of the commit string of `P(ℓ_p) ‖ packBits(wv)` under salt
`ws`, as the registration commits a `sha512/row/v2` row (after V3). -/
def regLeafC (c : Circuit) (p : ℕ) (wv : List Bool) (ws : List UInt8) : ByteArray :=
  Flock.Hm96.Default512.treeLeaf
    ⟨(Binding.commitString ((HmRow.rowPrefixV2 c.ports[p]!.bits).data.toList ++ HmRow.packBits wv) ws).toArray⟩

/-- **`RegisteredRowC`** (after V3): in a drawn statement `Stmt.setupHidden` accepts on a canonical circuit, whose
registered reads the verifier's check (`Flock.Registered.check`, against its own registered ports `own`) accepts, in
every satisfying witness at block `o` and VU `g`: where row `p` is registered port `reg`'s row and the VU's unit reads
position `pos`, and the registrant committed bits `wv` (of the row's `ℓ`) under salt `ws` there (its leaf opening
`reg`'s root at `pos` through `wpath`), the VU's own `b ‖ c` opens the root there through some `path`, and through it
the row's bits at the VU's message columns are `wv`, or SHA-512 collides explicitly: in hm96's leaf, or between the
two openings. -/
def RegisteredRowC : Prop :=
  ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)} {d : Lean.Json}
    {partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt},
    tags.typed = false → Stmt.setupHidden tags cf pf coins tables (some d) partition program = .ok st →
    ∀ {D : Draw.UnitDraw}, Draw.UnitDraw.ofJson d = .ok D → (∀ u ∈ D.proved, u < D.population) →
    formatOk st.c = true →
    ∀ {own : Array Flock.Registered.Port},
    Flock.Registered.check st.c st.pub (some own) (Flock.Registered.derived st.partition) = .ok () →
    ∀ {indices : Array ℕ}, Flock.Registered.unitIndices st.pub = .ok indices →
    ∀ (L : FlockSoundness.Layout st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))),
    (∀ r ∈ st.regions.toList, Refine.regionOf st.c.kLog (st.m - st.c.kLog) r ∈ regs) →
    ∀ {z : Model.Witness (L.model regs).m}, (L.model regs).Satisfies z →
    ∀ (o : Fin (2 ^ (L.model regs).nbl)) {g p pos : ℕ}, g < st.c.g → o.val * st.c.g + g < D.proved.size →
    ∀ reg ∈ own, (List.range st.c.ports.size).find? (fun q => st.c.ports[q]!.name == reg.name) = some p →
    reg.reads[indices.getD (o.val * st.c.g + g) 0]? = some (some pos) →
    ∀ (wv : List Bool) (ws : List UInt8) (wpath : ByteArray), wv.length = st.c.ports[p]!.bits →
    Flock.Registered.opensLeaf reg (regLeafC st.c p wv ws) pos wpath = true →
    ∃ path : ByteArray,
      Flock.Registered.opensLeaf reg
        (Flock.Hm96.Default512.treeLeaf (st.pub.digest (o.val * st.c.g + g) p st.c.ports.size)) pos path = true ∧
      (rowBits (Hm.blockBit (L.model regs) z o) st.c g p = wv ∨
        Binding.Collides (Binding.hm96Pair
          ((HmRow.rowPrefixV2 st.c.ports[p]!.bits).data.toList ++
            HmRow.packBits (rowBits (Hm.blockBit (L.model regs) z o) st.c g p))
          (Hm.bytesOf (Hm.blockBit (L.model regs) z o) 192 (Hm.saltCol st.c g p))
          ((HmRow.rowPrefixV2 st.c.ports[p]!.bits).data.toList ++ HmRow.packBits wv) ws) ∨
        Flock.Registered.Collides (Flock.Registered.openPair reg pos
          (Flock.Hm96.Default512.treeLeaf (st.pub.digest (o.val * st.c.g + g) p st.c.ports.size)) path
          (regLeafC st.c p wv ws) wpath))

/-- **A registered read is the registrant's row, or SHA-512 collides** (frozen; after V3, `rows_committedC` and
`Flock.Registered.check_ok`). -/
theorem registered_rowC : RegisteredRowC := by
  sorry

end FlockSoundness.Discharge.Canonical
