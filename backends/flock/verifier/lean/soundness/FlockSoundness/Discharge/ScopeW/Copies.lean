import FlockSoundness.Discharge.ScopeW.Delta
import FlockSoundness.Discharge.HiddenExec.Copies

set_option Elab.async false

/-!
# Δ's copies at the widened scope's rows: frozen statements

The row lemmas read Δ only at a row net's slot's bits, and there every statement the headlines take has the same Δ:
`HmRow.delta`'s pairs, plus pairs (`extra`) only at bits `HmRow.delta` never writes, filtered only at bits outside the
row nets (`RowDelta`). Three instances:

* `rowDelta_flat`: `Stmt.setupH`'s Δ, `HmRow.delta`'s;
* `rowDelta_hidden`: `Stmt.setupHidden`'s, `HmOut.delta`'s (`HmRow.delta`'s and `Hidden.outPairs`, `hidden_delta'`),
  whose pairs at the output rows' message bits `HmRow.delta` never writes (`outPairs_freshW`, `flat_count_msgW`);
* `rowDelta_hmIn`: a statement with the verifier's public inputs, `HmIn.delta`'s, which is `HmOut.delta`'s with no public
  port (`hmIn_delta_empty`) and otherwise filters only public leaves' bits, none of a row net's slot (`hpub`, which
  ZkPubIn's `publicCol_rowNet` gives).

So each copy lemma is stated once, at `RowDelta`, with the twin's circuit facts (`Aliased.UnitFacts` at
`Hidden.flatTwin st`, which reads only the circuit and its pin) as ZkPubIn's are; the plain forms are corollaries
(`twin_of_flat`, `inBit_copyLW_flat`, `row_copies_flatW_flat`).

* `inBit_copyLW`: an input bit of a row net's slot holds its one copy of the rows' copies and wires (`rowCopiesW`);
* `row_copies_flatW`: VU `g`'s row `p`'s copies, for any row, read off `HmRow.chainStart` and `HmRow.msgBit`;
* `row_copies_wholeW`, `row_copies_shortW`, `row_copies_segW`: what they give at each shape;
* `msg_copyW`: an output row's message bit holds its copy, for output rows of every shape (`OutRowsS`).
-/

namespace FlockSoundness.Discharge.ScopeW

open Flock FlockLevel3 Hm Compose Flock.DeriveAll Types.Dag

/-! ## What the row lemmas read of Δ -/

/-- **Δ at the rows**: `HmRow.delta`'s pairs and `extra`'s, filtered by `keep` at their row; `extra` writes no row that
`HmRow.delta` writes, and `keep` keeps every bit of a row net's slot. -/
def RowDelta (st : Stmt) (extra : List (ℕ × ℕ)) (keep : ℕ → Bool) : Prop :=
  st.da = ((HmRow.delta st.c st.pin).1 ++ extra.toArray).filter (fun e => keep e.1) ∧
    (∀ e ∈ extra, ∀ j, (HmRow.delta st.c st.pin).1.toList.count (e.1, j) = 0) ∧
    ∀ {n s off x : ℕ}, (n = st.c.shaNet ∨ n = st.c.hmNet) → AtSlot st.c n s off x → keep x = true

/-- `Stmt.setupH`'s Δ. -/
theorem rowDelta_flat {st : Stmt} (hda : st.da = (HmRow.delta st.c st.pin).1) : RowDelta st [] (fun _ => true) :=
  ⟨by rw [hda, Array.filter_eq_self.2 fun _ _ => rfl]; simp, by simp, fun _ _ => rfl⟩

/-- **`HmIn.delta` with no public port is `HmOut.delta`.** -/
theorem hmIn_delta_empty (c : Circuit) {pi : HmIn.Ports} (pin : ℕ) (h : pi.ports.isEmpty = true) :
    HmIn.delta c pi pin = HmOut.delta c pin := by
  simp [HmIn.delta, h]

/-- An untyped statement whose Δ is `HmRow.delta`'s is its own flat twin. -/
theorem flatTwin_eq {st : Stmt} (L : Layout st) (ht : st.c.typed = none)
    (hda : st.da = (HmRow.delta st.c st.pin).1) : Hidden.flatTwin st = st := by
  have hdb : st.db = (HmRow.delta st.c st.pin).2 := by rw [L.delta_b, hda, delta_snd _ _ ht]
  rw [Hidden.flatTwin, ← hda, ← hdb]

/-- **The plain forms' twin facts**: at a statement whose Δ is `HmRow.delta`'s, its own. -/
theorem twin_of_flat {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
    (hda : st.da = (HmRow.delta st.c st.pin).1) :
    Aliased.UnitFacts tags (Hidden.flatTwin st) ∧ Layout (Hidden.flatTwin st) := by
  rw [flatTwin_eq L F.parse.typed hda]
  exact ⟨F, L⟩

/-- **The output rows `msg_copyW` reads**: every row of a widened shape (`scopeOkW_ports`), each output row's port after
the input rows. -/
def OutRowsS (c : Circuit) : Prop :=
  (∀ p ∈ c.ports.toList, rowShapeW p = true) ∧ ∀ j < c.outRowPorts.size, HmOut.nIn c + j < c.ports.size

section Msg

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
include L F

/-- **`HmRow.delta` writes no row's message bit**, at a row of any widened shape: a row bit of a whole row's compressions,
of a short row's `hm96` slot or of a segment's data blocks. -/
theorem flat_count_msgW {g p b t : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size)
    (hs : rowShapeW st.c.ports[p]! = true) (hb : b < 2 * st.c.ports[p]!.words) (ht : t < 8) (j : ℕ) :
    (flatPairsW st.c st.pin).count (Hidden.deltaMsgCol st.c g p b t, j) = 0 := by
  sorry

/-- **`Hidden.outPairs` writes no row `HmRow.delta` writes.** -/
theorem outPairs_freshW (hrows : OutRowsS st.c) :
    ∀ e ∈ Hidden.outPairs st.c, ∀ j, (HmRow.delta st.c st.pin).1.toList.count (e.1, j) = 0 := by
  sorry

/-- `Stmt.setupHidden`'s Δ (`hidden_delta'`). -/
theorem rowDelta_hidden (hrows : OutRowsS st.c)
    (hda : st.da = (HmRow.delta st.c st.pin).1 ++ (Hidden.outPairs st.c).toArray) :
    RowDelta st (Hidden.outPairs st.c) (fun _ => true) :=
  ⟨by rw [hda, Array.filter_eq_self.2 fun _ _ => rfl], outPairs_freshW L F hrows, fun _ _ => rfl⟩

/-- **A statement with the verifier's public inputs**: `HmIn.delta`'s Δ, given that no row net's slot's bit is a public
column (ZkPubIn's `publicCol_rowNet`). -/
theorem rowDelta_hmIn (hrows : OutRowsS st.c) {pi : HmIn.Ports} (hda : st.da = (HmIn.delta st.c pi st.pin).1)
    (hpub : ∀ {n s off x : ℕ}, (n = st.c.shaNet ∨ n = st.c.hmNet) → AtSlot st.c n s off x →
      HmIn.publicCol st.c pi x = false) :
    RowDelta st (Hidden.outPairs st.c) (fun x => pi.ports.isEmpty || !HmIn.publicCol st.c pi x) := by
  refine ⟨?_, outPairs_freshW L F hrows, fun hn hat => by simp [hpub hn hat]⟩
  rw [hda, HmIn.delta, Hidden.hidden_delta']
  by_cases h : pi.ports.isEmpty
  · simp only [h, ↓reduceIte, Bool.true_or]
    exact (Array.filter_eq_self.2 fun _ _ => rfl).symm
  · simp [h]

end Msg

/-! ## The rows' copies -/

section Rows

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {extra : List (ℕ × ℕ)} {keep : ℕ → Bool} (hΔ : RowDelta st extra keep)
include L F hΔ

/-- **An input bit of a row net's slot holds its one copy**, for rows of any shape. At a self row `x` of a `sha512x3` or
`hm96` slot below the net's input rows, where the rows' copies and wires write exactly one `(x, src)`, every block of a
satisfying witness holds at `x` what it holds at `src`. -/
theorem inBit_copyLW {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopiesW st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) :
    blockBit (L.model regs) z o x = blockBit (L.model regs) z o src := by
  sorry

variable (Lf : Layout (Hidden.flatTwin st))
include Lf

/-- **VU `g`'s row `p`'s copies, for a row of any shape**, in every block of a satisfying witness: a chain start's `cv`
its midstate, every other compression's `cv` (the `hm96` slot's included) the previous one's output, a constant message
bit its constant, a digest bit the segment's last output's. -/
theorem row_copies_flatW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ j, HmRow.chainStart st.c.ports[p]! k = some j → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = wordsBit (startMid st.c.ports[p]! j) t) ∧
    (∀ k, 0 < k → k ≤ st.c.ports[p]!.chunks → HmRow.chainStart st.c.ports[p]! k = none → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p (k - 1) + t)) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ v, HmRow.msgBit st.c.ports[p]! k t = .const v →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = v) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ kk col, HmRow.msgBit st.c.ports[p]! k t = .digest kk col →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p kk + col)) := by
  sorry

/-- **The four copies into a row of whole blocks** (`RowWhole`), as `Hidden.row_copies_tgt`'s for any rows: the first
compression's `cv` the row prefix's midstate, each later one's the previous output, the `hm96` slot's `cv` the last
output, and its `pad` the padding block. -/
theorem row_copies_wholeW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (h : RowWhole st.c.ports[p]!) :
    (∀ t < 512, blockBit (L.model regs) z o (shaBase st.c g p 0 + shaIn st.c (2 * shaSub st.c p 0) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ k, k + 1 < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (L.model regs) z o (shaBase st.c g p (k + 1) + shaIn st.c (2 * shaSub st.c p (k + 1)) + t) =
        blockBit (L.model regs) z o (shaBase st.c g p k + shaOut st.c (shaSub st.c p k) + t)) ∧
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      blockBit (L.model regs) z o
        (shaBase st.c g p (st.c.ports[p]!.chunks - 1) + shaOut st.c (shaSub st.c p (st.c.ports[p]!.chunks - 1)) + t)) ∧
    (∀ t < 1024, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
      wordsBit (HmRow.padWords st.c.ports[p]!.words) t) := by
  sorry

/-- **The copies into a short row** (`RowV2Short`, no compression): its `hm96` slot's `cv` the row prefix's midstate, and
its `pad` past the row's `2 · words` bytes the row's padding. -/
theorem row_copies_shortW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (h : RowV2Short st.c.ports[p]!) :
    (∀ t < 512, blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 0 + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ t < 1024, 2 * st.c.ports[p]!.words ≤ (HmRow.blockBit t).1 →
      blockBit (L.model regs) z o (st.c.slotCol st.c.hmNet g p + hmIn st.c 1 + t) =
        tailBit (HmRow.tailOf st.c.ports[p]!) (2 * st.c.ports[p]!.words) t) := by
  sorry

/-- **The copies into a whole-block segmented row** (`RowSegWhole`): each segment's chain from its midstate through its
data blocks, its last block the padding block; the outer chain from the row prefix's midstate into the `hm96` slot, its
message the segments' digests, each byte copied from its segment's last output, then the padding. -/
theorem row_copies_segW (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) (h : RowSegWhole st.c.ports[p]!) :
    (∀ j < (segsOf st.c.ports[p]!).size, ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p (segFirst st.c.ports[p]! j) + t) =
        wordsBit (HmRow.segMidstate st.c.ports[p]! j) t) ∧
    (∀ j < (segsOf st.c.ports[p]!).size, ∀ i < ((segsOf st.c.ports[p]!)[j]!).1 / 1024, ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p (segFirst st.c.ports[p]! j + i + 1) + t) =
        blockBit (L.model regs) z o (outAt st.c g p (segFirst st.c.ports[p]! j + i) + t)) ∧
    (∀ j < (segsOf st.c.ports[p]!).size, ∀ t < 1024,
      blockBit (L.model regs) z o
          (msgAt st.c g p (segFirst st.c.ports[p]! j + ((segsOf st.c.ports[p]!)[j]!).1 / 1024) + t) =
        wordsBit (HmRow.wordsBE (HmRow.padBlock (128 + ((segsOf st.c.ports[p]!)[j]!).1 / 8)) 16) t) ∧
    (∀ t < 512, blockBit (L.model regs) z o (cvAt st.c g p (segFirst st.c.ports[p]! (segsOf st.c.ports[p]!).size) + t) =
      wordsBit (Sha512.compress Sha512.iv (HmRow.prefixOf st.c.ports[p]!) 0) t) ∧
    (∀ k, segFirst st.c.ports[p]! (segsOf st.c.ports[p]!).size ≤ k → k < st.c.ports[p]!.chunks → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p (k + 1) + t) = blockBit (L.model regs) z o (outAt st.c g p k + t)) ∧
    (∀ e < 64 * (segsOf st.c.ports[p]!).size, ∀ b < 8,
      blockBit (L.model regs) z o (msgAt st.c g p (segFirst st.c.ports[p]! (segsOf st.c.ports[p]!).size + e / 128) +
          (64 * (e % 128 / 8) + 56 - 8 * (e % 128 % 8) + b)) =
        blockBit (L.model regs) z o (outAt st.c g p (segFirst st.c.ports[p]! (e / 64 + 1) - 1) +
          (64 * (e % 64 / 8) + 56 - 8 * (e % 64 % 8) + b))) ∧
    (∀ t < 1024, 64 * (segsOf st.c.ports[p]!).size % 128 ≤ (HmRow.blockBit t).1 →
      blockBit (L.model regs) z o (msgAt st.c g p st.c.ports[p]!.chunks + t) =
        tailBit (HmRow.tailOf st.c.ports[p]!) (64 * (segsOf st.c.ports[p]!).size % 128) t) := by
  sorry

end Rows

section Out

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags (Hidden.flatTwin st))
  {keep : ℕ → Bool} (hΔ : RowDelta st (Hidden.outPairs st.c) keep) (hrows : OutRowsS st.c)
include L F hΔ hrows

/-- **An output row's message bit holds its copy**, for output rows of every widened shape and runs of output ports: at
VU `g`, output row `j`'s word `k`, bit `t`, what the witness holds at `outSrc`. -/
theorem msg_copyW {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {g j k t : ℕ} (hg : g < st.c.g)
    (hj : j < st.c.outRowPorts.size) (hk : k < (st.c.ports.getD (HmOut.nIn st.c + j) default).words) (ht : t < 16) :
    blockBit (L.model regs) z o (Hidden.deltaMsgCol st.c g (HmOut.nIn st.c + j) (2 * k + t / 8) (t % 8)) =
      blockBit (L.model regs) z o (Hidden.outSrc st.c g j k t) := by
  sorry

end Out

/-! ## The plain forms -/

section Plain

variable {tags : Tags} {st : Stmt} (L : Layout st) (F : Aliased.UnitFacts tags st)
  (hda : st.da = (HmRow.delta st.c st.pin).1)
include L F hda

/-- `inBit_copyLW` at a statement whose Δ is `HmRow.delta`'s (`Stmt.setupH`). -/
theorem inBit_copyLW_flat {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))} {z : Model.Witness (L.model regs).m}
    (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl)) {n s off x src : ℕ}
    (hn : n = st.c.shaNet ∨ n = st.c.hmNet) (hat : AtSlot st.c n s off x) {l : Flock.HmNets.Laid} {name : String}
    (hl : Net.ofRows name l.useful l.constPos l.inGroups l.outGroups l.ra l.rb = .ok (st.c.nets.getD n default).2)
    (hoff : off < (st.c.nets.getD n default).2.inWords * Flock.WORD)
    (hself : l.ra.getD off #[] = #[off] ∧ l.rb.getD off #[] = #[off])
    (hf : (rowCopiesW st.c st.pin ++
        wireCopiesOf st.c (HmRow.rowWires st.c.ports st.c.shaNet st.c.hmNet).toList).filter
      (fun e => e.1 = x) = [(x, src)]) :
    blockBit (L.model regs) z o x = blockBit (L.model regs) z o src :=
  inBit_copyLW L (twin_of_flat L F hda).1 (rowDelta_flat hda) hz o hn hat hl hoff hself hf

/-- `row_copies_flatW` at a statement whose Δ is `HmRow.delta`'s (`Stmt.setupH`). -/
theorem row_copies_flatW_flat (hupv : 0 < st.c.upv) {regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))}
    {z : Model.Witness (L.model regs).m} (hz : (L.model regs).Satisfies z) (o : Fin (2 ^ (L.model regs).nbl))
    {g p : ℕ} (hg : g < st.c.g) (hp : p < st.c.ports.size) :
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ j, HmRow.chainStart st.c.ports[p]! k = some j → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = wordsBit (startMid st.c.ports[p]! j) t) ∧
    (∀ k, 0 < k → k ≤ st.c.ports[p]!.chunks → HmRow.chainStart st.c.ports[p]! k = none → ∀ t < 512,
      blockBit (L.model regs) z o (cvAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p (k - 1) + t)) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ v, HmRow.msgBit st.c.ports[p]! k t = .const v →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = v) ∧
    (∀ k ≤ st.c.ports[p]!.chunks, ∀ t < 1024, ∀ kk col, HmRow.msgBit st.c.ports[p]! k t = .digest kk col →
      blockBit (L.model regs) z o (msgAt st.c g p k + t) = blockBit (L.model regs) z o (outAt st.c g p kk + col)) :=
  row_copies_flatW L (twin_of_flat L F hda).1 (rowDelta_flat hda) (twin_of_flat L F hda).2 hupv hz o hg hp

end Plain

end FlockSoundness.Discharge.ScopeW
