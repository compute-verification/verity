import Proofs.Flock.Claims.Recursion.VStarPartsHolo
import Proofs.Flock.Claims.Recursion.VStarPartsSparse

/-!
# V*'s part links, shared across the shapes

Every V* part, at the dense, holo or sparse shape, is a table of one of the units the verifier rebuilds itself: a
level (`rec-open/…`, or R's `rec-open-stacked/…`), a table of residuals (an algebra part at either holo flag, or one
of R's opening parts) or the RecSparse unit. Its link to the inner check has three parts: the staging (the V*
statement its coins are shown on decodes to such a table), the input tie (the copy's input columns hold the messages
it holds and the coins' values) and the output tie (the unit's outputs there give the inner check). This file states
that link once for every kind and every shape, with what is the circuit's own proved.

* **The unit's check is the parser's** (`parseFacts_of_parse`): a V* statement that decodes to an untyped statement
  without public input words has run `HmRow.parse` at every table, which runs each V* unit's check (`RecOpen.check`,
  `RecResiduals.check`, `RecSparse.check`, `RecOpening.check`). So no link carries a unit check.
* **The input tie's values are the columns'** (`openIns_reads`, `inCols_readElems`, `sparseIn_reads_any`): any copy's
  input columns hold, at each port, the values they read there (`openRow`…, `rowsRead`, `vRead`, `sparseW`, `sparseV`,
  and with `T > 1` the slots' types `sparseT`). So no link carries an existential for them.
* **The output tie from `Flock.check`** (`unitOuts_of_check`, at every copy `allOuts_of_check`): where the outer
  relation holds, each copy's output columns hold the unit's outputs at the values its input columns read (`UnitOuts`:
  `LevelOuts`, `ResOuts`, `SparseReadOuts`): `climbAt_of_check`, `partCols_of_check_at`, `partCols_of_check_opening`,
  `sparseOuts_of_check`.
* **A part's links** (`PartReads` at one V* statement, `PartReadsTo` at every V* statement the part's coins are shown
  on, with `PartChecks`' fallback): the staging, and every V* copy's read outputs (`AllOuts`) giving the goal on every
  list of messages agreeing with what the part holds; a goal read off one copy is `partReads_of_copy`.
  `partChecks_of_partReadsTo` and `partLogs_of_partReadsTo` give `PartChecks` and `PartLogs` at any shape. M's algebra
  links are instances (`partReadsTo_of_resReads`).
* **A linked part** (`LinkedPart`: `PartReadsTo` toward its check, and its calls either none where its coins are shown,
  `LogsNil`, or `PartReadsTo` toward them), at any shape. `vstarAccepts_of_linked`, `vstarParts_of_linked` and
  `vstarPartsStaging_of_linked` are `VStarAcceptsR`, `VStarParts` and `VStarPartsStaging` at any shape from every
  part's links, with `PartOpens` through registered climbs (`partOpens_of_climbs`). At `deployRec` over the outer
  record `deployZkRec relation`: `vstarAccepts_deployRec_of_linked`, `vstarParts_deployRec_of_linked` (what
  `ksA_deployed_parts` reads at each of the three shapes) and `vstarPartsStaging_deployRec_of_linked`.
* **The held messages off a copy's columns** (`held_of_colsHold`, `agrees_colsHold`, `agrees_openRow`): where the
  layout reads round `i`'s message at a run of a copy's columns (`copyPs`) that hold bytes `b` (`ColsHold`), the part
  holds `b` there, and every agreeing list of messages has `b` at round `i`. Bits read back as the bytes they spell
  (`bytesOf_bitL`). This is the message half of the input tie, at any layout that reads its messages at the copies'
  input columns.
-/

namespace Flock.Readable

open FlockSoundness FlockSoundness.Discharge.Hm FlockVBridge.OfVStar

/-! ## The units' checks, from the parse -/

/-- **What the parser accepted at a table, without the check**: a statement that decodes to an untyped statement
without public input words parses each table with `HmRow.parse` (`parseFacts_of_tablesOf`), so each table carries
`ParseFacts`, among them each V* unit's own check. -/
theorem parseFacts_of_parse {decode : Claims.Msg → Option Read} {x : Claims.Msg} {p : Parsed}
    (hx : parseBytes decode x = some p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ}
    (htl : t < p.sts.size) : ∃ tags, FlockSoundness.ParseFacts tags p.sts[t].c := by
  unfold parseBytes at hx
  obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.1 hx
  exact parseFacts_of_tablesOf (tablesOf_of_parses (parses_of_parse hr)) ht hpi htl

/-! ## A rec-sparse copy's inputs off its columns, at any shape -/

/-- The slots' types a rec-sparse copy's columns `v` hold on `tau`'s port, 16 bits each. -/
def sparseT (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) : Array ℕ :=
  Array.ofFn fun s : Fin sh.S =>
    FlockVBridge.Family.natOfBits (fun b => v (FlockVBridge.Sparse.tauBase sh + (16 * s + b))) 16

/-- **A rec-sparse copy's input columns hold the values they read**, at any shape: `sparseIn_reads` with the slots'
types read off `tau`'s port where the shape stacks them (`sparseT`). -/
theorem sparseIn_reads_any (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) :
    FlockVBridge.Sparse.SparseIn sh ((sparseW sh v)[·]!) ((sparseV sh v)[·]!) ((sparseT sh v)[·]!) v := by
  have hW : (fun e => (sparseW sh v)[e]!) =
      readElems (rowCol (Flock.RecSparse.messagePorts sh) 0) ((Flock.RecSparse.messagePorts sh).map (·.2)).sum v :=
    funext ofFn_readElems
  have hV : (fun e => (sparseV sh v)[e]!) =
      readElems (FlockVBridge.Sparse.vBaseS sh + ·) (Flock.RecSparse.nv sh) v :=
    funext ofFn_readElems
  refine ⟨fun q hq => ?_, fun q hq => ?_, fun _ s hs b hb => ?_, fun e he => ?_, fun e he => ?_⟩
  · rw [hW]
    exact (wbit_readElems hq).symm
  · rw [hV]
    exact (wbit_readElems hq).symm
  · rw [getElem!_pos (sparseT sh v) s (by simpa [sparseT] using hs)]
    simp [sparseT, FlockVBridge.Family.testBit_natOfBits, hb]
  · exact (ofFn_readElems e).trans (readElems_ge he)
  · exact (ofFn_readElems e).trans (readElems_ge he)

/-! ## The units' outputs at the values their inputs read -/

/-- **A level's outputs at what its inputs read** (`OpenOuts` at `openIns_reads`' values, the hm96-sha512 scheme):
the climb from the row's leaf and the consistency term. -/
def LevelOuts (v : ℕ → Bool) (LANES H : ℕ) : Prop :=
  OpenOuts v Flock.MerkleScheme.hm96Sha512 LANES H (openRow v LANES) (openAcc v LANES H) (openCoefs v LANES H)
    (openRest v LANES H) (openIdx v LANES H) (openSibs v LANES H)

/-- **A table of residuals' outputs at what its inputs read** (`PartCols` at `rowsRead` and `vRead`). -/
def ResOuts (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) (v : ℕ → Bool) : Prop :=
  PartCols S (rowsRead ns nv v) (vRead ns nv v) v (partBase S ns nv)

/-- **The RecSparse unit's outputs at what its inputs read** (`SparseOuts` at `sparseW`, `sparseV`, `sparseT`). -/
def SparseReadOuts (sh : Flock.RecSparse.Shape) (v : ℕ → Bool) : Prop :=
  SparseOuts sh v (sparseW sh v) (sparseV sh v) (sparseT sh v)

/-- **A table of residuals at either holo flag**: `ResPart` with the algebra's part at the flag its META gives, so the
holo shape's algebra parts (`holo = true`) are among them. -/
def ResPartAny (c : Flock.Circuit) (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ) : Prop :=
  (∃ (m k cl j : ℕ) (sh : Flock.Shape) (holo : Bool), RecResiduals.params c.unit.name = some (m, k, cl, j) ∧
    RecResiduals.shape? m k cl = some sh ∧ RecResiduals.holoOf c = .ok holo ∧
    RecResiduals.part { sh with holo } j = .ok (S, ns, nv)) ∨
  ∃ (m cl j : ℕ) (sh : Flock.Shape), RecOpening.params c.unit.name = some (m, cl, j) ∧
    RecOpening.shape? m cl = some sh ∧ RecOpening.part sh j = .ok (S, ns, nv)

theorem resPartAny_of_resPart {c : Flock.Circuit} {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : ResPart c S ns nv) : ResPartAny c S ns nv := by
  rcases h with ⟨m, k, cl, j, sh, hp, hsh, hh, hpart⟩ | h
  · have e : { sh with holo := false } = sh := by
      have h0 := shape?_holo hsh
      cases sh
      simp only at h0
      subst h0
      rfl
    exact Or.inl ⟨m, k, cl, j, sh, false, hp, hsh, hh, e ▸ hpart⟩
  · exact Or.inr h

/-- **The circuit's unit is one of V*'s, its copies' outputs at what their inputs read being `K`**: a level under
either name (`LevelOuts`), a table of residuals (`ResOuts`) or the RecSparse unit (`SparseReadOuts`). -/
def UnitOuts (c : Flock.Circuit) (K : (ℕ → Bool) → Prop) : Prop :=
  (∃ LANES H : ℕ, (Flock.RecOpen.params c.unit.name = some (LANES, H) ∨
    Flock.RecOpen.stackedParams c.unit.name = some (LANES, H)) ∧ K = fun v => LevelOuts v LANES H) ∨
  (∃ (S : Flock.RecOpen.GfS) (ns : List ℕ) (nv : ℕ), ResPartAny c S ns nv ∧ K = ResOuts S ns nv) ∨
  ∃ sh : Flock.RecSparse.Shape, Flock.RecSparse.params c.unit.name = some sh ∧ K = SparseReadOuts sh

/-- **The output tie from `Flock.check`**: at the deployed parser, wherever the outer relation holds at an untyped
statement without public input words, every copy of a V* unit holds its outputs at the values its input columns read
(`climbAt_of_check` at `openIns_reads`, `partCols_of_check_at` or `partCols_of_check_opening` at
`inCols_readElems`, `sparseOuts_of_check` at `sparseIn_reads_any`; the units' checks from the parse). -/
theorem unitOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) {t : ℕ} (htl : t < p.sts.size)
    {K : (ℕ → Bool) → Prop} (hK : UnitOuts p.sts[t].c K) {o g u : ℕ} (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog))
    (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv) : K (copyOf p t htl w o g u) := by
  obtain ⟨_, P⟩ := parseFacts_of_parse hx ht hpi htl
  rcases hK with ⟨LANES, H, hp, rfl⟩ | ⟨S, ns, nv, hpart, rfl⟩ | ⟨sh, hp, rfl⟩
  · exact climbAt_of_check h hx ht hpi htl hp ho hg hu Flock.MerkleScheme.hm96Sha512 (fun _ _ => rfl)
      (openIns_reads _ LANES H)
  · rcases hpart with ⟨m, k, cl, j, sh, holo, hp, hsh, hh, hpart⟩ | ⟨m, cl, j, sh, hp, hsh, hpart⟩
    · exact partCols_of_check_at h hx ht hpi htl P.recResiduals hp hsh hh hpart ho hg hu _ _
        (fun _ he => readElems_ge he) (fun _ he => readElems_ge he) (inCols_readElems ns nv _)
    · exact partCols_of_check_opening h hx ht hpi htl P.recOpening hp hsh hpart ho hg hu _ _
        (fun _ he => readElems_ge he) (fun _ he => readElems_ge he) (inCols_readElems ns nv _)
  · exact sparseOuts_of_check h hx ht hpi htl hp ho hg hu _ _ _ (sparseIn_reads_any sh _)

/-- **Every copy of every V\* unit holds its outputs at what its inputs read**: at each table whose unit is one of
V*'s, with outputs `K` there (`UnitOuts`), every copy of the unit (block `o`, VU `g`, copy `u`). -/
def AllOuts (p : Parsed) (w : Claims.Flock.Wit) : Prop :=
  ∀ (t : ℕ) (ht : t < p.sts.size) (K : (ℕ → Bool) → Prop), UnitOuts p.sts[t].c K → ∀ o g u,
    o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog) → g < p.sts[t].c.g → u < p.sts[t].c.upv → K (copyOf p t ht w o g u)

/-- **The output tie at every copy** (`unitOuts_of_check` at each). -/
theorem allOuts_of_check {decode : Claims.Msg → Option Read} {x : Claims.Msg} {w : Claims.Flock.Wit}
    (h : Claims.Flock.check (parseOf relation decode) x w = true) {p : Parsed} (hx : parseBytes decode x = some p)
    (ht : p.tags.typed = false) (hpi : p.publicInputs = none) : AllOuts p w :=
  fun _ htl _ hK _ _ _ ho hg hu => unitOuts_of_check h hx ht hpi htl hK ho hg hu

end Flock.Readable

namespace Claims.Recursion.VStar

open FlockVBridge.OfVStar
open FlockSoundness.Discharge.Hm (bitL)

/-! ## The held messages off a copy's columns -/

section Held

/-- The number bits spell, the low bit first (`byte`'s fold, in `ℕ`). -/
def bitsNat (bs : List Bool) : ℕ := bs.foldr (fun b a => (if b then 1 else 0) + 2 * a) 0

theorem toNat_byte (bs : List Bool) : (byte bs).toNat = bitsNat bs % 256 := by
  induction bs with
  | nil => rfl
  | cons b bs ih =>
    simp only [byte, bitsNat, List.foldr_cons] at ih ⊢
    rw [UInt8.toNat_add, UInt8.toNat_mul, ih]
    cases b <;> simp

theorem bitsNat_testBits (n : ℕ) : ∀ k, bitsNat ((List.range k).map n.testBit) = n % 2 ^ k := by
  intro k
  induction k generalizing n with
  | zero => simp [bitsNat, Nat.mod_one]
  | succ k ih =>
    rw [List.range_succ_eq_map, List.map_cons, List.map_map]
    have hf : n.testBit ∘ Nat.succ = (n / 2).testBit := funext fun i => Nat.testBit_succ n i
    simp only [bitsNat, List.foldr_cons] at ih ⊢
    rw [hf, ih, Nat.testBit_zero, Nat.pow_succ', Nat.mod_mul]
    by_cases h : n % 2 = 1 <;> simp [h]
    omega

/-- A byte's eight bits, the low bit first, are that byte. -/
theorem byte_testBits (b : UInt8) : byte ((List.range 8).map b.toNat.testBit) = b := by
  apply UInt8.toNat_inj.1
  rw [toNat_byte, bitsNat_testBits]
  have := UInt8.toNat_lt b
  omega

/-- **Bits read back as the bytes they spell**: `bytesOf` at `l`'s bits, the low bit of each byte first (`bitL`), is
`l`. -/
theorem bytesOf_bitL (l : List UInt8) : bytesOf ((List.range (8 * l.length)).map (bitL l)) = l := by
  apply List.ext_getElem
  · simp [bytesOf]
    omega
  · intro j h1 h2
    simp only [bytesOf, List.getElem_map, List.getElem_range]
    have hj : j < l.length := h2
    have e : ((List.map (bitL l) (List.range (8 * l.length))).drop (8 * j)).take 8 =
        (List.range 8).map (l[j]).toNat.testBit := by
      apply List.ext_getElem
      · simp
        omega
      · intro k hk1 hk2
        simp only [List.getElem_take, List.getElem_drop, List.getElem_map, List.getElem_range, bitL]
        have hk : k < 8 := by simpa using hk2
        have e1 : (8 * j + k) / 8 = j := by omega
        have e2 : (8 * j + k) % 8 = k := by omega
        rw [e1, e2, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hj, Option.getD_some]
    rw [e, byte_testBits]

/-- **A witness read at positions holding a byte string's bits is that byte string.** -/
theorem readAt_of_bits {w : Flock.Wit} {ps : List ℕ} {l : List UInt8} (hlen : ps.length = 8 * l.length)
    (h : ∀ q (hq : q < ps.length), w.hid.getD ps[q] false = bitL l q) : readAt w ps = l := by
  unfold readAt
  have e : ps.map (fun p => w.hid.getD p false) = (List.range (8 * l.length)).map (bitL l) :=
    List.ext_getElem (by simp [hlen]) fun q h1 _ => by
      simp only [List.getElem_map, List.getElem_range]
      exact h q (by simpa using h1)
  rw [e, bytesOf_bitL]

/-- The hidden value column `j` of copy `u` of VU `g` in block `o` of table `t` sits at (`copyCol`'s index). -/
def copyIdx (p : Flock.Readable.Parsed) (t : ℕ) (ht : t < p.sts.size) (o g u j : ℕ) : ℕ :=
  (o * 2 ^ p.sts[t].c.kLog + (p.sts[t].c.unitRange.slot (g * p.sts[t].c.upv + u) + j)) * p.sts.size + t

theorem copyOf_eq {p : Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size} {w : Flock.Wit} {o g u j : ℕ}
    (hj : j ≤ p.sts[t].c.unit.constPos) :
    Flock.Readable.copyOf p t ht w o g u j = w.hid.getD (copyIdx p t ht o g u j) false := by
  simp [Flock.Readable.copyOf, Flock.Readable.copyCol, copyIdx, hj]

/-- The hidden values a copy's columns `base`, …, `base + n − 1` sit at, in order. -/
def copyPs (p : Flock.Readable.Parsed) (t : ℕ) (ht : t < p.sts.size) (o g u base n : ℕ) : List ℕ :=
  (List.range n).map fun q => copyIdx p t ht o g u (base + q)

/-- **What a part holds at a run of a copy's columns is what they hold**: where the layout reads round `i`'s message
at the copy's columns from `base` (`copyPs`), within the copy's columns, and those columns hold the bytes `b`
(`ColsHold`), the part holds `b` at round `i`. -/
theorem held_of_colsHold {L : Layout} {x : Msg} {s i : ℕ} {p : Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Flock.Wit} {o g u base : ℕ} {b : ByteArray}
    (hm : L.msgAt x s i = some (copyPs p t ht o g u base (8 * b.size)))
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (Flock.Readable.copyOf p t ht w o g u) base b) :
    L.held x s w i = some b.data.toList := by
  simp only [Layout.held, hm, Option.map_some]
  congr 1
  apply readAt_of_bits
  · simp only [copyPs, List.length_map, List.length_range, Array.length_toList]
    rfl
  · intro q hq
    have hq' : q < 8 * b.size := by simpa [copyPs] using hq
    simp only [copyPs, List.getElem_map, List.getElem_range]
    rw [← copyOf_eq (by omega)]
    exact hb q hq'

/-- **The input tie's messages off a copy's columns**: on every list of messages agreeing with what the part holds,
round `i`'s message is the bytes the copy's columns hold where the layout reads it (`held_of_colsHold`). -/
theorem agrees_colsHold {L : Layout} {x : Msg} {s i : ℕ} {p : Flock.Readable.Parsed} {t : ℕ}
    {ht : t < p.sts.size} {w : Flock.Wit} {ms : List Msg} (hA : Agrees L x s w ms) (hi : i < L.rounds x)
    {o g u base : ℕ} {b : ByteArray} (hm : L.msgAt x s i = some (copyPs p t ht o g u base (8 * b.size)))
    (htop : base + 8 * b.size ≤ p.sts[t].c.unit.constPos + 1)
    (hb : FlockVBridge.ColsHold (Flock.Readable.copyOf p t ht w o g u) base b) : ms.getD i [] = b.data.toList :=
  hA i hi _ (held_of_colsHold hm htop hb)

/-- **A level's row is the held message**: where the layout reads round `i`'s message at a level copy's row columns
(its first `128 LANES`), every agreeing list of messages has, at round `i`, the bytes of the row the copy's climb
hashes (`openRow`, whose bytes `openIns_reads` puts there). -/
theorem agrees_openRow {L : Layout} {x : Msg} {s i : ℕ} {p : Flock.Readable.Parsed} {t : ℕ} {ht : t < p.sts.size}
    {w : Flock.Wit} {ms : List Msg} (hA : Agrees L x s w ms) (hi : i < L.rounds x) {o g u LANES H : ℕ}
    (hm : L.msgAt x s i = some (copyPs p t ht o g u 0 (128 * LANES)))
    (htop : 128 * LANES ≤ p.sts[t].c.unit.constPos + 1) :
    ms.getD i [] = (_root_.Flock.MerkleScheme.rowBytes
      (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES)).data.toList := by
  have hs : 8 * (_root_.Flock.MerkleScheme.rowBytes
      (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES)).size = 128 * LANES := by
    rw [FlockVBridge.size_rowBytes, show (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES).size =
      LANES from FlockVBridge.Family.size_elems _ _ _]
    omega
  exact agrees_colsHold (b := _root_.Flock.MerkleScheme.rowBytes
      (Flock.Readable.openRow (Flock.Readable.copyOf p t ht w o g u) LANES)) hA hi (by rw [hs]; exact hm)
    (by omega) (Flock.Readable.openIns_reads (Flock.Readable.copyOf p t ht w o g u) LANES H).2.2.2.2.2.1

end Held

/-! ## A part's links, read off its copy's columns -/

section Links

variable {Coin Col T : Type}

/-- **A part's links at one V\* statement** (a code fact about V*'s staging, the layout and the goal at part `s`, none
of it the circuit's semantics): at the compiled statement `x` and the V* statement `xO`,
* the staging: `xO` decodes (`decode`) to an untyped statement without public input words;
* the input and output ties: on every witness and every list of inner messages agreeing with what the part holds
  (`Agrees`), the outputs of every copy of every V* unit among its tables, at the values the copy's input columns read
  (`AllOuts`), give the goal `G`.
The units' checks, the input tie's values and the units' semantics are the circuit's (`allOuts_of_check`). A part's
goal may read many copies (a level part's every query); `partReads_of_copy` is the case of one. -/
def PartReads (decode : Msg → Option Flock.Readable.Read) (L : Layout) (s : ℕ) (x xO : Msg)
    (G : Flock.Wit → List Msg → Prop) : Prop :=
  ∃ p, Flock.Readable.parseBytes decode xO = some p ∧ p.tags.typed = false ∧ p.publicInputs = none ∧
    ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms → Flock.Readable.AllOuts p w → G w ms

/-- **A part's links through one copy**: a table with a V* unit (`UnitOuts`) and a copy of it (block `o`, VU `g`, copy
`u`) whose outputs give the goal on every agreeing list of messages. -/
theorem partReads_of_copy {decode : Msg → Option Flock.Readable.Read} {L : Layout} {s : ℕ} {x xO : Msg}
    {G : Flock.Wit → List Msg → Prop} {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes decode xO = some p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none)
    {t : ℕ} (htl : t < p.sts.size) {K : (ℕ → Bool) → Prop} (hK : Flock.Readable.UnitOuts p.sts[t].c K) {o g u : ℕ}
    (ho : o < 2 ^ (p.sts[t].m - p.sts[t].c.kLog)) (hg : g < p.sts[t].c.g) (hu : u < p.sts[t].c.upv)
    (h : ∀ (w : Flock.Wit) (ms : List Msg), Agrees L x s w ms → K (Flock.Readable.copyOf p t htl w o g u) → G w ms) :
    PartReads decode L s x xO G :=
  ⟨p, hx, ht, hpi, fun w ms hA hO => h w ms hA (hO t htl K hK o g u ho hg hu)⟩

/-- **The goal from a part's links at one V\* statement**: at the deployed parser, wherever the outer relation holds at
`xO`, the links there give their goal on every agreeing list of messages (`allOuts_of_check`). -/
theorem goal_of_partReads {DO : Flock.Deploy Coin Col T} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode) {L : Layout} {s : ℕ} {x xO : Msg}
    {w : Flock.Wit} {G : Flock.Wit → List Msg → Prop} (hck : Flock.check DO.parse xO w = true)
    (h : PartReads decode L s x xO G) {ms : List Msg} (hms : Agrees L x s w ms) : G w ms := by
  rw [hDO] at hck
  obtain ⟨p, hx, ht, hpi, hlink⟩ := h
  exact hlink w ms hms (Flock.Readable.allOuts_of_check hck hx ht hpi)

/-- **Part `s`'s links toward a goal** (`PartLinksTo` read off the copy's columns): wherever a V* statement `xO` of a
compiled statement `x` of the shape shows the coins `c` for part `s`, the part's links hold there (`PartReads`), each
output tie giving the goal `G x c` at the witness and the messages, or the fallback (`PartColl`: a collision among the
part's recorded calls, or a tie with the setup's evaluations at `xO`). -/
def PartReadsTo (H : Msg → Msg) (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (G : Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop) (s : ℕ) : Prop :=
  ∀ (x : Msg) (st : Stmt), D.parse x = some st → shape st.form → ∀ (xO : Msg) (c : ℕ → Coin), shows s x xO c →
    PartReads decode L s x xO fun w ms => G x c w ms ∨ PartColl H D L s x xO w

/-- **The goal from a part's links, or the fallback** (`goal_of_links` for `PartReadsTo`). -/
theorem goal_of_partReadsTo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T}
    {decode : Msg → Option Flock.Readable.Read} {L : Layout} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {G : Msg → (ℕ → Coin) → Flock.Wit → List Msg → Prop} {s : ℕ}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartReadsTo H D decode L shape shows G s) (x : Msg) (st : Stmt) (hst : D.parse x = some st)
    (hf : shape st.form) (xO : Msg) (w : Flock.Wit) (hck : Flock.check DO.parse xO w = true) :
    (∀ (c : ℕ → Coin), shows s x xO c → ∀ ms : List Msg, Agrees L x s w ms → G x c w ms) ∨
      PartColl H D L s x xO w := by
  by_cases hfb : PartColl H D L s x xO w
  · exact Or.inr hfb
  · exact Or.inl fun c hc ms hms => (goal_of_partReads hDO hck (h x st hst hf xO c hc) hms).resolve_right hfb

/-- **`PartChecks` from a part's links**, at any shape (`goal_of_partReadsTo` toward `chk s`). -/
theorem partChecks_of_partReadsTo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s) :
    PartChecks H D DO L shape shows chk s :=
  fun x st hst hf xO w hck _ => goal_of_partReadsTo hDO h x st hst hf xO w hck

/-- **`PartLogs` from a part's links for its calls**, at any shape (`goal_of_partReadsTo` toward the calls `logs s`
being among the part's recorded calls). -/
theorem partLogs_of_partReadsTo {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {s : ℕ}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (h : PartReadsTo H D decode L shape shows (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s) :
    PartLogs H D DO L shape shows logs s :=
  fun x st hst hf xO w hck _ => goal_of_partReadsTo hDO h x st hst hf xO w hck

/-- **M's algebra links are a part's links** (`ResReads`, the plain algebra): its table is a table of residuals at
the plain flag (`ResPartAny`), its output tie `ResOuts`'. -/
theorem partReadsTo_of_resReads {H : Msg → Msg} {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {s : ℕ} (h : ResReads D decode L shape shows chk s) :
    PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s := by
  intro x st hst hf xO c hc
  obtain ⟨p, hx, ht, hpi, t, htl, m, k, cl, j, sh, S, ns, nv, -, hp, hsh, hh, hpart, o, g, u, ho, hg, hu, hlink⟩ :=
    h x st hst hf xO c hc
  exact partReads_of_copy hx ht hpi htl (Or.inr (Or.inl ⟨S, ns, nv,
    Flock.Readable.resPartAny_of_resPart (Or.inl ⟨m, k, cl, j, sh, hp, hsh, hh, hpart⟩), rfl⟩)) ho hg hu
    fun w ms hms hout => Or.inl (hlink w ms hms hout)

/-- **Part `s` is linked** (at any shape): its links toward its inner check (`PartReadsTo`), and for its calls either
none where its coins are shown (`LogsNil`: the residuals' and the RecSparse unit's checks make no SHA-512 call) or its
links toward them (`PartReadsTo`: a level's climb, whose evaluations the part records). -/
def LinkedPart (H : Msg → Msg) (D : Deploy Coin) (decode : Msg → Option Flock.Readable.Read) (L : Layout)
    (shape : Form → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg))
    (s : ℕ) : Prop :=
  PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s ∧
    (LogsNil D shape shows logs s ∨
      PartReadsTo H D decode L shape shows (fun x c w ms => ∀ e ∈ logs s x c ms, e ∈ L.calls x s w) s)

/-- **`PartChecks` at every linked part.** -/
theorem partChecks_linked {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s) : ∀ s, PartChecks H D DO L shape shows chk s :=
  fun s => partChecks_of_partReadsTo hDO (hP s).1

/-- **`PartLogs` at every linked part**: nil where its coins are shown (`partLogs_of_shown_nil`), or from its links
for its calls (`partLogs_of_partReadsTo`). -/
theorem partLogs_linked {H : Msg → Msg} {D : Deploy Coin} {DO : Flock.Deploy Coin Col T} {L : Layout}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s) : ∀ s, PartLogs H D DO L shape shows logs s :=
  fun s => (hP s).2.elim partLogs_of_shown_nil (partLogs_of_partReadsTo hDO)

/-- **M's algebra part at any shape is linked** (`ResReads`, its check making no SHA-512 call: `logs s` empty). -/
theorem linkedPart_of_resReads {H : Msg → Msg} {D : Deploy Coin} {decode : Msg → Option Flock.Readable.Read}
    {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {s : ℕ} (h : ResReads D decode L shape shows chk s) (h0 : ∀ x c ms, logs s x c ms = []) :
    LinkedPart H D decode L shape shows chk logs s :=
  ⟨partReadsTo_of_resReads h, Or.inl fun x _ _ _ _ c _ ms => h0 x c ms⟩

end Links

/-! ## V*'s parts at any shape, from every part's links -/

section Bundles

variable {Coin W Col T : Type} [Inhabited Coin]

/-- **`VStarAcceptsR` at any shape from the parts' links**: `vstarAccepts_of_parts` with every part's `PartChecks`
from its links (`partChecks_of_partReadsTo`) and its `PartOpens` through registered climbs (`partOpens_of_climbs`). -/
theorem vstarAccepts_of_linked {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {IV : Verifier Coin} {L : Layout} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hR : ReadsThrough H D IV L.msgs L.calls) (hVL : VstarLen D)
    (hC : ∀ s, PartReadsTo H D decode L shape shows (fun x c _ ms => chk s x c ms) s)
    (hO : ∀ s, HeldOpens D DO L shape shows opening dig s) (hRC : ∀ s, RegClimbs H D DO L shape shows cmt dig s)
    (hS : Staged H D I L shape shows cmt) (hD : Decomp H D IV shape chk) : VStarAcceptsR H D I DO shape :=
  vstarAccepts_of_parts hR hVL (fun s => partChecks_of_partReadsTo hDO (hC s))
    (fun s => partOpens_of_climbs (hO s) (hRC s)) hS hD

/-- **V\*'s parts at any shape, from every part's links** (`VStarParts`, what `codeFacts_of_parts` and
`ksA_deployed_parts` read at each shape): every part linked (`LinkedPart`), its `PartOpens` through registered climbs,
with the staging, the decompositions and the held rounds. -/
theorem vstarParts_of_linked {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W} {DO : Flock.Deploy Coin Col T}
    {L : Layout} {opening : Opening} {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop} {cmt dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)} {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L shape shows opening dig s) (hRC : ∀ s, RegClimbs H D DO L shape shows cmt dig s)
    (hS : Staged H D I L shape shows cmt) (hD : Decomp H D I.V shape chk) (hLD : LogDecomp H D I.V shape logs)
    (hH : Held D I L shape) : VStarParts H D I DO L opening shape :=
  ⟨shows, chk, cmt, logs, partChecks_linked hDO hP, fun s => partOpens_of_climbs (hO s) (hRC s),
    partLogs_linked hDO hP, hS, hD, hLD, hH⟩

/-- **V\*'s parts at any shape with the staging off the run, from every part's links** (`VStarPartsStaging`, what
`KS_of_parts` reads): `vstarParts_of_linked` with the staging's fact about `D.vstar` (`VstarShows`) for `Staged`. -/
theorem vstarPartsStaging_of_linked {H : Msg → Msg} {D : Deploy Coin} {I : Inner Coin W}
    {DO : Flock.Deploy Coin Col T} {L : Layout} {opening : Opening} {decC : Msg → Msg → Coin} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → Coin) → List Msg → List (Msg × Msg)}
    {decode : Msg → Option Flock.Readable.Read}
    (hDO : DO.parse = Flock.Readable.parseOf Flock.Readable.relation decode)
    (hP : ∀ s, LinkedPart H D decode L shape shows chk logs s)
    (hO : ∀ s, HeldOpens D DO L shape shows opening dig s) (hRC : ∀ s, RegClimbs H D DO L shape shows cmt dig s)
    (hV : VstarShows D I L shape decC shows cmt) (hD : Decomp H D I.V shape chk)
    (hLD : LogDecomp H D I.V shape logs) (hH : Held D I L shape) : VStarPartsStaging D I DO H L opening decC shape :=
  ⟨shows, chk, cmt, logs, partChecks_linked hDO hP, fun s => partOpens_of_climbs (hO s) (hRC s),
    partLogs_linked hDO hP, hV, hD, hLD, hH⟩

end Bundles

/-! ## At `deployRec` -/

section Deployed

variable {W : Type} (H : Msg → Msg) (I : Inner FlockDeploy.Coin W) (ivScope : Msg → Bool) (b : RecDeploy.Brings)

/-- **`VStarAcceptsR` at any shape at `deployRec`**, over the outer record `deployZkRec relation`: `deployRec` at the
layout's readers reads through them (`RecDeploy.readsThrough_deployRec`), its V* statements number `nS`
(`RecDeploy.vstarLen_deployRec`), the outer record parses at `parseOf relation readOfO` by definition, and the rest is
`vstarAccepts_of_linked`. -/
theorem vstarAccepts_deployRec_of_linked {L : Layout} (hb : b.rd = RecDeploy.readersOf L) {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {opening : Opening} {cmt dig : Msg → ℕ → Msg → Msg}
    (hC : ∀ s, PartReadsTo H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape shows
      (fun x c _ ms => chk s x c ms) s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L shape shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk) :
    VStarAcceptsR H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) shape :=
  vstarAccepts_of_linked rfl (RecDeploy.readsThrough_deployRec H I.V ivScope b L hb)
    (RecDeploy.vstarLen_deployRec I.V ivScope b) hC hO hRC hS hD

/-- **V\*'s parts at any shape at `deployRec`** (`ksA_deployed_parts`' premise at each of the dense, holo and sparse
shapes, at `relO := relation`): `vstarParts_of_linked` with the outer record `deployZkRec relation`'s parser by
definition. -/
theorem vstarParts_deployRec_of_linked {L : Layout} {opening : Opening} {shape : Form → Prop}
    {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop} {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop}
    {cmt dig : Msg → ℕ → Msg → Msg} {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, LinkedPart H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows cmt dig s)
    (hS : Staged H (RecDeploy.deployRec I.V ivScope b) I L shape shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk)
    (hLD : LogDecomp H (RecDeploy.deployRec I.V ivScope b) I.V shape logs)
    (hH : Held (RecDeploy.deployRec I.V ivScope b) I L shape) :
    VStarParts H (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) L opening
      shape :=
  vstarParts_of_linked rfl hP hO hRC hS hD hLD hH

/-- **V\*'s parts at any shape at `deployRec`, with the staging off the run** (`KS_of_parts`' premise there):
`vstarPartsStaging_of_linked` with the outer record `deployZkRec relation`'s parser by definition. -/
theorem vstarPartsStaging_deployRec_of_linked {L : Layout} {opening : Opening} {decC : Msg → Msg → FlockDeploy.Coin}
    {shape : Form → Prop} {shows : ℕ → Msg → Msg → (ℕ → FlockDeploy.Coin) → Prop}
    {chk : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → Prop} {cmt dig : Msg → ℕ → Msg → Msg}
    {logs : ℕ → Msg → (ℕ → FlockDeploy.Coin) → List Msg → List (Msg × Msg)}
    (hP : ∀ s, LinkedPart H (RecDeploy.deployRec I.V ivScope b) RecDeploy.readOfO L shape shows chk logs s)
    (hO : ∀ s, HeldOpens (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows opening dig s)
    (hRC : ∀ s, RegClimbs H (RecDeploy.deployRec I.V ivScope b) (RecDeploy.deployZkRec Flock.Readable.relation) L
      shape shows cmt dig s)
    (hV : VstarShows (RecDeploy.deployRec I.V ivScope b) I L shape decC shows cmt)
    (hD : Decomp H (RecDeploy.deployRec I.V ivScope b) I.V shape chk)
    (hLD : LogDecomp H (RecDeploy.deployRec I.V ivScope b) I.V shape logs)
    (hH : Held (RecDeploy.deployRec I.V ivScope b) I L shape) :
    VStarPartsStaging (RecDeploy.deployRec I.V ivScope b) I (RecDeploy.deployZkRec Flock.Readable.relation) H L
      opening decC shape :=
  vstarPartsStaging_of_linked rfl hP hO hRC hV hD hLD hH

end Deployed

end Claims.Recursion.VStar
