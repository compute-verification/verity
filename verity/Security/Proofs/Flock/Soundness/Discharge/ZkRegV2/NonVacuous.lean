import Proofs.Flock.Soundness.Discharge.ZkRegV2.Scope
import Proofs.Flock.Verifier.Registered
import Proofs.Flock.Soundness.ExecSetup

set_option Elab.async false

/-!
# The registered check against `Layout.scopeOk` and `scopeOkV2`

`check_refuses_scopeOk`: at a circuit `Layout.scopeOk` admits, `Registered.check` refuses every non-empty list of
registered ports, because `scopeOk`'s rows are `sha512/row/v1` rows and `Registered.Port.fits` admits only
`sha512/row/v2` bit rows; so a theorem that takes both `scopeOk c` and the registered check's acceptance holds of no
session with a registered port.

`scopeOkV2` and the registered check's port test hold together of the rows C-Flock's composer writes for a registered
statement (`verity_flock.circuit.compose` with `port_bits` each input row's `16 · words` and every output row in
`v2_out_rows`): `v2Rows c`, each row of `c` a `sha512/row/v2` bit row of its `16 · words` bits, unsegmented, its
compressions unchanged. A row of whole SHA-512 blocks has as many either way (the composer's `Port.comps`,
`⌈(bytes + 17) / 128⌉ − 1`, of `2 · words = 128 · k` bytes is `k`), and every registered row of 1,024 or 2,048 bits is
one. `registered_v2_nonvacuous`: at every circuit whose unit meets `scopeOk`'s unit conditions (every circuit
`Layout.scopeOk` admits, `registered_v2_nonvacuous_of_scopeOk`) with rows of whole blocks, `v2Rows c` has the same unit,
`scopeOkV2` admits it, `Layout.scopeOk` refuses it, and every registered port naming one of its rows with that row's
bits passes the registered check's port test (`Registered.check_ok`'s `p < c.ports.size`, the name and `Port.fits`).
`witness`: one such circuit, a unit of one 1,024-bit input row whose one computed row reads two of the row's bits, with
a registered port on that row; its conditions are checked by evaluation. `registered_v2_nonvacuousH` and `witnessH` are
the same at `scopeOkHV2`, the hidden-output statements' scope, which also asks one output port per output row.
-/

namespace FlockSoundness.Discharge.ZkRegV2

open Flock

/-- **`Layout.scopeOk`'s rows are never rows the registered check admits.** -/
theorem not_fits_of_scopeOk {c : Circuit} (h : Layout.scopeOk c = true) (reg : Registered.Port) {q : Flock.Port}
    (hq : q ∈ c.ports.toList) : reg.fits q = false := by
  have hv := (Layout.scopeOk_ports h q hq).1
  simp [Registered.Port.fits, hv]

/-- **At a `Layout.scopeOk` circuit the registered check refuses every non-empty `own`.** -/
theorem check_refuses_scopeOk {c : Circuit} {pub : Public} {own : Array Registered.Port} {d : Bool}
    (hscope : Layout.scopeOk c = true) (hown : 0 < own.size) : Registered.check c pub (some own) d ≠ .ok () := by
  intro h
  obtain ⟨-, _, -, hall⟩ := Registered.check_ok h
  obtain ⟨p, hp, -, hfit, -⟩ := hall own[0] (Array.getElem_mem hown)
  rw [not_fits_of_scopeOk hscope _ (getElem_mem_toList' hp)] at hfit
  exact Bool.false_ne_true hfit

/-- **`scopeOkV2`'s rows are rows the registered check admits**: at a `scopeOkV2` circuit, a registered port whose rows
are `words * wordBits` bits fits each of the circuit's rows of that many bits. -/
theorem fits_of_scopeOkV2 {c : Circuit} (h : scopeOkV2 c = true) (reg : Registered.Port) {q : Flock.Port}
    (hq : q ∈ c.ports.toList) (hb : q.bits = reg.words * reg.wordBits) : reg.fits q = true := by
  obtain ⟨hv, hs, -, -⟩ := scopeOkV2_ports h q hq
  simp [Registered.Port.fits, hv, hs, hb]

/-! ## The composer's `sha512/row/v2` rows -/

/-- A row as C-Flock's composer writes it in a registered statement: a `sha512/row/v2` bit row of its `16 · words` bits,
not segmented, its compressions unchanged. -/
def v2Row (p : Flock.Port) : Flock.Port := { p with bits := 16 * p.words, v2 := true, segs := none }

/-- **`c` with `v2Row` rows**: its unit, nets, layout and outputs unchanged. -/
def v2Rows (c : Circuit) : Circuit := { c with ports := c.ports.map v2Row }

theorem v2Rows_unit (c : Circuit) : (v2Rows c).unit = c.unit := rfl

theorem v2Rows_size (c : Circuit) : (v2Rows c).ports.size = c.ports.size := Array.size_map ..

theorem v2Rows_getElem! (c : Circuit) {p : ℕ} (hp : p < c.ports.size) : (v2Rows c).ports[p]! = v2Row c.ports[p]! := by
  rw [getElem!_pos (v2Rows c).ports p (by rw [v2Rows_size]; exact hp), getElem!_pos c.ports p hp]
  exact Array.getElem_map ..

/-- **`scopeOkV2` admits `v2Rows c`** at a circuit whose unit meets `scopeOk`'s unit conditions and whose rows are whole
SHA-512 blocks. -/
theorem scopeOkV2_v2Rows {c : Circuit} (h1 : c.unit.inGroups.size = 1) (ho : c.outNet = c.unitNet)
    (hu : c.unit.unitLog ≤ 32) (hpr : Layout.portReadsB c.unit = true)
    (hw : ∀ p ∈ c.ports.toList, 2 * p.words = 128 * p.chunks) : scopeOkV2 (v2Rows c) = true := by
  have hO : (v2Rows c).outNet = c.outNet := rfl
  have hN : (v2Rows c).unitNet = c.unitNet := rfl
  unfold scopeOkV2
  rw [v2Rows_unit, hO, hN]
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq]
  refine ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, ?_⟩
  unfold v2Ports
  rw [Array.all_eq_true']
  intro q hq
  obtain ⟨p, hp, rfl⟩ := Array.mem_map.1 hq
  have hwp := hw p (Array.mem_toList_iff.2 hp)
  simp [v2Row, hwp]

/-- **`Layout.scopeOk` refuses `v2Rows c`** when `c` has a row. -/
theorem scopeOk_v2Rows {c : Circuit} (hne : 0 < c.ports.size) : Layout.scopeOk (v2Rows c) = false := by
  rw [Bool.eq_false_iff]
  intro h
  have hq : (v2Rows c).ports[0]! ∈ (v2Rows c).ports.toList :=
    getElem_mem_toList' (by rw [v2Rows_size]; exact hne)
  have hv := (Layout.scopeOk_ports h _ hq).1
  rw [v2Rows_getElem! c hne] at hv
  simp [v2Row] at hv

/-- **A registered port fits each `v2Row` of its bits.** -/
theorem fits_v2Row (reg : Registered.Port) (p : Flock.Port) (hb : 16 * p.words = reg.words * reg.wordBits) :
    reg.fits (v2Row p) = true := by
  simp [Registered.Port.fits, v2Row, hb]

/-- **The v2 registered statements' circuit premises hold together**, at the composer's `sha512/row/v2` rows of every
circuit whose unit meets `scopeOk`'s unit conditions, with rows of whole SHA-512 blocks: `v2Rows c` has `c`'s unit (so
`Built (v2Rows c).unit` is `Built c.unit`), `scopeOkV2` admits it and `Layout.scopeOk` refuses it, and a registered port
`reg` naming row `p` with that row's bits passes the registered check's port test (`Registered.check_ok`'s first three
facts of each port: `p < c.ports.size`, the name and `Port.fits`). -/
theorem registered_v2_nonvacuous {c : Circuit} (h1 : c.unit.inGroups.size = 1) (ho : c.outNet = c.unitNet)
    (hu : c.unit.unitLog ≤ 32) (hpr : Layout.portReadsB c.unit = true)
    (hw : ∀ p ∈ c.ports.toList, 2 * p.words = 128 * p.chunks)
    (reg : Registered.Port) {p : ℕ} (hp : p < c.ports.size) (hname : c.ports[p]!.name = reg.name)
    (hb : 16 * c.ports[p]!.words = reg.words * reg.wordBits) :
    (v2Rows c).unit = c.unit ∧ scopeOkV2 (v2Rows c) = true ∧ Layout.scopeOk (v2Rows c) = false ∧
      p < (v2Rows c).ports.size ∧ (v2Rows c).ports[p]!.name = reg.name ∧ reg.fits (v2Rows c).ports[p]! = true := by
  refine ⟨rfl, scopeOkV2_v2Rows h1 ho hu hpr hw, scopeOk_v2Rows (lt_of_le_of_lt (Nat.zero_le p) hp),
    by rw [v2Rows_size]; exact hp, ?_, ?_⟩
  · rw [v2Rows_getElem! c hp]; exact hname
  · rw [v2Rows_getElem! c hp]; exact fits_v2Row reg _ hb

/-- **`registered_v2_nonvacuous` at every circuit `Layout.scopeOk` admits** with rows of whole SHA-512 blocks: the v1
statements' circuits, with the rows a registered statement commits. -/
theorem registered_v2_nonvacuous_of_scopeOk {c : Circuit} (hscope : Layout.scopeOk c = true)
    (hw : ∀ p ∈ c.ports.toList, 2 * p.words = 128 * p.chunks)
    (reg : Registered.Port) {p : ℕ} (hp : p < c.ports.size) (hname : c.ports[p]!.name = reg.name)
    (hb : 16 * c.ports[p]!.words = reg.words * reg.wordBits) :
    (v2Rows c).unit = c.unit ∧ scopeOkV2 (v2Rows c) = true ∧ Layout.scopeOk (v2Rows c) = false ∧
      p < (v2Rows c).ports.size ∧ (v2Rows c).ports[p]!.name = reg.name ∧ reg.fits (v2Rows c).ports[p]! = true := by
  unfold Layout.scopeOk at hscope
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hscope
  obtain ⟨⟨⟨⟨h1, ho⟩, hu⟩, hpr⟩, -⟩ := hscope
  exact registered_v2_nonvacuous h1 ho hu hpr hw reg hp hname hb

/-! ## A witness -/

/-- One column per row: A's row 1024 is `[0]`, B's is `[1]`, every other row empty. -/
def witnessRows (col : ℕ) : Sparse where
  rows := 2048
  width := 1026
  data := ⟨#[col.toUInt8, 0, 0, 0]⟩
  start := Array.replicate 1025 0
  len := (Array.replicate 1024 0).push 1

/-- A unit of one 1,024-bit input row (input words 0 … 7, columns 0 … 1023), one computed row (column 1024, the AND of
the row's bits 0 and 1), the constant (column 1025) and one output bit. -/
def witnessNet : Net where
  name := "witness"
  useful := 1026
  constPos := 1025
  inGroups := #[(⟨0, 8, #[1024]⟩ : Flock.Group)]
  outGroups := #[(⟨8, 1, #[1]⟩ : Flock.Group)]
  inWords := 8
  a := witnessRows 0
  b := witnessRows 1
  unitLog := 11

/-- A circuit whose unit is `witnessNet` and whose one row, `w`, is a `sha512/row/v1` row of 64 u16 words (1,024 bits)
in one compression. -/
def witnessCircuit : Circuit :=
  { (default : Circuit) with
    name := "witness", nets := #[("unit", witnessNet)], unitNet := 0, outNet := 0,
    ports := #[{ name := "w", words := 64, chunks := 1, nb := 1 }], outPorts := #[("y", 1)], g := 1, upv := 1 }

/-- A registered value `W` read through row `w`: one row of 64 16-bit leaves. -/
def witnessReg : Registered.Port :=
  { name := "w", value := "W", root := ByteArray.empty, leaves := 1, words := 64, wordBits := 16,
    domain := ByteArray.empty, reads := #[some 0] }

theorem witness_conditions :
    witnessCircuit.unit.inGroups.size = 1 ∧ witnessCircuit.outNet = witnessCircuit.unitNet ∧
      witnessCircuit.unit.unitLog ≤ 32 ∧ Layout.portReadsB witnessCircuit.unit = true ∧
      (∀ p ∈ witnessCircuit.ports.toList, 2 * p.words = 128 * p.chunks) ∧ 0 < witnessCircuit.ports.size ∧
      witnessCircuit.ports[0]!.name = witnessReg.name ∧
      16 * witnessCircuit.ports[0]!.words = witnessReg.words * witnessReg.wordBits := by
  decide +kernel

/-- **A circuit with a registered port that the v2 statements' circuit premises admit together**: `v2Rows
witnessCircuit`, whose computed row reads its registered row's bits, is admitted by `scopeOkV2`, refused by
`Layout.scopeOk`, and passes the registered check's port test for `witnessReg`. -/
theorem witness :
    scopeOkV2 (v2Rows witnessCircuit) = true ∧ Layout.scopeOk (v2Rows witnessCircuit) = false ∧
      0 < (v2Rows witnessCircuit).ports.size ∧ (v2Rows witnessCircuit).ports[0]!.name = witnessReg.name ∧
      witnessReg.fits (v2Rows witnessCircuit).ports[0]! = true := by
  obtain ⟨h1, ho, hu, hpr, hw, hp, hname, hb⟩ := witness_conditions
  obtain ⟨-, h2, h3, h4, h5, h6⟩ := registered_v2_nonvacuous h1 ho hu hpr hw witnessReg hp hname hb
  exact ⟨h2, h3, h4, h5, h6⟩

/-! ## Hidden outputs -/

/-- **`scopeOkHV2` admits `v2Rows c`** when `c` also has one output port per output row, as the composer writes a
hidden-output statement (`v2Rows` keeps the output rows' ports). -/
theorem scopeOkHV2_v2Rows {c : Circuit} (h1 : c.unit.inGroups.size = 1) (ho : c.outNet = c.unitNet)
    (hu : c.unit.unitLog ≤ 32) (hpr : Layout.portReadsB c.unit = true)
    (hw : ∀ p ∈ c.ports.toList, 2 * p.words = 128 * p.chunks)
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) : scopeOkHV2 (v2Rows c) = true := by
  unfold scopeOkHV2
  rw [scopeOkV2_v2Rows h1 ho hu hpr hw]
  simp only [Bool.true_and, beq_iff_eq]
  exact hr

/-- **The hidden-output v2 registered statements' circuit premises hold together**: `registered_v2_nonvacuous` with
`scopeOkHV2`, at a circuit with one output port per output row. -/
theorem registered_v2_nonvacuousH {c : Circuit} (h1 : c.unit.inGroups.size = 1) (ho : c.outNet = c.unitNet)
    (hu : c.unit.unitLog ≤ 32) (hpr : Layout.portReadsB c.unit = true)
    (hw : ∀ p ∈ c.ports.toList, 2 * p.words = 128 * p.chunks)
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1)
    (reg : Registered.Port) {p : ℕ} (hp : p < c.ports.size) (hname : c.ports[p]!.name = reg.name)
    (hb : 16 * c.ports[p]!.words = reg.words * reg.wordBits) :
    (v2Rows c).unit = c.unit ∧ scopeOkHV2 (v2Rows c) = true ∧ Layout.scopeOk (v2Rows c) = false ∧
      p < (v2Rows c).ports.size ∧ (v2Rows c).ports[p]!.name = reg.name ∧ reg.fits (v2Rows c).ports[p]! = true := by
  obtain ⟨hU, -, h3⟩ := registered_v2_nonvacuous h1 ho hu hpr hw reg hp hname hb
  exact ⟨hU, scopeOkHV2_v2Rows h1 ho hu hpr hw hr, h3⟩

/-- `witnessCircuit` with its output `y` in an output row of its own, as the composer writes a hidden-output statement:
the row after the input row, of 64 u16 words (its one output word zero-padded to a whole SHA-512 block), one output
port per output row. -/
def witnessCircuitH : Circuit :=
  { witnessCircuit with
    ports := #[{ name := "w", words := 64, chunks := 1, nb := 1 }, { name := "y", words := 64, chunks := 1, nb := 1 }]
    outRowPorts := #[1] }

/-- **A hidden-output circuit with a registered port that the v2 statements' circuit premises admit together**:
`v2Rows witnessCircuitH` is admitted by `scopeOkHV2`, refused by `Layout.scopeOk`, and passes the registered check's
port test for `witnessReg` at its input row. -/
theorem witnessH :
    scopeOkHV2 (v2Rows witnessCircuitH) = true ∧ Layout.scopeOk (v2Rows witnessCircuitH) = false ∧
      0 < (v2Rows witnessCircuitH).ports.size ∧ (v2Rows witnessCircuitH).ports[0]!.name = witnessReg.name ∧
      witnessReg.fits (v2Rows witnessCircuitH).ports[0]! = true := by
  obtain ⟨h1, ho, hu, hpr, -⟩ := witness_conditions
  have hw : ∀ p ∈ witnessCircuitH.ports.toList, 2 * p.words = 128 * p.chunks := by decide +kernel
  have hr : witnessCircuitH.outRowPorts = Array.replicate witnessCircuitH.outPorts.size 1 := by decide +kernel
  obtain ⟨-, h2, h3, h4, h5, h6⟩ := registered_v2_nonvacuousH (c := witnessCircuitH) h1 ho hu hpr hw hr witnessReg
    (p := 0) (by decide +kernel) (by decide +kernel) (by decide +kernel)
  exact ⟨h2, h3, h4, h5, h6⟩

end FlockSoundness.Discharge.ZkRegV2
