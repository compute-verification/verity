import Proofs.Flock.Claims.Recursion.VerdictRowEAccR
import Proofs.Flock.Claims.Recursion.SlotsCheck
import Proofs.Flock.Claims.Recursion.VStarPartsSparse
import Proofs.Flock.C3.RegLevels.OutRegs
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret
import Proofs.Flock.Soundness.Discharge.Hidden.Rows
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope

/-!
# A rec-open copy's `acc` and `acc_out` columns are its unit's rows

`accHolds_recorded_layR` reads a piece's running sums off the bytes `bi`, `bo` its copy's columns hold
(`FlockVBridge.ColsHold`) at the verifier's layout, `acc` from column `128 L + 1024 max(1, ⌈H/2⌉)` and `acc_out` from
`Family.outCol L H + 1024`, and asks that they be the rows `layR` commits: `bi` the bytes `accSegs` reads at the unit's
`acc` row, `bo` the bytes `accOutForms` reads at its `acc_out` row. Both are the copy's own columns:

* `colsHold_inRow`: input row `q` of a circuit whose unit reads its input words in order (`rowHidden_eq`) is the copy's
  columns from the row's first input column;
* `colsHold_acc`: at a circuit `RecOpen.check` accepts at `rec-open/l<L>-h<H>`, the `acc` row is three rows in, at the
  copy's column `128 L + 1024 max(1, ⌈H/2⌉)` (`rowsOk`, `leavesOk`, its input group at column 0);
* `colsHold_outRow`, `colsHold_accOut`: output row `jo` of V*'s output layout (`OutsLaid`, one output port per output
  row) is the copy's columns from the output group's first column plus `1024 jo` (`outCol_leaf`), and `acc_out`'s row
  is row 1, from `Family.outCol L H + 1024` (`outCols_of_laid` at `RecOpen.unitLaid`);
* `accBytes_staged`, `accOutBytes_staged`, `accOutBytes_hidden`: `bi` and `bo` at a staged level part's copy
  (`copyAtR`'s, at `p.sts[0]!`), with `accHolds_recorded_layR`'s `hbi`, `hszi`, `hbi'` and `hbo`, `hszo`, `hbo'`.

`accOutBytes_staged` takes the output rows one port each (`c.outRowPorts = replicate 2 1`). On the hidden-output path
`out_rows_v1` gives it (`HmOut.checkRows` makes a `sha512/row/v1` output row a lone padded port), so
`accOutBytes_hidden` doesn't. On the public-output path `HmRow.parse` refuses a run of no ports, which with
`out_row_ports`' sum gives it too, but `parse_rowPorts` exposes only the sum.
-/

namespace Flock.Readable

section Hidden

open FlockSoundness FlockSoundness.Discharge Exec

/-- **A hidden-output table's output rows are one port each**: its circuit passed `HmOut.parse`, so `checkRows`, and
at `sha512/row/v1` rows (`out_rows_v1`) every output row is a lone port. -/
theorem outRowPorts_of_hidden {r : Read} {sts : Array Stmt} (hs : r.tablesOf = some sts) (ht : r.tags.typed = false)
    (hpi : r.publicInputs = none) (hh : r.tags.hiddenOutputs = true) {j : ℕ} (hj : j < sts.size)
    (hv1 : ∀ q ∈ sts[j].c.ports.toList, q.v2 = false) :
    sts[j].c.outRowPorts = Array.replicate sts[j].c.outPorts.size 1 := by
  obtain ⟨spec, hset⟩ := setup_of_tablesOf hs
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt r.tags r.I.count j).typed = false := (Exec.tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [Flock.Readable.tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  obtain ⟨hpa, hcr⟩ := Hidden.parse_untyped ht' (Hidden.setupHidden_spec' hd).parse
  exact (Hidden.out_rows_v1 hpa hcr hv1).2.1

end Hidden

/-- `outRowPorts_of_hidden` at a statement the parser read. -/
theorem outRowPorts_hidden {decode : Claims.Msg → Option Read} {x : Claims.Msg} {p : Parsed}
    (hx : parseBytes decode x = some p) (ht : p.tags.typed = false) (hpi : p.publicInputs = none)
    (hh : p.tags.hiddenOutputs = true) {t : ℕ} (htl : t < p.sts.size)
    (hv1 : ∀ q ∈ p.sts[t].c.ports.toList, q.v2 = false) :
    p.sts[t].c.outRowPorts = Array.replicate p.sts[t].c.outPorts.size 1 := by
  unfold parseBytes at hx
  obtain ⟨r, -, hr⟩ := Option.bind_eq_some_iff.1 hx
  exact outRowPorts_of_hidden (tablesOf_of_parses (parses_of_parse hr)) ht hpi hh htl hv1

end Flock.Readable

namespace RecDeploy

open Flock
open Claims (Msg)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Hm (bitL)
open FlockVBridge (ColsHold)

/-! ## Bytes and columns -/

theorem extract_two (ps : Array Port) (h : 2 ≤ ps.size) :
    (ps.extract 0 2).foldl (· + ·.words) 0 = (ps[0]'(by omega)).words + (ps[1]'(by omega)).words := by
  rw [extract_words Port.words]
  rcases ps with ⟨_ | ⟨a, _ | ⟨b, l⟩⟩⟩
  · simp at h
  · simp at h
  · simp

theorem length_bytesOf (bs : List Bool) : (Claims.Recursion.VStar.bytesOf bs).length = (bs.length + 7) / 8 := by
  simp [Claims.Recursion.VStar.bytesOf]

/-- **Columns that hold a bit string hold its bytes**: where columns `base`, … hold the bits `bs`, a whole number of
bytes, they hold `bytesOf bs`. -/
theorem colsHold_bytesOf {v : ℕ → Bool} {base : ℕ} {bs : List Bool} (h8 : bs.length % 8 = 0)
    (h : ∀ t (ht : t < bs.length), v (base + t) = bs[t]) :
    ColsHold v base ⟨(Claims.Recursion.VStar.bytesOf bs).toArray⟩ := by
  unfold FlockVBridge.ColsHold
  intro t ht
  have hs : (⟨(Claims.Recursion.VStar.bytesOf bs).toArray⟩ : ByteArray).size = (bs.length + 7) / 8 := by
    show (Claims.Recursion.VStar.bytesOf bs).toArray.size = _
    rw [List.size_toArray, length_bytesOf]
  rw [hs] at ht
  have ht' : t < bs.length := by omega
  show v (base + t) = bitL (Claims.Recursion.VStar.bytesOf bs) t
  rw [bitL_bytesOf ht']
  exact h t ht'

/-! ## An input row -/

/-- **A copy's columns hold its unit's input row**: where the unit reads its input words in order (`hl`, so
`rowHidden_eq`), the bytes `accSegs` reads at row `q`'s instance `n` are what copy `n` (`n / g`, `n % g`) holds from
the row's first input column. -/
theorem colsHold_inRow {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) (w : Claims.Flock.Wit) {N q : ℕ} {nm : String} (n : ℕ)
    (hl : c.leavesIn.getD 0 #[] = (Array.range N).map Int.ofNat) (hq : c.ports.findIdx? (·.name == nm) = some q)
    (hqs : q < c.ports.size) (hbw : c.ports[q].bits = 16 * c.ports[q].words)
    (hN : 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + c.ports[q].bits ≤ 16 * N)
    (hC : (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * N ≤ c.unit.constPos) {base : ℕ}
    (hbase : (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 =
      base) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) base
        ⟨(readSegs w (accSegs c (nm, n))).toArray⟩ ∧
      (readSegs w (accSegs c (nm, n))).length = 2 * c.ports[q].words := by
  subst hbase
  have hR : readSegs w (accSegs c (nm, n)) = Claims.Recursion.VStar.bytesOf
      ((((List.range c.ports[q].bits).map fun i => n / c.g * 2 ^ c.kLog +
        (c.unitRange.slot (n % c.g * c.upv) + ((c.unit.inGroups.getD 0 default).col * Flock.WORD +
          16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 + i))).map HmNets.Lin.var).map (formBit w)) := by
    rw [readSegs_accSegs]
    show readForms w (((rowHidden c nm n).getD []).map HmNets.Lin.var) = _
    rw [rowHidden_eq n hl hq hqs hN, Option.getD_some]
    rfl
  rw [hR]
  refine ⟨colsHold_bytesOf (by simp only [List.length_map, List.length_range, hbw]; omega) fun t ht => ?_, by
    rw [length_bytesOf]; simp only [List.length_map, List.length_range, hbw]; omega⟩
  have ht' : t < c.ports[q].bits := by simpa only [List.length_map, List.length_range] using ht
  have hj : (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 q).foldl (· + ·.words) 0 +
      t ≤ p.sts[0].c.unit.constPos := by
    rw [hc0]; omega
  simp only [List.getElem_map, List.getElem_range, formBit_var]
  rw [Claims.Recursion.VStar.copyOf_eq hj]
  simp only [Claims.Recursion.VStar.copyIdx, hc0, h1, Nat.mul_one, Nat.add_zero]

/-- **A rec-open copy's `acc` columns hold its unit's `acc` row**: at a circuit `RecOpen.check` accepts at
`rec-open/l<L>-h<H>`, its rows are `row`, `aux`, `acc`, … (`rowsOk`), its unit's input group is at column 0 and reads
its input words in order (`leavesOk`), so its `acc` row of 64 words is the copy's columns from
`128 L + 1024 max(1, ⌈H/2⌉)`. -/
theorem colsHold_acc {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) {tags : Flock.Tags} (P : FlockSoundness.ParseFacts tags c) {LANES HH : ℕ}
    (hn : c.unit.name = Flock.RecOpen.unitName LANES HH) (w : Claims.Flock.Wit) (n : ℕ) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) (128 * LANES + 1024 * max 1 ((HH + 1) / 2))
        ⟨(readSegs w (accSegs c ("acc", n))).toArray⟩ ∧
      (readSegs w (accSegs c ("acc", n))).length = 128 := by
  have hrec := P.recNet
  have hp := FlockVBridge.params_of_check hrec hn
  obtain ⟨hrows, hlv⟩ := FlockC3.RegLevels.check_rows hrec (.inl hp)
  obtain ⟨hq0, hn0, hw0, -⟩ := FlockC3.RegLevels.rowsOk_get hrows (q := 0) (by omega)
  obtain ⟨hq1, hn1, hw1, -⟩ := FlockC3.RegLevels.rowsOk_get hrows (q := 1) (by omega)
  obtain ⟨hq2, hn2, hw2, hb2⟩ := FlockC3.RegLevels.rowsOk_get hrows (q := 2) (by omega)
  rw [show ((Flock.RecOpen.rows LANES HH)[0]!).1 = "row" from rfl] at hn0
  rw [show ((Flock.RecOpen.rows LANES HH)[1]!).1 = "aux" from rfl] at hn1
  rw [show ((Flock.RecOpen.rows LANES HH)[2]!).1 = "acc" from rfl] at hn2
  rw [show (Flock.RecOpen.ports LANES HH).bits[0]! = 128 * LANES from rfl] at hw0
  rw [show (Flock.RecOpen.ports LANES HH).bits[1]! = 1024 * max 1 ((HH + 1) / 2) from rfl] at hw1
  rw [show (Flock.RecOpen.ports LANES HH).bits[2]! = 1024 from rfl] at hw2 hb2
  have hqa : c.ports.findIdx? (·.name == "acc") = some 2 := by
    rw [Array.findIdx?_eq_some_iff_getElem]
    refine ⟨hq2, by simp [hn2], fun j hj => ?_⟩
    rcases (by omega : j = 0 ∨ j = 1) with rfl | rfl
    · simp [hn0]
    · simp [hn1]
  have hx2 : (c.ports.extract 0 2).foldl (· + ·.words) 0 = c.ports[0].words + c.ports[1].words :=
    extract_two c.ports (by omega)
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  have hl0 := FlockC3.RegLevels.leavesOk_get hlv (j := 0) (C.leaves.1 ▸ C.upv_pos)
  have hU : (Flock.RecOpen.ports LANES HH).bits.foldl (· + ·) 0 =
      128 * LANES + 1024 * max 1 ((HH + 1) / 2) + 1024 + 256 * LANES + 1024 := FlockVBridge.unitIn_eq LANES HH
  have hcol : (c.unit.inGroups.getD 0 default).col = 0 := by
    rw [FlockC3.RegLevels.inGroups_of_laid (Flock.RecOpen.check_ok hrec hp).2]
    rfl
  have hC := leaves_le_constPos P hl0 (by rw [hU]; generalize max 1 ((HH + 1) / 2) = M; omega)
  have hN : 16 * (c.ports.extract 0 2).foldl (· + ·.words) 0 + c.ports[2].bits ≤
      16 * ((Flock.RecOpen.ports LANES HH).bits.foldl (· + ·) 0 / 16) := by
    rw [hx2, hb2, hU]
    generalize max 1 ((HH + 1) / 2) = M at hw1 ⊢
    omega
  have hbase : (c.unit.inGroups.getD 0 default).col * Flock.WORD + 16 * (c.ports.extract 0 2).foldl (· + ·.words) 0 =
      128 * LANES + 1024 * max 1 ((HH + 1) / 2) := by
    rw [hcol, hx2]
    generalize max 1 ((HH + 1) / 2) = M at hw1 ⊢
    omega
  obtain ⟨hcs, hlen⟩ := colsHold_inRow h1 h0 hc0 w n hl0 hqa hq2 (by omega) hN hC hbase
  exact ⟨hcs, by rw [hlen]; omega⟩

/-! ## An output row -/

/-- **The unit's returned output `j` is at its output group's column `16 j`** (`outCol_leaf` at copy 0), wherever the
unit returns the outputs. -/
theorem outCol_unit {tags : Flock.Tags} {c : Circuit} (P : FlockSoundness.ParseFacts tags c)
    (ho : c.outNet = c.unitNet) (g : ℕ) {j : ℕ} (hj : j < (c.leavesOut.getD 0 #[]).size) :
    HmOut.outCol c g (c.leavesOut.getD 0 #[])[j] =
      c.unitRange.slot (g * c.upv) + ((c.unit.outGroups.getD 0 default).col * Flock.WORD + 16 * j) := by
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  obtain ⟨h16, -, hlo⟩ := FlockSoundness.Discharge.Layout.check_out P.typed ho hls
  have hget : ∀ u < c.upv, c.leavesOut.getD u #[] = c.leavesOut[u]! := fun u hu => by
    rw [FlockSoundness.Discharge.Hidden.getElem!_eq_getD']; rfl
  have hupv : 0 < c.upv := FlockSoundness.Discharge.Hm.upv_pos C ho
  obtain ⟨hur, hk, hcount⟩ := FlockSoundness.unit_range_facts P C
  have hpv := FlockSoundness.perVu_unit C hur hk hcount
    (FlockSoundness.Discharge.Layout.check_cuts P.typed hls).2
  have h := FlockSoundness.Discharge.HiddenExec.outCol_leaf P.typed ho
    (FlockSoundness.Discharge.HiddenExec.check_returned P.typed ho hls)
    (fun u hu o' ho' => by rw [hget u hu] at ho'; exact (hlo u hu).2 o' ho')
    h16 (fun u hu => by rw [hget u hu]; exact (hlo u hu).1) hpv g hupv hj
  rwa [Nat.add_zero] at h

/-- **A copy's columns hold its unit's output row `jo`**: at V*'s output layout (`OutsLaid`), one output port per
output row, the bytes `accOutForms` reads at output row `jo < 2`'s instance `n` are what copy `n` holds from the output
group's first column `B` plus `1024 jo`. -/
theorem colsHold_outRow {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) {tags : Flock.Tags} (P : FlockSoundness.ParseFacts tags c) (ho : c.outNet = c.unitNet)
    (hlaid : FlockC3.RegLevels.OutsLaid c 0) (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) {B : ℕ}
    (hB : (c.unit.outGroups.getD 0 default).col * Flock.WORD = B) (hE : B + 2048 ≤ c.unit.constPos)
    (w : Claims.Flock.Wit) (n : ℕ) {jo : ℕ} (hjo : jo < 2) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) (B + 1024 * jo)
        ⟨(readForms w (accOutForms c (HmOut.nIn c + jo) n)).toArray⟩ ∧
      (readForms w (accOutForms c (HmOut.nIn c + jo) n)).length = 128 := by
  obtain ⟨hop, hlo, hq, -, hw0, -, hw1⟩ := hlaid
  have hlt : HmOut.nIn c + jo < c.ports.size := by omega
  have hwj : (c.ports.getD (HmOut.nIn c + jo) default).words = 64 := by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hlt, Option.getD_some]
    rcases (by omega : jo = 0 ∨ jo = 1) with rfl | rfl
    · exact hw0
    · exact hw1
  have hjo' : jo < c.outPorts.size := by rw [hop]; exact hjo
  have hgrp : HmOut.outGroup c jo = (jo, 1, 64 * jo, 64) := by
    rw [FlockSoundness.Discharge.Hidden.outGroup_single hr hjo']
    unfold FlockSoundness.Discharge.Hidden.outAt
    rw [hop]
    rcases (by omega : jo = 0 ∨ jo = 1) with rfl | rfl <;> rfl
  have hlen : (accOutForms c (HmOut.nIn c + jo) n).length = 1024 := by
    simp only [accOutForms, List.length_map, List.length_range, hwj]
  have hf : ∀ t (ht : t < (accOutForms c (HmOut.nIn c + jo) n).length),
      (accOutForms c (HmOut.nIn c + jo) n)[t] = HmNets.Lin.var (n / c.g * 2 ^ c.kLog +
        (HmOut.outCol c (n % c.g) (64 * jo + t / 16) + t % 16)) := fun t ht => by
    rw [hlen] at ht
    simp only [accOutForms, List.getElem_map, List.getElem_range, Nat.add_sub_cancel_left, hgrp]
    simp only [show t / 16 < 64 by omega, ↓reduceIte]
  have hcol : ∀ k < 128, HmOut.outCol c (n % c.g) k = c.unitRange.slot (n % c.g * c.upv) + (B + 16 * k) := by
    intro k hk
    have h := outCol_unit P ho (n % c.g) (j := k) (by rw [hlo, Array.size_range]; exact hk)
    simp only [hlo, Array.getElem_range] at h
    rw [h, hB]
  have hR : readForms w (accOutForms c (HmOut.nIn c + jo) n) =
      Claims.Recursion.VStar.bytesOf ((accOutForms c (HmOut.nIn c + jo) n).map (formBit w)) := rfl
  rw [hR]
  refine ⟨colsHold_bytesOf (by rw [List.length_map, hlen]) fun t ht => ?_, by
    rw [length_bytesOf, List.length_map, hlen]⟩
  have ht' : t < 1024 := by simpa only [List.length_map, hlen] using ht
  have hj : B + 1024 * jo + t ≤ p.sts[0].c.unit.constPos := by rw [hc0]; omega
  have e : ∀ S : ℕ, S + (B + 16 * (64 * jo + t / 16)) + t % 16 = S + (B + 1024 * jo + t) := fun S => by omega
  rw [List.getElem_map, hf t (by rw [hlen]; exact ht'), formBit_var, hcol (64 * jo + t / 16) (by omega), e,
    Claims.Recursion.VStar.copyOf_eq hj]
  simp only [Claims.Recursion.VStar.copyIdx, hc0, h1, Nat.mul_one, Nat.add_zero]

/-- **A rec-open copy's `acc_out` columns hold its unit's `acc_out` row**: at a circuit `RecOpen.check` accepts at
`rec-open/l<L>-h<H>`, of V*'s seven rows, one output port per output row, `acc_out` is output row 1 (its rows'
names, `rowsOk` and `OutsLaid`), at the copy's columns from `Family.outCol L H + 1024` (`outCols_of_laid` at
`RecOpen.unitLaid`, which also bounds the outputs within the computed columns). -/
theorem colsHold_accOut {p : Flock.Readable.Parsed} (h1 : p.sts.size = 1) (h0 : 0 < p.sts.size) {c : Circuit}
    (hc0 : p.sts[0].c = c) {tags : Flock.Tags} (P : FlockSoundness.ParseFacts tags c) (ho : c.outNet = c.unitNet)
    {LANES HH : ℕ} (hn : c.unit.name = Flock.RecOpen.unitName LANES HH) (hsz : c.ports.size = 7)
    (hr : c.outRowPorts = Array.replicate c.outPorts.size 1) {qo : ℕ}
    (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo) (w : Claims.Flock.Wit) (n : ℕ) :
    ColsHold (Flock.Readable.copyOf p 0 h0 w (n / c.g) (n % c.g) 0) (FlockVBridge.Family.outCol LANES HH + 1024)
        ⟨(readForms w (accOutForms c qo n)).toArray⟩ ∧ (readForms w (accOutForms c qo n)).length = 128 := by
  have hrec := P.recNet
  have hp := FlockVBridge.params_of_check hrec hn
  obtain ⟨ls, hls⟩ := P.check
  have C := FlockSoundness.check_facts P.typed hls
  have hlo0 : 0 < c.leavesOut.size := by
    rw [(FlockSoundness.Discharge.Layout.check_out P.typed ho hls).2.1]
    exact FlockSoundness.Discharge.Hm.upv_pos C ho
  have hlaid := FlockC3.RegLevels.outsLaid_of_outsOk (Flock.RecOpen.outsOk_of_check hrec hp) hlo0
  have hU := FlockSoundness.unit_built P C
  obtain ⟨hB, -, hE⟩ := FlockC3.RegLevels.outCols_of_laid (hU := hU) (Flock.RecOpen.check_ok hrec hp).2
  have hB' : (c.unit.outGroups.getD 0 default).col * Flock.WORD = FlockVBridge.Family.outCol LANES HH := hB
  rw [FlockVBridge.foldl_wordPorts, show (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp =
    c.unit.constPos from FlockSoundness.Rows.ofNet_one hU] at hE
  have hE' : FlockVBridge.Family.outCol LANES HH + 2048 ≤ c.unit.constPos := by omega
  obtain ⟨hrows, -⟩ := FlockC3.RegLevels.check_rows hrec (.inl hp)
  have gd : ∀ k (hk : k < c.ports.size), c.ports.getD k default = c.ports[k] := fun k hk => by
    rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hk, Option.getD_some]
  have hnm : ∀ k < 5, (c.ports.getD k default).name ≠ "out" ∧ (c.ports.getD k default).name ≠ "acc_out" := by
    intro k hk
    obtain ⟨hk', hnk, -, -⟩ := FlockC3.RegLevels.rowsOk_get hrows hk
    rw [gd k hk', hnk]
    rcases (by omega : k = 0 ∨ k = 1 ∨ k = 2 ∨ k = 3 ∨ k = 4) with rfl | rfl | rfl | rfl | rfl
    · rw [show ((Flock.RecOpen.rows LANES HH)[0]!).1 = "row" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[1]!).1 = "aux" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[2]!).1 = "acc" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[3]!).1 = "coef" from rfl]; decide
    · rw [show ((Flock.RecOpen.rows LANES HH)[4]!).1 = "dirs" from rfl]; decide
  obtain ⟨-, -, hq, hno, -, hna, -⟩ := id hlaid
  have hn5 : HmOut.nIn c = 5 := by
    rcases Nat.lt_or_ge (HmOut.nIn c) 5 with h | h
    · have e : (c.ports.getD (HmOut.nIn c) default).name = "out" := by
        rw [gd (HmOut.nIn c) (by omega)]; exact hno
      exact absurd e (hnm _ h).1
    · omega
  have hq6 : qo = HmOut.nIn c + 1 := by
    obtain ⟨hqs, hqn, hlt⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqo
    have hqn' : (c.ports.getD qo default).name = "acc_out" := by rw [gd qo hqs]; simpa using hqn
    have hle : qo ≤ HmOut.nIn c + 1 := by
      by_contra hgt
      exact hlt (HmOut.nIn c + 1) (by omega) (by simp [hna])
    rcases Nat.lt_or_ge qo 5 with h | h
    · exact absurd hqn' (hnm qo h).2
    · have hne : qo ≠ HmOut.nIn c := fun e => by
        rw [e, gd (HmOut.nIn c) (by omega), hno] at hqn'
        exact absurd hqn' (by decide)
      omega
  subst hq6
  have := colsHold_outRow h1 h0 hc0 P ho hlaid hr hB' hE' w n (jo := 1) (by omega)
  rwa [Nat.mul_one] at this

/-! ## At a staged level part -/

/-- **What a staged level part's statement gives its one table**: where part `s` is V*'s level statement
`.recOpen lv L H n` and stages, its statement has one table, whose circuit is the part's (`partCircuit_staged`),
accepted by `HmRow.parse` and named `rec-open/l<L>-h<H>` (`stagedTables`), within the headlines' scope (`staged_scope`:
it returns its outputs, every row a `sha512/row/v1` row), with V*'s seven row ports (`checkParsed_ok`). -/
theorem stagedRecOpen {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : Flock.RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c) :
    p.sts.size = 1 ∧ p.sts[0].c = c ∧ (∃ tags, FlockSoundness.ParseFacts tags c) ∧
      c.unit.name = Flock.RecOpen.unitName LANES HH ∧ c.outNet = c.unitNet ∧ c.ports.size = 7 ∧
      (∀ q ∈ c.ports.toList, q.v2 = false) ∧ p.tags.typed = false ∧ p.publicInputs = none := by
  obtain ⟨p', r', sv, hx', ht, hpi, hr', ⟨ss', hss', hsv'⟩, -, htab⟩ := stagedTables hS
  have e1 : p = p' := Option.some.inj (hx.symm.trans hx')
  subst e1
  have e2 : r = r' := Option.some.inj (hr.symm.trans hr')
  subst e2
  have e3 : ss = ss' := Except.ok.inj (hss.symm.trans hss')
  subst e3
  have e4 : VStar.Stmt.recOpen lv LANES HH n = sv := Option.some.inj (hsv.symm.trans hsv')
  subst e4
  have hc0 := partCircuit_staged hS hx ht hpi hr hc h0
  have h1 : p.sts.size = 1 := by
    obtain ⟨-, i, spec, sts, -, hsu, -⟩ := id hS
    have e : p = { i.base.read with sts } := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
    rw [e]
    exact (oneFull_staged hS hsu).1
  obtain ⟨tags, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  rw [hc0] at P
  have hn : c.unit.name = Flock.RecOpen.unitName LANES HH := by rw [← hc0]; exact (htab 0 h0).1
  have hsc := (staged_scope hS hx h0).1
  rw [hc0, FlockSoundness.Discharge.FailClosed.scopeOk_eq] at hsc
  have ho : c.outNet = c.unitNet := by
    have hs := hsc
    unfold FlockSoundness.Discharge.Layout.scopeOk at hs
    simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hs
    exact hs.1.1.1.2
  obtain ⟨u, hu, -, -, -, hrows⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  rw [hc0] at hrows
  obtain ⟨-, -, hin, hout⟩ := FlockVBridge.OfVStar.recOpenUnit_ok (LANES := LANES) (H := HH) hu
  have h7 : c.ports.size = 7 := by
    have hlen := congrArg List.length hrows
    rw [List.length_map, Array.length_toList] at hlen
    rw [hlen]
    simp [Flock.RecVStar.vstarRows, VStar.Stmt.sub, VStar.recOpenSub, hin, hout, VStar.recOpenPorts]
  exact ⟨h1, hc0, ⟨tags, P⟩, hn, ho, h7, fun q hq => (FlockSoundness.Discharge.Layout.scopeOk_ports hsc q hq).1,
    ht, hpi⟩

/-- **`accHolds_recorded_layR`'s `bi`**: at a staged level part `.recOpen lv L H n` with circuit `c`, the bytes
`accSegs` reads at the unit's `acc` row, instance `m`, are what copy `m` of its one table (`copyAtR`'s, at
`p.sts[0]!`) holds from column `128 L + 1024 max(1, ⌈H/2⌉)`, at least 32 of them. -/
theorem accBytes_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : Flock.RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (w : Claims.Flock.Wit) (m : ℕ) :
    ∃ bi : ByteArray, ColsHold (Flock.Readable.copyOf p 0 h0 w (m / p.sts[0]!.c.g) (m % p.sts[0]!.c.g) 0)
        (128 * LANES + 1024 * max 1 ((HH + 1) / 2)) bi ∧ 32 ≤ bi.size ∧
      bi.data.toList = readSegs w (accSegs c ("acc", m)) := by
  obtain ⟨h1, hc0, ⟨tags, P⟩, hn, -⟩ := stagedRecOpen hS hx h0 hr hss hsv hc
  have hg : p.sts[0]!.c = c := by rw [getElem!_pos p.sts 0 h0]; exact hc0
  rw [hg]
  obtain ⟨hcs, hlen⟩ := colsHold_acc h1 h0 hc0 P hn w m
  refine ⟨_, hcs, ?_, rfl⟩
  show 32 ≤ (readSegs w (accSegs c ("acc", m))).toArray.size
  rw [List.size_toArray, hlen]
  omega

/-- **`accHolds_recorded_layR`'s `bo`**, where the part's output rows are one port each: the bytes `accOutForms` reads
at the unit's `acc_out` row `qo`, instance `m`, are what copy `m` holds from column `Family.outCol L H + 1024`. -/
theorem accOutBytes_staged {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size) {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : Flock.RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    (hor : c.outRowPorts = Array.replicate c.outPorts.size 1) {qo : ℕ}
    (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo) (w : Claims.Flock.Wit) (m : ℕ) :
    ∃ bo : ByteArray, ColsHold (Flock.Readable.copyOf p 0 h0 w (m / p.sts[0]!.c.g) (m % p.sts[0]!.c.g) 0)
        (FlockVBridge.Family.outCol LANES HH + 1024) bo ∧ 32 ≤ bo.size ∧
      bo.data.toList = readForms w (accOutForms c qo m) := by
  obtain ⟨h1, hc0, ⟨tags, P⟩, hn, ho, h7, -⟩ := stagedRecOpen hS hx h0 hr hss hsv hc
  have hg : p.sts[0]!.c = c := by rw [getElem!_pos p.sts 0 h0]; exact hc0
  rw [hg]
  obtain ⟨hcs, hlen⟩ := colsHold_accOut h1 h0 hc0 P ho hn h7 hor hqo w m
  refine ⟨_, hcs, ?_, rfl⟩
  show 32 ≤ (readForms w (accOutForms c qo m)).toArray.size
  rw [List.size_toArray, hlen]
  omega

/-- `accOutBytes_staged` at a hidden-output statement, whose output rows are one port each
(`Flock.Readable.outRowPorts_hidden`). -/
theorem accOutBytes_hidden {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {p : Flock.Readable.Parsed}
    (hx : Flock.Readable.parseBytes readOfO xO = some p) (h0 : 0 < p.sts.size) (hh : p.tags.hiddenOutputs = true)
    {r : XR} (hr : readX x = some r)
    {ss : Array VStar.Stmt} (hss : Flock.RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ}
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {c : Circuit} (hc : r.w.parts[s]?.bind partCircuit = some c)
    {qo : ℕ} (hqo : c.ports.findIdx? (·.name == "acc_out") = some qo) (w : Claims.Flock.Wit) (m : ℕ) :
    ∃ bo : ByteArray, ColsHold (Flock.Readable.copyOf p 0 h0 w (m / p.sts[0]!.c.g) (m % p.sts[0]!.c.g) 0)
        (FlockVBridge.Family.outCol LANES HH + 1024) bo ∧ 32 ≤ bo.size ∧
      bo.data.toList = readForms w (accOutForms c qo m) := by
  obtain ⟨-, hc0, -, -, -, -, hv1, ht, hpi⟩ := stagedRecOpen hS hx h0 hr hss hsv hc
  have hor := Flock.Readable.outRowPorts_hidden hx ht hpi hh h0 (by rw [hc0]; exact hv1)
  rw [hc0] at hor
  exact accOutBytes_staged hS hx h0 hr hss hsv hc hor hqo w m

end RecDeploy
