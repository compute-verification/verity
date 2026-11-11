module

public import Proofs.Accounting.ReadUnit.Check.Parts

/-!
# A template the read's check accepts computes the read

`read_of_check`: where `Flock.ReadNet.check V H` accepts a circuit's unit template, the unrolled unit's rows hold the
read of a table of `V` rows of `H` words. Wherever they hold with the constant set, the outputs at the unit's output
group (the row's `16 H` bits, then the verdict) are `Hash.EmbeddingRead`'s at the input bits in port order, at SHA-512
and the pinned hm96 key: `gatesRefine_of_vals`'s `hsound` for a run whose rows are none and whose outputs are those
columns (`HoldsRead`).

The check makes the template `readTmpl V H`, so `unrolled_spec` gives each root call copy its part's run, which holds at
the copy's columns (`Steps`, one lemma for every part, from `stHolds_of_realizes` and `outs_of_realizes`). Each part's
statement then gives its outputs' values, and the root composes them as the read's net does.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open FlockSoundness FlockSoundness.TmplSpec FlockSoundness.Discharge.Hm Flock.ReadNet
open Flock.HmNets hiding and

/-! ## The read's ports in the unrolled unit -/

/-- Column `c` of `R`, or its constant column past its width. -/
def readCol (R : FlockSoundness.Rows) (c : ℕ) : Fin R.w := if h : c < R.w then ⟨c, h⟩ else R.one

theorem readCol_val {R : FlockSoundness.Rows} {c : ℕ} (h : c < R.w) : (readCol R c).val = c := by
  simp [readCol, h]

/-- The read's ports' columns: input bit `v < N` at column `v`, output `j` at `oStart + j`. -/
def readPhi (R : FlockSoundness.Rows) (N oStart v : ℕ) : Fin R.w :=
  readCol R (if v < N then v else oStart + (v - N))

/-- `z` at every column, `false` past the width. -/
def readZ (R : FlockSoundness.Rows) (z : Fin R.w → Bool) (c : ℕ) : Bool := if h : c < R.w then z ⟨c, h⟩ else false

theorem readZ_fin (R : FlockSoundness.Rows) (z : Fin R.w → Bool) : (fun c : Fin R.w => readZ R z c.val) = z :=
  funext fun c => by simp [readZ, c.isLt]

theorem readZ_col {R : FlockSoundness.Rows} {z : Fin R.w → Bool} {c : ℕ} (h : c < R.w) :
    z (readCol R c) = readZ R z c := by
  simp [readCol, readZ, h]

theorem unrolled_outGroups {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) : (Flock.HmOut.unrolled ct).unit.outGroups = tu.t.outGroups := by
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  rw [unrolled_unit_of ht hun hsz]
  have h' := hun
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h'
  obtain ⟨-, -, -, -, -, -, ho, -⟩ := Flock.Net.ofRows_ok hn
  exact ho

theorem checked_outGroup {V H : ℕ} {t : Flock.Tmpl.Tmpl} (hc : Checked V H t) :
    groupKey (t.outGroups.getD 0 default) =
      ((lay V H).oStart / Flock.WORD, (lay V H).outW, Array.replicate (H + 1) 16) := by
  have h := congrArg (fun a : Array (ℕ × ℕ × Array ℕ) => a[0]?) hc.outG
  simp only [Array.getElem?_map] at h
  have hr0 : (readTmpl V H).outGroups[0]? =
      some ⟨(lay V H).oStart / Flock.WORD, (lay V H).outW, Array.replicate (H + 1) 16⟩ := rfl
  rw [hr0, Option.map_some] at h
  rw [Array.getD_eq_getD_getElem?]
  cases hg : t.outGroups[0]? with
  | none => rw [hg] at h; simp at h
  | some g =>
    rw [hg, Option.map_some, Option.some.injEq] at h
    rw [Option.getD_some]
    exact h

theorem checked_outCol {V H : ℕ} {t : Flock.Tmpl.Tmpl} (hc : Checked V H t) :
    (t.outGroups.getD 0 default).col = (lay V H).oStart / Flock.WORD :=
  congrArg Prod.fst (checked_outGroup hc)

theorem checked_outPorts {V H : ℕ} {t : Flock.Tmpl.Tmpl} (hc : Checked V H t) :
    (t.outGroups.getD 0 default).bits.size = H + 1 :=
  (congrArg (fun p : ℕ × ℕ × Array ℕ => p.2.2.size) (checked_outGroup hc)).trans Array.size_replicate

/-- **The unrolled unit returns `H + 1` outputs**: its output group's ports, the row's `H` words and then the verdict's
word, which `ExecCheck`'s `leavesOut` makes each copy's output leaves where the unit returns the outputs. -/
theorem unrolled_outPorts {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true) :
    ((Flock.HmOut.unrolled ct).unit.outGroups.getD 0 default).bits.size = H + 1 := by
  rw [unrolled_outGroups ht hsz]
  exact checked_outPorts (checked_of_check hcheck)

/-- A template's committed net has its template's output ports: `Net.ofRowsAt` moves each output group's column only. -/
theorem tmplUnit_outBits (tu : Flock.TmplUnit) :
    (tu.net.outGroups.getD 0 default).bits = (tu.t.outGroups.getD 0 default).bits := by
  obtain ⟨_, _, _, _, hat, _⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  obtain ⟨_, _, _, _, _, _, _, _, _, _, _, hog, _⟩ := Flock.Net.ofRowsAt_ok hat
  rw [hog, Array.getD_eq_getD_getElem?, Array.getD_eq_getD_getElem?, Array.getElem?_map]
  cases tu.t.outGroups[0]? <;> rfl

/-- **Each copy of the unrolled unit returns one leaf per output port**, where the unit returns the outputs (`ho`) and
the circuit's unit is its template's committed net (`hu`): `CheckFacts`' `leavesOut`, which unrolling keeps, as it keeps
the copies and the output group's ports (`tmplUnit_outBits`). -/
theorem unrolled_leaves {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hf : CheckFacts ct) (ho : ct.outNet = ct.unitNet) (hu : ct.unit = tu.net) :
    0 < (Flock.HmOut.unrolled ct).upv ∧ ∀ j < (Flock.HmOut.unrolled ct).upv,
      ((Flock.HmOut.unrolled ct).leavesOut.getD j #[]).size =
        ((Flock.HmOut.unrolled ct).unit.outGroups.getD 0 default).bits.size := by
  rw [FlockSoundness.unrolled_upv, FlockSoundness.unrolled_leavesOut, unrolled_outGroups ht hsz, ← tmplUnit_outBits,
    ← hu]
  refine ⟨hf.upv_pos, fun j hj => ?_⟩
  have hj' : j < ct.leavesOut.size := by rw [hf.leavesOut.1]; exact hj
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem hj', Option.getD_some, ← getElem!_pos ct.leavesOut j hj']
  exact hf.leavesOut.2 j hj ho

/-- **`unrolled_leaves` for a parsed templated file**: `HmOut.parseTmpl` puts its template's committed net at the unit
and has passed `HmRow.check` (`TmplValue.parseTmpl_facts`, `check_facts`). -/
theorem unrolled_leaves_of_parse {tags : Flock.Tags} {bytes : ByteArray} {tables : List (String × ByteArray)}
    {ct : Flock.Circuit} (h : Flock.HmOut.parseTmpl tags bytes tables = .ok ct) (ho : ct.outNet = ct.unitNet) :
    0 < (Flock.HmOut.unrolled ct).upv ∧ ∀ j < (Flock.HmOut.unrolled ct).upv,
      ((Flock.HmOut.unrolled ct).leavesOut.getD j #[]).size =
        ((Flock.HmOut.unrolled ct).unit.outGroups.getD 0 default).bits.size := by
  obtain ⟨tu, c₀, P, -, rfl, h0, hn0⟩ := TmplValue.parseTmpl_facts h
  obtain ⟨ls, hck⟩ := P.check
  have C := check_facts P.typed hck
  refine unrolled_leaves rfl P.unitNet ?_ ho ?_
  · cases C
    constructor <;> assumption
  · show (c₀.nets.getD c₀.unitNet default).2 = tu.net
    rw [h0, hn0]

theorem lay_oStart_dvd (V H : ℕ) : (lay V H).oStart / Flock.WORD * Flock.WORD = (lay V H).oStart := by
  refine Nat.div_mul_cancel ?_
  show Flock.WORD ∣ ((lay V H).base (nCalls V H) + (lay V H).outW * Flock.WORD - 1) /
    ((lay V H).outW * Flock.WORD) * ((lay V H).outW * Flock.WORD)
  exact Dvd.dvd.mul_left (Nat.dvd_mul_left _ _) _

/-- **A template the read's check accepts computes the read**: the unrolled unit's rows hold a run of the read's ports,
input variable `x` at input column `x`, whose outputs are the unit's output group's first `16 H + 1` columns, and those
are the read's row and verdict at SHA-512 under the pinned hm96 key (`ReadSound`, `gatesRefine_of_vals`'s `hsound`). -/
theorem read_of_check {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
    (hB : Built (Flock.HmOut.unrolled ct).unit) :
    ∃ (run : (List Lin × Lin) × St)
      (hr : HoldsRead (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])),
      ReadSound pinnedKey V H hr ∧ run.1.1.length = 16 * H ∧
        (∀ x < readIn H (Hash.Tree.depth V), (hr.φ x).val = x) ∧
        ∀ j < 16 * H + 1, (hr.out j).val = Discharge.Layout.outCol (Flock.HmOut.unrolled ct) (j / 16) (j % 16) := by
  have hc := checked_of_check hcheck
  have hzero := lay_zero V H
  have hconst := lay_const V H
  have hbase : (lay V H).zero < (lay V H).base (nCalls V H) := by rw [lay_base]; omega
  have hoS := lay_oStart_ge V H
  have hog := lay_outW_ge V H
  have hig := lay_inW_ge V H
  have hN : readIn H (Hash.Tree.depth V) = 2592 + 16 * H + 512 * depth V := (portBits_sum V H).symm
  have hone : (Rows.ofNet hB).one.val = (lay V H).const := by rw [unrolled_one ht hsz hB, checked_constPos hc]
  have hone' : (Rows.ofNet hB).one.val = (Rows.ofNet hB).nIn + (Rows.ofNet hB).nComp := rfl
  have hw : (Rows.ofNet hB).w = (lay V H).const + 1 := by
    show (Rows.ofNet hB).nIn + (Rows.ofNet hB).nComp + 1 = _
    omega
  have hnIn : (Rows.ofNet hB).nIn = (lay V H).inW * Flock.WORD := by
    rw [unrolled_nIn ht hsz hB, checked_inCols hc, checked_constPos hc]
    exact Nat.min_eq_left (by omega)
  have hlist : ((List.range (16 * H)).map fun j => Lin.var (readIn H (Hash.Tree.depth V) + j)) ++
      [Lin.var (readIn H (Hash.Tree.depth V) + 16 * H)] =
      (List.range (16 * H + 1)).map fun j => Lin.var (readIn H (Hash.Tree.depth V) + j) := by
    simp [List.range_succ]
  refine ⟨(((List.range (16 * H)).map fun j => Lin.var (readIn H (Hash.Tree.depth V) + j),
      Lin.var (readIn H (Hash.Tree.depth V) + 16 * H)), ⟨readIn H (Hash.Tree.depth V) + 16 * H + 1, []⟩),
    ⟨readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart,
      fun j => readCol (Rows.ofNet hB) ((lay V H).oStart + j), ?_, ?_, ?_, ?_⟩, ?_, ?_, ?_, ?_⟩
  · intro v hv
    show (readCol (Rows.ofNet hB) (if v < readIn H (Hash.Tree.depth V) then v else _)).val < _
    rw [if_pos hv, readCol_val (show v < (Rows.ofNet hB).w by omega), hnIn]
    omega
  · intro j hj
    have hj' : j < 16 * H + 1 := by simpa using hj
    refine ⟨⟨(lay V H).oStart + j - (Rows.ofNet hB).nIn, by omega⟩, Fin.ext ?_⟩
    show (readCol (Rows.ofNet hB) ((lay V H).oStart + j)).val =
      (Rows.ofNet hB).nIn + ((lay V H).oStart + j - (Rows.ofNet hB).nIn)
    rw [readCol_val (show (lay V H).oStart + j < (Rows.ofNet hB).w by omega)]
    omega
  · intro _ _ _ r hr
    simp at hr
  · intro z _ _ j hj
    dsimp only at hj ⊢
    rw [List.getElem_of_eq hlist hj, List.getElem_map, List.getElem_range, ev_var]
    show z (readCol _ _) = z (readCol _ (if readIn H (Hash.Tree.depth V) + j < readIn H (Hash.Tree.depth V) then _
      else _))
    rw [if_neg (show ¬(readIn H (Hash.Tree.depth V) + j < readIn H (Hash.Tree.depth V)) by omega),
      Nat.add_sub_cancel_left]
  · intro z' hs' hone''
    dsimp only
    have hs : (Rows.ofNet hB).Sat fun c => readZ (Rows.ofNet hB) z' c.val := by rw [readZ_fin]; exact hs'
    have h1 : readZ (Rows.ofNet hB) z' (Rows.ofNet hB).one.val = true := by
      rw [← readZ_col (Rows.ofNet hB).one.isLt]
      simpa [readCol] using hone''
    obtain ⟨hf, hcc, hout, hcp⟩ := root_facts ht hsz hc hB hs h1
    obtain ⟨hrow, hok⟩ := root_read hc.hV32 hc.hH64 hf hcc hcp
    have hwin : ∀ v < 2592 + 16 * H + 512 * depth V,
        z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v) =
          rootVal V H (readZ (Rows.ofNet hB) z') v := fun v hv => by
      have h1 : v < readIn H (Hash.Tree.depth V) := by omega
      have h2 : v < (Rows.ofNet hB).w := by omega
      have h3 : v ≤ (lay V H).const := by omega
      rw [readPhi, if_pos h1, readZ_col (z := z') h2, rootVal_le h3]
    have hwout : ∀ j ≤ 16 * H,
        z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart
          (readIn H (Hash.Tree.depth V) + j)) =
          rootVal V H (readZ (Rows.ofNet hB) z') ((lay V H).outAt (segs H + 4 + depth V) j) := fun j hj => by
      have h1 : ¬(readIn H (Hash.Tree.depth V) + j < readIn H (Hash.Tree.depth V)) := by omega
      have h2 : (lay V H).oStart + j < (Rows.ofNet hB).w := by omega
      rw [readPhi, if_neg h1, Nat.add_sub_cancel_left, readZ_col (z := z') h2, hout j (by omega)]
    have hcol : ∀ k n, k + n ≤ 2592 + 16 * H + 512 * depth V →
        evs (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) (bitsAt k n) =
          colV V H (readZ (Rows.ofNet hB) z') k n := by
      intro k n hkn
      rw [evs_bitsAt, colV]
      exact List.map_congr_left fun t ht => hwin _ (by have := List.mem_range.1 ht; omega)
    have hdig : ∀ k, k + 512 ≤ 2592 + 16 * H + 512 * depth V →
        wordsDigest (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v))
          (wordsAt k 8) = ReadUnit.digestOf (colV V H (readZ (Rows.ofNet hB) z') k 512) := by
      intro k hk
      show ReadUnit.digestOf (evs _ (wordsAt k 8).flatten) = _
      rw [flatten_wordsAt]
      exact congrArg ReadUnit.digestOf (hcol k (64 * 8) (by omega))
    have htok : readTok (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) =
        ReadUnit.num (colV V H (readZ (Rows.ofNet hB) z') 0 32) :=
      congrArg ReadUnit.num (hcol 0 32 (by omega))
    have hroot : readRoot (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) =
        ReadUnit.digestOf (colV V H (readZ (Rows.ofNet hB) z') ROOT_AT 512) :=
      hdig ROOT_AT (by simp only [ROOT_AT]; omega)
    have hdom : readDomain (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) =
        ReadUnit.digestOf (colV V H (readZ (Rows.ofNet hB) z') DOMAIN_AT 512) :=
      hdig DOMAIN_AT (by simp only [DOMAIN_AT]; omega)
    have hopen : readOpening H (Hash.Tree.depth V)
        (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) =
        openV V H (readZ (Rows.ofNet hB) z') := by
      have hrow' := hcol ROW_AT (16 * H) (by simp only [ROW_AT]; omega)
      have hsalt : wordsBytes (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v))
          (wordsAt (saltAt H) 24) = ReadUnit.bytesOf (colV V H (readZ (Rows.ofNet hB) z') (saltAt H) 1536) := by
        show ReadUnit.bytesOf (evs _ (wordsAt (saltAt H) 24).flatten) = _
        rw [flatten_wordsAt]
        exact congrArg ReadUnit.bytesOf (hcol (saltAt H) (64 * 24) (by simp only [saltAt]; omega))
      have hpath : ((List.range (Hash.Tree.depth V)).map fun j => insW (2592 + 16 * H + 512 * j) 8).map
          (wordsDigest fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) =
          (List.range (depth V)).map (sibV V H (readZ (Rows.ofNet hB) z')) := by
        rw [List.map_map]
        exact List.map_congr_left fun j hj =>
          hdig (pathAt H + 512 * j)
            (by have := List.mem_range.1 hj; have := depth_eq V; simp only [pathAt]; omega)
      have e : readOpening H (Hash.Tree.depth V)
          (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v)) =
          ⟨evs (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v))
              (bitsAt ROW_AT (16 * H)),
            wordsBytes (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v))
              (wordsAt (saltAt H) 24),
            ((List.range (Hash.Tree.depth V)).map fun j => insW (2592 + 16 * H + 512 * j) 8).map
              (wordsDigest fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v))⟩ :=
        rfl
      rw [e, hrow', hsalt, hpath]
      rfl
    have hevs : evs (fun v => z' (readPhi (Rows.ofNet hB) (readIn H (Hash.Tree.depth V)) (lay V H).oStart v))
        ((List.range (16 * H)).map fun j => Lin.var (readIn H (Hash.Tree.depth V) + j)) =
        outV V H (readZ (Rows.ofNet hB) z') (segs H + 4 + depth V) (16 * H) := by
      simp only [evs, outV, List.map_map]
      refine List.map_congr_left fun j hj => ?_
      simp only [Function.comp_apply, ev_var]
      exact hwout j (List.mem_range.1 hj).le
    refine ⟨?_, fun hv => ?_⟩
    · rw [hevs, hrow, htok, hopen]
    · have hv' : rootVal V H (readZ (Rows.ofNet hB) z') ((lay V H).outAt (segs H + 4 + depth V) (16 * H)) = true := by
        rw [← hwout (16 * H) le_rfl]
        simpa using hv
      rw [htok, hopen, hroot, hdom]
      exact hok hv'
  · simp
  · intro x hx
    show (readCol (Rows.ofNet hB) (if x < readIn H (Hash.Tree.depth V) then x else _)).val = x
    rw [if_pos hx, readCol_val (show x < (Rows.ofNet hB).w by omega)]
  · intro j hj
    show (readCol (Rows.ofNet hB) ((lay V H).oStart + j)).val = _
    rw [readCol_val (show (lay V H).oStart + j < (Rows.ofNet hB).w by omega), Discharge.Layout.outCol,
      unrolled_outGroups ht hsz, checked_outCol hc, Nat.div_add_mod, lay_oStart_dvd]

end Accounting.ReadUnit.Gates
