module

public import Flock.ReadNet.Template
public import Proofs.Accounting.ReadUnit.Tie
public import Proofs.Accounting.ReadUnit.RowSeg
public import Proofs.Accounting.ReadUnit.Leaf
public import Proofs.Accounting.ReadUnit.Select
public import Proofs.Accounting.ReadUnit.Frame
public import Proofs.Flock.Soundness.TmplSpec
public import Properties.Flock.Circuit.Outs

/-!
# The read's template: each copy holds its part's run

The check makes the template `readTmpl V H`. Each part's Definition holds its generator's run at every copy whose rows
hold (`partHolds_of_rows`, from `stHolds_of_realizes` and `outs_of_realizes`), so the template's Definitions meet
`readSpec` (`steps_read`), and `unrolled_spec` gives the root's facts wherever the unrolled unit's rows hold with the
constant set (`root_facts`): the zero and constant columns, the output columns, and each call's copy holding its part's
run on the values its inputs bind.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open FlockSoundness FlockSoundness.TmplSpec FlockSoundness.Discharge.Hm Flock.ReadNet
open Flock.HmNets hiding and
open Flock.Tmpl (Def Call Seg canon dropPairs)

/-! ## A part's Definition holds its run -/

/-- Where a part's Definition reads variable `x` (`defOf`'s `col`): input `x` past its columns, else own column
`x - ins`. -/
def colOf (ins : ℕ) (r : List Lin × St) (x : Var) : ℕ :=
  if x < ins then (defOf ins r).cols + x else x - ins

/-- The value a copy of a part's Definition (own columns `v`, inputs `inp`) gives its run's variable `x`. -/
def partVal (ins : ℕ) (r : List Lin × St) (v : ℕ → Bool) (inp : List Bool) (x : Var) : Bool :=
  (defOf ins r).refV v inp true (colOf ins r x)

/-- **A part's run holds at a copy**: its rows hold at the copy's values, and each form it returns is the copy's own
column that commits it. -/
def PartHolds (ins : ℕ) (r : List Lin × St) (v : ℕ → Bool) (inp : List Bool) : Prop :=
  stHolds (partVal ins r v inp) r.2 ∧
    ∀ j (hj : j < r.1.length), v (r.2.next - ins + j) = ev (partVal ins r v inp) r.1[j]

theorem defOf_cols (ins : ℕ) (r : List Lin × St) : (defOf ins r).cols = r.2.next - ins + r.1.length := rfl

theorem defOf_ins (ins : ℕ) (r : List Lin × St) : (defOf ins r).ins = ins := rfl

theorem canon_single (x : ℕ) : canon #[x] = #[x] := by
  simp [canon, dropPairs]

/-- **A part's Definition holds its run** at every copy whose own rows hold, the constant 1, where each of the run's
rows writes a fresh variable (`runOk`). -/
theorem partHolds_of_rows {ins : ℕ} {r : List Lin × St} (hok : runOk ins r = true) {v : ℕ → Bool} {inp : List Bool}
    (hrows : ∀ w ∈ (defOf ins r).rows.toList, v w.1 = Flock.Tmpl.rowVal ((defOf ins r).refV v inp true) w) :
    PartHolds ins r v inp := by
  refine ⟨stHolds_of_realizes (φ := colOf ins r) ?_ hrows,
    outs_of_realizes (oc := fun j => r.2.next - ins + j) ?_ hrows⟩
  · intro w hw
    -- `w.out : Var`, whose `<` and `-` omega doesn't read as `ℕ`'s
    obtain ⟨o, ho⟩ : ∃ o : ℕ, w.out = o := ⟨_, rfl⟩
    have hb : ins ≤ o ∧ o < r.2.next := by
      have := List.all_eq_true.1 hok w hw
      rw [← ho]
      simpa using this
    have hc : colOf ins r w.out = o - ins := by
      rw [ho]
      exact if_neg (show ¬o < ins by omega)
    refine ⟨by rw [hc, defOf_cols]; omega,
      (colOf ins r w.out, refsOf (defOf ins r) (colOf ins r) w.a, refsOf (defOf ins r) (colOf ins r) w.b), ?_, rfl, rfl,
      rfl⟩
    show _ ∈ ((r.2.rows.reverse.map _).toArray ++ _).toList
    rw [Array.toList_append, List.mem_append]
    left
    rw [List.toList_toArray]
    exact List.mem_map.2 ⟨w, List.mem_reverse.2 hw, rfl⟩
  · intro j hj
    refine ⟨by rw [defOf_cols]; omega,
      (r.2.next - ins + j, refsOf (defOf ins r) (colOf ins r) r.1[j], #[(defOf ins r).cols + (defOf ins r).ins]), ?_,
      rfl, rfl, canon_single _⟩
    show _ ∈ (_ ++ (r.1.mapIdx _).toArray).toList
    rw [Array.toList_append, List.mem_append]
    right
    rw [List.toList_toArray]
    have hj' : j < (r.1.mapIdx fun k f => (r.2.next - ins + k, refsOf (defOf ins r) (colOf ins r) f,
        #[(defOf ins r).cols + (defOf ins r).ins])).length := by rw [List.length_mapIdx]; exact hj
    have := List.getElem_mem hj'
    rwa [List.getElem_mapIdx] at this

/-- The read's specification of Definition `d`: a copy of a part holds the part's run. -/
def readSpec (V H d : ℕ) (v : ℕ → Bool) (inp : List Bool) : Prop :=
  ∀ p ∈ (parts V H)[d]?, PartHolds p.ins p.run v inp

theorem parts_size (V H : ℕ) : (parts V H).size = 11 := rfl

theorem partDefs_size (V H : ℕ) : (partDefs V H).size = 11 := by simp [partDefs, parts]

theorem readTmpl_defs (V H : ℕ) :
    (readTmpl V H).defs = (partDefs V H).push (rootDef V H (partDefs V H)) := rfl

/-- **Every Definition of the read's template meets `readSpec`** (`Steps`), where every part's run writes fresh
variables. -/
theorem steps_read {V H : ℕ} (hok : (parts V H).all (fun p => runOk p.ins p.run) = true) :
    Steps (readTmpl V H).defs (readSpec V H) := by
  intro d _ v inp _ hrows _ _ p hp
  obtain ⟨hd, hpd⟩ := Array.getElem?_eq_some_iff.1 hp
  have hdef : (readTmpl V H).defs.getD d default = defOf p.ins p.run := by
    rw [readTmpl_defs, Array.getD_eq_getD_getElem?,
      Array.getElem?_push_lt (by rw [partDefs_size, ← parts_size V H]; exact hd)]
    simp [partDefs, hpd]
  rw [hdef] at hrows
  exact partHolds_of_rows (by have := Array.all_eq_true.1 hok d hd; rwa [hpd] at this) hrows

/-! ## The root, through `unrolled_spec` -/

theorem readTmpl_root (V H : ℕ) : (readTmpl V H).defs.getD ROOT default = rootDef V H (partDefs V H) := by
  rw [readTmpl_defs, show ROOT = (partDefs V H).size by rw [partDefs_size]; rfl]
  simp [Array.getD]

theorem boundV_single {c : Call} {l : List ℕ}
    (hc : c.segs = (l.map fun r => ({ len := 1, at_ := r, step := 0, reps := 1, rstride := 0 } : Seg)).toArray)
    (g : ℕ → Bool) : c.boundV g 0 = l.map g := by
  simp [Flock.Tmpl.Call.boundV, hc, List.flatMap_map]
  clear hc
  induction l with
  | nil => rfl
  | cons a l ih => simp [ih]

/-- What `check V H t` establishes. -/
structure Checked (V H : ℕ) (t : Flock.Tmpl.Tmpl) : Prop where
  hV : 2 ≤ V
  hV32 : V < 2 ^ 32
  hH : 1 ≤ H
  hH64 : 16 * H < 2 ^ 64
  ok : (parts V H).all (fun p => runOk p.ins p.run) = true
  root : t.root = ROOT
  inG : t.inGroups.map groupKey = (readTmpl V H).inGroups.map groupKey
  outG : t.outGroups.map groupKey = (readTmpl V H).outGroups.map groupKey
  defs : t.defs = (readTmpl V H).defs
  log : Flock.log2ceil (t.defs.getD t.root default).cols ≤ 32

theorem checked_of_check {V H : ℕ} {t : Flock.Tmpl.Tmpl} (h : Flock.ReadNet.check V H t = true) :
    Checked V H t := by
  simp only [Flock.ReadNet.check, Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h
  obtain ⟨⟨⟨⟨⟨⟨⟨⟨⟨h1, h2⟩, h3⟩, h0⟩, h4⟩, h5⟩, h6⟩, h7⟩, h8⟩, h9⟩ := h
  exact ⟨h1, h2, h3, h0, h4, h5, h6, h7, h8, h9⟩

/-- The read's ports' bits: `readIn`. -/
theorem portBits_sum (V H : ℕ) : 2592 + 16 * H + 512 * depth V = readIn H (depth V) := by
  unfold readIn; omega

theorem depth_eq (V : ℕ) : depth V = Hash.Tree.depth V := rfl

section layout

variable (V H : ℕ)

/-- The root's layout. -/
abbrev lay : Layout := layout V H (partDefs V H)

/-- The root's references' values at the root's columns `z`. -/
abbrev rootVal (z : ℕ → Bool) : ℕ → Bool := (rootDef V H (partDefs V H)).refV z [] true

theorem lay_zero : (lay V H).zero = (lay V H).inW * Flock.WORD := rfl

theorem lay_const : (lay V H).const = (lay V H).oStart + (lay V H).outW * Flock.WORD := rfl

theorem rootDef_cols : (rootDef V H (partDefs V H)).cols = (lay V H).const + 1 := rfl

theorem lay_base (i : ℕ) : (lay V H).base i = (lay V H).zero + 1 +
    ((List.range i).map fun i => ((partDefs V H).getD (partOf V H i) default).cols).sum := rfl

theorem lay_base_succ (i : ℕ) : (lay V H).base (i + 1) =
    (lay V H).base i + ((partDefs V H).getD (partOf V H i) default).cols := by
  rw [lay_base, lay_base, List.range_succ, List.map_append, List.sum_append]
  simp only [List.map_cons, List.map_nil, List.sum_cons, List.sum_nil]
  omega

theorem lay_base_mono {i j : ℕ} (h : i ≤ j) : (lay V H).base i ≤ (lay V H).base j := by
  induction h with
  | refl => exact le_rfl
  | step _ ih => rw [lay_base_succ]; omega

theorem lay_outW_pos : 0 < (lay V H).outW :=
  Nat.pos_of_isPowerOfTwo (Nat.isPowerOfTwo_nextPowerOfTwo _)

theorem lay_inW_ge : 2592 + 16 * H + 512 * depth V ≤ (lay V H).inW * Flock.WORD := by
  have h := Nat.le_nextPowerOfTwo ((2592 + 16 * H + 512 * depth V + Flock.WORD - 1) / Flock.WORD)
  have hw : Flock.WORD = 128 := rfl
  show _ ≤ Nat.nextPowerOfTwo ((2592 + 16 * H + 512 * depth V + Flock.WORD - 1) / Flock.WORD) * Flock.WORD
  rw [hw] at h ⊢
  omega

theorem lay_outW_ge : 16 * H + 16 ≤ (lay V H).outW * Flock.WORD := by
  have h := Nat.le_nextPowerOfTwo ((16 * H + 16 + Flock.WORD - 1) / Flock.WORD)
  have hw : Flock.WORD = 128 := rfl
  show _ ≤ Nat.nextPowerOfTwo ((16 * H + 16 + Flock.WORD - 1) / Flock.WORD) * Flock.WORD
  rw [hw] at h ⊢
  omega

theorem lay_oStart_ge : (lay V H).base (nCalls V H) ≤ (lay V H).oStart := by
  have hs : 0 < (lay V H).outW * Flock.WORD := Nat.mul_pos (lay_outW_pos V H) (by decide)
  show _ ≤ ((lay V H).base (nCalls V H) + (lay V H).outW * Flock.WORD - 1) / ((lay V H).outW * Flock.WORD) *
    ((lay V H).outW * Flock.WORD)
  generalize (lay V H).outW * Flock.WORD = s at hs
  generalize (lay V H).base (nCalls V H) = b
  have h1 := Nat.div_add_mod (b + s - 1) s
  have h2 := Nat.mod_lt (b + s - 1) hs
  rw [Nat.mul_comm] at h1
  omega

end layout

/-! ## The root's facts -/

theorem unrolled_one {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hB : Built (Flock.HmOut.unrolled ct).unit) :
    (Rows.ofNet hB).one.val = tu.t.constPos := by
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  have hU := unrolled_unit_of ht hun hsz
  generalize hm : (Flock.HmOut.unrolled ct).unit = m at hB
  obtain rfl : m = n := hm.symm.trans hU
  exact unroll_one hun hB

theorem unrolled_nIn {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hB : Built (Flock.HmOut.unrolled ct).unit) :
    (Rows.ofNet hB).nIn = min tu.t.inCols tu.t.constPos := by
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  have hU := unrolled_unit_of ht hun hsz
  generalize hm : (Flock.HmOut.unrolled ct).unit = m at hB
  obtain rfl : m = n := hm.symm.trans hU
  rw [Rows.ofNet_nIn]
  have h' := hun
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h'
  obtain ⟨-, -, -, -, hc, -, -, -, -, -, -, -, -, hw, -⟩ := Flock.Net.ofRows_ok hn
  rw [hc, hw]
  rfl

theorem checked_inCols {V H : ℕ} {t : Flock.Tmpl.Tmpl} (hc : Checked V H t) :
    t.inCols = (lay V H).inW * Flock.WORD := by
  have h := congrArg (fun a : Array (ℕ × ℕ × Array ℕ) => (a.back?.map fun k => k.1 + k.2.1).getD 0) hc.inG
  simp only [Array.back?_map, Option.map_map] at h
  unfold Flock.Tmpl.Tmpl.inCols
  rw [show (t.inGroups.back?.map fun g => g.col + g.words) = (t.inGroups.back?.map ((fun k : ℕ × ℕ × Array ℕ =>
    k.1 + k.2.1) ∘ groupKey)) from rfl, h]
  show (0 + (lay V H).inW) * Flock.WORD = _
  rw [Nat.zero_add]

theorem checked_constPos {V H : ℕ} {t : Flock.Tmpl.Tmpl} (hc : Checked V H t) : t.constPos = (lay V H).const := by
  unfold Flock.Tmpl.Tmpl.constPos
  rw [hc.defs, hc.root, readTmpl_root, rootDef_cols]
  rfl

/-- The unrolled unit has at most `2^32` rows: its slot is its root's columns', which the check bounds. -/
theorem unrolled_unitLog {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hc : Checked V H tu.t) : (Flock.HmOut.unrolled ct).unit.unitLog ≤ 32 := by
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  rw [unrolled_unit_of ht hun hsz]
  have h' := hun
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h'
  rw [(Flock.Net.ofRows_ok hn).1]
  exact Nat.max_le.2 ⟨by decide, hc.log⟩

/-- **The root's facts** wherever the unrolled unit's rows hold with the constant 1, at the root's references' values
`rootVal z`: the zero column is 0 and the constant 1, each output column is the select copy's, and each call's copy holds
its part's run, bound to its inputs. -/
theorem root_facts {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hc : Checked V H tu.t) (hB : Built (Flock.HmOut.unrolled ct).unit)
    {z : ℕ → Bool} (hs : (Rows.ofNet hB).Sat fun c => z c.val)
    (hone : z (Rows.ofNet hB).one.val = true) :
    rootVal V H z (lay V H).zero = false ∧ rootVal V H z ((lay V H).const + 1) = true ∧
      (∀ k < 16 * H + 1, z ((lay V H).oStart + k) = rootVal V H z ((lay V H).outAt (segs H + 4 + depth V) k)) ∧
      ∀ i < nCalls V H, ∀ p ∈ (parts V H)[partOf V H i]?,
        PartHolds p.ins p.run (fun y => z ((lay V H).base i + y)) ((inputs V H (lay V H) i).map (rootVal V H z)) := by
  have hS : Steps tu.t.defs (readSpec V H) := hc.defs ▸ steps_read hc.ok
  obtain ⟨hrows, hcalls⟩ := unrolled_spec ht hsz hB (unrolled_unitLog ht hsz hc) hS hs hone
  have hroot : tu.t.defs.getD tu.t.root default = rootDef V H (partDefs V H) := by
    rw [hc.defs, hc.root]; exact readTmpl_root V H
  rw [hroot, checked_inCols hc] at hrows hcalls
  have hC : rootVal V H z ((lay V H).const + 1) = true := by
    simp [Flock.Tmpl.Def.refV, rootDef_cols]
  have hval : ∀ x ≤ (lay V H).const, rootVal V H z x = z x := fun x hx => by
    simp [Flock.Tmpl.Def.refV, rootDef_cols, show x < (lay V H).const + 1 by omega]
  have hzc : (lay V H).zero < (lay V H).const := by
    have := lay_oStart_ge V H
    have h1 : (lay V H).zero + 1 ≤ (lay V H).base (nCalls V H) := by rw [lay_base]; omega
    have h2 := lay_outW_pos V H
    rw [lay_const]
    have : 0 < (lay V H).outW * Flock.WORD := Nat.mul_pos h2 (by decide)
    omega
  refine ⟨?_, hC, ?_, ?_⟩
  · rw [hval _ hzc.le]
    have hmem : ((lay V H).zero, #[(lay V H).const + 1], #[]) ∈ (rootDef V H (partDefs V H)).rows.toList := by
      show _ ∈ (Array.push _ _ ++ _).toList
      rw [Array.toList_append, List.mem_append, Array.toList_push]
      exact Or.inl (List.mem_concat_self ..)
    have := hrows _ hmem (by rw [lay_zero])
    rw [this]
    simp [Flock.Tmpl.rowVal, Flock.Tmpl.xorAt]
  · intro k hk
    have hmem : ((lay V H).oStart + k, #[(lay V H).outAt (segs H + 4 + depth V) k], #[(lay V H).const + 1]) ∈
        (rootDef V H (partDefs V H)).rows.toList := by
      show _ ∈ (_ ++ ((List.range (16 * H + 1)).map _).toArray).toList
      rw [Array.toList_append, List.mem_append, List.toList_toArray]
      exact Or.inr (List.mem_map.2 ⟨k, List.mem_range.2 hk, rfl⟩)
    have hge : (lay V H).inW * Flock.WORD ≤ (lay V H).oStart + k := by
      have := lay_oStart_ge V H
      have h1 : (lay V H).zero + 1 ≤ (lay V H).base (nCalls V H) := by rw [lay_base]; omega
      rw [lay_zero] at h1
      omega
    rw [hrows _ hmem hge]
    simp [Flock.Tmpl.rowVal, Flock.Tmpl.xorAt, hC]
  · intro i hi p hp
    have hmem : (⟨partOf V H i, 1, (lay V H).base i, ((partDefs V H).getD (partOf V H i) default).cols,
        ((inputs V H (lay V H) i).map fun r =>
          ({ len := 1, at_ := r, step := 0, reps := 1, rstride := 0 } : Seg)).toArray⟩ : Call) ∈
        (rootDef V H (partDefs V H)).calls.toList := by
      show _ ∈ ((List.range (nCalls V H)).map _).toArray.toList
      rw [List.toList_toArray]
      exact List.mem_map.2 ⟨i, List.mem_range.2 hi, rfl⟩
    have hge : (lay V H).inW * Flock.WORD ≤ (lay V H).base i := by rw [lay_base, lay_zero]; omega
    have := hcalls _ hmem hge 0 Nat.one_pos p
    simp only [Nat.zero_mul, Nat.add_zero] at this
    rw [boundV_single rfl] at this
    exact this hp

/-- `gatesRefine_of_vals`'s `hsound` for `hr`: wherever `R`'s rows hold with the constant set, the run's outputs are the
read's row and, where its verdict is set, an opening of it under `key`. -/
def ReadSound (key : Hash.Msg) (V H : ℕ) {R : FlockSoundness.Rows} {run : (List Lin × Lin) × St}
    (hr : HoldsRead R (readIn H (Hash.Tree.depth V)) run.2 (run.1.1 ++ [run.1.2])) : Prop :=
  ∀ z : Fin R.w → Bool, R.Sat z → z R.one = true →
    evs (fun v => z (hr.φ v)) run.1.1 =
        Hash.EmbeddingRead.h V H (readTok fun v => z (hr.φ v))
          (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v)) ∧
      (ev (fun v => z (hr.φ v)) run.1.2 = true →
        Hash.EmbeddingRead.ok Verity.sha512 Verity.sha512 ⟨readDomain fun v => z (hr.φ v), V, fun r => r⟩ key
          (readRoot fun v => z (hr.φ v)) V (readTok fun v => z (hr.φ v))
          (readOpening H (Hash.Tree.depth V) fun v => z (hr.φ v)))

end Accounting.ReadUnit.Gates
