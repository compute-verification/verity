module

public import Proofs.Accounting.ReadUnit.Binding
public import Proofs.Flock.Bridge.BitsCopyGate
import Proofs.Flock.Soundness.Discharge.HiddenExec.Ret

/-!
# A read's ports are Bits' gates

`portsAt` puts a table read's ports on the copy's gates at the read template's columns: input bit `x` on column `x`,
the outputs on `Layout.outCol`'s. `Spot.Bits.inGate` and `outGate` name a gate by an input port's bit and by a returned
output's bit. In the unit a template the read's check accepts unrolls to, they are the same gates:

* `unrolled_inGroups`, `checked_inGroups`, `unrolled_inCol`: the unrolled unit has its template's input groups, the
  read's one group of 16-bit ports from column 0, so input port `p` starts at column `16 p`.
* `unrolled_readIn_lt`: the read's input bits are input columns.
* `copyGate_colAt_inGate`: the gate Bits gives on bit `t` of input port `p` is `portsAt`'s gate at column `16 p + t`.
* `copyGate_colAt_outGate`: the gate Bits gives on bit `t` of returned output `i` is `portsAt`'s gate at
  `Layout.outCol`'s column, in any circuit.

No verifier path reads this file.
-/

@[expose] public section

namespace Accounting.ReadUnit.Gates

open FlockSoundness FlockSoundness.TmplSpec FlockSoundness.Discharge.Hm Flock.ReadNet
open Flock.HmNets hiding and

/-- A templated circuit's unrolled unit has its template's input groups. -/
theorem unrolled_inGroups {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) : (Flock.HmOut.unrolled ct).unit.inGroups = tu.t.inGroups := by
  obtain ⟨-, n, -, -, -, hun⟩ := TmplValue.unroll_of_unrollAt tu.unrolled
  rw [unrolled_unit_of ht hun hsz]
  have h' := hun
  unfold Flock.Tmpl.unroll at h'
  obtain ⟨_, -, hn⟩ := Flock.TmplOrder.bind_ok h'
  obtain ⟨-, -, -, -, -, hi, -⟩ := Flock.Net.ofRows_ok hn
  exact hi

/-- An array whose image is a one-element array's is one element, of that image. -/
theorem map_single {α β : Type} (f : α → β) {gs : Array α} {g0 : α} (h : gs.map f = #[g0].map f) :
    ∃ g, gs = #[g] ∧ f g = f g0 := by
  have hl : gs.toList.map f = [f g0] := by
    have h' := congrArg Array.toList h
    rwa [Array.toList_map, Array.toList_map] at h'
  match hgs : gs.toList, hl with
  | [g], hl => exact ⟨g, Array.toList_inj.1 hgs, List.head_eq_of_cons_eq hl⟩
  | [], hl => cases hl
  | _ :: _ :: _, hl =>
    have hlen := congrArg List.length hl
    simp only [List.length_map, List.length_cons, List.length_nil] at hlen
    omega

/-- **A template the read's check accepts has one input group**: the read's ports, from column 0. -/
theorem checked_inGroups {V H : ℕ} {t : Flock.Tmpl.Tmpl} (hc : Checked V H t) :
    ∃ g : Flock.Group, t.inGroups = #[g] ∧ g.col = 0 ∧ g.bits = Flock.ReadNet.portBits V H := by
  obtain ⟨w, hr⟩ : ∃ w, (readTmpl V H).inGroups = #[⟨0, w, Flock.ReadNet.portBits V H⟩] := ⟨_, rfl⟩
  have h := hc.inG
  rw [hr] at h
  obtain ⟨g, hg, hk⟩ := map_single groupKey h
  exact ⟨g, hg, congrArg Prod.fst hk, congrArg (fun p : ℕ × ℕ × Array ℕ => p.2.2) hk⟩

/-- **The unrolled unit's input port `p` starts at column `16 p`**, where the read's check accepts its template. -/
theorem unrolled_inCol {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true) {p : ℕ}
    (hp : p < 162 + H + 32 * Hash.Tree.depth V) :
    ((Flock.portCols (Flock.HmOut.unrolled ct).unit.inGroups).getD p default).1 = 16 * p := by
  obtain ⟨g, hg, hcol, hbits⟩ := checked_inGroups (checked_of_check hcheck)
  have h16 : ∀ b ∈ g.bits.toList, b = 16 := by
    intro b hb
    rw [hbits, Flock.ReadNet.portBits, Array.toList_replicate] at hb
    exact List.eq_of_mem_replicate hb
  have hq : p < g.bits.size := by
    rw [hbits, Flock.ReadNet.portBits, Array.size_replicate]
    exact hp
  rw [unrolled_inGroups ht hsz, hg, FlockSoundness.Discharge.HiddenExec.portCols_16_get h16 hq, hcol, Nat.zero_mul,
    Nat.zero_add]

/-- **The read's input bits are input columns** of the unit a template the read's check accepts unrolls to. -/
theorem unrolled_readIn_lt {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
    (hB : Built (Flock.HmOut.unrolled ct).unit) {x : ℕ} (hx : x < readIn H (Hash.Tree.depth V)) :
    x < (Rows.ofNet hB).nIn := by
  obtain ⟨run, hr, -, -, hφ, -⟩ := read_of_check ht hsz hcheck hB
  exact lt_of_eq_of_lt (hφ x hx).symm (hr.input x hx)

end Accounting.ReadUnit.Gates

namespace Accounting.ReadUnit

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Gates

/-- `colAt` at a column of the rows is that column. -/
theorem colAt_val {R : FlockSoundness.Rows} {x : ℕ} (h : x < R.w) : (colAt R x).val = x :=
  congrArg Fin.val (colAt_of_val (c := ⟨x, h⟩) rfl)

/-- **Bits' gate on a read's input port's bit is `portsAt`'s gate at the bit's column.** In a templated circuit
whose unit template the read's check accepts, at copy `j` of unit `u`: where `Spot.Bits.inGate` gives the gate `g` on
bit `t` of input port `p`, the copy's gate at column `16 p + t` is `g`. -/
theorem copyGate_colAt_inGate {V H : ℕ} {ct : Flock.Circuit} {tu : Flock.TmplUnit} (ht : ct.tmpl = some tu)
    (hsz : ct.unitNet < ct.nets.size) (hcheck : Flock.ReadNet.check V H tu.t = true)
    (hB : FlockSoundness.Built (Flock.HmOut.unrolled ct).unit) (n : ℕ) (u : Fin n) {j : ℕ}
    (hj : j < (Flock.HmOut.unrolled ct).upv) {p t g : ℕ} (hp : p < 162 + H + 32 * Hash.Tree.depth V) (ht16 : t < 16)
    (h : Spot.Bits.inGate (Flock.HmOut.unrolled ct) n u.val j p t = some g) :
    (Flock.Bridge.copyGate (.tmpl ct) hB n u hj (colAt (FlockSoundness.Rows.ofNet hB) (16 * p + t))).val = g := by
  have hx : 16 * p + t < readIn H (Hash.Tree.depth V) := by
    unfold readIn
    omega
  have hin : 16 * p + t < (FlockSoundness.Rows.ofNet hB).nIn := unrolled_readIn_lt ht hsz hcheck hB hx
  have hv : (colAt (FlockSoundness.Rows.ofNet hB) (16 * p + t)).val = 16 * p + t :=
    colAt_val (Nat.lt_succ_of_lt (Nat.lt_of_lt_of_le hin (Nat.le_add_right _ _)))
  exact Flock.Bridge.copyGate_inGate (.tmpl ct) hB n u hj _ (lt_of_eq_of_lt hv hin)
    (hv.trans (congrArg (fun x => x + t) (unrolled_inCol ht hsz hcheck hp).symm)) h

/-- **Bits' gate on a returned output's bit is `portsAt`'s gate at the output's column.** At copy `j` of unit `u` of
the circuit `D` describes: where `Spot.Bits.outGate` gives the gate `g` on bit `t` of returned output `i`, the copy's
gate at `Layout.outCol`'s column is `g`. -/
theorem copyGate_colAt_outGate (D : Flock.Desc) (hU : FlockSoundness.Built D.unroll.unit) (n : ℕ) (u : Fin n) {j : ℕ}
    (hj : j < D.unroll.upv) {i t g : ℕ} (h : Spot.Bits.outGate D.unroll n u.val j i t = some g) :
    (Flock.Bridge.copyGate D hU n u hj (colAt (FlockSoundness.Rows.ofNet hU)
      (FlockSoundness.Discharge.Layout.outCol D.unroll i t))).val = g := by
  have hlt : FlockSoundness.Discharge.Layout.outCol D.unroll i t <
      (FlockSoundness.Rows.ofNet hU).nIn + (FlockSoundness.Rows.ofNet hU).nComp :=
    Flock.Bridge.outGate_col_lt D hU n h
  exact Flock.Bridge.copyGate_outGate D hU n u hj _ (colAt_val (Nat.lt_succ_of_lt hlt)) h

end Accounting.ReadUnit
