import Proofs.Flock.Soundness.Compose
import Proofs.Flock.Soundness.Types.Dag

/-!
# A unit's rows in logical order, over the type DAG (verified-lowering phase 1, 1d step 2)

`Compose.ofBlock words done u` is the unit's `Logical` from `deriveChecked`'s layouts: `Flock.DeriveAll.order` (the list
`logicalText` prints), and each column's row as the block holds it: Δ's constant copies copy `one`, a Δ binding `src` is
`src · src`, and any other column its row as the verifier folds it (`Flock.DeriveAll.fullRow`: the derived row, a read's
product row with its table-direct side from its record).

For every statement `deriveChecked` accepts:
- `ofBlock_ordered`: the rows are `Ordered` (`Types.Dag.order_sound`, through `Compose.ordered_of_take`);
- `outCols_mem`: every output column of the unit type is in the order, so it has a logical column;
- `Rows.compose_eval_unit`: satisfied composed rows, with `one` at 1, carry on each output's logical column the unit
  type's output (`evalT`) on the input columns' values. This is `Types.Dag.unit_sound` read through the order.
-/

open Flock Flock.DeriveAll

namespace FlockSoundness

open Audit Types.Dag

namespace Compose

/-- **The unit's rows in logical order**, over the type DAG: `order done u`, and each column's row as the block holds it:
Δ's constant copies copy `one`, a Δ binding `src` reads `src · src`, and any other column its row as the verifier folds it
(`fullRow`, a read's product row with its table-direct side), the tables' words by SHA-512 in `words`. -/
abbrev ofBlock (words : String → Option (Array ℕ)) (done : Array D) (u : D) : Logical :=
  ⟨u.inCols.size, order done u, fun c =>
    match u.delta.find? (·.1 == c) with
    | some (_, none) => none
    | some (_, some s) => some (s, s)
    | none => some ((fullRow words u c).getD ([], []))⟩

/-- `ofBlock`'s rows are `blockRow`'s, whatever `one` is: a copy of `one`, the same row, or no row (then a zero row). -/
theorem ofBlock_cases (words : String → Option (Array ℕ)) (done : Array D) (u : D) (c : ℕ) :
    ((ofBlock words done u).row c = none ∧ ∀ one, blockRow words u one c = some ([one], [one])) ∨
    (∃ p, (ofBlock words done u).row c = some p ∧ ∀ one, blockRow words u one c = some p) ∨
    ((ofBlock words done u).row c = some ([], []) ∧ ∀ one, blockRow words u one c = none) := by
  rcases hf : u.delta.find? (·.1 == c) with _ | ⟨x, _ | s⟩
  · rcases hr : fullRow words u c with _ | r
    · exact Or.inr (Or.inr ⟨by simp [hf, hr], fun _ => by simp [blockRow, hf, hr]⟩)
    · exact Or.inr (Or.inl ⟨r, by simp [hf, hr], fun _ => by simp [blockRow, hf, hr]⟩)
  · exact Or.inl ⟨by simp [hf], fun _ => by simp [blockRow, hf]⟩
  · exact Or.inr (Or.inl ⟨(s, s), by simp [hf], fun _ => by simp [blockRow, hf]⟩)

theorem order_getD_inCols {done : Array D} {u : D} (htake : (order done u).take u.inCols.size = u.inCols.toList)
    {i : ℕ} (hi : i < u.inCols.size) : (order done u).getD i 0 = u.inCols.getD i 0 := by
  have h := congrArg (fun l => l[i]?) htake
  simp only [List.getElem?_take, hi, ite_true, Array.getElem?_toList] at h
  rw [List.getD_eq_getElem?_getD, h, Array.getD_eq_getD_getElem?]

variable {L : Logical} {h : Ordered L}

theorem physical_of_mem (z : Fin (Rows.compose L h).w → Bool) {c : ℕ} (hc : c ∈ L.order) :
    physical z c = z ⟨lidx L c, by rw [Rows.compose_w]; have := List.idxOf_lt_length_iff.2 hc; unfold lidx; omega⟩ := by
  unfold physical
  split
  · rfl
  · rename_i h'; exact absurd (List.idxOf_lt_length_iff.2 hc) h'

/-- A row that copies `one`, satisfied with `one` at 1, carries 1. -/
theorem comp_one {z : Fin (Rows.compose L h).w → Bool} (hz : (Rows.compose L h).Sat z)
    (h1 : z (Rows.compose L h).one = true) (k : Fin (Rows.compose L h).nComp)
    (hr : L.row (L.order.getD (L.nIn + k.val) 0) = none) : z ((Rows.compose L h).comp k) = true := by
  have hone : ∀ M : List (Fin (Rows.compose L h).w), M.map Fin.val = [L.order.length] → xorSum M z = true := by
    intro M hM
    cases M with
    | nil => cases hM
    | cons x M' =>
      cases M' with
      | nil =>
        simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true] at hM
        have : x = (Rows.compose L h).one := Fin.ext (by rw [Rows.compose_one]; exact hM)
        simp [xorSum, this, h1]
      | cons y M'' => simp at hM
  rw [hz k, hone _ (by rw [Rows.compose_a, lrow_none hr]), hone _ (by rw [Rows.compose_b, lrow_none hr])]
  rfl

end Compose

open Compose

section unit

variable {types : String → Option CircuitType} {ls : Array Layout} {info : String → Option TableInfo}
  {words : String → Option (Array ℕ)} {unit : ℕ} {done : Array D}

/-- **The unit's composed rows read only earlier columns**, or `one`. -/
theorem ofBlock_ordered (hd : deriveChecked types ls info words unit = .ok done) :
    Ordered (ofBlock words done (done.getD unit default)) := by
  have hs := fun one => order_sound hd one
  generalize done.getD unit default = u at *
  obtain ⟨-, -, htake, -, -⟩ := hs 0
  refine ordered_of_take ?_ fun ℓ hℓ hn p hp c hc => ?_
  · have := congrArg List.length htake
    simp only [List.length_take, Array.length_toList] at this
    show u.inCols.size ≤ (order done u).length
    omega
  · have hp' : (ofBlock words done u).row ((order done u).getD ℓ 0) = some p := hp
    have hℓ' : ℓ < (order done u).length := hℓ
    rcases ofBlock_cases words done u ((order done u).getD ℓ 0) with ⟨hr, -⟩ | ⟨p', hr, hb⟩ | ⟨hr, -⟩
    · rw [hr] at hp'; cases hp'
    · rw [hr, Option.some.injEq] at hp'
      subst hp'
      obtain ⟨-, -, -, hread, -⟩ := hs (c + 1)
      exact hread ℓ hℓ' hn p' (by rw [← List.getD_eq_getElem _ 0 hℓ']; exact hb (c + 1)) c hc (by omega)
    · rw [hr, Option.some.injEq] at hp'
      subst hp'
      simp at hc

/-- **Every output column of the unit type is in the order**: its row copies the output's form, with the constant. -/
theorem outCols_mem (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) {o : ℕ} (ho : o < t.outputs.size) :
    (done.getD unit default).outCols.getD o 0 ∈ order done (done.getD unit default) := by
  obtain ⟨-, -, -, -, -, -, wire, -, hout⟩ := checkLayout_spec (deriveChecked_all hd unit (lt_of_type ht)) ht
  have hk : (unit == unit) = true := by simp
  have hout' := hout o ho
  rw [hk] at hout'
  have hrow := rowIs_unit words hout' 0
  obtain ⟨-, -, -, -, hz⟩ := order_sound hd 0
  by_contra hm
  have := hz _ _ hm hrow
  simp at this

/-- **The unit's composed rows compute its type.** Column values that satisfy `Rows.compose` of the unit's rows, with
`one` at 1, carry on output `o`'s logical column the unit type's output `o` (`evalT`) on the input columns' values. -/
theorem Rows.compose_eval_unit (hd : deriveChecked types ls info words unit = .ok done) {t : CircuitType}
    (ht : (ls.getD unit default).type.bind types = some t) {h : Ordered (ofBlock words done (done.getD unit default))}
    {z : Fin (Rows.compose (ofBlock words done (done.getD unit default)) h).w → Bool}
    (hz : (Rows.compose (ofBlock words done (done.getD unit default)) h).Sat z)
    (h1 : z (Rows.compose (ofBlock words done (done.getD unit default)) h).one = true) {x : ℕ → Bool}
    (hx : ∀ i : Fin (Rows.compose (ofBlock words done (done.getD unit default)) h).w,
      i.val < (done.getD unit default).inCols.size → x i.val = z i)
    (F : ℕ) (hF : unit < F) {o : ℕ} (ho : o < t.outputs.size)
    (ℓ : Fin (Rows.compose (ofBlock words done (done.getD unit default)) h).w)
    (hℓ : ℓ.val = lidx (ofBlock words done (done.getD unit default)) ((done.getD unit default).outCols.getD o 0)) :
    z ℓ = (evalT types words F t x).getD o false := by
  have hmo := outCols_mem hd ht ho
  have hsound := unit_sound hd ht (done.getD unit default).rows.size
  have hins := inputs_self hd ht (done.getD unit default).rows.size
  obtain ⟨hn, hlt, htake, -, hzero⟩ := order_sound hd (done.getD unit default).rows.size
  obtain ⟨-, -, hin, -⟩ := checkLayout_spec (deriveChecked_all hd unit (lt_of_type ht)) ht
  have hΔ : ∀ e ∈ (done.getD unit default).delta, e.1 ∈ order done (done.getD unit default) := by
    have ho' := (deriveChecked_spec hd).2
    unfold orderChecked at ho'
    simp only [Bool.and_eq_true] at ho'
    exact (outsideOk_spec ho'.2).2.1
  generalize done.getD unit default = u at *
  have hle := h.le
  -- the physical values, with `one` (past every row) at 1
  let v : ℕ → Bool := fun c => if c = u.rows.size then true else physical z c
  have hv : ∀ c ∈ order done u, v c = physical z c := fun c hc => by
    have := hlt c hc
    simp only [v]
    rw [ite_eq_right (by omega)]
  have hsat : SatAt (blockRow words u u.rows.size) v := by
    intro c r hr
    by_cases hc1 : c = u.rows.size
    · exfalso
      subst hc1
      have hf : u.delta.find? (·.1 == u.rows.size) = none := by
        rw [Array.find?_eq_none]
        intro e he hee
        have := hlt _ (hΔ e he)
        simp only [beq_iff_eq] at hee
        omega
      have hfr : fullRow words u u.rows.size = none := by
        unfold fullRow
        split <;> simp
      simp [blockRow, hf, hfr] at hr
    by_cases hm : c ∈ order done u
    · have hlc := List.idxOf_lt_length_iff.2 hm
      obtain ⟨ℓ, hℓ, rfl⟩ : ∃ ℓ, ℓ < (order done u).length ∧ (order done u).getD ℓ 0 = c :=
        ⟨(order done u).idxOf c, hlc, by rw [List.getD_eq_getElem _ _ hlc]; exact List.getElem_idxOf hlc⟩
      by_cases hin' : ℓ < u.inCols.size
      · have hcol := order_getD_inCols htake hin'
        rw [hcol] at hr ⊢
        rw [hins ℓ hin', Option.some.injEq] at hr
        subst hr
        simp [xorSum]
      · obtain ⟨k, rfl⟩ : ∃ k, ℓ = u.inCols.size + k := ⟨ℓ - u.inCols.size, by omega⟩
        have hk : k < (Rows.compose (ofBlock words done u) h).nComp := by
          show k < (order done u).length - u.inCols.size; omega
        rw [hv _ hm, physical_getD hn z hℓ]
        rcases ofBlock_cases words done u ((order done u).getD (u.inCols.size + k) 0) with ⟨hr', hb⟩ | ⟨p, hr', hb⟩ |
          ⟨hr', hb⟩
        · rw [hb, Option.some.injEq] at hr
          subst hr
          have := comp_one hz h1 ⟨k, hk⟩ hr'
          simp only [xorSum, List.map_cons, List.map_nil, List.foldr_cons, List.foldr_nil, v, ite_true,
            Bool.xor_false, Bool.and_self]
          exact this
        · rw [hb, Option.some.injEq] at hr
          subst hr
          have hmem := h.mem k hk p hr'
          have hzk := hz ⟨k, hk⟩
          rw [xorSum_physical z _ _ (by rw [Rows.compose_a, lrow_some hr'])
              (fun c hc => hmem c (List.mem_append_left _ hc)),
            xorSum_physical z _ _ (by rw [Rows.compose_b, lrow_some hr'])
              (fun c hc => hmem c (List.mem_append_right _ hc))] at hzk
          rw [xorSum_congr (fun c hc => hv c (hmem c (List.mem_append_left _ hc))),
            xorSum_congr (fun c hc => hv c (hmem c (List.mem_append_right _ hc)))]
          exact hzk
        · rw [hb] at hr; cases hr
    · rw [hzero c r hm hr]
      simp only [v]
      rw [ite_eq_right hc1, physical_not_mem z hm]
      rfl
  have hmain := hsound v hsat (by simp [v]) F hF o ho
  rw [hv _ hmo, physical_of_mem z hmo] at hmain
  have hb : lidx (ofBlock words done u) (u.outCols.getD o 0) < (Rows.compose (ofBlock words done u) h).w := by
    rw [Rows.compose_w]; exact Nat.lt_succ_of_lt (List.idxOf_lt_length_iff.2 hmo)
  rw [show ℓ = ⟨_, hb⟩ from Fin.ext hℓ, hmain]
  congr 1
  refine evalT_congr types words F t fun i hi => ?_
  have hi' : i < u.inCols.size := by omega
  have hlt' : i < (order done u).length := by
    have := congrArg List.length htake
    simp only [List.length_take, Array.length_toList] at this
    omega
  rw [← order_getD_inCols htake hi', hv _ (by rw [List.getD_eq_getElem _ _ hlt']; exact List.getElem_mem hlt'), physical_getD hn z hlt']
  exact (hx ⟨i, by rw [Rows.compose_w]; exact Nat.lt_succ_of_lt hlt'⟩ hi').symm

end unit

end FlockSoundness
