import Proofs.Core.TC.Proofs.Step

/-!
# Completeness: the honest witness satisfies every predicate

`honestGroup` is the group row `step_witness` builds (the same columns, computed from the rows' exponents and scaled
magnitudes), and `group_complete` checks each of the group's predicates on it, for any admissible accumulator and any
sixteen BF16 operand pairs.
-/

namespace Verity.TC.Proofs

open Verity.TC Verity.TC.Spec Verity.TC.Relation

/-- The bounds every row of a Hopper group satisfies. -/
structure RowOK (x : TermIn) : Prop where
  s0 : 0 ≤ x.scaled
  s1 : x.scaled < 2 ^ 27
  e0 : -252 ≤ x.e
  e1 : x.e ≤ 254
  n01 : x.neg = 0 ∨ x.neg = 1

/-- The alignment columns `step_witness` builds for a row. -/
def honestTerm (E : Int) (x : TermIn) : TermCols :=
  let dd := min (E - x.e).toNat 27
  let aligned := x.scaled / 2 ^ dd
  let r := x.scaled - aligned * 2 ^ dd
  { d := E - x.e, pow := 2 ^ dd, invpow := 2 ^ (27 - dd), aligned := aligned, r := r, rinv := r * 2 ^ (27 - dd),
    signed := x.neg * aligned }

theorem maxE_le (U : Int) : ∀ (xs : List TermIn) (m : Int), m ≤ U → (∀ x ∈ xs, x.e ≤ U) → maxE m xs ≤ U
  | [], _, hm, _ => hm
  | x :: xs, m, hm, h => by
    rw [maxE_cons]
    exact maxE_le U xs _ (by have := h x (List.mem_cons_self ..); omega) (fun y hy => h y (List.mem_cons_of_mem _ hy))

theorem chain_zero_of (E : Int) : ∀ (xs : List TermIn) (c : Int), (c = 0 ∨ ∃ x ∈ xs, E = x.e) →
    xs.foldl (fun c x => c * (E - x.e)) c = 0
  | [], c, h => by
    rcases h with h | ⟨x, hx, _⟩
    · exact h
    · cases hx
  | x :: xs, c, h => by
    simp only [List.foldl_cons]
    apply chain_zero_of E xs
    rcases h with h | ⟨y, hy, he⟩
    · left; rw [h, Int.zero_mul]
    · rcases List.mem_cons.mp hy with rfl | hy
      · left; rw [he, Int.sub_self, Int.mul_zero]
      · exact Or.inr ⟨y, hy, he⟩

theorem contrib_bound (E : Int) (x : TermIn) (hx : RowOK x) : -(2 ^ 27) < contrib E x ∧ contrib E x < 2 ^ 27 := by
  unfold contrib
  have hp := pow2_pos (min (E - x.e).toNat 27)
  have q0 : 0 ≤ x.scaled / 2 ^ (min (E - x.e).toNat 27) := Int.ediv_nonneg hx.s0 (Int.le_of_lt hp)
  have q1 : x.scaled / 2 ^ (min (E - x.e).toNat 27) ≤ x.scaled := Int.ediv_le_self _ hx.s0
  have := hx.s1
  generalize x.scaled / 2 ^ (min (E - x.e).toNat 27) = q at *
  rcases hx.n01 with h | h <;> rw [h] <;> omega

theorem sum_bound (E : Int) : ∀ (xs : List TermIn), (∀ x ∈ xs, RowOK x) →
    -(xs.length * 2 ^ 27 : Int) ≤ (xs.map (contrib E)).sum ∧ (xs.map (contrib E)).sum ≤ xs.length * 2 ^ 27
  | [], _ => by simp
  | x :: xs, h => by
    have ih := sum_bound E xs (fun y hy => h y (List.mem_cons_of_mem _ hy))
    have hb := contrib_bound E x (h x (List.mem_cons_self ..))
    simp only [List.map_cons, List.sum_cons, List.length_cons]
    push_cast
    constructor <;> omega

theorem take_succ_foldl {α β : Type} (f : β → α → β) (c : β) (l : List α) (i : Nat) (h : i < l.length) :
    (l.take (i + 1)).foldl f c = f ((l.take i).foldl f c) l[i] := by
  rw [List.take_succ, List.getElem?_eq_getElem h, Option.toList_some, List.foldl_append]
  rfl

theorem term_complete (align : Align) (E : Int) (i : Nat) (Pprev : Int) (x : TermIn) (hx : RowOK x)
    (hd0 : 0 ≤ E - x.e) (hd1 : E - x.e < 512)
    (htab : i ≠ 0 → x.scaled = Pprev * 2 ^ 11 ∧ 0 ≤ Pprev ∧ Pprev < 2 ^ 16) :
    termHolds HP align E i Pprev x (honestTerm E x) := by
  have hdd : min (E - x.e).toNat 27 ≤ 27 := Nat.min_le_right _ _
  have hp := pow2_pos (min (E - x.e).toNat 27)
  have hp' := pow2_pos (27 - min (E - x.e).toNat 27)
  have h27 : (2 : Int) ^ 27 = 2 ^ min (E - x.e).toNat 27 * 2 ^ (27 - min (E - x.e).toNat 27) := by
    rw [← Int.pow_add]; congr 1; omega
  refine ⟨?_, ?_, ?_, ?_⟩
  · simp [LIN, honestTerm]
  · unfold LOOKUP tPow honestTerm
    simp only [HP]
    rw [if_pos ⟨hd0, hd1⟩]
  · split
    · unfold honestTerm
      simp only [MUL, RANGE, HP]
      have q0 : 0 ≤ x.scaled / 2 ^ (min (E - x.e).toNat 27) := Int.ediv_nonneg hx.s0 (Int.le_of_lt hp)
      have q1 : x.scaled / 2 ^ (min (E - x.e).toNat 27) ≤ x.scaled := Int.ediv_le_self _ hx.s0
      have hr := Int.emod_def x.scaled (2 ^ (min (E - x.e).toNat 27))
      have hr0 := Int.emod_nonneg x.scaled (Int.ne_of_gt hp)
      have hr1 := Int.emod_lt_of_pos x.scaled hp
      have hrr : x.scaled - x.scaled / 2 ^ min (E - x.e).toNat 27 * 2 ^ min (E - x.e).toNat 27 =
          x.scaled % 2 ^ min (E - x.e).toNat 27 := by rw [hr, Int.mul_comm]
      rw [hrr]
      refine ⟨by omega, ⟨q0, by have := hx.s1; omega⟩, trivial, ⟨Int.mul_nonneg hr0 (Int.le_of_lt hp'), ?_⟩⟩
      rw [h27]
      exact Int.mul_lt_mul_of_pos_right hr1 hp'
    · rename_i hi
      obtain ⟨hsc, hP0, hP1⟩ := htab (fun e => hi (Or.inl e))
      unfold LOOKUP tShift honestTerm
      simp only [HP]
      rw [if_pos ⟨hP0, hP1, hd0, hd1⟩]
      congr 1
      rw [Nat.shiftLeft_eq, Nat.shiftRight_eq_div_pow, natCast_div_pow, hsc, Int.natCast_mul, Int.toNat_of_nonneg hP0]
      simp
  · simp [MUL, honestTerm]

/-- The group row `step_witness` builds for the input state `st` and the product rows `prods`. -/
def honestGroup (st : State) (prods : List ProdCols) : GroupCols :=
  let ins := terms HP st prods
  let E := maxE (-133) ins
  let tcs := ins.map (honestTerm E)
  let T := (tcs.map (fun t => t.aligned - 2 * t.signed)).sum
  let L := bitLength T.natAbs
  let neg_t : Int := if T < 0 then 1 else 0
  let mag_t : Int := T.natAbs
  let normpow : Int := 2 ^ (32 - L)
  let z_t : Int := if (L : Int) = 0 then 1 else 0
  let normB := mag_t * normpow
  let magW : Int := (magWOf T : Nat)
  let e_out : Int := E + L - 26
  let e_fin := max e_out (-126)
  let sh := min (e_fin - e_out + 2) 36
  let pow_s : Int := 2 ^ sh.toNat
  let invpow_s : Int := 2 ^ (36 - sh).toNat
  let ovf : Int := if e_fin > 127 then 1 else 0
  let mag24 := magW / pow_s
  let r_s := magW - mag24 * pow_s
  let z_out : Int := if mag24 = 0 then 1 else 0
  let pinf0 := st.inf * (1 - st.neg)
  let ninf0 := st.inf * st.neg
  let S_nan := st.nan + (prods.map (·.nan)).sum
  let S_pinf := pinf0 + (prods.map (·.pinf)).sum
  let S_ninf := ninf0 + (prods.map (·.ninf)).sum
  let any_nan : Int := if S_nan ≠ 0 then 1 else 0
  let any_pinf : Int := if S_pinf ≠ 0 then 1 else 0
  let any_ninf : Int := if S_ninf ≠ 0 then 1 else 0
  let ovf_eff := ovf * (1 - z_t)
  let any_inf := any_pinf + any_ninf - any_pinf * any_ninf
  let nan_out := any_nan + (1 - any_nan) * (any_pinf * any_ninf)
  let inf_out := (1 - nan_out) * (any_inf + (1 - any_inf) * ovf_eff)
  let neg_inf := any_inf * any_ninf + (1 - any_inf) * neg_t
  let fin := 1 - nan_out - inf_out
  let neg_fin := neg_t * (1 - z_out)
  let e_state := e_fin + z_out * (-133 - e_fin)
  { Estar := E, terms := tcs, total := T, neg_t := neg_t, mag_t := mag_t, L := L, normpow := normpow, z_t := z_t,
    normB := normB, magW := magW, r_n := normB - magW * 2 ^ 6, e_out := e_out, e_fin := e_fin, pow_s := pow_s,
    invpow_s := invpow_s, ovf := ovf, mag24 := mag24, r_s := r_s, rsinv := r_s * invpow_s, z_out := z_out,
    pinf0 := pinf0, ninf0 := ninf0, S_nan := S_nan, S_pinf := S_pinf, S_ninf := S_ninf, any_nan := any_nan,
    any_pinf := any_pinf, any_ninf := any_ninf, ovf_eff := ovf_eff, any_inf := any_inf, nan_out := nan_out,
    inf_out := inf_out, neg_inf := neg_inf, fin := fin, neg_fin := neg_fin, e_state := e_state,
    out := ⟨nan_out, inf_out, inf_out * neg_inf + fin * neg_fin, -133 + fin * (e_state - -133), fin * mag24⟩ }

theorem honest_contrib (E : Int) (x : TermIn) :
    (honestTerm E x).aligned - 2 * (honestTerm E x).signed = contrib E x := by
  unfold honestTerm contrib
  simp only
  rw [Int.sub_mul, Int.one_mul, Int.mul_assoc]

theorem any_lookup (n : Nat) (S : Int) (h0 : 0 ≤ S) (h1 : S ≤ n) : LOOKUP (tAny n) S (if S ≠ 0 then 1 else 0) := by
  unfold LOOKUP tAny
  rw [if_pos ⟨h0, by omega⟩]

theorem stateOf_M_lt (v : Value) (hv : Admissible v) : (stateOf (-133) v).M < 2 ^ 24 := by
  cases v with
  | nan => simp [stateOf]
  | inf n => simp [stateOf]
  | fin t =>
    simp only [Admissible] at hv
    simp only [stateOf]
    split
    · simp
    · have := hv.1; simp only; omega

theorem group_complete (align : Align) (v : Value) (hv : Admissible v) (ab : List (Nat × Nat))
    (hlen : ab.length = 16) (hw : ∀ x ∈ ab, x.1 < 2 ^ 16 ∧ x.2 < 2 ^ 16) :
    groupHolds HP align 16 (stateOf (-133) v) (ab.map (fun x => prodCols (-133) x.1 x.2))
        (honestGroup (stateOf (-133) v) (ab.map (fun x => prodCols (-133) x.1 x.2))) ∧
      (honestGroup (stateOf (-133) v) (ab.map (fun x => prodCols (-133) x.1 x.2))).terms.length = 17 := by
  have hok := prodOK_mem ab hw
  obtain ⟨eN, eP, eNg⟩ := prods_sum ab hok
  obtain ⟨fnan, fpinf, fninf, fM, fe1, fe2, fneg, _, _⟩ := stateOf_facts v hv
  have fM24 := stateOf_M_lt v hv
  generalize hst : stateOf (-133) v = st at *
  have hins : terms HP st (ab.map (fun x => prodCols (-133) x.1 x.2)) =
      ⟨st.neg, st.e, st.M * 2 ^ HP.rho⟩ :: (ab.map (fun x => prodCols (-133) x.1 x.2)).map prodIn := rfl
  -- every row is bounded
  have hrows : ∀ x ∈ terms HP st (ab.map (fun x => prodCols (-133) x.1 x.2)), RowOK x := by
    rw [hins]
    intro x hx
    rcases List.mem_cons.mp hx with rfl | hx
    · exact ⟨Int.mul_nonneg fM (by decide), by simp only [HP]; omega, by dsimp only; omega, by dsimp only; omega, fneg⟩
    · obtain ⟨q, hq, rfl⟩ := List.mem_map.mp hx
      obtain ⟨y, hy, rfl⟩ := List.mem_map.mp hq
      have h := hok y hy
      have := h.P_nonneg
      have := h.P_lt
      have := h.em_lo
      have := h.em_hi
      exact ⟨Int.mul_nonneg h.P_nonneg (by decide), by simp only [prodIn, HP]; omega, by simp only [prodIn]; omega,
        by simp only [prodIn]; omega, h.sp01⟩
  have hilen : (terms HP st (ab.map (fun x => prodCols (-133) x.1 x.2))).length = 17 := by rw [hins]; simp [hlen]
  -- the sums of the special flags
  have hS : ∀ (p : Value → Bool) (x0 : Int), x0 = ind p v →
      0 ≤ x0 + ((ab.map (fun x => bf16ProductTotal x.1 x.2)).map (ind p)).sum ∧
        x0 + ((ab.map (fun x => bf16ProductTotal x.1 x.2)).map (ind p)).sum ≤ ((16 + 1 : Nat) : Int) := by
    intro p x0 hx0
    have := ind_sum p (v :: ab.map (fun x => bf16ProductTotal x.1 x.2))
    simp only [List.map_cons, List.sum_cons, List.length_cons, List.length_map, hlen] at this
    rw [hx0]; constructor <;> omega
  have hSn := hS Value.isNaN _ fnan
  have hSp := hS (Value.isInf false) _ fpinf
  have hSng := hS (Value.isInf true) _ fninf
  rw [← eN] at hSn
  rw [← eP] at hSp
  rw [← eNg] at hSng
  refine ⟨?_, by simp [honestGroup, hilen]⟩
  unfold groupHolds honestGroup
  simp only []
  generalize hI : terms HP st (ab.map (fun x => prodCols (-133) x.1 x.2)) = ins at *
  have hE0 := (le_maxE ins (-133)).1
  have hE1 := (le_maxE ins (-133)).2
  have hE2 : maxE (-133) ins ≤ 254 := maxE_le 254 ins _ (by decide) (fun x hx => (hrows x hx).e1)
  have hE3 := maxE_attained ins (-133)
  generalize maxE (-133) ins = E at *
  have hTm : (List.map (fun t => t.aligned - 2 * t.signed) (List.map (honestTerm E) ins)).sum =
      (ins.map (contrib E)).sum := by
    rw [List.map_map]; congr 1; apply List.map_congr_left; intro x _; exact honest_contrib E x
  have hTb := sum_bound E ins hrows
  rw [hilen] at hTb
  rw [hTm]
  generalize (ins.map (contrib E)).sum = T at *
  have hT32 : T.natAbs < 2 ^ 32 := by push_cast at hTb; omega
  have hbl := bitLength_le T.natAbs 32 hT32
  -- the window, for a nonzero total
  have hwin : T ≠ 0 → 2 ^ 31 ≤ T.natAbs * 2 ^ (32 - bitLength T.natAbs) ∧
      T.natAbs * 2 ^ (32 - bitLength T.natAbs) < 2 ^ 32 ∧ 2 ^ 25 ≤ magWOf T ∧ magWOf T < 2 ^ 26 ∧
      1 ≤ bitLength T.natAbs := by
    intro hT0
    obtain ⟨_, h1, h2⟩ := window T.natAbs (by omega) hT32
    have := (bitLength_spec T.natAbs (by omega)).2.2
    unfold magWOf
    refine ⟨?_, ?_, h1, h2, this⟩
    · have := (Nat.le_div_iff_mul_le (Nat.two_pow_pos 6)).mp h1; omega
    · have := (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos 6)).mp h2; omega
  have hW0 : T = 0 → magWOf T = 0 ∧ bitLength T.natAbs = 0 := by
    intro h; subst h; simp [magWOf, bitLength]
  have hWlt : magWOf T < 2 ^ 26 := by
    by_cases hT0 : T = 0
    · rw [(hW0 hT0).1]; decide
    · exact (hwin hT0).2.2.2.1
  have hk2 : 2 ≤ (min (max (E + ↑(bitLength T.natAbs) - 26) (-126) - (E + ↑(bitLength T.natAbs) - 26) + 2) 36).toNat := by
    omega
  have hk36 : (min (max (E + ↑(bitLength T.natAbs) - 26) (-126) - (E + ↑(bitLength T.natAbs) - 26) + 2) 36).toNat ≤ 36 := by
    omega
  generalize hsh : (min (max (E + ↑(bitLength T.natAbs) - 26) (-126) - (E + ↑(bitLength T.natAbs) - 26) + 2) 36) = sh at *
  have hm24 : magWOf T / 2 ^ sh.toNat < 2 ^ 24 := by
    apply (Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _)).mpr
    have : 2 ^ 26 ≤ 2 ^ 24 * 2 ^ sh.toNat := by
      calc 2 ^ 26 = 2 ^ 24 * 2 ^ 2 := by decide
        _ ≤ _ := Nat.mul_le_mul_left _ (Nat.pow_le_pow_right (by decide) hk2)
    omega
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_,
    ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  all_goals (try simp only [MUL, LIN, BOOL, HP])
  all_goals (try omega)
  case refine_1 =>
    intro i hi hi'
    rw [List.getElem_map]
    apply term_complete align E i _ _ (hrows _ (List.getElem_mem hi))
    · have := hE1 _ (List.getElem_mem hi); omega
    · have := (hrows _ (List.getElem_mem hi)).e0; omega
    · intro hi0
      subst hI
      obtain ⟨j, rfl⟩ : ∃ j, i = j + 1 := ⟨i - 1, by omega⟩
      have hj : j < ab.length := by rw [hilen] at hi; omega
      have hP := (hok _ (List.getElem_mem hj)).P_nonneg
      have hP1 := (hok _ (List.getElem_mem hj)).P_lt
      obtain ⟨e1, e2⟩ := tab_fact ab ⟨st.neg, st.e, st.M * 2 ^ HP.rho⟩ j hj hP (by rw [← hins]; exact hi)
      refine ⟨e1, e2, ?_⟩
      rw [if_neg (Nat.succ_ne_zero j), Nat.add_sub_cancel]
      have e : (ab.map (fun x => prodCols (-133) x.1 x.2))[j]!.P = (prodCols (-133) ab[j].1 ab[j].2).P := by
        simp [hj]
      rw [e]; exact hP1
  case refine_2 =>
    intro i hi
    exact (take_succ_foldl (fun c x => c * (E - x.e)) (E - -133) ins i hi).symm
  case refine_3 =>
    apply chain_zero_of
    rcases hE3 with h | ⟨x, hx, h⟩
    · left; omega
    · exact Or.inr ⟨x, hx, h⟩
  case refine_5 => split <;> simp
  case refine_6 => split <;> omega
  case refine_7 => simp only [RANGE, HP]; omega
  case refine_8 =>
    unfold LOOKUP tNorm
    simp only [HP]
    rw [if_pos ⟨by omega, by omega⟩]
    simp
  case refine_10 =>
    simp only [RANGE, HP]
    by_cases hT0 : T = 0
    · obtain ⟨_, h2⟩ := hW0 hT0
      subst hT0; simp [bitLength]
    · obtain ⟨h1, h2, _, _, h5⟩ := hwin hT0
      have hne : ¬ ((bitLength T.natAbs : Nat) : Int) = 0 := by omega
      simp only [hne, ite_false]
      have hc : (T.natAbs : Int) * 2 ^ (32 - bitLength T.natAbs) = ((T.natAbs * 2 ^ (32 - bitLength T.natAbs) : Nat) : Int) := by
        simp
      rw [hc]; omega
  case refine_12 =>
    simp only [RANGE, HP]
    have hc : (T.natAbs : Int) * 2 ^ (32 - bitLength T.natAbs) = ((T.natAbs * 2 ^ (32 - bitLength T.natAbs) : Nat) : Int) := by
      simp
    rw [hc]
    unfold magWOf
    generalize T.natAbs * 2 ^ (32 - bitLength T.natAbs) = N
    have := Nat.div_add_mod N (2 ^ 6)
    have := Nat.mod_lt N (Nat.two_pow_pos 6)
    omega
  case refine_14 =>
    unfold LOOKUP tExp
    rw [if_pos ⟨by simp only [E_OFF]; omega, by simp only [E_OFF]; omega⟩]
    simp [E_OFF, FP32_MIN_E, FP32_MAX_E, HP, hsh]
  case refine_16 =>
    simp only [RANGE]
    rw [← natCast_div_pow]
    generalize magWOf T / 2 ^ sh.toNat = q at hm24 ⊢
    omega
  case refine_18 =>
    simp only [RANGE]
    have hp := pow2_pos sh.toNat
    have hp' := pow2_pos (36 - sh).toNat
    have hr : ((magWOf T : Nat) : Int) - ((magWOf T : Nat) : Int) / 2 ^ sh.toNat * 2 ^ sh.toNat =
        ((magWOf T : Nat) : Int) % 2 ^ sh.toNat := by
      rw [Int.emod_def, Int.mul_comm]
    rw [hr]
    have hr0 := Int.emod_nonneg ((magWOf T : Nat) : Int) (Int.ne_of_gt hp)
    have hr1 := Int.emod_lt_of_pos ((magWOf T : Nat) : Int) hp
    have h36 : (2 : Int) ^ 36 = 2 ^ sh.toNat * 2 ^ (36 - sh).toNat := by rw [← Int.pow_add]; congr 1; omega
    exact ⟨Int.mul_nonneg hr0 (Int.le_of_lt hp'), by rw [h36]; exact Int.mul_lt_mul_of_pos_right hr1 hp'⟩
  case refine_19 =>
    refine ⟨?_, rfl⟩
    simp only [RANGE]
    rw [← natCast_div_pow]
    generalize magWOf T / 2 ^ sh.toNat = q at hm24 ⊢
    omega
  case refine_25 => exact any_lookup _ _ hSn.1 hSn.2
  case refine_26 => exact any_lookup _ _ hSp.1 hSp.2
  case refine_27 => exact any_lookup _ _ hSng.1 hSng.2

end Verity.TC.Proofs
