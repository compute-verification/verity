import Pouw.Protocol.PearlC.TileCircuit
import Pouw.Protocol.PearlC.PeelExactP
import Pouw.SecurityProofs.PearlC.AccumProofs
import Verity.SecurityProofs.Circuit

/-!
# The tile unit computes `C̃` and `U` (proofs)

`TileCircuit.tileSpec p K`, the tile template at `TM = TN = 1`, evaluates to the pinned order's words: on inputs holding
the codes of `A′_i` and `B̃_j` and the factor rows `P_A,i` and `P_B,j`, its two outputs are `chainWordP? p 4` over the
row's atoms and `uWord?` from it (`tile_words`), at every depth `K = 128·nw` (sm_120's `K = 8192` among them). A
transcript whose one unit is correct carries them (`tile_correct`, by core's `Verity.Partition.compose`). This is L3's
step 4 for the tile template.

* **Per gate** (`tileVal_atom`, `tileVal_add`, `tileVal_peel`, `tileVal_input`): each gate's value from the gates it reads.
* **A window** (`tileVal_window`): its four atoms write the window's word, a step word (`AccumProofs.StepWord`).
* **The promotions** (`tileVal_chain`): the template adds each window's word into the total by `F32Add_v2`, which does
  not read `−0` as `+0` as `promoteWord?` does. They agree because a sum of two step words is never `−0`
  (`addWord_step`): FP32 values lie on the grid `2^−149`, so a nonzero sum is at least the least subnormal, which
  `f32Word` writes as a nonzero word (`f32Word_ne_negZero`), and an exact zero sum is `−0` only for `(−0) + (−0)`.
* **U** (`tileVal_U`): the four K16 steps from `C̃`'s word.
-/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact Pouw.PearlC.PeelExactP

theorem val_of_lt {n : ℕ} {β : Fin n → ℕ} (v : (a : Fin n) → BitVec (β a)) {g : ℕ} (hg : g < n) :
    val v g = (v ⟨g, hg⟩).toNat := by
  rw [val, dite_eq_left_of_eq_true (eq_true hg)]

namespace Spec

variable (S : Spec)

theorem eval_gate (a : S.toCircuit.Transcript) (g : Fin S.N) (hg : (S.gate g).input = false) :
    S.toCircuit.eval a g = BitVec.ofNat _ ((S.gate g).f ((S.gate g).rd.map (val (S.toCircuit.eval a)))) :=
  S.toCircuit.eval_op a hg

theorem val_input (a : S.toCircuit.Transcript) {g : ℕ} (hg : g < S.N) (hi : (S.gate g).input = true) :
    val (S.toCircuit.eval a) g = (a ⟨g, hg⟩).toNat := by
  rw [val_of_lt _ hg]
  exact congrArg BitVec.toNat (S.toCircuit.eval_input a (g := (⟨g, hg⟩ : Fin S.toCircuit.N)) hi)

theorem val_gate (a : S.toCircuit.Transcript) {g : ℕ} (hg : g < S.N) (hi : (S.gate g).input = false) :
    val (S.toCircuit.eval a) g =
      (S.gate g).f ((S.gate g).rd.map (val (S.toCircuit.eval a))) % 2 ^ (S.gate g).w := by
  rw [val_of_lt _ hg, S.eval_gate a ⟨g, hg⟩ hi, BitVec.toNat_ofNat]

end Spec

section Gates

variable (p : Pipeline) (K : ℕ)

theorem gate_input {g : ℕ} (hg : g < 2 * K + 128) : (tileGate p K g).input = true := by
  unfold tileGate
  split_ifs <;> first | omega | rfl

theorem gate_atom {q r : ℕ} (hq : q < K / 128) (hr : r < 4) :
    tileGate p K (2 * K + 128 + (5 * q + r)) =
      ⟨32, false, (if r = 0 then [] else [2 * K + 128 + (5 * q + r) - 1]) ++
          (List.range 32).map (32 * (4 * q + r) + ·) ++ (List.range 32).map (K + 32 * (4 * q + r) + ·),
        atomF p (r == 0)⟩ := by
  have hj : 2 * K + 128 + (5 * q + r) - (2 * K + 128) = 5 * q + r := by omega
  have h5 : (5 * q + r) % 5 = r := by omega
  have h5' : (5 * q + r) / 5 = q := by omega
  unfold tileGate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), hj, ite_eq_left (by omega), h5, ite_eq_left hr, h5']

theorem gate_add {q : ℕ} (hq : q < K / 128) :
    tileGate p K (2 * K + 128 + (5 * q + 4)) =
      ⟨32, false, (if q = 0 then [] else [2 * K + 128 + (5 * q + 4) - 5]) ++ [2 * K + 128 + (5 * q + 4) - 1],
        addF (q == 0)⟩ := by
  have hj : 2 * K + 128 + (5 * q + 4) - (2 * K + 128) = 5 * q + 4 := by omega
  have h5 : (5 * q + 4) % 5 = 4 := by omega
  have h5' : (5 * q + 4) / 5 = q := by omega
  unfold tileGate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), hj, ite_eq_left (by omega), h5, ite_eq_right (by omega), h5']

theorem gate_peel {s : ℕ} :
    tileGate p K (2 * K + 128 + 5 * (K / 128) + s) =
      ⟨32, false, [2 * K + 128 + 5 * (K / 128) + s - 1] ++ (List.range 16).map (2 * K + 16 * s + ·) ++
          (List.range 16).map (2 * K + 64 + 16 * s + ·), peelF⟩ := by
  have hj : 2 * K + 128 + 5 * (K / 128) + s - (2 * K + 128) - 5 * (K / 128) = s := by omega
  unfold tileGate
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), hj]

theorem getD_pair {n : ℕ} (f h V : ℕ → ℕ) {i : ℕ} (hi : i < n) :
    (((List.range n).map f ++ (List.range n).map h).map V).getD i 0 = V (f i) ∧
      (((List.range n).map f ++ (List.range n).map h).map V).getD (n + i) 0 = V (h i) := by
  simp [List.getD_eq_getElem?_getD, List.getElem?_append_left, hi]

/-- The circuit's values. -/
abbrev tileVal (a : (tileSpec p K).toCircuit.Transcript) : ℕ → ℕ := val ((tileSpec p K).toCircuit.eval a)

theorem tileVal_atom (a : (tileSpec p K).toCircuit.Transcript) {q r : ℕ} (hq : q < K / 128) (hr : r < 4) :
    tileVal p K a (2 * K + 128 + (5 * q + r)) =
      p.step (if r = 0 then 0 else tileVal p K a (2 * K + 128 + (5 * q + r) - 1))
        (List.ofFn fun i : Fin 32 => code (tileVal p K a (32 * (4 * q + r) + i)))
        (List.ofFn fun i : Fin 32 => code (tileVal p K a (K + 32 * (4 * q + r) + i))) := by
  have hN : 2 * K + 128 + (5 * q + r) < (tileSpec p K).N := by
    show _ < tileN K
    unfold tileN; omega
  have hin : ((tileSpec p K).gate (2 * K + 128 + (5 * q + r))).input = false := by
    show (tileGate p K _).input = false
    rw [gate_atom p K hq hr]
  rw [tileVal, Spec.val_gate _ a hN hin]
  show (tileGate p K _).f ((tileGate p K _).rd.map _) % 2 ^ (tileGate p K _).w = _
  rw [gate_atom p K hq hr]
  dsimp only
  rw [Nat.mod_eq_of_lt (by unfold atomF; exact AccumProofs.step_lt _ _ _ _)]
  rcases Nat.eq_zero_or_pos r with rfl | h0
  · simp only [atomF, beq_self_eq_true, ↓reduceIte, List.nil_append]
    congr 2 <;> funext i
    · rw [(getD_pair _ _ _ i.isLt).1]
    · rw [(getD_pair _ _ _ i.isLt).2]
  · have hb : (r == 0) = false := by simp; omega
    simp only [atomF, hb, Bool.false_eq_true, ↓reduceIte, ite_eq_right (show r ≠ 0 by omega),
      List.cons_append, List.nil_append, List.map_cons, List.headD_cons, List.tail_cons]
    congr 2 <;> funext i
    · rw [(getD_pair _ _ _ i.isLt).1]
    · rw [(getD_pair _ _ _ i.isLt).2]

theorem tileVal_input (a : (tileSpec p K).toCircuit.Transcript) {g : ℕ} (hg : g < 2 * K + 128) :
    tileVal p K a g = val a g := by
  have hN : g < (tileSpec p K).N := by show g < tileN K; unfold tileN; omega
  rw [tileVal, Spec.val_input _ a hN (gate_input p K hg), val_of_lt _ hN]

theorem tileVal_add (a : (tileSpec p K).toCircuit.Transcript) {q : ℕ} (hq : q < K / 128) :
    tileVal p K a (2 * K + 128 + (5 * q + 4)) =
      f32Add (if q = 0 then 0 else tileVal p K a (2 * K + 128 + (5 * q + 4) - 5)) (tileVal p K a (2 * K + 128 + (5 * q + 4) - 1)) %
        2 ^ 32 := by
  have hN : 2 * K + 128 + (5 * q + 4) < (tileSpec p K).N := by
    show _ < tileN K
    unfold tileN; omega
  have hin : ((tileSpec p K).gate (2 * K + 128 + (5 * q + 4))).input = false := by
    show (tileGate p K _).input = false
    rw [gate_add p K hq]
  rw [tileVal, Spec.val_gate _ a hN hin]
  show (tileGate p K _).f ((tileGate p K _).rd.map _) % 2 ^ (tileGate p K _).w = _
  rw [gate_add p K hq]
  dsimp only
  rcases Nat.eq_zero_or_pos q with rfl | h0
  · simp [addF]
  · have hb : (q == 0) = false := by simp; omega
    simp [addF, hb, ite_eq_right (show q ≠ 0 by omega)]

theorem tileVal_peel (a : (tileSpec p K).toCircuit.Transcript) {s : ℕ} (hs : s < 4) :
    tileVal p K a (2 * K + 128 + 5 * (K / 128) + s) =
      k16Step (tileVal p K a (2 * K + 128 + 5 * (K / 128) + s - 1)) (fun l => tileVal p K a (2 * K + 16 * s + l))
        (fun l => tileVal p K a (2 * K + 64 + 16 * s + l)) := by
  have hN : 2 * K + 128 + 5 * (K / 128) + s < (tileSpec p K).N := by
    show _ < tileN K
    unfold tileN; omega
  have hin : ((tileSpec p K).gate (2 * K + 128 + 5 * (K / 128) + s)).input = false := by
    show (tileGate p K _).input = false
    rw [gate_peel p K]
  rw [tileVal, Spec.val_gate _ a hN hin]
  show (tileGate p K _).f ((tileGate p K _).rd.map _) % 2 ^ (tileGate p K _).w = _
  rw [gate_peel p K]
  dsimp only
  rw [Nat.mod_eq_of_lt (by unfold peelF k16Step; exact AccumProofs.packValue_lt _)]
  simp only [peelF, List.cons_append, List.nil_append, List.map_cons, List.headD_cons]
  congr 1 <;> funext l
  · rw [Nat.add_comm 1, List.getD_cons_succ, (getD_pair _ _ _ l.isLt).1]
  · rw [show 17 + (l : ℕ) = (16 + l) + 1 by omega, List.getD_cons_succ, (getD_pair _ _ _ l.isLt).2]

theorem code_val (c : Code) : code c.val = c := Fin.ext (Nat.mod_eq_of_lt c.isLt)

/-- The atoms' inputs: atom `t`'s codes are positions `32t, …` of the two rows. -/
theorem atom_inputs (a : (tileSpec p K).toCircuit.Transcript) (A B : Fin K → Code)
    (hA : ∀ i : Fin K, val a i = (A i).val) (hB : ∀ i : Fin K, val a (K + i) = (B i).val) {t : ℕ}
    (ht : t < (rowAtoms A B).length) :
    (List.ofFn fun i : Fin 32 => code (tileVal p K a (32 * t + i))) = List.ofFn (rowAtoms A B)[t].1 ∧
      (List.ofFn fun i : Fin 32 => code (tileVal p K a (K + 32 * t + i))) = List.ofFn (rowAtoms A B)[t].2 := by
  have hlen : (rowAtoms A B).length = K / 32 := List.length_ofFn
  rw [hlen] at ht
  have hk : 32 * t + 32 ≤ K := by have := Nat.mul_div_le K 32; omega
  simp only [rowAtoms, List.getElem_ofFn]
  constructor <;> congr 1 <;> funext i
  · rw [tileVal_input p K a (by omega), hA ⟨32 * t + i, by omega⟩, code_val]
  · rw [tileVal_input p K a (by omega), Nat.add_assoc, hB ⟨32 * t + i, by omega⟩, code_val]

theorem take_drop_four {α : Type} (l : List α) {n : ℕ} (h : n + 3 < l.length) :
    (l.drop n).take 4 = [l[n], l[n + 1], l[n + 2], l[n + 3]] := by
  have h0 : n < l.length := by omega
  have h1 : n + 1 < l.length := by omega
  have h2 : n + 2 < l.length := by omega
  rw [List.drop_eq_getElem_cons h0, List.drop_eq_getElem_cons h1, List.drop_eq_getElem_cons h2,
    List.drop_eq_getElem_cons h]
  rfl

/-- **A window's word**: its four atoms' gates write the window's word, a step word. -/
theorem tileVal_window (a : (tileSpec p K).toCircuit.Transcript) (A B : Fin K → Code)
    (hA : ∀ i : Fin K, val a i = (A i).val) (hB : ∀ i : Fin K, val a (K + i) = (B i).val) {q : ℕ}
    (hq : q < K / 128) {w : ℕ} (hw : windowWordP? p (((rowAtoms A B).drop (4 * q)).take 4) = some w) :
    tileVal p K a (2 * K + 128 + (5 * q + 3)) = w ∧ AccumProofs.StepWord w := by
  have hlen : (rowAtoms A B).length = K / 32 := List.length_ofFn
  have h3 : 4 * q + 3 < (rowAtoms A B).length := by rw [hlen]; omega
  rw [take_drop_four _ h3] at hw
  simp only [windowWordP?, List.foldlM_cons, List.foldlM_nil] at hw
  obtain ⟨w1, h1, hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain ⟨w2, h2, hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain ⟨w3, h3', hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain ⟨w4, h4, hw⟩ := Option.bind_eq_some_iff.1 hw
  obtain rfl : w4 = w := Option.some.inj hw
  have ai := fun t (ht : t < (rowAtoms A B).length) => atom_inputs p K a A B hA hB ht
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h1
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h2
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h3'
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h4
  have e0 : tileVal p K a (2 * K + 128 + (5 * q + 0)) = p.step 0 (List.ofFn (rowAtoms A B)[4 * q].1)
      (List.ofFn (rowAtoms A B)[4 * q].2) := by
    rw [tileVal_atom p K a hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  have e1 : tileVal p K a (2 * K + 128 + (5 * q + 1)) = p.step (tileVal p K a (2 * K + 128 + (5 * q + 0)))
      (List.ofFn (rowAtoms A B)[4 * q + 1].1) (List.ofFn (rowAtoms A B)[4 * q + 1].2) := by
    rw [tileVal_atom p K a hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  have e2 : tileVal p K a (2 * K + 128 + (5 * q + 2)) = p.step (tileVal p K a (2 * K + 128 + (5 * q + 1)))
      (List.ofFn (rowAtoms A B)[4 * q + 2].1) (List.ofFn (rowAtoms A B)[4 * q + 2].2) := by
    rw [tileVal_atom p K a hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  have e3 : tileVal p K a (2 * K + 128 + (5 * q + 3)) = p.step (tileVal p K a (2 * K + 128 + (5 * q + 2)))
      (List.ofFn (rowAtoms A B)[4 * q + 3].1) (List.ofFn (rowAtoms A B)[4 * q + 3].2) := by
    rw [tileVal_atom p K a hq (by omega), ← (ai _ (by omega)).1, ← (ai _ (by omega)).2]; rfl
  refine ⟨by rw [e3, e2, e1, e0], AccumProofs.step_lt _ _ _ _, AccumProofs.step_ne_negZero _ _ _ _, h4⟩

/-- **U's gate**: from `C̃`'s word `c` on `C̃`'s gate, the four K16 steps over the factor rows write `uWord? c pa pb`. -/
theorem tileVal_U (a : (tileSpec p K).toCircuit.Transcript) (pa pb : List ℕ)
    (hpa : ∀ i < 64, val a (2 * K + i) = pa.getD i 0) (hpb : ∀ i < 64, val a (2 * K + 64 + i) = pb.getD i 0)
    {c u : ℕ} (hc : tileVal p K a (cGate K) = c) (hu : uWord? c pa pb = some u) : tileVal p K a (uGate K) = u := by
  have sl : ∀ s < 4, (fun l : Fin 16 => tileVal p K a (2 * K + 16 * s + l)) = slice16 pa s ∧
      (fun l : Fin 16 => tileVal p K a (2 * K + 64 + 16 * s + l)) = slice16 pb s := fun s hs => by
    constructor <;> funext l
    · rw [tileVal_input p K a (by omega), Nat.add_assoc, hpa _ (by omega)]; rfl
    · rw [tileVal_input p K a (by omega), Nat.add_assoc, hpb _ (by omega)]; rfl
  rw [uWord?, show List.range 4 = [0, 1, 2, 3] from rfl] at hu
  simp only [List.foldlM_cons, List.foldlM_nil] at hu
  obtain ⟨u1, h1, hu⟩ := Option.bind_eq_some_iff.1 hu
  obtain ⟨u2, h2, hu⟩ := Option.bind_eq_some_iff.1 hu
  obtain ⟨u3, h3, hu⟩ := Option.bind_eq_some_iff.1 hu
  obtain ⟨u4, h4, hu⟩ := Option.bind_eq_some_iff.1 hu
  obtain rfl : u4 = u := Option.some.inj hu
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h1
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h2
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h3
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h4
  have e : ∀ s < 4, tileVal p K a (2 * K + 128 + 5 * (K / 128) + s) =
      k16Step (tileVal p K a (2 * K + 128 + 5 * (K / 128) + s - 1)) (slice16 pa s) (slice16 pb s) := fun s hs => by
    rw [tileVal_peel p K a hs, (sl s hs).1, (sl s hs).2]
  have e0 := e 0 (by omega)
  have e1 := e 1 (by omega)
  have e2 := e 2 (by omega)
  have e3 := e 3 (by omega)
  rw [show 2 * K + 128 + 5 * (K / 128) + 0 - 1 = cGate K by unfold cGate; omega, hc] at e0
  rw [show 2 * K + 128 + 5 * (K / 128) + 1 - 1 = 2 * K + 128 + 5 * (K / 128) + 0 by omega, e0] at e1
  rw [show 2 * K + 128 + 5 * (K / 128) + 2 - 1 = 2 * K + 128 + 5 * (K / 128) + 1 by omega, e1] at e2
  rw [show 2 * K + 128 + 5 * (K / 128) + 3 - 1 = 2 * K + 128 + 5 * (K / 128) + 2 by omega, e2] at e3
  rw [show uGate K = 2 * K + 128 + 5 * (K / 128) + 3 by unfold uGate tileN; omega, e3]

end Gates

/-! ## The promotion on words -/

section AddWord

open Pouw.Fp8Atom.Proofs AccumProofs

theorem f32Word_N {a : ℚ} (ha : 0 < a) :
    (roundEven (a / 2 ^ (max (binade a) (-126) - 23))).toNat ≤ 2 ^ 24 := by
  obtain ⟨-, hb2⟩ := binade_spec ha
  have hu : (0 : ℚ) < 2 ^ (max (binade a) (-126) - 23) := two_zpow_pos _
  have hlt : a / 2 ^ (max (binade a) (-126) - 23) < 2 ^ 24 := by
    rw [div_lt_iff₀ hu]
    calc a < 2 ^ (binade a + 1) := hb2
      _ ≤ 2 ^ (max (binade a) (-126) + 1) := zpow_le_zpow_right₀ one_le_two (by omega)
      _ = 2 ^ 24 * 2 ^ (max (binade a) (-126) - 23) := (two_pow_24 _).symm
  have h1 := roundEven_le (a / 2 ^ (max (binade a) (-126) - 23))
  have h2 : ⌊a / 2 ^ (max (binade a) (-126) - 23)⌋ < 2 ^ 24 := Int.floor_lt.2 (by exact_mod_cast hlt)
  omega

theorem f32Word_N_pos {a : ℚ} (ha : 2 ^ (-149 : ℤ) ≤ a) :
    1 ≤ (roundEven (a / 2 ^ (max (binade a) (-126) - 23))).toNat := by
  have ha0 : 0 < a := lt_of_lt_of_le (two_zpow_pos _) ha
  obtain ⟨hb1, -⟩ := binade_spec ha0
  have hu : (0 : ℚ) < 2 ^ (max (binade a) (-126) - 23) := two_zpow_pos _
  have hge : 1 ≤ a / 2 ^ (max (binade a) (-126) - 23) := by
    rw [le_div_iff₀ hu, one_mul]
    rcases le_or_gt (-126) (binade a) with h | h
    · rw [max_eq_left h]
      exact le_trans (zpow_le_zpow_right₀ one_le_two (by omega)) hb1
    · rw [max_eq_right h.le]
      exact ha
  have h1 := floor_le_roundEven (a / 2 ^ (max (binade a) (-126) - 23))
  have h2 : 1 ≤ ⌊a / 2 ^ (max (binade a) (-126) - 23)⌋ := Int.le_floor.2 (by exact_mod_cast hge)
  omega

/-- `f32Word`'s words are 32-bit. -/
theorem f32Word_lt (y : ℚ) : f32Word y < 2 ^ 32 := by
  unfold f32Word
  dsimp only
  by_cases h0 : |y| = 0
  · rw [ite_eq_left h0]; norm_num
  rw [ite_eq_right h0]
  have hN := f32Word_N (lt_of_le_of_ne (abs_nonneg y) (Ne.symm h0))
  generalize (roundEven (|y| / 2 ^ (max (binade |y|) (-126) - 23))).toNat = N₀ at hN ⊢
  generalize max (binade |y|) (-126) = E₀
  split_ifs <;> omega

/-- `f32Word` writes `−0` only below the least subnormal. -/
theorem f32Word_ne_negZero {y : ℚ} (hy : 2 ^ (-149 : ℤ) ≤ |y|) : f32Word y ≠ 0x80000000 := by
  have ha : 0 < |y| := lt_of_lt_of_le (two_zpow_pos _) hy
  unfold f32Word
  dsimp only
  rw [ite_eq_right ha.ne']
  have hN := f32Word_N ha
  have hN1 := f32Word_N_pos hy
  have hE : -126 ≤ max (binade |y|) (-126) := le_max_right _ _
  generalize (roundEven (|y| / 2 ^ (max (binade |y|) (-126) - 23))).toNat = N₀ at hN hN1 ⊢
  generalize max (binade |y|) (-126) = E₀ at hE ⊢
  split_ifs <;> omega

theorem wordQ_grid {w : ℕ} {x : ℚ} (h : wordQ w = some x) : ∃ m : ℤ, x = m * 2 ^ (-149 : ℤ) := by
  have he : (w >>> 23) % 256 ≠ 255 := by
    intro he
    unfold wordQ fp32Value at h
    simp only [he, ↓reduceIte] at h
    split_ifs at h <;> simp [Value.finite?] at h
  rw [PeelExactProofs.wordQ_fields he] at h
  obtain rfl := Option.some.inj h
  have hk : 1 ≤ max ((w >>> 23) % 256) 1 := le_max_right _ _
  generalize max ((w >>> 23) % 256) 1 = k at hk ⊢
  obtain ⟨j, rfl⟩ : ∃ j, k = j + 1 := ⟨k - 1, by omega⟩
  generalize ((if (w >>> 23) % 256 = 0 then 0 else 2 ^ 23) + w % 2 ^ 23 : ℕ) = N
  refine ⟨(if (w >>> 31) % 2 = 1 then -1 else 1) * ((N : ℤ) * 2 ^ j), ?_⟩
  rw [show ((j + 1 : ℕ) : ℤ) - 150 = (j : ℤ) + (-149) by push_cast; ring, zpow_add₀ two_ne_zero, zpow_natCast]
  split_ifs <;> push_cast <;> ring

theorem le_abs_add {x y : ℚ} (hx : ∃ m : ℤ, x = m * 2 ^ (-149 : ℤ)) (hy : ∃ m : ℤ, y = m * 2 ^ (-149 : ℤ))
    (h : x + y ≠ 0) : 2 ^ (-149 : ℤ) ≤ |x + y| := by
  obtain ⟨m, rfl⟩ := hx
  obtain ⟨n, rfl⟩ := hy
  rw [← add_mul, abs_mul, abs_of_pos (two_zpow_pos _)]
  have hmn : m + n ≠ 0 := by
    rintro hmn
    apply h
    rw [← add_mul, show ((m : ℚ) + n) = ((m + n : ℤ) : ℚ) by push_cast; ring, hmn]
    simp
  have h1 : (1 : ℚ) ≤ |((m + n : ℤ) : ℚ)| := by exact_mod_cast Int.one_le_abs hmn
  push_cast at h1
  nlinarith [two_zpow_pos (-149)]

/-- **One FP32 add of step words is a step word**: a sum of two finite words is `−0` only for `(−0) + (−0)`, and a
step word is never `−0`. -/
theorem addWord_step {t w s : ℕ} (ht : StepWord t) (h : addWord? t w = some s) : StepWord s := by
  unfold addWord? at h
  rcases hx : wordQ t with _ | x
  · rw [hx] at h; simp at h
  rcases hy : wordQ w with _ | y
  · rw [hx, hy] at h; simp at h
  rw [hx, hy] at h
  dsimp only at h
  obtain ⟨rfl, -⟩ := PeelExactProofs.finite?_eq_some h
  refine ⟨?_, ?_, h⟩
  · unfold rnWord
    split_ifs
    · norm_num
    · norm_num
    · exact f32Word_lt _
  · unfold rnWord
    by_cases h0 : x + y = 0
    · rw [ite_eq_left h0]
      have : signBit t = false ∨ x ≠ 0 := by
        by_cases hx0 : x = 0
        · left; rw [wordQ_zero_word ht (hx0 ▸ hx)]; decide
        · exact Or.inr hx0
      rcases this with hs | hs
      · simp [hs]
      · simp [hs]
    · rw [ite_eq_right h0]
      exact f32Word_ne_negZero (le_abs_add (wordQ_grid hx) (wordQ_grid hy) h0)

theorem promote_step {t w c : ℕ} (ht : StepWord t) (h : promoteWord? t w = some c) :
    addWord? t w = some c ∧ StepWord c := by
  unfold promoteWord? at h
  obtain ⟨s, hs, rfl⟩ := Option.map_eq_some_iff.1 h
  have hst := addWord_step ht hs
  rw [ite_eq_right hst.2.1]
  exact ⟨hs, hst⟩

theorem f32Add_of_addWord {a b s : ℕ} (h : addWord? a b = some s) : f32Add a b = s := by
  unfold addWord? at h
  rcases hx : wordQ a with _ | x
  · rw [hx] at h; simp at h
  rcases hy : wordQ b with _ | y
  · rw [hx, hy] at h; simp at h
  rw [hx, hy] at h
  simp only [f32Add, hx, hy]
  exact (PeelExactProofs.finite?_eq_some h).1.symm

end AddWord

section Chain

open AccumProofs

variable (p : Pipeline)

/-- **`C̃`'s gate** at depth `128·nw`: the windows' promotions write the chain's word. -/
theorem tileVal_chain {nw : ℕ} (hnw : 0 < nw) (a : (tileSpec p (128 * nw)).toCircuit.Transcript)
    (A B : Fin (128 * nw) → Code) (hA : ∀ i : Fin (128 * nw), val a i = (A i).val)
    (hB : ∀ i : Fin (128 * nw), val a (128 * nw + i) = (B i).val) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms A B) = some c) : tileVal p (128 * nw) a (cGate (128 * nw)) = c := by
  have hK : 128 * nw / 128 = nw := by omega
  have hlen : (rowAtoms A B).length = 4 * nw := by
    rw [show (rowAtoms A B).length = 128 * nw / 32 from List.length_ofFn]; omega
  set G : ℕ → ℕ → Option ℕ := fun t q => (windowWordP? p (((rowAtoms A B).drop (4 * q)).take 4)).bind
    (promoteWord? t) with hG
  have hch : chainWordP? p 4 (rowAtoms A B) = (List.range nw).foldlM G 0 := by
    rw [chainWordP?, Freeze.toChunks_eq (by norm_num), hlen, show (4 * nw + 4 - 1) / 4 = nw by omega,
      List.foldlM_map]
  have key : ∀ n ≤ nw, ∀ c, (List.range n).foldlM G 0 = some c →
      StepWord c ∧ (if n = 0 then 0 else tileVal p (128 * nw) a (2 * (128 * nw) + 128 + (5 * (n - 1) + 4))) = c := by
    intro n
    induction n with
    | zero =>
      intro _ c h
      simp only [List.range_zero, List.foldlM_nil, pure, Option.some.injEq] at h
      subst h
      exact ⟨⟨by norm_num, by decide, by decide⟩, rfl⟩
    | succ n ih =>
      intro hn c h
      rw [List.range_succ, List.foldlM_append] at h
      obtain ⟨t, ht, h⟩ := Option.bind_eq_some_iff.1 h
      simp only [List.foldlM_cons, List.foldlM_nil, hG] at h
      obtain ⟨c', hc', h⟩ := Option.bind_eq_some_iff.1 h
      obtain rfl : c' = c := Option.some.inj h
      obtain ⟨w, hw, hp⟩ := Option.bind_eq_some_iff.1 hc'
      obtain ⟨hts, htv⟩ := ih (by omega) t ht
      obtain ⟨hwv, -⟩ := tileVal_window p (128 * nw) a A B hA hB (by omega) hw
      obtain ⟨hadd, hcs⟩ := promote_step hts hp
      refine ⟨hcs, ?_⟩
      rw [ite_eq_right (Nat.succ_ne_zero n), show n + 1 - 1 = n by omega, tileVal_add p (128 * nw) a (by omega),
        show 2 * (128 * nw) + 128 + (5 * n + 4) - 1 = 2 * (128 * nw) + 128 + (5 * n + 3) by omega, hwv]
      have hprev : (if n = 0 then 0 else tileVal p (128 * nw) a (2 * (128 * nw) + 128 + (5 * n + 4) - 5)) = t := by
        rcases Nat.eq_zero_or_pos n with rfl | hn0
        · simpa using htv
        · rw [ite_eq_right (by omega), show 2 * (128 * nw) + 128 + (5 * n + 4) - 5 =
            2 * (128 * nw) + 128 + (5 * (n - 1) + 4) by omega]
          rwa [ite_eq_right (by omega)] at htv
      rw [hprev, f32Add_of_addWord hadd, Nat.mod_eq_of_lt hcs.1]
  obtain ⟨-, hv⟩ := key nw le_rfl c (hch ▸ hc)
  rw [ite_eq_right (by omega)] at hv
  rw [show cGate (128 * nw) = 2 * (128 * nw) + 128 + (5 * (nw - 1) + 4) by unfold cGate; omega, hv]

/-- **The tile unit's words** at depth `128·nw`: on inputs holding the codes of `A′_i` and `B̃_j` and the factor rows
`P_A,i` and `P_B,j`, its two outputs are `C̃_ij` and `U_ij` under the pinned order (`chainWordP? p 4`, `uWord?`). -/
theorem tile_words {nw : ℕ} (hnw : 0 < nw) (a : (tileSpec p (128 * nw)).toCircuit.Transcript)
    (A B : Fin (128 * nw) → Code) (pa pb : List ℕ) (hA : ∀ i : Fin (128 * nw), val a i = (A i).val)
    (hB : ∀ i : Fin (128 * nw), val a (128 * nw + i) = (B i).val)
    (hpa : ∀ i < 64, val a (2 * (128 * nw) + i) = pa.getD i 0)
    (hpb : ∀ i < 64, val a (2 * (128 * nw) + 64 + i) = pb.getD i 0) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms A B) = some c) :
    tileVal p (128 * nw) a (cGate (128 * nw)) = c ∧
      ∀ u, uWord? c pa pb = some u → tileVal p (128 * nw) a (uGate (128 * nw)) = u := by
  have hC := tileVal_chain p hnw a A B hA hB hc
  exact ⟨hC, fun u hu => tileVal_U p (128 * nw) a pa pb hpa hpb hC hu⟩

/-- **The tile unit, correct**: a transcript whose one unit is correct carries `C̃_ij` and `U_ij` on its outputs. -/
theorem tile_correct {nw : ℕ} (hnw : 0 < nw) (X : (tileSpec p (128 * nw)).toCircuit.Transcript)
    (hX : (tileSpec p (128 * nw)).part.Correct X 0)
    (A B : Fin (128 * nw) → Code) (pa pb : List ℕ) (hA : ∀ i : Fin (128 * nw), val X i = (A i).val)
    (hB : ∀ i : Fin (128 * nw), val X (128 * nw + i) = (B i).val)
    (hpa : ∀ i < 64, val X (2 * (128 * nw) + i) = pa.getD i 0)
    (hpb : ∀ i < 64, val X (2 * (128 * nw) + 64 + i) = pb.getD i 0) {c : ℕ}
    (hc : chainWordP? p 4 (rowAtoms A B) = some c) :
    val X (cGate (128 * nw)) = c ∧ ∀ u, uWord? c pa pb = some u → val X (uGate (128 * nw)) = u := by
  have hw : (tileSpec p (128 * nw)).part.wrong X = ∅ := by
    ext u
    simp only [Verity.Partition.mem_wrong, Finset.notMem_empty, iff_false, not_not]
    rwa [Subsingleton.elim u 0]
  have hout : ∀ g : ℕ, g ∈ ({cGate (128 * nw), uGate (128 * nw)} : Finset ℕ) → val X g =
      tileVal p (128 * nw) X g := by
    intro g hg
    have hlt : g < (tileSpec p (128 * nw)).toCircuit.N := by
      show g < tileN (128 * nw)
      simp only [Finset.mem_insert, Finset.mem_singleton] at hg
      rcases hg with rfl | rfl
      · unfold cGate tileN; omega
      · unfold uGate tileN; omega
    have hg' : (⟨g, hlt⟩ : Fin (tileSpec p (128 * nw)).toCircuit.N) ∈ (tileSpec p (128 * nw)).part.committed :=
      Verity.Partition.mem_committed_of_output _ (Finset.mem_filter.2 ⟨Finset.mem_univ _, hg⟩)
    rw [tileVal, val_of_lt _ hlt, val_of_lt _ hlt,
      (tileSpec p (128 * nw)).part.compose X X (fun _ _ => rfl) hw _ hg']
  obtain ⟨h1, h2⟩ := tile_words p hnw X A B pa pb hA hB hpa hpb hc
  refine ⟨(hout _ (by simp)).trans h1, fun u hu => (hout _ (by simp)).trans (h2 u hu)⟩

end Chain

end Pouw.PearlC.TileCircuit
