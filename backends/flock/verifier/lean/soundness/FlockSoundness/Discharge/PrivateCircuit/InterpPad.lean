import FlockSoundness.Discharge.PrivateCircuit.Interp
import FlockSoundness.Discharge.PrivateCircuit.Statements

/-!
# Protocol 2, the private circuit: padding computes the circuit

* `pad_correct`: `PadCorrect` at the universal unit (`Interp.lean`). A unit of `C` padded to the class's `(G, nIn)`, run
  by `UniversalUnit_v1{G, nIn, nOut}`, computes what `interpret` computes on the unit; every unit past `C`'s is a dummy,
  which outputs `nOut` zeros on every input.
  - `wires_sim`: gate by gate, the padded unit's wires hold the netlist's, each at its shifted index, while the
    indices fit `K` bits (`Nat.lt_log2_self`).
  - `gateFn_op`: `build`'s gate function at op's two bits is `interpret`'s rule.
* `pad_wrong`: `PadWrong`, the wrong units' count through `PadCorrect`, at any scheme.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

namespace Universal

theorem read_append_of_lt {w : List Bool} (t : List Bool) {i : Nat} (h : i < w.length) :
    read (w ++ t) i = read w i := by
  simp [read, List.getD_eq_getElem?_getD, List.getElem?_append_left h]

theorem read_append_length (w t : List Bool) (j : Nat) : read (w ++ t) (w.length + j) = read t j := by
  simp [read, List.getD_eq_getElem?_getD, List.getElem?_append_right]

theorem read_ofFn {n : Nat} (f : Fin n → Bool) (i : Nat) (h : i < n) : read (List.ofFn f) i = f ⟨i, h⟩ := by
  simp [read, List.getD_eq_getElem?_getD, h]

theorem wires_length {K : Nat} (gs : List (Gate K)) (w : List Bool) : (wires gs w).length = w.length + gs.length := by
  induction gs generalizing w with
  | nil => rfl
  | cons g gs ih => simp only [wires, List.foldl_cons] at ih ⊢; rw [ih]; simp; omega

theorem nwires_length (gs : List NGate) (w : List Bool) : (Netlist.wires gs w).length = w.length + gs.length := by
  induction gs generalizing w with
  | nil => rfl
  | cons g gs ih => simp only [Netlist.wires, List.foldl_cons] at ih ⊢; rw [ih]; simp; omega

theorem wires_append {K : Nat} (A B : List (Gate K)) (w : List Bool) : wires (A ++ B) w = wires B (wires A w) := by
  simp [wires, List.foldl_append]

theorem wires_prefix {K : Nat} (gs : List (Gate K)) (w : List Bool) : ∃ t, wires gs w = w ++ t := by
  induction gs generalizing w with
  | nil => exact ⟨[], by simp [wires]⟩
  | cons g gs ih =>
    obtain ⟨t, ht⟩ := ih (w ++ [g.eval w])
    exact ⟨[g.eval w] ++ t, by simp only [wires, List.foldl_cons] at ht ⊢; rw [ht]; simp⟩

theorem noop_eval {K : Nat} (w : List Bool) : (noop : Gate K).eval w = false := by
  simp [Gate.eval, noop, gateFn]

theorem wires_noops {K : Nat} (k : Nat) (w : List Bool) :
    wires (List.replicate k (noop : Gate K)) w = w ++ List.replicate k false := by
  induction k generalizing w with
  | zero => simp [wires]
  | succ k ih =>
    simp only [List.replicate_succ, wires, List.foldl_cons] at ih ⊢
    rw [ih, noop_eval]; simp

/-- `build`'s gate function at op's two bits is `interpret`'s rule. -/
theorem gateFn_op (op : Fin 4) (a b : Bool) :
    gateFn ((BitVec.ofNat 2 op.val).getLsbD 0) ((BitVec.ofNat 2 op.val).getLsbD 1) a b =
      xor (if op.val % 2 = 1 then a && b else xor a b) (decide (op.val / 2 = 1)) := by
  revert op a b; decide

/-- Every wire of the class fits `K` bits: `nIn + G ≤ 2^K`. -/
theorem wires_fit (G nIn : Nat) : nIn + G ≤ 2 ^ idxBits G nIn := by
  have := Nat.lt_log2_self (n := nIn + G - 1)
  have : 0 < 2 ^ idxBits G nIn := Nat.two_pow_pos _
  unfold idxBits at *; omega

theorem toNat_ofNat_lt {K i : Nat} (h : i < 2 ^ K) : (BitVec.ofNat K i).toNat = i := by
  simp [Nat.mod_eq_of_lt h]

theorem shiftWire_lt (u : Netlist) (c : Cls) {i m : Nat} (hi : i < m) :
    shiftWire u c i < m + (c.nIn - u.nIn) := by
  unfold shiftWire; split <;> omega

/-- **The padded unit's wires hold the netlist's.** Run on wires `ws` and `wp` whose reads agree at the shifted indices,
the netlist's gates `gs` (each source below its own wire) and their encodings keep the reads agreeing, while every
index fits `K` bits. -/
theorem wires_sim (u : Netlist) (c : Cls) :
    ∀ (gs : List NGate) (ws wp : List Bool), u.nIn ≤ ws.length → wp.length = ws.length + (c.nIn - u.nIn) →
      (∀ i < ws.length, read wp (shiftWire u c i) = read ws i) →
      (∀ j (h : j < gs.length), gs[j].a < ws.length + j ∧ gs[j].b < ws.length + j) →
      wp.length + gs.length ≤ 2 ^ idxBits c.G c.nIn →
      ∀ i < ws.length + gs.length,
        read (wires (gs.map (encGate u c)) wp) (shiftWire u c i) = read (Netlist.wires gs ws) i := by
  intro gs
  induction gs with
  | nil => intro ws wp _ _ hr _ _ i hi; exact hr i (by simpa using hi)
  | cons g gs ih =>
    intro ws wp hn hl hr hg hK
    have hab := hg 0 (by simp)
    simp only [List.getElem_cons_zero, Nat.add_zero] at hab
    have hlt : ∀ a, a < ws.length → (BitVec.ofNat (idxBits c.G c.nIn) (shiftWire u c a)).toNat = shiftWire u c a :=
      fun a ha => toNat_ofNat_lt (by have := shiftWire_lt u c ha; simp at hK; omega)
    have hv : (encGate u c g).eval wp = g.eval ws := by
      simp only [encGate, Gate.eval, NGate.eval]
      rw [hlt _ hab.1, hlt _ hab.2, hr _ hab.1, hr _ hab.2, gateFn_op]
    have hsw : shiftWire u c ws.length = wp.length := by unfold shiftWire; split <;> omega
    have := ih (ws ++ [g.eval ws]) (wp ++ [(encGate u c g).eval wp]) (by simp; omega) (by simp; omega)
      (by
        intro i hi
        simp only [List.length_append, List.length_singleton] at hi
        rcases Nat.lt_or_ge i ws.length with h | h
        · rw [read_append_of_lt _ h, read_append_of_lt _ (by have := shiftWire_lt u c h; omega), hr i h]
        · have : i = ws.length := by omega
          subst this
          rw [hsw, hv]
          have h1 := read_append_length wp [g.eval ws] 0
          have h2 := read_append_length ws [g.eval ws] 0
          simp only [Nat.add_zero] at h1 h2
          rw [h1, h2])
      (by
        intro j hj
        have := hg (j + 1) (by simp; omega)
        simp only [List.getElem_cons_succ, List.length_append, List.length_singleton] at this ⊢
        omega)
      (by simp at hK ⊢; omega)
    intro i hi
    simp only [List.map_cons, wires, Netlist.wires, List.foldl_cons] at this ⊢
    exact this i (by simp at hi ⊢; omega)

theorem padUnit_gates (c : Cls) (u : Netlist) (hG : u.gates.length ≤ c.G) :
    List.ofFn (padUnit c u).gates = u.gates.map (encGate u c) ++ List.replicate (c.G - u.gates.length) noop := by
  apply List.ext_getElem
  · simp; omega
  · intro k h1 h2
    simp only [List.getElem_ofFn, padUnit]
    split
    · rename_i hk
      rw [List.getElem_append_left (by simpa using hk)]
      simp
    · rename_i hk
      rw [List.getElem_append_right (by simp; omega)]
      simp

/-- **A padded unit computes its netlist.** -/
theorem interp_padUnit (c : Cls) (u : Netlist) (hu : u.Fits c) (x : Nat → Bool) :
    List.ofFn (interp (padUnit c u) fun j => x j.val) = interpret u x := by
  obtain ⟨hG, hn, ho, hg, hout⟩ := hu
  have hfit := wires_fit c.G c.nIn
  set ws := List.ofFn fun j : Fin u.nIn => x j.val
  set wp := List.ofFn fun j : Fin c.nIn => x j.val
  have hsim := wires_sim u c u.gates ws wp (by simp [ws]) (by simp [ws, wp]; omega)
    (by
      intro i hi
      simp only [ws, List.length_ofFn] at hi
      have : shiftWire u c i = i := by simp [shiftWire, hi]
      rw [this, read_ofFn _ _ (by omega), read_ofFn _ _ hi])
    (by intro j hj; simpa [ws] using hg j hj)
    (by simp [wp]; omega)
  obtain ⟨t, ht⟩ := wires_prefix (List.replicate (c.G - u.gates.length) (noop : Gate (idxBits c.G c.nIn)))
    (wires (u.gates.map (encGate u c)) wp)
  apply List.ext_getElem
  · simp [interpret, ho]
  · intro i h1 h2
    simp only [List.getElem_ofFn, interp, interpret, List.getElem_map]
    have hi : i < u.outputs.length := by simpa [interpret] using h2
    have hoi := hout _ (List.getElem_mem hi)
    have hsl := shiftWire_lt u c (m := u.nIn + u.gates.length) hoi
    rw [padUnit_gates c u hG, wires_append, ht]
    simp only [padUnit, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
    rw [toNat_ofNat_lt (by omega),
      read_append_of_lt _ (by rw [wires_length]; simp [wp]; omega)]
    exact hsim _ (by simp [ws]; omega)

/-- **The dummy unit outputs zeros**, on every input. -/
theorem interp_dummyUnit (c : Cls) (hG : 1 ≤ c.G) (x : Fin c.nIn → Bool) (i : Fin c.nOut) :
    interp (dummyUnit c) x i = false := by
  have hfit := wires_fit c.G c.nIn
  have hof : List.ofFn (dummyUnit c).gates = List.replicate c.G noop := by
    apply List.ext_getElem <;> simp [dummyUnit]
  simp only [interp, hof, wires_noops]
  have : (BitVec.ofNat (idxBits c.G c.nIn) c.nIn).toNat = c.nIn := toNat_ofNat_lt (by omega)
  simp only [dummyUnit, this]
  have h := read_append_length (List.ofFn x) (List.replicate c.G false) 0
  simp only [List.length_ofFn, Nat.add_zero] at h
  rw [h]
  simp [read, List.getD_eq_getElem?_getD, show 0 < c.G by omega]

end Universal

open Universal in
/-- **Padding computes the circuit, at the universal unit**: `PadCorrect` for `UniversalUnit_v1`'s interpreter, the
padding of `universal.pad`, and the dummy unit, whose output `dummy c` is exactly `nOut` zeros; `nOut` is exact, since
outputs are not padded. It is unit by unit; the wiring between units is the class's public topology, and `wired_eval`
(`Wiring.lean`) composes the units along it into `C` end to end. `interp` is tested equal to `UniversalUnit_v1` (`test_lean_universal.py`, by program digest), not
proved. -/
theorem pad_correct :
    PadCorrect Universal.fits Universal.pad Universal.units Universal.size Universal.run Universal.eval Universal.dummy := by
  intro C c ⟨hG, hN, hu⟩
  refine ⟨hN, ?_, ?_⟩
  · intro i hi x
    have hiN : i < c.N := by unfold size at hi; omega
    simp only [run, hiN, dite_true, pad, show i < C.length from hi, eval, List.getElem?_eq_getElem hi]
    exact interp_padUnit c _ (hu _ (List.getElem_mem hi)) x
  · intro i hi hiN x
    have : ¬ i < C.length := by unfold size at hi; omega
    simp only [run, show i < c.N from hiN, dite_true, pad, this, dite_false, dummy]
    rw [← List.ofFn_const]
    congr 1
    funext j
    exact interp_dummyUnit c hG _ j

/-- **Padding adds no wrong unit, and hides none.** -/
theorem pad_wrong : PadWrong := by
  intro Cls Circ In Out _ Prog fits pad N n run eval dummy hpad C c hfit x y
  obtain ⟨hn, hreal, hdummy⟩ := hpad C c hfit
  refine ⟨Finset.card_le_card ?_, fun hy => congrArg Finset.card ?_⟩
  · intro i hi
    simp only [Finset.mem_filter, Finset.mem_range] at hi ⊢
    exact ⟨by omega, by rw [hreal i hi.1]; exact hi.2⟩
  · ext i
    simp only [Finset.mem_filter, Finset.mem_range]
    constructor
    · rintro ⟨hiN, hne⟩
      rcases Nat.lt_or_ge i (n C) with h | h
      · exact ⟨h, by rw [← hreal i h]; exact hne⟩
      · exact absurd (by rw [hdummy i h hiN, hy i h hiN]) hne
    · rintro ⟨hi, hne⟩
      exact ⟨by omega, by rw [hreal i hi]; exact hne⟩

end FlockSoundness.Discharge.PrivateCircuit
