import Definitions.Pouw.PearlC.WholeCircuit
import Proofs.Pouw.PearlC.TileCircuit

/-!
# A unit of a composed table is its template (proofs)

`WholeCircuit.Whole` composes unit templates by their ports. Here a unit of it is cut out as its template's own
transcript (`cut`): the template evaluated on the values the composed transcript gives its ports. The cut is correct in
the template's one-unit partition whatever the transcript (`cut_correct`, an evaluation is correct for every unit:
`localEval_eval`), and it agrees with the composed transcript's local evaluation of the unit at every gate
(`cut_val`), so at the unit's inputs (`cut_input`) and, where the unit is correct, at every committed gate it computes
(`cut_committed`). The unit lemmas proved of a template's own table (`TileCircuit.tile_correct`,
`RowCircuit.row8A_code`, …) therefore hold of each unit of the composed table unchanged. A port that a unit's gate
reads is committed (`port_committed`), being an input or another unit's gate.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.PearlC.TileCircuit

/-- **An evaluation is correct for every unit**: recomputing a unit from the evaluation gives the evaluation back. -/
theorem localEval_eval {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n) (a : C.Transcript) (u : Fin n) :
    P.localEval (C.eval a) u = C.eval a := by
  funext g
  induction g using WellFoundedLT.induction with
  | _ g ih =>
    by_cases h : P.unit g = some u
    · rw [P.localEval_of_mem _ h]
      have hi : C.isInput g = false := by
        cases hc : C.isInput g
        · rfl
        · exact absurd ((P.input_iff g).2 hc) (by rw [h]; simp)
      rw [C.eval_op a hi]
      exact C.op_congr g _ _ fun x hx => ih x (C.reads_lt g x hx)
    · exact P.localEval_of_not_mem _ h

theorem correct_eval {C : Verity.Circuit} {n : ℕ} (P : Verity.Partition C n) (a : C.Transcript) (u : Fin n) :
    P.Correct (C.eval a) u := fun g _ _ => by rw [localEval_eval]

/-! ## Laying out blocks -/

theorem blocksLen_append : ∀ (L1 L2 : List Block), blocksLen (L1 ++ L2) = blocksLen L1 + blocksLen L2
  | [], _ => by simp [blocksLen]
  | B :: L1, L2 => by simp only [List.cons_append, blocksLen, blocksLen_append L1 L2]; omega

theorem unitCount_append : ∀ (L1 L2 : List Block), unitCount (L1 ++ L2) = unitCount L1 + unitCount L2
  | [], _ => by simp [unitCount]
  | B :: L1, L2 => by simp only [List.cons_append, unitCount, unitCount_append L1 L2]; omega

theorem locate_left : ∀ (L1 L2 : List Block) (o u g : ℕ), o ≤ g → g < o + blocksLen L1 →
    locate (L1 ++ L2) o u g = locate L1 o u g
  | [], _, o, _, g, h1, h2 => by simp only [blocksLen] at h2; omega
  | B :: L1, L2, o, u, g, h1, h2 => by
    simp only [List.cons_append, locate]
    split_ifs with h
    · rfl
    · exact locate_left L1 L2 _ _ g (by omega) (by simp only [blocksLen] at h2; omega)

theorem locate_u_lt : ∀ (L : List Block) (o u g : ℕ) {l : Loc}, o ≤ g → locate L o u g = some l →
    l.u < u + unitCount L
  | [], _, _, _, _, _, h => by simp [locate] at h
  | B :: L, o, u, g, l, hog, h => by
    unfold locate at h
    split_ifs at h with hg
    · cases h
      have hL : 0 < B.len := Nat.pos_of_ne_zero fun h0 => by rw [h0] at hg; omega
      have hq : (g - o) / B.len < B.cnt := (Nat.div_lt_iff_lt_mul hL).2 (by omega)
      simp only [unitCount]; omega
    · have := locate_u_lt L _ _ g (by omega) h
      simp only [unitCount]; omega

theorem locate_mid (B : Block) (L2 : List Block) : ∀ (L1 : List Block) (o u g : ℕ),
    o + blocksLen L1 ≤ g → g < o + blocksLen L1 + B.cnt * B.len →
    locate (L1 ++ B :: L2) o u g = some ⟨B, o + blocksLen L1, u + unitCount L1 + (g - (o + blocksLen L1)) / B.len,
      (g - (o + blocksLen L1)) / B.len, B.nIn + (g - (o + blocksLen L1)) % B.len⟩
  | [], o, u, g, _, h2 => by
    simp only [List.nil_append, blocksLen, unitCount, Nat.add_zero] at h2 ⊢
    unfold locate
    rw [ite_eq_left h2]
  | B' :: L1, o, u, g, h1, h2 => by
    simp only [List.cons_append, blocksLen, unitCount] at h1 h2 ⊢
    unfold locate
    rw [ite_eq_right (by omega), locate_mid B L2 L1 (o + B'.cnt * B'.len) (u + B'.cnt) g (by omega) (by omega)]
    simp only [Nat.add_assoc]

theorem blocksOK_mid (B : Block) (L2 : List Block) : ∀ (L1 : List Block) (o : ℕ), BlocksOK (L1 ++ B :: L2) o →
    ∀ q < B.cnt, ∀ a < B.nIn, B.port q a < o + blocksLen L1
  | [], o, h => by simpa [blocksLen] using h.1
  | _ :: L1, o, h => by
    intro q hq a ha
    have := blocksOK_mid B L2 L1 _ h.2 q hq a ha
    simp only [blocksLen]
    omega

/-! ## One instance of a block -/

namespace Whole

variable (W : Whole) {L1 L2 : List Block} {B : Block} (hW : W.blocks = L1 ++ B :: L2) {q : ℕ} (hq : q < B.cnt)
include hW hq

theorem unit_lt : unitCount L1 + q < unitCount W.blocks := by
  rw [hW, unitCount_append]; simp only [unitCount]; omega

theorem inst_lt {a : ℕ} (ha1 : B.nIn ≤ a) (ha : a < B.S.N) :
    W.nIn + blocksLen L1 + q * B.len + (a - B.nIn) < W.spec.toCircuit.N := by
  have := mul_add_lt (L := B.len) hq (show a - B.nIn < B.len by unfold Block.len; omega)
  show _ < W.nIn + blocksLen W.blocks
  rw [hW, blocksLen_append]; simp only [blocksLen]; omega

theorem port_lt {a : ℕ} (ha : a < B.nIn) : B.port q a < W.nIn + blocksLen L1 :=
  blocksOK_mid B L2 L1 W.nIn (hW ▸ W.ok) q hq a ha

theorem port_lt_N {a : ℕ} (ha : a < B.nIn) : B.port q a < W.spec.toCircuit.N :=
  lt_of_lt_of_le (W.port_lt hW hq ha) (by
    show _ ≤ W.nIn + blocksLen W.blocks
    rw [hW, blocksLen_append]; omega)

theorem locate_inst {a : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N) :
    locate W.blocks W.nIn 0 (W.nIn + blocksLen L1 + q * B.len + (a - B.nIn)) =
      some ⟨B, W.nIn + blocksLen L1, unitCount L1 + q, q, a⟩ := by
  have hlt : a - B.nIn < B.len := by unfold Block.len; omega
  have := mul_add_lt (L := B.len) hq hlt
  rw [hW, locate_mid B L2 L1 _ 0 _ (by omega) (by omega)]
  have hd : (W.nIn + blocksLen L1 + q * B.len + (a - B.nIn) - (W.nIn + blocksLen L1)) = q * B.len + (a - B.nIn) := by
    omega
  have hL : 0 < B.len := by omega
  rw [hd, Nat.mul_comm q, Nat.mul_add_div hL, Nat.div_eq_of_lt hlt, Nat.mul_add_mod, Nat.mod_eq_of_lt hlt]
  simp only [Nat.add_zero, Nat.zero_add]
  congr 3
  omega

theorem gate_inst {a : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N) :
    W.gate (W.nIn + blocksLen L1 + q * B.len + (a - B.nIn)) = B.gateAt (W.nIn + blocksLen L1) q a := by
  unfold Whole.gate
  rw [ite_eq_right (by omega), W.locate_inst hW hq ha1 ha2]

theorem unit_inst {a : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N) :
    W.part.unit ⟨_, W.inst_lt hW hq ha1 ha2⟩ = some ⟨_, W.unit_lt hW hq⟩ := by
  show (W.unitOf (W.nIn + blocksLen L1 + q * B.len + (a - B.nIn))).bind _ = _
  unfold Whole.unitOf
  rw [ite_eq_right (by omega), W.locate_inst hW hq ha1 ha2]
  simp [W.unit_lt hW hq]

theorem unit_ne {g : Fin W.spec.toCircuit.N} (hg : g.val < W.nIn + blocksLen L1) :
    W.part.unit g ≠ some ⟨_, W.unit_lt hW hq⟩ := by
  show (W.unitOf g.val).bind _ ≠ _
  unfold Whole.unitOf
  split_ifs with h
  · simp
  · have hs := locate_isSome W.blocks W.nIn 0 g.val (not_lt.1 h) g.isLt
    obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 hs
    have hl' : locate L1 W.nIn 0 g.val = some l := by
      rw [← locate_left L1 (B :: L2) _ _ _ (not_lt.1 h) hg, ← hW]
      exact hl
    have hu := locate_u_lt L1 _ 0 g.val (not_lt.1 h) hl'
    rw [hl]
    simp only [Option.map_some, Option.bind_some]
    split_ifs with hlt
    · simp only [ne_eq, Option.some.injEq, Fin.mk.injEq]; omega
    · simp

/-- Instance `q` of a block cut out of `X`: the template evaluated on the values `X` gives its ports. -/
def cut (X : W.spec.toCircuit.Transcript) (B : Block) (q : ℕ) : B.S.toCircuit.Transcript :=
  B.S.toCircuit.eval fun a => BitVec.ofNat _ (val X (B.port q a))

omit hW hq in
theorem cut_correct (X : W.spec.toCircuit.Transcript) (B : Block) (q : ℕ) : B.S.part.Correct (W.cut X B q) 0 :=
  correct_eval _ _ _

/-- **The cut is the composed table's unit, gate by gate**: where the template's inputs are its first `nIn` gates and
each port's value fits its input's width, every template gate's value in the cut is the unit's local evaluation at its
copy. -/
theorem cut_val (X : W.spec.toCircuit.Transcript) (hin : ∀ a < B.S.N, (B.S.gate a).input = true ↔ a < B.nIn)
    (hfit : ∀ a < B.nIn, val X (B.port q a) < 2 ^ (B.S.gate a).w) :
    ∀ a < B.S.N, val (W.cut X B q) a =
      val (W.part.localEval X ⟨_, W.unit_lt hW hq⟩) (B.reloc (W.nIn + blocksLen L1) q a) := by
  intro a
  induction a using Nat.strong_induction_on with
  | _ a ih =>
    intro ha
    by_cases hi : a < B.nIn
    · have hp := W.port_lt hW hq hi
      have hpN := W.port_lt_N hW hq hi
      rw [cut, Spec.val_input _ _ ha ((hin a ha).2 hi), BitVec.toNat_ofNat, Nat.mod_eq_of_lt (hfit a hi),
        Block.reloc, ite_eq_left hi, val_of_lt _ hpN, val_of_lt _ hpN,
        W.part.localEval_of_not_mem X (W.unit_ne hW hq (g := ⟨_, hpN⟩) hp)]
    · have hc : (B.S.gate a).input = false := by
        cases h : (B.S.gate a).input
        · rfl
        · exact absurd ((hin a ha).1 h) hi
      have hN := W.inst_lt hW hq (not_lt.1 hi) ha
      rw [cut, Spec.val_gate _ _ ha hc, Block.reloc, ite_eq_right hi, val_of_lt _ hN,
        W.part.localEval_of_mem X (W.unit_inst hW hq (not_lt.1 hi) ha)]
      show _ = (BitVec.ofNat (W.gate _).w ((W.gate _).f ((W.gate _).rd.map _))).toNat
      rw [BitVec.toNat_ofNat, W.gate_inst hW hq (not_lt.1 hi) ha]
      simp only [Block.gateAt, List.map_map]
      congr 2
      apply List.map_congr_left
      intro a' ha'
      have := B.S.rd_lt a ha a' ha'
      exact ih a' this (by omega)

/-- The cut's inputs are the ports' values. -/
theorem cut_input (X : W.spec.toCircuit.Transcript) (hin : ∀ a < B.S.N, (B.S.gate a).input = true ↔ a < B.nIn)
    (hfit : ∀ a < B.nIn, val X (B.port q a) < 2 ^ (B.S.gate a).w) (hnN : B.nIn ≤ B.S.N) {a : ℕ} (ha : a < B.nIn) :
    val (W.cut X B q) a = val X (B.port q a) := by
  have hpN := W.port_lt_N hW hq ha
  rw [W.cut_val hW hq X hin hfit a (by omega), Block.reloc, ite_eq_left ha, val_of_lt _ hpN, val_of_lt _ hpN,
    W.part.localEval_of_not_mem X (W.unit_ne hW hq (g := ⟨_, hpN⟩) (W.port_lt hW hq ha))]

/-- **A correct unit's committed gates are the cut's.** -/
theorem cut_committed (X : W.spec.toCircuit.Transcript) (hin : ∀ a < B.S.N, (B.S.gate a).input = true ↔ a < B.nIn)
    (hfit : ∀ a < B.nIn, val X (B.port q a) < 2 ^ (B.S.gate a).w) (hX : W.part.Correct X ⟨_, W.unit_lt hW hq⟩)
    {a : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N)
    (hcom : (⟨_, W.inst_lt hW hq ha1 ha2⟩ : Fin W.spec.toCircuit.N) ∈ W.part.committed) :
    val X (W.nIn + blocksLen L1 + q * B.len + (a - B.nIn)) = val (W.cut X B q) a := by
  rw [W.cut_val hW hq X hin hfit a ha2, Block.reloc, ite_eq_right (by omega), val_of_lt _ (W.inst_lt hW hq ha1 ha2),
    val_of_lt _ (W.inst_lt hW hq ha1 ha2), hX _ hcom (W.unit_inst hW hq ha1 ha2)]

/-- A unit's gate reads the ports of the template inputs it reads. -/
theorem port_reads {a a' : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N) (ha' : a' < B.nIn) (h : a' ∈ (B.S.gate a).rd) :
    (⟨B.port q a', W.port_lt_N hW hq ha'⟩ : Fin W.spec.toCircuit.N) ∈
      W.spec.toCircuit.reads ⟨_, W.inst_lt hW hq ha1 ha2⟩ := by
  simp only [Spec.toCircuit, Finset.mem_filter, Finset.mem_univ, true_and]
  show B.port q a' ∈ (W.gate _).rd
  rw [W.gate_inst hW hq ha1 ha2]
  simp only [Block.gateAt, List.mem_map]
  exact ⟨a', h, by simp [Block.reloc, ha']⟩

/-- **A port a unit's gate reads is committed**: an input, or a gate of an earlier unit. -/
theorem port_committed {a a' : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N) (ha' : a' < B.nIn)
    (h : a' ∈ (B.S.gate a).rd) :
    (⟨B.port q a', W.port_lt_N hW hq ha'⟩ : Fin W.spec.toCircuit.N) ∈ W.part.committed :=
  W.part.mem_committed_of_read (W.port_reads hW hq ha1 ha2 ha' h) (by
    rw [W.unit_inst hW hq ha1 ha2]
    exact fun e => W.unit_ne hW hq (g := ⟨_, W.port_lt_N hW hq ha'⟩) (W.port_lt hW hq ha') e.symm)

end Whole

end Pouw.PearlC.WholeCircuit
