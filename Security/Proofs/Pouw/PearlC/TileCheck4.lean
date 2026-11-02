import Definitions.Pouw.PearlC.TileCheck4
import Proofs.Pouw.PearlC.TileCheck8
import Proofs.Pouw.PearlC.Tile4Circuit

/-!
# `pc4`'s tile check: its units are its templates (proofs)

`WholeCircuit.whole4` is `Pc4Check`'s table, its units the instances of `Pc4RowB_v1`, `Pc4RowA_v1` and `Pc4Tile_v1`.
As for `pc8` (`TileCheck8`), each unit an entry's words depend on is cut out as its template's transcript
(`Whole.cut`), and the cuts are chained along the ports, so the unit lemmas proved of each template's own table apply to
the composed table unchanged:

* every port reads a gate of its input's width (`hfit4B`, `hfit4A`, and `hfit4T` where `16 ∣ K`, so the tile's scale
  bytes are the rows' `UE4M3` casts), so a cut holds on its inputs the transcript's values at its ports (`cutB4_input`,
  `cutA4_input`, `cutT4_input`) and, where its unit is correct, the transcript holds on the unit's committed gates the
  cut's values (`cutB4_at`, `cutA4_at`, `cutT4_at`);
* entry `(i, j)`'s tile unit is in the cone of its `C̃` (`cone4T`), and RowA `i` and RowB `j` in the cone of its `U`,
  whose last K16 step reads their `P_A` and `P_B` words (`cone4A`, `cone4B`);
* where the three units are correct, the entry's `C̃` and `U` hold the tile cut's (`chain4_C`, `chain4_U`), and the tile
  cut's `A′` and `B̃` rows, codes and scale bytes, and its `P_A` and `P_B` rows are the row cuts' (`chain4_A`,
  `chain4_SA`, `chain4_B`, `chain4_SB`, `chain4_PA`, `chain4_PB`), each gate a tile reads being committed
  (`Whole.port_committed`).
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

variable (ops : RowOps)

/-! ## The row tables' gates -/

theorem pre4Runs_blk (K b : ℕ) (rest : List Run) :
    pre4Runs K b ++ rest = (statsRuns .bf16Fabs .bf16 K b ++ [alpha4Run K b, d4Run K b]) ++ blk4Run K b :: rest := by
  simp [pre4Runs]

theorem bk4At_len (K b : ℕ) :
    b + runsLen (statsRuns .bf16Fabs .bf16 K b ++ [alpha4Run K b, d4Run K b]) = bk4At K b := by
  have := statsRuns_len .bf16Fabs .bf16 K b
  simp only [runsLen_append] at this ⊢
  simp [runsLen, alpha4Run, d4Run, Run.ofList, bk4At, d4At, a4At]
  omega

/-- A `pc4` row's code gates are E2M1 casts, 4 bits wide. -/
theorem rows4_code_w (T : Rows) {K : ℕ} {rest : List Run} (hT : T.runs = pre4Runs K T.nIn ++ rest)
    {t : ℕ} (ht : t < 16 * (K / 16)) : ((T.spec ops).gate (code4 K T.nIn t)).w = 4 := by
  have hl := bk4At_len K T.nIn
  rw [Rows.gate_run ops T (hT.trans (pre4Runs_blk K T.nIn rest)) (by unfold code4; omega)
    (by simp only [blk4Run, Run.rep]; unfold code4; omega)]
  have e1 : code4 K T.nIn t - (T.nIn + runsLen (statsRuns .bf16Fabs .bf16 K T.nIn ++
      [alpha4Run K T.nIn, d4Run K T.nIn])) = t / 16 * 52 + (21 + 2 * (t % 16)) := by
    unfold code4; omega
  have e2 : (t / 16 * 52 + (21 + 2 * (t % 16))) / 52 = t / 16 := by omega
  have e3 : (t / 16 * 52 + (21 + 2 * (t % 16))) % 52 = 21 + 2 * (t % 16) := by omega
  rw [e1]
  simp only [blk4Run, Run.rep, e2, e3, blk4, show ¬ (21 + 2 * (t % 16) < 15) by omega,
    show 21 + 2 * (t % 16) ≠ 15 by omega, show 21 + 2 * (t % 16) ≠ 16 by omega, show 21 + 2 * (t % 16) ≠ 17 by omega,
    show 21 + 2 * (t % 16) ≠ 18 by omega, show 21 + 2 * (t % 16) ≠ 19 by omega,
    show (21 + 2 * (t % 16) - 20) % 2 ≠ 0 by omega, ↓reduceIte, RGate.toGate, Prim.w]

/-- A `pc4` row's scale bytes are UE4M3 casts, 8 bits wide. -/
theorem rows4_scale_w (T : Rows) {K : ℕ} {rest : List Run} (hT : T.runs = pre4Runs K T.nIn ++ rest)
    {q : ℕ} (hq : q < K / 16) : ((T.spec ops).gate (scale4 K T.nIn q)).w = 8 := by
  have hl := bk4At_len K T.nIn
  rw [Rows.gate_run ops T (hT.trans (pre4Runs_blk K T.nIn rest)) (by unfold scale4; omega)
    (by simp only [blk4Run, Run.rep]; unfold scale4; omega)]
  have e1 : scale4 K T.nIn q - (T.nIn + runsLen (statsRuns .bf16Fabs .bf16 K T.nIn ++
      [alpha4Run K T.nIn, d4Run K T.nIn])) = q * 52 + 17 := by
    unfold scale4; omega
  have e2 : (q * 52 + 17) / 52 = q := by omega
  have e3 : (q * 52 + 17) % 52 = 17 := by omega
  rw [e1]
  simp only [blk4Run, Run.rep, e2, e3, blk4, show ¬ ((17 : ℕ) < 15) by omega, show (17 : ℕ) ≠ 15 by omega,
    show (17 : ℕ) ≠ 16 by omega, ↓reduceIte, RGate.toGate, Prim.w]

theorem row4A_code_w {K t : ℕ} (ht : t < 16 * (K / 16)) :
    (((row4A K).spec ops).gate (code4 K (rowIn K false) t)).w = 4 :=
  rows4_code_w ops (row4A K) rfl ht

theorem row4B_code_w {K t : ℕ} (ht : t < 16 * (K / 16)) :
    (((row4B K).spec ops).gate (code4 K (rowIn K true) t)).w = 4 :=
  rows4_code_w ops (row4B K) rfl ht

theorem row4A_scale_w {K q : ℕ} (hq : q < K / 16) :
    (((row4A K).spec ops).gate (scale4 K (rowIn K false) q)).w = 8 :=
  rows4_scale_w ops (row4A K) rfl hq

theorem row4B_scale_w {K q : ℕ} (hq : q < K / 16) :
    (((row4B K).spec ops).gate (scale4 K (rowIn K true) q)).w = 8 :=
  rows4_scale_w ops (row4B K) rfl hq

/-- RowA's `P_A` words are BF16 words: its `Ẽ` (`PouwETilde_v1`) and its chains' `f2fpBf16` casts, 16 bits wide. -/
theorem row4A_pa_w {K w : ℕ} (hw : w < 64) : (((row4A K).spec ops).gate (paOut4 K w)).w = 16 := by
  have hl := pre4Runs_len K (rowIn K false)
  unfold paOut4
  split_ifs with h
  · have hr : (row4A K).runs = (pre4Runs K (rowIn K false) ++ [af4Run K (rowIn K false)]) ++
        et4Run K (rowIn K false) :: [okwRun (a4At K (rowIn K false) + 12)] := by simp [row4A]
    have hn : runsLen (pre4Runs K (rowIn K false) ++ [af4Run K (rowIn K false)]) =
        runsLen (pre4Runs K (rowIn K false)) + 32 * (K / 64 + 1) := by
      simp [runsLen, af4Run, Run.rep]
    rw [Rows.gate_run ops (row4A K) hr (by rw [hn]; show rowIn K false + _ ≤ _; omega)
      (by rw [hn]; show _ < rowIn K false + _ + _; simp only [et4Run]; omega)]
    simp only [et4Run, RGate.toGate, Prim.w]
  · have hr : (row4A K).runs = pre4Runs K (rowIn K false) ++ af4Run K (rowIn K false) ::
        [et4Run K (rowIn K false), okwRun (a4At K (rowIn K false) + 12)] := by simp [row4A]
    have hq := mul_add_lt (L := K / 64 + 1) (show w - 32 < 32 by omega) (show K / 64 < _ by omega)
    rw [Rows.gate_run ops (row4A K) hr (by show rowIn K false + _ ≤ _; omega)
      (by show _ < rowIn K false + _ + _; simp only [af4Run, Run.rep]; omega)]
    have e1 : post4At K (rowIn K false) + (w - 32) * (K / 64 + 1) + K / 64 -
        ((row4A K).nIn + runsLen (pre4Runs K (rowIn K false))) = (w - 32) * (K / 64 + 1) + K / 64 := by
      show _ - (rowIn K false + _) = _
      omega
    have e2 : ((w - 32) * (K / 64 + 1) + K / 64) / (K / 64 + 1) = w - 32 := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega), Nat.zero_add]
    have e3 : ((w - 32) * (K / 64 + 1) + K / 64) % (K / 64 + 1) = K / 64 := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    rw [e1]
    simp only [af4Run, Run.rep, e2, e3, lt_irrefl, ↓reduceIte, RGate.toGate, Prim.w]

/-- RowB's `P_B` words are BF16 words: its blocks' `f2fpBf16` casts and its `bf16Neg`s, 16 bits wide. -/
theorem row4B_pb_w {K w : ℕ} (hw : w < 64) : (((row4B K).spec ops).gate (pbOut4 K w)).w = 16 := by
  have hl := pre4Runs_len K (rowIn K true)
  unfold pbOut4
  split_ifs with h
  · have hr : (row4B K).runs = (pre4Runs K (rowIn K true) ++ [et4Run K (rowIn K true)]) ++
        mid4Run K (rowIn K true) :: [nb4Run (post4At K (rowIn K true)), okwRun (a4At K (rowIn K true) + 12)] := by
      simp [row4B]
    have hn : runsLen (pre4Runs K (rowIn K true) ++ [et4Run K (rowIn K true)]) =
        runsLen (pre4Runs K (rowIn K true)) + 32 := by
      simp [runsLen, et4Run]
    have hq := mul_add_lt (L := K / 64 + 35) h (show K / 64 + 34 < _ by omega)
    rw [Rows.gate_run ops (row4B K) hr (by rw [hn]; show rowIn K true + _ ≤ _; omega)
      (by rw [hn]; show _ < rowIn K true + _ + _; simp only [mid4Run, Run.rep]; omega)]
    have e1 : post4At K (rowIn K true) + 32 + w * (K / 64 + 35) + K / 64 + 34 -
        ((row4B K).nIn + (runsLen (pre4Runs K (rowIn K true)) + 32)) = w * (K / 64 + 35) + (K / 64 + 34) := by
      show _ - (rowIn K true + _) = _
      omega
    have e2 : (w * (K / 64 + 35) + (K / 64 + 34)) / (K / 64 + 35) = w := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega), Nat.zero_add]
    have e3 : (w * (K / 64 + 35) + (K / 64 + 34)) % (K / 64 + 35) = K / 64 + 34 := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    rw [hn, e1]
    simp only [mid4Run, Run.rep, e2, e3, mid4, show ¬ (K / 64 + 34 < 32) by omega,
      show ¬ (K / 64 + 34 < 32 + K / 64) by omega, show K / 64 + 34 ≠ 32 + K / 64 by omega,
      show K / 64 + 34 ≠ 33 + K / 64 by omega, ↓reduceIte, RGate.toGate, Prim.w]
  · have hr : (row4B K).runs = (pre4Runs K (rowIn K true) ++ [et4Run K (rowIn K true), mid4Run K (rowIn K true)]) ++
        nb4Run (post4At K (rowIn K true)) :: [okwRun (a4At K (rowIn K true) + 12)] := by simp [row4B]
    have hn : runsLen (pre4Runs K (rowIn K true) ++ [et4Run K (rowIn K true), mid4Run K (rowIn K true)]) =
        runsLen (pre4Runs K (rowIn K true)) + 32 + 32 * (K / 64 + 35) := by
      simp [runsLen, et4Run, mid4Run, Run.rep]; omega
    rw [Rows.gate_run ops (row4B K) hr (by rw [hn]; show rowIn K true + _ ≤ _; omega)
      (by rw [hn]; show _ < rowIn K true + _ + _; simp only [nb4Run]; omega)]
    simp only [nb4Run, RGate.toGate, Prim.w]

/-! ## `whole4`'s blocks -/

section Whole4

variable (M K N TM TN : ℕ)

/-- `whole4`'s RowB block. -/
def blockB4 : BlockKind := ⟨(row4B K).spec ops, rowIn K true, N, portB8 M N K⟩

/-- `whole4`'s RowA block. -/
def blockA4 : BlockKind := ⟨(row4A K).spec ops, rowIn K false, M, portA8 M N K⟩

/-- `whole4`'s tile block. -/
def blockT4 : BlockKind := ⟨tile4Spec K TM TN, tile4In K TM TN, M / TM * (N / TN), portT4 M N K TM TN⟩

theorem whole4_B : (whole4 ops M K N TM TN).blocks =
    [] ++ blockB4 ops M K N :: [blockA4 ops M K N, blockT4 M K N TM TN] := rfl

theorem whole4_A : (whole4 ops M K N TM TN).blocks =
    [blockB4 ops M K N] ++ blockA4 ops M K N :: [blockT4 M K N TM TN] := rfl

theorem whole4_T : (whole4 ops M K N TM TN).blocks =
    [blockB4 ops M K N, blockA4 ops M K N] ++ blockT4 M K N TM TN :: [] := rfl

theorem rowB4_eq (j a : ℕ) : rowB4 M N K j a = (whole4 ops M K N TM TN).nIn + blocksLen [] +
    j * (blockB4 ops M K N).len + (a - (blockB4 ops M K N).nIn) := by
  simp only [rowB4, whole4, blocksLen, BlockKind.len, blockB4, Rows.spec, add_zero]

theorem rowA4_eq (i a : ℕ) : rowA4 M N K i a = (whole4 ops M K N TM TN).nIn + blocksLen [blockB4 ops M K N] +
    i * (blockA4 ops M K N).len + (a - (blockA4 ops M K N).nIn) := by
  simp only [rowA4, whole4, blocksLen, BlockKind.len, blockB4, blockA4, Rows.spec, add_zero]

theorem tile4_eq (e a : ℕ) : tile4 M N K TM TN e a = (whole4 ops M K N TM TN).nIn +
    blocksLen [blockB4 ops M K N, blockA4 ops M K N] + e * (blockT4 M K N TM TN).len +
      (a - (blockT4 M K N TM TN).nIn) := by
  simp only [tile4, tiles4, blocksLen, BlockKind.len, blockB4, blockA4, blockT4, whole4, add_zero, Nat.add_assoc]
  rfl

theorem whole4_N : (whole4 ops M K N TM TN).spec.toCircuit.N = in8 M N K + N * ((row4B K).N - rowIn K true) +
    M * ((row4A K).N - rowIn K false) + M / TM * (N / TN) * (tile4N K TM TN - tile4In K TM TN) := by
  show in8 M N K + blocksLen _ = _
  simp only [blocksLen, BlockKind.len, whole4, add_zero, Nat.add_assoc]
  rfl

/-! ### Inputs -/

theorem inW4_portA8 {i a : ℕ} (hi : i < M) (ha : a < rowIn K false) :
    inW4 M N K (portA8 M N K i a) = (row4A K).inW a := by
  have e1 : (M + N) * (K + 32) = M * K + N * K + 32 * M + 32 * N := by ring
  have e2 : M * (K + 32) = M * K + 32 * M := by ring
  have hiK : i * K + K ≤ M * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hi
  simp only [rowIn, Bool.false_eq_true, ↓reduceIte, add_zero] at ha
  simp only [row4A]
  unfold inW4 portA8 eaAt bAt ebAt faAt fbAt gAt
  rw [e1, e2]
  split_ifs <;> omega

theorem inW4_portB8 {j a : ℕ} (hj : j < N) (ha : a < rowIn K true) :
    inW4 M N K (portB8 M N K j a) = (row4B K).inW a := by
  have e1 : (M + N) * (K + 32) = M * K + N * K + 32 * M + 32 * N := by ring
  have e2 : M * (K + 32) = M * K + 32 * M := by ring
  have hjK : j * K + K ≤ N * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hj
  simp only [rowIn, ↓reduceIte] at ha
  simp only [row4B]
  unfold inW4 portB8 eaAt bAt ebAt faAt fbAt gAt
  rw [e1, e2]
  split_ifs <;> omega

theorem hinB4 : ∀ a < (blockB4 ops M K N).S.N,
    ((blockB4 ops M K N).S.gate a).input = true ↔ a < (blockB4 ops M K N).nIn :=
  Rows.input_iff ops (row4B K)

theorem hinA4 : ∀ a < (blockA4 ops M K N).S.N,
    ((blockA4 ops M K N).S.gate a).input = true ↔ a < (blockA4 ops M K N).nIn :=
  Rows.input_iff ops (row4A K)

theorem tile4_input_iff : ∀ a < (tile4Spec K TM TN).N,
    ((tile4Spec K TM TN).gate a).input = true ↔ a < tile4In K TM TN := by
  intro a _
  refine ⟨fun h => ?_, gate4_input K TM TN⟩
  by_contra hc
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have h1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have h2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hk : (TM + TN) * row4 K ≤ tile4In K TM TN := by unfold tile4In; omega
  have : (tile4Gate K TM TN a).input = false := by
    unfold tile4Gate
    rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
      ite_eq_right hc]
    simp only [entry4Gate]
    split_ifs <;> rfl
  exact absurd h (by rw [show (tile4Spec K TM TN).gate a = tile4Gate K TM TN a from rfl, this]; simp)

theorem hinT4 : ∀ a < (blockT4 M K N TM TN).S.N,
    ((blockT4 M K N TM TN).S.gate a).input = true ↔ a < (blockT4 M K N TM TN).nIn :=
  tile4_input_iff K TM TN

theorem hnB4 : (blockB4 ops M K N).nIn ≤ (blockB4 ops M K N).S.N :=
  le_of_lt (row4B_in_lt K)

theorem hnA4 : (blockA4 ops M K N).nIn ≤ (blockA4 ops M K N).S.N :=
  le_of_lt (row4A_in_lt K)

theorem hnT4 : (blockT4 M K N TM TN).nIn ≤ (blockT4 M K N TM TN).S.N := by
  show tile4In K TM TN ≤ tile4N K TM TN
  unfold tile4N; omega

/-! ### Every port reads a gate of its input's width -/

theorem hfit4B (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {j : ℕ} (hj : j < N) :
    ∀ a < (blockB4 ops M K N).nIn,
      val X ((blockB4 ops M K N).port j a) < 2 ^ ((blockB4 ops M K N).S.gate a).w := by
  intro a ha
  have hlt : portB8 M N K j a < (whole4 ops M K N TM TN).nIn := portB8_lt hj ha
  have h := (whole4 ops M K N TM TN).val_lt X (show _ < (whole4 ops M K N TM TN).spec.toCircuit.N from
    lt_of_lt_of_le hlt (Nat.le_add_right _ _))
  rw [(whole4 ops M K N TM TN).gate_in hlt] at h
  show val X (portB8 M N K j a) < 2 ^ (((row4B K).spec ops).gate a).w
  rw [Rows.gate_in ops (row4B K) ha, ← inW4_portB8 M K N hj ha]
  exact h

theorem hfit4A (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {i : ℕ} (hi : i < M) :
    ∀ a < (blockA4 ops M K N).nIn,
      val X ((blockA4 ops M K N).port i a) < 2 ^ ((blockA4 ops M K N).S.gate a).w := by
  intro a ha
  have hlt : portA8 M N K i a < (whole4 ops M K N TM TN).nIn := portA8_lt hi ha
  have h := (whole4 ops M K N TM TN).val_lt X (show _ < (whole4 ops M K N TM TN).spec.toCircuit.N from
    lt_of_lt_of_le hlt (Nat.le_add_right _ _))
  rw [(whole4 ops M K N TM TN).gate_in hlt] at h
  show val X (portA8 M N K i a) < 2 ^ (((row4A K).spec ops).gate a).w
  rw [Rows.gate_in ops (row4A K) ha, ← inW4_portA8 M K N hi ha]
  exact h

omit ops in
theorem tile4Gate_w {a : ℕ} (ha : a < tile4In K TM TN) : (tile4Gate K TM TN a).w =
    if a < TM * K then 4 else if a < TM * row4 K then 8 else if a < TM * row4 K + TN * K then 4
    else if a < (TM + TN) * row4 K then 8 else 16 := by
  unfold tile4Gate
  split_ifs <;> rfl

theorem hfit4T (h16 : 16 ∣ K) (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {e : ℕ}
    (he : e < M / TM * (N / TN)) : ∀ a < (blockT4 M K N TM TN).nIn,
      val X ((blockT4 M K N TM TN).port e a) < 2 ^ ((blockT4 M K N TM TN).S.gate a).w := by
  intro a ha
  have hN : portT4 M N K TM TN e a < (whole4 ops M K N TM TN).spec.toCircuit.N := by
    rw [whole4_N]; have := portT4_lt he ha; unfold tiles4 at this; omega
  refine lt_of_lt_of_eq ((whole4 ops M K N TM TN).val_lt X hN) ?_
  congr 1
  change a < tile4In K TM TN at ha
  show ((whole4 ops M K N TM TN).gate (portT4 M N K TM TN e a)).w = (tile4Gate K TM TN a).w
  rw [tile4Gate_w K TM TN ha]
  obtain ⟨c, hc⟩ := h16
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hsplit' : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  unfold portT4
  unfold tile4In at ha
  split_ifs with h1 h2 h3 h4 h5
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by simp [h] at h1
    have hq := tile_row_lt he ((Nat.div_lt_iff_lt_mul hK).2 h1)
    have hb := code4_bounds (b := rowIn K false) (Nat.mod_lt a hK)
    rw [rowA4_eq ops M K N TM TN, Whole.inst_w _ (whole4_A ops M K N TM TN) hq hb.1 (code4A_lt (Nat.mod_lt a hK))]
    exact row4A_code_w ops (show a % K < 16 * (K / 16) by have := Nat.mod_lt a hK; omega)
  · have hS0 : 0 < K / 16 := Nat.pos_of_ne_zero fun h => by
      have : TM * (K / 16) = 0 := by rw [h, Nat.mul_zero]
      omega
    have hq := tile_row_lt he ((Nat.div_lt_iff_lt_mul hS0).2 (show a - TM * K < TM * (K / 16) by omega))
    have hb := scale4_bounds (b := rowIn K false) (Nat.mod_lt (a - TM * K) hS0)
    rw [rowA4_eq ops M K N TM TN, Whole.inst_w _ (whole4_A ops M K N TM TN) hq hb.1
      (scale4A_lt (Nat.mod_lt _ hS0))]
    exact row4A_scale_w ops (Nat.mod_lt _ hS0)
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by
      have : TN * K = 0 := by rw [h, Nat.mul_zero]
      omega
    have hq := tile_col_lt he ((Nat.div_lt_iff_lt_mul hK).2 (show a - TM * row4 K < TN * K by omega))
    have hb := code4_bounds (b := rowIn K true) (Nat.mod_lt (a - TM * row4 K) hK)
    rw [rowB4_eq ops M K N TM TN, Whole.inst_w _ (whole4_B ops M K N TM TN) hq hb.1
      (code4B_lt (Nat.mod_lt _ hK))]
    exact row4B_code_w ops (show (a - TM * row4 K) % K < 16 * (K / 16) by
      have := Nat.mod_lt (a - TM * row4 K) hK; omega)
  · have hS0 : 0 < K / 16 := Nat.pos_of_ne_zero fun h => by
      have : TN * (K / 16) = 0 := by rw [h, Nat.mul_zero]
      omega
    have hq := tile_col_lt he ((Nat.div_lt_iff_lt_mul hS0).2
      (show a - (TM * row4 K + TN * K) < TN * (K / 16) by omega))
    have hb := scale4_bounds (b := rowIn K true) (Nat.mod_lt (a - (TM * row4 K + TN * K)) hS0)
    rw [rowB4_eq ops M K N TM TN, Whole.inst_w _ (whole4_B ops M K N TM TN) hq hb.1
      (scale4B_lt (Nat.mod_lt _ hS0))]
    exact row4B_scale_w ops (Nat.mod_lt _ hS0)
  · have hq := tile_row_lt he ((Nat.div_lt_iff_lt_mul (by norm_num)).2
      (show a - (TM + TN) * row4 K < TM * 64 by omega))
    have hb := paOut4_bounds (K := K) (Nat.mod_lt (a - (TM + TN) * row4 K) (show 0 < 64 by norm_num))
    rw [rowA4_eq ops M K N TM TN, Whole.inst_w _ (whole4_A ops M K N TM TN) hq hb.1 hb.2]
    exact row4A_pa_w ops (Nat.mod_lt _ (by norm_num))
  · have hq := tile_col_lt he ((Nat.div_lt_iff_lt_mul (by norm_num)).2
      (show a - ((TM + TN) * row4 K + 64 * TM) < TN * 64 by omega))
    have hb := pbOut4_bounds (K := K) (Nat.mod_lt (a - ((TM + TN) * row4 K + 64 * TM)) (show 0 < 64 by norm_num))
    rw [rowB4_eq ops M K N TM TN, Whole.inst_w _ (whole4_B ops M K N TM TN) hq hb.1 hb.2]
    exact row4B_pb_w ops (Nat.mod_lt _ (by norm_num))

end Whole4

/-! ## What a tile's entry reads -/

section Reads4

variable (K TM TN : ℕ)

/-- Entry `(i, j)`'s firing `l / 64` reads code `l` of `A′_i` and of `B̃_j`. -/
theorem tile4_reads_code {i j l : ℕ} (hj : j < TN) (hl : l < 64 * (K / 64)) :
    i * K + l ∈ (tile4Gate K TM TN (entry4Base K TM TN i j + l / 64)).rd ∧
      TM * row4 K + j * K + l ∈ (tile4Gate K TM TN (entry4Base K TM TN i j + l / 64)).rd := by
  have hq : l / 64 < K / 64 := by omega
  rw [gate4_entry K TM TN hj (by unfold block4Len; omega), entry4_atom K _ _ _ _ _ _ _ hq]
  exact ⟨List.mem_append_left _ (List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨l % 64, List.mem_range.2 (by omega), by omega⟩)))),
    List.mem_append_left _ (List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨l % 64, List.mem_range.2 (by omega), by omega⟩)))⟩

/-- Entry `(i, j)`'s firing `l / 4` reads scale byte `l` of `A′_i` and of `B̃_j`. -/
theorem tile4_reads_scale {i j l : ℕ} (hj : j < TN) (hl : l < 4 * (K / 64)) :
    TM * K + i * (K / 16) + l ∈ (tile4Gate K TM TN (entry4Base K TM TN i j + l / 4)).rd ∧
      TM * row4 K + TN * K + j * (K / 16) + l ∈ (tile4Gate K TM TN (entry4Base K TM TN i j + l / 4)).rd := by
  have hq : l / 4 < K / 64 := by omega
  rw [gate4_entry K TM TN hj (by unfold block4Len; omega), entry4_atom K _ _ _ _ _ _ _ hq]
  exact ⟨List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨l % 4, List.mem_range.2 (by omega), by omega⟩)),
    List.mem_append_right _ (List.mem_map.2 ⟨l % 4, List.mem_range.2 (by omega), by omega⟩)⟩

/-- Entry `(i, j)`'s K16 step `q / 16` reads word `q` of `P_A,i` and of `P_B,j`. -/
theorem tile4_reads_P {i j q : ℕ} (hj : j < TN) (hq : q < 64) :
    (TM + TN) * row4 K + 64 * i + q ∈ (tile4Gate K TM TN (entry4Base K TM TN i j + (K / 64 + q / 16))).rd ∧
      (TM + TN) * row4 K + 64 * TM + 64 * j + q ∈
        (tile4Gate K TM TN (entry4Base K TM TN i j + (K / 64 + q / 16))).rd := by
  rw [gate4_entry K TM TN hj (by unfold block4Len; omega), entry4_peel K _ _ _ _ _ _ _ (q / 16)]
  exact ⟨List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨q % 16, List.mem_range.2 (by omega), by omega⟩)),
    List.mem_append_right _ (List.mem_map.2 ⟨q % 16, List.mem_range.2 (by omega), by omega⟩)⟩

theorem u4Gate_eq (i j : ℕ) : u4Gate K TM TN i j = entry4Base K TM TN i j + (K / 64 + 48 / 16) := by
  unfold u4Gate block4Len; omega

theorem c4Gate_eq (hK : 0 < K / 64) (i j : ℕ) : c4Gate K TM TN i j = entry4Base K TM TN i j + (K / 64 - 1) := by
  unfold c4Gate; omega

theorem entry4_bounds {i j r : ℕ} (hi : i < TM) (hj : j < TN) (hr : r < block4Len K) :
    tile4In K TM TN ≤ entry4Base K TM TN i j + r ∧ entry4Base K TM TN i j + r < tile4N K TM TN := by
  have := entry4_lt K TM TN hi hj
  unfold entry4Base at this ⊢
  omega

end Reads4

section Ports4

variable {M N K TM TN : ℕ} (e : ℕ)

theorem portT4_A {ti l : ℕ} (hti : ti < TM) (hl : l < K) :
    portT4 M N K TM TN e (ti * K + l) = rowA4 M N K (e / (N / TN) * TM + ti) (code4 K (rowIn K false) l) := by
  have hK : 0 < K := by omega
  unfold portT4
  rw [ite_eq_left (mul_add_lt hti hl), show (ti * K + l) / K = ti by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (ti * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem portT4_SA {ti l : ℕ} (hti : ti < TM) (hl : l < K / 16) :
    portT4 M N K TM TN e (TM * K + ti * (K / 16) + l) =
      rowA4 M N K (e / (N / TN) * TM + ti) (scale4 K (rowIn K false) l) := by
  have hK : 0 < K / 16 := by omega
  have h1 := mul_add_lt hti hl
  have hT := row4_mul K TM
  unfold portT4
  rw [ite_eq_right (by omega), ite_eq_left (by omega), show TM * K + ti * (K / 16) + l - TM * K = ti * (K / 16) + l by
      omega,
    show (ti * (K / 16) + l) / (K / 16) = ti by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (ti * (K / 16) + l) % (K / 16) = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem portT4_B {tj l : ℕ} (htj : tj < TN) (hl : l < K) :
    portT4 M N K TM TN e (TM * row4 K + tj * K + l) =
      rowB4 M N K (e % (N / TN) * TN + tj) (code4 K (rowIn K true) l) := by
  have hK : 0 < K := by omega
  have h1 := mul_add_lt htj hl
  have hT := row4_mul K TM
  unfold portT4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show TM * row4 K + tj * K + l - TM * row4 K = tj * K + l by omega,
    show (tj * K + l) / K = tj by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (tj * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem portT4_SB {tj l : ℕ} (htj : tj < TN) (hl : l < K / 16) :
    portT4 M N K TM TN e (TM * row4 K + TN * K + tj * (K / 16) + l) =
      rowB4 M N K (e % (N / TN) * TN + tj) (scale4 K (rowIn K true) l) := by
  have hK : 0 < K / 16 := by omega
  have h1 := mul_add_lt htj hl
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  unfold portT4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show TM * row4 K + TN * K + tj * (K / 16) + l - (TM * row4 K + TN * K) = tj * (K / 16) + l by omega,
    show (tj * (K / 16) + l) / (K / 16) = tj by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (tj * (K / 16) + l) % (K / 16) = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem portT4_PA {ti q : ℕ} (hti : ti < TM) (hq : q < 64) :
    portT4 M N K TM TN e ((TM + TN) * row4 K + 64 * ti + q) =
      rowA4 M N K (e / (N / TN) * TM + ti) (paOut4 K q) := by
  have h1 := mul_add_lt (L := 64) hti hq
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  unfold portT4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_left (by omega), show (TM + TN) * row4 K + 64 * ti + q - (TM + TN) * row4 K = ti * 64 + q by omega,
    show (ti * 64 + q) / 64 = ti by omega, show (ti * 64 + q) % 64 = q by omega]

theorem portT4_PB {tj q : ℕ} (_htj : tj < TN) (hq : q < 64) :
    portT4 M N K TM TN e ((TM + TN) * row4 K + 64 * TM + 64 * tj + q) =
      rowB4 M N K (e % (N / TN) * TN + tj) (pbOut4 K q) := by
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  unfold portT4
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    ite_eq_right (by omega),
    show (TM + TN) * row4 K + 64 * TM + 64 * tj + q - ((TM + TN) * row4 K + 64 * TM) = tj * 64 + q by omega,
    show (tj * 64 + q) / 64 = tj by omega, show (tj * 64 + q) % 64 = q by omega]

end Ports4

/-! ## The units' cuts -/

section Cuts4

variable (M K N TM TN : ℕ)

/-- RowB `j`'s cut holds on its inputs the transcript's values at its ports. -/
theorem cutB4_input (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {j : ℕ} (hj : j < N) {a : ℕ}
    (ha : a < rowIn K true) :
    val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) a = val X (portB8 M N K j a) :=
  (whole4 ops M K N TM TN).cut_input (whole4_B ops M K N TM TN) hj X (hinB4 ops M K N)
    (hfit4B ops M K N TM TN X hj) (hnB4 ops M K N) ha

/-- RowA `i`'s cut holds on its inputs the transcript's values at its ports. -/
theorem cutA4_input (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {i : ℕ} (hi : i < M) {a : ℕ}
    (ha : a < rowIn K false) :
    val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) a = val X (portA8 M N K i a) :=
  (whole4 ops M K N TM TN).cut_input (whole4_A ops M K N TM TN) hi X (hinA4 ops M K N)
    (hfit4A ops M K N TM TN X hi) (hnA4 ops M K N) ha

/-- Tile `e`'s cut holds on its inputs the transcript's values at its ports. -/
theorem cutT4_input (h16 : 16 ∣ K) (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {e : ℕ}
    (he : e < M / TM * (N / TN)) {a : ℕ} (ha : a < tile4In K TM TN) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) e) a = val X (portT4 M N K TM TN e a) :=
  (whole4 ops M K N TM TN).cut_input (whole4_T ops M K N TM TN) he X (hinT4 M K N TM TN)
    (hfit4T ops M K N TM TN h16 X he) (hnT4 M K N TM TN) ha

/-- Where RowB `j` is correct, its committed gates hold its cut's values. -/
theorem cutB4_at (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {j : ℕ} (hj : j < N)
    (hX : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_B ops M K N TM TN) hj⟩)
    {a : ℕ} (ha1 : rowIn K true ≤ a) (ha2 : a < (row4B K).N) {h : rowB4 M N K j a < _}
    (hcom : (⟨rowB4 M N K j a, h⟩ : Fin (whole4 ops M K N TM TN).spec.toCircuit.N) ∈
      (whole4 ops M K N TM TN).part.committed) :
    val X (rowB4 M N K j a) = val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) a := by
  rw [rowB4_eq ops M K N TM TN]
  exact (whole4 ops M K N TM TN).cut_committed (whole4_B ops M K N TM TN) hj X (hinB4 ops M K N)
    (hfit4B ops M K N TM TN X hj) hX ha1 ha2 (fin_mem (rowB4_eq ops M K N TM TN j a) hcom)

/-- Where RowA `i` is correct, its committed gates hold its cut's values. -/
theorem cutA4_at (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {i : ℕ} (hi : i < M)
    (hX : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_A ops M K N TM TN) hi⟩)
    {a : ℕ} (ha1 : rowIn K false ≤ a) (ha2 : a < (row4A K).N) {h : rowA4 M N K i a < _}
    (hcom : (⟨rowA4 M N K i a, h⟩ : Fin (whole4 ops M K N TM TN).spec.toCircuit.N) ∈
      (whole4 ops M K N TM TN).part.committed) :
    val X (rowA4 M N K i a) = val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) a := by
  rw [rowA4_eq ops M K N TM TN]
  exact (whole4 ops M K N TM TN).cut_committed (whole4_A ops M K N TM TN) hi X (hinA4 ops M K N)
    (hfit4A ops M K N TM TN X hi) hX ha1 ha2 (fin_mem (rowA4_eq ops M K N TM TN i a) hcom)

/-- Where tile `e` is correct, its committed gates hold its cut's values. -/
theorem cutT4_at (h16 : 16 ∣ K) (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {e : ℕ}
    (he : e < M / TM * (N / TN))
    (hX : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_T ops M K N TM TN) he⟩)
    {a : ℕ} (ha1 : tile4In K TM TN ≤ a) (ha2 : a < tile4N K TM TN) {h : tile4 M N K TM TN e a < _}
    (hcom : (⟨tile4 M N K TM TN e a, h⟩ : Fin (whole4 ops M K N TM TN).spec.toCircuit.N) ∈
      (whole4 ops M K N TM TN).part.committed) :
    val X (tile4 M N K TM TN e a) = val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) e) a := by
  rw [tile4_eq ops M K N TM TN]
  exact (whole4 ops M K N TM TN).cut_committed (whole4_T ops M K N TM TN) he X (hinT4 M K N TM TN)
    (hfit4T ops M K N TM TN h16 X he) hX ha1 ha2 (fin_mem (tile4_eq ops M K N TM TN e a) hcom)

end Cuts4

/-! ## An entry's cone -/

section Cone4

variable (M K N TM TN : ℕ)

theorem rowA4_ltN {i a : ℕ} (hi : i < M) (ha : a < (row4A K).N) :
    rowA4 M N K i a < (whole4 ops M K N TM TN).spec.toCircuit.N := by
  rw [whole4_N]; have := rowA4_lt (N := N) hi ha; unfold tiles4 at this; omega

theorem rowB4_ltN {j a : ℕ} (hj : j < N) (ha : a < (row4B K).N) :
    rowB4 M N K j a < (whole4 ops M K N TM TN).spec.toCircuit.N := by
  rw [whole4_N]; have := rowB4_lt (M := M) hj ha; omega

theorem tile4_ltN {e a : ℕ} (he : e < M / TM * (N / TN)) (ha1 : tile4In K TM TN ≤ a) (ha2 : a < tile4N K TM TN) :
    tile4 M N K TM TN e a < (whole4 ops M K N TM TN).spec.toCircuit.N := by
  rw [tile4_eq ops M K N TM TN]
  exact (whole4 ops M K N TM TN).inst_lt (whole4_T ops M K N TM TN) he ha1 ha2

/-- Entry `(i, j)`'s `U` is its tile's copy of the block's gate `K / 64 + 3`. -/
theorem uGlob4_eq (i j : ℕ) : uGlob4 M N K TM TN i j = (whole4 ops M K N TM TN).nIn +
    blocksLen [blockB4 ops M K N, blockA4 ops M K N] + (i / TM * (N / TN) + j / TN) * (blockT4 M K N TM TN).len +
      (entry4Base K TM TN (i % TM) (j % TN) + (K / 64 + 48 / 16) - (blockT4 M K N TM TN).nIn) := by
  unfold uGlob4; rw [u4Gate_eq]; exact tile4_eq ops M K N TM TN _ _

/-- Entry `(i, j)`'s tile unit is in the cone of its `C̃`. -/
theorem cone4T {T : Finset (Fin (whole4 ops M K N TM TN).spec.toCircuit.N)} {i j : ℕ} (hTM : TM ∣ M)
    (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hK : 0 < K / 64)
    (hc : ∀ g : Fin (whole4 ops M K N TM TN).spec.toCircuit.N, g.val = cGlob4 M N K TM TN i j → g ∈ T) :
    (⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_T ops M K N TM TN) (tile_index hTM hTN hi hj).1⟩ :
      Fin (unitCount (whole4 ops M K N TM TN).blocks)) ∈ (whole4 ops M K N TM TN).part.cone T := by
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 - 1) (by unfold block4Len; omega)
  rw [← c4Gate_eq K TM TN hK] at hb
  exact (whole4 ops M K N TM TN).part.mem_cone (hc _ (tile4_eq ops M K N TM TN _ _).symm) (.refl _)
    ((whole4 ops M K N TM TN).unit_inst (whole4_T ops M K N TM TN) (tile_index hTM hTN hi hj).1 hb.1 hb.2)

/-- Entry `(i, j)`'s RowA unit is in the cone of its `U`, whose last K16 step reads `P_A,i`'s word 48. -/
theorem cone4A {T : Finset (Fin (whole4 ops M K N TM TN).spec.toCircuit.N)} {i j : ℕ} (hTM : TM ∣ M)
    (hTN : TN ∣ N) (hi : i < M) (hj : j < N)
    (hu : ∀ g : Fin (whole4 ops M K N TM TN).spec.toCircuit.N, g.val = uGlob4 M N K TM TN i j → g ∈ T) :
    (⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_A ops M K N TM TN) hi⟩ :
      Fin (unitCount (whole4 ops M K N TM TN).blocks)) ∈ (whole4 ops M K N TM TN).part.cone T := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 + 48 / 16) (by unfold block4Len; omega)
  have hpa := tile4_reads_P K TM TN (i := i % TM) htj (q := 48) (by norm_num)
  have hpb := paOut4_bounds (K := K) (show 48 < 64 by norm_num)
  have hm := (whole4 ops M K N TM TN).port_reads (whole4_T ops M K N TM TN) he hb.1 hb.2
    (show (TM + TN) * row4 K + 64 * (i % TM) + 48 < tile4In K TM TN by unfold tile4In; omega) hpa.1
  refine (whole4 ops M K N TM TN).part.mem_cone (hu _ (uGlob4_eq ops M K N TM TN i j).symm)
    (.step (reads_eq (ht' := (whole4 ops M K N TM TN).inst_lt (whole4_T ops M K N TM TN) he hb.1 hb.2) ?_ rfl hm)
      (.refl _))
    ((whole4 ops M K N TM TN).unit_inst (whole4_A ops M K N TM TN) hi hpb.1 hpb.2)
  show portT4 M N K TM TN _ _ = _
  rw [portT4_PA _ hti (by norm_num), hie, rowA4_eq ops M K N TM TN]

/-- Entry `(i, j)`'s RowB unit is in the cone of its `U`, whose last K16 step reads `P_B,j`'s word 48. -/
theorem cone4B {T : Finset (Fin (whole4 ops M K N TM TN).spec.toCircuit.N)} {i j : ℕ} (hTM : TM ∣ M)
    (hTN : TN ∣ N) (hi : i < M) (hj : j < N)
    (hu : ∀ g : Fin (whole4 ops M K N TM TN).spec.toCircuit.N, g.val = uGlob4 M N K TM TN i j → g ∈ T) :
    (⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_B ops M K N TM TN) hj⟩ :
      Fin (unitCount (whole4 ops M K N TM TN).blocks)) ∈ (whole4 ops M K N TM TN).part.cone T := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 + 48 / 16) (by unfold block4Len; omega)
  have hpa := tile4_reads_P K TM TN (i := i % TM) htj (q := 48) (by norm_num)
  have hpb := pbOut4_bounds (K := K) (show 48 < 64 by norm_num)
  have hm := (whole4 ops M K N TM TN).port_reads (whole4_T ops M K N TM TN) he hb.1 hb.2
    (show (TM + TN) * row4 K + 64 * TM + 64 * (j % TN) + 48 < tile4In K TM TN by unfold tile4In; omega) hpa.2
  refine (whole4 ops M K N TM TN).part.mem_cone (hu _ (uGlob4_eq ops M K N TM TN i j).symm)
    (.step (reads_eq (ht' := (whole4 ops M K N TM TN).inst_lt (whole4_T ops M K N TM TN) he hb.1 hb.2) ?_ rfl hm)
      (.refl _))
    ((whole4 ops M K N TM TN).unit_inst (whole4_B ops M K N TM TN) hj hpb.1 hpb.2)
  show portT4 M N K TM TN _ _ = _
  rw [portT4_PB _ htj (by norm_num), hje, rowB4_eq ops M K N TM TN]

theorem outs4_C {i j : ℕ} (hi : i < M) (hj : j < N) : cGlob4 M N K TM TN i j ∈ outs4 M N K TM TN := by
  unfold outs4
  exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_left _
    (Finset.mem_image.2 ⟨(i, j), Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hj⟩, rfl⟩)))

theorem outs4_U {i j : ℕ} (hi : i < M) (hj : j < N) : uGlob4 M N K TM TN i j ∈ outs4 M N K TM TN := by
  unfold outs4
  exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_right _
    (Finset.mem_image.2 ⟨(i, j), Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hj⟩, rfl⟩)))

end Cone4

/-! ## A tile's cut reads its rows' cuts -/

theorem h16_of {K : ℕ} (hk : 64 ∣ K) : 16 ∣ K := Nat.dvd_trans (by norm_num) hk

section Chain4

variable (M K N TM TN : ℕ) (X : (whole4 ops M K N TM TN).spec.toCircuit.Transcript) {i j : ℕ}
  (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hk : 64 ∣ K)

include hTM hTN hi hj hk

/-- Where its tile unit is correct, entry `(i, j)`'s `C̃` holds the tile cut's. -/
theorem chain4_C (hK : 0 < K / 64)
    (hXT : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_T ops M K N TM TN) (tile_index hTM hTN hi hj).1⟩) :
    val X (cGlob4 M N K TM TN i j) = val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN)
      (i / TM * (N / TN) + j / TN)) (c4Gate K TM TN (i % TM) (j % TN)) := by
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 - 1) (by unfold block4Len; omega)
  rw [← c4Gate_eq K TM TN hK] at hb
  exact cutT4_at ops M K N TM TN (h16_of hk) X (tile_index hTM hTN hi hj).1 hXT hb.1 hb.2
    (h := tile4_ltN ops M K N TM TN (tile_index hTM hTN hi hj).1 hb.1 hb.2)
    ((whole4 ops M K N TM TN).part.mem_committed_of_output (by
      simp only [Spec.toCircuit, Finset.mem_filter, Finset.mem_univ, true_and]
      exact outs4_C M K N TM TN hi hj))

/-- Where its tile unit is correct, entry `(i, j)`'s `U` holds the tile cut's. -/
theorem chain4_U
    (hXT : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_T ops M K N TM TN) (tile_index hTM hTN hi hj).1⟩) :
    val X (uGlob4 M N K TM TN i j) = val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN)
      (i / TM * (N / TN) + j / TN)) (u4Gate K TM TN (i % TM) (j % TN)) := by
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 + 48 / 16) (by unfold block4Len; omega)
  rw [← u4Gate_eq] at hb
  exact cutT4_at ops M K N TM TN (h16_of hk) X (tile_index hTM hTN hi hj).1 hXT hb.1 hb.2
    (h := tile4_ltN ops M K N TM TN (tile_index hTM hTN hi hj).1 hb.1 hb.2)
    ((whole4 ops M K N TM TN).part.mem_committed_of_output (by
      simp only [Spec.toCircuit, Finset.mem_filter, Finset.mem_univ, true_and]
      exact outs4_U M K N TM TN hi hj))

/-- Where RowA `i` is correct, the tile cut's row `i mod TM` of `A′`'s codes is RowA `i`'s cut's. -/
theorem chain4_A
    (hXA : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_A ops M K N TM TN) hi⟩) {l : ℕ} (hl : l < K) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN)) (i % TM * K + l) =
      val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (code4 K (rowIn K false) l) := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hl' : l < 64 * (K / 64) := by omega
  have hlt := mul_add_lt hti hl
  have ha : i % TM * K + l < tile4In K TM TN := by unfold tile4In; omega
  have hb := entry4_bounds K TM TN hti htj (r := l / 64) (by unfold block4Len; omega)
  have hc := code4_bounds (b := rowIn K false) hl
  have hcom := (whole4 ops M K N TM TN).port_committed (whole4_T ops M K N TM TN) he hb.1 hb.2 ha
    (tile4_reads_code K TM TN htj hl').1
  rw [cutT4_input ops M K N TM TN (h16_of hk) X he ha]
  show val X (portT4 M N K TM TN _ _) = _
  rw [portT4_A _ hti hl, hie]
  exact cutA4_at ops M K N TM TN X hi hXA hc.1 (code4A_lt hl)
    (h := rowA4_ltN ops M K N TM TN hi (code4A_lt hl))
    (fin_mem (by show portT4 M N K TM TN _ _ = _; rw [portT4_A _ hti hl, hie]) hcom)

/-- Where RowA `i` is correct, the tile cut's row `i mod TM` of `A′`'s scale bytes is RowA `i`'s cut's. -/
theorem chain4_SA
    (hXA : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_A ops M K N TM TN) hi⟩) {l : ℕ} (hl : l < K / 16) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN))
        (TM * K + i % TM * (K / 16) + l) =
      val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (scale4 K (rowIn K false) l) := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hl' : l < 4 * (K / 64) := by omega
  have hlt := mul_add_lt hti hl
  have ha : TM * K + i % TM * (K / 16) + l < tile4In K TM TN := by unfold tile4In; omega
  have hb := entry4_bounds K TM TN hti htj (r := l / 4) (by unfold block4Len; omega)
  have hc := scale4_bounds (b := rowIn K false) hl
  have hcom := (whole4 ops M K N TM TN).port_committed (whole4_T ops M K N TM TN) he hb.1 hb.2 ha
    (tile4_reads_scale K TM TN htj hl').1
  rw [cutT4_input ops M K N TM TN (h16_of hk) X he ha]
  show val X (portT4 M N K TM TN _ _) = _
  rw [portT4_SA _ hti hl, hie]
  exact cutA4_at ops M K N TM TN X hi hXA hc.1 (scale4A_lt hl)
    (h := rowA4_ltN ops M K N TM TN hi (scale4A_lt hl))
    (fin_mem (by show portT4 M N K TM TN _ _ = _; rw [portT4_SA _ hti hl, hie]) hcom)

/-- Where RowB `j` is correct, the tile cut's row `j mod TN` of `B̃`'s codes is RowB `j`'s cut's. -/
theorem chain4_B
    (hXB : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_B ops M K N TM TN) hj⟩) {l : ℕ} (hl : l < K) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN))
        (TM * row4 K + j % TN * K + l) =
      val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (code4 K (rowIn K true) l) := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hl' : l < 64 * (K / 64) := by omega
  have hlt := mul_add_lt htj hl
  have ha : TM * row4 K + j % TN * K + l < tile4In K TM TN := by unfold tile4In; omega
  have hb := entry4_bounds K TM TN hti htj (r := l / 64) (by unfold block4Len; omega)
  have hc := code4_bounds (b := rowIn K true) hl
  have hcom := (whole4 ops M K N TM TN).port_committed (whole4_T ops M K N TM TN) he hb.1 hb.2 ha
    (tile4_reads_code K TM TN (i := i % TM) htj hl').2
  rw [cutT4_input ops M K N TM TN (h16_of hk) X he ha]
  show val X (portT4 M N K TM TN _ _) = _
  rw [portT4_B _ htj hl, hje]
  exact cutB4_at ops M K N TM TN X hj hXB hc.1 (code4B_lt hl)
    (h := rowB4_ltN ops M K N TM TN hj (code4B_lt hl))
    (fin_mem (by show portT4 M N K TM TN _ _ = _; rw [portT4_B _ htj hl, hje]) hcom)

/-- Where RowB `j` is correct, the tile cut's row `j mod TN` of `B̃`'s scale bytes is RowB `j`'s cut's. -/
theorem chain4_SB
    (hXB : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_B ops M K N TM TN) hj⟩) {l : ℕ} (hl : l < K / 16) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN))
        (TM * row4 K + TN * K + j % TN * (K / 16) + l) =
      val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (scale4 K (rowIn K true) l) := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hT := row4_mul K TM
  have hS := row4_mul K (TM + TN)
  have hs1 : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hs2 : (TM + TN) * (K / 16) = TM * (K / 16) + TN * (K / 16) := Nat.add_mul _ _ _
  have hl' : l < 4 * (K / 64) := by omega
  have hlt := mul_add_lt htj hl
  have ha : TM * row4 K + TN * K + j % TN * (K / 16) + l < tile4In K TM TN := by unfold tile4In; omega
  have hb := entry4_bounds K TM TN hti htj (r := l / 4) (by unfold block4Len; omega)
  have hc := scale4_bounds (b := rowIn K true) hl
  have hcom := (whole4 ops M K N TM TN).port_committed (whole4_T ops M K N TM TN) he hb.1 hb.2 ha
    (tile4_reads_scale K TM TN (i := i % TM) htj hl').2
  rw [cutT4_input ops M K N TM TN (h16_of hk) X he ha]
  show val X (portT4 M N K TM TN _ _) = _
  rw [portT4_SB _ htj hl, hje]
  exact cutB4_at ops M K N TM TN X hj hXB hc.1 (scale4B_lt hl)
    (h := rowB4_ltN ops M K N TM TN hj (scale4B_lt hl))
    (fin_mem (by show portT4 M N K TM TN _ _ = _; rw [portT4_SB _ htj hl, hje]) hcom)

/-- Where RowA `i` is correct, the tile cut's `P_A` row `i mod TM` is RowA `i`'s cut's `P_A` words. -/
theorem chain4_PA
    (hXA : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_A ops M K N TM TN) hi⟩) {q : ℕ} (hq : q < 64) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN))
        ((TM + TN) * row4 K + 64 * (i % TM) + q) =
      val ((whole4 ops M K N TM TN).cut X (blockA4 ops M K N) i) (paOut4 K q) := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have ha : (TM + TN) * row4 K + 64 * (i % TM) + q < tile4In K TM TN := by unfold tile4In; omega
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 + q / 16) (by unfold block4Len; omega)
  have hc := paOut4_bounds (K := K) hq
  have hcom := (whole4 ops M K N TM TN).port_committed (whole4_T ops M K N TM TN) he hb.1 hb.2 ha
    (tile4_reads_P K TM TN (i := i % TM) htj hq).1
  rw [cutT4_input ops M K N TM TN (h16_of hk) X he ha]
  show val X (portT4 M N K TM TN _ _) = _
  rw [portT4_PA _ hti hq, hie]
  exact cutA4_at ops M K N TM TN X hi hXA hc.1 hc.2 (h := rowA4_ltN ops M K N TM TN hi hc.2)
    (fin_mem (by show portT4 M N K TM TN _ _ = _; rw [portT4_PA _ hti hq, hie]) hcom)

/-- Where RowB `j` is correct, the tile cut's `P_B` row `j mod TN` is RowB `j`'s cut's `P_B` words. -/
theorem chain4_PB
    (hXB : (whole4 ops M K N TM TN).part.Correct X
      ⟨_, (whole4 ops M K N TM TN).unit_lt (whole4_B ops M K N TM TN) hj⟩) {q : ℕ} (hq : q < 64) :
    val ((whole4 ops M K N TM TN).cut X (blockT4 M K N TM TN) (i / TM * (N / TN) + j / TN))
        ((TM + TN) * row4 K + 64 * TM + 64 * (j % TN) + q) =
      val ((whole4 ops M K N TM TN).cut X (blockB4 ops M K N) j) (pbOut4 K q) := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have ha : (TM + TN) * row4 K + 64 * TM + 64 * (j % TN) + q < tile4In K TM TN := by unfold tile4In; omega
  have hb := entry4_bounds K TM TN hti htj (r := K / 64 + q / 16) (by unfold block4Len; omega)
  have hc := pbOut4_bounds (K := K) hq
  have hcom := (whole4 ops M K N TM TN).port_committed (whole4_T ops M K N TM TN) he hb.1 hb.2 ha
    (tile4_reads_P K TM TN (i := i % TM) htj hq).2
  rw [cutT4_input ops M K N TM TN (h16_of hk) X he ha]
  show val X (portT4 M N K TM TN _ _) = _
  rw [portT4_PB _ htj hq, hje]
  exact cutB4_at ops M K N TM TN X hj hXB hc.1 hc.2 (h := rowB4_ltN ops M K N TM TN hj hc.2)
    (fin_mem (by show portT4 M N K TM TN _ _ = _; rw [portT4_PB _ htj hq, hje]) hcom)

end Chain4

end Pouw.PearlC.WholeCircuit
