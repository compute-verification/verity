import Definitions.Pouw.PearlC.TileCheck8
import Proofs.Pouw.PearlC.WholeCircuit
import Proofs.Pouw.PearlC.RowCircuit

/-!
# `pc8`'s tile check: its units are its templates (proofs)

`WholeCircuit.whole8` is `Pc8Check`'s table, its units the instances of `Pc8RowB_v1`, `Pc8RowA_v1` and `Pc8Tile_v1`.
Here each unit an entry's words depend on is cut out as its template's transcript (`Whole.cut`), and the cuts are
chained along the ports, so the unit lemmas proved of each template's own table apply to the composed table unchanged:

* every port reads a gate of its input's width (`hfit8B`, `hfit8A`, `hfit8T`), so a cut holds on its inputs the
  transcript's values at its ports (`cutB8_input`, `cutA8_input`, `cutT8_input`) and, where its unit is correct, the
  transcript holds on the unit's committed gates the cut's values (`cutB8_at`, `cutA8_at`, `cutT8_at`);
* entry `(i, j)`'s tile unit is in the cone of its `C̃` (`cone8T`), and RowA `i` and RowB `j` in the cone of its `U`,
  whose last K16 step reads their `P_A` and `P_B` words (`cone8A`, `cone8B`);
* where the three units are correct, the entry's `C̃` and `U` hold the tile cut's (`chain8_C`, `chain8_U`), and the tile
  cut's `A′` and `B̃` rows and `P_A` and `P_B` rows are the row cuts' codes and words (`chain8_A`, `chain8_B`,
  `chain8_PA`, `chain8_PB`), each gate a tile reads being committed (`Whole.port_committed`).
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

variable (ops : RowOps)

/-! ## The row tables' gates -/

theorem runsAt_mid (R : Run) (L2 : List Run) : ∀ (L1 : List Run) (o g : ℕ), o + runsLen L1 ≤ g →
    g < o + runsLen L1 + R.n → runsAt (L1 ++ R :: L2) o g = some (R.gate (g - (o + runsLen L1)))
  | [], o, g, h1, h2 => by
    simp only [runsLen, add_zero] at h1 h2 ⊢
    simp only [List.nil_append, runsAt, h2, ↓reduceIte]
  | R' :: L1, o, g, h1, h2 => by
    simp only [runsLen] at h1 h2 ⊢
    simp only [List.cons_append, runsAt, show ¬ g < o + R'.n by omega, ↓reduceIte]
    rw [runsAt_mid R L2 L1 (o + R'.n) g (by omega) (by omega)]
    congr 3
    omega

theorem runsAt_isSome : ∀ (L : List Run) (o g : ℕ), o ≤ g → g < o + runsLen L → (runsAt L o g).isSome
  | [], _, _, _, h => by simp only [runsLen] at h; omega
  | R :: L, o, g, h1, h2 => by
    unfold runsAt
    split_ifs with h
    · rfl
    · exact runsAt_isSome L _ g (by omega) (by simp only [runsLen] at h2; omega)

theorem Rows.gate_run (T : Rows) {L1 L2 : List Run} {R : Run} (hT : T.runs = L1 ++ R :: L2) {g : ℕ}
    (h1 : T.nIn + runsLen L1 ≤ g) (h2 : g < T.nIn + runsLen L1 + R.n) :
    (T.spec ops).gate g = (R.gate (g - (T.nIn + runsLen L1))).toGate ops := by
  show T.gate ops g = _
  unfold Rows.gate
  rw [ite_eq_right (by omega), hT, runsAt_mid R L2 L1 T.nIn g h1 h2]

theorem Rows.gate_in (T : Rows) {g : ℕ} (hg : g < T.nIn) : (T.spec ops).gate g = ⟨T.inW g, true, [], fun _ => 0⟩ := by
  show T.gate ops g = _
  unfold Rows.gate
  rw [ite_eq_left hg]

/-- A row table's gates are inputs exactly below `nIn`. -/
theorem Rows.input_iff (T : Rows) : ∀ a < (T.spec ops).N, ((T.spec ops).gate a).input = true ↔ a < T.nIn := by
  intro a ha
  by_cases h : a < T.nIn
  · rw [Rows.gate_in ops T h]; simp [h]
  · obtain ⟨G, hG⟩ := Option.isSome_iff_exists.1 (runsAt_isSome T.runs T.nIn a (by omega) ha)
    show (T.gate ops a).input = true ↔ _
    unfold Rows.gate
    rw [ite_eq_right h, hG]
    simp [RGate.toGate, h]

theorem pre8Runs_code (p : Pipeline) (K b : ℕ) (rest : List Run) :
    pre8Runs p K b ++ rest = (statsRuns .fabs .id K b ++ [alpha8Run K b, sq8Run K b, th8Run K b, dead8Run K b,
      ok8Run K b, ep8Run K b, ew8Run K b]) ++ code8Run p K b :: rest := by
  simp [pre8Runs]

theorem code8At_len (K b : ℕ) :
    b + runsLen (statsRuns .fabs .id K b ++ [alpha8Run K b, sq8Run K b, th8Run K b, dead8Run K b, ok8Run K b,
      ep8Run K b, ew8Run K b]) = code8At K b := by
  have := statsRuns_len .fabs .id K b
  simp only [runsLen_append] at this ⊢
  simp [runsLen, alpha8Run, sq8Run, th8Run, dead8Run, ok8Run, ep8Run, ew8Run, Run.ofList, code8At, ew8At, ep8At,
    ok8At, dead8At, th8At, sq8At, al8At]
  omega

/-- A `pc8` row's code gates are E4M3 casts, 8 bits wide. -/
theorem rows8_code_w (T : Rows) (p : Pipeline) {K : ℕ} {rest : List Run} (hT : T.runs = pre8Runs p K T.nIn ++ rest)
    {t : ℕ} (ht : t < K) : ((T.spec ops).gate (code8 K T.nIn t)).w = 8 := by
  have hl := code8At_len K T.nIn
  rw [Rows.gate_run ops T (hT.trans (pre8Runs_code p K T.nIn rest)) (by unfold code8; omega)
    (by simp only [code8Run]; unfold code8; omega)]
  have : code8 K T.nIn t - (T.nIn + runsLen (statsRuns .fabs .id K T.nIn ++ [alpha8Run K T.nIn, sq8Run K T.nIn,
      th8Run K T.nIn, dead8Run K T.nIn, ok8Run K T.nIn, ep8Run K T.nIn, ew8Run K T.nIn])) = 3 * t + 2 := by
    unfold code8; omega
  rw [this]
  simp only [code8Run, RGate.toGate, show (3 * t + 2) % 3 = 2 by omega, show (2 : ℕ) ≠ 0 by omega,
    show (2 : ℕ) ≠ 1 by omega, ↓reduceIte, Prim.w]

theorem row8A_code_w (p : Pipeline) {K t : ℕ} (ht : t < K) :
    (((row8A p K).spec ops).gate (code8 K (rowIn K false) t)).w = 8 :=
  rows8_code_w ops (row8A p K) p rfl ht

theorem row8B_code_w (p : Pipeline) {K t : ℕ} (ht : t < K) :
    (((row8B p K).spec ops).gate (code8 K (rowIn K true) t)).w = 8 :=
  rows8_code_w ops (row8B p K) p rfl ht

/-- RowA's `P_A` words are BF16 words: `f2fpBf16` casts, 16 bits wide. -/
theorem row8A_pa_w (p : Pipeline) {K w : ℕ} (hw : w < 64) :
    (((row8A p K).spec ops).gate (paOut8 K w)).w = 16 := by
  have hl := pre8Runs_len p K (rowIn K false)
  unfold paOut8
  split_ifs with h
  · have hr : (row8A p K).runs = (pre8Runs p K (rowIn K false) ++ [af8Run p K (rowIn K false)]) ++
        bew8Run K (rowIn K false) :: [okwRun (ok8At K (rowIn K false) + 7)] := by simp [row8A]
    have hn : runsLen (pre8Runs p K (rowIn K false) ++ [af8Run p K (rowIn K false)]) =
        runsLen (pre8Runs p K (rowIn K false)) + 32 * (5 * (K / 128) + 1) := by
      simp [runsLen, af8Run, Run.rep]
    rw [Rows.gate_run ops (row8A p K) hr (by rw [hn]; show rowIn K false + _ ≤ _; omega)
      (by rw [hn]; show _ < rowIn K false + _ + _; simp only [bew8Run]; omega)]
    simp only [bew8Run, RGate.toGate, Prim.w]
  · have hr : (row8A p K).runs = pre8Runs p K (rowIn K false) ++ af8Run p K (rowIn K false) ::
        [bew8Run K (rowIn K false), okwRun (ok8At K (rowIn K false) + 7)] := by simp [row8A]
    have hq := mul_add_lt (L := 5 * (K / 128) + 1) (show w - 32 < 32 by omega) (show 5 * (K / 128) < _ by omega)
    rw [Rows.gate_run ops (row8A p K) hr (by show rowIn K false + _ ≤ _; omega)
      (by show _ < rowIn K false + _ + _; simp only [af8Run, Run.rep]; omega)]
    have e1 : post8At K (rowIn K false) + (w - 32) * (5 * (K / 128) + 1) + 5 * (K / 128) -
        ((row8A p K).nIn + runsLen (pre8Runs p K (rowIn K false))) =
          (w - 32) * (5 * (K / 128) + 1) + 5 * (K / 128) := by
      show _ - (rowIn K false + _) = _
      omega
    have e2 : ((w - 32) * (5 * (K / 128) + 1) + 5 * (K / 128)) / (5 * (K / 128) + 1) = w - 32 := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega), Nat.zero_add]
    have e3 : ((w - 32) * (5 * (K / 128) + 1) + 5 * (K / 128)) % (5 * (K / 128) + 1) = 5 * (K / 128) := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    rw [e1]
    simp only [af8Run, Run.rep, e2, e3, lt_irrefl, ↓reduceIte, RGate.toGate, Prim.w]

/-- RowB's `P_B` words are BF16 words: an `f2fpBf16` cast or a `bf16Neg`, 16 bits wide. -/
theorem row8B_pb_w (p : Pipeline) {K w : ℕ} (hw : w < 64) :
    (((row8B p K).spec ops).gate (pbOut8 K w)).w = 16 := by
  have hl := pre8Runs_len p K (rowIn K true)
  unfold pbOut8
  split_ifs with h
  · have hr : (row8B p K).runs = pre8Runs p K (rowIn K true) ++ mid8Run p K (rowIn K true) ::
        [nb8Run K (rowIn K true) (post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35)),
          okwRun (ok8At K (rowIn K true) + 7)] := by simp [row8B]
    have hq := mul_add_lt (L := 5 * (K / 128) + 35) h (show 5 * (K / 128) + 34 < _ by omega)
    rw [Rows.gate_run ops (row8B p K) hr (by show rowIn K true + _ ≤ _; omega)
      (by show _ < rowIn K true + _ + _; simp only [mid8Run, Run.rep]; omega)]
    have e1 : post8At K (rowIn K true) + w * (5 * (K / 128) + 35) + 5 * (K / 128) + 34 -
        ((row8B p K).nIn + runsLen (pre8Runs p K (rowIn K true))) =
          w * (5 * (K / 128) + 35) + (5 * (K / 128) + 34) := by
      show _ - (rowIn K true + _) = _
      omega
    have e2 : (w * (5 * (K / 128) + 35) + (5 * (K / 128) + 34)) / (5 * (K / 128) + 35) = w := by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ (by omega), Nat.div_eq_of_lt (by omega), Nat.zero_add]
    have e3 : (w * (5 * (K / 128) + 35) + (5 * (K / 128) + 34)) % (5 * (K / 128) + 35) = 5 * (K / 128) + 34 := by
      rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt (by omega)]
    rw [e1]
    simp only [mid8Run, Run.rep, e2, e3, mid8, show ¬ (5 * (K / 128) + 34 < 32 + 5 * (K / 128)) by omega,
      show ¬ (5 * (K / 128) + 34 < 32) by omega, show 5 * (K / 128) + 34 ≠ 32 + 5 * (K / 128) by omega,
      show 5 * (K / 128) + 34 ≠ 33 + 5 * (K / 128) by omega, ↓reduceIte, RGate.toGate, Prim.w]
  · have hr : (row8B p K).runs = (pre8Runs p K (rowIn K true) ++ [mid8Run p K (rowIn K true)]) ++
        nb8Run K (rowIn K true) (post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35)) ::
          [okwRun (ok8At K (rowIn K true) + 7)] := by simp [row8B]
    have hn : runsLen (pre8Runs p K (rowIn K true) ++ [mid8Run p K (rowIn K true)]) =
        runsLen (pre8Runs p K (rowIn K true)) + 32 * (5 * (K / 128) + 35) := by
      simp [runsLen, mid8Run, Run.rep]
    rw [Rows.gate_run ops (row8B p K) hr (by rw [hn]; show rowIn K true + _ ≤ _; omega)
      (by rw [hn]; show _ < rowIn K true + _ + _; simp only [nb8Run]; omega)]
    rw [hn, show post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 2 * (w - 32) + 1 -
      ((row8B p K).nIn + (runsLen (pre8Runs p K (rowIn K true)) + 32 * (5 * (K / 128) + 35))) = 2 * (w - 32) + 1 by
        show _ - (rowIn K true + _) = _; omega]
    simp only [nb8Run, show (2 * (w - 32) + 1) % 2 = 1 by omega, show (1 : ℕ) ≠ 0 by omega, ↓reduceIte,
      RGate.toGate, Prim.w]

/-! ## Composed tables -/

theorem fin_mem {n : ℕ} {s : Finset (Fin n)} {x y : ℕ} {hx : x < n} {hy : y < n} (h : x = y)
    (hm : (⟨x, hx⟩ : Fin n) ∈ s) : (⟨y, hy⟩ : Fin n) ∈ s := by
  subst h; exact hm

theorem reads_eq {C : Verity.Circuit} {g g' t t' : ℕ} {hg : g < C.N} {hg' : g' < C.N} {ht : t < C.N} {ht' : t' < C.N}
    (e1 : g = g') (e2 : t = t') (m : (⟨g, hg⟩ : Fin C.N) ∈ C.reads ⟨t, ht⟩) :
    (⟨g', hg'⟩ : Fin C.N) ∈ C.reads ⟨t', ht'⟩ := by
  subst e1; subst e2; exact m

theorem Whole.val_lt (W : Whole) (X : W.spec.toCircuit.Transcript) {g : ℕ} (hg : g < W.spec.toCircuit.N) :
    val X g < 2 ^ (W.gate g).w := by
  rw [val_of_lt X hg]
  exact (X ⟨g, hg⟩).isLt

theorem Whole.gate_in (W : Whole) {g : ℕ} (hg : g < W.nIn) : W.gate g = ⟨W.inW g, true, [], fun _ => 0⟩ := by
  unfold Whole.gate
  rw [ite_eq_left hg]

theorem Whole.inst_w (W : Whole) {L1 L2 : List BlockKind} {B : BlockKind} (hW : W.blocks = L1 ++ B :: L2) {q : ℕ}
    (hq : q < B.cnt) {a : ℕ} (ha1 : B.nIn ≤ a) (ha2 : a < B.S.N) :
    (W.gate (W.nIn + blocksLen L1 + q * B.len + (a - B.nIn))).w = (B.S.gate a).w := by
  rw [W.gate_inst hW hq ha1 ha2]
  rfl

/-! ## `whole8`'s blocks -/

section Whole8

variable (p : Pipeline) (M K N TM TN : ℕ)

/-- `whole8`'s RowB block. -/
def blockB8 : BlockKind := ⟨(row8B p K).spec ops, rowIn K true, N, portB8 M N K⟩

/-- `whole8`'s RowA block. -/
def blockA8 : BlockKind := ⟨(row8A p K).spec ops, rowIn K false, M, portA8 M N K⟩

/-- `whole8`'s tile block. -/
def blockT8 : BlockKind := ⟨tileSpec p K TM TN, tileIn K TM TN, M / TM * (N / TN), portT8 p M N K TM TN⟩

theorem whole8_B : (whole8 ops p M K N TM TN).blocks =
    [] ++ blockB8 ops p M K N :: [blockA8 ops p M K N, blockT8 p M K N TM TN] := rfl

theorem whole8_A : (whole8 ops p M K N TM TN).blocks =
    [blockB8 ops p M K N] ++ blockA8 ops p M K N :: [blockT8 p M K N TM TN] := rfl

theorem whole8_T : (whole8 ops p M K N TM TN).blocks =
    [blockB8 ops p M K N, blockA8 ops p M K N] ++ blockT8 p M K N TM TN :: [] := rfl

theorem rowB8_eq (j a : ℕ) : rowB8 p M N K j a = (whole8 ops p M K N TM TN).nIn + blocksLen [] +
    j * (blockB8 ops p M K N).len + (a - (blockB8 ops p M K N).nIn) := by
  simp only [rowB8, whole8, blocksLen, BlockKind.len, blockB8, Rows.spec, add_zero]

theorem rowA8_eq (i a : ℕ) : rowA8 p M N K i a = (whole8 ops p M K N TM TN).nIn + blocksLen [blockB8 ops p M K N] +
    i * (blockA8 ops p M K N).len + (a - (blockA8 ops p M K N).nIn) := by
  simp only [rowA8, whole8, blocksLen, BlockKind.len, blockB8, blockA8, Rows.spec, add_zero]

theorem tile8_eq (e a : ℕ) : tile8 p M N K TM TN e a = (whole8 ops p M K N TM TN).nIn +
    blocksLen [blockB8 ops p M K N, blockA8 ops p M K N] + e * (blockT8 p M K N TM TN).len +
      (a - (blockT8 p M K N TM TN).nIn) := by
  simp only [tile8, blocksLen, BlockKind.len, blockB8, blockA8, blockT8, whole8, add_zero, Nat.add_assoc]
  rfl

theorem whole8_N : (whole8 ops p M K N TM TN).spec.toCircuit.N = in8 M N K + N * ((row8B p K).N - rowIn K true) +
    M * ((row8A p K).N - rowIn K false) + M / TM * (N / TN) * (tileN K TM TN - tileIn K TM TN) := by
  show in8 M N K + blocksLen _ = _
  simp only [blocksLen, BlockKind.len, whole8, add_zero, Nat.add_assoc]
  rfl

/-! ### Inputs -/

theorem inW8_portA8 {i a : ℕ} (hi : i < M) (ha : a < rowIn K false) :
    inW8 M N K (portA8 M N K i a) = (row8A p K).inW a := by
  have e1 : (M + N) * (K + 32) = M * K + N * K + 32 * M + 32 * N := by ring
  have e2 : M * (K + 32) = M * K + 32 * M := by ring
  have hiK : i * K + K ≤ M * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hi
  simp only [rowIn, Bool.false_eq_true, ↓reduceIte, add_zero] at ha
  simp only [row8A]
  unfold inW8 portA8 eaAt bAt ebAt faAt fbAt gAt
  rw [e1, e2]
  split_ifs <;> omega

theorem inW8_portB8 {j a : ℕ} (hj : j < N) (ha : a < rowIn K true) :
    inW8 M N K (portB8 M N K j a) = (row8B p K).inW a := by
  have e1 : (M + N) * (K + 32) = M * K + N * K + 32 * M + 32 * N := by ring
  have e2 : M * (K + 32) = M * K + 32 * M := by ring
  have hjK : j * K + K ≤ N * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hj
  simp only [rowIn, ↓reduceIte] at ha
  simp only [row8B]
  unfold inW8 portB8 eaAt bAt ebAt faAt fbAt gAt
  rw [e1, e2]
  split_ifs <;> omega

theorem hinB8 : ∀ a < (blockB8 ops p M K N).S.N,
    ((blockB8 ops p M K N).S.gate a).input = true ↔ a < (blockB8 ops p M K N).nIn :=
  Rows.input_iff ops (row8B p K)

theorem hinA8 : ∀ a < (blockA8 ops p M K N).S.N,
    ((blockA8 ops p M K N).S.gate a).input = true ↔ a < (blockA8 ops p M K N).nIn :=
  Rows.input_iff ops (row8A p K)

theorem tile_input_iff : ∀ a < (tileSpec p K TM TN).N,
    ((tileSpec p K TM TN).gate a).input = true ↔ a < tileIn K TM TN := by
  intro a _
  refine ⟨fun h => ?_, gate_input p K TM TN⟩
  by_contra hc
  have hk : (TM + TN) * K ≤ tileIn K TM TN := by unfold tileIn; omega
  have : (tileGate p K TM TN a).input = false := by
    unfold tileGate
    rw [ite_eq_right (by omega), ite_eq_right hc]
    simp only [entryGate]
    split_ifs <;> rfl
  exact absurd h (by rw [show (tileSpec p K TM TN).gate a = tileGate p K TM TN a from rfl, this]; simp)

theorem hinT8 : ∀ a < (blockT8 p M K N TM TN).S.N,
    ((blockT8 p M K N TM TN).S.gate a).input = true ↔ a < (blockT8 p M K N TM TN).nIn :=
  tile_input_iff p K TM TN

theorem hnB8 : (blockB8 ops p M K N).nIn ≤ (blockB8 ops p M K N).S.N := by
  show rowIn K true ≤ (row8B p K).N
  have := post8At_ge K (rowIn K true)
  rw [row8B_N]; omega

theorem hnA8 : (blockA8 ops p M K N).nIn ≤ (blockA8 ops p M K N).S.N := by
  show rowIn K false ≤ (row8A p K).N
  have := post8At_ge K (rowIn K false)
  rw [row8A_N]; omega

theorem hnT8 : (blockT8 p M K N TM TN).nIn ≤ (blockT8 p M K N TM TN).S.N := by
  show tileIn K TM TN ≤ tileN K TM TN
  unfold tileN; omega

/-! ### Every port reads a gate of its input's width -/

theorem hfit8B (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {j : ℕ} (hj : j < N) :
    ∀ a < (blockB8 ops p M K N).nIn,
      val X ((blockB8 ops p M K N).port j a) < 2 ^ ((blockB8 ops p M K N).S.gate a).w := by
  intro a ha
  have hlt : portB8 M N K j a < (whole8 ops p M K N TM TN).nIn := portB8_lt hj ha
  have h := (whole8 ops p M K N TM TN).val_lt X (show _ < (whole8 ops p M K N TM TN).spec.toCircuit.N from
    lt_of_lt_of_le hlt (Nat.le_add_right _ _))
  rw [(whole8 ops p M K N TM TN).gate_in hlt] at h
  show val X (portB8 M N K j a) < 2 ^ (((row8B p K).spec ops).gate a).w
  rw [Rows.gate_in ops (row8B p K) ha, ← inW8_portB8 p M K N hj ha]
  exact h

theorem hfit8A (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {i : ℕ} (hi : i < M) :
    ∀ a < (blockA8 ops p M K N).nIn,
      val X ((blockA8 ops p M K N).port i a) < 2 ^ ((blockA8 ops p M K N).S.gate a).w := by
  intro a ha
  have hlt : portA8 M N K i a < (whole8 ops p M K N TM TN).nIn := portA8_lt hi ha
  have h := (whole8 ops p M K N TM TN).val_lt X (show _ < (whole8 ops p M K N TM TN).spec.toCircuit.N from
    lt_of_lt_of_le hlt (Nat.le_add_right _ _))
  rw [(whole8 ops p M K N TM TN).gate_in hlt] at h
  show val X (portA8 M N K i a) < 2 ^ (((row8A p K).spec ops).gate a).w
  rw [Rows.gate_in ops (row8A p K) ha, ← inW8_portA8 p M K N hi ha]
  exact h

theorem tileGate_w8 {a : ℕ} (h : a < (TM + TN) * K) : (tileGate p K TM TN a).w = 8 := by
  unfold tileGate; rw [ite_eq_left h]

theorem tileGate_w16 {a : ℕ} (h1 : (TM + TN) * K ≤ a) (h2 : a < tileIn K TM TN) : (tileGate p K TM TN a).w = 16 := by
  unfold tileGate; rw [ite_eq_right (by omega), ite_eq_left h2]

theorem hfit8T (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {e : ℕ} (he : e < M / TM * (N / TN)) :
    ∀ a < (blockT8 p M K N TM TN).nIn,
      val X ((blockT8 p M K N TM TN).port e a) < 2 ^ ((blockT8 p M K N TM TN).S.gate a).w := by
  intro a ha
  have hN : portT8 p M N K TM TN e a < (whole8 ops p M K N TM TN).spec.toCircuit.N := by
    rw [whole8_N]; have := portT8_lt p he ha; omega
  refine lt_of_lt_of_eq ((whole8 ops p M K N TM TN).val_lt X hN) ?_
  congr 1
  have hNA : post8At K (rowIn K false) < (row8A p K).N := by rw [row8A_N]; omega
  have hNB : post8At K (rowIn K true) < (row8B p K).N := by rw [row8B_N]; omega
  change a < tileIn K TM TN at ha
  show ((whole8 ops p M K N TM TN).gate (portT8 p M N K TM TN e a)).w = (tileGate p K TM TN a).w
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  unfold portT8
  unfold tileIn at ha
  split_ifs with h1 h2 h3
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by simp [h] at h1
    have hq := tile_row_lt he ((Nat.div_lt_iff_lt_mul hK).2 h1)
    have hb := code8_bounds (b := rowIn K false) (Nat.mod_lt a hK)
    rw [rowA8_eq ops p M K N TM TN, Whole.inst_w _ (whole8_A ops p M K N TM TN) hq hb.1 (lt_trans hb.2 hNA),
      tileGate_w8 p K TM TN (by omega)]
    exact row8A_code_w ops p (Nat.mod_lt a hK)
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by simp [h] at h2
    have hq := tile_col_lt he ((Nat.div_lt_iff_lt_mul hK).2 (show a - TM * K < TN * K by omega))
    have hb := code8_bounds (b := rowIn K true) (Nat.mod_lt (a - TM * K) hK)
    rw [rowB8_eq ops p M K N TM TN, Whole.inst_w _ (whole8_B ops p M K N TM TN) hq hb.1 (lt_trans hb.2 hNB),
      tileGate_w8 p K TM TN h2]
    exact row8B_code_w ops p (Nat.mod_lt _ hK)
  · have hq := tile_row_lt he ((Nat.div_lt_iff_lt_mul (by norm_num)).2 (show a - (TM + TN) * K < TM * 64 by omega))
    have hb := paOut8_bounds p (K := K) (Nat.mod_lt (a - (TM + TN) * K) (show 0 < 64 by norm_num))
    rw [rowA8_eq ops p M K N TM TN, Whole.inst_w _ (whole8_A ops p M K N TM TN) hq hb.1 hb.2,
      tileGate_w16 p K TM TN (by omega) (by unfold tileIn; omega)]
    exact row8A_pa_w ops p (Nat.mod_lt _ (by norm_num))
  · have hq := tile_col_lt he ((Nat.div_lt_iff_lt_mul (by norm_num)).2
      (show a - ((TM + TN) * K + 64 * TM) < TN * 64 by omega))
    have hb := pbOut8_bounds p (K := K) (Nat.mod_lt (a - ((TM + TN) * K + 64 * TM)) (show 0 < 64 by norm_num))
    rw [rowB8_eq ops p M K N TM TN, Whole.inst_w _ (whole8_B ops p M K N TM TN) hq hb.1 hb.2,
      tileGate_w16 p K TM TN (by omega) (by unfold tileIn; omega)]
    exact row8B_pb_w ops p (Nat.mod_lt _ (by norm_num))

end Whole8

/-! ## What a tile's entry reads -/

section Reads8

variable (p : Pipeline) (K TM TN : ℕ)

/-- Entry `(i, j)`'s atom `5·(l / 128) + (l mod 128) / 32` reads code `l` of `A′_i` and of `B̃_j`. -/
theorem tile_reads_code {i j l : ℕ} (hj : j < TN) (hl : l < 128 * (K / 128)) :
    i * K + l ∈ (tileGate p K TM TN (entryBase K TM TN i j + (5 * (l / 128) + l % 128 / 32))).rd ∧
      TM * K + j * K + l ∈ (tileGate p K TM TN (entryBase K TM TN i j + (5 * (l / 128) + l % 128 / 32))).rd := by
  have hq : l / 128 < K / 128 := by omega
  have hr : l % 128 / 32 < 4 := by omega
  rw [gate_entry p K TM TN hj (by unfold blockLen; omega), entry_atom p K _ _ _ _ _ hq hr]
  exact ⟨List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨l % 32, List.mem_range.2 (by omega), by omega⟩)),
    List.mem_append_right _ (List.mem_map.2 ⟨l % 32, List.mem_range.2 (by omega), by omega⟩)⟩

/-- Entry `(i, j)`'s K16 step `q / 16` reads word `q` of `P_A,i` and of `P_B,j`. -/
theorem tile_reads_P {i j q : ℕ} (hj : j < TN) (hq : q < 64) :
    (TM + TN) * K + 64 * i + q ∈ (tileGate p K TM TN (entryBase K TM TN i j + (5 * (K / 128) + q / 16))).rd ∧
      (TM + TN) * K + 64 * TM + 64 * j + q ∈
        (tileGate p K TM TN (entryBase K TM TN i j + (5 * (K / 128) + q / 16))).rd := by
  rw [gate_entry p K TM TN hj (by unfold blockLen; omega), entry_peel p K _ _ _ _ _ (q / 16)]
  exact ⟨List.mem_append_left _ (List.mem_append_right _
      (List.mem_map.2 ⟨q % 16, List.mem_range.2 (by omega), by omega⟩)),
    List.mem_append_right _ (List.mem_map.2 ⟨q % 16, List.mem_range.2 (by omega), by omega⟩)⟩

theorem uGate_eq (i j : ℕ) : uGate K TM TN i j = entryBase K TM TN i j + (5 * (K / 128) + 48 / 16) := by
  unfold uGate blockLen; omega

theorem entry_bounds {i j r : ℕ} (hi : i < TM) (hj : j < TN) (hr : r < blockLen K) :
    tileIn K TM TN ≤ entryBase K TM TN i j + r ∧ entryBase K TM TN i j + r < tileN K TM TN := by
  have := entry_lt K TM TN hi hj
  unfold entryBase at this ⊢
  omega

end Reads8

/-- Entry `(i, j)` lies in tile `i / TM · (N / TN) + j / TN` at `(i mod TM, j mod TN)`. -/
theorem tile_index {M N TM TN i j : ℕ} (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N) :
    i / TM * (N / TN) + j / TN < M / TM * (N / TN) ∧
      (i / TM * (N / TN) + j / TN) / (N / TN) * TM + i % TM = i ∧
      (i / TM * (N / TN) + j / TN) % (N / TN) * TN + j % TN = j := by
  have hiq : i / TM < M / TM := Nat.div_lt_div_of_lt_of_dvd hTM hi
  have hjq : j / TN < N / TN := Nat.div_lt_div_of_lt_of_dvd hTN hj
  have hn : 0 < N / TN := lt_of_le_of_lt (Nat.zero_le _) hjq
  have h1 : (i / TM * (N / TN) + j / TN) / (N / TN) = i / TM := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hn, Nat.div_eq_of_lt hjq, Nat.zero_add]
  have h2 : (i / TM * (N / TN) + j / TN) % (N / TN) = j / TN := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hjq]
  refine ⟨mul_add_lt hiq hjq, ?_, ?_⟩
  · rw [h1]; exact Nat.div_add_mod' i TM
  · rw [h2]; exact Nat.div_add_mod' j TN

section Ports8

variable (p : Pipeline) {M N K TM TN : ℕ} (e : ℕ)

theorem portT8_A {ti l : ℕ} (hti : ti < TM) (hl : l < K) :
    portT8 p M N K TM TN e (ti * K + l) = rowA8 p M N K (e / (N / TN) * TM + ti) (code8 K (rowIn K false) l) := by
  have hK : 0 < K := by omega
  unfold portT8
  rw [ite_eq_left (mul_add_lt hti hl), show (ti * K + l) / K = ti by
      rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (ti * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem portT8_B {tj l : ℕ} (htj : tj < TN) (hl : l < K) :
    portT8 p M N K TM TN e (TM * K + tj * K + l) =
      rowB8 p M N K (e % (N / TN) * TN + tj) (code8 K (rowIn K true) l) := by
  have hK : 0 < K := by omega
  have h1 := mul_add_lt htj hl
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  unfold portT8
  rw [ite_eq_right (by omega), ite_eq_left (by omega), show TM * K + tj * K + l - TM * K = tj * K + l by omega,
    show (tj * K + l) / K = tj by rw [Nat.add_comm, Nat.add_mul_div_right _ _ hK, Nat.div_eq_of_lt hl, Nat.zero_add],
    show (tj * K + l) % K = l by rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hl]]

theorem portT8_PA {ti q : ℕ} (hti : ti < TM) (hq : q < 64) :
    portT8 p M N K TM TN e ((TM + TN) * K + 64 * ti + q) =
      rowA8 p M N K (e / (N / TN) * TM + ti) (paOut8 K q) := by
  have h1 := mul_add_lt (L := 64) hti hq
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  unfold portT8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_left (by omega),
    show (TM + TN) * K + 64 * ti + q - (TM + TN) * K = ti * 64 + q by omega,
    show (ti * 64 + q) / 64 = ti by omega, show (ti * 64 + q) % 64 = q by omega]

theorem portT8_PB {tj q : ℕ} (_htj : tj < TN) (hq : q < 64) :
    portT8 p M N K TM TN e ((TM + TN) * K + 64 * TM + 64 * tj + q) =
      rowB8 p M N K (e % (N / TN) * TN + tj) (pbOut8 K q) := by
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  unfold portT8
  rw [ite_eq_right (by omega), ite_eq_right (by omega), ite_eq_right (by omega),
    show (TM + TN) * K + 64 * TM + 64 * tj + q - ((TM + TN) * K + 64 * TM) = tj * 64 + q by omega,
    show (tj * 64 + q) / 64 = tj by omega, show (tj * 64 + q) % 64 = q by omega]

end Ports8

/-! ## The units' cuts -/

section Cuts8

variable (p : Pipeline) (M K N TM TN : ℕ)

/-- RowB `j`'s cut holds on its inputs the transcript's values at its ports. -/
theorem cutB8_input (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {j : ℕ} (hj : j < N) {a : ℕ}
    (ha : a < rowIn K true) :
    val ((whole8 ops p M K N TM TN).cut X (blockB8 ops p M K N) j) a = val X (portB8 M N K j a) :=
  (whole8 ops p M K N TM TN).cut_input (whole8_B ops p M K N TM TN) hj X (hinB8 ops p M K N)
    (hfit8B ops p M K N TM TN X hj) (hnB8 ops p M K N) ha

/-- RowA `i`'s cut holds on its inputs the transcript's values at its ports. -/
theorem cutA8_input (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {i : ℕ} (hi : i < M) {a : ℕ}
    (ha : a < rowIn K false) :
    val ((whole8 ops p M K N TM TN).cut X (blockA8 ops p M K N) i) a = val X (portA8 M N K i a) :=
  (whole8 ops p M K N TM TN).cut_input (whole8_A ops p M K N TM TN) hi X (hinA8 ops p M K N)
    (hfit8A ops p M K N TM TN X hi) (hnA8 ops p M K N) ha

/-- Tile `e`'s cut holds on its inputs the transcript's values at its ports. -/
theorem cutT8_input (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {e : ℕ} (he : e < M / TM * (N / TN))
    {a : ℕ} (ha : a < tileIn K TM TN) :
    val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN) e) a = val X (portT8 p M N K TM TN e a) :=
  (whole8 ops p M K N TM TN).cut_input (whole8_T ops p M K N TM TN) he X (hinT8 p M K N TM TN)
    (hfit8T ops p M K N TM TN X he) (hnT8 p M K N TM TN) ha

/-- Where RowB `j` is correct, its committed gates hold its cut's values. -/
theorem cutB8_at (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {j : ℕ} (hj : j < N)
    (hX : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_B ops p M K N TM TN) hj⟩)
    {a : ℕ} (ha1 : rowIn K true ≤ a) (ha2 : a < (row8B p K).N) {h : rowB8 p M N K j a < _}
    (hcom : (⟨rowB8 p M N K j a, h⟩ : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N) ∈
      (whole8 ops p M K N TM TN).part.committed) :
    val X (rowB8 p M N K j a) = val ((whole8 ops p M K N TM TN).cut X (blockB8 ops p M K N) j) a := by
  rw [rowB8_eq ops p M K N TM TN]
  exact (whole8 ops p M K N TM TN).cut_committed (whole8_B ops p M K N TM TN) hj X (hinB8 ops p M K N)
    (hfit8B ops p M K N TM TN X hj) hX ha1 ha2 (fin_mem (rowB8_eq ops p M K N TM TN j a) hcom)

/-- Where RowA `i` is correct, its committed gates hold its cut's values. -/
theorem cutA8_at (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {i : ℕ} (hi : i < M)
    (hX : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_A ops p M K N TM TN) hi⟩)
    {a : ℕ} (ha1 : rowIn K false ≤ a) (ha2 : a < (row8A p K).N) {h : rowA8 p M N K i a < _}
    (hcom : (⟨rowA8 p M N K i a, h⟩ : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N) ∈
      (whole8 ops p M K N TM TN).part.committed) :
    val X (rowA8 p M N K i a) = val ((whole8 ops p M K N TM TN).cut X (blockA8 ops p M K N) i) a := by
  rw [rowA8_eq ops p M K N TM TN]
  exact (whole8 ops p M K N TM TN).cut_committed (whole8_A ops p M K N TM TN) hi X (hinA8 ops p M K N)
    (hfit8A ops p M K N TM TN X hi) hX ha1 ha2 (fin_mem (rowA8_eq ops p M K N TM TN i a) hcom)

/-- Where tile `e` is correct, its committed gates hold its cut's values. -/
theorem cutT8_at (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {e : ℕ} (he : e < M / TM * (N / TN))
    (hX : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_T ops p M K N TM TN) he⟩)
    {a : ℕ} (ha1 : tileIn K TM TN ≤ a) (ha2 : a < tileN K TM TN) {h : tile8 p M N K TM TN e a < _}
    (hcom : (⟨tile8 p M N K TM TN e a, h⟩ : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N) ∈
      (whole8 ops p M K N TM TN).part.committed) :
    val X (tile8 p M N K TM TN e a) = val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN) e) a := by
  rw [tile8_eq ops p M K N TM TN]
  exact (whole8 ops p M K N TM TN).cut_committed (whole8_T ops p M K N TM TN) he X (hinT8 p M K N TM TN)
    (hfit8T ops p M K N TM TN X he) hX ha1 ha2 (fin_mem (tile8_eq ops p M K N TM TN e a) hcom)

end Cuts8

/-! ## An entry's cone -/

section Cone8

variable (p : Pipeline) (M K N TM TN : ℕ)

theorem pos_of_dvd_lt {a b i : ℕ} (h : a ∣ b) (hi : i < b) : 0 < a := Nat.pos_of_dvd_of_pos h (by omega)

theorem rowA8_ltN {i a : ℕ} (hi : i < M) (ha : a < (row8A p K).N) :
    rowA8 p M N K i a < (whole8 ops p M K N TM TN).spec.toCircuit.N := by
  rw [whole8_N]; have := rowA8_lt p (N := N) hi ha; omega

theorem rowB8_ltN {j a : ℕ} (hj : j < N) (ha : a < (row8B p K).N) :
    rowB8 p M N K j a < (whole8 ops p M K N TM TN).spec.toCircuit.N := by
  rw [whole8_N]; have := rowB8_lt p (M := M) hj ha; omega

theorem tile8_ltN {e a : ℕ} (he : e < M / TM * (N / TN)) (ha1 : tileIn K TM TN ≤ a) (ha2 : a < tileN K TM TN) :
    tile8 p M N K TM TN e a < (whole8 ops p M K N TM TN).spec.toCircuit.N := by
  rw [tile8_eq ops p M K N TM TN]
  exact (whole8 ops p M K N TM TN).inst_lt (whole8_T ops p M K N TM TN) he ha1 ha2

/-- Entry `(i, j)`'s `U` is its tile's copy of the block's gate `5·(K / 128) + 3`. -/
theorem uGlob8_eq (i j : ℕ) : uGlob8 p M N K TM TN i j = (whole8 ops p M K N TM TN).nIn +
    blocksLen [blockB8 ops p M K N, blockA8 ops p M K N] + (i / TM * (N / TN) + j / TN) * (blockT8 p M K N TM TN).len +
      (entryBase K TM TN (i % TM) (j % TN) + (5 * (K / 128) + 48 / 16) - (blockT8 p M K N TM TN).nIn) := by
  unfold uGlob8; rw [uGate_eq]; exact tile8_eq ops p M K N TM TN _ _

/-- Entry `(i, j)`'s tile unit is in the cone of its `C̃`. -/
theorem cone8T {T : Finset (Fin (whole8 ops p M K N TM TN).spec.toCircuit.N)} {i j : ℕ} (hTM : TM ∣ M)
    (hTN : TN ∣ N) (hi : i < M) (hj : j < N) (hK : 0 < K / 128)
    (hc : ∀ g : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N, g.val = cGlob8 p M N K TM TN i j → g ∈ T) :
    (⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_T ops p M K N TM TN) (tile_index hTM hTN hi hj).1⟩ :
      Fin (unitCount (whole8 ops p M K N TM TN).blocks)) ∈ (whole8 ops p M K N TM TN).part.cone T := by
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) - 1) (by unfold blockLen; omega)
  have hcg : cGate K TM TN (i % TM) (j % TN) = entryBase K TM TN (i % TM) (j % TN) + (5 * (K / 128) - 1) := by
    unfold cGate; omega
  rw [← hcg] at hb
  exact (whole8 ops p M K N TM TN).part.mem_cone (hc _ (tile8_eq ops p M K N TM TN _ _).symm) (.refl _)
    ((whole8 ops p M K N TM TN).unit_inst (whole8_T ops p M K N TM TN) (tile_index hTM hTN hi hj).1 hb.1 hb.2)

/-- Entry `(i, j)`'s RowA unit is in the cone of its `U`, whose last K16 step reads `P_A,i`'s word 48. -/
theorem cone8A {T : Finset (Fin (whole8 ops p M K N TM TN).spec.toCircuit.N)} {i j : ℕ} (hTM : TM ∣ M)
    (hTN : TN ∣ N) (hi : i < M) (hj : j < N)
    (hu : ∀ g : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N, g.val = uGlob8 p M N K TM TN i j → g ∈ T) :
    (⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_A ops p M K N TM TN) hi⟩ :
      Fin (unitCount (whole8 ops p M K N TM TN).blocks)) ∈ (whole8 ops p M K N TM TN).part.cone T := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) + 48 / 16) (by unfold blockLen; omega)
  have hpa := tile_reads_P p K TM TN (i := i % TM) htj (q := 48) (by norm_num)
  have hpb := paOut8_bounds p (K := K) (show 48 < 64 by norm_num)
  have hm := (whole8 ops p M K N TM TN).port_reads (whole8_T ops p M K N TM TN) he hb.1 hb.2
    (show (TM + TN) * K + 64 * (i % TM) + 48 < tileIn K TM TN by unfold tileIn; omega) hpa.1
  refine (whole8 ops p M K N TM TN).part.mem_cone (hu _ (uGlob8_eq ops p M K N TM TN i j).symm)
    (.step (reads_eq (ht' := (whole8 ops p M K N TM TN).inst_lt (whole8_T ops p M K N TM TN) he hb.1 hb.2) ?_ rfl hm)
      (.refl _))
    ((whole8 ops p M K N TM TN).unit_inst (whole8_A ops p M K N TM TN) hi hpb.1 hpb.2)
  show portT8 p M N K TM TN _ _ = _
  rw [portT8_PA p _ hti (by norm_num), hie, rowA8_eq ops p M K N TM TN]

/-- Entry `(i, j)`'s RowB unit is in the cone of its `U`, whose last K16 step reads `P_B,j`'s word 48. -/
theorem cone8B {T : Finset (Fin (whole8 ops p M K N TM TN).spec.toCircuit.N)} {i j : ℕ} (hTM : TM ∣ M)
    (hTN : TN ∣ N) (hi : i < M) (hj : j < N)
    (hu : ∀ g : Fin (whole8 ops p M K N TM TN).spec.toCircuit.N, g.val = uGlob8 p M N K TM TN i j → g ∈ T) :
    (⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_B ops p M K N TM TN) hj⟩ :
      Fin (unitCount (whole8 ops p M K N TM TN).blocks)) ∈ (whole8 ops p M K N TM TN).part.cone T := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) + 48 / 16) (by unfold blockLen; omega)
  have hpa := tile_reads_P p K TM TN (i := i % TM) htj (q := 48) (by norm_num)
  have hpb := pbOut8_bounds p (K := K) (show 48 < 64 by norm_num)
  have hm := (whole8 ops p M K N TM TN).port_reads (whole8_T ops p M K N TM TN) he hb.1 hb.2
    (show (TM + TN) * K + 64 * TM + 64 * (j % TN) + 48 < tileIn K TM TN by unfold tileIn; omega) hpa.2
  refine (whole8 ops p M K N TM TN).part.mem_cone (hu _ (uGlob8_eq ops p M K N TM TN i j).symm)
    (.step (reads_eq (ht' := (whole8 ops p M K N TM TN).inst_lt (whole8_T ops p M K N TM TN) he hb.1 hb.2) ?_ rfl hm)
      (.refl _))
    ((whole8 ops p M K N TM TN).unit_inst (whole8_B ops p M K N TM TN) hj hpb.1 hpb.2)
  show portT8 p M N K TM TN _ _ = _
  rw [portT8_PB p _ htj (by norm_num), hje, rowB8_eq ops p M K N TM TN]

theorem outs8_C {i j : ℕ} (hi : i < M) (hj : j < N) : cGlob8 p M N K TM TN i j ∈ outs8 p M N K TM TN := by
  unfold outs8
  exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_left _
    (Finset.mem_image.2 ⟨(i, j), Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hj⟩, rfl⟩)))

theorem outs8_U {i j : ℕ} (hi : i < M) (hj : j < N) : uGlob8 p M N K TM TN i j ∈ outs8 p M N K TM TN := by
  unfold outs8
  exact Finset.mem_union_left _ (Finset.mem_union_left _ (Finset.mem_union_right _
    (Finset.mem_image.2 ⟨(i, j), Finset.mem_product.2 ⟨Finset.mem_range.2 hi, Finset.mem_range.2 hj⟩, rfl⟩)))

end Cone8

/-! ## A tile's cut reads its rows' cuts -/

section Chain8

variable (p : Pipeline) (M K N TM TN : ℕ) (X : (whole8 ops p M K N TM TN).spec.toCircuit.Transcript) {i j : ℕ}
  (hTM : TM ∣ M) (hTN : TN ∣ N) (hi : i < M) (hj : j < N)

include hTM hTN hi hj

/-- Where its tile unit is correct, entry `(i, j)`'s `C̃` holds the tile cut's. -/
theorem chain8_C (hK : 0 < K / 128)
    (hXT : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_T ops p M K N TM TN) (tile_index hTM hTN hi hj).1⟩) :
    val X (cGlob8 p M N K TM TN i j) = val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN)
      (i / TM * (N / TN) + j / TN)) (cGate K TM TN (i % TM) (j % TN)) := by
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) - 1) (by unfold blockLen; omega)
  have hcg : cGate K TM TN (i % TM) (j % TN) = entryBase K TM TN (i % TM) (j % TN) + (5 * (K / 128) - 1) := by
    unfold cGate; omega
  rw [← hcg] at hb
  exact cutT8_at ops p M K N TM TN X (tile_index hTM hTN hi hj).1 hXT hb.1 hb.2
    (h := tile8_ltN ops p M K N TM TN (tile_index hTM hTN hi hj).1 hb.1 hb.2)
    ((whole8 ops p M K N TM TN).part.mem_committed_of_output (by
      simp only [Spec.toCircuit, Finset.mem_filter, Finset.mem_univ, true_and]
      exact outs8_C p M K N TM TN hi hj))

/-- Where its tile unit is correct, entry `(i, j)`'s `U` holds the tile cut's. -/
theorem chain8_U
    (hXT : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_T ops p M K N TM TN) (tile_index hTM hTN hi hj).1⟩) :
    val X (uGlob8 p M N K TM TN i j) = val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN)
      (i / TM * (N / TN) + j / TN)) (uGate K TM TN (i % TM) (j % TN)) := by
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) + 48 / 16) (by unfold blockLen; omega)
  rw [← uGate_eq] at hb
  exact cutT8_at ops p M K N TM TN X (tile_index hTM hTN hi hj).1 hXT hb.1 hb.2
    (h := tile8_ltN ops p M K N TM TN (tile_index hTM hTN hi hj).1 hb.1 hb.2)
    ((whole8 ops p M K N TM TN).part.mem_committed_of_output (by
      simp only [Spec.toCircuit, Finset.mem_filter, Finset.mem_univ, true_and]
      exact outs8_U p M K N TM TN hi hj))

/-- Where RowA `i` is correct, the tile cut's row `i mod TM` of `A′` is RowA `i`'s cut's codes. -/
theorem chain8_A (hk : 128 ∣ K)
    (hXA : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_A ops p M K N TM TN) hi⟩) {l : ℕ} (hl : l < K) :
    val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN) (i / TM * (N / TN) + j / TN)) (i % TM * K + l) =
      val ((whole8 ops p M K N TM TN).cut X (blockA8 ops p M K N) i) (code8 K (rowIn K false) l) := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hl' : l < 128 * (K / 128) := by omega
  have hlt := mul_add_lt hti hl
  have ha : i % TM * K + l < tileIn K TM TN := by unfold tileIn; omega
  have hb := entry_bounds K TM TN hti htj (r := 5 * (l / 128) + l % 128 / 32) (by unfold blockLen; omega)
  have hc := code8_bounds (b := rowIn K false) hl
  have hNA : post8At K (rowIn K false) < (row8A p K).N := by rw [row8A_N]; omega
  have hcom := (whole8 ops p M K N TM TN).port_committed (whole8_T ops p M K N TM TN) he hb.1 hb.2 ha
    (tile_reads_code p K TM TN htj hl').1
  rw [cutT8_input ops p M K N TM TN X he ha]
  show val X (portT8 p M N K TM TN _ _) = _
  rw [portT8_A p _ hti hl, hie]
  exact cutA8_at ops p M K N TM TN X hi hXA hc.1 (lt_trans hc.2 hNA)
    (h := rowA8_ltN ops p M K N TM TN hi (lt_trans hc.2 hNA))
    (fin_mem (by show portT8 p M N K TM TN _ _ = _; rw [portT8_A p _ hti hl, hie]) hcom)

/-- Where RowB `j` is correct, the tile cut's row `j mod TN` of `B̃` is RowB `j`'s cut's codes. -/
theorem chain8_B (hk : 128 ∣ K)
    (hXB : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_B ops p M K N TM TN) hj⟩) {l : ℕ} (hl : l < K) :
    val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN) (i / TM * (N / TN) + j / TN))
        (TM * K + j % TN * K + l) =
      val ((whole8 ops p M K N TM TN).cut X (blockB8 ops p M K N) j) (code8 K (rowIn K true) l) := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  have hl' : l < 128 * (K / 128) := by omega
  have hlt := mul_add_lt htj hl
  have ha : TM * K + j % TN * K + l < tileIn K TM TN := by unfold tileIn; omega
  have hb := entry_bounds K TM TN hti htj (r := 5 * (l / 128) + l % 128 / 32) (by unfold blockLen; omega)
  have hc := code8_bounds (b := rowIn K true) hl
  have hNB : post8At K (rowIn K true) < (row8B p K).N := by rw [row8B_N]; omega
  have hcom := (whole8 ops p M K N TM TN).port_committed (whole8_T ops p M K N TM TN) he hb.1 hb.2 ha
    (tile_reads_code p K TM TN (i := i % TM) htj hl').2
  rw [cutT8_input ops p M K N TM TN X he ha]
  show val X (portT8 p M N K TM TN _ _) = _
  rw [portT8_B p _ htj hl, hje]
  exact cutB8_at ops p M K N TM TN X hj hXB hc.1 (lt_trans hc.2 hNB)
    (h := rowB8_ltN ops p M K N TM TN hj (lt_trans hc.2 hNB))
    (fin_mem (by show portT8 p M N K TM TN _ _ = _; rw [portT8_B p _ htj hl, hje]) hcom)

/-- Where RowA `i` is correct, the tile cut's `P_A` row `i mod TM` is RowA `i`'s cut's `P_A` words. -/
theorem chain8_PA
    (hXA : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_A ops p M K N TM TN) hi⟩) {q : ℕ} (hq : q < 64) :
    val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN) (i / TM * (N / TN) + j / TN))
        ((TM + TN) * K + 64 * (i % TM) + q) =
      val ((whole8 ops p M K N TM TN).cut X (blockA8 ops p M K N) i) (paOut8 K q) := by
  obtain ⟨he, hie, -⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have ha : (TM + TN) * K + 64 * (i % TM) + q < tileIn K TM TN := by unfold tileIn; omega
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) + q / 16) (by unfold blockLen; omega)
  have hc := paOut8_bounds p (K := K) hq
  have hcom := (whole8 ops p M K N TM TN).port_committed (whole8_T ops p M K N TM TN) he hb.1 hb.2 ha
    (tile_reads_P p K TM TN (i := i % TM) htj hq).1
  rw [cutT8_input ops p M K N TM TN X he ha]
  show val X (portT8 p M N K TM TN _ _) = _
  rw [portT8_PA p _ hti hq, hie]
  exact cutA8_at ops p M K N TM TN X hi hXA hc.1 hc.2 (h := rowA8_ltN ops p M K N TM TN hi hc.2)
    (fin_mem (by show portT8 p M N K TM TN _ _ = _; rw [portT8_PA p _ hti hq, hie]) hcom)

/-- Where RowB `j` is correct, the tile cut's `P_B` row `j mod TN` is RowB `j`'s cut's `P_B` words. -/
theorem chain8_PB
    (hXB : (whole8 ops p M K N TM TN).part.Correct X
      ⟨_, (whole8 ops p M K N TM TN).unit_lt (whole8_B ops p M K N TM TN) hj⟩) {q : ℕ} (hq : q < 64) :
    val ((whole8 ops p M K N TM TN).cut X (blockT8 p M K N TM TN) (i / TM * (N / TN) + j / TN))
        ((TM + TN) * K + 64 * TM + 64 * (j % TN) + q) =
      val ((whole8 ops p M K N TM TN).cut X (blockB8 ops p M K N) j) (pbOut8 K q) := by
  obtain ⟨he, -, hje⟩ := tile_index hTM hTN hi hj
  have hti := Nat.mod_lt i (pos_of_dvd_lt hTM hi)
  have htj := Nat.mod_lt j (pos_of_dvd_lt hTN hj)
  have ha : (TM + TN) * K + 64 * TM + 64 * (j % TN) + q < tileIn K TM TN := by unfold tileIn; omega
  have hb := entry_bounds K TM TN hti htj (r := 5 * (K / 128) + q / 16) (by unfold blockLen; omega)
  have hc := pbOut8_bounds p (K := K) hq
  have hcom := (whole8 ops p M K N TM TN).port_committed (whole8_T ops p M K N TM TN) he hb.1 hb.2 ha
    (tile_reads_P p K TM TN (i := i % TM) htj hq).2
  rw [cutT8_input ops p M K N TM TN X he ha]
  show val X (portT8 p M N K TM TN _ _) = _
  rw [portT8_PB p _ htj hq, hje]
  exact cutB8_at ops p M K N TM TN X hj hXB hc.1 hc.2 (h := rowB8_ltN ops p M K N TM TN hj hc.2)
    (fin_mem (by show portT8 p M N K TM TN _ _ = _; rw [portT8_PB p _ htj hq, hje]) hcom)

end Chain8

end Pouw.PearlC.WholeCircuit
