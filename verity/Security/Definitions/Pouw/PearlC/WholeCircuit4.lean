import Definitions.Pouw.PearlC.WholeCircuit
import Definitions.Pouw.PearlC.Row4Circuit
import Definitions.Pouw.PearlC.Tile4Circuit

/-!
# `pc4`'s whole Program as a gate table (definitions only)

`Pc4Check{M, K, N, TM, TN}` (`verity.protocols.accounting.work.pouw.circuit.pc4`) as a `WholeCircuit.Whole`, as `whole8` writes `Pc8Check`.
Its input regions have `pc8`'s sizes and order (`a`, `ea`, `b`, `eb`, `fa`, `fb`, `g`), so their first gates are
`eaAt` … `gAt` and `in8`, and a row unit reads them through `portA8` and `portB8`: RowA `i` reads row `i` of `a`, line
`i` of `ea`, `fa` (its form basis) and `fb` (its chain basis); RowB `j` reads row `j` of `b`, line `j` of `eb`, `fb`,
`fa` and `g`. Only the widths differ (`inW4`): BF16 rows, E2M1 lines and bases, and `G`'s FP32 words.

The units are `N` instances of `Pc4RowB_v1{K}` (`row4B`), `M` of `Pc4RowA_v1{K}` (`row4A`) and `(M / TM)·(N / TN)` of
`Pc4Tile_v1{K, TM, TN}` (`tile4Spec`), the tiles row-major. Tile `(rs, cs)` reads row `i` of its `A′` (codes, then scale
bytes) and `P_A` row `i` from RowA `rs·TM + i`, and row `j` of its `B̃` and `P_B` row `j` from RowB `cs·TN + j`
(`portT4`). As in `pc8`, the lines and `G` are verifier-derived public inputs, so no unit computes `G`.

`tests/test_circuit_whole4_lean.py` checks the inputs' widths, the ports, the unit numbering and the outputs against
`pc4.unit_cut` of `Pc4Check`, with its descriptor digests pinned.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

/-- `Pc4Check`'s input gates' widths: BF16 rows of `a`, E2M1 lines of `ea`, BF16 rows of `b`, E2M1 lines of `eb` and
bases `fa`, `fb`, and `G`'s FP32 words. -/
def inW4 (M N K g : ℕ) : ℕ :=
  if g < eaAt M K then 16 else if g < bAt M K then 4 else if g < ebAt M N K then 16 else if g < gAt M N K then 4
  else 32

/-- RowB `j`'s copy of its template's gate `a`. -/
def rowB4 (M N K j a : ℕ) : ℕ := in8 M N K + j * ((row4B K).N - rowIn K true) + (a - rowIn K true)

/-- RowA `i`'s copy of its template's gate `a`, after the `N` RowB units. -/
def rowA4 (M N K i a : ℕ) : ℕ :=
  in8 M N K + N * ((row4B K).N - rowIn K true) + i * ((row4A K).N - rowIn K false) + (a - rowIn K false)

/-- The first gate of the tiles, after the row units. -/
def tiles4 (M N K : ℕ) : ℕ :=
  in8 M N K + N * ((row4B K).N - rowIn K true) + M * ((row4A K).N - rowIn K false)

/-- Tile `e`'s copy of its template's gate `a`, after the row units. -/
def tile4 (M N K TM TN e a : ℕ) : ℕ :=
  tiles4 M N K + e * (tile4N K TM TN - tile4In K TM TN) + (a - tile4In K TM TN)

/-- RowA's `P_A` word `w`: its `Ẽ` word `w` (`w < 32`), else its chain against the chain basis' column `w − 32`, cast
to BF16. -/
def paOut4 (K w : ℕ) : ℕ :=
  if w < 32 then post4At K (rowIn K false) + 32 * (K / 64 + 1) + w
  else post4At K (rowIn K false) + (w - 32) * (K / 64 + 1) + K / 64

/-- RowB's `P_B` word `w`: its block `w`'s last gate (`w < 32`), else its `−Ẽ` word `w − 32`. -/
def pbOut4 (K w : ℕ) : ℕ :=
  if w < 32 then post4At K (rowIn K true) + 32 + w * (K / 64 + 35) + K / 64 + 34
  else post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + (w - 32)

/-- Tile `e = rs·(N / TN) + cs`'s inputs (`tile4In K TM TN`): row `i` of its `A′`'s codes and scale bytes and its
`P_A` row `i` from RowA `rs·TM + i`, row `j` of its `B̃`'s and its `P_B` row `j` from RowB `cs·TN + j`. -/
def portT4 (M N K TM TN e a : ℕ) : ℕ :=
  if a < TM * K then rowA4 M N K (e / (N / TN) * TM + a / K) (code4 K (rowIn K false) (a % K))
  else if a < TM * row4 K then
    rowA4 M N K (e / (N / TN) * TM + (a - TM * K) / (K / 16)) (scale4 K (rowIn K false) ((a - TM * K) % (K / 16)))
  else if a < TM * row4 K + TN * K then
    rowB4 M N K (e % (N / TN) * TN + (a - TM * row4 K) / K) (code4 K (rowIn K true) ((a - TM * row4 K) % K))
  else if a < (TM + TN) * row4 K then
    rowB4 M N K (e % (N / TN) * TN + (a - (TM * row4 K + TN * K)) / (K / 16))
      (scale4 K (rowIn K true) ((a - (TM * row4 K + TN * K)) % (K / 16)))
  else if a < (TM + TN) * row4 K + 64 * TM then
    rowA4 M N K (e / (N / TN) * TM + (a - (TM + TN) * row4 K) / 64) (paOut4 K ((a - (TM + TN) * row4 K) % 64))
  else
    rowB4 M N K (e % (N / TN) * TN + (a - ((TM + TN) * row4 K + 64 * TM)) / 64)
      (pbOut4 K ((a - ((TM + TN) * row4 K + 64 * TM)) % 64))

/-- `C̃_ij`'s gate: entry `(i mod TM, j mod TN)` of tile `(i / TM, j / TN)`. -/
def cGlob4 (M N K TM TN i j : ℕ) : ℕ :=
  tile4 M N K TM TN (i / TM * (N / TN) + j / TN) (c4Gate K TM TN (i % TM) (j % TN))

/-- `U_ij`'s gate. -/
def uGlob4 (M N K TM TN i j : ℕ) : ℕ :=
  tile4 M N K TM TN (i / TM * (N / TN) + j / TN) (u4Gate K TM TN (i % TM) (j % TN))

/-- `Pc4Check`'s outputs: every `C̃_ij` and `U_ij`, then each A row's flag and each B row's. -/
def outs4 (M N K TM TN : ℕ) : Finset ℕ :=
  (Finset.range M ×ˢ Finset.range N).image (fun e => cGlob4 M N K TM TN e.1 e.2) ∪
    (Finset.range M ×ˢ Finset.range N).image (fun e => uGlob4 M N K TM TN e.1 e.2) ∪
    (Finset.range M).image (fun i => rowA4 M N K i (post4At K (rowIn K false) + 32 * (K / 64 + 1) + 32)) ∪
    (Finset.range N).image (fun j => rowB4 M N K j (post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + 32))

theorem row4A_N (K : ℕ) : (row4A K).N = post4At K (rowIn K false) + 32 * (K / 64 + 1) + 33 := by
  have hlen := pre4Runs_len K (rowIn K false)
  show rowIn K false + runsLen (row4A K).runs = _
  simp only [row4A, runsLen_append, runsLen, af4Run, Run.rep, et4Run, okwRun, Run.ofList, List.length_cons,
    List.length_nil]
  omega

theorem row4B_N (K : ℕ) : (row4B K).N = post4At K (rowIn K true) + 32 + 32 * (K / 64 + 35) + 33 := by
  have hlen := pre4Runs_len K (rowIn K true)
  show rowIn K true + runsLen (row4B K).runs = _
  simp only [row4B, runsLen_append, runsLen, mid4Run, Run.rep, et4Run, nb4Run, okwRun, Run.ofList, List.length_cons,
    List.length_nil]
  omega

theorem post4At_ge (K b : ℕ) : b ≤ post4At K b := by
  simp only [post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
  omega

theorem code4_bounds {K b t : ℕ} (ht : t < K) : b ≤ code4 K b t ∧ code4 K b t < post4At K b + 52 := by
  simp only [code4, post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
  omega

theorem scale4_bounds {K b i : ℕ} (hi : i < K / 16) : b ≤ scale4 K b i ∧ scale4 K b i < post4At K b := by
  simp only [scale4, post4At, bk4At, d4At, a4At, rhoAt, bf4At, bf3At, bf2At, bf1At, bf0At, fmaAt]
  omega

theorem paOut4_bounds {K w : ℕ} (hw : w < 64) : rowIn K false ≤ paOut4 K w ∧ paOut4 K w < (row4A K).N := by
  rw [row4A_N]
  have := post4At_ge K (rowIn K false)
  unfold paOut4
  split_ifs with h
  · omega
  · have := mul_add_lt (L := K / 64 + 1) (show w - 32 < 32 by omega) (show K / 64 < _ by omega)
    omega

theorem pbOut4_bounds {K w : ℕ} (hw : w < 64) : rowIn K true ≤ pbOut4 K w ∧ pbOut4 K w < (row4B K).N := by
  rw [row4B_N]
  have := post4At_ge K (rowIn K true)
  unfold pbOut4
  split_ifs with h
  · have := mul_add_lt (L := K / 64 + 35) h (show K / 64 + 34 < _ by omega)
    omega
  · omega

theorem code4A_lt {K t : ℕ} (ht : t < K) : code4 K (rowIn K false) t < (row4A K).N := by
  rw [row4A_N]; have := (code4_bounds (b := rowIn K false) ht).2; omega

theorem code4B_lt {K t : ℕ} (ht : t < K) : code4 K (rowIn K true) t < (row4B K).N := by
  rw [row4B_N]; have := (code4_bounds (b := rowIn K true) ht).2; omega

theorem scale4A_lt {K i : ℕ} (hi : i < K / 16) : scale4 K (rowIn K false) i < (row4A K).N := by
  rw [row4A_N]; have := (scale4_bounds (b := rowIn K false) hi).2; omega

theorem scale4B_lt {K i : ℕ} (hi : i < K / 16) : scale4 K (rowIn K true) i < (row4B K).N := by
  rw [row4B_N]; have := (scale4_bounds (b := rowIn K true) hi).2; omega

theorem row4A_in_lt (K : ℕ) : rowIn K false < (row4A K).N := by
  have := post4At_ge K (rowIn K false); rw [row4A_N]; omega

theorem row4B_in_lt (K : ℕ) : rowIn K true < (row4B K).N := by
  have := post4At_ge K (rowIn K true); rw [row4B_N]; omega

theorem rowA4_lt {M N K i a : ℕ} (hi : i < M) (ha : a < (row4A K).N) :
    rowA4 M N K i a < tiles4 M N K := by
  unfold rowA4 tiles4
  have := mul_add_lt (L := (row4A K).N - rowIn K false) hi (show a - rowIn K false < _ by
    have := row4A_in_lt K; omega)
  omega

theorem rowB4_lt {M N K j a : ℕ} (hj : j < N) (ha : a < (row4B K).N) :
    rowB4 M N K j a < in8 M N K + N * ((row4B K).N - rowIn K true) := by
  unfold rowB4
  have := mul_add_lt (L := (row4B K).N - rowIn K true) hj (show a - rowIn K true < _ by
    have := row4B_in_lt K; omega)
  omega

theorem rowB4_lt' {M N K j a : ℕ} (hj : j < N) (ha : a < (row4B K).N) : rowB4 M N K j a < tiles4 M N K :=
  lt_of_lt_of_le (rowB4_lt hj ha) (by unfold tiles4; omega)

theorem portT4_lt {M N K TM TN e a : ℕ} (he : e < M / TM * (N / TN)) (ha : a < tile4In K TM TN) :
    portT4 M N K TM TN e a < tiles4 M N K := by
  unfold portT4
  unfold tile4In at ha
  have e1 : TM * row4 K = TM * K + TM * (K / 16) := by unfold row4; ring
  have e2 : (TM + TN) * row4 K = TM * K + TM * (K / 16) + TN * K + TN * (K / 16) := by unfold row4; ring
  split_ifs with h1 h2 h3 h4 h5
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by simp [h] at h1
    exact rowA4_lt (tile_row_lt he ((Nat.div_lt_iff_lt_mul hK).2 h1)) (code4A_lt (Nat.mod_lt _ hK))
  · have hS : 0 < K / 16 := Nat.pos_of_ne_zero fun h => by
      have : TM * (K / 16) = 0 := by rw [h, Nat.mul_zero]
      omega
    exact rowA4_lt (tile_row_lt he ((Nat.div_lt_iff_lt_mul hS).2 (by omega))) (scale4A_lt (Nat.mod_lt _ hS))
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by
      have : TN * K = 0 := by rw [h, Nat.mul_zero]
      omega
    exact rowB4_lt' (tile_col_lt he ((Nat.div_lt_iff_lt_mul hK).2 (by omega))) (code4B_lt (Nat.mod_lt _ hK))
  · have hS : 0 < K / 16 := Nat.pos_of_ne_zero fun h => by
      have : TN * (K / 16) = 0 := by rw [h, Nat.mul_zero]
      omega
    exact rowB4_lt' (tile_col_lt he ((Nat.div_lt_iff_lt_mul hS).2 (by omega))) (scale4B_lt (Nat.mod_lt _ hS))
  · exact rowA4_lt (tile_row_lt he ((Nat.div_lt_iff_lt_mul (by norm_num)).2 (by omega)))
      (paOut4_bounds (Nat.mod_lt _ (by norm_num))).2
  · exact rowB4_lt' (tile_col_lt he ((Nat.div_lt_iff_lt_mul (by norm_num)).2 (by omega)))
      (pbOut4_bounds (Nat.mod_lt _ (by norm_num))).2

/-- **`Pc4Check{M, K, N, TM, TN}`'s table**, with `ops` for core's primitives. -/
def whole4 (ops : RowOps) (M K N TM TN : ℕ) : Whole where
  nIn := in8 M N K
  inW := inW4 M N K
  blocks := [⟨(row4B K).spec ops, rowIn K true, N, portB8 M N K⟩,
    ⟨(row4A K).spec ops, rowIn K false, M, portA8 M N K⟩,
    ⟨tile4Spec K TM TN, tile4In K TM TN, M / TM * (N / TN), portT4 M N K TM TN⟩]
  outs := outs4 M N K TM TN
  ok := ⟨fun _ hj _ ha => portB8_lt hj ha, fun _ hi _ ha => lt_of_lt_of_le (portA8_lt hi ha) (Nat.le_add_right _ _),
    fun _ he _ ha => portT4_lt he ha, trivial⟩

end Pouw.PearlC.WholeCircuit
