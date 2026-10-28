import Definitions.Pouw.PearlC.RowCircuit
import Definitions.Core.Partition

/-!
# A whole Program's gate table from its units' tables (definitions only)

A tile-check Program (`verity.protocols.accounting.work.pouw.circuit.pc8`'s `Pc8Check`) is its proof units' templates wired together: each row
unit reads the Program's input regions, and each tile unit reads its rows' outputs. Its table here is a `Whole`: the
Program's input gates, then its units' computed gates, each unit an instance of a `Block`, a unit template's own table
(`TileCircuit.Spec`, whose inputs are its first `nIn` gates) with the gate each instance's input reads (`port`). An
instance's computed gates are the template's, in order, with every read relocated (`Block.reloc`): an input to the gate
its port names, a computed gate to the instance's own copy. These ports are the only wires between units.

The partition (`Whole.part`) puts each instance's gates in a unit of its own, numbered in the blocks' order, as
`verity.protocols.accounting.work.pouw.circuit.pc4.units` numbers the root body's members.

* **`pc8`** (`whole8`, `Pc8Check{M, K, N, TM, TN, DEV}`): the inputs in `Pc8Check`'s parameter-leaf order (`a`, `ea`,
  `b`, `eb`, `fa`, `fb`, `g`), then `N` instances of `Pc8RowB_v1{K}` (`row8B`), `M` of `Pc8RowA_v1{K}` (`row8A`) and
  `(M / TM)·(N / TN)` of `Pc8Tile_v1{K, TM, TN}` (`tileSpec`), the tiles row-major. Tile `(rs, cs)` reads `A′`'s codes and
  `P_A` from RowA `rs·TM + i` and `B̃`'s codes and `P_B` from RowB `cs·TN + j`. The noise lines and `G` are input
  regions: in the deployed Program they are verifier-derived public inputs, so no unit computes `G`.

The inputs' widths, the ports, the unit numbering and the outputs are checked against `pc8.unit_cut` of `Pc8Check`, with
its descriptor digests pinned, by `tests/test_circuit_whole_lean.py`; each unit's own table is tied to its template by
`test_circuit_row_lean.py` and `test_circuit_tile_lean.py`.
-/

namespace Pouw.PearlC.WholeCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.TileCircuit Pouw.PearlC.RowCircuit

/-! ## Composed tables -/

/-- One kind of proof unit of a composed table: the unit template's own table `S`, whose inputs are its first `nIn`
gates, and its `cnt` instances, instance `q`'s input `a` reading the composed table's gate `port q a`. -/
structure Block where
  S : Spec
  nIn : ℕ
  cnt : ℕ
  port : ℕ → ℕ → ℕ

namespace Block

variable (B : Block)

/-- An instance's computed gates. -/
def len : ℕ := B.S.N - B.nIn

/-- Gate `a` of instance `q` of a block whose gates start at `o`: an input is the gate its port names, a computed gate
the instance's own copy. -/
def reloc (o q a : ℕ) : ℕ := if a < B.nIn then B.port q a else o + q * B.len + (a - B.nIn)

/-- Gate `a` of the template as instance `q`'s copy: the template's width and value, its reads relocated. -/
def gateAt (o q a : ℕ) : GateRow := ⟨(B.S.gate a).w, false, (B.S.gate a).rd.map (B.reloc o q), (B.S.gate a).f⟩

end Block

/-- Where a computed gate lies: its block, the block's first gate, its unit, the instance and the template's gate. -/
structure Loc where
  B : Block
  o : ℕ
  u : ℕ
  q : ℕ
  a : ℕ

/-- Gate `g` of blocks laid out from gate `o`, their units numbered from `u`. -/
def locate : List Block → ℕ → ℕ → ℕ → Option Loc
  | [], _, _, _ => none
  | B :: L, o, u, g =>
    if g < o + B.cnt * B.len then some ⟨B, o, u + (g - o) / B.len, (g - o) / B.len, B.nIn + (g - o) % B.len⟩
    else locate L (o + B.cnt * B.len) (u + B.cnt) g

/-- Blocks laid out from gate `o` read earlier gates only: every port of an instance names a gate before its block. -/
def BlocksOK : List Block → ℕ → Prop
  | [], _ => True
  | B :: L, o => (∀ q < B.cnt, ∀ a < B.nIn, B.port q a < o) ∧ BlocksOK L (o + B.cnt * B.len)

/-- The blocks' computed gates. -/
def blocksLen : List Block → ℕ
  | [] => 0
  | B :: L => B.cnt * B.len + blocksLen L

/-- The blocks' units. -/
def unitCount : List Block → ℕ
  | [] => 0
  | B :: L => B.cnt + unitCount L

theorem locate_spec : ∀ (L : List Block) (o u g : ℕ) {l : Loc}, BlocksOK L o → o ≤ g → locate L o u g = some l →
    (∀ q < l.B.cnt, ∀ a < l.B.nIn, l.B.port q a < l.o) ∧ l.o ≤ g ∧ l.q < l.B.cnt ∧ l.B.nIn ≤ l.a ∧
      l.a < l.B.S.N ∧ g = l.o + l.q * l.B.len + (l.a - l.B.nIn) ∧ l.u < u + unitCount L
  | [], _, _, _, _, _, _, h => by simp [locate] at h
  | B :: L, o, u, g, l, hok, hog, h => by
    unfold locate at h
    split_ifs at h with hg
    · cases h
      have hL : 0 < B.len := Nat.pos_of_ne_zero fun h0 => by rw [h0] at hg; omega
      have hq : (g - o) / B.len < B.cnt := (Nat.div_lt_iff_lt_mul hL).2 (by omega)
      have hr := Nat.mod_lt (g - o) hL
      have hd := Nat.div_add_mod' (g - o) B.len
      refine ⟨hok.1, hog, hq, Nat.le_add_right _ _, ?_, ?_, ?_⟩
      · simp only [Block.len] at hL hr ⊢; omega
      · simp only [Nat.add_sub_cancel_left]; omega
      · simp only [unitCount]; omega
    · obtain ⟨h1, h2, h3, h4, h5, h6, h7⟩ := locate_spec L _ _ g hok.2 (by omega) h
      exact ⟨h1, h2, h3, h4, h5, h6, by simp only [unitCount]; omega⟩

theorem locate_isSome : ∀ (L : List Block) (o u g : ℕ), o ≤ g → g < o + blocksLen L → (locate L o u g).isSome
  | [], _, _, _, _, h => by simp only [blocksLen] at h; omega
  | B :: L, o, u, g, hog, hg => by
    unfold locate
    split_ifs with h
    · rfl
    · exact locate_isSome L _ _ g (by omega) (by simp only [blocksLen] at hg; omega)

/-- A composed table: `nIn` input gates, gate `a`'s width `inW a`, then each block's instances in turn, and the
outputs. -/
structure Whole where
  nIn : ℕ
  inW : ℕ → ℕ
  blocks : List Block
  outs : Finset ℕ
  ok : BlocksOK blocks nIn

namespace Whole

variable (W : Whole)

/-- The gate count. -/
def N : ℕ := W.nIn + blocksLen W.blocks

/-- Gate `g`: an input, or the copy of its template's gate in its instance. -/
def gate (g : ℕ) : GateRow :=
  if g < W.nIn then ⟨W.inW g, true, [], fun _ => 0⟩
  else match locate W.blocks W.nIn 0 g with
    | some l => l.B.gateAt l.o l.q l.a
    | none => ⟨0, true, [], fun _ => 0⟩

/-- **The composed table as a word circuit.** -/
def spec : Spec where
  N := W.N
  gate := W.gate
  outs := W.outs
  rd_lt g _ a ha := by
    unfold gate at ha
    split_ifs at ha with h
    · simp at ha
    · split at ha
      · next l hl =>
        obtain ⟨hp, hog, hq, -, haN, hg, -⟩ := locate_spec _ _ _ _ W.ok (not_lt.1 h) hl
        simp only [Block.gateAt, List.mem_map] at ha
        obtain ⟨a', ha', rfl⟩ := ha
        have := l.B.S.rd_lt l.a haN a' ha'
        unfold Block.reloc
        split_ifs with h'
        · exact lt_of_lt_of_le (hp l.q hq a' h') hog
        · omega
      · simp at ha
  input_rd g _ hi := by
    unfold gate at hi ⊢
    split_ifs with h
    · rfl
    · split
      · next l hl => simp [h, hl, Block.gateAt] at hi
      · rfl

/-- The unit of gate `g`: none for an input, else its instance's. -/
def unitOf (g : ℕ) : Option ℕ := if g < W.nIn then none else (locate W.blocks W.nIn 0 g).map Loc.u

/-- **The partition into the instances**, one unit each, numbered in the blocks' order. -/
def part : Verity.Partition W.spec.toCircuit (unitCount W.blocks) where
  unit g := (W.unitOf g.val).bind fun u => if h : u < unitCount W.blocks then some ⟨u, h⟩ else none
  input_iff g := by
    show _ ↔ (W.gate g.val).input = true
    unfold unitOf gate
    by_cases h : g.val < W.nIn
    · simp [h]
    · have hs := locate_isSome W.blocks W.nIn 0 g.val (not_lt.1 h) g.isLt
      obtain ⟨l, hl⟩ := Option.isSome_iff_exists.1 hs
      have hu := (locate_spec _ _ _ _ W.ok (not_lt.1 h) hl).2.2.2.2.2.2
      simp only [h, ↓reduceIte, hl, Option.map_some, Option.bind_some, Block.gateAt]
      simp only [zero_add] at hu
      simp [hu]

end Whole

/-! ## `pc8`'s whole Program -/

/-- `Pc8Check`'s input regions in its parameter-leaf order: `a`'s `M` rows of `K` FP32 words from gate `0` (row `i` at
`i·K`), `ea`'s `M` noise lines of 32 E4M3 codes from `eaAt` (line `i` at `·+ 32·i`), `b`'s `N` rows from `bAt`, `eb`'s
`N` lines from `ebAt`, the bases `fa` and `fb`, `K` lines each, from `faAt` and `fbAt` (line `t` at `·+ 32·t`), and
`g`'s 32 × 32 FP32 words from `gAt` (`g_{s,q}` at `·+ 32·s + q`). -/
def eaAt (M K : ℕ) : ℕ := M * K

/-- `b`'s first gate. -/
def bAt (M K : ℕ) : ℕ := M * (K + 32)

/-- `eb`'s first gate. -/
def ebAt (M N K : ℕ) : ℕ := M * (K + 32) + N * K

/-- `fa`'s first gate. -/
def faAt (M N K : ℕ) : ℕ := (M + N) * (K + 32)

/-- `fb`'s first gate. -/
def fbAt (M N K : ℕ) : ℕ := (M + N) * (K + 32) + 32 * K

/-- `g`'s first gate. -/
def gAt (M N K : ℕ) : ℕ := (M + N) * (K + 32) + 64 * K

/-- The input gates. -/
def in8 (M N K : ℕ) : ℕ := gAt M N K + 1024

/-- The input gates' widths: FP32 words, E4M3 codes and `G`'s FP32 words. -/
def inW8 (M N K g : ℕ) : ℕ :=
  if g < eaAt M K then 32 else if g < bAt M K then 8 else if g < ebAt M N K then 32 else if g < gAt M N K then 8
  else 32

/-- RowA `i`'s inputs (`rowIn K false`): row `i` of `a`, line `i` of `ea`, `fa` (its form basis) and `fb` (its chain
basis). -/
def portA8 (M N K i a : ℕ) : ℕ :=
  if a < K then i * K + a
  else if a < K + 32 then eaAt M K + 32 * i + (a - K)
  else if a < 33 * K + 32 then faAt M N K + (a - (K + 32))
  else fbAt M N K + (a - (33 * K + 32))

/-- RowB `j`'s inputs (`rowIn K true`): row `j` of `b`, line `j` of `eb`, `fb` (its form basis), `fa` (its chain basis)
and `g`. -/
def portB8 (M N K j a : ℕ) : ℕ :=
  if a < K then bAt M K + j * K + a
  else if a < K + 32 then ebAt M N K + 32 * j + (a - K)
  else if a < 33 * K + 32 then fbAt M N K + (a - (K + 32))
  else if a < 65 * K + 32 then faAt M N K + (a - (33 * K + 32))
  else gAt M N K + (a - (65 * K + 32))

/-- RowB `j`'s copy of its template's gate `a`. -/
def rowB8 (p : Pipeline) (M N K j a : ℕ) : ℕ := in8 M N K + j * ((row8B p K).N - rowIn K true) + (a - rowIn K true)

/-- RowA `i`'s copy of its template's gate `a`, after the `N` RowB units. -/
def rowA8 (p : Pipeline) (M N K i a : ℕ) : ℕ :=
  in8 M N K + N * ((row8B p K).N - rowIn K true) + i * ((row8A p K).N - rowIn K false) + (a - rowIn K false)

/-- Tile `e`'s copy of its template's gate `a`, after the row units. -/
def tile8 (p : Pipeline) (M N K TM TN e a : ℕ) : ℕ :=
  in8 M N K + N * ((row8B p K).N - rowIn K true) + M * ((row8A p K).N - rowIn K false) +
    e * (tileN K TM TN - tileIn K TM TN) + (a - tileIn K TM TN)

/-- RowA's `P_A` word `w`: its rescaled noise' BF16 word `w` (`w < 32`), else its chain against the chain basis' column
`w − 32`, cast to BF16. -/
def paOut8 (K w : ℕ) : ℕ :=
  if w < 32 then post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + w
  else post8At K (rowIn K false) + (w - 32) * (5 * (K / 128) + 1) + 5 * (K / 128)

/-- RowB's `P_B` word `w`: its block `w`'s last gate (`w < 32`), else its rescaled noise' word `w − 32` negated. -/
def pbOut8 (K w : ℕ) : ℕ :=
  if w < 32 then post8At K (rowIn K true) + w * (5 * (K / 128) + 35) + 5 * (K / 128) + 34
  else post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 2 * (w - 32) + 1

/-- Tile `e = rs·(N / TN) + cs`'s inputs (`tileIn K TM TN`): row `i` of its `A′` and its `P_A` row `i` from RowA
`rs·TM + i`, row `j` of its `B̃` and its `P_B` row `j` from RowB `cs·TN + j`. -/
def portT8 (p : Pipeline) (M N K TM TN e a : ℕ) : ℕ :=
  if a < TM * K then rowA8 p M N K (e / (N / TN) * TM + a / K) (code8 K (rowIn K false) (a % K))
  else if a < (TM + TN) * K then
    rowB8 p M N K (e % (N / TN) * TN + (a - TM * K) / K) (code8 K (rowIn K true) ((a - TM * K) % K))
  else if a < (TM + TN) * K + 64 * TM then
    rowA8 p M N K (e / (N / TN) * TM + (a - (TM + TN) * K) / 64) (paOut8 K ((a - (TM + TN) * K) % 64))
  else
    rowB8 p M N K (e % (N / TN) * TN + (a - ((TM + TN) * K + 64 * TM)) / 64)
      (pbOut8 K ((a - ((TM + TN) * K + 64 * TM)) % 64))

/-- `C̃_ij`'s gate: entry `(i mod TM, j mod TN)` of tile `(i / TM, j / TN)`. -/
def cGlob8 (p : Pipeline) (M N K TM TN i j : ℕ) : ℕ :=
  tile8 p M N K TM TN (i / TM * (N / TN) + j / TN) (cGate K TM TN (i % TM) (j % TN))

/-- `U_ij`'s gate. -/
def uGlob8 (p : Pipeline) (M N K TM TN i j : ℕ) : ℕ :=
  tile8 p M N K TM TN (i / TM * (N / TN) + j / TN) (uGate K TM TN (i % TM) (j % TN))

/-- `Pc8Check`'s outputs: every `C̃_ij` and `U_ij`, then each A row's flag and each B row's. -/
def outs8 (p : Pipeline) (M N K TM TN : ℕ) : Finset ℕ :=
  (Finset.range M ×ˢ Finset.range N).image (fun e => cGlob8 p M N K TM TN e.1 e.2) ∪
    (Finset.range M ×ˢ Finset.range N).image (fun e => uGlob8 p M N K TM TN e.1 e.2) ∪
    (Finset.range M).image (fun i => rowA8 p M N K i (post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + 32)) ∪
    (Finset.range N).image (fun j => rowB8 p M N K j (post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 64))

theorem row8A_N (p : Pipeline) (K : ℕ) :
    (row8A p K).N = post8At K (rowIn K false) + 32 * (5 * (K / 128) + 1) + 33 := by
  have hlen := pre8Runs_len p K (rowIn K false)
  show rowIn K false + runsLen (row8A p K).runs = _
  simp only [row8A, runsLen_append, runsLen, af8Run, Run.rep, bew8Run, okwRun, Run.ofList, List.length_cons,
    List.length_nil]
  omega

theorem row8B_N (p : Pipeline) (K : ℕ) :
    (row8B p K).N = post8At K (rowIn K true) + 32 * (5 * (K / 128) + 35) + 65 := by
  have hlen := pre8Runs_len p K (rowIn K true)
  show rowIn K true + runsLen (row8B p K).runs = _
  simp only [row8B, runsLen_append, runsLen, mid8Run, Run.rep, nb8Run, okwRun, Run.ofList, List.length_cons,
    List.length_nil]
  omega

theorem mul_add_lt {i n L a : ℕ} (hi : i < n) (ha : a < L) : i * L + a < n * L := by
  have : (i + 1) * L ≤ n * L := Nat.mul_le_mul_right L hi
  rw [Nat.succ_mul] at this
  omega

theorem in8_eq (M N K : ℕ) : in8 M N K = M * K + N * K + 32 * M + 32 * N + 64 * K + 1024 := by
  unfold in8 gAt; ring

theorem portA8_lt {M N K i a : ℕ} (hi : i < M) (ha : a < rowIn K false) : portA8 M N K i a < in8 M N K := by
  simp only [rowIn, Bool.false_eq_true, ↓reduceIte, add_zero] at ha
  rw [in8_eq]
  unfold portA8 eaAt faAt fbAt
  have e1 : (M + N) * (K + 32) = M * K + N * K + 32 * M + 32 * N := by ring
  rw [e1]
  split_ifs with h1 h2 h3
  · have := mul_add_lt (L := K) hi h1; omega
  · omega
  · omega
  · omega

theorem portB8_lt {M N K j a : ℕ} (hj : j < N) (ha : a < rowIn K true) : portB8 M N K j a < in8 M N K := by
  simp only [rowIn, ↓reduceIte] at ha
  rw [in8_eq]
  unfold portB8 bAt ebAt faAt fbAt gAt
  have e1 : (M + N) * (K + 32) = M * K + N * K + 32 * M + 32 * N := by ring
  have e2 : M * (K + 32) = M * K + 32 * M := by ring
  rw [e1, e2]
  split_ifs with h1 h2 h3 h4
  · have := mul_add_lt (L := K) hj h1; omega
  · omega
  · omega
  · omega
  · omega

theorem code8_bounds {K b t : ℕ} (ht : t < K) : b ≤ code8 K b t ∧ code8 K b t < post8At K b := by
  simp only [code8, post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At,
    bf1At, bf0At, fmaAt]
  omega

theorem post8At_ge (K b : ℕ) : b ≤ post8At K b := by
  simp only [post8At, code8At, ew8At, ep8At, ok8At, dead8At, th8At, sq8At, al8At, rhoAt, bf4At, bf3At, bf2At, bf1At,
    bf0At, fmaAt]
  omega

theorem paOut8_bounds (p : Pipeline) {K w : ℕ} (hw : w < 64) :
    rowIn K false ≤ paOut8 K w ∧ paOut8 K w < (row8A p K).N := by
  rw [row8A_N]
  have := post8At_ge K (rowIn K false)
  unfold paOut8
  split_ifs with h
  · omega
  · have := mul_add_lt (L := 5 * (K / 128) + 1) (show w - 32 < 32 by omega) (show 5 * (K / 128) < _ by omega)
    omega

theorem pbOut8_bounds (p : Pipeline) {K w : ℕ} (hw : w < 64) :
    rowIn K true ≤ pbOut8 K w ∧ pbOut8 K w < (row8B p K).N := by
  rw [row8B_N]
  have := post8At_ge K (rowIn K true)
  unfold pbOut8
  split_ifs with h
  · have := mul_add_lt (L := 5 * (K / 128) + 35) h (show 5 * (K / 128) + 34 < _ by omega)
    omega
  · omega

theorem rowA8_lt (p : Pipeline) {M N K i a : ℕ} (hi : i < M) (ha : a < (row8A p K).N) :
    rowA8 p M N K i a < in8 M N K + N * ((row8B p K).N - rowIn K true) + M * ((row8A p K).N - rowIn K false) := by
  unfold rowA8
  have hN : rowIn K false ≤ (row8A p K).N := by have := post8At_ge K (rowIn K false); rw [row8A_N]; omega
  have := mul_add_lt (L := (row8A p K).N - rowIn K false) hi (show a - rowIn K false < _ by
    have := post8At_ge K (rowIn K false); rw [row8A_N] at ha ⊢; omega)
  omega

theorem rowB8_lt (p : Pipeline) {M N K j a : ℕ} (hj : j < N) (ha : a < (row8B p K).N) :
    rowB8 p M N K j a < in8 M N K + N * ((row8B p K).N - rowIn K true) := by
  unfold rowB8
  have := mul_add_lt (L := (row8B p K).N - rowIn K true) hj (show a - rowIn K true < _ by
    have := post8At_ge K (rowIn K true); rw [row8B_N] at ha ⊢; omega)
  omega

/-- Row `i < TM` of tile `e < (M / TM)·(N / TN)`'s tile row is a row of A. -/
theorem tile_row_lt {M N TM TN e i : ℕ} (he : e < M / TM * (N / TN)) (hi : i < TM) : e / (N / TN) * TM + i < M := by
  have hn : 0 < N / TN := Nat.pos_of_ne_zero fun h => by rw [h] at he; simp at he
  have hrs : e / (N / TN) < M / TM := (Nat.div_lt_iff_lt_mul hn).2 he
  have := mul_add_lt (L := TM) hrs hi
  have := Nat.div_mul_le_self M TM
  omega

/-- Row `j < TN` of tile `e`'s tile column is a row of B. -/
theorem tile_col_lt {M N TM TN e j : ℕ} (he : e < M / TM * (N / TN)) (hj : j < TN) : e % (N / TN) * TN + j < N := by
  have hn : 0 < N / TN := Nat.pos_of_ne_zero fun h => by rw [h] at he; simp at he
  have := mul_add_lt (L := TN) (Nat.mod_lt e hn) hj
  have := Nat.div_mul_le_self N TN
  omega

theorem portT8_lt (p : Pipeline) {M N K TM TN e a : ℕ} (he : e < M / TM * (N / TN)) (ha : a < tileIn K TM TN) :
    portT8 p M N K TM TN e a <
      in8 M N K + N * ((row8B p K).N - rowIn K true) + M * ((row8A p K).N - rowIn K false) := by
  have hNA : post8At K (rowIn K false) < (row8A p K).N := by rw [row8A_N]; omega
  have hNB : post8At K (rowIn K true) < (row8B p K).N := by rw [row8B_N]; omega
  unfold portT8
  unfold tileIn at ha
  have hsplit : (TM + TN) * K = TM * K + TN * K := Nat.add_mul _ _ _
  split_ifs with h1 h2 h3
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by simp [h] at h1
    exact rowA8_lt p (tile_row_lt he ((Nat.div_lt_iff_lt_mul hK).2 h1))
      (lt_trans (code8_bounds (Nat.mod_lt _ hK)).2 hNA)
  · have hK : 0 < K := Nat.pos_of_ne_zero fun h => by simp [h] at h2
    have hj : (a - TM * K) / K < TN := (Nat.div_lt_iff_lt_mul hK).2 (by omega)
    exact lt_of_lt_of_le (rowB8_lt p (tile_col_lt he hj) (lt_trans (code8_bounds (Nat.mod_lt _ hK)).2 hNB))
      (Nat.le_add_right _ _)
  · have hi : (a - (TM + TN) * K) / 64 < TM := (Nat.div_lt_iff_lt_mul (by norm_num)).2 (by omega)
    exact rowA8_lt p (tile_row_lt he hi) (paOut8_bounds p (Nat.mod_lt _ (by norm_num))).2
  · have hj : (a - ((TM + TN) * K + 64 * TM)) / 64 < TN := (Nat.div_lt_iff_lt_mul (by norm_num)).2 (by omega)
    exact lt_of_lt_of_le (rowB8_lt p (tile_col_lt he hj) (pbOut8_bounds p (Nat.mod_lt _ (by norm_num))).2)
      (Nat.le_add_right _ _)

/-- **`Pc8Check{M, K, N, TM, TN}`'s table** on the step `p`, with `ops` for core's primitives. -/
def whole8 (ops : RowOps) (p : Pipeline) (M K N TM TN : ℕ) : Whole where
  nIn := in8 M N K
  inW := inW8 M N K
  blocks := [⟨(row8B p K).spec ops, rowIn K true, N, portB8 M N K⟩,
    ⟨(row8A p K).spec ops, rowIn K false, M, portA8 M N K⟩,
    ⟨tileSpec p K TM TN, tileIn K TM TN, M / TM * (N / TN), portT8 p M N K TM TN⟩]
  outs := outs8 p M N K TM TN
  ok := ⟨fun _ hj _ ha => portB8_lt hj ha, fun _ hi _ ha => lt_of_lt_of_le (portA8_lt hi ha) (Nat.le_add_right _ _),
    fun _ he _ ha => portT8_lt p he ha, trivial⟩

end Pouw.PearlC.WholeCircuit
