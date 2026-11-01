import Definitions.Pouw.PearlC.TileCircuit
import Definitions.Pouw.PearlC.DeviceFp4

/-!
# Pearl-C4's tile unit as a core circuit (definitions only)

`pc4`'s tile template `Pc4Tile_v1{K, TM, TN}` (`verity.protocols.accounting.work.pouw.circuit.pc4`, its `_tile_body` and `_chain`) as a
`Verity.Circuit`, a gate table (`TileCircuit.Spec`) as `TileCircuit` writes `pc8`'s.

* **Inputs**, in the template's parameter-leaf order, with `S = K/16` scale bytes per row: the E2M1 codes of `A′`'s
  `TM` rows (row `i` at gates `i·K, …`, 4 bits) and their UE4M3 scale bytes (from `TM·K + i·S`, 8 bits), the same for
  `B̃`'s `TN` rows (codes from `TM·(K + S) + j·K`, bytes from `TM·(K + S) + TN·K + j·S`), then the 64 BF16 words of
  each `P_A` row (from `(TM + TN)·(K + S) + 64·i`, 16 bits) and each `P_B` row (from
  `(TM + TN)·(K + S) + 64·TM + 64·j`).
* **Entry `(i, j)`'s block** (`entry4Gate`), the entries in row-major order:
  * **the chain**: `K/64` firings of `BlackwellNvf4OmmaDot64_v1` (`atom4F`, `nvf4Word`) from `+0`, each over the next
    64 code pairs and 4 scale bytes per operand, with no promotion. The last is `C̃_ij` (`c4Gate`);
  * **U**: four `HopperBF16WgmmaDot16_v1` steps (`TileCircuit.peelF`) from `C̃_ij` over the slices of `P_A,i` and
    `P_B,j`, as in `pc8`. The last is `U_ij` (`u4Gate`).
* **One proof unit** (`Spec.part`) holds every computed gate.

Which gate reads which, the widths and the outputs are checked against the template's `CallGraph`, with the checked
templates' descriptor digests pinned, by `tests/test_circuit_tile_lean.py`. The trusted part is as in `TileCircuit`: the
order of a gate's reads and each primitive's semantics, here `nvf4Word` for the FP4 step (the primitive's three rules,
with `Fp4Atom.step sm120Nvf4` on a finite accumulator and valid scale bytes).
-/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.Fp4

/-- An E2M1 code from a natural (its low nibble). -/
def nib (n : ℕ) : Nib := ⟨n % 16, Nat.mod_lt _ (by decide)⟩

/-- A scale byte from a natural (its low byte). -/
def byte (n : ℕ) : Fin 256 := ⟨n % 256, Nat.mod_lt _ (by decide)⟩

/-- `BlackwellNvf4OmmaDot64_v1` on words: a scale byte whose low seven bits are `0x7F` (the UE4M3 NaN) writes NaN
(`0x7FFFFFFF`); else a NaN accumulator writes NaN and an infinite one itself; else the firing `Fp4Atom.step sm120Nvf4`,
which reads a scale byte's low seven bits, with NaN where it leaves the model. -/
def nvf4Word (acc : ℕ) (a b : List Nib) (sa sb : List (Fin 256)) : ℕ :=
  if (sa ++ sb).any (fun s => s.val % 128 == 127) then 0x7FFFFFFF
  else if (acc >>> 23) % 256 = 255 then (if acc % 2 ^ 23 = 0 then acc else 0x7FFFFFFF)
  else (sm120Nvf4.step acc a b sa sb).getD 0x7FFFFFFF

/-- One firing of the chain: the accumulator (`+0` for the first, else the first value read), then the 64 code pairs
and each operand's 4 scale bytes read after it. -/
def atom4F (first : Bool) (l : List ℕ) : ℕ :=
  let r := if first then l else l.tail
  nvf4Word (if first then 0 else l.headD 0) (List.ofFn fun i : Fin 64 => nib (r.getD i 0))
    (List.ofFn fun i : Fin 64 => nib (r.getD (64 + i) 0)) (List.ofFn fun i : Fin 4 => byte (r.getD (128 + i) 0))
    (List.ofFn fun i : Fin 4 => byte (r.getD (132 + i) 0))

/-- One gate of an entry's block, at position `r` of the block and gate `g`: the chain's `K/64` firings over `A′_i`'s
codes from gate `a` and scale bytes from `sa` and `B̃_j`'s from `b` and `sb`, then U's four K16 steps over the 64 BF16
words of `P_A,i` from gate `pa` and of `P_B,j` from gate `pb`. -/
def entry4Gate (K a b sa sb pa pb g r : ℕ) : GateRow :=
  if r < K / 64 then
    ⟨32, false, (if r = 0 then [] else [g - 1]) ++ (List.range 64).map (a + 64 * r + ·) ++
        (List.range 64).map (b + 64 * r + ·) ++ (List.range 4).map (sa + 4 * r + ·) ++
        (List.range 4).map (sb + 4 * r + ·), atom4F (r == 0)⟩
  else
    ⟨32, false, [g - 1] ++ (List.range 16).map (pa + 16 * (r - K / 64) + ·) ++
        (List.range 16).map (pb + 16 * (r - K / 64) + ·), peelF⟩

/-- An entry's block: the chain's firings, then U's four K16 steps. -/
def block4Len (K : ℕ) : ℕ := K / 64 + 4

/-- A row's inputs: its `K` codes and `K/16` scale bytes. -/
def row4 (K : ℕ) : ℕ := K + K / 16

/-- The input gates: `A′`'s `TM` rows and `B̃`'s `TN` rows (codes, then scale bytes, for each operand), then each `P_A`
row and each `P_B` row. -/
def tile4In (K TM TN : ℕ) : ℕ := (TM + TN) * row4 K + 64 * (TM + TN)

/-- The first gate of entry `(i, j)`'s block; the blocks follow the inputs in row-major order. -/
def entry4Base (K TM TN i j : ℕ) : ℕ := tile4In K TM TN + (i * TN + j) * block4Len K

/-- `Pc4Tile_v1{K, TM, TN}`'s gates: the inputs (see the module docstring), then entry `(i, j)`'s block (`entry4Gate`)
for each entry in row-major order. -/
def tile4Gate (K TM TN g : ℕ) : GateRow :=
  if g < TM * K then ⟨4, true, [], fun _ => 0⟩
  else if g < TM * row4 K then ⟨8, true, [], fun _ => 0⟩
  else if g < TM * row4 K + TN * K then ⟨4, true, [], fun _ => 0⟩
  else if g < (TM + TN) * row4 K then ⟨8, true, [], fun _ => 0⟩
  else if g < tile4In K TM TN then ⟨16, true, [], fun _ => 0⟩
  else
    let e := (g - tile4In K TM TN) / block4Len K
    entry4Gate K (e / TN * K) (TM * row4 K + e % TN * K) (TM * K + e / TN * (K / 16))
      (TM * row4 K + TN * K + e % TN * (K / 16)) ((TM + TN) * row4 K + 64 * (e / TN))
      ((TM + TN) * row4 K + 64 * TM + 64 * (e % TN)) g ((g - tile4In K TM TN) % block4Len K)

/-- The gate count: the inputs and one block per entry. -/
def tile4N (K TM TN : ℕ) : ℕ := tile4In K TM TN + TM * TN * block4Len K

/-- `C̃_ij`'s gate: the last firing of entry `(i, j)`'s block. -/
def c4Gate (K TM TN i j : ℕ) : ℕ := entry4Base K TM TN i j + K / 64 - 1

/-- `U_ij`'s gate: the last K16 step of entry `(i, j)`'s block. -/
def u4Gate (K TM TN i j : ℕ) : ℕ := entry4Base K TM TN i j + block4Len K - 1

theorem block4Len_pos (K : ℕ) : 0 < block4Len K := by unfold block4Len; omega

/-- Gate `g` past the inputs lies in entry `e`'s block at position `r`, with `e < TM·TN` below the gate count. -/
theorem tile4_block {K TM TN g : ℕ} (hg : g < tile4N K TM TN) (h : tile4In K TM TN ≤ g) :
    g = tile4In K TM TN + (g - tile4In K TM TN) / block4Len K * block4Len K + (g - tile4In K TM TN) % block4Len K ∧
      (g - tile4In K TM TN) / block4Len K / TN < TM ∧ (g - tile4In K TM TN) / block4Len K % TN < TN ∧
      (g - tile4In K TM TN) % block4Len K < block4Len K := by
  have hL := block4Len_pos K
  have he : (g - tile4In K TM TN) / block4Len K < TM * TN := by
    rw [Nat.div_lt_iff_lt_mul hL]; unfold tile4N at hg; omega
  have hTN : 0 < TN := Nat.pos_of_ne_zero (by rintro rfl; simp at he)
  refine ⟨?_, (Nat.div_lt_iff_lt_mul hTN).2 he, Nat.mod_lt _ hTN, Nat.mod_lt _ hL⟩
  have := Nat.div_add_mod' (g - tile4In K TM TN) (block4Len K)
  omega

/-- **Pearl-C4's tile unit as a word circuit** at depth `K` and tile `TM × TN`; its outputs are every entry's `C̃` and
`U`. -/
def tile4Spec (K TM TN : ℕ) : Spec where
  N := tile4N K TM TN
  gate := tile4Gate K TM TN
  outs := (Finset.range TM ×ˢ Finset.range TN).image (fun e => c4Gate K TM TN e.1 e.2) ∪
    (Finset.range TM ×ˢ Finset.range TN).image (fun e => u4Gate K TM TN e.1 e.2)
  rd_lt g hg a ha := by
    unfold tile4Gate at ha
    split_ifs at ha with h1 h2 h3 h4 h5
    · simp at ha
    · simp at ha
    · simp at ha
    · simp at ha
    · simp at ha
    obtain ⟨hgb, hi, hj, hr⟩ := tile4_block hg (not_lt.1 h5)
    generalize (g - tile4In K TM TN) / block4Len K = e at ha hgb hi hj hr
    generalize (g - tile4In K TM TN) % block4Len K = r at ha hgb hr
    have hA : e / TN * K + K ≤ TM * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hi
    have hB : e % TN * K + K ≤ TN * K := by rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right K hj
    have hSA : e / TN * (K / 16) + K / 16 ≤ TM * (K / 16) := by
      rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hi
    have hSB : e % TN * (K / 16) + K / 16 ≤ TN * (K / 16) := by
      rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right _ hj
    have hIn : (TM + TN) * row4 K + 64 * (TM + TN) = tile4In K TM TN := rfl
    have hT : TM * row4 K = TM * K + TM * (K / 16) := by unfold row4; ring
    have hS : (TM + TN) * row4 K = TM * K + TM * (K / 16) + TN * K + TN * (K / 16) := by unfold row4; ring
    unfold block4Len at hr
    unfold entry4Gate at ha
    split_ifs at ha with h6 h7
    · simp only [List.append_assoc, List.mem_append, List.mem_nil_iff, false_or, List.mem_map, List.mem_range] at ha
      rcases ha with ⟨l, hl, rfl⟩ | ⟨l, hl, rfl⟩ | ⟨l, hl, rfl⟩ | ⟨l, hl, rfl⟩ <;> omega
    · simp only [List.append_assoc, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false, List.mem_map,
        List.mem_range] at ha
      rcases ha with rfl | ⟨l, hl, rfl⟩ | ⟨l, hl, rfl⟩ | ⟨l, hl, rfl⟩ | ⟨l, hl, rfl⟩ <;> omega
    · simp only [List.mem_append, List.mem_cons, List.mem_map, List.mem_range, List.singleton_append] at ha
      rcases ha with (rfl | ⟨l, hl, rfl⟩) | ⟨l, hl, rfl⟩ <;> omega
  input_rd g _ hi := by
    unfold tile4Gate at hi ⊢
    split_ifs at hi ⊢
    · rfl
    · rfl
    · rfl
    · rfl
    · rfl
    · unfold entry4Gate at hi; split_ifs at hi

end Pouw.PearlC.TileCircuit
