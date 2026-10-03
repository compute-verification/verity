import Verity.Protocol.Partition
import Pouw.Protocol.PearlC.PeelExact

/-!
# The tile unit as a core circuit (definitions only)

`pc8`'s tile template `Pc8Tile_v1{K, TM = 1, TN = 1, DEV}` (`verity_pouw.circuit.pc8`, its `_tile_body` and `_chain`) as
a `Verity.Circuit`, written as a gate table (`Spec`): for each gate its width, whether it is an input, the earlier gates
it reads in order, and its value from theirs. No evaluator for IR Programs: the table is the template's computing graph
(`verity.ir.cut.CallGraph`), with each IR primitive's semantics a Lean function on words.

* **Inputs**, in the template's parameter-leaf order: the E4M3 codes of `A′_i` (gates `0, …, K − 1`, 8 bits) and of
  `B̃_j` (`K, …, 2K − 1`), and the 64 BF16 words of `P_A,i` (`2K, …, 2K + 63`, 16 bits) and of `P_B,j` (`2K + 64, …`).
* **The chain** on the step `p` (`BlackwellE4m3QmmaDot32_v1` on sm_120, `p = sm120E4m3`): per window of 4 atoms, four
  atoms from `+0` (`atomF`) and one `F32Add_v2` (`addF`, the total `f32Add`) promoting the window's word into the total.
  The last promotion is `C̃_ij` (`cGate`).
* **U**: four `HopperBF16WgmmaDot16_v1` steps (`peelF`, `PeelExact.k16Step`) from `C̃_ij` over the slices of `P_A,i` and
  `P_B,j`. The last is `U_ij` (`uGate`).
* **One proof unit** (`Spec.part`) holds every computed gate.

Which gate reads which, the widths and the outputs are checked against the template's `CallGraph`, with the checked
templates' descriptor digests pinned, by a test beside `pc8`'s. The order of a gate's reads is not in the `CallGraph`;
it and each primitive's semantics are the trusted part: `F32Add_v2` is `f32Add`, the atom is `p.step` and the K16 step
is `k16Step`, each pinned by its conformance vectors.
-/

namespace Pouw.PearlC.TileCircuit

open Pouw.Fp8Atom Pouw.PearlC Pouw.PearlC.PeelExact

/-- One gate of a word circuit: its width, whether it is an input, the gates it reads in order, and its value from
theirs as naturals. -/
structure Gate where
  w : ℕ
  input : Bool
  rd : List ℕ
  f : List ℕ → ℕ

/-- A word circuit: `N` gates, each reading earlier gates only, inputs reading none. -/
structure Spec where
  N : ℕ
  gate : ℕ → Gate
  outs : Finset ℕ
  rd_lt : ∀ g < N, ∀ a ∈ (gate g).rd, a < g
  input_rd : ∀ g < N, (gate g).input = true → (gate g).rd = []

/-- Gate `a`'s value in `v` as a natural (`0` past the last gate). -/
def val {n : ℕ} {β : Fin n → ℕ} (v : (a : Fin n) → BitVec (β a)) (a : ℕ) : ℕ :=
  if h : a < n then (v ⟨a, h⟩).toNat else 0

namespace Spec

variable (S : Spec)

/-- The word circuit as a core circuit. -/
@[reducible] def toCircuit : Verity.Circuit where
  N := S.N
  w g := (S.gate g).w
  isInput g := (S.gate g).input
  reads g := Finset.univ.filter fun a => a.val ∈ (S.gate g).rd
  reads_lt g a ha := by
    simp only [Finset.mem_filter, Finset.mem_univ, true_and] at ha
    exact S.rd_lt g g.isLt a ha
  input_reads g hg := by
    ext a
    simp [S.input_rd g g.isLt hg]
  op g v := BitVec.ofNat _ ((S.gate g).f ((S.gate g).rd.map (val v)))
  op_congr g v v' h := by
    congr 2
    apply List.map_congr_left
    intro a ha
    unfold val
    split
    · next hlt => rw [h ⟨a, hlt⟩ (by simp [ha])]
    · rfl
  outputs := Finset.univ.filter fun g => g.val ∈ S.outs

/-- One proof unit holding every computed gate. -/
def part : Verity.Partition S.toCircuit 1 where
  unit g := if (S.gate g).input then none else some 0
  input_iff g := by
    show (if (S.gate g).input then none else some 0) = none ↔ (S.gate g).input = true
    split <;> simp_all

end Spec

/-- `F32Add_v2`: IEEE binary32 addition of two FP32 words, NaN written as `0x7FFFFFFF`. On finite words it is
`PeelExact.addWord?`'s rounding without its finiteness check, so an overflow writes the infinity `f32Word` rounds to. -/
def f32Add (a b : ℕ) : ℕ :=
  match wordQ a, wordQ b with
  | some x, some y => rnWord (x + y) (decide (x = 0) && decide (y = 0) && signBit a && signBit b)
  | _, _ =>
    packValue <| match fp32Value a, fp32Value b with
      | .inf s, .inf t => if s = t then .inf s else .nan
      | .inf s, .fin _ => .inf s
      | .fin _, .inf t => .inf t
      | _, _ => .nan

/-- An E4M3 code from a natural (its low byte). -/
def code (n : ℕ) : Code := ⟨n % 256, Nat.mod_lt _ (by decide)⟩

/-- One atom of the chain on the step `p`: the accumulator (`+0` for a window's first atom, else the first value read)
and the 32 code pairs read after it. -/
def atomF (p : Pipeline) (first : Bool) (l : List ℕ) : ℕ :=
  let r := if first then l else l.tail
  p.step (if first then 0 else l.headD 0) (List.ofFn fun i : Fin 32 => code (r.getD i 0))
    (List.ofFn fun i : Fin 32 => code (r.getD (32 + i) 0))

/-- One promotion: the window's word added into the total (`+0` for the first window). -/
def addF (first : Bool) (l : List ℕ) : ℕ := if first then f32Add 0 (l.headD 0) else f32Add (l.headD 0) (l.getD 1 0)

/-- One K16 step of U: the accumulator and 16 BF16 word pairs. -/
def peelF (l : List ℕ) : ℕ := k16Step (l.headD 0) (fun i => l.getD (1 + i) 0) (fun i => l.getD (17 + i) 0)

/-- `Pc8Tile_v1{K, TM = 1, TN = 1}`'s gates, the chain on the step `p` in windows of 4 atoms: the codes of `A′_i`
(gates `0, …, K − 1`) and `B̃_j` (`K, …, 2K − 1`), the 64 BF16 words of `P_A,i` (`2K, …`) and `P_B,j` (`2K + 64, …`),
then per window four atoms and one promotion, then U's four K16 steps. -/
def tileGate (p : Pipeline) (K g : ℕ) : Gate :=
  if g < 2 * K then ⟨8, true, [], fun _ => 0⟩
  else if g < 2 * K + 128 then ⟨16, true, [], fun _ => 0⟩
  else if g - (2 * K + 128) < 5 * (K / 128) then
    if (g - (2 * K + 128)) % 5 < 4 then
      ⟨32, false, (if (g - (2 * K + 128)) % 5 = 0 then [] else [g - 1]) ++
          (List.range 32).map (32 * (4 * ((g - (2 * K + 128)) / 5) + (g - (2 * K + 128)) % 5) + ·) ++
          (List.range 32).map (K + 32 * (4 * ((g - (2 * K + 128)) / 5) + (g - (2 * K + 128)) % 5) + ·),
        atomF p ((g - (2 * K + 128)) % 5 == 0)⟩
    else ⟨32, false, (if (g - (2 * K + 128)) / 5 = 0 then [] else [g - 5]) ++ [g - 1],
      addF ((g - (2 * K + 128)) / 5 == 0)⟩
  else
    ⟨32, false, [g - 1] ++ (List.range 16).map (2 * K + 16 * (g - (2 * K + 128) - 5 * (K / 128)) + ·) ++
        (List.range 16).map (2 * K + 64 + 16 * (g - (2 * K + 128) - 5 * (K / 128)) + ·), peelF⟩

/-- The gate count: the inputs, five gates per window, four K16 steps. -/
def tileN (K : ℕ) : ℕ := 2 * K + 128 + 5 * (K / 128) + 4

/-- `C̃`'s gate: the last promotion. -/
def cGate (K : ℕ) : ℕ := 2 * K + 128 + 5 * (K / 128) - 1

/-- `U`'s gate: the last K16 step. -/
def uGate (K : ℕ) : ℕ := tileN K - 1

/-- **The tile unit as a word circuit** at depth `K` on the step `p`. -/
def tileSpec (p : Pipeline) (K : ℕ) : Spec where
  N := tileN K
  gate := tileGate p K
  outs := {cGate K, uGate K}
  rd_lt g hg a ha := by
    unfold tileN at hg
    unfold tileGate at ha
    have h128 : 128 * (K / 128) ≤ K := Nat.mul_div_le K 128
    split_ifs at ha with h1 h2 h3 h4 h5 h6
    · simp at ha
    · simp at ha
    · simp only [List.append_assoc, List.mem_append, List.mem_nil_iff, false_or, List.mem_map, List.mem_range] at ha
      rcases ha with ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩ <;> omega
    · simp only [List.append_assoc, List.mem_append, List.mem_cons, List.mem_nil_iff, or_false, List.mem_map,
        List.mem_range] at ha
      rcases ha with rfl | ⟨i, hi, rfl⟩ | ⟨i, hi, rfl⟩ <;> omega
    · simp only [List.nil_append, List.mem_cons, List.mem_nil_iff, or_false] at ha
      omega
    · simp only [List.mem_append, List.mem_cons, List.mem_nil_iff, or_false] at ha
      omega
    · simp only [List.mem_append, List.mem_cons, List.mem_map, List.mem_range, List.singleton_append] at ha
      rcases ha with (rfl | ⟨i, hi, rfl⟩) | ⟨i, hi, rfl⟩ <;> omega
  input_rd g _ hi := by
    unfold tileGate at hi ⊢
    split_ifs at hi ⊢ <;> simp_all

end Pouw.PearlC.TileCircuit
