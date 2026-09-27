import FlockSoundness.GateRows
import FlockSoundness.Rope.Gates
import FlockSoundness.Rope.Pinned

/-!
# RoPE's rows are its Boolean export's rows

The pinned `rope-head/neox-bf16/pair` unit (`flock-ir-unit/v2`, sha256 `933c4ef8…`,
`verity_flock.templates.rope_head.PINS`) and its gate circuit, the IR's Boolean export
(`verity_flock.boolean_export`: 64 inputs, 5,996 AND, 9,774 XOR and 1,733 NOT gates, 32 outputs),
are generated data (`Rope/Pinned.lean`, `Rope/Gates.lean`). `backends/flock/tests/test_lean_rope.py`
regenerates both and checks them byte for byte.

* `pinned_eq`: the pinned rows are the rows `GateRows.rows` derives from the gate circuit, by a
  kernel computation (`decide +kernel`, on bitmasks).
* `rope_sound`: so every column assignment that satisfies the pinned rows, with the constant 1,
  carries on its 32 output columns the gate circuit's outputs on its 64 input columns
  (`GateRows.rows_sound`).
-/

namespace FlockSoundness.Rope

open GateRows Audit

def decSrc (n : ℕ) : Src := if n = 0 then .zero else if n = 1 then .one else .wire (n - 2)

/-- A gate from its code `op + 3·(a + 2^17·b)`, operands coded `0`, `1` (constants) or `2 + wire`. -/
def decGate (c : ℕ) : Gate :=
  if c % 3 = 0 then .and (decSrc (c / 3 % 2 ^ 17)) (decSrc (c / 3 / 2 ^ 17))
  else if c % 3 = 1 then .xor (decSrc (c / 3 % 2 ^ 17)) (decSrc (c / 3 / 2 ^ 17))
  else .not (decSrc (c / 3 % 2 ^ 17))

/-- A row's columns from their code: base-8192 digits, each column plus one. -/
def unpack : ℕ → ℕ → List ℕ
  | 0, _ => []
  | f + 1, n => if n = 0 then [] else (n % 8192 - 1) :: unpack f (n / 8192)

/-- The gate circuit: the Boolean export of the unit. -/
def circ : Circ := ⟨nIn, gateCodes.map decGate, outCodes.map decSrc⟩

/-- The v2 layout of the unit: one input group (four 16-bit ports in one 128-bit word, columns
`0 … 63`), the ANDs from column 128, and one output group (two 16-bit ports) at word 48, so the
output region runs from column 6124 to 6271 with its copy rows at `6144 … 6175`. -/
def lay : Lay := ⟨fun k => k, 128, 148, fun r => if 6144 ≤ r ∧ r < 6176 then some (r - 6144) else none⟩

/-- The pinned rows, as the file lists their columns. -/
def pinned : List (List ℕ × List ℕ) := pinnedCodes.map fun p => (unpack 512 p.1, unpack 512 p.2)

/-- **The pinned rows are the derived rows**, checked by the kernel. -/
theorem pinned_eq : pinned.map (fun p => (listMask p.1, listMask p.2)) = rows circ lay := by
  decide +kernel

theorem facts : pinned.length = 6273 ∧ (pinned.all fun p => p.1.all (· < 6273) && p.2.all (· < 6273)) = true ∧
    gatesWF circ.nIn circ.gates = true ∧ circ.nIn = 64 ∧ circ.gates.length = 17503 ∧ nAnd circ = 5996 ∧
    circ.outs.length = 32 ∧ (circ.outs.all fun o => o.below (64 + 17503)) = true := by
  decide +kernel

/-- Rows satisfied by column values `z`, reading each row's columns as a list. -/
def SatList (R : List (List ℕ × List ℕ)) (z : ℕ → Bool) : Prop :=
  ∀ r p, R[r]? = some p → z r = (xorSum p.1 z && xorSum p.2 z)

/-- **RoPE's pinned rows compute its Boolean export.** Column values that satisfy the pinned rows,
with the constant (column 6272) at 1, carry on output column `6144 + o` the gate circuit's output
`o` on input columns `0 … 63`. -/
theorem rope_sound (z : ℕ → Bool) (hz : SatList pinned z) (h1 : z 6272 = true) (o : ℕ) (ho : o < 32) :
    z (6144 + o) = (circ.eval z)[o]'(by simpa [Circ.eval, facts.2.2.2.2.2.2.1] using ho) := by
  obtain ⟨hlen, hcols, hwf, hin, hgl, hand, hol, hob⟩ := facts
  have hR : (rows circ lay).length = 6273 := by rw [← pinned_eq, List.length_map, hlen]
  have hsat : Sat (rows circ lay) z := by
    intro r p hp
    rw [← pinned_eq, List.getElem?_map] at hp
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.mp hp
    have hqm : q ∈ pinned := List.mem_of_getElem? hq
    have hall := List.all_eq_true.mp hcols q hqm
    simp only [Bool.and_eq_true, List.all_eq_true, decide_eq_true_eq] at hall
    rw [hR, hz r q hq, xorSum_eq_psum z 6273 q.1 hall.1, xorSum_eq_psum z 6273 q.2 hall.2]
  have hc : constCol circ lay = 6272 := by simp [constCol, lay, hand]
  have houts : ∀ s ∈ circ.outs, s.below (circ.nIn + circ.gates.length) = true := by
    intro s hs; rw [hin, hgl]; exact List.all_eq_true.mp hob s hs
  have key := rows_sound circ lay hwf (by rw [hin, hgl]; decide) (fun k hk => by simp [lay]; omega) houts z hsat
    (by rw [hc]; exact h1) (20 + o) (by simp [lay]; omega) o
    (by simp only [lay, hand]; rw [if_pos (by omega)]; congr 1; omega) (by rw [hol]; exact ho)
  simp only [lay, hand, show 128 + 5996 + (20 + o) = 6144 + o by omega] at key
  exact key

end FlockSoundness.Rope
