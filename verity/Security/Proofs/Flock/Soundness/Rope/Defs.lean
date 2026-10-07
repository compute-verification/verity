import Proofs.Flock.Soundness.GateRows
import Proofs.Flock.Soundness.Rope.Gates
import Proofs.Flock.Soundness.Rope.Pinned
import Proofs.Flock.Soundness.Rope.Checkpoints

/-!
# RoPE's unit as data: its gate circuit, its layout and its pinned rows

Decoders for the generated data (`Rope/Gates.lean`, `Rope/Pinned.lean`, `Rope/Checkpoints.lean`;
`verity_flock.lean_rows`).
-/

namespace FlockSoundness.Rope

open GateRows

def decSrc (n : ℕ) : Src := if n = 0 then .zero else if n = 1 then .one else .wire (n - 2)

/-- A gate from its code `op + 3·(a + 2^17·b)`, operands coded `0`, `1` (constants) or `2 + wire`. -/
def decGate (c : ℕ) : Gate :=
  if c % 3 = 0 then .and (decSrc (c / 3 % 2 ^ 17)) (decSrc (c / 3 / 2 ^ 17))
  else if c % 3 = 1 then .xor (decSrc (c / 3 % 2 ^ 17)) (decSrc (c / 3 / 2 ^ 17))
  else .not (decSrc (c / 3 % 2 ^ 17))

/-- A row's columns from their code: its runs `(first, length)`, base-2^22 digits `first·512 + length + 1`. -/
def unpackRuns : ℕ → ℕ → List (ℕ × ℕ)
  | 0, _ => []
  | f + 1, n =>
    if n = 0 then [] else ((n % 2 ^ 22 - 1) / 512, (n % 2 ^ 22 - 1) % 512) :: unpackRuns f (n / 2 ^ 22)

/-- The gate circuit: the Boolean export of the unit. -/
def circ : Circ := ⟨nIn, gateCodes.map decGate, outCodes.map decSrc⟩

/-- The v2 layout of the unit: one input group (four 16-bit ports in one 128-bit word, columns
`0 … 63`), the ANDs from column 128, and one output group (two 16-bit ports) at word 47, so the
output region runs from column 5956 to 6143 with its copy rows at `6016 … 6047`. -/
def lay : Lay := ⟨fun k => k, 128, 188, fun r => if 6016 ≤ r ∧ r < 6048 then some (r - 6016) else none⟩

/-- A pinned row's masks, from its runs. -/
def pinMasks (c : ℕ × ℕ) : ℕ × ℕ := (runsMask (unpackRuns 512 c.1), runsMask (unpackRuns 512 c.2))

/-- A pinned row's runs lie below the constant's column. -/
def runsBounded (c : ℕ × ℕ) : Bool := (unpackRuns 512 c.1 ++ unpackRuns 512 c.2).all fun r => r.1 + r.2 ≤ 6145

/-- A checkpoint from its data, each form's runs as a mask. -/
def decCP (d : ℕ × ℕ × List (ℕ × ℕ × Bool)) : CP :=
  ⟨d.1, d.2.1, d.2.2.map fun e => (e.1, (runsMask (unpackRuns 512 e.2.1), e.2.2))⟩

/-- The output region's row at column `r` (`GateRows.outRow`) with the constant's column given. -/
def outRowK (k : ℕ) (S : St) (r : ℕ) : ℕ × ℕ :=
  match lay.outAt r with
  | some o => (mask k (srcForm S.forms (circ.outs.getD o .zero)), 2 ^ k)
  | none => (0, 0)

/-- The pinned rows, as runs of columns. -/
def pinnedRuns : List (List (ℕ × ℕ) × List (ℕ × ℕ)) :=
  pinnedCodes.map fun p => (unpackRuns 512 p.1, unpackRuns 512 p.2)

/-- The pinned rows, as the file lists their columns. -/
def pinned : List (List ℕ × List ℕ) := pinnedRuns.map fun p => (runsList p.1, runsList p.2)

end FlockSoundness.Rope
