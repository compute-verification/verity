import Mathlib.Data.ZMod.Defs
import Pous.Protocol.Model.SecureErasure
import Pous.Protocol.Model.Digest

/-!
# The secure-erasure accept rule, evaluated by Lean on the difftest's cases

Reads cases on stdin, one per line, as `tests/pous_erasure_cases.py` writes them:

    Tfill Tread s M pageBytes g z fAnswered fTime n d_1 … d_n F_0 … F_(M−1) (i a v t)×s

and prints one verdict per case, `1` accept and `0` reject, on one line.  `tests/pous_erasure_cases.py` runs it
(`lake env lean --run` in `lean/`, after `lake build Pous.Protocol.Model.SecureErasure Pous.Protocol.Model.Digest`) and
pins the verdicts in `fixtures/lean_erasure_verdicts.json`, against which `test_pous_erasure_difftest.py` checks
`verity_pous.erasure.ErasureVerifier`.

**The verdict is the model's.**  It is `Pous.eraseWins` (`Pous/Protocol/Model/SecureErasure.lean`) on one outcome of
the secure-erasure game in the plain model, with no foreign content (`S_φ = 0`, so keeping it is free and `eraseWins`
is the verifier's accept event):

- the scheme's `M` blocks are the fill's bytes `F_i` (`F_K`, as Python computed it: Lean has no SHAKE256);
- the digest `dig` is `Pous.Digest.digest g z` over `GF(2^127 − 1)` of the fill's pages, each `⌈pageBytes / 15⌉`
  15-byte little-endian words, zero-padded, written as 16 bytes little-endian: Lean computes the digest itself;
- the outcome's digest key is `(g, z)` and its `s` indices are the `i` of the reads;
- the prover returns the digest bytes `d` for the fill and the value `v` of read `j` for round `j`.

**The deadlines are not.**  Lean's game has no clock: its responders are `(D, Q)`-bounded, so they always answer in time
(`Pous/Protocol/Model/Audit.lean`).  The one rule written here is the verifier's reading of a missed deadline: an
answer that did not come within it (`a = 0`, or `t > T`) is one that no digest or block is (`none`, `late`).
-/

namespace PousErasureCheck

open Pous

/-- The digest's field (`verity_pous.erasure.DIGEST_PRIME`). -/
abbrev F : Type := ZMod (2 ^ 127 - 1)

/-- A block: one byte of the fill, as 64 bits, wide enough for every answer the cases give. -/
def block (v : ℕ) : Bits 64 := fun k => v.testBit k

/-- The block no byte is, `2^64 − 1`: a read answered after its deadline, or not at all. -/
def late : Bits 64 := fun _ => true

/-- The byte a block of the fill holds. -/
def byte (b : Bits 64) : ℕ :=
  (List.finRange 8).foldr (fun k acc => (if b (Fin.castLE (by decide) k) then 1 else 0) + 2 * acc) 0

/-- The scheme of one case in the plain model: `m + 1` blocks, block `i` the fill's byte `i`. -/
def scheme (m : ℕ) (fill : Array ℕ) : Scheme IdealModel.plain where
  n := 0
  t := 0
  B := m + 1
  ℓ := 64
  B_pos := Nat.succ_pos m
  Coins := Unit
  enc := fun _ _ _ => (fun i => i.elim0, fun i => block (fill.getD i 0))
  dec := fun _ _ _ i => i.elim0

def pages (B pageBytes : ℕ) : ℕ := (B + pageBytes - 1) / pageBytes

def words (pageBytes : ℕ) : ℕ := (pageBytes + 14) / 15

/-- Word `w` of page `r`: bytes `15w … 15w + 14` of the page, little-endian, zero past the page or the fill. -/
def word {B : ℕ} (pageBytes : ℕ) (C : Fin B → Bits 64) (r w : ℕ) : ℕ :=
  (List.range 15).foldr (fun k acc =>
    let q := r * pageBytes + 15 * w + k
    (if h : 15 * w + k < pageBytes ∧ q < B then byte (C ⟨q, h.2⟩) else 0) + 256 * acc) 0

/-- The fill check's digest: `Pous.Digest.digest g z` of the pages, as 16 bytes little-endian. -/
def dig {B : ℕ} (pageBytes : ℕ) (κ : F × F) (C : Fin B → Bits 64) : Option (List ℕ) :=
  let xs : Fin (pages B pageBytes) → Digest.Blk (words pageBytes) F := fun r w => (word pageBytes C r w : F)
  let v := (Digest.digest κ.1 κ.2 xs).val
  some ((List.range 16).map fun k => v / 256 ^ k % 256)

/-- The prover of one case: its digest for the fill, `reads[j]` for round `j`, nothing kept. -/
def prover (m : ℕ) (fill : Array ℕ) (digest : Option (List ℕ)) (reads : Array (Bits 64)) :
    EraseProver (scheme m fill) 0 0 0 0 (F × F) (Option (List ℕ)) where
  fill := fun _ _ _ _ _ _ => (digest, fun i => i.elim0)
  read := ⟨fun _ _ j σ _ => .ret (reads.getD j late, σ), fun _ _ _ _ _ _ => ⟨Nat.le_refl 0, Nat.le_refl 0⟩⟩
  keep := fun _ _ i => i.elim0

/-- `eraseWins`, decided conjunct by conjunct; the last over the list of answers, not over every value of an answer
(the instance search's choice, `Fintype.decidableForallFintype`, would run through all `2^64` blocks). -/
def decideWins {M : IdealModel} (sch : Scheme M) {S S_φ D Q : ℕ} {K V : Type} [DecidableEq V]
    (dig : K → sch.Code → V) (s : ℕ) (W : Bits sch.n) (P : EraseProver sch S S_φ D Q K V)
    (x : EraseOutcome sch S_φ K s) : Decidable (eraseWins sch dig s W P x) := by
  dsimp only [eraseWins]
  exact @instDecidableAnd _ _ inferInstance (@instDecidableAnd _ _ inferInstance (List.decidableBAll _ _))

/-- What the verifier takes from an answer: its value, if it came within the deadline `T`. -/
def inTime {α : Type} (T : ℕ) (answered : Bool) (time : ℕ) (v : α) : Option α :=
  if answered && decide (time ≤ T) then some v else none

structure Case where
  Tfill : ℕ
  Tread : ℕ
  s : ℕ
  m : ℕ
  pageBytes : ℕ
  key : F × F
  digest : Option (List ℕ)
  fill : Array ℕ
  indices : Array ℕ
  reads : Array (Bits 64)

/-- `eraseWins` on the case's outcome. -/
def accepts (c : Case) : Bool :=
  let x : EraseOutcome (scheme c.m c.fill) 0 (F × F) c.s :=
    (fun i => i.elim0, c.key, (), (), fun j => ⟨c.indices.getD j 0 % (c.m + 1), Nat.mod_lt _ (Nat.succ_pos _)⟩)
  @decide _ (decideWins (scheme c.m c.fill) (dig (B := c.m + 1) c.pageBytes) c.s (fun i => i.elim0)
    (prover c.m c.fill c.digest c.reads) x)

def quads : List ℕ → Option (List (ℕ × ℕ × ℕ × ℕ))
  | [] => some []
  | i :: a :: v :: t :: rest => ((i, a, v, t) :: ·) <$> quads rest
  | _ => none

def parse (line : String) : Option Case := do
  let ns ← ((line.splitOn " ").filter (· ≠ "")).mapM String.toNat?
  match ns with
  | tf :: tr :: s :: size :: pb :: g :: z :: fa :: ft :: n :: rest =>
    let digest := rest.take n
    let fill := (rest.drop n).take size
    let reads ← quads ((rest.drop n).drop size)
    guard (1 ≤ size ∧ 1 ≤ pb ∧ digest.length = n ∧ fill.length = size ∧ reads.length = s ∧ fa ≤ 1)
    guard (g < 2 ^ 127 - 1 ∧ z < 2 ^ 127 - 1 ∧ digest.all (· < 256) ∧ fill.all (· < 256))
    guard (reads.all fun (i, a, v, _) => i < size ∧ a ≤ 1 ∧ v + 1 < 2 ^ 64)
    pure {
      Tfill := tf, Tread := tr, s, m := size - 1, pageBytes := pb, key := ((g : F), (z : F))
      digest := inTime tf (fa == 1) ft digest
      fill := fill.toArray
      indices := (reads.map (·.1)).toArray
      reads := (reads.map fun (_, a, v, t) => ((inTime tr (a == 1) t v).map block).getD late).toArray }
  | _ => none

partial def loop (h : IO.FS.Stream) (acc : String) : IO String := do
  let line ← h.getLine
  if line.isEmpty then
    return acc
  let line := line.trimAsciiEnd.toString
  if line.isEmpty then
    loop h acc
  else
    match parse line with
    | some c => loop h (acc.push (if accepts c then '1' else '0'))
    | none => throw <| IO.userError s!"not a case: {line}"

end PousErasureCheck

def main : IO Unit := do
  IO.println (← PousErasureCheck.loop (← IO.getStdin) "")
