import Mathlib.Data.ZMod.Defs
import Pous.Protocol.Model.ReadDraw
import Pous.Protocol.Model.PageFill
import Pous.Protocol.Model.Digest

/-!
# The secure-erasure accept rule over distinct reads of the page fill, evaluated by Lean on the difftest's cases

Reads cases on stdin, one per line, as `tests/pous_erasure_cases.py` writes them:

    Tfill Tread Tround s M pageBytes g z fAnswered fTime n d_1 … d_n K_0 … K_31 R (T_0 … T_39 X_0 … X_(pageBytes−1))×R
      (i a v t r)×s

and prints one verdict per case, `1` accept and `0` reject, on one line.  `tests/pous_erasure_cases.py` runs it
(`lake env lean --run` in `lean/`, after `lake build` of the modules it imports) and pins the verdicts in
`fixtures/lean_erasure_verdicts.json`, against which `test_pous_erasure_difftest.py` checks
`verity_pous.erasure.ErasureVerifier`.

**The verdict is the model's.**  It is `Pous.eraseWinsDraw` with `Pous.distinctReads`
(`Pous/Protocol/Model/ReadDraw.lean`, the game `Pous.Guarantees.SecureErasurePagesFamily` and
`SecureErasurePages96112` are about) on one outcome of the secure-erasure game, with no foreign content (`S_φ = 0`, so
keeping it is free and `eraseWinsDraw` is the verifier's accept event):

- the fill is the page fill of `Pous/Protocol/Model/PageFill.lean` under the case's key `K`, computed by its own
  `PageFill.tweak`, `PageFill.page` and `PageFill.byte`, at the point `ω` of the ideal model where `π_T` for the
  tweak `T` Python gave page `q` swaps the zero page with Python's page `X` (Lean has no SHAKE256), and every other
  `π_T` is the identity.  Page `p` is then `π_{tweak K p}(0)`: Python's page `q` only if Python's tweak for it is
  Lean's `tweak K p`, else the zero page.  Byte `i` is byte `i mod pageBytes` of page `⌊i / pageBytes⌋`, which is
  `pageScheme`'s code when `M` is whole pages (`pageScheme_code`); a fill of part of a page is its first `M` bytes;
- the read indices are the embedding `Fin s ↪ Fin M` of the case's `i`, so a case whose indices repeat is refused;
- the digest `dig` is `Pous.Digest.digest g z` over `GF(2^127 − 1)` of the fill's pages, each `⌈pageBytes / 15⌉`
  15-byte little-endian words, zero-padded, written as 16 bytes little-endian: Lean computes the digest itself;
- the prover returns the digest bytes `d` for the fill and the value `v` of read `j` for round `j`.

A read's byte is held as a 64-bit block, the byte zero-extended, so that an answer no byte is can be told apart.

**The deadlines are not.**  Lean's game has no clock: its responders are `(D, Q)`-bounded, so they always answer in time
(`Pous/Protocol/Model/Audit.lean`), and its rounds follow one another.  The one rule written here is the verifier's
reading of a missed deadline: an answer that did not come within it (`a = 0`, or `t > T`), and one to a read revealed
more than `Tround` after the read before it (`r > Tround`, with `r` the time between the two reveals), is one that no
digest or block is (`none`, `late`).
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

/-- A little-endian byte string as bits: bit `j` is bit `j mod 8` of byte `⌊j / 8⌋`. -/
def bitsOf (n : ℕ) (bs : Array ℕ) : Bits n := fun j => (bs.getD (j.val / 8) 0).testBit (j.val % 8)

/-- A byte of the page fill as a block, zero-extended. -/
def widen (b : Bits 8) : Bits 64 := fun k => if h : k.val < 8 then b ⟨k.val, h⟩ else false

/-- The case's point of the page model: `π_T` swaps the zero page and page `X` for each `(T, X)` Python gave, and is
the identity at every other tweak. -/
def omega (pb : ℕ) (pages : List (PageFill.Tweak × PageFill.Page pb)) : PageFill.PageOmega pb := fun T =>
  match pages.find? (fun q => decide (q.1 = T)) with
  | some (_, x) => Equiv.swap (PageFill.zeroPage pb) x
  | none => Equiv.refl _

/-- With `M` whole pages, the fill's byte `i` as `code` computes it is `pageScheme`'s: byte `i mod pb` of page
`⌊i / pb⌋`. -/
theorem pageScheme_code (pb P : ℕ) [NeZero pb] [NeZero P] (K : Bits 256) (W : Bits 0) (ω : PageFill.PageOmega pb)
    (c : Unit) (i : Fin (P * pb)) :
    ((PageFill.pageScheme pb P K).enc W ω c).2 i =
      PageFill.byte (PageFill.page K ω (i.val / pb)) ⟨i.val % pb, Nat.mod_lt _ (NeZero.pos pb)⟩ := rfl

/-- The page Python gave for Lean's tweak `tweak K p`, if any. -/
def lookup (pb : ℕ) (pages : List (PageFill.Tweak × PageFill.Page pb)) (K : Bits 256) (p : ℕ) :
    Option (PageFill.Tweak × PageFill.Page pb) :=
  pages.find? fun q => decide (q.1 = PageFill.tweak K p)

def pageOf {pb : ℕ} : Option (PageFill.Tweak × PageFill.Page pb) → PageFill.Page pb
  | some (_, x) => x
  | none => PageFill.zeroPage pb

/-- Page `p` at the case's point of the model is the page `lookup` finds: `code` evaluates the right side, with the
lookup done once per page (a `Page` is a function, so the left side would redo it at every bit). -/
theorem page_omega (pb : ℕ) (pages : List (PageFill.Tweak × PageFill.Page pb)) (K : Bits 256) (p : ℕ) :
    PageFill.page K (omega pb pages) p = pageOf (lookup pb pages K p) := by
  unfold PageFill.page omega lookup
  generalize pages.find? (fun q => decide (q.1 = PageFill.tweak K p)) = o
  rcases o with _ | ⟨T, x⟩
  · rfl
  · exact Equiv.swap_apply_left _ _

/-- The case's fill, `B` bytes as blocks: byte `i` is `PageFill.byte (PageFill.page K (omega pb pages) ⌊i / pb⌋)
(i mod pb)` (`page_omega`). -/
def code (pb : ℕ) (hpb : 0 < pb) (K : Bits 256) (pages : List (PageFill.Tweak × PageFill.Page pb)) (R B : ℕ) :
    Array (Bits 64) :=
  let found := (Array.range R).map (lookup pb pages K)
  Array.ofFn (n := B) fun i =>
    widen (PageFill.byte (pageOf (found.getD (i.val / pb) none)) ⟨i.val % pb, Nat.mod_lt _ hpb⟩)

/-- The scheme of one case in the plain model: `m + 1` blocks, the case's fill. -/
def scheme (m : ℕ) (fill : Array (Bits 64)) : Scheme IdealModel.plain where
  n := 0
  t := 0
  B := m + 1
  ℓ := 64
  B_pos := Nat.succ_pos m
  Coins := Unit
  enc := fun _ _ _ => (fun i => i.elim0, fun i => fill.getD i late)
  dec := fun _ _ _ i => i.elim0

def pageCount (B pageBytes : ℕ) : ℕ := (B + pageBytes - 1) / pageBytes

def words (pageBytes : ℕ) : ℕ := (pageBytes + 14) / 15

/-- Word `w` of page `r`: bytes `15w … 15w + 14` of the page, little-endian, zero past the page or the fill. -/
def word {B : ℕ} (pageBytes : ℕ) (C : Fin B → Bits 64) (r w : ℕ) : ℕ :=
  (List.range 15).foldr (fun k acc =>
    let q := r * pageBytes + 15 * w + k
    (if h : 15 * w + k < pageBytes ∧ q < B then byte (C ⟨q, h.2⟩) else 0) + 256 * acc) 0

/-- The fill check's digest: `Pous.Digest.digest g z` of the pages, as 16 bytes little-endian. -/
def dig {B : ℕ} (pageBytes : ℕ) (κ : F × F) (C : Fin B → Bits 64) : Option (List ℕ) :=
  let xs : Fin (pageCount B pageBytes) → Digest.Blk (words pageBytes) F := fun r w => (word pageBytes C r w : F)
  let v := (Digest.digest κ.1 κ.2 xs).val
  some ((List.range 16).map fun k => v / 256 ^ k % 256)

/-- The prover of one case: its digest for the fill, `reads[j]` for round `j`, nothing kept. -/
def prover (m : ℕ) (fill : Array (Bits 64)) (digest : Option (List ℕ)) (reads : Array (Bits 64)) :
    EraseProver (scheme m fill) 0 0 0 0 (F × F) (Option (List ℕ)) where
  fill := fun _ _ _ _ _ _ => (digest, fun i => i.elim0)
  read := ⟨fun _ _ j σ _ => .ret (reads.getD j late, σ), fun _ _ _ _ _ _ => ⟨Nat.le_refl 0, Nat.le_refl 0⟩⟩
  keep := fun _ _ i => i.elim0

/-- `eraseWinsDraw`, decided conjunct by conjunct; the last over the list of answers, not over every value of an answer
(the instance search's choice, `Fintype.decidableForallFintype`, would run through all `2^64` blocks). -/
def decideWins {M : IdealModel} (sch : Scheme M) {S S_φ D Q : ℕ} {K V R : Type} [DecidableEq V] {s : ℕ}
    (rd : R → Fin s → Fin sch.B) (dig : K → sch.Code → V) (W : Bits sch.n) (P : EraseProver sch S S_φ D Q K V)
    (x : EraseOutcomeDraw sch S_φ K R) : Decidable (eraseWinsDraw sch rd dig W P x) := by
  dsimp only [eraseWinsDraw]
  exact @instDecidableAnd _ _ inferInstance (@instDecidableAnd _ _ inferInstance (List.decidableBAll _ _))

/-- What the verifier takes from an answer: its value, if it came within the deadline `T`. -/
def inTime {α : Type} (T : ℕ) (answered : Bool) (time : ℕ) (v : α) : Option α :=
  if answered && decide (time ≤ T) then some v else none

/-- The read indices as distinct reads: the embedding, if no index repeats. -/
def embed {s B : ℕ} (f : Fin s → Fin B) : Option (Fin s ↪ Fin B) :=
  if h : Function.Injective f then some ⟨f, h⟩ else none

structure Case where
  s : ℕ
  m : ℕ
  pageBytes : ℕ
  key : F × F
  digest : Option (List ℕ)
  fill : Array (Bits 64)
  indices : Array ℕ
  reads : Array (Bits 64)

/-- `eraseWinsDraw` with `distinctReads` on the case's outcome; none if its indices repeat. -/
def accepts (c : Case) : Option Bool := do
  let sch := scheme c.m c.fill
  let r ← embed (s := c.s) (B := c.m + 1) fun j =>
    ⟨c.indices.getD j 0 % (c.m + 1), Nat.mod_lt _ (Nat.succ_pos _)⟩
  let x : EraseOutcomeDraw sch 0 (F × F) (Fin c.s ↪ Fin sch.B) := (fun i => i.elim0, c.key, (), (), r)
  pure (@decide _ (decideWins sch (distinctReads sch c.s) (dig (B := c.m + 1) c.pageBytes) (fun i => i.elim0)
    (prover c.m c.fill c.digest c.reads) x))

def quints : List ℕ → Option (List (ℕ × ℕ × ℕ × ℕ × ℕ))
  | [] => some []
  | i :: a :: v :: t :: r :: rest => ((i, a, v, t, r) :: ·) <$> quints rest
  | _ => none

/-- Page `q`'s tweak and bytes from the line's `R · (40 + pb)` numbers. -/
def pagesOf (pb : ℕ) (raw : List ℕ) (R : ℕ) : List (PageFill.Tweak × PageFill.Page pb) :=
  (List.range R).map fun q =>
    let chunk := raw.drop (q * (40 + pb))
    (bitsOf 320 (chunk.take 40).toArray, bitsOf (pb * 8) ((chunk.drop 40).take pb).toArray)

def parse (line : String) : Option Case := do
  let ns ← ((line.splitOn " ").filter (· ≠ "")).mapM String.toNat?
  match ns with
  | tf :: tr :: tround :: s :: size :: pb :: g :: z :: fa :: ft :: n :: rest =>
    let digest := rest.take n
    let rest := rest.drop n
    let key := rest.take 32
    match rest.drop 32 with
    | R :: rest =>
      let raw := rest.take (R * (40 + pb))
      let reads ← quints (rest.drop (R * (40 + pb)))
      guard (1 ≤ size ∧ digest.length = n ∧ key.length = 32 ∧ R = pageCount size pb ∧ raw.length = R * (40 + pb))
      guard (reads.length = s ∧ fa ≤ 1)
      guard (g < 2 ^ 127 - 1 ∧ z < 2 ^ 127 - 1 ∧ digest.all (· < 256) ∧ key.all (· < 256) ∧ raw.all (· < 256))
      guard (reads.all fun (i, a, v, _, _) => i < size ∧ a ≤ 1 ∧ v + 1 < 2 ^ 64)
      if hpb : 0 < pb then
        pure {
          s, m := size - 1, pageBytes := pb, key := ((g : F), (z : F))
          digest := inTime tf (fa == 1) ft digest
          fill := code pb hpb (bitsOf 256 key.toArray) (pagesOf pb raw R) R size
          indices := (reads.map (·.1)).toArray
          reads := (reads.map fun (_, a, v, t, r) =>
            ((inTime tr (a == 1 && decide (r ≤ tround)) t v).map block).getD late).toArray }
      else none
    | _ => none
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
    | some c => match accepts c with
      | some v => loop h (acc.push (if v then '1' else '0'))
      | none => throw <| IO.userError s!"the read indices repeat: {line}"
    | none => throw <| IO.userError s!"not a case: {line}"

end PousErasureCheck

def main : IO Unit := do
  IO.println (← PousErasureCheck.loop (← IO.getStdin) "")
