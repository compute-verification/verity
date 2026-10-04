/-!
# The secure-erasure accept rule, evaluated by Lean on the difftest's cases

Reads cases on stdin, one per line, as `tests/pous_erasure_cases.py` writes them:

    Tfill Tread s M expected fAnswered fDigest fTime r F_0 … F_(M−1) (i a v t)×r

and prints one verdict per case, `1` accept and `0` reject, on one line.  `tests/pous_erasure_cases.py` runs it
(`lean --run`, from `lean/` for the package's toolchain) and pins the verdicts in `fixtures/lean_erasure_verdicts.json`,
against which `test_pous_erasure_difftest.py` checks `verity_pous.erasure.ErasureVerifier`.

BINDING.  `erasureAccepts` below is the accept rule of `SecureErasureMeets` as the lanes' contract states it, written
here until `cursor/pous-lean-secure-erasure-3cf5` lands its definitions.  Then this file imports that module, `accepts`
calls that module's rule, and the fixture pins that module's text as well as this file's.
-/

namespace PousErasureCheck

/-- One step's answer as the verifier saw it: whether one came, its value, and its delay from the send. -/
structure Answer where
  answered : Bool
  value : Nat
  time : Nat

/-- A case: the deadlines, `s`, the fill `F` (`M` bytes), the expected digest, the fill's answer and the reads,
each an index and its answer. -/
structure Case where
  Tfill : Nat
  Tread : Nat
  s : Nat
  F : Array Nat
  expected : Nat
  fill : Answer
  reads : List (Nat × Answer)

/-- A step passes: answered, within its deadline `T`, with the expected value. -/
def passes (T expected : Nat) (a : Answer) : Bool :=
  a.answered && decide (a.time ≤ T) && a.value == expected

/-- The accept rule: the digest of the fill is `Digest_K(F_K)` within `Tfill`, and each of the `s` reads `(i, a)`
is `F i` within `Tread`. -/
def erasureAccepts (Tfill Tread s : Nat) (F : Nat → Nat) (expected : Nat) (fill : Answer)
    (reads : List (Nat × Answer)) : Bool :=
  passes Tfill expected fill && reads.length == s && reads.all fun (i, a) => passes Tread (F i) a

def accepts (c : Case) : Bool :=
  erasureAccepts c.Tfill c.Tread c.s (fun i => c.F.getD i 256) c.expected c.fill c.reads

def readsOf : Nat → List Nat → Option (List (Nat × Answer))
  | 0, [] => some []
  | r + 1, i :: a :: v :: t :: rest => ((i, ⟨a == 1, v, t⟩) :: ·) <$> readsOf r rest
  | _, _ => none

def parse (line : String) : Option Case := do
  let ns ← ((line.splitOn " ").filter (· ≠ "")).mapM String.toNat?
  match ns with
  | tf :: tr :: s :: m :: e :: fa :: fd :: ft :: r :: rest =>
    if rest.length < m then none else
    let reads ← readsOf r (rest.drop m)
    pure ⟨tf, tr, s, (rest.take m).toArray, e, ⟨fa == 1, fd, ft⟩, reads⟩
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
