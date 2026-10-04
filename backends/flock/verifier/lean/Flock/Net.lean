/-
PROTOCOL.md §16.5: an expanded circuit in `flock-ir-unit/v2` form (a unit, a tail stage, or a lookup slot built from its
table), and the text helpers its parser uses.
-/
import Lean.Data.Json
import Flock.Comp

namespace Flock

open Lean (Json)

def WORD : Nat := 128

def log2ceil (n : Nat) : Nat := if n ≤ 1 then 0 else Nat.log2 (n - 1) + 1
def isPow2 (n : Nat) : Bool := n > 0 && 2 ^ Nat.log2 n == n
def nextPow2 (n : Nat) : Nat := 2 ^ log2ceil n

def tokens (s : String) : Array String := Id.run do
  let mut out := #[]
  let mut cur := ""
  for c in s.toList do
    if c.isWhitespace then
      if cur != "" then
        out := out.push cur
        cur := ""
    else cur := cur.push c
  if cur != "" then out := out.push cur
  return out

/-- A port group of an expanded circuit: its first 128-bit word column, its word count (an aligned power of two) and its
ports' widths. -/
structure Group where
  col : Nat
  words : Nat
  bits : Array Nat
  deriving Inhabited, Repr

/-- Every port of a group list: its first column in the slot and its width, groups in order. -/
def portCols (gs : Array Group) : Array (Nat × Nat) := Id.run do
  let mut out := #[]
  for g in gs do
    let mut at_ := g.col * WORD
    for w in g.bits do
      out := out.push (at_, w)
      at_ := at_ + w
  return out

/-- A lookup slot's table and where its product rows sit (`Flock.Lookup`). -/
structure LookupTable where
  table : ByteArray
  n : Nat
  lo : Nat
  bits : Nat
  prod0 : Nat
  /-- the low decoder's top block: low minterm `l` is row `loTop + l` -/
  loTop : Nat
  deriving Inhabited

/-- An expanded circuit in `flock-ir-unit/v2` form, `2^unitLog` rows (rows past `useful` are empty). -/
structure Net where
  name : String
  useful : Nat
  constPos : Nat
  inGroups : Array Group
  outGroups : Array Group
  inWords : Nat
  a : Sparse
  b : Sparse
  unitLog : Nat
  /-- a lookup slot: what the fold reads from its table beyond the explicit rows -/
  lookup : Option LookupTable := none
  deriving Inhabited

namespace Net

/-- Topological order: every computed row (after the `inCols` input rows, before the constant) reads, in A and in B, only
earlier columns that are not input padding (an empty input row: a forced zero), or the constant. With the input rows and
the constant row, every column a row reads is then below `useful`. -/
def checkOrder (name : String) (inCols constPos : Nat) (ra rb : Array (Array Nat)) : V Unit := do
  let padding := fun (c : Nat) => c < inCols && (ra.getD c #[]).isEmpty && (rb.getD c #[]).isEmpty
  for r in [inCols:constPos] do
    for c in ra.getD r #[] ++ rb.getD r #[] do
      if c == constPos then continue
      if c ≥ r then throw s!"{name}: row {r} reads column {c}, not an earlier one"
      if padding c then throw s!"{name}: row {r} reads column {c}, an input padding column"

private def num (h : Array String) (i : Nat) : V Nat :=
  match h[i]?.bind String.toNat? with
  | some n => pure n
  | none => throw s!"flock-ir-unit/v2: header field {i} is not a number"

private def groups (h : Array String) (at0 : Nat) : V (Array Group × Nat) := do
  let count ← num h at0
  let mut at_ := at0 + 1
  let mut out := #[]
  for _ in [0:count] do
    let col ← num h at_
    let words ← num h (at_ + 1)
    let np ← num h (at_ + 2)
    let mut bits := #[]
    for i in [0:np] do bits := bits.push (← num h (at_ + 3 + i))
    if !(isPow2 words && col % words == 0) then throw "flock-ir-unit/v2: a port group is not an aligned power of two of words"
    out := out.push ⟨col, words, bits⟩
    at_ := at_ + 3 + np
  return (out, at_)

/-- A net's input groups ascend and are disjoint, and each input port bit's row is a self row, `[c]·[c]`: only the padding
past a group's ports may be empty. So each port bit is its own column, which one copy pair of Δ fills. -/
def inputsOk (inGroups : Array Group) (ra rb : Array (Array Nat)) : Bool :=
  (List.range (inGroups.size - 1)).all (fun i => (inGroups.getD i default).col + (inGroups.getD i default).words ≤
      (inGroups.getD (i + 1) default).col) &&
    inGroups.all fun g => (List.range (g.bits.foldl (· + ·) 0)).all fun j =>
      ra.getD (g.col * WORD + j) #[] == #[g.col * WORD + j] && rb.getD (g.col * WORD + j) #[] == #[g.col * WORD + j]

/-- An expanded circuit from its parts, as `parse` reads them from text or `Flock.Typed` has them from `derive`: its
useful rows' count, its constant, its port groups, and each useful row's A and B columns. -/
def ofRows (name : String) (useful constPos : Nat) (inGroups outGroups : Array Group) (ra rb : Array (Array Nat)) :
    V Net := do
  if (inGroups ++ outGroups).any (fun g => !(isPow2 g.words && g.col % g.words == 0)) then
    throw "flock-ir-unit/v2: a port group is not an aligned power of two of words"
  if outGroups.isEmpty then throw "flock-ir-unit/v2: no output group"
  if constPos + 1 != useful then throw "flock-ir-unit/v2: the constant is not the last useful column"
  let inWords := (inGroups.back?.map fun g => g.col + g.words).getD 0
  if inGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD || g.col + g.words > inWords)
      || outGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD) then
    throw "flock-ir-unit/v2: a port group's ports overrun its words, or an input group the input rows"
  if inWords * WORD > constPos then throw "flock-ir-unit/v2: the input rows reach the constant"
  let unitLog := max 7 (log2ceil useful)
  match (List.range (inWords * WORD)).find? fun i =>
      !((ra.getD i #[] == #[i] && rb.getD i #[] == #[i]) || ((ra.getD i #[]).isEmpty && (rb.getD i #[]).isEmpty)) with
  | some i => throw s!"flock-ir-unit/v2: row {i} is not an input row"
  | none => pure ()
  if ra.getD constPos #[] != #[constPos] || rb.getD constPos #[] != #[constPos] then
    throw "flock-ir-unit/v2: the constant row is not A = B = [const]"
  checkOrder "flock-ir-unit/v2" (inWords * WORD) constPos ra rb
  if !inputsOk inGroups ra rb then throw "flock-ir-unit/v2: the input groups overlap, or an input port bit has no self row"
  let a ← Sparse.ofRows ra (2 ^ unitLog)
  let b ← Sparse.ofRows rb (2 ^ unitLog)
  -- D11: an output group lies inside its unit's slot, so a region at its column (`Circuit.outRegion`) stays in the slot
  if let some g := outGroups.find? (fun g => g.col + g.words > 2 ^ unitLog / WORD) then
    throw s!"flock-ir-unit/v2: output group at word {g.col} ({g.words} words) overruns its unit's {2 ^ unitLog / WORD} words (D11)"
  return { name, useful, constPos, inGroups, outGroups, inWords, a, b, unitLog }

/-- A net `ofRows` builds holds exactly the rows it was given: its A and B are theirs at its slot size, and its size,
constant and port groups are the ones given. And the checks it ran hold: the input rows end before the constant, each is
a self row or empty, the constant row is `[const]·[const]`, and the row order (`checkOrder`) passed; it has no lookup; and
every output group lies inside the unit's slot (D11). -/
theorem ofRows_ok {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)} {n : Net}
    (h : ofRows name useful constPos ig og ra rb = .ok n) :
    n.unitLog = max 7 (log2ceil useful) ∧ Sparse.ofRows ra (2 ^ n.unitLog) = .ok n.a ∧
      Sparse.ofRows rb (2 ^ n.unitLog) = .ok n.b ∧ n.useful = useful ∧ n.constPos = constPos ∧ n.inGroups = ig ∧
      n.outGroups = og ∧ n.inWords * WORD ≤ constPos ∧ ra.getD constPos #[] = #[constPos] ∧
      rb.getD constPos #[] = #[constPos] ∧ checkOrder "flock-ir-unit/v2" (n.inWords * WORD) constPos ra rb = .ok () ∧
      (∀ i < n.inWords * WORD,
        ((ra.getD i #[] == #[i] && rb.getD i #[] == #[i]) || ((ra.getD i #[]).isEmpty && (rb.getD i #[]).isEmpty)) = true) ∧
      n.lookup = none ∧ n.inWords = (ig.back?.map fun g => g.col + g.words).getD 0 ∧
      ∀ g ∈ og.toList, g.col + g.words ≤ 2 ^ n.unitLog / WORD := by
  unfold ofRows at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' (first | split at h | (injection h with h; subst h))
  all_goals first
    | exact absurd h (by simp)
    | (rename_i _ _ _ _ hreach _ hfind hconst _ _ hord _ _ _ ha _ _ hb _ hd3
       simp only [Bool.or_eq_true, bne_iff_ne, ne_eq, not_or, Decidable.not_not] at hconst
       have hslot : ∀ g ∈ og.toList, g.col + g.words ≤ 2 ^ max 7 (log2ceil useful) / WORD := fun g hg =>
         Nat.not_lt.mp fun hc => by
           have hs : (og.find? fun g => decide (g.col + g.words > 2 ^ max 7 (log2ceil useful) / WORD)).isSome :=
             Array.find?_isSome.mpr ⟨g, Array.mem_toList_iff.mp hg, decide_eq_true hc⟩
           obtain ⟨g', hg'⟩ := Option.isSome_iff_exists.mp hs
           exact hd3 g' hg'
       refine ⟨rfl, ha, hb, rfl, rfl, rfl, rfl, Nat.not_lt.mp hreach, hconst.1, hconst.2, hord, fun i hi => ?_, rfl, rfl,
         hslot⟩
       have := List.find?_eq_none.mp hfind i (List.mem_range.mpr hi)
       revert this
       generalize ((ra.getD i #[] == #[i] && rb.getD i #[] == #[i]) || ((ra.getD i #[]).isEmpty && (rb.getD i #[]).isEmpty)) = b
       cases b <;> simp)

/-- A net `ofRows` builds passed `inputsOk`: its input groups ascend and are disjoint, and every input port bit's row is a
self row. -/
theorem ofRows_inputs {name : String} {useful constPos : Nat} {ig og : Array Group} {ra rb : Array (Array Nat)} {n : Net}
    (h : ofRows name useful constPos ig og ra rb = .ok n) :
    (∀ i (hi : i + 1 < ig.size), ig[i].col + ig[i].words ≤ ig[i + 1].col) ∧
      ∀ g ∈ ig.toList, ∀ j < g.bits.foldl (· + ·) 0,
        ra.getD (g.col * WORD + j) #[] = #[g.col * WORD + j] ∧ rb.getD (g.col * WORD + j) #[] = #[g.col * WORD + j] := by
  have hin : inputsOk ig ra rb = true := by
    unfold ofRows at h
    simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
    repeat' (first | split at h | (injection h with h; subst h))
    all_goals first
      | exact absurd h (by simp)
      | (rename_i _ _ _ _ _ _ _ _ _ _ _ h1 h2 h3 _ _ _ _ _ _
         first | simpa using h1 | simpa using h2 | simpa using h3)
  unfold inputsOk at hin
  simp only [Bool.and_eq_true, List.all_eq_true, List.mem_range, decide_eq_true_eq, Array.all_eq_true', beq_iff_eq] at hin
  refine ⟨fun i hi => ?_, fun g hg j hj => hin.2 g (Array.mem_toList_iff.1 hg) j hj⟩
  have := hin.1 i (by omega)
  simpa [Array.getD, hi, Nat.lt_of_succ_lt hi] using this

/-- Parse one expanded circuit (the lines of a `NET` block). -/
def parse (lines : Array String) : V Net := do
  let h := tokens (lines.getD 0 "")
  if h[0]? != some "flock-ir-unit/v2" then throw "not a flock-ir-unit/v2 circuit"
  let name := h.getD 1 ""
  let useful ← num h 2
  let constPos ← num h 3
  let (inGroups, at1) ← groups h 4
  let (outGroups, at2) ← groups h at1
  if at2 != h.size then throw "flock-ir-unit/v2: trailing header fields"
  if outGroups.isEmpty then throw "flock-ir-unit/v2: no output group"
  if constPos + 1 != useful then throw "flock-ir-unit/v2: the constant is not the last useful column"
  let inWords := (inGroups.back?.map fun g => g.col + g.words).getD 0
  if inGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD || g.col + g.words > inWords)
      || outGroups.any (fun g => g.bits.foldl (· + ·) 0 > g.words * WORD) then
    throw "flock-ir-unit/v2: a port group's ports overrun its words, or an input group the input rows"
  if inWords * WORD > constPos then throw "flock-ir-unit/v2: the input rows reach the constant"
  let mut ra := #[]
  let mut rb := #[]
  for i in [0:useful] do
    let v ← (tokens (lines.getD (1 + i) "")).mapM fun t => match t.toNat? with
      | some n => pure n
      | none => throw s!"flock-ir-unit/v2: row {i} is not numbers"
    let na := v.getD 0 0
    let nb := v.getD (1 + na) 0
    if v.size == 0 || v.size != 2 + na + nb then throw s!"flock-ir-unit/v2: row {i} is malformed"
    ra := ra.push (v.extract 1 (1 + na))
    rb := rb.push (v.extract (2 + na) (2 + na + nb))
  -- after the rows: an optional `LEAVES <json>` line (bound by the pin, not read), and nothing else
  let mut at_ := 1 + useful
  if (lines.getD at_ "").startsWith "LEAVES " then
    match Json.parse ((lines.getD at_ "").drop 7).toString with
    | .ok _ => at_ := at_ + 1
    | .error e => throw s!"flock-ir-unit/v2: the LEAVES line is not JSON: {e}"
  if (lines.getD at_ "").startsWith "CUT " then
    throw s!"{name}: a CUT line (a tail the verifier evaluates); the verifier evaluates no part of the computation"
  if at_ != lines.size then throw "flock-ir-unit/v2: text after the rows that is not its LEAVES line"
  ofRows name useful constPos inGroups outGroups ra rb

end Net

end Flock
