import Flock.Field

/-!
# The coin server's record, in its own format

`rec_live`'s coin server writes its record of a session (`public.json`) with `json.dumps(…, indent=1)`, its rounds
first (`rec_live.record_bytes`). `recordW` writes those bytes, and `readRecord` is the record format's own total reader:
the rounds, in the fixed keys, their fixed order and the fixed indentation, and nothing past them. `recordFullW` and
`readFull` go on past the rounds to the link points: the record's kind and tops' format, the prover's Hello in its
fixed keys (`helloW`, `readHello`) and the link's level-0 tops and points (`objW`, `readObj`). `Record.msgs` is the
inner rounds before the last as the record shows them: round 0's from the link, each later one's from its round. It reads no general
JSON, so what reads the record back (`Proofs.Flock.Claims.Recursion.RecRecord`: `readRecord (recordW xs rest) =
some xs`) rests on no property of a JSON library. The verifier's other readers of the record (`RecVStar.innerCommits`)
parse it as JSON.

A round's stream name is written as `json.dumps` writes an ASCII string (`"` and `\` escaped), its numbers in decimal,
its tops, commit strings and coins in lowercase hex, a coin as `rec_live.f128_hex` writes it (`hi` then `lo`, 16 digits
each).
-/

namespace Flock.RecRecord

/-- A reader of the record's bytes: what it read, and the bytes after it. -/
abbrev P (α : Type) : Type := List UInt8 → Option (α × List UInt8)

/-- `d` spaces: the indentation at depth `d`. -/
def sp (d : Nat) : List UInt8 := List.replicate d 32

/-- The bytes after the literal `l`, where `s` starts with it. -/
def lit : List UInt8 → List UInt8 → Option (List UInt8)
  | [], s => some s
  | _ :: _, [] => none
  | a :: l, b :: s => if a = b then lit l s else none

/-! ## Hex -/

/-- A lowercase hex digit. -/
def hexDigit (n : Nat) : UInt8 := if n < 10 then UInt8.ofNat (48 + n) else UInt8.ofNat (87 + n)

/-- A lowercase hex digit's value. -/
def unhexDigit (c : UInt8) : Option Nat :=
  if 48 ≤ c.toNat ∧ c.toNat ≤ 57 then some (c.toNat - 48)
  else if 97 ≤ c.toNat ∧ c.toNat ≤ 102 then some (c.toNat - 87) else none

/-- A byte's two hex digits. -/
def hexB (b : UInt8) : List UInt8 := [hexDigit (b.toNat / 16), hexDigit (b.toNat % 16)]

/-- Bytes in hex, as Python's `bytes.hex` writes them. -/
def hexW (bs : List UInt8) : List UInt8 := bs.flatMap hexB

/-- The byte two hex digits say. -/
def unhexPair (a b : UInt8) : Option UInt8 := do
  let x ← unhexDigit a
  let y ← unhexDigit b
  pure (UInt8.ofNat (16 * x + y))

/-- The bytes of a hex string's digits, up to its closing quote. -/
def hexBytes : List UInt8 → Option (List UInt8 × List UInt8)
  | [] => none
  | [c] => if c = 34 then some ([], []) else none
  | a :: b :: s =>
    if a = 34 then some ([], b :: s) else do
      let x ← unhexPair a b
      let (r, s') ← hexBytes s
      pure (x :: r, s')

/-- A hex string. -/
def hexStr (bs : List UInt8) : List UInt8 := 34 :: hexW bs ++ [34]

/-- A hex string's bytes. -/
def readHex : P (List UInt8) := fun s => (lit [34] s).bind hexBytes

/-! ## The stream's name -/

/-- A byte of a string, escaped as `json.dumps` escapes `"` and `\`. -/
def esc (c : UInt8) : List UInt8 := if c = 34 ∨ c = 92 then [92, c] else [c]

/-- A string's bytes, escaped. -/
def escW (bs : List UInt8) : List UInt8 := bs.flatMap esc

/-- A string. -/
def strStr (bs : List UInt8) : List UInt8 := 34 :: escW bs ++ [34]

/-- An escaped string's bytes, up to its closing quote. -/
def unesc : List UInt8 → Option (List UInt8 × List UInt8)
  | [] => none
  | [c] => if c = 34 then some ([], []) else if c = 92 then none else some ([c], [])
  | a :: b :: s =>
    if a = 34 then some ([], b :: s)
    else if a = 92 then (unesc s).map fun (r, s') => (b :: r, s')
    else (unesc (b :: s)).map fun (r, s') => (a :: r, s')

/-- A string's bytes. -/
def readStr : P (List UInt8) := fun s => (lit [34] s).bind unesc

/-! ## Numbers -/

/-- The decimal digit of `n`'s last place. -/
def digit (n : Nat) : UInt8 := UInt8.ofNat (48 + n % 10)

/-- `n` in decimal. -/
def decW (n : Nat) : List UInt8 := if n < 10 then [digit n] else decW (n / 10) ++ [digit n]

/-- Whether a byte is a decimal digit. -/
def isDigit (c : UInt8) : Bool := 48 ≤ c.toNat && c.toNat ≤ 57

/-- The decimal digits at the start of a list, after the value `acc`, and the bytes after them. -/
def digits (acc : Nat) : List UInt8 → Nat × List UInt8
  | [] => (acc, [])
  | c :: s => if isDigit c then digits (acc * 10 + (c.toNat - 48)) s else (acc, c :: s)

/-- A number in decimal, of at least one digit. -/
def readNat : P Nat := fun s => match s with
  | c :: _ => if isDigit c then some (digits 0 s) else none
  | [] => none

/-! ## Lists -/

section Lists
variable {α : Type}

/-- A list's elements after its first, and its close, at indent `d`. -/
def tailW (d : Nat) (w : α → List UInt8) : List α → List UInt8
  | [] => 10 :: (sp d ++ [93])
  | e :: es => 44 :: 10 :: (sp (d + 1) ++ (w e ++ tailW d w es))

/-- A list at indent `d`, as `json.dumps(…, indent=1)` writes it: `[]` when empty. -/
def listW (d : Nat) (w : α → List UInt8) : List α → List UInt8
  | [] => [91, 93]
  | e :: es => 91 :: 10 :: (sp (d + 1) ++ (w e ++ tailW d w es))

/-- A list's elements from its first, and its close, at indent `d`: at most `fuel` of them. -/
def elems (d : Nat) (p : P α) : Nat → P (List α)
  | 0, _ => none
  | fuel + 1, s => (p s).bind fun (e, s) => match s with
    | c :: t =>
      if c = 44 then (lit (10 :: sp (d + 1)) t).bind fun t =>
        (elems d p fuel t).map fun (es, t) => (e :: es, t)
      else (lit (10 :: (sp d ++ [93])) s).map fun t => ([e], t)
    | [] => none

/-- A list at indent `d`, its elements read by `p`. -/
def readList (d : Nat) (p : P α) : P (List α) := fun s => (lit [91] s).bind fun s => match s with
  | c :: t => if c = 93 then some ([], t) else (lit (10 :: sp (d + 1)) s).bind fun s => elems d p s.length s
  | [] => none

end Lists

/-! ## Coins -/

/-- `v`'s `w` bytes, most significant first. -/
def be : Nat → Nat → List UInt8
  | 0, _ => []
  | w + 1, v => UInt8.ofNat (v / 256 ^ w) :: be w (v % 256 ^ w)

/-- The number bytes say, most significant first. -/
def unbe (bs : List UInt8) : Nat := bs.foldl (fun a x => a * 256 + x.toNat) 0

/-- A coin's 16 bytes: `hi`, then `lo`, each most significant first. -/
def coinB (c : F128) : List UInt8 := be 8 c.hi.toNat ++ be 8 c.lo.toNat

/-- A coin, as `rec_live.f128_hex` writes it. -/
def coinStr (c : F128) : List UInt8 := hexStr (coinB c)

/-- A coin of 32 hex digits. -/
def readCoin : P F128 := fun s => (readHex s).bind fun (bs, s) =>
  if bs.length = 16 then some (⟨UInt64.ofNat (unbe (bs.drop 8)), UInt64.ofNat (unbe (bs.take 8))⟩, s) else none

/-! ## Rounds -/

/-- **A round of the coin server's record**, as `rec_live` writes it: its stream's name, its index on the stream, its
salted tops (each a list of nodes), how many level-0 tops it absorbed, its commit strings, its coin count and the
coins it drew. -/
structure Round where
  stream : List UInt8
  k : Nat
  tops : List (List (List UInt8))
  level0 : Nat
  commits : List (List UInt8)
  n : Nat
  coins : List F128
  deriving DecidableEq

/-- An object's key, its colon and the space after it. -/
def key (s : String) : List UInt8 := 34 :: (s.toUTF8.toList ++ [34, 58, 32])

/-- A field after an object's first, at indent `d`. -/
def fld (d : Nat) (s : String) : List UInt8 := 44 :: 10 :: (sp d ++ key s)

/-- A round, at indent `2`. -/
def roundW (x : Round) : List UInt8 :=
  [123, 10] ++ sp 3 ++ key "stream" ++ strStr x.stream ++ fld 3 "k" ++ decW x.k ++ fld 3 "tops" ++
    listW 3 (listW 4 hexStr) x.tops ++ fld 3 "level0" ++ decW x.level0 ++ fld 3 "commits" ++
    listW 3 hexStr x.commits ++ fld 3 "n" ++ decW x.n ++ fld 3 "coins" ++ listW 3 coinStr x.coins ++
    [10] ++ sp 2 ++ [125]

/-- A round, at indent `2`. -/
def readRound : P Round := fun s => do
  let s ← lit (123 :: 10 :: (sp 3 ++ key "stream")) s
  let (stream, s) ← readStr s
  let s ← lit (fld 3 "k") s
  let (k, s) ← readNat s
  let s ← lit (fld 3 "tops") s
  let (tops, s) ← readList 3 (readList 4 readHex) s
  let s ← lit (fld 3 "level0") s
  let (level0, s) ← readNat s
  let s ← lit (fld 3 "commits") s
  let (commits, s) ← readList 3 readHex s
  let s ← lit (fld 3 "n") s
  let (n, s) ← readNat s
  let s ← lit (fld 3 "coins") s
  let (coins, s) ← readList 3 readCoin s
  let s ← lit (10 :: (sp 2 ++ [125])) s
  pure ({ stream, k, tops, level0, commits, n, coins }, s)

/-! ## The record -/

/-- **The coin server's record**, as `rec_live.record_bytes` writes it: its rounds first, then `rest`, the bytes of the
record's other fields and its close, which the reader doesn't read. -/
def recordW (xs : List Round) (rest : List UInt8) : List UInt8 :=
  [123, 10] ++ sp 1 ++ key "rounds" ++ listW 1 roundW xs ++ rest

/-- **The record's rounds**, read off its bytes in the record's own format. -/
def readRecord (s : List UInt8) : Option (List Round) :=
  ((lit (123 :: 10 :: (sp 1 ++ key "rounds")) s).bind (readList 1 readRound)).map Prod.fst

/-! ## Objects -/

section Objects
variable {α : Type}

/-- An object's entry: its key, a colon and a space, and its value. -/
def entW (w : α → List UInt8) (e : List UInt8 × α) : List UInt8 := strStr e.1 ++ [58, 32] ++ w e.2

/-- An object's entry. -/
def readEnt (p : P α) : P (List UInt8 × α) := fun s => (readStr s).bind fun (k, s) =>
  (lit [58, 32] s).bind fun s => (p s).map fun (v, s) => ((k, v), s)

/-- An object's entries after its first, and its close, at indent `d`. -/
def objTailW (d : Nat) (w : α → List UInt8) : List (List UInt8 × α) → List UInt8
  | [] => 10 :: (sp d ++ [125])
  | e :: es => 44 :: 10 :: (sp (d + 1) ++ (entW w e ++ objTailW d w es))

/-- An object at indent `d`, as `json.dumps(…, indent=1)` writes a dict: `{}` when empty. -/
def objW (d : Nat) (w : α → List UInt8) : List (List UInt8 × α) → List UInt8
  | [] => [123, 125]
  | e :: es => 123 :: 10 :: (sp (d + 1) ++ (entW w e ++ objTailW d w es))

/-- An object's entries from its first, and its close, at indent `d`: at most `fuel` of them. -/
def objElems (d : Nat) (p : P α) : Nat → P (List (List UInt8 × α))
  | 0, _ => none
  | fuel + 1, s => (readEnt p s).bind fun (e, s) => match s with
    | c :: t =>
      if c = 44 then (lit (10 :: sp (d + 1)) t).bind fun t =>
        (objElems d p fuel t).map fun (es, t) => (e :: es, t)
      else (lit (10 :: (sp d ++ [125])) s).map fun t => ([e], t)
    | [] => none

/-- An object at indent `d`, its values read by `p`, its entries in order. -/
def readObj (d : Nat) (p : P α) : P (List (List UInt8 × α)) := fun s => (lit [123] s).bind fun s => match s with
  | c :: t => if c = 125 then some ([], t) else (lit (10 :: sp (d + 1)) s).bind fun s => objElems d p s.length s
  | [] => none

end Objects

/-! ## The record past its rounds -/

/-- **The prover's Hello as the record holds it**, in its keys' order: its digests, flavor, the link's `m` and
points, profile, reps, rounds and tables. -/
structure Hello where
  digests : List UInt8
  flavor : List UInt8
  m : Nat
  points : Nat
  profile : List UInt8
  reps : Nat
  rounds : List UInt8
  tables : List (List UInt8)
  deriving DecidableEq

/-- Hello, at indent `1`. -/
def helloW (h : Hello) : List UInt8 :=
  [123, 10] ++ sp 2 ++ key "digests" ++ strStr h.digests ++ fld 2 "flavor" ++ strStr h.flavor ++ fld 2 "link" ++
    [123, 10] ++ sp 3 ++ key "m" ++ decW h.m ++ fld 3 "points" ++ decW h.points ++ [10] ++ sp 2 ++ [125] ++
    fld 2 "profile" ++ strStr h.profile ++ fld 2 "reps" ++ decW h.reps ++ fld 2 "rounds" ++ strStr h.rounds ++
    fld 2 "tables" ++ listW 2 strStr h.tables ++ [10] ++ sp 1 ++ [125]

/-- Hello, at indent `1`. -/
def readHello : P Hello := fun s => do
  let s ← lit (123 :: 10 :: (sp 2 ++ key "digests")) s
  let (digests, s) ← readStr s
  let s ← lit (fld 2 "flavor") s
  let (flavor, s) ← readStr s
  let s ← lit (fld 2 "link") s
  let s ← lit (123 :: 10 :: (sp 3 ++ key "m")) s
  let (m, s) ← readNat s
  let s ← lit (fld 3 "points") s
  let (points, s) ← readNat s
  let s ← lit (10 :: (sp 2 ++ [125])) s
  let s ← lit (fld 2 "profile") s
  let (profile, s) ← readStr s
  let s ← lit (fld 2 "reps") s
  let (reps, s) ← readNat s
  let s ← lit (fld 2 "rounds") s
  let (rounds, s) ← readStr s
  let s ← lit (fld 2 "tables") s
  let (tables, s) ← readList 2 readStr s
  let s ← lit (10 :: (sp 1 ++ [125])) s
  pure ({ digests, flavor, m, points, profile, reps, rounds, tables }, s)

/-- **The coin server's record up to its link points** (`rec_live.CoinServer.record`): its rounds, kind, tops'
format, the prover's Hello, and the link's level-0 tops (each table's nodes, `b ‖ c` each) and points. -/
structure Record where
  rounds : List Round
  kind : List UInt8
  tops : List UInt8
  hello : Hello
  roots : List (List UInt8 × List (List UInt8))
  points : List F128
  deriving DecidableEq

/-- **The coin server's record**, as `rec_live.record_bytes` writes it, up to its link points, then `rest`: the link's
`y`, the record's streams, counts and end, and its close, which the reader doesn't read. -/
def recordFullW (r : Record) (rest : List UInt8) : List UInt8 :=
  recordW r.rounds (fld 1 "kind" ++ strStr r.kind ++ fld 1 "tops" ++ strStr r.tops ++ fld 1 "hello" ++
    helloW r.hello ++ fld 1 "link" ++ [123, 10] ++ sp 2 ++ key "roots" ++ objW 2 (listW 3 hexStr) r.roots ++
    fld 2 "points" ++ listW 2 coinStr r.points ++ rest)

/-- **The record up to its link points**, read off its bytes in the record's own format. -/
def readFull (s : List UInt8) : Option Record := do
  let s ← lit (123 :: 10 :: (sp 1 ++ key "rounds")) s
  let (rounds, s) ← readList 1 readRound s
  let s ← lit (fld 1 "kind") s
  let (kind, s) ← readStr s
  let s ← lit (fld 1 "tops") s
  let (tops, s) ← readStr s
  let s ← lit (fld 1 "hello") s
  let (hello, s) ← readHello s
  let s ← lit (fld 1 "link") s
  let s ← lit (123 :: 10 :: (sp 2 ++ key "roots")) s
  let (roots, s) ← readObj 2 (readList 3 readHex) s
  let s ← lit (fld 2 "points") s
  let (points, _) ← readList 2 readCoin s
  pure { rounds, kind, tops, hello, roots, points }

/-- **The inner rounds before the last, as the record shows them**: round 0, the link, each table's level-0 top's
nodes, joined, and the link points; then each recorded round, its tops' nodes then its `c_k`, joined, and its coins.
The inner verifier's last round, the openings, is no round of the record. -/
def Record.msgs (r : Record) : List (List UInt8 × List F128) :=
  ((r.roots.map fun t => t.2.flatten).flatten, r.points) ::
    r.rounds.map fun x => ((x.tops.map List.flatten).flatten ++ x.commits.flatten, x.coins)

end Flock.RecRecord
