/-!
# The certifier program: the lead's calls

The network certifier as the lead's coin source and per-forward count: the definition
`verity.core.protocols.certifier.calls` mirrors, held to it by the difftest (`NetworkCertifier.Calls.CallsDifftest`).
Lean core only.

* **A call's shape** (`Shape`): `rows` rows of A, `cols` output columns and inner dimension `K`, committed over whole MMA
  tiles (`Shape.m`, `Shape.n`) in `Shape.S` k32 steps, with `Shape.H` row words a row and `Shape.tiles` coin tiles, as
  the lead's `padRows`, `padCols` and `stepsFor` have them. The certifier takes only a shape the verifier admits
  (`Shape.valid`, the lead's `Layout.Admits`).
* **The flip** (`flip`): the call's coins from the certifier's entropy, a list of bytes, read in the release's order:
  the row words' bytes, each row's bits from `S + 3` up cleared (`rowBytes`); G's 32-bit little-endian words, each below
  `gBound` taken as its residue mod 7⁴ and each other rejected (`words`); the coin lanes' bytes, each one's low 7 bits a
  `u`.
* **The release** (`release`): the bytes of the coin feed the lead's kernel reads, 32-bit words little endian: the row
  words (bit `q < 4` of a row's is its step-1 bit a₁, bit `t + 3` its tag bit at a step `t ≥ 1`; step 0 reads none),
  the G words (byte `q` of column `j`'s the int8 `G[q, j]`), then a coin tile at a time its lane-coin words
  (`laneBytes`: three int8 coin lanes, and byte 3 the column code's top digit at step 0 and 1 later) and its column
  sources (`source`), which are no coins but a public function of them.
* **The count** (`step`): the window's forwards as the certifier keeps them. A call's A root is answered with the call's
  release, once; a forward's first call opens it, while no other forward is open. A bucket of a boundary link at
  allocation `r` counts `timingBits r` and its real frames' content bits once, against the open forward. A forward's
  words root closes it, its release's bits (`releaseBits`) counted, and refuses it if its count passes its calls' bits
  summed (`Cap.passed`); the window's close refuses the forward still open. Once the entropy fails the window flips
  nothing more, and the forward whose call it failed is refused.
-/

namespace NetworkCertifier.Calls

/-- Rows a coin tile. -/
def tile : Nat := 64

/-- G's rejection bound, `7⁴ · ⌊2³² / 7⁴⌋`: a 32-bit word at or above it is rejected. -/
def gBound : Nat := 7 ^ 4 * (2 ^ 32 / 7 ^ 4)

/-- The most MMA tiles a call's committed words may span over its steps for the verifier to admit it. -/
def admittedTiles : Nat := 2 ^ 40

/-- A column source's offset, `1.5 · 2²³` as an fp32's bits. -/
def sourceBias : Int := 0x4B400000

/-- Row class `c`'s weight on coin lane `ℓ` (the lead's `rotWeight`): 127 on lane `c` and 1 on the others. -/
def rotW (c ℓ : Nat) : Int := if ℓ = c then 127 else 1

/-- A call's registered shape. -/
structure Shape where
  rows : Nat
  cols : Nat
  K : Nat
  deriving DecidableEq, Repr

namespace Shape

/-- Rows its words commit, over whole MMA tiles of 16 (`padRows`). -/
def m (s : Shape) : Nat := 16 * ((s.rows + 15) / 16)

/-- Columns its words commit, over whole MMA tiles of 8 (`padCols`). -/
def n (s : Shape) : Nat := 8 * ((s.cols + 7) / 8)

/-- k32 steps a word: step 0 holds 21 data lanes and each later step 28 (`stepsFor`). -/
def S (s : Shape) : Nat := 1 + (s.K - 21 + 27) / 28

/-- Row words a row: `⌈(S + 3) / 32⌉`, for its four step-1 bits and `S − 1` tag bits. -/
def H (s : Shape) : Nat := (s.S + 3 + 31) / 32

/-- Coin tiles: `⌈m / 64⌉`. -/
def tiles (s : Shape) : Nat := (s.m + tile - 1) / tile

/-- Coins the call takes: `S + 3` bits a row, four tag weights a column and `3·S` coin lanes a tile and column. -/
def coins (s : Shape) : Nat := s.m * (s.S + 3) + 4 * s.n + 3 * s.S * s.tiles * s.n

/-- A shape the certifier takes: every dimension positive, and its committed words at most `admittedTiles` MMA tiles over
its steps. -/
def valid (s : Shape) : Bool :=
  decide (0 < s.rows ∧ 0 < s.cols ∧ 0 < s.K ∧ s.m / 16 * (s.n / 8) * s.S ≤ admittedTiles)

/-- Its MACs a token, the share of the egress cap it brings its forward: `rows · n · K` (the lead's `callBits`). -/
def macs (s : Shape) : Nat := s.rows * s.n * s.K

end Shape

/-- A call's coins as the certifier flips them, in the release's bytes: `rows`, the row words'; `g`, the G words'; `u`,
the coin lanes' `u` in `[0, 128)`, `u[((k·S + t)·n + j)·3 + ℓ]` at tile `k`, step `t`, column `j` and lane `ℓ`. -/
structure CallCoins where
  rows : List Nat
  g : List Nat
  u : List Nat
  deriving DecidableEq, Repr

/-- The first `k` bytes of the entropy and the rest, or none when it holds fewer. -/
def take? (k : Nat) (e : List Nat) : Option (List Nat × List Nat) :=
  if k ≤ e.length then some (e.take k, e.drop k) else none

/-- The row words' bytes from as many bytes of entropy: byte `i` keeps those of its bits that are below its row's `S + 3`
(a row is `4H` bytes). -/
def rowBytes (s : Shape) (raw : List Nat) : List Nat :=
  raw.mapIdx fun i x => x % 2 ^ min 8 (s.S + 3 - 8 * (i % (4 * s.H)))

/-- G's words: the entropy read as 32-bit little-endian words, each below `gBound` taken as its residue mod `7⁴` and each
other rejected, until `k` more are taken (after `acc`, the taken ones reversed); with the entropy left, or none when it
ends first. -/
def words : Nat → List Nat → List Nat → Option (List Nat × List Nat)
  | 0, e, acc => some (acc.reverse, e)
  | k + 1, a :: b :: c :: d :: e, acc =>
    let w := a + 256 * (b + 256 * (c + 256 * d))
    if w < gBound then words k e (w % 7 ^ 4 :: acc) else words (k + 1) e acc
  | _ + 1, _, _ => none

/-- An int8's byte, two's complement: `x mod 256`. -/
def int8 (x : Int) : Nat := (x % 256).toNat

/-- The G words' bytes: column `j`'s byte `q` is the int8 of word `j`'s base-7 digit `q`, less 3. -/
def gBytes (ws : List Nat) : List Nat :=
  ws.flatMap fun w => (List.range 4).map fun q => int8 ((w / 7 ^ q % 7 : Nat) - 3)

/-- **The flip**: the call's coins from the entropy, and the entropy left, or none when it ends first. -/
def flip (s : Shape) (e : List Nat) : Option (CallCoins × List Nat) := do
  let (raw, e) ← take? (4 * s.m * s.H) e
  let (ws, e) ← words s.n e []
  let (ub, e) ← take? (3 * s.S * s.tiles * s.n) e
  pure (⟨rowBytes s raw, gBytes ws, ub.map (· % 128)⟩, e)

/-- A 32-bit word's four bytes, little endian. -/
def u32 (w : Nat) : List Nat := [w % 256, w / 2 ^ 8 % 256, w / 2 ^ 16 % 256, w / 2 ^ 24 % 256]

/-- Digit `q` of column `j`'s code, in `[−6, 6]`: the balanced base-13 digits of `j + 1` (the lead's `colCode`). -/
def colCode (j q : Nat) : Int := ((j + 14281) / 13 ^ q % 13 : Nat) - 6

/-- A tile's lane-coin words' bytes from its `u`, from word `w` on (word `t·n + j` is step `t`, column `j`): bytes 0–2
the int8 coin lanes `2u − 127`, byte 3 the column code's top digit at step 0 and 1 at a later step. -/
def laneBytes (n : Nat) : Nat → List Nat → List Nat
  | w, a :: b :: c :: rest =>
    [int8 (2 * (a : Int) - 127), int8 (2 * (b : Int) - 127), int8 (2 * (c : Int) - 127),
      if w < n then int8 (colCode w 3) else 1] ++ laneBytes n (w + 1) rest
  | _, _ => []

/-- Column `j`'s source for row class `c` in a tile of `S` steps whose coin lanes' `u` are `u`: `0x4B400000 − σ mod 2³²`,
with `σ = Σ_t Σ_ℓ rotW c ℓ · (2u − 127)`. -/
def source (S n : Nat) (u : Array Nat) (c j : Nat) : Nat :=
  let σ := (List.range S).foldl (fun acc t => (List.range 3).foldl (fun acc ℓ =>
    acc + rotW c ℓ * (2 * (u[(t * n + j) * 3 + ℓ]! : Int) - 127)) acc) 0
  ((sourceBias - σ) % 2 ^ 32).toNat

/-- **The release**: the call's row words and G words, then a coin tile at a time its lane-coin words and its column
sources (class 0's `n`, then class 1's). -/
def release (s : Shape) (c : CallCoins) : List Nat :=
  c.rows ++ c.g ++ (List.range s.tiles).flatMap fun k =>
    let u := (c.u.drop (k * (3 * s.S * s.n))).take (3 * s.S * s.n)
    let ua := u.toArray
    laneBytes s.n 0 u ++
      (List.range 2).flatMap fun cl => (List.range s.n).flatMap fun j => u32 (source s.S s.n ua cl j)

/-- A boundary link's bucket's timing bits at allocation `r`: `⌈log2 (r + 1)⌉`, the bit length of `r`. -/
def timingBits (r : Nat) : Nat := if r = 0 then 0 else Nat.log2 r + 1

/-- The egress cap the window's workload registers: `cap` bits a token over the forward's `fwdMacs` MACs a token. -/
structure Cap where
  cap : Nat
  fwdMacs : Nat
  deriving DecidableEq, Repr

/-- Whether `count` bits pass the bits of calls of `macs` MACs (`Shape.macs` summed), `cap · macs / fwdMacs`, compared in
integers. -/
def Cap.passed (c : Cap) (macs count : Nat) : Bool :=
  c.cap * macs < count * c.fwdMacs

/-- A forward's verdict in the record. -/
inductive ForwardVerdict
  | accepted
  | pastBits
  | unclosed
  | noCoins
  deriving DecidableEq, Repr

/-- A verdict's name in the record. -/
def ForwardVerdict.name : ForwardVerdict → String
  | .accepted => "accepted"
  | .pastBits => "past its bits"
  | .unclosed => "open when the window closed"
  | .noCoins => "a call's coins were not flipped"

/-- A call as the certifier keeps it: its id, shape and A root. -/
structure Call where
  id : Nat
  shape : Shape
  aRoot : List Nat
  deriving Repr

/-- A forward as the certifier keeps it: its id, its calls in id order, its count, its words root and its verdict. -/
structure Forward where
  id : Nat
  calls : List Call
  count : Nat
  wordsRoot : Option (List Nat)
  verdict : Option ForwardVerdict
  deriving Repr

/-- A forward refused for `v`, unless it was refused already. -/
def Forward.refuse (v : ForwardVerdict) (f : Forward) : Forward := { f with verdict := some (f.verdict.getD v) }

/-- What reaches the certifier: the prover's registrations on the control channel, each boundary link's bucket at its
tick (its allocation and its real frames' content bits), and the window's close. -/
inductive Event
  | aRoot (forward call : Nat) (root : List Nat) (shape : Shape)
  | bucket (r : Nat) (frames : List Nat)
  | wordsRoot (forward : Nat) (root : List Nat)
  | close

/-- What the certifier answers: an A root's release, a counted bucket, a words root's verdict, every forward's verdict
at the close, a refused event, or a call whose coins weren't flipped. -/
inductive CallAnswer
  | release (bytes : List Nat)
  | counted
  | verdict (v : ForwardVerdict)
  | verdicts (vs : List (Nat × ForwardVerdict))
  | refusal
  | noCoins

/-- The window's forwards, in id order, the open one, the entropy left, and whether the window is closed. -/
structure State where
  forwards : List Forward
  isOpen : Option Nat
  entropy : List Nat
  closed : Bool

/-- A window's start, with its entropy. -/
def State.init (e : List Nat) : State := ⟨[], none, e, false⟩

/-- `x` in its place by `id`. -/
def insertById {α : Type} (id : α → Nat) (x : α) : List α → List α
  | [] => [x]
  | y :: ys => if id x < id y then x :: y :: ys else y :: insertById id x ys

/-- The state with forward `k` changed by `g`. -/
def State.update (st : State) (k : Nat) (g : Forward → Forward) : State :=
  { st with forwards := st.forwards.map fun f => if f.id == k then g f else f }

/-- **The certifier's step** on one event, with the egress cap `cap` and the node's release at a forward's end,
`releaseBits`. -/
def step (cap : Cap) (releaseBits : Nat) (st : State) : Event → State × CallAnswer
  | .aRoot forward call root shape =>
    let isNew := !st.forwards.any (·.id == forward)
    let f := (st.forwards.find? (·.id == forward)).getD ⟨forward, [], 0, none, none⟩
    if st.closed || !shape.valid || root.length != 64 ||
        !(st.isOpen == some forward || (st.isOpen.isNone && isNew)) || f.calls.any (·.id == call) then
      (st, .refusal) else
    let f := { f with calls := insertById Call.id ⟨call, shape, root⟩ f.calls }
    let opened := { st with forwards := insertById Forward.id f (st.forwards.filter (·.id != forward)),
                            isOpen := some forward }
    match flip shape st.entropy with
    | some (c, e) => ({ opened with entropy := e }, .release (release shape c))
    | none => ({ opened.update forward (Forward.refuse .noCoins) with entropy := [] }, .noCoins)
  | .bucket r frames =>
    if st.closed || decide (r < frames.length) then (st, .refusal) else
    match st.isOpen with
    | some k => (st.update k fun f => { f with count := f.count + timingBits r + frames.sum }, .counted)
    | none => (st, .counted)
  | .wordsRoot forward root =>
    if st.closed || st.isOpen != some forward || root.length != 64 then (st, .refusal) else
    match st.forwards.find? (·.id == forward) with
    | none => (st, .refusal)
    | some f =>
      let count := f.count + releaseBits
      let v := f.verdict.getD
        (if cap.passed (f.calls.map (·.shape.macs)).sum count then .pastBits else .accepted)
      ({ st.update forward (fun f => { f with count, wordsRoot := some root, verdict := some v }) with isOpen := none },
        .verdict v)
  | .close =>
    if st.closed then (st, .refusal) else
    let st := match st.isOpen with
      | some k => st.update k (Forward.refuse .unclosed)
      | none => st
    ({ st with isOpen := none, closed := true },
      .verdicts (st.forwards.map fun f => (f.id, f.verdict.getD .unclosed)))

/-- The window's answers to `evs` in order, and its state after them. -/
def run (cap : Cap) (releaseBits : Nat) (st : State) : List Event → List CallAnswer × State
  | [] => ([], st)
  | ev :: evs =>
    let (st', a) := step cap releaseBits st ev
    let (as, st'') := run cap releaseBits st' evs
    (a :: as, st'')

end NetworkCertifier.Calls
