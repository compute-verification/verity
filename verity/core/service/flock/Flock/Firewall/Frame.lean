/-
# The outer frame (live PROTOCOL.md §10.1): what leaves a masked rep

The outer firewall's release function. `round` takes one device round as the device frames it (transcript-v2, its values
unmasked, ending in its squeeze) and the rep's masking state, and returns the round that leaves for the verifier's stream
with the state after it: every label, message and squeeze as the device framed it, the pads' label and root right after
the zerocheck's label, and each value absorbed in a masked phase sent as `v + h_i` (GF(2^128) addition) under the next pad
of the layout's absorbed order (`absorbed`). The masked phases are the algebraic phase, from the zerocheck's label to the
opening's, and, with a masked Ligerito level 0, the values between `Zk.LABEL_OPEN` and `Zk.LABEL_CLOSE`. The domain frame
is the stream's own and never leaves.

`flock-firewall frame` serves `round` to the firewall (`cmd`), which releases only what it returns; Rust's
`zk_veil::Masking` is its speed reference, held to it on every round the firewall passes.
-/
import Flock.Zk
import Flock.ZkLigerito

namespace Flock.Firewall.Frame

open Lean (Json)
open Transcript (header padded)

def LABEL_ZC : String := "flock-zerocheck-v0"
def LABEL_PCS : String := "flock-pcs-open-batch-v0"
/-- The univariate skip's evaluations per sent vector (`zk_veil::SKIP`). -/
def SKIP : Nat := 64

/-- How many of the absorbed values belong to a masked Ligerito level 0 (none when `k0 = 0`: level 0 is not masked). -/
def nLig (l : Zk.PadLayout) : Nat := if l.k0 > 0 then l.nOod0 + 4 * l.k0 + 4 else 0

/-- The pads in the order the prover absorbs the values they mask (`zk_veil::Layout::absorbed`): the zerocheck's
`round1_ab`, `round1_c`, multilinear rounds, `final_a`, `final_b` (`final_c` is never absorbed), the lincheck's rounds and
`z_partial`, then level 0's OOD values, lane-round messages and the pair's sums. -/
def absorbed (l : Zk.PadLayout) : Array Nat := Id.run do
  let mut v : Array Nat := #[]
  for i in [0:SKIP] do v := v.push (l.r1ab i)
  for i in [0:SKIP] do v := v.push (l.r1c i)
  for j in [0:l.nMlv] do v := (v.push (l.ml j 0)).push (l.ml j 1)
  v := (v.push l.fa).push l.fb
  for j in [0:l.nLc] do v := (v.push (l.lc j 0)).push (l.lc j 1)
  for i in [0:SKIP] do v := v.push (l.zp i)
  if l.k0 > 0 then
    for i in [0:l.nOod0] do v := v.push (l.lgOod i)
    for j in [0:l.k0] do
      for t in [0:4] do v := v.push (l.lgMsg j t)
    v := v ++ #[l.lgE 0 0, l.lgE 0 1, l.lgE 1 0, l.lgE 1 1]
  return v

/-- A rep's masking state: its pads `h`, the absorbed order, the pads' root, and where the rep's stream is. -/
structure Masker where
  h : Array F128
  order : Array Nat
  /-- how many of `order` the algebraic phase absorbs -/
  nPiop : Nat
  lig : Bool
  root : ByteArray
  /-- values absorbed masked so far -/
  pos : Nat := 0
  on : Bool := false
  done : Bool := false
  doneL0 : Bool := false
  l0 : Bool := false
  deriving Inhabited

/-- The state at a rep's start, for layout `l`, pads `h` and the pads' 64-byte root. -/
def Masker.new (l : Zk.PadLayout) (h : Array F128) (root : ByteArray) : V Masker := do
  if root.size != 64 then throw s!"the pads' root has {root.size} bytes, not 64"
  let order := absorbed l
  if let some i := order.find? (· ≥ h.size) then throw s!"pad {i} is outside the rep's {h.size} pads"
  return { h, order, nPiop := order.size - nLig l, lig := l.k0 > 0, root }

def labelFrame (b : ByteArray) : ByteArray := header 2 0 b.size ++ padded b

def bytesFrame (b : ByteArray) : ByteArray := header 5 0 b.size ++ padded b

def isLabel (b : ByteArray) (l : String) : Bool := b.data == l.toUTF8.data

/-- A label as the masking challenger takes it: the phase it ends, its frame, and the phase it starts (after the
zerocheck's label, the pads' label and root). -/
def Masker.label (s : Masker) (l : ByteArray) : V (Masker × ByteArray) := do
  let mut s := s
  if isLabel l LABEL_PCS && s.on then
    if s.pos != s.nPiop then
      throw s!"the opening began after {s.pos} of the algebraic phase's {s.nPiop} masked values"
    s := { s with on := false, done := true }
  if isLabel l Zk.LABEL_CLOSE && s.l0 then
    if s.pos != s.order.size then
      throw s!"level 0 closed after {s.pos} of the layout's {s.order.size} masked values"
    s := { s with l0 := false, on := false, doneL0 := true }
  let mut out := labelFrame l
  if isLabel l LABEL_ZC && !s.done then
    s := { s with on := true }
    out := out ++ labelFrame Zk.LABEL_PADS.toUTF8 ++ bytesFrame s.root
  if isLabel l Zk.LABEL_OPEN && s.done && s.lig && !s.doneL0 then
    s := { s with on := true, l0 := true }
  return (s, out)

/-- An absorbed value: `v + h_i` under the next pad of the order in a masked phase, `v` outside one. -/
def Masker.absorb (s : Masker) (v : F128) : V (Masker × F128) := do
  if !s.on then return (s, v)
  let some i := s.order[s.pos]? | throw s!"more masked values than the layout's {s.order.size}"
  let some p := s.h[i]? | throw s!"pad {i} is outside the rep's pads"
  return ({ s with pos := s.pos + 1 }, v + p)

/-- Whether the rep's stream ended with every masked phase closed and every pad of the order used. -/
def Masker.finished (s : Masker) : Bool := s.done && (s.doneL0 || !s.lig) && s.pos == s.order.size

/-- One device round `content` that squeezes `n` coins: the round that leaves, and the state after it; or why the round
is not a device round (a frame that overruns it, an unknown op, a squeeze before its end or of other than `n` coins, a
round without its squeeze, more masked values than the layout's, a masked phase closed early). -/
def round (s : Masker) (content : ByteArray) (n : Nat) : V (ByteArray × Masker) := do
  let mut s := s
  let mut out := ByteArray.empty
  let mut pos := 0
  for _ in [0:content.size + 1] do
    if pos + 16 > content.size then
      if pos == content.size then throw "a round without its squeeze"
      throw "a truncated frame header"
    let op := content.get! pos
    let kind := content.get! (pos + 1)
    let len := (u64OfLE content (pos + 8)).toNat
    pos := pos + 16
    let pad := (len + 15) / 16 * 16
    let body := fun (w : Nat) => if pos + w > content.size then throw "a frame overruns its round" else
      pure (content.extract pos (pos + w))
    match op.toNat with
    | 1 =>
      let _ ← body pad
      pos := pos + pad
    | 2 =>
      let (s', f) ← s.label (← body len)
      s := s'
      out := out ++ f
      pos := pos + pad
    | 5 =>
      out := out ++ bytesFrame (← body len)
      pos := pos + pad
    | 3 =>
      let b ← body (16 * len)
      if !(kind == 2 || (kind == 1 && len == 1)) then throw s!"an observe of kind {kind} and length {len}"
      out := out ++ header 3 kind len
      for i in [0:len] do
        let (s', v) ← s.absorb ⟨u64OfLE b (16 * i), u64OfLE b (16 * i + 8)⟩
        s := s'
        out := F128.pushBytes out v
      pos := pos + 16 * len
    | 4 =>
      if len == 0 && kind == 2 then
        out := out ++ header 4 2 0
      else
        if pos != content.size then throw "a squeeze before the round's end"
        if len != n then throw s!"the round squeezes {len} coins, the caller asked for {n}"
        if !(kind == 2 || (kind == 1 && n == 1)) then throw s!"a squeeze of kind {kind} and length {n}"
        return (out ++ header 4 kind n, s)
    | x => throw s!"unknown frame op {x}"
  throw "a round without its squeeze"

/-- The rep's start, from `flock-firewall frame`'s first line: `{"layout": {"n_mlv", "n_lc", "n_ood0", "k0"}, "pads":
[hex], "root": hex}`, each pad 32 hex digits (lo then hi, little-endian, as the frames carry them). -/
def Masker.ofJson (j : Json) : V Masker := do
  let lay ← j.getObjVal? "layout"
  let nat := fun (k : String) => (lay.getObjValAs? Nat k)
  let l : Zk.PadLayout := ⟨← nat "n_mlv", ← nat "n_lc", ← nat "n_ood0", ← nat "k0"⟩
  let pads ← (← j.getObjValAs? (Array String) "pads").mapM fun x => do
    let b ← unhex x
    if b.size != 16 then throw s!"a pad of {b.size} bytes"
    pure (⟨u64OfLE b 0, u64OfLE b 8⟩ : F128)
  Masker.new l pads (← unhex (← j.getObjValAs? String "root"))

/-- `n` bytes of `h`. Each read that returns nothing ends the input, so `n` reads that each return a byte or more
suffice. -/
def readExact (h : IO.FS.Stream) (n : Nat) : IO ByteArray := go n .empty
where
  go : Nat → ByteArray → IO ByteArray
    | 0, acc =>
      if acc.size ≥ n then pure acc
      else throw (IO.userError s!"frame: {acc.size} bytes after {n} reads of a {n}-byte round")
    | fuel + 1, acc => do
      if acc.size ≥ n then return acc
      let b ← h.read (n - acc.size).toUSize
      if b.isEmpty then throw (IO.userError s!"frame: input closed {acc.size} bytes into a {n}-byte round")
      go fuel (acc ++ b)

/-- `flock-firewall frame`: one rep's release, served on stdin and stdout. The first line starts the rep
(`Masker.ofJson`) and is answered `ok`; then `round N LEN` followed by the device round's `LEN` bytes is answered
`ok LEN'` followed by the `LEN'` bytes that leave, and `finish` is answered `ok` when every masked phase closed with every
pad used. Anything else, or a round `round` refuses, is answered `err WHY` and ends the rep: nothing after it leaves. -/
def cmd (_ : List String) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let refuse := fun (why : String) => do
    stdout.putStrLn s!"err {why.replace "\n" " "}"
    stdout.flush
    return (1 : UInt32)
  let first ← stdin.getLine
  let mut s ← match Json.parse first >>= Masker.ofJson with
    | .ok s => pure s
    | .error e => return (← refuse s!"the rep's start: {e}")
  stdout.putStrLn "ok"
  stdout.flush
  repeat
    let line := (← stdin.getLine).replace "\n" ""
    if line.isEmpty then break
    match line.splitOn " " with
    | ["round", n, len] =>
      let (some n, some len) := (n.toNat?, len.toNat?) | return (← refuse s!"a request `{line}`")
      let content ← readExact stdin len
      match round s content n with
      | .ok (out, s') =>
        s := s'
        stdout.putStrLn s!"ok {out.size}"
        stdout.write out
        stdout.flush
      | .error e => return (← refuse e)
    | ["finish"] =>
      if s.finished then
        stdout.putStrLn "ok"
        stdout.flush
      else return (← refuse s!"the rep ended after {s.pos} of the layout's {s.order.size} masked values")
    | _ => return (← refuse s!"a request `{line}`")
  return 0

end Flock.Firewall.Frame
