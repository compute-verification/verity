/-
The sha512x3 slot's circuit as a Lean builder (`verity_flock.sha512_circuit`, over `verity.ml.boolean.forms`): GF(2) linear
forms over committed bits, rows `out = a · b`, and the gadgets that build SHA-512's compression from them. Definitions only,
on core Lean and `Flock.Hash` (the round constants), so the executable verifier can run the same builder; what the rows
force is proved in `Gadget`, `Word` and `Compress`.
-/
import Flock.Hash
import Flock.Hm96

namespace FlockSoundness.Discharge.Hm

/-- A committed bit's index. -/
abbrev Var := Nat

/-- A GF(2) linear form: the XOR of the committed bits `s` (ascending, each once: `forms.L`'s frozenset) and the
constant `c`. -/
structure Lin where
  s : List Var
  c : Bool
  deriving DecidableEq, Inhabited

/-- The symmetric difference of two ascending lists, ascending. -/
def symmDiff : List Var → List Var → List Var
  | [], t => t
  | a :: s, [] => a :: s
  | a :: s, b :: t =>
    if a < b then a :: symmDiff s (b :: t)
    else if b < a then b :: symmDiff (a :: s) t
    else symmDiff s t
termination_by s t => s.length + t.length

namespace Lin

def zero : Lin := ⟨[], false⟩
def one : Lin := ⟨[], true⟩
def var (v : Var) : Lin := ⟨[v], false⟩
def const (b : Bool) : Lin := ⟨[], b⟩

/-- XOR of two forms (`forms.L.__xor__`). -/
def xor (f g : Lin) : Lin := ⟨symmDiff f.s g.s, f.c ^^ g.c⟩

instance : HXor Lin Lin Lin := ⟨xor⟩

def isConst (f : Lin) : Bool := f.s.isEmpty

end Lin

/-- A committed row: bit `out` is the product of the forms `a` and `b`. -/
structure Row where
  out : Var
  a : Lin
  b : Lin

/-- The builder's state: the next committed bit and the rows so far (newest first). -/
structure St where
  next : Nat
  rows : List Row

/-- The circuit builder (`forms.Circuit`). -/
abbrev M := StateM St

/-- A new committed bit, the product of `a` and `b`. -/
def fresh (a b : Lin) : M Lin := fun s => (Lin.var s.next, ⟨s.next + 1, ⟨s.next, a, b⟩ :: s.rows⟩)

/-- `n` committed input bits (no rows). -/
def input (n : Nat) : M (List Lin) := fun s => ((List.range n).map fun i => Lin.var (s.next + i), ⟨s.next + n, s.rows⟩)

/-- `forms.Circuit.AND` (no common-subexpression table): constants and equal supports simplify, else a new bit. -/
def and (a b : Lin) : M Lin :=
  if a.isConst then pure (if a.c then b else Lin.zero)
  else if b.isConst then pure (if b.c then a else Lin.zero)
  else if a.s = b.s then pure (if a.c = b.c then a else Lin.zero)
  else fresh a b

/-- `forms.maj`: the majority with one AND, `z ⊕ (x ⊕ z)(y ⊕ z)`; two or more constants simplify. -/
def maj (x y z : Lin) : M Lin :=
  let consts := [x, y, z].filter (·.isConst)
  if 2 ≤ consts.length then
    let vars := [x, y, z].filter (fun w => !w.isConst)
    let s := (consts.map fun w => w.c.toNat).sum
    if s = 0 then pure Lin.zero
    else if s = 2 then pure Lin.one
    else pure (match vars with
      | v :: _ => v
      | [] => Lin.const (decide (2 ≤ ([x, y, z].map fun w => w.c.toNat).sum)))
  else do
    let r ← and (x ^^^ z) (y ^^^ z)
    pure (z ^^^ r)

/-- `sha512_circuit.commit` on one bit: a committed bit or a constant stays; any other form becomes a new bit,
the product of the form and 1. -/
def commit (f : Lin) : M Lin :=
  if (f.s.length = 1 ∧ f.c = false) ∨ f.isConst then pure f else fresh f Lin.one

/-! ## Words -/

/-- A 64-bit word of the circuit: its bits' forms, least significant first (`sha512_circuit`'s words). -/
abbrev Word := List Lin

/-- `sha512_circuit._rot`: `x[n:] + x[:n]`, the word rotated right by `n`. -/
def rot (w : Word) (n : Nat) : Word := w.drop n ++ w.take n

/-- `sha512_circuit._shr`: `x[n:] + [0] * n`, the word shifted right by `n`. -/
def shr (w : Word) (n : Nat) : Word := w.drop n ++ List.replicate n Lin.zero

/-- XOR of two words, bit by bit. -/
def xorW (a b : Word) : Word := List.zipWith (· ^^^ ·) a b

/-- `sha512_circuit._xor3`: `p ^ q ^ r` bit by bit. -/
def xor3 (a b c : Word) : Word := xorW (xorW a b) c

/-- `forms.const_bits value 64`. -/
def constW (k : BitVec 64) : Word := (List.range 64).map fun i => Lin.const (k.getLsbD i)

/-- `compress_gadget`'s `ch`: `g ⊕ AND(e, f ⊕ g)` bit by bit. -/
def chW (e f g : Word) : M Word :=
  (List.zip e (List.zip f g)).mapM fun x => do
    let r ← and x.1 (x.2.1 ^^^ x.2.2)
    pure (x.2.2 ^^^ r)

/-- `forms.maj` bit by bit. -/
def majW (a b c : Word) : M Word := (List.zip a (List.zip b c)).mapM fun x => maj x.1 x.2.1 x.2.2

/-- `sha512_circuit.commit`. -/
def commitW (w : Word) : M Word := w.mapM commit

/-! ## The ripple-carry adder -/

/-- `sha512_circuit.CARRY_EVERY`. -/
def CARRY_EVERY : Nat := 5

/-- The carry out of bit `i`, committed at every bit `i ≡ 4 (mod 5)`. -/
def carryCommit (i : Nat) (c : Lin) : M Lin :=
  if i % CARRY_EVERY = CARRY_EVERY - 1 then commit c else pure c

/-- `sha512_circuit._add2`'s loop from bit `i` with carry `c`: the sum bit `x ⊕ y ⊕ c`, then, before the last bit, the
carry `maj(x, y, c)`, committed at every bit `i ≡ 4 (mod 5)`. -/
def add2Go : Word → Word → Lin → Nat → M Word
  | x :: a, y :: b, c, i =>
    if a.isEmpty then pure [x ^^^ y ^^^ c]
    else do
      let c₁ ← maj x y c
      let c₂ ← carryCommit i c₁
      let r ← add2Go a b c₂ (i + 1)
      pure ((x ^^^ y ^^^ c) :: r)
  | _, _, _, _ => pure []

/-- `sha512_circuit._add2`: the sum of two words mod `2^64`, carry in 0. -/
def add2 (a b : Word) : M Word := add2Go a b Lin.zero 0

/-- `sha512_circuit._add`: `acc + x₁ + … + xₖ`, added left to right. -/
def addN : Word → List Word → M Word
  | acc, [] => pure acc
  | acc, x :: xs => do
    let acc ← add2 acc x
    addN acc xs

/-! ## The compression -/

/-- `Σ₀`, `Σ₁`, `σ₀`, `σ₁` as XOR forms (no rows). -/
def bsig0W (a : Word) : Word := xor3 (rot a 28) (rot a 34) (rot a 39)
def bsig1W (e : Word) : Word := xor3 (rot e 14) (rot e 18) (rot e 41)
def ssig0W (w : Word) : Word := xor3 (rot w 1) (rot w 8) (shr w 7)
def ssig1W (w : Word) : Word := xor3 (rot w 19) (rot w 61) (shr w 6)

/-- SHA-512's eight working variables. -/
structure Regs (α : Type) where
  a : α
  b : α
  c : α
  d : α
  e : α
  f : α
  g : α
  h : α

namespace Regs

def ofList {α : Type} [Inhabited α] (l : List α) : Regs α :=
  ⟨l[0]!, l[1]!, l[2]!, l[3]!, l[4]!, l[5]!, l[6]!, l[7]!⟩

def toList {α : Type} (r : Regs α) : List α := [r.a, r.b, r.c, r.d, r.e, r.f, r.g, r.h]

def map {α β : Type} (f : α → β) (r : Regs α) : Regs β := ⟨f r.a, f r.b, f r.c, f r.d, f r.e, f r.f, f r.g, f r.h⟩

end Regs

/-- Schedule word `w.length` from the words before it, committed: `commit(_add(w[t-16], σ₀ w[t-15], w[t-7], σ₁ w[t-2]))`. -/
def schedStep (w : List Word) : M (List Word) := do
  let t := w.length
  let x ← addN (w.getD (t - 16) []) [ssig0W (w.getD (t - 15) []), w.getD (t - 7) [], ssig1W (w.getD (t - 2) [])]
  let x ← commitW x
  pure (w ++ [x])

/-- `n` more schedule words. -/
def schedule : Nat → List Word → M (List Word)
  | 0, w => pure w
  | n + 1, w => do
    let w ← schedStep w
    schedule n w

/-- One round with constant `k` and schedule word `wt`; the new `e` and `a` are committed, `e` first. -/
def roundW (k : BitVec 64) (wt : Word) (r : Regs Word) : M (Regs Word) := do
  let ch ← chW r.e r.f r.g
  let t1 ← addN r.h [bsig1W r.e, ch, constW k, wt]
  let mj ← majW r.a r.b r.c
  let t2 ← add2 (bsig0W r.a) mj
  let e ← add2 r.d t1
  let e ← commitW e
  let a ← add2 t1 t2
  let a ← commitW a
  pure ⟨a, r.a, r.b, r.c, e, r.e, r.f, r.g⟩

/-- Round constant `t` (`Flock.Sha512.k`). -/
def kAt (t : Nat) : BitVec 64 := (Flock.Sha512.k[t]!).toBitVec

/-- Rounds `t, t+1, …, t+n-1` over the schedule `w`. -/
def rounds (w : List Word) : Nat → Nat → Regs Word → M (Regs Word)
  | 0, _, r => pure r
  | n + 1, t, r => do
    let r ← roundW (kAt t) (w.getD t []) r
    rounds w n (t + 1) r

/-- `sha512_circuit.compress_gadget`: the schedule, 80 rounds, the feed-forward. -/
def compress (cv msg : List Word) : M (List Word) := do
  let w ← schedule 64 msg
  let r ← rounds w 80 0 (Regs.ofList cv)
  (cv.zip r.toList).mapM fun x => add2 x.1 x.2

/-- `sha512_circuit._words`: `n` words of 64 bits. -/
def words (bits : List Lin) (n : Nat) : List Word := (List.range n).map fun i => (bits.drop (64 * i)).take 64

/-- `sha512_circuit.compression n`: the inputs `cv{i}` (512 bits) and `msg{i}` (1024 bits) for each `i`, then each
compression's output, its words concatenated. -/
def compression (n : Nat) : M (List (List Lin × List Lin) × List (List Lin)) := do
  let ins ← (List.range n).mapM fun _ => do
    let cv ← input 512
    let msg ← input 1024
    pure (cv, msg)
  let outs ← ins.mapM fun x => do
    let out ← compress (words x.1 8) (words x.2 16)
    pure out.flatten
  pure (ins, outs)

/-- The sha512x3 slot's circuit. -/
def compression3 : M (List (List Lin × List Lin) × List (List Lin)) := compression 3

/-! ## The hm96 slot -/

/-- `sha512_circuit.word_bit`: the word-order position of byte-string bit `t` (bit `t % 8` of byte `t / 8`; words
big-endian). -/
def wordBit (t : Nat) : Nat := 64 * (t / 64) + 56 - 8 * (t / 8 % 8) + t % 8

/-- The salt's 24 message words: word `i`'s bit `j` is the byte-string bit of `y` that `wordBit` sends to `64 i + j`. -/
def saltWords (y : List Lin) : List Word :=
  (List.range 24).map fun i => (List.range 64).map fun j => y.getD (64 * i + 8 * (7 - j / 8) + j % 8) Lin.zero

/-- Bit `i` of `b = x ⊕ M·y` from `x`'s bit: `M[i][j]` is key bit `i + j` (`Hm96.mask`), `j < 1536`, added in order. -/
def maskBit (key : ByteArray) (y : List Lin) (i : Nat) (x : Lin) : Lin :=
  (List.range 1536).foldl (fun acc j => if Flock.Hm96.bit key (i + j) then y.getD j Lin.zero ^^^ acc else acc) x

/-- The chaining value after `hm96-sha512/v1`'s salt prefix (one block). -/
def saltMid : List (BitVec 64) :=
  (List.range 8).map fun i => (Flock.Sha512.compress Flock.Sha512.iv Flock.Hm96.Default512.saltPrefix 0)[i]!.toBitVec

/-- The salt digest's second block after the salt's last 64 bytes: SHA-512's padding of a 320-byte message. -/
def saltTail : List (BitVec 64) := [0x8000000000000000#64, 0#64, 0#64, 0#64, 0#64, 0#64, 0#64, 2560#64]

/-- `sha512_circuit.hm96_row`: inputs `cv` (512 bits), `pad` (1024) and the salt `y` (1536, byte-string order); output
`b ‖ c` (1024 bits, byte-string order), `x = F(cv, pad)`, `c = SHA-512(salt_prefix ‖ y)` from the prefix's midstate,
`b = x ⊕ M(key)·y`. -/
def hm96Row (key : ByteArray) : M (List Lin × List Lin × List Lin × List Lin) := do
  let cv ← input 512
  let pad ← input 1024
  let y ← input 1536
  let x ← compress (words cv 8) (words pad 16)
  let yw := saltWords y
  let st ← compress (saltMid.map constW) (yw.take 16)
  let st ← compress st (yw.drop 16 ++ saltTail.map constW)
  let b := (List.range 512).map fun i => maskBit key y i (x.flatten.getD (wordBit i) Lin.zero)
  let c := (List.range 512).map fun t => st.flatten.getD (wordBit t) Lin.zero
  pure (cv, pad, y, b ++ c)

end FlockSoundness.Discharge.Hm
