module

public import Poub.Line

/-!
# γ's terms

`audit.ledger`'s terms for an accepted session's record against a capacity R, in exact rationals, as
`Verifier.ledgers` declares them (`audit.GammaTerms`): each a fraction of R over the session's span, and over each
window's own span `[b_(t−1), D_t]` (`WindowGamma`). The record credits its registered blocks only, and charges no host, link
or carrier term: no other form is computed here.

* timing, `1 − credited/whole`: each pass's credited bytes are its registered receipts' (its length less its post-seed
  reads) times the block's bytes, and `whole` is `R·span/10^9`;
* sample, the window's incorrect receipts at δ (`incorrectReceipts`, `Stratified.count_bound` of one uniform sample of its
  `drawn` receipts of its U) times the block's bytes;
* chip, S_chip a pass; counting, AuditRO's bits a pass, `namedBits k + SLACK_LOG2` (`audit.named_bits`);
* rebuild, the squaring ceiling's bytes a second, `squarings·BLOCK_BYTES/encode_squarings`, the codec's encode;
* pooled (the ledger's `horizon`) and floor, the pooled bound (`Pooled.pooled` over the record's samples, g windows a
  sample): W incorrect receipts' bytes over the session, and each window's share at most min(1, f·U_g/U_t) of its
  receipts.

The errors are every window's δ (the pooled bound's one δ) plus AuditRO's 2^-`SLACK_LOG2` a pass.

`namedBits k` is k·`QUERIES_LOG2` − max(0, ⌊log₂ k! · (1 − 2^-40)⌋ − 1), with log₂ k! bracketed: exactly from k! up to
`EXACT_FACTORIAL`, by Stirling's series with Robbins' bounds above it, its logarithms in fixed point rounded outward
(`log2Iv`), ln 2 and π from series. A k whose bracket leaves the integer open is refused. The exact value is at most
log₂ k!, so the charge is at least log₂ C(2^`QUERIES_LOG2`, k). `ledger.py`'s floats are held to this
(`session/verifier.py`'s `gamma_answer`, `tests/test_gamma.py`). Code: Lean core only.
-/

@[expose] public section

namespace Poub.Gamma

/-- `audit.QUERIES_LOG2`: AuditRO's queries a pass, 2^80. -/
def QUERIES_LOG2 : Nat := 80

/-- `audit.SLACK_LOG2`: AuditRO's slack a pass. -/
def SLACK_LOG2 : Nat := 140

/-- Fractional bits of `log2Iv`'s brackets. -/
def LOG_BITS : Nat := 160

/-- Bits the mantissa is held to while it is squared. -/
def FIX_BITS : Nat := 256

/-- Up to this k, log₂ k! is bracketed from k! itself. -/
def EXACT_FACTORIAL : Nat := 1024

/-- A closed interval of rationals, `lo ≤ hi`. -/
structure Iv where
  lo : Rat
  hi : Rat

def pow2 (e : Int) : Rat :=
  if 0 ≤ e then ((2 ^ e.toNat : Nat) : Rat) else 1 / ((2 ^ (-e).toNat : Nat) : Rat)

def floorQ (q : Rat) : Int := q.num.fdiv q.den

/-- ⌊log₂ x⌋ for a positive rational. -/
def ilog2 (x : Rat) : Int :=
  let e : Int := (Nat.log2 x.num.toNat : Int) - (Nat.log2 x.den : Int)
  if pow2 e ≤ x then e else e - 1

/-- The bits of log₂ m for a mantissa held between `lo` and `hi` (scaled by 2^`FIX_BITS`, 1 ≤ m ≤ 2): square, and
take a bit where the square reaches 2, halving it. `acc + 2^-i·log₂ m` is the same after each step, and rounding the
lower held value down and the upper up moves it only down or up: `(a, b)` with `a/2^n ≤ log₂ m ≤ (b + 1)/2^n`. -/
def logBits (n : Nat) (lo hi : Nat) : Nat × Nat := Id.run do
  let one := 2 ^ FIX_BITS
  let two := 2 * one
  let mut l := lo
  let mut h := hi
  let mut a := 0
  let mut b := 0
  for _ in [0:n] do
    l := l * l / one
    h := (h * h + one - 1) / one
    if two ≤ l then
      a := 2 * a + 1
      l := l / 2
    else
      a := 2 * a
    if two ≤ h then
      b := 2 * b + 1
      h := (h + 1) / 2
    else
      b := 2 * b
  return (a, b)

/-- A bracket of log₂ x for a positive rational x, `LOG_BITS` wide. -/
def log2Iv (x : Rat) : Iv :=
  let e := ilog2 x
  let m := x / pow2 e
  let n := m.num.toNat * 2 ^ FIX_BITS
  let (a, b) := logBits LOG_BITS (n / m.den) ((n + m.den - 1) / m.den)
  let s := ((2 ^ LOG_BITS : Nat) : Rat)
  ⟨(e : Rat) + (a : Rat) / s, (e : Rat) + ((b + 1 : Nat) : Rat) / s⟩

/-- ln 2 = Σ_{j ≥ 1} 1/(j·2^j): the first 200 terms, the rest at most 1/(201·2^200). -/
def ln2 : Iv :=
  let s := (List.range 200).foldl (fun acc j => acc + 1 / (((j + 1) * 2 ^ (j + 1) : Nat) : Rat)) 0
  ⟨s, s + 1 / ((201 * 2 ^ 200 : Nat) : Rat)⟩

/-- log₂ e = 1/ln 2. -/
def log2e : Iv := ⟨1 / ln2.hi, 1 / ln2.lo⟩

/-- arctan(1/x) = Σ_j (−1)^j/((2j+1)·x^(2j+1)), alternating and falling: through term 2n it is above, and less term
2n + 1 below. -/
def atanInv (x n : Nat) : Iv :=
  let t : Nat → Rat := fun j => 1 / (((2 * j + 1) * x ^ (2 * j + 1) : Nat) : Rat)
  let s := (List.range (2 * n + 1)).foldl (fun acc j => if j % 2 = 0 then acc + t j else acc - t j) 0
  ⟨s - t (2 * n + 1), s⟩

/-- π = 16·arctan(1/5) − 4·arctan(1/239) (Machin). -/
def pi : Iv :=
  let a := atanInv 5 60
  let b := atanInv 239 20
  ⟨16 * a.lo - 4 * b.hi, 16 * a.hi - 4 * b.lo⟩

/-- ½·log₂(2π). -/
def halfLog2TwoPi : Iv := ⟨(1 + (log2Iv pi.lo).lo) / 2, (1 + (log2Iv pi.hi).hi) / 2⟩

def fact : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * fact n

/-- log₂ k! for k ≥ 1 (Stirling, with Robbins' 1/(12k + 1) < r_k < 1/(12k)):
(k + ½)·log₂ k − k·log₂ e + ½·log₂(2π) + r_k·log₂ e. -/
def stirling (k : Nat) : Iv :=
  let l := log2Iv k
  let c : Rat := (k : Rat) + 1 / 2
  let h := halfLog2TwoPi
  ⟨c * l.lo - (k : Rat) * log2e.hi + h.lo + log2e.lo / ((12 * k + 1 : Nat) : Rat),
   c * l.hi - (k : Rat) * log2e.lo + h.hi + log2e.hi / ((12 * k : Nat) : Rat)⟩

/-- A bracket of log₂ k!. -/
def log2Fact (k : Nat) : Iv := if k ≤ EXACT_FACTORIAL then log2Iv (fact k) else stirling k

/-- `audit.named_bits k`: k·`QUERIES_LOG2` − max(0, ⌊log₂ k! · (1 − 2^-40)⌋ − 1), refused where the bracket of log₂ k!
doesn't decide it. -/
def namedBits (k : Nat) : Except String Nat := do
  let x := log2Fact k
  let c : Rat := 1 - 1 / ((2 ^ 40 : Nat) : Rat)
  let q : Int := (k * QUERIES_LOG2 : Nat)
  let least := q - max 0 (floorQ (x.hi * c) - 1)
  let most := q - max 0 (floorQ (x.lo * c) - 1)
  if least != most then
    throw s!"the named bits of a pass of {k} receipts are not decided at {LOG_BITS} bits of log₂ {k}!"
  pure least.toNat

/-- `C(n − m, k)/C(n, k) ≥ δ`: m of n units incorrect all miss a uniform sample of k with probability at least δ. -/
def misses (n k m : Nat) (δ : Rat) : Bool :=
  let (a, b) := (List.range k).foldl (fun (a, b) i => (a * (n - m - i), b * (n - i))) (1, 1)
  decide (δ.num * (b : Int) ≤ (a : Int) * δ.den)

/-- `Stratified.count_bound` of one uniform sample of `k` of `n` receipts: the largest m ≤ n with
`C(n − m, k)/C(n, k) ≥ δ`, none past the n − k a positive miss probability allows (all n at k = 0), by bisection. -/
def incorrectReceipts (n k : Nat) (δ : Rat) : Nat :=
  let finite := if k = 0 then n else n - k
  let rec go : Nat → Nat → Nat → Nat
    | 0, lo, _ => lo
    | fuel + 1, lo, hi =>
      if hi ≤ lo + 1 then lo
      else
        let mid := (lo + hi) / 2
        if decide (mid ≤ finite) && misses n k mid δ then go fuel mid hi else go fuel lo mid
  go (Nat.log2 (n + 1) + 2) 0 (n + 1)

/-- One γ's terms, each a fraction of the capacity over a span. -/
structure Terms where
  timing : Rat
  sample : Rat
  chip : Rat
  counting : Rat
  rebuild : Rat
  host : Rat
  link : Rat
  carrier : Rat
  pooled : Rat
  floor : Rat
  deriving Inhabited

/-- γ with each window's sample at δ, the terms' sum. -/
def Terms.gamma (t : Terms) : Rat :=
  t.timing + t.sample + t.chip + t.counting + t.rebuild + t.host + t.link + t.carrier

/-- γ with the pooled sample. -/
def Terms.gammaPooled (t : Terms) : Rat := t.gamma - t.sample + t.pooled

/-- Window `window`'s terms over its own span. -/
structure WindowGamma where
  window : Nat
  spanNs : Nat
  terms : Terms
  deriving Inhabited

/-- `audit.GammaTerms` for the credit this module computes. -/
structure Gamma where
  terms : Terms
  windows : Array WindowGamma
  error : Rat
  errorPooled : Rat

/-- The first window of the largest γ (`by`), the windows nonempty. -/
def worst (ws : Array WindowGamma) (by_ : Terms → Rat) : WindowGamma :=
  ws.foldl (fun w x => if by_ w.terms < by_ x.terms then x else w) ws[0]!

/-- A block's encode in squarings (`Codec.encode_squarings`): k roots a word, each the bit length of (p + 1)/4 less one. -/
def encodeSquarings (c : Codec.Codec) : Nat := c.field.blockWords * c.rounds * Nat.log2 ((c.field.p + 1) / 4)

def ns : Rat := 1000000000

/-- γ's terms of record `sm` of registration `reg` (its digest `digest`), sampled `g` windows a coin, at δ, blocks of
`blockBytes`, S_chip `chipBytes`, against capacity R (bytes a second) and a squaring ceiling at the codec's width
(squarings a second); or why there are none, word for word `audit.ledger`'s. -/
def gamma (reg : Registration) (digest : ByteArray) (sm : Session.Summary) (g : Nat) (δ : Rat)
    (blockBytes chipBytes : Nat) (capacity squaringsPerS : Rat) : Except String Gamma := do
  if let some why := Line.unregistered reg digest sm then throw why
  if sm.windows.size = 0 then throw "a session has one or more windows"
  if g = 0 then throw "a sample covers one or more windows"
  if capacity ≤ 0 then throw "the capacity is positive and the chip's storage nonnegative"
  if squaringsPerS < 0 then throw "a squaring ceiling is a finite nonnegative rate"
  let ws := sm.windows
  let b := blockBytes
  let span := ws[ws.size - 1]!.deadlineNs - ws[0]!.releasedNs
  let whole := capacity * (span : Rat) / ns
  let us := ws.map fun w => w.lengths.foldl (· + ·) 0
  let receipts := us.foldl (· + ·) 0
  if whole < ((receipts * b : Nat) : Rat) then
    throw s!"the record certifies {receipts * b} bytes read, more than the capacity allows in its span: the capacity is understated"
  let groups := (List.range ((ws.size + g - 1) / g)).toArray.map fun a =>
    let ts := List.range' (a * g) (min g (ws.size - a * g))
    (ts.foldl (fun n t => n + us[t]!) 0, ts.foldl (fun n t => n + ws[t]!.drawn) 0)
  let pooled ← Pooled.pooled groups δ
  let mut named : Array (Nat × Nat) := #[]
  for w in ws do
    for k in w.lengths do
      if !(named.any (·.1 == k)) then named := named.push (k, (← namedBits k) + SLACK_LOG2)
  let bits : Nat → Nat := fun k => ((named.find? (·.1 == k)).map (·.2)).getD 0
  let rebuild := squaringsPerS * ((Codec.BLOCK_BYTES : Nat) : Rat) / (((encodeSquarings reg.codec : Nat) : Rat) * capacity)
  let mut out : Array WindowGamma := #[]
  let mut wrong := 0
  let mut credited := 0
  let mut counted := 0
  let mut passes := 0
  for t in [0:ws.size] do
    let w := ws[t]!
    let spanT := w.deadlineNs - w.releasedNs
    let wholeT := capacity * (spanT : Rat) / ns
    let readsT := us[t]! * b
    if wholeT < (readsT : Rat) then
      throw s!"window {t} certifies {readsT} bytes read, more than the capacity allows in its span: the capacity is understated"
    let e := incorrectReceipts us[t]! w.drawn δ
    let cr := ((w.lengths.zip w.post).foldl (fun n (l, p) => n + (l - p)) 0) * b
    let c := w.lengths.foldl (fun n k => n + bits k) 0
    let f := min 1 (pooled.floors[t / g]! * (groups[t / g]!.1 : Rat) / (us[t]! : Rat))
    out := out.push
      { window := t, spanNs := spanT,
        terms := { timing := 1 - (cr : Rat) / wholeT, sample := ((e * b : Nat) : Rat) / wholeT,
                   chip := ((w.lengths.size * chipBytes : Nat) : Rat) / wholeT, counting := (c : Rat) / 8 / wholeT,
                   rebuild, host := 0, link := 0, carrier := 0, pooled := f * (readsT : Rat) / wholeT, floor := f } }
    wrong := wrong + e
    credited := credited + cr
    counted := counted + c
    passes := passes + w.lengths.size
  let beside := (passes : Rat) / ((2 ^ SLACK_LOG2 : Nat) : Rat)
  pure
    { terms := { timing := 1 - (credited : Rat) / whole, sample := ((wrong * b : Nat) : Rat) / whole,
                 chip := ((passes * chipBytes : Nat) : Rat) / whole, counting := (counted : Rat) / 8 / whole,
                 rebuild, host := 0, link := 0, carrier := 0, pooled := pooled.wrong * (b : Rat) / whole,
                 floor := pooled.floor },
      windows := out, error := (ws.size : Rat) * δ + beside, errorPooled := δ + beside }

end Poub.Gamma
