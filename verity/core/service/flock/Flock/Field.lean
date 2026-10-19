/-
PROTOCOL.md §2.1, §2.2, §2.4: the fields.
-/
namespace Flock

/-- GF(2^128) = F2[x]/(x^128 + x^7 + x^2 + x + 1). Bit `i` of `hi·2^64 + lo` is the coefficient of `x^i`. -/
structure F128 where
  lo : UInt64
  hi : UInt64
  deriving BEq, Inhabited, Repr, DecidableEq, Hashable

namespace F128

def zero : F128 := ⟨0, 0⟩
def one : F128 := ⟨1, 0⟩

@[inline] def add (a b : F128) : F128 := ⟨a.lo ^^^ b.lo, a.hi ^^^ b.hi⟩

instance instAdd : Add F128 := ⟨add⟩
instance instOfNatOfNatNat : OfNat F128 0 := ⟨zero⟩
instance instOfNatOfNatNat_1 : OfNat F128 1 := ⟨one⟩

@[inline] def isZero (a : F128) : Bool := a.lo == 0 && a.hi == 0

/-- Bits `i, i + 1, …` of the carry-less product: for each set bit of `bb` (bit `i` of the multiplier) the 128-bit value
`a · x^i` is added into `(lo, hi)`. -/
def clmulGo (a : UInt64) : Nat → Nat → UInt64 → UInt64 → UInt64 → UInt64 × UInt64
  | 0, _, _, lo, hi => (lo, hi)
  | fuel + 1, i, bb, lo, hi =>
    if bb &&& 1 = 1 then
      clmulGo a fuel (i + 1) (bb >>> 1) (lo ^^^ (a <<< i.toUInt64)) (if i = 0 then hi else hi ^^^ (a >>> (64 - i).toUInt64))
    else clmulGo a fuel (i + 1) (bb >>> 1) lo hi

/-- Carry-less product of two 64-bit words, as (low word, high word). -/
def clmul (a b : UInt64) : UInt64 × UInt64 := clmulGo a 64 0 b 0 0

/-- The low word of `lo + a · x^i` if bit `t` of `bb` is set, else `lo`, without a branch (`i < 64`). -/
@[inline] def clmulStepLo (a bb t i lo : UInt64) : UInt64 :=
  lo ^^^ ((a <<< i) &&& ((0 : UInt64) - ((bb >>> t) &&& 1)))

/-- The high word, likewise: `a · x^i`'s high word is `(a >>> 1) >>> (63 − i)`, `0` at `i = 0`. -/
@[inline] def clmulStepHi (a bb t i hi : UInt64) : UInt64 :=
  hi ^^^ (((a >>> 1) >>> (63 - i)) &&& ((0 : UInt64) - ((bb >>> t) &&& 1)))

/-- `clmulStepLo`, `clmulStepHi` on bits `0, 1, 2, 3` of `bb`, as `a · x^s, …, a · x^(s+3)`, each word its own chain. -/
@[inline] def clmulWindow (a bb s lo hi : UInt64) : F128 :=
  ⟨clmulStepLo a bb 3 (s + 3) (clmulStepLo a bb 2 (s + 2) (clmulStepLo a bb 1 (s + 1) (clmulStepLo a bb 0 s lo))),
    clmulStepHi a bb 3 (s + 3) (clmulStepHi a bb 2 (s + 2) (clmulStepHi a bb 1 (s + 1) (clmulStepHi a bb 0 s hi)))⟩

/-- `clmulGo` four bits at a time: `fuel` windows of `bb`, from bit `s`, stopping once no bit of `bb` is left. -/
def clmulWin (a : UInt64) : Nat → UInt64 → UInt64 → UInt64 → UInt64 → F128
  | 0, _, _, lo, hi => ⟨lo, hi⟩
  | fuel + 1, s, bb, lo, hi =>
    if bb = 0 then ⟨lo, hi⟩ else
    clmulWin a fuel (s + 4) (bb >>> 4) (clmulWindow a bb s lo hi).lo (clmulWindow a bb s lo hi).hi

/-- `clmul` as compiled code computes it (`clmul_eq_clmulFast`): up to sixteen 4-bit windows, each branch-free, with no
`Nat` but the fuel. -/
@[inline] def clmulFast (a b : UInt64) : UInt64 × UInt64 :=
  let r := clmulWin a 16 0 b 0 0
  (r.lo, r.hi)

theorem clmulMask_eq (x c : UInt64) : x &&& ((0 : UInt64) - (c &&& 1)) = if c &&& 1 = 1 then x else 0 := by
  have h2 : (c &&& 1).toNat < 2 := by
    rw [UInt64.toNat_and]
    exact Nat.lt_of_le_of_lt Nat.and_le_right (by decide)
  rcases (by omega : (c &&& 1).toNat = 0 ∨ (c &&& 1).toNat = 1) with h | h
  · have e : c &&& 1 = 0 := UInt64.toNat_inj.mp (by simp [h])
    rw [e]
    apply UInt64.toNat_inj.mp
    simp
  · have e : c &&& 1 = 1 := UInt64.toNat_inj.mp (by simp [h])
    simp only [e, ↓reduceIte]
    apply UInt64.toNat_inj.mp
    rw [UInt64.toNat_and, show ((0 : UInt64) - 1).toNat = 2 ^ 64 - 1 from rfl, Nat.and_two_pow_sub_one_eq_mod,
      Nat.mod_eq_of_lt x.toNat_lt]

theorem clmulHi_eq (a j : UInt64) (i : Nat) (hj : j.toNat = i) (h64 : i < 64) :
    (a >>> 1) >>> (63 - j) = if i = 0 then 0 else a >>> (64 - i).toUInt64 := by
  have e63 : (63 - j).toNat = 63 - i := by
    rw [UInt64.toNat_sub_of_le _ _ (by rw [UInt64.le_iff_toNat_le]; simp; omega)]
    simp [hj]
  have hl : ((a >>> 1) >>> (63 - j)).toNat = a.toNat >>> (64 - i) := by
    rw [UInt64.toNat_shiftRight, UInt64.toNat_shiftRight, e63, show (1 : UInt64).toNat = 1 from rfl,
      ← Nat.shiftRight_add]
    congr 1
    omega
  apply UInt64.toNat_inj.mp
  rw [hl]
  split
  · rename_i h0
    subst h0
    rw [UInt64.toNat_zero, Nat.shiftRight_eq_div_pow]
    exact Nat.div_eq_of_lt a.toNat_lt
  · rw [UInt64.toNat_shiftRight, Nat.toUInt64, UInt64.toNat_ofNat']
    congr 1
    omega

theorem clmulGo_succ (a : UInt64) (n i : Nat) (bb lo hi j : UInt64) (hj : j.toNat = i) (h64 : i < 64) :
    clmulGo a (n + 1) i bb lo hi =
      clmulGo a n (i + 1) (bb >>> 1) (clmulStepLo a bb 0 j lo) (clmulStepHi a bb 0 j hi) := by
  have hjt : i.toUInt64 = j := UInt64.toNat_inj.mp (by simp [Nat.toUInt64, hj]; omega)
  simp only [clmulGo, clmulStepLo, clmulStepHi, UInt64.shiftRight_zero, clmulMask_eq, clmulHi_eq a j i hj h64, hjt]
  by_cases hb : bb &&& 1 = 1
  · by_cases hi0 : i = 0
    · simp [hb, hi0]
    · simp [hb, hi0]
  · simp [hb]

theorem clmulShr_shr (x m n p : UInt64) (h : m.toNat + n.toNat = p.toNat) (hp : p.toNat < 64) :
    (x >>> m) >>> n = x >>> p := by
  apply UInt64.toNat_inj.mp
  rw [UInt64.toNat_shiftRight, UInt64.toNat_shiftRight, UInt64.toNat_shiftRight, ← Nat.shiftRight_add]
  congr 1
  omega

theorem clmulGo_four (a : UInt64) (n i : Nat) (s bb lo hi : UInt64) (hs : s.toNat = i) (h : i + 4 ≤ 64) :
    clmulGo a (n + 4) i bb lo hi =
      clmulGo a n (i + 4) (bb >>> 4)
        (clmulStepLo a bb 3 (s + 3) (clmulStepLo a bb 2 (s + 2) (clmulStepLo a bb 1 (s + 1) (clmulStepLo a bb 0 s lo))))
        (clmulStepHi a bb 3 (s + 3) (clmulStepHi a bb 2 (s + 2) (clmulStepHi a bb 1 (s + 1) (clmulStepHi a bb 0 s hi)))) := by
  have hk : ∀ k : UInt64, k.toNat < 4 → (s + k).toNat = i + k.toNat := fun k hk => by
    rw [UInt64.toNat_add]; omega
  rw [clmulGo_succ a (n + 3) i bb lo hi s hs (by omega),
    clmulGo_succ a (n + 2) (i + 1) _ _ _ (s + 1) (hk 1 (by decide)) (by omega),
    clmulGo_succ a (n + 1) (i + 1 + 1) _ _ _ (s + 2) (hk 2 (by decide)) (by omega),
    clmulGo_succ a n (i + 1 + 1 + 1) _ _ _ (s + 3) (hk 3 (by decide)) (by omega),
    clmulShr_shr bb 1 1 2 (by decide) (by decide), clmulShr_shr bb 2 1 3 (by decide) (by decide),
    clmulShr_shr bb 3 1 4 (by decide) (by decide)]
  simp only [clmulStepLo, clmulStepHi, UInt64.shiftRight_zero]

theorem clmulGo_zero (a : UInt64) : ∀ (n i : Nat) (lo hi : UInt64), clmulGo a n i 0 lo hi = (lo, hi)
  | 0, _, _, _ => rfl
  | n + 1, i, lo, hi => by
    rw [clmulGo, show (0 : UInt64) >>> 1 = 0 from rfl]
    simpa using clmulGo_zero a n (i + 1) lo hi

theorem clmulWin_eq (a : UInt64) : ∀ (f : Nat) (s bb lo hi : UInt64), s.toNat + 4 * f ≤ 64 →
    clmulWin a f s bb lo hi = ⟨(clmulGo a (4 * f) s.toNat bb lo hi).1, (clmulGo a (4 * f) s.toNat bb lo hi).2⟩
  | 0, _, _, _, _, _ => rfl
  | f + 1, s, bb, lo, hi, h => by
    rw [clmulWin]
    split
    · rename_i hb
      rw [hb, clmulGo_zero]
    · have hs4 : (s + 4).toNat = s.toNat + 4 := by rw [UInt64.toNat_add]; simp; omega
      rw [clmulWin_eq a f (s + 4) _ _ _ (by omega), hs4, show 4 * (f + 1) = 4 * f + 4 by omega,
        clmulGo_four a (4 * f) s.toNat s bb lo hi rfl (by omega)]
      simp only [clmulWindow]

theorem clmulFast_eq_clmul (a b : UInt64) : clmulFast a b = clmul a b := by
  have h := clmulWin_eq a 16 0 b 0 0 (by decide)
  rw [show 4 * 16 = 64 from rfl, show (0 : UInt64).toNat = 0 from rfl] at h
  rw [clmulFast, h, clmul]

/-- Compiled code runs `clmulFast`; the kernel and every proof see `clmul`. `@[csimp]` rewrites only code compiled after
it, so it stays above `mul`. -/
@[csimp] theorem clmul_eq_clmulFast : @clmul = @clmulFast :=
  funext fun a => funext fun b => (clmulFast_eq_clmul a b).symm

/-- Reduce the 256-bit carry-less product `r3·2^192 + r2·2^128 + r1·2^64 + r0` modulo the GHASH polynomial. -/
@[inline] def reduce (r0 r1 r2 r3 : UInt64) : F128 :=
  let tl := r2 ^^^ (r2 <<< 1) ^^^ (r2 <<< 2) ^^^ (r2 <<< 7)
  let th := r3 ^^^ ((r3 <<< 1) ||| (r2 >>> 63)) ^^^ ((r3 <<< 2) ||| (r2 >>> 62)) ^^^ ((r3 <<< 7) ||| (r2 >>> 57))
  let o := (r3 >>> 63) ^^^ (r3 >>> 62) ^^^ (r3 >>> 57)
  ⟨r0 ^^^ tl ^^^ o ^^^ (o <<< 1) ^^^ (o <<< 2) ^^^ (o <<< 7), r1 ^^^ th⟩

def mul (a b : F128) : F128 :=
  let (p0l, p0h) := clmul a.lo b.lo
  let (p2l, p2h) := clmul a.hi b.hi
  let (ml, mh) := clmul (a.lo ^^^ a.hi) (b.lo ^^^ b.hi)
  let m1l := ml ^^^ p0l ^^^ p2l
  let m1h := mh ^^^ p0h ^^^ p2h
  reduce p0l (p0h ^^^ m1l) (p2l ^^^ m1h) p2h

instance instMul : Mul F128 := ⟨mul⟩

def square (a : F128) : F128 := mul a a

/-- `k` square-and-multiply steps by `a` from `acc`: `acc^(2^k) · a^(2^k − 1)`. -/
def invGo (a : F128) : Nat → F128 → F128
  | 0, acc => acc
  | k + 1, acc => invGo a k (mul (square acc) a)

/-- `a^(2^128 − 2)`: the inverse, with `inv 0 = 0` (§2.1). 2^128 − 2 = 0b111…10: 127 ones, then one zero. -/
def inv (a : F128) : F128 := square (invGo a 127 one)

/-- The 16-byte encoding `le64(lo) ‖ le64(hi)`. -/
def toBytes (a : F128) : ByteArray :=
  ⟨Array.ofFn (n := 16) fun i =>
    if i.val < 8 then (a.lo >>> (8 * i.val).toUInt64).toUInt8 else (a.hi >>> (8 * (i.val - 8)).toUInt64).toUInt8⟩

def pushBytes (out : ByteArray) (a : F128) : ByteArray := out ++ a.toBytes

@[inline] def bit (a : F128) (i : Nat) : Bool :=
  if i < 64 then (a.lo >>> i.toUInt64) &&& 1 == 1 else (a.hi >>> (i - 64).toUInt64) &&& 1 == 1

/-- `x^i` for `i < 128`. -/
def monomial (i : Nat) : F128 :=
  if i < 64 then ⟨(1 : UInt64) <<< i.toUInt64, 0⟩ else ⟨0, (1 : UInt64) <<< (i - 64).toUInt64⟩

private def hexDigit (n : UInt64) : Char :=
  let d := n.toNat
  if d < 10 then Char.ofNat (48 + d) else Char.ofNat (87 + d)

/-- Record hex: `{hi:016x}{lo:016x}` (§5.1). -/
def toHex (a : F128) : String := Id.run do
  let mut s := ""
  for i in [0:16] do s := s.push (hexDigit ((a.hi >>> (4 * (15 - i)).toUInt64) &&& 0xf))
  for i in [0:16] do s := s.push (hexDigit ((a.lo >>> (4 * (15 - i)).toUInt64) &&& 0xf))
  return s

end F128

/-- GF(2^256) = F[u]/(u^2 + u + x^{-1}) (§2.4); an element is `c0 + c1·u`. -/
structure F256 where
  c0 : F128
  c1 : F128
  deriving BEq, Inhabited, Repr, DecidableEq

namespace F256

/-- `x^{-1}` in F. -/
def xinv : F128 := ⟨0x43, 0x8000000000000000⟩

def zero : F256 := ⟨0, 0⟩
def one : F256 := ⟨1, 0⟩
/-- `u`. -/
def u : F256 := ⟨0, 1⟩
def ofF (a : F128) : F256 := ⟨a, 0⟩

@[inline] def add (a b : F256) : F256 := ⟨a.c0 + b.c0, a.c1 + b.c1⟩

def mul (a b : F256) : F256 :=
  let p0 := a.c0 * b.c0
  let p1 := a.c1 * b.c1
  let p2 := (a.c0 + a.c1) * (b.c0 + b.c1)
  ⟨p0 + xinv * p1, p2 + p0⟩

/-- `K · F`, coordinate-wise. -/
@[inline] def smul (a : F256) (s : F128) : F256 := ⟨a.c0 * s, a.c1 * s⟩

instance instAdd : Add F256 := ⟨add⟩
instance instMul : Mul F256 := ⟨mul⟩
instance instOfNatOfNatNat : OfNat F256 0 := ⟨zero⟩
instance instOfNatOfNatNat_1 : OfNat F256 1 := ⟨one⟩

end F256

/-- `φ8(α) = g`, a root of `X^8 + X^4 + X^3 + X + 1` in F (§2.2). -/
def phi8Gen : F128 := ⟨0x6b8330483c2e9849, 0x0dcb364640a222fe⟩

/-- `g^i`. -/
def phi8Pow : Nat → F128
  | 0 => F128.one
  | i + 1 => phi8Pow i * phi8Gen

/-- `Σ_{i < n} bit_i(v) · g^i`. -/
def phi8Sum (v : Nat) : Nat → F128
  | 0 => F128.zero
  | i + 1 => if v.testBit i then phi8Sum v i + phi8Pow i else phi8Sum v i

/-- `φ8(v) = Σ bit_i(v) · g^i` for bytes `v` (§2.2), tabulated. -/
def phi8Table : Array F128 := Array.ofFn (n := 256) fun v => phi8Sum v.val 8

@[inline] def phi8 (v : Nat) : F128 := phi8Table.getD (v % 256) 0

end Flock
