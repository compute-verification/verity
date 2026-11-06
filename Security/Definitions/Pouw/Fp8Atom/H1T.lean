import Definitions.Pouw.Fp8Atom.Fp32

/-!
# H-1T: the forming map, the checked chain and its credited words (trusted definitions)

H-1T is `docs/pouw/hardness-shaped-matrices.md` §3.1.
It is the map `h1_tagged.py` computes (`h1t_ref`, `tag_tuples`, `chain`), with the
salt fields of the kernel `h1t_form_xq` (`h1_kernel.py`).

* **Layout.** Each k32 slice has 29 real lanes (0..28) and 3 tag lanes (29, 30, 31). There are `T = ⌈k/29⌉` slices,
  and the row and the columns are zero-padded to `29·T`.
* **Salt** (change 1). Per (row, slice) the kernel reads 18 raw XOF words `r0..r17` (`RawSlice`). The map reads
  exactly these fields of them (`extract`, `SliceSalt`):
  * the sign of real lane `ℓ` is bit 15 (`ℓ` even) or bit 31 (`ℓ` odd) of `r_(ℓ/2)`;
  * tag lane 29 takes its sign and 3 mantissa bits from bits 31 and 25..23 of `r17`;
  * tag lanes 30 and 31 take theirs from bits 15 and 9..7, and 31 and 25..23, of `r16`.
  No other bit enters any word.
* **Real lanes.** `M` is the slice's largest magnitude, `F = max(2^(e(M) − 6), 2^−2)`, `p = 2^⌊log₂ |x|⌋` (0 at
  `x = 0`) and `μ = max(p/8, F)`. A set sign bit gives `s = +1`. Then `x1 = RNE_satfinite(x + s·μ)` and `x2 = −s·μ`,
  with `x + s·μ` exact over ℚ, as in `h1t_ref`.
* **Tag lanes.** `v = ±32F·(1 + m/8)`, negative when the sign bit is set, with `x1 = v` and `x2 = −v`.
* **B′** is the registered real codes plus the fixed tag rows (change 3). Column `j` carries `t0 = A0[j mod 23]`,
  `t1 = A1[(j / 23) mod 62]` and `t2 = A1[(j / 1426) mod 62]` on lanes 29, 30 and 31. Here `A0` is the 23 codes
  `0x68..0x7E` and `A1` the 62 codes `0x60..0x7E` and `0xE0..0xFE`.
* **Chain.** Position `p < T` is block 1 (`x1`) of slice `p`, and `T ≤ p < 2T` is block 2 (`x2`) of slice `p − T`.
  `I_p = hopperStep(+0, x_b, B′)` over the slice's 32 lanes, `R_0 = I_0` and `R_(p+1) = RN_f32(R_p + I_(p+1))`. Words
  are values in ℚ: two words equal as values count as equal.
* **Credited words** (`Pos`, change 2), 0-based: `I_p` for `p < 2T` and `R_p` for `1 ≤ p ≤ 2T − 3`. This is §3.1's
  `I_1..I_2T` and `R_2..R_(2T−2)`, `4T − 3` words per output for `T ≥ 2`.
* **A unit** (`Unit`) is `m` rows of `X_q` against `n` columns of B′ over `k`. It has its own salt, and its salt
  context binds the unit id (change 4), so different units' salts are independent coordinates.
* **Legal** (change 5): `k ≤ 32,768`, `n ≤ 88,412`, every code finite. There is no `|x| ≤ 416` clause, which is
  correctness and not security.
-/

namespace Pouw.Fp8Atom.H1T

open Pouw.Fp8Atom

/-! ## E4M3 encoding -/

/-- A code's value (`0` for the two NaNs, which legal inputs exclude). -/
def val (c : Code) : ℚ := (decode c).getD 0

/-- The code nearest a positive rational `y`, ties to the even code, saturating at `0x7E` (448). With
`e = max (binade y) (−6)` the grid step is `2^(e − 3)`, and the code is `8·(e + 6)` plus the rounded multiple of the
step (a subnormal for `e = −6`; a carry into the next binade is the next code). -/
def encPos (y : ℚ) : ℕ :=
  let e := max (binade y) (-6)
  min 126 (8 * (e + 6) + roundEven (y / 2 ^ (e - 3))).toNat

theorem encPos_le (y : ℚ) : encPos y ≤ 126 := by unfold encPos; exact Nat.min_le_left _ _

/-- `cvt.rn.satfinite.e4m3`: round to nearest even on the E4M3 grid, and clamp to ±448. Zero is `+0`. -/
def enc (y : ℚ) : Code :=
  if y = 0 then ⟨0, by norm_num⟩
  else if 0 < y then ⟨encPos y, by have := encPos_le y; omega⟩
  else ⟨128 + encPos (-y), by have := encPos_le (-y); omega⟩

/-! ## The salt -/

/-- One (row, slice)'s 18 raw XOF words `r0..r17`. -/
abbrev RawSlice := Fin 18 → Fin (2 ^ 32)

/-- The raw words all zero. -/
def rawZero : RawSlice := fun _ => ⟨0, by norm_num⟩

/-- The fields of one (row, slice)'s salt that the map reads. -/
structure SliceSalt where
  /-- Real lane `ℓ`'s sign bit: set means `s = +1` (`x1 = x + μ`, `x2 = −μ`). -/
  pos : Fin 29 → Bool
  /-- Tag lane `29 + i`'s sign bit: set means `v` is negative. -/
  neg : Fin 3 → Bool
  /-- Tag lane `29 + i`'s three mantissa bits `m`. -/
  man : Fin 3 → Fin 8

/-- `a`, `b` or `c` at index 0, 1 or 2. -/
def pick3 {α : Type} (a b c : α) (i : Fin 3) : α := if i.val = 0 then a else if i.val = 1 then b else c

/-- Bit `b` of `w`. -/
def bitAt (w b : ℕ) : Bool := (w >>> b) % 2 == 1

/-- Bits `b + 2 .. b` of `w`. -/
def bits3 (w b : ℕ) : Fin 8 := ⟨(w >>> b) % 8, Nat.mod_lt _ (by norm_num)⟩

/-- The fields `h1t_form_xq` reads (`h1_kernel.py`): real lanes `2i` and `2i + 1` from bits 15 and 31 of `r_i`; tag
lane 29 from bits 31 and 25..23 of `r17`; tag lanes 30 and 31 from bits 15, 9..7 and 31, 25..23 of `r16`. -/
def extract (r : RawSlice) : SliceSalt where
  pos l := bitAt (r ⟨l.val / 2, by have := l.isLt; omega⟩) (if l.val % 2 = 0 then 15 else 31)
  neg := pick3 (bitAt (r 17) 31) (bitAt (r 16) 15) (bitAt (r 16) 31)
  man := pick3 (bits3 (r 17) 23) (bits3 (r 16) 7) (bits3 (r 16) 23)

/-! ## The forming map (§3.1) -/

/-- The slice maximum `M`: the largest magnitude over the 29 real lanes. -/
def sliceMax (xs : Fin 29 → Code) : ℚ := (List.ofFn fun l => |val (xs l)|).foldl max 0

/-- `F = max(2^(e(M) − 6), 2^−2)` (`c0 = 6`, `c1 = 10`); `2^−2` at `M = 0`. -/
def floorF (M : ℚ) : ℚ := if M = 0 then 1 / 4 else max (2 ^ (binade M - 6)) (1 / 4)

/-- `p = 2^⌊log₂ |x|⌋`, `0` at `x = 0`. -/
def pow2Floor (x : ℚ) : ℚ := if x = 0 then 0 else 2 ^ binade |x|

/-- `μ = max(p/8, F)`. -/
def mu (F x : ℚ) : ℚ := max (pow2Floor x / 8) F

/-- A real lane with floor `F`, code `x` and sign bit `pos`: `(x1, x2) = (RNE_satfinite(x + s·μ), −s·μ)`. -/
def formReal (F : ℚ) (x : Code) (pos : Bool) : Code × Code :=
  let s : ℚ := if pos then 1 else -1
  (enc (val x + s * mu F (val x)), enc (-(s * mu F (val x))))

/-- A tag lane's value `v = ±32F·(1 + m/8)`, negative when the sign bit is set. -/
def tagVal (F : ℚ) (neg : Bool) (m : Fin 8) : ℚ := (if neg then -1 else 1) * (32 * F) * (1 + (m : ℚ) / 8)

/-- One slice's operands `(x1, x2)`: the real lanes formed with the slice's `F`, and the tag lanes `v` and `−v`. -/
def formSlice (xs : Fin 29 → Code) (σ : SliceSalt) : (Fin 32 → Code) × (Fin 32 → Code) :=
  let F := floorF (sliceMax xs)
  (fun l => if h : l.val < 29 then (formReal F (xs ⟨l.val, h⟩) (σ.pos ⟨l.val, h⟩)).1
    else enc (tagVal F (σ.neg ⟨l.val - 29, by have := l.isLt; omega⟩) (σ.man ⟨l.val - 29, by have := l.isLt; omega⟩)),
   fun l => if h : l.val < 29 then (formReal F (xs ⟨l.val, h⟩) (σ.pos ⟨l.val, h⟩)).2
    else enc (-tagVal F (σ.neg ⟨l.val - 29, by have := l.isLt; omega⟩) (σ.man ⟨l.val - 29, by have := l.isLt; omega⟩)))

/-! ## The tag rows -/

/-- `A0`: the 23 positive codes `0x68..0x7E` (64 to 448). -/
def A0 (i : Fin 23) : Code := ⟨0x68 + i.val, by have := i.isLt; omega⟩

/-- `A1`: the 62 codes `0x60..0x7E` (32 to 448) and `0xE0..0xFE` (their negations). -/
def A1 (i : Fin 62) : Code :=
  ⟨if i.val < 31 then 0x60 + i.val else 0xE0 + (i.val - 31), by have := i.isLt; split <;> omega⟩

/-- Column `j`'s tag tuple `(t0, t1, t2)`, on lanes 29, 30 and 31. -/
def tuple (j : ℕ) : Fin 3 → Code :=
  pick3 (A0 ⟨j % 23, Nat.mod_lt _ (by norm_num)⟩) (A1 ⟨j / 23 % 62, Nat.mod_lt _ (by norm_num)⟩)
    (A1 ⟨j / 1426 % 62, Nat.mod_lt _ (by norm_num)⟩)

/-! ## The chain of one row and column -/

/-- Real lane `ℓ` of slice `τ` of a zero-padded row: element `29τ + ℓ`. -/
def realLanes (x : ℕ → Code) (τ : ℕ) : Fin 29 → Code := fun l => x (29 * τ + l.val)

/-- Column `j`'s 32 lanes in slice `τ`: its zero-padded real codes `b` on lanes 0..28, its tuple on 29..31. -/
def colLanes (b : ℕ → Code) (j τ : ℕ) : Fin 32 → Code :=
  fun l => if h : l.val < 29 then b (29 * τ + l.val) else tuple j ⟨l.val - 29, by have := l.isLt; omega⟩

/-- The fresh-step word at position `p` of a row `x` and a column (`b`, `j`) under the slices' salts `σ`: block 1
(`x1`) of slice `p` for `p < T`, block 2 (`x2`) of slice `p − T` otherwise. -/
def freshWord (T : ℕ) (x b : ℕ → Code) (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) : ℕ :=
  if p < T then hopperStep 0 (formSlice (realLanes x p) (σ p)).1 (colLanes b j p)
  else hopperStep 0 (formSlice (realLanes x (p - T)) (σ (p - T))).2 (colLanes b j (p - T))

/-- A word's value (`0` if it is not finite, which on legal inputs no H-1T word is). -/
def wordVal (w : ℕ) : ℚ := (wordQ w).getD 0

/-- `I_p` as a value. -/
def freshVal (T : ℕ) (x b : ℕ → Code) (j : ℕ) (σ : ℕ → SliceSalt) (p : ℕ) : ℚ := wordVal (freshWord T x b j σ p)

/-- The running words: `R_0 = I_0` and `R_(p+1) = RN_f32(R_p + I_(p+1))`. -/
def runVal (T : ℕ) (x b : ℕ → Code) (j : ℕ) (σ : ℕ → SliceSalt) : ℕ → ℚ
  | 0 => freshVal T x b j σ 0
  | p + 1 => rnAdd (runVal T x b j σ p) (freshVal T x b j σ (p + 1))

/-! ## Units -/

/-- The credited positions of one output (0-based): `I_p` for `p < 2T` (`inl`) and `R_p` for `1 ≤ p ≤ 2T − 3`
(`inr`). -/
abbrev Pos (T : ℕ) := Fin (2 * T) ⊕ {p : Fin (2 * T) // 1 ≤ p.val ∧ p.val + 3 ≤ 2 * T}

/-- A unit: `m` rows of `X_q` (`xq`, E4M3 codes) against `n` columns of B′'s registered real codes (`b`) over `k`. -/
structure Unit where
  m : ℕ
  k : ℕ
  n : ℕ
  xq : Fin m → Fin k → Code
  b : Fin k → Fin n → Code

namespace Unit

/-- The number of slices, `⌈k/29⌉`. -/
def T (U : Unit) : ℕ := (U.k + 28) / 29

/-- Row `i` of `X_q`, zero-padded. -/
def row (U : Unit) (i : Fin U.m) : ℕ → Code := fun r => if h : r < U.k then U.xq i ⟨r, h⟩ else ⟨0, by norm_num⟩

/-- Column `j` of B′'s real codes, zero-padded. -/
def col (U : Unit) (j : Fin U.n) : ℕ → Code := fun r => if h : r < U.k then U.b ⟨r, h⟩ j else ⟨0, by norm_num⟩

/-- The unit's salt: 18 raw XOF words per (row, slice). -/
abbrev Salt (U : Unit) := Fin U.m → Fin U.T → RawSlice

/-- Row `i`'s salt fields per slice. -/
def rowSalt (U : Unit) (g : U.Salt) (i : Fin U.m) : ℕ → SliceSalt :=
  fun τ => if h : τ < U.T then extract (g i ⟨τ, h⟩) else extract rawZero

/-- The credited word of row `i`, column `j`, position `a`, as a function of the unit's salt. -/
def word (U : Unit) (c : Fin U.m × Fin U.n × Pos U.T) (g : U.Salt) : ℚ :=
  match c.2.2 with
  | .inl p => freshVal U.T (U.row c.1) (U.col c.2.1) c.2.1.val (U.rowSalt g c.1) p.val
  | .inr p => runVal U.T (U.row c.1) (U.col c.2.1) c.2.1.val (U.rowSalt g c.1) p.val.val

/-- The free values on the unit's salt (W1): the constants, which include every salt-independent function of the
committed input, and the raw XOF words. -/
def free (U : Unit) : Set (U.Salt → ℚ) :=
  {f | (∃ q : ℚ, f = fun _ => q) ∨ ∃ (i : Fin U.m) (τ : Fin U.T) (ℓ : Fin 18), f = fun g => ((g i τ ℓ : ℕ) : ℚ)}

/-- **Legal inputs**: `k ≤ 32,768`, `n ≤ 88,412` columns per unit, and every code of `X_q` and of B′ finite. -/
def Legal (U : Unit) : Prop :=
  U.k ≤ 32768 ∧ U.n ≤ 88412 ∧ (∀ i l, (U.xq i l).isNaN = false) ∧ ∀ l j, (U.b l j).isNaN = false

end Unit

end Pouw.Fp8Atom.H1T
