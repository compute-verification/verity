import Definitions.Core.TC.Pipeline

/-!
# The step relation

The tensor-core step relation, the definition of record: its tables (`tBf16`, `tPcls`, `tPow`, `tNorm`, `tExp`, `tShift`,
`tAny`), its predicate kinds (`MUL`, `BOOL`, `LIN`, `LOOKUP`, `RANGE`, `ISZERO`) and `StepHolds`, over the integers. It
reads nothing from `Verity.TC.Spec`: the relation and the semantics are independent definitions, and
`Verity.SecurityProofs.TC.Sound` proves them equivalent.

* **The witness has a fixed shape**: `k` product rows, one group row per group, `size + 1` term rows per group
  (`Witness.Shaped`), the columns the census counts.
* **The operands and the input state are arguments**, not witness fields.
* **`RANGE` and `ISZERO` are their meaning**: `RANGE x bits` is `0 ≤ x < 2^bits`, which the census's range gadget
  accepts (one `T_range` lookup when `bits ≤ 16`; otherwise non-negativity, the 16-bit chunk split and a lookup per
  chunk), and `ISZERO x bits z` is `x` in range with `z = [x = 0]`, which the census's 12-bit chunks, lookups and muls
  accept. `Verity.SecurityProofs.TC.Gadgets` proves the chunked forms equivalent.
* **Products the checker computes itself** (`both`, `fin_inf`, `any_inf * any_ninf`, `inf_out * neg_inf`, the running
  product of the "attained" chain) appear as those products, and the `MUL`s that check them against themselves are kept.
-/

namespace Verity.TC.Relation

def CLS_FIN : Int := 0
def CLS_ZERO : Int := 1
def CLS_INF : Int := 2
def CLS_NAN : Int := 3
def FP32_MIN_E : Int := -126
def FP32_MAX_E : Int := 127
def E_OFF : Int := 192

/-- `relation.Params`. -/
structure Params where
  W : Nat
  rho : Nat
  A : Nat
  B : Nat
  K : Nat
  Z : Int
  n : Nat

/-- `relation.params`. `rho = W - 24` is a natural number: the relation is defined for BF16 pipelines (`W ≥ 24`). -/
def params (pipe : Pipeline) : Params :=
  let W := pipe.width
  let n := pipe.groups.foldl max 0
  { W := W, rho := W - 24, A := W + 1, B := W + 1 + bitLength n, K := W + 10, Z := pipe.zeroExponent, n := n }

/-! ## Tables: a key off the table is `none` (the Python's `KeyError`) -/

/-- `T_bf16`: word → (sign, effective exponent `e or 1`, significand with the hidden bit (0 unless finite nonzero),
class). -/
def tBf16 (word : Int) : Option (Int × Int × Int × Int) :=
  if 0 ≤ word ∧ word < 2 ^ 16 then
    let w := word.toNat
    let s := w >>> 15
    let e := (w >>> 7) &&& 0xFF
    let m := w &&& 0x7F
    if e = 0xFF then some (s, 1, 0, if m ≠ 0 then CLS_NAN else CLS_INF)
    else if e = 0 ∧ m = 0 then some (s, 1, 0, CLS_ZERO)
    else some (s, (if e ≠ 0 then e else 1 : Nat), (m ||| (if e ≠ 0 then 0x80 else 0) : Nat), CLS_FIN)
  else none

/-- `T_pcls`: (class a, class b, sign a, sign b) → (nan, +inf, -inf, zero-in-the-finite-path, sign) of the product.
The Python's `sa ^ sb` on signs in `{0, 1}` is `if sa = sb then 0 else 1`. -/
def tPcls (key : Int × Int × Int × Int) : Option (Int × Int × Int × Int × Int) :=
  let (ca, cb, sa, sb) := key
  if 0 ≤ ca ∧ ca < 4 ∧ 0 ≤ cb ∧ cb < 4 ∧ (sa = 0 ∨ sa = 1) ∧ (sb = 0 ∨ sb = 1) then
    let sp : Int := if sa = sb then 0 else 1
    if ca = CLS_NAN ∨ cb = CLS_NAN ∨ (ca = CLS_INF ∧ cb = CLS_ZERO) ∨ (cb = CLS_INF ∧ ca = CLS_ZERO) then
      some (1, 0, 0, 1, sp)
    else if ca = CLS_INF ∨ cb = CLS_INF then some (0, 1 - sp, sp, 1, sp)
    else some (0, 0, 0, if ca ≠ CLS_FIN ∨ cb ≠ CLS_FIN then 1 else 0, sp)
  else none

/-- `T_pow`: `d ∈ [0, 512)` → `(2^min(d, A), 2^(A - min(d, A)))`. -/
def tPow (p : Params) (d : Int) : Option (Int × Int) :=
  if 0 ≤ d ∧ d < 512 then
    let dd := min d.toNat p.A
    some (2 ^ dd, 2 ^ (p.A - dd))
  else none

/-- `T_norm`: `L ∈ [0, B]` → `(2^(B - L), [L = 0])`. -/
def tNorm (p : Params) (L : Int) : Option (Int × Int) :=
  if 0 ≤ L ∧ L ≤ p.B then some (2 ^ (p.B - L.toNat), if L = 0 then 1 else 0) else none

/-- `T_exp`: `e_out + E_OFF` → (final exponent, `2^(subnormal shift + rho)`, `2^(K - that)`, overflow). -/
def tExp (p : Params) (idx : Int) : Option (Int × Int × Int × Int) :=
  if 0 ≤ idx ∧ idx < 512 then
    let e_out := idx - E_OFF
    let e_fin := max e_out FP32_MIN_E
    let sh := min (e_fin - e_out + p.rho) p.K
    some (e_fin, 2 ^ sh.toNat, 2 ^ (p.K - sh).toNat, if e_fin > FP32_MAX_E then 1 else 0)
  else none

/-- `T_shift`: `(P, d)` → `floor((P << (9 + rho)) / 2^d)`, the alternative product alignment. -/
def tShift (p : Params) (key : Int × Int) : Option Int :=
  let (P, d) := key
  if 0 ≤ P ∧ P < 2 ^ 16 ∧ 0 ≤ d ∧ d < 512 then some ((P.toNat <<< (9 + p.rho)) >>> min d.toNat p.A : Nat)
  else none

/-- `T_any(n)`: `s ∈ [0, n]` → `[s ≠ 0]`. -/
def tAny (n : Nat) (s : Int) : Option Int :=
  if 0 ≤ s ∧ s ≤ n then some (if s ≠ 0 then 1 else 0) else none

/-! ## Predicate kinds -/

/-- `Census.MUL`: `a * b = c`. -/
def MUL (a b c : Int) : Prop := a * b = c

/-- `Census.BOOL`: `x * (1 - x) = 0`. -/
def BOOL (x : Int) : Prop := MUL x (1 - x) 0

/-- `Census.LIN`: a linear residual is zero. -/
def LIN (residual : Int) : Prop := residual = 0

/-- `Census.LOOKUP`: `(key, outs)` is a row of the table. -/
def LOOKUP {κ ο : Type} (T : κ → Option ο) (key : κ) (outs : ο) : Prop := T key = some outs

/-- `Census.RANGE`: `x ∈ [0, 2^bits)`. -/
def RANGE (x : Int) (bits : Nat) : Prop := 0 ≤ x ∧ x < 2 ^ bits

/-- `Census.ISZERO`: `x ∈ [0, 2^bits)` and `z = [x = 0]`. -/
def ISZERO (x : Int) (bits : Nat) (z : Int) : Prop := RANGE x bits ∧ z = if x = 0 then 1 else 0

/-! ## State and witness -/

/-- `relation.State`: the decoded accumulator. -/
structure State where
  nan : Int
  inf : Int
  neg : Int
  e : Int
  M : Int
deriving Inhabited

/-- `relation.zero_state`. -/
def zeroState (pipe : Pipeline) : State := ⟨0, 0, 0, pipe.zeroExponent, 0⟩

/-- `relation.decode_state`: the boundary decode of an FP32 word below `2^32` (not a predicate: a VU starts from the
constant `zero_state`, and each step's input is the previous step's output). -/
def decodeState (word : Nat) (pipe : Pipeline) : State :=
  let Z := pipe.zeroExponent
  let s := word >>> 31
  let E := (word >>> 23) &&& 0xFF
  let m := word &&& 0x7FFFFF
  if E = 0xFF then (if m ≠ 0 then ⟨1, 0, 0, Z, 0⟩ else ⟨0, 1, s, Z, 0⟩)
  else if E = 0 then (if m = 0 then ⟨0, 0, 0, Z, 0⟩ else ⟨0, 0, s, FP32_MIN_E, m⟩)
  else ⟨0, 0, s, (E : Int) - 127, ((1 <<< 23) ||| m : Nat)⟩

/-- `relation.pack_state` on a canonical state (its fields are non-negative, except the exponent). -/
def packState (st : State) : Nat :=
  if st.nan ≠ 0 then 0x7FFFFFFF
  else if st.inf ≠ 0 then (st.neg.toNat <<< 31) ||| 0x7F800000
  else if st.M = 0 then 0
  else
    (st.neg.toNat <<< 31) ||| ((st.e + 127 - (if st.M.toNat >>> 23 ≠ 0 then 0 else 1)).toNat <<< 23)
      ||| (st.M.toNat &&& 0x7FFFFF)

/-- The product columns `step_witness` puts in `w["prod"][j]`. -/
structure ProdCols where
  sa : Int
  ea : Int
  SA : Int
  ca : Int
  sb : Int
  eb : Int
  SB : Int
  cb : Int
  nan : Int
  pinf : Int
  ninf : Int
  zero : Int
  sp : Int
  P : Int
  ep : Int
  em : Int
deriving Inhabited

/-- The alignment columns of one term, `g["terms"][i]`. -/
structure TermCols where
  d : Int
  pow : Int
  invpow : Int
  aligned : Int
  r : Int
  rinv : Int
  signed : Int
deriving Inhabited

/-- The columns of one group, `w["groups"][g]`, its output state included. -/
structure GroupCols where
  Estar : Int
  terms : List TermCols
  total : Int
  neg_t : Int
  mag_t : Int
  L : Int
  normpow : Int
  z_t : Int
  normB : Int
  magW : Int
  r_n : Int
  e_out : Int
  e_fin : Int
  pow_s : Int
  invpow_s : Int
  ovf : Int
  mag24 : Int
  r_s : Int
  rsinv : Int
  z_out : Int
  pinf0 : Int
  ninf0 : Int
  S_nan : Int
  S_pinf : Int
  S_ninf : Int
  any_nan : Int
  any_pinf : Int
  any_ninf : Int
  ovf_eff : Int
  any_inf : Int
  nan_out : Int
  inf_out : Int
  neg_inf : Int
  fin : Int
  neg_fin : Int
  e_state : Int
  out : State
deriving Inhabited

/-- A step witness: `w["prod"]` and `w["groups"]`. -/
structure Witness where
  prod : List ProdCols
  groups : List GroupCols
deriving Inhabited

/-- The alignment mode: `ALIGN_DIV` (every term by division with remainder) or `ALIGN_TABLE` (products by one
`T_shift` lookup, the accumulator term by division). -/
inductive Align where
  | div
  | table
deriving DecidableEq

/-- The witness's shape: `k` product rows, one group row per group, `size + 1` term rows in each. -/
def Witness.Shaped (pipe : Pipeline) (w : Witness) : Prop :=
  w.prod.length = pipe.k ∧ w.groups.length = pipe.groups.length ∧
    ∀ i (h : i < w.groups.length) (h' : i < pipe.groups.length), (w.groups[i]).terms.length = pipe.groups[i] + 1

/-! ## The predicates of `check_step` -/

/-- One term of a group as `_terms` lists it: (sign, masked exponent, scaled magnitude). -/
structure TermIn where
  neg : Int
  e : Int
  scaled : Int

/-- `relation._terms`: the accumulator, then each product of the group. -/
def terms (p : Params) (st : State) (prods : List ProdCols) : List TermIn :=
  ⟨st.neg, st.e, st.M * 2 ^ p.rho⟩ :: prods.map (fun q => ⟨q.sp, q.em, q.P * 2 ^ (9 + p.rho)⟩)

/-- The predicates on one product: decode, class logic, exact significand product, exponent, zero mask. -/
def prodHolds (p : Params) (a b : Nat) (q : ProdCols) : Prop :=
  LOOKUP tBf16 (a : Int) (q.sa, q.ea, q.SA, q.ca) ∧
  LOOKUP tBf16 (b : Int) (q.sb, q.eb, q.SB, q.cb) ∧
  LOOKUP tPcls (q.ca, q.cb, q.sa, q.sb) (q.nan, q.pinf, q.ninf, q.zero, q.sp) ∧
  MUL q.SA q.SB q.P ∧
  LIN (q.ep - (q.ea + q.eb - 254)) ∧
  MUL q.zero (p.Z - q.ep) (q.em - q.ep)

/-- The predicates on term `i` of a group: `E* - e_i` through `T_pow`, then the truncating alignment and the sign. -/
def termHolds (p : Params) (align : Align) (Estar : Int) (i : Nat) (Pprev : Int) (x : TermIn) (t : TermCols) : Prop :=
  LIN (t.d - (Estar - x.e)) ∧
  LOOKUP (tPow p) t.d (t.pow, t.invpow) ∧
  (if i = 0 ∨ align = .div then
    MUL t.aligned t.pow (x.scaled - t.r) ∧ RANGE t.aligned p.A ∧ MUL t.r t.invpow t.rinv ∧ RANGE t.rinv p.A
   else LOOKUP (tShift p) (Pprev, t.d) t.aligned) ∧
  MUL x.neg t.aligned t.signed

/-- The predicates of one group of `check_step`, on its input state `st` and its `size` product rows. -/
def groupHolds (p : Params) (align : Align) (size : Nat) (st : State) (prods : List ProdCols) (g : GroupCols) : Prop :=
  let ins := terms p st prods
  -- E*: at least every masked exponent and the floor (d in [0, 512) via T_pow), and attained
  (∀ i (h : i < ins.length) (h' : i < g.terms.length),
    termHolds p align g.Estar i (if i = 0 then 0 else (prods[i - 1]!).P) ins[i] g.terms[i]) ∧
  (∀ i (h : i < ins.length), MUL (ins.take i |>.foldl (fun c x => c * (g.Estar - x.e)) (g.Estar - p.Z))
      (g.Estar - ins[i].e) ((ins.take (i + 1)).foldl (fun c x => c * (g.Estar - x.e)) (g.Estar - p.Z))) ∧
  LIN (ins.foldl (fun c x => c * (g.Estar - x.e)) (g.Estar - p.Z)) ∧
  -- truncating alignment, signed sum
  LIN (g.total - (g.terms.map (fun t => t.aligned - 2 * t.signed)).sum) ∧
  -- magnitude and sign of the total; normalisation to W bits; subnormal shift and rescale
  BOOL g.neg_t ∧
  MUL (1 - 2 * g.neg_t) g.total g.mag_t ∧
  RANGE g.mag_t p.B ∧
  LOOKUP (tNorm p) g.L (g.normpow, g.z_t) ∧
  MUL g.mag_t g.normpow g.normB ∧
  RANGE (g.normB - (1 - g.z_t) * 2 ^ (p.B - 1)) (p.B - 1) ∧
  LIN (g.magW * 2 ^ (p.B - p.W) + g.r_n - g.normB) ∧
  RANGE g.r_n (p.B - p.W) ∧
  LIN (g.e_out - (g.Estar + g.L - p.W)) ∧
  LOOKUP (tExp p) (g.e_out + E_OFF) (g.e_fin, g.pow_s, g.invpow_s, g.ovf) ∧
  MUL g.mag24 g.pow_s (g.magW - g.r_s) ∧
  RANGE g.mag24 24 ∧
  MUL g.r_s g.invpow_s g.rsinv ∧
  RANGE g.rsinv p.K ∧
  ISZERO g.mag24 24 g.z_out ∧
  -- special values and the output state
  MUL st.inf (1 - st.neg) g.pinf0 ∧
  MUL st.inf st.neg g.ninf0 ∧
  LIN (g.S_nan - st.nan - (prods.map (·.nan)).sum) ∧
  LIN (g.S_pinf - g.pinf0 - (prods.map (·.pinf)).sum) ∧
  LIN (g.S_ninf - g.ninf0 - (prods.map (·.ninf)).sum) ∧
  LOOKUP (tAny (size + 1)) g.S_nan g.any_nan ∧
  LOOKUP (tAny (size + 1)) g.S_pinf g.any_pinf ∧
  LOOKUP (tAny (size + 1)) g.S_ninf g.any_ninf ∧
  MUL g.ovf (1 - g.z_t) g.ovf_eff ∧
  MUL g.any_pinf g.any_ninf (g.any_pinf * g.any_ninf) ∧
  LIN (g.any_inf - (g.any_pinf + g.any_ninf - g.any_pinf * g.any_ninf)) ∧
  MUL (1 - g.any_nan) (g.any_pinf * g.any_ninf) (g.nan_out - g.any_nan) ∧
  MUL (1 - g.any_inf) g.ovf_eff (g.any_inf + (1 - g.any_inf) * g.ovf_eff - g.any_inf) ∧
  MUL (1 - g.nan_out) (g.any_inf + (1 - g.any_inf) * g.ovf_eff) g.inf_out ∧
  MUL g.any_inf g.any_ninf (g.any_inf * g.any_ninf) ∧
  MUL (1 - g.any_inf) g.neg_t (g.neg_inf - g.any_inf * g.any_ninf) ∧
  LIN (g.fin - (1 - g.nan_out - g.inf_out)) ∧
  MUL g.neg_t (1 - g.z_out) g.neg_fin ∧
  MUL g.z_out (p.Z - g.e_fin) (g.e_state - g.e_fin) ∧
  MUL g.inf_out g.neg_inf (g.inf_out * g.neg_inf) ∧
  MUL g.fin g.neg_fin (g.out.neg - g.inf_out * g.neg_inf) ∧
  MUL g.fin (g.e_state - p.Z) (g.out.e - p.Z) ∧
  MUL g.fin g.mag24 g.out.M ∧
  LIN (g.out.nan - g.nan_out) ∧
  LIN (g.out.inf - g.inf_out)

/-- The groups of `check_step` in order: each group's input state is the previous group's output state. -/
def groupsHold (p : Params) (align : Align) : List Nat → State → List ProdCols → List GroupCols → Prop
  | size :: sizes, st, prods, g :: gs =>
    groupHolds p align size st (prods.take size) g ∧ groupsHold p align sizes g.out (prods.drop size) gs
  | _, _, _, _ => True

/-- The state `check_step` returns: the last group's output. -/
def stepOut (st : State) : List GroupCols → State
  | [] => st
  | g :: gs => stepOut g.out gs

/-- `check_step` accepts the witness `w` for the input state `st` and the operand words `a`, `b`: the shape, then every
product's predicates, then every group's. -/
def StepHolds (pipe : Pipeline) (align : Align) (st : State) (a b : List Nat) (w : Witness) : Prop :=
  let p := params pipe
  w.Shaped pipe ∧
  (∀ j (h : j < w.prod.length), prodHolds p (a[j]!) (b[j]!) w.prod[j]) ∧
  groupsHold p align pipe.groups st w.prod w.groups

/-- A verification unit's steps from the input state `st`, as `test_verification_unit_on_captured_gemm_coordinates` chains
them: each step's input state is the previous step's output, and `out` is the last one's. -/
def ChainHolds (pipe : Pipeline) (align : Align) : State → List (List Nat × List Nat × Witness) → State → Prop
  | st, [], out => out = st
  | st, (a, b, w) :: rest, out => StepHolds pipe align st a b w ∧ ChainHolds pipe align (stepOut st w.groups) rest out

/-- The columns of the unit boundary, `pack_relation`: the FP32 word of the final state and its `cvt.rn.bf16.f32`
rounding. `low = M - hidden * 2^23`, `fin = 1 - nan - inf`, `nz = 1 - z` and `t = lo + 0x7FFF + b16` are the linear
expressions the Python computes from them. -/
structure PackCols where
  hidden : Int
  z : Int
  nh_nz : Int
  fin_nz : Int
  Ebias : Int
  mant : Int
  word : Int
  hi : Int
  lo : Int
  b16 : Int
  hi2 : Int
  carry : Int
  rem : Int
  rounded : Int
  wrap : Int
  bf16 : Int

/-- The first half of `relation.pack_relation` on the final state `st`: its FP32 word. -/
def packWordHolds (st : State) (c : PackCols) : Prop :=
  let low := st.M - c.hidden * 2 ^ 23
  let fin := 1 - st.nan - st.inf
  BOOL c.hidden ∧
  RANGE low 23 ∧
  ISZERO st.M 24 c.z ∧
  MUL (1 - c.hidden) (1 - c.z) c.nh_nz ∧
  MUL c.nh_nz (st.e - FP32_MIN_E) 0 ∧
  MUL fin (1 - c.z) c.fin_nz ∧
  MUL c.fin_nz (st.e + 126 + c.hidden) (c.Ebias - (st.nan + st.inf) * 255) ∧
  MUL fin low (c.mant - st.nan * 0x7FFFFF) ∧
  LIN (c.word - (st.neg * 2 ^ 31 + c.Ebias * 2 ^ 23 + c.mant))

/-- The second half of `relation.pack_relation`, `cvt.rn.bf16.f32` of the word: every NaN to `0x7FFF`; otherwise RNE,
carrying into the exponent. -/
def cvtHolds (nan : Int) (c : PackCols) : Prop :=
  RANGE c.lo 16 ∧
  RANGE c.hi 16 ∧
  LIN (c.hi * 65536 + c.lo - c.word) ∧
  BOOL c.b16 ∧
  RANGE c.hi2 15 ∧
  LIN (c.hi2 * 2 + c.b16 - c.hi) ∧
  BOOL c.carry ∧
  RANGE c.rem 16 ∧
  LIN (c.carry * 65536 + c.rem - (c.lo + 0x7FFF + c.b16)) ∧
  BOOL c.wrap ∧
  RANGE c.rounded 16 ∧
  LIN (c.rounded + c.wrap * 65536 - c.hi - c.carry) ∧
  MUL nan (0x7FFF - c.rounded) (c.bf16 - c.rounded)

/-- `relation.pack_relation` on the final state `st`: the FP32 word, then its `cvt.rn.bf16.f32` rounding. -/
def packHolds (st : State) (c : PackCols) : Prop := packWordHolds st c ∧ cvtHolds st.nan c

end Verity.TC.Relation
