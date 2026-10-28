import Mathlib.Data.ZMod.Basic
import Definitions.Pous.Accounting.Params
import Definitions.Pous.Model.M1p

/-!
# P2's cost model: a generic ring over `Z/p` with random odd labels (trusted)

P2's maps are ring operations mod `p` (`sq`, its root `ρ`) mixed with operations on the integers' representation
(parity, and the mask `σ_K(r) = r ⊕ K`). This file models them so that the encoder's two roots can be charged in
sequential multiplications, not in oracle queries.

**The model `ringRO B w p`.** Setup draws, uniformly:

* an *odd labeling* `φ : Z/p ≃ [0, p)` with `φ(−x) = −φ(x)` (that is, `p − φ(x)` for `x ≠ 0`); every element is
  seen only as its label, a `w`-bit string below `p`;
* block `j`'s keys `(K_j, τ_j)`: a `w`-bit mask `K_j` and the label `τ_j < p` of the offset `t_j = φ⁻¹(τ_j)`.

Queries: `add a b` and `mul a b` on labels, answering the label of the sum or product, and `one`, answering `φ(1)`.
A label `≥ p` names no element: a ring query on one is answered by the junk label `junkLabel a b = a`, its first
argument unchanged (as `permAct` fixes the tail `[p, 2^w)` in `Security/Definitions/Pous/Model/M1p.lean`).
`key j part` answers one half of block `j`'s keys: `K_j` (`mask`) or `τ_j` (`offset`), framed per (block, purpose) as
P2-EXP-IO frames SHAKE256. Parity, negation, equality and XOR act on labels, locally and for free. The labeling is
dense, so naming a label hands the adversary a uniform element for free. At `φ = id`, `p2ring` is `schemes/p2.py`'s
`encode_segment` and `decode_segment` for given `(K, t)`.

**The cost bound `Prog.RingBounded D Q L`**, beside `Prog.Bounded`: on every primitive, at most `D` rounds that
contain a `mul` (rounds of only `add`, `one` and `key` cost no depth), at most `Q` key queries and at most `L` ring
queries (`add`, `mul`, `one`) in total. `Q` counts key queries, not Keccak-f calls: the certificate converts, at the
key expander's cost per query. Hardware enters only through the certificate's numbers, never here: `D` from the
claim `ring-mul-floor` (no device in scope completes one multiplication in `Z/p` at the scheme's width faster than
`t_mul`, using all of its resources; ASICs and FPGAs are out of its scope), `Q` from `keccak-rate-ceiling`, and `L`
from `ring-op-ceiling` (no responder, with whatever devices it uses, performs more than `L` ring operations within one
answer's deadline; throughput adds up across devices, so the certificate prices `L` from a device count or a budget).
The model itself is the claim `generic-ring`. Every statement over these is parametric in `D`, `Q` and `L`.

`SeqResponderOn`, `SimResponderOn`, `AuditSecureOn` and `MeetsOn` are `SeqResponder`, `SimResponder`, `AuditSecure`
and `Meets` with `RingBounded D Q L` in place of `Bounded D Q`. `Bounded`, `AuditSecure` and `Meets` are unchanged.
`MeetsWithSlack` is `Meets` with `lam` more bits of adversary state per block, the requirement P2's margin pins read.
-/

namespace Pous

open scoped ENNReal

/-! ## The ring model -/

/-- An odd labeling of `Z/p`: a bijection onto the labels `[0, p)` with `φ(−x) = −φ(x)` in `Fin p`. -/
abbrev OddLabeling (p : ℕ) [NeZero p] : Type := {φ : ZMod p ≃ Fin p // ∀ x, φ (-x) = -φ x}

/-- Which half of a block's keys a key query asks for: the mask `K_j`, or the label of the offset `t_j`. -/
inductive KeyPart : Type
  | mask
  | offset

/-- The ring model's queries: ring operations on labels, the constant `1`, and block `j`'s keys. -/
inductive RingQry (B w : ℕ) : Type
  | add : Bits w → Bits w → RingQry B w
  | mul : Bits w → Bits w → RingQry B w
  | one : RingQry B w
  | key : Fin B → KeyPart → RingQry B w

/-- A multiplication query. A round that contains one costs one round of depth. -/
def RingQry.isMul {B w : ℕ} : RingQry B w → Bool
  | .mul _ _ => true
  | _ => false

/-- A key query. The others (`add`, `mul`, `one`) are ring queries. -/
def RingQry.isKey {B w : ℕ} : RingQry B w → Bool
  | .key _ _ => true
  | _ => false

variable {B w p : ℕ} [NeZero p]

/-- The element behind a label below `p`, and `0` for an invalid label (`≥ p`). -/
def eltD (φ : OddLabeling p) (a : Bits w) : ZMod p :=
  if h : bitsVal a < p then φ.1.symm ⟨bitsVal a, h⟩ else 0

/-- The label of an element, as a `w`-bit string. -/
def lab (w : ℕ) (φ : OddLabeling p) (x : ZMod p) : Bits w := bitsOfNat w (φ.1 x)

/-- **The junk label**: a ring query on a label `≥ p` is answered by its first argument, unchanged. -/
def junkLabel (a _b : Bits w) : Bits w := a

/-- A ring operation on labels: the label of `op x y` when both labels are below `p`, else the junk label. -/
def ringOp (φ : OddLabeling p) (op : ZMod p → ZMod p → ZMod p) (a b : Bits w) : Bits w :=
  if bitsVal a < p ∧ bitsVal b < p then lab w φ (op (eltD φ a) (eltD φ b)) else junkLabel a b

/-- **The generic ring over `Z/p` with a random odd labeling**, and per-block keys. -/
def ringRO (B w p : ℕ) [NeZero p] : IdealModel where
  Ω := OddLabeling p × (Fin B → Bits w × Fin p)
  nonemptyΩ := ⟨(⟨(ZMod.finEquiv p).symm.toEquiv, fun x => map_neg (ZMod.finEquiv p).symm x⟩,
    fun _ => (fun _ => false, 0))⟩
  Qry := RingQry B w
  Ans := Bits w
  answer ω
    | .add a b => ringOp ω.1 (· + ·) a b
    | .mul a b => ringOp ω.1 (· * ·) a b
    | .one => lab w ω.1 1
    | .key j .mask => (ω.2 j).1
    | .key j .offset => bitsOfNat w (ω.2 j).2

/-! ## The cost bound, beside `Prog.Bounded` -/

namespace Prog

variable {α : Type}

/-- The number of rounds that contain a `mul`, on the path taken when queries are answered by `f`. -/
def mulDepth (f : (ringRO B w p).Qry → (ringRO B w p).Ans) : Prog (ringRO B w p) α → ℕ
  | ret _ => 0
  | round qs k => (if qs.any RingQry.isMul then 1 else 0) + mulDepth f (k (qs.map f))

/-- The number of key queries on the path taken. -/
def keyWork (f : (ringRO B w p).Qry → (ringRO B w p).Ans) : Prog (ringRO B w p) α → ℕ
  | ret _ => 0
  | round qs k => qs.countP RingQry.isKey + keyWork f (k (qs.map f))

/-- The number of ring queries (`add`, `mul`, `one`) on the path taken. -/
def ringWork (f : (ringRO B w p).Qry → (ringRO B w p).Ans) : Prog (ringRO B w p) α → ℕ
  | ret _ => 0
  | round qs k => qs.countP (fun q => !q.isKey) + ringWork f (k (qs.map f))

/-- **`(D, Q, L)`-bounded in the ring model**: at most `D` rounds with a multiplication, `Q` key queries and `L` ring
queries, against every primitive. -/
def RingBounded (D Q L : ℕ) (P : Prog (ringRO B w p) α) : Prop :=
  ∀ ω : (ringRO B w p).Ω, P.mulDepth ((ringRO B w p).answer ω) ≤ D ∧ P.keyWork ((ringRO B w p).answer ω) ≤ Q ∧
    P.ringWork ((ringRO B w p).answer ω) ≤ L

end Prog

/-! ## The audit game over the ring bound -/

/-- `SimResponder` with `RingBounded D Q L` in place of `Bounded D Q`. -/
abbrev SimResponderOn (sch : Scheme (ringRO B w p)) (S D Q L k : ℕ) : Type :=
  SimResponderBy sch S k (Prog.RingBounded D Q L)

/-- `SeqResponder` with `RingBounded D Q L` in place of `Bounded D Q`. -/
abbrev SeqResponderOn (sch : Scheme (ringRO B w p)) (S D Q L : ℕ) : Type :=
  SeqResponderBy sch (fun _ => S) (Prog.RingBounded D Q L)

/-- `simAccepts` for a `SimResponderOn`. -/
def simAcceptsOn (sch : Scheme (ringRO B w p)) {S D Q L k : ℕ} (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SimResponderOn sch S D Q L k) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  (A₂.prog W pp (A₁ W pp C ω) x.2.2).run ((ringRO B w p).answer ω) = fun j => C (x.2.2 j)

/-- `seqAccepts` for a `SeqResponderOn`. -/
def seqAcceptsOn (sch : Scheme (ringRO B w p)) {S D Q L : ℕ} (k : ℕ) (W : Bits sch.n) (A₁ : Preprocessor sch S)
    (A₂ : SeqResponderOn sch S D Q L) (x : Outcome sch k) : Prop :=
  let ω := x.1
  let pp := (sch.enc W ω x.2.1).1
  let C := (sch.enc W ω x.2.1).2
  seqAnswers ((ringRO B w p).answer ω) (A₂.step W pp) 0 (List.ofFn x.2.2) (A₁ W pp C ω) =
    List.ofFn fun j => C (x.2.2 j)

/-- **`AuditSecure` over the ring bound**: every `S`-bit preprocessing and every `(D, Q, L)`-bounded responder is
accepted with probability at most `δ + ε`. -/
def AuditSecureOn (sch : Scheme (ringRO B w p)) : Reveal → (S D Q L k : ℕ) → (δ ε : ℝ≥0∞) → Prop
  | .simultaneous, S, D, Q, L, k, δ, ε =>
      ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SimResponderOn sch S D Q L k),
        pr (simAcceptsOn sch W A₁ A₂) ≤ δ + ε
  | .sequential, S, D, Q, L, k, δ, ε =>
      ∀ (W : Bits sch.n) (A₁ : Preprocessor sch S) (A₂ : SeqResponderOn sch S D Q L),
        pr (seqAcceptsOn sch k W A₁ A₂) ≤ δ + ε

/-- **The requirement over the ring bound**: `Meets`' conjuncts, with `AuditSecureOn` in place of `AuditSecure`. -/
def MeetsOn (sch : Scheme (ringRO B w p)) (reveal : Reveal) (D Q L k : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    AuditSecureOn sch reveal (sch.stateBound Params.ρ) D Q L k (ENNReal.ofReal Params.δ) ε

/-- **The requirement with `lam` bits of slack per block**: `Meets`, except that the adversary's state is
`⌊ρ·|C|⌋ + lam·B` bits. -/
def MeetsWithSlack {M : IdealModel} (sch : Scheme M) (reveal : Reveal) (lam D Q k : ℕ) (ε : ℝ≥0∞) : Prop :=
  ε ≤ Params.εMax ∧ sch.Correct ∧ sch.CodeHoldsW ∧ sch.SpaceBound Params.expansion ∧
    AuditSecure sch reveal (sch.stateBound Params.ρ + lam * sch.B) D Q k (ENNReal.ofReal Params.δ) ε

/-! ## P2 in the ring model -/

/-- `sq` on elements, read through the labeling: `a²`, negated when `a`'s label is odd. -/
def sqE (φ : OddLabeling p) (a : ZMod p) : ZMod p :=
  if (φ.1 a : ℕ) % 2 = 1 then -(a * a) else a * a

/-- `ρ`, a right inverse of `sqE` wherever one exists. `sqE` is a bijection when `p` is a prime `≡ 3 mod 4`. -/
noncomputable def rhoE (φ : OddLabeling p) : ZMod p → ZMod p := Function.invFun (sqE φ)

/-- `σ_K` on labels, local: `r ⊕ K` when that is below `p`, else `r`. -/
def sigmaK (p : ℕ) {w : ℕ} (K r : Bits w) : Bits w :=
  if bitsVal (fun i => xor (r i) (K i) : Bits w) < p then (fun i => xor (r i) (K i)) else r

/-- Negation on labels, local: `p − a` for a label `a ≠ 0`, and `0` for `0`. -/
def negL (p : ℕ) {w : ℕ} (a : Bits w) : Bits w :=
  if bitsVal a = 0 then a else bitsOfNat w (p - bitsVal a)

/-- `sq` on labels, local once a `mul` query has returned `s`, the label of `a²`: `s`, negated when `a` is odd. -/
def sqFix (p : ℕ) {w : ℕ} (a s : Bits w) : Bits w :=
  if bitsVal a % 2 = 1 then negL p s else s

/-- The `i`-th answer of a round (all-zero past the end). -/
def nthAns {w : ℕ} (as : List (Bits w)) (i : ℕ) : Bits w := as.getD i fun _ => false

/-- **P2's per-block decoder as a program**: `m_j = sq(σ_K(sq(c_j))) − t_j`. Round 1 asks for both keys and `c²`;
round 2 for `u²`, where `u = σ_K(sq(c))`; round 3 adds `−t`. Two rounds contain a multiplication. The output is the
label read mod `2^b`. -/
def decBlock (B w p b : ℕ) [NeZero p] (j : Fin B) (c : Bits w) : Prog (ringRO B w p) (Bits b) :=
  .round [RingQry.key j .mask, RingQry.key j .offset, RingQry.mul c c] fun r₁ =>
    let u := sigmaK p (nthAns r₁ 0) (sqFix p c (nthAns r₁ 2))
    .round [RingQry.mul u u] fun r₂ =>
      .round [RingQry.add (sqFix p u (nthAns r₂ 0)) (negL p (nthAns r₁ 1))] fun r₃ =>
        .ret (bitsOfNat b (bitsVal (nthAns r₃ 0)))

/-- Block `j`'s code block `ρ(σ_K(ρ(m + t)))`, with `m` the payload label's element and `(K, τ)` its keys. The
trusted encoder sees the labeling whole, so it takes the roots. -/
noncomputable def encBlock (φ : OddLabeling p) (K : Bits w) (τ : Fin p) (v : Bits w) : Bits w :=
  lab w φ (rhoE φ (eltD φ (sigmaK p K (lab w φ (rhoE φ (eltD φ v + φ.1.symm τ))))))

/-- **P2 in the ring model**, with `b` payload bits per block zero-extended to a `w`-bit label (as in `m1p`). No `pp`
and no coins; decoding runs `decBlock` on every block. The definitions are total. The side conditions (`p` a prime
`≡ 3 mod 4`, `2^b ≤ p ≤ 2^w`) are hypotheses of the theorems that need them. -/
noncomputable def p2ring (B w p b : ℕ) [NeZero B] [NeZero p] : Scheme (ringRO B w p) where
  n := B * b
  t := 0
  B := B
  ℓ := w
  B_pos := Nat.pos_of_ne_zero (NeZero.ne B)
  Coins := Unit
  enc W ω _ := (Fin.elim0, fun j =>
    encBlock ω.1 (ω.2 j).1 (ω.2 j).2 (bitsOfNat w (bitsVal ((codeEquiv B b).symm W j))))
  dec _ C f := codeEquiv B b fun j => (decBlock B w p b j (C j)).run f

set_option exponentiation.threshold 20000 in
/-- P2's registered 16,448-bit modulus is nonzero, so the ring model is defined at it. -/
instance instNeZeroP16448 : NeZero (2 ^ 16448 - 21065) := ⟨by decide +kernel⟩

end Pous
