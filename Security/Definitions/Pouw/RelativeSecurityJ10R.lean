import Definitions.Pouw.RelativeSecurity

/-!
# Relative security at call scope under J10-R: the call design its floors read (trusted)

Theorem K's design (`DBits`, `dPrefix`, `dFrame`) puts the noise `U` on the rows, shared across blocks, and a fresh
coin on every entry of the columns. J10-R's noise is H1's, one SHA-512 stream per call row and per call column, seeded
once a call, so a call's coins are shared by the whole call on both sides. This is that call as one stage of `n` rows
(the call's `M`), `n'` columns (its `N`) and `B` segments of `L` (`KS` segments of 32), as `segment.call_words`
computes it:

* the call's bits (`JBits`): row `i`'s dither bits `b(i, t, j)`, fresh a segment and read by every column; column `l`'s
  coin bits `c(t, j, l)`, fresh a segment and read by every row; and the 8 bits of each of column `l`'s `L` bytes of
  `U` (`β(l, j, ·)`), the same in every segment;
* the operands, int8 as served (`jS8`): `X(i, tL + j) = s8 (A + b(i, t, j))` on the rows and
  `W'(l, tL + j) = s8 (W + c(t, j, l) + U(l, j))` on the columns, `U(l, j) = −64 + (byte mod 129)` (`jU`), on the
  committed `A` (`aq`) and the registered `W` (`wq`);
* the committed prefix words `T_t(i, l) = Σ_{t' ≤ t} Σ_j X(i, t'L + j) W'(l, t'L + j)` mod `2^32` (`jIncr`,
  `jPrefix`): a segment's coin × coin part is `b^(t) N^(t)ᵀ` with `N = c + U`, one `⟨n, L, n'⟩` product over the whole
  call, not `n n' / L²` independent `⟨L, L, L⟩` ones;
* cell-block `(i, l, t)`'s frame (`jFrame`): its `x` bits are row `i`'s dither bits of segment `t`, its `y` bits column
  `l`'s coin bits of segment `t`, so mod 2 its target is `⟨x, y⟩` plus terms blind to one side, with no plane weight;
* charging purity at those frames (`SlpG.ChargePureJ`; across draws, `SlpG.ChargePureAtJ`), one-sided products per draw
  (`OneSidedAtJ`), a round's coins (`JStage`; overlapped, `OJBits`, `OJDraws`, `OJSpot`) and an adversary's window run
  on the call design (`JStagedRun`, `JRunMap`; `JOverlapRun`, `JORunMap`).

The digit words `O'` and `D_d` (`U · A_d`, over the digit coins `z`) are not in the design, as no digit word is in
Theorem K's.
-/

namespace Pouw.RelativeSecurity

open Finset

section J10R

/-- The int8 wrap on `Z/2³²`: `x` sign-extended from its low byte, `((x + 128) mod 256) − 128`. -/
def jS8 (x : ZMod (2 ^ 32)) : ZMod (2 ^ 32) := (((x.val + 128) % 256 : ℕ) : ZMod (2 ^ 32)) - 128

/-- **`U`'s value from its byte**, H1's keyed word in ±64: `−64 + (byte mod 129)`, the byte's 8 bits least
significant first (`h1tile.u129`). -/
def jU (β : Fin 8 → ZMod (2 ^ 32)) : ZMod (2 ^ 32) :=
  (((∑ p : Fin 8, 2 ^ p.val * (β p).val) % 129 : ℕ) : ZMod (2 ^ 32)) - 64

/-- **A call's bits under J10-R**: row `i`'s dither bits `b(i, t, j)` (segment `t`, position `j < L`), column `l`'s
coin bits `c(t, j, l)`, and the bits `β(l, j, p)` of column `l`'s byte `j` of `U`, which every segment reads. The rows'
bits are the one side, the columns' the other. -/
abbrev JBits (n L B n' : ℕ) : Type :=
  (Fin n × Fin B × Fin L) ⊕ ((Fin B × Fin L × Fin n') ⊕ (Fin n' × Fin L × Fin 8))

variable {n L B n' : ℕ}

/-- The row operand `X(i, tL + j) = s8 (A(i, tL + j) + b(i, t, j))` (`A` in `aq`). -/
def jX (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32)) (i : Fin n) (t : Fin B) (j : Fin L)
    (z : JBits n L B n' → ZMod (2 ^ 32)) : ZMod (2 ^ 32) :=
  jS8 (aq i t j + z (.inl (i, t, j)))

/-- The column operand `W'(l, tL + j) = s8 (W(l, tL + j) + c(t, j, l) + U(l, j))` (`W` in `wq`). -/
def jW (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (t : Fin B) (j : Fin L) (l : Fin n')
    (z : JBits n L B n' → ZMod (2 ^ 32)) : ZMod (2 ^ 32) :=
  jS8 (wq t j l + z (.inr (.inl (t, j, l))) + jU fun p => z (.inr (.inr (l, j, p))))

/-- Cell-block `(i, l, t)`'s target: segment `t`'s increment `Σ_j X(i, tL + j) W'(l, tL + j)` of the committed prefix
word at cell `(i, l)`. -/
def jIncr (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32)) (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (i : Fin n)
    (l : Fin n') (t : Fin B) : (JBits n L B n' → ZMod (2 ^ 32)) → ZMod (2 ^ 32) :=
  fun z => ∑ j, jX aq i t j z * jW wq t j l z

/-- The committed prefix word `T_t(i, l) = Σ_{t' ≤ t} Σ_j X(i, t'L + j) W'(l, t'L + j)` mod `2^32`. -/
def jPrefix (aq : Fin n → Fin B → Fin L → ZMod (2 ^ 32)) (wq : Fin B → Fin L → Fin n' → ZMod (2 ^ 32)) (i : Fin n)
    (l : Fin n') (t : ℕ) : (JBits n L B n' → ZMod (2 ^ 32)) → ZMod (2 ^ 32) :=
  fun z => ∑ t' : Fin B, if t'.val ≤ t then jIncr aq wq i l t' z else 0

/-- **Cell-block `(i, l, t)`'s frame**: its `x` bits are row `i`'s dither bits of segment `t`, its `y` bits column `l`'s
coin bits of segment `t`. -/
def jFrame (c : DCells n n' B) : Fin L ⊕ Fin L → JBits n L B n' :=
  Sum.elim (fun j => .inl (c.1, c.2.2, j)) fun j => .inr (.inl (c.2.2, j, c.2.1))

/-- **Charging purity of a call's run**: no product is two-sided, mod 2 through `φ`, at the frames (`jFrame`) of two
cell-blocks whose words read it with odd coefficients. A product two-sided at one cell-block's frame reads that row's
and that column's coins of one segment; a product shared by two rows or two columns of a segment is outside it. -/
def SlpG.ChargePureJ {Ar : Type} [CommRing Ar] (φ : Ar →+* ZMod 2) {N : ℕ} (P : SlpG (JBits n L B n') Ar N)
    (word : DCells n n' B → Fin N) : Prop :=
  ∀ c c' i, c ≠ c' → φ (P.coef (word c) i) ≠ 0 → TwoSidedAt (jFrame c) φ (P.prod i) →
    φ (P.coef (word c') i) ≠ 0 → ¬TwoSidedAt (jFrame c') φ (P.prod i)

end J10R

/-- A round's coins with `St` calls: each call's bits, then the check's call and cell-block. -/
abbrev JStage (St n L B n' : ℕ) : Type := Paths (Finset (JBits n L B n')) St × (Fin St × DCells n n' B)

/-- **An adversary's window run on the call design**, as the analysis reads it: its staged play (per round and call,
the error table fixed by the call's operand commitment before its draw), the per-round binding and zero-knowledge break
events, the noise reads `V`, and along every play `x` each round's calls as straight-line programs over `Ar` on the
call's bits (`prog x w s`), on the committed `A` (`aq x w s`) and the registered `W` (`wq x w s`), with the slot of
each cell's committed prefix word (`word x w s`). -/
structure JStagedRun (R St n L B n' : ℕ) (Ar : Type) [CommRing Ar] where
  play : StagedPlay (DCells n n' B) (JBits n L B n') Ar St R
  Bind : Fin R → Paths (JStage St n L B n') R → Prop
  ZK : Fin R → Paths (JStage St n L B n') R → Prop
  V : Paths (JStage St n L B n') R → ℝ
  N : Paths (JStage St n L B n') R → Fin R → Fin St → ℕ
  prog : ∀ x w s, SlpG (JBits n L B n') Ar (N x w s)
  aq : Paths (JStage St n L B n') R → Fin R → Fin St → Fin n → Fin B → Fin L → Ar
  wq : Paths (JStage St n L B n') R → Fin R → Fin St → Fin B → Fin L → Fin n' → Ar
  word : ∀ x w s, DCells n n' B → Fin (N x w s)

/-- **The analysis's map from adversaries to their window runs on the call design.** -/
abbrev JRunMap (Mc : Machine) (S : Service) (R St n L B n' : ℕ) (Ar : Type) [CommRing Ar] : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → JStagedRun R St n L B n' Ar

/-- A round's bits with overlapping calls: call `t`'s bits, of shape `(n t, L, B t, n' t)`. -/
abbrev OJBits (St : ℕ) (n : Fin St → ℕ) (L : ℕ) (B n' : Fin St → ℕ) : Type := Σ t, JBits (n t) L (B t) (n' t)

/-- A round's draws with overlapping calls: call `t`'s bits. -/
abbrev OJDraws (St : ℕ) (n : Fin St → ℕ) (L : ℕ) (B n' : Fin St → ℕ) : Type :=
  ∀ t, Finset (JBits (n t) L (B t) (n' t))

/-- A round's coins with overlapping calls: the check's cell-block, among every call's, and every call's draw. -/
abbrev OJSpot (St : ℕ) (n : Fin St → ℕ) (L : ℕ) (B n' : Fin St → ℕ) : Type :=
  OCells St n n' B × OJDraws St n L B n'

section J10ROverlap

variable {St : ℕ} {Ar : Type} [CommRing Ar] {n B n' : Fin St → ℕ} {L : ℕ}

/-- **One-sided products per call**: at every bit point of the other calls, one-sided on call `t`'s bits (its rows'
against its columns'), for every call `t`. -/
def OneSidedAtJ (π : (OJBits St n L B n' → Ar) → Ar) : Prop :=
  ∀ (W : OJDraws St n L B n') (t : Fin St), OneSided fun z => π (mergeAt W t z)

/-- **Charging purity across calls**: at every bit point of the draws, no product is two-sided at the frames
(`jFrame`) of two cell-blocks, of any calls, whose words read it with odd coefficients (mod 2 through `φ`). -/
def SlpG.ChargePureAtJ (φ : Ar →+* ZMod 2) {N : ℕ} (P : SlpG (OJBits St n L B n') Ar N)
    (word : OCells St n n' B → Fin N) : Prop :=
  ∀ (W : OJDraws St n L B n') (c c' : OCells St n n' B) i, c ≠ c' → φ (P.coef (word c) i) ≠ 0 →
    TwoSidedAt (jFrame c.2) φ (fun z => P.prod i (mergeAt W c.1 z)) → φ (P.coef (word c') i) ≠ 0 →
      ¬TwoSidedAt (jFrame c'.2) φ fun z => P.prod i (mergeAt W c'.1 z)

end J10ROverlap

/-- **An adversary's window run with overlapping calls**, as the analysis reads it: its play over the round's draws
(per round, the error tables of every call fixed before the round's first draw), the per-round binding and
zero-knowledge break events, the noise reads `V`, and along every play `x` each round's run as one straight-line
program over `Ar` on every call's bits (`prog x w`), on each call's committed `A` and registered `W` (`aq x w t`,
`wq x w t`), with the slot of each cell's committed prefix word (`word x w`). -/
structure JOverlapRun (R St : ℕ) (n : Fin St → ℕ) (L : ℕ) (B n' : Fin St → ℕ) (Ar : Type) [CommRing Ar] where
  play : OvPlay (fun t => DCells (n t) (n' t) (B t)) (fun t => JBits (n t) L (B t) (n' t)) Ar R
  Bind : Fin R → Paths (OJSpot St n L B n') R → Prop
  ZK : Fin R → Paths (OJSpot St n L B n') R → Prop
  V : Paths (OJSpot St n L B n') R → ℝ
  N : Paths (OJSpot St n L B n') R → Fin R → ℕ
  prog : ∀ x w, SlpG (OJBits St n L B n') Ar (N x w)
  aq : Paths (OJSpot St n L B n') R → Fin R → ∀ t, Fin (n t) → Fin (B t) → Fin L → Ar
  wq : Paths (OJSpot St n L B n') R → Fin R → ∀ t, Fin (B t) → Fin L → Fin (n' t) → Ar
  word : ∀ x w, OCells St n n' B → Fin (N x w)

/-- **The analysis's map from adversaries to their window runs with overlapping calls.** -/
abbrev JORunMap (Mc : Machine) (S : Service) (R St : ℕ) (n : Fin St → ℕ) (L : ℕ) (B n' : Fin St → ℕ) (Ar : Type)
    [CommRing Ar] : Type 1 :=
  ∀ (Ω' : Type) [Fintype Ω'], Mc.Alg (S.Seed × Ω') (S.Com × S.Ans) → Ω' → JOverlapRun R St n L B n' Ar

end Pouw.RelativeSecurity
