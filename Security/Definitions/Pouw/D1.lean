import Definitions.Pouw.Game.Defs

/-!
# Protocol D1: the [KW] transcript with low-rank integer noise (trusted; §3.2, §4.4, §6 D1)

Parameters (the `tt-attacks` and `llm-shapes` threads may change every one of them):
* `r`, the noise rank; `d`, the checked inner depth (TT asks `0 < d ≤ r`);
* the noise derivation: unit `u`'s weight factors `(F_L, F_R)` are an expansion `expWt` of `H`'s answers to the
  queries `qWt s u w(u)`, and its activation factors `(E_L, E_R)` an expansion `expAct` of `H`'s answers to the
  queries `qAct s u A_u w(u)` and to `qWt s u w(u)`. Sharing is a property of an instance (`GroupShared`): the
  weight noise per (epoch, weight), and the activations' right factor per (epoch, weight, group of `G` indices).
  The factor distribution is the push-forward of the uniform oracle through `expAct`, `expWt`;
* `W_ref`, from the layout only (generic, the milestone's included).

The named milestone instance is `D1.m1` (`Definitions/Pouw/M1.lean`).

Noise `E = E_L·E_R` (`m × k`, rank `≤ r`), `F = F_L·F_R` (`k × n`); noised operands `A' = A + E`, `B' = B + F`.
Checked values: the running partial sums of `A'·B'` at inner depths `d, 2d, …`, the last block capped at `k`.
`H_ref` computes them with an int32 accumulator, and decodes `C = C' − (A·F_L)·F_R − E_L·(E_R·B')` in int32.
-/

namespace Pouw.D1

open Finset

/-- D1's parameters. -/
structure Params (Q R S : Type) where
  r : ℕ
  d : ℕ
  qAct : S → ℕ → Mat → ℕ → List Q
  qWt : S → ℕ → ℕ → List Q
  expAct : Shape → List R → List R → Mat × Mat
  expWt : Shape → List R → Mat × Mat
  Wref : UnitLayout → ℕ → ℕ

/-- The number of checked blocks, `⌈k/d⌉` (`0` when `d = 0`). -/
def nBlocks (d k : ℕ) : ℕ := (k + d - 1) / d

/-- Block `t` ends at inner index `min ((t + 1)·d, k)`. -/
def blockEnd (d k t : ℕ) : ℕ := min ((t + 1) * d) k

namespace Params

variable {Q R S : Type} (P : Params Q R S)

/-- Unit `u`'s activation factors `(E_L, E_R)`. -/
def actF (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat × Mat :=
  P.expAct (U.layout.shape u) ((P.qAct s u A (U.layout.wid u)).map H) ((P.qWt s u (U.layout.wid u)).map H)

/-- Unit `u`'s weight factors `(F_L, F_R)`. -/
def wtF (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat × Mat :=
  P.expWt (U.layout.shape u) ((P.qWt s u (U.layout.wid u)).map H)

/-- `A' = A + E_L·E_R`. -/
def noisedAct (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  A + mm P.r (P.actF H s U u A).1 (P.actF H s U u A).2

/-- `B' = B_{w(u)} + F_L·F_R`. -/
def noisedWt (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat :=
  U.weight (U.layout.wid u) + mm P.r (P.wtF H s U u).1 (P.wtF H s U u).2

/-- The checked values: block `t` is the running partial sum of `A'·B'` up to `blockEnd d k t`, in exact integers. -/
def Y (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mm (blockEnd P.d (U.layout.shape u).k t) (P.noisedAct H s U u A) (P.noisedWt H s U u)

/-- The compared entries: blocks `t < ⌈k/d⌉`, rows `i < m`, columns `j < n`. -/
def checkIdx (sh : Shape) : Finset (ℕ × ℕ × ℕ) :=
  range (nBlocks P.d sh.k) ×ˢ range sh.m ×ˢ range sh.n

/-- `H_ref`'s noised activations in int32: `wrap32 (A + E_L·E_R)`. -/
def refAct (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  fun i l => wrap32 (A i l + mmW P.r (P.actF H s U u A).1 (P.actF H s U u A).2 i l)

/-- `H_ref`'s noised weights in int32: `wrap32 (B + F_L·F_R)`. -/
def refWt (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat :=
  fun l j => wrap32 (U.weight (U.layout.wid u) l j + mmW P.r (P.wtF H s U u).1 (P.wtF H s U u).2 l j)

/-- `H_ref`'s checked values: the int32 accumulator of `A'·B'` read at each block end. -/
def refChecked (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmW (blockEnd P.d (U.layout.shape u).k t) (P.refAct H s U u A) (P.refWt H s U u)

/-- `H_ref`'s useful output: the int32 decode `C' − (A·F_L)·F_R − E_L·(E_R·B')`, every step wrapping. -/
def refOut (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  let k := (U.layout.shape u).k
  let E : Mat × Mat := P.actF H s U u A
  let F : Mat × Mat := P.wtF H s U u
  fun i j => wrap32 (wrap32 (mmW k (P.refAct H s U u A) (P.refWt H s U u) i j - mmW P.r (mmW k A F.1) F.2 i j)
    - mmW P.r E.1 (mmW k E.2 (P.refWt H s U u)) i j)

/-- D1 as a protocol. -/
def toProtocol : Protocol Q R S where
  Y := P.Y
  checkIdx := P.checkIdx
  refChecked := P.refChecked
  refOut := P.refOut
  Wref := P.Wref

/-- The factor distribution keeps every noise entry in `[−63, 63]`, for every shape and every oracle answer. -/
def NoiseRange : Prop :=
  (∀ (sh : Shape) (a b : List R), ∀ i < sh.m, ∀ l < sh.k,
      InNoise (mm P.r (P.expAct sh a b).1 (P.expAct sh a b).2 i l)) ∧
  (∀ (sh : Shape) (ans : List R), ∀ l < sh.k, ∀ j < sh.n,
      InNoise (mm P.r (P.expWt sh ans).1 (P.expWt sh ans).2 l j))

/-- Group sharing (tt-attacks V4, replacing per-epoch sharing of `E_R`, which tt-attacks' V3 breaks): the
weight noise `(F_L, F_R)` depends only on the salt and the weight, and the activations' right factor `E_R` only on
the salt, the weight and the group `⌊u/G⌋`, for every oracle. Part of `IsMilestone`; no generic theorem assumes it. -/
def GroupShared (G : ℕ) : Prop :=
  (∀ (s : S) (u u' w : ℕ), P.qWt s u w = P.qWt s u' w) ∧
  ∀ (H : Q → R) (s : S) (sh : Shape) (u u' : ℕ) (A A' : Mat) (w : ℕ), u / G = u' / G →
    (P.expAct sh ((P.qAct s u A w).map H) ((P.qWt s u w).map H)).2 =
      (P.expAct sh ((P.qAct s u' A' w).map H) ((P.qWt s u' w).map H)).2

/-- At most `c` oracle queries per unit for its noise. -/
def QueryBound (c : ℕ) : Prop :=
  ∀ (s : S) (u : ℕ) (A : Mat) (w : ℕ), (P.qAct s u A w).length + (P.qWt s u w).length ≤ c

/-- The milestone's parameters: rank 16, checked depth `d = r = 16`, the weight noise shared per (epoch, weight) and
`E_R` per (epoch, weight, group of 16 consecutive indices) (W4). `W_ref` is not fixed here. -/
def IsMilestone : Prop := P.r = 16 ∧ P.d = 16 ∧ P.GroupShared 16

end Params

end Pouw.D1
