module

public import Definitions.Pouw.Game.Defs

/-! Protocol D1, the [KW] transcript with low-rank integer noise, whose named milestone instance is `D1.m1`. -/

@[expose] public section

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
  Wref : Layout → ℕ → ℕ

/-- The number of checked blocks, `⌈k/d⌉` (`0` when `d = 0`). -/
def nSegments (d k : ℕ) : ℕ := (k + d - 1) / d

/-- Block `t` ends at inner index `min ((t + 1)·d, k)`. -/
def blockEnd (d k t : ℕ) : ℕ := min ((t + 1) * d) k

namespace Params

variable {Q R S : Type} (P : Params Q R S)

def actF (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat × Mat :=
  P.expAct (U.layout.shape u) ((P.qAct s u A (U.layout.fixedId u)).map H) ((P.qWt s u (U.layout.fixedId u)).map H)

def wtF (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat × Mat :=
  P.expWt (U.layout.shape u) ((P.qWt s u (U.layout.fixedId u)).map H)

def noisedAct (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  A + mm P.r (P.actF H s U u A).1 (P.actF H s U u A).2

def noisedWt (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat :=
  U.fixed (U.layout.fixedId u) + mm P.r (P.wtF H s U u).1 (P.wtF H s U u).2

def Y (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mm (blockEnd P.d (U.layout.shape u).k t) (P.noisedAct H s U u A) (P.noisedWt H s U u)

def checkIdx (sh : Shape) : Finset (ℕ × ℕ × ℕ) :=
  range (nSegments P.d sh.k) ×ˢ range sh.m ×ˢ range sh.n

def refAct (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  fun i l => wrap32 (A i l + mmW P.r (P.actF H s U u A).1 (P.actF H s U u A).2 i l)

def refWt (H : Q → R) (s : S) (U : Workload) (u : ℕ) : Mat :=
  fun l j => wrap32 (U.fixed (U.layout.fixedId u) l j + mmW P.r (P.wtF H s U u).1 (P.wtF H s U u).2 l j)

def refChecked (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Checked :=
  fun t => mmW (blockEnd P.d (U.layout.shape u).k t) (P.refAct H s U u A) (P.refWt H s U u)

def refOut (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) : Mat :=
  let k := (U.layout.shape u).k
  let E : Mat × Mat := P.actF H s U u A
  let F : Mat × Mat := P.wtF H s U u
  fun i j => wrap32 (wrap32 (mmW k (P.refAct H s U u A) (P.refWt H s U u) i j - mmW P.r (mmW k A F.1) F.2 i j)
    - mmW P.r E.1 (mmW k E.2 (P.refWt H s U u)) i j)

def toProtocol : Protocol Q R S where
  Y := P.Y
  checkIdx := P.checkIdx
  refChecked := P.refChecked
  refOut := P.refOut
  Wref := P.Wref

def NoiseRange : Prop :=
  (∀ (sh : Shape) (a b : List R), ∀ i < sh.m, ∀ l < sh.k,
      InNoise (mm P.r (P.expAct sh a b).1 (P.expAct sh a b).2 i l)) ∧
  (∀ (sh : Shape) (ans : List R), ∀ l < sh.k, ∀ j < sh.n,
      InNoise (mm P.r (P.expWt sh ans).1 (P.expWt sh ans).2 l j))

/-- Part of `IsMilestone`; no generic theorem assumes it. -/
def GroupShared (G : ℕ) : Prop :=
  (∀ (s : S) (u u' w : ℕ), P.qWt s u w = P.qWt s u' w) ∧
  ∀ (H : Q → R) (s : S) (sh : Shape) (u u' : ℕ) (A A' : Mat) (w : ℕ), u / G = u' / G →
    (P.expAct sh ((P.qAct s u A w).map H) ((P.qWt s u w).map H)).2 =
      (P.expAct sh ((P.qAct s u' A' w).map H) ((P.qWt s u' w).map H)).2

def QueryBound (c : ℕ) : Prop :=
  ∀ (s : S) (u : ℕ) (A : Mat) (w : ℕ), (P.qAct s u A w).length + (P.qWt s u w).length ≤ c

/-- `W_ref` is not fixed here. -/
def IsMilestone : Prop := P.r = 16 ∧ P.d = 16 ∧ P.GroupShared 16

end Params

end Pouw.D1
