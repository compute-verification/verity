module

public import Definitions.Pouw.Game.Defs
public import Definitions.Pouw.NCP.Defs
public import Mathlib.Order.Interval.Finset.Nat

/-! The degenerate-input barrier's definitions: cheap points, an abstract product cost model and three protocol classes. -/

@[expose] public section

namespace Pouw.Barrier

open Finset

variable {Q R S : Type}

def CheapPoint (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (A : Adversary CM)
    (c₀ : ℝ) (q : ℕ) : Prop :=
  U.InDomain D ∧ 0 ≤ c₀ ∧ q ≤ 2 ^ 64 ∧ A.Bounded U c₀ q ∧
    ∀ (H : Q → R) (s : S), correctSet P U H s (A.trace U H s) = range U.layout.N ∧ ¬ A.Breaks U H s

inductive MExpr where
  | act : MExpr
  | wt : MExpr
  | noise : ℕ → MExpr
  | const : Mat → MExpr
  | reg : ℕ → MExpr
  | add : MExpr → MExpr → MExpr
  | smul : ℤ → MExpr → MExpr
  | gemm : Shape → ℕ → MExpr → MExpr → MExpr

namespace MExpr

def eval (A B : Mat) (nz rg : ℕ → Mat) : MExpr → Mat
  | act => A
  | wt => B
  | noise i => nz i
  | const M => M
  | reg i => rg i
  | add x y => fun i j => eval A B nz rg x i j + eval A B nz rg y i j
  | smul c x => fun i j => c * eval A B nz rg x i j
  | gemm sh lo x y => fun i j =>
      if i < sh.m ∧ j < sh.n then ∑ l ∈ Ico lo (lo + sh.k), eval A B nz rg x i l * eval A B nz rg y l j else 0

def cost : MExpr → ℕ
  | add x y => cost x + cost y
  | smul _ x => cost x
  | gemm sh _ x y => sh.Wmm + cost x + cost y
  | _ => 0

end MExpr

structure Plan where
  act : Option Mat
  nReg : ℕ
  reg : ℕ → MExpr
  out : MExpr

def Plan.cost (p : Plan) : ℕ := ∑ t ∈ range p.nReg, (p.reg t).cost + p.out.cost

def Plan.regs (p : Plan) (A B : Mat) (nz : ℕ → Mat) (t : ℕ) : Mat :=
  if t < p.nReg then (p.reg t).eval A B nz (fun i => if _ : i < t then p.regs A B nz i else 0) else 0
termination_by t

abbrev NoiseSrc (Q R S : Type) : Type := (Q → R) → S → Workload → ℕ → Mat → ℕ → Mat

def Plan.commit (p : Plan) (N : NoiseSrc Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Mat) (u : ℕ) :
    Commit :=
  let A := p.act.getD (act u)
  let B := U.fixed (U.layout.fixedId u)
  let nz := N H s U u A
  ⟨A, p.regs A B nz, p.out.eval A B nz (p.regs A B nz)⟩

/-- Hash calls are free: a program's oracle use is only through the noise atoms, and it makes no counted calls. -/
def gemmModel (N : NoiseSrc Q R S) : CostModel Q R S where
  Prog := Layout → ℕ → Option Plan
  run prog H U act _ s u := (prog U.layout u).map fun p => p.commit N H s U act u
  cost prog _ U _ _ _ := ∑ u ∈ range U.layout.N, ((((prog U.layout u).map Plan.cost).getD 0 : ℕ) : ℝ)
  calls _ _ _ _ _ _ := 0

abbrev wtOf (U : Workload) (u : ℕ) : Mat := U.fixed (U.layout.fixedId u)

/-- `M` on the `m × n` corner, `0` elsewhere; `H_ref`'s outputs are stated on the corner so that a plan can reproduce
them exactly (`AdmitsRef`). -/
def corner (m n : ℕ) (M : Mat) : Mat := fun i j => if i < m ∧ j < n then M i j else 0

def addProto (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) : Protocol Q R S where
  Y H s U u A := fun _ => mm (U.layout.shape u).k (A + N H s U u A 0) (wtOf U u + N H s U u A 1)
  checkIdx sh := {0} ×ˢ (range sh.m ×ˢ range sh.n)
  refChecked H s U u A := fun t =>
    if t = 0 then corner (U.layout.shape u).m (U.layout.shape u).n
      (mm (U.layout.shape u).k (A + N H s U u A 0) (wtOf U u + N H s U u A 1)) else 0
  refOut _ _ U u A := corner (U.layout.shape u).m (U.layout.shape u).n (mm (U.layout.shape u).k A (wtOf U u))
  Wref := W

def relProd (N : NoiseSrc Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (i : ℕ) : Mat :=
  mm (U.layout.shape u).k (A + N H s U u A (2 * i)) (wtOf U u + N H s U u A (2 * i + 1))

def relProto (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) : Protocol Q R S where
  Y H s U u A := relProd N H s U u A
  checkIdx sh := range t ×ˢ (range sh.m ×ˢ range sh.n)
  refChecked H s U u A := fun i =>
    if i < t then corner (U.layout.shape u).m (U.layout.shape u).n (relProd N H s U u A i) else 0
  refOut H s U u A := corner (U.layout.shape u).m (U.layout.shape u).n
    (fun a b => ∑ i ∈ range t, c i * relProd N H s U u A i a b)
  Wref := W

def RelNoise (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) : Prop :=
  ∑ i ∈ range t, c i = 1 ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat),
      (∀ a b, ∑ i ∈ range t, c i * N H s U u A (2 * i) a b = 0) ∧
      (∀ a b, ∑ i ∈ range t, c i * N H s U u A (2 * i + 1) a b = 0) ∧
      (∀ a b, ∑ i ∈ range t, c i * mm (U.layout.shape u).k (N H s U u A (2 * i)) (N H s U u A (2 * i + 1)) a b = 0)

def segProto (L d : ℕ) (α β : Mat → Mat) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) : Protocol Q R S where
  Y H s U u A := fun t => mm ((t + 1) * d) (α A + N H s U u A 0) (β (wtOf U u) + N H s U u A 1)
  checkIdx sh := range L ×ˢ (range sh.m ×ˢ range sh.n)
  refChecked H s U u A := fun t =>
    if t < L then corner (U.layout.shape u).m (U.layout.shape u).n
      (mm ((t + 1) * d) (α A + N H s U u A 0) (β (wtOf U u) + N H s U u A 1)) else 0
  refOut H s U u A := corner (U.layout.shape u).m (U.layout.shape u).n
    (mm (L * d) (α A + N H s U u A 0) (β (wtOf U u) + N H s U u A 1))
  Wref := W

def cancelNoise (e f r : (Q → R) → S → Workload → ℕ → Mat → Mat) : NoiseSrc Q R S := fun H s U u A i =>
  let k := (U.layout.shape u).k
  match i with
  | 0 => e H s U u A
  | 1 => f H s U u A
  | 2 => NCP.E2 k (e H s U u A) (r H s U u A)
  | 3 => NCP.F2 k (r H s U u A) (f H s U u A)
  | 4 => NCP.E3 k (e H s U u A) (r H s U u A)
  | 5 => NCP.F3 k (r H s U u A) (f H s U u A)
  | _ => 0

def stackNoise (k : ℕ) (Rm : Mat) (e f : (Q → R) → S → Workload → ℕ → Mat → Mat) : NoiseSrc Q R S :=
  fun H s U u A i =>
    if i = 0 then NCP.stackX k 0 (e H s U u A) Rm else if i = 1 then NCP.stackY k 0 Rm (f H s U u A) else 0

end Pouw.Barrier
