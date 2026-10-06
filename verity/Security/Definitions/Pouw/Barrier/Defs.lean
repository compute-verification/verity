import Definitions.Pouw.Game.Defs
import Definitions.Pouw.NCP.Defs
import Mathlib.Order.Interval.Finset.Nat

/-!
# The degenerate-input barrier: definitions (trusted)

* `CheapPoint`: an admissible point of the game (an in-domain workload and an adversary, whose committed activations
  are its own choice) at which one program commits every unit correct at cost at most `c₀`, for every oracle and salt.
* `gemmModel N`: a small abstract cost model. A program assigns each unit a straight-line plan over matrix
  expressions whose atoms are the committed activations, the unit's weight, the protocol's noise matrices (hash calls
  are free) and known constants; sums and integer multiples are free, and a plain product of shape `(m, k, n)` over
  an inner index window of length `k` costs `m·k·n`. Plans are fixed by the layout (oblivious), so a constant is a
  function of the layout only.
* The protocol classes: `addProto` (class (a), one checked noised product with a separate decode), `relProto`
  (class (b), several checked products whose final values give the useful output through a known linear relation)
  and `segProto` (class (c), the running sums of one product at depth `d`, the last of which is the useful output).
-/

namespace Pouw.Barrier

open Finset

variable {Q R S : Type}

/-! ## Cheap points -/

/-- `(U, 𝒜)` is a **cheap point at cost `c₀`** in the game: `U` is in the domain (so its weights, like the
activations `𝒜` commits, are the adversary's choice within int7), `c₀ ≥ 0`, `𝒜` is within `(c₀, q)`, and every unit
of `U` is correct in `𝒜`'s transcript for every oracle and salt. A *degenerate input* is the special case where
`𝒜`'s committed activations or `U`'s weights make the useful output a known constant (`A = 0` or `B = 0`), so that
`c₀` is the cost of the checked values alone. -/
def CheapPoint (CM : CostModel Q R S) (P : Protocol Q R S) (D : Layout → Prop) (U : Workload) (A : Adversary CM)
    (c₀ : ℝ) (q : ℕ) : Prop :=
  U.InDomain D ∧ 0 ≤ c₀ ∧ q ≤ 2 ^ 64 ∧ A.Bounded U c₀ q ∧
    ∀ (H : Q → R) (s : S), correctSet P U H s (A.transcript U H s) = range U.layout.N

/-! ## The abstract product model -/

/-- Matrix expressions: the committed activations, the unit's weight, the protocol's noise matrix `i`, a known
constant, an earlier register of the same plan, a sum, an integer multiple, and a plain product
`gemm sh lo x y` over the inner window `[lo, lo + sh.k)`, masked to the `sh.m × sh.n` corner. -/
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

/-- The value of an expression, given the activations `A`, the weight `B`, the noise matrices `nz` and the
registers `rg`. -/
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

/-- The cost: `m·k·n` per plain product; atoms, sums and integer multiples cost nothing. -/
def cost : MExpr → ℕ
  | add x y => cost x + cost y
  | smul _ x => cost x
  | gemm sh _ x y => sh.Wmm + cost x + cost y
  | _ => 0

end MExpr

/-- A unit's plan: the activations it commits (`none`: the activation input; `some M`: the constant `M`), registers
`0, …, nReg − 1` (register `t` is checked block `t` and may read registers below `t`), and the useful output, which
may read every register. -/
structure Plan where
  act : Option Mat
  nReg : ℕ
  reg : ℕ → MExpr
  out : MExpr

/-- A plan's cost: every register's and the output's. -/
def Plan.cost (p : Plan) : ℕ := ∑ t ∈ range p.nReg, (p.reg t).cost + p.out.cost

/-- The registers' values; register `t ≥ nReg` is `0`. -/
def Plan.regs (p : Plan) (A B : Mat) (nz : ℕ → Mat) (t : ℕ) : Mat :=
  if t < p.nReg then (p.reg t).eval A B nz (fun i => if _ : i < t then p.regs A B nz i else 0) else 0
termination_by t

/-- A noise source: noise matrix `i` of unit `u`, from the oracle, the salt, the workload and the committed
activations. -/
abbrev NoiseSrc (Q R S : Type) : Type := (Q → R) → S → Workload → ℕ → Mat → ℕ → Mat

/-- Unit `u`'s commitment under plan `p`. -/
def Plan.commit (p : Plan) (N : NoiseSrc Q R S) (H : Q → R) (s : S) (U : Workload) (act : ℕ → Mat) (u : ℕ) :
    Commit :=
  let A := p.act.getD (act u)
  let B := U.weight (U.layout.wid u)
  let nz := N H s U u A
  ⟨A, p.regs A B nz, p.out.eval A B nz (p.regs A B nz)⟩

/-- **The abstract product model** for the noise source `N`: a program gives each unit of a layout a plan or a
blank; its cost is the sum of its units' plan costs. Its oracle use is only through the noise atoms, and it makes no
counted calls (hash calls are free in the game). -/
def gemmModel (N : NoiseSrc Q R S) : CostModel Q R S where
  Prog := Layout → ℕ → Option Plan
  run prog H U act _ s u := (prog U.layout u).map fun p => p.commit N H s U act u
  cost prog _ U _ _ _ := ∑ u ∈ range U.layout.N, ((((prog U.layout u).map Plan.cost).getD 0 : ℕ) : ℝ)
  calls _ _ _ _ _ _ := 0

/-! ## Protocol classes -/

/-- The unit's weight. -/
abbrev wtOf (U : Workload) (u : ℕ) : Mat := U.weight (U.layout.wid u)

/-- `M` on the `m × n` corner, `0` elsewhere: what a plan's products produce. `H_ref`'s outputs are stated on the
corner, so that a plan can reproduce them exactly (`AdmitsRef`); correctness reads only the corner. -/
def corner (m n : ℕ) (M : Mat) : Mat := fun i j => if i < m ∧ j < n then M i j else 0

/-- **Class (a)**: one checked product, the noised product `(A + E)(B + F)` with `E`, `F` the noise matrices 0 and 1,
compared on block 0's `m × n` entries; `H_ref`'s useful output is `A·B` by a decode outside the checked product, and
`W_ref` is a parameter. -/
def addProto (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) : Protocol Q R S where
  Y H s U u A := fun _ => mm (U.layout.shape u).k (A + N H s U u A 0) (wtOf U u + N H s U u A 1)
  checkIdx sh := {0} ×ˢ (range sh.m ×ˢ range sh.n)
  refChecked H s U u A := fun t =>
    if t = 0 then corner (U.layout.shape u).m (U.layout.shape u).n
      (mm (U.layout.shape u).k (A + N H s U u A 0) (wtOf U u + N H s U u A 1)) else 0
  refOut _ _ U u A := corner (U.layout.shape u).m (U.layout.shape u).n (mm (U.layout.shape u).k A (wtOf U u))
  Wref := W

/-- Checked product `i` of class (b): `(A + E_i)(B + F_i)` with `E_i`, `F_i` the noise matrices `2i` and `2i + 1`. -/
def relProd (N : NoiseSrc Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat) (i : ℕ) : Mat :=
  mm (U.layout.shape u).k (A + N H s U u A (2 * i)) (wtOf U u + N H s U u A (2 * i + 1))

/-- **Class (b)**: `t` checked products, each compared on its final `m × n` entries (block `i < t`); `H_ref`'s useful
output is the known linear combination `∑_{i < t} c_i·P_i`, and `W_ref` is a parameter. -/
def relProto (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) : Protocol Q R S where
  Y H s U u A := relProd N H s U u A
  checkIdx sh := range t ×ˢ (range sh.m ×ˢ range sh.n)
  refChecked H s U u A := fun i =>
    if i < t then corner (U.layout.shape u).m (U.layout.shape u).n (relProd N H s U u A i) else 0
  refOut H s U u A := corner (U.layout.shape u).m (U.layout.shape u).n
    (fun a b => ∑ i ∈ range t, c i * relProd N H s U u A i a b)
  Wref := W

/-- The relation condition of class (b): `∑ c_i = 1`, and for every oracle, salt, workload, unit and committed
activations, `∑ c_i·E_i = 0`, `∑ c_i·F_i = 0` and `∑ c_i·E_i·F_i = 0` (inner dimension `k` of the unit), so that
`∑ c_i·(A + E_i)(B + F_i) = A·B`. -/
def RelNoise (t : ℕ) (c : ℕ → ℤ) (N : NoiseSrc Q R S) : Prop :=
  ∑ i ∈ range t, c i = 1 ∧
    ∀ (H : Q → R) (s : S) (U : Workload) (u : ℕ) (A : Mat),
      (∀ a b, ∑ i ∈ range t, c i * N H s U u A (2 * i) a b = 0) ∧
      (∀ a b, ∑ i ∈ range t, c i * N H s U u A (2 * i + 1) a b = 0) ∧
      (∀ a b, ∑ i ∈ range t, c i * mm (U.layout.shape u).k (N H s U u A (2 * i)) (N H s U u A (2 * i + 1)) a b = 0)

/-- **Class (c)**: one product `X·Y` of inner dimension `L·d`, with `X = α(A) + E` and `Y = β(B) + F` (`E`, `F` the
noise matrices 0 and 1, `α`, `β` the protocol's operand maps), whose checked values are its running sums after every
depth-`d` step: block `t < L` is `∑_{l < (t + 1)·d} X_{il}·Y_{lj}` on the `m × n` entries. `H_ref`'s useful output is
the last running sum; `W_ref` is a parameter. -/
def segProto (L d : ℕ) (α β : Mat → Mat) (N : NoiseSrc Q R S) (W : Layout → ℕ → ℕ) : Protocol Q R S where
  Y H s U u A := fun t => mm ((t + 1) * d) (α A + N H s U u A 0) (β (wtOf U u) + N H s U u A 1)
  checkIdx sh := range L ×ˢ (range sh.m ×ˢ range sh.n)
  refChecked H s U u A := fun t =>
    if t < L then corner (U.layout.shape u).m (U.layout.shape u).n
      (mm ((t + 1) * d) (α A + N H s U u A 0) (β (wtOf U u) + N H s U u A 1)) else 0
  refOut H s U u A := corner (U.layout.shape u).m (U.layout.shape u).n
    (mm (L * d) (α A + N H s U u A 0) (β (wtOf U u) + N H s U u A 1))
  Wref := W

/-- A noise source of the lead's form, from per-unit factors `E₁ = e`, `F₁ = f` and `R = r` (inner dimension `k` of
the unit): noise matrices `0, …, 5` are `E₁, F₁, E₂, F₂, E₃, F₃` (`Pouw.NCP`), and the rest are `0`. -/
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

/-- The lead's noise as a class (c) noise source at a fixed inner dimension `k` and a fixed `R = Rm`, from per-unit
factors `E₁ = e`, `F₁ = f`: source `0` is `[E₁ | E₂ | −E₃]`, source `1` is `[F₁; F₂; F₃]`, the rest `0`. -/
def stackNoise (k : ℕ) (Rm : Mat) (e f : (Q → R) → S → Workload → ℕ → Mat → Mat) : NoiseSrc Q R S :=
  fun H s U u A i =>
    if i = 0 then NCP.stackX k 0 (e H s U u A) Rm else if i = 1 then NCP.stackY k 0 Rm (f H s U u A) else 0

end Pouw.Barrier
