import Definitions.Pouw.Barrier.Defs

/-! The proof-only part of `Definitions.Pouw.Barrier.Defs`: the definitions and lemmas that no guarantee
statement or assumption reads. -/

namespace Pouw.Barrier

open Finset

variable {Q R S : Type}

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

end Pouw.Barrier
