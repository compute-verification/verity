module

public import Definitions.Pouw.Assumptions.NCP.Chain
public import Definitions.Pouw.NCP.Relation
public import Definitions.Pouw.NCP.RouteU

/-! The word layer on route U, where the reduction's "only if" transfers and its converse does not, so the floors take NzNCP of the route-U values as a hypothesis of their own. -/

@[expose] public section

namespace Pouw.NCP

open Finset Pouw.NCP.Relation

def checkedUPos (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  checkedU β k (A u) (B u) (e1Of ω u) (f1Of ω) R (p / (m * n) + 1) (p % (m * n) / n) (p % (m * n) % n)

/-- Route U's protocol checked values in the word model. -/
def wrappedU (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (ω : Noise N m n k) (u p : ℕ) : ℤ :=
  wrap32 (checkedUPos β k m n N R A B ω u p)

namespace RouteU

def ReductionOnlyIfU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (u p : ℕ) (r : Word.Rule), u < N →
    (∀ ω : Noise N m n k, checkedUPos β k m n N R A B ω u p = r.target (checkedUPos β k m n N R A B ω u)) →
    ∀ e < m * n, ∀ a < k, ∀ b < k,
      coefAt k m n R p e a b = r.target (fun q => coefAt k m n R q e a b) - r.target (fun _ => 0)

def NCPFinalOnlyIndU : Prop :=
  ∀ (β : ℤ) (k m n N : ℕ) (R : Mat) (A B : ℕ → Mat) (adv : Word.Adversary), 16 ∣ k →
    CheckpointIndependent k 16 R → Word.Valid (fun _ => positions k m n) adv →
    ∀ u < N, Word.identCount (fun _ => positions k m n) (checkedUPos β k m n N R A B) adv u ≤ m * n

end RouteU

end Pouw.NCP
