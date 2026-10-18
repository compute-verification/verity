import Models.Hash.Assumptions

/-!
# The hash area's guarantee statements (proposals, not yet proved)

Statement first (Daniel's ruling): these are the statements for the red team and for Daniel's card, with no proof yet,
each under `shape.frozen` as `unstated` in `Security/lean-audit.json` until a theorem `Hash.SecurityProofs.G : G` in
`Security/Proofs/Hash/` proves it and the lock lists it.

* `TreeBinding`: a Merkle double opening at one position gives the same leaf input, or a collision of the hash among
  the inputs the two checks hashed. Deterministic, for every hash.
* `SisTreeBinding`: for every matrix, fixed or drawn, a double opening of an SIS leaf with different digits gives an SIS
  solution (the digits' difference), or a collision of the tree's hash among the inputs the two checks hashed.
* `SisEpochBinding`: for a matrix drawn uniformly (from the epoch's live coin), a prover's double opening with no hash
  collision on its paths has probability at most `(E[cost] + 1)/2^bits`, given `SISHard` at the finder it gives.
* `EpochBinding`: for a tree under a hash keyed by the epoch's live coin, a prover of at most `q` evaluations in the
  window opens one position two ways with probability at most `(q + 2·(depth + 1))² / 2^(2·lam + 1)`, given
  `KeyedWindowCR` at the finder it gives.

Each break is tied to the run (ruling of 8 Oct): the collision is among inputs the run's own checks hashed, and the SIS
solution is the difference of the run's own openings.
-/

namespace Hash.Guarantees

/-- **Merkle binding, as a break tied to the run.** Two openings accepted at position `i` against one root, in a
depth-`d` tree over `H`, have the same leaf input, or the inputs their checks hashed hold a collision of `H`. -/
def TreeBinding : Prop :=
  ∀ (H : Msg → Digest) (tag : Msg) (d : ℕ) (r : Digest) (i : ℕ) (leaf₁ leaf₂ : Msg) (sibs₁ sibs₂ : List Digest),
    Path.accepts H tag d r i leaf₁ sibs₁ → Path.accepts H tag d r i leaf₂ sibs₂ →
    leaf₁ = leaf₂ ∨ CollisionIn H (Path.hashed H tag i leaf₁ sibs₁ ++ Path.hashed H tag i leaf₂ sibs₂)

/-- **SIS leaf binding, as a break tied to the run.** For every parameter set, format, hash and matrix `A` (a fixed
public matrix included), two openings accepted at position `i` against one root have the same digits, or their
difference is an SIS solution for `A`, or the inputs their checks hashed hold a collision of `H`. -/
def SisTreeBinding : Prop :=
  ∀ (P : Sis.Params) (F : Sis.Format) (H : Msg → Digest) (A : Sis.Mat P) (r : Digest) (i : ℕ) (o₁ o₂ : Sis.Opening P),
    Sis.accepts P F H A r i o₁ → Sis.accepts P F H A r i o₂ →
    o₁.w = o₂.w ∨ Sis.Solution P A (Sis.diff o₁ o₂) ∨ CollisionIn H (Sis.hashed P F H A i o₁ ++ Sis.hashed P F H A i o₂)

/-- **SIS binding with a live matrix.** For a matrix drawn uniformly and a prover's coins, the probability that the
prover's claim is a binding break whose checks hashed no collision of `H` is at most the expected cost of the finder it
gives (its cost plus `m`) plus one, over `2^bits`, given `SISHard` at that finder. -/
def SisEpochBinding : Prop :=
  ∀ (P : Sis.Params) (F : Sis.Format) (H : Msg → Digest) (V : Sis.Prover P),
    Assumptions.SISHard P (Sis.reduce V) →
    Pr (fun s : Sis.Mat P × V.Ω =>
        Sis.Break P F H s.1 (V.out s.1 s.2) ∧ ¬ CollisionIn H (Sis.claimHashed P F H s.1 (V.out s.1 s.2))) ≤
      (Ex (fun s : Sis.Mat P × V.Ω => ((V.cost s.1 s.2 + P.m : ℕ) : ℝ)) + 1) / 2 ^ P.bits

/-- **Epoch-keyed binding against a compute-bounded key-holder.** For a key drawn uniformly (the epoch's live coin) and
a prover's coins, a prover that evaluates `h k` at most `q` times in the window outputs a binding break of the tree
under `h k` with probability at most `(q + 2·(depth + 1))² / 2^(2·lam + 1)`, given `KeyedWindowCR` at the finder it
gives. -/
def EpochBinding : Prop :=
  ∀ {K : Type} [Fintype K] [Nonempty K] (h : K → Msg → Digest) (F : Keyed.Format) (lam q : ℕ) (V : Keyed.Prover K q),
    Assumptions.KeyedWindowCR h lam (Keyed.reduce F V) →
    Pr (fun p : K × V.Ω => Keyed.Break F (h p.1) ((V.prog p.1 p.2).run (h p.1))) ≤
      ((q + 2 * (F.depth + 1) : ℕ) : ℝ) ^ 2 / 2 ^ (2 * lam + 1)

end Hash.Guarantees
