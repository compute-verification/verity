import Definitions.Accounting.Remote.Draw
import Definitions.Accounting.TS.Drawn

/-!
# The receipts' fork game (trusted)

What the receipts' binding term (`Receipts.binding`) is bounded by: two runs of the session from window `t`'s cut on
independent openings coins (`cont t`, `Remote.Draw`), and what each break of them costs. `Receipts.held` is the
plurality of the rows a receipt shows (`Receipts.Shows`) over those continuations, so a move squared is at most a fork.

* `Receipts.Broken D prf rest t ω ζ`: **the fork's break**. Some receipt window `t` opens in both runs, `ω` and
  `cont t ω ζ`, has two different messages of one SHA-512 digest among what the two openings' checks hashed: under the
  leaf and path checks, or under the row digests (`Hash.RowLeaf.outerHashed`, `innerHashed`).
* `Receipts.Split D prf rest t ω ζ`: **the roots split**. Some receipt window `t` opens in both runs has its verifier,
  root or rank differ between them. Two accepted openings bind each other only at one root (`Hash.Guarantees.RowBinding`),
  so a fork whose roots split breaks nothing. A window whose receipts' roots, ranks and verifiers are fixed by its cut
  never splits. The deployed prover is sent window `t`'s coin only after its roots are in, and the challenger's
  commitment to the coins only with the outcome, so its roots read only what precedes window `t`'s coin; that the cut
  fixes them is not yet a premise of the guarantees, and nothing here bounds `split` (the gap `binding-split`).
* `Receipts.fork` and `Receipts.split`: their probabilities over a uniform pair `(ω, ζ)`.
* `Receipts.finder D prf rest t q`: **window `t`'s fork finder**, a SHA-512 collision finder (`Claims.Finder`) on the
  coins `(ω, ζ)`. It runs the session on `ω` and on `cont t ω ζ`, and for a receipt both runs open it outputs two
  different messages of one digest among what the two openings' checks hashed, when there are (`Receipts.Finds`). Its
  cost is `q`, the SHA-512 calls of two runs of the prover and the verifier's checks of their openings; claim
  `cr/sha-512` (`Claims.StrictBirthday`) bounds its chance by `q²/2^513`.
-/

namespace Accounting.Remote.Draw

open scoped ENNReal

variable {n κ : ℕ} {X : Type}

namespace Receipts

variable (R : Receipts n κ X) {m : ℕ} {s : Fin n → ℕ}

/-- **The fork's break at window `t`**: some receipt the window opens in both `ω` and its continuation on `ζ` has two
different messages of one digest among what the two openings' checks hashed. -/
def Broken (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ)
    (t : Fin n) (ω ζ : Coins n κ X) : Prop :=
  ∃ p ∈ D.opened prf rest t ω, p ∈ D.opened prf rest t (cont t ω ζ) ∧
    (Hash.CollisionIn R.hash
        (Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t ω p) (R.rank t ω p) (R.opened t ω p) ++
          Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t (cont t ω ζ) p) (R.rank t (cont t ω ζ) p)
            (R.opened t (cont t ω ζ) p)) ∨
      Hash.CollisionIn R.hash
        (Hash.RowLeaf.innerHashed R.hash (R.verifier t ω p) (R.opened t ω p) ++
          Hash.RowLeaf.innerHashed R.hash (R.verifier t (cont t ω ζ) p) (R.opened t (cont t ω ζ) p)))

/-- **Window `t`'s roots split** between `ω` and its continuation on `ζ`: some receipt the window opens in both has its
verifier, root or rank differ between them. -/
def Split (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ)
    (t : Fin n) (ω ζ : Coins n κ X) : Prop :=
  ∃ p ∈ D.opened prf rest t ω, p ∈ D.opened prf rest t (cont t ω ζ) ∧
    ¬ (R.verifier t (cont t ω ζ) p = R.verifier t ω p ∧ R.root t (cont t ω ζ) p = R.root t ω p ∧
      R.rank t (cont t ω ζ) p = R.rank t ω p)

/-- The probability of window `t`'s fork break over a uniform pair of coins. -/
noncomputable def fork [Fintype X] (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) : ℝ≥0∞ :=
  Pous.pr fun y : Coins n κ X × Coins n κ X => R.Broken D prf rest t y.1 y.2

/-- The probability that window `t`'s roots split over a uniform pair of coins. -/
noncomputable def split [Fintype X] (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) : ℝ≥0∞ :=
  Pous.pr fun y : Coins n κ X × Coins n κ X => R.Split D prf rest t y.1 y.2

/-- **What window `t`'s fork finder outputs on `(ω, ζ)`**: two different messages `uv` of one SHA-512 digest, both
among what the checks of the two openings of one receipt the window opens in both runs hashed. -/
def Finds (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m) (rest : Coins n κ X → Fin n → ℕ → ℕ)
    (t : Fin n) (ω ζ : Coins n κ X) (uv : Hash.Msg × Hash.Msg) : Prop :=
  Claims.Collision (fun u => (R.hash u).bytes) uv.1 uv.2 ∧
    ∃ p ∈ D.opened prf rest t ω, p ∈ D.opened prf rest t (cont t ω ζ) ∧
      ((uv.1 ∈ Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t ω p) (R.rank t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t (cont t ω ζ) p) (R.rank t (cont t ω ζ) p)
              (R.opened t (cont t ω ζ) p) ∧
          uv.2 ∈ Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t ω p) (R.rank t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.outerHashed R.hash R.hash (R.verifier t (cont t ω ζ) p) (R.rank t (cont t ω ζ) p)
              (R.opened t (cont t ω ζ) p)) ∨
        (uv.1 ∈ Hash.RowLeaf.innerHashed R.hash (R.verifier t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.innerHashed R.hash (R.verifier t (cont t ω ζ) p) (R.opened t (cont t ω ζ) p) ∧
          uv.2 ∈ Hash.RowLeaf.innerHashed R.hash (R.verifier t ω p) (R.opened t ω p) ++
            Hash.RowLeaf.innerHashed R.hash (R.verifier t (cont t ω ζ) p) (R.opened t (cont t ω ζ) p)))

open Classical in
/-- **Window `t`'s fork finder** at `q` SHA-512 calls: on coins `(ω, ζ)` it outputs what it finds (`Finds`), if
anything. -/
noncomputable def finder [Fintype X] [Nonempty X] (D : Session n m s κ X) (prf : Fin n → Pous.Bits κ → Blocks m)
    (rest : Coins n κ X → Fin n → ℕ → ℕ) (t : Fin n) (q : ℕ) : Claims.Finder where
  S := Coins n κ X × Coins n κ X
  out y := if h : ∃ uv, R.Finds D prf rest t y.1 y.2 uv then some h.choose else none
  cost _ := q

end Receipts

end Accounting.Remote.Draw
