import Definitions.Hash

/-!
# The hash area's named assumptions

Each is a `Prop` taken as a hypothesis by the statements that use it, never an `axiom`, and each is stated at one
finder, the one a prover gives, because a bound over every finder of a fixed function is false (some finder hard-codes
the answer). Both are proposals: each new assumption goes to Daniel as a card before any guarantee rests on it.
-/

namespace Hash.Assumptions

/-- **`SISHard` (proposed claim `sis/<param-set>`): SIS is hard at the parameter set `P`, for the finder `F`.** For a
uniform matrix `A ∈ Z_q^{n×m}` and `F`'s coins, `F` outputs an SIS solution for `A` (nonzero, entries within
`β = hi − lo`, `A·z = 0 mod q`) with probability at most its expected cost plus one, over `2^bits`. The `+ 1` is the
chance a finder of no work has: a fixed nonzero `z` is a solution of a uniform `A` with probability `q^-n`, so
`2^bits ≤ q^n` is part of choosing `P`. `P.bits` is what the lattice estimator gives the parameter set (128 for a fixed
public matrix that must hold against offline work; a window's for a matrix drawn from the epoch's live coin). It is
stated over uniform matrices only: a fixed matrix derived from a public label reaches it through the
common-reference-string step `hash-derived-matrix`, and one expanded from a live coin through `random-oracle`. -/
def SISHard (P : Sis.Params) (F : Sis.Finder P) : Prop :=
  Pr (fun s : Sis.Mat P × F.S => ∃ z, F.out s.1 s.2 = some z ∧ Sis.Solution P s.1 z) ≤
    (Ex (fun s : Sis.Mat P × F.S => (F.cost s.1 s.2 : ℝ)) + 1) / 2 ^ P.bits

/-- **`KeyedWindowCR` (proposed claim `wcr/<instance>`, e.g. `wcr/sha-512-r48`): the keyed hash `h` resists a
key-holder's collision search in the window, at `lam` bits, for the finder `F`.** For a uniform key `k` and `F`'s
coins, if `F` evaluates `h k` at most `q` times in every world, it outputs a collision of `h k` with probability at most
`q² / 2^(2·lam + 1)`: the birthday bound of a `2·lam`-bit hash, for a finder that *holds the key*. Its content is that
the family leaves a key-holder no shortcut within the window: for SHA-512 itself it is `cr/sha-512` restricted to a
fresh IV (`lam = 256`), and for a round-reduced SHA-512 it is the premise that the remaining rounds keep a margin over
the best collision attacks from a known IV. A universal or linear family is false here at `q` of a few evaluations,
which is the trap it states. -/
def KeyedWindowCR {K : Type} [Fintype K] [Nonempty K] (h : K → Msg → Digest) (lam : ℕ) (F : Keyed.Finder K) : Prop :=
  ∀ q : ℕ, (∀ (k : K) (s : F.S), ((F.prog k s).evals (h k)).length ≤ q) →
    Pr (fun p : K × F.S => ∃ u v, (F.prog p.1 p.2).run (h p.1) = some (u, v) ∧ Collision (h p.1) u v) ≤
      (q : ℝ) ^ 2 / 2 ^ (2 * lam + 1)

end Hash.Assumptions
