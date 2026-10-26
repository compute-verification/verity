import Definitions.Pouw.Fp8Atom.H1TLemmas

/-!
# H-1T's running words (a named assumption, M4b and M4c)

* **`H1TRunningWords`** is `DistinctLiveH1T` on the pairs of words the one-flip argument does not cover (`Hard`): O1
  and O2 of `internal/pouw-fp8/lean-atom-m2-plan.md` §0.4, less what one flip separates by column and near the start
  of block 2. On a legal unit, two distinct credited words of one row are distinct functions of the salt if they are
  two block-1 running words at one position of columns whose tuples are less than 72 apart in every lane and have no
  lane of opposite signs where either is at least 36 in magnitude, or a block-2 running word against any word but a
  fresh-step word at a later position, the block-1 words of `OppLater`, and, for an `Early` running word, the words
  that skip the last slice; and no `Stuck` block-2 running word is constant.

A flip moves a fresh-step word by at least 960 and a block-1 running word by at least 448, but the difference of two
columns' running words by as little as `2·|v|·|Δt| = 64` at `F = 1/4`, against an ulp of 512. So two block-1 running
words at one position separate only when the tuples are at least 72 apart in the flipped lane, or when they move
strictly in opposite directions. The move a block-1 slice `s` makes in a block-2 running word `R_p` passes the
roundings from `R_s` to `R_p`, `p − s + 1` of them (`p` for `s = 0`, since `R_0 = I_0`), each of which can absorb up
to 512 of it; the worst case is `2T − 3`, at `s = 0` and `p = 2T − 3`. From the last slice, `s = T − 1`, it is
`p − T + 2`, so a large tag lane still moves `R_p` near the start of block 2 (`Early`). Otherwise it separates the word only from a word that moves strictly the other
way: a block-1 word of a later slice and of a column whose tuple has the opposite sign in the flipped lane. It is
implied by `DistinctLiveH1T` (`h1tRunningOfDistinct`), and with `H1TOneFlip` it implies `DistinctLiveH1T`
(`h1tDistinctLive`): the two are equivalent. It is Assumed, with the red team's simulation evidence (pair frequencies
at most 0.375), and it is a hypothesis of the pins that use it.
-/

namespace Pouw.Fp8Atom.H1T

/-- **`H1TRunningWords`** (O1 and O2, narrowed by column and near the start of block 2): on every legal unit and row
`i`, two distinct credited words of row `i` that are `Hard` are distinct functions of the unit's salt, and a `Stuck`
block-2 running word is not a constant. -/
def H1TRunningWords : Prop :=
  ∀ U : Unit, U.Legal → ∀ (i : Fin U.m) (a a' : Fin U.n × Pos U.T),
    (a ≠ a' → Hard a.1.val a'.1.val a.2 a'.2 → U.word (i, a) ≠ U.word (i, a')) ∧
      (Stuck a.1.val a.2 → ∀ q : ℚ, U.word (i, a) ≠ fun _ => q)

end Pouw.Fp8Atom.H1T
