module

public import Definitions.Flock.Merkle

/-!
# The two colliding inputs of two Merkle openings

Two openings of one position that verify against one tree top from different leaves go up through a node where the two
walks first agree, and the two inputs hashed at that node differ. `walkPair` returns those two inputs. `openPair` and
`openPairL` return the pair for two openings to different columns: the leaf's own pair when the two leaves agree, the
walk's pair otherwise. Each is a function of the two openings alone, so a collision finder can output it.

That each pair is a collision of `H` is proved in `Proofs.Flock.Soundness.MerklePair` (`walkPair_spec`,
`openPair_spec`, `openPairL_spec`).

Two digests are compared by their bytes (`Enc.dig`, which is injective), so no function here needs equality on the
digest type to be decidable.
-/

@[expose] public section

namespace FlockSoundness.Merkle

variable {Col D : Type}
variable (H : List UInt8 → D) (E : Enc Col D)

/-- **The two inputs hashed where two walks first agree.** The walks go up from `a` and from `a'` at index `idx`,
through the siblings `sibs` and `sibs'`. At the first step whose two parents agree, the pair is that step's two inputs,
in the order `up` hashes them. It is the empty pair when a path ends before the parents agree. -/
def walkPair : List D → List D → ℕ → D → D → List UInt8 × List UInt8
  | s :: rest, s' :: rest', idx, a, a' =>
      if E.dig (up H E idx a s) = E.dig (up H E idx a' s') then
        if idx % 2 = 0 then (E.dig a ++ E.dig s, E.dig a' ++ E.dig s')
        else (E.dig s ++ E.dig a, E.dig s' ++ E.dig a')
      else walkPair rest rest' (idx / 2) (up H E idx a s) (up H E idx a' s')
  | _, _, _, _, _ => ([], [])

/-- **The pair of two openings of position `pos` to the columns `c` and `c'`**, under the leaf `H (encode column)`: the
two columns' encodings when their leaves agree, the walk's pair from the two leaves otherwise. -/
def openPair (pos : ℕ) (c c' : Col) (sibs sibs' : List D) : List UInt8 × List UInt8 :=
  if E.dig (H (E.col c)) = E.dig (H (E.col c')) then (E.col c, E.col c')
  else walkPair H E sibs sibs' pos (H (E.col c)) (H (E.col c'))

variable {S : Type}

/-- **The pair of two openings under a leaf scheme**, of position `pos` to the columns `c` and `c'` under the salts
`s` and `s'`: the scheme's own pair (`Leaf.pair`) when the two leaves agree, and the walk's pair from the two leaves
otherwise. -/
def openPairL (Lf : Leaf Col S D H) (pos : ℕ) (c c' : Col) (s s' : S) (sibs sibs' : List D) :
    List UInt8 × List UInt8 :=
  if E.dig (Lf.leaf c s) = E.dig (Lf.leaf c' s') then Lf.pair c s c' s'
  else walkPair H E sibs sibs' pos (Lf.leaf c s) (Lf.leaf c' s')

/-- The members of a list taken two at a time, in the list's order: what a search for two of a run's openings that
clash goes through. -/
def pairsOf {γ : Type} (xs : List γ) : List (γ × γ) := xs.flatMap fun x => xs.map fun y => (x, y)

end FlockSoundness.Merkle
