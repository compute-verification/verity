module

public import Definitions.Core.Hash
public import Definitions.Core.Hm96

/-!
# String commitments

A string-commitment scheme over values `α`, the hash `H` its public parameter: `commit` takes a string and the
committer's coins and returns a commitment and the committer's opening information, `open` takes that information and
a position and returns the value there with an opening, and `Opens H c p v π` is the check that `π` opens position `p`
of the commitment `c` to the value `v`; `hashed` lists the messages that check hashed. An opening at a set of positions
is an opening at each.

`Correct`: whatever `open` returns for an honest commitment is the string's value and is accepted. `Binds`: two
accepted openings of one position to different values are a collision of `H` among the messages their checks hashed,
the form of the area's other bindings (`Guarantees.TreeBinding`, `Guarantees.RowBinding`), so a binding break is tied to
the run that made it. `Hides`: for one `H`, commitments to any two strings of one length are within `ε` on every event,
the form of `Claims.Hm96Hiding`.
-/

@[expose] public section

namespace Hash

open scoped ENNReal

/-- **A string-commitment scheme** over values `α`: its commitments, the committer's opening information, its openings
and the committer's coins; `commit`, `open`, the opening check `Opens` and what that check hashed. -/
structure StringCommitment (α : Type) where
  Com : Type
  Dec : Type
  Op : Type
  Coins : Type
  [coinsFintype : Fintype Coins]
  commit : (Msg → Digest) → List α → Coins → Com × Dec
  «open» : (Msg → Digest) → Dec → ℕ → α × Op
  Opens : (Msg → Digest) → Com → ℕ → α → Op → Prop
  hashed : (Msg → Digest) → Com → ℕ → α → Op → List Msg

attribute [instance] StringCommitment.coinsFintype

namespace StringCommitment

variable {α : Type}

/-- **Correctness**: at each position of the string, the honest opening of an honest commitment opens it to the
string's value. -/
def Correct (S : StringCommitment α) (H : Msg → Digest) : Prop :=
  ∀ (s : List α) (r : S.Coins) (p : ℕ) (hp : p < s.length),
    (S.open H (S.commit H s r).2 p).1 = s[p] ∧
      S.Opens H (S.commit H s r).1 p (S.open H (S.commit H s r).2 p).1 (S.open H (S.commit H s r).2 p).2

/-- **Binding, as a break tied to the run**: two openings of one position of one commitment, each accepted, open it to
one value, or two different messages that the two checks hashed have one digest under `H`. -/
def Binds (S : StringCommitment α) (H : Msg → Digest) : Prop :=
  ∀ (c : S.Com) (p : ℕ) (v v' : α) (π π' : S.Op), S.Opens H c p v π → S.Opens H c p v' π' →
    v = v' ∨ CollisionIn H (S.hashed H c p v π ++ S.hashed H c p v' π')

/-- **Statistical hiding with error `ε`**, for one `H`: for strings `s` and `s'` of one length and uniform coins, no
event on the commitment is more likely under `s` than under `s'` by more than `ε`. -/
def Hides (S : StringCommitment α) (H : Msg → Digest) (ε : ℝ≥0∞) : Prop :=
  ∀ (s s' : List α), s.length = s'.length → ∀ E : S.Com → Prop,
    Claims.PrENNReal (fun r : S.Coins => E (S.commit H s r).1) ≤
      Claims.PrENNReal (fun r : S.Coins => E (S.commit H s' r).1) + ε

end StringCommitment

end Hash
