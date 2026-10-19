import Definitions.Accounting.TS.Drawn

/-!
# A padded window's records: the registration and the epoch (trusted)

A padded window (`SelectLayer`) reads two kinds of record proved outside its own rounds
(note:20261018T2200Z-draft-padded-credit-rule, round 5): the weights' registration, proved once, before every salt,
whose certificates `ok_B,j` gate the window's tiles and whose sums `S_B` its tally reads; and the salt's epoch, proved
once per salt, whose formed B rows `B̃_j ‖ P_B,j` are the operands the salt's windows' tiles open. Each is a sampled run
of proofs (`DrawnLayer`) whose record units are always proved, so its records are right wherever its drawn units are
(`Accounting.Guarantees.TS.RegisteredRightOff`, `EpochFormedRightOff`).

* `Registration`: a `DrawnLayer` with each weight's rows, each row's raw row, its certificate and its weight's sum
  positions, the forming verdict and the count a value reads as, each record's units and the run's acceptance.
* `Registration.Right`: record `r`'s certificates are their raw rows' forming verdicts and its sum is their count.
* `Epoch`: a `DrawnLayer` whose first round is the epoch's (the salt, its formings) and whose later rounds are the
  salt's windows', with each record's raw row and formed row positions, the forming under the salt's noise on a path,
  each record's units, the formed row a window unit's slot reads, and the run's acceptance.
* `Epoch.Right`: record `e`'s formed row is its raw row's forming under the salt's noise.
* `Registration.AlwaysProved`, `Epoch.AlwaysProved`, `Registration.Of`, `Epoch.Of`, `Epoch.DomainBound`: protocol
  structure.
* `Registration.Computes`, `Epoch.Computes`: a property of the forming verdict's, the sum's and the forming's
  Definitions, proved for each registration and each epoch.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- **The weights' registration**: a sampled run of proofs (`DrawnLayer`) and
* `rows r`: the rows of weight record `r`;
* `rawPos j`, `okPos j`: row `j`'s raw row `B_j` and its certificate `ok_B,j`, committed together;
* `sumPos r`: weight `r`'s registered sum `S_B`;
* `verdict`: the forming's verdict on a raw row (finite, `ρ·α ≥ 1`, at most `k/64` dead lanes), a function of the raw
  row alone, which no salt changes;
* `count`: the count a committed certificate or sum reads as;
* `units r`: the units proving record `r` (each row's verdict, the sum's limbs), each always proved;
* `accepts`: the registration's acceptance. -/
structure Registration (C B : Type) (S m s : ℕ) (J Q Pos V : Type) extends DrawnLayer C B S m s J Q Pos V where
  rows : ℕ → Finset ℕ
  rawPos : ℕ → Pos
  okPos : ℕ → Pos
  sumPos : ℕ → Pos
  verdict : V → V
  count : V → ℕ
  units : ℕ → Finset (Fin S × J)
  accepts : Paths (Paths C m × B) S → Prop

namespace Registration

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (rg : Registration C B S m s J Q Pos V)

/-- **Record `r` is right on `x`**: each row's committed certificate `ok_B,j` is the forming verdict of its committed
raw row `B_j`, and the committed sum `S_B` counts `Σ_j ok_B,j` over the weight's rows. -/
def Right [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (r : ℕ) : Prop :=
  (∀ j ∈ rg.rows r,
      rg.toDrawnLayer.committed x (rg.okPos j) = rg.verdict (rg.toDrawnLayer.committed x (rg.rawPos j))) ∧
    rg.count (rg.toDrawnLayer.committed x (rg.sumPos r)) =
      ∑ j ∈ rg.rows r, rg.count (rg.toDrawnLayer.committed x (rg.okPos j))

/-- **Every record unit is in every draw of its round.** Protocol structure. -/
def AlwaysProved : Prop := ∀ r, ∀ u ∈ rg.units r, ∀ b : B, ∃ i, rg.draw b i = u.2

/-- **The registration is accepted only if every drawn unit's proof is.** Protocol structure. -/
def Of : Prop := ∀ x, rg.accepts x → ∀ w i, rg.accepted x w i

/-- **Right record units make a right record**: on every path where each unit of record `r` is right at its committed
slots, record `r` is right. A property of the verdict's and the sum's Definitions (each `ok_B,j` unit reads `B_j` and
writes `ok_B,j`; the sum's limbs read every `ok_B,j` of the weight and write `S_B`), proved for each registration. -/
def Computes [Fintype C] [Fintype B] [Nonempty V] : Prop :=
  ∀ x r, (∀ u ∈ rg.units r, rg.right x u.1 u.2 (rg.toDrawnLayer.view x u.1 u.2)) → rg.Right x r

end Registration

/-- **A salt's epoch and its windows**: a sampled run of proofs (`DrawnLayer`) whose first round is the epoch's (its
coins the salt, the formings proved once per salt) and whose later rounds are the salt's windows', and
* `rawPos e`, `formedPos e`: record `e`'s raw row `B_j` (registered) and its formed row `B̃_j ‖ P_B,j`, a row of the
  salt-bound epoch tree;
* `form x v`: the forming of raw row `v` under the salt's noise on `x` (the noise the verifier derives from the salt
  and the hiding roots);
* `units e`: the units proving record `e` (the forming's pieces), each always proved;
* `reads u k`: the epoch record unit `u`'s slot `k` reads, if any (a tile's B operand);
* `accepts`: the run's acceptance. -/
structure Epoch (C B : Type) (S m s : ℕ) (J Q Pos V : Type) extends DrawnLayer C B S m s J Q Pos V where
  rawPos : ℕ → Pos
  formedPos : ℕ → Pos
  form : Paths (Paths C m × B) S → V → V
  units : ℕ → Finset (Fin S × J)
  reads : Fin S × J → Q → Option ℕ
  accepts : Paths (Paths C m × B) S → Prop

namespace Epoch

variable {C B : Type} {S m s : ℕ} {J Q Pos V : Type} (ep : Epoch C B S m s J Q Pos V)

/-- **Record `e` is right on `x`**: its committed formed row is the forming of its committed raw row under the salt's
noise. -/
def Right [Fintype C] [Fintype B] [Nonempty V] (x : Paths (Paths C m × B) S) (e : ℕ) : Prop :=
  ep.toDrawnLayer.committed x (ep.formedPos e) = ep.form x (ep.toDrawnLayer.committed x (ep.rawPos e))

/-- **Every record unit is in every draw of its round.** Protocol structure. -/
def AlwaysProved : Prop := ∀ e, ∀ u ∈ ep.units e, ∀ b : B, ∃ i, ep.draw b i = u.2

/-- **The run is accepted only if every drawn unit's proof is.** Protocol structure. -/
def Of : Prop := ∀ x, ep.accepts x → ∀ w i, ep.accepted x w i

/-- **A window's slot reading an epoch record is that record's row of this salt's tree**: every slot that reads record
`e` is at `e`'s formed row position. The tree's domain binds `(salt, root_W)` and is a public input of every tile
instance, which the verifier sets from the live salt and the registered root, so a tile opens its B operand in this
salt's tree and no other: a stale salt's tree, or another weight's, has another domain, so its rows are other
positions. Binding across the salt's windows is then the positions' binding (`DrawnLayer.MovedIn`): every window
opening record `e` opens `e`'s committed formed row. Protocol structure. -/
def DomainBound : Prop := ∀ u k e, ep.reads u k = some e → ep.pos u.1 u.2 k = ep.formedPos e

/-- **Right record units make a right record**: on every path where each unit of record `e` is right at its committed
slots, record `e` is right. A property of the forming's Definitions (its pieces read the raw row and the salt's noise
and write the formed row), proved for each epoch. -/
def Computes [Fintype C] [Fintype B] [Nonempty V] : Prop :=
  ∀ x e, (∀ u ∈ ep.units e, ep.right x u.1 u.2 (ep.toDrawnLayer.view x u.1 u.2)) → ep.Right x e

end Epoch

end Accounting.TS
