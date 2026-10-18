import Definitions.Accounting.TS.Play

/-!
# A sampled window's proofs: what (I8) reads of its drawn units (trusted)

`ProofLayer`'s counterpart for 𝒫_samp (note:20261018T0449Z-draft-i8-sampled). A round's coins are `A × B`: on `A` the
coins released before the draws, after which the round's units, and every input of their seeds, are committed; on `B`
the live draw coin, which names `s` units (`DrawnLayer.draw`). Each drawn unit is proved alone, with the generation of
its own noise (decision 105). A unit is a self-contained `r × c` tile (decision 106): its proof opens the unit's values
(its operand rows, its seed's inputs and its outputs), checks the seed, derives the noise it reads and computes the
tile, so its relation and its rightness read its own values only, and its position is the unit's index `j`, not a
value the prover picks.

`ProofLayer` forks before a round's coins (`splice`), which redraws the round's draws, so proof `i` names a different
unit on each continuation. Here the fork is at the draw (`spliceDraw`), and unit `j`'s committed values are its
opening on the plurality continuation (`plurality`), a function of the coins before the draw whoever is drawn.

* `coinAt x w`: round `w`'s coins on `x`.
* `spliceDraw w x z`: `x`'s coins up to round `w`'s draw, then `z`'s.
* `plurality sp acc op x`: among `x`'s continuations `sp x z` on which `acc` holds, one whose `op` is the most frequent.
* `DrawnLayer`: per path, round `w` and unit `j`, whether unit `j`'s proof was accepted and the values it opened; unit
  `j`'s relation and rightness; the units the draw coin names (`drawnAt`); and unit `j`'s values as committed before
  the draw (`committed`).
* `Moved`, `Forked`: binding's failure for a unit, and its fork form; `Unsound`, `SoundAt`: proof soundness, per draw;
  `Pins`: pinning, per unit; `Of`: the play's tie to its drawn units' proofs.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- Round `w`'s coins on the path `x`. -/
def coinAt {Ω : Type} : {n : ℕ} → Paths Ω n → Fin n → Ω
  | 0, _, w => w.elim0
  | _ + 1, x, ⟨0, _⟩ => x.1
  | _ + 1, x, ⟨w + 1, h⟩ => coinAt x.2 ⟨w, Nat.lt_of_succ_lt_succ h⟩

/-- `x`'s coins before round `w`'s draws (its rounds before `w`, and round `w`'s coins on `A`), then `z`'s: the run that
shares `x`'s commitments up to round `w`'s draws and continues on fresh draws. -/
def spliceDraw {A B : Type} : {n : ℕ} → ℕ → Paths (A × B) n → Paths (A × B) n → Paths (A × B) n
  | 0, _, _, _ => ()
  | _ + 1, 0, x, z => ((x.1.1, z.1.2), z.2)
  | _ + 1, w + 1, x, z => (x.1, spliceDraw w x.2 z.2)

section Plurality

variable {X V : Type} [Fintype X]

open Classical in
/-- The fresh coins `z` on whose continuation of `x` (`sp x z`) `acc` holds. -/
noncomputable def accAt (sp : X → X → X) (acc : X → Prop) (x : X) : Finset X :=
  Finset.univ.filter fun z => acc (sp x z)

open Classical in
/-- How many of `accAt sp acc x` give `op` the value the continuation on `z` gives it. -/
noncomputable def agree (sp : X → X → X) (acc : X → Prop) (op : X → V) (x z : X) : ℕ :=
  ((accAt sp acc x).filter fun z' => op (sp x z') = op (sp x z)).card

open Classical in
/-- **The plurality continuation**: among the continuations of `x` on which `acc` holds, one whose value of `op` is the
most frequent; `x` when there is none. -/
noncomputable def plurality (sp : X → X → X) (acc : X → Prop) (op : X → V) (x : X) : X :=
  if h : (accAt sp acc x).Nonempty then sp x ((accAt sp acc x).exists_max_image (agree sp acc op x) h).choose
  else x

end Plurality

/-- **A sampled window's proofs** over `S` rounds of coins on `A × B`, `s` draws a round of units `J`, each unit's proof
opening its values in `V`: the units the draw coin names (`draw`); on each path, whether unit `j`'s proof in round `w`
was accepted (`accepted`) and the values it opened (`opened`); unit `j`'s relation in round `w` (`rel`), the public
Program its proof passes; and its rightness (`right`): its outputs are its tile's, from its operands, under the noise
its seed derives. -/
structure DrawnLayer (A B : Type) (S s : ℕ) (J V : Type) where
  draw : B → Fin s → J
  accepted : Paths (A × B) S → Fin S → J → Prop
  opened : Paths (A × B) S → Fin S → J → V
  rel : Fin S → J → V → Prop
  right : Fin S → J → V → Prop

namespace DrawnLayer

variable {A B : Type} {S s : ℕ} {J V : Type} (pf : DrawnLayer A B S s J V)

/-- The unit round `w`'s draw `i` names on `x`. -/
def drawnAt (x : Paths (A × B) S) (w : Fin S) (i : Fin s) : J := pf.draw (coinAt x w).2 i

/-- **Unit `j`'s values as round `w` committed them before its draws**: its proof's opening on the plurality
continuation of `x` from round `w`'s draws, among those on which its proof is accepted. -/
noncomputable def committed [Fintype A] [Fintype B] (x : Paths (A × B) S) (w : Fin S) (j : J) : V :=
  pf.opened (plurality (spliceDraw w) (fun y => pf.accepted y w j) (fun y => pf.opened y w j) x) w j

/-- **Binding fails for unit `j` in round `w`**: its proof is accepted and opens values other than those committed
before the draws. -/
def Moved [Fintype A] [Fintype B] (x : Paths (A × B) S) (w : Fin S) (j : J) : Prop :=
  pf.accepted x w j ∧ pf.opened x w j ≠ pf.committed x w j

/-- **Two runs off one commitment open a unit differently** (the fork form): on `x` and on its continuation from round
`w`'s draws on fresh coins `z`, which share every value committed before the draws, unit `j`'s proof is accepted, and
the two open it differently. Where a unit's values are hm96 rows of one frame-v3-sha512 tree, two openings are two
inputs hashed to one SHA-512 digest along their paths, which `cr/sha-512` bounds at a finder that runs the window twice
from round `w`'s draws. -/
def Forked (w : Fin S) (j : J) (x z : Paths (A × B) S) : Prop :=
  pf.accepted x w j ∧ pf.accepted (spliceDraw w x z) w j ∧ pf.opened x w j ≠ pf.opened (spliceDraw w x z) w j

/-- Unit `j`'s proof in round `w` is accepted, and its opening fails its relation. -/
def Unsound (x : Paths (A × B) S) (w : Fin S) (j : J) : Prop := pf.accepted x w j ∧ ¬ pf.rel w j (pf.opened x w j)

/-- **Proof soundness at `δ_ZK`, per draw**: the unit round `w`'s draw `i` names has its proof accepted with an opening
that fails its relation with probability at most `δ_ZK` over the window's coins. -/
def SoundAt [Fintype A] [Fintype B] (δzk : ℝ) : Prop :=
  ∀ w i, prob (fun x => pf.Unsound x w (pf.drawnAt x w i)) ≤ δzk

/-- **Each unit's relation pins its values**: values that satisfy unit `j`'s relation are right. A check with no
absorbed word as witness (`PouwCallCheck_v1`, `PouwWarpCheck_v1`, `PouwSegCheck_v3`, #2417) recomputes every word it
absorbs from the unit's operands, its seed and its noise. -/
def Pins : Prop := ∀ w j v, pf.rel w j v → pf.right w j v

/-- **The play is its drawn units' proofs'** (𝒫_samp): the window is accepted only if every drawn unit's proof is, and
the ideal game's check is the drawn units' values as committed before the draws, so drawn units right at those values
are accepted by the ideal game. -/
def Of [Fintype A] [Fintype B] {K T L U : Type} (P : Play (A × B) S K T L U) : Prop :=
  (∀ x, P.accepts x → ∀ w i, pf.accepted x w (pf.drawnAt x w i)) ∧
    ∀ x, (∀ w i, pf.right w (pf.drawnAt x w i) (pf.committed x w (pf.drawnAt x w i))) → P.game.accepts x

end DrawnLayer

end Accounting.TS
