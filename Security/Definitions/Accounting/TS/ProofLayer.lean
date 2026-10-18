import Definitions.Accounting.TS.Play

/-!
# A window's proofs: what (I8) reads of the proof service (trusted)

EPS1-SPEC §2.7's (I8), `Accounting.Assumptions.TS.ProofsSound`, reads three pieces of the proof service and one of the
TS model's accounting (note:20261017T1817Z-draft-i8-lean):

* **binding at `δ_bind`**: an accepted proof opens its round's leaves (hm96 rows in frame-v3-sha512 trees, committed
  before the round's σ) to the values the round committed (`ProofLayer.Moved`, its failure). Its fork form
  (`ProofLayer.Forked`, the 8 Oct ruling's): two runs off one commitment, both accepted, open it differently;
* **proof soundness at `δ_ZK`**: an accepted proof's opening satisfies its relation, outside `δ_ZK` a proof
  (`ProofLayer.SoundAt`); C-Flock's `Flock.Guarantees.EndToEndHiddenDrawn` is this bound for a session, against the
  committed transcript fixed at registration;
* **pinning**: the relations pin every committed value's derivation (`ProofLayer.Pins`), a property of the
  Definitions the proofs pass, which `PouwSegCheck_v2` lacks: its window's absorbed prefix-word sets but the drawn
  step's are free witness (note:20261017T1723Z-draft-ledger §1.2).

`ProofLayer` is what a window's proofs carry on each path of coins: per round `w < S` and proof `i < n_π`, whether the
verifier accepted it, the values its opening gives the round's leaves (its session's committed transcript, salts
included), and its relation (the public Program it must pass, with the leaves' rows among its outputs). The values the
round committed (`ProofLayer.committed`) are its proofs' opening on a reference continuation: among the continuations
of the path's coins before round `w` (`splice`), which share the leaves committed before σ_w, and on which every proof
of round `w` is accepted, one whose opening is the most frequent. They are a function of the coins before round `w`.
-/

namespace Accounting.TS

open Pouw.RelativeSecurity

/-- `x`'s coins before round `w`, then `z`'s from round `w` on: the run that shares `x`'s commitments up to σ_w and
continues on fresh coins. -/
def splice {Ω : Type} : {n : ℕ} → ℕ → Paths Ω n → Paths Ω n → Paths Ω n
  | 0, _, _, _ => ()
  | _ + 1, 0, _, z => z
  | _ + 1, w + 1, x, z => (x.1, splice w x.2 z.2)

/-- **A window's proofs** over `S` rounds of coins on `Ω`, `n_π` proofs a round, opening the round's leaves to values in
`V`: on each path, whether the verifier accepted proof `i` of round `w` (`accepted`) and the values its opening gives
the round's leaves (`opened`); proof `i` of round `w`'s relation on those values (`rel`), its Definition with the
leaves' rows among its outputs; and `right`, that every value of the window is right for its derivation from the
values it is derived from (each round's and proof's, in `Fin S → Fin n_π → V`). -/
structure ProofLayer (Ω : Type) (S nπ : ℕ) (V : Type) where
  accepted : Paths Ω S → Fin S → Fin nπ → Prop
  opened : Paths Ω S → Fin S → Fin nπ → V
  rel : Fin S → Fin nπ → V → Prop
  right : (Fin S → Fin nπ → V) → Prop

namespace ProofLayer

variable {Ω : Type} {S nπ : ℕ} {V : Type} (pf : ProofLayer Ω S nπ V)

/-- Every proof of round `w` is accepted. -/
def Proved (x : Paths Ω S) (w : Fin S) : Prop := ∀ i, pf.accepted x w i

open Classical in
/-- The fresh coins `z` on whose continuation of `x` from round `w` (`splice w x z`) every proof of round `w` is
accepted. -/
noncomputable def provedAt [Fintype Ω] (x : Paths Ω S) (w : Fin S) : Finset (Paths Ω S) :=
  Finset.univ.filter fun z => pf.Proved (splice w x z) w

open Classical in
/-- How many of `provedAt x w` open round `w`'s leaves as the continuation on `z` does. -/
noncomputable def weight [Fintype Ω] (x z : Paths Ω S) (w : Fin S) : ℕ :=
  ((pf.provedAt x w).filter fun z' => ∀ i, pf.opened (splice w x z') w i = pf.opened (splice w x z) w i).card

open Classical in
/-- **The continuation round `w`'s commitment is read at**: among the continuations of `x` from round `w` on which every
proof of round `w` is accepted, one whose opening of round `w`'s leaves is the most frequent; `x` when there is none. -/
noncomputable def ref [Fintype Ω] (x : Paths Ω S) (w : Fin S) : Paths Ω S :=
  if h : (pf.provedAt x w).Nonempty then
    splice w x ((pf.provedAt x w).exists_max_image (fun z => pf.weight x z w) h).choose
  else x

/-- **The values round `w` committed before σ_w**: its proofs' opening of its leaves on the reference continuation. -/
noncomputable def committed [Fintype Ω] (x : Paths Ω S) (w : Fin S) (i : Fin nπ) : V := pf.opened (pf.ref x w) w i

/-- **Binding fails at round `w`**: every proof of round `w` is accepted, and one opens the round's leaves to values
other than those the round committed. -/
def Moved [Fintype Ω] (x : Paths Ω S) (w : Fin S) : Prop := pf.Proved x w ∧ ∃ i, pf.opened x w i ≠ pf.committed x w i

/-- **Two runs off one commitment open it differently** (the fork form): on `x` and on its continuation from round `w`
on fresh coins `z`, which share round `w`'s leaves, every proof of round `w` is accepted, and some proof's openings
differ. Where the proofs open hm96 rows of one frame-v3-sha512 tree against its root, different openings are two
different inputs hashed to one SHA-512 digest along their paths: the break the 8 Oct ruling ties to the run, which
`cr/sha-512` bounds at a finder that runs the window twice from round `w`. -/
def Forked (w : Fin S) (x z : Paths Ω S) : Prop :=
  pf.Proved x w ∧ pf.Proved (splice w x z) w ∧ ∃ i, pf.opened x w i ≠ pf.opened (splice w x z) w i

/-- Proof `i` of round `w` is accepted, and its opening fails its relation. -/
def Unsound (x : Paths Ω S) (w : Fin S) (i : Fin nπ) : Prop := pf.accepted x w i ∧ ¬ pf.rel w i (pf.opened x w i)

/-- **Proof soundness at `δ_ZK`**: each proof is accepted with an opening that fails its relation with probability at
most `δ_ZK` over the window's coins. -/
def SoundAt [Fintype Ω] (δzk : ℝ) : Prop := ∀ w i, prob (fun x => pf.Unsound x w i) ≤ δzk

/-- **The relations pin every committed value**: values that satisfy every proof's relation are right for their
derivations. For a window check this says every word it absorbs is fixed by what it reads (the committed operands and
the committed incoming set), so the committed receipt word is the fold of the honest prefix words. -/
def Pins : Prop := ∀ c : Fin S → Fin nπ → V, (∀ w i, pf.rel w i (c w i)) → pf.right c

/-- **The play is its proofs'** (𝒫_full): the window is accepted only if every proof of every round is, and the ideal
game's check is the rounds' proofs as if every proof were sound, so committed values that satisfy every proof's relation
are accepted by the ideal game. -/
def Of [Fintype Ω] {K T L U : Type} (P : Play Ω S K T L U) : Prop :=
  (∀ x, P.accepts x → ∀ w, pf.Proved x w) ∧ ∀ x, (∀ w i, pf.rel w i (pf.committed x w i)) → P.game.accepts x

end ProofLayer

end Accounting.TS
