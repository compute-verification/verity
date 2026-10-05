import FlockSoundness.Discharge.Recursion.Stage
import FlockSoundness.Audit.OneStage
import RunModel.Guarantees

/-!
# The run's soundness, from C-Flock's recursive audit (plan; not built)

The chain, each step a declaration below:

1. `tree_hybrid`: the coin server that draws the whole coin tree at `Hello` against one that draws each round's coins
   when it reveals them (`lazy`, to be written beside `CoinServer`): within `ErrorTerms.tree`, by `Hm96Hiding` on the
   tree's leaves.
2. `view_le_game`: a view adversary of the lazy run is a strategy of `soundGame` (it sees less than a strategy does),
   so the run's probability is at most the game's.
3. `loop_refines_recGame`: `soundGame` with the auditor's machines plays C-Flock's `recGame L Reg I Cm outer`:
   the registration commit is `Reg`, the draw `L.Ω`, each inner slot one of `rounds Cm I.Coin I.r`, the outer session
   V*'s `ZkOuter.game`; the verifier accepting implies `LiveAcceptsZKR`. Every event schedule the developer picks is
   one interleaving of those rounds, refused frames included, because each party is a `Core.seq` and the coin server
   answers each slot once, after its commitment (`Run.Rules`). C-Flock's `Game` is its own type, core's shape: the
   step also carries a strategy across.
4. `zk_session_recursive_full` bounds `recGame`'s accept-and-false-inner-statement by `ErrorTerms.outer` and `.inner`,
   and by `cr/sha-512` on its finders (`hCRo`, `hCR`), which are `extraction`'s.
5. OneStage (`audit_count`, through the inner statement `x R ω` holding on the drawn units) and `LawDelta` give the
   draw's `δ` on the committed transcript, the extraction's.
-/

namespace RunModelProofs

open RunModel Run
open scoped ENNReal

/-- The run's extraction: at each history, V*'s witness read off the outer strategy (`ZkOuter.layer`), decoded into the
inner messages (`wmsg`), and the registered rows they open, as core's transcript; its finders are
`zk_session_recursive_full`'s. Trusted text in the end (`RunModel.Extraction`). -/
noncomputable def extraction (A : Agreement) (cl : Claim A) : Extraction A cl := sorry

/-- The run's terms: `ZkOuter.bound` and the stages' slack (`outer`), the inner protocol's `εin` (`inner`), and
`tree_hybrid`'s (`tree`). -/
noncomputable def terms (A : Agreement) (cl : Claim A) : ErrorTerms := sorry

/-- The coin server that draws each outer round's coins when it reveals them, and answers `Hello` with a root and key
of fresh bytes. -/
noncomputable def lazy (A : Agreement) : Machine := sorry

/-- The soundness setting with `lazy` in the coin server's place. -/
noncomputable def lazySetting (A : Agreement) : Setting :=
  { Setting.sound A with roles := Function.update (Roles.honest A) .coinServer (lazy A) }

/-- The soundness run in `lazySetting`, as `soundRun`. -/
noncomputable def lazyRun (A : Agreement) (dev : Adversary) :
    ℕ → World (lazySetting A) → PMF (World (lazySetting A)) := sorry

/-- An accepting run of either setting while more than `threshold` units of `X` are wrong. -/
def Bad {S : Setting} (A : Agreement) (cl : Claim A) (X : cl.C.Transcript) (w : World S) : Prop :=
  w.accepted.isSome ∧ A.threshold < (cl.P.wrong X).card

/-- Step 1. -/
theorem tree_hybrid (A : Agreement) (cl : Claim A) (δ₁ : ℝ≥0∞) (h : Assumptions.Hm96Hiding δ₁) (fuel : ℕ)
    (dev : Adversary) :
    (soundRun A dev fuel (World.init _)).toOuterMeasure
        {w | Violated A cl ((extraction A cl).committed fuel dev) w} ≤
      (lazyRun A dev fuel (World.init _)).toOuterMeasure
          {w | Bad A cl ((extraction A cl).committed fuel dev) w} + (terms A cl).tree := sorry

/-- Step 2: a strategy of core's game sees every coin; in `lazySetting` every coin drawn is one already revealed. -/
theorem view_le_game (A : Agreement) (cl : Claim A) (fuel : ℕ) (dev : Adversary) :
    ∃ s : Verity.Game.Strategy (soundGame (lazySetting A) fuel (World.init _)),
      (lazyRun A dev fuel (World.init _)).toOuterMeasure
          {w | Bad A cl ((extraction A cl).committed fuel dev) w} ≤
        Verity.Game.prob (fun w => Bad A cl ((extraction A cl).committed fuel dev) w)
          (soundGame (lazySetting A) fuel (World.init _)) s := sorry

/-- Step 3, the one new piece of soundness: the auditor's loop refines C-Flock's recursive audit. Its statement names
`recGame`'s parameters as functions of `A`, which this plan does not fix. -/
theorem loop_refines_recGame (A : Agreement) : True := sorry

/-- The guarantee. -/
theorem soundness (A : Agreement) (cl : Claim A) :
    Guarantees.Soundness A cl (extraction A cl) (terms A cl) := sorry

end RunModelProofs
