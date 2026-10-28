import Definitions.Pouw.PearlC.HiddenTileGame
import Proofs.Pouw.PearlC.FlockTile.Bridge

/-!
# C-Flock's hidden audit: the TileRead session per drawn tile

The hidden audit (`HiddenAudit`) plays, for each drawn tile, a proof game on the window's commitment. For C-Flock that
game is the live `--zk` session of the TileRead statement at the tile's public `t_s`, which reads the window's committed
rows at `tile_at(t_s)`, carried into core's games by the bridge (`toVerity`) and relabelled to flock-verify's verdict.

TileRead's Definition and its C-Flock soundness (the restack plan's T1 and T3) are not in Lean yet, so the session is a
parameter (`TileReadSession`): its window root, its live game per tile, its verdict, the class of strategies its
soundness speaks of, and the event that soundness bounds. From it:

* `TileReadSession.audit`: the `HiddenAudit` value, with `Bounded = BoundedByInduced TileReadSession.cls`;
* `TileReadSession.tileGame`: a `TileGame` for every tile of every in-domain workload, from the two open obligations
  `TileReadSound` (T3, giving `bad_le`) and `TileReadBinds` (the binding, giving `good_of_accept`).
-/

namespace Pouw.PearlC.FlockTile

open scoped ENNReal

variable {Q R S : Type}

/-- **C-Flock's TileRead session, as the hidden audit plays it per drawn tile.** A parameter until TileRead's
Definition and its C-Flock soundness land.

* `Root` is the window root, the frame-v3-sha512 root over the window's hm96-sha512 rows, which the verifier holds, and
  `root U τ o` is the window root of the transcript `τ` committed under the opening `o` (its rows' salts).
* `game H s U r g` is the live `--zk` session of the TileRead statement at the public `t_s` with `tile_at(t_s) = g`, on
  the window root `r`, with the anchors the verifier recomputes from the oracle `H` and the salt `s`: C-Flock's game
  (`FlockSoundness.Game`), ending in the live audit's output (for C-Flock's `auditLive`, the registration, the draw and
  the session's transcript). `verdict` is flock-verify's verdict on that output.
* `Cls U τ o H s g σ q` is the class C-Flock's soundness speaks of, for a strategy `σ` of tile `g`'s session, at budget
  `q`: T3's CR hypotheses, `cr/sha-512` for each table's knowledge finders (`TableCRZ`) and `ecr/sha-512` for the link
  finders (`LinkCRZC`), of the finders built from `σ`. It is stated without reading which of `τ`'s tiles are good.
* `Bad U τ o H s g σ x` is the event T3 bounds (for C-Flock, a drawn unit wrong under the extracted witness, or a
  finder's success). -/
structure TileReadSession (Q R S : Type) where
  Root : Type
  Opening : Type
  root : Workload → CommitLog → Opening → Root
  Out : Workload → Type
  game : (Q → R) → S → (U : Workload) → Root → ℕ → FlockSoundness.Game (Out U)
  verdict : {U : Workload} → Out U → Bool
  Cls : (U : Workload) → (τ : CommitLog) → (o : Opening) → (H : Q → R) → (s : S) → (g : ℕ) →
    FlockSoundness.Game.Strategy (game H s U (root U τ o) g) → ℕ → Prop
  Bad : (U : Workload) → (τ : CommitLog) → (o : Opening) → (H : Q → R) → (s : S) → (g : ℕ) →
    FlockSoundness.Game.Strategy (game H s U (root U τ o) g) → Out U → Prop

namespace TileReadSession

variable (F : TileReadSession Q R S)

/-- Tile `g`'s proof game as the hidden audit plays it: the session's live game, through the bridge, relabelled to
flock-verify's verdict. -/
def proofGame (H : Q → R) (s : S) (U : Workload) (r : F.Root) (g : ℕ) : Verity.Game Bool :=
  (toVerity (F.game H s U r g)).map F.verdict

/-- The strategy of the session's game that a strategy of tile `g`'s proof game is. -/
def stratOf {H : Q → R} {s : S} {U : Workload} {r : F.Root} {g : ℕ}
    (σ : Verity.Game.Strategy (F.proofGame H s U r g)) : FlockSoundness.Game.Strategy (F.game H s U r g) :=
  stratOfVerity _ (Verity.Game.Strategy.ofMap F.verdict _ σ)

/-- **The hidden audit's single-game class.** A strategy `σ` of a game `G`, on the transcript `τ` committed under the
opening `o`, is in the class at `q` when, at every tile whose proof game `G` is (for every workload, oracle and salt), the
session's strategy that `σ` is meets the session's class `Cls` there. -/
def cls (τ : CommitLog) (o : F.Opening) (G : Verity.Game Bool) (σ : Verity.Game.Strategy G) (q : ℕ) : Prop :=
  ∀ (U : Workload) (H : Q → R) (s : S) (g : ℕ) (e : F.proofGame H s U (F.root U τ o) g = G),
    F.Cls U τ o H s g (F.stratOf (cast (congrArg Verity.Game.Strategy e.symm) σ)) q

/-- **C-Flock's hidden audit.** The root is the window root; each drawn tile's proof game is the TileRead session's live
game through the bridge (`proofGame`); and the class is the batch provers whose every induced strategy on a tile's game
is in the session's class (`BoundedByInduced cls`). -/
def audit : HiddenAudit Q R S where
  Root := F.Root
  Opening := F.Opening
  root := F.root
  game := F.proofGame
  Bounded := BoundedByInduced F.cls

end TileReadSession

/-- **T3 at the TileRead session (open obligation, to be discharged by C-Flock's soundness).** For every in-domain
workload, transcript `τ`, opening `o`, oracle, salt and tile `g`, and every strategy `σ` of tile `g`'s session in the
session's class at `q`: flock-verify accepts on a run in the event `Bad` with probability at most `ηzk q`. To be met by
the restack plan's T3 (`Flock.Guarantees.EndToEndCanonicalP` at the drawn units, with its CR hypotheses as `Cls` and
its event as `Bad`), at the TileRead statement. -/
def TileReadSound (P : Protocol Q R S) (D : UnitLayout → Prop) (F : TileReadSession Q R S) (ηzk : ℕ → ℝ) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (τ : CommitLog) (o : F.Opening) (H : Q → R) (s : S) (g : ℕ)
    (σ : FlockSoundness.Game.Strategy (F.game H s U (F.root U τ o) g)) (q : ℕ), F.Cls U τ o H s g σ q →
    FlockSoundness.Game.prob (fun x => F.verdict x = true ∧ F.Bad U τ o H s g σ x) (F.game H s U (F.root U τ o) g) σ ≤
      ENNReal.ofReal (ηzk q)

/-- **The binding (open obligation, to be proved).** For every in-domain workload, transcript `τ`, opening `o`, oracle,
salt and tile `g`: an output of tile `g`'s session that flock-verify accepts outside `Bad` happens only when tile `g` is
good or blank in `τ`. To be met by TileRead's gates (an accepted instance at the public `t_s` read the window's committed
rows at `tile_at(t_s)` and the tile's leaf matches), the registered reads binding those rows to the window root (L4),
hm96-sha512's binding to `τ`'s rows under `o`, the tile check computing the good-tile predicate (L3) and the pinned
anchors (L8). -/
def TileReadBinds (P : Protocol Q R S) (TR : TileRules Q R S) (D : UnitLayout → Prop) (F : TileReadSession Q R S) :
    Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (τ : CommitLog) (o : F.Opening) (H : Q → R) (s : S) (g : ℕ)
    (σ : FlockSoundness.Game.Strategy (F.game H s U (F.root U τ o) g)) (x : F.Out U),
    F.verdict x = true → ¬ F.Bad U τ o H s g σ x → g ∈ goodTiles P TR U H s τ ∪ blankTiles TR U τ

/-- **A `TileGame` per tile from T3 and the binding.** Tile `g`'s proof game is the session's game through the bridge
(`game_eq` by definition); its `Bad` is the session's, read at the session's strategy; `bad_le` is `TileReadSound`
through the bridge's `prob_ofVerity`; and `good_of_accept` is `TileReadBinds`. -/
def TileReadSession.tileGame [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : UnitLayout → Prop) (F : TileReadSession Q R S) (ηzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ ηzk q)
    (hT3 : TileReadSound P D F ηzk) (hbind : TileReadBinds P TR D F) (U : Workload) (hU : U.InDomain P D)
    (τ : CommitLog) (o : F.Opening) (H : Q → R) (s : S) (g : ℕ) :
    TileGame P TR F.audit F.cls ηzk U τ o H s g where
  Out := F.Out U
  G := toVerity (F.game H s U (F.root U τ o) g)
  verdict := F.verdict
  game_eq := rfl
  Bad σ x := F.Bad U τ o H s g (stratOfVerity _ σ) x
  bad_le q σ hcls := by
    have hc := hcls U H s g rfl
    change F.Cls U τ o H s g (stratOfVerity _ (Verity.Game.Strategy.ofMap F.verdict _
      (Verity.Game.Strategy.toMap F.verdict (toVerity (F.game H s U (F.root U τ o) g)) σ))) q at hc
    rw [ofMap_toMap] at hc
    have hb := hT3 U hU τ o H s g _ q hc
    rw [prob_ofVerity] at hb
    exact ENNReal.toReal_le_of_le_ofReal (h0 q) hb
  good_of_accept σ x hx hb := hbind U hU τ o H s g _ x hx hb

end Pouw.PearlC.FlockTile
