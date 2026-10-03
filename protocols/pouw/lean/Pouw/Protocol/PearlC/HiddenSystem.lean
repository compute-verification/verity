import Pouw.Protocol.PearlC.HiddenTileGame
import Pouw.Protocol.Draft.SampledProofs
import Pouw.Protocol.Draft.BatchSeq

/-!
# The hidden audit of tile checks sound against registered inputs (trusted)

`HiddenAudit.ofSystem` is the hidden audit whose drawn tiles are proved by proof systems of sampled proofs
(`Pouw.Draft.SampledProofs.System`, provisional). Tile `t` of workload `U` has a tile-check circuit of kind
`kind U t`, whose inputs on the oracle `H`, the salt `s` and the transcript `τ` are `valueAt H s U τ t`: the tile's rows
in `τ`, and the anchors the verifier recomputes from `H`, `s` and `U` (L8). The commitment to `τ` is each tile's
commitment to its inputs, under the tile's own opening. The class is `BoundedByInduced` of the tile checks' own
classes.

A tile's proof game does not say which tile it is, or on which oracle, salt and workload. Commitments to different
inputs can collide, and then their games are equal. A class read off the game must hold the proof system's class for
every reading of the game, and the honest prover for one tile's inputs is not in the class for a colliding reading's
inputs (the registered-path finder finds the collision). Some other oracle's inputs collide with any tile's, by
counting. So the commitment names the oracle, salt and workload it is for, the verifier rejects one that names others
(for C-Flock, the anchors in the commitment's public part against the verifier's own recomputation: L8), and the class
reads a game at the name. The honest prover is then in the class unless two of its own tiles' proof games are equal on
different inputs (`HiddenSystem.Distinct`), or its batch can be read as another list of games
(`Pouw.Draft.Game.Unambiguous`).

`ChecksGood` is L3 (a placeholder for the statement of the l3-tilecheck lane): a tile whose tile-check inputs evaluate
to the public values at the circuit's outputs is good or blank. `Covers`: an accepted tile-check proof has drawn every
unit that feeds an output.
-/

namespace Pouw.PearlC

open Pouw.Draft.SampledProofs

variable {Q R S : Type}

/-- **The tile checks as proof systems**: the kinds `K` of tile-check circuit, each a proof system on openings of type
`O` and roots of type `Rt`; the kind of tile `t` of workload `U`; and the tile check's inputs on the oracle `H`, the
salt `s` and the transcript `τ`: `τ`'s rows of tile `t` and the anchors from `H`, `s` and `U`. -/
structure HiddenSystem (Q R S : Type) where
  K : Type
  O : Type
  Rt : Type
  sys : K → System O Rt
  kind : Workload → ℕ → K
  valueAt : (Q → R) → S → (U : Workload) → Transcript → (t : ℕ) → (sys (kind U t)).C.Transcript

namespace HiddenSystem

variable (HS : HiddenSystem Q R S)

/-- The opening: the oracle, salt and workload the commitment names, and each tile's opening. -/
abbrev Opening : Type := ((Q → R) × S × Workload) × (ℕ → HS.O)

/-- The root on each oracle, salt and tile: the name, and the tile's commitment to its tile-check inputs. -/
abbrev Root : Type := (Q → R) → S → ℕ → ((Q → R) × S × Workload) × HS.Rt

/-- The commitment of `τ` under `o`, on workload `U`. -/
def root (U : Workload) (τ : Transcript) (o : HS.Opening) : HS.Root :=
  fun H s t => (o.1, (HS.sys (HS.kind U t)).commit (HS.valueAt H s U τ t) (o.2 t))

open Classical in
/-- Tile `t`'s proof game on the root `r`: the tile check's game on the tile's commitment, relabelled to its verdict,
when the root names the verifier's `H`, `s` and `U`; reject otherwise. -/
noncomputable def game (H : Q → R) (s : S) (U : Workload) (r : HS.Root) (t : ℕ) : Verity.Game Bool :=
  if (r H s t).1 = (H, s, U) then ((HS.sys (HS.kind U t)).game (r H s t).2).map (HS.sys (HS.kind U t)).accept
  else .ret false

/-- Tile `t`'s proof game on `τ` under `o`, read at `H`, `s` and `U`: the tile check's game on the tile's commitment,
relabelled to its verdict. -/
def tileGame (U : Workload) (H : Q → R) (s : S) (τ : Transcript) (o : HS.Opening) (t : ℕ) : Verity.Game Bool :=
  ((HS.sys (HS.kind U t)).game ((HS.sys (HS.kind U t)).commit (HS.valueAt H s U τ t) (o.2 t))).map
    (HS.sys (HS.kind U t)).accept

/-- **The single-game class.** The strategy `σ` of the game `g`, on `τ` under `o`, is in the class at `q` when, for
every tile `t` whose proof game, read at the oracle, salt and workload `o` names, is `g`, `σ` is in tile `t`'s
tile-check class on its inputs and opening. -/
def Cls (τ : Transcript) (o : HS.Opening) (g : Verity.Game Bool) (σ : Verity.Game.Strategy g) (q : ℕ) : Prop :=
  ∀ (t : ℕ) (e : g = HS.tileGame o.1.2.2 o.1.1 o.1.2.1 τ o t),
    (HS.sys (HS.kind o.1.2.2 t)).Cls (HS.valueAt o.1.1 o.1.2.1 o.1.2.2 τ t) (o.2 t)
      (Verity.Game.Strategy.ofMap (HS.sys (HS.kind o.1.2.2 t)).accept _ (cast (congrArg Verity.Game.Strategy e) σ)) q

/-- **The tile checks' draw covers their outputs**: an accepted proof has drawn every unit that feeds an output. -/
def Covers : Prop :=
  ∀ (k : HS.K) (x : (HS.sys k).Out), (HS.sys k).accept x = true →
    (HS.sys k).P.cone (HS.sys k).C.outputs ⊆ (HS.sys k).drawn x

/-- **L3 (placeholder for the l3-tilecheck lane's statement): the tile check computes the good-tile predicate.** On
every in-domain workload, a tile whose tile-check inputs evaluate to the public values at the circuit's outputs is good
or blank. -/
def ChecksGood [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (τ : Transcript) (t : ℕ), t < (TR.tiling U.layout).NT →
    (∀ g ∈ (HS.sys (HS.kind U t)).C.outputs, ∀ b, (HS.sys (HS.kind U t)).fixed g = some b →
      (HS.sys (HS.kind U t)).C.eval (HS.valueAt H s U τ t) g = b) →
    t ∈ goodTiles P TR U H s τ ∪ blankTiles TR U τ

end HiddenSystem

/-- **The hidden audit of the tile checks**: each tile's commitment to its tile-check inputs, the tile check's proof
game, and the class `BoundedByInduced` of the tile checks' own classes. -/
noncomputable def HiddenAudit.ofSystem (HS : HiddenSystem Q R S) : HiddenAudit Q R S where
  Root := HS.Root
  Opening := HS.Opening
  root := HS.root
  game := HS.game
  Bounded := BoundedByInduced HS.Cls

namespace HiddenSystem

variable (HS : HiddenSystem Q R S)

/-- The honest opening on `U`: it names the oracle, salt and workload, with each tile's opening from `salts`. -/
def honestOpening (U : Workload) (salts : (Q → R) → S → ℕ → HS.O) : (Q → R) → S → HS.Opening :=
  fun H s => ((H, s, U), salts H s)

open Classical in
/-- The honest prover of tile `t`'s proof game: the tile check's honest prover on the tile's inputs and opening (a game
that rejects at once has no moves). -/
noncomputable def honestTile (U : Workload) (τ : Transcript) (o : HS.Opening) (H : Q → R) (s : S) (t : ℕ) :
    Verity.Game.Strategy (HS.game H s U (HS.root U τ o) t) :=
  if h : o.1 = (H, s, U) then
    cast (congrArg Verity.Game.Strategy (ite_eq_left h).symm)
      (Verity.Game.Strategy.toMap (HS.sys (HS.kind U t)).accept _
        ((HS.sys (HS.kind U t)).honest (HS.valueAt H s U τ t) (o.2 t)))
  else cast (congrArg Verity.Game.Strategy (ite_eq_right h).symm) PUnit.unit

/-- **The honest prover of the proof phase**: under the honest opening, the drawn tiles' honest provers, one tile
after another in draw order. -/
noncomputable def honestProver (TR : TileRules Q R S) (U : Workload) (τ : (Q → R) → S → Transcript)
    (salts : (Q → R) → S → ℕ → HS.O) (w : ℕ → ℕ) (t : ℕ) :
    (HiddenAudit.ofSystem HS).Prover TR U τ (HS.honestOpening U salts) w t :=
  fun x => cast (congrArg (fun L => Verity.Game.Strategy (Verity.Game.batch L)) List.map_ofFn)
    (Pouw.Draft.Game.Strategy.batchSeq _
      (Pouw.Draft.Game.Strategies.ofMap
        (HS.game x.1 x.2.1 U (HS.root U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1)))
        (HS.honestTile U (τ x.1 x.2.1) (HS.honestOpening U salts x.1 x.2.1) x.1 x.2.1)
        (List.ofFn fun i => ((x.2.2 i).1 : ℕ))))

/-- **No two tiles' proof games are equal on different inputs**: on `τ` under `o`, read at `H`, `s` and `U`, tiles
whose proof games are equal have the same kind, tile-check inputs and opening. -/
def Distinct (U : Workload) (H : Q → R) (s : S) (τ : Transcript) (o : HS.Opening) : Prop :=
  ∀ t t' : ℕ, HS.tileGame U H s τ o t = HS.tileGame U H s τ o t' →
    (⟨HS.kind U t, HS.valueAt H s U τ t, o.2 t⟩ : (k : HS.K) × (HS.sys k).C.Transcript × HS.O) =
      ⟨HS.kind U t', HS.valueAt H s U τ t', o.2 t'⟩

end HiddenSystem

end Pouw.PearlC
