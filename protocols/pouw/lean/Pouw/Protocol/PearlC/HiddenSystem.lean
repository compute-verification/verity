import Pouw.Protocol.PearlC.HiddenTileGame
import Pouw.Protocol.PearlC.TileCheck
import Pouw.Protocol.Draft.SampledProofs
import Pouw.Protocol.Draft.BatchSeq

/-!
# The hidden audit of tile checks sound against registered values (trusted)

`HiddenAudit.ofSystem` is the hidden audit whose drawn tiles are proved by proof systems of sampled proofs
(`Pouw.Draft.SampledProofs.System`, provisional). Tile `t` of workload `U` is checked by a circuit of kind
`kind U t` at its checked gates `out (kind U t)`, and the values its commitment registers on the oracle `H`, the salt
`s` and the transcript `τ` are `valueAt H s U τ t`: the anchors the verifier derives from `H`, `s` and `U` (L8), and
the rows and words of `τ`'s unit. The commitment to `τ` is each tile's commitment to its registered values, under the
tile's own opening. The class is `BoundedByInduced` of the tile checks' own classes. `ofTileCheck` is the hidden system
of a tile check (`TileCheck`): a tile is checked on its unit's circuit at its words.

A tile's proof game does not say which tile it is, or on which oracle, salt and workload. Commitments to different
values can collide, and then their games are equal. A class read off the game must hold the proof system's class for
every reading of the game, and the honest prover for one tile's values is not in the class for a colliding reading's
values (the registered-path finder finds the collision). Some other oracle's values collide with any tile's, by
counting. So the commitment names the oracle, salt and workload it is for, the verifier rejects one that names others
(for C-Flock, the anchors in the commitment's public part against the verifier's own recomputation: L8), and the class
reads a game at the name. The honest prover is then in the class unless two of its own tiles' proof games are equal on
different values (`HiddenSystem.Distinct`), or its batch can be read as another list of games
(`Pouw.Draft.Game.Unambiguous`).

`ChecksGood` is L3 on the hidden system: a tile with a transcript correct on the cone of its checked gates and holding
its registered values is good or blank. `TileCheck.Computes` gives it for `ofTileCheck`. `Covers`: an accepted proof
has drawn the cone of the checked gates.
-/

namespace Pouw.PearlC

open Pouw.Draft.SampledProofs

variable {Q R S : Type}

/-- **The tile checks as proof systems**: the kinds `K` of tile check, each a circuit `C k` cut into proof units by
`part k`, checked at the gates `out k`, and a proof system on openings of type `O` and roots of type `Rt`; the kind of
tile `t` of workload `U`; and the values the tile check registers on the oracle `H`, the salt `s` and the transcript
`τ`: the anchors from `H`, `s` and `U`, and `τ`'s rows and words of the tile's unit. -/
structure HiddenSystem (Q R S : Type) where
  K : Type
  C : K → Verity.Circuit
  n : K → ℕ
  part : (k : K) → Verity.Partition (C k) (n k)
  out : (k : K) → Finset (Fin (C k).N)
  O : Type
  Rt : Type
  sys : (k : K) → System (C k) (n k) (part k) O Rt
  kind : Workload → ℕ → K
  valueAt : (Q → R) → S → (U : Workload) → Transcript → (t : ℕ) → (C (kind U t)).Transcript

namespace HiddenSystem

variable (HS : HiddenSystem Q R S)

/-- The opening: the oracle, salt and workload the commitment names, and each tile's opening. -/
abbrev Opening : Type := ((Q → R) × S × Workload) × (ℕ → HS.O)

/-- The root on each oracle, salt and tile: the name, and the tile's commitment to its registered values. -/
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
tile-check class on its registered values and opening. -/
def Cls (τ : Transcript) (o : HS.Opening) (g : Verity.Game Bool) (σ : Verity.Game.Strategy g) (q : ℕ) : Prop :=
  ∀ (t : ℕ) (e : g = HS.tileGame o.1.2.2 o.1.1 o.1.2.1 τ o t),
    (HS.sys (HS.kind o.1.2.2 t)).Cls (HS.valueAt o.1.1 o.1.2.1 o.1.2.2 τ t) (o.2 t)
      (Verity.Game.Strategy.ofMap (HS.sys (HS.kind o.1.2.2 t)).accept _ (cast (congrArg Verity.Game.Strategy e) σ)) q

/-- **The tile checks' draw covers their checked gates**: an accepted proof has drawn every unit of their cone. -/
def Covers : Prop :=
  ∀ (k : HS.K) (x : (HS.sys k).Out), (HS.sys k).accept x = true → (HS.part k).cone (HS.out k) ⊆ (HS.sys k).drawn x

/-- **L3 on the hidden system: the tile check computes the good-tile predicate.** On every in-domain workload, a tile
with a transcript `X` of its tile check, correct on every unit of the cone of its checked gates and holding its
registered values, is good or blank. -/
def ChecksGood [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S) (TR : TileRules Q R S)
    (D : Layout → Prop) : Prop :=
  ∀ U : Workload, U.InDomain P D → ∀ (H : Q → R) (s : S) (τ : Transcript) (t : ℕ), t < (TR.tiling U.layout).NT →
    ∀ X : (HS.C (HS.kind U t)).Transcript,
      (∀ v ∈ (HS.part (HS.kind U t)).cone (HS.out (HS.kind U t)), (HS.part (HS.kind U t)).Correct X v) →
      (∀ g ∈ (HS.sys (HS.kind U t)).registered, X g = HS.valueAt H s U τ t g) →
      t ∈ goodTiles P TR U H s τ ∪ blankTiles TR U τ

end HiddenSystem

/-- The values unit `u`'s tile check registers on `τ`: the anchors on the public wires and the commitment's values on
the others, and zero for a blank unit, whose tiles are never checked. -/
def TileCheck.registeredAt (TC : TileCheck Q R S) (H : Q → R) (s : S) (U : Workload) (τ : Transcript) (u : ℕ) :
    (TC.C (U.layout.shape u)).Transcript :=
  match τ u with
  | some c => fun w => if w ∈ TC.pub _ then TC.anchorsOf H s U u c.act w else TC.rowsOf U u c w
  | none => fun _ => 0

/-- **The hidden system of a tile check**: a tile of kind (its unit's shape, its rows, its columns) is checked on its
unit's circuit at its words (`TileCheck.out`), by the proof system `sys` of its kind, on its unit's registered values
(`TileCheck.registeredAt`). -/
def HiddenSystem.ofTileCheck (TC : TileCheck Q R S) (TR : TileRules Q R S) (O Rt : Type)
    (sys : (k : Shape × Finset ℕ × Finset ℕ) → System (TC.C k.1) (TC.n k.1) (TC.part k.1) O Rt) :
    HiddenSystem Q R S where
  K := Shape × Finset ℕ × Finset ℕ
  C k := TC.C k.1
  n k := TC.n k.1
  part k := TC.part k.1
  out k := TC.out k.1 k.2.1 k.2.2
  O := O
  Rt := Rt
  sys := sys
  kind U t := (U.layout.shape ((TR.tiling U.layout).unit t), (TR.tiling U.layout).rows t, (TR.tiling U.layout).cols t)
  valueAt H s U τ t := TC.registeredAt H s U τ ((TR.tiling U.layout).unit t)

/-- **The hidden audit of the tile checks**: each tile's commitment to its registered values, the tile check's proof
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
/-- The honest prover of tile `t`'s proof game: the tile check's honest prover on the tile's registered values and
opening (a game that rejects at once has no moves). -/
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

/-- **No two tiles' proof games are equal on different values**: on `τ` under `o`, read at `H`, `s` and `U`, tiles
whose proof games are equal have the same kind, registered values and opening. -/
def Distinct (U : Workload) (H : Q → R) (s : S) (τ : Transcript) (o : HS.Opening) : Prop :=
  ∀ t t' : ℕ, HS.tileGame U H s τ o t = HS.tileGame U H s τ o t' →
    (⟨HS.kind U t, HS.valueAt H s U τ t, o.2 t⟩ : (k : HS.K) × (HS.C k).Transcript × HS.O) =
      ⟨HS.kind U t', HS.valueAt H s U τ t', o.2 t'⟩

end HiddenSystem

end Pouw.PearlC
