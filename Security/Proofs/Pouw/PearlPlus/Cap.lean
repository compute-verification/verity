import Proofs.Pouw.PearlPlus.Assembly

/-!
# Pearl+'s bridge, part (c): a passed tile is within the game's inert-work bound

The bound the game reads is `TileGood`'s first conjunct, `capOK` of `pearlCTilesDevRev1 devH100` at the deployed
semantics and `ρ = 1/400`: the tile's replayed inert work (`tileDebit`: its flagged atom-words and its forming) on its
rows that pass the rules and all its columns is at most `ρ·creditOf` times their share. Acceptance doesn't read it:
`Draw` puts a sampled tile with good words in `goodDeclTiles` within the bound and in `overCapTiles` over it. The credit
does: the tolerant audit's deduction grows with the samples on `overCapTiles`, and a passed tile within the bound keeps
those to the samples the check finds over its own bound.

The check's bound (`Rule.checkOpened`'s `capFailure`) sums the chain's inert work over the record's credited rows and
columns only, those neither excluded nor voluntary, and has no forming term. So `TilePassGivesTheCap` is stated and not
proved: a voluntary row passes the rules, and a voluntary column's cells are in the tile, so the game's bound counts
inert work the check never reads.
-/

namespace Pouw.PearlPlus.Bridge

open Pouw.PearlC Pouw.Fp8Atom PouwCheck.PearlPlus

/-- **(c) A passed tile is within the inert-work bound**: a tile of a pass `load` gives, which is audit tile `g` of `U`,
under the semantics' salt and weight root, whose weight rows all pass the rules (`TileGood`'s second conjunct, as (d)
takes it), whose openings are its unit's commitment `c`, and that `checkTile` passes on its weight's noise, has its
replayed inert work within the game's bound at `ρ = 1/400`. -/
def TilePassGivesTheCap : Prop :=
  ∀ {Q R S : Type} (saltOf : S → List ℕ) (rootOf : ℕ → List ℕ) (H : Q → R) (s : S)
    (coins : PouwCommit.CoinCommitment) (man : Lean.Json) (p : Pass) (tid : ℕ) (o : TileOpening)
    (U : Pouw.PearlC.Workload) (g : ℕ) (c : Pouw.PearlC.Commit),
    PouwCheck.PearlPlus.load coins man = .ok p → TileOf p tid U g → bytesNat p.salt = saltOf s →
    bytesNat ((p.weightRoots.lookup (U.layout.wid ((auditTiling 64 64 U.layout).unit g))).getD ByteArray.empty) =
      rootOf (U.layout.wid ((auditTiling 64 64 U.layout).unit g)) →
    (∀ j ∈ (auditTiling 64 64 U.layout).cols g,
      (protocolH3 saltOf rootOf (1 / 400) : Pouw.PearlC.Protocol Q R S).weightOK
        (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
        (U.weight (U.layout.wid ((auditTiling 64 64 U.layout).unit g))) j) →
    OpensCommitted p tid o U ((auditTiling 64 64 U.layout).unit g) c →
    checkTile p (Properties.Pouw.PearlPlus.noiseOf p tid) tid o = none →
      (tilesH3 saltOf rootOf (1 / 400) : Pouw.PearlC.TileRules Q R S).capOK H s U g c.act

end Pouw.PearlPlus.Bridge
