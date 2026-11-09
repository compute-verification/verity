module

public import Definitions.Pouw.PearlC.FlockTile.Audit
public import Proofs.Core.Game.Prob

/-!
# Flock's hidden audit: a `TileGame` per sampled tile

The TileRead session, Flock's `HiddenAudit` value over it and its two open obligations are in
`Definitions.Pouw.PearlC.FlockTile.Audit`. Here: `TileReadSession.tileGame`, a `TileGame` for every tile of every
in-domain workload, from `TileReadSound` (T3, giving `bad_le`) and `TileReadBinds` (the binding, giving
`good_of_accept`).
-/

@[expose] public section

namespace Pouw.PearlC.FlockTile

variable {Q R S : Type}

/-- **A `TileGame` per tile from T3 and the binding.** Tile `g`'s proof game is the session's game (`game_eq` by
definition); its `Bad` is the session's; `bad_le` is `TileReadSound`; and `good_of_accept` is `TileReadBinds`. -/
def TileReadSession.tileGame [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (F : TileReadSession Q R S) (δzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ δzk q)
    (hT3 : TileReadSound P D F δzk) (hbind : TileReadBinds P TR D F) (U : Workload) (hU : U.InDomain P D)
    (τ : Transcript) (o : F.Opening) (H : Q → R) (s : S) (g : ℕ) :
    TileGame P TR F.audit F.cls δzk U τ o H s g where
  Out := F.Out U
  G := F.game H s U (F.root U τ o) g
  verdict := F.verdict
  game_eq := rfl
  Bad σ x := F.Bad U τ o H s g σ x
  bad_le q σ hcls := by
    have hc := hcls U H s g rfl
    change F.Cls U τ o H s g (Verity.Game.Strategy.ofMap F.verdict _
      (Verity.Game.Strategy.toMap F.verdict (F.game H s U (F.root U τ o) g) σ)) q at hc
    rw [Verity.Game.Strategy.ofMap_toMap] at hc
    exact ENNReal.toReal_le_of_le_ofReal (h0 q) (hT3 U hU τ o H s g _ q hc)
  good_of_accept σ x hx hb := hbind U hU τ o H s g _ x hx hb

end Pouw.PearlC.FlockTile
