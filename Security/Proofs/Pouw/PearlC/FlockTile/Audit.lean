import Definitions.Pouw.PearlC.FlockTile.Audit
import Proofs.Pouw.PearlC.FlockTile.Bridge

/-!
# C-Flock's hidden audit: a `TileGame` per drawn tile

The TileRead session, C-Flock's `HiddenAudit` value over it and its two open obligations are in
`Definitions.Pouw.PearlC.FlockTile.Audit`. Here: `TileReadSession.tileGame`, a `TileGame` for every tile of every
in-domain workload, from `TileReadSound` (T3, giving `bad_le`) and `TileReadBinds` (the binding, giving
`good_of_accept`).
-/

namespace Pouw.PearlC.FlockTile

variable {Q R S : Type}

/-- **A `TileGame` per tile from T3 and the binding.** Tile `g`'s proof game is the session's game through the bridge
(`game_eq` by definition); its `Bad` is the session's, read at the session's strategy; `bad_le` is `TileReadSound`
through the bridge's `prob_ofVerity`; and `good_of_accept` is `TileReadBinds`. -/
def TileReadSession.tileGame [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] (P : Protocol Q R S)
    (TR : TileRules Q R S) (D : Layout → Prop) (F : TileReadSession Q R S) (ηzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ ηzk q)
    (hT3 : TileReadSound P D F ηzk) (hbind : TileReadBinds P TR D F) (U : Workload) (hU : U.InDomain P D)
    (τ : Transcript) (o : F.Opening) (H : Q → R) (s : S) (g : ℕ) :
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
