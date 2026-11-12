module

public import Proofs.Pouw.Lemmas.PearlCRecords
public import Proofs.Pouw.PearlC.FlockTile.Audit
public import Proofs.Pouw.PearlC.HiddenTileGame

/-!
# Pearl-C's gap under Flock's hidden audit, conditional on T3 and the binding

* `tileProofSoundAll_flock`: at Flock's hidden audit (`TileReadSession.audit`), the hidden audit's open obligation
  `TileProofSoundAll` follows from T3 at the TileRead session (`TileReadSound`) and the binding (`TileReadBinds`),
  through one `TileGame` per tile (`TileReadSession.tileGame`) and `tileProofSoundAll_of_tileGame`.
* `pearlCHiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192`: so `pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192`'s
  `hzk` is discharged there, and v1 under rev1 per tile at 1/1,000, 8192³ keeps 0.36949% under Flock's hidden audit
  at W1, given `TTOutTilePearlCDevRev1W1`, T3 and the binding.
* `Pouw.SecurityProofs.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192`: the guarantee
  `Pouw.Guarantees.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192` (`Proofs.Pouw.Lemmas.PearlCRecords`),
  which is the last.

Both open obligations are about TileRead, whose Definition and Flock soundness are not in Lean yet; the session is a
parameter (`TileReadSession`). A g under this audit is cited only beside the theorem that the honest prover, at its stated
`q`, is in the audit's class (`HiddenAudit.ProverBounded`, `HiddenAudit`'s docstring): for `TileReadSession.audit`, that
every strategy the honest batch prover induces on a tile's proof game meets the session's `Cls` at every tile whose
proof game it is. It needs the honest TileRead prover, which is not in Lean yet either.
-/

@[expose] public section

namespace Pouw.PearlC.FlockTile

open Pouw.PearlC.Assumptions

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **The hidden audit's open obligation at Flock's hidden audit**, from T3 at the TileRead session and the binding:
for every session `F`, `TileProofSoundAll` holds at `F.audit` and `δzk`, given `0 ≤ δzk`. -/
theorem tileProofSoundAll_flock (P : Protocol Q R S) (TR : TileRules Q R S) (D : Layout → Prop)
    (F : TileReadSession Q R S) (δzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ δzk q) (hT3 : TileReadSound P D F δzk)
    (hbind : TileReadBinds P TR D F) : TileProofSoundAll P TR D F.audit δzk :=
  tileProofSoundAll_of_tileGame P TR D F.audit F.cls δzk h0 (fun _ _ _ _ _ h => h)
    (fun U hU τ o H s g => F.tileGame P TR D δzk h0 hT3 hbind U hU τ o H s g)

/-- **v1 under rev1 per audit tile at the cap 1/1,000, 8192³, under Flock's hidden audit, conditional on T3 and the
binding**, FP32 at 8.376, the honest cast at 8.72: `g = 1648180439/446072750000` (0.36949%), at `F.audit` for the TileRead
session `F`, at `t·δzk(q)` more error than the sampled audit, at the cost model `W1.w1Priced` on
`W1Device.sm120Loop sha fetch`. `pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192` with its `hzk` discharged by
`tileProofSoundAll_flock`.

Its hypotheses: `TTOutTilePearlCDevRev1W1` (the device's timing, as in the sampled audit); `TileReadSound` (T3 at the
session, to be discharged by Flock's soundness of the TileRead statement, whose CR hypotheses are `F.Cls`); and
`TileReadBinds` (the binding, to be proved from TileRead's gates, L4, hm96-sha512's binding, L3 and L8). Cite it only as
conditional on the last two, and beside the honest prover's `ProverBounded` at `F.audit` (the module docstring). It says
something about Flock only at the session whose `game` is TileRead's live session and whose `Cls` is T3's CR
hypotheses: at a session with an empty `Cls`, `F.audit`'s class is empty too and both obligations hold with `Bad`
everything, as at `pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192`'s degenerate hidden audit. -/
theorem pearlCHiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192 (sha fetch : ℚ) (salt : S → List ℕ)
    (sem : PearlCSem Q R S) (F : TileReadSession Q R S) (δzk : ℕ → ℝ) (h0 : ∀ q, 0 ≤ δzk q)
    (hTT : TTOutTilePearlCDevRev1W1 (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt (devSm120v1 Prices.sm120Loop)
      sem (1 / 1000))
    (hT3 : TileReadSound (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F δzk)
    (hbind : TileReadBinds (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F) :
    GGapHidden (Pouw.PearlC.W1.w1Priced (Pouw.PearlC.W1.W1Device.sm120Loop sha fetch) salt)
      (pearlCProtocolDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCTilesDevRev1K (devSm120v1 Prices.sm120Loop) sem (1 / 1000) (218 / 25 - 8953 / 1000))
      (pearlCDomainDevAt (devSm120v1 Prices.sm120Loop) sh8192) F.audit
      ((1648180439 / 446072750000 : ℚ) : ℝ) δPearlC δzk :=
  pearlCHiddenSm120v1LoopCast8p72Rev1Cap1000_8192 sha fetch salt sem F.audit δzk hTT
    (tileProofSoundAll_flock _ _ _ F δzk h0 hT3 hbind)

end Pouw.PearlC.FlockTile

namespace Pouw.SecurityProofs.PearlC

theorem HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192 :
    Pouw.Guarantees.PearlC.HiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192 :=
  fun sha fetch salt sem F δzk h0 hTT hT3 hbind =>
    Pouw.PearlC.FlockTile.pearlCHiddenFlockSm120v1LoopCast8p72Rev1Cap1000_8192 sha fetch salt sem F δzk h0 hTT hT3
      hbind

end Pouw.SecurityProofs.PearlC
