import Proofs.Flock.Claims.Complete.ZkTables.Level0
import Proofs.Flock.Claims.Complete.ZkTables.Table
import Proofs.Flock.Claims.Complete.ZkTables.View

/-!
# Masked completeness from its parts

`complete_jsonT` takes four table-level premises (`RepShapedT`, `ProverRunsT`, `LigeritoT`, `InnerT`). Each splits into
level 0's masked Ligerito, stated once over `ligeritoPZk` and `Zk.ligeritoWith` (`Level0`: `ProverRuns`, `ProverShape`,
`Lanes`, `Verifier`), and the rest, which the plain proofs and the closed forms of the masked PIOP cover: what each
T-Prop asks beyond `Level0` (`ShapeRest`, `RunsRest`, `LigeritoRest`, `InnerRest`). `LigeritoRest` and `InnerRest`
hold (`ligeritoRest`, `innerRest`, from `View.view`), so `complete_jsonT_of_parts` composes `Level0`'s four with
`ShapeRest` and `RunsRest`; `RunsRest` holds too (`RunsGo.runsRest`).

The prover's rank refusal (`rankCheckP`) enters `RunsRest` only: a rep returns where `ProverRunsT`'s `MaskRankT`
holds, and `complete_jsonT` takes the mask-rank bound at the verifier's coins (`MaskRank.MaskRankBound`) for the
other side, that a good case's coins pass the check.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT

section Rest

variable (H : Claims.Msg → Claims.Msg)

/-- **`RepShapedT` beyond level 0's shape**: the masked PIOP's and the inner proof's lengths, and the encoding's bound,
given `Level0.ProverShape`. -/
def ShapeRest : Prop := Level0.ProverShape → RepShapedT H

/-- **`ProverRunsT` beyond level 0**: the masked PIOP returns in closed form, the rep's constraints vanish at its pads
(level 0's lane rows from `Level0.Lanes`), so the inner proof returns, and the rep records `repRoundsT` rounds, given
`Level0.ProverRuns`, `Level0.Lanes` and `Level0.Verifier` (whose final position gives the Ligerito's round count). -/
def RunsRest : Prop := Level0.ProverRuns → Level0.Lanes → Level0.Verifier → ProverRunsT H

/-- **`LigeritoT` beyond level 0**: the verifier's replay and ring switch return the prover's values at the prover's
position, so its masked Ligerito runs on the prover's inputs, given `Level0.Verifier`. -/
def LigeritoRest : Prop := Level0.Verifier → LigeritoT H

/-- **`InnerT` beyond level 0**: as `LigeritoRest`, and the verifier's constraints are the ones the prover's inner
proof checked before it returned, given `Level0.Verifier`. -/
def InnerRest : Prop := Level0.Verifier → InnerT H

end Rest

open FlockComplete.Replay in
/-- **`LigeritoT` from level 0's verifier premise** (`view`). -/
theorem ligeritoRest (H : Claims.Msg → Claims.Msg) : LigeritoRest H := by
  intro hV h64 st tr hT ω i d k r v cm pts e rep ic hin hrep prior ws l p t' l' _ _ hgo R hAg q hq sw hsw
  obtain ⟨_, _, _, _, _, h, -⟩ :=
    view hV h64 st tr hT ω i d r v cm pts e rep ic hin hrep prior ws l p t' l' hgo R hAg q hq sw hsw
  exact ⟨_, h⟩

open FlockComplete.Replay in
/-- **`InnerT` from level 0's verifier premise**: the verifier's constraints are those the prover's inner proof
returned on (`view`), so its batched constraint held at its coins (`innerP_ret`), which is what the verifier's inner
check needs (`verifyInnerWith_of'`). -/
theorem innerRest (H : Claims.Msg → Claims.Msg) : InnerRest H := by
  intro hV h64 st tr hT ω i d k r v cm pts e rep ic hin hrep prior ws l p t' l' _ _ hgo R hAg q hq sw hsw lg hlg lanes
    hlanes
  obtain ⟨rec, tI, lI, rp, lgc, hLg, htk, hlgc, hq1, hinn⟩ :=
    view hV h64 st tr hT ω i d r v cm pts e rep ic hin hrep prior ws l p t' l' hgo R hAg q hq sw hsw
  obtain rfl : lg = (rec, ⟨R, tI.k, tI.pending⟩) := Except.ok.inj (hlg.symm.trans hLg)
  obtain rfl : lanes = lgc := Except.ok.inj (hlanes.symm.trans hlgc)
  rw [hq1]
  obtain ⟨hz, hb, hpi, ht'⟩ := innerP_ret _ (Claims.hOf H) _ _ _ _ _ _ _ hinn
  rw [hpi]
  subst ht'
  exact ⟨_, verifyInnerWith_of' (Claims.hOf H) (JsonRT.hOf_size h64) _ ω i rep (ω.innerSalt i rep) hin.innerCommit _ _
    hz hb tI htk (hAg.mono (Nat.zero_le _))⟩

/-- **Masked completeness from its parts**: `complete_jsonT` from level 0's four premises (`Level0`), the rest of
`RepShapedT` and `ProverRunsT`, and the mask-rank bound; `LigeritoT` and `InnerT` follow from `Level0.Verifier`
(`ligeritoRest`, `innerRest`). -/
theorem complete_jsonT_of_parts (H : Claims.Msg → Claims.Msg) (S : Claims.JsonSpec printJ Flock.canon)
    (h0 : Level0.ProverRuns) (h1 : Level0.ProverShape) (h2 : Level0.Lanes) (h3 : Level0.Verifier)
    (gS : ShapeRest H) (gR : RunsRest H) (hMR : MaskRank.MaskRankBound) :
    Claims.Flock.Complete H (deployZk Readable.relation) :=
  complete_jsonT H S (gS h1) (gR h0 h2 h3) (ligeritoRest H h3) (innerRest H h3) hMR

end FlockComplete.ZkTables
