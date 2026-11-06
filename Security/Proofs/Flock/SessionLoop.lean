module

public import Proofs.Flock.SessionLoop.Statement
public import Proofs.Flock.SessionLoop.TakenVals
public import Proofs.Flock.SessionLoop.FinalC
public import Proofs.Flock.SessionLoop.TakesTau
public import Proofs.Flock.SessionLoop.Inner
public import Proofs.Flock.SessionLoop.LiveInner
public import Proofs.Flock.SessionLoop.Link
public import Proofs.Flock.SessionLoop.FinishParts
public import Proofs.Flock.SessionLoop.FinCommits
public import Proofs.Flock.SessionLoop.FinPoints
public import Proofs.Flock.SessionLoop.FinRounds
public import Proofs.Flock.SessionLoop.TablesSide
public import Proofs.Flock.SessionLoop.FinishIsLoop
public import Proofs.Flock.SessionLoop.Phase
public import Proofs.Flock.SessionLoop.PiopState
public import Proofs.Flock.SessionLoop.CutPiopLive
public import Proofs.Flock.SessionLoop.CutInnerLive
public import Proofs.Flock.SessionLoop.InnerSlot
public import Proofs.Flock.SessionLoop.InnerUnrefused
public import Proofs.Flock.SessionLoop.LiveInnerAt
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.ZK.FrameUnrefused
import Proofs.Flock.Soundness.CROnly.Hiding

/-!
# Proof of C-Flock's zero knowledge at the session loop's own frames

`Flock.SecurityProofs.SessionLoopZK : Flock.Guarantees.SessionLoopZK` is `Claims.ZKServer.Flock.zk_server_lean_ofLoop`
at the deployed relation and at the hiding gap of hm96-sha512's pinned key (`CROnly.hm96Hiding_gap`, finite as a real),
at the loop's own tables (`Loop.realOf`, whose tables are `Count.RealTabs`: `Loop.realTabs_realOf`), its `FinalRoundEq`
from `FinishIs`, the loop's frames refusing every coin fault (`LeanRelease.frameUnrefused_ofLoop`) and the tables
admitting unrefused coins (`LeanRelease.finalRoundEq_ofLoop`). `SessionLoopZKDeployed` is it at the deployed tables,
which admit unrefused coins (`LeanRelease.tablesAdmitUnrefused_dataE`), at the loop's own link and `Finish`
(`LeanRelease.ofLoopLink`),
whose `LinkRoundEq` holds in the masked scope where the loop draws the tape's level 0
(`LeanRelease.linkRoundEq_ofLoopLink`, from `Loop.Src`'s level-0 half, `Loop.Level0Src.linkLevel0`), and whose `FinishIs`
holds there at `LoopInnerRound0`, `Loop.Src`, `LiveRoundEq` and `SwCausal` (`Loop.finishIs`).

`SessionLoopZKDeployed` is over every worker. Its premises on the worker (`LiveRoundEq`, `FillRoundEq`) read it in the
algebraic phase only (`Proofs.Flock.SessionLoop.Phase`): a worker that, until the algebraic phase's last coins, reads
only the nonce, what the workers see of the statement's tables (`RefSession.seenOf`, what `vs` reads) and the answers as
coins (`Loop.PhaseBlind`) is `Loop.phased` at an algebraic phase's worker, whatever it reads after them
(`Loop.PhaseBlind.exists_phased`), and the premises at it are the premises at that worker as a pad-blind one
(`Loop.liveRoundEq_phased`, `Loop.fillRoundEq_phased`).
-/

@[expose] public section

namespace Claims.ZKServer.Loop

/-- **The loop's own tables are `RealTabs`**: each is `tabOfLevel0` at a level 0 `Level0.ofDraws` drew. -/
theorem realTabs_realOf {R : Type}
    (src : Msg → Flock.Wit → R → ℕ → _root_.Flock.V (Array _root_.Flock.F128 × ByteArray × Array ByteArray))
    (x : Msg) (w : Flock.Wit) (ρ : R) {s0 : _} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hs : FlockDeploy.setupOf true x = some (s0, spec, sts)) : Count.RealTabs sts (realOf src x w ρ) := by
  intro k tr hr h
  unfold realOf at h
  rw [hs] at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i a _
    obtain ⟨g, d, hi⟩ := a
    dsimp only at h
    split at h
    · cases h
    · rename_i l hl
      split at h
      · cases h
      · rename_i t ht
        cases h
        exact ⟨g, d, l, hl, ht⟩

/-- **Every worker that reads only what the workers see is a pad-blind one** (`Loop.blind`): one that returns the same
at two tapes alike there (`Masked.seenOf`) and two histories whose answers carry the same coins (`CoinLayer.rd`). -/
theorem exists_blind (L : LeanRelease Flock.Wit
    ((ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Nonce ×
      (ZKMal.CoinLayer.session loopNr ZKMal.Masked.dataE).refP.Rand))
    {wk : L.toSrv.Worker} (hb : L.toSrv.Blind ZKMal.Masked.seenOf ZKMal.CoinLayer.rd wk) : ∃ bw, blind L bw = wk :=
  hb.exists_lift ZKMal.Masked.seenOf_surjective

end Claims.ZKServer.Loop

namespace Flock.SecurityProofs

open FlockSoundness.Discharge

/-- **C-Flock's zero knowledge at the session loop's own frames**, at every statement the loop serves, at tables that
admit unrefused coins. -/
theorem SessionLoopZK : Flock.Guarantees.SessionLoopZK := by
  intro T hA vs t ht hS hV hD hc hsc hT L₀ src real hreal wk x w hR hServe hL0 hLk hFin hE hEf Θ Vs θ hk Ev
  have h := Claims.ZKServer.Flock.zk_server_lean_ofLoop _root_.Flock.Readable.relation T vs t ht
    (FlockSoundness.CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) ENNReal.ofReal_ne_top hS hV hD hc hsc hT L₀ real wk
    x w hR hServe (fun ρ _ _ _ hs => hreal ▸ Claims.ZKServer.Loop.realTabs_realOf src x w ρ hs) hL0 hLk
    (Claims.ZKServer.LeanRelease.finalRoundEq_ofLoop T vs L₀ real _ wk x w hA
      (Claims.ZKServer.LeanRelease.frameUnrefused_ofLoop T L₀ real _ wk x w hServe) hFin) hE hEf Vs θ hk Ev
  rwa [ENNReal.toReal_ofReal (FlockSoundness.CROnly.hidingGap_nonneg ZkView.My ZkView.saltHash)] at h

/-- **C-Flock's zero knowledge at the session loop's own frames and link, at the deployed tables**: `SessionLoopZK` at
`Masked.dataE`, whose tables admit unrefused coins and hold the tables' side but `SwCausal` (`Loop.tables_dataE`,
`complete_dataE`, `dummyReg_dataE`, `causal_dataE`, `tapeOK_dataE`), and the release `withFiles (withLink L₀ src) src`,
whose `LinkRoundEq` is `linkRoundEq_ofLoopLink` and whose `FinishIs` is `Loop.finishIs`. -/
theorem SessionLoopZKDeployed : Flock.Guarantees.SessionLoopZKDeployed := by
  intro vs t ht hsc L₀ src real hreal wk x w hR hServe hL0 hScope hSrc hE hEf Θ Vs θ hk Ev
  subst hreal
  exact SessionLoopZK _ Claims.ZKServer.LeanRelease.tablesAdmitUnrefused_dataE vs t ht Claims.ZKServer.Loop.tables_dataE
    Claims.ZKServer.Loop.complete_dataE Claims.ZKServer.Loop.dummyReg_dataE Claims.ZKServer.Loop.causal_dataE hsc
    Claims.ZKServer.Loop.tapeOK_dataE
    (Claims.ZKServer.LeanRelease.withFiles (Claims.ZKServer.LeanRelease.withLink L₀ src) src) src _ rfl wk x w hR hServe
    hL0 (Claims.ZKServer.LeanRelease.linkRoundEq_ofLoopLink vs L₀ src _ hScope hR hSrc.1.linkLevel0)
    (Claims.ZKServer.Loop.finishIs vs L₀ src wk x w hScope hR hServe hL0 hSrc hE hsc) hE hEf Vs θ hk Ev

end Flock.SecurityProofs
