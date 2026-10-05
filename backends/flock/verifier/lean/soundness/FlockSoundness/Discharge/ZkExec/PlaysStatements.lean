import FlockSoundness.Discharge.ZkExec.RunsStatements
import FlockSoundness.Protocol.Discharge.ZkExec.PlaysStatements

/-!
# `RunsZC`'s plays half: `repProg`'s run is a run of the compiled `--zk` rep (statements)

`RunsZC` (`Coin.lean`) asks each rep for a `RepRun`: a run of the compiled rep (`ZkLink.repZCAt`) on its rounds' coins
whose messages, cut at the spine's coins, are the decoder's groups. `ZkCut` gives the cut for the rep's messages
`repMsgsZ`; here is the run. As M0's `rep_refines` composes `zerocheck_refines`, `lincheck_refines`, `opening_refines`
and `ligerito_refines`, the run is composed phase by phase from flock-verify's: each phase that succeeds from `t` to
`t'` gives a run of the model's phase on the coins of the rounds it consumed (`consumed t.rounds t.k t'.k`), played with
the proof's messages.

The compiled rep's only branch on values is the opening's verdict (`openingZK` returns `none` when a switched claim
fails its check); its other branches read the schedule. So the masked zerocheck, lincheck, Ligerito and inner proof
play at any pads, link layout and batched value; the opening plays `some` by M0's `opening_refines`, on claims whose
points the replay's coins give and whose values are the claim checks.

* `ZkReplayPlays`: `Zk.replay` gives runs of `zerocheckZK` and `lincheckZK`, and the two masked claims' points.
* `ZkLigeritoPlays`: `Zk.ligerito` gives a run of `ligeritoPadZC` whose Ligerito part is not a reject.
* `ZkInnerPlays`: `Zk.verifyInner` on seven rows gives a run of `innerGameC` (after `final_c'`, which the compiled rep
  receives before it).
* `ZkTabSeam`: a compiled table whose dimensions are the executable's.
* `ZkRepPlays`: `repProg`'s run, consuming every round, is a run of `repZCAt` with the rep's messages and the rounds'
  coins, whose Ligerito part is not a reject.
* `ZkRepRun`: with `ZkRepFrames` and `ZkCut`, that run is a `RepRun` along `repShapeZ` for the decoder `decZ`, with no
  tail.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-- **A `--zk` rep's `RepRun`** (`RunsZC`'s per-rep half). At the `--zk` schedule, 64-byte digests and a compiled table
at the executable's dimensions whose extra claims are the executable's, on a decoded proof whose pads root has 64 bytes
and on well-formed retained rounds, a run of rep `r`'s transcript program that consumes every round gives a result,
not a reject in its Ligerito part, at which the compiled rep has a `RepRun` on the rounds along `repShapeZ` for the
decoder `decZ`, with no tail. -/
def ZkRepRun : Prop :=
  ∀ {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽)
    (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray)
    (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle.digestLen = 64 → ZkTabSeam T st sch tPad →
    extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length →
    (∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j]) →
    Zk.decode (shapeZK st sch extras) bytes = .ok zp → zp.inner.root.size = 64 →
    RoundsWF rounds → AllRetained rounds →
    (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size →
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      PlaysCoin.RepRun (ZkLink.repZCAt execArith Hs T pts r)
        (repShapeZ 𝔽 𝕂 Dig T (Model.extraClaims T.S T.ptLocal mPts pts).length)
        (decZ (wRestZ st.m st.kLog (2 + extras.size) (schedOf sch) tPad (layZ st sch).len)) rounds.toList [] pend ∧
      pend.lig ≠ PendL.reject

end FlockSoundness.Discharge.ZkExec
