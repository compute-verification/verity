import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements

/-!
# `RunsZC`'s plays half: `repProg`'s run is a run of the compiled `--zk` rep (statements)

`RunsZC` (`Coin.lean`) asks each rep for a `RepRun`: a run of the compiled rep (`ZkLink.repZCAt`) on its rounds' coins
whose messages, cut at the spine's coins, are the decoder's groups. `ZkCut` gives the cut for the rep's messages
`repMsgsZ`; here is the run. As the non-ZK session's `rep_refines` composes `zerocheck_refines`, `lincheck_refines`,
`opening_refines` and `ligerito_refines`, the run is composed phase by phase from flock-verify's: each phase that
succeeds from `t` to `t'` gives a run of the model's phase on the coins of the rounds it consumed
(`consumed t.rounds t.k t'.k`), played with the proof's messages.

The compiled rep's only branch on values is the opening's verdict (`openingZK` returns `none` when a switched claim
fails its check); its other branches read the schedule. So the masked zerocheck, lincheck, Ligerito and inner proof
play at any pads, link layout and batched value; the opening plays `some` by the non-ZK session's `opening_refines`, on
claims whose points the replay's coins give and whose values are the claim checks.

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

/-- **The masked replay refines the model's masked zerocheck and lincheck** (live PROTOCOL.md §1, §4). When
`Zk.replay` succeeds from `t` to `t'` on well-formed rounds, at a statement whose `m` (at least 15) and `k_log` are the
executable's, then at any pads and triple the model's masked zerocheck and lincheck, played with the proof's messages,
run on the coins of the rounds consumed, split at some round `k₁`; and the replay's two claims have the model's points:
`ab`'s is the lincheck's claim's, `c`'s is `(L_s(z), r)` at the zerocheck's coins. -/
def ZkReplayPlays : Prop :=
  ∀ (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (t t' : Flock.Transcript) (rp : Zk.Replayed)
    (S : Model.Statement), st.m = S.m → st.kLog = S.kLog → 15 ≤ S.m → RoundsWF t.rounds →
    Zk.replay st lay p root t = .ok (rp, t') →
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∀ {np : ℕ} (P : ZkSession.PadsA np) (tr : ZkSession.Triple np), ∃ (k₁ : ℕ) (zc : ZkSession.ZcZK 𝔽 np S.m)
        (row : ZkSession.LinF 𝔽 np) (ab : ZkSession.ClaimZK 𝔽 np S.m) (row' : ZkSession.LinF 𝔽 np),
        t.k ≤ k₁ ∧ k₁ ≤ t'.k ∧
        Plays fOf (ZkSession.zerocheckZK execArith S.m P tr) (zcMsgs S.m p.zerocheck) (consumed t.rounds t.k k₁)
          (zc, row) ∧
        Plays fOf (ZkSession.lincheckZK execArith S P zc) (lcMsgs S.kLog p.lincheck) (consumed t.rounds k₁ t'.k)
          (ab, row') ∧
        (∀ v : F128, ClaimRel S.m { rp.claims[0]! with value := v } ⟨fOf v, ab.skip, ab.x⟩) ∧
        ∀ v : F128, ClaimRel S.m { rp.claims[1]! with value := v } ⟨fOf v, execArith.lagS zc.z, zc.r⟩

/-- **The hiding Ligerito refines the model's masked level 0** (live PROTOCOL.md §7). When `Zk.ligerito` succeeds from
`t` to `t'` on well-formed rounds, at a schedule whose dimensions and levels are the model's (`DimSeam`, `LevelSeam`),
whose levels' consistency coins are not one word, and with at least one lane round, then at any pads of two extra
lanes, padding constant, rep, batched value and basis, the model's masked level 0 with Ligerito's levels compiled,
played with the proof's messages (`ligMsgsZ`, `ybar` of `t_pad` elements), runs on the coins of the rounds consumed to
a result whose Ligerito part is not a reject. -/
def ZkLigeritoPlays : Prop :=
  ∀ (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray) (p : LigeritoProof)
    (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (t t' : Flock.Transcript) (rec : Zk.L0Rec),
    DimSeam sch → (∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default)) →
    (∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) → 1 ≤ sch.k0 →
    RoundsWF t.rounds → Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t') →
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∀ {np : ℕ} (P : ZkSession.Pads0 np 2) (c₀ : 𝔽) (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽),
        ∃ out : ZkSession.LinK 𝕂 np × PendL 𝔽 Dig,
          Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P (schedOf sch) zl.tPad c₀ r Tm b)
            (ligMsgsZ p capDig sch zl.tPad zl.msgs) (consumed t.rounds t.k t'.k) out ∧ out.2 ≠ PendL.reject

/-- **The inner proof refines the model's compiled inner proof** (live PROTOCOL.md §5). When `Zk.verifyInner` succeeds
from `t` to `t'` on seven rows and well-formed rounds, then at any hashes, pads code, rows, triple and pads root, the
model's compiled inner proof on seven rows and `Q` queries, played with the proof's `[ρ, σ]`, `τ`'s commitment and `Y`,
runs on the coins of the rounds consumed (`final_c'`, which `verifyInner` absorbs first, is the compiled rep's message
before it). -/
def ZkInnerPlays : Prop :=
  ∀ (lay : Zk.PadLayout) (pr : Zk.InnerProof) (finalC : F128) (cons : Array (String × Zk.Lin))
    (t t' : Flock.Transcript), cons.size = 7 → RoundsWF t.rounds →
    Zk.verifyInner lay pr finalC cons t = .ok ((), t') →
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∀ {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) {N : ℕ} (dom : Fin N ↪ 𝔽) (np nP : ℕ) (pos : 𝔽 → Fin N)
        (rows : Fin 7 → ZkSession.Lin 𝔽 np) (tr : ZkSession.Triple np) (root : Dig),
        ∃ v, Plays fOf (ZkLink.innerGameC Hs.H Hs.Ep Hs.Lp Hs.Lτ dom np nP Zk.Q pos rows tr root)
          [⟨𝔽 × 𝔽, (fOf pr.rho, fOf pr.sigma)⟩, ⟨Dig, toDig pr.tauCommit⟩, ⟨Fin (np + nP) → 𝔽, vecOf (np + nP) pr.y⟩]
          (consumed t.rounds t.k t'.k) v

/-- **A compiled `--zk` table at the executable's dimensions**: its statement's `m` and `k_log`, its schedule
(`schedOf`), two extra lanes, `t_pad` padding columns, the pads layout's `len` pads, `K_PAD` padding symbols and `Q`
inner queries. -/
structure ZkTabSeam {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) :
    Prop where
  hm : T.S.m = st.m
  hk : T.S.kLog = st.kLog
  hsch : T.sch = schedOf sch
  he : T.e = 2
  ht : T.t = tPad
  hnp : T.np = (layZ st sch).len
  hnP : T.nP = Zk.K_PAD
  hq : T.q = Zk.Q

/-- **`repProg`'s run is a run of the compiled `--zk` rep** (the plays half of `--zk`'s `rep_refines`). At the `--zk`
schedule and a compiled table at the executable's dimensions (`ZkTabSeam`), whose extra claims at the link points are
the executable's (`ClaimRel`, claim by claim), a run of rep `r`'s transcript program from the domain `dom` on
well-formed rounds that consumes every round gives a run of the compiled rep `repZCAt`, played with the rep's messages
(`repMsgsZ`) and the rounds' coins, to a result whose Ligerito part is not a reject. -/
def ZkRepPlays : Prop :=
  ∀ {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽)
    (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript),
    Zk.schedule st.m = .ok (sch, tPad) → ZkTabSeam T st sch tPad →
    extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length →
    (∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j]) →
    RoundsWF rounds → (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size →
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧ pend.lig ≠ PendL.reject

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
