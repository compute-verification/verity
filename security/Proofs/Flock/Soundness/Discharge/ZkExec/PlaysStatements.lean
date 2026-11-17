module

public import Proofs.Flock.Soundness.Discharge.ZkExec.RunsStatements
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

/-! The statements of `RunsZC`'s plays half: `repProg`'s run is a run of the compiled `--zk` rep. -/

@[expose] public section

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

def ZkReplayPlays : Prop :=
  ∀ (st : Setup) (lay : Zk.PadLayout) (p : Proof) (root : ByteArray) (t t' : Flock.Transcript) (rp : Zk.Replayed)
    (S : Model.Statement), st.m = S.m → st.kLog = S.kLog → 15 ≤ S.m → RoundsWF t.rounds →
    Zk.replay st lay p root t = .ok (rp, t') →
    t'.rounds = t.rounds ∧ t'.k = t.k + (S.m + (S.kLog - 6)) ∧
      ∀ {np : ℕ} (P : ZkSession.PadsA np) (tr : ZkSession.Triple np), ∃ (k₁ : ℕ) (zc : ZkSession.ZcZK 𝔽 np S.m)
        (row : ZkSession.LinF 𝔽 np) (ab : ZkSession.ClaimZK 𝔽 np S.m) (row' : ZkSession.LinF 𝔽 np),
        t.k ≤ k₁ ∧ k₁ ≤ t'.k ∧
        Plays fOf (ZkSession.zerocheckZK execArith S.m P tr) (zcMsgs S.m p.zerocheck) (consumed t.rounds t.k k₁)
          (zc, row) ∧
        Plays fOf (ZkSession.lincheckZK execArith S P zc) (lcMsgs S.kLog p.lincheck) (consumed t.rounds k₁ t'.k)
          (ab, row') ∧
        (∀ v : F128, ClaimRel S.m { rp.claims[0]! with value := v } ⟨fOf v, ab.skip, ab.x⟩) ∧
        ∀ v : F128, ClaimRel S.m { rp.claims[1]! with value := v } ⟨fOf v, execArith.lagS zc.z, zc.r⟩

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

def ZkRepPlays : Prop :=
  ∀ {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽)
    (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) → ZkTabSeam T st sch tPad →
    extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length →
    (∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j]) →
    RoundsWF rounds → (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size →
    ∃ pend : ZkLink.PendZ 𝔽 Dig Sl,
      Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧ pend.lig ≠ PendL.reject

def ZkRepRun : Prop :=
  ∀ {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽)
    (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray)
    (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript),
    Zk.schedule st.wideLeaf st.m = .ok (sch, tPad) → st.merkle.digestLen = 64 → ZkTabSeam T st sch tPad →
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
