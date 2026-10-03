import FlockSoundness.Discharge.ZkExec.VerdictStatements

/-!
# `ZkRepAccepts`'s Ligerito half: a `--zk` rep's Ligerito verdict accepts the proof's rows

`ZkRepAccepts` (`VerdictStatements.lean`) has three conclusions at every result of the compiled rep: the Ligerito
verdict accepts the opened rows, the inner check holds, and every requested row verifies (`OpensOKZ`). This file is the
first: `ZkLigAccepts` takes exactly `ZkRepAccepts`'s hypotheses and concludes `pend.lig.verdict os.rows = true`.

The compiled masked phase's Ligerito (`ZkLink.ligeritoPadZC`, then `ZkLink.ligTailC`) is M0's compiled Ligerito
(`Model.recursiveRoundsC`, `Model.finalCheck`) at levels ≥ 1; at level 0 it batches the padded column
`ZkLink.padColumnO`: the witness lanes folded, `ρ₁·row[2^k₀+2r] + ρ₂·row[2^k₀+2r+1]`, and the pads' correction
`Model.padCorr` at `ȳ`, which is the executable's `Zk.padAt` at `PAD_C`.
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL

local notation "𝔽" => FlockLevel3.GF128

/-- **A `--zk` rep's Ligerito verdict accepts** (live PROTOCOL.md §7, PROTOCOL.md §13). Under `ZkRepAccepts`'s
hypotheses, every result of the compiled rep, played with the rep's messages and the rounds' coins, has a Ligerito
verdict that accepts the openings whose rep-`r` rows are the proof's. -/
def ZkLigAccepts : Prop :=
  ∀ {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript) (cap₀ : Cap Dig)
    (os : ZkLink.OpensZ 𝔽 Dig (List UInt8)),
    Zk.schedule st.m = .ok (sch, tPad) → st.merkle = MerkleScheme.hm96Sha512 → ZkTabSeam T st sch tPad →
    ZkTabLay T st sch → FoldRealizes st.fold T.S →
    extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length →
    (∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j]) →
    Zk.decode (shapeZK st sch extras) bytes = .ok zp → RoundsWF rounds →
    (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr) →
    tr.k = rounds.size → cap₀ = capDig zp.proof.cap →
    (∀ ℓ j, os.rows r.val ℓ j = openRow zp.proof.ligerito sch ℓ j) →
    (∀ ℓ j, os.salts r.val ℓ j = saltRow zp.proof.ligerito sch ℓ j) → os.inner r.val = innerOpensOf zp.inner →
    ∀ pend : ZkLink.PendZ 𝔽 Dig (List UInt8),
      Plays fOf (ZkLink.repZCAt execArith hs512 T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend →
      pend.lig.verdict os.rows = true

theorem zk_lig_accepts : ZkLigAccepts := by
  sorry

end FlockSoundness.Discharge.ZkExec
