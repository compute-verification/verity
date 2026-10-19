import Proofs.Flock.Claims.Complete.Zk.Run
import Proofs.Flock.Claims.Complete.Sim
import Flock.With.SimZk

/-!
# The honest masked run as a pure session

The plain prover's simulation (`simS`, `Complete/Sim.lean`) on the masked prover: every masked prover function is
related to its pure run (`zerocheckPZk_sim`, …, `sessionPZk_sim`), and the honest masked run's sends are the pure
masked session's bodies (`sessionSZk`, `honestRunZk_eq`).
-/

namespace FlockComplete

open Flock FlockDeploy With

theorem zerocheckPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (m : ℕ) (digest cap root : ByteArray)
    (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) :
    ((simS H resp).st Transcript).R (zerocheckPZk asks m digest cap root lay h tr)
      (zerocheckPZk (asksS resp) m digest cap root lay h tr) := by
  unfold zerocheckPZk sampleVecP
  sim_walk [squeezeP_sim H resp, sampleP_sim H resp]

theorem lincheckPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (c : CircuitFold) (m : ℕ)
    (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (zc : ZcOut) :
    ((simS H resp).st Transcript).R (lincheckPZk asks c m lay h tr zc) (lincheckPZk (asksS resp) c m lay h tr zc) := by
  unfold lincheckPZk
  sim_walk [sampleP_sim H resp]

theorem commitRowsP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (ms : MerkleScheme) (lv : Level)
    (rows : Array (Array F128)) (salt : ℕ → ByteArray) :
    (simS H resp).R (commitRowsP Hashes.pi ms lv rows salt)
      (commitRowsP (Hashes.ofSha512 (Claims.hOf H)) ms lv rows salt) := by
  have ho := opsWith_sim (simS H resp) (simS_hashes H resp) ms
  unfold commitRowsP
  sim_walk [ho.leaf, ho.node]

theorem commitPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (ms : MerkleScheme) (lv : Level)
    (tPad : ℕ) (g : Array F128) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ByteArray) :
    (simS H resp).R (commitPZk Hashes.pi ms lv tPad g extra lanePad salt)
      (commitPZk (Hashes.ofSha512 (Claims.hOf H)) ms lv tPad g extra lanePad salt) :=
  commitRowsP_sim H resp ms lv _ salt

theorem ligeritoPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (t b : Array F128) (lay : Zk.PadLayout) (h : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) :
    ((simS H resp).st Transcript).R
      (ligeritoPZk asks Hashes.pi ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) ms sch T cm0 t b lay h rep tPad extra lanePad salt
        nonce) := by
  unfold ligeritoPZk sampleVecP
  sim_walk [liftT_sim H resp, squeezeP_sim H resp, sampleP_sim H resp, sampleKP_sim H resp, sendMsg_sim H resp,
    powP_sim H resp, challengeP_sim H resp, oodIntroP_sim H resp, queryP_sim H resp, commitP_sim H resp]

theorem innerCommitP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (p : PadsZk) (salt : ℕ → ByteArray) :
    (simS H resp).R (innerCommitP Hashes.pi p salt) (innerCommitP (Hashes.ofSha512 (Claims.hOf H)) p salt) := by
  have hh := simS_hashes H resp
  unfold innerCommitP MerkleOps.hm96With
  sim_walk [leafWith_sim (simS H resp) hh]

theorem innerP_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (lay : Zk.PadLayout) (p : PadsZk)
    (ic : InnerCommit) (finalC : F128) (cons : Array (String × Zk.Lin)) :
    ((simS H resp).st Transcript).R (innerP asks Hashes.pi lay p ic finalC cons)
      (innerP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) lay p ic finalC cons) := by
  have hh := simS_hashes H resp
  unfold innerP sampleVecP
  sim_walk [sampleP_sim H resp, squeezeP_sim H resp, zk_tauCommitmentWith_sim (simS H resp) hh]

theorem repPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (st : Setup) (sch : Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (tr : Trace) (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk)
    (ic : InnerCommit) (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) :
    ((simS H resp).st Transcript).R
      (repPZk asks Hashes.pi st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior words)
      (repPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) st sch tPad rep dom tr t cm0 extras pads ic extra lanePad
        salt nonce prior words) := by
  unfold repPZk rankCheckP
  sim_walk [zerocheckPZk_sim H resp, lincheckPZk_sim H resp, ringSwitchP_sim H resp, ligeritoPZk_sim H resp,
    innerP_sim H resp]

theorem sessionPZk_sim (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec)
    (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace) (tape : TapeZk) :
    (simS H resp).R (sessionPZk asks Hashes.pi spec tabs words trs tape)
      (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape) := by
  unfold sessionPZk
  sim_walk [simS_link H resp, commitPZk_sim H resp, innerCommitP_sim H resp, repPZk_sim H resp,
    recordJson_sim H resp]

/-- **The pure honest masked session** at the replies `resp`: its verdict (the record and the proof files) and its
sends' bodies. -/
def sessionSZk (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg) (spec : SessionSpec) (sts : Array Flock.Stmt)
    (w : Claims.Flock.Wit) (ω : TapeZk) : Except String (String × Array ByteArray) × Array Claims.Msg :=
  (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w)
    ω).go #[]

/-- **The honest masked run is the pure masked session's**: its sends are the session's bodies, each of tag `0`, and its
last message the session's record and proof files after the statement. -/
theorem honestRunZk_eq (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : TapeZk) {k : ℕ}
    (r : Fin k → Coin) {s : Ins} {spec : SessionSpec} {sts : Array Flock.Stmt}
    (hs : setupOf true x = some (s, spec, sts)) {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : sessionSZk H (fun j => (Claims.coinSeq r j).bytes) spec sts w ω = (.ok (rec, pfs), bs)) :
    honestRunZk H x w ω r = (bs.toList.map (List.cons 0), Final.encode ⟨x, rec, pfs⟩) := by
  obtain ⟨bs', h1, h2⟩ := sessionPZk_sim H (fun j => (Claims.coinSeq r j).bytes) spec (tabsOf sts)
    (MaskRank.masksOf sts) (tracesOf sts w) ω #[]
  unfold sessionSZk at hS
  rw [hS] at h1 h2
  dsimp only at h1 h2
  unfold honestRunZk progZk
  rw [hs]
  dsimp only
  change (PI.bind (sessionPZk asks Hashes.pi spec (tabsOf sts) (MaskRank.masksOf sts) (tracesOf sts w) ω).run _).sim
    _ _ 0 = _
  rw [PI.sim_bind]
  have h1' := h1
  simp only [Array.size_empty] at h1'
  rw [h1']
  rw [h2, Array.empty_append, List.toList_toArray]
  show (_ ++ [], _) = _
  rw [List.append_nil]
  rfl

end FlockComplete
