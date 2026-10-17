import Proofs.Flock.Claims.Complete.ZkTables.Files

/-!
# The reps' checks at several tables

`Zk/Replay.lean`, `Zk/Lanes.lean` and `Zk/Consumed.lean` at a run of several tables, so that of the rep's checks
(`OpenAt`) only those `complete_zk` takes at one table stay named: per rep the ring switch, the masked Ligerito and
the inner proof (`RepChecksAt`).

* `Run.walkJ`: `RunZk.walk` at several tables. The record decodes to the streams the session returned, every recorded
  round retained, table `i`'s rep `r`'s stream the `2 i + r`th, and that rep's decoded masked proof the proof it
  returned, ending at that stream (`RepRunJ`).
* `replayJ`: the masked replay passes, since the prover ran it on rounds the stream carries (`replay_goal`).
* `lanesJ`: level 0's lane phase from the masked Ligerito returning (`laneConstraints_goal`).
* `consumedJ`: every recorded round consumed where the rep's steps return (`consumedZk_goal`): each rep's inner proof
  ends at its table's `repRounds` within its stream (`Run.inner_kJ`), and the streams together fit the session's
  rounds, `2 · repRounds` per table (`sends_goal`, `rounds_eq`), so each stream ends there (`eq_of_sum_le`).
* `openJ_of`, `completeJO`: `complete` at several tables from the ring switch, the masked Ligerito and the inner proof
  there (`RepChecksJ`), with `RepShapedJ` and `RecordFitsJ` for the proof files.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay
open Lean (Json JsonNumber)

/-- **Rep `rep` of table `i`'s run**: the rep, on the table's trace `tr`, its packed table, the masked level-0
commitment `cm`, the extra claims `e`, and the tape's pads at table `i` committed as `ic`, returns a proof with `q`,
from the empty transcript at its domain, ending at the rounds `rs` (`RepRunZk` at table `i`). -/
def RepRunJ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup)
    (v : Schedule × ℕ) (tr : Trace) (cm : Commit) (e : Array Claim) (tape : TapeZk) (i rep : ℕ) (rs : Array Round)
    (q : Zk.ZkProof → Prop) : Prop :=
  ∃ p ic l t' l', innerCommitP (Hashes.ofSha512 h : Hashes V)
      (padsOf tape i rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) (tape.innerSalt i rep) = .ok ic ∧ q p ∧
    rs = t'.rounds ∧
    ((repPZk (asksS resp) (Hashes.ofSha512 h) st v.1 v.2 rep (spec.domain spec.tables[i]! rep).toUTF8 tr
      (packed tr st.m) cm e (padsOf tape i rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) ic (tape.extra i)
      (tape.lanePad i) (tape.salt i rep) (tape.nonce i rep)).run
        (Transcript.init #[] (spec.domain spec.tables[i]! rep).toUTF8)).go l = (.ok (p, t'), l')

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest masked run of several tables, rep by rep**: the record decodes to the streams the session returned,
`2` per table, every round retained; the sends are the link of the tables' caps, then the streams' rounds; and at
each table, with `--zk` schedule `schs[i]!`, rep `r`'s recorded rounds are the `2 i + r`th stream's as the decoder
reads them back, and its decoded masked proof is the proof it returned, ending at that stream. -/
theorem Run.walkJ (S : JsonSpec) (h64 : Out64 H) (hS : StmtOkJ H rel) (hR : RepShapedJ H rel) (o : Run H rel)
    (hc : 1 < o.sts.size) :
    ∃ (schs : Array (Schedule × ℕ)) (cms : Array Commit) (streams : Array (Array Round)),
      streams.size = 2 * o.sts.size ∧ (∀ rs ∈ streams, ∀ x ∈ rs, ∃ c, x.bytes = some c) ∧
      (∃ bs, o.bodies.toList = linkBody (cms.map fun c => flat c.cap).toList
        (o.tabs.map fun _ => ByteArray.empty).toList :: bs ∧ Recs o.resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      o.sess.points = pointsOf o.spec (Claims.coinSeq o.r 0) ∧
      ∀ i < o.sts.size, o.tabs[i]! = Setup.ofCircuit o.sts[i]! ∧ Zk.schedule o.sts[i]!.m = .ok schs[i]! ∧
        o.sch i = schs[i]!.1 ∧ ∀ r < 2,
          (o.stream i r).rounds = streams[2 * i + r]!.map (roundRec o.spec (Claims.hOf H)) ∧
          RepRunJ o.resp (Claims.hOf H) o.spec (Setup.ofCircuit o.sts[i]!) schs[i]! (traces o.sts o.w)[i]! cms[i]!
            (o.extras i) o.ω i r streams[2 * i + r]! fun p => o.zp i r = p := by
  obtain ⟨hsig, hdraw, -, m, kvs, hown, hm, hkv, hco, hg⟩ := hS o.toCase hc
  obtain ⟨hz, hcz⟩ := S.canon_ok _ hg
  have hH : HelloOk o.spec hz := ⟨m, kvs, hown, hm, hkv, hco, hg, hcz⟩
  obtain ⟨ct, hct⟩ : ∃ ct, o.spec.coinTree = some ct := Option.isSome_iff_exists.1 o.ready_parts.2.2.2.2
  have hn := o.toCase.tables_nodup
  obtain ⟨-, -, hsz, -⟩ := o.toCase.tables
  have hP := proverOkJ (rel := rel) h64 o.toCase
  have htsz : (tabsOf o.sts).size = o.sts.size := tabsOf_size _
  have htab : ∀ i (hi : i < o.sts.size), (tabsOf o.sts)[i]! = Setup.ofCircuit o.sts[i]! := fun i hi => by
    rw [getElem!_pos (tabsOf o.sts) i (by rw [htsz]; exact hi), getElem!_pos o.sts i hi]
    simp [tabsOf]
  have hst : ∀ a ∈ o.spec.repStreams, a.1 < (tabsOf o.sts).size := by
    intro a ha
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 ha
    have hk' := hk
    rw [repStreams_size] at hk'
    rw [repStreams_getElem _ hk, tabsOf_size]
    dsimp only
    omega
  have hrun := o.ran
  unfold sessionS at hrun
  obtain ⟨schs, cms, ss, hsc, hcsz, hrec, hss, hrs, ⟨bs, hbs, hrecs⟩, hpsz, hfiles⟩ :=
    sessionPZk_filesJ H _ hct hown hcz (tabsOf o.sts) _ _
      (fun st hst' => by
        obtain ⟨st', hst'', rfl⟩ := Array.mem_map.1 hst'
        exact (hP st' hst'').2) hst hrun
  dsimp only at hrec hpsz hfiles
  rw [htsz] at hcsz
  simp only [ptsOf_coin] at hrec hfiles
  have hsc' : ∀ i (hi : i < o.sts.size), Zk.schedule (Setup.ofCircuit o.sts[i]!).m = .ok schs[i]! ∧
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit o.sts[i]!).merkle
        (schs[i]!.1.levels.getD 0 default) schs[i]!.2 (packed (traces o.sts o.w)[i]! (Setup.ofCircuit o.sts[i]!).m)
        (o.ω.extra i) (o.ω.lanePad i) (o.ω.salt i 0 0) = .ok cms[i]! := fun i hi => by
    rw [← htab i hi]
    exact hsc i (by rw [htsz]; exact hi)
  have hrb : ∀ rb ∈ cms.map fun c => flat c.cap, 0 < rb.size := by
    intro rb hrb
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hrb
    rw [Array.size_map] at hk
    obtain ⟨hv, hcm⟩ := hsc' k (by omega)
    rw [getElem!_pos o.sts k (by omega), getElem!_pos cms k hk] at hcm
    rw [getElem!_pos o.sts k (by omega)] at hv
    rw [Array.getElem_map]
    exact ((hP _ (Array.getElem_mem (by omega))).1 _ hv _ _ _ _ _ hcm).1
  have hrbs : (cms.map fun c => flat c.cap).size = o.spec.tables.size := by simp [hcsz, hsz]
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  have hdec : Record.decodeWith o.hs o.spec o.record = .ok (sessOf o.spec (Claims.hOf H)
      (cms.map fun c => flat c.cap) (pointsOf o.spec (Claims.coinSeq o.r 0)) ss o.proofs) := by
    rw [hrec]
    exact decode_recordObjZkJ S (hOf_size h64) hn hct hH hdraw hsig hrbs hrb hpts hpm hss hrs o.proofs
  have hsess : o.sess = sessOf o.spec (Claims.hOf H) (cms.map fun c => flat c.cap)
      (pointsOf o.spec (Claims.coinSeq o.r 0)) ss o.proofs := by
    unfold Run.sess
    rw [hdec]
    rfl
  have hzsz : ((o.spec.repStreams.zip ss).map
      (streamRec o.spec (Claims.hOf H) (cms.map fun c => flat c.cap))).size = o.spec.repStreams.size := by
    simp [hss]
  have hnames : (((o.spec.repStreams.zip ss).map
      (streamRec o.spec (Claims.hOf H) (cms.map fun c => flat c.cap))).toList.map StreamRec.name).Nodup := by
    have e : (((o.spec.repStreams.zip ss).map
        (streamRec o.spec (Claims.hOf H) (cms.map fun c => flat c.cap))).toList.map StreamRec.name) =
        o.spec.repStreams.toList.map (·.2.2) := by
      refine List.ext_getElem (by simp [hss]) fun k h₁ h₂ => ?_
      simp [streamRec]
    rw [e]
    exact repNames_nodup hn
  have hstream : ∀ i r, 2 * i + r < o.spec.repStreams.size → r < 2 →
      (o.stream i r).rounds = ss[2 * i + r]!.map (roundRec o.spec (Claims.hOf H)) := by
    intro i r hk hr
    have hf := find?_name StreamRec.name _ hnames (2 * i + r) (by simp [hss, hk])
    simp only [Array.getElem_toList, Array.find?_toList, Array.getElem_map, Array.getElem_zip] at hf
    have hname : s!"{o.spec.tables[i]!}/rep{r}" = (streamRec o.spec (Claims.hOf H) (cms.map fun c => flat c.cap)
        (o.spec.repStreams[2 * i + r], ss[2 * i + r]'(by rw [hss]; exact hk))).name := by
      simp only [streamRec, repStreams_getElem _ hk]
      rw [show (2 * i + r) / 2 = i by omega, show (2 * i + r) % 2 = r by omega]
    unfold Run.stream
    rw [hsess]
    simp only [sessOf]
    rw [hname, hf, getElem!_pos ss _ (by rw [hss]; exact hk)]
    rfl
  have hpsz2 : ss.size = 2 * o.sts.size := by rw [hss, repStreams_size, hsz]
  refine ⟨schs, cms, ss, hpsz2,
    fun rs hrs' x hx => Recs.bytes hrecs x
      (List.mem_flatMap.2 ⟨rs, Array.mem_toList_iff.2 hrs', Array.mem_toList_iff.2 hx⟩),
    ⟨bs, hbs, hrecs⟩, by rw [hsess]; rfl, fun i hi => ?_⟩
  have htabi : o.tabs[i]! = Setup.ofCircuit o.sts[i]! := htab i hi
  obtain ⟨hv, hcm⟩ := hsc' i hi
  have hsch : o.sch i = schs[i]!.1 := by
    unfold Run.sch zkSchedsOf Run.tabs
    rw [getElem!_pos _ i (by simp [htsz]; exact hi), Array.getElem_map,
      ← getElem!_pos (tabsOf o.sts) i (by rw [htsz]; exact hi), htab i hi]
    unfold schedZkOf
    rw [hv]
  refine ⟨htabi, hv, hsch, fun r hr => ⟨hstream i r (by rw [repStreams_size, hsz]; omega) hr, ?_⟩⟩
  have hj : 2 * i + r < o.proofs.size := by rw [hpsz, repStreams_size, hsz]; omega
  obtain ⟨e, p, ic, l, t', l', hex, hic, hb, hrr, hp⟩ := hfiles (2 * i + r) hj
  have hji : (2 * i + r) / 2 = i := by omega
  have hjr : (2 * i + r) % 2 = r := by omega
  rw [hji] at hex hic hb hp
  rw [hjr] at hic hp
  rw [htab i hi] at hex hic hb hp
  have hext : o.extras i = e := by
    unfold Run.extras
    rw [hsess, htabi]
    simp only [sessOf, hex]
    rfl
  have hps := Ret.go (hR o.toCase hc i hi _ _ e hv hcm hex r ic hic) hp
  refine ⟨p, ic, l, t', l', hic, ?_, hrr, by rw [hext]; exact hp⟩
  have hdec' : Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok p := by
    unfold Run.bytes
    rw [getElem!_pos o.proofs _ hj, hb, htabi, hsch, hext]
    exact FlockSoundness.Discharge.ZkView.decode_encode _ _ hps
  unfold Run.zp
  rw [hdec']
  rfl

theorem Run.lt_sts {o : Run H rel} {i : ℕ} (hi : i < o.tabs.size) : i < o.sts.size := by
  simpa [Run.tabs, tabsOf_size] using hi

/-- **The masked replay passes** at every rep of a run of several tables, from its recorded rounds (`replay_goal`). -/
theorem replayJ (S : JsonSpec) (h64 : Out64 H) (hS : StmtOkJ H rel) (hR : RepShapedJ H rel) (o : Run H rel)
    (hc : 1 < o.sts.size) : ∀ i < o.tabs.size, ∀ r < 2,
      (Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r) =
        .ok (o.rp i r) := by
  intro i hi r hr
  obtain ⟨schs, cms, streams, hss, hbytes, -, -, hrep⟩ := Run.walkJ S h64 hS hR o hc
  obtain ⟨htabi, -, hsch, hreps⟩ := hrep i (Run.lt_sts hi)
  obtain ⟨hstream, p, ic, l, t', l', -, hzp, hst', hrun⟩ := hreps r hr
  have hroot := (Ret.go (repPZk_fields _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun).2.2
  obtain ⟨R, rp, t₁, hre, C, hC⟩ := Ret.go (repPZk_replay _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun
  refine FlockComplete.Replay.ok_val ?_
  have hlay : o.lay i = Zk.PadLayout.new (Setup.ofCircuit o.sts[i]!).m (Setup.ofCircuit o.sts[i]!).kLog
      schs[i]!.1.k0 := by
    unfold Run.lay
    rw [htabi, hsch]
  rw [hzp, hroot, hlay, htabi]
  rw [← With.zk_replayWith_std] at hre
  have hcarry : Carry (Transcript.init R (o.spec.domain o.spec.tables[i]! r).toUTF8) (o.t₀ i r) := by
    refine ⟨rfl, rfl, fun j hj => ?_⟩
    change j < R.size at hj
    have hj' : j < t'.rounds.size := by
      rw [hC, Array.size_append]
      exact Nat.lt_of_lt_of_le hj (Nat.le_add_right _ _)
    have hRj : R[j] = t'.rounds[j] := by
      have h1 : t'.rounds[j]? = R[j]? := by rw [hC, Array.getElem?_append_left hj]
      rw [Array.getElem?_eq_getElem hj', Array.getElem?_eq_getElem hj] at h1
      exact (Option.some.inj h1).symm
    have hmem : t'.rounds[j] ∈ streams[2 * i + r]! := by
      rw [hst']
      exact Array.getElem_mem hj'
    have hk : 2 * i + r < streams.size := by
      have := Run.lt_sts hi
      omega
    obtain ⟨c, hcb⟩ := hbytes _ (by rw [getElem!_pos streams _ hk]; exact Array.getElem_mem _) _ hmem
    change R[j].bytes.isSome = true ∧ Carries (o.stream i r).rounds j R[j]
    rw [hRj, hcb]
    refine ⟨rfl, roundRec o.spec (Claims.hOf H) t'.rounds[j], ?_, rfl, rfl, by simp [roundRec, hcb]⟩
    rw [hstream, hst']
    simp [hj']
  obtain ⟨u', hu', -⟩ := replayWith_xfer With.std o.hs _ _ _ _ _ _ _ _ hcarry hre
  exact ⟨_, hu'⟩

/-- **Level 0's lane phase gives its constraints** wherever the masked Ligerito returns on a decoded proof
(`laneConstraints_goal`). -/
theorem lanesJ {o : Run H rel} {i r : ℕ}
    (hLg : (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap
      (o.zp i r).proof.ligerito (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 =
        .ok (o.lg i r))
    (hD : Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.zp i r)) :
    Zk.ligeritoConstraints (o.lay i) (o.lg i r).1 (o.zp i r).proof.ligerito (o.zp i r).level0 = .ok (o.lanes i r) := by
  obtain ⟨ho, hcl⟩ := zkLigeritoWith_rec _ _ _ _ _ _ _ _ (t' := (o.lg i r).2) (rc := (o.lg i r).1) hLg
  have hm := (FlockSoundness.Discharge.ZkExec.zk_decode_sizes _ _ _ hD).1.messages
  obtain ⟨v, hv⟩ := ligeritoConstraints_ok (lay := o.lay i) (lg := (o.zp i r).proof.ligerito)
    (msgs := (o.zp i r).level0) ho hcl (by simp only [Run.lay, Zk.PadLayout.new, shapeZk] at hm ⊢; omega)
  unfold Run.lanes
  rw [hv]
  rfl

end FlockComplete.ZkTables
