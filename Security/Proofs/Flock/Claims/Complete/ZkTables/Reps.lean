import Proofs.Flock.Claims.Complete.ZkTables.Files
import Proofs.Flock.Claims.Complete.Json.Canon

/-!
# The reps' checks at several tables

`Zk/Replay.lean`, `Zk/Lanes.lean` and `Zk/Consumed.lean` at a run of any number of tables, so that of the rep's checks
(`OpenAt`) only those `complete_zk` takes at one table stay named: per rep the ring switch, the masked Ligerito and
the inner proof (`RepChecksAt`).

* `Run.walkJ`: `RunZk.walk` at every table. The record decodes to the streams the session returned, every recorded
  round retained, table `i`'s rep `r`'s stream the `2 i + r`th, and that rep's decoded masked proof the proof it
  returned, ending at that stream (`RepRunJ`).
* `replayJ`: the masked replay passes, since the prover ran it on rounds the stream carries (`replay_goal`).
* `lanesJ`: level 0's lane phase from the masked Ligerito returning (`laneConstraints_goal`).
* `consumedJ`: every recorded round consumed where the rep's steps return (`consumedZk_goal`): each rep's inner proof
  ends at its table's `repRounds` within its stream (`Run.inner_kJ`), and the streams together fit the session's
  rounds, `2 · repRounds` per table (`sends_goal`, `rounds_eq`), so each stream ends there (`eq_of_sum_le`).
* `openAt_of`, `openJ_of`, `completeJO`: `complete` at several tables from the ring switch, the masked Ligerito and the
  inner proof there (`RepChecksJ`), with `RepShapedJ` for the proof files.
* `completeAll`: since nothing above asks for several tables, the same at every number of tables from the per-table
  premises at every good case (`StmtAt`, `ProverRunsAt`, `RepShapedAt`, `RepChecksAt`), one table's included, through
  `complete_of_at`, in place of `complete_zk`'s.
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT FlockComplete.Replay
open Lean (Json JsonNumber)

/-- **Rep `rep` of table `i`'s run**: the rep, on the table's trace `tr`, its packed table, its masked level-0
commitment `cm` under the tape's level `i`, the extra claims `e`, and the tape's pads at table `i` committed as `ic`,
returns a proof with `q`, from the empty transcript at its domain and `n` sends, ending at the rounds `rs`, at some
earlier claim points and mask words (`RepRunZk` at table `i`). -/
def RepRunJ (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (spec : SessionSpec) (st : Setup)
    (v : Schedule × ℕ) (tr : Trace) (cm : Commit) (e : Array Claim) (tape : TapeZk) (i rep : ℕ) (rs : Array Round)
    (n : ℕ) (q : Zk.ZkProof → Prop) : Prop :=
  ∃ p ic l t' l' prior ws, innerCommitP (Hashes.ofSha512 h : Hashes V)
      (padsOf tape i rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) (tape.innerSalt i rep) = .ok ic ∧ q p ∧
    rs = t'.rounds ∧ l.size = n ∧
    commitPZk (Hashes.ofSha512 h : Hashes V) st.merkle (v.1.levels.getD 0 default) v.2 (packed tr st.m)
      (tape.extra i) (tape.lanePad i) (tape.salt i 0 0) = .ok cm ∧
    ((repPZk (asksS resp) (Hashes.ofSha512 h) st v.1 v.2 rep (spec.domain spec.tables[i]! rep).toUTF8 tr
      (packed tr st.m) cm e (padsOf tape i rep (Zk.PadLayout.new st.m st.kLog v.1.k0)) ic (tape.extra i)
      (tape.lanePad i) (tape.salt i rep) (tape.nonce i rep) prior ws).run
        (Transcript.init #[] (spec.domain spec.tables[i]! rep).toUTF8)).go l = (.ok (p, t'), l')

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest masked run, rep by rep**: the record decodes to the streams the session returned, `2` per table,
every round retained; the sends are the link of the tables' caps, then the streams' rounds; and at each table, with
`--zk` schedule `schs[i]!`, rep `r`'s recorded rounds are the `2 i + r`th stream's as the decoder reads them back, and
its decoded masked proof is the proof it returned, ending at that stream. -/
theorem Run.walkJ (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) (hR : RepShapedAt o.toCase) :
    ∃ (schs : Array (Schedule × ℕ)) (cms : Array Commit) (streams : Array (Array Round)),
      streams.size = 2 * o.sts.size ∧ (∀ rs ∈ streams, ∀ x ∈ rs, ∃ c, x.bytes = some c) ∧
      (∃ bs, o.bodies.toList = linkBody (cms.map fun c => flat c.cap).toList
        (o.tabs.map fun _ => ByteArray.empty).toList :: bs ∧ Recs o.resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      o.sess.points = pointsOf o.spec (Claims.coinSeq o.r 0) ∧
      ∀ i < o.sts.size, o.tabs[i]! = Setup.ofCircuit o.sts[i]! ∧ Zk.schedule o.sts[i]!.m = .ok schs[i]! ∧
        o.sch i = schs[i]!.1 ∧ ∀ r < 2,
          (o.stream i r).rounds = streams[2 * i + r]!.map (roundRec o.spec (Claims.hOf H)) ∧
          RepRunJ o.resp (Claims.hOf H) o.spec (Setup.ofCircuit o.sts[i]!) schs[i]! (traces o.sts o.w)[i]! cms[i]!
            (o.extras i) o.ω i r streams[2 * i + r]!
            (1 + ((streams.toList.take (2 * i + r)).flatMap Array.toList).length) fun p => o.zp i r = p := by
  obtain ⟨hsig, hdraw, -, m, kvs, hown, hm, hkv, hco, hg, ha, hT⟩ := id hS
  obtain ⟨hz, hcz⟩ := FlockComplete.JsonRT.canon_ok _ hg ha
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
    sessionPZk_filesJ H _ hct hown hcz (tabsOf o.sts) _ _ _
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
    exact decode_recordObjZkJ (hOf_size h64) hn hct (hT ct hct) hH hdraw hsig hrbs hrb hpts hpm hss hrs o.proofs
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
  obtain ⟨e, p, ic, l, t', l', prior, hex, hic, hb, hrr, hl, hp⟩ := hfiles (2 * i + r) hj
  have hji : (2 * i + r) / 2 = i := by omega
  have hjr : (2 * i + r) % 2 = r := by omega
  rw [hji] at hex hic hb hp
  rw [hjr] at hic hp
  rw [htab i hi] at hex hic hb hp
  rw [MaskRank.masksOf_getElem! hi] at hp
  have hext : o.extras i = e := by
    unfold Run.extras
    rw [hsess, htabi]
    simp only [sessOf, hex]
    rfl
  have hps := Ret.go (hR i hi _ _ e hv hcm hex r ic hic prior) hp
  refine ⟨p, ic, l, t', l', prior, _, hic, ?_, hrr, hl, hcm, by rw [hext]; exact hp⟩
  have hdec' : Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok p := by
    unfold Run.bytes
    rw [getElem!_pos o.proofs _ hj, hb, htabi, hsch, hext]
    exact FlockSoundness.Discharge.ZkView.decode_encode _ _ hps
  unfold Run.zp
  rw [hdec']
  rfl

theorem Run.lt_sts {o : Run H rel} {i : ℕ} (hi : i < o.tabs.size) : i < o.sts.size := by
  simpa [Run.tabs, tabsOf_size] using hi

/-- **The masked replay passes** at every rep of a run, from its recorded rounds (`replay_goal`). -/
theorem replayJ (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) (hR : RepShapedAt o.toCase) :
    ∀ i < o.tabs.size, ∀ r < 2,
      (Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r) =
        .ok (o.rp i r) := by
  intro i hi r hr
  obtain ⟨schs, cms, streams, hss, hbytes, -, -, hrep⟩ := Run.walkJ h64 o hS hR
  obtain ⟨htabi, -, hsch, hreps⟩ := hrep i (Run.lt_sts hi)
  obtain ⟨hstream, p, ic, l, t', l', prior, ws, -, hzp, hst', -, -, hrun⟩ := hreps r hr
  have hroot := (Ret.go (repPZk_fields _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun).2.2
  obtain ⟨R, rp, t₁, hre, C, hC⟩ := Ret.go (repPZk_replay _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _) hrun
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

section Consumed

open FlockSoundness.Refine FlockSoundness.Discharge.ZkExec

/-- **Where a masked rep's steps return, the inner proof ends at the rep's round count**, within the rep's stream, at
any table (`RunZk.inner_k`). -/
theorem Run.inner_kJ (o : Run H rel) {i r : ℕ} {st : Flock.Stmt} (ht : o.tabs[i]! = Setup.ofCircuit st)
    {v : Schedule × ℕ} (hv : Zk.schedule st.m = .ok v) (hsch : o.sch i = v.1)
    (hRp : (Zk.replayWith o.hs o.tabs[i]! (o.lay i) (o.zp i r).proof (o.zp i r).inner.root).run (o.t₀ i r) =
      .ok (o.rp i r))
    (hRS : (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
      (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r))
    (hLg : (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap
      (o.zp i r).proof.ligerito (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 =
        .ok (o.lg i r))
    (hIn : (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
      (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)) :
    (o.inner i r).2.k = (st.m + ((Setup.ofCircuit st).kLog - 6)) + ((2 + (o.extras i).size) + 1) +
        (v.1.k0 + 7 + 11 * (v.1.levels.size - 1)) + 4 ∧
      (o.inner i r).2.k ≤ (o.stream i r).rounds.size := by
  obtain ⟨h25, -, hsm, hk0, -, hdim, hlv, -, -⟩ := zk_schedule_seams st.m v.1 v.2 hv
  obtain ⟨hn7, h2, -, -, -, -, -⟩ := hdim
  rw [hsch] at hLg
  obtain ⟨hr₁, hk₁⟩ := replayWith_rounds o.hs (t' := (o.rp i r).2) (rp := (o.rp i r).1)
    (by rw [ht]; exact (by omega : 14 ≤ st.m)) hRp
  obtain ⟨hr₂, hk₂⟩ := ringSwitchWith_rounds o.hs (t' := (o.sw i r).2) (out := (o.sw i r).1)
    (by simp [ringClaimsZk]) hRS
  obtain ⟨hr₃, hk₃⟩ := zkLigeritoWith_rounds o.hs _ v.1 _ _ _ _ _ (t' := (o.lg i r).2) (rc := (o.lg i r).1)
    (by omega) hk0 h2 (fun i hi => (hlv i hi).q_ne_zero) hLg
  obtain ⟨hr₄, hk₄⟩ := verifyInnerWith_rounds o.hs (t' := (o.inner i r).2) (u := (o.inner i r).1) hIn
  have hend := verifyInnerWith_end o.hs (t' := (o.inner i r).2) (u := (o.inner i r).1) hIn
  refine ⟨?_, ?_⟩
  · rw [hk₄, hk₃, hk₂, hk₁, ht]
    simp [ringClaimsZk, Run.t₀, Transcript.init, Setup.ofCircuit]
  · rw [hr₄, hr₃, hr₂, hr₁] at hend
    exact hend

/-- **Values bounded below whose sum is at most their bounds' sum** equal their bounds. -/
theorem eq_of_sum_le {α : Type} (a b : α → ℕ) :
    ∀ l : List α, (∀ j ∈ l, b j ≤ a j) → (l.map a).sum ≤ (l.map b).sum → ∀ j ∈ l, a j = b j
  | [], _, _, j, hj => absurd hj List.not_mem_nil
  | x :: l, hle, hs, j, hj => by
    simp only [List.map_cons, List.sum_cons] at hs
    have h1 := hle x List.mem_cons_self
    have h2 : (l.map b).sum ≤ (l.map a).sum := List.sum_le_sum fun j hj => hle j (List.mem_cons_of_mem _ hj)
    rcases List.mem_cons.1 hj with rfl | hj
    · omega
    · exact eq_of_sum_le a b l (fun j hj => hle j (List.mem_cons_of_mem _ hj)) (by omega) j hj

/-- **Every recorded round of a masked rep is consumed** at a run, wherever its steps return (`consumedZk_goal`): each
rep's inner proof ends at its table's `repRounds` within its stream (`Run.inner_kJ`), and the streams together are the
sends after the link, at most `2 · repRounds` per table (`sends_goal`, `rounds_eq`). -/
theorem consumedJ (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) (hR : RepShapedAt o.toCase)
    (hRS : ∀ i < o.tabs.size, ∀ r < 2,
      (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
        (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r))
    (hLg : ∀ i < o.tabs.size, ∀ r < 2,
      (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
        (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r))
    (hIn : ∀ i < o.tabs.size, ∀ r < 2,
      (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
        (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)) :
    ∀ i < o.tabs.size, ∀ r < 2, ((o.inner i r).2.k != (o.stream i r).rounds.size) = false := by
  have hRp := replayJ h64 o hS hR
  have hJ := recordJsonJ h64 o hS
  obtain ⟨schs, cms, streams, hss, -, ⟨bs, hbs, hrecs⟩, hpts, hrep⟩ := Run.walkJ h64 o hS hR
  have htsz : o.tabs.size = o.sts.size := tabsOf_size _
  let K : ℕ → ℕ := fun i => repRounds true o.spec o.sts[i]!
  have hk : ∀ i (hi : i < o.sts.size), ∀ r < 2, (o.inner i r).2.k = K i ∧
      K i ≤ streams[2 * i + r]!.size := by
    intro i hi r hr
    obtain ⟨htabi, hv, hsch, hreps⟩ := hrep i hi
    have hi' : i < o.tabs.size := by rw [htsz]; exact hi
    obtain ⟨k₀, le₀⟩ := Run.inner_kJ o htabi hv hsch (hRp i hi' r hr) (hRS i hi' r hr) (hLg i hi' r hr)
      (hIn i hi' r hr)
    have hx : (o.extras i).size = o.spec.points * o.sts[i]!.regions.size := by
      have he := hJ.repExtras i hi'
      rw [htabi] at he
      rw [FlockDeploy.extraClaims_size (st := o.sts[i]!) he, hpts]
      simp [pointsOf, Nat.mul_comm]
    have hK : K i = (o.inner i r).2.k := by
      rw [k₀, hx]
      simp [K, repRounds, hv, Setup.ofCircuit]
    refine ⟨hK.symm, ?_⟩
    rw [hK]
    refine le₀.trans (le_of_eq ?_)
    rw [(hreps r hr).1, Array.size_map]
  have hsum : ((List.range (2 * o.sts.size)).map fun j => streams[j]!.size).sum ≤
      ((List.range (2 * o.sts.size)).map fun j => K (j / 2)).sum := by
    have hsend := sends_goal H rel o
    rw [rounds_eq] at hsend
    have hlen := Recs.length_eq hrecs
    have hb : o.bodies.size = 1 + bs.length := by
      rw [← Array.length_toList, hbs]
      simp only [List.length_cons]
      omega
    rw [List.length_flatMap, toList_range streams, List.map_map, hss] at hlen
    rw [sum_range_two]
    have : ((List.range o.sts.size).map fun i => repRounds true o.spec o.sts[i]!).sum =
        ((List.range o.sts.size).map K).sum := rfl
    simp only [Function.comp_def, Array.length_toList] at hlen
    omega
  have heq := eq_of_sum_le (fun j => streams[j]!.size) (fun j => K (j / 2)) (List.range (2 * o.sts.size))
    (fun j hj => by
      have hj' := List.mem_range.1 hj
      have := (hk (j / 2) (by omega) (j % 2) (by omega)).2
      rwa [show 2 * (j / 2) + j % 2 = j by omega] at this) hsum
  intro i hi r hr
  have hi' : i < o.sts.size := by rw [← htsz]; exact hi
  obtain ⟨-, -, -, hreps⟩ := hrep i hi'
  have h1 := heq (2 * i + r) (List.mem_range.2 (by omega))
  simp only [show (2 * i + r) / 2 = i by omega] at h1
  rw [(hreps r hr).1, Array.size_map, h1, (hk i hi' r hr).1]
  exact bne_self_eq_false _

end Consumed

section Several

variable (H rel)

/-- **The rep's checks `complete_zk` takes at one table, at a run**: per rep the ring switch, the masked Ligerito and
the inner proof. -/
structure RepChecksAt (o : Run H rel) : Prop where
  ringSwitch : ∀ i < o.tabs.size, ∀ r < 2,
    (ringSwitchWith o.hs o.tabs[i]!.m (ringClaimsZk (o.rp i r).1 (o.zp i r).proof ++ o.extras i)
      (o.zp i r).proof.ringSwitches).run (o.rp i r).2 = .ok (o.sw i r)
  ligerito : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.ligeritoWith o.hs o.tabs[i]!.merkle (o.sch i) (o.sw i r).1.1 (o.zp i r).proof.cap (o.zp i r).proof.ligerito
      (basis (o.sw i r).1.2) ⟨o.tPad i, r, (o.zp i r).level0⟩).run (o.sw i r).2 = .ok (o.lg i r)
  inner : ∀ i < o.tabs.size, ∀ r < 2,
    (Zk.verifyInnerWith o.hs (o.lay i) (o.zp i r).inner (o.zp i r).proof.zerocheck.finalC
      (repConsZk (o.rp i r).1 (o.zp i r).proof ++ o.lanes i r)).run (o.lg i r).2 = .ok (o.inner i r)

/-- **The rep's checks `complete_zk` takes at one table, at every run of several tables.** -/
def RepChecksJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → RepChecksAt H rel o

end Several

/-- **The rep's checks at a run**, from the ring switch, the masked Ligerito and the inner proof there: the replay
(`replayJ`), level 0's lanes (`lanesJ`) and every round consumed (`consumedJ`). -/
theorem openAt_of (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) (hR : RepShapedAt o.toCase)
    (hC : RepChecksAt H rel o) : OpenAt H rel o :=
  { replay := replayJ h64 o hS hR
    ringSwitch := hC.ringSwitch
    ligerito := hC.ligerito
    lanes := fun i hi r hr => lanesJ (hC.ligerito i hi r hr) ((Run.repJ h64 o hS hR).2.2 i hi r hr).1
    inner := hC.inner
    consumed := consumedJ h64 o hS hR hC.ringSwitch hC.ligerito hC.inner }

/-- **The rep's checks at every run of several tables**, from the ring switch, the masked Ligerito and the inner proof
there. -/
theorem openJ_of (h64 : Out64 H) (hS : StmtOkJ H rel) (hR : RepShapedJ H rel)
    (hC : RepChecksJ H rel) : OpenJ H rel := fun o hc =>
  openAt_of h64 o (hS o.toCase hc) (hR o.toCase hc) (hC o hc)

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables: from `complete_zk`'s premises, and at
several tables the statement's and the prover's pieces' (`StmtOkJ`, `ProverRunsJ`), the proof files' (`RepShapedJ`),
and per rep the ring switch, the masked Ligerito and the inner proof (`RepChecksJ`), the premises `complete_zk` takes
at one table. -/
theorem completeJO (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hRS : FlockComplete.RingSwitchZk H rel) (hLg : FlockComplete.LigeritoZk H rel)
    (hIn : FlockComplete.Inner H rel) (hSJ : StmtOkJ H rel) (hPRJ : ProverRunsJ H rel) (hRJ : RepShapedJ H rel)
    (hCJ : RepChecksJ H rel) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) :
    Claims.Flock.Complete H (deployZk rel) :=
  completeJR H rel h64 hS hR hPR hRS hLg hIn hSJ hPRJ hRJ (openJ_of h64 hSJ hRJ hCJ) hMR hMF

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables, from premises stated per table at every good case, one table's as several's: the statement's
`hello` (`StmtAt`), the prover's pieces (`ProverRunsAt`), the rep proofs' shapes (`RepShapedAt`), and per rep the
ring switch, the masked Ligerito and the inner proof (`RepChecksAt`). -/
theorem completeAll (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (h64 : JsonRT.Out64 H) (hS : ∀ c : Case H rel, StmtAt c)
    (hPR : ∀ c : Case H rel, ProverRunsAt c) (hR : ∀ c : Case H rel, RepShapedAt c)
    (hC : ∀ o : Run H rel, RepChecksAt H rel o) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) :
    Claims.Flock.Complete H (deployZk rel) :=
  complete_of_at H rel h64 (fun c => c.sessionRuns (hS c) (hPR c))
    (fun o => recordAt_of h64 o (hS o.toCase) (filesAt h64 o (hS o.toCase) (hR o.toCase)))
    (fun o => openAt_of h64 o (hS o.toCase) (hR o.toCase) (hC o)) hMR hMF

end FlockComplete.ZkTables
