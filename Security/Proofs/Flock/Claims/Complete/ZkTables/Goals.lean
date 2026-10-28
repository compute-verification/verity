import Proofs.Flock.Claims.Complete.ZkTables.Prover
import Proofs.Flock.Claims.Complete.ZkTables.Session
import Proofs.Flock.Claims.Complete.ZkTables.Decode
import Proofs.Flock.Claims.Complete.Json.Text

/-!
# The record's checks at the honest masked run of several tables

`Zk/Json/Goals.lean` at a run of any number of tables (`Run`), from `StmtAt` and the prover's pieces
(`proverOkJ`): the record is the printed honest masked record of one `root_B` per table (`sessionPZk_factsJ`), which
`Record.decodeWith` reads as `sessOf` (`decode_recordObjZkJ`) and `liveOf` as its live session (`live_recordObjZkJ`):
`Run.jsonJ`. Its checks follow (`recordJsonJ`): the draw, the live reads, custody (each table's `root_B`, no public
words, the points, one round per round message and each round its message), the decode, each table's public words, the
proof digests listing the rep streams (each found under its name, `find?_name`), every rep's stream, and each table's
extra claims. What `RecordAt` asks of the proof files (`FilesAt`: their sizes, each rep's file decoding at the table's
shape to a proof of its parameters, cap and pads' root) is left named, and `RecordAt` follows from it (`recordAt_of`;
at several tables `recordJ_of`); so `complete` holds with `FilesJ` in place of `RecordJ` (`completeJ`).
-/

namespace FlockComplete.ZkTables

open Flock Transcript FlockComplete FlockDeploy JsonRT
open Lean (Json JsonNumber)

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

theorem tabsOf_size (sts : Array Flock.Stmt) : (tabsOf sts).size = sts.size := by simp [tabsOf]

/-- **The honest masked run's record**, at any number of tables: it is the printed honest masked record of one
`root_B` per table, `Record.decodeWith` reads it as `sessOf` and `liveOf` as its live session, the link is its
`root_B`s and no public words, and the sends after the link are its streams' rounds. -/
theorem Run.jsonJ (S : JsonSpec) (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) :
    ∃ rbs streams hz ct, rbs.size = o.spec.tables.size ∧ streams.size = o.spec.repStreams.size ∧
      o.proofs.size = o.spec.repStreams.size ∧
      o.link = (rbs.toList, (o.tabs.map fun _ => ByteArray.empty).toList) ∧
      (∃ bs, o.bodies.toList = linkBody rbs.toList (o.tabs.map fun _ => ByteArray.empty).toList :: bs ∧
        Recs o.resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      o.record = (printJ (recordObjZk o.spec (Claims.hOf H) hz ct rbs (pointsOf o.spec (Claims.coinSeq o.r 0)) streams
        o.proofs)) ∧
      o.sess = sessOf o.spec (Claims.hOf H) rbs (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs ∧
      Record.decodeWith o.hs o.spec o.record = .ok o.sess ∧
      liveOf o.spec o.record = .ok o.live ∧
      o.live = ⟨rbs, o.spec.tables.map fun _ => ByteArray.empty, pointsOf o.spec (Claims.coinSeq o.r 0),
        streams.map (·.map liveRound)⟩ := by
  obtain ⟨hsig, hdraw, -, m, kvs, hown, hm, hkv, hco, hg⟩ := hS
  obtain ⟨hz, hcz, -⟩ := S.canon_ok _ hg
  have hH : HelloOk o.spec hz := ⟨m, kvs, hown, hm, hkv, hco, hg, hcz⟩
  obtain ⟨ct, hct⟩ : ∃ ct, o.spec.coinTree = some ct := Option.isSome_iff_exists.1 o.ready_parts.2.2.2.2
  have hn := o.toCase.tables_nodup
  obtain ⟨-, hmax, hsz, -⟩ := o.toCase.tables
  have hP := proverOkJ (rel := rel) h64 o.toCase
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
  obtain ⟨rbs, streams, hrbs, hrec, hrb, hss, hrs, bs, hbs, hrecs⟩ :=
    sessionPZk_factsJ H _ hct hown hcz (tabsOf o.sts) _ _ _
      (fun st hst' => by
        obtain ⟨st', hst'', rfl⟩ := Array.mem_map.1 hst'
        exact (hP st' hst'').1)
      (fun st hst' => by
        obtain ⟨st', hst'', rfl⟩ := Array.mem_map.1 hst'
        exact (hP st' hst'').2) hst hrun
  have hpf := sessionPZk_proofs (Claims.hOf H) _ o.spec (tabsOf o.sts) (MaskRank.masksOf o.sts) (traces o.sts o.w) o.ω
    #[] _ _ o.ran
  dsimp only at hrec
  rw [ptsOf_coin] at hrec
  rw [tabsOf_size, ← hsz] at hrbs
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  have hdec : Record.decodeWith o.hs o.spec o.record =
      .ok (sessOf o.spec (Claims.hOf H) rbs (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) := by
    rw [hrec]
    exact decode_recordObjZkJ S (hOf_size h64) hn hct hH hdraw hsig hrbs (fun rb h => (hrb rb h).1) hpts hpm hss hrs
      o.proofs
  have hlv : liveOf o.spec o.record = .ok ⟨rbs, o.spec.tables.map fun _ => ByteArray.empty,
      pointsOf o.spec (Claims.coinSeq o.r 0), streams.map (·.map liveRound)⟩ := by
    rw [hrec]
    exact live_recordObjZkJ S hn (good_draw hdraw) hz ct hrbs hss o.proofs
  have hsend : o.sends.headD [] = 0 :: linkBody rbs.toList (o.tabs.map fun _ => ByteArray.empty).toList := by
    unfold Run.sends
    rw [hbs]
    rfl
  have hsess : o.sess = sessOf o.spec (Claims.hOf H) rbs (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs := by
    unfold Run.sess
    rw [hdec]
    rfl
  refine ⟨rbs, streams, hz, ct, hrbs, hss, hpf, ?_, ⟨bs, hbs, hrecs⟩, hrec, hsess, by rw [hdec, hsess],
    by unfold Run.live; rw [hlv], by unfold Run.live; rw [hlv]⟩
  unfold Run.link
  have hlen : rbs.toList.length < 2 ^ 64 := by
    rw [Array.length_toList, hrbs, hsz]
    unfold Tags.MAX_TABLES at hmax
    omega
  have hlen' : (o.tabs.map fun _ => ByteArray.empty).toList.length < 2 ^ 64 := by
    rw [Array.length_toList, Array.size_map, Run.tabs, tabsOf_size]
    unfold Tags.MAX_TABLES at hmax
    omega
  rw [hsend, linkOf_linkBody hlen hlen'
    (fun b hb => (hrb b (Array.mem_toList_iff.1 hb)).2) (by simp)]
  rfl

/-- **The round messages and their coins**: the sends after the first, the `j`th with coin `1 + j`. -/
theorem Run.mids_eq (o : Run H rel) {b : Claims.Msg} {bs : List Claims.Msg} (hbs : o.bodies.toList = b :: bs) :
    o.mids = bs.zip ((List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j)) := by
  have hn : o.bodies.size = bs.length + 1 := by
    rw [← Array.length_toList, hbs, List.length_cons]
  unfold Run.mids Run.sends
  rw [hbs, hn, List.range_succ_eq_map]
  simp only [List.map_cons, List.cons_append, midsOf, List.map_map]
  rw [List.dropLast_concat]
  have : ((List.range (bs.length + 1)).map (Claims.coinSeq o.r ∘ Nat.succ)).dropLast =
      (List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j) := by
    rw [range_succ_dropLast]
    congr 1
    funext j
    simp [Nat.add_comm]
  rw [this]
  congr 1
  simp [Function.comp_def]

/-- **The record's checks on the record, at a run of several tables**: `RecordAt`'s fields but the proof files'. -/
structure RecordJsonAt (o : Run H rel) : Prop where
  draw : drawEq (FlockSoundness.Discharge.Exec.recordDraw o.record) o.s.read.draw = true
  liveReads : liveOf o.spec o.record = .ok o.live
  rootB : (o.live.rootB.toList == o.link.1) = true
  publicsLink : (o.live.publics.toList == o.link.2) = true
  points : (o.live.points == pointsOf o.spec (Claims.coinSeq o.r 0)) = true
  count : ((o.live.streams.toList.flatMap (·.toList)).length == o.mids.length) = true
  rounds : ((o.live.streams.toList.flatMap (·.toList)).zip o.mids).all
    (fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n) = true
  decodes : Record.decodeWith o.hs o.spec o.record = .ok o.sess
  publics : ∀ q ∈ (o.tabs.zip o.sess.publics).toList, q.1.publics q.2 = .ok ()
  proofList : (o.sess.proofSha.size != o.spec.repStreams.size ||
    !o.spec.repStreams.all fun a => o.sess.proofSha.any fun x => x.1 == a.2.2) = false
  proofDigests : ∀ q ∈ (o.spec.repStreams.zip o.proofs).toList,
    ((o.sess.proofSha.find? (·.1 == q.1.2.2)).map (·.2) != some (hexOf (dgOf o.spec (Claims.hOf H) q.2))) = false
  repStream : ∀ i < o.tabs.size, ∀ r < 2,
    (o.sess.streams.find? (·.name == s!"{o.spec.tables[i]!}/rep{r}")).isSome = true
  repExtras : ∀ i < o.tabs.size, o.tabs[i]!.extra o.sess.points = .ok (o.extras i)
  canonical : Flock.RecordText.check o.record = .ok ()

/-- **The record's checks hold at every run** where the statement's `hello` prints. -/
theorem recordJsonJ (S : JsonSpec) (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) : RecordJsonAt o := by
  obtain ⟨rbs, streams, hz, ct, hrbs, hss, hpf, hlink, ⟨bs, hbs, hrecs⟩, hrec, hsess, hdec, hlv, hlive⟩ :=
    o.jsonJ S h64 hS
  obtain ⟨-, hdraw, hdr, -⟩ := hS
  have hn := o.toCase.tables_nodup
  obtain ⟨-, -, hsz, -⟩ := o.toCase.tables
  have hrsz : o.spec.repStreams.size = 2 * o.tabs.size := by
    rw [repStreams_size, hsz, Run.tabs, tabsOf_size]
  have hpsz : (o.spec.repStreams.zip o.proofs).size = o.spec.repStreams.size := by simp [hpf]
  have hpk : ∀ k (hk : k < o.spec.repStreams.size),
      (o.spec.repStreams.zip o.proofs)[k]'(by rw [hpsz]; exact hk) =
        (o.spec.repStreams[k], o.proofs[k]'(by rw [hpf]; exact hk)) := by
    intro k hk
    simp
  have hnames : ((((o.spec.repStreams.zip o.proofs).map fun q =>
      (q.1.2.2, hexOf (dgOf o.spec (Claims.hOf H) q.2))).toList.map (·.1))).Nodup := by
    have e : (((o.spec.repStreams.zip o.proofs).map fun q =>
        (q.1.2.2, hexOf (dgOf o.spec (Claims.hOf H) q.2))).toList.map (·.1)) =
        o.spec.repStreams.toList.map (·.2.2) := by
      refine List.ext_getElem (by simp [hpf]) fun k h₁ h₂ => ?_
      simp
    rw [e]
    exact repNames_nodup hn
  refine
    { draw := ?_
      liveReads := hlv
      canonical := by
        rw [hrec]
        exact check_printJ S (good_recordObjZkJ hn _ hz ct hrbs _ streams o.proofs (good_draw hdraw))
      rootB := by rw [hlive, hlink]; exact list_bytes_beq_self _
      publicsLink := ?_
      points := by rw [hlive]; exact beq_self_eq_true _
      count := ?_
      rounds := ?_
      decodes := hdec
      publics := ?_
      proofList := ?_
      proofDigests := ?_
      repStream := ?_
      repExtras := ?_ }
  · have hd := recordDraw_recordObjZkJ S (h := Claims.hOf H) hn hdraw hz ct hrbs
      (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs
    rw [hrec, ← hdr]
    revert hd
    cases o.spec.unitDraw with
    | none => intro hd; rw [hd]; rfl
    | some d =>
      rintro ⟨d', hd', hs⟩
      rw [hd']
      exact (drawEq_iff _ _).2 (.some hs)
  · rw [hlive, hlink]
    dsimp only
    have : (o.spec.tables.map fun _ => ByteArray.empty).toList = (o.tabs.map fun _ => ByteArray.empty).toList := by
      simp only [Array.toList_map, List.map_const', Array.length_toList, Run.tabs, tabsOf_size, hsz]
    rw [this]
    exact list_bytes_beq_self _
  · rw [o.mids_eq hbs, hlive]
    have := Recs.length_eq hrecs
    simp only [Array.toList_map, List.flatMap_map, List.length_zip, List.length_map, List.length_range, Nat.min_self,
      beq_iff_eq]
    rw [← this, List.length_flatMap, List.length_flatMap]
    simp
  · rw [o.mids_eq hbs, hlive]
    have := recs_all o.r 1 _ bs hrecs
    simpa [Array.toList_map, List.flatMap_map, List.map_flatMap] using this
  · intro q hq
    rw [hsess] at hq
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hq
    simp only [Array.getElem_toList, Array.getElem_zip, sessOf, Array.getElem_map, Run.tabs, tabsOf]
    simp [Setup.ofCircuit]
    rfl
  · rw [hsess]
    simp only [sessOf]
    refine Bool.or_eq_false_iff.2 ⟨by simp [hpf], ?_⟩
    simp only [Bool.not_eq_false']
    refine Array.all_eq_true.2 fun k hk => ?_
    refine Array.any_eq_true.2 ⟨k, by simp [hpf]; exact hk, ?_⟩
    simp
  · intro q hq
    obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hq
    have hk' : k < o.spec.repStreams.size := by simpa [hpf] using hk
    rw [hsess]
    simp only [sessOf]
    have hf := find?_name (fun x : String × String => x.1) _ hnames k (by simp [hpf, hk'])
    simp only [Array.getElem_toList, Array.find?_toList, Array.getElem_map, hpk k hk'] at hf ⊢
    rw [hf]
    simp
  · intro i hi r hr
    rw [hsess]
    simp only [sessOf]
    have hk : 2 * i + r < o.spec.repStreams.size := by omega
    refine Array.find?_isSome.2 ⟨streamRec o.spec (Claims.hOf H) rbs
      (o.spec.repStreams[2 * i + r], streams[2 * i + r]'(by rw [hss]; exact hk)), ?_, ?_⟩
    · refine Array.mem_map.2 ⟨_, Array.mem_iff_getElem.2 ⟨2 * i + r, by simp [hss, hk], ?_⟩, rfl⟩
      simp
    · simp only [streamRec, repStreams_getElem _ hk, beq_iff_eq]
      have h1 : (2 * i + r) / 2 = i := by omega
      have h2 : (2 * i + r) % 2 = r := by omega
      rw [h1, h2]
  · intro i hi
    have hfit := o.ready_parts.2.2.1
    have hi' : i < o.sts.size := by simpa [Run.tabs, tabsOf_size] using hi
    obtain ⟨e, he⟩ := extraClaims_ok o.sts[i] (hfit _ (Array.getElem_mem hi')) (pointsOf o.spec (Claims.coinSeq o.r 0))
    have ht : o.tabs[i]! = Setup.ofCircuit o.sts[i] := by
      rw [getElem!_pos o.tabs i hi]
      simp [Run.tabs, tabsOf]
    unfold Run.extras
    rw [ht, hsess]
    simp only [sessOf]
    rw [show (Setup.ofCircuit o.sts[i]).extra (pointsOf o.spec (Claims.coinSeq o.r 0)) = .ok e from he]
    rfl

/-- **What the record's checks ask of the proof files, at a run**: their sizes, and each rep's file decoding at the
table's masked shape to a proof of the table's PCS parameters, a 64-byte pads' root, and the table's `root_B` as cap. -/
structure FilesAt (o : Run H rel) : Prop where
  finalFits : o.record.toUTF8.size < 2 ^ 64 ∧ o.proofs.size < 2 ^ 64 ∧ ∀ b ∈ o.proofs, b.size < 2 ^ 64
  repDecodes : ∀ i < o.tabs.size, ∀ r < 2,
    Zk.decode (shapeZk o.tabs[i]! (o.sch i) (o.extras i).size) (o.bytes i r) = .ok (o.zp i r)
  repParams : ∀ i < o.tabs.size, ∀ r < 2, ((o.zp i r).proof.params != pcsParams o.tabs[i]!.merkle (o.sch i)) = false
  padsRoot : ∀ i < o.tabs.size, ∀ r < 2, ((o.zp i r).inner.root.size != 64) = false
  capCommitted : ∀ i < o.tabs.size, ∀ r < 2, (flat (o.zp i r).proof.cap != o.sess.rootB[i]!) = false

variable (H rel) in
/-- **What the record's checks ask of the proof files, at every run of several tables.** -/
def FilesJ : Prop := ∀ o : Run H rel, 1 < o.sts.size → FilesAt o

/-- **The record's checks at a run** where the statement's `hello` prints, from what they ask of the proof files. -/
theorem recordAt_of (S : JsonSpec) (h64 : Out64 H) (o : Run H rel) (hS : StmtAt o.toCase) (hFi : FilesAt o) :
    RecordAt H rel o :=
  have hJ := recordJsonJ S h64 o hS
  { finalFits := hFi.finalFits
    draw := hJ.draw
    liveReads := hJ.liveReads
    canonical := hJ.canonical
    rootB := hJ.rootB
    publicsLink := hJ.publicsLink
    points := hJ.points
    count := hJ.count
    rounds := hJ.rounds
    decodes := hJ.decodes
    publics := hJ.publics
    proofList := hJ.proofList
    proofDigests := hJ.proofDigests
    repStream := hJ.repStream
    repExtras := hJ.repExtras
    repDecodes := hFi.repDecodes
    repParams := hFi.repParams
    padsRoot := hFi.padsRoot
    capCommitted := hFi.capCommitted }

/-- **The record's checks at every run of several tables**, from what they ask of the proof files. -/
theorem recordJ_of (S : JsonSpec) (h64 : Out64 H) (hS : StmtOkJ H rel) (hF : FilesJ H rel) : RecordJ H rel :=
  fun o hc => recordAt_of S h64 o (hS o.toCase hc) (hF o hc)

/-- **Flock's masked verifier is complete** for the honest masked prover on each table's share, with error `δc H`, at
any number of tables, the record's checks discharged: from `complete`'s premises at one table and at several tables
`StmtOkJ`, `ProverRunsJ`, what the record's checks ask of the proof files (`FilesJ`) and the rep's checks (`OpenJ`). -/
theorem completeJ (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (S : JsonRT.JsonSpec) (h64 : JsonRT.Out64 H) (hS : JsonRT.StmtOkZk H rel) (hR : JsonRT.RepShapedZk H rel)
    (hPR : JsonRT.ProverRunsZk H rel) (hRS : FlockComplete.RingSwitchZk H rel) (hLg : FlockComplete.LigeritoZk H rel)
    (hIn : FlockComplete.Inner H rel) (hSJ : StmtOkJ H rel) (hPRJ : ProverRunsJ H rel) (hFJ : FilesJ H rel)
    (hOp : OpenJ H rel) (hMR : MaskRank.MaskRankBound) (hMF : MaskFree rel) : Claims.Flock.Complete H (deployZk rel) :=
  complete H rel S h64 hS hR hPR hRS hLg hIn hSJ hPRJ (recordJ_of S h64 hSJ hFJ) hOp hMR hMF

end FlockComplete.ZkTables
