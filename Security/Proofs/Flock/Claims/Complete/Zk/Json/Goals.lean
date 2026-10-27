import Proofs.Flock.Claims.Complete.Json.Goals
import Proofs.Flock.Claims.Complete.Custody
import Proofs.Flock.Claims.Complete.Zk.Json.Session

/-!
# B's record checks at the honest masked run

`Json/Goals.lean` at the masked run (`RunZk`): the goals of `GoalsZk` on the record, from the plain ones' JSON
specification (`JsonSpec`) and 64-byte oracle outputs (`Out64`), and two named hypotheses in place of `StmtOk` and
`ProverOk`:

* `StmtOkZk`: a non-empty `Σ`, a printable draw that decodes (`DrawOk`), the statement's draw the one it was set up at,
  and the verifier's `Hello` a JSON object whose `coins` is an object, printable with the prover's nonce put in (what
  the prover's `Hello` and R7 need). The keyed coin tree is `readyZk`'s. It holds at every good masked case under
  `JsonSpec` (`stmtOkZk`, `Zk/Json/Stmt.lean`).
* `ProverOkZk`: at the case's table, the masked commitment's cap, at the `--zk` schedule's first level, is non-empty and
  under `2^64` bytes, and each masked rep, on the table's extra claims at the link's points, records a round, each of
  at most `65536` coins (`RepOk`).

`RunZk.json` is the record at the run: the printed honest masked record (`sessionPZk_facts`), which `Record.decodeWith`
reads as `sessOf` (`decode_recordObjZk`) and `liveOf` as its live session (`live_recordObjZk`). The goals follow as the
plain ones: `DecodesZk`, `LiveReadsZk`, `DrawMatchesZk`, `CustodyRootBZk`, `CustodyPublicsZk`, `CustodyPointsZk`,
`CustodyCountZk`, `CustodyRoundsZk`, `PublicsZk`, `ProofListZk`, `ProofDigestsZk`, `RepStreamZk` and `RepExtrasZk`.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy
open Lean (Json JsonNumber)

/-- **What the masked record needs of the statement**, at every good masked case: a non-empty `Σ`, a printable draw
that decodes to the statement's instances, the statement's draw the one it was set up at, and the verifier's `Hello` a
JSON object whose `coins` is an object, printable with the prover's nonce put in. -/
def StmtOkZk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : CaseZk H rel, 0 < c.spec.sigma.size ∧ DrawOk c.spec ∧ c.spec.unitDraw = c.s.read.draw ∧
    ∃ m kvs : Std.TreeMap.Raw String Json compare, Json.parse c.spec.hello = .ok (.obj m) ∧ m.WF ∧ kvs.WF ∧
      m.get? "coins" = some (.obj kvs) ∧ Good (helloJ (.obj m))

/-- **What the masked record needs of the prover's pieces**, at every good masked case's table: the masked
commitment's cap, at the `--zk` schedule's first level, is non-empty and under `2^64` bytes, and each masked rep, on the
table's extra claims at the link's points, records a round, each of at most `65536` coins. -/
def ProverOkZk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : CaseZk H rel, ∀ st ∈ c.sts,
    (∀ v, Zk.schedule (Setup.ofCircuit st).m = .ok v → ∀ g extra lanePad salt cm,
      commitPZk (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle (v.1.levels.getD 0 default)
        v.2 g extra lanePad salt = .ok cm → 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64) ∧
    ∀ v, Zk.schedule (Setup.ofCircuit st).m = .ok v →
      ∀ rep dom tr tb cm₀ extras pads ic extra lanePad salt nonce prior ws,
      (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok extras →
      Ret ((repPZk (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) v.1 v.2 rep dom tr tb cm₀
        extras pads ic extra lanePad salt nonce prior ws).run (Transcript.init #[] dom)) fun pt => RepOk pt.2.rounds

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest masked run's record**, at one table: it is the printed honest masked record, `Record.decodeWith`
reads it as `sessOf` and `liveOf` as its live session, its points the first coin's, the link is its `root_B` and no
public words, and the sends after the link are its streams' rounds. -/
theorem RunZk.json (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) (o : RunZk H rel) :
    ∃ st rb streams hz ct, o.sts = #[st] ∧ o.spec.tables = #[st.tags.table] ∧ o.tabs = #[Setup.ofCircuit st] ∧
      streams.size = o.spec.repStreams.size ∧ o.proofs.size = o.spec.repStreams.size ∧
      (∃ e, (Setup.ofCircuit st).extra (pointsOf o.spec (Claims.coinSeq o.r 0)) = .ok e) ∧
      o.link = ([rb], [ByteArray.empty]) ∧
      (∃ bs, o.bodies.toList = linkBody [rb] [ByteArray.empty] :: bs ∧
        Recs o.resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      o.record = printJ (recordObjZk o.spec (Claims.hOf H) hz ct #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams
        o.proofs) ∧
      Record.decodeWith o.hs o.spec o.record =
        .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) ∧
      o.sess = sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs ∧
      liveOf o.spec o.record = .ok o.live ∧
      o.live = ⟨#[rb], o.spec.tables.map fun _ => ByteArray.empty, pointsOf o.spec (Claims.coinSeq o.r 0),
        streams.map (·.map liveRound)⟩ := by
  obtain ⟨st, hst, hsp, -, -⟩ := o.table
  obtain ⟨hsig, hdraw, -, m, kvs, hown, hm, hkv, hco, hg⟩ := hS o.toCaseZk
  obtain ⟨hz, hc⟩ := S.canon_ok _ hg
  have hH : HelloOk o.spec hz := ⟨m, kvs, hown, hm, hkv, hco, hg, hc⟩
  obtain ⟨ct, hct⟩ : ∃ ct, o.spec.coinTree = some ct := Option.isSome_iff_exists.1 o.ready_parts.2.2.2.2
  obtain ⟨hC, hR⟩ := hP o.toCaseZk st (by rw [hst]; simp)
  have ht : o.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun := o.ran
  unfold sessionSZk at hrun
  rw [hst, htabs] at hrun
  obtain ⟨rb, streams, hrec, hrb, hrb64, hss, hrs, hex, bs, hbs, hrecs⟩ :=
    sessionPZk_facts H _ ht hct hown hc (Setup.ofCircuit st) _ _ _ hC hR hrun
  have hpf := sessionPZk_proofs (Claims.hOf H) _ o.spec (tabsOf o.sts) (MaskRank.masksOf o.sts) (tracesOf o.sts o.w)
    o.ω #[] _ _ o.ran
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, ptsOf_coin] at hrec hex
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  dsimp only at hrec
  have hdec : Record.decodeWith o.hs o.spec o.record =
      .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) := by
    rw [hrec]
    exact decode_recordObjZk S (hOf_size h64) ht hct hH hdraw hsig hrb hpts hpm hss hrs o.proofs
  have hlv : liveOf o.spec o.record = .ok ⟨#[rb], o.spec.tables.map fun _ => ByteArray.empty,
      pointsOf o.spec (Claims.coinSeq o.r 0), streams.map (·.map liveRound)⟩ := by
    rw [hrec]
    exact live_recordObjZk S ht (good_draw hdraw) hz ct hss o.proofs
  have hsend : o.sends.headD [] = 0 :: linkBody [rb] [ByteArray.empty] := by
    unfold RunZk.sends
    rw [hbs]
    rfl
  refine ⟨st, rb, streams, hz, ct, hst, ht, by simp [RunZk.tabs, tabsOf, hst], hss, hpf, hex, ?_, ⟨bs, hbs, hrecs⟩,
    hrec, hdec, by unfold RunZk.sess; rw [hdec]; rfl, by unfold RunZk.live; rw [hlv],
    by unfold RunZk.live; rw [hlv]⟩
  unfold RunZk.link
  rw [hsend, linkOf_linkBody (by simp) (by simp) (by simpa using hrb64) (by simp)]
  rfl

/-- **The round messages and their coins**: the sends after the first, the `j`th with coin `1 + j`. -/
theorem RunZk.mids_eq (o : RunZk H rel) {b : Claims.Msg} {bs : List Claims.Msg} (hbs : o.bodies.toList = b :: bs) :
    o.mids = bs.zip ((List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j)) := by
  have hn : o.bodies.size = bs.length + 1 := by
    rw [← Array.length_toList, hbs, List.length_cons]
  unfold RunZk.mids RunZk.sends
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

/-- **The record decodes** (S1–S14): the goal `Decodes`. -/
theorem decodesZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : DecodesZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  rw [hdec, ← hsess]

/-- **The record's live session reads**: the goal `LiveReads`. -/
theorem liveReadsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : LiveReadsZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  exact hlv

/-- **The record is in its canonical text** (S0): the goal `RecordCanonicalZk`, the printed honest record
(`check_printJ`). -/
theorem canonicalZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : RecordCanonicalZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  obtain ⟨-, hdraw, -, -⟩ := hS o.toCaseZk
  rw [hrec]
  exact check_printJ S (good_recordObjZk ht _ hz ct rb _ streams o.proofs (good_draw hdraw))

/-- **The record's draw is the statement's**: the goal `DrawMatches`. -/
theorem drawMatchesZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : DrawMatchesZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  obtain ⟨-, hdraw, hdr, -⟩ := hS o.toCaseZk
  have hd := recordDraw_recordObjZk S (h := Claims.hOf H) ht hdraw hz ct rb (pointsOf o.spec (Claims.coinSeq o.r 0))
    streams o.proofs
  rw [hrec, ← hdr]
  revert hd
  cases o.spec.unitDraw with
  | none => intro hd; rw [hd]; rfl
  | some d =>
    rintro ⟨d', hd', hs⟩
    rw [hd']
    exact (drawEq_iff _ _).2 (.some hs)

/-- **Custody: the record's `root_B` is the link message's**: the goal `CustodyRootB`. -/
theorem custodyRootBZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : CustodyRootBZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  rw [hlive, hlink]
  exact list_bytes_beq_self _

/-- **Custody: the record's public words are the link message's**: the goal `CustodyPublics`. -/
theorem custodyPublicsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : CustodyPublicsZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  rw [hlive, hlink]
  dsimp only
  rw [ht]
  simp only [List.map_toArray, List.map_cons, List.map_nil, List.toList_toArray]
  exact list_bytes_beq_self _

/-- **Custody: the record's link points are the first coin's**: the goal `CustodyPoints`. -/
theorem custodyPointsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : CustodyPointsZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  rw [hlive]
  exact beq_self_eq_true _

/-- **Each table's public words are its statement's**: the goal `Publics`. -/
theorem publicsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : PublicsZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  intro q hq
  rw [htabs, hsess] at hq
  simp only [sessOf, ht, List.map_toArray, List.map_cons, List.map_nil, List.zip_toArray, List.zip_cons_cons,
    List.zip_nil_right, List.mem_cons, List.not_mem_nil, or_false] at hq
  subst hq
  simp [Setup.ofCircuit]
  rfl

/-- **The record's proof digests list exactly the rep streams** (S17): the goal `ProofList`. -/
theorem proofListZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : ProofListZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  have hrs1 := repStreams_one ht
  obtain ⟨p0, p1, hp⟩ := two_of (xs := o.proofs) (by rw [hpf, hrs1]; rfl)
  rw [hsess, hp]
  simp only [sessOf, hrs1, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray, List.map_cons,
    List.map_nil]
  simp only [List.all_toArray, List.any_toArray]
  simp

/-- **Each proof file is the one the record's digest names** (S17): the goal `ProofDigests`. -/
theorem proofDigestsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : ProofDigestsZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  have hrs1 := repStreams_one ht
  obtain ⟨p0, p1, hp⟩ := two_of (xs := o.proofs) (by rw [hpf, hrs1]; rfl)
  intro q hq
  rw [hp, hrs1] at hq
  simp only [List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.toList_toArray, List.mem_cons,
    List.not_mem_nil, or_false] at hq
  rw [hsess, hp]
  simp only [sessOf, hrs1, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray, List.map_cons,
    List.map_nil, List.find?_toArray]
  rcases hq with rfl | rfl
  · simp
  · simp

/-- **The rep's stream is recorded** (S11): the goal `RepStream`. -/
theorem repStreamZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : RepStreamZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  have hrs1 := repStreams_one ht
  obtain ⟨s0, s1, rfl⟩ := two_of (xs := streams) (by rw [hss, hrs1]; rfl)
  intro i hi r hr
  rw [htabs] at hi
  have : i = 0 := by simpa using hi
  subst this
  rw [hsess]
  simp only [sessOf, hrs1, ht, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray,
    List.map_cons, List.map_nil, List.find?_toArray, streamRec]
  rcases (show r = 0 ∨ r = 1 by omega) with rfl | rfl
  · simp
  · simp

/-- **The table's extra claims evaluate** at the link's points: the goal `RepExtras`. -/
theorem repExtrasZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : RepExtrasZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    RunZk.json S h64 hS hP o
  obtain ⟨e, he⟩ := hex
  intro i hi
  rw [htabs] at hi
  have : i = 0 := by simpa using hi
  subst this
  unfold RunZk.extras
  rw [htabs, hsess]
  rw [show (#[Setup.ofCircuit st])[0]! = Setup.ofCircuit st from rfl]
  simp only [sessOf, he]
  rfl

/-- **Custody: each recorded round is its message and its coin's first `n` elements**: the goal `CustodyRounds`. -/
theorem custodyRoundsZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : CustodyRoundsZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, ⟨bs, hbs, hrecs⟩, hrec, hdec, hsess, hlv,
    hlive⟩ :=
    RunZk.json S h64 hS hP o
  have hn : o.bodies.size = bs.length + 1 := by
    rw [← Array.length_toList, hbs, List.length_cons]
  have hmids : o.mids = bs.zip ((List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j)) := by
    unfold RunZk.mids RunZk.sends
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
  rw [hmids, hlive]
  have := recs_all o.r 1 _ bs hrecs
  simpa [Array.toList_map, List.flatMap_map, List.map_flatMap] using this

/-- **Custody: the record has one round per round message**: the goal `CustodyCount`. -/
theorem custodyCountZk_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hS : StmtOkZk H rel)
    (hP : ProverOkZk H rel) : CustodyCountZk H rel := by
  intro o
  obtain ⟨st, rb, streams, hz, ct, hst, ht, htabs, hss, hpf, hex, hlink, ⟨bs, hbs, hrecs⟩, hrec, hdec, hsess, hlv,
    hlive⟩ :=
    RunZk.json S h64 hS hP o
  rw [RunZk.mids_eq o hbs, hlive]
  have := Recs.length_eq hrecs
  simp only [Array.toList_map, List.flatMap_map, List.length_zip, List.length_map, List.length_range, Nat.min_self,
    beq_iff_eq]
  rw [← this, List.length_flatMap, List.length_flatMap]
  simp

end FlockComplete.JsonRT
