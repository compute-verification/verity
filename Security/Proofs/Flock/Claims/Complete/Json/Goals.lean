import Proofs.Flock.Claims.Complete.Json.Live
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Soundness.Discharge.ZkView.Decode
import Proofs.Flock.Claims.Complete.Json.Text

/-!
# B's record checks at the honest run

The goals of `Goals` on the record, proved at every honest run (`Run`) from three named hypotheses:

* `Claims.JsonSpec printJ canon`: the JSON library's specification at the honest prover's printer `printJ` (`Spec`),
  since `Json.parse`, `canon` and `Json`'s `==` are opaque.
* `Claims.Out64 H`: the oracle's outputs are 64 bytes, as SHA-512's, which `Claims.Complete` reads. S12 compares each
  round digest's length with the empty message's, so an oracle with outputs of varying length fails the honest record.
* `ProverOk`: at the case's table, the commitment's cap is non-empty and under `2^64` bytes, and each rep, on the
  table's extra claims, records a round, each of at most `65536` coins (`RepOk`); it holds at `Out64`
  (`Replay.proverOk`).

What they read of the statement, `StmtOk` (no keyed coin tree, a non-empty `Σ`, a printable draw that decodes, and the
statement's draw), holds at every good case (`stmtOk`, `Stmt`).

`Run.json` is the record at the run: `Record.decodeWith` reads `sessOf` and `liveOf` the live session, its points the
first coin's (`coinOf_bytes`: a coin's bytes read back are the coin), the first send is the link of its `root_B`, and
the rest are its streams' rounds (`Recs`). The goals follow: `Decodes`, `LiveReads`, `DrawMatches`, `CustodyRootB`,
`CustodyPublics`, `CustodyPoints`, `Publics`, `ProofList`, `ProofDigests`, `RepStream`, `RepExtras` and `CustodyRounds`.
-/

namespace FlockComplete.JsonRT

open Flock Transcript FlockComplete FlockDeploy
open Lean (Json JsonNumber)

theorem hOf_size {H : Claims.Msg → Claims.Msg} (h : Claims.Out64 H) (b : ByteArray) : (Claims.hOf H b).size = 64 := by
  simp only [Claims.hOf, Claims.ofMsg, ByteArray.size, List.size_toArray]
  exact h _

/-- **What the record needs of the prover's pieces**, at every good case's table: the commitment's cap, at the
schedule's first level, is non-empty and under `2^64` bytes, and each rep, on the table's extra claims at the link's
points, records a round, each of at most `65536` coins. -/
def ProverOk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) : Prop :=
  ∀ c : Case H rel, ∀ st ∈ c.sts,
    (∀ sch, fast100 (Setup.ofCircuit st).m = some sch → ∀ g salt cm,
      commitP (Hashes.ofSha512 (Claims.hOf H) : Hashes V) (Setup.ofCircuit st).merkle (sch.levels.getD 0 default) g
        salt = .ok cm → 0 < (flat cm.cap).size ∧ (flat cm.cap).size < 2 ^ 64) ∧
    ∀ sch, fast100 (Setup.ofCircuit st).m = some sch → ∀ tr tb cm₀ extras salt nonce d,
      (Setup.ofCircuit st).extra (ptsOf c.spec (coinOf (c.resp 0))) = .ok extras →
      Ret ((repP (asksS c.resp) (Hashes.ofSha512 (Claims.hOf H)) (Setup.ofCircuit st) sch tr tb cm₀ extras salt
        nonce).run (Transcript.init #[] d)) fun pt => RepOk pt.2.rounds

theorem toBytes_length' (a : F128) : a.toBytes.data.toList.length = 16 := by simp [F128.toBytes]

theorem coin_bytes_length (c : Coin) : c.bytes.length = 16 * 65536 := by
  simp only [Coin.bytes, List.length_flatMap, toBytes_length', List.map_const', List.length_finRange,
    List.sum_replicate, smul_eq_mul]

theorem coin_bytes_split (c : Coin) (i : Fin 65536) :
    ∃ pre post : List UInt8, c.bytes = pre ++ (c i).toBytes.data.toList ++ post ∧ pre.length = 16 * i.val := by
  refine ⟨((List.finRange 65536).take i.val).flatMap fun j => (c j).toBytes.data.toList,
    ((List.finRange 65536).drop (i.val + 1)).flatMap fun j => (c j).toBytes.data.toList, ?_, ?_⟩
  · have hi : i.val < (List.finRange 65536).length := by simp
    conv_lhs => rw [Coin.bytes, ← List.take_append_drop i.val (List.finRange 65536),
      List.drop_eq_getElem_cons hi]
    simp only [List.flatMap_append, List.flatMap_cons, List.getElem_finRange, Fin.cast_mk, Fin.eta,
      List.append_assoc]
  · simp only [List.length_flatMap, toBytes_length', List.map_const', List.length_take, List.length_finRange,
      List.sum_replicate, smul_eq_mul]
    omega

theorem u64OfLE_past (b : ByteArray) (o : ℕ) (ho : b.size ≤ o) : u64OfLE b o = 0 := by
  rw [FlockSoundness.Discharge.ZkView.u64OfLE_eq]
  have : ∀ k, b.get! (o + k) = 0 := fun k => by
    simp only [ByteArray.get!, ByteArray.size] at ho ⊢
    rw [getElem!_neg _ _ (by omega)]
    rfl
  simp only [this]
  decide

/-- A coin's bytes read back: each element is the coin's, and `0` past its 65536. -/
theorem coinOf_bytes (c : Coin) (i : ℕ) : coinOf c.bytes i = c.at i := by
  by_cases hi : i < 65536
  · rw [show c.at i = c ⟨i, hi⟩ by simp [Coin.at, hi]]
    obtain ⟨pre, post, e, hl⟩ := coin_bytes_split c ⟨i, hi⟩
    have hb : Claims.ofMsg c.bytes = ⟨pre.toArray⟩ ++ le64 (c ⟨i, hi⟩).lo ++ (le64 (c ⟨i, hi⟩).hi ++ ⟨post.toArray⟩) := by
      apply ByteArray.ext
      apply Array.toList_inj.1
      simp only [Claims.ofMsg, e, FlockSoundness.Discharge.ZkView.toBytes_split, FlockSoundness.Refine.data_append,
        List.append_assoc]
    have hp : (⟨pre.toArray⟩ : ByteArray).size = 16 * i := by simp [ByteArray.size, hl]
    have hp8 : (⟨pre.toArray⟩ ++ le64 (c ⟨i, hi⟩).lo : ByteArray).size = 16 * i + 8 := by
      rw [ByteArray.size_append, hp, FlockSoundness.Discharge.ZkView.le64_size]
    unfold coinOf
    rw [hb, ← hp8, ← hp, FlockSoundness.Discharge.ZkView.u64OfLE_mid, ← ByteArray.append_assoc,
      FlockSoundness.Discharge.ZkView.u64OfLE_mid]
  · rw [show c.at i = 0 by simp [Coin.at, hi]]
    have hs : (Claims.ofMsg c.bytes).size = 16 * 65536 := by
      simp [Claims.ofMsg, ByteArray.size, coin_bytes_length]
    unfold coinOf
    rw [u64OfLE_past _ _ (by omega), u64OfLE_past _ _ (by omega)]
    rfl

theorem ptsOf_coin (spec : SessionSpec) (c : Coin) : ptsOf spec (coinOf c.bytes) = pointsOf spec c := by
  simp only [ptsOf, pointsOf, coinOf_bytes]

variable {H : Claims.Msg → Claims.Msg} {rel : Readable.Parsed → Readable.Relation}

/-- **The honest run's record**, at one table: it is the printed honest record, `Record.decodeWith` reads it as `sessOf`
and `liveOf` as its live session, its points the first coin's, the link is its `root_B` and no public words, and the
sends after the link are its streams' rounds. -/
theorem Run.json (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) (o : Run H rel) :
    ∃ st rb streams, o.sts = #[st] ∧ o.spec.tables = #[st.tags.table] ∧ o.tabs = #[Setup.ofCircuit st] ∧
      streams.size = o.spec.repStreams.size ∧ o.proofs.size = o.spec.repStreams.size ∧
      (∃ e, (Setup.ofCircuit st).extra (pointsOf o.spec (Claims.coinSeq o.r 0)) = .ok e) ∧
      o.link = ([rb], [ByteArray.empty]) ∧
      (∃ bs, o.bodies.toList = linkBody [rb] [ByteArray.empty] :: bs ∧
        Recs o.resp 1 (streams.toList.flatMap Array.toList) bs) ∧
      o.record = printJ (recordObj o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams
        o.proofs) ∧
      Record.decodeWith o.hs o.spec o.record =
        .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) ∧
      o.sess = sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs ∧
      liveOf o.spec o.record = .ok o.live ∧
      o.live = ⟨#[rb], o.spec.tables.map fun _ => ByteArray.empty, pointsOf o.spec (Claims.coinSeq o.r 0),
        streams.map (·.map liveRound)⟩ := by
  obtain ⟨st, hst, hsp, -, -⟩ := o.table
  obtain ⟨hct, hsig, hdraw, -⟩ := stmtOk o.toCase
  obtain ⟨hC, hR⟩ := hP o.toCase st (by rw [hst]; simp)
  have ht : o.spec.tables = #[st.tags.table] := by rw [hsp]; rfl
  have htabs : tabsOf #[st] = #[Setup.ofCircuit st] := by simp [tabsOf]
  have hrun := o.ran
  unfold sessionS at hrun
  rw [hst, htabs] at hrun
  obtain ⟨rb, streams, hrec, hrb, hrb64, hss, hrs, hex, bs, hbs, hrecs⟩ :=
    sessionP_facts H o.resp ht hct (Setup.ofCircuit st) _ _ hC hR hrun
  have hpf := sessionP_proofs (Claims.hOf H) o.resp o.spec (tabsOf o.sts) (tracesOf o.sts o.w) o.ω #[] _ _ o.ran
  rw [show o.resp 0 = (Claims.coinSeq o.r 0).bytes from rfl, ptsOf_coin] at hrec hex
  have hpts : (pointsOf o.spec (Claims.coinSeq o.r 0)).size = o.spec.points := by simp [pointsOf]
  have hpm : ∀ p ∈ pointsOf o.spec (Claims.coinSeq o.r 0), p.size = o.spec.mPts := by
    intro p hp
    obtain ⟨i, -, rfl⟩ := Array.mem_map.1 hp
    simp
  have hd : Good (o.spec.unitDraw.getD Json.null) := by
    unfold DrawOk at hdraw
    revert hdraw
    cases o.spec.unitDraw with
    | none => exact fun _ => .null
    | some d => exact fun hd => hd.2.1
  dsimp only at hrec
  have hdec : Record.decodeWith o.hs o.spec o.record =
      .ok (sessOf o.spec (Claims.hOf H) #[rb] (pointsOf o.spec (Claims.coinSeq o.r 0)) streams o.proofs) := by
    rw [hrec]
    exact decode_recordObj S (hOf_size h64) ht hct hdraw hsig hrb hpts hpm hss hrs o.proofs
  have hlv : liveOf o.spec o.record = .ok ⟨#[rb], o.spec.tables.map fun _ => ByteArray.empty,
      pointsOf o.spec (Claims.coinSeq o.r 0), streams.map (·.map liveRound)⟩ := by
    rw [hrec]
    exact live_recordObj S ht hd hss o.proofs
  have hsend : o.sends.headD [] = 0 :: linkBody [rb] [ByteArray.empty] := by
    unfold Run.sends
    rw [hbs]
    rfl
  refine ⟨st, rb, streams, hst, ht, by simp [Run.tabs, tabsOf, hst], hss, hpf, hex, ?_, ⟨bs, hbs, hrecs⟩, hrec, hdec,
    by unfold Run.sess; rw [hdec]; rfl, by unfold Run.live; rw [hlv], by unfold Run.live; rw [hlv]⟩
  unfold Run.link
  rw [hsend, linkOf_linkBody (by simp) (by simp) (by simpa using hrb64) (by simp)]
  rfl

/-- **The record decodes** (S1–S14): the goal `Decodes`. -/
theorem decodes_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    Decodes H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  rw [hdec, ← hsess]

/-- **The record's live session reads**: the goal `LiveReads`. -/
theorem liveReads_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    LiveReads H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  exact hlv

/-- **The record is in its canonical text** (S0): the goal `RecordCanonical`, the printed honest record
(`check_printJ`). -/
theorem canonical_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    RecordCanonical H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  obtain ⟨-, -, hdraw, -⟩ := stmtOk o.toCase
  have hd : Good (o.spec.unitDraw.getD Json.null) := by
    unfold DrawOk at hdraw
    revert hdraw
    cases o.spec.unitDraw with
    | none => exact fun _ => .null
    | some d => exact fun hd => hd.2.1
  rw [hrec]
  exact check_printJ S (good_recordObj ht _ rb _ streams o.proofs hd)

/-- **Alike values are equivalent** (`JsonEquiv`, what `drawEq` decides). -/
theorem Same.jsonEquiv {a b : Json} (h : Same a b) : JsonEquiv a b := by
  induction h with
  | null => rfl
  | bool => rfl
  | num => rfl
  | str => rfl
  | arr hs _ ih => exact (jsonEquiv_arr_iff _ _).2 ⟨hs, ih⟩
  | obj hm hn hk _ ih => exact (jsonEquiv_obj_iff hm hn).2 ⟨hk, ih⟩

/-- **The record's draw is the statement's**: the goal `DrawMatches`. -/
theorem drawMatches_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    DrawMatches H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  obtain ⟨-, -, hdraw, hdr⟩ := stmtOk o.toCase
  have hd := recordDraw_recordObj S (h := Claims.hOf H) ht hdraw rb (pointsOf o.spec (Claims.coinSeq o.r 0)) streams
    o.proofs
  rw [hrec, ← hdr]
  revert hd
  cases o.spec.unitDraw with
  | none => intro hd; rw [hd]; rfl
  | some d =>
    rintro ⟨d', hd', hs⟩
    rw [hd']
    exact (drawEq_iff _ _).2 (.some hs)

theorem list_bytes_beq_self (l : List ByteArray) : (l == l) = true := by
  induction l with
  | nil => rfl
  | cons b l ih => simp only [List.cons_beq_cons, bytes_beq_self, ih, Bool.and_self]

/-- **Custody: the record's `root_B` is the link message's**: the goal `CustodyRootB`. -/
theorem custodyRootB_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    CustodyRootB H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  rw [hlive, hlink]
  exact list_bytes_beq_self _

/-- **Custody: the record's public words are the link message's**: the goal `CustodyPublics`. -/
theorem custodyPublics_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    CustodyPublics H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  rw [hlive, hlink]
  dsimp only
  rw [ht]
  simp only [List.map_toArray, List.map_cons, List.map_nil, List.toList_toArray]
  exact list_bytes_beq_self _

/-- **Custody: the record's link points are the first coin's**: the goal `CustodyPoints`. -/
theorem custodyPoints_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    CustodyPoints H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  rw [hlive]
  exact beq_self_eq_true _

/-- **Each table's public words are its statement's**: the goal `Publics`. -/
theorem publics_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    Publics H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  intro q hq
  rw [htabs, hsess] at hq
  simp only [sessOf, ht, List.map_toArray, List.map_cons, List.map_nil, List.zip_toArray, List.zip_cons_cons,
    List.zip_nil_right, List.mem_cons, List.not_mem_nil, or_false] at hq
  subst hq
  simp [Setup.ofCircuit]
  rfl

theorem two_of {α : Type} {xs : Array α} (h : xs.size = 2) : ∃ a b, xs = #[a, b] := by
  rcases xs with ⟨_ | ⟨a, _ | ⟨b, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at h
  exact ⟨a, b, rfl⟩

/-- **The record's proof digests list exactly the rep streams** (S17): the goal `ProofList`. -/
theorem proofList_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    ProofList H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  have hrs1 := repStreams_one ht
  obtain ⟨p0, p1, hp⟩ := two_of (xs := o.proofs) (by rw [hpf, hrs1]; rfl)
  rw [hsess, hp]
  simp only [sessOf, hrs1, List.zip_toArray, List.zip_cons_cons, List.zip_nil_right, List.map_toArray, List.map_cons,
    List.map_nil]
  simp only [List.all_toArray, List.any_toArray]
  simp

/-- **Each proof file is the one the record's digest names** (S17): the goal `ProofDigests`. -/
theorem proofDigests_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    ProofDigests H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
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
theorem repStream_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    RepStream H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
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
theorem repExtras_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    RepExtras H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, -, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  obtain ⟨e, he⟩ := hex
  intro i hi
  rw [htabs] at hi
  have : i = 0 := by simpa using hi
  subst this
  unfold Run.extras
  rw [htabs, hsess]
  rw [show (#[Setup.ofCircuit st])[0]! = Setup.ofCircuit st from rfl]
  simp only [sessOf, he]
  rfl

/-- Rounds that are sends from index `i` on, each answered by its coin's bytes, pass custody's round check against
those sends and coins. -/
theorem recs_all {k : ℕ} (r : Fin k → Coin) :
    ∀ (i : ℕ) (rs : List Round) (bs : List Claims.Msg), Recs (fun j => (Claims.coinSeq r j).bytes) i rs bs →
      ((rs.map liveRound).zip (bs.zip ((List.range bs.length).map fun j => Claims.coinSeq r (i + j)))).all
        (fun (x, a, c) => x.bytes == some ⟨a.toArray⟩ && x.coins == c.vals x.n) = true
  | _, [], [], _ => rfl
  | i, x :: rs, b :: bs, h => by
    obtain ⟨⟨c, hb, rfl, -, hc⟩, h⟩ := h
    rw [List.length_cons, List.range_succ_eq_map]
    have hm : ((List.range bs.length).map Nat.succ).map (fun j => Claims.coinSeq r (i + j)) =
        (List.range bs.length).map fun j => Claims.coinSeq r (i + 1 + j) := by
      simp only [List.map_map, Function.comp_def, Nat.succ_eq_add_one]
      congr 1
      funext j
      rw [Nat.add_comm j 1, Nat.add_assoc]
    simp only [List.map_cons, List.zip_cons_cons, List.all_cons, hm, recs_all r (i + 1) rs bs h, Bool.and_true,
      Nat.add_zero]
    have hcv : x.coins = (Claims.coinSeq r i).vals x.n := by
      rw [hc, Coin.vals]
      congr 1
      funext j
      exact coinOf_bytes _ j
    simp only [liveRound, hb, Option.getD_some, Claims.toMsg, Array.toArray_toList, hcv]
    simp only [Bool.and_eq_true]
    exact ⟨bytes_beq_self c, beq_self_eq_true _⟩
  | _, [], _ :: _, h => h.elim
  | _, _ :: _, [], h => h.elim

theorem range_succ_dropLast {α : Type} (f : ℕ → α) (n : ℕ) :
    ((List.range (n + 1)).map f).dropLast = (List.range n).map f := by
  rw [List.range_succ, List.map_append, List.map_singleton, List.dropLast_concat]

/-- **Custody: each recorded round is its message and its coin's first `n` elements**: the goal `CustodyRounds`. -/
theorem custodyRounds_goal (S : Claims.JsonSpec printJ Flock.canon) (h64 : Claims.Out64 H) (hP : ProverOk H rel) :
    CustodyRounds H rel := by
  intro o
  obtain ⟨st, rb, streams, hst, ht, htabs, hss, hpf, hex, hlink, ⟨bs, hbs, hrecs⟩, hrec, hdec, hsess, hlv, hlive⟩ :=
    Run.json S h64 hP o
  have hn : o.bodies.size = bs.length + 1 := by
    rw [← Array.length_toList, hbs, List.length_cons]
  have hmids : o.mids = bs.zip ((List.range bs.length).map fun j => Claims.coinSeq o.r (1 + j)) := by
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
  rw [hmids, hlive]
  have := recs_all o.r 1 _ bs hrecs
  simpa [Array.toList_map, List.flatMap_map, List.map_flatMap] using this

end FlockComplete.JsonRT
