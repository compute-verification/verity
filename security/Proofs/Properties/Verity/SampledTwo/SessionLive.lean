module

public import Proofs.Properties.Verity.SampledTwo.SessionMiddle

/-!
# A session the masked verdict accepts has messages

A session's messages (`SessionMsgs.sessionMsgs`) are read from its record: the link message from the live session
`FlockDeploy.liveOf` reads, then each recorded round's retained bytes, then the last message. There are none where the
record doesn't read as a live session or a round kept no bytes. Here both hold of a one-table session the masked
verdict (`Flock.Zk.verify`) accepts at a setup that retains rounds.

* **The record reads as a live session** (`live_of_verify`). The verdict decoded the record
  (`Verity.Sampled.zkVerify_decode`), and `FlockDeploy.liveOf` reads no field `Flock.Record.decode` did not: the link,
  each table's `root_B` (`rootField_reads`), public words (`pubField_reads`) and link points (`pointField_reads`),
  as `Claims.Bridge.Custody.decode_ok` and `FlockDeploy.KS.decode_pubs` give them, and each rep stream's rounds
  (`roundFields_reads`, `streamJ_of_decode`). The verdict ran both reps of the one table
  (`Verity.Sampled.verifyRep_of_accepted`), and each rep's check found its stream by name among the decoded
  (`FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok`), so the record's streams hold it (`streamJ_of_decode`).
* **Each round kept its bytes**: the live session's rounds have the decoded rounds' bytes, and the decoder refuses a
  round without them where the setup retains rounds (`FlockSoundness.Discharge.Exec.decode_retained`).
* **The session has messages** (`sessionMsgs_of_verify`).

**Not established here**: that a session's setup retains rounds and has one table, each a hypothesis.
-/

@[expose] public section

namespace Verity.SampledTwo.SessionLive

open Claims
open Lean (Json)
open Verity.Sampled (repStreamName)
open Claims.Bridge.Custody (RootField PointField RoundFields DecStream)

/-! ## Reads that return -/

/-- A read that returns, then a read of what it returned that returns, returns. -/
theorem bind_returns {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

theorem list_mapM_returns {α β ε : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, List.Forall₂ (fun a b => f a = .ok b) l l' → l.mapM f = .ok l'
  | _, _, .nil => rfl
  | _, _, .cons hab h => by
    rw [List.mapM_cons]
    exact bind_returns hab (bind_returns (list_mapM_returns h) rfl)

/-- A read of each member that returns reads the array (`Claims.Bridge.Custody.mapM_forall₂`, turned around). -/
theorem mapM_returns {α β ε : Type} {f : α → Except ε β} {xs : Array α} {ys : Array β}
    (h : List.Forall₂ (fun a b => f a = .ok b) xs.toList ys.toList) : xs.mapM f = .ok ys := by
  rw [Array.mapM_eq_mapM_toList, list_mapM_returns h]
  rfl

theorem mapM_some_of {α β : Type} {f : α → Option β} :
    ∀ {l : List α}, (∀ a ∈ l, ∃ b, f a = some b) → ∃ bs, l.mapM f = some bs
  | [], _ => ⟨[], rfl⟩
  | a :: l, h => by
    obtain ⟨b, hb⟩ := h a List.mem_cons_self
    obtain ⟨bs, hbs⟩ := mapM_some_of (l := l) fun a' ha' => h a' (List.mem_cons_of_mem a ha')
    exact ⟨b :: bs, by rw [List.mapM_cons, hb, hbs]; rfl⟩

/-! ## What the live session reads, of what the decoder read -/

/-- The live session reads a table's `root_B` the decoder read. -/
theorem rootField_reads {l : Json} {t : String} {rb : ByteArray} (h : RootField l t rb) :
    (do Flock.unhex (← Flock.Record.str (← Flock.Record.fld l "root_b") t) : Flock.V ByteArray) = .ok rb := by
  obtain ⟨v, s, h1, h2, h3⟩ := h
  exact bind_returns h1 (bind_returns h2 h3)

/-- The live session reads a table's public words the decoder read. -/
theorem pubField_reads {pubs : Json} {t : String} {p : ByteArray} (h : FlockDeploy.KS.PubField pubs t p) :
    (if Flock.Record.isNull pubs t then pure ByteArray.empty
      else do Flock.unhex (← Flock.Record.str pubs t) : Flock.V ByteArray) = .ok p := by
  cases hn : Flock.Record.isNull pubs t
  · obtain ⟨s, h1, h2⟩ := h.2 hn
    rw [ite_eq_right (by simp)]
    exact bind_returns h1 h2
  · rw [h.1 hn, ite_eq_left rfl]
    rfl

/-- The live session reads a link point the decoder read. -/
theorem pointField_reads {p : Json} {pt : Array Flock.F128} (h : PointField p pt) :
    (match p.getArr? with
      | .ok a => Flock.Record.f128s a
      | .error _ => .error "record: a link point is not a list" : Flock.V (Array Flock.F128)) = .ok pt := by
  obtain ⟨a, h1, h2⟩ := h
  rw [h1]
  exact h2

/-- **The live session reads a round the decoder read**: its coin count, coins and retained bytes, with no digest. -/
theorem roundFields_reads {r : Json} {n : ℕ} {coins : Array Flock.F128} {bytes : Option ByteArray}
    (h : RoundFields r n coins bytes) : FlockDeploy.roundJ r = .ok ⟨n, ByteArray.empty, coins, bytes⟩ := by
  obtain ⟨hn, ⟨cs, hcs, hcoins⟩, h0, h1⟩ := h
  unfold FlockDeploy.roundJ
  refine bind_returns hn (bind_returns hcs (bind_returns hcoins ?_))
  dsimp only
  cases hnull : Flock.Record.isNull r "msg"
  · obtain ⟨s, b, hs, hb, rfl⟩ := h1 hnull
    rw [ite_eq_right (by simp)]
    refine bind_returns hs (bind_returns (a := some b) ?_ rfl)
    show some <$> Flock.unhex s = _
    rw [hb]
    rfl
  · rw [h0 hnull, ite_eq_left rfl]
    rfl

/-- **The live session reads a rep stream the decoded record has by name**: the record's first stream so named is
the one the decoder read as it (`Claims.Bridge.Custody.stream_pred`), and its rounds read as the decoded rounds, each
with the decoded round's bytes. -/
theorem streamJ_of_decode {spec : Flock.SessionSpec} {recs : Array Json} {sess : Flock.RunRecord} {name : String}
    {y : Flock.StreamRec} (hstr : List.Forall₂ (DecStream spec) recs.toList sess.streams.toList)
    (hy : sess.streams.find? (fun x => x.name == name) = some y) :
    ∃ rds, FlockDeploy.streamJ recs name = .ok rds ∧
      List.Forall₂ (fun (rd rd' : Flock.Round) => rd'.bytes = rd.bytes) y.rounds.toList rds.toList := by
  have hy' : sess.streams.toList.find? (fun x => x.name == name) = some y := by rw [Array.find?_toList]; exact hy
  obtain ⟨x, ⟨-, rs, hrs, hrds⟩, hx⟩ := Claims.Bridge.Custody.forall₂_find?
    (R := fun (y : Flock.StreamRec) (x : Json) => DecStream spec x y) (p := fun y => y.name == name)
    (q := fun s => (s.getObjValAs? String "stream").toOption == some name)
    (fun a b hab => Claims.Bridge.Custody.stream_pred hab.1) (Verity.Sampled.forall₂_swap hstr) hy'
  rw [Array.find?_toList] at hx
  refine ⟨y.rounds.map fun rd => ⟨rd.n, ByteArray.empty, rd.coins, rd.bytes⟩, ?_, ?_⟩
  · unfold FlockDeploy.streamJ
    rw [hx]
    refine bind_returns hrs (mapM_returns ?_)
    rw [Array.toList_map, List.forall₂_map_right_iff]
    exact hrds.imp fun r rd h => roundFields_reads h.1
  · rw [Array.toList_map, List.forall₂_map_right_iff]
    exact List.forall₂_same.2 fun _ _ => rfl

/-! ## The live session of an accepted record -/

/-- **The record of a one-table session the masked verdict accepts reads as a live session, each of whose rounds
kept its bytes**, where the session retains rounds. The verdict decoded the record
(`Verity.Sampled.zkVerify_decode`); the live session's link, roots, public words and link points are fields the
decoder read (`Claims.Bridge.Custody.decode_ok`, `FlockDeploy.KS.decode_pubs`); each of the table's two rep streams
is one a rep's check found among the decoded (`Verity.Sampled.verifyRep_of_accepted`,
`FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok`), so the live session reads it (`streamJ_of_decode`) with the
decoded rounds' bytes, which the decoder refuses a round without
(`FlockSoundness.Discharge.Exec.decode_retained`). -/
theorem live_of_verify {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} {record : String}
    {proofs : Array ByteArray} {T : String} (hv : Flock.Zk.verify spec tabs record proofs = .ok ())
    (hT : spec.tables = #[T]) (hret : spec.retainRounds = true) :
    ∃ lv, FlockDeploy.liveOf spec record = .ok lv ∧
      ∀ r ∈ lv.streams.toList.flatMap Array.toList, ∃ b, r.bytes = some b := by
  obtain ⟨sess, hd⟩ := Verity.Sampled.zkVerify_decode hv
  obtain ⟨j, l, recs, hp, hl, hr, hroot, ⟨ps, hps, hpts⟩, hstr⟩ := Claims.Bridge.Custody.decode_ok hd
  obtain ⟨j', l', pubs, hp', hl', hpubs, hpf⟩ := FlockDeploy.KS.decode_pubs hd
  rw [hp] at hp'
  obtain rfl := Except.ok.inj hp'
  rw [hl] at hl'
  obtain rfl := Except.ok.inj hl'
  have hrep : ∀ q : Fin 2, ∃ rds, FlockDeploy.streamJ recs (repStreamName T q.val) = .ok rds ∧
      ∀ r ∈ rds.toList, ∃ b, r.bytes = some b := by
    intro q
    obtain ⟨_, sch, tPad, c, -, hrepq⟩ := Verity.Sampled.verifyRep_of_accepted hv hd q
    obtain ⟨y, _, _, _, hf, -⟩ := FlockSoundness.Discharge.ZkExec.zk_verifyRep_ok _ _ _ _ _ _ _ _ _ hrepq
    have hf' : sess.streams.find? (fun x => x.name == repStreamName T q.val) = some y := by
      rw [hT] at hf
      exact hf
    obtain ⟨rds, hrds, hb⟩ := streamJ_of_decode hstr hf'
    refine ⟨rds, hrds, fun r hr => ?_⟩
    obtain ⟨rd, hrd, hbytes⟩ := Claims.Bridge.Custody.forall₂_mem_left (Verity.Sampled.forall₂_swap hb) r hr
    obtain ⟨b, hb'⟩ := FlockSoundness.Discharge.Exec.decode_retained hret hd y
      (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hf')) rd hrd
    exact ⟨b, hbytes.trans hb'⟩
  obtain ⟨rds0, h0, hb0⟩ := hrep 0
  obtain ⟨rds1, h1, hb1⟩ := hrep 1
  refine ⟨⟨sess.rootB, sess.publics, sess.points, #[rds0, rds1]⟩, ?_, ?_⟩
  · rw [FlockDeploy.liveOf_of_parse spec hp]
    unfold FlockDeploy.liveOfJ
    refine bind_returns hl (bind_returns (mapM_returns (hroot.imp fun _ _ h => rootField_reads h))
      (bind_returns hpubs (bind_returns (mapM_returns (hpf.imp fun _ _ h => pubField_reads h))
        (bind_returns hps (bind_returns (mapM_returns (hpts.imp fun _ _ h => pointField_reads h))
          (bind_returns hr (bind_returns (mapM_returns ?_) rfl)))))))
    rw [Verity.Sampled.repStreams_table hT]
    exact .cons h0 (.cons h1 .nil)
  · intro r hr
    simp only [List.flatMap_cons, List.flatMap_nil, List.append_nil, List.mem_append] at hr
    rcases hr with hr | hr
    · exact hb0 r hr
    · exact hb1 r hr

/-- **A one-table session the masked verdict accepts has messages**, where the session retains rounds: its record
reads as a live session, each of whose rounds kept its bytes (`live_of_verify`), which is all
`SessionMsgs.sessionMsgs` reads of it. -/
theorem sessionMsgs_of_verify {spec : Flock.SessionSpec} {tabs : Array Flock.Setup} (x : Msg) {record : String}
    {proofs : Array ByteArray} {T : String} (hv : Flock.Zk.verify spec tabs record proofs = .ok ())
    (hT : spec.tables = #[T]) (hret : spec.retainRounds = true) :
    ∃ ms, SessionMsgs.sessionMsgs spec x record proofs = some ms := by
  obtain ⟨lv, hl, hb⟩ := live_of_verify hv hT hret
  obtain ⟨mids, hmids⟩ := mapM_some_of
    (f := fun r : Flock.Round => r.bytes.map fun b => (0 : UInt8) :: b.data.toList) (fun r hr => by
      obtain ⟨b, hb'⟩ := hb r hr
      exact ⟨_, by rw [hb']; rfl⟩)
  refine ⟨(0 :: FlockDeploy.linkBody lv.rootB.toList lv.publics.toList) :: mids ++
    [FlockDeploy.Final.encode ⟨x, record, proofs⟩], ?_⟩
  unfold SessionMsgs.sessionMsgs
  simp only [Option.bind_eq_bind, Option.bind_eq_some_iff, Option.pure_def, Option.some.injEq]
  exact ⟨lv, by rw [hl]; rfl, mids, hmids, rfl⟩

end Verity.SampledTwo.SessionLive
