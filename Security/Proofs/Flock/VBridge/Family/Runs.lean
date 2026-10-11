import Proofs.Flock.VBridge.Runs.Rep
import Proofs.Flock.VBridge.Family
import Proofs.Flock.Soundness.Discharge.DecodeZL.ZL

/-!
# `TableFiles.Runs` from both reps' algebra

`runs_of_files` proves `TableFiles.Runs G` for table files `G` whose fields are what the family's other objects make
them, from `AlgebraHolds` and `AccHolds` at each rep's values `d r` (the family's `F.rep`): each rep's run is
`Runs.run_of_frames`. Its round-by-round frame hypotheses come from one statement about the rep's stream, `FramedAt`:
its rounds hold the chunks of the rep's frames (`repFrames`, as `rep_rounds` reads them off an accepting `verifyRep`),
each round retaining its chunk's bytes and issuing the coins its chunk's closing squeeze asks for.

* `FramedAt.block`, `FramedAt.flat`: a chunk, and a loop's chunks, from the front of the frames.
* `framed_frames`: the zerocheck's, the lincheck's, ring switching's and Ligerito's rounds from `FramedAt`.
* `run_of_files`: one rep's run; `runs_of_files`: `TableFiles.Runs` at table files `RunsFrom` describes;
  `hRuns_of`: `HRuns` at a family whose table files it describes.
* `valid_fast`, `valid_extras_wf`, `valid_roundsWF`, `valid_decode`: the fields of `RunsFrom` that `TableFiles.Valid`
  already gives.
-/

namespace FlockVBridge.Family

open Flock Flock.RecAlgebra FlockSoundness.Refine FlockVBridge.Compiled FlockVBridge.Stages
open FlockSoundness.Discharge.Exec (chunks closes repFrames zcFrames lcFrames rsFrames ligFrames)
open FlockSoundness.Discharge.DecodeZL (chunks_single chunks_append_of_closed)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockVBridge.Runs (run_of_frames)

/-- The coins a chunk's closing squeeze asks for. -/
def sqCount (c : List Frame) : ℕ :=
  match c.getLast? with
  | some (.sq _ n) => n
  | _ => 0

/-- **The rounds from `o` hold the frames `fs`**: round `o + k` retained the bytes of chunk `k` of `fs` and issued the
coins the chunk's closing squeeze asks for, which `squeeze` checks (Transcript.lean:75-86). -/
def FramedAt (rs : Array Round) (o : ℕ) (fs : List Frame) : Prop :=
  ∀ k < (chunks fs).1.length, Retains rs[o + k]! (encs ((chunks fs).1.getD k [])) ∧
    (rs[o + k]!).n = sqCount ((chunks fs).1.getD k [])

theorem chunks_flat (F : ℕ → List Frame) (hF : ∀ i, chunks (F i) = ([F i], [])) :
    ∀ (a n : ℕ), chunks ((List.range' a n).flatMap F) = ((List.range' a n).map F, [])
  | a, 0 => by simp [chunks]
  | a, n + 1 => by
      rw [List.range'_succ, List.flatMap_cons, chunks_append_of_closed (by rw [hF]), hF, chunks_flat F hF (a + 1) n]
      simp

namespace FramedAt

variable {rs : Array Round} {o : ℕ}

theorem split {F₁ F₂ : List Frame} {C : List (List Frame)} (h₁ : chunks F₁ = (C, [])) (h : FramedAt rs o (F₁ ++ F₂)) :
    FramedAt rs o F₁ ∧ FramedAt rs (o + C.length) F₂ := by
  have e := chunks_append_of_closed (F₁ := F₁) (by rw [h₁]) F₂
  simp only [h₁] at e
  refine ⟨fun k hk => ?_, fun k hk => ?_⟩
  · simp only [h₁] at hk ⊢
    have := h k (by rw [e, List.length_append]; omega)
    simp only [e, List.getD_eq_getElem?_getD] at this
    rwa [List.getElem?_append_left hk] at this
  · have := h (C.length + k) (by rw [e, List.length_append]; omega)
    simp only [e, List.getD_eq_getElem?_getD] at this
    rwa [List.getElem?_append_right (by omega), Nat.add_sub_cancel_left, ← Nat.add_assoc,
      ← List.getD_eq_getElem?_getD] at this

/-- A chunk at the front of the frames is round `o`'s. -/
theorem block {F₁ F₂ : List Frame} (h₁ : chunks F₁ = ([F₁], [])) (h : FramedAt rs o (F₁ ++ F₂)) :
    (Retains rs[o]! (encs F₁) ∧ (rs[o]!).n = sqCount F₁) ∧ FramedAt rs (o + 1) F₂ := by
  obtain ⟨h₁', h₂⟩ := split h₁ h
  refine ⟨?_, by simpa using h₂⟩
  have := h₁' 0 (by rw [h₁]; simp)
  simpa [h₁] using this

/-- A loop's chunks at the front of the frames, one per iteration, are rounds `o` to `o + n - 1`. -/
theorem flat {F : ℕ → List Frame} (hF : ∀ i, chunks (F i) = ([F i], [])) {a n : ℕ} {F₂ : List Frame}
    (h : FramedAt rs o ((List.range' a n).flatMap F ++ F₂)) :
    (∀ i < n, Retains rs[o + i]! (encs (F (a + i))) ∧ (rs[o + i]!).n = sqCount (F (a + i))) ∧
      FramedAt rs (o + n) F₂ := by
  obtain ⟨h₁, h₂⟩ := split (chunks_flat F hF a n) h
  rw [List.length_map, List.length_range'] at h₂
  refine ⟨fun i hi => ?_, h₂⟩
  have := h₁ i (by rw [chunks_flat F hF a n, List.length_map, List.length_range']; exact hi)
  simpa [chunks_flat F hF a n, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range' hi] using this

end FramedAt

/-- A rep's frames (`repFrames`, `n + 1` claims), cut into its chunks: the zerocheck's three rounds and its loop, the
lincheck's first two rounds (the first after the zerocheck's final values), its loop and its last round, ring
switching's first claim (after the batching label), its other claims and `γ`'s round, then Ligerito's frames. -/
theorem repFrames_eq (st : Setup) (sch : Flock.Schedule) (D : ByteArray) (n : ℕ) (p : Proof) (T : F128) :
    repFrames st sch D (n + 1) p T =
      ([.dom D, .label "flock-r1cs-v0".toUTF8, .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty),
          .label "flock-zerocheck-v0".toUTF8] ++ [.sq 2 6]) ++
      (([] ++ [.sq 2 (st.m - 13)]) ++
      (([.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList] ++ [.sq 1 1]) ++
      ((List.range' 0 (st.m - 6)).flatMap
          (fun i => [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1]) ++
      (([.one p.zerocheck.finalA, .one p.zerocheck.finalB, .label "flock-lincheck-v0".toUTF8] ++ [.sq 1 1]) ++
      (([] ++ [.sq 1 1]) ++
      ((List.range' 0 (st.fold.kLog - 6)).flatMap
          (fun j => [.one (p.lincheck.rounds[j]!).1, .one (p.lincheck.rounds[j]!).2, .sq 1 1]) ++
      (([.vec p.lincheck.zPartial.toList] ++ [.sq 1 1]) ++
      (([.label "flock-pcs-open-batch-v0".toUTF8, .label "flock-ring-switch-v0".toUTF8,
          .vec (p.ringSwitches[0]!).toList] ++ [.sq 2 7]) ++
      ((List.range' 1 n).flatMap
          (fun k => [.label "flock-ring-switch-v0".toUTF8, .vec (p.ringSwitches[k]!).toList, .sq 2 7]) ++
      (([] ++ [.sq 2 (n + 1)]) ++
      ligFrames sch T p.cap p.ligerito)))))))))) := by
  simp [repFrames, zcFrames, lcFrames, rsFrames, List.range'_succ]

/-- A round that retained `L` answers every byte string whose bytes are `L`. -/
theorem answersL_of_retains {r : Round} {L : List UInt8} (h : Retains r L) : AnswersL r L := by
  intro c hc
  unfold Retains at h
  unfold Round.answers
  cases hb : r.bytes with
  | none => rw [hb] at h; cases h
  | some B =>
      rw [hb] at h
      simp only [Option.map_some, Option.some.injEq] at h
      exact ByteArray.ext (Array.ext' (hc.trans h.symm))

theorem oneCoin_of {r : Round} {L : List UInt8} (h : Retains r L) (hn : r.n = 1) : OneCoin r L := by
  unfold OneCoin
  exact ⟨hn, answers_of_retains h⟩

theorem sq_enc (k : UInt8) (n : ℕ) : (Transcript.header 4 k n).data.toList = (Frame.sq k n).enc := by
  simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]

/-- The bytes ring switching's first claim answers (Opening.lean:68-81): the batching label, then the claim's. -/
theorem rsFramed_zero (s : Array F128) : (rsFramed (rsPending ByteArray.empty 0) s).data.toList =
    encs [.label "flock-pcs-open-batch-v0".toUTF8, .label "flock-ring-switch-v0".toUTF8, .vec s.toList, .sq 2 7] := by
  simp only [rsFramed, rsPending]
  rw [data_append, absorbSlice_enc]
  unfold labelled
  rw [label_enc, label_enc, sq_enc]
  simp [encs]

/-- The bytes ring switching's later claims answer (Opening.lean:76-81). -/
theorem rsFramed_succ (s : Array F128) (k : ℕ) : (rsFramed (rsPending ByteArray.empty (k + 1)) s).data.toList =
    encs [.label "flock-ring-switch-v0".toUTF8, .vec s.toList, .sq 2 7] := by
  simp only [rsFramed, rsPending]
  rw [data_append, absorbSlice_enc]
  unfold labelled
  rw [label_enc, sq_enc]
  simp [encs]

/-- `Flock.fast100` has schedules for `m ≥ 22` only (Schedule.lean:47-48). -/
theorem le_of_fast100 {m : ℕ} {s : Flock.Schedule} (h : Flock.fast100 m = some s) : 22 ≤ m := by
  by_contra hc
  have : Flock.fast100 m = none := by
    unfold Flock.fast100
    split <;> simp_all
  rw [this] at h
  cases h

/-- **A rep's rounds, from its stream holding its frames**: when the rounds of `t` (at round 0, the domain pending)
hold the chunks of `repFrames` at `n` claims, they carry the zerocheck's, the lincheck's, ring switching's and
Ligerito's rounds as `Runs.run_of_frames` asks for them. -/
theorem framed_frames (st : Setup) (sch : Flock.Schedule) (D : ByteArray) (n : ℕ) (hn : n ≠ 0) (p : Proof) (T : F128)
    (t : Transcript) (ht : t.k = 0) (hp : t.pending.data.toList = (Frame.dom D).enc) (hm14 : 14 ≤ st.m)
    (h : FramedAt t.rounds 0 (repFrames st sch D n p T)) :
    (t.rounds[t.k]!).n = 6 ∧ (t.rounds[t.k + 1]!).n = st.m - 13 ∧ (t.rounds[t.k + 2]!).n = 1 ∧
    (∀ i < st.m - 6, (t.rounds[t.k + 3 + i]!).n = 1) ∧
    AnswersL (t.rounds[t.k]!) (t.pending.data.toList ++ encs [.label "flock-r1cs-v0".toUTF8,
      .bytes st.digest, .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6]) ∧
    AnswersL (t.rounds[t.k + 1]!) (encs [.sq 2 (st.m - 13)]) ∧
    AnswersL (t.rounds[t.k + 2]!)
      (encs [.vec p.zerocheck.round1ab.toList, .vec p.zerocheck.round1c.toList, .sq 1 1]) ∧
    (∀ i < st.m - 6, AnswersL (t.rounds[t.k + 3 + i]!)
      (encs [.one (p.zerocheck.rounds[i]!).1, .one (p.zerocheck.rounds[i]!).2, .sq 1 1])) ∧
    LcAnswered ⟨t.rounds, t.k + 3 + (st.m - 6),
      ⟨(encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray⟩⟩ st.fold.kLog p.lincheck ∧
    (∀ k < n, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).n = 7) ∧
    (∀ k < n, (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + k]!).answers
      (rsFramed (rsPending ByteArray.empty k) p.ringSwitches[k]!)) ∧
    (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + n]!).n = n ∧
    (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + n]!).answers
      (ByteArray.empty ++ Transcript.header 4 2 n) ∧
    (∀ i < (chunks (ligFrames sch T p.cap p.ligerito)).1.length,
      Retains (t.rounds[t.k + 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + n + 1 + i]!)
        (encs ((chunks (ligFrames sch T p.cap p.ligerito)).1.getD i []))) := by
  obtain ⟨n, rfl⟩ : ∃ n', n = n' + 1 := ⟨n - 1, by omega⟩
  rw [ht, hp]
  simp only [Nat.zero_add]
  rw [repFrames_eq] at h
  have h13 : st.m - 13 ≠ 0 := by omega
  obtain ⟨⟨r0, n0⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  obtain ⟨⟨r1, n1⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes]; omega)) h
  obtain ⟨⟨r2, n2⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  obtain ⟨hz, h⟩ := FramedAt.flat (fun i => by simp [chunks, closes]) h
  obtain ⟨⟨r3, n3⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  obtain ⟨⟨r4, n4⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  obtain ⟨hl, h⟩ := FramedAt.flat (fun i => by simp [chunks, closes]) h
  obtain ⟨⟨r5, n5⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  obtain ⟨⟨r6, n6⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  obtain ⟨hr, h⟩ := FramedAt.flat (fun i => by simp [chunks, closes]) h
  obtain ⟨⟨r7, n7⟩, h⟩ := FramedAt.block (chunks_single (by simp [closes]) (by simp [closes])) h
  have k₁ : 0 + 1 + 1 + 1 + (st.m - 6) = 3 + (st.m - 6) := by omega
  have k₂ : 0 + 1 + 1 + 1 + (st.m - 6) + 1 + 1 + (st.fold.kLog - 6) + 1 =
      3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 := by omega
  rw [k₁] at r3 n3 r4 n4 hl r5 n5
  rw [k₂] at r6 n6 hr r7 n7 h
  refine ⟨n0, n1, n2.trans rfl, fun i hi => ?_, ?_, answersL_of_retains r1, answersL_of_retains r2,
    fun i hi => ?_, ⟨?_, ?_, fun j hj => ?_, ?_⟩, fun k hk => ?_, fun k hk => ?_, ?_, ?_, fun i hi => ?_⟩
  · have := (hz i hi).2
    rw [show 0 + 1 + 1 + 1 + i = 3 + i by omega] at this
    exact this.trans rfl
  · refine answersL_of_retains ?_
    have e : encs ([.dom D, .label "flock-r1cs-v0".toUTF8, .bytes st.digest,
        .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8] ++ [.sq 2 6]) =
        (Frame.dom D).enc ++ encs [.label "flock-r1cs-v0".toUTF8, .bytes st.digest,
          .bytes (p.cap.foldl (· ++ ·) ByteArray.empty), .label "flock-zerocheck-v0".toUTF8, .sq 2 6] := by
      simp [encs]
    rwa [e] at r0
  · have := (hz i hi).1
    rw [show 0 + 1 + 1 + 1 + i = 3 + i by omega, Nat.zero_add] at this
    exact answersL_of_retains this
  · have e : encs ([.one p.zerocheck.finalA, .one p.zerocheck.finalB, .label "flock-lincheck-v0".toUTF8] ++
        [.sq 1 1]) = (ByteArray.mk (encs [.one p.zerocheck.finalA, .one p.zerocheck.finalB]).toArray).data.toList ++
          encs [.label "flock-lincheck-v0".toUTF8, .sq 1 1] := by
      simp [encs]
    rw [e] at r3
    exact oneCoin_of r3 (n3.trans rfl)
  · exact oneCoin_of r4 (n4.trans rfl)
  · have := hl j hj
    rw [show 3 + (st.m - 6) + 1 + 1 + j = 3 + (st.m - 6) + 2 + j by omega, Nat.zero_add] at this
    exact oneCoin_of this.1 (this.2.trans rfl)
  · rw [show 3 + (st.m - 6) + 1 + 1 + (st.fold.kLog - 6) = 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) by omega] at r5 n5
    exact oneCoin_of r5 (n5.trans rfl)
  · rcases k with _ | k
    · exact n6.trans rfl
    · have := (hr k (by omega)).2
      rw [show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + 1 + k =
        3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (k + 1) by omega] at this
      exact this.trans rfl
  · rcases k with _ | k
    · refine answersL_of_retains ?_ _ (rsFramed_zero _)
      simpa [encs] using r6
    · have := (hr k (by omega)).1
      rw [show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + 1 + k =
        3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (k + 1) by omega, Nat.add_comm 1 k] at this
      exact answersL_of_retains this _ (rsFramed_succ _ k)
  · rw [show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + 1 + n =
      3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (n + 1) by omega] at n7
    exact n7.trans rfl
  · rw [show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + 1 + n =
      3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (n + 1) by omega] at r7
    refine answersL_of_retains r7 _ ?_
    rw [data_append, sq_enc]
    simp [encs]
  · have := (h i hi).1
    rwa [show 3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + 1 + n + 1 + i =
      3 + (st.m - 6) + 2 + (st.fold.kLog - 6) + 1 + (n + 1) + 1 + i by omega] at this

/-! ## The runs -/

variable {cls : HClass}

/-- **Rep `r`'s run of the table files** (`TableFiles.Runs`' body at `r`): `Runs.run_of_frames` at the table's setup,
schedule, extra claims, rep `r`'s proof file and its stream's replay transcript, its frame hypotheses from the stream
holding the rep's frames (`framed_frames`). -/
theorem run_of_files (G : TableFiles cls) (r : ℕ) (sh : Shape) (d : RepData) (halg : AlgebraHolds sh d)
    (hacc : AccHolds sh d)
    -- the table's setup and schedule
    (hm : G.st.m = sh.m) (hkL : G.st.fold.kLog = sh.kLog) (hkS : G.st.kLog = sh.kLog) (hkm : G.st.kLog ≤ G.st.m)
    (hsch : sh.sched = G.sch) (hf : Flock.fast100 G.st.m = some G.sch)
    -- the table's extra claims
    (hxs : d.extras = G.extras)
    (hext : ∀ j < G.extras.size, G.extras[j]!.skip.size = 64 ∧ G.extras[j]!.x.size = G.st.m - 6)
    -- the stream's coins
    (hd : d.rounds = (G.stream r).rounds.map (·.coins)) (hne : NoEmpty d.rounds) (hwf : RoundsWF (G.stream r).rounds)
    -- the proof file: its decode, the messages w holds, the comb port and the levels' openings
    (hdec : decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r))
    (hzp : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ZcProofOf (decoded sh d.w d.comb d.cons) (info.zc.vc.eval (valOf sh d.w d.comb d.cons)) (G.proof r).zerocheck)
    (hlp : LcProofOf (decoded sh d.w d.comb d.cons) (G.proof r).lincheck) (hpl : sh.holo = false)
    (hrsw : (G.proof r).ringSwitches = (decoded sh d.w d.comb d.cons).rs)
    (hmessages : (G.proof r).ligerito.messages =
      (decoded sh d.w d.comb d.cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2))
    (hoodValues : (G.proof r).ligerito.oodValues = (decoded sh d.w d.comb d.cons).ood)
    (hyr : (G.proof r).ligerito.finalProof.yr = (decoded sh d.w d.comb d.cons).yr)
    (hinitialCap : ((G.proof r).ligerito.initialCap == (G.proof r).cap) = true)
    (hfoldPts : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      G.st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
        .ok d.comb)
    (hopen : ∀ v info kappa, verifierRows sh d.rounds d.extras d.comb = .ok (v, info, kappa) →
      ∀ cs ∈ info.lg.conss.toList, Opened G.st.merkle G.sch (G.proof r).cap (G.proof r).ligerito
        (decoded sh d.w d.comb d.cons) cs)
    -- the stream's rounds: the rep's frames at w's batched value
    (hfr : FramedAt (G.stream r).rounds 0 (repFrames G.st G.sch (G.spec.domain G.spec.tables[G.i]! r).toUTF8
      (2 + G.extras.size) (G.proof r) (valOf sh d.w d.comb d.cons .T))) :
    ∃ t₄ : Flock.Transcript, repProg G.st G.sch (G.proof r) G.extras
        (Transcript.init (G.stream r).rounds (G.spec.domain G.spec.tables[G.i]! r).toUTF8) = .ok ((), t₄) ∧
      t₄.k = (G.stream r).rounds.size := by
  have hm14 : 14 ≤ G.st.m := by have := le_of_fast100 hf; omega
  obtain ⟨hn₀, hn₁, hn₂, hnr, ha₀, ha₁, ha₂, har, hlc, hrn, hra, hgn, hga, hlg⟩ :=
    framed_frames G.st G.sch _ (2 + G.extras.size) (by omega) (G.proof r) _
      (Transcript.init (G.stream r).rounds (G.spec.domain G.spec.tables[G.i]! r).toUTF8) rfl (init_enc _ _) hm14 hfr
  exact run_of_frames sh d halg hacc G.st G.sch (G.proof r) (G.bytes r) G.extras _ rfl hd hne hwf hm hm14 hkL hkS hkm
    hxs hext hsch hf hdec hzp hlp hpl hrsw hmessages hoodValues hyr hinitialCap hfoldPts hopen hn₀ hn₁ hn₂ hnr ha₀ ha₁
    ha₂ har hlc hrn hra hgn hga hlg

/-- **What `TableFiles.Runs` asks of the table files**, at the verifier's shape `sh` and each rep's values `d r` (the
family's `F.rep`): the setup's dimensions are the shape's and its schedule `fast100`'s; the extra claims are each rep's
and of the shapes the verifier's claims have; and for each rep, the stream's coins are the rep's, its rounds well
formed, its proof file decodes to messages w holds with Merkle openings that open, and its rounds hold the rep's frames
(`FramedAt`, at w's batched value `T`). -/
structure RunsFrom (G : TableFiles cls) (sh : Shape) (d : Fin 2 → RepData) : Prop where
  m : G.st.m = sh.m
  foldKLog : G.st.fold.kLog = sh.kLog
  kLog : G.st.kLog = sh.kLog
  kLog_le : G.st.kLog ≤ G.st.m
  sched : sh.sched = G.sch
  fast : Flock.fast100 G.st.m = some G.sch
  extras : ∀ r, (d r).extras = G.extras
  extras_wf : ∀ j < G.extras.size, G.extras[j]!.skip.size = 64 ∧ G.extras[j]!.x.size = G.st.m - 6
  coins : ∀ r : Fin 2, (d r).rounds = (G.stream r).rounds.map (·.coins)
  noEmpty : ∀ r : Fin 2, NoEmpty (d r).rounds
  roundsWF : ∀ r : Fin 2, RoundsWF (G.stream r).rounds
  decode : ∀ r : Fin 2, decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r)
  zerocheck : ∀ r : Fin 2, ∀ v info kappa, verifierRows sh (d r).rounds (d r).extras (d r).comb = .ok (v, info, kappa) →
    ZcProofOf (decoded sh (d r).w (d r).comb (d r).cons) (info.zc.vc.eval (valOf sh (d r).w (d r).comb (d r).cons))
      (G.proof r).zerocheck
  lincheck : ∀ r : Fin 2, LcProofOf (decoded sh (d r).w (d r).comb (d r).cons) (G.proof r).lincheck
  ringSwitches : ∀ r : Fin 2, (G.proof r).ringSwitches = (decoded sh (d r).w (d r).comb (d r).cons).rs
  messages : ∀ r : Fin 2, (G.proof r).ligerito.messages =
    (decoded sh (d r).w (d r).comb (d r).cons).msgs.map fun mm => (RecAlgebra.kv mm.1, RecAlgebra.kv mm.2)
  oodValues : ∀ r : Fin 2, (G.proof r).ligerito.oodValues = (decoded sh (d r).w (d r).comb (d r).cons).ood
  yr : ∀ r : Fin 2, (G.proof r).ligerito.finalProof.yr = (decoded sh (d r).w (d r).comb (d r).cons).yr
  initialCap : ∀ r : Fin 2, ((G.proof r).ligerito.initialCap == (G.proof r).cap) = true
  foldPts : ∀ r : Fin 2, ∀ v info kappa, verifierRows sh (d r).rounds (d r).extras (d r).comb = .ok (v, info, kappa) →
    G.st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
      .ok (d r).comb
  opened : ∀ r : Fin 2, ∀ v info kappa, verifierRows sh (d r).rounds (d r).extras (d r).comb = .ok (v, info, kappa) →
    ∀ cs ∈ info.lg.conss.toList, Opened G.st.merkle G.sch (G.proof r).cap (G.proof r).ligerito
      (decoded sh (d r).w (d r).comb (d r).cons) cs
  framed : ∀ r : Fin 2, FramedAt (G.stream r).rounds 0 (repFrames G.st G.sch
    (G.spec.domain G.spec.tables[G.i]! r).toUTF8 (2 + G.extras.size) (G.proof r)
      (valOf sh (d r).w (d r).comb (d r).cons .T))

/-- **`TableFiles.Runs` from both reps' algebra**: at table files that `RunsFrom` describes, when both reps meet E's
preconditions, each rep's program runs (`run_of_files`). -/
theorem runs_of_files (G : TableFiles cls) (sh : Shape) (d : Fin 2 → RepData) (h : RunsFrom G sh d)
    (hpl : sh.holo = false) (halg : ∀ r, AlgebraHolds sh (d r) ∧ AccHolds sh (d r)) : G.Runs := by
  intro r hr
  let r' : Fin 2 := ⟨r, hr⟩
  exact run_of_files G r sh (d r') (halg r').1 (halg r').2 h.m h.foldKLog h.kLog h.kLog_le h.sched h.fast
    (h.extras r') h.extras_wf (h.coins r') (h.noEmpty r') (h.roundsWF r') (h.decode r') (h.zerocheck r') (h.lincheck r')
    hpl (h.ringSwitches r') (h.messages r') (h.oodValues r') (h.yr r') (h.initialCap r') (h.foldPts r') (h.opened r')
    (h.framed r')

/-! ## The fields `TableFiles.Valid` gives -/

section valid

open FlockSoundness.Model FlockSoundness.Discharge.Recursion

variable {G : TableFiles cls} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {M : Hidden cls} {nT nR : ℕ}
  {T : Fin nT → Type} {CT : Fin nR → Type} {t : List (List (SMsg T) × ((k : Fin nR) → CT k))}

/-- `RunsFrom.fast` from `Valid`. -/
theorem valid_fast (hv : G.Valid regs M T CT t) : Flock.fast100 G.st.m = some G.sch := hv.2.2.2.2.1

/-- `RunsFrom.extras_wf` from `Valid`: the extra claims realize the model's (`ClaimRel`). -/
theorem valid_extras_wf (hv : G.Valid regs M T CT t) :
    ∀ j < G.extras.size, G.extras[j]!.skip.size = 64 ∧ G.extras[j]!.x.size = G.st.m - 6 := by
  obtain ⟨hmS, -, -, -, -, hext, -, hex, -⟩ := hv
  intro j hj
  obtain ⟨hsz, hrel⟩ := hext G.extras hex
  have c := hrel j (by rw [← hsz]; exact hj)
  exact ⟨c.skip_size, by rw [hmS]; exact c.x_size⟩

/-- `RunsFrom.roundsWF` from `Valid`: rep `r`'s stream is one of the record's. -/
theorem valid_roundsWF (hv : G.Valid regs M T CT t) (r : ℕ) (hr : r < 2) : RoundsWF (G.stream r).rounds := by
  obtain ⟨-, -, -, -, -, -, hwf, -, hrep, -⟩ := hv
  exact hwf _ (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some (hrep r hr).1))

/-- `RunsFrom.decode` from `Valid`. -/
theorem valid_decode (hv : G.Valid regs M T CT t) (r : ℕ) (hr : r < 2) :
    decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r) := by
  obtain ⟨-, -, -, -, -, -, -, -, hrep, -⟩ := hv
  exact (hrep r hr).2.1

end valid

/-! ## `HRuns` -/

section family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.Model
open FlockSoundness.Audit FlockLevel3 FlockVBridge.InnerAccepts

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

/-- **`HRuns` at any family whose table files `RunsFrom` describes** at its shape and reps, at every accepted outer
proof whose reps meet E's preconditions. -/
theorem hRuns_of (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (h : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
      (∀ r, AlgebraHolds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r) ∧
        AccHolds F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r)) →
      RunsFrom (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)) F.sh
        (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ))) :
    F.HRuns dg N k Rw :=
  fun R ω t R₂ τ o hA halg =>
    runs_of_files _ _ _ (h R ω t R₂ τ o hA halg) (FlockVBridge.OfVStar.shape?_holo F.hsh) halg

end family

end FlockVBridge.Family
