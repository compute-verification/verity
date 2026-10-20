import Proofs.Flock.Claims.Complete.ZkTables.Round1

/-!
# The masked prover's third message: table `0` rep `0`'s outer zerocheck point

Round `2` of the masked session on the executable's side. Right after the zerocheck's first squeeze (`rSkip`, six
coins) the masked zerocheck squeezes its outer point (`rOuter`, `m − 13` coins) with nothing absorbed in between, so its
third body is that squeeze's frame of a transcript whose pending bytes are empty: the header of `m − 13` coins of kind
`2` alone (`Replay.framed_afterSq`), whatever the coins, the cut or the witness.

* `pv_sim_pre`, `st_sim_pre`, `pv_sim_cons`, `st_sim_cons`: a bind's bodies start with its first part's, and a part
  that sends one body and returns is followed by the rest's from the next index.
* `sampleVecP_pv`: a squeeze of `n ≠ 0` coins in `PV` sends its frame and returns the reply's first `n` coins.
* `zerocheckPZk_pv_head2`, `repPZk_pv_head2`, `afterLinkPZk_pv_head2`: the first two bodies of a rep, and of the session
  after the link's coin where rep `0` of table `0` is the first stream and its extra claims return.
* `sessionPZk_pv_round2`, `sessionPZk_go_round2`: the session's first three bodies, and its third at the pure session.
-/

open Flock FlockDeploy

namespace FlockComplete

variable (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg)

theorem pv_sim_pre {α β : Type} {x : PV α} {f : α → PV β} {i : ℕ} {ms : List Claims.Msg}
    (hx : ∃ bs, (x.run.sim H resp i).1 = ms ++ bs) : ∃ bs', ((x >>= f).run.sim H resp i).1 = ms ++ bs' := by
  obtain ⟨bs, hx⟩ := hx
  rw [pv_sim_bind, hx, List.append_assoc]
  exact ⟨_, rfl⟩

theorem st_sim_pre {σ α β : Type} {x : StateT σ PV α} {f : α → StateT σ PV β} {s : σ} {i : ℕ}
    {ms : List Claims.Msg} (hx : ∃ bs, ((x.run s).run.sim H resp i).1 = ms ++ bs) :
    ∃ bs', (((x >>= f).run s).run.sim H resp i).1 = ms ++ bs' := by
  rw [StateT.run_bind]
  exact pv_sim_pre H resp hx

theorem pv_sim_cons {α β : Type} {x : PV α} {f : α → PV β} {i : ℕ} {m : Claims.Msg} {a : α}
    (hx : x.run.sim H resp i = ([m], .ok a)) :
    ((x >>= f).run.sim H resp i).1 = m :: ((f a).run.sim H resp (i + 1)).1 := by
  rw [pv_sim_bind, hx]
  rfl

theorem st_sim_cons {σ α β : Type} {x : StateT σ PV α} {f : α → StateT σ PV β} {s s' : σ} {i : ℕ}
    {m : Claims.Msg} {a : α} (hx : (x.run s).run.sim H resp i = ([m], .ok (a, s'))) :
    (((x >>= f).run s).run.sim H resp i).1 = m :: (((f a).run s').run.sim H resp (i + 1)).1 := by
  rw [StateT.run_bind]
  exact pv_sim_cons H resp hx

/-- **A squeeze of `n ≠ 0` coins in `PV`** sends its frame, tagged `0`, and returns the reply's first `n` coins and
the transcript after it. -/
theorem sampleVecP_pv {n : ℕ} (hn : n ≠ 0) (t : Transcript) (i : ℕ) :
    ((sampleVecP asks n).run t).run.sim H resp i =
      ([0 :: Claims.toMsg (Transcript.framed t 2 n)],
        .ok ((Array.range n).map (coinOf (resp i)), Replay.afterSq t 2 n (coinOf (resp i)))) := by
  simp only [sampleVecP, squeezeP, StateT.run, hn, ite_false]
  rfl

/-- **After a squeeze nothing is pending**: the next frame is its header alone. -/
theorem Replay.framed_afterSq (t : Transcript) (kind : UInt8) (n : ℕ) (coin : ℕ → F128) (kind' : UInt8) (n' : ℕ) :
    Transcript.framed (Replay.afterSq t kind n coin) kind' n' = Transcript.framed ⟨#[], 0, ByteArray.empty⟩ kind' n' :=
  rfl

/-- **The masked zerocheck's first two bodies**: its first squeeze's frame, of the transcript with the circuit's
digest, the level-0 cap and the pads' root absorbed (`Replay.ZcZ.t₁`), then its outer point's, of `m − 13` coins
squeezed right after. -/
theorem zerocheckPZk_pv_head2 (m : ℕ) (hm : 14 ≤ m) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (h : Array F128) (tr : Trace) (t : Transcript) (i : ℕ) :
    ∃ bs, (((zerocheckPZk asks m digest cap root lay h tr).run t).run.sim H resp i).1 =
      [0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ digest cap root t) 2 6),
        0 :: Claims.toMsg (Transcript.framed (Replay.afterSq (Replay.ZcZ.t₁ digest cap root t) 2 6
          (coinOf (resp i))) 2 (m - 13))] ++ bs := by
  unfold zerocheckPZk
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.label_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.absorbBytes_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.absorbBytes_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.label_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.label_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.absorbBytes_eq _ _)))]
  rw [st_sim_cons H resp (sampleVecP_pv H resp (n := 6) (by decide) _ i)]
  rw [st_sim_cons H resp (sampleVecP_pv H resp (n := m - 13) (by omega) _ (i + 1))]
  exact ⟨_, rfl⟩

end FlockComplete

namespace FlockComplete

variable (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg)

theorem repPZk_pv_head2 (st : Setup) (hm : 14 ≤ st.m) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (T : Transcript) (i : ℕ) :
    ∃ bs, (((repPZk asks Hashes.pi st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior
      words).run T).run.sim H resp i).1 =
      [0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ st.digest (flat cm0.cap) ic.root T) 2 6),
        0 :: Claims.toMsg (Transcript.framed (Replay.afterSq (Replay.ZcZ.t₁ st.digest (flat cm0.cap) ic.root T) 2 6
          (coinOf (resp i))) 2 (st.m - 13))] ++ bs := by
  unfold repPZk
  exact st_sim_pre H resp (zerocheckPZk_pv_head2 H resp st.m hm st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr T i)

/-- **The session's first two bodies after the link's coin**, where table `0`'s rep `0` is the first stream and its
extra claims and pads' commitment return: that rep's first zerocheck squeeze, from the domain's transcript, then its
outer point's. -/
theorem afterLinkPZk_pv_head2 (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
    (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic)
    (i : ℕ) :
    ∃ bs, ((afterLinkPZk asks Hashes.pi spec tabs words trs tape schs tbls cms c1).run.sim H resp i).1 =
      [0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap) ic.root
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6),
        0 :: Claims.toMsg (Transcript.framed (Replay.afterSq (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap) ic.root
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6 (coinOf (resp i))) 2
          (tabs[0]!.m - 13))] ++ bs := by
  unfold afterLinkPZk
  dsimp only
  rw [← Array.forIn_toList, h0, List.forIn_cons]
  refine pv_sim_pre H resp ?_
  refine pv_sim_pre H resp ?_
  rw [pv_sim_quiet H resp ((monadLift_pv H resp _ i).trans (congrArg _ he))]
  rw [pv_sim_quiet H resp ((innerCommitP_pv H resp _ _ i).trans (congrArg _ hi))]
  exact pv_sim_pre H resp (repPZk_pv_head2 H resp tabs[0]! hm schs[0]!.1 schs[0]!.2 0
    (spec.domain spec.tables[0]! 0).toUTF8 trs[0]! tbls[0]! cms[0]! extras
    (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) ic
    (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0) (tape.nonce 0 0) (priorOf tabs[0]! #[] 0) words[0]!
    (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8) i)

/-- **The masked session's first three bodies** in `PV`: the link message of the level-0 caps, then table `0` rep `0`'s
first zerocheck squeeze and its outer point's. -/
theorem sessionPZk_pv_round2 (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
    (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic) :
    ∃ bs, ((sessionPZk asks Hashes.pi spec tabs words trs tape).run.sim H resp 0).1 =
      (0 :: linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList) ::
        ([0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap) ic.root
            (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6),
          0 :: Claims.toMsg (Transcript.framed (Replay.afterSq (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap)
            ic.root (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6 (coinOf (resp 1))) 2
            (tabs[0]!.m - 13))] ++ bs) := by
  rw [sessionPZk_eq]
  rw [pv_sim_quiet H resp ((schedsPZk_pv H resp tabs 0).trans (congrArg _ hs))]
  rw [pv_sim_quiet H resp ((commitsPZk_pv H resp tabs schs _ tape 0).trans (congrArg _ hc))]
  rw [link_pv_sim]
  obtain ⟨bs, hb⟩ := afterLinkPZk_pv_head2 H resp spec tabs hm words trs tape schs (tablesP tabs trs) cms
    (coinOf (resp 0)) h0 he hi 1
  exact ⟨bs, by rw [hb]⟩

/-- **The pure masked session's third body** is table `0` rep `0`'s outer zerocheck point squeeze, untagged: the
header of `m − 13` coins of kind `2` alone, where the schedules, the level-0 commitments, that table's extra claims
at the link points and its pads' commitment return. -/
theorem sessionPZk_go_round2 (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
    (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic) :
    2 < ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2.size ∧
      ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2[2]! =
        Claims.toMsg (Transcript.framed ⟨#[], 0, ByteArray.empty⟩ 2 (tabs[0]!.m - 13)) := by
  obtain ⟨bs, h1, h2⟩ := sessionPZk_sim H resp spec tabs words trs tape #[]
  obtain ⟨bs', hb⟩ := sessionPZk_pv_round2 H resp spec tabs hm words trs tape hs hc h0 he hi
  simp only [Array.size_empty] at h1
  rw [h1] at hb
  rw [h2, Array.empty_append]
  rw [Replay.framed_afterSq] at hb
  rcases bs with _ | ⟨a, _ | ⟨b, _ | ⟨c, bs⟩⟩⟩
  · simp at hb
  · simp at hb
  · simp at hb
  · simp only [List.map_cons, List.cons_append, List.nil_append, List.cons.injEq] at hb
    refine ⟨by simp, ?_⟩
    simp [hb.2.2.1]

end FlockComplete
