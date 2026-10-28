import Proofs.Flock.Claims.Complete.ZkTables.Round2

/-!
# The masked prover's zerocheck bodies in the session

The masked zerocheck's sent bodies, in order (`Replay.ZcZ.sends`): its two squeezes' frames (`rSkip`, `rOuter`), the
frame of `z`'s squeeze after the padded `round1_ab` and `round1_c`, then each residual round's frame after its padded
message. Table `0`'s rep `0` zerocheck is the session's first after the link message, so its `k`-th body is the
session's `1 + k`-th, where that table's extra claims at the link points return.

* `Replay.zerocheckPZk_go_sends`: `Replay.zerocheckPZk_go` with its sends given, `ZcZ.sends`.
* `zerocheckPZk_pv_sends`, `repPZk_pv_sends`, `afterLinkPZk_pv_sends`, `sessionPZk_pv_sends`: the same bodies, tagged
  `0`, first in the zerocheck's, the rep's, the session's after the link's coin, and the session's after the link
  message.
* `sessionPZk_go_sends`: the pure masked session's body `1 + k` is the zerocheck's `k`-th, for every `k` it sends;
  `sessionPZk_go_round3`: so its fourth is `z`'s squeeze frame after the padded `round1_ab`, `round1_c`.
-/

namespace FlockComplete.Replay

open Flock Transcript

namespace ZcZ

/-- **The masked zerocheck's sent bodies**, untagged: its two squeezes' frames (`rSkip`, `rOuter`), `z`'s squeeze frame
after the padded `round1_ab` and `round1_c` (`t₄`), then residual round `i`'s after its padded message. -/
def sends (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (digest cap root : ByteArray) (cs : Zc.Challenges) (tr : Trace)
    (t : Transcript) : List Claims.Msg :=
  Claims.toMsg (framed (t₁ digest cap root t) 2 6) :: Claims.toMsg (framed (t₂ digest cap root cs t) 2 (m - 13)) ::
    Claims.toMsg (framed (t₄ lay h m digest cap root cs tr t) 1 1) ::
      (List.range (m - 6)).map fun i => Claims.toMsg (framed
        ⟨(loopT lay h m cs tr (t₅ lay h m digest cap root cs tr t) i).rounds,
          (loopT lay h m cs tr (t₅ lay h m digest cap root cs tr t) i).k,
          pF (msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ i).2 (pF (msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ i).1
            (loopT lay h m cs tr (t₅ lay h m digest cap root cs tr t) i).pending)⟩ 1 1)

theorem sends_length (lay : Zk.PadLayout) (h : Array F128) (m : ℕ) (digest cap root : ByteArray) (cs : Zc.Challenges)
    (tr : Trace) (t : Transcript) : (sends lay h m digest cap root cs tr t).length = 3 + (m - 6) := by
  simp [sends]
  omega

end ZcZ

theorem push3_append {α : Type} (l : Array α) (a b c : α) (xs : List α) :
    ((l.push a).push b).push c ++ xs.toArray = l ++ (a :: b :: c :: xs).toArray := by
  rw [List.push_append_toArray, List.push_append_toArray, List.push_append_toArray]

/-- **The masked prover's zerocheck, in closed form, with its sends**: `zerocheckPZk_go`, its sent bodies `ZcZ.sends`. -/
theorem zerocheckPZk_go_sends (resp : ℕ → Claims.Msg) {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray)
    (lay : Zk.PadLayout) (h : Array F128) (tr : Trace) (t : Transcript) (l : Array Claims.Msg) :
    ((zerocheckPZk (asksS resp) m digest cap root lay h tr).run t).go l =
      (.ok ((ZcZ.proof lay h m (Zc.coinsAt resp l.size) tr, Zc.out m (Zc.coinsAt resp l.size) tr),
        ZcZ.tEnd lay h m digest cap root (Zc.coinsAt resp l.size) tr t),
        l ++ (ZcZ.sends lay h m digest cap root (Zc.coinsAt resp l.size) tr t).toArray) := by
  unfold zerocheckPZk
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) _)]
  rw [bind_go (sampleVecP_go resp (by decide) _ _)]
  rw [bind_go (sampleVecP_go resp (by omega) _ _)]
  dsimp only
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbSlice_eq _ _) _)]
  rw [bind_go (sampleP_go resp _ _)]
  dsimp only
  simp only [Array.size_push]
  let cs := Zc.coinsAt resp l.size
  let T := ZcZ.loopT lay h m cs tr (ZcZ.t₅ lay h m digest cap root cs tr t)
  let g := ZcZ.msg lay h tr (m - 6) (cs.r m) cs.z cs.ρ
  refine Eq.trans (bind_go (forIn_go _
    (fun i => (Zc.tab (m - 6) cs.z cs.ρ tr.az i, Zc.tab (m - 6) cs.z cs.ρ tr.bz i, pushes g i, pushes cs.ρ i)) T
    (fun i => Claims.toMsg (framed ⟨(T i).rounds, (T i).k, pF (g i).2 (pF (g i).1 (T i).pending)⟩ 1 1))
    (l.size + 3) ?hbody (m - 6) _ (by simp))) ?tail
  case tail =>
    dsimp only
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [pure_go]
    exact Prod.ext rfl (push3_append _ _ _ _ _)
  case hbody =>
    intro i l₁ hl₁
    dsimp only
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) _)]
    rw [bind_go (sampleP_go resp _ _), hl₁]
    rfl

end FlockComplete.Replay

namespace FlockComplete

open Flock FlockDeploy

variable (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg)

/-- **The masked zerocheck's bodies in `PV`** are its sends, tagged `0`. -/
theorem zerocheckPZk_pv_sends {m : ℕ} (hm : 14 ≤ m) (digest cap root : ByteArray) (lay : Zk.PadLayout)
    (h : Array F128) (tr : Trace) (t : Transcript) (i : ℕ) :
    (((zerocheckPZk asks m digest cap root lay h tr).run t).run.sim H resp i).1 =
      (Replay.ZcZ.sends lay h m digest cap root (Replay.Zc.coinsAt resp i) tr t).map (List.cons 0) := by
  obtain ⟨bs, h1, h2⟩ := (simS H resp).run (zerocheckPZk_sim H resp m digest cap root lay h tr) t
    (Array.replicate i [])
  rw [Replay.zerocheckPZk_go_sends resp hm digest cap root lay h tr t] at h2
  simp only [Array.size_replicate] at h1 h2
  have hb : bs = Replay.ZcZ.sends lay h m digest cap root (Replay.Zc.coinsAt resp i) tr t :=
    List.toArray_inj ((Array.append_right_inj _).1 h2.symm)
  rw [h1, hb]

theorem repPZk_pv_sends (st : Setup) (hm : 14 ≤ st.m) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (T : Transcript) (i : ℕ) :
    ∃ bs, (((repPZk asks Hashes.pi st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior
      words).run T).run.sim H resp i).1 =
      (Replay.ZcZ.sends (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h st.m st.digest (flat cm0.cap) ic.root
        (Replay.Zc.coinsAt resp i) tr T).map (List.cons 0) ++ bs := by
  unfold repPZk
  exact st_sim_pre H resp ⟨[], by
    rw [List.append_nil]
    exact zerocheckPZk_pv_sends H resp hm st.digest (flat cm0.cap) ic.root
      (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr T i⟩

/-- **The session's bodies after the link's coin start with table `0` rep `0`'s zerocheck's**, where that rep is the
first stream and its extra claims and pads' commitment return. -/
theorem afterLinkPZk_pv_sends (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
    (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic)
    (i : ℕ) :
    ∃ bs, ((afterLinkPZk asks Hashes.pi spec tabs words trs tape schs tbls cms c1).run.sim H resp i).1 =
      (Replay.ZcZ.sends (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)
        (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)).h tabs[0]!.m tabs[0]!.digest
        (flat cms[0]!.cap) ic.root (Replay.Zc.coinsAt resp i) trs[0]!
        (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)).map (List.cons 0) ++ bs := by
  unfold afterLinkPZk
  dsimp only
  rw [← Array.forIn_toList, h0, List.forIn_cons]
  refine pv_sim_pre H resp ?_
  refine pv_sim_pre H resp ?_
  rw [pv_sim_quiet H resp ((monadLift_pv H resp _ i).trans (congrArg _ he))]
  rw [pv_sim_quiet H resp ((innerCommitP_pv H resp _ _ i).trans (congrArg _ hi))]
  exact pv_sim_pre H resp (repPZk_pv_sends H resp tabs[0]! hm schs[0]!.1 schs[0]!.2 0
    (spec.domain spec.tables[0]! 0).toUTF8 trs[0]! tbls[0]! cms[0]! extras
    (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) ic
    (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0) (tape.nonce 0 0) (priorOf tabs[0]! #[] 0) words[0]!
    (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8) i)

/-- **The masked session's bodies** in `PV`: the link message of the level-0 caps, then table `0` rep `0`'s zerocheck's
sends. -/
theorem sessionPZk_pv_sends (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
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
        ((Replay.ZcZ.sends (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)
          (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)).h tabs[0]!.m tabs[0]!.digest
          (flat cms[0]!.cap) ic.root (Replay.Zc.coinsAt resp 1) trs[0]!
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)).map (List.cons 0) ++ bs) := by
  rw [sessionPZk_eq]
  rw [pv_sim_quiet H resp ((schedsPZk_pv H resp tabs 0).trans (congrArg _ hs))]
  rw [pv_sim_quiet H resp ((commitsPZk_pv H resp tabs schs _ tape 0).trans (congrArg _ hc))]
  rw [link_pv_sim]
  obtain ⟨bs, hb⟩ := afterLinkPZk_pv_sends H resp spec tabs hm words trs tape schs (tablesP tabs trs) cms
    (coinOf (resp 0)) h0 he hi 1
  exact ⟨bs, by rw [hb]⟩

/-- **The pure masked session's body `1 + k` is table `0` rep `0`'s zerocheck's `k`-th send**, for every `k` it sends,
where the schedules, the level-0 commitments, that table's extra claims at the link points and its pads' commitment
return. -/
theorem sessionPZk_go_sends (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
    (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic)
    {k : ℕ} (hk : k < 3 + (tabs[0]!.m - 6)) :
    1 + k < ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2.size ∧
      ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2[1 + k]! =
        (Replay.ZcZ.sends (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)
          (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)).h tabs[0]!.m tabs[0]!.digest
          (flat cms[0]!.cap) ic.root (Replay.Zc.coinsAt resp 1) trs[0]!
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8))[k]! := by
  obtain ⟨bs, h1, h2⟩ := sessionPZk_sim H resp spec tabs words trs tape #[]
  obtain ⟨bs', hb⟩ := sessionPZk_pv_sends H resp spec tabs hm words trs tape hs hc h0 he hi
  simp only [Array.size_empty] at h1
  rw [h1] at hb
  rw [h2, Array.empty_append]
  have hlen := Replay.ZcZ.sends_length (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)
    (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)).h tabs[0]!.m tabs[0]!.digest
    (flat cms[0]!.cap) ic.root (Replay.Zc.coinsAt resp 1) trs[0]!
    (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)
  generalize Replay.ZcZ.sends (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)
    (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)).h tabs[0]!.m tabs[0]!.digest
    (flat cms[0]!.cap) ic.root (Replay.Zc.coinsAt resp 1) trs[0]!
    (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8) = S at hb hlen ⊢
  dsimp only at hb
  obtain ⟨a, l₁, rfl, -, hl₁⟩ := List.map_eq_cons_iff.1 hb
  obtain ⟨L₁, L₂, rfl, hL₁, -⟩ := List.map_eq_append_iff.1 hl₁
  have hL : L₁ = S := (List.map_inj_right (f := List.cons (0 : UInt8)) (fun _ _ h => (List.cons.inj h).2)).1 hL₁
  rw [← hL] at hlen ⊢
  have hkL : k < L₁.length := by omega
  refine ⟨by simp only [List.size_toArray, List.length_cons, List.length_append]; omega, ?_⟩
  rw [List.getElem!_toArray, Nat.add_comm 1 k, List.getElem!_eq_getElem?_getD, List.getElem!_eq_getElem?_getD,
    List.getElem?_cons_succ, List.getElem?_append_left hkL]

/-- **The pure masked session's fourth body is `z`'s squeeze frame after the padded `round1_ab` and `round1_c`** of
table `0` rep `0`'s zerocheck (`Replay.ZcZ.t₄`), at the coins the replies from body `1` on give. -/
theorem sessionPZk_go_round3 (spec : SessionSpec) (tabs : Array Setup) (hm : 14 ≤ tabs[0]!.m)
    (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic) :
    3 < ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2.size ∧
      ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2[3]! =
        Claims.toMsg (Transcript.framed (Replay.ZcZ.t₄ (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)
          (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)).h tabs[0]!.m tabs[0]!.digest
          (flat cms[0]!.cap) ic.root (Replay.Zc.coinsAt resp 1) trs[0]!
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 1 1) := by
  obtain ⟨h1, h2⟩ := sessionPZk_go_sends H resp spec tabs hm words trs tape hs hc h0 he hi (k := 2) (by omega)
  exact ⟨h1, h2.trans (List.getElem!_cons_succ.trans (List.getElem!_cons_succ.trans List.getElem!_cons_zero))⟩

end FlockComplete
