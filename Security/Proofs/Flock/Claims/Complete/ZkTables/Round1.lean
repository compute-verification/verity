import Proofs.Flock.Claims.Complete.ZkTables.Round0
import Proofs.Flock.Claims.Complete.Zk.Sim
import Proofs.Flock.Claims.Complete.Zk.PiopGo
import Proofs.Flock.Claims.Complete.Zk.Count

/-!
# The masked prover's second message: table `0` rep `0`'s first zerocheck coin

Round `1` of the masked session on the executable's side. After the link message's coin the session opens table `0`'s
rep `0`: its extra claims at the link points, its pads' commitment (quiet), then the zerocheck's first squeeze. Its
second body is that squeeze's frame, of a transcript the coin does not reach: the domain's, with the circuit digest,
the level-0 cap and the pads' root absorbed (`Replay.ZcZ.t₁`). Where the extra claims throw, the session stops after
the link message.

* `pv_sim_bind`, `pv_sim_quiet`, `pv_sim_head`, `pv_sim_err` (and `st_sim_quiet`, `st_sim_head` under `StateT`): the
  bodies of a bind in the verdict-and-oracle monad `PV`, the first part's then the rest's.
* `pv_sim_of_quiet`: a `PV` program related (`simS`) to a quiet pure one sends nothing and returns its verdict; so
  the schedules, the level-0 commitments and the pads' commitment (`schedsPZk_pv`, `commitsPZk_pv`, `innerCommitP_pv`).
* `zerocheckPZk_pv_head`, `repPZk_pv_head`, `afterLinkPZk_pv_head`: the first body of a rep, and of the session after
  the link's coin where rep `0` of table `0` is the first stream and its extra claims return.
* `sessionPZk_pv_round1`, `sessionPZk_go_round1`: the session's first two bodies, the link message and that squeeze,
  in `PV` and at the pure session.
* `afterLinkPZk_pv_err`, `sessionPZk_pv_round1_err`, `sessionPZk_go_round1_err`: where table `0`'s extra claims throw,
  the session stops having sent the link message alone.
-/

open Flock FlockDeploy

namespace FlockComplete

variable (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg)

/-- **A bind's bodies in `PV`**: the first part's, then, where it returns, the rest's from the next index. -/
theorem pv_sim_bind {α β : Type} (x : PV α) (f : α → PV β) (i : ℕ) :
    (x >>= f).run.sim H resp i =
      ((x.run.sim H resp i).1 ++ (match (x.run.sim H resp i).2 with
        | .ok a => ((f a).run.sim H resp (i + (x.run.sim H resp i).1.length)).1
        | .error _ => []), match (x.run.sim H resp i).2 with
        | .ok a => ((f a).run.sim H resp (i + (x.run.sim H resp i).1.length)).2
        | .error e => .error e) := by
  rw [ExceptT.run_bind]
  change (PI.bind x.run _).sim H resp i = _
  rw [PI.sim_bind]
  generalize x.run.sim H resp i = p
  rcases p with ⟨bs, r⟩
  cases r with
  | error e => exact Prod.ext (by simp; rfl) rfl
  | ok a => rfl

theorem pv_sim_quiet {α β : Type} {x : PV α} {f : α → PV β} {i : ℕ} {a : α}
    (hx : x.run.sim H resp i = ([], .ok a)) : (x >>= f).run.sim H resp i = (f a).run.sim H resp i := by
  rw [pv_sim_bind, hx]
  simp

theorem pv_sim_head {α β : Type} {x : PV α} {f : α → PV β} {i : ℕ} {m : Claims.Msg}
    (hx : ∃ bs, (x.run.sim H resp i).1 = m :: bs) : ∃ bs', ((x >>= f).run.sim H resp i).1 = m :: bs' := by
  obtain ⟨bs, hx⟩ := hx
  rw [pv_sim_bind, hx]
  exact ⟨_, rfl⟩

theorem st_sim_quiet {σ α β : Type} {x : StateT σ PV α} {f : α → StateT σ PV β} {s s' : σ} {i : ℕ} {a : α}
    (hx : (x.run s).run.sim H resp i = ([], .ok (a, s'))) :
    ((x >>= f).run s).run.sim H resp i = ((f a).run s').run.sim H resp i := by
  rw [StateT.run_bind]
  exact pv_sim_quiet H resp hx

theorem st_sim_head {σ α β : Type} {x : StateT σ PV α} {f : α → StateT σ PV β} {s : σ} {i : ℕ} {m : Claims.Msg}
    (hx : ∃ bs, ((x.run s).run.sim H resp i).1 = m :: bs) :
    ∃ bs', (((x >>= f).run s).run.sim H resp i).1 = m :: bs' := by
  rw [StateT.run_bind]
  exact pv_sim_head H resp hx

/-- **A `PV` program related to a quiet pure one sends nothing** and returns the pure one's verdict. -/
theorem pv_sim_of_quiet {α : Type} {x : PV α} {y : PS α} {v : V α} (hR : (simS H resp).R x y)
    (hq : ∀ l, y.go l = (v, l)) (i : ℕ) : x.run.sim H resp i = ([], v) := by
  obtain ⟨bs, h1, h2⟩ := hR (Array.replicate i [])
  rw [hq] at h1 h2
  simp only [Array.size_replicate] at h1
  have : bs = [] := by
    have := congrArg Array.size h2
    simp at this
    exact this
  subst this
  rw [h1]
  rfl

end FlockComplete

namespace FlockComplete

variable (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg)

theorem liftTM_pv {α : Type} (x : Transcript.TM α) (t : Transcript) (i : ℕ) :
    ((liftTM x : TMW PV α).run t).run.sim H resp i = ([], x t) := by
  show (PI.ret (x t)).sim H resp i = _
  rfl

theorem squeezeP_pv_head (kind : UInt8) {n : ℕ} (hn : n ≠ 0) (t : Transcript) (i : ℕ) :
    ∃ bs, (((squeezeP asks kind n).run t).run.sim H resp i).1 = (0 :: Claims.toMsg (Transcript.framed t kind n)) :: bs := by
  simp only [squeezeP, StateT.run, hn, ite_false]
  exact ⟨_, rfl⟩

/-- **The masked zerocheck's first body** is its first squeeze's frame, of the transcript with the circuit's digest,
the level-0 cap and the pads' root absorbed (`Replay.ZcZ.t₁`). -/
theorem zerocheckPZk_pv_head (m : ℕ) (digest cap root : ByteArray) (lay : Zk.PadLayout) (h : Array F128) (tr : Trace)
    (t : Transcript) (i : ℕ) :
    ∃ bs, (((zerocheckPZk asks m digest cap root lay h tr).run t).run.sim H resp i).1 =
      (0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ digest cap root t) 2 6)) :: bs := by
  unfold zerocheckPZk
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.label_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.absorbBytes_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.absorbBytes_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.label_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.label_eq _ _)))]
  rw [st_sim_quiet H resp ((liftTM_pv H resp _ _ _).trans (congrArg _ (Flock.RecAlgebra.absorbBytes_eq _ _)))]
  exact st_sim_head H resp (squeezeP_pv_head H resp 2 (n := 6) (by decide) _ i)

end FlockComplete

namespace FlockComplete

variable (H : Claims.Msg → Claims.Msg) (resp : ℕ → Claims.Msg)

theorem repPZk_pv_head (st : Setup) (sch : Schedule) (tPad rep : ℕ) (dom : ByteArray) (tr : Trace)
    (t : Array F128) (cm0 : Commit) (extras : Array Claim) (pads : PadsZk) (ic : InnerCommit)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64)
    (prior : Array (List F128)) (words : List ℕ) (T : Transcript) (i : ℕ) :
    ∃ bs, (((repPZk asks Hashes.pi st sch tPad rep dom tr t cm0 extras pads ic extra lanePad salt nonce prior
      words).run T).run.sim H resp i).1 =
      (0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ st.digest (flat cm0.cap) ic.root T) 2 6)) :: bs := by
  unfold repPZk
  exact st_sim_head H resp (zerocheckPZk_pv_head H resp st.m st.digest (flat cm0.cap) ic.root
    (Zk.PadLayout.new st.m st.kLog sch.k0) pads.h tr T i)

/-- The pads' commitment in `PV` sends nothing and returns its verdict at `hOf H`. -/
theorem innerCommitP_pv (p : PadsZk) (salt : ℕ → ByteArray) (i : ℕ) :
    (innerCommitP Hashes.pi p salt).run.sim H resp i =
      ([], (innerCommitP (Hashes.ofSha512 (Claims.hOf H)) p salt : V InnerCommit)) :=
  pv_sim_of_quiet H resp (innerCommitP_sim H resp p salt) (innerCommitP_quiet (Claims.hOf H) p salt) i

theorem monadLift_pv {α : Type} (v : V α) (i : ℕ) : (monadLift v : PV α).run.sim H resp i = ([], v) := rfl

/-- **The session's first body after the link's coin**, where table `0`'s rep `0` is the first stream and its extra
claims and pads' commitment return: that rep's first zerocheck squeeze, from the domain's transcript. -/
theorem afterLinkPZk_pv_head (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic)
    (i : ℕ) :
    ∃ bs, ((afterLinkPZk asks Hashes.pi spec tabs words trs tape schs tbls cms c1).run.sim H resp i).1 =
      (0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap) ic.root
        (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6)) :: bs := by
  unfold afterLinkPZk
  dsimp only
  rw [← Array.forIn_toList, h0, List.forIn_cons]
  refine pv_sim_head H resp ?_
  refine pv_sim_head H resp ?_
  rw [pv_sim_quiet H resp ((monadLift_pv H resp _ i).trans (congrArg _ he))]
  rw [pv_sim_quiet H resp ((innerCommitP_pv H resp _ _ i).trans (congrArg _ hi))]
  exact pv_sim_head H resp (repPZk_pv_head H resp tabs[0]! schs[0]!.1 schs[0]!.2 0
    (spec.domain spec.tables[0]! 0).toUTF8 trs[0]! tbls[0]! cms[0]! extras
    (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) ic
    (tape.extra 0) (tape.lanePad 0) (tape.salt 0 0) (tape.nonce 0 0) (priorOf tabs[0]! #[] 0) words[0]!
    (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8) i)

theorem schedsPZk_pv (tabs : Array Setup) (i : ℕ) :
    (schedsPZk tabs : PV _).run.sim H resp i = ([], (schedsPZk tabs : V _)) :=
  pv_sim_of_quiet H resp (by unfold schedsPZk; sim_walk) (schedsPZk_quiet tabs) i

theorem commitsPZk_pv (tabs : Array Setup) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (tape : TapeZk)
    (i : ℕ) :
    (commitsPZk Hashes.pi tabs schs tbls tape : PV _).run.sim H resp i =
      ([], (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs tbls tape : V _)) :=
  pv_sim_of_quiet H resp (by unfold commitsPZk; sim_walk [commitPZk_sim H resp])
    (commitsPZk_quiet (Claims.hOf H) tabs schs tbls tape) i

theorem link_pv_sim {β : Type} (rb pub : Array ByteArray) (f : (ℕ → F128) → PV β) (i : ℕ) :
    ((asks.link rb pub >>= f).run.sim H resp i).1 =
      (0 :: linkBody rb.toList pub.toList) :: ((f (coinOf (resp i))).run.sim H resp (i + 1)).1 := by
  rw [pv_sim_bind]
  rfl

/-- **The masked session's first two bodies** in `PV`: the link message of the level-0 caps, then table `0` rep `0`'s
first zerocheck squeeze. -/
theorem sessionPZk_pv_round1 (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
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
        (0 :: Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap) ic.root
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6)) :: bs := by
  rw [sessionPZk_eq]
  rw [pv_sim_quiet H resp ((schedsPZk_pv H resp tabs 0).trans (congrArg _ hs))]
  rw [pv_sim_quiet H resp ((commitsPZk_pv H resp tabs schs _ tape 0).trans (congrArg _ hc))]
  rw [link_pv_sim]
  obtain ⟨bs, hb⟩ := afterLinkPZk_pv_head H resp spec tabs words trs tape schs (tablesP tabs trs) cms
    (coinOf (resp 0)) h0 he hi 1
  exact ⟨bs, by rw [hb]⟩

theorem pv_sim_err {α β : Type} {x : PV α} {f : α → PV β} {i : ℕ} {e : String}
    (hx : x.run.sim H resp i = ([], .error e)) : (x >>= f).run.sim H resp i = ([], .error e) := by
  rw [pv_sim_bind, hx]
  rfl

theorem link_pv_sim' {β : Type} (rb pub : Array ByteArray) (f : (ℕ → F128) → PV β) (i : ℕ) :
    (asks.link rb pub >>= f).run.sim H resp i =
      ((0 :: linkBody rb.toList pub.toList) :: ((f (coinOf (resp i))).run.sim H resp (i + 1)).1,
        ((f (coinOf (resp i))).run.sim H resp (i + 1)).2) := by
  rw [pv_sim_bind]
  rfl

/-- **Where table `0`'s extra claims throw**, the session after the link's coin sends nothing and throws. -/
theorem afterLinkPZk_pv_err (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) (schs : Array (Schedule × ℕ)) (tbls : Array (Array F128)) (cms : Array Commit) (c1 : ℕ → F128)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {e : String} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)) = .error e) (i : ℕ) :
    (afterLinkPZk asks Hashes.pi spec tabs words trs tape schs tbls cms c1).run.sim H resp i = ([], .error e) := by
  unfold afterLinkPZk
  dsimp only
  rw [← Array.forIn_toList, h0, List.forIn_cons]
  refine pv_sim_err H resp ?_
  refine pv_sim_err H resp ?_
  exact pv_sim_err H resp ((monadLift_pv H resp _ i).trans (congrArg _ he))

theorem sessionPZk_pv_round1_err (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ))
    (trs : Array Trace) (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {e : String} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .error e) :
    (sessionPZk asks Hashes.pi spec tabs words trs tape).run.sim H resp 0 =
      ([0 :: linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList], .error e) := by
  rw [sessionPZk_eq]
  rw [pv_sim_quiet H resp ((schedsPZk_pv H resp tabs 0).trans (congrArg _ hs))]
  rw [pv_sim_quiet H resp ((commitsPZk_pv H resp tabs schs _ tape 0).trans (congrArg _ hc))]
  rw [link_pv_sim', afterLinkPZk_pv_err H resp spec tabs words trs tape schs (tablesP tabs trs) cms
    (coinOf (resp 0)) h0 he]

/-- **The pure masked session's second body** is table `0` rep `0`'s first zerocheck squeeze, untagged, where the
schedules, the level-0 commitments, that table's extra claims at the link points and its pads' commitment return. -/
theorem sessionPZk_go_round1 (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ)) (trs : Array Trace)
    (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {extras : Array Claim} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .ok extras)
    {ic : InnerCommit} (hi : (innerCommitP (Hashes.ofSha512 (Claims.hOf H))
      (padsOf tape 0 0 (Zk.PadLayout.new tabs[0]!.m tabs[0]!.kLog schs[0]!.1.k0)) (tape.innerSalt 0 0) : V _) = .ok ic) :
    1 < ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2.size ∧
      ((sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[]).2[1]! =
        Claims.toMsg (Transcript.framed (Replay.ZcZ.t₁ tabs[0]!.digest (flat cms[0]!.cap) ic.root
          (Transcript.init #[] (spec.domain spec.tables[0]! 0).toUTF8)) 2 6) := by
  obtain ⟨bs, h1, h2⟩ := sessionPZk_sim H resp spec tabs words trs tape #[]
  obtain ⟨bs', hb⟩ := sessionPZk_pv_round1 H resp spec tabs words trs tape hs hc h0 he hi
  simp only [Array.size_empty] at h1
  rw [h1] at hb
  rw [h2, Array.empty_append]
  rcases bs with _ | ⟨a, _ | ⟨b, bs⟩⟩
  · simp at hb
  · simp at hb
  · simp only [List.map_cons, List.cons.injEq] at hb
    refine ⟨by simp, ?_⟩
    simp [hb.2.1]

/-- **Where table `0`'s extra claims throw, the pure masked session stops after the link message**: it throws
having sent that message alone. -/
theorem sessionPZk_go_round1_err (spec : SessionSpec) (tabs : Array Setup) (words : Array (List ℕ))
    (trs : Array Trace) (tape : TapeZk) {schs : Array (Schedule × ℕ)} {cms : Array Commit}
    (hs : (schedsPZk tabs : V _) = .ok schs)
    (hc : (commitsPZk (Hashes.ofSha512 (Claims.hOf H)) tabs schs (tablesP tabs trs) tape : V _) = .ok cms)
    {n : String} {rest : List (ℕ × ℕ × String)} (h0 : spec.repStreams.toList = (0, 0, n) :: rest)
    {e : String} (he : tabs[0]!.extra ((Array.range spec.points).map fun p =>
      (Array.range spec.mPts).map fun j => coinOf (resp 0) (p * spec.mPts + j)) = .error e) :
    (sessionPZk (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) spec tabs words trs tape).go #[] =
      (.error e, #[linkBody (cms.map fun c => flat c.cap).toList (tabs.map fun _ => ByteArray.empty).toList]) := by
  obtain ⟨bs, h1, h2⟩ := sessionPZk_sim H resp spec tabs words trs tape #[]
  have hb := sessionPZk_pv_round1_err H resp spec tabs words trs tape hs hc h0 he
  simp only [Array.size_empty] at h1
  rw [h1] at hb
  obtain ⟨hb1, hb2⟩ := Prod.mk.inj hb
  rcases bs with _ | ⟨a, _ | ⟨b, bs⟩⟩
  · simp at hb1
  · simp only [List.map_cons, List.map_nil, List.cons.injEq, and_true, true_and] at hb1
    refine Prod.ext hb2 ?_
    rw [h2, hb1]
    rfl
  · simp at hb1

end FlockComplete
