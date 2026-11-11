module

public import Proofs.Flock.ZkAny.RelayLoopJ
public import Proofs.Flock.ZkAny.RelayLoopRound
public import Proofs.Flock.ZkAny.RelayReps

/-!
# The relay's invariant kept at a frame round

At a frame round's slot the round is the model's next, on its rep stream's rep, and every other seen rep stream's
frame is kept; so `LoopJ` is kept there given `RoundFacts` and `RepJ` of the round's own rep one round on:

* `rep_at`, `rep_index`: rep stream `2 * i + r` is table `i`'s rep `r`. `rep_eq`: two rep streams at one index are
  one. `rep_of_name`: where the rep streams' names are distinct, two of one name are at one index; `tables_inj`: and
  two tables of one name are one.
* `locate_fst_succ`: the next round's stream is at or after this round's.
* **`round_rep`**: a frame round's rep is the model's next round's rep stream, among the setup's. `repJ_cur`: past
  its stream's first round the rep has `RepJ`; `freshAt_cur`: at it, the rep's frame is fresh.
* `seenAt_round`: a round of the model's next round's stream keeps `SeenAt` one round on.
* `streamAt_old`, `repJ_old`: a rep stream before the next round's has `StreamAt` vacuously one round on, so its
  `RepJ` is kept by updates that keep its frame and what the session's rounds read.
* `LoopModel.hist_succ`: the history one message on.
* `opens_snd`, `frameAnswer_snd`, `tookEarlier_snd`, **`answered_snd`**: the coins' answer reads the history only
  through its answers, where the frame's states and the transport's coins do; so at the loop's own frames
  (`ofLoop_answered_snd`). `frameAnswerR_snd`, `answeredR_snd`, `ofLoop_answeredR_snd`: so does its answer before
  the rank stop, and `SameInv` with it (`sameInv_hist`, `sameInv_snd`), and the coin step (`ofLoop_fed_snd`).
* `stopped_roundRun`: the session's stop one frame round on.
* **`loopKept_round`**: `LoopJ` is kept at a frame round, given `RoundFacts` there and the round's rep's `RepJ` at the
  session's next request with the model's coin step taking the round's coins.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock.Firewall.Session (State Placed byte? reqAt)
open _root_.Flock (SessionSpec Stmt)
open _root_.Flock.Firewall.Frames (Sessions Rep)

/-- **Rep stream `m` is table `i`'s rep `r` where `m = 2 * i + r`.** -/
theorem rep_at {spec : SessionSpec} {m i r : Nat} {s : String} (hme : spec.repStreams[m]? = some (i, r, s)) :
    2 * i + r = m := by
  have hm : m < spec.repStreams.size := by
    by_contra hc
    rw [Array.getElem?_eq_none (by omega)] at hme
    cases hme
  have hT : m / 2 < spec.tables.size := by
    rw [Claims.Bridge.Custody.repStreams_size] at hm
    omega
  obtain ⟨hlt, hg⟩ := Claims.Bridge.Custody.repStreams_getElem spec hT (Nat.mod_lt m (by decide))
  have g := Array.getElem?_eq_getElem hlt
  rw [hg, Nat.div_add_mod, hme, Option.some.injEq, Prod.mk.injEq, Prod.mk.injEq] at g
  obtain ⟨rfl, rfl, -⟩ := g
  exact Nat.div_add_mod m 2

/-- **Table `i`'s rep `r` is rep stream `2 * i + r`.** -/
theorem rep_index {spec : SessionSpec} {i r : Nat} {s : String} (h : (i, r, s) ∈ spec.repStreams.toList) :
    spec.repStreams[2 * i + r]? = some (i, r, s) := by
  obtain ⟨m, hme⟩ := List.mem_iff_getElem?.mp h
  rw [Array.getElem?_toList] at hme
  rw [rep_at hme]
  exact hme

/-- Two rep streams at one index are one. -/
theorem rep_eq {spec : SessionSpec} {i r i' r' : Nat} {s s' : String} (h : (i, r, s) ∈ spec.repStreams.toList)
    (h' : (i', r', s') ∈ spec.repStreams.toList) (he : 2 * i + r = 2 * i' + r') : (i, r, s) = (i', r', s') := by
  have a := rep_index h
  rw [he, rep_index h'] at a
  exact (Option.some.inj a).symm

/-- **Where the rep streams' names are distinct, two rep streams of one name are at one index.** -/
theorem rep_of_name {spec : SessionSpec} (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) {i r i' r' : Nat}
    {s : String} (h : (i, r, s) ∈ spec.repStreams.toList) (h' : (i', r', s) ∈ spec.repStreams.toList) :
    2 * i + r = 2 * i' + r' := by
  have g : ∀ {i r : Nat}, (i, r, s) ∈ spec.repStreams.toList →
      ∃ hl : 2 * i + r < (spec.repStreams.toList.map (·.2.2)).length,
        (spec.repStreams.toList.map (·.2.2))[2 * i + r] = s := by
    intro i r hm
    apply List.getElem?_eq_some_iff.mp
    simp only [List.getElem?_map, Array.getElem?_toList, rep_index hm, Option.map_some]
  obtain ⟨hl, e⟩ := g h
  obtain ⟨hl', e'⟩ := g h'
  have a := List.Nodup.idxOf_getElem hnd _ hl
  have b := List.Nodup.idxOf_getElem hnd _ hl'
  rw [e] at a
  rw [e'] at b
  exact a.symm.trans b

/-- **Where the rep streams' names are distinct, two tables of one name are one**: each table's rep 0 stream is named
after it. -/
theorem tables_inj {spec : SessionSpec} (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) {i j : Nat}
    {name : String} (hi : spec.tables[i]? = some name) (hj : spec.tables[j]? = some name) : i = j := by
  have mem : ∀ k, spec.tables[k]? = some name →
      (k, 0, s!"{name}/rep{(0 : Nat)}") ∈ spec.repStreams.toList := by
    intro k hk
    obtain ⟨hlt, he⟩ := Array.getElem?_eq_some_iff.mp hk
    rw [Array.mem_toList_iff, _root_.Flock.SessionSpec.repStreams, Array.mem_flatMap]
    refine ⟨k, Array.mem_range.mpr hlt, Array.mem_map.mpr ⟨0, by simp, ?_⟩⟩
    rw [getElem!_pos spec.tables k hlt, he]
  have := rep_of_name hnd (mem i hi) (mem j hj)
  omega

/-- **The next round's stream is at or after this round's.** -/
theorem locate_fst_succ : ∀ (rs : List Nat) (k i : Nat),
    (ZKMal.CoinLayer.locate rs k i).1 ≤ (ZKMal.CoinLayer.locate rs k (i + 1)).1
  | [], _, _ => Nat.le_refl _
  | c :: cs, k, i => by
    by_cases h1 : i + 1 < c
    · rw [Loop.locate_cons_lt h1, Loop.locate_cons_lt (Nat.lt_of_succ_lt h1)]
    · rw [Loop.locate_cons_ge h1]
      by_cases h0 : i < c
      · rw [Loop.locate_cons_lt h0]
        exact Nat.le_of_succ_le (Loop.locate_fst_ge cs (k + 1) _)
      · rw [Loop.locate_cons_ge h0, show i + 1 - c = i - c + 1 by omega]
        exact locate_fst_succ cs (k + 1) (i - c)

/-- **A frame round's rep is the model's next round's rep stream**, among the setup's: at the slot of a round of
`stream` that the relay steps through table `i`'s rep `r`, where the session's plan is the setup's and its rep
streams' names are distinct, rep stream `2 * i + r` is the stream of the model's next round at its history so far. -/
theorem round_rep {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) :
    2 * i + r = (nextAt spec sts (M.hist σ.c)).1 ∧ (i, r, stream) ∈ spec.repStreams.toList := by
  obtain ⟨hsent, hop, -, hq⟩ := slot_round hsl
  have hmem : (i, r, stream) ∈ spec.repStreams.toList := by
    have h := streamRep_mem hF.setup hsent (any_of_extract (slot_round_open hsl))
    rwa [hrep] at h
  refine ⟨?_, hmem⟩
  have hlen : (M.hist σ.c).length - 2 = opened st - 1 := by
    rw [LoopModel.hist, length_histOf_take, Nat.min_eq_left (Nat.le_of_lt hlt), hF.count hsent]
    omega
  rw [hJ.plan hsent] at hq
  have hsz : opened st - 1 < (_root_.Flock.Firewall.Session.planOf spec sts).size := by
    by_contra hc
    rw [Array.getElem?_eq_none (by omega)] at hq
    cases hq
  have hsum : opened st - 1 < (Loop.roundsOf spec sts).sum := by
    rw [← Loop.plan_size]
    exact hsz
  have hfst := Loop.fst_plan hsum
  rw [Array.getD_eq_getD_getElem?, hq, Option.getD_some] at hfst
  unfold nextAt
  rw [hlen]
  generalize (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (opened st - 1)).1 = k at hfst ⊢
  unfold _root_.Flock.Firewall.Session.repName at hfst
  rcases hek : spec.repStreams[k]? with _ | ⟨a, b, c⟩
  · simp only [hek, Option.map_none, Option.getD_none] at hfst
    exact absurd hfst (Loop.repStreams_name hmem)
  · simp only [hek, Option.map_some, Option.getD_some] at hfst
    have hmk : (a, b, c) ∈ spec.repStreams.toList :=
      List.mem_iff_getElem?.mpr ⟨k, by rw [Array.getElem?_toList]; exact hek⟩
    rw [← hfst] at hmk
    rw [← rep_at hek]
    exact rep_of_name hnd hmem hmk

/-- **Past its stream's first round, a frame round's rep has `RepJ`**: its stream is seen (`SeenAt`). -/
theorem repJ_cur {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (hpos : 0 < (nextAt spec sts (M.hist σ.c)).2) :
    RepJ M sid spec sts i r σ := by
  obtain ⟨hk, hmem⟩ := round_rep hsl hF hJ hrep hlt hnd
  exact hJ.reps i r stream hmem ((hJ.seenAt i r stream hmem).2 hk hpos)

/-- **At its stream's first round, a frame round's rep's frame is fresh**: its stream is not seen (`SeenAt`). -/
theorem freshAt_cur {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {seen : List String} {st : State} {pl : Placed} {σ : LoopAt} {stream : String} {n i r : Nat}
    (hsl : slot st pl = some (.round stream n)) (hF : LoopFacts sid spec seen st pl σ)
    (hJ : LoopJ M sid spec sts seen st σ) (hrep : streamRep st stream = (i, r)) (hlt : σ.c < M.τ.length)
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (h0 : (nextAt spec sts (M.hist σ.c)).2 = 0) :
    FreshAt σ.R sid i r := by
  obtain ⟨hk, hmem⟩ := round_rep hsl hF hJ hrep hlt hnd
  refine hF.fresh i r stream hmem ?_
  cases hc : seen.contains stream
  · rfl
  · exfalso
    rcases (hJ.seenAt i r stream hmem).1 hc with h | ⟨-, h⟩ <;> omega

/-- **A round of the model's next round's stream keeps `SeenAt` one round on**: where that stream is table `i`'s rep
`r`'s, `s`, the rep streams' names are distinct, and the history `h'` is one message past `h`, past the `Commit`, the
streams seen and `s` are those with a round before the model's next at `h'`. -/
theorem seenAt_round {spec : SessionSpec} {sts : Array Stmt} {h h' : List (Claims.Msg × Claims.Msg)}
    {seen : List String} {i r : Nat} {s : String} (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (hmem : (i, r, s) ∈ spec.repStreams.toList) (hk : 2 * i + r = (nextAt spec sts h).1) (hl : 2 ≤ h.length)
    (hl' : h'.length = h.length + 1) (hS : SeenAt spec sts h seen) : SeenAt spec sts h' (s :: seen) := by
  have hq : h'.length - 2 = h.length - 2 + 1 := by omega
  intro i' r' s' hmem'
  obtain ⟨ha, hb⟩ := hS i' r' s' hmem'
  unfold nextAt at ha hb hk ⊢
  rw [hq]
  have hmono := locate_fst_succ (Loop.roundsOf spec sts) 0 (h.length - 2)
  have hsucc := Loop.locate_succ (Loop.roundsOf spec sts) 0 (h.length - 2)
  have hpred := Loop.locate_pred (Loop.roundsOf spec sts) 0 (h.length - 2)
  refine ⟨fun hc => ?_, fun he hpos => ?_⟩
  · have hle : 2 * i' + r' ≤ (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1 := by
      simp only [List.contains_cons, Bool.or_eq_true, beq_iff_eq] at hc
      rcases hc with hc | hc
      · subst hc
        have := rep_of_name hnd hmem' hmem
        omega
      · rcases ha hc with h1 | ⟨h1, -⟩ <;> omega
    by_cases heq : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2 + 1)).1 =
        (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1
    · have h2 := hsucc heq
      rcases Nat.lt_or_eq_of_le hle with h3 | h3
      · exact Or.inl (by omega)
      · exact Or.inr ⟨by omega, by omega⟩
    · exact Or.inl (by omega)
  · have hj : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2 + 1)).2 =
        (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2 + 1)).2 - 1 + 1 := by
      omega
    have hL : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1 =
        (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2 + 1)).1 := by
      rw [hpred _ hj]
    have hs' : s' = s := congrArg (fun x => x.2.2) (rep_eq hmem' hmem (by omega))
    simp [hs']

/-- **A rep stream before the next round's has `StreamAt` vacuously one round on**: past the `Commit`, at the history
`h'` one message past `h`, the next round is neither the stream's nor the next stream's first. -/
theorem streamAt_old {W Rr : Type} {M : LoopModel W Rr} {spec : SessionSpec} {sts : Array Stmt} {k : Nat}
    {h h' : List (Claims.Msg × Claims.Msg)} {rel : _root_.Flock.Firewall.Frames.Release} {name : String}
    {l0 : _root_.Flock.Firewall.Level0.Level0} {p : Rep} {R : _root_.Flock.Firewall.Frames.Sessions}
    (hk : k < (nextAt spec sts h).1) (hl : 2 ≤ h.length) (hl' : h'.length = h.length + 1) :
    StreamAt M spec sts k h' rel name l0 p R := by
  have hq : h'.length - 2 = h.length - 2 + 1 := by omega
  unfold nextAt at hk
  have hmono := locate_fst_succ (Loop.roundsOf spec sts) 0 (h.length - 2)
  have hsucc := Loop.locate_succ (Loop.roundsOf spec sts) 0 (h.length - 2)
  unfold StreamAt
  rw [hq]
  refine ⟨fun he => absurd he (by omega), fun he h0 => ?_⟩
  exfalso
  by_cases heq : (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2 + 1)).1 =
      (ZKMal.CoinLayer.locate (Loop.roundsOf spec sts) 0 (h.length - 2)).1
  · have := hsucc heq
    omega
  · omega

/-- **A rep stream before the next round's keeps `RepJ` one round on** across updates that keep its frame and what
the session's rounds read of it. -/
theorem repJ_old {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {i r : Nat} {σ σ' : LoopAt} (hk : Kept sid i r σ.R σ'.R) (hs : Stays sid σ σ')
    (hold : 2 * i + r < (nextAt spec sts (M.hist σ.c)).1) (hl : 2 ≤ (M.hist σ.c).length)
    (hl' : (M.hist σ'.c).length = (M.hist σ.c).length + 1) (h : RepJ M sid spec sts i r σ) :
    RepJ M sid spec sts i r σ' := by
  obtain ⟨p, rel, name, l0, hf, hss, hn, hl0, -⟩ := h
  refine ⟨p, rel, name, l0, ?_, ?_, hn, ?_, streamAt_old hold hl hl'⟩
  · rw [frameAt_kept hk (by rw [hf]; rfl)]
    exact hf
  · rw [sessAt_stay (hs.2 name l0 hl0) (by rw [hss]; rfl)]
    exact hss
  · rw [hs.1 name]
    exact hl0

open Claims.ZKMal in
/-- **The history one message on** is the history with one more round. -/
theorem LoopModel.hist_succ {W Rr : Type} (M : LoopModel W Rr) {j : Nat} (hj : j < M.τ.length) :
    ∃ e', M.hist (j + 1) = M.hist j ++ [e'] := by
  unfold LoopModel.hist histOf
  rw [List.take_add_one, List.getElem?_eq_getElem hj, Option.toList_some, List.map_append, List.map_cons,
    List.map_nil]
  exact ⟨_, rfl⟩

/-- The history's later answers, decoded alike, where its answers are alike. -/
theorem tail_snd {β : Type} {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) (g : Msg → β) :
    h.tail.map (fun p => g p.2) = h'.tail.map (fun p => g p.2) := by
  have e : ∀ l : List (Msg × Msg), l.map (fun p => g p.2) = (l.map Prod.snd).map g := by
    intro l
    induction l with
    | nil => rfl
    | cons b l ih => exact congrArg (g b.2 :: ·) ih
  have e' : ∀ l : List (Msg × Msg), l.tail.map (fun p => g p.2) = (l.map Prod.snd).tail.map g := by
    intro l
    cases l with
    | nil => rfl
    | cons b l => exact e l
  rw [e', e', hs]

/-- The configuration reads the history only through its length. -/
theorem cfg_length {W Rr : Type} (L : LeanRelease W Rr) (x : Msg) (ρ : Rr) {h h' : List (Msg × Msg)}
    (hl : h.length = h'.length) : L.cfg x ρ h = L.cfg x ρ h' := by
  unfold LeanRelease.cfg
  rw [hl]

/-- **The transport's coin check reads the history only through its answers.** -/
theorem opens_snd {W Rr : Type} (L : LeanRelease W Rr) (x : Msg) (ρ : Rr) {h h' : List (Msg × Msg)}
    (hs : h.map Prod.snd = h'.map Prod.snd) : L.opens x ρ h = L.opens x ρ h' := by
  match h, h', hs with
  | [], [], _ => rfl
  | [(_, _)], [(_, _)], _ => rfl
  | (_, a) :: o :: ops, (_, a') :: o' :: ops', hs =>
    simp only [List.map_cons, List.cons.injEq] at hs
    obtain ⟨rfl, ho, hops⟩ := hs
    simp only [LeanRelease.opens, List.map_cons, ho, hops]
  | [], _ :: _, hs => simp at hs
  | _ :: _, [], hs => simp at hs
  | [_], _ :: _ :: _, hs => simp at hs
  | _ :: _ :: _, [_], hs => simp at hs

/-- **The frame's taking of the coins reads the history only through its answers**, where the frame's states and the
transport's coins do. -/
theorem frameAnswer_snd {W Rr : Type} (L : LeanRelease W Rr) {x : Msg} {ρ : Rr}
    (hp : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.prog x h = L.prog x h')
    (hf : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.fresh x h = L.fresh x h')
    (hc : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.coinsOf x h = L.coinsOf x h')
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    L.frameAnswer x ρ h = L.frameAnswer x ρ h' := by
  have k : ∀ (e : Msg × Msg) (t : List (Msg × Msg)), L.frameAnswer x ρ (e :: t) =
      LeanRelease.takeAt (e :: t).length (decide (L.rep x (e :: t).length = L.rep x ((e :: t).length - 1)))
        (L.cfg x ρ (e :: t)) (L.cfgAt x ρ ((e :: t).length - 1)) (L.prog x (e :: t)) (L.fresh x (e :: t))
        (L.coinsOf x (e :: t)) := by
    intro e t
    unfold LeanRelease.frameAnswer
    cases hg : (e :: t).getLast? with
    | none => simp at hg
    | some _ => rfl
  have hl : h.length = h'.length := by simpa using congrArg List.length hs
  match h, h', hs, hl with
  | [], [], _, _ => rfl
  | [], _ :: _, hs, _ => simp at hs
  | _ :: _, [], hs, _ => simp at hs
  | e :: t, e' :: t', hs, hl =>
    rw [k, k, hp _ _ hs, hf _ _ hs, hc _ _ hs, cfg_length L x ρ hl, hl]

/-- **Whether the frame took every earlier round's coins reads the history only through its answers**, where the
frame's states and the transport's coins do. -/
theorem tookEarlier_snd {W Rr : Type} (L : LeanRelease W Rr) {x : Msg} {ρ : Rr}
    (hp : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.prog x h = L.prog x h')
    (hf : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.fresh x h = L.fresh x h')
    (hc : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.coinsOf x h = L.coinsOf x h')
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    L.tookEarlier x ρ h = L.tookEarlier x ρ h' := by
  have hl : h.length = h'.length := by simpa using congrArg List.length hs
  unfold LeanRelease.tookEarlier
  rw [hl]
  congr 1
  funext j
  rw [frameAnswer_snd L hp hf hc (h := h.take j) (h' := h'.take j) (by rw [List.map_take, List.map_take, hs])]

/-- **The coins' answer reads the history only through its answers**, where the frame's states and the transport's
coins do: the transport's coin check (`opens_snd`), the earlier rounds' coins (`tookEarlier_snd`) and the last's
(`frameAnswer_snd`). -/
theorem answered_snd {W Rr : Type} (L : LeanRelease W Rr) {x : Msg} {ρ : Rr}
    (hp : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.prog x h = L.prog x h')
    (hf : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.fresh x h = L.fresh x h')
    (hc : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.coinsOf x h = L.coinsOf x h')
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    L.answered x ρ h = L.answered x ρ h' := by
  unfold LeanRelease.answered
  rw [opens_snd L x ρ hs, tookEarlier_snd L hp hf hc hs, frameAnswer_snd L hp hf hc hs]

/-- **At the loop's own frames the coins' answer reads the history only through its answers**: the frame's states
are of the decoded coins (`ofLoop_progReads`), and so are the transport's coins (`loop_coinsRead`). -/
theorem ofLoop_answered_snd (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    (LeanRelease.ofLoop T L₀ real).answered x ρ h = (LeanRelease.ofLoop T L₀ real).answered x ρ h' := by
  refine answered_snd _ (fun h h' hs => ?_) (fun h h' hs => ?_) (fun h h' hs => ?_) hs
  · exact congrArg (Loop.progOf x) (tail_snd hs ZKMal.CoinLayer.rd)
  · exact congrArg (Loop.freshOf x) (tail_snd hs ZKMal.CoinLayer.rd)
  · rcases h with _ | ⟨⟨m, a⟩, t⟩ <;> rcases h' with _ | ⟨⟨m', a'⟩, t'⟩
    · rfl
    · simp at hs
    · simp at hs
    · exact (LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x m a t).trans
        ((congrArg (Loop.readCoins x) (tail_snd hs ZKMal.CoinLayer.rd)).trans
          (LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x m' a' t').symm)

/-- **The frame's coin step reads the history only through its answers**, as `frameAnswer_snd`. -/
theorem frameAnswerR_snd {W Rr : Type} (L : LeanRelease W Rr) {x : Msg} {ρ : Rr}
    (hp : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.prog x h = L.prog x h')
    (hf : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.fresh x h = L.fresh x h')
    (hc : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.coinsOf x h = L.coinsOf x h')
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    L.frameAnswerR x ρ h = L.frameAnswerR x ρ h' := by
  have hl : h.length = h'.length := by simpa using congrArg List.length hs
  match h, h', hs, hl with
  | [], [], _, _ => rfl
  | [], _ :: _, hs, _ => simp at hs
  | _ :: _, [], hs, _ => simp at hs
  | e :: t, e' :: t', hs, hl =>
    rw [LeanRelease.frameAnswerR_cons, LeanRelease.frameAnswerR_cons, hp _ _ hs, hf _ _ hs, hc _ _ hs,
      cfg_length L x ρ hl, hl]

/-- **The coins' answer before the rank stop reads the history only through its answers**, as `answered_snd`. -/
theorem answeredR_snd {W Rr : Type} (L : LeanRelease W Rr) {x : Msg} {ρ : Rr}
    (hp : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.prog x h = L.prog x h')
    (hf : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.fresh x h = L.fresh x h')
    (hc : ∀ h h' : List (Msg × Msg), h.map Prod.snd = h'.map Prod.snd → L.coinsOf x h = L.coinsOf x h')
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    L.answeredR x ρ h = L.answeredR x ρ h' := by
  unfold LeanRelease.answeredR
  rw [opens_snd L x ρ hs, tookEarlier_snd L hp hf hc hs, frameAnswerR_snd L hp hf hc hs]

/-- **At the loop's own frames the coins' answer before the rank stop reads the history only through its answers**,
as `ofLoop_answered_snd`. -/
theorem ofLoop_answeredR_snd (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    (LeanRelease.ofLoop T L₀ real).answeredR x ρ h = (LeanRelease.ofLoop T L₀ real).answeredR x ρ h' := by
  refine answeredR_snd _ (fun h h' hs => ?_) (fun h h' hs => ?_) (fun h h' hs => ?_) hs
  · exact congrArg (Loop.progOf x) (tail_snd hs ZKMal.CoinLayer.rd)
  · exact congrArg (Loop.freshOf x) (tail_snd hs ZKMal.CoinLayer.rd)
  · rcases h with _ | ⟨⟨m, a⟩, t⟩ <;> rcases h' with _ | ⟨⟨m', a'⟩, t'⟩
    · rfl
    · simp at hs
    · simp at hs
    · exact (LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x m a t).trans
        ((congrArg (Loop.readCoins x) (tail_snd hs ZKMal.CoinLayer.rd)).trans
          (LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x m' a' t').symm)

/-- **At the loop's own frames the coin step after a history reads it only through its answers**, as
`ofLoop_answered_snd`. -/
theorem ofLoop_fed_snd (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand))
    (real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (x : Msg)
    (ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) :
    (LeanRelease.ofLoop T L₀ real).fed x ρ h = (LeanRelease.ofLoop T L₀ real).fed x ρ h' := by
  have hl : h.length = h'.length := by simpa using congrArg List.length hs
  have hp : (LeanRelease.ofLoop T L₀ real).prog x h = (LeanRelease.ofLoop T L₀ real).prog x h' :=
    congrArg (Loop.progOf x) (tail_snd hs ZKMal.CoinLayer.rd)
  have hc : (LeanRelease.ofLoop T L₀ real).coinsOf x h = (LeanRelease.ofLoop T L₀ real).coinsOf x h' := by
    rcases h with _ | ⟨⟨m, a⟩, t⟩ <;> rcases h' with _ | ⟨⟨m', a'⟩, t'⟩
    · rfl
    · simp at hs
    · simp at hs
    · exact (LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x m a t).trans
        ((congrArg (Loop.readCoins x) (tail_snd hs ZKMal.CoinLayer.rd)).trans
          (LeanRelease.loop_coinsRead Loop.loopNr T (LeanRelease.withFrames T L₀ real) x m' a' t').symm)
  unfold LeanRelease.fed
  rw [hp, hc, hl]

/-- **`SameInv` reads the history only through the configuration and the coins' answer there** before the rank
stop. -/
theorem sameInv_hist {W Rr : Type} {L : LeanRelease W Rr} {x : Msg} {ρ : Rr} {h h' : List (Msg × Msg)}
    {cs : List FlockDeploy.Coin} {q j : Nat} {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray}
    {xs : Array (Array _root_.Flock.F128)} {hi : Array ByteArray} {pre : List _root_.Flock.Firewall.Frame.Req}
    {ev : List (Nat × ByteArray × _root_.Flock.Firewall.Frame.Place × Array _root_.Flock.F128)} {p : Rep}
    (hc : L.cfg x ρ h' = L.cfg x ρ h) (ha : L.answeredR x ρ h' = L.answeredR x ρ h)
    (hI : SameInv L x ρ h cs q j t dom xs hi pre ev p) : SameInv L x ρ h' cs q j t dom xs hi pre ev p :=
  ⟨hI.run, hI.cfg.trans hc.symm, fun m hm => hI.st m (ha.symm.trans hm), hI.coins, hI.inv⟩

/-- **At the loop's own frames `SameInv` reads the history only through its answers.** -/
theorem sameInv_snd {T : ZKMal.CoinLayer.TableData}
    {L₀ : LeanRelease Claims.Flock.Wit
      ((ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand)}
    {real : Msg → Claims.Flock.Wit →
      (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand →
      Nat → _root_.Flock.V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray)} {x : Msg}
    {ρ : (ZKMal.CoinLayer.session Loop.loopNr T).refP.Nonce × (ZKMal.CoinLayer.session Loop.loopNr T).refP.Rand}
    {h h' : List (Msg × Msg)} (hs : h.map Prod.snd = h'.map Prod.snd) {cs : List FlockDeploy.Coin} {q j : Nat}
    {t : _root_.Flock.Firewall.Finish.Tab} {dom : ByteArray} {xs : Array (Array _root_.Flock.F128)}
    {hi : Array ByteArray} {pre : List _root_.Flock.Firewall.Frame.Req}
    {ev : List (Nat × ByteArray × _root_.Flock.Firewall.Frame.Place × Array _root_.Flock.F128)} {p : Rep}
    (hI : SameInv (LeanRelease.ofLoop T L₀ real) x ρ h cs q j t dom xs hi pre ev p) :
    SameInv (LeanRelease.ofLoop T L₀ real) x ρ h' cs q j t dom xs hi pre ev p :=
  sameInv_hist (cfg_length _ x ρ (h := h') (h' := h) (by simpa using (congrArg List.length hs).symm))
    (ofLoop_answeredR_snd T L₀ real x ρ hs).symm hI

/-- **The session's stop one frame round on**: at a frame round with `RoundFacts` that releases, whose coins the
auditor answered and the session and the rep's frame took, the loop's frames at the session's next request have
stopped the session exactly where a round up to this one failed. -/
theorem stopped_roundRun {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {seen : List String}
    {stream : String} {n i r : Nat} {dom : ByteArray} {v : Claims.Msg} {σ : LoopAt} {st : State} {pl : Placed}
    {e : Env} {R₃ : Sessions} (hA : RoundFacts M sid spec seen stream n i r dom v σ)
    (hv : M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false))
    (hr : RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 R₃)
    (hreq : st.request pl (.round stream n) = none) {o a : ByteArray} (ho : e.round = .ok o)
    (ha : e.ans = some a) (hb : byte? a 0 = some 2) (hs : (st.answer (.round stream n) a).1.stop = none)
    (hc : e.coins = .ok ()) (e' : Claims.Msg × Claims.Msg) :
    R₃.stopped sid = !M.L.toSrv.liveBefore M.wk M.x M.w M.ρ M.κ (M.hist σ.c ++ [e']) := by
  obtain ⟨m, q, f, nd, hm, hrp, hnd, hcfg, hu, -⟩ := hA.place
  have hcn := hA.coins
  subst hcn
  have ho' := (hr.round hreq).symm.trans ho
  obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
  obtain ⟨R₁, R₂, R₂', p', h01, h12, hcs, h22, h23⟩ := hr.coins hreq ho hput ha hb hs hc
  have hfa : frameAt R₂ sid i r = some p₁ := frameAt_own hA.frame (roundStep_key hput) (h01.1 i r) (h12.1 i r)
  obtain ⟨j, hj⟩ := find_of_frameAt hfa
  obtain ⟨st₂, hst, rfl⟩ := coinsStep_inv hj hcs
  exact stopped_round_next M.L M.wk M.x M.w M.ρ M.κ (M.hist σ.c) hA.kind hA.last hv hA.cross hm hA.frame hrp hnd
    hcfg hu hA.stopped ho' hput (h01.1 i r) h01.2 ((h12.1 i r).trans (h22.1 i r)) (h12.2.trans h22.2) hst h23.2 e'

/-- **`LoopJ` is kept at a frame round**, where the rep streams' names are distinct, `RoundFacts` holds at each
request at the slot as the loop answers it (`slotCase_round`'s premise), and the round's rep has `RepJ` at the
session's next request with the model's coin step taking the round's coins (`hcur`): the session's stop one round on
is `stopped_roundRun`'s; every other seen rep stream is before the round's, on another rep, so its frame and `RepJ`
are kept; the round's stream is seen; and, where the model's release requests the loop's openings (`hrq`), the
session's coin tree past it is the one its tree before opens the model's answer to at the schedule's next opening
(`LoopModel.TookAt.succ`), the frame having taken the coins before (`RoundFacts.place`) and the model's coin step the
round's (`hcur`). -/
theorem loopKept_round {W Rr : Type} {M : LoopModel W Rr} {sid : Nat} {spec : SessionSpec} {sts : Array Stmt}
    {nonce : ByteArray} {stream : String} {n : Nat} (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    {s0 : FlockDeploy.Ins} (hx : FlockDeploy.setupOf true M.x = some (s0, spec, sts))
    (hrq : ∀ j, M.L.req M.x j = Loop.reqOf M.x j)
    (hA : ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat) (dom : ByteArray)
      (v : Claims.Msg), slot st pl = some (.round stream n) → LoopFacts sid spec seen st pl σ →
      LoopJ M sid spec sts seen st σ → σ.c < M.τ.length → M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → SaltsSrc M sid σ →
      InnerSrc M sid i r σ → StartOk sts sid i r σ → (∃ o, e.round = .ok o) →
      RoundFacts M sid spec seen stream n i r dom v σ)
    (hcur : ∀ (seen : List String) (st : State) (pl : Placed) (e : Env) (σ σ' : LoopAt) (i r : Nat)
      (dom : ByteArray) (v : Claims.Msg), slot st pl = some (.round stream n) → e.setup = .ok (spec, sts) →
      LoopFacts sid spec seen st pl σ → LoopJ M sid spec sts seen st σ → M.Ans σ.c e →
      M.wk M.x M.w M.ρ (M.hist σ.c) 0 = some (v, false) →
      RoundRun sid st pl stream n i r dom ⟨v.toArray⟩ e σ.R σ.L0 σ'.R →
      (∀ p, frameAt σ.R sid i r = some p → p.cfg = M.L.cfg M.x M.ρ (M.hist σ.c)) → σ'.c = σ.c + 1 →
      Stays sid σ σ' → RoundFacts M sid spec seen stream n i r dom v σ → ∀ o a, e.round = .ok o →
      e.ans = some a → byte? a 0 = some 2 → (st.answer (.round stream n) a).1.stop = none → e.coins = .ok () →
      RepJ M sid spec sts i r σ' ∧ M.L.fed M.x M.ρ (M.hist σ'.c) = true) :
    LoopKept M sid spec sts nonce (LoopJ M sid spec sts) (.round stream n) := by
  intro seen st pl e σ σ' hsl he hF hJ hra hgo
  obtain ⟨hans, ⟨i, r, dom, v, hv, hr, hpc, hin, hsk⟩, hcc, hst, hsal⟩ := hra
  have hlt := hans.1
  obtain ⟨hsent, hop, -, -⟩ := slot_round hsl
  have hreq := request_round_slot hF.reach hF.live hsl
  obtain ⟨⟨o, hro⟩, a, hae, hb, hsa, hco⟩ := stepK_round_go hF.reach hF.live hsl hgo
  have hRA := hA seen st pl e σ σ' i r dom v hsl hF hJ hlt hv hr hpc hsal hin hsk ⟨o, hro⟩
  have hk : (Except.ok (.round stream n) : Except String _root_.Flock.Firewall.Session.Kind) ≠ .ok .hello := by
    intro h
    cases h
  have hsent' := stepK_sent (hold := true) (st := st) (pl := pl) (b := some (tagOf (.round stream n))) (e := e) hk
  obtain ⟨hkr, hmem⟩ := round_rep hsl hF hJ hr.rep_eq hlt hnd
  have hl : 2 ≤ (M.hist σ.c).length := by
    rw [LoopModel.hist, length_histOf_take, Nat.min_eq_left (Nat.le_of_lt hlt), hF.count hsent]
    omega
  have hl' : (M.hist σ'.c).length = (M.hist σ.c).length + 1 := by
    rw [hcc, LoopModel.hist, LoopModel.hist, length_histOf_take, length_histOf_take]
    omega
  obtain ⟨e', he'⟩ := LoopModel.hist_succ M hlt
  have hsess : SessOn spec sts σ'.R sid := by
    obtain ⟨_, _, _, _, -, hrp₀, -⟩ := hRA.place
    obtain ⟨p₀, hput₀⟩ := roundStep_put_some hrp₀
    obtain ⟨R₁, R₂, R₂', p', h01, h12, -, h22, h23⟩ := hr.coins hreq hro hput₀ hae hb hsa hco
    have h1 := sessOn_sessKept h01.2 (hJ.sess hsent)
    have h2 := sessOn_sessKept h22.2 (sessOn_sessKept h12.2 h1)
    exact sessOn_sessKept h23.2 h2
  refine ⟨?_, fun h => ?_, fun i' r' s' hmem' hseen => ?_, stepK_plan he hJ.plan,
    seenAt_round hnd hmem hkr hl hl' hJ.seenAt, fun h => absurd h (by rw [hcc, hF.count hsent]; omega),
    fun _ => builtAt_stays hst (hJ.built (by rw [hF.count hsent]; omega)), fun _ => hsess, stepK_links he hJ.links,
    fun _ => linkOn_stays (hJ.sess hsent) hst (by rw [hF.count hsent]; omega) (by omega)
      (hJ.link (by rw [hF.count hsent]; omega)), fun _ => hJ.sizes hsent, ?_⟩
  · rw [hsent', hsent, hcc, he', Bool.true_and]
    exact stopped_roundRun hRA hv hr hreq hro hae hb hsa hco e'
  · rw [hsent', hsent] at h
    cases h
  · by_cases hs' : s' = stream
    · obtain ⟨rfl, rfl⟩ : i' = i ∧ r' = r := by
        have hm2 : (i, r, s') ∈ spec.repStreams.toList := by
          rw [hs']
          exact hmem
        have h3 := rep_eq hmem' hm2 (rep_of_name hnd hmem' hm2)
        exact ⟨congrArg Prod.fst h3, congrArg (fun x => x.2.1) h3⟩
      exact (hcur seen st pl e σ σ' _ _ dom v hsl he hF hJ hans hv hr hpc hcc hst hRA o a hro hae hb hsa hco).1
    · have hseen' : seen.contains s' = true := by
        simpa [seenAfter, hs'] using hseen
      have hne := hRA.other i' r' s' hmem' hs'
      have hne2 : 2 * i' + r' ≠ 2 * i + r := fun h2 =>
        hne (congrArg (fun x => (x.1, x.2.1)) (rep_eq hmem' hmem h2))
      have hold : 2 * i' + r' < (nextAt spec sts (M.hist σ.c)).1 := by
        rcases (hJ.seenAt i' r' s' hmem').1 hseen' with h1 | ⟨h1, -⟩ <;> omega
      obtain ⟨_, _, _, _, -, hrp, -⟩ := hRA.place
      obtain ⟨p₁, hput⟩ := roundStep_put_some hrp
      obtain ⟨R₁, R₂, R₂', p', h01, h12, hcs, h22, h23⟩ := hr.coins hreq hro hput hae hb hsa hco
      have hK : Kept sid i' r' σ.R σ'.R :=
        (((((h01.1 i' r').step (.put _ _ (key_ne (roundStep_key hput) hne))).trans (h12.1 i' r')).trans
          (h22.1 i' r')).step (.put _ _ (key_ne (coinsStep_key hcs) hne))).trans (h23.1 i' r')
      exact repJ_old hK hst hold hl hl' (hJ.reps i' r' s' hmem' hseen')
  · intro _
    have hfd := (hcur seen st pl e σ σ' i r dom v hsl he hF hJ hans hv hr hpc hcc hst hRA o a hro hae hb hsa hco).2
    obtain ⟨hj, hAn⟩ := hans
    obtain ⟨C, hC⟩ := Option.isSome_iff_exists.mp (hF.reach.sess hsent)
    obtain ⟨a', C', c, hae', hb', ho, hs'⟩ := stepK_coin_open hF.live (.inr ⟨stream, n, rfl⟩) hC hgo
    have hm := hAn a' hae' hb'
    have hc := hF.count hsent
    rw [opened_of hC] at hc
    have hq : reqAt st.plan C.hello.links C.opened.length = M.L.req M.x (σ.c - 1) := by
      rw [hrq, Loop.reqOf_eq hx, hJ.plan hsent, hJ.links.2 C hC, show C.opened.length = σ.c - 1 by omega]
    obtain ⟨m, -, -, -, hAm, -⟩ := hRA.place
    rw [hcc] at hfd ⊢
    refine LoopModel.TookAt.succ hj (hJ.took hj) (by omega) hC (by rw [← hq, hm]; exact ho) hs' ?_ hfd
    rw [LeanRelease.frameAnswer_of_answered hAm] <;> rfl

end Claims.ZKServer.Relay
