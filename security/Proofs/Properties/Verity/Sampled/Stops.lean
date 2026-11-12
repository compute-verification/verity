module

public import Proofs.Properties.Verity.Sampled.Readings
public import Proofs.Properties.Verity.Sampled.RecordRounds

/-!
# Sampled proofs' coupling: a stop's written coins in the play's log

Where the projection writes a round's coins into a stop's opening, the play on the tape the trace ends on releases them
there.

* **The walk keeps a coin it passes** (`Pos.walk_le`, `Pos.le_toStop`, `Pos.walk_coin_keeps`): the walk extends the
  position at any cuts, and below a coin node no cut fixes, every tape through the walk's stop holds there the coin the
  node's tape holds.
* **A trace keeps the coin at its start** (`RoundsReach.le_tape`, `RoundsReach.playReaches`, `RoundsReach.head`,
  `firstStop_playReaches`): the tape a trace ends on is one through its start, so its play is one the start's node
  reaches, and from a coin node no cut fixes, it holds there the coin the node's tape holds.
* **An opening the device serves, at its coin** (`optOpenK_next`, `OpeningAt.ofNext`, `run_headCoin`): the node after
  an opening frame the device answers is that opening's coin node, described at the close its releases continue to,
  and the play from it is the play after the coin a tape holds there.
* **The release in the log** (`log_of_opening`, `opened_of_head`, `RoundsReach.release`): from an opening's coin node
  whose releases continue to the close after a log `pre`, every play's log is `pre`, the opening answered with the
  release of the coin the tape holds there, then the rest. So off the opening's empty class at that close, the play on
  the tape a trace from the stop ends on releases there the bits the coin map wrote.
* **Each stop's written coins in the log** (`RoundsReach.answers`, `StopsAt`, `StopsAt.next`, `LoggedAfter`,
  `LoggedAt`, `RoundsReach.stop_logged`, `RoundsReach.answers_logged`, `stopOf_firstStop`, `stopsAt_firstStop`,
  `firstStop_logged`): a trace's answered stops, each where `StopOf` says from a log and the close after it and the
  prover's request, the first from the empty log; where every round's reading picks only openings the device serves,
  the play on the tape the trace ends on releases, at a stop where not every cut is past, the bits written there (the
  link points' at the first stop, a round's coins at an answered one), at an opening the rounds whose cuts the stop
  fixes pick.
* **One round a stop** (`DigestCut.modeHere_eq_none`, `DigestCut.modeHere_some`, `Pos.modes_some`, `sub_of_single`,
  `stop_slot`, `LoggedAfter.slot`): at the readings and decoders at an injective layout of distinct tables, a stop's
  submission is its one fixed round's, a rep round, with its most likely message, and its bits are released at that
  round's opening of its rep stream.
* **The trace's answered stops, in order in the log** (`PicksIn`, `PicksOnce`, `AnsweredBy`, `AnsweredBy.seg`,
  `opening_log`, `RoundsReach.order`): where each round picks at most one release of the log the play on the trace's
  tape has, the trace's answered stops release their coins after logs each longer than the one before, and every
  release a round picks is the link's or an answered stop's, unless the trace answered all its rounds or ended at a stop
  the decoders submit nothing at.
* **Each opening the device logs, served** (`LogServed`, `Served.of_next`, `logServed_relK`, `logServed_run`,
  `link_picks`): every opening of a play's log is one the device serves after the releases before it, so a log with an
  opening has a first one, which the link's slot picks.
* **The first stop fixes a cut** (`RoundsReach.run_ret`, `firstStop_skip_log`, `firstStop_fixes`): where every cut is
  past at the first stop no round picks in the play's log, so where the log has an opening, not every cut is past.
* **Each round picks once** (`firstOpened_append`, `helloSpecs_getElem?_append`, `openedOf_lt`,
  `picksOnce_readingAt`, `covers_of_length`): where each tree of the log names its streams distinctly, each round of
  the readings at a layout picks at most one release of the log; and picks after as many logs as the rounds but one
  are every pick.
* **The rounds a trace's submissions append** (`foldl_addRoundJ`): each stream's rounds are the submissions on it, in
  order.
* **The logs before a stream's openings** (`prefixesWhere`, `length_prefixesWhere`, `mem_prefixesWhere`,
  `pairwise_prefixesWhere`, `countP_prefixesWhere`, `map_countP_prefixesWhere`, `eq_of_mem_iff_of_sorted`,
  `filter_logs`): the logs of a log before each of its openings of a stream, as many as the openings, each longer
  than the one before and the `k`-th with `k` of them; and logs each longer than the one before that are exactly where
  the log opens the stream are those.
* **The link's release, then the answered stops'** (`prefix_le`, `firstOpened_ne_none`, `readingAt_served`,
  `forall₂_zip_right`, `firstStop_order`): at the readings and decoders at an injective layout of distinct tables,
  where each round picks at most one release of the log, the trace from the first stop releases the link points' bits
  after a log `L₀`, then each answered stop's coins after a longer log; and where the session's rounds but one are at
  most the trace's budget, every release a round picks is one of those.
* **An answered stop's modes** (`RoundsReach.stop_modes`, `RoundsReach.stop_modes_noMiss`): each cut fixed at an
  answered stop has, on the tape the trace ends on, the most likely message the stop decoded; off the misses, the
  message the play reveals there.
* **What a stop's fixed cuts reveal** (`Pos.revealOn_here`, `RoundsReach.stop_reveal`, `CutsAt`, `openCut_pos`,
  `cutsAt_openCuts`, `LoggedAfter.pre_eq`, `RoundsReach.stop_modes_reveal`): a cut fixed at a stop reveals, on the
  tape a trace from it ends on, its reading of the play there; after an opening frame each is at the bytes the frame
  answers, so off the misses the stop decodes the message the play reveals for its round wherever that message's
  SHA-512 is those bytes.
* **The answered stops of one stream** (`opening_of_opensAt`, `firstOpened_opening`, `firstOpened_past`,
  `helloSpecs_bind_append`, `stream_eq`, `rep_ne_session`, `firstOpened_link`, `stream_logs`): at a layout with a link
  slot, the first stop's log has no opening; and where the opening after it is of the session's stream and the layout
  has a slot for each of the log's openings of a rep stream, the stops submitting on that stream release their coins
  after the logs before each of its openings, in order.
* **The rounds of one stream** (`map_fst_filter_zip`, `rounds_of_logs`, `stream_rounds`): so the rounds a trace appends
  to a stream are one per opening of it, the `k`-th after the log that has opened it `k` times.
* **No more openings than the record's rounds** (`openedOf_le_recordRounds`): where
  the device's log opens only the record's rounds (`OpenedRecorded`) and serves each opening, it opens each named
  stream at most the rounds the record gives it.
-/

@[expose] public section

namespace Verity.Game

variable {α V ι : Type}

namespace Pos

variable {g : Game α} {σ : Strategy g}

open Classical in
/-- **The walk extends the position**, at any cuts. -/
theorem walk_le : ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
    (back : CoinTape g' s' → CoinTape g σ), (⟨g', s', Ks, x, back⟩ : Pos g σ V ι).Le (walk g' s' Ks x back)
  | .ret _, _, _, _, _ => Le.refl _
  | .send M k, s', Ks, x, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · rw [walk_here _ s' Ks x back hf]
      exact Le.refl _
    · rw [walk_send s' Ks x back fun i => by simpa using fun h => hf ⟨i, h⟩]
      exact walk_le (k s'.1) s'.2 _ x back
  | @Game.coin _ C iF iN k, s', Ks, x, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · rw [walk_here _ s' Ks x back hf]
      exact Le.refl _
    · rw [walk_coin s' Ks x back fun i => by simpa using fun h => hf ⟨i, h⟩]
      obtain ⟨f, hf'⟩ := walk_le (k x.1) (s' x.1) (fun i => (Ks i).atCoin x.1) (x.2 x.1)
        fun y => back (x.1, Function.update x.2 x.1 y)
      exact ⟨fun y => (x.1, Function.update x.2 x.1 (f y)), hf'⟩

/-- **The next stop extends the position**, at any cuts. -/
theorem le_toStop (p : Pos g σ V ι) : p.Le p.toStop := walk_le p.g' p.s' p.Ks p.x p.back

open Classical in
/-- **Below a coin node no cut fixes, the walk keeps the node's coin**: every tape through the walk's stop is one
through the node, holding there the coin the node's tape holds. -/
theorem walk_coin_keeps {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} {g' : Game α}
    (hg : g' = @Game.coin _ C iF iN k) (s' : Strategy g') (Ks : ι → DigestCut V g') (x : CoinTape g' s')
    (back : CoinTape g' s' → CoinTape g σ) (hK : ∀ i, (Ks i).isHere = false) :
    ∃ h : CoinTape (walk g' s' Ks x back).g' (walk g' s' Ks x back).s' → CoinTape g' s',
      (∀ y, (walk g' s' Ks x back).back y = back (h y)) ∧ ∀ y, headCoin hg s' (h y) = headCoin hg s' x := by
  subst hg
  rw [walk_coin s' Ks x back hK]
  obtain ⟨f, hf⟩ := walk_le (k x.1) (s' x.1) (fun i => (Ks i).atCoin x.1) (x.2 x.1)
    fun y => back (x.1, Function.update x.2 x.1 y)
  exact ⟨fun y => (x.1, Function.update x.2 x.1 (f y)), hf, fun _ => rfl⟩

/-- **A cut not fixed at a node decodes nothing there.** -/
theorem _root_.Verity.Game.DigestCut.modeHere_eq_none [Nonempty V] {g' : Game α} {K : DigestCut V g'}
    (h : K.isHere = false) (s : Strategy g') : K.modeHere s = none := by
  cases K <;> first | rfl | cases h

/-- **A round whose cut is not fixed at a position decodes nothing there.** -/
theorem modes_eq_none [Nonempty V] {p : Pos g σ V ι} {i : ι} (h : (p.Ks i).isHere = false) : p.modes i = none :=
  DigestCut.modeHere_eq_none h p.s'

/-- **A cut fixed at a node decodes a message there.** -/
theorem _root_.Verity.Game.DigestCut.modeHere_some [Nonempty V] {g' : Game α} {K : DigestCut V g'}
    (h : K.isHere = true) (s : Strategy g') : ∃ m, K.modeHere s = some m := by
  cases K <;> first | exact ⟨_, rfl⟩ | cases h

/-- **A round whose cut is fixed at a position decodes a message there.** -/
theorem modes_some [Nonempty V] {p : Pos g σ V ι} {i : ι} (h : (p.Ks i).isHere = true) : ∃ m, p.modes i = some m :=
  DigestCut.modeHere_some h p.s'

/-- **No cut is fixed at a consumed position.** -/
theorem isHere_consume (p : Pos g σ V ι) (i : ι) : (p.consume.Ks i).isHere = false := by
  show (if (p.Ks i).isHere then (.skip : DigestCut V p.g') else p.Ks i).isHere = false
  by_cases h : (p.Ks i).isHere = true
  · rw [ite_eq_left h]
    rfl
  · rw [ite_eq_right h]
    simpa using h

/-- **What a cut fixed at a position reveals on every later tape**: at a position that keeps the invariant at the
cuts `Ks₀`, a cut fixed there at the reading `rv` reveals, on every tape through a later position, `rv` of the play on
it. -/
theorem revealOn_here {Ks₀ : ι → DigestCut V g} {p q : Pos g σ V ι} (h : p.Inv Ks₀) (hpq : p.Le q) {i : ι}
    {rv : α → Option V} (hi : p.Ks i = .here rv) (y : CoinTape q.g' q.s') :
    (Ks₀ i).revealOn σ (q.back y) = rv (CoinTape.run g σ (q.back y)) := by
  obtain ⟨f, hf⟩ := hpq
  rw [hf y]
  rcases h.cut i with hc | hc
  · refine (hc _ DigestCut.revealReading (f y)).trans ?_
    rw [h.run (f y), hi]
    rfl
  · rw [hi] at hc
    cases hc

end Pos

/-- **The play from a coin node is the play after the coin the tape holds there.** -/
theorem run_headCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (s' : Strategy g') (z : CoinTape g' s') :
    ∃ (s₀ : Strategy (k (headCoin hg s' z))) (y : CoinTape (k (headCoin hg s' z)) s₀),
      CoinTape.run g' s' z = CoinTape.run (k (headCoin hg s' z)) s₀ y := by
  subst hg
  exact ⟨s' z.1, z.2 z.1, rfl⟩

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ addRoundJ addRoundJ_self addRoundJ_ne)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

/-! ## An opening the device serves, at its coin -/

/-- **The node after an opening frame the device answers is the opening's coin node**, where its trees have room for
it. -/
theorem optOpenK_next {β : Type} {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {i s n : ℕ} {b : ByteArray} (hc : treeLines ts < maxTreeLines) {T : OpenTree} (hT : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p) :
    optOpenK k fs ts f i s n b ts[i]? = treeOpenK k fs ts f i s n b T (some p) := by
  unfold optOpenK
  rw [ite_eq_left hc, hT]
  show treeOpenK k fs ts f i s n b T (T.next s n) = _
  rw [hp]

/-- **The node after an opening frame the device answers, described**: the tree `T` it holds and the round `p` it
opens, where the releases continue to the close `E` after the log `pre` and them. -/
def OpeningAt.ofNext {β : Type} {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {E : List Release → Game β} {pre : List Release} {i s n : ℕ} {b : ByteArray} (hf : ∀ log, f log = E (pre ++ log))
    (hc : treeLines ts < maxTreeLines) {T : OpenTree} (hT : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p) :
    OpeningAt E (optOpenK k fs ts f i s n b ts[i]?) :=
  ⟨T, p, i, s, n, b, k, fs, ts, pre, _, _, ((optOpenK_next hc hT hp).trans
    (congrArg (fun f' => treeOpenK k fs ts f' i s n b T (some p)) (funext hf))).trans rfl⟩

/-! ## The release in the log -/

/-- **The release at an opening's coin node**: from the node after an opening frame of the `i`-th tree's stream `s`
that the device answers, where the releases continue to the close after the log `pre` and them, every play's log is
`pre`, then the opening answering `b` with the release of the coin the tape holds there, then the rest. -/
theorem log_of_opening {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray} {k : ℕ}
    {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release} {i s n : ℕ}
    {b : ByteArray} (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log)) (hc : treeLines ts < maxTreeLines)
    {T : OpenTree}
    (hT : ts[i]? = some T) {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p)
    (s' : Strategy (optOpenK k fs ts f i s n b ts[i]?)) (z : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s') :
    ∃ L, (CoinTape.run _ s' z).log = pre ++ ⟨.opening i s n, b,
      TreeDraw.opened (headCoin (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp).eq s' z).1 n⟩ :: L := by
  obtain ⟨s₀, y, hrun⟩ := run_headCoin (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp).eq s' z
  have h₀ : PlayReaches _ s₀ (CoinTape.run (optOpenK k fs ts f i s n b ts[i]?) s' z) := by
    rw [hrun]
    exact run_reaches _ s₀ y
  obtain ⟨L, s₁, h₁⟩ := playReaches_relK k fs (ts.set i (T.bump p.1)) _ s₀ h₀
  exact ⟨L, log_of_finish h₁⟩

/-- **The release of a coin a tape keeps from a written one is the written bits**: at the node after an opening frame
the device answers, its releases continuing to a close `E` that records its log after the log `pre` and them, off the
opening's empty class at `bits`, where a tape holds the coin the tape with `bits` written in holds, its coin there is
released as `bits`. -/
theorem opened_of_head {β : Type} {lg : β → List Release} {E : List Release → Game β} (hRL : RecordsLog lg E)
    {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β} {pre : List Release}
    {i s n : ℕ} {b : ByteArray} (hf : ∀ log, f log = E (pre ++ log)) (hc : treeLines ts < maxTreeLines)
    {T : OpenTree} (hT : ts[i]? = some T) {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)}
    (hp : T.next s n = some p) {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)} {bits : List Bool}
    (hE : ¬ EmptyAt E (optOpenK k fs ts f i s n b ts[i]?) bits)
    (x'' z : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'')
    (hh : headCoin (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp).eq s'' z =
      headCoin (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp).eq s'' (writeOpening E s'' bits x'')) :
    TreeDraw.opened (headCoin (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp).eq s'' z).1 n = bits := by
  obtain ⟨oa, hoa⟩ := openingOf_some (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp)
  rw [hh]
  exact opened_writeOpening_off hRL s'' bits x'' hE oa hoa (OpeningAt.ofNext (k := k) (b := b) hf hc hT hp)

/-! ## A trace keeps the coins written at its stops -/

section Trace

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} {D : Decoders R Reg}
  {M : CoinMap g σ Ω R mPts} {dg : ByteArray → ByteArray} {J : ℕ} {B : RoundHold}

/-- **The tape a trace ends on is one through its start**, at any cuts. -/
theorem RoundsReach.le_tape {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)}
    {o : β × CoinTape g σ} (hr : RoundsReach D M dg J B fin rootB pts N rs p o) :
    ∃ q : Pos g σ Claims.Msg (Fin R), p.Le q ∧ o.2 = q.tape := by
  induction hr with
  | zero rs p => exact ⟨p, Pos.Le.refl p, rfl⟩
  | stop N rs p _ => exact ⟨p.toStop, Pos.le_toStop p, rfl⟩
  | round N rs p s c o _ _ _ ih =>
    obtain ⟨q, hq, ho⟩ := ih
    exact ⟨q, ((Pos.le_toStop p).trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans hq, ho⟩

/-- **A trace's start reaches the play of the tape it ends on**, where the start keeps the play's start (`Pos.Inv`):
the tape is one through the start, whose play is the play from the start's node. -/
theorem RoundsReach.playReaches {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {p : Pos g σ Claims.Msg (Fin R)} {o : β × CoinTape g σ} (hr : RoundsReach D M dg J B fin rootB pts N rs p o)
    (hp : p.Inv Ks₀) : PlayReaches p.g' p.s' (CoinTape.run g σ o.2) := by
  obtain ⟨q, ⟨h, hh⟩, hoq⟩ := hr.le_tape
  rw [hoq, Pos.tape, hh, hp.run]
  exact run_reaches _ _ _

/-- **A trace from a coin node no cut fixes keeps the node's coin**: the tape it ends on is one through the node,
holding there the coin the node's tape holds. -/
theorem RoundsReach.head {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)}
    {o : β × CoinTape g σ} (hr : RoundsReach D M dg J B fin rootB pts N rs p o) {C : Type} {iF : Fintype C}
    {iN : Nonempty C} {k : C → Game Exchange} (hg : p.g' = @Game.coin _ C iF iN k)
    (hK : ∀ i, (p.Ks i).isHere = false) :
    ∃ z, o.2 = p.back z ∧ headCoin hg p.s' z = headCoin hg p.s' p.x := by
  cases hr with
  | zero => exact ⟨p.x, rfl, rfl⟩
  | stop => exact ⟨p.x, Pos.toStop_tape p, rfl⟩
  | round _ _ _ s c _ _ _ ho =>
    obtain ⟨h, hh, hc⟩ := Pos.walk_coin_keeps hg p.s' p.Ks p.x p.back hK
    obtain ⟨q, hq, hoq⟩ := ho.le_tape
    obtain ⟨h₁, hh₁⟩ := ((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq
    exact ⟨h (h₁ q.x), hoq.trans ((hh₁ q.x).trans (hh _)), hc _⟩

/-- **A stop's written release is in the play's log**: at a stop after an opening frame of the `i`-th tree's stream `s`
that the device serves after the log `pre` (`Served`, the trees the log shows `TreesOf`), its releases continuing to the
close after `pre` and them, off the opening's empty class at `bits`, the play on the tape any trace from the stop,
`bits` written into its opening and consumed, ends on has as its log `pre`, then the opening answering `b` with `bits`,
then the rest. -/
theorem RoundsReach.release {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray} {k : ℕ}
    {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release} {i s n : ℕ}
    {b : ByteArray} {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)}
    {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)}
    {Ks : Fin R → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?)}
    {x'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s''}
    {back'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'' → CoinTape g σ}
    (hq : q = ⟨optOpenK k fs ts f i s n b ts[i]?, s'', Ks, x'', back''⟩) (hi : q.Inv Ks₀)
    (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log)) (hT : TreesOf pre ts) (hS : Served pre i s n)
    {bits : List Bool} (hE : ¬ EmptyAt (finish sg θ F c rq) q.g' bits) {β : Type} {fin : LiveOutJ J mPts → β}
    {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round}
    {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite (writeOpening (finish sg θ F c rq) q.s' bits)).consume o) :
    ∃ L, (CoinTape.run g σ o.2).log = pre ++ ⟨.opening i s n, b, bits⟩ :: L := by
  subst hq
  obtain ⟨T, hTi, _, _, hp, -⟩ := Served.next hT hS
  have hc := Served.cap hT hS
  obtain ⟨z, hz, hh⟩ := hr.head (OpeningAt.ofNext (k := k) (b := b) hf hc hTi hp).eq (Pos.isHere_consume _)
  have hw := opened_of_head (finish_recordsLog sg θ F c rq) hf hc hTi hp hE x'' z hh
  have hX : CoinTape.run g σ o.2 = CoinTape.run (optOpenK k fs ts f i s n b ts[i]?) s'' z := by
    rw [hz]
    exact hi.run z
  obtain ⟨L, hL⟩ := log_of_opening hf hc hTi hp s'' z
  refine ⟨L, ?_⟩
  rw [hX, hL, hw]

/-- **A trace from a play's end ends there**: from a position that keeps the invariant at the play's end `y`, any
rewriting of its tape consumed, the play on the tape the trace ends on is `y`. -/
theorem RoundsReach.run_ret {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀)
    {y : Exchange} (hy : q.g' = .ret y) {φ : CoinTape q.g' q.s' → CoinTape q.g' q.s'} {β : Type}
    {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite φ).consume o) : CoinTape.run g σ o.2 = y := by
  obtain ⟨q', hq', ho⟩ := hr.le_tape
  obtain ⟨h, hh⟩ := ((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq'
  have key : ∀ p : Pos g σ Claims.Msg (Fin R), p.g' = .ret y → ∀ t, CoinTape.run p.g' p.s' t = y :=
    fun ⟨_, _, _, _, _⟩ hg t => by
      subst hg
      rfl
  rw [ho]
  show CoinTape.run g σ (q'.back q'.x) = y
  rw [hh, hi.run]
  exact key q hy _

/-- **A trace's answered stops**: `RoundsReach.out`'s list, at each answered submission a position the trace passes
through that keeps any invariant `Q` its start keeps and each consumed stop passes on, the submission decoded at the
stop after it, its coins, which `B` doesn't hold of there, and the rest of the trace from the stop with them written in
and consumed. -/
theorem RoundsReach.answers {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)}
    {o : β × CoinTape g σ} (hr : RoundsReach D M dg J B fin rootB pts N rs p o)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) :
    ∃ as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))),
      o.1 = fin ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
          D.proofs (CoinTape.run g σ o.2)⟩ ∧
      ∀ a ∈ as, Q a.1 ∧ D.sub J a.1.toStop.modes = some a.2.1 ∧ ¬ B a.1.toStop.g' J a.2.1 a.2.2 ∧
        ∃ N' rs', RoundsReach D M dg J B fin rootB pts N' rs'
          (a.1.toStop.rewrite (M.round a.1.toStop J a.2.1 a.2.2)).consume o := by
  induction hr with
  | zero rs p => exact ⟨[], rfl, fun _ h => by simp at h⟩
  | stop N rs p _ => exact ⟨[], rfl, fun _ h => by simp at h⟩
  | round N rs p s c o hs hB ho ih =>
    obtain ⟨as, h1, h2⟩ := ih (hstep p _ hQ)
    refine ⟨(p, ⟨s, c⟩) :: as, h1, fun a ha => ?_⟩
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨hQ, hs, hB, N, _, ho⟩
    · exact h2 a ha

/-- **An answered stop's modes are the cuts' on the trace's tape**: at a position that keeps the invariant at the
cuts `Ks₀`, where the rest of a trace runs from the stop after it, its tape rewritten and consumed, each cut fixed at
the stop has, on the tape the trace ends on, the most likely message the stop decoded; every other cut decoded none
there (`Pos.modes_eq`, as `RoundsReach.toStop_modes` at a trace's first stop). -/
theorem RoundsReach.stop_modes {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {Ks₀ : Fin R → DigestCut Claims.Msg g} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {q : Pos g σ Claims.Msg (Fin R)} (hq : q.Inv Ks₀)
    {φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.toStop.rewrite φ).consume o) :
    q.toStop.modes = fun i => if (q.toStop.Ks i).isHere then (Ks₀ i).modeOn σ o.2 else none := by
  obtain ⟨_, _, hi⟩ := Pos.toStop_spec hq
  obtain ⟨q', hq', _, hoq⟩ := hr.through (Pos.inv_consume (Pos.inv_rewrite hi _))
  rw [hoq]
  exact Pos.modes_eq hi (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq') _

/-- **What a cut fixed at an answered stop reveals on the trace's tape**: at a position that keeps the invariant at
the cuts `Ks₀`, where the rest of a trace runs from the stop after it, its tape rewritten and consumed, a cut fixed at
the stop at the reading `rv` reveals, on the tape the trace ends on, `rv` of the play on it (`Pos.revealOn_here`). -/
theorem RoundsReach.stop_reveal {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {Ks₀ : Fin R → DigestCut Claims.Msg g} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {q : Pos g σ Claims.Msg (Fin R)} (hq : q.Inv Ks₀)
    {φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.toStop.rewrite φ).consume o) {i : Fin R}
    {rv : Exchange → Option Claims.Msg} (hi : q.toStop.Ks i = .here rv) :
    (Ks₀ i).revealOn σ o.2 = rv (CoinTape.run g σ o.2) := by
  obtain ⟨_, _, hi'⟩ := Pos.toStop_spec hq
  obtain ⟨q', hq', _, hoq⟩ := hr.through (Pos.inv_consume (Pos.inv_rewrite hi' _))
  rw [hoq]
  exact Pos.revealOn_here hi' (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans hq') hi _

end Trace

/-! ## Where each stop of a trace is -/

section StopsAt

variable {g : Game Exchange} {σ : Strategy g} {R : ℕ}

/-- **Where a position's walk stops, at a log and the close after it**: as `StopOf` says, from some log `pre`, at
releases continuing to the close under the signer `sg` after the request `rq` and `pre` and them. -/
def StopsAt (P : Fin R → List Release → Frame → Prop) (rv : Fin R → Exchange → Option Claims.Msg) (sg : Signer)
    (θ : ByteArray) (F : Array ByteArray) (c rq : ByteArray) (q : Pos g σ Claims.Msg (Fin R)) : Prop :=
  ∃ (pre : List Release) (f : List Release → Game Exchange) (a : Fin R → Prop),
    (∀ log, f log = finish sg θ F c rq (pre ++ log)) ∧ StopOf P rv pre f a q

/-- **The walk from a stop, consumed, stops at a log and the close after it**, on any rewriting of the tape there,
where each round's reading picks only openings. -/
theorem StopsAt.next {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n) {q : Pos g σ Claims.Msg (Fin R)}
    (h : StopsAt P rv sg θ F c rq q) (φ : CoinTape q.g' q.s' → CoinTape q.g' q.s') :
    StopsAt P rv sg θ F c rq (q.rewrite φ).consume.toStop := by
  obtain ⟨pre, f, a, hf, L, hL, h⟩ := h
  rcases h with ⟨hK, y, hy, s₀, x₀, hrun⟩ |
    ⟨k, fs, ts, f', i, s, n, b, hf', hT, -, s'', Ks, x'', back'', hq, hKs⟩
  · obtain ⟨g', s', Ks, x, back⟩ := q
    obtain rfl : g' = .ret y := hy
    refine ⟨pre, f, a, hf, L, hL, Or.inl ⟨fun j => ?_, y, rfl, s₀, x₀, hrun⟩⟩
    have hKj : Ks j = .skip := hK j
    show (if (Ks j).isHere then (.skip : DigestCut Claims.Msg (Game.ret y)) else Ks j) = .skip
    rw [hKj, ite_self]
  · refine ⟨pre ++ L, f', _, fun log => ?_, stopOf_next hP s'' Ks x'' back'' hq hT hKs φ⟩
    rw [hf', hf, List.append_assoc]

end StopsAt

/-! ## Each stop's written coins in the log -/

section Logged

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} {D : Decoders R Reg}
  {M : CoinMap g σ Ω R mPts} {dg : ByteArray → ByteArray} {J : ℕ} {B : RoundHold}

/-- **The bits `bits` released in the log `log` after the log `pre`, at an opening the rounds `q` fixes pick**: an
opening of the `i`-th tree's stream `s` that round `j`, whose cut `q` fixes, picks after `pre`, as does every round
whose cut `q` fixes, answered with `bits`. -/
def LoggedAfter (P : Fin R → List Release → Frame → Prop) (q : Pos g σ Claims.Msg (Fin R)) (pre : List Release)
    (bits : List Bool) (log : List Release) : Prop :=
  ∃ (j : Fin R) (i s n : ℕ) (b : ByteArray) (L : List Release),
    (q.Ks j).isHere = true ∧ (∀ j', (q.Ks j').isHere = true → P j' pre (.opening i s n)) ∧
      P j pre (.opening i s n) ∧ log = pre ++ ⟨.opening i s n, b, bits⟩ :: L

/-- **The bits `bits` released in the log `log` at an opening the rounds `q` fixes pick**, after some log
(`LoggedAfter`). -/
def LoggedAt (P : Fin R → List Release → Frame → Prop) (q : Pos g σ Claims.Msg (Fin R)) (bits : List Bool)
    (log : List Release) : Prop :=
  ∃ pre, LoggedAfter P q pre bits log

/-- **The cuts `q` fixes are at the bytes the release of the log `log` after the log `pre` answers**: each is fixed
there at its round's revealed messages that open those bytes. -/
def CutsAt (rv : Fin R → Exchange → Option Claims.Msg) (q : Pos g σ Claims.Msg (Fin R)) (pre log : List Release) :
    Prop :=
  ∀ j r L, (q.Ks j).isHere = true → log = pre ++ r :: L → q.Ks j = .here (digestReveal (rv j) r.answers.data.toList)

open Classical in
/-- **A round's cut after an opening frame its reading picks** is fixed there, at its reading's messages that open
the bytes the frame answers. -/
theorem openCut_pos {β : Type} {P : List Release → Frame → Prop} {rv : β → Option Claims.Msg} {k : ℕ}
    {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β} {pre : List Release} {i s n : ℕ}
    {b : ByteArray} (h : P pre (.opening i s n)) (o : Option OpenTree) :
    openCut P rv k fs ts f pre i s n b o = .here (digestReveal rv b.data.toList) := by
  unfold openCut
  rw [ite_eq_left h]
  rfl

/-- **At a stop after an opening frame, the cuts fixed there are at the bytes the frame answers** (`CutsAt`), in a
log with that opening after the log `pre`. -/
theorem cutsAt_openCuts {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release}
    {i s n : ℕ} {b : ByteArray} {a : Fin R → Prop} {q : Pos g σ Claims.Msg (Fin R)}
    {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)}
    {Ks : Fin R → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?)}
    {x'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s''}
    {back'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'' → CoinTape g σ}
    (hq : q = ⟨optOpenK k fs ts f i s n b ts[i]?, s'', Ks, x'', back''⟩)
    (hKs : DigestCut.LiveCuts a (fun j => openCut (P j) (rv j) k fs ts f pre i s n b ts[i]?) Ks) {bits : List Bool}
    {L log : List Release} (hL : log = pre ++ ⟨.opening i s n, b, bits⟩ :: L) : CutsAt rv q pre log := by
  intro j r L₂ hj he
  rw [hL] at he
  obtain ⟨rfl, -⟩ := List.cons.inj (List.append_cancel_left he)
  subst hq
  obtain ⟨ha, hp⟩ := (isHere_openCuts hKs j).1 hj
  exact (hKs.eq_of ha).trans (openCut_pos hp _)

/-- **A stop's written coins are in the play's log, at an opening its fixed rounds pick**: at a position that keeps
the invariant at the cuts `Ks₀`, where `StopsAt` says and not every cut is past, where each round's reading picks only
openings the device serves, off the opening's empty class at `bits`, the play on the tape any trace from the position,
`bits` written into its opening and consumed, ends on releases `bits` at an opening the rounds whose cuts the position
fixes pick (`LoggedAt`). -/
theorem RoundsReach.stop_logged {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {sg : Signer} {θ : ByteArray} {F : Array ByteArray}
    {c rq : ByteArray} (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n)
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀)
    (hS : StopsAt P rv sg θ F c rq q) (hK : ¬ ∀ j, q.Ks j = .skip) {bits : List Bool}
    (hE : ¬ EmptyAt (finish sg θ F c rq) q.g' bits) {β : Type} {fin : LiveOutJ J mPts → β}
    {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round}
    {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite (writeOpening (finish sg θ F c rq) q.s' bits)).consume o) :
    LoggedAt P q bits (CoinTape.run g σ o.2).log := by
  obtain ⟨pre, f, a, hf, L, -, h⟩ := hS
  rcases h with ⟨hK', -⟩ | ⟨k, fs, ts, f', i, s, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · exact absurd hK' hK
  · have hjK : (q.Ks j).isHere = true := by
      rw [hq]
      exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
    have hall : ∀ j', (q.Ks j').isHere = true → P j' (pre ++ L) (.opening i s n) := fun j' h' => by
      rw [hq] at h'
      exact ((isHere_openCuts hKs j').1 h').2
    obtain ⟨L', hL'⟩ := RoundsReach.release (sg := sg) (θ := θ) (F := F) (c := c) (rq := rq) hq hi
      (fun log => by rw [hf', hf, List.append_assoc]) hT (hPS j _ i s n hj) hE hr
    exact ⟨pre ++ L, j, i, s, n, b, L', hjK, hall, hj, hL'⟩

/-- **Each answered stop's coins are in the play's log**: on a trace from a position that keeps the invariant at the
cuts `Ks₀` and whose walk stops where `StopsAt` says, where each round's reading picks only openings the device serves,
the decoders submit nothing at a stop that fixes no cut, the coin map writes a round's coins into the stop's opening as
their bits `bitsOf`, and off `B` the opening's class at those bits is not empty, the play on the tape the trace ends on
releases, at each answered stop (`RoundsReach.answers`), the bits of the stop's round's coins at an opening the rounds
whose cuts the stop fixes pick (`LoggedAt`, `RoundsReach.stop_logged`). -/
theorem RoundsReach.answers_logged {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {sg : Signer} {θ : ByteArray} {F : Array ByteArray}
    {c rq : ByteArray} (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) (hsub : D.sub J (fun _ => none) = none)
    (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool)
    (hM : ∀ q s x, M.round q J s x = writeOpening (finish sg θ F c rq) q.s' (bitsOf s x))
    (hB : ∀ g' s x, ¬ B g' J s x → ¬ EmptyAt (finish sg θ F c rq) g' (bitsOf s x))
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)}
    {o : β × CoinTape g σ} (hr : RoundsReach D M dg J B fin rootB pts N rs p o) (hi : p.Inv Ks₀)
    (hS : StopsAt P rv sg θ F c rq p.toStop) :
    ∃ as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))),
      o.1 = fin ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
          D.proofs (CoinTape.run g σ o.2)⟩ ∧
      ∀ a ∈ as, D.sub J a.1.toStop.modes = some a.2.1 ∧
        LoggedAt P a.1.toStop (bitsOf a.2.1 a.2.2) (CoinTape.run g σ o.2).log := by
  obtain ⟨as, h1, h2⟩ := hr.answers (fun q => q.Inv Ks₀ ∧ StopsAt P rv sg θ F c rq q.toStop) ⟨hi, hS⟩
    fun q φ hq => ⟨Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hq.1).2.2 φ), StopsAt.next hP hq.2 φ⟩
  refine ⟨as, h1, fun a ha => ?_⟩
  obtain ⟨⟨hai, has⟩, hsa, hBa, _, _, hra⟩ := h2 a ha
  refine ⟨hsa, ?_⟩
  rw [hM] at hra
  refine RoundsReach.stop_logged hPS (Pos.toStop_spec hai).2.2 has (fun hK => ?_) (hB _ _ _ hBa) hra
  have hm : a.1.toStop.modes = fun _ => none := funext fun j =>
    (congrArg (fun K => DigestCut.modeHere K a.1.toStop.s') (hK j)).trans rfl
  rw [hm, hsub] at hsa
  cases hsa

end Logged

/-! ## One round a stop -/

/-- **A submission where one round decodes is that round's slot's**: where every round but `j` decodes none, a
submission the decoders at the layout `L` decode is round `j`'s, a rep round of a table before `J`, with its most
likely message. -/
theorem sub_of_single {R : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) (L : Fin R → Slot) (J : ℕ)
    {modes : Fin R → Option Claims.Msg} {j : Fin R} (hs : ∀ j', j' ≠ j → modes j' = none) {sub : SubmitJ J}
    (h : (decodersAt reg L).sub J modes = some sub) :
    ∃ (t : ℕ) (ht : t < J) (r : Fin 2) (k : ℕ) (m : Claims.Msg), L j = .rep t r k ∧ modes j = some m ∧
      sub = ⟨⟨t, ht⟩, r, ⟨m.toArray⟩, tailCount m⟩ := by
  have hf : ∀ j', j' ≠ j → subAt J L modes j' = none := fun j' hj => by
    rw [subAt, hs j' hj, Option.bind_none]
  have e : (decodersAt reg L).sub J modes = subAt J L modes j := findSome?_of_single hf (List.mem_finRange j)
  rw [e] at h
  unfold subAt at h
  cases hm : modes j with
  | none =>
    rw [hm, Option.bind_none] at h
    cases h
  | some m =>
    rw [hm, Option.bind_some] at h
    cases hL : L j with
    | link =>
      rw [hL] at h
      cases h
    | rep t r k =>
      rw [hL] at h
      have h' : (if h : t < J then some (⟨⟨t, h⟩, r, ⟨m.toArray⟩, tailCount m⟩ : SubmitJ J) else none) = some sub :=
        h
      by_cases ht : t < J
      · rw [dite_eq_left ht] at h'
        exact ⟨t, ht, r, k, m, rfl, rfl, (Option.some.inj h').symm⟩
      · simp [ht] at h'

/-- **A stop answers its one fixed round's slot**, at the readings and the decoders at the layout `L`, injective, of
distinct tables: where every round whose cut the stop `q` fixes picks one opening after the log `pre`, as round `j`
does, a submission decoded at `q` is round `j`'s, round `k` of table `t`'s rep `r` stream, `t < J`, with its most
likely message. -/
theorem stop_slot {R J : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) {tables : Array String}
    {L : Fin R → Slot} (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {sub : SubmitJ J} (hsub : (decodersAt reg L).sub J q.modes = some sub) {j : Fin R} {pre : List Release}
    {i s n : ℕ}
    (hall : ∀ j', (q.Ks j').isHere = true → (readingAt tables L record recordDigest).round j' pre (.opening i s n))
    (hP : (readingAt tables L record recordDigest).round j pre (.opening i s n)) :
    ∃ (t : ℕ) (ht : t < J) (r : Fin 2) (k : ℕ) (m : Claims.Msg), L j = .rep t r k ∧ q.modes j = some m ∧
      sub = ⟨⟨t, ht⟩, r, ⟨m.toArray⟩, tailCount m⟩ :=
  sub_of_single reg L J (modes := q.modes) (fun j' hj' => Pos.modes_eq_none (p := q) (i := j') (by
    cases h : (q.Ks j').isHere
    · rfl
    · exact absurd (readingAt_unique hL ht record recordDigest (hall j' h) hP) hj')) hsub

/-- **A stop's logged bits are at its slot's opening**, at the readings and the decoders at the layout `L`, injective,
of distinct tables: where the decoders submit `sub` at the stop `q` and `bits` are released after the log `pre` at an
opening its fixed rounds pick (`LoggedAfter`), `sub` is round `k` of table `t`'s rep `r` stream with the most likely
message of round `j`, whose slot that is and whose cut `q` fixes, and `pre`'s first opening is of tree `i`, the opening
is of that tree's stream named table `t`'s rep `r`, and `pre` has opened that stream `k` times. -/
theorem LoggedAfter.slot {R J : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) {tables : Array String}
    {L : Fin R → Slot} (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {sub : SubmitJ J} (hsub : (decodersAt reg L).sub J q.modes = some sub) {pre : List Release} {bits : List Bool}
    {log : List Release} (h : LoggedAfter (readingAt tables L record recordDigest).round q pre bits log) :
    ∃ (t : ℕ) (ht : t < J) (r : Fin 2) (k : ℕ) (m : Claims.Msg) (j : Fin R) (i s n : ℕ) (b : ByteArray)
      (L' : List Release), L j = .rep t r k ∧ (q.Ks j).isHere = true ∧ q.modes j = some m ∧
        sub = ⟨⟨t, ht⟩, r, ⟨m.toArray⟩, tailCount m⟩ ∧ firstOpened pre = some i ∧
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = tables[t]?.map (repStreamName · r.val) ∧
        openedOf pre i s = k ∧ log = pre ++ ⟨.opening i s n, b, bits⟩ :: L' := by
  obtain ⟨j, i, s, n, b, L', hj, hall, hP, he⟩ := h
  obtain ⟨t, ht', r, k, m, hLj, hm, hs⟩ := stop_slot reg hL ht record recordDigest hsub hall hP
  have hP' : slotRound tables (L j) pre (.opening i s n) := hP
  rw [hLj] at hP'
  obtain ⟨i', s', n', hfr, -, hfo, hname, hk⟩ := hP'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  exact ⟨t, ht', r, k, m, j, _, _, _, b, L', hLj, hj, hm, hs, hfo, hname, hk, he⟩

/-! ## The trace's answered stops, in order in the log -/

/-- **A release of `pre ++ (L ++ L₃)` past `pre` and before `L₃` is one of `L`'s.** -/
theorem mid_of_append {α : Type} {pre L L₃ L₁ L₂ : List α} {r : α} (he : L₁ ++ r :: L₂ = pre ++ (L ++ L₃))
    (h₁ : pre.length ≤ L₁.length) (h₂ : L₁.length < pre.length + L.length) :
    ∃ c d, L₁ = pre ++ c ∧ L = c ++ r :: d := by
  rcases List.append_eq_append_iff.1 he with ⟨a', hpre, hr⟩ | ⟨c', hL₁, hr⟩
  · have ha : a' = [] := by
      have hl := congrArg List.length hpre
      rw [List.length_append] at hl
      exact List.eq_nil_of_length_eq_zero (by omega)
    subst ha
    rw [List.append_nil] at hpre
    subst hpre
    cases L with
    | nil => simp at h₂
    | cons x d =>
      rw [List.nil_append, List.cons_append] at hr
      obtain ⟨rfl, -⟩ := List.cons.inj hr
      exact ⟨[], d, (List.append_nil _).symm, rfl⟩
  · subst hL₁
    rw [List.length_append] at h₂
    rcases List.append_eq_append_iff.1 hr with ⟨a'', hc, -⟩ | ⟨c'', hL, hr'⟩
    · subst hc
      rw [List.length_append] at h₂
      omega
    · cases c'' with
      | nil =>
        rw [List.append_nil] at hL
        subst hL
        omega
      | cons x d =>
        rw [List.cons_append] at hr'
        obtain ⟨rfl, -⟩ := List.cons.inj hr'
        exact ⟨c', d, rfl, hL⟩

/-- **Of two splits of one list, the one with the longer front extends the other's front.** -/
theorem prefix_le {α : Type} {L₁ L₀ R₁ R₀ : List α} (h : L₁ ++ R₁ = L₀ ++ R₀) (hle : L₀.length ≤ L₁.length) :
    ∃ d, L₁ = L₀ ++ d := by
  rcases List.append_eq_append_iff.1 h with ⟨a', h₀, -⟩ | ⟨c', h₁, -⟩
  · have hl := congrArg List.length h₀
    rw [List.length_append] at hl
    have ha : a' = [] := List.eq_nil_of_length_eq_zero (by omega)
    subst ha
    exact ⟨[], by rw [h₀, List.append_nil, List.append_nil]⟩
  · exact ⟨c', h₁⟩

/-- **A log through an opening has a first opening.** -/
theorem firstOpened_ne_none (L₀ : List Release) {r : Release} {i s n : ℕ} (hr : r.frame = .opening i s n)
    (d : List Release) : firstOpened (L₀ ++ r :: d) ≠ none := fun h => by
  have h' := List.findSome?_eq_none_iff.1 h r (List.mem_append_right _ (List.mem_cons_self ..))
  simp [hr] at h'

/-- **The readings at a layout pick only openings the device serves.** -/
theorem readingAt_served {R : ℕ} (tables : Array String) (L : Fin R → Slot) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) :
    ∀ j pre i s n, (readingAt tables L record recordDigest).round j pre (.opening i s n) → Served pre i s n := by
  intro j pre i s n h
  have h' : slotRound tables (L j) pre (.opening i s n) := h
  cases hL : L j with
  | link =>
    rw [hL] at h'
    obtain ⟨_, _, _, hfr, hS, -⟩ := h'
    obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
    exact hS
  | rep t r k =>
    rw [hL] at h'
    obtain ⟨_, _, _, hfr, hS, -⟩ := h'
    obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
    exact hS

/-- **Each element of a list a `Forall₂` relates to another is beside one of the other's in their zip.** -/
theorem forall₂_zip_right {α : Type _} {β : Type _} {Rel : α → β → Prop} {l₁ : List α} {l₂ : List β}
    (h : List.Forall₂ Rel l₁ l₂) {b : β} (hb : b ∈ l₂) : ∃ a, (a, b) ∈ l₁.zip l₂ := by
  induction h with
  | nil => exact absurd hb List.not_mem_nil
  | @cons a b' l₁ l₂ _ _ ih =>
    rw [List.zip_cons_cons]
    rcases List.mem_cons.1 hb with e | hb
    · rw [e]
      exact ⟨a, List.mem_cons_self ..⟩
    · obtain ⟨a', ha'⟩ := ih hb
      exact ⟨a', List.mem_cons_of_mem _ ha'⟩

section Order

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} {D : Decoders R Reg}
  {M : CoinMap g σ Ω R mPts} {dg : ByteArray → ByteArray} {J : ℕ} {B : RoundHold}

/-- **Round `j` picks the release of the log `Lg` after the log `L₁`.** -/
def PicksIn (P : Fin R → List Release → Frame → Prop) (Lg : List Release) (j : Fin R) (L₁ : List Release) : Prop :=
  ∃ r L₂, Lg = L₁ ++ r :: L₂ ∧ P j L₁ r.frame

/-- **Each round picks at most one release of the log `Lg`.** -/
def PicksOnce (P : Fin R → List Release → Frame → Prop) (Lg : List Release) : Prop :=
  ∀ j L₁ L₁', PicksIn P Lg j L₁ → PicksIn P Lg j L₁' → L₁ = L₁'

/-- **A stop's release is after one log**, where each round picks at most one release of the log. -/
theorem LoggedAfter.pre_eq {P : Fin R → List Release → Frame → Prop} {Lg : List Release} (hU : PicksOnce P Lg)
    {q : Pos g σ Claims.Msg (Fin R)} {pre pre' : List Release} {bits bits' : List Bool}
    (h : LoggedAfter P q pre bits Lg) (h' : LoggedAfter P q pre' bits' Lg) : pre = pre' := by
  obtain ⟨j, i, s, n, b, L, hj, -, hp, he⟩ := h
  obtain ⟨-, i', s', n', b', L', -, hall', -, he'⟩ := h'
  exact hU j pre pre' ⟨_, L, he, hp⟩ ⟨_, L', he', hall' j hj⟩

/-- **What a trace has answered by a stop**, in the log `Lg`, with the releases after the logs `done` answered: `Lg`
has a release after the log `pre`, an answered one; each round not live in `a` picks after an answered log; each round
picking before `pre` picks after an answered log; and no live round picks after one. -/
structure AnsweredBy (P : Fin R → List Release → Frame → Prop) (Lg pre : List Release) (a : Fin R → Prop)
    (done : List (List Release)) : Prop where
  pre_mem : pre ∈ done
  pre_log : ∃ r L₂, Lg = pre ++ r :: L₂
  past : ∀ j, ¬ a j → ∃ L₁ ∈ done, PicksIn P Lg j L₁
  before : ∀ j L₁, PicksIn P Lg j L₁ → L₁.length < pre.length → L₁ ∈ done
  live : ∀ j L₁, L₁ ∈ done → PicksIn P Lg j L₁ → ¬ a j

/-- **A pick before the end of a segment no live round picks in is after an answered log**: where each round picks at
most one release of `Lg`, `Lg` is `pre`, the segment `L` and the rest, and no live round picks in `L` after `pre`. -/
theorem AnsweredBy.seg {P : Fin R → List Release → Frame → Prop} {Lg pre : List Release} {a : Fin R → Prop}
    {done : List (List Release)} (hA : AnsweredBy P Lg pre a done) (hU : PicksOnce P Lg) {L L₃ : List Release}
    (hNP : NoPick P a pre L) (hLg : Lg = pre ++ (L ++ L₃)) {j : Fin R} {L₁ : List Release}
    (hp : PicksIn P Lg j L₁) (h₂ : L₁.length < pre.length + L.length) : L₁ ∈ done := by
  by_cases h₁ : pre.length ≤ L₁.length
  · obtain ⟨r, L₂, he, hP⟩ := id hp
    obtain ⟨c', d, rfl, hL⟩ := mid_of_append (he.symm.trans hLg) h₁ h₂
    by_cases ha : a j
    · exact absurd hP (hNP j ha c' r d hL)
    · obtain ⟨L₀, hL₀, hp₀⟩ := hA.past j ha
      rw [hU j _ L₀ hp hp₀]
      exact hL₀
  · exact hA.before j L₁ hp (by omega)

/-- **The play on a position's tape at a play's end is that end**, at a position that keeps the invariant. -/
theorem run_tape_ret {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀)
    {y : Exchange} (hy : q.g' = .ret y) : CoinTape.run g σ q.tape = y := by
  have key : ∀ p : Pos g σ Claims.Msg (Fin R), p.g' = .ret y → ∀ t, CoinTape.run p.g' p.s' t = y :=
    fun ⟨_, _, _, _, _⟩ hg t => by
      subst hg
      rfl
  exact (hi.run q.x).trans (key q hy _)

/-- **The play on a stop's tape logs the stop's opening**: at a position that keeps the invariant, after an opening
frame of the `i`-th tree's stream `s` that the device serves after the log `pre`, its releases continuing to the
close after `pre` and them, the play on the position's tape has as its log `pre`, then that opening, then the rest. -/
theorem opening_log {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray} {k : ℕ}
    {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release} {i s n : ℕ}
    {b : ByteArray} {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)}
    {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)}
    {Ks : Fin R → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?)}
    {x'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s''}
    {back'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'' → CoinTape g σ}
    (hq : q = ⟨optOpenK k fs ts f i s n b ts[i]?, s'', Ks, x'', back''⟩) (hi : q.Inv Ks₀)
    (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log)) (hT : TreesOf pre ts) (hS : Served pre i s n) :
    ∃ (bits : List Bool) (L : List Release),
      (CoinTape.run g σ q.tape).log = pre ++ ⟨.opening i s n, b, bits⟩ :: L := by
  subst hq
  obtain ⟨T, hTi, _, _, hp, -⟩ := Served.next hT hS
  obtain ⟨L, hL⟩ := log_of_opening hf (Served.cap hT hS) hTi hp s'' x''
  have hX : CoinTape.run g σ (back'' x'') = CoinTape.run (optOpenK k fs ts f i s n b ts[i]?) s'' x'' := hi.run x''
  exact ⟨_, L, (congrArg Exchange.log hX).trans hL⟩

/-- **A trace's answered stops, in order in the log**: on a trace from a position that keeps the invariant at the
cuts `Ks₀` and whose walk stops where `StopOf` says after the log `pre` at the live rounds `a`, with the releases after
the logs `done` answered (`AnsweredBy`) in the log `Lg` of the play on the tape the trace ends on, where each round
picks at most one release of `Lg` and only openings the device serves, the decoders submit nothing at a stop that fixes
no cut, the coin map writes a round's coins into the stop's opening as their bits `bitsOf`, and off `B` the opening's
class at those bits is not empty: the trace's answered stops (`RoundsReach.answers`) each release their round's coins
after a log (`LoggedAfter`), each longer than `pre` and than the one before, the cuts the stop fixes at the bytes that
release answers (`CutsAt`), each at a position that keeps the invariant, the rest of the trace running from the stop
after it; and every release of `Lg` a round picks is after an answered log or one of those, unless
the trace answered `N` rounds or ended at a stop the decoders submit nothing at, whose fixed rounds pick after a log
not answered. -/
theorem RoundsReach.order {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {sg : Signer} {θ : ByteArray} {F : Array ByteArray}
    {c rq : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) (hsub : D.sub J (fun _ => none) = none)
    (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool)
    (hM : ∀ q s x, M.round q J s x = writeOpening (finish sg θ F c rq) q.s' (bitsOf s x))
    (hB : ∀ g' s x, ¬ B g' J s x → ¬ EmptyAt (finish sg θ F c rq) g' (bitsOf s x))
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)}
    {o : β × CoinTape g σ} {Lg : List Release} (hU : PicksOnce P Lg) (hr : RoundsReach D M dg J B fin rootB pts N rs p o) :
    p.Inv Ks₀ → (CoinTape.run g σ o.2).log = Lg →
    ∀ (pre : List Release) (f : List Release → Game Exchange) (a : Fin R → Prop) (done : List (List Release)),
      (∀ log, f log = finish sg θ F c rq (pre ++ log)) → StopOf P rv pre f a p.toStop → AnsweredBy P Lg pre a done →
      ∃ (as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))) (ms : List (List Release)),
        o.1 = fin ⟨rootB, pts, as.foldl (fun rs a =>
          addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
            D.proofs (CoinTape.run g σ o.2)⟩ ∧
        List.Forall₂ (fun a m => D.sub J a.1.toStop.modes = some a.2.1 ∧
          LoggedAfter P a.1.toStop m (bitsOf a.2.1 a.2.2) Lg ∧ CutsAt rv a.1.toStop m Lg) as ms ∧
        (∀ a ∈ as, a.1.Inv Ks₀ ∧ ∃ N' rs', RoundsReach D M dg J B fin rootB pts N' rs'
          (a.1.toStop.rewrite (M.round a.1.toStop J a.2.1 a.2.2)).consume o) ∧
        (∀ m ∈ ms, pre.length < m.length) ∧ ms.Pairwise (fun m m' => m.length < m'.length) ∧
        ((∀ j L₁, PicksIn P Lg j L₁ → L₁ ∈ done ∨ L₁ ∈ ms) ∨ as.length = N ∨
          ∃ (q : Pos g σ Claims.Msg (Fin R)) (L₁ : List Release), D.sub J q.modes = none ∧ L₁ ∉ done ∧
            ∃ j, (q.Ks j).isHere = true ∧ ∀ j', (q.Ks j').isHere = true → PicksIn P Lg j' L₁) := by
  induction hr with
  | zero rs p =>
    intro _ _ pre f a done _ _ _
    exact ⟨[], [], rfl, .nil, fun _ h => by simp at h, fun _ h => by simp at h, .nil, Or.inr (Or.inl rfl)⟩
  | stop N rs p hs =>
    intro hi hLg pre f a done hf hSO hA
    refine ⟨[], [], rfl, .nil, fun _ h => by simp at h, fun _ h => by simp at h, .nil, ?_⟩
    have hi' := (Pos.toStop_spec hi).2.2
    obtain ⟨L, hNP, ⟨-, y, hy, s₀, x₀, hrun⟩ |
      ⟨k, fs, ts, f', i, s₁, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩⟩ := hSO
    · obtain ⟨s₂, h₂⟩ := playReaches_of_eq (hf L) s₀ (run_reaches _ s₀ x₀)
      rw [hrun] at h₂
      have hL : Lg = pre ++ (L ++ []) := by
        rw [← hLg, List.append_nil, ← log_of_finish h₂]
        exact congrArg Exchange.log (run_tape_ret hi' hy)
      refine Or.inl fun j L₁ hp => Or.inl ?_
      obtain ⟨r, L₂, he, -⟩ := id hp
      have hlen := congrArg List.length (he.symm.trans hL)
      simp only [List.length_append, List.length_cons, List.length_nil, Nat.add_zero] at hlen
      exact hA.seg hU hNP hL hp (by omega)
    · have hf'' : ∀ log, f' log = finish sg θ F c rq ((pre ++ L) ++ log) := fun log => by
        rw [hf', hf, List.append_assoc]
      obtain ⟨bits, L', hLo⟩ := opening_log hq hi' hf'' hT (hPS j _ i s₁ n hj)
      have hLo' : Lg = (pre ++ L) ++ ⟨.opening i s₁ n, b, bits⟩ :: L' := hLg.symm.trans hLo
      have hjK : (p.toStop.Ks j).isHere = true := by
        rw [hq]
        exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
      refine Or.inr (Or.inr ⟨p.toStop, pre ++ L, hs, fun hd => hA.live j _ hd ⟨_, L', hLo', hj⟩ haj, j, hjK,
        fun j' h' => ?_⟩)
      rw [hq] at h'
      exact ⟨_, L', hLo', ((isHere_openCuts hKs j').1 h').2⟩
  | round N rs p s cs o hs hBc ho ih =>
    intro hi hLg pre f a done hf hSO hA
    have hi' := (Pos.toStop_spec hi).2.2
    obtain ⟨L, hNP, ⟨hK, -⟩ |
      ⟨k, fs, ts, f', i, s₁, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩⟩ := hSO
    · have hm : p.toStop.modes = fun _ => none := funext fun j =>
        (congrArg (fun K => DigestCut.modeHere K p.toStop.s') (hK j)).trans rfl
      rw [hm, hsub] at hs
      cases hs
    · have hf'' : ∀ log, f' log = finish sg θ F c rq ((pre ++ L) ++ log) := fun log => by
        rw [hf', hf, List.append_assoc]
      have ho' := ho
      rw [hM] at ho'
      obtain ⟨L', hL'⟩ := RoundsReach.release hq hi' hf'' hT (hPS j _ i s₁ n hj) (hB _ _ _ hBc) ho'
      rw [hLg] at hL'
      have hjK : (p.toStop.Ks j).isHere = true := by
        rw [hq]
        exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
      have hall : ∀ j', (p.toStop.Ks j').isHere = true → P j' (pre ++ L) (.opening i s₁ n) := fun j' h' => by
        rw [hq] at h'
        exact ((isHere_openCuts hKs j').1 h').2
      have hpick : ∀ j', PicksIn P Lg j' (pre ++ L) → P j' (pre ++ L) (.opening i s₁ n) := by
        rintro j' ⟨r, L₂, he, hp⟩
        rw [hL'] at he
        obtain ⟨rfl, -⟩ := List.cons.inj (List.append_cancel_left he)
        exact hp
      have hseg : Lg = pre ++ (L ++ (⟨.opening i s₁ n, b, bitsOf s cs⟩ :: L')) := by
        rw [hL', List.append_assoc]
      have hA' : AnsweredBy P Lg (pre ++ L) (fun j => a j ∧ ¬ P j (pre ++ L) (.opening i s₁ n))
          ((pre ++ L) :: done) := by
        refine ⟨List.mem_cons_self .., ⟨_, L', hL'⟩, fun j' hj' => ?_, fun j' L₁ hp hl => ?_,
          fun j' L₁ hd hp => ?_⟩
        · by_cases ha : a j'
          · exact ⟨pre ++ L, List.mem_cons_self .., _, L', hL', not_not.1 fun h => hj' ⟨ha, h⟩⟩
          · obtain ⟨L₀, hL₀, hp₀⟩ := hA.past j' ha
            exact ⟨L₀, List.mem_cons_of_mem _ hL₀, hp₀⟩
        · rw [List.length_append] at hl
          exact List.mem_cons_of_mem _ (hA.seg hU hNP hseg hp hl)
        · rcases List.mem_cons.1 hd with rfl | hd
          · exact fun h => h.2 (hpick j' hp)
          · exact fun h => hA.live j' L₁ hd hp h.1
      obtain ⟨as, ms, h1, hF, hI, hgt, hpw, hcov⟩ := ih (Pos.inv_consume (Pos.inv_rewrite hi' _)) hLg (pre ++ L) f'
        _ ((pre ++ L) :: done) hf'' (stopOf_next hP s'' Ks x'' back'' hq hT hKs _) hA'
      have hLne : pre.length < (pre ++ L).length := by
        rw [List.length_append]
        rcases L with _ | ⟨x, L⟩
        · rw [List.append_nil] at hL' hj
          exact absurd haj (hA.live j pre hA.pre_mem ⟨_, L', hL', hj⟩)
        · simp only [List.length_cons]
          omega
      refine ⟨(p, ⟨s, cs⟩) :: as, (pre ++ L) :: ms, h1,
        .cons ⟨hs, ⟨j, i, s₁, n, b, L', hjK, hall, hj, hL'⟩, cutsAt_openCuts hq hKs hL'⟩ hF,
        fun a ha => ?_, fun m hm => ?_, List.pairwise_cons.2 ⟨hgt, hpw⟩, ?_⟩
      · rcases List.mem_cons.1 ha with rfl | ha
        · exact ⟨hi, N, _, ho⟩
        · exact hI a ha
      · rcases List.mem_cons.1 hm with rfl | hm
        · exact hLne
        · exact hLne.trans (hgt m hm)
      · rcases hcov with hc | hN | ⟨q, L₁, h1', h2', h3'⟩
        · refine Or.inl fun j' L₁ hp => ?_
          rcases hc j' L₁ hp with hd | hm
          · rcases List.mem_cons.1 hd with rfl | hd
            · exact Or.inr (List.mem_cons_self ..)
            · exact Or.inl hd
          · exact Or.inr (List.mem_cons_of_mem _ hm)
        · exact Or.inr (Or.inl (by rw [List.length_cons, hN]))
        · exact Or.inr (Or.inr ⟨q, L₁, h1', fun h => h2' (List.mem_cons_of_mem _ h), h3'⟩)

end Order

/-! ## Each round picks once -/

/-- **A log's first opening stays first** as the log grows. -/
theorem firstOpened_append {L₁ : List Release} {i : ℕ} (h : firstOpened L₁ = some i) (d : List Release) :
    firstOpened (L₁ ++ d) = some i := by
  unfold firstOpened at h ⊢
  rw [List.findSome?_append, h]
  rfl

/-- **A tree's spec stays as the log grows.** -/
theorem helloSpecs_getElem?_append {L₁ : List Release} {i : ℕ} {sp : Flock.CoinTree.Spec}
    (h : (helloSpecs L₁)[i]? = some sp) (d : List Release) : (helloSpecs (L₁ ++ d))[i]? = some sp := by
  rw [Verity.Sampled.helloSpecs_append, List.getElem?_append_left (List.getElem?_eq_some_iff.1 h).1, h]

/-- **A log through an opening of a stream has opened it more often than the log before it.** -/
theorem openedOf_lt (L₁ : List Release) {r : Release} {i s n : ℕ} (hr : r.frame = .opening i s n)
    (d : List Release) : openedOf L₁ i s < openedOf (L₁ ++ r :: d) i s := by
  have hp : r.frame.opensAt i s = true := by simp [hr, Frame.opensAt]
  have e : openedOf (L₁ ++ r :: d) i s = openedOf L₁ i s + (openedOf d i s + 1) := by
    unfold openedOf
    rw [List.countP_append, List.countP_cons, if_pos hp]
  omega

/-- **Each round of the readings at a layout picks at most one release of a log** whose trees each name their streams
distinctly: the link's picks the first opening, and a rep round's picks an opening of the first tree opened, of its
one stream of that name, after a count of that stream's openings that each opening of it raises. -/
theorem picksOnce_readingAt {R : ℕ} (tables : Array String) (L : Fin R → Slot) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {Lg : List Release} (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) :
    PicksOnce (readingAt tables L record recordDigest).round Lg := by
  suffices h : ∀ j L₁ L₁', PicksIn (readingAt tables L record recordDigest).round Lg j L₁ →
      PicksIn (readingAt tables L record recordDigest).round Lg j L₁' → L₁.length < L₁'.length → False by
    intro j L₁ L₁' h₁ h₁'
    rcases Nat.lt_trichotomy L₁.length L₁'.length with hl | hl | hl
    · exact (h j L₁ L₁' h₁ h₁' hl).elim
    · obtain ⟨r, L₂, he, -⟩ := h₁
      obtain ⟨r', L₂', he', -⟩ := h₁'
      exact (List.append_inj (he.symm.trans he') hl).1
    · exact (h j L₁' L₁ h₁' h₁ hl).elim
  intro j L₁ L₁' h₁ h₁' hl
  obtain ⟨r, L₂, he, hp⟩ := h₁
  obtain ⟨r', L₂', he', hp'⟩ := h₁'
  obtain ⟨d, rfl⟩ := prefix_le (he'.symm.trans he) hl.le
  rw [List.append_assoc] at he'
  have he₂ := List.append_cancel_left (he.symm.trans he')
  cases d with
  | nil => simp at hl
  | cons x d =>
    rw [List.cons_append] at he₂
    obtain ⟨hx, -⟩ := List.cons.inj he₂
    subst hx
    have hq : slotRound tables (L j) L₁ r.frame := hp
    have hq' : slotRound tables (L j) (L₁ ++ r :: d) r'.frame := hp'
    cases hLj : L j with
    | link =>
      rw [hLj] at hq hq'
      obtain ⟨i, s, n, hfr, -, -⟩ := hq
      obtain ⟨-, -, -, -, -, hnone⟩ := hq'
      exact firstOpened_ne_none L₁ hfr d hnone
    | rep t q k =>
      rw [hLj] at hq hq'
      obtain ⟨i, s, n, hfr, ⟨sp, hsp, hs, -⟩, hfo, hname, hk⟩ := hq
      obtain ⟨i', s', n', -, -, hfo', hname', hk'⟩ := hq'
      rw [firstOpened_append hfo] at hfo'
      obtain rfl := Option.some.inj hfo'
      rw [helloSpecs_getElem?_append hsp, Option.bind_some] at hname'
      rw [hsp, Option.bind_some] at hname
      have hX : sp.streams.toList[s]? = some sp.streams[s] := by
        rw [Array.getElem?_toList, Array.getElem?_eq_getElem hs]
      have hX' : sp.streams.toList[s']? = some sp.streams[s] := by
        rw [Array.getElem?_toList, hname', ← hname, Array.getElem?_eq_getElem hs]
      have hmem : sp ∈ helloSpecs Lg := by
        rw [he]
        exact List.mem_of_getElem? (helloSpecs_getElem?_append hsp _)
      obtain rfl := nodup_getElem?_inj (hn sp hmem) hX hX'
      have := openedOf_lt L₁ hfr d
      omega

/-- **Picks after as many logs as the rounds but one are every pick**: where each round picks at most one release of
`Lg`, round `j₀` picks the release after `L₀`, and some round picks the release after each of the distinct logs `ms`,
each longer than `L₀` and at least as many as the rounds but one, every release a round picks is after `L₀` or after
one of `ms`. -/
theorem covers_of_length {R : ℕ} {P : Fin R → List Release → Frame → Prop} {Lg : List Release}
    (hU : PicksOnce P Lg) {L₀ : List Release} {j₀ : Fin R} (h₀ : PicksIn P Lg j₀ L₀) {ms : List (List Release)}
    (hpick : ∀ m ∈ ms, ∃ j, PicksIn P Lg j m) (hgt : ∀ m ∈ ms, L₀.length < m.length) (hnd : ms.Nodup)
    (hlen : R ≤ ms.length + 1) : ∀ j L₁, PicksIn P Lg j L₁ → L₁ = L₀ ∨ L₁ ∈ ms := by
  classical
  intro j L₁ hp
  by_contra hne
  rw [not_or] at hne
  let g : List Release → Fin R := fun m => if h : ∃ j, PicksIn P Lg j m then h.choose else j₀
  have hg : ∀ m ∈ ms, PicksIn P Lg (g m) m := fun m hm => by
    have h := hpick m hm
    show PicksIn P Lg (if h : ∃ j, PicksIn P Lg j m then h.choose else j₀) m
    rw [dif_pos h]
    exact h.choose_spec
  have hnd' : (j₀ :: j :: ms.map g).Nodup := by
    refine List.nodup_cons.2 ⟨fun h => ?_, List.nodup_cons.2 ⟨fun h => ?_, List.Nodup.map_on
      (fun x hx y hy e => hU (g y) x y (e ▸ hg x hx) (hg y hy)) hnd⟩⟩
    · rcases List.mem_cons.1 h with h | h
      · exact hne.1 (hU j L₁ L₀ hp (h ▸ h₀))
      · obtain ⟨m, hm, e⟩ := List.mem_map.1 h
        have h' := hU j₀ m L₀ (e ▸ hg m hm) h₀
        have h₂ := hgt m hm
        rw [h'] at h₂
        exact lt_irrefl _ h₂
    · obtain ⟨m, hm, e⟩ := List.mem_map.1 h
      have h' := hU j m L₁ (e ▸ hg m hm) hp
      exact hne.2 (h' ▸ hm)
  have hc := hnd'.length_le_card
  simp only [List.length_cons, List.length_map, Fintype.card_fin] at hc
  omega

/-! ## The rounds a trace's submissions append -/

/-- **The rounds a list of submissions appends to a stream**: the stream's rounds before, then the rounds of the
submissions on it, in order. -/
theorem foldl_addRoundJ {J : ℕ} {γ : Type _} (tab : γ → Fin J) (rep : γ → Fin 2) (rd : γ → Flock.Round) :
    ∀ (as : List γ) (rs₀ : Fin J → Fin 2 → List Flock.Round) (t : Fin J) (r : Fin 2),
      as.foldl (fun rs a => addRoundJ rs (tab a) (rep a) (rd a)) rs₀ t r =
        rs₀ t r ++ (as.filter fun a => tab a = t ∧ rep a = r).map rd
  | [], _, _, _ => by simp
  | a :: as, rs₀, t, r => by
    rw [List.foldl_cons, foldl_addRoundJ tab rep rd as]
    by_cases h : tab a = t ∧ rep a = r
    · obtain ⟨rfl, rfl⟩ := h
      have hp : decide (tab a = tab a ∧ rep a = rep a) = true := by simp
      rw [addRoundJ_self, List.filter_cons, if_pos hp, List.map_cons, List.append_assoc, List.singleton_append]
    · have hp : ¬ (decide (tab a = t ∧ rep a = r) = true) := by simpa using h
      rw [addRoundJ_ne _ _ (fun h' => h ⟨h'.1.symm, h'.2.symm⟩), List.filter_cons, if_neg hp]

/-! ## The logs before a stream's openings -/

/-- **The logs before each element of `l` that `q` holds of**, in order. -/
def prefixesWhere {α : Type} (q : α → Bool) : List α → List (List α)
  | [] => []
  | x :: xs => (if q x then [[]] else []) ++ (prefixesWhere q xs).map (x :: ·)

theorem length_prefixesWhere {α : Type} (q : α → Bool) :
    ∀ l : List α, (prefixesWhere q l).length = l.countP q
  | [] => rfl
  | x :: xs => by
    rw [prefixesWhere, List.length_append, List.length_map, length_prefixesWhere q xs, List.countP_cons]
    cases q x <;> simp <;> omega

theorem mem_prefixesWhere {α : Type} {q : α → Bool} :
    ∀ {l pre : List α}, pre ∈ prefixesWhere q l ↔ ∃ r L₂, l = pre ++ r :: L₂ ∧ q r = true
  | [], pre => by
    constructor
    · intro h
      simp [prefixesWhere] at h
    · rintro ⟨r, L₂, he, -⟩
      cases pre <;> cases he
  | x :: xs, [] => by
    constructor
    · intro h
      rw [prefixesWhere] at h
      rcases List.mem_append.1 h with h | h
      · split at h
        · exact ⟨x, xs, rfl, ‹q x = true›⟩
        · simp at h
      · obtain ⟨a, -, e⟩ := List.mem_map.1 h
        cases e
    · rintro ⟨r, L₂, he, hr⟩
      obtain ⟨rfl, rfl⟩ := List.cons.inj he
      rw [prefixesWhere, if_pos hr]
      exact List.mem_append_left _ (List.mem_singleton_self _)
  | x :: xs, y :: p => by
    constructor
    · intro h
      rw [prefixesWhere] at h
      rcases List.mem_append.1 h with h | h
      · split at h <;> simp at h
      · obtain ⟨a, ha, e⟩ := List.mem_map.1 h
        obtain ⟨rfl, rfl⟩ := List.cons.inj e
        obtain ⟨r, L₂, he, hr⟩ := mem_prefixesWhere.1 ha
        exact ⟨r, L₂, by rw [he, List.cons_append], hr⟩
    · rintro ⟨r, L₂, he, hr⟩
      rw [List.cons_append] at he
      obtain ⟨rfl, he'⟩ := List.cons.inj he
      rw [prefixesWhere]
      exact List.mem_append_right _ (List.mem_map.2 ⟨p, mem_prefixesWhere.2 ⟨r, L₂, he', hr⟩, rfl⟩)

theorem pairwise_prefixesWhere {α : Type} (q : α → Bool) :
    ∀ l : List α, (prefixesWhere q l).Pairwise (fun a b => a.length < b.length)
  | [] => List.Pairwise.nil
  | x :: xs => by
    rw [prefixesWhere]
    refine List.pairwise_append.2 ⟨?_, List.pairwise_map.2 ((pairwise_prefixesWhere q xs).imp fun h => by
      simp only [List.length_cons]
      omega), fun a ha b hb => ?_⟩
    · split <;> simp
    · obtain ⟨c, -, rfl⟩ := List.mem_map.1 hb
      split at ha
      · rw [List.mem_singleton.1 ha]
        exact Nat.succ_pos _
      · simp at ha

/-- **The `k`-th log before an element `q` holds of has `k` of them.** -/
theorem countP_prefixesWhere {α : Type} (q : α → Bool) :
    ∀ (l : List α) (k : ℕ) (pre : List α), (prefixesWhere q l)[k]? = some pre → pre.countP q = k
  | [], k, pre, h => by simp [prefixesWhere] at h
  | x :: xs, k, pre, h => by
    rw [prefixesWhere] at h
    cases hx : q x
    · have hx' : ¬ (q x = true) := by simp [hx]
      rw [if_neg hx', List.nil_append, List.getElem?_map] at h
      obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 h
      rw [List.countP_cons, countP_prefixesWhere q xs k p hp]
      simp [hx]
    · rw [if_pos hx, List.singleton_append] at h
      cases k with
      | zero =>
        rw [List.getElem?_cons_zero] at h
        obtain rfl := Option.some.inj h
        rfl
      | succ k =>
        rw [List.getElem?_cons_succ, List.getElem?_map] at h
        obtain ⟨p, hp, rfl⟩ := Option.map_eq_some_iff.1 h
        rw [List.countP_cons, countP_prefixesWhere q xs k p hp]
        simp [hx]

/-- **The logs before each element `q` holds of have none, one, and so on of them.** -/
theorem map_countP_prefixesWhere {α : Type} (q : α → Bool) (l : List α) :
    (prefixesWhere q l).map (·.countP q) = List.range (l.countP q) := by
  refine List.ext_getElem (by rw [List.length_map, List.length_range, length_prefixesWhere]) fun k h₁ h₂ => ?_
  rw [List.getElem_map, List.getElem_range]
  exact countP_prefixesWhere q l k _ (List.getElem?_eq_getElem _)

/-- **Two lists of logs, each strictly longer than the one before, with the same members are one.** -/
theorem eq_of_mem_iff_of_sorted {α : Type} : ∀ {l₁ l₂ : List (List α)},
    l₁.Pairwise (fun a b => a.length < b.length) → l₂.Pairwise (fun a b => a.length < b.length) →
      (∀ a, a ∈ l₁ ↔ a ∈ l₂) → l₁ = l₂
  | [], [], _, _, _ => rfl
  | [], b :: _, _, _, h => absurd ((h b).2 (List.mem_cons_self ..)) List.not_mem_nil
  | a :: _, [], _, _, h => absurd ((h a).1 (List.mem_cons_self ..)) List.not_mem_nil
  | a :: l₁, b :: l₂, h₁, h₂, h => by
    rw [List.pairwise_cons] at h₁ h₂
    have hab : a = b := by
      rcases List.mem_cons.1 ((h a).1 (List.mem_cons_self ..)) with e | ha
      · exact e
      · rcases List.mem_cons.1 ((h b).2 (List.mem_cons_self ..)) with e | hb
        · exact e.symm
        · exact absurd ((h₂.1 a ha).trans (h₁.1 b hb)) (lt_irrefl _)
    subst hab
    congr 1
    refine eq_of_mem_iff_of_sorted h₁.2 h₂.2 fun c => ⟨fun hc => ?_, fun hc => ?_⟩
    · rcases List.mem_cons.1 ((h c).1 (List.mem_cons_of_mem _ hc)) with e | hc'
      · exact absurd (h₁.1 c hc) (by rw [e]; exact lt_irrefl _)
      · exact hc'
    · rcases List.mem_cons.1 ((h c).2 (List.mem_cons_of_mem _ hc)) with e | hc'
      · exact absurd (h₂.1 c hc) (by rw [e]; exact lt_irrefl _)
      · exact hc'

/-- **Logs, each longer than the one before, that are where a log has its openings of a stream, are the logs before
them**: where the logs of `xs` are each longer than the one before, and one of them whose element `p` holds of is a log
exactly where `Lg` opens tree `i`'s stream `s` after it, those logs are the ones before each of `Lg`'s openings of that
stream, in order. -/
theorem filter_logs {γ : Type _} {Lg : List Release} {i s : ℕ} {xs : List (γ × List Release)}
    (hpw : (xs.map Prod.snd).Pairwise (fun m m' => m.length < m'.length)) (p : γ → Bool)
    (hp : ∀ m, (∃ x ∈ xs, p x.1 = true ∧ x.2 = m) ↔ ∃ r L₂, Lg = m ++ r :: L₂ ∧ r.frame.opensAt i s = true) :
    (xs.filter fun x => p x.1).map Prod.snd = prefixesWhere (fun r => r.frame.opensAt i s) Lg := by
  refine eq_of_mem_iff_of_sorted (hpw.sublist ((List.filter_sublist ..).map _)) (pairwise_prefixesWhere _ Lg)
    fun m => ?_
  refine Iff.trans ?_ ((hp m).trans mem_prefixesWhere.symm)
  rw [List.mem_map]
  constructor
  · rintro ⟨x, hx, rfl⟩
    obtain ⟨hx, hpx⟩ := List.mem_filter.1 hx
    exact ⟨x, hx, hpx, rfl⟩
  · rintro ⟨x, hx, hpx, rfl⟩
    exact ⟨x, List.mem_filter.2 ⟨hx, hpx⟩, rfl⟩

/-! ## Each opening the device logs, served -/

/-- **Each opening of the log `log` is one the device serves after the releases before it.** -/
def LogServed (log : List Release) : Prop :=
  ∀ L₁ r L₂ i s n, log = L₁ ++ r :: L₂ → r.frame = .opening i s n → Served L₁ i s n

theorem logServed_nil : LogServed [] := fun L₁ _ _ _ _ _ h => by
  cases L₁ <;> simp at h

theorem LogServed.snoc {log : List Release} {r : Release} (h : LogServed log)
    (hr : ∀ i s n, r.frame = .opening i s n → Served log i s n) : LogServed (log ++ [r]) := by
  intro L₁ r' L₂ i s n he hf
  rcases List.append_eq_append_iff.1 he with ⟨a', rfl, ha'⟩ | ⟨c', rfl, hc'⟩
  · cases a' with
    | nil =>
      obtain ⟨rfl, -⟩ := List.cons.inj ha'
      rw [List.append_nil]
      exact hr i s n hf
    | cons x a'' =>
      have h₂ := (List.cons.inj ha').2
      cases a'' <;> simp at h₂
  · cases c' with
    | nil =>
      obtain ⟨rfl, -⟩ := List.cons.inj hc'
      have h₂ := hr i s n hf
      rwa [List.append_nil] at h₂
    | cons x c'' =>
      obtain ⟨rfl, -⟩ := List.cons.inj hc'
      exact h L₁ r' c'' i s n rfl hf

/-- **An opening the device's trees have room for and answer is one it serves**, where the trees are the ones the log
shows (`Served.next`'s and `Served.cap`'s converse). -/
theorem Served.of_next {pre : List Release} {ts : List OpenTree} (hT : TreesOf pre ts) {i s n : ℕ}
    (hc : treeLines ts < maxTreeLines) {T : OpenTree} (hTi : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p) : Served pre i s n := by
  obtain ⟨hs, ho, hn⟩ := OpenTree.next_some hp
  have hs' : s < T.spec.streams.size := by
    rw [← hs]
    exact p.1.isLt
  refine ⟨T.spec, ?_, hs', ?_, hn, ?_⟩
  · rw [← hT.specs, List.getElem?_map, hTi]
    rfl
  · rw [← hT.opened i s, openedIn_some hTi hs']
    have e : (⟨s, hs'⟩ : Fin T.spec.streams.size) = p.1 := Fin.ext hs.symm
    rw [e, ← ho]
    exact p.2.isLt
  · rw [← hT.lines]
    exact hc

theorem logServed_end {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    {f : List Release → Game Exchange} {pre : List Release} (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log))
    (hS : LogServed pre) (s : Strategy (f [])) {ex : Exchange} (h : PlayReaches (f []) s ex) : LogServed ex.log := by
  obtain ⟨s', h'⟩ := playReaches_of_eq (hf []) s h
  rw [log_of_finish h', List.append_nil]
  exact hS

/-- **An output a play of an optional node reaches**, by the node's two cases. -/
theorem reaches_optGame {β γ : Type} {g₀ : Game β} {K : γ → Game β} {Q : β → Prop} (o : Option γ)
    (h0 : ∀ s e, PlayReaches g₀ s e → Q e) (h1 : ∀ x, o = some x → ∀ s e, PlayReaches (K x) s e → Q e)
    (s : Strategy (optGame o g₀ K)) (e : β) (h : PlayReaches (optGame o g₀ K) s e) : Q e := by
  cases o with
  | none => exact h0 s e h
  | some x => exact h1 x rfl s e h

/-- **An output a play of a decision's node reaches**, by the branch it takes. -/
theorem reaches_ite {β : Type} {c : Prop} {dc : Decidable c} {t e : Game β} {Q : β → Prop}
    (ht : c → ∀ s x, PlayReaches t s x → Q x) (he : ¬ c → ∀ s x, PlayReaches e s x → Q x)
    (s : Strategy (@ite _ c dc t e)) (x : β) (h : PlayReaches (@ite _ c dc t e) s x) : Q x := by
  cases dc with
  | isFalse hc => exact he hc s x h
  | isTrue hc => exact ht hc s x h

/-- **Each opening a play of the device's releases logs is one it serves**: from a log each of whose openings the
device serves, at the trees that log shows, every play of the releases continuing to the close after the log and
them ends on a log each of whose openings the device serves after the releases before it. -/
theorem logServed_relK {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray} (k : ℕ) :
    ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game Exchange) (pre : List Release),
      (∀ log, f log = finish sg θ F c rq (pre ++ log)) → TreesOf pre ts → LogServed pre →
      ∀ (st : Strategy (relK k fs ts f)) (ex : Exchange), PlayReaches (relK k fs ts f) st ex → LogServed ex.log := by
  induction k with
  | zero => exact fun _ _ _ _ hf _ hS st _ h => logServed_end hf hS st h
  | succ k ih =>
    intro fs ts f pre hf hT hS st ex h
    obtain ⟨m, s₁⟩ := st
    rcases m with _ | ⟨l | spec | ⟨i, s, n⟩, b⟩
    · exact logServed_end hf hS s₁ h
    · refine reaches_optGame (Q := fun e => LogServed e.log) (takeCoin l fs)
        (fun s₀ _ h₀ => logServed_end hf hS s₀ h₀) (fun fs' _ s₂ e₂ h₂ => ?_) s₁ ex h
      obtain ⟨x, hx⟩ := h₂
      exact ih fs' ts (fun log => f (⟨.fresh l, b, List.ofFn x⟩ :: log)) (pre ++ [⟨.fresh l, b, List.ofFn x⟩])
        (fun log => by simp only [hf, List.append_assoc, List.singleton_append]) (hT.fresh l b _)
        (hS.snoc fun _ _ _ he => by cases he) (s₂ x) e₂ hx
    · refine reaches_ite (Q := fun e => LogServed e.log) (fun _ s₄ e₄ h₄ => ?_)
        (fun _ s₀ _ h₀ => logServed_end hf hS s₀ h₀) s₁ ex h
      obtain ⟨t, ht⟩ := h₄
      exact ih fs (ts ++ [⟨b, spec, t, fun _ => 0⟩])
        (fun log => f (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ :: log))
        (pre ++ [⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩])
        (fun log => by simp only [hf, List.append_assoc, List.singleton_append]) (hT.hello b spec t _)
        (hS.snoc fun _ _ _ he => by cases he) (s₄ t) e₄ ht
    · refine reaches_ite (Q := fun e => LogServed e.log) (fun hc s₄ e₄ h₄ => ?_)
        (fun _ s₀ _ h₀ => logServed_end hf hS s₀ h₀) s₁ ex h
      refine reaches_optGame (Q := fun e => LogServed e.log) ts[i]?
        (fun s₀ _ h₀ => logServed_end hf hS s₀ h₀) (fun T hTi s₂ e₂ h₂ => ?_) s₄ e₄ h₄
      refine reaches_optGame (Q := fun e => LogServed e.log) (T.next s n)
        (fun s₀ _ h₀ => logServed_end hf hS s₀ h₀) (fun p hp s₃ e₃ h₃ => ?_) s₂ e₂ h₂
      obtain ⟨d, hd⟩ := h₃
      exact ih fs (ts.set i (T.bump p.1)) (fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log))
        (pre ++ [⟨.opening i s n, b, TreeDraw.opened d.1 n⟩])
        (fun log => by simp only [hf, List.append_assoc, List.singleton_append]) (hT.opening hTi hp b _)
        (hS.snoc fun _ _ _ he => by
          obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj he
          exact Served.of_next hT hc hTi hp) (s₃ d) e₃ hd

/-- **A log with an opening has a first one, which the link's slot picks**, where each opening of the log is one the
device serves. -/
theorem link_picks (tables : Array String) {log : List Release} (hS : LogServed log) :
    ∀ (m : ℕ) (L₁ : List Release), L₁.length < m → ∀ (r : Release) (L₂ : List Release) (i s n : ℕ),
      log = L₁ ++ r :: L₂ → r.frame = .opening i s n →
        ∃ L₁ r L₂, log = L₁ ++ r :: L₂ ∧ slotRound tables .link L₁ r.frame
  | 0, _, h => absurd h (Nat.not_lt_zero _)
  | m + 1, L₁, hm => by
    intro r L₂ i s n he hr
    cases hf : firstOpened L₁ with
    | none => exact ⟨L₁, r, L₂, he, i, s, n, hr, hS L₁ r L₂ i s n he hr, hf⟩
    | some i' =>
      obtain ⟨r', hmem, hr'⟩ := List.exists_of_findSome?_eq_some hf
      obtain ⟨A, C, rfl⟩ := List.append_of_mem hmem
      have ho : ∃ i s n, r'.frame = .opening i s n := by
        try dsimp only at hr'
        cases hfr : r'.frame <;> rw [hfr] at hr' <;> first | exact ⟨_, _, _, rfl⟩ | cases hr'
      obtain ⟨i₁, s₁, n₁, ho⟩ := ho
      have hA : A.length < m := by
        rw [List.length_append, List.length_cons] at hm
        omega
      exact link_picks tables hS m A hA r' (C ++ r :: L₂) i₁ s₁ n₁
        (by rw [he, List.append_assoc, List.cons_append]) ho

/-! ## The first stop -/

section First

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α] (V : Vocab d α) (θ : ByteArray)
  (F : Array ByteArray) (c : ByteArray)

/-- **The first stop is where `StopOf` says**, from the empty log with every round live, at the releases continuing
to the close after `σ`'s request, where each round's reading picks only openings (`walk_send_cast`). -/
theorem stopOf_firstStop (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (ω : (A.extend M.widening).L.Ω)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopOf rd.round rd.reveal [] (finish (V.signer θ) θ F c σ.1) (fun _ => True) (firstStop V θ F c A rd σ M ω) := by
  have h := walk_send_cast (rv := rd.reveal) (f := finish (V.signer θ) θ F c) (a := fun _ => True) hP
    (game_eq_relK V θ F c).symm σ (roundCuts V θ F c rd) (M.start ω) id fun _ => Or.inl ⟨trivial, rfl⟩
  have e : (cast (congrArg Strategy (game_eq_relK V θ F c)) σ).1 = σ.1 := cast_send_fst (game_eq_relK V θ F c) σ
  rw [e] at h
  exact h

/-- **The first stop is where `StopsAt` says**, from the empty log after `σ`'s request with every round live, where
each round's reading picks only openings. -/
theorem stopsAt_firstStop (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (ω : (A.extend M.widening).L.Ω)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 (firstStop V θ F c A rd σ M ω) :=
  ⟨[], _, fun _ => True, fun _ => rfl, stopOf_firstStop V θ F c A rd σ M ω hP⟩

/-- **The link points are in the play's log**: on a trace from the first stop, the link points written in and
consumed, where each round's reading picks only openings the device serves, not every cut is past at the first stop,
the coin map writes the link points into the stop's opening as their bits `bitsOf`, and off `BL` the opening's class at
those bits is not empty, the play on the tape the trace ends on releases the link points' bits at an opening the
rounds whose cuts the first stop fixes pick (`LoggedAt`, `RoundsReach.stop_logged`). -/
theorem firstStop_logged (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    {D : Decoders A.rounds A.Reg} (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    {B : RoundHold} (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, rd.round j pre (.opening i s n) → Served pre i s n)
    (bitsOf : (Fin 2 → Fin A.mP → 𝔽) → List Bool)
    (hML : ∀ q pts, M.link q pts = writeOpening (finish (V.signer θ) θ F c σ.1) q.s' (bitsOf pts))
    (hBL : ∀ g' pts, ¬ BL g' pts → ¬ EmptyAt (finish (V.signer θ) θ F c σ.1) g' (bitsOf pts))
    (ω : (A.extend M.widening).L.Ω)
    (pts : Fin 2 → Fin A.mP → 𝔽) (hK : ¬ ∀ j, (firstStop V θ F c A rd σ M ω).Ks j = .skip)
    (hL : ¬ BL (firstStop V θ F c A rd σ M ω).g' pts) {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β}
    {rootB : Array ByteArray} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o) :
    LoggedAt rd.round (firstStop V θ F c A rd σ M ω) (bitsOf pts) (CoinTape.run (V.game θ F c) σ o.2).log := by
  rw [hML] at hr
  exact RoundsReach.stop_logged hPS (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))).2.2
    (stopsAt_firstStop V θ F c A rd σ M ω hP) hK (hBL _ _ hL) hr

/-- **The first stop reaches the play a trace from it ends on**, any rewriting of its tape consumed
(`RoundsReach.playReaches`). -/
theorem firstStop_playReaches (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    {D : Decoders A.rounds A.Reg} (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) {B : RoundHold}
    (ω : (A.extend M.widening).L.Ω)
    {f : CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s' →
      CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s'}
    {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs ((firstStop V θ F c A rd σ M ω).rewrite f).consume o) :
    PlayReaches (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s'
      (CoinTape.run (V.game θ F c) σ o.2) :=
  hr.playReaches (Pos.inv_consume (Pos.inv_rewrite
    (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))).2.2 f))

/-- **Each opening a play logs is one the device serves** after the releases before it (`logServed_relK`). -/
theorem logServed_run (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ) :
    LogServed (CoinTape.run (V.game θ F c) σ x).log := by
  obtain ⟨s', h⟩ := playReaches_of_eq (game_eq_relK V θ F c) σ (run_reaches _ σ x)
  exact logServed_relK (sg := V.signer θ) (θ := θ) (F := F) (c := c) (rq := s'.1) _ _ _
    (finish (V.signer θ) θ F c s'.1) [] (fun _ => rfl) TreesOf.nil logServed_nil s'.2 _ h

/-- **Where every cut is past at the first stop, no round picks in the play's log**: on a trace from the first stop,
any rewriting of its tape consumed, where each round's reading picks only openings, no round picks any release of the
play's log on the tape the trace ends on, after the releases before it. -/
theorem firstStop_skip_log (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    {D : Decoders A.rounds A.Reg} (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) {B : RoundHold}
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) (ω : (A.extend M.widening).L.Ω)
    (hK : ∀ j, (firstStop V θ F c A rd σ M ω).Ks j = .skip)
    (φ : CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s' →
      CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s')
    {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs ((firstStop V θ F c A rd σ M ω).rewrite φ).consume o) :
    NoPick rd.round (fun _ => True) [] (CoinTape.run (V.game θ F c) σ o.2).log := by
  obtain ⟨L, hL, h⟩ := stopOf_firstStop V θ F c A rd σ M ω hP
  rcases h with ⟨-, y, hy, s₀, x₀, hrun⟩ |
    ⟨k, fs, ts, f', i, s, n, b, -, -, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · have h₀ : PlayReaches (finish (V.signer θ) θ F c σ.1 L) s₀ y := by
      rw [← hrun]
      exact run_reaches _ s₀ x₀
    rw [RoundsReach.run_ret (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))).2.2 hy hr,
      log_of_finish h₀]
    exact hL
  · have hjK : ((firstStop V θ F c A rd σ M ω).Ks j).isHere = true := by
      rw [hq]
      exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
    rw [hK j] at hjK
    cases hjK

/-- **Where the play's log has an opening, the first stop fixes some cut**: on a trace from the first stop, any
rewriting of its tape consumed, where each round's reading picks only openings and round `j₀`'s picks what the link's
slot picks, and the play's log on the tape the trace ends on has an opening, not every cut is past at the first stop
(`firstStop_skip_log`, `link_picks`). -/
theorem firstStop_fixes (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    {D : Decoders A.rounds A.Reg} (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) {B : RoundHold}
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) (tables : Array String) (j₀ : Fin A.rounds)
    (hj₀ : ∀ pre fr, slotRound tables .link pre fr → rd.round j₀ pre fr) (ω : (A.extend M.widening).L.Ω)
    (φ : CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s' →
      CoinTape (firstStop V θ F c A rd σ M ω).g' (firstStop V θ F c A rd σ M ω).s')
    {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs ((firstStop V θ F c A rd σ M ω).rewrite φ).consume o)
    (hO : ∃ L₁ r L₂ i s n, (CoinTape.run (V.game θ F c) σ o.2).log = L₁ ++ r :: L₂ ∧ r.frame = .opening i s n) :
    ¬ ∀ j, (firstStop V θ F c A rd σ M ω).Ks j = .skip := by
  intro hK
  have hN := firstStop_skip_log V θ F c A rd σ M hP ω hK φ hr
  obtain ⟨L₁, r, L₂, i, s, n, he, hr'⟩ := hO
  obtain ⟨L₁', r', L₂', he', hl⟩ :=
    link_picks tables (logServed_run V θ F c σ o.2) _ L₁ (Nat.lt_succ_self _) r L₂ i s n he hr'
  exact hN j₀ trivial L₁' r' L₂' he' (hj₀ _ _ hl)

/-- **The trace from the first stop, in order in the log**, at the readings `rd` and the decoders `D` at a layout `L`,
injective, of distinct tables, its rep slots of tables before `J`: where each round picks at most one release of the
log `Lg` of the play on the tape the trace ends on, the coin map writes the link points and a round's coins into a
stop's opening as their bits, off `BL` and `B` the opening's classes at those bits are not empty, not every cut is past
at the first stop, `BL` doesn't hold there, and the rounds but one are at most the trace's `N`: `Lg` has the link
points' bits after a log `L₀` (`LoggedAfter`), and the trace's answered stops (`RoundsReach.answers`) each release their
round's coins after a log, each longer than `L₀` and than the one before, each stop's fixed cuts at the bytes its
release answers (`CutsAt`), each at a position that keeps the invariant at the rounds' cuts, the rest of the trace
running from the stop after it; and every release of `Lg` a round picks is after `L₀` or one of those
(`RoundsReach.order`, `covers_of_length`). -/
theorem firstStop_order (A : Live) {tables : Array String} {L : Fin A.rounds → Slot} (hL : Function.Injective L)
    (ht : tables.toList.Nodup) {J : ℕ} (hJ : ∀ j t r k, L j = .rep t r k → t < J)
    (record : Exchange → Option Claims.Msg) (recordDigest : Claims.Msg) (reg : Option Claims.Msg → A.Reg)
    {rd : Reading A.rounds} (hrd : rd = readingAt tables L record recordDigest) {D : Decoders A.rounds A.Reg}
    (hD : D = decodersAt reg L) (σ : Strategy (V.game θ F c)) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (BL : LinkHold A.mP) {B : RoundHold} (bitsL : (Fin 2 → Fin A.mP → 𝔽) → List Bool)
    (hML : ∀ q pts, M.link q pts = writeOpening (finish (V.signer θ) θ F c σ.1) q.s' (bitsL pts))
    (hBL : ∀ g' pts, ¬ BL g' pts → ¬ EmptyAt (finish (V.signer θ) θ F c σ.1) g' (bitsL pts))
    (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool)
    (hM : ∀ q s x, M.round q J s x = writeOpening (finish (V.signer θ) θ F c σ.1) q.s' (bitsOf s x))
    (hB : ∀ g' s x, ¬ B g' J s x → ¬ EmptyAt (finish (V.signer θ) θ F c σ.1) g' (bitsOf s x))
    (ω : (A.extend M.widening).L.Ω)
    (pts : Fin 2 → Fin A.mP → 𝔽) (hK : ¬ ∀ j, (firstStop V θ F c A rd σ M ω).Ks j = .skip)
    (hL0 : ¬ BL (firstStop V θ F c A rd σ M ω).g' pts) {N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β}
    {rootB : Array ByteArray} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o)
    {Lg : List Release} (hLg : (CoinTape.run (V.game θ F c) σ o.2).log = Lg) (hU : PicksOnce rd.round Lg)
    (hRN : A.rounds ≤ N + 1) :
    ∃ (L₀ : List Release) (as : List (Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) ×
        ((s : SubmitJ J) × (Fin s.n → 𝔽)))) (ms : List (List Release)),
      (LoggedAfter rd.round (firstStop V θ F c A rd σ M ω) L₀ (bitsL pts) Lg ∧
        CutsAt rd.reveal (firstStop V θ F c A rd σ M ω) L₀ Lg) ∧
      o.1 = fin ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf A.dg a.2.1.bytes a.2.1.n a.2.2)) (fun _ _ => []),
          D.proofs (CoinTape.run (V.game θ F c) σ o.2)⟩ ∧
      List.Forall₂ (fun a m => D.sub J a.1.toStop.modes = some a.2.1 ∧
        LoggedAfter rd.round a.1.toStop m (bitsOf a.2.1 a.2.2) Lg ∧ CutsAt rd.reveal a.1.toStop m Lg) as ms ∧
      (∀ a ∈ as, a.1.Inv (roundCuts V θ F c rd) ∧ ∃ N' rs', RoundsReach D M A.dg J B fin rootB pts N' rs'
        (a.1.toStop.rewrite (M.round a.1.toStop J a.2.1 a.2.2)).consume o) ∧
      (∀ m ∈ ms, L₀.length < m.length) ∧ ms.Pairwise (fun m m' => m.length < m'.length) ∧
      ∀ j L₁, PicksIn rd.round Lg j L₁ → L₁ = L₀ ∨ L₁ ∈ ms := by
  subst hrd hD
  have hP := readingAt_picks tables L record recordDigest
  have hPS := readingAt_served tables L record recordDigest
  have hi₀ := (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c (readingAt tables L record recordDigest))
    (M.start ω))).2.2
  have hSO := stopOf_firstStop V θ F c A (readingAt tables L record recordDigest) σ M ω hP
  obtain ⟨L₀, hNP, ⟨hK', -⟩ | ⟨k, fs, ts, f', i, s, n, b, hf', hT, ⟨j, -, hj⟩, s'', Ks, x'', back'', hq, hKs⟩⟩ :=
    hSO
  · exact absurd hK' hK
  have hf'' : ∀ log, f' log = finish (V.signer θ) θ F c σ.1 (L₀ ++ log) := hf'
  have hr' := hr
  rw [hML] at hr'
  obtain ⟨L', hL'⟩ := RoundsReach.release hq hi₀ hf'' hT (hPS j _ i s n hj) (hBL _ _ hL0) hr'
  rw [hLg] at hL'
  have hjK : ((firstStop V θ F c A (readingAt tables L record recordDigest) σ M ω).Ks j).isHere = true := by
    rw [hq]
    exact (isHere_openCuts hKs j).2 ⟨trivial, hj⟩
  have hall : ∀ j', ((firstStop V θ F c A (readingAt tables L record recordDigest) σ M ω).Ks j').isHere = true →
      (readingAt tables L record recordDigest).round j' L₀ (.opening i s n) := fun j' h' => by
    rw [hq] at h'
    exact ((isHere_openCuts hKs j').1 h').2
  have hA : AnsweredBy (readingAt tables L record recordDigest).round Lg L₀
      (fun j => True ∧ ¬ (readingAt tables L record recordDigest).round j L₀ (.opening i s n)) [L₀] := by
    refine ⟨List.mem_singleton_self _, ⟨_, L', hL'⟩,
      fun j' hj' => ⟨L₀, List.mem_singleton_self _, _, L', hL', not_not.1 fun h => hj' ⟨trivial, h⟩⟩,
      fun j' L₁ hp hl => ?_, fun j' L₁ hd hp => ?_⟩
    · obtain ⟨r, L₂, he, hp'⟩ := id hp
      obtain ⟨c', d, hc, hLd⟩ := mid_of_append (pre := []) (L := L₀) (L₃ := ⟨.opening i s n, b, bitsL pts⟩ :: L')
        (L₁ := L₁) (r := r) (L₂ := L₂) (by rw [List.nil_append]; exact he.symm.trans hL') (Nat.zero_le _)
        (by simp only [List.length_nil, Nat.zero_add]; exact hl)
      rw [hc] at hp'
      exact absurd hp' (hNP j' trivial c' r d hLd)
    · rw [List.mem_singleton] at hd
      subst hd
      obtain ⟨r, L₂, he, hp'⟩ := hp
      rw [hL'] at he
      obtain ⟨rfl, -⟩ := List.cons.inj (List.append_cancel_left he)
      exact fun h => h.2 hp'
  obtain ⟨as, ms, h1, hF, hI, hgt, hpw, hcov⟩ := RoundsReach.order hP hPS (sub_eq_none reg L J fun _ _ => rfl)
    bitsOf hM hB hU hr (Pos.inv_consume (Pos.inv_rewrite hi₀ _)) hLg L₀ f' _ [L₀] hf''
    (stopOf_next hP s'' Ks x'' back'' hq hT hKs _) hA
  refine ⟨L₀, as, ms, ⟨⟨j, i, s, n, b, L', hjK, hall, hj, hL'⟩, cutsAt_openCuts hq hKs hL'⟩, h1, hF, hI, hgt, hpw,
    ?_⟩
  rcases hcov with hc | hN | ⟨q, L₁, hnone, hnd, j₁, hj₁, hall₁⟩
  · exact fun j' L₁ hp => (hc j' L₁ hp).imp_left List.mem_singleton.1
  · refine covers_of_length hU (j₀ := j) ⟨_, L', hL', hj⟩ (fun m hm => ?_) hgt
      (hpw.imp fun h e => by subst e; exact lt_irrefl _ h) (by rw [← hF.length_eq, hN]; exact hRN)
    obtain ⟨a, ha⟩ := forall₂_zip_right hF hm
    obtain ⟨-, ⟨j', i', s', n', b', L'', -, -, hj', he'⟩, -⟩ := List.forall₂_zip hF ha
    exact ⟨j', _, L'', he', hj'⟩
  · exfalso
    obtain ⟨r, L₂, he, hp₁⟩ := hall₁ j₁ hj₁
    have hone : ∀ j', j' ≠ j₁ → q.modes j' = none := fun j' hj' => Pos.modes_eq_none (p := q) (i := j') (by
      cases h : (q.Ks j').isHere
      · rfl
      · obtain ⟨r', L₂', he', hp'⟩ := hall₁ j' h
        rw [he] at he'
        obtain ⟨rfl, -⟩ := List.cons.inj (List.append_cancel_left he')
        exact absurd (readingAt_unique hL ht record recordDigest hp' hp₁) hj')
    obtain ⟨m, hm⟩ := Pos.modes_some hj₁
    cases hLj : L j₁ with
    | rep t r' k =>
      rw [sub_single reg L J hLj (hJ j₁ t r' k hLj) hm hone] at hnone
      cases hnone
    | link =>
      have hfo : firstOpened L₁ = none := by
        have h' : slotRound tables (L j₁) L₁ r.frame := hp₁
        rw [hLj] at h'
        obtain ⟨-, -, -, -, -, h''⟩ := h'
        exact h''
      by_cases hlt : L₁.length < L₀.length
      · exact hnd (hA.before j₁ L₁ ⟨r, L₂, he, hp₁⟩ hlt)
      · obtain ⟨d, rfl⟩ := prefix_le (he.symm.trans hL') (by omega)
        rw [List.append_assoc] at he
        have he₂ := List.append_cancel_left (he.symm.trans hL')
        cases d with
        | nil => exact hnd (by simp)
        | cons x d =>
          rw [List.cons_append] at he₂
          obtain ⟨rfl, -⟩ := List.cons.inj he₂
          exact firstOpened_ne_none L₀ (r := ⟨.opening i s n, b, bitsL pts⟩) rfl d hfo

variable {V θ F c} in
/-- **Off the misses, an answered stop decodes the message the play reveals there**: at a position that keeps the
invariant at the rounds' cuts, where the rest of a trace runs from the stop after it, its tape rewritten and consumed,
the tape the trace ends on misses at no round, and its play reveals `w` at the cut of a round fixed at the stop, the
stop's most likely message for that round is `w` (`RoundsReach.stop_modes`). -/
theorem RoundsReach.stop_modes_noMiss {A : Live} {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {B : RoundHold} {J N : ℕ}
    {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {q : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds)}
    (hq : q.Inv (roundCuts V θ F c rd))
    {φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs (q.toStop.rewrite φ).consume o)
    (hn : NoMiss V θ F c rd σ o.2) {i : Fin A.rounds} (hi : (q.toStop.Ks i).isHere = true) {w : Claims.Msg}
    (hw : (roundCut V θ F c rd i).cut.revealOn σ o.2 = some w) : q.toStop.modes i = some w := by
  rw [RoundsReach.stop_modes hq hr]
  dsimp only
  rw [ite_eq_left hi]
  by_contra hne
  exact hn.1 i ((DigestCut.missOn_iff _ _ _).2 ⟨w, hw, hne⟩)

variable {V θ F c} in
/-- **Off the misses, an answered stop decodes the message the play reveals for its round**: as
`RoundsReach.stop_modes_noMiss`, where the cuts the stop fixes are at the bytes the release `r` of the play's log
answers (`CutsAt`), and the play reveals for a round fixed there a message `w` whose SHA-512 is those bytes
(`RoundsReach.stop_reveal`). -/
theorem RoundsReach.stop_modes_reveal {A : Live} {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {B : RoundHold} {J N : ℕ}
    {β : Type} {fin : LiveOutJ J A.mP → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin A.mP → 𝔽}
    {rs : Fin J → Fin 2 → List Flock.Round} {q : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds)}
    (hq : q.Inv (roundCuts V θ F c rd))
    {φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N rs (q.toStop.rewrite φ).consume o)
    (hn : NoMiss V θ F c rd σ o.2) {pre L : List Release} {r : Release}
    (hc : CutsAt rd.reveal q.toStop pre (CoinTape.run (V.game θ F c) σ o.2).log)
    (hL : (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ r :: L) {i : Fin A.rounds}
    (hi : (q.toStop.Ks i).isHere = true) {w : Claims.Msg}
    (hw : rd.reveal i (CoinTape.run (V.game θ F c) σ o.2) = some w) (hd : Verity.sha512Msg w = r.answers.data.toList) :
    q.toStop.modes i = some w := by
  refine RoundsReach.stop_modes_noMiss hq hr hn hi ?_
  refine (RoundsReach.stop_reveal hq hr (hc i r L hi hL)).trans ?_
  unfold digestReveal
  rw [hw]
  simp [Option.filter, hd]

end First

/-! ## The answered stops of one stream -/

/-- **A frame that opens tree `i`'s stream `s` is an opening of it.** -/
theorem opening_of_opensAt {fr : Frame} {i s : ℕ} (h : fr.opensAt i s = true) : ∃ n, fr = .opening i s n := by
  cases fr with
  | opening i' s' n =>
    have h' : i' = i ∧ s' = s := by simpa [Frame.opensAt] using h
    exact ⟨n, by rw [h'.1, h'.2]⟩
  | _ => simp [Frame.opensAt] at h

/-- **A log with no opening, then an opening of tree `i`, has first opened tree `i`.** -/
theorem firstOpened_opening {L₀ : List Release} (h : firstOpened L₀ = none) {r : Release} {i s n : ℕ}
    (hr : r.frame = .opening i s n) (d : List Release) : firstOpened (L₀ ++ r :: d) = some i := by
  unfold firstOpened at h ⊢
  rw [List.findSome?_append, h, Option.none_or]
  simp [List.findSome?_cons, hr]

/-- **A log past a log with no opening and the opening of tree `i` after it has first opened tree `i`.** -/
theorem firstOpened_past {L₀ : List Release} (h₀ : firstOpened L₀ = none) {r₀ : Release} {i s n : ℕ}
    (hr₀ : r₀.frame = .opening i s n) {L' m R₁ : List Release} (he : m ++ R₁ = L₀ ++ r₀ :: L')
    (hlt : L₀.length < m.length) : firstOpened m = some i := by
  obtain ⟨d, rfl⟩ := prefix_le he hlt.le
  cases d with
  | nil => simp at hlt
  | cons x d =>
    rw [List.append_assoc, List.cons_append] at he
    obtain ⟨hx, -⟩ := List.cons.inj (List.append_cancel_left he)
    rw [hx]
    exact firstOpened_opening h₀ hr₀ d

/-- **What a tree's stream is named stays as the log grows.** -/
theorem helloSpecs_bind_append {β : Type} {L₁ : List Release} {i : ℕ} {f : Flock.CoinTree.Spec → Option β} {y : β}
    (h : ((helloSpecs L₁)[i]?.bind f) = some y) (d : List Release) :
    ((helloSpecs (L₁ ++ d))[i]?.bind f) = some y := by
  cases hsp : (helloSpecs L₁)[i]? with
  | none =>
    rw [hsp, Option.bind_none] at h
    cases h
  | some sp =>
    rw [helloSpecs_getElem?_append hsp d]
    rw [hsp] at h
    exact h

/-- **Two streams of one tree with one name are one**, where each tree of the log `Lg` names its streams distinctly:
a stream of tree `i` after a log `Lg` continues, and a stream of that tree in `Lg`, with one name. -/
theorem stream_eq {Lg m R₁ : List Release} (he : Lg = m ++ R₁) (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup)
    {i s s' : ℕ} {o : Option String} (hsp : ∃ sp, (helloSpecs m)[i]? = some sp ∧ s' < sp.streams.size)
    (h' : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s']?) = o)
    (h : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = o) : s' = s := by
  obtain ⟨sp, hsp, hlt⟩ := hsp
  have hsp' : (helloSpecs Lg)[i]? = some sp := by
    rw [he]
    exact helloSpecs_getElem?_append hsp R₁
  rw [hsp, Option.bind_some] at h'
  rw [hsp', Option.bind_some, ← h'] at h
  refine nodup_getElem?_inj (hn sp (List.mem_of_getElem? hsp')) (a := sp.streams[s']) ?_ ?_
  · rw [Array.getElem?_toList, Array.getElem?_eq_getElem hlt]
  · rw [Array.getElem?_toList, h, Array.getElem?_eq_getElem hlt]

/-- **No rep stream is named the session's.** -/
theorem rep_ne_session {tables : Array String} {t r : ℕ} :
    tables[t]?.map (repStreamName · r) ≠ some _root_.Spot.Challenger.TREE_SESSION := by
  intro h
  cases hta : tables[t]? with
  | none =>
    rw [hta] at h
    cases h
  | some a =>
    rw [hta] at h
    have e : repStreamName a r = _root_.Spot.Challenger.TREE_SESSION := Option.some.inj h
    have := repStreamName_ne a r
    rw [e] at this
    simp at this

/-- **The release after the first stop's log is the log's first opening**, at the readings at a layout with a link
slot: where each opening of `Lg` is one the device serves, every release a round picks in `Lg` is after `L₀` or one
of the logs `ms`, each longer than `L₀`, and `Lg` has an opening after `L₀`, `L₀` has none (`link_picks`). -/
theorem firstOpened_link {R : ℕ} {tables : Array String} {L : Fin R → Slot} {record : Exchange → Option Claims.Msg}
    {recordDigest : Claims.Msg} {Lg L₀ L' : List Release} {ms : List (List Release)} (hS : LogServed Lg)
    {jl : Fin R} (hjl : L jl = .link)
    (hcov : ∀ j L₁, PicksIn (readingAt tables L record recordDigest).round Lg j L₁ → L₁ = L₀ ∨ L₁ ∈ ms)
    (hgt : ∀ m ∈ ms, L₀.length < m.length) {r₀ : Release} {i s n : ℕ} (hL₀ : Lg = L₀ ++ r₀ :: L')
    (hr₀ : r₀.frame = .opening i s n) : firstOpened L₀ = none := by
  obtain ⟨L₁, r, L₂, he, hl⟩ := link_picks tables hS _ L₀ (Nat.lt_succ_self _) r₀ L' i s n hL₀ hr₀
  obtain ⟨-, -, -, -, -, hfo⟩ := id hl
  rcases hcov jl L₁ ⟨r, L₂, he, by show slotRound tables (L jl) L₁ r.frame; rw [hjl]; exact hl⟩ with e | hm
  · rw [← e]
    exact hfo
  · obtain ⟨d, rfl⟩ := prefix_le (he.symm.trans hL₀) (hgt L₁ hm).le
    cases d with
    | nil => exact absurd (hgt _ hm) (by rw [List.append_nil]; exact lt_irrefl _)
    | cons x d =>
      rw [List.append_assoc, List.cons_append] at he
      obtain ⟨hx, -⟩ := List.cons.inj (List.append_cancel_left (he.symm.trans hL₀))
      rw [hx] at hfo
      exact absurd hfo (firstOpened_ne_none L₀ hr₀ d)

section Stream

variable {g : Game Exchange} {σ : Strategy g} {R J : ℕ} {Reg : Type}

/-- **A stop answered on table `t`'s rep `r` stream is at an opening of that stream**, at the readings and the decoders
at a layout `L`, injective, of distinct tables: where the decoders submit `sub` on that stream at the stop `q` and
`bits` are released after the log `m` at an opening its fixed rounds pick (`LoggedAfter`), `m` longer than `L₀`, each
opening of `Lg` is one the device serves, each tree of `Lg` names its streams distinctly, `L₀` has no opening and is
followed by an opening of tree `i`, and tree `i`'s stream `s` is named table `t`'s rep `r`, `sub` is round `k` of that
stream with the most likely message of round `j`, whose slot that is and whose cut `q` fixes, and the opening is of
stream `s` of tree `i`, after `k` openings of it in `m`. -/
theorem LoggedAfter.at_stream (reg : Option Claims.Msg → Reg) {tables : Array String} {L : Fin R → Slot}
    (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {q : Pos g σ Claims.Msg (Fin R)} {sub : SubmitJ J}
    (hsub : (decodersAt reg L).sub J q.modes = some sub) {m Lg : List Release} {bits : List Bool}
    (hlog : LoggedAfter (readingAt tables L record recordDigest).round q m bits Lg) (hS : LogServed Lg)
    (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) {L₀ L' : List Release} {r₀ : Release} {i s₀ n₀ : ℕ}
    (hL₀ : Lg = L₀ ++ r₀ :: L') (hr₀ : r₀.frame = .opening i s₀ n₀) (h₀ : firstOpened L₀ = none)
    (hlt : L₀.length < m.length) (t : Fin J) (r : Fin 2) {s : ℕ}
    (hs : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = tables[t.val]?.map (repStreamName · r.val))
    (htr : sub.tab = t ∧ sub.rep = r) :
    ∃ (k : ℕ) (w : Claims.Msg) (j : Fin R) (n : ℕ) (b : ByteArray) (L'' : List Release), L j = .rep t.val r k ∧
      (q.Ks j).isHere = true ∧ q.modes j = some w ∧ sub = ⟨t, r, ⟨w.toArray⟩, tailCount w⟩ ∧
      ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) = tables[t.val]?.map (repStreamName · r.val) ∧
      openedOf m i s = k ∧ Lg = m ++ ⟨.opening i s n, b, bits⟩ :: L'' := by
  obtain ⟨t', ht', r', k, w, j, i', s', n, b, L'', hLj, hj, hw, hsa, hfo', hname, hk, he⟩ :=
    LoggedAfter.slot reg hL ht record recordDigest hsub hlog
  have ht't : t' = t.val := by
    have h := congrArg Fin.val htr.1
    rw [hsa] at h
    exact h
  have hr'r : r' = r := by
    have h := htr.2
    rw [hsa] at h
    exact h
  have hii : i' = i := by
    rw [firstOpened_past h₀ hr₀ (he.symm.trans hL₀) hlt] at hfo'
    exact (Option.some.inj hfo').symm
  subst ht't hr'r hii
  obtain ⟨sp, hsp, hslt, -⟩ := hS m _ L'' i' s' n he rfl
  obtain rfl := stream_eq he hn ⟨sp, hsp, hslt⟩ hname hs
  exact ⟨k, w, j, n, b, L'', hLj, hj, hw, hsa, hname, hk, he⟩

/-- **The answered stops of one stream are at its openings, in order**, at the readings and the decoders at a layout
`L`, injective, of distinct tables: where the answered stops `as` each release their round's coins after the logs `ms`
of `Lg` (`firstStop_order`), each longer than `L₀` and than the one before, every release a round picks in `Lg` is
after `L₀` or one of `ms`, each opening of `Lg` is one the device serves, each tree of `Lg` names its streams
distinctly, `L₀` has no opening and is followed by an opening of tree `i`'s stream named the session's, tree `i`'s
stream `s` is named table `t`'s rep `r`, and the layout has a slot for each of `Lg`'s openings of it, the logs after
which the stops submitting on table `t`'s rep `r` release their coins are the logs before each of `Lg`'s openings of
`s`, in order. -/
theorem stream_logs (reg : Option Claims.Msg → Reg) {tables : Array String} {L : Fin R → Slot}
    (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool)
    {as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))} {ms : List (List Release)}
    {Lg L₀ L' : List Release}
    (hF : List.Forall₂ (fun a m => (decodersAt reg L).sub J a.1.toStop.modes = some a.2.1 ∧
      LoggedAfter (readingAt tables L record recordDigest).round a.1.toStop m (bitsOf a.2.1 a.2.2) Lg) as ms)
    (hgt : ∀ m ∈ ms, L₀.length < m.length) (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hcov : ∀ j L₁, PicksIn (readingAt tables L record recordDigest).round Lg j L₁ → L₁ = L₀ ∨ L₁ ∈ ms)
    (hS : LogServed Lg) (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) {r₀ : Release} {i s₀ n₀ : ℕ}
    (hL₀ : Lg = L₀ ++ r₀ :: L') (hr₀ : r₀.frame = .opening i s₀ n₀) (h₀ : firstOpened L₀ = none)
    (hses : ((helloSpecs L₀)[i]?.bind fun sp => sp.streams[s₀]?) =
      some _root_.Spot.Challenger.TREE_SESSION)
    (t : Fin J) (r : Fin 2) {s : ℕ}
    (hs : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = tables[t.val]?.map (repStreamName · r.val))
    (hcovL : ∀ k < openedOf Lg i s, ∃ j, L j = .rep t.val r k) :
    ((as.zip ms).filter fun x => decide (x.1.2.1.tab = t ∧ x.1.2.1.rep = r)).map Prod.snd =
      prefixesWhere (fun r => r.frame.opensAt i s) Lg := by
  have hlen := hF.length_eq
  have hfo : ∀ m R₁, Lg = m ++ R₁ → L₀.length < m.length → firstOpened m = some i := fun m R₁ he hlt =>
    firstOpened_past h₀ hr₀ (he.symm.trans hL₀) hlt
  refine filter_logs (xs := as.zip ms) (by rw [List.map_snd_zip hlen.ge]; exact hpw)
    (fun a => decide (a.2.1.tab = t ∧ a.2.1.rep = r)) fun m => ⟨?_, ?_⟩
  · rintro ⟨⟨⟨q, sub, cs⟩, m'⟩, hx, hpx, hxm⟩
    have hxm' : m' = m := hxm
    rw [← hxm']
    obtain ⟨hsub0, hlog0⟩ := List.forall₂_zip hF hx
    have hsub : (decodersAt reg L).sub J q.toStop.modes = some sub := hsub0
    have hlog : LoggedAfter (readingAt tables L record recordDigest).round q.toStop m' (bitsOf sub cs) Lg := hlog0
    obtain ⟨_, _, _, n', _, L'', -, -, -, -, -, -, he⟩ := LoggedAfter.at_stream reg hL ht record recordDigest hsub
      hlog hS hn hL₀ hr₀ h₀ (hgt m' (List.of_mem_zip hx).2) t r hs (of_decide_eq_true hpx)
    exact ⟨_, L'', he, by simp [Frame.opensAt]⟩
  · rintro ⟨x, L₂, he, hx⟩
    obtain ⟨n, hxn⟩ := opening_of_opensAt hx
    have hsv := hS m x L₂ i s n he hxn
    have hlt : L₀.length < m.length := by
      by_contra hle
      obtain ⟨d, hd⟩ := prefix_le (hL₀.symm.trans he) (by omega)
      cases d with
      | nil =>
        have h1 := helloSpecs_bind_append hses (r₀ :: L')
        rw [← hL₀] at h1
        rw [List.append_nil] at hd
        rw [hd] at hL₀
        obtain ⟨hrx, -⟩ := List.cons.inj (List.append_cancel_left (hL₀.symm.trans he))
        rw [hrx, hxn] at hr₀
        obtain ⟨-, hss, -⟩ := Frame.opening.inj hr₀
        rw [← hss, hs] at h1
        exact rep_ne_session h1
      | cons y d =>
        rw [hd, List.append_assoc, List.cons_append] at hL₀
        obtain ⟨hxy, -⟩ := List.cons.inj (List.append_cancel_left (hL₀.symm.trans he))
        rw [hd, hxy] at h₀
        exact firstOpened_ne_none m hxn d h₀
    have hfm := hfo m _ he hlt
    obtain ⟨sp, hsp, -⟩ := id hsv
    have hname : ((helloSpecs m)[i]?.bind fun sp => sp.streams[s]?) =
        tables[t.val]?.map (repStreamName · r.val) := by
      have hsp' : (helloSpecs Lg)[i]? = some sp := by
        rw [he]
        exact helloSpecs_getElem?_append hsp _
      rw [hsp, Option.bind_some]
      rw [hsp', Option.bind_some] at hs
      exact hs
    have hk : openedOf m i s < openedOf Lg i s := by
      rw [he]
      exact openedOf_lt m hxn L₂
    obtain ⟨j, hj⟩ := hcovL _ hk
    have hpick : (readingAt tables L record recordDigest).round j m x.frame := by
      show slotRound tables (L j) m x.frame
      rw [hj]
      exact ⟨i, s, n, hxn, hsv, hfm, hname, rfl⟩
    have hmm : m ∈ ms := by
      rcases hcov j m ⟨x, L₂, he, hpick⟩ with e | h
      · rw [e] at hlt
        exact absurd hlt (lt_irrefl _)
      · exact h
    obtain ⟨⟨q, sub, cs⟩, hz⟩ := forall₂_zip_right hF hmm
    obtain ⟨hsub0, hlog0⟩ := List.forall₂_zip hF hz
    have hsub : (decodersAt reg L).sub J q.toStop.modes = some sub := hsub0
    obtain ⟨j', i', s', n', b', L'', -, hall, hP', he'⟩ := hlog0
    obtain ⟨hxe, -⟩ := List.cons.inj (List.append_cancel_left (he.symm.trans he'))
    rw [hxe] at hpick
    obtain ⟨t', ht', r', k', mm, hLj', -, hsa⟩ := stop_slot reg hL ht record recordDigest hsub hall hP'
    rw [readingAt_unique hL ht record recordDigest hP' hpick, hj] at hLj'
    obtain ⟨ht't, hr'r, -⟩ := Slot.rep.inj hLj'
    refine ⟨((q, ⟨sub, cs⟩), m), hz, decide_eq_true ?_, rfl⟩
    show sub.tab = t ∧ sub.rep = r
    rw [hsa]
    exact ⟨Fin.ext ht't.symm, hr'r.symm⟩

end Stream

/-! ## The rounds of one stream -/

/-- **The elements of a zip whose first `p` holds of, first parts only, are the first list's that `p` holds of**,
where the lists are of one length. -/
theorem map_fst_filter_zip {α : Type _} {β : Type _} (p : α → Bool) :
    ∀ (xs : List α) (ys : List β), xs.length = ys.length →
      ((xs.zip ys).filter fun x => p x.1).map Prod.fst = xs.filter p
  | [], _, _ => by simp
  | _ :: _, [], h => by simp at h
  | x :: xs, y :: ys, h => by
    have ih := map_fst_filter_zip p xs ys (by simp only [List.length_cons] at h; omega)
    rw [List.zip_cons_cons, List.filter_cons, List.filter_cons]
    by_cases hx : p x = true
    · rw [if_pos (show p (x, y).1 = true from hx), if_pos hx, List.map_cons]
      exact congrArg (x :: ·) ih
    · rw [if_neg (show ¬ p (x, y).1 = true from hx), if_neg hx]
      exact ih

/-- **The rounds of the elements `p` holds of are one per log `q` holds after, in order**: where the logs `ms` beside
the elements `xs` whose first `p` holds of are the logs of `Lg` before each release `q` holds of (`stream_logs`), and
each such element's round is `f` at the count of those releases its log has, the rounds of `xs`'s elements `p` holds
of are `f` at none, one, and so on up to the count of them in `Lg`. -/
theorem rounds_of_logs {γ : Type _} {δ : Type _} {xs : List γ} {ms : List (List Release)} (hlen : xs.length = ms.length)
    (p : γ → Bool) (rd : γ → δ) (f : ℕ → δ) {q : Release → Bool} {Lg : List Release}
    (h : ((xs.zip ms).filter fun x => p x.1).map Prod.snd = prefixesWhere q Lg)
    (hf : ∀ a m, (a, m) ∈ xs.zip ms → p a = true → rd a = f (m.countP q)) :
    (xs.filter p).map rd = (List.range (Lg.countP q)).map f := by
  rw [← map_fst_filter_zip p xs ms hlen, List.map_map, ← map_countP_prefixesWhere q Lg, ← h, List.map_map,
    List.map_map]
  refine List.map_congr_left fun x hx => ?_
  obtain ⟨hx, hpx⟩ := List.mem_filter.1 hx
  exact hf x.1 x.2 hx hpx

section StreamRounds

variable {g : Game Exchange} {σ : Strategy g} {R J : ℕ} {Reg : Type}

/-- **The rounds the answered stops leave on table `t`'s rep `r` stream are one per opening of it**, at the readings
and the decoders at a layout `L`, injective, of distinct tables, under `stream_logs`'s premises: where the round each
answered stop on that stream appends is `f` at the count of the stream's openings its log has, the stream's rounds are
`f` at none, one, and so on up to the count of them in `Lg`. -/
theorem stream_rounds (reg : Option Claims.Msg → Reg) {tables : Array String} {L : Fin R → Slot}
    (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool) (dg : ByteArray → ByteArray)
    {as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))} {ms : List (List Release)}
    {Lg L₀ L' : List Release}
    (hF : List.Forall₂ (fun a m => (decodersAt reg L).sub J a.1.toStop.modes = some a.2.1 ∧
      LoggedAfter (readingAt tables L record recordDigest).round a.1.toStop m (bitsOf a.2.1 a.2.2) Lg) as ms)
    (hgt : ∀ m ∈ ms, L₀.length < m.length) (hpw : ms.Pairwise (fun m m' => m.length < m'.length))
    (hcov : ∀ j L₁, PicksIn (readingAt tables L record recordDigest).round Lg j L₁ → L₁ = L₀ ∨ L₁ ∈ ms)
    (hS : LogServed Lg) (hn : ∀ sp ∈ helloSpecs Lg, sp.streams.toList.Nodup) {r₀ : Release} {i s₀ n₀ : ℕ}
    (hL₀ : Lg = L₀ ++ r₀ :: L') (hr₀ : r₀.frame = .opening i s₀ n₀) (h₀ : firstOpened L₀ = none)
    (hses : ((helloSpecs L₀)[i]?.bind fun sp => sp.streams[s₀]?) =
      some _root_.Spot.Challenger.TREE_SESSION)
    (t : Fin J) (r : Fin 2) {s : ℕ}
    (hs : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = tables[t.val]?.map (repStreamName · r.val))
    (hcovL : ∀ k < openedOf Lg i s, ∃ j, L j = .rep t.val r k) (f : ℕ → Flock.Round)
    (hf : ∀ a m, (a, m) ∈ as.zip ms → a.2.1.tab = t ∧ a.2.1.rep = r →
      roundOf dg a.2.1.bytes a.2.1.n a.2.2 = f (openedOf m i s)) :
    as.foldl (fun rs a => addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2))
      (fun _ _ => []) t r = (List.range (openedOf Lg i s)).map f := by
  refine (foldl_addRoundJ (fun a => a.2.1.tab) (fun a => a.2.1.rep) (fun a => roundOf dg a.2.1.bytes a.2.1.n a.2.2)
    as (fun _ _ => []) t r).trans ((List.nil_append _).trans ?_)
  exact rounds_of_logs hF.length_eq (fun a => decide (a.2.1.tab = t ∧ a.2.1.rep = r))
    (fun a => roundOf dg a.2.1.bytes a.2.1.n a.2.2) f
    (stream_logs reg hL ht record recordDigest bitsOf hF hgt hpw hcov hS hn hL₀ hr₀ h₀ hses t r hs hcovL)
    fun a m hx hp => hf a m hx (of_decide_eq_true hp)

end StreamRounds

/-! ## No more openings than the record's rounds -/

/-- **A log that opens only the record's rounds opens each named stream at most the rounds the record gives it**:
where the device serves each opening of the log `Lg` and, before each, `Lg` has opened that stream fewer times than
the rounds `record` gives it (`OpenedRecorded`), stream `s` of the `i`-th tree, named `name`, is opened at most
`recordRounds record name` times. -/
theorem openedOf_le_recordRounds {record : Lean.Json} {Lg : List Release} (hR : OpenedRecorded record Lg)
    (hS : LogServed Lg) {i s : ℕ} {name : String}
    (hname : ((helloSpecs Lg)[i]?.bind fun sp => sp.streams[s]?) = some name) :
    openedOf Lg i s ≤ recordRounds record name := by
  suffices h : ∀ pre post, Lg = pre ++ post → openedOf pre i s ≤ recordRounds record name from
    h Lg [] (List.append_nil Lg).symm
  intro pre
  induction pre using List.reverseRecOn with
  | nil => intro _ _; exact Nat.zero_le _
  | append_singleton pre r ih =>
    intro post he
    rw [List.append_assoc, List.singleton_append] at he
    rw [openedOf_append]
    by_cases hr : r.frame.opensAt i s = true
    · obtain ⟨n, hn⟩ := opening_of_opensAt hr
      rw [if_pos hr]
      obtain ⟨sp, hsp, hs, -⟩ := hS pre r post i s n he hn
      have h1 : ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = some sp.streams[s] := by
        rw [hsp, Option.bind_some, Array.getElem?_eq_getElem hs]
      have h2 := helloSpecs_bind_append h1 (r :: post)
      rw [← he, hname] at h2
      have hlt := hR pre r post he i s n hn name (h1.trans h2.symm)
      omega
    · rw [if_neg hr]
      exact ih (r :: post) he

end Verity.Sampled
