import Proofs.Properties.Verity.Sampled.Readings

/-!
# Sampled proofs' coupling: a stop's written coins in the play's log

Where the projection writes a round's coins into a stop's opening, the play on the tape the trace ends on releases them
there.

* **The walk keeps a coin it passes** (`Pos.walk_le`, `Pos.le_toStop`, `Pos.walk_coin_keeps`): the walk extends the
  position at any cuts, and below a coin node no cut fixes, every tape through the walk's stop holds there the coin the
  node's tape holds.
* **A trace keeps the coin at its start** (`RoundsReach.le_tape`, `RoundsReach.head`): the tape a trace ends on is one
  through its start, and from a coin node no cut fixes, it holds there the coin the node's tape holds.
* **An opening the device serves, at its coin** (`optOpenK_next`, `OpeningAt.ofNext`, `openingOf_some`, `run_headCoin`):
  the node after an opening frame the device answers is that opening's coin node, and the play from it is the play
  after the coin a tape holds there.
* **The release in the log** (`playReaches_relK`, `log_of_evidenceG`, `log_of_opening`, `opened_of_head`,
  `RoundsReach.release`): from an opening's coin node whose releases continue to the evidence after a log `pre`, every
  play's log is `pre`, the opening answered with the release of the coin the tape holds there, then the rest. So off
  the opening's empty class, the play on the tape a trace from the stop ends on releases there the bits the coin map
  wrote.
* **Each stop's written coins in the log** (`RoundsReach.answers`, `StopsAt`, `StopsAt.next`, `LoggedAt`,
  `RoundsReach.stop_logged`, `RoundsReach.answers_logged`, `stopsAt_firstStop`, `firstStop_logged`): a trace's
  answered stops, each where `StopOf` says from a log and the evidence after it, the first from the empty log; where
  every round's reading picks only openings the device serves, the play on the tape the trace ends on releases, at a
  stop where not every cut is past, the bits written there (the link points' at the first stop, a round's coins at an
  answered one), at an opening the rounds whose cuts the stop fixes pick.
* **One round a stop** (`DigestCut.modeHere_eq_none`, `sub_of_single`, `stop_slot`, `LoggedAt.slot`): at the readings
  and decoders at an injective layout of distinct tables, a stop's submission is its one fixed round's, a rep round,
  with its most likely message, and its bits are released at that round's opening of its rep stream.
* **Each opening the device logs, served** (`LogServed`, `Served.of_next`, `logServed_relK`, `logServed_run`,
  `link_picks`): every opening of a play's log is one the device serves after the releases before it, so a log with an
  opening has a first one, which the link's slot picks.
* **The first stop fixes a cut** (`RoundsReach.run_ret`, `firstStop_skip_log`, `firstStop_fixes`): where every cut is
  past at the first stop no round picks in the play's log, so where the log has an opening, not every cut is past.
-/

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

/-- **No cut is fixed at a consumed position.** -/
theorem isHere_consume (p : Pos g σ V ι) (i : ι) : (p.consume.Ks i).isHere = false := by
  show (if (p.Ks i).isHere then (.skip : DigestCut V p.g') else p.Ks i).isHere = false
  by_cases h : (p.Ks i).isHere = true
  · rw [ite_eq_left h]
    rfl
  · rw [ite_eq_right h]
    simpa using h

end Pos

/-- **The play from a coin node is the play after the coin the tape holds there.** -/
theorem run_headCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (s' : Strategy g') (z : CoinTape g' s') :
    ∃ (s₀ : Strategy (k (headCoin hg s' z))) (y : CoinTape (k (headCoin hg s' z)) s₀),
      CoinTape.run g' s' z = CoinTape.run (k (headCoin hg s' z)) s₀ y := by
  subst hg
  exact ⟨s' z.1, z.2 z.1, rfl⟩

/-- **An output a play reaches, at an equal game.** -/
theorem playReaches_of_eq {g g' : Game α} (e : g = g') (s : Strategy g) {o : α} (h : PlayReaches g s o) :
    ∃ s' : Strategy g', PlayReaches g' s' o := by
  subst e
  exact ⟨s, h⟩

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ addRoundJ)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

/-! ## An opening the device serves, at its coin -/

/-- **The node after an opening frame the device answers is the opening's coin node.** -/
theorem optOpenK_next {β : Type} {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {i s n : ℕ} {b : ByteArray} {T : OpenTree} (hT : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p) :
    optOpenK k fs ts f i s n b ts[i]? = treeOpenK k fs ts f i s n b T (some p) := by
  rw [hT]
  show treeOpenK k fs ts f i s n b T (T.next s n) = _
  rw [hp]

/-- **The node after an opening frame the device answers, described**: the tree `T` it holds and the round `p` it
opens. -/
def OpeningAt.ofNext {β : Type} {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {i s n : ℕ} {b : ByteArray} {T : OpenTree} (hT : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p) :
    OpeningAt (optOpenK k fs ts f i s n b ts[i]?) :=
  ⟨T, p, i, s, n, b, k, fs, ts, f, _, _, (optOpenK_next hT hp).trans rfl⟩

/-- **A node with a description is one `openingOf` describes.** -/
theorem openingOf_some {β : Type} {g' : Game β} (o' : OpeningAt g') : ∃ o, openingOf g' = some o :=
  ⟨Classical.choice ⟨o'⟩, by
    unfold openingOf
    exact dite_eq_left ⟨o'⟩⟩

/-! ## The release in the log -/

/-- **An output a play of the device's releases reaches is one a play of the continuation at some log reaches.** -/
theorem playReaches_relK {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (f : List Release → Game β) (s : Strategy (relK k fs ts f)) {o : β} (h : PlayReaches (relK k fs ts f) s o) :
    ∃ L, ∃ s' : Strategy (f L), PlayReaches (f L) s' o := by
  obtain ⟨s', h'⟩ := playReaches_of_eq (releases_bind k fs ts f).symm s h
  exact playReaches_bind _ s' h'

/-- **The evidence after a log ends the play at that log.** -/
theorem log_of_evidenceG {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {log : List Release}
    {s : Strategy (evidenceG θ F c log)} {ex : Exchange} (h : PlayReaches (evidenceG θ F c log) s ex) :
    ex.log = log := by
  have he : ex = ⟨θ, F, c, log, s.1.1, s.1.2⟩ := h
  rw [he]

/-- **The release at an opening's coin node**: from the node after an opening frame of the `i`-th tree's stream `s`
that the device answers, where the releases continue to the evidence after the log `pre` and them, every play's log is
`pre`, then the opening answering `b` with the release of the coin the tape holds there, then the rest. -/
theorem log_of_opening {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {k : ℕ} {fs : List (String × ℕ)}
    {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release} {i s n : ℕ} {b : ByteArray}
    (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) {T : OpenTree} (hT : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p)
    (s' : Strategy (optOpenK k fs ts f i s n b ts[i]?)) (z : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s') :
    ∃ L, (CoinTape.run _ s' z).log = pre ++ ⟨.opening i s n, b,
      TreeDraw.opened (headCoin (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp).eq s' z).1 n⟩ :: L := by
  obtain ⟨s₀, y, hrun⟩ := run_headCoin (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp).eq s' z
  have h₀ : PlayReaches _ s₀ (CoinTape.run (optOpenK k fs ts f i s n b ts[i]?) s' z) := by
    rw [hrun]
    exact run_reaches _ s₀ y
  obtain ⟨L, s₁, h₁⟩ := playReaches_relK k fs (ts.set i (T.bump p.1)) _ s₀ h₀
  obtain ⟨s₂, h₂⟩ := playReaches_of_eq (hf _) s₁ h₁
  exact ⟨L, log_of_evidenceG h₂⟩

/-- **The release of a coin a tape keeps from a written one is the written bits**: at the node after an opening frame
the device answers, off the opening's empty class at `bits`, where a tape holds the coin the tape with `bits` written
in holds, its coin there is released as `bits`. -/
theorem opened_of_head {β : Type} {k : ℕ} {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game β}
    {i s n : ℕ} {b : ByteArray} {T : OpenTree} (hT : ts[i]? = some T)
    {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p)
    {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)} {bits : List Bool}
    (hE : ¬ EmptyAt (optOpenK k fs ts f i s n b ts[i]?) bits) (x'' z : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'')
    (hh : headCoin (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp).eq s'' z =
      headCoin (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp).eq s'' (writeOpening s'' bits x'')) :
    TreeDraw.opened (headCoin (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp).eq s'' z).1 n = bits := by
  obtain ⟨oa, hoa⟩ := openingOf_some (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp)
  rw [hh]
  exact opened_writeOpening_off s'' bits x'' hE oa hoa (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hT hp)

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
evidence after `pre` and them, off the opening's empty class at `bits`, the play on the tape any trace from the stop,
`bits` written into its opening and consumed, ends on has as its log `pre`, then the opening answering `b` with `bits`,
then the rest. -/
theorem RoundsReach.release {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {k : ℕ} {fs : List (String × ℕ)}
    {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release} {i s n : ℕ} {b : ByteArray}
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)}
    {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)}
    {Ks : Fin R → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?)}
    {x'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s''}
    {back'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'' → CoinTape g σ}
    (hq : q = ⟨optOpenK k fs ts f i s n b ts[i]?, s'', Ks, x'', back''⟩) (hi : q.Inv Ks₀)
    (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) (hT : TreesOf pre ts) (hS : Served pre i s n)
    {bits : List Bool} (hE : ¬ EmptyAt q.g' bits) {β : Type} {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray}
    {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite (writeOpening q.s' bits)).consume o) :
    ∃ L, (CoinTape.run g σ o.2).log = pre ++ ⟨.opening i s n, b, bits⟩ :: L := by
  subst hq
  obtain ⟨T, hTi, _, _, hp, -⟩ := Served.next hT hS
  obtain ⟨z, hz, hh⟩ := hr.head (OpeningAt.ofNext (k := k) (fs := fs) (f := f) (b := b) hTi hp).eq
    (Pos.isHere_consume _)
  have hw := opened_of_head hTi hp hE x'' z hh
  have hX : CoinTape.run g σ o.2 = CoinTape.run (optOpenK k fs ts f i s n b ts[i]?) s'' z := by
    rw [hz]
    exact hi.run z
  obtain ⟨L, hL⟩ := log_of_opening hf hTi hp s'' z
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

end Trace

/-! ## Where each stop of a trace is -/

section StopsAt

variable {g : Game Exchange} {σ : Strategy g} {R : ℕ}

/-- **Where a position's walk stops, at a log and the evidence after it**: as `StopOf` says, from some log `pre`, at
releases continuing to the evidence after `pre` and them. -/
def StopsAt (P : Fin R → List Release → Frame → Prop) (rv : Fin R → Exchange → Option Claims.Msg) (θ : ByteArray)
    (F : Array ByteArray) (c : ByteArray) (q : Pos g σ Claims.Msg (Fin R)) : Prop :=
  ∃ (pre : List Release) (f : List Release → Game Exchange) (a : Fin R → Prop),
    (∀ log, f log = evidenceG θ F c (pre ++ log)) ∧ StopOf P rv pre f a q

/-- **The walk from a stop, consumed, stops at a log and the evidence after it**, on any rewriting of the tape there,
where each round's reading picks only openings. -/
theorem StopsAt.next {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n) {q : Pos g σ Claims.Msg (Fin R)}
    (h : StopsAt P rv θ F c q) (φ : CoinTape q.g' q.s' → CoinTape q.g' q.s') :
    StopsAt P rv θ F c (q.rewrite φ).consume.toStop := by
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

/-- **The bits `bits` released in the log `log` at an opening the rounds `q` fixes pick**: after a log `pre`, an
opening of the `i`-th tree's stream `s` that round `j`, whose cut `q` fixes, picks there, as does every round whose cut
`q` fixes, answered with `bits`. -/
def LoggedAt (P : Fin R → List Release → Frame → Prop) (q : Pos g σ Claims.Msg (Fin R)) (bits : List Bool)
    (log : List Release) : Prop :=
  ∃ (j : Fin R) (pre : List Release) (i s n : ℕ) (b : ByteArray) (L : List Release),
    (q.Ks j).isHere = true ∧ (∀ j', (q.Ks j').isHere = true → P j' pre (.opening i s n)) ∧
      P j pre (.opening i s n) ∧ log = pre ++ ⟨.opening i s n, b, bits⟩ :: L

/-- **A stop's written coins are in the play's log, at an opening its fixed rounds pick**: at a position that keeps
the invariant at the cuts `Ks₀`, where `StopsAt` says and not every cut is past, where each round's reading picks only
openings the device serves, off the opening's empty class at `bits`, the play on the tape any trace from the position,
`bits` written into its opening and consumed, ends on releases `bits` at an opening the rounds whose cuts the position
fixes pick (`LoggedAt`). -/
theorem RoundsReach.stop_logged {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀) (hS : StopsAt P rv θ F c q) (hK : ¬ ∀ j, q.Ks j = .skip)
    {bits : List Bool} (hE : ¬ EmptyAt q.g' bits) {β : Type} {fin : LiveOutJ J mPts → β}
    {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round}
    {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite (writeOpening q.s' bits)).consume o) :
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
    obtain ⟨L', hL'⟩ := RoundsReach.release (θ := θ) (F := F) (c := c) hq hi
      (fun log => by rw [hf', hf, List.append_assoc]) hT (hPS j _ i s n hj) hE hr
    exact ⟨j, pre ++ L, i, s, n, b, L', hjK, hall, hj, hL'⟩

/-- **Each answered stop's coins are in the play's log**: on a trace from a position that keeps the invariant at the
cuts `Ks₀` and whose walk stops where `StopsAt` says, where each round's reading picks only openings the device serves,
the decoders submit nothing at a stop that fixes no cut, the coin map writes a round's coins into the stop's opening as
their bits `bitsOf`, and off `B` the opening's class at those bits is not empty, the play on the tape the trace ends on
releases, at each answered stop (`RoundsReach.answers`), the bits of the stop's round's coins at an opening the rounds
whose cuts the stop fixes pick (`LoggedAt`, `RoundsReach.stop_logged`). -/
theorem RoundsReach.answers_logged {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) (hsub : D.sub J (fun _ => none) = none)
    (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool)
    (hM : ∀ q s c, M.round q J s c = writeOpening q.s' (bitsOf s c))
    (hB : ∀ g' s c, ¬ B g' J s c → ¬ EmptyAt g' (bitsOf s c)) {Ks₀ : Fin R → DigestCut Claims.Msg g} {β : Type}
    {fin : LiveOutJ J mPts → β} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p : Pos g σ Claims.Msg (Fin R)} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs p o) (hi : p.Inv Ks₀) (hS : StopsAt P rv θ F c p.toStop) :
    ∃ as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))),
      o.1 = fin ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
          D.proofs (CoinTape.run g σ o.2)⟩ ∧
      ∀ a ∈ as, D.sub J a.1.toStop.modes = some a.2.1 ∧
        LoggedAt P a.1.toStop (bitsOf a.2.1 a.2.2) (CoinTape.run g σ o.2).log := by
  obtain ⟨as, h1, h2⟩ := hr.answers (fun q => q.Inv Ks₀ ∧ StopsAt P rv θ F c q.toStop) ⟨hi, hS⟩
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
  sub_of_single reg L J (fun j' hj' => Pos.modes_eq_none (by
    cases h : (q.Ks j').isHere
    · rfl
    · exact absurd (readingAt_unique hL ht record recordDigest (hall j' h) hP) hj')) hsub

/-- **A stop's logged bits are at its slot's opening**, at the readings and the decoders at the layout `L`, injective,
of distinct tables: where the decoders submit `sub` at the stop `q` and `bits` are released at an opening its fixed
rounds pick (`LoggedAt`), `sub` is round `k` of table `t`'s rep `r` stream with the most likely message of round `j`,
whose slot that is, and `bits` are released after a log whose first opening is of tree `i`, at an opening of that tree's
stream named table `t`'s rep `r`, which the log has opened `k` times. -/
theorem LoggedAt.slot {R J : ℕ} {Reg : Type} (reg : Option Claims.Msg → Reg) {tables : Array String}
    {L : Fin R → Slot} (hL : Function.Injective L) (ht : tables.toList.Nodup) (record : Exchange → Option Claims.Msg)
    (recordDigest : Claims.Msg) {g : Game Exchange} {σ : Strategy g} {q : Pos g σ Claims.Msg (Fin R)}
    {sub : SubmitJ J} (hsub : (decodersAt reg L).sub J q.modes = some sub) {bits : List Bool} {log : List Release}
    (h : LoggedAt (readingAt tables L record recordDigest).round q bits log) :
    ∃ (t : ℕ) (ht : t < J) (r : Fin 2) (k : ℕ) (m : Claims.Msg) (j : Fin R) (pre : List Release) (i s n : ℕ)
      (b : ByteArray) (L' : List Release), L j = .rep t r k ∧ q.modes j = some m ∧
        sub = ⟨⟨t, ht⟩, r, ⟨m.toArray⟩, tailCount m⟩ ∧ firstOpened pre = some i ∧
        ((helloSpecs pre)[i]?.bind fun sp => sp.streams[s]?) = tables[t]?.map (repStreamName · r.val) ∧
        openedOf pre i s = k ∧ log = pre ++ ⟨.opening i s n, b, bits⟩ :: L' := by
  obtain ⟨j, pre, i, s, n, b, L', -, hall, hP, he⟩ := h
  obtain ⟨t, ht', r, k, m, hLj, hm, hs⟩ := stop_slot reg hL ht record recordDigest hsub hall hP
  have hP' : slotRound tables (L j) pre (.opening i s n) := hP
  rw [hLj] at hP'
  obtain ⟨i', s', n', hfr, -, hfo, hname, hk⟩ := hP'
  obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj hfr
  exact ⟨t, ht', r, k, m, j, pre, _, _, _, b, L', hLj, hm, hs, hfo, hname, hk, he⟩

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

/-- **An opening the device's trees answer is one it serves**, where the trees are the ones the log shows
(`Served.next`'s converse). -/
theorem Served.of_next {pre : List Release} {ts : List OpenTree} (hT : TreesOf pre ts) {i s n : ℕ} {T : OpenTree}
    (hTi : ts[i]? = some T) {p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)} (hp : T.next s n = some p) :
    Served pre i s n := by
  obtain ⟨hs, ho, hn⟩ := OpenTree.next_some hp
  have hs' : s < T.spec.streams.size := by
    rw [← hs]
    exact p.1.isLt
  refine ⟨T.spec, ?_, hs', ?_, hn⟩
  · rw [← hT.specs, List.getElem?_map, hTi]
    rfl
  · rw [← hT.opened i s, openedIn_some hTi hs']
    have e : (⟨s, hs'⟩ : Fin T.spec.streams.size) = p.1 := Fin.ext hs.symm
    rw [e, ← ho]
    exact p.2.isLt

theorem logServed_end {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} {f : List Release → Game Exchange}
    {pre : List Release} (hf : ∀ log, f log = evidenceG θ F c (pre ++ log)) (hS : LogServed pre)
    (s : Strategy (f [])) {ex : Exchange} (h : PlayReaches (f []) s ex) : LogServed ex.log := by
  obtain ⟨s', h'⟩ := playReaches_of_eq (hf []) s h
  rw [log_of_evidenceG h', List.append_nil]
  exact hS

/-- **An output a play of an optional node reaches**, by the node's two cases. -/
theorem reaches_optGame {β γ : Type} {g₀ : Game β} {K : γ → Game β} {Q : β → Prop} (o : Option γ)
    (h0 : ∀ s e, PlayReaches g₀ s e → Q e) (h1 : ∀ x, o = some x → ∀ s e, PlayReaches (K x) s e → Q e)
    (s : Strategy (optGame o g₀ K)) (e : β) (h : PlayReaches (optGame o g₀ K) s e) : Q e := by
  cases o with
  | none => exact h0 s e h
  | some x => exact h1 x rfl s e h

/-- **Each opening a play of the device's releases logs is one it serves**: from a log each of whose openings the
device serves, at the trees that log shows, every play of the releases continuing to the evidence after the log and
them ends on a log each of whose openings the device serves after the releases before it. -/
theorem logServed_relK {θ : ByteArray} {F : Array ByteArray} {c : ByteArray} (k : ℕ) :
    ∀ (fs : List (String × ℕ)) (ts : List OpenTree) (f : List Release → Game Exchange) (pre : List Release),
      (∀ log, f log = evidenceG θ F c (pre ++ log)) → TreesOf pre ts → LogServed pre →
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
    · obtain ⟨t, ht⟩ := h
      exact ih fs (ts ++ [⟨b, spec, t, fun _ => 0⟩])
        (fun log => f (⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩ :: log))
        (pre ++ [⟨.hello spec, b, bitsOfBytes (treeRoot b spec t ++ treeKey t.1)⟩])
        (fun log => by simp only [hf, List.append_assoc, List.singleton_append]) (hT.hello b spec t _)
        (hS.snoc fun _ _ _ he => by cases he) (s₁ t) ex ht
    · refine reaches_optGame (Q := fun e => LogServed e.log) ts[i]?
        (fun s₀ _ h₀ => logServed_end hf hS s₀ h₀) (fun T hTi s₂ e₂ h₂ => ?_) s₁ ex h
      refine reaches_optGame (Q := fun e => LogServed e.log) (T.next s n)
        (fun s₀ _ h₀ => logServed_end hf hS s₀ h₀) (fun p hp s₃ e₃ h₃ => ?_) s₂ e₂ h₂
      obtain ⟨d, hd⟩ := h₃
      exact ih fs (ts.set i (T.bump p.1)) (fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log))
        (pre ++ [⟨.opening i s n, b, TreeDraw.opened d.1 n⟩])
        (fun log => by simp only [hf, List.append_assoc, List.singleton_append]) (hT.opening hTi hp b _)
        (hS.snoc fun _ _ _ he => by
          obtain ⟨rfl, rfl, rfl⟩ := Frame.opening.inj he
          exact Served.of_next hT hTi hp) (s₃ d) e₃ hd

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

/-- **The first stop is where `StopsAt` says**, from the empty log with every round live, where each round's reading
picks only openings. -/
theorem stopsAt_firstStop (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (ω : (A.extend M.widening).L.Ω)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopsAt rd.round rd.reveal θ F c (firstStop V θ F c A rd σ M ω) :=
  ⟨[], evidenceG θ F c, fun _ => True, fun _ => rfl,
    walk_cast hP (game_eq_relK V θ F c).symm σ (roundCuts V θ F c rd) (M.start ω) id TreesOf.nil
      fun _ => Or.inl ⟨trivial, rfl⟩⟩

/-- **The link points are in the play's log**: on a trace from the first stop, the link points written in and
consumed, where each round's reading picks only openings the device serves, not every cut is past at the first stop,
the coin map writes the link points into the stop's opening as their bits `bitsOf`, and off `BL` the opening's class at
those bits is not empty, the play on the tape the trace ends on releases the link points' bits at an opening the
rounds whose cuts the first stop fixes pick (`LoggedAt`, `RoundsReach.stop_logged`). -/
theorem firstStop_logged (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    {D : Decoders A.rounds A.Reg} (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP)
    {B : RoundHold} (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, rd.round j pre (.opening i s n) → Served pre i s n)
    (bitsOf : (Fin 2 → Fin A.mP → 𝔽) → List Bool) (hML : ∀ q pts, M.link q pts = writeOpening q.s' (bitsOf pts))
    (hBL : ∀ g' pts, ¬ BL g' pts → ¬ EmptyAt g' (bitsOf pts)) (ω : (A.extend M.widening).L.Ω)
    (pts : Fin 2 → Fin A.mP → 𝔽) (hK : ¬ ∀ j, (firstStop V θ F c A rd σ M ω).Ks j = .skip)
    (hL : ¬ BL (firstStop V θ F c A rd σ M ω).g' pts) {J N : ℕ} {β : Type} {fin : LiveOutJ J A.mP → β}
    {rootB : Array ByteArray} {o : β × CoinTape (V.game θ F c) σ}
    (hr : RoundsReach D M A.dg J B fin rootB pts N (fun _ _ => [])
      ((firstStop V θ F c A rd σ M ω).rewrite (M.link (firstStop V θ F c A rd σ M ω) pts)).consume o) :
    LoggedAt rd.round (firstStop V θ F c A rd σ M ω) (bitsOf pts) (CoinTape.run (V.game θ F c) σ o.2).log := by
  rw [hML] at hr
  exact RoundsReach.stop_logged hPS (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))).2.2
    (stopsAt_firstStop V θ F c A rd σ M ω hP) hK (hBL _ _ hL) hr

/-- **Each opening a play logs is one the device serves** after the releases before it (`logServed_relK`). -/
theorem logServed_run (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ) :
    LogServed (CoinTape.run (V.game θ F c) σ x).log := by
  obtain ⟨s', h⟩ := playReaches_of_eq (game_eq_relK V θ F c) σ (run_reaches _ σ x)
  exact logServed_relK (θ := θ) (F := F) (c := c) _ _ _ (evidenceG θ F c) [] (fun _ => rfl) TreesOf.nil
    logServed_nil s' _ h

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
  obtain ⟨L, hL, h⟩ := walk_cast (rv := rd.reveal) (a := fun _ => True) hP (game_eq_relK V θ F c).symm σ
    (roundCuts V θ F c rd) (M.start ω) id TreesOf.nil fun _ => Or.inl ⟨trivial, rfl⟩
  rcases h with ⟨-, y, hy, s₀, x₀, hrun⟩ |
    ⟨k, fs, ts, f', i, s, n, b, -, -, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · have h₀ : PlayReaches (evidenceG θ F c L) s₀ y := by
      rw [← hrun]
      exact run_reaches _ s₀ x₀
    rw [RoundsReach.run_ret (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))).2.2 hy hr,
      log_of_evidenceG h₀]
    exact hL
  · have hjK : ((firstStop V θ F c A rd σ M ω).Ks j).isHere = true := by
      show ((Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) (M.start ω) id).Ks j).isHere = true
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

end First

end Verity.Sampled
