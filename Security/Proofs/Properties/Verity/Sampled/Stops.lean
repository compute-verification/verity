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
* **Each stop's written coins in the log** (`RoundsReach.answers`, `StopsAt`, `StopsAt.next`,
  `RoundsReach.stop_logged`, `RoundsReach.answers_logged`, `stopsAt_firstStop`, `firstStop_logged`): a trace's
  answered stops, each where `StopOf` says from a log and the evidence after it, the first from the empty log; where
  every round's reading picks only openings the device serves, the play on the tape the trace ends on releases, at a
  stop where not every cut is past, the bits written there (the link points' at the first stop, a round's coins at an
  answered one), at an opening a round whose cut the stop fixes picks.
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

/-- **A stop's written coins are in the play's log, at an opening a round picks**: at a position that keeps the
invariant at the cuts `Ks₀`, where `StopsAt` says and not every cut is past, where each round's reading picks only
openings the device serves, off the opening's empty class at `bits`, the play on the tape any trace from the position,
`bits` written into its opening and consumed, ends on releases `bits` at an opening, after a log at which a round whose
cut the position fixes picks that opening. -/
theorem RoundsReach.stop_logged {P : Fin R → List Release → Frame → Prop}
    {rv : Fin R → Exchange → Option Claims.Msg} {θ : ByteArray} {F : Array ByteArray} {c : ByteArray}
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀) (hS : StopsAt P rv θ F c q) (hK : ¬ ∀ j, q.Ks j = .skip)
    {bits : List Bool} (hE : ¬ EmptyAt q.g' bits) {β : Type} {fin : LiveOutJ J mPts → β}
    {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ} {rs : Fin J → Fin 2 → List Flock.Round}
    {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs (q.rewrite (writeOpening q.s' bits)).consume o) :
    ∃ (j : Fin R) (pre : List Release) (i s n : ℕ) (b : ByteArray) (L : List Release),
      (q.Ks j).isHere = true ∧ P j pre (.opening i s n) ∧
        (CoinTape.run g σ o.2).log = pre ++ ⟨.opening i s n, b, bits⟩ :: L := by
  obtain ⟨pre, f, a, hf, L, -, h⟩ := hS
  rcases h with ⟨hK', -⟩ | ⟨k, fs, ts, f', i, s, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · exact absurd hK' hK
  · have hjK : (q.Ks j).isHere = true := by
      rw [hq]
      exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
    obtain ⟨L', hL'⟩ := RoundsReach.release (θ := θ) (F := F) (c := c) hq hi
      (fun log => by rw [hf', hf, List.append_assoc]) hT (hPS j _ i s n hj) hE hr
    exact ⟨j, pre ++ L, i, s, n, b, L', hjK, hj, hL'⟩

/-- **Each answered stop's coins are in the play's log**: on a trace from a position that keeps the invariant at the
cuts `Ks₀` and whose walk stops where `StopsAt` says, where each round's reading picks only openings the device serves,
the decoders submit nothing at a stop that fixes no cut, the coin map writes a round's coins into the stop's opening as
their bits `bitsOf`, and off `B` the opening's class at those bits is not empty, the play on the tape the trace ends on
releases, at each answered stop (`RoundsReach.answers`), an opening answered with the bits of the stop's round's coins,
after a log at which a round whose cut the stop fixes picks that opening (`RoundsReach.stop_logged`). -/
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
        ∃ (j : Fin R) (pre : List Release) (i s n : ℕ) (b : ByteArray) (L : List Release),
          (a.1.toStop.Ks j).isHere = true ∧ P j pre (.opening i s n) ∧
            (CoinTape.run g σ o.2).log = pre ++ ⟨.opening i s n, b, bitsOf a.2.1 a.2.2⟩ :: L := by
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
those bits is not empty, the play on the tape the trace ends on releases the link points' bits at an opening, after a
log at which a round whose cut the first stop fixes picks that opening (`RoundsReach.stop_logged`). -/
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
    ∃ (j : Fin A.rounds) (pre : List Release) (i s n : ℕ) (b : ByteArray) (L : List Release),
      ((firstStop V θ F c A rd σ M ω).Ks j).isHere = true ∧ rd.round j pre (.opening i s n) ∧
        (CoinTape.run (V.game θ F c) σ o.2).log = pre ++ ⟨.opening i s n, b, bitsOf pts⟩ :: L := by
  rw [hML] at hr
  exact RoundsReach.stop_logged hPS (Pos.toStop_spec (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))).2.2
    (stopsAt_firstStop V θ F c A rd σ M ω hP) hK (hBL _ _ hL) hr

end First

end Verity.Sampled
