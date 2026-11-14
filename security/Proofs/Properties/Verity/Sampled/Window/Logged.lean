module

public import Proofs.Properties.Verity.Sampled.Window.Reach

/-!
# Sampled proofs' coupling over a window: each member's written coins in the play's log

Where the projection writes a member's link points and its rounds' coins into its stops' openings, the play on the
tape the window's walk ends on releases them there, whichever members' sessions follow.

* **The release in the log, past the stop** (`release_past`, `stop_logged_past`): `RoundsReach.release` and
  `RoundsReach.stop_logged` at any position past the stop with the bits written in and consumed (`Pos.Past`), where
  the one-session lemmas take a trace from it. A later session's walk goes on from the stop's coin node to its next
  stop, which keeps the coin written there.
* **A member's session keeps where the walk stops** (`MemberReach.stopsAt`, `stopsAt_rootStop`): the stop after a
  member's session's end is where `StopsAt` says, from a log and the close after it, as the stop its session starts
  at is; the play's first stop is, on any tape.
* **A member's written coins in the log** (`MemberReach.logged`): on a member's session's trace from a position
  whose stop is where `StopsAt` says and fixes a cut, where the coin map writes the link points and each round's
  coins into the stops' openings as bits and off the held-off events the openings' classes at them are not empty, the
  play on the tape of any position past the trace's end releases the link points' bits and each answered round's
  coins at an opening the rounds whose cuts the stop fixes pick, and the session's outcome is those rounds in order.
* **The members' steps, at where the walk stops** (`MemberSteps`, `memberDecodes_of_stops`): `MemberDecodes` at any
  invariant, beside the play's start kept and `StopsAt` at the stop a member's session starts at, both handed to each
  member's step.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ addRoundJ)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

/-! ## The release in the log, past the stop -/

section Past

variable {g : Game Exchange} {σ : Strategy g} {R : ℕ}

/-- **A stop's written release is in the play's log, at any position past the stop**: at a stop after an opening
frame of the `i`-th tree's stream `s` that the device serves after the log `pre`, its releases continuing to the close
after `pre` and them, off the opening's empty class at `bits`, the play on the tape of any position past the stop
with `bits` written into its opening and consumed has as its log `pre`, then the opening answering `b` with `bits`,
then the rest (`RoundsReach.release`, whose trace ends at such a position). -/
theorem release_past {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray} {k : ℕ}
    {fs : List (String × ℕ)} {ts : List OpenTree} {f : List Release → Game Exchange} {pre : List Release} {i s n : ℕ}
    {b : ByteArray} {Ks₀ : Fin R → DigestCut Claims.Msg g} {q : Pos g σ Claims.Msg (Fin R)}
    {s'' : Strategy (optOpenK k fs ts f i s n b ts[i]?)}
    {Ks : Fin R → DigestCut Claims.Msg (optOpenK k fs ts f i s n b ts[i]?)}
    {x'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s''}
    {back'' : CoinTape (optOpenK k fs ts f i s n b ts[i]?) s'' → CoinTape g σ}
    (hq : q = ⟨optOpenK k fs ts f i s n b ts[i]?, s'', Ks, x'', back''⟩) (hi : q.Inv Ks₀)
    (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log)) (hT : TreesOf pre ts) (hS : Served pre i s n)
    {bits : List Bool} (hE : ¬ EmptyAt (finish sg θ F c rq) q.g' bits) {r : Pos g σ Claims.Msg (Fin R)}
    (hr : (q.rewrite (writeOpening (finish sg θ F c rq) q.s' bits)).consume.Past r) :
    ∃ L, (CoinTape.run g σ r.tape).log = pre ++ ⟨.opening i s n, b, bits⟩ :: L := by
  subst hq
  obtain ⟨T, hTi, _, _, hp, -⟩ := Served.next hT hS
  have hc := Served.cap hT hS
  obtain ⟨z, hz, hh⟩ := hr.head (OpeningAt.ofNext (k := k) (b := b) hf hc hTi hp).eq (Pos.isHere_consume _)
  have hw := opened_of_head (finish_recordsLog sg θ F c rq) hf hc hTi hp hE x'' z hh
  have hX : CoinTape.run g σ r.tape = CoinTape.run (optOpenK k fs ts f i s n b ts[i]?) s'' z := by
    rw [hz]
    exact hi.run z
  obtain ⟨L, hL⟩ := log_of_opening hf hc hTi hp s'' z
  refine ⟨L, ?_⟩
  rw [hX, hL, hw]

/-- **A stop's written coins are in the play's log, at an opening its fixed rounds pick, at any position past the
stop**: at a position that keeps the invariant at the cuts `Ks₀`, where `StopsAt` says and not every cut is past,
where each round's reading picks only openings the device serves, off the opening's empty class at `bits`, the play
on the tape of any position past the position with `bits` written into its opening and consumed releases `bits` at an
opening the rounds whose cuts the position fixes pick (`LoggedAt`; `RoundsReach.stop_logged`, whose trace ends at
such a position). -/
theorem stop_logged_past {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) {Ks₀ : Fin R → DigestCut Claims.Msg g}
    {q : Pos g σ Claims.Msg (Fin R)} (hi : q.Inv Ks₀) (hS : StopsAt P rv sg θ F c rq q)
    (hK : ¬ ∀ j, q.Ks j = .skip) {bits : List Bool} (hE : ¬ EmptyAt (finish sg θ F c rq) q.g' bits)
    {r : Pos g σ Claims.Msg (Fin R)}
    (hr : (q.rewrite (writeOpening (finish sg θ F c rq) q.s' bits)).consume.Past r) :
    LoggedAt P q bits (CoinTape.run g σ r.tape).log := by
  obtain ⟨pre, f, a, hf, L, -, h⟩ := hS
  rcases h with ⟨hK', -⟩ | ⟨k, fs, ts, f', i, s, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩
  · exact absurd hK' hK
  · have hjK : (q.Ks j).isHere = true := by
      rw [hq]
      exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
    have hall : ∀ j', (q.Ks j').isHere = true → P j' (pre ++ L) (.opening i s n) := fun j' h' => by
      rw [hq] at h'
      exact ((isHere_openCuts hKs j').1 h').2
    obtain ⟨L', hL'⟩ := release_past (sg := sg) (θ := θ) (F := F) (c := c) (rq := rq) hq hi
      (fun log => by rw [hf', hf, List.append_assoc]) hT (hPS j _ i s n hj) hE hr
    exact ⟨pre ++ L, j, i, s, n, b, L', hjK, hall, hj, hL'⟩

end Past

/-! ## A member's written coins in the log -/

section Member

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} {D : ℕ → PosDecoders g σ R Reg}
  {M : WindowCoinMap g σ Ω R} {BL : WindowLinkHold} {B : WindowRoundHold} {S : Finset (Fin nJ)} {Rg : Reg}

/-- **The stop after a member's session's end is where `StopsAt` says**, from a log and the close after it, where the
stop its session starts at is and each round's reading picks only openings (`StopsAt.next` at each consumed stop). -/
theorem MemberReach.stopsAt {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n) {i : ℕ}
    {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out)
    (hS : StopsAt P rv sg θ F c rq p.toStop) : StopsAt P rv sg θ F c rq q.toStop :=
  h.keeps (fun q => StopsAt P rv sg θ F c rq q.toStop) hS (fun q φ hq => StopsAt.next hP hq φ) fun q hq => by
    show StopsAt P rv sg θ F c rq q.toStop.toStop
    rw [Pos.toStop_toStop]
    exact hq

/-- **A member's written coins are in the play's log**: on the trace of the session of the member at place `i` from a
position that keeps the invariant at the cuts `Ks₀`, whose stop is where `StopsAt` says and where not every cut is
past, where each round's reading picks only openings the device serves, the member's decoders submit nothing at a stop
that fixes no cut, the member's coin map writes the link points and a round's coins into a stop's opening as their
bits (`bitsL`, `bitsOf`), and off the member's held-off events the opening's class at those bits is not empty: the
session's outcome is the root decoded at its first stop, link points `pts`, the answered submissions' rounds in order
and the member's proof files at its end; and the play on the tape of any position `r` past the trace's end (the
position the last member's session ends at is one) releases `pts`' bits at an opening the rounds whose cuts the first
stop fixes pick, and each answered submission's coins at an opening the rounds whose cuts its stop fixes pick
(`firstStop_logged` and `RoundsReach.answers_logged`, at a member of a window). -/
theorem MemberReach.logged {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) {i : ℕ}
    {m : Flock.Guarantees.MemberV5 Reg nJ} (hsub : (D i).sub (memberTabs S Rg m) (fun _ => none) = none)
    (bitsL : (Fin 2 → Fin (memberPts m) → 𝔽) → List Bool)
    (hML : ∀ q pts, (M.member i (memberPts m)).link q pts = writeOpening (finish sg θ F c rq) q.s' (bitsL pts))
    (hBL : ∀ g' pts, ¬ BL i (memberPts m) g' pts → ¬ EmptyAt (finish sg θ F c rq) g' (bitsL pts))
    (bitsOf : (s : SubmitJ (memberTabs S Rg m)) → (Fin s.n → 𝔽) → List Bool)
    (hM : ∀ q s x, (M.member i (memberPts m)).round q (memberTabs S Rg m) s x =
      writeOpening (finish sg θ F c rq) q.s' (bitsOf s x))
    (hB : ∀ g' s x, ¬ B i g' (memberTabs S Rg m) s x → ¬ EmptyAt (finish sg θ F c rq) g' (bitsOf s x))
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {p q r : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out)
    (hqr : q.Past r) (hi : p.Inv Ks₀) (hS : StopsAt P rv sg θ F c rq p.toStop)
    (hK : ¬ ∀ j, p.toStop.Ks j = .skip) :
    ∃ (pts : Fin 2 → Fin (memberPts m) → 𝔽)
      (as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ (memberTabs S Rg m)) × (Fin s.n → 𝔽)))),
      out = ⟨(D i).root p.toStop.modes, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf m.dg a.2.1.bytes a.2.1.n a.2.2)) (fun _ _ => []),
          (D i).files q⟩ ∧
      LoggedAt P p.toStop (bitsL pts) (CoinTape.run g σ r.tape).log ∧
      ∀ a ∈ as, (D i).sub (memberTabs S Rg m) a.1.toStop.modes = some a.2.1 ∧
        LoggedAt P a.1.toStop (bitsOf a.2.1 a.2.2) (CoinTape.run g σ r.tape).log := by
  obtain ⟨pts, hL, he⟩ := h
  have hi' := (Pos.toStop_spec hi).2.2
  rw [hML] at he
  have hlink := stop_logged_past hPS hi' hS hK (hBL _ _ hL) (he.past.trans hqr)
  obtain ⟨as, h1, h2⟩ := he.answers (fun q => q.Inv Ks₀ ∧ StopsAt P rv sg θ F c rq q.toStop)
    ⟨Pos.inv_consume (Pos.inv_rewrite hi' _), StopsAt.next hP hS _⟩
    fun q φ hq => ⟨Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hq.1).2.2 φ), StopsAt.next hP hq.2 φ⟩
  refine ⟨pts, as, h1, hlink, fun a ha => ?_⟩
  obtain ⟨⟨hai, has⟩, hsa, hBa, hpa⟩ := h2 a ha
  refine ⟨hsa, ?_⟩
  rw [hM] at hpa
  refine stop_logged_past hPS (Pos.toStop_spec hai).2.2 has (fun hK' => ?_) (hB _ _ _ hBa) (hpa.trans hqr)
  have hm : a.1.toStop.modes = fun _ => none := funext fun j =>
    (congrArg (fun K => DigestCut.modeHere K a.1.toStop.s') (hK' j)).trans rfl
  rw [hm, hsub] at hsa
  cases hsa

end Member

/-! ## The decoders' premise at each member, at where the walk stops -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The play's first stop is where `StopOf` says**, on any tape: from the empty log, the close after `σ`'s request
and what the device releases, with every round live, where each round's reading picks only openings
(`stopOf_firstStop`, at any readings and tape). -/
theorem stopOf_rootStop {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopOf rd.round rd.reveal [] (finish (V.signer θ) θ F c σ.1) (fun _ => True)
      (Pos.root (roundCuts V θ F c rd) x).toStop := by
  have h := walk_send_cast (rv := rd.reveal) (f := finish (V.signer θ) θ F c) (a := fun _ => True) hP
    (game_eq_relK V θ F c).symm σ (roundCuts V θ F c rd) x id fun _ => Or.inl ⟨trivial, rfl⟩
  have e : (cast (congrArg Strategy (game_eq_relK V θ F c)) σ).1 = σ.1 := cast_send_fst (game_eq_relK V θ F c) σ
  rw [e] at h
  exact h

/-- **The play's first stop is where `StopsAt` says**, on any tape: from the empty log after `σ`'s request with every
round live, where each round's reading picks only openings (`stopOf_rootStop`). -/
theorem stopsAt_rootStop {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 (Pos.root (roundCuts V θ F c rd) x).toStop :=
  ⟨[], _, fun _ => True, fun _ => rfl, stopOf_rootStop V θ F c rd σ x hP⟩

/-- **The members' steps at a window, at where the walk stops**, at an invariant `I` of the widened coins, the position
`r` the members' traces end at, a member's place and the position its session starts at: `I` holds at the play's
start on the coin map's tape, whatever `r` (`start`); the session's trace of each member with a proof block in the
sample, from a position `I` holds at that keeps the play's start and whose stop is where `StopsAt` says, ending at a
position `q` that `r` is past, where `r` keeps the play's start and its tape's play the sampled verdict accepts,
misses at no round and not at the record and `P` holds of, has `I` at the next place and the position past the stop
after `q`, the session's closing round's, and writes a transcript verity-verify-proof `--zk` accepts at the member's `own`
(`step`); and a member with no proof block in the sample, which has
no statement and no session in the play, has `I` at the next place and the same position (`skip`). It is
`MemberDecodes` with what every member's trace keeps handed to the step. -/
structure MemberSteps (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : WindowDecoders V θ F c W σ)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (I : (W.extend M.widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop) : Prop where
  start : ∀ ω r, I ω r 0 (Pos.root (roundCuts V θ F c rd) (M.start ω))
  step : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Flock.Guarantees.MemberV5 ByteArray W.nJ)
    (p q r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds))
    (out : LiveOutJ (memberTabs ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) m) (memberPts m)),
    ((W.extend M.widening).joint t).ms[i]? = some m → (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
    I ω r i p → p.Inv (roundCuts V θ F c rd) →
    StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 p.toStop →
    MemberReach D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) i m p q out → q.Past r →
    r.Inv (roundCuts V θ F c rd) →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧
      P (CoinTape.run (V.game θ F c) σ r.tape) →
    I ω r (i + 1) q.toStop.consume ∧ FlockSoundness.Discharge.ZkHidden.AcceptsZKP m.I m.own
      (m.wr (m.part ((W.extend M.widening).L.draw ω)) (windowReg V θ F c W rd σ D) out)
  skip : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Flock.Guarantees.MemberV5 ByteArray W.nJ)
    (p r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
    ((W.extend M.widening).joint t).ms[i]? = some m → ¬ (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
    I ω r i p → I ω r (i + 1) p
  ends : ∀ (ω : (W.extend M.widening).L.Ω) (r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
    (Pos.root (roundCuts V θ F c rd) (M.start ω)).Past r →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧
      P (CoinTape.run (V.game θ F c) σ r.tape) →
    ((W.extend M.widening).joint t).ended ω = true

/-- **The decoders' premise at each member's session, at where the walk stops** (`MemberDecodes`): from the members'
steps (`MemberSteps`) at an invariant `I`, it holds at `I` beside the play's start kept and `StopsAt` at the stop a
member's session starts at, wherever each round's reading picks only openings. The play's start and where the walk
stops are kept by every member's trace (`MemberReach.inv`, `MemberReach.stopsAt`), so `MemberReach.logged` applies at
each member. -/
theorem memberDecodes_of_stops (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : WindowDecoders V θ F c W σ)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (I : (W.extend M.widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n)
    (h : MemberSteps V θ F c k P W t rd σ D M BL B I) :
    MemberDecodes V θ F c k P W t rd σ D M BL B fun ω r i p =>
      I ω r i p ∧ p.Inv (roundCuts V θ F c rd) ∧ StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 p.toStop where
  start ω r := ⟨h.start ω r, Pos.inv_root _ _, stopsAt_rootStop V θ F c rd σ (M.start ω) hP⟩
  step ω i m p q r out hm hd hI hmr hqr hri hacc := by
    obtain ⟨hI', hacc'⟩ := h.step ω i m p q r out hm hd hI.1 hI.2.1 hI.2.2 hmr hqr hri hacc
    exact ⟨⟨hI', Pos.inv_consume (Pos.toStop_spec (hmr.inv hI.2.1)).2.2,
      StopsAt.next hP (hmr.stopsAt hP hI.2.2) id⟩, hacc'⟩
  skip ω i m p r hm hd hI := ⟨h.skip ω i m p r hm hd hI.1, hI.2.1, hI.2.2⟩
  ends := h.ends

end Verity.Sampled
