module

public import Proofs.Properties.Verity.Sampled.Window.Walk
public import Proofs.Properties.Verity.Sampled.Stops

/-!
# Sampled proofs' coupling over a window: each member's session's trace in the window's

The joint law's support over a window (`WindowReaches`, `SeqReach`) nests each member's session's trace in what
follows it. Here it is read a member at a time, each session's trace from the position it starts at to the position
it ends at, which the next member's session starts from:

* **A session's rounds' trace to its end** (`RoundsEnd`): the rounds' trace (`RoundsReach`, `RoundsReachK`) with the
  position it ends at and the live session's outcome there. A trace then what follows it is a trace to its end and
  what follows from there (`RoundsReachK.ends`, `RoundsEnd.reachK`); the one-session trace is a trace to its end
  beside the tape there (`RoundsReach.ends`, `RoundsEnd.reach`), so what `Stops` proves of a `RoundsReach` trace holds
  of each member's. The end is a later position that keeps the invariant (`RoundsEnd.le`, `RoundsEnd.inv`).
* **Past a position** (`Pos.Past`, `Pos.toStop_toStop`, `Pos.Past.head`): the position itself or a later one than
  its next stop, which is where a walk on from it goes; a position past a coin node no cut fixes keeps the node's
  coin. A trace's end is past its start (`RoundsEnd.past`), and at each answered stop past the stop with its coins
  written in and consumed (`RoundsEnd.answers`); a trace keeps to its end what each consumed stop passes on and the
  next stop keeps (`RoundsEnd.keeps`).
* **A member's session's trace** (`MemberReach`): link points its held-off event doesn't hold of at the session's
  first stop, and the rounds' trace to its end from the tape with them written in (`MemberReach.le`,
  `MemberReach.inv`, `MemberReach.past`, `MemberReach.keeps`).
* **The members' traces in order** (`SeqChain`): each member's from where the one before it ended, with its
  transcript; the sessions' joint law's support is such a chain, beside the tape at its end (`SeqReach.chain`,
  `SeqChain.le`, `SeqChain.inv`, `SeqChain.past`).
* **A property of every member's transcript, a member at a time** (`SeqChain.forall₂`): from an invariant `I` of the
  position each member's session starts at, kept by each member's trace.
* **The decoders' premise at the window, a member at a time** (`WindowReaches.chain`, `MemberDecodes`,
  `windowDecodes_of_members`): `WindowDecodes` holds wherever each member's session's trace, in a chain whose end's
  tape plays an accepted play that misses nowhere, keeps the invariant and writes a transcript flock-verify `--zk`
  accepts when the member has a proof block in the sample.
-/

@[expose] public section

namespace Verity.Game.Pos

variable {α V ι : Type} {g : Game α} {σ : Strategy g}

/-- **The next stop from a stop is the stop.** -/
theorem toStop_toStop (p : Pos g σ V ι) : p.toStop.toStop = p.toStop := by
  rcases p.toStop_stops with h | ⟨a, ha⟩
  · exact walk_here p.toStop.g' p.toStop.s' p.toStop.Ks p.toStop.x p.toStop.back h
  · have key : ∀ q : Pos g σ V ι, q.g' = .ret a → q.toStop = q := fun ⟨_, _, _, _, _⟩ hg => by
      subst hg
      rfl
    exact key p.toStop ha

/-- **The position `r` is past the position `p`**: `p` itself, or a later position than `p`'s next stop. A walk on
from `p` goes to its next stop first, so each position it passes is past `p`. -/
def Past (p r : Pos g σ V ι) : Prop := r = p ∨ p.toStop.Le r

theorem Past.refl (p : Pos g σ V ι) : p.Past p := Or.inl rfl

/-- **A position's next stop is past it.** -/
theorem past_toStop (p : Pos g σ V ι) : p.Past p.toStop := Or.inr (Le.refl _)

/-- **A position past `p` is a later position than `p`.** -/
theorem Past.le {p r : Pos g σ V ι} (h : p.Past r) : p.Le r := by
  rcases h with h | h
  · rw [h]
    exact Le.refl p
  · exact (le_toStop p).trans h

theorem Past.trans {p q r : Pos g σ V ι} (h₁ : p.Past q) (h₂ : q.Past r) : p.Past r := by
  rcases h₁ with h₁ | h₁
  · rw [h₁] at h₂
    exact h₂
  · exact Or.inr (h₁.trans h₂.le)

/-- **A position past a coin node no cut fixes keeps the node's coin**: its tape is one through the node, holding
there the coin the node's tape holds (`walk_coin_keeps`). -/
theorem Past.head {p r : Pos g σ V ι} (h : p.Past r) {C : Type} {iF : Fintype C} {iN : Nonempty C}
    {k : C → Game α} (hg : p.g' = @Game.coin _ C iF iN k) (hK : ∀ i, (p.Ks i).isHere = false) :
    ∃ z, r.tape = p.back z ∧ headCoin hg p.s' z = headCoin hg p.s' p.x := by
  rcases h with h | h
  · rw [h]
    exact ⟨p.x, rfl, rfl⟩
  · obtain ⟨f, hf, hc⟩ := walk_coin_keeps hg p.s' p.Ks p.x p.back hK
    obtain ⟨f', hf'⟩ := h
    exact ⟨f (f' r.x), (hf' r.x).trans (hf _), hc _⟩

end Verity.Game.Pos

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ addRoundJ)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

/-! ## A session's rounds' trace to its end -/

section Rounds

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ) (B : RoundHold)

/-- **The rounds' trace off `B` to its end** from the position `p`: the position `q` the trace ends at, after the
last round or at a stop whose decoding submits nothing, and the live session's outcome there, with the proof files
decoded from the play on the tape at `q`. -/
inductive RoundsEnd (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ℕ → (Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → Pos g σ Claims.Msg (Fin R) →
      LiveOutJ J mPts → Prop
  | zero (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) :
      RoundsEnd rootB pts 0 rs p p ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.tape)⟩
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (h : D.sub J p.toStop.modes = none) :
      RoundsEnd rootB pts (N + 1) rs p p.toStop ⟨rootB, pts, rs, D.proofs (CoinTape.run g σ p.toStop.tape)⟩
  | round (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (s : SubmitJ J)
      (c : Fin s.n → 𝔽) (q : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ J mPts)
      (h : D.sub J p.toStop.modes = some s) (hB : ¬ B p.toStop.g' J s c)
      (ho : RoundsEnd rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (p.toStop.rewrite (M.round p.toStop J s c)).consume q out) :
      RoundsEnd rootB pts (N + 1) rs p q out

variable {D M dg J B} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
  {rs : Fin J → Fin 2 → List Flock.Round} {p q : Pos g σ Claims.Msg (Fin R)} {out : LiveOutJ J mPts}

/-- **A trace, then what follows the session, is a trace to its end and what follows from there.** -/
theorem RoundsReachK.ends {τ ο : Type} {wr : LiveOutJ J mPts → τ}
    {next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop} {o : ο}
    (hr : RoundsReachK D M dg J B wr next rootB pts N rs p o) :
    ∃ (q : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ J mPts),
      RoundsEnd D M dg J B rootB pts N rs p q out ∧ next q (wr out) o := by
  induction hr with
  | zero rs p o hn => exact ⟨p, _, .zero rs p, hn⟩
  | stop N rs p o h hn => exact ⟨p.toStop, _, .stop N rs p h, hn⟩
  | round N rs p s c o h hB _ ih =>
    obtain ⟨q, out, hq, hn⟩ := ih
    exact ⟨q, out, .round N rs p s c q out h hB hq, hn⟩

/-- **A trace to its end, and what follows from there, is a trace then what follows the session.** -/
theorem RoundsEnd.reachK {τ ο : Type} (wr : LiveOutJ J mPts → τ)
    (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (o : ο)
    (he : RoundsEnd D M dg J B rootB pts N rs p q out) :
    next q (wr out) o → RoundsReachK D M dg J B wr next rootB pts N rs p o := by
  induction he with
  | zero rs p => exact fun hn => .zero rs p o hn
  | stop N rs p h => exact fun hn => .stop N rs p o h hn
  | round N rs p s c q out h hB _ ih => exact fun hn => .round N rs p s c o h hB (ih hn)

/-- **A trace to its end is the one-session trace** (`RoundsReach`) ending on the outcome there, beside the tape at
its end. -/
theorem RoundsEnd.reach {β : Type} (fin : LiveOutJ J mPts → β)
    (he : RoundsEnd D M dg J B rootB pts N rs p q out) :
    RoundsReach D M dg J B fin rootB pts N rs p (fin out, q.tape) := by
  induction he with
  | zero rs p => exact .zero rs p
  | stop N rs p h => exact .stop N rs p h
  | round N rs p s c q out h hB _ ih => exact .round N rs p s c _ h hB ih

/-- **The one-session trace is a trace to its end**, ending on the outcome there beside the tape at its end. -/
theorem RoundsReach.ends {β : Type} {fin : LiveOutJ J mPts → β} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs p o) :
    ∃ (q : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ J mPts),
      RoundsEnd D M dg J B rootB pts N rs p q out ∧ o = (fin out, q.tape) := by
  induction hr with
  | zero rs p => exact ⟨p, _, .zero rs p, rfl⟩
  | stop N rs p h => exact ⟨p.toStop, _, .stop N rs p h, rfl⟩
  | round N rs p s c o h hB _ ih =>
    obtain ⟨q, out, hq, ho⟩ := ih
    exact ⟨q, out, .round N rs p s c q out h hB hq, ho⟩

/-- **A trace's end is a later position than its start.** -/
theorem RoundsEnd.le (he : RoundsEnd D M dg J B rootB pts N rs p q out) : p.Le q := by
  induction he with
  | zero rs p => exact Pos.Le.refl p
  | stop N rs p _ => exact Pos.le_toStop p
  | round N rs p s c q out _ _ _ ih =>
    exact ((Pos.le_toStop p).trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans ih

/-- **A trace's end keeps the invariant its start keeps.** -/
theorem RoundsEnd.inv {Ks₀ : Fin R → DigestCut Claims.Msg g} (he : RoundsEnd D M dg J B rootB pts N rs p q out) :
    p.Inv Ks₀ → q.Inv Ks₀ := by
  induction he with
  | zero rs p => exact id
  | stop N rs p _ => exact fun hp => (Pos.toStop_spec hp).2.2
  | round N rs p s c q out _ _ _ ih =>
    exact fun hp => ih (Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hp).2.2 _))

/-- **A trace's end is past its start.** -/
theorem RoundsEnd.past (he : RoundsEnd D M dg J B rootB pts N rs p q out) : p.Past q := by
  induction he with
  | zero rs p => exact Pos.Past.refl p
  | stop N rs p _ => exact Pos.past_toStop p
  | round N rs p s c q out _ _ _ ih => exact Or.inr (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans ih.le)

/-- **What a trace keeps to its end**: a property `Q` of positions its start has, that each consumed stop passes on
and a position's next stop keeps. -/
theorem RoundsEnd.keeps (he : RoundsEnd D M dg J B rootB pts N rs p q out)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) (hstop : ∀ q, Q q → Q q.toStop) : Q q := by
  induction he with
  | zero rs p => exact hQ
  | stop N rs p _ => exact hstop p hQ
  | round N rs p s c q out _ _ _ ih => exact ih (hstep p _ hQ)

/-- **A trace's answered stops, to its end**: the outcome's rounds are the answered submissions' in order, and at
each a position the trace passes through that keeps any invariant `Q` its start keeps and each consumed stop passes
on, the submission decoded at the stop after it, its coins, which `B` doesn't hold of there, and the trace's end past
the stop with them written in and consumed (`RoundsReach.answers`, with where the trace ends). -/
theorem RoundsEnd.answers (he : RoundsEnd D M dg J B rootB pts N rs p q out)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) :
    ∃ as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))),
      out = ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
          D.proofs (CoinTape.run g σ q.tape)⟩ ∧
      ∀ a ∈ as, Q a.1 ∧ D.sub J a.1.toStop.modes = some a.2.1 ∧ ¬ B a.1.toStop.g' J a.2.1 a.2.2 ∧
        (a.1.toStop.rewrite (M.round a.1.toStop J a.2.1 a.2.2)).consume.Past q := by
  induction he with
  | zero rs p => exact ⟨[], rfl, fun _ h => by simp at h⟩
  | stop N rs p _ => exact ⟨[], rfl, fun _ h => by simp at h⟩
  | round N rs p s c q out hs hB ho ih =>
    obtain ⟨as, h1, h2⟩ := ih (hstep p _ hQ)
    refine ⟨(p, ⟨s, c⟩) :: as, h1, fun a ha => ?_⟩
    rcases List.mem_cons.1 ha with rfl | ha
    · exact ⟨hQ, hs, hB, ho.past⟩
    · exact h2 a ha

end Rounds

/-! ## The members' traces, in order -/

section Seq

open FlockSoundness.Discharge.Exec (Transcript)

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} (D : ℕ → Decoders R Reg)
  (M : WindowCoinMap g σ Ω R) (BL : WindowLinkHold) (B : WindowRoundHold) (S : Finset (Fin nJ)) (Rg : Reg)

/-- **The trace of the session of the member at place `i`** from the position `p` to the position `q` it ends at,
with the live session's outcome `out` there: link points the member's held-off event doesn't hold of at the session's
first stop, and the rounds' trace to its end from the tape with them written in. -/
def MemberReach (i : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (p q : Pos g σ Claims.Msg (Fin R))
    (out : LiveOutJ (memberTabs S Rg m) (memberPts m)) : Prop :=
  ∃ pts : Fin 2 → Fin (memberPts m) → 𝔽, ¬ BL i (memberPts m) p.toStop.g' pts ∧
    RoundsEnd (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) ((D i).root p.toStop.modes) pts m.N
      (fun _ _ => []) (p.toStop.rewrite ((M.member i (memberPts m)).link p.toStop pts)).consume q out

variable {D M BL B S Rg}

/-- **A member's session's trace ends at a later position than it starts from.** -/
theorem MemberReach.le {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out) : p.Le q := by
  obtain ⟨pts, -, he⟩ := h
  exact ((Pos.le_toStop p).trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans he.le

/-- **A member's session's trace ends at a position that keeps the invariant its start keeps.** -/
theorem MemberReach.inv {Ks₀ : Fin R → DigestCut Claims.Msg g} {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ}
    {p q : Pos g σ Claims.Msg (Fin R)} {out : LiveOutJ (memberTabs S Rg m) (memberPts m)}
    (h : MemberReach D M BL B S Rg i m p q out) (hp : p.Inv Ks₀) : q.Inv Ks₀ := by
  obtain ⟨pts, -, he⟩ := h
  exact he.inv (Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hp).2.2 _))

/-- **A member's session's trace ends past the position it starts from.** -/
theorem MemberReach.past {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out) : p.Past q := by
  obtain ⟨pts, -, he⟩ := h
  exact Or.inr (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans he.le)

/-- **What a member's session's trace keeps to its end**: a property `Q` of positions its start has, that each
consumed stop passes on and a position's next stop keeps. -/
theorem MemberReach.keeps {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) (hstop : ∀ q, Q q → Q q.toStop) : Q q := by
  obtain ⟨pts, -, he⟩ := h
  exact he.keeps Q (hstep p _ hQ) hstep hstop

variable (D M BL B S Rg) in
/-- **The members' sessions' traces, in order**, from the position `p` to the position `r` the last ends at: the
trace of the member at place `i`, its transcript, and the rest from the position it ends at. -/
inductive SeqChain : ℕ → List (Flock.Guarantees.MemberV5 Reg nJ) → Pos g σ Claims.Msg (Fin R) →
    List Transcript → Pos g σ Claims.Msg (Fin R) → Prop
  | nil (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : SeqChain i [] p [] p
  | cons (i : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (l : List (Flock.Guarantees.MemberV5 Reg nJ))
      (p q r : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ (memberTabs S Rg m) (memberPts m)) (ts : List Transcript)
      (hm : MemberReach D M BL B S Rg i m p q out) (hl : SeqChain (i + 1) l q ts r) :
      SeqChain i (m :: l) p (m.wr (m.part S) Rg out :: ts) r

/-- **The sessions' joint law's support off the held-off events is the members' traces in order**, beside the tape
at the position the last ends at. -/
theorem SeqReach.chain : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R))
    (o : List Transcript × CoinTape g σ), SeqReach D M BL B S Rg i l p o →
    ∃ r : Pos g σ Claims.Msg (Fin R), SeqChain D M BL B S Rg i l p o.1 r ∧ o.2 = r.tape
  | [], i, p, o, h => by
    have ho : o = ([], p.tape) := h
    rw [ho]
    exact ⟨p, .nil i p, rfl⟩
  | m :: l, i, p, o, h => by
    have hs : SessionReachK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) (m.wr (m.part S) Rg)
        (BL i (memberPts m))
        (fun q a o' => ∃ ts, SeqReach D M BL B S Rg (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts) m.N p o := h
    obtain ⟨pts, hL, hr⟩ := hs
    obtain ⟨q, out, he, ts, hl, ho⟩ := hr.ends
    obtain ⟨r, hc, ht⟩ := SeqReach.chain l (i + 1) q (ts, o.2) hl
    refine ⟨r, ?_, ht⟩
    rw [ho]
    exact .cons i m l p q r out ts ⟨pts, hL, he⟩ hc

/-- **The members' traces end at a later position than they start from.** -/
theorem SeqChain.le {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)} {p r : Pos g σ Claims.Msg (Fin R)}
    {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) : p.Le r := by
  induction hc with
  | nil i p => exact Pos.Le.refl p
  | cons i m l p q r out ts hm _ ih => exact hm.le.trans ih

/-- **The members' traces end past the position they start from.** -/
theorem SeqChain.past {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)} {p r : Pos g σ Claims.Msg (Fin R)}
    {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) : p.Past r := by
  induction hc with
  | nil i p => exact Pos.Past.refl p
  | cons i m l p q r out ts hm _ ih => exact hm.past.trans ih

/-- **The members' traces end at a position that keeps the invariant their start keeps.** -/
theorem SeqChain.inv {Ks₀ : Fin R → DigestCut Claims.Msg g} {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)}
    {p r : Pos g σ Claims.Msg (Fin R)} {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) :
    p.Inv Ks₀ → r.Inv Ks₀ := by
  induction hc with
  | nil i p => exact id
  | cons i m l p q r out ts hm _ ih => exact fun hp => ih (hm.inv hp)

/-- **A property of every member's transcript, a member at a time**: `Acc` holds of each member and its transcript
along the members' traces wherever an invariant `I` of the place and the position a member's session starts at holds
at the chain's start and, at each member's trace from a position it holds at, ending at a position the chain's end
is past, holds at the next place and the position the trace ends at, with `Acc` of the member and the transcript the
trace writes. -/
theorem SeqChain.forall₂ {Acc : Flock.Guarantees.MemberV5 Reg nJ → Transcript → Prop}
    (I : ℕ → Pos g σ Claims.Msg (Fin R) → Prop) {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)}
    {p r : Pos g σ Claims.Msg (Fin R)} {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) :
    I i p →
    (∀ (j k : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (p' q : Pos g σ Claims.Msg (Fin R))
      (out : LiveOutJ (memberTabs S Rg m) (memberPts m)), j = i + k → l[k]? = some m → I j p' →
      MemberReach D M BL B S Rg j m p' q out → q.Past r → I (j + 1) q ∧ Acc m (m.wr (m.part S) Rg out)) →
    List.Forall₂ Acc l ts := by
  induction hc with
  | nil i p => exact fun _ _ => .nil
  | cons i m l p q r out ts hm hl ih =>
    intro hI h
    obtain ⟨hI', hacc⟩ := h i 0 m p q out rfl rfl hI hm hl.past
    exact .cons hacc (ih hI' fun j k m' p' q' out' hj hk =>
      h j (k + 1) m' p' q' out' (by omega) (by simpa using hk))

end Seq

/-! ## The decoders' premise at the window, a member at a time -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

variable {V θ F c} in
/-- **The joint law's support off the held-off events is the members' traces in order** from the play's start on the
coin map's tape at some widened coins, at the decoded registration and those coins' sample, beside the tape at the
position the last ends at. -/
theorem WindowReaches.chain {W : Window} {t : ℕ} {rd : Reading W.rounds} {σ : Strategy (V.game θ F c)}
    {D : ℕ → Decoders W.rounds ByteArray} {M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds} {BL : WindowLinkHold}
    {B : WindowRoundHold} {o : W.Out × CoinTape (V.game θ F c) σ}
    (h : WindowReaches V θ F c W t rd σ D M BL B o) :
    ∃ (ω : (W.extend M.widening).L.Ω) (r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
      o.1.1 = windowReg V θ F c W rd σ D ∧ o.1.2.1 = (W.extend M.widening).L.draw ω ∧
      SeqChain D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) 0
        ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)) o.1.2.2 r ∧
      o.2 = r.tape := by
  obtain ⟨ω, hR, hS, hs⟩ := h
  obtain ⟨r, hc, ht⟩ := SeqReach.chain _ 0 _ (o.1.2.2, o.2) hs
  exact ⟨ω, r, hR, hS, hc, ht⟩

/-- **The decoders' premise at each member's session**, at an invariant `I` of the widened coins, the place and the
position a member's session starts at: `I` holds at the play's start on the coin map's tape (`start`); and each
member's session's trace from a position `I` holds at, ending at a position `q` that a position `r` is past
(`Pos.Past`), where `r` keeps the play's start and its tape's play the sampled verdict accepts, misses at no round and
not at the record and `P` holds of, has `I` at the next place and `q`, and, when the member has a proof block in the
sample, writes a transcript flock-verify `--zk` accepts at the member's `own` (`step`). -/
structure MemberDecodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ℕ → Decoders W.rounds ByteArray)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (I : (W.extend M.widening).L.Ω → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop) : Prop where
  start : ∀ ω, I ω 0 (Pos.root (roundCuts V θ F c rd) (M.start ω))
  step : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Flock.Guarantees.MemberV5 ByteArray W.nJ)
    (p q r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds))
    (out : LiveOutJ (memberTabs ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) m) (memberPts m)),
    ((W.extend M.widening).joint t).ms[i]? = some m → I ω i p →
    MemberReach D M BL B ((W.extend M.widening).L.draw ω) (windowReg V θ F c W rd σ D) i m p q out → q.Past r →
    r.Inv (roundCuts V θ F c rd) →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧
      P (CoinTape.run (V.game θ F c) σ r.tape) →
    I ω (i + 1) q ∧ ((m.part ((W.extend M.widening).L.draw ω)).Nonempty →
      FlockSoundness.Discharge.ZkHidden.AcceptsZKP m.I m.own
        (m.wr (m.part ((W.extend M.widening).L.draw ω)) (windowReg V θ F c W rd σ D) out))

/-- **The decoders' premise at the window, a member at a time** (`WindowDecodes`): it holds wherever the decoders'
premise holds at each member's session (`MemberDecodes`), at any invariant of the position a member's session starts
at. The joint law's support is the members' traces in order, each from where the one before it ended, and the tape an
accepted play is on is the one at the position the last ends at. -/
theorem windowDecodes_of_members (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : Window) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ℕ → Decoders W.rounds ByteArray)
    (M : WindowCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : WindowLinkHold) (B : WindowRoundHold)
    (I : (W.extend M.widening).L.Ω → ℕ → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (h : MemberDecodes V θ F c k P W t rd σ D M BL B I) : WindowDecodes V θ F c k P W t rd σ D M BL B := by
  intro o ho hacc
  obtain ⟨ω, r, -, hS, hc, ht⟩ := ho.chain
  rw [ht] at hacc
  have hri := hc.inv (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))
  have hf : List.Forall₂ (fun (m : Flock.Guarantees.MemberV5 ByteArray W.nJ) ts =>
      (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
        FlockSoundness.Discharge.ZkHidden.AcceptsZKP m.I m.own ts) ((W.extend M.widening).joint t).ms o.1.2.2 :=
    hc.forall₂ (I ω) (h.start ω) fun j k m p q out hj hk hI hmr hqr => by
      have e : k = j := by omega
      subst e
      exact h.step ω _ m p q r out hk hI hmr hqr hri hacc
  have ho1 : o.1 = (o.1.1, (W.extend M.widening).L.draw ω, o.1.2.2) := Prod.ext rfl (Prod.ext hS rfl)
  rw [ho1]
  exact hf

end Verity.Sampled
