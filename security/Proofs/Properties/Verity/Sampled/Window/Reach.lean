module

public import Proofs.Properties.Verity.Sampled.Window.Walk
public import Proofs.Properties.Verity.Sampled.Stops

/-! SPOT's coupling over a window: each member's session's trace in the window's. -/

@[expose] public section

namespace Verity.Game.Pos

variable {α V ι : Type} {g : Game α} {σ : Strategy g}

/-- **The position `r` is past the position `p`**: `p` itself, or a later position than `p`'s next stop. A walk on
from `p` goes to its next stop first, so each position it passes is past `p`. -/
def Past (p r : Pos g σ V ι) : Prop := r = p ∨ p.toStop.Le r

theorem Past.refl (p : Pos g σ V ι) : p.Past p := Or.inl rfl

theorem past_toStop (p : Pos g σ V ι) : p.Past p.toStop := Or.inr (Le.refl _)

theorem Past.le {p r : Pos g σ V ι} (h : p.Past r) : p.Le r := by
  rcases h with h | h
  · rw [h]
    exact Le.refl p
  · exact (le_toStop p).trans h

theorem past_consume (p : Pos g σ V ι) : p.Past p.toStop.consume := Or.inr (le_consume _)

theorem Past.trans {p q r : Pos g σ V ι} (h₁ : p.Past q) (h₂ : q.Past r) : p.Past r := by
  rcases h₁ with h₁ | h₁
  · rw [h₁] at h₂
    exact h₂
  · exact Or.inr (h₁.trans h₂.le)

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

section Rounds

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : PosDecoders g σ R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ) (B : RoundHold)

inductive RoundsEnd (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ℕ → (Fin J → Fin 2 → List Flock.Round) → Pos g σ Claims.Msg (Fin R) → Pos g σ Claims.Msg (Fin R) →
      LiveOutJ J mPts → Prop
  | zero (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) :
      RoundsEnd rootB pts 0 rs p p ⟨rootB, pts, rs, D.files p⟩
  | stop (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R))
      (h : D.sub J p.toStop.modes = none) :
      RoundsEnd rootB pts (N + 1) rs p p.toStop ⟨rootB, pts, rs, D.files p.toStop⟩
  | round (N : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (p : Pos g σ Claims.Msg (Fin R)) (s : SubmitJ J)
      (c : Fin s.n → 𝔽) (q : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ J mPts)
      (h : D.sub J p.toStop.modes = some s) (hB : ¬ B p.toStop.g' J s c)
      (ho : RoundsEnd rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (p.toStop.rewrite (M.round p.toStop J s c)).consume q out) :
      RoundsEnd rootB pts (N + 1) rs p q out

variable {D M dg J B} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
  {rs : Fin J → Fin 2 → List Flock.Round} {p q : Pos g σ Claims.Msg (Fin R)} {out : LiveOutJ J mPts}

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

theorem RoundsEnd.reachK {τ ο : Type} (wr : LiveOutJ J mPts → τ)
    (next : Pos g σ Claims.Msg (Fin R) → τ → ο → Prop) (o : ο)
    (he : RoundsEnd D M dg J B rootB pts N rs p q out) :
    next q (wr out) o → RoundsReachK D M dg J B wr next rootB pts N rs p o := by
  induction he with
  | zero rs p => exact fun hn => .zero rs p o hn
  | stop N rs p h => exact fun hn => .stop N rs p o h hn
  | round N rs p s c q out h hB _ ih => exact fun hn => .round N rs p s c o h hB (ih hn)

theorem RoundsEnd.reach {D : Decoders R Reg} {β : Type} (fin : LiveOutJ J mPts → β)
    (he : RoundsEnd D.atEnd M dg J B rootB pts N rs p q out) :
    RoundsReach D M dg J B fin rootB pts N rs p (fin out, q.tape) := by
  induction he with
  | zero rs p => exact .zero rs p
  | stop N rs p h => exact .stop N rs p h
  | round N rs p s c q out h hB _ ih => exact .round N rs p s c _ h hB ih

theorem RoundsReach.ends {D : Decoders R Reg} {β : Type} {fin : LiveOutJ J mPts → β} {o : β × CoinTape g σ}
    (hr : RoundsReach D M dg J B fin rootB pts N rs p o) :
    ∃ (q : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ J mPts),
      RoundsEnd D.atEnd M dg J B rootB pts N rs p q out ∧ o = (fin out, q.tape) := by
  induction hr with
  | zero rs p => exact ⟨p, _, .zero rs p, rfl⟩
  | stop N rs p h => exact ⟨p.toStop, _, .stop N rs p h, rfl⟩
  | round N rs p s c o h hB _ ih =>
    obtain ⟨q, out, hq, ho⟩ := ih
    exact ⟨q, out, .round N rs p s c q out h hB hq, ho⟩

theorem RoundsEnd.le (he : RoundsEnd D M dg J B rootB pts N rs p q out) : p.Le q := by
  induction he with
  | zero rs p => exact Pos.Le.refl p
  | stop N rs p _ => exact Pos.le_toStop p
  | round N rs p s c q out _ _ _ ih =>
    exact ((Pos.le_toStop p).trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans ih

theorem RoundsEnd.inv {Ks₀ : Fin R → DigestCut Claims.Msg g} (he : RoundsEnd D M dg J B rootB pts N rs p q out) :
    p.Inv Ks₀ → q.Inv Ks₀ := by
  induction he with
  | zero rs p => exact id
  | stop N rs p _ => exact fun hp => (Pos.toStop_spec hp).2.2
  | round N rs p s c q out _ _ _ ih =>
    exact fun hp => ih (Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hp).2.2 _))

theorem RoundsEnd.past (he : RoundsEnd D M dg J B rootB pts N rs p q out) : p.Past q := by
  induction he with
  | zero rs p => exact Pos.Past.refl p
  | stop N rs p _ => exact Pos.past_toStop p
  | round N rs p s c q out _ _ _ ih => exact Or.inr (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans ih.le)

theorem RoundsEnd.keeps (he : RoundsEnd D M dg J B rootB pts N rs p q out)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) (hstop : ∀ q, Q q → Q q.toStop) : Q q := by
  induction he with
  | zero rs p => exact hQ
  | stop N rs p _ => exact hstop p hQ
  | round N rs p s c q out _ _ _ ih => exact ih (hstep p _ hQ)

theorem RoundsEnd.answers (he : RoundsEnd D M dg J B rootB pts N rs p q out)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) :
    ∃ as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽))),
      out = ⟨rootB, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs,
          D.files q⟩ ∧
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

section Seq

open FlockSoundness.Discharge.Exec (Transcript)

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} (D : ℕ → PosDecoders g σ R Reg)
  (M : ExecutionPhaseCoinMap g σ Ω R) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold) (S : Finset (Fin nJ)) (Rg : Reg)

def MemberReach (i : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (p q : Pos g σ Claims.Msg (Fin R))
    (out : LiveOutJ (memberTabs S Rg m) (memberPts m)) : Prop :=
  ∃ pts : Fin 2 → Fin (memberPts m) → 𝔽, ¬ BL i (memberPts m) p.toStop.g' pts ∧
    RoundsEnd (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) ((D i).root p.toStop.modes) pts m.N
      (fun _ _ => []) (p.toStop.rewrite ((M.member i (memberPts m)).link p.toStop pts)).consume q out

variable {D M BL B S Rg}

theorem MemberReach.le {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out) : p.Le q := by
  obtain ⟨pts, -, he⟩ := h
  exact ((Pos.le_toStop p).trans ((Pos.le_rewrite _ _).trans (Pos.le_consume _))).trans he.le

theorem MemberReach.inv {Ks₀ : Fin R → DigestCut Claims.Msg g} {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ}
    {p q : Pos g σ Claims.Msg (Fin R)} {out : LiveOutJ (memberTabs S Rg m) (memberPts m)}
    (h : MemberReach D M BL B S Rg i m p q out) (hp : p.Inv Ks₀) : q.Inv Ks₀ := by
  obtain ⟨pts, -, he⟩ := h
  exact he.inv (Pos.inv_consume (Pos.inv_rewrite (Pos.toStop_spec hp).2.2 _))

theorem MemberReach.past {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out) : p.Past q := by
  obtain ⟨pts, -, he⟩ := h
  exact Or.inr (((Pos.le_rewrite _ _).trans (Pos.le_consume _)).trans he.le)

theorem MemberReach.keeps {i : ℕ} {m : Flock.Guarantees.MemberV5 Reg nJ} {p q : Pos g σ Claims.Msg (Fin R)}
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} (h : MemberReach D M BL B S Rg i m p q out)
    (Q : Pos g σ Claims.Msg (Fin R) → Prop) (hQ : Q p)
    (hstep : ∀ q (φ : CoinTape q.toStop.g' q.toStop.s' → CoinTape q.toStop.g' q.toStop.s'), Q q →
      Q (q.toStop.rewrite φ).consume) (hstop : ∀ q, Q q → Q q.toStop) : Q q := by
  obtain ⟨pts, -, he⟩ := h
  exact he.keeps Q (hstep p _ hQ) hstep hstop

variable (D M BL B S Rg) in
inductive SeqChain : ℕ → List (Flock.Guarantees.MemberV5 Reg nJ) → Pos g σ Claims.Msg (Fin R) →
    List Transcript → Pos g σ Claims.Msg (Fin R) → Prop
  | nil (i : ℕ) (p : Pos g σ Claims.Msg (Fin R)) : SeqChain i [] p [] p
  | cons (i : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (l : List (Flock.Guarantees.MemberV5 Reg nJ))
      (p q r : Pos g σ Claims.Msg (Fin R)) (out : LiveOutJ (memberTabs S Rg m) (memberPts m)) (ts : List Transcript)
      (hd : memberSampled S m = true) (hm : MemberReach D M BL B S Rg i m p q out)
      (hl : SeqChain (i + 1) l q.toStop.consume ts r) :
      SeqChain i (m :: l) p (m.wr (m.part S) Rg out :: ts) r
  | skip (i : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (l : List (Flock.Guarantees.MemberV5 Reg nJ))
      (p r : Pos g σ Claims.Msg (Fin R)) (pts : Fin 2 → Fin (memberPts m) → 𝔽) (ts : List Transcript)
      (hd : memberSampled S m = false) (hl : SeqChain (i + 1) l p ts r) :
      SeqChain i (m :: l) p (m.wr (m.part S) Rg (idleOut (memberTabs S Rg m) pts) :: ts) r

theorem SeqReach.chain : ∀ (l : List (Flock.Guarantees.MemberV5 Reg nJ)) (i : ℕ) (p : Pos g σ Claims.Msg (Fin R))
    (o : List Transcript × CoinTape g σ), SeqReach D M BL B S Rg i l p o →
    ∃ r : Pos g σ Claims.Msg (Fin R), SeqChain D M BL B S Rg i l p o.1 r ∧ o.2 = r.tape
  | [], i, p, o, h => by
    have ho : o = ([], p.tape) := h
    rw [ho]
    exact ⟨p, .nil i p, rfl⟩
  | m :: l, i, p, o, h => by
    have hs : MemberReachK (D i) (M.member i (memberPts m)) m.dg (memberTabs S Rg m) (B i) (m.wr (m.part S) Rg)
        (BL i (memberPts m))
        (fun q a o' => ∃ ts, SeqReach D M BL B S Rg (i + 1) l q (ts, o'.2) ∧ o'.1 = a :: ts) m.N p o
        (memberSampled S m) := h
    cases hd : memberSampled S m with
    | true =>
      rw [hd] at hs
      obtain ⟨pts, hL, hr⟩ := hs
      obtain ⟨q, out, he, ts, hl, ho⟩ := hr.ends
      obtain ⟨r, hc, ht⟩ := SeqReach.chain l (i + 1) q.toStop.consume (ts, o.2) hl
      refine ⟨r, ?_, ht⟩
      rw [ho]
      exact .cons i m l p q r out ts hd ⟨pts, hL, he⟩ hc
    | false =>
      rw [hd] at hs
      obtain ⟨pts, ts, hl, ho⟩ := hs
      obtain ⟨r, hc, ht⟩ := SeqReach.chain l (i + 1) p (ts, o.2) hl
      refine ⟨r, ?_, ht⟩
      rw [ho]
      exact .skip i m l p r pts ts hd hc

theorem SeqChain.le {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)} {p r : Pos g σ Claims.Msg (Fin R)}
    {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) : p.Le r := by
  induction hc with
  | nil i p => exact Pos.Le.refl p
  | cons i m l p q r out ts _ hm _ ih =>
    exact hm.le.trans (((Pos.le_toStop q).trans (Pos.le_consume _)).trans ih)
  | skip i m l p r pts ts _ _ ih => exact ih

theorem SeqChain.past {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)} {p r : Pos g σ Claims.Msg (Fin R)}
    {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) : p.Past r := by
  induction hc with
  | nil i p => exact Pos.Past.refl p
  | cons i m l p q r out ts _ hm _ ih =>
    exact (hm.past.trans (Pos.past_consume q)).trans ih
  | skip i m l p r pts ts _ _ ih => exact ih

theorem SeqChain.inv {Ks₀ : Fin R → DigestCut Claims.Msg g} {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)}
    {p r : Pos g σ Claims.Msg (Fin R)} {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) :
    p.Inv Ks₀ → r.Inv Ks₀ := by
  induction hc with
  | nil i p => exact id
  | cons i m l p q r out ts _ hm _ ih =>
    exact fun hp => ih (Pos.inv_consume (Pos.toStop_spec (hm.inv hp)).2.2)
  | skip i m l p r pts ts _ _ ih => exact ih

theorem SeqChain.forall₂ {Acc : Flock.Guarantees.MemberV5 Reg nJ → Transcript → Prop}
    (I : ℕ → Pos g σ Claims.Msg (Fin R) → Prop) {i : ℕ} {l : List (Flock.Guarantees.MemberV5 Reg nJ)}
    {p r : Pos g σ Claims.Msg (Fin R)} {ts : List Transcript} (hc : SeqChain D M BL B S Rg i l p ts r) :
    I i p →
    (∀ (j k : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (p' q : Pos g σ Claims.Msg (Fin R))
      (out : LiveOutJ (memberTabs S Rg m) (memberPts m)), j = i + k → l[k]? = some m → memberSampled S m = true →
      I j p' → MemberReach D M BL B S Rg j m p' q out → q.Past r →
      I (j + 1) q.toStop.consume ∧ Acc m (m.wr (m.part S) Rg out)) →
    (∀ (j k : ℕ) (m : Flock.Guarantees.MemberV5 Reg nJ) (p' : Pos g σ Claims.Msg (Fin R))
      (pts : Fin 2 → Fin (memberPts m) → 𝔽), j = i + k → l[k]? = some m → memberSampled S m = false → I j p' →
      I (j + 1) p' ∧ Acc m (m.wr (m.part S) Rg (idleOut (memberTabs S Rg m) pts))) →
    List.Forall₂ Acc l ts := by
  induction hc with
  | nil i p => exact fun _ _ _ => .nil
  | cons i m l p q r out ts hd hm hl ih =>
    intro hI h hs
    obtain ⟨hI', hacc⟩ := h i 0 m p q out rfl rfl hd hI hm ((Pos.past_consume q).trans hl.past)
    exact .cons hacc (ih hI'
      (fun j k m' p' q' out' hj hk => h j (k + 1) m' p' q' out' (by omega) (by simpa using hk))
      fun j k m' p' pts' hj hk => hs j (k + 1) m' p' pts' (by omega) (by simpa using hk))
  | skip i m l p r pts ts hd hl ih =>
    intro hI h hs
    obtain ⟨hI', hacc⟩ := hs i 0 m p pts rfl rfl hd hI
    exact .cons hacc (ih hI'
      (fun j k m' p' q' out' hj hk => h j (k + 1) m' p' q' out' (by omega) (by simpa using hk))
      fun j k m' p' pts' hj hk => hs j (k + 1) m' p' pts' (by omega) (by simpa using hk))

end Seq

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

variable {V θ F c} in
theorem ExecutionPhaseReaches.chain {W : ExecutionPhase} {t : ℕ} {rd : Reading W.rounds} {σ : Strategy (V.game θ F c)}
    {D : ExecutionPhaseDecoders V θ F c W σ} {M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds} {BL : ExecutionPhaseLinkHold}
    {B : ExecutionPhaseRoundHold} {o : W.Out × CoinTape (V.game θ F c) σ}
    (h : ExecutionPhaseReaches V θ F c W t rd σ D M BL B o) :
    ∃ (ω : (W.extend M.widening).L.Ω) (r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
      o.1.1 = (executionPhaseReg V θ F c W rd σ D, ((W.extend M.widening).joint t).ended ω) ∧
      o.1.2.1 = (W.extend M.widening).L.draw ω ∧
      SeqChain D M BL B ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) 0
        ((W.extend M.widening).joint t).ms (Pos.root (roundCuts V θ F c rd) (M.start ω)) o.1.2.2 r ∧
      o.2 = r.tape := by
  obtain ⟨ω, hR, hS, hs⟩ := h
  obtain ⟨r, hc, ht⟩ := SeqReach.chain _ 0 _ (o.1.2.2, o.2) hs
  exact ⟨ω, r, hR, hS, hc, ht⟩

structure MemberDecodes (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ExecutionPhaseDecoders V θ F c W σ)
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold)
    (I : (W.extend M.widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop) : Prop where
  start : ∀ ω r, I ω r 0 (Pos.root (roundCuts V θ F c rd) (M.start ω))
  step : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Flock.Guarantees.MemberV5 ByteArray W.nJ)
    (p q r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds))
    (out : LiveOutJ (memberTabs ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) m) (memberPts m)),
    ((W.extend M.widening).joint t).ms[i]? = some m → (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
    I ω r i p →
    MemberReach D M BL B ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) i m p q out → q.Past r →
    r.Inv (roundCuts V θ F c rd) →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧
      P (CoinTape.run (V.game θ F c) σ r.tape) →
    I ω r (i + 1) q.toStop.consume ∧ FlockSoundness.Discharge.ZkHidden.AcceptsZKP m.I m.own
      (m.wr (m.part ((W.extend M.widening).L.draw ω)) (executionPhaseReg V θ F c W rd σ D) out)
  skip : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Flock.Guarantees.MemberV5 ByteArray W.nJ)
    (p r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
    ((W.extend M.widening).joint t).ms[i]? = some m → ¬ (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
    I ω r i p → I ω r (i + 1) p
  ends : ∀ (ω : (W.extend M.widening).L.Ω) (r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds)),
    (Pos.root (roundCuts V θ F c rd) (M.start ω)).Past r →
    V.Verdict k .sampled (CoinTape.run (V.game θ F c) σ r.tape) ∧ NoMiss V θ F c rd σ r.tape ∧
      P (CoinTape.run (V.game θ F c) σ r.tape) →
    ((W.extend M.widening).joint t).ended ω = true

theorem executionPhaseDecodes_of_members (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ExecutionPhaseDecoders V θ F c W σ)
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold)
    (I : (W.extend M.widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop)
    (h : MemberDecodes V θ F c k P W t rd σ D M BL B I) : ExecutionPhaseDecodes V θ F c k P W t rd σ D M BL B := by
  intro o ho hacc
  obtain ⟨ω, r, hR, hS, hc, ht⟩ := ho.chain
  rw [ht] at hacc
  have hE := h.ends ω r hc.past hacc
  have hri := hc.inv (Pos.inv_root (roundCuts V θ F c rd) (M.start ω))
  have hf : List.Forall₂ (fun (m : Flock.Guarantees.MemberV5 ByteArray W.nJ) ts =>
      (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
        FlockSoundness.Discharge.ZkHidden.AcceptsZKP m.I m.own ts) ((W.extend M.widening).joint t).ms o.1.2.2 :=
    hc.forall₂ (I ω r) (h.start ω r)
      (fun j k m p q out hj hk hd hI hmr hqr => by
        have e : k = j := by omega
        subst e
        obtain ⟨hI', ha⟩ := h.step ω _ m p q r out hk (memberSampled_iff.1 hd) hI hmr hqr hri hacc
        exact ⟨hI', fun _ => ha⟩)
      fun j k m p _ hj hk hd hI => by
        have e : k = j := by omega
        subst e
        exact ⟨h.skip ω _ m p r hk (not_nonempty_of_memberSampled hd) hI,
          fun hne => absurd hne (not_nonempty_of_memberSampled hd)⟩
  have ho1 : o.1 = ((executionPhaseReg V θ F c W rd σ D, ((W.extend M.widening).joint t).ended ω),
      (W.extend M.widening).L.draw ω, o.1.2.2) := Prod.ext hR (Prod.ext hS rfl)
  rw [ho1]
  exact And.intro hE hf

end Verity.Sampled
