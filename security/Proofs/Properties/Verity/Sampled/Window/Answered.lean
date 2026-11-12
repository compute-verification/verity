module

public import Proofs.Properties.Verity.Sampled.Window.Logged

/-!
# Sampled proofs' coupling over a window: the members' answered stops, in order in the play's log

The members' sessions' traces run one after another, each from where the one before it ended, and the play the
sampled verdict accepts is the one on the tape of the position the last ends at. This module says where in that one
play's log each member's trace answered, and what the walk has answered by the stop the next member's starts at.

* **What the traces have answered by a stop** (`Walked`): in the log `Lg`, with the releases after the logs `done`
  answered, after the log `pre` at the live rounds `a`. Each answered log is no longer than `pre` and some round picks
  after it; each round no longer live picks after an answered log; each pick before `pre` is after an answered log;
  and no live round picks after one. It holds at the play's start with nothing answered and every round live
  (`walked_nil`), where `AnsweredBy` takes a first answered log.
* **Answering a stop** (`Walked.next`): where `Lg` has, after `pre` and releases no live round picks, a release a
  live round picks, the log before that release is longer than every answered log, and `Walked` holds there with that
  log answered and the rounds that pick the release no longer live.
* **The rounds' trace to its end, in order in the log** (`RoundsEnd.order`): `RoundsReach.order` on a trace to its
  end, at the log of the play on the tape of any position past the end: each answered stop releases its round's coins
  after a log longer than every answered one and than the one before, the walk from the trace's end stops where
  `StopOf` says with `Walked` at those logs answered too, and the trace answered its `N` rounds or ended at a stop the
  decoders submit nothing at.
* **A member's session's trace, in order in the log** (`MemberReach.order`): the same from the session's first stop,
  where not every cut is past: the link points' bits are released after a log longer than every answered one, each
  answered round's coins after a longer one, and the walk from the session's end stops where `StopOf` says with
  `Walked` at all of those answered.
-/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ addRoundJ)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

/-! ## What the traces have answered by a stop -/

section Walked

variable {R : ℕ}

/-- **What the members' traces have answered by a stop**, in the log `Lg`, with the releases after the logs `done`
answered, after the log `pre` at the live rounds `a`: each answered log is no longer than `pre` and some round picks
after it; each round not live in `a` picks after an answered log; each round picking before `pre` picks after an
answered log; and no live round picks after one. -/
structure Walked (P : Fin R → List Release → Frame → Prop) (Lg pre : List Release) (a : Fin R → Prop)
    (done : List (List Release)) : Prop where
  picked : ∀ L₁ ∈ done, L₁.length ≤ pre.length ∧ ∃ j, PicksIn P Lg j L₁
  past : ∀ j, ¬ a j → ∃ L₁ ∈ done, PicksIn P Lg j L₁
  before : ∀ j L₁, PicksIn P Lg j L₁ → L₁.length < pre.length → L₁ ∈ done
  live : ∀ j L₁, L₁ ∈ done → PicksIn P Lg j L₁ → ¬ a j

/-- **Nothing is answered at the play's start**, from the empty log with every round live. -/
theorem walked_nil (P : Fin R → List Release → Frame → Prop) (Lg : List Release) :
    Walked P Lg [] (fun _ => True) [] where
  picked _ h := absurd h List.not_mem_nil
  past _ h := absurd trivial h
  before _ _ _ h := absurd h (Nat.not_lt_zero _)
  live _ _ h := absurd h List.not_mem_nil

/-- **A pick before the end of a segment no live round picks in is after an answered log**: where each round picks at
most one release of `Lg`, `Lg` is `pre`, the segment `L` and the rest, and no live round picks in `L` after `pre`
(`AnsweredBy.seg`). -/
theorem Walked.seg {P : Fin R → List Release → Frame → Prop} {Lg pre : List Release} {a : Fin R → Prop}
    {done : List (List Release)} (hA : Walked P Lg pre a done) (hU : PicksOnce P Lg) {L L₃ : List Release}
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

/-- **Answering a stop**: where each round picks at most one release of `Lg`, and `Lg` has, after `pre` and releases
`L` no live round picks, a release `r` that the live round `j` picks, the log `pre ++ L` before `r` is longer than
every answered log, and `Walked` holds after it with it answered and the rounds that pick `r` there no longer live. -/
theorem Walked.next {P : Fin R → List Release → Frame → Prop} {Lg pre : List Release} {a : Fin R → Prop}
    {done : List (List Release)} (hA : Walked P Lg pre a done) (hU : PicksOnce P Lg) {L L' : List Release}
    {r : Release} (hNP : NoPick P a pre L) (hLg : Lg = (pre ++ L) ++ r :: L') {j : Fin R} (haj : a j)
    (hj : P j (pre ++ L) r.frame) :
    (∀ L₁ ∈ done, L₁.length < (pre ++ L).length) ∧
      Walked P Lg (pre ++ L) (fun j => a j ∧ ¬ P j (pre ++ L) r.frame) ((pre ++ L) :: done) := by
  have hpick : ∀ j', PicksIn P Lg j' (pre ++ L) → P j' (pre ++ L) r.frame := by
    rintro j' ⟨r', L₂, he, hp⟩
    rw [hLg] at he
    obtain ⟨rfl, -⟩ := List.cons.inj (List.append_cancel_left he)
    exact hp
  have hseg : Lg = pre ++ (L ++ r :: L') := by rw [hLg, List.append_assoc]
  have hlt : ∀ L₁ ∈ done, L₁.length < (pre ++ L).length := fun L₁ hd => by
    obtain ⟨hle, j', r', L₂, he, -⟩ := hA.picked L₁ hd
    rw [List.length_append]
    by_cases hL : L.length = 0
    · have hnil : L = [] := List.eq_nil_of_length_eq_zero hL
      have hne : L₁.length ≠ pre.length := fun hl => by
        rw [hnil, List.append_nil] at hLg hj
        obtain ⟨e, -⟩ := List.append_inj (he.symm.trans hLg) hl
        rw [e] at hd
        exact hA.live j pre hd ⟨r, L', hLg, hj⟩ haj
      omega
    · omega
  refine ⟨hlt, fun L₁ hd => ?_, fun j' hj' => ?_, fun j' L₁ hp hl => ?_, fun j' L₁ hd hp => ?_⟩
  · rcases List.mem_cons.1 hd with rfl | hd
    · exact ⟨Nat.le_refl _, j, r, L', hLg, hj⟩
    · exact ⟨Nat.le_of_lt (hlt L₁ hd), (hA.picked L₁ hd).2⟩
  · by_cases ha : a j'
    · exact ⟨pre ++ L, List.mem_cons_self .., r, L', hLg, not_not.1 fun h => hj' ⟨ha, h⟩⟩
    · obtain ⟨L₀, hL₀, hp₀⟩ := hA.past j' ha
      exact ⟨L₀, List.mem_cons_of_mem _ hL₀, hp₀⟩
  · rw [List.length_append] at hl
    exact List.mem_cons_of_mem _ (hA.seg hU hNP hseg hp hl)
  · rcases List.mem_cons.1 hd with rfl | hd
    · exact fun h => h.2 (hpick j' hp)
    · exact fun h => hA.live j' L₁ hd hp h.1

end Walked

/-! ## The rounds' trace to its end, in order in the log -/

section Rounds

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} {D : PosDecoders g σ R Reg}
  {M : CoinMap g σ Ω R mPts} {dg : ByteArray → ByteArray} {J : ℕ} {B : RoundHold}

/-- **The rounds' trace to its end, in order in the log** (`RoundsReach.order`, on a trace to its end): on a trace
from a position that keeps the invariant at the cuts `Ks₀` and whose walk stops where `StopOf` says after the log `pre`
at the live rounds `a`, with the releases after the logs `done` answered (`Walked`) in the log `Lg` of the play on the
tape of a position `r` past the trace's end, where each round picks at most one release of `Lg` and only openings the
device serves, the decoders submit nothing at a stop that fixes no cut, the coin map writes a round's coins into the
stop's opening as their bits `bitsOf`, and off `B` the opening's class at those bits is not empty: the outcome's rounds
are the answered submissions' in order; each answered stop releases its round's coins after a log (`LoggedAfter`)
longer than every answered one and than the one before, the cuts the stop fixes at the bytes that release answers
(`CutsAt`), at a position that keeps the invariant and whose stop, the coins written in and consumed, the trace's end
is past; the walk from the trace's end stops where `StopOf` says, with those logs answered too (`Walked`); and the
trace answered `N` rounds or ended at a stop the decoders submit nothing at. -/
theorem RoundsEnd.order {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
    {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray}
    (hP : ∀ j pre fr, P j pre fr → ∃ i s n, fr = .opening i s n)
    (hPS : ∀ j pre i s n, P j pre (.opening i s n) → Served pre i s n) (hsub : D.sub J (fun _ => none) = none)
    (bitsOf : (s : SubmitJ J) → (Fin s.n → 𝔽) → List Bool)
    (hM : ∀ q s x, M.round q J s x = writeOpening (finish sg θ F c rq) q.s' (bitsOf s x))
    (hB : ∀ g' s x, ¬ B g' J s x → ¬ EmptyAt (finish sg θ F c rq) g' (bitsOf s x))
    {Ks₀ : Fin R → DigestCut Claims.Msg g} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {N : ℕ}
    {rs : Fin J → Fin 2 → List Flock.Round} {p q r : Pos g σ Claims.Msg (Fin R)} {out : LiveOutJ J mPts}
    {Lg : List Release} (hU : PicksOnce P Lg) (hLg : (CoinTape.run g σ r.tape).log = Lg)
    (he : RoundsEnd D M dg J B rootB pts N rs p q out) :
    p.Inv Ks₀ → q.Past r →
    ∀ (pre : List Release) (f : List Release → Game Exchange) (a : Fin R → Prop) (done : List (List Release)),
      (∀ log, f log = finish sg θ F c rq (pre ++ log)) → StopOf P rv pre f a p.toStop → Walked P Lg pre a done →
      ∃ (as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ J) × (Fin s.n → 𝔽)))) (ms : List (List Release))
        (pre' : List Release) (f' : List Release → Game Exchange) (a' : Fin R → Prop)
        (done' : List (List Release)),
        out = ⟨rootB, pts, as.foldl (fun rs a =>
          addRoundJ rs a.2.1.tab a.2.1.rep (roundOf dg a.2.1.bytes a.2.1.n a.2.2)) rs, D.files q⟩ ∧
        List.Forall₂ (fun a m => D.sub J a.1.toStop.modes = some a.2.1 ∧
          LoggedAfter P a.1.toStop m (bitsOf a.2.1 a.2.2) Lg ∧ CutsAt rv a.1.toStop m Lg) as ms ∧
        (∀ a ∈ as, a.1.Inv Ks₀ ∧ (a.1.toStop.rewrite (M.round a.1.toStop J a.2.1 a.2.2)).consume.Past q) ∧
        (∀ m ∈ ms, ∀ L₁ ∈ done, L₁.length < m.length) ∧ ms.Pairwise (fun m m' => m.length < m'.length) ∧
        (∀ log, f' log = finish sg θ F c rq (pre' ++ log)) ∧ StopOf P rv pre' f' a' q.toStop ∧
        (∀ L₁, L₁ ∈ done' ↔ L₁ ∈ ms ∨ L₁ ∈ done) ∧ Walked P Lg pre' a' done' ∧
        (as.length = N ∨ D.sub J q.toStop.modes = none) := by
  induction he with
  | zero rs p =>
    intro _ _ pre f a done hf hSO hA
    exact ⟨[], [], pre, f, a, done, rfl, .nil, fun _ h => by simp at h, fun _ h => by simp at h, .nil, hf, hSO,
      fun L₁ => by simp, hA, Or.inl rfl⟩
  | stop N rs p hs =>
    intro _ _ pre f a done hf hSO hA
    refine ⟨[], [], pre, f, a, done, rfl, .nil, fun _ h => by simp at h, fun _ h => by simp at h, .nil, hf, ?_,
      fun L₁ => by simp, hA, Or.inr ?_⟩
    · rw [Pos.toStop_toStop]
      exact hSO
    · rw [Pos.toStop_toStop]
      exact hs
  | round N rs p s cs q out hs hBc ho ih =>
    intro hi hqr pre f a done hf hSO hA
    have hi' := (Pos.toStop_spec hi).2.2
    obtain ⟨L, hNP, ⟨hK, -⟩ |
      ⟨k, fs, ts, f', i, s₁, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩⟩ := hSO
    · have hm : p.toStop.modes = fun _ => none := funext fun j =>
        (congrArg (fun K => DigestCut.modeHere K p.toStop.s') (hK j)).trans rfl
      rw [hm, hsub] at hs
      cases hs
    · have hf'' : ∀ log, f' log = finish sg θ F c rq ((pre ++ L) ++ log) := fun log => by
        rw [hf', hf, List.append_assoc]
      have ho' := ho.past
      rw [hM] at ho'
      obtain ⟨L', hL'⟩ := release_past hq hi' hf'' hT (hPS j _ i s₁ n hj) (hB _ _ _ hBc) (ho'.trans hqr)
      rw [hLg] at hL'
      have hjK : (p.toStop.Ks j).isHere = true := by
        rw [hq]
        exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
      have hall : ∀ j', (p.toStop.Ks j').isHere = true → P j' (pre ++ L) (.opening i s₁ n) := fun j' h' => by
        rw [hq] at h'
        exact ((isHere_openCuts hKs j').1 h').2
      obtain ⟨hlt, hA'⟩ := hA.next hU hNP (r := ⟨.opening i s₁ n, b, bitsOf s cs⟩) hL' haj hj
      obtain ⟨as, ms, pre', f₁, a', done', h1, hF, hI, hgt, hpw, hf₁, hSO', hd', hW, hend⟩ :=
        ih (Pos.inv_consume (Pos.inv_rewrite hi' _)) hqr (pre ++ L) f' _ ((pre ++ L) :: done) hf''
          (stopOf_next hP s'' Ks x'' back'' hq hT hKs _) hA'
      refine ⟨(p, ⟨s, cs⟩) :: as, (pre ++ L) :: ms, pre', f₁, a', done', h1,
        .cons ⟨hs, ⟨j, i, s₁, n, b, L', hjK, hall, hj, hL'⟩, cutsAt_openCuts hq hKs hL'⟩ hF,
        fun x hx => ?_, fun m hm L₁ hd => ?_,
        List.pairwise_cons.2 ⟨fun m hm => hgt m hm _ (List.mem_cons_self ..), hpw⟩, hf₁, hSO', fun L₁ => ?_, hW,
        hend.imp_left fun h => by rw [List.length_cons, h]⟩
      · rcases List.mem_cons.1 hx with rfl | hx
        · exact ⟨hi, ho.past⟩
        · exact hI x hx
      · rcases List.mem_cons.1 hm with rfl | hm
        · exact hlt L₁ hd
        · exact hgt m hm L₁ (List.mem_cons_of_mem _ hd)
      · rw [hd' L₁, List.mem_cons, List.mem_cons]
        constructor
        · rintro (h | h | h)
          · exact Or.inl (Or.inr h)
          · exact Or.inl (Or.inl h)
          · exact Or.inr h
        · rintro ((h | h) | h)
          · exact Or.inr (Or.inl h)
          · exact Or.inl h
          · exact Or.inr (Or.inr h)

end Rounds

/-! ## A member's session's trace, in order in the log -/

section Member

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} {D : ℕ → PosDecoders g σ R Reg}
  {M : WindowCoinMap g σ Ω R} {BL : WindowLinkHold} {B : WindowRoundHold} {S : Finset (Fin nJ)} {Rg : Reg}

/-- **A member's session's trace, in order in the log**: on the trace of the session of the member at place `i` from a
position that keeps the invariant at the cuts `Ks₀`, whose walk stops where `StopOf` says after the log `pre` at the
live rounds `a` and where not every cut is past, with the releases after the logs `done` answered (`Walked`) in the
log `Lg` of the play on the tape of a position `r` past the trace's end, where each round picks at most one release of
`Lg` and only openings the device serves, the member's decoders submit nothing at a stop that fixes no cut, the
member's coin map writes the link points and a round's coins into a stop's opening as their bits (`bitsL`, `bitsOf`),
and off the member's held-off events the opening's class at those bits is not empty: the session's outcome is the root
decoded at its first stop, link points `pts`, the answered submissions' rounds in order and the member's proof files
at its end; `Lg` has `pts`' bits after a log `L₀` longer than every answered one, at an opening the rounds whose cuts
the first stop fixes pick, those cuts at the bytes it answers; each answered stop releases its round's coins after a
log longer than `L₀` and than the one before (`RoundsEnd.order`); the walk from the session's end stops where `StopOf`
says, with `L₀` and those logs answered too (`Walked`); and the trace answered the member's `N` rounds or ended at a
stop the decoders submit nothing at. -/
theorem MemberReach.order {P : Fin R → List Release → Frame → Prop} {rv : Fin R → Exchange → Option Claims.Msg}
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
    {out : LiveOutJ (memberTabs S Rg m) (memberPts m)} {Lg : List Release} (hU : PicksOnce P Lg)
    (hLg : (CoinTape.run g σ r.tape).log = Lg) (h : MemberReach D M BL B S Rg i m p q out) (hqr : q.Past r)
    (hi : p.Inv Ks₀) {pre : List Release} {f : List Release → Game Exchange} {a : Fin R → Prop}
    {done : List (List Release)} (hf : ∀ log, f log = finish sg θ F c rq (pre ++ log))
    (hSO : StopOf P rv pre f a p.toStop) (hA : Walked P Lg pre a done) (hK : ¬ ∀ j, p.toStop.Ks j = .skip) :
    ∃ (pts : Fin 2 → Fin (memberPts m) → 𝔽) (L₀ : List Release)
      (as : List (Pos g σ Claims.Msg (Fin R) × ((s : SubmitJ (memberTabs S Rg m)) × (Fin s.n → 𝔽))))
      (ms : List (List Release)) (pre' : List Release) (f' : List Release → Game Exchange) (a' : Fin R → Prop)
      (done' : List (List Release)),
      out = ⟨(D i).root p.toStop.modes, pts, as.foldl (fun rs a =>
        addRoundJ rs a.2.1.tab a.2.1.rep (roundOf m.dg a.2.1.bytes a.2.1.n a.2.2)) (fun _ _ => []),
          (D i).files q⟩ ∧
      (LoggedAfter P p.toStop L₀ (bitsL pts) Lg ∧ CutsAt rv p.toStop L₀ Lg) ∧
      List.Forall₂ (fun a m' => (D i).sub (memberTabs S Rg m) a.1.toStop.modes = some a.2.1 ∧
        LoggedAfter P a.1.toStop m' (bitsOf a.2.1 a.2.2) Lg ∧ CutsAt rv a.1.toStop m' Lg) as ms ∧
      (∀ a ∈ as, a.1.Inv Ks₀ ∧ (a.1.toStop.rewrite
        ((M.member i (memberPts m)).round a.1.toStop (memberTabs S Rg m) a.2.1 a.2.2)).consume.Past q) ∧
      (∀ L₁ ∈ done, L₁.length < L₀.length) ∧ (∀ m' ∈ ms, L₀.length < m'.length) ∧
      ms.Pairwise (fun m' m'' => m'.length < m''.length) ∧
      (∀ log, f' log = finish sg θ F c rq (pre' ++ log)) ∧ StopOf P rv pre' f' a' q.toStop ∧
      (∀ L₁, L₁ ∈ done' ↔ L₁ = L₀ ∨ L₁ ∈ ms ∨ L₁ ∈ done) ∧ Walked P Lg pre' a' done' ∧
      (as.length = m.N ∨ (D i).sub (memberTabs S Rg m) q.toStop.modes = none) := by
  obtain ⟨pts, hL, he⟩ := h
  have hi' := (Pos.toStop_spec hi).2.2
  obtain ⟨L, hNP, ⟨hK', -⟩ |
    ⟨k, fs, ts, f', i', s, n, b, hf', hT, ⟨j, haj, hj⟩, s'', Ks, x'', back'', hq, hKs⟩⟩ := hSO
  · exact absurd hK' hK
  have hf'' : ∀ log, f' log = finish sg θ F c rq ((pre ++ L) ++ log) := fun log => by
    rw [hf', hf, List.append_assoc]
  have hp' := he.past
  rw [hML] at hp'
  obtain ⟨L', hL'⟩ := release_past hq hi' hf'' hT (hPS j _ i' s n hj) (hBL _ _ hL) (hp'.trans hqr)
  rw [hLg] at hL'
  have hjK : (p.toStop.Ks j).isHere = true := by
    rw [hq]
    exact (isHere_openCuts hKs j).2 ⟨haj, hj⟩
  have hall : ∀ j', (p.toStop.Ks j').isHere = true → P j' (pre ++ L) (.opening i' s n) := fun j' h' => by
    rw [hq] at h'
    exact ((isHere_openCuts hKs j').1 h').2
  obtain ⟨hlt, hA'⟩ := hA.next hU hNP (r := ⟨.opening i' s n, b, bitsL pts⟩) hL' haj hj
  obtain ⟨as, ms, pre', f₁, a', done', h1, hF, hI, hgt, hpw, hf₁, hSO', hd', hW, hend⟩ :=
    RoundsEnd.order hP hPS hsub bitsOf hM hB hU hLg he (Pos.inv_consume (Pos.inv_rewrite hi' _)) hqr (pre ++ L) f'
      _ ((pre ++ L) :: done) hf'' (stopOf_next hP s'' Ks x'' back'' hq hT hKs _) hA'
  refine ⟨pts, pre ++ L, as, ms, pre', f₁, a', done', h1,
    ⟨⟨j, i', s, n, b, L', hjK, hall, hj, hL'⟩, cutsAt_openCuts hq hKs hL'⟩, hF, hI, hlt,
    fun m' hm' => hgt m' hm' _ (List.mem_cons_self ..), hpw, hf₁, hSO', fun L₁ => ?_, hW, hend⟩
  rw [hd' L₁, List.mem_cons]
  constructor
  · rintro (h | h | h)
    · exact Or.inr (Or.inl h)
    · exact Or.inl h
    · exact Or.inr (Or.inr h)
  · rintro (h | h | h)
    · exact Or.inr (Or.inl h)
    · exact Or.inl h
    · exact Or.inr (Or.inr h)

end Member

end Verity.Sampled
