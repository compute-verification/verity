module

public import Proofs.Properties.Verity.Sampled.Window.Reach

/-! SPOT's coupling over a window: each member's written coins in the play's log. -/

@[expose] public section

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ addRoundJ)
open FlockSoundness.Refine (roundOf)

local notation "𝔽" => FlockLevel3.GF128

section Past

variable {g : Game Exchange} {σ : Strategy g} {R : ℕ}

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

section Member

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R nJ : ℕ} {D : ℕ → PosDecoders g σ R Reg}
  {M : ExecutionPhaseCoinMap g σ Ω R} {BL : ExecutionPhaseLinkHold} {B : ExecutionPhaseRoundHold} {S : Finset (Fin nJ)} {Rg : Reg}

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

variable {ι : Type} {d : ResourceAccounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

theorem stopOf_rootStop {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopOf rd.round rd.reveal [] (finish (V.signer θ) θ F c σ.1) (fun _ => True)
      (Pos.root (roundCuts V θ F c rd) x).toStop := by
  have h := walk_send_cast (rv := rd.reveal) (f := finish (V.signer θ) θ F c) (a := fun _ => True) hP
    (game_eq_relK V θ F c).symm σ (roundCuts V θ F c rd) x id fun _ => Or.inl ⟨trivial, rfl⟩
  have e : (cast (congrArg Strategy (game_eq_relK V θ F c)) σ).1 = σ.1 := cast_send_fst (game_eq_relK V θ F c) σ
  rw [e] at h
  exact h

theorem stopsAt_rootStop {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c)) (x : CoinTape (V.game θ F c) σ)
    (hP : ∀ j pre fr, rd.round j pre fr → ∃ i s n, fr = .opening i s n) :
    StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 (Pos.root (roundCuts V θ F c rd) x).toStop :=
  ⟨[], _, fun _ => True, fun _ => rfl, stopOf_rootStop V θ F c rd σ x hP⟩

structure MemberSteps (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ExecutionPhaseDecoders V θ F c W σ)
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold)
    (I : (W.extend M.widening).L.Ω → Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → ℕ →
      Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds) → Prop) : Prop where
  start : ∀ ω r, I ω r 0 (Pos.root (roundCuts V θ F c rd) (M.start ω))
  step : ∀ (ω : (W.extend M.widening).L.Ω) (i : ℕ) (m : Flock.Guarantees.MemberV5 ByteArray W.nJ)
    (p q r : Pos (V.game θ F c) σ Claims.Msg (Fin W.rounds))
    (out : LiveOutJ (memberTabs ((W.extend M.widening).L.draw ω) (executionPhaseReg V θ F c W rd σ D) m) (memberPts m)),
    ((W.extend M.widening).joint t).ms[i]? = some m → (m.part ((W.extend M.widening).L.draw ω)).Nonempty →
    I ω r i p → p.Inv (roundCuts V θ F c rd) →
    StopsAt rd.round rd.reveal (V.signer θ) θ F c σ.1 p.toStop →
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

theorem memberDecodes_of_stops (k : Verity.Registration.Composite) (P : Exchange → Prop) (W : ExecutionPhase) (t : ℕ)
    (rd : Reading W.rounds) (σ : Strategy (V.game θ F c)) (D : ExecutionPhaseDecoders V θ F c W σ)
    (M : ExecutionPhaseCoinMap (V.game θ F c) σ W.Ω W.rounds) (BL : ExecutionPhaseLinkHold) (B : ExecutionPhaseRoundHold)
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
