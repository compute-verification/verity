import Proofs.Flock.Claims.ZK.FrameBeta

/-!
# The frames refuse an outer coin `1` (`FrameUnrefused`'s outer fault)

At a `Live` statement the loop serves, on a live history of `refK x` openings whose last coins the frame takes, no
table's rep has a zerocheck outer coin `r_e = 1` among the opened coins (`Masked.Refused`'s first disjunct): the rep's
round 1 squeezes them, and the frame's taking of a round's coins in the algebraic phase stops at `r_e = 1`
(`Masker.answer`, `zk_veil::Masking::refuse_outer`).

* `Sched.rounds_two`: the algebraic phase's first two scheduled rounds are the zerocheck's binding with `r_skip`'s
  squeeze of `6` and `r_outer`'s squeeze of `m − 13` alone (`zerocheckFw_head`, then a recording whose first rounds stay
  put, `recLiftPre`).
* `Loop.run_outer`: the frame, from a rep's start, takes the binding (the algebraic phase on, `roundFree_zc`), its six
  coins, and `r_outer`'s round (`roundFree_zcOuter`): it owes `m − 13` coins on the six it kept.
* `Loop.answer_outer`: where it then takes coins, none of the first `n_mlv − 7` is `1`.
* **`LeanRelease.outer_unrefused`**: the fault. The opening at `Masked.outerSlot x j r` is rep `2j + r`'s round 1, whose
  state is that one (`repState_outer`).
-/

namespace Claims.ZKServer

open Claims Claims.ZKMal
open _root_.Flock (V F128 Stmt SessionSpec Transcript TMW)
open _root_.Flock.Transcript (header framed)
open _root_.Flock.Firewall.Frame (Cfg Masker K_SKIP N_INNER roundFree)
open FlockSoundness.Refine (Frame encs)

namespace Sched

section Skel

open _root_.Flock.Firewall.Skeleton (RecM recAsks padsFrames)
open _root_.Flock.Firewall.Finish (Asks squeezeP sampleP sampleVecP zerocheckFw lincheckFw repPiop)

/-- Two recordings alike, both holding the rounds `pre` first: as many coins drawn. -/
def RecPre (pre : List (ByteArray × ℕ)) (s s' : Array (ByteArray × ℕ) × ℕ) : Prop :=
  s.2 = s'.2 ∧ pre <+: s.1.toList ∧ pre <+: s'.1.toList

/-- Two recording programs alike from recordings holding `pre` first. -/
def RecPreR (pre : List (ByteArray × ℕ)) {α β : Type} (S : α → β → Prop) (x : RecM α) (y : RecM β) : Prop :=
  ∀ s s', RecPre pre s s' → RelV (fun a b => S a.1 b.1 ∧ RecPre pre a.2 b.2) (x s) (y s')

/-- **The recording's lift** from recordings holding `pre` first: a round only adds to them. -/
def recLiftPre (pre : List (ByteArray × ℕ)) : Lift (M := RecM) recAsks where
  R := RecPreR pre
  pure hab := fun _ _ hs => And.intro hab hs
  bind hxy hfg := fun s s' hs => by
    rw [stateT_bind_apply, stateT_bind_apply]
    exact relV_bind (hxy s s' hs) fun a b hab => hfg a.1 b.1 hab.1 a.2 b.2 hab.2
  throw _ _ := fun _ _ _ => True.intro
  liftOk hab := fun _ _ hs => And.intro hab hs
  round := by
    intro c c' hc
    obtain ⟨n, hn, hn', -⟩ := roundAt_of_alike hc
    rintro ⟨rs, k⟩ ⟨rs', k'⟩ ⟨hk, hp, hp'⟩
    dsimp only at hk
    subst hk
    unfold recAsks
    dsimp only
    rw [hn, hn']
    refine And.intro rfl (And.intro rfl (And.intro ?_ ?_))
    · rw [Array.toList_push]; exact hp.trans (List.prefix_append _ _)
    · rw [Array.toList_push]; exact hp'.trans (List.prefix_append _ _)

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- The masked zerocheck after `r_outer`'s squeeze, at its coins `rOuter`. -/
def zcTail (io : Asks M) (m : ℕ) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (rOuter : Array F128) :
    TMW M (_root_.Flock.ZerocheckProof × _root_.Flock.ZcOut) := do
  let r := _root_.Flock.pinned ++ rOuter
  let r1ab' := (Array.range 64).map fun i => mk (lay.r1ab i)
  let r1c' := (Array.range 64).map fun i => mk (lay.r1c i)
  _root_.Flock.liftTM (_root_.Flock.Transcript.absorbSlice r1ab')
  _root_.Flock.liftTM (_root_.Flock.Transcript.absorbSlice r1c')
  let z ← sampleP io
  let mut rounds : Array (F128 × F128) := #[]
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let g1' := mk (lay.ml i 0)
    let ginf' := mk (lay.ml i 1)
    _root_.Flock.liftTM (_root_.Flock.Transcript.absorbF g1')
    _root_.Flock.liftTM (_root_.Flock.Transcript.absorbF ginf')
    let rr ← sampleP io
    rho := rho.push rr
    rounds := rounds.push (g1', ginf')
  _root_.Flock.liftTM (_root_.Flock.Transcript.absorbF (mk lay.fa))
  _root_.Flock.liftTM (_root_.Flock.Transcript.absorbF (mk lay.fb))
  return (⟨r1ab', r1c', rounds, mk lay.fa, mk lay.fb, mk lay.fc⟩, { z, rho, rRest := r, vA := 0, vB := 0, vC := 0 })

theorem zcRest_eq (io : Asks M) (m : ℕ) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    zcRest io m lay mk = sampleVecP io (m - 13) >>= zcTail io m lay mk :=
  rfl

theorem zcTail_sim {io : Asks M} {L : Lift io} (m : ℕ) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (rO : Array F128) : Sim L (· = ·) (zcTail io m lay mk rO) (zcTail io m lay mk rO) := by
  unfold zcTail
  refine sim_bind (sim_absorbSlice rfl (by simp)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbSlice rfl (by simp)) fun _ _ _ => ?_
  refine sim_bind sim_sampleP fun z z' h => ?_
  subst h
  refine sim_bind (sim_forIn (m - 6) _ _ (fun _ u u' => u = u') (fun i _ u u' hu => ?_) rfl) fun u u' h => ?_
  · subst hu
    refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
    refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
    refine sim_bind sim_sampleP fun a b h => ?_
    subst h
    exact sim_pure ⟨_, _, rfl, rfl, rfl⟩
  · subst h
    refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
    refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
    exact sim_pure rfl

/-- A squeeze of `n ≠ 0` coins on the recording: its framed bytes recorded with their count. -/
theorem sampleVecP_rec {n : ℕ} (h0 : n ≠ 0) (hn : n < 2 ^ 64) (t : Transcript) (rs : Array (ByteArray × ℕ)) (k : ℕ) :
    sampleVecP recAsks n t (rs, k) =
      .ok (((Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
          { t with pending := ByteArray.empty, k := t.k + 1,
                   rounds := t.rounds.push ⟨n, ByteArray.empty,
                     (Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
                     some (framed t 2 n)⟩ }),
          (rs.push (framed t 2 n, n), k + n)) := by
  have e : sampleVecP recAsks n t (rs, k) =
      .ok (((Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
          { t with pending := ByteArray.empty, k := t.k + 1,
                   rounds := t.rounds.push ⟨n, ByteArray.empty,
                     (Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
                     some (framed t 2 n)⟩ }),
          (rs.push (framed t 2 n, (_root_.Flock.u64OfLE (framed t 2 n) ((framed t 2 n).size - 8)).toNat),
           k + (_root_.Flock.u64OfLE (framed t 2 n) ((framed t 2 n).size - 8)).toNat)) := by
    simp only [sampleVecP, squeezeP, h0, if_false]
    rfl
  rw [e, Count.framed_count t 2 hn]

/-- The second round the recording holds from a rep's start: `r_outer`'s squeeze, framed on an empty pending. -/
theorem framed_skelT8 (c : ByteArray) (n : ℕ) : framed (skelT8 c) 2 n = ByteArray.empty ++ header 4 2 n :=
  rfl

theorem ok_bind' {α β : Type} (a : α) (f : α → V β) : ((Except.ok a : V α) >>= f) = f a :=
  rfl

/-- **The recorded PIOP's first two rounds**, from a rep's start: the zerocheck's binding with `r_skip`'s squeeze, then
`r_outer`'s squeeze alone. -/
theorem skel_two (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root dom : ByteArray)
    (hm : 14 ≤ st.m) (hm' : st.m < 2 ^ 64) :
    RelV (fun a b => RecPre [(zcPend st.digest cap root dom ++ header 4 2 6, 6),
        (framed (skelT8 (zcPend st.digest cap root dom ++ header 4 2 6)) 2 (st.m - 13), st.m - 13)] a.2 b.2)
      (((repPiop recAsks st lay cap root #[] fun _ => 0) >>= fun _ => pure () : TMW RecM Unit).run'
        (Transcript.init #[] dom) |>.run (#[], 0))
      (((repPiop recAsks st lay cap root #[] fun _ => 0) >>= fun _ => pure () : TMW RecM Unit).run'
        (Transcript.init #[] dom) |>.run (#[], 0)) := by
  have hcnt : (_root_.Flock.u64OfLE (zcPend st.digest cap root dom ++ header 4 2 6)
      ((zcPend st.digest cap root dom ++ header 4 2 6).size - 8)).toNat = 6 :=
    count_header _ 2 6 (by decide)
  have hs1 : skelS1 (zcPend st.digest cap root dom ++ header 4 2 6) =
      ((#[] : Array (ByteArray × ℕ)).push (zcPend st.digest cap root dom ++ header 4 2 6, 6), 0 + 6) := by
    unfold skelS1
    rw [hcnt]
  have h0 : st.m - 13 ≠ 0 := by omega
  rw [tmw_run'_run]
  refine relV_bind (S := fun a b => RecPre _ a.2 b.2) ?_ fun a b h => by exact h
  rw [tmw_bind_apply]
  refine relV_bind (S := fun a b => RecPre _ a.2 b.2) ?_ fun a b h => by exact h
  unfold repPiop
  rw [tmw_bind_apply, zerocheckFw_head, zcRest_eq, tmw_bind_apply, hs1, sampleVecP_rec h0 (by omega), ok_bind']
  dsimp only
  refine relV_bind (zcTail_sim (L := recLiftPre _) st.m lay _ _ _ _ (by exact pendAlike_empty) _ _
    ⟨rfl, List.prefix_refl _, List.prefix_refl _⟩) fun a b hab => ?_
  obtain ⟨⟨x, t1⟩, s1⟩ := a
  obtain ⟨⟨y, t1'⟩, s1'⟩ := b
  obtain ⟨⟨hxy, ht⟩, hs⟩ := hab
  dsimp only at hxy ht hs
  subst hxy
  obtain ⟨zp, zc⟩ := x
  dsimp only
  exact relV_mono (sim_bind (L := recLiftPre _) (S := (· = ·)) (T := (· = ·))
    (lincheckFw_sim st.fold lay (fun _ => 0) zc) (fun u v h => by subst h; exact sim_pure rfl)
    t1 t1' ht s1 s1' hs) fun _ _ h => h.2

/-- The algebraic phase's schedule, related to itself, holds its first two rounds. -/
theorem rounds_two_rel (st : _root_.Flock.Setup) (k0 : ℕ) (cap dom : ByteArray) (hm : 14 ≤ st.m)
    (hm' : st.m < 2 ^ 64) :
    RelV (fun (a : Array (ByteArray × ℕ)) (_ : Array (ByteArray × ℕ)) =>
        a[0]? = some (_root_.Flock.Firewall.Skeleton.head st cap dom ++ header 4 2 6, 6) ∧
          a[1]? = some (ByteArray.empty ++ header 4 2 (st.m - 13), st.m - 13))
      (_root_.Flock.Firewall.Skeleton.rounds st k0 cap dom) (_root_.Flock.Firewall.Skeleton.rounds st k0 cap dom) := by
  unfold _root_.Flock.Firewall.Skeleton.rounds
  dsimp only
  refine relV_bind (skel_two st _ cap _ dom hm hm') fun a b hab => ?_
  obtain ⟨_, rs, k⟩ := a
  obtain ⟨_, rs', k'⟩ := b
  obtain ⟨-, ⟨l, hl⟩, ⟨l', hl'⟩⟩ := hab
  dsimp only at hl hl'
  rw [framed_skelT8] at hl hl'
  have hr0 : rs[0]? = some (zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6, 6) := by
    rw [← Array.getElem?_toList, ← hl]; rfl
  have hr0' : rs'[0]? = some (zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6, 6) := by
    rw [← Array.getElem?_toList, ← hl']; rfl
  have hr1 : rs[1]? = some (ByteArray.empty ++ header 4 2 (st.m - 13), st.m - 13) := by
    rw [← Array.getElem?_toList, ← hl]; rfl
  rw [hr0, hr0']
  dsimp only
  have key : zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6 =
      _root_.Flock.Firewall.Skeleton.head st cap dom ++ (padsFrames (ByteArray.mk (Array.replicate 64 0)) ++
        header 4 2 6) := by
    rw [zcPend_eq, head_eq, ByteArray.append_assoc]
  have e1 : (zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).extract 0
      (_root_.Flock.Firewall.Skeleton.head st cap dom).size = _root_.Flock.Firewall.Skeleton.head st cap dom := by
    rw [key]
    exact ByteArray.extract_append_eq_left rfl
  have e2 : (zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).extract
      (_root_.Flock.Firewall.Skeleton.head st cap dom).size ((_root_.Flock.Firewall.Skeleton.head st cap dom).size +
        (padsFrames (ByteArray.mk (Array.replicate 64 0))).size) =
      padsFrames (ByteArray.mk (Array.replicate 64 0)) := by
    have := @ByteArray.extract_append_size_add (_root_.Flock.Firewall.Skeleton.head st cap dom)
      (padsFrames (ByteArray.mk (Array.replicate 64 0)) ++ header 4 2 6) 0
      (padsFrames (ByteArray.mk (Array.replicate 64 0))).size
    simp only [Nat.add_zero] at this
    rw [key, this]
    exact ByteArray.extract_append_eq_left rfl
  have e3 : (zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).extract
      ((_root_.Flock.Firewall.Skeleton.head st cap dom).size +
        (padsFrames (ByteArray.mk (Array.replicate 64 0))).size)
      (zcPend st.digest cap (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).size = header 4 2 6 := by
    rw [key, ← ByteArray.append_assoc]
    exact ByteArray.extract_append_eq_right ByteArray.size_append.symm ByteArray.size_append
  simp only [e1, e2, e3, bne_self_eq_false, Bool.false_eq_true, or_self, ↓reduceIte]
  have hs : 1 < rs.size := (Array.getElem?_eq_some_iff.1 hr1).1
  show _ ∧ _
  simp only [Array.set!_eq_setIfInBounds]
  refine ⟨?_, ?_⟩
  · rw [Array.getElem?_setIfInBounds_self_of_lt (by omega)]
  · rw [Array.getElem?_setIfInBounds_ne (by omega), hr1]

/-- **The algebraic phase's first two scheduled rounds**: the zerocheck's binding (the statement's, without the pads'
frames) with `r_skip`'s squeeze of `6`, and `r_outer`'s squeeze of `m − 13` alone. -/
theorem rounds_two {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (hm : 14 ≤ st.m) (hm' : st.m < 2 ^ 64)
    (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) :
    alg[0]? = some (_root_.Flock.Firewall.Skeleton.head st cap dom ++ header 4 2 6, 6) ∧
      alg[1]? = some (ByteArray.empty ++ header 4 2 (st.m - 13), st.m - 13) := by
  have := rounds_two_rel st k0 cap dom hm hm'
  rw [h] at this
  exact this

end Skel

end Sched

end Claims.ZKServer
