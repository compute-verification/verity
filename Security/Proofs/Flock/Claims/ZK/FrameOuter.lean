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

omit [MonadExceptOf String M] in
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
    simp only [sampleVecP, squeezeP, h0, ↓reduceIte]
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

/-! ## The frame through `r_outer`'s round -/

/-- **The zerocheck's binding round at a rep's start**: `roundFree_zcBind` with the stream's domain frame first
(`Transcript.init`'s), which the frame skips. -/
theorem roundFree_zcHead (c : Cfg) (s : Masker) (digest cap dom : ByteArray)
    (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (hon : s.on = false) (hd : s.done = false)
    (hi : s.inner = none) (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64)
    (hb : s.pos ≤ c.order.size) :
    ∃ out fr, roundFree c s
        (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap,
          .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray) 6 =
        .ok (out, { s with on := true, owed := some 6 }, fr) := by
  obtain ⟨out, fr, hr, -⟩ := roundFree_zc c s [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest, .bytes cap] []
    2 6 hc hon hd hi
    (by
      intro f hf
      simp only [List.cons_append, List.nil_append, List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl | rfl
      · exact hdom
      · show "flock-r1cs-v0".toUTF8.size < 2 ^ 64; decide
      · exact hdg
      · exact hcap
      · show _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8.size < 2 ^ 64; decide)
    (by
      intro f hf
      simp only [List.mem_cons, List.not_mem_nil, or_false] at hf
      rcases hf with rfl | rfl | rfl | rfl
      · trivial
      · show plainLabel "flock-r1cs-v0".toUTF8; unfold plainLabel; decide
      · trivial
      · trivial)
    (by simp) (.inl rfl) (by decide) (by decide) hb
  exact ⟨out, fr, hr⟩

/-- The algebraic phase's first scheduled round is the binding's frames, then `r_skip`'s squeeze. -/
theorem head_encs (st : _root_.Flock.Setup) (cap dom : ByteArray) :
    _root_.Flock.Firewall.Skeleton.head st cap dom ++ header 4 2 6 =
      ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes st.digest, .bytes cap,
        .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray := by
  apply ByteArray.ext
  apply Array.ext'
  rw [List.toList_toArray, data_append', Sched.head_eq]
  unfold Sched.zcPre
  rw [FlockSoundness.Refine.label_enc, FlockSoundness.Refine.absorbBytes_enc,
    FlockSoundness.Refine.absorbBytes_enc, FlockSoundness.Refine.label_enc, FlockSoundness.Refine.init_enc]
  simp only [encs, List.flatMap_cons, List.flatMap_nil, List.append_nil, List.append_assoc, sq_enc,
    _root_.Flock.Firewall.Frame.LABEL_ZC]

/-- `r_outer`'s scheduled round is its squeeze alone. -/
theorem outer_encs (n : ℕ) :
    ByteArray.empty ++ header 4 2 n = ByteArray.mk (encs [] ++ (Frame.sq 2 n).enc).toArray := by
  apply ByteArray.ext
  apply Array.ext'
  rw [List.toList_toArray, data_append', sq_enc]
  rfl

namespace Loop

open _root_.Flock.Firewall.Frame (check)

/-- A loop each of whose steps yields or refuses, where it returns, yielded at each. -/
theorem forIn_yield_each {g : ℕ → PUnit → V (ForInStep PUnit)}
    (hg : ∀ i, g i PUnit.unit = .ok (.yield PUnit.unit) ∨ ∃ e, g i PUnit.unit = .error e) :
    ∀ (l : List ℕ) (y : PUnit), forIn l PUnit.unit g = .ok y → ∀ i ∈ l, g i PUnit.unit = .ok (.yield PUnit.unit)
  | [], _, _, i, hi => by simp at hi
  | a :: l, y, h, i, hi => by
    rw [List.forIn_cons] at h
    rcases hg a with ha | ⟨e, ha⟩
    · rw [ha] at h
      simp only [bind, Except.bind] at h
      rcases List.mem_cons.1 hi with rfl | hi
      · exact ha
      · exact forIn_yield_each hg l y h i hi
    · rw [ha] at h
      simp only [bind, Except.bind, reduceCtorEq] at h

theorem getElem!_append_size {a b : Array F128} {e : ℕ} (he : e < b.size) : (a ++ b)[a.size + e]! = b[e]! := by
  rw [getElem!_pos (a ++ b) (a.size + e) (by rw [Array.size_append]; omega), getElem!_pos b e he]
  simp

/-- Where the frame takes coins for a round owing `n`, they are `n`. -/
theorem answer_size {cfg : Cfg} {s m : Masker} {cs : Array F128} {n : ℕ} (ho : s.owed = some n)
    (h : s.answer cfg cs = .ok m) : cs.size = n := by
  unfold Masker.answer at h
  simp only [ho] at h
  by_contra hc
  have hne : (cs.size != n) = true := by simp [hc]
  simp only [hne, ↓reduceIte] at h
  cases h

/-- **Where the frame takes a round's coins in the algebraic phase**, none at a zerocheck outer coin's place is `1`
(`Masker.answer`, `refuse_outer`). -/
theorem answer_outer {cfg : Cfg} {s m : Masker} {cs : Array F128} {n : ℕ} (ho : s.owed = some n)
    (hi : s.inner = none) (hl : s.l0 = false) (hon : s.on = true) (h : s.answer cfg cs = .ok m) :
    cs.size = n ∧ ∀ e < n, K_SKIP ≤ s.coins.size + e → s.coins.size + e < K_SKIP + (cfg.lay.nMlv - N_INNER) →
      (cs[e]! + 1).isZero = false := by
  have hc := answer_size ho h
  refine ⟨hc, fun e he h1 h2 => ?_⟩
  unfold Masker.answer at h
  have hne : (cs.size != n) = false := by simp [hc]
  simp only [ho, hi, hl, hne, hon, Bool.false_eq_true, ↓reduceIte, Bool.not_true] at h
  obtain ⟨_, hf, -⟩ := except_bind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hf
  have hk := forIn_yield_each (fun i => by
      try dsimp only
      split
      · exact Or.inr ⟨_, rfl⟩
      · exact Or.inl rfl) _ _ hf (s.coins.size + e) (by
      rw [List.mem_range']
      refine ⟨e, ?_, by simp⟩
      simp only [Std.Legacy.Range.size, Array.size_append]
      omega)
  by_contra hz
  rw [Bool.not_eq_false] at hz
  have hz' : ((s.coins ++ cs)[s.coins.size + e]! + 1).isZero = true := by
    rw [getElem!_append_size (by omega)]
    exact hz
  try dsimp only at hk
  simp only [h1, h2, hz', decide_true, Bool.and_self, ↓reduceIte] at hk
  cases hk

/-- **The frame's check of a round against itself**: its state is `roundFree`'s, at the rep's state or its stop. -/
theorem check_self {c : Cfg} {s : Masker} {r : ByteArray} {n : ℕ} {out : ByteArray} {sx : Masker}
    {why : Option String} (h : check c s r r n = .ok (out, sx, why)) :
    (∃ fr, roundFree c s r n = .ok (out, sx, fr)) ∨
      ∃ fr, roundFree c { s with stopped := true } r n = .ok (out, sx, fr) := by
  have key : ∀ {s' : Masker}, _root_.Flock.Firewall.Frame.round c s' r n = .ok (out, sx) →
      ∃ fr, roundFree c s' r n = .ok (out, sx, fr) := by
    intro s' hs
    unfold _root_.Flock.Firewall.Frame.round at hs
    rcases hr : roundFree c s' r n with e | ⟨o, t, fr⟩
    · rw [hr] at hs
      simp only [bind, Except.bind, reduceCtorEq] at hs
    · rw [hr] at hs
      simp only [bind, Except.bind, pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hs
      obtain ⟨rfl, rfl⟩ := hs
      exact ⟨fr, rfl⟩
  cases why with
  | none => exact Or.inl (key (check_none h))
  | some w => exact Or.inr (key (check_some h))

/-- A round with itself in its place, past which the rep runs on: the frame took it, or refused it and stopped. -/
theorem run_sched_cons {c : Cfg} {st : _root_.Flock.Firewall.Frame.State} {n : ℕ} {r : ByteArray} {rs : List _root_.Flock.Firewall.Frame.Req}
    (ho : st.m.owed = none) (h2 : (_root_.Flock.Firewall.Frame.run c st (.round n r (.sched r) :: rs)).2 = false) :
    (∃ out sx why, check c st.m r r n = .ok (out, sx, why) ∧
      _root_.Flock.Firewall.Frame.run c st (.round n r (.sched r) :: rs) = _root_.Flock.Firewall.Frame.run c { m := sx, begun := true } rs) ∨
    _root_.Flock.Firewall.Frame.run c st (.round n r (.sched r) :: rs) = _root_.Flock.Firewall.Frame.run c { m := { st.m with stopped := true }, begun := true } rs := by
  rcases hc : check c st.m r r n with e | ⟨out, sx, why⟩
  · right
    cases hs : e.startsWith _root_.Flock.Firewall.Frame.STOP
    · simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ho, hc, hs, Bool.false_eq_true, ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends]
    · exfalso
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ho, hc, hs, ↓reduceIte, _root_.Flock.Firewall.Frame.Reply.ends] at h2
      cases h2
  · left
    refine ⟨out, sx, why, rfl, ?_⟩
    simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, ho, hc, _root_.Flock.Firewall.Frame.Reply.ends, Bool.false_eq_true, ↓reduceIte]

/-- An earlier rep's coins, past which the rep runs on, add claim points alone. -/
theorem run_prior (c : Cfg) (P : Array F128) (rs : List _root_.Flock.Firewall.Frame.Req)
    (h2 : (_root_.Flock.Firewall.Frame.run c {} ((if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
      rs)).2 = false) :
    ∃ p, _root_.Flock.Firewall.Frame.run c {} ((if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++ rs) =
      _root_.Flock.Firewall.Frame.run c { m := { prior := p } } rs := by
  by_cases hP : P.isEmpty = true
  · exact ⟨#[], by simp only [hP, ↓reduceIte, List.nil_append]⟩
  · simp only [hP, Bool.false_eq_true, ↓reduceIte, List.singleton_append] at h2 ⊢
    rcases hcp : _root_.Flock.Firewall.Frame.claimPoints c.lay
        (_root_.Flock.Firewall.Frame.words (_root_.Flock.Firewall.Frame.wordBytes P) 0 P.size) with e | p
    · exfalso
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hcp,
        _root_.Flock.Firewall.Frame.Reply.ends] at h2
      cases h2
    · refine ⟨#[] ++ p, ?_⟩
      simp only [_root_.Flock.Firewall.Frame.run, _root_.Flock.Firewall.Frame.step, hcp,
        _root_.Flock.Firewall.Frame.Reply.ends, Bool.false_eq_true, ↓reduceIte]

/-- **The frame through `r_outer`'s round**, from a rep's start past an earlier rep's coins: where the binding's
round, its coins and `r_outer`'s round (`n` coins) neither end the rep nor leave it owing none, the frame is in the
algebraic phase, owing `n` coins, on the binding's six. -/
theorem run_outer {c : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size) (P : Array F128)
    (sched : Array (ByteArray × ℕ)) (cr : Array (Array F128)) {digest cap dom : ByteArray} {n : ℕ}
    (hdg : digest.size < 2 ^ 64) (hcap : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64) (hn0 : n ≠ 0)
    (hn : n < 2 ^ 64)
    (h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8, .bytes digest,
      .bytes cap, .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6))
    (h1 : sched.getD 1 (.empty, 0) = (ByteArray.mk (encs [] ++ (Frame.sq 2 n).enc).toArray, n))
    (h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).2 = false) (ho : (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).1.m.owed ≠ none) :
    (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).1.m.owed = some n ∧ (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).1.m.on = true ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).1.m.inner = none ∧ (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).1.m.l0 = false ∧
      (_root_.Flock.Firewall.Frame.run c {} (schedReqs P sched cr 1)).1.m.coins.size = 6 := by
  have e1 : schedReqs P sched cr 1 =
      (if P.isEmpty then [] else [.prior P.size (_root_.Flock.Firewall.Frame.wordBytes P)]) ++
        (.round (sched.getD 0 (.empty, 0)).2 (sched.getD 0 (.empty, 0)).1 (.sched (sched.getD 0 (.empty, 0)).1) ::
          .coins (.ok (cr.getD 0 #[])) ::
          [.round (sched.getD 1 (.empty, 0)).2 (sched.getD 1 (.empty, 0)).1 (.sched (sched.getD 1 (.empty, 0)).1)]) := by
    rw [schedReqs_succ]
    unfold schedReqs
    simp only [List.range_zero, List.flatMap_nil, List.append_nil, Nat.zero_add, List.append_assoc,
      List.cons_append, List.nil_append]
  rw [e1, h0, h1] at h2 ho ⊢
  dsimp only at h2 ho ⊢
  obtain ⟨p, hp⟩ := run_prior c P _ h2
  rw [hp] at h2 ho ⊢
  rcases run_sched_cons (st := { m := { prior := p } }) rfl h2 with ⟨out, sx, why, hck, he⟩ | he
  · rw [he] at h2 ho ⊢
    have hsx : sx = { ({ prior := p } : Masker) with on := true, owed := some 6 } ∨
        sx = { ({ prior := p } : Masker) with stopped := true, on := true, owed := some 6 } := by
      rcases check_self hck with ⟨fr, hr⟩ | ⟨fr, hr⟩
      · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { prior := p } digest cap dom hc rfl rfl rfl hdg hcap hdom
          (Nat.zero_le _)
        rw [hr'] at hr
        simp only [Except.ok.injEq, Prod.mk.injEq] at hr
        exact Or.inl hr.2.1.symm
      · obtain ⟨o, fr', hr'⟩ := roundFree_zcHead c { ({ prior := p } : Masker) with stopped := true } digest cap dom
          hc rfl rfl rfl hdg hcap hdom (Nat.zero_le _)
        rw [hr'] at hr
        simp only [Except.ok.injEq, Prod.mk.injEq] at hr
        exact Or.inr hr.2.1.symm
    have hsxo : sx.owed = some 6 := by rcases hsx with rfl | rfl <;> rfl
    have hsxi : sx.inner = none := by rcases hsx with rfl | rfl <;> rfl
    have hsxl : sx.l0 = false := by rcases hsx with rfl | rfl <;> rfl
    have hsxon : sx.on = true := by rcases hsx with rfl | rfl <;> rfl
    have hsxc : sx.coins = #[] := by rcases hsx with rfl | rfl <;> rfl
    have hsxp : sx.pos = 0 := by rcases hsx with rfl | rfl <;> rfl
    obtain ⟨s1, ha, he1⟩ := run_coins_cons c { m := sx, begun := true } _ _ h2
    rw [he1] at h2 ho ⊢
    have hsz := answer_size hsxo ha
    have hs1 := answer_alg c sx _ hsxi hsxl ha
    have hs1o : s1.owed = none := by rw [hs1]
    have hs1on : s1.on = true := by rw [hs1]; exact hsxon
    have hs1i : s1.inner = none := by rw [hs1]; exact hsxi
    have hs1l : s1.l0 = false := by rw [hs1]; exact hsxl
    have hs1p : s1.pos = 0 := by rw [hs1]; exact hsxp
    have hs1c : s1.coins.size = 6 := by rw [hs1]; simp only [hsxon, hsxc, ↓reduceIte, Array.empty_append]; exact hsz
    rcases run_sched_cons (st := { ({ m := sx, begun := true } : _root_.Flock.Firewall.Frame.State) with m := s1 })
      hs1o h2 with ⟨out1, sx1, why1, hck1, he2⟩ | he2
    · rw [he2]
      have hsx1 : sx1.owed = some n ∧ sx1.on = true ∧ sx1.inner = none ∧ sx1.l0 = false ∧ sx1.coins.size = 6 := by
        rcases check_self hck1 with ⟨fr, hr⟩ | ⟨fr, hr⟩
        · obtain ⟨o, fr', hr', -⟩ := roundFree_zcOuter c s1 n hc hs1on hs1i hn0 hn (by rw [hs1p]; exact Nat.zero_le _)
          have e := Except.ok.inj (hr.symm.trans hr')
          simp only [Prod.mk.injEq] at e
          rw [e.2.1]
          exact ⟨rfl, hs1on, hs1i, hs1l, hs1c⟩
        · obtain ⟨o, fr', hr', -⟩ := roundFree_zcOuter c { s1 with stopped := true } n hc hs1on hs1i hn0 hn
            (by rw [hs1p]; exact Nat.zero_le _)
          have e := Except.ok.inj (hr.symm.trans hr')
          simp only [Prod.mk.injEq] at e
          rw [e.2.1]
          exact ⟨rfl, hs1on, hs1i, hs1l, hs1c⟩
      exact hsx1
    · rw [he2] at ho
      exact (ho hs1o).elim
  · rw [he] at h2
    obtain ⟨s1, ha, -⟩ := run_coins_cons c _ _ _ h2
    exact absurd ha (answer_owed_none _ rfl)

/-- A rep's schedule so far starts with the algebraic phase's. -/
theorem schedSoFar_pre {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr = .ok sched) : ∃ rest, sched = alg ++ rest := by
  unfold schedSoFar at hs
  rw [hal] at hs
  simp only [bind, Except.bind] at hs
  split at hs
  · simp only [pure, Except.pure, Except.ok.injEq] at hs
    exact ⟨#[], by rw [← hs, Array.append_empty]⟩
  · split at hs
    · simp only [pure, Except.pure, Except.ok.injEq] at hs
      exact ⟨_, hs.symm⟩
    · simp only [pure, Except.pure, Except.ok.injEq] at hs
      exact ⟨_, hs.symm⟩
    · cases hs

/-- **The loop's state at a rep's `r_outer` round**: where the frame takes coins there, they are `m − 13`, and none
in an outer place of the layout `c''` is `1`. -/
theorem repState_outer {c c'' : Cfg} (hc : ∀ p, p < c.order.size → c.order[p]! < c.h.size)
    {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray} {xs : Array (Array F128)}
    {hi : Array ByteArray} {cr0 cr : Array (Array F128)} {ns : ℕ → ℕ} {alg sched : Array (ByteArray × ℕ)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched) (hm : 14 ≤ t.st.m) (hm' : t.st.m < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hcap : (_root_.Flock.Firewall.Finish.flat t.cm0.cap).size < 2 ^ 64)
    (hdom : dom.size < 2 ^ 64) {cs : Array F128} {m : Masker}
    (ha : (repState c t rep dom xs hi cr0 cr ns (1 + 1)).answer c'' cs = .ok m) :
    cs.size = t.st.m - 13 ∧ ∀ e < t.st.m - 13, e < c''.lay.nMlv - N_INNER → (cs[e]! + 1).isZero = false := by
  obtain ⟨ha0, ha1⟩ := Sched.rounds_two hm hm' hal
  obtain ⟨rest, hrest⟩ := schedSoFar_pre hal hs
  have hg : ∀ k, k < alg.size → sched.getD k (.empty, 0) = (alg[k]?).getD (.empty, 0) := fun k hk => by
    rw [hrest, Array.getD_eq_getD_getElem?, Array.getElem?_append_left hk]
  have h1s : 1 < alg.size := (Array.getElem?_eq_some_iff.1 ha1).1
  have h0 : sched.getD 0 (.empty, 0) = (ByteArray.mk (encs [.dom dom, .label "flock-r1cs-v0".toUTF8,
      .bytes t.st.digest, .bytes (_root_.Flock.Firewall.Finish.flat t.cm0.cap),
      .label _root_.Flock.Firewall.Frame.LABEL_ZC.toUTF8] ++ (Frame.sq 2 6).enc).toArray, 6) := by
    rw [hg 0 (by omega), ha0, Option.getD_some, head_encs]
  have h1 : sched.getD 1 (.empty, 0) =
      (ByteArray.mk (encs [] ++ (Frame.sq 2 (t.st.m - 13)).enc).toArray, t.st.m - 13) := by
    rw [hg 1 h1s, ha1, Option.getD_some, outer_encs]
  rw [repState_at c t rep dom xs hi cr0 cr ns 1 hal hs (by rw [hrest, Array.size_append]; omega)] at ha
  dsimp only at ha
  cases h2 : (_root_.Flock.Firewall.Frame.run c {} (schedReqs
      (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched cr 1)).2
  · rw [h2] at ha
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    obtain ⟨hO, hon, hin, hl, hcs⟩ := run_outer hc _ sched cr hdg hcap hdom (by omega) (by omega) h0 h1 h2
      (fun e => answer_owed_none cs e ha)
    obtain ⟨hsz, hall⟩ := answer_outer hO hin hl hon ha
    refine ⟨hsz, fun e he hlt => hall e he ?_ ?_⟩
    · rw [hcs]; show 6 ≤ 6 + e; omega
    · rw [hcs]; show 6 + e < 6 + (c''.lay.nMlv - N_INNER); omega
  · rw [h2] at ha
    simp only [↓reduceIte] at ha
    exact absurd ha (answer_owed_none cs rfl)

/-- A rep's start lays out `m − 6` residual rounds. -/
theorem start_nMlv {st : Stmt} {lay : _root_.Flock.Zk.PadLayout} {mask : _root_.Flock.Firewall.Frame.MaskSlot}
    (h : _root_.Flock.Firewall.Drive.start st = .ok (lay, mask)) : lay.nMlv = st.m - 6 := by
  unfold _root_.Flock.Firewall.Drive.start at h
  obtain ⟨⟨sch, tp⟩, -, h⟩ := bindV_ok h
  obtain ⟨ms, -, h⟩ := bindV_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  rw [← h.1]
  rfl

end Loop


namespace LeanRelease

open Loop

/-- **The frames refuse an outer coin `1`**: at a `Live` statement the loop serves, on a live history of `refK x`
openings whose last coins the frame takes, table `j`'s rep `r`'s outer coin `e`, where it is among the opened coins, is
not `1`. The opening at `outerSlot` is the rep's round 1, `r_outer`'s, its frame there in the algebraic phase on the
binding's six coins (`repState_outer`), and the frame took its coins (`coins_taken_of_live`), which it does only at no
outer coin `1` (`answer_outer`). -/
theorem outer_unrefused (T : ZKMal.CoinLayer.TableData)
    (L₀ : LeanRelease Flock.Wit
      ((ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand))
    (real : Msg → Flock.Wit →
      (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand →
      ℕ → V (_root_.Flock.Firewall.Finish.Tab × Array ByteArray))
    (wk : (ofLoop T L₀ real).toSrv.Worker) (x : Msg) (w : Flock.Wit) (hx : Masked.Live x) (hSv : Serves x)
    (ρ : (ZKMal.CoinLayer.session loopNr T).refP.Nonce × (ZKMal.CoinLayer.session loopNr T).refP.Rand)
    (κ : (ofLoop T L₀ real).toSrv.Chk) (m₀ ρ₀ : Msg) (ops : List (Msg × Msg)) (hl : ops.length = refK x)
    (hlive : (ofLoop T L₀ real).toSrv.live wk x w ρ κ ((m₀, ρ₀) :: ops) = true)
    (hr : (ofLoop T L₀ real).refusesCoins x ρ ((m₀, ρ₀) :: ops) = false)
    {j r e : ℕ} (hj : j < (Masked.stsOf x).size) (hr2 : r < 2) (hb : Masked.outerSlot x j r < refK x)
    (he : e < (Masked.stsOf x)[j]!.m - 13) :
    Masked.outerAt (x := x) ((ZKMal.CoinLayer.session loopNr T).pad (K := refK x)
      (ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2)) j r e ≠ 1 := by
  obtain ⟨s0, spec, hsx⟩ := Masked.setupOf_of_pos (lt_of_le_of_lt (Nat.zero_le _) hj)
  obtain ⟨y⟩ := hx.2 j hj
  obtain ⟨-, -, hspec, -⟩ := FlockComplete.ZkTables.tables_of_setup hx.1 hsx
  have hq2 : 2 * j + r < spec.repStreams.size := by rw [FlockDeploy.repStreams_size, hspec]; omega
  have hst : (Masked.stsOf x)[(2 * j + r) / 2]! = (Masked.stsOf x)[j] := by
    rw [show (2 * j + r) / 2 = j by omega]
    exact getElem!_pos _ j hj
  have hstj : (Masked.stsOf x)[j]! = (Masked.stsOf x)[j] := getElem!_pos _ j hj
  have hz : _root_.Flock.Zk.schedule (Masked.stsOf x)[(2 * j + r) / 2]!.m = .ok (y.sch, y.tPad) := by
    rw [hst]; exact y.hz
  have hk6 : 6 ≤ (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog := by rw [hst]; exact y.wf.1.hk6
  obtain ⟨N, hN⟩ : ∃ N, N = FlockComplete.piopRoundsZk spec (Masked.stsOf x)[(2 * j + r) / 2]! +
      (y.sch.k0 + 7 + 11 * (y.sch.levels.size - 1)) := ⟨_, rfl⟩
  have hs : Masked.outerSlot x j r = FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) + 1 := by
    unfold Masked.outerSlot
    rw [hsx]
  have hrsl : (roundsOf spec (Masked.stsOf x)).length = spec.repStreams.size := by simp [roundsOf]
  have hrq : (roundsOf spec (Masked.stsOf x)).getD (2 * j + r) 0 = N + 4 := by
    simp only [roundsOf, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hq2, Option.map_some,
      Option.getD_some]
    have e1 : (Masked.stsOf x)[(2 * j + r) / 2]!.m + (Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6 =
        (Masked.stsOf x)[(2 * j + r) / 2]!.m + ((Masked.stsOf x)[(2 * j + r) / 2]!.c.kLog - 6) := by omega
    rw [repRounds_true]
    simp only [hz]
    rw [hN]
    unfold FlockComplete.piopRoundsZk
    rw [e1]
  have hs1 : 1 ≤ FlockComplete.repStartZk spec (Masked.stsOf x) (2 * j + r) := by
    unfold FlockComplete.repStartZk; omega
  obtain ⟨s, hsd⟩ : ∃ s, Masked.outerSlot x j r = s := ⟨_, rfl⟩
  rw [hsd] at hs hb
  have hlast : (ofLoop T L₀ real).last x = refK x + 1 := loop_last loopNr T (withFrames T L₀ real) x
  obtain ⟨-, m, ha⟩ := coins_taken_of_live (ofLoop T L₀ real) wk x w ρ κ ((m₀, ρ₀) :: ops) hlive
    (by rw [hlast]; simp [hl]) hr (i := s + 2) (by omega) (by simp only [List.length_cons]; omega)
  obtain ⟨⟨dm, da⟩, hd⟩ : ∃ d, ops[s]? = some d := ⟨ops[s]'(by omega), List.getElem?_eq_getElem _⟩
  have htk : ops.take (s + 1) = ops.take s ++ [(dm, da)] := by rw [List.take_add_one, hd]; rfl
  have hh : ((m₀, ρ₀) :: ops).take (s + 2) = (m₀, ρ₀) :: (ops.take s ++ [(dm, da)]) := by
    rw [List.take_succ_cons, htk]
  have hts : (ops.take s).length = s := by rw [List.length_take]; omega
  rw [hh] at ha
  change (progOf x ((ops.take s ++ [(dm, da)]).map fun p => ZKMal.CoinLayer.rd p.2)).answer
    ((ofLoop T L₀ real).cfgAt x ρ (s + 1))
    (coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)]))) = .ok m at ha
  have hca : coinsAt x ((m₀, ρ₀) :: (ops.take s ++ [(dm, da)])) =
      (ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX) := by
    unfold coinsAt
    simp only [List.tail_cons, List.getLast?_append, List.getLast?_singleton, List.length_append,
      hts, List.length_singleton, Nat.add_sub_cancel]
    exact feedOf_rd _ _
  rw [hca, List.map_append, List.map_cons, List.map_nil] at ha
  have hloc : ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1) = (2 * j + r, 1) := by
    have e := locate_take_sum (roundsOf spec (Masked.stsOf x)) 0 (2 * j + r) 1 (by rw [hrsl]; exact hq2)
      (by rw [hrq]; omega)
    rw [Nat.zero_add] at e
    rw [← e, hs, repStartZk_eq spec _ hq2.le]
    congr 1
    omega
  have hlay : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay =
      ((startOf x (ZKMal.CoinLayer.locate (roundsOf spec (Masked.stsOf x)) 0 (s - 1)).1).toOption.map
        (·.1)).getD default := by
    show ((startOf x (repOf x (s + 1))).toOption.map (·.1)).getD default = _
    rw [show s + 1 = s - 1 + 2 by omega, repOf_add_two hsx]
  rw [progOf_snoc x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) (ZKMal.CoinLayer.rd da) hsx (q := s - 1)
    (by rw [List.length_map, hts]; omega), hloc] at ha
  rw [hloc] at hlay
  dsimp only at ha
  rcases hph : placeholder spec (Masked.stsOf x) (2 * j + r)
      (wordsAt x ((ops.take s).map fun p => ZKMal.CoinLayer.rd p.2) 0) with e' | ⟨c', t, dom, xs, hi⟩
  · rw [hph] at ha
    exact absurd ha (answer_owed_none _ rfl)
  rw [hph] at ha
  dsimp only at ha
  obtain ⟨alg, sched, hal, hsc⟩ := repState_ok ha
  obtain ⟨hts', -, -⟩ := Count.placeholder_tab hph
  obtain ⟨hm36, -, -, -⟩ := Count.sched_facts (show _root_.Flock.Zk.schedule _ = .ok (y.sch, y.tPad) from hz)
  obtain ⟨-, hdom', -, -⟩ := sizesOk_at (sizesOk_of_serves hSv s0 spec _ hsx) hq2
  obtain ⟨lay, mask, hds, hc', hdm, -⟩ := placeholder_ok hph
  have hdg : t.st.digest.size < 2 ^ 64 := by
    rw [hts']
    simp only [_root_.Flock.Setup.ofCircuit, ByteArray.size_extract]
    omega
  have hmb : t.st.m < 2 ^ 64 := by rw [hts']; exact Nat.lt_trans hm36 (by decide)
  have htm : t.st.m = (Masked.stsOf x)[j]!.m := by rw [hts', hst, hstj]; rfl
  have hcc : ∀ p, p < c'.order.size → c'.order[p]! < c'.h.size := by rw [hc']; exact ofDraws_order _ _ _
  have hnm : ((ofLoop T L₀ real).cfgAt x ρ (s + 1)).lay.nMlv = (Masked.stsOf x)[j]!.m - 6 := by
    rw [hlay]
    unfold startOf
    rw [hsx]
    dsimp only
    rw [hds]
    show lay.nMlv = _
    rw [start_nMlv hds, hst, hstj]
  obtain ⟨hsz, hall⟩ := repState_outer hcc hal hsc (by omega) hmb hdg (placeholder_cap hph) (hdm ▸ hdom') ha
  rw [htm] at hsz hall
  have hek : e < min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX := by
    have := hsz
    simp only [FlockDeploy.Coin.vals, Array.size_map, Array.size_range] at this
    omega
  have h0 : ((ZKMal.CoinLayer.rd da).vals (min (loopNr x s) _root_.Flock.Firewall.Frames.COIN_MAX))[e]! =
      (ZKMal.CoinLayer.rd da).at e := by
    rw [getElem!_pos _ e (by simp only [FlockDeploy.Coin.vals, Array.size_map, Array.size_range]; exact hek)]
    simp [FlockDeploy.Coin.vals]
  have hk := hall e he (by rw [hnm, N_INNER]; omega)
  rw [h0] at hk
  unfold Masked.outerAt
  rw [hsd, dite_eq_left_of_eq_true (eq_true hb)]
  show ((ops.map fun p => (ZKMal.CoinLayer.session loopNr T).decOf p.2).getD s default).at e ≠ 1
  rw [List.getD_eq_getElem?_getD, List.getElem?_map, hd, Option.map_some, Option.getD_some]
  show (ZKMal.CoinLayer.rd da).at e ≠ 1
  intro h1
  rw [h1] at hk
  exact absurd hk (by decide)

end LeanRelease

end Claims.ZKServer
