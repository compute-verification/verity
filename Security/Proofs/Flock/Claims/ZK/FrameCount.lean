import Flock.Firewall.Skeleton
import Flock.Firewall.SessionPlan
import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The algebraic phase's coin counts

The loop requests each rep round's coins at the plan's count (`Session.repPlan`, `Loop.loopNr_succ`), and the frame
checks the device against the rep's scheduled rounds (`schedSoFar`), so a round's frame takes its coins only where the
two counts agree. This file counts the algebraic phase's (`Skeleton.rounds`): the recording channel (`Skeleton.recAsks`)
keeps each round's count as its squeeze header holds it, a squeeze of no coins is no round, and the masked zerocheck and
lincheck squeeze `replayCoins`' counts in order.

* `Tally ns Q p`: the transcript program `p` over `RecM`, wherever it returns, returns a value satisfying `Q` and has
  recorded the rounds `ns` after those it started with; with its `pure`, `bind`, `throw`, transcript steps and loops over
  a range (`tally_pure`, `tally_bind`, `tally_throw`, `tally_liftTM`, `tally_forIn`).
* `tally_squeezeP`, `zerocheckFw_tally`, `lincheckFw_tally`, `repPiop_tally`: the squeezes' counts, the zerocheck's
  and lincheck's, and the PIOP's.
-/

namespace Claims.ZKServer.Count

open _root_.Flock (V F128 Transcript TMW liftTM)
open _root_.Flock.Firewall.Skeleton (RecM recAsks)
open _root_.Flock.Firewall.Finish (squeezeP sampleP sampleVecP zerocheckFw lincheckFw repPiop)

/-- The rounds' counts the recording channel has kept. -/
def cnts (s : Array (ByteArray × ℕ) × ℕ) : List ℕ := s.1.toList.map Prod.snd

/-- **A transcript program's counts**: wherever `p` returns, its value satisfies `Q`, and the recording channel has
kept the rounds `ns` after those it started with. -/
def Tally {α : Type} (ns : List ℕ) (Q : α → Prop) (p : TMW RecM α) : Prop :=
  ∀ t s a t' s', ((p.run t).run s) = .ok ((a, t'), s') → Q a ∧ cnts s' = cnts s ++ ns

/-- A round's count: none for a squeeze of no coins. -/
def sq (n : ℕ) : List ℕ := if n = 0 then [] else [n]

theorem rec_bind_ok {α β : Type} {x : RecM α} {f : α → RecM β} {s : Array (ByteArray × ℕ) × ℕ} {b : β}
    {s' : Array (ByteArray × ℕ) × ℕ} (h : (x >>= f).run s = .ok (b, s')) :
    ∃ a s₁, x.run s = .ok (a, s₁) ∧ (f a).run s₁ = .ok (b, s') := by
  rw [StateT.run_bind] at h
  rcases hx : x.run s with _ | ⟨a, s₁⟩
  · rw [hx] at h; cases h
  · rw [hx] at h
    exact ⟨a, s₁, rfl, h⟩

theorem tally_pure {α : Type} {Q : α → Prop} {a : α} (h : Q a) : Tally [] Q (pure a : TMW RecM α) := by
  intro t s a' t' s' e
  rw [StateT.run_pure, StateT.run_pure] at e
  cases e
  exact ⟨h, by simp⟩

theorem tally_bind {α β : Type} {ns ms : List ℕ} {Q : α → Prop} {R : β → Prop} {p : TMW RecM α}
    {f : α → TMW RecM β} (hp : Tally ns Q p) (hf : ∀ a, Q a → Tally ms R (f a)) : Tally (ns ++ ms) R (p >>= f) := by
  intro t s b t' s' h
  rw [StateT.run_bind] at h
  obtain ⟨⟨a, t₁⟩, s₁, h1, h2⟩ := rec_bind_ok h
  obtain ⟨hq, hc⟩ := hp t s a t₁ s₁ h1
  obtain ⟨hr, hc'⟩ := hf a hq t₁ s₁ b t' s' h2
  exact ⟨hr, by rw [hc', hc, List.append_assoc]⟩

theorem tally_eq {α : Type} {ns ns' : List ℕ} {Q : α → Prop} {p : TMW RecM α} (h : Tally ns Q p) (e : ns = ns') :
    Tally ns' Q p :=
  e ▸ h

theorem tally_mono {α : Type} {ns : List ℕ} {Q Q' : α → Prop} {p : TMW RecM α} (h : Tally ns Q p)
    (hQ : ∀ a, Q a → Q' a) : Tally ns Q' p :=
  fun t s a t' s' e => let ⟨hq, hc⟩ := h t s a t' s' e; ⟨hQ a hq, hc⟩

theorem tally_throw {α : Type} {ns : List ℕ} {Q : α → Prop} (e : String) :
    Tally ns Q (throw e : TMW RecM α) := by
  intro t s a t' s' h
  cases h

theorem tally_liftTM {α : Type} (x : _root_.Flock.Transcript.TM α) :
    Tally [] (fun _ => True) (liftTM x : TMW RecM α) := by
  intro t s a t' s' h
  have e : ((liftTM x : TMW RecM α).run t).run s = (do let r ← x t; pure (r, s)) := rfl
  rw [e] at h
  rcases hx : x t with _ | r
  · rw [hx] at h; cases h
  · rw [hx] at h
    cases h
    exact ⟨trivial, by simp⟩

/-- **A loop over `[0:n]`'s counts**: each step from any value yields, having recorded `ms`. -/
theorem tally_forIn {σ : Type} (n : ℕ) (ms : List ℕ) (body : ℕ → σ → TMW RecM (ForInStep σ))
    (hb : ∀ i < n, ∀ u, Tally ms (fun r => ∃ v, r = .yield v) (body i u)) (u : σ) :
    Tally (List.replicate n ms).flatten (fun _ => True) (forIn [:n] u body) := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ), a + k = n →
      Tally (List.replicate k ms).flatten (fun _ => True) (forIn (List.range' a k) u body) from
    key n 0 u (by omega)
  intro k
  induction k with
  | zero =>
    intro a u _
    exact tally_pure trivial
  | succ k ih =>
    intro a u ha
    rw [List.range'_succ, List.forIn_cons, List.replicate_succ, List.flatten_cons]
    refine tally_bind (hb a (by omega) u) fun r hr => ?_
    obtain ⟨v, rfl⟩ := hr
    exact ih (a + 1) v (by omega)

theorem tally_forIn_bind {σ β : Type} {n : ℕ} {ms ms' : List ℕ} {R : β → Prop} {body : ℕ → σ → TMW RecM (ForInStep σ)}
    {u : σ} {f : σ → TMW RecM β} (hb : ∀ i < n, ∀ u, Tally ms (fun r => ∃ v, r = .yield v) (body i u))
    (hf : ∀ r, Tally ms' R (f r)) : Tally ((List.replicate n ms).flatten ++ ms') R (forIn [:n] u body >>= f) :=
  tally_bind (tally_forIn n ms body hb u) fun r _ => hf r

/-! ## The squeezes -/

/-- A squeeze's framed bytes end with its count. -/
theorem framed_count (t : Transcript) (kind : UInt8) {n : ℕ} (hn : n < 2 ^ 64) :
    (_root_.Flock.u64OfLE (Transcript.framed t kind n) ((Transcript.framed t kind n).size - 8)).toNat = n := by
  have e : Transcript.framed t kind n =
      (t.pending ++ ByteArray.mk #[4, kind, 0, 0, 0, 0, 0, 0]) ++ _root_.Flock.le64 n.toUInt64 ++ ByteArray.empty := by
    rw [ByteArray.append_empty, ByteArray.append_assoc]
    rfl
  have hs : (Transcript.framed t kind n).size - 8 = (t.pending ++ ByteArray.mk #[4, kind, 0, 0, 0, 0, 0, 0]).size := by
    rw [e, ByteArray.append_empty, ByteArray.size_append, FlockSoundness.Discharge.ZkView.le64_size]
    omega
  rw [hs, e, FlockSoundness.Discharge.ZkView.u64OfLE_mid]
  rw [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt hn]

/-- **A squeeze's count**: a squeeze of `n` coins records one round of `n`, none for `n = 0`. -/
theorem tally_squeezeP (kind : UInt8) (n : ℕ) (hn : n < 2 ^ 64) :
    Tally (sq n) (fun _ => True) (squeezeP recAsks kind n) := by
  intro t s a t' s' h
  obtain ⟨rs, k⟩ := s
  by_cases h0 : n = 0
  · subst h0
    have e : ((squeezeP recAsks kind 0).run t).run (rs, k) =
        .ok ((#[], { t with pending := Transcript.framed t kind 0 }), (rs, k)) := rfl
    rw [e] at h
    cases h
    exact ⟨trivial, by simp [sq]⟩
  · have e : ((squeezeP recAsks kind n).run t).run (rs, k) =
        .ok (((Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
          { t with pending := ByteArray.empty, k := t.k + 1,
                   rounds := t.rounds.push ⟨n, ByteArray.empty,
                     (Array.range n).map fun i => _root_.Flock.Firewall.Skeleton.coin (k + i + 1),
                     some (Transcript.framed t kind n)⟩ }),
          (rs.push (Transcript.framed t kind n,
            (_root_.Flock.u64OfLE (Transcript.framed t kind n) ((Transcript.framed t kind n).size - 8)).toNat),
           k + (_root_.Flock.u64OfLE (Transcript.framed t kind n) ((Transcript.framed t kind n).size - 8)).toNat)) := by
      simp only [squeezeP, StateT.run, h0, if_false]
      rfl
    rw [e, framed_count t kind hn] at h
    cases h
    exact ⟨trivial, by simp [cnts, sq, h0]⟩

theorem tally_sampleP : Tally [1] (fun _ => True) (sampleP recAsks) :=
  tally_eq (tally_bind (tally_squeezeP 1 1 (by decide)) fun _ _ => tally_pure trivial) (by simp [sq])

theorem tally_sampleVecP (n : ℕ) (hn : n < 2 ^ 64) : Tally (sq n) (fun _ => True) (sampleVecP recAsks n) :=
  tally_squeezeP 2 n hn

theorem tally_throw_bind {α β : Type} {ns : List ℕ} {R : β → Prop} (e : String) (f : α → TMW RecM β) :
    Tally ns R (throw e >>= f) :=
  tally_eq (tally_bind (Q := fun _ => False) (ms := []) (tally_throw e) fun _ h => h.elim) (List.append_nil ns)

/-! ## The algebraic phase -/

/-- The masked zerocheck's counts: `6`, `m - 13` (none when it is `0`), `z`'s, and one a round. -/
def zcCounts (m : ℕ) : List ℕ := [6] ++ sq (m - 13) ++ [1] ++ (List.replicate (m - 6) [1]).flatten

/-- The masked lincheck's counts: `α`'s and `β`'s, one a round, and `r'`'s. -/
def lcCounts (kLog : ℕ) : List ℕ := [1, 1] ++ (List.replicate (kLog - 6) [1]).flatten ++ [1]

theorem zerocheckFw_tally (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hm : m < 2 ^ 64) :
    Tally (zcCounts m) (fun _ => True) (zerocheckFw recAsks m digest cap root lay mk) := by
  unfold zerocheckFw
  apply tally_eq
  · apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_sampleVecP 6 (by decide)); intro _ _
    apply tally_bind (tally_sampleVecP (m - 13) (by omega)); intro _ _
    dsimp only
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind tally_sampleP; intro _ _
    apply tally_forIn_bind
    · intro i _ u
      try dsimp only
      apply tally_bind (tally_liftTM _); intro _ _
      apply tally_bind (tally_liftTM _); intro _ _
      apply tally_bind tally_sampleP; intro _ _
      exact tally_pure ⟨_, rfl⟩
    · intro r
      try dsimp only
      apply tally_bind (tally_liftTM _); intro _ _
      apply tally_bind (tally_liftTM _); intro _ _
      exact tally_pure trivial
  · simp [zcCounts, sq]

theorem lincheckFw_tally (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut) :
    Tally (lcCounts c.kLog) (fun _ => True) (lincheckFw recAsks c lay mk zc) := by
  unfold lincheckFw
  apply tally_eq
  · apply tally_bind (tally_liftTM _); intro _ _
    apply tally_bind tally_sampleP; intro _ _
    apply tally_bind tally_sampleP; intro _ _
    dsimp only
    split
    · apply tally_bind (tally_pure (Q := fun _ => True) trivial); intro _ _
      try dsimp only
      apply tally_forIn_bind
      · intro i _ u
        try dsimp only
        apply tally_bind (tally_liftTM _); intro _ _
        apply tally_bind (tally_liftTM _); intro _ _
        apply tally_bind tally_sampleP; intro _ _
        exact tally_pure ⟨_, rfl⟩
      · intro r
        try dsimp only
        apply tally_bind (tally_liftTM _); intro _ _
        apply tally_bind tally_sampleP; intro _ _
        exact tally_pure trivial
    · exact tally_throw_bind _ _
  · simp [lcCounts]

/-- **A rep's PIOP's counts**: the zerocheck's, then the lincheck's. -/
theorem repPiop_tally (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hm : st.m < 2 ^ 64) :
    Tally (zcCounts st.m ++ lcCounts st.fold.kLog) (fun _ => True) (repPiop recAsks st lay cap root extras mk) := by
  unfold repPiop
  apply tally_eq
  · apply tally_bind (zerocheckFw_tally st.m st.digest cap root lay mk hm); intro a _
    obtain ⟨zp, zc⟩ := a
    dsimp only
    apply tally_bind (lincheckFw_tally st.fold lay mk zc); intro b _
    obtain ⟨lp, lc⟩ := b
    dsimp only
    exact tally_pure trivial
  · simp

theorem tally_run' {α : Type} {ns : List ℕ} {Q : α → Prop} {p : TMW RecM α} (hp : Tally ns Q p) {t : Transcript}
    {s : Array (ByteArray × ℕ) × ℕ} {a : α} {s' : Array (ByteArray × ℕ) × ℕ}
    (h : (p.run' t).run s = .ok (a, s')) : Q a ∧ cnts s' = cnts s ++ ns := by
  rw [StateT.run'_eq, StateT.run_map] at h
  rcases hx : (p.run t).run s with _ | ⟨⟨a', t'⟩, s''⟩
  · rw [hx] at h; cases h
  · rw [hx] at h
    cases h
    exact hp t s a' t' s'' hx

/-- **The algebraic phase's scheduled rounds' counts** are the PIOP's. -/
theorem rounds_counts {st : _root_.Flock.Setup} {k0 : ℕ} {cap dom : ByteArray} {alg : Array (ByteArray × ℕ)}
    (hm : st.m < 2 ^ 64) (h : _root_.Flock.Firewall.Skeleton.rounds st k0 cap dom = .ok alg) :
    alg.toList.map Prod.snd = zcCounts st.m ++ lcCounts st.fold.kLog := by
  have hT := tally_bind (repPiop_tally st (_root_.Flock.Zk.PadLayout.new st.m st.kLog k0) cap
    (ByteArray.mk (Array.replicate 64 0)) #[] (fun _ => 0) hm) fun _ _ => tally_pure (Q := fun _ => True) (a := ()) trivial
  unfold _root_.Flock.Firewall.Skeleton.rounds at h
  simp only [bind, Except.bind] at h
  split at h
  · cases h
  · rename_i v hv
    obtain ⟨_, rs, k⟩ := v
    obtain ⟨-, hc⟩ := tally_run' (p := _) hT hv
    simp only [cnts, List.append_nil, List.map_nil, List.nil_append] at hc
    split at h
    · rename_i r0 n0 hr0
      split at h
      · cases h
      · cases h
        rw [← hc]
        dsimp only at hr0 ⊢
        obtain ⟨h0, hg⟩ := Array.getElem?_eq_some_iff.1 hr0
        have hn : (rs.toList.map Prod.snd)[0]'(by simp; omega) = n0 := by simp [hg]
        simp only [Array.set!, Array.toList_setIfInBounds, List.map_set]
        rw [← hn]
        exact List.set_getElem_self _
    · cases h

/-- The PIOP's counts are the masked replay's, a squeeze of no coins no round. -/
theorem piop_counts (m kLog : ℕ) :
    zcCounts m ++ lcCounts kLog = (_root_.Flock.Firewall.Session.replayCoins m kLog).toList.filter (· != 0) := by
  have e : ∀ n, (List.replicate n [1]).flatten = List.replicate n 1 := fun n => by
    induction n with
    | zero => rfl
    | succ n ih => rw [List.replicate_succ, List.flatten_cons, ih]; rfl
  by_cases h : m - 13 = 0
  · simp [zcCounts, lcCounts, sq, h, e, _root_.Flock.Firewall.Session.replayCoins]
  · simp [zcCounts, lcCounts, sq, h, e, _root_.Flock.Firewall.Session.replayCoins]

end Claims.ZKServer.Count
