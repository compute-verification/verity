import Proofs.Flock.Claims.ZK.FrameLoop
import Proofs.Flock.Claims.ZK.FramePass

/-!
# Two runs of the firewall's prover alike in shape (Stage B, R2)

The loop's release runs each rep's frame on the rep's scheduled rounds at placeholder inputs (a zero level 0 of the
statement's shape, zero salts for levels ≥ 1), and the deployed frame checks the device against the scheduled rounds at
the tape's (`note:20261022T0001Z-report-zkfin-stageb`). Both are the firewall's own prover (`Finish.repPiop`,
`Finish.repOpening`) run at two inputs of one shape, and the frame's state reads a round's shape alone (`FrameShape`), so
what is left is that the two runs send rounds of one shape. The prover is generic in its monad, so the comparison is
too:

* `PendAlike`, `TrAlike`, `RoundAlike`: pending bytes that are `encs` of alike frames (`Alike`), mid-frames all;
  transcripts with such pending bytes; and a squeeze's framed bytes over them, with the squeeze's header as `roundFree`
  takes it (`roundFree_alike`'s and `passes_frames`' hypotheses).
* `Lift`: a relation between two runs in a monad `M` that keeps `pure`, `bind`, `throw`, an `ok` lifted from `V`, and
  the prover's `round` at alike framed bytes (equal coins).
* `Sim`: two transcript programs over `M` alike: from alike transcripts, related results at alike transcripts. With its
  `bind`, `pure`, `throw`, `if` and loops over a range (`sim_bind`, `sim_pure`, `sim_throw`, `sim_ite`, `sim_forIn`),
  and each transcript step (`sim_label`, `sim_absorbF`, `sim_absorbSlice`, `sim_absorbBytes`, `sim_absorbK`, `sim_pow`,
  `sim_squeezeP`, `sim_sampleP`, `sim_sampleVecP`, `sim_sampleKP`).
* **The algebraic phase** (`zerocheckFw_sim`, `lincheckFw_sim`, `repPiop_sim`): at two level-0 caps of one size, the
  PIOP returns the same proof and claims and leaves alike transcripts.
* **The ring switch** (`ringSwitchP_sim`): at switched vectors of the same sizes, it returns the same switched claims.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128 F256 Transcript TMW liftTM)
open _root_.Flock.Transcript (header framed label absorbF absorbSlice absorbBytes absorbK pow)
open _root_.Flock.Firewall.Finish (Asks squeezeP sampleP sampleVecP sampleKP zerocheckFw lincheckFw repPiop ringSwitchP)
open FlockSoundness.Refine (Frame encs)

namespace Sched

/-- Every frame a mid-frame. -/
def MidAll (fs : List Frame) : Prop := ∀ f ∈ fs, midFrame f

/-- **Two pending byte strings of one shape**: `encs` of alike frames, all mid-frames. -/
def PendAlike (p p' : ByteArray) : Prop :=
  ∃ fs fs' : List Frame, p.data.toList = encs fs ∧ p'.data.toList = encs fs' ∧ List.Forall₂ Alike fs fs' ∧ MidAll fs

/-- Two transcripts with pending bytes of one shape. -/
def TrAlike (t t' : Transcript) : Prop := PendAlike t.pending t'.pending

/-- **Two rounds of one shape**: a squeeze's framed bytes over pending bytes of one shape, its kind and count as
`roundFree` takes them. -/
def RoundAlike (c c' : ByteArray) : Prop :=
  ∃ (p p' : ByteArray) (k : UInt8) (n : ℕ), PendAlike p p' ∧ c = p ++ header 4 k n ∧ c' = p' ++ header 4 k n ∧
    (k = 2 ∨ (k = 1 ∧ n = 1)) ∧ n ≠ 0 ∧ n < 2 ^ 64

theorem pendAlike_empty : PendAlike ByteArray.empty ByteArray.empty :=
  ⟨[], [], rfl, rfl, .nil, fun _ h => nomatch h⟩

theorem forall₂_snoc {fs fs' : List Frame} {f f' : Frame} (h : List.Forall₂ Alike fs fs') (hf : Alike f f') :
    List.Forall₂ Alike (fs ++ [f]) (fs' ++ [f']) := by
  induction h with
  | nil => exact .cons hf .nil
  | cons ha _ ih => exact .cons ha ih

theorem encs_snoc (fs : List Frame) (f : Frame) : encs (fs ++ [f]) = encs fs ++ f.enc := by
  simp [encs]

/-- **A frame more on each side** keeps the shape. -/
theorem pendAlike_snoc {p p' q q' : ByteArray} {f f' : Frame} (h : PendAlike p p')
    (hq : q.data.toList = p.data.toList ++ f.enc) (hq' : q'.data.toList = p'.data.toList ++ f'.enc) (hf : Alike f f')
    (hm : midFrame f) : PendAlike q q' := by
  obtain ⟨fs, fs', hp, hp', hA, hM⟩ := h
  refine ⟨fs ++ [f], fs' ++ [f'], ?_, ?_, forall₂_snoc hA hf, ?_⟩
  · rw [hq, hp, encs_snoc]
  · rw [hq', hp', encs_snoc]
  · intro g hg
    rcases List.mem_append.1 hg with hg | hg
    · exact hM g hg
    · rw [List.mem_singleton.1 hg]; exact hm

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **Two runs in `M` alike**, for the prover's channel `io`: a relation, for each relation on results, kept by `pure`,
`bind`, `throw`, an `ok` lifted from `V`, and `io.round` at rounds of one shape (with equal coins). -/
structure Lift (io : Asks M) where
  R : {α β : Type} → (α → β → Prop) → M α → M β → Prop
  pure : ∀ {α β : Type} {S : α → β → Prop} {a : α} {b : β}, S a b → R S (Pure.pure a) (Pure.pure b)
  bind : ∀ {α β γ δ : Type} {S : α → β → Prop} {T : γ → δ → Prop} {x : M α} {y : M β} {f : α → M γ}
    {g : β → M δ}, R S x y → (∀ a b, S a b → R T (f a) (g b)) → R T (x >>= f) (y >>= g)
  throw : ∀ {α β : Type} {S : α → β → Prop} (e e' : String), R S (throwThe String e : M α) (throwThe String e' : M β)
  liftOk : ∀ {α β : Type} {S : α → β → Prop} {a : α} {b : β}, S a b →
    R S (monadLift (Except.ok a : V α) : M α) (monadLift (Except.ok b : V β))
  round : ∀ {c c' : ByteArray}, RoundAlike c c' → R (· = ·) (io.round c) (io.round c')

variable {io : Asks M} (L : Lift io)

/-- **Two transcript programs alike**: from alike transcripts, results related by `S` at alike transcripts. -/
def Sim {α β : Type} (S : α → β → Prop) (p : TMW M α) (q : TMW M β) : Prop :=
  ∀ t t', TrAlike t t' → L.R (fun x y => S x.1 y.1 ∧ TrAlike x.2 y.2) (p t) (q t')

variable {L}

theorem sim_pure {α β : Type} {S : α → β → Prop} {a : α} {b : β} (h : S a b) :
    Sim L S (pure a : TMW M α) (pure b : TMW M β) :=
  fun _ _ ht => L.pure ⟨h, ht⟩

theorem sim_bind {α β γ δ : Type} {S : α → β → Prop} {T : γ → δ → Prop} {p : TMW M α} {q : TMW M β}
    {f : α → TMW M γ} {g : β → TMW M δ} (h : Sim L S p q) (hfg : ∀ a b, S a b → Sim L T (f a) (g b)) :
    Sim L T (p >>= f) (q >>= g) :=
  fun t t' ht => L.bind (h t t' ht) fun x y hxy => hfg x.1 y.1 hxy.1 x.2 y.2 hxy.2

theorem sim_throw {α β : Type} {S : α → β → Prop} (e e' : String) :
    Sim L S (throw e : TMW M α) (throw e' : TMW M β) :=
  fun _ _ _ => L.bind (S := fun _ _ => False) (L.throw e e') fun _ _ h => h.elim

theorem sim_ite {α β : Type} {S : α → β → Prop} (c : Prop) [Decidable c] {p p' : TMW M α} {q q' : TMW M β}
    (h1 : c → Sim L S p q) (h2 : ¬ c → Sim L S p' q') :
    Sim L S (if c then p else p') (if c then q else q') := by
  by_cases hc : c
  · rw [if_pos hc, if_pos hc]; exact h1 hc
  · rw [if_neg hc, if_neg hc]; exact h2 hc

theorem sim_mono {α β : Type} {S S' : α → β → Prop} {p : TMW M α} {q : TMW M β} (h : Sim L S p q)
    (hS : ∀ a b, S a b → S' a b) (hp : p >>= pure = p) (hq : q >>= pure = q) : Sim L S' p q := by
  rw [← hp, ← hq]
  exact sim_bind h fun a b hab => sim_pure (hS a b hab)

/-- **A loop over `[0:n]` alike**: if step `i` from values related by `I i` yields values related by `I (i + 1)`,
the loops end related by `I n`. -/
theorem sim_forIn {σ τ : Type} (n : ℕ) (body : ℕ → σ → TMW M (ForInStep σ)) (body' : ℕ → τ → TMW M (ForInStep τ))
    (I : ℕ → σ → τ → Prop)
    (hb : ∀ i < n, ∀ u u', I i u u' →
      Sim L (fun r r' => ∃ v v', r = .yield v ∧ r' = .yield v' ∧ I (i + 1) v v') (body i u) (body' i u'))
    {u : σ} {u' : τ} (h₀ : I 0 u u') : Sim L (I n) (forIn [:n] u body) (forIn [:n] u' body') := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (u' : τ), a + k = n → I a u u' →
      Sim L (I n) (forIn (List.range' a k) u body) (forIn (List.range' a k) u' body') from
    key n 0 u u' (by omega) h₀
  intro k
  induction k with
  | zero =>
    intro a u u' ha hI
    have e : a = n := by omega
    exact sim_pure (e ▸ hI)
  | succ k ih =>
    intro a u u' ha hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine sim_bind (hb a (by omega) u u' hI) fun r r' hr => ?_
    obtain ⟨v, v', rfl, rfl, hv⟩ := hr
    exact ih (a + 1) v v' (by omega) hv

/-! ## The transcript's steps -/

/-- A transcript step that only appends a frame to the pending bytes. -/
theorem sim_liftTM_mod {x x' : _root_.Flock.Transcript.TM Unit} {F F' : Transcript → Transcript}
    (hx : ∀ t, x t = .ok ((), F t)) (hx' : ∀ t, x' t = .ok ((), F' t))
    (hF : ∀ t t', TrAlike t t' → TrAlike (F t) (F' t')) : Sim L (· = ·) (liftTM x : TMW M Unit) (liftTM x') := by
  intro t t' ht
  show L.R _ (monadLift (x t)) (monadLift (x' t'))
  rw [hx, hx']
  exact L.liftOk ⟨rfl, hF t t' ht⟩

theorem le64_size (x : UInt64) : (_root_.Flock.le64 x).size = 8 := by
  have := congrArg List.length (FlockSoundness.Refine.le64_data x)
  simpa using this

theorem sim_label (l : String) (hl : l.toUTF8.size < 2 ^ 64) :
    Sim L (· = ·) (liftTM (label l) : TMW M Unit) (liftTM (label l)) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.label_enc _ l) (FlockSoundness.Refine.label_enc _ l) rfl hl

theorem sim_absorbF (v v' : F128) : Sim L (· = ·) (liftTM (absorbF v) : TMW M Unit) (liftTM (absorbF v')) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.absorbF_enc _ v) (FlockSoundness.Refine.absorbF_enc _ v') trivial trivial

theorem sim_absorbSlice {vs vs' : Array F128} (h : vs.size = vs'.size) (hn : vs.size < 2 ^ 64) :
    Sim L (· = ·) (liftTM (absorbSlice vs) : TMW M Unit) (liftTM (absorbSlice vs')) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.absorbSlice_enc _ vs) (FlockSoundness.Refine.absorbSlice_enc _ vs')
      (show vs.toList.length = vs'.toList.length by simpa using h) (show vs.toList.length < 2 ^ 64 by simpa using hn)

theorem sim_absorbBytes {b b' : ByteArray} (h : b.size = b'.size) (hn : b.size < 2 ^ 64) :
    Sim L (· = ·) (liftTM (absorbBytes b) : TMW M Unit) (liftTM (absorbBytes b')) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.absorbBytes_enc _ b) (FlockSoundness.Refine.absorbBytes_enc _ b') h hn

theorem sim_absorbK (v v' : F256) : Sim L (· = ·) (liftTM (absorbK v) : TMW M Unit) (liftTM (absorbK v')) :=
  sim_absorbSlice rfl (show 2 < 2 ^ 64 by decide)

theorem sim_pow (x x' : UInt64) : Sim L (· = ·) (liftTM (pow x) : TMW M Unit) (liftTM (pow x')) :=
  sim_absorbBytes (by rw [le64_size, le64_size]) (by rw [le64_size]; decide)

theorem sim_squeezeP (kind : UInt8) (n : ℕ) (hk : kind = 2 ∨ (kind = 1 ∧ n = 1)) (hn : n < 2 ^ 64) :
    Sim L (· = ·) (squeezeP io kind n) (squeezeP io kind n) := by
  intro t t' ht
  unfold squeezeP
  by_cases h0 : n = 0
  · subst h0
    have hk2 : kind = 2 := hk.resolve_right fun h => absurd h.2 (by decide)
    simp only [↓reduceIte]
    exact L.pure ⟨rfl, pendAlike_snoc ht (FlockSoundness.Refine.framed_enc t kind 0)
      (FlockSoundness.Refine.framed_enc t' kind 0) ⟨rfl, rfl⟩ ⟨hk2, rfl⟩⟩
  · simp only [h0, ↓reduceIte]
    refine L.bind (L.round ⟨t.pending, t'.pending, kind, n, ht, rfl, rfl, hk, h0, hn⟩) fun a b hab => ?_
    subst hab
    exact L.pure ⟨rfl, pendAlike_empty⟩

theorem sim_sampleP : Sim L (· = ·) (sampleP io) (sampleP io) :=
  sim_bind (sim_squeezeP 1 1 (.inr ⟨rfl, rfl⟩) (by decide)) fun _ _ h => h ▸ sim_pure rfl

theorem sim_sampleVecP (n : ℕ) (hn : n < 2 ^ 64) : Sim L (· = ·) (sampleVecP io n) (sampleVecP io n) :=
  sim_squeezeP 2 n (.inl rfl) hn

theorem sim_sampleKP : Sim L (· = ·) (sampleKP io) (sampleKP io) :=
  sim_bind (sim_squeezeP 2 2 (.inl rfl) (by decide)) fun _ _ h => h ▸ sim_pure rfl

/-! ## The algebraic phase -/

/-- **The masked zerocheck at two caps of one size** returns the same proof and leaves alike transcripts. -/
theorem zerocheckFw_sim (m : ℕ) (digest cap cap' root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hc : cap.size = cap'.size) (hd : digest.size < 2 ^ 64) (hcn : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64)
    (hm : m < 2 ^ 64) :
    Sim L (· = ·) (zerocheckFw io m digest cap root lay mk) (zerocheckFw io m digest cap' root lay mk) := by
  unfold zerocheckFw
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes rfl hd) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes hc hcn) fun _ _ _ => ?_
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes rfl hr) fun _ _ _ => ?_
  refine sim_bind (sim_sampleVecP 6 (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_sampleVecP (m - 13) (by omega)) fun rO rO' h => ?_
  subst h
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

/-- **The masked lincheck from one coin state** returns the same proof and leaves alike transcripts. -/
theorem lincheckFw_sim (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut) :
    Sim L (· = ·) (lincheckFw io c lay mk zc) (lincheckFw io c lay mk zc) := by
  unfold lincheckFw
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind sim_sampleP fun a a' h => ?_
  subst h
  refine sim_bind sim_sampleP fun b b' h => ?_
  subst h
  dsimp only
  split
  case h_2 => exact sim_bind (L := L) (S := fun _ _ => False) (sim_throw _ _) fun _ _ h => h.elim
  refine sim_bind (L := L) (S := (· = ·)) (sim_pure rfl) fun f f' h => ?_
  subst h
  refine sim_bind (sim_forIn _ _ _ (fun _ u u' => u = u') (fun i _ u u' hu => ?_) rfl) fun u u' h => ?_
  · subst hu
    refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
    refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
    refine sim_bind sim_sampleP fun a b h => ?_
    subst h
    exact sim_pure ⟨_, _, rfl, rfl, rfl⟩
  · subst h
    refine sim_bind (sim_absorbSlice rfl (by simp)) fun _ _ _ => ?_
    refine sim_bind sim_sampleP fun a b h => ?_
    subst h
    exact sim_pure rfl

/-- **A rep's PIOP at two caps of one size** returns the same proofs and claims and leaves alike transcripts. -/
theorem repPiop_sim (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap cap' root : ByteArray)
    (extras : Array _root_.Flock.Claim) (mk : ℕ → F128) (hc : cap.size = cap'.size) (hd : st.digest.size < 2 ^ 64)
    (hcn : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64) (hm : st.m < 2 ^ 64) :
    Sim L (· = ·) (repPiop io st lay cap root extras mk) (repPiop io st lay cap' root extras mk) := by
  unfold repPiop
  refine sim_bind (zerocheckFw_sim st.m st.digest cap cap' root lay mk hc hd hcn hr hm) fun a a' h => ?_
  subst h
  obtain ⟨zp, zc⟩ := a
  refine sim_bind (lincheckFw_sim st.fold lay mk zc) fun b b' h => ?_
  subst h
  obtain ⟨lp, lc⟩ := b
  exact sim_pure rfl

/-! ## The ring switch -/

/-- **The ring switch at switched vectors of the same sizes** returns the same switched claims. -/
theorem ringSwitchP_sim (sv sv' : ℕ → Array F128) (claims : Array _root_.Flock.Claim)
    (hs : ∀ k, (sv k).size = (sv' k).size) (hn : ∀ k, (sv k).size < 2 ^ 64) (hc : claims.size < 2 ^ 64) :
    Sim L (fun r r' => r.2.2 = r'.2.2) (ringSwitchP io sv claims) (ringSwitchP io sv' claims) := by
  unfold ringSwitchP
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_forIn _ _ _ (fun _ u u' => u.snd.snd = u'.snd.snd) (fun k _ u u' hu => ?_) rfl)
    fun u u' h => ?_
  · obtain ⟨ss, betas, pc⟩ := u
    obtain ⟨ss', betas', pc'⟩ := u'
    obtain rfl : pc = pc' := hu
    refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
    refine sim_bind (sim_absorbSlice (hs k) (hn k)) fun _ _ _ => ?_
    refine sim_bind (sim_sampleVecP 7 (by decide)) fun a b h => ?_
    subst h
    exact sim_pure ⟨_, _, rfl, rfl, rfl⟩
  · obtain ⟨ss, betas, pc⟩ := u
    obtain ⟨ss', betas', pc'⟩ := u'
    obtain rfl : pc = pc' := h
    refine sim_bind (sim_sampleVecP _ hc) fun a b h => ?_
    subst h
    exact sim_pure rfl

/-! ## The Ligerito prover's state -/

open _root_.Flock.Firewall.Finish (LigState LP liftT sendMsg powP challengeP oodIntroP queryP foldK splitK embedF
  roundPoly liftU addScaled maskMsg dotK ligeritoPZk repOpening Commit flat)

/-- **Two Ligerito states of one shape**: tables of one size, and the same weight (the basis and the coins'). -/
def LigAlike (s s' : LigState) : Prop := s.tab.size = s'.tab.size ∧ s.w = s'.w

variable (L) in
/-- **Two Ligerito programs alike**: from states of one shape, results related by `S` at states of one shape. -/
def SimL {α β : Type} (S : α → β → Prop) (p : LP M α) (q : LP M β) : Prop :=
  ∀ s s', LigAlike s s' → Sim L (fun x y => S x.1 y.1 ∧ LigAlike x.2 y.2) (p.run s) (q.run s')

theorem simL_pure {α β : Type} {S : α → β → Prop} {a : α} {b : β} (h : S a b) :
    SimL L S (pure a : LP M α) (pure b : LP M β) :=
  fun _ _ hs => sim_pure ⟨h, hs⟩

theorem simL_bind {α β γ δ : Type} {S : α → β → Prop} {T : γ → δ → Prop} {p : LP M α} {q : LP M β}
    {f : α → LP M γ} {g : β → LP M δ} (h : SimL L S p q) (hfg : ∀ a b, S a b → SimL L T (f a) (g b)) :
    SimL L T (p >>= f) (q >>= g) :=
  fun s s' hs => sim_bind (h s s' hs) fun x y hxy => hfg x.1 y.1 hxy.1 x.2 y.2 hxy.2

theorem simL_liftT {α β : Type} {S : α → β → Prop} {m : TMW M α} {m' : TMW M β} (h : Sim L S m m') :
    SimL L S (liftT m : LP M α) (liftT m') :=
  fun _ _ hs => sim_bind h fun _ _ hab => sim_pure ⟨hab, hs⟩

theorem simL_get : SimL L LigAlike (get : LP M LigState) get :=
  fun _ _ hs => sim_pure ⟨hs, hs⟩

theorem simL_set {a b : LigState} (h : LigAlike a b) : SimL L (· = ·) (set a : LP M PUnit) (set b) :=
  fun _ _ _ => sim_pure ⟨rfl, h⟩

theorem simL_modify {F F' : LigState → LigState} (h : ∀ s s', LigAlike s s' → LigAlike (F s) (F' s')) :
    SimL L (· = ·) (modify F : LP M PUnit) (modify F') :=
  fun s s' hs => sim_pure ⟨rfl, h s s' hs⟩

theorem simL_throw {α β : Type} {S : α → β → Prop} (e e' : String) :
    SimL L S (throw e : LP M α) (throw e' : LP M β) :=
  fun s s' _ => by
    show Sim L _ ((throw e : TMW M α) >>= fun a => pure (a, s)) ((throw e' : TMW M β) >>= fun a => pure (a, s'))
    exact sim_bind (L := L) (S := fun _ _ => False) (sim_throw e e') fun _ _ h => h.elim

theorem simL_ite {α β : Type} {S : α → β → Prop} (c : Prop) [Decidable c] {p p' : LP M α} {q q' : LP M β}
    (h1 : c → SimL L S p q) (h2 : ¬ c → SimL L S p' q') :
    SimL L S (if c then p else p') (if c then q else q') := by
  by_cases hc : c
  · rw [if_pos hc, if_pos hc]; exact h1 hc
  · rw [if_neg hc, if_neg hc]; exact h2 hc

theorem simL_forIn {σ τ : Type} (n : ℕ) (body : ℕ → σ → LP M (ForInStep σ)) (body' : ℕ → τ → LP M (ForInStep τ))
    (I : ℕ → σ → τ → Prop)
    (hb : ∀ i < n, ∀ u u', I i u u' →
      SimL L (fun r r' => ∃ v v', r = .yield v ∧ r' = .yield v' ∧ I (i + 1) v v') (body i u) (body' i u'))
    {u : σ} {u' : τ} (h₀ : I 0 u u') : SimL L (I n) (forIn [:n] u body) (forIn [:n] u' body') := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.forIn_eq_forIn_range']
  simp only [Std.Legacy.Range.size, Nat.sub_zero, Nat.add_sub_cancel, Nat.div_one]
  suffices key : ∀ k a (u : σ) (u' : τ), a + k = n → I a u u' →
      SimL L (I n) (forIn (List.range' a k) u body) (forIn (List.range' a k) u' body') from
    key n 0 u u' (by omega) h₀
  intro k
  induction k with
  | zero =>
    intro a u u' ha hI
    have e : a = n := by omega
    exact simL_pure (e ▸ hI)
  | succ k ih =>
    intro a u u' ha hI
    rw [List.range'_succ, List.forIn_cons, List.forIn_cons]
    refine simL_bind (hb a (by omega) u u' hI) fun r r' hr => ?_
    obtain ⟨v, v', rfl, rfl, hv⟩ := hr
    exact ih (a + 1) v v' (by omega) hv

/-- **A loop over two lists of one length**, its steps alike from any two elements. -/
theorem simL_forIn_list {α β σ τ : Type} (body : α → σ → LP M (ForInStep σ)) (body' : β → τ → LP M (ForInStep τ))
    (I : σ → τ → Prop)
    (hb : ∀ x y u u', I u u' →
      SimL L (fun r r' => ∃ v v', r = .yield v ∧ r' = .yield v' ∧ I v v') (body x u) (body' y u')) :
    ∀ (xs : List α) (ys : List β) (u : σ) (u' : τ), xs.length = ys.length → I u u' →
      SimL L I (forIn xs u body) (forIn ys u' body')
  | [], [], _, _, _, h => simL_pure h
  | [], _ :: _, _, _, h, _ => by simp at h
  | _ :: _, [], _, _, h, _ => by simp at h
  | x :: xs, y :: ys, u, u', h, hI => by
    rw [List.forIn_cons, List.forIn_cons]
    refine simL_bind (hb x y u u' hI) fun r r' hr => ?_
    obtain ⟨v, v', rfl, rfl, hv⟩ := hr
    exact simL_forIn_list body body' I hb xs ys v v' (by simpa using h) hv

/-- **A loop over two arrays of one size**, its steps alike from any two elements. -/
theorem simL_forIn_array {α β σ τ : Type} (body : α → σ → LP M (ForInStep σ)) (body' : β → τ → LP M (ForInStep τ))
    (I : σ → τ → Prop)
    (hb : ∀ x y u u', I u u' →
      SimL L (fun r r' => ∃ v v', r = .yield v ∧ r' = .yield v' ∧ I v v') (body x u) (body' y u'))
    (xs : Array α) (ys : Array β) {u : σ} {u' : τ} (h : xs.size = ys.size) (hI : I u u') :
    SimL L I (forIn xs u body) (forIn ys u' body') := by
  rw [← Array.forIn_toList, ← Array.forIn_toList]
  exact simL_forIn_list body body' I hb _ _ u u' (by simpa using h) hI

theorem sim_lift {α β : Type} {S : α → β → Prop} {x : M α} {y : M β} (h : L.R S x y) :
    Sim L S (StateT.lift x : TMW M α) (StateT.lift y) :=
  fun _ _ ht => L.bind h fun _ _ hab => L.pure ⟨hab, ht⟩

theorem splitK_size (f : Array F256) : (splitK f).size = 2 * f.size := by
  unfold splitK
  rw [← Array.foldl_toList]
  suffices h : ∀ (l : List F256) (acc : Array F128),
      (l.foldl (fun acc v => (acc.push v.c0).push v.c1) acc).size = acc.size + 2 * l.length by
    simpa using h f.toList #[]
  intro l
  induction l with
  | nil => intro acc; simp
  | cons a l ih => intro acc; simp only [List.foldl_cons, ih, Array.size_push, List.length_cons]; ring

theorem foldK_size (r : F256) (a : Array F256) : (foldK r a).size = a.size / 2 := by
  simp [foldK]

/-! ## The Ligerito prover's steps -/

theorem sendMsg_sim (m m' : F256 × F256) : SimL L (· = ·) (sendMsg m : LP M Unit) (sendMsg m') := by
  unfold sendMsg
  refine simL_bind (simL_modify fun s s' h => h) fun _ _ _ => ?_
  refine simL_bind (simL_liftT (sim_absorbK _ _)) fun _ _ _ => ?_
  exact simL_liftT (sim_absorbK _ _)

theorem powP_sim (nonce nonce' : ℕ → ℕ → UInt64) (kind : ℕ) :
    SimL L (· = ·) (powP nonce kind : LP M Unit) (powP nonce' kind) := by
  unfold powP
  refine simL_bind simL_get fun st st' h => ?_
  refine simL_bind (simL_set ?_) fun _ _ _ => ?_
  · rcases kind with _ | _ | kind <;> exact h
  exact simL_liftT (sim_pow _ _)

theorem challengeP_sim (split : Bool) : SimL L (· = ·) (challengeP io split : LP M F256) (challengeP io split) := by
  unfold challengeP
  refine simL_bind (simL_liftT sim_sampleKP) fun r r' h => ?_
  subst h
  refine simL_bind (simL_modify fun s s' h => ⟨by simp only [foldK_size, h.1], by rw [h.2]⟩) fun _ _ _ => ?_
  have tail : SimL L (· = ·)
      (do let st ← get; sendMsg (roundPoly st.tab st.w); pure r : LP M F256)
      (do let st ← get; sendMsg (roundPoly st.tab st.w); pure r) :=
    simL_bind simL_get fun _ _ _ => simL_bind (sendMsg_sim _ _) fun _ _ _ => simL_pure rfl
  cases split <;> simp only [Bool.false_eq_true, ↓reduceIte]
  · exact tail
  · exact simL_bind (simL_modify fun s s' h => ⟨by simp only [embedF, Array.size_map, splitK_size, h.1], by rw [h.2]⟩)
      fun _ _ _ => tail

theorem oodIntroP_sim (nonce nonce' : ℕ → ℕ → UInt64) (g g' : Array F128) (dim : ℕ) (hd : dim < 2 ^ 64) :
    SimL L (· = ·) (oodIntroP io nonce g dim : LP M Unit) (oodIntroP io nonce' g' dim) := by
  unfold oodIntroP
  refine simL_bind (simL_liftT (sim_sampleVecP dim hd)) fun z z' h => ?_
  subst h
  refine simL_bind (simL_modify fun s s' h => h) fun _ _ _ => ?_
  refine simL_bind (simL_liftT (sim_absorbF _ _)) fun _ _ _ => ?_
  refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
  refine simL_bind (powP_sim nonce nonce' 2) fun _ _ _ => ?_
  refine simL_bind (simL_liftT sim_sampleP) fun b b' h => ?_
  subst h
  exact simL_modify fun s s' h => ⟨h.1, by rw [h.2]⟩

theorem log2_succ_lt (q : ℕ) (hq : q < 2 ^ 64) : Nat.log2 (q - 1) + 1 < 2 ^ 64 := by
  rcases Nat.eq_zero_or_pos (q - 1) with h0 | h0
  · rw [h0, Nat.log2_zero]; decide
  · have h := (Nat.log2_lt h0.ne').2 (show q - 1 < 2 ^ 64 by omega)
    have : Nat.log2 (q - 1) + 1 ≤ 65 := by omega
    exact lt_of_le_of_lt this (by decide)

theorem queryP_sim (nonce nonce' : ℕ → ℕ → UInt64) (ms : _root_.Flock.MerkleScheme) (lv : _root_.Flock.Level)
    (cm cm' : _root_.Flock.Firewall.Finish.Commit) (hq : lv.q < 2 ^ 64) :
    SimL L (fun r r' => r.2 = r'.2) (queryP io nonce ms lv cm : LP M _) (queryP io nonce' ms lv cm') := by
  unfold queryP
  refine simL_bind (powP_sim _ _ 0) fun _ _ _ => ?_
  refine simL_bind (simL_liftT (sim_sampleVecP lv.q hq)) fun c c' h => ?_
  subst h
  refine simL_bind (powP_sim _ _ 1) fun _ _ _ => ?_
  refine simL_bind (simL_liftT (sim_sampleVecP _ (log2_succ_lt lv.q hq))) fun a a' h => ?_
  subst h
  exact simL_pure rfl

/-! ## The masked Ligerito and the opening -/

/-- **Two commitments of one shape**: flattened caps of one size, a `u64` length. -/
def CapAlike (cm cm' : Commit) : Prop := (flat cm.cap).size = (flat cm'.cap).size ∧ (flat cm.cap).size < 2 ^ 64

theorem sim_run {α β : Type} {S : α → β → Prop} {p : LP M α} {q : LP M β} {s s' : LigState} (h : SimL L S p q)
    (hs : LigAlike s s') : Sim L (fun x y => S x.1 y.1 ∧ LigAlike x.2 y.2) (p.run s) (q.run s') :=
  h s s' hs

theorem splitK_size' (f : Array F256) :
    (f.foldl (fun acc v => (acc.push v.c0).push v.c1) #[]).size = 2 * f.size :=
  splitK_size f

/-- **The masked Ligerito at two tables of one size and two level-0 commitments of one shape**, every other input
the prover's own or of one size, with `commit` taking tables of one size to commitments of one shape: alike runs. -/
theorem ligeritoPZk_sim (commit : _root_.Flock.Level → Array F128 → (ℕ → ByteArray) → M Commit)
    (hcom : ∀ lv (g g' : Array F128) s s', g.size = g'.size → L.R CapAlike (commit lv g s) (commit lv g' s'))
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule) (T T' : F128) (cm0 cm0' : Commit)
    (t t' b : Array F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array F128) (rep tPad : ℕ)
    (extra extra' lanePad lanePad' : ℕ → ℕ → F128) (salt salt' : ℕ → ℕ → ByteArray)
    (nonce nonce' : ℕ → ℕ → UInt64) (hcm : CapAlike cm0 cm0') (ht : t.size = t'.size) (hn : sch.n + 1 < 2 ^ 64)
    (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 64) (hp : 2 * tPad < 2 ^ 64) :
    Sim L (fun _ _ => True) (ligeritoPZk io commit ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk io commit ms sch T' cm0' t' b lay h rep tPad extra' lanePad' salt' nonce') := by
  unfold ligeritoPZk
  dsimp only
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes hcm.1 hcm.2) fun _ _ _ => ?_
  refine sim_bind (sim_label _ (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_run (S := fun _ _ => True) ?_ ⟨by simp [embedF, ht], rfl⟩) fun r r' _ => ?_
  rotate_left
  · obtain ⟨⟨_, _, _, _, _, _, _, _, _⟩, _⟩ := r
    obtain ⟨⟨_, _, _, _, _, _, _, _, _⟩, _⟩ := r'
    exact sim_pure trivial
  refine simL_bind (simL_liftT (sim_sampleVecP _ (by omega))) fun z z' hz => ?_
  subst hz
  refine simL_bind (simL_modify fun s s' h => h) fun _ _ _ => ?_
  refine simL_bind (simL_liftT (sim_absorbF _ _)) fun _ _ _ => ?_
  refine simL_bind (powP_sim nonce nonce' 2) fun _ _ _ => ?_
  refine simL_bind (simL_liftT sim_sampleP) fun b0 b0' hb => ?_
  subst hb
  refine simL_bind (simL_modify fun s s' h => ⟨h.1, by rw [h.2]⟩) fun _ _ _ => ?_
  refine simL_bind simL_get fun _ _ _ => ?_
  refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
  refine simL_bind (simL_forIn _ _ _ (fun _ u u' => u.1 = u'.1 ∧ u.2.1 = u'.2.1) (fun j _ u u' hu => ?_)
    ⟨rfl, rfl⟩) fun u u' hu => ?_
  · obtain ⟨ic, rh, zl⟩ := u
    obtain ⟨ic', rh', zl'⟩ := u'
    obtain ⟨rfl, rfl⟩ := hu
    refine simL_bind (simL_liftT sim_sampleKP) fun r r' hr => ?_
    subst hr
    refine simL_bind (simL_modify fun s s' h => ⟨by simp only [foldK_size, h.1], by rw [h.2]⟩) fun _ _ _ => ?_
    refine simL_ite _ (fun _ => ?_) (fun _ => ?_)
    · refine simL_bind simL_get fun st st' hst => ?_
      refine simL_bind (simL_liftT (sim_absorbK _ _)) fun _ _ _ => ?_
      refine simL_bind (simL_liftT (sim_absorbK _ _)) fun _ _ _ => ?_
      refine simL_bind (simL_liftT sim_sampleKP) fun a a' ha => ?_
      subst ha
      refine simL_bind (simL_liftT sim_sampleKP) fun c c' hc => ?_
      subst hc
      refine simL_ite _ (fun _ => ?_) (fun _ => ?_)
      · exact simL_bind (L := L) (S := fun _ _ => False) (simL_throw _ _) fun _ _ h => h.elim
      refine simL_bind (simL_liftT (sim_label _ (by decide))) fun _ _ _ => ?_
      refine simL_bind (simL_liftT (sim_absorbK _ _)) fun _ _ _ => ?_
      refine simL_bind (simL_liftT (sim_absorbSlice ?_ ?_)) fun _ _ _ => ?_
      · rw [splitK_size', splitK_size']; simp only [Array.size_map, Array.size_range]
      · rw [splitK_size']; simpa only [Array.size_map, Array.size_range] using hp
      refine simL_bind (simL_modify fun s s' h =>
        ⟨by simp only [embedF, Array.size_map, splitK_size, Array.size_range, hst.1], by rw [h.2]⟩) fun _ _ _ => ?_
      refine simL_bind simL_get fun _ _ _ => ?_
      refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
      exact simL_pure ⟨_, _, rfl, rfl, rfl, rfl⟩
    · refine simL_bind simL_get fun _ _ _ => ?_
      refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
      exact simL_pure ⟨_, _, rfl, rfl, rfl, rfl⟩
  obtain ⟨ic, rh, zl⟩ := u
  obtain ⟨ic', rh', zl'⟩ := u'
  obtain ⟨rfl, rfl⟩ := hu
  refine simL_bind simL_get fun s1 s1' hs1 => ?_
  refine simL_bind (simL_liftT (sim_lift (hcom _ _ _ _ _ (by simp only [Array.size_map, hs1.1]))))
    fun cm1 cm1' hcm1 => ?_
  refine simL_bind (simL_liftT (sim_absorbBytes hcm1.1 hcm1.2)) fun _ _ _ => ?_
  refine simL_bind (oodIntroP_sim nonce nonce' _ _ _ (by omega)) fun _ _ _ => ?_
  refine simL_bind (oodIntroP_sim nonce nonce' _ _ _ (by omega)) fun _ _ _ => ?_
  refine simL_bind (queryP_sim nonce nonce' ms _ cm0 cm0' (hq 0)) fun q q' hqq => ?_
  obtain ⟨op0, v0⟩ := q
  obtain ⟨op0', v0'⟩ := q'
  obtain rfl : v0 = v0' := hqq
  refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
  refine simL_bind (powP_sim nonce nonce' 2) fun _ _ _ => ?_
  refine simL_bind (simL_liftT sim_sampleP) fun bq bq' hb => ?_
  subst hb
  refine simL_bind (simL_modify fun s s' h => ⟨h.1, by rw [h.2]⟩) fun _ _ _ => ?_
  refine simL_bind (simL_forIn _ _ _ (fun _ u u' => u.1 = u'.1 ∧ u.1 < 2 ^ 64 ∧ CapAlike u.2.1 u'.2.1)
    (fun i _ u u' hu => ?_) ⟨rfl, by omega, hcm1⟩) fun u u' _ => ?_
  · obtain ⟨sd, pc, caps, rps, fin⟩ := u
    obtain ⟨sd', pc', caps', rps', fin'⟩ := u'
    obtain ⟨rfl, hsd, hpc⟩ := hu
    refine simL_bind (simL_forIn 4 _ _ (fun _ f f' => f.size = f'.size) (fun j _ f f' hf => ?_) rfl)
      fun f f' hf => ?_
    · refine simL_ite _ (fun _ => ?_) (fun _ => ?_)
      · refine simL_bind (simL_liftT sim_sampleKP) fun r r' hr => ?_
        subst hr
        refine simL_bind (simL_modify fun s s' h => ⟨by simp only [foldK_size, h.1], by rw [h.2]⟩)
          fun _ _ _ => ?_
        refine simL_bind simL_get fun d d' hd => ?_
        refine simL_bind (simL_modify fun s s' h =>
          ⟨by simp only [embedF, Array.size_map, splitK_size, h.1], by rw [h.2]⟩) fun _ _ _ => ?_
        refine simL_bind simL_get fun _ _ _ => ?_
        refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
        exact simL_pure ⟨_, _, rfl, rfl, hd.1⟩
      · refine simL_bind (challengeP_sim false) fun _ _ _ => ?_
        exact simL_pure ⟨_, _, rfl, rfl, hf⟩
    refine simL_ite _ (fun _ => ?_) (fun _ => ?_)
    · refine simL_bind (simL_liftT (sim_lift (hcom _ _ _ _ _ (by simp only [splitK_size, hf]))))
        fun cmN cmN' hN => ?_
      refine simL_bind (simL_liftT (sim_absorbBytes hN.1 hN.2)) fun _ _ _ => ?_
      refine simL_bind (oodIntroP_sim nonce nonce' _ _ _ (by omega)) fun _ _ _ => ?_
      refine simL_bind (oodIntroP_sim nonce nonce' _ _ _ (by omega)) fun _ _ _ => ?_
      refine simL_bind (queryP_sim nonce nonce' ms _ pc pc' (hq _)) fun q q' hqq => ?_
      obtain ⟨op, v⟩ := q
      obtain ⟨op', v'⟩ := q'
      obtain rfl : v = v' := hqq
      refine simL_bind (sendMsg_sim _ _) fun _ _ _ => ?_
      refine simL_bind (powP_sim nonce nonce' 2) fun _ _ _ => ?_
      refine simL_bind (simL_liftT sim_sampleP) fun bb bb' hb => ?_
      subst hb
      refine simL_bind (simL_modify fun s s' h => ⟨h.1, by rw [h.2]⟩) fun _ _ _ => ?_
      exact simL_pure ⟨_, _, rfl, rfl, rfl, by omega, hN⟩
    · refine simL_bind (simL_forIn_array _ _ (fun _ _ => True) (fun _ _ _ _ _ => ?_) _ _ ?_ trivial)
        fun _ _ _ => ?_
      · refine simL_bind (simL_liftT (sim_absorbF _ _)) fun _ _ _ => ?_
        exact simL_pure ⟨_, _, rfl, rfl, trivial⟩
      · rw [splitK_size', splitK_size', hf]
      refine simL_bind (queryP_sim nonce nonce' ms _ pc pc' (hq _)) fun q q' _ => ?_
      obtain ⟨op, v⟩ := q
      obtain ⟨op', v'⟩ := q'
      refine simL_bind (powP_sim nonce nonce' 2) fun _ _ _ => ?_
      refine simL_bind (simL_liftT sim_sampleP) fun _ _ _ => ?_
      exact simL_pure ⟨_, _, rfl, rfl, rfl, hsd, hpc⟩
  obtain ⟨_, _, _, _, _⟩ := u
  obtain ⟨_, _, _, _, _⟩ := u'
  exact simL_pure trivial

/-- **A rep's opening at two tables of one size and two level-0 commitments of one shape**: alike runs. -/
theorem repOpening_sim (commit : _root_.Flock.Level → Array F128 → (ℕ → ByteArray) → M Commit)
    (hcom : ∀ lv (g g' : Array F128) s s', g.size = g'.size → L.R CapAlike (commit lv g s) (commit lv g' s'))
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule) (lay : _root_.Flock.Zk.PadLayout) (tPad rep : ℕ)
    (g g' : Array F128) (cm0 cm0' : Commit) (claims : Array _root_.Flock.Claim) (h : Array F128)
    (extra extra' lanePad lanePad' : ℕ → ℕ → F128) (salt salt' : ℕ → ℕ → ByteArray) (hcm : CapAlike cm0 cm0')
    (hg : g.size = g'.size) (hc : claims.size < 2 ^ 64) (hn : sch.n + 1 < 2 ^ 64)
    (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 64) (hp : 2 * tPad < 2 ^ 64) :
    Sim L (fun _ _ => True) (repOpening io commit st sch lay tPad rep g cm0 claims h extra lanePad salt)
      (repOpening io commit st sch lay tPad rep g' cm0' claims h extra' lanePad' salt') := by
  unfold repOpening
  refine sim_bind (ringSwitchP_sim _ _ claims (fun _ => by simp [_root_.Flock.Firewall.Finish.switchVec])
    (fun _ => by simp [_root_.Flock.Firewall.Finish.switchVec]) hc) fun r r' hr => ?_
  obtain ⟨ss, T, sws⟩ := r
  obtain ⟨ss', T', sws'⟩ := r'
  obtain rfl : sws = sws' := hr
  refine sim_bind (ligeritoPZk_sim commit hcom _ sch T T' cm0 cm0' g g' _ lay h rep tPad extra extra' lanePad lanePad'
    salt salt' _ _ hcm hg hn hq hp) fun r r' _ => ?_
  obtain ⟨_, _, _⟩ := r
  obtain ⟨_, _, _⟩ := r'
  exact sim_pure trivial

end Sched

end Claims.ZKServer
