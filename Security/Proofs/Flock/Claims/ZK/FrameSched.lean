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

end Sched

end Claims.ZKServer
