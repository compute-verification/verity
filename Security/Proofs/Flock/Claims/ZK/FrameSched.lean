import Proofs.Flock.Claims.ZK.FrameLoop
import Proofs.Flock.Claims.ZK.FramePass
import Proofs.Flock.Claims.ZK.FinishLevel0

/-!
# Two runs of the firewall's prover alike in shape

The loop's release runs each rep's frame on the rep's scheduled rounds at placeholder inputs (a zero level 0 of the
statement's shape, zero salts for levels ≥ 1), and the deployed frame checks the device against the scheduled rounds at
the tape's. Both are the firewall's own prover (`Finish.repPiop`,
`Finish.repOpening`) run at two inputs of one shape, and the frame's state reads a round's shape alone (`FrameShape`), so
what is left is that the two runs send rounds of one shape. The prover is generic in its monad, so the comparison is
too:

* `PendAlike`, `TrAlike`, `RoundAlike`: pending bytes that are `encs` of alike frames (`Alike`), mid-frames all and
  none the inner proof's label (each transcript step's label is the prover's own, `sim_label`);
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

/-- Every frame a mid-frame, and none the inner proof's label (`Masker.label`'s start of the inner proof). -/
def MidAll (fs : List Frame) : Prop := ∀ f ∈ fs, midFrame f ∧ f ≠ .label _root_.Flock.Zk.LABEL_INNER.toUTF8

/-- **Two pending byte strings of one shape**: `encs` of alike frames, all mid-frames and none the inner proof's label
(`MidAll`). -/
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
    (hm : midFrame f) (hi : f ≠ .label _root_.Flock.Zk.LABEL_INNER.toUTF8) : PendAlike q q' := by
  obtain ⟨fs, fs', hp, hp', hA, hM⟩ := h
  refine ⟨fs ++ [f], fs' ++ [f'], ?_, ?_, forall₂_snoc hA hf, ?_⟩
  · rw [hq, hp, encs_snoc]
  · rw [hq', hp', encs_snoc]
  · intro g hg
    rcases List.mem_append.1 hg with hg | hg
    · exact hM g hg
    · rw [List.mem_singleton.1 hg]; exact ⟨hm, hi⟩

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

theorem sim_label (l : String) (hl : l.toUTF8.size < 2 ^ 64) (hi : l.toUTF8 ≠ _root_.Flock.Zk.LABEL_INNER.toUTF8) :
    Sim L (· = ·) (liftTM (label l) : TMW M Unit) (liftTM (label l)) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.label_enc _ l) (FlockSoundness.Refine.label_enc _ l) rfl hl
      fun e => hi (Frame.label.inj e)

theorem sim_absorbF (v v' : F128) : Sim L (· = ·) (liftTM (absorbF v) : TMW M Unit) (liftTM (absorbF v')) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.absorbF_enc _ v) (FlockSoundness.Refine.absorbF_enc _ v') trivial trivial
      nofun

theorem sim_absorbSlice {vs vs' : Array F128} (h : vs.size = vs'.size) (hn : vs.size < 2 ^ 64) :
    Sim L (· = ·) (liftTM (absorbSlice vs) : TMW M Unit) (liftTM (absorbSlice vs')) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.absorbSlice_enc _ vs) (FlockSoundness.Refine.absorbSlice_enc _ vs')
      (show vs.toList.length = vs'.toList.length by simpa using h) (show vs.toList.length < 2 ^ 64 by simpa using hn)
      nofun

theorem sim_absorbBytes {b b' : ByteArray} (h : b.size = b'.size) (hn : b.size < 2 ^ 64) :
    Sim L (· = ·) (liftTM (absorbBytes b) : TMW M Unit) (liftTM (absorbBytes b')) :=
  sim_liftTM_mod (fun _ => rfl) (fun _ => rfl) fun t t' ht =>
    pendAlike_snoc ht (FlockSoundness.Refine.absorbBytes_enc _ b) (FlockSoundness.Refine.absorbBytes_enc _ b') h hn
      nofun

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
      (FlockSoundness.Refine.framed_enc t' kind 0) ⟨rfl, rfl⟩ ⟨hk2, rfl⟩ nofun⟩
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

/-- **The masked zerocheck after its first round** (`zerocheckFw` from `rOuter`'s squeeze on), which reads neither
the digest, the cap nor the pads' root. -/
def zcRest (io : Asks M) (m : ℕ) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) :
    TMW M (_root_.Flock.ZerocheckProof × _root_.Flock.ZcOut) := do
  let rOuter ← sampleVecP io (m - 13)
  let r := _root_.Flock.pinned ++ rOuter
  let r1ab' := (Array.range 64).map fun i => mk (lay.r1ab i)
  let r1c' := (Array.range 64).map fun i => mk (lay.r1c i)
  liftTM (absorbSlice r1ab')
  liftTM (absorbSlice r1c')
  let z ← sampleP io
  let mut rounds : Array (F128 × F128) := #[]
  let mut rho : Array F128 := #[]
  for i in [0:m - 6] do
    let g1' := mk (lay.ml i 0)
    let ginf' := mk (lay.ml i 1)
    liftTM (absorbF g1')
    liftTM (absorbF ginf')
    let rr ← sampleP io
    rho := rho.push rr
    rounds := rounds.push (g1', ginf')
  liftTM (absorbF (mk lay.fa))
  liftTM (absorbF (mk lay.fb))
  return (⟨r1ab', r1c', rounds, mk lay.fa, mk lay.fb, mk lay.fc⟩, { z, rho, rRest := r, vA := 0, vB := 0, vC := 0 })

theorem zerocheckFw_split (io : Asks M) (m : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) :
    zerocheckFw io m digest cap root lay mk = (do
      liftTM (label "flock-r1cs-v0")
      liftTM (absorbBytes digest)
      liftTM (absorbBytes cap)
      liftTM (label "flock-zerocheck-v0")
      liftTM (label _root_.Flock.Zk.LABEL_PADS)
      liftTM (absorbBytes root)
      let _ ← sampleVecP io 6
      zcRest io m lay mk) :=
  rfl

/-- The masked zerocheck after its first round returns the same proof from alike transcripts. -/
theorem zcRest_sim (m : ℕ) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128) (hm : m < 2 ^ 64) :
    Sim L (· = ·) (zcRest io m lay mk) (zcRest io m lay mk) := by
  unfold zcRest
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

/-- **The masked zerocheck at two caps of one size** returns the same proof and leaves alike transcripts. -/
theorem zerocheckFw_sim (m : ℕ) (digest cap cap' root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (hc : cap.size = cap'.size) (hd : digest.size < 2 ^ 64) (hcn : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64)
    (hm : m < 2 ^ 64) :
    Sim L (· = ·) (zerocheckFw io m digest cap root lay mk) (zerocheckFw io m digest cap' root lay mk) := by
  rw [zerocheckFw_split, zerocheckFw_split]
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes rfl hd) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes hc hcn) fun _ _ _ => ?_
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes rfl hr) fun _ _ _ => ?_
  refine sim_bind (sim_sampleVecP 6 (by decide)) fun _ _ _ => ?_
  exact zcRest_sim m lay mk hm

/-- **The masked lincheck from one coin state** returns the same proof and leaves alike transcripts. -/
theorem lincheckFw_sim (c : _root_.Flock.CircuitFold) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    (zc : _root_.Flock.ZcOut) :
    Sim L (· = ·) (lincheckFw io c lay mk zc) (lincheckFw io c lay mk zc) := by
  unfold lincheckFw
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
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
    Sim L (fun r r' => r = r' ∧ r.2.2.size = extras.size + 2) (repPiop io st lay cap root extras mk)
      (repPiop io st lay cap' root extras mk) := by
  unfold repPiop
  refine sim_bind (zerocheckFw_sim st.m st.digest cap cap' root lay mk hc hd hcn hr hm) fun a a' h => ?_
  subst h
  obtain ⟨zp, zc⟩ := a
  refine sim_bind (lincheckFw_sim st.fold lay mk zc) fun b b' h => ?_
  subst h
  obtain ⟨lp, lc⟩ := b
  refine sim_pure ⟨rfl, ?_⟩
  show (_root_.Flock.abcClaims st.m st.kLog zc lc ++ extras).size = extras.size + 2
  have : (_root_.Flock.abcClaims st.m st.kLog zc lc).size = 2 := rfl
  rw [Array.size_append, this]
  omega

/-! ## The ring switch -/

/-- **The ring switch at switched vectors of the same sizes** returns the same switched claims. -/
theorem ringSwitchP_sim (sv sv' : ℕ → Array F128) (claims : Array _root_.Flock.Claim)
    (hs : ∀ k, (sv k).size = (sv' k).size) (hn : ∀ k, (sv k).size < 2 ^ 64) (hc : claims.size < 2 ^ 64) :
    Sim L (fun r r' => r.2.2 = r'.2.2) (ringSwitchP io sv claims) (ringSwitchP io sv' claims) := by
  unfold ringSwitchP
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_forIn _ _ _ (fun _ u u' => u.snd.snd = u'.snd.snd) (fun k _ u u' hu => ?_) rfl)
    fun u u' h => ?_
  · obtain ⟨ss, betas, pc⟩ := u
    obtain ⟨ss', betas', pc'⟩ := u'
    obtain rfl : pc = pc' := hu
    refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
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
    (ms : _root_.Flock.MerkleScheme) (sch : _root_.Flock.Schedule)
    (hcom : ∀ i (g g' : Array F128) s s', g.size = g'.size →
      L.R CapAlike (commit (sch.levels.getD i default) g s) (commit (sch.levels.getD i default) g' s'))
    (T T' : F128) (cm0 cm0' : Commit)
    (t t' b : Array F128) (lay : _root_.Flock.Zk.PadLayout) (h : Array F128) (rep tPad : ℕ)
    (extra extra' lanePad lanePad' : ℕ → ℕ → F128) (salt salt' : ℕ → ℕ → ByteArray)
    (nonce nonce' : ℕ → ℕ → UInt64) (hcm : CapAlike cm0 cm0') (ht : t.size = t'.size) (hn : sch.n + 1 < 2 ^ 64)
    (hq : ∀ i, (sch.levels.getD i default).q < 2 ^ 64) (hp : 2 * tPad < 2 ^ 64) :
    Sim L (fun _ _ => True) (ligeritoPZk io commit ms sch T cm0 t b lay h rep tPad extra lanePad salt nonce)
      (ligeritoPZk io commit ms sch T' cm0' t' b lay h rep tPad extra' lanePad' salt' nonce') := by
  unfold ligeritoPZk
  dsimp only
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
  refine sim_bind (sim_absorbF _ _) fun _ _ _ => ?_
  refine sim_bind (sim_absorbBytes hcm.1 hcm.2) fun _ _ _ => ?_
  refine sim_bind (sim_label _ (by decide) (by decide)) fun _ _ _ => ?_
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
      refine simL_bind (simL_liftT (sim_label _ (by decide) (by decide))) fun _ _ _ => ?_
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
    (st : _root_.Flock.Setup) (sch : _root_.Flock.Schedule)
    (hcom : ∀ i (g g' : Array F128) s s', g.size = g'.size →
      L.R CapAlike (commit (sch.levels.getD i default) g s) (commit (sch.levels.getD i default) g' s'))
    (lay : _root_.Flock.Zk.PadLayout) (tPad rep : ℕ)
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
  refine sim_bind (ligeritoPZk_sim commit _ sch hcom T T' cm0 cm0' g g' _ lay h rep tPad extra extra' lanePad lanePad'
    salt salt' _ _ hcm hg hn hq hp) fun r r' _ => ?_
  obtain ⟨_, _, _⟩ := r
  obtain ⟨_, _, _⟩ := r'
  exact sim_pure trivial

/-! ## Levels ≥ 1's commitments -/

theorem flat_size (A : Array ByteArray) (h : ∀ b ∈ A, b.size = 64) : (flat A).size = 64 * A.size := by
  unfold flat
  rw [← Array.foldl_toList]
  have key : ∀ (l : List ByteArray) (acc : ByteArray), (∀ b ∈ l, b.size = 64) →
      (l.foldl (· ++ ·) acc).size = acc.size + 64 * l.length := by
    intro l
    induction l with
    | nil => intro acc _; simp
    | cons a l ih =>
      intro acc hl
      rw [List.foldl_cons, ih _ fun b hb => hl b (by simp [hb]), ByteArray.size_append, hl a (by simp),
        List.length_cons]
      ring
  rw [key _ _ fun b hb => h b (by simpa using hb)]
  simp

theorem layer_size (d : ℕ) (lf : ℕ → ByteArray) : ∀ h, (_root_.Flock.Firewall.Link.layer d lf h).size = 2 ^ d / 2 ^ h
  | 0 => by simp [_root_.Flock.Firewall.Link.layer]
  | h + 1 => by
    simp only [_root_.Flock.Firewall.Link.layer, Array.size_map, Array.size_range, layer_size d lf h]
    rw [Nat.div_div_eq_div_mul, pow_succ]

theorem layer_mem (d : ℕ) (lf : ℕ → ByteArray) (hl : ∀ q, (lf q).size = 64) :
    ∀ h, ∀ b ∈ _root_.Flock.Firewall.Link.layer d lf h, b.size = 64
  | 0 => by
    intro b hb
    simp only [_root_.Flock.Firewall.Link.layer, Array.mem_map] at hb
    obtain ⟨q, -, rfl⟩ := hb
    exact hl q
  | h + 1 => by
    intro b hb
    simp only [_root_.Flock.Firewall.Link.layer, Array.mem_map] at hb
    obtain ⟨q, -, rfl⟩ := hb
    exact _root_.Flock.Sha512.hash_size _

/-- **A level's commitment has a cap of one shape** whatever its table and salts: `2^cap` digests. -/
theorem commitRows_capAlike (lv : _root_.Flock.Level) (rows rows' : Array (Array F128)) (s s' : ℕ → ByteArray)
    (hd : 64 * 2 ^ lv.d < 2 ^ 64) :
    CapAlike (_root_.Flock.Firewall.Finish.commitRows lv rows s) (_root_.Flock.Firewall.Finish.commitRows lv rows' s') := by
  have key : ∀ (rows : Array (Array F128)) (s : ℕ → ByteArray),
      (flat (_root_.Flock.Firewall.Finish.commitRows lv rows s).cap).size = 64 * (2 ^ lv.d / 2 ^ (lv.d - lv.cap)) := by
    intro rows s
    rw [(_root_.Flock.Firewall.Finish.commitRows_tree lv _ s).2,
      flat_size _ (layer_mem _ (fun q => _root_.Flock.MerkleScheme.hm96Sha512.leaf
        rows[q]! (_root_.Flock.Firewall.Finish.fitBytes 192 (s q)))
        (fun _ => _root_.Flock.Hm96.Default512.leaf_size _ _) _), layer_size]
  refine ⟨by rw [key, key], ?_⟩
  rw [key]
  exact lt_of_le_of_lt (Nat.mul_le_mul_left 64 (Nat.div_le_self _ _)) hd

theorem commitFw_capAlike (lv : _root_.Flock.Level) (g g' : Array F128) (s s' : ℕ → ByteArray)
    (hd : 64 * 2 ^ lv.d < 2 ^ 64) :
    CapAlike (_root_.Flock.Firewall.Finish.commitFw lv g s) (_root_.Flock.Firewall.Finish.commitFw lv g' s') :=
  commitRows_capAlike lv _ _ s s' hd

/-! ## Rounds of one shape, with their coin counts -/

/-- **Two scheduled rounds of one shape**: their framed bytes alike (`RoundAlike`) at one coin count, the count their
squeeze headers hold. -/
def RoundAt (r r' : ByteArray × ℕ) : Prop :=
  r.2 = r'.2 ∧ ∃ (p p' : ByteArray) (k : UInt8), PendAlike p p' ∧ r.1 = p ++ header 4 k r.2 ∧
    r'.1 = p' ++ header 4 k r.2 ∧ (k = 2 ∨ (k = 1 ∧ r.2 = 1)) ∧ r.2 ≠ 0 ∧ r.2 < 2 ^ 64

/-- Two schedules of one shape, round by round. -/
def RoundsAt (rs rs' : Array (ByteArray × ℕ)) : Prop := List.Forall₂ RoundAt rs.toList rs'.toList

theorem forall₂_push {α β : Type} {R : α → β → Prop} {l : List α} {l' : List β} {a : α} {b : β}
    (h : List.Forall₂ R l l') (hab : R a b) : List.Forall₂ R (l ++ [a]) (l' ++ [b]) := by
  induction h with
  | nil => exact .cons hab .nil
  | cons h1 _ ih => exact .cons h1 ih

/-- A squeeze's framed bytes hold its coin count in their last eight bytes. -/
theorem count_header (p : ByteArray) (k : UInt8) (n : ℕ) (hn : n < 2 ^ 64) :
    (_root_.Flock.u64OfLE (p ++ header 4 k n) ((p ++ header 4 k n).size - 8)).toNat = n := by
  have hL : (p ++ header 4 k n).data.toList = p.data.toList ++ (Frame.sq k n).enc := by
    simp [Frame.enc, Frame.op, Frame.kind, Frame.len, Frame.pay]
  have hd : (p ++ header 4 k n).data.toList.drop p.data.toList.length = (Frame.sq k n).enc ++ [] := by
    rw [hL, List.drop_left, List.append_nil]
  have h := (header_at _ _ _ _ hd hn).2.2
  have hs : (p ++ header 4 k n).size - 8 = p.data.toList.length + 8 := by
    have := congrArg List.length hL
    rw [List.length_append, Frame.enc_length] at this
    simp only [Frame.payLen, Array.length_toList] at this
    simp only [ByteArray.size, Array.length_toList]
    omega
  rw [hs]
  have e : ({ data := (p ++ header 4 k n).data.toList.toArray } : ByteArray) = p ++ header 4 k n := by
    rw [Array.toArray_toList]
  rw [e] at h
  exact h

/-- Alike framed bytes, at their coin count. -/
theorem roundAt_of_alike {c c' : ByteArray} (h : RoundAlike c c') :
    ∃ n, (_root_.Flock.u64OfLE c (c.size - 8)).toNat = n ∧ (_root_.Flock.u64OfLE c' (c'.size - 8)).toNat = n ∧
      RoundAt (c, n) (c', n) := by
  obtain ⟨p, p', k, n, hp, rfl, rfl, hk, h0, hn⟩ := h
  exact ⟨n, count_header p k n hn, count_header p' k n hn, rfl, p, p', k, hp, rfl, rfl, hk, h0, hn⟩

/-! ## The opening's resumptions -/

open _root_.Flock.Firewall.Drive (Step OpenM stepAsks openStep)

/-- **Two resumptions of one shape**: both end at related results, both stop, or both ask for rounds of one shape at
one count and go on alike from the same coins. -/
inductive StepRel {α β : Type} (S : α → β → Prop) : Step α → Step β → Prop
  | done {a : α} {b : β} : S a b → StepRel S (.done a) (.done b)
  | fail (e e' : String) : StepRel S (.fail e) (.fail e')
  | ask {c c' : ByteArray} {n : ℕ} {k : Array F128 → Step α} {k' : Array F128 → Step β} :
      RoundAt (c, n) (c', n) → (∀ cs, StepRel S (k cs) (k' cs)) → StepRel S (.ask c n k) (.ask c' n k')

theorem stepRel_bind {α β γ δ : Type} {S : α → β → Prop} {T : γ → δ → Prop} {p : Step α} {q : Step β}
    {f : α → Step γ} {g : β → Step δ} (h : StepRel S p q) (hfg : ∀ a b, S a b → StepRel T (f a) (g b)) :
    StepRel T (p.bind f) (q.bind g) := by
  induction h with
  | done hab => exact hfg _ _ hab
  | fail e e' => exact .fail e e'
  | ask hr _ ih => exact .ask hr fun cs => ih cs

/-- **Alike resumptions fed the same coins** end alike, having sent rounds of one shape. -/
theorem stepRel_feed {α β : Type} {S : α → β → Prop} :
    ∀ {p : Step α} {q : Step β} (cs : List (Array F128)) {rs rs' : Array (ByteArray × ℕ)},
      StepRel S p q → RoundsAt rs rs' →
      StepRel S (p.feed cs rs).1 (q.feed cs rs').1 ∧ RoundsAt (p.feed cs rs).2 (q.feed cs rs').2
  | _, _, [], _, _, h, hr => by
    cases h with
    | done hab => exact ⟨.done hab, hr⟩
    | fail e e' => exact ⟨.fail e e', hr⟩
    | ask hc hk => exact ⟨.ask hc hk, hr⟩
  | _, _, x :: cs, rs, rs', h, hr => by
    cases h with
    | done hab => exact ⟨.done hab, hr⟩
    | fail e e' => exact ⟨.fail e e', hr⟩
    | ask hc hk =>
      exact stepRel_feed cs (hk x) (by unfold RoundsAt; rw [Array.toList_push, Array.toList_push]; exact forall₂_push hr hc)

/-- The opening's runs, related: at one state, results related by `S` at one state. -/
def ORel {α β : Type} (S : α → β → Prop) (x : OpenM α) (y : OpenM β) : Prop :=
  ∀ s, StepRel (fun a b => S a.1 b.1 ∧ a.2 = b.2) (x s) (y s)

theorem orel_pure {α β : Type} {S : α → β → Prop} {a : α} {b : β} (h : S a b) :
    ORel S (pure a : OpenM α) (pure b : OpenM β) :=
  fun _ => .done ⟨h, rfl⟩

theorem orel_bind {α β γ δ : Type} {S : α → β → Prop} {T : γ → δ → Prop} {x : OpenM α} {y : OpenM β}
    {f : α → OpenM γ} {g : β → OpenM δ} (hx : ORel S x y) (hf : ∀ a b, S a b → ORel T (f a) (g b)) :
    ORel T (x >>= f) (y >>= g) := by
  intro s
  refine stepRel_bind (p := x s) (q := y s) (hx s) fun p q hpq => ?_
  obtain ⟨a, s1⟩ := p
  obtain ⟨b, s2⟩ := q
  obtain ⟨hab, rfl⟩ := hpq
  exact hf a b hab s1

theorem orel_throw {α β : Type} {S : α → β → Prop} (e e' : String) :
    ORel S (throwThe String e : OpenM α) (throwThe String e' : OpenM β) :=
  fun _ => .fail e e'

theorem orel_ite {α β : Type} {S : α → β → Prop} (c : Prop) [Decidable c] {p p' : OpenM α} {q q' : OpenM β}
    (h1 : c → ORel S p q) (h2 : ¬ c → ORel S p' q') :
    ORel S (if c then p else p') (if c then q else q') := by
  by_cases hc : c
  · rw [if_pos hc, if_pos hc]; exact h1 hc
  · rw [if_neg hc, if_neg hc]; exact h2 hc

theorem orel_get : ORel (· = ·) (get : OpenM (Array (Array F128))) get :=
  fun _ => .done ⟨rfl, rfl⟩

theorem orel_set (s : Array (Array F128)) : ORel (· = ·) (set s : OpenM PUnit) (set s) :=
  fun _ => .done ⟨rfl, rfl⟩

theorem orel_ask {c c' : ByteArray} {n : ℕ} (h : RoundAt (c, n) (c', n)) :
    ORel (· = ·) (StateT.lift (Step.ask c n .done) : OpenM (Array F128)) (StateT.lift (Step.ask c' n .done)) :=
  fun _ => .ask h fun _ => .done ⟨rfl, rfl⟩

/-- **The opening's runs as a `Lift`**: at one state of coins taken, `stepAsks` asks for rounds of one shape. -/
def openLift (coins : Array (Array F128)) (live : Bool) : Lift (M := OpenM) (stepAsks coins live) where
  R := ORel
  pure := orel_pure
  bind := orel_bind
  throw := orel_throw
  liftOk h := fun _ => .done ⟨h, rfl⟩
  round := by
    intro c c' hc
    obtain ⟨n, hn, hn', hr⟩ := roundAt_of_alike hc
    unfold stepAsks
    dsimp only
    rw [hn, hn']
    refine orel_bind orel_get fun got got' hg => ?_
    subst hg
    rcases coins[got.size]? with _ | cs <;> cases live <;> dsimp only
    · exact orel_bind (S := fun _ _ => False) (orel_throw _ _) fun _ _ h => h.elim
    all_goals
      refine orel_bind (S := (· = ·)) ?_ fun cs cs' hcs => ?_
      rotate_left
      · subst hcs
        refine orel_ite _ (fun _ => orel_bind (S := fun _ _ => False) (orel_throw _ _) fun _ _ h => h.elim)
          fun _ => ?_
        exact orel_bind (orel_set _) fun _ _ _ => orel_pure rfl
      first | exact orel_ask hr | exact orel_pure rfl

theorem sim_getThe (coins : Array (Array F128)) (live : Bool) :
    Sim (openLift coins live) (· = ·) (getThe (Array (Array F128)) : TMW OpenM _) (getThe _) :=
  fun _ _ ht _ => .done ⟨⟨rfl, ht⟩, rfl⟩

/-- Every opening lift relates programs by `ORel`, whatever its coins are for. -/
theorem sim_relift (coins : Array (Array F128)) (live live' : Bool) {α β : Type} {S : α → β → Prop}
    {p : TMW OpenM α} {q : TMW OpenM β} (h : Sim (openLift coins live) S p q) : Sim (openLift coins live') S p q :=
  h

/-- A stream's transcript at its start, its domain a `u64` length. -/
theorem trAlike_init (dom : ByteArray) (hd : dom.size < 2 ^ 64) :
    TrAlike (Transcript.init #[] dom) (Transcript.init #[] dom) :=
  ⟨[.dom dom], [.dom dom], by simp [encs, FlockSoundness.Refine.init_enc],
    by simp [encs, FlockSoundness.Refine.init_enc], .cons rfl .nil, fun f hf => by
      simp only [List.mem_singleton] at hf
      subst hf
      exact ⟨hd, nofun⟩⟩

/-- **Two opening programs alike, run** from alike transcripts at one state: alike resumptions. -/
theorem stepRel_run {coins : Array (Array F128)} {live : Bool} {α β : Type} {S : α → β → Prop}
    {p : TMW OpenM α} {q : TMW OpenM β} (h : Sim (openLift coins live) S p q) {t t' : Transcript}
    (ht : TrAlike t t') (s : Array (Array F128)) :
    StepRel (fun _ _ => True) ((p.run' t).run' s) ((q.run' t').run' s) := by
  refine stepRel_bind (T := fun _ _ => True) (stepRel_bind (T := fun _ _ => True) (h t t' ht s) fun _ _ _ => ?_)
    fun _ _ _ => .done trivial
  exact .done trivial

/-- **A rep's opening at two tables of one shape**, the same statement, schedule and `t_pad`, packed tables of one
size and level-0 commitments of one shape, on the same coins: alike resumptions, whatever the salts for levels ≥ 1. -/
theorem openStep_alike (t t' : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray) (xs : Array (Array F128)) (hi hi' : Array ByteArray)
    (coins : Array (Array F128)) (hst : t.st = t'.st) (hsch : t.sch = t'.sch) (htp : t.tPad = t'.tPad)
    (hg : t.g.size = t'.g.size) (hcm : CapAlike t.cm0 t'.cm0) (hdom : dom.size < 2 ^ 64)
    (hdg : t.st.digest.size < 2 ^ 64) (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64)
    (hn : t.sch.n + 1 < 2 ^ 64) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 64) (hp : 2 * t.tPad < 2 ^ 64)
    (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64) :
    StepRel (fun _ _ => True) (openStep t rep dom xs hi coins) (openStep t' rep dom xs hi' coins) := by
  obtain ⟨st, sch, tPad, g, extra, lanePad, cm0⟩ := t
  obtain ⟨st', sch', tPad', g', extra', lanePad', cm0'⟩ := t'
  obtain rfl : st = st' := hst
  obtain rfl : sch = sch' := hsch
  obtain rfl : tPad = tPad' := htp
  unfold openStep
  dsimp only
  refine stepRel_run (coins := coins) (live := true) (S := fun _ _ => True) ?_ (trAlike_init dom hdom) _
  have h1 := repPiop_sim (L := openLift coins false) st (_root_.Flock.Zk.PadLayout.new st.m st.kLog sch.k0)
    (flat cm0.cap) (flat cm0'.cap) ByteArray.empty (_root_.Flock.Firewall.Finish.claimsOf xs) (fun _ => 0) hcm.1 hdg
    hcm.2 (by decide) hm
  refine sim_bind (sim_relift coins false true h1) fun r r' h => ?_
  obtain ⟨rfl, hcs⟩ := h
  have hc : r.2.2.size < 2 ^ 64 := by
    simp only [_root_.Flock.Firewall.Finish.claimsOf, Array.size_map] at hcs
    omega
  refine sim_bind (sim_getThe coins true) fun k k' h => ?_
  subst h
  refine sim_ite _ (fun _ => ?_) (fun _ => ?_)
  · first
    | exact sim_throw _ _
    | exact sim_bind (S := fun _ _ => False) (sim_throw _ _) fun _ _ h => h.elim
  exact sim_bind (S := fun _ _ => True) (repOpening_sim (L := openLift coins true)
    (fun lv g salt => pure (_root_.Flock.Firewall.Finish.commitFw lv g salt)) st sch
    (fun i g g' s s' _ => orel_pure (commitFw_capAlike _ g g' s s' (hd i))) _ tPad rep g g' cm0 cm0' r.2.2 _ extra
    extra' lanePad lanePad' _ _ hcm hg hc hn hq hp) fun _ _ _ => sim_pure trivial

/-! ## The algebraic phase's schedule -/

/-- **Two `V` results alike**: both refused, or both values, related by `S`. -/
def RelV {α β : Type} (S : α → β → Prop) : V α → V β → Prop
  | .ok a, .ok b => S a b
  | .error _, .error _ => True
  | _, _ => False

theorem relV_bind {α β γ δ : Type} {S : α → β → Prop} {T : γ → δ → Prop} {x : V α} {y : V β} {f : α → V γ}
    {g : β → V δ} (h : RelV S x y) (hfg : ∀ a b, S a b → RelV T (f a) (g b)) : RelV T (x >>= f) (y >>= g) := by
  cases x <;> cases y
  · exact True.intro
  · exact (h : False).elim
  · exact (h : False).elim
  · exact hfg _ _ h

theorem relV_mono {α β : Type} {S T : α → β → Prop} {x : V α} {y : V β} (h : RelV S x y)
    (hST : ∀ a b, S a b → T a b) : RelV T x y := by
  cases x <;> cases y
  · exact True.intro
  · exact (h : False).elim
  · exact (h : False).elim
  · exact hST _ _ h

theorem stateT_bind_apply {σ α β : Type} (x : StateT σ V α) (f : α → StateT σ V β) (s : σ) :
    (x >>= f) s = x s >>= fun p => f p.1 p.2 :=
  rfl

theorem tmw_bind_apply {σ α β : Type} (x : TMW (StateT σ V) α) (f : α → TMW (StateT σ V) β) (t : Transcript)
    (s : σ) : (x >>= f) t s = x t s >>= fun p => f p.1.1 p.1.2 p.2 :=
  rfl

theorem tmw_run'_run {σ α : Type} (x : TMW (StateT σ V) α) (t : Transcript) (s : σ) :
    (x.run' t).run s = x t s >>= fun p => pure (p.1.1, p.2) :=
  rfl

section Skel

open _root_.Flock.Firewall.Skeleton (RecM recAsks padsFrames)

/-- **Two recordings alike, pinned at their first rounds** `h` and `h'`: as many coins drawn, rounds of one shape. -/
def RecP (h h' : ByteArray × ℕ) (s s' : Array (ByteArray × ℕ) × ℕ) : Prop :=
  s.2 = s'.2 ∧ RoundsAt s.1 s'.1 ∧ s.1[0]? = some h ∧ s'.1[0]? = some h'

/-- Two recording programs alike from recordings pinned at `h` and `h'`. -/
def RecR (h h' : ByteArray × ℕ) {α β : Type} (S : α → β → Prop) (x : RecM α) (y : RecM β) : Prop :=
  ∀ s s', RecP h h' s s' → RelV (fun a b => S a.1 b.1 ∧ RecP h h' a.2 b.2) (x s) (y s')

theorem getElem?_push_zero {α : Type} {a : Array α} {x y : α} (h : a[0]? = some y) : (a.push x)[0]? = some y := by
  obtain ⟨hs, _⟩ := Array.getElem?_eq_some_iff.1 h
  rw [Array.getElem?_push, if_neg (Nat.ne_of_lt hs)]
  exact h

/-- **The recording's lift**, from recordings pinned at their first rounds `h` and `h'`. -/
def recLift (h h' : ByteArray × ℕ) : Lift (M := RecM) recAsks where
  R := RecR h h'
  pure hab := fun _ _ hs => And.intro hab hs
  bind hxy hfg := fun s s' hs => by
    rw [stateT_bind_apply, stateT_bind_apply]
    exact relV_bind (hxy s s' hs) fun a b hab => hfg a.1 b.1 hab.1 a.2 b.2 hab.2
  throw _ _ := fun _ _ _ => True.intro
  liftOk hab := fun _ _ hs => And.intro hab hs
  round := by
    intro c c' hc
    obtain ⟨n, hn, hn', hr⟩ := roundAt_of_alike hc
    rintro ⟨rs, k⟩ ⟨rs', k'⟩ ⟨hk, hR, h0, h0'⟩
    dsimp only at hk
    subst hk
    unfold recAsks
    dsimp only
    rw [hn, hn']
    refine And.intro rfl (And.intro rfl (And.intro ?_ (And.intro (getElem?_push_zero h0) (getElem?_push_zero h0'))))
    unfold RoundsAt
    rw [Array.toList_push, Array.toList_push]
    exact forall₂_push hR hr

/-- What the masked zerocheck frames before its first squeeze, up to its label, at `dom`, `digest` and `cap`. -/
def zcPre (digest cap dom : ByteArray) : ByteArray :=
  (Transcript.init #[] dom).pending ++ header 2 0 "flock-r1cs-v0".toUTF8.size ++
    _root_.Flock.Transcript.padded "flock-r1cs-v0".toUTF8 ++ header 5 0 digest.size ++
    _root_.Flock.Transcript.padded digest ++ header 5 0 cap.size ++ _root_.Flock.Transcript.padded cap ++
    header 2 0 "flock-zerocheck-v0".toUTF8.size ++ _root_.Flock.Transcript.padded "flock-zerocheck-v0".toUTF8

/-- What the masked zerocheck frames before its first squeeze, the pads' label and root `root` included. -/
def zcPend (digest cap root dom : ByteArray) : ByteArray :=
  zcPre digest cap dom ++ header 2 0 _root_.Flock.Zk.LABEL_PADS.toUTF8.size ++
    _root_.Flock.Transcript.padded _root_.Flock.Zk.LABEL_PADS.toUTF8 ++ header 5 0 root.size ++
    _root_.Flock.Transcript.padded root

/-- The transcript after the masked zerocheck's first round `c`, the recording's coins. -/
def skelT8 (c : ByteArray) : Transcript :=
  { rounds := (#[] : Array _root_.Flock.Round).push
      ⟨6, ByteArray.empty, (Array.range 6).map fun i => _root_.Flock.Firewall.Skeleton.coin (0 + i + 1), some c⟩,
    k := 0 + 1, pending := ByteArray.empty }

/-- The recording after the masked zerocheck's first round `c`. -/
def skelS1 (c : ByteArray) : Array (ByteArray × ℕ) × ℕ :=
  ((#[] : Array (ByteArray × ℕ)).push (c, (_root_.Flock.u64OfLE c (c.size - 8)).toNat),
    0 + (_root_.Flock.u64OfLE c (c.size - 8)).toNat)

/-- **The recorded zerocheck up to its first round**, from a rep's start: the rest at that round recorded. -/
theorem zerocheckFw_head (m : ℕ) (digest cap root dom : ByteArray) (lay : _root_.Flock.Zk.PadLayout)
    (mk : ℕ → F128) :
    zerocheckFw recAsks m digest cap root lay mk (Transcript.init #[] dom) (#[], 0) =
      zcRest recAsks m lay mk (skelT8 (zcPend digest cap root dom ++ header 4 2 6))
        (skelS1 (zcPend digest cap root dom ++ header 4 2 6)) := by
  rw [zerocheckFw_split]
  rfl

theorem pendAlike_label {p p' : ByteArray} (h : PendAlike p p') (l : String) (hl : l.toUTF8.size < 2 ^ 64)
    (hi : l.toUTF8 ≠ _root_.Flock.Zk.LABEL_INNER.toUTF8) :
    PendAlike (p ++ header 2 0 l.toUTF8.size ++ _root_.Flock.Transcript.padded l.toUTF8)
      (p' ++ header 2 0 l.toUTF8.size ++ _root_.Flock.Transcript.padded l.toUTF8) :=
  pendAlike_snoc h (FlockSoundness.Refine.label_enc _ l) (FlockSoundness.Refine.label_enc _ l) rfl hl
    fun e => hi (Frame.label.inj e)

theorem pendAlike_bytes {p p' b b' : ByteArray} (h : PendAlike p p') (hb : b.size = b'.size)
    (hn : b.size < 2 ^ 64) :
    PendAlike (p ++ header 5 0 b.size ++ _root_.Flock.Transcript.padded b)
      (p' ++ header 5 0 b'.size ++ _root_.Flock.Transcript.padded b') :=
  pendAlike_snoc h (FlockSoundness.Refine.absorbBytes_enc _ b) (FlockSoundness.Refine.absorbBytes_enc _ b') hb hn
    nofun

theorem zcPre_alike (digest cap cap' dom : ByteArray) (hc : cap.size = cap'.size) (hd : digest.size < 2 ^ 64)
    (hcn : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64) : PendAlike (zcPre digest cap dom) (zcPre digest cap' dom) :=
  pendAlike_label (pendAlike_bytes (pendAlike_bytes (pendAlike_label (trAlike_init dom hdom) _ (by decide) (by decide))
    rfl hd) hc hcn) _ (by decide) (by decide)

theorem zcPend_alike (digest cap cap' root dom : ByteArray) (hc : cap.size = cap'.size)
    (hd : digest.size < 2 ^ 64) (hcn : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64) :
    PendAlike (zcPend digest cap root dom) (zcPend digest cap' root dom) :=
  pendAlike_bytes (pendAlike_label (zcPre_alike digest cap cap' dom hc hd hcn hdom) _ (by decide) (by decide)) rfl hr

theorem head_eq (st : _root_.Flock.Setup) (cap dom : ByteArray) :
    _root_.Flock.Firewall.Skeleton.head st cap dom = zcPre st.digest cap dom := by
  simp only [_root_.Flock.Firewall.Skeleton.head, zcPre, _root_.Flock.Firewall.Frame.labelFrame,
    _root_.Flock.Firewall.Frame.bytesFrame, _root_.Flock.Firewall.Frame.LABEL_ZC, ByteArray.append_assoc]

theorem zcPend_eq (digest cap root dom : ByteArray) :
    zcPend digest cap root dom = zcPre digest cap dom ++ padsFrames root := by
  simp only [zcPend, padsFrames, _root_.Flock.Firewall.Frame.labelFrame, _root_.Flock.Firewall.Frame.bytesFrame,
    ByteArray.append_assoc]

theorem forall₂_set0 {α β : Type} {R : α → β → Prop} {l : List α} {l' : List β} {a : α} {b : β}
    (h : List.Forall₂ R l l') (hab : R a b) : List.Forall₂ R (l.set 0 a) (l'.set 0 b) := by
  cases h with
  | nil => exact .nil
  | cons _ ht => exact .cons hab ht

/-- **The recorded PIOP at two caps of one size**, from a rep's start: recordings of one shape, pinned at their first
rounds. -/
theorem skel_run (st : _root_.Flock.Setup) (lay : _root_.Flock.Zk.PadLayout) (cap cap' root dom : ByteArray)
    (hc : cap.size = cap'.size) (hd : st.digest.size < 2 ^ 64) (hcn : cap.size < 2 ^ 64) (hr : root.size < 2 ^ 64)
    (hdom : dom.size < 2 ^ 64) (hm : st.m < 2 ^ 64) :
    RelV (fun a b => RecP (zcPend st.digest cap root dom ++ header 4 2 6, 6)
        (zcPend st.digest cap' root dom ++ header 4 2 6, 6) a.2 b.2)
      (((repPiop recAsks st lay cap root #[] fun _ => 0) >>= fun _ => pure () : TMW RecM Unit).run'
        (Transcript.init #[] dom) |>.run (#[], 0))
      (((repPiop recAsks st lay cap' root #[] fun _ => 0) >>= fun _ => pure () : TMW RecM Unit).run'
        (Transcript.init #[] dom) |>.run (#[], 0)) := by
  have hcnt : ∀ c, (_root_.Flock.u64OfLE (zcPend st.digest c root dom ++ header 4 2 6)
      ((zcPend st.digest c root dom ++ header 4 2 6).size - 8)).toNat = 6 :=
    fun c => count_header _ 2 6 (by decide)
  have hpin : RecP (zcPend st.digest cap root dom ++ header 4 2 6, 6)
      (zcPend st.digest cap' root dom ++ header 4 2 6, 6)
      (skelS1 (zcPend st.digest cap root dom ++ header 4 2 6))
      (skelS1 (zcPend st.digest cap' root dom ++ header 4 2 6)) := by
    unfold RecP skelS1
    dsimp only
    rw [hcnt, hcnt]
    exact ⟨rfl, List.Forall₂.cons ⟨rfl, _, _, 2, zcPend_alike st.digest cap cap' root dom hc hd hcn hr hdom, rfl, rfl,
      .inl rfl, show (6 : ℕ) ≠ 0 by decide, show (6 : ℕ) < 2 ^ 64 by decide⟩ List.Forall₂.nil, rfl, rfl⟩
  rw [tmw_run'_run, tmw_run'_run]
  refine relV_bind (S := fun a b => RecP (zcPend st.digest cap root dom ++ header 4 2 6, 6)
    (zcPend st.digest cap' root dom ++ header 4 2 6, 6) a.2 b.2) ?_ fun a b h => by exact h
  rw [tmw_bind_apply, tmw_bind_apply]
  refine relV_bind (S := fun a b => RecP (zcPend st.digest cap root dom ++ header 4 2 6, 6)
    (zcPend st.digest cap' root dom ++ header 4 2 6, 6) a.2 b.2) ?_ fun a b h => by exact h
  unfold repPiop
  rw [tmw_bind_apply, tmw_bind_apply, zerocheckFw_head, zerocheckFw_head]
  refine relV_bind (zcRest_sim (L := recLift _ _) st.m lay _ hm (skelT8 _) (skelT8 _) pendAlike_empty _ _ hpin) fun a b hab => ?_
  obtain ⟨⟨x, t1⟩, s1⟩ := a
  obtain ⟨⟨y, t1'⟩, s1'⟩ := b
  obtain ⟨⟨hxy, ht⟩, hs⟩ := hab
  dsimp only at hxy ht hs
  subst hxy
  obtain ⟨zp, zc⟩ := x
  dsimp only
  exact relV_mono (sim_bind (L := recLift (zcPend st.digest cap root dom ++ header 4 2 6, 6)
      (zcPend st.digest cap' root dom ++ header 4 2 6, 6)) (S := (· = ·)) (T := (· = ·))
    (lincheckFw_sim st.fold lay (fun _ => 0) zc) (fun u v h => by subst h; exact sim_pure rfl)
    t1 t1' ht s1 s1' hs) fun _ _ h => h.2

/-- **The algebraic phase's scheduled rounds at two level-0 caps of one size** are rounds of one shape, or both
refused. -/
theorem rounds_alike (st : _root_.Flock.Setup) (k0 : ℕ) (cap cap' dom : ByteArray) (hc : cap.size = cap'.size)
    (hd : st.digest.size < 2 ^ 64) (hcn : cap.size < 2 ^ 64) (hdom : dom.size < 2 ^ 64) (hm : st.m < 2 ^ 64) :
    RelV RoundsAt (_root_.Flock.Firewall.Skeleton.rounds st k0 cap dom)
      (_root_.Flock.Firewall.Skeleton.rounds st k0 cap' dom) := by
  unfold _root_.Flock.Firewall.Skeleton.rounds
  dsimp only
  refine relV_bind (skel_run st _ cap cap' _ dom hc hd hcn
    (show (Array.replicate 64 (0 : UInt8)).size < 2 ^ 64 by simp) hdom hm) fun a b hab => ?_
  obtain ⟨_, rs, k⟩ := a
  obtain ⟨_, rs', k'⟩ := b
  obtain ⟨_, hR, h0, h0'⟩ := hab
  dsimp only at hR h0 h0'
  dsimp only
  rw [h0, h0']
  dsimp only
  have key : ∀ c, zcPend st.digest c (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6 =
      _root_.Flock.Firewall.Skeleton.head st c dom ++ (padsFrames (ByteArray.mk (Array.replicate 64 0)) ++
        header 4 2 6) := fun c => by
    rw [zcPend_eq, head_eq, ByteArray.append_assoc]
  have e1 : ∀ c, (zcPend st.digest c (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).extract 0
      (_root_.Flock.Firewall.Skeleton.head st c dom).size = _root_.Flock.Firewall.Skeleton.head st c dom := fun c => by
    rw [key]
    exact ByteArray.extract_append_eq_left rfl
  have e2 : ∀ c, (zcPend st.digest c (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).extract
      (_root_.Flock.Firewall.Skeleton.head st c dom).size ((_root_.Flock.Firewall.Skeleton.head st c dom).size +
        (padsFrames (ByteArray.mk (Array.replicate 64 0))).size) =
      padsFrames (ByteArray.mk (Array.replicate 64 0)) := fun c => by
    have := @ByteArray.extract_append_size_add (_root_.Flock.Firewall.Skeleton.head st c dom)
      (padsFrames (ByteArray.mk (Array.replicate 64 0)) ++ header 4 2 6) 0
      (padsFrames (ByteArray.mk (Array.replicate 64 0))).size
    simp only [Nat.add_zero] at this
    rw [key, this]
    exact ByteArray.extract_append_eq_left rfl
  have e3 : ∀ c, (zcPend st.digest c (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).extract
      ((_root_.Flock.Firewall.Skeleton.head st c dom).size +
        (padsFrames (ByteArray.mk (Array.replicate 64 0))).size)
      (zcPend st.digest c (ByteArray.mk (Array.replicate 64 0)) dom ++ header 4 2 6).size = header 4 2 6 :=
    fun c => by
      rw [key, ← ByteArray.append_assoc]
      exact ByteArray.extract_append_eq_right ByteArray.size_append.symm ByteArray.size_append
  simp only [e1, e2, e3, bne_self_eq_false, Bool.false_eq_true, or_self, ↓reduceIte]
  have hpre : PendAlike (_root_.Flock.Firewall.Skeleton.head st cap dom)
      (_root_.Flock.Firewall.Skeleton.head st cap' dom) := by
    rw [head_eq, head_eq]
    exact zcPre_alike st.digest cap cap' dom hc hd hcn hdom
  show RoundsAt _ _
  unfold RoundsAt
  simp only [Array.set!_eq_setIfInBounds, Array.toList_setIfInBounds]
  exact forall₂_set0 hR ⟨rfl, _, _, 2, hpre, rfl, rfl, .inl rfl, show (6 : ℕ) ≠ 0 by decide,
    show (6 : ℕ) < 2 ^ 64 by decide⟩

end Skel

theorem relV_ite {α β : Type} {S : α → β → Prop} (c : Prop) [Decidable c] {x x' : V α} {y y' : V β}
    (h1 : c → RelV S x y) (h2 : ¬ c → RelV S x' y') : RelV S (if c then x else x') (if c then y else y') := by
  by_cases hc : c
  · rw [if_pos hc, if_pos hc]; exact h1 hc
  · rw [if_neg hc, if_neg hc]; exact h2 hc

theorem relV_refl {α : Type} (x : V α) : RelV (· = ·) x x := by
  cases x
  · exact True.intro
  · exact (rfl : _ = _)

/-- **Two tables of one shape**: the same statement, schedule and `t_pad`, packed tables of one size and level-0
commitments of one shape. -/
def TabAlike (t t' : _root_.Flock.Firewall.Finish.Tab) : Prop :=
  t.st = t'.st ∧ t.sch = t'.sch ∧ t.tPad = t'.tPad ∧ t.g.size = t'.g.size ∧ CapAlike t.cm0 t'.cm0

/-- **Two level 0s the loop draws at one `m`**, of packed tables of one size, give tables of one shape, or are both
refused. -/
theorem tabOfLevel0_alike (st : _root_.Flock.Setup) {m : ℕ} {g g' : Array F128} {d d' : ByteArray}
    {l l' : _root_.Flock.Firewall.Level0.Level0} (h : _root_.Flock.Firewall.Level0.ofDraws m g d = .ok l)
    (h' : _root_.Flock.Firewall.Level0.ofDraws m g' d' = .ok l') (hg : g.size = g'.size)
    (hd : 64 * 2 ^ l.lv.d < 2 ^ 64) :
    RelV TabAlike (_root_.Flock.Firewall.Finish.tabOfLevel0 st l) (_root_.Flock.Firewall.Finish.tabOfLevel0 st l') := by
  obtain ⟨hm, hgl, ⟨s, hs, hlv⟩, -, -, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok h
  obtain ⟨hm', hgl', ⟨s', hs', hlv'⟩, -, -, -⟩ := _root_.Flock.Firewall.Level0.ofDraws_ok h'
  rw [hs] at hs'
  obtain ⟨rfl, htp⟩ := Prod.mk.inj (Except.ok.inj hs')
  have hll : l'.lv = l.lv := by rw [hlv, hlv']
  have hcap : CapAlike (_root_.Flock.Firewall.Finish.commitOf l) (_root_.Flock.Firewall.Finish.commitOf l') := by
    rw [_root_.Flock.Firewall.Finish.commitOf_eq h, _root_.Flock.Firewall.Finish.commitOf_eq h']
    unfold _root_.Flock.Firewall.Finish.commit0
    rw [hll]
    exact commitRows_capAlike _ _ _ _ _ hd
  unfold _root_.Flock.Firewall.Finish.tabOfLevel0
  rw [hm, hm', ← htp]
  dsimp only
  refine relV_ite _ (fun _ => True.intro) fun _ => ?_
  refine relV_ite _ (fun _ => True.intro) fun _ => ?_
  refine relV_bind (relV_refl _) fun p p' hp => ?_
  subst hp
  refine relV_ite _ (fun _ => True.intro) fun _ => ?_
  show TabAlike _ _
  exact ⟨rfl, rfl, rfl, by rw [hgl, hgl']; exact hg, hcap⟩

theorem forall₂_append {α β : Type} {R : α → β → Prop} {l₁ l₂ : List α} {l₁' l₂' : List β}
    (h₁ : List.Forall₂ R l₁ l₁') (h₂ : List.Forall₂ R l₂ l₂') : List.Forall₂ R (l₁ ++ l₂) (l₁' ++ l₂') := by
  induction h₁ with
  | nil => exact h₂
  | cons h _ ih => exact .cons h ih

theorem roundsAt_append {rs rs' qs qs' : Array (ByteArray × ℕ)} (h : RoundsAt rs rs') (hq : RoundsAt qs qs') :
    RoundsAt (rs ++ qs) (rs' ++ qs') := by
  unfold RoundsAt
  rw [Array.toList_append, Array.toList_append]
  exact forall₂_append h hq

/-- **A rep's scheduled rounds so far at two tables of one shape**, on the same coins: rounds of one shape, or both
refused, whatever the salts for levels ≥ 1. -/
theorem schedSoFar_alike (t t' : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi hi' : Array ByteArray) (cr : Array (Array F128)) (hst : t.st = t'.st)
    (hsch : t.sch = t'.sch) (htp : t.tPad = t'.tPad) (hg : t.g.size = t'.g.size) (hcm : CapAlike t.cm0 t'.cm0)
    (hdom : dom.size < 2 ^ 64) (hdg : t.st.digest.size < 2 ^ 64) (hm : t.st.m < 2 ^ 64) (hx : xs.size + 2 < 2 ^ 64)
    (hn : t.sch.n + 1 < 2 ^ 64) (hq : ∀ i, (t.sch.levels.getD i default).q < 2 ^ 64) (hp : 2 * t.tPad < 2 ^ 64)
    (hd : ∀ i, 64 * 2 ^ (t.sch.levels.getD i default).d < 2 ^ 64) :
    RelV RoundsAt (Loop.schedSoFar t rep dom xs hi cr) (Loop.schedSoFar t' rep dom xs hi' cr) := by
  unfold Loop.schedSoFar
  rw [← hst, ← hsch]
  refine relV_bind (rounds_alike t.st t.sch.k0 (flat t.cm0.cap) (flat t'.cm0.cap) dom hcm.1 hdg hcm.2 hdom hm)
    fun alg alg' ha => ?_
  have hsz : alg.size = alg'.size := by
    have := ha.length_eq
    simpa using this
  rw [← hsz]
  by_cases hlt : cr.size < alg.size
  · simp only [hlt, ↓reduceIte]
    exact ha
  · simp only [hlt, ↓reduceIte]
    obtain ⟨hs, hrs⟩ := stepRel_feed (cr.extract alg.size cr.size).toList (rs := #[]) (rs' := #[])
      (openStep_alike t t' rep dom xs hi hi' (cr.extract 0 alg.size) hst hsch htp hg hcm hdom hdg hm hx hn hq hp hd)
      List.Forall₂.nil
    revert hs hrs
    generalize (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
      (cr.extract alg.size cr.size).toList #[] = A
    generalize (_root_.Flock.Firewall.Drive.openStep t' rep dom xs hi' (cr.extract 0 alg.size)).feed
      (cr.extract alg.size cr.size).toList #[] = B
    obtain ⟨sA, rsA⟩ := A
    obtain ⟨sB, rsB⟩ := B
    intro hs hrs
    dsimp only at hs hrs
    cases hs with
    | done => exact roundsAt_append ha hrs
    | fail => exact True.intro
    | ask hc _ =>
      refine roundsAt_append ha ?_
      unfold RoundsAt
      rw [Array.toList_push, Array.toList_push]
      exact forall₂_push hrs hc

end Sched

namespace Loop

/-! ## The loop's bookkeeping -/

/-- **The next rep round, in the same rep stream, is the stream's next round.** -/
theorem locate_succ : ∀ (rs : List ℕ) (k i : ℕ),
    (ZKMal.CoinLayer.locate rs k (i + 1)).1 = (ZKMal.CoinLayer.locate rs k i).1 →
    (ZKMal.CoinLayer.locate rs k (i + 1)).2 = (ZKMal.CoinLayer.locate rs k i).2 + 1
  | [], _, _, _ => rfl
  | c :: cs, k, i, h => by
    by_cases h1 : i + 1 < c
    · rw [locate_cons_lt h1, locate_cons_lt (Nat.lt_of_succ_lt h1)]
    · by_cases h0 : i < c
      · rw [locate_cons_ge h1, locate_cons_lt h0] at h
        have := locate_fst_ge cs (k + 1) (i + 1 - c)
        omega
      · rw [locate_cons_ge h1, locate_cons_ge h0] at h ⊢
        have e : i + 1 - c = i - c + 1 := by omega
        rw [e] at h ⊢
        exact locate_succ cs (k + 1) (i - c) h

theorem locate_snd_le : ∀ (rs : List ℕ) (k i : ℕ), (ZKMal.CoinLayer.locate rs k i).2 ≤ i
  | [], _, _ => Nat.le_refl _
  | c :: cs, k, i => by
    by_cases h : i < c
    · rw [locate_cons_lt h]
    · rw [locate_cons_ge h]
      exact Nat.le_trans (locate_snd_le cs (k + 1) (i - c)) (Nat.sub_le _ _)

theorem flatMap_range_length {α : Type} (f : ℕ → List α) (hf : ∀ k, (f k).length = 2) :
    ∀ i, ((List.range i).flatMap f).length = 2 * i
  | 0 => rfl
  | i + 1 => by
    rw [List.range_succ, List.flatMap_append, List.length_append, flatMap_range_length f hf i, List.flatMap_singleton,
      hf]
    omega

/-- The first `2 i + 1` requests of rounds of two requests each: the first `i` rounds' and the first of round `i`'s. -/
theorem flatMap_range_take {α : Type} (f : ℕ → List α) (hf : ∀ k, (f k).length = 2) {i n : ℕ} (h : i < n) :
    ((List.range n).flatMap f).take (2 * i + 1) = (List.range i).flatMap f ++ (f i).take 1 := by
  obtain ⟨d, rfl⟩ : ∃ d, n = (i + 1) + d := ⟨n - (i + 1), by omega⟩
  rw [List.range_add, List.flatMap_append, List.range_succ, List.flatMap_append, List.flatMap_singleton,
    List.take_append_of_le_length (by rw [List.length_append, flatMap_range_length f hf, hf]; omega)]
  rw [show 2 * i + 1 = ((List.range i).flatMap f).length + 1 by rw [flatMap_range_length f hf]]
  exact List.take_length_add_append 1

theorem flatMap_range_congr {α : Type} {f g : ℕ → List α} {i : ℕ} (h : ∀ k, k < i → f k = g k) :
    (List.range i).flatMap f = (List.range i).flatMap g :=
  List.flatMap_congr fun k hk => h k (List.mem_range.1 hk)

/-- **The drive's requests up to round `i`'s**, a scheduled round: the earlier rep's coins, the rounds before it with
their coins, then round `i` without its coins. -/
theorem reqs_take (prior : Array F128) (sched own : Array (ByteArray × ℕ)) (dev : ℕ → ByteArray)
    (coins : Array (Array F128)) {i : ℕ} (hi : i < sched.size) :
    (_root_.Flock.Firewall.Drive.reqs prior sched own dev coins).take ((if prior.isEmpty then 0 else 1) + 2 * i + 1) =
      (if prior.isEmpty then [] else [.prior prior.size (_root_.Flock.Firewall.Frame.wordBytes prior)]) ++
      (List.range i).flatMap (fun k =>
        [.round (sched.getD k (.empty, 0)).2 (dev k) (.sched (sched.getD k (.empty, 0)).1),
          .coins (.ok (coins.getD k #[]))]) ++
      [.round (sched.getD i (.empty, 0)).2 (dev i) (.sched (sched.getD i (.empty, 0)).1)] := by
  unfold _root_.Flock.Firewall.Drive.reqs
  have hl : ∀ k, ([.round (sched.getD k (.empty, 0)).2 (dev k) (.sched (sched.getD k (.empty, 0)).1),
      .coins (.ok (coins.getD k #[]))] : List _root_.Flock.Firewall.Frame.Req).length = 2 := fun _ => rfl
  have hr := flatMap_range_take _ hl hi
  have hlen := flatMap_range_length _ hl sched.size
  rw [List.append_assoc, List.append_assoc]
  cases hp : prior.isEmpty
  · simp only [Bool.false_eq_true, ↓reduceIte]
    rw [show 1 + 2 * i + 1 = [_root_.Flock.Firewall.Frame.Req.prior prior.size
        (_root_.Flock.Firewall.Frame.wordBytes prior)].length + (2 * i + 1) by simp; omega,
      List.take_length_add_append, List.take_append_of_le_length (by omega), hr]
    rfl
  · simp only [↓reduceIte, List.nil_append, Nat.zero_add]
    rw [List.take_append_of_le_length (by omega), hr]
    rfl

theorem feed_append {α : Type} : ∀ (s : _root_.Flock.Firewall.Drive.Step α) (l₁ l₂ : List (Array F128))
    (rs : Array (ByteArray × ℕ)), s.feed (l₁ ++ l₂) rs = (s.feed l₁ rs).1.feed l₂ (s.feed l₁ rs).2
  | s, [], l₂, rs => by cases s <;> rfl
  | .ask c n k, a :: l₁, l₂, rs => feed_append (k a) l₁ l₂ (rs.push (c, n))
  | .done _, _ :: _, l₂, _ => by cases l₂ <;> rfl
  | .fail _, _ :: _, l₂, _ => by cases l₂ <;> rfl

theorem feed_nil {α : Type} (s : _root_.Flock.Firewall.Drive.Step α) (rs : Array (ByteArray × ℕ)) :
    s.feed [] rs = (s, rs) := by
  cases s <;> rfl

/-- **The schedule so far only grows**: one more round's coins keep every round scheduled so far. -/
theorem schedSoFar_snoc (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr : Array (Array F128)) (a : Array F128)
    {s s' : Array (ByteArray × ℕ)} (h : schedSoFar t rep dom xs hi cr = .ok s)
    (h' : schedSoFar t rep dom xs hi (cr.push a) = .ok s') : ∃ u, s' = s ++ u := by
  unfold schedSoFar at h h'
  cases hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom with
  | error e => rw [hal] at h; cases h
  | ok alg =>
    rw [hal] at h h'
    simp only [bind, Except.bind] at h h'
    by_cases hlt : cr.size < alg.size
    · simp only [hlt, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
      subst h
      by_cases hlt' : (cr.push a).size < alg.size
      · simp only [hlt', ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h'
        exact ⟨#[], by rw [h', Array.append_empty]⟩
      · simp only [hlt', ↓reduceIte] at h'
        have e : ((cr.push a).extract alg.size (cr.push a).size).toList = [] := by
          rw [Array.extract_empty_of_size_le_start (by rw [Array.size_push] at hlt' ⊢; omega)]
        rw [e, feed_nil] at h'
        revert h'
        generalize _root_.Flock.Firewall.Drive.openStep t rep dom xs hi ((cr.push a).extract 0 alg.size) = S
        intro h'
        cases S with
        | ask c n k =>
          simp only [pure, Except.pure, Except.ok.injEq] at h'
          exact ⟨#[(c, n)], by rw [← h']; rfl⟩
        | done _ =>
          simp only [pure, Except.pure, Except.ok.injEq] at h'
          exact ⟨#[], by rw [← h']⟩
        | fail _ => cases h'
    · have hlt' : ¬ (cr.push a).size < alg.size := by rw [Array.size_push]; omega
      simp only [hlt, ↓reduceIte] at h
      simp only [hlt', ↓reduceIte] at h'
      have e1 : (cr.push a).extract 0 alg.size = cr.extract 0 alg.size := by
        rw [Array.extract_push, if_pos (by omega)]
      have e2 : ((cr.push a).extract alg.size (cr.push a).size).toList =
          (cr.extract alg.size cr.size).toList ++ [a] := by
        rw [Array.size_push, Array.extract_push, if_neg (by omega), if_pos (by omega), Array.toList_push]
      rw [e1, e2, feed_append] at h'
      revert h h'
      generalize (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
        (cr.extract alg.size cr.size).toList #[] = P
      obtain ⟨S, rs⟩ := P
      intro h h'
      cases S with
      | ask c n k =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        dsimp only at h'
        rw [show (_root_.Flock.Firewall.Drive.Step.ask c n k).feed [a] rs = (k a).feed [] (rs.push (c, n)) from rfl,
          feed_nil] at h'
        cases hk : k a with
        | ask c' n' _ =>
          rw [hk] at h'
          simp only [pure, Except.pure, Except.ok.injEq] at h'
          exact ⟨#[(c', n')], by rw [← h']; simp⟩
        | done _ =>
          rw [hk] at h'
          simp only [pure, Except.pure, Except.ok.injEq] at h'
          exact ⟨#[], by rw [← h', Array.append_empty]⟩
        | fail _ => rw [hk] at h'; cases h'
      | done _ =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        dsimp only at h'
        rw [show ∀ x, (_root_.Flock.Firewall.Drive.Step.done x : _root_.Flock.Firewall.Drive.Step Unit).feed [a] rs =
          (.done x, rs) from fun _ => rfl] at h'
        simp only [pure, Except.pure, Except.ok.injEq] at h'
        exact ⟨#[], by rw [← h', Array.append_empty]⟩
      | fail _ => cases h

/-- A step still asking after the coins `l` sent a round for each. -/
theorem feed_ask_size {α : Type} : ∀ (s : _root_.Flock.Firewall.Drive.Step α) (l : List (Array F128))
    (rs : Array (ByteArray × ℕ)) {c n k rs'}, s.feed l rs = (.ask c n k, rs') → rs'.size = rs.size + l.length
  | s, [], rs, _, _, _, _, h => by rw [feed_nil] at h; cases h; rfl
  | .ask c' n' k', a :: l, rs, _, _, _, _, h => by
    have := feed_ask_size (k' a) l (rs.push (c', n')) h
    rw [this, Array.size_push, List.length_cons]
    omega
  | .done _, _ :: _, _, _, _, _, _, h => by cases h
  | .fail _, _ :: _, _, _, _, _, _, h => by cases h

/-- **A schedule that has stopped growing**: where the schedule so far holds no round past its coins, the opening has
ended, and one more round's coins schedule nothing more. -/
theorem schedSoFar_snoc_done (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr : Array (Array F128)) (a : Array F128)
    {s s' : Array (ByteArray × ℕ)} (h : schedSoFar t rep dom xs hi cr = .ok s) (hle : s.size ≤ cr.size)
    (h' : schedSoFar t rep dom xs hi (cr.push a) = .ok s') : s' = s := by
  unfold schedSoFar at h h'
  cases hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom with
  | error e => rw [hal] at h; cases h
  | ok alg =>
    rw [hal] at h h'
    simp only [bind, Except.bind] at h h'
    by_cases hlt : cr.size < alg.size
    · simp only [hlt, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
      subst h
      omega
    · have hlt' : ¬ (cr.push a).size < alg.size := by rw [Array.size_push]; omega
      simp only [hlt, ↓reduceIte] at h
      simp only [hlt', ↓reduceIte] at h'
      have e1 : (cr.push a).extract 0 alg.size = cr.extract 0 alg.size := by
        rw [Array.extract_push, if_pos (by omega)]
      have e2 : ((cr.push a).extract alg.size (cr.push a).size).toList =
          (cr.extract alg.size cr.size).toList ++ [a] := by
        rw [Array.size_push, Array.extract_push, if_neg (by omega), if_pos (by omega), Array.toList_push]
      rw [e1, e2, feed_append] at h'
      have hlen : (cr.extract alg.size cr.size).toList.length = cr.size - alg.size := by
        rw [Array.length_toList, Array.size_extract]
        omega
      revert h h'
      generalize hP : (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
        (cr.extract alg.size cr.size).toList #[] = P
      obtain ⟨S, rs⟩ := P
      intro h h'
      cases S with
      | ask c n k =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        have := feed_ask_size _ _ _ hP
        rw [Array.size_append, Array.size_push, this, hlen, Array.size_empty] at hle
        omega
      | done _ =>
        simp only [pure, Except.pure, Except.ok.injEq] at h
        subst h
        dsimp only at h'
        rw [show ∀ x, (_root_.Flock.Firewall.Drive.Step.done x : _root_.Flock.Firewall.Drive.Step Unit).feed [a] rs =
          (.done x, rs) from fun _ => rfl] at h'
        simp only [pure, Except.pure, Except.ok.injEq] at h'
        exact h'.symm
      | fail _ => cases h

/-- **The schedule on one more round's coins reaches one round further**: where the schedule on the coins `cr` has
round `j + 1` and the coins before the last (`cr.pop`, `j` of them) schedule `sched`, `sched` has round `j`. -/
theorem schedSoFar_pop_lt (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) {cr : Array (Array F128)} {j : ℕ}
    {sched sched' : Array (ByteArray × ℕ)} (hs : schedSoFar t rep dom xs hi cr.pop = .ok sched)
    (hs' : schedSoFar t rep dom xs hi cr = .ok sched') (hcr : cr.size = j + 1) (hj : j + 1 < sched'.size) :
    j < sched.size := by
  rw [Array.eq_push_pop_back!_of_size_ne_zero (show cr.size ≠ 0 by omega)] at hs'
  refine Nat.lt_of_not_le fun hle => ?_
  have e := schedSoFar_snoc_done t rep dom xs hi cr.pop cr.back! hs (by rw [Array.size_pop]; omega) hs'
  subst e
  omega

theorem schedSoFar_ok_rounds {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {s : Array (ByteArray × ℕ)}
    (h : schedSoFar t rep dom xs hi cr = .ok s) :
    ∃ alg, _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg := by
  unfold schedSoFar at h
  cases hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom with
  | error e => rw [hal] at h; cases h
  | ok alg => exact ⟨alg, rfl⟩

/-- **The schedule on fewer coins is not refused** where the schedule on one more round's is not: a refused opening
stays refused. -/
theorem schedSoFar_push_ok (t : _root_.Flock.Firewall.Finish.Tab) (rep : ℕ) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr : Array (Array F128)) (a : Array F128)
    {s : Array (ByteArray × ℕ)} (h : schedSoFar t rep dom xs hi (cr.push a) = .ok s) :
    ∃ s0, schedSoFar t rep dom xs hi cr = .ok s0 := by
  unfold schedSoFar at h ⊢
  cases hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom with
  | error e => rw [hal] at h; cases h
  | ok alg =>
    rw [hal] at h
    simp only [bind, Except.bind] at h ⊢
    by_cases hlt : cr.size < alg.size
    · simp only [hlt, ↓reduceIte, pure, Except.pure]
      exact ⟨alg, rfl⟩
    · have hlt' : ¬ (cr.push a).size < alg.size := by rw [Array.size_push]; omega
      simp only [hlt, ↓reduceIte]
      simp only [hlt', ↓reduceIte] at h
      have e1 : (cr.push a).extract 0 alg.size = cr.extract 0 alg.size := by
        rw [Array.extract_push, if_pos (by omega)]
      have e2 : ((cr.push a).extract alg.size (cr.push a).size).toList =
          (cr.extract alg.size cr.size).toList ++ [a] := by
        rw [Array.size_push, Array.extract_push, if_neg (by omega), if_pos (by omega), Array.toList_push]
      rw [e1, e2, feed_append] at h
      revert h
      generalize (_root_.Flock.Firewall.Drive.openStep t rep dom xs hi (cr.extract 0 alg.size)).feed
        (cr.extract alg.size cr.size).toList #[] = P
      obtain ⟨S, rs⟩ := P
      intro h
      cases S with
      | ask c n k => exact ⟨_, rfl⟩
      | done _ => exact ⟨_, rfl⟩
      | fail e => cases h

theorem schedSoFar_pop_ok {t : _root_.Flock.Firewall.Finish.Tab} {rep : ℕ} {dom : ByteArray}
    {xs : Array (Array F128)} {hi : Array ByteArray} {cr : Array (Array F128)} {s : Array (ByteArray × ℕ)}
    (h : schedSoFar t rep dom xs hi cr = .ok s) : ∃ s0, schedSoFar t rep dom xs hi cr.pop = .ok s0 := by
  by_cases h0 : cr.size = 0
  · have e : cr.pop = cr := by
      rw [Array.eq_empty_of_size_eq_zero h0]
      rfl
    rw [e]
    exact ⟨s, h⟩
  · rw [Array.eq_push_pop_back!_of_size_ne_zero h0] at h
    exact schedSoFar_push_ok t rep dom xs hi cr.pop cr.back! h

end Loop

end Claims.ZKServer
