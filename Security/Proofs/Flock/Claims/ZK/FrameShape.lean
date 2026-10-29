import Proofs.Flock.Claims.ZK.FrameParse

/-!
# The frame reads a round's shape alone

The loop's release runs each rep's frame on the scheduled rounds
at placeholder draws, salts and level-0 cap, so the frame's state after a round must not read what those change: the
draws' values (the pads, their root and rows), a bytes frame's body (a commitment's cap), an observe's values, and
whether the rep is stopped. Over a round's frames (`roundFree_encs`), the frame's step (`stepAt`) reads a label's bytes,
a bytes frame's size, an observe's length and the configuration's layout, and nothing else of them:

* `RelM f x y`: two runs alike in `f` (both return, at results equal under `f`, or both fail with one error), with its
  `bind` and `map` (`relM_bind`, `relM_map`).
* `CfgAlike`: two configurations of one layout, mask slot and pad count, whatever their draws; `unstop`: a state but
  for `stopped`.
* `Alike`: two frames of one shape (the same frame but for a bytes frame's body and an observe's values).
* `absorb_alike`, `label_alike`, `absorb1_alike`, `foldlM_absorb1_alike`, `stepAt_alike`: each step, at alike frames and
  configurations, from states equal but for `stopped` and the round's release so far, returns states equal but for
  them, or the same error.
* **`foldlM_stepAt_alike`**: so does the fold over alike frames.
* **`roundFree_alike`**: and so the frame's round over alike frames, outside the inner proof, returns the same state but
  for `stopped` and the same free offsets, or the same error; **`roundFree_alike_ok`**: where both rounds are taken,
  inside the inner proof too (whose `innerOwn` reads the round's values), the same state and offsets.
-/

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Transcript (header)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS roundFree)
open FlockSoundness.Refine (Frame encs)

/-- **Two runs alike in `f`**: both return, at results equal under `f`, or both fail with one error. -/
def RelM {α β : Type} (f : α → β) : V α → V α → Prop
  | .ok a, .ok b => f a = f b
  | .error e, .error e' => e = e'
  | _, _ => False

theorem relM_refl {α β : Type} (f : α → β) (x : V α) : RelM f x x := by
  cases x
  · exact (rfl : _ = _)
  · exact (rfl : _ = _)

theorem relM_bind {α β γ δ : Type} {f : α → γ} {g : β → δ} {x y : V α} {k k' : α → V β} (h : RelM f x y)
    (hk : ∀ a b, x = .ok a → y = .ok b → f a = f b → RelM g (k a) (k' b)) : RelM g (x >>= k) (y >>= k') := by
  cases x with
  | error e =>
    cases y with
    | error e' => exact (h : e = e')
    | ok b => exact (h : False).elim
  | ok a =>
    cases y with
    | error e' => exact (h : False).elim
    | ok b => exact hk a b rfl rfl h

theorem relM_map {α β γ δ : Type} {f : α → γ} {g : β → δ} {x y : V α} {k k' : α → β} (h : RelM f x y)
    (hk : ∀ a b, f a = f b → g (k a) = g (k' b)) : RelM g (k <$> x) (k' <$> y) := by
  cases x with
  | error e =>
    cases y with
    | error e' => exact (h : e = e')
    | ok b => exact (h : False).elim
  | ok a =>
    cases y with
    | error e' => exact (h : False).elim
    | ok b => exact hk a b h

/-- **Two configurations of one rep's layout**: equal but for the draws (the pads' values, their root and the rows). -/
structure CfgAlike (c c' : Cfg) : Prop where
  lay : c.lay = c'.lay
  order : c.order = c'.order
  nPiop : c.nPiop = c'.nPiop
  lig : c.lig = c'.lig
  mask : c.mask = c'.mask
  size : c.h.size = c'.h.size

/-- A state but for `stopped`. -/
def unstop (s : Masker) : Masker := { s with stopped := false }

theorem eq_unstop {s s' : Masker} (h : unstop s = unstop s') : s' = { s with stopped := s'.stopped } := by
  have e : s' = { unstop s' with stopped := s'.stopped } := rfl
  rw [e, ← h]
  rfl

theorem relM_pure {α β : Type} {f : α → β} {a b : α} (h : f a = f b) : RelM f (pure a) (pure b) := h

theorem relM_throw {α β : Type} (f : α → β) (e : String) : RelM f (throw e) (throw e) := rfl

theorem getElem?_none_iff_of_size {α β : Type} {xs : Array α} {ys : Array β} (h : xs.size = ys.size) (i : ℕ) :
    xs[i]? = none ↔ ys[i]? = none := by
  rw [Array.getElem?_eq_none_iff, Array.getElem?_eq_none_iff, h]

theorem absorb_alike {c c' : Cfg} (hc : CfgAlike c c') (s : Masker) (b : Bool) (v v' : F128) :
    RelM (fun q => unstop q.1) (s.absorb c v) ({ s with stopped := b }.absorb c' v') := by
  unfold Masker.absorb
  rw [← hc.order]
  by_cases hon : (!s.on) = true
  · simp only [hon, ↓reduceIte]
    exact relM_pure rfl
  · simp only [hon, Bool.false_eq_true, ↓reduceIte]
    cases hi : c.order[s.pos]? with
    | none => exact relM_throw _ _
    | some i =>
      dsimp only
      cases hp : c.h[i]? with
      | none =>
        rw [(getElem?_none_iff_of_size hc.size i).mp hp]
        exact relM_throw _ _
      | some p =>
        cases hp' : c'.h[i]? with
        | none => rw [← getElem?_none_iff_of_size hc.size i] at hp'; rw [hp] at hp'; cases hp'
        | some p' => exact relM_pure rfl

theorem relM_of_map {α β : Type} {f : α → β} {x y : V α} (h : x.map f = y.map f) : RelM f x y := by
  cases x <;> cases y <;> first | exact Except.error.inj h | exact Except.ok.inj h | cases h

theorem exMap_bind {α β γ : Type} (f : β → γ) (x : V α) (k : α → V β) :
    (x >>= k).map f = x >>= fun a => (k a).map f := by
  cases x <;> rfl

theorem exMap_pure {α β : Type} (f : α → β) (a : α) : (pure a : V α).map f = pure (f a) := rfl

theorem label_alike {c c' : Cfg} (hc : CfgAlike c c') (s : Masker) (b : Bool) (l : ByteArray) :
    RelM (fun q => unstop q.1) (s.label c l) ({ s with stopped := b }.label c' l) := by
  apply relM_of_map
  unfold Masker.label
  rw [← hc.nPiop, ← hc.mask, ← hc.lay, ← hc.order, ← hc.lig]
  cases hm : c.mask <;>
  simp only [apply_ite (Except.map _), exMap_bind, exMap_pure, unstop]

def Alike : Frame → Frame → Prop
  | .dom b, .dom b' => b.size = b'.size
  | .label b, .label b' => b = b'
  | .bytes b, .bytes b' => b.size = b'.size
  | .one _, .one _ => True
  | .vec vs, .vec vs' => vs.length = vs'.length
  | .sq k n, .sq k' n' => k = k' ∧ n = n'
  | _, _ => False

/-- The part of a step's state the frame keeps, but for `stopped`. -/
abbrev keep (r : Masker × ByteArray × Array ℕ × ℕ) : Masker × Array ℕ × ℕ := (unstop r.1, r.2.2)

theorem keep_with {r r' : Masker × ByteArray × Array ℕ × ℕ} (h : keep r = keep r') :
    ∃ b o', r' = ({ r.1 with stopped := b }, o', r.2.2) := by
  obtain ⟨m, o, fr, p⟩ := r
  obtain ⟨m', o', fr', p'⟩ := r'
  simp only [keep, Prod.mk.injEq] at h
  obtain ⟨hm, rfl, rfl⟩ := h
  exact ⟨m'.stopped, o', by rw [← eq_unstop hm]⟩

theorem absorb1_alike {c c' : Cfg} (hc : CfgAlike c c') (r : Masker × ByteArray × Array ℕ × ℕ) (b : Bool)
    (o' : ByteArray) (v v' : F128) :
    RelM keep (absorb1 c r v) (absorb1 c' ({ r.1 with stopped := b }, o', r.2.2) v') := by
  unfold absorb1
  exact relM_map (absorb_alike hc r.1 b v v') fun a a' ha => by simp only [keep, ha, hc.nPiop]

theorem foldlM_absorb1_alike {c c' : Cfg} (hc : CfgAlike c c') : ∀ (vs vs' : List F128) (_ : vs.length = vs'.length)
    (r r' : Masker × ByteArray × Array ℕ × ℕ) (_ : keep r = keep r'),
    RelM keep (vs.foldlM (absorb1 c) r) (vs'.foldlM (absorb1 c') r')
  | [], [], _, _, _, hr => hr
  | [], _ :: _, h, _, _, _ => by simp at h
  | _ :: _, [], h, _, _, _ => by simp at h
  | v :: vs, v' :: vs', h, r, r', hr => by
    obtain ⟨b, o', rfl⟩ := keep_with hr
    rw [List.foldlM_cons, List.foldlM_cons]
    exact relM_bind (absorb1_alike hc r b _ v v') fun a a' _ _ ha =>
      foldlM_absorb1_alike hc vs vs' (by simpa using h) a a' ha

/-- **The frame's step at alike frames and configurations**, from states alike but for `stopped`, reads only their
shape. -/
theorem stepAt_alike {c c' : Cfg} (hc : CfgAlike c c') (r r' : Masker × ByteArray × Array ℕ × ℕ)
    (hr : keep r = keep r') (f f' : Frame) (hf : Alike f f') : RelM keep (stepAt c r f) (stepAt c' r' f') := by
  obtain ⟨b, o', rfl⟩ := keep_with hr
  obtain ⟨m, o, fr, p⟩ := r
  cases f <;> cases f' <;> try exact (hf : False).elim
  case dom.dom x x' =>
    exact relM_pure (by rw [show x.size = x'.size from hf]; rfl)
  case label.label x x' =>
    rw [show x = x' from hf]
    exact relM_map (label_alike hc m b x') fun a a' ha => by simp only [keep, ha]
  case bytes.bytes x x' =>
    exact relM_pure (by rw [show x.size = x'.size from hf]; rfl)
  case one.one v v' => exact foldlM_absorb1_alike hc [v] [v'] rfl _ _ rfl
  case vec.vec vs vs' => exact foldlM_absorb1_alike hc vs vs' hf _ _ rfl
  case sq.sq => exact relM_pure rfl

theorem foldlM_stepAt_alike {c c' : Cfg} (hc : CfgAlike c c') : ∀ (fs fs' : List Frame) (_ : List.Forall₂ Alike fs fs')
    (r r' : Masker × ByteArray × Array ℕ × ℕ) (_ : keep r = keep r'),
    RelM keep (fs.foldlM (stepAt c) r) (fs'.foldlM (stepAt c') r')
  | [], [], _, _, _, hr => hr
  | f :: fs, f' :: fs', .cons hf h, r, r', hr => by
    rw [List.foldlM_cons, List.foldlM_cons]
    exact relM_bind (stepAt_alike hc r r' hr f f' hf) fun a b _ _ hab => foldlM_stepAt_alike hc fs fs' h a b hab

theorem exMap_ok_eq {α : Type} {P q : α} {x : V Unit} (h : (fun _ => P) <$> x = .ok q) : q = P := by
  cases x with
  | error e => cases h
  | ok u => exact (Except.ok.inj h).symm

theorem relM_map_unit {α β : Type} {g : α → β} {x y : V Unit} (hx : x = pure ()) (hy : y = pure ()) {P P' : α}
    (h : g P = g P') : RelM g ((fun _ => P) <$> x) ((fun _ => P') <$> y) := by
  subst hx
  subst hy
  exact h

theorem roundFree_alike_core {c c' : Cfg} (hc : CfgAlike c c') (s : Masker) (b : Bool) (fs fs' : List Frame) (k : UInt8)
    (n : ℕ) (X X' : ByteArray) (hA : List.Forall₂ Alike fs fs')
    (he : roundFree c s X n =
      (do
        let r ← fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0)
        (fun (_ : Unit) => (r.2.1 ++ header 4 k n, { r.1 with owed := some n }, r.2.2.1)) <$>
          (match r.1.inner with
           | some j => _root_.Flock.Firewall.Frame.innerOwn c r.1 j X n
           | none => pure () : V Unit)))
    (he' : roundFree c' { s with stopped := b } X' n =
      (do
        let r ← fs'.foldlM (stepAt c') ({ s with stopped := b }, ByteArray.empty, #[], 0)
        (fun (_ : Unit) => (r.2.1 ++ header 4 k n, { r.1 with owed := some n }, r.2.2.1)) <$>
          (match r.1.inner with
           | some j => _root_.Flock.Firewall.Frame.innerOwn c' r.1 j X' n
           | none => pure () : V Unit)))
    (hI : ∀ q, fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none) :
    RelM (fun q : ByteArray × Masker × Array ℕ => (unstop q.2.1, q.2.2)) (roundFree c s X n)
      (roundFree c' { s with stopped := b } X' n) := by
  rw [he, he']
  refine relM_bind (foldlM_stepAt_alike hc fs fs' hA _ _ rfl) fun a a' ha _ hab => ?_
  have e1 : unstop a.1 = unstop a'.1 := congrArg Prod.fst hab
  have e2 : a.2.2.1 = a'.2.2.1 := congrArg (fun q => q.2.1) hab
  have hi : a.1.inner = none := hI a ha
  have hi' : a'.1.inner = none := by
    rw [show a'.1.inner = (unstop a'.1).inner from rfl, ← e1]
    exact hi
  dsimp only
  refine relM_map_unit (by rw [hi]) (by rw [hi']) ?_
  show ({ unstop a.1 with owed := some n }, a.2.2.1) = ({ unstop a'.1 with owed := some n }, a'.2.2.1)
  rw [e1, e2]

/-- **The frame's round, where both alike rounds are taken**, leaves states alike but for `stopped`, with the same free
offsets: inside the inner proof too, where `innerOwn` reads the round's values. -/
theorem roundFree_alike_ok_core {c c' : Cfg} (hc : CfgAlike c c') (s : Masker) (b : Bool) (fs fs' : List Frame) (k : UInt8)
    (n : ℕ) (X X' : ByteArray) (hA : List.Forall₂ Alike fs fs')
    (he : roundFree c s X n =
      (do
        let r ← fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0)
        (fun (_ : Unit) => (r.2.1 ++ header 4 k n, { r.1 with owed := some n }, r.2.2.1)) <$>
          (match r.1.inner with
           | some j => _root_.Flock.Firewall.Frame.innerOwn c r.1 j X n
           | none => pure () : V Unit)))
    (he' : roundFree c' { s with stopped := b } X' n =
      (do
        let r ← fs'.foldlM (stepAt c') ({ s with stopped := b }, ByteArray.empty, #[], 0)
        (fun (_ : Unit) => (r.2.1 ++ header 4 k n, { r.1 with owed := some n }, r.2.2.1)) <$>
          (match r.1.inner with
           | some j => _root_.Flock.Firewall.Frame.innerOwn c' r.1 j X' n
           | none => pure () : V Unit)))
    {q q' : ByteArray × Masker × Array ℕ} (hq : roundFree c s X n = .ok q)
    (hq' : roundFree c' { s with stopped := b } X' n = .ok q') : unstop q.2.1 = unstop q'.2.1 ∧ q.2.2 = q'.2.2 := by
  rw [he] at hq
  rw [he'] at hq'
  cases ha : fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) with
  | error e => rw [ha] at hq; cases hq
  | ok a =>
    cases ha' : fs'.foldlM (stepAt c') ({ s with stopped := b }, ByteArray.empty, #[], 0) with
    | error e => rw [ha'] at hq'; cases hq'
    | ok a' =>
      have hab : keep a = keep a' := by
        have h := foldlM_stepAt_alike hc fs fs' hA (s, ByteArray.empty, #[], 0)
          ({ s with stopped := b }, ByteArray.empty, #[], 0) rfl
        rw [ha, ha'] at h
        exact h
      rw [ha] at hq
      rw [ha'] at hq'
      have e1 : unstop a.1 = unstop a'.1 := congrArg Prod.fst hab
      have e2 : a.2.2.1 = a'.2.2.1 := congrArg (fun q => q.2.1) hab
      have hq1 : q = (a.2.1 ++ header 4 k n, { a.1 with owed := some n }, a.2.2.1) := exMap_ok_eq hq
      have hq2 : q' = (a'.2.1 ++ header 4 k n, { a'.1 with owed := some n }, a'.2.2.1) := exMap_ok_eq hq'
      rw [hq1, hq2]
      exact ⟨show { unstop a.1 with owed := some n } = { unstop a'.1 with owed := some n } by rw [e1], e2⟩


/-- **The frame's round over alike frames and configurations**, from states equal but for `stopped`, outside the inner
proof, returns the same state but for `stopped` and the same free offsets, or the same error. -/
theorem roundFree_alike {c c' : Cfg} (hc : CfgAlike c c') (s : Masker) (b : Bool) (fs fs' : List Frame) (k : UInt8)
    (n : ℕ) (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hm' : ∀ f ∈ fs', midFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hA : List.Forall₂ Alike fs fs')
    (hI : ∀ q, fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none) :
    RelM (fun q : ByteArray × Masker × Array ℕ => (unstop q.2.1, q.2.2))
      (roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n)
      (roundFree c' { s with stopped := b } (ByteArray.mk (encs fs' ++ (Frame.sq k n).enc).toArray) n) :=
  roundFree_alike_core hc s b fs fs' k n _ _ hA (roundFree_encs c s fs k n hs hm hk h0 hn)
    (roundFree_encs c' { s with stopped := b } fs' k n hs hm' hk h0 hn) hI

/-- **The frame's round where both alike rounds are taken** leaves the same state but for `stopped` and the same free
offsets, inside the inner proof too. -/
theorem roundFree_alike_ok {c c' : Cfg} (hc : CfgAlike c c') (s : Masker) (b : Bool) (fs fs' : List Frame) (k : UInt8)
    (n : ℕ) (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hm' : ∀ f ∈ fs', midFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hA : List.Forall₂ Alike fs fs')
    {q q' : ByteArray × Masker × Array ℕ}
    (hq : roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n = .ok q)
    (hq' : roundFree c' { s with stopped := b } (ByteArray.mk (encs fs' ++ (Frame.sq k n).enc).toArray) n = .ok q') :
    unstop q.2.1 = unstop q'.2.1 ∧ q.2.2 = q'.2.2 :=
  roundFree_alike_ok_core hc s b fs fs' k n _ _ hA (roundFree_encs c s fs k n hs hm hk h0 hn)
    (roundFree_encs c' { s with stopped := b } fs' k n hs hm' hk h0 hn) hq hq'

end Claims.ZKServer
