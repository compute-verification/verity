import Proofs.Flock.Claims.ZK.FrameParse

/-!
# The frame reads a round's shape alone

Stage B (`note:20261022T0001Z-report-zkfin-stageb`, R1): the loop's release runs each rep's frame on the scheduled rounds
at placeholder salts and level-0 cap, so the frame's state after a round must not read what those change: a bytes
frame's body (a commitment's cap) and an observe's values. Over a round's frames (`roundFree_encs`), the frame's step
(`stepAt`) reads a label's bytes, a bytes frame's size and an observe's length, and nothing else of them:

* `RelM f x y`: two runs alike in `f` (both return, at results equal under `f`, or both fail with one error), with its
  `bind` and `map` (`relM_bind`, `relM_map`).
* `Alike`: two frames of one shape (the same frame but for a bytes frame's body and an observe's values).
* `absorb_alike`, `absorb1_alike`, `foldlM_absorb1_alike`, `stepAt_alike`: each step, at alike frames, from states
  equal but for the round's release so far, returns states equal but for it, or the same error.
* **`foldlM_stepAt_alike`**: so does the fold over alike frames.
* **`roundFree_alike`**: and so the frame's round over alike frames, outside the inner proof, returns the same state and
  free offsets, or the same error.
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

/-- **Two frames of one shape**: the same frame but for a bytes frame's body and an observe's values. -/
def Alike : Frame → Frame → Prop
  | .dom b, .dom b' => b.size = b'.size
  | .label b, .label b' => b = b'
  | .bytes b, .bytes b' => b.size = b'.size
  | .one _, .one _ => True
  | .vec vs, .vec vs' => vs.length = vs'.length
  | .sq k n, .sq k' n' => k = k' ∧ n = n'
  | _, _ => False

/-- The part of a step's state the frame keeps: its masking state, the round's free offsets and its position. -/
abbrev keep (r : Masker × ByteArray × Array ℕ × ℕ) : Masker × Array ℕ × ℕ := (r.1, r.2.2)

/-- **An absorbed value's state reads none of it**, nor does its error. -/
theorem absorb_alike (c : Cfg) (s : Masker) (v v' : F128) :
    RelM Prod.fst (s.absorb c v) (s.absorb c v') := by
  unfold Masker.absorb
  cases hon : s.on
  · exact (rfl : s = s)
  · cases hi : c.order[s.pos]? with
    | none => exact relM_refl _ _
    | some i =>
      cases hp : c.h[i]? with
      | none => exact relM_refl _ _
      | some p => exact (rfl : ({ s with pos := s.pos + 1 } : Masker) = { s with pos := s.pos + 1 })

theorem absorb1_alike (c : Cfg) {r r' : Masker × ByteArray × Array ℕ × ℕ} (hr : keep r = keep r') (v v' : F128) :
    RelM keep (absorb1 c r v) (absorb1 c r' v') := by
  obtain ⟨m, o, fr, p⟩ := r
  obtain ⟨m', o', fr', p'⟩ := r'
  simp only [keep, Prod.mk.injEq] at hr
  obtain ⟨rfl, rfl, rfl⟩ := hr
  unfold absorb1
  exact relM_map (absorb_alike c m v v') fun a b hab => by simp only [keep, hab]

theorem foldlM_absorb1_alike (c : Cfg) : ∀ (vs vs' : List F128) (_ : vs.length = vs'.length)
    (r r' : Masker × ByteArray × Array ℕ × ℕ) (_ : keep r = keep r'),
    RelM keep (vs.foldlM (absorb1 c) r) (vs'.foldlM (absorb1 c) r')
  | [], [], _, _, _, hr => hr
  | [], _ :: _, h, _, _, _ => by simp at h
  | _ :: _, [], h, _, _, _ => by simp at h
  | v :: vs, v' :: vs', h, r, r', hr => by
    rw [List.foldlM_cons, List.foldlM_cons]
    exact relM_bind (absorb1_alike c hr v v') fun a b _ _ hab =>
      foldlM_absorb1_alike c vs vs' (by simpa using h) a b hab

/-- **The frame's step at alike frames** reads only their shape. -/
theorem stepAt_alike (c : Cfg) (r r' : Masker × ByteArray × Array ℕ × ℕ) (hr : keep r = keep r')
    (f f' : Frame) (hf : Alike f f') : RelM keep (stepAt c r f) (stepAt c r' f') := by
  obtain ⟨m, o, fr, p⟩ := r
  obtain ⟨m', o', fr', p'⟩ := r'
  simp only [keep, Prod.mk.injEq] at hr
  obtain ⟨rfl, rfl, rfl⟩ := hr
  cases f <;> cases f' <;> try exact (hf : False).elim
  case dom.dom b b' =>
    show (m, fr, p + 16 + (b.size + 15) / 16 * 16) = (m, fr, p + 16 + (b'.size + 15) / 16 * 16)
    rw [show b.size = b'.size from hf]
  case label.label b b' =>
    rw [show b = b' from hf]
    exact relM_map (relM_refl Prod.fst (m.label c b')) fun a a' ha => by simp only [keep, ha]
  case bytes.bytes b b' =>
    show (m, fr, p + 16 + (b.size + 15) / 16 * 16) = (m, fr, p + 16 + (b'.size + 15) / 16 * 16)
    rw [show b.size = b'.size from hf]
  case one.one v v' => exact foldlM_absorb1_alike c [v] [v'] rfl _ _ rfl
  case vec.vec vs vs' => exact foldlM_absorb1_alike c vs vs' hf _ _ rfl
  case sq.sq k n k' n' => exact (rfl : (m, fr, p + 16) = (m, fr, p + 16))

/-- **The fold of the frame's step over alike frames** reads only their shape. -/
theorem foldlM_stepAt_alike (c : Cfg) : ∀ (fs fs' : List Frame) (_ : List.Forall₂ Alike fs fs')
    (r r' : Masker × ByteArray × Array ℕ × ℕ) (_ : keep r = keep r'),
    RelM keep (fs.foldlM (stepAt c) r) (fs'.foldlM (stepAt c) r')
  | [], [], _, _, _, hr => hr
  | [], _ :: _, h, _, _, _ => nomatch h
  | _ :: _, [], h, _, _, _ => nomatch h
  | f :: fs, f' :: fs', h, r, r', hr => by
    obtain ⟨hf, h⟩ := List.forall₂_cons.mp h
    rw [List.foldlM_cons, List.foldlM_cons]
    exact relM_bind (stepAt_alike c r r' hr f f' hf) fun a b _ _ hab => foldlM_stepAt_alike c fs fs' h a b hab

/-- **The frame's round over alike frames** returns the same state and free offsets, or the same error, where the inner
proof has not begun by the round's squeeze. -/
theorem roundFree_alike (c : Cfg) (s : Masker) (fs fs' : List Frame) (k : UInt8) (n : ℕ)
    (hs : s.inner ≠ some INNER_ROUNDS) (hm : ∀ f ∈ fs, midFrame f) (hm' : ∀ f ∈ fs', midFrame f)
    (hk : k = 2 ∨ (k = 1 ∧ n = 1)) (h0 : ¬ (n = 0 ∧ k = 2)) (hn : n < 2 ^ 64) (hA : List.Forall₂ Alike fs fs')
    (hI : ∀ q, fs.foldlM (stepAt c) (s, ByteArray.empty, #[], 0) = .ok q → q.1.inner = none) :
    RelM (fun q : ByteArray × Masker × Array ℕ => q.2)
      (roundFree c s (ByteArray.mk (encs fs ++ (Frame.sq k n).enc).toArray) n)
      (roundFree c s (ByteArray.mk (encs fs' ++ (Frame.sq k n).enc).toArray) n) := by
  rw [roundFree_encs c s fs k n hs hm hk h0 hn, roundFree_encs c s fs' k n hs hm' hk h0 hn]
  refine relM_bind (foldlM_stepAt_alike c fs fs' hA _ _ rfl) fun a b ha _ hab => ?_
  have e1 : a.1 = b.1 := congrArg Prod.fst hab
  have e2 : a.2.2.1 = b.2.2.1 := congrArg (fun q => q.2.1) hab
  have hi : a.1.inner = none := hI a ha
  have hi' : b.1.inner = none := by rw [← e1]; exact hi
  rw [hi, hi']
  show ({ a.1 with owed := some n }, a.2.2.1) = ({ b.1 with owed := some n }, b.2.2.1)
  rw [e1, e2]

end Claims.ZKServer
