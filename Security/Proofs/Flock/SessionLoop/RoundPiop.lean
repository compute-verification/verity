import Proofs.Flock.SessionLoop.CutPiopLive
import Proofs.Flock.SessionLoop.PiopState
import Proofs.Flock.SessionLoop.TakenVals
import Proofs.Flock.SessionLoop.RepSrc
import Proofs.Flock.SessionLoop.FinCommits
import Proofs.Flock.SessionLoop.FinRounds

/-!
# A rep's PIOP round past its first, in frames

At a live round of rep stream `2j + r`'s PIOP past its first round, the frame checks the worker's round against the
loop's own scheduled round, the PIOP's frames at zero values (`Sched.rounds_piop`), and the switched session sends the
PIOP's frames at its masked values (`Masked.runCut_piopBody`). This file shows the two are one round in frames
(`LeanRelease.RoundFrames`), so the frame releases the session's message:

* **Alike frames** (`alike_cases`, `plainFrames_of_alike`, `noDoms_of_alike`, `fvals_length_alike`): frames alike to
  ones inside a phase are inside it, and carry as many values. `eq_of_alike_fvals`: alike frames with no domain or
  bytes frame (`fixedFrame`) and the same values are the same.
* **The release's values** (`alike_maskFrames`, `fvals_maskFrames`, `maskVals_getElem?`): the release of frames is
  alike to them, and its values are theirs through the mask, each at its position.
* **The PIOP's frames** (`piopFr_alike`, `fvals_piopFr_getElem?`, `piopFr_eq_maskFrames`): a PIOP round's frames past
  round `0` are alike at any values, round `jj`'s value `k` is the masked value at the absorbed order's position
  `P + k`, `P` the values before it, and so the round's frames are the release of any alike frames whose values, through
  the mask at `P`, are the masked values there.
-/

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

namespace Claims.ZKServer

open _root_.Flock (V F128)
open FlockSoundness.Refine (Frame encs)

/-! ## Alike frames -/

theorem alike_cases : ∀ {f f' : Frame}, Alike f f' →
    (∃ b b', f = .dom b ∧ f' = .dom b') ∨ (∃ b, f = .label b ∧ f' = .label b) ∨
    (∃ b b', f = .bytes b ∧ f' = .bytes b') ∨ (∃ v v', f = .one v ∧ f' = .one v') ∨
    (∃ vs vs', f = .vec vs ∧ f' = .vec vs' ∧ vs.length = vs'.length) ∨ (∃ k n, f = .sq k n ∧ f' = .sq k n)
  | .dom _, .dom _, _ => .inl ⟨_, _, rfl, rfl⟩
  | .label _, .label _, h => by
    simp only [Alike] at h
    subst h
    exact .inr (.inl ⟨_, rfl, rfl⟩)
  | .bytes _, .bytes _, _ => .inr (.inr (.inl ⟨_, _, rfl, rfl⟩))
  | .one _, .one _, _ => .inr (.inr (.inr (.inl ⟨_, _, rfl, rfl⟩)))
  | .vec _, .vec _, h => .inr (.inr (.inr (.inr (.inl ⟨_, _, rfl, rfl, h⟩))))
  | .sq _ _, .sq _ _, h => by
    simp only [Alike] at h
    obtain ⟨rfl, rfl⟩ := h
    exact .inr (.inr (.inr (.inr (.inr ⟨_, _, rfl, rfl⟩))))

theorem plainFrame_of_alike {f f' : Frame} (h : Alike f f') (hp : plainFrame f) : plainFrame f' := by
  rcases alike_cases h with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
    ⟨vs, vs', rfl, rfl, -⟩ | ⟨k, n, rfl, rfl⟩
  · exact True.intro
  · exact hp
  · exact True.intro
  · exact True.intro
  · exact True.intro
  · exact hp

theorem noDom_of_alike {f f' : Frame} (h : Alike f f') (hn : noDom f) : noDom f' := by
  rcases alike_cases h with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
    ⟨vs, vs', rfl, rfl, -⟩ | ⟨k, n, rfl, rfl⟩
  · exact hn
  all_goals exact True.intro

theorem plainFrames_of_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') (hp : ∀ f ∈ fs, plainFrame f) :
    ∀ f ∈ fs', plainFrame f := by
  induction h with
  | nil => intro f hf; cases hf
  | cons ha _ ih =>
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · exact plainFrame_of_alike ha (hp _ (List.mem_cons_self ..))
    · exact ih (fun g hg => hp g (List.mem_cons_of_mem _ hg)) f hf

theorem noDoms_of_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') (hn : ∀ f ∈ fs, noDom f) :
    ∀ f ∈ fs', noDom f := by
  induction h with
  | nil => intro f hf; cases hf
  | cons ha _ ih =>
    intro f hf
    rcases List.mem_cons.1 hf with rfl | hf
    · exact noDom_of_alike ha (hn _ (List.mem_cons_self ..))
    · exact ih (fun g hg => hn g (List.mem_cons_of_mem _ hg)) f hf

theorem fvals_length_alike {fs fs' : List Frame} (h : List.Forall₂ Alike fs fs') :
    (fvals fs').length = (fvals fs).length := by
  induction h with
  | nil => rfl
  | cons ha _ ih =>
    rcases alike_cases ha with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
      ⟨vs, vs', rfl, rfl, hl⟩ | ⟨k, n, rfl, rfl⟩ <;>
      (simp only [fvals, List.length_cons, List.length_append, ih]; omega)

/-- A frame its values fix among the frames alike to it: neither a domain frame nor a bytes frame. -/
def fixedFrame : Frame → Prop
  | .dom _ => False
  | .bytes _ => False
  | _ => True

/-- **Alike frames with the same values are the same**, where neither has a domain or bytes frame. -/
theorem eq_of_alike_fvals {A B : List Frame} (h : List.Forall₂ Alike A B) :
    (∀ f ∈ A, fixedFrame f) → fvals A = fvals B → A = B := by
  induction h with
  | nil => intro _ _; rfl
  | @cons a b l₁ l₂ ha hr ih =>
    intro hf hv
    have hfA : ∀ f ∈ l₁, fixedFrame f := fun g hg => hf g (List.mem_cons_of_mem _ hg)
    have ha' := hf a (List.mem_cons_self ..)
    rcases alike_cases ha with ⟨b, b', rfl, rfl⟩ | ⟨b, rfl, rfl⟩ | ⟨b, b', rfl, rfl⟩ | ⟨v, v', rfl, rfl⟩ |
      ⟨vs, vs', rfl, rfl, hl⟩ | ⟨k, n, rfl, rfl⟩
    · exact (ha' : False).elim
    · simp only [fvals] at hv
      rw [ih hfA hv]
    · exact (ha' : False).elim
    · simp only [fvals, List.cons.injEq] at hv
      rw [hv.1, ih hfA hv.2]
    · simp only [fvals] at hv
      obtain ⟨h1, h2⟩ := List.append_inj hv hl
      rw [h1, ih hfA h2]
    · simp only [fvals] at hv
      rw [ih hfA hv]

/-! ## The release's values -/

theorem alike_maskFrames (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (G : List Frame), (∀ f ∈ G, noDom f) → List.Forall₂ Alike G (maskFrames g p G)
  | _, [], _ => .nil
  | p, f :: G, h => by
    have hG : ∀ f ∈ G, noDom f := fun y hy => h y (List.mem_cons_of_mem _ hy)
    cases f with
    | dom b => exact (h _ (List.mem_cons_self ..) : False).elim
    | label b =>
      simp only [maskFrames]
      exact .cons rfl (alike_maskFrames g p G hG)
    | bytes b =>
      simp only [maskFrames]
      exact .cons rfl (alike_maskFrames g p G hG)
    | one v =>
      simp only [maskFrames]
      exact .cons True.intro (alike_maskFrames g (p + 1) G hG)
    | vec vs =>
      simp only [maskFrames]
      exact .cons (maskVals_length g p vs).symm (alike_maskFrames g (p + vs.length) G hG)
    | sq k n =>
      simp only [maskFrames]
      exact .cons ⟨rfl, rfl⟩ (alike_maskFrames g p G hG)

theorem maskVals_append (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (a b : List F128), maskVals g p (a ++ b) = maskVals g p a ++ maskVals g (p + a.length) b
  | p, [], b => by simp [maskVals]
  | p, v :: a, b => by
    simp only [maskVals, List.cons_append, List.length_cons]
    rw [maskVals_append g (p + 1) a b, show p + 1 + a.length = p + (a.length + 1) by omega]

theorem fvals_maskFrames (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (G : List Frame), fvals (maskFrames g p G) = maskVals g p (fvals G)
  | _, [] => rfl
  | p, .dom _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .label _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .bytes _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .sq _ _ :: G => by
    simp only [maskFrames, fvals]
    exact fvals_maskFrames g p G
  | p, .one v :: G => by
    simp only [maskFrames, fvals, maskVals]
    rw [fvals_maskFrames g (p + 1) G]
  | p, .vec vs :: G => by
    simp only [maskFrames, fvals]
    rw [fvals_maskFrames g (p + vs.length) G, maskVals_append]

theorem maskVals_getElem? (g : ℕ → F128 → F128) :
    ∀ (p : ℕ) (vs : List F128) (i : ℕ), (maskVals g p vs)[i]? = vs[i]?.map (g (p + i))
  | _, [], _ => by simp [maskVals]
  | p, v :: vs, 0 => by simp [maskVals]
  | p, v :: vs, i + 1 => by
    simp only [maskVals, List.getElem?_cons_succ]
    rw [maskVals_getElem? g (p + 1) vs i, show p + 1 + i = p + (i + 1) by omega]

/-! ## The PIOP's frames -/

/-- A PIOP round's frames past round `0` are alike at any values and layout. -/
theorem piopFr_alike (m kLog : ℕ) {digest cap root digest' cap' root' : ByteArray}
    {lay lay' : _root_.Flock.Zk.PadLayout} (mk mk' : ℕ → F128) {jj : ℕ} (h0 : jj ≠ 0) :
    List.Forall₂ Alike (piopFr m kLog digest cap root lay mk jj) (piopFr m kLog digest' cap' root' lay' mk' jj) := by
  unfold piopFr
  rw [ite_eq_right h0, ite_eq_right h0]
  split_ifs <;> simp [Alike]

theorem piopFr_fixed {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout} {mk : ℕ → F128}
    {jj : ℕ} (h : jj ≠ 0) : ∀ f ∈ piopFr m kLog digest cap root lay mk jj, fixedFrame f := by
  intro f hf
  unfold piopFr at hf
  rw [ite_eq_right h] at hf
  split_ifs at hf <;> simp only [List.mem_cons, List.not_mem_nil, or_false] at hf <;>
    rcases hf with rfl | rfl | rfl <;> exact True.intro

theorem fvals_piopFr_length (m kLog : ℕ) {digest cap root digest' cap' root' : ByteArray}
    {lay lay' : _root_.Flock.Zk.PadLayout} (mk mk' : ℕ → F128) (jj : ℕ) :
    (fvals (piopFr m kLog digest cap root lay mk jj)).length =
      (fvals (piopFr m kLog digest' cap' root' lay' mk' jj)).length := by
  unfold piopFr
  split_ifs <;> simp [fvals]

theorem piopV_length_congr (m kLog : ℕ) {digest cap root digest' cap' root' : ByteArray}
    {lay lay' : _root_.Flock.Zk.PadLayout} (mk mk' : ℕ → F128) :
    ∀ a, (piopV m kLog digest cap root lay mk a).length = (piopV m kLog digest' cap' root' lay' mk' a).length
  | 0 => by simp [piopV]
  | a + 1 => by
    rw [piopV_succ, piopV_succ, List.length_append, List.length_append]
    exact congrArg₂ (· + ·) (piopV_length_congr m kLog mk mk' a) (fvals_piopFr_length m kLog mk mk' a)

theorem piopV_prefix (m kLog : ℕ) (digest cap root : ByteArray) (lay : _root_.Flock.Zk.PadLayout) (mk : ℕ → F128)
    {a b : ℕ} (h : a ≤ b) :
    ∃ rest, piopV m kLog digest cap root lay mk b = piopV m kLog digest cap root lay mk a ++ rest := by
  induction b with
  | zero =>
    obtain rfl : a = 0 := by omega
    exact ⟨[], (List.append_nil _).symm⟩
  | succ b ih =>
    rcases Nat.eq_or_lt_of_le h with rfl | h'
    · exact ⟨[], (List.append_nil _).symm⟩
    · obtain ⟨rest, hr⟩ := ih (by omega)
      exact ⟨rest ++ fvals (piopFr m kLog digest cap root lay mk b), by rw [piopV_succ, hr, List.append_assoc]⟩

/-- **A PIOP round's value `k`** is the masked value at the absorbed order's pad of position `P + k`, `P` the values the
rounds before it carry. -/
theorem fvals_piopFr_getElem? {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout}
    {mk : ℕ → F128} (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6) {jj P k : ℕ}
    (hj : jj < piopNR m kLog) (hP : (piopV m kLog digest cap root lay mk jj).length = P)
    (hk' : k < (fvals (piopFr m kLog digest cap root lay mk jj)).length) :
    P + k < lay.fa + 2 + (2 * lay.nLc + 64) ∧
      (fvals (piopFr m kLog digest cap root lay mk jj))[k]? =
        some (mk (if P + k < lay.fa + 2 then P + k else P + k + 1)) := by
  obtain ⟨rest, hr⟩ := piopV_prefix m kLog digest cap root lay mk (show jj + 1 ≤ piopNR m kLog by omega)
  rw [piopV_all m kLog digest cap root lay mk hm hk hn hl, piopV_succ, List.append_assoc] at hr
  have hlen := congrArg List.length hr
  simp only [List.length_map, List.length_append, List.length_range, List.length_range', hP] at hlen
  have hlt : P + k < lay.fa + 2 + (2 * lay.nLc + 64) := by omega
  refine ⟨hlt, ?_⟩
  have e := congrArg (·[P + k]?) hr
  rw [List.getElem?_append_right (by omega), hP, Nat.add_sub_cancel_left, List.getElem?_append_left hk'] at e
  rw [← e, List.getElem?_map]
  split_ifs with h1
  · rw [List.getElem?_append_left (by simpa using h1), List.getElem?_range h1]
    rfl
  · rw [List.getElem?_append_right (by simp only [List.length_range]; omega), List.length_range,
      List.range'_eq_map_range, List.getElem?_map, List.getElem?_range (by omega)]
    exact congrArg (fun v => some (mk v)) (show lay.fa + 3 + (P + k - (lay.fa + 2)) = P + k + 1 by omega)

/-- **A PIOP round's frames are the release of alike frames** whose values, through the mask `g` from position `P` on,
are the masked values at the absorbed order's pads. -/
theorem piopFr_eq_maskFrames {m kLog : ℕ} {digest cap root : ByteArray} {lay : _root_.Flock.Zk.PadLayout}
    {mk : ℕ → F128} (hm : 14 ≤ m) (hk : 6 ≤ kLog) (hn : lay.nMlv = m - 6) (hl : lay.nLc = kLog - 6) {jj P : ℕ}
    (h0 : jj ≠ 0) (hj : jj < piopNR m kLog) (hP : (piopV m kLog digest cap root lay mk jj).length = P)
    (g : ℕ → F128 → F128) {G : List Frame} (hG : List.Forall₂ Alike (piopFr m kLog digest cap root lay mk jj) G)
    (hv : ∀ k (hkG : k < (fvals G).length),
      mk (if P + k < lay.fa + 2 then P + k else P + k + 1) = g (P + k) ((fvals G)[k]'hkG)) :
    piopFr m kLog digest cap root lay mk jj = maskFrames g P G := by
  have hnd := noDoms_of_alike hG (piopFr_noDom h0)
  have hA := Sched.forall₂_alike_trans hG (alike_maskFrames g P G hnd)
  refine eq_of_alike_fvals hA (piopFr_fixed h0) ?_
  rw [fvals_maskFrames]
  apply List.ext_getElem?
  intro k
  rw [maskVals_getElem?]
  have hlen := fvals_length_alike hG
  rcases Nat.lt_or_ge k (fvals G).length with hk' | hk'
  · obtain ⟨-, e⟩ := fvals_piopFr_getElem? hm hk hn hl hj hP (k := k) (by omega)
    rw [e, List.getElem?_eq_getElem hk', Option.map_some, hv k hk']
  · rw [List.getElem?_eq_none hk', List.getElem?_eq_none (by omega)]
    rfl

end Claims.ZKServer
