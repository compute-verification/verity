module

public import Proofs.Flock.Claims.ZK.FrameRanked
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# A rep's frame stops only where its check fails

The frame's `stopped` is the rep's stop: once set it stays, and only a failed check sets it. The loop reads it as the
session's stop (`Flock.Firewall.Frames.Sessions.stopped`):

* `absorb_stopped`, `label_stopped`, `roundFree_stopped`, `round_stopped`: a masked value, a label and a round keep
  `stopped`, whatever its bytes.
* **`check_stopped`**: after the frame's `check` of a device round, the rep is stopped where it was or where the
  check failed the round (`why`), whose stand-in is the scheduled round of a stopped rep (`check_some`).
* `roundFree_empty`, **`check_empty`**: no device round (`ByteArray.empty`, what the loop hands a stopped session's
  frames) is a round, so the check fails it and the rep is stopped after.
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel LABEL_PCS LABEL_ZC claimPoints)

namespace Ranked

theorem absorb_stopped {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) :
    q.1.stopped = s.stopped := by
  unfold Masker.absorb at h
  cases hon : s.on
  · simp only [hon, Bool.not_false, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl
  · simp only [hon, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at h
    cases hi : c.order[s.pos]? with
    | none => simp only [hi] at h; cases h
    | some i =>
      cases hp : c.h[i]? with
      | none => simp only [hi, hp] at h; cases h
      | some p =>
        simp only [hi, hp, pure, Except.pure, Except.ok.injEq] at h
        subst h
        rfl

theorem label_stopped {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (h : s.label c l = .ok q) :
    q.1.stopped = s.stopped := by
  by_cases hP : isLabel l LABEL_PCS = true
  · rw [eq_of_isLabel hP] at h
    cases hon : s.on
    · have e : s.label c LABEL_PCS.toUTF8 = .ok (s, _root_.Flock.Firewall.Frame.labelFrame LABEL_PCS.toUTF8) := by
        simp [Masker.label, hon]
        rfl
      rw [e] at h
      cases h
      rfl
    · obtain ⟨p1, p2, p3, p4⟩ := pcs_other
      by_cases hpos : s.pos = c.nPiop
      · cases hm : c.mask with
        | none =>
          simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos] at h
          cases h
          rfl
        | some ms =>
          cases hcp : claimPoints c.lay s.coins with
          | error e =>
            simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
            cases h
          | ok P =>
            simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos, hcp] at h
            rw [show ∀ f : Array (Array F128) → V (Masker × ByteArray), (Except.ok P >>= f) = f P from
              fun _ => rfl] at h
            split at h
            · cases h
              rfl
            · cases h
      · cases hm : c.mask <;> simp [Masker.label, hon, hm, isLabel_self, p1, p2, p3, p4, hpos] at h <;> cases h
  by_cases hC : isLabel l _root_.Flock.Zk.LABEL_CLOSE = true
  · rw [eq_of_isLabel hC] at h
    obtain ⟨p1, p2, p3, p4⟩ := close_other
    cases hl0 : s.l0
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0] at h
      cases h
      rfl
    · by_cases hpos : s.pos = c.order.size
      · by_cases hrho : s.rho.size = _root_.Flock.Firewall.Frame.RHO_COINS
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
          rfl
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos, hrho] at h
          cases h
      · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hl0, hpos] at h
        cases h
  by_cases hI : isLabel l _root_.Flock.Zk.LABEL_INNER = true
  · rw [eq_of_isLabel hI] at h
    obtain ⟨p1, p2, p3, p4⟩ := inner_other
    cases hi : s.inner with
    | some k =>
      simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi] at h
      cases h
    | none =>
      cases hd : s.done
      · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd] at h
        cases h
      · cases hon : s.on
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          split at h
          · cases h
          · cases h
            rfl
        · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hi, hd, hon] at h
          cases h
  by_cases hZ : isLabel l LABEL_ZC = true
  · rw [eq_of_isLabel hZ] at h
    obtain ⟨p1, p2, p3, p4⟩ := zc_other
    cases hd : s.done
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      rfl
    · simp [Masker.label, isLabel_self, p1, p2, p3, p4, hd] at h
      cases h
      rfl
  by_cases hO : isLabel l _root_.Flock.Zk.LABEL_OPEN = true
  · rw [eq_of_isLabel hO] at h
    obtain ⟨p1, p2, p3, p4⟩ := open_other
    simp [Masker.label, isLabel_self, p1, p2, p3, p4] at h
    split at h
    · cases h
      rfl
    · cases h
      rfl
  simp only [Bool.not_eq_true] at hP hC hI hZ hO
  rw [label_plain c s l ⟨hP, hC, hI, hZ, hO⟩] at h
  cases h
  rfl

open _root_.Flock.Firewall.Frame (roundFree INNER_ROUNDS) in
/-- **A round keeps `stopped`**, whatever its bytes. -/
theorem roundFree_stopped {c : Cfg} {s : Masker} {content : ByteArray} {n : Nat} {q : ByteArray × Masker × Array Nat}
    (h : roundFree c s content n = .ok q) : q.2.1.stopped = s.stopped := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := forIn_inv (fun x : Option (ByteArray × Masker × Array Nat) × Masker × ByteArray × Array Nat × Nat =>
      x.2.1.stopped = s.stopped ∧ ∀ v, x.1 = some v → v.2.1.stopped = s.stopped) _ ?step _ _ _
    ⟨rfl, fun _ hv => by cases hv⟩ hr
  · obtain ⟨o, m, out, free, pos⟩ := r
    cases o with
    | none => cases h
    | some v => cases h; exact hQ.2 _ rfl
  · intro i x y hx hy
    obtain ⟨o, m, out, free, pos⟩ := x
    obtain ⟨hk, -⟩ := hx
    dsimp only at hy hk
    by_cases h1 : pos + 16 > content.size
    · rw [ite_eq_left h1] at hy
      by_cases h2 : (pos == content.size) = true
      · rw [ite_eq_left h2] at hy
        cases hy
      · rw [ite_eq_right h2] at hy
        cases hy
    rw [ite_eq_right h1] at hy
    generalize (content.get! pos).toNat = op at hy
    rcases op with _ | _ | _ | _ | _ | _ | op
    all_goals simp only [] at hy
    · cases hy
    · have hy := ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · have hy := ite_bind_ok hy
      obtain ⟨x, hx, hy⟩ := bindV_ok hy
      cases hy
      exact ⟨(label_stopped hx).trans hk, fun _ hv => by cases hv⟩
    · have hy := ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : r'.1.stopped = s.stopped := by
        refine forIn_inv (fun x : Masker × ByteArray × Array Nat => x.1.stopped = s.stopped) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact (absorb_stopped ha).trans hx
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact (absorb_stopped ha).trans hx
      exact ⟨hQ', fun _ hv => by cases hv⟩
    · generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h5 : (len == 0 && kind == 2) = true
      · rw [ite_eq_left h5] at hy
        cases hy
        exact ⟨hk, fun _ hv => by cases hv⟩
      rw [ite_eq_right h5] at hy
      by_cases h4 : (pos + 16 != content.size) = true
      · rw [ite_eq_left h4] at hy
        cases hy
      rw [ite_eq_right h4] at hy
      by_cases h6 : (len != n) = true
      · rw [ite_eq_left h6] at hy
        cases hy
      rw [ite_eq_right h6] at hy
      by_cases h7 : (!(kind == 2 || kind == 1 && n == 1)) = true
      · rw [ite_eq_left h7] at hy
        cases hy
      rw [ite_eq_right h7] at hy
      split at hy
      · obtain ⟨u, -, hy⟩ := bindV_ok hy
        cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk⟩
    · have hy := ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

theorem round_stopped {c : Cfg} {s s' : Masker} {content out : ByteArray} {n : Nat}
    (h : _root_.Flock.Firewall.Frame.round c s content n = .ok (out, s')) : s'.stopped = s.stopped := by
  unfold _root_.Flock.Firewall.Frame.round at h
  obtain ⟨⟨o, t, fr⟩, hr, h⟩ := bindV_ok h
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨-, rfl⟩ := h
  exact roundFree_stopped hr

open _root_.Flock.Firewall.Frame (check) in
/-- **The frame's check stops the rep only where it fails**: after it the rep is stopped where it was, or where the
check failed the device round. -/
theorem check_stopped {c : Cfg} {s : Masker} {v r : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker}
    {why : Option String} (h : check c s v r n = .ok (out, s', why)) : s'.stopped = (s.stopped || why.isSome) := by
  cases why with
  | none =>
    rw [round_stopped (check_none h)]
    simp
  | some w =>
    rw [round_stopped (check_some h)]
    simp

open _root_.Flock.Firewall.Frame (roundFree INNER_ROUNDS) in
/-- No device round is no round: it has no squeeze. -/
theorem roundFree_empty (c : Cfg) (s : Masker) (n : Nat) (q : ByteArray × Masker × Array Nat) :
    roundFree c s ByteArray.empty n ≠ .ok q := by
  intro h
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  cases h

open _root_.Flock.Firewall.Frame (check) in
/-- **The frame's check of no device round stops the rep**: it fails, whatever the scheduled round. -/
theorem check_empty {c : Cfg} {s : Masker} {r : ByteArray} {n : Nat} {out : ByteArray} {s' : Masker}
    {why : Option String} (h : check c s ByteArray.empty r n = .ok (out, s', why)) : s'.stopped = true := by
  rw [check_stopped h]
  cases why with
  | none =>
    have hr := check_none h
    unfold _root_.Flock.Firewall.Frame.round at hr
    obtain ⟨q, hq, -⟩ := bindV_ok hr
    exact absurd hq (roundFree_empty c s n q)
  | some w => simp

end Ranked

end Claims.ZKServer
