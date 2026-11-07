module

public import Proofs.Flock.Claims.ZK.FrameInner
public import Proofs.Flock.Claims.ZK.FrameStop
import all Init.Data.String.Defs
import all Init.Data.Array.Basic

/-!
# The inner proof's `ε` and `β` at the model's state

Past a rep's schedule the loop's frame releases the inner proof's rounds itself (`Frame.innerSched`), its round 1
reading the frame's `ε` and its round 3 its `β`, which `Masker.answer` sets from the coins of the inner proof's rounds 0
and 2. The model's own round there (`Loop.innerOf`) reads them off the history's coins instead. Here the two meet at
the model's masking state:

* `roundFree_keeps`: a round keeps whatever its labels, masked values and squeeze keep. `label_eb`, `absorb_eb`: a
  label and a masked value keep `ε` and `β`, and so a round does (`roundFree_eb`).
* `answer_eb`: coins inside the inner proof set `ε` at its round 0 and `β` at its round 2 (`INNER_BETA`) to their first
  word, and keep them otherwise.
* `run_ownReqs_eb`: along a rep's requests up to the inner proof's round `i`, once past its round 0 `ε` is the first
  word of that round's coins, and once past its round 2 `β` is the first word of that round's.
* **`repState_own_eb`**: so where the frame takes the coins of the inner proof's round `i` at the rep's state after it.
-/

@[expose] public section

namespace Claims.ZKServer.Relay

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker isLabel LABEL_PCS LABEL_ZC claimPoints roundFree INNER_ROUNDS INNER_BETA)
open Ranked

/-- **A round keeps what its labels, masked values and squeeze keep**: where `Masker.label`, `Masker.absorb` and the
squeeze's `owed` keep `g`, a round the frame takes keeps it, whatever its bytes. -/
theorem roundFree_keeps {α : Type} (g : Masker → α) {c : Cfg}
    (hl : ∀ {s : Masker} {l : ByteArray} {q : Masker × ByteArray}, s.label c l = .ok q → g q.1 = g s)
    (ha : ∀ {s : Masker} {v : F128} {q : Masker × F128}, s.absorb c v = .ok q → g q.1 = g s)
    (ho : ∀ (s : Masker) (n : Nat), g { s with owed := some n } = g s)
    {s : Masker} {content : ByteArray} {n : Nat} {q : ByteArray × Masker × Array Nat}
    (h : roundFree c s content n = .ok q) : g q.2.1 = g s := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := forIn_inv (fun x : Option (ByteArray × Masker × Array Nat) × Masker × ByteArray × Array Nat × Nat =>
      g x.2.1 = g s ∧ ∀ v, x.1 = some v → g v.2.1 = g s) _ ?step _ _ _
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
      exact ⟨(hl hx).trans hk, fun _ hv => by cases hv⟩
    · have hy := ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : g r'.1 = g s := by
        refine forIn_inv (fun x : Masker × ByteArray × Array Nat => g x.1 = g s) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha', hy⟩ := bindV_ok hy
          cases hy
          exact (ha ha').trans hx
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha', hy⟩ := bindV_ok hy
          cases hy
          exact (ha ha').trans hx
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
        exact ⟨hk, fun v hv => by cases hv; exact (ho _ _).trans hk⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact (ho _ _).trans hk⟩
    · have hy := ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

/-- **A masked value keeps `ε` and `β`.** -/
theorem absorb_eb {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) :
    (q.1.eps, q.1.beta) = (s.eps, s.beta) := by
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

/-- **A label keeps `ε` and `β`.** -/
theorem label_eb {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (h : s.label c l = .ok q) :
    (q.1.eps, q.1.beta) = (s.eps, s.beta) := by
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

/-- **A round keeps `ε` and `β`**, whatever its bytes. -/
theorem roundFree_eb {c : Cfg} {s : Masker} {content : ByteArray} {n : Nat} {q : ByteArray × Masker × Array Nat}
    (h : roundFree c s content n = .ok q) : q.2.1.eps = s.eps ∧ q.2.1.beta = s.beta := by
  have e := roundFree_keeps (fun m : Masker => (m.eps, m.beta)) label_eb absorb_eb (fun _ _ => rfl) h
  exact ⟨congrArg Prod.fst e, congrArg Prod.snd e⟩

/-- **Coins inside the inner proof set `ε` at its round 0 and `β` at its round 2**, each to the coins' first word, and
keep them otherwise. -/
theorem answer_eb {c : Cfg} {s s' : Masker} {cs : Array F128} {k : Nat} (hs : s.inner = some k)
    (h : s.answer c cs = .ok s') :
    s'.eps = (if k = 0 then cs[0]! else s.eps) ∧ s'.beta = (if k = INNER_BETA then cs[0]! else s.beta) := by
  unfold Masker.answer at h
  simp only [bind, Except.bind, pure, Except.pure, hs] at h
  repeat' split at h
  all_goals (try cases h)
  all_goals
    by_cases hk0 : k = 0
    · subst hk0
      simp_all [INNER_BETA]
    · by_cases hk2 : k = INNER_BETA
      · subst hk2
        simp_all [INNER_BETA]
      · simp_all [INNER_BETA]

open _root_.Flock.Firewall.Frame (ownRound) in
/-- **`ε` and `β` along a rep's requests up to the inner proof's round `i`**: where they neither end the rep nor leave
it owing no coins, once past the inner proof's round 0 `ε` is the first word of its coins (`cr`'s past the schedule),
and once past its round 2 `β` is the first word of its coins (`answer_eb`; each round keeps both, `roundFree_eb`). -/
theorem run_ownReqs_eb (c : Cfg) (prior : Array F128) (sched own : Array (ByteArray × Nat))
    (cr : Array (Array F128))
    (h0 : ∃ s w t n, w.size = 16 ∧ t.size = 64 ∧ own.getD 0 (.empty, 0) = (ownRound c s w t n 0, n)) :
    ∀ i, (_root_.Flock.Firewall.Frame.run c {} (Loop.ownReqs prior sched own cr i)).2 = false →
      (_root_.Flock.Firewall.Frame.run c {} (Loop.ownReqs prior sched own cr i)).1.m.owed ≠ none →
      (0 < i → (_root_.Flock.Firewall.Frame.run c {} (Loop.ownReqs prior sched own cr i)).1.m.eps =
        (cr.getD sched.size #[])[0]!) ∧
      (INNER_BETA < i → (_root_.Flock.Firewall.Frame.run c {} (Loop.ownReqs prior sched own cr i)).1.m.beta =
        (cr.getD (sched.size + INNER_BETA) #[])[0]!)
  | 0, _, _ => ⟨fun h => absurd h (Nat.lt_irrefl 0), fun h => absurd h (Nat.not_lt_zero _)⟩
  | i + 1, h2, ho => by
    rw [Loop.ownReqs_succ] at h2 ho ⊢
    obtain ⟨hb, s1, ha, out, s', fr, hr, -, hm⟩ := Loop.run_snoc_coins_own c {} _ _ _ _ h2 ho
    have hon : (_root_.Flock.Firewall.Frame.run c {} (Loop.ownReqs prior sched own cr i)).1.m.owed ≠ none :=
      fun e => Loop.answer_owed_none _ e ha
    have hI := Loop.run_ownReqs_inner c prior sched own cr h0 i hb hon
    have ih := run_ownReqs_eb c prior sched own cr h0 i hb hon
    obtain ⟨he, hbt⟩ := answer_eb hI ha
    obtain ⟨hke, hkb⟩ := roundFree_eb hr
    rw [hm]
    refine ⟨fun _ => ?_, fun hlt => ?_⟩
    · rw [hke, he]
      by_cases hi0 : i = 0
      · subst hi0
        rw [if_pos rfl, Nat.add_zero]
      · rw [if_neg hi0]
        exact ih.1 (Nat.pos_of_ne_zero hi0)
    · rw [hkb, hbt]
      by_cases hi2 : i = INNER_BETA
      · subst hi2
        rw [if_pos rfl]
      · rw [if_neg hi2]
        exact ih.2 (by unfold INNER_BETA at hlt hi2 ⊢; omega)

/-- **`ε` and `β` where the frame takes the inner proof's round `i`'s coins `cs`** at the rep's state after it
(`repState`, past the schedule `sched` on the rep's coins before the last): `ε` is `cs`'s first word at round 0 and the
first word of round 0's coins past it, and from round 2 on `β` is `cs`'s first word at round 2 and the first word of
round 2's coins past it (`run_ownReqs_eb`, `answer_eb`). -/
theorem repState_own_eb (c c' : Cfg) (t : _root_.Flock.Firewall.Finish.Tab) (rep : Nat) (dom : ByteArray)
    (xs : Array (Array F128)) (hi : Array ByteArray) (cr0 cr : Array (Array F128)) (ns : Nat → Nat) {i : Nat}
    {alg sched : Array (ByteArray × Nat)}
    (hal : _root_.Flock.Firewall.Skeleton.rounds t.st t.sch.k0 (_root_.Flock.Firewall.Finish.flat t.cm0.cap) dom =
      .ok alg)
    (hs : Loop.schedSoFar t rep dom xs hi cr.pop = .ok sched) (hi4 : i < INNER_ROUNDS) {cs : Array F128}
    {m : Masker} (ha : (Loop.repState c t rep dom xs hi cr0 cr ns (sched.size + i + 1)).answer c' cs = .ok m) :
    m.eps = (if i = 0 then cs[0]! else (cr.getD sched.size #[])[0]!) ∧
      (INNER_BETA ≤ i →
        m.beta = (if i = INNER_BETA then cs[0]! else (cr.getD (sched.size + INNER_BETA) #[])[0]!)) := by
  rw [Loop.repState_own_at c t rep dom xs hi cr0 cr ns i hal hs hi4] at ha
  dsimp only at ha
  generalize hR : _root_.Flock.Firewall.Frame.run c {}
    (Loop.ownReqs (if rep = 0 then #[] else (cr0.extract 0 alg.size).foldl (· ++ ·) #[]) sched
      (Loop.ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)) cr i) = R at ha
  obtain ⟨S, b⟩ := R
  cases b with
  | true =>
    simp only [↓reduceIte] at ha
    exact absurd ha (Loop.answer_owed_none cs rfl)
  | false =>
    simp only [Bool.false_eq_true, ↓reduceIte] at ha
    have h0 := (⟨_, _, _, _, Loop.zeros_size 16, Loop.zeros_size 64,
      Loop.ownRounds_getD c (cr.extract sched.size cr.size) (fun j => ns (sched.size + j))
        (show 0 < INNER_ROUNDS by decide)⟩ : ∃ s w t n, w.size = 16 ∧ t.size = 64 ∧
          (Loop.ownRounds c (cr.extract sched.size cr.size) fun j => ns (sched.size + j)).getD 0 (.empty, 0) =
            (_root_.Flock.Firewall.Frame.ownRound c s w t n 0, n))
    have hon : S.m.owed ≠ none := fun e => Loop.answer_owed_none cs e ha
    have hI := Loop.run_ownReqs_inner c _ sched _ cr h0 i (by rw [hR]) (by rw [hR]; exact hon)
    have hE := run_ownReqs_eb c _ sched _ cr h0 i (by rw [hR]) (by rw [hR]; exact hon)
    rw [hR] at hI hE
    obtain ⟨he, hb⟩ := answer_eb hI ha
    refine ⟨?_, fun hle => ?_⟩
    · rw [he]
      by_cases hi0 : i = 0
      · rw [if_pos hi0, if_pos hi0]
      · rw [if_neg hi0, if_neg hi0]
        exact hE.1 (Nat.pos_of_ne_zero hi0)
    · rw [hb]
      by_cases hi2 : i = INNER_BETA
      · rw [if_pos hi2, if_pos hi2]
      · rw [if_neg hi2, if_neg hi2]
        exact hE.2 (by omega)

/-- A coin array's first word, `0` where it has none. -/
theorem getElem!_zero_getD (a : Array F128) : a[0]! = a.getD 0 0 := by
  rw [getElem!_def, Array.getD_eq_getD_getElem?]
  cases a[0]? <;> first | rfl | decide

open _root_.Flock.Firewall.Frame (innerSched ownRound) in
/-- **The frame's own inner round reads `ε` only at round 1 and `β` only at round 3**: at two states alike there,
it is one round. -/
theorem innerSched_congr (c : Cfg) {s s' : Masker} (fin : Array ByteArray) (n : Nat) :
    ∀ {k : Nat}, k < INNER_ROUNDS → (k = 1 → s.eps = s'.eps) → (k = 3 → s.beta = s'.beta) →
      innerSched c s fin n k = innerSched c s' fin n k
  | 0, _, _, _ => by simp only [innerSched, ownRound]
  | 1, _, he, _ => by simp only [innerSched, ownRound, he rfl]
  | 2, _, _, _ => by simp only [innerSched, ownRound]
  | 3, _, _, hb => by simp only [innerSched, ownRound, hb rfl]
  | k + 4, h, _, _ => absurd h (by unfold INNER_ROUNDS; omega)

end Claims.ZKServer.Relay
