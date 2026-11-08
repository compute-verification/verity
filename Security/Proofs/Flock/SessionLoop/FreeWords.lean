module

public import Proofs.Flock.Claims.ZK.FrameInner
public import Proofs.Flock.Claims.ZK.FrameRanked
public import Proofs.Flock.Claims.ZK.FrameRounds
public import Proofs.Flock.Claims.ZK.FrameState

/-!
# The words a round frees, and what the frame's state after it reads

`Frame.roundFree` returns, beside a round's release and the frame's state after it, the offsets of the 16-byte words the
round absorbs in the algebraic phase (its free words), which a device round may carry at any value:

* **The positions** (`roundFree_pos`): the frame's position after a round is past the words it freed, each freed at a
  position below `n_piop`.
* **The free words alone differ** (`roundFree_agree`): at alike configurations (`CfgAlike`) and from states equal but
  for `stopped`, two rounds of one size that agree outside the words the first frees free the same words and leave
  states equal but for `stopped`.
* **The check** (`check_unstop`): the state the frame's `check` returns on any device round, of a stopped rep or not,
  is the one the scheduled round leaves, but for `stopped`.
-/

@[expose] public section

namespace Claims.ZKServer

open _root_.Flock (V F128)
open _root_.Flock.Firewall.Frame (Cfg Masker INNER_ROUNDS roundFree)

namespace Free

/-- **A label leaves the frame's position.** -/
theorem label_pos {c : Cfg} {s : Masker} {l : ByteArray} {q : Masker × ByteArray} (h : s.label c l = .ok q) :
    q.1.pos = s.pos := by
  have e : (s.label c l).map (fun q => q.1.pos) = (s.label c l).map (fun _ => s.pos) := by
    unfold Masker.label
    cases hm : c.mask <;> simp only [apply_ite (Except.map _), exMap_bind, exMap_pure]
  rw [h] at e
  exact Except.ok.inj e

/-- **An absorbed value moves the frame's position by one in a masked phase**, by none outside one. -/
theorem absorb_pos {c : Cfg} {s : Masker} {v : F128} {q : Masker × F128} (h : s.absorb c v = .ok q) :
    q.1.pos = if s.on then s.pos + 1 else s.pos := by
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

/-- **Taking coins leaves the frame's position.** -/
theorem answer_pos {cfg : Cfg} {s s' : Masker} {cs : Array F128} (h : s.answer cfg cs = .ok s') : s'.pos = s.pos := by
  cases hi : s.inner with
  | none =>
    cases hl : s.l0
    · rw [answer_alg cfg s cs hi hl h]
    · unfold Masker.answer at h
      cases ho : s.owed with
      | none => rw [ho] at h; cases h
      | some n =>
        by_cases hc : (cs.size != n) = true
        · simp only [ho, hc, ↓reduceIte] at h
          cases h
        simp only [ho, hc, Bool.false_eq_true, ↓reduceIte, hi, hl] at h
        by_cases hp : (s.pos != cfg.order.size) = true
        · simp only [hp, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h
          subst h
          rfl
        · simp only [hp, Bool.false_eq_true, ↓reduceIte] at h
          obtain ⟨r, -, h⟩ := bindV_ok h
          simp only [pure, Except.pure, Except.ok.injEq] at h
          subst h
          rfl
  | some k =>
    unfold Masker.answer at h
    cases ho : s.owed with
    | none => rw [ho] at h; cases h
    | some n =>
      by_cases hc : (cs.size != n) = true
      · simp only [ho, hc, ↓reduceIte] at h
        cases h
      simp only [ho, hc, Bool.false_eq_true, ↓reduceIte, hi] at h
      obtain ⟨u, -, h⟩ := bindV_ok h
      simp only [pure, Except.pure, Except.ok.injEq] at h
      subst h
      split <;> (try split) <;> rfl

/-- The frame's position `m.pos` is past the `free.size` words freed since `s`, each at a position below `n_piop`. -/
def PosInv (c : Cfg) (s m : Masker) (free : Array ℕ) : Prop :=
  s.pos + free.size ≤ m.pos ∧ ∀ k < free.size, s.pos + k < c.nPiop

theorem posInv_absorb {c : Cfg} {s m : Masker} {free : Array ℕ} {v : F128} {q : Masker × F128}
    (hp : PosInv c s m free) (h : m.absorb c v = .ok q) : PosInv c s q.1 free := by
  have e := absorb_pos h
  refine ⟨?_, hp.2⟩
  have := hp.1
  split at e <;> omega

theorem posInv_push {c : Cfg} {s m : Masker} {free : Array ℕ} {v : F128} {q : Masker × F128} (e : ℕ)
    (hp : PosInv c s m free) (hon : m.on = true) (hlt : m.pos < c.nPiop) (h : m.absorb c v = .ok q) :
    PosInv c s q.1 (free.push e) := by
  have e' := absorb_pos h
  rw [hon, ite_eq_left rfl] at e'
  obtain ⟨h1, h2⟩ := hp
  refine ⟨by rw [Array.size_push, e']; omega, fun k hk => ?_⟩
  rw [Array.size_push] at hk
  rcases Nat.lt_or_ge k free.size with hk' | hk'
  · exact h2 k hk'
  · omega

/-- **A round's free words sit at the frame's next positions**: the frame's position after the round is past the words
it freed, each freed at a position below `n_piop`. -/
theorem roundFree_pos {c : Cfg} {s : Masker} {content : ByteArray} {n : Nat} {q : ByteArray × Masker × Array Nat}
    (h : roundFree c s content n = .ok q) :
    s.pos + q.2.2.size ≤ q.2.1.pos ∧ ∀ k < q.2.2.size, s.pos + k < c.nPiop := by
  unfold roundFree at h
  by_cases hI : (s.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h
    cases h
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h
  obtain ⟨r, hr, h⟩ := bindV_ok h
  have hQ := Ranked.forIn_inv (fun x : Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ =>
      PosInv c s x.2.1 x.2.2.2.1 ∧ ∀ v, x.1 = some v → PosInv c s v.2.1 v.2.2) _ ?step _ _ _
    ⟨⟨by simp, fun k hk => by simp at hk⟩, fun _ hv => by cases hv⟩ hr
  · obtain ⟨o, m, out, free, pos⟩ := r
    cases o with
    | none => cases h
    | some v => cases h; exact hQ.2 _ rfl
  · intro i x y hx hy
    obtain ⟨o, m, out, free, pos⟩ := x
    obtain ⟨hk, -⟩ := hx
    dsimp only at hy
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
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · have hy := Ranked.ite_bind_ok hy
      obtain ⟨x, hx, hy⟩ := bindV_ok hy
      cases hy
      refine ⟨?_, fun _ hv => by cases hv⟩
      obtain ⟨h1, h2⟩ := hk
      exact ⟨by rw [label_pos hx]; exact h1, h2⟩
    · have hy := Ranked.ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : PosInv c s r'.1 r'.2.2 := by
        refine Ranked.forIn_inv (fun x : Masker × ByteArray × Array ℕ => PosInv c s x.1 x.2.2) _ ?_ _ _ _ hk hr'
        intro i x y hx hy
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          simp only [Bool.and_eq_true, decide_eq_true_eq] at h8
          exact posInv_push _ hx h8.1 h8.2 ha
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact posInv_absorb hx ha
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
      have hk' : PosInv c s { m with owed := some n } free := hk
      split at hy
      · obtain ⟨u, -, hy⟩ := bindV_ok hy
        cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk'⟩
      · cases hy
        exact ⟨hk, fun v hv => by cases hv; exact hk'⟩
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨hk, fun _ hv => by cases hv⟩
    · cases hy

/-- `Ranked.forIn_inv` with each step's index in the loop's list. -/
theorem forIn_inv_mem {τ : Type} (Q : τ → Prop) (g : ℕ → τ → V (ForInStep τ)) :
    ∀ (l : List ℕ), (∀ i ∈ l, ∀ x y, Q x → g i x = .ok y → Q (match y with | .done b => b | .yield b => b)) →
      ∀ (x y : τ), Q x → forIn l x g = .ok y → Q y
  | [], _, x, y, hx, h => by cases h; exact hx
  | a :: l, hg, x, y, hx, h => by
    rw [List.forIn_cons] at h
    cases hga : g a x with
    | error e => rw [hga] at h; cases h
    | ok st =>
      rw [hga] at h
      cases st with
      | done b => cases h; exact hg a (List.mem_cons_self ..) x _ hx hga
      | yield b =>
        exact forIn_inv_mem Q g l (fun i hi => hg i (List.mem_cons_of_mem _ hi)) b y
          (hg a (List.mem_cons_self ..) x _ hx hga) h

/-- Two steps alike in `R`: both go on, or both return, at states alike in `R`. -/
def StepRel {τ₁ τ₂ : Type} (R : τ₁ → τ₂ → Prop) : ForInStep τ₁ → ForInStep τ₂ → Prop
  | .yield a, .yield b => R a b
  | .done a, .done b => R a b
  | _, _ => False

/-- **Two loops alike step by step end alike**: where each pair of steps from states alike in `R` is alike, at the
first loop's run past the step (its rest and its end, which `Rest` describes), both loops end at states alike in `R`. -/
theorem forIn_rel {τ₁ τ₂ : Type} (R : τ₁ → τ₂ → Prop) (Rest : τ₁ → Prop)
    (g₁ : ℕ → τ₁ → V (ForInStep τ₁)) (g₂ : ℕ → τ₂ → V (ForInStep τ₂))
    (hg : ∀ (i : ℕ) (l : List ℕ) (x₁ : τ₁) (x₂ : τ₂) (y₁ : ForInStep τ₁) (y₂ : ForInStep τ₂) (z₁ : τ₁),
      R x₁ x₂ → Rest z₁ → g₁ i x₁ = .ok y₁ → g₂ i x₂ = .ok y₂ → forIn (i :: l) x₁ g₁ = .ok z₁ → StepRel R y₁ y₂) :
    ∀ (l : List ℕ) (x₁ : τ₁) (x₂ : τ₂) (z₁ : τ₁) (z₂ : τ₂), R x₁ x₂ → Rest z₁ →
      forIn l x₁ g₁ = .ok z₁ → forIn l x₂ g₂ = .ok z₂ → R z₁ z₂
  | [], x₁, x₂, z₁, z₂, hR, _, h₁, h₂ => by cases h₁; cases h₂; exact hR
  | i :: l, x₁, x₂, z₁, z₂, hR, hZ, h₁, h₂ => by
    have h₁' := h₁
    rw [List.forIn_cons] at h₁ h₂
    cases hg₁ : g₁ i x₁ with
    | error e => rw [hg₁] at h₁; cases h₁
    | ok y₁ =>
      cases hg₂ : g₂ i x₂ with
      | error e => rw [hg₂] at h₂; cases h₂
      | ok y₂ =>
        have hs := hg i l x₁ x₂ y₁ y₂ z₁ hR hZ hg₁ hg₂ h₁'
        rw [hg₁] at h₁
        rw [hg₂] at h₂
        cases y₁ with
        | done a =>
          cases y₂ with
          | done b => cases h₁; cases h₂; exact hs
          | yield b => exact (hs : False).elim
        | yield a =>
          cases y₂ with
          | done b => exact (hs : False).elim
          | yield b => exact forIn_rel R Rest g₁ g₂ hg l a b z₁ z₂ hs hZ h₁ h₂

/-- The state of `roundFree`'s loop: the round's return, once it returned, the frame's state, the round's release so
far, the free offsets so far and the position in the round. -/
abbrev St : Type := Option (ByteArray × Masker × Array ℕ) × Masker × ByteArray × Array ℕ × ℕ

/-- **One step of the round's loop**: on, past the frame header it read (16 bytes on), with the offsets it freed
appended, each past that header and before the step's end; or returned, at the offsets so far. -/
def GrowStep (x : St) : ForInStep St → Prop
  | .yield b => b.1 = none ∧ x.2.2.2.2 + 16 ≤ b.2.2.2.2 ∧
      ∃ L : Array ℕ, b.2.2.2.1 = x.2.2.2.1 ++ L ∧ ∀ e ∈ L, x.2.2.2.2 + 16 ≤ e ∧ e + 16 ≤ b.2.2.2.2
  | .done b => (∃ v, b.1 = some v ∧ v.2.2 = x.2.2.2.1) ∧ x.2.2.2.2 + 16 ≤ b.2.2.2.2 ∧ b.2.2.2.1 = x.2.2.2.1

/-- Every step of the loop body `g` grows (`GrowStep`). -/
def Grows (g : ℕ → St → V (ForInStep St)) : Prop := ∀ i x y, g i x = .ok y → GrowStep x y

/-- `Grows g`, with `g` read off a run of the loop. -/
theorem grows_at {l : List ℕ} {x z : St} {g : ℕ → St → V (ForInStep St)} (_ : forIn l x g = .ok z) (hG : Grows g) :
    Grows g := hG

/-- **The offsets a round's loop frees from a state on** are past that state's frame header. -/
theorem suffix_of_grows {g : ℕ → St → V (ForInStep St)} (hG : Grows g) :
    ∀ (l : List ℕ) (x z : St), x.1 = none → forIn l x g = .ok z → ∀ v, z.1 = some v →
      ∃ L : Array ℕ, v.2.2 = x.2.2.2.1 ++ L ∧ ∀ e ∈ L, x.2.2.2.2 + 16 ≤ e
  | [], x, z, hx, h, v, hv => by
    cases h
    rw [hx] at hv
    cases hv
  | a :: l, x, z, _, h, v, hv => by
    rw [List.forIn_cons] at h
    cases hga : g a x with
    | error e => rw [hga] at h; cases h
    | ok st =>
      rw [hga] at h
      have hs := hG a x st hga
      cases st with
      | done b =>
        cases h
        obtain ⟨⟨v', hv', hvf⟩, -, -⟩ := hs
        rw [hv'] at hv
        cases hv
        exact ⟨#[], by rw [hvf, Array.append_empty], fun e he => by simp at he⟩
      | yield b =>
        obtain ⟨hb, hpos, L₁, hL₁, hL₁e⟩ := hs
        obtain ⟨L₂, hL₂, hL₂e⟩ := suffix_of_grows hG l b z hb h v hv
        refine ⟨L₁ ++ L₂, by rw [hL₂, hL₁, Array.append_assoc], fun e he => ?_⟩
        rcases Array.mem_append.1 he with he | he
        · exact (hL₁e e he).1
        · have := hL₂e e he
          omega

/-- `suffix_of_grows` from a step of the loop, whatever that step's state returned. -/
theorem suffix_cons {g : ℕ → St → V (ForInStep St)} (hG : Grows g) (a : ℕ) (l : List ℕ) (x z : St)
    (h : forIn (a :: l) x g = .ok z) (v : ByteArray × Masker × Array ℕ) (hv : z.1 = some v) :
    ∃ L : Array ℕ, v.2.2 = x.2.2.2.1 ++ L ∧ ∀ e ∈ L, x.2.2.2.2 + 16 ≤ e := by
  rw [List.forIn_cons] at h
  cases hga : g a x with
  | error e => rw [hga] at h; cases h
  | ok st =>
    rw [hga] at h
    have hs := hG a x st hga
    cases st with
    | done b =>
      cases h
      obtain ⟨⟨v', hv', hvf⟩, -, -⟩ := hs
      rw [hv'] at hv
      cases hv
      exact ⟨#[], by rw [hvf, Array.append_empty], fun e he => by simp at he⟩
    | yield b =>
      obtain ⟨hb, hpos, L₁, hL₁, hL₁e⟩ := hs
      obtain ⟨L₂, hL₂, hL₂e⟩ := suffix_of_grows hG l b z hb h v hv
      refine ⟨L₁ ++ L₂, by rw [hL₂, hL₁, Array.append_assoc], fun e he => ?_⟩
      rcases Array.mem_append.1 he with he | he
      · exact (hL₁e e he).1
      · have := hL₂e e he
        omega

theorem inFree_false {free : Array ℕ} {i : ℕ} (h : ∀ f ∈ free, i < f ∨ f + 16 ≤ i) :
    _root_.Flock.Firewall.Frame.inFree free i = false := by
  unfold _root_.Flock.Firewall.Frame.inFree
  rw [Array.any_eq_false]
  intro k hk
  rcases h free[k] (Array.getElem_mem hk) with h' | h' <;> simp <;> omega

theorem get!_of_lt {D : ByteArray} {i : ℕ} (h : i < D.size) : D.get! i = D.data[i] := by
  simp [ByteArray.get!, h]

theorem u64OfLE_congr {d d' : ByteArray} {o : ℕ} (h : ∀ i < 8, d.get! (o + i) = d'.get! (o + i)) :
    _root_.Flock.u64OfLE d o = _root_.Flock.u64OfLE d' o := by
  rw [FlockSoundness.Discharge.ZkView.u64OfLE_eq, FlockSoundness.Discharge.ZkView.u64OfLE_eq]
  refine List.foldl_ext _ _ _ fun a b hb => ?_
  rw [h b (List.mem_range'_1.1 hb).2]

theorem extract_congr {D D' : ByteArray} (hsz : D.size = D'.size) {a b : ℕ} (hb : b ≤ D.size)
    (h : ∀ i, a ≤ i → i < b → D.get! i = D'.get! i) : D.extract a b = D'.extract a b := by
  apply ByteArray.ext
  rw [ByteArray.data_extract, ByteArray.data_extract]
  refine Array.ext (by simp only [Array.size_extract, ByteArray.size] at hsz ⊢; rw [hsz]) fun i h1 h2 => ?_
  rw [Array.getElem_extract, Array.getElem_extract]
  simp only [Array.size_extract] at h1
  have hi : a + i < D.size := by simp only [ByteArray.size] at hb ⊢; omega
  have hi' : a + i < D'.size := by rw [← hsz]; exact hi
  have e := h (a + i) (by omega) (by simp only [ByteArray.size] at hb; omega)
  rw [get!_of_lt hi, get!_of_lt hi'] at e
  exact e

/-- Two returns alike: none yet, or both, with the same free offsets and states equal but for `stopped`. -/
def OptRel : Option (ByteArray × Masker × Array ℕ) → Option (ByteArray × Masker × Array ℕ) → Prop
  | none, none => True
  | some v₁, some v₂ => v₁.2.2 = v₂.2.2 ∧ unstop v₁.2.1 = unstop v₂.2.1
  | _, _ => False

/-- Two states of `roundFree`'s loop alike: one position, the same free offsets, frames equal but for `stopped`, and
alike returns. -/
def Rel0 (x₁ x₂ : St) : Prop :=
  x₁.2.2.2.2 = x₂.2.2.2.2 ∧ x₁.2.2.2.1 = x₂.2.2.2.1 ∧ unstop x₁.2.1 = unstop x₂.2.1 ∧ OptRel x₁.1 x₂.1

/-- A state's free offsets lie behind its position. -/
def Fwd (x : St) : Prop := ∀ e ∈ x.2.2.2.1, e + 16 ≤ x.2.2.2.2

theorem fwd_step {x : St} {y : ForInStep St} (hx : Fwd x) (h : GrowStep x y) :
    Fwd (match y with | .done b => b | .yield b => b) := by
  cases y with
  | yield b =>
    obtain ⟨-, hp, L, hL, hLe⟩ := h
    intro e he
    dsimp only at he ⊢
    rw [hL] at he
    rcases Array.mem_append.1 he with he | he
    · have := hx e he
      omega
    · exact (hLe e he).2
  | done b =>
    obtain ⟨-, hp, hf⟩ := h
    intro e he
    dsimp only at he ⊢
    rw [hf] at he
    have := hx e he
    omega

/-- **Two runs of a round's loop alike step by step end alike** (`forIn_rel`), the first's free offsets behind its
position at every step. -/
theorem forIn_rel_grows {g₁ g₂ : ℕ → St → V (ForInStep St)} (Rest : St → Prop) {l : List ℕ} {x₁ x₂ z₁ z₂ : St}
    (h₁ : forIn l x₁ g₁ = .ok z₁) (h₂ : forIn l x₂ g₂ = .ok z₂) (hG : Grows g₁)
    (hg : ∀ (i : ℕ) (l : List ℕ) (x₁ x₂ : St) (y₁ y₂ : ForInStep St) (z₁ : St),
      Rel0 x₁ x₂ → Fwd x₁ → Rest z₁ → g₁ i x₁ = .ok y₁ → g₂ i x₂ = .ok y₂ → forIn (i :: l) x₁ g₁ = .ok z₁ →
        StepRel Rel0 y₁ y₂)
    (hR : Rel0 x₁ x₂) (hF : Fwd x₁) (hZ : Rest z₁) : Rel0 z₁ z₂ :=
  (forIn_rel (fun a b => Rel0 a b ∧ Fwd a) Rest g₁ g₂ (fun i l x₁ x₂ y₁ y₂ z₁ hR hZ hy₁ hy₂ hrun => by
    have hs := hg i l x₁ x₂ y₁ y₂ z₁ hR.1 hR.2 hZ hy₁ hy₂ hrun
    have hf := fwd_step hR.2 (hG i x₁ y₁ hy₁)
    cases y₁ <;> cases y₂ <;> first | exact (hs : False).elim | exact ⟨hs, hf⟩) l x₁ x₂ z₁ z₂ ⟨hR, hF⟩ hZ h₁ h₂).1

theorem forIn_cons_yield {τ : Type} {g : ℕ → τ → V (ForInStep τ)} {i : ℕ} {l : List ℕ} {x y z : τ}
    (h : forIn (i :: l) x g = .ok z) (hy : g i x = .ok (.yield y)) : forIn l y g = .ok z := by
  rw [List.forIn_cons, hy] at h
  exact h

theorem inFree_split {free L : Array ℕ} {a b i : ℕ} (hf : ∀ e ∈ free, e + 16 ≤ a) (hL : ∀ e ∈ L, b ≤ e)
    (ha : a ≤ i) (hb : i < b) : _root_.Flock.Firewall.Frame.inFree (free ++ L) i = false :=
  inFree_false fun f hf' => by
    rcases Array.mem_append.1 hf' with h | h
    · right
      have := hf f h
      omega
    · left
      have := hL f h
      omega

theorem absorb_unstop {c c' : Cfg} (hc : CfgAlike c c') {s₁ s₂ : Masker} (hs : unstop s₁ = unstop s₂)
    {v v' : F128} {q₁ q₂ : Masker × F128} (h₁ : s₁.absorb c v = .ok q₁) (h₂ : s₂.absorb c' v' = .ok q₂) :
    unstop q₁.1 = unstop q₂.1 := by
  have hr := absorb_alike hc s₁ s₂.stopped v v'
  rw [← eq_unstop hs, h₁, h₂] at hr
  exact hr

theorem ite_bind_ok' {α β : Type} {p : Prop} [Decidable p] {e : String} {a : α} {f : α → V β} {y : β}
    (h : ((if p then throw e else pure a) >>= f) = .ok y) : ¬ p ∧ f a = .ok y := by
  by_cases hp : p
  · rw [ite_eq_left hp] at h
    cases h
  · rw [ite_eq_right hp] at h
    exact ⟨hp, h⟩

open _root_.Flock.Firewall.Frame (inFree) in
/-- **Two runs of a round alike outside its free words end alike**: at alike configurations, from states equal but
for `stopped`, on two rounds of one size that agree outside the words the first run frees, both runs free the same
offsets and end at states equal but for `stopped`. -/
theorem roundFree_agree {c c' : Cfg} (hc : CfgAlike c c') {s₁ s₂ : Masker} (hs : unstop s₁ = unstop s₂)
    {content content' : ByteArray} {n : ℕ} (hsz : content.size = content'.size)
    {q₁ q₂ : ByteArray × Masker × Array ℕ}
    (h₁ : roundFree c s₁ content n = .ok q₁) (h₂ : roundFree c' s₂ content' n = .ok q₂)
    (hX : ∀ i < content.size, inFree q₁.2.2 i = false → content.get! i = content'.get! i) :
    q₂.2.2 = q₁.2.2 ∧ unstop q₁.2.1 = unstop q₂.2.1 := by
  have hin : s₂.inner = s₁.inner := (congrArg Masker.inner hs).symm
  unfold roundFree at h₁ h₂
  by_cases hI : (s₁.inner == some INNER_ROUNDS) = true
  · simp only [hI, ↓reduceIte] at h₁
    cases h₁
  have hI' : ¬ (s₂.inner == some INNER_ROUNDS) = true := by rw [hin]; exact hI
  simp only [hI, Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h₁
  simp only [hI', Bool.false_eq_true, ↓reduceIte, Std.Legacy.Range.forIn_eq_forIn_range', Std.Legacy.Range.size] at h₂
  rw [← hsz] at h₂
  obtain ⟨r₁, hr₁, h₁⟩ := bindV_ok h₁
  obtain ⟨r₂, hr₂, h₂⟩ := bindV_ok h₂
  have hz₁ : r₁.1 = some q₁ := by
    obtain ⟨o, _, _, _, _⟩ := r₁
    cases o with
    | none => cases h₁
    | some v => cases h₁; rfl
  have hz₂ : r₂.1 = some q₂ := by
    obtain ⟨o, _, _, _, _⟩ := r₂
    cases o with
    | none => cases h₂
    | some v => cases h₂; rfl
  have hG := grows_at hr₁ ?grow
  · have hR := forIn_rel_grows (fun z : St => z.1 = some q₁) hr₁ hr₂ hG ?step (by exact ⟨rfl, rfl, hs, trivial⟩)
      (by intro e he; simp at he) hz₁
    · obtain ⟨-, -, -, hO⟩ := hR
      rw [hz₁, hz₂] at hO
      obtain ⟨e1, e2⟩ := hO
      exact ⟨e1.symm, e2⟩
    · intro i l x₁ x₂ y₁ y₂ z₁ hR hF hZ hy₁ hy₂ hrun
      have hy₁₀ := hy₁
      obtain ⟨o₁, m₁, out₁, free₁, pos₁⟩ := x₁
      obtain ⟨o₂, m₂, out₂, free₂, pos₂⟩ := x₂
      obtain ⟨hp, hf, hm, -⟩ := hR
      dsimp only at hp hf hm hy₁ hy₂
      subst hp hf
      obtain ⟨L, hL, hLe⟩ := suffix_cons hG i l _ z₁ hrun q₁ hZ
      dsimp only at hL hLe
      by_cases h1 : pos₁ + 16 > content.size
      · rw [ite_eq_left h1] at hy₁
        by_cases h2 : (pos₁ == content.size) = true
        · rw [ite_eq_left h2] at hy₁
          cases hy₁
        · rw [ite_eq_right h2] at hy₁
          cases hy₁
      rw [ite_eq_right h1] at hy₁ hy₂
      have hh : ∀ b < 16, content.get! (pos₁ + b) = content'.get! (pos₁ + b) := fun b hb =>
        hX _ (by omega) (by
          rw [hL]
          exact inFree_split (free := free₁) (a := pos₁) hF hLe (by omega) (by omega))
      have e0 : content.get! pos₁ = content'.get! pos₁ := hh 0 (by omega)
      have e1 : content.get! (pos₁ + 1) = content'.get! (pos₁ + 1) := hh 1 (by omega)
      have e8 : _root_.Flock.u64OfLE content (pos₁ + 8) = _root_.Flock.u64OfLE content' (pos₁ + 8) :=
        u64OfLE_congr fun j hj => by rw [Nat.add_assoc]; exact hh (8 + j) (by omega)
      rw [← e0, ← e1, ← e8] at hy₂
      generalize (content.get! pos₁).toNat = op at hy₁ hy₂
      rcases op with _ | _ | _ | _ | _ | _ | op
      all_goals simp only [] at hy₁ hy₂
      · cases hy₁
      · have hy₁ := Ranked.ite_bind_ok hy₁
        have hy₂ := Ranked.ite_bind_ok hy₂
        cases hy₁
        cases hy₂
        exact ⟨rfl, rfl, hm, trivial⟩
      · generalize (_root_.Flock.u64OfLE content (pos₁ + 8)).toNat = len at hy₁ hy₂
        obtain ⟨hb, hy₁⟩ := ite_bind_ok' hy₁
        have hy₂ := Ranked.ite_bind_ok hy₂
        obtain ⟨x, hx, hy₁⟩ := bindV_ok hy₁
        obtain ⟨x', hx', hy₂⟩ := bindV_ok hy₂
        cases hy₁
        cases hy₂
        obtain ⟨L', hL', hL'e⟩ := suffix_of_grows hG l _ z₁ rfl (forIn_cons_yield hrun hy₁₀) q₁ hZ
        dsimp only at hL' hL'e
        have he := extract_congr hsz (a := pos₁ + 16) (b := pos₁ + 16 + len) (by omega) fun j hj1 hj2 =>
          hX _ (by omega) (by
            rw [hL']
            exact inFree_split (free := free₁) (a := pos₁) hF hL'e (by omega) (by omega))
        rw [← he] at hx'
        have hr := label_alike hc m₁ m₂.stopped (content.extract (pos₁ + 16) (pos₁ + 16 + len))
        rw [← eq_unstop hm, hx, hx'] at hr
        exact ⟨rfl, rfl, hr, trivial⟩
      · have hy₁ := Ranked.ite_bind_ok hy₁
        have hy₂ := Ranked.ite_bind_ok hy₂
        generalize (_root_.Flock.u64OfLE content (pos₁ + 8)).toNat = len at hy₁ hy₂
        generalize content.get! (pos₁ + 1) = kind at hy₁ hy₂
        by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
        · rw [ite_eq_left h3] at hy₁
          cases hy₁
        rw [ite_eq_right h3] at hy₁ hy₂
        obtain ⟨r', hr', hy₁⟩ := bindV_ok hy₁
        obtain ⟨r'', hr'', hy₂⟩ := bindV_ok hy₂
        cases hy₁
        cases hy₂
        have hi := forIn_rel (fun (a b : Masker × ByteArray × Array ℕ) => unstop a.1 = unstop b.1 ∧ a.2.2 = b.2.2)
          (fun _ => True) _ _ ?inner _ _ _ _ _ (by exact ⟨hm, rfl⟩) trivial hr' hr''
        · exact ⟨rfl, hi.2, hi.1, trivial⟩
        · intro j l' a b ya yb za hab _ hya hyb _
          obtain ⟨ma, oa, fa⟩ := a
          obtain ⟨mb, ob, fb⟩ := b
          obtain ⟨hu, hfab⟩ := hab
          dsimp only at hu hfab hya hyb
          subst hfab
          have hon : mb.on = ma.on := (congrArg Masker.on hu).symm
          have hpo : mb.pos = ma.pos := (congrArg Masker.pos hu).symm
          rw [hon, hpo, ← hc.nPiop] at hyb
          by_cases h8 : (ma.on && decide (ma.pos < c.nPiop)) = true
          · rw [ite_eq_left h8] at hya hyb
            obtain ⟨u, hu1, hya⟩ := bindV_ok hya
            obtain ⟨u', hu2, hyb⟩ := bindV_ok hyb
            cases hya
            cases hyb
            exact ⟨absorb_unstop hc hu hu1 hu2, rfl⟩
          · rw [ite_eq_right h8] at hya hyb
            obtain ⟨u, hu1, hya⟩ := bindV_ok hya
            obtain ⟨u', hu2, hyb⟩ := bindV_ok hyb
            cases hya
            cases hyb
            exact ⟨absorb_unstop hc hu hu1 hu2, rfl⟩
      · generalize (_root_.Flock.u64OfLE content (pos₁ + 8)).toNat = len at hy₁ hy₂
        generalize content.get! (pos₁ + 1) = kind at hy₁ hy₂
        by_cases h5 : (len == 0 && kind == 2) = true
        · rw [ite_eq_left h5] at hy₁ hy₂
          cases hy₁
          cases hy₂
          exact ⟨rfl, rfl, hm, trivial⟩
        rw [ite_eq_right h5] at hy₁ hy₂
        by_cases h4 : (pos₁ + 16 != content.size) = true
        · rw [ite_eq_left h4] at hy₁
          cases hy₁
        rw [ite_eq_right h4] at hy₁ hy₂
        by_cases h6 : (len != n) = true
        · rw [ite_eq_left h6] at hy₁
          cases hy₁
        rw [ite_eq_right h6] at hy₁ hy₂
        by_cases h7 : (!(kind == 2 || kind == 1 && n == 1)) = true
        · rw [ite_eq_left h7] at hy₁
          cases hy₁
        rw [ite_eq_right h7] at hy₁ hy₂
        have hd₁ : y₁ = .done (some (out₁ ++ _root_.Flock.Transcript.header 4 kind n, { m₁ with owed := some n },
            free₁), m₁, out₁, free₁, pos₁ + 16) := by
          split at hy₁
          · obtain ⟨_, -, hy₁⟩ := bindV_ok hy₁
            cases hy₁
            rfl
          · cases hy₁
            rfl
        have hd₂ : y₂ = .done (some (out₂ ++ _root_.Flock.Transcript.header 4 kind n, { m₂ with owed := some n },
            free₁), m₂, out₂, free₁, pos₁ + 16) := by
          split at hy₂
          · obtain ⟨_, -, hy₂⟩ := bindV_ok hy₂
            cases hy₂
            rfl
          · cases hy₂
            rfl
        subst hd₁ hd₂
        exact ⟨rfl, rfl, hm, rfl, congrArg (fun s : Masker => { s with owed := some n }) hm⟩
      · have hy₁ := Ranked.ite_bind_ok hy₁
        have hy₂ := Ranked.ite_bind_ok hy₂
        cases hy₁
        cases hy₂
        exact ⟨rfl, rfl, hm, trivial⟩
      · cases hy₁
  · intro i x y hy
    obtain ⟨o, m, out, free, pos⟩ := x
    dsimp only at hy
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
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨rfl, by dsimp only; omega, #[], by simp, fun e he => by simp at he⟩
    · have hy := Ranked.ite_bind_ok hy
      obtain ⟨x, hx, hy⟩ := bindV_ok hy
      cases hy
      exact ⟨rfl, by dsimp only; omega, #[], by simp, fun e he => by simp at he⟩
    · have hy := Ranked.ite_bind_ok hy
      generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h3 : (!(kind == 2 || kind == 1 && len == 1)) = true
      · rw [ite_eq_left h3] at hy
        cases hy
      rw [ite_eq_right h3] at hy
      obtain ⟨r', hr', hy⟩ := bindV_ok hy
      cases hy
      have hQ' : ∃ L : Array ℕ, r'.2.2 = free ++ L ∧ ∀ e ∈ L, pos + 16 ≤ e ∧ e + 16 ≤ pos + 16 + 16 * len := by
        refine forIn_inv_mem (fun x : Masker × ByteArray × Array ℕ =>
          ∃ L : Array ℕ, x.2.2 = free ++ L ∧ ∀ e ∈ L, pos + 16 ≤ e ∧ e + 16 ≤ pos + 16 + 16 * len) _ _ ?_ _ _
          ⟨#[], by simp, fun e he => by simp at he⟩ hr'
        intro i hi x y hx hy
        have hi' := (List.mem_range'_1.1 hi).2
        obtain ⟨L, hL, hLe⟩ := hx
        by_cases h8 : (x.1.on && decide (x.1.pos < c.nPiop)) = true
        · rw [ite_eq_left h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          refine ⟨L.push _, by rw [hL, Array.append_push], fun e he => ?_⟩
          rcases Array.mem_push.1 he with he | he
          · exact hLe e he
          · subst he
            constructor <;> omega
        · rw [ite_eq_right h8] at hy
          obtain ⟨a, ha, hy⟩ := bindV_ok hy
          cases hy
          exact ⟨L, hL, hLe⟩
      obtain ⟨L, hL, hLe⟩ := hQ'
      exact ⟨rfl, by dsimp only; omega, L, hL, hLe⟩
    · generalize (_root_.Flock.u64OfLE content (pos + 8)).toNat = len at hy
      generalize content.get! (pos + 1) = kind at hy
      by_cases h5 : (len == 0 && kind == 2) = true
      · rw [ite_eq_left h5] at hy
        cases hy
        exact ⟨rfl, by dsimp only; omega, #[], by simp, fun e he => by simp at he⟩
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
        exact ⟨⟨_, rfl, rfl⟩, by dsimp only; omega, rfl⟩
      · cases hy
        exact ⟨⟨_, rfl, rfl⟩, by dsimp only; omega, rfl⟩
    · have hy := Ranked.ite_bind_ok hy
      cases hy
      exact ⟨rfl, by dsimp only; omega, #[], by simp, fun e he => by simp at he⟩
    · cases hy

theorem roundFree_of_round {c : Cfg} {s : Masker} {r : ByteArray} {n : ℕ} {o : ByteArray} {t : Masker}
    (h : _root_.Flock.Firewall.Frame.round c s r n = .ok (o, t)) : ∃ fr, roundFree c s r n = .ok (o, t, fr) := by
  unfold _root_.Flock.Firewall.Frame.round at h
  obtain ⟨⟨o', t', fr⟩, hq, h⟩ := bindV_ok h
  cases h
  exact ⟨fr, hq⟩

open _root_.Flock.Firewall.Frame (check) in
/-- **The frame's check leaves the scheduled round's state** but for `stopped`: whatever the device round, at a state
of a stopped rep or not, the state the check returns is the one the scheduled round `r` leaves the rep in. -/
theorem check_unstop {c : Cfg} {m : Masker} {st : Bool} {v r : ByteArray} {n : ℕ} {o : ByteArray} {t : Masker}
    {f : Array ℕ} {out : ByteArray} {s' : Masker} {why : Option String}
    (hr : roundFree c { m with stopped := false } r n = .ok (o, t, f))
    (h : check c { m with stopped := st } v r n = .ok (out, s', why)) : unstop s' = unstop t := by
  have hc : CfgAlike c c := ⟨rfl, rfl, rfl, rfl, rfl, rfl⟩
  cases why with
  | some w =>
    obtain ⟨fr, hq⟩ := roundFree_of_round (check_some h)
    exact (roundFree_agree hc (by exact rfl) (by exact rfl) hr hq fun _ _ _ => rfl).2.symm
  | none =>
    obtain ⟨q, hq, hsz, hd⟩ := passes_none (check_none_passes h)
    obtain ⟨fr, hv⟩ := roundFree_of_round (check_none h)
    have h1 := roundFree_agree hc (by exact rfl) hsz.symm hq hv fun i hi hf =>
      (firstDiff_none hd i (by omega) hf).symm
    have h2 := roundFree_agree hc (by exact rfl) (by exact rfl) hr hq fun _ _ _ => rfl
    exact h1.2.symm.trans h2.2.symm

end Free

end Claims.ZKServer
