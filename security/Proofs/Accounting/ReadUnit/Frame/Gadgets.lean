/-
What the rows of the frame-v3 read's small gadgets force: a list's `mapM` entry by entry (`sound_mapM_getD`), the AND
tree and the digest comparison (`sound_eqDigest`), one bit of the layouts' selection (`sound_selectGo`, whose sum
telescopes to the selected layout's bit, `selV_telescope`) and the flags (`sound_flags`: flag `j` is set iff the hidden
number is at least `2 ^ (8 (j + 1))`).
-/
module

public import Proofs.Accounting.ReadUnit.Frame.Bytes

@[expose] public section

namespace Accounting.ReadUnit.FrameRead

open Flock.HmNets hiding and
open FlockSoundness.Discharge.Hm
open Flock.ReadNet Flock.ReadNet.Frame
open Accounting.ReadUnit.Gates

/-! ## `mapM` -/

theorem sound_mapM_getD {β γ : Type} (d : γ) (f : β → M γ) (Q : (Var → Bool) → β → γ → Prop)
    (hf : ∀ x, Sound (f x) fun z r => Q z x r) :
    ∀ l : List β, Sound (l.mapM f) fun z rs => rs.length = l.length ∧ ∀ i (hi : i < l.length), Q z l[i] (rs.getD i d)
  | [] => Sound.pure' fun z => ⟨rfl, fun i hi => absurd hi (by simp)⟩
  | x :: l => by
    rw [List.mapM_cons]
    refine Sound.bind' (hf x) fun r => Sound.bind' (sound_mapM_getD d f Q hf l) fun rs =>
      Sound.pure' fun z hl hr => ?_
    obtain ⟨hlen, hall⟩ := hl
    refine ⟨by simp [hlen], fun i hi => ?_⟩
    cases i with
    | zero => simpa using hr
    | succ i => simpa using hall i (by simpa using hi)

/-! ## The body's constant bits -/

/-- The cached constant bits hold their values at `z`. -/
def ConstsOk (z : Var → Bool) (c : Consts) : Prop :=
  (∀ f, c.1 = some f → ev z f = false) ∧ ∀ f, c.2 = some f → ev z f = true

/-- With `BL.constant` a constant form, the constant bits are `pure`: the cache gains only constant forms. -/
theorem constBits_constForm (c : Consts) (v w : ℕ) : ∃ c' : Consts, constBits constForm c v w =
      pure (c', (List.range w).map fun j => if v.testBit j then c'.2.getD Lin.one else c'.1.getD Lin.zero) ∧
    (∀ f, c'.1 = some f → f = Lin.const false ∨ c.1 = some f) ∧
      ∀ f, c'.2 = some f → f = Lin.const true ∨ c.2 = some f := by
  simp only [Frame.constBits, constForm]
  split_ifs <;> simp only [map_pure, pure_bind] <;>
    exact ⟨_, rfl, fun f hf => by first | exact Or.inr hf | exact Or.inl (Option.some.inj hf).symm,
      fun f hf => by first | exact Or.inr hf | exact Or.inl (Option.some.inj hf).symm⟩

/-- **The constant bits** (`_body.const` over `BL.constant` in a flat net): `v`'s low `w` bits. -/
theorem sound_constBits (c : Consts) (v w : ℕ) : Sound (constBits constForm c v w) fun z r =>
    ConstsOk z c → ConstsOk z r.1 ∧ evs z r.2 = (List.range w).map v.testBit ∧ r.2.length = w := by
  obtain ⟨c', he, h1, h2⟩ := constBits_constForm c v w
  rw [he]
  refine Sound.pure' fun z hc => ?_
  have e0 : ∀ f, c'.1 = some f → ev z f = false := fun f hf => by
    rcases h1 f hf with rfl | h
    · rfl
    · exact hc.1 f h
  have e1 : ∀ f, c'.2 = some f → ev z f = true := fun f hf => by
    rcases h2 f hf with rfl | h
    · rfl
    · exact hc.2 f h
  have g0 : ev z (c'.1.getD Lin.zero) = false := by
    cases h : c'.1 with
    | none => rfl
    | some f => exact e0 f h
  have g1 : ev z (c'.2.getD Lin.one) = true := by
    cases h : c'.2 with
    | none => rfl
    | some f => exact e1 f h
  refine ⟨⟨e0, e1⟩, ?_, by simp⟩
  simp only [evs, List.map_map]
  refine List.map_congr_left fun j _ => ?_
  simp only [Function.comp_apply]
  split <;> rename_i hj <;> simp [hj, g0, g1]

/-! ## The AND tree and the comparison -/

theorem sound_andPairs : ∀ l : List Lin, Sound (andPairs l) fun z r =>
    r.length = (l.length + 1) / 2 ∧ ((∀ f ∈ r, ev z f = true) ↔ ∀ f ∈ l, ev z f = true)
  | [] => Sound.pure' fun z => by simp
  | [a] => Sound.pure' fun z => by simp
  | a :: b :: rest => by
    show Sound (Flock.HmNets.and a b >>= fun r => andPairs rest >>= fun rs => pure (r :: rs)) _
    refine Sound.bind' (sound_and a b) fun r => Sound.bind' (sound_andPairs rest) fun rs =>
      Sound.pure' fun z hrs hr => ?_
    obtain ⟨hl, hall⟩ := hrs
    refine ⟨by simp only [List.length_cons, hl]; omega, ?_⟩
    simp only [List.mem_cons, forall_eq_or_imp, hr, Bool.and_eq_true, hall]
    tauto

theorem sound_andTree : ∀ (k : ℕ) (l : List Lin), Sound (andTree k l) fun z r =>
    l.length ≤ k + 1 → (ev z r = true ↔ ∀ f ∈ l, ev z f = true)
  | 0, [] => Sound.pure' fun z _ => by simp
  | _ + 1, [] => Sound.pure' fun z _ => by simp
  | 0, [a] => Sound.pure' fun z _ => by simp
  | _ + 1, [a] => Sound.pure' fun z _ => by simp
  | 0, a :: b :: l => Sound.pure' fun z h => by simp at h
  | k + 1, a :: b :: l => by
    show Sound (andPairs (a :: b :: l) >>= fun l' => andTree k l') _
    refine Sound.bind' (sound_andPairs _) fun l' => Sound.mono (sound_andTree k l') fun z r h hP hl => ?_
    obtain ⟨hlen, hall⟩ := hP
    rw [← hall]
    refine h ?_
    rw [hlen]
    simp only [List.length_cons] at hl ⊢
    omega

theorem sound_andReduce (l : List Lin) : Sound (andReduce l) fun z r => ev z r = true → ∀ f ∈ l, ev z f = true :=
  Sound.mono (sound_andTree l.length l) fun z r h hr => (h (by omega)).1 hr

theorem map_eq_of_zip {α β : Type} (f : α → β) :
    ∀ {l l' : List α}, l.length = l'.length → (∀ p ∈ l.zip l', f p.1 = f p.2) → l.map f = l'.map f
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: l, a' :: l', h, hp => by
    rw [List.map_cons, List.map_cons, hp (a, a') (by simp),
      map_eq_of_zip f (by simpa using h) fun p hp' => hp p (List.mem_cons_of_mem _ hp')]

/-- **The comparison** (`LmsEqual_v1{N = 64}`): whenever its rows hold, its bit is set only where the two digests agree. -/
theorem sound_eqDigest (x y : List Word) : Sound (eqDigest x y) fun z r =>
    Shaped 8 x → Shaped 8 y → ev z r = true → wordsDigest z x = wordsDigest z y := by
  refine Sound.mono (sound_andReduce _) fun z r h hx hy hr => ?_
  have hall := h hr
  have hw : x.map (evs z) = y.map (evs z) := by
    refine map_eq_of_zip _ (by rw [hx.1, hy.1]) fun ab hab => ?_
    have hl : ab.1.length = ab.2.length := by
      rw [hx.2 _ (List.of_mem_zip hab).1, hy.2 _ (List.of_mem_zip hab).2]
    refine map_eq_of_zip _ hl fun uv huv => ?_
    have := hall _ (List.mem_flatMap.2 ⟨ab, hab, List.mem_map.2 ⟨uv, huv, rfl⟩⟩)
    simp only [ev_xor, ev_one] at this
    cases h1 : ev z uv.1 <;> cases h2 : ev z uv.2 <;> simp_all
  simp only [wordsDigest, evs, List.map_flatten]
  exact congrArg _ (congrArg _ hw)

/-! ## The selection -/

/-- `_select`'s sum over values: `m`, then for each flag `z` and adjacent bits `(a, b)`, `z (a ⊕ b)` added. -/
def selV : Bool → List Bool → List (Bool × Bool) → Bool
  | m, z :: zs, (a, b) :: ps => selV (m ^^ (z && (a ^^ b))) zs ps
  | m, _, _ => m

theorem sound_selectGo : ∀ (m : Lin) (fl : List Lin) (ps : List (Lin × Lin)), Sound (selectGo m fl ps) fun z r =>
    ev z r = selV (ev z m) (evs z fl) (ps.map fun ab => (ev z ab.1, ev z ab.2))
  | m, [], ps => Sound.pure' fun z => rfl
  | m, _ :: _, [] => Sound.pure' fun z => rfl
  | m, f :: fl, (a, b) :: ps => by
    have hk : ∀ m' : Lin, Sound (selectGo m' fl ps) fun z r =>
        ev z m' = (ev z m ^^ (ev z f && (ev z a ^^ ev z b))) →
          ev z r = selV (ev z m) (evs z (f :: fl)) (((a, b) :: ps).map fun ab => (ev z ab.1, ev z ab.2)) :=
      fun m' => Sound.mono (sound_selectGo m' fl ps) fun z r h hm' => by
        rw [h, hm']
        simp [selV, evs]
    simp only [selectGo]
    refine Sound.ite' (fun hc => ?_) fun _ => ?_
    · simp only [pure_bind]
      refine Sound.mono (hk _) fun z r h => h ?_
      have hd := ev_of_isConst hc z
      rw [ev_xor] at hd
      rw [hd]
      split_ifs with hdc <;> simp [hdc]
    · refine Sound.bind' (sound_and _ _) fun r => ?_
      simp only [pure_bind]
      refine Sound.mono (hk _) fun z x h hr => h ?_
      simp [hr]

theorem selV_false (m : Bool) : ∀ (k : ℕ) (ps : List (Bool × Bool)), selV m (List.replicate k false) ps = m
  | 0, ps => by cases ps <;> rfl
  | k + 1, [] => rfl
  | k + 1, (a, b) :: ps => by
    rw [List.replicate_succ, selV, Bool.false_and, Bool.xor_false, selV_false m k ps]

/-- **The sum telescopes**: under flags whose first `k` are set and the rest clear, the sum over adjacent bits of
`c :: cs` is its bit `k`. -/
theorem selV_telescope : ∀ (k r : ℕ) (c : Bool) (cs : List Bool), k ≤ cs.length →
    selV c (List.replicate k true ++ List.replicate r false) ((c :: cs).zip cs) = (c :: cs).getD k false
  | 0, r, c, cs, _ => by simp [selV_false]
  | k + 1, r, c, [], h => absurd h (by simp)
  | k + 1, r, c, c' :: cs, h => by
    rw [List.replicate_succ, List.cons_append, List.zip_cons_cons, selV, Bool.true_and,
      show (c ^^ (c ^^ c')) = c' by cases c <;> cases c' <;> rfl,
      selV_telescope k r c' cs (by simpa using h), List.getD_cons_succ]

theorem map_zip_tail (z : Var → Bool) : ∀ col : List Lin,
    (col.zip col.tail).map (fun ab => (ev z ab.1, ev z ab.2)) = (evs z col).zip (evs z col).tail
  | [] => rfl
  | [_] => rfl
  | a :: b :: col => by
    have := map_zip_tail z (b :: col)
    simp only [List.tail_cons, List.zip_cons_cons, List.map_cons, evs] at this ⊢
    rw [this]

/-- **One bit of the selection**: under flags holding `k` ones then zeros, with `k` below the layouts' count, the
selected bit holds layout `k`'s. -/
theorem sound_selectBit (fl col : List Lin) : Sound (selectBit fl col) fun z r => ∀ k s : ℕ,
    evs z fl = List.replicate k true ++ List.replicate s false → k < col.length →
      ev z r = ev z (col.getD k Lin.zero) := by
  refine Sound.mono (sound_selectGo _ _ _) fun z r h k s hfl hk => ?_
  rw [h, hfl, map_zip_tail]
  cases col with
  | nil => simp at hk
  | cons c cs =>
    simp only [List.headD_cons, evs, List.map_cons, List.tail_cons]
    rw [selV_telescope k s _ _ (by simp at hk ⊢; omega)]
    exact getD_evs z (c :: cs) k

/-! ## The flags -/

/-- Whether any of bits `lo … hi - 1` of `bits` is set at `z`. -/
def anyFrom (z : Var → Bool) (bits : List Lin) (lo hi : ℕ) : Bool :=
  (List.range' lo (hi - lo)).any fun i => ev z (bits.getD i Lin.zero)

theorem anyFrom_eq_true {z : Var → Bool} {bits : List Lin} {lo hi : ℕ} :
    anyFrom z bits lo hi = true ↔ ∃ i, lo ≤ i ∧ i < hi ∧ ev z (bits.getD i Lin.zero) = true := by
  simp only [anyFrom, List.any_eq_true, List.mem_range'_1]
  constructor
  · rintro ⟨i, ⟨h1, h2⟩, h3⟩
    exact ⟨i, h1, by omega, h3⟩
  · rintro ⟨i, h1, h2, h3⟩
    exact ⟨i, ⟨h1, by omega⟩, h3⟩

theorem anyFrom_self (z : Var → Bool) (bits : List Lin) (n : ℕ) : anyFrom z bits n n = false := by
  simp [anyFrom]

theorem anyFrom_step (z : Var → Bool) (bits : List Lin) {lo hi : ℕ} (h : lo < hi) :
    anyFrom z bits lo hi = (ev z (bits.getD lo Lin.zero) || anyFrom z bits (lo + 1) hi) := by
  rw [anyFrom, anyFrom, show hi - lo = (hi - (lo + 1)) + 1 by omega, List.range'_succ, List.any_cons]

theorem sound_orL (a b : Lin) : Sound (orL a b) fun z r => ev z r = (ev z a || ev z b) := by
  unfold orL
  refine Sound.bind' (sound_and a b) fun r => Sound.pure' fun z hr => ?_
  rw [ev_xor, ev_xor, hr]
  cases ev z a <;> cases ev z b <;> rfl

/-- The flags' OR chain over bits `t + 7, …, 8`, the bits above it being `t + 8 … N - 1`. -/
theorem sound_flagsGo (bits : List Lin) (N : ℕ) : ∀ (t : ℕ) (acc : Option Lin) (out : List Lin),
    Sound (flagsGo bits t acc out) fun z r => t + 8 ≤ N → acc.elim false (ev z) = anyFrom z bits (t + 8) N →
      evs z r = (List.range ((t + 7) / 8)).map (fun j => anyFrom z bits (8 * (j + 1)) N) ++ evs z out
  | 0, acc, out => Sound.pure' fun z _ _ => by simp
  | t + 1, acc, out => by
    have tail : ∀ a : Lin, Sound (flagsGo bits t (some a) (if (t + 8) % 8 = 0 then a :: out else out)) fun z r =>
        t + 1 + 8 ≤ N → ev z a = anyFrom z bits (t + 8) N →
          evs z r = (List.range ((t + 1 + 7) / 8)).map (fun j => anyFrom z bits (8 * (j + 1)) N) ++ evs z out :=
      fun a => Sound.mono (sound_flagsGo bits N t (some a) _) fun z r h hN ha => by
        rw [h (by omega) ha]
        split_ifs with h8
        · rw [show (t + 1 + 7) / 8 = (t + 7) / 8 + 1 by omega, List.range_succ, List.map_append, List.append_assoc]
          simp only [List.map_cons, List.map_nil, evs, List.singleton_append]
          rw [ha, show 8 * ((t + 7) / 8 + 1) = t + 8 by omega]
        · rw [show (t + 1 + 7) / 8 = (t + 7) / 8 by omega]
    cases acc with
    | none =>
      simp only [flagsGo, pure_bind]
      refine Sound.mono (tail _) fun z r h hN hacc => h hN ?_
      rw [anyFrom_step z bits (show t + 8 < N by omega), show t + 8 + 1 = t + 1 + 8 by omega, ← hacc]
      simp
    | some a₀ =>
      simp only [flagsGo]
      refine Sound.bind' (sound_orL a₀ _) fun a => Sound.mono (tail a) fun z r h hr hN hacc => h hN ?_
      rw [hr, anyFrom_step z bits (show t + 8 < N by omega), show t + 8 + 1 = t + 1 + 8 by omega, ← hacc]
      exact Bool.or_comm _ _

/-- **The flags** (`frame_v3._flags`): flag `j < n - 1` is set iff bits `8 (j + 1) … 8 n - 1` of `bits` have one set. -/
theorem sound_flags (bits : List Lin) (n : ℕ) : Sound (flags bits n) fun z fl => 1 ≤ n →
    evs z fl = (List.range (n - 1)).map fun j => anyFrom z bits (8 * (j + 1)) (8 * n) := by
  refine Sound.mono (sound_flagsGo bits (8 * n) _ none []) fun z fl h hn => ?_
  rw [h (by omega) (by simp [anyFrom_self, show 8 * n - 8 + 8 = 8 * n by omega]),
    show (8 * n - 8 + 7) / 8 = n - 1 by omega]
  simp [evs]

end Accounting.ReadUnit.FrameRead
