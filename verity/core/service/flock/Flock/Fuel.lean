/-
The fuel-bounded recursions of the sampled verifier's reading: past its fuel each refuses, and a result it accepts at
some fuel it gives, unchanged, at any more (`defOfJsonIn_mono`, `cutIn_mono`, `evalDefIn_mono`; the walks' is
`Steps.run_mono`). So the verifier's answer at `Steps.FUEL` is the unbounded recursion's wherever it accepts.
-/
import Flock.Qword
import Flock.Qcall
import Flock.Extract

namespace Flock.Fuel

open Program

theorem bind_ok {ε α β} {x y : Except ε α} {k k' : α → Except ε β} (hxy : ∀ a, x = .ok a → y = .ok a)
    (hk : ∀ a b, k a = .ok b → k' a = .ok b) {b : β} (h : (x >>= k) = .ok b) : (y >>= k') = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; rw [hxy a hx]; exact hk a b h

theorem list_mapM {ε α β} {f g : α → Except ε β} (h : ∀ a b, f a = .ok b → g a = .ok b) :
    ∀ {l : List α} {bs : List β}, l.mapM f = .ok bs → l.mapM g = .ok bs
  | [], bs, hf => by simpa using hf
  | a :: l, bs, hf => by
    rw [List.mapM_cons] at hf ⊢
    exact bind_ok (h a) (fun _ _ ht => bind_ok (fun _ hl => list_mapM h hl) (fun _ _ e => e) ht) hf

theorem array_mapM {ε α β} {f g : α → Except ε β} (h : ∀ a b, f a = .ok b → g a = .ok b) {as : Array α}
    {bs : Array β} (hf : as.mapM f = .ok bs) : as.mapM g = .ok bs := by
  rw [Array.mapM_eq_mapM_toList] at hf ⊢
  exact bind_ok (fun _ hl => list_mapM h hl) (fun _ _ e => e) hf

theorem list_forIn {ε α σ} {f g : α → σ → Except ε (ForInStep σ)} (h : ∀ a s t, f a s = .ok t → g a s = .ok t) :
    ∀ {l : List α} {s r : σ}, forIn l s f = .ok r → forIn l s g = .ok r
  | [], s, r, hf => by simpa using hf
  | a :: l, s, r, hf => by
    rw [List.forIn_cons] at hf ⊢
    refine bind_ok (h a s) (fun t r ht => ?_) hf
    cases t with
    | done b => exact ht
    | yield b => exact list_forIn h ht

theorem range_forIn {ε σ} {f g : Nat → σ → Except ε (ForInStep σ)} (h : ∀ a s t, f a s = .ok t → g a s = .ok t)
    {rg : Std.Legacy.Range} {s r : σ} (hf : forIn rg s f = .ok r) : forIn rg s g = .ok r := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hf ⊢
  exact list_forIn h hf

theorem array_forIn {ε α σ} {f g : α → σ → Except ε (ForInStep σ)} (h : ∀ a s t, f a s = .ok t → g a s = .ok t)
    {xs : Array α} {s r : σ} (hf : forIn xs s f = .ok r) : forIn xs s g = .ok r := by
  rw [← Array.forIn_toList] at hf ⊢
  exact list_forIn h hf

/-- A Definition `evaluate` reads at some nesting bound, it reads alike at any larger one. -/
theorem defOfJsonIn_mono : ∀ {n m : Nat} {j : Lean.Json} {d : Qword.Def}, n ≤ m →
    Qword.defOfJsonIn n j = .ok d → Qword.defOfJsonIn m j = .ok d
  | 0, _, _, _, _, h => by simp [Qword.defOfJsonIn] at h
  | _ + 1, 0, _, _, hnm, _ => by omega
  | n + 1, m + 1, j, d, hnm, h => by
    have ih : ∀ j d, Qword.defOfJsonIn n j = .ok d → Qword.defOfJsonIn m j = .ok d :=
      fun _ _ => defOfJsonIn_mono (by omega)
    unfold Qword.defOfJsonIn at h ⊢
    simp only at h ⊢
    generalize Qword.defOfJsonIn n = f at h ih
    generalize Qword.defOfJsonIn m = g at ih ⊢
    split at h
    · exact h
    · refine bind_ok (fun l hl => array_mapM (fun a b hab => ?_) hl) (fun _ _ e => e) h
      split at hab
      · split at hab
        · simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hab
        · rename_i hne
          simp only [hne] at ⊢
          split at hab
          · exact bind_ok (ih _) (fun _ _ e => e) hab
          · exact hab
      · exact hab
    · exact h
    · exact h

/-- A Q_call cut the verifier finishes at some nesting bound, it finishes alike at any larger one. -/
theorem cutIn_mono {P : Prog} {G : Qcall.Gates} {X : Nat} : ∀ {n m : Nat} {lo hi : Nat} {path : Array (Array Nat)}
    {how : Qcall.Piece} {acc r : Array Qcall.ProofBlock}, n ≤ m →
    Qcall.cutIn n P G X lo hi path how acc = .ok r → Qcall.cutIn m P G X lo hi path how acc = .ok r
  | 0, _, _, _, _, _, _, _, _, h => by simp [Qcall.cutIn] at h
  | _ + 1, 0, _, _, _, _, _, _, hnm, _ => by omega
  | n + 1, m + 1, lo, hi, path, how, acc, r, hnm, h => by
    have ih : ∀ lo hi path how acc r, Qcall.cutIn n P G X lo hi path how acc = .ok r →
        Qcall.cutIn m P G X lo hi path how acc = .ok r := fun _ _ _ _ _ _ => cutIn_mono (by omega)
    unfold Qcall.cutIn at h ⊢
    simp only at h ⊢
    generalize Qcall.cutIn n P G X = f at h ih
    generalize Qcall.cutIn m P G X = g at ih ⊢
    by_cases hc : (!G.anyComputing lo hi) = true
    · simp only [hc] at h ⊢; exact h
    by_cases hw : G.wOut lo hi ≤ X
    · simp only [hc, hw] at h ⊢; exact h
    simp only [hc, hw, ↓reduceIte] at h ⊢
    cases how with
    | act d off =>
      exact bind_ok (fun s hs => range_forIn (fun k s t hk => bind_ok (ih _ _ _ _ _) (fun _ _ e => e) hk) hs)
        (fun _ _ e => e) h
    | members d off k =>
      exact bind_ok (fun s hs => range_forIn (fun k s t hk => bind_ok (ih _ _ _ _ _) (fun _ _ e => e) hk) hs)
        (fun _ _ e => e) h
    | gate => exact h

/-- A Q_word cut of a Definition the verifier finishes at some nesting bound, with its memo, it finishes alike at any
larger one. -/
theorem evalDefIn_mono {P : Prog} {X : Nat} : ∀ {n m : Nat} {memo : Std.HashMap Nat Qword.DCut} {d : Nat}
    {r : Qword.DCut × Std.HashMap Nat Qword.DCut}, n ≤ m →
    Extract.evalDefIn n P X memo d = .ok r → Extract.evalDefIn m P X memo d = .ok r
  | 0, _, _, _, _, _, h => by simp [Extract.evalDefIn] at h
  | _ + 1, 0, _, _, _, hnm, _ => by omega
  | n + 1, m + 1, memo, d, r, hnm, h => by
    have ih : ∀ memo d r, Extract.evalDefIn n P X memo d = .ok r → Extract.evalDefIn m P X memo d = .ok r :=
      fun _ _ _ => evalDefIn_mono (by omega)
    unfold Extract.evalDefIn at h ⊢
    simp only at h ⊢
    generalize Extract.evalDefIn n P X = f at h ih
    generalize Extract.evalDefIn m P X = g at ih ⊢
    cases hm : memo[d]? with
    | some c => simp only [hm] at h ⊢; exact h
    | none =>
      simp only [hm] at h ⊢
      by_cases hp : P.defs[d]!.prim = true
      · simp only [hp, ↓reduceIte] at h ⊢; exact h
      · simp only [hp, Bool.false_eq_true, ↓reduceIte] at h ⊢
        cases hs : Extract.separable P d with
        | some parts =>
          simp only [hs] at h ⊢
          exact bind_ok (fun s hs' => array_forIn (fun x s t hk => bind_ok (ih _ _) (fun _ _ e => e) hk) hs')
            (fun _ _ e => e) h
        | none => simp only [hs] at h ⊢; exact h

end Flock.Fuel
