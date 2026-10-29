import Proofs.Properties.PoubCheck.Coins

/-!
# A drawn window's record is kept (`RecordKeptAfterDraw`)

Each event keeps every window's `KeepW`: once drawn, its draws, challenge, roots, plans and heads stay as they are;
once final, it stays final; and it is drawn only if it was, or is final. No event changes the digest, and the appends
only grow (`Frz`, `frz_step`). An event that adds a pass adds it to an undrawn window: the window it fills is not
final (so undrawn, `DF`), or is the one after it, which has no root yet (`Inv.beyond`), and a drawn window has
`min_passes ≥ 1` of them (`Inv.few`). So a run keeps `Frz` from a state on (`run_frz`), and a drawn window's record,
which reads its own fields and the window before it (drawn too, `Prefix`), is kept.

Its openable receipts are kept with it: `checkOpening` reads the appends only for the append holding its block
(`appendOf`, `checkOpening_congr`), the block a drawn pass reads is below the store's extent then (`record_reads`), and
every append after starts at that extent or above (`firsts_big`), so `appendOf` finds the same one (`appendOf_extend`).
-/

open PoubCheck PoubCheck.Session

namespace Properties.PoubCheck

/-! ## Each window -/

/-- What a window's record reads of it, and its draw, unchanged. -/
def Fixed (w w' : Win) : Prop :=
  w'.drawn = w.drawn ∧ w'.challenge = w.challenge ∧ w'.roots = w.roots ∧ w'.plans = w.plans ∧ w'.heads = w.heads

/-- `w'` keeps what `w` fixed once drawn, stays final once `w` is, and is drawn only if `w` was or `w'` is final. -/
def KeepW (w w' : Win) : Prop :=
  (w.drawn.isSome → Fixed w w') ∧ (w.final → w'.final) ∧ (w'.drawn.isSome → w.drawn.isSome ∨ w'.final)

/-- Each window of `ws'` keeps its namesake's in `ws` (an index past `ws` is a default, undrawn and not final). -/
def KeepsF (ws ws' : Array Win) : Prop := ∀ u : Nat, KeepW ws[u]! ws'[u]!

/-- Every drawn window is final. -/
def DF (ws : Array Win) : Prop := ∀ u : Nat, ws[u]!.drawn.isSome → ws[u]!.final

theorem keepW_refl (w : Win) : KeepW w w := ⟨fun _ => ⟨rfl, rfl, rfl, rfl, rfl⟩, id, Or.inl⟩

theorem keepW_trans {a b c : Win} (h₁ : KeepW a b) (h₂ : KeepW b c) : KeepW a c := by
  obtain ⟨f1, g1, d1⟩ := h₁
  obtain ⟨f2, g2, d2⟩ := h₂
  refine ⟨fun ha => ?_, fun ha => g2 (g1 ha), fun hc => ?_⟩
  · obtain ⟨e1, e2, e3, e4, e5⟩ := f1 ha
    have hb : b.drawn.isSome := by rw [e1]; exact ha
    obtain ⟨e1', e2', e3', e4', e5'⟩ := f2 hb
    exact ⟨e1'.trans e1, e2'.trans e2, e3'.trans e3, e4'.trans e4, e5'.trans e5⟩
  · rcases d2 hc with hb | hc'
    · rcases d1 hb with ha | hb'
      · exact Or.inl ha
      · exact Or.inr (g2 hb')
    · exact Or.inr hc'

theorem keepsF_refl (ws : Array Win) : KeepsF ws ws := fun _ => keepW_refl _

theorem keepsF_trans {a b c : Array Win} (h₁ : KeepsF a b) (h₂ : KeepsF b c) : KeepsF a c :=
  fun u => keepW_trans (h₁ u) (h₂ u)

theorem getElem!_modify' (ws : Array Win) (u t : Nat) (f : Win → Win) :
    (ws.modify u f)[t]! = if u = t ∧ t < ws.size then f ws[t]! else ws[t]! := by
  by_cases ht : t < ws.size
  · rw [getElem!_pos (ws.modify u f) t (by simpa using ht), getElem!_pos ws t ht, Array.getElem_modify]
    by_cases hu : u = t <;> simp [hu, ht]
  · rw [getElem!_neg (ws.modify u f) t (by simpa using ht), getElem!_neg ws t ht]
    simp [ht]

theorem keepsF_modify' (ws : Array Win) (u : Nat) (f : Win → Win) (hf : u < ws.size → KeepW ws[u]! (f ws[u]!)) :
    KeepsF ws (ws.modify u f) := by
  intro t
  rw [getElem!_modify']
  split
  · rename_i h; obtain ⟨rfl, h⟩ := h; exact hf h
  · exact keepW_refl _

theorem keepsF_modify (ws : Array Win) (u : Nat) (f : Win → Win) (hf : KeepW ws[u]! (f ws[u]!)) :
    KeepsF ws (ws.modify u f) :=
  keepsF_modify' ws u f fun _ => hf

theorem keepsF_push (ws : Array Win) (w : Win) (hw : w.drawn = none) : KeepsF ws (ws.push w) := by
  intro t
  by_cases ht : t < ws.size
  · rw [getElem!_pos (ws.push w) t (by simp; omega), getElem!_pos ws t ht, Array.getElem_push_lt]
    exact keepW_refl _
  · rw [getElem!_neg ws t ht]
    refine ⟨fun h => (by cases h), fun h => (by cases h), fun h => ?_⟩
    by_cases he : t = ws.size
    · subst he
      rw [getElem!_pos (ws.push w) ws.size (by simp), Array.getElem_push_eq, hw] at h
      cases h
    · rw [getElem!_neg (ws.push w) t (by simp; omega)] at h
      cases h

/-- A change of fields a record and a draw don't read. -/
theorem keepW_of {w w' : Win} (h : Fixed w w') (hf : w.final → w'.final) : KeepW w w' :=
  ⟨fun _ => h, hf, fun hd => Or.inl (h.1 ▸ hd)⟩

/-- A change of an undrawn window that leaves it undrawn and as final as it was. -/
theorem keepW_undrawn {w w' : Win} (hd : w.drawn = none) (hd' : w'.drawn = w.drawn) (hf : w'.final = w.final) :
    KeepW w w' :=
  ⟨fun h => (by rw [hd] at h; cases h), fun h => (by rw [hf]; exact h), fun h => (by rw [hd', hd] at h; cases h)⟩

/-- The draw of an undrawn window, which makes it final. -/
theorem keepW_draw {w w' : Win} (hd : w.drawn = none) (hf : w'.final = true) : KeepW w w' :=
  ⟨fun h => (by rw [hd] at h; cases h), fun _ => hf, fun _ => Or.inr hf⟩

theorem df_keepsF {ws ws' : Array Win} (h : DF ws) (k : KeepsF ws ws') : DF ws' := by
  intro u hu
  obtain ⟨-, g, d⟩ := k u
  rcases d hu with hd | hf
  · exact g (h u hd)
  · exact hf

theorem keepsF_fail (s : State) (t : Nat) (f : Failure) (a : Option (Nat × Nat)) :
    KeepsF s.windows (s.fail t f a).2.windows := by
  unfold State.fail
  dsimp only
  split
  · exact keepsF_modify _ _ _ (keepW_of ⟨rfl, rfl, rfl, rfl, rfl⟩ id)
  · exact keepsF_refl _


/-! ## Each event -/

/-- What the rest of a session keeps of `s`: each window (`KeepW`), the digest, and the appends, only extended. -/
def Frz (s s' : State) : Prop :=
  KeepsF s.windows s'.windows ∧ s'.digest = s.digest ∧ ∃ more, s'.appends = s.appends ++ more

theorem frz_of {s s' : State} (hk : KeepsF s.windows s'.windows) (hd : s'.digest = s.digest)
    (ha : s'.appends = s.appends) : Frz s s' :=
  ⟨hk, hd, #[], by rw [ha, Array.append_empty]⟩

theorem frz_refl (s : State) : Frz s s := frz_of (keepsF_refl _) rfl rfl

theorem frz_trans {a b c : State} (h₁ : Frz a b) (h₂ : Frz b c) : Frz a c := by
  obtain ⟨k1, d1, m1, e1⟩ := h₁
  obtain ⟨k2, d2, m2, e2⟩ := h₂
  exact ⟨keepsF_trans k1 k2, d2.trans d1, m1 ++ m2, by rw [e2, e1, Array.append_assoc]⟩

theorem frz_fail (s : State) (t : Nat) (f : Failure) (a : Option (Nat × Nat)) : Frz s (s.fail t f a).2 :=
  frz_of (keepsF_fail s t f a) rfl rfl

theorem keepsF_draw (ws : Array Win) (u : Nat) (d : Array (Nat × Nat)) (now : Nat) (hu : ¬ws[u]!.drawn.isSome = true) :
    KeepsF ws ((ws.modify u ({· with final := true})).modify u ({· with drawn := some d, openedNs := some now})) := by
  refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ⟨rfl, rfl, rfl, rfl, rfl⟩ (fun _ => rfl)))
    (keepsF_modify' _ _ _ fun hl => ?_)
  rw [Array.size_modify] at hl
  have e : (ws.modify u ({· with final := true}))[u]! = {ws[u]! with final := true} := by
    rw [getElem!_modify']; simp [hl]
  rw [e]
  have : ws[u]!.drawn = none := by cases h : ws[u]!.drawn <;> simp_all
  exact keepW_draw this rfl

theorem frz_start {s : State} {now : Nat} {r : Result} {s' : State} (he : start s now = (r, s')) : Frz s s' := by
  unfold start State.releaseCoin at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact frz_refl _
    | exact frz_fail _ _ _ _
    | (refine frz_of ?_ rfl rfl
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))
    | (refine frz_trans (frz_of ?_ rfl rfl) (frz_fail _ _ _ _)
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))

theorem frz_boundary {s : State} {now : Nat} {r : Result} {s' : State} (he : boundary s now = (r, s')) : Frz s s' := by
  unfold boundary State.releaseCoin at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals cases he
  all_goals first
    | exact frz_refl _
    | exact frz_fail _ _ _ _
    | (refine frz_of ?_ rfl rfl
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))
    | (refine frz_trans (frz_of ?_ rfl rfl) (frz_fail _ _ _ _)
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))

theorem frz_append {s : State} {root : ByteArray} {count : Int} {r : Result} {s' : State}
    (he : append s root count = .ok (r, s')) : Frz s s' := by
  unfold append at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact frz_refl _
    | exact frz_fail _ _ _ _
    | (refine frz_of ?_ rfl rfl
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))
    | (refine frz_trans (frz_of ?_ rfl rfl) (frz_fail _ _ _ _)
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))
    | exact ⟨keepsF_refl _, rfl, #[_], Array.push_eq_append⟩

theorem frz_release {s : State} {runs withdrawn : Array (Int × Int)} {r : Result} {s' : State}
    (he : release s runs withdrawn = (r, s')) : Frz s s' := by
  unfold release at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact frz_refl _
    | exact frz_fail _ _ _ _
    | (refine frz_of ?_ rfl rfl
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))
    | (refine frz_trans (frz_of ?_ rfl rfl) (frz_fail _ _ _ _)
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))

theorem frz_verify {s : State} {t : Int} {openings : Array Opening} {r : Result} {s' : State}
    (he : verify s t openings = (r, s')) : Frz s s' := by
  unfold verify at he
  simp only [Id.run] at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals first
    | exact frz_refl _
    | exact frz_fail _ _ _ _
    | (refine frz_of ?_ rfl rfl
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))
    | (refine frz_trans (frz_of ?_ rfl rfl) (frz_fail _ _ _ _)
       first
         | exact keepsF_refl _
         | (refine keepsF_push _ _ ?_; rfl)
         | (refine keepsF_modify _ _ _ (keepW_of ?_ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | (intro; rfl))
         | (refine keepsF_trans (keepsF_modify _ _ _ (keepW_of ?_ ?_)) (keepsF_push _ _ ?_) <;>
             first | exact ⟨rfl, rfl, rfl, rfl, rfl⟩ | exact id | rfl))

theorem frz_close {s : State} {t : Int} {now : Nat} {r : Result} {s' : State}
    (he : close s t now = .ok (r, s')) : Frz s s' := by
  unfold close at he
  dsimp only at he
  repeat' split at he
  all_goals (try dsimp only at he)
  all_goals cases he
  all_goals refine frz_of ?_ rfl rfl
  all_goals (try exact keepsF_refl _)
  all_goals first
    | (have _ := ‹draws _ _ _ _ _ = some _›
       show KeepsF s.windows ((s.windows.modify t.toNat _).modify t.toNat _)
       exact keepsF_draw _ _ _ _ ‹¬s.windows[t.toNat]!.drawn.isSome = true›)
    | (refine keepsF_trans ?_ (keepsF_fail _ _ _ _)
       exact keepsF_modify _ _ _ (keepW_of ⟨rfl, rfl, rfl, rfl, rfl⟩ (fun _ => rfl)))

theorem df_undrawn {ws : Array Win} (h : DF ws) (u : Nat) (hf : ¬ws[u]!.final = true) : ws[u]!.drawn = none := by
  cases hd : ws[u]!.drawn with
  | none => rfl
  | some _ => exact absurd (h u (by rw [hd]; rfl)) hf

theorem advance_undrawn {s : State} (h : Inv s) (hm : 1 ≤ s.params.minPasses) {t i : Int}
    (hc : (t == ↑s.filling + 1 && i == 0 && decide (t < ↑s.windows.size)) = true) :
    (s.windows.modify s.filling ({· with final := true}))[t.toNat]!.drawn = none := by
  simp only [Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq] at hc
  obtain ⟨⟨h1, -⟩, h3⟩ := hc
  have hne : ¬(s.filling = t.toNat ∧ t.toNat < s.windows.size) := fun ⟨e, _⟩ => by omega
  rw [getElem!_modify', ite_eq_right hne]
  have ht : t.toNat < s.windows.size := by omega
  have h' : InvOn _ _ _ _ _ _ _ _ _ _ := h
  have hb := h'.beyond t.toNat ht (by omega)
  rw [getElem!_pos s.windows t.toNat ht]
  cases hd : s.windows[t.toNat].drawn with
  | none => rfl
  | some _ =>
    have := h'.few t.toNat ht (by rw [hd]; rfl)
    omega

theorem frz_receive {s : State} {t i : Int} {root : ByteArray} {runs : Option (Array (Int × Int))}
    {change : Option (Array (Int × Int) × Array (Int × Int))} {st : Option Int} {now : Nat} {r : Result}
    {s' : State} (h : Inv s) (hdf : DF s.windows) (hm : 1 ≤ s.params.minPasses)
    (he : receive s t i root runs change st now = (r, s')) : Frz s s' := by
  unfold receive at he
  split at he
  · cases he; exact frz_refl _
  split at he
  · cases he; exact frz_refl _
  split at he
  · cases he; exact frz_fail _ _ _ _
  change (have s := s; _) = _ at he
  extract_lets s₀ jp s₁ at he
  have key : ∀ s₂ : State, Frz s s₂ → s₂.windows[s₂.filling]!.drawn = none → jp () s₂ = (r, s') → Frz s s' := by
    intro s₂ k₂ hu hj
    dsimp only [jp] at hj
    repeat' split at hj
    all_goals (try dsimp only at hj)
    all_goals cases hj
    all_goals refine frz_trans k₂ (frz_of ?_ rfl rfl)
    all_goals first
      | (split
         · exact keepsF_modify _ _ _ (keepW_of ⟨rfl, rfl, rfl, rfl, rfl⟩ id)
         · exact keepsF_refl _)
      | (refine keepsF_modify _ _ _ (keepW_undrawn hu ?_ ?_) <;> rfl)
  split at he
  · exact key s₁ (frz_of (keepsF_modify _ _ _ (keepW_of ⟨rfl, rfl, rfl, rfl, rfl⟩ (fun _ => rfl))) rfl rfl)
      (advance_undrawn h hm ‹_›) he
  · split at he
    · cases he; exact frz_fail _ _ _ _
    · refine key s₀ (frz_refl _) (df_undrawn hdf _ ?_) he
      intro hf
      apply ‹¬_›
      simp only [Bool.or_eq_true]
      exact Or.inl (Or.inr hf)

theorem frz_step {s : State} {e : Event} {r : Result} {s' : State} (h : Inv s) (hdf : DF s.windows)
    (hm : 1 ≤ s.params.minPasses) (he : step s e = .ok (r, s')) : Frz s s' := by
  cases e with
  | start now => injection he with he; exact frz_start he
  | boundary now => injection he with he; exact frz_boundary he
  | append root count => exact frz_append he
  | release runs withdrawn => injection he with he; exact frz_release he
  | receive t i root runs change st now => injection he with he; exact frz_receive h hdf hm he
  | close t now => exact frz_close he
  | verify t openings => injection he with he; exact frz_verify he

/-- What `Frz` needs of a state: the invariant, every drawn window final, and `min_passes` at least 1. -/
def FInv (s : State) : Prop := Inv s ∧ DF s.windows ∧ 1 ≤ s.params.minPasses

theorem finv_step {s : State} {e : Event} {r : Result} {s' : State} (h : FInv s) (he : step s e = .ok (r, s')) :
    FInv s' ∧ Frz s s' := by
  obtain ⟨hi, hdf, hm⟩ := h
  have hf := frz_step hi hdf hm he
  obtain ⟨hi', hp, -, -⟩ := step_inv hi he
  exact ⟨⟨hi', df_keepsF hdf hf.1, by rw [hp]; exact hm⟩, hf⟩

theorem run_frz (s₀ : State) (events : Array Event) (rs : Array Result) (s : State) (h : FInv s₀)
    (hr : run s₀ events = .ok (rs, s)) : FInv s ∧ Frz s₀ s :=
  run_inv (fun s => FInv s ∧ Frz s₀ s) (fun _ _ _ _ ⟨hs, hf⟩ he =>
    have ⟨hs', hf'⟩ := finv_step hs he; ⟨hs', frz_trans hf hf'⟩) s₀ events rs s ⟨h, frz_refl _⟩ hr

theorem finv_init (params : Params) (reg : Registration) (coins appendCoins : Array ByteArray) (notBeforeNs : Nat)
    (hm : 1 ≤ params.minPasses) : FInv (State.init params reg coins appendCoins notBeforeNs) :=
  ⟨inv_init .., fun u h => (by simp [State.init] at h; cases h), hm⟩

/-! ## The record and the appends -/

theorem dr_eq (ws : Array Win) (t : Nat) : dr ws t = ws[t]!.drawn := by
  by_cases ht : t < ws.size
  · simp [dr, Array.getElem?_eq_getElem ht, getElem!_pos ws t ht]
  · rw [dr_none (by omega), getElem!_neg ws t ht]; rfl

theorem map_first (xs : Array Append) (a : Nat) (h : a < xs.size) : (xs.map (·.first))[a]! = xs[a]!.first := by
  rw [getElem!_pos _ a (by simpa using h), Array.getElem_map, getElem!_pos xs a h]

/-- Appending appends whose firsts lie above `j` leaves the append holding `j` as it was. -/
theorem appendOf_extend (A more : Array Append) (j : Nat) (hmono : Monotone' ((A ++ more).map (·.first)))
    (hbig : ∀ k, A.size ≤ k → k < (A ++ more).size → j < (A ++ more)[k]!.first) :
    appendOf (A ++ more) j = appendOf A j := by
  have hF : ∀ a, a < A.size → ((A ++ more).map (·.first))[a]! = (A.map (·.first))[a]! := by
    intro a ha
    rw [map_first _ a (by simp; omega), map_first _ a ha, getElem!_pos (A ++ more) a (by simp; omega),
      getElem!_pos A a ha, Array.getElem_append_left ha]
  have hmono' : Monotone' (A.map (·.first)) := by
    intro a b hab hb
    simp only [Array.size_map] at hb
    rw [← hF a (by omega), ← hF b hb]
    exact hmono a b hab (by simp; omega)
  obtain ⟨l', b', a'⟩ := bisectRight_spec ((A ++ more).map (·.first)) j hmono
  obtain ⟨l, b, a⟩ := bisectRight_spec (A.map (·.first)) j hmono'
  simp only [Array.size_map, Array.size_append] at l' l a' a
  have hk' : bisectRight ((A ++ more).map (·.first)) j ≤ A.size := by
    rcases Nat.lt_or_ge A.size (bisectRight ((A ++ more).map (·.first)) j) with hc | hc
    · have h1 := b' A.size hc
      rw [map_first _ _ (by simp; omega)] at h1
      have h2 := hbig A.size (Nat.le_refl _) (by simp; omega)
      omega
    · exact hc
  have heq : bisectRight ((A ++ more).map (·.first)) j = bisectRight (A.map (·.first)) j := by
    rcases Nat.lt_trichotomy (bisectRight ((A ++ more).map (·.first)) j) (bisectRight (A.map (·.first)) j) with
      hc | hc | hc
    · have h1 := b _ hc
      have h2 := a' _ (Nat.le_refl _) (by omega)
      rw [hF _ (by omega)] at h2
      omega
    · exact hc
    · have h1 := b' _ hc
      have h2 := a _ (Nat.le_refl _) (by omega)
      rw [hF _ (by omega)] at h1
      omega
  unfold appendOf
  dsimp only
  rw [heq]
  split
  · rfl
  · rw [Array.getElem?_append_left (by omega)]

theorem checkOpening_congr (reg : Registration) (digest : ByteArray) (A A' : Array Append) (w : WindowRecord)
    (rc : Receipter) (i p : Nat) (o : Opening)
    (h : appendOf A' (w.passes[i]!.plan.block p) = appendOf A (w.passes[i]!.plan.block p)) :
    checkOpening reg digest A' w rc i p o = checkOpening reg digest A w rc i p o := by
  unfold checkOpening
  simp only [Id.run, h]

theorem record_pass (s : State) (t i : Nat) (hi : i < (s.record t).passes.size) :
    (s.record t).passes[i]!.plan = s.windows[t]!.plans[i]! ∧ i < s.windows[t]!.roots.size := by
  unfold State.record at hi ⊢
  simp only [Array.size_map, Array.size_range] at hi
  refine ⟨?_, hi⟩
  rw [getElem!_pos _ i (by simp [hi]), Array.getElem_map, Array.getElem_range]

/-- A drawn window's receipts read blocks below the store's extent. -/
theorem record_reads {s : State} (h : Inv s) (t : Nat) (ht : t < s.windows.size) (i p : Nat)
    (hi : i < (s.record t).passes.size) (hp : p < (s.record t).passes[i]!.plan.length) :
    (s.record t).passes[i]!.plan.block p < s.extent := by
  obtain ⟨e, hr⟩ := record_pass s t i hi
  rw [e] at hp ⊢
  have h' : InvOn _ _ _ _ _ _ _ _ _ _ := h
  rw [getElem!_pos s.windows t ht] at hr hp ⊢
  have hh : i < s.windows[t].heads.size := by rw [(h'.sizes t ht).2.2]; exact hr
  obtain ⟨m, hm, -, hb⟩ := h'.heads t ht i hh
  have := hb p hp
  have hle : ((s.appends.toList.take m).map (·.count)).sum ≤ (s.appends.toList.map (·.count)).sum := by
    rw [List.map_take]
    have := sum_take_mono (s.appends.toList.map (·.count)) m (s.appends.toList.map (·.count)).length
      (by simp; omega)
    rwa [List.take_length] at this
  rw [h'.extentSum]
  omega

theorem firsts_mono {s : State} (h : Inv s) : Monotone' (s.appends.map (·.first)) := by
  have h' : InvOn _ _ _ _ _ _ _ _ _ _ := h
  intro a b hab hb
  simp only [Array.size_map] at hb
  rw [map_first _ a (by omega), map_first _ b hb, getElem!_pos _ a (by omega), getElem!_pos _ b hb,
    h'.firsts a (by omega), h'.firsts b hb, List.map_take, List.map_take]
  have := sum_take_mono (s.appends.toList.map (·.count)) a b hab
  omega

/-- The appends past `s₁`'s start at its extent or above. -/
theorem firsts_big {s₁ s : State} (h₁ : Inv s₁) (h : Inv s) (more : Array Append) (ha : s.appends = s₁.appends ++ more)
    (hr : s.reg = s₁.reg) (k : Nat) (hk : s₁.appends.size ≤ k) (hk' : k < s.appends.size) :
    s₁.extent ≤ s.appends[k]!.first := by
  have h₁' : InvOn _ _ _ _ _ _ _ _ _ _ := h₁
  have h' : InvOn _ _ _ _ _ _ _ _ _ _ := h
  rw [getElem!_pos _ k hk', h'.firsts k hk', h₁'.extentSum, hr, List.map_take]
  have hs := sum_take_mono (s.appends.toList.map (·.count)) s₁.appends.size k hk
  have e : (s.appends.toList.map (·.count)).take s₁.appends.size = s₁.appends.toList.map (·.count) := by
    rw [← List.map_take, ha, Array.toList_append, List.take_left' (by simp)]
  rw [e] at hs
  omega

theorem recordKeptAfterDraw : RecordKeptAfterDraw := by
  intro params reg coins appendCoins notBeforeNs events events' rs rs' s s' h₁ h₂ t d hm hd
  obtain ⟨-, -, hr₁, -⟩ := inv_run h₁
  obtain ⟨hF₁, -⟩ := run_frz _ _ _ _ (finv_init params reg coins appendCoins notBeforeNs hm) h₁
  obtain ⟨hF, hk, hdig, more, happ⟩ := run_frz _ _ _ _ hF₁ h₂
  have hI₁ : Inv s := hF₁.1
  have hI : Inv s' := hF.1
  have hr : s'.reg = s.reg := (run_inv (fun x => Inv x ∧ x.reg = s.reg)
    (fun _ _ _ _ ⟨hx, e⟩ he => have ⟨hx', f⟩ := step_inv hx he; ⟨hx', f.2.1.trans e⟩) _ _ _ _ ⟨hI₁, rfl⟩ h₂).2
  have hpre : Prefix s.windows :=
    run_inv (fun s => Prefix s.windows) (fun _ _ _ _ hs he => prefix_step hs (adv_step he)) _ _ _ _
      (fun t j _ ht => by simp [dr, State.init] at ht) h₁
  have hd' : s.windows[t]!.drawn = some d := by rw [← dr_eq]; exact hd
  have ht : t < s.windows.size := by
    rcases Nat.lt_or_ge t s.windows.size with h | h
    · exact h
    · rw [getElem!_neg _ _ (by omega)] at hd'; cases hd'
  obtain ⟨e1, e2, e3, e4, e5⟩ := (hk t).1 (by rw [hd']; rfl)
  have hrec : s'.record t = s.record t := by
    have hprev : (if t > 0 then s'.windows[t - 1]!.roots.back?.getD ByteArray.empty else s'.digest) =
        (if t > 0 then s.windows[t - 1]!.roots.back?.getD ByteArray.empty else s.digest) := by
      by_cases htp : t > 0
      · have hs : (dr s.windows (t - 1)).isSome := hpre t (t - 1) (by omega) (by show (drawnOf s t).isSome; rw [hd]; rfl)
        rw [dr_eq] at hs
        rw [ite_eq_left htp, ite_eq_left htp, ((hk (t - 1)).1 hs).2.2.1]
      · rw [ite_eq_right htp, ite_eq_right htp, hdig]
    unfold State.record
    simp only [e2, e3, e4, e5]
    rw [hprev]
  refine ⟨by rw [show drawnOf s' t = s'.windows[t]!.drawn from dr_eq _ _, e1, hd'], hrec, fun u => ?_⟩
  rw [hrec, happ]
  have key : ∀ i p o, i < (s.record t).passes.size → p < (s.record t).passes[i]!.plan.length →
      checkOpening reg reg.digest (s.appends ++ more) (s.record t) (receipterOf reg (s.record t) i) i p o =
      checkOpening reg reg.digest s.appends (s.record t) (receipterOf reg (s.record t) i) i p o := by
    intro i p o hi hp
    apply checkOpening_congr
    apply appendOf_extend
    · rw [← happ]; exact firsts_mono hI
    · intro k hk hk'
      have := firsts_big hI₁ hI more happ hr k hk (by rw [happ]; exact hk')
      have hb := record_reads hI₁ t ht i p hi hp
      rw [← happ]; omega
  constructor
  · rintro ⟨i, p, o, hi, hp, hu, hc⟩; exact ⟨i, p, o, hi, hp, hu, (key i p o hi hp) ▸ hc⟩
  · rintro ⟨i, p, o, hi, hp, hu, hc⟩; exact ⟨i, p, o, hi, hp, hu, (key i p o hi hp).symm ▸ hc⟩

end Properties.PoubCheck
