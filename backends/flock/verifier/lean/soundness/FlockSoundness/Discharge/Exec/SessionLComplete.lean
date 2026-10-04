import FlockSoundness.Discharge.Exec.SessionL
import FlockSoundness.Audit.OneStage

/-!
# Weak completeness of the salted batched session

A game is decisive (`Decisive`) when every message type it asks for is inhabited (`Productive`) and two of its runs,
for some messages and some coins (`Reaches`), end differently. These are PoUW's `Pouw.Draft.Game.Decisive`,
`Productive` and `Reaches`, restated over `FlockSoundness.Game`.

The salted batched session `batchedSessionL` is decisive on every plan with a table, and so is the one-stage audit
over it (`audit_tile_decisive`, for the one-table plans of a tiling). No table needs to be satisfiable and no schedule
needs a side condition:

* **acceptance** (`batchedSessionL_reaches_true`): the all-zero run accepts every table. With zero messages and zero
  coins every running claim stays `0` (zerocheck, lincheck, every Ligerito fold, out-of-domain sample and consistency
  batch), except the region claims, whose ring-switch message puts the claim's value on the packed position where the
  claim's weight is `1` (at zero link points the region points are `0/1`). A zero residual `yr` makes the final sum
  `0`. `finalCheck`'s length conjunct `orig.length + ext = m − 7` is `Schedule.WellShaped`'s shape identity, which
  `TabSpec.hsch` gives through `fast100`. Every level's rows are zero rows under the default salt, so one value per
  level reaches every cap entry, whatever the positions;
* **rejection** (`batchedSessionL_reaches_false`): a rep whose zerocheck sends `final_a_eval = final_b_eval = 1` on
  the zero claim rejects, and with it the table and the session;
* **productivity** (`batchedSessionL_productive`): every message type of the session is inhabited.
-/

namespace FlockSoundness.Discharge.Exec.SessionLComplete

open Game Model Accounting Audit

section Defs

variable {α β : Type}

/-- Every message type `g` asks for is inhabited (coin types are, by `Game.coin`'s `Nonempty`). -/
def Productive : Game α → Prop
  | .ret _ => True
  | .send M k => Nonempty M ∧ ∀ m, Productive (k m)
  | @Game.coin _ _ _ _ k => ∀ c, Productive (k c)

/-- Some run of `g`, for some messages and some coins, ends with `a`. -/
def Reaches : Game α → α → Prop
  | .ret b, a => b = a
  | .send _ k, a => ∃ m, Reaches (k m) a
  | @Game.coin _ _ _ _ k, a => ∃ c, Reaches (k c) a

/-- **Decisive**: every message type of `g` is inhabited, and two runs of `g` end with different outputs. -/
def Decisive (g : Game α) : Prop :=
  Productive g ∧ ∃ a b, a ≠ b ∧ Reaches g a ∧ Reaches g b

theorem reaches_bind {f : α → Game β} {b : β} :
    ∀ (g : Game α) (a : α), Reaches g a → Reaches (f a) b → Reaches (g.bind f) b
  | .ret _, _, h, hf => by cases h; exact hf
  | .send _ k, a, ⟨m, h⟩, hf => ⟨m, reaches_bind (k m) a h hf⟩
  | @Game.coin _ _ _ _ k, a, ⟨c, h⟩, hf => ⟨c, reaches_bind (k c) a h hf⟩

theorem reaches_map (f : α → β) {g : Game α} {a : α} (h : Reaches g a) : Reaches (g.map f) (f a) :=
  reaches_bind (f := fun a => .ret (f a)) g a h rfl

theorem productive_bind {f : α → Game β} (hf : ∀ a, Productive (f a)) :
    ∀ g : Game α, Productive g → Productive (g.bind f)
  | .ret a, _ => hf a
  | .send _ k, ⟨hM, h⟩ => ⟨hM, fun m => productive_bind hf (k m) (h m)⟩
  | @Game.coin _ _ _ _ k, h => fun c => productive_bind hf (k c) (h c)

theorem productive_map (f : α → β) {g : Game α} (h : Productive g) : Productive (g.map f) :=
  productive_bind (f := fun a => .ret (f a)) (fun _ => trivial) g h

end Defs

local notation "𝔽" => FlockLevel3.GF128

variable {D Sl : Type} [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

/-! ## The schedule's shape -/

/-- Every `fast100` schedule folds `k0` lanes at level 0 and one fewer than each later level's lanes after it, and
these with `ext` make up the `m − 7` packed coordinates. -/
theorem fast100_orig_length : ∀ m s, fast100 m = some s →
    s.k0 + ((s.levels.drop 1).map fun lv => lv.lanes - 1).sum + s.ext = s.m - 7 := by
  intro m s h
  have hm : 22 ≤ m ∧ m ≤ 35 := by
    by_contra hc
    have : fast100 m = none := by
      unfold fast100
      split <;> first | rfl | omega
    simp [this] at h
  obtain ⟨h1, h2⟩ := hm
  interval_cases m <;> (simp only [fast100, Option.some.injEq] at h; subst h; decide +kernel)

/-- **`finalCheck`'s length condition, for a table**: the level-0 folds and the later levels' folds past their first
challenge, with the clear residual, are `m − 7` coordinates. -/
theorem TabSpec.orig_length (T : TabSpec mPts) :
    T.k0 + ((T.l₁ :: T.rest).map fun lv => lv.lanes - 1).sum + T.ext = T.m - 7 :=
  fast100_orig_length T.S.m T.sch T.hsch

/-! ## The session -/

/-- **Can accept.** For every hash, encoding, leaf scheme, plan, draw and registration, some run of the salted batched
session accepts. -/
theorem batchedSessionL_reaches_true (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) :
    Reaches (batchedSessionL H E Lf plan S R) true := by
  sorry

/-- **Can reject.** When the plan has a table on the draw, some run of the salted batched session rejects. -/
theorem batchedSessionL_reaches_false (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) (hne : plan S R ≠ []) :
    Reaches (batchedSessionL H E Lf plan S R) false := by
  sorry

/-- **Productive.** Every message type the salted batched session asks for is inhabited. -/
theorem batchedSessionL_productive (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) :
    Productive (batchedSessionL H E Lf plan S R) := by
  sorry

/-- **The salted batched session is decisive** on a draw where the plan has a table. -/
theorem batchedSessionL_decisive (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (S : Finset (Fin n)) (R : Reg) (hne : plan S R ≠ []) :
    Decisive (batchedSessionL H E Lf plan S R) :=
  ⟨batchedSessionL_productive H E Lf plan S R, true, false, by decide,
    batchedSessionL_reaches_true H E Lf plan S R, batchedSessionL_reaches_false H E Lf plan S R hne⟩

/-! ## The one-stage audit over it -/

/-- **The audit can accept**, on every registration and every draw. -/
theorem audit_reaches_accept (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (R : Reg) (ω : L.Ω) :
    Reaches (audit L Reg (batchedSessionL H E Lf plan)) (true, L.draw ω) :=
  ⟨R, ω, reaches_map (fun ok => (ok, L.draw ω)) (batchedSessionL_reaches_true H E Lf plan (L.draw ω) R)⟩

/-- **The audit can reject**, on every registration and every draw where the plan has a table. -/
theorem audit_reaches_reject (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) (R : Reg) (ω : L.Ω)
    (hne : plan (L.draw ω) R ≠ []) :
    Reaches (audit L Reg (batchedSessionL H E Lf plan)) (false, L.draw ω) :=
  ⟨R, ω, reaches_map (fun ok => (ok, L.draw ω)) (batchedSessionL_reaches_false H E Lf plan (L.draw ω) R hne)⟩

/-- **The audit is productive** when registrations exist. -/
theorem audit_productive [Nonempty Reg] (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) :
    Productive (audit L Reg (batchedSessionL H E Lf plan)) :=
  ⟨inferInstance, fun R ω => productive_map _ (batchedSessionL_productive H E Lf plan (L.draw ω) R)⟩

/-- **The audit is decisive** when registrations exist and the plan has a table on every draw. -/
theorem audit_decisive [Nonempty Reg] (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (hne : ∀ S R, plan S R ≠ []) :
    Decisive (audit L Reg (batchedSessionL H E Lf plan)) := by
  obtain ⟨R⟩ := ‹Nonempty Reg›
  obtain ⟨ω⟩ := L.nonempty
  exact ⟨audit_productive L H E Lf plan, (true, L.draw ω), (false, L.draw ω), by simp,
    audit_reaches_accept L H E Lf plan R ω, audit_reaches_reject L H E Lf plan R ω (hne _ _)⟩

/-- **A tile's audit is decisive**: the one-stage audit over the salted batched session of a one-table plan. -/
theorem audit_tile_decisive [Nonempty Reg] (L : Law n) (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (T : Finset (Fin n) → Reg → TabSpec mPts) :
    Decisive (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  audit_decisive L H E Lf _ fun _ _ => List.cons_ne_nil _ _

end FlockSoundness.Discharge.Exec.SessionLComplete
