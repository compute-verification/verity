module

public import Properties.Flock.Circuit.Keyed

/-!
# `uniform`'s gates, units and rows

`Placed.uniform k R ins one hone n` has `k + n·nComp` gates (`gates_eq`). Gate `g < k` is an input no unit holds
(`op_uniform_input`, `unit_uniform_val`), and gate `k + v·nComp + i` is unit `v`'s row `i`, its columns wired as
`colVal` says (`op_uniform_row`).
-/

@[expose] public section

namespace Flock.Bridge.FilePart

open FlockSoundness FlockSoundness.Discharge.Placed

theorem gates_eq (k : ℕ) (R : Rows) : ∀ n, gates k R n = k + n * R.nComp
  | 0 => by simp [gates]
  | n + 1 => by rw [gates, gates_eq k R n, Nat.add_mul, Nat.one_mul, Nat.add_assoc]

/-- Unit `v`'s column `c` as `uniform` wires it, a gate: its input's, its computed row's, or the constant's. -/
def colVal (k : ℕ) (R : Rows) (ins : ℕ → Fin R.nIn → Fin k) (one : Fin k) (v : ℕ) (c : Fin R.w) : ℕ :=
  if h : c.val < R.nIn then (ins v ⟨c.val, h⟩).val
  else if c.val < R.nIn + R.nComp then k + v * R.nComp + (c.val - R.nIn) else one.val

variable {k : ℕ} {R : Rows} {ins : ℕ → Fin R.nIn → Fin k} {one : Fin k} {hone : ∀ u c, ins u c ≠ one}

/-- **`uniform`'s units**: an input gate is none's, and gate `k + v·nComp + i` is unit `v`'s. -/
theorem unit_uniform_val : ∀ (n : ℕ) (g : Fin (gates k R n)),
    ((uniform k R ins one hone n).unit g).map Fin.val = if g.val < k then none else some ((g.val - k) / R.nComp)
  | 0, g => by
    have : g.val < k := g.isLt
    simp [uniform, Prog.unit, this]
  | n + 1, g => by
    have hg := g.isLt
    have hgn : gates k R (n + 1) = k + n * R.nComp + R.nComp := by rw [gates_eq]; ring
    change ((Prog.snoc (uniform k R ins one hone n) R _ _ _).unit g).map Fin.val = _
    by_cases h : g.val < gates k R n
    · simp only [Prog.unit, h, ↓reduceDIte, Option.map_map]
      have := unit_uniform_val n ⟨g.val, h⟩
      rw [← this]
      rfl
    · simp only [Prog.unit, h, ↓reduceDIte, Option.map_some, Fin.val_last]
      have hG := gates_eq k R n
      have hk : ¬ g.val < k := by omega
      rw [if_neg hk]
      congr 1
      symm
      apply Nat.div_eq_of_lt_le
      · omega
      · rw [Nat.add_mul, Nat.one_mul]; omega

/-- `uniform`'s input gates are inputs. -/
theorem op_uniform_input : ∀ (n : ℕ) (g : Fin (gates k R n)), g.val < k →
    (uniform k R ins one hone n).op g = .input
  | 0, _, _ => rfl
  | n + 1, g, hg => by
    have h : g.val < gates k R n := lt_of_lt_of_le hg (le_gates k R n)
    change (Prog.snoc (uniform k R ins one hone n) R _ _ _).op g = _
    simp only [Prog.op, h, ↓reduceDIte, op_uniform_input n ⟨g.val, h⟩ hg]
    rfl

/-- **`uniform`'s computed gates are rows**: gate `k + v·nComp + i` is unit `v`'s row `i`, its columns wired as
`colVal` says. -/
theorem op_uniform_row : ∀ (n : ℕ) (g : Fin (gates k R n)), k ≤ g.val →
    ∃ (i : Fin R.nComp) (a b : List (Fin (gates k R n))),
      (g.val - k) % R.nComp = i.val ∧ (uniform k R ins one hone n).op g = .row a b ∧
      a.map Fin.val = (R.a i).map (colVal k R ins one ((g.val - k) / R.nComp)) ∧
      b.map Fin.val = (R.b i).map (colVal k R ins one ((g.val - k) / R.nComp))
  | 0, g, hg => absurd g.isLt (by show ¬ g.val < k; omega)
  | n + 1, g, hg => by
    have hgl := g.isLt
    have hgn : gates k R (n + 1) = k + n * R.nComp + R.nComp := by rw [gates_eq]; ring
    change ∃ (i : Fin R.nComp) (a b : List (Fin (gates k R (n + 1)))), _ ∧
      (Prog.snoc (uniform k R ins one hone n) R _ _ _).op g = .row a b ∧ _
    by_cases h : g.val < gates k R n
    · obtain ⟨i, a, b, hi, ho, ha, hb⟩ := op_uniform_row n ⟨g.val, h⟩ hg
      refine ⟨i, a.map (Fin.castAdd R.nComp), b.map (Fin.castAdd R.nComp), hi, ?_, ?_, ?_⟩
      · simp only [Prog.op, h, ↓reduceDIte, ho]; rfl
      · exact List.map_map.trans ha
      · exact List.map_map.trans hb
    · have hG := gates_eq k R n
      have hv : (g.val - k) / R.nComp = n := by
        apply Nat.div_eq_of_lt_le
        · omega
        · rw [Nat.add_mul, Nat.one_mul]; omega
      have hm : (g.val - k) % R.nComp = g.val - gates k R n := by
        have := Nat.div_add_mod' (g.val - k) R.nComp
        rw [hv] at this
        omega
      have hlt : g.val - gates k R n < R.nComp := by omega
      refine ⟨⟨g.val - gates k R n, hlt⟩,
        (R.a ⟨g.val - gates k R n, hlt⟩).map
          (Prog.colWire R (fun c => Fin.castLE (le_gates k R n) (ins n c)) (Fin.castLE (le_gates k R n) one)),
        (R.b ⟨g.val - gates k R n, hlt⟩).map
          (Prog.colWire R (fun c => Fin.castLE (le_gates k R n) (ins n c)) (Fin.castLE (le_gates k R n) one)),
        hm, ?_, ?_, ?_⟩
      · simp only [Prog.op]
        rw [dif_neg h]
        rfl
      · rw [hv]
        refine List.map_map.trans (List.map_congr_left fun c _ => ?_)
        simp only [Function.comp, Prog.colWire_val, colVal, Fin.val_castLE, hG]
      · rw [hv]
        refine List.map_map.trans (List.map_congr_left fun c _ => ?_)
        simp only [Function.comp, Prog.colWire_val, colVal, Fin.val_castLE, hG]

end Flock.Bridge.FilePart
