module

public import Proofs.Properties.Nci.RecordPrim
public import Proofs.Properties.Nci.RecordRowBits
public import Spot.WiresSound
public import Mathlib.Tactic

/-!
# A unit's rows

A unit reads through its parameters, and the record cuts each parameter into rows: parameter `i` of a unit, cut
`k` ways, is `k` rows of `leaves / k` leaves each.

* `rowLeaf_of_param`: a leaf of a unit's parameter is a leaf of one of the unit's rows. Leaf `t` is leaf
  `t % (leaves / k)` of row `t / (leaves / k)`. The cut divides the parameter's leaves: where the record declares
  none it is 1, and a declared cut divides the port's outer length (`Spot.Wires.cutsOk`, inside `Spot.Wires.check`),
  which is the parameter's (`TemplateOf`) and divides its leaves.
* `ranked`: the wires `Spot.Wires.check` accepts rank the units, every row's writer below the row's unit.
-/

@[expose] public section

namespace NciCheck.Properties.Record.SecurityProofs

open Flock.Program (Prog Scope Ref Ty)
open Flock.TemplateQuery (TRange)
open NciCheck.RecordText (Row outerOf)
open NciCheck.Properties.SecurityProofs

variable {P : Prog} {rs : Array TRange} {W : Spot.Wires.Wiring} {S : Spot.Wires.Summary}

/-! ## The cut of a port divides its parameter's leaves -/

/-- A parameter's outer length divides its leaves. -/
private theorem outerOf_dvd (ty : Ty) : outerOf ty ∣ ty.leaves := by
  cases ty with
  | array n e =>
    rw [BodiesOk.leaves_array]
    show n ∣ n * e.leaves
    exact Nat.dvd_mul_right n _
  | value w => exact Nat.one_dvd _
  | tuple fs => exact Nat.one_dvd _
  | record fs => exact Nat.one_dvd _

/-- A wiring `Spot.Wires.check` accepts has sound cuts. -/
private theorem check_cuts (hS : Spot.Wires.check W = .ok S) : Spot.Wires.cutsOk W = true := by
  unfold Spot.Wires.check at hS
  split at hS
  · cases hS
  · rename_i hcut
    simpa using hcut

/-- A declared cut names a template of the wiring and one of its rows, is positive and divides the row's outer
length. -/
private theorem cut_dvd (hc : Spot.Wires.cutsOk W = true) {tid port : String} {kv : List (String × ℕ)} {k : ℕ}
    (hkv : W.cuts.lookup tid = some kv) (hk : kv.lookup port = some k) :
    ∃ t q, (tid, t) ∈ W.templates ∧ (port, q) ∈ t.rows ∧ 0 < k ∧ k ∣ q.outer := by
  unfold Spot.Wires.cutsOk at hc
  have h1 := List.all_eq_true.1 hc (tid, kv) (Spot.Wires.lookup_mem hkv)
  dsimp only at h1
  split at h1
  · rename_i t ht
    have h2 := List.all_eq_true.1 h1 (port, k) (Spot.Wires.lookup_mem hk)
    dsimp only at h2
    split at h2
    · rename_i q hq
      simp only [Bool.and_eq_true, decide_eq_true_eq, beq_iff_eq] at h2
      exact ⟨t, q, Spot.Wires.lookup_mem ht, Spot.Wires.lookup_mem hq, h2.1.1, Nat.dvd_of_mod_eq_zero h2.1.2⟩
    · cases h2
  · cases h1

/-- **The cut of a unit's port is positive and divides the parameter's leaves.** -/
private theorem cutsOf_dvd (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i : ℕ} (hi : i < (P.defs[sc.d]!).params.size) :
    0 < W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1 ∧
      W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1 ∣ ((P.defs[sc.d]!).params[i]!).2.leaves := by
  unfold Spot.Wires.Wiring.cutsOf
  cases hkv : W.cuts.lookup tid with
  | none => simp
  | some kv =>
    cases hk : kv.lookup ((P.defs[sc.d]!).params[i]!).1 with
    | none => simp [hk]
    | some k =>
      obtain ⟨t, q, ht, hq, hk0, hkq⟩ := cut_dvd (check_cuts hS) hkv hk
      obtain ⟨hinj, -, hrows, -⟩ := (hT tid t ht).2 u sc hu
      obtain ⟨i', hi', hname, -, hout⟩ := (hrows _ q).1 hq
      have hii : i' = i := hinj i' i hi' hi hname.symm
      rw [hii] at hout
      rw [hout] at hkq
      simp only [Option.getD_some, hk]
      exact ⟨hk0, Nat.dvd_trans hkq (outerOf_dvd _)⟩

/-! ## R1 -/

/-- **A leaf of a unit's parameter is a leaf of one of the unit's rows**: the row its cut puts it in. -/
theorem rowLeaf_of_param (hT : Templates P rs W) (hS : Spot.Wires.check W = .ok S) {tid : String} {u : ℕ}
    {sc : Scope} (hu : UnitAt P rs tid u sc) {i t g : ℕ} (hi : i < (P.defs[sc.d]!).params.size)
    (ht : t < ((P.defs[sc.d]!).params[i]!).2.leaves)
    (hg : Flock.Program.resolve P sc (⟨true, i, t⟩ : Ref) = .ok g) :
    ∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l g lo hi := by
  obtain ⟨hk0, q, hq⟩ := cutsOf_dvd hT hS hu hi
  obtain ⟨k, hkdef⟩ : ∃ k, W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1 = k := ⟨_, rfl⟩
  rw [hkdef] at hk0 hq
  have ht' : t < k * q := by rw [← hq]; exact ht
  have hq0 : 0 < q := by
    rcases Nat.eq_zero_or_pos q with h | h
    · rw [h, Nat.mul_zero] at ht'
      exact absurd ht' (Nat.not_lt_zero _)
    · exact h
  have hdiv : ((P.defs[sc.d]!).params[i]!).2.leaves / k = q := by
    rw [hq]
    exact Nat.mul_div_cancel_left q hk0
  have hrow : t / q < k := (Nat.div_lt_iff_lt_mul hq0).2 ht'
  refine ⟨⟨tid, u, ((P.defs[sc.d]!).params[i]!).1, t / q⟩, t % q, _, _, rfl, rfl,
    sc, hu, i, hi, rfl, ?_, ?_, ?_, rfl, rfl⟩
  · show t / q < W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1
    rw [hkdef]
    exact hrow
  · show t % q < (inRow (P.defs[sc.d]!) i (W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1) (t / q)).2.size
    rw [hkdef, inRow_size _ _ _ _ hrow, hdiv]
    exact Nat.mod_lt _ hq0
  · show Flock.Program.resolve P sc (⟨true, i, t / q *
      (((P.defs[sc.d]!).params[i]!).2.leaves / W.cutsOf tid ((P.defs[sc.d]!).params[i]!).1) + t % q⟩ : Ref) = .ok g
    rw [hkdef, hdiv, Nat.div_add_mod' t q]
    exact hg

/-! ## R5 -/

/-- **The wires rank the units**: under a rank `Spot.Wires.check` gives, every row's writer ranks below the row's
unit. -/
theorem ranked (hS : Spot.Wires.check W = .ok S) : ∃ ρ : String × ℤ → ℕ, Ranked S ρ := by
  obtain ⟨ρ, hρ⟩ := Spot.Wires.check_ranked hS
  refine ⟨ρ, ?_⟩
  intro r wt wu out hw
  exact hρ _ _ (Spot.Wires.writerOf_reads hS hw)

end NciCheck.Properties.Record.SecurityProofs
