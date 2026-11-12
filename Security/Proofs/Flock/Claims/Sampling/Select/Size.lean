module

public import Proofs.Flock.Claims.Sampling.Select.Take
public import Proofs.Flock.Bridge.Draw
public import Proofs.Core.QSort

@[expose] public section

open Spot

namespace Properties.Spot.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- `subset k n` draws `k` units when it draws. -/
theorem subset_size {k n : ℕ} {src : ByteArray} {D : Array ℕ} (h : Flock.Draw.subset k n src = some D) :
    D.size = k := by
  rw [DrawExec.subset_eq] at h
  cases hl : DrawExec.loopSw n src (List.range' 0 k) ∅ 0 with
  | none => rw [hl] at h; exact absurd h (by simp)
  | some sw =>
    rw [hl, Option.map_some, Option.some.injEq] at h
    rw [← h, Verity.QSort.size_qsort, Array.size_map, Array.size_range]

theorem sum_range'_getElem! {α : Type} [Inhabited α] (St : Array α) (f : α → ℕ) :
    ((List.range' 0 St.size).map fun i => f St[i]!).sum = (St.toList.map f).sum := by
  congr 1
  apply List.ext_getElem (by simp)
  intro i h1 _
  simp only [List.length_map, List.length_range'] at h1
  simp [List.getElem_range', getElem!_pos St i h1]

/-- **`stratifiedWith` on `take` draws its strata's `k`s**: a run that returns a draw `D` returns as many units as its
strata's draws sum to. -/
theorem stratifiedWith_size (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p p' : ℕ) (D : Array ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run p = .ok (some D, p')) :
    D.size = (strata.toList.map (·.k)).sum := by
  unfold Flock.Draw.stratifiedWith at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl] at h
  have loop : ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ) (r : Option (Option (Array ℕ))) (out' : Array ℕ) (pos₁ : ℕ),
      (forIn (m := StateT ℕ (Except String)) l ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = .ok ((r, out'), pos₁) →
      r = none ∧ out'.size = out.size + (l.map fun i => strata[i]!.k).sum := by
    intro l
    induction l with
    | nil =>
      intro out pos r out' pos₁ hl
      simp only [List.forIn_nil, StateT.run_pure] at hl
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hl)
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
      exact ⟨rfl, by simp⟩
    | cons i is ih =>
      intro out pos r out' pos₁ hl
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind] at hl
      cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)).run pos with
      | error e => rw [ho] at hl; exact absurd hl (by rw [bind_error, bind_error]; simp)
      | ok x =>
        obtain ⟨v, pos₂⟩ := x
        obtain ⟨j, Dl, rfl, rfl, hst⟩ := onWith_take src _ (by unfold Flock.Draw.Stratum.need; omega) _ pos _ _ ho
        rw [ho, bind_ok] at hl
        dsimp only at hl
        rw [StateT.run_pure, pure_bind] at hl
        dsimp only at hl
        obtain ⟨hr, hsz⟩ := ih _ _ r out' pos₁ hl
        refine ⟨hr, ?_⟩
        rw [hsz, Array.size_append, Array.size_map, subset_size hst.2.2.1, List.map_cons, List.sum_cons]
        omega
  rw [StateT.run_bind] at h
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run p with
  | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨⟨r, out'⟩, pos₁⟩ := y
    rw [hl, bind_ok] at h
    dsimp only at h
    obtain ⟨rfl, hsz⟩ := loop _ _ _ _ _ _ hl
    rw [StateT.run_pure] at h
    obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
    have hD : D = out'.qsort (fun x1 x2 => decide (x1 < x2)) := (Option.some.inj h1).symm
    rw [hD, Verity.QSort.size_qsort, hsz, Array.size_empty, Nat.zero_add]
    exact sum_range'_getElem! strata (·.k)

/-- **`drawUnits` draws the law's `k`**: a draw from `src` under a law on strata of sizes `ns` with draw `kd` (`LawOn`)
returns `kd` units. -/
theorem drawUnits_size {law : Flock.Draw.Law} {ns : List ℕ} {kd n : ℕ} {src : ByteArray} {units : Array ℕ}
    {read : ℕ} (hlaw : Flock.Bridge.LawOn law ns kd) (h : Draw.drawUnits law n src = .ok (units, read)) :
    units.size = kd := by
  unfold Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take src need) law n).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok x =>
    obtain ⟨r, read'⟩ := x
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hu, -⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hu
    cases law with
    | subset k =>
      obtain ⟨hk, -⟩ := hlaw
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) (2 * k * Flock.Draw.width n + 64)
          (Flock.Draw.subset k n)).run 0 with
      | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
      | ok y =>
        obtain ⟨v, p⟩ := y
        obtain ⟨j, D, rfl, -, hst⟩ := onWith_take src _ (by omega) _ 0 _ _ ho
        rw [ho, bind_ok, StateT.run_pure] at hw
        obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
        obtain rfl := (Option.some.inj h1).symm
        exact (subset_size hst.2.2.1).trans hk
    | bernoulli a b => exact (hlaw : False).elim
    | stratified k strata =>
      obtain ⟨-, -, -, -, hsum⟩ := hlaw
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run 0 with
      | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
      | ok y =>
        obtain ⟨v, p⟩ := y
        rw [ho, bind_ok, StateT.run_pure] at hw
        obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
        cases v with
        | none => exact absurd h1 (by simp)
        | some D =>
          obtain rfl := (Option.some.inj h1).symm
          exact (stratifiedWith_size src strata 0 p D ho).trans hsum
    | work k strata =>
      obtain ⟨-, -, hsum⟩ := hlaw
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) (strata.map (·.toStratum))).run 0 with
      | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
      | ok y =>
        obtain ⟨v, p⟩ := y
        rw [ho, bind_ok, StateT.run_pure] at hw
        obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
        cases v with
        | none => exact absurd h1 (by simp)
        | some D =>
          obtain rfl := (Option.some.inj h1).symm
          refine (stratifiedWith_size src _ 0 p D ho).trans ?_
          rw [Array.toList_map, List.map_map]
          exact hsum

end Properties.Spot.SecurityProofs
