import Proofs.Flock.EndToEnd.Served.Statement
import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The honest served draw ends

`ServedDraw`'s honest lemma takes a draw that ends: the served stratified draw of one of the units `[0, 2)`
(`Flock.Guarantees.Served.honestSt`) on the `honestLc` coins all `0`. Its one stratum's first read of `need` bytes is
the whole stream, on which `subset 1 2`'s one `uniform 2` read is the word `0`, below the largest multiple of `2`.

* `onWith_take_ok`: `onWith` on `take` ends at its first read when the sampler succeeds on it.
* `stratifiedWith_one`: a stratified draw of one stratum ends where that stratum's `onWith` ends.
* `drawUnits_stratified_ok`: `drawUnits` returns what `stratifiedWith` draws.
* `honest_drawUnits`: the honest draw ends.
-/

namespace FlockSoundness.Audit.Served

open SampledProofs Properties.SampledProofs.SecurityProofs

/-- A `repeat` loop whose first body is `done` returns its value. -/
theorem loop_done {β : Type} (f : Unit → β → StateT ℕ (Except String) (ForInStep β)) (b b' : β) (s s' : ℕ)
    (r : Except String (β × ℕ)) (hf : (f () b).run s = .ok (.done b', s'))
    (hl : (Lean.Loop.forIn Lean.Loop.mk b f).run s = r) : r = .ok (b', s') := by
  subst hl
  rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind, hf]
  rfl

/-- **`onWith` on `take` ends at its first read** when the sampler succeeds on it. -/
theorem onWith_take_ok (src : ByteArray) (N p : ℕ) (run : ByteArray → Option (Array ℕ)) (u : Array ℕ)
    (hp : p + N ≤ src.size) (hr : run (src.extract p (p + N)) = some u) :
    (Flock.Draw.onWith (fun need => Draw.take src need) N run).run p = .ok (some u, p + N) := by
  unfold Flock.Draw.onWith
  rw [StateT.run_bind, take_run, ite_eq_right (show ¬ src.size < p + N by omega), bind_ok]
  dsimp only
  rw [StateT.run_bind]
  cases hl : (forIn (m := StateT ℕ (Except String)) Lean.Loop.mk
      ((none : Option (Option (Array ℕ))), src.extract p (p + N)) _).run (p + N) with
  | error e =>
    exact absurd (loop_done _ _ ((some (some u) : Option (Option (Array ℕ))), src.extract p (p + N)) _ (p + N) _
      (by dsimp only; rw [hr]; rfl) hl) (by simp)
  | ok y =>
    obtain rfl := Except.ok.inj (loop_done _ _ ((some (some u) : Option (Option (Array ℕ))), src.extract p (p + N))
      _ (p + N) _ (by dsimp only; rw [hr]; rfl) hl)
    rfl

/-- The strata loop of `stratifiedWith` on one stratum whose `onWith` ends. -/
theorem strata_loop_one (src : ByteArray) (strata : Array Flock.Draw.Stratum) (out Dl : Array ℕ) (p : ℕ)
    (ho : (Flock.Draw.onWith (fun need => Draw.take src need) strata[0]!.need
      (Flock.Draw.subset strata[0]!.k strata[0]!.size)).run 0 = .ok (some Dl, p)) :
    (forIn (m := StateT ℕ (Except String)) [0] ((none : Option (Option (Array ℕ))), out) (fun i __s => do
        let __x ← Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)
        match __x with
        | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
        | _ => pure (ForInStep.done (some none, __s.2)))).run 0 =
      .ok (((none : Option (Option (Array ℕ))), out ++ Array.map strata[0]!.unit Dl), p) := by
  rw [List.forIn_cons, StateT.run_bind, StateT.run_bind, ho, bind_ok]
  rfl

/-- **A stratified draw of one stratum ends where its stratum's `onWith` ends.** -/
theorem stratifiedWith_one (src : ByteArray) (strata : Array Flock.Draw.Stratum) (h1 : strata.size = 1)
    (Dl : Array ℕ) (p : ℕ)
    (ho : (Flock.Draw.onWith (fun need => Draw.take src need) strata[0]!.need
      (Flock.Draw.subset strata[0]!.k strata[0]!.size)).run 0 = .ok (some Dl, p)) :
    ∃ D, (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run 0 = .ok (some D, p) := by
  unfold Flock.Draw.stratifiedWith
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl, h1, StateT.run_bind]
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 1 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run 0 with
  | error e => exact absurd (hl.symm.trans (strata_loop_one src strata #[] Dl p ho)) (by simp)
  | ok y =>
    obtain rfl := Except.ok.inj (hl.symm.trans (strata_loop_one src strata #[] Dl p ho))
    exact ⟨_, rfl⟩

/-- **`drawUnits` returns what `stratifiedWith` draws**, and where it stopped. -/
theorem drawUnits_stratified_ok (K pop : ℕ) (strata : Array Flock.Draw.Stratum) (src : ByteArray) (D : Array ℕ)
    (p : ℕ) (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run 0 = .ok (some D, p)) :
    Draw.drawUnits (.stratified K strata) pop src = .ok (D, p) := by
  unfold Draw.drawUnits Flock.Draw.drawWith
  dsimp only
  rw [StateT.run_bind, h]
  rfl

/-- A big-endian word of `0` bytes is `0`. -/
theorem beNat_zeros (b : ByteArray) (off len : ℕ) (h : ∀ i < len, b.get! (off + i) = 0) :
    Flock.Draw.beNat b off len = 0 := by
  unfold Flock.Draw.beNat
  have key : ∀ l : List ℕ, (∀ i ∈ l, b.get! (off + i) = 0) →
      l.foldl (fun acc i => acc * 256 + (b.get! (off + i)).toNat) 0 = 0 := by
    intro l
    induction l with
    | nil => intro _; rfl
    | cons i is ih =>
      intro hl
      have h0 : (b.get! (off + i)).toNat = 0 := by rw [hl i List.mem_cons_self, UInt8.toNat_zero]
      show List.foldl _ (0 * 256 + (b.get! (off + i)).toNat) is = 0
      rw [h0]
      exact ih fun j hj => hl j (List.mem_cons_of_mem _ hj)
  exact key _ fun i hi => h i (List.mem_range.1 hi)

/-- **`uniform 2` on a stream that begins with a word of `0` bytes draws `0`.** -/
theorem uniform_two_zeros (src : ByteArray) (fuel : ℕ) (hw : Flock.Draw.width 2 ≤ src.size)
    (hz : ∀ i < Flock.Draw.width 2, src.get! (0 + i) = 0) :
    Flock.Draw.uniform 2 src (fuel + 1) 0 = some (0, Flock.Draw.width 2) := by
  have hx := beNat_zeros src 0 _ hz
  have hmod : 2 ^ (8 * Flock.Draw.width 2) % 2 = 0 := by
    obtain ⟨k, hk⟩ : ∃ k, 8 * Flock.Draw.width 2 = k + 1 :=
      ⟨8 * Flock.Draw.width 2 - 1, by have := DrawExec.width_pos 2; omega⟩
    rw [hk, Nat.pow_succ]
    exact Nat.mul_mod_left _ _
  have hs : 0 < 2 ^ (8 * Flock.Draw.width 2) - 2 ^ (8 * Flock.Draw.width 2) % 2 := by
    rw [hmod, Nat.sub_zero]
    exact Nat.two_pow_pos _
  unfold Flock.Draw.uniform
  dsimp only
  split
  · omega
  · split
    · simp [hx]
    · rename_i h2
      rw [hx] at h2
      exact absurd hs h2

/-- **`subset 1 2` draws on a stream that begins with a word of `0` bytes.** -/
theorem subset_one_two (src : ByteArray) (hw : Flock.Draw.width 2 ≤ src.size)
    (hz : ∀ i < Flock.Draw.width 2, src.get! (0 + i) = 0) : (Flock.Draw.subset 1 2 src).isSome := by
  rw [DrawExec.subset_eq, show List.range' 0 1 = [0] from rfl]
  simp only [DrawExec.loopSw, Nat.sub_zero, uniform_two_zeros src src.size hw hz, Option.map_some,
    Option.isSome_some]

/-- **The honest served draw ends**: on the `honestLc` coins all `0`, the served stratified draw of one of the units
`[0, 2)` draws its unit. -/
theorem honest_drawUnits : ∃ units read,
    Draw.drawUnits (.stratified 1 (Array.ofFn Flock.Guarantees.Served.honestSt)) 2
      (Law.streamOf Flock.Guarantees.Served.honestZeros) = .ok (units, read) := by
  have hsz : (Law.streamOf Flock.Guarantees.Served.honestZeros).size = Flock.Guarantees.Served.honestStratum.need :=
    DrawExec.size_streamOf _
  have hneed : Flock.Guarantees.Served.honestStratum.need = 2 * 1 * Flock.Draw.width 2 + 64 := rfl
  have hz : ∀ i < Flock.Draw.width 2, (Law.streamOf Flock.Guarantees.Served.honestZeros).get! (0 + i) = 0 := by
    intro i hi
    rw [DrawExec.get!_streamOf Flock.Guarantees.Served.honestZeros (0 + i)
      (by show 0 + i < Flock.Guarantees.Served.honestStratum.need; omega)]
    rfl
  obtain ⟨u, hu⟩ := Option.isSome_iff_exists.1
    (subset_one_two (Law.streamOf Flock.Guarantees.Served.honestZeros) (by omega) hz)
  have ho := onWith_take_ok (Law.streamOf Flock.Guarantees.Served.honestZeros)
    Flock.Guarantees.Served.honestStratum.need 0 (Flock.Draw.subset 1 2) u (by omega)
    (by rw [Nat.zero_add, ← hsz, ByteArray.extract_zero_size]; exact hu)
  obtain ⟨D, hD⟩ := stratifiedWith_one _ (Array.ofFn Flock.Guarantees.Served.honestSt) rfl u _ ho
  exact ⟨D, _, drawUnits_stratified_ok 1 2 _ _ D _ hD⟩

end FlockSoundness.Audit.Served
