import Proofs.Flock.Soundness.ExecStratified
import Proofs.Flock.Soundness.Assumptions
import Init.Internal.Order.While

/-!
# The live draw: `drawOS`, extend-and-rerun included, escapes no more often than the model's

`Flock.Draw.drawOS` is the verifier's draw from its operating-system randomness. It is `Flock.Draw.drawWith` with every
stream's bytes from `IO.getRandomBytes` (`drawOS_def`, by `rfl`): each stratum's `subset` runs on a stream of
`need` bytes, and `Flock.Draw.onWith` extends the stream by `need` more and reruns it whenever it runs out.

To Lean's logic `IO.getRandomBytes` is an opaque constant, so the theorems read `drawWith` on a model byte source,
`tapeGet`: stream `i` reads a tape in order, and a read past its end fails. `runOS f law pop` is the draw on the bytes
`f i` of each stream `i` (`some d`), or `none` if it would read past them. The bytes come from a source `t` over a finite
probability space `Ω`, uniform by assumption A3 (`Assumptions.UniformRandomBytes`, `uniform/io-getrandombytes`).

* `subset_append`: a `subset` draw that succeeds is the same on a longer stream, so rerunning on the extended stream
  never changes a draw once it succeeds.
* `onWith_tape`: `onWith` in the model, through `Lean.Loop.forIn_eq_of_monadTail`, by an invariant (`loop_inv`): the
  stream so far is the tape's first bytes, so a draw it returns is the sampler's on the whole tape.
* `stratifiedWith_tape`: a stratified run on unread streams returns `Flock.Draw.stratified`'s draw on the whole tapes.
* `drawOS_subset_escape_le`, `drawOS_stratified_escape_le`, `drawOS_work_escape_le`: the live draw misses `B` with
  probability at most the model's escape, `Law.subset`'s or `Law.stratified`'s, for every number of bytes `L` read,
  through `subset_exec_escape_le` and `stratified_exec_escape_le`.
* `execOS`: the live draw as a law, a run that would read past the source's bytes drawing every unit.
  `execOS_escape_le`, `execOS_work_escape_le` and `execOS_subset_escape_le` bound its escape, and `execOS_miss_le` its
  count curve.

The live verifier reads as many bytes as its reruns need, with no bound. On a run that reads at most `L i` bytes of each
stream it returns what `runOS` does on those bytes, and these runs exhaust every run as `L` grows, so its escape is the
limit of the ones bounded here, each at most the model's. A3 is what makes those bytes uniform. The limit itself is
measure theory the theorems don't state.

The Bernoulli law's live draw is `Proofs.Flock.Claims.Sampling.ExecBernoulli`'s `drawOS_bernoulli_escape_le`.
-/

namespace FlockSoundness.Audit.Law

open Game Finset
open scoped ENNReal

variable {n m : ℕ}

theorem get!_append (a b : ByteArray) (j : ℕ) (hj : j < a.size) : (a ++ b).get! j = a.get! j := by
  show (a ++ b).data[j]! = a.data[j]!
  have hj' : j < (a ++ b).data.size := by
    rw [ByteArray.data_append, Array.size_append]; have : j < a.data.size := hj; omega
  rw [getElem!_pos (a ++ b).data j hj', getElem!_pos a.data j hj]
  simp only [ByteArray.data_append]
  exact Array.getElem_append_left hj

theorem beNat_append (a b : ByteArray) (off len : ℕ) (h : off + len ≤ a.size) :
    Flock.Draw.beNat (a ++ b) off len = Flock.Draw.beNat a off len := by
  unfold Flock.Draw.beNat
  refine List.foldl_ext _ _ _ fun acc i hi => ?_
  rw [get!_append a b (off + i) (by have := List.mem_range.1 hi; omega)]

/-- **A draw of `uniform` that succeeds is the same on a longer stream**, with at least as much fuel. -/
theorem uniform_append (n : ℕ) (a b : ByteArray) : ∀ fuel pos r p, Flock.Draw.uniform n a fuel pos = some (r, p) →
    ∀ fuel', fuel ≤ fuel' → Flock.Draw.uniform n (a ++ b) fuel' pos = some (r, p) := by
  intro fuel
  induction fuel with
  | zero => intro pos r p h; simp [Flock.Draw.uniform] at h
  | succ f ih =>
    intro pos r p h fuel' hf
    obtain ⟨f', rfl⟩ : ∃ f', fuel' = f' + 1 := ⟨fuel' - 1, by omega⟩
    unfold Flock.Draw.uniform at h ⊢
    have hsz : (a ++ b).size = a.size + b.size := ByteArray.size_append
    dsimp only at h ⊢
    by_cases hw : pos + Flock.Draw.width n > a.size
    · simp only [hw, ↓reduceIte] at h; exact absurd h (by simp)
    · have hw' : ¬ pos + Flock.Draw.width n > (a ++ b).size := by omega
      simp only [hw, hw', ↓reduceIte] at h ⊢
      rw [beNat_append a b pos _ (by omega)]
      split_ifs at h ⊢ with hx
      · exact h
      · exact ih _ _ _ h f' (by omega)

theorem loopSw_append (n : ℕ) (a b : ByteArray) : ∀ (l : List ℕ) (sw : Std.HashMap ℕ ℕ) (pos : ℕ) sw',
    DrawExec.loopSw n a l sw pos = some sw' → DrawExec.loopSw n (a ++ b) l sw pos = some sw' := by
  intro l
  induction l with
  | nil => intro sw pos sw' h; exact h
  | cons i is ih =>
    intro sw pos sw' h
    simp only [DrawExec.loopSw] at h ⊢
    cases hu : Flock.Draw.uniform (n - i) a (a.size + 1) pos with
    | none => rw [hu] at h; exact absurd h (by simp)
    | some rp =>
      obtain ⟨r, p⟩ := rp
      rw [hu] at h
      rw [uniform_append (n - i) a b _ pos r p hu _ (by rw [ByteArray.size_append]; omega)]
      exact ih _ _ _ h

/-- **A `subset` draw that succeeds is the same on a longer stream**: extending the stream and rerunning changes nothing
once the draw has succeeded. -/
theorem subset_append (k n : ℕ) (a b : ByteArray) (D : Array ℕ) (h : Flock.Draw.subset k n a = some D) :
    Flock.Draw.subset k n (a ++ b) = some D := by
  rw [DrawExec.subset_eq] at h ⊢
  cases hl : DrawExec.loopSw n a (List.range' 0 k) ∅ 0 with
  | none => rw [hl] at h; exact absurd h (by simp)
  | some sw => rw [hl] at h; rw [loopSw_append n a b _ _ _ sw hl]; exact h

/-- A `repeat` loop in `StateT σ Option`, by an invariant `I` and a measure `μ` that each `yield` lowers: when it
returns, its value and state meet `Q`. -/
theorem loop_inv {σ β : Type} (f : Unit → β → StateT σ Option (ForInStep β)) (I Q : β → σ → Prop) (μ : β → σ → ℕ)
    (hf : ∀ b s, I b s → ∀ b' s', (f () b).run s = some (.done b', s') → Q b' s')
    (hy : ∀ b s, I b s → ∀ b' s', (f () b).run s = some (.yield b', s') → I b' s' ∧ μ b' s' < μ b s) :
    ∀ N b s, μ b s < N → I b s → ∀ r s', (Lean.Loop.forIn Lean.Loop.mk b f).run s = some (r, s') → Q r s' := by
  intro N
  induction N with
  | zero => intro b s h; omega
  | succ N ih =>
    intro b s hμ hI r s' h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    rw [StateT.run_bind] at h
    cases hb : (f () b).run s with
    | none => rw [hb] at h; exact absurd h (by simp)
    | some x =>
      obtain ⟨step, s₁⟩ := x
      rw [hb] at h
      cases step with
      | done b' =>
        change (pure b' : StateT σ Option β).run s₁ = some (r, s') at h
        rw [StateT.run_pure] at h
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
        exact hf b s hI _ _ hb
      | yield b' =>
        obtain ⟨hI', hμ'⟩ := hy b s hI _ _ hb
        exact ih b' s₁ (by omega) hI' r s' h


/-- The model's byte source: stream `i` reads the tape `T i` in order, from its position in the state; a read past the
tape's end fails. -/
def tapeGet (T : ℕ → ByteArray) (i need : ℕ) : StateT (ℕ → ℕ) Option ByteArray := fun pos =>
  if pos i + need ≤ (T i).size then some ((T i).extract (pos i) (pos i + need), Function.update pos i (pos i + need))
  else none

/-- A tape splits at any point up to its end. -/
theorem extract_split (a : ByteArray) (j : ℕ) (hj : j ≤ a.size) : a = a.extract 0 j ++ a.extract j a.size := by
  rw [← ByteArray.extract_eq_extract_append_extract j (Nat.zero_le _) hj, ByteArray.extract_zero_size]

/-- **`onWith` in the model**: a run from a stream not yet read that returns the sampler's `some v` has `v` from the
sampler on the whole tape (`run` keeping a success on a longer stream, as `subset` does), and moves no other stream. -/
theorem onWith_tape (T : ℕ → ByteArray) (i need : ℕ) (hneed : 0 < need) (run : ByteArray → Option (Array ℕ))
    (hrun : ∀ a b u, run a = some u → run (a ++ b) = some u) (pos pos' : ℕ → ℕ) (u : Option (Array ℕ))
    (hpos : pos i = 0) (h : (Flock.Draw.onWith (tapeGet T i) need run).run pos = some (u, pos')) :
    (∀ j, j ≠ i → pos' j = pos j) ∧ ∀ v, u = some v → run (T i) = some v := by
  unfold Flock.Draw.onWith at h
  rw [StateT.run_bind] at h
  cases hg : (tapeGet T i need).run pos with
  | none => rw [hg] at h; exact absurd h (by simp)
  | some x =>
    obtain ⟨src, pos₁⟩ := x
    rw [hg] at h
    have hg' := hg
    simp only [tapeGet, StateT.run] at hg'
    split at hg'
    · rename_i hle
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hg')
      rw [Option.bind_eq_bind, Option.bind_some] at h
      dsimp only at h
      rw [StateT.run_bind] at h
      cases hl : (forIn (m := StateT (ℕ → ℕ) Option) Lean.Loop.mk
          ((none : Option (Option (Array ℕ))), (T i).extract (pos i) (pos i + need)) _).run
          (Function.update pos i (pos i + need)) with
      | none => rw [hl] at h; exact absurd h (by simp)
      | some y =>
        obtain ⟨⟨r, src'⟩, pos₂⟩ := y
        rw [hl, Option.bind_eq_bind, Option.bind_some] at h
        dsimp only at h
        have key := loop_inv _
          (fun (b : Option (Option (Array ℕ)) × ByteArray) (s : ℕ → ℕ) =>
            b.1 = none ∧ b.2 = (T i).extract 0 (s i) ∧ (∀ j, j ≠ i → s j = pos j) ∧ s i ≤ (T i).size)
          (fun (b : Option (Option (Array ℕ)) × ByteArray) (s : ℕ → ℕ) =>
            (∀ j, j ≠ i → s j = pos j) ∧ ∀ v, b.1 = some (some v) → run (T i) = some v)
          (fun _ s => (T i).size + 1 - s i) ?_ ?_ _ _ _ (Nat.lt_succ_self _) ?_ _ _ hl
        · obtain ⟨hpos₂, hrun₂⟩ := key
          cases r with
          | none =>
            rw [StateT.run_pure] at h
            obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
            exact ⟨hpos₂, fun v hv => absurd hv (by simp)⟩
          | some r₀ =>
            rw [StateT.run_pure] at h
            obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj h)
            exact ⟨hpos₂, fun v hv => hrun₂ v (by rw [hv])⟩
        · intro b s hI b' s' hstep
          obtain ⟨_, hb2, hbs, hbi⟩ := hI
          split at hstep
          · rename_i units hu
            rw [StateT.run_pure] at hstep
            obtain ⟨h1, rfl⟩ := Prod.mk.inj (Option.some.inj hstep)
            injection h1 with h1
            subst h1
            refine ⟨hbs, fun v hv => ?_⟩
            obtain rfl : units = v := Option.some.inj (Option.some.inj hv)
            rw [extract_split (T i) (s i) hbi, ← hb2]
            exact hrun _ _ _ hu
          · rw [StateT.run_bind] at hstep
            cases hg2 : (tapeGet T i need).run s with
            | none => rw [hg2] at hstep; exact absurd hstep (by simp)
            | some z =>
              rw [hg2, Option.bind_eq_bind, Option.bind_some] at hstep
              rw [StateT.run_pure] at hstep
              obtain ⟨h1, _⟩ := Prod.mk.inj (Option.some.inj hstep)
              exact absurd h1 (by simp)
        · intro b s hI b' s' hstep
          obtain ⟨_, hb2, hbs, hbi⟩ := hI
          split at hstep
          · rw [StateT.run_pure] at hstep
            obtain ⟨h1, _⟩ := Prod.mk.inj (Option.some.inj hstep)
            exact absurd h1 (by simp)
          · rw [StateT.run_bind] at hstep
            cases hg2 : (tapeGet T i need).run s with
            | none => rw [hg2] at hstep; exact absurd hstep (by simp)
            | some z =>
              obtain ⟨bytes, s₁⟩ := z
              rw [hg2, Option.bind_eq_bind, Option.bind_some] at hstep
              rw [StateT.run_pure] at hstep
              obtain ⟨h1, rfl⟩ := Prod.mk.inj (Option.some.inj hstep)
              injection h1 with h1
              subst h1
              simp only [tapeGet, StateT.run] at hg2
              split at hg2
              · rename_i hle2
                obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Option.some.inj hg2)
                dsimp only
                refine ⟨⟨rfl, ?_, fun j hj => ?_, ?_⟩, ?_⟩
                · simp only [Function.update_self]
                  rw [hb2, ByteArray.extract_append_extract, Nat.min_eq_left (Nat.zero_le _),
                    Nat.max_eq_right (Nat.le_add_right _ _)]
                · rw [Function.update_of_ne hj]; exact hbs j hj
                · simp only [Function.update_self]; exact hle2
                · simp only [Function.update_self]; omega
              · exact absurd hg2 (by simp)
        · refine ⟨rfl, ?_, fun j hj => ?_, ?_⟩
          · simp only [Function.update_self, hpos, Nat.zero_add]
          · rw [Function.update_of_ne hj]
          · simp only [Function.update_self]; exact hle
    · exact absurd hg' (by simp)


/-- The stratified loop as a list recursion: stratum `i`'s sampler `g i`, its values appended under `u i`. -/
def catOpt (g : ℕ → Option (Array ℕ)) (u : ℕ → ℕ → ℕ) : List ℕ → Array ℕ → Option (Array ℕ)
  | [], out => some out
  | i :: is, out => (g i).bind fun l => catOpt g u is (out ++ l.map (u i))

theorem forIn_cat (g : ℕ → Option (Array ℕ)) (u : ℕ → ℕ → ℕ) (F : ℕ → Array ℕ → Option (ForInStep (Array ℕ)))
    (hF : ∀ i out, F i out = (g i).bind fun l => some (ForInStep.yield (out ++ l.map (u i)))) :
    ∀ (l : List ℕ) (out : Array ℕ), forIn l out F = catOpt g u l out := by
  intro l
  induction l with
  | nil => intro out; rfl
  | cons i is ih =>
    intro out
    rw [List.forIn_cons, hF]
    simp only [catOpt]
    cases g i with
    | none => rfl
    | some l => exact ih (out ++ Array.map (u i) l)

/-- `Flock.Draw.stratified` is `catOpt` over the strata, sorted. -/
theorem stratified_cat (strata : Array Flock.Draw.Stratum) (streams : Array ByteArray) :
    Flock.Draw.stratified strata streams =
      (catOpt (fun i => Flock.Draw.subset strata[i]!.k strata[i]!.size (streams.getD i ByteArray.empty))
        (fun i => strata[i]!.unit) (List.range' 0 strata.size) #[]).map (·.qsort (· < ·)) := by
  unfold Flock.Draw.stratified
  dsimp only
  rw [Std.Legacy.Range.forIn_eq_forIn_range']
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl]
  rw [forIn_cat (fun i => Flock.Draw.subset strata[i]!.k strata[i]!.size (streams.getD i ByteArray.empty))
    (fun i => strata[i]!.unit) (fun i __s => do
      let local_ ← Flock.Draw.subset strata[i]!.k strata[i]!.size (streams.getD i ByteArray.empty)
      pure (ForInStep.yield (__s ++ Array.map strata[i]!.unit local_))) (fun i out => rfl)]
  cases catOpt _ _ (List.range' 0 strata.size 1) #[] <;> rfl

theorem need_pos (s : Flock.Draw.Stratum) : 0 < s.need := by
  unfold Flock.Draw.Stratum.need; omega

/-- **`stratifiedWith` in the model**: a run from unread streams that returns a draw returns `Flock.Draw.stratified`'s on
the whole tapes. -/
theorem stratifiedWith_tape (strata : Array Flock.Draw.Stratum) (streams : Array ByteArray) (pos' : ℕ → ℕ)
    (D : Array ℕ) (h : (Flock.Draw.stratifiedWith (tapeGet fun i => streams.getD i ByteArray.empty) strata).run
      (fun _ => 0) = some (some D, pos')) :
    Flock.Draw.stratified strata streams = some D := by
  set T : ℕ → ByteArray := fun i => streams.getD i ByteArray.empty with hT
  unfold Flock.Draw.stratifiedWith at h
  dsimp only at h
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl] at h
  let g : ℕ → Option (Array ℕ) := fun i => Flock.Draw.subset strata[i]!.k strata[i]!.size (T i)
  have loop : ∀ (l : List ℕ), l.Nodup → ∀ (out : Array ℕ) (pos : ℕ → ℕ), (∀ j ∈ l, pos j = 0) →
      ∀ (r : Option (Option (Array ℕ))) (out' : Array ℕ) (pos₁ : ℕ → ℕ),
      (forIn (m := StateT (ℕ → ℕ) Option) l ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (tapeGet T i) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = some ((r, out'), pos₁) →
      (r = none → catOpt g (fun i => strata[i]!.unit) l out = some out') ∧ ∀ r₀, r = some r₀ → r₀ = none := by
    intro l
    induction l with
    | nil =>
      intro _ out pos _ r out' pos₁ hl
      simp only [List.forIn_nil, StateT.run_pure] at hl
      obtain ⟨h1, _⟩ := Prod.mk.inj (Option.some.inj hl)
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
      exact ⟨fun _ => rfl, fun r₀ h => absurd h (by simp)⟩
    | cons i is ih =>
      intro hnd out pos hpos r out' pos₁ hl
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind] at hl
      cases ho : (Flock.Draw.onWith (tapeGet T i) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)).run pos with
      | none => rw [ho] at hl; exact absurd hl (by simp)
      | some x =>
        obtain ⟨v, pos₂⟩ := x
        obtain ⟨hmv, hv⟩ := onWith_tape T i _ (need_pos _) _ (fun a b u h => subset_append _ _ a b u h) pos pos₂ v
          (hpos i (List.mem_cons_self)) ho
        rw [ho] at hl
        simp only [Option.bind_eq_bind, Option.bind_some] at hl
        cases v with
        | none =>
          simp only [StateT.run_pure] at hl
          obtain ⟨h1, _⟩ := Prod.mk.inj (Option.some.inj hl)
          obtain ⟨rfl, _⟩ := Prod.mk.inj h1
          exact ⟨fun h => absurd h (by simp), fun r₀ h => (Option.some.inj h).symm⟩
        | some li =>
          simp only [StateT.run_pure] at hl
          have hg : g i = some li := hv li rfl
          have key := ih (List.nodup_cons.1 hnd).2 _ pos₂ (fun j hj => ?_) r out' pos₁ hl
          · refine ⟨fun hr => ?_, key.2⟩
            show (g i).bind _ = _
            rw [hg, Option.bind_some]
            exact key.1 hr
          · rw [hmv j (fun e => (List.nodup_cons.1 hnd).1 (e ▸ hj))]
            exact hpos j (List.mem_cons_of_mem _ hj)
  rw [StateT.run_bind] at h
  cases hl : (forIn (m := StateT (ℕ → ℕ) Option) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run (fun _ => 0) with
  | none => rw [hl] at h; exact absurd h (by simp)
  | some y =>
    obtain ⟨⟨r, out'⟩, pos₁⟩ := y
    rw [hl, Option.bind_eq_bind, Option.bind_some] at h
    have key := loop _ List.nodup_range' #[] (fun _ => 0) (fun _ _ => rfl) r out' pos₁ hl
    cases r with
    | some r₀ =>
      rw [key.2 r₀ rfl] at h
      simp only [StateT.run_pure] at h
      exact absurd (Prod.mk.inj (Option.some.inj h)).1 (by simp)
    | none =>
      simp only [StateT.run_pure] at h
      have hD := (Prod.mk.inj (Option.some.inj h)).1
      rw [stratified_cat, key.1 rfl, Option.map_some]
      exact congrArg some (Option.some.inj hD)


theorem drawWith_stratified_run (get : ℕ → ℕ → StateT (ℕ → ℕ) Option ByteArray) (K pop : ℕ)
    (strata : Array Flock.Draw.Stratum) (s : ℕ → ℕ) :
    (Flock.Draw.drawWith get (.stratified K strata) pop).run s =
      ((Flock.Draw.stratifiedWith get strata).run s).map fun p =>
        (p.1.map fun units => ({ law := .stratified K strata, population := pop, units } : Flock.Draw.UnitDraw), p.2) := by
  unfold Flock.Draw.drawWith
  rw [StateT.run_bind]
  cases (Flock.Draw.stratifiedWith get strata).run s <;> rfl

theorem drawWith_work_run (get : ℕ → ℕ → StateT (ℕ → ℕ) Option ByteArray) (K pop : ℕ)
    (strata : Array Flock.Draw.WorkStratum) (s : ℕ → ℕ) :
    (Flock.Draw.drawWith get (.work K strata) pop).run s =
      ((Flock.Draw.stratifiedWith get (strata.map (·.toStratum))).run s).map fun p =>
        (p.1.map fun units => ({ law := .work K strata, population := pop, units } : Flock.Draw.UnitDraw), p.2) := by
  unfold Flock.Draw.drawWith
  rw [StateT.run_bind]
  cases (Flock.Draw.stratifiedWith get (strata.map (·.toStratum))).run s <;> rfl

theorem drawWith_subset_run (get : ℕ → ℕ → StateT (ℕ → ℕ) Option ByteArray) (k pop : ℕ) (s : ℕ → ℕ) :
    (Flock.Draw.drawWith get (.subset k) pop).run s =
      ((Flock.Draw.onWith (get 0) (2 * k * Flock.Draw.width pop + 64) (Flock.Draw.subset k pop)).run s).map fun p =>
        (p.1.map fun units => ({ law := .subset k, population := pop, units } : Flock.Draw.UnitDraw), p.2) := by
  unfold Flock.Draw.drawWith
  rw [StateT.run_bind]
  cases (Flock.Draw.onWith (get 0) (2 * k * Flock.Draw.width pop + 64) (Flock.Draw.subset k pop)).run s <;> rfl

/-- A source's streams as tapes, stream `i` the bytes `f i`, as `Flock.Draw.stratified` reads its streams. -/
def tapesOf {m : ℕ} {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256) : ℕ → ByteArray :=
  fun i => (Array.ofFn fun s => streamOf (f s)).getD i ByteArray.empty

/-- **The live draw on given bytes**: `Flock.Draw.drawOS`'s `drawWith`, each stream `i` reading the bytes `f i`; `none`
when it would read past them. -/
def runOS {m : ℕ} {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256) (law : Flock.Draw.Law) (pop : ℕ) :
    Option Flock.Draw.UnitDraw :=
  match (Flock.Draw.drawWith (tapeGet (tapesOf f)) law pop).run fun _ => 0 with
  | some (some d, _) => some d
  | _ => none

theorem runOS_strata {m : ℕ} {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256) (strata : Array Flock.Draw.Stratum)
    (law : Flock.Draw.Law) (pop : ℕ) (d : Flock.Draw.UnitDraw)
    (hlaw : ∀ (get : ℕ → ℕ → StateT (ℕ → ℕ) Option ByteArray) (s : ℕ → ℕ), ∃ g : Array ℕ → Flock.Draw.UnitDraw,
      (∀ D, (g D).units = D) ∧ (Flock.Draw.drawWith get law pop).run s =
        ((Flock.Draw.stratifiedWith get strata).run s).map fun p => (p.1.map g, p.2))
    (h : runOS f law pop = some d) :
    Flock.Draw.stratified strata (Array.ofFn fun s => streamOf (f s)) = some d.units := by
  obtain ⟨g, hg, hrun⟩ := hlaw (tapeGet (tapesOf f)) (fun _ => 0)
  unfold runOS at h
  rw [hrun] at h
  cases hs : (Flock.Draw.stratifiedWith (tapeGet (tapesOf f)) strata).run (fun _ => 0) with
  | none => rw [hs] at h; exact absurd h (by simp)
  | some p =>
    obtain ⟨u, pos'⟩ := p
    rw [hs] at h
    cases u with
    | none => exact absurd h (by simp)
    | some D =>
      have hd : g D = d := Option.some.inj h
      rw [← hd, hg]
      exact stratifiedWith_tape strata _ pos' D hs

/-- **The live stratified draw escapes no more often than the model's** (`Law.stratified`), on a uniform byte source
(A3). -/
theorem drawOS_stratified_escape_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (K pop : ℕ) {Ω : Type} [Fintype Ω] (L : Fin m → ℕ)
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    prCoin (fun ω => ∃ d, runOS (t ω) (.stratified K (Array.ofFn S)) pop = some d ∧ ∀ u ∈ B, u.val ∉ d.units) ≤
      (stratified σ k hk).escape B := by
  rw [ht fun f => ∃ d, runOS f (.stratified K (Array.ofFn S)) pop = some d ∧ ∀ u ∈ B, u.val ∉ d.units]
  refine le_trans (prCoin_mono fun f ⟨d, hd, hmiss⟩ => ⟨d.units, ?_, hmiss⟩) (stratified_exec_escape_le σ k hk S hS L B)
  exact runOS_strata f _ _ pop d (fun get s => ⟨_, fun _ => rfl, drawWith_stratified_run get K pop _ s⟩) hd

/-- **The live work draw escapes no more often than the model's**: the work law's strata, as the stratified law's. -/
theorem drawOS_work_escape_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (K pop : ℕ) (W : Array Flock.Draw.WorkStratum)
    (hW : W.map (·.toStratum) = Array.ofFn S) {Ω : Type} [Fintype Ω] (L : Fin m → ℕ)
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    prCoin (fun ω => ∃ d, runOS (t ω) (.work K W) pop = some d ∧ ∀ u ∈ B, u.val ∉ d.units) ≤
      (stratified σ k hk).escape B := by
  rw [ht fun f => ∃ d, runOS f (.work K W) pop = some d ∧ ∀ u ∈ B, u.val ∉ d.units]
  refine le_trans (prCoin_mono fun f ⟨d, hd, hmiss⟩ => ⟨d.units, ?_, hmiss⟩) (stratified_exec_escape_le σ k hk S hS L B)
  refine runOS_strata f _ _ pop d (fun get s => ⟨fun D => { law := .work K W, population := pop, units := D },
    fun _ => rfl, ?_⟩) hd
  rw [drawWith_work_run, hW]

/-- **The live subset draw escapes no more often than the uniform `k`-subset**, on a uniform byte source (A3). -/
theorem drawOS_subset_escape_le (n k : ℕ) (hk : k ≤ n) {Ω : Type} [Fintype Ω] (L : Fin 1 → ℕ)
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    prCoin (fun ω => ∃ d, runOS (t ω) (.subset k) n = some d ∧ ∀ u ∈ B, u.val ∉ d.units) ≤ (subset n k hk).escape B := by
  classical
  rw [ht fun f => ∃ d, runOS f (.subset k) n = some d ∧ ∀ u ∈ B, u.val ∉ d.units]
  calc prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => ∃ d, runOS f (.subset k) n = some d ∧ ∀ u ∈ B, u.val ∉ d.units)
      ≤ prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => ∀ s ∈ (univ : Finset (Fin 1)),
          (fun _ (g : Fin (L s) → Fin 256) => ∃ D, Flock.Draw.subset k n (streamOf g) = some D ∧ ∀ u ∈ B, u.val ∉ D)
            s (f s)) := by
        refine prCoin_mono fun f ⟨d, hd, hmiss⟩ s _ => ?_
        obtain rfl : s = 0 := Subsingleton.elim _ _
        refine ⟨d.units, ?_, hmiss⟩
        unfold runOS at hd
        rw [drawWith_subset_run] at hd
        cases ho : (Flock.Draw.onWith (tapeGet (tapesOf f) 0) (2 * k * Flock.Draw.width n + 64)
            (Flock.Draw.subset k n)).run (fun _ => 0) with
        | none => rw [ho] at hd; exact absurd hd (by simp)
        | some p =>
          obtain ⟨u, pos'⟩ := p
          rw [ho] at hd
          cases u with
          | none => exact absurd hd (by simp)
          | some D =>
            have hD : D = d.units := by rw [← Option.some.inj hd]
            have h1 := (onWith_tape (tapesOf f) 0 _ (by omega) _ (fun a b u h => subset_append _ _ a b u h) _ pos' _
              rfl ho).2 D rfl
            have h2 : tapesOf f 0 = streamOf (f 0) := by simp [tapesOf, Array.getD_eq_getD_getElem?]
            rw [h2] at h1
            rw [← hD]; exact h1
    _ = ∏ s : Fin 1, prCoin (fun g : Fin (L s) → Fin 256 =>
          ∃ D, Flock.Draw.subset k n (streamOf g) = some D ∧ ∀ u ∈ B, u.val ∉ D) :=
        prCoin_pi_forall (Ω := fun s => Fin (L s) → Fin 256) univ (fun _ (g : Fin (L _) → Fin 256) =>
          ∃ D, Flock.Draw.subset k n (streamOf g) = some D ∧ ∀ u ∈ B, u.val ∉ D)
    _ ≤ (subset n k hk).escape B := by
        rw [Fin.prod_univ_one]
        exact subset_exec_escape_le n k (L 0) hk B


/-- **The live draw as a law**: `Flock.Draw.drawOS`'s draw on a byte source `t` over `Ω`, reading each stream's first `L i`
bytes; a run that would read past them draws every unit. -/
def execOS {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} {L : Fin m → ℕ} (t : Ω → (i : Fin m) → Fin (L i) → Fin 256)
    (law : Flock.Draw.Law) (pop : ℕ) : Law n where
  Ω := Ω
  draw ω := match runOS (t ω) law pop with
    | some d => univ.filter fun u => u.val ∈ d.units
    | none => univ

theorem execOS_escape_le_of {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} {L : Fin m → ℕ}
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (law : Flock.Draw.Law) (pop : ℕ) (B : Finset (Fin n)) (e : ℝ≥0∞)
    (h0 : B = ∅ → 1 ≤ e)
    (h : prCoin (fun ω => ∃ d, runOS (t ω) law pop = some d ∧ ∀ u ∈ B, u.val ∉ d.units) ≤ e) :
    (execOS (n := n) t law pop).escape B ≤ e := by
  classical
  rcases B.eq_empty_or_nonempty with rfl | ⟨b, hb⟩
  · exact (DrawExec.prCoin_le_one _).trans (h0 rfl)
  · refine le_trans (prCoin_mono fun ω hd => ?_) h
    change Disjoint (match runOS (t ω) law pop with
      | some d => univ.filter fun u => u.val ∈ d.units
      | none => univ) B at hd
    split at hd
    · rename_i d hd'
      exact ⟨d, hd', fun u hu hud => Finset.disjoint_left.1 hd (Finset.mem_filter.2 ⟨Finset.mem_univ _, hud⟩) hu⟩
    · exact absurd hb (Finset.disjoint_left.1 hd (Finset.mem_univ b))

theorem stratified_escape_empty (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card) :
    (stratified σ k hk).escape ∅ = 1 := by
  rw [stratified_escape]
  refine Finset.prod_eq_one fun s _ => ?_
  simp only [Finset.filter_empty, Finset.card_empty, Nat.sub_zero]
  exact ENNReal.div_self (by exact_mod_cast (Nat.choose_pos (hk s)).ne') (ENNReal.natCast_ne_top _)

/-- **The live stratified draw, as a law, escapes no more often than `Law.stratified`**, on a uniform byte source (A3),
for every number of bytes `L` the model reads. -/
theorem execOS_escape_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t)
    (B : Finset (Fin n)) :
    (execOS (n := n) t (.stratified K (Array.ofFn S)) pop).escape B ≤ (stratified σ k hk).escape B :=
  execOS_escape_le_of t _ pop B _ (fun h => by rw [h, stratified_escape_empty])
    (drawOS_stratified_escape_le σ k hk S hS K pop L t ht B)

/-- **The live work draw, as a law, escapes no more often than `Law.stratified`**, on a uniform byte source (A3): the work
law's strata as the stratified law's. -/
theorem execOS_work_escape_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (K pop : ℕ) (W : Array Flock.Draw.WorkStratum)
    (hW : W.map (·.toStratum) = Array.ofFn S) {Ω : Type} [Fintype Ω] [Nonempty Ω] (L : Fin m → ℕ)
    (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    (execOS (n := n) t (.work K W) pop).escape B ≤ (stratified σ k hk).escape B :=
  execOS_escape_le_of t _ pop B _ (fun h => by rw [h, stratified_escape_empty])
    (drawOS_work_escape_le σ k hk S hS K pop W hW L t ht B)

/-- **The live subset draw, as a law, escapes no more often than the uniform `k`-subset**, on a uniform byte source (A3). -/
theorem execOS_subset_escape_le (n k : ℕ) (hk : k ≤ n) {Ω : Type} [Fintype Ω] [Nonempty Ω] (L : Fin 1 → ℕ)
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (B : Finset (Fin n)) :
    (execOS (n := n) t (.subset k) n).escape B ≤ (subset n k hk).escape B :=
  execOS_escape_le_of t _ n B _ (fun h => by rw [h, subset_escape]; simp [ENNReal.div_self,
    (Nat.choose_pos hk).ne'])
    (drawOS_subset_escape_le n k hk L t ht B)

/-- The live draw's count curve is at most `Law.stratified`'s, which `stratified_miss_eq_greedy` computes. -/
theorem execOS_miss_le (σ : Fin n → Fin m) (k : Fin m → ℕ) (hk : ∀ s, k s ≤ (stratum σ s).card)
    (S : Fin m → Flock.Draw.Stratum) (hS : ExecStrata σ k S) (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) (c : ℕ) :
    (execOS (n := n) t (.stratified K (Array.ofFn S)) pop).miss c ≤ (stratified σ k hk).miss c :=
  iSup₂_mono fun B _ => execOS_escape_le σ k hk S hS K pop L t ht B

/-- **`drawOS` is `drawWith` on `IO.getRandomBytes`**: the live draw is the one the theorems above are about. -/
theorem drawOS_def (law : Flock.Draw.Law) (pop : ℕ) :
    Flock.Draw.drawOS law pop = (do
      let some d ← Flock.Draw.drawWith (fun _ need => IO.getRandomBytes need.toUSize) law pop
        | throw (IO.userError "unreachable")
      return d) := rfl

end FlockSoundness.Audit.Law
