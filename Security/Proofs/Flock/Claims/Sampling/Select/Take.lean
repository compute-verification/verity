import Proofs.Flock.Claims.Sampling.Select.Seq
import SampledProofs.Draw
import Init.Internal.Order.While

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open FlockSoundness FlockSoundness.Audit

/-- A `repeat` loop in `StateT σ (Except ε)`, by an invariant `I` and a measure `μ` that each `yield` lowers: when it
returns, its value and state meet `Q`. -/
theorem loop_inv_ex {σ ε β : Type} (f : Unit → β → StateT σ (Except ε) (ForInStep β)) (I Q : β → σ → Prop)
    (μ : β → σ → ℕ)
    (hf : ∀ b s, I b s → ∀ b' s', (f () b).run s = .ok (.done b', s') → Q b' s')
    (hy : ∀ b s, I b s → ∀ b' s', (f () b).run s = .ok (.yield b', s') → I b' s' ∧ μ b' s' < μ b s) :
    ∀ N b s, μ b s < N → I b s → ∀ r s', (Lean.Loop.forIn Lean.Loop.mk b f).run s = .ok (r, s') → Q r s' := by
  intro N
  induction N with
  | zero => intro b s h; omega
  | succ N ih =>
    intro b s hμ hI r s' h
    rw [Lean.Loop.forIn_eq_of_monadTail] at h
    rw [StateT.run_bind] at h
    cases hb : (f () b).run s with
    | error e => rw [hb] at h; exact absurd h (by simp [bind, Except.bind])
    | ok x =>
      obtain ⟨step, s₁⟩ := x
      rw [hb] at h
      cases step with
      | done b' =>
        change (pure b' : StateT σ (Except ε) β).run s₁ = .ok (r, s') at h
        rw [StateT.run_pure] at h
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
        exact hf b s hI _ _ hb
      | yield b' =>
        obtain ⟨hI', hμ'⟩ := hy b s hI _ _ hb
        exact ih b' s₁ (by omega) hI' r s' h

theorem take_run (src : ByteArray) (need p : ℕ) :
    (Draw.take src need).run p =
      if src.size < p + need then Except.error "the randomness source ran out"
      else Except.ok (src.extract p (p + need), p + need) := by
  unfold Draw.take
  by_cases h : src.size < p + need <;> simp [h, StateT.run, bind, StateT.bind, get, getThe, MonadStateOf.get,
    StateT.get, set, StateT.set, pure, StateT.pure, throw, throwThe, MonadExceptOf.throw, StateT.lift, Except.bind,
    Except.pure]

theorem bind_ok {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

theorem bind_error {ε α β : Type} (e : ε) (f : α → Except ε β) : (Except.error e >>= f) = Except.error e := rfl

theorem bind_eq_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} :
    (x >>= f) = .ok b ↔ ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x <;> simp [bind, Except.bind]

/-- **`onWith` on `take`**: a run from `p` that ends returns the first of the runs on `j·N` bytes from `p` that succeeds,
at `p + j·N`. -/
theorem onWith_take (src : ByteArray) (N : ℕ) (hN : 0 < N) (run : ByteArray → Option (Array ℕ)) (p p' : ℕ)
    (u : Option (Array ℕ))
    (h : (Flock.Draw.onWith (fun need => Draw.take src need) N run).run p = .ok (u, p')) :
    ∃ j D, u = some D ∧ p' = p + j * N ∧ StopsAt N run src p j D := by
  unfold Flock.Draw.onWith at h
  rw [StateT.run_bind, take_run] at h
  split at h
  · exact absurd h (by rw [bind_error]; simp)
  · rename_i hle
    rw [bind_ok] at h
    dsimp only at h
    rw [StateT.run_bind] at h
    cases hl : (forIn (m := StateT ℕ (Except String)) Lean.Loop.mk
        ((none : Option (Option (Array ℕ))), src.extract p (p + N)) _).run (p + N) with
    | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨⟨r, src'⟩, p₂⟩ := y
      rw [hl, bind_ok] at h
      dsimp only at h
      have key := loop_inv_ex _
        (fun (b : Option (Option (Array ℕ)) × ByteArray) (s : ℕ) =>
          b.1 = none ∧ ∃ j, 1 ≤ j ∧ s = p + j * N ∧ s ≤ src.size ∧ b.2 = src.extract p s ∧
            ∀ i, 1 ≤ i → i < j → run (src.extract p (p + i * N)) = none)
        (fun (b : Option (Option (Array ℕ)) × ByteArray) (s : ℕ) =>
          ∃ j D, b.1 = some (some D) ∧ s = p + j * N ∧ StopsAt N run src p j D)
        (fun _ s => src.size + 1 - s) ?_ ?_ _ _ _ (Nat.lt_succ_self _) ?_ _ _ hl
      · obtain ⟨j, D, hr, hp₂, hstop⟩ := key
        dsimp only at hr
        subst hr
        rw [StateT.run_pure] at h
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
        exact ⟨j, D, rfl, hp₂, hstop⟩
      · intro b s hI b' s' hstep
        obtain ⟨_, j, hj, hs, hsz, hb2, hmin⟩ := hI
        split at hstep
        · rename_i units hu
          rw [StateT.run_pure] at hstep
          obtain ⟨h1, rfl⟩ := Prod.mk.inj (Except.ok.inj hstep)
          injection h1 with h1
          subst h1
          refine ⟨j, units, rfl, hs, hj, hs ▸ hsz, ?_, hmin⟩
          rw [← hs, ← hb2]; exact hu
        · rw [StateT.run_bind, take_run] at hstep
          split at hstep
          · exact absurd hstep (by rw [bind_error]; simp)
          · rw [bind_ok, StateT.run_pure] at hstep
            obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hstep)
            exact absurd h1 (by simp)
      · intro b s hI b' s' hstep
        obtain ⟨_, j, hj, hs, hsz, hb2, hmin⟩ := hI
        split at hstep
        · rw [StateT.run_pure] at hstep
          obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hstep)
          exact absurd h1 (by simp)
        · rename_i hnone
          rw [StateT.run_bind, take_run] at hstep
          split at hstep
          · exact absurd hstep (by rw [bind_error]; simp)
          · rename_i hle2
            rw [bind_ok, StateT.run_pure] at hstep
            obtain ⟨h1, rfl⟩ := Prod.mk.inj (Except.ok.inj hstep)
            injection h1 with h1
            subst h1
            refine ⟨⟨rfl, j + 1, by omega, by rw [hs]; ring, by omega, ?_, fun i hi hij => ?_⟩, by omega⟩
            · dsimp only
              rw [hb2, ByteArray.extract_append_extract, Nat.min_eq_left (by omega),
                Nat.max_eq_right (by omega)]
            · rcases Nat.lt_succ_iff_lt_or_eq.1 hij with hlt | rfl
              · exact hmin i hi hlt
              · rw [← hs, ← hb2]
                cases hrun : run b.2 with
                | none => rfl
                | some x => exact absurd hrun (hnone x)
      · refine ⟨rfl, 1, le_refl _, by ring, by omega, rfl, fun i hi hlt => by omega⟩

/-- A stratum's stage: its `subset` run on `need` bytes at a time, missing as `miss` says. -/
def stage (s : Flock.Draw.Stratum) (miss : Array ℕ → Prop) : Stage := ⟨s.need, Flock.Draw.subset s.k s.size, miss⟩

theorem seqMiss_mono (strata : ℕ → Flock.Draw.Stratum) (m m' : ℕ → Array ℕ → Prop) :
    ∀ (l : List ℕ), (∀ i ∈ l, ∀ D, m i D → m' i D) → ∀ src p,
      SeqMiss (l.map fun i => stage (strata i) (m i)) src p → SeqMiss (l.map fun i => stage (strata i) (m' i)) src p
  | [], _, _, _, _ => trivial
  | i :: is, hm, src, _, ⟨j, D, hst, hD, hrest⟩ =>
    ⟨j, D, hst, hm i List.mem_cons_self D hD,
      seqMiss_mono strata m m' is (fun i' hi' => hm i' (List.mem_cons_of_mem _ hi')) src _ hrest⟩

/-- **`stratifiedWith` on `take`**: a run from `p` that returns a draw `D` runs each stratum's `subset` in turn, each
stopping where `onWith` stops it, every unit it draws in `D`. -/
theorem stratifiedWith_take (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p p' : ℕ) (D : Array ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run p = .ok (some D, p')) :
    SeqMiss ((List.range' 0 strata.size).map fun i =>
      stage strata[i]! fun Dl => ∀ x ∈ Dl, strata[i]!.unit x ∈ D) src p := by
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
      r = none ∧ (∀ x ∈ out, x ∈ out') ∧
        SeqMiss (l.map fun i => stage strata[i]! fun Dl => ∀ x ∈ Dl, strata[i]!.unit x ∈ out') src pos := by
    intro l
    induction l with
    | nil =>
      intro out pos r out' pos₁ hl
      simp only [List.forIn_nil, StateT.run_pure] at hl
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hl)
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
      exact ⟨rfl, fun _ hx => hx, trivial⟩
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
        obtain ⟨hr, hsub, hseq⟩ := ih _ _ r out' pos₁ hl
        refine ⟨hr, fun x hx => hsub x (Array.mem_append_left _ hx), j, Dl, hst, fun x hx => ?_, hseq⟩
        exact hsub _ (Array.mem_append_right _ (Array.mem_map_of_mem hx))
  rw [StateT.run_bind] at h
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run p with
  | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨⟨r, out'⟩, pos₁⟩ := y
    rw [hl, bind_ok] at h
    dsimp only at h
    obtain ⟨rfl, _, hseq⟩ := loop _ _ _ _ _ _ hl
    rw [StateT.run_pure] at h
    obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
    have hD : D = out'.qsort (fun x1 x2 => decide (x1 < x2)) := (Option.some.inj h1).symm
    refine seqMiss_mono (fun i => strata[i]!) _ _ _ (fun i _ Dl hDl x hx => ?_) _ _ hseq
    rw [hD, DrawExec.mem_qsort]; exact hDl x hx

/-- The stages a law's draw over `[0, n)` runs (`Flock.Draw.drawWith`), stage `i` missing as `miss i` says: one for
`subset` and `bernoulli`, one per stratum for `stratified` and `work`. -/
def lawStages (law : Flock.Draw.Law) (n : ℕ) (miss : ℕ → Array ℕ → Prop) : List Stage :=
  match law with
  | .subset k => [⟨2 * k * Flock.Draw.width n + 64, Flock.Draw.subset k n, miss 0⟩]
  | .bernoulli a b => [⟨2 * n * Flock.Draw.width b + 64, Flock.Draw.bernoulli a b n, miss 0⟩]
  | .stratified _ strata => (List.range' 0 strata.size).map fun i => stage strata[i]! (miss i)
  | .work _ strata =>
    (List.range' 0 (strata.map (·.toStratum)).size).map fun i => stage (strata.map (·.toStratum))[i]! (miss i)

/-- The unit stage `i`'s local index `x` is. -/
def unitOf (law : Flock.Draw.Law) (i x : ℕ) : ℕ :=
  match law with
  | .stratified _ strata => strata[i]!.unit x
  | .work _ strata => (strata.map Flock.Draw.WorkStratum.toStratum)[i]!.unit x
  | _ => x

/-- **`drawUnits` as stages**: a draw from `src` that ends runs `law`'s stages in turn from its first byte, every unit
each stage draws among the units it returns. -/
theorem drawUnits_seq (law : Flock.Draw.Law) (n : ℕ) (src : ByteArray) (units : Array ℕ) (read : ℕ)
    (h : Draw.drawUnits law n src = .ok (units, read)) :
    SeqMiss (lawStages law n fun i Dl => ∀ x ∈ Dl, unitOf law i x ∈ units) src 0 := by
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
        exact ⟨j, D, hst, fun x hx => hx, trivial⟩
    | bernoulli a b =>
      unfold Flock.Draw.drawWith at hw
      dsimp only at hw
      rw [StateT.run_bind] at hw
      cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) (2 * n * Flock.Draw.width b + 64)
          (Flock.Draw.bernoulli a b n)).run 0 with
      | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
      | ok y =>
        obtain ⟨v, p⟩ := y
        obtain ⟨j, D, rfl, -, hst⟩ := onWith_take src _ (by omega) _ 0 _ _ ho
        rw [ho, bind_ok, StateT.run_pure] at hw
        obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
        obtain rfl := (Option.some.inj h1).symm
        exact ⟨j, D, hst, fun x hx => hx, trivial⟩
    | stratified k strata =>
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
          exact stratifiedWith_take src strata 0 p D ho
    | work k strata =>
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
          exact stratifiedWith_take src _ 0 p D ho

end Properties.SampledProofs.SecurityProofs
