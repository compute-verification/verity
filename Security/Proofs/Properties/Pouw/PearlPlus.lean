import Properties.Pouw.PearlPlus
import Proofs.Properties.Evaluate
import Proofs.Properties.Pouw.Commit

/-!
# Proofs of Pearl+'s run verdict properties (`Properties.Pouw.PearlPlus`)

By unfolding the port's definitions step by step: each `Except` chain one bind at a time (`bind_ok`), each `for` loop by
induction over its list with the loop's state as the invariant's, and the replay's tasks by their pure model (a task's
`get` is its function's value). The unscale is the Definition's through `Properties.Evaluator`'s FP32 equalities.
-/

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

/-! ## `Except` and `BEq` -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem eq_of_bne_false {α : Type} [BEq α] [LawfulBEq α] {a b : α} (h : (a != b) = false) : a = b := by
  simpa using h

theorem throw_bind {ε α β : Type} (e : ε) (f : α → Except ε β) : (throw e >>= f : Except ε β) = .error e := rfl

/-! ## The plan -/

/-- `plan` once its source is fixed: the draw's key, the draw and the plan. -/
def planTail (passes : List Pass) (draws : Nat) (source : ByteArray) : Except Refusal Plan := do
  let root := runCommitment passes
  let key ← match PouwCheck.PearlPlus.drawKey source root with
    | .ok k => pure k
    | .error e => throw { reason := s!"ValueError: {e}" }
  let drawn ← match PouwCheck.PearlPlus.draw passes key draws with
    | .ok d => pure d
    | .error e => throw { reason := s!"ValueError: {e}" }
  pure ⟨passes, root, source, todo passes drawn, drawn.length⟩

theorem plan_eq (i : Input) : plan i = (do
    if i.manifests.isEmpty then throw { reason := "the run holds no pass with calls" }
    throwRefusal (passesDistinct i.manifests) fun _ => do
    let passes ← i.manifests.mapM (load i.coins)
    let root := runCommitment passes
    let source ← match i.held with
      | some h =>
        if h.commitment != root || (i.source.isSome && i.source != some h.source) then
          throw { refused := some AFTER_COIN,
                  reason := s!"{AFTER_COIN.1}: {AFTER_COIN.2} (its coin was drawn on run {(hex h.commitment).take 16}, " ++
                    s!"these passes are run {(hex root).take 16})" }
        pure h.source
      | none => match i.source with
        | some s => pure s
        | none => throw { reason := "the verifier's coin: a source, or its record of the coin" }
    planTail passes i.draws source) := rfl

theorem planTail_ok {passes : List Pass} {draws : Nat} {source : ByteArray} {pl : Plan}
    (h : planTail passes draws source = .ok pl) :
    ∃ key drawn, PouwCheck.PearlPlus.drawKey source (runCommitment passes) = .ok key ∧
      PouwCheck.PearlPlus.draw passes key draws = .ok drawn ∧
      pl = ⟨passes, runCommitment passes, source, todo passes drawn, drawn.length⟩ := by
  unfold planTail at h
  cases hk : PouwCheck.PearlPlus.drawKey source (runCommitment passes) with
  | error e => simp only [hk, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h; cases h
  | ok key =>
  cases hd : PouwCheck.PearlPlus.draw passes key draws with
  | error e =>
    simp only [hk, hd, bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h; cases h
  | ok drawn =>
  simp only [hk, hd, bind, Except.bind, pure, Except.pure] at h
  cases h
  exact ⟨key, drawn, rfl, hd, rfl⟩

theorem planMeans : PlanMeans := by
  intro i pl h
  rw [plan_eq] at h
  dsimp only at h
  split at h
  · rw [throw_bind] at h; cases h
  cases hd : passesDistinct i.manifests with
  | some w => simp only [throwRefusal, hd] at h; cases h
  | none =>
  simp only [throwRefusal, hd] at h
  obtain ⟨passes, hp, h⟩ := bind_ok h
  obtain ⟨source, hheld, hnone, h⟩ : ∃ source, (∀ c, i.held = some c → c.commitment = runCommitment passes ∧
      source = c.source ∧ (i.source = none ∨ i.source = some c.source)) ∧ (i.held = none → i.source = some source) ∧
      planTail passes i.draws source = .ok pl := by
    cases hh : i.held with
    | some c =>
      simp only [hh] at h
      split at h
      · rw [throw_bind] at h; cases h
      rename_i hc
      simp only [Bool.or_eq_true, Bool.and_eq_true, not_or, not_and, Bool.not_eq_true, bne_eq_false_iff_eq] at hc
      refine ⟨c.source, fun c' hc' => ?_, fun h' => (by cases h'), h⟩
      cases hc'
      refine ⟨hc.1, rfl, ?_⟩
      cases hs : i.source with
      | none => exact .inl rfl
      | some s => exact .inr (by simpa [hs] using hc.2 (by simp [hs]))
    | none =>
      simp only [hh] at h
      cases hs : i.source with
      | none => simp only [hs] at h; rw [throw_bind] at h; cases h
      | some s =>
        simp only [hs] at h
        exact ⟨s, fun c hc => (by cases hc), fun _ => rfl, h⟩
  obtain ⟨key, drawn, hk, hdr, rfl⟩ := planTail_ok h
  exact ⟨rfl, hp, rfl, hheld, hnone, key, drawn, hk, hdr, rfl, rfl⟩

/-! ## The replay -/

theorem lookup_mem {α β : Type} [BEq α] [LawfulBEq α] {a : α} {b : β} :
    ∀ {l : List (α × β)}, l.lookup a = some b → (a, b) ∈ l
  | [], h => by cases h
  | (k, v) :: l, h => by
    rw [List.lookup_cons] at h
    by_cases hk : a = k
    · subst hk
      simp only [beq_self_eq_true] at h
      cases h
      exact List.mem_cons_self
    · simp only [beq_eq_false_iff_ne.2 hk] at h
      exact List.mem_cons_of_mem _ (lookup_mem h)

/-- A loop that, on an invariant of its first component, only yields, pushing one task of value `V a` per item. -/
theorem forIn_tasks {α σ τ : Type} (V : α → τ) (I : σ → Prop) (f : α → σ × Array (Task τ) → Id (ForInStep (σ × Array (Task τ))))
    (hf : ∀ a n ts, I n → ∃ n', I n' ∧ f a (n, ts) = .yield (n', ts.push (Task.pure (V a)))) :
    ∀ (l : List α) (n : σ) (ts : Array (Task τ)), I n →
      ((forIn l (n, ts) f : Id _).2.toList.map Task.get) = ts.toList.map Task.get ++ l.map V
  | [], n, ts, _ => by simp; rfl
  | a :: l, n, ts, hn => by
    obtain ⟨n', hn', he⟩ := hf a n ts hn
    rw [List.forIn_cons, he]
    change List.map Task.get (forIn (m := Id) l (n', ts.push ⟨V a⟩) f).snd.toList = _
    rw [forIn_tasks V I f hf l n' _ hn']
    simp

/-- The units of one weight have one k. -/
def KConsistent (p : Pass) : Prop :=
  ∀ r ∈ p.records, ∀ r' ∈ p.records, r.matmul.weight = r'.matmul.weight → r.matmul.k = r'.matmul.k

/-- One tile's verdict as `replay` computes it, on the fixed operand's noise of its weight. -/
def verdictOf (pl : Plan) (opener : Nat → Nat → Except String TileOpening) (x : Nat × Nat) : TileVerdict :=
  let p := pl.passes[x.1]!
  match p.tile? x.2 with
  | none => ⟨x.1, x.2, some "tile does not match the verifier's own enumeration"⟩
  | some _ => match opener x.1 x.2 with
    | .error e => ⟨x.1, x.2, some s!"the prover's files don't open it: {e}"⟩
    | .ok o => ⟨x.1, x.2, checkTile p (noiseOf p x.2) x.2 o⟩

/-- A search loop: each item yields on, or stops with a value `P` of it; a value found is some item's. -/
theorem forIn_found {α β γ : Type} (P : α → β → Prop) (f : α → Option (Option β) × γ → Id (ForInStep (Option (Option β) × γ)))
    (hf : ∀ a g, (∃ g', f a (none, g) = .yield (none, g')) ∨ ∃ v g', f a (none, g) = .done (some (some v), g') ∧ P a v) :
    ∀ (l : List α) (g : γ) (v : β), (forIn l (none, g) f : Id _).1 = some (some v) → ∃ a ∈ l, P a v
  | [], _, _, h => by cases h
  | a :: l, g, v, h => by
    rw [List.forIn_cons] at h
    rcases hf a g with ⟨g', he⟩ | ⟨v', g', he, hp⟩
    · rw [he] at h
      obtain ⟨a', ha', hp⟩ := forIn_found P f hf l g' v h
      exact ⟨a', List.mem_cons_of_mem _ ha', hp⟩
    · rw [he] at h
      cases h
      exact ⟨a, List.mem_cons_self, hp⟩

theorem tile?_pos {p : Pass} {tid pos : Nat} {rows cols : List Nat} (h : p.tile? tid = some (pos, rows, cols)) :
    pos < p.records.length := by
  unfold Pass.tile? at h
  dsimp only at h
  simp only [Id.run_bind] at h
  split at h
  · rename_i r heq
    obtain ⟨⟨_, pos'⟩, hm, hp⟩ := forIn_found (fun (x : UnitRecord × Nat) (v : Nat × List Nat × List Nat) => v.1 = x.2)
      _ (by
        intro a g
        dsimp only
        split
        · exact .inr ⟨_, _, rfl, rfl⟩
        · exact .inl ⟨_, rfl⟩) _ _ _ (heq.trans (congrArg some h))
    cases hp
    exact (List.getElem?_eq_some_iff.1 (List.mem_zipIdx_iff_getElem?.1 hm)).1
  · cases h

/-- What the replay's noise cache holds: each pass and weight's noise at a unit of that weight. -/
def NoiseHeld (pl : Plan) (noise : List ((Nat × Nat) × Task (Except String FixedNoise))) : Prop :=
  ∀ e ∈ noise, ∃ r ∈ (pl.passes[e.1.1]!).records, r.matmul.weight = e.1.2 ∧
    e.2.get = fixedNoise (pl.passes[e.1.1]!).salt (((pl.passes[e.1.1]!).weightRoots.lookup e.1.2).getD .empty) r.matmul.k

theorem replay_eq (pl : Plan) (opener : Nat → Nat → Except String TileOpening) (hk : ∀ i : Nat, KConsistent pl.passes[i]!) :
    replay pl opener = pl.todo.map (verdictOf pl opener) := by
  unfold replay
  refine (forIn_tasks (verdictOf pl opener) (NoiseHeld pl) _ ?_ pl.todo [] #[] (by intro e he; cases he)).trans
    (by simp)
  intro x noise ts hn
  obtain ⟨i, tid⟩ := x
  dsimp only
  cases ht : pl.passes[i]!.tile? tid with
  | none => exact ⟨noise, hn, by simp only [verdictOf, ht]; rfl⟩
  | some x =>
  obtain ⟨pos, rows, cols⟩ := x
  simp only
  have hmem : pl.passes[i]!.records[pos]! ∈ pl.passes[i]!.records := by
    have hp := tile?_pos ht
    rw [getElem!_pos pl.passes[i]!.records pos hp]; exact List.getElem_mem hp
  have hv (t : Task (Except String FixedNoise))
      (hget : t.get = fixedNoise pl.passes[i]!.salt
        ((List.lookup pl.passes[i]!.records[pos]!.matmul.weight pl.passes[i]!.weightRoots).getD ByteArray.empty)
        pl.passes[i]!.records[pos]!.matmul.k) :
      Task.map (fun wn => match opener i tid with
        | Except.error e => { pass := i, tile := tid, reason := some (toString "the prover's files don't open it: " ++ toString e) }
        | Except.ok o => ({ pass := i, tile := tid, reason := checkTile pl.passes[i]! wn tid o } : TileVerdict)) t =
        ⟨verdictOf pl opener (i, tid)⟩ := by
    show Task.pure _ = _
    congr 1
    simp only [verdictOf, ht, noiseOf, hget]
  cases hl : List.lookup (i, pl.passes[i]!.records[pos]!.matmul.weight) noise with
  | some t =>
    obtain ⟨r, hr, hw, hget⟩ := hn _ (lookup_mem hl)
    dsimp only at hr hw hget
    have hkk : r.matmul.k = pl.passes[i]!.records[pos]!.matmul.k := hk i r hr _ hmem hw
    exact ⟨noise, hn, congrArg (fun x => (ForInStep.yield (noise, ts.push x) : ForInStep _))
      (hv t (by rw [hget, hkk, ← hw]))⟩
  | none =>
    refine ⟨((i, pl.passes[i]!.records[pos]!.matmul.weight), Task.spawn fun _ => fixedNoise pl.passes[i]!.salt
        ((List.lookup pl.passes[i]!.records[pos]!.matmul.weight pl.passes[i]!.weightRoots).getD ByteArray.empty)
        pl.passes[i]!.records[pos]!.matmul.k) :: noise, ?_,
      congrArg (fun x => (ForInStep.yield (_, ts.push x) : ForInStep _)) (hv _ rfl)⟩
    intro e he
    rcases List.mem_cons.1 he with rfl | he
    · exact ⟨_, hmem, rfl, rfl⟩
    · exact hn e he

/-! ## The run's refusal before the coin -/

theorem refusedBeforeTheCoin : RefusedBeforeTheCoin := by
  intro i opener r h
  unfold check
  rw [h]

/-! ## The served y -/

theorem unscale_eq (u alphaA alphaB : Nat) : Rule.unscale u alphaA alphaB = Pouw.PearlC.unscale u alphaA alphaB := by
  unfold Rule.unscale Pouw.PearlC.unscale
  rw [Properties.Evaluator.Fp32.f32MulV2, Properties.Evaluator.Fp32.f32DivV2,
    Properties.Evaluator.Fp32.f32ToBf16RnBits, Properties.Evaluator.Fp32.bf16NanF2fp]
  rfl

theorem yIsTheUnscale : YIsTheUnscale := by
  refine ⟨unscale_eq, fun u alphaA alphaB y h => ?_⟩
  unfold Rule.yOf at h
  dsimp only at h
  split at h
  · cases h
  · cases h
    exact unscale_eq u alphaA alphaB

end Properties.Pouw.PearlPlus
