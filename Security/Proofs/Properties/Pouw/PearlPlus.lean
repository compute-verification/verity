import Properties.Pouw.PearlPlus
import Proofs.Properties.Evaluate
import Proofs.Properties.Pouw.Commit

/-!
# Proofs of Pearl+'s run verdict properties (`Properties.Pouw.PearlPlus`)

By unfolding the port's definitions step by step: each `Except` chain one bind at a time (`bind_ok`), each `for` loop by
induction over its list with the loop's state as the invariant's, and the replay's tasks by their pure model (a task's
`get` is its function's value). The unscale is the Definition's through `Properties.Evaluator`'s FP32 equalities.

Proved: `PlanMeans`, `AcceptMeans`, `RefusedBeforeTheCoin`, `LoadedGivesUp`, `TileAcceptMeans` and `YIsTheUnscale`.
`TileKeepsTheCap` and `WorkIsTheCredit` state the rule that a sampled over-cap row refuses its tile and that the work is
the credit over every unit; under tolerant acceptance, where a sampled over-cap row is credited nothing, both are to be
restated before they are proved. `OpeningBinds` is the trees' and not proved here.
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

theorem eq_of_not_bne {α : Type} [BEq α] [LawfulBEq α] {a b : α} (h : ¬(a != b) = true) : a = b := by
  simpa using h

theorem of_not_not {b : Bool} (h : ¬(!b) = true) : b = true := by
  cases b
  · exact absurd rfl h
  · rfl

theorem not_or_true {a b : Bool} (h : ¬(a || b) = true) : ¬a = true ∧ ¬b = true := by
  cases a <;> cases b <;> simp_all

/-! ## A loaded pass -/

theorem refuse_bind {α β : Type} (t w : String) (f : α → Except Refusal β) : (refuse t w >>= f) = refuse t w := rfl

theorem refuse_ne {α : Type} (t w : String) (a : α) : refuse t w ≠ .ok a := nofun

/-- Beta-reduce `h` without unfolding its `have`s: `do`'s join points stay shared, so a chain stays linear. -/
local macro "bz " h:ident : tactic => `(tactic| try dsimp (config := { zeta := false }) only at $h:ident)

/-- Peel one bind of the chain `h`: `a` is its value and `ha` the bound computation's equation. -/
local macro "peel " h:ident a:ident ha:ident : tactic =>
  `(tactic| (have h' := bind_ok $h; clear $h; rcases h' with ⟨$a:ident, $ha:ident, $h:ident⟩; bz $h))

/-- Close a branch of the chain `h` that refuses. -/
local macro "nope " h:ident : tactic => `(tactic| first
  | exact absurd $h (refuse_ne _ _ _)
  | (rw [refuse_bind] at $h:ident; exact absurd $h (refuse_ne _ _ _))
  | cases $h:ident)

theorem mapM_ok_mem {ε α β : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → ∀ b ∈ l', ∃ a ∈ l, f a = .ok b
  | [], l', h, b, hb => by cases h; cases hb
  | a :: l, l', h, b, hb => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    cases h
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, List.mem_cons_self, h0⟩
    · obtain ⟨a', ha', h'⟩ := mapM_ok_mem hbs b hb
      exact ⟨a', List.mem_cons_of_mem _ ha', h'⟩

theorem forIn_all {α β γ : Type} (Q : α → Prop)
    (f : α → Option (Option β) × γ → Id (ForInStep (Option (Option β) × γ)))
    (hf : ∀ a g, (∃ g', f a (none, g) = .yield (none, g') ∧ Q a) ∨ ∃ v g', f a (none, g) = .done (some (some v), g')) :
    ∀ (l : List α) (g : γ), (∀ v, (forIn l (none, g) f : Id _).1 ≠ some (some v)) → ∀ a ∈ l, Q a
  | [], _, _, a, ha => by cases ha
  | a :: l, g, h, b, hb => by
    rcases hf a g with ⟨g', he, hq⟩ | ⟨v, g', he⟩
    · rw [List.forIn_cons, he] at h
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hq
      · exact forIn_all Q f hf l g' h b hb
    · rw [List.forIn_cons, he] at h
      exact absurd rfl (h v)

theorem forIn_all' {α β γ : Type} (Q : α → Prop)
    (f : α → Option (Option β) × γ → Id (ForInStep (Option (Option β) × γ)))
    (hf : ∀ a g, (∃ g', f a (none, g) = .yield (none, g') ∧ Q a) ∨ ∃ v g', f a (none, g) = .done (some (some v), g'))
    (l : List α) (g : γ) (res : Option (Option β) × γ) (hres : (forIn l (none, g) f : Id _) = res)
    (hne : ∀ v, res.1 ≠ some (some v)) : ∀ a ∈ l, Q a :=
  forIn_all Q f hf l g (hres ▸ hne)

def RecordKept (splits : List (Nat × Nat)) (r : UnitRecord) : Prop :=
  r.matmul.k = kUnit ((splits.lookup r.matmul.weight).getD 0) ∧ r.kKept = (splits.lookup r.matmul.weight).getD 0 ∧
    r.voluntaryA.length ≤ voluntaryCap r.matmul.m ∧ r.voluntaryB.length ≤ voluntaryCap r.matmul.n

theorem recordKept_of {splits : List (Nat × Nat)} {r : UnitRecord}
    (hk : ¬((r.matmul.k, r.kKept) != (kUnit ((splits.lookup r.matmul.weight).getD 0),
      (splits.lookup r.matmul.weight).getD 0)) = true)
    (hc : ¬(decide (r.voluntaryA.length > voluntaryCap r.matmul.m) ||
      decide (r.voluntaryB.length > voluntaryCap r.matmul.n)) = true) : RecordKept splits r := by
  simp only [bne_iff_ne, ne_eq, Decidable.not_not, Prod.mk.injEq, Bool.or_eq_true, decide_eq_true_eq,
    not_or, Nat.not_lt] at hk hc
  exact ⟨hk.1, hk.2, hc.1, hc.2⟩

theorem checkRecords_kept {splits : List (Nat × Nat)} {records : List UnitRecord} (h : checkRecords splits records = none) :
    ∀ r ∈ records, RecordKept splits r := by
  unfold checkRecords at h
  simp only [Id.run_bind] at h
  generalize hres : (forIn records (none, [], []) _ : Id (Option (Option String) × List (Nat × List Nat) × List (Nat × List Nat))) = res at h
  refine forIn_all' (RecordKept splits) _ ?_ records _ res hres ?_
  · intro a g
    dsimp only
    split
    · exact .inr ⟨_, _, rfl⟩
    rename_i hk
    repeat' split
    all_goals first
      | exact .inr ⟨_, _, rfl⟩
      | exact .inl ⟨_, rfl, recordKept_of hk ‹_›⟩
  · intro v hv
    have hv' : res.run.fst = some (some v) := hv
    rw [hv'] at h
    cases h

theorem load_facts {coins : PouwCommit.CoinCommitment} {man : Lean.Json} {p : Pass} (h : load coins man = .ok p) :
    (∀ r ∈ p.records, r.reason = none) ∧
      ∃ splits, checkRecords splits p.records = none ∧ recordsRoot p.salt p.weightsRoot splits p.records = p.recordsRoot := by
  unfold load at h
  peel h u0 hman
  cases hd : passesDistinct [man] with
  | some w => simp [throwRefusal, hd] at h
  | none =>
  rw [throwRefusal.eq_def, hd] at h
  bz h
  peel h scheme hsch
  peel h mj hmj
  extract_lets j1 at h
  split at h
  case h_2 => nope h
  rename_i usA
  peel h us hus
  cases hus
  dsimp (config := { zeta := false }) only [j1] at h
  clear j1
  peel h matmuls hmm
  peel h wj hwj
  extract_lets ws j2 at h
  split at h
  case h_2 => nope h
  rename_i kv
  peel h weights hweights
  cases hweights
  dsimp (config := { zeta := false }) only [j2] at h
  clear j2
  peel h roots hroots
  split at h
  case h_2 => nope h
  rename_i opening0 _
  peel h claimedJ hcj
  extract_lets wtj jo ju at h
  split at h
  case isTrue => nope h
  dsimp (config := { zeta := false }) only [ju] at h
  clear ju
  split at h
  case h_2 => nope h
  rename_i opening hop
  peel h opening' hop'
  cases hop'
  dsimp (config := { zeta := false }) only [jo] at h
  clear jo
  extract_lets e1 e2 e3 e4 e5 at h
  split at h
  case h_2 => nope h
  rename_i claimed hcl
  peel h claimed' hc'
  cases hc'
  dsimp (config := { zeta := false }) only [e5] at h
  clear e5
  split at h
  case isTrue => nope h
  rename_i hep
  dsimp (config := { zeta := false }) only [e4] at h
  clear e4
  split at h
  case h_1 => nope h
  rename_i wr hwr
  dsimp (config := { zeta := false }) only [e3] at h
  clear e3
  split at h
  case isTrue => nope h
  rename_i hcov
  dsimp (config := { zeta := false }) only [e2] at h
  clear e2
  split at h
  case h_2 => nope h
  rename_i weightsRoot hwroot
  peel h weightsRoot' hw'
  cases hw'
  dsimp (config := { zeta := false }) only [e1] at h
  clear e1
  extract_lets f1 at h
  split at h
  case h_2 => nope h
  case h_3 => nope h
  rename_i salt hsalt
  peel h salt' hs'
  cases hs'
  dsimp (config := { zeta := false }) only [f1] at h
  clear f1
  extract_lets g1 at h
  split at h
  case isTrue => nope h
  rename_i hsaltj
  dsimp (config := { zeta := false }) only [g1] at h
  clear g1
  peel h records hrecords
  peel h splits hsplits
  extract_lets k1 at h
  split at h
  case h_1 => nope h
  rename_i cr hcr
  dsimp (config := { zeta := false }) only [k1] at h
  clear k1
  peel h pubj hpubj
  peel h published hpub
  extract_lets l1 at h
  split at h
  case isTrue => nope h
  rename_i hroot
  dsimp (config := { zeta := false }) only [l1] at h
  clear l1
  peel h rootsA hrootsA
  peel h c hc
  peel h rootj hrootj
  peel h root hroot'
  peel h countj hcountj
  split at h
  case h_2 => nope h
  rename_i count hcount
  peel h yRoots hyRoots
  peel h u1 hdomain
  extract_lets pp m1 at h
  split at h
  case isTrue => nope h
  rename_i htiles
  dsimp (config := { zeta := false }) only [m1] at h
  have hp : pp = p := Except.ok.inj h
  clear h m1
  subst hp
  refine ⟨fun r hr => ?_, splits, ?_, ?_⟩
  · obtain ⟨x, -, hfx⟩ := mapM_ok_mem hrecords r hr
    bz hfx
    extract_lets wt at hfx
    peel hfx real hreal
    peel hfx kk hkk
    split at hfx
    case h_2 => nope hfx
    split at hfx
    case h_2 => nope hfx
    peel hfx eaj heaj
    peel hfx ea hea
    peel hfx ebj hebj
    peel hfx eb heb
    peel hfx va hva
    peel hfx vb hvb
    extract_lets rr jr at hfx
    split at hfx
    case h_1 => nope hfx
    rename_i hrr
    dsimp (config := { zeta := false }) only [jr] at hfx
    have hp : rr = r := Except.ok.inj hfx
    subst hp
    cases hw : rr.reason with
    | none => rfl
    | some w => exact absurd hw (hrr w)
  · cases hc : checkRecords splits records with
    | none => rfl
    | some w => exact absurd hc (hcr w)
  · simpa using hroot

theorem loadedGivesUp : LoadedGivesUp := by
  intro coins man p h
  obtain ⟨hr, splits, hc, hroot⟩ := load_facts h
  exact ⟨fun r hm => ⟨hr r hm, (checkRecords_kept hc r hm).2.2.1, (checkRecords_kept hc r hm).2.2.2⟩, splits, hc, hroot⟩

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

/-! ## A tile -/

theorem tileAcceptMeans : TileAcceptMeans := by
  intro p wn tid o h
  unfold checkTile at h
  dsimp only at h
  split at h
  case h_2 => cases h
  rename_i pos rows cols ht
  split at h
  case isTrue => cases h
  rename_i hcov
  split at h
  case isTrue => cases h
  rename_i hA
  split at h
  case isTrue => cases h
  rename_i hB
  split at h
  case isTrue => cases h
  rename_i hT
  split at h
  case h_2 => cases h
  rename_i ops ch hw
  split at h
  case isTrue => cases h
  rename_i hleaf
  split at h
  case h_1 => cases h
  rename_i hco
  obtain ⟨hca, hcb⟩ := not_or_true hcov
  obtain ⟨hlp, hlo⟩ := not_or_true hT
  refine ⟨pos, rows, cols, ops, ch, ht, eq_of_not_bne hca, eq_of_not_bne hcb, eq_of_not_bne hlp, of_not_not hA,
    of_not_not hB, of_not_not hlo, hw, eq_of_not_bne hleaf, ?_, fun yRoot hy => ?_⟩
  · cases hc : Rule.checkOpened p.scheme p.records[pos]! rows cols ops with
    | none => rfl
    | some w => exact absurd hc (hco w)
  · rw [hy] at h
    dsimp only at h
    split at h
    case h_2 => cases h
    rename_i ys hys
    rw [pure_bind] at h
    split at h
    case isTrue => cases h
    rename_i hy'
    obtain ⟨hyp, hyo⟩ := not_or_true hy'
    exact ⟨ys, hys, eq_of_not_bne hyp, of_not_not hyo, h⟩

/-! ## The run's refusal before the coin -/

theorem refusedBeforeTheCoin : RefusedBeforeTheCoin := by
  intro i opener r h
  unfold check
  rw [h]

/-! ## The run's acceptance -/

/-- A loaded pass's units of one weight have one k: its split's `k_unit` (`checkRecords`). -/
theorem load_kconsistent {coins : CoinCommitment} {man : Lean.Json} {p : Pass} (h : load coins man = .ok p) :
    KConsistent p := by
  obtain ⟨-, splits, hc, -⟩ := load_facts h
  intro r hr r' hr' hw
  rw [(checkRecords_kept hc r hr).1, (checkRecords_kept hc r' hr').1, hw]

theorem passes_kconsistent {i : Input} {pl : Plan} (h : plan i = .ok pl) (j : Nat) : KConsistent pl.passes[j]! := by
  obtain ⟨-, hm, -⟩ := planMeans i pl h
  by_cases hj : j < pl.passes.length
  · rw [getElem!_pos pl.passes j hj]
    obtain ⟨man, -, hl⟩ := mapM_ok_mem hm _ (List.getElem_mem hj)
    exact load_kconsistent hl
  · rw [getElem!_neg pl.passes j hj]
    intro r hr
    cases hr

theorem verdictOf_key (pl : Plan) (opener : Nat → Nat → Except String TileOpening) (x : Nat × Nat) :
    ((verdictOf pl opener x).pass, (verdictOf pl opener x).tile) = x := by
  unfold verdictOf
  dsimp only
  split
  · rfl
  · split <;> rfl

theorem default_tile? (tid : Nat) : (default : Pass).tile? tid = none := rfl

/-- A tile whose verdict passed is a pass's, opened, and passed `checkTile` on its weight's noise. -/
theorem verdictOf_none {pl : Plan} {opener : Nat → Nat → Except String TileOpening} {pi tid : Nat}
    (h : (verdictOf pl opener (pi, tid)).reason = none) :
    ∃ p o, pl.passes[pi]? = some p ∧ opener pi tid = .ok o ∧ checkTile p (noiseOf p tid) tid o = none := by
  unfold verdictOf at h
  dsimp only at h
  split at h
  · cases h
  rename_i x ht
  split at h
  · cases h
  rename_i o ho
  by_cases hj : pi < pl.passes.length
  · refine ⟨pl.passes[pi], o, List.getElem?_eq_getElem hj, ho, ?_⟩
    rw [← getElem!_pos pl.passes pi hj]
    exact h
  · rw [getElem!_neg pl.passes pi hj, default_tile?] at ht
    cases ht

theorem acceptMeans : AcceptMeans := by
  intro i opener hacc
  cases hp : plan i with
  | error r => rw [refusedBeforeTheCoin i opener r hp] at hacc; cases hacc
  | ok pl =>
  unfold check at hacc ⊢
  rw [hp] at hacc ⊢
  dsimp only at hacc ⊢
  split at hacc
  case isFalse => cases hacc
  rename_i hb
  split
  case isFalse hb' => exact absurd hb hb'
  have hr := replay_eq pl opener (passes_kconsistent hp)
  have hnone : ∀ v ∈ replay pl opener, v.reason = none := by
    simp only [List.isEmpty_iff, List.filterMap_eq_nil_iff, Option.map_eq_none_iff] at hb
    exact hb
  refine ⟨pl, rfl, rfl, ?_, hnone, fun pi tid hm => verdictOf_none (hnone _ ?_)⟩
  · dsimp only
    rw [hr, List.map_map]
    conv => rhs; rw [← List.map_id pl.todo]
    exact List.map_congr_left fun x _ => verdictOf_key pl opener x
  · rw [hr]
    exact List.mem_map_of_mem hm

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
