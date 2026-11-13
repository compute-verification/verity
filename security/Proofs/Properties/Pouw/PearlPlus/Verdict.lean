module

public import Properties.Pouw.PearlPlus
public import Properties.Pouw.PearlPlus.Witnesses
public import Proofs.Properties.Core.Evaluate
public import Proofs.Properties.Pouw.Commit

/-!
# What Pearl+'s run verdict establishes (`Properties.Pouw.PearlPlus`)

`PlanMeans`, `AcceptMeans`, `RefusedBeforeTheCoin`, `LoadedGivesUp`, `TileAcceptMeans`, `TileKeepsTheCap`,
`YIsTheUnscale` and `WorkIsTheCredit`, by unfolding the port's definitions step by step: each `Except` chain one bind
at a time (`bind_ok`),
each `for` loop by induction over its list with the loop's state as the invariant's, and the replay's tasks by their
pure model (a task's `get` is its function's value). The unscale is the Definition's through `Properties.Evaluator`'s FP32 equalities. A tile
`checkOpened` passes keeps the inert-work bound (`TileKeepsTheCap`); one over it is `checkTile`'s `overCap`, which the
run accepts only when it credits such a tile nothing (`AcceptMeans`). The work an accepted run publishes is the credit
on the verdicts it answers with, and a rejected run publishes none (`WorkIsTheCredit`, `check`'s two branches).
-/

@[expose] public section

namespace Properties.Pouw.PearlPlus

open PouwCommit PouwCheck.PearlPlus PouwCheck.PearlPlus.Words

/-! ## `Except` and `BEq` -/

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : x >>= f = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

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

theorem checkRecords_kept {splits : List (Nat × Nat)} {records : List UnitRecord}
    (h : checkRecords splits records = none) :
    ∀ r ∈ records, RecordKept splits r := by
  unfold checkRecords at h
  simp only [Id.run_bind] at h
  generalize hres : (forIn records (none, [], []) _ :
    Id (Option (Option String) × List (Nat × List Nat) × List (Nat × List Nat))) = res at h
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

theorem load_facts {coins : EpochCoins} {man : Lean.Json} {p : Pass} (h : load coins man = .ok p) :
    (∀ r ∈ p.records, r.reason = none) ∧
      ∃ splits, checkRecords splits p.records = none ∧
        recordsRoot p.salt p.weightsRoot splits p.records = p.recordsRoot := by
  unfold load at h
  obtain ⟨u0, hman, h⟩ := bind_ok h
  cases hd : passesDistinct [man] with
  | some w => simp [throwRefusal, hd] at h
  | none =>
  rw [throwRefusal.eq_def, hd] at h
  obtain ⟨scheme, hsch, h⟩ := bind_ok h
  obtain ⟨mj, hmj, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder at h
  split at h
  case h_2 => cases h
  rename_i usA
  obtain ⟨us, hus, h⟩ := bind_ok h
  cases hus
  dsimp (config := { zeta := false, zetaDelta := true }) only at h
  obtain ⟨matmuls, hmm, h⟩ := bind_ok h
  obtain ⟨wj, hwj, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder at h
  split at h
  case h_2 => cases h
  rename_i kv
  obtain ⟨weights, hweights, h⟩ := bind_ok h
  cases hweights
  dsimp (config := { zeta := false, zetaDelta := true }) only at h
  obtain ⟨roots, hroots, h⟩ := bind_ok h
  split at h
  case h_2 => cases h
  rename_i openingJ hoj
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder cj lv jo at h
  generalize hb : lv = b at h
  obtain ⟨opening, h⟩ : ∃ opening, jo opening = .ok p := by
    split at h
    all_goals (split at h; rotate_left; cases h; obtain ⟨o', ho', h⟩ := bind_ok h; cases ho'; exact ⟨_, h⟩)
  dsimp (config := { zeta := false }) only [jo] at h
  simp (config := { zeta := false }) only [hb] at h
  generalize hc : cj = co at h
  cases co with
  | none => cases h
  | some claimedJ =>
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder jc at h
  obtain ⟨claimed, h⟩ : ∃ claimed, jc claimed = .ok p := by
    split at h
    all_goals (split at h; rotate_left; cases h; obtain ⟨o', ho', h⟩ := bind_ok h; cases ho'; exact ⟨_, h⟩)
  dsimp (config := { zeta := false }) only [jc] at h
  extract_lets -underBinder same jsame at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [jsame] at h
  extract_lets -underBinder jw at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [jw] at h
  extract_lets -underBinder ws jws at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [jws] at h
  extract_lets -underBinder jwr at h
  obtain ⟨weightsRoot, h⟩ : ∃ r, jwr r = .ok p := by
    split at h
    · obtain ⟨o', ho', h⟩ := bind_ok h; cases ho'; exact ⟨_, h⟩
    · cases h
  dsimp (config := { zeta := false }) only [jwr] at h
  extract_lets -underBinder jsalt at h
  obtain ⟨salt, h⟩ : ∃ s, jsalt s = .ok p := by
    split at h
    all_goals first
      | cases h
      | (split at h <;> first
          | cases h
          | (obtain ⟨o', ho', h⟩ := bind_ok h; cases ho'; exact ⟨_, h⟩))
  dsimp (config := { zeta := false }) only [jsalt] at h
  extract_lets -underBinder jsl at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [jsl] at h
  extract_lets -underBinder wj at h
  obtain ⟨records, hrecords, h⟩ := bind_ok h
  obtain ⟨splits, hsplits, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder jcr at h
  split at h
  · cases h
  rename_i hcr
  dsimp (config := { zeta := false }) only [jcr] at h
  obtain ⟨pubj, hpubj, h⟩ := bind_ok h
  obtain ⟨published, hpub, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder jrr at h
  split at h
  · cases h
  rename_i hroot
  dsimp (config := { zeta := false }) only [jrr] at h
  obtain ⟨rootsA, hrootsA, h⟩ := bind_ok h
  obtain ⟨c, hc, h⟩ := bind_ok h
  obtain ⟨rootj, hrootj, h⟩ := bind_ok h
  obtain ⟨root, hroot', h⟩ := bind_ok h
  obtain ⟨countj, hcountj, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  split at h
  case h_2 => cases h
  rename_i count hcount
  obtain ⟨yRoots, hyRoots, h⟩ := bind_ok h
  obtain ⟨u1, hdomain, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder pp at h
  split at h
  · cases h
  have hp : pp = p := Except.ok.inj h
  subst hp
  refine ⟨fun r hr => ?_, splits, ?_, ?_⟩
  · obtain ⟨x, -, hfx⟩ := mapM_ok_mem hrecords r hr
    try dsimp (config := { zeta := false }) only at hfx
    extract_lets -underBinder wt at hfx
    obtain ⟨real, hreal, hfx⟩ := bind_ok hfx
    obtain ⟨kk, hkk, hfx⟩ := bind_ok hfx
    split at hfx
    case h_2 => cases hfx
    split at hfx
    case h_2 => cases hfx
    obtain ⟨eaj, heaj, hfx⟩ := bind_ok hfx
    obtain ⟨ea, hea, hfx⟩ := bind_ok hfx
    obtain ⟨ebj, hebj, hfx⟩ := bind_ok hfx
    obtain ⟨eb, heb, hfx⟩ := bind_ok hfx
    obtain ⟨va, hva, hfx⟩ := bind_ok hfx
    obtain ⟨vb, hvb, hfx⟩ := bind_ok hfx
    try dsimp (config := { zeta := false }) only at hfx
    extract_lets -underBinder rr jr at hfx
    split at hfx
    · cases hfx
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
  · show recordsRoot salt weightsRoot splits records = published
    simpa using hroot

theorem loadedGivesUp : LoadedGivesUp := by
  intro coins man p h
  obtain ⟨hr, splits, hc, hroot⟩ := load_facts h
  exact ⟨fun r hm => ⟨hr r hm, (checkRecords_kept hc r hm).2.2.1, (checkRecords_kept hc r hm).2.2.2⟩, splits, hc, hroot⟩

/-! ## The plan -/

/-- What `afterTie` establishes once it gives a plan. -/
theorem afterTie_ok {i : Input} {passes : List Pass} {tie : Option Tied} {pl : Plan}
    (h : afterTie i passes tie = .ok pl) :
    pl.passes = passes ∧ pl.root = runCommitment passes ∧
    (∀ c, i.held = some c → c.commitment = pl.root ∧ pl.source = c.source ∧
      (i.source = none ∨ i.source = some c.source)) ∧
    (i.held = none → i.source = some pl.source) ∧
    ∃ key drawn, PouwCheck.PearlPlus.drawKey pl.source pl.root = .ok key ∧
      PouwCheck.PearlPlus.draw pl.passes key i.draws = .ok drawn ∧
      pl.todo = todo pl.passes drawn ∧ pl.drawn = drawn.length ∧ pl.picks = drawn := by
  unfold afterTie at h
  extract_lets -underBinder root jsrc at h
  obtain ⟨source, hheld, hnone, h⟩ : ∃ source, (∀ c, i.held = some c → c.commitment = root ∧ source = c.source ∧
      (i.source = none ∨ i.source = some c.source)) ∧ (i.held = none → i.source = some source) ∧
      jsrc source = .ok pl := by
    split at h
    · rename_i c hh
      extract_lets -underBinder jj at h
      split at h
      · cases h
      rename_i hc
      dsimp (config := { zeta := false }) only [jj] at h
      obtain ⟨s', hs', h⟩ := bind_ok h
      cases hs'
      simp only [Bool.or_eq_true, Bool.and_eq_true, not_or, not_and, Bool.not_eq_true, bne_eq_false_iff_eq] at hc
      refine ⟨c.source, fun c' hc' => ?_, fun h' => (by rw [hh] at h'; cases h'), h⟩
      rw [hh] at hc'
      cases hc'
      refine ⟨hc.1, rfl, ?_⟩
      cases hs : i.source with
      | none => exact .inl rfl
      | some s => exact .inr (by simpa [hs] using hc.2 (by simp [hs]))
    · rename_i hh
      split at h
      · rename_i s hs
        obtain ⟨s', hs', h⟩ := bind_ok h
        cases hs'
        exact ⟨s, fun c hc => (by rw [hh] at hc; cases hc), fun _ => hs, h⟩
      · cases h
  dsimp (config := { zeta := false }) only [jsrc] at h
  extract_lets -underBinder jd at h
  obtain ⟨u, h⟩ : ∃ u, jd u = .ok pl := by
    split at h
    · split at h
      · cases h
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  dsimp (config := { zeta := false }) only [jd] at h
  extract_lets -underBinder jo at h
  obtain ⟨u', h⟩ : ∃ u, jo u = .ok pl := by
    split at h
    · split at h
      · cases h
      · exact ⟨(), h⟩
    · exact ⟨(), h⟩
  dsimp (config := { zeta := false }) only [jo] at h
  extract_lets -underBinder jk at h
  obtain ⟨key, hk, h⟩ : ∃ key, PouwCheck.PearlPlus.drawKey source root = .ok key ∧ jk key = .ok pl := by
    split at h
    · rename_i k hk
      obtain ⟨k', hk', h⟩ := bind_ok h
      cases hk'
      exact ⟨k, hk, h⟩
    · cases h
  dsimp (config := { zeta := false }) only [jk] at h
  extract_lets -underBinder jdr at h
  obtain ⟨drawn, hdr, h⟩ : ∃ drawn, PouwCheck.PearlPlus.draw passes key i.draws = .ok drawn ∧ jdr drawn = .ok pl := by
    split at h
    · rename_i d hd
      obtain ⟨d', hd', h⟩ := bind_ok h
      cases hd'
      exact ⟨d, hd, h⟩
    · cases h
  dsimp (config := { zeta := false }) only [jdr] at h
  cases h
  exact ⟨rfl, rfl, hheld, hnone, key, drawn, hk, hdr, rfl, rfl, rfl⟩

theorem planMeans : PlanMeans := by
  intro i pl h
  unfold plan at h
  extract_lets -underBinder j0 at h
  split at h
  · cases h
  dsimp (config := { zeta := false }) only [j0] at h
  cases hd : passesDistinct i.manifests with
  | some w => rw [throwRefusal.eq_def, hd] at h; cases h
  | none =>
  rw [throwRefusal.eq_def, hd] at h
  try dsimp (config := { zeta := false }) only at h
  obtain ⟨passes, hp, h⟩ := bind_ok h
  try dsimp (config := { zeta := false }) only at h
  extract_lets -underBinder js at h
  split at h
  · cases h
  rename_i hsalt
  dsimp (config := { zeta := false }) only [js] at h
  extract_lets -underBinder jt at h
  obtain ⟨u, hserved, h⟩ : ∃ u : Unit, (i.testInstance = false → ∀ p ∈ passes,
      p.scheme.kMin = K_MIN ∧ p.scheme.tileM = TILE ∧ p.scheme.tileN = TILE) ∧ jt u = .ok pl := by
    split at h
    · split at h
      · cases h
      · rename_i hf
        refine ⟨(), fun _ p hm => ?_, h⟩
        obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hm
        have hz : (passes[k], k) ∈ passes.zip (List.range passes.length) :=
          List.mem_iff_getElem.2 ⟨k, by simpa using hk, by simp⟩
        have hn := List.find?_eq_none.1 hf _ hz
        simp only [Bool.or_eq_true, not_or, bne_iff_ne, ne_eq, Decidable.not_not] at hn
        exact ⟨hn.1.1, hn.1.2, hn.2⟩
    · rename_i hti
      refine ⟨(), fun h' => absurd (by simp [h']) hti, h⟩
  dsimp (config := { zeta := false }) only [jt] at h
  extract_lets -underBinder jtie at h
  obtain ⟨tie, h⟩ : ∃ tie, jtie tie = .ok pl := by
    split at h
    · obtain ⟨t', ht', h⟩ := bind_ok h; cases ht'; exact ⟨_, h⟩
    · split at h
      · obtain ⟨t', ht', h⟩ := bind_ok h; cases ht'; exact ⟨_, h⟩
      · cases h
  dsimp (config := { zeta := false }) only [jtie] at h
  cases ha : afterTie i passes tie with
  | error r => rw [ha] at h; cases h
  | ok pl' =>
  rw [ha] at h
  have hpl : pl' = pl := Except.ok.inj h
  subst hpl
  obtain ⟨hpass, hroot, hheld, hnone, key, drawn, hk, hdr, htodo, hdrawn, hpicks⟩ := afterTie_ok ha
  subst hpass
  exact ⟨rfl, hp, fun p hm => eq_of_not_bne fun hb => hsalt (List.any_eq_true.2 ⟨p, hm, hb⟩), hserved, hroot,
    hheld, hnone, key, drawn, hk, hdr, htodo, hdrawn, hpicks⟩

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
theorem forIn_tasks {α σ τ : Type} (V : α → τ) (I : σ → Prop)
    (f : α → σ × Array (Task τ) → Id (ForInStep (σ × Array (Task τ))))
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
  | none => ⟨x.1, x.2, .fail "tile does not match the verifier's own enumeration"⟩
  | some _ => match opener x.1 x.2 with
    | .error e => ⟨x.1, x.2, .fail s!"the prover's files don't open it: {e}"⟩
    | .ok o => ⟨x.1, x.2, checkTile p (noiseOf p x.2) x.2 o⟩

/-- A search loop: each item yields on, or stops with a value `P` of it; a value found is some item's. -/
theorem forIn_found {α β γ : Type} (P : α → β → Prop)
    (f : α → Option (Option β) × γ → Id (ForInStep (Option (Option β) × γ)))
    (hf : ∀ a g, (∃ g', f a (none, g) = .yield (none, g')) ∨
      ∃ v g', f a (none, g) = .done (some (some v), g') ∧ P a v) :
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

theorem replay_eq (pl : Plan) (opener : Nat → Nat → Except String TileOpening)
    (hk : ∀ i : Nat, KConsistent pl.passes[i]!) :
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
        | Except.error e =>
          { pass := i, tile := tid, outcome := .fail (toString "the prover's files don't open it: " ++ toString e) }
        | Except.ok o => ({ pass := i, tile := tid, outcome := checkTile pl.passes[i]! wn tid o } : TileVerdict)) t =
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
  intro p wn tid o bound h0
  generalize h : checkTile p wn tid o = t at h0
  have ht := h0.symm
  clear h0
  have hnf : ∀ w, t ≠ .fail w := by
    intro w hw
    subst hw
    cases bound <;> cases ht
  unfold checkTile at h
  dsimp only at h
  split at h
  case h_2 => exact (hnf _ h.symm).elim
  rename_i pos rows cols ht'
  split at h
  · exact (hnf _ h.symm).elim
  split at h
  · exact (hnf _ h.symm).elim
  rename_i hcov
  split at h
  · exact (hnf _ h.symm).elim
  rename_i hA
  split at h
  · exact (hnf _ h.symm).elim
  rename_i hB
  split at h
  · exact (hnf _ h.symm).elim
  rename_i hT
  split at h
  case h_2 => exact (hnf _ h.symm).elim
  rename_i ops ch hw
  split at h
  · exact (hnf _ h.symm).elim
  rename_i hleaf
  split at h
  · exact (hnf _ h.symm).elim
  rename_i hrows
  obtain ⟨d, hdb, hdt, hy⟩ : ∃ d : TileOutcome,
      (Rule.capCheck p.scheme p.records[pos]! rows cols ops = .ok none ∧ d = .pass ∨
        ∃ w, Rule.capCheck p.scheme p.records[pos]! rows cols ops = .ok (some w) ∧ d = .overCap w) ∧ d = t ∧
      ∀ yRoot, p.yRoots[pos]! = some yRoot → ∃ ys, o.y = .ok ys ∧
        ys.map (·.position) = rows.filter (· < p.records[pos]!.realRows) ∧
        ys.all (opens (rowsDomain (yBinding p.salt p.records[pos]!.matmul.index) p.records[pos]!.matmul.index
          p.records[pos]!.realRows HM96 false) yRoot ROW_SCHEMA) ∧
        Rule.yReason rows cols ops ch (ys.map fun y => words32 (openedValue y.value)) = none := by
    split at h
    case h_3 => exact (hnf _ h.symm).elim
    all_goals
      rename_i hc
      try simp only [pure_bind] at h
      refine ⟨_, by first | exact .inl ⟨hc, rfl⟩ | exact .inr ⟨_, hc, rfl⟩, ?_⟩
      split at h
      · rename_i yRoot hyr
        split at h
        case h_2 => exact (hnf _ h.symm).elim
        rename_i ys hys
        try simp only [pure_bind] at h
        split at h
        · exact (hnf _ h.symm).elim
        rename_i hyo
        split at h
        · exact (hnf _ h.symm).elim
        rename_i hyr'
        refine ⟨h, fun yRoot' hy' => ?_⟩
        rw [hyr] at hy'
        cases hy'
        obtain ⟨hyp, hyo'⟩ := not_or_true hyo
        exact ⟨ys, hys, eq_of_not_bne hyp, of_not_not hyo', hyr'⟩
      · rename_i hnone
        exact ⟨h, fun yRoot hy => absurd hy (hnone yRoot)⟩
  have hbound : Rule.capCheck p.scheme p.records[pos]! rows cols ops = .ok bound := by
    subst hdt
    rcases hdb with ⟨hc, rfl⟩ | ⟨w, hc, rfl⟩
    · cases bound with
      | none => exact hc
      | some w => cases ht
    · cases bound with
      | none => cases ht
      | some w' => cases ht; exact hc
  obtain ⟨hca, hcb⟩ := not_or_true hcov
  obtain ⟨hlp, hlo⟩ := not_or_true hT
  refine ⟨pos, rows, cols, ops, ch, ht', eq_of_not_bne hca, eq_of_not_bne hcb, eq_of_not_bne hlp, of_not_not hA,
    of_not_not hB, of_not_not hlo, hw, eq_of_not_bne hleaf, ?_, hbound, hy⟩
  cases hc : Rule.rowsFailure p.records[pos]! rows cols ops with
  | none => rfl
  | some w => exact absurd hc (hrows w)

/-! ## A tile's inert-work bound -/

theorem mapM_ok_length {ε α β : Type} {f : α → Except ε β} :
    ∀ {l : List α} {l' : List β}, l.mapM f = .ok l' → l'.length = l.length
  | [], l', h => by cases h; rfl
  | a :: l, l', h => by
    rw [List.mapM_cons] at h
    obtain ⟨b0, h0, h⟩ := bind_ok h
    obtain ⟨bs, hbs, h⟩ := bind_ok h
    cases h
    exact congrArg (· + 1) (mapM_ok_length hbs)

theorem mem_zip_snd {α β : Type} :
    ∀ {l : List α} {l' : List β}, l'.length ≤ l.length → ∀ b ∈ l', ∃ a, (a, b) ∈ l.zip l'
  | _, [], _, _, hb => by cases hb
  | [], _ :: _, h, _, _ => absurd h (Nat.not_succ_le_zero _)
  | a :: l, b' :: l', h, b, hb => by
    rcases List.mem_cons.1 hb with rfl | hb
    · exact ⟨a, List.mem_cons_self⟩
    · obtain ⟨a', ha'⟩ := mem_zip_snd (Nat.le_of_succ_le_succ h) b hb
      exact ⟨a', List.mem_cons_of_mem _ ha'⟩

/-- An `Except` loop whose steps only yield on, or stop with `some (some _)`, and that ends holding no such value, yielded
on every item. -/
theorem forIn_ok_all {ε α β γ : Type} (Q : α → Prop)
    (f : α → Option (Option β) × γ → Except ε (ForInStep (Option (Option β) × γ)))
    (hf : ∀ a g, (∃ g', f a (none, g) = .ok (.yield (none, g')) ∧ Q a) ∨
      ∃ v g', f a (none, g) = .ok (.done (some (some v), g'))) :
    ∀ (l : List α) (g : γ) (s : Option (Option β) × γ), forIn l (none, g) f = .ok s →
      (∀ v, s.1 ≠ some (some v)) → ∀ a ∈ l, Q a
  | [], _, _, _, _, a, ha => by cases ha
  | a :: l, g, s, h, hs, b, hb => by
    rcases hf a g with ⟨g', he, hq⟩ | ⟨v, g', he⟩
    · rw [List.forIn_cons, he] at h
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hq
      · exact forIn_ok_all Q f hf l g' s h hs b hb
    · rw [List.forIn_cons, he] at h
      have hs' : (some (some v), g') = s := Except.ok.inj h
      exact absurd (by rw [← hs']) (hs v)

theorem capCheck_of_checkOpened {s : Scheme} {r : UnitRecord} {rows cols : List Nat} {ops : Ops}
    (h : Rule.checkOpened s r rows cols ops = none) : Rule.capCheck s r rows cols ops = .ok none := by
  unfold Rule.checkOpened at h
  cases hr : Rule.rowsFailure r rows cols ops with
  | some w => rw [hr] at h; cases h
  | none =>
  rw [hr] at h
  change Rule.capFailure s r rows cols ops = none at h
  unfold Rule.capFailure at h
  split at h
  · rename_i why hc; rw [hc, h]
  · cases h

theorem tileKeepsTheCap : TileKeepsTheCap := by
  intro s r rows cols ops h
  have hc := capCheck_of_checkOpened h
  unfold Rule.capCheck at hc
  revert hc
  cases r.credited rows cols with
  | mk cr cc =>
  intro hc
  dsimp only at hc ⊢
  intro hcr hcc
  split at hc
  · rename_i hE
    simp only [Bool.or_eq_true, List.isEmpty_iff] at hE
    exact (hE.elim hcr hcc).elim
  split at hc
  · cases hc
  rename_i rowDebit hrd
  obtain ⟨words, hw, hrd⟩ := bind_ok hrd
  obtain ⟨elems, he, hrd⟩ := bind_ok hrd
  obtain rfl := (Except.ok.inj hrd).symm
  split at hc
  · cases hc
  rename_i hcap
  refine ⟨words, elems, hw, he, Nat.not_lt.mp hcap, ?_⟩
  obtain ⟨st, hst, hc⟩ := bind_ok hc
  have hall := forIn_ok_all (fun x : Nat × Nat => x.2 * Rule.CAP_DEN ≤
      Rule.CAP_NUM * creditOf 1 r.kKept r.matmul.n * cc.length * (if s.censusKs.contains r.matmul.k then 4 else 1)) _
    (by
      intro x g
      by_cases hb : x.2 * Rule.CAP_DEN ≤
          Rule.CAP_NUM * creditOf 1 r.kKept r.matmul.n * cc.length * (if s.censusKs.contains r.matmul.k then 4 else 1)
      · exact .inl ⟨_, by rw [ite_eq_right_of_eq_false _ _ (eq_false (Nat.not_lt.mpr hb))]; rfl, hb⟩
      · exact .inr ⟨_, _, by rw [ite_eq_left_of_eq_true _ _ (eq_true (Nat.lt_of_not_le hb))]; rfl⟩) _ _ _ hst (by
      intro v hv
      rw [hv] at hc
      cases hc)
  have hlw : words.length = cr.length := mapM_ok_length hw
  have hle : elems.length = cr.length := by rw [mapM_ok_length he, List.length_zip, hlw, Nat.min_self]
  intro d hd
  obtain ⟨i, hi⟩ := mem_zip_snd (l := cr)
    (Nat.le_of_eq (by rw [List.length_map, List.length_zip, hlw, hle, Nat.min_self])) d hd
  exact hall (i, d) hi

/-! ## The run's acceptance -/

/-- A loaded pass's units of one weight have one k: its split's `k_unit` (`checkRecords`). -/
theorem load_kconsistent {coins : EpochCoins} {man : Lean.Json} {p : Pass} (h : load coins man = .ok p) :
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

/-- A tile whose verdict is ok under `m` is a pass's, opened, and passed `checkTile` on its weight's noise, or was over its
inert-work bound with `m` crediting it nothing. -/
theorem verdictOf_ok {pl : Plan} {opener : Nat → Nat → Except String TileOpening} {m : OverCap} {pi tid : Nat}
    (h : (verdictOf pl opener (pi, tid)).ok m = true) :
    ∃ p o, pl.passes[pi]? = some p ∧ opener pi tid = .ok o ∧
      (checkTile p (noiseOf p tid) tid o = .pass ∨
        m = .uncredited ∧ ∃ why, checkTile p (noiseOf p tid) tid o = .overCap why) := by
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
    unfold TileVerdict.ok at h
    dsimp only at h
    revert h
    cases checkTile pl.passes[pi]! (noiseOf pl.passes[pi]! tid) tid o with
    | pass => exact fun _ => .inl rfl
    | overCap why =>
      intro h
      cases m with
      | uncredited => exact .inr ⟨rfl, why, rfl⟩
      | reject => cases h
    | fail why => intro h; cases h
  · rw [getElem!_neg pl.passes pi hj, default_tile?] at ht
    cases ht

/-- A refusal of `plan` is `check`'s whole answer. -/
theorem refusedBeforeTheCoin : RefusedBeforeTheCoin := by
  intro i opener r h
  unfold check
  rw [h]

theorem acceptMeans : AcceptMeans := by
  intro i opener hacc
  cases hp : plan i with
  | error r => unfold check at hacc; rw [hp] at hacc; cases hacc
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
  have hok : ∀ v ∈ replay pl opener, v.ok i.overCap = true := by
    intro v hv
    simp only [List.isEmpty_iff, List.filterMap_eq_nil_iff] at hb
    have hv' := hb v hv
    by_cases hk : v.ok i.overCap = true
    · exact hk
    · rw [ite_eq_right_of_eq_false _ _ (eq_false hk), Option.map_eq_none_iff] at hv'
      unfold TileVerdict.reason at hv'
      unfold TileVerdict.ok at hk
      revert hv' hk
      cases v.outcome <;> simp
  refine ⟨pl, rfl, rfl, rfl, ?_, hok, fun pi tid hm => verdictOf_ok (hok _ ?_)⟩
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

/-! ## The published work -/

/-- Once `plan` succeeds, `check` publishes the credit on its own verdicts when it accepts, and none when it rejects. -/
theorem check_work {i : Input} {pl : Plan} (opener : Nat → Nat → Except String TileOpening) (hp : plan i = .ok pl) :
    ((check i opener).accept = true → (check i opener).work = some (testCredit i.testInstance
        (creditAtAssumption pl.passes i.draws (overCapDraws pl (check i opener).verdicts) DELTA))) ∧
      ((check i opener).accept = false → (check i opener).work = none) := by
  unfold check
  rw [hp]
  dsimp only
  split
  · exact ⟨fun _ => rfl, fun h => by cases h⟩
  · exact ⟨fun h => (by cases h), fun _ => rfl⟩

theorem workIsTheCredit : WorkIsTheCredit := by
  refine ⟨fun i opener hacc => ?_, fun i opener hrej => ?_⟩
  · cases hp : plan i with
    | error r => rw [refusedBeforeTheCoin i opener r hp] at hacc; cases hacc
    | ok pl => exact ⟨pl, rfl, (check_work opener hp).1 hacc⟩
  · cases hp : plan i with
    | error r => exact (congrArg RunAnswer.work (refusedBeforeTheCoin i opener r hp)).trans rfl
    | ok pl => exact (check_work opener hp).2 hrej

/-- A verdict that accepts every run (`acceptAllCheck`) publishes no work, so not the credit of the plan that a `plan`
accepting every run returns (`acceptAllPlan`). `Witnesses`' `workIsTheCredit_nontrivial` says so of every credit; this
one names its plan, since `WorkIsTheCredit` is a conjunction and the lock reads no `∃` before one. -/
theorem workIsTheCredit_acceptAll :
    ¬ (acceptAllCheck emptyRun noOpening).work = some (testCredit emptyRun.testInstance
      (creditAtAssumption acceptAllPlan.passes emptyRun.draws
        (overCapDraws acceptAllPlan (acceptAllCheck emptyRun noOpening).verdicts) DELTA)) := by
  intro h
  cases h

end Properties.Pouw.PearlPlus
