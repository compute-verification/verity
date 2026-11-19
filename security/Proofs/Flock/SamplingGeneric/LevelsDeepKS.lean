module

public import Proofs.Flock.SamplingGeneric.LevelsDeep

/-!
# The compiler at levels is knowledge sound

`CompilerKSLev`: at one level `ks_one_lev`, at two `ks_two_lev`, and at three or more `ks_deep`, `KSLinOn` at the
extractor with one wait per level (`Elev`) on the statements with a middle level.

* `outerListLev`, `foundLev`, `Elev_eval`, `ownLog_Elev`: **the extractor's parts at levels**, and what it keeps; the
  kept openings pass their checks with the checks' calls among the extractor's own (`passes_below`, `passes_foundLev`),
  and a good block's openings at a kept inner run, below kept runs at every level, are kept (`good_below`,
  `mem_below_cons`, `mem_foundLev`).
* `splitDeep`: **the deterministic half**. At an accepted run whose extraction fails, `FailsDeep` holds or the
  extractor's own calls collide: the incorrect set at a best interior per top block is a candidate `Y`, each kept outer
  run's incorrect set is below `Y` in every first-level block, and every run below a kept outer run shows its interior
  roots (`Sessions.interior`), so a kept inner run at which one of its proof blocks is good would make it correct.
* `ks_deep`, `compilerKSLev`: **KS at three levels or more**, and the compiler's statement at every number of levels.
-/

@[expose] public section

open scoped ENNReal

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Opening Partition InScopeLev RLev slot topOf incorrectLev levM forkOn waitAccC
  flatMapRP assemble ownLog ownLog_bind accList OwnColl flatMapRP_eval waitAccC_eval_nil accList_map mem_zip_finRange
  getElem?_map_pair agree_of_passes val_assemble ιOf ιOf_eq acc_of_accList accList_length_eq AllBad KSLinOn Bext
  replayOf)

namespace CompiledLev

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The extractor's parts at levels -/

/-- The outer wait's kept runs at levels, indexed. -/
noncomputable def outerListLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (c : L.Elev.C x) :
    List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen)) :=
  (List.finRange (L.kOut x)).zip ((waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map c.1) []).eval H run').1

/-- **What the extractor at levels keeps** at the outer runs `Z`: below each (`below`). -/
noncomputable def foundLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (c : L.Elev.C x)
    (Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))) : List (List Msg × Opening Bool) :=
  ((flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
    L.below x s (L.forks x) jo.2.1 (c.2 jo.1)) Z).eval H run').1

/-- **The extractor at levels at an accepted run** assembles what it keeps. -/
theorem Elev_eval (H : Msg → Msg) (L : CompiledLev Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.Elev.C x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : (run H P L.V x ω r₀).out = .acc) :
    (L.Elev.eval H P x ω r₀ c).1 =
      some (assemble s (L.foundLev H x s (run H P L.V x ω) c (L.outerListLev H x (run H P L.V x ω) c))) := by
  have hacc' : (run H P L.V x ω r₀).seen.out = .acc := hacc
  unfold foundLev outerListLev
  simp only [Extractor.eval, Elev, RP.eval, hs]
  rw [ite_eq_left hacc', (RP.eval_bind _ _ _ _).1, (RP.eval_bind _ _ _ _).1]
  rfl

/-- The kept openings' checks are the extractor's own calls. -/
theorem ownLog_Elev (H : Msg → Msg) (L : CompiledLev Coin A) {Ω : Type} (P : Prover Ω) (x : Msg) (ω : Ω)
    (r₀ : Fin (L.V.k x) → Coin) (c : L.Elev.C x) {s : AuditStatement} (hs : L.parse x = some s)
    (hacc : (run H P L.V x ω r₀).out = .acc) :
    ∀ t ∈ ownLog H (run H P L.V x ω) (flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
        L.below x s (L.forks x) jo.2.1 (c.2 jo.1)) (L.outerListLev H x (run H P L.V x ω) c)),
      t ∈ ownLog H (run H P L.V x ω) (L.Elev.ext x r₀ c) := by
  intro t ht
  have hacc' : (run H P L.V x ω r₀).seen.out = .acc := hacc
  simp only [Elev, ownLog, hs]
  rw [ite_eq_left hacc', ownLog_bind, ownLog_bind]
  exact List.mem_append_right _ (List.mem_append_left _ ht)

/-- What the extractor keeps below a run passes its opening check, with the check's calls its own. -/
theorem passes_below (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (s : AuditStatement) :
    ∀ (fs : List (Finset (Fin (L.V.k x)))) (r : Fin (L.V.k x) → Coin) (c : L.WaitC x fs.length),
      Sampling.Passes H run' L.commit (L.below x s fs r c)
  | [], _, _ => Sampling.Passes.bind _ fun _ => Sampling.Passes.flatMapRP fun li _ =>
      L.toCompiled.passes_runExt H x run' s li.2.1 li.2.2 _
  | _ :: fs, _, c => Sampling.Passes.bind _ fun _ => Sampling.Passes.flatMapRP fun jo _ =>
      passes_below H L x run' s fs jo.2.1 (c.2 jo.1)

theorem passes_foundLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (s : AuditStatement) (c : L.Elev.C x)
    (Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))) :
    Sampling.Passes H run' L.commit (flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
      L.below x s (L.forks x) jo.2.1 (c.2 jo.1)) Z) :=
  Sampling.Passes.flatMapRP fun jo _ => L.passes_below H x run' s _ jo.2.1 _

theorem mem_outerListLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (c : L.Elev.C x) {j : Fin (L.kOut x)} {r : Fin (L.V.k x) → Coin}
    (hj : (accList (fun r => (run' r).out = .acc) (L.kOut x) ((List.finRange (L.capOut x)).map c.1))[j.val]? =
      some r) :
    (j, (r, (run' r).seen)) ∈ L.outerListLev H x run' c := by
  unfold outerListLev
  rw [waitAccC_eval_nil]
  exact mem_zip_finRange (getElem?_map_pair run' hj)

/-- What a kept outer run's extraction keeps is kept. -/
theorem mem_foundLev (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (c : L.Elev.C x)
    {Z : List (Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen))} {j : Fin (L.kOut x)} {r : Fin (L.V.k x) → Coin}
    (hj : (j, (r, (run' r).seen)) ∈ Z) {q : List Msg × Opening Bool}
    (hq : q ∈ ((L.below x s (L.forks x) r (c.2 j)).eval H run').1) :
    q ∈ L.foundLev H x s run' c Z := by
  unfold foundLev
  rw [flatMapRP_eval]
  exact List.mem_flatMap.2 ⟨_, hj, hq⟩

/-- What the extraction below a kept run of a middle wait keeps is kept below the run above it. -/
theorem mem_below_cons (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (f : Finset (Fin (L.V.k x))) (fs : List (Finset (Fin (L.V.k x))))
    (r : Fin (L.V.k x) → Coin) (c : L.WaitC x (fs.length + 1)) {j : Fin (L.kOut x)} {d : Fin (L.V.k x) → Coin}
    (hj : (accList (fun d => (run' (forkOn f r d)).out = .acc) (L.kOut x)
      ((List.finRange (L.capOut x)).map c.1))[j.val]? = some d) {q : List Msg × Opening Bool}
    (hq : q ∈ ((L.below x s fs (forkOn f r d) (c.2 j)).eval H run').1) :
    q ∈ ((L.below x s (f :: fs) r c).eval H run').1 := by
  show q ∈ (((waitAccC (L.kOut x) ((List.finRange (L.capOut x)).map fun l => forkOn f r (c.1 l)) []).bind
      fun kept => flatMapRP (fun jo : Fin (L.kOut x) × ((Fin (L.V.k x) → Coin) × Seen) =>
        L.below x s fs jo.2.1 (c.2 jo.1)) ((List.finRange (L.kOut x)).zip kept)).eval H run').1
  rw [(RP.eval_bind _ _ _ _).1, flatMapRP_eval, waitAccC_eval_nil]
  have hm : ((List.finRange (L.capOut x)).map fun l => forkOn f r (c.1 l)) =
      ((List.finRange (L.capOut x)).map c.1).map (forkOn f r) := by
    rw [List.map_map]; rfl
  rw [hm, accList_map, List.map_map]
  refine List.mem_flatMap.2 ⟨(j, (forkOn f r d, (run' (forkOn f r d)).seen)), mem_zip_finRange ?_, hq⟩
  rw [List.getElem?_map, hj]; rfl

/-- What a kept inner run's extractions keep is kept below the run above it. -/
theorem mem_below_nil (H : Msg → Msg) (L : CompiledLev Coin A) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript) (r : Fin (L.V.k x) → Coin) (c : L.WaitC x 0) {l : Fin (L.kIn x)}
    {d : Fin (L.V.k x) → Coin}
    (hl : (accList (fun d => (run' (forkOn (L.secondStage x) r d)).out = .acc) (L.kIn x)
      ((List.finRange (L.capIn x)).map c.1))[l.val]? = some d) {q : List Msg × Opening Bool}
    (hq : q ∈ ((L.runExt x s (forkOn (L.secondStage x) r d) (run' (forkOn (L.secondStage x) r d)).seen
      (c.2 l)).eval H run').1) :
    q ∈ ((L.below x s [] r c).eval H run').1 := by
  show q ∈ (((waitAccC (L.kIn x) ((List.finRange (L.capIn x)).map fun l => forkOn (L.secondStage x) r (c.1 l))
      []).bind fun inner => flatMapRP (fun li : Fin (L.kIn x) × ((Fin (L.V.k x) → Coin) × Seen) =>
        L.runExt x s li.2.1 li.2.2 (c.2 li.1)) ((List.finRange (L.kIn x)).zip inner)).eval H run').1
  rw [(RP.eval_bind _ _ _ _).1, flatMapRP_eval, waitAccC_eval_nil]
  have hm : ((List.finRange (L.capIn x)).map fun l => forkOn (L.secondStage x) r (c.1 l)) =
      ((List.finRange (L.capIn x)).map c.1).map (forkOn (L.secondStage x) r) := by
    rw [List.map_map]; rfl
  rw [hm, accList_map, List.map_map]
  refine List.mem_flatMap.2 ⟨(l, (forkOn (L.secondStage x) r d, (run' (forkOn (L.secondStage x) r d)).seen)),
    mem_zip_finRange ?_, hq⟩
  rw [List.getElem?_map, hl]; rfl

/-- **A good block's openings are kept below a run**: at a kept inner run `d` below `r`, a block `v` good at the
extractor's coins there is sampled and shows its session, of a witness in `A`'s relation whose revealed openings at
`v` that pass are among what the extraction below `r` keeps (`Compiled.good_found`'s argument). -/
theorem good_below (H : Msg → Msg) (L : CompiledLev Coin A) {Ω : Type} (P : Prover Ω) (ω : Ω) (x : Msg)
    (s : AuditStatement)
    (hshared : ∀ r i j u u', (run H P L.V x ω r).out = .acc → (L.checked x r)[i]? = some u →
      (L.checked x r)[j]? = some u' → L.skey x (L.checked x r) i = L.skey x (L.checked x r) j →
      L.session x u (run H P L.V x ω r).seen = L.session x u' (run H P L.V x ω r).seen)
    (r : Fin (L.V.k x) → Coin) (c : L.WaitC x 0) {l : Fin (L.kIn x)} {d : Fin (L.V.k x) → Coin}
    (hl : (accList (fun d => (run H P L.V x ω (forkOn (L.secondStage x) r d)).out = .acc) (L.kIn x)
      ((List.finRange (L.capIn x)).map c.1))[l.val]? = some d) {v : ℕ}
    (hgood : L.GoodAt x (L.Found H x (run H P L.V x ω)) (forkOn (L.secondStage x) r d) v (c.2 l)) :
    v ∈ L.checked x (forkOn (L.secondStage x) r d) ∧ ∃ xS vS w,
      L.session x v (run H P L.V x ω (forkOn (L.secondStage x) r d)).seen = some (xS, vS) ∧ A.R xS w ∧
      ∀ q ∈ L.reveal s v (run H P L.V x ω (forkOn (L.secondStage x) r d)).seen w, L.commit.Opens H q.1 q.2 = true →
        q ∈ ((L.below x s [] r c).eval H (run H P L.V x ω)).1 := by
  obtain ⟨i, hvlt, hvi, i₀, u₀, hlead, hu₀, xS, vS, hsess, hsh, w, hw, hR⟩ := hgood
  have hacc : (run H P L.V x ω (forkOn (L.secondStage x) r d)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨hk, hc, hfirst⟩ := Compiled.of_leadOf hlead
  have hsv := hshared _ _ _ _ _ hacc hu₀ hvi hk
  refine ⟨List.mem_of_getElem? hvi, xS, vS, w, by rw [← hsv]; exact hsess, hR, fun q hq hok => ?_⟩
  have hvb : v ∈ L.sessBlocks x (L.checked x (forkOn (L.secondStage x) r d)) i₀ := by
    rw [Compiled.sessBlocks, hc]
    exact List.mem_filterMap.2 ⟨i, Compiled.self_mem_sessSlots hvi hvlt, hvi⟩
  exact L.mem_below_nil H x s _ r c hl (L.toCompiled.mem_runExt H x _ s _ _ (c.2 l) hu₀ u₀.isLt hfirst
    (L.toCompiled.sessExt_mem H x _ s _ _ _ _ hsess hsh hw hvb hq hok))

/-! ## The split -/

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem badOuterTwo_of_singleton (L : CompiledTwo Coin A) {x : Msg} {run' : (Fin (L.V.k x) → Coin) → Transcript}
    {found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop} {X : Finset ℕ}
    {Xs : Finset (Finset ℕ)} (hX : X ∈ Xs) {g : (Fin (L.V.k x) → Coin) → Fin (L.V.k x) → Coin} {e : L.E1 x}
    (h : L.BadOuterTwo x run' found {X} g e) : L.BadOuterTwo x run' found Xs g e := by
  rcases h with h | ⟨X', hX', hk, hbad⟩
  · exact Or.inl h
  · obtain rfl := mem_singleton.1 hX'
    exact Or.inr ⟨X', hX, hk, hbad⟩

/-- **Bad below, level by level**: if every run that keeps an invariant `Pr` is bad for `X` past the middle levels
wherever what the extraction below it keeps is kept, so is every such run below the middle levels' rounds `fs` (each
some level's rounds, which keep `Pr`), for any sets holding `X`. -/
theorem badBelow_of (L : CompiledLev Coin A) (H : Msg → Msg) (x : Msg) (s : AuditStatement)
    (run' : (Fin (L.V.k x) → Coin) → Transcript)
    (found : (u : Fin (L.nPU x)) → (Fin (L.V.k x) → Coin) → A.EC (L.sh x u) → Prop) (X : Finset ℕ)
    (fd : List (List Msg × Opening Bool)) (Pr' : (Fin (L.V.k x) → Coin) → Prop)
    (hP : ∀ j r d, Pr' r → Pr' (forkOn (L.stages x j) r d))
    (hin : ∀ r (c : L.WaitC x 0), Pr' r → (∀ q ∈ ((L.below x s [] r c).eval H run').1, q ∈ fd) →
      L.twoView.BadOuterTwo x run' found {X} (forkOn (L.secondStage x) r) c) :
    ∀ (fs : List (Finset (Fin (L.V.k x)))), (∀ f ∈ fs, ∃ j, f = L.stages x j) → ∀ (Xs : Finset (Finset ℕ)), X ∈ Xs →
      ∀ (r : Fin (L.V.k x) → Coin) (c : L.WaitC x fs.length), Pr' r →
        (∀ q ∈ ((L.below x s fs r c).eval H run').1, q ∈ fd) → L.BadBelow x run' found fs Xs r c := by
  intro fs
  induction fs with
  | nil =>
    intro _ Xs hX r c hr hkeep
    exact badOuterTwo_of_singleton L.twoView hX (hin r c hr hkeep)
  | cons f fs ih =>
    intro hfs Xs hX r c hr hkeep
    obtain ⟨j, rfl⟩ := hfs f List.mem_cons_self
    by_cases hshort : (accList (fun d => (run' (forkOn (L.stages x j) r d)).out = .acc) (L.kOut x)
      ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x
    · exact Or.inl hshort
    refine Or.inr ⟨X, hX, accList_length_eq hshort, fun i d hi => ?_⟩
    exact ih (fun f hf => hfs f (List.mem_cons_of_mem _ hf)) {X} (mem_singleton_self _) (forkOn (L.stages x j) r d)
      (c.2 i) (hP j r d hr) fun q hq => hkeep q (L.mem_below_cons H x s run' _ fs r c hi hq)

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem forks_stages (L : CompiledLev Coin A) (x : Msg) : ∀ f ∈ L.forks x, ∃ j, f = L.stages x j := by
  intro f hf
  obtain ⟨j, -, rfl⟩ := List.mem_map.1 hf
  exact ⟨j, rfl⟩

/-- **The split at three levels or more**: at an accepted run whose extraction fails, the failure event holds or the
extractor's own calls collide. Otherwise the kept openings assemble a witness; each first-level block's proof blocks
share one top block, so the incorrect set at a best interior per top block (`exists_best`, at the levels after the
first, `escAt`) is a candidate `Y`, and each kept outer run's incorrect set, at the interiors the kept openings under
its interior roots give, is below `Y` in every first-level block (`missProb_incorrectLev_congr`); every run below it
shows its interior roots (`Sessions.interior`, at each level's rounds and the last's), so a kept inner run at which one
of its proof blocks is good would make it correct (`Sessions.opens` at its top block, `good_below`). -/
theorem splitDeep {H : Msg → Msg} {L : CompiledLev Coin A} (hbind : L.commit.Binds H) (hS : L.Sessions H) (δ : ℝ)
    (x : Msg) (hm : 0 < L.mid x) {Ω : Type} (P : Prover Ω) (ω : Ω) (r₀ : Fin (L.V.k x) → Coin)
    (c : L.Elev.C x) (hacc : (run H P L.V x ω r₀).out = .acc)
    (hfail : ¬ (RLev H L.commit L.parse L.partition L.levels L.top δ).ext x (L.Elev.eval H P x ω r₀ c).1) :
    L.FailsDeep δ x (run H P L.V x ω) (L.toCompiled.Found H x (run H P L.V x ω)) c ∨ OwnColl H L.Elev P x ω r₀ c := by
  classical
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin⟩ := hS.refuses x Ω P ω r₀ hacc
  have hmid : L.mid x + 2 = Lv.levels.length := by
    have h := mid_eq hs hPt hLv
    omega
  obtain ⟨Q, hQ⟩ : ∃ Q, L.top s = some Q := by
    cases hq : L.top s with
    | none =>
      have h := hin.2.2.2
      rw [hq] at h
      have h' : Lv.levels.length = 1 := h
      omega
    | some Q => exact ⟨Q, rfl⟩
  have hT := hin.2.2.2
  rw [hQ] at hT
  obtain ⟨-, hsome, l₀, rest, hL, hrest, hiff⟩ := hT
  have hins : ∀ Z ∈ l₀.blocks.parts, ∀ ω, ∀ b ∈ (Verity.Level.sampleBelow (Lv.levels.drop (0 + 1)) Z).draw ω,
      b ⊆ Z := by
    have hi := Lv.inside
    rw [hL] at hi
    obtain ⟨l₁, rest', hr⟩ := List.exists_cons_of_ne_nil hrest
    rw [hr] at hi
    have hd : Lv.levels.drop (0 + 1) = l₁ :: rest' := by rw [hL, hr]; rfl
    intro Z hZ
    rw [hd]
    exact fun ω b hb => (Verity.Level.sampleBelow_draw hi.2 hZ ω b hb).2
  have hρ : ∀ Z ∈ l₀.blocks.parts, ∃ ρ, ∀ i ∈ Z, replayOf Pt Q i.val = some ρ := by
    intro Z hZ
    obtain ⟨i, hi⟩ := l₀.blocks.nonempty_of_mem_parts hZ
    obtain ⟨ρ, hρ⟩ := Option.isSome_iff_exists.1 (hsome i.val i.isLt)
    exact ⟨ρ, fun j hj => ((hiff j i).2 ⟨Z, hZ, hj, hi⟩).trans hρ⟩
  by_cases hcoll : OwnColl H L.Elev P x ω r₀ c
  · exact Or.inr hcoll
  refine Or.inl ⟨s, Pt, Lv, l₀, rest, hs, hPt, hLv, hin, hL, hm, ?_⟩
  set run' := run H P L.V x ω with hrun'
  set fd := L.foundLev H x s run' c (L.outerListLev H x run' c) with hfd
  have hpass : ∀ q ∈ fd, L.commit.Opens H q.1 q.2 = true ∧
      ∀ t ∈ ((L.commit.opens q.1 q.2).eval H).2, t ∈ Claims.Sampling.ownLog H run' (L.Elev.ext x r₀ c) := by
    intro q hq
    obtain ⟨h1, h2⟩ := L.passes_foundLev H x run' s c _ q hq
    exact ⟨h1, fun t ht => L.ownLog_Elev H P x ω r₀ c hs hacc t (h2 t ht)⟩
  have hagree : ∀ q ∈ fd, ∀ q' ∈ fd, q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => agree_of_passes hbind hpass hcoll hq hq' h1 h2
  rw [L.Elev_eval H P x ω r₀ c hs hacc] at hfail
  set w := assemble s fd with hw
  set I : (ℕ → ℕ → Bool) → Finset ℕ := fun ι => incorrectLev s Pt (some Q) (w.val H L.commit s) ι with hI
  let m : Finset (Fin Pt.blocks.length) → Finset ℕ → ℝ≥0∞ := fun Z X => escAt Pt Lv 0 Z X
  have hbest : ∀ ρ : ℕ, ∃ ιρ : ℕ → Bool, ∀ Z ∈ l₀.blocks.parts, (∀ i ∈ Z, replayOf Pt Q i.val = some ρ) →
      ∀ ι' : ℕ → Bool, m Z (I fun _ => ι') ≤ m Z (I fun _ => ιρ) := by
    intro ρ
    by_cases hex : ∃ Z₀ ∈ l₀.blocks.parts, ∀ i ∈ Z₀, replayOf Pt Q i.val = some ρ
    · obtain ⟨Z₀, hZ₀, hZ₀ρ⟩ := hex
      obtain ⟨ιρ, hιρ⟩ := exists_best (fun ι' : ℕ → Bool => I fun _ => ι') (range Pt.blocks.length)
        (fun _ => incorrectLev_subset _ _ _ _ _) (m Z₀)
      refine ⟨ιρ, fun Z hZ hZρ ι' => ?_⟩
      obtain ⟨i, hi⟩ := l₀.blocks.nonempty_of_mem_parts hZ
      obtain ⟨j, hj⟩ := l₀.blocks.nonempty_of_mem_parts hZ₀
      obtain ⟨X, hX, hiX, hjX⟩ := (hiff i j).1 ((hZρ i hi).trans (hZ₀ρ j hj).symm)
      have h1 : Z = X := l₀.blocks.eq_of_mem_parts hZ hX hi hiX
      have h2 : Z₀ = X := l₀.blocks.eq_of_mem_parts hZ₀ hX hj hjX
      rw [h1, ← h2]
      exact hιρ ι'
    · refine ⟨fun _ => false, fun Z hZ hZρ _ => absurd ⟨Z, hZ, hZρ⟩ hex⟩
  choose ιs hιs using hbest
  have hYc : I ιs ∈ cands Pt Lv δ := by
    refine mem_filter.2 ⟨mem_powerset.2 (incorrectLev_subset _ _ _ _ _), ?_⟩
    by_contra hlt
    refine hfail ⟨s, Pt, Lv, hs, hPt, hLv, ιs, ?_⟩
    rw [hQ]
    exact lt_of_not_ge hlt
  by_cases hshort : (accList (fun r => (run' r).out = .acc) (L.kOut x)
    ((List.finRange (L.capOut x)).map c.1)).length < L.kOut x
  · exact Or.inl hshort
  refine Or.inr ⟨_, hYc, accList_length_eq hshort, fun j r hj => ?_⟩
  have hjZ := L.mem_outerListLev H x run' c hj
  let ιr : ℕ → ℕ → Bool := fun ρ => ιOf fd (L.iroots x (run' r).seen ρ)
  have hXr : I ιr ∈ aboveDeep Pt Lv l₀ (I ιs) := by
    refine mem_filter.2 ⟨mem_powerset.2 (incorrectLev_subset _ _ _ _ _), fun Z hZ => ?_⟩
    obtain ⟨ρ, hZρ⟩ := hρ Z hZ
    have e1 := missProb_incorrectLev_congr (s := s) (val := w.val H L.commit s) (hins Z hZ) hZρ ιr
      (fun _ => ιr ρ) rfl
    have e2 := missProb_incorrectLev_congr (s := s) (val := w.val H L.commit s) (hins Z hZ) hZρ ιs
      (fun _ => ιs ρ) rfl
    exact e1.trans_le ((hιs ρ Z hZ hZρ (ιr ρ)).trans_eq e2.symm)
  refine L.badBelow_of H x s run' (L.toCompiled.Found H x run') (I ιr) fd
    (fun r' => ∀ ρ, L.iroots x (run' r').seen ρ = L.iroots x (run' r).seen ρ)
    (fun j' r' d hr' ρ => ((hS.interior x Ω P ω r' d ρ).2 j').trans (hr' ρ)) ?_ (L.forks x) (L.forks_stages x)
    _ hXr r (c.2 j) (fun _ => rfl) fun q hq => L.mem_foundLev H x s run' c hjZ hq
  intro r' c' hr' hkeep
  by_cases hshort2 : (accList (fun d => (run' (forkOn (L.secondStage x) r' d)).out = .acc) (L.kIn x)
    ((List.finRange (L.capIn x)).map c'.1)).length < L.kIn x
  · exact Or.inl hshort2
  refine Or.inr ⟨_, mem_singleton_self _, accList_length_eq hshort2, fun l' e hl v hv hgood => ?_⟩
  obtain ⟨hvc, xS, vS, wS, hsess, hRw, hkept⟩ := L.good_below H P ω x s (hS.shared x Ω P ω) r' c' hl hgood
  have hacc' : (run' (forkOn (L.secondStage x) r' e)).out = .acc := by
    have h := acc_of_accList hl
    exact h
  obtain ⟨b, ρ, rb, hb, hρv, hrb, hcor⟩ :=
    hS.opens x s Pt Lv Ω P ω (forkOn (L.secondStage x) r' e) v xS vS wS hs hPt hLv hin hacc' hvc hsess hRw
  rw [hQ] at hρv hrb
  have hcorr := hcor (w.val H L.commit s) (ιr ρ) fun q hq hok => by
    have hqf : q ∈ fd := hkeep q (hkept q hq hok)
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · rw [hr] at hok
      exact val_assemble hagree hqf hr hok
    · rw [(hS.interior x Ω P ω r' e ρ).1, hr' ρ] at hi
      show ιOf fd (L.iroots x (run' r).seen ρ) q.2.pos = q.2.val
      rw [← hi]
      exact ιOf_eq hagree hqf
  have hvI : v ∈ incorrectLev s Pt (some Q) (w.val H L.commit s) ιr := hv
  unfold incorrectLev at hvI
  exact (mem_filter.1 hvI).2 ⟨b, ρ, rb, hb, hρv, hrb, hcorr⟩

/-! ## KS at levels -/

set_option synthInstance.maxSize 1024 in
open Classical in
/-- **`CompiledLev.KS` at three levels or more**: `KSLinOn` at `Elev` on the statements in scope with a middle level
(`Sampling.KSLin_of_ownColl` at `FailsDeep`, `pr_failsDeep_le` and `splitDeep`, with the bound `1` off them). -/
theorem ks_deep {H : Msg → Msg} {L : CompiledLev Coin A} (hA : A.KS H) (hbind : L.commit.Binds H)
    (hS : L.Sessions H) (δ : ℝ) (hδ : 0 ≤ δ) :
    KSLinOn H L.V (fun x => L.Scope x ∧ 0 < L.mid x) (RLev H L.commit L.parse L.partition L.levels L.top δ) L.Elev
      (L.κLev δ) L.γLev (fun _ => 1) (Bext H L.V L.Elev) := by
  have hSK : L.toCompiled.SessionKS H := Compiled.sessionKS_of hA ⟨hS.fixed, hS.runs⟩
  have h := Sampling.KSLin_of_ownColl H L.V (RLev H L.commit L.parse L.partition L.levels L.top δ) L.Elev
    (fun x => if 0 < L.mid x then L.κLev δ x else 1) L.γLev
    (fun x Ω P => if 0 < L.mid x then
        fun s => (run H P L.V x s.1 s.2.1).out = .acc ∧
          L.FailsDeep δ x (run H P L.V x s.1) (L.toCompiled.Found H x (run H P L.V x s.1)) s.2.2
      else fun s => (run H P L.V x s.1 s.2.1).out = .acc ∧
          ¬ (RLev H L.commit L.parse L.partition L.levels L.top δ).ext x (L.Elev.eval H P x s.1 s.2.1 s.2.2).1)
    (fun x Ω P s hacc hfail => by
      by_cases h1 : 0 < L.mid x
      · rw [if_pos h1]
        exact (splitDeep hbind hS δ x h1 P s.1 s.2.1 s.2.2 hacc hfail).imp_left fun h => ⟨hacc, h⟩
      · rw [if_neg h1]
        exact Or.inl ⟨hacc, hfail⟩)
    (fun x Ω _ _ P => by
      by_cases h1 : 0 < L.mid x
      · rw [if_pos h1, if_pos h1]
        obtain ⟨β, hβ0, hβ, hκ⟩ := hSK x Ω P
        exact L.pr_failsDeep_le H hδ hS x Ω P (fun ω => L.toCompiled.Found H x (run H P L.V x ω)) β hβ0 hβ hκ
      · rw [if_neg h1, if_neg h1]
        exact (Pr_le_one' _).trans (le_add_of_nonneg_right (Real.sqrt_nonneg _)))
  intro x hx Ω _ _ P
  have hx' := h x Ω P
  dsimp only at hx'
  rwa [if_pos hx.2] at hx'

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- A statement in scope has one level, two, or a middle one. -/
theorem oneLevel_or_twoLevel_or_mid {L : CompiledLev Coin A} {x : Msg} (hx : L.Scope x) :
    L.OneLevel x ∨ L.TwoLevel x ∨ 0 < L.mid x := by
  obtain ⟨s, Pt, Lv, hs, hPt, hLv, hin⟩ := hx
  have hm := mid_eq hs hPt hLv
  cases htop : L.top s with
  | none =>
    refine Or.inl fun s' hs' => ?_
    rw [hs, Option.some.injEq] at hs'
    subst hs'
    exact htop
  | some Q =>
    have hT := hin.2.2.2
    rw [htop] at hT
    obtain ⟨-, -, l₀, rest, hL, hrest, -⟩ := hT
    obtain ⟨l₁, rest', rfl⟩ := List.exists_cons_of_ne_nil hrest
    cases rest' with
    | nil =>
      refine Or.inr (Or.inl fun s' Pt' Lv' hs' hPt' hLv' => ?_)
      rw [hs, Option.some.injEq] at hs'
      subst hs'
      rw [hPt, Option.some.injEq] at hPt'
      subst hPt'
      rw [hLv, Option.some.injEq] at hLv'
      subst hLv'
      exact ⟨l₀, l₁, hL⟩
    | cons l₂ rest'' =>
      refine Or.inr (Or.inr ?_)
      rw [hm, hL]
      simp

/-- **The sampled compiler at levels is knowledge sound**, at every number of levels: `ks_one_lev` at one,
`ks_two_lev` at two and `ks_deep` at three or more, each at `Elev`, `κLev` and `γLev`. -/
theorem compilerKSLev : CompilerKSLev := by
  intro Coin _ _ _ A L H hA hbind hS δ hδ x hx Ω _ _ P
  rcases oneLevel_or_twoLevel_or_mid hx with h | h | h
  · exact ks_one_lev hA hbind hS δ hδ x ⟨hx, h⟩ Ω P
  · exact ks_two_lev hA hbind hS δ hδ x ⟨hx, h⟩ Ω P
  · exact ks_deep hA hbind hS δ hδ x ⟨hx, h⟩ Ω P

end CompiledLev

end Claims.SamplingGeneric
