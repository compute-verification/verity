import Proofs.Flock.Soundness.Session
import Proofs.Flock.Soundness.Game.Prob

/-!
# Several tables in one batched session (PROTOCOL.md §5)

The protocol batches tables. `Commit` sends every table's level-0 cap, the link points are one
draw for all tables, and then every table's reps and openings run concurrently (`sessionB`). The
session accepts when every table accepts (`acceptedB`).

For each table `j`, everything after `Commit` is a batch with table `j` at position `j` and the
other tables around it (`sessAfterB_gamesB`), so `session_knowledge_sound` applies to it. Per
table, this gives the form the audit's randomized-extractor analysis consumes:

* `extractProbB`: the probability, over `K` reruns of everything after `Commit`, that the table
  the extractor fills for table `j` has a given property; `failProbB` is its failure (no
  satisfying witness near it), and the link event is any other property;
* `joint_le_B`: `Pr[the session accepts] · failProbB ≤ ksBoundB`, with `ksBoundB` table `j`'s
  `ε_c⁻ + K·adv₀ + N₀/(eK)`, computed in the session;
* `one_le_fail_add_B`: if commitment implies the event, every extracted table is uncommitted or has
  it, so the two probabilities add up to at least `1` (the audit's `cover`).

For a unit read from table `j`, these are the unit's `ksFail` (`failProbB`), its `linkFail`
(`extractProbB` of the link event), its `cover` (`one_le_fail_add_B'`) and its knowledge bound
(`joint_le_B'`), in `ℝ≥0∞` as `Audit.ExtractionAnalysis` takes them.
-/

namespace FlockSoundness

open Game Model Accounting

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K]
variable {D : Type}

/-- A table of a batched session: a statement, its `fast100` schedule by levels, and its link
layout for the session's `m_pts`. -/
structure TabSpec (mPts : ℕ) where
  S : Statement
  m : ℕ
  k0 : ℕ
  l₀ : Level
  l₁ : Level
  rest : List Level
  ext : ℕ
  hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
  hm : 13 ≤ S.m
  ptLocal : ℕ
  hlay : S.LinkLayout ptLocal mPts

variable {mPts : ℕ}

namespace TabSpec

/-- The table's schedule. -/
abbrev sch (T : TabSpec mPts) : Schedule := ⟨T.m, T.k0, T.l₀ :: T.l₁ :: T.rest, T.ext⟩

/-- The table's run after the link points, with level-0 cap `cap₀`. -/
noncomputable def after (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (T : TabSpec mPts) (cap₀ : Cap D) (pts : Fin 2 → Fin mPts → F) : Game (OutC F D) :=
  tableAfter A H E T.S T.sch T.hm T.ptLocal mPts cap₀ pts

end TabSpec

/-! ## The session -/

/-- The tables' runs after `Commit`, with the caps it sent. -/
noncomputable def gamesB (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) :
    List ((Fin 2 → Fin mPts → F) → Game (OutC F D)) :=
  List.ofFn fun i => (ts.get i).after A H E (caps i)

/-- After `Commit`: one link-point draw, then the games concurrently. -/
noncomputable def sessAfterB (fs : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) :
    Game (List (OutC F D)) :=
  Game.coin (Fin 2 → Fin mPts → F) fun pts => batch (fs.map (· pts))

/-- **A batched session** (PROTOCOL.md §5): `Commit` sends every table's level-0 cap, then one
link-point draw, then every table's reps and openings concurrently. -/
noncomputable def sessionB (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (ts : List (TabSpec mPts)) : Game (List (OutC F D)) :=
  Game.send (Fin ts.length → Cap D) fun caps => sessAfterB (gamesB A H E ts caps)

/-- The session accepts when every table accepts. -/
def acceptedB [Inhabited D] (ts : List (TabSpec mPts)) (outs : List (OutC F D)) : Prop :=
  ∀ i : Fin ts.length, (outs.getD i dfltOut).accepted = true

theorem ofFn_split {T₀ : Type*} {J : ℕ} (f : Fin J → T₀) (j : Fin J) :
    List.ofFn f = (List.ofFn f).take j ++ f j :: (List.ofFn f).drop (j + 1) := by
  have h := List.take_append_drop (j : ℕ) (List.ofFn f)
  rw [List.drop_eq_getElem_cons (by simp), List.getElem_ofFn] at h
  exact h.symm

theorem sessAfterB_eq (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (T : TabSpec mPts) (cap₀ : Cap D) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) :
    sessAfterB (pre ++ T.after A H E cap₀ :: post) =
      sessAfter A H E T.S T.sch T.hm T.ptLocal mPts pre post cap₀ := by
  simp only [sessAfterB, sessAfter, List.map_append, List.map_cons]
  rfl

/-- **Table `j`'s place in the session**: everything after `Commit` is a batch with table `j` at
position `j`. -/
theorem sessAfterB_gamesB (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) (j : Fin ts.length) :
    sessAfterB (gamesB A H E ts caps) =
      sessAfter A H E (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesB A H E ts caps).take j) ((gamesB A H E ts caps).drop (j + 1)) (caps j) := by
  have hs : gamesB A H E ts caps = (gamesB A H E ts caps).take j ++
      (ts.get j).after A H E (caps j) :: (gamesB A H E ts caps).drop (j + 1) :=
    ofFn_split (fun i => (ts.get i).after A H E (caps i)) j
  exact (congrArg sessAfterB hs).trans (sessAfterB_eq A H E (ts.get j) (caps j) _ _)

theorem take_gamesB_length (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (ts : List (TabSpec mPts)) (caps : Fin ts.length → Cap D) (j : Fin ts.length) :
    ((gamesB A H E ts caps).take j).length = j := by
  simp [gamesB, j.isLt.le]

/-- The session strategy after `Commit`, seen with table `j` at its position. -/
noncomputable def stratB (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (ts : List (TabSpec mPts)) (j : Fin ts.length) (τ : Strategy (sessionB A H E ts)) :
    Strategy (sessAfter A H E (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j)) :=
  cast (congrArg Strategy (sessAfterB_gamesB A H E ts τ.1 j)) τ.2

theorem expectN_expect_cast {α : Type} {g₁ g₂ : Game α} (h : g₁ = g₂) (s : Strategy g₁) (n : ℕ)
    (Φ : List α → α → ℝ) :
    expectN g₂ (cast (congrArg Strategy h) s) n
        (fun bs => expect (Φ bs) g₂ (cast (congrArg Strategy h) s)) =
      expectN g₁ s n (fun bs => expect (Φ bs) g₁ s) := by
  subst h; rfl

open Classical in
/-- A fresh run's event and an event of independent runs: their joint probability is the product. -/
theorem expect_mul_expectN {β : Type} (g : Game β) (s : Strategy g) (P : β → Prop)
    (Q : List β → Prop) (n : ℕ) :
    expect (ind P) g s * expectN g s n (ind Q) =
      expectN g s n (fun bs => expect (ind fun o => P o ∧ Q bs) g s) := by
  rw [← expectN_const_mul]
  refine congrArg (expectN g s n) (funext fun bs => ?_)
  by_cases hQ : Q bs
  · have : (ind fun o => P o ∧ Q bs) = ind P := funext fun o => by simp [ind, hQ]
    rw [this]
    simp [ind, hQ]
  · have : (ind fun o => P o ∧ Q bs) = fun _ => 0 := funext fun o => by simp [ind, hQ]
    rw [this, expect_const]
    simp [ind, hQ]

/-! ## Per table: the extractor's probabilities and the knowledge bound -/

open Classical in
/-- **An event of table `j`'s extraction**: the probability, over `Kr` reruns of everything after
`Commit`, that the table the extractor fills from table `j`'s openings satisfies `G`. -/
noncomputable def extractProbB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop) : ℝ :=
  expectN (sessAfterB (gamesB A H E ts τ.1)) τ.2 Kr
    (ind fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs))

/-- **Table `j`'s extraction fails**: no codeword near the extracted table packs a satisfying
witness. -/
noncomputable def failProbB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionB A H E ts)) : ℝ :=
  extractProbB A H E ts j Kr τ fun T => ¬ Committed A (ts.get j).S (ts.get j).sch T

/-- **Table `j`'s knowledge bound in the session**: `ε_c⁻ + K·adv₀ + N₀/(eK)`, every term the
table's own, computed in the session. -/
noncomputable def ksBoundB [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionB A H E ts)) : ℝ :=
  epsS A H E (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest
      (ts.get j).ext (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm
      (ts.get j).ptLocal mPts ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1))
      (τ.1 j) (stratB A H E ts j τ) +
    (Kr * adv₀S A H E (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j)
        (stratB A H E ts j τ) +
      2 ^ (ts.get j).l₀.logLen / (Real.exp 1 * Kr))

theorem extractProbB_nonneg [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop) :
    0 ≤ extractProbB A H E ts j Kr τ G :=
  expectN_nonneg _ _ (fun _ => ind_nonneg _ _) _

open Classical in
/-- **Knowledge soundness of table `j` in a batched session**, in the product form: table `j`
accepts in a fresh run after `Commit`, while the extraction from `Kr` independent reruns fails,
with probability at most `ε_c⁻ + K·adv₀ + N₀/(eK)`. -/
theorem joint_le_B (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (Kr : ℕ) (hK : 1 ≤ Kr) (τ : Strategy (sessionB A H E ts)) :
    expect (ind fun outs : List (OutC F D) => (outs.getD j dfltOut).accepted = true)
        (sessAfterB (gamesB A H E ts τ.1)) τ.2 * failProbB A H E ts j Kr τ ≤
      ksBoundB A H E ts j Kr τ := by
  have h := session_knowledge_sound A hA H E (ts.get j).S (ts.get j).m (ts.get j).k0
    (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext (ts.get j).hsch (ts.get j).hm
    (ts.get j).ptLocal mPts (ts.get j).hlay ((gamesB A H E ts τ.1).take j)
    ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j) (stratB A H E ts j τ) Kr hK
  rw [take_gamesB_length] at h
  unfold stratB at h
  rw [expectN_expect_cast (sessAfterB_gamesB A H E ts τ.1 j) τ.2 Kr] at h
  unfold failProbB extractProbB ksBoundB
  rw [expect_mul_expectN]
  exact h

open Classical in
/-- **The same, for the session's verdict**, in the audit's `ℝ≥0∞` form:
`Pr[the session accepts] · Pr[table j's extraction fails] ≤ ksBoundB`. -/
theorem joint_le_B' (A : Arith F K) (hA : A.Correct)
    [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (Kr : ℕ) (hK : 1 ≤ Kr) (τ : Strategy (sessionB A H E ts)) :
    prob (acceptedB ts) (sessionB A H E ts) τ * ENNReal.ofReal (failProbB A H E ts j Kr τ) ≤
      ENNReal.ofReal (ksBoundB A H E ts j Kr τ) := by
  have hmono : prob (acceptedB ts) (sessionB A H E ts) τ ≤
      prob (fun outs : List (OutC F D) => (outs.getD j dfltOut).accepted = true)
        (sessAfterB (gamesB A H E ts τ.1)) τ.2 :=
    prob_mono (fun outs h => h j) _ _
  have hfin : prob (fun outs : List (OutC F D) => (outs.getD j dfltOut).accepted = true)
      (sessAfterB (gamesB A H E ts τ.1)) τ.2 ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (prob_le_one _ _ _)
  calc prob (acceptedB ts) (sessionB A H E ts) τ * ENNReal.ofReal (failProbB A H E ts j Kr τ)
      ≤ prob (fun outs : List (OutC F D) => (outs.getD j dfltOut).accepted = true)
          (sessAfterB (gamesB A H E ts τ.1)) τ.2 * ENNReal.ofReal (failProbB A H E ts j Kr τ) :=
        mul_le_mul' hmono le_rfl
    _ = ENNReal.ofReal ((prob (fun outs : List (OutC F D) => (outs.getD j dfltOut).accepted = true)
          (sessAfterB (gamesB A H E ts τ.1)) τ.2).toReal * failProbB A H E ts j Kr τ) := by
        rw [ENNReal.ofReal_mul ENNReal.toReal_nonneg, ENNReal.ofReal_toReal hfin]
    _ ≤ ENNReal.ofReal (ksBoundB A H E ts j Kr τ) := by
        refine ENNReal.ofReal_le_ofReal ?_
        rw [prob_toReal]
        exact joint_le_B A hA H E ts j Kr hK τ

open Classical in
/-- **The audit's `cover`, per table.** If commitment implies `G`, every extracted table is
uncommitted or satisfies `G`, so table `j`'s failure and `G` have probabilities adding up to at
least `1`. -/
theorem one_le_fail_add_B [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop)
    (hG : ∀ T, Committed A (ts.get j).S (ts.get j).sch T → G T) :
    1 ≤ failProbB A H E ts j Kr τ + extractProbB A H E ts j Kr τ G := by
  unfold failProbB extractProbB
  rw [← expectN_add]
  calc (1 : ℝ) = expectN (sessAfterB (gamesB A H E ts τ.1)) τ.2 Kr (fun _ => 1) :=
        (expectN_const _ _ 1 Kr).symm
    _ ≤ _ := expectN_mono _ _ (fun bs => by
        by_cases hc : Committed A (ts.get j).S (ts.get j).sch
            (extractTableS H E (ts.get j).sch j (τ.1 j) bs)
        · have hg := hG _ hc
          unfold ind
          rw [if_neg (not_not.2 hc), if_pos hg]
          norm_num
        · have h1 : ind (fun bs => ¬ Committed A (ts.get j).S (ts.get j).sch
              (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs = 1 := by
            unfold ind; rw [if_pos hc]
          rw [h1]
          linarith [ind_nonneg (fun bs => G (extractTableS H E (ts.get j).sch j (τ.1 j) bs)) bs]) Kr

/-- `one_le_fail_add_B` in the audit's `ℝ≥0∞` form. -/
theorem one_le_fail_add_B' [Inhabited D] (A : Arith F K) (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts)) (j : Fin ts.length) (Kr : ℕ)
    (τ : Strategy (sessionB A H E ts)) (G : Oracle F → Prop)
    (hG : ∀ T, Committed A (ts.get j).S (ts.get j).sch T → G T) :
    1 ≤ ENNReal.ofReal (failProbB A H E ts j Kr τ) + ENNReal.ofReal (extractProbB A H E ts j Kr τ G) := by
  have h0 : 0 ≤ failProbB A H E ts j Kr τ := extractProbB_nonneg A H E ts j Kr τ _
  rw [← ENNReal.ofReal_add h0 (extractProbB_nonneg A H E ts j Kr τ G)]
  exact ENNReal.one_le_ofReal.2 (one_le_fail_add_B A H E ts j Kr τ G hG)

end FlockSoundness
