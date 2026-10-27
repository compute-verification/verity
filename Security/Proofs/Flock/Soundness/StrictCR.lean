import Proofs.Flock.Soundness.Assumptions
import Proofs.Flock.Soundness.Game.Seq
import Proofs.Flock.Soundness.CompiledSound
import Proofs.Flock.Soundness.Audit.FlockBatchedAcc

/-!
# The compiled and knowledge terms, and `δ_tree`, under strict-time collision resistance

`SHA512CRStrict` (`cr/sha-512`) is a hypothesis about one collision finder. This file builds, from the prover, the
finders that the compiled theorem, the knowledge term and `δ_tree` count, and restates each term over them.

* A finder (`Finder`) is a game played by a fixed strategy that outputs two byte strings or nothing. `Finder.CR H F q` is
  `SHA512CRStrict` for `F` at the budget `q`, and bounds `F`'s success by `q²/2^513` (`Finder.CR.le`).
* The rewinding finders run the prover to a commitment and continue it twice: `twoRuns` after a cap every run sends
  first, `forkFinder` at a rep's cap along the run (the fork `Lock.forkE` measures). They output the two openings that
  conflict. The self-clash finder runs the prover once (`oneRun`). Merkle binding makes every pair they count a collision
  (`op₀_conflict_collision`, `opR_conflict_collision`, `SelfClash.collision`), so each term is at most its finder's
  success.
* So the compiled theorem (`table_sound_compiled_strict`), a table's knowledge bound in a session
  (`ksBoundAccB_le_strict`), the knowledge term (`ksAvgBE_le_strict`) and the rewinding bound at a registered root
  (`tree_le_strict`; `TreeRewind.le` for an event a rewinding covers, which is how `δ_tree` enters the end-to-end
  bounds) are closed forms in the finders' budgets: `qF` for a rewinding finder (two runs of the prover and two
  verifications), `qS` for a one-run finder (one run and one verification).

The budgets are the hypotheses' to state: the model plays the prover as a strategy and counts no evaluations.
-/

namespace FlockSoundness.StrictCR

open Game Model Accounting
open scoped ENNReal

/-- The executable's field. -/
local notation "𝔽" => FlockLevel3.GF128

/-- A finder's output: two byte strings, or nothing. -/
abbrev Pair : Type := Option (List UInt8 × List UInt8)

section Finders

variable {D : Type} (H : List UInt8 → D)

/-- The output is a collision of `H`. -/
def IsColl (o : Pair) : Prop := ∃ x y, o = some (x, y) ∧ x ≠ y ∧ H x = H y

open Classical in
/-- A collision of `H` when `p` holds (`h` gives one), nothing otherwise. -/
noncomputable def collOf (p : Prop) (h : p → Merkle.Collision H) : Pair :=
  if hp : p then some ((h hp).choose, (h hp).choose_spec.choose) else none

theorem collOf_isColl {p : Prop} (h : p → Merkle.Collision H) (hp : p) : IsColl H (collOf H p h) := by
  unfold collOf
  rw [dite_eq_left hp]
  exact ⟨_, _, rfl, (h hp).choose_spec.choose_spec⟩

/-- An event of a run, as a finder's output: a collision whenever the event holds. -/
noncomputable def onePick {α : Type} (q : α → Prop) (hq : ∀ a, q a → Merkle.Collision H) (a : α) : Pair :=
  collOf H (q a) (hq a)

theorem ind_le_onePick {α : Type} (q : α → Prop) (hq : ∀ a, q a → Merkle.Collision H) (a : α) :
    ind q a ≤ ind (IsColl H) (onePick H q hq a) := by
  unfold ind
  by_cases h : q a
  · have hc : IsColl H (onePick H q hq a) := collOf_isColl H (hq a) h
    rw [ite_eq_left h, ite_eq_left hc]
  · rw [ite_eq_right h]
    split_ifs <;> norm_num

/-- Two runs' conflict at some position, as a finder's output. -/
noncomputable def conflictPick {α P V : Type} (op : α → P → Option V)
    (hcol : ∀ x p, Rewinding.ConflictAt op x p → Merkle.Collision H) (b b' : α) : Pair :=
  onePick H (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (fun x h => h.elim (hcol x)) (b, b')

/-- **A collision finder**: a game played by a fixed strategy, with what it found as its output. -/
structure Finder (γ : Type) : Type 1 where
  g : Game γ
  s : Strategy g

/-- The expectation of `f` over a finder's coins. -/
noncomputable def Finder.ev {γ : Type} (Fd : Finder γ) (f : γ → ℝ) : ℝ := expect f Fd.g Fd.s

/-- **`SHA512CRStrict` for one finder**, at the budget `q` on every outcome: `q` bounds the SHA-512 evaluations of the
prover's runs the finder plays and of its verifications. The cost is `q` itself (`fun _ => q`), so Lean doesn't check
that the finder stays within it: a budget below the finder's real count makes this a claim about SHA-512 that
`cr/sha-512` doesn't make. Whoever instantiates it states an honest `q` (`ASSUMPTIONS.md`).

Assumption record. Claim: `cr/sha-512` (`Claims.StrictBirthday`). It assumes that this one finder, which plays the
prover's runs and outputs what they open two ways, finds a SHA-512 collision with probability at most `q²/2^513`. That
is the generic birthday bound for 512-bit outputs at `q` evaluations, and no SHA-512 collision is known. A hash premise:
it is `Assumptions.SHA512CRStrict` at the constant cost `q`, so it reduces to `StrictBirthday` and asks nothing of the
verifier. The one condition Lean doesn't check is the budget: `q` must bound the finder's real evaluations. -/
def Finder.CR (Fd : Finder Pair) (q : ℝ) : Prop :=
  Assumptions.SHA512CRStrict H Fd.g Fd.s id (fun _ => q) q

theorem Finder.CR.le {H : List UInt8 → D} {Fd : Finder Pair} {q : ℝ} (h : Fd.CR H q) :
    Fd.ev (ind (IsColl H)) ≤ q ^ 2 / 2 ^ (513 : ℝ) := by
  have h' := h fun _ => le_rfl
  rw [prob_toReal] at h'
  exact h'

/-- **One run** of `g` against `s`, its output read by `pick`. -/
noncomputable def oneRun {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → γ) : Finder γ :=
  ⟨g.map pick, Strategy.map pick g s⟩

/-- **Two independent runs** of `g` against `s`, their outputs read by `pick`. -/
noncomputable def twoRuns {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → β → γ) : Finder γ :=
  ⟨g.bind fun b => g.map (pick b), Strategy.bind g _ s fun b => Strategy.map (pick b) g s⟩

theorem oneRun_ev {β γ : Type} (f : γ → ℝ) (g : Game β) (s : Strategy g) (pick : β → γ) :
    (oneRun g s pick).ev f = expect (fun b => f (pick b)) g s :=
  expect_map f pick g s

theorem twoRuns_ev {β γ : Type} (f : γ → ℝ) (g : Game β) (s : Strategy g) (pick : β → β → γ) :
    (twoRuns g s pick).ev f = expect₂ (fun b b' => f (pick b b')) g s := by
  show expect f (g.bind fun b => g.map (pick b)) (Strategy.bind g _ s fun b => Strategy.map (pick b) g s) = _
  rw [expect_bind]
  exact congrArg (fun F => expect F g s) (funext fun b => expect_map f (pick b) g s)

theorem expect_one_le {α : Type} (g : Game α) (s : Strategy g) (q : α → Prop)
    (hq : ∀ a, q a → Merkle.Collision H) :
    expect (ind q) g s ≤ (oneRun g s (onePick H q hq)).ev (ind (IsColl H)) := by
  rw [oneRun_ev]
  exact expect_mono (ind_le_onePick H q hq) g s

theorem expect₂_conflict_le {α P V : Type} (g : Game α) (s : Strategy g) (op : α → P → Option V)
    (hcol : ∀ x p, Rewinding.ConflictAt op x p → Merkle.Collision H) :
    expect₂ (fun b b' => ind (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (b, b')) g s ≤
      (twoRuns g s (conflictPick H op hcol)).ev (ind (IsColl H)) := by
  rw [twoRuns_ev]
  exact expect_mono (fun b => expect_mono (fun b' =>
    ind_le_onePick H (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (fun x h => h.elim (hcol x))
      (b, b')) _ _) _ _

section Fork

variable {α β T Dc I γ : Type} {get : I → ℕ → Option (T × Dc)} {R : I → α → β → Prop}

/-- **The rewinding finder at a table node** (the finder `Lock.forkE` measures): along the run, at the node of `slot`,
two continuations from the cap the prover sends there, their outputs read by `pick`. It plays the strategy, and at every
other table node the table `plur` chooses, as the oracle prover does; a run that doesn't reach the slot outputs `dflt`. -/
noncomputable def forkFinder (plur : Lock.Chooser T Dc β) (slot : ℕ) (pick : Dc → β → β → γ) (dflt : γ) :
    {i : I} → {g : Game α} → {g' : Game β} → Lock T Dc get R i g g' → Strategy g' → Finder γ
  | _, _, _, .ret _, _ => ⟨.ret dflt, PUnit.unit⟩
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C iF iN _ _ L, σ' =>
      ⟨@Game.coin _ C iF iN fun c => (forkFinder plur slot pick dflt (L c) (σ' c)).g,
        fun c => (forkFinder plur slot pick dflt (L c) (σ' c)).s⟩
  | _, _, _, .send _ L, σ' => forkFinder plur slot pick dflt (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' =>
      if s = slot then twoRuns (k' σ'.1) σ'.2 (pick σ'.1)
      else forkFinder plur slot pick dflt (L (plur s σ'.1 (k' σ'.1) σ'.2) σ'.1) σ'.2

omit H in
theorem forkE_le_ev (plur : Lock.Chooser T Dc β) (slot : ℕ) {Φ : Dc → β → β → ℝ} {pick : Dc → β → β → γ}
    {dflt : γ} {f : γ → ℝ} (hΦ : ∀ d b b', Φ d b b' ≤ f (pick d b b')) (hdflt : 0 ≤ f dflt) :
    ∀ {i : I} {g : Game α} {g' : Game β} (L : Lock T Dc get R i g g') (σ' : Strategy g'),
      Lock.forkE plur slot Φ L σ' ≤ (forkFinder plur slot pick dflt L σ').ev f
  | _, _, _, .ret _, _ => hdflt
  | _, _, _, @Lock.coin _ _ _ _ _ _ _ _ C _ _ _ _ L, σ' => by
      simp only [Lock.forkE, forkFinder, Finder.ev, expect]
      exact div_le_div_of_nonneg_right
        (Finset.sum_le_sum fun c _ => forkE_le_ev plur slot hΦ hdflt (L c) (σ' c)) (Nat.cast_nonneg _)
  | _, _, _, .send _ L, σ' => forkE_le_ev plur slot hΦ hdflt (L σ'.1) σ'.2
  | _, _, _, @Lock.table _ _ _ _ _ _ _ _ s _ _ k' _ _ _ L, σ' => by
      simp only [Lock.forkE, forkFinder]
      split_ifs
      · rw [twoRuns_ev]
        exact expect_mono (fun b => expect_mono (hΦ σ'.1 b) _ _) _ _
      · exact forkE_le_ev plur slot hΦ hdflt _ σ'.2

theorem forkE_conflict_le {P V : Type} (plur : Lock.Chooser T Dc β) (slot : ℕ) (op : Dc → β → P → Option V)
    (hcol : ∀ d x p, Rewinding.ConflictAt (op d) x p → Merkle.Collision H) {i : I} {g : Game α} {g' : Game β}
    (L : Lock T Dc get R i g g') (σ' : Strategy g') :
    Lock.forkE plur slot (fun d b b' => ind (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p) (b, b')) L σ' ≤
      (forkFinder plur slot (fun d => conflictPick H (op d) (hcol d)) none L σ').ev (ind (IsColl H)) :=
  forkE_le_ev plur slot (fun d b b' => ind_le_onePick H (fun x : β × β => ∃ p, Rewinding.ConflictAt (op d) x p)
    (fun x h => h.elim (hcol d x)) (b, b')) (ind_nonneg _ _) L σ'

end Fork

open Classical in
/-- **`δ_tree`'s rewinding bound under `SHA512CRStrict`.** Runs of `g` against `s` open positions `P` of a registered
root to values (`op`), and `vstar` is the plurality of those openings (`Rewinding.IsPluralityE`), the leaves the bound
reads the root as. If each run opens at most `Q` positions and two runs' conflicting openings are a collision (`hcol`,
Merkle binding), then the probability that a run opens a position off `vstar` is at most `√(N·Q·q²/2^513)`, for `q` the
budget of the finder that runs the prover twice after the registration (`DESIGN.md` §1: `δ_tree ≤ √(N_s·Q_s·Adv)`). -/
theorem tree_le_strict {α P V : Type} [Fintype P] [Fintype V] [DecidableEq V] (op : α → P → Option V)
    (g : Game α) (s : Strategy g) (vstar : P → V) (hv : Rewinding.IsPluralityE op g s vstar) (Q : ℕ)
    (hQ : ∀ a, (Finset.univ.filter fun p => (op a p).isSome).card ≤ Q)
    (hcol : ∀ x p, Rewinding.ConflictAt op x p → Merkle.Collision H) {q : ℝ}
    (hCR : (twoRuns g s (conflictPick H op hcol)).CR H q) :
    expect (ind fun a => ∃ p, Rewinding.Off op vstar a p) g s ≤
      Real.sqrt (Fintype.card P * Q * (q ^ 2 / 2 ^ (513 : ℝ))) :=
  (Rewinding.expect_bad_le op g s vstar hv Q hQ).trans (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
    ((expect₂_conflict_le H g s op hcol).trans hCR.le) (by positivity)))

/-- **A rewinding at registered roots that covers an event** `bad` of the game `G` played by `S`. Runs of `g` against
`s` (the prover after its registration, with the unit draw in place of the proof coins) open positions `P` of the roots
to values (`op`), at most `Q` per run (`opens`), and `vstar` is the plurality of the openings (`plur`). Two runs'
conflicting openings are a collision of `H` (`binds`, Merkle binding), and `bad` is no likelier than a run opening a
position off `vstar` (`covers`). -/
structure TreeRewind {β : Type} (G : Game β) (S : Strategy G) (bad : β → Prop) : Type 1 where
  α : Type
  P : Type
  V : Type
  [fintypeP : Fintype P]
  [fintypeV : Fintype V]
  [decEqV : DecidableEq V]
  op : α → P → Option V
  g : Game α
  s : Strategy g
  vstar : P → V
  plur : Rewinding.IsPluralityE op g s vstar
  Q : ℕ
  opens : ∀ a, (Finset.univ.filter fun p => (op a p).isSome).card ≤ Q
  binds : ∀ x p, Rewinding.ConflictAt op x p → Merkle.Collision H
  covers : prob bad G S ≤ ENNReal.ofReal (expect (ind fun a => ∃ p, Rewinding.Off op vstar a p) g s)

namespace TreeRewind

attribute [local instance] fintypeP fintypeV decEqV

variable {H} {β : Type} {G : Game β} {S : Strategy G} {bad : β → Prop}

/-- The finder `δ_tree` counts: two runs after the registration, and a position they open to different values. -/
noncomputable def finder (tr : TreeRewind H G S bad) : Finder Pair :=
  twoRuns tr.g tr.s (conflictPick H tr.op tr.binds)

/-- **`δ_tree` at the budget `q`**: `√(N·Q·q²/2^513)`, for `N` the positions and `Q` the openings per run. -/
noncomputable def bound (tr : TreeRewind H G S bad) (q : ℝ) : ℝ :=
  Real.sqrt (Fintype.card tr.P * tr.Q * (q ^ 2 / 2 ^ (513 : ℝ)))

/-- **`δ_tree` under `SHA512CRStrict`**: if the rewinding's finder satisfies `SHA512CRStrict` at the budget `q`, the
covered event has probability at most `bound q` (`tree_le_strict`). -/
theorem le (tr : TreeRewind H G S bad) {q : ℝ} (h : tr.finder.CR H q) : prob bad G S ≤ ENNReal.ofReal (tr.bound q) :=
  tr.covers.trans (ENNReal.ofReal_le_ofReal
    (tree_le_strict H tr.op tr.g tr.s tr.vstar tr.plur tr.Q tr.opens tr.binds h))

/-- **A rewinding with no positions** covers an event that never happens. The skeleton's `hExec` (every executable
accept is the model's) gives one for `execAccept ∧ o.1 = false`, at `δ_tree = 0` (`ofNever_bound`). -/
def ofNever (h : ∀ b, ¬ bad b) : TreeRewind H G S bad where
  α := Unit
  P := Empty
  V := Unit
  op _ e := e.elim
  g := .ret ()
  s := PUnit.unit
  vstar e := e.elim
  plur e := e.elim
  Q := 0
  opens _ := by simp
  binds _ e := e.elim
  covers := (prob_mono (p := bad) (q := fun _ => False) h G S).trans (by rw [prob_false]; exact zero_le)

theorem ofNever_bound (h : ∀ b, ¬ bad b) (q : ℝ) : (ofNever h : TreeRewind H G S bad).bound q = 0 := by
  simp [bound, ofNever]

end TreeRewind

end Finders

/-! ## The compiled theorem -/

section Compiled

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

/-- **`Adv₀`'s finder**: two continuations after the prover's level-0 cap, and a level-0 position they open to different
rows. -/
noncomputable def adv₀Finder (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableC A H E S sch hm ptLocal mPts)) :
    Finder Pair :=
  twoRuns (tableCmp A H E S sch hm ptLocal mPts σ.1) σ.2
    (conflictPick H (op₀ H E sch σ.1) fun _ _ h => op₀_conflict_collision h)

/-- **`Adv_{r,ℓ}`'s finder**: along the run, at rep `r`'s level-`ℓ` cap, two continuations, and a level-`ℓ` position
they open to different rows. -/
noncomputable def advRFinder [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) : Finder Pair :=
  forkFinder (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (fun d => conflictPick H (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d) fun _ _ h => opR_conflict_collision h)
    none
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2

/-- **The self-clash finder**: one run, and a position it opens two ways. -/
noncomputable def selfFinder (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableC A H E S sch hm ptLocal mPts)) :
    Finder Pair :=
  oneRun (tableC A H E S sch hm ptLocal mPts) σ (onePick H (SelfClash H E sch) fun _ h => h.collision)

/-- **`SHA512CRStrict` for the compiled theorem's finders**: the rewinding finders at budget `qF`, the self-clash finder at
budget `qS`. -/
structure CompiledCR [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) (qF qS : ℝ) : Prop where
  fork : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
    (advRFinder A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ).CR H qF
  level0 : (adv₀Finder A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ).CR H qF
  self : (selfFinder A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ).CR H qS

omit [Fintype K] in
theorem adv₀_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableC A H E S sch hm ptLocal mPts))
    {q : ℝ} (h : (adv₀Finder A H E S sch hm ptLocal mPts σ).CR H q) :
    adv₀ A H E S sch hm ptLocal mPts σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect₂_conflict_le H (tableCmp A H E S sch hm ptLocal mPts σ.1) σ.2 (op₀ H E sch σ.1)
    fun _ _ h => op₀_conflict_collision h).trans h.le

theorem advR_le_strict [Inhabited D] (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0)
    (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (r ℓ : ℕ)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) {q : ℝ}
    (h : (advRFinder A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ).CR H q) :
    advR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (forkE_conflict_le H (plurC H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (slotOf r ℓ)
    (opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ) (fun _ _ _ h => opR_conflict_collision h)
    (lockTable A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts
      (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2) σ.1) σ.2).trans h.le

omit [Fintype K] in
theorem self_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (σ : Strategy (tableC A H E S sch hm ptLocal mPts))
    {q : ℝ} (h : (selfFinder A H E S sch hm ptLocal mPts σ).CR H q) :
    expect (ind (SelfClash H E sch)) (tableC A H E S sch hm ptLocal mPts) σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect_one_le H _ σ (SelfClash H E sch) fun _ h => h.collision).trans h.le

/-- **The compiled theorem under `SHA512CRStrict`** (`table_sound_compiled` at its finders). If the rewinding finders
each satisfy `SHA512CRStrict` at budget `qF` and the self-clash finder at `qS`, a compiled run accepts while the table
`P*` committed to is not a codeword of the statement with probability at most
$$\text{tableError}+\sqrt{2^{\log N_0}\cdot 2Q_0\cdot\frac{q_F^2}{2^{513}}}+\sum_{r<2}\sum_{\ell}
\sqrt{2^{\log N_\ell}Q_\ell\cdot\frac{q_F^2}{2^{513}}}+\frac{q_S^2}{2^{513}}.$$ -/
theorem table_sound_compiled_strict (A : Arith F K) (hA : A.Correct) [Inhabited D] (H : List UInt8 → D)
    (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ)
    (hsch : fast100 S.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (hlay : S.LinkLayout ptLocal mPts)
    (σ : Strategy (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts)) {qF qS : ℝ}
    (hCR : CompiledCR A H E S m k0 l₀ l₁ rest ext (fast100_shape S.m _ hsch).2.1 hm ptLocal mPts σ qF qS) :
    expect (ind fun out : OutC F D => out.accepted = true ∧
        ¬ Committed A S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
          (C0star A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts σ.1 σ.2))
      (tableC A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts) σ ≤
    tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
      Real.sqrt (2 ^ l₀.logLen * (2 * l₀.queries) * (qF ^ 2 / 2 ^ (513 : ℝ))) +
      ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
        Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
          (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
      qS ^ 2 / 2 ^ (513 : ℝ) :=
  (table_sound_compiled A hA H E S m k0 l₀ l₁ rest ext hsch hm ptLocal mPts hlay σ).trans
    (add_le_add (add_le_add (add_le_add le_rfl
      (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
        (adv₀_le_strict A H E S _ hm ptLocal mPts σ hCR.level0) (by positivity))))
      (Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ => Real.sqrt_le_sqrt
        (mul_le_mul_of_nonneg_left (advR_le_strict A H E S m k0 l₀ l₁ rest ext _ hm ptLocal mPts r ℓ σ
          (hCR.fork r hr ℓ hℓ)) (by positivity))))
      (self_le_strict A H E S _ hm ptLocal mPts σ hCR.self))

end Compiled

/-! ## The knowledge term -/

section Session

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}
  [Inhabited D] {mPts : ℕ}

/-- **`Adv_{r,ℓ}`'s finder in a session**: along the session, at the table's rep `r`'s level-`ℓ` cap, two continuations,
and a level-`ℓ` position they open to different rows. -/
noncomputable def advRSFinder (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D) (r ℓ : ℕ)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) :
    Finder Pair :=
  forkFinder (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
    (fun d => conflictPick H
      (fun outs : List (OutC F D) => opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOut))
      fun x _ h => opR_conflict_collision (x := (x.1.getD pre.length dfltOut, x.2.getD pre.length dfltOut)) h)
    none (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post default cap₀) σ

/-- **`Adv₀`'s finder in a session**: two runs after `Commit`, and a level-0 position of the table they open to different
rows. -/
noncomputable def adv₀SFinder (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D)))
    (cap₀ : Cap D) (σ : Strategy (sessAfter A H E S sch hm ptLocal mPts pre post cap₀)) : Finder Pair :=
  twoRuns (sessAfter A H E S sch hm ptLocal mPts pre post cap₀) σ
    (conflictPick H (fun outs : List (OutC F D) => op₀ H E sch cap₀ (outs.getD pre.length dfltOut))
      fun x _ h => op₀_conflict_collision (x := (x.1.getD pre.length dfltOut, x.2.getD pre.length dfltOut)) h)

/-- **The self-clash finder in a session**: one run after `Commit`, and a position of the table it opens two ways. -/
noncomputable def selfSFinder (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D)))
    (cap₀ : Cap D) (σ : Strategy (sessAfter A H E S sch hm ptLocal mPts pre post cap₀)) : Finder Pair :=
  oneRun (sessAfter A H E S sch hm ptLocal mPts pre post cap₀) σ
    (onePick H (fun outs : List (OutC F D) => SelfClash H E sch (outs.getD pre.length dfltOut))
      fun _ h => h.collision)

/-- **`SHA512CRStrict` for a table's finders in a session**: the rewinding finders at budget `qF`, the self-clash finder
at budget `qS`. -/
structure SessCR (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement) (m k0 : ℕ)
    (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ)
    (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀))
    (qF qS : ℝ) : Prop where
  fork : ∀ r ∈ Finset.range 2, ∀ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
    (advRSFinder A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ).CR H qF
  level0 : (adv₀SFinder A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ).CR H qF
  self : (selfSFinder A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀ σ).CR H qS

theorem advRS_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D) (r ℓ : ℕ)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) {q : ℝ}
    (h : (advRSFinder A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ).CR H q) :
    advRS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ r ℓ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (forkE_conflict_le H (plurS H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ pre.length) (slotOf r ℓ)
    (fun d (outs : List (OutC F D)) => opR H E ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ r ℓ d (outs.getD pre.length dfltOut))
    (fun _ x _ h => opR_conflict_collision (x := (x.1.getD pre.length dfltOut, x.2.getD pre.length dfltOut)) h)
    (lockSess A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post default cap₀) σ).trans h.le

omit [Fintype K] in
theorem adv₀S_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D)))
    (cap₀ : Cap D) (σ : Strategy (sessAfter A H E S sch hm ptLocal mPts pre post cap₀)) {q : ℝ}
    (h : (adv₀SFinder A H E S sch hm ptLocal mPts pre post cap₀ σ).CR H q) :
    adv₀S A H E S sch hm ptLocal mPts pre post cap₀ σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect₂_conflict_le H (sessAfter A H E S sch hm ptLocal mPts pre post cap₀) σ
    (fun outs : List (OutC F D) => op₀ H E sch cap₀ (outs.getD pre.length dfltOut))
    fun x _ h => op₀_conflict_collision (x := (x.1.getD pre.length dfltOut, x.2.getD pre.length dfltOut)) h).trans
    h.le

omit [Fintype K] in
theorem selfS_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (sch : Schedule) (hm : 13 ≤ S.m) (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D)))
    (cap₀ : Cap D) (σ : Strategy (sessAfter A H E S sch hm ptLocal mPts pre post cap₀)) {q : ℝ}
    (h : (selfSFinder A H E S sch hm ptLocal mPts pre post cap₀ σ).CR H q) :
    expect (ind fun outs : List (OutC F D) => SelfClash H E sch (outs.getD pre.length dfltOut))
      (sessAfter A H E S sch hm ptLocal mPts pre post cap₀) σ ≤ q ^ 2 / 2 ^ (513 : ℝ) :=
  (expect_one_le H _ σ (fun outs : List (OutC F D) => SelfClash H E sch (outs.getD pre.length dfltOut))
    fun _ h => h.collision).trans h.le

/-- **`ε_c⁻` in the session under `SHA512CRStrict`.** -/
theorem epsS_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (S : Statement)
    (m k0 : ℕ) (l₀ l₁ : Level) (rest : List Level) (ext : ℕ) (hk0 : l₀.lanes = k0) (hm : 13 ≤ S.m)
    (ptLocal mPts : ℕ) (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutC F D))) (cap₀ : Cap D)
    (σ : Strategy (sessAfter A H E S ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ hm ptLocal mPts pre post cap₀)) {qF qS : ℝ}
    (h : SessCR A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ σ qF qS) :
    epsS A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal mPts pre post cap₀ σ ≤
      tableError ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (S.shape mPts) +
        ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 (l₀ :: l₁ :: rest).length,
          Real.sqrt (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).logLen *
            (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
        qS ^ 2 / 2 ^ (513 : ℝ) :=
  add_le_add (add_le_add le_rfl (Finset.sum_le_sum fun r hr => Finset.sum_le_sum fun ℓ hℓ =>
      Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left (advRS_le_strict A H E S m k0 l₀ l₁ rest ext hk0 hm ptLocal
        mPts pre post cap₀ r ℓ σ (h.fork r hr ℓ hℓ)) (by positivity))))
    (selfS_le_strict A H E S _ hm ptLocal mPts pre post cap₀ σ h.self)

/-- **`SHA512CRStrict` for table `j`'s finders in a batched session**, with the prover's session strategy `τ`. -/
abbrev TableCR (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D) (ts : List (TabSpec mPts))
    (j : Fin ts.length) (τ : Strategy (sessionB A H E ts)) (qF qS : ℝ) : Prop :=
  SessCR A H E (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest (ts.get j).ext
    (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm (ts.get j).ptLocal mPts
    ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j) (stratB A H E ts j τ) qF qS

omit [Inhabited D] in
/-- **Table `j`'s knowledge bound under `SHA512CRStrict`**, `ε_c⁻ + √(k·adv₀)` at the finders' budgets: a function of the
table and the budgets alone. -/
noncomputable def ksStrict (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) : ℝ :=
  tableError (ts.get j).sch ((ts.get j).S.shape mPts) +
    ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 ((ts.get j).l₀ :: (ts.get j).l₁ :: (ts.get j).rest).length,
      Real.sqrt (2 ^ (lvOf (ts.get j).sch ℓ).logLen * (lvOf (ts.get j).sch ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    qS ^ 2 / 2 ^ (513 : ℝ) + Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

theorem ksBoundAccB_le_strict (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) (τ : Strategy (sessionB A H E ts)) {qF qS : ℝ}
    (h : TableCR A H E ts j τ qF qS) : ksBoundAccB A H E ts j k τ ≤ ksStrict ts j k qF qS :=
  add_le_add
    (epsS_le_strict A H E (ts.get j).S (ts.get j).m (ts.get j).k0 (ts.get j).l₀ (ts.get j).l₁ (ts.get j).rest
      (ts.get j).ext (fast100_shape (ts.get j).S.m _ (ts.get j).hsch).2.1 (ts.get j).hm (ts.get j).ptLocal mPts
      ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j) (stratB A H E ts j τ) h)
    (Real.sqrt_le_sqrt (mul_le_mul_of_nonneg_left
      (adv₀S_le_strict A H E (ts.get j).S (ts.get j).sch (ts.get j).hm (ts.get j).ptLocal mPts
        ((gamesB A H E ts τ.1).take j) ((gamesB A H E ts τ.1).drop (j + 1)) (τ.1 j) (stratB A H E ts j τ) h.level0)
      (Nat.cast_nonneg k)))

omit [Inhabited D] in
/-- **The knowledge term under `SHA512CRStrict`**: after each draw, the largest over the plan's tables of
`ksStrict/(1 − N₀/(ek))`, averaged over the draw. It reads no prover: the budgets stand in for its finders. -/
noncomputable def ksAvgStrict {n : ℕ} {Reg : Type} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (L : Audit.Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) : ℝ≥0∞ :=
  avg fun ω : L.Ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksStrict (plan (L.draw ω) R) j k qF qS / (1 - rateB (plan (L.draw ω) R) j k))

end Session

/-- **The knowledge term with the expected-time extractor, under `SHA512CRStrict`**: `ksAvgBE` is at most `ksAvgStrict`
at a prover each of whose tables' finders, after each draw, satisfies `SHA512CRStrict` at budgets `qF`, `qS`
(`TableCR`). -/
theorem ksAvgBE_le_strict {n : ℕ} {Reg D : Type} [Inhabited D] (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    {mPts : ℕ} (plan : Finset (Fin n) → Reg → List (TabSpec mPts)) {L : Audit.Law n} (k : ℕ) {qF qS : ℝ} (R : Reg)
    (τ : Audit.Cont L Reg (Audit.Partition.batchedSession H E plan) R)
    (hCR : ∀ ω j, TableCR execArith H E (plan (L.draw ω) R) j (Audit.Partition.batchStrat H E plan (τ ω)) qF qS) :
    Audit.Partition.ksAvgBE H E plan k R τ ≤ ksAvgStrict plan L k qF qS R := by
  refine avg_mono fun ω => iSup_mono fun j => ?_
  have ha := ksBoundAccB_nonneg execArith H E (plan (L.draw ω) R) j k (Audit.Partition.batchStrat H E plan (τ ω))
  rcases le_or_gt (1 - rateB (plan (L.draw ω) R) j k) 0 with h0 | h0
  · exact (le_of_eq (ENNReal.ofReal_of_nonpos (div_nonpos_of_nonneg_of_nonpos ha h0))).trans zero_le
  · exact ENNReal.ofReal_le_ofReal (div_le_div_of_nonneg_right
      (ksBoundAccB_le_strict execArith H E (plan (L.draw ω) R) j k _ (hCR ω j)) h0.le)

end FlockSoundness.StrictCR
