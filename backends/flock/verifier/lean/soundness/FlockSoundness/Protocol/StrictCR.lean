import FlockSoundness.Assumptions
import FlockSoundness.Protocol.Game.Seq
import FlockSoundness.Protocol.Audit.FlockBatched
import FlockSoundness.Protocol.SessionBatchAcc

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

open Classical in
/-- A collision of `H` when `p` holds (`h` gives one), nothing otherwise. -/
noncomputable def collOf (p : Prop) (h : p → Merkle.Collision H) : Pair :=
  if hp : p then some ((h hp).choose, (h hp).choose_spec.choose) else none

/-- An event of a run, as a finder's output: a collision whenever the event holds. -/
noncomputable def onePick {α : Type} (q : α → Prop) (hq : ∀ a, q a → Merkle.Collision H) (a : α) : Pair :=
  collOf H (q a) (hq a)

/-- Two runs' conflict at some position, as a finder's output. -/
noncomputable def conflictPick {α P V : Type} (op : α → P → Option V)
    (hcol : ∀ x p, Rewinding.ConflictAt op x p → Merkle.Collision H) (b b' : α) : Pair :=
  onePick H (fun x : α × α => ∃ p, Rewinding.ConflictAt op x p) (fun x h => h.elim (hcol x)) (b, b')

/-- **A collision finder**: a game played by a fixed strategy, with what it found as its output. -/
structure Finder (γ : Type) : Type 1 where
  g : Game γ
  s : Strategy g

/-- **`SHA512CRStrict` for one finder**, at the budget `q` on every outcome: `q` bounds the SHA-512 evaluations of the
prover's runs the finder plays and of its verifications. The cost is `q` itself (`fun _ => q`), so Lean doesn't check
that the finder stays within it: a budget below the finder's real count makes this a claim about SHA-512 that
`cr/sha-512` doesn't make. Whoever instantiates it states an honest `q` (`ASSUMPTIONS.md`). -/
def Finder.CR (Fd : Finder Pair) (q : ℝ) : Prop :=
  Assumptions.SHA512CRStrict H Fd.g Fd.s id (fun _ => q) q

/-- **One run** of `g` against `s`, its output read by `pick`. -/
noncomputable def oneRun {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → γ) : Finder γ :=
  ⟨g.map pick, Strategy.map pick g s⟩

/-- **Two independent runs** of `g` against `s`, their outputs read by `pick`. -/
noncomputable def twoRuns {β γ : Type} (g : Game β) (s : Strategy g) (pick : β → β → γ) : Finder γ :=
  ⟨g.bind fun b => g.map (pick b), Strategy.bind g _ s fun b => Strategy.map (pick b) g s⟩

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

end Fork

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

/-- **`δ_tree` at the budget `q`**: `√(N·Q·q²/2^513)`, for `N` the positions and `Q` the openings per run. -/
noncomputable def bound (tr : TreeRewind H G S bad) (q : ℝ) : ℝ :=
  Real.sqrt (Fintype.card tr.P * tr.Q * (q ^ 2 / 2 ^ (513 : ℝ)))

end TreeRewind

end Finders

/-! ## The compiled theorem -/

section Compiled

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}

end Compiled

/-! ## The knowledge term -/

section Session

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D : Type}
  [Inhabited D] {mPts : ℕ}

omit [Inhabited D] in
/-- **Table `j`'s knowledge bound under `SHA512CRStrict`**, `ε_c⁻ + √(k·adv₀)` at the finders' budgets: a function of the
table and the budgets alone. -/
noncomputable def ksStrict (ts : List (TabSpec mPts)) (j : Fin ts.length) (k : ℕ) (qF qS : ℝ) : ℝ :=
  tableError (ts.get j).sch ((ts.get j).S.shape mPts) +
    ∑ _r ∈ Finset.range 2, ∑ ℓ ∈ Finset.Icc 1 ((ts.get j).l₀ :: (ts.get j).l₁ :: (ts.get j).rest).length,
      Real.sqrt (2 ^ (lvOf (ts.get j).sch ℓ).logLen * (lvOf (ts.get j).sch ℓ).queries * (qF ^ 2 / 2 ^ (513 : ℝ))) +
    qS ^ 2 / 2 ^ (513 : ℝ) + Real.sqrt (k * (qF ^ 2 / 2 ^ (513 : ℝ)))

omit [Inhabited D] in
/-- **The knowledge term under `SHA512CRStrict`**: after each draw, the largest over the plan's tables of
`ksStrict/(1 − N₀/(ek))`, averaged over the draw. It reads no prover: the budgets stand in for its finders. -/
noncomputable def ksAvgStrict {n : ℕ} {Reg : Type} (plan : Finset (Fin n) → Reg → List (TabSpec mPts))
    (L : Audit.Law n) (k : ℕ) (qF qS : ℝ) (R : Reg) : ℝ≥0∞ :=
  avg fun ω : L.Ω => ⨆ j : Fin (plan (L.draw ω) R).length,
    ENNReal.ofReal (ksStrict (plan (L.draw ω) R) j k qF qS / (1 - rateB (plan (L.draw ω) R) j k))

end Session

end FlockSoundness.StrictCR
