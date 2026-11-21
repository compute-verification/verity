module

public import Proofs.Flock.Soundness.Discharge.HiddenExec.HeadlineExec
public import Proofs.Flock.Soundness.Discharge.HiddenExec.KeyH
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Setup
public import Proofs.Flock.Soundness.Discharge.HiddenExec.Sites
public import Proofs.Flock.Soundness.Discharge.ZkBind.At
public import Proofs.Flock.Soundness.Discharge.ZkBind.BindX
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Count
public import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt

@[expose] public section

set_option Elab.async false

/-!
# The sites, the layout and the rows at the hidden-output `--zk` tables

lp-zkbind's `sitesZAt`, `layZAt`, `rsZAt` and `rsZAt_computes` (`ZkBind/At.lean`) at the hidden-output tables
`tablesAtZKH dj y₀`, as HiddenExec's `sitesAtH`, `layAtH`, `rsAtH` and `rsAtH_computes` are at the public path's
`tablesAtH`:

* `tableShH`: the public hidden-output table (`HiddenExec.TableH`) at the same sample: the `SampleSetupZKH`'s
  `SampleSetupH` at `zkInputs I` and its `table`, or `y₀`'s refused one. Its statement is the `--zk` table's
  (`tableAtZKH_S`, both `stmtOf` of setupHidden's statement at the same regions).
* `sitesShH`: HiddenExec's sites at `tableShH`.
* `placeZH`: each sampled proof block's place in its `--zk` table, `keyPlaceH`'s along `tableAtZKH_S`.
-/

namespace FlockSoundness.Discharge.ZkHidden

open Flock FlockLevel3 Game Model Accounting Refine Spot.Sampling ZkSession ZkLink ZkExec ZkBind HiddenExec Placed
open FlockSoundness.Discharge.Exec (Inputs)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## A unit's place along an equality of statements -/

/-- A unit's place in `T`, as one in `T' = T`. -/
def placeCast {C : Verity.Boolean.Circuit} {n : ℕ} {P : Spot.Sampling.Partition C n} {T' T : Statement} {u : Fin n} (e : T' = T)
    (p : UnitPlace P T u) : UnitPlace P T' u := by
  subst e
  exact p

theorem decode_placeCast {C : Verity.Boolean.Circuit} {n : ℕ} {P : Spot.Sampling.Partition C n} {T' T : Statement} {u : Fin n}
    (e : T' = T) (p : UnitPlace P T u) (z : Witness T'.m) : (placeCast e p).decode z = p.decode (witOf e z) := by
  subst e
  rfl

/-! ## The public hidden-output table at each `--zk` table's sample -/

variable {n : ℕ} {I : Inputs} {mPts : ℕ} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : SampleSetupZKH I mPts S₀ (dj S₀)) {Reg : Type}

open Classical in
/-- **The public hidden-output table at the `--zk` table's sample**: the sample's `SampleSetupZKH`'s table, or `y₀`'s
refused one. -/
noncomputable def tableShH (S : Finset (Fin n)) : TableH (zkInputs I) mPts S :=
  if h : Nonempty (SampleSetupZKH I mPts S (dj S)) then (Classical.choice h).x.table else y₀.x.refused S

/-- **The `--zk` table's statement is the public table's**, at every sample: both are `stmtOf` of the same setupHidden
statement at the same regions. -/
theorem tableAtZKH_S (S : Finset (Fin n)) : (tableAtZKH dj y₀ S).S = (tableShH dj y₀ S).T.S := by
  unfold tableAtZKH tableShH
  split_ifs <;> rfl

/-- The public plan at the same samples. -/
noncomputable abbrev planShH : Finset (Fin n) → Reg → List (TabSpec mPts) := fun S _ => [(tableShH dj y₀ S).T]

abbrev tabShH : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planShH (Reg := Reg) dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

theorem planAtZKH_S : ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n),
    (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S =
      (Partition.tabOf (planShH dj y₀) (tabShH dj y₀) S R u).S :=
  fun S _ _ => tableAtZKH_S dj y₀ S

variable (ht : I.tags.typed = false) {c : Flock.Circuit} (hc : HmOut.parse (zkTags I) I.circuitFile I.tables = .ok c)
  (hscope : Layout.scopeOkH c = true)

/-- **The public plan's sites** (`sitesAtH`'s, at `tableShH`). -/
noncomputable def sitesShH : SitesH (planShH (Reg := Reg) dj y₀) (tabShH dj y₀) where
  A S _ _ _ := (tableShH dj y₀ S).twinA
  B S _ _ _ := (tableShH dj y₀ S).twinB
  sat S _ _ _ := (tableShH dj y₀ S).sat (zkTyped ht) hc
  pin S _ _ _ := (tableShH dj y₀ S).pin (zkTyped ht) hc
  acc S _ _ _ := (tableShH dj y₀ S).acc (zkTyped ht) hc (rowsFit_of_scopeH (zkTyped ht) hc hscope)
  vu S _ u h := (tableShH dj y₀ S).vu u h
  hvu S _ u h := (tableShH dj y₀ S).hvu u h
  o S _ u h := (tableShH dj y₀ S).o u h

/-- **One circuit per session**: every site's statement circuit is the session's one circuit `c`. -/
theorem sitesShH_parse :
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S),
      HmOut.unrolled ((sitesShH dj y₀ ht hc hscope).acc S R u h).st.c = c :=
  fun S _ _ _ => (tableShH dj y₀ S).acc_unrolled (zkTyped ht) hc (rowsFit_of_scopeH (zkTyped ht) hc hscope)

variable (hU : Built c.unit) {pub : Public} (hpub : HmOut.loadPublic (zkTags I) c I.publicFile = .ok pub)
  (hpt : pub.tables = none)

/-! ## The `--zk` places, layout and rows -/

/-- **Each sampled proof block's place in its hidden-output `--zk` table**: `keyPlaceH`'s place in the public table, along
`planAtZKH_S`. -/
noncomputable def placeZH (S : Finset (Fin n)) (R : Reg) (u : Fin n) (h : u ∈ S) :
    UnitPlace ((keyProg c hU n).partition (Layout.outsOf c hU n))
      (tabOfZ (planAtZKH dj y₀) (tabAtZKH dj y₀) S R u).S u :=
  placeCast (planAtZKH_S dj y₀ S R u)
    (keyPlaceH (sitesShH_parse dj y₀ ht hc hscope) (Layout.outsOf c hU n) S R u h)

end FlockSoundness.Discharge.ZkHidden
