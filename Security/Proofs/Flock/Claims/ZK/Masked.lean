import Proofs.Flock.Claims.ZK.Session
import Proofs.Flock.Claims.ZK.MaskedTables

/-!
# The deployed masked session's zero knowledge

`Flock.zk_deployZk_ref_on_rounds` at the deployed masked session (`FlockDeploy.setupOf true`): an honest-verifier
zero-knowledge session whose leaves are hiding salted commitments and whose coins `Hello` commits (Goldreich–Kahan),
on the statements `Masked.Live` (the masked scope's, at a witness: `live_iff`). Its table data (`tableData`) is
concrete where the tree fixes it: the tables (`jOf`, `tabC`: `setupOf true x`'s, each `coinedOf` at its `--zk` table
from its statement alone, `ZkHidden.ZkTab.table`), the witness's table witnesses (`wtOf`, each table's share) and the
dummy's (`dummy`, a witness of the regions). The rest is two plans:

* `RoundPlan`: each table's coins at the session's coins (`cz`, `sessionPZk`'s per-table derivation) and the round
  encodings (`enc`, `fin`, `off`: `sessionPZk`'s ask messages and `Final.encode`);
* `TapePlan`: the finite tape, read as the masked prover's (`tapeZk`) and as the tables' randomness (`tape`).

The session's Props at every plan:

* `Tables` (`tables`), `DummyReg` (`dummyReg`) and `Complete` on the masked scope (`complete`) are theorems
  (`MaskedTables`).
* `Causal` is the round plan's (`RoundPlan.Causal`): the cut view round `i` encodes is at the tables' coins `cz x c`,
  whose type reads every coin, so no typing of `enc` makes it hold.
* `TapeOK` is `FibersEq (U.tape x)` on `Live` statements (`Flock.zk_deployZk_masked`, a Prop of the tape plan), or, on
  the tape of one statement's tables' randomness (`TapePlan.at`), a theorem (`fibersEq_at`,
  `Flock.zk_deployZk_masked_at`): what statement-indexed tapes read at that statement.
* `HonestMsgs` is its three rounds (`HonestPreOn`, `HonestFinOn`, `HonestOffOn`) on `Live`.
-/

namespace Claims.ZKMal

open Claims FlockSoundness.Merkle FlockSoundness.ZK
open FlockSoundness.Discharge
open CoinLayer

set_option synthInstance.maxSize 100000
set_option synthInstance.maxHeartbeats 1000000

/-- **`Claims.Flock.ZK` at a larger `ε_zk`**, larger at the statements with a witness. -/
theorem zk_mono {Coin Col T : Type} {H : Msg → Msg} {D : Claims.Flock.ZkDeploy Coin Col T} {ε ε' γ : Msg → ℝ}
    (h : Claims.Flock.ZK H D ε γ) (hle : ∀ x w, Claims.Flock.R D.parse x w → ε x ≤ ε' x) :
    Claims.Flock.ZK H D ε' γ :=
  ⟨h.1, fun x w hR Θ Vs θ hk Ev => (h.2 x w hR Θ Vs θ hk Ev).trans (by linarith [hle x w hR])⟩

/-- **`ε_zk` on `A` and `1` outside it, as `ε_zk` on `B`**, where the statements with a witness are in `A` exactly
when they are in `B`. -/
theorem zk_if_congr {Coin Col T : Type} {H : Msg → Msg} {D : Claims.Flock.ZkDeploy Coin Col T} {A B : Msg → Prop}
    [DecidablePred A] [DecidablePred B] {E E' γ : Msg → ℝ} (h : Claims.Flock.ZK H D (fun x => if A x then E x else 1) γ)
    (hAB : ∀ x w, Claims.Flock.R D.parse x w → (A x ↔ B x)) (hE : ∀ x, E x ≤ E' x) :
    Claims.Flock.ZK H D (fun x => if B x then E' x else 1) γ :=
  zk_mono h fun x w hR => by
    have e := hAB x w hR
    split_ifs with h1 h2 h2
    · exact hE x
    · exact absurd (e.1 h1) h2
    · exact absurd (e.2 h2) h1
    · exact le_rfl

namespace Masked

/-- **A family of tables**: at each statement, the masked session's table count of tables. -/
abbrev Tabs : Type 1 := (x : Msg) → Fin (jOf x) → Composed.CoinedSetup

/-- **The tables' randomness at `x`**: each table's base, every pre-final leaf's salt and its ideal seed. -/
abbrev Rand (C : Tabs) (x : Msg) : Type :=
  (∀ j, ZkView.Base (C x j).z (C x j).Hi) × (ZkView.Leaves (fun j => (C x j).z) → ZkView.Salt) ×
    (ZkView.Leaves (fun j => (C x j).z) → ZkView.Bits × ZkView.Salt)

/-- The session's pre-final leaves at `x`. -/
noncomputable def nL (P : Composed.LigProver) (x : Msg) : ℕ := Fintype.card (ZkView.Leaves fun j => (tabC P x j).z)

/-- **The round plan** at the tables `C`: each table's coins at the session's coins, and the round encodings of the cut
view, of the tables' views where they admit the coins, and of the cut view where one refuses. -/
structure RoundPlan (C : Tabs) where
  cz : (x : Msg) → (Fin (refK x) → FlockDeploy.Coin) → ∀ j, Composed.CoinsZK (C x j).Rest
  enc : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → ℕ → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg
  fin : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((C x j).T (cz x c j)).tab.View) → Msg
  off : (x : Msg) → (c : Fin (refK x) → FlockDeploy.Coin) → (∀ j, ((C x j).T (cz x c j)).tab.Cut) → Msg

/-- **The tape plan** at the tables `C`: a finite tape, read as the masked prover's (`tapeZk`) and as the tables'
randomness (`tape`). -/
structure TapePlan (C : Tabs) where
  Tape : Type
  [finTape : Fintype Tape]
  [neTape : Nonempty Tape]
  tapeZk : Tape → FlockComplete.TapeZk
  tape : (x : Msg) → Tape → Rand C x

attribute [instance] TapePlan.finTape TapePlan.neTape

/-- **The deployed masked session's table data** at the Ligerito prover `P` and the plans `Q` and `U`: the tables
`tabC P`, the witness's table witnesses its shares (`wtOf`), the dummy's a witness of the regions (`dummy`). -/
noncomputable def tableData (P : Composed.LigProver) (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) : TableData where
  Tape := U.Tape
  tapeZk := U.tapeZk
  J := jOf
  C := tabC P
  cz := Q.cz
  wt := wtOf P
  wt₀ x j := dummy (tabC P x j).S
  enc := Q.enc
  fin := Q.fin
  off := Q.off
  tape := U.tape

/-- **`Tables`**: every table of the masked session is a table it sets up (`setupTable_tabC`). -/
theorem tables (P : Composed.LigProver) (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) (nr : Msg → ℕ → ℕ) :
    (session nr (tableData P Q U)).Tables :=
  fun x j => setupTable_tabC P x j

/-- **`DummyReg`**: the dummy's table witnesses carry the public region data (`dummyReg_tabC`). -/
theorem dummyReg (P : Composed.LigProver) (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) (nr : Msg → ℕ → ℕ) :
    (session nr (tableData P Q U)).DummyReg :=
  fun x c j => dummyReg_tabC P x j (Q.cz x c j)

/-- **`Complete`** on any class of statements, at the deployed relation: each table's share is valid
(`complete_tabC`). -/
theorem complete (P : Composed.LigProver) (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) (nr : Msg → ℕ → ℕ)
    (A : Msg → Prop) :
    (session nr (tableData P Q U)).Complete fun x w =>
      A x ∧ Claims.Flock.R (FlockDeploy.deployZk _root_.Flock.Readable.relation).parse x w :=
  fun _ _ h j => complete_tabC P h.2 j

/-- **Round `i`'s encoding reads the coins before `i` only** (`RefSession.Causal` at the round plan), whatever each
pre-final leaf's pair `(M·y, c(y))`. -/
def RoundPlan.Causal {C : Tabs} (Q : RoundPlan C) : Prop :=
  ∀ (x : Msg) (wt : ∀ j, FlockSoundness.Model.Witness (C x j).S.m) (ω : ∀ j, ZkView.Base (C x j).z (C x j).Hi)
    (p : ZkView.Leaves (fun j => (C x j).z) → ZkView.Bits × ByteArray) (c c' : Fin (refK x) → FlockDeploy.Coin)
    (i : ℕ), (∀ j : Fin (refK x), j.val < i → c j = c' j) →
    Q.enc x c i (cutAt (C x) (Q.cz x c) wt ω fun l =>
        ZkView.leafH (digOf (C x) (Q.cz x c) wt ω l + (p l).1, (p l).2)) =
      Q.enc x c' i (cutAt (C x) (Q.cz x c') wt ω fun l =>
        ZkView.leafH (digOf (C x) (Q.cz x c') wt ω l + (p l).1, (p l).2))

theorem RoundPlan.causal {P : Composed.LigProver} {Q : RoundPlan (tabC P)} (h : Q.Causal) (U : TapePlan (tabC P))
    (nr : Msg → ℕ → ℕ) : (session nr (tableData P Q U)).Causal :=
  h

open Classical in
/-- **The tape of one statement's tables' randomness**: at `x₀` the tape is `x₀`'s tables' randomness, read as itself;
elsewhere it reads as one fixed randomness. `z` reads it as the masked prover's. -/
noncomputable def TapePlan.at {C : Tabs} (x₀ : Msg) (z : Rand C x₀ → FlockComplete.TapeZk) : TapePlan C where
  Tape := Rand C x₀
  tapeZk := z
  tape x r := if h : x₀ = x then h ▸ r else Classical.arbitrary _

/-- **At its statement, the tape is read with fibers of one size**: as itself. -/
theorem fibersEq_at {C : Tabs} (x₀ : Msg) (z : Rand C x₀ → FlockComplete.TapeZk) :
    FibersEq ((TapePlan.at x₀ z).tape x₀) := by
  have e : (TapePlan.at x₀ z).tape x₀ = id := funext fun r => dite_eq_left rfl
  rw [e]
  exact fibersEq_of_bijective Function.bijective_id

end Masked

open Masked

open Classical in
/-- **Flock's zero knowledge at the deployed masked session**, today's form: `Claims.Flock.ZK` at `zkDeployed` at the
masked session's table data, at `ε_zk = 8·n_L·δ₁ + 8t·n_L·δ₁ + 2·3^{-t}` on the masked scope and `1` outside it, under
`Claims.Hm96Hiding` at the pinned key, the round plan's `Causal`, the tape plan read with fibers of one size at the
`Live` statements, and `refP`'s three rounds `ZkTables.honestP`'s there. `Tables`, `DummyReg` and `Complete` are
theorems (`Masked.tables`, `Masked.dummyReg`, `Masked.complete`). -/
theorem Flock.zk_deployZk_masked (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (U : TapePlan (tabC P)) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal} (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hδ₁ : δ₁ ≠ ⊤) (hc : Q.Causal) (hT : ∀ x, Live x → FibersEq (U.tape x))
    (hpre : (session nr (tableData P Q U)).HonestPreOn Live H coinOfD fun ω => U.tapeZk ω.2)
    (hfin : (session nr (tableData P Q U)).HonestFinOn Live H coinOfD fun ω => U.tapeZk ω.2)
    (hoff : (session nr (tableData P Q U)).HonestOffOn Live H coinOfD fun ω => U.tapeZk ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr (tableData P Q U) t)
      (fun x => if Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * δ₁.toReal + 8 * t * nL P x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t := by
  exact zk_if_congr (Flock.zk_deployZk_ref_on_rounds H _root_.Flock.Readable.relation nr (tableData P Q U) Live t ht
    hT1 hδ₁ (tables P Q U nr) (complete P Q U nr Live) (dummyReg P Q U nr) (RoundPlan.causal hc U nr) hT hpre hfin
    hoff) (fun _ _ hR => live_iff hR) fun _ => le_rfl

open Classical in
/-- **Flock's zero knowledge at the deployed masked session, at one statement's tape**: at `x₀`, the tape `x₀`'s
tables' randomness (`TapePlan.at`, read with fibers of one size: `fibersEq_at`), `Claims.Flock.ZK` at `ε_zk` at `x₀`
if it is in the masked scope and `1` elsewhere, under `Claims.Hm96Hiding` at the pinned key, the round plan's `Causal`,
and `refP`'s three rounds `ZkTables.honestP`'s at `x₀`. Statement-indexed tapes read the claim at `x₀` at this
record. -/
theorem Flock.zk_deployZk_masked_at (H : Msg → Msg) (nr : Msg → ℕ → ℕ) (P : Composed.LigProver)
    (Q : RoundPlan (tabC P)) (x₀ : Msg) (z : Rand (tabC P) x₀ → FlockComplete.TapeZk) (t : ℕ) (ht : 0 < t) {δ₁ : ENNReal}
    (hT1 : Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁) (hδ₁ : δ₁ ≠ ⊤) (hc : Q.Causal)
    (hpre : (session nr (tableData P Q (TapePlan.at x₀ z))).HonestPreOn (fun x => x = x₀ ∧ Live x) H coinOfD
      fun ω => z ω.2)
    (hfin : (session nr (tableData P Q (TapePlan.at x₀ z))).HonestFinOn (fun x => x = x₀ ∧ Live x) H coinOfD
      fun ω => z ω.2)
    (hoff : (session nr (tableData P Q (TapePlan.at x₀ z))).HonestOffOn (fun x => x = x₀ ∧ Live x) H coinOfD
      fun ω => z ω.2) :
    Claims.Flock.ZK H (zkDeployed _root_.Flock.Readable.relation nr (tableData P Q (TapePlan.at x₀ z)) t)
      (fun x => if x = x₀ ∧ Claims.Flock.Scope (FlockDeploy.deployZk _root_.Flock.Readable.relation) x then
        8 * nL P x * δ₁.toReal + 8 * t * nL P x * δ₁.toReal + 2 * (1 / 3) ^ t else 1)
      fun _ => 4 * t * 3 ^ t := by
  exact zk_if_congr (Flock.zk_deployZk_ref_on_rounds H _root_.Flock.Readable.relation nr
    (tableData P Q (TapePlan.at x₀ z)) (fun x => x = x₀ ∧ Live x) t ht hT1 hδ₁ (tables P Q _ nr) (complete P Q _ nr _)
    (dummyReg P Q _ nr) (RoundPlan.causal hc _ nr) (by rintro x ⟨rfl, -⟩; exact fibersEq_at x z) hpre hfin hoff)
    (fun _ _ hR => and_congr_right fun _ => live_iff hR) fun _ => le_rfl

end Claims.ZKMal
