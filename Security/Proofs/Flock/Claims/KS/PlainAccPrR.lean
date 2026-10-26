import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainRdRunK

/-!
# The acceptance bound at `VR rd` against the reader's read sets

`PlainRdRunK`'s `plain_accPr_le_rdAllK` over the deployed recursive verifier `VR rd`: at `nextR rd`'s columns
(`colsRV rd`) and its level columns (`plainRawLevelColsRV rd`, read off the run `runOfRV rd` that `decR` decides on).

* `runOfRV rd v j`: `runOfR`'s run, table `j`'s, on a view (`colsRV`'s reading), and `colsRV_runOfRV`: `colsRV` is its
  level-0 columns.
* `plainRawLevelColsRV rd v k ℓ`: `plainRawLevelCols` at that run, rep `k`'s level-`ℓ` columns of table 0.
* `PlainAccPrRdR rd x`: `plain_accPr_le_rdAllK`'s statement at `VR rd`, with `GoodTable` at `deployPlain` (it reads
  neither `V` nor `cols`, so it is the same `GoodTable` at any record built on `deployPlain` with `V := VR rd`).
-/

namespace FlockDeploy

open Flock
open RecDeploy (outCoin)

/-- **The run `colsRV rd` reads on a view**: at the statement of the view's last message (`rd`), its messages and the
coins its answers show (`outCoin`), `runOfR`'s run, table `j`'s, at what `decR` decides on. -/
noncomputable def runOfRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (j : ℕ) : Option Claims.Flock.ZkRun :=
  match ((v.msgs.map Prod.fst).getLast?).bind rd with
  | some fl =>
    match decR rd fl.stmt (v.msgs.map Prod.fst) ((v.msgs.map Prod.snd).map outCoin) with
    | some d => runOfDecR d j
    | none => none
  | none => none

theorem colsRV_runOfRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (j : ℕ) :
    colsRV rd v j = colsOf (runOfRV rd v j) := by
  unfold colsRV runOfRV
  cases ((v.msgs.map Prod.fst).getLast?).bind rd with
  | none => rfl
  | some fl => cases decR rd fl.stmt (v.msgs.map Prod.fst) ((v.msgs.map Prod.snd).map outCoin) <;> rfl

end FlockDeploy

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge
  FlockSoundness.Discharge.Exec FlockSoundness.Discharge.Tables
open Claims.Bridge.CustodyPlain Claims.Bridge.DecodePlain Claims.Bridge.Custody Claims.Bridge.PlainDraw
open Claims.Bridge.Live
open Lean (Json)

/-- **Rep `k`'s level-`ℓ` columns of table 0 at `VR rd`, on a view**: `plainRawLevelCols` at the run `runOfRV rd`
reads. -/
noncomputable def plainRawLevelColsRV (rd : Claims.Msg → Option FinalP) (v : Claims.Seen) (k ℓ : ℕ) :
    List (ℕ × List 𝔽) :=
  match runOfRV rd v 0 with
  | some r =>
    match r.reps[k]? with
    | some y => plainLevelColsOf r.plainShape y.1 y.2 ℓ
    | none => []
  | none => []

/-- **An accepting run at `VR rd` with a collision among the verifier's calls** (`plainEv₁K`'s). -/
def plainEv₁R (rd : Claims.Msg → Option FinalP) {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (s : Fin ((V false).k x) → Coin) : Prop :=
  (Claims.run sha512 P (VR rd) x ω s).out = .acc ∧ Claims.TiedColl sha512 (Claims.run sha512 P (VR rd) x ω s).vLog

open Classical in
/-- **`plain_accPr_le_rdAllK`'s statement at `VR rd`**: at the plain table's setup and schedule, at any prover, tape
and committed word `C` off the good tables, and any read sets `rdD` holding every level-`ℓ + 1` row an accepting run
opens at a slot (`plainRawLevelColsRV`), with `w` a guess fixed by the coins before the slot's fork, the acceptance
probability is at most `ε_table`, the collision run's, the read sets' spread and the off-`C` opening's at level 0
(`colsRV rd`). -/
def PlainAccPrRdR (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) : Prop :=
  ∃ (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (m k0 : ℕ)
      (l₀ l₁ : Accounting.Level) (rest : List Accounting.Level) (ext : ℕ),
    setupOf false x = some (ins, spec, sts) ∧ sts[0]? = some st ∧ fast100 st.m = some sch ∧
    schedOf sch = ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧ tables false x = [(spec, st)] ∧
    Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ ∧
    ∀ {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Claims.Prover Ω) (ω : Ω) (C : Fin (wordLen false x) → List 𝔽),
      (∀ p, wide false x (C p)) → ¬ Claims.Flock.GoodTable (deployPlain Readable.relation) (wordLen false) x C →
      ∀ (rdD : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
          Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
            Finset (Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽))
        (w : (r : Fin 2) → (ℓ : Fin (l₀ :: l₁ :: rest).length) → (Fin ((V false).k x) → Coin) →
          Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).logLen) →
            Fin (2 ^ (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes) → 𝔽),
      (∀ r ℓ (c c' : Fin ((V false).k x) → Coin),
        (∀ i : Fin ((V false).k x), i.val < forkRound spec st k0 r.val ℓ.val → c i = c' i) →
          w r ℓ c = w r ℓ c') →
      (∀ r ℓ (c : Fin ((V false).k x) → Coin), (Claims.run sha512 P (VR rd) x ω c).out = .acc →
        ∀ q a, (q.val, a) ∈ plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen r.val (ℓ.val + 1) →
          rowFin (lvOf ⟨m, k0, l₀ :: l₁ :: rest, ext⟩ (ℓ.val + 1)).lanes a ∈ rdD r ℓ c q) →
      Claims.accPr sha512 (VR rd) P x ω ≤ εtable false x +
        (Claims.Pr fun s : Fin ((V false).k x) → Coin => plainEv₁R rd P x ω s) +
        ∑ r : Fin 2, ∑ ℓ : Fin (l₀ :: l₁ :: rest).length, ∑ q,
          Claims.Pr (fun c : Fin ((V false).k x) → Coin => (rdD r ℓ c q).Nonempty ∧ rdD r ℓ c q ≠ {w r ℓ c q}) +
        Claims.Pr fun r : Fin ((V false).k x) → Coin => (Claims.run sha512 P (VR rd) x ω r).out = .acc ∧
          ∃ (p : Fin (wordLen false x)) (u : List 𝔽),
            Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) p.val = some u ∧
              u ≠ C p

end FlockDeploy.KS
