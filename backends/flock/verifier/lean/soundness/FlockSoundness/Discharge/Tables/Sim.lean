import FlockSoundness.Discharge.Tables.Steps
import FlockSoundness.Protocol.Discharge.Tables.Sim

/-!
# The simulation of a session of `J` tables (the transfer, R11c, at `--session-tables J`)

flock-circuit's `--session-tables J` (`session_tables`, `config_with`) and flock-verify's `Stmt.setupTables` run one
session of `J` tables: one `Commit` with every table's root, one link exchange whose two points every table reads (the
tables share `m_pts`), the `2J` rep streams `(table j, rep r)` in any order the prover picks, each round's coins drawn
after its bytes, and every proof file after the last coin of every stream (the barrier).

* `liveSession dg J mPts N`: that live session, on the coin server with retained round bytes. `liveSession` at `J = 1`
  is `Refine.liveTable` up to the submissions' table index.
* `modelSession out G`: the model's session of `J` tables, `Commit`'s caps, the link points, then `batch` over the
  tables, each its two reps concurrently (`G j pts r`) and its openings. `sessionBL` is one (`Session.lean`).
* `DecodesTable … j`: table `j`'s decoder is right on accepted sessions: on every live outcome the live verdict
  accepts, table `j`'s reps run on the tapes decoded from its streams `(j, 0)`, `(j, 1)` and its proof files, and the
  model accepts table `j`'s output on those runs. One hypothesis per table.
* `simSession`, `live_le_session`: the simulation, and the transfer: against any live prover, the live verdict holds
  with probability at most the model session's acceptance (every table's) against the simulated prover, whenever
  every table's decoder is right.
-/

namespace FlockSoundness.Discharge.Tables

open Flock FlockLevel3 Game Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## The live session -/

/-! ## The model's session -/

section Model

variable {P OutT Cp Op : Type} {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)

end Model

/-! ## Tapes -/

section Tapes

variable {J mPts : ℕ} (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg)

end Tapes

/-! ## Following decoded messages -/

section Follow

variable {α : Type}

end Follow

/-! ## The simulation -/

section Sim

variable {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
  (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
  (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)

variable {out ok pL opn}

variable (out)

variable {out}

variable (out ok pL opn)

/-- **The transfer, for a session of `J` tables.** Against any live prover, the live verdict holds with probability at
most the model session's acceptance of every table against the simulated prover, whenever every table's decoder is
right on accepted sessions. -/
theorem live_le_session (N : ℕ) (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (hinh : ∀ j pts r, AllInh (G j pts r)) (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j)
    (Pr : Strategy (liveSession dg J mPts N)) :
    prob pL (liveSession dg J mPts N) Pr ≤
      prob (AllOk ok dflt) (modelSession out G)
        ((simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec).strategy Pr) :=
  Sim.prob_le _ Pr

end Sim

end FlockSoundness.Discharge.Tables
