import FlockSoundness.Discharge.ZkLink.Inh
import FlockSoundness.Protocol.Discharge.ZkLink.One
import FlockSoundness.Protocol.Discharge.ZkLink.StrictZ
import FlockSoundness.Protocol.Discharge.ZkLink.DrawOk
import Std.Data.TreeMap.Raw.Lemmas

/-!
# flock-verify `--zk`'s headline at the compiled `--zk` table

`Refines.lean`'s one-table refinement and headline at [wave2-zk-link]'s compiled `--zk` table. At each draw `S` and
registration `R` the table is `T S R : TabZK 𝔽 mPts`, its reps `ZkLink.repZCAt A Hs (T S R)` (`repZ`), the level-0 cap
first (`Fin 1 → Cap D`), the openings `ZkLink.OpensZ 𝔽 D Sl` last, and the verdict `ZkLink.acceptedZC` (`outZ`, `accZ`).
That table's `sessionZC` is `ZkLink.batchedSessionZC` at the one-table plan (`sessionZC_repZ`, by
`ZkLink.batchedSessionZC_one`), every rep's game has strategies (`ZkLink.allInh_repZCAt`), and the simulated prover
`simAuditZK … τ` is a strategy of that audit (`simAuditZC`), where the compiled count curve `ZkLink.zk_flock_countC`
applies:

* `zk_headline_exec_zc`: against any live prover, at any law,
  `Pr[flock-verify --zk accepts ∧ at least K wrong units] ≤ miss(K) + ε_ks + δ_link`;
* `zk_headline_exec_zc_execOS`: the same at the live draw (`Law.execOS`, `drawOS`'s stratified law) under A3
  (`UniformRandomBytes`), with `Law.stratified`'s count curve for `miss(K)`.

The wrong units are those of the simulated prover's committed values (`ZkLink.XplurZC` at `simAuditZC … τ`). Open,
named in the hypotheses: `hdec : DecodesZK …` (the per-transcript refinement, R11b at `--zk`), `hj : JointZC …`
(`ε_ks`), and `hlow : LoweringSoundZK …` with the value binding `vb`, as at [wave2-zk-link]; `δ_link` is `⊤` at a
prover outside A2 (`linkBoundZC`). Open and not a hypothesis: the coin tree (the live audit draws each round's coins
fresh; [wave2-zk-coins]), and `J > 1` (the plan has one table).
-/

namespace FlockSoundness.Discharge.ZkExec

open Game Audit Exec Refine Model Accounting ZkSession
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

variable {K D Sl : Type} [Field K] [Fintype K] [DecidableEq K] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
  (A : Arith 𝔽 K) (Hs : ZkLink.Hashes 𝔽 D Sl) (T : Finset (Fin n) → Reg → TabZK 𝔽 mPts)

/-- The compiled `--zk` table's reps at each draw and registration (`Refine.modelTable`'s `G`). -/
noncomputable def repZ (S : Finset (Fin n)) (R : Reg) : (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game (ZkLink.PendZ 𝔽 D Sl) :=
  ZkLink.repZCAt A Hs (T S R)

/-- The compiled `--zk` table's output: the one table's `outZC` at its level-0 cap. -/
noncomputable def outZ (S : Finset (Fin n)) (R : Reg) (caps : Fin 1 → Cap D)
    (pend : ZkLink.PendZ 𝔽 D Sl × ZkLink.PendZ 𝔽 D Sl) (os : ZkLink.OpensZ 𝔽 D Sl) : List (ZkLink.OutZC 𝔽 D Sl) :=
  [ZkLink.outZC Hs (T S R) (caps 0) pend os]

open Classical in
/-- The compiled `--zk` table's verdict. -/
noncomputable def accZ (S : Finset (Fin n)) (R : Reg) (outs : List (ZkLink.OutZC 𝔽 D Sl)) : Bool :=
  decide (ZkLink.acceptedZC [T S R] outs)

variable (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → ZkLink.OpensZ 𝔽 D Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → Fin 1 → Cap D) (N : ℕ) {L : Law n}

end FlockSoundness.Discharge.ZkExec
