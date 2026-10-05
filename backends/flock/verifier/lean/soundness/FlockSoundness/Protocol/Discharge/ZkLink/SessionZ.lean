import FlockSoundness.Protocol.Discharge.ZkLink.BadZ

/-!
# Knowledge soundness of a compiled `--zk` table in the session, from accepted reruns

`SessionSoundL`, `SessionAccL` and `SessionBatchAccL` for the compiled `--zk` session. The table at position
`pre.length` is replaced by its primed form (`expect_sessAtZ_push`), whose lockstep with the oracle session
(`lockSessZ`) fixes a table at every committed object: each level's cap, each rep's pads root and `τ`'s commitment. A
bad run is split by `bad_splitZ`, and the mass of runs off a fixed table is a fork term (`Lock.leafE_off_le`):

* `session_sound_of_table_accZ`: for an event `Acc` of the session's outputs that implies the table accepts, and any
  level-0 oracle `C`, `Pr[Acc ∧ C uncommitted] ≤ ε_c⁻ + Pr[Acc ∧ a run opens a level-0 position off C]`, with `ε_c⁻ =
  epsSZ`: the table's bound, the levels', pads roots' and `τ` commitments' fork terms, and the self-clash;
* `session_knowledge_sound_accZ`: `ε · Pr[the table extracted from k ≥ 1 accepted reruns is uncommitted] ≤
  ε_c⁻ + N₀·ε/(ek) + √(k·adv₀)`;
* `jointZC_ksBound`: `JointZC` at `ksBoundAccZ`, table `j`'s `ε_c⁻ + √(k·adv₀)` in the session, for `k ≥ 1`.
-/

namespace FlockSoundness.Discharge.ZkLink

open Game Model Accounting ZkSession
open scoped ENNReal

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
variable {D Sl : Type}

/-! ## Small facts -/

/-! ## The bad runs, pointwise -/

/-! ## Rewinding in the session -/

open Classical in
/-- **The tables `P̃` sends in the session**: at each committed object of the table at position `n`, the plurality of
the table's openings there: rep `s`'s pads at slot `s < 2`, rep `s - 2`'s `τ` at `s < 4`, and rep `(s - 4) % 2`'s level
`(s - 4) / 2` above. -/
noncomputable def plurSZ [Inhabited D] [Inhabited Sl] (Hs : Hashes F D Sl) (T : TabZK F mPts) (n : ℕ) :
    Lock.Chooser (Oracle F) (Cap D) (List (OutZC F D Sl)) :=
  fun s d g σ =>
    if s < 2 then padsTab T.N (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutZC F D Sl) => opPadsZ Hs T.N T.q s d (outs.getD n dfltOutZC)) g σ))
    else if s < 4 then tauTab (Classical.choose (Rewinding.exists_pluralityE
      (fun outs : List (OutZC F D Sl) => opTauZ Hs (s - 2) d (outs.getD n dfltOutZC)) g σ))
    else tabOf (lvOf T.sch ((s - 4) / 2)).lanes (lvOf T.sch ((s - 4) / 2)).logLen
      (Classical.choose (Rewinding.exists_pluralityE
        (fun outs : List (OutZC F D Sl) => opRZ Hs.H Hs.E Hs.Lf T.sch ((s - 4) % 2) ((s - 4) / 2) d
          (outs.getD n dfltOutZC)) g σ))

section Terms

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (T : TabZK F mPts)
  (pre post : List ((Fin 2 → Fin mPts → F) → Game (OutZC F D Sl))) (cap₀ : Cap D)

end Terms

/-! ## The compiled bound for any level-0 oracle, in the session -/

/-! ## Knowledge soundness in the session -/

/-! ## Several tables in one batched session -/

section Batch

variable [Inhabited D] [Inhabited Sl] (A : Arith F K) (Hs : Hashes F D Sl) (ts : List (TabZK F mPts))

/-- Table `j`'s strategy in the session, read at its position. -/
noncomputable def stratZ (j : Fin ts.length) (τ : Strategy (sessionZC A Hs ts)) :
    Strategy (sessAtZ ((gamesZC A Hs ts τ.1).take j) ((gamesZC A Hs ts τ.1).drop (j + 1))
      (tabAfterZC A Hs (ts.get j) (τ.1 j))) :=
  cast (congrArg Strategy (sessAfterZC_split A Hs ts τ.1 j)) τ.2

end Batch

end FlockSoundness.Discharge.ZkLink
