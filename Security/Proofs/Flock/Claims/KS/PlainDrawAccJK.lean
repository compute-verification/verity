import Proofs.Flock.Claims.KS.PlainDrawAccK
import Proofs.Flock.Claims.KS.PlainDrawJ

/-!
# flock-verify's acceptance at a draw value, at the parsed record, through `keep`

`PlainDrawAccK` at `AcceptsAtDJK`: each lemma's proof on the walkers of `verifyJWithMK keep` (`verifyJK_ok_tables`,
`verifyJK_one_ok`) and of `Record.decodeJWithK keep` (`decodeJWithK_points`, `decodeJWithK_roundsWF`,
`decodeJWithK_allRetained`), which have the text walkers' statements at the parsed value. Each name is its `K`
original's with `J` before the `K` (`accepts_oneAnyDJK`), or after it where it has none (`LeafFormJ`).
-/

namespace FlockSoundness.Discharge.AtDraw

open Claims.Bridge.PlainDraw (setupAtD AcceptsAtDJK TranscriptJ named_of_setupAtD)

local notation "𝔽" => FlockLevel3.GF128

section Any

open Flock FlockLevel3 Refine Game Transcript Model Accounting
open FlockSoundness.Discharge.Exec FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **A one-table acceptance at `dr` is one `setupOne` statement's at `dr`**, whose table has a registry scheme
(`accepts_oneAny`). -/
theorem accepts_oneAnyDJK (keep : Bool → Bool) {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ}
    (hc : I.count ≤ 1) (h : AcceptsAtDJK keep I dr t) :
    ∃ st, setupOne I.tags I.circuitFile I.publicFile I.coins I.tables dr I.partition I.program = .ok st ∧
      (Setup.ofCircuit st).merkle.Named ∧
      verifyJWithMK keep With.std st.spec #[Setup.ofCircuit st] t.j t.proofs = .ok () := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  have hN := named_of_setupAtD hs
  rcases setupTables_okAny hs with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  · exact ⟨st, hst, hN _ (by simp), by simpa using hv⟩
  · omega

/-- **flock-verify's acceptance at `dr`, round by round** (`accepts_rep_roundsAny`). -/
theorem accepts_rep_roundsAnyDJK (keep : Bool → Bool) {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ}
    (hr : I.tags.retainRounds = true) (h : AcceptsAtDJK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeJWithK keep With.std spec t.j = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 0 t.proofs[2 * j]! ∧
        RepRounds spec (Setup.ofCircuit sts[j]) sch sess j 1 t.proofs[2 * j + 1]! := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verifyJK_ok_tables keep (named_of_setupAtD hs) hv
  have hret := decodeJWithK_allRetained keep With.std ((setupTables_retainAny hs).trans hr) hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨hdim, -⟩ := fast100_dims _ _ hf
  have hlv : ∀ i < schs[j].levels.size, LevelSeam (schs[j].levels.getD i default) := fun i hi =>
    fast100_level _ _ hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hm : 14 ≤ (Setup.ofCircuit sts[j]).m := by have := exec_fast100_ge hf; omega
  exact ⟨schs[j], hf, rep_rounds_retained hm hdim hlv hret h0, rep_rounds_retained hm hdim hlv hret h1⟩

/-- **flock-verify's acceptance at `dr` makes each table's salted verdict accept** (`accepts_verdict_hm96Any`). -/
theorem accepts_verdict_hm96AnyDJK (keep : Bool → Bool) {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ}
    (hl : I.tags.merkleLeaf = "hm96-sha512/v1") (h : AcceptsAtDJK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeJWithK keep With.std spec t.j = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
        ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ ∧
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ ∧
          (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, hsz, -, -, hall⟩ := verifyJK_ok_tables keep (named_of_setupAtD hs) hv
  obtain ⟨-, hsetup⟩ := setupTables_setupOne hs
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hs
  have hpts := decodeJWithK_points keep With.std hsess
  rw [Array.size_map] at hsz
  refine ⟨spec, sts, sess, hs, hsess, hsz, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  obtain ⟨d, hset, -⟩ := hsetup j hj
  refine ⟨schs[j], hf, fun hwf hr hm => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun p hp => (hpts.2 p hp).trans (hmp j hj).symm)
  exact table_verdict_hm96 spec (Setup.ofCircuit sts[j]) (stmtOf sts[j] hwf (regionsOf sts[j])) hm rfl rfl
    (ofCircuit_fold sts[j] hwf _) (setupOne_tagsAt_hm96 hl hset) _ sess hext
    (decodeJWithK_roundsWF keep With.std hsess) schs[j] hf j _ _
    c h0 h1

/-- **Each table's verdict at the model leaf `lf` accepts, at flock-verify's acceptance at `dr` through `keep`**: what
`accepts_verdict_hm96AnyDJK` gives at `leaf512` and `accepts_verdict_plainAnyDK` at `leafPlain`. -/
def TableVerdictsDJK (keep : Bool → Bool) (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512) (I : Exec.Inputs)
    (dr : Option Json) (t : TranscriptJ) : Prop :=
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeJWithK keep With.std spec t.j = .ok sess ∧ sts.size = spec.tables.size ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m),
        ∃ (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ ∧
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ ∧
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ ∧
          (tableOutL H512 enc512 lf (capOfRoot sess.rootB[j]!) (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!])).accepted =
            true

/-- **A leaf form at hidden outputs**: the tag `merkleLeaf` names, the model leaf `lf`, each table's verdict accepting
at `lf` under that tag (`TableVerdictsDJK`), and the 64-byte digests of the scheme a hidden setup under the tag commits
with. -/
structure LeafFormJ where
  tag : String
  lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512
  verdicts : ∀ (keep : Bool → Bool) {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ},
    I.tags.hiddenOutputs = true → I.tags.merkleLeaf = tag → AcceptsAtDJK keep I dr t → TableVerdictsDJK keep lf I dr t
  digestLen : ∀ {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Json} {program : Option ByteArray} {st : Flock.Stmt},
    Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st → tags.merkleLeaf = tag →
      st.merkle.digestLen = 64

/-- **The verify command's tags under hidden outputs at a leaf form**: `TagsOkH` at `F`'s tag. -/
def LeafFormJ.TagsOk (F : LeafFormJ) (I : Exec.Inputs) : Prop :=
  I.tags.hiddenOutputs = true ∧ I.tags.retainRounds = true ∧ I.tags.merkleLeaf = F.tag ∧ I.count ≤ 1

/-- **`hm96-sha512/v1` leaves as a leaf form**: `leaf512`, with `accepts_verdict_hm96AnyDJK`. -/
noncomputable abbrev LeafFormJ.hm96 : LeafFormJ where
  tag := "hm96-sha512/v1"
  lf := leaf512
  verdicts := by
    intro keep _ _ _ _ hl h
    exact accepts_verdict_hm96AnyDJK keep hl h
  digestLen := by
    intro _ _ _ _ _ _ _ _ _ hst hl
    rw [HiddenExec.setupHidden_hm96 hl hst]
    rfl

end Any

section Req0

open Flock Refine Model Game FlockSoundness.Discharge.Exec
open Lean (Json)

/-- **An accepted rep at `dr` requests level 0 first** (`accepts_req0`). -/
theorem accepts_req0DJK (keep : Bool → Bool)
    {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ} (h : AcceptsAtDJK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeJWithK keep With.std spec t.j = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (r : ℕ), r < 2 →
        ∀ (s : StreamRec) (p : Proof) (pend : PendL 𝔽 Dig),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString r) =
            some s →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + r]! = .ok p →
          Plays fOf (Model.repC execArith r (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) pend →
          (∃ (R : Req) (rest : List Req), pend.reqs = R :: rest ∧ R.level = 0 ∧
            R.positions = (Flock.positions (sch.levels.getD 0 default)
              (s.rounds[Claims.Flock.plainQueryRound sts[j].m sts[j].c.kLog sch.k0
                (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length)]!).coins).toList ∧
            p.ligerito.initialProof.rows.size = R.positions.length) ∧
          ∀ R ∈ pend.reqs, R.lanes = (lvOf (schedOf sch) R.level).lanes ∧
            R.depth = Model.pathLen (lvOf (schedOf sch) R.level) := by
  obtain ⟨spec, sts, hs, hv⟩ := h
  obtain ⟨sess, schs, hsess, -, -, -, hall⟩ := verifyJK_ok_tables keep (named_of_setupAtD hs) hv
  obtain ⟨hp2, hmp⟩ := HiddenExec.setupTables_pointsAny hs
  have hpts := decodeJWithK_points keep With.std hsess
  refine ⟨spec, sts, sess, hs, hsess, fun j hj => ?_⟩
  obtain ⟨hs', c, hf, h0, h1⟩ := hall j (by rw [Array.size_map]; exact hj)
  simp only [Array.getElem_map] at hf h0 h1
  refine ⟨schs[j], hf, fun hwf hr hm r hr2 s p pend hsr hp hpl => ?_⟩
  have hext := ofCircuit_extra sts[j] hwf hr sess.points (by rw [hpts.1, hp2])
    (fun q hq => (hpts.2 q hq).trans (hmp j hj).symm)
  have h15 : 15 ≤ sts[j].m := by have : 22 ≤ sts[j].m := exec_fast100_ge hf; omega
  obtain ⟨hdim, hsm⟩ := fast100_dims _ schs[j] hf
  have hlv : ∀ i < schs[j].levels.size, LevelSeam (schs[j].levels.getD i default) := fun i hi =>
    fast100_level _ schs[j] hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < schs[j].levels.size, Accounting.ceilLog2 (schs[j].levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha _ schs[j] hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hrep : verifyRep spec (Setup.ofCircuit sts[j]) schs[j] sess j r t.proofs[2 * j + r]! = .ok c := by
    rcases r with _ | _ | r
    · exact h0
    · exact h1
    · omega
  obtain ⟨s', p', hs'', hp', -, -, -, -, hk⟩ := rep_refines_req0 (D := Dig) spec (Setup.ofCircuit sts[j])
    (stmtOf sts[j] hwf (regionsOf sts[j])) hm h15 schs[j] rfl rfl (ofCircuit_fold sts[j] hwf _) _ sess hext hdim hlv
    hα1 hsm (decodeJWithK_roundsWF keep With.std hsess) j r _ _ hrep
  obtain rfl : s' = s := Option.some.inj (hs''.symm.trans hsr)
  obtain rfl : p' = p := Except.ok.inj (hp'.symm.trans hp)
  obtain ⟨pend', hpl', -, hok, -, R, rest, hRe, hRl, hRp⟩ :=
    hk capDig (fun _ ℓ k => (((openAt p'.ligerito (schs[j].levels.size - 1) ℓ).1)[k]!.toList.map fOf, []))
      (fun _ _ _ _ => rfl)
  obtain rfl := ZkExec.plays_det hpl hpl'
  have hmc := (hok R (by rw [hRe]; exact List.mem_cons_self)).2.2.2
  rw [hRl] at hmc
  simp only [openAt, ↓reduceIte] at hmc
  refine ⟨⟨R, rest, hRe, hRl, hRp, by simpa using (merkleCheck_rows hmc).1⟩, fun R' hR' => ?_⟩
  obtain ⟨hlt, hl, hd, -⟩ := hok R' hR'
  rw [lvOf_schedOf hlt]
  exact ⟨hl, hd⟩

/-- **At flock-verify's acceptance at `dr` through `keep`, each table's level-0 openings at the model leaf `lf` are its
reps' deployed entries** (`accepts_opens₀L`). -/
theorem accepts_opens₀LDLfJK (keep : Bool → Bool) (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512)
    {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ} (h : AcceptsAtDJK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeJWithK keep With.std spec t.j = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 lf cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          out.accepted = true →
          (∀ x ∈ opens₀L (schedOf sch) out, VerifiedL H512 enc512 lf cap
            (lvOf (schedOf sch) 0).lanes (Model.pathLen (lvOf (schedOf sch) 0)) x ∧
              x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
          (opens₀L (schedOf sch) out).map (fun x => (x.1, x.2.1)) =
            Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
              Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! := by
  obtain ⟨spec, sts, sess, hs, hsess, hall⟩ := accepts_req0DJK keep h
  refine ⟨spec, sts, sess, hs, hsess, fun j hj => ?_⟩
  obtain ⟨sch, hf, hk⟩ := hall j hj
  refine ⟨sch, hf, fun hwf hr hm p₀ p₁ s₀ s₁ pend₀ pend₁ cap out hs₀ hs₁ hp₀ hp₁ hpl₀ hpl₁ hout hacc => ?_⟩
  subst hout
  obtain ⟨⟨R₀, rest₀, hR₀, hl₀, hpos₀, hrw₀⟩, hsh₀⟩ := hk hwf hr hm 0 (by norm_num) s₀ p₀ pend₀ hs₀ hp₀ hpl₀
  obtain ⟨⟨R₁, rest₁, hR₁, hl₁, hpos₁, hrw₁⟩, hsh₁⟩ := hk hwf hr hm 1 (by norm_num) s₁ p₁ pend₁ hs₁ hp₁ hpl₁
  have h2 := (fast100_dims _ sch hf).1.2.1
  have hseam : LevelSeam (sch.levels.getD 0 default) :=
    fast100_level _ sch hf _ (by unfold Array.getD; rw [dite_eq_left (by omega)]; exact Array.getElem_mem _)
  have hlv0 := lvOf_schedOf (s := sch) (i := 0) (by omega)
  have hQ : (lvOf (schedOf sch) 0).queries = (sch.levels.getD 0 default).q := by rw [hlv0]; rfl
  have hq : ∀ (R : Req) (coins : Array F128),
      R.positions = (Flock.positions (sch.levels.getD 0 default) coins).toList →
      R.positions.length = (sch.levels.getD 0 default).q := fun R coins h => by
    rw [h, Array.length_toList, positions_size _ hseam]
  have hent : (opens₀L (schedOf sch) (tableOutL H512 enc512 lf cap (pend₀, pend₁)
      (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
        sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]))).map
        (fun x => (x.1, x.2.1)) =
      Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
          (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
            (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
        Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
          (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
            (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! := by
    unfold opens₀L
    rw [List.map_append, hQ, Claims.Flock.plainRepColsOf_ok hp₀, Claims.Flock.plainRepColsOf_ok hp₁,
      opnOf_pair hp₀ hp₁]
    congr 1
    · exact Claims.Flock.slotOpensL_entries _ 0 _ _ _ R₀ rest₀ hR₀ hl₀ hpos₀ (hq _ _ hpos₀) hrw₀ fun _ => rfl
    · exact Claims.Flock.slotOpensL_entries _ 1 _ _ _ R₁ rest₁ hR₁ hl₁ hpos₁ (hq _ _ hpos₁) hrw₁ fun _ => rfl
  refine ⟨fun x hx => ⟨?_, ?_⟩, hent⟩
  · refine Claims.Flock.verified_opens₀L H512 enc512 lf (schedOf sch) _ (pend₀, pend₁) _ hacc
      (fun r hr R hR hlR => ?_) x hx
    rcases r with _ | _ | r
    · exact hlR ▸ hsh₀ R hR
    · exact hlR ▸ hsh₁ R hR
    · omega
  · have hm := List.mem_map_of_mem (f := fun x => (x.1, x.2.1)) hx
    rw [hent, Claims.Flock.plainRepColsOf_ok hp₀, Claims.Flock.plainRepColsOf_ok hp₁] at hm
    rcases List.mem_append.1 hm with hm | hm
    · exact Claims.Flock.lt_of_mem_positions hseam (Claims.Flock.fst_mem_positions_of_mem_zkRepCols hm)
    · exact Claims.Flock.lt_of_mem_positions hseam (Claims.Flock.fst_mem_positions_of_mem_zkRepCols hm)

/-- **At flock-verify's acceptance at `dr`, each table's level-0 openings are its reps' deployed entries**:
`accepts_opens₀LDLfJK` at `leaf512`. -/
theorem accepts_opens₀LDJK (keep : Bool → Bool)
    {I : Exec.Inputs} {dr : Option Json} {t : TranscriptJ} (h : AcceptsAtDJK keep I dr t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : Session), setupAtD I dr = .ok (spec, sts) ∧
      Record.decodeJWithK keep With.std spec t.j = .ok sess ∧
      ∀ (j : ℕ) (hj : j < sts.size), ∃ sch : Flock.Schedule, Flock.fast100 sts[j].m = some sch ∧
        ∀ (hwf : StmtWF sts[j]) (hr : RegionsWF sts[j]) (hm : 13 ≤ sts[j].m) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec)
          (pend₀ pend₁ : PendL 𝔽 Dig) (cap : Cap Dig) (out : OutCL 𝔽 Dig (List UInt8)),
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 0) =
            some s₀ →
          sess.streams.find? (fun x => x.name == toString spec.tables[j]! ++ toString "/rep" ++ toString 1) =
            some s₁ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j]! = .ok p₀ →
          decodeProof (shapeOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
            PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) t.proofs[2 * j + 1]! = .ok p₁ →
          Plays fOf (Model.repC execArith 0 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₀ capDig)
            (s₀.rounds.toList.map (·.coins)) pend₀ →
          Plays fOf (Model.repC execArith 1 (stmtOf sts[j] hwf (regionsOf sts[j])) (schedOf sch) hm
              (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                (ptsOf sts[j].mPts sess.points)))
            (repMsgs (Setup.ofCircuit sts[j]) sch (2 + (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j]))
              PT_LOCAL sts[j].mPts (ptsOf sts[j].mPts sess.points)).length) p₁ capDig)
            (s₁.rounds.toList.map (·.coins)) pend₁ →
          out = tableOutL H512 enc512 leaf512 cap (pend₀, pend₁)
            (opnOf (Setup.ofCircuit sts[j]) sch (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL
              sts[j].mPts (ptsOf sts[j].mPts sess.points)).length #[t.proofs[2 * j]!, t.proofs[2 * j + 1]!]) →
          out.accepted = true →
          (∀ x ∈ opens₀L (schedOf sch) out, VerifiedL H512 enc512 leaf512 cap
            (lvOf (schedOf sch) 0).lanes (Model.pathLen (lvOf (schedOf sch) 0)) x ∧
              x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
          (opens₀L (schedOf sch) out).map (fun x => (x.1, x.2.1)) =
            Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₀.rounds t.proofs[2 * j]! ++
              Claims.Flock.plainRepColsOf (shapeOf (Setup.ofCircuit sts[j]) sch
                (Model.extraClaims (stmtOf sts[j] hwf (regionsOf sts[j])) PT_LOCAL sts[j].mPts
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! :=
  accepts_opens₀LDLfJK keep leaf512 h

end Req0

section Reps

open Flock Refine Exec Game Model Accounting Audit
open FlockSoundness.Discharge.DecodeZL FlockSoundness.Discharge.HiddenExec
open Lean (Json)

/-- **flock-verify's acceptance at `dr` through `keep` of a one-table session, rep by rep, where each table's verdict
accepts at the model leaf `lf`**, the one table's statement `st` `setupOne`'s at `dr`. -/
theorem accepts_repsOneDLfJK (keep : Bool → Bool) (lf : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512)
    {I : Inputs} {dr : Option Json} {t : TranscriptJ} (hrr : I.tags.retainRounds = true) (hc1 : I.count ≤ 1)
    (hv : TableVerdictsDJK keep lf I dr t) (h : AcceptsAtDJK keep I dr t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : Session) (sch : Flock.Schedule),
      setupOne I.tags I.circuitFile I.publicFile I.coins I.tables dr I.partition I.program = .ok st ∧
      setupAtD I dr = .ok (spec, #[st]) ∧ Record.decodeJWithK keep With.std spec t.j = .ok sess ∧
      Flock.fast100 st.m = some sch ∧
      ∀ (hwf : StmtWF st) (_hr : RegionsWF st) (hm : 13 ≤ st.m), ∃ pend : Fin 2 → PendL 𝔽 Dig,
        (∀ r : Fin 2, ∃ (s : StreamRec) (p : Proof) (Tv : F128),
          sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
            some s ∧
          decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
            st.mPts (ptsOf st.mPts sess.points)).length) t.proofs[r.val]! = .ok p ∧
          Plays fOf (Model.repC execArith r.val (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
              (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
            (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) (pend r) ∧
          (chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r.val).toUTF8
            (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
              (ptsOf st.mPts sess.points)).length) p Tv)).1.length = s.rounds.size ∧
          ∀ k < s.rounds.size, Retains (s.rounds[k]!) (encs ((chunks (repFrames (Setup.ofCircuit st) sch
            (spec.domain spec.tables[0]! r.val).toUTF8 (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p Tv)).1.getD k []))) ∧
        (tableOutL H512 enc512 lf (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true := by
  obtain ⟨spec, sts, sess, hsu, hdec, -, hall⟩ := hv
  obtain ⟨spec', sts', sess', hsu', hdec', -, hallR⟩ := accepts_rep_roundsAnyDJK keep hrr h
  rw [hsu, Except.ok.injEq, Prod.mk.injEq] at hsu'
  obtain ⟨rfl, rfl⟩ := hsu'
  rw [hdec, Except.ok.injEq] at hdec'
  subst hdec'
  obtain ⟨hp2, hmp⟩ := setupTables_pointsAny hsu
  have hpts := decodeJWithK_points keep With.std hdec
  rcases setupTables_okAny hsu with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc', -⟩
  swap
  · omega
  have h0 : 0 < (#[st] : Array Flock.Stmt).size := by simp
  obtain ⟨sch, hf, hk⟩ := hall 0 h0
  obtain ⟨sch', hf', hR0, hR1⟩ := hallR 0 h0
  obtain rfl : sch = sch' := Option.some.inj (hf.symm.trans hf')
  refine ⟨st, st.spec, sess, sch, hst, hsu, hdec, hf, fun hwf hr hm => ?_⟩
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hv⟩ := hk hwf hr hm
  have hext := ofCircuit_extra st hwf hr sess.points (by rw [hpts.1, hp2])
    (fun q hq => (hpts.2 q hq).trans (hmp 0 h0).symm)
  obtain ⟨e₀, f₀, xs₀, T₀, hf₀, hx₀, hq₀, hl₀, hr₀⟩ := hR0
  obtain ⟨e₁, f₁, xs₁, T₁, hf₁, hx₁, hq₁, hl₁, hr₁⟩ := hR1
  rw [(hext xs₀ hx₀).1] at hq₀ hl₀ hr₀
  rw [(hext xs₁ hx₁).1] at hq₁ hl₁ hr₁
  obtain rfl : f₀ = p₀ := Except.ok.inj (hq₀.symm.trans hp₀)
  obtain rfl : f₁ = p₁ := Except.ok.inj (hq₁.symm.trans hp₁)
  obtain rfl : e₀ = s₀ := Option.some.inj (hf₀.symm.trans hs₀)
  obtain rfl : e₁ = s₁ := Option.some.inj (hf₁.symm.trans hs₁)
  refine ⟨![pend₀, pend₁], fun r => ?_, hv⟩
  rcases r with ⟨_ | _ | k, hk2⟩
  · exact ⟨e₀, f₀, T₀, hf₀, hq₀, hpl₀, hl₀, hr₀⟩
  · exact ⟨e₁, f₁, T₁, hf₁, hq₁, hpl₁, hl₁, hr₁⟩
  · omega

/-- **flock-verify's acceptance at `dr` of a one-table session, rep by rep**: `accepts_reps` and `accepts_repsH`, with
the one table's statement `st` `setupOne`'s at `dr` under either tag: `accepts_repsOneDLfJK` at `leaf512`. -/
theorem accepts_repsOneDJK (keep : Bool → Bool)
    {I : Inputs} {dr : Option Json} {t : TranscriptJ} (hrr : I.tags.retainRounds = true)
    (hl : I.tags.merkleLeaf = "hm96-sha512/v1") (hc1 : I.count ≤ 1) (h : AcceptsAtDJK keep I dr t) :
    ∃ (st : Flock.Stmt) (spec : SessionSpec) (sess : Session) (sch : Flock.Schedule),
      setupOne I.tags I.circuitFile I.publicFile I.coins I.tables dr I.partition I.program = .ok st ∧
      setupAtD I dr = .ok (spec, #[st]) ∧ Record.decodeJWithK keep With.std spec t.j = .ok sess ∧
      Flock.fast100 st.m = some sch ∧
      ∀ (hwf : StmtWF st) (_hr : RegionsWF st) (hm : 13 ≤ st.m), ∃ pend : Fin 2 → PendL 𝔽 Dig,
        (∀ r : Fin 2, ∃ (s : StreamRec) (p : Proof) (Tv : F128),
          sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
            some s ∧
          decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
            st.mPts (ptsOf st.mPts sess.points)).length) t.proofs[r.val]! = .ok p ∧
          Plays fOf (Model.repC execArith r.val (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
              (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
            (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p capDig)
            (s.rounds.toList.map (·.coins)) (pend r) ∧
          (chunks (repFrames (Setup.ofCircuit st) sch (spec.domain spec.tables[0]! r.val).toUTF8
            (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
              (ptsOf st.mPts sess.points)).length) p Tv)).1.length = s.rounds.size ∧
          ∀ k < s.rounds.size, Retains (s.rounds[k]!) (encs ((chunks (repFrames (Setup.ofCircuit st) sch
            (spec.domain spec.tables[0]! r.val).toUTF8 (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL
              st.mPts (ptsOf st.mPts sess.points)).length) p Tv)).1.getD k []))) ∧
        (tableOutL H512 enc512 leaf512 (capOfRoot sess.rootB[0]!) (pend 0, pend 1)
          (opnOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
            (ptsOf st.mPts sess.points)).length #[t.proofs[0]!, t.proofs[1]!])).accepted = true :=
  accepts_repsOneDLfJK keep leaf512 hrr hc1 (accepts_verdict_hm96AnyDJK keep hl h) h

end Reps

section Tables

open Flock FlockLevel3 Refine
open FlockSoundness.Discharge.Exec (Inputs verifyJK_one_ok)
open FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.HiddenExec

/-- **A draw flock-verify accepts at sets up**: `drawSetup_of_accepts` at `AcceptsAtDJK keep I (some d)`, with no
premise on the record's draw. -/
theorem drawSetup_of_acceptsDJK (keep : Bool → Bool)
    {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) {t : TranscriptJ} (h : AcceptsAtDJK keep I (some d) t) :
    ∃ st D, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st ∧
      Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetup I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hN, hv⟩ := accepts_oneAnyDJK keep hc1 h
  rw [setupOne_public hh] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verifyJK_one_ok keep hN hv
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  exact ⟨st, D, hst, hD, fun hS hm => drawSetup_of_setupH ht hc ho hst hf hD hS hm⟩

/-- **At a draw flock-verify accepts at, the headline's table is `stmtOf` of the statement it accepted**:
`tableAt_of_accepts` at `AcceptsAtDJK keep I (some (dj S))`, with no premise on the record's draw. -/
theorem tableAt_of_acceptsDJK (keep : Bool → Bool) {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmRow.parse I.tags I.circuitFile I.tables = .ok c) (ho : c.outNet = c.unitNet)
    (hh : I.tags.hiddenOutputs = false) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetup I mPts S₀ (dj S₀)) {t : TranscriptJ}
    (h : AcceptsAtDJK keep I (some (dj S)) t)
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program = .ok st ∧
      ∃ hwf : StmtWF st, (tableAt ht dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := drawSetup_of_acceptsDJK keep (S := S) (mPts := mPts) ht hc ho hh hc1 h
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAt_stmtOf ht dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, e⟩⟩

/-- **A draw flock-verify accepts at under hidden outputs sets up**: `drawSetupH_of_accepts` at
`AcceptsAtDJK keep I (some d)`, with no premise on the record's draw. -/
theorem drawSetupH_of_acceptsDJK (keep : Bool → Bool)
    {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {d : Lean.Json} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) {t : TranscriptJ} (h : AcceptsAtDJK keep I (some d) t) :
    ∃ st D, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program =
        .ok st ∧ Draw.UnitDraw.ofJson d = .ok D ∧
      ((∀ u ∈ S, u.val ∈ D.proved) → st.mPts = mPts → ∃ x : DrawSetupH I mPts S d, x.st = st) := by
  obtain ⟨st, hst, hN, hv⟩ := accepts_oneAnyDJK keep hc1 h
  rw [setupOne_hidden hh] at hst
  obtain ⟨-, sch, -, -, hf, -, -⟩ := verifyJK_one_ok keep hN hv
  obtain ⟨pop, -, hdr⟩ := setupHidden_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec (c := { st.c with outPorts := #[] }) hdr
  exact ⟨st, D, hst, hD, fun hS hm => drawSetupH_of_setupHiddenI ht hc hst hf hD hS hm⟩

/-- **At a draw flock-verify accepts at under hidden outputs, the hidden table is `stmtOf` of the statement it
accepted**: `tableAtH_of_acceptsI` at `AcceptsAtDJK keep I (some (dj S))`, with no premise on the record's draw. -/
theorem tableAtH_of_acceptsDJK (keep : Bool → Bool) {I : Inputs} {mPts n : ℕ} {S : Finset (Fin n)} {c : Flock.Circuit}
    (ht : I.tags.typed = false) (hc : HmOut.parse I.tags I.circuitFile I.tables = .ok c)
    (hh : I.tags.hiddenOutputs = true) (hc1 : I.count ≤ 1) (dj : Finset (Fin n) → Lean.Json)
    {S₀ : Finset (Fin n)} (x₀ : DrawSetupH I mPts S₀ (dj S₀)) {t : TranscriptJ}
    (h : AcceptsAtDJK keep I (some (dj S)) t)
    (hS : ∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved)
    (hmPts : ∀ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition
      I.program = .ok st → st.mPts = mPts) :
    ∃ st, Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program =
        .ok st ∧ ∃ hwf : StmtWF st, (tableAtH dj x₀ S).T.S = stmtOf st hwf (regionsOf st) := by
  obtain ⟨st, D, hst, hD, hx⟩ := drawSetupH_of_acceptsDJK keep (S := S) (mPts := mPts) ht hc hh hc1 h
  obtain ⟨x, rfl⟩ := hx (hS D hD) (hmPts _ hst)
  obtain ⟨y, hy, e⟩ := tableAtH_stmt dj x₀ x
  exact ⟨x.st, hst, hy ▸ ⟨y.wf.1, congrArg (fun T : TableH I mPts S => T.T.S) e⟩⟩

end Tables

end FlockSoundness.Discharge.AtDraw
