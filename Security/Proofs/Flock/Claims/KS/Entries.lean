import Proofs.Flock.Claims.KS.OffSC
import Proofs.Flock.Claims.KS.LevelZeroAcc

/-!
# The deployed reader's level-0 entries at flock-verify's acceptance

`exists_ne_of_off₀L_sc` (`OffSC`) needs, of a salted output, that its level-0 openings verify, sit below `2^d`, and are,
with their rows, the deployed reader's entries. This file proves all three at flock-verify's acceptance:

* `Claims.Flock.slotOpensL_entries`: a rep whose first request is level 0's, at the positions `Flock.positions` draws
  from `coins`, with one opened row per position, opens at level 0 exactly the entries `zkRepCols` reads off `coins`
  and those rows, in order.
* `Claims.Flock.plainRepColsOf_ok`, `Claims.Flock.lt_of_mem_positions`: the deployed plain reader's entries at a proof
  that decodes are `zkRepCols`'s, and a drawn position is below `2^d` at a level `fast100` uses.
* `Claims.Flock.exists_ne_of_entries`: `exists_ne_of_off₀L_sc` with the entries given as one list equation.
* `accepts_opens₀L`: when `Accepts I t`, whether or not the outputs are hidden, at every table, every pair of the
  reps' runs on the proof files' messages and the streams' coins whose salted output accepts at a cap has every level-0
  opening verifying against that cap, at level 0's lanes and path length, below `2^d`, and its level-0 openings'
  positions and rows are its reps' `plainRepColsOf` entries at the session's shape, stream rounds and proof files
  (`accepts_req0`, `verified_opens₀L`); `accepts_verdict_hm96` gives such runs, accepted at `capOfRoot` of the table's
  `root_B`.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Discharge.Exec

local notation "𝔽" => FlockLevel3.GF128

/-- **A rep's level-0 openings are the reader's entries**: when the rep's first request is level 0's, at the positions
`Flock.positions` draws from `coins`, `q` of them, and its openings are `rows`, one per position, its level-0 openings'
positions and rows are `zkRepCols`'s entries, in order. -/
theorem slotOpensL_entries {D Sl : Type} (out : Model.OutCL 𝔽 D Sl) (r : ℕ) (lv : _root_.Flock.Level)
    (coins : Array _root_.Flock.F128) (rows : Array (Array _root_.Flock.F128)) (R : Model.Req)
    (rest : List Model.Req) (hreqs : (if r = 0 then out.pend.1 else out.pend.2).reqs = R :: rest)
    (hl : R.level = 0) (hpos : R.positions = (_root_.Flock.positions lv coins).toList)
    (hq : R.positions.length = lv.q) (hrows : rows.size = R.positions.length)
    (hop : ∀ j, (out.opens r 0 j).1 = rows[j]!.toList.map FlockLevel3.GF128.of) :
    (slotOpensL out r 0 lv.q).map (fun x => (x.1, x.2.1)) = zkRepCols lv coins rows := by
  unfold slotOpensL zkRepCols
  rw [hreqs, List.find?_cons_of_pos (by simp [hl])]
  simp only [List.map_map]
  rw [← hq, List.take_length, ← hpos]
  apply List.ext_getElem
  · simp [hrows]
  · intro i h1 h2
    simp only [List.length_map, List.length_zipIdx] at h1
    simp [hop, hrows, h1]

/-- An entry `zkRepCols` reads is at a drawn position. -/
theorem fst_mem_positions_of_mem_zkRepCols {lv : _root_.Flock.Level} {coins : Array _root_.Flock.F128}
    {rows : Array (Array _root_.Flock.F128)} {e : ℕ × List 𝔽} (he : e ∈ zkRepCols lv coins rows) :
    e.1 ∈ (_root_.Flock.positions lv coins).toList := by
  unfold zkRepCols at he
  obtain ⟨pr, hpr, rfl⟩ := List.mem_map.1 he
  exact (List.of_mem_zip hpr).1

/-- A position `Flock.positions` draws at a level of the model's shape is below `2^d`. -/
theorem lt_of_mem_positions {lv : _root_.Flock.Level} (hl : Refine.LevelSeam lv) {coins : Array _root_.Flock.F128}
    {p : ℕ} (hp : p ∈ (_root_.Flock.positions lv coins).toList) : p < 2 ^ lv.d := by
  rw [← Refine.positions_eq lv hl coins] at hp
  rw [hl.2.2.2.2]
  exact positions_lt _ _ _ _ p hp

/-- A round's coins read with a default round are the coins read with no coins as the default. -/
theorem coins_getElem! (rs : Array _root_.Flock.Round) (i : ℕ) :
    (rs[i]!).coins = (rs[i]?.map (·.coins)).getD #[] := by
  by_cases h : i < rs.size
  · simp [h]
  · simp [h]
    rfl

/-- The deployed plain reader's entries at a proof that decodes: `zkRepCols` at level 0, the coins of round
`plainQueryRound` and the proof's level-0 rows. -/
theorem plainRepColsOf_ok {sh : _root_.Flock.Shape} {rounds : Array _root_.Flock.Round} {bytes : ByteArray}
    {p : _root_.Flock.Proof} (h : _root_.Flock.decodeProof sh bytes = .ok p) :
    plainRepColsOf sh rounds bytes = zkRepCols (sh.sched.levels.getD 0 default)
      (rounds[plainQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]!).coins p.ligerito.initialProof.rows := by
  unfold plainRepColsOf
  rw [h, coins_getElem!]

/-- **`exists_ne_of_off₀L_sc` at a list of entries**: when the level-0 openings verify below `2^d` and their positions
and rows are `os`, an output that opens a position off the oracle of `C` without a self-clash has the reader look up a
column off `C`. -/
theorem exists_ne_of_entries {D Sl : Type} (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D)
    (Lf : Merkle.Leaf (List 𝔽) Sl D H) (sch : Accounting.Schedule) (cap₀ : Model.Cap D)
    (out : Model.OutCL 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf sch 0).logLen) (C : Fin N → List 𝔽) (d : ℕ)
    (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀L sch out,
      VerifiedL H E Lf cap₀ (lvOf sch 0).lanes (Model.pathLen (lvOf sch 0)) x ∧ x.1 < 2 ^ d)
    (hent : (opens₀L sch out).map (fun x => (x.1, x.2.1)) = os)
    (hoff : Off₀L H E Lf sch cap₀ (oracleOf C) out) (hsc : ¬ SelfClashL H E Lf sch out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p :=
  exists_ne_of_off₀L_sc H E Lf sch cap₀ out hN C d os hcap (fun x hx => (hver x hx).1) (fun x hx => (hver x hx).2)
    (fun e he => by rw [← hent] at he; exact List.mem_map.1 he)
    (fun x hx => by rw [← hent]; exact List.mem_map_of_mem hx) hoff hsc

end Claims.Flock

namespace FlockSoundness.Discharge.Exec

open Flock Refine Model Game

local notation "𝔽" => FlockLevel3.GF128

/-- **At flock-verify's acceptance, each table's level-0 openings are its reps' deployed entries** (§9–§14). When
`Accepts I t`, at every table `j`, its `fast100` schedule and the statement read off its layout,
every pair of the reps' runs on the proof files' messages and the streams' coins whose salted output accepts at a cap
has its level-0 openings verifying against that cap at level 0's lanes and path length, below `2^d`, and its level-0
openings' positions and rows are the reps' `plainRepColsOf` entries, rep 0's then rep 1's, at the table's shape, the
reps' stream rounds and the proof files. -/
theorem accepts_opens₀L {I : Inputs} {t : Transcript} (h : Accepts I t) :
    ∃ (spec : SessionSpec) (sts : Array Flock.Stmt) (sess : RunRecord), setupOf I t.record = .ok (spec, sts) ∧
      Record.decode spec t.record = .ok sess ∧
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
                  (ptsOf sts[j].mPts sess.points)).length) s₁.rounds t.proofs[2 * j + 1]! := by
  obtain ⟨spec, sts, sess, hs, hsess, hall⟩ := accepts_req0 h
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
  have hent : (opens₀L (schedOf sch) (tableOutL H512 enc512 leaf512 cap (pend₀, pend₁)
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
  · refine Claims.Flock.verified_opens₀L H512 enc512 leaf512 (schedOf sch) _ (pend₀, pend₁) _ hacc
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

end FlockSoundness.Discharge.Exec
