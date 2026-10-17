import Proofs.Flock.Claims.KS.ZkOffSC
import Proofs.Flock.Claims.KS.Entries
import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictOpens

/-!
# The masked reader's level-0 entries at a rep's run

`exists_ne_of_offW₀Z_sc` (`ZkOffSC`) needs, of a masked output, that its level-0 openings verify, sit below `2^d`, and
are, with their rows' witness lanes, the masked reader's entries. This file proves the entries at a rep's run, the
masked `Entries`:

* `Claims.Flock.slotOpensZ_entries`: a masked rep whose first request is level 0's, at the positions `Flock.positions`
  draws from `coins`, with one opened row per position, opens at level 0 exactly the entries `zkRepCols` reads off
  `coins` and those rows, in order, at every width the rows are cut to.
* `Claims.Flock.zkRepColsOf_ok`: the masked reader's entries at a proof that decodes are `zkRepCols`'s at level 0, the
  coins of round `zkQueryRound` and the proof's level-0 rows, cut to the witness lanes.
* `FlockSoundness.Discharge.ZkExec.zk_rep_entries`: when `repProg` runs on a rep's stream and consumes every round, the
  compiled rep played with the rep's messages and the rounds' coins opens at level 0, on the witness lanes, the masked
  reader's entries at the rep's stream rounds and proof bytes (`zk_rep_reqs`, whose first request is level 0's at the
  coins of round `zkQueryRound`).
* `Claims.Flock.exists_ne_of_zk_entries`: `exists_ne_of_offW₀Z_sc` with the entries given as one list equation.
-/

namespace Claims.Flock

open FlockSoundness FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession

local notation "𝔽" => FlockLevel3.GF128

/-- **A masked rep's level-0 openings are the reader's entries**: when the rep's first request is level 0's, at the
positions `Flock.positions` draws from `coins`, `q` of them, and its openings are `rows`, one per position, its level-0
openings' positions and rows, each cut to `w` symbols, are `zkRepCols`'s entries cut to `w`, in order. -/
theorem slotOpensZ_entries {D Sl : Type} (out : OutZC 𝔽 D Sl) (r : ℕ) (lv : _root_.Flock.Level)
    (coins : Array _root_.Flock.F128) (rows : Array (Array _root_.Flock.F128)) (R : Model.Req)
    (rest : List Model.Req) (hreqs : (if r = 0 then out.pend.1 else out.pend.2).lig.reqs = R :: rest)
    (hl : R.level = 0) (hpos : R.positions = (_root_.Flock.positions lv coins).toList)
    (hq : R.positions.length = lv.q) (hrows : rows.size = R.positions.length)
    (hop : ∀ j, (out.opens.rows r 0 j).1 = rows[j]!.toList.map FlockLevel3.GF128.of) (w : ℕ) :
    (slotOpensZ out r 0 lv.q).map (fun x => (x.1, x.2.1.take w)) =
      (zkRepCols lv coins rows).map fun p => (p.1, p.2.take w) := by
  unfold slotOpensZ zkRepCols
  rw [hreqs, List.find?_cons_of_pos (by simp [hl])]
  simp only [List.map_map]
  rw [← hq, List.take_length, ← hpos]
  apply List.ext_getElem
  · simp [hrows]
  · intro i h1 h2
    simp only [List.length_map, List.length_zipIdx] at h1
    simp [hop, hrows, h1]

/-- The masked reader's entries at a proof that decodes: `zkRepCols` at level 0, the coins of round `zkQueryRound`
and the proof's level-0 rows, each row cut to its first `2^k0` symbols. -/
theorem zkRepColsOf_ok {sh : _root_.Flock.Shape} {rounds : Array _root_.Flock.Round} {bytes : ByteArray}
    {zp : _root_.Flock.Zk.ZkProof} (h : _root_.Flock.Zk.decode sh bytes = .ok zp) :
    zkRepColsOf sh rounds bytes = (zkRepCols (sh.sched.levels.getD 0 default)
      (rounds[zkQueryRound sh.m sh.kLog sh.sched.k0 sh.claims]!).coins zp.proof.ligerito.initialProof.rows).map
        fun p => (p.1, p.2.take (2 ^ (sh.sched.levels.getD 0 default).lanes)) := by
  unfold zkRepColsOf
  rw [h, coins_getElem!]

/-- **`exists_ne_of_offW₀Z_sc` at a list of entries**: when the level-0 openings verify at `width₀` below `2^d` and
their positions and rows' witness lanes are `os`, an output that opens a position off the oracle of `C` on a witness
lane without a self-clash has the reader look up a column off `C`. -/
theorem exists_ne_of_zk_entries {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (cap₀ : Model.Cap D) (out : OutZC 𝔽 D Sl) {N : ℕ} (hN : N = 2 ^ (lvOf T.sch 0).logLen) (C : Fin N → List 𝔽)
    (d : ℕ) (os : List (ℕ × List 𝔽)) (hcap : out.cap₀ = cap₀)
    (hver : ∀ x ∈ opens₀Z T.sch out,
      VerifiedZ Hs.H Hs.E Hs.Lf cap₀ (width₀ T) (Model.pathLen (lvOf T.sch 0)) x ∧ x.1 < 2 ^ d)
    (hent : (opens₀Z T.sch out).map (fun x => (x.1, x.2.1.take (2 ^ T.sch.k0))) = os)
    (hoff : OffW₀Z Hs T cap₀ (oracleOf C) out) (hsc : ¬ SelfClashZ Hs T out) :
    ∃ (p : Fin N) (u : List 𝔽), lookCol (distinctBelow d os) p.val = some u ∧ u ≠ C p :=
  exists_ne_of_offW₀Z_sc Hs T cap₀ out hN C d os hcap (fun x hx => (hver x hx).1) (fun x hx => (hver x hx).2)
    (fun e he => by rw [← hent] at he; exact List.mem_map.1 he)
    (fun x hx => by rw [← hent]; exact List.mem_map_of_mem hx) hoff hsc

end Claims.Flock

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128

/-- **A masked rep's level-0 openings are its deployed entries** (live PROTOCOL.md §7). When rep `r`'s transcript
program runs on its stream's rounds and consumes every round, at a compiled table at the executable's dimensions whose
extra claims are the executable's, and the rep's proof bytes decode to its proof, every output whose rep-`r` result is
a play of the compiled rep with the rep's messages and the rounds' coins, and whose rep-`r` level-0 openings are the
proof's level-0 rows, opens at level 0, each row cut to its first `2^k0` symbols, the masked reader's entries at the
rep's stream rounds and proof bytes. -/
theorem zk_rep_entries {mPts : ℕ} {Sl : Type} (Hs : ZkLink.Hashes 𝔽 Dig Sl) (T : ZkSession.TabZK 𝔽 mPts)
    (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup) (sch : Flock.Schedule) (tPad : ℕ)
    (extras : Array Flock.Claim) (zp : Zk.ZkProof) (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (bytes : ByteArray) (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (hdec : Zk.decode (shapeZK st sch extras) bytes = .ok zp)
    (out : ZkLink.OutZC 𝔽 Dig Sl)
    (hp : Plays fOf (ZkLink.repZCAt execArith Hs T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
      (rounds.toList.map (·.coins)) (if r.val = 0 then out.pend.1 else out.pend.2))
    (ho : ∀ j, (out.opens.rows r.val 0 j).1 = zp.proof.ligerito.initialProof.rows[j]!.toList.map fOf) :
    (ZkLink.slotOpensZ out r.val 0 (lvOf T.sch 0).queries).map (fun x => (x.1, x.2.1.take (2 ^ T.sch.k0))) =
      Claims.Flock.zkRepColsOf (shapeZK st sch extras) rounds bytes := by
  obtain ⟨-, -, -, -, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨-, h2, hl0, -⟩ := hdim
  obtain ⟨pend', hp', -, hreqs, R, rest, hR, hl, hpos⟩ :=
    zk_rep_reqs Hs T pts r st sch tPad extras zp rounds dom tr hsch hseam hlen hext hwf hrun hk
  have he := plays_det hp hp'
  have hseam0 : LevelSeam (sch.levels.getD 0 default) := hlv 0 (by omega)
  have hQ : (lvOf T.sch 0).queries = (sch.levels.getD 0 default).q := by
    rw [hseam.hsch, lvOf_schedOf (by omega)]
    rfl
  have hw : 2 ^ T.sch.k0 = 2 ^ ((shapeZK st sch extras).sched.levels.getD 0 default).lanes := by
    rw [hseam.hsch]
    show 2 ^ sch.k0 = 2 ^ (sch.levels.getD 0 default).lanes
    rw [hl0]
  have hmc := ((hreqs R (by rw [hR]; exact List.mem_cons_self)).1 hl).2
  have hrows : zp.proof.ligerito.initialProof.rows.size = R.positions.length := by
    simpa using (merkleCheck_rows hmc).1
  rw [hQ, hw, Claims.Flock.zkRepColsOf_ok hdec]
  exact Claims.Flock.slotOpensZ_entries out r.val _ _ _ R rest (by rw [he]; exact hR) hl hpos
    (by rw [hpos, Array.length_toList, positions_size _ hseam0]) hrows ho _

end FlockSoundness.Discharge.ZkExec
