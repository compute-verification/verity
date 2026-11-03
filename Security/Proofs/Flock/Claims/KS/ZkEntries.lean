import Proofs.Flock.Claims.KS.Entries
import Proofs.Flock.Soundness.Discharge.ZkExec.Verdict

/-!
# The masked reader's level-0 entries at a rep's run

`exists_ne_of_offW₀Z_sc` (`ZkOffSC`) needs, of a masked output, that its level-0 openings verify, sit below `2^d`, and
are, with their rows' witness lanes, the masked reader's entries. This file proves all three at `Zk.verifyRep`'s
acceptance, the masked `Entries`:

* `Claims.Flock.slotOpensZ_entries`: a masked rep whose first request is level 0's, at the positions `Flock.positions`
  draws from `coins`, with one opened row per position, opens at level 0 exactly the entries `zkRepCols` reads off
  `coins` and those rows, in order, at every width the rows are cut to.
* `Claims.Flock.zkRepColsOf_ok`: the masked reader's entries at a proof that decodes are `zkRepCols`'s at level 0, the
  coins of round `zkQueryRound` and the proof's level-0 rows, cut to the witness lanes.
* `FlockSoundness.Discharge.ZkExec.zk_rep_entries`: when `repProg` runs on a rep's stream and consumes every round, the
  compiled rep played with the rep's messages and the rounds' coins opens at level 0, on the witness lanes, the masked
  reader's entries at the rep's stream rounds and proof bytes (`zk_rep_reqs`, whose first request is level 0's at the
  coins of round `zkQueryRound`).
* `Claims.Flock.verified_slotOpensZ`, `Claims.Flock.verified_opens₀Z`: an accepted masked table's level-0 openings
  verify against its cap at `width₀` when both reps' level-0 requests have level 0's path length.
* `FlockSoundness.Discharge.ZkExec.zk_table_opens₀Z`: when both reps of a table pass `Zk.verifyRep`, every pair of the
  reps' results whose output at the proof files' openings accepts at a cap has every level-0 opening verifying against
  that cap at `width₀`, below `2^d`, and its level-0 openings' positions and witness lanes are the reps' `zkRepColsOf`
  entries, rep 0's then rep 1's (`zk_rep_entries`, `Claims.Flock.verified_opens₀Z`).
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

/-- **A rep's level-0 openings verify** against the output's cap at `width₀` when its openings are `OpensOKZ` and its
level-0 requests have paths of length `depth`. -/
theorem verified_slotOpensZ {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts)
    (out : OutZC 𝔽 D Sl) (r Q depth : ℕ)
    (hok : OpensOKZ Hs.H Hs.E Hs.Lf (width₀ T) out.cap₀ r (if r = 0 then out.pend.1 else out.pend.2).lig
      out.opens.rows out.opens.salts)
    (h0 : ∀ R ∈ (if r = 0 then out.pend.1 else out.pend.2).lig.reqs, R.level = 0 → R.depth = depth) :
    ∀ x ∈ slotOpensZ out r 0 Q, VerifiedZ Hs.H Hs.E Hs.Lf out.cap₀ (width₀ T) depth x := by
  intro x hx
  unfold slotOpensZ at hx
  split at hx
  · simp at hx
  · rename_i req hreq
    have hlv : req.level = 0 := by simpa using List.find?_some hreq
    have hmem := List.mem_of_find?_eq_some hreq
    have hd := h0 req hmem hlv
    obtain ⟨cap, hcap, hall⟩ := hok req hmem
    rw [hlv] at hcap hall
    obtain rfl : out.cap₀ = cap := by simpa [Model.capOf] using hcap
    obtain ⟨pj, hpj, rfl⟩ := List.mem_map.1 hx
    have hg := List.mem_zipIdx_iff_getElem?.1 hpj
    rw [List.getElem?_take] at hg
    split_ifs at hg with hq
    have hjl : pj.2 < req.positions.length := (List.getElem?_eq_some_iff.1 hg).1
    have hpos : req.positions.getD pj.2 0 = pj.1 := by simp [List.getD, hg]
    obtain ⟨h1, h2⟩ := hall pj.2 hjl
    rw [ite_eq_left_of_eq_true _ _ (eq_true rfl)] at h1
    rw [hd, hpos] at h2
    exact ⟨h1, h2⟩

/-- **An accepted masked table's level-0 openings verify** against its cap at `width₀` and level 0's path length, when
both reps' level-0 requests have it: acceptance is both reps' `OpensOKZ` at the cap. -/
theorem verified_opens₀Z {D Sl : Type} {mPts : ℕ} (Hs : Hashes 𝔽 D Sl) (T : TabZK 𝔽 mPts) (cap : Model.Cap D)
    (pend : PendZ 𝔽 D Sl × PendZ 𝔽 D Sl) (os : OpensZ 𝔽 D Sl) (hacc : (outZC Hs T cap pend os).accepted = true)
    (h0 : ∀ r < 2, ∀ R ∈ (if r = 0 then pend.1 else pend.2).lig.reqs, R.level = 0 →
      R.depth = Model.pathLen (lvOf T.sch 0)) :
    ∀ x ∈ opens₀Z T.sch (outZC Hs T cap pend os),
      VerifiedZ Hs.H Hs.E Hs.Lf cap (width₀ T) (Model.pathLen (lvOf T.sch 0)) x := by
  have hok := hacc
  simp only [outZC, Bool.and_eq_true, decide_eq_true_eq] at hok
  intro x hx
  rcases List.mem_append.1 hx with hx | hx
  · exact verified_slotOpensZ Hs T _ 0 _ _ hok.2.2.2.1 (h0 0 (by norm_num)) x hx
  · exact verified_slotOpensZ Hs T _ 1 _ _ hok.2.2.2.2 (h0 1 (by norm_num)) x hx

end Claims.Flock

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model

local notation "𝔽" => FlockLevel3.GF128

/-- **A masked rep's level-0 openings are its deployed entries** (live PROTOCOL.md §7). When rep `r`'s transcript
program runs on its stream's rounds and consumes every round, at a compiled table at the executable's dimensions whose
extra claims are the executable's, and the rep's proof bytes decode to its proof, every output whose rep-`r` result is
a play of the compiled rep with the rep's messages and the rounds' coins, and whose rep-`r` level-0 openings are the
proof's level-0 rows, requests level 0 at its path length and opens at level 0, each row cut to its first `2^k0`
symbols, the masked reader's entries at the rep's stream rounds and proof bytes. -/
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
    (∀ R ∈ (if r.val = 0 then out.pend.1 else out.pend.2).lig.reqs, R.level = 0 →
      R.depth = Model.pathLen (lvOf T.sch 0)) ∧
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
  refine ⟨fun R' hR' hl' => ?_, ?_⟩
  · rw [he] at hR'
    rw [((hreqs R' hR').1 hl').1, hseam.hsch, lvOf_schedOf (by omega)]
  rw [hQ, hw, Claims.Flock.zkRepColsOf_ok hdec]
  exact Claims.Flock.slotOpensZ_entries out r.val _ _ _ R rest (by rw [he]; exact hR) hl hpos
    (by rw [hpos, Array.length_toList, positions_size _ hseam0]) hrows ho _

/-- **At `verify --zk`'s acceptance of a table, its level-0 openings are its reps' masked entries** (live PROTOCOL.md
§1–§7). When both reps of table `i` pass `Zk.verifyRep` with one cap, at a compiled table at the executable's
dimensions whose extra claims at the link points `pts` are the session's, on well-formed streams, every pair of the
reps' results, played with their proof files' messages and their streams' coins, whose output at the openings `opnZ`
reads off the proof files accepts at a cap has every level-0 opening verifying against that cap at `width₀` and level
0's path length, below `2^d`, and its level-0 openings' positions and witness lanes are the reps' `zkRepColsOf`
entries, rep 0's then rep 1's, at the table's shape, the reps' stream rounds and the proof files. -/
theorem zk_table_opens₀Z {n mPts : ℕ} {Reg : Type} (T : Finset (Fin n) → Reg → ZkSession.TabZK 𝔽 mPts)
    (S : Finset (Fin n)) (R : Reg) (pts : Fin 2 → Fin mPts → 𝔽) (spec : SessionSpec) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (sess : RunRecord) (i : ℕ) (b₀ b₁ c : ByteArray) (N : ℕ)
    (hN : (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts).length = N)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam (T S R) st sch tPad)
    (hext : ExtraRealizes st (T S R).S sess.points (Model.extraClaims (T S R).S (T S R).ptLocal mPts pts))
    (hwf : ∀ s ∈ sess.streams.toList, RoundsWF s.rounds)
    (h0 : Zk.verifyRep spec st sch tPad sess i 0 b₀ = .ok c) (h1 : Zk.verifyRep spec st sch tPad sess i 1 b₁ = .ok c)
    (s₀ s₁ : StreamRec) (zp₀ zp₁ : Zk.ZkProof)
    (hs₀ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 0) =
      some s₀)
    (hs₁ : sess.streams.find? (fun x => x.name == toString spec.tables[i]! ++ toString "/rep" ++ toString 1) =
      some s₁)
    (hd₀ : Zk.decode (shapeZKn st sch N) b₀ = .ok zp₀) (hd₁ : Zk.decode (shapeZKn st sch N) b₁ = .ok zp₁)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8) × ZkLink.PendZ 𝔽 Dig (List UInt8)) (cap : Cap Dig)
    (hp₀ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 0) (repMsgsZ st sch (2 + N) tPad zp₀)
      (s₀.rounds.toList.map (·.coins)) pend.1)
    (hp₁ : Plays fOf (ZkLink.repZCAt execArith hs512 (T S R) pts 1) (repMsgsZ st sch (2 + N) tPad zp₁)
      (s₁.rounds.toList.map (·.coins)) pend.2)
    (hacc : (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])).accepted = true) :
    (∀ x ∈ ZkLink.opens₀Z (T S R).sch (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])),
      ZkLink.VerifiedZ H512 enc512 leaf512 cap (ZkLink.width₀ (T S R)) (Model.pathLen (lvOf (T S R).sch 0)) x ∧
        x.1 < 2 ^ (sch.levels.getD 0 default).d) ∧
    (ZkLink.opens₀Z (T S R).sch (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁]))).map
        (fun x => (x.1, x.2.1.take (2 ^ (T S R).sch.k0))) =
      Claims.Flock.zkRepColsOf (shapeZKn st sch N) s₀.rounds b₀ ++
        Claims.Flock.zkRepColsOf (shapeZKn st sch N) s₁.rounds b₁ := by
  obtain ⟨-, -, -, -, -, hdim, hlv, -, -⟩ := zk_schedule_seams _ _ _ hsch
  obtain ⟨-, h2, -, -⟩ := hdim
  have hseam0 : LevelSeam (sch.levels.getD 0 default) := hlv 0 (by omega)
  obtain ⟨s₀', ex, zp₀', tr₀, hs₀', hx₀, hd₀', -, -, hrun₀, hk₀, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 0 b₀ c h0
  obtain ⟨s₁', ex₁, zp₁', tr₁, hs₁', hx₁, hd₁', -, -, hrun₁, hk₁, -, -⟩ :=
    zk_verifyRep_ok spec st sch tPad sess i 1 b₁ c h1
  obtain rfl : ex = ex₁ := Except.ok.inj (hx₀.symm.trans hx₁)
  obtain ⟨hsz, hrel⟩ := hext ex hx₀
  have hsh : shapeZK st sch ex = shapeZKn st sch N := by
    unfold shapeZK shapeZKn; rw [hsz, hN]
  obtain rfl : s₀' = s₀ := Option.some.inj (hs₀'.symm.trans hs₀)
  obtain rfl : s₁' = s₁ := Option.some.inj (hs₁'.symm.trans hs₁)
  have hz₀ := hd₀'
  have hz₁ := hd₁'
  rw [hsh] at hz₀ hz₁
  obtain rfl : zp₀' = zp₀ := Except.ok.inj (hz₀.symm.trans hd₀)
  obtain rfl : zp₁' = zp₁ := Except.ok.inj (hz₁.symm.trans hd₁)
  have hn : ex.size = N := hsz.trans hN
  rw [← hn] at hp₀ hp₁
  obtain ⟨hro, -, -, -⟩ := opnZ_pair hd₀ hd₁
  obtain ⟨hdp₀, he₀⟩ := zk_rep_entries hs512 (T S R) pts 0 st sch tPad ex zp₀' s₀'.rounds _ tr₀ b₀ hsch hseam hsz
    hrel (hwf s₀' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₀))) hrun₀ hk₀ hd₀'
    (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) hp₀
    (fun j => by show ((opnZ st sch N #[b₀, b₁]).rows 0 0 j).1 = _; rw [hro]; simp [opensOf, openRow, openAt])
  obtain ⟨hdp₁, he₁⟩ := zk_rep_entries hs512 (T S R) pts 1 st sch tPad ex zp₁' s₁'.rounds _ tr₁ b₁ hsch hseam hsz
    hrel (hwf s₁' (Array.mem_toList_iff.2 (Array.mem_of_find?_eq_some hs₁))) hrun₁ hk₁ hd₁'
    (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁])) hp₁
    (fun j => by show ((opnZ st sch N #[b₀, b₁]).rows 1 0 j).1 = _; rw [hro]; simp [opensOf, openRow, openAt])
  rw [hsh] at he₀ he₁
  have hent : (ZkLink.opens₀Z (T S R).sch (ZkLink.outZC hs512 (T S R) cap pend (opnZ st sch N #[b₀, b₁]))).map
      (fun x => (x.1, x.2.1.take (2 ^ (T S R).sch.k0))) =
      Claims.Flock.zkRepColsOf (shapeZKn st sch N) s₀'.rounds b₀ ++
        Claims.Flock.zkRepColsOf (shapeZKn st sch N) s₁'.rounds b₁ := by
    unfold ZkLink.opens₀Z
    rw [List.map_append]
    exact congrArg₂ (· ++ ·) he₀ he₁
  refine ⟨fun x hx => ⟨?_, ?_⟩, hent⟩
  · refine Claims.Flock.verified_opens₀Z hs512 (T S R) cap pend _ hacc (fun r hr R hR hlR => ?_) x hx
    rcases r with _ | _ | r
    · exact hdp₀ R hR hlR
    · exact hdp₁ R hR hlR
    · omega
  · have hm := List.mem_map_of_mem (f := fun x => (x.1, x.2.1.take (2 ^ (T S R).sch.k0))) hx
    rw [hent, Claims.Flock.zkRepColsOf_ok hd₀, Claims.Flock.zkRepColsOf_ok hd₁] at hm
    rcases List.mem_append.1 hm with hm | hm
    · obtain ⟨e, he, hxe⟩ := List.mem_map.1 hm
      rw [show x.1 = e.1 from (congrArg Prod.fst hxe).symm]
      exact Claims.Flock.lt_of_mem_positions hseam0 (Claims.Flock.fst_mem_positions_of_mem_zkRepCols he)
    · obtain ⟨e, he, hxe⟩ := List.mem_map.1 hm
      rw [show x.1 = e.1 from (congrArg Prod.fst hxe).symm]
      exact Claims.Flock.lt_of_mem_positions hseam0 (Claims.Flock.fst_mem_positions_of_mem_zkRepCols he)

end FlockSoundness.Discharge.ZkExec
