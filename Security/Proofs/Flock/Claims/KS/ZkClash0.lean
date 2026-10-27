import Proofs.Flock.Claims.KS.ZkRows
import Proofs.Flock.Claims.KS.ZkStep

/-!
# The masked view's self-clash at level 0

`ZkViewClash` splits along `SelfClashZ`'s three cases: a clash of two level-0 openings at the witness cap (`Clash₀Z`),
one past level 0 at a rep's recorded cap (`LaterClashZ`), and a pads clash (`PadsZ`); `ZkViewClashQ` is
`ZkViewClash` at one case (`zkViewClash_of_Q`). A level-0 clash compares whole opened rows, the witness lanes with both
reps' extra lanes, so where the reader's columns cut the rows, it is a collision in the run's log by the run's rows:

* `zkRawRowsJ`, `rawRows_zk`: the rows table `j`'s reps open in an accepted masked record, every one and each whole,
  are its reps' `zkRepRowsOf` rows on the live session's rep streams and proof files (`cols_zk` with whole rows and
  no `distinctBelow`).
* `zk_rowJ_raw`, `zk_rowClash_tiedJ`: each such row is a Merkle opening of table `j`'s level 0 against its `root_B`,
  every SHA-512 input of it in the run's log (`zk_openingJ_raw` at whole rows), so two at one position that differ
  collide in that log; two accepted runs share `P`'s first message, so `root_B`, and a row of each at one position that
  differ collide in the two logs (`zk_rowClash_tiedJ₂`).
* `zkViewRows₀`, `zkViewClashQ₀`: at the run's view the deployed decoders are right at `OpensRows`
  (`decodesTableR_deploy`), whose rows at `wrS`'s transcript are the run's (`recordsLiveAtS_wrS`, `rawRows_zk`), so a
  level-0 self-clash there is two of the run's rows at one position that differ, a collision in its log.
* `zk_KS_of_later`, `zk_tableBoundJOnAccR_of_later`, `zk_KSFlock_of_later`: so masked KS and the recursion's chain
  rest on the fork bounds and the view's clashes past level 0 and at the pads.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Rows

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody

theorem zkRepRowsOf_congr (sh : Flock.Shape) {a b : Array Flock.Round} (bytes : ByteArray)
    (h : ∀ j : ℕ, a[j]?.map (fun rd : Flock.Round => rd.coins) = b[j]?.map (fun rd : Flock.Round => rd.coins)) :
    Claims.Flock.zkRepRowsOf sh a bytes = Claims.Flock.zkRepRowsOf sh b bytes := by
  unfold Claims.Flock.zkRepRowsOf
  rw [h]

theorem getD_ofFnJ {α : Type} {n : ℕ} (f : Fin n → α) (d : α) (p : Fin n) : (List.ofFn f).getD p d = f p := by
  rw [List.getD_eq_getElem _ _ (by simp), List.getElem_ofFn]

/-- **The rows table `j`'s reps open in a masked record**, every one and each whole: `zkRawColsJ` with no row cut. -/
def zkRawRowsJ (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) :=
  match runOf true s j with
  | some r => r.reps.flatMap fun y => Claims.Flock.zkRepRowsOf r.shape y.1 y.2
  | none => []

namespace Gen

variable [S : ZkSetup]

/-- **The rows table `j`'s reps open in a masked record**, every one and each whole: `zkRawColsJ` with no row cut. -/
def zkRawRowsJ (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) :=
  match runOf true s j with
  | some r => r.reps.flatMap fun y => Claims.Flock.zkRepRowsOf r.shape y.1 y.2
  | none => []

/-- **An accepted masked run's rows of table `j`**: where the run's last message `f` sets up at `(spec, sts)` with two
link points, the masked verdict accepts its record and proof files at any hashes and `liveOf` reads the record, the
rows table `j`'s reps open are `zkRepRowsOf` at `shapeZKn` with `2 · |regions|` extra claims, on the live session's
rep streams `2j`, `2j + 1` and proof files (`cols_zk` with whole rows). -/
theorem rawRows_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) :
    zkRawRowsJ seen j =
      Claims.Flock.zkRepRowsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j) #[]) (f.proofs.getD (2 * j) ByteArray.empty) ++
        Claims.Flock.zkRepRowsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j + 1) #[]) (f.proofs.getD (2 * j + 1) ByteArray.empty) := by
  obtain ⟨hsz, hall⟩ := rep_zk hs hv hlv
  obtain ⟨sess, sch', tPad', stream, c, hpts, hpn, hf', hrep, -, -⟩ := hall (2 * j) (by omega)
  obtain ⟨-, extras, -, hex, -⟩ := zkVerifyRepWith_rounds hs hf' hrep
  have hj' : sts[2 * j / 2]! = sts[j] := by
    rw [show 2 * j / 2 = j by omega]
    exact getElem!_pos sts j hj
  rw [hj', hpts] at hex
  have hx : extras.size = 2 * sts[j].regions.size := by
    rw [extraClaims_size (st := sts[j]) hex, ← hpts, hpn, hp2, Nat.mul_comm]
  have hzs : Claims.Flock.zkSched (Setup.ofCircuit sts[j]).m = (sch, tPad) := by
    simp only [Claims.Flock.zkSched]
    rw [show (Setup.ofCircuit sts[j]).m = sts[j].m from rfl, hsch]
  have hsh : Claims.Flock.zkProofShape (Setup.ofCircuit sts[j]).m (Setup.ofCircuit sts[j]).kLog (2 + extras.size)
      (Setup.ofCircuit sts[j]).merkle.digestLen (saltLenOf true (Setup.ofCircuit sts[j])) =
      ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
    simp only [Claims.Flock.zkProofShape, hzs, ZkExec.shapeZKn, saltLenOf, hx, ite_true]
  have hrun : runOf true seen j = some
      { m := (Setup.ofCircuit sts[j]).m, kLog := (Setup.ofCircuit sts[j]).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit sts[j]).merkle.digestLen, saltLen := saltLenOf true (Setup.ofCircuit sts[j]),
        reps := [(lv.streams.getD (2 * j) #[], f.proofs.getD (2 * j) ByteArray.empty),
                 (lv.streams.getD (2 * j + 1) #[], f.proofs.getD (2 * j + 1) ByteArray.empty)] } := by
    unfold runOf
    simp [hf, hset, Array.getElem?_eq_getElem hj, hlv, hex, Except.toOption]
  unfold zkRawRowsJ
  rw [hrun]
  simp only [Claims.Flock.ZkRun.shape, hsh, List.flatMap_cons, List.flatMap_nil, List.append_nil]

end Gen

/-- **An accepted masked run's rows of table `j`**: where the run's last message `f` sets up at `(spec, sts)` with two
link points, the masked verdict accepts its record and proof files at any hashes and `liveOf` reads the record, the
rows table `j`'s reps open are `zkRepRowsOf` at `shapeZKn` with `2 · |regions|` extra claims, on the live session's
rep streams `2j`, `2j + 1` and proof files (`cols_zk` with whole rows). -/
theorem rawRows_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) :
    zkRawRowsJ seen j =
      Claims.Flock.zkRepRowsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j) #[]) (f.proofs.getD (2 * j) ByteArray.empty) ++
        Claims.Flock.zkRepRowsOf (ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size))
          (lv.streams.getD (2 * j + 1) #[]) (f.proofs.getD (2 * j + 1) ByteArray.empty) :=
  Gen.rawRows_zk (S := S₀) (hs := hs) (seen := seen) (f := f) (ins := ins) (spec := spec) (sts := sts) (lv := lv)
    (j := j) (sch := sch) (tPad := tPad) (hf := hf) (hset := hset) (hlv := hlv) (hv := hv) (hj := hj) (hsch := hsch)
    (hp2 := hp2)

end Rows

section Raw

open Flock
open FlockSoundness.Refine (ebind_ok entries_eq_of_flat flat_data merkleCheck_cap)

namespace Gen

variable [S : ZkSetup]

/-- **An accepted masked run's reps open a row of table `j` only as a Merkle opening** of table `j`'s level 0 in its
own log: `zk_openingJ_raw` at each row whole. -/
theorem zk_rowJ_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    {j q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : (q, u) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf true x = some (s, spec, sts) ∧ sts[j]? = some st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) link.1.toArray[j]!
        (Claims.run sha512 P (V true) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := acc_within h
  generalize hL : (Claims.run sha512 P (V true) x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold zkRawRowsJ at hu
  split at hu
  swap
  · simp at hu
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eqJ hf hx hs hlv hrr
  obtain ⟨ρ, hρ, hp⟩ := List.mem_flatMap.1 hu
  obtain ⟨k, hk, rfl⟩ : ∃ k < 2,
      ρ = (lv.streams.getD (2 * j + k) #[], f.proofs.getD (2 * j + k) ByteArray.empty) := by
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hρ
    rcases hρ with rfl | rfl
    · exact ⟨0, by omega, rfl⟩
    · exact ⟨1, by omega, rfl⟩
  have hstsj : j < sts.size := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, -⟩
    exact h0
  have hstj : sts[j] = st := by
    rcases Array.getElem?_eq_some_iff.1 hst with ⟨h0, e⟩
    exact e
  have hj : j < (tabsOf sts).size := by simpa [tabsOf] using hstsj
  have htj : (tabsOf sts)[j] = Setup.ofCircuit st := by rw [tabsOf_getElem hj]; simp only [hstj]
  replace hv : Zk.verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs = .ok () := hv
  obtain ⟨sess, hdec, hsz, hreps⟩ :=
    FlockSoundness.ZkQuery.zkQueryRound_verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs hv
  obtain ⟨sch, tPad, hsch, hrep⟩ := hreps j hj
  obtain ⟨stream, extras', zp, hstream, hext', hzp, -, -, hmc, hcap⟩ := hrep k hk
  simp only [htj] at hsch hext' hzp hmc
  have hdec' : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes (L.map Prod.fst)) spec f.record) sess hdec
  obtain ⟨hrB, hpts, hlsz, hstreams⟩ := Claims.Bridge.Custody.decode_live hdec' hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : j < spec.tables.size := by rw [← hsz]; exact hj
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * j + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hrowsEq : Claims.Flock.zkRepRowsOf
      (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
      (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) =
      Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows := by
    have e1 : lv.streams.getD (2 * j + k) #[] = lv.streams[2 * j + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : f.proofs.getD (2 * j + k) ByteArray.empty = f.proofs[2 * j + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    have hzs : Claims.Flock.zkSched (Setup.ofCircuit st).m = (sch, tPad) := by
      simp only [Claims.Flock.zkSched]
      rw [hsch]
    have hsh : Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)) =
        FlockSoundness.Discharge.ZkExec.shapeZK (Setup.ofCircuit st) sch extras := by
      simp only [Claims.Flock.zkProofShape, hzs, FlockSoundness.Discharge.ZkExec.shapeZK, saltLenOf, ite_true]
    rw [e1, e2, hsh]
    refine (zkRepRowsOf_congr _ _ fun i => ?_).trans (Claims.Flock.zkRepRowsOf_ok hzp)
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  have hp2 : (q, u) ∈ Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows := by
    rw [← hrowsEq]
    exact hp
  obtain ⟨i, hi, -, hiq, hiu⟩ := mem_zkRepCols hp2
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := by
    have := checks_hm96 hc (setupOf_some hs).2 (setupOf_zk hs) j hstsj
    rw [hstj] at this
    exact this
  rw [hms] at hmc
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) MerkleScheme.hm96Sha512)
    MerkleScheme.hm96Sha512 _ _ _ _ _ _ _ _ () hmc
  rw [opsWith_hm96] at hruns
  obtain ⟨hpure, hlog⟩ := merkleCheck_logs hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  refine ⟨s, spec, sts, st, sch, tPad, link, zp.proof.ligerito.initialProof.rows[i]!, hs, hst, hsch, hlink,
    ⟨Or.inl rfl, zp.proof.cap, zp.proof.ligerito.initialProof.salts.getD i ByteArray.empty,
      sibsOf zp.proof.ligerito.initialProof.siblings
        ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) i,
      hcs, hcw, ?_, ?_, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i, ?_⟩, hiu⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hiq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi
  · rw [opensInsOf_hm96, ← hiq]
    exact hlog i hi

end Gen

/-- **An accepted masked run's reps open a row of table `j` only as a Merkle opening** of table `j`'s level 0 in its
own log: `zk_openingJ_raw` at each row whole. -/
theorem zk_rowJ_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    {j q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : (q, u) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf true x = some (s, spec, sts) ∧ sts[j]? = some st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD 0 default)
        (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) link.1.toArray[j]!
        (Claims.run sha512 P (V true) x ω r).vLog q row ∧
      u = row.toList.map FlockLevel3.GF128.of :=
  Gen.zk_rowJ_raw (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (j := j) (q := q) (u := u) (h := h) (hu := hu)

namespace Gen

variable [S : ZkSetup]

/-- **Two different rows two accepted masked runs open at one position collide in their logs**: the runs share `P`'s
first message, so table `j`'s level-0 root, and a row of each at one position that differ give a collision among the
two runs' verifier calls, at SHA-512. -/
theorem zk_rowClash_tiedJ₂ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r r' : Fin ((V true).k x) → Coin} {j q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (h' : (Claims.run sha512 P (V true) x ω r').out = .acc)
    (hu : (q, u) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j)
    (hv : (q, v) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r').seen j) (huv : u ≠ v) :
    Claims.TiedColl sha512
      ((Claims.run sha512 P (V true) x ω r).vLog ++ (Claims.run sha512 P (V true) x ω r').vLog) := by
  obtain ⟨s, spec, sts, st, sch, tPad, link, row, hs, hst, hsch, hlink, ho, hu'⟩ := zk_rowJ_raw h hu
  obtain ⟨s', spec', sts', st', sch', tPad', link', row', hs', hst', hsch', hlink', ho', hv'⟩ := zk_rowJ_raw h' hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e])

end Gen

/-- **Two different rows two accepted masked runs open at one position collide in their logs**: the runs share `P`'s
first message, so table `j`'s level-0 root, and a row of each at one position that differ give a collision among the
two runs' verifier calls, at SHA-512. -/
theorem zk_rowClash_tiedJ₂ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r r' : Fin ((V true).k x) → Coin} {j q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (h' : (Claims.run sha512 P (V true) x ω r').out = .acc)
    (hu : (q, u) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j)
    (hv : (q, v) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r').seen j) (huv : u ≠ v) :
    Claims.TiedColl sha512
      ((Claims.run sha512 P (V true) x ω r).vLog ++ (Claims.run sha512 P (V true) x ω r').vLog) :=
  Gen.zk_rowClash_tiedJ₂ (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (r' := r') (j := j) (q := q) (u := u)
    (v := v) (h := h) (h' := h') (hu := hu) (hv := hv) (huv := huv)

namespace Gen

variable [S : ZkSetup]

/-- **Two different rows an accepted masked run opens at one position collide in its own log**: two rows table `j`'s
reps open at one position that differ give a collision among the run's verifier calls, at SHA-512. -/
theorem zk_rowClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : (q, u) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j)
    (hv : (q, v) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j) (huv : u ≠ v) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog :=
  tiedColl_of_append_self (zk_rowClash_tiedJ₂ h h hu hv huv)

end Gen

/-- **Two different rows an accepted masked run opens at one position collide in its own log**: two rows table `j`'s
reps open at one position that differ give a collision among the run's verifier calls, at SHA-512. -/
theorem zk_rowClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : (q, u) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j)
    (hv : (q, v) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j) (huv : u ≠ v) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog :=
  Gen.zk_rowClash_tiedJ (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (j := j) (q := q) (u := u) (v := v)
    (h := h) (hu := hu) (hv := hv) (huv := huv)

end Raw

section Clash

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

/-- **`SelfClashZ`'s level-0 case** at SHA-512: two of the output's level-0 openings verify at one position against the
witness cap with different rows. -/
def Clash₀Z (m : ℕ) (T : TabZK 𝔽 m) (out : OutZC 𝔽 Refine.Dig (List UInt8)) : Prop :=
  ClashW ZkExec.hs512.H ZkExec.hs512.E ZkExec.hs512.Lf out.cap₀ (width₀ T) (Model.pathLen (lvOf T.sch 0))
    (opens₀Z T.sch out)

/-- **`SelfClashZ`'s case past level 0** at SHA-512: two of a rep's openings at a level `ℓ ≥ 1` verify at one position
against the rep's recorded cap there with different rows. -/
def LaterClashZ (m : ℕ) (T : TabZK 𝔽 m) (out : OutZC 𝔽 Refine.Dig (List UInt8)) : Prop :=
  ∃ r < 2, ∃ ℓ, 1 ≤ ℓ ∧ ∃ cap, (if r = 0 then out.pend.1 else out.pend.2).lig.caps[ℓ - 1]? = some cap ∧
    Exec.ClashL ZkExec.hs512.H ZkExec.hs512.E ZkExec.hs512.Lf cap (lvOf T.sch ℓ).lanes
      (Model.pathLen (lvOf T.sch ℓ)) (slotOpensZ out r ℓ (lvOf T.sch ℓ).queries)

/-- **`SelfClashZ`'s pads case** at SHA-512: two of a rep's pads openings, both drawn at one position, verify there to
one root with different pairs. -/
def PadsZ (m : ℕ) (T : TabZK 𝔽 m) (out : OutZC 𝔽 Refine.Dig (List UInt8)) : Prop :=
  PadsClashAtZ ZkExec.hs512 T.N T.q out

/-- **`ZkViewClash` at one case of the self-clash**: at every masked statement in scope that sets up, at its draw
`d`, every prover and tape, every table `j` and table of full-width columns, and every coins at which `V true` accepts
`P` deciding a record, a decoded output of table `j` at the record's live outcome in `Q` (`LiveSelfClashZ` at
`viewOf` with `Q` for `SelfClashZ`) gives two different columns the run's reps open at one position of some table, or
a collision among the run's verifier calls. -/
def ZkViewClashQ (x : Claims.Msg) (Q : (m : ℕ) → TabZK 𝔽 m → OutZC 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (_ : parse true Readable.relation x = some s) (_ : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ r : Fin ((V true).k x) → Coin, (Claims.run sha512 P (V true) x ω r).out = .acc →
    ∀ s' spec' sts' link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s' spec' sts' link c₁ mids f →
      (∃ a, Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs j => Exec.capOfRoot rs[j.val]!)
          (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a ∧
        Q spec.mPts (zkTabs h0 L hz hk hlay j) (a.getD j dfltOutZC)) →
        (∃ (j' q : ℕ) (a b : List 𝔽), (q, a) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧
          (q, b) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧ a ≠ b) ∨
          Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog

namespace Gen

variable [S : ZkSetup]

/-- **`ZkViewClash` at one case of the self-clash**: at every masked statement in scope that sets up, at its draw
`d`, every prover and tape, every table `j` and table of full-width columns, and every coins at which `V true` accepts
`P` deciding a record, a decoded output of table `j` at the record's live outcome in `Q` (`LiveSelfClashZ` at
`viewOf` with `Q` for `SelfClashZ`) gives two different columns the run's reps open at one position of some table, or
a collision among the run's verifier calls. -/
def ZkViewClashQ (x : Claims.Msg) (Q : (m : ℕ) → TabZK 𝔽 m → OutZC 𝔽 Refine.Dig (List UInt8) → Prop) : Prop :=
  ∀ (s : Claims.Flock.Stmt) (_ : parse true Readable.relation x = some s) (_ : Claims.Flock.InScope true s)
    (ins : Ins) (spec : SessionSpec) (sts : Array Stmt) (_ : setupOf true x = some (ins, spec, sts))
    (h0 : 0 < sts.size) (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    (d : Lean.Json) (_ : ins.read.draw = some d) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (j : ℕ)
    (_ : j < (tables true x).length) (C : Fin (wordLenJ true x j) → List 𝔽), (∀ p, wideJ true x j (C p)) →
    ∀ r : Fin ((V true).k x) → Coin, (Claims.run sha512 P (V true) x ω r).out = .acc →
    ∀ s' spec' sts' link c₁ mids f,
      act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
        .decide s' spec' sts' link c₁ mids f →
      (∃ a, Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
          (fun rs j => Exec.capOfRoot rs[j.val]!)
          (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
          (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a ∧
        Q spec.mPts (zkTabs h0 L hz hk hlay j) (a.getD j dfltOutZC)) →
        (∃ (j' q : ℕ) (a b : List 𝔽), (q, a) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧
          (q, b) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j' ∧ a ≠ b) ∨
          Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog

/-- **The view's clash from its three cases**: `SelfClashZ` is `Clash₀Z`, `LaterClashZ` or `PadsZ`. -/
theorem zkViewClash_of_Q {x : Claims.Msg} (h₀ : ZkViewClashQ x Clash₀Z) (h₁ : ZkViewClashQ x LaterClashZ)
    (h₂ : ZkViewClashQ x PadsZ) : ZkViewClash x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC r hacc s' spec' sts' link c₁ mids f hact
    ⟨a, ha, hq⟩
  rcases hq with hq | hq | hq
  · exact h₀ s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC r hacc s' spec' sts' link c₁ mids f hact
      ⟨a, ha, hq⟩
  · exact h₁ s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC r hacc s' spec' sts' link c₁ mids f hact
      ⟨a, ha, hq⟩
  · exact h₂ s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC r hacc s' spec' sts' link c₁ mids f hact
      ⟨a, ha, hq⟩

end Gen

/-- **The view's clash from its three cases**: `SelfClashZ` is `Clash₀Z`, `LaterClashZ` or `PadsZ`. -/
theorem zkViewClash_of_Q {x : Claims.Msg} (h₀ : ZkViewClashQ x Clash₀Z) (h₁ : ZkViewClashQ x LaterClashZ)
    (h₂ : ZkViewClashQ x PadsZ) : ZkViewClash x :=
  Gen.zkViewClash_of_Q (S := S₀) (x := x) (h₀ := h₀) (h₁ := h₁) (h₂ := h₂)

namespace Gen

variable [S : ZkSetup]

/-- **An accepting run's view opens the run's level-0 rows**: the reps' pendings at the view play the deployed decoders'
tapes, so table `j`'s output there is right at `OpensRows` (`decodesTableR_deploy`); its level-0 openings are the reps'
`zkRepRowsOf` rows at `wrS`'s transcript, which records the run's rep streams and proof files (`recordsLiveAtS_wrS`),
so they are the run's rows (`rawRows_zk`). -/
theorem zkViewRows₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} {r : Fin ((V true).k x) → Coin}
    (hacc : (Claims.run sha512 P (V true) x ω r).out = .acc) {s' spec' sts' link c₁ mids f}
    (hact : act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
      .decide s' spec' sts' link c₁ mids f)
    {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (hret : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a) :
    (opens₀Z (zkTabs h0 L hz hk hlay j).sch (a.getD j dfltOutZC)).map (fun x => (x.1, x.2.1)) =
      zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j := by
  obtain ⟨hV, pend, hpl, ha⟩ := hret
  subst a
  have e := (act_decide hact).symm.trans hset
  simp only [Option.some.injEq, Prod.mk.injEq] at e
  obtain ⟨e₁, e₂, e₃⟩ := e
  subst s' spec' sts'
  have hrun : Runs sha512 (decision true ins x link c₁ mids f) ()
      ((Claims.run sha512 P (V true) x ω r).vLog.map Prod.fst) := by
    obtain ⟨s₂, spec₂, sts₂, link₂, c₂, mids₂, f₂, hact₂, hrun⟩ := acc_decide hacc
    rw [hact] at hact₂
    cases hact₂
    exact hrun
  have hjJ : j < (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length := by
    rw [tabsJ_length]
    exact hj
  obtain ⟨jj, rfl⟩ : ∃ jj : Fin (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length, jj.val = j :=
    ⟨⟨j, hjJ⟩, rfl⟩
  obtain ⟨-, spec₂, sts₂, sess, sch, tPad, s₀, s₁, hj₂, hsA₂, hdec, hsch, hs₀, hs₁, hrows, -⟩ :=
    Tables.DecodesTable.ok_of_plays (decodesTableR_deploy hs hin hset h0 L hz hk hlay hd
      (viewOf _ spec.mPts spec link c₁ f) jj) ⟨hV.1, rfl⟩ (pend jj) (hpl jj 0) (hpl jj 1)
  rw [tabsJ_get] at hrows
  have hacc' := run_acc_next sha512 P (V true) x ω r hacc
  replace hacc' : ((next true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
      ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map
        (Claims.coinSeq r))).eval sha512).1 = .acc := hacc'
  obtain ⟨-, hc⟩ := (next_decide_sha512 hact).1 hacc'
  obtain ⟨hins, hsu⟩ := setupOf_some (act_decide hact)
  obtain ⟨hvr, -⟩ := decision_runs hsu hrun
  have hv' : Zk.verifyWithM (hashesIn ((Claims.run sha512 P (V true) x ω r).vLog.map Prod.fst)) spec
      (tabsOf sts) f.record f.proofs = .ok () := verdict_within hvr
  have hzk : ins.read.zk = true := readOf_zk (x := x) (by simp [readOf, hins])
  have hsA := setupAt_of_setup hsu hzk (publicInputs_none_of_checks hc)
  rw [hd] at hsA
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsA.symm.trans hsA₂))
  obtain ⟨hpr, hall⟩ := recordsLiveAtS_wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f) hV.1
  obtain ⟨-, hrs⟩ := hall spec sts sess hsA hdec
  obtain ⟨y₀, hy₀, hr₀⟩ := (hrs jj).2 ⟨0, by omega⟩
  obtain ⟨y₁, hy₁, hr₁⟩ := (hrs jj).2 ⟨1, by omega⟩
  obtain rfl : y₀ = s₀ := Option.some.inj (hy₀.symm.trans hs₀)
  obtain rfl : y₁ = s₁ := Option.some.inj (hy₁.symm.trans hs₁)
  obtain ⟨hx, lv, hlv, -⟩ := checks_ok hc
  have hf : lastFinal (Claims.run sha512 P (V true) x ω r).seen = some f := by
    have hl := act_decide_final hact
    rw [List.getLast?_map, Option.bind_map] at hl
    exact hl
  have hraw := rawRows_zk _ hf (by rw [hx]; exact act_decide hact) hlv hv' hj₂ hsch
    (HiddenExec.setupTables_pointsAny hsA).1
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (q : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[q]!).toList.map (withSha spec) →
      ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
        (lv.streams.getD q #[])[i]?.map (fun rd : Flock.Round => rd.coins) := by
    intro q y hy i
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  rw [hpr', zkRepRowsOf_congr _ _ (hco (2 * jj.val) _ hr₀), zkRepRowsOf_congr _ _ (hco (2 * jj.val + 1) _ hr₁),
    ← getD_empty f.proofs (2 * jj.val), ← getD_empty f.proofs (2 * jj.val + 1), ← hraw] at hrows
  rw [getD_ofFnJ]
  exact hrows

end Gen

/-- **An accepting run's view opens the run's level-0 rows**: the reps' pendings at the view play the deployed decoders'
tapes, so table `j`'s output there is right at `OpensRows` (`decodesTableR_deploy`); its level-0 openings are the reps'
`zkRepRowsOf` rows at `wrS`'s transcript, which records the run's rep streams and proof files (`recordsLiveAtS_wrS`),
so they are the run's rows (`rawRows_zk`). -/
theorem zkViewRows₀ {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} {r : Fin ((V true).k x) → Coin}
    (hacc : (Claims.run sha512 P (V true) x ω r).out = .acc) {s' spec' sts' link c₁ mids f}
    (hact : act true x ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (V true) x ω r).msgs.map Prod.fst).length).map (Claims.coinSeq r)) =
      .decide s' spec' sts' link c₁ mids f)
    {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (hret : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a) :
    (opens₀Z (zkTabs h0 L hz hk hlay j).sch (a.getD j dfltOutZC)).map (fun x => (x.1, x.2.1)) =
      zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j :=
  Gen.zkViewRows₀ (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (r := r) (hacc := hacc) (s' := s') (spec' := spec') (sts' := sts')
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (hact := hact) (a := a) (hret := hret)

namespace Gen

variable [S : ZkSetup]

/-- **A level-0 self-clash at an accepting run's view is a collision in its log**: its level-0 openings are the run's
rows (`zkViewRows₀`), and two of them at one position that differ collide in the run's log (`zk_rowClash_tiedJ`). -/
theorem zkViewClashQ₀ (x : Claims.Msg) : ZkViewClashQ x Clash₀Z := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C _ r hacc s' spec' sts' link c₁ mids f hact hret
  obtain ⟨a, ha, hcl⟩ := hret
  have hrows := zkViewRows₀ hs hin hset h0 L hz hk hlay hd hj hacc hact ha
  obtain ⟨u, hu, v, hv, huv, -, -, hne⟩ := hcl
  have hu' : (u.1, u.2.1) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j := by
    rw [← hrows]
    exact List.mem_map_of_mem hu
  have hv'' : (u.1, v.2.1) ∈ zkRawRowsJ (Claims.run sha512 P (V true) x ω r).seen j := by
    rw [huv, ← hrows]
    exact List.mem_map_of_mem hv
  exact Or.inr (zk_rowClash_tiedJ hacc hu' hv'' hne)

end Gen

/-- **A level-0 self-clash at an accepting run's view is a collision in its log**: its level-0 openings are the run's
rows (`zkViewRows₀`), and two of them at one position that differ collide in the run's log (`zk_rowClash_tiedJ`). -/
theorem zkViewClashQ₀ (x : Claims.Msg) : ZkViewClashQ x Clash₀Z :=
  Gen.zkViewClashQ₀ (S := S₀) (x := x)

end Clash

open Flock

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's fork bounds and the view's clashes past level 0 and at the pads at the statements some run
accepts: the view's clash at level 0 holds at every statement (`zkViewClashQ₀`, `zkViewClash_of_Q`;
`zk_KS_of_fork`). -/
theorem zk_KS_of_later (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hlater : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x LaterClashZ)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_fork hfork fun x hno => zkViewClash_of_Q (zkViewClashQ₀ x) (hlater x hno) (hpads x hno)

namespace Gen

variable [S : ZkSetup]

/-- **The masked record the claims read: KS**, at SHA-512 at the deployed relation, at every number of tables, from
the masked simulator's fork bounds and the view's clashes past level 0 and at the pads at the statements some run
accepts: the view's clash at level 0 holds at every statement (`zkViewClashQ₀`, `zkViewClash_of_Q`;
`zk_KS_of_fork`). -/
theorem zk_KS_of_later (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hlater : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x LaterClashZ)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    Claims.Flock.KS sha512 (deployZkTrim Readable.relation) :=
  zk_KS_of_fork hfork fun x hno => zkViewClash_of_Q (zkViewClashQ₀ x) (hlater x hno) (hpads x hno)

end Gen

end FlockDeploy.KS

namespace FlockDeploy.KS

open Flock
open Claims.Recursion

/-- **Every masked table's bound with escapes on accepted runs counted per round**, at full-width columns, from the
fork bounds and the view's clashes past level 0 and at the pads at the statements some run accepts
(`zk_tableBoundJOnAccR_of_fork`): `zk_KSFlockJR`'s premise. -/
theorem zk_tableBoundJOnAccR_of_later (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hlater : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x LaterClashZ)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) :
    TableBoundJOnAccR sha512 (deployZk Readable.relation) (wordLenJ true) (wideJ true) :=
  zk_tableBoundJOnAccR_of_fork hfork fun x hno => zkViewClash_of_Q (zkViewClashQ₀ x) (hlater x hno) (hpads x hno)

/-- **KS of the recursion over the masked record the claims read** at the deployed relation, at every number of
tables, from the fork bounds and the view's clashes past level 0 and at the pads at the statements some run accepts
(`zk_KSFlock_of_fork`). -/
theorem zk_KSFlock_of_later (D : Deploy Coin)
    (DO : Claims.Flock.Deploy Coin (List FlockLevel3.GF128) Claims.Flock.ZkTrace)
    (opening : Claims.Recursion.Opening) (hF : Claims.Flock.KS sha512 DO)
    (hfork : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkSimForkLe x)
    (hlater : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x LaterClashZ)
    (hpads : ∀ x, ¬ Claims.Flock.NoAcc sha512 (V true) x → ZkViewClashQ x PadsZ) (hRef : Refuses sha512 D)
    (hRounds : ∀ x, (deployZkTrim Readable.relation).V.k (xI D x) ≤ D.V.k x) (hN : SessionNums D DO)
    (hε : TableErrNonneg D) (hRd : ∀ decI, SessionRounds D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hS : SessionAccepts D) (hR : ∀ decI, SessionRuns sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hV : ∀ decI, InnerViewRun sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hT : ∀ decI, CallsTrue sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hL : ∀ decI, CallsLogColl sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hO : ∀ decI, CallsOpenPlayed sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO opening)
    (hAcc : ∀ decI, InnerAccepts sha512 D (Inner.ofFlock (deployZkTrim Readable.relation) decI) DO)
    (hN0 : CallsNil sha512 D) :
    KSFlock sha512 D (deployZkTrim Readable.relation) DO :=
  zk_KSFlock_of_fork D DO opening hF hfork
    (fun x hno => zkViewClash_of_Q (zkViewClashQ₀ x) (hlater x hno) (hpads x hno)) hRef hRounds hN hε hRd hS hR hV
    hT hL hO hAcc hN0

end FlockDeploy.KS
