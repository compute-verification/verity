import Proofs.Flock.Claims.KS.ZkFork
import Proofs.Flock.Claims.KS.PlainForkClash

/-!
# The masked live fork event's collision

`ZkForkClash` (`ZkFork`) asks that two accepting runs of `P` at the masked simulator's coins whose decoded live outcomes
have `ForkEvZ` at table `j`'s rep `r`'s level `ℓ + 1` collide among the verifier's calls in the two. This file proves it
with no hypothesis, as `PlainForkClash` does for the plain verifier. Each live outcome is its run's view
(`zkSimRunAt`), where the deployed decoders are right at `OpensRows` (`decodesTableR_deploy`): rep `r`'s recorded caps
are its decoded proof file's recursive caps, and its level-`ℓ + 1` openings its `zkLevelRowsOf` rows, the run's own
(`rawLevelRows_zk`), each an `Opening` in the run's log against the flattened `ℓ`-th recursive cap
(`zk_levelOpeningJ`). One recorded cap in both runs is one cap of 64-byte digests (`capDig_inj`), so one root, and two
rows opened at one position of it that differ collide in the joined logs (`tiedColl_of_openings`).

* `runOf_shape_zk`, `zk_levelOpeningJ`: an accepted masked run's table `j` is read at its `--zk` shape, and a row its
  rep `k` opens at a later level is an `Opening` in its log.
* `opRZ_eq_some`, `forkEvZ_entries`: `ForkEvZ` is one recorded cap and two openings of one position of it to
  different rows.
* `zkViewOpening`, `zkAccOpening`: at an accepting run's view, and at a decoded live outcome of the masked simulator,
  table `j`'s rep `r` records its proof's recursive caps and opens rows of them in the run's log.
* `zkForkClash`: `ZkForkClash` holds at every statement.
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Rows

open Flock FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **An accepted masked run's table `j` at its `--zk` shape**: the reader's table `j` is read at the shape
`Zk.verifyRep` decodes its proof files at, at `2 · |regions|` extra claims (`rawLevelRows_zk`'s `runOf`). -/
theorem runOf_shape_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) :
    ∃ rr, runOf true seen j = some rr ∧
      rr.shape = ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) := by
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
  exact ⟨_, hrun, hsh⟩

end Gen

/-- **An accepted masked run's table `j` at its `--zk` shape**: the reader's table `j` is read at the shape
`Zk.verifyRep` decodes its proof files at, at `2 · |regions|` extra claims (`rawLevelRows_zk`'s `runOf`). -/
theorem runOf_shape_zk (hs : Hashes Flock.V) {seen : Claims.Seen} {f : Final} {ins : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {lv : Live} {j : ℕ} {sch : Schedule} {tPad : ℕ} (hf : lastFinal seen = some f)
    (hset : setupOf true f.stmt = some (ins, spec, sts)) (hlv : liveOf spec f.record = .ok lv)
    (hv : Zk.verifyWithM hs spec (tabsOf sts) f.record f.proofs = .ok ()) (hj : j < sts.size)
    (hsch : Zk.schedule sts[j].m = .ok (sch, tPad)) (hp2 : spec.points = 2) :
    ∃ rr, runOf true seen j = some rr ∧
      rr.shape = ZkExec.shapeZKn (Setup.ofCircuit sts[j]) sch (2 * sts[j].regions.size) :=
  Gen.runOf_shape_zk (S := S₀) (hs := hs) (seen := seen) (f := f) (ins := ins) (spec := spec) (sts := sts) (lv := lv)
    (j := j) (sch := sch) (tPad := tPad) (hf := hf) (hset := hset) (hlv := hlv) (hv := hv) (hj := hj) (hsch := hsch)
    (hp2 := hp2)

namespace Gen

variable [S : ZkSetup]

/-- **A row one rep of an accepted masked run opens at a later level is an `Opening` in its log**: a level-`ℓ` row of
table `j`'s rep `k` is an `Opening` of its position against the flattened `ℓ − 1`-th recursive cap of the rep's proof
(its proof file decoded at the reader's shape), a cap of `2^cap` 64-byte digests, each SHA-512 input of it in the run's
verifier log (`zk_levelClash_tiedJ` at one row): its verdict accepts with SHA-512 answered within that log
(`acc_within`), so level `ℓ`'s Merkle check passed on the rep's rows at those positions (`zkLevels_verifyWithM`), and
the check, carried back to `OC`, logged them (`merkleCheck_logs`). -/
theorem zk_levelOpeningJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j k ℓ q : ℕ} {a : List 𝔽}
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (ha : (q, a) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ) :
    ∃ (rr : Claims.Flock.ZkRun) (f : Final) (sch : Schedule) (zp : Zk.ZkProof) (row : Array F128),
      runOf true (Claims.run sha512 P (V true) x ω r).seen j = some rr ∧ rr.shape.sched = sch ∧
      lastFinal (Claims.run sha512 P (V true) x ω r).seen = some f ∧
      Zk.decode rr.shape f.proofs[2 * j + k]! = .ok zp ∧ a = row.toList.map FlockLevel3.GF128.of ∧
      zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
      (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
        (Claims.run sha512 P (V true) x ω r).vLog q row := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := acc_within h
  generalize hL : (Claims.run sha512 P (V true) x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, -⟩ := Claims.Bridge.Custody.checks_ok hc
  unfold zkRawLevelRowsJ at ha
  split at ha
  swap
  · simp at ha
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOf_eqJ hf hx hs hlv hrr
  obtain ⟨hk, ha⟩ : k < 2 ∧
      (q, a) ∈ zkLevelRowsOf (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
        (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ := by
    rcases k with _ | _ | k
    · exact ⟨by omega, ha⟩
    · exact ⟨by omega, ha⟩
    · simp at ha
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
    zkLevels_verifyWithM (hashesIn (L.map Prod.fst)) spec (tabsOf sts) f.record f.proofs hv
  obtain ⟨sch, tPad, hsch, hrep⟩ := hreps j hj
  obtain ⟨stream, extras', zp, hstream, hext', hzp, hlev⟩ := hrep k hk
  simp only [htj] at hsch hext' hzp hlev
  have hdec' : Record.decode spec f.record = .ok sess := by
    rw [← With.decodeWith_std]
    exact (With.decodeWith_sim sub (sub_hashes (L.map Prod.fst)) spec f.record) sess hdec
  obtain ⟨-, hpts, -, hstreams⟩ := Claims.Bridge.Custody.decode_live hdec' hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : j < spec.tables.size := by rw [← hsz]; exact hj
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * j + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hzs : Claims.Flock.zkSched (Setup.ofCircuit st).m = (sch, tPad) := by
    simp only [Claims.Flock.zkSched]
    rw [hsch]
  have hsh : Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)) =
      FlockSoundness.Discharge.ZkExec.shapeZK (Setup.ofCircuit st) sch extras := by
    simp only [Claims.Flock.zkProofShape, hzs, FlockSoundness.Discharge.ZkExec.shapeZK, saltLenOf, ite_true]
  have hsched : (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st))).sched = sch := by
    rw [hsh]
    rfl
  have hℓ : 1 ≤ ℓ ∧ ℓ < sch.levels.size := by
    by_contra hn
    rw [zkLevelRowsOf, ite_eq_right (by rw [hsched]; exact hn)] at ha
    simp at ha
  obtain ⟨hent, hmc⟩ := hlev ℓ hℓ.1 hℓ.2
  have hrowsEq : zkLevelRowsOf (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
      (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
      (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
        (y.rounds[zkLevelRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size)
          (sch.levels.size - 1) ℓ]!).coins (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1 := by
    rw [← hent, hsh]
    have e1 : lv.streams.getD (2 * j + k) #[] = lv.streams[2 * j + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : f.proofs.getD (2 * j + k) ByteArray.empty = f.proofs[2 * j + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine zkLevelRowsOf_congr _ _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  rw [hrowsEq] at ha
  obtain ⟨i, hi, -, hiq, hia⟩ := mem_zkRepCols ha
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
  have hop : Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
      (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) L
      (positions (sch.levels.getD ℓ default) (y.rounds[zkLevelRound (Setup.ofCircuit st).m
        (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins)[i]!
      (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).1[i]! :=
    ⟨Or.inl rfl, zp.proof.ligerito.recursiveCaps[ℓ - 1]!,
      (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.2.getD i ByteArray.empty,
      sibsOf (openAt zp.proof.ligerito (sch.levels.size - 1) ℓ).2.1
        ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) i,
      hcs, hcw, rfl, merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i,
      by rw [opensInsOf_hm96]; exact hlog i hi⟩
  rw [hiq] at hop
  refine ⟨_, f, sch, zp, _, hrr, hsched, hf, ?_, hia, hcs, hcw, hop⟩
  show Zk.decode (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
    (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st))) f.proofs[2 * j + k]! = .ok zp
  rw [hsh]
  exact hzp

end Gen

/-- **A row one rep of an accepted masked run opens at a later level is an `Opening` in its log**: a level-`ℓ` row of
table `j`'s rep `k` is an `Opening` of its position against the flattened `ℓ − 1`-th recursive cap of the rep's proof
(its proof file decoded at the reader's shape), a cap of `2^cap` 64-byte digests, each SHA-512 input of it in the run's
verifier log (`zk_levelClash_tiedJ` at one row): its verdict accepts with SHA-512 answered within that log
(`acc_within`), so level `ℓ`'s Merkle check passed on the rep's rows at those positions (`zkLevels_verifyWithM`), and
the check, carried back to `OC`, logged them (`merkleCheck_logs`). -/
theorem zk_levelOpeningJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j k ℓ q : ℕ} {a : List 𝔽}
    (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (ha : (q, a) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen j k ℓ) :
    ∃ (rr : Claims.Flock.ZkRun) (f : Final) (sch : Schedule) (zp : Zk.ZkProof) (row : Array F128),
      runOf true (Claims.run sha512 P (V true) x ω r).seen j = some rr ∧ rr.shape.sched = sch ∧
      lastFinal (Claims.run sha512 P (V true) x ω r).seen = some f ∧
      Zk.decode rr.shape f.proofs[2 * j + k]! = .ok zp ∧ a = row.toList.map FlockLevel3.GF128.of ∧
      zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
      (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
      Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
        (Claims.run sha512 P (V true) x ω r).vLog q row :=
  Gen.zk_levelOpeningJ (S := S₀) (Ω := Ω) (P := P) (x := x) (ω := ω) (r := r) (j := j) (k := k) (ℓ := ℓ) (q := q)
    (a := a) (h := h) (ha := ha)

end Rows

section Event

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge

variable {D Sl : Type} [Inhabited D] [Inhabited Sl]

omit [Inhabited D] [Inhabited Sl] in
open Classical in
/-- **An opened position at a masked rep's cap is a verifying opening there**: where rep `r`'s level-`ℓ` openings at
`cap` open `p`, one of them is at `p`, opened to that row. -/
theorem opRZ_eq_some {H : List UInt8 → D} {E : Merkle.Enc (List 𝔽) D} {Lf : Merkle.Leaf (List 𝔽) Sl D H}
    {sch : Accounting.Schedule} {r ℓ : ℕ} {cap : Model.Cap D} {out : OutZC 𝔽 D Sl}
    {p : Fin (2 ^ (lvOf sch ℓ).logLen)} (h : (opRZ H E Lf sch r ℓ cap out p).isSome) :
    ∃ x ∈ slotOpensZ out r ℓ (lvOf sch ℓ).queries, x.1 = p.val ∧
      opRZ H E Lf sch r ℓ cap out p = some (rowFin (lvOf sch ℓ).lanes x.2.1) := by
  unfold opRZ Exec.opOfL at h ⊢
  rw [Option.isSome_map] at h
  obtain ⟨x, hx⟩ := Option.isSome_iff_exists.1 h
  have hmem := List.mem_of_mem_head? hx
  rw [List.mem_filter, decide_eq_true_eq] at hmem
  exact ⟨x, hmem.1, hmem.2.1, by rw [hx]; rfl⟩

/-- **The masked fork event's two entries**: `ForkEvZ` at table `j`'s rep `r`'s level `ℓ + 1` is one recorded cap `d`
in both outputs and an opening in each of one position to different rows. -/
theorem forkEvZ_entries {Hs : Hashes 𝔽 D Sl} {sch : Accounting.Schedule} {j r ℓ : ℕ} {a a' : List (OutZC 𝔽 D Sl)}
    (h : ForkEvZ Hs sch j r ℓ a a') :
    ∃ dc, CapAtZ r ℓ (a.getD j dfltOutZC) dc ∧ CapAtZ r ℓ (a'.getD j dfltOutZC) dc ∧
      ∃ x₁ ∈ slotOpensZ (a.getD j dfltOutZC) r (ℓ + 1) (lvOf sch (ℓ + 1)).queries,
      ∃ x₂ ∈ slotOpensZ (a'.getD j dfltOutZC) r (ℓ + 1) (lvOf sch (ℓ + 1)).queries,
        x₁.1 = x₂.1 ∧ x₁.2.1 ≠ x₂.2.1 := by
  obtain ⟨dc, hc₁, hc₂, p, hs₁, hs₂, hne⟩ := h
  obtain ⟨x₁, hx₁, hp₁, he₁⟩ := opRZ_eq_some (out := a.getD j dfltOutZC) hs₁
  obtain ⟨x₂, hx₂, hp₂, he₂⟩ := opRZ_eq_some (out := a'.getD j dfltOutZC) hs₂
  refine ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hp₁.trans hp₂.symm, fun he => hne ?_⟩
  show opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) dc (a.getD j dfltOutZC) p =
    opRZ Hs.H Hs.E Hs.Lf sch r (ℓ + 1) dc (a'.getD j dfltOutZC) p
  rw [he₁, he₂, he]

end Event

section View

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Custody

namespace Gen

variable [S : ZkSetup]

/-- **An accepting run's view records its reps' caps and opens rows of them in its log**: at the view the deployed
decoders are right at `OpensRows` (`decodesTableR_deploy`), so table `j`'s rep `r` records its decoded proof file's
recursive caps, and its openings at a level `ℓ ≥ 1` are its `zkLevelRowsOf` rows at `wrS`'s transcript, which records
the run's rep streams and proof files (`recordsLiveAtS_wrS`), so they are the run's level-`ℓ` rows
(`rawLevelRows_zk`), each an `Opening` in the run's log against the flattened `ℓ − 1`-th of those caps
(`zk_levelOpeningJ`, at the shape `runOf_shape_zk` reads). -/
theorem zkViewOpening {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
      (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a)
    (rr : Fin 2) :
    ∃ (hj' : j < sts.size) (sch : Schedule) (tPad : ℕ) (zp : Zk.ZkProof), Zk.schedule sts[j].m = .ok (sch, tPad) ∧
      (∀ ℓ dc, CapAtZ rr.val ℓ (a.getD j dfltOutZC) dc → Refine.capDig zp.proof.ligerito.recursiveCaps[ℓ]! = dc) ∧
      ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensZ (a.getD j dfltOutZC) rr.val ℓ
          (lvOf (zkTabs h0 L hz hk hlay j).sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
        ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
          zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
          (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
          Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
            (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (V true) x ω r).vLog q row := by
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
  rw [getD_ofFnJ]
  obtain ⟨-, spec₂, sts₂, sess, sch, tPad, s₀, s₁, hj₂, hsA₂, hdec, hsch, hs₀, hs₁, -, hlater, -, -, -, hcaps⟩ :=
    Tables.DecodesTable.ok_of_plays (decodesTableR_deploy hs hin hset h0 L hz hk hlay hd
      (viewOf _ spec.mPts spec link c₁ f) jj) ⟨hV.1, rfl⟩ (pend jj) (hpl jj 0) (hpl jj 1)
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
  have hp2 := (HiddenExec.setupTables_pointsAny hsA).1
  have hraw := rawLevelRows_zk _ hf (by rw [hx]; exact act_decide hact) hlv hv' hj₂ hsch hp2
  obtain ⟨rr₀, hrr₀, hsh₀⟩ := runOf_shape_zk _ hf (by rw [hx]; exact act_decide hact) hlv hv' hj₂ hsch hp2
  have hD : liveD spec f.record = lv := by simp [liveD, hlv]
  have hco : ∀ (q : ℕ) (y : StreamRec),
      y.rounds.toList = ((liveD spec f.record).streams[q]!).toList.map (withSha spec) →
      ∀ i : ℕ, y.rounds[i]?.map (fun rd : Flock.Round => rd.coins) =
        (lv.streams.getD q #[])[i]?.map (fun rd : Flock.Round => rd.coins) := by
    intro q y hy i
    rw [← Array.getElem?_toList, hy, hD, show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    simp [withSha, Option.map_map, Function.comp_def]
  have hpr' : (wrS ins.read.I d (viewOf _ spec.mPts spec link c₁ f)).proofs = f.proofs := hpr
  obtain ⟨zp, hzp, hcz⟩ := hcaps rr.val rr.isLt
  rw [hpr'] at hzp
  refine ⟨hj₂, sch, tPad, zp, hsch, fun ℓ dc hc => ?_, fun ℓ h1 q v hqv => ?_⟩
  · unfold CapAtZ at hc
    rw [hcz, List.getElem?_map] at hc
    by_cases hl : ℓ < sch.levels.size - 1
    · rw [List.getElem?_range hl] at hc
      exact Option.some.inj hc
    · rw [List.getElem?_eq_none (by simpa using hl)] at hc
      cases hc
  obtain ⟨e₀, e₁⟩ := hlater ℓ h1
  rw [tabsJ_get] at e₀ e₁
  have hraw' : (q, v) ∈ zkRawLevelRowsJ (Claims.run sha512 P (V true) x ω r).seen jj.val rr.val ℓ := by
    obtain ⟨_ | _ | k, hk2⟩ := rr
    · rw [hpr', zkLevelRowsOf_congr _ _ _ (hco (2 * jj.val) _ hr₀), ← getD_empty f.proofs (2 * jj.val)] at e₀
      rw [hraw 0 (by omega) ℓ, Nat.add_zero, ← e₀]
      exact hqv
    · rw [hpr', zkLevelRowsOf_congr _ _ _ (hco (2 * jj.val + 1) _ hr₁),
        ← getD_empty f.proofs (2 * jj.val + 1)] at e₁
      rw [hraw 1 (by omega) ℓ, ← e₁]
      exact hqv
    · omega
  obtain ⟨rr', f', sch', zp', row, hrr', hsch', hf', hzp', hrow, hcs, hcw, hop⟩ := zk_levelOpeningJ hacc hraw'
  obtain rfl := Option.some.inj (hrr₀.symm.trans hrr')
  obtain rfl := Option.some.inj (hf.symm.trans hf')
  obtain rfl : sch' = sch := by rw [← hsch', hsh₀]; rfl
  rw [hsh₀] at hzp'
  obtain rfl := Except.ok.inj (hzp.symm.trans hzp')
  exact ⟨row, hrow, hcs, hcw, hop⟩

end Gen

/-- **An accepting run's view records its reps' caps and opens rows of them in its log**: at the view the deployed
decoders are right at `OpensRows` (`decodesTableR_deploy`), so table `j`'s rep `r` records its decoded proof file's
recursive caps, and its openings at a level `ℓ ≥ 1` are its `zkLevelRowsOf` rows at `wrS`'s transcript, which records
the run's rep streams and proof files (`recordsLiveAtS_wrS`), so they are the run's level-`ℓ` rows
(`rawLevelRows_zk`), each an `Opening` in the run's log against the flattened `ℓ − 1`-th of those caps
(`zk_levelOpeningJ`, at the shape `runOf_shape_zk` reads). -/
theorem zkViewOpening {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
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
      (viewOf (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec.mPts spec' link c₁ f) a)
    (rr : Fin 2) :
    ∃ (hj' : j < sts.size) (sch : Schedule) (tPad : ℕ) (zp : Zk.ZkProof), Zk.schedule sts[j].m = .ok (sch, tPad) ∧
      (∀ ℓ dc, CapAtZ rr.val ℓ (a.getD j dfltOutZC) dc → Refine.capDig zp.proof.ligerito.recursiveCaps[ℓ]! = dc) ∧
      ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensZ (a.getD j dfltOutZC) rr.val ℓ
          (lvOf (zkTabs h0 L hz hk hlay j).sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
        ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
          zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
          (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
          Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
            (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (V true) x ω r).vLog q row :=
  Gen.zkViewOpening (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (r := r) (hacc := hacc) (s' := s') (spec' := spec') (sts' := sts')
    (link := link) (c₁ := c₁) (mids := mids) (f := f) (hact := hact) (a := a) (hret := hret) (rr := rr)

end View

section Live

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open FlockSoundness.Discharge.Composed (outJ repJ)
open Claims.Bridge.Live

namespace Gen

variable [S : ZkSetup]

/-- **A decoded live outcome of the masked simulator records an accepting run's caps and opens rows of them in its
log**: where the live verdict holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of
the record it decides (`zkSimRunAt`), and there table `j`'s rep `r` records its proof's recursive caps and opens rows of
them in the run's log (`zkViewOpening`). -/
theorem zkAccOpening {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (ha : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a) (rr : Fin 2) :
    ∃ (hj' : j < sts.size) (sch : Schedule) (tPad : ℕ) (zp : Zk.ZkProof), Zk.schedule sts[j].m = .ok (sch, tPad) ∧
      (∀ ℓ dc, CapAtZ rr.val ℓ (a.getD j dfltOutZC) dc → Refine.capDig zp.proof.ligerito.recursiveCaps[ℓ]! = dc) ∧
      ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensZ (a.getD j dfltOutZC) rr.val ℓ
          (lvOf (zkTabs h0 L hz hk hlay j).sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
        ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
          zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
          (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
          Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
            (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog q row := by
  have hacc : (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).out = .acc := by
    by_contra hn
    exact zkSimAbortAt x s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r' hn ha.1
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, -⟩ := acc_decide hacc
  have hv := zkSimRunAt x s hs hin ins spec sts hset h0 L hz hk hlay Ω P ω u r' hacc s' spec' sts' link c₁ mids f hact
  rw [hv] at ha
  exact zkViewOpening hs hin hset h0 L hz hk hlay hd hj hacc hact ha rr

end Gen

/-- **A decoded live outcome of the masked simulator records an accepting run's caps and opens rows of them in its
log**: where the live verdict holds, `P`'s run at `zkΦ`'s coins accepts (`zkSimAbortAt`), the outcome is the view of
the record it decides (`zkSimRunAt`), and there table `j`'s rep `r` records its proof's recursive caps and opens rows of
them in the run's log (`zkViewOpening`). -/
theorem zkAccOpening {x : Claims.Msg} {s : Claims.Flock.Stmt} (hs : parse true Readable.relation x = some s)
    (hin : Claims.Flock.InScope true s) {ins : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : setupOf true x = some (ins, spec, sts)) (h0 : 0 < sts.size)
    (L : (j : ℕ) → (h : j < sts.size) → Layout sts[j])
    (hz : ∀ j (h : j < sts.size), Zk.schedule sts[j].m = .ok (Claims.Flock.zkSched sts[j].m))
    (hk : ∀ j (h : j < sts.size), sts[j].c.kLog ≤ 27)
    (hlay : ∀ j (h : j < sts.size), ((L j h).model (Refine.regionsOf sts[j])).LinkLayout PT_LOCAL spec.mPts)
    {d : Lean.Json} (hd : ins.read.draw = some d) {Ω : Type} {P : Claims.Prover Ω} {ω : Ω} {j : ℕ}
    (hj : j < (tables true x).length) {C : Fin (wordLenJ true x j) → List 𝔽} (hC : ∀ p, wideJ true x j (C p))
    (u : zkTape x) (r' : Fin (zkN x + 2) → LCoin 65536) {a : List (OutZC 𝔽 Refine.Dig (List UInt8))}
    (ha : Claims.Bridge.RetOK (outJ ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (AccTxJ ins.read.I d j (zkRoot P x ω) C) (opnD sts _) (decD _) (fun _ _ => true) (fun _ _ _ => [])
      (fun rs j => Exec.capOfRoot rs[j.val]!)
      (repJ execArith ZkExec.hs512 (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)))
      (liveOut (fun _ => (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length) (fun _ => spec.mPts) zkN
        spec.dg (zkW (tabsJ (tables true x).length (zkTabs h0 L hz hk hlay)).length spec (zk_mPts_le hs hin hset))
        sha512 (zkSim P x spec sts) x (ω, u) r') a) (rr : Fin 2) :
    ∃ (hj' : j < sts.size) (sch : Schedule) (tPad : ℕ) (zp : Zk.ZkProof), Zk.schedule sts[j].m = .ok (sch, tPad) ∧
      (∀ ℓ dc, CapAtZ rr.val ℓ (a.getD j dfltOutZC) dc → Refine.capDig zp.proof.ligerito.recursiveCaps[ℓ]! = dc) ∧
      ∀ ℓ, 1 ≤ ℓ → ∀ q v, (q, v) ∈ (slotOpensZ (a.getD j dfltOutZC) rr.val ℓ
          (lvOf (zkTabs h0 L hz hk hlay j).sch ℓ).queries).map (fun x => (x.1, x.2.1)) →
        ∃ row : Array F128, v = row.toList.map FlockLevel3.GF128.of ∧
          zp.proof.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
          (∀ b ∈ zp.proof.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
          Opening MerkleScheme.hm96Sha512 (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
            (zp.proof.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty)
            (Claims.run sha512 P (V true) x ω (zkΦ x spec sts (u, r'))).vLog q row :=
  Gen.zkAccOpening (S := S₀) (x := x) (s := s) (hs := hs) (hin := hin) (ins := ins) (spec := spec) (sts := sts)
    (hset := hset) (h0 := h0) (L := L) (hz := hz) (hk := hk) (hlay := hlay) (d := d) (hd := hd) (Ω := Ω) (P := P)
    (ω := ω) (j := j) (hj := hj) (C := C) (hC := hC) (u := u) (r' := r') (a := a) (ha := ha) (rr := rr)

namespace Gen

variable [S : ZkSetup]

/-- **The masked live fork event's collision, with no hypothesis**: two accepting runs of `P` whose live outcomes
decode to `ForkEvZ` at table `j`'s rep `r`'s level `ℓ + 1` record one cap `d` there, the `capDig` of each run's
proof's `ℓ`-th recursive cap, so one cap of 64-byte digests (`capDig_inj`), and open one position of it to two
different rows, each an `Opening` against that cap in its run's log (`zkAccOpening`), which collide in the joined
logs (`tiedColl_of_openings`). -/
theorem zkForkClash (x : Claims.Msg) : ZkForkClash x := by
  intro s hs hin ins spec sts hset h0 L hz hk hlay d hd Ω P ω j hj C hC u r₁ r₂ r ℓ hq _ _
  obtain ⟨a₁, a₂, ha₁, ha₂, hF⟩ := hq
  obtain ⟨dc, hc₁, hc₂, x₁, hx₁, x₂, hx₂, hq, hne⟩ := forkEvZ_entries hF
  obtain ⟨hj₁, sch₁, tPad₁, zp₁, hsch₁, hcap₁, hop₁⟩ :=
    zkAccOpening hs hin hset h0 L hz hk hlay hd hj hC u r₁ ha₁ r
  obtain ⟨hj₂, sch₂, tPad₂, zp₂, hsch₂, hcap₂, hop₂⟩ :=
    zkAccOpening hs hin hset h0 L hz hk hlay hd hj hC u r₂ ha₂ r
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj (hsch₁.symm.trans hsch₂))
  have hd₁ := hcap₁ ℓ.val dc hc₁
  have hd₂ := hcap₂ ℓ.val dc hc₂
  obtain ⟨row₁, hrow₁, hcs₁, hcw₁, ho₁⟩ := hop₁ (ℓ.val + 1) (by omega) x₁.1 x₁.2.1 (List.mem_map_of_mem hx₁)
  obtain ⟨row₂, hrow₂, hcs₂, hcw₂, ho₂⟩ := hop₂ (ℓ.val + 1) (by omega) x₂.1 x₂.2.1 (List.mem_map_of_mem hx₂)
  simp only [Nat.add_sub_cancel] at hcs₁ hcw₁ ho₁ hcs₂ hcw₂ ho₂
  have hcap : zp₁.proof.ligerito.recursiveCaps[ℓ.val]! = zp₂.proof.ligerito.recursiveCaps[ℓ.val]! :=
    capDig_inj (hd₁.trans hd₂.symm) (hcs₁.trans hcs₂.symm) hcw₁ hcw₂
  rw [hcap, hq] at ho₁
  exact tiedColl_of_openings ho₁ ho₂ fun he => hne (by rw [hrow₁, hrow₂, he])

end Gen

/-- **The masked live fork event's collision, with no hypothesis**: two accepting runs of `P` whose live outcomes
decode to `ForkEvZ` at table `j`'s rep `r`'s level `ℓ + 1` record one cap `d` there, the `capDig` of each run's
proof's `ℓ`-th recursive cap, so one cap of 64-byte digests (`capDig_inj`), and open one position of it to two
different rows, each an `Opening` against that cap in its run's log (`zkAccOpening`), which collide in the joined
logs (`tiedColl_of_openings`). -/
theorem zkForkClash (x : Claims.Msg) : ZkForkClash x :=
  Gen.zkForkClash (S := S₀) (x := x)

end Live

end FlockDeploy.KS
