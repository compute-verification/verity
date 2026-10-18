import Proofs.Flock.Claims.KS.ZkSimW

/-!
# A masked self-clash from two raw columns

`PlainRawClashRun` at `--zk`, at every table. Table `j`'s reps' columns before `distinctBelow` keeps one per position
(`zkRawColsJ`, holding every column the reader opens: `mem_zkRawColsJ`) are each the witness lanes of an opened row
whose level-0 opening against table `j`'s `root_B` passed, every SHA-512 input of it in the run's own log
(`zk_openingJ_raw`, `zk_openingJ` before `distinctBelow`), so two of them at one position that differ collide in that
log (`zk_selfClash_tiedJ`). So the self-clash clause of `ZkSimWJ` (`ZkClashRun`) follows from a live self-clash giving
an accepting run whose reps open two different columns at one position of some table, or a collision
(`ZkRawClashRun`, `zkClashRun_of_raw`).
-/

namespace FlockDeploy.KS

local notation "𝔽" => FlockLevel3.GF128

section Raw

open Flock
open FlockSoundness.Refine (ebind_ok entries_eq_of_flat flat_data merkleCheck_cap)

/-- **The columns table `j`'s reps open in a masked record**, every one, before `distinctBelow` keeps one per
position. -/
def zkRawColsJ (s : Claims.Seen) (j : ℕ) : List (ℕ × List 𝔽) :=
  match runOf true s j with
  | some r => r.reps.flatMap fun y => Claims.Flock.zkRepColsOf r.shape y.1 y.2
  | none => []

theorem mem_zkRawColsJ {s : Claims.Seen} {j : ℕ} {p : ℕ × List 𝔽} (h : p ∈ cols true s j) : p ∈ zkRawColsJ s j := by
  unfold cols at h
  unfold zkRawColsJ
  cases hr : runOf true s j with
  | none => simp [hr] at h
  | some r =>
    simp only [hr, ↓reduceIte, Claims.Flock.zkRunCols] at h ⊢
    exact mem_of_distinctBelow h

/-- **An accepted masked run's reps open a column of table `j` only as a Merkle opening** of table `j`'s level 0 in
its own log: `zk_openingJ` at every column a rep opens. -/
theorem zk_openingJ_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {r : Fin ((V true).k x) → Coin}
    {j q : ℕ} {u : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : (q, u) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ)
      (link : List ByteArray × List ByteArray) (row : Array F128),
      setupOf true x = some (s, spec, sts) ∧ sts[j]? = some st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
      linkOf ((P.next x [] ω).eval sha512).1 = some link ∧
      Opening (sch.levels.getD 0 default) (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES)
        link.1.toArray[j]! (Claims.run sha512 P (V true) x ω r).vLog q row ∧
      u = (row.toList.map FlockLevel3.GF128.of).take (2 ^ (sch.levels.getD 0 default).lanes) := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hs, hf, hlink, hc, hv⟩ := acc_within h
  generalize hL : (Claims.run sha512 P (V true) x ω r).vLog = L at hv ⊢
  obtain ⟨hx, lv, hlv, hcu⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hroot, -⟩ := Claims.Bridge.Custody.custody_ok hcu
  unfold zkRawColsJ at hu
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
  obtain ⟨stream, extras', zp, hstream, hext', hzp, hlt, hcols, hmc, hcap⟩ := hrep k hk
  simp only [htj] at hsch hext' hzp hlt hcols hmc
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
  have hcolsEq : Claims.Flock.zkRepColsOf
      (Claims.Flock.zkProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
        (Setup.ofCircuit st).merkle.digestLen (saltLenOf true (Setup.ofCircuit st)))
      (lv.streams.getD (2 * j + k) #[]) (f.proofs.getD (2 * j + k) ByteArray.empty) =
      (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
        (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
          (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows).map
        (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) := by
    rw [← hcols]
    have e1 : lv.streams.getD (2 * j + k) #[] = lv.streams[2 * j + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]
    have e2 : f.proofs.getD (2 * j + k) ByteArray.empty = f.proofs[2 * j + k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine zkRepColsOf_congr _ _ fun i => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins i).symm
  have hp2 : (q, u) ∈ (Claims.Flock.zkRepCols (sch.levels.getD 0 default)
      (y.rounds[Claims.Flock.zkQueryRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0
        (2 + extras.size)]!).coins zp.proof.ligerito.initialProof.rows).map
      (fun p => (p.1, p.2.take (2 ^ (sch.levels.getD 0 default).lanes))) := by
    rw [← hcolsEq]
    exact hp
  obtain ⟨⟨q', w⟩, hw, hqw⟩ := List.mem_map.1 hp2
  simp only [Prod.mk.injEq] at hqw
  obtain ⟨rfl, rfl⟩ := hqw
  obtain ⟨i, hi, -, hiq, hiu⟩ := mem_zkRepCols hw
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 := by
    have := checks_hm96 hc (setupOf_some hs).2 j hstsj
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
    ⟨zp.proof.cap, zp.proof.ligerito.initialProof.salts.getD i ByteArray.empty,
      sibsOf zp.proof.ligerito.initialProof.siblings
        ((sch.levels.getD 0 default).d - (sch.levels.getD 0 default).cap) i,
      hcs, hcw, ?_, ?_, merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i, ?_⟩, by rw [hiu]⟩
  · rw [hcap, hrB, ← hroot, Array.toArray_toList]
  · rw [← hiq]
    exact merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi
  · rw [← hiq]
    exact hlog i hi

/-- **A self-clash among an accepted masked run's columns collides in its own log**: two columns table `j`'s reps
open at one position that differ give a collision among the run's verifier calls, at SHA-512. -/
theorem zk_selfClash_tiedJ {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((V true).k x) → Coin} {j q : ℕ} {u v : List 𝔽} (h : (Claims.run sha512 P (V true) x ω r).out = .acc)
    (hu : (q, u) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j)
    (hv : (q, v) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω r).seen j) (huv : u ≠ v) :
    Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω r).vLog := by
  obtain ⟨s, spec, sts, st, sch, tPad, link, row, hs, hst, hsch, hlink, ho, hu'⟩ := zk_openingJ_raw h hu
  obtain ⟨s', spec', sts', st', sch', tPad', link', row', hs', hst', hsch', hlink', ho', hv'⟩ := zk_openingJ_raw h hv
  rw [hs] at hs'
  cases hs'
  rw [hst] at hst'
  cases hst'
  rw [hsch] at hsch'
  cases hsch'
  rw [hlink] at hlink'
  cases hlink'
  exact tiedColl_of_append_self (tiedColl_of_openings ho ho' fun e => huv (by rw [hu', hv', e]))

end Raw

section Clash

open Flock FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkSession
open FlockSoundness.Discharge
open FlockSoundness.Discharge.Tables (LiveOutJ liveSession DecodesTable)
open Claims.Bridge.Live

/-- **A live event at `P'`'s outcome gives an accepting masked run of `P` whose reps open two different columns at
one position of some table, or a collision**: `ZkClashRun` with a self-clash at level 0 asked without the hashes
(`zkRawColsJ`). -/
def ZkRawClashRun {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (mPts : ℕ) (J N : Claims.Msg → ℕ)
    (dg : ByteArray → ByteArray) (w : (x : Claims.Msg) → Wire 65536 (J x) mPts) {τ : Type}
    (P' : Claims.Prover (Ω × τ)) (Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin)
    (pL : LiveOutJ (J x) mPts → Prop) : Prop :=
  ∀ u r', pL (liveOut J (fun _ => mPts) N dg w sha512 P' x (ω, u) r') →
    (Claims.run sha512 P (V true) x ω (Φ (u, r'))).out = .acc ∧
      ((∃ (j q : ℕ) (a b : List 𝔽),
        (q, a) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω (Φ (u, r'))).seen j ∧
        (q, b) ∈ zkRawColsJ (Claims.run sha512 P (V true) x ω (Φ (u, r'))).seen j ∧ a ≠ b) ∨
        Claims.TiedColl sha512 (Claims.run sha512 P (V true) x ω (Φ (u, r'))).vLog)

/-- **The clash gives the collision**: two different columns of an accepting masked run at one position of one table
collide in its log (`zk_selfClash_tiedJ`), so `ZkRawClashRun` gives `ZkClashRun`. -/
theorem zkClashRun_of_raw {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω} {mPts : ℕ}
    {J N : Claims.Msg → ℕ} {dg : ByteArray → ByteArray} {w : (x : Claims.Msg) → Wire 65536 (J x) mPts} {τ : Type}
    {P' : Claims.Prover (Ω × τ)} {Φ : τ × (Fin (N x + 2) → LCoin 65536) → Fin ((V true).k x) → Coin}
    {pL : LiveOutJ (J x) mPts → Prop} (h : ZkRawClashRun P x ω mPts J N dg w P' Φ pL) :
    ZkClashRun P x ω mPts J N dg w P' Φ pL := fun u r' hp => by
  obtain ⟨hacc, ⟨j, q, a, b, ha, hb, hab⟩ | ht⟩ := h u r' hp
  · exact ⟨hacc, zk_selfClash_tiedJ hacc ha hb hab⟩
  · exact ⟨hacc, ht⟩

end Clash

end FlockDeploy.KS
