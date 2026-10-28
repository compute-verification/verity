import Proofs.Flock.Claims.KS.PlainRLevels
import Proofs.Flock.Claims.KS.PlainCapsDep

/-!
# A cap position's read sets at the deployed reader `VR rd`

`CapHitAt W lc` (`Pr_rdAt_le_capHit`'s form) follows at any verifier `W` with `V false`'s cap positions from two facts
about the level columns `lc` (`capHitAt_of`, `Pr_rdAt_le_capHitK`'s proof):

* `QueriedAt W lc`: an accepted run has a level-`ℓ` column at `q` exactly where the level's query coin draws `q`
  (`plain_levelCols_queriedK`'s form);
* `TieAt W lc`: two accepted runs sharing the coins before the plain fork round, reading two rows at one later-level
  position, tie a collision among their calls (`plain_levelTieK`'s form).

At `VR rd` and its level columns `plainRawLevelColsRV rd`:

* `queriedAt_R`: `plain_levelCols_queriedK`'s proof at `nextR`'s verdict at the parsed value (`verifyJWithMK`): the
  columns are the positions of round `plainLevelRound`'s coins zipped with the proof's rows
  (`plainLevels_verifyRepWith`), and those coins are the run's coin `plainQRound`'s first `plainQN`: custody records
  each round's coin with the decision's mids, whose coins are the run's after the first (`decR_some`).
* `plainR_levelOpening`: a later-level row is an `Opening` in the run's log against the flattened recursive cap of the
  proof decoded from the read proofs' files (`plain_levelOpeningK`'s proof).
* `CapPrefixR rd`: two accepted runs sharing the coins before the plain fork round decode rep `r`'s file to proofs
  with one flattened `ℓ`-th recursive cap (`plain_capPrefixK`'s form at `VR rd`). `tieAt_R` from it.
* `capHitAt_R`, `plainR_tableBoundAbortLin_of_capPrefix`, `plainR_KSFlockLin_of_capPrefix`: the table bound and
  `KSFlockLin` at the deployed reader from `PlainRdInR` (`rdAllAt_R`) and `CapPrefixR`.
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Exec
open Claims.Bridge.Custody (checksAtJ_ok custody_ok)
open Claims.Bridge.PlainDraw (decideR_sha512)
open RecDeploy (outCoin)
open Lean (Json)

local notation "𝔽" => FlockLevel3.GF128

/-! ## `CapHitAt` from what it reads -/

/-- **Which positions an accepted run's level columns read at `W`** (`plain_levelCols_queriedK`'s form): at an
accepted run of `W`, a level-`ℓ ≥ 1` column of table 0's rep `r < 2` at `q` exists iff `q` is among the positions the
level's query coin draws. -/
def QueriedAt (W : Claims.Verifier Coin) (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) : Prop :=
  ∀ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (c : Fin (W.k x) → Coin),
    (Claims.run sha512 P W x ω c).out = .acc →
    ∀ {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ},
      setupOf false x = some (s, spec, sts) → sts[0]? = some st → fast100 st.m = some sch → r < 2 → 1 ≤ ℓ →
      ℓ < sch.levels.size → ∀ q : ℕ,
      ((∃ a, (q, a) ∈ lc x (Claims.run sha512 P W x ω c).seen r ℓ) ↔
        q ∈ positions (sch.levels.getD ℓ default)
          ((Claims.coinSeq c (plainQRound spec st sch r ℓ)).vals (plainQN spec st sch ℓ)))

/-- **Two rows at one later-level position tie a collision at `W`** (`plain_levelTieK`'s form): at two accepted runs
of `W` at one tape whose coins agree before `forkRound spec st k0 r ℓ`, two different rows their level columns read at
one position of table 0's rep `r < 2` and level `ℓ + 1` tie a collision among their verifier calls. -/
def TieAt (W : Claims.Verifier Coin) (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽)) : Prop :=
  ∀ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (c₁ c₂ : Fin (W.k x) → Coin),
    (Claims.run sha512 P W x ω c₁).out = .acc → (Claims.run sha512 P W x ω c₂).out = .acc →
    ∀ {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ},
      setupOf false x = some (s, spec, sts) → sts[0]? = some st → fast100 st.m = some sch → r < 2 →
      ℓ + 1 < sch.levels.size → (∀ i : Fin (W.k x), i.val < forkRound spec st sch.k0 r ℓ → c₁ i = c₂ i) →
      ∀ {q : ℕ} {a b : List 𝔽}, (q, a) ∈ lc x (Claims.run sha512 P W x ω c₁).seen r (ℓ + 1) →
        (q, b) ∈ lc x (Claims.run sha512 P W x ω c₂).seen r (ℓ + 1) → a ≠ b →
        Claims.TiedColl sha512 ((Claims.run sha512 P W x ω c₁).vLog ++ (Claims.run sha512 P W x ω c₂).vLog)

/-- **`CapHitAt` from which positions the level columns read and their ties** (`Pr_rdAt_le_capHitK`'s proof at `W`):
`Pr_rdOff_le_capHit` at `CpDep`'s position `i` and `rdAtW W lc`. A run reads something at the position exactly where it
accepts and its coins draw the position (`QueriedAt`), and two runs sharing the coins before the plain fork round
reading two rows there tie a collision (`TieAt`). -/
theorem capHitAt_of (W : Claims.Verifier Coin) (lc : Claims.Msg → Claims.Seen → ℕ → ℕ → List (ℕ × List 𝔽))
    (hq : QueriedAt W lc) (ht : TieAt W lc) : CapHitAt W lc := by
  intro Ω A _ _ _ _ P x keep M i lanes ω
  set j := (Fintype.equivFin (DepIdx x)).symm i with hj
  have hW : 0 < depL x := by
    have := j.2.1.isLt
    omega
  obtain ⟨⟨spec, st, sch⟩, hd⟩ : ∃ t, depSetup x = some t := by
    unfold depL at hW
    rcases hd : depSetup x with _ | t
    · rw [hd] at hW
      exact absurd hW (Nat.lt_irrefl 0)
    · exact ⟨t, rfl⟩
  obtain ⟨s, sts, hs, hst, hsch⟩ := depSetup_some hd
  obtain ⟨hround, hopens⟩ := CpDep_at hs hst hsch i
  have hℓ : j.2.1.val + 1 < sch.levels.size := by
    have := j.2.1.isLt
    have := depL_eq hs hst hsch
    omega
  refine Claims.Recursion.Pr_rdOff_le_capHit sha512 W CpDep P x keep M i ω _ (fun d => ?_)
    (fun d d' u u' hc hu hu' hne => ?_)
  · constructor
    · rintro ⟨u, hu⟩
      obtain ⟨hk, hr, a, ha, -⟩ := mem_rdAtW.1 hu
      refine ⟨(Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := W) (x := x) (keep := keep)
        (ω := ω) (d := d)).2 ⟨hk, hr⟩, ?_⟩
      rw [hopens, decide_eq_true_eq]
      exact (hq P x ω d.1 hr hs hst hsch j.1.isLt (by omega) hℓ _).1 ⟨a, ha⟩
    · rintro ⟨hacc, hq'⟩
      obtain ⟨hk, hr⟩ := (Claims.Recursion.abortRun_acc_iff (H := sha512) (P := P) (V := W) (x := x)
        (keep := keep) (ω := ω) (d := d)).1 hacc
      rw [hopens, decide_eq_true_eq] at hq'
      obtain ⟨a, ha⟩ := (hq P x ω d.1 hr hs hst hsch j.1.isLt (by omega) hℓ _).2 hq'
      exact ⟨_, mem_rdAtW.2 ⟨hk, hr, a, ha, rfl⟩⟩
  · obtain ⟨hk, hr, a, ha, rfl⟩ := mem_rdAtW.1 hu
    obtain ⟨hk', hr', a', ha', rfl⟩ := mem_rdAtW.1 hu'
    rw [Claims.Recursion.abortRun_of_keep (H := sha512) (P := P) (V := W) (x := x) (keep := keep) (ω := ω)
        (t := d) hk,
      Claims.Recursion.abortRun_of_keep (H := sha512) (P := P) (V := W) (x := x) (keep := keep) (ω := ω)
        (t := d') hk']
    rw [hround] at hc
    exact ht P x ω d.1 d'.1 hr hr' hs hst hsch j.1.isLt hℓ hc ha ha' fun he => hne (by rw [he])

/-! ## An accepted run of `VR rd`, read off its view -/

/-- **The run `runOfRV rd` reads on a view is `runOfR` at the view's messages and the run's coins** (`colsRV_eq`'s
proof): the decision's checks fix its statement to `x`, and `decR` doesn't read the last coin. -/
theorem runOfRV_eqR {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
    {v : Claims.Seen} (hm : v.msgs.map Prod.fst = ms)
    (hcs : (v.msgs.map Prod.snd).map outCoin = cs.dropLast ++ [default]) (hk : cs.length = (V false).k x)
    (hacc : ((nextR rd x ms cs).eval sha512).1 = .acc) : runOfRV rd v = runOfR rd x ms cs := by
  cases hact : actR rd x ms cs with
  | rej => revert hacc; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert hacc; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    have hk' : (V false).k x = 2 + rounds false spec sts := by
      simp only [V, actR_setupOf hact]
    have h2 : 2 ≤ cs.length := by omega
    obtain ⟨fl, hfl, hfs⟩ := decR_rd (actR_decide hact)
    have hx : fl.stmt = x := hfs.symm.trans (checksAtJ_ok (decideR_sha512 hact hacc).2.1).1
    funext j
    unfold runOfRV
    rw [runOfR_of_actR hact, hm, hcs, hfl]
    simp only [hx, decR_dropLast rd x ms h2, actR_decide hact]

/-- **`runOfRV rd` on an accepted run's view is `runOfR` at its messages and coins** (`colsRV_run`'s proof). -/
theorem runOfRV_run {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω c).out = .acc) :
    runOfRV rd (Claims.run sha512 P (VR rd) x ω c).seen =
      runOfR rd x ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst).length).map (Claims.coinSeq c)) := by
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω c h
  refine runOfRV_eqR rfl ?_ ?_ hacc
  · rw [List.length_map]
    exact run_outCoin_VR rd P x ω c h
  · have hk := accLast_VR rd x _ _ (by simp) hacc
    rw [List.length_map] at hk
    simp only [List.length_map, List.length_range]
    exact hk

/-- **Each plain rep stream of an accepted parsed record draws its table's `plainRepNs`** (`streams_ns_plainK`'s
proof, at `verifyJWithMK` and `liveOfJ`). -/
theorem streams_ns_plainJ (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array Stmt}
    {j : Json} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : liveOfJ spec j = .ok lv)
    (hpn : lv.points.size = spec.points) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].toList.map Round.n = plainRepNs spec sts[k / 2]! := by
  obtain ⟨sess, schs, hdec, hsz, -, -, hall⟩ := verifyJWithMK_tables_cap keep hs hv
  obtain ⟨-, hpts, hlsz, hstr⟩ := decodeJWithK_live keep hs hdec hl
  have hn : sts.size = spec.tables.size := by simpa [tabsOf] using hsz
  have hk₂ : ∀ k, k < lv.streams.size → k < spec.repStreams.size := fun k hk => hlsz ▸ hk
  refine ⟨by rw [hlsz, FlockDeploy.repStreams_size, hn], fun k hk => ?_⟩
  obtain ⟨y, hfind, hfa⟩ := hstr k (hk₂ k hk)
  have hi : k / 2 < (tabsOf sts).size := by
    have := hk₂ k hk
    rw [FlockDeploy.repStreams_size, ← hn] at this
    simp only [tabsOf, Array.size_map]
    omega
  have hi' : k / 2 < sts.size := by simpa [tabsOf] using hi
  obtain ⟨_, c, hf, h₀, h₁⟩ := hall (k / 2) hi
  have htab : (tabsOf sts)[k / 2] = Setup.ofCircuit sts[k / 2]! := by
    rw [getElem!_pos sts (k / 2) hi']; exact Array.getElem_map ..
  rw [htab] at h₀ h₁ hf
  simp only [FlockDeploy.repStreams_getElem spec (hk₂ k hk)] at hfind
  obtain ⟨bytes, hrep⟩ : ∃ bytes, verifyRepWith hs spec (Setup.ofCircuit sts[k / 2]!) schs[k / 2] sess (k / 2)
      (k % 2) bytes = .ok c := by
    rcases Nat.mod_two_eq_zero_or_one k with h | h <;> rw [h]
    · exact ⟨_, h₀⟩
    · exact ⟨_, h₁⟩
  obtain ⟨stream', extras, hfind', hex, hns⟩ := verifyRepWith_readsN hs rfl hf hrep
  rw [hfind] at hfind'
  cases hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := FlockDeploy.extraClaims_size hex
  have hsch : Claims.Flock.plainSched sts[k / 2]!.m = schs[k / 2] := by
    have e7 : (Setup.ofCircuit sts[k / 2]!).m = sts[k / 2]!.m := rfl
    rw [e7] at hf
    simp only [Claims.Flock.plainSched, hf, Option.getD_some]
  have hmap : lv.streams[k].toList.map Round.n = y.rounds.toList.map Round.n := by
    rw [← getElem!_pos lv.streams k hk]
    exact (map_n_of_forall₂ (hfa.imp fun _ _ h => ⟨h.1, h.2.1⟩)).symm
  rw [hmap, hns, hx, hpts, hpn, plainRepNs, hsch, Nat.mul_comm]
  rfl

/-! ## Which positions an accepted run of `VR rd` opens -/

/-- **An accepted run of `VR rd` opens a later-level position exactly where its query coin draws it**
(`plain_levelCols_queriedK`'s proof, at `nextR`'s verdict at the parsed value): at table 0's rep `r < 2` and level
`1 ≤ ℓ`, a column at `q` exists iff `q` is among the positions the run's coin `plainQRound` draws at count `plainQN`.
The stream round's coins are the decision's mid coin there (`custody_ok`), which is the run's (`decR_some`). -/
theorem plainR_levelCols_queried {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω}
    {x : Claims.Msg} {ω : Ω} {c : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω c).out = .acc)
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ}
    (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch)
    (hr : r < 2) (h1 : 1 ≤ ℓ) (hℓ : ℓ < sch.levels.size) (q : ℕ) :
    (∃ a, (q, a) ∈ plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen r ℓ) ↔
      q ∈ positions (sch.levels.getD ℓ default)
        ((Claims.coinSeq c (plainQRound spec st sch r ℓ)).vals (plainQN spec st sch ℓ)) := by
  obtain ⟨s', spec', sts', link, c₁, mids, f, hact, hrun⟩ := accR_decide h
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω c h
  obtain ⟨-, hc, hsu, -, -⟩ := decideR_sha512 hact hacc
  obtain ⟨rfl, rfl, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = sts' := by
    simpa using hs.symm.trans (actR_setupOf hact)
  have hv := verdictJ_within (decisionSF_runs hsu hrun)
  generalize (Claims.run sha512 P (VR rd) x ω c).vLog.map Prod.fst = L at hv
  obtain ⟨-, lv, hlv, hcus⟩ := checksAtJ_ok hc
  have hpn : lv.points.size = spec.points := by
    rw [liveOfJ_points spec c₁ mids _ hlv]
    simp [pointsOf]
  obtain ⟨hsts0, hst0⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, schs, hdec, hsz, -, -, hall⟩ := verifyJWithMK_tables_cap id _ hv
  obtain ⟨hi', cB, hsch', h0r, h1r⟩ := hall 0 h0
  simp only [ht0] at hsch' h0r h1r
  have e : schs[0] = sch := Option.some.inj (hsch'.symm.trans hsch)
  rw [e] at h0r h1r
  have hrep : verifyRepWith (hashesIn L) spec (Setup.ofCircuit st) sch sess 0 r
      (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + r]! = .ok cB := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · simpa using h0r
    · exact h1r
  obtain ⟨hfi, hkl⟩ := tabsOf_fold sts (Setup.ofCircuit st) (ht0 ▸ Array.getElem_mem h0)
  obtain ⟨stream, extras, p, hstream, hext, hp, hlev⟩ :=
    plainLevels_verifyRepWith _ spec (Setup.ofCircuit st) sch sess 0 r _ cB hsch hfi hkl hrep
  obtain ⟨-, hpts, -, hstreams⟩ := decodeJWithK_live id _ hdec hlv
  rw [hpts] at hext
  have hn : extras.size = st.regions.size * spec.points := by
    rw [extraClaims_size (st := st) hext, hpn]
  have hcols0 : plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen r ℓ =
      plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
        (lv.streams.getD r #[]) ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD r ByteArray.empty) ℓ := by
    unfold plainRawLevelColsRV
    rw [runOfRV_run h]
    rcases hrr : runOfR rd x ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst)
        ((List.range ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst).length).map (Claims.coinSeq c)) 0 with
      _ | rr
    · rw [runOfR_of_actR hact] at hrr
      simp [runOfDecR, hst, hlv, hext, Except.toOption] at hrr
    · obtain ⟨st', extras', hst', hext', rfl⟩ := runOfR_eq hact hlv hrr
      obtain rfl := Option.some.inj (hst'.symm.trans hst)
      rw [hext] at hext'
      cases hext'
      rcases r with _ | _ | r
      · rfl
      · rfl
      · omega
  have hJJ : plainLevelRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size)
      (sch.levels.size - 1) ℓ =
      plainLevelRound st.m st.c.kLog sch.k0 (2 + spec.points * st.regions.size) (sch.levels.size - 1) ℓ := by
    rw [hn, Nat.mul_comm]
    rfl
  obtain ⟨hent, hmc⟩ := hlev ℓ h1 hℓ
  rw [hJJ] at hent hmc
  set J := plainLevelRound st.m st.c.kLog sch.k0 (2 + spec.points * st.regions.size) (sch.levels.size - 1) ℓ
    with hJ
  have hJR : J < repRounds false spec st := plainLevelRound_lt hsch hℓ
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hr
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + r) hk'
  rw [hrs] at hy
  obtain rfl : y = stream := Option.some.inj (hy.symm.trans hstream)
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hcolsEq : plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen r ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default) (y.rounds[J]!).coins
        (openAt p.ligerito (sch.levels.size - 1) ℓ).1 := by
    rw [hcols0, ← hent]
    have e1 : lv.streams.getD r #[] = lv.streams[2 * 0 + r]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD r ByteArray.empty =
        (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[2 * 0 + r]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]; simp
    rw [e1, e2]
    refine plainLevelColsOf_congr _ _ _ fun j => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins j).symm
  obtain ⟨hsz2, hsizes⟩ := streams_size_plainJ id _ hv hlv hpn
  have hR0 : ∀ k (hk : k < lv.streams.size), k < 2 → lv.streams[k].size = repRounds false spec st := by
    intro k hk hk2
    rw [hsizes k hk, show k / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
  have hq2 : 2 ≤ lv.streams.size := by omega
  have hjr : J < (lv.streams[r]'(by omega)).size := by rw [hR0 r _ hr]; exact hJR
  have hfl := (custody_ok hcus).2.2.2
  have e1 := flat_two_getElem? lv.streams hq2 hr hjr
  rw [hR0 0 (by omega) (by omega), Array.getElem?_eq_getElem hjr] at e1
  obtain ⟨am, hm, hram⟩ := forall₂_getElem?_some hfl e1
  obtain ⟨_, _, _, _, _, _, pf, _, _, _, _, _, _, hdd⟩ := decR_some (actR_decide hact)
  simp only [DecR.mk.injEq] at hdd
  obtain ⟨-, -, -, -, -, hmids, -⟩ := hdd
  rw [hmids] at hm
  obtain ⟨-, hb⟩ := List.getElem?_zip_eq_some.1 hm
  obtain ⟨_, hget⟩ := List.getElem?_eq_some_iff.1 hb
  simp only [List.getElem_dropLast, List.getElem_tail, List.getElem_map, List.getElem_range] at hget
  obtain ⟨-, hns⟩ := streams_ns_plainJ id _ hv hlv hpn
  have hnr := hns r (by omega)
  rw [show r / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0] at hnr
  have hnJ : (lv.streams[r]'(by omega))[J].n = plainQN spec st sch ℓ := by
    have hlen : J < (plainRepNs spec st).length := by rw [← hnr]; simpa using hjr
    unfold plainQN
    rw [← hJ, getElem!_pos _ J hlen]
    simp only [← hnr, List.getElem_map, Array.getElem_toList]
  simp only [Nat.mul_zero, Nat.zero_add] at hfa
  rw [getElem!_pos lv.streams r (by omega)] at hfa
  have hsize : y.rounds.size = repRounds false spec st := by
    have := hfa.length_eq
    rw [Array.length_toList, Array.length_toList, hR0 r _ hr] at this
    exact this
  have hfj := Claims.Bridge.Custody.forall₂_getElem hfa (show J < y.rounds.size by omega)
  have hcJ : (y.rounds[J]!).coins =
      (Claims.coinSeq c (plainQRound spec st sch r ℓ)).vals (plainQN spec st sch ℓ) := by
    rw [getElem!_pos y.rounds J (by omega), hfj.2.1, getElem!_pos _ J hjr, hram.2, hnJ, ← hget]
    rfl
  rw [hcolsEq, hcJ]
  rw [hcJ] at hmc
  constructor
  · rintro ⟨a, ha⟩
    obtain ⟨j, hj, -, hjq, -⟩ := mem_zkRepCols ha
    rw [getElem!_pos _ j hj] at hjq
    exact hjq ▸ Array.getElem_mem hj
  · intro hq
    obtain ⟨j, hj, hjq⟩ := Array.mem_iff_getElem.1 hq
    have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
        (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
      rcases checksAtJ_leaf hc hsu 0 hsts0 with e | ⟨-, e⟩
      · rw [hst0] at e
        exact Or.inl e
      · rw [hst0] at e
        exact Or.inr e
    have hruns := With.merkleCheckWith_sim (back L)
      (With.opsWith_sim (back L) (back_hashes L) (Setup.ofCircuit st).merkle)
      (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
    obtain ⟨hpure, -⟩ := merkleCheck_logsOf hms hruns
    obtain ⟨hrows, -⟩ := merkleCheck_rows hpure
    have hseam : LevelSeam (sch.levels.getD ℓ default) :=
      fast100_level _ _ hsch _ (by unfold Array.getD; rw [dite_eq_left hℓ]; exact Array.getElem_mem hℓ)
    have hlq := positions_size _ hseam ((Claims.coinSeq c (plainQRound spec st sch r ℓ)).vals (plainQN spec st sch ℓ))
    exact ⟨_, hjq ▸ mem_zkRepCols_of hj (by rw [hrows]; exact hj) (by rw [← hlq]; exact hj)⟩

/-- **`QueriedAt` at `VR rd` and its level columns.** -/
theorem queriedAt_R (rd : Claims.Msg → Option FinalP) : QueriedAt (VR rd) (fun _ => plainRawLevelColsRV rd) :=
  fun _ _ _ _ h _ _ _ _ _ _ _ hs hst hsch hr h1 hℓ q => plainR_levelCols_queried h hs hst hsch hr h1 hℓ q

/-! ## The openings an accepted run of `VR rd` makes at a later level -/

/-- **A later-level row an accepted run of `VR rd` opens is an `Opening` in its log** (`plain_levelOpeningK`'s proof,
at `nextR`'s verdict at the parsed value): a level-`ℓ` row of table 0's rep `k` is an `Opening` of its position
against the flattened `ℓ − 1`-th recursive cap of the proof decoded from rep `k`'s file of the read proofs, a cap of
`2^cap` 64-byte digests, each SHA-512 input of it in the run's verifier log. -/
theorem plainR_levelOpening {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg}
    {ω : Ω} {c : Fin ((VR rd).k x) → Coin} {k ℓ q : ℕ} {a : List 𝔽}
    (h : (Claims.run sha512 P (VR rd) x ω c).out = .acc)
    (ha : (q, a) ∈ plainRawLevelColsRV rd (Claims.run sha512 P (VR rd) x ω c).seen k ℓ) :
    ∃ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
      (mids : List (Claims.Msg × Coin)) (f : FinalP) (st : Stmt) (sch : Flock.Schedule) (p : Proof)
      (row : Array F128),
      actR rd x ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω c).msgs.map Prod.fst).length).map (Claims.coinSeq c)) =
        .decide s spec sts link c₁ mids f ∧
      sts[0]? = some st ∧ fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points))
        (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[k]! = .ok p ∧
      a = row.toList.map FlockLevel3.GF128.of ∧
      p.ligerito.recursiveCaps[ℓ - 1]!.size = 2 ^ (sch.levels.getD ℓ default).cap ∧
      (∀ b ∈ p.ligerito.recursiveCaps[ℓ - 1]!.toList, b.size = 64) ∧
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) (Claims.run sha512 P (VR rd) x ω c).vLog
        q row := by
  obtain ⟨s, spec, sts, link, c₁, mids, f, hact, hrun⟩ := accR_decide h
  have hacc := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω c h
  obtain ⟨-, hc, hsu, -, -⟩ := decideR_sha512 hact hacc
  have hv := verdictJ_within (decisionSF_runs hsu hrun)
  generalize hL : (Claims.run sha512 P (VR rd) x ω c).vLog = L at hv ⊢
  obtain ⟨-, lv, hlv, -⟩ := checksAtJ_ok hc
  have hpn : lv.points.size = spec.points := by
    rw [liveOfJ_points spec c₁ mids _ hlv]
    simp [pointsOf]
  unfold plainRawLevelColsRV at ha
  rw [runOfRV_run h] at ha
  split at ha
  swap
  · simp at ha
  rename_i rr hrr
  obtain ⟨st, extras, hst, hext, rfl⟩ := runOfR_eq hact hlv hrr
  obtain ⟨hk, ha⟩ : k < 2 ∧
      (q, a) ∈ plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
        (lv.streams.getD k #[]) ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) ℓ := by
    rcases k with _ | _ | k
    · exact ⟨by omega, ha⟩
    · exact ⟨by omega, ha⟩
    · simp at ha
  have hsts0 : 0 < sts.size := (Array.getElem?_eq_some_iff.1 hst).1
  have hst0 : sts[0] = st := (Array.getElem?_eq_some_iff.1 hst).2
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, schs, hdec, hsz, -, -, hall⟩ := verifyJWithMK_tables_cap id _ hv
  obtain ⟨hi', cB, hsch, h0r, h1r⟩ := hall 0 h0
  simp only [ht0] at hsch h0r h1r
  generalize schs[0] = sch at hsch h0r h1r
  have hrep : verifyRepWith (hashesIn (L.map Prod.fst)) spec (Setup.ofCircuit st) sch sess 0 k
      (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[k]! = .ok cB := by
    rcases (by omega : k = 0 ∨ k = 1) with rfl | rfl
    · simpa using h0r
    · simpa using h1r
  obtain ⟨hfi, hkl⟩ := tabsOf_fold sts (Setup.ofCircuit st) (ht0 ▸ Array.getElem_mem h0)
  obtain ⟨stream, extras', p, hstream, hext', hp', hlev⟩ :=
    plainLevels_verifyRepWith _ spec (Setup.ofCircuit st) sch sess 0 k _ cB hsch hfi hkl hrep
  obtain ⟨-, hpts, -, hstreams⟩ := decodeJWithK_live id _ hdec hlv
  rw [hpts, hext] at hext'
  cases hext'
  have htab : 0 < spec.tables.size := by rw [← hsz]; exact h0
  obtain ⟨hk', hrs⟩ := Claims.Bridge.Custody.repStreams_getElem spec htab hk
  obtain ⟨y, hy, hfa⟩ := hstreams (2 * 0 + k) hk'
  rw [hrs] at hy
  have hys : y = stream := Option.some.inj (hy.symm.trans hstream)
  subst hys
  have hcoins := forall₂_coins (fun _ _ hab => hab.2.1) hfa
  have hsched : (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog (2 + extras.size)
      (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st))).sched = sch := by
    simp only [Claims.Flock.plainProofShape, Claims.Flock.plainSched, hsch]
    rfl
  have hℓ : 1 ≤ ℓ ∧ ℓ < sch.levels.size := by
    by_contra hn
    rw [plainLevelColsOf, ite_eq_right (by rw [hsched]; exact hn)] at ha
    simp at ha
  obtain ⟨hent, hmc⟩ := hlev ℓ hℓ.1 hℓ.2
  have hcolsEq : plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
      (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
      (lv.streams.getD k #[]) ((filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty) ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default)
        (y.rounds[plainLevelRound (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size)
          (sch.levels.size - 1) ℓ]!).coins (openAt p.ligerito (sch.levels.size - 1) ℓ).1 := by
    rw [← hent]
    have e1 : lv.streams.getD k #[] = lv.streams[2 * 0 + k]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps).getD k ByteArray.empty =
        (filesAt (tabsOf sts) (pointsOf spec c₁) f.ps)[k]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]
    rw [e1, e2]
    refine plainLevelColsOf_congr _ _ _ fun j => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins j).symm
  rw [hcolsEq] at ha
  obtain ⟨j, hj, -, hjq, hja⟩ := mem_zkRepCols ha
  have hms : (Setup.ofCircuit st).merkle = MerkleScheme.hm96Sha512 ∨
      (Setup.ofCircuit st).merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rcases checksAtJ_leaf hc hsu 0 hsts0 with e | ⟨-, e⟩
    · rw [hst0] at e
      exact Or.inl e
    · rw [hst0] at e
      exact Or.inr e
  have hdl : (Setup.ofCircuit st).merkle.digestLen = 64 := by
    rcases hms with e | e <;> rw [e] <;> rfl
  have hruns := With.merkleCheckWith_sim (back (L.map Prod.fst))
    (With.opsWith_sim (back (L.map Prod.fst)) (back_hashes (L.map Prod.fst)) (Setup.ofCircuit st).merkle)
    (Setup.ofCircuit st).merkle _ _ _ _ _ _ _ _ () hmc
  obtain ⟨hpure, hlog⟩ := merkleCheck_logsOf hms hruns
  obtain ⟨hcs, hcw⟩ := merkleCheck_cap hpure
  have hop : ∀ i < (positions (sch.levels.getD ℓ default) (y.rounds[plainLevelRound (Setup.ofCircuit st).m
      (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins).size,
      Opening (Setup.ofCircuit st).merkle (sch.levels.getD ℓ default) (2 ^ (sch.levels.getD ℓ default).lanes)
        (p.ligerito.recursiveCaps[ℓ - 1]!.foldl (· ++ ·) ByteArray.empty) L
        (positions (sch.levels.getD ℓ default) (y.rounds[plainLevelRound (Setup.ofCircuit st).m
          (Setup.ofCircuit st).kLog sch.k0 (2 + extras.size) (sch.levels.size - 1) ℓ]!).coins)[i]!
        (openAt p.ligerito (sch.levels.size - 1) ℓ).1[i]! := fun i hi =>
    ⟨hms, p.ligerito.recursiveCaps[ℓ - 1]!,
      (openAt p.ligerito (sch.levels.size - 1) ℓ).2.2.getD i ByteArray.empty,
      sibsOf (openAt p.ligerito (sch.levels.size - 1) ℓ).2.1
        ((sch.levels.getD ℓ default).d - (sch.levels.getD ℓ default).cap) i,
      hcs, fun b hb => (hcw b hb).trans hdl, rfl, merkleCheck_opens _ _ _ _ _ _ _ _ _ hpure i hi,
      fun b hb => (merkleCheck_sibs _ _ _ _ _ _ _ _ _ hpure i b hb).trans hdl, hlog i hi⟩
  have hn : extras.size = st.regions.size * spec.points := by
    rw [extraClaims_size (st := st) hext, hpn]
  rw [hn] at hp'
  exact ⟨s, spec, sts, link, c₁, mids, f, st, sch, p, _, hact, hst, hsch, hp', hja, hcs,
    fun b hb => (hcw b hb).trans hdl, hjq ▸ hop j hj⟩

/-! ## The caps two accepted runs of `VR rd` absorb -/

/-- **Two accepted runs of `VR rd` sharing the coins before the plain fork round decode one cap there**
(`plain_capPrefixK`'s form at `VR rd`): at two accepted runs of `VR rd` at one tape whose coins agree before
`forkRound spec st k0 r ℓ`, rep `r < 2` of table 0's files of the two decisions' read proofs, decoded at the table's
`fast100` schedule and `|regions| · points` extra claims, have one flattened `ℓ`-th recursive cap, for each `ℓ + 1`
below the schedule's levels. -/
def CapPrefixR (rd : Claims.Msg → Option FinalP) : Prop :=
  ∀ {Ω : Type} (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω) (c₁ c₂ : Fin ((VR rd).k x) → Coin),
    (Claims.run sha512 P (VR rd) x ω c₁).out = .acc → (Claims.run sha512 P (VR rd) x ω c₂).out = .acc →
    ∀ {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ},
      setupOf false x = some (s, spec, sts) → sts[0]? = some st → fast100 st.m = some sch → r < 2 →
      ℓ + 1 < sch.levels.size →
      (∀ i : Fin ((VR rd).k x), i.val < forkRound spec st sch.k0 r ℓ → c₁ i = c₂ i) →
      ∀ {link₁ link₂ : List ByteArray × List ByteArray} {e₁ e₂ : Coin} {mids₁ mids₂ : List (Claims.Msg × Coin)}
        {f₁ f₂ : FinalP} {p₁ p₂ : Proof},
      actR rd x ((Claims.run sha512 P (VR rd) x ω c₁).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω c₁).msgs.map Prod.fst).length).map (Claims.coinSeq c₁)) =
        .decide s spec sts link₁ e₁ mids₁ f₁ →
      actR rd x ((Claims.run sha512 P (VR rd) x ω c₂).msgs.map Prod.fst)
          ((List.range ((Claims.run sha512 P (VR rd) x ω c₂).msgs.map Prod.fst).length).map (Claims.coinSeq c₂)) =
        .decide s spec sts link₂ e₂ mids₂ f₂ →
      decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points))
          (filesAt (tabsOf sts) (pointsOf spec e₁) f₁.ps)[r]! = .ok p₁ →
      decodeProof (shapeOf (Setup.ofCircuit st) sch (st.regions.size * spec.points))
          (filesAt (tabsOf sts) (pointsOf spec e₂) f₂.ps)[r]! = .ok p₂ →
      p₁.ligerito.recursiveCaps[ℓ]!.foldl (· ++ ·) ByteArray.empty =
        p₂.ligerito.recursiveCaps[ℓ]!.foldl (· ++ ·) ByteArray.empty

/-- **`TieAt` at `VR rd` from `CapPrefixR`** (`plain_levelTieK`'s proof): each row is an `Opening` of the position in
its run's log against the run's decoded rep-`r` flattened `ℓ`-th recursive cap (`plainR_levelOpening` at level
`ℓ + 1`), the two caps are one (`CapPrefixR`), and two openings of one position against one root to different rows tie
a collision (`tiedColl_of_openings`). -/
theorem tieAt_R (rd : Claims.Msg → Option FinalP) (hcp : CapPrefixR rd) :
    TieAt (VR rd) (fun _ => plainRawLevelColsRV rd) := by
  intro Ω P x ω c₁ c₂ h₁ h₂ s spec sts st sch r ℓ hs hst hsch hr hℓ hc q a b ha hb hab
  obtain ⟨s₁, spec₁, sts₁, link₁, e₁, mids₁, f₁, st₁, sch₁, p₁, row₁, hact₁, hst₁, hsch₁, hp₁, hra, -, -, hop₁⟩ :=
    plainR_levelOpening h₁ ha
  obtain ⟨s₂, spec₂, sts₂, link₂, e₂, mids₂, f₂, st₂, sch₂, p₂, row₂, hact₂, hst₂, hsch₂, hp₂, hrb, -, -, hop₂⟩ :=
    plainR_levelOpening h₂ hb
  obtain ⟨rfl, e₁'⟩ := Prod.mk.inj (Option.some.inj ((actR_setupOf hact₁).symm.trans hs))
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj e₁'
  obtain ⟨rfl, e₂'⟩ := Prod.mk.inj (Option.some.inj ((actR_setupOf hact₂).symm.trans hs))
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj e₂'
  obtain rfl := Option.some.inj (hst₁.symm.trans hst)
  obtain rfl := Option.some.inj (hst₂.symm.trans hst)
  obtain rfl := Option.some.inj (hsch₁.symm.trans hsch)
  obtain rfl := Option.some.inj (hsch₂.symm.trans hsch)
  have hcap := hcp P x ω c₁ c₂ h₁ h₂ hs hst hsch hr hℓ hc hact₁ hact₂ hp₁ hp₂
  rw [Nat.add_sub_cancel] at hop₁ hop₂
  rw [hcap] at hop₁
  exact tiedColl_of_openings hop₁ hop₂ fun he => hab (by rw [hra, hrb, he])

/-! ## The table bound at the deployed reader -/

/-- **`CapHitAt` at `VR rd` and its level columns, from `CapPrefixR`.** -/
theorem capHitAt_R (rd : Claims.Msg → Option FinalP) (hcp : CapPrefixR rd) :
    CapHitAt (VR rd) (fun _ => plainRawLevelColsRV rd) :=
  capHitAt_of (VR rd) _ (queriedAt_R rd) (tieAt_R rd hcp)

/-- **The table bound at the deployed reader's plain record** (`plainR_tableBoundAbortLin` at the level columns
`plainRawLevelColsRV rd`): from the reader's bound's hypothesis `PlainRdInR` wherever some run accepts (`rdAllAt_R`)
and `CapPrefixR`; the top level's emptiness is `levelTopAt_R`. -/
theorem plainR_tableBoundAbortLin_of_capPrefix (rd : Claims.Msg → Option FinalP)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc sha512 (VR rd) x → PlainRdInR rd x) (hcp : CapPrefixR rd) :
    Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR Readable.relation rd) (wordLen false) (wide false)
      CpDep :=
  plainR_tableBoundAbortLin Readable.relation rd (fun _ => plainRawLevelColsRV rd) (rdAllAt_R rd hIn)
    (levelTopAt_R rd) (capHitAt_R rd hcp)

/-- **`KSFlockLin` at the deployed reader's trimmed plain record**, from `PlainRdInR` and `CapPrefixR`
(`plainR_KSFlockLin_of_reader`). -/
theorem plainR_KSFlockLin_of_capPrefix (rd : Claims.Msg → Option FinalP)
    (hIn : ∀ x, ¬ Claims.Flock.NoAcc sha512 (VR rd) x → PlainRdInR rd x) (hcp : CapPrefixR rd) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR Readable.relation rd) CpDep :=
  plainR_KSFlockLin_of_reader Readable.relation rd (fun _ => plainRawLevelColsRV rd) (rdAllAt_R rd hIn)
    (levelTopAt_R rd) (capHitAt_R rd hcp)

end FlockDeploy.KS
