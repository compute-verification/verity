import Proofs.Flock.Claims.KS.PlainCapPrefix
import Proofs.Flock.Claims.Recursion.PlainK
import Proofs.Flock.Claims.Recursion.CallsBridgeE

/-!
# Which positions an accepted plain rep opens at a later level

An accepted plain run opens, at table 0's rep `r` and level `ℓ ≥ 1`, exactly the positions drawn from that level's
query coins, and those coins are a deployed coin with a count the statement fixes. So "the run opens `q` there" is a
function of the run's coins alone, which is what a cap position's `opens` reads (`Claims.Recursion.Caps`).

* `plainQRound`, `plainQN`: the deployed coin of rep `r`'s level-`ℓ` query round, `r · R + J + 1` for `R` the table's
  `repRounds` and `J` the stream round `plainLevelRound`, and its count, `plainRepNs`' entry `J`.
* `plain_levelCols_queried`: at an accepted plain run, a level-`ℓ` column at `q` exists iff `q` is among the positions
  that coin draws. The columns are those positions zipped with the proof's rows (`plainLevels_verifyWithM`), the rows
  as many as the positions (`merkleCheck_rows`), the positions `q` of the level (`positions_size`), and the stream
  round's coins the deployed coin's first `n` (`custody_ok`, `streams_ns_plain`).
-/

namespace FlockDeploy.KS

open Flock Transcript FlockSoundness FlockSoundness.Refine FlockSoundness.Discharge.Exec

/-- **The deployed coin of rep `r`'s level-`ℓ` query round** at table 0: rep `r`'s `J`-th stream round, `J` the
level's `plainLevelRound`, after the link message and rep `r`'s earlier reps. -/
def plainQRound (spec : SessionSpec) (st : Stmt) (sch : Flock.Schedule) (r ℓ : ℕ) : ℕ :=
  r * repRounds false spec st +
    plainLevelRound st.m st.c.kLog sch.k0 (2 + spec.points * st.regions.size) (sch.levels.size - 1) ℓ + 1

/-- **The coin count of that round**, which the statement fixes (`plainRepNs`). -/
def plainQN (spec : SessionSpec) (st : Stmt) (sch : Flock.Schedule) (ℓ : ℕ) : ℕ :=
  (plainRepNs spec st)[plainLevelRound st.m st.c.kLog sch.k0 (2 + spec.points * st.regions.size)
    (sch.levels.size - 1) ℓ]!

/-- A later level's query round is a round of the rep. -/
theorem plainLevelRound_lt {spec : SessionSpec} {st : Stmt} {sch : Flock.Schedule} (hsch : fast100 st.m = some sch)
    {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) :
    plainLevelRound st.m st.c.kLog sch.k0 (2 + spec.points * st.regions.size) (sch.levels.size - 1) ℓ <
      repRounds false spec st := by
  rw [repRounds_plain hsch]
  unfold plainLevelRound levRound preLig
  dsimp only
  split <;> omega

/-- **Every drawn position within the level's queries has its row among the columns**: the converse of
`mem_zkRepCols`. -/
theorem mem_zkRepCols_of {lv : Flock.Level} {coins : Array F128} {rows : Array (Array F128)} {j : ℕ}
    (hj : j < (positions lv coins).size) (hr : j < rows.size) (hq : j < lv.q) :
    ((positions lv coins)[j], rows[j].toList.map FlockLevel3.GF128.of) ∈ Claims.Flock.zkRepCols lv coins rows := by
  unfold Claims.Flock.zkRepCols
  have hlen : j < ((positions lv coins).toList.zip (rows.toList.take lv.q)).length := by
    simp only [List.length_zip, List.length_take, Array.length_toList]
    omega
  refine List.mem_map.2 ⟨_, List.getElem_mem hlen, ?_⟩
  simp [List.getElem_zip]

/-- **An accepted plain rep opens a later-level position iff its query coin draws it.** At an accepted plain run, at
table 0's rep `r < 2` and level `1 ≤ ℓ` of the table's `fast100` schedule, some level-`ℓ` column is at `q` iff `q` is
among the level's positions at the first `plainQN` elements of the run's coin `plainQRound`. -/
theorem plain_levelCols_queriedK (keep : Bool → Bool) {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((V false).k x) → Coin} (h : (Claims.run sha512 P (VK keep) x ω c).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ}
    (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch)
    (hr : r < 2) (h1 : 1 ≤ ℓ) (hℓ : ℓ < sch.levels.size) (q : ℕ) :
    (∃ a, (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c).seen r ℓ) ↔
      q ∈ positions (sch.levels.getD ℓ default)
        ((Claims.coinSeq c (plainQRound spec st sch r ℓ)).vals (plainQN spec st sch ℓ)) := by
  obtain ⟨s', spec', sts', link, c₁, mids, f, hs', hf, hc, hv, hmids⟩ := plain_acc_midsK keep h
  obtain ⟨rfl, rfl, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = sts' := by simpa using hs.symm.trans hs'
  generalize (Claims.run sha512 P (VK keep) x ω c).vLog.map Prod.fst = L at hv
  obtain ⟨hx, lv, hlv, hcus⟩ := Claims.Bridge.Custody.checks_ok hc
  obtain ⟨hsts0, hst0⟩ := Array.getElem?_eq_some_iff.1 hst
  have h0 : 0 < (tabsOf sts).size := by simpa [tabsOf] using hsts0
  have ht0 : (tabsOf sts)[0] = Setup.ofCircuit st := by rw [tabsOf_getElem h0]; simp only [hst0]
  obtain ⟨sess, hdec, hsz, hreps⟩ :=
    plainLevels_verifyWithMK keep (hashesIn L) spec (tabsOf sts) f.record f.proofs (tabsOf_fold sts) hv
  obtain ⟨sch', hsch', hrep⟩ := hreps 0 h0
  obtain ⟨stream, extras, p, hstream, hext, hp, hlev⟩ := hrep r hr
  simp only [ht0] at hsch' hext hp hlev
  obtain rfl := Option.some.inj (hsch.symm.trans hsch')
  obtain ⟨-, hpts, -, hstreams⟩ := decodeWithK_live keep _ hdec hlv
  rw [hpts] at hext
  obtain ⟨hsz2, hall⟩ := rep_plainK keep _ hv hlv
  have hn : extras.size = st.regions.size * spec.points := by
    obtain ⟨sess', -, -, -, hpts', hpn', -⟩ := hall 0 (by omega)
    rw [extraClaims_size (st := st) hext, ← hpts', hpn']
  have hro : runOf false (Claims.run sha512 P (VK keep) x ω c).seen 0 = some
      { m := (Setup.ofCircuit st).m, kLog := (Setup.ofCircuit st).kLog, claims := 2 + extras.size,
        hashLen := (Setup.ofCircuit st).merkle.digestLen, saltLen := saltLenOf false (Setup.ofCircuit st),
        reps := [(lv.streams.getD 0 #[], f.proofs.getD 0 ByteArray.empty),
                 (lv.streams.getD 1 #[], f.proofs.getD 1 ByteArray.empty)] } := by
    simp [runOf, hf, hx, hs, hst, hlv, hext, Except.toOption]
  have hcols0 : plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c).seen r ℓ =
      plainLevelColsOf (Claims.Flock.plainProofShape (Setup.ofCircuit st).m (Setup.ofCircuit st).kLog
        (2 + extras.size) (Setup.ofCircuit st).merkle.digestLen (saltLenOf false (Setup.ofCircuit st)))
        (lv.streams.getD r #[]) (f.proofs.getD r ByteArray.empty) ℓ := by
    unfold plainRawLevelCols
    rw [hro]
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
  have hcolsEq : plainRawLevelCols (Claims.run sha512 P (VK keep) x ω c).seen r ℓ =
      Claims.Flock.zkRepCols (sch.levels.getD ℓ default) (y.rounds[J]!).coins
        (openAt p.ligerito (sch.levels.size - 1) ℓ).1 := by
    rw [hcols0, ← hent]
    have e1 : lv.streams.getD r #[] = lv.streams[2 * 0 + r]! := by
      rw [show (#[] : Array Flock.Round) = default from rfl, getD_eq_getElem!]; simp
    have e2 : f.proofs.getD r ByteArray.empty = f.proofs[2 * 0 + r]! := by
      rw [show ByteArray.empty = default from rfl, getD_eq_getElem!]; simp
    rw [e1, e2]
    refine plainLevelColsOf_congr _ _ _ fun j => ?_
    rw [← Array.getElem?_toList, ← Array.getElem?_toList]
    exact (hcoins j).symm
  obtain ⟨-, hsizes⟩ := streams_size_plainK keep _ hv hlv
  have hR0 : ∀ k (hk : k < lv.streams.size), k < 2 → lv.streams[k].size = repRounds false spec st := by
    intro k hk hk2
    rw [hsizes k hk, show k / 2 = 0 by omega, getElem!_pos sts 0 hsts0, hst0]
  have hq2 : 2 ≤ lv.streams.size := by omega
  have hjr : J < (lv.streams[r]'(by omega)).size := by rw [hR0 r _ hr]; exact hJR
  have hfl := (Claims.Bridge.Custody.custody_ok hcus).2.2.2
  have e1 := flat_two_getElem? lv.streams hq2 hr hjr
  rw [hR0 0 (by omega) (by omega), Array.getElem?_eq_getElem hjr] at e1
  obtain ⟨am, hm, hram⟩ := forall₂_getElem?_some hfl e1
  have hk : r * repRounds false spec st + J < plainN x := by
    have := (List.getElem?_eq_some_iff.1 hm).1
    rw [hmids, List.length_map, List.length_range] at this
    exact this
  rw [hmids, List.getElem?_map, List.getElem?_range hk] at hm
  simp only [Option.map_some, Option.some.injEq] at hm
  subst hm
  obtain ⟨-, hns⟩ := streams_ns_plainK keep _ hv hlv
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
    rw [getElem!_pos y.rounds J (by omega), hfj.2.1, getElem!_pos _ J hjr, hram.2, hnJ]
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
      rcases checks_leaf hc (setupOf_some hs).2 0 hsts0 with e | ⟨-, e⟩
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

/-- `plain_levelCols_queriedK` at the deployed plain verifier (`keep := id`). -/
theorem plain_levelCols_queried {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {c : Fin ((V false).k x) → Coin} (h : (Claims.run sha512 P (V false) x ω c).out = .acc) {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {st : Stmt} {sch : Flock.Schedule} {r ℓ : ℕ}
    (hs : setupOf false x = some (s, spec, sts)) (hst : sts[0]? = some st) (hsch : fast100 st.m = some sch)
    (hr : r < 2) (h1 : 1 ≤ ℓ) (hℓ : ℓ < sch.levels.size) (q : ℕ) :
    (∃ a, (q, a) ∈ plainRawLevelCols (Claims.run sha512 P (V false) x ω c).seen r ℓ) ↔
      q ∈ positions (sch.levels.getD ℓ default)
        ((Claims.coinSeq c (plainQRound spec st sch r ℓ)).vals (plainQN spec st sch ℓ)) :=
  plain_levelCols_queriedK id (Ω := Ω) (P := P) (x := x) (ω := ω) (c := c) (h := h) (s := s) (spec := spec) (sts := sts) (st := st) (sch := sch) (r := r) (ℓ := ℓ) (hs := hs) (hst := hst) (hsch := hsch) (hr := hr) (h1 := h1) (hℓ := hℓ) (q := q)

end FlockDeploy.KS
