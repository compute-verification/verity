import Proofs.Flock.Claims.KS.PlainDrawRColsE
import Proofs.Flock.Claims.KS.PlainKSLin
import Proofs.Flock.Claims.Recursion.FlockKSDeployE

/-!
# The plain record at the deployed reader `VR rd`, and its code facts

`deployPlainR rel rd` is the plain record with the deployed reader `VR rd` (`nextR rd` at `V false`'s rounds) as its
verifier and the reader's columns on a view (`colsRV rd`) as its opened columns; `deployPlainTrimR rel rd` is it as the
claims read it, its columns and code trimmed of trailing zeros. The bound record's element-level record at
`nextE := nextR rd`, `colsE := (deployPlainTrimR rel rd).cols` and `wE := (deployPlainTrimR rel rd).weight` is it
(`RecDeploy.deE_R_trim`, as `deE_VE_trim`).

Its code facts, each from `VR rd`'s decision at SHA-512 (`decideR_sha512`), by `VE`'s proofs (`FlockKSDeployE`):

* it accepts only served tables in scope (`nextR_acc_check`, `accR_inScope`), so at one table (`plainR_hone`) and
  with `k ≥ 8` (`kR_range`, `plainR_kSplits`);
* an accepted run's statement is its last message's (`accR_last`), so the reader's columns are table 0's of the
  statement and have the zero column's width (`accR_cols`, `plainR_colsOk`);
* it accepts only at its last round (`accLast_VR`): custody's count of the record's rounds (`streams_size_plainJ`)
  is the round messages', one per opened round (`readPs_length`);
* `plainR_KSFlockLin`: `KSFlockLin` at the trimmed record and `CpDep`, from conflicts tied to collisions and the table
  bound at the untrimmed record (`plainE_KSFlockLin`'s assembly).
-/

namespace FlockDeploy

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-- **The plain record at the deployed reader**: `deployPlain rel` with `VR rd` as its verifier and the reader's
columns on a view (`colsRV rd`) as its opened columns. -/
noncomputable def deployPlainR (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace :=
  { deployPlain rel with V := VR rd, cols := colsRV rd }

/-- **The plain record at the deployed reader, as the claims read it**: `deployPlainR rel rd` with its opened columns
and its code trimmed of trailing zeros. -/
noncomputable def deployPlainTrimR (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.Deploy Coin (List 𝔽) Claims.Flock.PlainTrace :=
  (deployPlainR rel rd).mapCols trimCol

end FlockDeploy

namespace RecDeploy

/-- **The element-level record at the deployed reader is the trimmed plain record there.** -/
theorem deE_R_trim (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (rd : Claims.Msg → Option FlockDeploy.FinalP) :
    deE rel (FlockDeploy.nextR rd) (FlockDeploy.deployPlainTrimR rel rd).cols
        (FlockDeploy.deployPlainTrimR rel rd).weight = FlockDeploy.deployPlainTrimR rel rd :=
  rfl

end RecDeploy

namespace FlockDeploy.KS

open Flock FlockSoundness.Discharge.Exec
open FlockSoundness.Refine (ebind_ok)
open Claims.Bridge.Custody (checksAtJ_ok checksAtJ_scope)
open Claims.Bridge.PlainDraw (decideR_sha512)
open Claims.Recursion

local notation "𝔽" => FlockLevel3.GF128

/-! ## Scope -/

/-- **An accepting step of `VR rd` at SHA-512** is at a statement that sets up, serves only `fast100`'s
`22 ≤ m ≤ 35` and passed the scope check (`decideR_sha512`, `verifyJWithMK_tables_cap`, `checksAtJ_scope`). -/
theorem nextR_acc_check {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : ((nextR rd x as cs).eval sha512).1 = .acc) :
    ∃ ins spec sts, setupOf false x = some (ins, spec, sts) ∧ 0 < sts.size ∧
      (∀ (i : ℕ) (hi : i < sts.size), 22 ≤ sts[i].m ∧ sts[i].m ≤ 35) ∧ ins.read.check = .ok () := by
  cases ha : actR rd x as cs with
  | rej => revert h; simp [nextR, ha, Claims.OC.eval]
  | coin c => revert h; simp [nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨hv, hc, -, -, -⟩ := decideR_sha512 ha h
    obtain ⟨_, schs, -, -, hpos, -, hall⟩ := verifyJWithMK_tables_cap id With.std hv
    have hsz : (tabsOf sts).size = sts.size := Array.size_map ..
    refine ⟨s, spec, sts, actR_setupOf ha, hsz ▸ hpos, fun i hi => ?_, checksAtJ_scope hc⟩
    obtain ⟨_, _, hf, -⟩ := hall i (hsz ▸ hi)
    have hm : (tabsOf sts)[i].m = sts[i].m := by simp [tabsOf, Setup.ofCircuit]
    exact hm ▸ fast100_range hf

theorem accR_check {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (VR rd) x) :
    ∃ ins spec sts, setupOf false x = some (ins, spec, sts) ∧ 0 < sts.size ∧
      (∀ (i : ℕ) (hi : i < sts.size), 22 ≤ sts[i].m ∧ sts[i].m ≤ 35) ∧ ins.read.check = .ok () := by
  unfold Claims.Flock.NoAcc at hacc
  push Not at hacc
  obtain ⟨Ω, P, ω, r, hr⟩ := hacc
  obtain ⟨as, has⟩ := FlockDeploy.KS.stepWith_acc _ _ _ _ _ _ _ _ _ hr
  exact nextR_acc_check has

/-- **`VR rd` accepts only in scope**, at SHA-512: a statement some run accepts parses, in scope (`accE_inScope`'s
proof, at `nextR_acc_check`). -/
theorem accR_inScope (rel : Readable.Parsed → Readable.Relation) {rd : Claims.Msg → Option FinalP} {x : Claims.Msg}
    (hacc : ¬ Claims.Flock.NoAcc sha512 (VR rd) x) :
    ∃ s, parse false rel x = some s ∧ Claims.Flock.InScope false s := by
  obtain ⟨ins, spec, sts, hset, h0, hm, hc⟩ := accR_check hacc
  obtain ⟨hi, hs⟩ := setupOf_some hset
  have hr : readOf false x = some ins.read := by unfold readOf; rw [hi]; rfl
  have hzk : ins.read.zk = false := readOf_zk hr
  have hrp := read_parse_of_setup hs
  have hparse : parse false rel x = some (Readable.toReadable rel { ins.read with sts }) := by
    unfold parse Readable.parseOf Readable.parseBytes
    rw [hr, Option.bind_some, hrp, Option.map_some]
  refine ⟨_, hparse, ?_⟩
  have hin := (Readable.inScope_iff rel (Readable.parses_of_parse hrp)).2 ⟨hc, ?_⟩
  · exact hzk ▸ hin
  · simpa only [Readable.toReadable, Array.getElem?_eq_getElem h0, Option.map_some, Option.getD_some] using
      (hm 0 h0).1

/-- **`VR rd` accepts only at one table**, at SHA-512: the plain scope admits one (`accR_inScope`). -/
theorem plainR_hone (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) :
    (∃ s, (deployPlainR rel rd).parse x = some s ∧ s.tables = 1) ∨
      Claims.Flock.NoAcc sha512 (deployPlainR rel rd).V x := by
  by_cases hno : Claims.Flock.NoAcc sha512 (VR rd) x
  · exact .inr hno
  · obtain ⟨s, hs, hin⟩ := accR_inScope rel hno
    obtain ⟨-, -, -, -, -, -, -, -, ht, -⟩ := hin
    exact .inl ⟨s, hs, (Claims.Flock.Stmt.tables_eq_one_iff s).2 (by simpa using ht)⟩

/-- **`VR rd`'s `k` range**: at every statement, `k ≥ 8` or `VR rd` accepts no run at SHA-512 (`kE_range`'s proof). -/
theorem kR_range (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) (x : Claims.Msg) :
    8 ≤ (deployPlainR rel rd).k x ∨ Claims.Flock.NoAcc sha512 (deployPlainR rel rd).V x := by
  by_cases hacc : Claims.Flock.NoAcc sha512 (VR rd) x
  · exact .inr hacc
  obtain ⟨s, spec, sts, hs, h0, hm, -⟩ := accR_check hacc
  obtain ⟨h22, h35⟩ := hm 0 h0
  have hd := (Claims.Flock.plainShape_table ⟨sts[0].m - 22, by omega⟩)
  have hi : Claims.Flock.PlainIdx.m ⟨sts[0].m - 22, by omega⟩ = sts[0].m := by
    simp only [Claims.Flock.PlainIdx.m]
    omega
  have hlen : 10 ≤ (Claims.Flock.plainLevel0 sts[0].m).d := by
    unfold Claims.Flock.plainLen Claims.Flock.plainCols at hd
    rw [hi] at hd
    omega
  refine .inl (eight_le_k ?_)
  show 2 ^ 10 ≤ N₀ false x
  exact le_trans (two_pow_le hlen) (N₀_ge_of_setup hs h0)

/-- **The trimmed record at `VR rd`'s `k` range, with the relation split**, where the statement has one table wherever
`VR rd` accepts some run (`plainR_hone`). -/
theorem plainR_kSplits (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (x : Claims.Msg) :
    (8 ≤ (deployPlainTrimR rel rd).k x ∧ Claims.Flock.Splits (deployPlainTrimR rel rd) x) ∨
      Claims.Flock.NoAcc sha512 (deployPlainTrimR rel rd).V x :=
  (kR_range rel rd x).elim
    (fun hk => ((plainR_hone rel rd x).imp_left fun ⟨_, hs, h1⟩ =>
      splits_of_one (deployPlainTrimR rel rd) rfl rfl hs h1).imp (fun h => ⟨hk, h⟩) id) .inr

/-! ## Columns -/

/-- **An accepted run of `VR rd` ends on a message the reader reads at the statement**, at SHA-512: `rd` reads its last
message, on the run's own statement (`decR_rd`, `checksAtJ_ok`). -/
theorem accR_last {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) :
    ∃ fl, (((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst).getLast?).bind rd = some fl ∧ fl.stmt = x := by
  have ha := FlockDeploy.KS.run_acc_next sha512 P (VR rd) x ω r h
  generalize (List.range _).map (Claims.coinSeq r) = cs at ha
  replace ha : ((nextR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs).eval sha512).1 = .acc := ha
  cases hact : actR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst) cs with
  | rej => revert ha; simp [nextR, hact, Claims.OC.eval]
  | coin c => revert ha; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨-, hc, -⟩ := decideR_sha512 hact ha
    obtain ⟨fl, hfl, hfs⟩ := decR_rd (actR_decide hact)
    exact ⟨fl, hfl, hfs.symm.trans (checksAtJ_ok hc).1⟩

/-- **`runOfDecR`'s run is at its table's `m`.** -/
theorem runOfDecR_m {d : DecR} {j : ℕ} {rr : Claims.Flock.ZkRun} (h : runOfDecR d j = some rr) :
    ∃ st, d.sts[j]? = some st ∧ rr.m = st.m := by
  unfold runOfDecR at h
  simp only [Option.bind_eq_bind, Option.pure_def] at h
  obtain ⟨st, hst, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  obtain ⟨_, -, h⟩ := Option.bind_eq_some_iff.1 h
  refine ⟨st, hst, ?_⟩
  rw [← Option.some.inj h]
  simp [Setup.ofCircuit]

/-- **An accepted run of `VR rd` opens columns of the statement's table 0**: each has its level-0 width
(`accE_cols`'s proof, at the reader's columns). -/
theorem accR_cols {rd : Claims.Msg → Option FinalP} {Ω : Type} {P : Claims.Prover Ω} {x : Claims.Msg} {ω : Ω}
    {r : Fin ((VR rd).k x) → Coin} (h : (Claims.run sha512 P (VR rd) x ω r).out = .acc) {q : ℕ} {u : List 𝔽}
    (hu : Claims.Flock.lookCol (colsRV rd (Claims.run sha512 P (VR rd) x ω r).seen 0) q = some u) :
    ∃ spec st, (tables false x).head? = some (spec, st) ∧
      u.length = 2 ^ (if false then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes := by
  obtain ⟨fl, hfl, hx⟩ := accR_last h
  unfold colsRV at hu
  rw [show (Claims.run sha512 P (VR rd) x ω r).seen.msgs = (Claims.run sha512 P (VR rd) x ω r).msgs from rfl, hfl]
    at hu
  dsimp only at hu
  rw [hx] at hu
  cases hd : decR rd x ((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.fst)
      (((Claims.run sha512 P (VR rd) x ω r).msgs.map Prod.snd).map RecDeploy.outCoin) with
  | none => rw [hd] at hu; simp [Claims.Flock.lookCol] at hu
  | some d =>
    rw [hd] at hu
    cases hrr : runOfDecR d 0 with
    | none => simp [colsOf, hrr, Claims.Flock.lookCol] at hu
    | some rr =>
      simp only [colsOf, hrr] at hu
      obtain ⟨st, hst, hm⟩ := runOfDecR_m hrr
      obtain ⟨⟨s, spec, sts⟩, _, _, _, _, _, _, hv, -, -, -, -, -, hdd⟩ := decR_some hd
      have hst' : sts[0]? = some st := by rw [hdd] at hst; exact hst
      refine ⟨spec, st, tables_head hv hst', ?_⟩
      obtain ⟨p, hp, rfl⟩ := exists_of_lookCol hu
      simp only [Bool.false_eq_true, ↓reduceIte]
      rw [plainRunCols_width hp, hm]

/-- **`ColsOk` at the plain record at `VR rd`**: an accepted run opens only columns of the zero column's width. -/
theorem plainR_colsOk (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP) :
    Claims.Flock.ColsOk sha512 (deployPlainR rel rd) (wordLen false) (wide false) := by
  intro x Ω P ω r q u hacc _ hu
  obtain ⟨spec, st, hh, hl⟩ := accR_cols hacc hu
  unfold wide
  rw [hl, zeroCol_length hh]

/-! ## The round count -/

theorem cutLens_length {α : Type} : ∀ (ns : List ℕ) (l : List α), (cutLens ns l).length = ns.length
  | [], _ => rfl
  | n :: ns, l => by simp [cutLens, cutLens_length ns]

/-- **`readPs` reads one round message per opened round**: each rep's frames are as many as its block of rows
(`FitsRep.rounds`), and the blocks cut all the rounds (`cutLens_flatten`). -/
theorem readPs_length {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {r₀ : List ByteArray}
    {rss : List (List ByteArray)} {cs : List Coin} {fl : FinalP} {pf : Array Flock.Proof × List Claims.Msg}
    (h : readPs spec sts c₁ r₀ rss cs fl = some pf) : pf.2.length = rss.length := by
  unfold readPs at h
  by_cases hc : fl.ps.size = 2 * sts.size ∧ (repLens spec sts).sum = rss.length
  · rw [if_pos hc] at h
    obtain ⟨rs, hrs, rfl⟩ := Option.map_eq_some_iff.1 h
    obtain ⟨hlen, hget⟩ := mapM_some_get hrs
    rw [List.length_range] at hlen
    have hcl : (cutLens (repLens spec sts) rss).length = 2 * sts.size := by
      rw [cutLens_length, repLens, List.length_map, List.length_range]
    have hflat := cutLens_flatten (repLens spec sts) rss hc.2.ge
    dsimp only
    rw [List.length_flatten, List.map_map]
    conv_rhs => rw [← hflat, List.length_flatten]
    congr 1
    apply List.ext_getElem
    · simp [hlen, hcl]
    · intro k h₁ h₂
      have hk : k < rs.length := by simpa using h₁
      have hk' : k < (cutLens (repLens spec sts) rss).length := by simpa using h₂
      have hg := hget k (by rw [List.length_range]; omega) hk
      rw [List.getElem_range] at hg
      obtain ⟨T, hT, hms⟩ := readRep_fits hg
      simp only [List.get_eq_getElem] at hT hms
      simp only [List.getElem_map, Function.comp_apply]
      rw [hms, framesOf, List.length_map, hT.rounds, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hk',
        Option.getD_some]
  · rw [if_neg hc] at h
    cases h

/-- **Each plain rep stream of an accepted parsed record has its table's `repRounds`** (`streams_size_plainK`'s
proof, at `verifyJWithMK` and `liveOfJ`): where `liveOfJ` reads the link's points of the session's count, there are
two rep streams per table, and stream `k` (table `k / 2`'s rep `k % 2`) has `repRounds false spec sts[k / 2]`
rounds. -/
theorem streams_size_plainJ (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {sts : Array _root_.Flock.Stmt}
    {j : Lean.Json} {proofs : Array ByteArray} {lv : Live}
    (hv : verifyJWithMK keep hs spec (tabsOf sts) j proofs = .ok ()) (hl : liveOfJ spec j = .ok lv)
    (hpn : lv.points.size = spec.points) :
    lv.streams.size = 2 * sts.size ∧
      ∀ (k : ℕ) (hk : k < lv.streams.size), lv.streams[k].size = repRounds false spec sts[k / 2]! := by
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
  have hlen : lv.streams[k].size = y.rounds.size := by
    have e := hfa.length_eq
    rw [getElem!_pos lv.streams k hk] at e
    simpa using e.symm
  obtain ⟨stream', extras, p, hfind', hex, -, hsize, -⟩ := verifyRepWith_rounds hs rfl hf hrep
  rw [hfind] at hfind'
  have hys := Option.some.inj hfind'
  have hx : extras.size = sts[k / 2]!.regions.size * sess.points.size := FlockDeploy.extraClaims_size hex
  rw [hlen, hys, hsize]
  unfold repRounds
  simp only [Bool.false_eq_true, ↓reduceIte]
  rw [show fast100 sts[k / 2]!.m = some schs[k / 2] from hf, hx, hpts, hpn, Nat.mul_comm]
  all_goals rfl

/-- **At SHA-512, `VR rd` accepts only at its last round** (`accLast_VE`'s count, at `nextR`'s decision): custody
holds one recorded round per round message (`custody_length`), the record's rounds are `rounds` (`streams_size_plainJ`),
and the round messages are the rounds after the link but the last (`readPs_length`), so the messages are
`2 + rounds`; table 0's rounds make `rounds` positive. -/
theorem accR_length (rd : Claims.Msg → Option FinalP) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (hcs : cs.length = as.length) (h : ((nextR rd x as cs).eval sha512).1 = .acc) :
    ∃ s spec sts, setupOf false x = some (s, spec, sts) ∧ as.length = 2 + rounds false spec sts ∧
      0 < rounds false spec sts := by
  cases ha : actR rd x as cs with
  | rej => revert h; simp [nextR, ha, Claims.OC.eval]
  | coin c => revert h; simp [nextR, ha, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    obtain ⟨hv, hc, -, -, -⟩ := decideR_sha512 ha h
    obtain ⟨-, lv, hlv, hcus⟩ := checksAtJ_ok hc
    have hpn : lv.points.size = spec.points := by
      rw [liveOfJ_points spec c₁ mids _ hlv]
      simp [pointsOf]
    obtain ⟨hsz, hk⟩ := streams_size_plainJ id With.std hv hlv hpn
    have hmr : mids.length = rounds false spec sts := by
      rw [← custody_length hcus, rounds_sum, List.length_flatMap, ← sum_pairs (fun i => repRounds false spec sts[i]!)]
      congr 1
      apply List.ext_getElem
      · simp [hsz]
      · intro k h₁ h₂
        simp only [List.getElem_map, List.getElem_range, Array.getElem_toList, Array.length_toList]
        exact hk k _
    obtain ⟨_, schs, -, -, hpos, -, hall⟩ := verifyJWithMK_tables_cap id With.std hv
    have hs0 : 0 < sts.size := by simpa [tabsOf] using hpos
    obtain ⟨_, _, hf, -⟩ := hall 0 hpos
    have hm0 : (tabsOf sts)[0].m = sts[0].m := by simp [tabsOf, Setup.ofCircuit]
    rw [hm0, ← getElem!_pos sts 0 hs0] at hf
    have hr0 : 0 < repRounds false spec sts[0]! := by
      unfold repRounds
      simp only [Bool.false_eq_true, ↓reduceIte, hf]
      omega
    have hrpos : 2 * repRounds false spec sts[0]! ≤ rounds false spec sts := by
      rw [rounds_sum]
      obtain ⟨n, hn⟩ : ∃ n, sts.size = n + 1 := ⟨sts.size - 1, by omega⟩
      simp only [hn, List.range_succ_eq_map, List.map_cons, List.sum_cons]
      omega
    obtain ⟨⟨s', spec', sts'⟩, c₁', r₀, rss, fl, link', pf, hso, -, -, h7, -, h9, hd⟩ := decR_some (actR_decide ha)
    simp only [DecR.mk.injEq] at hd
    obtain ⟨rfl, rfl, rfl, -, rfl, rfl, -⟩ := hd
    have hrss : rss.length = as.length - 2 := by
      rw [(mapM_some_get h7).1, List.length_dropLast, List.length_tail]
      omega
    have hmids : (pf.2.zip cs.tail.dropLast).length = as.length - 2 := by
      rw [List.length_zip, readPs_length h9, hrss, List.length_dropLast, List.length_tail, hcs]
      omega
    have key : rounds false spec sts = as.length - 2 := hmr.symm.trans hmids
    have hrp : 0 < rounds false spec sts := by omega
    refine ⟨_, _, _, hso, ?_, hrp⟩
    rw [key] at hrp ⊢
    omega

/-- **At SHA-512, `VR rd` accepts only at its last round** (`accR_length`). -/
theorem accLast_VR (rd : Claims.Msg → Option FinalP) : AccLast sha512 (VR rd) := fun x as cs hcs h => by
  obtain ⟨s, spec, sts, hso, hl, -⟩ := accR_length rd hcs h
  show as.length = (match setupOf false x with | some (_, spec, sts) => 2 + rounds false spec sts | none => 1)
  rw [hso]
  dsimp only
  exact hl

/-- **At SHA-512, `VR rd` decides without the round's coin** (`decideFree_VK`'s proof, at `nextR`): an accepting
decision is past the link's coin (`accLast_VR`), its last message asks no coin, and `decR` reads only the coins before
the last (`decR`'s `cs.head?` and `cs.tail.dropLast`). -/
theorem decideFree_VR (rd : Claims.Msg → Option FinalP) : DecideFree sha512 (VR rd) := by
  intro x as cs c c' hl hacc
  show (nextR rd x as (cs ++ [c'])).eval sha512 = (nextR rd x as (cs ++ [c])).eval sha512
  have hacc' : ((nextR rd x as (cs ++ [c])).eval sha512).1 = .acc := hacc
  cases hact : actR rd x as (cs ++ [c]) with
  | rej => revert hacc'; simp [nextR, hact, Claims.OC.eval]
  | coin c₂ => revert hacc'; simp [nextR, hact, Claims.OC.eval]
  | decide s spec sts link c₁ mids f =>
    have hso := actR_setupOf hact
    obtain ⟨s', spec', sts', -, hk, hkp⟩ := accR_length rd (x := x) (as := as) (cs := cs ++ [c]) (by simp [hl]) hacc'
    have hcl : 0 < cs.length := by
      have e := hl.symm.trans hk
      have h1 : 1 < cs.length + 1 := by
        rw [e]
        exact Nat.lt_of_lt_of_le (by decide) (Nat.le_add_right 2 _)
      exact Nat.lt_of_succ_lt_succ h1
    obtain ⟨c₀, cs', rfl⟩ : ∃ c₀ cs', cs = c₀ :: cs' := by
      rcases cs with _ | ⟨c₀, cs'⟩
      · exact absurd hcl (Nat.lt_irrefl 0)
      · exact ⟨c₀, cs', rfl⟩
    have ha : ∀ t, as.getLast? ≠ some (0 :: t) := fun t hlast => by
      have hcoin : act false x as (c₀ :: cs' ++ [c]) = .coin c := by
        unfold act
        rw [hso]
        simp only [List.getLast?_concat, hlast]
      have e := hact
      unfold actR at e
      rw [hcoin] at e
      cases e
    have hd : decR rd x as (c₀ :: cs' ++ [c']) = decR rd x as (c₀ :: cs' ++ [c]) := by
      unfold decR
      simp only [List.cons_append, List.head?_cons, List.tail_cons, List.dropLast_concat]
    unfold nextR actR
    rw [act_cons_snoc false x as c₀ cs' c c' ha, hd]

/-! ## `KSFlockLin` at the trimmed record -/

/-- **`KSFlockLin` at the plain record at `VR rd`**, at SHA-512, at the trimmed record and the deployed caps `CpDep`:
`KSFlockLin_of_tableBoundAbortLin` from conflicts tied to collisions and the table bound at the untrimmed record,
carried to the trimmed columns (`tableBoundAbortLin_mapCols`), with the record's numbers, one table wherever some run
accepts (`plainR_hone`) and `ColsOk` (`plainR_colsOk`), as `plainE_KSFlockLin`. -/
theorem plainR_KSFlockLin (rel : Readable.Parsed → Readable.Relation) (rd : Claims.Msg → Option FinalP)
    (hconf : Claims.Flock.ConflictTied sha512 (deployPlainR rel rd) (wordLen false))
    (htab : Claims.Recursion.TableBoundAbortLin sha512 (deployPlainR rel rd) (wordLen false) (wide false) CpDep) :
    Claims.Recursion.KSFlockLin sha512 (deployPlainTrimR rel rd) CpDep :=
  Claims.Recursion.KSFlockLin_of_tableBoundAbortLin sha512 (deployPlainTrimR rel rd) (wordLen false)
    (trimWide false) CpDep (wordLen_le_N₀ false) (fun x => εtable_nonneg false x) (fun _ => cap_pos)
    (plainR_kSplits rel rd) (plainR_hone rel rd) (trimWide_nil false)
    (Claims.Flock.colsOk_mapCols sha512 (deployPlainR rel rd) (wordLen false) trimCol (wide false)
      (plainR_colsOk rel rd))
    (Claims.Flock.conflictTied_mapCols sha512 (deployPlainR rel rd) (wordLen false) trimCol hconf)
    (Claims.Recursion.tableBoundAbortLin_mapCols sha512 (deployPlainR rel rd) (wordLen false) trimCol rfl
      (wide false) CpDep (plainR_colsOk rel rd) (wide_trimCol_inj false) htab)

end FlockDeploy.KS
