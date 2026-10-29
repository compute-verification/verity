import Proofs.Flock.Claims.Recursion.RowDLevels
import Proofs.Flock.Claims.Recursion.RoundCounts
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.NextRCoins
import Proofs.Flock.Claims.Recursion.NextRFits
import Proofs.Flock.Claims.Recursion.HonestVerdictR

/-!
# Row D's `hlev` at the honest recursive prover

`DecAt.hlev_of_repP` (`RowDLevels`) gives row D's `hlev` at a rep whose decided proof is a returning honest rep's
(`hrun`), on a `commitP` of its table (`h0`), whose replies' coins at the queried rounds are the rep's recorded ones
(`hcoin`). Here the three hold at the honest prover of `nextR`: the pure session `sessionS`, whose replies are the
coins' bytes, at a decision on its honest messages (`honestFileMsgs`).

* **The decision is the honest one** (`honest_decR_files`): on the honest messages, `decR` decides at the statement's spec,
  tables and first coin, and its proofs are the ones the session's files decode to (`filePs`), by `readsBack`.
* **`hrun` and `h0`** (`DecAt.hlev_honest_of`): rep `r` of table `i` is the session's rep `2 i + r` (`sessionP_repRuns`),
  whose file decodes to its proof (`repShapedOk`), at the table's setup, `fast100` schedule and extra claims at the
  link's points (`points_eq`), and the level-0 commitment the session made.
* **`hcoin`**: wherever the verifier rows run (where `hlev` is asked), the round `lgStart + qOff sch ℓ` has level `ℓ`'s
  `q` coins (`rows_sz`); the recorded round there is the live coin `canonRounds` reads (`rounds_canon`), and the reply
  after the link's send and the earlier reps' rounds (`sendsP_eq`) is that coin's bytes.
* **Row D's `hlev` at the honest run** (`DecAt.hlev_honest`): all of it, with the widths `hn` and `hlf` from the table's
  registered scheme (`opsWith_sizes`); only `hops` stays.
-/

namespace FlockVBridge.Family

open Flock FlockSoundness.Refine
open FlockComplete.Replay.Lg (qOff)

/-- **Each queried round has its level's `q` coins**: at any run of `verifierRows` on rounds with no empty round, at
`m ≥ 14` and a schedule of the seams, round `lgStart + qOff sch ℓ` has level `ℓ`'s `q` coins (`ligeritoRun_pos`). -/
theorem rows_sz {sh : Shape} {rounds : Array (Array F128)} {extras : Array Claim} {comb v : Array F128}
    {info : Flock.RecAlgebra.Info Flock.RecAlgebra.Lin} {kappa : Array (Array (Array F128))}
    (hne : Flock.RecAlgebra.NoEmpty rounds) (hm : 14 ≤ sh.m) (hdim : DimSeam sh.sched)
    (hlv : ∀ i < sh.sched.levels.size, LevelSeam (sh.sched.levels.getD i default))
    (h : Flock.RecAlgebra.verifierRows sh rounds extras comb = .ok (v, info, kappa)) :
    ∀ ℓ < sh.sched.levels.size,
      (rounds[lgStart sh extras.size + qOff sh.sched ℓ]!).size = (sh.sched.levels.getD ℓ default).q := by
  have : Fact (Flock.RecAlgebra.NoEmpty rounds) := ⟨hne⟩
  obtain ⟨lins, k, hr⟩ := FlockVBridge.Runs.run_of_rows h
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, h2, h3, -⟩ := Flock.RecAlgebra.run_stages hr
  obtain ⟨-, -, -, -, -, -, -, -, -, hk1, -⟩ := Flock.RecAlgebra.zerocheck_coins hm h0
  obtain ⟨-, -, -, -, -, -, -, -, -, -, -, -, -, -, hk2, -⟩ := Flock.RecAlgebra.lincheck_coins h1
  have hcs : (Flock.RecAlgebra.claimsOf sh.m sh.kLog info.zc info.lc extras).size = 2 + extras.size := by
    simp [Flock.RecAlgebra.claimsOf]
  obtain ⟨_, -, -, -, hk3, -, -, -⟩ := Flock.RecAlgebra.ringSwitch_spec (by omega) h2
  have hzc : (Flock.RecAlgebra.symbolic sh).zc.size = sh.m - 6 := by simp [Flock.RecAlgebra.symbolic]
  have hlc : (Flock.RecAlgebra.symbolic sh).lc.size = sh.kLog - 6 := by simp [Flock.RecAlgebra.symbolic]
  rw [hzc] at hk1
  rw [hlc] at hk2
  rw [hcs] at hk3
  have hks : k3 = lgStart sh extras.size := by
    unfold lgStart
    omega
  have h3' := Flock.RecAlgebra.ligerito_sim (Flock.RecAlgebra.Lin.eval_hom fun _ => 0) _ _ _ _ _ _ _ _ h3
  obtain ⟨st₀, out, stV, k₄, hst0, hrun, -⟩ := ligerito_run_conss h3'
  obtain ⟨-, hsz⟩ := ligeritoRun_pos hst0 hdim hlv hrun
  intro ℓ hℓ
  rw [← hks]
  exact hsz ℓ hℓ

end FlockVBridge.Family

namespace FlockDeploy

open Flock FlockComplete Rows

/-- **The honest prover's messages at its own files**: each table's level-0 cap (`fileCaps`), each rep's rows of the
proof its file decodes to (`filePs`) at its batched value `Ts k`, then the last message. -/
def honestFileMsgs (spec : SessionSpec) (sts : Array _root_.Flock.Stmt) (c₁ : Coin) (pfs : Array ByteArray)
    (cs : List Coin) (Ts : ℕ → F128) (last : Claims.Msg) : List Claims.Msg :=
  honestMsgs (fileCaps spec sts c₁ pfs) ((List.range (filePs spec sts c₁ pfs).size).map fun k =>
    repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) (filePs spec sts c₁ pfs)[k]!
      (Ts k)) last

/-- **The decision on the honest messages is the honest one**: at a statement that sets up, at a `sessionS` run whose
replies are the coins' bytes, for coins that number at least the reps' rounds and two, at `TableOk` tables with at
most 2,048 extra claims at the first coin's link points, where the last message reads as `filePs`' openings
(`opensOf`, what `rdR` returns) at the statement, `decR` decides at the statement's spec, tables and first coin, with
the proofs `filePs` (`readsBack`, through `decR_opensOf`). -/
theorem honest_decR_files {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {rd : Claims.Msg → Option FinalP}
    {x last : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} {c₁ : Coin}
    {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : Tape} {fl : FinalP} (Ts : ℕ → F128)
    (hx : setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) (hrd : rd last = some fl)
    (hps : fl.ps = (filePs spec sts c₁ pfs).map RecDeploy.DeployedR.opensOf) (hst : fl.stmt = x) :
    ∃ d, decR rd x (honestFileMsgs spec sts c₁ pfs cs Ts last) cs = some d ∧ d.spec = spec ∧ d.sts = sts ∧
      d.c₁ = c₁ ∧ d.f.ps = filePs spec sts c₁ pfs := by
  have hc : resp 0 = c₁.bytes := by
    cases cs with
    | nil => simp at hcs
    | cons c cs' =>
      simp only [List.head?_cons, Option.some.injEq] at hcs
      subst hcs
      simpa using hresp 0 (by simp)
  obtain ⟨h64t, hts⟩ := setup_tables hx
  have hfc := filesFinalC h64 hc hts.symm htab hS hresp hlen
  obtain ⟨hsz, hall⟩ := sessionS_reps h64 hc hts.symm htab hS (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast
  have hsh : ShapedAt spec sts c₁ (filePs spec sts c₁ pfs) := fun k hk => by
    have hk' : k < 2 * sts.size := by
      have h := hk
      rw [filePs_size, hsz] at h
      exact h
    have h := (hall k hk').2.2.2.2.2.2
    rwa [getElem!_pos (filePs spec sts c₁ pfs) k hk] at h
  have hcaps : ∀ i < (fileCaps spec sts c₁ pfs).size,
      (fileCaps spec sts c₁ pfs)[i]!.size = RecDeploy.level0Nodes sts[i]! := by
    intro i hi
    rw [fileCaps_size] at hi
    have h := (hall (2 * i) (by omega)).2.2.2.2.1
    rw [show 2 * i / 2 = i by omega] at h
    simp only [fileCaps, JsonRT.range_map_getElem! _ hi]
    exact h
  have hrep : ∀ k < (filePs spec sts c₁ pfs).size,
      SchedOf (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k) ∧
        HonestRep (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
          (filePs spec sts c₁ pfs)[k]! ∧
          (filePs spec sts c₁ pfs)[k]!.cap = (fileCaps spec sts c₁ pfs)[k / 2]! := by
    intro k hk
    rw [filePs_size] at hk
    have hf := hfc k hk
    rw [hsz] at hk
    obtain ⟨-, h2, h3, h4, -, -⟩ := hall k hk
    refine ⟨h2, h3 hf, ?_⟩
    simp only [fileCaps, JsonRT.range_map_getElem! _ (show k / 2 < sts.size by omega)]
    exact h4
  have hfit0 : RowsFit ((capRows (fileCaps spec sts c₁ pfs)).map nodeRow) := by
    refine capRows_fit (by rw [fileCaps_size]; exact h64t) fun a ha => ?_
    obtain ⟨i, hi, rfl⟩ := Array.mem_map.1 ha
    have hi' := Array.mem_range.1 hi
    have hk2 : 2 * i / 2 < sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall (2 * i) (by omega)
    have hok := htab (2 * i / 2) hk2
    exact rep_cap_le hsc (hon (hfc (2 * i) (by rw [hsz]; omega))) (repCtx_bounds hk2 hok.1 hok.2).1
  have hfit : ∀ k < (filePs spec sts c₁ pfs).size,
      ∀ rs ∈ repRows (repCtx spec sts c₁ (capRows (fileCaps spec sts c₁ pfs)) cs.tail.dropLast k)
        (filePs spec sts c₁ pfs)[k]! (Ts k), RowsFit rs := by
    intro k hk rs hrs
    rw [filePs_size, hsz] at hk
    have hk2 : k / 2 < sts.size := by omega
    obtain ⟨-, hsc, hon, -, -, -⟩ := hall k hk
    have hok := htab (k / 2) hk2
    obtain ⟨hd, hkl, hn⟩ := repCtx_bounds (r₀ := capRows (fileCaps spec sts c₁ pfs)) (cs := cs.tail.dropLast) hk2
      hok.1 hok.2
    exact repRows_fit (Ts k) hsc (hon (hfc k (by rw [hsz]; omega))) hd hkl hn rs hrs
  have hl : (honestFileMsgs spec sts c₁ pfs cs Ts last).getLast? = some last := by
    unfold honestFileMsgs honestMsgs
    exact List.getLast?_concat
  rw [decR_opensOf (rd' := fun _ => some { fl with ps := filePs spec sts c₁ pfs })
    (fl' := { fl with ps := filePs spec sts c₁ pfs }) hl hrd rfl rfl hps]
  exact ⟨_, readsBack (fun _ => some { fl with ps := filePs spec sts c₁ pfs }) x last cs s spec sts c₁
    (fileCaps spec sts c₁ pfs) (filePs spec sts c₁ pfs) Ts { fl with ps := filePs spec sts c₁ pfs } hx hcs
    (fileCaps_size ..) hcaps (by rw [filePs_size, hsz]) hrep hfit0 hfit rfl rfl hst hsh, rfl, rfl, rfl, rfl⟩

end FlockDeploy

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf filePs repLens honestFileMsgs)
open FlockVBridge.Family (plain cfOf lgStart)
open Flock.RecAlgebra (verifierRows)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockComplete.Replay.Lg (qOff)
open FlockComplete (repP asksS commitP Commit Trace)

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass) (acc : ℕ → Array F128)

/-- **Row D's `hlev` at a decision whose proofs are the honest session's files'**: where the decision's spec, tables
and first coin are a `sessionS` run's, and its proofs the ones the run's files decode to (`filePs`), at replies that are
the coins' bytes, for coins that number at least the reps' rounds and two, at `TableOk` tables with at most 2,048 extra
claims at the first coin's link points. `hlev_of_repP`'s `hrun` and `h0` are the run's rep `2 i + r`
(`sessionP_repRuns`), its `hcoin` the replies' coins at the rounds the verifier rows query (`rows_sz`,
`rounds_canon`), and its widths the table's registered scheme's (`opsWith_sizes`). -/
theorem hlev_honest_of {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {s : FlockDeploy.Ins} {spec : SessionSpec}
    {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : FlockComplete.Tape}
    (hx : FlockDeploy.setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), FlockComplete.JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs))
    (hdsp : D.d.spec = spec) (hdsts : D.d.sts = sts) (hdc : D.d.c₁ = c₁) (hdps : D.d.f.ps = filePs spec sts c₁ pfs)
    {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2)
    (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup)
    (hops : plain.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes Flock.V) = MerkleOps.ofScheme plain) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () := by
  intro v info kappa hv
  have hc : resp 0 = c₁.bytes := by
    cases cs with
    | nil => simp at hcs
    | cons c cs' =>
      simp only [List.head?_cons, Option.some.injEq] at hcs
      subst hcs
      simpa using hresp 0 (by simp)
  obtain ⟨-, hts⟩ := FlockDeploy.setup_tables hx
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  have hDsp : D.spec = spec := hsp.trans hdsp
  have hDsts : D.sts = sts := hsts.trans hdsts
  have hi : i < sts.size := by
    rw [← hts, ← hDsp]
    exact hti
  -- the session's rep `2 i + r`
  have htabs : (tabsOf sts).size = sts.size := by
    unfold tabsOf
    rw [Array.size_map]
  obtain ⟨schs, cms, hsc, hsz, hk⟩ := FlockComplete.JsonRT.sessionP_repRuns H resp spec (tabsOf sts)
    (FlockComplete.tracesOf sts w) ω (by rw [htabs, hts]) hS
  rw [htabs] at hsc hsz hk
  replace hsz : pfs.size = 2 * sts.size := hsz
  have hk2 : 2 * i + (r : ℕ) < 2 * sts.size := by omega
  have hdiv : (2 * i + (r : ℕ)) / 2 = i := by omega
  have hmod : (2 * i + (r : ℕ)) % 2 = r := by omega
  obtain ⟨e, pt, l, l', he, hls, hrun, hf⟩ := hk _ hk2
  rw [hdiv] at he hf
  rw [hdiv, hmod] at hrun
  obtain ⟨hs, hcm⟩ := hsc i hi
  rw [hc, FlockComplete.JsonRT.ptsOf_coin] at he
  -- table `i`'s files are the session's
  have hFst : (D.files cls i).st = (tabsOf sts)[i]! := by
    show (tabsOf D.sts)[i]! = _
    rw [hDsts]
  have hFsch : (D.files cls i).sch = schs[i]! := by
    show (Flock.fast100 (tabsOf D.sts)[i]!.m).getD default = _
    rw [hDsts, hs]
    rfl
  have hFex : (D.files cls i).extras = e := by
    show ((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[] = _
    rw [D.points_eq, hDsp, hdc, hDsts, he]
    rfl
  have ht := FlockDeploy.tabsOf_get! hi
  have hFpf : (D.files cls i).proof r = pt.1 := by
    show D.d.f.ps[2 * i + (r : ℕ)]! = _
    rw [hdps]
    have hsh : FlockDeploy.shapeAt (tabsOf sts) (pointsOf spec c₁) (2 * i + (r : ℕ)) =
        FlockComplete.shapeOf (tabsOf sts)[i]! schs[i]! e.size := by
      simp only [FlockDeploy.shapeAt, hdiv, hs, he, Option.getD_some]
      rfl
    have hs1 := hs
    have hcm1 := hcm
    have hrun1 := hrun
    rw [ht] at hs1 hcm1 hrun1
    have he1 : sts[i].extraClaims (pointsOf spec c₁) = .ok e := by
      rw [ht] at he
      exact he
    have hp := FlockComplete.JsonRT.repShapedOk h64 (htab i hi).1 hs1 resp _ hcm1 ((htab i hi).2 e he1) _ _ _ l pt l'
      hrun1
    have hdec : decodeProof (FlockDeploy.shapeAt (tabsOf sts) (pointsOf spec c₁) (2 * i + (r : ℕ)))
        pfs[2 * i + (r : ℕ)]! = .ok pt.1 := by
      rw [hsh, show pfs[2 * i + (r : ℕ)]! = (rec, pfs).2[2 * i + (r : ℕ)]! from rfl, hf, ht]
      exact FlockVBridge.Family.decodeProof_encodeProof _ _ hp
    exact FlockDeploy.filePs_get (by rw [hsz]; exact hk2) hdec
  -- `hrun` and `h0`
  rw [← hFst, ← hFsch, ← hFex] at hrun
  have hrun' : ((repP (asksS resp) (Hashes.ofSha512 (Claims.hOf H)) (D.files cls i).st (D.files cls i).sch
      (FlockComplete.tracesOf sts w)[i]! (FlockComplete.packed (FlockComplete.tracesOf sts w)[i]! (D.files cls i).st.m)
      cms[i]! (D.files cls i).extras (ω.salt i r) (ω.nonce i r)).run
        (Transcript.init #[] (spec.domain spec.tables[i]! r).toUTF8)).go l =
      (.ok ((D.files cls i).proof r, pt.2), l') := by
    rw [hFpf]
    exact hrun
  rw [← hFst, ← hFsch] at hcm
  -- the widths
  have hms : (D.files cls i).st.merkle ∈ MerkleScheme.all := by
    rw [hFst, ht]
    exact (htab i hi).1.scheme
  obtain ⟨hn, hlf⟩ := FlockComplete.JsonRT.opsWith_sizes h64 hms
  -- `hcoin`, at the rounds the verifier rows query
  obtain ⟨h22, -⟩ := FlockVBridge.Family.fast100_range hG.fast
  have hdim := (FlockSoundness.Refine.fast100_dims _ _ hG.fast).1
  have hlv : ∀ j < (D.files cls i).sch.levels.size,
      FlockSoundness.Refine.LevelSeam ((D.files cls i).sch.levels.getD j default) :=
    fun j hj => FlockSoundness.Refine.fast100_level _ _ hG.fast _
      (by unfold Array.getD; rw [dite_eq_left hj]; exact Array.getElem_mem hj)
  have hqsz : ∀ j < (D.files cls i).sch.levels.size,
      ((D.rep cls acc i r).rounds[lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j]!).size =
        ((D.files cls i).sch.levels.getD j default).q :=
    FlockVBridge.Family.rows_sz (D.noEmpty cls acc hti r) (by show 14 ≤ (D.files cls i).st.m; omega) hdim hlv hv
  have eR : (D.rep cls acc i r).rounds = canonRounds spec sts cs.tail.dropLast (2 * i + r) := by
    have hR := D.rounds_canon cls i hnd (by rw [hDsts]; exact hi) hti r
    rw [hDsp, hDsts] at hR
    exact hR
  have h6 : ∀ j < 2 * sts.size, 6 ≤ sts[j / 2]!.c.kLog := fun j hj => by
    rw [getElem!_pos sts (j / 2) (by omega)]
    exact (htab _ (by omega)).1.k
  obtain ⟨-, hall⟩ := FlockDeploy.sessionS_reps h64 hc hts.symm htab hS [] cs.tail.dropLast
  have hsum : ((List.range (2 * i + r)).map (FlockComplete.JsonRT.sendsP resp spec (tabsOf sts))).sum =
      ((repLens spec sts).take (2 * i + r)).sum := by
    rw [FlockDeploy.take_eq_map_getD _ _ (by rw [FlockDeploy.repLens_length]; omega)]
    congr 1
    refine List.map_congr_left fun j hj => ?_
    have hj' : j < 2 * sts.size := by
      have := List.mem_range.1 hj
      omega
    exact FlockDeploy.sendsP_eq hc hj' (h6 j hj') (hall j hj').2.1
  have hS1 := FlockDeploy.sum_take_succ_getD (repLens spec sts) (2 * i + r)
  have hS2 := FlockDeploy.take_sum_le (repLens spec sts) (2 * i + r + 1)
  have hcoin : ∀ j < (D.files cls i).sch.levels.size, ∀ t < ((D.files cls i).sch.levels.getD j default).q,
      FlockComplete.Replay.Lg.cfOf resp l.size
          (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t =
        cfOf (D.rep cls acc i r).rounds (lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j) t := by
    intro j hj t htq
    have hJsz := hqsz j hj
    generalize lgStart (D.sh cls i) (D.files cls i).extras.size + qOff (D.files cls i).sch j = J at hJsz ⊢
    generalize ((D.files cls i).sch.levels.getD j default).q = q at hJsz htq
    rw [eR] at hJsz
    unfold cfOf
    rw [eR]
    have hCsz : (canonRounds spec sts cs.tail.dropLast (2 * i + r)).size = (repLens spec sts).getD (2 * i + r) 0 := by
      simp [canonRounds]
    have hJlt : J < (repLens spec sts).getD (2 * i + r) 0 := by
      by_contra hn'
      rw [getElem!_neg _ J (by rw [hCsz]; exact hn')] at hJsz
      have : (default : Array F128).size = 0 := rfl
      omega
    have hCJ : (canonRounds spec sts cs.tail.dropLast (2 * i + r))[J]! =
        (cs.tail.dropLast.getD (((repLens spec sts).take (2 * i + r)).sum + J) default).vals
          ((RecDeploy.repSchedAt spec sts[(2 * i + r) / 2]!).getD J default).1 := by
      rw [getElem!_pos _ J (by rw [hCsz]; exact hJlt)]
      simp [canonRounds]
    rw [hCJ] at hJsz ⊢
    simp only [FlockDeploy.Coin.vals, Array.size_map, Array.size_range] at hJsz
    have hval : ∀ (c : Coin) (n : ℕ), t < n → (c.vals n).getD t 0 = c.at t := fun c n hn => by
      simp [FlockDeploy.Coin.vals, Array.getD_eq_getD_getElem?, hn]
    have htn : t < ((RecDeploy.repSchedAt spec sts[(2 * i + r) / 2]!).getD J default).1 := by omega
    rw [hval _ _ htn, FlockDeploy.getD_tail_dropLast (default : Coin) cs (((repLens spec sts).take (2 * i + r)).sum + J)
      (by omega)]
    unfold FlockComplete.Replay.Lg.cfOf
    rw [hls, hsum, show 1 + ((repLens spec sts).take (2 * i + r)).sum + J =
      ((repLens spec sts).take (2 * i + r)).sum + J + 1 by omega,
      hresp (((repLens spec sts).take (2 * i + r)).sum + J + 1) (by omega), FlockComplete.JsonRT.coinOf_bytes]
  exact D.hlev_of_repP cls acc hG hti r (Claims.hOf H) hops resp _ _ _ _ _ hn hlf hcm
    (Transcript.init #[] (spec.domain spec.tables[i]! r).toUTF8) rfl l hrun' hcoin v info kappa hv

/-- **Row D's `hlev` at the honest recursive prover**: at a decision on the honest messages (`honestFileMsgs`), at a
statement that sets up, at a `sessionS` run whose replies are the coins' bytes, for coins that number at least the
reps' rounds and two, at `TableOk` tables with at most 2,048 extra claims at the first coin's link points, where the
last message reads as `filePs`' openings at the statement (`honest_decR_files`, `hlev_honest_of`). -/
theorem hlev_honest {H : Claims.Msg → Claims.Msg} (h64 : Claims.Out64 H) {last : Claims.Msg} {s : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit}
    {ω : FlockComplete.Tape} {Ts : ℕ → F128} {fl : FinalP}
    (hx : FlockDeploy.setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), FlockComplete.JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS H resp spec sts w ω = (.ok (rec, pfs), bs)) (hrd : rd last = some fl)
    (hps : fl.ps = (filePs spec sts c₁ pfs).map RecDeploy.DeployedR.opensOf) (hst : fl.stmt = x)
    (D : DecAt rd x (honestFileMsgs spec sts c₁ pfs cs Ts last) cs) {i : ℕ} (hG : D.RowG cls acc i)
    (hti : i < D.spec.tables.size) (r : Fin 2) (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup)
    (hops : plain.opsWith (Hashes.ofSha512 (Claims.hOf H) : Hashes Flock.V) = MerkleOps.ofScheme plain) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () := by
  obtain ⟨d, hdec, hdsp, hdsts, hdc, hdps⟩ :=
    FlockDeploy.honest_decR_files h64 Ts hx hcs hresp hlen htab hS hrd hps hst
  have hDd : D.d = d := Option.some.inj (D.dec.symm.trans hdec)
  exact D.hlev_honest_of cls acc h64 hx hcs hresp hlen htab hS (by rw [hDd]; exact hdsp) (by rw [hDd]; exact hdsts)
    (by rw [hDd]; exact hdc) (by rw [hDd]; exact hdps) hG hti r hnd hops

theorem sha512_out64 : Claims.Out64 FlockDeploy.sha512 := fun m => by
  rw [FlockDeploy.sha512, Claims.toMsg, Array.length_toList, ByteArray.size_data]
  exact Sha512.hash_size _

theorem sha512_ops : plain.opsWith (Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512) : Hashes Flock.V) =
    MerkleOps.ofScheme plain := by
  rw [FlockDeploy.hOf_sha512]
  exact plain_ops_std

/-- **Row D's `hlev` at the honest recursive prover, at the deployed hashes**: `hlev_honest_of` at `FlockDeploy.sha512`,
whose Merkle closures are the plain scheme's (`plain_ops_std`), so no `hops` remains. -/
theorem hlev_honest_of_deployed {s : FlockDeploy.Ins} {spec : SessionSpec}
    {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit} {ω : FlockComplete.Tape}
    (hx : FlockDeploy.setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), FlockComplete.JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS FlockDeploy.sha512 resp spec sts w ω = (.ok (rec, pfs), bs))
    (hdsp : D.d.spec = spec) (hdsts : D.d.sts = sts) (hdc : D.d.c₁ = c₁) (hdps : D.d.f.ps = filePs spec sts c₁ pfs)
    {i : ℕ} (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2)
    (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () :=
  D.hlev_honest_of cls acc sha512_out64 hx hcs hresp hlen htab hS hdsp hdsts hdc hdps hG hti r hnd sha512_ops

/-- **Row D's `hlev` at the honest recursive prover's messages, at the deployed hashes**: `hlev_honest` at
`FlockDeploy.sha512`, with no `hops`. -/
theorem hlev_honest_deployed {last : Claims.Msg} {s : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array _root_.Flock.Stmt} {c₁ : Coin} {resp : ℕ → Claims.Msg} {w : Claims.Flock.Wit}
    {ω : FlockComplete.Tape} {Ts : ℕ → F128} {fl : FinalP}
    (hx : FlockDeploy.setupOf false x = some (s, spec, sts)) (hcs : cs.head? = some c₁)
    (hresp : ∀ i < cs.length, resp i = (cs.getD i default).bytes) (hlen : (repLens spec sts).sum + 2 ≤ cs.length)
    (htab : ∀ (i : ℕ) (hi : i < sts.size), FlockComplete.JsonRT.TableOk sts[i] ∧
      ∀ e, sts[i].extraClaims (pointsOf spec c₁) = .ok e → e.size ≤ 2048)
    {rec : String} {pfs : Array ByteArray} {bs : Array Claims.Msg}
    (hS : FlockComplete.sessionS FlockDeploy.sha512 resp spec sts w ω = (.ok (rec, pfs), bs))
    (hrd : rd last = some fl) (hps : fl.ps = (filePs spec sts c₁ pfs).map RecDeploy.DeployedR.opensOf)
    (hst : fl.stmt = x) (D : DecAt rd x (honestFileMsgs spec sts c₁ pfs cs Ts last) cs) {i : ℕ}
    (hG : D.RowG cls acc i) (hti : i < D.spec.tables.size) (r : Fin 2)
    (hnd : (D.spec.repStreams.toList.map (·.2.2)).Nodup) :
    ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ ℓ < (D.files cls i).sch.levels.size, ((D.files cls i).sch.levels[ℓ]!).q ≤ (info.lg.conss[ℓ]!).pos.size ∧
        levelCheck (D.files cls i).sch ((D.files cls i).proof r).cap ((D.files cls i).proof r).ligerito ℓ
          (info.lg.conss[ℓ]!).pos = .ok () :=
  hlev_honest cls acc sha512_out64 hx hcs hresp hlen htab hS hrd hps hst D hG hti r hnd sha512_ops

end DecAt

end RecDeploy.Rows
