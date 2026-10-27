import Proofs.Flock.Claims.Recursion.SameJAt
import Proofs.Flock.Claims.Recursion.VerdictFramed
import Proofs.Flock.Claims.Recursion.VerdictFramedJson
import Proofs.Flock.Claims.Recursion.VerdictPublics
import Proofs.Flock.Claims.Recursion.VerdictRowG
import Proofs.Flock.Claims.Recursion.VerdictParts
import Proofs.Flock.Claims.Recursion.InnerForm
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Claims.Complete.Json.Stmt
import Proofs.Flock.Recursive.HoloRepCoins
import Proofs.Flock.C3.Compose.HoloStreamSizes
import Proofs.Flock.VBridge.Family.Lig
import Proofs.Flock.VBridge.Family.Encode

/-!
# The record the verdict reads decodes, at a decision of `nextR`

At a dense inner statement where the last step reads and the base checks hold, the record value the verdict reads
(`jAt`) decodes at the setup SHA-512 gives (`decode_at`, RECINT's `TargetDecode` with `At` unfolded), with no JSON
premise: the verifier keeps the public header and the draw normalized, so both are well-formed (`unitDraw_of`,
`draw_wf`), and the statement's coin tree is its coins' (`setupOne_stmt`), none at `.os` or `.seed` (`tree?_none`). `jAt` is the reader's record (`recJ`) with its digests and coins
substituted, so it is alike the honest record whose rounds hold the fill's coins (`same_coinSubJ`, with the fill finding
a coin wherever `liveOf` reads, `fill_of_live`), and an honest record of a one-table statement decodes
(`decodeJ_same`). What `decodeJ_same` asks:

* **The statement** (`dense_stmt`): one table, no coin tree, a non-empty Σ, a printable draw that decodes to the
  statement's instances, two link points, at most 1024 regions, and a `fast100` schedule.
* **The link's `root_B`** is round 0's level-0 nodes joined, which the first proof's cap is, so it is non-empty (the
  `shaped` base check).
* **The streams** (`repSchedAt_bounds`): rep `k`'s rounds are its schedule's, which has a round, each of between 1
  and 65536 coins.
-/

namespace RecDeploy.DecodeAt

open Flock
open Lean (Json)
open Claims (Msg)
open Claims.Json (Same Good)
open FlockDeploy (Coin FinalP ReadOkR holdsR liveOfJ coinSubJ pointsOf tabsOf shapeAt cutLens level0Lens repLens
  recStreams)
open FlockComplete.JsonRT (DrawOk)

/-! ## Bytes -/

theorem foldl_append_size : ∀ (l : List ByteArray) (a : ByteArray),
    (l.foldl (· ++ ·) a).size = a.size + (l.map ByteArray.size).sum
  | [], a => by simp
  | b :: l, a => by
    rw [List.foldl_cons, foldl_append_size l (a ++ b), ByteArray.size_append, List.map_cons, List.sum_cons]
    omega

theorem le_sum_of_mem' : ∀ {l : List ℕ} {a : ℕ}, a ∈ l → a ≤ l.sum
  | _ :: _, _, .head _ => by rw [List.sum_cons]; omega
  | _ :: _, _, .tail _ h => by rw [List.sum_cons]; have := le_sum_of_mem' h; omega

/-- Bytes joined from a list holding a non-empty one are non-empty. -/
theorem foldl_append_pos {l : List ByteArray} {b : ByteArray} (hb : b ∈ l) (h : 0 < b.size) :
    0 < (l.foldl (· ++ ·) ByteArray.empty).size := by
  rw [foldl_append_size]
  have := le_sum_of_mem' (List.mem_map.2 ⟨b, hb, rfl⟩ : b.size ∈ l.map ByteArray.size)
  omega

/-! ## The statement -/

/-- **The inner statement is drawn**: the plain verifier sets an inner statement the reader accepts up at a draw
(`checkForm_ok`: a draw of every one of the files' units). -/
theorem inner_drawn {x : Msg} {r : XR} (hr : readX x = some r) (ha : RecScope.accepts (readOf r) = true)
    {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt}
    (hset : FlockDeploy.setupOf false r.w.inner = some (s, spec, sts)) : s.read.draw.isSome = true := by
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨-, ⟨u, hsh⟩⟩, hheld⟩ := ha
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  have hheld : (filesOf r).isSome = true := hheld
  obtain ⟨p, hp⟩ : ∃ p, r.partition = some p := by
    cases hq : r.partition with
    | none => simp [filesOf, hq] at hheld
    | some p => exact ⟨p, rfl⟩
  have hfs : filesOf r = some (.files ⟨r.innerRaw.circuit, r.innerRaw.publicFile, p, "", none,
      some (innerFormOf r.innerRaw)⟩) := by rw [filesOf, hp]; rfl
  rw [hfs] at hsh
  obtain ⟨spec', st', hstm, -, -, -, -, hform⟩ := checkShape_files hsh
  obtain ⟨hs1, hs2, -, b, d, D, hb, hjd, -, -⟩ := checkForm_ok hform
  have hs1 : r.innerRaw.statement = Tags.circuitPlainLeaves.name := hs1
  have hs2 : r.innerRaw.osCoins = true := hs2
  have hb : r.innerRaw.draw = some b := hb
  obtain ⟨hdec, -, -⟩ := readX_inner hr
  obtain ⟨hins, -⟩ := FlockDeploy.setupOf_some hset
  have hw : r.innerRaw.zk = false ∧ r.innerRaw.ins = .ok s := by
    unfold FlockDeploy.insOf at hins
    obtain ⟨w, hw0, hw⟩ := Option.bind_eq_some_iff.1 hins
    rw [hdec] at hw0
    cases hw0
    split_ifs at hw with hz
    · cases hw
    change r.innerRaw.ins.toOption = some s at hw
    exact ⟨not_not.1 hz, toOption_eq_some hw⟩
  obtain ⟨-, -, -, -, -, -, -, -, ⟨dr, hm, hd9⟩, -⟩ := ins_plainLeaves hw.2 hw.1 hs1 hs2
  rw [hb] at hm
  simp only [Option.mapM_some, hjd, Functor.map, Except.map, Except.ok.injEq] at hm
  rw [hd9, ← hm]
  rfl

/-- **What the record needs of a dense inner statement**: the plain verifier sets it up as one table, with no coin
tree, a non-empty Σ, a printable draw that decodes to its instances, two link points, at most 1024 regions and a
`fast100` schedule. -/
theorem dense_stmt {xo : Msg} {sto : Claims.Recursion.Stmt}
    (hst : parse xo = some sto) (hf : Claims.Recursion.Form.Dense sto.form) {s : FlockDeploy.Ins}
    {spec : SessionSpec} {sts : Array Stmt} (hset : FlockDeploy.setupOf false (xIOf xo) = some (s, spec, sts)) :
    ∃ st t, sts = #[st] ∧ spec.tables = #[t] ∧ spec.coinTree = none ∧ 0 < spec.sigma.size ∧ DrawOk spec ∧
      Good (spec.unitDraw.getD Json.null) ∧ spec.points = 2 ∧ st.regions.size ≤ 1024 ∧ (fast100 st.m).isSome := by
  obtain ⟨s', spec', st, sch, hset', htags, hpi, hsch⟩ := Rows.dense_inner hst hf
  rw [hset] at hset'
  obtain ⟨rfl, rfl, rfl⟩ : s = s' ∧ spec = spec' ∧ sts = #[st] := by simpa using hset'
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hx : xIOf xo = r.w.inner := by unfold xIOf; rw [hst]; rfl
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hread := FlockDeploy.Ins.setup_read hsu
  unfold Readable.Read.setup at hread
  rw [hpi] at hread
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okAny hread with ⟨-, st1, h1, hs1, hsts1⟩ | ⟨hc1, hsz, -⟩
  swap
  · simp only [List.size_toArray, List.length_cons, List.length_nil, zero_add] at hsz
    omega
  obtain rfl : st = st1 := by simpa using hsts1
  subst hs1
  have hhid : s.read.tags.hiddenOutputs = true := by rw [htags] <;> rfl
  have hreg : st.regions.size ≤ 1024 := by
    have h1' := h1
    rw [FlockSoundness.Discharge.HiddenExec.setupOne_hidden hhid] at h1'
    exact FlockVBridge.Family.spec_regions_le (FlockSoundness.Discharge.Hidden.setupHidden_spec' h1')
  obtain ⟨⟨c', pop, hl, hd⟩, hinst, hsig, -, hct⟩ := FlockComplete.JsonRT.setupOne_stmt h1
  obtain ⟨w, hwz, hw⟩ := FlockComplete.JsonRT.insOf_false hins
  obtain ⟨hzk, hcoins, hdraw⟩ := FlockComplete.JsonRT.Raw.ins_read hw
  have hz : s.read.zk = false := hzk.trans hwz
  have hset2 := hset
  rw [hx] at hset2
  have hds := inner_drawn hr ha hset2
  obtain ⟨d, hdd⟩ := Option.isSome_iff_exists.1 hds
  have hu := FlockComplete.JsonRT.unitDraw_of hl (hd d hdd)
  have hU : st.spec.unitDraw = some d := by simp [Stmt.spec, hu, Except.toOption]
  obtain ⟨D, hD, hn⟩ := hinst d hu
  have hdl := FlockComplete.JsonRT.Ins.setup_drawLaw hsu hds
  rw [hdd, hz] at hdl
  simp only [ProvedScope.drawLaw, hD] at hdl
  have hlw : ProvedScope.lawWhy false s.read.tags false D = none := by
    cases e : ProvedScope.lawWhy false s.read.tags false D with
    | none => rfl
    | some _ => rw [e] at hdl; cases hdl
  have hg : Good d := by
    rcases FlockComplete.JsonRT.law_of_lawWhy hlw with ⟨K, strata, hK⟩ | ⟨k, hk⟩
    · exact FlockComplete.JsonRT.good_stratified (FlockComplete.JsonRT.draw_wf hdraw hdd) hD hK
    · exact FlockComplete.JsonRT.good_subset (FlockComplete.JsonRT.draw_wf hdraw hdd) hD hk
  have hnn : d ≠ Json.null := by
    intro hn'
    rw [hn'] at hD
    cases hD
  refine ⟨st, st.tags.table, rfl, rfl, ?_, hsig, ?_, ?_, rfl, hreg, by simp [hsch]⟩
  · change st.coinTree = none
    rw [hct]
    refine FlockComplete.JsonRT.tree?_none ?_
    simp only [Readable.Read.coins, hz, Bool.false_eq_true, ↓reduceIte]
    exact hcoins hwz
  · unfold DrawOk
    rw [hU]
    dsimp only
    exact ⟨hnn, hg, D, hD, hn⟩
  · rw [hU]
    exact hg

/-! ## The schedule -/

/-- **A rep's schedule at a `fast100` table**: it has a round, and each round counts between 1 and 65536 coins, at
a claim count of at most 65536. -/
theorem repSchedAt_bounds {spec : SessionSpec} {st : Stmt} (hf : (fast100 st.m).isSome)
    (hc : 2 + spec.points * st.regions.size ≤ 65536) :
    0 < (RecDeploy.repSchedAt spec st).length ∧
      ∀ x ∈ RecDeploy.repSchedAt spec st, 1 ≤ x.1 ∧ x.1 ≤ 65536 := by
  obtain ⟨sch, hsch⟩ := Option.isSome_iff_exists.1 hf
  obtain ⟨-, h35, -⟩ := fast100_sizes hsch
  have key : ∀ m < 36, (fast100 m).isSome → (RecVStar.repRounds m 6 1).isSome := by decide +kernel
  have key2 : ∀ m < 36, ((RecVStar.repRounds m 6 1).all fun o => o.all fun x => decide (x.1 ≤ 65536)) = true := by
    decide +kernel
  obtain ⟨out, hout⟩ := Option.isSome_iff_exists.1 (key _ (by omega) hf)
  obtain ⟨T, hT, hall⟩ := FlockC3.Compose.repRounds_tail hout
  have h2 := hall st.c.kLog (2 + spec.points * st.regions.size)
  have hTb : ∀ x ∈ T, x.1 ≤ 65536 := by
    have k2 := key2 st.m (by omega)
    rw [hout, hT] at k2
    simp only [Option.all_some, Array.all_append, Bool.and_eq_true, Array.all_eq_true_iff_forall_mem,
      decide_eq_true_eq] at k2
    exact k2.2
  have hHb : ∀ x ∈ FlockC3.Compose.repHead st.m st.c.kLog (2 + spec.points * st.regions.size), x.1 ≤ 65536 := by
    intro x hx
    simp only [FlockC3.Compose.repHead, Array.mem_append, Array.mem_replicate, List.mem_toArray, List.mem_cons,
      List.not_mem_nil, or_false] at hx
    rcases hx with ((((((h | h | h) | ⟨-, h⟩) | h) | ⟨-, h⟩) | h) | ⟨-, h⟩) | h
    all_goals (subst h; first | omega | (dsimp only; omega))
  have hpos := Flock.Guarantees.HoloRepCoins.repRounds_coins (by omega) h2
  have e : RecDeploy.repSchedAt spec st =
      (FlockC3.Compose.repHead st.m st.c.kLog (2 + spec.points * st.regions.size) ++ T).toList := by
    unfold RecDeploy.repSchedAt
    rw [h2] <;> rfl
  rw [e]
  refine ⟨?_, fun x hx => ?_⟩
  · rw [Array.length_toList, Array.size_append]
    simp only [FlockC3.Compose.repHead, Array.size_append, Array.size_replicate, List.size_toArray,
      List.length_cons, List.length_nil]
    omega
  · rw [Array.mem_toList_iff] at hx
    refine ⟨hpos x hx, ?_⟩
    rcases Array.mem_append.1 hx with h | h
    · exact hHb x h
    · exact hTb x h

/-! ## The decode -/

/-- **The record the verdict reads decodes** at a decision at a dense inner statement where the last step reads and
the base checks hold, at the setup SHA-512 gives. -/
theorem decode_jAt {rd : Msg → Option FinalP} {x : Msg}
    {ms : List Msg} {cs : List Coin} {xo : Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (hr : ReadOkR rd x ms cs)
    (hb : ∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) {d : FlockDeploy.DecR}
    (hd : FlockDeploy.decR rd x ms cs = some d) {spec : SessionSpec} {sts : Array Stmt}
    (hs : FlockDeploy.setupWith (Claims.hOf FlockDeploy.sha512) d.s.read = .ok (spec, sts)) :
    ∃ sess, Record.decodeJWith (Flock.Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512) : Flock.Hashes Flock.V) spec
      (FlockDeploy.jAt FlockDeploy.sha512 spec sts d.c₁ d.mids d.f) = .ok sess := by
  subst hx
  have hact : FlockDeploy.actR rd (xIOf xo) ms cs = .decide d.s d.spec d.sts d.link d.c₁ d.mids d.f := by
    unfold ReadOkR at hr
    revert hr
    cases h : FlockDeploy.actR rd (xIOf xo) ms cs with
    | rej => intro hr; exact (hr : False).elim
    | coin c => intro hr; exact (hr : False).elim
    | decide s' sp' ts' link' c' mids' f' =>
      intro _
      rw [← Option.some.inj ((FlockDeploy.actR_decide h).symm.trans hd)]
  have hlive : ∃ lv, liveOfJ d.spec (coinSubJ d.spec d.c₁ d.mids d.f.j) = .ok lv := by
    unfold ReadOkR at hr
    rw [hact] at hr
    exact hr
  have hsv : FlockDeploy.setupOf false (xIOf xo) = some (d.s, d.spec, d.sts) := by
    obtain ⟨v, c₁, r₀, rss, fl, link, pf, h1, -, -, -, -, -, rfl⟩ := FlockDeploy.decR_some hd
    exact h1
  have hsu : d.s.setup = .ok (d.spec, d.sts) := (FlockDeploy.setupOf_some hsv).2
  have hs0 := hs
  rw [FlockDeploy.hOf_sha512, FlockDeploy.setupWith_sha512, FlockDeploy.Ins.setup_read hsu] at hs
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs)
  obtain ⟨st, t, hsts, ht, hct, hsig, hdraw, hdg, hpts2, hreg, hfast⟩ := dense_stmt hst hf hsv
  have hnames : FlockDeploy.NamesDistinct d.spec := by
    have h := hb .names (by simp [VerdictP.bases])
    unfold holdsR at h
    rw [hact] at h
    exact h d.spec d.sts hs0
  have hnd : (d.spec.repStreams.toList.map (·.2.2)).Nodup := by
    refine List.nodup_iff_injective_get.2 fun i k e => Fin.ext ?_
    simp only [List.get_eq_getElem, List.getElem_map, Array.getElem_toList] at e
    exact hnames i k (by simpa using i.2) (by simpa using k.2) e
  obtain ⟨r₀, rss, ms', h6, -, hj, hfr⟩ := FlockDeploy.decR_framesR hd
  obtain ⟨r₀', h6', hlink, -⟩ := FlockDeploy.decR_rootB hd
  obtain rfl : r₀ = r₀' := Option.some.inj (h6.symm.trans h6')
  obtain ⟨-, -, -, -, hsz, -⟩ := FlockDeploy.decR_fits hd
  have hk0 : 0 < d.f.ps.size := by rw [hsz, hsts]; simp
  have hsh0 : FlockVBridge.Family.ProofShaped (shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) 0) d.f.ps[0] := by
    have h := hb .shaped (by simp [VerdictP.bases])
    unfold holdsR at h
    rw [hact] at h
    exact h d.spec d.sts hs0 0 hk0
  obtain ⟨hhl, -, -, ⟨hcs, hce⟩, -⟩ := hsh0
  obtain ⟨T0, hT0, -⟩ := hfr 0 (by rw [hsts]; simp)
  have hcap : d.f.ps[0].cap.toList = r₀.take (RecDeploy.level0Nodes st) := by
    rw [← getElem!_pos d.f.ps 0 hk0, hT0.cap]
    show (cutLens (level0Lens d.sts) r₀).getD (0 / 2) [] = _
    rw [hsts] <;> rfl
  have hrb : 0 < ((r₀.take (RecDeploy.level0Nodes st)).foldl (· ++ ·) ByteArray.empty).size := by
    rw [← hcap]
    have hc0 : 0 < d.f.ps[0].cap.size := by rw [hcs]; exact Nat.two_pow_pos _
    refine foldl_append_pos (b := d.f.ps[0].cap[0]) (Array.mem_toList_iff.2 (Array.getElem_mem hc0)) ?_
    rw [hce _ (Array.getElem_mem hc0)]
    exact hhl
  have hlk : d.link.1.toArray = #[(r₀.take (RecDeploy.level0Nodes st)).foldl (· ++ ·) ByteArray.empty] := by
    rw [hlink, hsts] <;> rfl
  have hbounds := repSchedAt_bounds (spec := d.spec) hfast (by rw [hpts2]; omega)
  have hdiv : ∀ k < 2 * d.sts.size, d.sts[k / 2]! = st := by
    intro k hk
    rw [hsts] at hk ⊢
    have : k / 2 = 0 := by simp at hk; omega
    rw [this] <;> rfl
  have hsize : ∀ k (hk : k < (recStreams d.spec d.sts ms' cs.tail.dropLast).size),
      (recStreams d.spec d.sts ms' cs.tail.dropLast)[k].size = (RecDeploy.repSchedAt d.spec st).length := by
    intro k hk
    have hk2 : k < 2 * d.sts.size := by rwa [FlockDeploy.recStreams_size] at hk
    obtain ⟨T, hT, hfk⟩ := hfr k hk2
    rw [FlockDeploy.recStreams_size_k hk, hfk, FlockDeploy.framesOf, List.length_map, hT.rounds, ← hT.sched]
    show (RecDeploy.repSchedAt d.spec d.sts[k / 2]!).length = _
    rw [hdiv k hk2]
  have hn : ∀ k (hk : k < (recStreams d.spec d.sts ms' cs.tail.dropLast).size) j
      (hj : j < (recStreams d.spec d.sts ms' cs.tail.dropLast)[k].size),
      1 ≤ (recStreams d.spec d.sts ms' cs.tail.dropLast)[k][j].n ∧
        (recStreams d.spec d.sts ms' cs.tail.dropLast)[k][j].n ≤ 65536 := by
    intro k hk j hj
    have hk2 : k < 2 * d.sts.size := by rwa [FlockDeploy.recStreams_size] at hk
    have hjL : j < (RecDeploy.repSchedAt d.spec st).length := by rw [← hsize k hk]; exact hj
    rw [FlockDeploy.recStreams_n hk hj, hdiv k hk2, List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hjL,
      Option.getD_some]
    exact hbounds.2 _ (List.getElem_mem hjL)
  have hss : ∀ off, (fillStreams d.spec d.mids off (recStreams d.spec d.sts ms' cs.tail.dropLast)).size =
      d.spec.repStreams.size := by
    intro off
    simp only [fillStreams, Array.size_mapIdx]
    rw [FlockDeploy.recStreams_size, FlockDeploy.repStreams_size, ht, hsts] <;> rfl
  have hrs : ∀ off, ∀ rs ∈ fillStreams d.spec d.mids off (recStreams d.spec d.sts ms' cs.tail.dropLast),
      0 < rs.size ∧ ∀ r ∈ rs, 1 ≤ r.n ∧ r.n ≤ 65536 ∧ r.coins.size = r.n := by
    intro off rs hm
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hm
    have hi' : i < (recStreams d.spec d.sts ms' cs.tail.dropLast).size := by simpa [fillStreams] using hi
    simp only [fillStreams, Array.getElem_mapIdx, Array.size_mapIdx]
    refine ⟨by rw [hsize i hi']; exact hbounds.1, fun r hr => ?_⟩
    obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hr
    have hk' : k < (recStreams d.spec d.sts ms' cs.tail.dropLast)[i].size := by simpa using hk
    simp only [Array.getElem_mapIdx, FlockDeploy.Coin.vals, Array.size_map, Array.size_range]
    first | exact ⟨(hn i hi' k hk').1, (hn i hi' k hk').2, trivial⟩ | exact ⟨(hn i hi' k hk').1, (hn i hi' k hk').2, rfl⟩
  have hpts : (pointsOf d.spec d.c₁).size = d.spec.points := by
    unfold pointsOf
    rw [Array.size_map, Array.size_range]
  have hpm : ∀ p ∈ pointsOf d.spec d.c₁, p.size = d.spec.mPts := by
    intro p hp
    unfold pointsOf at hp
    obtain ⟨q, -, rfl⟩ := Array.mem_map.1 hp
    rw [Array.size_map, Array.size_range]
  have h64 : ∀ b, (Claims.hOf FlockDeploy.sha512 b).size = 64 := fun b => by
    rw [FlockDeploy.hOf_sha512, Sha512.hash_size]
  obtain ⟨lv, hlv⟩ := hlive
  rw [hj, FlockDeploy.recJ, hlk] at hlv
  unfold FlockDeploy.jAt
  rw [hj, FlockDeploy.recJ, hlk]
  refine ⟨_, FlockComplete.JsonRT.decodeJ_same h64 ht hct hdraw hsig hrb hpts hpm (hss _) (hrs _) _
    (same_coinSubJ ht _ _ _ _ _ _ hdg ?_)⟩
  intro i hi hS k hk
  rw [streamOffset_shaSubJ]
  exact fill_of_live hnd hlv i hi hS k hk

/-- **`TargetDecode`** (RECINT's, `At` unfolded): at a dense inner statement where the last step reads and the base
checks hold, a decision's record value the verdict reads decodes at the setup SHA-512 gives. -/
theorem decode_at (rd : Msg → Option FinalP)
    (P : Msg → List Msg → List Coin → Prop) :
    ∀ (x : Msg) (ms : List Msg) (cs : List Coin),
      ((∃ (xo : Msg) (sto : Claims.Recursion.Stmt), parse xo = some sto ∧ Claims.Recursion.Form.Dense sto.form ∧
        x = xIOf xo) ∧ ReadOkR rd x ms cs ∧ (∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) ∧
        P x ms cs) →
      ∀ d : FlockDeploy.DecR, FlockDeploy.decR rd x ms cs = some d → ∀ (spec : SessionSpec) (sts : Array Stmt),
        FlockDeploy.setupWith (Claims.hOf FlockDeploy.sha512) d.s.read = .ok (spec, sts) →
        ∃ sess, Record.decodeJWith (Flock.Hashes.ofSha512 (Claims.hOf FlockDeploy.sha512) : Flock.Hashes Flock.V)
          spec (FlockDeploy.jAt FlockDeploy.sha512 spec sts d.c₁ d.mids d.f) = .ok sess :=
  fun _ _ _ ⟨⟨_, _, hst, hf, hx⟩, hr, hb, _⟩ _ hd _ _ hs => decode_jAt hst hf hx hr hb hd hs

end RecDeploy.DecodeAt
