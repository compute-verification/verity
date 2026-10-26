import Proofs.Flock.Claims.Recursion.OutSlotRead
import Proofs.Flock.Claims.Recursion.OpenDirsDeployed
import Proofs.Flock.Claims.Recursion.TopsOpens
import Proofs.Flock.Claims.Recursion.InnerExtras
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Claims.Recursion.VerdictRowDTie
import Proofs.Flock.C3.Opening.SessionCoins
import Proofs.Flock.C3.Reps.AtList

/-!
# The level's `out` slot and RecCap's `top` slot carry one commit string (`OutCommitTie` at `layR`)

`OutCommitTie` asks, at a level part's piece `q` (instance `N = rep · q_l + query`), that the level part's `out` slot
at `N` and RecCap's `top` slot at `capIdx` carry one commit string. Both are rows of the coin server's
`rec-top-L<l>`:

* `outRow_bytes`: the level's check 5 runs `checkOpenTops` (`checkLink_open`), whose `out` reads `rec-top-L<l>` at
  instance `N`'s node `first + pos >>> HH` (`checkLevelTopReads_ok`, `outRead`), the labels being `0, …, n − 1`
  (`CopySite.idx_staged`), so its slot at `N` is `topRow lk l` there (`checkTopRows_rows`, as `capRow_bytes`).
* `posRaw_of`: that position is the one `Rows.posAt` reads off the coins (`posTieR_of`'s argument, at the raw
  position rather than its `dirs` row), and rep 1's rows start after rep 0's `2^cap` nodes at a later level
  (`laterCap_size`, `top0_eq`), so the node is `capIdx`.
* `root_link`, `tops_link`: the two parts' links agree on `root_B` and on each rep's later tops. Both parts' readers'
  message is the run's at every non-fixed round before the last, on the bytes its pieces span (`CmtAgrees`): round 0
  has no pieces, so it is the whole of `root_B` (`cmt_zero`); a stream round's tops lie inside its pieces' span,
  each top's piece being `128 · n` bytes (`round_view`, `pieceLen_topPiece`). Equal bytes of 128-byte nodes are equal
  nodes (`eq_of_flatten_nodes`), so `topRow` reads one row at both (`topRow_link`).

`outCommitTie_deployed` assembles them at `layR`, the dense form, pieces in range and the inner verifier's coins
within the bound's.
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open FlockSoundness.Discharge.Recursion (innerLevels openRows openDirs)

/-! ## Lists of nodes -/

theorem eq_of_flatten_lengths {α : Type} : ∀ {L L' : List (List α)}, L.map List.length = L'.map List.length →
    L.flatten = L'.flatten → L = L'
  | [], [], _, _ => rfl
  | [], _ :: _, h, _ => by simp at h
  | _ :: _, [], h, _ => by simp at h
  | a :: L, b :: L', h, he => by
    simp only [List.map_cons, List.cons.injEq] at h
    simp only [List.flatten_cons] at he
    have hab : a = b := by
      have := congrArg (List.take a.length) he
      rwa [List.take_left, h.1, List.take_left] at this
    subst hab
    rw [eq_of_flatten_lengths h.2 (List.append_cancel_left he)]

theorem eq_of_flatten_nodes {L L' : List (List UInt8)} (h : ∀ a ∈ L, a.length = 128)
    (h' : ∀ a ∈ L', a.length = 128) (he : L.flatten = L'.flatten) : L = L' := by
  have hm : L.map List.length = L.map fun _ => 128 := List.map_congr_left h
  have hm' : L'.map List.length = L'.map fun _ => 128 := List.map_congr_left h'
  have hl : L.length = L'.length := by
    have := congrArg List.length he
    rw [List.length_flatten, List.length_flatten, hm, hm', sum_map_const, sum_map_const] at this
    omega
  refine eq_of_flatten_lengths ?_ he
  rw [hm, hm']
  exact List.ext_getElem (by simpa using hl) (fun i _ _ => by simp)

theorem byteArrays_eq {a b : Array ByteArray} (h : a.toList.map (·.toList) = b.toList.map (·.toList)) : a = b := by
  have hinj : Function.Injective (fun c : ByteArray => c.toList) := fun c d hcd => by
    have e : c.data.toList = d.data.toList := by
      rw [← byteArray_toList_eq, ← byteArray_toList_eq]
      exact hcd
    cases c
    cases d
    congr
    exact Array.toList_inj.1 e
  exact Array.toList_inj.1 ((List.map_injective_iff.2 hinj) h)

theorem nodes_length {a : Array ByteArray} (h : ∀ b ∈ a, b.size = 128) :
    ∀ l ∈ a.toList.map (·.toList), l.length = 128 := by
  intro l hl
  obtain ⟨b, hb, rfl⟩ := List.mem_map.1 hl
  rw [byteArray_length, h b (Array.mem_toList_iff.1 hb)]

theorem nodes_eq {a b : Array ByteArray} (ha : ∀ c ∈ a, c.size = 128) (hb : ∀ c ∈ b, c.size = 128)
    (h : (a.toList.map (·.toList)).flatten = (b.toList.map (·.toList)).flatten) : a = b :=
  byteArrays_eq (eq_of_flatten_nodes (nodes_length ha) (nodes_length hb) h)

/-! ## The link's `root_B` and `topRow` -/

/-- **Every node of `root_B` is 128 bytes**: the link reads it through `RecVStar.hexList`. -/
theorem innerLink_root {tr : Lean.Json} {record : String} {lk : RecVStar.InnerLink}
    (h : RecVStar.innerLink tr record = .ok lk) : ∀ b ∈ lk.root, b.size = 128 := by
  unfold RecVStar.innerLink at h
  simp only [bind, Except.bind, pure, Except.pure, throw, throwThe, MonadExceptOf.throw] at h
  repeat' split at h
  all_goals try (cases h; done)
  cases h
  exact hexList_size (by assumption)

/-- **`topRow` reads one row at two links that agree where it reads**: `root_B` at level 0, both reps' level-`l` tops
above it. -/
theorem topRow_link {lk lk' : RecVStar.InnerLink} {l m : ℕ} {wh wh' : String} {c c' : ByteArray}
    (h0 : l = 0 → lk.root = lk'.root)
    (hs : l ≠ 0 → ∀ rep < 2,
      (lk.tops[rep]?.bind (·[l - 1]?)).map (·.2) = (lk'.tops[rep]?.bind (·[l - 1]?)).map (·.2))
    (h : RecVStar.topRow lk l m = .ok (wh, c)) (h' : RecVStar.topRow lk' l m = .ok (wh', c')) : c = c' := by
  unfold RecVStar.topRow at h h'
  by_cases hl : l = 0
  · subst hl
    simp only [beq_self_eq_true, ↓reduceIte] at h h'
    rw [h0 rfl] at h
    cases hm : lk'.root[m]? with
    | none => simp only [hm, reduceCtorEq] at h
    | some d =>
      simp only [hm, Except.ok.injEq, Prod.mk.injEq] at h h'
      exact h.2.symm.trans h'.2
  · have hb : (l == 0) = false := by simpa using hl
    simp only [hb, Bool.false_eq_true, ↓reduceIte] at h h'
    have e0 := hs hl 0 (by omega)
    have e1 := hs hl 1 (by omega)
    revert h h' e0 e1
    generalize lk.tops[0]?.bind (·[l - 1]?) = o0
    generalize lk.tops[1]?.bind (·[l - 1]?) = o1
    generalize lk'.tops[0]?.bind (·[l - 1]?) = o0'
    generalize lk'.tops[1]?.bind (·[l - 1]?) = o1'
    intro h h' e0 e1
    rcases o0 with _ | ⟨k0, t0⟩ <;> rcases o1 with _ | ⟨k1, t1⟩ <;> try (simp at h; done)
    rcases o0' with _ | ⟨k0', t0'⟩ <;> rcases o1' with _ | ⟨k1', t1'⟩ <;> try (simp at e0 e1 h'; done)
    simp only [Option.map_some, Option.some.injEq] at e0 e1
    subst e0 e1
    simp only at h h'
    cases h0m : t0[m]? <;> cases h1m : t1[m - t0.size]? <;>
      simp only [h0m, h1m, Except.ok.injEq, Prod.mk.injEq, reduceCtorEq] at h h' <;>
      exact h.2.symm.trans h'.2

/-! ## Check 5's link check at a level -/

theorem capParams_open (L H : ℕ) : VStar.capParams (RecOpen.unitName L H) = none := by
  cases h : VStar.capParams (RecOpen.unitName L H) with
  | none => rfl
  | some p =>
    obtain ⟨l, N⟩ := p
    exact absurd (capName_of_params h).symm (capName_ne_open l N L H)

theorem capParams_of_open {name : String} {L H : ℕ} (h : RecOpen.params name = some (L, H)) :
    VStar.capParams name = none := by
  have hn : name = RecOpen.unitName L H := (FlockVBridge.unitName_params h).symm
  rw [hn]
  exact capParams_open L H

/-- **The link check at a level runs `checkOpenTops`**, at the registered reads, the inner session check 5 loads and
the link. -/
theorem checkLink_open {st : Stmt} {own : Array Registered.Port} {f : RecCoins.Source} {lk : RecVStar.InnerLink}
    (h : RecCoins.checkLink st (some own) (some f) (some lk) = .ok ()) {LANES H : ℕ}
    (ho : RecOpen.params st.c.unit.name = some (LANES, H)) :
    ∃ inn, f.load st.c.unit.name = .ok inn ∧ RecCoins.checkOpenTops inn st.c own lk LANES H = .ok () := by
  have hc := capParams_of_open ho
  have h' : (f.load st.c.unit.name >>= fun inn => RecCoins.checkOpenTops inn st.c own lk LANES H) = .ok () := by
    revert h
    unfold RecCoins.checkLink
    simp [ho, hc]
  exact FlockVBridge.PartS.bind_ok h'

/-- Rep 0's level-`l` top's node count, as the link holds it (`checkOpenTops`' `top0`). -/
def top0 (lk : RecVStar.InnerLink) (l : ℕ) : ℕ := ((lk.tops[0]?.bind (·[l - 1]?)).map (·.2.size)).getD 0

/-- **The node instance `N` of a level's `out` reads** (`checkLevelTopReads`' `want`): rep 0's openings at their
positions' nodes, rep 1's after `t0` rows above level 0. -/
def outRead (t0 l q H : ℕ) (pos0 pos1 : Array ℕ) (N : ℕ) : ℕ :=
  if N < q then pos0[N]! >>> H else (if l == 0 then 0 else t0) + (pos1[N - q]! >>> H)

theorem checkLevelTopReads_ok {unit : String} {own : Array Registered.Port} {l q H t0 : ℕ} {pos0 pos1 : Array ℕ}
    (h : RecCoins.checkLevelTopReads unit own l q H t0 pos0 pos1 = .ok ()) :
    ∃ reg, own.find? (·.name == "out") = some reg ∧ reg.value = s!"rec-top-L{l}" ∧
      reg.reads = (Array.range (2 * q)).map fun i => some (outRead t0 l q H pos0 pos1 i) := by
  unfold RecCoins.checkLevelTopReads at h
  cases hf : own.find? (·.name == "out") with
  | none => simp [hf, throw, throwThe, MonadExceptOf.throw] at h
  | some reg =>
    simp only [hf] at h
    cases hv : (reg.value != s!"rec-top-L{l}")
    · rw [hv] at h
      cases hr : (reg.reads != (Array.range (2 * q)).map fun i =>
          some (if i < q then pos0[i]! >>> H else (if l == 0 then 0 else t0) + (pos1[i - q]! >>> H)))
      · exact ⟨reg, rfl, by simpa using hv, by simpa [outRead] using hr⟩
      · rw [hr] at h
        simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    · rw [hv] at h
      simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- **What `checkOpenTops` accepting checked**: the inner session's algebra, its one level `l` of `LANES` lanes and
`H` siblings, both reps' `q` query positions there, and the level's `out` reads at them (`checkLevelTopReads`). -/
theorem checkOpenTops_ok {inn : RecCoins.Inner} {c : Circuit} {own : Array Registered.Port}
    {lk : RecVStar.InnerLink} {LANES H : ℕ} (h : RecCoins.checkOpenTops inn c own lk LANES H = .ok ()) :
    ∃ sh extras l r0 r1, inn.algebra = .ok (sh, extras) ∧ innerLevels sh LANES H = [l] ∧
      inn.rep sh extras 0 = .ok r0 ∧ inn.rep sh extras 1 = .ok r1 ∧
      RecCoins.checkLevelTopReads c.unit.name own l (sh.sched.levels[l]!).q H (top0 lk l)
        (r0.info.lg.conss[l]!).pos (r1.info.lg.conss[l]!).pos = .ok () := by
  unfold RecCoins.checkOpenTops at h
  dsimp only at h
  obtain ⟨⟨sh, extras⟩, halg, h⟩ := FlockVBridge.PartS.bind_ok h
  try dsimp only at h
  split at h
  · rename_i l hls
    have hls' : innerLevels sh LANES H = [l] := hls
    obtain ⟨pos, hpos, h⟩ := FlockVBridge.PartS.bind_ok h
    obtain ⟨a, b, ha, hb, rfl⟩ := FlockC3.Opening.mapM_two hpos
    obtain ⟨r0, h0, ha⟩ := FlockVBridge.PartS.bind_ok ha
    obtain ⟨r1, h1, hb⟩ := FlockVBridge.PartS.bind_ok hb
    have ea : (r0.info.lg.conss[l]!).pos = a := by
      revert ha
      cases hcs : r0.info.lg.conss[l]? with
      | none => intro ha; simp [throw, throwThe, MonadExceptOf.throw] at ha
      | some cs =>
        intro ha
        simp only at ha
        split at ha
        · cases ha
          rw [getElem!_def, hcs]
        · cases ha
    have eb : (r1.info.lg.conss[l]!).pos = b := by
      revert hb
      cases hcs : r1.info.lg.conss[l]? with
      | none => intro hb; simp [throw, throwThe, MonadExceptOf.throw] at hb
      | some cs =>
        intro hb
        simp only at hb
        split at hb
        · cases hb
          rw [getElem!_def, hcs]
        · cases hb
    refine ⟨sh, extras, l, r0, r1, halg, hls', h0, h1, ?_⟩
    rw [ea, eb]
    exact h
  · cases h

/-! ## The level's `out` slot is a row of `rec-top-L<l>` -/

/-- **The level's `out` slot at instance `N` is `topRow lk l` at `outRead`**: at a staged `.recOpen lv LANES HH n`
part the witness checks against, the setup's check 5 loads the inner session `inn` and its link check runs
`checkOpenTops` (`checkLink_open`), so `out` reads `rec-top-L<l>` at instance `N`'s node `outRead` (rep `N / q`'s
opening's position's node, `checkLevelTopReads_ok`); the labels are `0, …, n − 1` (`CopySite.idx_staged`, since
check 5's `dirs` rows are the statement's `n = 2 q` instances, `drSize_staged`), and `checkTopRows` pins the row at
that position to `topRow lk l` (`checkTopRows_rows`). The slot is its instance's public row (`slotPub_table`). -/
theorem outRow_bytes {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {lv LANES HH n : ℕ} (hsv : ss[s]? = some (.recOpen lv LANES HH n))
    {st0 : Claims.Recursion.Stmt} (hst : parse x = some st0) (hdn : Claims.Recursion.Form.Dense st0.form)
    (hlev : s < (OpenLower.schAt (xIOf x) 0).levels.size)
    {i : RecIns} {lk : RecVStar.InnerLink} (hi : insO xO = some i) (hlk : i.link = some lk)
    {qo : ℕ} (hqo : c.ports.findIdx? (·.name == "out") = some qo)
    (hm : ∀ (spec : SessionSpec) (sts : Array Stmt), setupO xO = some (i, spec, sts) → ∀ (_ : 0 < sts.size)
      (f : RecCoins.Source) (inn : RecCoins.Inner), RecCoins.Source.atC3 i.src i.inner i.commits = some f →
      f.load sts[0].c.unit.name = .ok inn →
      ∃ v, FlockDeploy.setupOf false (xIOf x) = some v ∧ inn.m = (FlockDeploy.tabsOf v.2.2)[0]!.m) :
    ∃ (spec : SessionSpec) (sts : Array Stmt) (h0 : 0 < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner)
      (sh : Shape) (extras : Array Claim) (l : ℕ) (r0 r1 : RecCoins.Rep),
      setupO xO = some (i, spec, sts) ∧ RecCoins.Source.atC3 i.src i.inner i.commits = some f ∧
      f.load sts[0].c.unit.name = .ok inn ∧ inn.algebra = .ok (sh, extras) ∧ innerLevels sh LANES HH = [l] ∧
      inn.rep sh extras 0 = .ok r0 ∧ inn.rep sh extras 1 = .ok r1 ∧
      ∀ N < n, ∃ wh cc, RecVStar.topRow lk l (outRead (top0 lk l) l (sh.sched.levels[l]!).q HH
          (r0.info.lg.conss[l]!).pos (r1.info.lg.conss[l]!).pos N) = .ok (wh, cc) ∧
        readForms w (hmCommit c qo N) = cc.toList := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss', hss', hsv'⟩, h0p, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hS' := hS
  obtain ⟨ms, i', spec, sts, rfl, hsu, -⟩ := hS'
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain rfl : i' = i := Option.some.inj (hins.symm.trans hi)
  have hp := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  obtain ⟨hsz1, hpopn⟩ := oneFull_staged hS hsu
  have h0 : 0 < sts.size := by omega
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hsz : p.sts.size = 1 := by rw [hp]; exact hsz1
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0p
  -- the unit, check 5's site, its rows and its link check
  obtain ⟨_, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0p
  have hn : p.sts[0].c.unit.name = Flock.RecOpen.unitName LANES HH := (htab 0 h0p).1
  have hpar := FlockVBridge.params_of_check P.recNet hn
  rw [e0] at hpar
  obtain ⟨own, f, hown, hf', hso, hop, hsp, -⟩ := CopySite.copySite_staged hsu h0 (.inl (by simp [hpar]))
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset) 0 h0
  rw [hown, hf'] at hcoin
  obtain ⟨inn, hinn, hco⟩ := FlockSoundness.Discharge.Recursion.openCheck_of_check hcoin hso hop hsp hpar
  obtain ⟨cr, dr, hrows, -, what, hdv⟩ := FlockSoundness.Discharge.Recursion.open_of_check hco
  have hlnk := Flock.Readable.array_forM_ok (RecIns.setup_checkLink hset) 0 h0
  rw [hown, hf', hlk] at hlnk
  obtain ⟨inn', hinn', htops⟩ := checkLink_open hlnk hpar
  rw [hinn] at hinn'
  cases hinn'
  obtain ⟨sh, extras, l, r0, r1, halg, hl, h0r, h1r, hlev'⟩ := checkOpenTops_ok htops
  obtain ⟨sh', extras', l', r0', r1', halg', hl', h0r', h1r', rfl⟩ := OpenLower.openRows_dirs hrows
  rw [halg] at halg'
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj halg')
  rw [hl] at hl'
  obtain rfl : l = l' := List.head_eq_of_cons_eq hl'
  rw [h0r] at h0r'
  cases h0r'
  rw [h1r] at h1r'
  cases h1r'
  -- check 5's `dirs` rows are the statement's `n = 2 q` instances, the labels `0, …, n − 1`
  have hdrs : (openDirs sh l HH r0 r1).size = n :=
    OpenDirs.drSize_staged hst hdn hr hss hsv hlev (hm spec sts hsu h0 f inn hf' hinn) hrows
  have hdr2 : (openDirs sh l HH r0 r1).size = 2 * (sh.sched.levels[l]!).q := by
    simp [FlockSoundness.Discharge.Recursion.openDirs]
  obtain ⟨-, r3, ss3, sv3, hr3, hss3, hsv3, hpop⟩ := staged_scope hS hx h0p
  rw [hr] at hr3
  cases hr3
  rw [hss] at hss3
  cases hss3
  rw [hsv] at hsv3
  cases hsv3
  have hpop' : Flock.RecVStar.population p.sts[0] = n := hpop
  have hle := hpopn 0 h0
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨_, hset0⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨d, hd, -⟩ := hall 0 h0p
  have hnn := setupOne_n hd
  rw [e0] at hpop' hnn
  have hidx := CopySite.idx_staged hsu h0 hown hdv (by omega) (CopySite.labels_staged hS hsu h0)
  -- the slot
  obtain ⟨L, hz⟩ := layout_of_check_regs hck hx ht hpi h0p
  have hz1 : (L.model (FlockSoundness.Refine.regionsOf p.sts[0])).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0p
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0p).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L _ hz1
  have htop := Flock.Readable.array_forM_ok (RecIns.setup_topRows hset) 0 h0
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0)
  rw [hown, hlk] at htop
  rw [hown] at hreg
  rw [← e0] at htop hreg hidx hlev'
  obtain ⟨reg, hfreg, hval, hreads⟩ := checkLevelTopReads_ok hlev'
  have hmem : reg ∈ own := Array.mem_of_find?_eq_some hfreg
  have hname : reg.name = "out" := by simpa using Array.find?_some hfreg
  have hlt' : RecVStar.topLevel reg.value = some l := by rw [hval]; exact topLevel_top l
  obtain ⟨t, htt, pp, positions, paths, hpp, hent, hrows'⟩ := checkTopRows_rows htop hmem hlt'
  obtain ⟨t', idx, ht'', hidx', hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  rw [htt] at ht''
  cases ht''
  obtain rfl : Array.range p.sts[0].pub.n = idx := Except.ok.inj (hidx.symm.trans hidx')
  obtain ⟨p', positions', paths', hp', hent', hall'⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  obtain ⟨p'', positions'', paths'', hp'', hent'', hsize⟩ := FlockC3.PubRows.checkPort_size (hports _ hmem)
  rw [hpp] at hp' hp''
  cases hp'
  cases hp''
  rw [hent] at hent' hent''
  cases hent'
  cases hent''
  rw [hname] at hpp
  subst hc0
  have hq2 : p.sts[0].c.ports.findIdx? (·.name == "out") = some pp :=
    findIdx_of_find_range (f := fun (x : Port) => x.name == "out") hpp
  rw [hqo] at hq2
  cases hq2
  refine ⟨spec, sts, h0, f, inn, sh, extras, l, r0, r1, hsu, hf', hinn, halg, hl, h0r, h1r, fun N hN => ?_⟩
  have hN2 : N < 2 * (sh.sched.levels[l]!).q := by omega
  have hnp : N < p.sts[0].pub.n := by rw [e0]; omega
  have hnN : N < p.sts[0].n := by rw [setupOne_n hd]; exact hnp
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hqo
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg0 hnN hnbl)
  rw [digest_tables htt] at hsl
  obtain ⟨hlt, hrd, -⟩ := hall' N hnp
  have hrn : ((Array.range (2 * (sh.sched.levels[l]!).q)).map fun i =>
      some (outRead (top0 lk l) l (sh.sched.levels[l]!).q HH (r0.info.lg.conss[l]!).pos
        (r1.info.lg.conss[l]!).pos i))[N]? =
      some (some (outRead (top0 lk l) l (sh.sched.levels[l]!).q HH (r0.info.lg.conss[l]!).pos
        (r1.info.lg.conss[l]!).pos N)) := by
    simp [hN2]
  rw [hreads, range_getD hnp, hrn] at hrd
  have hpos : positions.getD (Registered.ref t p.sts[0].c.ports.size qo N) 0 =
      outRead (top0 lk l) l (sh.sched.levels[l]!).q HH (r0.info.lg.conss[l]!).pos (r1.info.lg.conss[l]!).pos N :=
    (Option.some.inj (Option.some.inj hrd)).symm
  obtain ⟨wh, cc, htr, hrow⟩ := hrows' _ (hsize ▸ hlt)
  rw [hpos] at htr
  refine ⟨wh, cc, htr, ?_⟩
  rw [hsl, hrow, byteArray_toList_eq]

/-! ## The two parts' links agree -/

/-- **Round 0 is `root_B`, of 128-byte nodes**: a staged part's record's round 0 is its link's `root_B`
(`staged_reads_link`, `msgs_zero`). -/
theorem cmt_zero {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {i : RecIns}
    {lk : RecVStar.InnerLink} (hi : insO xO = some i) (hl : i.link = some lk) :
    cmtTr recRead x 0 xO = (lk.root.toList.map (·.toList)).flatten ∧ ∀ b ∈ lk.root, b.size = 128 := by
  obtain ⟨i', sh, cs, tt, tr, record, rc, lk', hi', -, -, hil, hR, hlk, hrd, hshown⟩ := staged_reads_link hS
  obtain ⟨lk'', rc', want, hlk', hrd', -, hroots, -⟩ := FlockC3.VStarList.checkReads_ok hR
  rw [hlk] at hlk'
  obtain rfl : lk' = lk'' := Except.ok.inj hlk'
  rw [hrd] at hrd'
  obtain rfl : rc = rc' := Option.some.inj hrd'
  rw [hi] at hi'
  obtain rfl : i = i' := Option.some.inj hi'
  rw [hl] at hil
  obtain rfl : lk = lk' := Option.some.inj hil
  exact ⟨by rw [cmtTr_record hshown, msgs_zero rc _ _ hroots], innerLink_root hlk⟩

/-- **Two staged parts' links share `root_B`**: round 0 has no pieces and is no fixed round, so at both parts the
readers' message there is the run's (`CmtAgrees`), each part's `root_B` (`cmt_zero`). -/
theorem root_link (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) {st : Stage} {s s' : ℕ}
    {x xO xO' : Msg} {ms : List Msg} (hS : StagedPart st s x xO) (hS' : StagedPart st s' x xO')
    (hK : 2 ≤ layoutOf.rounds x)
    (hA : (layoutBTops fin).CmtAgrees (cmtTr recRead) x xO ms)
    (hA' : (layoutBTops fin).CmtAgrees (cmtTr recRead) x xO' ms) {i i' : RecIns} {lk lk' : RecVStar.InnerLink}
    (hi : insO xO = some i) (hl : i.link = some lk) (hi' : insO xO' = some i') (hl' : i'.link = some lk') :
    lk.root = lk'.root := by
  have hp : (layoutBTops fin).pieceAt x 0 = [] := by
    show stepPieces fin x 0 = []
    unfold stepPieces
    rw [if_neg (by omega), if_pos rfl]
  have h1 := (hA 0 (by show 0 + 1 < layoutOf.rounds x; omega) rfl).1 hp
  have h2 := (hA' 0 (by show 0 + 1 < layoutOf.rounds x; omega) rfl).1 hp
  obtain ⟨e1, n1⟩ := cmt_zero hS hi hl
  obtain ⟨e2, n2⟩ := cmt_zero hS' hi' hl'
  exact nodes_eq n1 n2 (e1.symm.trans (h1.symm.trans (h2.trans e2)))

theorem getD_lt {a : Array ℕ} {e : ℕ} (h : e < a.size) : a.getD e 0 = a[e] := by
  simp [Array.getD, h]

theorem flatMap_inv {α β : Type} (f : β → List α) (d : β) :
    ∀ (L : List β) (g : ℕ), g < (L.flatMap f).length →
      ∃ k e, k < L.length ∧ e < (f (L.getD k d)).length ∧ ((L.take k).flatMap f).length + e = g
  | [], g, hg => by simp at hg
  | a :: L, g, hg => by
    simp only [List.flatMap_cons, List.length_append] at hg
    by_cases h : g < (f a).length
    · exact ⟨0, g, by simp, by simpa using h, by simp⟩
    · obtain ⟨k, e, hk, he, hke⟩ := flatMap_inv f d L (g - (f a).length) (by omega)
      refine ⟨k + 1, e, by simp; omega, by simpa using he, ?_⟩
      simp only [List.take_succ_cons, List.flatMap_cons, List.length_append]
      omega

/-- The schedule's `g`-th top is some round's `e`-th. -/
theorem topsBefore_inv {rs : Array RecVStar.SchedRound} {g : ℕ} (hg : g < (RecVStar.topSizes rs).length) :
    ∃ k e, k < rs.size ∧ e < ((rs.getD k default).2.1).size ∧ topsBefore rs k + e = g := by
  obtain ⟨k, e, hk, he, hke⟩ := flatMap_inv (fun w : RecVStar.SchedRound => w.2.1.toList) default rs.toList g hg
  have hgd : rs.toList.getD k default = rs.getD k default := by
    simp [Array.getD_eq_getD_getElem?, List.getD_eq_getElem?_getD]
  rw [hgd] at he
  exact ⟨k, e, by simpa using hk, by simpa using he, hke⟩

/-- **Two staged parts' links share each rep's later tops**: at stream round `j = rep · R + k` the readers' message
is the run's on the bytes its pieces span (`CmtAgrees`), which cover the round's tops (each top's piece is `128 · n`
bytes, `pieceLen_topPiece`); each part's record holds its link's tops there, at the schedule's sizes and of 128-byte
nodes (`round_view`), so equal bytes are equal tops. -/
theorem tops_link (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) {st : Stage} {s s' : ℕ}
    {x xO xO' : Msg} {ms : List Msg} (hS : StagedPart st s x xO) (hS' : StagedPart st s' x xO') {r : XR}
    (hr : readX x = some r) {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs)
    (hK : layoutOf.rounds x = 2 + 2 * rs.size)
    (hA : (layoutBTops fin).CmtAgrees (cmtTr recRead) x xO ms)
    (hA' : (layoutBTops fin).CmtAgrees (cmtTr recRead) x xO' ms) {i i' : RecIns} {lk lk' : RecVStar.InnerLink}
    (hi : insO xO = some i) (hl : i.link = some lk) (hi' : insO xO' = some i') (hl' : i'.link = some lk')
    {rep g : ℕ} (hrep : rep < 2) (hg : g < (RecVStar.topSizes rs).length) :
    ((lk.tops.getD rep #[])[g]?).map (·.2) = ((lk'.tops.getD rep #[])[g]?).map (·.2) := by
  obtain ⟨k, e, hk, he, rfl⟩ := topsBefore_inv hg
  obtain ⟨i1, lk1, cs1, hi1, hl1, -, hv⟩ := round_view hS hr hrs
  obtain ⟨i2, lk2, cs2, hi2, hl2, -, hv'⟩ := round_view hS' hr hrs
  rw [hi] at hi1
  obtain rfl : i = i1 := Option.some.inj hi1
  rw [hl] at hl1
  obtain rfl : lk = lk1 := Option.some.inj hl1
  rw [hi'] at hi2
  obtain rfl : i' = i2 := Option.some.inj hi2
  rw [hl'] at hl2
  obtain rfl : lk' = lk2 := Option.some.inj hl2
  obtain ⟨y, hmsg, hlens, htops, -⟩ := hv rep hrep k hk
  obtain ⟨y', hmsg', hlens', htops', -⟩ := hv' rep hrep k hk
  have hpos : 0 < rs.size := by omega
  have hj : rep * rs.size + k < 2 * rs.size := by
    have : rep * rs.size ≤ 1 * rs.size := Nat.mul_le_mul_right _ (by omega)
    omega
  have hjm : (rep * rs.size + k) % rs.size = k := by
    rw [Nat.add_comm, Nat.add_mul_mod_self_right, Nat.mod_eq_of_lt hk]
  have hjd : (rep * rs.size + k) / rs.size = rep := by
    rw [Nat.add_comm, Nat.add_mul_div_right _ _ hpos, Nat.div_eq_of_lt hk, Nat.zero_add]
  have hne : ((rs.getD k default).2.1).isEmpty = false := by
    simp only [Array.isEmpty, decide_eq_false_iff_not]
    omega
  have hsr : streamRound x (rep * rs.size + k) = some (rs.getD k default) := by
    simp [streamRound, hr, hrs, hj, hjm]
  have hfx : (layoutBTops fin).fixedAt x (rep * rs.size + k + 1) = none := by
    show (streamRound x (rep * rs.size + k)).bind _ = none
    rw [hsr]
    show (if ((rs.getD k default).2.1.isEmpty && !(rs.getD k default).2.2.2) = true then some [] else none) = none
    rw [hne]
    rfl
  have hpc : (layoutBTops fin).pieceAt x (rep * rs.size + k + 1) = topsPiecesAt r rs (rep * rs.size + k) := by
    show stepPieces fin x (rep * rs.size + k + 1) = _
    unfold stepPieces
    rw [if_neg (by omega), if_neg (by omega), Nat.add_sub_cancel]
    simp [topsPieces, hr, hrs]
  have hT := (hA _ (by show rep * rs.size + k + 1 + 1 < layoutOf.rounds x; omega) hfx).2
  have hT' := (hA' _ (by show rep * rs.size + k + 1 + 1 < layoutOf.rounds x; omega) hfx).2
  have heq := hT.symm.trans hT'
  rw [hmsg, hmsg'] at heq
  -- the tops' bytes and the pieces' span
  have hlenT := tops_length hlens
  have hlenT' := tops_length hlens'
  have hML : ∀ (z : RecRecord.Round), z.tops.map List.length = ((rs.getD k default).2.1).toList →
      (∀ e (he : e < z.tops.length), ∃ top : Array ByteArray,
        z.tops[e] = top.toList.map (·.toList) ∧ ∀ b ∈ top, b.size = 128) →
      (z.tops.map List.flatten).map List.length = ((rs.getD k default).2.1).toList.map (128 * ·) := by
    intro z hz hzt
    have hzl := tops_length hz
    apply List.ext_getElem (by simp [hzl])
    intro e h1 h2
    have he' : e < z.tops.length := by simpa using h1
    have hfl : (z.tops[e]).flatten.length = 128 * z.tops[e].length := by
      obtain ⟨top, hye, h128⟩ := hzt e he'
      rw [hye, flatten_nodes_length h128]
      simp
    simp only [List.getElem_map, Array.getElem_toList]
    rw [hfl, tops_len hz he']
    have he2 : e < ((rs.getD k default).2.1).size := by rw [← hzl]; exact he'
    rw [getD_lt he2]
  have hm1 := hML y hlens fun e he => by
    obtain ⟨kt, top, -, hye, h128⟩ := htops e he
    exact ⟨top, hye, h128⟩
  have hm2 := hML y' hlens' fun e he => by
    obtain ⟨kt, top, -, hye, h128⟩ := htops' e he
    exact ⟨top, hye, h128⟩
  have hsum : ∀ (z : RecRecord.Round),
      (z.tops.map List.flatten).map List.length = ((rs.getD k default).2.1).toList.map (128 * ·) →
      (z.tops.map List.flatten).flatten.length = (((rs.getD k default).2.1).toList.map (128 * ·)).sum := by
    intro z hz
    rw [List.length_flatten, hz]
  have hpl : (((rs.getD k default).2.1).toList.map (128 * ·)).sum ≤ (layoutBTops fin).piecesLen x
      (rep * rs.size + k + 1) := by
    unfold Claims.Recursion.VStarB.LayoutB.piecesLen
    rw [hpc]
    unfold topsPiecesAt
    rw [hjm, if_neg (by rw [hne]; decide)]
    rw [List.map_append, List.sum_append]
    have hcap : ((List.range ((rs.getD k default).2.1).size).map (capPieceAt r rs (rep * rs.size + k))).map
        (fun p => Claims.Recursion.VStarB.pieceLen p.2) = ((rs.getD k default).2.1).toList.map (128 * ·) := by
      apply List.ext_getElem (by simp)
      intro e h1 h2
      have he2 : e < ((rs.getD k default).2.1).size := by simpa using h1
      simp only [List.getElem_map, List.getElem_range, capPieceAt, pieceLen_topPiece, hjm, Array.getElem_toList]
      rw [getD_lt he2]
    rw [hcap]
    omega
  have hTe : (y.tops.map List.flatten).flatten = (y'.tops.map List.flatten).flatten := by
    have hl1 := hsum y hm1
    have hl2 := hsum y' hm2
    have := congrArg (List.take (y.tops.map List.flatten).flatten.length) heq
    rw [List.take_take, List.take_take, Nat.min_eq_left (by omega), List.take_left,
      show (y.tops.map List.flatten).flatten.length = (y'.tops.map List.flatten).flatten.length by omega,
      List.take_left] at this
    exact this
  have hflat := eq_of_flatten_lengths (hm1.trans hm2.symm) hTe
  have hey : e < y.tops.length := by rw [hlenT]; exact he
  have hey' : e < y'.tops.length := by rw [hlenT']; exact he
  obtain ⟨kt, top, hkt, hye, h128⟩ := htops e hey
  obtain ⟨kt', top', hkt', hye', h128'⟩ := htops' e hey'
  have hfe : (y.tops[e]).flatten = (y'.tops[e]).flatten := by
    have := congrArg (·[e]?) hflat
    simp only [List.getElem?_map, List.getElem?_eq_getElem hey, List.getElem?_eq_getElem hey', Option.map_some,
      Option.some.injEq] at this
    exact this
  rw [hye, hye'] at hfe
  rw [hkt, hkt', Option.map_some, Option.map_some, nodes_eq h128 h128' hfe]

/-! ## Rep 1's rows at a later level start after rep 0's `2^cap` -/

theorem levelCap_get {sh : VStar.Algebra.Shape} {K : Array VStar.Stmt} (h : VStar.levelCapStatements sh = .ok K)
    {g : ℕ} (hg : g < K.size) : ∃ s, sh.sched = .ok s ∧ K[g] = .cap (g + 1) (2 * 2 ^ s.levels[g + 1]!.cap) := by
  unfold VStar.levelCapStatements at h
  cases hs : sh.sched with
  | error e => rw [hs] at h; cases h
  | ok s =>
    rw [hs] at h
    cases h
    exact ⟨s, rfl, by simp⟩

/-- **A later cap's level-`l` top has `2^cap` nodes**: at a dense statement, a RecCap statement `.cap l N` with
`l ≠ 0` is V\*'s later cap at level `l` (`FlockC3.Reps.list_ok`, the list's parts holding no cap), holding
`2 · 2^cap` nodes (`VStar.levelCapStatements`), twice the schedule's `(l − 1)`-th top's (`RecVStar.checkTopsListed`),
at table 0's schedule (`inner_m`). -/
theorem laterCap_size {x : Msg} {st0 : Claims.Recursion.Stmt} (hst : parse x = some st0)
    (hdn : Claims.Recursion.Form.Dense st0.form) {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {s' l N : ℕ} (hsv : ss[s']? = some (.cap l N)) (hl : l ≠ 0)
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) :
    (RecVStar.topSizes rs)[l - 1]? = some (2 ^ ((OpenLower.schAt (xIOf x) 0).levels[l]!).cap) := by
  obtain ⟨r', hr', -, rfl⟩ := parse_read hst
  rw [hr] at hr'
  cases hr'
  have hsp : r.shape.sparse = none := by
    have h2 : (r.shape.sparse.map fun sp => (sp.nu, sp.ell, sp.sLog, sp.tLog)) = none := hdn.2
    cases hs : r.shape.sparse with
    | none => rfl
    | some _ => simp [hs] at h2
  obtain ⟨_, _, _, _, _, _, A, C, _, _, _, _, _, hA, hC, K, n0, hK, hT, rfl⟩ := FlockC3.Reps.list_ok hss
  obtain rfl := (FlockC3.VStarList.candidate3_ok hC).1 hsp
  obtain ⟨sch, sp, ps, hsch, -, -, hAe⟩ := FlockVBridge.Family.statementsOf_eq hA
  have hge : A.size ≤ s' := by
    by_contra hlt
    rw [Array.append_assoc, Array.getElem?_append_left (by omega)] at hsv
    have hm := Array.mem_of_getElem? hsv
    rw [hAe] at hm
    simp at hm
  rw [Array.append_assoc, Array.getElem?_append_right hge, Array.empty_append] at hsv
  have hsz := topsListed_size hT hrs
  by_cases hgK : s' - A.size < K.size
  · rw [Array.getElem?_append_left hgK, Array.getElem?_eq_getElem hgK] at hsv
    obtain ⟨s, hs, hKg⟩ := levelCap_get hK hgK
    rw [hKg] at hsv
    simp only [Option.some.injEq, VStar.Stmt.cap.injEq] at hsv
    obtain ⟨hl1, hN⟩ := hsv
    have hg : l - 1 = s' - A.size := by omega
    have hgt : s' - A.size < (RecVStar.topSizes rs).length := by rw [← hsz]; exact hgK
    obtain ⟨nn, hnn⟩ : ∃ nn, (RecVStar.topSizes rs)[s' - A.size]? = some nn :=
      ⟨_, List.getElem?_eq_getElem hgt⟩
    have hi := topsListed_at hT hrs hnn hgK
    rw [hKg] at hi
    simp only [VStar.Stmt.instances] at hi
    have hnn2 : nn = 2 ^ s.levels[s' - A.size + 1]!.cap := by omega
    have hfs := FlockVBridge.Family.sched_fast100 hs
    obtain ⟨s0, spec, t, hset, htm⟩ := OpenLower.inner_m hst hr
    have e : OpenLower.schAt (xIOf x) 0 = s := by
      unfold OpenLower.schAt
      rw [hset]
      have htabs : FlockDeploy.tabsOf #[t] = #[_root_.Flock.Setup.ofCircuit t] := by simp [FlockDeploy.tabsOf]
      have h0 : (FlockDeploy.tabsOf #[t])[0]!.m = t.m := by
        rw [htabs]
        rfl
      dsimp only
      rw [h0, htm]
      have : (FlockC3.Reps.vshOf r.shape).m = r.shape.m := rfl
      rw [← this, hfs]
      rfl
    rw [hg, hnn, e, hnn2, ← hl1]
  · exfalso
    rw [Array.getElem?_append_right (by omega)] at hsv
    by_cases he : s' - A.size - K.size = 0
    · rw [he] at hsv
      simp at hsv
      omega
    · rw [Array.getElem?_eq_none (by simp; omega)] at hsv
      cases hsv

/-- **Rep 0's level-`g + 1` top in the link has the schedule's `g`-th top's nodes** (`round_view`, `topSizes_at`). -/
theorem top0_eq {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {r : XR} (hr : readX x = some r)
    {rs : Array RecVStar.SchedRound} (hrs : repSched r.shape = some rs) {i : RecIns} {lk : RecVStar.InnerLink}
    (hi : insO xO = some i) (hl : i.link = some lk) {g nn : ℕ} (hn : (RecVStar.topSizes rs)[g]? = some nn) :
    (lk.tops[0]?.bind (·[g]?)).map (·.2.size) = some nn := by
  have hg : g < (RecVStar.topSizes rs).length := (List.getElem?_eq_some_iff.1 hn).1
  obtain ⟨k, e, hk, he, rfl⟩ := topsBefore_inv hg
  obtain ⟨i1, lk1, cs1, hi1, hl1, -, hv⟩ := round_view hS hr hrs
  rw [hi] at hi1
  obtain rfl : i = i1 := Option.some.inj hi1
  rw [hl] at hl1
  obtain rfl : lk = lk1 := Option.some.inj hl1
  obtain ⟨y, -, hlens, htops, -⟩ := hv 0 (by omega) k hk
  have hey : e < y.tops.length := by rw [tops_length hlens]; exact he
  obtain ⟨kt, top, hkt, hye, -⟩ := htops e hey
  rw [getD_getElem?] at hkt
  rw [hkt, Option.map_some, tops_size hlens hey hye]
  rw [topSizes_at hk he] at hn
  have := Option.some.inj hn
  rw [← this, getD_lt he]

end RecDeploy

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (openRows openDirs innerLevels)
open RecDeploy.Rows (canonRounds posAt sideAt schOf verifierRows_comb)

/-- **A piece's query position is the one `posAt` reads off the coins** (`posTieR_of`, at the raw position rather
than its low `HH` bits): at a staged level part whose check 5 loads the inner session `inn`, with its level `l` of
`LANES` lanes and `HH` siblings and both reps, a piece `ownQR` gives the part is at level `l`, one of its `q`
queries of rep 0 or 1, and its rep's position there is `posAt`'s. -/
theorem posRaw_of {untr : TrReader} {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {st : Claims.Recursion.Stmt} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) (hR : PiecesInRangeR pieces)
    (hI : InnerCoinsR untr b Bk s x c) {xO : Msg} (hsh : showsPart untr b s x xO c) {i : RecIns}
    {spec : SessionSpec} {sts : Array Stmt} (hsu : setupO xO = some (i, spec, sts)) (ht : 0 < sts.size)
    {f : RecCoins.Source} {inn : RecCoins.Inner} (hf : RecCoins.Source.atC3 i.src i.inner i.commits = some f)
    (hload : f.load sts[0].c.unit.name = .ok inn) {r : XR} {ss : Array VStar.Stmt} {lv LANES HH n : ℕ}
    (hr : readX x = some r) (hss : _root_.Flock.RecVStar.statements r.shape = .ok ss)
    (hsv : ss[s]? = some (.recOpen lv LANES HH n)) {sh : Shape} {extras : Array Claim} {l : ℕ}
    {r0 r1 : RecCoins.Rep} (halg : inn.algebra = .ok (sh, extras)) (hl : innerLevels sh LANES HH = [l])
    (h0 : inn.rep sh extras 0 = .ok r0) (h1 : inn.rep sh extras 1 = .ok r1) {q : OpenPiece}
    (hq : q ∈ pieces (xIOf x)) (ho : ownQR x q = some s) :
    q.level = l ∧ qAt (xIOf x) q.level = (sh.sched.levels[l]!).q ∧ q.query < (sh.sched.levels[l]!).q ∧
      q.rep < 2 ∧
      ((if q.rep = 0 then r0 else r1).info.lg.conss[l]!).pos[q.query]! =
        posAt (xIOf x) ((List.range (Bk (xIOf x))).map c) q := by
  obtain ⟨v, hv, hm, hk, hex, hrd⟩ := hI xO hsh i spec sts hsu ht f inn hf hload
  obtain ⟨hext, hsh'⟩ := algebra_ok halg
  obtain ⟨sch, hsch, hshm, hshk, hshc, hshs, hshh⟩ := shape?_ok hsh'
  obtain ⟨hq0, r', ss', hr', hss', L', H', n', hsv'⟩ := ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv'
  obtain ⟨hlvq, -, -, -⟩ := hsv'
  obtain ⟨hlvs, hLANES, hHH, -⟩ := levelMatch hst hd hr hss hsv
  have hS : schAt (xIOf x) 0 = sch := by
    unfold schAt
    rw [hv]
    dsimp only
    rw [← hm, hsch, Option.getD_some]
  obtain ⟨hrep, hquery⟩ := hR (xIOf x) q hq
  rw [hq0, hS] at hquery
  have hlev : q.level < sch.levels.size := by
    by_contra hge
    rw [getElem!_neg sch.levels q.level hge] at hquery
    exact Nat.not_lt_zero _ hquery
  rw [hS, ← hlvs, hlvq] at hLANES hHH
  have hql : q.level = l := innerLevels_unique hl (by rw [hshs]; exact hlev) (by rw [hshs]; exact hLANES.symm)
    (by rw [hshs]; exact hHH.symm)
  have hqa : qAt (xIOf x) q.level = (sh.sched.levels[l]!).q := by
    unfold qAt
    rw [hS, hshs, hql]
  refine ⟨hql, hqa, by rw [hshs, ← hql]; exact hquery, hrep, ?_⟩
  have hrq : inn.rep sh extras q.rep = .ok (if q.rep = 0 then r0 else r1) := by
    obtain h | h : q.rep = 0 ∨ q.rep = 1 := by omega
    · rw [h]; simpa using h0
    · rw [h]; simpa using h1
  obtain ⟨hrounds, hvr⟩ := rep_ok hrq
  rw [verifierRows_filter (hrd q.rep hrep _ hrounds (verifierRows_pos hvr)), verifierRows_comb] at hvr
  have hexs : ((tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default))) =
      .ok extras := by rw [← hex, hext]
  unfold posAt
  rw [hv]
  dsimp only
  rw [hq0, hexs]
  simp only [Except.toOption, Option.getD_some, Nat.mul_zero, Nat.zero_add]
  rw [verifierRows_congr (sh := shapeOf (tabsOf v.2.2)[0]! ((fast100 (tabsOf v.2.2)[0]!.m).getD default) extras.size)
    (sh' := sh) (by
      simp only [algFields, Prod.mk.injEq]
      refine ⟨by rw [hshm, hm]; rfl, by rw [hshk, hk]; rfl, by rw [hshc]; rfl, by rw [hshs, ← hm, hsch]; rfl,
        by rw [hshh]; rfl⟩)]
  cases h : RecAlgebra.verifierRows sh (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast q.rep)
      extras #[] with
  | error e => rw [h] at hvr; cases hvr
  | ok w =>
    rw [h] at hvr
    obtain ⟨w₁, w₂, w₃⟩ := w
    simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at hvr
    obtain ⟨-, hi, -⟩ := hvr
    rw [← hi, hql]

end RecDeploy.OpenLower

namespace RecDeploy.CapLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Claims.Recursion.VStarB
open RecDeploy.OpenLower (qAt schAt PiecesInRangeR ownQR_some levelMatch)

/-- **`OutCommitTie` at the layout of record**, at the deployed reader's statements at the dense form, pieces in
range and the inner verifier's coins within the bound's: the level's `out` slot at the piece's instance is
`topRow lk l` at its opening's node (`outRow_bytes`), which is `capIdx` (`posRaw_of`, `top0_eq`, `laterCap_size`);
RecCap's `top` slot there is `topRow lk' l` (`capRow_bytes`); and the two links agree where `topRow` reads
(`root_link`, `tops_link`), both parts' readers' messages being the run's (`CmtAgrees`). -/
theorem outCommitTie_deployed {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {IV : Claims.Verifier Coin}
    {shape : Claims.Recursion.Form → Prop}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece) (hP : ∀ x, (OpenDirs.DR b elem IV).parse x = parse x)
    (hSh : ∀ f, shape f → Claims.Recursion.Form.Dense f) (hR : PiecesInRangeR pieces)
    (hB : ∀ x, RestRead.kI x ≤ Bd.DE.V.k (xIOf x)) :
    OutCommitTie FlockDeploy.sha512 (OpenDirs.DR b elem IV) shape recRead b (cmtTr recRead) (DeployedR.layR elem) Bd
      pieces Rows.posAt := by
  intro x st hx hf c ws hJ r ss s lv LANES HH n hr hss hsv xO w p hw hst q hq ho ht r' ss' hr' hss' lv' n' cc qo hsv'
    hcc hqo s' N' c' w' qc hs' hc' hw' hqc hlt
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  cases hsv'
  have hx' : parse x = some st := (hP x).symm.trans hx
  have hd := hSh _ hf
  have hxI : Claims.Recursion.xI (OpenDirs.DR b elem IV) x = xIOf x := by
    unfold Claims.Recursion.xI xIOf
    rw [hP x]
  rw [hxI] at hlt ⊢
  rw [hxI] at hq
  obtain ⟨-, -, -, ⟨-, -, hA, -⟩, -, -, hshow, hck, hS⟩ := hst
  -- the piece's level is the statement's
  obtain ⟨h0q, r2, ss2, hr2, hss2, L2, H2, n2, hsv2⟩ := ownQR_some ho
  rw [hr] at hr2
  cases hr2
  rw [hss] at hss2
  cases hss2
  rw [hsv] at hsv2
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv2
  obtain ⟨hlvq, -, -, -⟩ := hsv2
  obtain ⟨hls, -, -, hnq⟩ := levelMatch hx' hd hr hss hsv
  have hql : q.level = s := hlvq.symm.trans hls
  obtain ⟨hrep, hqry⟩ := hR _ q hq
  rw [h0q, hql] at hqry
  have hlev : s < (schAt (xIOf x) 0).levels.size := by
    by_contra hge
    rw [getElem!_neg _ s hge] at hqry
    exact Nat.not_lt_zero _ hqry
  -- the level part's slot
  obtain ⟨i, -, -, -, -, -, -, lk, hi, -, -, hlk, -⟩ := staged_reads_link hS
  have hIC := OpenLower.innerCoinsR_staged (b := b) (Bk := Bd.DE.V.k) (s := s) (c := c) hx' hd (hB x)
  obtain ⟨spec, sts, h0, f, inn, sh, extras, l, r0, r1, hsu, hf', hinn, halg, hl, h0r, h1r, hbytes⟩ :=
    outRow_bytes hS hck hr hcc hss hsv hx' hd hlev hi hlk hqo fun spec sts hsu h0 f inn hf hload => by
      obtain ⟨v, hv, hm, -⟩ := hIC xO ⟨hshow, hS⟩ i spec sts hsu h0 f inn hf hload
      exact ⟨v, hv, hm⟩
  obtain ⟨hlq, hqa, hqq, -, hpos⟩ := OpenLower.posRaw_of hx' hd hR hIC ⟨hshow, hS⟩ hsu h0 hf' hinn hr hss hsv
    halg hl h0r h1r hq ho
  -- RecCap's slot
  have hs'lt : s' < ws.length := (List.getElem?_eq_some_iff.1 hw').1
  have hw'e : ws[s'] = w' := (List.getElem?_eq_some_iff.1 hw').2
  obtain ⟨-, xO', hck', -, -, ⟨-, hS'⟩, -, -, hA', -⟩ := hJ.2.2.2 s' hs'lt
  rw [hw'e] at hck'
  obtain ⟨i', -, -, -, -, -, -, lk', hi', -, -, hlk', -⟩ := staged_reads_link hS'
  have hsv'' : (RecVStar.statements r.shape).toOption.bind (·[s']?) = some (.cap q.level N') := by
    rw [hss]
    exact hs'
  obtain ⟨qc', hqc', -, hcap⟩ := capRow_bytes hS' hck' hr hc' hsv'' hi' hlk'
  rw [hqc] at hqc'
  cases hqc'
  obtain ⟨wh', cc', htr', hb'⟩ := hcap _ hlt
  -- the level's instance reads node `capIdx`
  have hNn : q.rep * qAt (xIOf x) q.level + q.query < n := by
    rw [hnq, ← hql]
    rw [← hql] at hqry
    simp only [qAt]
    generalize ((schAt (xIOf x) 0).levels[q.level]!).q = Q at hqry ⊢
    have h1 : q.rep * Q ≤ 1 * Q := Nat.mul_le_mul_right _ (by omega)
    omega
  obtain ⟨wh, cc0, htr, hb⟩ := hbytes _ hNn
  obtain ⟨r3, rs, hr3, hrs, hK⟩ := schedMatches_dense x st hx' hd
  rw [hr] at hr3
  cases hr3
  have hK' : layoutOf.rounds x = 2 + 2 * rs.size := hK
  have htop0 : l ≠ 0 → top0 lk l = 2 ^ ((schAt (xIOf x) 0).levels[l]!).cap := by
    intro hl0
    have hcs := laterCap_size hx' hd hr hss (hlq ▸ hs') hl0 hrs
    unfold top0
    rw [top0_eq hS hr hrs hi hlk hcs, Option.getD_some]
  have hidx : outRead (top0 lk l) l (sh.sched.levels[l]!).q HH (r0.info.lg.conss[l]!).pos
      (r1.info.lg.conss[l]!).pos (q.rep * qAt (xIOf x) q.level + q.query) =
      capIdx (xIOf x) q (Rows.posAt (xIOf x) ((List.range (Bd.DE.V.k (xIOf x))).map c) q) HH := by
    rw [← hpos, hqa]
    unfold outRead capIdx capFirst
    rw [hlq]
    obtain h | h : q.rep = 0 ∨ q.rep = 1 := by omega
    · rw [h]
      simp [hqq]
    · rw [h]
      have hge : ¬ (1 * (sh.sched.levels[l]!).q + q.query < (sh.sched.levels[l]!).q) := by omega
      rw [if_neg hge, Nat.one_mul, Nat.add_sub_cancel_left]
      by_cases hl0 : l = 0
      · simp [hl0]
      · simp [hl0, htop0 hl0]
  rw [hidx] at htr
  have hlink : cc0 = cc' := by
    refine topRow_link (fun hl0 => ?_) (fun hl0 rep hrep' => ?_) htr (hlq ▸ htr')
    · exact root_link _ hS hS' (by omega) hA hA' hi hlk hi' hlk'
    · have hcs := laterCap_size hx' hd hr hss (hlq ▸ hs') hl0 hrs
      have hg : l - 1 < (RecVStar.topSizes rs).length := (List.getElem?_eq_some_iff.1 hcs).1
      have := tops_link _ hS hS' hr hrs hK' hA hA' hi hlk hi' hlk' hrep' hg
      rwa [getD_getElem?, getD_getElem?] at this
  rw [hb, hb', hlink]

/-- **`OutBindRP` at the layout of record**, from `OutCommitTie` there (`outCommitTie_deployed`) and the instance
bound (`outBindRP_of_tie`). -/
theorem outBindRP_deployed {b : Brings}
    {elem : (r : XR) → List (Fin r.w.parts.length × List (Msg × List HmNets.Lin))} {IV : Claims.Verifier Coin}
    {shape : Claims.Recursion.Form → Prop}
    {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}
    (pieces : Msg → List OpenPiece) (hP : ∀ x, (OpenDirs.DR b elem IV).parse x = parse x)
    (hSh : ∀ f, shape f → Claims.Recursion.Form.Dense f) (hR : PiecesInRangeR pieces)
    (hB : ∀ x, RestRead.kI x ≤ Bd.DE.V.k (xIOf x)) :
    OutBindRP FlockDeploy.sha512 (OpenDirs.DR b elem IV) shape recRead b (cmtTr recRead) (DeployedR.layR elem) Bd
      pieces Rows.posAt :=
  outBindRP_of_tie (D := OpenDirs.DR b elem IV) (shape := shape) (untr := recRead) (b := b) (cmt := cmtTr recRead)
    (Bd := Bd) elem pieces Rows.posAt hP hSh hR (outCommitTie_deployed pieces hP hSh hR hB)

end RecDeploy.CapLower
