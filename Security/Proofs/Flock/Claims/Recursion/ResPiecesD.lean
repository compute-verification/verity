import Proofs.Flock.Claims.Recursion.VerdictRowRLeaf
import Proofs.Flock.Claims.Recursion.VerdictRowRDeployed
import Proofs.Flock.Claims.Recursion.PartsVStar
import Proofs.Flock.Claims.Recursion.VStarLinks
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed
import Proofs.Flock.VBridge.Family.Rep
import Proofs.Flock.VBridge.Family.Structure

/-!
# The concrete `rec-residuals` pieces and their owners (`resPD`, `ownRD`)

Row R's leaf asks `resP x` to hold, at every decision `At` admits, a piece of part `p`'s `S` keyed by group `p` at a
copy `(o, g)` with `o + g = r`, for each rep `r` and group `p` (`ResPiecesAt`), and every piece and `rec-open` query to
have an owner (`PiecesOwned`). Both hold at every statement `readX` reads, with no hypothesis.

* `shI spec sts`: the inner shape at the setup's first table: V\*'s (`RecResiduals.shape?`) at its `m`, `k_log` and
  two claims and one per region and point, plain.
* `shI_tie`: at a dense compiled statement, `shI` at its inner statement's setup is the verifier's own inner shape
  (`r.shape`, `--rec-inner`) at `RecResiduals.shape?`. The setup holds that shape's `m`, `k_log` and claims to the inner
  statement's files (`checkShape_files`), which Flock's plain verifier sets up as one table (`innerSetup_files`).
* `DecAt.algSh_tie`: at a dense decision, table 0's decode shape is `shI` in every field V\*'s algebra reads.
* `listed_of_statements`: V\*'s list holds `.inner p (unitName … p) _` for each group `p` of the verifier's algebra
  (`RecVStar.statements` refuses unless V\* has as many parts as groups), and `.recOpen ℓ …` for each level `ℓ` of the
  shape's `fast100` schedule.
* `part_of_readX`: each group's part builds at the read shape. Part `p`'s circuit is named as its statement and carries
  its META `holo`, and its part check passes (`readX_partNamed`: each part's session's `RecVStar.checkCircuit`), which
  builds the verifier's part at the group its name gives.
* `resPD x`: one piece per group `p` of `shI` and rep `r`, keyed `(p, r, 0, 0)` (instance `r`, `ResLower.keyCopy`), at
  part `p`'s `S`. `ownRD x q`: the first of V\*'s statements at `x` that is part `q.key.1` (`isResOf`).
* `resPiecesAt_D`, `openOwned_D`, `piecesOwned_D`: `ResPiecesAt` and `PiecesOwned` at `resPD`, `ownRD` and
  `OpenLower.ownQR`.
-/

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf tabsOf)
open Flock.RecAlgebra (structureOf)
open Flock.RecResiduals (portSizes groups)
open RecDeploy.Rows (DecAt algShape structureOf_alg portSizes_alg part_alg)
open RecDeploy.RestRead (DenseX)
open RecDeploy.OpenLower (ownQR isLevel)

/-! ## The inner shape at the setup -/

/-- The inner shape at the setup's first table: `RecResiduals.shape?` at its `m`, `k_log` and two claims and one per
region and point, plain. -/
def shI (spec : SessionSpec) (sts : Array Stmt) : Shape :=
  { (RecResiduals.shape? sts[0]!.m sts[0]!.c.kLog (2 + spec.points * sts[0]!.regions.size)).getD default with
    holo := false }

theorem getElem!_one (st : Stmt) : (#[st] : Array Stmt)[0]! = st :=
  (getElem!_pos (#[st] : Array Stmt) 0 (by simp)).trans rfl

/-- **The read shape is the setup's**: at a dense compiled statement whose inner statement sets up, the setup is one
table `st` at the verifier's own inner shape's `m`, and `shI` there is that shape at `RecResiduals.shape?`, plain. -/
theorem shI_tie {x : Msg} (hX : DenseX x) {v : FlockDeploy.Ins × SessionSpec × Array Stmt}
    (hv : setupOf false (xIOf x) = some v) :
    ∃ r ss sh0, readX x = some r ∧ RecVStar.statements r.shape = .ok ss ∧ r.shape.holo = false ∧
      RecResiduals.shape? r.shape.m r.shape.kLog r.shape.claims = some sh0 ∧
      shI v.2.1 v.2.2 = { sh0 with holo := r.shape.holo } ∧ ∃ st, v.2.2 = #[st] ∧ st.m = r.shape.m := by
  obtain ⟨sto, hst, hf⟩ := hX
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpt, -, -⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, e2, -, e4, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  have hx : xIOf x = r.w.inner := by unfold xIOf; rw [hst]; rfl
  rw [hx, hset] at hv
  cases hv
  obtain ⟨sh0, -, -, hsh0, -, -⟩ := Rows.groups_of_statements hss
  have hmm : st.m = r.shape.m := hm.trans e1.symm
  have hkk : st.c.kLog = r.shape.kLog := hk.trans e2.symm
  have hcc : 2 + spec.points * st.regions.size = r.shape.claims := by rw [e4, hreg, hpt]
  refine ⟨r, ss, sh0, hr, hss, hh, hsh0, ?_, st, rfl, hmm⟩
  simp only [shI, getElem!_one, hmm, hkk, hcc, hsh0, Option.getD_some, hh]

namespace DecAt

variable {rd : Msg → Option FinalP} {x : Msg} {ms : List Msg} {cs : List Coin} (D : DecAt rd x ms cs)
  (cls : FlockSoundness.Discharge.Recursion.HClass)

/-- **Table 0's decode shape is the setup's inner shape** (`shI`) in every field V\*'s algebra reads, at a dense
decision (`Rows.DecAt.sh_alg`, at the setup's numbers). -/
theorem algSh_tie {xo : Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (h0 : 0 < D.sts.size)
    {v : FlockDeploy.Ins × SessionSpec × Array Stmt} (hv : setupOf false x = some v) :
    algShape (D.sh cls 0) = shI v.2.1 v.2.2 := by
  have hG := D.rowG cls (fun _ => #[]) hst hf hx h0
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, hk, hreg, hpt, -, -⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, e2, -, e4, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  obtain ⟨sh0, -, -, hsh0, -, -⟩ := Rows.groups_of_statements hss
  obtain ⟨sch0, hf0, rfl⟩ := Rows.shape?_some hsh0
  have hx' : x = r.w.inner := by rw [hx]; unfold xIOf; rw [hst]; rfl
  rw [← hx'] at hset
  rw [hset] at hv
  cases hv
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  obtain ⟨v', c₁, r₀, rss, fl, link, pf, hv', -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hset] at hv'
  cases hv'
  have hdspec : D.d.spec = spec := congrArg (·.spec) hd
  have hdsts : D.d.sts = #[st] := congrArg (·.sts) hd
  have hsts1 : D.sts = #[st] := hsts.trans hdsts
  have htab : (tabsOf D.sts)[0]! = Setup.ofCircuit st := by
    rw [D.tab_eq h0]
    have h' : ∀ (a : Array Stmt) (h : 0 < a.size), a = #[st] → Setup.ofCircuit a[0] = Setup.ofCircuit st := by
      intro a h ha
      subst ha
      rfl
    exact h' D.sts h0 hsts1
  have hext : (Setup.ofCircuit st).extra D.sess.points = .ok (D.files cls 0).extras := by
    rw [← htab]
    exact hG.extra
  have hesz : (D.files cls 0).extras.size = st.regions.size * spec.points := by
    rw [Rows.extras_size (s := st) hext, D.points, hsp, hdspec]
    simp [FlockDeploy.pointsOf]
  have e : D.sh cls 0 = FlockSoundness.Refine.shapeOf (Setup.ofCircuit st) ((fast100 st.m).getD default)
      (D.files cls 0).extras.size := by
    show FlockSoundness.Refine.shapeOf (tabsOf D.sts)[0]! ((fast100 (tabsOf D.sts)[0]!.m).getD default) _ = _
    rw [htab]; rfl
  have hmm : st.m = r.shape.m := by rw [hm, e1]
  have hkk : st.c.kLog = r.shape.kLog := by rw [hk, e2]
  have hcc : 2 + (D.files cls 0).extras.size = r.shape.claims := by
    rw [hesz, e4, hreg, hpt, Nat.mul_comm]
  have hcc' : 2 + spec.points * st.regions.size = r.shape.claims := by rw [e4, hreg, hpt]
  have hsc : (fast100 st.m).getD default = sch0 := by rw [hmm, hf0]; rfl
  have hI : shI spec #[st] = { ({ m := r.shape.m, kLog := r.shape.kLog, claims := r.shape.claims, sched := sch0 } :
      Shape) with holo := false } := by
    simp only [shI, getElem!_one, hmm, hkk, hcc', hsh0, Option.getD_some]
  show algShape (D.sh cls 0) = shI spec #[st]
  rw [hI, e]
  unfold algShape FlockSoundness.Refine.shapeOf
  dsimp only
  congr 1
  all_goals first | exact hmm | exact hkk | exact hcc | exact hsc | rfl

end DecAt

/-! ## V\*'s list -/

/-- **V\*'s list holds each part and each level**: where `RecVStar.statements` returns at the inner shape `i`, it has
computed the verifier's groups at `RecResiduals.shape?` (`RecResiduals.lean:119-139`), V\* has as many parts as groups,
and the list holds part `p`'s statement `rec-residuals/m…-p<p>` for each group `p` and level `ℓ`'s `rec-open` statement
for each level of the shape's `fast100` schedule (`statementsOf_eq`). -/
theorem listed_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ sh0 S gs sc, RecResiduals.shape? i.m i.kLog i.claims = some sh0 ∧
      structureOf { sh0 with holo := i.holo } = .ok S ∧ groups S (portSizes { sh0 with holo := i.holo }) = .ok gs ∧
      fast100 i.m = some sc ∧
      (∀ p < gs.size, ∃ pt, VStar.Stmt.inner p (RecResiduals.unitName i.m i.kLog i.claims p) pt ∈ ss) ∧
      ∀ l < sc.levels.size, ∃ a b n, VStar.Stmt.recOpen l a b n ∈ ss := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i sh0 hsh
  split at h
  · cases h
  obtain ⟨ps, hps, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, hgs, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  rename_i hsz
  split at h
  · cases h
  obtain ⟨K, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨t, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨A, hA, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨C, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨K0, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u', -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨u'', -, h⟩ := FlockC3.VStarList.eok_bind h
  cases h
  obtain ⟨sp, hsp, hpp⟩ := FlockC3.VStarList.eok_bind (Rows.ok_of_mapError' hps)
  obtain ⟨S, hS, hg⟩ := FlockC3.VStarList.eok_bind (Rows.ok_of_mapError' hgs)
  obtain ⟨sc, sp', ps', hsc, hsp', hpp', hAe⟩ := FlockVBridge.Family.statementsOf_eq hA
  rw [hsp] at hsp'
  cases hsp'
  rw [hpp] at hpp'
  cases hpp'
  have hsz' : ps.size = gs.size := by simpa using hsz
  refine ⟨sh0, S, gs, sc, hsh, hS, hg, FlockVBridge.Family.sched_fast100 hsc, fun p hp => ⟨ps[p]!, ?_⟩,
    fun l hl => ⟨2 ^ (sc.levels[l]!).lanes, (sc.levels[l]!).d - (sc.levels[l]!).cap, 2 * (sc.levels[l]!).q, ?_⟩⟩
  · have hmem : VStar.Stmt.inner p (RecResiduals.unitName i.m i.kLog i.claims p) ps[p]! ∈ A := by
      rw [hAe]
      exact Array.mem_append.2 (Or.inr (List.mem_toArray.2 (List.mem_map.2 ⟨p, List.mem_range.2 (by omega), rfl⟩)))
    simp [hmem]
  · have hmem : VStar.Stmt.recOpen l (2 ^ (sc.levels[l]!).lanes) ((sc.levels[l]!).d - (sc.levels[l]!).cap)
        (2 * (sc.levels[l]!).q) ∈ A := by
      rw [hAe]
      exact Array.mem_append.2 (Or.inl (List.mem_toArray.2 (List.mem_map.2 ⟨l, List.mem_range.2 hl, rfl⟩)))
    simp [hmem]

/-- **Each group's part builds at the read shape**: part `p`'s statement is in V\*'s list (`listed_of_statements`), its
circuit is named as it and carries its META `holo`, and its part check passes (`readX_partNamed`); the part check
builds the verifier's part at the group the name gives (`checkPart_inner`), which is `p` (`resUnitName_inj`). -/
theorem part_of_readX {x : Msg} {r : XR} (hr : readX x = some r) {ss : Array VStar.Stmt}
    (hss : RecVStar.statements r.shape = .ok ss) {sh0 : Shape} {S : RecAlgebra.GfS}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hsh0 : RecResiduals.shape? r.shape.m r.shape.kLog r.shape.claims = some sh0)
    (hS : structureOf { sh0 with holo := r.shape.holo } = .ok S)
    (hg : groups S (portSizes { sh0 with holo := r.shape.holo }) = .ok gs) {p : ℕ} (hp : p < gs.size) :
    ∃ S' ns nv, RecResiduals.part { sh0 with holo := r.shape.holo } p = .ok (S', ns, nv) := by
  obtain ⟨sh1, S1, gs1, -, hsh1, hS1, hg1, -, hlist, -⟩ := listed_of_statements hss
  rw [hsh0] at hsh1
  cases hsh1
  rw [hS] at hS1
  cases hS1
  rw [hg] at hg1
  cases hg1
  obtain ⟨pt, hmem⟩ := hlist p hp
  obtain ⟨vs, hvs, hall⟩ := readX_partNamed hr
  rw [hss] at hvs
  cases hvs
  obtain ⟨s, hs, hget⟩ := Array.mem_iff_getElem.1 hmem
  obtain ⟨prt, c, -, -, hpart, hname, hholo⟩ := hall s hs
  rw [hget] at hpart hname hholo
  obtain ⟨m, k, cl, jj, sh, holo, S', ns, nv, hpar, hshp, hho, hpt⟩ := Flock.Readable.checkPart_inner hpart
  have hn : RecResiduals.unitName m k cl jj = RecResiduals.unitName r.shape.m r.shape.kLog r.shape.claims p :=
    (FlockVBridge.Family.resUnitName_params hpar).trans hname
  obtain ⟨rfl, rfl, rfl, rfl⟩ := FlockVBridge.Family.resUnitName_inj hn
  rw [hsh0] at hshp
  cases hshp
  have hh := FlockSoundness.Discharge.Recursion.checkHolo_ok hholo
  rw [hho] at hh
  have hh' : holo = r.shape.holo := by
    have := Except.ok.inj hh
    simpa [RecVStar.partHolo] using this
  subst hh'
  exact ⟨S', ns, nv, hpt⟩

/-! ## The pieces and their owners -/

/-- The verifier's algebra's groups at shape `sh`, counted; `0` where it doesn't group. -/
def groupsN (sh : Shape) : ℕ :=
  match structureOf sh with
  | .ok S =>
    match groups S (portSizes sh) with
    | .ok gs => gs.size
    | .error _ => 0
  | .error _ => 0

/-- Group `p`'s pieces at shape `sh`: one per rep `r`, keyed `(p, r, 0, 0)`, at part `p`'s `S`; none where the part
doesn't build. -/
def resGroup (sh : Shape) (p : ℕ) : List ResPiece :=
  match RecResiduals.part sh p with
  | .ok (S, _, _) => [⟨(p, 0, 0, 0), S⟩, ⟨(p, 1, 0, 0), S⟩]
  | .error _ => []

/-- **The `rec-residuals` pieces** at the inner statement `x`: each group's (`resGroup`) at the setup's inner shape
(`shI`). -/
def resPD (x : Msg) : List ResPiece :=
  match setupOf false x with
  | some v => (List.range (groupsN (shI v.2.1 v.2.2))).flatMap (resGroup (shI v.2.1 v.2.2))
  | none => []

/-- A V\* statement is the algebra's part `jj`. -/
def isResOf (jj : ℕ) : VStar.Stmt → Bool
  | .inner j _ _ => j == jj
  | _ => false

/-- **A piece's owner**: the first of V\*'s statements at the compiled statement `x` that is the algebra's part of the
piece's group. -/
def ownRD (x : Msg) (q : ResPiece) : Option ℕ :=
  (readX x).bind fun r => (_root_.Flock.RecVStar.statements r.shape).toOption.bind fun ss =>
    ss.findIdx? (isResOf q.key.1)

theorem groupsN_eq {sh : Shape} {S : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : structureOf sh = .ok S) (hg : groups S (portSizes sh) = .ok gs) : groupsN sh = gs.size := by
  simp [groupsN, hS, hg]

theorem mem_resPD {x : Msg} {q : ResPiece} (hq : q ∈ resPD x) :
    ∃ v, setupOf false x = some v ∧ q.key.1 < groupsN (shI v.2.1 v.2.2) := by
  unfold resPD at hq
  cases hv : setupOf false x with
  | none => rw [hv] at hq; cases hq
  | some v =>
    rw [hv] at hq
    obtain ⟨p, hp, hq⟩ := List.mem_flatMap.1 hq
    refine ⟨v, rfl, ?_⟩
    unfold resGroup at hq
    split at hq
    · rcases List.mem_cons.1 hq with rfl | hq
      · exact List.mem_range.1 hp
      · rw [List.mem_singleton] at hq
        subst hq
        exact List.mem_range.1 hp
    · cases hq

/-- **`ResPiecesAt` at the concrete pieces**: at a dense decision, table 0's shape is `shI` in every field the algebra
reads (`algSh_tie`), which is the read shape (`shI_tie`), where each group's part builds (`part_of_readX`); `resPD`
holds each group's two pieces there. -/
theorem resPiecesAt_D (rd : Msg → Option FinalP) (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ)
    (side : Msg → List Msg → OpenPiece → OpenSide) (posAt : Msg → List Coin → OpenPiece → ℕ) :
    ResPiecesAt rd resPD key side posAt := by
  intro x ms cs hA D cls i hi S gs hS hg r p hp
  obtain ⟨⟨xo, sto, hst, hf, hx⟩, -⟩ := id hA
  obtain ⟨hT1, -⟩ := D.tables_of_dense hst hf hx
  obtain rfl : i = 0 := by omega
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, -, -⟩ := Rows.decR_last D.dec
  have he := DecAt.algSh_tie D cls hst hf hx hi hv
  rw [← structureOf_alg, he] at hS
  rw [← portSizes_alg, he] at hg
  obtain ⟨r', ss, sh0, hr, hss, -, hsh0, hI, -⟩ := shI_tie ⟨sto, hst, hf⟩ (by rw [← hx]; exact hv)
  rw [hI] at hS hg
  obtain ⟨S', ns, nv, hpart⟩ := part_of_readX hr hss hsh0 hS hg hp
  have hN : groupsN (shI v.2.1 v.2.2) = gs.size := by rw [hI]; exact groupsN_eq hS hg
  refine ⟨⟨(p, r.val, 0, 0), S'⟩, ?_, ⟨ns, nv, ?_⟩, rfl, by simp⟩
  · unfold resPD
    rw [hv]
    refine List.mem_flatMap.2 ⟨p, List.mem_range.2 (by omega), ?_⟩
    unfold resGroup
    rw [hI, hpart]
    rcases r with ⟨_ | _ | n, hr⟩
    · exact List.mem_cons_self
    · exact List.mem_cons_of_mem _ List.mem_cons_self
    · omega
  · rw [← part_alg, he, hI]
    exact hpart

/-- **Every `rec-open` query has an owner**: a query of the dense inner statement's one table at level `ℓ` of its
`fast100` schedule, which is the read shape's (`shI_tie`), goes to V\*'s `.recOpen ℓ …` (`listed_of_statements`),
below the statements' count, which is the parts' (`readXPre_spec`). -/
theorem openOwned_D (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) (x : Msg) (hX : DenseX x) :
    ∀ q ∈ Rows.rowDPieces key (xIOf x), ∃ s < nS x, ownQR x q = some s := by
  intro q hq
  unfold Rows.rowDPieces at hq
  cases hv : setupOf false (xIOf x) with
  | none => rw [hv] at hq; cases hq
  | some v =>
    rw [hv] at hq
    obtain ⟨ht, -, hl, -, -⟩ := Rows.mem_rowDPiecesOf.1 hq
    obtain ⟨r, ss, sh0, hr, hss, -, -, -, st, hvs, hstm⟩ := shI_tie hX hv
    rw [hvs] at ht hl
    have h0 : q.table = 0 := by simp at ht; omega
    rw [h0] at hl
    obtain ⟨-, -, -, sc, -, -, -, hfast, -, hlev⟩ := listed_of_statements hss
    have htab : (tabsOf #[st])[0]! = Setup.ofCircuit st := Rows.tabsOf_get (sts := #[st]) (j := 0) (by simp)
    have hsch : Rows.schAt #[st] 0 = sc := by
      unfold Rows.schAt
      rw [htab]
      show (fast100 st.m).getD default = sc
      rw [hstm, hfast]
      rfl
    rw [hsch] at hl
    obtain ⟨a, b, n, hmem⟩ := hlev q.level hl
    have hsome : (ss.findIdx? (isLevel q.level)).isSome = true := by
      rw [Array.findIdx?_isSome]
      exact Array.any_eq_true'.2 ⟨_, hmem, by simp [isLevel]⟩
    obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 hsome
    obtain ⟨hsz, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs
    obtain ⟨-, vs, hvs', hlen⟩ := readXPre_spec (readXPre_of_readX hr)
    rw [hss] at hvs'
    cases hvs'
    refine ⟨s, ?_, ?_⟩
    · simp only [nS, hr]
      omega
    · simp [ownQR, h0, hr, hss, Except.toOption, hs]

/-- **`PiecesOwned` at the concrete pieces and owners**: each piece's group is one of the read shape's (`shI_tie`), so
V\*'s list holds its part (`listed_of_statements`), the first of which `ownRD` finds, below the statements' count, which
is the parts' (`readXPre_spec`); and each `rec-open` query has its level's statement (`openOwned_D`). -/
theorem piecesOwned_D (key : ℕ → ℕ → ℕ → ℕ → ℕ × ℕ × ℕ × ℕ) : PiecesOwned resPD key ownRD := by
  intro x hX
  refine ⟨fun q hq => ?_, openOwned_D key x hX⟩
  obtain ⟨v, hv, hp⟩ := mem_resPD hq
  obtain ⟨r, ss, sh0, hr, hss, -, hsh0, hI, -⟩ := shI_tie hX hv
  obtain ⟨sh1, S, gs, -, hsh1, hS, hg, -, hlist, -⟩ := listed_of_statements hss
  rw [hsh0] at hsh1
  cases hsh1
  have hN : groupsN (shI v.2.1 v.2.2) = gs.size := by rw [hI]; exact groupsN_eq hS hg
  obtain ⟨pt, hmem⟩ := hlist q.key.1 (by omega)
  have hsome : (ss.findIdx? (isResOf q.key.1)).isSome = true := by
    rw [Array.findIdx?_isSome]
    exact Array.any_eq_true'.2 ⟨_, hmem, by simp [isResOf]⟩
  obtain ⟨s, hs⟩ := Option.isSome_iff_exists.1 hsome
  obtain ⟨hsz, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hs
  obtain ⟨-, vs, hvs, hlen⟩ := readXPre_spec (readXPre_of_readX hr)
  rw [hss] at hvs
  cases hvs
  refine ⟨s, ?_, ?_⟩
  · simp only [nS, hr]
    omega
  · simp [ownRD, hr, hss, Except.toOption, hs]

end RecDeploy.DecompBR
