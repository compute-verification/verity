import Proofs.Flock.Claims.Recursion.VerdictAssembly
import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.C3.VStarList.Shape

/-!
# Row R at the deployed reader

`DecAt.runs_assembled` (`VerdictAssembly`) takes row R as S and its groups (`hS`, `hg`), the rep's coin rounds' sizes
(`hsz`), `w`'s size (`hw`) and every part's residuals vanishing (`hparts`, `PartsZero`). At a decision of an inner
statement the reader reads at the dense form:

* **S and its groups** (`DecAt.structure_ok`): the reader's scope check builds V\*'s list (`RecVStar.statements`), which
  computes the verifier's groups at the inner shape (`groups_of_statements`). That shape is table `i`'s decode shape
  (`DecAt.sh`) in every field V\*'s algebra reads (`DecAt.sh_alg`): `m`, `k_log` and the claims are the files'
  (`checkShape_files`, `innerSetup_files`; two claims a region at the session's two points, `extras_size`), the schedule
  `fast100`'s. The digest and salt widths, which `Refine.shapeOf` sets and `RecResiduals.shape?` leaves at their
  defaults, are read by neither `structureOf` nor `portSizes` (`structureOf_alg`, `portSizes_alg`, `part_alg`).
* **`w`'s size** (`DecAt.w_size`, `DecAt.hw_of_acc`): `w` is the frames' values (`chunks_vals`), one per element of the
  rounds (`vals_repFrames_rounds`), then the rep's accumulators, and S's `nw` is the rounds' elements and the four
  running sums (`structureOf_nw`). So `hw` is `(acc k).size = 4`.
* **The parts' residuals** (`DecAt.partsZero_of_pieces`): from the `rec-residuals` pieces as `resPieces_computes` gives
  them (`resHolds elemAt vvAt`), with one piece per (rep, group) at the copy `keyAt r p` (`hcover`), the copy's
  elements being `w` at the group's row ports (`hE`, the layout's tie) and its verifier values being `verifierRows`' `v`
  at the group's `V` (`hvv`, check 5's tie at the verifier's values, the vcalls tie).
* **`runs_assembled_R`**: `runs_assembled` with `hS`, `hg` and `hw` discharged, `hparts` as the pieces and their ties,
  and `hsz` kept: the record's rounds are the prover's JSON (`jAt` substitutes only coins and digests), so their
  sizes need the canonical record or the coin tie.
-/

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf pointsOf vals repCtx)
open FlockVBridge.Family (FramedAt ProofShaped roundSizes)
open FlockVBridge.AccChain (UnitSums ReadsEnds)
open Flock.RecAlgebra (verifierRows valOf NoEmpty structureOf el wNames accNames rounds)
open Flock.RecResiduals (portSizes groups)
open FlockVBridge.PartS (wIdx)
open FlockSoundness.Discharge.Exec (repFrames)
open FlockSoundness.Discharge.Recursion (HClass)
open FlockSoundness.Discharge.DecodeZL (ProofSizes decodeProof_sizes)

/-! ## The fields V\*'s algebra reads -/

/-- **A shape's algebra fields**: `m`, `k_log`, the claims, the schedule and `holo`, with the digest and salt widths at
their defaults. -/
def algShape (sh : Shape) : Shape :=
  { m := sh.m, kLog := sh.kLog, claims := sh.claims, sched := sh.sched, holo := sh.holo }

theorem wNames_alg (sh : Shape) : wNames (algShape sh) = wNames sh := by
  cases sh
  rfl

theorem wi_alg (sh : Shape) : RecAlgebra.wi (algShape sh) = RecAlgebra.wi sh := by
  funext k
  simp only [RecAlgebra.wi, wNames_alg]

theorem orderOf_alg (sh : Shape) : RecAlgebra.orderOf (algShape sh) = RecAlgebra.orderOf sh := by
  funext k
  cases sh
  rfl

theorem varOp_alg (sh : Shape) : RecAlgebra.varOp (algShape sh) = RecAlgebra.varOp sh := by
  funext c k
  cases sh
  rfl

theorem accTerms_alg (sh : Shape) : RecAlgebra.accTerms (algShape sh) = RecAlgebra.accTerms sh := by
  funext fin ri l
  cases sh
  rfl

theorem coefOps_alg (sh : Shape) : RecAlgebra.coefOps (algShape sh) = RecAlgebra.coefOps sh := by
  funext c ri l
  induction l with
  | nil => rfl
  | cons h rest ih =>
    obtain ⟨k, x⟩ := h
    simp only [RecAlgebra.coefOps, varOp_alg, ih]

theorem prodOps_alg (sh : Shape) : RecAlgebra.prodOps (algShape sh) = RecAlgebra.prodOps sh := by
  funext c ri l
  induction l with
  | nil => rfl
  | cons h rest ih =>
    obtain ⟨⟨a, b⟩, x⟩ := h
    simp only [RecAlgebra.prodOps, varOp_alg, ih]

theorem buildRes_alg (sh : Shape) : RecAlgebra.buildRes (algShape sh) = RecAlgebra.buildRes sh := by
  funext fin c ri f
  simp only [RecAlgebra.buildRes, accTerms_alg, coefOps_alg, prodOps_alg, orderOf_alg]

theorem buildAll_alg (sh : Shape) : RecAlgebra.buildAll (algShape sh) = RecAlgebra.buildAll sh := by
  funext fin c l ri
  induction l generalizing ri with
  | nil => rfl
  | cons f rest ih => simp only [RecAlgebra.buildAll, buildRes_alg, ih]

theorem build_alg (sh : Shape) : RecAlgebra.build (algShape sh) = RecAlgebra.build sh := by
  funext lins
  simp only [RecAlgebra.build, buildAll_alg, wNames_alg]
  rfl

theorem residualLins_alg (sh : Shape) : RecAlgebra.residualLins (algShape sh) = RecAlgebra.residualLins sh := by
  funext c e
  cases sh
  rfl

theorem structureOf_alg (sh : Shape) : structureOf (algShape sh) = structureOf sh := by
  have he : RecAlgebra.extrasLike (algShape sh) = RecAlgebra.extrasLike sh := rfl
  simp only [RecAlgebra.structureOf, residualLins_alg, he, build_alg]

theorem portSizes_alg (sh : Shape) : portSizes (algShape sh) = portSizes sh := by
  cases sh
  rfl

theorem part_alg (sh : Shape) (j : ℕ) : RecResiduals.part (algShape sh) j = RecResiduals.part sh j := by
  simp only [RecResiduals.part, structureOf_alg, portSizes_alg]

/-- **S's `nw` is w's names**: the rounds' elements, then the four running sums. -/
theorem structureOf_nw {sh : Shape} {S : RecAlgebra.GfS} (h : structureOf sh = .ok S) :
    S.nw = (rounds sh).flatten.size + 4 := by
  unfold RecAlgebra.structureOf at h
  obtain ⟨⟨lins, info⟩, -, h⟩ := FlockC3.VStarList.eok_bind h
  dsimp only at h
  obtain ⟨b, hb, h⟩ := FlockC3.VStarList.eok_bind h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  unfold RecAlgebra.build at hb
  obtain ⟨⟨res, st⟩, -, hb⟩ := FlockC3.VStarList.eok_bind hb
  dsimp only at hb
  split at hb
  · cases hb
  · simp only [pure, Except.pure, Except.ok.injEq] at hb
    subst hb
    simp [wNames, accNames]

/-! ## V\*'s list computes the groups -/

theorem ok_of_mapError' {ε A : Type} {e : Except ε A} {g : ε → ε} {a : A} (h : e.mapError g = .ok a) :
    e = .ok a := by
  cases e with
  | error _ => simp [Except.mapError] at h
  | ok _ => simpa [Except.mapError] using h

/-- `RecResiduals.shape?`'s shape: `fast100`'s schedule, the other fields at their defaults. -/
theorem shape?_some {m k c : ℕ} {sh : Shape} (h : RecResiduals.shape? m k c = some sh) :
    ∃ s, fast100 m = some s ∧ sh = { m := m, kLog := k, claims := c, sched := s } := by
  unfold RecResiduals.shape? at h
  cases hf : fast100 m with
  | none => rw [hf] at h; cases h
  | some s =>
    rw [hf] at h
    simp at h
    obtain ⟨-, rfl⟩ := h
    exact ⟨s, rfl, rfl⟩

/-- **V\*'s list holds the verifier's groups**: `RecVStar.statements` returns only where the inner shape is
`RecResiduals.shape?`'s and its algebra's S and groups are computed (`RecResiduals.lean:119-139`). -/
theorem groups_of_statements {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) :
    ∃ sh0 S gs, RecResiduals.shape? i.m i.kLog i.claims = some sh0 ∧
      structureOf { sh0 with holo := i.holo } = .ok S ∧
      groups S (portSizes { sh0 with holo := i.holo }) = .ok gs := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i sh0 hsh
  split at h
  · cases h
  obtain ⟨ps, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, hgs, -⟩ := FlockC3.VStarList.eok_bind h
  have hgs' := ok_of_mapError' hgs
  obtain ⟨S, hS, hg⟩ := FlockC3.VStarList.eok_bind hgs'
  exact ⟨sh0, S, gs, hsh, hS, hg⟩

/-- **The extra claims number the regions times the points.** -/
theorem extras_size {s : Stmt} {pts : Array (Array F128)} {e : Array Claim} (h : s.extraClaims pts = .ok e) :
    e.size = s.regions.size * pts.size := by
  obtain ⟨ess, hess, hfl⟩ := FlockSoundness.Refine.extraClaims_ok s pts e h
  have hl : ∀ es ∈ ess, es.length = pts.size := by
    intro es hes
    obtain ⟨reg, hreg⟩ := forall₂_right hess es hes
    simpa using hreg.length_eq.symm
  have hlen : ess.length = s.regions.size := by simpa using hess.length_eq.symm
  have hm : ess.map List.length = ess.map (fun _ => pts.size) := List.map_congr_left hl
  rw [← Array.length_toList, hfl, List.length_flatten, hm, List.map_const', List.sum_replicate, hlen, smul_eq_mul]

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs) (cls : HClass)

/-- **Table `i`'s decode shape is V\*'s inner shape** in every field V\*'s algebra reads, at a decision of an inner
statement the reader reads at the dense form, where the session's points are the link's. -/
theorem sh_alg {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁) :
    ∃ (inn : RecVStar.Inner) (ss : Array VStar.Stmt) (sh0 : Shape), RecVStar.statements inn = .ok ss ∧
      RecResiduals.shape? inn.m inn.kLog inn.claims = some sh0 ∧
      algShape (D.sh cls i) = { sh0 with holo := inn.holo } := by
  have hG := D.rowG cls (fun _ => #[]) hst hf hx hi
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
  obtain ⟨sh0, -, -, hsh0, -, -⟩ := groups_of_statements hss
  obtain ⟨sch0, hf0, rfl⟩ := shape?_some hsh0
  have hx' : x = r.w.inner := by rw [hx]; unfold xIOf; rw [hst]; rfl
  rw [← hx'] at hset
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hset] at hv
  cases hv
  have hdspec : D.d.spec = spec := congrArg (·.spec) hd
  have hdsts : D.d.sts = #[st] := congrArg (·.sts) hd
  have hsts1 : D.sts = #[st] := hsts.trans hdsts
  have hi0 : i = 0 := by
    have : i < 1 := by simpa [hsts1] using hi
    omega
  subst hi0
  have htab : (tabsOf D.sts)[0]! = Setup.ofCircuit st := by
    rw [D.tab_eq hi]
    have h' : ∀ (a : Array Stmt) (h : 0 < a.size), a = #[st] → Setup.ofCircuit a[0] = Setup.ofCircuit st := by
      intro a h ha
      subst ha
      rfl
    exact h' D.sts hi hsts1
  have hext : (Setup.ofCircuit st).extra D.sess.points = .ok (D.files cls 0).extras := by
    rw [← htab]
    exact hG.extra
  have hesz : (D.files cls 0).extras.size = st.regions.size * spec.points := by
    rw [extras_size (s := st) hext, hpts, hsp, hdspec]
    simp [pointsOf]
  refine ⟨r.shape, ss, _, hss, hsh0, ?_⟩
  have e : D.sh cls 0 = FlockSoundness.Refine.shapeOf (Setup.ofCircuit st) ((fast100 st.m).getD default)
      (D.files cls 0).extras.size := by
    show FlockSoundness.Refine.shapeOf (tabsOf D.sts)[0]! ((fast100 (tabsOf D.sts)[0]!.m).getD default) _ = _
    rw [htab]; rfl
  rw [e]
  have hmm : st.m = r.shape.m := by rw [hm, e1]
  have hkk : st.c.kLog = r.shape.kLog := by rw [hk, e2]
  have hcc : 2 + (D.files cls 0).extras.size = r.shape.claims := by
    rw [hesz, e4, hreg, hpt, Nat.mul_comm]
  have hsc : (fast100 st.m).getD default = sch0 := by rw [hmm, hf0]; rfl
  unfold algShape FlockSoundness.Refine.shapeOf
  dsimp only
  congr 1
  all_goals first | exact hmm | exact hkk | exact hcc | exact hsc | exact hh.symm

/-- **S and its groups at table `i`** (row R's `hS` and `hg`): V\*'s list, which the reader's scope check builds,
computes them at the inner shape, which is table `i`'s decode shape in the fields they read. -/
theorem structure_ok {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size)
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁) :
    ∃ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S ∧ groups S (portSizes (D.sh cls i)) = .ok gs := by
  obtain ⟨inn, ss, sh0, hss, hsh0, he⟩ := D.sh_alg cls hst hf hx hi hpts
  obtain ⟨sh1, S, gs, hsh1, hS, hg⟩ := groups_of_statements hss
  rw [hsh0] at hsh1
  cases hsh1
  refine ⟨S, gs, ?_, ?_⟩
  · rw [← structureOf_alg (D.sh cls i), he]
    exact hS
  · rw [← portSizes_alg (D.sh cls i), he]
    exact hg

/-! ## `w`'s size -/

variable (acc : ℕ → Array F128) (i : ℕ) (r : Fin 2)

/-- Rep `k`'s `w` is its frames' values, then its accumulators. -/
theorem w_toList (k : ℕ) (a : Array F128) :
    (D.w k a).toList =
      vals (repFrames (D.ctx k).st (D.ctx k).sch (D.ctx k).dom (D.ctx k).n D.d.f.ps[k]! (D.T k)) ++ a.toList := by
  rw [DecAt.w, Array.toList_append, List.toList_toArray, FlockDeploy.chunksOf, FlockDecodeTies.chunks_vals]

/-- **Rep `r`'s `w` has one value per element of the rounds, then its accumulators**, where the session's points
are the link's and the rep's proof is shaped for its file. -/
theorem w_size (hG : D.RowG cls acc i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (FlockDeploy.shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r))
      D.d.f.ps[2 * i + r]) :
    (D.rep cls acc i r).w.size = (rounds (D.sh cls i)).flatten.size + (acc (2 * i + r)).size := by
  have hext : (tabsOf D.sts)[i]!.extra (pointsOf D.spec D.d.c₁) = .ok (D.files cls i).extras := by
    rw [← hpts]
    exact hG.extra
  have hp : ProofSizes (D.sh cls i) ((D.files cls i).proof r) :=
    decodeProof_sizes _ _ _ (D.decode_rep cls i r hext hk hsh)
  have hn := D.ctx_n cls acc i r hG hext
  have hs := D.ctx_sch cls acc i r hG
  have hv := FlockDecodeTies.vals_repFrames_rounds (sh := D.sh cls i) (st := (D.ctx (2 * i + r)).st)
    (sch := (D.ctx (2 * i + r)).sch) (n := (D.ctx (2 * i + r)).n) (D.ctx (2 * i + r)).dom D.d.f.ps[2 * i + r]!
    (D.T (2 * i + r)) (by rw [D.ctx_st cls acc i r hG]; rfl)
    (by rw [D.ctx_st cls acc i r hG]; exact hG.foldKLog) hn hs rfl hp.r1ab hp.r1c hp.zPartial
    (fun k hk => FlockDecodeTies.rsEach_get hp (by rw [← hn]; exact hk))
    (by rw [hs]; exact FlockDecodeTies.yr_size hp)
  rw [← Array.length_toList]
  show (D.w (2 * i + r) (acc (2 * i + r))).toList.length = _
  rw [w_toList, List.length_append, hv, List.length_map, Array.length_toList, Array.length_toList]

/-- **Row R's `hw`**: at S, `w` has S's `nw` elements where the rep's accumulators are the four running sums. -/
theorem hw_of_acc (hG : D.RowG cls acc i) (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : 2 * i + r < D.d.f.ps.size)
    (hsh : ProofShaped (FlockDeploy.shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r))
      D.d.f.ps[2 * i + r])
    {S : RecAlgebra.GfS} (hS : structureOf (D.sh cls i) = .ok S) (hacc : (acc (2 * i + r)).size = 4) :
    (D.rep cls acc i r).w.size = S.nw := by
  rw [D.w_size cls acc i r hG hpts hk hsh, hacc, structureOf_nw hS]

/-! ## The parts' residuals from the `rec-residuals` pieces -/

/-- **Row R's `hparts` from the pieces**: where every `rec-residuals` piece holds at the decision (`resHolds`), each
group `p` of S has a piece at the copy `keyAt p` whose group is the verifier's part `p` (`hcover`), that copy's
elements are `w` at the group's row ports (`hE`) and its verifier values `verifierRows`' `v` at the group's `V` (`hvv`),
every part's residuals vanish at rep `r`'s values. -/
theorem partsZero_of_pieces {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (rpieces : List ResPiece)
    (elemAt : Claims.Msg → List Claims.Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → F128)
    (vvAt : Claims.Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → F128)
    (hres : ∀ q ∈ rpieces, resHolds elemAt vvAt q x ms cs) (keyAt : ℕ → ℕ × ℕ × ℕ × ℕ)
    (hcover : ∀ p (_hp : p < gs.size), ∃ q ∈ rpieces, q.key = keyAt p ∧
      ∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv))
    (hE : ∀ p (hp : p < gs.size) k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
      elemAt x ms cs (keyAt p) k = el (D.rep cls acc i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k])
    (hvv : ∀ v info kappa, verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
        (D.rep cls acc i r).comb = .ok (v, info, kappa) →
      ∀ p (hp : p < gs.size) k (hk : k < gs[p].2.2.size), vvAt x cs (keyAt p) k = el v gs[p].2.2[k]) :
    D.PartsZero cls acc i r gs := by
  intro p hp
  obtain ⟨q, hq, hkey, ns, nv, hpart⟩ := hcover p hp
  refine ⟨q.S, ns, nv, hpart, elemAt x ms cs (keyAt p), vvAt x cs (keyAt p), fun k hk => hE p hp k hk,
    fun v info kappa h k hk => hvv v info kappa h p hp k hk, ?_⟩
  rw [← hkey]
  exact hres q hq

/-! ## The assembly with row R discharged -/

/-- **Table `i`'s reps run at the decision**, as `runs_assembled`, with row R's S, groups and `w`'s size discharged,
its parts' residuals given by the `rec-residuals` pieces and their two ties (the layout's `hE`, check 5's `hvv`), and
its coin rounds' sizes (`hsz`) and the accumulators' number (`hacc`) as hypotheses. -/
theorem runs_assembled_R {xo : Claims.Msg} {sto : Claims.Recursion.Stmt} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) (hi : i < D.sts.size)
    (hti : i < D.spec.tables.size)
    (hframed : ∀ r : Fin 2, FramedAt ((D.files cls i).stream r).rounds 0 (repFrames (D.files cls i).st
      (D.files cls i).sch ((D.files cls i).spec.domain (D.files cls i).spec.tables[(D.files cls i).i]! r).toUTF8
      (2 + (D.files cls i).extras.size) ((D.files cls i).proof r)
      (valOf (D.sh cls i) (D.rep cls acc i r).w (D.rep cls acc i r).comb (D.rep cls acc i r).cons .T)))
    (hpts : D.sess.points = pointsOf D.spec D.d.c₁)
    (hk : ∀ r : Fin 2, 2 * i + r < D.d.f.ps.size)
    (hsh : ∀ r : Fin 2,
      ProofShaped (FlockDeploy.shapeAt (tabsOf D.sts) (pointsOf D.spec D.d.c₁) (2 * i + r))
        (D.d.f.ps[2 * i + r]'(hk r)))
    (hL : ∀ r, D.RowL cls acc i r)
    (pieces : List OpenPiece) (sideAt : Claims.Msg → List Claims.Msg → OpenPiece → OpenSide)
    (posAt : Claims.Msg → List Coin → OpenPiece → ℕ)
    (hholds : ∀ q ∈ pieces, openHolds sideAt posAt q x ms cs)
    (hside : ∀ r : Fin 2, ∀ q ∈ pieces, q.table = i → q.rep = r.val → sideAt x ms q = D.side cls i r)
    (hpos : ∀ r : Fin 2, ∀ v info kappa,
      verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras (D.rep cls acc i r).comb =
        .ok (v, info, kappa) →
      ∀ q ∈ pieces, q.table = i → q.rep = r.val → posAt x cs q = (info.lg.conss[q.level]!).pos[q.query]!)
    (hcover : ∀ r : Fin 2, ∀ ℓ < (D.files cls i).sch.levels.size, ∀ j < ((D.files cls i).sch.levels[ℓ]!).q,
      ∃ q ∈ pieces, q.table = i ∧ q.rep = r.val ∧ q.level = ℓ ∧ q.query = j)
    (hsz : ∀ r : Fin 2, (D.rep cls acc i r).rounds.toList.map Array.size = roundSizes (D.sh cls i))
    (hacc : ∀ r : Fin 2, (acc (2 * i + r)).size = 4)
    (rpieces : List ResPiece) (elemAt : Claims.Msg → List Claims.Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → F128)
    (vvAt : Claims.Msg → List Coin → ℕ × ℕ × ℕ × ℕ → ℕ → F128)
    (hres : ∀ q ∈ rpieces, resHolds elemAt vvAt q x ms cs) (keyAt : Fin 2 → ℕ → ℕ × ℕ × ℕ × ℕ)
    (hcoverR : ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
      ∀ (r : Fin 2) p (_hp : p < gs.size), ∃ q ∈ rpieces, q.key = keyAt r p ∧
        ∃ ns nv, RecResiduals.part (D.sh cls i) p = .ok (q.S, ns, nv))
    (hE : ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
      ∀ (r : Fin 2) p (hp : p < gs.size) k (hk : k < (wIdx (portSizes (D.sh cls i)) gs[p].2.1).size),
        elemAt x ms cs (keyAt r p) k = el (D.rep cls acc i r).w (wIdx (portSizes (D.sh cls i)) gs[p].2.1)[k])
    (hvv : ∀ (S : RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)),
      structureOf (D.sh cls i) = .ok S → groups S (portSizes (D.sh cls i)) = .ok gs →
      ∀ (r : Fin 2) v info kappa, verifierRows (D.sh cls i) (D.rep cls acc i r).rounds (D.rep cls acc i r).extras
          (D.rep cls acc i r).comb = .ok (v, info, kappa) →
        ∀ p (hp : p < gs.size) k (hk : k < gs[p].2.2.size), vvAt x cs (keyAt r p) k = el v gs[p].2.2[k])
    (accC : Fin 2 → ℕ → Array F128) (p0 : Fin 2 → ℕ)
    (hu : ∀ r : Fin 2, UnitSums (D.sh cls i) (accC r) (D.opens cls acc i r (p0 r)))
    (hrd : ∀ r : Fin 2, ReadsEnds (D.sh cls i) (D.rep cls acc i r) (accC r) (D.opens cls acc i r (p0 r))) :
    (D.files cls i).Runs := by
  obtain ⟨S, gs, hS, hg⟩ := D.structure_ok cls hst hf hx hi hpts
  have hG := D.rowG cls acc hst hf hx hi
  exact D.runs_assembled cls acc hst hf hx hi hti hframed hpts hk hsh hL pieces sideAt posAt hholds hside hpos hcover
    hS hg hsz (fun r => D.hw_of_acc cls acc i r hG hpts (hk r) (hsh r) hS (hacc r))
    (fun r => D.partsZero_of_pieces cls acc i r rpieces elemAt vvAt hres (keyAt r) (hcoverR S gs hS hg r)
      (hE S gs hS hg r) fun v info kappa h => hvv S gs hS hg r v info kappa h)
    accC p0 hu hrd

end DecAt

end RecDeploy.Rows
