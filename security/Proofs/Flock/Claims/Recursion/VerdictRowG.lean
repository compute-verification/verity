module

public import Proofs.Flock.Claims.Complete.Reps
public import Proofs.Flock.Claims.Recursion.VerdictRows
public import Proofs.Flock.SessionLoop.FinPoints
public import Proofs.Flock.VBridge.Family.Setup
import Flock.With.Returns
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Replay.LincheckAlgebra
import Proofs.Flock.Claims.Recursion.InnerForm
import Proofs.Flock.Claims.Recursion.SchedCount
import Proofs.Flock.Claims.Recursion.SchedDense
import Proofs.Flock.Level3.Assembly
import Proofs.Flock.Level3.FoldedBase
import Proofs.Flock.Recursive.ExecClass
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Refine.Arith
import Proofs.Flock.VBridge.Algebra.RunSkel

/-! Row G: the verdict's setup is the reader's (`DecAt.rowG`). -/

@[expose] public section

namespace RecDeploy.Rows

open Flock
open FlockDeploy (Coin FinalP tabsOf)

theorem forall₂_right {α β : Type} {R : α → β → Prop} :
    ∀ {l₁ : List α} {l₂ : List β}, List.Forall₂ R l₁ l₂ → ∀ b ∈ l₂, ∃ a, R a b
  | _, _, .nil, _, h => by cases h
  | _, _, .cons hab h, b, hm => by
    rcases List.mem_cons.1 hm with rfl | hm
    · exact ⟨_, hab⟩
    · exact forall₂_right h b hm

theorem extras_shape {s : Stmt} {pts : Array (Array F128)} {e : Array Claim} (h : s.extraClaims pts = .ok e)
    (hm : 6 ≤ s.m) : ∀ j < e.size, e[j]!.skip.size = 64 ∧ e[j]!.x.size = s.m - 6 := by
  intro j hj
  obtain ⟨ess, hess, hfl⟩ := FlockSoundness.Refine.extraClaims_ok s pts e h
  have hmem : e[j]! ∈ ess.flatten := by
    rw [← hfl, getElem!_pos e j hj]
    exact Array.getElem_mem_toList hj
  obtain ⟨es, hes, hmem'⟩ := List.mem_flatten.1 hmem
  obtain ⟨reg, hreg⟩ := forall₂_right hess es hes
  obtain ⟨pt, v, -, hv⟩ := forall₂_right hreg _ hmem'
  rw [hv]
  refine ⟨?_, ?_⟩
  · show (eqTable ((s.regionPoint reg pt).extract 0 6)).size = 64
    rw [FlockLevel3.eqTable_size, Array.size_extract, FlockSoundness.Refine.regionPoint_size, Nat.min_eq_left hm]
    all_goals rfl
  · show ((s.regionPoint reg pt).extract 6 (s.regionPoint reg pt).size).size = s.m - 6
    rw [Array.size_extract, FlockSoundness.Refine.regionPoint_size, Nat.min_self]

theorem slotOk_of_hidden {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) :
    ∀ r ∈ st.c.ranges.toList, FlockComplete.JsonRT.SlotOk st.c r := by
  obtain ⟨ls, hl⟩ := FlockSoundness.Discharge.Hidden.parseCommit_check ht
    (FlockSoundness.Discharge.Hidden.setupHidden_parseCommit h)
  exact FlockComplete.JsonRT.hm_check_kinds hl

theorem folds_ofCircuit {S : Setup} {s : Stmt} (hS : S = Setup.ofCircuit s) (hwf : FlockSoundness.Refine.StmtWF s)
    (hs : ∀ r ∈ s.c.ranges.toList, FlockComplete.JsonRT.SlotOk s.c r) (h14 : 14 ≤ s.m) {sh : Shape}
    (hm : sh.m = s.m) (hk : sh.kLog = s.c.kLog) {rounds : Array (Array F128)} {extras : Array Claim}
    {lins : Array RecAlgebra.Lin} {info : RecAlgebra.Info RecAlgebra.Lin}
    (h : RecAlgebra.residualLins sh (RecAlgebra.RoundCoins.ofRounds rounds) extras = .ok (lins, info)) :
    ∃ c, S.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn info.lc.fold.ts =
      .ok c ∧ c.size = 64 := by
  subst hS
  obtain ⟨j, hj⟩ := RecAlgebra.residualLins_run h
  obtain ⟨-, r0, r1, rsr, lgr, k1, k2, k3, h0, h1, -⟩ := RecAlgebra.run_stages hj
  obtain ⟨-, -, -, -, -, -, -, -, hrs, -⟩ := RecAlgebra.zerocheck_coins (by omega) h0
  obtain ⟨-, -, -, -, -, -, -, -, -, hlw, hrho, -⟩ := RecAlgebra.lincheck_coins h1
  have hz : (RecAlgebra.symbolic sh).zc.size = sh.m - 6 := by simp [RecAlgebra.symbolic]
  have hlw' : info.lc.fold.lw.size = 64 := by
    rw [hlw]
    exact FlockSoundness.Refine.skipWeights_size _
  have hrho' : info.lc.fold.rhoIn.size = s.c.kLog - 6 := by
    rw [hrho, Array.size_extract, hrs, hz]
    have := hwf.hk
    omega
  obtain ⟨v, hv⟩ := FlockComplete.JsonRT.fold_ok s hwf.hk6 hs info.lc.fold.alpha hlw' hrho'
  have hfd := FlockLevel3.folded_ok s info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
    info.lc.fold.ts v hv
  refine ⟨_, hfd, ?_⟩
  obtain ⟨v', -, hsz, hw⟩ := (FlockSoundness.Refine.ofCircuit_fold s hwf []).folded _ _ _ _ _ _ hfd
  rw [hw, ← Array.foldl_toList, FlockLevel3.foldl_halve_size, Array.size_modify, hsz, Array.length_toList,
    FlockVBridge.Family.ts_size h]
  show 2 ^ s.c.kLog / 2 ^ (sh.kLog - 6) = 64
  rw [hk]
  obtain ⟨d, hd⟩ : ∃ d, s.c.kLog = d + 6 := ⟨s.c.kLog - 6, by have := hwf.hk6; omega⟩
  rw [hd, Nat.add_sub_cancel, pow_add, Nat.mul_div_cancel_left _ (Nat.two_pow_pos _)]
  all_goals rfl

theorem dense_inner {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) :
    ∃ s spec st sch, FlockDeploy.setupOf false (xIOf xo) = some (s, spec, #[st]) ∧
      s.read.tags = RecCoins.innerTags ∧ s.read.publicInputs = none ∧ fast100 st.m = some sch := by
  obtain ⟨r, hr, ha, rfl⟩ := parse_read hst
  have hh : r.shape.holo = false := hf.1
  obtain ⟨f, spec', st', s, spec, st, hfs, hstm, hset, hm, -, -, -, htags, hpi⟩ := innerSetup_files hr ha
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨⟨⟨ss, hss⟩, -⟩, ⟨u, hsh⟩⟩, -⟩ := ha
  have hss : RecVStar.statements r.shape = .ok ss := hss
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  rw [hfs] at hsh
  obtain ⟨spec'', st'', hstm', e1, -, -, -, -⟩ := checkShape_files hsh
  rw [hstm] at hstm'
  cases hstm'
  obtain ⟨rs, hrs⟩ := repSched_of_statements hss
  have hrs' : RecVStar.repRounds r.shape.m r.shape.kLog r.shape.claims = some rs := by
    unfold repSched at hrs
    rw [hh] at hrs
    exact hrs
  obtain ⟨sch, hsch, -⟩ := repRounds_size hrs'
  have hx : xIOf xo = r.w.inner := by unfold xIOf; rw [hst]; rfl
  refine ⟨s, spec, st, sch, by rw [hx]; exact hset, htags, hpi, ?_⟩
  rw [hm, ← e1]
  exact hsch

namespace DecAt

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}
  (D : DecAt rd x ms cs)

theorem read_setup : D.d.s.read.setup = .ok (D.spec, D.sts) := by
  have h := D.setup
  rw [FlockDeploy.hOf_sha512, FlockDeploy.setupWith_sha512] at h
  exact h

theorem setup_eq : D.spec = D.d.spec ∧ D.sts = D.d.sts := by
  obtain ⟨⟨s, spec, sts⟩, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  obtain ⟨-, hs⟩ := FlockDeploy.setupOf_some hv
  have h1 := FlockDeploy.Ins.setup_read hs
  have h2 := D.read_setup
  rw [hd] at h2
  have h3 : (Except.ok (spec, sts) : Flock.V (SessionSpec × Array Stmt)) = .ok (D.spec, D.sts) := h1.symm.trans h2
  have e := Prod.mk.inj (Except.ok.inj h3)
  rw [hd]
  exact ⟨e.1.symm, e.2.symm⟩

theorem sts_hidden (hT : D.d.s.read.tags = RecVStar.innerTags ∧ D.d.s.read.publicInputs = none) :
    ∀ t ∈ D.sts, ∃ (tags : Tags) (cf pf : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
      (draw partition : Option Lean.Json) (program : Option ByteArray),
      tags.merkleLeaf = "flock-leaf/sha512-unsalted" ∧ tags.typed = false ∧
        Stmt.setupHidden tags cf pf coins tables draw partition program = .ok t := by
  have h := D.read_setup
  unfold Readable.Read.setup at h
  rw [hT.1, hT.2] at h
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH
      FlockSoundness.Discharge.Recursion.innerTags_hiddenOutputs h with
    ⟨-, st, hst, -, hsts⟩ | ⟨-, -, hall, -, -⟩
  · rw [hsts]
    intro t ht
    obtain rfl : t = st := by simpa using ht
    exact ⟨_, _, _, _, _, _, _, _, rfl, rfl, hst⟩
  · intro t ht
    obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 ht
    obtain ⟨d, hd⟩ := hall j hj
    exact ⟨_, _, _, _, _, _, _, _, rfl, rfl, hd⟩

theorem tab_eq {i : ℕ} (hi : i < D.sts.size) : (tabsOf D.sts)[i]! = Setup.ofCircuit D.sts[i] := by
  unfold FlockDeploy.tabsOf
  rw [getElem!_pos (D.sts.map Setup.ofCircuit) i (by simpa using hi), Array.getElem_map]

theorem merkle_of_tags (hT : D.d.s.read.tags = RecVStar.innerTags ∧ D.d.s.read.publicInputs = none) {i : ℕ}
    (hi : i < D.sts.size) :
    (tabsOf D.sts)[i]!.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" ∧
      (tabsOf D.sts)[i]!.merkle.Named := by
  have hm : (tabsOf D.sts)[i]!.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted" := by
    rw [D.tab_eq hi]
    obtain ⟨_, _, _, _, _, _, _, _, hl, -, hs⟩ := D.sts_hidden hT _ (Array.getElem_mem hi)
    exact FlockSoundness.Discharge.Hidden.setupHidden_plain hl hs
  refine ⟨hm, ?_⟩
  rw [hm]
  exact With.named_of_find? (id := "flock-leaf/sha512-unsalted") rfl

variable (cls : FlockSoundness.Discharge.Recursion.HClass)

theorem foldKLog_of {i : ℕ} (hi : i < D.sts.size) : (D.files cls i).st.fold.kLog = (D.sh cls i).kLog := by
  show (tabsOf D.sts)[i]!.fold.kLog = (tabsOf D.sts)[i]!.kLog
  rw [D.tab_eq hi]
  rfl

theorem kLog_le_of_tags (hT : D.d.s.read.tags = RecVStar.innerTags ∧ D.d.s.read.publicInputs = none) {i : ℕ}
    (hi : i < D.sts.size) :
    (D.files cls i).st.kLog ≤ (D.files cls i).st.m := by
  show (tabsOf D.sts)[i]!.kLog ≤ (tabsOf D.sts)[i]!.m
  rw [D.tab_eq hi]
  obtain ⟨_, _, _, _, _, _, _, _, -, -, hs⟩ := D.sts_hidden hT _ (Array.getElem_mem hi)
  exact (FlockSoundness.Discharge.Hidden.setupHidden_wf hs).1.hk

theorem rowG (acc : ℕ → Array F128) {xo : Claims.Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : x = xIOf xo) {i : ℕ} (hi : i < D.sts.size) :
    RowG D cls acc i := by
  obtain ⟨s, spec, st, sch, hset, htags, hpi, hsch⟩ := dense_inner hst hf
  rw [← hx] at hset
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  obtain ⟨v, c₁, r₀, rss, fl, link, pf, hv, -, -, -, -, -, hd⟩ := FlockDeploy.decR_some D.dec
  rw [hset] at hv
  cases hv
  have hds : D.d.s = s := congrArg (·.s) hd
  have hdsts : D.d.sts = #[st] := congrArg (·.sts) hd
  have hT : D.d.s.read.tags = RecVStar.innerTags ∧ D.d.s.read.publicInputs = none := by
    rw [hds]; exact ⟨htags, hpi⟩
  have hsts1 : D.sts = #[st] := hsts.trans hdsts
  have hi0 : i = 0 := by
    have : i < 1 := by simpa [hsts1] using hi
    omega
  have hsti : D.sts[i] = st := by
    have h' : ∀ (a : Array Stmt) (h : i < a.size), a = #[st] → a[i] = st := by
      intro a h ha
      subst ha
      subst hi0
      rfl
    exact h' D.sts hi hsts1
  have hfast : fast100 D.sts[i].m = some sch := by rw [hsti]; exact hsch
  have h22 := (FlockVBridge.Family.fast100_range hfast).1
  obtain ⟨tags, cf, pf', coins, tables, draw, part, prog, -, htyped, hhid⟩ :=
    D.sts_hidden hT _ (Array.getElem_mem hi)
  have hwf := (FlockSoundness.Discharge.Hidden.setupHidden_wf hhid).1
  have hslot := slotOk_of_hidden htyped hhid
  have hb := Claims.ZKMal.Masked.setupHidden_regionBytes hhid
  have hmk := D.merkle_of_tags hT hi
  obtain ⟨e, he⟩ := FlockVBridge.Family.extraClaims_exists D.sts[i] D.sess.points hb
  have hex : (D.files cls i).extras = e := by
    show ((tabsOf D.sts)[i]!.extra D.sess.points).toOption.getD #[] = e
    rw [D.tab_eq hi]
    show (D.sts[i].extraClaims D.sess.points).toOption.getD #[] = e
    rw [he]
    rfl
  have hshm : (D.sh cls i).m = D.sts[i].m := by
    show (tabsOf D.sts)[i]!.m = D.sts[i].m
    rw [D.tab_eq hi]
    rfl
  have hshk : (D.sh cls i).kLog = D.sts[i].c.kLog := by
    show (tabsOf D.sts)[i]!.kLog = D.sts[i].c.kLog
    rw [D.tab_eq hi]
    rfl
  refine
    { spec := hsp
      sts := hsts
      merkle := hmk.1
      named := hmk.2
      foldKLog := D.foldKLog_of cls hi
      kLog_le := D.kLog_le_of_tags cls hT hi
      fast := ?_
      extra := ?_
      extras_wf := ?_
      foldPts := ?_ }
  · show fast100 (tabsOf D.sts)[i]!.m = some ((fast100 (tabsOf D.sts)[i]!.m).getD default)
    rw [D.tab_eq hi]
    show fast100 D.sts[i].m = some ((fast100 D.sts[i].m).getD default)
    rw [hfast]
    rfl
  · rw [hex]
    show (tabsOf D.sts)[i]!.extra D.sess.points = .ok e
    rw [D.tab_eq hi]
    exact he
  · rw [hex]
    have hst' : (D.files cls i).st.m = D.sts[i].m := by
      show (tabsOf D.sts)[i]!.m = D.sts[i].m
      rw [D.tab_eq hi]
      rfl
    rw [hst']
    exact extras_shape he (by omega)
  · intro r v info kappa h
    obtain ⟨lins, hl⟩ := FlockVBridge.Family.residualLins_of_verifierRows h
    have hl' : RecAlgebra.residualLins (D.sh cls i) (RecAlgebra.RoundCoins.ofRounds (FlockVBridge.Family.roundsOf
        (D.files cls i) r)) (D.files cls i).extras = .ok (lins, info) := hl
    obtain ⟨c, hcf, hcs⟩ := folds_ofCircuit (S := (D.files cls i).st) (D.tab_eq hi) hwf hslot (by omega) hshm hshk hl'
    show (D.files cls i).st.fold.folded info.lc.fold.alpha info.lc.fold.beta info.lc.fold.lw info.lc.fold.rhoIn
      info.lc.fold.ts = .ok (FlockVBridge.Family.combOf (D.sh cls i) (D.files cls i)
        (FlockVBridge.Family.roundsOf (D.files cls i) r))
    rw [hcf]
    unfold FlockVBridge.Family.combOf
    rw [hl']
    simp only
    rw [hcf]
    simp [hcs]

end DecAt

end RecDeploy.Rows
