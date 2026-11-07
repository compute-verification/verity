module

public import Proofs.Flock.Claims.Recursion.Deploy
public import Proofs.Flock.Claims.Recursion.SetupCount
public import Proofs.Flock.Claims.Recursion.SetupValues
import Flock.With.SetupEq
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
import Proofs.Flock.Claims.Recursion.AcceptedScope
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.DeployScopeRuns
import Proofs.Flock.Claims.Recursion.InnerForm
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.Soundness.Discharge.Hidden.Spec
import Proofs.Flock.Soundness.Discharge.HiddenExec.Accepts
import Proofs.Flock.Soundness.Discharge.Layout.Drawn

/-!
# The inner session's setup calls at an accepted statement, in any hash

`kappa_eps_deployed` takes `hS`: the inner session's setup makes at most `2^32` SHA-512 evaluations
(`innerSetupIns`). Here it is a fact about the verifier's own setup at any statement some run of the compiled verifier
accepts, and it holds for every hash `h`, not only SHA-512 (`innerSetupCalls_le_acc`):

* `Stmt.setupTablesWith`'s calls are at most `innerB = 12 + 1024 · perRow 17`, where `perRow K = (K + 3) · 2^K`
  (`innerTables_cost`), from `SetupCount.setupHidden_cost` at the circuit's ports (at most `1024`), the public file's
  shared-row tables (at most `2^17` rows each) and the sampled population (at most `2^17`);
* those three are functions of the files alone, whatever the hash (`SetupValues`), so the bounds the verifier
  establishes at SHA-512 hold at `h`: the ports from `checkInRange`'s regions (`read_setup_in`, `regions_size`), and
  the rows and the files' statement's `n`, which the sample's population is (`Layout.drawn_spec`), from
  `RecBudget.innerRows` (`checkShape_rows`);
* `innerB = 12 + 5 · 2^29 ≤ 2^32` (`innerB_le`).

`RecDeploy.lamLin2_R_ge_in` (CallsClosed) takes `hS` and `c_B` from here.
-/

@[expose] public section

set_option Elab.async false

namespace RecDeploy.SetupCount

open Flock FlockComplete
open Lean (Json)

/-- The inner session's setup calls, at most. -/
def innerB : Nat := 12 + 1024 * perRow 17

theorem innerB_eq : innerB = 2684354572 := by
  unfold innerB perRow
  norm_num

theorem innerB_le : innerB ≤ 2 ^ 32 := by
  rw [innerB_eq]
  norm_num

/-- **`Stmt.setupHiddenWith`'s calls at the inner statement, in any hash**: at most `10 + 1024 · perRow 17`, where at
SHA-512 the sampled statement (`hst`) has at most `1024` regions and the unsampled one (`hst'`) at most `2^17` instances
and `2^17` rows in each shared-row table. -/
theorem innerHidden_cost (h : ByteArray → ByteArray) {tags : Tags} {cf pf : ByteArray} {d : Json} {co co' : Coins}
    {pa pa' : Option Json} {pr pr' : Option ByteArray} {st st' : Stmt} (ht : tags.typed = false)
    (hi : tags.isTmpl = false) (hst : Stmt.setupHidden tags cf pf co [] (some d) pa pr = .ok st)
    (hst' : Stmt.setupHidden tags cf pf co' [] none pa' pr' = .ok st')
    (hreg : st.regions.size ≤ 1024) (hn : st'.pub.n ≤ 2 ^ 17)
    (hrows : ∀ t, st'.pub.tables = some t → ∀ r ∈ t.counts, r ≤ 2 ^ 17)
    (coins : Coins) (partition : Option Json) (program publicInputs : Option ByteArray) :
    countC.P (Stmt.setupHiddenWith (Hashes.counted h) tags cf pf coins [] (some d) partition program publicInputs)
      (10 + 1024 * perRow 17) := by
  have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' hst
  have s' := FlockSoundness.Discharge.Hidden.setupHidden_spec' hst'
  have hc : st.c = st'.c := Except.ok.inj (s.parse.symm.trans s'.parse)
  have hP : st'.c.ports.size ≤ 1024 := by
    rw [← hc, ← RecDeploy.regions_size s.regions]
    exact hreg
  obtain ⟨pop, hpop, -, hd⟩ := s.pub
  obtain ⟨pop', hpop', hn', -⟩ := s'.pub
  have hdr : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hd d rfl
  rw [hc] at hpop
  have hpp : pop = pop' := Except.ok.inj (hpop.symm.trans hpop')
  have hpub : st'.pub = pop := (hn' rfl).trans hpp.symm
  rw [hpub] at hrows
  have hstd : HmOut.parseWith (M := Except String) (Hashes.ofSha512 Sha512.hash) tags cf [] = .ok st'.c :=
    (With.outParseWith_std tags cf []).trans (s'.parseF hi)
  have hrow : HmRow.loadPublicWith (M := Except String) (Hashes.ofSha512 Sha512.hash) tags
      { st'.c with outPorts := #[] } pf = .ok pop :=
    (With.loadPublicWith_std _ _ _).trans (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop)
  obtain ⟨D, hD, hDp, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
  have hN : D.population ≤ 2 ^ 17 := by
    rw [hDp, ← hpub]
    exact hn
  refine setupHidden_cost h tags cf pf coins [] d partition program publicInputs (fun c => c.ports = st'.c.ports)
    1024 17 ht hi (fun c hc' => by have e : c.ports = st'.c.ports := hc'; rw [e]; exact hP) ?_ ?_ ?_
  · exact okC_of_V (fun k => With.hmOutParseWith_sim (With.Sim.forget _) With.Throws.exceptT With.Throws.v
      (With.Sim.forget_counted h) tags cf [] k) fun c hc' => SetupValues.parse_ports_eq h Sha512.hash ht hc' hstd
  · intro c pi hc'
    exact okC_of_V (fun k => With.hmInLoadPublicWith_sim (With.Sim.forget _) With.Throws.exceptT With.Throws.v
      (With.Sim.forget_counted h) tags c pi pf k) fun p hp =>
      SetupValues.counts_le (SetupValues.load_counts_eq h Sha512.hash hp hrow hc') hrows
  · intro c pi p hpt
    exact okC_of_V (fun k => With.hmInDrawnWith_sim (With.Sim.forget _) With.Throws.exceptT With.Throws.v
      (With.Sim.forget_counted h) c pi p d k) fun pub hp => by
      obtain ⟨D', hD', hn, he⟩ := SetupValues.drawn_pop h hp
      rw [hD] at hD'
      cases hD'
      refine ⟨by unfold PT at hpt ⊢; exact SetupValues.counts_le he hpt, ?_⟩
      unfold PN
      rw [hn]
      exact hN

/-- **`Stmt.setupTablesWith`'s calls at the inner statement, in any hash**: at most `innerB`, at one table with hidden
outputs. -/
theorem innerTables_cost (h : ByteArray → ByteArray) {tags : Tags} {cf pf : ByteArray} {d : Json} {co co' : Coins}
    {pa pa' : Option Json} {pr pr' : Option ByteArray} {st st' : Stmt} (ht : tags.typed = false)
    (hi : tags.isTmpl = false) (hh : tags.hiddenOutputs = true)
    (hst : Stmt.setupHidden tags cf pf co [] (some d) pa pr = .ok st)
    (hst' : Stmt.setupHidden tags cf pf co' [] none pa' pr' = .ok st')
    (hreg : st.regions.size ≤ 1024) (hn : st'.pub.n ≤ 2 ^ 17)
    (hrows : ∀ t, st'.pub.tables = some t → ∀ r ∈ t.counts, r ≤ 2 ^ 17)
    (coins : Coins) (count : Nat) (hcount : count ≤ 1) (partition : Option Json) (program : Option ByteArray)
    (work : Option (Nat × Array (String × Nat × Nat))) (stratified : Option Nat)
    (closure : Option (Std.HashMap Nat (Array Nat))) (publicInputs : Option ByteArray) :
    Stmt.setupCalls h tags cf pf coins count [] (some d) partition program work stratified closure publicInputs ≤
      innerB := by
  have hc := countC.mono (setupTables_cost h tags cf pf coins count [] d partition program work stratified closure
    publicInputs _ hcount hh (innerHidden_cost h ht hi hst hst' hreg hn hrows coins partition program publicInputs))
    (show 2 + (10 + 1024 * perRow 17) ≤ innerB by unfold innerB; omega)
  have h0 := hc 0
  unfold Stmt.setupCalls
  omega

end RecDeploy.SetupCount

namespace RecDeploy

open Flock FlockSoundness.Refine
open Claims (Msg Out Prover run)
open FlockDeploy (Coin Ins)

/-- **The inner session's setup calls, in any hash**: where the setup's checks accept what the verifier reads of a
compiled statement, Flock's plain verifier sets the inner statement up (`innerSetup_files`) as one table (the files
name no lookup tables, `RecSetup.checkForm`), and its setup's hash calls are at most `SetupCount.innerB` with any hash
as the oracle. -/
theorem innerSetup_calls {x : Msg} {r : XR} (hr : readX x = some r) (ha : RecScope.accepts (readOf r) = true) :
    ∃ s spec sts, FlockDeploy.setupOf false r.w.inner = some (s, spec, sts) ∧ sts.size = 1 ∧
      ∀ h, FlockDeploy.setupCallsOf h s.read ≤ SetupCount.innerB := by
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
      some (innerFormOf r.innerRaw), r.innerRaw.program⟩) := by rw [filesOf, hp]; rfl
  rw [hfs] at hsh
  obtain ⟨spec', st', hstm, hn0, hrows0⟩ := checkShape_rows hsh
  obtain ⟨spec2, st2, hstm2, -, -, -, -, hform⟩ := checkShape_files hsh
  have e2 : st2 = st' := congrArg Prod.snd (Except.ok.inj (hstm2.symm.trans hstm))
  rw [e2] at hform
  obtain ⟨hst', -⟩ := files_stmt hstm
  have hn : st'.pub.n ≤ 2 ^ 17 := hn0.trans (by norm_num [RecBudget.MAX_INNER_ROWS])
  have hrows : ∀ t, st'.pub.tables = some t → ∀ r ∈ t.counts, r ≤ 2 ^ 17 := fun t ht r hrt =>
    (hrows0 t ht r hrt).trans (by norm_num [RecBudget.MAX_INNER_ROWS])
  obtain ⟨hs1, hs3, b, d, D, hb, hjd, hD, hpr⟩ := checkForm_ok hform
  have hs1 : r.innerRaw.statement = Tags.circuitPlainLeaves.name := hs1
  have hs3 : r.innerRaw.tables = [] := hs3
  have hb : r.innerRaw.draw = some b := hb
  obtain ⟨hdec, -, hok⟩ := readX_inner hr
  obtain ⟨s, spec, sts, hset, hchk⟩ := innerOk_ok hok
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hw : r.innerRaw.zk = false ∧ r.innerRaw.ins = .ok s := by
    unfold FlockDeploy.insOf at hins
    obtain ⟨w, hw0, hw⟩ := Option.bind_eq_some_iff.1 hins
    rw [hdec] at hw0
    cases hw0
    split_ifs at hw with hz
    · cases hw
    change r.innerRaw.ins.toOption = some s at hw
    exact ⟨not_not.1 hz, toOption_eq_some hw⟩
  obtain ⟨z1, z2, z3, z4, z5, -, z7, z8, z9, -⟩ := ins_plainLeaves hw.2 hw.1 hs1
  obtain ⟨hcount, hpi⟩ := check_plain hchk z1
  have hpart : s.read.I.partition = some p := by
    have hpa := (readX_spec hr).2.1
    rw [z8] at hpa
    simp only [Except.toOption, Option.some.injEq] at hpa
    rw [hpa, hp]
  have hdraw : s.read.draw = some (Flock.norm d) := by
    obtain ⟨dr, hz9, he⟩ := z9
    rw [hb] at hz9
    simp only [Option.mapM_some, hjd, Functor.map, Except.map, Except.ok.injEq] at hz9
    rw [he, ← hz9]
    rfl
  have hsetup := FlockDeploy.Ins.setup_read hsu
  obtain ⟨-, -, -, hrange⟩ := read_setup_in hsetup hpi
  unfold Readable.Read.setup at hsetup
  rw [z2, z3, z4, z5, z7, hs3, hdraw, hpart, hpi] at hsetup
  have hh : RecCoins.innerTags.hiddenOutputs = true := rfl
  have ht : RecCoins.innerTags.typed = false := rfl
  have hi : RecCoins.innerTags.isTmpl = false := by rfl
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH hh hsetup with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc, -⟩
  · refine ⟨s, _, #[st], hset, rfl, fun h => ?_⟩
    obtain ⟨-, hreg, -⟩ := (show _ ∧ st.regions.size ≤ 1024 ∧ _ from hrange st (by simp))
    unfold FlockDeploy.setupCallsOf
    rw [z2, z3, z4, z5, z7, hs3, hdraw, hpart, hpi]
    exact SetupCount.innerTables_cost h ht hi hh hst hst' hreg hn hrows _ _ hcount _ _ _ _ _ _
  · omega

section Runs

variable {H : Msg → Msg} {IV : Claims.Verifier Coin} {st : Stage} {x : Msg} {Ω : Type} {P : Prover Ω} {ω : Ω}
  {r : Fin ((V IV st).k x) → Coin}

/-- **The inner session's setup calls at an accepted statement, in any hash**: on a run the compiled verifier accepts,
the plain verifier sets the inner statement `xIOf x` up as one table, with at most `SetupCount.innerB` hash calls
whatever the hash. -/
theorem innerSetupCalls_le_acc (hacc : (run H P (V IV st) x ω r).out = .acc) :
    ∃ s spec sts, FlockDeploy.setupOf false (xIOf x) = some (s, spec, sts) ∧ sts.size = 1 ∧
      ∀ h, FlockDeploy.setupCallsOf h s.read ≤ SetupCount.innerB := by
  obtain ⟨xr, hx, hacc', -⟩ := run_acc_scope H IV st x P ω r hacc
  obtain ⟨as', ha⟩ := stepWith_acc H P x ω _ _ _ _ _ hacc
  obtain ⟨xr', hx', hp, -⟩ := acc_parse (IV := IV) (st := st) H ha
  have hxx : xr' = xr := Option.some.inj (hx'.symm.trans hx)
  rw [hxx] at hp
  have hxI : xIOf x = xr.w.inner := by
    unfold xIOf
    rw [hp]
    rfl
  rw [hxI]
  exact innerSetup_calls hx hacc'

/-- **`hS` at an accepted statement**: the inner session's setup inputs at SHA-512 number at most `2^32`. -/
theorem innerSetupIns_le_acc (hacc : (run H P (V IV st) x ω r).out = .acc) :
    (innerSetupIns (xIOf x)).length ≤ 2 ^ 32 := by
  obtain ⟨s, spec, sts, hset, -, hle⟩ := innerSetupCalls_le_acc hacc
  simp only [innerSetupIns, hset, List.length_map]
  rw [(FlockDeploy.setupM_eval FlockDeploy.sha512 s.read).2]
  exact (hle _).trans SetupCount.innerB_le

end Runs

end RecDeploy
