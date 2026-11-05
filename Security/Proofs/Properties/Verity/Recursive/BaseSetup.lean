import Proofs.Flock.Claims.DeployInputs
import SampledProofs.OneStageZk

/-!
# A step of `acc_of_accepts`: a V\* part's base setup, compiled and model

A V\* part's verdict (`VStar.Recursive.partVerdict`) sets its session up with `SampledProofs.OneStageZk.buildSessionOf`
at the statement `verity/flock-circuit`'s tags `tags0`, on one table's keyed coin tree, with `--zk`'s hooks at the
circuit's `mask_log`, and no lookup tables. The model's recursive setup (`RecDeploy.RecIns.setup`) starts with the same
calls at its statement's reading `r` (`Readable.Read`): the hm96-sha512 rows check, `Read.setup`, each table's
`checkRegistered` and `noIdentifier`, and `Registered.checkFull` at the reading's draw.

`baseSetup_of_build`: where `r` reads the part's inputs as `FlockDeploy.Raw.ins` reads a `--zk` statement (the tags
`tags0.withCoins coins`, the coin tree, one table, no tables, no `--work`, `--stratified`, `--closure` or
`--public-inputs`), the compiled setup's success gives each of those model calls, with the same `(spec, sts)`:

* the tags agree: `withCoins` is the identity on a coin tree, so `r.tags` (`ZkExec.zkTags`) is the compiled `tags`, and
  `r.coins` (`ZkExec.zkCoins`) the compiled `coins`;
* the session agrees: under hidden outputs the compiled setup runs `Stmt.setupTablesOf` on `ProvedScope.circuitOf`'s
  parse, which is `Stmt.setupTables` (`Stmt.setupTables_eq`);
* the compiled `checkIdentifier` refuses every header naming an identifier, which is `noIdentifier`'s refusal.

The compiled `checkProgram`, `checkComplete`, `checkCover` and `KnownRead.check` have no counterpart in the model's
setup, so this needs none of them.
-/

open Lean (Json)
open Flock (V)

namespace Verity.Recursive.BaseSetup

theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

theorem forIn_list_ok {α : Type} {l : List α} {f : α → PUnit → V (ForInStep PUnit)}
    (h : ∀ a ∈ l, f a PUnit.unit = .ok (.yield PUnit.unit)) : forIn l PUnit.unit f = .ok PUnit.unit := by
  induction l with
  | nil => rfl
  | cons a l ih =>
    rw [List.forIn_cons, h a List.mem_cons_self]
    exact ih fun b hb => h b (List.mem_cons_of_mem _ hb)

/-- A `for` loop over an array whose every step yields is `.ok`. -/
theorem forIn_ok {α : Type} {xs : Array α} {f : α → PUnit → V (ForInStep PUnit)}
    (h : ∀ a ∈ xs, f a PUnit.unit = .ok (.yield PUnit.unit)) : forIn xs PUnit.unit f = .ok PUnit.unit := by
  rw [← Array.forIn_toList]
  exact forIn_list_ok fun a ha => h a (Array.mem_toList_iff.1 ha)

theorem forIn_list_mem {α : Type} {l : List α} {f : α → PUnit → V (ForInStep PUnit)}
    (hy : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn l PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ l, f a PUnit.unit = .ok (.yield PUnit.unit) := by
  induction l with
  | nil => intro a ha; cases ha
  | cons a l ih =>
    rw [List.forIn_cons] at h
    rcases hf : f a PUnit.unit with e | s
    · rw [hf] at h; cases h
    · have hs := hy a s hf
      subst hs
      rw [hf] at h
      intro b hb
      rcases List.mem_cons.1 hb with rfl | hb
      · exact hf
      · exact ih h b hb

/-- A `.ok` `for` loop over an array, whose steps only yield, yields at each element. -/
theorem forIn_mem {α : Type} {xs : Array α} {f : α → PUnit → V (ForInStep PUnit)}
    (hy : ∀ a s, f a PUnit.unit = .ok s → s = .yield PUnit.unit)
    (h : forIn xs PUnit.unit f = .ok PUnit.unit) : ∀ a ∈ xs, f a PUnit.unit = .ok (.yield PUnit.unit) := by
  rw [← Array.forIn_toList] at h
  exact fun a ha => forIn_list_mem hy h a (Array.mem_toList_iff.2 ha)

/-- A loop step `do A a; B a; pure (.yield ())` yields exactly when both calls are `.ok`. -/
theorem body_ok {α : Type} {A B : α → V Unit} {a : α} :
    (do A a; B a; pure (ForInStep.yield PUnit.unit) : V (ForInStep PUnit)) = .ok (.yield PUnit.unit) ↔
      A a = .ok () ∧ B a = .ok () := by
  rcases hA : A a with e | ⟨⟩ <;> rcases hB : B a with e' | ⟨⟩ <;>
    simp [bind, Except.bind, pure, Except.pure]

theorem body_yield {α : Type} {A B : α → V Unit} :
    ∀ a s, (do A a; B a; pure (ForInStep.yield PUnit.unit) : V (ForInStep PUnit)) = .ok s →
      s = .yield PUnit.unit := by
  intro a s h
  rcases hA : A a with e | ⟨⟩ <;> rcases hB : B a with e' | ⟨⟩ <;>
    simp [hA, hB, bind, Except.bind, pure, Except.pure] at h
  exact h.symm

/-- The compiled `checkIdentifier` (no `--identifier`) passing is the model's `noIdentifier` passing: both pass exactly
when the public header names no identifier, and the compiled one refuses every header that names one. -/
theorem noIdentifier_of_check {st : Flock.Stmt} (h : SampledProofs.OneStageZk.checkIdentifier st = .ok ()) :
    FlockDeploy.noIdentifier st = .ok () := by
  unfold SampledProofs.OneStageZk.checkIdentifier at h
  unfold FlockDeploy.noIdentifier
  rcases hj : st.pub.header.getObjVal? "identifier" with e | j
  · rfl
  · exfalso
    rw [hj] at h
    cases j
    all_goals try (simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h; done)
    next kvs =>
      cases hd : Flock.Program.domainError (.obj kvs) "identifier" <;>
        simp [hd, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

end Verity.Recursive.BaseSetup

namespace Verity.Recursive

open BaseSetup in
/-- **A V\* part's base setup.** Let `tags0` be the part's statement tags, and `coins` and `tags` the coin tree
and tags `partVerdict` sets its session up on. Suppose the compiled `buildSessionOf` sets the part's circuit `circuitB`
and staged public file `pubB` up at the draw `draw`, with `partition`, `program` and the verifier's registered reads
`registered`, to `(spec, sts)`. Let `r` read the same inputs as `Raw.ins` reads a `--zk` statement. Then the model's
calls at the start of `RecIns.setup` all succeed at `r` with the same `(spec, sts)`: the hm96-sha512 rows check,
`Read.setup`, each table's `checkRegistered` and `noIdentifier` (pointwise, and as `RecIns.setup`'s loop), and
`Registered.checkFull` at `r`'s draw. -/
theorem baseSetup_of_build {tags0 : Flock.Tags} {circuitB pubB : ByteArray} {draw partition : Option Json}
    {program : Option ByteArray} {registered : Option (Array Flock.Registered.Port)} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt}
    (hb : SampledProofs.OneStageZk.buildSessionOf
      (Flock.ProvedScope.circuitOf
        (Flock.Zk.tags (tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1)))) (Flock.Zk.maskLogOf circuitB) 1)
        circuitB [])
      (Flock.Zk.tags (tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1)))) (Flock.Zk.maskLogOf circuitB) 1)
      circuitB pubB (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) [] 1 draw partition program registered =
        .ok (spec, sts))
    (r : Flock.Readable.Read)
    (hI : r.I =
      { tags := tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1)))
        circuitFile := circuitB
        publicFile := pubB
        coins := .tree (Flock.Zk.coinSpec (tags0.tableNames 1))
        count := 1
        tables := []
        partition := partition
        program := program
        work := none
        stratified := none
        closure := none })
    (hzk : r.zk = true) (hdraw : r.draw = draw) (hpi : r.publicInputs = none) :
    r.tags.hm96Rows = true ∧ r.setup = .ok (spec, sts) ∧
      (∀ st ∈ sts, Flock.checkRegistered st registered = .ok () ∧ FlockDeploy.noIdentifier st = .ok ()) ∧
      (forIn sts PUnit.unit fun st _ => do
          Flock.checkRegistered st registered
          FlockDeploy.noIdentifier st
          pure (ForInStep.yield PUnit.unit) : V PUnit) = .ok PUnit.unit ∧
      Flock.Registered.checkFull registered r.draw = .ok () := by
  have hos : (Flock.Coins.tree (Flock.Zk.coinSpec (tags0.tableNames 1)) == Flock.Coins.os) = false := rfl
  have hwc : tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) = tags0 := by
    simp [Flock.Tags.withCoins, hos]
  have hcoins : r.coins = .tree (Flock.Zk.coinSpec (tags0.tableNames 1)) := by
    simp [Flock.Readable.Read.coins, hzk, FlockSoundness.Discharge.ZkExec.zkCoins, hI, hwc]
  have htags : r.tags =
      Flock.Zk.tags (tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1)))) (Flock.Zk.maskLogOf circuitB) 1 := by
    simp [Flock.Readable.Read.tags, hzk, FlockSoundness.Discharge.ZkExec.zkTags,
      FlockSoundness.Discharge.ZkExec.zkCoins, hI, hwc]
  have hsetup : r.setup = Flock.Stmt.setupTables
      (Flock.Zk.tags (tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1)))) (Flock.Zk.maskLogOf circuitB) 1)
      circuitB pubB (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) 1 [] draw partition program := by
    simp only [Flock.Readable.Read.setup, htags, hcoins, hdraw, hpi, hI]
  unfold SampledProofs.OneStageZk.buildSessionOf at hb
  have hh : (Flock.Zk.tags (tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))))
      (Flock.Zk.maskLogOf circuitB) 1).hm96Rows = true := by
    refine Classical.byContradiction fun hne => ?_
    simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at hb
  simp only [hh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure_bind] at hb
  have hstep : ∃ x : Flock.SessionSpec × Array Flock.Stmt,
      Flock.Stmt.setupTables
        (Flock.Zk.tags (tags0.withCoins (.tree (Flock.Zk.coinSpec (tags0.tableNames 1)))) (Flock.Zk.maskLogOf circuitB) 1)
        circuitB pubB (.tree (Flock.Zk.coinSpec (tags0.tableNames 1))) 1 [] draw partition program = .ok x ∧
      (∀ st ∈ x.2, Flock.checkRegistered st registered = .ok () ∧
        SampledProofs.OneStageZk.checkIdentifier st = .ok ()) ∧
      Flock.Registered.checkFull registered draw = .ok () ∧ (x.1, x.2) = (spec, sts) := by
    split at hb
    · next hhid =>
      obtain ⟨x, hs, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hloop, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hfull, hb⟩ := bind_ok hb
      refine ⟨x, ?_, fun st hst => (body_ok (A := fun st => Flock.checkRegistered st registered)
        (B := SampledProofs.OneStageZk.checkIdentifier)).1 (forIn_mem body_yield hloop st hst), hfull,
        Except.ok.inj hb⟩
      rw [Flock.Stmt.setupTables_eq]
      simpa only [Flock.ProvedScope.circuitOf, hhid, ↓reduceIte] using hs
    · obtain ⟨x, hs, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hloop, hb⟩ := bind_ok hb
      obtain ⟨⟨⟩, hfull, hb⟩ := bind_ok hb
      exact ⟨x, hs, fun st hst => (body_ok (A := fun st => Flock.checkRegistered st registered)
        (B := SampledProofs.OneStageZk.checkIdentifier)).1 (forIn_mem body_yield hloop st hst), hfull,
        Except.ok.inj hb⟩
  obtain ⟨⟨spec', sts'⟩, hs, hpt, hfull, hx⟩ := hstep
  simp only [Prod.mk.injEq] at hx
  obtain ⟨rfl, rfl⟩ := hx
  have hpt' : ∀ st ∈ sts', Flock.checkRegistered st registered = .ok () ∧
      FlockDeploy.noIdentifier st = .ok () :=
    fun st hst => ⟨(hpt st hst).1, noIdentifier_of_check (hpt st hst).2⟩
  refine ⟨by rw [htags]; exact hh, hsetup.trans hs, hpt', forIn_ok fun st hst =>
    (body_ok (A := fun st => Flock.checkRegistered st registered) (B := FlockDeploy.noIdentifier)).2 (hpt' st hst), ?_⟩
  rw [hdraw]
  exact hfull

end Verity.Recursive
