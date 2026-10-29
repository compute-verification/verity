import Proofs.Flock.Claims.KS.ZkExtend
import Proofs.Flock.Claims.Recursion.DeployOuter

/-!
# Masked knowledge soundness at the outer verifier (`zk_KS_outer`)

The recursion's outer verifier (`RecDeploy.deployZkRec`) is `deployZk` at another setup, `SO`: masked, the recursive
setup (`RecDeploy.setupO`) at its Flock inputs, whose `Ok` is the recursive setup's success (`RecIns.setup`, which runs
the verifier's inner-shape refusals, `RecSetup.checkInner`, among its own), with the verdict's last refusals
(`RecDeploy.fullDrawO`) after the checks (`wrapO`); plain, `S₀`'s. At `SO` the masked chain's copy of the masked
instance (`KS.Gen.deployZk`) is `deployZkRec` (`deployZk_SO`): the same step (`act_SO`), the decision's refusal after the
verdict (`outOf_decision_SO`), the same reading of the statement, columns and numbers. So the chain's knowledge
soundness at `SO` (`KS.Gen.zk_KS_untrim`) is the outer verifier's (`zk_KS_outer`).
-/

namespace FlockDeploy.KS

open Flock
open RecDeploy (RecIns insO setupO readOfO fullDrawO actO nextO VO runOfO colsO tablesO εtableO N₀O weightO
  deployZkRec)

/-- **The outer verifier's refusal after the checks**: the verdict's last refusals (`fullDrawO`). -/
def wrapO (x : Claims.Msg) (f : Final) (v : Flock.V Unit) : Flock.V Unit :=
  if fullDrawO x f then v else .error "rec: refused by the verdict's last refusals (fullDraw, innerDrawn, denseOnly)"

/-- Under `--zk` the recursive setup passes every table's `Zk.shapeOk`. -/
theorem recIns_shapeOk {r : RecIns} {spec : SessionSpec} {sts : Array Stmt} (h : r.setup = .ok (spec, sts))
    (hz : r.base.read.zk = true) : sts.forM Zk.shapeOk = .ok () := by
  unfold RecDeploy.RecIns.setup at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  all_goals
    obtain ⟨-, h2⟩ := Prod.mk.inj (Except.ok.inj h)
    subst h2
    assumption

/-- **The outer verifier's setup**: masked, the recursive setup at its Flock inputs (`RecIns.base`), `Ok` where the
recursive setup succeeds, and the verdict's last refusals after the checks (`wrapO`); plain, `S₀`'s. -/
@[reducible] def SO : ZkSetup where
  ins zk x := match zk with
    | true => (insO x).map RecIns.base
    | false => insOf false x
  setup zk x := match zk with
    | true => (setupO x).map fun p => (p.1.base, p.2)
    | false => setupOf false x
  wrap zk x f v := match zk with
    | true => wrapO x f v
    | false => v
  Ok zk s spec sts := match zk with
    | true => ∃ r : RecIns, r.base = s ∧ r.setup = .ok (spec, sts)
    | false => s.setup = .ok (spec, sts)
  setup_some {zk x s spec sts} h := by
    cases zk
    · exact setupOf_some h
    · obtain ⟨⟨r, spec', sts'⟩, hr, he⟩ := Option.map_eq_some_iff.1 h
      obtain ⟨rfl, rfl, rfl⟩ : r.base = s ∧ spec' = spec ∧ sts' = sts := by simpa using he
      obtain ⟨hi, hs⟩ := RecDeploy.setupO_some hr
      exact ⟨by simp [hi], r, rfl, hs⟩
  ins_zk {zk x s} h := by
    cases zk
    · exact readOf_zk (zk := false) (x := x) (by simp [readOf, h])
    · obtain ⟨r, hr, rfl⟩ := Option.map_eq_some_iff.1 h
      exact RecDeploy.readOfO_zk (x := x) (by simp [readOfO, hr])
  ok_read {zk s spec sts} h := by
    cases zk
    · exact Ins.setup_read h
    · obtain ⟨r, rfl, hr⟩ := h
      exact RecDeploy.RecIns.setup_read hr
  wrap_ok {zk x f v} h := by
    cases zk
    · exact h
    · change wrapO x f v = .ok () at h
      unfold wrapO at h
      split at h
      · exact h
      · cases h
  ok_shape {zk s spec sts} h hz := by
    cases zk
    · exact Claims.Bridge.Custody.setup_shapeOk h hz
    · obtain ⟨r, rfl, hr⟩ := h
      exact recIns_shapeOk hr hz

theorem setupOf_SO (x : Claims.Msg) : @Gen.setupOf SO true x = (setupO x).map fun p => (p.1.base, p.2) := rfl

theorem checks_SO (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) :
    @Gen.checks SO true s spec x link c₁ mids f = wrapO x f (checks true s spec x link c₁ mids f) := rfl

/-- **The step at `SO` is the outer verifier's** (`actO`). -/
theorem act_SO (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : @Gen.act SO true x as cs = actO x as cs := by
  unfold Gen.act actO
  rw [setupOf_SO]
  rcases setupO x with _ | ⟨i, spec, sts⟩ <;> rfl

/-- A decision's outcome, read after a first step: a refusal of the step is a refusal. -/
theorem outOf_bind {β : Type} (m : Claims.OCV β) (G : β → Claims.OCV Unit) :
    outOf (m >>= G) = m.run.bind fun r => match r with
      | .ok b => outOf (G b)
      | .error _ => .ret .rej := by
  unfold outOf
  rw [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.bind_assoc']
  congr 1
  funext r
  cases r <;> rfl

/-- **The decision at `SO`** is the decision, refused where the verdict's last refusals fail (`nextO`'s decide
step). -/
theorem outOf_decision_SO (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) :
    outOf (@Gen.decision SO true s x link c₁ mids f) =
      (outOf (decision true s x link c₁ mids f)).bind fun o => .ret (if fullDrawO x f then o else .rej) := by
  unfold Gen.decision decision
  rw [outOf_bind, outOf_bind, Claims.OC.bind_assoc']
  congr 1
  funext r
  rcases r with e | ⟨spec, sts⟩
  · cases fullDrawO x f <;> rfl
  · show outOf (@Gen.decisionAt SO true s spec sts x link c₁ mids f) =
      (outOf (decisionAt true s spec sts x link c₁ mids f)).bind _
    unfold Gen.decisionAt decisionAt
    rw [outOf_bind, outOf_bind, Claims.OC.bind_assoc']
    congr 1
    funext r'
    rcases r' with e | ⟨⟩
    · cases fullDrawO x f <;> rfl
    · show outOf (monadLift (@Gen.checks SO true s spec x link c₁ mids f)) =
        (outOf (monadLift (checks true s spec x link c₁ mids f))).bind _
      rw [checks_SO]
      unfold wrapO
      cases fullDrawO x f <;> rfl

/-- **The verifier's next message at `SO` is the outer verifier's** (`nextO`). -/
theorem next_SO (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    @Gen.next SO true x as cs = nextO x as cs := by
  unfold Gen.next nextO
  rw [act_SO]
  cases actO x as cs with
  | rej => rfl
  | coin c => rfl
  | decide s spec sts link c₁ mids f => exact outOf_decision_SO s x link c₁ mids f

/-- **The verifier at `SO` is the outer verifier** (`VO`). -/
theorem V_SO : @Gen.V SO true = VO := by
  unfold Gen.V VO
  congr 1
  · funext x
    rw [setupOf_SO]
    rcases setupO x with _ | ⟨i, spec, sts⟩ <;> rfl
  · funext x as cs
    exact next_SO x as cs

theorem readOf_SO : @Gen.readOf SO true = readOfO := by
  funext x
  show ((insO x).map RecIns.base).map Ins.read = (insO x).map fun s => s.base.read
  rw [Option.map_map]
  rfl

theorem runOf_SO (s : Claims.Seen) (j : ℕ) : @Gen.runOf SO true s j = runOfO s j := by
  unfold Gen.runOf runOfO
  cases lastFinal s with
  | none => rfl
  | some f =>
    show (do let (_, spec, sts) ← @Gen.setupOf SO true f.stmt; _) = (do let (_, spec, sts) ← setupO f.stmt; _)
    rw [setupOf_SO]
    rcases setupO f.stmt with _ | ⟨i, spec, sts⟩ <;> rfl

theorem cols_SO : @Gen.cols SO true = colsO := by
  funext s j
  unfold Gen.cols colsO
  rw [runOf_SO]
  cases runOfO s j <;> rfl

theorem tables_SO (x : Claims.Msg) : @Gen.tables SO true x = tablesO x := by
  unfold Gen.tables tablesO
  rw [setupOf_SO]
  rcases setupO x with _ | ⟨i, spec, sts⟩ <;> rfl

/-- **The masked chain's instance at `SO` is the outer verifier's record** (`deployZkRec`). -/
theorem deployZk_SO (rel : Readable.Parsed → Readable.Relation) : @Gen.deployZk SO rel = deployZkRec rel := by
  have hε : @Gen.εtable SO true = εtableO := funext fun x => by unfold Gen.εtable εtableO; rw [tables_SO]
  have hN : @Gen.N₀ SO true = N₀O := funext fun x => by unfold Gen.N₀ N₀O; rw [tables_SO]
  have hw : @Gen.weight SO true = weightO := funext fun x => by unfold Gen.weight weightO; rw [tables_SO]
  unfold Gen.deployZk deployZkRec Gen.parse Gen.scopeCheck Gen.table
  rw [V_SO, readOf_SO, cols_SO, hε, hN, hw]

/-- **Masked knowledge soundness at the outer verifier**, at SHA-512 at the deployed relation, with no premise: the
masked chain's (`KS.Gen.zk_KS_untrim`) at the outer verifier's setup (`SO`), whose instance is `deployZkRec`
(`deployZk_SO`). -/
theorem zk_KS_outer : Claims.Flock.KS sha512 (RecDeploy.deployZkRec Readable.relation) := by
  rw [← deployZk_SO]
  exact @Gen.zk_KS_untrim SO

end FlockDeploy.KS
