module

public import Proofs.Flock.Claims.Complete.ZkTables.HelloSize
import Proofs.Flock.Claims.Complete.ZkTables.Stmt
import Proofs.Flock.Claims.Bridge.Custody
import Proofs.Flock.Claims.Scope.Setup
import Proofs.Flock.Soundness.Discharge.HiddenExec.AcceptsP
import Proofs.Flock.Soundness.Discharge.HiddenExec.SetupP
import Proofs.Flock.Soundness.Discharge.Hidden.SpecP
import Proofs.Flock.Soundness.Refine.Walk

/-!
# The one-table masked session's setup, from `setupOf true x` alone

At a statement `x` whose masked setup returns one table (`setupOf true x = some (s, spec, sts)`, `sts.size = 1`), the
table's statement `st` has the facts the length bound reads (`setup_facts`): `25 ≤ m ≤ 35` (`Zk.shapeOk`, `Spec.m35`),
the `hm96` leaf (the tags' leaf scheme, which `Tags.byName?` pins at `hm96Rows`), `k_log ≤ m`, at most `2^10` regions
and `m_pts ≤ 64` (`checkInRange`), a keyed coin tree (`zkCoins` is a tree, and `setupHidden` keeps it:
`setupHidden_coinTree`) and a `Hello` that reads back (`hello_ok`). It holds for a statement with hidden outputs, at any
public inputs.
-/

@[expose] public section

open _root_.Flock

namespace FlockDeploy

theorem insOf_ins {zk : Bool} {x : Claims.Msg} {s : Ins} (h : insOf zk x = some s) :
    ∃ w, decode x = some w ∧ w.zk = zk ∧ w.ins = .ok s := by
  unfold insOf at h
  obtain ⟨w, hw, h2⟩ := Option.bind_eq_some_iff.1 h
  split_ifs at h2 with hz
  · cases h2
  · change w.ins.toOption = some s at h2
    refine ⟨w, hw, not_not.1 hz, ?_⟩
    cases hi : w.ins with
    | error e => rw [hi] at h2; cases h2
    | ok s' =>
      rw [hi] at h2
      cases h2
      rfl

theorem Raw.ins_tags {w : Raw} {s : Ins} (h : w.ins = .ok s) :
    ∃ t, Tags.byName? w.statement = some t ∧ s.read.I.tags = t ∧ (w.zk = true → t.plainLeaves = false) := by
  unfold Raw.ins at h
  simp only [bind, Except.bind, pure, Except.pure] at h
  repeat' split at h
  all_goals first | (cases h; done) | (cases h; exact ⟨_, by assumption, rfl, fun hz => by simpa [hz] using ‹¬ (_ && _) = true›⟩)

end FlockDeploy

namespace FlockComplete.ZkTables

open Flock FlockDeploy FlockComplete JsonRT
open Lean (Json)

/-- A statement of `Tags.byName?` with the hm96 rows and no plain leaves commits with `hm96-sha512/v1`. -/
theorem byName_hm96 {n : String} {t : Tags} (h : Tags.byName? n = some t) (hp : t.plainLeaves = false)
    (h9 : t.hm96Rows = true) : t.merkleLeaf = "hm96-sha512/v1" := by
  have hall : ∀ t ∈ Tags.all, t.plainLeaves = false → t.hm96Rows = true → t.merkleLeaf = "hm96-sha512/v1" := by
    decide +kernel
  have htm : ∀ t ∈ Tags.templated, t.plainLeaves = false → t.hm96Rows = true → t.merkleLeaf = "hm96-sha512/v1" := by
    decide +kernel
  unfold Tags.byName? at h
  cases hf : Tags.find? n with
  | some t' =>
    rw [hf] at h
    change some t' = some t at h
    cases h
    unfold Tags.find? at hf
    exact hall t' (List.mem_of_find?_eq_some hf) hp h9
  | none =>
    rw [hf] at h
    change Tags.templated.find? (·.name == n) = some t at h
    exact htm t (List.mem_of_find?_eq_some h) hp h9

open _root_.Flock FlockSoundness.Refine in
set_option maxHeartbeats 4000000 in
/-- **A statement `Stmt.setupHidden` returns keeps the coin tree its setup was given.** -/
theorem setupHidden_coinTree {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program own : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program own = .ok st) :
    st.coinTree = coins.tree? := by
  cases draw <;> unfold Stmt.setupHidden at h <;> dsimp only at h <;> simp only [throw_bind', pure_bind] at h <;>
    repeat' walk_step
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    rfl

end FlockComplete.ZkTables
