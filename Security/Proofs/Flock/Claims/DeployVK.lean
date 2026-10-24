import Proofs.Flock.Claims.DeployE

/-!
# The deployed plain verifier and the element-level one as one verifier over `keep` (`FlockDeploy.VK`)

`VK keep` is `FlockDeploy.V false` with its verdict read through `Flock.verifyWithMK keep`, which reads each comparison
of a record digest with the digest of an encoding through `keep` (`Flock/With/VerifyE.lean`). Its rounds, its step's
case split (`act false`), its coins' answers, its setup and its custody checks are `V false`'s:

* `VK id` is the deployed plain verifier (`VK_id`): `verdictM false` is `verifyWithM`, which is `verifyWithMK id`;
* `VK (fun _ => true)` is the element-level one (`VK_true`): `verifyWithME` is `verifyWithMK (fun _ => true)`.

So a statement proved at `VK keep` for every `keep` holds at both. `Flock.verifyWithK keep h` is `verifyWith h` at
`keep := id` and `verifyWithE h` at `keep := fun _ => true`, both by `rfl`.
-/

namespace Flock

/-- **The verdict over `V` with SHA-512 replaced by `h`, its digest comparisons read through `keep`**: `verifyWith h`
at `keep := id`, `verifyWithE h` at `keep := fun _ => true`. -/
def verifyWithK (keep : Bool → Bool) (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) : V Unit :=
  verifyWithMK (M := V) keep (Hashes.ofSha512 h) spec tabs record proofs

theorem verifyWithK_id (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : verifyWithK id h spec tabs record proofs = verifyWith h spec tabs record proofs :=
  rfl

theorem verifyWithK_true (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    verifyWithK (fun _ => true) h spec tabs record proofs = verifyWithE h spec tabs record proofs :=
  rfl

end Flock

namespace FlockDeploy

open Flock

/-- **The decision at a session, its verdict read through `keep`**: the verdict over `OC` on its tables, then the
plain custody checks. -/
def decisionAtK (keep : Bool → Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit := do
  verifyWithMK keep Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs
  monadLift (checks false s spec x link c₁ mids f)

/-- **The decision, its verdict read through `keep`**: the statement's setup over `OC`, then `decisionAtK`. -/
def decisionK (keep : Bool → Bool) (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAtK keep s spec sts x link c₁ mids f

/-- **The step, its verdict read through `keep`**: `next false`'s, with `decisionK keep`. -/
def nextK (keep : Bool → Bool) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  match act false x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionK keep s x link c₁ mids f)

/-- **The deployed plain verifier's rounds with the steps `nextK keep`.** -/
def VK (keep : Bool → Bool) : Claims.Verifier Coin where
  k := (V false).k
  next := nextK keep

theorem nextK_id : nextK id = next false := by
  funext x as cs
  unfold nextK next
  cases act false x as cs <;> rfl

theorem nextK_true : nextK (fun _ => true) = nextE := by
  funext x as cs
  unfold nextK nextE
  cases act false x as cs <;> rfl

/-- **`VK id` is the deployed plain verifier.** -/
theorem VK_id : VK id = V false := by
  show (⟨(V false).k, nextK id⟩ : Claims.Verifier Coin) = ⟨(V false).k, next false⟩
  rw [nextK_id]

/-- **`VK (fun _ => true)` is the element-level verifier.** -/
theorem VK_true : VK (fun _ => true) = VE := by
  show (⟨(V false).k, nextK (fun _ => true)⟩ : Claims.Verifier Coin) = ⟨(V false).k, nextE⟩
  rw [nextK_true]

/-- `VK keep` takes the deployed plain verifier's rounds. -/
theorem VK_k (keep : Bool → Bool) : (VK keep).k = (V false).k := rfl

end FlockDeploy
