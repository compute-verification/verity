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

namespace Claims

open _root_.Flock

/-- **The verdict over `OC` read through `keep`, against `H`, is `verifyWithK keep (hOf H)`.** -/
theorem verifyOCK_eval (keep : Bool → Bool) (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) :
    ((verifyWithMK keep Hashes.oc spec tabs record proofs).run.eval H).1 =
      verifyWithK keep (hOf H) spec tabs record proofs := by
  rw [← (With.verifyWithMK_sim (ocLog H) (ocLog_hashes H) keep spec tabs record proofs #[]).1]
  exact With.verifyWithMK_sim (With.Sim.forget _) (With.Sim.forget_logged _) keep spec tabs record proofs #[]

end Claims

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

theorem decisionAtK_eval (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec)
    (sts : Array Stmt) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) :
    ((decisionAtK keep s spec sts x link c₁ mids f).run.eval H).1 =
      (do verifyWithK keep (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
          checks false s spec x link c₁ mids f) := by
  have he := Claims.verifyOCK_eval keep H spec (tabsOf sts) f.record f.proofs
  simp only [decisionAtK, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verifyWithMK keep Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H with ⟨r, l⟩
  rw [hv] at he
  simp only at he
  subst he
  cases verifyWithK keep (Claims.hOf H) spec (tabsOf sts) f.record f.proofs <;> rfl

theorem decisionK_eval (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    ((decisionK keep s x link c₁ mids f).run.eval H).1 =
      (do let (spec, sts) ← setupWith (Claims.hOf H) s.read
          verifyWithK keep (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
          checks false s spec x link c₁ mids f) := by
  obtain ⟨he, -⟩ := setupM_eval H s.read
  simp only [decisionK, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at he
  simp only at he
  subst he
  cases setupWith (Claims.hOf H) s.read with
  | error e => rfl
  | ok p =>
    obtain ⟨spec, sts⟩ := p
    exact decisionAtK_eval keep H s spec sts x link c₁ mids f

theorem nextK_rej (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} (h : act false x as cs = .rej) : (nextK keep x as cs).eval H = (.rej, []) := by
  simp only [nextK, h, Claims.OC.eval]

theorem nextK_coin (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {c : Coin} (h : act false x as cs = .coin c) :
    (nextK keep x as cs).eval H = (.msg c.bytes, []) := by
  simp only [nextK, h, Claims.OC.eval]

/-- **The decision read through `keep`, against `H`**: it accepts exactly when the setup with `hOf H` is `.ok`, and at
its session `verifyWithK keep (hOf H)` and the custody checks are `.ok` (`next_decide` at `keep := id`, `nextE_decide`
at `keep := fun _ => true`). -/
theorem nextK_decide (keep : Bool → Bool) (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f) :
    ((nextK keep x as cs).eval H).1 = .acc ↔
      ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verifyWithK keep (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs = .ok () ∧
          checks false s spec' x link c₁ mids f = .ok () := by
  have he := decisionK_eval keep H s x link c₁ mids f
  simp only [nextK, h, outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hs : setupWith (Claims.hOf H) s.read with
  | error e => simp [bind, Except.bind]
  | ok p =>
    obtain ⟨spec', sts'⟩ := p
    cases hv : verifyWithK keep (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs with
    | error e => simp [bind, Except.bind, hv]
    | ok u =>
      cases hc : checks false s spec' x link c₁ mids f with
      | error e => simp [bind, Except.bind, hv, hc]
      | ok u' => simp [bind, Except.bind, hv, hc]

end FlockDeploy
