import Proofs.Flock.Claims.Recursion.LayoutBTops
import Proofs.Flock.Claims.Recursion.VStarCross

/-!
# The plain setup makes a call no recorded part makes

`calls_layoutBTops_tagged` puts every call `layoutBTops` records at byte 7 `h` or `s` (`HmTagged`, hm96's prefixes).
The inner statement's setup, wherever it is `.ok` at `H`, makes a call that is neither (`setupM_log_untagged`): its
session digest `σ = SHA-512(sigmaTag ‖ …)`, whose tag `verity/flock-circuit/sigma` has `f` at byte 7. So a premise
asking every call of an accepted inner run to be a recorded one, or a collision among them, asks at every accepted run
with a setup for a collision among the recorded calls (`setupM_not_layoutBTops`), which `CallsLogCollAt` does not: it
searches the setup's calls too (`Deploy.callsAt`).

The proof runs the setup on a poisoned oracle (`poison`), SHA-512 refusing every input `p` holds of. Off a run whose
log `p` holds of nowhere, the poisoned run is the `OC` run (`poisonSim`, one more simulation of `Flock.With.Sim`, at
`setupTablesWith_sim`). With `p` the inputs not `HmTagged`, a hidden-output setup refuses at `σ`
(`setupHiddenWith_poison`), and so does every session of such tables (`setupTablesWith_poison`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)

/-- SHA-512 as `h`, refusing every input `p` holds of. -/
def poison (p : ByteArray → Bool) (h : ByteArray → ByteArray) : Hashes Flock.V :=
  { Hashes.ofSha512 Sha512.hash with sha512 := fun b => cond (p b) (.error "poison") (.ok (h b)) }

/-- An `OCV` run against `H` whose log `p` holds of nowhere is a run with `poison p (hOf H)`. -/
def poisonSim (p : ByteArray → Bool) (H : Msg → Msg) : With.Sim Claims.OCV Flock.V where
  R x y := (∀ e ∈ (x.run.eval H).2, p (Claims.ofMsg e.1) = false) → y = (x.run.eval H).1
  pure _ _ := rfl
  bind {α β x y f g} hxy hfg hn := by
    rw [ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind] at hn ⊢
    have hy := hxy fun e he => hn e (List.mem_append_left _ he)
    subst hy
    rcases hxr : x.run.eval H with ⟨rx, lx⟩
    rw [hxr] at hn
    cases rx with
    | error e => rfl
    | ok a => exact hfg a fun e he => hn e (List.mem_append_right _ he)
  throw _ _ := rfl
  lift _ _ := rfl

theorem poisonSim_hashes (p : ByteArray → Bool) (H : Msg → Msg) :
    (poisonSim p H).Hashes Claims.Hashes.oc (poison p (Claims.hOf H)) where
  sha512 b hn := by
    have hb : p (Claims.ofMsg (Claims.toMsg b)) = false := hn (Claims.toMsg b, H (Claims.toMsg b)) (by
      show _ ∈ (Claims.OC.eval H
        (Claims.OC.query (Claims.toMsg b) fun a => Claims.OC.ret (Except.ok (Claims.ofMsg a)))).2
      simp [Claims.OC.eval])
    rw [Claims.ofMsg_toMsg] at hb
    show cond (p b) _ _ = _
    rw [hb]
    rfl
  sha256 _ _ := rfl
  blake3 _ _ := rfl
  blake3Keyed _ _ _ := rfl

/-- **The setup over `OC` against `H`, off a log `p` holds of nowhere, is the setup with `poison p (hOf H)`.** -/
theorem setupM_poison (p : ByteArray → Bool) (H : Msg → Msg) (r : Readable.Read)
    (hn : ∀ e ∈ ((FlockDeploy.setupM Claims.Hashes.oc r).run.eval H).2, p (Claims.ofMsg e.1) = false) :
    FlockDeploy.setupM (poison p (Claims.hOf H)) r = FlockDeploy.setupWith (Claims.hOf H) r :=
  ((With.setupTablesWith_sim (poisonSim p H) With.Throws.exceptT With.Throws.v (poisonSim_hashes p H) _ _ _ _ _ _ _
    _ _ _ _ _ _) hn).trans (FlockDeploy.setupM_eval H r).1

theorem ebind_ok' {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases hx : x with
  | error e => rw [hx] at h; cases h
  | ok a => rw [hx] at h; exact ⟨a, rfl, h⟩

theorem ite_ok {α : Type} {c : Prop} [Decidable c] {a b : Flock.V α} {v : α} :
    (if c then a else b) = .ok v ↔ c ∧ a = .ok v ∨ ¬ c ∧ b = .ok v := by
  by_cases hc : c <;> simp [hc]

theorem throw_ok {α : Type} {e : String} {v : α} (h : (throw e : Flock.V α) = .ok v) : False := nomatch h

instance instDecidablePredMsgHmTagged : DecidablePred HmTagged := fun m => by unfold HmTagged; infer_instance

/-- Not one of hm96's prefixes at byte 7. -/
def untagged (b : ByteArray) : Bool := decide (¬ HmTagged (Claims.toMsg b))

/-- An input that starts with a tag carrying `f` at byte 7 is not `HmTagged`. -/
theorem untagged_of_tag {t : String} (ht : 8 ≤ t.toUTF8.data.toList.length ∧ t.toUTF8.data.toList[7]? = some 102)
    (a b c : ByteArray) : untagged (t.toUTF8 ++ a ++ b ++ c) = true := by
  obtain ⟨h1, h2⟩ := ht
  simp only [untagged, decide_eq_true_eq, HmTagged, toMsg_append, List.append_assoc]
  rw [List.getElem?_append_left (by simp only [Claims.toMsg]; omega),
    show (Claims.toMsg t.toUTF8)[7]? = some 102 from h2]
  simp

set_option linter.unusedTactic false in
set_option maxHeartbeats 4000000 in
/-- **A hidden-output setup refuses on an oracle refusing its `σ` input.** -/
theorem setupHiddenWith_poison {p : ByteArray → Bool} {h : ByteArray → ByteArray} {sig : String}
    (hp : ∀ a b c : ByteArray, p (sig.toUTF8 ++ a ++ b ++ c) = true) {tags : Tags} (htag : tags.sigmaTag = sig)
    (circuitFile publicFile : ByteArray) (coins : Coins) (tables : List (String × ByteArray))
    (draw partition : Option Lean.Json) (program publicInputs : Option ByteArray) {st : Stmt}
    (hs : Stmt.setupHiddenWith (poison p h) tags circuitFile publicFile coins tables draw partition program
      publicInputs = .ok st) : False := by
  unfold Stmt.setupHiddenWith at hs
  dsimp only at hs
  repeat' (first
    | (exact throw_ok hs)
    | (obtain ⟨_, hx, hs⟩ := ebind_ok' hs
       try (cases hx; done)
       try (simp only [poison, htag, hp, Bool.cond_true] at hx; cases hx; done))
    | (rw [ite_ok] at hs
       rcases hs with ⟨_, hs⟩ | ⟨_, hs⟩)
    | (split at hs))

theorem mapM_range_ne_ok {α : Type} {n : ℕ} (hn : 0 < n) {f : ℕ → Flock.V α} (hf : ∀ a, f 0 ≠ .ok a) {r : Array α}
    (h : (Array.range n).mapM f = .ok r) : False := by
  obtain ⟨k, rfl⟩ : ∃ k, n = k + 1 := ⟨n - 1, by omega⟩
  rw [Array.mapM_eq_mapM_toList, Array.toList_range, List.range_succ_eq_map, List.mapM_cons] at h
  cases h0 : f 0 with
  | error e => rw [h0] at h; cases h
  | ok a => exact hf a h0

set_option maxHeartbeats 4000000 in
/-- **A session of hidden-output tables refuses on an oracle refusing their `σ` input**: one table's setup is
`setupHiddenWith`, and several tables' first is. -/
theorem setupTablesWith_poison {p : ByteArray → Bool} {h : ByteArray → ByteArray} {sig : String}
    (hp : ∀ a b c : ByteArray, p (sig.toUTF8 ++ a ++ b ++ c) = true) {tags : Tags} (htag : tags.sigmaTag = sig)
    (hhid : tags.hiddenOutputs = true) (circuitFile publicFile : ByteArray) (coins : Coins) (count : ℕ)
    (tables : List (String × ByteArray)) (draw partition : Option Lean.Json) (program : Option ByteArray)
    (work : Option (ℕ × Array (String × ℕ × ℕ))) (stratified : Option ℕ) (closure : Option (Std.HashMap ℕ (Array ℕ)))
    (publicInputs : Option ByteArray) {v : SessionSpec × Array Stmt}
    (hs : Stmt.setupTablesWith (poison p h) tags circuitFile publicFile coins count tables draw partition program work
      stratified closure publicInputs = .ok v) : False := by
  unfold Stmt.setupTablesWith at hs
  dsimp only at hs
  repeat' (first
    | (exact throw_ok hs)
    | (obtain ⟨_, hx, hs⟩ := ebind_ok' hs
       try (cases hx; done)
       try (simp only [hhid, ↓reduceIte] at hx
            obtain ⟨_, hx, -⟩ := ebind_ok' hx
            refine setupHiddenWith_poison hp ?_ _ _ _ _ _ _ _ _ hx; exact htag)
       try (refine mapM_range_ne_ok (by omega) (fun a ha => ?_) hx
            simp only [hhid, ↓reduceIte] at ha
            obtain ⟨_, ha, -⟩ := ebind_ok' ha
            refine setupHiddenWith_poison hp ?_ _ _ _ _ _ _ _ _ ha
            exact htag))
    | (rw [ite_ok] at hs
       rcases hs with ⟨_, hs⟩ | ⟨_, hs⟩)
    | (split at hs))

/-- **The setup's calls include one that is not `HmTagged`**, wherever it is `.ok` at `H` on hidden outputs at a `σ`
tag with `f` at byte 7: off that, its poisoned run is its run (`setupM_poison`) and refuses
(`setupTablesWith_poison`). -/
theorem setupM_log_untagged (H : Msg → Msg) (r : Readable.Read) (hhid : r.tags.hiddenOutputs = true)
    (hsig : 8 ≤ r.tags.sigmaTag.toUTF8.data.toList.length ∧ r.tags.sigmaTag.toUTF8.data.toList[7]? = some 102)
    {v : SessionSpec × Array Stmt} (hok : FlockDeploy.setupWith (Claims.hOf H) r = .ok v) :
    ∃ e ∈ ((FlockDeploy.setupM Claims.Hashes.oc r).run.eval H).2, ¬ HmTagged e.1 := by
  by_contra hn
  push Not at hn
  have h := setupM_poison untagged H r fun e he => by
    simp only [untagged, toMsg_ofMsg, decide_eq_false_iff_not, not_not]
    exact hn e he
  rw [hok] at h
  exact setupTablesWith_poison (untagged_of_tag hsig) rfl hhid _ _ _ _ _ _ _ _ _ _ _ _ h

/-- **No `layoutBTops` part records that call**: every call it records is `HmTagged` (`calls_layoutBTops_tagged`). -/
theorem setupM_not_layoutBTops (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (H : Msg → Msg)
    (r : Readable.Read) (hhid : r.tags.hiddenOutputs = true)
    (hsig : 8 ≤ r.tags.sigmaTag.toUTF8.data.toList.length ∧ r.tags.sigmaTag.toUTF8.data.toList[7]? = some 102)
    {v : SessionSpec × Array Stmt} (hok : FlockDeploy.setupWith (Claims.hOf H) r = .ok v) :
    ∃ e ∈ ((FlockDeploy.setupM Claims.Hashes.oc r).run.eval H).2,
      ∀ (x : Msg) (s : ℕ) (w : Claims.Flock.Wit), e ∉ (layoutBTops fin).calls x s w := by
  obtain ⟨e, he, hn⟩ := setupM_log_untagged H r hhid hsig hok
  exact ⟨e, he, fun x s w hm => hn (calls_layoutBTops_tagged fin x s w e hm)⟩

/-- The inner tags meet both: hidden outputs, and `verity/flock-circuit/sigma`. -/
theorem innerTags_untagged : RecVStar.innerTags.hiddenOutputs = true ∧
    8 ≤ RecVStar.innerTags.sigmaTag.toUTF8.data.toList.length ∧
      RecVStar.innerTags.sigmaTag.toUTF8.data.toList[7]? = some 102 := by
  decide

/-- **`S(x_I)` holds a call no part records**: where the inner statement sets up on hidden outputs at a `σ` tag with
`f` at byte 7, its setup is `.ok` at SHA-512 (`setupWith_sha512_of`), so one of the inputs `innerSetupIns` lists, the
ones `Deploy.callsAt` searches beside the recorded calls, is not `HmTagged` (`setupM_log_untagged`). -/
theorem innerSetupIns_untagged {xI : Msg} {s : FlockDeploy.Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : FlockDeploy.setupOf false xI = some (s, spec, sts)) (hhid : s.read.tags.hiddenOutputs = true)
    (hsig : 8 ≤ s.read.tags.sigmaTag.toUTF8.data.toList.length ∧
      s.read.tags.sigmaTag.toUTF8.data.toList[7]? = some 102) :
    ∃ u ∈ innerSetupIns xI, ¬ HmTagged u := by
  obtain ⟨e, he, hn⟩ := setupM_log_untagged FlockDeploy.sha512 s.read hhid hsig (setupWith_sha512_of h)
  refine ⟨e.1, ?_, hn⟩
  unfold innerSetupIns
  rw [h]
  exact List.mem_map_of_mem he

end RecDeploy
