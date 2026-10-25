import Proofs.Flock.Claims.Deploy
import Flock.With.VerifyE

/-!
# The element-level inner verifier at the deployment (`FlockDeploy.VE`)

`FlockDeploy.V false` with its verdict replaced by the element-level one (`Flock.verifyWithME`: the shipped verdict's
text with its digests of encodings dropped, S5–S7, S12's digest width and retained bytes' digest, S13, S14 and S17).
Everything else is the deployed plain verifier's: the rounds (`k`), the step's case split (`act`), the coins' answers,
the statement's setup over `OC` and the custody checks after the verdict (`checks`). It is the inner verifier's steps
the recursion's bound record reads (`RecDeploy.bdDeployed`'s `nextE`), and it is never deployed.

**It accepts wherever the deployed plain verifier does** (`nextE_acc_of_next`, and on runs `run_out_VE`): at every
oracle `H`, on the same messages and coins. The element-level verdict keeps every check of the shipped one except the
digests of encodings, and every hash call (`Flock.verifyWithE_of_verifyWith`), so nothing that reads `VE`'s acceptance
reads one stronger than the deployed verifier's.
-/

namespace Claims

open _root_.Flock

/-- **The element-level verdict over `OC`, against `H`, is `verifyWithE (hOf H)`.** -/
theorem verifyOCE_eval (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithME Hashes.oc spec tabs record proofs).run.eval H).1 = verifyWithE (hOf H) spec tabs record proofs := by
  rw [← (With.verifyWithME_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).1]
  exact verifyLoggedE_fst _ _ _ _ _ _

/-- **Its oracle log is `verifyLoggedE`'s log**: the SHA-512 inputs, in call order. -/
theorem verifyOCE_log (H : Msg → Msg) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) :
    ((verifyWithME Hashes.oc spec tabs record proofs).run.eval H).2.map (fun p => ofMsg p.1) =
      (verifyLoggedE (hOf H) spec tabs record proofs).2.toList := by
  have := (With.verifyWithME_sim (ocLog H) (ocLog_hashes H) spec tabs record proofs #[]).2
  simp only [verifyLoggedE] at this ⊢
  rw [this]
  simp

end Claims

namespace FlockDeploy

open Flock

/-- **The element-level decision at a session**: the element-level verdict over `OC` on its tables, then the checks. -/
def decisionAtE (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit := do
  verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs
  monadLift (checks false s spec x link c₁ mids f)

/-- **The element-level decision**: the statement's setup over `OC`, then the element-level decision at its session. -/
def decisionE (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAtE s spec sts x link c₁ mids f

/-- **The element-level verifier's step**: `next false`'s, with the element-level decision. -/
def nextE (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  match act false x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionE s x link c₁ mids f)

/-- **The element-level inner verifier**: the deployed plain verifier's rounds, the element-level steps. -/
def VE : Claims.Verifier Coin where
  k := (V false).k
  next := nextE

theorem decisionAtE_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : Final) :
    ((decisionAtE s spec sts x link c₁ mids f).run.eval H).1 =
      (do verifyWithE (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
          checks false s spec x link c₁ mids f) := by
  have he := Claims.verifyOCE_eval H spec (tabsOf sts) f.record f.proofs
  simp only [decisionAtE, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verifyWithME Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H with ⟨r, l⟩
  rw [hv] at he
  simp only at he
  subst he
  cases verifyWithE (Claims.hOf H) spec (tabsOf sts) f.record f.proofs <;> rfl

theorem decisionE_eval (H : Claims.Msg → Claims.Msg) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    ((decisionE s x link c₁ mids f).run.eval H).1 =
      (do let (spec, sts) ← setupWith (Claims.hOf H) s.read
          verifyWithE (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
          checks false s spec x link c₁ mids f) := by
  obtain ⟨he, -⟩ := setupM_eval H s.read
  simp only [decisionE, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at he
  simp only at he
  subst he
  cases setupWith (Claims.hOf H) s.read with
  | error e => rfl
  | ok p =>
    obtain ⟨spec, sts⟩ := p
    exact decisionAtE_eval H s spec sts x link c₁ mids f

/-- **The element-level decision against `H`**: it accepts exactly when the setup with `hOf H` is `.ok`, and at its
session the element-level verdict with `hOf H` and the custody checks are `.ok`. -/
theorem nextE_decide (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act false x as cs = .decide s spec sts link c₁ mids f) :
    ((nextE x as cs).eval H).1 = .acc ↔
      ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verifyWithE (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs = .ok () ∧
          checks false s spec' x link c₁ mids f = .ok () := by
  have he := decisionE_eval H s x link c₁ mids f
  simp only [nextE, h, outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hs : setupWith (Claims.hOf H) s.read with
  | error e => simp [bind, Except.bind]
  | ok p =>
    obtain ⟨spec', sts'⟩ := p
    cases hv : verifyWithE (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs with
    | error e => simp [bind, Except.bind, hv]
    | ok u =>
      cases hc : checks false s spec' x link c₁ mids f with
      | error e => simp [bind, Except.bind, hv, hc]
      | ok u' => simp [bind, Except.bind, hv, hc]

/-- **The element-level verifier accepts wherever the deployed plain verifier does**: at every oracle, on the same
messages and coins. -/
theorem nextE_acc_of_next (H : Claims.Msg → Claims.Msg) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (h : ((next false x as cs).eval H).1 = .acc) : ((nextE x as cs).eval H).1 = .acc := by
  cases ha : act false x as cs with
  | rej => rw [next_rej H ha] at h; simp at h
  | coin c => rw [next_coin H ha] at h; simp at h
  | decide s spec sts link c₁ mids f =>
    obtain ⟨spec', sts', hs, hv, hc⟩ := (next_decide H ha).1 h
    exact (nextE_decide H ha).2 ⟨spec', sts', hs, verifyWithE_of_verifyWith _ _ _ _ _ hv, hc⟩

/-- **Where the deployed plain verifier answers, the element-level one answers alike**, with the same (empty) log. -/
theorem nextE_eval_of_msg (H : Claims.Msg → Claims.Msg) {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {m : Claims.Msg} (h : ((next false x as cs).eval H).1 = .msg m) :
    (nextE x as cs).eval H = (next false x as cs).eval H := by
  cases ha : act false x as cs with
  | rej => rw [next_rej H ha] at h; simp at h
  | coin c => simp only [nextE, next, ha]
  | decide s spec sts link c₁ mids f =>
    simp only [next, ha, outOf, Claims.OC.eval_bind, Claims.OC.eval] at h
    split at h <;> simp at h

theorem stepWith_out_VE {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (cs : List Claims.Msg → List Coin) :
    ∀ (n : ℕ) (hist : List (Claims.Msg × Claims.Msg)) (as : List Claims.Msg) (t : Claims.Trace),
      (Claims.stepWith H P x ω (fun as => next false x as (cs as)) n hist as t).out = .acc →
        (Claims.stepWith H P x ω (fun as => nextE x as (cs as)) n hist as t).out = .acc := by
  intro n
  induction n with
  | zero => intro hist as t h; simp [Claims.stepWith] at h
  | succ n ih =>
    intro hist as t h
    simp only [Claims.stepWith] at h ⊢
    generalize hv : (next false x (as ++ [((P.next x hist ω).eval H).1])
      (cs (as ++ [((P.next x hist ω).eval H).1]))).eval H = vb at h
    obtain ⟨b, l⟩ := vb
    cases b with
    | msg m =>
      have hm : ((next false x (as ++ [((P.next x hist ω).eval H).1])
          (cs (as ++ [((P.next x hist ω).eval H).1]))).eval H).1 = .msg m := by rw [hv]
      rw [nextE_eval_of_msg H hm, hv]
      exact ih _ _ _ h
    | acc =>
      have ha : ((next false x (as ++ [((P.next x hist ω).eval H).1])
          (cs (as ++ [((P.next x hist ω).eval H).1]))).eval H).1 = .acc := by rw [hv]
      have he := nextE_acc_of_next H _ _ _ ha
      generalize hw : (nextE x (as ++ [((P.next x hist ω).eval H).1])
        (cs (as ++ [((P.next x hist ω).eval H).1]))).eval H = wb at he
      obtain ⟨b', l'⟩ := wb
      simp only at he
      subst he
      rfl
    | rej => simp at h

/-- **The element-level verifier accepts every run the deployed plain verifier accepts**: at every oracle, prover,
statement, tape and coins (the two verifiers take the same rounds). -/
theorem run_out_VE {Ω : Type} (H : Claims.Msg → Claims.Msg) (P : Claims.Prover Ω) (x : Claims.Msg) (ω : Ω)
    (r : Fin ((V false).k x) → Coin) (h : (Claims.run H P (V false) x ω r).out = .acc) :
    (Claims.run H P VE x ω r).out = .acc :=
  stepWith_out_VE H P x ω (fun as => (List.range as.length).map (Claims.coinSeq r)) _ [] [] _ h

end FlockDeploy
