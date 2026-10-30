import Proofs.Flock.Claims.Recursion.BoundDeploy
import Proofs.Flock.Claims.DeployEfficient

/-!
# (P1) `Bound.PlainKSAbort` fails at an element-level verifier `KSBoundRoute` quantifies over

`RecDeploy.KSBoundRoute` takes (P1), `Bound.PlainKSAbort`, at the deployed bound record
`bdDeployed rel nextE colsE wE …`, at every choice of the element-level verifier's steps `nextE`. At the steps that
accept whatever they read (`acceptAllE`), (P1) fails at every false statement in the deployed plain verifier's scope
(`not_plainKSAbort_acceptAll_in`):

* **Generic** (`Claims.Recursion.not_KSAbort_of_acc`): an inner protocol whose verifier accepts at its first round fails
  `Inner.KSAbort` at any statement with no witness and `κ < 1`. The prover that never aborts is accepted at every coin,
  `E^rep` finds no witness there, and the verifier calls `H` nowhere, so `B` finds no collision: `1 ≤ κ`.
* **At `DE`** (`not_plainKSAbort_acceptAll`): `deE rel acceptAllE colsE wE` keeps `deployPlainTrim rel`'s parser, code
  and numbers, so its relation and `κ` are `deployPlain rel`'s, its decoders `deployPlainTrim rel`'s, and its verifier
  takes `FlockDeploy.V false`'s rounds, at least one at every statement. The deployed plain radius decoder meets the spec
  at the trimmed code (`FlockDeploy.listWithin_plainTrim`).
* **In scope** (`not_plainKSAbort_acceptAll_in`): at a plain session in scope (`FlockDeploy.PlainIn`, at most `200`
  tables), `κ ≤ 200 · 2^{−202.98} < 1` (`FlockDeploy.κ_plain_le`).

So (P1) at every `nextE`, as `KSBoundRoute` quantifies it, is false wherever the deployed plain verifier has a false
statement in scope: `deE` leaves the element-level verifier's steps free, while its `κ` is `deployPlain rel`'s, fixed by
the deployed numbers. That is why (P1) stays a hypothesis of `KSBoundRoute`; at the deployed plain verifier's own steps
it is `RecDeploy.plainKSAbort_bd`'s statement, and at the deployed relation it holds with no premise
(`RecDeploy.plainKSAbort_deployed`).
-/

namespace Claims.Recursion

open Claims

/-! ## An inner verifier that accepts at its first round -/

/-- A verifier whose steps accept whatever they read accepts at the first round and calls `H` nowhere. -/
theorem stepWith_acc {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω) (vnext : List Msg → OC Out)
    (hv : ∀ as, vnext as = .ret .acc) :
    ∀ (n : ℕ) (hist : List (Msg × Msg)) (as : List Msg) (t : Trace), 0 < n →
      (stepWith H P x ω vnext n hist as t).out = .acc ∧ (stepWith H P x ω vnext n hist as t).vLog = t.vLog
  | 0, _, _, _, h => absurd h (Nat.lt_irrefl 0)
  | _ + 1, _, _, _, _ => by simp [stepWith, hv, OC.eval]

theorem run_acc {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x) (ω : Ω) (r : Fin (V.k x) → Coin) :
    (run H P V x ω r).out = .acc :=
  (stepWith_acc H P x ω _ (fun _ => hv _ _) (V.k x) [] [] ⟨[], [], [], .rej⟩ hk).1

theorem run_vLog_acc {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin) (x : Msg)
    (hv : ∀ as cs, V.next x as cs = .ret .acc) (hk : 0 < V.k x) (ω : Ω) (r : Fin (V.k x) → Coin) :
    (run H P V x ω r).vLog = [] :=
  (stepWith_acc H P x ω _ (fun _ => hv _ _) (V.k x) [] [] ⟨[], [], [], .rej⟩ hk).2

theorem abortRun_keep {Coin Ω A : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (ω : Ω) (t : (Fin (V.k x) → Coin) × A) :
    abortRun H P V x (fun _ _ _ => true) ω t = run H P V x ω t.1 := by
  simp [abortRun]

section Acc

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- At a statement with no witness, a prover that never aborts against a verifier that accepts at its first round
fails `E^rep` at every draw. -/
theorem prBadAbort_acc (H : Msg → Msg) (I : Inner Coin W) (x : Msg) (hv : ∀ as cs, I.V.next x as cs = .ret .acc)
    (hk : 0 < I.V.k x) (hx : ∀ w, ¬ I.R x w) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A]
    (P : Prover Ω) :
    I.PrBadAbort H x P (fun _ _ _ => true : Ω → (Fin (I.V.k x) → Coin) → A → Bool) = 1 := by
  unfold Inner.PrBadAbort
  refine (Pr_congr (q := fun _ => True) fun s => iff_true_intro ⟨?_, ?_⟩).trans (Pr_const_of trivial)
  · rw [abortRun_keep]
    exact run_acc H P I.V x hv hk _ _
  · cases ((erep (I.waits x) (I.k x) (I.cap x) I.opens (I.assemble x) (I.check x) s.2.1 s.2.2).eval H
      (abortRun H P I.V x (fun _ _ _ => true) s.1)).1 with
    | none => exact id
    | some w => exact hx w

/-- Neither run of `B`'s fork calls `H` on the verifier's side, so `B` finds nothing. -/
theorem prBAbort_acc (H : Msg → Msg) (I : Inner Coin W) (x : Msg) (hv : ∀ as cs, I.V.next x as cs = .ret .acc)
    (hk : 0 < I.V.k x) {Ω A : Type} [Fintype Ω] [Nonempty Ω] [Fintype A] [Nonempty A] (P : Prover Ω) :
    PrBAbort H I.V P x (fun _ _ _ => true : Ω → (Fin (I.V.k x) → Coin) → A → Bool) = 0 := by
  simp only [PrBAbort, hk, ↓reduceDIte, Finder.Pr]
  refine (Pr_congr fun s => iff_false_intro ?_).trans Pr_false
  rintro ⟨u, v, h, -⟩
  simp only [BAbort, abortRun_keep, run_vLog_acc H P I.V x hv hk, List.append_nil, findColl] at h
  cases h

/-- **`Inner.KSAbort` fails at an inner protocol whose verifier accepts at its first round**, wherever a statement has
no witness and `κ < 1`: the prover that never aborts is accepted at every coin, `E^rep` finds no witness, and `B` finds
no collision, so `1 ≤ κ`. -/
theorem not_KSAbort_of_acc (H : Msg → Msg) (I : Inner Coin W) (x : Msg) (hv : ∀ as cs, I.V.next x as cs = .ret .acc)
    (hk : 0 < I.V.k x) (hx : ∀ w, ¬ I.R x w) (hκ : I.κ x < 1) : ¬ I.KSAbort H := by
  intro h
  have h1 := h x Unit ⟨fun _ _ _ => .ret []⟩ Unit fun _ _ _ => true
  rw [prBadAbort_acc H I x hv hk hx, prBAbort_acc H I x hv hk, mul_zero, Real.sqrt_zero, add_zero] at h1
  linarith

end Acc

end Claims.Recursion

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

/-- The element-level verifier's steps that accept whatever they read. -/
def acceptAllE : Msg → List Msg → List Coin → Claims.OC Out := fun _ _ _ => .ret .acc

/-- **(P1) fails at `acceptAllE`** at every statement with no witness in the deployed plain relation where the deployed
plain `κ` is below `1`, whatever the code side's other choices. -/
theorem not_plainKSAbort_acceptAll (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) (x : Msg)
    (hx : ∀ w, ¬ Claims.Flock.R (FlockDeploy.deployPlain rel).parse x w)
    (hκ : Claims.Flock.κ (FlockDeploy.deployPlain rel) x < 1) :
    ¬ (bdDeployed rel acceptAllE colsE wE join rowIn final w).PlainKSAbort FlockDeploy.sha512 := fun h =>
  Claims.Recursion.not_KSAbort_of_acc FlockDeploy.sha512
    (Claims.Recursion.Inner.ofFlock (deE rel acceptAllE colsE wE) Claims.GS.plainDecodeRD) x (fun _ _ => rfl)
    (V_k_pos_flock false x) hx hκ (h Claims.GS.plainDecodeRD (FlockDeploy.listWithin_plainTrim rel))

/-- **(P1) fails at `acceptAllE` at every false statement in the deployed plain verifier's scope**: a plain session
that sets up with at most `200` tables at the deployed parameters (`FlockDeploy.PlainIn`) and has no witness. -/
theorem not_plainKSAbort_acceptAll_in (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ) {x : Msg} {s spec sts}
    (hin : FlockDeploy.PlainIn x s spec sts) (hx : ∀ w, ¬ Claims.Flock.R (FlockDeploy.deployPlain rel).parse x w) :
    ¬ (bdDeployed rel acceptAllE colsE wE join rowIn final w).PlainKSAbort FlockDeploy.sha512 := by
  refine not_plainKSAbort_acceptAll rel colsE wE join rowIn final w x hx ?_
  have hs : (sts.size : ℝ) ≤ 200 := by exact_mod_cast hin.size_le
  have hp : (2 : ℝ) ^ (-(202.98 : ℝ)) ≤ 2 ^ (-((8 : ℕ) : ℝ)) :=
    Real.rpow_le_rpow_of_exponent_le (by norm_num) (by norm_num)
  rw [show (2 : ℝ) ^ (-((8 : ℕ) : ℝ)) = (2 ^ 8)⁻¹ by rw [Real.rpow_neg (by norm_num), Real.rpow_natCast]] at hp
  have h0 : (0 : ℝ) ≤ 2 ^ (-(202.98 : ℝ)) := by positivity
  calc Claims.Flock.κ (FlockDeploy.deployPlain rel) x ≤ sts.size * 2 ^ (-(202.98 : ℝ)) :=
        FlockDeploy.κ_plain_le rel hin
    _ ≤ 200 * (2 ^ 8)⁻¹ := mul_le_mul hs hp h0 (by norm_num)
    _ < 1 := by norm_num

end RecDeploy
