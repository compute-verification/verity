import Proofs.Flock.Claims.Sampling.Reference.DeployedCode

/-!
# The sampled verifier's sessions at the code with its Lean draw

At `SamplingDeploy.code c reg L bytes sr sess`, one proof session per registration proves every drawn unit: every unit
has the session `sess x` on the rounds `sr x`. So the slot at `S = 1` covers every checked unit (`slotsCover_code`),
and `SessionRuns`, `SessionOpens` and `InnerNumbers` are one statement about the run's one session each:

* `SessionRunsOne`: at coins that check some unit, the session is a Flock run on `sr x`; it gives `SessionRuns`
  (`sessionRuns_code`).
* `SessionOpensOne`: at an accepted run that checks some unit and shows the session, one Flock witness in its relation
  reveals, for every checked unit, openings that make it correct; it is `SessionOpens` (`sessionOpens_code_iff`).
* `innerNumbers_code_iff`: `InnerNumbers` is Flock's numbers at the one session's statement.
-/

namespace SamplingDeploy

open Claims Claims.Sampling

variable (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (c : Code) (reg : Msg → Option Reg)
  (L : Msg → ℕ) (bytes : (x : Msg) → (Fin (c.V.k x) → FlockDeploy.Coin) → Fin (L x) → Fin 256)
  (sr : (x : Msg) → Finset (Fin (c.V.k x))) (sess : Msg → Seen → Option (Msg × Seen))

/-- **One slot covers every checked unit**: every unit has the run's one session on its rounds. -/
theorem slotsCover_code : SlotsCover (deploy rel (code c reg L bytes sr sess)) (fun _ => 1) :=
  slotsCover_one _ (fun _ _ _ => rfl) (fun _ _ _ => rfl)

/-- **`SessionRuns` at one session**: at coins that check some unit, the session is a Flock run on `sr x`, of a Flock
prover fixed by the coins off it, and decides the run. -/
def SessionRunsOne (H : Msg → Msg) : Prop :=
  ∀ x (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (c.V.k x) → FlockDeploy.Coin),
    (code c reg L bytes sr sess).checked x r ≠ [] →
    ∃ (xS : Msg) (emb : Fin ((FlockDeploy.deployZk rel).V.k xS) → Fin (c.V.k x)) (P' : Prover Unit),
      StrictMono emb ∧ (∀ t, t ∈ sr x ↔ ∃ i, emb i = t) ∧
      (∀ r' : Fin (c.V.k x) → FlockDeploy.Coin, (∀ t ∉ sr x, r' t = r t) →
        ((run H P c.V x ω r').out = .acc ↔
          (run H P' (FlockDeploy.deployZk rel).V xS () fun i => r' (emb i)).out = .acc)) ∧
      ∀ r' : Fin (c.V.k x) → FlockDeploy.Coin, (∀ t ∉ sr x, (∀ i, t < emb i) → r' t = r t) →
        sess x (run H P c.V x ω r').seen =
            some (xS, (run H P' (FlockDeploy.deployZk rel).V xS () fun i => r' (emb i)).seen) ∧
          ∀ q ∈ (run H P' (FlockDeploy.deployZk rel).V xS () fun i => r' (emb i)).vLog, q ∈ (run H P c.V x ω r').vLog

/-- **`SessionRuns` from the one session's.** -/
theorem sessionRuns_code (H : Msg → Msg) (h : SessionRunsOne rel c reg L bytes sr sess H) :
    SessionRuns H (deploy rel (code c reg L bytes sr sess)) :=
  sessionRuns_of_slots (slotsCover_code rel c reg L bytes sr sess) fun x Ω P ω r p u hp hu =>
    h x Ω P ω r fun hn => by
      have h0 : p = 0 := Nat.lt_one_iff.1 hp
      subst h0
      change ((code c reg L bytes sr sess).checked x r)[0]? = some u at hu
      rw [hn] at hu
      simp at hu

/-- **`SessionOpens` at one session**: on a statement in scope, at an accepted run that checks some unit and shows the
session with statement `xS`, one Flock witness in its relation reveals, for every checked unit, openings that make that
unit correct. -/
def SessionOpensOne (H : Msg → Msg) (iroots : Msg → Seen → ℕ → List Msg) : Prop :=
  ∀ x s Pt (Ω : Type) (P : Prover Ω) ω (r : Fin (c.V.k x) → FlockDeploy.Coin) xS vS wF,
    c.parse x = some s → c.partition s.prog s.query = some Pt → InScope Pt s →
    (run H P c.V x ω r).out = .acc → sess x (run H P c.V x ω r).seen = some (xS, vS) →
    Claims.Flock.check (FlockDeploy.deployZk rel).parse xS wF = true →
    ∀ u, (∃ q : Fin (nPU (deploy rel (code c reg L bytes sr sess)) x),
        ((code c reg L bytes sr sess).checked x r)[q.val]? = some u) →
    ∃ rv a b, Pt.rus[u.1]? = some rv ∧ rv[u.2]? = some (a, b) ∧
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ c.reveal s u (run H P c.V x ω r).seen wF, (commit c).Opens H q.1 q.2 = true →
          (q.1 = s.roots → val q.2.pos = some q.2.val) ∧
          (q.1 = iroots x (run H P c.V x ω r).seen u.1 → ι q.2.pos = q.2.val)) →
        s.prog.Correct (RU.ref s.prog rv val ι) a b

/-- **`SessionOpens` is the one session's.** -/
theorem sessionOpens_code_iff (H : Msg → Msg) (iroots : Msg → Seen → ℕ → List Msg) :
    SessionOpens H (deploy rel (code c reg L bytes sr sess)) iroots ↔
      SessionOpensOne rel c reg L bytes sr sess H iroots := by
  constructor
  · intro h x s Pt Ω P ω r xS vS wF hs hP hin hacc hsess hck u hu
    exact h x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck
  · intro h x s Pt Ω P ω r u xS vS wF hs hP hin hacc hu hsess hck
    exact h x s Pt Ω P ω r xS vS wF hs hP hin hacc hsess hck u hu

/-- **`InnerNumbers` at one session**: Flock's numbers at the session's statement, whatever the run shows. -/
theorem innerNumbers_code_iff :
    InnerNumbers (deploy rel (code c reg L bytes sr sess)) ↔ ∀ x v xS vS, sess x v = some (xS, vS) →
      (FlockDeploy.deployZk rel).waits xS = Claims.Flock.Numbers.waits ∧
        (FlockDeploy.deployZk rel).cap xS = Claims.Flock.Numbers.cap ∧ (FlockDeploy.deployZk rel).N₀ xS = c.innerN₀ x ∧
        (FlockDeploy.deployZk rel).εtable xS ≤ c.tables x * 2 ^ (-(205 : ℝ)) ∧
        (FlockDeploy.deployZk rel).weight xS ≤ c.tables x * 2 ^ 33 :=
  innerNumbers_one_iff _ fun _ _ _ => rfl

end SamplingDeploy
