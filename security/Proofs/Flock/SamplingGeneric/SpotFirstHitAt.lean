module

public import Proofs.Flock.SamplingGeneric.SpotFirstHit
public import Proofs.Flock.SamplingGeneric.TwoFirstHitAt
public import Proofs.Flock.SpotFlock.RevealsOpen
public import Proofs.Flock.SpotFlock.RevealsLe
import Proofs.Flock.Claims.Prob

/-!
# The first-hit form at a budget a run, by gate, at SPOT[Flock]'s code

`CompiledTwo.KSFirstHitTwoX` takes two costs with the three bounds that make them costs. By gate, the bounds follow
from the two by-position first-hit bounds and the cost at a budget: where each run calls SHA-512 at most `T` times,
both parties', `c_R = Compiled.runCost` and `c_F = Compiled.firstHitCost` meet them under the linear expected-time
bound (`ExpectedCR`) at `B_conf,X` and at the two by-position finders of every cap
(`CompiledTwo.twoXWithin_at`). The form then holds at the error

`κ_s + √(k·γ_s·Pr[B]) + (2·(T·(1 + e_s·N_s) + n_PU·openCalls) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m)`
`+ (1 + K_s)·n_PU·openCalls))/2^256`.

`SpotFlock.firstHitTwo_at` is that at SPOT[Flock]'s two-stage code (`SpotFlock.twoZk`), from
`Flock.SecurityProofs.SpotFirstHitTwo`: the gates a proof block needs are the code's (`Reveal.needs`, by
`reveal_needs_spot`), and the opening checks' count is the code's (`revealsLe`). What stays a hypothesis there: the
budget, the three counts (`N_s`, `K_s`, `K_m`, on the coins alone) and the linear expected-time bound at the three
finders.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition footprintAt)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The event of `KSRegTwoX`, its threshold form and its headline at one error `ε`**, for one prover on one
statement: what `KSFirstHitTwoX` concludes at the error of its two costs. -/
def TwoXWithin (H : Msg → Msg) (L : CompiledTwo Coin A) (x : Msg) (s : AuditStatement) (Pt : Partition)
    (d : Sampling.Design.Two) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (ε : ℝ) : Prop :=
  Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX H P x p.1 p.2) ≤ ε ∧
    (∀ δ : ℝ, 0 ≤ δ →
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∀ ι : ℕ → ℕ → Bool, footprintAt s Pt d.part (L.regValX H P x) ι ∉ d.M Pt δ) ≤ δ + ε) ∧
    ∀ pb : ℝ, (∀ ι, d.Pa Pt (footprintAt s Pt d.part (L.regValX H P x) ι) ≤ pb) →
      Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc) ≤ pb + ε

/-- **The first-hit form at a budget `T` a run, by gate.** Where the form holds (`hK`), on a statement in scope, a
reveal for proof block `u` opens only gates of `needs u`, every run calls `H` at most `T` times, both parties' calls
counted (`hT`), the opening checks call `H` at most `openCalls` times a slot (`hopen`), `N_s` is at least the expected
number of sessions in a run's sample, `K_s` at least the expected number of gates the sample needs and `K_m` at least
the expected number of gates its sessions need, session by session, and `B_conf,X` and the two by-position finders of
every cap meet the linear expected-time bound, the event of `KSRegTwoX`, its threshold form and its headline hold at
the error `εfirstTwoX` of `c_R = runCost` and `c_F = firstHitCost`. -/
theorem twoXWithin_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hK : L.KSFirstHitTwoX H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hx : L.Scope x) (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u) (T : ℕ)
    (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      ((L.toCompiled.sampleNeeds x needs (L.checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin => (L.toCompiled.sessCount x needs (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXAt H P x s needs K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderIAt H P x s d.part needs K)) :
    L.TwoXWithin H x s Pt d P
      (L.εfirstTwoX H x P (L.toCompiled.runCost x T Ns) (L.toCompiled.firstHitCost x T Ns Ks Km)) := by
  have hc := Compiled.ex_calls_le_at hS.toCostContract hopen hs P needs T hT hNs hKs hKm
  have hdiv := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 2))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ))
  have hR := Compiled.bconfX_le_at hS.toCostContract hopen x P T hT hNs hB
  have hBm := (Verity.FirstHit.le_of_forall_cap fun K : ℕ =>
    (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
      (Compiled.pr_offRegX_le_needs_at_cap hbind hs P needs hpos K (hX K))).trans hdiv
  have hNm := (Verity.FirstHit.le_of_forall_cap fun K : ℕ =>
    (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
      (pr_offIreg_le_needs_at_cap hbind hS hs d.part P needs hpos K (hI K))).trans hdiv
  exact hK x s Pt d hx hs hPt hd Ω P _ _ hR hBm hNm

end CompiledTwo

end Claims.SamplingGeneric

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **SPOT[Flock]'s compiled two-stage verifier** at `SpotFirstHitTwo`'s registration, code, interior roots, design,
replay sample and replay cap. -/
noncomputable abbrev twoZk (h : Held) :=
  compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots (two h) (rs h) (cap h)

/-- **SPOT[Flock]'s two-stage verifier at the first-hit error, at a budget `T` a run, by gate.** At a held state with
a built session frame whose reveals address it, on a statement in scope: where every run calls SHA-512 at most `T`
times, both parties' calls counted, `N_s` is at least the expected number of sessions in a run's sample, `K_s` at
least the expected number of gates the sample's proof blocks need opened (`Reveal.needs`) and `K_m` at least the
expected number of gates its sessions need, session by session, and `B_conf,X` and the two by-position first-hit
finders of every cap meet the linear expected-time bound at SHA-512, the event of `KSRegTwoX`, its threshold form and
its headline hold at `εfirstTwoX` of `c_R = runCost` and `c_F = firstHitCost`. The gates a reveal opens and the
opening checks' count are the code's, proved (`reveal_needs_spot`, `revealsLe`). -/
theorem firstHitTwo_at (h : Held) (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement} {Pt : Partition}
    {d : Design.Two} (hx : (twoZk h).Scope x) (hs : (twoZk h).parse x = some s)
    (hPt : (twoZk h).partition s = some Pt) (hd : (twoZk h).two s = some d) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P (twoZk h).V x ω r).pLog.length +
      (run FlockDeploy.sha512 P (twoZk h).V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin ((twoZk h).V.k x) → FlockDeploy.Coin =>
      (((twoZk h).toCompiled.firstSlots x ((twoZk h).checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin ((twoZk h).V.k x) → FlockDeploy.Coin =>
      (((twoZk h).toCompiled.sampleNeeds x (fun u => (Reveal.needs (partition h) s u).toFinset)
        ((twoZk h).checked x r)).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin ((twoZk h).V.k x) → FlockDeploy.Coin =>
      ((twoZk h).toCompiled.sessCount x (fun u => (Reveal.needs (partition h) s u).toFinset)
        ((twoZk h).checked x r) : ℝ)) ≤ Km)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 (twoZk h).toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 ((twoZk h).toCompiled.finderXAt FlockDeploy.sha512 P x s
      (fun u => (Reveal.needs (partition h) s u).toFinset) K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 ((twoZk h).finderIAt FlockDeploy.sha512 P x s d.part
      (fun u => (Reveal.needs (partition h) s u).toFinset) K)) :
    (twoZk h).TwoXWithin FlockDeploy.sha512 x s Pt d P
      ((twoZk h).εfirstTwoX FlockDeploy.sha512 x P ((twoZk h).toCompiled.runCost x T Ns)
        ((twoZk h).toCompiled.firstHitCost x T Ns Ks Km)) :=
  Claims.SamplingGeneric.CompiledTwo.twoXWithin_at (_root_.Flock.SecurityProofs.SpotFirstHitTwo h hB hS)
    (SamplingDeploy.commit_binds (code h) _)
    (sessionsTwo_spot h (sessionOpensTwo_of_frameIs h _ (embeds_layoutFor h) (frameIs_built h hB hS)))
    (SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h))
    hx hs hPt hd P (fun u => (Reveal.needs (partition h) s u).toFinset) (reveal_needs_spot h _ s) T hT hNs hKs
    hKm hBc hX hI

end SpotFlock
