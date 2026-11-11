module

public import Proofs.Flock.SamplingGeneric.SpotFirstHitAt
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
import Proofs.Flock.Claims.Prob

/-!
# The first-hit form at a budget a run, by class of gates, at SPOT[Flock]'s code

`CompiledTwo.twoXWithin_at` discharges the two costs of `CompiledTwo.KSFirstHitTwoX` by gate. Here they are discharged
by class of gates: where a run's extractions open a class whole or not at all, under the statement's roots (`hallX`)
and under a replay block's interior roots (`hallI`), the two by-class first-hit bounds and the cost by class
(`Compiled.ex_calls_le_at_cls`) give `c_F = Compiled.firstHitCost` at the classes' counts, under the linear
expected-time bound (`ExpectedCR`) at `B_conf,X` and at the two by-class finders of every cap
(`CompiledTwo.twoXWithin_cls_at`). The form then holds at the error

`κ_s + √(k·γ_s·Pr[B]) + (2·(T·(1 + e_s·N_s) + n_PU·openCalls) + 4·(T·(1 + e_s·N_s + K_s + e_s·K_m)`
`+ (1 + K_s)·n_PU·openCalls))/2^256`,

with `K_s` at least the expected number of classes of the gates a run's sample needs and `K_m` at least the expected
number of the classes' representatives its sessions need, session by session.

`SpotFlock.firstHitTwo_cls_at` is that at SPOT[Flock]'s two-stage code (`SpotFlock.twoZk`), from
`Flock.SecurityProofs.SpotFirstHitTwo`, with the gates a proof block needs and the opening checks' count the code's.
What stays a hypothesis there: the classes with `hallX` and `hallI`, the budget, the three counts and the linear
expected-time bound at the three finders.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Partition)

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-- **The first-hit form at a budget `T` a run, by class of gates.** Where the form holds (`hK`), on a statement in
scope, a reveal for proof block `u` opens only gates of `needs u`, a run's extractions open a class of gates whole or
not at all under the statement's roots (`hallX`) and under a replay block's interior roots (`hallI`), every run calls
`H` at most `T` times, both parties' calls counted (`hT`), the opening checks call `H` at most `openCalls` times a
slot (`hopen`), `N_s` is at least the expected number of sessions in a run's sample, `K_s` at least the expected
number of classes of the gates the sample needs and `K_m` at least the expected number of the classes'
representatives its sessions need, session by session, and `B_conf,X` and the two by-class finders of every cap meet
the linear expected-time bound, the event of `KSRegTwoX`, its threshold form and its headline hold at the error
`εfirstTwoX` of `c_R = runCost` and `c_F = firstHitCost`. Every proof block that needs a gate needs its class's
representative (`_hrep`, which the proof doesn't read): under it a by-class finder's charge covers what it reads. -/
theorem twoXWithin_cls_at {H : Msg → Msg} {L : CompiledTwo Coin A} (hK : L.KSFirstHitTwoX H)
    (hbind : L.commit.Binds H) (hS : L.Sessions H) (hopen : L.toCompiled.OpenCallsLe H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hx : L.Scope x) (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (needs : ℕ → Finset ℕ) (hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u)
    (cls : Fin (L.nG x) → Fin (L.nG x))
    (_hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u)
    (hallX : ∀ (a : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x)) (g g' : Fin (L.nG x)), cls g = cls g' →
      (L.opensAt H P x a.1 a.2.1 a.2.2 g).isSome → (L.opensAt H P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin (L.V.k x) → Coin)) (ec : (Fin (L.V.k x) → Coin) × L.SlotEC x) (g g' : Fin (L.nG x)),
      cls g = cls g' → (L.iopTable H P x t.1 d.part t.2 ec g).isSome →
        (L.iopTable H P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ) (hT : ∀ ω r, (run H P L.V x ω r).pLog.length + (run H P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → Coin => ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hB : ExpectedCR H (Compiled.BconfX H L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR H (L.toCompiled.finderXCls H P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR H (L.finderICls H P x s d.part needs cls K)) :
    L.TwoXWithin H x s Pt d P
      (L.εfirstTwoX H x P (L.toCompiled.runCost x T Ns) (L.toCompiled.firstHitCost x T Ns Ks Km)) := by
  have hc := Compiled.ex_calls_le_at_cls hS.toCostContract hopen hs P needs cls T hT hNs hKs hKm
  have hdiv := div_le_div_of_nonneg_right (mul_le_mul_of_nonneg_left hc (by norm_num : (0 : ℝ) ≤ 2))
    (by positivity : (0 : ℝ) ≤ 2 ^ (256 : ℝ))
  have hR := Compiled.bconfX_le_at hS.toCostContract hopen x P T hT hNs hB
  have hBm := (Verity.FirstHit.le_of_forall_cap fun K : ℕ =>
    (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffRegX H P x q.1 q.2.1 q.2.2) fun _ h => h.2).trans
      (Compiled.pr_offRegX_le_cls_cap hbind hs P needs hpos cls hallX K (hX K))).trans hdiv
  have hNm := (Verity.FirstHit.le_of_forall_cap fun K : ℕ =>
    (Pr_mono (p := fun q : Ω × ((Fin (L.V.k x) → Coin) × L.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.OffIreg H P x q.1 d.part q.2.1 q.2.2) fun _ h => h.2).trans
      (pr_offIreg_le_cls_cap hbind hS hs d.part P needs hpos cls hallI K (hI K))).trans hdiv
  exact hK x s Pt d hx hs hPt hd Ω P _ _ hR hBm hNm

end CompiledTwo

end Claims.SamplingGeneric

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

/-- **SPOT[Flock]'s two-stage verifier at the first-hit error, at a budget `T` a run, by class of gates.** At a held
state with a built session frame whose reveals address it, on a statement in scope, for classes `cls` of the
program's gates that a run's extractions open whole or not at all (`hallX`, `hallI`): where every run calls SHA-512
at most `T` times, both parties' calls counted, `N_s` is at least the expected number of sessions in a run's sample,
`K_s` at least the expected number of classes of the gates the sample's proof blocks need opened (`Reveal.needs`) and
`K_m` at least the expected number of the classes' representatives its sessions need, session by session, and
`B_conf,X` and the two by-class first-hit finders of every cap meet the linear expected-time bound at SHA-512, the
event of `KSRegTwoX`, its threshold form and its headline hold at `εfirstTwoX` of `c_R = runCost` and
`c_F = firstHitCost`. The gates a reveal opens and the opening checks' count are the code's, proved. -/
theorem firstHitTwo_cls_at (h : Held) (hB : Built h) (hS : Addresses h) {x : Msg} {s : AuditStatement}
    {Pt : Partition} {d : Design.Two} (hx : (twoZk h).Scope x) (hs : (twoZk h).parse x = some s)
    (hPt : (twoZk h).partition s = some Pt) (hd : (twoZk h).two s = some d) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (cls : Fin ((twoZk h).nG x) → Fin ((twoZk h).nG x))
    (hrep : ∀ u (g : Fin ((twoZk h).nG x)), g.val ∈ (Reveal.needs (partition h) s u).toFinset →
      (cls g).val ∈ (Reveal.needs (partition h) s u).toFinset)
    (hallX : ∀ (a : Ω × ((Fin ((twoZk h).V.k x) → FlockDeploy.Coin) × (twoZk h).SlotEC x))
      (g g' : Fin ((twoZk h).nG x)), cls g = cls g' →
        ((twoZk h).opensAt FlockDeploy.sha512 P x a.1 a.2.1 a.2.2 g).isSome →
          ((twoZk h).opensAt FlockDeploy.sha512 P x a.1 a.2.1 a.2.2 g').isSome)
    (hallI : ∀ (t : Ω × (Fin ((twoZk h).V.k x) → FlockDeploy.Coin))
      (ec : (Fin ((twoZk h).V.k x) → FlockDeploy.Coin) × (twoZk h).SlotEC x) (g g' : Fin ((twoZk h).nG x)),
        cls g = cls g' → ((twoZk h).iopTable FlockDeploy.sha512 P x t.1 d.part t.2 ec g).isSome →
          ((twoZk h).iopTable FlockDeploy.sha512 P x t.1 d.part t.2 ec g').isSome)
    (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P (twoZk h).V x ω r).pLog.length +
      (run FlockDeploy.sha512 P (twoZk h).V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin ((twoZk h).V.k x) → FlockDeploy.Coin =>
      (((twoZk h).toCompiled.firstSlots x ((twoZk h).checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin ((twoZk h).V.k x) → FlockDeploy.Coin =>
      ((((twoZk h).toCompiled.sampleNeeds x (fun u => (Reveal.needs (partition h) s u).toFinset)
        ((twoZk h).checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin ((twoZk h).V.k x) → FlockDeploy.Coin =>
      ((twoZk h).toCompiled.sessCountIn x (fun u => (Reveal.needs (partition h) s u).toFinset)
        (fun l => ((twoZk h).toCompiled.sampleNeeds x (fun u => (Reveal.needs (partition h) s u).toFinset)
          l).image cls) ((twoZk h).checked x r) : ℝ)) ≤ Km)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 (twoZk h).toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 ((twoZk h).toCompiled.finderXCls FlockDeploy.sha512 P x s
      (fun u => (Reveal.needs (partition h) s u).toFinset) cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 ((twoZk h).finderICls FlockDeploy.sha512 P x s d.part
      (fun u => (Reveal.needs (partition h) s u).toFinset) cls K)) :
    (twoZk h).TwoXWithin FlockDeploy.sha512 x s Pt d P
      ((twoZk h).εfirstTwoX FlockDeploy.sha512 x P ((twoZk h).toCompiled.runCost x T Ns)
        ((twoZk h).toCompiled.firstHitCost x T Ns Ks Km)) :=
  Claims.SamplingGeneric.CompiledTwo.twoXWithin_cls_at (_root_.Flock.SecurityProofs.SpotFirstHitTwo h hB hS)
    (SamplingDeploy.commit_binds (code h) _)
    (sessionsTwo_spot h (sessionOpensTwo_of_frameIs h _ (embeds_layoutFor h) (frameIs_built h hB hS)))
    (SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h))
    hx hs hPt hd P (fun u => (Reveal.needs (partition h) s u).toFinset) (reveal_needs_spot h _ s) cls hrep hallX
    hallI T hT hNs hKs hKm hBc hX hI

end SpotFlock
