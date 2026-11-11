module

public import Proofs.Flock.SpotFlock.OpensClass
public import Proofs.Flock.SamplingGeneric.TwoFirstHitClassAt
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.Sampling.Assembled
import Proofs.Flock.SpotFlock.RevealsLe

/-!
# The by-class first-hit bound at a budget, at SPOT[Flock]'s code

`CompiledTwo.eventTwoX_le_cls_at_budget` bounds the event of `KSRegTwoX` by `CompiledTwo.εclsAt` for any compiled
two-stage verifier, a set of gates each proof block needs and classes of the gates that a run's extractions open whole
or not at all. `code_eventTwoX_le_cls_at_budget` is that bound at SPOT[Flock]'s compiled two-stage verifier, where

* a proof block needs the gates its reveal opens (`Reveal.needs`, `code_hpos`), and
* a gate's class is its key's (`gateKey`: the proof blocks that need the gate opened, the replay block that holds it,
  and whether it is on that block's boundary).

Under `Built` and `Addresses` at a statement in the two-stage scope, the code supplies Flock's knowledge soundness (the
masked record's), the commitment's binding, the sessions' contract, the opening checks' count, and the three
hypotheses on the classes: a proof block that needs a gate needs its class's representative (`gateKey_needs`), a class
lies in one replay block (`gateKey_block`), and a run's extractions open a class whole (`code_hwit`).

What the theorem still takes: the budget `T` a run, the three counts over the coins (the sample's sessions, the
classes of the gates the sample needs, and the classes its sessions need, session by session), and the linear
expected-time bound at `B_conf,X` and at the two by-class finders. Nothing here counts the classes.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The positions a proof block's reveal opens are the gates it needs**, at the code. -/
theorem code_hpos (s : AuditStatement) (u : ℕ) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.2.pos ∈ (Reveal.needs (partition h) s u).toFinset := by
  intro p hp
  obtain ⟨g, hg, hgp⟩ := (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hp
  have hpos : p.2.pos = g := congrArg (fun q : List Msg × Opening Bool => q.2.pos) hgp.symm
  rw [hpos]
  exact List.mem_toFinset.2 hg

/-- **The by-class first-hit bound at a budget, at the code and its key**: `CompiledTwo.eventTwoX_le_cls_at_budget`
at SPOT[Flock]'s compiled two-stage verifier `L`, with `needs` the gates each proof block's reveal opens and `cls` the
classes of `gateKey`, under `Built` and `Addresses` at a statement in the two-stage scope. Where every run calls
SHA-512 at most `T` times, both parties' calls counted (`hT`), `N_s` is at least the expected number of sessions in a
run's sample, `K_s` at least the expected number of classes of the gates the sample needs and `K_m` at least the
expected number of the classes' representatives its sessions need, session by session, and `B_conf,X` and the two
by-class finders of every cap meet the linear expected-time bound, an accepted run samples a proof block of
`FregTwoX` with probability at most `εclsAt`. -/
theorem code_eventTwoX_le_cls_at_budget (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω) {needs : ℕ → Finset ℕ}
    (hneeds : needs = fun u => (Reveal.needs (partition h) s u).toFinset) {cls : Fin (L.nG x) → Fin (L.nG x)}
    (hcls : cls = Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) (T : ℕ)
    (hT : ∀ ω r, (run FlockDeploy.sha512 P L.V x ω r).pLog.length +
      (run FlockDeploy.sha512 P L.V x ω r).vLog.length ≤ T) {Ns Ks Km : ℝ}
    (hNs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      ((L.toCompiled.firstSlots x (L.checked x r)).card : ℝ)) ≤ Ns)
    (hKs : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (((L.toCompiled.sampleNeeds x needs (L.checked x r)).image cls).card : ℝ)) ≤ Ks)
    (hKm : Ex (fun r : Fin (L.V.k x) → FlockDeploy.Coin =>
      (L.toCompiled.sessCountIn x needs (fun l => (L.toCompiled.sampleNeeds x needs l).image cls)
        (L.checked x r) : ℝ)) ≤ Km)
    (hBc : ExpectedCR FlockDeploy.sha512
      (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x))
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.toCompiled.finderXCls FlockDeploy.sha512 P x s needs cls K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.finderICls FlockDeploy.sha512 P x s d.part needs cls K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.εclsAt FlockDeploy.sha512 x P T Ns Ks Km := by
  have hA : (Claims.SamplingGeneric.Arg.ofFlock
      (FlockDeploy.deployZk _root_.Flock.Readable.relation)).KS FlockDeploy.sha512 :=
    Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _)
  have hbind : L.commit.Binds FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.commit_binds (code h) _
  have hSess : L.Sessions FlockDeploy.sha512 := by
    subst hL
    exact sessionsTwo_spot h
      (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS))
  have hopen : L.toCompiled.OpenCallsLe FlockDeploy.sha512 := by
    subst hL
    exact SamplingDeploy.openCallsLe_compiled _ (code h) iroots _ (revealsLe h)
  have hs : L.parse x = some s := by
    subst hL
    exact hx
  have hPt : L.partition s = some Pt := by
    subst hL
    exact hP
  have hd' : L.two s = some d := by
    subst hL
    exact hd
  have hpos : ∀ u v w, ∀ p ∈ L.reveal s u v w, p.2.pos ∈ needs u := by
    subst hL hneeds
    exact fun u v w => code_hpos h s u v w
  have hrep : ∀ u (g : Fin (L.nG x)), g.val ∈ needs u → (cls g).val ∈ needs u := by
    subst hneeds hcls
    intro u g hg
    exact List.mem_toFinset.2 (gateKey_needs h hP
      (Verity.FirstHit.key_clsOf (fun g : Fin (L.nG x) => gateKey h s Pt d g.val) g).symm (List.mem_toFinset.1 hg))
  have hblock : ∀ g g' : Fin (L.nG x), cls g = cls g' → gateBlock d.part g.val = gateBlock d.part g'.val := by
    subst hcls
    intro g g' hgg
    exact gateKey_block h ((Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg)
  have hkept : ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.SlotEC x),
      L.toCompiled.ClassWhole FlockDeploy.sha512 x cls (L.kept FlockDeploy.sha512 P x s ω r c) := by
    subst hcls
    exact fun ω r c => L.toCompiled.kept_classWhole_of_wit FlockDeploy.sha512 P x s ω _ r c
      (code_hwit h hB hS hL hx hP hd hin P ω r c)
  exact Claims.SamplingGeneric.CompiledTwo.eventTwoX_le_cls_at_budget hA hbind hSess hopen hs hPt hd' hin P needs
    hpos cls hrep
    (fun a g g' hgg hg =>
      L.toCompiled.opensAt_isSome_cls FlockDeploy.sha512 P x a.1 hs cls a.2.1 a.2.2 (hkept a.1 a.2.1 a.2.2) g g' hgg
        hg)
    (fun t ec g g' hgg hg =>
      L.iopTable_isSome_cls FlockDeploy.sha512 P x t.1 hs d.part cls hblock t.2 ec (hkept t.1 _ ec.2) g g' hgg hg)
    T hT hNs hKs hKm hBc hX hI

end SpotFlock
