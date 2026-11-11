module

public import Proofs.Flock.SpotFlock.OpensWhole
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHitClass
import Proofs.Flock.SamplingGeneric
import Proofs.Flock.Masked
import Proofs.Flock.Claims.Sampling.Assembled

/-!
# What the extracted witnesses reveal holds the classes of the code's key whole

The by-class first-hit bound (`CompiledTwo.eventTwoX_le_firstHit_key`) asks one thing of the code that instantiates
the reveal, the commitment and the sessions (`hwit`): at every run, for each session of the run's sample, extracted
at its first slot, each witness its extraction returns and each proof block of the session, what the witness reveals
there holds the classes of the key whole (`Compiled.ClassWhole`).

`code_hwit` is that hypothesis at SPOT[Flock]'s compiled two-stage verifier, for the key `gateKey` (the proof blocks
that need a gate opened, the replay block that holds it, and whether it is on that block's boundary), under `Built`
and `Addresses` at a statement in the two-stage scope. Each slot of the sample is its own session there, so the
session's proof blocks are the slot's own (`Compiled.sessBlocks_eq_singleton`), and `sessWit_reveal_key` is the rest.
The key's other two hypotheses are `gateKey_needs` and `gateKey_block`, and the positions a block's reveal opens are
the gates it needs (`code_hpos`).

`code_eventTwoX_le_firstHit_key` is the bound with all four in place: at the code and that key, under `Built` and
`Addresses`, what is left of its hypotheses is the linear expected-time bound at its two by-class finders.
-/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy

variable (h : Held)

/-- **The by-class bound's hypothesis on what the extracted witnesses reveal, at the code**: at SPOT[Flock]'s compiled
two-stage verifier `L`, under `Built` and `Addresses`, at a statement in the two-stage scope and every run, accepted
or refused, for each slot of the run's sample that is its session's first, each witness that session's extraction
returns there and each proof block of the session, what the witness reveals at the block holds the classes of
`gateKey` whole. -/
theorem code_hwit (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω) :
    ∀ (ω : Ω) (r : Fin (L.V.k x) → FlockDeploy.Coin) (c : L.SlotEC x),
      ∀ iu ∈ (List.finRange (L.nPU x)).zip (L.checked x r),
        ∀ hu : iu.2 < L.nPU x ∧ (L.sessSlots x (L.checked x r) iu.1).head? = some iu.1.val,
          ∀ w : Claims.Flock.Wit,
            ((L.sessWit x iu.2 r (run FlockDeploy.sha512 P L.V x ω r).seen (c iu.1 ⟨iu.2, hu.1⟩)).eval
                FlockDeploy.sha512 (run FlockDeploy.sha512 P L.V x ω)).1 = some w →
              ∀ u' ∈ L.sessBlocks x (L.checked x r) iu.1,
                L.toCompiled.ClassWhole FlockDeploy.sha512 x
                  (Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val)
                  (L.reveal s u' (run FlockDeploy.sha512 P L.V x ω r).seen w) := by
  subst hL
  intro ω r c iu hiu hu w hw u' hu'
  have hi := Claims.SamplingGeneric.getElem?_of_mem_zip_finRange (j := iu.1) (a := iu.2) hiu
  have hsb := Claims.SamplingGeneric.Compiled.sessBlocks_eq_singleton (by intro p q hpq; exact hpq) hi hu.1
  rw [hsb] at hu'
  have hu'' : u' = iu.2 := List.mem_singleton.1 hu'
  rw [hu'']
  intro p hp _ g g' hg hgg
  have hk : gateKey h s Pt d p.2.pos = gateKey h s Pt d g'.val := by
    rw [hg]
    exact (Verity.FirstHit.clsOf_eq_iff _ g g').1 hgg
  exact sessWit_reveal_key h hB hS hx hP hd hin P ω r (List.mem_of_getElem? hi) (c iu.1 ⟨iu.2, hu.1⟩) hw p hp
    g'.val hk

/-- **The positions a proof block's reveal opens are the gates it needs**, at the code. -/
theorem code_hpos (s : AuditStatement) (u : ℕ) (v : Seen) (w : Claims.Flock.Wit) :
    ∀ p ∈ (code h).reveal s u v w, p.2.pos ∈ (Reveal.needs (partition h) s u).toFinset := by
  intro p hp
  obtain ⟨g, hg, hgp⟩ := (Reveal.mem_reveal (partition h) (two h) (iroots []) (Reveal.frameAt h.frame)).1 hp
  have hpos : p.2.pos = g := congrArg (fun q : List Msg × Opening Bool => q.2.pos) hgp.symm
  rw [hpos]
  exact List.mem_toFinset.2 hg

/-- **The by-class first-hit bound at the code and its key**: `CompiledTwo.eventTwoX_le_firstHit_key` at SPOT[Flock]'s
compiled two-stage verifier `L`, the gates each proof block needs opened and the key `gateKey`, under `Built` and
`Addresses` at a statement in the two-stage scope. Flock's knowledge soundness is the masked record's, the commitment
binds, the sessions' contract is the code's, and the key's hypotheses are `code_hpos`, `gateKey_needs`,
`gateKey_block` and `code_hwit`. What is left is the linear expected-time bound at the two by-class finders (`hX`,
`hI`). The sum is over the classes of the gates the sample's proof blocks need; nothing here counts them. -/
theorem code_eventTwoX_le_firstHit_key (hB : Built h) (hS : Addresses h)
    {L : Claims.SamplingGeneric.CompiledTwo FlockDeploy.Coin
      (Claims.SamplingGeneric.Arg.ofFlock (FlockDeploy.deployZk _root_.Flock.Readable.relation))}
    (hL : L = SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (code h) iroots
      (two h) (rs h) (cap h))
    {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Design.Two} (hx : parse h x = some s)
    (hP : partition h s = some Pt) (hd : two h s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} [Fintype Ω]
    [Nonempty Ω] (P : Prover Ω)
    (hX : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.toCompiled.finderXCls FlockDeploy.sha512 P x s
      (fun u => (Reveal.needs (partition h) s u).toFinset)
      (Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) K))
    (hI : ∀ K : ℕ, ExpectedCR FlockDeploy.sha512 (L.finderICls FlockDeploy.sha512 P x s d.part
      (fun u => (Reveal.needs (partition h) s u).toFinset)
      (Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val) K)) :
    Pr (fun p : Ω × (Fin (L.V.k x) → FlockDeploy.Coin) =>
        (run FlockDeploy.sha512 P L.V x p.1 p.2).out = .acc ∧
          ∃ u ∈ L.checked x p.2, u ∈ L.FregTwoX FlockDeploy.sha512 P x p.1 p.2) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB FlockDeploy.sha512 L.V P x) +
          (Claims.SamplingGeneric.Compiled.BconfX FlockDeploy.sha512 L.toCompiled Ω P x).Pr FlockDeploy.sha512 +
        4 * Ex (fun q : Ω × ((Fin (L.V.k x) → FlockDeploy.Coin) × L.SlotEC x) =>
          ((L.callsAt FlockDeploy.sha512 P x s q.1 q.2.1 q.2.2 +
            ∑ c ∈ (Finset.univ.filter (fun g : Fin (L.nG x) =>
                decide (∃ u ∈ L.checked x q.2.1, g.val ∈ (Reveal.needs (partition h) s u).toFinset))).image
                  (Verity.FirstHit.clsOf fun g : Fin (L.nG x) => gateKey h s Pt d g.val),
              L.toCompiled.callsBy FlockDeploy.sha512 P x s (fun u => (Reveal.needs (partition h) s u).toFinset)
                c.val q.1 q.2.1 q.2.2 : ℕ) : ℝ)) /
          2 ^ (256 : ℝ) := by
  subst hL
  exact Claims.SamplingGeneric.CompiledTwo.eventTwoX_le_firstHit_key
    (Claims.SamplingGeneric.flockArgKS FlockDeploy.sha512 _ _root_.Flock.SecurityProofs.MaskedKS
      (Claims.Sampling.listWithin_deployZk _))
    (SamplingDeploy.commit_binds (code h) _)
    (sessionsTwo_spot h (sessionOpensTwo_of_frameIs h (layoutFor h) (embeds_layoutFor h) (frameIs_built h hB hS)))
    hx hP hd hin P (fun u => (Reveal.needs (partition h) s u).toFinset) (fun u v w => code_hpos h s u v w)
    (fun g => gateKey h s Pt d g.val)
    (fun g g' hk u hg => List.mem_toFinset.2 (gateKey_needs h hP hk (List.mem_toFinset.1 hg)))
    (fun g g' hk => gateKey_block h hk) (code_hwit h hB hS rfl hx hP hd hin P) hX hI

end SpotFlock
