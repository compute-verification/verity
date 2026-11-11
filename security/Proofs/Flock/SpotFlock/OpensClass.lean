module

public import Proofs.Flock.SpotFlock.OpensWhole
public import Proofs.Flock.SamplingGeneric.TwoRegFirstHitClass

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
The key's other two hypotheses are `gateKey_needs` and `gateKey_block`.
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

end SpotFlock
