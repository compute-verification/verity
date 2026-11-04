import Proofs.Properties.NetworkScan.Fits
import Proofs.Properties.NetworkScan.Causal

/-!
# Where a wire's gates lie in a flattening

A wire's reader leaves are leaves of a root input node, and its writer is an out leaf of a root node over whose leaves
a part of the root's return runs. In every flattening (`Properties.NetworkScan.Flattens`) of a Program C-Flock decodes
and lays out:

* `input_wired`: a root input node's leaf resolves to an input gate (`CausalProof.input_lands`, `Fits.gate_at_prim`);
* `writer_returned`: such an out leaf resolves to a gate the flattening returns (`Fits.seq_get_at`).
-/

namespace Properties.NetworkCertifier.WiredProof

open Flock.Program
open NciCheck.Properties.SecurityProofs
open Properties.NetworkScan
open Properties.NetworkScan.Fits

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Resolve.Good P sc → Resolve.BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
include hroot hB hL

/-- **A wired gate is an input**: leaf `j` of a root input node resolves to a gate every flattening makes an input. -/
theorem input_wired (C : Claims.Sampling.Prog) (hF : Flattens P C) {q j x : ℕ}
    (hq : q < (P.defs[P.root]!).body.nodes.size)
    (hin : NciCheck.Desc.isInputNode P ((P.defs[P.root]!).body.nodes[q]!) = true)
    (hj : j < ((P.defs[P.root]!).body.nodes[q]!).out.leaves)
    (hx : resolve P P.rootScope ⟨false, q, j⟩ = .ok x) : C.gates[x]? = some .input := by
  obtain ⟨hlo, hhi⟩ := CausalProof.input_lands P hroot hB hL hq hin hj _ x hx
  have hmono := Resolve.offsets_mono P P.root (hL _ hroot.1) hroot.2
  have hend := ((hL _ hroot.1) hroot.2).2.1
  have hlt : x < C.gates.length := by
    rw [hF.1]
    have := hmono (q + 1) _ hq (le_refl _)
    omega
  obtain ⟨p, ops, sc, hgate, hiff, -⟩ := hF.2.1 x hlt
  simp only [NciCheck.Desc.isInputNode, Bool.and_eq_true] at hin
  have hp := gate_at_prim P hroot hL hq hin.1 hlo hhi hgate
  exact hiff.2 (by rw [hp]; exact hin.2)

end

/-- **A writer is returned**: out leaf `j` of root node `ke`, over whose leaves from its first a part of the root's
return runs, `c` of them past `j`, resolves to a gate every flattening returns. -/
theorem writer_returned {P : Prog} (hroot : P.root < P.defs.size)
    (hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]!) (C : Claims.Sampling.Prog) (hF : Flattens P C)
    {ke j c w : ℕ} (hret : Part.affine false ke 0 c ∈ (P.defs[P.root]!).body.ret.parts) (hj : j < c)
    (hw : resolve P P.rootScope ⟨false, ke, j⟩ = .ok w) : w ∈ C.outs := by
  obtain ⟨-, -, ps, hps, -⟩ := (hok P.root hroot).2
  rw [hps, (Walk.ofParts_spec ps).1] at hret
  obtain ⟨i, hi, hqi⟩ := Array.mem_iff_getElem.1 hret
  have hpi : ps[i]! = .affine false ke 0 c := by rw [getElem!_pos ps i hi, hqi]
  have hcnt : j < ps[i]!.count := by rw [hpi]; simp only [Part.count]; omega
  obtain ⟨hidx, hget⟩ := seq_get_at ps hi hcnt
  rw [hpi] at hget
  change _ = (⟨false, ke, 0 + j⟩ : Flock.Program.Ref) at hget
  rw [Nat.zero_add, ← hps] at hget
  rw [← hps] at hidx
  exact (hF.2.2 w).2 ⟨_, hidx, by rw [hget]; exact hw⟩

end Properties.NetworkCertifier.WiredProof
