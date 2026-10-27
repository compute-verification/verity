import Proofs.Flock.Claims.Recursion.RestLowR
import Proofs.Flock.Claims.Recursion.HmOpen

/-!
# `SaltRowsJ` at `finalR` off the hm96 slots the openings sit at (`OpensSlots`)

`SaltRowsJ` asks, at every round before the last and each of its commit strings, the salt's hash and the row's digest
at the opening `finalR` cuts off the last message. Both are hm96's two recorded calls at the slot that commit string
came from (`HmOpen.checks_of_calls`), so `SaltRowsJ` is a fact about the layout alone, with no hash in it:

* `OpensSlots L`: at a dense compiled statement and witnesses one a part, each opening `finalR` cuts off the last
  message is some part `s`'s row at port `nm` (`rowHidden`) and the salt of port `q`'s hm96 slot at instance `n`
  (`hmSalt`), read off `ws[s]`; the commit string it opens is that slot's (`hmCommit`); the slot's calls (`hmCalls`)
  are among `s`'s recorded calls; and the port's row prefix is the one `rowInDeployed` lays at the row's length.
* `saltRowsJ_of_slots`: `OpensSlots L` gives `SaltRowsJ` at `finalR`.
-/

namespace RecDeploy.RestLow

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP)
open Claims.Recursion.VStarB
open RecDeploy.RestRead (DenseX kI)
open RecDeploy.DecompBR

/-- **Each opening `finalR` cuts is an hm96 slot's**, at some part `s`: its row at port `nm` and the salt of port `q`'s
slot at instance `n`, off `ws[s]`, opening that slot's commit string, with the slot's calls among `s`'s recorded calls
and the port's row prefix the one `rowInDeployed` lays at the row's length. -/
def OpensSlots (L : LayoutB) : Prop :=
  ∀ x, DenseX x → ∀ (ws : List Claims.Flock.Wit), ws.length = nS x → ∀ os last,
    DeployedR.finalR ((L.msgs x ws).getD (kI x - 1) []) = some (os, last) → ∀ j i o, (os.getD j [])[i]? = some o →
      ∃ s, ∃ hs : s < ws.length, ∃ (cc : Circuit) (nm : String) (q n : ℕ),
        (∀ p ∈ hmCalls cc nm q n, (p.1 ++ readForms ws[s] p.2.1, readForms ws[s] p.2.2) ∈ L.calls x s ws[s]) ∧
        rowPfx cc q = (HmRow.rowPrefix ((readForms ws[s]
          (((rowHidden cc nm n).getD []).map HmNets.Lin.var)).length / 2)).data.toList ∧
        o = (readForms ws[s] (((rowHidden cc nm n).getD []).map HmNets.Lin.var),
          DeployedR.saltOf (readForms ws[s] (hmSalt cc q n))) ∧
        commitAt ((L.msgs x ws).getD j []) i = readForms ws[s] (hmCommit cc q n)

/-- **`SaltRowsJ` at `finalR` from `OpensSlots`**: each opening's salt and row checks are its slot's two recorded calls,
true where every recorded call is. -/
theorem saltRowsJ_of_slots (rd : Msg → Option FinalP) (wE w : Msg → ℝ) (L : LayoutB) (hO : OpensSlots L) :
    SaltRowsJ rd wE DeployedR.finalR w L := by
  intro x hx j _ i _ c ws hl hT Q S
  have hslot : ∀ os last, DeployedR.finalR ((L.msgs x ws).getD (kI x - 1) []) = some (os, last) → ∀ o,
      (os.getD j [])[i]? = some o →
      ∃ cc nm q n w', o = (readForms w' (((rowHidden cc nm n).getD []).map HmNets.Lin.var),
          DeployedR.saltOf (readForms w' (hmSalt cc q n))) ∧
        commitAt ((L.msgs x ws).getD j []) i = readForms w' (hmCommit cc q n) ∧
        (FlockDeploy.sha512 (Claims.toMsg Hm96.Default512.saltPrefix ++
            List.ofFn (DeployedR.saltOf (readForms w' (hmSalt cc q n)))) = (readForms w' (hmCommit cc q n)).drop 64 ∧
          FlockDeploy.sha512 (rowInDeployed (readForms w' (((rowHidden cc nm n).getD []).map HmNets.Lin.var))) =
            Claims.Recursion.xorMask ((readForms w' (hmCommit cc q n)).take 64) (fun i : Fin 64 =>
              (Hm96.Default512.mask ⟨Array.ofFn (DeployedR.saltOf (readForms w' (hmSalt cc q n)))⟩).get! i.val)) := by
    intro os last hf o ho
    obtain ⟨s, hs, cc, nm, q, n, hC, hpfx, ho', hcm⟩ := hO x hx ws hl os last hf j i o ho
    refine ⟨cc, nm, q, n, ws[s], ho', hcm, HmOpen.checks_of_calls (fun p hp => ?_) hpfx⟩
    have he := RecDeploy.CapLower.mem_allCalls (List.getElem?_eq_getElem hs) (hC p hp)
    have hTe := hT _ he
    exact hTe
  have hk : (bdR rd wE DeployedR.finalR w).DE.V.k (xIOf x) = kI x := rfl
  have hfin : (bdR rd wE DeployedR.finalR w).final = DeployedR.finalR := rfl
  constructor
  · intro os last hf o ho
    rw [hk, hfin] at hf
    obtain ⟨cc, nm, q, n, w', rfl, hcm, hS, -⟩ := hslot os last hf o ho
    show FlockDeploy.sha512 (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn _) =
      (commitAt ((L.msgs x ws).getD j []) i).drop 64
    rw [hcm]
    exact hS
  · intro os last hf o ho
    rw [hk, hfin] at hf
    obtain ⟨cc, nm, q, n, w', rfl, hcm, -, hR⟩ := hslot os last hf o ho
    show FlockDeploy.sha512 (rowInDeployed _) = Claims.Recursion.xorMask ((commitAt ((L.msgs x ws).getD j []) i).take 64)
      (fun i : Fin 64 => (Hm96.Default512.mask ⟨Array.ofFn _⟩).get! i.val)
    rw [hcm]
    exact hR

end RecDeploy.RestLow
