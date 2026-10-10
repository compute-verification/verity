import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.FirewallRun

/-!
# `FirewallSends` from one run

`FirewallSends`' template half is a lemma about the firewall's code (`FirewallRun.run_of_one`): what it sends at every
salt list is fixed by what it sends at one. What stays an obligation is the honest worker's half, `WorkerSends`: at one
salt list, its requests run through the firewall to the template `σ` the schedule fixes, its commit strings those of the
messages, and finish. `firewallSends_of_worker` gives `FirewallSends` at the transcript `σ` filled with commit strings.
-/

namespace Flock.Guarantees

open Flock.Firewall FlockSoundness.Discharge.Recursion.FirewallRun

/-- Obligation (the honest worker's half of `FirewallSends`): at every private circuit `P`, witness `w`, registration
salts `r`, tape `a` and randomness `p`, there are salts `ys₀` at which the honest worker's requests `reqs P w r a p`,
run through the firewall (`Flock.Firewall.run s`), finish, send the messages `σ` but for their commit strings (`σ`
reads only the schedule), and send as commit strings, in salt order, the salted commit strings of the messages
`msgs P w r a p` under `ys₀`. -/
def WorkerSends {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → List ByteArray) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (σ : List Flock.Firewall.Sent) : Prop :=
  ∀ P w r a p, ∃ ys₀ : Fin nh → ByteArray, (Flock.Firewall.run s (reqs P w r a p) (List.ofFn ys₀)).2 = true ∧
    (Flock.Firewall.run s (reqs P w r a p) (List.ofFn ys₀)).1.toList.map blank = σ ∧
    holesOf (Flock.Firewall.run s (reqs P w r a p) (List.ofFn ys₀)).1 =
      List.ofFn fun i => Hm96.Default512.commitString (Sha512.hash (msgs P w r a p i)) (ys₀ i)

/-- **`FirewallSends` from the honest worker's half**: the firewall's transcript at every salt list is the template `σ`
filled with the messages' commit strings under those salts, in salt order (`fillAll`). -/
theorem firewallSends_of_worker {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → List ByteArray) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (σ : List Flock.Firewall.Sent) (h : WorkerSends s reqs msgs σ) :
    FirewallSends s reqs msgs fun cs => (fillAll σ (List.ofFn cs)).toArray := by
  intro P w r a p ys
  obtain ⟨ys₀, hfin, hσ, hh⟩ := h P w r a p
  exact run_of_one s _ σ (fun i => Sha512.hash (msgs P w r a p i)) ys₀ (fun i => Firewall.Contract.sha512_size _)
    hfin hσ hh ys

end Flock.Guarantees
