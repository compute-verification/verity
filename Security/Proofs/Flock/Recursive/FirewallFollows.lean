import Proofs.Flock.Recursive.Firewall
import Proofs.Flock.Recursive.FirewallRun

/-!
# `FirewallSends` from one run

`FirewallSends`' template half is a lemma about the firewall's code (`FirewallRun.run_of_one`): on one request stream,
what it sends at every salt list is fixed by what it sends at one. Under live inner coins the worker's requests read the
commit strings the firewall sent, so at the salts `ys` the stream is the requests at `firewallStrings msgs ys`. What
stays an obligation is the honest worker's half, `WorkerSends`: at any commit strings, at one salt list, its requests
run through the firewall to the template `σ` the schedule fixes, its commit strings those of the messages reading those
strings, and finish. `firewallSends_of_worker` gives `FirewallSends` at the transcript `σ` filled with commit strings.
-/

namespace Flock.Guarantees

open Flock.Firewall FlockSoundness.Discharge.Recursion.FirewallRun

/-- Obligation (the honest worker's half of `FirewallSends`): at every private circuit `P`, witness `w`, registration
salts `r`, tape `a`, randomness `p` and commit strings `l` the firewall sent it, there are salts `ys₀` at which the
honest worker's requests `reqs P w r a p l`, run through the firewall (`Flock.Firewall.run s`), finish, send the messages
`σ` but for their commit strings (`σ` reads only the schedule), and send as commit strings, in salt order, the salted
commit strings under `ys₀` of the messages `msgs P w r a p`, message `i` reading the strings in `l` before it. -/
def WorkerSends {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (σ : List Flock.Firewall.Sent) : Prop :=
  ∀ P w r a p (l : Fin nh → ByteArray), ∃ ys₀ : Fin nh → ByteArray,
    (Flock.Firewall.run s (reqs P w r a p l) (List.ofFn ys₀)).2 = true ∧
    (Flock.Firewall.run s (reqs P w r a p l) (List.ofFn ys₀)).1.toList.map blank = σ ∧
    holesOf (Flock.Firewall.run s (reqs P w r a p l) (List.ofFn ys₀)).1 =
      List.ofFn fun i =>
        Hm96.Default512.commitString (Sha512.hash (msgs P w r a p i fun j => l (Fin.castLE i.isLt.le j))) (ys₀ i)

/-- **`FirewallSends` from the honest worker's half**: the firewall's transcript at every salt list is the template `σ`
filled with the messages' commit strings under those salts, in salt order (`fillAll`), each message reading the strings
before it. -/
theorem firewallSends_of_worker {Prog Wit Rr Ωa Ωp : Type} {nh : ℕ} (s : Flock.Firewall.Schedule)
    (reqs : Prog → Wit → Rr → Ωa → Ωp → (Fin nh → ByteArray) → List ByteArray)
    (msgs : Prog → Wit → Rr → Ωa → Ωp → (i : Fin nh) → (Fin i → ByteArray) → ByteArray)
    (σ : List Flock.Firewall.Sent) (h : WorkerSends s reqs msgs σ) :
    FirewallSends s reqs msgs fun cs => (fillAll σ (List.ofFn cs)).toArray := by
  intro P w r a p ys
  obtain ⟨ys₀, hfin, hσ, hh⟩ := h P w r a p (firewallStrings (msgs P w r a p) ys)
  have hfs : (fun i => Hm96.Default512.commitString (Sha512.hash (msgs P w r a p i fun j =>
      firewallStrings (msgs P w r a p) ys (Fin.castLE i.isLt.le j))) (ys i)) = firewallStrings (msgs P w r a p) ys :=
    funext fun i => by rw [firewallStrings]
  refine (run_of_one s _ σ (fun i => Sha512.hash (msgs P w r a p i fun j =>
    firewallStrings (msgs P w r a p) ys (Fin.castLE i.isLt.le j))) ys₀ (fun i => Firewall.Contract.sha512_size _)
    hfin hσ hh ys).trans ?_
  exact congrArg (fun cs => ((fillAll σ (List.ofFn cs)).toArray, true)) hfs

end Flock.Guarantees
