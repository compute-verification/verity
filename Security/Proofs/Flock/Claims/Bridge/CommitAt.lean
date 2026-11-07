module

public import Proofs.Flock.Claims.Bridge.Transports
public import Proofs.Flock.Claims.Sampling.CommitAt

/-!
# The commitment at ports that read the roots, at SHA-512

At SHA-512 an opening passes `commitAt c own` exactly where Flock's registered read (`Registered.opensLeaf`) accepts its
hm96 leaf at the port that `regPath` finds among the ports of its root list and position (`opens_sha512At`, which is
`opens_sha512` at those ports). `opensLeaf` climbs to the port's own root, and at `rooted` ports the port found for the
list's `k`th root is port `k` with that root. So Flock's registered read at port `k`, with the run's root, passes
`commitAt` there (`opens_rooted_sha512`). Where `at_` reads the roots by port (`AtPorts`) and the root list has no
repeats, the read at the opening's own port is all it takes (`opens_atPorts`).
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling
open FlockSoundness.Refine FlockSoundness.Binding
open Properties.NetworkCertifier.Bytes (bs)

/-- **At SHA-512, `commitAt` is Flock's registered read** at the port `regPath` finds among the ports of the opening's
root list and position (`opens_sha512` there). -/
theorem opens_sha512At (c : Code) (own : List Msg → ℕ → Array _root_.Flock.Registered.Port) (roots : List Msg)
    (o : Opening Bool) :
    (commitAt c own).Opens sha512Msg roots o = true ↔
      ∃ ri reg, c.at_ roots o.pos = some ri ∧ (own roots o.pos).find? (fun p => decide (bs p.root = ri.1)) = some reg ∧
        _root_.Flock.Registered.opensLeaf reg (bytes (digBytes (H512 (hm96Input (row c.place c.bits o) (c.salt o)))))
          ri.2 (bytes (c.π o)) = true :=
  opens_sha512 { c with own := own roots o.pos } roots o

/-- **Flock's registered read at port `k`, with the run's root, passes `commitAt` at `rooted` ports** where `at_`
checks the opening under the list's `k`th root and no earlier entry of the list is that root. -/
theorem opens_rooted_sha512 {c : Code} {shapes : ℕ → Array _root_.Flock.Registered.Port} {roots : List Msg}
    {o : Opening Bool} {k idx : ℕ} {r : Msg} (hat : c.at_ roots o.pos = some (r, idx))
    (hk : k < (shapes o.pos).size) (hr : roots[k]? = some r) (hfirst : ∀ j < k, roots[j]? ≠ some r)
    (hread : _root_.Flock.Registered.opensLeaf { (shapes o.pos)[k] with root := bytes r }
      (bytes (digBytes (H512 (hm96Input (row c.place c.bits o) (c.salt o))))) idx (bytes (c.π o)) = true) :
    (commitAt c fun roots g => rooted (shapes g) roots).Opens sha512Msg roots o = true :=
  (opens_sha512At c _ roots o).2 ⟨(r, idx), _, hat, find?_rooted hk hr hfirst, hread⟩

/-- **A code's `at_` reads the roots by port**: it checks the value at `g` under the root list's entry for `g`'s port
`port g`, at leaf `idx g`, and nowhere past the list. -/
def AtPorts (c : Code) (port idx : ℕ → ℕ) : Prop :=
  ∀ roots g, c.at_ roots g = (roots[port g]?).map fun r => (r, idx g)

/-- **Flock's registered read at the opening's port passes `commitAt` at `rooted` ports**, where `at_` reads the roots
by port and the root list has no repeats. -/
theorem opens_atPorts {c : Code} {port idx : ℕ → ℕ} {shapes : ℕ → Array _root_.Flock.Registered.Port}
    {roots : List Msg} {o : Opening Bool} {r : Msg} (hA : AtPorts c port idx) (hk : port o.pos < (shapes o.pos).size)
    (hr : roots[port o.pos]? = some r) (hnd : roots.Nodup)
    (hread : _root_.Flock.Registered.opensLeaf { (shapes o.pos)[port o.pos] with root := bytes r }
      (bytes (digBytes (H512 (hm96Input (row c.place c.bits o) (c.salt o))))) (idx o.pos) (bytes (c.π o)) = true) :
    (commitAt c fun roots g => rooted (shapes g) roots).Opens sha512Msg roots o = true :=
  opens_rooted_sha512 (by simp [hA roots o.pos, hr]) hk hr (before_of_nodup hnd hr) hread

end SamplingDeploy
