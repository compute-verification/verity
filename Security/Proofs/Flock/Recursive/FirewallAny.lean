import Proofs.Flock.Recursive.FirewallAccepts

/-!
# The live firewall at any requests

`rec_live --lean` relays every request of an inner session to `flock-firewall live` (`Flock.Firewall.liveWith`), and the
coin server receives only what it sends. `liveSends_of_accepts` is about the honest worker's requests; here they are any
bytes at all. Whatever the inner worker sends, the live firewall sends the coin server the pinned schedule's transcript
(`FirewallTemplate.tmpl`), its Commit naming tables of the schedule, each once, filled with the commit strings of 64-byte
digests under the salts it read, in order, and cut after one of its messages: the whole transcript when the session
finishes, else the place where it stopped. The tables' order, the digests, the place and how the session ends read the
requests and how many salts were read for each, never a salt's value.

* `liveOut`: the live loop (`liveWith`'s, over `liveStep`) at any requests: what it sends until the first refusal, and
  how it ends.
* `run_any`: the same of `Flock.Firewall.run` at any requests and any salts of one length.
* `Flock.SecurityProofs.liveOut_any`: the theorem.

Commit's table order is the worker's: the firewall pins each named table's node count to the schedule's and refuses a
table named twice (`step`'s typed pins), and only the canonical compare, which the kernel can't evaluate, pins the set.
So the order `o` is an output here, among the schedule's tables (`Tables`).
-/

namespace FlockSoundness.Discharge.Recursion.FirewallAny

open Flock Flock.Firewall FirewallRun FirewallTemplate FirewallLive

/-- **A live session at any requests** (`liveWith`'s loop over `liveStep`): from the state `st`, each request with the
fresh salts read for it, what the firewall sends the coin server, in order, up to the first refusal, and how the session
ends: `some t` at the state `t` after the last request when nothing was refused, `none` at a refusal, after which
`liveWith` answers every request with the stop and sends nothing. A line that is not hex stops it the same way. -/
def liveOut (s : Schedule) : State → List (ByteArray × List ByteArray) → List Sent × Option State
  | st, [] => ([], some st)
  | st, (b, fresh) :: rest =>
    match liveStep s st b fresh with
    | .ok (sent, _, st') => (sent ++ (liveOut s st' rest).1, (liveOut s st' rest).2)
    | .error _ => ([], none)

/-- **The session finished**: no request was refused and `Finish` was taken. Otherwise it stopped (`end: stop`). -/
def liveFinished (s : Schedule) (L : List (ByteArray × List ByteArray)) : Bool :=
  match (liveOut s { salts := [] } L).2 with
  | some t => t.finished
  | none => false

/-- **What a live session's requests fix**: each request, and how many fresh salts were read for it. -/
def shape (L : List (ByteArray × List ByteArray)) : List (ByteArray × ℕ) := L.map fun x => (x.1, x.2.length)

/-- **Tables of the schedule, each named once**: the table lists a Commit the firewall takes may name, in the worker's
order (a table the schedule has no node count for is refused, and so is a table named twice). -/
def Tables (s : Schedule) (o : List String) : Prop :=
  o.Nodup ∧ ∀ t ∈ o, (natOf? (s.roots.getObjValD t)).isSome

/-- **The firewall at any requests**: for any requests and any number `n` of salts, there are tables `o` of the schedule,
each named once, 64-byte digests `ds`, a place `k` of the schedule's transcript and an end `fin` such that at every salt
list of length `n`, `run` sends the schedule's transcript with Commit naming `o` (`tmpl s o`), filled with the commit
strings of `ds` under the salts in order, cut after its first `k` messages, and finishes exactly when `fin`; a session
that finishes sends the whole transcript. -/
theorem run_any (s : Schedule) (reqs : List ByteArray) (n : ℕ) :
    ∃ (o : List String) (ds : List ByteArray) (k : ℕ) (fin : Bool),
      Tables s o ∧ (∀ d ∈ ds, d.size = 64) ∧ ds.length ≤ n ∧ k ≤ (tmpl s o).length ∧
        (fin = true → k = (tmpl s o).length) ∧
        ∀ ys : List ByteArray, ys.length = n →
          run s reqs ys =
            (((fillAll (tmpl s o) (List.zipWith Hm96.Default512.commitString ds ys)).take k).toArray, fin) := by
  sorry

end FlockSoundness.Discharge.Recursion.FirewallAny

namespace Flock.SecurityProofs

open Flock.Firewall FlockSoundness.Discharge.Recursion
open FirewallRun FirewallTemplate FirewallAny

/-- **The live firewall at any requests sends the schedule's transcript, filled at the strings it accepted, or stops at
a public place.** For any requests, each with how many fresh salts are read for it (`R`), there are tables `o` of the
schedule, each named once, 64-byte digests `ds`, a place `k` of the schedule's transcript and an end `fin` such that
every live session on those requests (`liveOut`, `flock-firewall live`'s loop), whatever salts it reads (`shape L = R`),
sends the coin server the schedule's transcript with Commit naming `o` (`tmpl s o`), filled with the commit strings of
`ds` under the salts read, in order, and cut after its first `k` messages, and finishes exactly when `fin`. A session
that finishes sends the whole transcript; one that stops has sent its first `k` messages, `k` at most the transcript's
length, which the schedule fixes (`tmpl s o`'s length doesn't read `o`). No condition on the requests: they are the inner
worker's, whatever it is. -/
theorem liveOut_any (s : Schedule) (R : List (ByteArray × ℕ)) :
    ∃ (o : List String) (ds : List ByteArray) (k : ℕ) (fin : Bool),
      Tables s o ∧ (∀ d ∈ ds, d.size = 64) ∧ ds.length ≤ (R.map (·.2)).sum ∧ k ≤ (tmpl s o).length ∧
        (fin = true → k = (tmpl s o).length) ∧
        ∀ L : List (ByteArray × List ByteArray), shape L = R →
          (liveOut s { salts := [] } L).1 =
              (fillAll (tmpl s o) (List.zipWith Hm96.Default512.commitString ds (L.flatMap (·.2)))).take k ∧
            liveFinished s L = fin := by
  sorry

end Flock.SecurityProofs
