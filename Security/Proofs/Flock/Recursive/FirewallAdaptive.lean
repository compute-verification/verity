import Proofs.Flock.Recursive.FirewallAny

/-!
# The live firewall at a worker that reads its answers

`FirewallAny.liveOut_any` fixes the inner worker's requests, and how many salts are read for each, before any salt. The
inner worker sends each request after reading what `flock-firewall live` answered it (`Flock.Firewall.liveWith`): the
messages it sent the coin server and the salts it drew for them, `{"sent", "salts"}`, and whatever the worker reads
from them, the live coins included. Here the worker is any function of those answers, and its requests are read at
the salts the session draws. Whatever it is, the live firewall sends the schedule's transcript filled with commit
strings `strs dg ys`, the `i`-th that of a 64-byte digest `dg i (ys.take i)` under salt `i`, cut after one of its
messages, or a Commit naming the schedule's tables in another order: digest `i` reads the salts before it, never salt
`i` or a later one.

* `adaptOut`: the live loop at a worker `W`, for at most `N` requests, reading the salts from the list `ys` in turn.
* `liveOut_prefix`, `adaptOut_prefix`: the sends on a prefix of the requests are a prefix of the sends.
* `adaptOut_liveOut`: each such session is a live session (`FirewallAny.liveOut`) on the requests it took.
* `adaptOut_rel`: two sessions whose salts agree before `i` send the commit strings of the same digests up to salt
  `i`: they take the same requests and draw the same salts until one draws salt `i`.
* `Flock.SecurityProofs.adaptOut_any`: the theorem.

The model reads the salts from a list, and a session that reads more salts than the list holds is given those left;
`liveWith` reads them from the operating system, which never runs out, so the two agree at a list long enough for the
salts the session reads. A line that isn't hex stops `liveWith`'s session as a refused request does, and the worker
model sends bytes: the empty request, which `decode` refuses, sends the same.
-/

namespace FlockSoundness.Discharge.Recursion.FirewallAdaptive

open Flock Flock.Firewall FirewallRun FirewallTemplate FirewallLive FirewallAny

/-- **What the live firewall answered so far**: per request, what it sent the coin server and the salts it drew. -/
abbrev History := List (List Sent × List ByteArray)

/-- **An inner worker**: its next request, read from the answers so far, or `none` when it closes the session. Its
tape, its randomness and the live coins it was sent are its parameters. -/
abbrev Worker := History → Option ByteArray

/-- How many fresh salts `liveWith` reads before a request runs. -/
def reads (b : ByteArray) : Nat := b.size / NODE_BYTES + 1

/-- **A live session at the worker `W`**, for at most `N` requests: from the state `st` and the answers `h`, the salts
read in turn from `ys`, what the firewall sends the coin server, in order, up to the first refusal, and how the session
ends, as `FirewallAny.liveOut`. -/
def adaptOut (s : Schedule) (W : Worker) : Nat → State → History → List ByteArray → List Sent × Option State
  | 0, st, _, _ => ([], some st)
  | N + 1, st, h, ys =>
    match W h with
    | none => ([], some st)
    | some b =>
      match liveStep s st b (ys.take (reads b)) with
      | .ok (sent, used, st') =>
        (sent ++ (adaptOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))).1,
          (adaptOut s W N st' (h ++ [(sent, used)]) (ys.drop (reads b))).2)
      | .error _ => ([], none)

/-- **The session finished**: no request was refused and `Finish` was taken. -/
def adaptFinished (s : Schedule) (W : Worker) (N : Nat) (ys : List ByteArray) : Bool :=
  match (adaptOut s W N { salts := [] } [] ys).2 with
  | some t => t.finished
  | none => false

/-- **The commit strings at digests that read the salts before them**: string `i` is digest `dg i (ys.take i)` under
salt `i`. -/
def strs (dg : Nat → List ByteArray → ByteArray) (ys : List ByteArray) : List ByteArray :=
  List.zipWith Hm96.Default512.commitString ((List.range ys.length).map fun i => dg i (ys.take i)) ys

/-! ## A prefix of the requests -/

/-- **The sends on a prefix of the requests are a prefix of the sends.** -/
theorem liveOut_prefix (s : Schedule) (st : State) (L₁ L₂ : List (ByteArray × List ByteArray)) :
    (liveOut s st L₁).1 <+: (liveOut s st (L₁ ++ L₂)).1 := by
  sorry

/-- **The sends on a worker's first requests are a prefix of its sends.** -/
theorem adaptOut_prefix (s : Schedule) (W : Worker) (N M : Nat) (st : State) (h : History) (ys : List ByteArray) :
    (adaptOut s W N st h ys).1 <+: (adaptOut s W (N + M) st h ys).1 := by
  sorry

/-! ## The requests a session took -/

/-- **A worker's session is a live session on the requests it took.** -/
theorem adaptOut_liveOut (s : Schedule) (W : Worker) (N : Nat) (st : State) (h : History) (ys : List ByteArray) :
    ∃ L : List (ByteArray × List ByteArray), adaptOut s W N st h ys = liveOut s st L := by
  sorry

/-! ## Two sessions whose salts agree before `i` -/

/-- **Two sessions whose salts agree before `i`**, from states equal but for undrawn salts of one length and the same
answers: their commit strings are those of digests `D₁ ++ D₂` and `D₁ ++ D₂'` under each session's salts, the state's
then those of `ys`, with `D₁` past salt `i` or nothing after it. -/
theorem adaptOut_rel (s : Schedule) (W : Worker) (N : Nat) (st : State) (z' : List ByteArray) (h : History)
    (ys ys' : List ByteArray) (i : Nat) (hz : st.salts.length = z'.length) (hy : ys.length = ys'.length)
    (ha : (st.salts ++ ys).take i = (z' ++ ys').take i) :
    ∃ D₁ D₂ D₂' : List ByteArray, (∀ d ∈ D₁ ++ D₂ ++ D₂', d.size = 64) ∧
      (D₁ ++ D₂).length ≤ (st.salts ++ ys).length ∧ (D₁ ++ D₂').length ≤ (z' ++ ys').length ∧
      (adaptOut s W N st h ys).1.flatMap holes =
        List.zipWith Hm96.Default512.commitString (D₁ ++ D₂) (st.salts ++ ys) ∧
      (adaptOut s W N { st with salts := z' } h ys').1.flatMap holes =
        List.zipWith Hm96.Default512.commitString (D₁ ++ D₂') (z' ++ ys') ∧
      (i < D₁.length ∨ (D₂ = [] ∧ D₂' = [])) := by
  sorry

end FlockSoundness.Discharge.Recursion.FirewallAdaptive

namespace Flock.SecurityProofs

open Flock.Firewall FlockSoundness.Discharge.Recursion
open FirewallRun FirewallTemplate FirewallAny FirewallAdaptive

/-- **The live firewall at any worker that reads its answers sends the schedule's transcript, filled at commit strings
whose digests read only the salts before them, or stops at a public place, unless its Commit named the schedule's
tables in another order.** `T` is the order of the schedule's tables that Commit is to name. For any inner worker `W`
whose next request reads the answers so far, the sends and the salts drawn, for at most `N` requests, and any number
`n` of salts, there are digests `dg`, `dg i pre` 64 bytes, and a stop `σ` such that at every salt list `ys` of length
`n`, the live session (`adaptOut`, `flock-firewall live`'s loop) either sends the coin server the schedule's transcript
with Commit naming `T` (`tmpl s T`), filled with `strs dg ys`, string `i` the commit string of `dg i (ys.take i)` under
salt `i`, cut after its first `(σ ys).1` messages, and finishes exactly when `(σ ys).2`; or sent a Commit naming the
schedule's tables, each once, in an order other than `T`, which `step` takes today. The stop lies in
`Fin ((tmpl s T).length + 1) × Bool`, and a session that finishes sends the whole transcript. Digest `i` reads the
salts before `i` alone: `W` may read every salt it was answered. -/
theorem adaptOut_any (s : Schedule) (T : List String) (W : Worker) (N n : Nat) :
    ∃ (dg : Nat → List ByteArray → ByteArray) (σ : List ByteArray → Nat × Bool),
      (∀ i pre, (dg i pre).size = 64) ∧
      ∀ ys : List ByteArray, ys.length = n →
        (σ ys).1 ≤ (tmpl s T).length ∧ ((σ ys).2 = true → (σ ys).1 = (tmpl s T).length) ∧
        (((adaptOut s W N { salts := [] } [] ys).1 = (fillAll (tmpl s T) (strs dg ys)).take (σ ys).1 ∧
            adaptFinished s W N ys = (σ ys).2) ∨
          (namesOf (adaptOut s W N { salts := [] } [] ys).1 ≠ T ∧
            Tables s (namesOf (adaptOut s W N { salts := [] } [] ys).1))) := by
  sorry

end Flock.SecurityProofs
