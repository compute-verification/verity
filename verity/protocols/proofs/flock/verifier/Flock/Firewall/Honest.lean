/-
The honest worker's requests in an inner session (live PROTOCOL.md §10.3), as a program: what an honest worker sends the
ZK server (`Flock.Firewall.run`), from the pinned schedule and what the worker brings beyond it (`Honest`).
`FirewallHonest` proves the ZK server finishes on them, and `firewall-honest-difftest` (`verity/core/difftest`) holds
`rec_live`'s requests on a real prover's recorded session to them, byte for byte
(`verity/ml/flock/tests/test_honest_reqs.py`).
-/
module

public import Flock.Firewall

@[expose] public section

namespace Flock.Firewall

open Lean (Json)

/-- One round of an honest worker: what its challenger absorbs, call by call, and the kind of the squeeze that ends the
round. -/
structure HonestRound where
  items : List Absorb := []
  kind : Nat := 0

/-- What an honest worker brings to an inner session beyond the pinned schedule: the Pings it measures its link with
before it registers, the hidden statement's Σ, `root_F`, each table's level-0 top in the order Commit names them, each
stream's `Open` label, the streams it opens before Commit (`opens 0`) and just before round `i` (`opens (i + 1)`), each
round by its place in the schedule (its statement digest, caps, nonces and elements among its items), and its proofs,
each a table's, a rep's and the proof's bytes. -/
structure Honest where
  pings : List ByteArray := []
  sigma : String
  rootF : ByteArray
  roots : List (String × ByteArray)
  label : String → ByteArray
  opens : Nat → List String
  round : Nat → HonestRound
  proofs : List (String × Nat × ByteArray) := []

/-- A Ping as the worker writes it, with its content (`decode` reads none of it). -/
def putPing (b : ByteArray) : ByteArray := ⟨#[9]⟩ ++ putBytes b

/-- A Proof as the worker writes it, with its table, rep and bytes (`decode` reads none of them). -/
def putProof (table : String) (rep : Nat) (proof : ByteArray) : ByteArray :=
  ⟨#[5]⟩ ++ (putStr table ++ (putLE 4 rep ++ putBytes proof))

/-- The schedule's Hello with Σ in its link, each object rebuilt from its members in order. -/
def helloWith (hello : Json) (sigma : String) : Json :=
  match hello with
  | .obj kvs => Json.mkObj (kvs.toArray.toList.map fun (k, v) => (k, if k == "link" then linkWith v else v))
  | j => j
where
  linkWith : Json → Json
    | .obj kvs => Json.mkObj (kvs.toArray.toList ++ [("sigma", .str sigma)])
    | j => j

/-- The `Open`s the worker sends at its `j`-th place (`Honest.opens`), each at the schedule's parent and position. -/
def honestOpens (s : Schedule) (h : Honest) (j : Nat) : List Req :=
  (h.opens j).filterMap fun stream =>
    (s.streams.find? (·.1 == stream)).map fun (_, parent, pos, _) => .openStream stream parent (h.label stream) pos

/-- Round `i`'s request: its content what the worker's challenger absorbed and the squeeze of the schedule's coin count. -/
def honestRound (s : Schedule) (h : Honest) (i : Nat) : List Req :=
  match roundSpec (s.rounds.getD i .null) with
  | some (stream, _, n, _) => [.round stream (putRound (h.round i).items (h.round i).kind n) n]
  | none => []

/-- **The honest worker's requests** in a session held to the schedule `s`, on the wire: its Pings, `Register`, Hello as
it prints it (`printJson`), its first `Open`s, Commit with its tables' level-0 tops and no public words, Link with the
schedule's count of zero elements, the schedule's rounds in order each after the `Open`s sent just before it, its proofs,
then Finish. -/
def honestReqs (s : Schedule) (h : Honest) : List ByteArray :=
  h.pings.map putPing ++ [putReq .register, putReq (.hello (printJson (helloWith s.hello h.sigma)))] ++
    (honestOpens s h 0).map putReq ++
    [putReq (.commit h.rootF h.roots []), putReq (.link ⟨Array.replicate (16 * s.y) 0⟩)] ++
    (List.range s.rounds.size).flatMap (fun i => (honestOpens s h (i + 1) ++ honestRound s h i).map putReq) ++
    h.proofs.map (fun p => putProof p.1 p.2.1 p.2.2) ++ [putReq .finish]

end Flock.Firewall
