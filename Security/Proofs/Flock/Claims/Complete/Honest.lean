import Proofs.Flock.Claims.Complete.Prover.Session
import Proofs.Flock.Claims.Deploy
import Proofs.Flock.Level3.Placed

/-!
# Flock's honest prover at the deployed plain verifier (`honestP`)

The plain record's reference prover (`(deployPlain rel).P`), the prover of `Claims.Flock.Complete H (deployPlain rel)`
(at its error `δc H`), against B's plain record (`FlockDeploy.V false`).

* **Its program** (`PI`): oracle queries, and sends, each answered by the verifier's reply. `PV := ExceptT String PI`
  is a verdict monad the session prover (`sessionP`) runs in, with SHA-512 the oracle (`Hashes.pi`) and the two sends of
  `Asks` (`asks`): the link message `0 :: linkBody rootB publics` and each round's framed bytes `0 :: c`, each answered
  by the coin's bytes (`Coin.bytes`), read back by `coinOf`.
* **Its program on a statement** (`prog`): on the session `setupOf false x` sets up, each table's trace from the
  witness (`traceOf`), the honest session, and the last message `Final.encode ⟨x, record, proofs⟩`.
* **The prover** (`honestP`): at the history `hist`, the program fed the verifier's replies so far (`feed`): the first
  send not yet answered, or the last message. It keeps no state: each message is the program replayed on the replies.

The witness is the trace (`Claims.Flock.Wit.hid`, table `i`'s address `a` at `i·2^m + a`); `A·z` and `B·z` are the level-0
matrices the verifier folds (`FlockLevel3.placedA`, `placedB`) applied block by block (`mulBlocks`), entry by entry.
-/

namespace FlockComplete

open Flock FlockDeploy

/-- **The prover's program**: a result, an oracle query, or a send answered by the verifier's reply. -/
inductive PI (α : Type) where
  | ret : α → PI α
  | query : Claims.Msg → (Claims.Msg → PI α) → PI α
  | ask : Claims.Msg → (Claims.Msg → PI α) → PI α

namespace PI

def bind {α β : Type} : PI α → (α → PI β) → PI β
  | .ret a, f => f a
  | .query q k, f => .query q fun r => bind (k r) f
  | .ask m k, f => .ask m fun r => bind (k r) f

instance instMonad : Monad PI where
  pure := .ret
  bind := bind

end PI

/-- The prover's verdict monad: an error ends the session. -/
abbrev PV := ExceptT String PI

/-- The prover's hashes: SHA-512 the oracle, SHA-256 and BLAKE3 as shipped (`Claims.Hashes.oc`'s). -/
def Hashes.pi : Flock.Hashes PV :=
  { Flock.Hashes.ofSha512 Sha512.hash with
    sha512 := fun b => ExceptT.lift (PI.query (Claims.toMsg b) fun a => PI.ret (Claims.ofMsg a)) }

/-- A coin read back from the verifier's reply: element `i` is the reply's bytes `16i` to `16i + 16`, `le64(lo) ‖ le64(hi)`,
as `Coin.bytes` writes it. -/
def coinOf (a : Claims.Msg) : Nat → F128 :=
  let b := Claims.ofMsg a
  fun i => ⟨u64OfLE b (16 * i), u64OfLE b (16 * i + 8)⟩

/-- **The prover's sends**: the link message and a round's framed bytes, each of tag `0`, each answered by its coin. -/
def asks : Asks PV where
  link rootB publics := ExceptT.lift (PI.ask (0 :: linkBody rootB.toList publics.toList) fun a => PI.ret (coinOf a))
  round c := ExceptT.lift (PI.ask (0 :: Claims.toMsg c) fun a => PI.ret (coinOf a))

/-- **The prover's next message**: the program on the replies `bs`: the first send they do not answer, or the program's
result. Each oracle query is the prover's. -/
def feed : PI Claims.Msg → List Claims.Msg → Claims.OC Claims.Msg
  | .ret a, _ => .ret a
  | .query q k, bs => .query q fun r => feed (k r) bs
  | .ask m _, [] => .ret m
  | .ask _ k, b :: bs => feed (k b) bs

/-- `placedA st` at row `r`, column `j`, as a bit. -/
def entryA (st : Flock.Stmt) (r j : Nat) : Bool :=
  decide (FlockLevel3.rangesEntry st (FlockLevel3.slotEntryA st) r j + FlockLevel3.deltaEntry st.da r j = 1)

/-- `placedB st` at row `r`, column `j`, as a bit. -/
def entryB (st : Flock.Stmt) (r j : Nat) : Bool :=
  decide (FlockLevel3.rangesEntry st (FlockLevel3.slotEntryB st) r j + FlockLevel3.deltaEntry st.db r j = 1)

/-- A level-0 matrix applied block by block over GF(2): `(M·z)[o·2^kLog + r] = Σ_j M[r][j]·z[o·2^kLog + j]`. -/
def mulBlocks (entry : Nat → Nat → Bool) (kLog : Nat) (z : Array Bool) : Array Bool :=
  let nb := 2 ^ kLog
  (Array.range z.size).map fun a =>
    (List.range nb).foldl (fun acc j => if entry (a % nb) j && z[a / nb * nb + j]! then !acc else acc) false

/-- **A table's trace** from the witness's bits, the table's addresses from `off`. -/
def traceOf (st : Flock.Stmt) (hid : Array Bool) (off : Nat) : Trace :=
  let z := (Array.range (2 ^ st.m)).map fun a => hid.getD (off + a) false
  { z, az := mulBlocks (entryA st) st.c.kLog z, bz := mulBlocks (entryB st) st.c.kLog z }

/-- The session's traces: table `i`'s from the witness's bits at `i·2^m`. -/
def tracesOf (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) : Array Trace :=
  let h := w.hid.toArray
  (Array.range sts.size).map fun i => traceOf sts[i]! h (i * 2 ^ sts[i]!.m)

/-- **The honest prover's program** on the statement `x`, the witness `w` and the tape `ω`: on the session the
statement sets up, the honest session, then the last message. A statement that sets up no session, or a session the
prover cannot run, gets the empty message, which the verifier refuses. -/
def prog (x : Claims.Msg) (w : Claims.Flock.Wit) (ω : Tape) : PI Claims.Msg :=
  match setupOf false x with
  | none => .ret []
  | some (_, spec, sts) => do
    match ← (sessionP asks Hashes.pi spec (tabsOf sts) (tracesOf sts w) ω).run with
    | .ok (record, proofs) => pure (Final.encode ⟨x, record, proofs⟩)
    | .error _ => pure []

/-- **Flock's honest prover** at the plain verifier: at each history, its program fed the verifier's replies. -/
def honestP (w : Claims.Flock.Wit) : Claims.Prover Tape := ⟨fun x hist ω => feed (prog x w ω) (hist.map Prod.snd)⟩

end FlockComplete
