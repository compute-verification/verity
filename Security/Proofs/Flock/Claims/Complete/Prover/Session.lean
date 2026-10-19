import Proofs.Flock.Claims.Complete.Prover.Ligerito
import Proofs.Flock.Claims.Complete.Prover.Print
import Proofs.Flock.VBridge.Family.Encode

/-!
# The honest prover's session: the reps, the proof files and the record

PROTOCOL.md §4–§8 and §16 from the prover's side, for the plain verifier (`verifyWithM`).

* **A rep** (`repP`): zerocheck, lincheck, the ring switch on `abcClaims ++ extras`, then Ligerito on the table's
  level-0 commitment for the batched claim `T`, from the rep's own transcript (`Transcript.init #[] domain`). The proof
  file is `FlockVBridge.Family.encodeProof` at the shape `verifyRepWith` reads (`shapeOf`), the encoder whose inverse
  `decodeProof_encodeProof` proves.
* **The session** (`sessionP`): each table's level-0 commitment (`commitP` of the packed table, salted from the tape's
  level `0`), the link message (each `root_B`, no public words), the link's points from its coin, then the rep streams in
  `spec.repStreams` order.
* **The record** (`recordJson`): the `flock-live-session/v1` record `Record.decodeWith` and `FlockDeploy.liveOf` read:
  `hello` (the statement's, or under a keyed coin tree its `Hello` with a zero nonce, canonical, and a `coin_tree` of a
  zero key and root), `config.require_link`, the exchange link (`Σ` as `root_F`, each `root_B`, the points, `y = 0`),
  the statement's `unit_draw`, every rep stream's rounds with their coins and retained bytes, and every digest the
  record binds (`root_b_*`, `root_f_*`, `msg_*`, `root_*`, `link_*`, `proof_*`) through the hashes, printed by the
  total `printJ` (the bytes `Json.compress` prints).

In deployment the coin server writes the record and draws the coin tree's key; here the prover writes what the server
would, so the record is the honest server's.
-/

namespace FlockComplete

open Flock Transcript
open Lean (Json)

/-- The shape `verifyRepWith` decodes a rep's proof file at, with `extras` extra claims. -/
def shapeOf (st : Setup) (sch : Schedule) (extras : Nat) : Shape :=
  let saltLen : Option Nat := if st.merkle.saltBytes = 0 && !st.openedSalts then none else some st.merkle.saltBytes
  { m := st.m, kLog := st.kLog, claims := 2 + extras, sched := sch, hashLen := st.merkle.digestLen, saltLen }

variable {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **One rep's proof** of the table `tr` committed as `cm0` (its packed table `t`), with the statement's extra claims:
the recursive levels salted from `salt l q`, the nonces from `nonce`. -/
def repP (io : Asks M) (hs : Hashes M) (st : Setup) (sch : Schedule) (tr : Trace) (t : Array F128) (cm0 : Commit)
    (extras : Array Claim) (salt : Nat → Nat → ByteArray) (nonce : Nat → Nat → UInt64) : TMW M Proof := do
  let (zp, zc) ← zerocheckP io st.m st.digest (flat cm0.cap) tr
  let (lp, lc) ← lincheckP io st.fold st.m tr zc
  let (ss, T, sws) ← ringSwitchP io tr (abcClaims st.m st.kLog zc lc ++ extras)
  let lg ← ligeritoP io hs st.merkle sch T cm0 t (basisVals sws (st.m - 7)) salt nonce
  return { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp, ringSwitches := ss,
           ligerito := lg }

/-- Lowercase hex, as the record writes bytes. -/
def hexJ (b : ByteArray) : Json := Json.str (hexOf b)

/-- An F128 as the record writes it (`{hi:016x}{lo:016x}`). -/
def f128J (x : F128) : Json := Json.str x.toHex

/-- `n` copies of the digit `0`. -/
def zeros (n : Nat) : String := String.ofList (List.replicate n '0')

/-- The record's `hello`, and its `coin_tree` under a keyed coin tree: the verifier's `Hello` with a zero nonce in
`coins`, serialized canonically (what `CoinTree.checkHello` compares), and a zero key and root. -/
def helloOf (spec : SessionSpec) : V (String × List (String × Json)) :=
  match spec.coinTree with
  | none => pure (spec.hello, [])
  | some ct => do
    let some own := (Json.parse spec.hello).toOption | throw "S2/R7: the verifier's Hello is not JSON"
    let nonce := zeros 64
    let coins := (own.getObjVal? "coins").toOption.getD Json.null
    let h ← canon (own.setObjVal! "coins" (coins.setObjVal! "nonce" (Json.str nonce)))
    pure (h, [("coin_tree", Json.mkObj [("key", Json.str (zeros 512)), ("nonce", Json.str nonce),
      ("root", Json.str (zeros 128)), ("spec", ct.toJson)])])

/-- A recorded round `k` of a stream: its count, coins and retained bytes, with their digest and length. -/
def roundJ (hs : Hashes M) (spec : SessionSpec) (k : Nat) (r : Round) : M Json := do
  let b := r.bytes.getD ByteArray.empty
  let d ← spec.dgWith hs b
  pure (Json.mkObj [("k", Lean.toJson k), ("n", Lean.toJson r.n), (s!"msg_{spec.dk}", hexJ d),
    ("coins", Json.arr (r.coins.map f128J)), ("msg", hexJ b), ("msg_len", Lean.toJson b.size)])

/-- **The session record**: each table's `root_B`, the link's points, each rep stream's recorded rounds (in
`spec.repStreams` order) and the proof files. -/
def recordJson (hs : Hashes M) (spec : SessionSpec) (rootB : Array ByteArray) (points : Array (Array F128))
    (streams : Array (Array Round)) (proofs : Array ByteArray) : M String := do
  let dk := spec.dk
  let (hello, coinTree) ← (monadLift (helloOf spec) : M _)
  let rootBs := (spec.tables.zip rootB).toList.map fun (t, rb) => (t, hexJ rb)
  let rootBds ← (spec.tables.zip rootB).toList.mapM fun (t, rb) => do pure (t, hexJ (← spec.dgWith hs rb))
  let y := Array.replicate (spec.points * spec.tables.size) (0 : F128)
  let link := Json.mkObj [("mode", Json.str "exchange"), ("sigma", hexJ spec.sigma),
    ("root_b", Json.mkObj rootBs), (s!"root_b_{dk}", Json.mkObj rootBds),
    ("root_f", hexJ spec.sigma), (s!"root_f_{dk}", hexJ (← spec.dgWith hs spec.sigma)),
    ("publics", Json.mkObj []), (s!"publics_{dk}", Json.mkObj []),
    ("points", Json.arr (points.map fun p => Json.arr (p.map f128J))), ("y", Json.arr (y.map f128J))]
  let want := spec.repStreams
  let mut ss : Array Json := #[]
  for ((i, r, name), rs) in want.zip streams do
    let rounds ← (Array.range rs.size).mapM fun k => roundJ hs spec k rs[k]!
    ss := ss.push (Json.mkObj [("stream", Json.str name), ("parent", Json.str ""),
      ("label", Json.str (spec.domain spec.tables[i]! r)), ("at", Lean.toJson (0 : Nat)),
      (s!"root_{dk}", hexJ (← spec.dgWith hs rootB[i]!)), ("rounds", Json.arr rounds)])
  let mut lb := spec.sigma
  for (_, rb) in (spec.tables.zip rootB).qsort (·.1 < ·.1) do lb := lb ++ rb
  for p in points do
    for c in p do lb := F128.pushBytes lb c
  for c in y do lb := F128.pushBytes lb c
  let mut pf : Array Json := #[]
  for ((_, _, name), b) in want.zip proofs do
    pf := pf.push (Json.mkObj [("stream", Json.str name), (dk, hexJ (← spec.dgWith hs b))])
  let draw := spec.unitDraw.getD Json.null
  let j := Json.mkObj ([("kind", Json.str "flock-live-session/v1"), ("aborted", Json.null), ("hello", Json.str hello),
    ("config", Json.mkObj [("require_link", Json.bool true)]), ("link_mode", Json.str "exchange"), ("link", link),
    ("unit_draw", draw), ("streams", Json.arr ss), (s!"link_{dk}", hexJ (← spec.dgWith hs lb)),
    (s!"proof_{dk}", Json.arr pf)] ++ coinTree)
  pure (printJ j)

/-- **The honest session** on each table's trace and packed table: the level-0 commitments, the link, the rep streams;
the record and the proof files (`spec.repStreams` order). -/
def sessionP (io : Asks M) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) (trs : Array Trace)
    (tape : Tape) : M (String × Array ByteArray) := do
  let schs ← tabs.mapM fun st => match fast100 st.m with
    | some sch => pure sch
    | none => throw s!"m = {st.m}: no fast100 schedule"
  let tbls := (Array.range tabs.size).map fun i => packed trs[i]! tabs[i]!.m
  let cms ← (Array.range tabs.size).mapM fun i =>
    commitP hs tabs[i]!.merkle (schs[i]!.levels.getD 0 default) tbls[i]! (tape.salt i 0 0)
  let rootB := cms.map fun c => flat c.cap
  let c1 ← io.link rootB (tabs.map fun _ => ByteArray.empty)
  let points := (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun j => c1 (p * spec.mPts + j)
  let mut streams : Array (Array Round) := #[]
  let mut proofs : Array ByteArray := #[]
  for (i, r, _) in spec.repStreams do
    let st := tabs[i]!
    let extras ← (monadLift (st.extra points) : M _)
    let (p, t) ← (repP io hs st schs[i]! trs[i]! tbls[i]! cms[i]! extras (tape.salt i r) (tape.nonce i r)).run
      (Transcript.init #[] (spec.domain spec.tables[i]! r).toUTF8)
    streams := streams.push t.rounds
    proofs := proofs.push (FlockVBridge.Family.encodeProof (shapeOf st schs[i]! extras.size) p)
  let record ← recordJson hs spec rootB points streams proofs
  return (record, proofs)

end FlockComplete
