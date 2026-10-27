import Proofs.Flock.Claims.ZkCols

/-!
The masked reader `zkRunCols` (`Proofs.Flock.Claims.ZkCols`) on a recorded `flock-circuit --zk` session, held to the
openings the verifier checks: the run-time counterpart of `zkQueryRound_verifyWithM`. A probe run by hand, not a proof
(`exempt` in `lean-audit.json`). For each table and rep, the reader's positions (`zkRepColsOf`, round `zkQueryRound` of
the rep's stream) must pass level 0's Merkle check, the verifier's `merkleCheck` on the proof's opened rows, siblings and
salts, against the table's committed `root_B`, which must be the proof's cap. A Merkle opening fixes its position (the
climb's left and right turns and its cap entry), so passing it is opening exactly where the verifier opened: the
positions of every other round of the stream with at least `q₀` coins must fail it, and so must the reader's with one
position's low bit flipped. The table's `zkRunCols` must be both reps' positions below `2^d`, once each, in any order,
each with its row's first `2^k0` symbols, the witness lanes. Prints one `ZKCOLS` JSON line per table; exits 1 if a
check fails.

    uv run research data fetch-fixtures verity/ml/flock/tests/fixtures/zk-sessions
    cd Security/Proofs && lake env lean --run Flock/Claims/ZkColsSession.lean SESSION_DIR M K_LOG CLAIMS TABLE...

`SESSION_DIR` holds `session.json` and each table's `<TABLE>.rep<r>.bin`; `M`, `K_LOG` and `CLAIMS` (`2 +` points
times regions) are each table's setup, under the `hm96-sha512/v1` leaf scheme. On `test_lean_zk.py`'s `--zk` RoPE d64
sessions (the `zk-sessions` fixture `fixtures/artifacts.json` declares; k_log 22, 8 claims): `sessions/drawn/*` at m 25
with table `circuit`, `sessions/j2/*` at m 25 with tables `circuit-00 circuit-01`, `sessions/m26/*` at m 26 with table
`circuit`.
-/

open Lean Flock

def fail {α : Type} (msg : String) : IO α := throw (IO.userError msg)

def ofV {α : Type} (what : String) : V α → IO α
  | .ok a => pure a
  | .error e => fail s!"{what}: {e}"

def roundsOf (j : Json) (name : String) : IO (Array Round) := do
  let streams ← ofV "streams" (j.getObjValAs? (Array Json) "streams")
  let some s := streams.find? (fun s => (s.getObjValAs? String "stream").toOption == some name)
    | fail s!"no stream {name}"
  let rs ← ofV name (s.getObjValAs? (Array Json) "rounds")
  rs.mapM fun r => do
    let n ← ofV name (r.getObjValAs? Nat "n")
    let coins ← ofV name ((r.getObjValAs? (Array String) "coins").bind (·.mapM f128OfHex))
    pure { n, sha := ByteArray.empty, coins }

def opens (ms : MerkleScheme) (lv : Flock.Level) (zp : Zk.ZkProof) (pos : Array Nat) : Bool :=
  let ip := zp.proof.ligerito.initialProof
  (merkleCheck ms zp.proof.cap lv.cap lv.d pos ip.rows ip.siblings ip.salts (2 ^ lv.lanes + Zk.EXTRA_LANES)).toBool

def sorted (xs : List Nat) : List Nat := xs.mergeSort (· ≤ ·)

def table (j : Json) (dir : String) (m kLog claims : Nat) (t : String) : IO Bool := do
  let ms := MerkleScheme.hm96Sha512
  let reps ← [0, 1].mapM fun r => do
    pure (← roundsOf j s!"{t}/rep{r}", ← IO.FS.readBinFile s!"{dir}/{t}.rep{r}.bin")
  let run : Claims.Flock.ZkRun := { m, kLog, claims, hashLen := ms.digestLen, saltLen := some ms.saltBytes, reps }
  let sh := run.shape
  let lv := sh.sched.levels.getD 0 default
  let q := Claims.Flock.zkQueryRound m kLog sh.sched.k0 claims
  let rootB ← ofV "root_b" ((j.getObjValAs? Json "link").bind (·.getObjValAs? Json "root_b") |>.bind
    (·.getObjValAs? String t) |>.bind unhex)
  let mut ok := true
  let mut out : Array Json := #[]
  let mut opened : Array Nat := #[]
  for ((rounds, bytes), r) in reps.zipIdx do
    let zp ← ofV s!"{t}/rep{r}: decode" (Zk.decode sh bytes)
    let pos := ((Claims.Flock.zkRepColsOf sh rounds bytes).map (·.1)).toArray
    opened := opened ++ pos
    let capIsRoot := zp.proof.cap.foldl (· ++ ·) ByteArray.empty == rootB
    let nAtQ := (rounds[q]?.map (·.n)).getD 0
    let here := opens ms lv zp pos
    let others := (List.range rounds.size).filter fun k => k != q && lv.q ≤ (rounds[k]!).coins.size
    let othersOpen := others.filter fun k => opens ms lv zp (positions lv (rounds[k]!).coins)
    let flipped := opens ms lv zp (pos.modify 0 (· ^^^ 1))
    let good := capIsRoot && nAtQ == lv.q && pos.size == lv.q && here && othersOpen.isEmpty && !flipped
    ok := ok && good
    out := out.push (Json.mkObj [("rep", toJson r), ("rounds", toJson rounds.size), ("coins_at_query_round", toJson nAtQ),
      ("cap_is_root_b", toJson capIsRoot), ("positions", toJson pos.size), ("opens", toJson here),
      ("other_rounds_with_q0_coins", toJson others), ("other_rounds_that_open", toJson othersOpen),
      ("one_flipped_position_opens", toJson flipped), ("ok", toJson good)])
  let cols := Claims.Flock.zkRunCols run
  let colPos := cols.map (·.1)
  let want := (opened.toList.filter (· < 2 ^ lv.d)).eraseDups
  let posOk := sorted colPos == sorted want
  let widthOk := cols.all (·.2.length == 2 ^ lv.lanes)
  ok := ok && posOk && widthOk
  IO.println s!"ZKCOLS {(Json.mkObj [("table", toJson t), ("m", toJson m), ("k_log", toJson kLog),
    ("claims", toJson claims), ("k0", toJson sh.sched.k0), ("q0", toJson lv.q), ("d", toJson lv.d),
    ("query_round", toJson q), ("reps", Json.arr out), ("opened", toJson opened.size), ("run_cols", toJson cols.length),
    ("run_cols_are_the_reps_positions", toJson posOk), ("run_cols_are_2^k0_witness_lanes", toJson widthOk),
    ("ok", toJson ok)]).compress}"
  return ok

def main (args : List String) : IO UInt32 := do
  let dir :: m :: kLog :: claims :: tables@(_ :: _) := args
    | IO.eprintln "usage: ZkColsSession.lean SESSION_DIR M K_LOG CLAIMS TABLE..."; return 2
  let (some m, some kLog, some claims) := (m.toNat?, kLog.toNat?, claims.toNat?)
    | IO.eprintln "M, K_LOG and CLAIMS are natural numbers"; return 2
  let j ← ofV "session.json" (Json.parse (← IO.FS.readFile s!"{dir}/session.json"))
  let mut ok := true
  for t in tables do
    ok := (← table j dir m kLog claims t) && ok
  return if ok then 0 else 1
