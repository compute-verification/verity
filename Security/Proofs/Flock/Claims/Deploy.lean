import Proofs.Flock.Claims.DeployOC
import Proofs.Flock.Claims.DeployInputs
import Proofs.Flock.Level3.ArithFacts

/-!
# Flock's deployed verifier in public-coin form (`Deploy.V`)

The shipped verifier (`flock-verify verify`, plain or `--zk`) as a `Claims.Verifier`: the prover's messages are the
live session's, each answered by that round's live coin, and the last is decided by the shipped verdict, with every
SHA-512 call an oracle query.

* **Coins.** A live coin (`Coin`) is the `65536` GF(2^128) elements a round may draw (S12: `1 ≤ n ≤ 65536`); a round of
  `n` coins reads the first `n` (`Coin.vals`), the link's points the first `points · m_pts` (`pointsOf`). The verifier's
  answer `b_j` is the coin's bytes (`Coin.bytes`).
* **Messages.** `a_1` is the link message (tag `0`: each table's `root_B` and public words, `linkBody`); `a_2, …` are the
  rounds' framed bytes (tag `0`), the session's rep streams in `spec.repStreams` order, each stream's rounds in order;
  the last (tag `1`, `Final`) carries the statement, the record's text and the proof files. A message of tag `0` is
  answered by its coin; one of tag `1` is decided; any other is rejected (`act`).
* **The decision** (`decision`): the shipped verdict over `OC` (`verifyWithM Hashes.oc`, or `Zk.verifyWithM`), whose
  every SHA-512 call is an `OC.query`, the live coins' digests among them (S12's `msg_sha512`, `dgWith`); then, with no
  hash, the live session's custody (`checks`): the last message's statement is `x`, the record's draw is the statement's,
  the record's link and rounds are the live messages and coins (`custody`); then the scope (`ProvedScope.check`) and a
  draw-less session's refusal (`drawLaw zk none`), as `verifyCmd` orders them.
* **Rounds.** `k(x) = 2 + Σ_tables 2 · repRounds`, the link, every recorded round and the last message (`rounds`).

`next_rej`, `next_coin` and `next_decide` are the verifier's three steps: a refusal and a coin make no query; a decision
accepts exactly when the shipped verdict against `H` (`verifyWith (hOf H)`, `Zk.verifyWith (hOf H)`) and the custody
checks are `.ok`, and its oracle log has `shaCalls (hOf H)` (`Zk.shaCalls`) entries.
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-! ## Coins -/

/-- **A live coin**: the `65536` GF(2^128) elements a round may draw. -/
def Coin : Type := Fin 65536 → F128

instance : Inhabited Coin := ⟨fun _ => 0⟩
instance : Nonempty Coin := ⟨default⟩
instance : Fintype Coin := inferInstanceAs (Fintype (Fin 65536 → FlockLevel3.GF128))

/-- The coin's element `i`, `0` past its end. -/
def Coin.at (c : Coin) (i : ℕ) : F128 := if h : i < 65536 then c ⟨i, h⟩ else 0

/-- A round's `n` coins: the first `n` elements. -/
def Coin.vals (c : Coin) (n : ℕ) : Array F128 := (Array.range n).map c.at

/-- **The verifier's answer**: the coin's bytes, every element's 16 bytes in order. -/
def Coin.bytes (c : Coin) : Claims.Msg := (List.finRange 65536).flatMap fun i => (c i).toBytes.data.toList

/-- The link's points: `spec.points` points of `spec.mPts` coordinates, the coin's first elements, point-major. -/
def pointsOf (spec : SessionSpec) (c : Coin) : Array (Array F128) :=
  (Array.range spec.points).map fun p => (Array.range spec.mPts).map fun i => c.at (p * spec.mPts + i)

/-! ## Messages -/

/-- The link message's body: each table's `root_B`, then each table's public words, in the session's table order. -/
def linkBody (rootB publics : List ByteArray) : Claims.Msg := encList encBytes rootB ++ encList encBytes publics

/-- The link message read back: tag `0`, then `linkBody`. -/
def linkOf : Claims.Msg → Option (List ByteArray × List ByteArray)
  | 0 :: body =>
    match (do let r ← rdList rdBytes; let p ← rdList rdBytes; pure (r, p) : Rd _).run body with
    | some (v, []) => some v
    | _ => none
  | _ => none

/-- **The prover's last message**: the statement, the record's text and the proof files (`spec.repStreams` order). -/
structure Final where
  stmt : Claims.Msg
  record : String
  proofs : Array ByteArray

/-- The last message: tag `1`, then the statement, the record's text and the proof files. -/
def Final.encode (f : Final) : Claims.Msg :=
  1 :: (encNat f.stmt.length ++ f.stmt ++ encStr f.record ++ encList encBytes f.proofs.toList)

def Final.read : Rd Final := do
  let n ← rdNat
  let stmt ← rdTake n
  let record ← rdStr
  let proofs ← rdList rdBytes
  pure ⟨stmt, record, proofs.toArray⟩

/-- The last message read back. -/
def finalOf : Claims.Msg → Option Final
  | 1 :: body =>
    match Final.read.run body with
    | some (f, []) => some f
    | _ => none
  | _ => none

/-! ## The live session in the record -/

/-- What the record says of the live session: each table's `root_B` and public words, the link's points, and each rep
stream's rounds (count, coins and retained bytes), in `spec.repStreams` order. -/
structure Live where
  rootB : Array ByteArray
  publics : Array ByteArray
  points : Array (Array F128)
  streams : Array (Array Round)

/-- The record's live session, read with no hash (`Record.decodeWith`'s fields, without its digests). -/
def liveOf (spec : SessionSpec) (record : String) : V Live := do
  let j ← match Json.parse record with
    | .ok j => pure j
    | .error e => throw s!"record: not JSON: {e}"
  let l ← Record.fld j "link"
  let rootB ← spec.tables.mapM fun t => do unhex (← Record.str (← Record.fld l "root_b") t)
  let pubs ← Record.fld l "publics"
  let publics ← spec.tables.mapM fun t =>
    if Record.isNull pubs t then pure ByteArray.empty else do unhex (← Record.str pubs t)
  let points ← (← Record.arr l "points").mapM fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => .error "record: a link point is not a list"
  let recs ← Record.arr j "streams"
  let streams ← spec.repStreams.mapM fun (_, _, name) => do
    let some s := recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name)
      | throw s!"S11: no stream {name}"
    (← Record.arr s "rounds").mapM fun r => do
      let n ← Record.nat r "n"
      let coins ← Record.f128s (← Record.arr r "coins")
      let bytes ← if Record.isNull r "msg" then pure none else some <$> unhex (← Record.str r "msg")
      pure (⟨n, ByteArray.empty, coins, bytes⟩ : Round)
  pure ⟨rootB, publics, points, streams⟩

/-- **The live session's custody**: the record's `root_B` and public words are the link message's, its points the
first coin's, and its rounds, in order, the round messages with their coins: each round's retained bytes the message,
its coins the first `n` of the message's coin. -/
def custody (spec : SessionSpec) (lv : Live) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) : V Unit := do
  Record.check (lv.rootB.toList == link.1) "live: the record's root_B is not the link message's"
  Record.check (lv.publics.toList == link.2) "live: the record's public words are not the link message's"
  Record.check (lv.points == pointsOf spec c₁) "live: the record's link points are not the first coin"
  let rounds := lv.streams.toList.flatMap (·.toList)
  Record.check (rounds.length == mids.length) "live: the record's rounds are not the round messages"
  Record.check ((rounds.zip mids).all fun (r, a, c) => r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n)
    "live: a recorded round is not its message and coin"

/-! ## The verifier -/

/-- The setups of the session's tables. -/
def tabsOf (sts : Array Stmt) : Array Setup := sts.map Setup.ofCircuit

/-- The shipped verdict with `hs` as the hashes: `Zk.verifyWithM` for the masked verifier, else `verifyWithM`. -/
def verdictM {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (zk : Bool) (hs : Hashes M)
    (spec : SessionSpec) (tabs : Array Setup) (record : String) (proofs : Array ByteArray) : M Unit :=
  if zk then Zk.verifyWithM hs spec tabs record proofs else verifyWithM hs spec tabs record proofs

/-- The shipped verdict with SHA-512 replaced by `h` (`verifyWith h`, `Zk.verifyWith h`). -/
def verdictWith (zk : Bool) (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : V Unit :=
  if zk then Zk.verifyWith h spec tabs record proofs else verifyWith h spec tabs record proofs

/-- The shipped verdict's SHA-512 calls with `h` (`shaCalls h`, `Zk.shaCalls h`). -/
def shaCallsOf (zk : Bool) (h : ByteArray → ByteArray) (spec : SessionSpec) (tabs : Array Setup) (record : String)
    (proofs : Array ByteArray) : ℕ :=
  if zk then Zk.shaCalls h spec tabs record proofs else shaCalls h spec tabs record proofs

/-- **The checks after the verdict**, with no hash: the last message's statement is `x`, the record's draw is the
statement's, the live session's custody, the verifier's scope check, and a draw-less session's refusal. -/
def checks (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) : V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (FlockSoundness.Discharge.Exec.recordDraw f.record == s.read.draw)
    "live: the record's draw is not the statement's"
  custody spec (← liveOf spec f.record) link c₁ mids
  s.read.check
  if s.read.draw.isNone then ProvedScope.drawLaw zk none

/-- **The decision**: the shipped verdict over `OC`, then the checks. -/
def decision (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit := do
  verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs
  monadLift (checks zk s spec x link c₁ mids f)

/-- What the verifier does at the prover's messages: refuse, answer a coin, or decide. -/
inductive Act where
  | rej
  | coin (c : Coin)
  | decide (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
      (mids : List (Claims.Msg × Coin)) (f : Final)

/-- **The verifier's step** at the prover's messages `as` and the coins `cs` (one per message): no setup, refuse; the
last message of tag `0`, answer its coin; of tag `1`, after the link message, decide on the link, its coin, the round
messages with theirs, and the last message; anything else, refuse. -/
def act (zk : Bool) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Act :=
  match setupOf zk x with
  | none => .rej
  | some (s, spec, sts) =>
    match as.getLast?, cs.getLast? with
    | some (0 :: _), some c => .coin c
    | some a, _ =>
      match finalOf a, as, cs with
      | some f, a₁ :: rest, c₁ :: crest =>
        match linkOf a₁ with
        | some link => .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) f
        | none => .rej
      | _, _, _ => .rej
    | none, _ => .rej

/-- A decision as the verifier's output: accept on `.ok`, else reject. -/
def outOf (d : Claims.OCV Unit) : Claims.OC Claims.Out :=
  d.run.bind fun r => .ret (match r with
    | .ok () => .acc
    | .error _ => .rej)

/-- The verifier's next message. -/
def next (zk : Bool) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  match act zk x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s spec sts link c₁ mids f => outOf (decision zk s spec sts x link c₁ mids f)

/-- A rep's recorded rounds: the masked replay's `m + kLog − 6`, else zerocheck's `m − 3` and lincheck's `kLog − 3`;
the ring switch's `claims + 1`; Ligerito's `k0 + 5 + 11R` (`R + 1` levels), with the masked one's level-0 out-of-domain
point and coin and its four inner rounds. -/
def repRounds (zk : Bool) (spec : SessionSpec) (st : Stmt) : ℕ :=
  let claims := 2 + spec.points * st.regions.size
  let kLog := st.c.kLog
  if zk then
    match Zk.schedule st.m with
    | .ok (sch, _) => (st.m + kLog - 6) + (claims + 1) + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4
    | .error _ => 0
  else
    match fast100 st.m with
    | some sch => (st.m - 3) + (kLog - 3) + (claims + 1) + (sch.k0 + 5 + 11 * (sch.levels.size - 1))
    | none => 0

/-- The session's recorded rounds: each table's two reps. -/
def rounds (zk : Bool) (spec : SessionSpec) (sts : Array Stmt) : ℕ :=
  sts.foldl (fun acc st => acc + 2 * repRounds zk spec st) 0

/-- **Flock's deployed verifier in public-coin form**, masked when `zk`: `k(x)` is the link message, the session's
recorded rounds and the last message (one round where the statement does not set up). -/
def V (zk : Bool) : Claims.Verifier Coin where
  k x := match setupOf zk x with
    | some (_, spec, sts) => 2 + rounds zk spec sts
    | none => 1
  next := next zk

/-! ## The verifier's steps against `H` -/

theorem next_rej (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    (h : act zk x as cs = .rej) : (next zk x as cs).eval H = (.rej, []) := by
  simp only [next, h, Claims.OC.eval]

/-- **A coin is answered with no query.** -/
theorem next_coin (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {c : Coin} (h : act zk x as cs = .coin c) : (next zk x as cs).eval H = (.msg c.bytes, []) := by
  simp only [next, h, Claims.OC.eval]

/-- The verdict over `OC` against `H` is the verdict with `hOf H`. -/
theorem verdictM_eval (H : Claims.Msg → Claims.Msg) (zk : Bool) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) :
    ((verdictM zk Claims.Hashes.oc spec tabs record proofs).run.eval H).1 =
      verdictWith zk (Claims.hOf H) spec tabs record proofs := by
  unfold verdictM verdictWith
  cases zk
  · exact Claims.verifyOC_eval H spec tabs record proofs
  · exact Claims.Zk.verifyOC_eval H spec tabs record proofs

/-- Its oracle log has the verdict's `shaCalls` entries. -/
theorem verdictM_log_length (H : Claims.Msg → Claims.Msg) (zk : Bool) (spec : SessionSpec) (tabs : Array Setup)
    (record : String) (proofs : Array ByteArray) :
    ((verdictM zk Claims.Hashes.oc spec tabs record proofs).run.eval H).2.length =
      shaCallsOf zk (Claims.hOf H) spec tabs record proofs := by
  unfold verdictM shaCallsOf
  cases zk
  · exact Claims.verifyOC_log_length H spec tabs record proofs
  · exact Claims.Zk.verifyOC_log_length H spec tabs record proofs

/-- The decision against `H`: its verdict is the shipped verdict with `hOf H`, then the checks; its log is the
verdict's. -/
theorem decision_eval (H : Claims.Msg → Claims.Msg) (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : Final) :
    ((decision zk s spec sts x link c₁ mids f).run.eval H).1 =
        (do verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
            checks zk s spec x link c₁ mids f) ∧
      ((decision zk s spec sts x link c₁ mids f).run.eval H).2.length =
        shaCallsOf zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs := by
  have he := verdictM_eval H zk spec (tabsOf sts) f.record f.proofs
  have hl := verdictM_log_length H zk spec (tabsOf sts) f.record f.proofs
  simp only [decision, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs).run.eval H with ⟨r, l⟩
  rw [hv] at he hl
  simp only at he hl
  subst he
  cases verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs with
  | error e =>
    refine ⟨rfl, ?_⟩
    show (l ++ []).length = _
    rw [List.append_nil, hl]
  | ok u =>
    refine ⟨rfl, ?_⟩
    show (l ++ []).length = _
    rw [List.append_nil, hl]

/-- **The verifier's decision against `H`**: it accepts exactly when the shipped verdict with SHA-512 replaced by `H`
(`verifyWith (hOf H)`, or `Zk.verifyWith (hOf H)`) and the custody checks are `.ok`, and its oracle log has that
verdict's `shaCalls (hOf H)` entries. -/
theorem next_decide (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval H).1 = .acc ↔
        verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs = .ok () ∧
          checks zk s spec x link c₁ mids f = .ok () := by
  obtain ⟨he, -⟩ := decision_eval H zk s spec sts x link c₁ mids f
  simp only [next, h, outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hv : verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs with
  | error e => simp [bind, Except.bind]
  | ok u =>
    cases checks zk s spec x link c₁ mids f with
    | error e => simp [bind, Except.bind]
    | ok u' => simp [bind, Except.bind]

/-- **The decision's oracle log** has the shipped verdict's `shaCalls (hOf H)` entries. -/
theorem next_decide_log (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval H).2.length = shaCallsOf zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs := by
  obtain ⟨-, hl⟩ := decision_eval H zk s spec sts x link c₁ mids f
  simp only [next, h, outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]
  exact hl

end FlockDeploy
