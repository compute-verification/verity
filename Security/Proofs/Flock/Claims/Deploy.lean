import Proofs.Flock.Claims.DeployOC
import Proofs.Flock.Claims.DeployInputs
import Proofs.Flock.Claims.JsonEq
import Proofs.Flock.Level3.ArithFacts
import Flock.With.Deployed
import Flock.With.SetupEq
import Flock.RecordText

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
* **The decision** (`decision`): the statement's setup over `OC` (`setupM`, `Stmt.setupTablesWith Hashes.oc`), then at
  its session the shipped verdict over `OC` (`verifyWithM Hashes.oc`, or `Zk.verifyWithM`), every SHA-512 call of both
  an `OC.query`, the session's digests and the live coins' among them (S12's `msg_sha512`, `dgWith`); then, with no
  hash, the live session's custody (`checks`): the last message's statement is `x`, the record's draw is the statement's,
  the record's link and rounds are the live messages and coins (`custody`); then the scope (`ProvedScope.check`) and a
  draw-less session's refusal (`drawLaw zk tags false none`), and last S0, the record in its canonical text (`RecordText.check`),
  as `verifyCmd` orders them.
* **Rounds.** `k(x) = 2 + Σ_tables 2 · repRounds`, the link, every recorded round and the last message (`rounds`), of
  the session the statement sets up at SHA-512 (`setupOf`).

`next_rej`, `next_coin` and `next_decide` are the verifier's three steps: a refusal and a coin make no query; a decision
accepts exactly when the setup against `H` (`setupWith (hOf H)`) and at its session the shipped verdict
(`verifyWith (hOf H)`, `Zk.verifyWith (hOf H)`) and the custody checks are `.ok`, and its oracle log has the setup's
`setupCallsOf (hOf H)` entries, then the verdict's `shaCalls (hOf H)` (`Zk.shaCalls`) (`next_decide_log`). At the
SHA-512 oracle (`sha512`) the setup is the statement's and the verdict the deployed `verify` (`Zk.verify`)
(`next_decide_sha512`, `next_decide_log_sha512`).
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-! ## Coins -/

/-- **A live coin**: the `65536` GF(2^128) elements a round may draw. -/
def Coin : Type := Fin 65536 → F128

instance instInhabitedCoin : Inhabited Coin := ⟨fun _ => 0⟩
instance instNonemptyCoin : Nonempty Coin := ⟨default⟩
instance instFintypeCoin : Fintype Coin := inferInstanceAs (Fintype (Fin 65536 → FlockLevel3.GF128))

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

/-- **The statement's setup** with `hs` as the hashes: `Stmt.setupTablesWith` on what `Read.setup` reads. -/
def setupM {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT V M] (hs : Hashes M) (r : Readable.Read) :
    M (SessionSpec × Array Stmt) :=
  Stmt.setupTablesWith hs r.tags r.I.circuitFile r.I.publicFile r.coins r.I.count r.I.tables r.draw r.I.partition
    r.I.program r.I.work r.I.stratified r.I.closure r.publicInputs

/-- The statement's setup with SHA-512 replaced by `h`. -/
def setupWith (h : ByteArray → ByteArray) (r : Readable.Read) : V (SessionSpec × Array Stmt) :=
  setupM (Hashes.ofSha512 h) r

/-- The setup's SHA-512 calls with `h` (`Stmt.setupCalls h`). -/
def setupCallsOf (h : ByteArray → ByteArray) (r : Readable.Read) : ℕ :=
  Stmt.setupCalls h r.tags r.I.circuitFile r.I.publicFile r.coins r.I.count r.I.tables r.draw r.I.partition r.I.program
    r.I.work r.I.stratified r.I.closure r.publicInputs

/-- **The checks after the verdict**, with no hash: the last message's statement is `x`, the record's draw is the
statement's whatever its key order (`drawEq`: `JsonEquiv`, objects as finite maps, as the service's dict equality), the
live session's custody, the verifier's scope check, a draw-less session's refusal, and last the record's canonical text
(S0, `RecordText.check`: the record is what `printT` prints of what it parses to). -/
def checks (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) : V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw)
    "live: the record's draw is not the statement's"
  custody spec (← liveOf spec f.record) link c₁ mids
  s.read.check
  (if s.read.draw.isNone then ProvedScope.drawLaw zk s.read.tags false none else pure ())
  RecordText.check f.record

/-- **The decision at a session**: the shipped verdict over `OC` on its tables, then the checks. -/
def decisionAt (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit := do
  verdictM zk Claims.Hashes.oc spec (tabsOf sts) f.record f.proofs
  monadLift (checks zk s spec x link c₁ mids f)

/-- **The decision**: the statement's setup over `OC`, then the decision at its session. -/
def decision (zk : Bool) (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAt zk s spec sts x link c₁ mids f

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
  | .decide s _ _ link c₁ mids f => outOf (decision zk s x link c₁ mids f)

/-- A rep's recorded rounds: the masked replay's `m + (kLog − 6)`, else zerocheck's `3 + (m − 6)` and lincheck's
`3 + (kLog − 6)`; the ring switch's `claims + 1`; Ligerito's `k0 + 5 + 11R` (`R + 1` levels), with the masked one's
two lane-phase coins and its four inner rounds. -/
def repRounds (zk : Bool) (spec : SessionSpec) (st : Stmt) : ℕ :=
  let claims := 2 + spec.points * st.regions.size
  let kLog := st.c.kLog
  if zk then
    match Zk.schedule st.m with
    | .ok (sch, _) => (st.m + (kLog - 6)) + (claims + 1) + (sch.k0 + 7 + 11 * (sch.levels.size - 1)) + 4
    | .error _ => 0
  else
    match fast100 st.m with
    | some sch => (3 + (st.m - 6)) + (3 + (kLog - 6)) + (claims + 1) + (sch.k0 + 5 + 11 * (sch.levels.size - 1))
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

/-- The setup over `OC` against `H` is the setup with `hOf H`, and its oracle log has `setupCallsOf (hOf H)` entries. -/
theorem setupM_eval (H : Claims.Msg → Claims.Msg) (r : Readable.Read) :
    ((setupM Claims.Hashes.oc r).run.eval H).1 = setupWith (Claims.hOf H) r ∧
      ((setupM Claims.Hashes.oc r).run.eval H).2.length = setupCallsOf (Claims.hOf H) r :=
  ⟨Claims.setupOC_eval H _ _ _ _ _ _ _ _ _ _ _ _ _, Claims.setupOC_log_length H _ _ _ _ _ _ _ _ _ _ _ _ _⟩

/-- The decision at a session against `H`: its verdict is the shipped verdict with `hOf H`, then the checks; its log is
the verdict's. -/
theorem decisionAt_eval (H : Claims.Msg → Claims.Msg) (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt)
    (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin))
    (f : Final) :
    ((decisionAt zk s spec sts x link c₁ mids f).run.eval H).1 =
        (do verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
            checks zk s spec x link c₁ mids f) ∧
      ((decisionAt zk s spec sts x link c₁ mids f).run.eval H).2.length =
        shaCallsOf zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs := by
  have he := verdictM_eval H zk spec (tabsOf sts) f.record f.proofs
  have hl := verdictM_log_length H zk spec (tabsOf sts) f.record f.proofs
  simp only [decisionAt, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
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

/-- The decision against `H`: its verdict is the setup with `hOf H`, then at its session the shipped verdict with
`hOf H` and the checks; its log is the setup's, then the verdict's where the setup is `.ok`. -/
theorem decision_eval (H : Claims.Msg → Claims.Msg) (zk : Bool) (s : Ins) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    ((decision zk s x link c₁ mids f).run.eval H).1 =
        (do let (spec, sts) ← setupWith (Claims.hOf H) s.read
            verdictWith zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
            checks zk s spec x link c₁ mids f) ∧
      ((decision zk s x link c₁ mids f).run.eval H).2.length =
        setupCallsOf (Claims.hOf H) s.read +
          match setupWith (Claims.hOf H) s.read with
          | .ok (spec, sts) => shaCallsOf zk (Claims.hOf H) spec (tabsOf sts) f.record f.proofs
          | .error _ => 0 := by
  obtain ⟨he, hl⟩ := setupM_eval H s.read
  simp only [decision, ExceptT.run_bind, Claims.OC.bind_eq, Claims.OC.eval_bind]
  rcases hv : (setupM Claims.Hashes.oc s.read).run.eval H with ⟨r, l⟩
  rw [hv] at he hl
  simp only at he hl
  subst he
  cases hr : setupWith (Claims.hOf H) s.read with
  | error e =>
    refine ⟨rfl, ?_⟩
    show (l ++ []).length = _
    rw [List.append_nil, hl, Nat.add_zero]
  | ok p =>
    obtain ⟨spec, sts⟩ := p
    obtain ⟨he', hl'⟩ := decisionAt_eval H zk s spec sts x link c₁ mids f
    exact ⟨he', by rw [List.length_append, hl, hl']⟩

/-- **The verifier's decision against `H`**: it accepts exactly when the setup with SHA-512 replaced by `H` is `.ok`,
and at its session the shipped verdict with `hOf H` (`verifyWith (hOf H)`, or `Zk.verifyWith (hOf H)`) and the custody
checks are `.ok`. -/
theorem next_decide (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval H).1 = .acc ↔
      ∃ spec' sts', setupWith (Claims.hOf H) s.read = .ok (spec', sts') ∧
        verdictWith zk (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs = .ok () ∧
          checks zk s spec' x link c₁ mids f = .ok () := by
  obtain ⟨he, -⟩ := decision_eval H zk s x link c₁ mids f
  simp only [next, h, outOf, Claims.OC.eval_bind, Claims.OC.eval]
  rw [he]
  cases hs : setupWith (Claims.hOf H) s.read with
  | error e => simp [bind, Except.bind]
  | ok p =>
    obtain ⟨spec', sts'⟩ := p
    cases hv : verdictWith zk (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs with
    | error e => simp [bind, Except.bind, hv]
    | ok u =>
      cases hc : checks zk s spec' x link c₁ mids f with
      | error e => simp [bind, Except.bind, hv, hc]
      | ok u' => simp [bind, Except.bind, hv, hc]

/-- **The decision's oracle log**: the setup's `setupCallsOf (hOf H)` entries, then, where the setup is `.ok`, the
shipped verdict's `shaCalls (hOf H)` (`Zk.shaCalls`) at its session. -/
theorem next_decide_log (H : Claims.Msg → Claims.Msg) {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg}
    {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval H).2.length =
      setupCallsOf (Claims.hOf H) s.read +
        match setupWith (Claims.hOf H) s.read with
        | .ok (spec', sts') => shaCallsOf zk (Claims.hOf H) spec' (tabsOf sts') f.record f.proofs
        | .error _ => 0 := by
  obtain ⟨-, hl⟩ := decision_eval H zk s x link c₁ mids f
  simp only [next, h, outOf, Claims.OC.eval_bind, Claims.OC.eval, List.append_nil]
  exact hl

/-! ## At SHA-512 -/

/-- A decision is made only on a statement that sets up, at its setup. -/
theorem act_decide {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    setupOf zk x = some (s, spec, sts) := by
  unfold act at h
  split at h
  · cases h
  · rename_i s' spec' sts' hs
    repeat' split at h
    all_goals first | (cases h; done) | (cases h; exact hs)

/-- SHA-512 as an oracle on messages. -/
def sha512 : Claims.Msg → Claims.Msg := fun m => Claims.toMsg (Sha512.hash (Claims.ofMsg m))

theorem hOf_sha512 : Claims.hOf sha512 = Sha512.hash := by
  funext b
  simp [Claims.hOf, sha512]

/-- **At SHA-512 the shipped verdict is the deployed verifier's** (`verify`, `Zk.verify`), on every session the
statement sets up. -/
theorem verdictWith_sha512 {zk : Bool} {x : Claims.Msg} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    (h : setupOf zk x = some (s, spec, sts)) (record : String) (proofs : Array ByteArray) :
    verdictWith zk (Claims.hOf sha512) spec (tabsOf sts) record proofs =
      if zk then Zk.verify spec (tabsOf sts) record proofs else verify spec (tabsOf sts) record proofs := by
  have hr := Ins.setup_read (setupOf_some h).2
  rw [hOf_sha512]
  unfold verdictWith tabsOf
  cases zk
  · exact (verify_eq_verifyWith_of_setupTables _ _ _ _ _ _ _ _ _ _ _ _ _ hr record proofs).symm
  · exact (Zk.verify_eq_verifyWith_of_setupTables _ _ _ _ _ _ _ _ _ _ _ _ _ hr record proofs).symm

/-- At SHA-512 the setup is the shipped one (`Read.setup`). -/
theorem setupWith_sha512 (r : Readable.Read) : setupWith Sha512.hash r = r.setup :=
  (Stmt.setupTables_eq_setupTablesWith _ _ _ _ _ _ _ _ _ _ _ _ _).symm

/-- On a decision, the setup at the SHA-512 oracle is the statement's. -/
theorem act_decide_setupWith {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    setupWith (Claims.hOf sha512) s.read = .ok (spec, sts) := by
  rw [hOf_sha512, setupWith_sha512]
  exact Ins.setup_read (setupOf_some (act_decide h)).2

/-- **The verifier's decision at SHA-512**: it accepts exactly when the deployed verifier (`verify`, or `Zk.verify`)
accepts the record and proof files and the custody checks pass. -/
theorem next_decide_sha512 {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval sha512).1 = .acc ↔
      (if zk then Zk.verify spec (tabsOf sts) f.record f.proofs else verify spec (tabsOf sts) f.record f.proofs) =
          .ok () ∧
        checks zk s spec x link c₁ mids f = .ok () := by
  rw [next_decide sha512 h, act_decide_setupWith h, ← verdictWith_sha512 (act_decide h)]
  constructor
  · rintro ⟨spec', sts', he, hv, hc⟩
    cases he
    exact ⟨hv, hc⟩
  · rintro ⟨hv, hc⟩
    exact ⟨spec, sts, rfl, hv, hc⟩

/-- **The decision's oracle log at SHA-512**: the setup's SHA-512 calls, then the shipped verdict's. -/
theorem next_decide_log_sha512 {zk : Bool} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : Final} (h : act zk x as cs = .decide s spec sts link c₁ mids f) :
    ((next zk x as cs).eval sha512).2.length =
      setupCallsOf Sha512.hash s.read + shaCallsOf zk Sha512.hash spec (tabsOf sts) f.record f.proofs := by
  rw [next_decide_log sha512 h, act_decide_setupWith h, hOf_sha512]

end FlockDeploy
