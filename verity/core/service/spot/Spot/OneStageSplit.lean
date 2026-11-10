/-
One sample proved in several sessions, in parts: a draft beside the one-stage audit on the proofs (`OneStageZk`), which
proves its sample in one session (`OneStageZk.oneSession`). No protocol reads this module and no root imports it.

* **bound**: the most proof blocks one session proves, an `Option Nat` the check takes as an argument (the parameters
  hold none). `none` is no bound. A bound is a power of two (`boundHeld`, `BOUND_FORM`).
* **rule** (`parts`): the sampled proof blocks, ascending, cut into consecutive runs of the bound, the last run the
  rest. With no bound, or a sample of no more proof blocks than it, there is one part, the whole sample. The rule reads
  the bound and the sampled proof blocks and nothing else, so how many parts there are and how large each is follow
  from how many proof blocks are sampled.
* **one part** (`several` false): the check is `OneStageZk.checkBlobsAt` itself, whatever the bound
  (`checkBlobsSplitAt_none`, `checkBlobsSplitAt_one`).
* **several parts** (`severalRun`, `severalAt`): the evidence names one session a part (`sessionsHeld`); the
  challenger's release log opens the sessions' rounds in turn, session `j + 1`'s first after session `j`'s last
  (`inTurnHeld`, `IN_TURN`); and session `j`'s record samples exactly part `j` (`partsHeld`), held before any session
  is verified. The rest is `checkBlobsAt`'s, line for line, its sessions verified by `OneStageZk.run` as it is.
* **refusal**: a sample proved in several sessions is refused even where every session accepts (`SPLIT_REFUSED`): no
  guarantee covers it. So the check accepts only what `checkBlobsAt` accepts (`accept_of_checkBlobsSplitAt`).
-/
module

public import CoreProtocol.Blobs
public import Spot.OneStage
public import Spot.ChallengerTrees
public import Spot.OneStageZk
public import Flock
public import Flock.JsonParse
meta import CoreProtocol.Blobs
meta import Spot.OneStage
meta import Spot.ChallengerTrees
meta import Spot.OneStageZk
meta import Flock
meta import Flock.JsonParse

@[expose] public section
set_option backward.privateInPublic true
set_option backward.privateInPublic.warn false

namespace Spot.OneStageZk.Split

open Lean (Json)
open Flock (V)

/-- The refusal of a bound that is no power of two. -/
def BOUND_FORM : String := "session_units is the most proof blocks one session proves, a power of two"

/-- `T` is a power of two, one included: not zero, and it shares no bit with `T - 1`. -/
def powerOfTwo (T : Nat) : Bool := T != 0 && (T &&& (T - 1)) == 0

/-- The bound held (module docstring, **bound**): none, or a power of two; else a refusal naming it (`BOUND_FORM`). -/
def boundHeld : Option Nat → V Unit
  | none => pure ()
  | some T => if powerOfTwo T then pure () else throw s!"{BOUND_FORM}, not {T}"

/-- **The rule** (module docstring, **rule**), stated here and nowhere else in Lean: the parts of the sampled proof
blocks `blocks`, ascending, one a session in order. With no bound, or no more proof blocks than it, the one part is
`blocks`; else consecutive runs of the bound `T`, the last run the rest, `⌈blocks.size / T⌉` of them. A bound of
zero, which `boundHeld` refuses, cuts nothing. -/
def parts (bound : Option Nat) (blocks : Array Nat) : Array (Array Nat) :=
  match bound with
  | none => #[blocks]
  | some T =>
    if T == 0 || blocks.size ≤ T then #[blocks]
    else (Array.range ((blocks.size + T - 1) / T)).map fun j => blocks.extract (j * T) ((j + 1) * T)

/-- Whether the rule cuts `blocks` into several parts, so that several sessions prove them. -/
def several (bound : Option Nat) (blocks : Array Nat) : Bool := (parts bound blocks).size != 1

/-- The evidence names one session a part (module docstring, **several parts**): where the rule cuts one part, the
evidence's one session, as today (`oneSession`); else as many sessions as parts. -/
def sessionsHeld (bound : Option Nat) (blocks : Array Nat) (sessions : Array RunFiles) : V Unit :=
  let m := (parts bound blocks).size
  if m == 1 then oneSession sessions
  else if sessions.size == m then pure ()
  else throw s!"the evidence names {sessions.size} sessions: the sampled proof blocks are proved in {m}, one a part"

/-- With no bound the sessions' check is today's. -/
theorem sessionsHeld_none (blocks : Array Nat) (sessions : Array RunFiles) :
    sessionsHeld none blocks sessions = oneSession sessions := rfl

/-- Session `j` proved exactly its part: `proved`, its proof blocks ascending, is `part`. A proof block of the part it
left out, one it proved twice and one of another part each refuse. -/
def partHeld (j : Nat) (part proved : Array Nat) : V Unit :=
  if proved == part then pure ()
  else throw s!"session {j} proved other proof blocks than part {j} of the sample"

/-- The sessions' proof blocks `proved`, in session order, each held to its part of `ps` (`partHeld`); a session past
the last part is held to a part of no proof block. -/
def eachPartHeld (ps proved : Array (Array Nat)) : V Unit :=
  proved.zipIdx.forM fun (blocks, j) => partHeld j (ps.getD j #[]) blocks

/-- A session's record on the blobs, refused as `step` refuses it when no blob holds it or it isn't UTF-8. -/
def recordOf (bs : Core.Blobs) (s : RunFiles) : V String := do
  let some record := String.fromUTF8? (← bs.get s.record s!"session {s.name}'s record")
    | throw s!"session {s.name}'s record is not UTF-8"
  pure record

/-- Each session's record samples exactly its part (module docstring, **several parts**): the proof blocks session
`j` proves, as `push` reads them from its record on the public file `pubB` (`provedUnits`), are part `j` of `blocks`
under the rule. The verifier cuts the parts itself; it reads each record and verifies no session. -/
def partsHeld (bound : Option Nat) (pubB : ByteArray) (blocks : Array Nat) (sessions : Array RunFiles)
    (bs : Core.Blobs) : V Unit := do
  eachPartHeld (parts bound blocks) (← sessions.mapM fun s => do provedUnits pubB (← recordOf bs s))

/-- The coin tree whose round release log line `l` opens, by its root (the line's `tree`); none for a line that opens no
round. -/
def openedTree (l : String) : Option String :=
  match Challenger.event l with
  | .ok j =>
    if jsonStrIs (getD j "event") Challenger.OPEN_EVENT then
      match getD j "tree" with
      | .str root => some root
      | _ => none
    else none
  | .error _ => none

/-- `inTurn`'s scan: `turn` the session whose rounds the lines so far opened last. -/
def inTurnFrom (roots : Array String) : Nat → List String → Bool
  | _, [] => true
  | turn, l :: ls =>
    if Challenger.openLine l then
      match (openedTree l).bind (roots.idxOf? ·) with
      | some j => turn ≤ j && inTurnFrom roots j ls
      | none => false
    else inTurnFrom roots turn ls

/-- **The sessions in turn**, in the release log's lines `log`: `roots` the sessions' coin trees' roots in session
order, every line that opens a coin tree's round (`Challenger.openLine`) opens one of a session's tree, and never one
of an earlier session's than the line before it. So session `j + 1`'s first round is opened after session `j`'s last.
Lines that open no round may come anywhere. -/
def inTurn (roots : Array String) (log : List String) : Bool := inTurnFrom roots 0 log

/-- The root of the coin tree a session's record names (its `coin_tree`'s `root`), read as `treesHeld` reads the
record; empty, a root no logged tree has, where it names none. -/
def treeRoot (record : String) : String :=
  match getD (getD ((Flock.JsonParse.parse record).toOption.getD Json.null) "coin_tree") "root" with
  | .str root => root
  | _ => ""

/-- The refusal of a release log that opens the sessions' rounds out of turn (`inTurn`). -/
def IN_TURN : String :=
  "the challenger's release log opens a session's round before an earlier session's last: the sessions are proved " ++
    "in turn, in the evidence's order"

/-- The sessions proved in turn (module docstring, **several parts**): the challenger session's release log, its lines
after the head as `orderHeld` reads them, opens the sessions' rounds in the evidence's order of sessions (`inTurn`),
each session's coin tree the one its record names (`treeRoot`). -/
def inTurnHeld (e : Evidence) (bs : Core.Blobs) : V Unit := do
  let roots ← e.sessions.mapM fun s => treeRoot <$> recordOf bs s
  let lines ← linesOf e
  unless inTurn roots lines.tail do throw s!"coins: {IN_TURN}"

/-- The refusal of a sample proved in several sessions, where every session accepts (module docstring, **refusal**). -/
def SPLIT_REFUSED : String :=
  "the sample is proved in several sessions, each of which accepts: no guarantee covers a sample proved in parts, " ++
    "so it is refused"

/-- The sessions' run on a sample the rule cuts into several parts (module docstring, **several parts**):
`checkBlobsAt`'s lines before v1's check, in its order, with one session a part where it holds one session
(`sessionsHeld`), the sessions in turn after the trees are held (`inTurnHeld`), and each session's record held to its
part before any is verified (`partsHeld`); then the sessions' verdicts on the blobs, by `run` as it is. -/
def severalRun (bound : Option Nat) (p : Params) (c : OneStage.Commitment) (e : Evidence) (bs : Core.Blobs) :
    V Run := do
  sessionsHeld bound e.base.provedUnits e.sessions
  let ts ← treesOf p.base e.base
  orderHeld p c e
  treesHeld p.own ts e bs
  inTurnHeld e bs
  popHeld p.base.verifier c.record
  let pubB ← bs.get p.own.pub "public file"
  publicHeld c.record pubB
  partsHeld bound pubB e.base.provedUnits e.sessions bs
  run p.own e.sessions (some e.base.provedUnits) bs

/-- The check at the coins `ξ` of a sample the rule cuts into several parts: the sessions' run (`severalRun`), then
v1's `checkAt` at `ξ` on its verdicts, as `checkBlobsAt` ends. Where every session accepts and v1 accepts, the refusal
`SPLIT_REFUSED`; what v1 refuses or leaves undecided stays v1's outcome. It never accepts (`severalAt_ne_accept`). -/
def severalAt (bound : Option Nat) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (bs : Core.Blobs) : Core.Outcome OneStage.Public Core.Integrity :=
  match severalRun bound p c e bs with
  | .error err => .reject s!"proofs: {err}"
  | .ok r =>
    match OneStage.checkAt p.base c (some ξ) { e.base with verdicts := r.out } with
    | .accept _ _ =>
      if r.all then .reject s!"proofs: {SPLIT_REFUSED}" else .reject "proofs: a session's verifier did not accept"
    | .reject why => .reject why
    | .undecided why => .undecided why

/-- The check at the coins `ξ` under the bound `bound`, beside `checkBlobsAt`: a bound that is no power of two refuses
(`boundHeld`); where the rule cuts the evidence's proved proof blocks into one part the check is `checkBlobsAt`; where
it cuts several, `severalAt`. -/
def checkBlobsSplitAt (bound : Option Nat) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (bs : Core.Blobs) : Core.Outcome OneStage.Public Core.Integrity :=
  match boundHeld bound with
  | .error why => .reject s!"proofs: {why}"
  | .ok () => if several bound e.base.provedUnits then severalAt bound p c ξ e bs else checkBlobsAt p c ξ e bs

/-- With no bound the check is today's. -/
theorem checkBlobsSplitAt_none (p : Params) (c : OneStage.Commitment) (ξ : ByteArray) (e : Evidence)
    (bs : Core.Blobs) : checkBlobsSplitAt none p c ξ e bs = checkBlobsAt p c ξ e bs := rfl

/-- Under a bound the rule cuts one part by, the check is today's. -/
theorem checkBlobsSplitAt_one (bound : Option Nat) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : Evidence) (bs : Core.Blobs) (hb : boundHeld bound = .ok ()) (h : several bound e.base.provedUnits = false) :
    checkBlobsSplitAt bound p c ξ e bs = checkBlobsAt p c ξ e bs := by
  simp [checkBlobsSplitAt, hb, h]

/-- A sample in several parts is never accepted. -/
theorem severalAt_ne_accept (bound : Option Nat) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : Evidence) (bs : Core.Blobs) (pub : OneStage.Public) (prof : Core.Integrity) :
    severalAt bound p c ξ e bs ≠ .accept pub prof := by
  unfold severalAt
  split
  · exact nofun
  · split
    · split <;> exact nofun
    · exact nofun
    · exact nofun

/-- The check accepts only what today's accepts, whatever the bound. -/
theorem accept_of_checkBlobsSplitAt (bound : Option Nat) (p : Params) (c : OneStage.Commitment) (ξ : ByteArray)
    (e : Evidence) (bs : Core.Blobs) (pub : OneStage.Public) (prof : Core.Integrity)
    (h : checkBlobsSplitAt bound p c ξ e bs = .accept pub prof) : checkBlobsAt p c ξ e bs = .accept pub prof := by
  unfold checkBlobsSplitAt at h
  split at h
  · cases h
  · split at h
    · exact absurd h (severalAt_ne_accept bound p c ξ e bs pub prof)
    · exact h

private def refusal : V Unit → String
  | .error e => e
  | .ok () => ""

/-- The pure checks on sessions' proof blocks `proved`, in session order: the bound, one session a part, each its
part. -/
private def held (bound : Option Nat) (blocks : Array Nat) (proved : Array (Array Nat)) : V Unit := do
  boundHeld bound
  sessionsHeld bound blocks (proved.map fun _ => ⟨"s", "", #[]⟩)
  eachPartHeld (parts bound blocks) proved

/-- Two samples the Python mirror's tests cut too (`check_split`'s): one of 8 proof blocks, one of 5. -/
private def SUBSET : Array Nat := #[0, 2, 3, 7, 11, 12, 16, 19]
private def SAMPLE : Array Nat := #[1, 3, 4, 7, 9]

#guard [0, 3, 5, 6, 12, 2 ^ 40 + 1].all fun T => refusal (boundHeld (some T)) == s!"{BOUND_FORM}, not {T}"
#guard [none, some 1, some 2, some 4, some 8, some 16, some (2 ^ 40)].all fun b => (boundHeld b).toBool

#guard parts (some 4) SUBSET == #[#[0, 2, 3, 7], #[11, 12, 16, 19]]
#guard parts (some 2) SUBSET == #[#[0, 2], #[3, 7], #[11, 12], #[16, 19]]
#guard parts (some 4) SAMPLE == #[#[1, 3, 4, 7], #[9]]
#guard parts (some 2) SAMPLE == #[#[1, 3], #[4, 7], #[9]]
#guard parts (some 1) SAMPLE == #[#[1], #[3], #[4], #[7], #[9]]
#guard [none, some 8, some 16, some (2 ^ 40)].all fun b =>
  [SUBSET, SAMPLE, #[0, 1, 2, 3, 4], #[1, 2, 3, 4], #[]].all fun blocks => parts b blocks == #[blocks]
#guard parts (some 4) #[1, 2, 3, 4] == #[#[1, 2, 3, 4]] && parts (some 1) #[] == #[#[]]
#guard parts (some 2) #[0, 1, 2, 3, 4] == #[#[0, 1], #[2, 3], #[4]] &&
  parts (some 2) #[1, 2, 3, 4] == #[#[1, 2], #[3, 4]]
#guard parts (some 4) #[0, 1, 2, 3, 4] == #[#[0, 1, 2, 3], #[4]] && parts (some 4) #[1, 2, 3, 4] == #[#[1, 2, 3, 4]]

-- The parts joined are the sample, and their sizes follow from how many proof blocks it has and the bound alone.
#guard (List.range 13).all fun K => [1, 2, 4, 8, 16].all fun T =>
  let odd := (Array.range K).map (2 * · + 1)
  let even := (Array.range K).map (2 * ·)
  let sizes := (Array.range (max 1 ((K + T - 1) / T))).map fun j => min T (K - j * T)
  (parts (some T) odd).flatten == odd && (parts (some T) odd).map (·.size) == sizes &&
    (parts (some T) even).map (·.size) == sizes && (held (some T) odd (parts (some T) odd)).toBool

-- One part: one session's proof blocks are held as `provedHeld` holds them, on `OneStageZk`'s own examples.
#guard [none, some 4, some 8].all fun b =>
  [(#[9, 3] ++ #[7], #[3, 7, 9]), (#[3, 7] ++ #[7, 9], #[3, 7, 9]), (#[3] ++ #[9], #[3, 7, 9])].all
    fun ((proved, want) : Array Nat × Array Nat) =>
      (held b want #[proved.qsort (· < ·)]).toBool == (provedHeld proved want).toBool
#guard (held none #[3, 7, 9] #[#[3, 7, 9]]).toBool && !(held none #[3, 7, 9] #[#[3, 7, 7, 9]]).toBool &&
  !(held none #[3, 7, 9] #[#[3, 9]]).toBool
#guard [none, some 4, some 8].all fun b =>
  [#[⟨"s", "", #[]⟩], #[], #[⟨"s", "", #[]⟩, ⟨"t", "", #[]⟩]].all fun (ss : Array RunFiles) =>
    refusal (sessionsHeld b #[3, 7, 9] ss) == refusal (oneSession ss)

-- Several parts, [1, 3], [4, 7], [9]: a dropped proof block, a doubled one, another part's, and a session count.
#guard (held (some 2) SAMPLE #[#[1, 3], #[4, 7], #[9]]).toBool
#guard refusal (held (some 2) SAMPLE #[#[1, 3], #[4], #[9]]) ==
  "session 1 proved other proof blocks than part 1 of the sample"
#guard refusal (held (some 2) SAMPLE #[#[1, 3], #[3, 4, 7], #[9]]) ==
  "session 1 proved other proof blocks than part 1 of the sample"
#guard refusal (held (some 2) SAMPLE #[#[1, 3], #[4, 7], #[9, 9]]) ==
  "session 2 proved other proof blocks than part 2 of the sample"
#guard refusal (held (some 2) SAMPLE #[#[4, 7], #[1, 3], #[9]]) ==
  "session 0 proved other proof blocks than part 0 of the sample"
#guard [#[#[1, 4], #[3, 7], #[9]], #[#[1], #[3, 4], #[7, 9]], #[#[9], #[4, 7], #[1, 3]], #[#[1, 3], #[4, 7], #[8]],
    #[#[3, 1], #[4, 7], #[9]]].all fun (proved : Array (Array Nat)) => !(held (some 2) SAMPLE proved).toBool
#guard refusal (held (some 2) SAMPLE #[#[1, 3], #[4, 7]]) ==
  "the evidence names 2 sessions: the sampled proof blocks are proved in 3, one a part"
#guard [#[#[1, 3], #[4, 7], #[9], #[]], #[#[1, 3, 4, 7, 9]], #[]].all fun (proved : Array (Array Nat)) =>
  (refusal (held (some 2) SAMPLE proved)).startsWith "the evidence names "
#guard !several none SAMPLE && !several (some 8) SAMPLE && several (some 4) SAMPLE && several (some 2) SAMPLE

private def openT (tree stream : String) : String :=
  "{\"after\":\"cd\",\"coins\":\"ef\",\"event\":\"tree-open\",\"n\":1,\"round\":0,\"stream\":\"" ++ stream ++
    "\",\"tree\":\"" ++ tree ++ "\"}"
private def treeT (tree : String) : String :=
  "{\"event\":\"coin-tree\",\"key\":\"02\",\"nonce\":\"03\",\"root\":\"" ++ tree ++ "\",\"spec\":{}}"

#guard inTurn #["aa", "bb"] [treeT "aa", treeT "bb", openT "aa" "session", openT "aa" "t/rep0", openT "bb" "session",
  openT "bb" "t/rep0"]
#guard inTurn #["aa", "bb"] [treeT "aa", openT "aa" "session", openT "aa" "t/rep0", treeT "bb", openT "bb" "session"]
#guard !inTurn #["aa", "bb"] [openT "aa" "session", openT "bb" "session", openT "aa" "t/rep0", openT "bb" "t/rep0"]
#guard !inTurn #["bb", "aa"] [openT "aa" "session", openT "aa" "t/rep0", openT "bb" "session", openT "bb" "t/rep0"]
#guard !inTurn #["aa"] [openT "aa" "session", openT "bb" "session"]
#guard inTurn #["aa"] [openT "aa" "session", openT "aa" "t/rep0", openT "aa" "t/rep0"]
#guard inTurn #["aa", "bb", "cc"] [openT "aa" "session", openT "cc" "session"]
#guard inTurn #[] [] && !inTurn #[] [openT "aa" "session"]
#guard treeRoot "{\"coin_tree\":{\"root\":\"aa\"}}" == "aa" && treeRoot "{}" == "" && treeRoot "no record" == ""

end Spot.OneStageZk.Split
