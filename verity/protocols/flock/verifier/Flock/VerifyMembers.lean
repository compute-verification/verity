module

public import Flock.JsonParse
public import Lean.Data.Json
meta import Flock.JsonParse
meta import Lean.Data.Json

/-!
# `verify-members`: the verifier's `verify` for each member of a run, in one process

A run's members are verified one `verify` call each, a process of its own, and every call reads the run's registered
program, the same bytes, again. `verify-members [--archive DIR] --program F` reads `F` once and keeps what `verify` derives
from a program's bytes alone: its digest and its two readings (`ProgramRead`). `run` is its loop over the members; the
member's check it is given is `verifyMember`, which is `verify` (`verifyMember_eq`).

**What it reads.** Standard input, one line a member, in the members' order. A line is a JSON array of strings, `verify`'s
argument list for that member (what follows the word `verify` on its command line), so a member names its statement,
its files and its sessions exactly as it does to `verify`, its own `--program` among them. A member's index is its
line's, from 0, and every line is a member. A line is read once the member before it is done, so a caller may write a
member's line when that member is ready and read its verdicts before it writes the next. `verify` reads no standard input.

**What it prints.** For each member, on standard output: the lines `verify` prints for that member alone, byte for byte (a
`VERDICT` line for each of its sessions, in order), then one line `MEMBER {"exit":E,"index":I}` (`memberLine`), after
which the output is flushed. `E` is the exit status `verify` ends with on those arguments: 0 when every session is
accepted, 1 when one is refused or an exception stopped it, 2 when it stopped before any session (its usage, an input it
cannot read). A line that is no argument list gets no verdict and `MEMBER {"error":WHY,"exit":2,"index":I}`. Standard
error holds what `verify` writes there for each member, in order, and is flushed before that member's `MEMBER` line is
written: a caller that writes a member's line only once the member before it is answered finds there, when a `MEMBER`
line arrives, the lines of the members answered so far, that member's last, and nothing else.

**Its exit status** is 0 when every member's is 0, else 1; 2, before any member is read, when its own `--program` is not
given or cannot be read.

**Nothing crosses from one member to the next but the kept program's digest and readings.** The loop hands a member's check
that member's argument list and nothing else, and keeps of it only its exit status, for its own. The check itself
(`verifyMember`) takes the kept program and the member's arguments: every file is read again, every setup made again and
every session's coin key held fresh among that member's sessions alone, as in a `verify` call of its own. What is kept is
a function of the kept program's bytes, and a member receives it only where its own program's bytes are those bytes,
compared in full; a member with another program has it read anew, once, as `verify` reads its own. No session's
record, coins, tables, registered reads or verdict is kept, and nothing is keyed by anything but the program's bytes.

**It reaches no other host.** The process reads its standard input, its own program and the files its members' lines
name, and writes its two streams: it opens no connection and writes no file. So it may be started before a member's
session exists and kept while later sessions run: what a challenger's release log holds, and in what order, is decided
by who talks to the challenger, and this process does not.

**It fails closed.** A member that `verify` refuses, cannot read or stops on has that member's lines and exit status, and
the members after it have theirs. If the process dies, the members it did not reach have no `MEMBER` line. A caller
takes as refused: a session whose `VERDICT` line says so; a session of a member with no `VERDICT` line before that
member's `MEMBER` line; and every session of a member with no `MEMBER` line. Such a member can be handed to `verify`
alone, which is the same check.
-/

@[expose] public section

namespace Flock.VerifyMembers

open Lean (Json)

def USAGE : String :=
  "usage: verify-members [--archive DIR] --program F  (standard input: a line a member, the JSON array of verify's arguments for it)"

/-- A member's line as `verify`'s argument list: a JSON array of strings, and nothing else. -/
def memberArgs (line : String) : Except String (List String) := do
  let items ← match Flock.JsonParse.parse line with
    | .ok (.arr items) => pure items
    | .ok _ => throw "a member's line is a JSON array of strings"
    | .error e => throw s!"a member's line is a JSON array of strings: {e}"
  items.toList.mapM fun item => match item with
    | .str s => pure s
    | _ => throw "a member's line is a JSON array of strings: one of its items is not a string"

/-- The line that ends member `index`'s output: its exit status, and for a line that was no argument list why. -/
def memberLine (index : Nat) (status : UInt32) (error : Option String := none) : String :=
  let why := match error with | some e => [("error", Json.str e)] | none => []
  s!"MEMBER {(Json.mkObj ([("exit", Lean.toJson status.toNat), ("index", Lean.toJson index)] ++ why)).compress}"

#guard (memberArgs "[\"--zk\", \"--session\", \"a b\"]\n").toOption == some ["--zk", "--session", "a b"]
#guard (memberArgs "[]").toOption == some []
#guard (memberArgs "[\"--zk\", 1]").toOption == none
#guard (memberArgs "{\"args\": []}").toOption == none
#guard (memberArgs "\n").toOption == none
#guard (memberArgs "[\"--zk\"] [\"--zk\"]").toOption == none
#guard memberLine 3 1 == "MEMBER {\"exit\":1,\"index\":3}"
#guard memberLine 0 2 (some "a\nb") == "MEMBER {\"error\":\"a\\nb\",\"exit\":2,\"index\":0}"

/-- `verify-members`' loop: `member` on each line of standard input in order, as the module's text above says. `member` is
handed one member's argument list and nothing else, and the loop keeps of a member only whether its exit status was 0.
An exception `member` throws ends that member alone, with the line and the status 1 an uncaught one ends `verify` with.
Standard error is flushed before a member's `MEMBER` line is written. -/
def run (member : List String → IO UInt32) : IO UInt32 := do
  let stdin ← IO.getStdin
  let stdout ← IO.getStdout
  let stderr ← IO.getStderr
  let mut index := 0
  let mut all := true
  repeat
    let line ← stdin.getLine
    if line.isEmpty then break
    let done : UInt32 × Option String ← match memberArgs line with
      | .ok args =>
        try
          let status ← member args
          pure (status, none)
        catch e =>
          IO.eprintln s!"uncaught exception: {e}"
          pure (1, none)
      | .error e =>
        IO.eprintln s!"verify-members: member {index}: {e}"
        pure (2, some e)
    stderr.flush
    stdout.putStrLn (memberLine index done.1 done.2)
    stdout.flush
    all := all && done.1 == 0
    index := index + 1
  return (if all then 0 else 1)

end Flock.VerifyMembers
