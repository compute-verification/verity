module

/-!
# The connection rows' bounds

The constants the period's check (`NetworkCheck.Period`) holds a schedule's connection rows to: the bytes a cell's
payload holds under lengths rows, the most slots, ids and classes a row or a schedule has, a class's size, the keys a
schedule has only where it sets them, and what the check says of a schedule with connection rows.
-/

@[expose] public section

namespace NetworkCheck.Period

/-- The byte of an id request past its first prompt id (`io_request.layout`'s `start` and `active.ID_FIELD_BYTES`):
with lengths rows a cell's payload holds at least this much, so a request's head and first prompt id are in its first
cell, each field at its place (`io_request.layout` raises for a smaller cell). -/
def firstPromptEnd : Nat := 378
/-- The most slots a lengths row has (`connection_rows.check_slots`): its proof block returns the whole row, 96 bits a
slot, after its 2 words of 16 bits, in a public size class of at most 262,144 bits. -/
def mostSlots : Nat := 2730
/-- The most ids a response row has (`content_rows.most_ids`): its tie's proof block returns the whole row, 32 bits an
id after its count, behind 16 bits of verdict in a public size class of at most 262,144 bits, and a class's size is a
power of two. -/
def mostIds : Nat := 4096
/-- The most classes a schedule lists (`content_rows.MOST_CLASSES`): the connection rows' head counts its tables in a
u16, three a class after the lengths and the fails tables. -/
def mostClasses : Nat := 21844
/-- The most classes a schedule with request words lists (`content_rows.most_classes`): a class has a fourth table
there, its words table. -/
def mostClassesWords : Nat := 16383

/-- ⌈a / b⌉, and 0 at b = 0. -/
def divCeil (a b : Nat) : Nat := (a + b - 1) / b

/-- The keys a schedule has only where it sets them (`warden.OPTIONAL_FIELDS`). -/
def optionalScheduleKeys : List String := ["content", "lengths", "max_tokens_cap", "request_words"]

/-- A class's size, of prompt ids or of ids released: a power of two of at least 2 (`content_rows.check_size`). -/
def classSize (n : Nat) : Bool := 2 ≤ n && n &&& (n - 1) == 0

/-- What the check says of a schedule with connection rows (`Schedule.connectionRows`). It reads each execution
phase's cells' rows and no connection rows' registration (`connection_rows.check_registration`), so it would hold
nothing of such a schedule's lengths, fails and content tables to the schedule. -/
def connectionRowsRefusal : String :=
  "schedule: connection rows: the check reads no connection rows' registration, so it refuses a schedule with \
    lengths rows, content rows or request words (connectionRowsCheck)"

end NetworkCheck.Period
