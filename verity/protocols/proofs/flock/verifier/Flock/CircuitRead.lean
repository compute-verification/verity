/-
The verifier's circuit file, hashed and parsed once.

`verify` hashes its circuit file (the archive's check that its copy hashes to its key; without an archive, the file's
SHA-512 itself) and parses it (`circuitOfKnowing`). `KeptAt f` is a function's value at one input, made when first asked
for and kept with the equation that says so; `reuse` gives it back where it is asked for at that input, compared in
full, and the function's value anew anywhere else, so what it gives is the function's value on every input
(`KeptAt.reuse_eq`). `CircuitAt` is everything the parse of a circuit file is a function of, the statement by its name
(`CircuitAt.parse`, which is `circuitOfKnowing` at that statement's tags: `CircuitAt.parse_eq`). `CircuitRead` is what a
call keeps of its circuit file: its SHA-512 and its parse. `CircuitRead.Readers` is the two as functions, what a call of
the verifier reads its circuit file through, and whatever is kept they are the readers anew (`CircuitRead.readers_eq`):
that is what lets a call read through what another call kept.
-/
module

public import Flock.ProvedScope
public import Flock.With.SetupEq
public import Flock.Zk

@[expose] public section

namespace Flock

/-- `f`'s value at `input`, made when first asked for and kept. -/
structure KeptAt {α β : Type} (f : α → β) where
  input : α
  value : Thunk β
  value_eq : value.get = f input

namespace KeptAt

variable {α β : Type} {f : α → β}

/-- `f` at `a`, not made before it is asked for. -/
def of (f : α → β) (a : α) : KeptAt f := ⟨a, Thunk.mk fun _ => f a, rfl⟩

/-- `k` at `a`: `k` itself where `a` is its input, compared in full (one pointer comparison where `a` is that very
object), and `f` at `a` anew anywhere else. -/
def reuse [DecidableEq α] (k : KeptAt f) (a : α) : KeptAt f :=
  match withPtrEqDecEq a k.input (fun _ => inferInstance) with
  | isTrue _ => k
  | isFalse _ => of f a

theorem eq_of (k : KeptAt f) : k = of f k.input := by
  cases k with
  | mk input value value_eq =>
    have h : value = Thunk.mk fun _ => f input := Thunk.ext value_eq
    subst h
    rfl

/-- **What is kept, reused, is the function anew, whatever is kept.** -/
theorem reuse_eq [DecidableEq α] (k : KeptAt f) : k.reuse = of f := by
  funext a
  unfold reuse
  split
  · rename_i h _
    rw [h]
    exact k.eq_of
  · rfl

/-- What `reuse` gives at `a` is at `a`, and holds `f a`. -/
theorem reuse_at [DecidableEq α] (k : KeptAt f) (a : α) : (k.reuse a).input = a ∧ (k.reuse a).value.get = f a := by
  rw [reuse_eq]
  exact ⟨rfl, rfl⟩

end KeptAt

/-- The deployed hashes (`With.std`) with the SHA-512 of `file` known to be `sha`; `b == file` is one pointer comparison
when `b` is `file` itself. -/
def stdKnowing (file sha : ByteArray) : Hashes V :=
  Hashes.ofSha512 fun b => if b == file then sha else Sha512.hash b

theorem stdKnowing_eq (file : ByteArray) : stdKnowing file (Sha512.hash file) = With.std := by
  have h : (fun b => if b == file then Sha512.hash file else Sha512.hash b) = Sha512.hash := by
    funext b
    split
    · rename_i h
      rw [ByteArray.ext (eq_of_beq (show (b.data == file.data) = true from h))]
    · rfl
  unfold stdKnowing
  rw [h]

/-- `ProvedScope.circuitOf` with the file's SHA-512 `sha` computed beforehand (`circuitOfKnowing_eq`). -/
def circuitOfKnowing (sha : ByteArray) (tags : Tags) (file : ByteArray) (tables : List (String × ByteArray)) :
    V Circuit :=
  if tags.hiddenOutputs then HmOut.parseCommitWithV (stdKnowing file sha) tags file tables
  else HmRow.parseWith (stdKnowing file sha) tags file tables

theorem circuitOfKnowing_eq (tags : Tags) (file : ByteArray) (tables : List (String × ByteArray)) :
    circuitOfKnowing (Sha512.hash file) tags file tables = ProvedScope.circuitOf tags file tables := by
  unfold circuitOfKnowing ProvedScope.circuitOf
  rw [HmOut.parseCommitWithV_eq, stdKnowing_eq, With.outParseCommitWith_std, With.parseWith_std]

/-- Everything `verify`'s parse of its circuit file is a function of: the statement by its name (`--statement`),
whether its plain proof files are upstream's (`agree --upstream-plain`), whether the session is `--zk`'s and its tables
(`--session-tables`), which set the statement's tags; the SHA-512 the parse takes as the file's; the lookup tables; and
the file. The file is last, so two that differ in anything else are told apart before their files are compared. -/
structure CircuitAt where
  statement : String
  upstreamPlain : Bool
  zk : Bool
  count : Nat
  sha : ByteArray
  tables : List (String × ByteArray)
  file : ByteArray
  deriving DecidableEq

/-- `circuitOfKnowing` at the tags `verify` gives the statement `named`: upstream's plain form where `upstreamPlain`, and
under `--zk` the `--zk` identity at the file's `mask_log` and the session's tables (`Zk.tags`). -/
def circuitOfNamed (named : Option Tags) (upstreamPlain zk : Bool) (count : Nat) (sha : ByteArray)
    (tables : List (String × ByteArray)) (file : ByteArray) : V Circuit :=
  match named with
  | none => .error "unknown statement"
  | some tags =>
    let tags := if upstreamPlain then { tags with upstreamPlain := true } else tags
    circuitOfKnowing sha (if zk then Zk.tags tags (Zk.maskLogOf file) count else tags) file tables

/-- `verify`'s parse of its circuit file (`CircuitAt.parse_eq`). -/
def CircuitAt.parse (a : CircuitAt) : V Circuit :=
  circuitOfNamed (Tags.byName? a.statement) a.upstreamPlain a.zk a.count a.sha a.tables a.file

/-- **`CircuitAt.parse` is `circuitOfKnowing` at the statement's tags as `verify` sets them.** -/
theorem CircuitAt.parse_eq {statement : String} {tags : Tags} (h : Tags.byName? statement = some tags)
    (upstreamPlain zk : Bool) (count : Nat) (sha : ByteArray) (tables : List (String × ByteArray)) (file : ByteArray) :
    CircuitAt.parse ⟨statement, upstreamPlain, zk, count, sha, tables, file⟩ =
      circuitOfKnowing sha
        (if zk then
            Zk.tags (if upstreamPlain then { tags with upstreamPlain := true } else tags) (Zk.maskLogOf file) count
          else if upstreamPlain then { tags with upstreamPlain := true } else tags)
        file tables := by
  show circuitOfNamed (Tags.byName? statement) upstreamPlain zk count sha tables file = _
  generalize Tags.byName? statement = named at h
  subst h
  rfl

/-- What a call keeps of its circuit file: the file's SHA-512, and its parse at everything the parse is a function of. -/
structure CircuitRead where
  hash : KeptAt Sha512.hash
  parse : KeptAt CircuitAt.parse

/-- A circuit file's SHA-512 and its parse, each with the input it is at: what a call of the verifier reads its circuit
file through. -/
structure CircuitRead.Readers where
  hash : ByteArray → KeptAt Sha512.hash
  parse : CircuitAt → KeptAt CircuitAt.parse

/-- The SHA-512 and the parse themselves, each made anew of what it is given. -/
def CircuitRead.Readers.anew : CircuitRead.Readers := ⟨KeptAt.of Sha512.hash, KeptAt.of CircuitAt.parse⟩

/-- The readers through what `kept` holds: a file's SHA-512 is the kept one where the file's bytes are the kept file's,
and its parse the kept one where everything the parse is a function of is what the kept parse is at, each compared in
full and with nothing else; anything else is made anew. With nothing kept, the readers anew. -/
def CircuitRead.readers : Option CircuitRead → CircuitRead.Readers
  | none => .anew
  | some kept => ⟨kept.hash.reuse, kept.parse.reuse⟩

/-- **The readers through what is kept are the readers anew, whatever is kept.** -/
theorem CircuitRead.readers_eq (kept : Option CircuitRead) : CircuitRead.readers kept = .anew := by
  cases kept with
  | none => rfl
  | some kept =>
    show (⟨kept.hash.reuse, kept.parse.reuse⟩ : CircuitRead.Readers) = ⟨KeptAt.of _, KeptAt.of _⟩
    rw [KeptAt.reuse_eq, KeptAt.reuse_eq]

end Flock
