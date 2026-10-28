/-
`flock-verify rec-compose`: a recursive statement's verdict from its V* parts' verdicts. Each part is one
`flock-verify verify --zk` process at one V* statement (`--rec-part J`), with one session. The compiled verifier the
recursion's guarantee reads (`RecDeploy.V`, `Proofs.Flock.Claims.Recursion.Deploy`) runs part `s` at the statement
`RecDeploy.partRaw … s p`: of what `verify` reads, only the part's Flock statement (its circuit, partition, program and
registered reads, with its staged public file), its index and the verifier's own bindings of it vary by part, and every
other input is one value for every part (`Shared`, with the shared rows `--rec-rows`): the verifier's inner shape, the
inner statement's files and how it is set up, the inner session's record, the coin server's transcript and the circuit
registration. No part sees another, so only the composition can hold every part to one set. Each part's `VERDICT` names
the SHA-512 of each (`rec_shared`, `rec_rows_sha512`), whether it ran `--zk` (`zk`), and its session's coin key
(`coin_key`).

It accepts exactly when there is one log per V* statement, each with one `VERDICT`; `rec_part` `0` to `n − 1` each once;
every `VERDICT` accepts, ran `--zk`, names no identifier and names the composition's own rows file's digest; the first
part's `rec_shared` holds every input the compiled verifier's parts read, at the composition's own inner shape
(`Shared.Complete`), and every part's is that one; and no two sessions share a coin key (`verify` refuses a key reused
within one run, `CoinTree.checkFresh`, and this across runs). `Accepts` states it, and `check_accepts` proves it of
`check`; `Proofs.Flock.Claims.Recursion.ComposeParts` carries it to the parts' statements.

It doesn't check that the parts' sessions are one run of the compiled verifier, which is the open condition C10
`OneRun`: one session per part, every session the coin server opened for the statement's parts recorded, and the
staging before the first part's Hello. A prover can retry a part and hand over only the session that accepted, since
`checkFresh` and this see only the keys they are handed. `OneRun` needs the service's interactive-run record, a later
PR; until it holds, the recursive verdict is not a claim a guarantee backs.
-/
import Lean.Data.Json
import Flock.Bytes
import Flock.Hash

namespace Flock
namespace RecCompose

open Lean (Json)

/-- A file's SHA-512, in lowercase hex. -/
def digest (b : ByteArray) : String := hexOf (Sha512.hash b)

/-- **What every part of one recursive statement reads alike**, as its `VERDICT` names it (`rec_shared`): the SHA-512 of
the verifier's own inner shape (`--rec-inner`), of the inner statement's files (`--rec-inner-circuit`,
`--rec-inner-public`, `--rec-inner-partition`), of a record of check 5's coins (`--rec-inner-coins`, which `verify`
refuses), of the inner session's record (`--rec-inner-session`'s `session.json`), of the coin server's transcript
(`--rec-inner-transcript`) and of the circuit registration (`--rec-inner-registration`); how the inner statement is set
up (`--rec-inner-statement`, its coins `os` as `true`, each lookup table's name and SHA-512, and the SHA-512 of
`--rec-inner-draw`). `none` where `verify` held no such input. -/
structure Shared where
  recInner : Option String
  innerCircuit : Option String
  innerPublic : Option String
  innerPartition : Option String
  innerCoins : Option String
  innerSession : Option String
  transcript : Option String
  registration : Option String
  innerStatement : Option String
  innerStatementCoins : Option Bool
  innerTables : Option (List (String × String))
  innerDraw : Option String
  deriving DecidableEq, Repr, Inhabited

/-- **The shared inputs' digests**, from the bytes `verify` read. -/
def Shared.ofFiles (recInner innerCircuit innerPublic innerPartition innerCoins innerSession transcript registration :
    Option ByteArray) (innerStatement : Option String) (innerStatementCoins : Option Bool)
    (innerTables : Option (List (String × ByteArray))) (innerDraw : Option ByteArray) : Shared :=
  ⟨recInner.map digest, innerCircuit.map digest, innerPublic.map digest, innerPartition.map digest,
    innerCoins.map digest, innerSession.map digest, transcript.map digest, registration.map digest, innerStatement,
    innerStatementCoins, innerTables.map (·.map fun t => (t.1, digest t.2)), innerDraw.map digest⟩

/-- A string or `null`. -/
def optJson : Option String → Json
  | none => .null
  | some s => .str s

def Shared.toJson (s : Shared) : Json :=
  Json.mkObj [("rec_inner", optJson s.recInner), ("inner_circuit", optJson s.innerCircuit),
    ("inner_public", optJson s.innerPublic), ("inner_partition", optJson s.innerPartition),
    ("inner_coins", optJson s.innerCoins), ("inner_session", optJson s.innerSession),
    ("inner_transcript", optJson s.transcript), ("inner_registration", optJson s.registration),
    ("inner_statement", optJson s.innerStatement),
    ("inner_statement_coins", optJson (s.innerStatementCoins.map fun os => if os then "os" else "seed")),
    ("inner_tables", match s.innerTables with
      | none => .null
      | some ts => .arr (ts.map fun t => Json.arr #[.str t.1, .str t.2]).toArray),
    ("inner_draw", optJson s.innerDraw)]

/-- Field `k` of `j`: a string, or `null` as `none`; nothing when absent or another value. -/
def readOpt (j : Json) (k : String) : Option (Option String) :=
  match j.getObjVal? k with
  | .ok .null => some none
  | .ok (.str s) => some (some s)
  | _ => none

/-- A lookup table as `Shared.toJson` writes it: its name and SHA-512. -/
def readTable : Json → Option (String × String)
  | .arr a => match a.toList with
    | [.str n, .str d] => some (n, d)
    | _ => none
  | _ => none

/-- `Shared.toJson` read back; nothing where a field is absent or not in its form. -/
def Shared.ofJson? (j : Json) : Option Shared := do
  let coins ← match j.getObjVal? "inner_statement_coins" with
    | .ok .null => some none
    | .ok (.str "os") => some (some true)
    | .ok (.str "seed") => some (some false)
    | _ => none
  let tables ← match j.getObjVal? "inner_tables" with
    | .ok .null => some none
    | .ok (.arr ts) => some <$> ts.toList.mapM readTable
    | _ => none
  pure ⟨← readOpt j "rec_inner", ← readOpt j "inner_circuit", ← readOpt j "inner_public", ← readOpt j "inner_partition",
    ← readOpt j "inner_coins", ← readOpt j "inner_session", ← readOpt j "inner_transcript",
    ← readOpt j "inner_registration", ← readOpt j "inner_statement", coins, tables, ← readOpt j "inner_draw"⟩

/-- The first field in which `s` differs from `t`, the first part's, as a refusal's reason. -/
def Shared.diff (s t : Shared) : String :=
  let show? := fun (o : Option String) => o.getD "none"
  let one := fun (key what : String) (a b : Option String) =>
    if a != b then some s!"its {key} ({what}) is {show? a}, not {show? b} as the first part's" else none
  (one "rec_inner" "the verifier's inner shape, --rec-inner" s.recInner t.recInner <|>
    one "inner_circuit" "the inner statement's circuit, --rec-inner-circuit" s.innerCircuit t.innerCircuit <|>
    one "inner_public" "the inner statement's public file, --rec-inner-public" s.innerPublic t.innerPublic <|>
    one "inner_partition" "the inner statement's partition, --rec-inner-partition" s.innerPartition t.innerPartition <|>
    one "inner_coins" "a record of check 5's coins, --rec-inner-coins" s.innerCoins t.innerCoins <|>
    one "inner_session" "the inner session's record, --rec-inner-session" s.innerSession t.innerSession <|>
    one "inner_transcript" "the coin server's transcript, --rec-inner-transcript" s.transcript t.transcript <|>
    one "inner_registration" "the circuit registration, --rec-inner-registration" s.registration t.registration <|>
    one "inner_statement" "how the inner statement is proved, --rec-inner-statement" s.innerStatement t.innerStatement <|>
    one "inner_statement_coins" "the inner statement's coins, --rec-inner-statement-coins"
      (s.innerStatementCoins.map toString) (t.innerStatementCoins.map toString) <|>
    one "inner_tables" "the inner statement's lookup tables, --rec-inner-tables"
      (s.innerTables.map toString) (t.innerTables.map toString) <|>
    one "inner_draw" "the inner statement's draw, --rec-inner-draw" s.innerDraw t.innerDraw).getD
    "its rec_shared is not the first part's"

/-- **A part's verdict as `rec-compose` reads it** off its `VERDICT` line. -/
structure Verdict where
  accepted : Bool
  why : String
  part : Option Nat
  rows : Option String
  shared : Option Shared
  zk : Option Bool
  coinKey : Option String
  identifier : Bool
  deriving Repr, Inhabited

def Verdict.ofJson (j : Json) : Verdict where
  accepted := (j.getObjValAs? Bool "accepted").toOption == some true
  why := (j.getObjValAs? String "why").toOption.getD ""
  part := (j.getObjValAs? Nat "rec_part").toOption
  rows := (j.getObjValAs? String "rec_rows_sha512").toOption
  shared := (j.getObjVal? "rec_shared").toOption.bind Shared.ofJson?
  zk := (j.getObjValAs? Bool "zk").toOption
  coinKey := (j.getObjValAs? String "coin_key").toOption
  identifier := (j.getObjVal? "identifier").toOption.isSome

/-- **The shared inputs in the compiled verifier's form**: every part of `RecDeploy.V` reads the verifier's own inner
shape, here the composition's (`inner`, its SHA-512), the inner statement's three files, the inner session's record,
the coin server's transcript and how the inner statement is proved, and no record of check 5's coins. -/
def Shared.Complete (inner : String) (s : Shared) : Prop :=
  s.recInner = some inner ∧ s.innerCircuit.isSome ∧ s.innerPublic.isSome ∧ s.innerPartition.isSome ∧
    s.innerCoins = none ∧ s.innerSession.isSome ∧ s.transcript.isSome ∧ s.innerStatement.isSome ∧
    s.innerStatementCoins.isSome ∧ s.innerTables.isSome

/-- **`rec-compose`'s acceptance** over the parts' verdicts, at V*'s `n` statements, the composition's own rows file's
SHA-512 `rows` and inner shape's `inner`: one verdict per statement, `rec_part` `j` once for each `j < n`; each accepted
at one of them,
`--zk`, with no identifier, at `rows`; one `Shared` in the compiled verifier's form at `inner` that every part names; and
every session's coin key named, no two alike. -/
def Accepts (n : Nat) (rows inner : String) (vs : List Verdict) : Prop :=
  0 < n ∧ vs.length = n ∧ (∀ j, j < n → (vs.map (·.part)).count (some j) = 1) ∧
    (∀ v ∈ vs, v.accepted = true ∧ v.zk = some true ∧ v.identifier = false ∧ v.rows = some rows ∧
      ∃ j, v.part = some j ∧ j < n) ∧
    (∃ sh, sh.Complete inner ∧ ∀ v ∈ vs, v.shared = some sh) ∧
    (∀ v ∈ vs, v.coinKey.isSome) ∧ (vs.map (·.coinKey)).Nodup

/-- A log's `VERDICT` lines, each a JSON object, in order. -/
def verdicts (log : String) : Except String (Array Json) :=
  (log.splitOn "\n").foldlM (init := #[]) fun out line =>
    if line.startsWith "VERDICT " then
      match Json.parse (line.drop 8).toString with
      | .ok j => pure (out.push j)
      | .error e => throw s!"a VERDICT line that is not JSON: {e}"
    else pure out

/-- **One part's log**: it has exactly one `VERDICT` (a V* part is one session). Its verdict. -/
def partOf (name log : String) : Except String (String × Verdict) := do
  let vs ← (verdicts log).mapError (s!"{name}: " ++ ·)
  match vs.toList with
  | [v] => pure (name, Verdict.ofJson v)
  | [] => throw s!"{name}: no VERDICT"
  | _ => throw s!"{name}: {vs.size} VERDICTs: a V* part is one session (RecDeploy.V runs one per part)"

/-- **One part's verdict on its own**: accepted, at one of V*'s `n` statements, `--zk`, without an identifier, and at
the composition's rows file of SHA-512 `rows`. -/
def checkOne (n : Nat) (rows : String) (name : String) (v : Verdict) : Except String Unit :=
  if v.accepted = false then .error s!"{name}: a session is not accepted ({v.why})"
  else match v.part with
  | none => .error s!"{name}: its VERDICT names no rec_part: not one of V*'s parts (--rec-part J)"
  | some j =>
    if n ≤ j then .error s!"rec_part {j}: V* has {n} statements"
    else if v.zk ≠ some true then
      .error s!"{name}: not a --zk session: V*'s parts are masked (RecDeploy.insO), the only form the recursion's proof reads"
    else if v.identifier = true then
      .error s!"{name}: its VERDICT names an identifier (--identifier): V*'s parts set up without one (RecIns.setup), the only form the recursion's proof reads"
    else match v.rows with
    | none => .error s!"{name}: its VERDICT names no rec_rows_sha512: it was not checked against the shared rows (--rec-rows)"
    | some h =>
      if h = rows then .ok () else .error s!"{name}: checked against the rows file {h}, not the one staged rows file {rows}"

/-- The refusal of a part whose verify held no `what`. -/
def held (name what : String) : String :=
  s!"{name}: its verify held no {what}, which every part of the compiled verifier reads"

/-- **The first part's shared inputs are the compiled verifier's** at the composition's inner shape of SHA-512 `inner`
(`Shared.Complete`), each refusal naming its input. -/
def checkForm (name inner : String) (s : Shared) : Except String Unit :=
  match s.recInner with
  | none => .error s!"{name}: its VERDICT names no rec_inner: it was not verified at V*'s inner shape (--rec-inner)"
  | some h =>
    if h ≠ inner then .error s!"{name}: verified at the inner shape {h}, not the composition's own --rec-inner {inner}"
    else if s.innerCircuit.isNone then .error (held name "inner statement's circuit (--rec-inner-circuit)")
    else if s.innerPublic.isNone then .error (held name "inner statement's public file (--rec-inner-public)")
    else if s.innerPartition.isNone then .error (held name "inner statement's partition (--rec-inner-partition)")
    else if s.innerCoins.isSome then
      .error s!"{name}: it read a record of check 5's coins (--rec-inner-coins), which no part of the compiled verifier reads"
    else if s.innerSession.isNone then .error (held name "inner session's record (--rec-inner-session)")
    else if s.transcript.isNone then .error (held name "coin server's transcript (--rec-inner-transcript)")
    else if s.innerStatement.isNone then .error (held name "word on how the inner statement is proved (--rec-inner-statement)")
    else if s.innerStatementCoins.isNone then .error (held name "inner statement's coins (--rec-inner-statement)")
    else if s.innerTables.isNone then .error (held name "inner statement's lookup tables (--rec-inner-statement)")
    else .ok ()

/-- **Every part names the one shared set** `sh`. -/
def checkShared (sh : Shared) (name : String) (v : Verdict) : Except String Unit :=
  match v.shared with
  | none => .error s!"{name}: its VERDICT names no rec_shared: the inputs it shares with the other parts are not named"
  | some s => if s = sh then .ok () else .error s!"{name}: {s.diff sh}"

/-- **No two sessions share a coin key**, each named. -/
def checkKeys : List (String × Verdict) → Except String Unit
  | [] => .ok ()
  | (name, v) :: rest =>
    match v.coinKey with
    | none => .error s!"{name}: its VERDICT names no coin_key: its session's coins are not named"
    | some k =>
      if rest.any (·.2.coinKey == some k) then
        .error s!"{name}: its session's coin key is another part's: a session key is never reused (CoinTree.checkFresh)"
      else checkKeys rest

/-- `rec_part j` has one part. -/
def checkIndex (vs : List (String × Verdict)) (j : Nat) : Except String Unit :=
  if (vs.map (·.2.part)).count (some j) = 1 then .ok ()
  else .error s!"rec_part {j}: {(vs.map (·.2.part)).count (some j)} logs, not one"

/-- **The composition on the parts' verdicts** (name, verdict): `checkOne` on each, the first part's shared inputs in
the compiled verifier's form at `inner` (`checkForm`) and every part's the same (`checkShared`), one part per statement
(`checkIndex`, and as many parts as statements) and fresh coin keys (`checkKeys`). -/
def checkVerdicts (n : Nat) (rows inner : String) (vs : List (String × Verdict)) : Except String Unit :=
  if n = 0 then .error "no V* statement" else
  (vs.forM fun p => checkOne n rows p.1 p.2) >>= fun _ =>
  match vs with
  | [] => .error s!"0 parts, not V*'s {n} statements"
  | (name, v) :: _ =>
    match v.shared with
    | none => .error s!"{name}: its VERDICT names no rec_shared: the inputs it shares with the other parts are not named"
    | some sh =>
      checkForm name inner sh >>= fun _ =>
      (vs.forM fun p => checkShared sh p.1 p.2) >>= fun _ =>
      ((List.range n).forM (checkIndex vs)) >>= fun _ =>
      (if vs.length = n then .ok () else .error s!"{vs.length} parts, not V*'s {n} statements") >>= fun _ =>
      checkKeys vs

/-- **The composition**: `parts`' logs (name, text), one `VERDICT` each (`partOf`), composed at V*'s `n` statements,
the composition's own rows file's SHA-512 `rows` and inner shape's `inner` (`checkVerdicts`). -/
def check (n : Nat) (rows inner : String) (parts : List (String × String)) : Except String Unit :=
  (parts.mapM fun p => partOf p.1 p.2) >>= checkVerdicts n rows inner

/-! ## `check` accepts only what `Accepts` states -/

theorem bind_ok {α β : Type} {x : Except String α} {f : α → Except String β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem forM_ok {α : Type} {f : α → Except String Unit} :
    ∀ {l : List α} {u : Unit}, l.forM f = .ok u → ∀ a ∈ l, f a = .ok ()
  | [], _, _, a, ha => by cases ha
  | b :: l, _, h, a, ha => by
    dsimp only [List.forM] at h
    obtain ⟨⟨⟩, hb, h⟩ := bind_ok h
    cases ha with
    | head => exact hb
    | tail _ hm => exact forM_ok h a hm

theorem checkOne_ok {n : Nat} {rows name : String} {v : Verdict} (h : checkOne n rows name v = .ok ()) :
    v.accepted = true ∧ v.zk = some true ∧ v.identifier = false ∧ v.rows = some rows ∧
      ∃ j, v.part = some j ∧ j < n := by
  unfold checkOne at h
  split at h
  · cases h
  rename_i ha
  split at h
  · cases h
  rename_i j hj
  split at h
  · cases h
  rename_i hn
  split at h
  · cases h
  rename_i hz
  split at h
  · cases h
  rename_i hi
  split at h
  · cases h
  rename_i hr
  split at h
  · rename_i hh
    exact ⟨by simpa using ha, by simpa using hz, by simpa using hi, by rw [hr, hh], j, hj, by omega⟩
  · cases h

theorem checkForm_ok {name inner : String} {s : Shared} (h : checkForm name inner s = .ok ()) : s.Complete inner := by
  unfold checkForm at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i hr hi h1 h2 h3 h4 h5 h6 h7 h8 h9
  refine ⟨by rw [hr]; simpa using hi, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩ <;>
    simp_all [Option.isNone_iff_eq_none, Option.isSome_iff_ne_none]

theorem checkShared_ok {sh : Shared} {name : String} {v : Verdict} (h : checkShared sh name v = .ok ()) :
    v.shared = some sh := by
  unfold checkShared at h
  repeat' split at h
  all_goals first | (cases h; done) | skip
  rename_i hs e
  rw [hs, e]

theorem checkKeys_ok : ∀ {vs : List (String × Verdict)}, checkKeys vs = .ok () →
    (∀ p ∈ vs, p.2.coinKey.isSome) ∧ (vs.map (·.2.coinKey)).Nodup
  | [], _ => ⟨by simp, List.nodup_nil⟩
  | (name, v) :: rest, h => by
    unfold checkKeys at h
    split at h
    · cases h
    rename_i k hk
    split at h
    · cases h
    rename_i hany
    obtain ⟨h1, h2⟩ := checkKeys_ok h
    refine ⟨?_, ?_⟩
    · intro p hp
      cases hp with
      | head => simp [hk]
      | tail _ hm => exact h1 p hm
    · rw [List.map_cons, List.nodup_cons]
      refine ⟨?_, h2⟩
      intro hm
      obtain ⟨q, hq, he⟩ := List.mem_map.1 hm
      apply hany
      rw [List.any_eq_true]
      exact ⟨q, hq, by simp [he, hk]⟩

theorem checkIndex_ok {vs : List (String × Verdict)} {j : Nat} (h : checkIndex vs j = .ok ()) :
    (vs.map (·.2.part)).count (some j) = 1 := by
  unfold checkIndex at h
  split at h
  · assumption
  · cases h

/-- **`checkVerdicts` accepts only what `Accepts` states.** -/
theorem checkVerdicts_accepts {n : Nat} {rows inner : String} {vs : List (String × Verdict)}
    (h : checkVerdicts n rows inner vs = .ok ()) : Accepts n rows inner (vs.map (·.2)) := by
  unfold checkVerdicts at h
  split at h
  · cases h
  rename_i hn
  obtain ⟨_, hone, h⟩ := bind_ok h
  obtain _ | ⟨⟨name, v⟩, rest⟩ := vs
  · cases h
  dsimp only at h
  split at h
  · cases h
  rename_i sh hsh
  obtain ⟨_, hform, h⟩ := bind_ok h
  obtain ⟨_, hshared, h⟩ := bind_ok h
  obtain ⟨_, hidx, h⟩ := bind_ok h
  obtain ⟨_, hlen, h⟩ := bind_ok h
  have hl : ((name, v) :: rest).length = n := by
    split at hlen
    · assumption
    · cases hlen
  obtain ⟨hk1, hk2⟩ := checkKeys_ok h
  refine ⟨Nat.pos_of_ne_zero hn, by simpa using hl, ?_, ?_, ⟨sh, checkForm_ok hform, ?_⟩, ?_, ?_⟩
  · intro j hj
    rw [List.map_map]
    exact checkIndex_ok (forM_ok hidx j (List.mem_range.2 hj))
  · intro w hw
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact checkOne_ok (forM_ok hone p hp)
  · intro w hw
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact checkShared_ok (forM_ok hshared p hp)
  · intro w hw
    obtain ⟨p, hp, rfl⟩ := List.mem_map.1 hw
    exact hk1 p hp
  · rw [List.map_map]
    exact hk2

/-- **`check` accepts only what `Accepts` states**: where it accepts the logs, each holds one `VERDICT`, and their
verdicts meet `Accepts`. -/
theorem check_accepts {n : Nat} {rows inner : String} {parts : List (String × String)}
    (h : check n rows inner parts = .ok ()) :
    ∃ vs, parts.mapM (fun p => partOf p.1 p.2) = .ok vs ∧ Accepts n rows inner (vs.map (·.2)) := by
  obtain ⟨vs, hvs, h⟩ := bind_ok h
  exact ⟨vs, hvs, checkVerdicts_accepts h⟩

end RecCompose
end Flock
