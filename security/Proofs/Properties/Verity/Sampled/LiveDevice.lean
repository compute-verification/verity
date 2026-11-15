module

public import Proofs.Properties.Verity.Sampled.Instance
public import Proofs.Properties.Challenger.Entry
public import Proofs.Properties.Challenger.Lines
public import Proofs.Flock.Claims.Complete.Json.RoundTrip
public import Lms.Sign
public import Base.JsonParse

/-!
# The live challenger as deployed: a deployment that holds a key, flips its coins and keeps the ledger

`Device.Deployment` is what fixes the device's history at a play (`Device.dev`). `Instance.Honest.deployment` is the
idle one: no key, no coins, no ledger. This file builds the one the auditor runs (`deployment`), the SPOT live
challenger on its auditor device, each field what the host's code does:

* **The host** (`Host`): what the auditor's machine holds when an audit starts. Its LMS key over SHA-512 at the
  deployed parameters (`Lms.deployed`, n = 64; `serve.py`'s and `auditor.py`'s `load_key` in
  `verity/composite/device/live_challenger/`), the key's next unused leaf (`lms.LeafState`), its name (each session's
  record signs `<name>.<session>`), the commit its tree is pinned at, the session id the gate reserves for the audit
  (`gate_reserve`'s `s<UTC>-<token>`), the ledger's next `seq` and its last line's SHA-512, the host's clock at each entry,
  and the body fields of its own each entry adds that the device's model doesn't compute (the pins log's seal, the coin
  link's port, the session secret's SHA-512, the session's other files' digests, serve.py's exit code). All of it is fixed
  before the play: none of it is the prover's.
* **The coins** (`families`, `draws`, `run`): the coin commitment's one family is the challenger family v2's parameters
  name, with `MAX_COINS` coins (`Instance.coins`'s count). At the close the device samples, for every position of the layout,
  the coin's `COIN_BYTES` bytes and its hm96 salt's `SALT_BYTES` bytes, as `LiveChallenger.__init__`
  (`verity/protocols/spot/device/challenger.py`) samples `(coin, salt)` per position; the play's released coins override
  the sampled ones at their positions (`Device.coinAt`). The run's scheme and epoch are the reserve request's, its host
  `<name>.<session>`, and how it ended and its lines' times are what the prover's close leaves to it (`closeJson`).
* **The ledger** (`opens`, `closes`): the auditor's signed entries for the period's one window, each `append`'s line
  (`auditor.py`): `period-open`, `reserve` and `open` before any coin, `end` and `period-close` at the close, chained by
  `prev` from the host's ledger head and numbered from its `seq`, each signed at its leaf. Their bodies carry the
  prover's request where the gate copies it (the reserve's request, the window's root) and the device's own fields
  where the gate computes them (the layout, `pinned`, `key_sha512`, `host`, the times, the record's leaf, the run's
  commitment, the counts released, whether the client closed the session).
* **The signatures** (`sign`): LMS `signAt` (`Lms.Sign`, `lms.sign_at`) under the host's key, the `i`-th message of the play
  at the host's leaf plus `i`: the ledger's three opening entries, then the run's record and close (`serve.py`'s `sign`),
  then the ledger's two closing entries, the order the host signs them in.
* **The deployment** (`deployment`): pinned at the host's key (`Host.pin`), on the windows the auditor's pins designate
  for the challenger. The host's key has no earlier history in this model, so no earlier line opens a designated
  window and no earlier run shares the play's id (`earlier_closed`, `fresh_id`). Its ledger holds one line per `seq`
  (`numbered`), and an open names the play's session (`opens_run`), where each line it prints reads back as the record
  it printed (`ReadsBack`, the open premise `Properties.Challenger.LedgerReadsBack` at each of the play's records;
  `readsBack_of_plain` where their JSON is plain).
* **At it** (`signer`, `dev`, `cost`, `base`): the deployed device's signer at `families`, its history at each play, the
  declared calls covering its reading (`costs`), and a vocabulary at it, pinned at the host's key alone, whose fields
  `Instance` ties to the deployment (`base_signer`, `coins_families`, `costs`).
* **The audit's own parameters** (`Holds`): v2's parameters in `θ` name a challenger whose key is the host's. There the
  parameters' key is the host's, the pinned one the history sits under (`pubOf_holds`). Whatever challenger they name, the layout gives it `MAX_COINS` coins
  (`families_params`), the family clause of the session v2 sets up (`SetsUp.params`).
-/

@[expose] public section

namespace Verity.Sampled.LiveDevice

open scoped ENNReal
open Lean (Json)
open Spot Spot.Challenger
open Verity.Sampled.Device (RunSetup Deployment)
open Properties.Challenger (History OpensOn LedgerReadsBack)

/-! ## The host -/

/-- **The auditor device the live challenger runs on**, as it stands when an audit starts at the parameters `θ`: its LMS
key at the deployed parameters (`deployed`), its name and pinned commit, and at each `θ` the session id the gate
reserves, the key's next unused leaf, the ledger's next `seq` and its last line's SHA-512 (`prev`, `ZERO` on an empty
ledger), its clock's UTC reading at the audit's `k`-th entry, the body fields of its own the entry of each kind
adds (`extra`), and the files the audit registers: at each `θ` the circuit and public files, and the pinned table
files (`circuitFile`, `publicFile`, `tables`). -/
structure Host where
  key : Lms.PrivateKey
  deployed : key.params = Lms.deployed
  name : String
  pinned : String
  session : ByteArray → String
  leaf : ByteArray → ℕ
  seq : ByteArray → ℕ
  prev : ByteArray → String
  clock : ByteArray → ℕ → String
  extra : ByteArray → String → List (String × Json)
  circuitFile : ByteArray → ByteArray
  publicFile : ByteArray → ByteArray
  tables : Array ByteArray

namespace Host

variable (H : Host)

/-- The host's pinned table lookup, derived from the table bytes it registers. -/
def tableFiles : PinnedTables := .ofFiles H.tables

/-- The host's public key, `I ‖ T[1]` (`lms.public_key`). -/
def pub : ByteArray := H.key.publicKey

/-- `key_sha512`: the SHA-512 of the host's public key, as each opening entry names the key (`key_signer`). -/
def keySha512 : String := sha512Hex H.pub

/-- **The host's pin**: the challenger's, its public key under the deployed LMS parameters. -/
def pin : AuditorPin :=
  { role := .challenger, key := H.pub.data.toList, scheme := (Lms.deployed.N, Lms.deployed.W, Lms.deployed.H) }

/-- The host a session's record signs, `<name>.<session>` (`reserve`'s `host`). -/
def hostAt (θ : ByteArray) : String := H.name ++ "." ++ H.session θ

/-- **The host's LMS signature at leaf `q`** (`Lms.signAt`, encoded), the empty string where the leaf is past the
key's. -/
def signAt (q : ℕ) (m : ByteArray) : ByteArray :=
  match Lms.signAt H.key q m with
  | .ok s => s.encode
  | .error _ => .empty

end Host

/-! ## The coins -/

/-- **The coin commitment's families at the parameters `θ`**: the challenger family v2's parameters name, with
`MAX_COINS` coins; none where they name no challenger. -/
def families (θ : ByteArray) : Families :=
  match paramsOf θ with
  | some p =>
    match p.base.challenger with
    | some (_, l) => [(l, MAX_COINS)]
    | none => []
  | none => []

/-- **Each family samples `MAX_COINS` coins** (`Instance.coins`). -/
theorem coins_families {θ : ByteArray} {l : String} {m : ℕ} (h : (l, m) ∈ families θ) : m = MAX_COINS := by
  unfold families at h
  split at h
  · split at h
    · simp only [List.mem_cons, Prod.mk.injEq, List.not_mem_nil, or_false] at h
      exact h.2
    · simp at h
  · simp at h

/-- The bits the device samples for one position of its layout: the coin's `COIN_BYTES` bytes and its hm96 salt's. -/
def SLOT_BITS : ℕ := 8 * (COIN_BYTES + _root_.Spot.Challenger.SALT_BYTES)

/-- **The bits the device samples at the close**: every position's coin and salt (`LiveChallenger.__init__`). -/
def draws (θ : ByteArray) : ℕ := _root_.Spot.Challenger.total (families θ) * SLOT_BITS

/-- The bytes of the `n` bits of `ρ` from bit `off` (`Lines.coinBytes`). -/
def bytesAt (ρ : List Bool) (off n : ℕ) : ByteArray := Lines.coinBytes ((ρ.drop off).take n)

/-- Bytes read as JSON, `null` where they aren't UTF-8 text that parses. -/
def jsonOf (b : ByteArray) : Json :=
  match String.fromUTF8? b with
  | some s => (Flock.JsonParse.parse s).toOption.getD .null
  | none => .null

/-- **The prover's request**, as the gate takes it: `reserve` (its `--peer`, `--scheme`, `--families`, `--epoch`,
`--minutes`, `--for`, `--withhold` and its `by`), `open` (the window's root and its opener) and `period` (`period-open`'s
fields), each an object of the request's JSON. -/
def requestPart (a : ByteArray) (k : String) : Json := getD (jsonOf a) k

/-- **What the prover's close leaves to the device**: `ended` (`null` when the client closed the session, else why
not) and `times` (each release log line's time), the parts of the run the deployed device's clock and its client's
frames decide. -/
def closeJson (b : ByteArray) : Json := jsonOf b

/-- An object's fields, none for any other value. -/
def fieldsOf (j : Json) : List (String × Json) :=
  match j with
  | .obj kvs => kvs.toArray.toList
  | _ => []

/-- A string's text, the empty string for any other value. -/
def strOf (j : Json) : String :=
  match j with
  | .str s => s
  | _ => ""

/-- **The run's setup at a play** but for its key, layout and ledger, which `Deployment.setupAt` sets: the session id
the gate reserved and the host the record signs, the reserve's scheme and epoch, each position's coin and salt sampled at
the close, how it ended and each line's time. -/
def run (H : Host) (θ a : ByteArray) (_ : List Release) (b : ByteArray) (ρ : List Bool) : RunSetup where
  key := H.pub
  id := H.session θ
  host := H.hostAt θ
  scheme := strOf (getD (requestPart a "reserve") "scheme")
  epoch := (nat? (getD (requestPart a "reserve") "epoch")).getD 0
  families := families θ
  coins p := bytesAt ρ (p * SLOT_BITS) (8 * COIN_BYTES)
  salts p := bytesAt ρ (p * SLOT_BITS + 8 * COIN_BYTES) (8 * _root_.Spot.Challenger.SALT_BYTES)
  ended := match getD (closeJson b) "ended" with
    | .str s => some s
    | _ => none
  time k := match getD (closeJson b) "times" with
    | .arr ts => (ts[k]?.bind nat?).getD 0
    | _ => 0
  ledger := []

/-- The run's setup at a play, under the host's key, the pinned one (`Deployment.setupAt`), and at `families`, before
its ledger. -/
def setup (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : RunSetup :=
  { run H θ a log b ρ with families := families θ }

/-- The run's coin commitment (`Properties.Challenger.Run.commitment`), its layout's and its coins' and salts'. -/
def commitmentAt (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    Option LiveCommitment :=
  (Device.runOf (setup H θ a log b ρ) log []).commitment

/-- The run's release log, its head and the lines the device printed (`Device.printLog`); none where it printed none. -/
def releaseLog (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    Option (List String) :=
  (Device.printLog (setup H θ a log b ρ).time Device.PrintState.init log).map
    (Device.headLine (commitmentAt H θ a log b ρ) :: ·)

/-! ## The ledger -/

/-- **An entry's record, as `append` writes it**: entry `seq θ + j` of `kind` for `session`, after the line whose
SHA-512 is `prev`, its `sig` the host's signature at leaf `q` over `ledger_message` of the entry without it. -/
def entryRecord (H : Host) (θ : ByteArray) (j q : ℕ) (prev kind : String) (session : Option String) (body : Json) :
    _root_.Core.Record :=
  let unsigned := Json.mkObj [("body", body), ("kind", .str kind), ("prev", .str prev), ("schema", .str LEDGER_SCHEMA),
    ("seq", Lean.toJson (H.seq θ + j)), ("session", (session.map Json.str).getD .null)]
  let m := (ledgerMessage (Properties.Challenger.ident H.pub) unsigned).toOption.getD .empty
  ⟨body, kind, prev, LEDGER_SCHEMA, H.seq θ + j, session, H.signAt q m⟩

/-- A record's ledger line, its canonical text (`ledger_text`). -/
def recordLine (r : _root_.Core.Record) : Json := .str ((PyJson.canonical r.toJson).toOption.getD "")

/-- **An entry's line, as `append` writes it**: its record's, in the ledger's canonical text. -/
def entryLine (H : Host) (θ : ByteArray) (j q : ℕ) (prev kind : String) (session : Option String) (body : Json) :
    Json :=
  recordLine (entryRecord H θ j q prev kind session body)

/-- **Consecutive entries' records**: from entry `seq θ + j` at leaf `q`, after the line whose SHA-512 is `prev`, each
line the next one's `prev` (`Core.Record.prevOf`). -/
def records (H : Host) (θ : ByteArray) :
    ℕ → ℕ → String → List (String × Option String × Json) → List _root_.Core.Record
  | _, _, _, [] => []
  | j, q, prev, (kind, s, body) :: es =>
    let r := entryRecord H θ j q prev kind s body
    r :: records H θ (j + 1) (q + 1) (_root_.Core.Record.prevOf (strOf (recordLine r))) es

theorem records_length (H : Host) (θ : ByteArray) :
    ∀ (j q : ℕ) (prev : String) (es : List (String × Option String × Json)), (records H θ j q prev es).length = es.length
  | _, _, _, [] => rfl
  | j, q, prev, (_, _, _) :: es => by
    simp only [records, List.length_cons]
    rw [records_length H θ]

/-- A body: the host's own fields of the entry's kind, then the request's `keep` fields of `part`, then the device's,
each later field in place of an earlier one of its name. -/
def body (H : Host) (θ : ByteArray) (kind : String) (part : Json) (keep : List String) (own : List (String × Json)) :
    Json :=
  Json.mkObj (H.extra θ kind ++ (fieldsOf part).filter (keep.contains ·.1) ++ own)

/-- The request fields a reserve records and its open carries (`RESERVED`), and whether its session is withheld. -/
def RESERVE_FIELDS : List String := ["by", "for", "peer", "scheme", "epoch", "minutes", "withheld"]

/-- **The period's open** (`period`'s `period-open`): the request's period, the pinned commit, the key and the time. -/
def periodOpenBody (H : Host) (θ a : ByteArray) : Json :=
  body H θ "period-open" (requestPart a "period")
    ["period", "lane", "by", "cap", "circuit", "draw_k", "rows", "weights_root", "model", "circuit_files", "length",
      "policy", "alpha", "gamma", "registration", "policy_name"]
    [("pinned", .str H.pinned), ("key_sha512", .str H.keySha512), ("opened_utc", .str (H.clock θ 0))]

/-- **The window's reserve** (`reserve`): the request, the layout the session commits to, the pinned commit, the key, the
record's host, and when it starts and expires (`later`). -/
def reserveBody (H : Host) (θ a : ByteArray) : Json :=
  body H θ "reserve" (requestPart a "reserve") RESERVE_FIELDS
    [("families", familiesJson (families θ)), ("pinned", .str H.pinned), ("key_sha512", .str H.keySha512),
      ("host", .str (H.hostAt θ)), ("started_utc", .str (H.clock θ 1)),
      ("expires_utc", .str ((later (H.clock θ 1) ((nat? (getD (requestPart a "reserve") "minutes")).getD 0)).getD ""))]

/-- **The window's open** (`admit`): the window's root the opener named, its reserve's `seq`, the reserve's request and
layout, its opener, the pinned commit, the key, the record's host and the time. -/
def openBody (H : Host) (θ a : ByteArray) : Json :=
  body H θ "open" (requestPart a "reserve") RESERVE_FIELDS
    [("window", Json.mkObj [("root", getD (requestPart a "open") "root")]), ("reserve", Lean.toJson (H.seq θ + 1)),
      ("families", familiesJson (families θ)), ("opened_by", getD (requestPart a "open") "by"),
      ("pinned", .str H.pinned), ("key_sha512", .str H.keySha512), ("host", .str (H.hostAt θ)),
      ("started_utc", .str (H.clock θ 2))]

/-- **The ledger's entries before any coin**: the period's open, the window's reserve and its open, entries `seq θ` to
`seq θ + 2` at the host's leaves `leaf θ` to `leaf θ + 2`, after the host's ledger head. -/
def openRecords (H : Host) (θ a : ByteArray) : List _root_.Core.Record :=
  records H θ 0 (H.leaf θ) (H.prev θ)
    [("period-open", none, periodOpenBody H θ a), ("reserve", some (H.session θ), reserveBody H θ a),
      ("open", some (H.session θ), openBody H θ a)]

/-- Their lines. -/
def opens (H : Host) (θ a : ByteArray) : List Json := (openRecords H θ a).map recordLine

/-- Whether the client closed the session: its close left no reason it ended otherwise. -/
def closed (b : ByteArray) : Bool :=
  match getD (closeJson b) "ended" with
  | .str _ => false
  | _ => true

/-- How many coins of each family the run released (`released_of`), and whether every family released all of its
(`complete`). -/
def releasedJson (θ : ByteArray) (log : List Release) : Json :=
  Json.mkObj ((families θ).map fun (l, _) => (l, Lean.toJson (Device.coinsOf l log).length))

def complete (θ : ByteArray) (log : List Release) : Bool :=
  (families θ).all fun (l, n) => (Device.coinsOf l log).length == n

/-- **The window's end** (`run`'s `end`): whether the client closed the session and how it ended, the release log's
SHA-512 among the session's files, the coins released and whether all were, the leaf the record was signed at (the
fourth of the play's), its commitment's root and the time. -/
def endBody (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : Json :=
  let files := Json.mkObj ((fieldsOf (getD (Json.mkObj (H.extra θ "end")) "files")) ++
    ((releaseLog H θ a log b ρ).map fun ls => [(LOG_FILE, Json.str (logDigest ls))]).getD [])
  body H θ "end" .null []
    [("closed", .bool (closed b)),
      ("ended", match getD (closeJson b) "ended" with
        | .str s => .str s
        | _ => .str "closed by the client"),
      ("files", files), ("released", releasedJson θ log), ("complete", .bool (complete θ log)),
      ("record_leaf", Lean.toJson (H.leaf θ + 3)),
      ("commitment", ((commitmentAt H θ a log b ρ).map fun c => Json.str (Flock.hexOf c.root)).getD .null),
      ("ended_utc", .str (H.clock θ 3))]

/-- **The period's close** (`period`'s `period-close`): its one window, `{root, session, complete}`, and the time. -/
def periodCloseBody (H : Host) (θ a : ByteArray) (log : List Release) : Json :=
  let window := Json.mkObj [("root", getD (requestPart a "open") "root"), ("session", .str (H.session θ)),
    ("complete", .bool (complete θ log))]
  body H θ "period-close" (requestPart a "period") ["period", "lane", "by"]
    [("windows", .arr #[window]), ("complete", .bool (complete θ log)), ("closed_utc", .str (H.clock θ 4))]

/-- **The ledger's entries at the close**: the window's end and the period's close, entries `seq θ + 3` and
`seq θ + 4`, after the window's open, at the leaves after the run's record and close (`leaf θ + 5`, `leaf θ + 6`). -/
def closeRecords (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    List _root_.Core.Record :=
  records H θ 3 (H.leaf θ + 5) (_root_.Core.Record.prevOf (strOf ((opens H θ a).getLastD .null)))
    [("end", some (H.session θ), endBody H θ a log b ρ), ("period-close", none, periodCloseBody H θ a log)]

/-- Their lines. -/
def closes (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : List Json :=
  (closeRecords H θ a log b ρ).map recordLine

theorem opens_length (H : Host) (θ a : ByteArray) : (opens H θ a).length = 3 := by
  rw [opens, List.length_map, openRecords, records_length]
  rfl

theorem closes_length (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    (closes H θ a log b ρ).length = 2 := by
  rw [closes, List.length_map, closeRecords, records_length]
  rfl

/-- The play's records' numbers: `seq θ` to `seq θ + 4`, in order. -/
theorem records_seq (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    (openRecords H θ a ++ closeRecords H θ a log b ρ).map (·.seq) =
      [H.seq θ + 0, H.seq θ + 1, H.seq θ + 2, H.seq θ + 3, H.seq θ + 4] := by
  simp [openRecords, closeRecords, records, entryRecord]

/-- The play's records' sessions: the gate's session for the reserve, the open and the end, none for the period's. -/
theorem records_session (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    (openRecords H θ a ++ closeRecords H θ a log b ρ).map (·.session) =
      [none, some (H.session θ), some (H.session θ), some (H.session θ), none] := by
  simp [openRecords, closeRecords, records, entryRecord]

/-! ## The read-back -/

/-- **Each ledger line the host prints at a play reads back as its record** (`LedgerReadsBack`): the three it prints
before any coin and the two at the close. -/
def ReadsBack (H : Host) : Prop :=
  ∀ θ a log b ρ, ∀ r ∈ openRecords H θ a ++ closeRecords H θ a log b ρ, LedgerReadsBack r

-- `mvcgen` splits `ledgerEntry?` into one goal per field it parses
set_option maxHeartbeats 1000000 in
/-- `ledgerEntry?` read its entry from the line's text, which parses to it. -/
theorem ledgerEntry_parse {line : Json} {e : LedgerEntry} (h : ledgerEntry? line = some e) :
    line = .str e.text ∧ (Flock.JsonParse.parse e.text).toOption = some e.entry := by
  refine Properties.Pouw.LiveSalt.Ledger.of_triple_opt
    (Q := fun e => line = .str e.text ∧ (Flock.JsonParse.parse e.text).toOption = some e.entry) ?_ h
  clear h
  mvcgen [ledgerEntry?, guard, -Std.Do.Spec.throw_Option, Properties.Pouw.LiveSalt.Ledger.none_spec,
    Properties.Pouw.LiveSalt.Ledger.failure_spec, Properties.Pouw.LiveSalt.Ledger.toOption_spec,
    Properties.Pouw.LiveSalt.Ledger.ledgerTag_spec, Properties.Pouw.LiveSalt.Ledger.nat_spec,
    Properties.Pouw.LiveSalt.Ledger.some_pure_spec]
  all_goals first | exact ⟨rfl, by assumption⟩ | (constructor <;> first | rfl | assumption)

/-- **A record whose JSON is plain reads back** (`LedgerReadsBack`): its canonical text is `Flock.canon`'s
(`Plain.canonical_canon`), which parses to a value alike it (`parse_canon`), whose `seq`, kind and session are its. -/
theorem ledgerReadsBack_of_plain {r : _root_.Core.Record}
    (hp : Properties.Challenger.Plain.PlainAt PyJson.DEPTH r.toJson) : LedgerReadsBack r := by
  intro e he
  obtain ⟨t, ht⟩ := FlockComplete.JsonRT.canon_ok_some r.toJson (Properties.Challenger.Entry.good_of_plain hp)
  have hline : PyJson.canonical r.toJson = .ok t := (Properties.Challenger.Plain.canonical_canon hp).trans ht
  rw [hline] at he
  obtain ⟨j', hj', hsame⟩ :=
    FlockComplete.JsonRT.parse_canon r.toJson t (Properties.Challenger.Entry.good_of_plain hp) ht
  obtain ⟨htext, hparse⟩ := ledgerEntry_parse he
  simp only [Except.toOption, Option.getD_some, Json.str.injEq] at htext
  have hent : e.entry = j' := by
    rw [← htext, hj'] at hparse
    simp only [Except.toOption, Option.some.injEq] at hparse
    exact hparse.symm
  subst hent
  obtain ⟨r', hr', hq⟩ := Properties.Challenger.Lines.ledgerEntry_seq he
  obtain ⟨n, hn, hrn⟩ := Properties.Challenger.Lines.recordOfJson_seq
    (Properties.Challenger.ChallengePhase.toOption_some hr')
  obtain ⟨-, -, hk, -, -, hopen, -⟩ := Properties.Challenger.Window.ledgerEntry_fields he
  obtain ⟨-, hkr, -, -, hsr, hses, -⟩ := Properties.Challenger.Entry.toJson_vals r
  have hsk := Properties.Challenger.Entry.same_getD hsame "kind"
  have hsq := Properties.Challenger.Entry.same_getD hsame "seq"
  have hss := Properties.Challenger.Entry.same_getD hsame "session"
  rw [hkr] at hsk
  rw [hsr] at hsq
  rw [hses] at hss
  have hkind : e.kind = r.kind := by
    have := FlockComplete.JsonRT.same_str hsk
    rw [hk] at this
    exact Json.str.inj this
  refine ⟨?_, hkind, fun ho => ?_⟩
  · have := FlockComplete.JsonRT.same_num hsq
    rw [hn] at this
    obtain rfl := Json.num.inj this
    rw [← hq, hrn]
    rfl
  · have hs := hopen ho
    revert hss
    cases r.session with
    | none =>
      intro hss
      have := FlockComplete.JsonRT.same_null hss
      rw [hs] at this
      cases this
    | some s =>
      intro hss
      have := FlockComplete.JsonRT.same_str hss
      rw [hs] at this
      rw [Json.str.inj this]

/-- **The host's lines read back where its records' JSON is plain** (`ledgerReadsBack_of_plain`). -/
theorem readsBack_of_plain (H : Host)
    (h : ∀ θ a log b ρ, ∀ r ∈ openRecords H θ a ++ closeRecords H θ a log b ρ,
      Properties.Challenger.Plain.PlainAt PyJson.DEPTH r.toJson) : ReadsBack H :=
  fun θ a log b ρ r hr => ledgerReadsBack_of_plain (h θ a log b ρ r hr)

/-! ## The deployment -/

/-- A filter-map whose every value is `g`'s at its element is a sublist of `g`'s map. -/
theorem filterMap_sublist {α β : Type} {f : α → Option β} {g : α → β} :
    ∀ {l : List α}, (∀ x ∈ l, ∀ y, f x = some y → y = g x) → (l.filterMap f).Sublist (l.map g)
  | [], _ => by simp
  | x :: l, h => by
    have ih := filterMap_sublist (l := l) fun y hy => h y (List.mem_cons_of_mem _ hy)
    cases hx : f x with
    | none =>
      simp only [List.filterMap_cons, hx, List.map_cons]
      exact ih.cons _
    | some y =>
      simp only [List.filterMap_cons, hx, List.map_cons]
      rw [h x (by simp) y hx]
      exact ih.cons₂ _

/-- **The host's ledger holds one line per `seq`** where its lines read back (`ReadsBack`): a line that reads as an
entry carries its record's `seq`, and the play's records are numbered `seq θ` to `seq θ + 4`. The host's key has no
earlier history in this model, so its ledger is the play's lines alone. -/
theorem numbered (H : Host) (hR : ReadsBack H) (θ a : ByteArray) (log : List Release) (b : ByteArray)
    (ρ : List Bool) : Device.Numbered (opens H θ a ++ closes H θ a log b ρ) := by
  have hl : opens H θ a ++ closes H θ a log b ρ =
      (openRecords H θ a ++ closeRecords H θ a log b ρ).map recordLine := by
    simp [opens, closes]
  unfold Device.Numbered
  rw [hl, List.filterMap_map]
  refine List.Nodup.sublist (filterMap_sublist (g := (·.seq)) fun r hr s hs => ?_) ?_
  · change (ledgerEntry? (recordLine r)).map (·.seq) = some s at hs
    obtain ⟨e, he, rfl⟩ := Option.map_eq_some_iff.1 hs
    exact (hR θ a log b ρ r hr e he).1
  · rw [records_seq]
    simp [List.nodup_cons]

/-- **An open the host prints names the play's session** where its lines read back (`ReadsBack`): an open line's
session is its record's, and the host's records name the gate's session or none. -/
theorem opens_run (H : Host) (hR : ReadsBack H) (θ a : ByteArray) (log : List Release) (b : ByteArray)
    (ρ : List Bool) (pk : ByteArray) (w id : String)
    (h : OpensOn ⟨fun _ => [], fun _ => opens H θ a ++ closes H θ a log b ρ⟩ pk w id) : id = H.session θ := by
  obtain ⟨line, hl, e, he, hk, -, hid, -, -⟩ := h
  have hl' : line ∈ (openRecords H θ a ++ closeRecords H θ a log b ρ).map recordLine := by
    simpa [opens, closes] using hl
  obtain ⟨r, hr, rfl⟩ := List.mem_map.1 hl'
  have hs := (hR θ a log b ρ r hr e he).2.2 hk
  have hses : r.session ∈ [none, some (H.session θ), some (H.session θ), some (H.session θ), none] := by
    rw [← records_session H θ a log b ρ]
    exact List.mem_map_of_mem hr
  rw [hs] at hses
  simp at hses
  rw [← hid, hses]

/-- **No earlier line of the host's key opens a designated window**: the host's key has no earlier history in this
model (`Challenger.earlier` at its default, no run and no ledger line), so this holds with nothing to close. -/
theorem earlier_closed (pk : ByteArray) (w id : String) : ¬ OpensOn ⟨fun _ => [], fun _ => []⟩ pk w id :=
  fun ⟨_, hl, _⟩ => absurd hl List.not_mem_nil

/-- **No earlier run of the host's key has the play's session's id**: the host's key has no earlier history in this
model, so there is no earlier run and the id is fresh with nothing to compare. -/
theorem fresh_id (pk : ByteArray) (id : String) :
    ∀ r ∈ (⟨fun _ => [], fun _ => []⟩ : History).runs pk, r.id ≠ id :=
  fun _ h => absurd h List.not_mem_nil

/-- **The live challenger as deployed** on the host `H`, pinned at its key (`Host.pin`), on the windows `W` the
auditor designated: the run of the session the gate reserved, its coins and salts sampled at the close, the ledger's
three entries before any coin and two at the close, and each of the play's messages signed with the host's key at its
leaf. Its lines read back (`hR`) give its numbering and its opens. -/
def deployment (H : Host) (W : List Hash.Msg) (hR : ReadsBack H) : Deployment where
  pin := H.pin
  sessions := W
  run := run H
  opens := opens H
  closes := closes H
  ledgerLines _ := 5
  ledger_le θ a log b ρ := by rw [opens_length, closes_length]
  draws := draws
  sign θ i m := (H.signAt (H.leaf θ + i) ⟨m.toArray⟩).data.toList
  numbered θ a log b ρ := by
    simpa using numbered H hR θ a log b ρ
  earlier_closed pk w id _ := earlier_closed pk w id
  opens_run θ a log b ρ w id _ h := opens_run H hR θ a log b ρ _ w id h
  fresh_id θ a log b ρ pk := fresh_id pk _

/-- The deployed device's signer at the parameters `θ`, at `families`. -/
def signer (H : Host) (W : List Hash.Msg) (hR : ReadsBack H) (θ : ByteArray) : Signer :=
  (deployment H W hR).signer families θ

/-- The deployed device's history at each play (`Deployment.dev`). -/
noncomputable abbrev dev (H : Host) (W : List Hash.Msg) (hR : ReadsBack H) : Exchange → History :=
  (deployment H W hR).dev families

/-- **The declared calls**: the most, over the play's tapes, of what the device's reading of what the prover sent
hashes (`Binding.breakCalls`) and one per blob it sent. -/
noncomputable def cost (H : Host) (W : List Hash.Msg) (hR : ReadsBack H) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Verity.Game.Strategy (Verity.play (families θ) (signer H W hR θ) [] θ F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play (families θ) (signer H W hR θ) [] θ F c) σ =>
    Binding.breakCalls (dev H W hR (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-! ## A vocabulary at the deployment -/

/-- **v2's reading at the deployed device**: `Idle.vocab δ 1` with the deployment's families, signer and signature count,
the auditor's pins `S` with the host's key the one pinned (`Host.pin`) and the windows `S` designates for the
challenger the deployment's, v2's reading of the committed files at the host's pinned tables (`Host.tableFiles`) and its
digests, the host's circuit and public files then its pinned table files as the registered files, the declared calls
`cost`, each forger at the key pinned at its index, and every name that reads a strategy of the play at them. -/
noncomputable def base (H : Host) (S : AuditorPins) (hR : ReadsBack H) (δ : ℝ≥0∞) : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab δ 1 with
    families := families
    signer := signer H (S.sessions .challenger) hR
    pins := { S with keys := [H.pin] }
    pinned := fun _ => rfl
    disjoint := fun _ l h => by
      change l ∈ List.map (·.1) (Parties.families []) at h
      simp [Parties.families] at h
    signer_families := fun _ => rfl
    read := fun θ F => .one (readOf H.tableFiles θ F)
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    registered := fun θ => #[H.circuitFile θ, H.publicFile θ] ++ H.tables
    place := fun θ F _ g => placeOfFiles (.one (readOf H.tableFiles θ F)) g
    extract := fun _ _ _ _ _ => 0
    finder := fun _ _ _ _ _ => Verity.Idle.finder
    cost := cost H (S.sessions .challenger) hR
    finder_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    finders := fun _ _ _ _ => []
    premises := fun _ _ _ _ _ => True
    forger := fun _ i _ _ _ _ => { Verity.Idle.forger with key := (([H.pin][i]?).map (·.key)).getD [] }
    forger_pinned := fun _ _ _ _ _ _ _ => rfl
    forger_calls := fun _ _ _ _ _ _ _ => Nat.zero_le _
    signatures := (deployment H (S.sessions .challenger) hR).signatures
    signs_le := fun θ P hP a log b ρ => by
      rcases List.mem_cons.1 hP with rfl | h
      · exact (deployment H (S.sessions .challenger) hR).signs_le families θ a log b ρ
      · exact absurd h List.not_mem_nil
    forger_signed := fun _ _ _ _ _ _ _ => Nat.zero_le _
    computed := fun _ _ _ _ _ => ∅
    macs := fun _ _ _ _ _ => 0
    world := fun _ _ _ _ _ => Verity.Idle.world
    world_window := fun _ _ _ _ _ => rfl
    world_capacity := fun _ _ _ _ _ => rfl
    schedule := fun θ F => Instance.Honest.blocks (readOf H.tableFiles θ F).circuit
    wipe := fun θ F => Instance.Honest.wipe (readOf H.tableFiles θ F).circuit
    wipeRun := fun θ F c _ =>
      Instance.Honest.ofRun (readOf H.tableFiles θ F).circuit
        (Verity.play (families θ) (signer H (S.sessions .challenger) hR θ) [] θ F c)
    honest := fun θ F c =>
      Verity.requestThen (families θ) (signer H (S.sessions .challenger) hR θ) [] θ F c .empty
        ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

/-- `Instance.signer` at `base`: its signer is the deployment's at its families. -/
theorem base_signer (H : Host) (S : AuditorPins) (hR : ReadsBack H) (δ : ℝ≥0∞) (θ : ByteArray) :
    (base H S hR δ).signer θ = (deployment H (S.sessions .challenger) hR).signer (base H S hR δ).families θ :=
  rfl

/-- `Instance.coins` at `base`: each family it names samples `MAX_COINS` coins. -/
theorem base_coins (H : Host) (S : AuditorPins) (hR : ReadsBack H) (δ : ℝ≥0∞) (θ : ByteArray) (l : String) (m : ℕ)
    (h : (l, m) ∈ (base H S hR δ).families θ) : m = MAX_COINS :=
  coins_families h

/-- **The pins designate the deployment's windows for the challenger** (`Deployment.binds`'s hypothesis). -/
theorem base_sessions (H : Host) (S : AuditorPins) (hR : ReadsBack H) (δ : ℝ≥0∞) :
    (base H S hR δ).pins.sessions .challenger = (deployment H (S.sessions .challenger) hR).sessions :=
  rfl

/-- `Instance.costs` at `base`: its declared calls cover the deployed device's reading at its history. -/
theorem costs (H : Host) (S : AuditorPins) (hR : ReadsBack H) (δ : ℝ≥0∞) :
    CostsRead (base H S hR δ) ((deployment H (S.sessions .challenger) hR).dev (base H S hR δ).families) :=
  fun θ F c σ t =>
    Finset.le_sup (f := fun t : Verity.Game.Tape
        (Verity.play (families θ) (signer H (S.sessions .challenger) hR θ) [] θ F c) σ =>
      Binding.breakCalls (dev H (S.sessions .challenger) hR (Verity.Game.Tape.out _ σ t))
        (Verity.Game.Tape.out _ σ t).sent + (Verity.Game.Tape.out _ σ t).sent.blobs.size) (Finset.mem_univ t)

/-! ## The audit's own parameters -/

/-- The public key of the parameters' challenger key (`Binding.keyOf`), the empty string where they name none. -/
def pubOf (θ : ByteArray) : ByteArray := ((Binding.keyOf θ).map (·.pub)).getD .empty

/-- **The deployment holds the audit's registration at `θ`**: v2's parameters in `θ` name a challenger whose key is the
host's, its public key and the deployed LMS parameters. -/
def Holds (H : Host) (θ : ByteArray) : Prop :=
  ∃ p key l, ParamsAt θ p ∧ p.base.challenger = some (key, l) ∧ key.pub = H.pub ∧ key.params = Lms.deployed

/-- At the audit's own parameters the layout is the parameters' family, at `MAX_COINS` coins. -/
theorem families_holds {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {key : ChallengerKey} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : families θ = [(l, MAX_COINS)] := by
  simp only [families, paramsOf_eq hp, hc]

/-- **Each challenger v2's parameters name gets `MAX_COINS` coins** from the deployment's layout: the family clause
`SetsUp.params` reads at an `Instance` (`Instance.coins`), with no hypothesis. -/
theorem families_params {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {key : ChallengerKey} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : (l, MAX_COINS) ∈ families θ := by
  rw [families_holds hp hc]
  exact List.mem_singleton_self _

/-- The first component of SPOT's id names the parameters' challenger key (`Binding.keyOf`). -/
theorem findSome_keyAt : ∀ {ps : List _root_.Core.Composition.Named} {x : _root_.Core.Composition.Named}
    {k : ChallengerKey}, ps.find? (·.id == VStar.Recursive.Sampled.PROTOCOL) = some x → Binding.keyAt x = some k →
    ps.findSome? Binding.keyAt = some k
  | [], _, _, h, _ => by simp at h
  | n :: ps, x, k, h, hk => by
    by_cases hn : n.id = VStar.Recursive.Sampled.PROTOCOL
    · rw [List.find?_cons_of_pos (by simpa using hn)] at h
      obtain rfl := Option.some.inj h
      simp only [List.findSome?_cons, hk]
    · rw [List.find?_cons_of_neg (by simpa using hn)] at h
      have hn' : Binding.keyAt n = none := by simp [Binding.keyAt, hn]
      rw [List.findSome?_cons, hn']
      exact findSome_keyAt h hk

/-- **At the audit's own parameters their challenger key is the host's**: the pinned key every run and ledger line of
the deployed device's history sits under (`Deployment.setupAt`'s `pub`). -/
theorem pubOf_holds {H : Host} {θ : ByteArray} (h : Holds H θ) : pubOf θ = H.pub := by
  obtain ⟨p, key, l, ⟨ps, hps, x, hx, hp⟩, hc, hpub, -⟩ := h
  have hid : x.id = VStar.Recursive.Sampled.PROTOCOL := by simpa using List.find?_some hx
  have hk : Binding.keyAt x = some key := by
    unfold Binding.keyAt
    rw [hp]
    simp [hid, hc]
  have hθ : Binding.keyOf θ = some key := by
    simp only [Binding.keyOf, hps]
    exact findSome_keyAt hx hk
  simp [pubOf, hθ, hpub]

end Verity.Sampled.LiveDevice
