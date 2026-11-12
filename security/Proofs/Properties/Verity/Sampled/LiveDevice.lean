module

public import Proofs.Properties.Verity.Sampled.Instance
public import Lms.Sign
public import Flock.JsonParse

/-!
# The live challenger as deployed: a deployment that holds a key, draws its coins and keeps the ledger

`Device.Deployment` is what fixes the device's history at a play (`Device.dev`). `Instance.Honest.deployment` is the
idle one: no key, no coins, no ledger. This file builds the one the auditor runs (`deployment`), the SPOT live
challenger on its auditor host, each field what the host's code does:

* **The host** (`Host`): what the auditor's machine holds when an audit starts. Its LMS key over SHA-512 at the
  deployed parameters (`Lms.deployed`, n = 64; `serve.py`'s and `auditor.py`'s `load_key` in
  `verity/composite/device/live_challenger/`), the key's next unused leaf (`lms.LeafState`), its name (each session's
  record signs `<name>.<session>`), the commit its tree is pinned at, the session id the gate reserves for the audit
  (`gate_reserve`'s `s<UTC>-<token>`), the ledger's next `seq` and its last line's SHA-512, the host's clock at each entry,
  and the body fields of its own each entry adds that the device's model doesn't compute (the pins log's seal, the coin
  link's port, the session secret's SHA-512, the session's other files' digests, serve.py's exit code). All of it is fixed
  before the play: none of it is the prover's.
* **The coins** (`families`, `draws`, `run`): the coin commitment's one family is the challenger family v2's parameters
  name, with `MAX_COINS` coins (`Instance.coins`'s count). At the close the device draws, for every position of the layout,
  the coin's `COIN_BYTES` bytes and its hm96 salt's `SALT_BYTES` bytes, as `LiveChallenger.__init__`
  (`verity/protocols/spot/device/challenger.py`) draws `(coin, salt)` per position; the play's released coins override
  the drawn ones at their positions (`Device.coinAt`). The run's scheme and epoch are the reserve request's, its host
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
* **At it** (`signer`, `dev`, `cost`, `base`): the deployed device's signer at `families`, its history at each play, the
  declared calls covering its reading (`costs`), and a vocabulary at it whose fields `Instance` ties to the deployment
  (`base_signer`, `coins_families`, `costs`).
* **The audit's own parameters** (`Holds`): v2's parameters in `θ` name a challenger whose key is the host's. There the
  history's key is the host's (`pubOf_holds`). Whatever challenger they name, the layout gives it `MAX_COINS` coins
  (`families_params`), the family clause of the session v2 sets up (`SetsUp.params`).
-/

@[expose] public section

namespace Verity.Sampled.LiveDevice

open Lean (Json)
open Spot Spot.Challenger
open Verity.Sampled.Device (RunSetup Deployment pubOf)

/-! ## The host -/

/-- **The auditor host the live challenger runs on**, as it stands when an audit starts at the parameters `θ`: its LMS
key at the deployed parameters (`deployed`), its name and pinned commit, and at each `θ` the session id the gate
reserves, the key's next unused leaf, the ledger's next `seq` and its last line's SHA-512 (`prev`, `ZERO` on an empty
ledger), its clock's UTC reading at the audit's `k`-th entry, and the body fields of its own the entry of each kind
adds (`extra`). -/
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

namespace Host

variable (H : Host)

/-- The host's public key, `I ‖ T[1]` (`lms.public_key`). -/
def pub : ByteArray := H.key.publicKey

/-- `key_sha512`: the SHA-512 of the host's public key, as each opening entry names the key (`key_signer`). -/
def keySha512 : String := sha512Hex H.pub

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

/-- The bits the device draws for one position of its layout: the coin's `COIN_BYTES` bytes and its hm96 salt's. -/
def SLOT_BITS : ℕ := 8 * (COIN_BYTES + _root_.Spot.Challenger.SALT_BYTES)

/-- **The bits the device draws at the close**: every position's coin and salt (`LiveChallenger.__init__`). -/
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
the gate reserved and the host the record signs, the reserve's scheme and epoch, each position's coin and salt drawn at
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

/-- The run's setup at a play, under the parameters' key and at `families`, before its ledger. -/
def setup (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : RunSetup :=
  { run H θ a log b ρ with key := pubOf θ, families := families θ }

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

/-- **An entry's line, as `append` writes it**: entry `seq θ + j` of `kind` for `session`, after the line whose SHA-512 is
`prev`, its `sig` the host's signature at leaf `q` over `ledger_message` of the entry without it, in the ledger's
canonical text (`ledger_text`). -/
def entryLine (H : Host) (θ : ByteArray) (j q : ℕ) (prev kind : String) (session : Option String) (body : Json) :
    Json :=
  let unsigned := Json.mkObj [("body", body), ("kind", .str kind), ("prev", .str prev), ("schema", .str LEDGER_SCHEMA),
    ("seq", Lean.toJson (H.seq θ + j)), ("session", (session.map Json.str).getD .null)]
  let m := (ledgerMessage (Properties.Challenger.ident H.pub) unsigned).toOption.getD .empty
  let r : _root_.Core.Record := ⟨body, kind, prev, LEDGER_SCHEMA, H.seq θ + j, session, H.signAt q m⟩
  .str ((PyJson.canonical r.toJson).toOption.getD "")

/-- **Consecutive entries' lines**: from entry `seq θ + j` at leaf `q`, after the line whose SHA-512 is `prev`, each
the next one's `prev` (`Core.Record.prevOf`). -/
def entries (H : Host) (θ : ByteArray) : ℕ → ℕ → String → List (String × Option String × Json) → List Json
  | _, _, _, [] => []
  | j, q, prev, (kind, s, body) :: es =>
    let l := H.entryLine θ j q prev kind s body
    l :: entries H θ (j + 1) (q + 1) (_root_.Core.Record.prevOf (strOf l)) es

theorem entries_length (H : Host) (θ : ByteArray) :
    ∀ (j q : ℕ) (prev : String) (es : List (String × Option String × Json)), (entries H θ j q prev es).length = es.length
  | _, _, _, [] => rfl
  | j, q, prev, (_, _, _) :: es => by
    simp only [entries, List.length_cons]
    rw [entries_length H θ]

/-- A body: the host's own fields of the entry's kind, then the request's `keep` fields of `part`, then the device's,
each later field in place of an earlier one of its name. -/
def body (H : Host) (θ : ByteArray) (kind : String) (part : Json) (keep : List String) (own : List (String × Json)) :
    Json :=
  Json.mkObj (H.extra θ kind ++ (fieldsOf part).filter (keep.contains ·.1) ++ own)

/-- The request fields a reserve records and its open carries (`RESERVED`), and whether its session is withheld. -/
def RESERVE_FIELDS : List String := ["by", "for", "peer", "scheme", "epoch", "minutes", "withheld"]

/-- **The period's open** (`period`'s `period-open`): the request's period, the pinned commit, the key and the time. -/
def periodOpenBody (H : Host) (θ a : ByteArray) : Json :=
  H.body θ "period-open" (requestPart a "period")
    ["period", "lane", "by", "cap", "circuit", "draw_k", "rows", "weights_root", "model", "circuit_files", "length",
      "policy", "alpha", "gamma", "registration", "policy_name"]
    [("pinned", .str H.pinned), ("key_sha512", .str H.keySha512), ("opened_utc", .str (H.clock θ 0))]

/-- **The window's reserve** (`reserve`): the request, the layout the session commits to, the pinned commit, the key, the
record's host, and when it starts and expires (`later`). -/
def reserveBody (H : Host) (θ a : ByteArray) : Json :=
  H.body θ "reserve" (requestPart a "reserve") RESERVE_FIELDS
    [("families", familiesJson (families θ)), ("pinned", .str H.pinned), ("key_sha512", .str H.keySha512),
      ("host", .str (H.hostAt θ)), ("started_utc", .str (H.clock θ 1)),
      ("expires_utc", .str ((later (H.clock θ 1) ((nat? (getD (requestPart a "reserve") "minutes")).getD 0)).getD ""))]

/-- **The window's open** (`admit`): the window's root the opener named, its reserve's `seq`, the reserve's request and
layout, its opener, the pinned commit, the key, the record's host and the time. -/
def openBody (H : Host) (θ a : ByteArray) : Json :=
  H.body θ "open" (requestPart a "reserve") RESERVE_FIELDS
    [("window", Json.mkObj [("root", getD (requestPart a "open") "root")]), ("reserve", Lean.toJson (H.seq θ + 1)),
      ("families", familiesJson (families θ)), ("opened_by", getD (requestPart a "open") "by"),
      ("pinned", .str H.pinned), ("key_sha512", .str H.keySha512), ("host", .str (H.hostAt θ)),
      ("started_utc", .str (H.clock θ 2))]

/-- **The ledger's entries before any coin**: the period's open, the window's reserve and its open, entries `seq θ` to
`seq θ + 2` at the host's leaves `leaf θ` to `leaf θ + 2`, after the host's ledger head. -/
def opens (H : Host) (θ a : ByteArray) : List Json :=
  entries H θ 0 (H.leaf θ) (H.prev θ)
    [("period-open", none, periodOpenBody H θ a), ("reserve", some (H.session θ), reserveBody H θ a),
      ("open", some (H.session θ), openBody H θ a)]

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
  H.body θ "end" .null []
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
  H.body θ "period-close" (requestPart a "period") ["period", "lane", "by"]
    [("windows", .arr #[window]), ("complete", .bool (complete θ log)), ("closed_utc", .str (H.clock θ 4))]

/-- **The ledger's entries at the close**: the window's end and the period's close, entries `seq θ + 3` and
`seq θ + 4`, after the window's open, at the leaves after the run's record and close (`leaf θ + 5`, `leaf θ + 6`). -/
def closes (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : List Json :=
  entries H θ 3 (H.leaf θ + 5) (_root_.Core.Record.prevOf (strOf ((opens H θ a).getLastD .null)))
    [("end", some (H.session θ), endBody H θ a log b ρ), ("period-close", none, periodCloseBody H θ a log)]

theorem opens_length (H : Host) (θ a : ByteArray) : (opens H θ a).length = 3 :=
  entries_length H θ _ _ _ _

theorem closes_length (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    (closes H θ a log b ρ).length = 2 :=
  entries_length H θ _ _ _ _

/-! ## The deployment -/

/-- **The live challenger as deployed** on the host `H`: the run of the session the gate reserved, its coins and salts
drawn at the close, the ledger's three entries before any coin and two at the close, and each of the play's messages
signed with the host's key at its leaf. -/
def deployment (H : Host) : Deployment where
  run := run H
  opens := opens H
  closes := closes H
  ledgerLines _ := 5
  ledger_le θ a log b ρ := by rw [opens_length, closes_length]
  draws := draws
  sign θ i m := (H.signAt (H.leaf θ + i) ⟨m.toArray⟩).data.toList

/-- The deployed device's signer at the parameters `θ`, at `families`. -/
def signer (H : Host) (θ : ByteArray) : Signer := (deployment H).signer families θ

/-- The deployed device's history at each play (`Device.dev`). -/
noncomputable abbrev dev (H : Host) : Exchange → Properties.Challenger.History :=
  Device.dev ((deployment H).setupOf families)

/-- **The declared calls**: the most, over the play's tapes, of what the device's reading of what the prover sent
hashes (`Binding.breakCalls`) and one per blob it sent. -/
noncomputable def cost (H : Host) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play (families θ) (signer H θ) θ F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play (families θ) (signer H θ) θ F c) σ =>
    Binding.breakCalls (dev H (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

/-! ## A vocabulary at the deployment -/

/-- **v2's reading at the deployed device**: `Idle.vocab δ 1` with the deployment's families, signer and signature count,
v2's reading of the committed files and its digests, the files the digests name as the registered files
(`registeredOf`), the declared calls `cost`, and every name that reads a strategy of the play at them. -/
noncomputable def base (H : Host) (δ : ℝ≥0∞) : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab δ 1 with
    families := families
    signer := signer H
    read := readOf
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    registered := registeredOf
    place := fun θ F _ g => placeOfRead (readOf θ F) g
    extract := fun _ _ _ _ _ => 0
    finder := fun _ _ _ _ _ => Verity.Idle.finder
    cost := cost H
    finder_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    finders := fun _ _ _ _ => []
    premises := fun _ _ _ _ _ => True
    forger := fun _ _ _ _ _ => Verity.Idle.forger
    forger_calls := fun _ _ _ _ _ _ => Nat.zero_le _
    signatures := (deployment H).signatures
    signs_le := (deployment H).signs_le families
    forger_signed := fun _ _ _ _ _ _ => Nat.zero_le _
    computed := fun _ _ _ _ _ => ∅
    world := fun _ _ _ _ _ => Verity.Idle.world
    world_window := fun _ _ _ _ _ => rfl
    world_capacity := fun _ _ _ _ _ => rfl
    theta := fun _ _ => ∅
    theta_input := fun _ _ _ h => absurd h (Finset.notMem_empty _)
    schedule := fun θ F => Instance.Honest.blocks (readOf θ F).circuit
    wipe := fun θ F => Instance.Honest.wipe (readOf θ F).circuit
    wipeRun := fun θ F c _ =>
      Instance.Honest.ofRun (readOf θ F).circuit (Verity.play (families θ) (signer H θ) θ F c)
    honest := fun θ F c =>
      Verity.requestThen (families θ) (signer H θ) θ F c .empty ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

/-- `Instance.signer` at `base`: its signer is the deployment's at its families. -/
theorem base_signer (H : Host) (δ : ℝ≥0∞) (θ : ByteArray) :
    (base H δ).signer θ = (deployment H).signer (base H δ).families θ :=
  rfl

/-- `Instance.coins` at `base`: each family it names samples `MAX_COINS` coins. -/
theorem base_coins (H : Host) (δ : ℝ≥0∞) (θ : ByteArray) (l : String) (m : ℕ) (h : (l, m) ∈ (base H δ).families θ) :
    m = MAX_COINS :=
  coins_families h

/-- `Instance.costs` at `base`: its declared calls cover the deployed device's reading at its history. -/
theorem costs (H : Host) (δ : ℝ≥0∞) : CostsRead (base H δ) (Device.dev ((deployment H).setupOf (base H δ).families)) :=
  fun θ F c σ t =>
    Finset.le_sup (f := fun t : Verity.Game.Tape (Verity.play (families θ) (signer H θ) θ F c) σ =>
      Binding.breakCalls (dev H (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
        (Verity.Game.Tape.out _ σ t).sent.blobs.size) (Finset.mem_univ t)

/-! ## The audit's own parameters -/

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

/-- The first component of sampled proofs' id names the parameters' challenger key (`Binding.keyOf`). -/
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

/-- **At the audit's own parameters the history's key is the host's**: the key every run and ledger line of the
deployed device's history sits under (`Deployment.setupAt`'s `pubOf`). -/
theorem pubOf_holds {H : Host} {θ : ByteArray} (h : Holds H θ) : pubOf θ = H.pub := by
  obtain ⟨p, key, l, ⟨ps, hps, x, hx, hp⟩, hc, hpub, -⟩ := h
  have hid : x.id = VStar.Recursive.Sampled.PROTOCOL := by simpa using List.find?_some hx
  have hk : Binding.keyAt x = some key := by simp [Binding.keyAt, hid, hp, hc]
  have hθ : Binding.keyOf θ = some key := by
    simp only [Binding.keyOf, hps]
    exact findSome_keyAt hx hk
  simp [pubOf, hθ, hpub]

end Verity.Sampled.LiveDevice
