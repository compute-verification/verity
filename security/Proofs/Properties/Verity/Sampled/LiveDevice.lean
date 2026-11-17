module

public import Proofs.Properties.Verity.Sampled.Instance
public import Lms.Sign
public import Base.JsonParse

/-! The live challenger as deployed: a deployment that holds a key, flips its coins and keeps the ledger. -/

@[expose] public section

namespace Verity.Sampled.LiveDevice

open scoped ENNReal
open Lean (Json)
open Spot Spot.Challenger
open Verity.Sampled.Device (RunSetup Deployment pubOf)

/-- The auditor device the live challenger runs on, as it stands when an audit starts at the parameters
`θ`; `prev` is `ZERO` on an empty ledger. -/
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

def tableFiles : PinnedTables := .ofFiles H.tables

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

def families (θ : ByteArray) : Families :=
  match paramsOf θ with
  | some p =>
    match p.base.challenger with
    | some (_, l) => [(l, MAX_COINS)]
    | none => []
  | none => []

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

def draws (θ : ByteArray) : ℕ := _root_.Spot.Challenger.total (families θ) * SLOT_BITS

def bytesAt (ρ : List Bool) (off n : ℕ) : ByteArray := Lines.coinBytes ((ρ.drop off).take n)

/-- Bytes read as JSON, `null` where they aren't UTF-8 text that parses. -/
def jsonOf (b : ByteArray) : Json :=
  match String.fromUTF8? b with
  | some s => (Flock.JsonParse.parse s).toOption.getD .null
  | none => .null

def requestPart (a : ByteArray) (k : String) : Json := getD (jsonOf a) k

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

def setup (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : RunSetup :=
  { run H θ a log b ρ with key := pubOf θ, families := families θ }

def commitmentAt (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    Option LiveCommitment :=
  (Device.runOf (setup H θ a log b ρ) log []).commitment

def releaseLog (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    Option (List String) :=
  (Device.printLog (setup H θ a log b ρ).time Device.PrintState.init log).map
    (Device.headLine (commitmentAt H θ a log b ρ) :: ·)

def entryLine (H : Host) (θ : ByteArray) (j q : ℕ) (prev kind : String) (session : Option String) (body : Json) :
    Json :=
  let unsigned := Json.mkObj [("body", body), ("kind", .str kind), ("prev", .str prev), ("schema", .str LEDGER_SCHEMA),
    ("seq", Lean.toJson (H.seq θ + j)), ("session", (session.map Json.str).getD .null)]
  let m := (ledgerMessage (Properties.Challenger.ident H.pub) unsigned).toOption.getD .empty
  let r : _root_.Core.Record := ⟨body, kind, prev, LEDGER_SCHEMA, H.seq θ + j, session, H.signAt q m⟩
  .str ((PyJson.canonical r.toJson).toOption.getD "")

def entries (H : Host) (θ : ByteArray) : ℕ → ℕ → String → List (String × Option String × Json) → List Json
  | _, _, _, [] => []
  | j, q, prev, (kind, s, body) :: es =>
    let l := entryLine H θ j q prev kind s body
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

def RESERVE_FIELDS : List String := ["by", "for", "peer", "scheme", "epoch", "minutes", "withheld"]

def periodOpenBody (H : Host) (θ a : ByteArray) : Json :=
  body H θ "period-open" (requestPart a "period")
    ["period", "lane", "by", "cap", "circuit", "draw_k", "rows", "weights_root", "model", "circuit_files", "length",
      "policy", "alpha", "gamma", "registration", "policy_name"]
    [("pinned", .str H.pinned), ("key_sha512", .str H.keySha512), ("opened_utc", .str (H.clock θ 0))]

def reserveBody (H : Host) (θ a : ByteArray) : Json :=
  body H θ "reserve" (requestPart a "reserve") RESERVE_FIELDS
    [("families", familiesJson (families θ)), ("pinned", .str H.pinned), ("key_sha512", .str H.keySha512),
      ("host", .str (H.hostAt θ)), ("started_utc", .str (H.clock θ 1)),
      ("expires_utc", .str ((later (H.clock θ 1) ((nat? (getD (requestPart a "reserve") "minutes")).getD 0)).getD ""))]

def openBody (H : Host) (θ a : ByteArray) : Json :=
  body H θ "open" (requestPart a "reserve") RESERVE_FIELDS
    [("window", Json.mkObj [("root", getD (requestPart a "open") "root")]), ("reserve", Lean.toJson (H.seq θ + 1)),
      ("families", familiesJson (families θ)), ("opened_by", getD (requestPart a "open") "by"),
      ("pinned", .str H.pinned), ("key_sha512", .str H.keySha512), ("host", .str (H.hostAt θ)),
      ("started_utc", .str (H.clock θ 2))]

def opens (H : Host) (θ a : ByteArray) : List Json :=
  entries H θ 0 (H.leaf θ) (H.prev θ)
    [("period-open", none, periodOpenBody H θ a), ("reserve", some (H.session θ), reserveBody H θ a),
      ("open", some (H.session θ), openBody H θ a)]

def closed (b : ByteArray) : Bool :=
  match getD (closeJson b) "ended" with
  | .str _ => false
  | _ => true

def releasedJson (θ : ByteArray) (log : List Release) : Json :=
  Json.mkObj ((families θ).map fun (l, _) => (l, Lean.toJson (Device.coinsOf l log).length))

def complete (θ : ByteArray) (log : List Release) : Bool :=
  (families θ).all fun (l, n) => (Device.coinsOf l log).length == n

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

def periodCloseBody (H : Host) (θ a : ByteArray) (log : List Release) : Json :=
  let window := Json.mkObj [("root", getD (requestPart a "open") "root"), ("session", .str (H.session θ)),
    ("complete", .bool (complete θ log))]
  body H θ "period-close" (requestPart a "period") ["period", "lane", "by"]
    [("windows", .arr #[window]), ("complete", .bool (complete θ log)), ("closed_utc", .str (H.clock θ 4))]

def closes (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) : List Json :=
  entries H θ 3 (H.leaf θ + 5) (_root_.Core.Record.prevOf (strOf ((opens H θ a).getLastD .null)))
    [("end", some (H.session θ), endBody H θ a log b ρ), ("period-close", none, periodCloseBody H θ a log)]

theorem opens_length (H : Host) (θ a : ByteArray) : (opens H θ a).length = 3 :=
  entries_length H θ _ _ _ _

theorem closes_length (H : Host) (θ a : ByteArray) (log : List Release) (b : ByteArray) (ρ : List Bool) :
    (closes H θ a log b ρ).length = 2 :=
  entries_length H θ _ _ _ _

def deployment (H : Host) : Deployment where
  run := run H
  opens := opens H
  closes := closes H
  ledgerLines _ := 5
  ledger_le θ a log b ρ := by rw [opens_length, closes_length]
  draws := draws
  sign θ i m := (H.signAt (H.leaf θ + i) ⟨m.toArray⟩).data.toList

def signer (H : Host) (θ : ByteArray) : Signer := (deployment H).signer families θ

noncomputable abbrev dev (H : Host) : Exchange → Properties.Challenger.History :=
  Device.dev ((deployment H).setupOf families)

noncomputable def cost (H : Host) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)
    (σ : Verity.Game.Strategy (Verity.play (families θ) (signer H θ) θ F c)) : ℕ :=
  Finset.univ.sup fun t : Verity.Game.Tape (Verity.play (families θ) (signer H θ) θ F c) σ =>
    Binding.breakCalls (dev H (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
      (Verity.Game.Tape.out _ σ t).sent.blobs.size

noncomputable def base (H : Host) (δ : ℝ≥0∞) : Vocab Verity.Idle.device Unit :=
  { Verity.Idle.vocab δ 1 with
    families := families
    signer := signer H
    read := fun θ F => .one (readOf H.tableFiles θ F)
    programDigest := programDigestOf
    publicDigest := publicDigestOf
    registered := fun θ => #[H.circuitFile θ, H.publicFile θ] ++ H.tables
    place := fun θ F _ g => placeOfFiles (.one (readOf H.tableFiles θ F)) g
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
    macs := fun _ _ _ _ _ => 0
    world := fun _ _ _ _ _ => Verity.Idle.world
    world_window := fun _ _ _ _ _ => rfl
    world_capacity := fun _ _ _ _ _ => rfl
    schedule := fun θ F => Instance.Honest.blocks (readOf H.tableFiles θ F).circuit
    wipe := fun θ F => Instance.Honest.wipe (readOf H.tableFiles θ F).circuit
    wipeRun := fun θ F c _ =>
      Instance.Honest.ofRun (readOf H.tableFiles θ F).circuit (Verity.play (families θ) (signer H θ) θ F c)
    honest := fun θ F c =>
      Verity.requestThen (families θ) (signer H θ) θ F c .empty ⟨.empty, fun _ => ⟨(.empty, #[]), ⟨⟩⟩⟩ }

theorem base_signer (H : Host) (δ : ℝ≥0∞) (θ : ByteArray) :
    (base H δ).signer θ = (deployment H).signer (base H δ).families θ :=
  rfl

theorem base_coins (H : Host) (δ : ℝ≥0∞) (θ : ByteArray) (l : String) (m : ℕ) (h : (l, m) ∈ (base H δ).families θ) :
    m = MAX_COINS :=
  coins_families h

theorem costs (H : Host) (δ : ℝ≥0∞) : CostsRead (base H δ) (Device.dev ((deployment H).setupOf (base H δ).families)) :=
  fun θ F c σ t =>
    Finset.le_sup (f := fun t : Verity.Game.Tape (Verity.play (families θ) (signer H θ) θ F c) σ =>
      Binding.breakCalls (dev H (Verity.Game.Tape.out _ σ t)) (Verity.Game.Tape.out _ σ t).sent +
        (Verity.Game.Tape.out _ σ t).sent.blobs.size) (Finset.mem_univ t)

def Holds (H : Host) (θ : ByteArray) : Prop :=
  ∃ p key l, ParamsAt θ p ∧ p.base.challenger = some (key, l) ∧ key.pub = H.pub ∧ key.params = Lms.deployed

theorem families_holds {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {key : ChallengerKey} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : families θ = [(l, MAX_COINS)] := by
  simp only [families, paramsOf_eq hp, hc]

theorem families_params {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} {key : ChallengerKey} {l : String}
    (hp : ParamsAt θ p) (hc : p.base.challenger = some (key, l)) : (l, MAX_COINS) ∈ families θ := by
  rw [families_holds hp hc]
  exact List.mem_singleton_self _

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
