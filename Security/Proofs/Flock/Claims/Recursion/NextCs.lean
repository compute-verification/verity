import Proofs.Flock.Claims.Recursion.PartsCompose
import Proofs.Flock.Claims.Complete.Json.Decode
import Proofs.Flock.Soundness.Refine.Exec
import Std.Data.TreeMap.Raw.AdditionalOperations

/-!
# The deployed plain verifier with the record's coins read from the live coins (`nextCs`)

The plain verifier's custody (`FlockDeploy.custody`) compares the coins the record carries with the live coins `cs`,
and its verdict (`verdictM`) reads the coins off the record. `nextCs` is the plain verifier's step with the record's
coins substituted after parsing: at a decision whose record parses to `j`, it runs the verdict (`verifyJWithM`) and
custody's reading (`liveOfJ`) at `coinSubJ spec c₁ mids j`, `j` with its coins replaced by the live coins, while the
statement and draw checks and S0 (`RecordText.check`) read the record's text as the plain verifier does. A record that
is not JSON is decided as the plain verifier decides it.

* **Reading at a parsed value** (`liveOfJ`, `verdictJM`, `checksJ`, `decisionJ`): the plain verifier's readers after
  the parse; at a record that parses to `j` each is the plain one (`liveOf_of_parse`, `verdictM_of_parse`,
  `checks_of_parse`, `decisionAt_of_parse`).
* **The substitution** (`coinSubJ`): the link's points and every round's coins erased (`eraseCoins`), then filled
  (`fillCoins`): the points `pointsOf spec c₁`, round `k` of rep stream `name` the first `n` of the coin at
  `streamOffset + k`. It maps an object's values in place (`mapKey`, `Std.TreeMap.Raw.map`), keeping its keys and its
  tree, so a lookup reads through it whatever the parsed tree (`getObjVal?_mapKey_self`, `getObjVal?_mapKey_ne`).

Three facts, and custody:

* **Plain acceptance carries over** (`decisionCsAt_canonical`, `decisionCs_canonical`, `nextCs_canonical`): at a record that parses to a value
  the substitution fixes, `nextCs` is the plain `next`.
* **Soundness transfers at the parsed value** (`nextCs_decide`, `decisionCsAt_parse`): `nextCs` at a decision is the
  plain verdict and checks at the substituted value, so a route that proves the plain step at every parsed value
  (`verifyJWithM`) covers it.
* **The coin text is not read** (`decisionCsAt_coinBlind`, `nextCs_coinBlind`): two decisions whose records are in
  their canonical text (S0 passes, `RecordText.check_of`) and parse to values equal once their coins are erased, with
  the same statement, proofs and draw, step alike.
* **Custody holds by construction** (`liveOfJ_points`, `liveOfJ_coins`, `custody_coinSubJ`): at the substituted value
  custody's points are the first coin's and, at rep streams of distinct names, every recorded round it pairs with a
  round message carries the first `n` of that message's coin (`f128s_map`, the hex round trip), so custody is its
  byte comparisons alone. No JSON premise is taken.

A coin is answered as the plain verifier answers it (`nextCs_coin`), so `bdDeployed` at `nextCs` splits as at `next`
(`RecDeploy.decompB_nextCs`).
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-! ## Reading the record at a parsed value -/

/-- A recorded round as `liveOf` reads it. -/
def roundJ (r : Json) : Flock.V Round := do
  let n ← Record.nat r "n"
  let coins ← Record.f128s (← Record.arr r "coins")
  let bytes ← if Record.isNull r "msg" then pure none else some <$> unhex (← Record.str r "msg")
  pure (⟨n, ByteArray.empty, coins, bytes⟩ : Round)

/-- Rep stream `name`'s rounds as `liveOf` reads them: the first stream of that name's. -/
def streamJ (recs : Array Json) (name : String) : Flock.V (Array Round) := do
  let some s := recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name)
    | throw s!"S11: no stream {name}"
  (← Record.arr s "rounds").mapM fun r => roundJ r

/-- **`liveOf` at the record's parsed value `j`.** -/
def liveOfJ (spec : SessionSpec) (j : Json) : Flock.V Live := do
  let l ← Record.fld j "link"
  let rootB ← spec.tables.mapM fun t => do unhex (← Record.str (← Record.fld l "root_b") t)
  let pubs ← Record.fld l "publics"
  let publics ← spec.tables.mapM fun t =>
    if Record.isNull pubs t then pure ByteArray.empty else do unhex (← Record.str pubs t)
  let points ← (← Record.arr l "points").mapM fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => .error "record: a link point is not a list"
  let recs ← Record.arr j "streams"
  let streams ← spec.repStreams.mapM fun (_, _, name) => streamJ recs name
  pure ⟨rootB, publics, points, streams⟩

theorem liveOf_of_parse (spec : SessionSpec) {record : String} {j : Json} (h : Json.parse record = .ok j) :
    liveOf spec record = liveOfJ spec j := by
  unfold liveOf
  rw [h]
  rfl

/-- The shipped verdict at the record's parsed value `j`. -/
def verdictJM {M : Type → Type} [Monad M] [MonadExceptOf String M] [MonadLiftT Flock.V M] (zk : Bool) (hs : Hashes M)
    (spec : SessionSpec) (tabs : Array Setup) (j : Json) (proofs : Array ByteArray) : M Unit :=
  if zk then Zk.verifyJWithM hs spec tabs j proofs else verifyJWithM hs spec tabs j proofs

theorem verdictM_of_parse {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT Flock.V M]
    (zk : Bool) (hs : Hashes M) (spec : SessionSpec) (tabs : Array Setup) {record : String} {j : Json}
    (proofs : Array ByteArray) (h : Json.parse record = .ok j) :
    verdictM zk hs spec tabs record proofs = verdictJM zk hs spec tabs j proofs := by
  unfold verdictM verdictJM
  rw [verifyWithM_of_parse hs spec tabs proofs h, Zk.verifyWithM_of_parse hs spec tabs proofs h]

/-- `checks` with custody reading the parsed value `j`; the statement and draw checks and S0 read the record's text. -/
def checksJ (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) (j : Json) : Flock.V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (drawEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw)
    "live: the record's draw is not the statement's"
  custody spec (← liveOfJ spec j) link c₁ mids
  s.read.check
  (if s.read.draw.isNone then ProvedScope.drawLaw zk s.read.tags false none else pure ())
  RecordText.check f.record

theorem checks_of_parse (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (h : Json.parse f.record = .ok j) : checks zk s spec x link c₁ mids f = checksJ zk s spec x link c₁ mids f j := by
  unfold checks checksJ
  rw [liveOf_of_parse spec h]

/-- `decisionAt` with the verdict and custody at the parsed value `j`. -/
def decisionJ (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) (j : Json) :
    Claims.OCV Unit := do
  verdictJM zk Claims.Hashes.oc spec (tabsOf sts) j f.proofs
  monadLift (checksJ zk s spec x link c₁ mids f j)

theorem decisionAt_of_parse (zk : Bool) (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (h : Json.parse f.record = .ok j) :
    decisionAt zk s spec sts x link c₁ mids f = decisionJ zk s spec sts x link c₁ mids f j := by
  unfold decisionAt decisionJ
  rw [verdictM_of_parse zk _ spec _ _ h, checks_of_parse zk s spec x link c₁ mids h]

/-! ## Maps over a parsed value -/

/-- `j`'s value at key `k` mapped by `g`, its keys and tree kept; a value that is not an object is kept. -/
def mapKey (k : String) (g : Json → Json) : Json → Json
  | .obj m => .obj (m.map fun k' v => if k' = k then g v else v)
  | j => j

/-- An array's elements mapped by `g`; a value that is not an array is kept. -/
def mapArr (g : Json → Json) : Json → Json
  | .arr a => .arr (a.map g)
  | j => j

/-- An array's elements mapped by `g` with their indices. -/
def mapArrIdx (g : ℕ → Json → Json) : Json → Json
  | .arr a => .arr (a.mapIdx g)
  | j => j

theorem impl_get?_map (f : String → Json → Json) :
    ∀ (t : Std.DTreeMap.Internal.Impl String (fun _ => Json)) (k : String),
      Std.DTreeMap.Internal.Impl.Const.get? (t.map f) k = (Std.DTreeMap.Internal.Impl.Const.get? t k).map (f k)
  | .leaf, _ => rfl
  | .inner _ k' v l r, k => by
    simp only [Std.DTreeMap.Internal.Impl.map, Std.DTreeMap.Internal.Impl.Const.get?]
    split
    · exact impl_get?_map f l k
    · exact impl_get?_map f r k
    · rename_i h
      rw [Std.LawfulEqOrd.eq_of_compare h]
      rfl

/-- **A lookup reads through a value map**, with no hypothesis on the tree. -/
theorem get?_map (f : String → Json → Json) (m : Std.TreeMap.Raw String Json compare) (k : String) :
    (m.map f).get? k = (m.get? k).map (f k) :=
  impl_get?_map f m.inner.inner k

theorem getObjVal?_mapKey_self (k : String) (g : Json → Json) (j : Json) :
    (mapKey k g j).getObjVal? k = g <$> j.getObjVal? k := by
  cases j with
  | obj m =>
    simp only [mapKey, Json.getObjVal?, get?_map]
    cases m.get? k <;> simp <;> rfl
  | _ => rfl

theorem getObjVal?_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (h : k' ≠ k) :
    (mapKey k g j).getObjVal? k' = j.getObjVal? k' := by
  cases j with
  | obj m =>
    simp only [mapKey, Json.getObjVal?, get?_map]
    cases m.get? k' <;> simp [h]
  | _ => rfl

/-! ## The coin substitution -/

/-- A stream's name as the record writes it, `""` if none. -/
def nameOf (s : Json) : String := (s.getObjValAs? String "stream").toOption.getD ""

/-- A round's count as `liveOf` reads it, `0` if none. -/
def nOf (r : Json) : ℕ := match Record.nat r "n" with
  | .ok n => n
  | .error _ => 0

/-- Each round of a stream, `g name k round` at its round `k`. -/
def mapStream (g : String → ℕ → Json → Json) (s : Json) : Json := mapKey "rounds" (mapArrIdx (g (nameOf s))) s

/-- Each round of each stream of the record. -/
def mapRounds (g : String → ℕ → Json → Json) : Json → Json := mapKey "streams" (mapArr (mapStream g))

/-- The record with the link's points `v`. -/
def setPoints (v : Json) : Json → Json := mapKey "link" (mapKey "points" fun _ => v)

/-- **The record with its coins erased**: the link's points and every round's coins `null`. -/
def eraseCoins (j : Json) : Json := mapRounds (fun _ _ => mapKey "coins" fun _ => .null) (setPoints .null j)

/-- Field elements as the record writes them, hex strings. -/
def hexArr (a : Array F128) : Json := .arr (a.map FlockComplete.f128J)

/-- Stream `name`'s recorded rounds as `liveOf` finds the stream, `0` if none. -/
def roundsOf (j : Json) (name : String) : ℕ :=
  match (Record.arr j "streams").toOption.bind
      (·.find? fun s => (s.getObjValAs? String "stream").toOption == some name) with
  | some s => ((Record.arr s "rounds").toOption.map (·.size)).getD 0
  | none => 0

/-- Where stream `name`'s rounds start among the round coins: the rounds of the rep streams before it. -/
def streamOffset (spec : SessionSpec) (j : Json) (name : String) : ℕ :=
  (((spec.repStreams.toList.map (·.2.2)).takeWhile (· ≠ name)).map (roundsOf j)).sum

/-- **The record's coins filled from the live coins**: the link's points the first coin's (`pointsOf`), round `k` of
stream `name` the first `n` of its coin at `streamOffset + k`. -/
def fillCoins (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) : Json :=
  mapRounds (fun name k r => match mids[streamOffset spec j name + k]? with
      | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
      | none => r)
    (setPoints (.arr ((pointsOf spec c₁).map hexArr)) j)

/-- **The record's parsed value with its coins the live coins**: erased, then filled. -/
def coinSubJ (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) : Json :=
  fillCoins spec c₁ mids (eraseCoins j)

/-! ## The step -/

/-- The decision at a session with the record's coins substituted after parsing; a record that is not JSON is
decided as the plain verifier decides it. -/
def decisionCsAt (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit :=
  match Json.parse f.record with
  | .ok j => decisionJ false s spec sts x link c₁ mids f (coinSubJ spec c₁ mids j)
  | .error _ => decisionAt false s spec sts x link c₁ mids f

/-- **The decision with the record's coins the live coins**: the statement's setup over `OC`, then `decisionCsAt`. -/
def decisionCs (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : Final) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionCsAt s spec sts x link c₁ mids f

/-- **The plain verifier's step with the record's coins `cs`'s.** -/
def nextCs (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  match act false x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionCs s x link c₁ mids f)

/-- **A coin is answered as the plain verifier answers it**, with no query (`deploySplit`'s `hcoin`). -/
theorem nextCs_coin (H : Claims.Msg → Claims.Msg) : ∀ (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (c : Coin), act false x as cs = .coin c → ((nextCs x as cs).eval H).1 = .msg c.bytes := by
  intro x as cs c h
  simp only [nextCs, h, Claims.OC.eval]

/-- At a decision, `nextCs` decides by `decisionCs`. -/
theorem nextCs_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f) :
    nextCs x as cs = outOf (decisionCs s x link c₁ mids f) := by
  simp only [nextCs, h]

/-- **Soundness transfers at the parsed value**: at a record that parses to `j`, the decision is the plain verdict and
checks at `coinSubJ spec c₁ mids j`. -/
theorem decisionCsAt_parse (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (hj : Json.parse f.record = .ok j) :
    decisionCsAt s spec sts x link c₁ mids f = decisionJ false s spec sts x link c₁ mids f (coinSubJ spec c₁ mids j) := by
  simp only [decisionCsAt, hj]

/-- **Plain acceptance carries over** at a session: at a record that parses to a value the substitution fixes, the
decision is the plain one. -/
theorem decisionCsAt_canonical (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (hj : Json.parse f.record = .ok j) (hk : coinSubJ spec c₁ mids j = j) :
    decisionCsAt s spec sts x link c₁ mids f = decisionAt false s spec sts x link c₁ mids f := by
  rw [decisionCsAt_parse s spec sts x link c₁ mids hj, hk, decisionAt_of_parse false s spec sts x link c₁ mids hj]

/-- **Plain acceptance carries over** at the decision: at a record that parses to a value the substitution fixes at
every session, `decisionCs` is the plain `decision`. -/
theorem decisionCs_canonical (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json} (hj : Json.parse f.record = .ok j)
    (hk : ∀ spec, coinSubJ spec c₁ mids j = j) : decisionCs s x link c₁ mids f = decision false s x link c₁ mids f := by
  unfold decisionCs decision
  congr 1
  funext p
  obtain ⟨spec, sts⟩ := p
  exact decisionCsAt_canonical s spec sts x link c₁ mids hj (hk spec)

/-- **Plain acceptance carries over**: at a decision whose record parses to a value the substitution fixes at every
session, `nextCs` is the plain `next`. -/
theorem nextCs_canonical {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    {j : Json} (h : act false x as cs = .decide s spec sts link c₁ mids f) (hj : Json.parse f.record = .ok j)
    (hk : ∀ spec', coinSubJ spec' c₁ mids j = j) : nextCs x as cs = next false x as cs := by
  simp only [nextCs, next, h, decisionCs_canonical s x link c₁ mids hj hk]

/-- **The coin text is not read** at a session: two records in their canonical text that parse to values equal once
their coins are erased, with the same statement, proofs and draw, are decided alike. -/
theorem decisionCsAt_coinBlind (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f₁ f₂ : Final}
    {j₁ j₂ : Json} (hj₁ : Json.parse f₁.record = .ok j₁) (hj₂ : Json.parse f₂.record = .ok j₂)
    (he : eraseCoins j₁ = eraseCoins j₂) (hst : f₁.stmt = f₂.stmt) (hp : f₁.proofs = f₂.proofs)
    (hd : FlockSoundness.Discharge.Exec.recordDraw f₁.record = FlockSoundness.Discharge.Exec.recordDraw f₂.record)
    (hc₁ : RecordText.printT j₁ = f₁.record) (hc₂ : RecordText.printT j₂ = f₂.record) :
    decisionCsAt s spec sts x link c₁ mids f₁ = decisionCsAt s spec sts x link c₁ mids f₂ := by
  rw [decisionCsAt_parse s spec sts x link c₁ mids hj₁, decisionCsAt_parse s spec sts x link c₁ mids hj₂]
  simp only [decisionJ, checksJ, coinSubJ, he, hst, hp, hd, RecordText.check_of hj₁ hc₁, RecordText.check_of hj₂ hc₂]

/-- **The coin text is not read**: two decisions on the same session, link, first coin and round messages whose
records are in their canonical text and parse to values equal once their coins are erased, with the same statement,
proofs and draw, step alike. -/
theorem nextCs_coinBlind {x : Claims.Msg} {as₁ as₂ : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec₁ spec₂ : SessionSpec} {sts₁ sts₂ : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f₁ f₂ : Final} (h₁ : act false x as₁ cs = .decide s spec₁ sts₁ link c₁ mids f₁)
    (h₂ : act false x as₂ cs = .decide s spec₂ sts₂ link c₁ mids f₂) {j₁ j₂ : Json}
    (hj₁ : Json.parse f₁.record = .ok j₁) (hj₂ : Json.parse f₂.record = .ok j₂) (he : eraseCoins j₁ = eraseCoins j₂)
    (hst : f₁.stmt = f₂.stmt) (hp : f₁.proofs = f₂.proofs)
    (hd : FlockSoundness.Discharge.Exec.recordDraw f₁.record = FlockSoundness.Discharge.Exec.recordDraw f₂.record)
    (hc₁ : RecordText.printT j₁ = f₁.record) (hc₂ : RecordText.printT j₂ = f₂.record) :
    nextCs x as₁ cs = nextCs x as₂ cs := by
  simp only [nextCs, h₁, h₂]
  congr 1
  unfold decisionCs
  congr 1
  funext p
  obtain ⟨spec, sts⟩ := p
  exact decisionCsAt_coinBlind s spec sts x link c₁ mids hj₁ hj₂ he hst hp hd hc₁ hc₂

/-! ## Custody at the substituted value -/

section Custody

open FlockComplete.JsonRT (f128s_map array_mapM_ok_of)

theorem bind_ok {α β : Type} {x : Flock.V α} {f : α → Flock.V β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => cases h
  | ok a => exact ⟨a, rfl, h⟩

theorem fld_mapKey_eq (k : String) (g : Json → Json) (j : Json) :
    Record.fld (mapKey k g j) k = g <$> Record.fld j k := by
  unfold Record.fld
  rw [getObjVal?_mapKey_self]
  cases j.getObjVal? k <;> rfl

theorem fld_mapKey_self {k : String} {g : Json → Json} {j v : Json} (h : Record.fld (mapKey k g j) k = .ok v) :
    ∃ v₀, Record.fld j k = .ok v₀ ∧ v = g v₀ := by
  rw [fld_mapKey_eq] at h
  cases hj : Record.fld j k with
  | error e => rw [hj] at h; cases h
  | ok v₀ => rw [hj] at h; cases h; exact ⟨v₀, rfl, rfl⟩

theorem fld_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.fld (mapKey k g j) k' = Record.fld j k' := by
  unfold Record.fld
  rw [getObjVal?_mapKey_ne g j hk]

theorem arr_mapKey_const {k : String} {a b : Array Json} {j : Json}
    (h : Record.arr (mapKey k (fun _ => .arr a) j) k = .ok b) : b = a := by
  unfold Record.arr at h
  cases hf : Record.fld (mapKey k (fun _ => .arr a) j) k with
  | error e => rw [hf] at h; cases h
  | ok v =>
    obtain ⟨v₀, -, rfl⟩ := fld_mapKey_self hf
    rw [hf] at h
    cases h
    rfl

theorem arr_mapKey_mapArr (k : String) (g : Json → Json) (j : Json) :
    Record.arr (mapKey k (mapArr g) j) k = (·.map g) <$> Record.arr j k := by
  unfold Record.arr
  rw [fld_mapKey_eq]
  cases Record.fld j k with
  | error e => rfl
  | ok v => cases v <;> rfl

theorem arr_mapKey_mapArrIdx (k : String) (g : ℕ → Json → Json) (j : Json) :
    Record.arr (mapKey k (mapArrIdx g) j) k = (·.mapIdx g) <$> Record.arr j k := by
  unfold Record.arr
  rw [fld_mapKey_eq]
  cases Record.fld j k with
  | error e => rfl
  | ok v => cases v <;> rfl

theorem arr_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.arr (mapKey k g j) k' = Record.arr j k' := by
  unfold Record.arr
  rw [fld_mapKey_ne g j hk]

theorem nat_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    Record.nat (mapKey k g j) k' = Record.nat j k' := by
  unfold Record.nat
  rw [fld_mapKey_ne g j hk]

theorem getObjValAs_mapKey_ne {k k' : String} (g : Json → Json) (j : Json) (hk : k' ≠ k) :
    (mapKey k g j).getObjValAs? String k' = j.getObjValAs? String k' := by
  unfold Json.getObjValAs? Json.getObjValD
  rw [getObjVal?_mapKey_ne g j hk]

theorem list_mapM_ok_inv {α β : Type} (f : α → Flock.V β) :
    ∀ (l : List α) (ys : List β), l.mapM f = .ok ys →
      ∃ hl : ys.length = l.length, ∀ (i : ℕ) (h : i < l.length), f l[i] = .ok (ys[i]'(hl ▸ h))
  | [], ys, h => by
    simp only [List.mapM_nil] at h
    cases h
    exact ⟨rfl, fun i h => absurd h (by simp)⟩
  | a :: l, ys, h => by
    rw [List.mapM_cons] at h
    cases ha : f a with
    | error e => rw [ha] at h; cases h
    | ok b =>
      rw [ha] at h
      cases hl : l.mapM f with
      | error e => rw [hl] at h; cases h
      | ok bs =>
        rw [hl] at h
        cases h
        obtain ⟨hlen, hf⟩ := list_mapM_ok_inv f l bs hl
        refine ⟨by simp [hlen], fun i hi => ?_⟩
        cases i with
        | zero => exact ha
        | succ i => exact hf i (by simpa using hi)

theorem array_mapM_ok_inv {α β : Type} (f : α → Flock.V β) (a : Array α) (ys : Array β) (h : a.mapM f = .ok ys) :
    ∃ hl : ys.size = a.size, ∀ (i : ℕ) (h : i < a.size), f a[i] = .ok (ys[i]'(hl ▸ h)) := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : a.toList.mapM f with
  | error e => rw [hl] at h; cases h
  | ok bs =>
    rw [hl] at h
    cases h
    obtain ⟨hlen, hf⟩ := list_mapM_ok_inv f a.toList bs hl
    refine ⟨by simpa using hlen, fun i hi => ?_⟩
    have := hf i (by simpa using hi)
    simp only [Array.getElem_toList] at this
    exact this

theorem flatten_getElem {α : Type} :
    ∀ (L : List (List α)) (t : ℕ) (ht : t < L.flatten.length),
      ∃ (i : ℕ) (hi : i < L.length) (k : ℕ) (hk : k < L[i].length),
        L.flatten[t] = L[i][k] ∧ t = ((L.take i).map List.length).sum + k
  | [], t, ht => by simp at ht
  | s :: L, t, ht => by
    by_cases hs : t < s.length
    · refine ⟨0, by simp, t, hs, ?_, by simp⟩
      simp only [List.flatten_cons]
      rw [List.getElem_append_left hs]
      rfl
    · have ht' : t - s.length < L.flatten.length := by
        simp only [List.flatten_cons, List.length_append] at ht
        omega
      obtain ⟨i, hi, k, hk, he, hsum⟩ := flatten_getElem L (t - s.length) ht'
      refine ⟨i + 1, by simpa using hi, k, by simpa using hk, ?_, ?_⟩
      · simp only [List.flatten_cons]
        rw [List.getElem_append_right (by omega)]
        simpa using he
      · simp only [List.take_succ_cons, List.map_cons, List.sum_cons]
        omega

theorem takeWhile_ne_nodup {α : Type} [DecidableEq α] :
    ∀ (l : List α) (_ : l.Nodup) (i : ℕ) (hi : i < l.length), l.takeWhile (· ≠ l[i]) = l.take i
  | [], _, i, hi => by simp at hi
  | a :: l, hn, i, hi => by
    cases i with
    | zero => simp
    | succ i =>
      have hl : i < l.length := by simpa using hi
      have hne : a ≠ l[i] := by
        intro h
        exact (List.nodup_cons.1 hn).1 (h ▸ List.getElem_mem hl)
      simp only [List.getElem_cons_succ, List.takeWhile_cons, List.take_succ_cons]
      simp only [ne_eq, hne, not_false_eq_true, decide_true]
      rw [takeWhile_ne_nodup l (List.nodup_cons.1 hn).2 i hl]
      simp

/-- The rep-stream test is blind to the rounds. -/
theorem find_mapStream (g : String → ℕ → Json → Json) (recs : Array Json) (name : String) :
    (recs.map (mapStream g)).find? (fun s => (s.getObjValAs? String "stream").toOption == some name) =
      (recs.find? fun s => (s.getObjValAs? String "stream").toOption == some name).map (mapStream g) := by
  rw [Array.find?_map]
  congr 2
  funext s
  simp only [Function.comp, mapStream, getObjValAs_mapKey_ne _ _ (by decide : "stream" ≠ "rounds")]

theorem nameOf_of_find {recs : Array Json} {name : String} {s : Json}
    (h : recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some s) :
    nameOf s = name := by
  have := Array.find?_some h
  simp only [beq_iff_eq] at this
  simp only [nameOf, this, Option.getD_some]

/-- **A rep stream read at substituted rounds**: the stream the record's test finds, its rounds mapped. -/
theorem streamJ_map {g : String → ℕ → Json → Json} {recs : Array Json} {name : String} {rs : Array Round}
    (h : streamJ (recs.map (mapStream g)) name = .ok rs) :
    ∃ s rs₀, recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some s ∧
      Record.arr s "rounds" = .ok rs₀ ∧ (rs₀.mapIdx (g name)).mapM (fun r => roundJ r) = .ok rs := by
  unfold streamJ at h
  rw [find_mapStream] at h
  cases hf : recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) with
  | none => rw [hf] at h; cases h
  | some s =>
    rw [hf] at h
    obtain ⟨rs', hrs', h⟩ := bind_ok h
    simp only [mapStream] at hrs'
    rw [arr_mapKey_mapArrIdx, nameOf_of_find hf] at hrs'
    cases ha : Record.arr s "rounds" with
    | error e => rw [ha] at hrs'; cases hrs'
    | ok rs₀ =>
      rw [ha] at hrs'
      cases hrs'
      exact ⟨s, rs₀, rfl, ha, h⟩

/-- `roundJ`'s round count and coins are the record's `n` and `coins`. -/
theorem roundJ_ok {r : Json} {rd : Round} (h : roundJ r = .ok rd) :
    Record.nat r "n" = .ok rd.n ∧ (Record.arr r "coins" >>= Record.f128s) = .ok rd.coins := by
  unfold roundJ at h
  obtain ⟨n, hn, h⟩ := bind_ok h
  obtain ⟨cs, hcs, h⟩ := bind_ok h
  obtain ⟨coins, hcoins, h⟩ := bind_ok h
  have hc : (Record.arr r "coins" >>= Record.f128s) = .ok coins := by rw [hcs]; exact hcoins
  split at h
  · cases h
    exact ⟨hn, hc⟩
  · obtain ⟨_, -, h⟩ := bind_ok h
    obtain ⟨_, -, h⟩ := bind_ok h
    cases h
    exact ⟨hn, hc⟩

/-- **A round with its coins replaced reads them** (`f128s_map`). -/
theorem roundJ_coins (c : Coin) (r : Json) {rd : Round}
    (h : roundJ (mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r) = .ok rd) : rd.coins = c.vals rd.n := by
  obtain ⟨hn, hc⟩ := roundJ_ok h
  rw [nat_mapKey_ne _ _ (by decide)] at hn
  have hnr : nOf r = rd.n := by simp only [nOf, hn]
  rw [hnr] at hc
  obtain ⟨cs, hcs, hc⟩ := bind_ok hc
  obtain rfl := arr_mapKey_const hcs
  rw [f128s_map] at hc
  injection hc with hc
  exact hc.symm

/-- `roundsOf` at a stream `streamJ` reads: its rounds. -/
theorem roundsOf_of {j : Json} {recs : Array Json} {name : String} {s : Json} {rs : Array Json}
    (hr : Record.arr j "streams" = .ok recs)
    (hf : recs.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) = some s)
    (ha : Record.arr s "rounds" = .ok rs) : roundsOf j name = rs.size := by
  unfold roundsOf
  have hro : (Record.arr j "streams").toOption = some recs := by rw [hr]; rfl
  have hsa : (Record.arr s "rounds").toOption = some rs := by rw [ha]; rfl
  rw [hro, Option.bind_some, hf]
  dsimp only
  rw [hsa]
  rfl

/-- **Custody's points at the substituted value are the first coin's.** -/
theorem liveOfJ_points (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) {lv : Live}
    (h : liveOfJ spec (coinSubJ spec c₁ mids j) = .ok lv) : lv.points = pointsOf spec c₁ := by
  unfold liveOfJ at h
  obtain ⟨l, hl, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨ps, hps, h⟩ := bind_ok h
  obtain ⟨points, hpoints, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  cases h
  show points = _
  simp only [coinSubJ, fillCoins, mapRounds, setPoints] at hl
  rw [fld_mapKey_ne _ _ (by decide)] at hl
  obtain ⟨l₀, -, rfl⟩ := fld_mapKey_self hl
  obtain rfl := arr_mapKey_const hps
  have := array_mapM_ok_of (fun p => match p.getArr? with
    | .ok a => Record.f128s a
    | .error _ => .error "record: a link point is not a list") ((pointsOf spec c₁).map hexArr) (pointsOf spec c₁)
    (by simp) fun i h₁ h₂ => by
      simp only [Array.getElem_map, hexArr, Json.getArr?]
      exact f128s_map _
  rw [this] at hpoints
  cases hpoints
  rfl

/-- **Custody's round coins at the substituted value are the live coins**: at rep streams of distinct names, every
recorded round custody pairs with a round message carries the first `n` of that message's coin. -/
theorem liveOfJ_coins (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) {lv : Live}
    (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup) (h : liveOfJ spec (coinSubJ spec c₁ mids j) = .ok lv) :
    ∀ (t : ℕ) (h₁ : t < (lv.streams.toList.flatMap fun s : Array Round => s.toList).length) (h₂ : t < mids.length),
      (lv.streams.toList.flatMap fun s : Array Round => s.toList)[t].coins =
        mids[t].2.vals (lv.streams.toList.flatMap fun s : Array Round => s.toList)[t].n := by
  unfold liveOfJ at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨recs, hrecs, h⟩ := bind_ok h
  obtain ⟨streams, hstreams, h⟩ := bind_ok h
  cases h
  simp only [coinSubJ, fillCoins, mapRounds, setPoints] at hrecs
  rw [arr_mapKey_mapArr, arr_mapKey_ne _ _ (by decide : "streams" ≠ "link")] at hrecs
  cases hE : Record.arr (eraseCoins j) "streams" with
  | error e => rw [hE] at hrecs; cases hrecs
  | ok aE =>
  rw [hE] at hrecs
  injection hrecs with hrecs
  subst hrecs
  obtain ⟨hsz, hf⟩ := array_mapM_ok_inv _ _ _ hstreams
  have hS : ∀ (i : ℕ) (hi : i < spec.repStreams.size), ∃ rs₁ : Array Json,
      roundsOf (eraseCoins j) spec.repStreams[i].2.2 = rs₁.size ∧
      (rs₁.mapIdx fun k r => match mids[streamOffset spec (eraseCoins j) spec.repStreams[i].2.2 + k]? with
          | some (_, c) => mapKey "coins" (fun _ => hexArr (c.vals (nOf r))) r
          | none => r).mapM (fun r => roundJ r) = .ok (streams[i]'(hsz ▸ hi)) := by
    intro i hi
    obtain ⟨s₁, rs₁, hfind, ha, hm⟩ := streamJ_map (hf i hi)
    exact ⟨rs₁, roundsOf_of hE hfind ha, hm⟩
  have hsize : ∀ (i : ℕ) (hi : i < spec.repStreams.size),
      roundsOf (eraseCoins j) spec.repStreams[i].2.2 = (streams[i]'(hsz ▸ hi)).size := by
    intro i hi
    obtain ⟨rs₁, hr, hm⟩ := hS i hi
    obtain ⟨hl, -⟩ := array_mapM_ok_inv _ _ _ hm
    rw [hr, hl, Array.size_mapIdx]
  intro t h₁ h₂
  have hfl : (streams.toList.flatMap fun s : Array Round => s.toList) =
      (streams.toList.map fun s : Array Round => s.toList).flatten := rfl
  simp only [hfl] at h₁ ⊢
  obtain ⟨i, hi, k, hk, he, ht⟩ := flatten_getElem _ t h₁
  simp only [he]
  have hi' : i < spec.repStreams.size := by simpa [hsz] using hi
  have hk' : k < (streams[i]'(hsz ▸ hi')).size := by simpa using hk
  have hget : (streams.toList.map fun s : Array Round => s.toList)[i][k] = (streams[i]'(hsz ▸ hi'))[k] := by simp
  rw [hget]
  have hoff : streamOffset spec (eraseCoins j) spec.repStreams[i].2.2 + k = t := by
    rw [ht]
    congr 1
    have hnames : spec.repStreams[i].2.2 = (spec.repStreams.toList.map (·.2.2))[i]'(by simpa using hi') := by simp
    unfold streamOffset
    rw [hnames, takeWhile_ne_nodup _ hnd i (by simpa using hi'), List.map_take, List.map_take]
    congr 2
    apply List.ext_getElem (by simp [hsz])
    intro i' h₁' h₂'
    have hi'' : i' < spec.repStreams.size := by simpa using h₁'
    simp only [List.getElem_map, Array.getElem_toList]
    exact hsize i' hi''
  obtain ⟨rs₁, -, hm⟩ := hS i hi'
  obtain ⟨hl, hr⟩ := array_mapM_ok_inv _ _ _ hm
  have hk₁ : k < rs₁.size := by
    have := hl ▸ hk'
    simpa using this
  have hrk := hr k (by simpa using hk₁)
  simp only [Array.getElem_mapIdx, hoff, List.getElem?_eq_getElem h₂] at hrk
  exact roundJ_coins _ _ hrk

theorem all_congr_mem {α : Type} {p q : α → Bool} : ∀ {l : List α}, (∀ a ∈ l, p a = q a) → l.all p = l.all q
  | [], _ => rfl
  | a :: l, h => by
    simp only [List.all_cons, h a (List.mem_cons_self ..),
      all_congr_mem (l := l) fun b hb => h b (List.mem_cons_of_mem _ hb)]

/-- Rounds whose coins are their paired messages' pass custody's round comparison exactly when their bytes do. -/
theorem all_zip_coins {rounds : List Round} {mids : List (Claims.Msg × Coin)}
    (hco : ∀ (t : ℕ) (h₁ : t < rounds.length) (h₂ : t < mids.length), rounds[t].coins = mids[t].2.vals rounds[t].n)
    (p : Round × Claims.Msg × Coin → Bool)
    (hp : ∀ r a c, p (r, a, c) = (r.bytes == some ⟨a.toArray⟩ && r.coins == c.vals r.n)) :
    (rounds.zip mids).all p = (rounds.zip mids).all fun q => q.1.bytes == some ⟨q.2.1.toArray⟩ := by
  apply all_congr_mem
  intro q hq
  obtain ⟨t, ht, rfl⟩ := List.mem_iff_getElem.1 hq
  have ht₁ : t < rounds.length := by simp only [List.length_zip] at ht; omega
  have ht₂ : t < mids.length := by simp only [List.length_zip] at ht; omega
  rw [List.getElem_zip]
  show p (rounds[t], mids[t].1, mids[t].2) = _
  rw [hp, hco t ht₁ ht₂, beq_self_eq_true, Bool.and_true]

theorem check_true (why : String) : Record.check true why = pure () := rfl

/-- **Custody holds by construction at the substituted value**: at rep streams of distinct names, custody of what
`liveOfJ` reads there is its byte comparisons alone, the points and every paired round's coins being the live coins'. -/
theorem custody_coinSubJ (spec : SessionSpec) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (j : Json) {lv : Live} (hnd : (spec.repStreams.toList.map (·.2.2)).Nodup)
    (h : liveOfJ spec (coinSubJ spec c₁ mids j) = .ok lv) :
    custody spec lv link c₁ mids = (do
      Record.check (lv.rootB.toList == link.1) "live: the record's root_B is not the link message's"
      Record.check (lv.publics.toList == link.2) "live: the record's public words are not the link message's"
      Record.check ((lv.streams.toList.flatMap fun s : Array Round => s.toList).length == mids.length)
        "live: the record's rounds are not the round messages"
      Record.check (((lv.streams.toList.flatMap fun s : Array Round => s.toList).zip mids).all
          fun q => q.1.bytes == some ⟨q.2.1.toArray⟩)
        "live: a recorded round is not its message and coin" : Flock.V Unit) := by
  have hp : (lv.points == pointsOf spec c₁) = true := by rw [liveOfJ_points spec c₁ mids j h]; exact beq_self_eq_true _
  unfold custody
  dsimp only
  rw [hp, all_zip_coins (liveOfJ_coins spec c₁ mids j hnd h), check_true, pure_bind]
  intro r a c
  rfl

end Custody

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin nextCs nextCs_coin)
open Claims.Recursion.VStarB

/-- **The composition at the element-level verifier `nextCs`**: `bdDeployed` with `nextCs` as its steps, split by
`deploySplit` (a coin answered as the plain verifier answers it, `nextCs_coin`), at the default inventory. Where the
verdict's split `LS` gives its checks to the statement's parts (`hR`), the checks no part holds hold off any witnesses
(`hrest`, less the setup and the tags, which the statement fixes), and each part's circuit accepting implies its share
(`hlow`), `DecompB` holds at what each part's circuit accepts. -/
theorem decompB_nextCs (H : Msg → Msg)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    {R : Type} (LS : LastSplit H nextCs R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s, ownR x r = some s →
        s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel nextCs colsE wE join rowIn final w)
          (deploySplit H rel nextCs colsE wE join rowIn final w (nextCs_coin H) LS)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy ownR) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel nextCs colsE wE join rowIn final w)
            (deploySplit H rel nextCs colsE wE join rowIn final w (nextCs_coin H) LS)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel nextCs colsE wE join rowIn final w)
        (deploySplit H rel nextCs colsE wE join rowIn final w (nextCs_coin H) LS)
        (ownHolders (recBound rel' b L IV) L (ownDeploy ownR)) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel nextCs colsE wE join rowIn final w).IVStar L shape
      (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_partAccepts_of_parts H (recBound rel' b L IV) DO _ L shape shows cmt
    (deploySplit H rel nextCs colsE wE join rowIn final w (nextCs_coin H) LS) _ _ hlow
    (ownedBy_deploy H rel nextCs colsE wE join rowIn final w (nextCs_coin H) LS ownR _ L shape hR)
    (stmtFixes_deploy H rel nextCs colsE wE hj rowIn final w (nextCs_coin H) LS ownR rel' b L IV shape hrest)

end RecDeploy
