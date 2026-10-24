import Proofs.Flock.Claims.Recursion.PartsCompose
import Proofs.Flock.Claims.Complete.Prover.Print

/-!
# The deployed plain verifier with the record's coins read from the live coins (`nextCs`)

The plain verifier's custody (`FlockDeploy.custody`) compares the coins the record's text carries with the live coins
`cs`, and its verdict (`verdictM`) reads the coins off that text. `nextCs` is the plain verifier's step at the prover's
messages with the last message's record substituted (`lastCs`): its coins replaced by `cs`'s, so that what the verdict
reads is the live coins whatever the record's text said.

* `coinSub`: the record's text parsed, its coins erased (`eraseCoins`: the link's points and every round's coins set to
  `null`), then filled from the first coin and the round coins (`fillCoins`: the points `pointsOf spec c₁`, round `k` of
  rep stream `name` the first `n` of the coin at its place in `spec.repStreams` order), and printed (`printJ`). A text
  that is not JSON is left as it is.
* `lastCs sub x as cs`: at a decision, every message but the last, then the last with its record `sub spec c₁ mids`'d;
  else `as`. `nextCs sub = next false x (lastCs sub x as cs) cs`.

Three facts:

* **Plain acceptance carries over** (`nextCs_canonical`): at a decision whose last message is `f.encode` and whose record
  the substitution fixes, `nextCs` is the plain `next`.
* **Soundness transfers** (`nextCs_eq`, `lastCs_length`): `nextCs` is the plain `next` at the substituted messages, as
  many as `as`, so `nextCs` accepting is plain `next` accepting there.
* **The coin text is not read** (`nextCs_congr`, `coinSub_congr`, `nextCs_coinBlind`): two decisions on the same link,
  coins and round messages whose last messages differ only in the record's coins step alike.

A coin is answered as the plain verifier answers it (`nextCs_coin`), so `bdDeployed` at `nextCs` splits as at `next`
(`RecDeploy.decompB_nextCs`), with custody's coin comparison satisfied by construction wherever the substituted record
reads back its coins (`printJ` and `Json.parse`, `F128.toHex` and `f128OfHex` round trips): that and the route at this
element-level verifier are the open pieces.
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-! ## The coin substitution -/

/-- Each round of each stream of the record, `g name k round` at stream `name`'s round `k`. -/
def mapRounds (g : String → ℕ → Json → Json) (j : Json) : Json :=
  match j.getObjValD "streams" with
  | .arr ss => j.setObjVal! "streams" (.arr (ss.map fun s =>
      let name := (s.getObjValAs? String "stream").toOption.getD ""
      match s.getObjValD "rounds" with
      | .arr rs => s.setObjVal! "rounds" (.arr (rs.mapIdx fun k r => g name k r))
      | _ => s))
  | _ => j

/-- The record with the link's points set to `v`. -/
def setPoints (v : Json) (j : Json) : Json := j.setObjVal! "link" ((j.getObjValD "link").setObjVal! "points" v)

/-- **The record with its coins erased**: the link's points and every round's coins `null`. -/
def eraseCoins (j : Json) : Json := mapRounds (fun _ _ r => r.setObjVal! "coins" .null) (setPoints .null j)

/-- Field elements as the record writes them, hex strings. -/
def hexArr (a : Array F128) : Json := .arr (a.map fun e => .str e.toHex)

/-- Stream `name`'s recorded rounds: the first stream of that name's, `0` if none. -/
def roundsOf (j : Json) (name : String) : ℕ :=
  match j.getObjValD "streams" with
  | .arr ss =>
    match ss.find? (fun s => (s.getObjValAs? String "stream").toOption == some name) with
    | some s =>
      match s.getObjValD "rounds" with
      | .arr rs => rs.size
      | _ => 0
    | none => 0
  | _ => 0

/-- Where stream `name`'s rounds start among the round coins: the rounds of the rep streams before it. -/
def streamOffset (spec : SessionSpec) (j : Json) (name : String) : ℕ :=
  (((spec.repStreams.toList.map (·.2.2)).takeWhile (· ≠ name)).map (roundsOf j)).sum

/-- **The record's coins filled from the live coins**: the link's points the first coin's (`pointsOf`), round `k` of
stream `name` the first `n` of its coin at `streamOffset + k`. -/
def fillCoins (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (j : Json) : Json :=
  mapRounds (fun name k r =>
      match mids[streamOffset spec j name + k]? with
      | some (_, c) => r.setObjVal! "coins" (hexArr (c.vals ((r.getObjValAs? Nat "n").toOption.getD 0)))
      | none => r)
    (setPoints (.arr ((pointsOf spec c₁).map hexArr)) j)

/-- **The record's text with its coins the live coins**: parsed, erased, filled and printed; left as it is if it is not
JSON. -/
def coinSub (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (r : String) : String :=
  match Json.parse r with
  | .ok j => FlockComplete.printJ (fillCoins spec c₁ mids (eraseCoins j))
  | .error _ => r

/-! ## The step -/

/-- **The prover's messages with the record substituted**: at a decision, the last message's record `sub spec c₁ mids`'d;
else the messages. -/
def lastCs (sub : SessionSpec → Coin → List (Claims.Msg × Coin) → String → String) (x : Claims.Msg)
    (as : List Claims.Msg) (cs : List Coin) : List Claims.Msg :=
  match act false x as cs with
  | .decide _ spec _ _ c₁ mids f => as.dropLast ++ [Final.encode { f with record := sub spec c₁ mids f.record }]
  | _ => as

/-- **The plain verifier's step with the record's coins `cs`'s.** -/
def nextCs (sub : SessionSpec → Coin → List (Claims.Msg × Coin) → String → String) (x : Claims.Msg)
    (as : List Claims.Msg) (cs : List Coin) : Claims.OC Claims.Out :=
  next false x (lastCs sub x as cs) cs

variable (sub : SessionSpec → Coin → List (Claims.Msg × Coin) → String → String)

/-- **`nextCs` is the plain `next` at the substituted messages**, so its acceptance is the plain verifier's there. -/
theorem nextCs_eq (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    nextCs sub x as cs = next false x (lastCs sub x as cs) cs := rfl

theorem lastCs_of_coin {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {c : Coin}
    (h : act false x as cs = .coin c) : lastCs sub x as cs = as := by
  simp only [lastCs, h]

theorem lastCs_of_decide {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f) :
    lastCs sub x as cs = as.dropLast ++ [Final.encode { f with record := sub spec c₁ mids f.record }] := by
  simp only [lastCs, h]

theorem act_nil_ne_decide {x : Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec} {sts : Array Stmt}
    {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x [] cs = .decide s spec sts link c₁ mids f) : False := by
  unfold act at h
  split at h
  · cases h
  · simp at h

/-- The substitution keeps the number of messages. -/
theorem lastCs_length (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    (lastCs sub x as cs).length = as.length := by
  unfold lastCs
  split
  · rename_i h
    cases as with
    | nil => exact (act_nil_ne_decide h).elim
    | cons a as => simp
  · rfl

/-- **A coin is answered as the plain verifier answers it**, with no query (`deploySplit`'s `hcoin`). -/
theorem nextCs_coin (H : Claims.Msg → Claims.Msg) : ∀ (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin)
    (c : Coin), act false x as cs = .coin c → ((nextCs sub x as cs).eval H).1 = .msg c.bytes := by
  intro x as cs c h
  rw [nextCs_eq, lastCs_of_coin sub h, next_coin H h]

/-- **At a record the substitution fixes, the messages are kept**: at a decision whose last message is `f.encode`. -/
theorem lastCs_canonical {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f) (hl : as.getLast? = some f.encode)
    (hk : sub spec c₁ mids f.record = f.record) : lastCs sub x as cs = as := by
  rw [lastCs_of_decide sub h, hk]
  obtain ⟨ys, rfl⟩ := List.getLast?_eq_some_iff.1 hl
  rw [List.dropLast_concat]
  all_goals rfl

/-- **Plain acceptance carries over**: at a decision whose last message is `f.encode` and whose record the
substitution fixes (the record's coins already `cs`'s, as the plain verifier's custody asks), `nextCs` is the plain
`next`. -/
theorem nextCs_canonical {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)} {f : Final}
    (h : act false x as cs = .decide s spec sts link c₁ mids f) (hl : as.getLast? = some f.encode)
    (hk : sub spec c₁ mids f.record = f.record) : nextCs sub x as cs = next false x as cs := by
  rw [nextCs_eq, lastCs_canonical sub h hl hk]

/-- **`nextCs` reads the record only through `sub`**: two decisions on the same link, first coin and round messages,
whose messages before the last agree and whose last messages agree but for records `sub` takes to one text, step
alike. -/
theorem nextCs_congr {x : Claims.Msg} {as₁ as₂ : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f₁ f₂ : Final} (h₁ : act false x as₁ cs = .decide s spec sts link c₁ mids f₁)
    (h₂ : act false x as₂ cs = .decide s spec sts link c₁ mids f₂) (hd : as₁.dropLast = as₂.dropLast)
    (hst : f₁.stmt = f₂.stmt) (hp : f₁.proofs = f₂.proofs)
    (hsub : sub spec c₁ mids f₁.record = sub spec c₁ mids f₂.record) :
    nextCs sub x as₁ cs = nextCs sub x as₂ cs := by
  rw [nextCs_eq, nextCs_eq, lastCs_of_decide sub h₁, lastCs_of_decide sub h₂, hd, hsub, hst, hp]

/-- **`coinSub` does not read the coin text**: two JSON records equal once their coins are erased substitute to one
text. -/
theorem coinSub_congr (spec : SessionSpec) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {r₁ r₂ : String}
    {j₁ j₂ : Json} (h₁ : Json.parse r₁ = .ok j₁) (h₂ : Json.parse r₂ = .ok j₂) (he : eraseCoins j₁ = eraseCoins j₂) :
    coinSub spec c₁ mids r₁ = coinSub spec c₁ mids r₂ := by
  simp only [coinSub, h₁, h₂, he]

/-- **The coin text is not read**, at `coinSub`: two decisions as in `nextCs_congr` whose records are JSON equal once
their coins are erased step alike. -/
theorem nextCs_coinBlind {x : Claims.Msg} {as₁ as₂ : List Claims.Msg} {cs : List Coin} {s : Ins} {spec : SessionSpec}
    {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin} {mids : List (Claims.Msg × Coin)}
    {f₁ f₂ : Final} (h₁ : act false x as₁ cs = .decide s spec sts link c₁ mids f₁)
    (h₂ : act false x as₂ cs = .decide s spec sts link c₁ mids f₂) (hd : as₁.dropLast = as₂.dropLast)
    (hst : f₁.stmt = f₂.stmt) (hp : f₁.proofs = f₂.proofs) {j₁ j₂ : Json} (hj₁ : Json.parse f₁.record = .ok j₁)
    (hj₂ : Json.parse f₂.record = .ok j₂) (he : eraseCoins j₁ = eraseCoins j₂) :
    nextCs coinSub x as₁ cs = nextCs coinSub x as₂ cs :=
  nextCs_congr coinSub h₁ h₂ hd hst hp (coinSub_congr spec c₁ mids hj₁ hj₂ he)

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin nextCs nextCs_coin)
open Claims.Recursion.VStarB

/-- **The composition at the element-level verifier `nextCs`**: `bdDeployed` with `nextCs sub` as its steps, split by
`deploySplit` (a coin answered as the plain verifier answers it, `nextCs_coin`), at the default inventory. Where the
verdict's split `LS` gives its checks to the statement's parts (`hR`), the checks no part holds hold off any witnesses
(`hrest`, less the setup and the tags, which the statement fixes), and each part's circuit accepting implies its share
(`hlow`), `DecompB` holds at what each part's circuit accepts. -/
theorem decompB_nextCs (H : Msg → Msg) (sub : SessionSpec → Coin → List (Msg × Coin) → String → String)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    {R : Type} (LS : LastSplit H (nextCs sub) R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ LS.checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s, ownR x r = some s →
        s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.Stmt), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextCs sub) colsE wE join rowIn final w)
          (deploySplit H rel (nextCs sub) colsE wE join rowIn final w (nextCs_coin sub H) LS)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy ownR) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextCs sub) colsE wE join rowIn final w)
            (deploySplit H rel (nextCs sub) colsE wE join rowIn final w (nextCs_coin sub H) LS)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextCs sub) colsE wE join rowIn final w)
        (deploySplit H rel (nextCs sub) colsE wE join rowIn final w (nextCs_coin sub H) LS)
        (ownHolders (recBound rel' b L IV) L (ownDeploy ownR)) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextCs sub) colsE wE join rowIn final w).IVStar L shape
      (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_partAccepts_of_parts H (recBound rel' b L IV) DO _ L shape shows cmt
    (deploySplit H rel (nextCs sub) colsE wE join rowIn final w (nextCs_coin sub H) LS) _ _ hlow
    (ownedBy_deploy H rel (nextCs sub) colsE wE join rowIn final w (nextCs_coin sub H) LS ownR _ L shape hR)
    (stmtFixes_deploy H rel (nextCs sub) colsE wE hj rowIn final w (nextCs_coin sub H) LS ownR rel' b L IV shape hrest)

end RecDeploy
