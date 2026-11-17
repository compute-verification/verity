module

public import Proofs.Flock.Claims.Steps.NextCs
public import Proofs.Flock.Claims.Recursion.PartsCompose
import all Lean.Data.Json.Basic
import all Lean.Data.Json.FromToJson.Basic
import all Init.Data.Array.Basic
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Tactic.ReduceModChar
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.Claims.Complete.Json.Decode

/-!
# The deployed plain verifier with the record's coins read from the live coins (`nextCs`)

The plain verifier's custody (`FlockDeploy.custody`) compares the coins the record carries with the live coins `cs`,
and its verdict (`verdictM`) reads the coins off the record. `nextCs` is the plain verifier's step with the record's
coins substituted after parsing: at a decision whose record parses to `j`, it runs the verdict (`verifyJWithM`) and
custody's reading (`liveOfJ`) at `coinSubJ spec c₁ mids j`, `j` with its coins replaced by the live coins, while the
statement and sample checks and S0 (`RecordText.check`) read the record's text as the plain verifier does. A record that
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
  the same statement, proofs and sample, step alike.
* **Custody holds by construction** (`liveOfJ_points`, `liveOfJ_coins`, `custody_coinSubJ`): at the substituted value
  custody's points are the first coin's and, at rep streams of distinct names, every recorded round it pairs with a
  round message carries the first `n` of that message's coin (`f128s_map`, the hex round trip), so custody is its
  byte comparisons alone. No JSON premise is taken.

A coin is answered as the plain verifier answers it (`nextCs_coin`), so `bdDeployed` at `nextCs` splits as at `next`
(`RecDeploy.decompB_nextCs`).
-/

@[expose] public section

namespace FlockDeploy

open Flock

open Lean (Json)

/-! ## Reading the record at a parsed value -/

theorem liveOf_of_parse (spec : SessionSpec) {record : String} {j : Json} (h : Flock.JsonParse.parse record = .ok j) :
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
    (proofs : Array ByteArray) (h : Flock.JsonParse.parse record = .ok j) :
    verdictM zk hs spec tabs record proofs = verdictJM zk hs spec tabs j proofs := by
  unfold verdictM verdictJM
  rw [verifyWithM_of_parse hs spec tabs proofs h, Zk.verifyWithM_of_parse hs spec tabs proofs h]

/-- `checks` with custody reading the parsed value `j`; the statement and sample checks and S0 read the record's text. -/
def checksJ (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray)
    (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) (j : Json) : Flock.V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (sampleEq (FlockSoundness.Discharge.Exec.recordDraw f.record) s.read.draw)
    "live: the record's draw is not the statement's"
  custody spec (← liveOfJ spec j) link c₁ mids
  s.read.check
  (if s.read.draw.isNone then ProvedScope.sampleDesign zk s.read.tags false none else pure ())
  RecordText.check f.record

theorem checks_of_parse (zk : Bool) (s : Ins) (spec : SessionSpec) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (h : Flock.JsonParse.parse f.record = .ok j) : checks zk s spec x link c₁ mids f = checksJ zk s spec x link c₁ mids f j := by
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
    (h : Flock.JsonParse.parse f.record = .ok j) :
    decisionAt zk s spec sts x link c₁ mids f = decisionJ zk s spec sts x link c₁ mids f j := by
  unfold decisionAt decisionJ
  rw [verdictM_of_parse zk _ spec _ _ h, checks_of_parse zk s spec x link c₁ mids h]

/-! ## Maps over a parsed value -/

/-! ## The coin substitution -/

/-! ## The step -/

/-- The decision at a session with the record's coins substituted after parsing; a record that is not JSON is
decided as the plain verifier decides it. -/
def decisionCsAt (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : Final) :
    Claims.OCV Unit :=
  match Flock.JsonParse.parse f.record with
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
    (hj : Flock.JsonParse.parse f.record = .ok j) :
    decisionCsAt s spec sts x link c₁ mids f = decisionJ false s spec sts x link c₁ mids f (coinSubJ spec c₁ mids j) := by
  simp only [decisionCsAt, hj]

/-- **Plain acceptance carries over** at a session: at a record that parses to a value the substitution fixes, the
decision is the plain one. -/
theorem decisionCsAt_canonical (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (hj : Flock.JsonParse.parse f.record = .ok j) (hk : coinSubJ spec c₁ mids j = j) :
    decisionCsAt s spec sts x link c₁ mids f = decisionAt false s spec sts x link c₁ mids f := by
  rw [decisionCsAt_parse s spec sts x link c₁ mids hj, hk, decisionAt_of_parse false s spec sts x link c₁ mids hj]

/-- **Plain acceptance carries over** at the decision: at a record that parses to a value the substitution fixes at
every session, `decisionCs` is the plain `decision`. -/
theorem decisionCs_canonical (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json} (hj : Flock.JsonParse.parse f.record = .ok j)
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
    {j : Json} (h : act false x as cs = .decide s spec sts link c₁ mids f) (hj : Flock.JsonParse.parse f.record = .ok j)
    (hk : ∀ spec', coinSubJ spec' c₁ mids j = j) : nextCs x as cs = next false x as cs := by
  simp only [nextCs, next, h, decisionCs_canonical s x link c₁ mids hj hk]

/-- **The coin text is not read** at a session: two records in their canonical text that parse to values equal once
their coins are erased, with the same statement, proofs and sample, are decided alike. -/
theorem decisionCsAt_coinBlind (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f₁ f₂ : Final}
    {j₁ j₂ : Json} (hj₁ : Flock.JsonParse.parse f₁.record = .ok j₁) (hj₂ : Flock.JsonParse.parse f₂.record = .ok j₂)
    (he : eraseCoins j₁ = eraseCoins j₂) (hst : f₁.stmt = f₂.stmt) (hp : f₁.proofs = f₂.proofs)
    (hd : FlockSoundness.Discharge.Exec.recordDraw f₁.record = FlockSoundness.Discharge.Exec.recordDraw f₂.record)
    (hc₁ : RecordText.printT j₁ = f₁.record) (hc₂ : RecordText.printT j₂ = f₂.record) :
    decisionCsAt s spec sts x link c₁ mids f₁ = decisionCsAt s spec sts x link c₁ mids f₂ := by
  rw [decisionCsAt_parse s spec sts x link c₁ mids hj₁, decisionCsAt_parse s spec sts x link c₁ mids hj₂]
  simp only [decisionJ, checksJ, coinSubJ, he, hst, hp, hd, RecordText.check_of hj₁ hc₁, RecordText.check_of hj₂ hc₂]

/-- **The coin text is not read**: two decisions on the same session, link, first coin and round messages whose
records are in their canonical text and parse to values equal once their coins are erased, with the same statement,
proofs and sample, step alike. -/
theorem nextCs_coinBlind {x : Claims.Msg} {as₁ as₂ : List Claims.Msg} {cs : List Coin} {s : Ins}
    {spec₁ spec₂ : SessionSpec} {sts₁ sts₂ : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f₁ f₂ : Final} (h₁ : act false x as₁ cs = .decide s spec₁ sts₁ link c₁ mids f₁)
    (h₂ : act false x as₂ cs = .decide s spec₂ sts₂ link c₁ mids f₂) {j₁ j₂ : Json}
    (hj₁ : Flock.JsonParse.parse f₁.record = .ok j₁) (hj₂ : Flock.JsonParse.parse f₂.record = .ok j₂) (he : eraseCoins j₁ = eraseCoins j₂)
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
