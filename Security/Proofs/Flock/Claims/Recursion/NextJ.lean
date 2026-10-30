import Proofs.Flock.Claims.Recursion.NextCs

/-!
# `IVStar`'s last step at the record's parsed value (`nextJ`)

`nextCs` still parses the record's text (`Json.parse`, opaque), so a split of its last step has a check, the parse's
success, that no V\* part computes and the statement does not fix: the record is a prover message, a witness under a
hiding leaf, and `StmtFixes` asks a check to hold at the readers' messages of every witnesses. `nextJ` is the
element-level step whose last message carries the record's parsed value: the layout's reader `rd` (the code side's, as
`final` is) reads it as the statement, the parsed value and the proof files (`FinalJ`), and the decision runs the
verdict and custody's reading at the coin-substituted value (`decisionAtJ`), the draw check at the parsed value
(`drawOfJ`). There is no `Json.parse` in it, so parse success is no check of `IVStar`.

* **The step** (`actJ`, `nextJ`): a coin is answered as the deployed plain verifier answers it (`nextJ_coin`, so
  `deploySplit` applies); at a decision it is `decisionJF` (`nextJ_decide`).
* **The parsed value's decision is `nextCs`'s** (`decisionAtJ_eq`, `decisionJF_eq`): at the same session, link, first
  coin and round messages, `nextJ`'s decision on a parsed value `j` is `nextCs`'s on the record that parses to `j` in
  its canonical text (`printT j`, which S0 passes: `RecordText.check_of`), so a route through the plain step at every
  parsed value (`verifyJWithM`) covers both.
* **The reader parts' checks** (`ReadOk`, `LastSplitJ`, `LastSplitJ.toLast`, `ownJ`): a split of `nextJ`'s last step
  whose soundness may assume the reading, the last message read by `rd` and the record value's fields custody reads
  present with their shapes (`liveOfJ` at the substituted value `.ok`). The reading is one check, `read`, given to the
  first part holding the last round whole, the reader part whose message carries the record value (`ownJ`); the
  verdict's checks keep their owners (`ownJ_lt`). `RecDeploy.decompB_nextJ` is the composition at `nextJ`.

**Proof-file decoding.** The proof files (`FinalJ.proofs`) are bytes the verdict decodes inside `verifyRepWith`. That
decoding stays among the verdict's checks here (`LastSplitJ`'s `checks`), and no part's lowering fact computes it, so a
`LastSplitJ` whose checks are owned by parts that don't decode the proofs has an unsatisfiable `hlow`. The route is
`NextS`: its last message carries the decoded proofs and its decision decodes no proof file (`RecDeploy.decompB_nextS`).
-/

namespace FlockDeploy

open Flock
open Lean (Json)

/-- **The prover's last message at the parsed value**: the statement, the record's parsed value and the proof files. -/
structure FinalJ where
  stmt : Claims.Msg
  j : Json
  proofs : Array ByteArray

/-- The record's draw at its parsed value, rebuilt (`recordDraw` after the parse). -/
def drawOfJ (j : Json) : Option Json :=
  match j.getObjVal? "unit_draw" with
  | .ok Json.null => none
  | .ok d => some (Flock.norm d)
  | .error _ => none

theorem recordDraw_of_parse {record : String} {j : Json} (h : Json.parse record = .ok j) :
    FlockSoundness.Discharge.Exec.recordDraw record = drawOfJ j := by
  unfold FlockSoundness.Discharge.Exec.recordDraw
  rw [h]
  rfl

/-- The checks after the verdict at the parsed value: the statement, the draw at `f.j`, custody at `j'`. -/
def checksAtJ (s : Ins) (spec : SessionSpec) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalJ) (j' : Json) : Flock.V Unit := do
  Record.check (f.stmt == x) "live: the last message's statement is not the statement"
  Record.check (drawEq (drawOfJ f.j) s.read.draw) "live: the record's draw is not the statement's"
  custody spec (← liveOfJ spec j') link c₁ mids
  s.read.check
  if s.read.draw.isNone then ProvedScope.drawLaw false s.read.tags false none

/-- **The decision at a session at the parsed value**: the plain verdict and custody's reading at the coin-substituted
value. -/
def decisionAtJ (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) (f : FinalJ) :
    Claims.OCV Unit := do
  verifyJWithM Claims.Hashes.oc spec (tabsOf sts) (coinSubJ spec c₁ mids f.j) f.proofs
  monadLift (checksAtJ s spec x link c₁ mids f (coinSubJ spec c₁ mids f.j))

/-- The decision at the parsed value: the statement's setup over `OC`, then `decisionAtJ`. -/
def decisionJF (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) (f : FinalJ) : Claims.OCV Unit := do
  let (spec, sts) ← setupM Claims.Hashes.oc s.read
  decisionAtJ s spec sts x link c₁ mids f

/-- What `nextJ` does at the prover's messages. -/
inductive ActJ where
  | rej
  | coin (c : Coin)
  | decide (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (link : List ByteArray × List ByteArray) (c₁ : Coin)
      (mids : List (Claims.Msg × Coin)) (f : FinalJ)

/-- **`nextJ`'s step**: a coin where the deployed plain verifier answers one; else, at a statement that sets up, decide
where `rd` reads the last message and the link message reads; anything else, refuse. -/
def actJ (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) : ActJ :=
  match act false x as cs with
  | .coin c => .coin c
  | _ =>
    match setupOf false x with
    | none => .rej
    | some (s, spec, sts) =>
      match as.getLast?, as, cs with
      | some a, a₁ :: rest, c₁ :: crest =>
        match rd a, linkOf a₁ with
        | some f, some link => .decide s spec sts link c₁ ((rest.dropLast.map List.tail).zip crest.dropLast) f
        | _, _ => .rej
      | _, _, _ => .rej

/-- **The element-level step at the record's parsed value.** -/
def nextJ (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (as : List Claims.Msg) (cs : List Coin) :
    Claims.OC Claims.Out :=
  match actJ rd x as cs with
  | .rej => .ret .rej
  | .coin c => .ret (.msg c.bytes)
  | .decide s _ _ link c₁ mids f => outOf (decisionJF s x link c₁ mids f)

/-- **A coin is answered as the plain verifier answers it** (`deploySplit`'s `hcoin`). -/
theorem nextJ_coin (rd : Claims.Msg → Option FinalJ) (H : Claims.Msg → Claims.Msg) : ∀ (x : Claims.Msg)
    (as : List Claims.Msg) (cs : List Coin) (c : Coin), act false x as cs = .coin c →
    ((nextJ rd x as cs).eval H).1 = .msg c.bytes := by
  intro x as cs c h
  simp only [nextJ, actJ, h, Claims.OC.eval]

theorem nextJ_decide {rd : Claims.Msg → Option FinalJ} {x : Claims.Msg} {as : List Claims.Msg} {cs : List Coin}
    {s : Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalJ} (h : actJ rd x as cs = .decide s spec sts link c₁ mids f) :
    nextJ rd x as cs = outOf (decisionJF s x link c₁ mids f) := by
  simp only [nextJ, h]

/-- **The parsed value's decision at a session is `nextCs`'s** on a record in its canonical text that parses to it. -/
theorem decisionAtJ_eq (s : Ins) (spec : SessionSpec) (sts : Array Stmt) (x : Claims.Msg)
    (link : List ByteArray × List ByteArray) (c₁ : Coin) (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json}
    (hj : Json.parse f.record = .ok j) (hc : RecordText.printT j = f.record) :
    decisionAtJ s spec sts x link c₁ mids ⟨f.stmt, j, f.proofs⟩ = decisionCsAt s spec sts x link c₁ mids f := by
  have hs : RecordText.check f.record = pure () := RecordText.check_of hj hc
  rw [decisionCsAt_parse s spec sts x link c₁ mids hj]
  simp only [decisionAtJ, decisionJ, verdictJM, checksAtJ, checksJ, recordDraw_of_parse hj, hs, bind_pure_unit,
    Bool.false_eq_true, ↓reduceIte]

/-- **The parsed value's decision is `nextCs`'s** on a record in its canonical text that parses to it. -/
theorem decisionJF_eq (s : Ins) (x : Claims.Msg) (link : List ByteArray × List ByteArray) (c₁ : Coin)
    (mids : List (Claims.Msg × Coin)) {f : Final} {j : Json} (hj : Json.parse f.record = .ok j)
    (hc : RecordText.printT j = f.record) :
    decisionJF s x link c₁ mids ⟨f.stmt, j, f.proofs⟩ = decisionCs s x link c₁ mids f := by
  unfold decisionJF decisionCs
  congr 1
  funext p
  obtain ⟨spec, sts⟩ := p
  exact decisionAtJ_eq s spec sts x link c₁ mids hj hc

/-! ## The reader parts' checks -/

/-- **The reading at the last step**: `rd` reads the last message and the link message reads (`actJ` decides), and the
record value's fields custody reads are there with their shapes (`liveOfJ` at the substituted value is `.ok`). -/
def ReadOk (rd : Claims.Msg → Option FinalJ) (x : Claims.Msg) (ms : List Claims.Msg) (cs : List Coin) : Prop :=
  match actJ rd x ms cs with
  | .decide _ spec _ _ c₁ mids f => ∃ lv, liveOfJ spec (coinSubJ spec c₁ mids f.j) = .ok lv
  | _ => False

end FlockDeploy

namespace RecDeploy

open Flock
open Claims (Msg Out)
open FlockDeploy (Coin FinalJ nextJ nextJ_coin ReadOk)
open Claims.Recursion.VStarB

/-- **A split of `nextJ`'s last step that may assume the reading**: checks the inner statement fixes, and the proof that
where the reading holds (`ReadOk`) and every check does, the last step accepts. -/
structure LastSplitJ (H : Msg → Msg) (rd : Msg → Option FinalJ) (R : Type) where
  checks : Msg → List R
  holds : R → Msg → List Msg → List Coin → Prop
  sound : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), ms.length = (FlockDeploy.V false).k x →
    cs.length = ms.length → ReadOk rd x ms cs → (∀ r ∈ checks x, holds r x ms cs) → ((nextJ rd x ms cs).eval H).1 = .acc

/-- A check of `nextJ`'s last step: the reading (`read`) or the verdict's check `r` (`verdict r`). -/
inductive ReadCheck (R : Type) where
  | read
  | verdict (r : R)

/-- **The split with the reading as its own check.** -/
def LastSplitJ.toLast {H : Msg → Msg} {rd : Msg → Option FinalJ} {R : Type} (LJ : LastSplitJ H rd R) :
    LastSplit H (nextJ rd) (ReadCheck R) where
  checks x := .read :: (LJ.checks x).map .verdict
  holds q x ms cs := match q with
    | .read => ReadOk rd x ms cs
    | .verdict r => LJ.holds r x ms cs
  sound x ms cs hk hc h := LJ.sound x ms cs hk hc (h .read (List.mem_cons_self ..)) fun r hr =>
    h (.verdict r) (List.mem_cons_of_mem _ (List.mem_map.2 ⟨r, hr, rfl⟩))

/-- **The reading to the reader part**: the first part holding the last round whole, whose message carries the record
value; the verdict's checks as `ownR` gives them. -/
def ownJ {R : Type} (D : Claims.Recursion.Deploy Coin) (L : LayoutB) (ownR : Msg → R → Option ℕ) (x : Msg) :
    ReadCheck R → Option ℕ
  | .read => firstHolder D L x ((FlockDeploy.V false).k (Claims.Recursion.xI D x) - 1)
  | .verdict r => ownR x r

/-- `ownJ`'s owners are among the statement's parts wherever `ownR`'s are. -/
theorem ownJ_lt {H : Msg → Msg} {rd : Msg → Option FinalJ} {R : Type} (LJ : LastSplitJ H rd R)
    (ownR : Msg → R → Option ℕ) (D : Claims.Recursion.Deploy Coin) (L : LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LJ.checks (Claims.Recursion.xI D x), ∀ s, ownR x r = some s → s < D.nS x) :
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → shape st.form →
      ∀ r ∈ LJ.toLast.checks (Claims.Recursion.xI D x), ∀ s, ownJ D L ownR x r = some s → s < D.nS x := by
  intro x st hx hf r hr s ho
  cases r with
  | read => exact firstHolder_lt ho
  | verdict r =>
    refine hR x st hx hf r ?_ s ho
    rcases List.mem_cons.1 hr with h | h
    · cases h
    · obtain ⟨r', hr', h⟩ := List.mem_map.1 h
      cases h
      exact hr'

/-- **The composition at `nextJ`**: `bdDeployed` with `nextJ rd` as its steps, split by `deploySplit` at the reading's
split (`LastSplitJ.toLast`), the reading given to the reader part of the last round (`ownJ`). Where the verdict's owners
are the statement's parts (`hR`), the checks no part holds hold off any witnesses (`hrest`), and each part's circuit
accepting implies its share (`hlow`), `DecompB` holds at what each part's circuit accepts. Proof-file decoding is among
the verdict's checks, owned by no reader part; `RecDeploy.decompB_nextS` is the composition without it. -/
theorem decompB_nextJ (H : Msg → Msg) (rd : Msg → Option FinalJ)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) {join : List Msg → Msg}
    (hj : ∀ rs, ∃ a, join rs = 0 :: a)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    {R : Type} (LJ : LastSplitJ H rd R) (ownR : Msg → R → Option ℕ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (b : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) (shape : Claims.Recursion.Form → Prop) {Col' T' : Type}
    (DO : Claims.Flock.Deploy Coin Col' T') (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg)
    (hR : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ r ∈ LJ.checks (Claims.Recursion.xI (recBound rel' b L IV) x), ∀ s, ownR x r = some s →
        s < (recBound rel' b L IV).nS x)
    (hrest : ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), (recBound rel' b L IV).parse x = some st → shape st.form →
      ∀ q ∈ ivChecks (bdDeployed rel (nextJ rd) colsE wE join rowIn final w)
          (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast)
          (Claims.Recursion.xI (recBound rel' b L IV) x),
        ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR)) x q = none →
        (∀ r, q = .elem r → ∃ r', r = .last r') →
        ∀ (c : ℕ → Coin) (ws : List Claims.Flock.Wit), ws.length = (recBound rel' b L IV).nS x →
          (∀ e ∈ L.allCalls x ws, H e.1 = e.2) →
          q.Holds H (bdDeployed rel (nextJ rd) colsE wE join rowIn final w)
            (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast)
            (Claims.Recursion.xI (recBound rel' b L IV) x) c (L.msgs x ws))
    (hlow : ∀ s x c ms, PartAccepts H (recBound rel' b L IV) DO L shape shows cmt s x c ms →
      PartComputes H (recBound rel' b L IV) (bdDeployed rel (nextJ rd) colsE wE join rowIn final w)
        (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast)
        (ownHolders (recBound rel' b L IV) L (ownDeploy (ownJ (recBound rel' b L IV) L ownR))) s x c ms) :
    DecompB H (recBound rel' b L IV) (bdDeployed rel (nextJ rd) colsE wE join rowIn final w).IVStar L shape
      (PartAccepts H (recBound rel' b L IV) DO L shape shows cmt) :=
  decompB_partAccepts_of_parts H (recBound rel' b L IV) DO _ L shape shows cmt
    (deploySplit H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast) _ _ hlow
    (ownedBy_deploy H rel (nextJ rd) colsE wE join rowIn final w (nextJ_coin rd H) LJ.toLast
      (ownJ (recBound rel' b L IV) L ownR) _ L shape (ownJ_lt LJ ownR _ L shape hR))
    (stmtFixes_deploy H rel (nextJ rd) colsE wE hj rowIn final w (nextJ_coin rd H) LJ.toLast
      (ownJ (recBound rel' b L IV) L ownR) rel' b L IV shape hrest)

end RecDeploy
