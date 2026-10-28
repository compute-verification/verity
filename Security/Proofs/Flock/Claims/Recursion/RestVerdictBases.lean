import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.Recursion.DeployChecked
import Proofs.Flock.Claims.JsonNorm

/-!
# `hrest`'s base checks: the verdict's other checks hold off any witnesses

`decompB_lastSplitP` asks every check the inventory gives no part to hold at the readers' messages of any witnesses
(`hrest`). Among them are the four checks of `VerdictR` other than the verdict (`VerdictP.bases`), wherever `ownB`
gives them to no part. At SHA-512 each holds at every decision of `nextR`, from what the outer reader checks of the
inner statement (`parse_innerOk`: it sets up, its scope check passes, it carries a draw) and what the reading fixes:

* **`.setup`** (`holdsR_setup`): the setup at `hOf sha512` is the statement's (`setupWith_sha512_of`).
* **`.names`** (`holdsR_names`): the scope check leaves no public inputs, so the rep streams' names are distinct
  (`namesDistinct_of_setup`).
* **`.shaped`** (`holdsR_shaped`): where the decided proofs are shaped for their files (`DecShaped`).
* **`.after`** (`holdsR_after`): the last message's statement is the statement (`DecStmt`), the reader's record's draw
  is the statement's (`DrawTied`), custody passes at the reader's record (`CustodyRecJ`), the scope check passes, and
  the draw-less refusal is not reached.

`RestBaseHolds` is the statement `hrest` reads at a base check: at every compiled statement the outer reader parses,
every coin sequence and every messages, the base check holds (`restBase_holds`). It reads no witness, so it gives
`hrest`'s base cases at the readers' messages of any witnesses.

**What the reading must fix.** `DecStmt` and `DecShaped` are facts about `decR`: the last message's statement and the
opened proofs' lengths are the prover's (`rd` reads them from the last message, which `final` assembles from the parts'
pieces), so no statement fact gives them. `decR` fixes both where it takes the statement for the last message's and
refuses proofs that are not shaped. `DrawTied` is a statement fact: the session's `unitDraw` is the statement's draw.
At several tables the setup sets it to the draw; at one table it is read back from the public file's parsed header,
which only `JsonSpec.parse_wf` (a field of a named premise) makes a well-formed tree.
-/

namespace RecDeploy

open Flock
open Lean (Json)
open Claims (Msg)
open FlockDeploy (Coin FinalP nextR nextR_coin VerdictR holdsR decR DecR actR jAt checksAtJ drawOfJ drawEq custody
  liveOfJ setupOf setupWith tabsOf pointsOf shapeAt NamesDistinct)
open Claims.Recursion.VStarB

noncomputable section

/-! ## What the reading and the statement fix -/

/-- **The reading fixes the last message's statement**: at every decision of `decR`, it is the inner statement. -/
def DecStmt (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (d : DecR), decR rd x as cs = some d → d.f.stmt = x

/-- **The reading refuses proofs not shaped for their files**: at every decision of `decR`, each decided proof is
shaped at the shape its file is read at. -/
def DecShaped (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (d : DecR), decR rd x as cs = some d →
    ∀ (k : ℕ) (hk : k < d.f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) k) d.f.ps[k]

/-- **The session's draw is the statement's**: at every statement that sets up and passes its scope check, the draw a
record carrying the session's `unitDraw` has is the statement's draw up to `JsonEquiv`. -/
def DrawTied : Prop :=
  ∀ (x : Msg) (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), setupOf false x = some (s, spec, sts) →
    s.read.check = .ok () → drawEq (drawOfU spec.unitDraw) s.read.draw = true

/-- **Custody passes at the reader's record**: at every decision of `decR` on as many coins as messages, at a
statement whose scope check passes, what `liveOfJ` reads of the substituted record (`jAt` at SHA-512) passes custody
against the decision's link and round messages. -/
def CustodyRecJ (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (as : List Msg) (cs : List Coin) (d : DecR), as.length = cs.length → decR rd x as cs = some d →
    d.s.read.check = .ok () →
    ∃ lv, liveOfJ d.spec (jAt FlockDeploy.sha512 d.spec d.sts d.c₁ d.mids d.f) = .ok lv ∧
      custody d.spec lv d.link d.c₁ d.mids = .ok ()

/-! ## A decision's setup -/

/-- **A decision of `actR` is a decision of `decR` at the statement's setup.** -/
theorem actR_decide_setup {rd : Msg → Option FinalP} {x : Msg} {as : List Msg} {cs : List Coin}
    {s : FlockDeploy.Ins} {spec : SessionSpec} {sts : Array Stmt} {link : List ByteArray × List ByteArray} {c₁ : Coin}
    {mids : List (Msg × Coin)} {f : FinalP} (h : actR rd x as cs = .decide s spec sts link c₁ mids f) :
    setupOf false x = some (s, spec, sts) ∧ decR rd x as cs = some ⟨s, spec, sts, link, c₁, mids, f⟩ := by
  have hd := FlockDeploy.actR_decide h
  obtain ⟨v, c₁', r₀, rss, fl, link', pf, hv, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
  obtain ⟨hs, hspec, hsts, -⟩ := DecR.mk.inj he
  subst hs hspec hsts
  exact ⟨hv, hd⟩

/-! ## The base checks at a decision -/

/-- **`.setup` holds at every decision at SHA-512**: the setup at `hOf sha512` is the statement's. -/
theorem holdsR_setup (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin) :
    holdsR FlockDeploy.sha512 rd .setup x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    exact ⟨_, setupWith_sha512_of (actR_decide_setup h).1⟩
  · trivial

/-- **`.names` holds at every decision at SHA-512** of a statement whose scope check passes: it leaves no public
inputs, so the rep streams' names are distinct. -/
theorem holdsR_names (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hck : ∀ s spec sts, setupOf false x = some (s, spec, sts) → s.read.check = .ok ()) :
    holdsR FlockDeploy.sha512 rd .names x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    have hx := (actR_decide_setup h).1
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    exact FlockDeploy.namesDistinct_of_setup (FlockDeploy.setupOf_some hx).2
      (Claims.Bridge.Custody.publicInputs_none_of_check (hck _ _ _ hx))
  · trivial

/-- **`.shaped` holds at every decision at SHA-512 whose proofs are shaped** for their files. -/
theorem holdsR_shaped (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hsh : ∀ d, decR rd x as cs = some d → ∀ (k : ℕ) (hk : k < d.f.ps.size),
      FlockVBridge.Family.ProofShaped (shapeAt (tabsOf d.sts) (pointsOf d.spec d.c₁) k) d.f.ps[k]) :
    holdsR FlockDeploy.sha512 rd .shaped x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    obtain ⟨hx, hd⟩ := actR_decide_setup h
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    exact hsh _ hd
  · trivial

/-- **The checks after the verdict pass** where each does: the statement, the draw, custody, the scope check, and a
drawn session. -/
theorem checksAtJ_ok_of {s : FlockDeploy.Ins} {spec : SessionSpec} {x : Msg} {link : List ByteArray × List ByteArray}
    {c₁ : Coin} {mids : List (Msg × Coin)} {f : FlockDeploy.FinalJ} {j' : Json} {lv : FlockDeploy.Live}
    (hS : f.stmt = x) (hD : drawEq (drawOfJ f.j) s.read.draw = true) (hL : liveOfJ spec j' = .ok lv)
    (hC : custody spec lv link c₁ mids = .ok ()) (hQ : s.read.check = .ok ()) (hN : s.read.draw.isSome = true) :
    checksAtJ s spec x link c₁ mids f j' = .ok () := by
  have hN' : s.read.draw.isNone = false := by
    cases h : s.read.draw with
    | none => rw [h] at hN; cases hN
    | some _ => rfl
  unfold checksAtJ
  simp only [hS, beq_self_eq_true, hD, hL, hC, hQ, hN', Record.check, bind, Except.bind, pure, Except.pure,
    Bool.false_eq_true, ↓reduceIte]
  try rfl

/-- **The honest record's draw is its session's `unitDraw`, rebuilt**, as `drawOfJ` reads it. -/
theorem drawOfJ_recordObj (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Flock.Round)) (proofs : Array ByteArray) :
    drawOfJ (FlockComplete.JsonRT.recordObj spec h rootB points streams proofs) =
      (drawOfU spec.unitDraw).map Flock.norm := by
  unfold drawOfJ FlockComplete.JsonRT.recordObj
  rw [FlockComplete.JsonRT.getObjVal?_mkObj (FlockComplete.JsonRT.recordList_distinct spec h rootB points streams
    proofs) (k := "unit_draw") (v := spec.unitDraw.getD Json.null) (by simp [FlockComplete.JsonRT.recordList])]
  cases spec.unitDraw with
  | none => rfl
  | some d => cases d <;> first | rfl | exact congrArg some (FlockDeploy.norm_norm _).symm

/-- **Rebuilding one side keeps the draw check** where the other side's draw is already rebuilt. -/
theorem drawEq_map_norm {a b : Option Json} (hb : ∀ e, b = some e → Flock.norm e = e) (h : drawEq a b = true) :
    drawEq (a.map Flock.norm) b = true := by
  rw [FlockDeploy.drawEq_iff] at h ⊢
  cases h with
  | none => exact .none
  | @some a' b' h' =>
    refine .some ?_
    show FlockDeploy.JsonEquiv (Flock.norm a') b'
    rw [FlockDeploy.norm_congr a' b' h', hb b' rfl]
    exact FlockDeploy.JsonEquiv.refl _

/-- **`.after` holds at every decision at SHA-512** on as many coins as messages, of a statement whose scope check
passes and which carries a draw, where the reading fixes the last message's statement, the session's draw is the
statement's and custody passes at the reader's record. -/
theorem holdsR_after (rd : Msg → Option FinalP) (x : Msg) (as : List Msg) (cs : List Coin)
    (hlen : as.length = cs.length)
    (hck : ∀ s spec sts, setupOf false x = some (s, spec, sts) → s.read.check = .ok () ∧ s.read.draw.isSome = true)
    (hstmt : DecStmt rd) (hud : DrawTied) (hcu : CustodyRecJ rd) :
    holdsR FlockDeploy.sha512 rd .after x as cs := by
  unfold holdsR
  split
  · next s spec sts link c₁ mids f h =>
    obtain ⟨hx, hd⟩ := actR_decide_setup h
    intro spec' sts' hs'
    rw [setupWith_sha512_of hx] at hs'
    obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj hs')
    obtain ⟨hq, hn⟩ := hck _ _ _ hx
    obtain ⟨lv, hL, hC⟩ := hcu x as cs _ hlen hd hq
    have hD : drawEq (drawOfJ f.j) s.read.draw = true := by
      obtain ⟨v, c₁', r₀, rss, fl, link', pf, -, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
      obtain ⟨-, hspec, -, -, -, -, hf⟩ := DecR.mk.inj he
      rw [hf]
      dsimp only
      rw [FlockDeploy.recJ, drawOfJ_recordObj, ← hspec]
      exact drawEq_map_norm (fun e he => (FlockDeploy.insOf_draw_norm (FlockDeploy.setupOf_some hx).1 he).1)
        (hud _ _ _ _ hx hq)
    exact checksAtJ_ok_of (f := ⟨f.stmt, f.j, #[]⟩) (hstmt x as cs _ hd) hD hL hC hq hn
  · trivial

/-! ## The base checks at the outer reader's statements -/

/-- **`hrest`'s base checks, at the reading**: at the inner statement of every compiled statement the outer reader
parses, on as many coins as messages, each of the verdict's other checks holds. -/
def BasesHold (rd : Msg → Option FinalP) : Prop :=
  ∀ (x : Msg) (st : Claims.Recursion.Stmt), parse x = some st → ∀ b ∈ VerdictP.bases, ∀ (as : List Msg)
    (cs : List Coin), as.length = cs.length → holdsR FlockDeploy.sha512 rd b (xIOf x) as cs

/-- **The base checks hold** where the reading fixes the last message's statement and refuses unshaped proofs, the
session's draw is the statement's and custody passes at the reader's record. -/
theorem basesHold (rd : Msg → Option FinalP) (hstmt : DecStmt rd) (hsh : DecShaped rd) (hud : DrawTied)
    (hcu : CustodyRecJ rd) : BasesHold rd := by
  intro x st hx b hb as cs hlen
  have hck : ∀ s spec sts, setupOf false (xIOf x) = some (s, spec, sts) →
      s.read.check = .ok () ∧ s.read.draw.isSome = true := by
    intro s spec sts h
    obtain ⟨s', spec', sts', h', hq, hn⟩ := parse_innerOk hx
    rw [h] at h'
    obtain ⟨rfl, -⟩ := Prod.mk.inj (Option.some.inj h')
    exact ⟨hq, hn⟩
  simp only [VerdictP.bases, List.mem_cons, List.not_mem_nil, or_false] at hb
  rcases hb with rfl | rfl | rfl | rfl
  · exact holdsR_setup rd _ as cs
  · exact holdsR_shaped rd _ as cs (hsh _ as cs)
  · exact holdsR_names rd _ as cs fun s spec sts h => (hck s spec sts h).1
  · exact holdsR_after rd _ as cs hlen hck hstmt hud hcu

/-- **`hrest` at a base check**: at every compiled statement the outer reader parses, every coins and every
messages, `IVStar`'s check of the split verdict's base check `b` holds, whatever the pieces and the deployment's other
parameters. -/
theorem restBase_holds (rd : Msg → Option FinalP) (hB : BasesHold rd) {Q : Type} (pieces : Msg → List Q)
    (pholds : Q → Msg → List Msg → List Coin → Prop)
    (hpieces : ∀ (x : Msg) (ms : List Msg) (cs : List Coin), FlockDeploy.ReadOkR rd x ms cs →
      (∀ b ∈ VerdictP.bases, holdsR FlockDeploy.sha512 rd b x ms cs) → (∀ q ∈ pieces x, pholds q x ms cs) →
        holdsR FlockDeploy.sha512 rd .verdict x ms cs)
    (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (rel' : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation) (bB : Brings) (L : LayoutB)
    (IV : Claims.Verifier Coin) {x : Msg} {st : Claims.Recursion.Stmt} (hx : (recBound rel' bB L IV).parse x = some st)
    {b : VerdictR} (hb : b ∈ VerdictP.bases) (c : ℕ → Coin) (ms : List Msg) :
    (IVCheck.elem (DeployCheck.last (ReadCheck.verdict (VerdictP.base b))) :
        IVCheck (DeployCheck (ReadCheck (VerdictP Q)))).Holds FlockDeploy.sha512
      (bdDeployed rel (nextR rd) colsE wE join rowIn final w)
      (deploySplit FlockDeploy.sha512 rel (nextR rd) colsE wE join rowIn final w (nextR_coin rd FlockDeploy.sha512)
        (lastSplitP FlockDeploy.sha512 rd pieces pholds hpieces).toLast)
      (Claims.Recursion.xI (recBound rel' bB L IV) x) c ms := by
  intro os last _ hos
  refine hB x st hx b hb _ _ ?_
  rw [List.length_map, List.length_range, ← hos]
  simp [Claims.Recursion.Bound.opened]

end

end RecDeploy
