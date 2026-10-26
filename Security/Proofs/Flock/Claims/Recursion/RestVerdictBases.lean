import Proofs.Flock.Claims.Recursion.VerdictPieces
import Proofs.Flock.Claims.Recursion.NextShaDec
import Proofs.Flock.Claims.Recursion.DeployChecked

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

/-- **The session's draw is the statement's**: at every statement that sets up, `unitDraw` is the statement's draw up
to `JsonEquiv`. -/
def DrawTied : Prop :=
  ∀ (x : Msg) (s : FlockDeploy.Ins) (spec : SessionSpec) (sts : Array Stmt), setupOf false x = some (s, spec, sts) →
    drawEq spec.unitDraw s.read.draw = true

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
  obtain ⟨v, -, -, -, -, -, -, hv, -, -, -, -, -, he⟩ := FlockDeploy.decR_some hd
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
  simp only [hS, beq_self_eq_true, hD, hL, hC, hQ, hN', FlockDeploy.Custody.check_true, bind, Except.bind, pure,
    Except.pure, Bool.false_eq_true, ↓reduceIte]
  rfl

end

end RecDeploy
