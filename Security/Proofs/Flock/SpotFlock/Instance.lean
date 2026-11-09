module

public import Proofs.Flock.SpotFlock.FrameIsBuilt
public import Proofs.Flock.SpotFlock.Cap
import all Init.Data.Array.Basic

/-!
# A two-stage registration SPOT[Flock]'s premises hold at

`SpotFlock.Instance.held R B desc q c₀`: the auditor's copies (`Held`) of a registration of one template whose one
request is two units, of which the second stage proves one, over the template's reference statement `R` (`Ref`) and a
boundary registration `B` (`Bnd`) of one request sampled `subset:1`, at the descriptor `desc` and the statement's query
`q`, with SPOT's own frame (`frameAt` at the slot map, `revealCode` at the site map, from the code `c₀`). Its round 2
reads `R`'s links and reference registered reads, each present (`some`), since an absent one refuses every run, and no
wires (`some #[]`), since there is one template. No slot is always checked. A request is two units and the count is
one since the second stage refuses a request of one unit and a count proving a request whole, and with it every run
(`Select.secondProblem`, `secondProblem_held`).

Each bundle holds the facts the two-stage design reads of it, each a decision on its own data:
* `Ref`: the statement is in Flock's masked scope (`zkOk`), reads as a member (`templateOk`), and with its two units is
  in the two-stage scope (`scopeOk` over the members `ms2 R.x` and their replay partition `part2 R.x`);
* `Bnd`: the boundary's layout is one proof block (`oneLayout`), its first stage's design is `subset:1` (`slotDesign`,
  `firstK`), and its population is in slot order (`popOk`).

At them the design evaluates (`two_held`) and the second stage takes the counts (`secondProblem_held`), so the
registration doesn't refute a checked run before any session; the frame is SPOT's own (`built_held`); SPOT's layout
embeds the statement (`embeds_held`); and round 2's parts are the one template's (`partsAt_held`).

## What this does not show

`SpotKSTwo` reads two premises of a registration, `Built h` and `Sites h`; SPOT's layout embeds the statement at every
registration (`embeds_layoutFor`). At `held R bnd0 …` (`bnd0`, `SpotFlock.InstanceBoundary`'s concrete boundary) the
design evaluates, the second stage takes the counts and `Built` holds, for every `R : Ref`. `Sites` is not proved here.
So the two-stage guarantee's non-vacuity is reduced, not proved: it rests on one existence fact, a `Ref`, that is, a
reference statement with hidden outputs and `sha512/row/v2` ports, in the two-stage scope at two units a request,
whose registered reads carry roots and whose masked setup (`Stmt.setupHidden`, through `FlockDeploy.setupOf true`)
holds at the open guarantee that the verifier's fixed nets build (`Properties.Flock.Nets.FixedNetsBuild`). No such
statement is proved in Lean. That one exists is evidence, not proof: the Lean verifier accepts the honest run of the
two-stage toy (`benchmarks/two_stage/`), whose reference statements are of this kind, one of its templates at two
units a request.
-/

@[expose] public section

namespace SpotFlock.Instance

open Claims Claims.Sampling SamplingDeploy Properties.Spot
open Lean (Json)

/-- **One unit's boundary layout**: a population of one stratum of one proof block, unit `0` at the gates `gs`, and an
empty input class. -/
def oneLayout (gs : ℕ × ℕ) : Spot.Slots.Layout := ⟨#[#[#[⟨some 0, gs⟩]], #[]]⟩

/-- **The first stage's design** `subset:1`: one unit of the population, and every input slot. -/
def design1 : Design := ⟨[.subset 1, .all]⟩

/-- **One template with two units**: the statement circuit of `x` over two VUs. -/
def ms2 (x : Msg) : Spot.Bits.Members := [(stmtOf x, #[0, 1])]

/-- **Its replay partition** over one request's boundary layout, two units a request. -/
def part2 (x : Msg) : Partition := filesReplay (ms2 x) #[2] (oneLayout (0, 0))

theorem filesReplay_oneLayout (ms : Spot.Bits.Members) (ns : Array ℕ) (gs : ℕ × ℕ) :
    filesReplay ms ns (oneLayout gs) = filesReplay ms ns (oneLayout (0, 0)) := rfl

/-- **A boundary registration of one unit, sampled `subset:1`**, with the facts the two-stage design reads of it: its
layout is one proof block, its first stage's design is `subset:1`, which Flock reads as a uniform `1`-subset with no
closure map, and its population is in slot order. -/
structure Bnd where
  g : Reg
  gs : ℕ × ℕ
  layout : Spot.Slots.layout g.program g.query = .ok (oneLayout gs)
  design : slotDesign g.rule g.program g.query g.table = .ok design1
  firstK : firstK g = some 1
  pop : ∀ n, Spot.Partition.units g.program g.query = .ok n → Spot.Slots.popOk g.program g.query n = true

/-- **A template's reference statement**, named `name`, with its replay block's links (`links`: its input and output
ports at the boundary record) and its reference registered reads (`reads`: each port's registered read, `.null` for
none), and the facts the two-stage design reads of it: it sets up in Flock's masked scope, reads as a member, and with
its two units is in the two-stage scope. -/
structure Ref where
  x : Msg
  name : String
  links : Array Spot.SecondStage.Link
  reads : Json
  zk : zkOk x = true
  tmpl : ∀ h : Held, h.xS₀ 0 = x → templateOk h 0 = true
  scope : scopeOk (bitsProgs (ms2 x)) (membersPart (ms2 x)) ⟨part2 x, design1, [1]⟩ = true

/-- **The two-stage registration** over the boundary `B`: two units a request and a count of one (`ns`, `ks`), request
`ρ`'s unit `p` its block `2ρ + p` (the members', `filesBlockOf`), and no slot always checked. -/
def reg (B : Bnd) : RegTwo :=
  { first := B.g, ns := #[2], ks := #[1], blockOf := fun ρ _ p => ρ * 2 + p, always := fun _ => false }

/-- **The auditor's copies at the frame `c₀`**: one template, `R`'s, with two VUs, the second stage's count `1`, `R`'s
links and reference registered reads and no wires, at the descriptor `desc` and the query `q`. -/
def held₀ (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code) : Held where
  desc := desc
  vus := [#[0, 1]]
  query := q
  reg := reg B
  k := [1]
  names := [R.name]
  links := some R.links
  wires := some #[]
  xS₀ := fun _ => R.x
  xS₀_ok := fun _ => R.zk
  refReads := fun _ => R.reads
  frame := c₀

/-- **The auditor's copies**: `held₀`'s, with SPOT's own frame from `c₀`. -/
noncomputable def held (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code) : Held :=
  let h := held₀ R B desc q c₀
  { h with frame := frameAt (slotMap h) (revealCode h (siteMap h) c₀) }

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

theorem ms_held : (held R B desc q c₀).ms = ms2 R.x := rfl

theorem replayDesign_held : replayDesign (held R B desc q c₀) = some design1 := by
  show (slotDesign B.g.rule B.g.program B.g.query B.g.table).toOption = some design1
  rw [B.design]
  rfl

theorem layout_held :
    (Spot.Slots.layout (held R B desc q c₀).reg.first.program (held R B desc q c₀).reg.first.query).toOption =
      some (oneLayout B.gs) := by
  show (Spot.Slots.layout B.g.program B.g.query).toOption = some (oneLayout B.gs)
  rw [B.layout]
  rfl

theorem filesOk_held : filesOk (reg B) (ms2 R.x) (oneLayout B.gs) [1] = true := rfl

theorem templates_held : (List.range (held R B desc q c₀).ms.length).all (templateOk (held R B desc q c₀)) = true := by
  rw [ms_held]
  show [0].all (templateOk (held R B desc q c₀)) = true
  simp only [List.all_cons, List.all_nil, Bool.and_true]
  exact R.tmpl _ rfl

/-- **The design evaluates**: the members' replay partition over one request's boundary layout, the design `subset:1`
and the count `1`. -/
theorem two_held (s : AuditStatement) :
    two (held R B desc q c₀) s = some ⟨filesReplay (ms2 R.x) #[2] (oneLayout B.gs), design1, [1]⟩ := by
  have hc : (scopeOk (bitsProgs (held R B desc q c₀).ms) (membersPart (held R B desc q c₀).ms)
      ⟨filesReplay (held R B desc q c₀).ms (held R B desc q c₀).reg.ns (oneLayout B.gs), design1,
        (held R B desc q c₀).k⟩ &&
      filesOk (held R B desc q c₀).reg (held R B desc q c₀).ms (oneLayout B.gs) (held R B desc q c₀).k &&
      (List.range (held R B desc q c₀).ms.length).all (templateOk (held R B desc q c₀))) = true := by
    rw [templates_held, ms_held]
    simp only [Bool.and_true, Bool.and_eq_true]
    refine ⟨?_, filesOk_held R B⟩
    rw [filesReplay_oneLayout]
    exact R.scope
  unfold two
  rw [layout_held, replayDesign_held]
  dsimp only
  rw [ite_eq_left hc, ms_held]
  rfl

theorem two_held_isSome (s : AuditStatement) : (two (held R B desc q c₀) s).isSome = true := by
  rw [two_held]
  rfl

/-- **The frame is SPOT's own.** -/
theorem built_held : Built (held R B desc q c₀) := built_spot (held₀ R B desc q c₀) c₀

/-- **SPOT's layout embeds the statement.** -/
theorem embeds_held : Embeds (held R B desc q c₀) (layoutFor (held R B desc q c₀)) := embeds_layoutFor _

/-- **Round 2's parts**: the one template's, named `R.name`, from slot `0`, of two units. -/
theorem partsAt_held (n : ℕ) : partsAt (held R B desc q c₀) n = #[⟨R.name, 0, 2⟩] := rfl

/-- **The second stage takes the counts**: at the replay sample of the one request, `Select.selectSecond` raises none
of its refusals over the registration's cells and counts, one unit of the request's two. -/
theorem secondProblem_held :
    Spot.Select.secondProblem #[0] (held R B desc q c₀).reg.ns (held R B desc q c₀).reg.ks = none := by
  show Spot.Select.secondProblem #[0] #[2] #[1] = none
  decide +kernel

end SpotFlock.Instance
