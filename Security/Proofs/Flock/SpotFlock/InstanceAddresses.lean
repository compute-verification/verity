module

public import Proofs.Flock.SpotFlock.Instance
public import Proofs.Flock.SpotFlock.ReadsAt
public import Proofs.Flock.Claims.Sampling.Select.Second

/-!
# Where every accepted run of the two-stage instance stands

At `SpotFlock.Instance.held R B desc q c₀`, for every `R` and `B`, every accepted run (`CheckedRun.Ok`) has:
* the replay sample `#[0]` (`sample_ok`): round 2 sampled the second stage (`secondDraw`), which `Select.secondProblem`
  lets sample only over a nonempty, ascending sample, and the first stage samples unit slots (`firstSlots_held`) of a
  layout of one proof block (`sample_getD`);
* replay block `0` (`rho_ok`), the replay partition's one, and a unit below two (`u_lt_two`);
* its unit in template `0`, whose part is two units from unit `0` (`drawnPart_ok`);
and round 2 reads no wire at any port (`wireOf_held`).

From these, a registered read's slot opens the position the session header gives the unit's instance's row,
`ReadsRegistered`'s last conjunct, at each arm the instance has:
* a port a link names (`leafAt_link`): at a slot with no units, stride `0` and position `0`. Round 2 holds a linked
  read at row `ρ · per + x` of the port's rows (`placedOk_linked`, `linkedRow_lt`), with `x < per`; the instance's
  rows have `per = 1` (`per_held`) and `ρ = 0`, so the row is `0`;
* the template's own port (`leafAt_own`): at a slot with no units, stride `2` and position the unit's, where each entry
  round 2 accepts for the read takes the own arm (`CheckedRun.leafAt_own`).

Each goes through the entry round 2 accepts (`ReadsAt.regEntry`) rather than an arm fact for every entry, since an
entry's arm turns on its value, which `ReadsAt.ReadFields` doesn't state.
-/

@[expose] public section

namespace SpotFlock.Instance

open Claims Claims.Sampling SamplingDeploy Properties.Spot

/-- An ascending replay sample of one replay block or more, each `0`, is `#[0]`. -/
theorem eq_zero_of_asc {s : Array ℕ} (hne : s.isEmpty = false)
    (hasc : (s.toList.zip s.toList.tail).all (fun (a, b) => decide (a < b)) = true)
    (h0 : ∀ i < s.size, s.getD i 0 = 0) : s = #[0] := by
  rcases s with ⟨_ | ⟨a, _ | ⟨b, l⟩⟩⟩
  · simp at hne
  · have ha := h0 0 (by simp)
    simp only [Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_cons_zero, Option.getD_some] at ha
    rw [ha]
  · have ha := h0 0 (by simp)
    have hb := h0 1 (by simp)
    simp only [Array.getD_eq_getD_getElem?, List.getElem?_toArray, List.getElem?_cons_zero, List.getElem?_cons_succ,
      Option.getD_some] at ha hb
    simp only [List.tail_cons, List.zip_cons_cons, List.all_cons, decide_eq_true_eq, Bool.and_eq_true] at hasc
    omega

variable (R : Ref) (B : Bnd) (desc : Msg) (q : Query) (c₀ : Code)

/-- **The first stage samples replay block `0` only**, on any coins: it samples unit slots of a layout of one proof
block. -/
theorem sample_getD (src : ByteArray) :
    ∀ i < ((held R B desc q c₀).reg.sample src).size, ((held R B desc q c₀).reg.sample src).getD i 0 = 0 := by
  intro i hi
  obtain ⟨b, hb, -⟩ := firstSlots_held R B desc q c₀ (oneLayout B.gs) B.layout src i hi
  obtain ⟨hlt, -⟩ := Array.getElem?_eq_some_iff.1 hb
  have h1 : (oneLayout B.gs).blocks.size = 1 := rfl
  omega

/-- **Round 2 reads no wire**: the one template's reads are its own or linked. -/
theorem wireOf_held (st : Roots) (rec draw : Lean.Json) (κ : FlockDeploy.Coin) (t port : String) :
    Spot.SecondStage.wireOf (ownOf (held R B desc q c₀) st rec draw κ) t port = none := rfl

variable {R B desc q c₀}

/-- **An accepted run's replay sample is replay block `0`.** -/
theorem sample_ok {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) :
    (held R B desc q c₀).reg.sample (src r.c₀) = #[0] := by
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, -, -, -, -, -⟩ := hr
  have hat := stmtOkAt_unit _ hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨_, _, _, _, _, -, -, hdraw, -⟩ := readsOk_true _ hat.2
  obtain ⟨_, hsel⟩ := secondDraw_ok _ hdraw
  obtain ⟨hne, hasc⟩ :=
    Properties.Spot.SecurityProofs.secondProblem_sample (Properties.Spot.SecurityProofs.selectSecond_ok hsel).1
  exact eq_zero_of_asc hne hasc (sample_getD R B desc q c₀ _)

/-- **An accepted run's replay block is `0`**, the replay partition's one. -/
theorem rho_ok {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) : r.ρ = 0 := by
  obtain ⟨_, _, _, -, -, h2, -, -, -, -, -, hρ, -, -⟩ := hr
  rw [two_held] at h2
  cases h2
  unfold replayOf at hρ
  have hlt : r.ρ < (filesReplay (ms2 R.x) #[2] (oneLayout B.gs)).blocks.length :=
    List.mem_range.1 (List.mem_of_find?_eq_some hρ)
  have h1 : (filesReplay (ms2 R.x) #[2] (oneLayout B.gs)).blocks.length = 1 := rfl
  omega

/-- **An accepted run's unit is one of the request's two.** -/
theorem u_lt_two {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) : r.u < 2 := by
  obtain ⟨_, _, _, hx, -, -, -, -, hu, -⟩ := hr
  have hlt := mem_checked_lt hu
  have hN : unitCount (held R B desc q c₀) r.x = 2 := by
    unfold unitCount
    rw [hx]
    exact (_root_.Flock.Bridge.Members.length_blocks_membersPart (ms := (held R B desc q c₀).ms)).trans rfl
  omega

/-- **An accepted run's unit is in template `0`**, whose part is the replay block's two units from unit `0`. -/
theorem drawnPart_ok {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) :
    r.drawnPart (held R B desc q c₀) = ⟨R.name, 0, 2⟩ := by
  obtain ⟨_, _, _, hx, -, -, -, -, hu, -⟩ := hr
  have ht : templateOf (held R B desc q c₀).ms r.u < 1 := templateOf_lt_of_checked _ hx hu
  show partAt (held R B desc q c₀) _ (templateOf (held R B desc q c₀).ms r.u) = _
  rw [Nat.lt_one_iff.1 ht]
  rfl

/-- **A linked read round 2 accepts is placed at the port's rows**: at a port a link names, `readOk` accepts an entry
only where the boundary record states a population `M` and the design gives the port rows `ro`, and its arm is then
`M · ro.per` leaves, slot `u` at the row `linkedRow` gives, which `placedOk` holds the entry's reads to. -/
theorem placedOk_linked {own : Spot.SecondStage.Own} {named : Bool} {p : Spot.SecondStage.Part} {port : String}
    {e : Lean.Json} (h : Spot.SecondStage.readOk own named p port e = true) {l : Spot.SecondStage.Link}
    (hl : Spot.SecondStage.linkOf own p.template port = some l) :
    ∃ M ro, Spot.SecondStage.rowsOf own p.template port = some ro ∧
      ReadsAt.armOf own p port e = some (M * ro.per, Spot.SecondStage.linkedRow own p ro) ∧
      Spot.SecondStage.placedOk own.sample.size p e (M * ro.per) (Spot.SecondStage.linkedRow own p ro) = true := by
  have hr := h
  unfold Spot.SecondStage.readOk at h
  rw [hl] at h
  revert h
  generalize hv : Spot.getD e "value" = val
  generalize Spot.SecondStage.wireOf own p.template port = wr
  intro h
  rcases val with _ | _ | _ | v | _ | _ <;> rcases wr with _ | w <;>
    simp only [Bool.and_eq_true, reduceCtorEq] at h
  all_goals
    obtain ⟨-, h⟩ := h
    split at h
    · have ha := ReadsAt.armOf_link hv hl ‹_› ‹_›
      exact ⟨_, _, ‹_›, ha, ReadsAt.placedOk_of_readOk hr (.inl (by simp [hl])) ha⟩
    · cases h

/-- **A linked row is below its rows' `per`** in its replay block's: `linkedRow` gives a row only there. -/
theorem linkedRow_lt {own : Spot.SecondStage.Own} {p : Spot.SecondStage.Part} {ro : Spot.SecondStage.Rows} {u b : ℕ}
    (hb : Spot.SecondStage.linkedRow own p ro u = some b) :
    ∃ a x, own.sample[p.span u]? = some a ∧ x < ro.per ∧ b = a * ro.per + x := by
  obtain ⟨a, x, hsa, hrow, rfl⟩ := ReadsAt.linkedRow_some hb
  refine ⟨a, x, hsa, ?_, rfl⟩
  by_contra hx
  unfold Spot.SecondStage.linkedRow at hb
  simp [hsa, hrow, hx] at hb

/-- **The instance's rows are one a replay block**, at every port the design gives rows. -/
theorem per_held {st rec draw c} {t port : String} {ro : Spot.SecondStage.Rows}
    (hro : Spot.SecondStage.rowsOf (ownOf (held R B desc q c₀) st rec draw c) t port = some ro) : ro.per = 1 := by
  have hm : ro ∈ R.links.map (fun l => (⟨l.template, l.port, 1, #[⟨0, 2, 0, 0⟩]⟩ : Spot.SecondStage.Rows)) :=
    Array.mem_of_find?_eq_some hro
  obtain ⟨l, -, rfl⟩ := Array.mem_map.1 hm
  rfl

/-- **A linked read opens the header's position at the instance** at a slot with no units, stride `0` and position
`0`: round 2 holds a linked read at row `ρ · per + x` of the port's rows (`placedOk_linked`, `ReadsAt.pos_of_placedOk`),
with `x < per` (`linkedRow_lt`), and the replay sample is `#[0]` (`sample_ok`) and `per = 1` (`per_held`), so the
read's position is `0`. -/
theorem leafAt_link {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) {own : Array _root_.Flock.Registered.Port}
    {reg : _root_.Flock.Registered.Port} (hown : r.ins.registered = some own) (hreg : reg ∈ own) {p : ℕ}
    (hp : (List.range r.sts[0]!.c.ports.size).find? (fun k => r.sts[0]!.c.ports[k]!.name == reg.name) = some p)
    {l : Spot.SecondStage.Link} (hl : R.links.find? (fun l => l.template == R.name && l.port == reg.name) = some l)
    {roots : List Msg} {s : Slot} (hs0 : s.units = 0) (hst : s.stride = 0) (hpos : s.pos = 0)
    {t : _root_.Flock.Tables} {positions : Array ℕ} {paths : Array ByteArray} (ht : r.sts[0]!.pub.tables = some t)
    (hent : _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths)) (en : Entry) :
    leafAt roots s en =
      positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size p (r.n (held R B desc q c₀))) 0 := by
  have hsam := sample_ok hr
  obtain ⟨D, -, hn, -, hj⟩ :=
    CheckedRun.readPosAt _ hr (CheckedRun.count_le_one hr) (counted_held _ _ _ _ _ hr) hown hreg hp ht hent
  obtain ⟨_, _, _, hx, -, -, -, hok, hu, -, -, -, -, hset⟩ := hr
  have ht0 : templateOf (held R B desc q c₀).ms r.u < 1 := templateOf_lt_of_checked _ hx hu
  obtain ⟨st, rec, draw, e, -, -, -, hrd, hf⟩ := ReadsAt.regEntry _ hx hok hu hset hown hreg
  rw [Nat.lt_one_iff.1 ht0] at hrd
  obtain ⟨M, ro, hro, -, hpl⟩ := placedOk_linked hrd (l := l) hl
  obtain ⟨-, b, hb, -, hpos'⟩ := ReadsAt.pos_of_placedOk hpl hf hj
  obtain ⟨a, x, hsa, hx, rfl⟩ := linkedRow_lt hb
  have hper := per_held hro
  change ((held R B desc q c₀).reg.sample (src r.c₀))[_]? = some a at hsa
  rw [hsam] at hsa
  have ha0 : a = 0 := by simpa using Array.mem_of_getElem? hsa
  have hleaf : leafAt roots s en = 0 := by simp [leafAt, unitsAt, hs0, hst, hpos]
  rw [hper] at hx
  rw [hleaf, hpos', ha0]
  omega

/-- **An own-port read opens the header's position at the instance** at a slot with no units, stride `2` and position
the unit's: where each entry round 2 accepts for the read takes the own arm, `CheckedRun.leafAt_own` at the instance's
part of two units from unit `0` (`drawnPart_ok`) and replay block `0` (`rho_ok`). -/
theorem leafAt_own {r : CheckedRun} (hr : r.Ok (held R B desc q c₀)) {own : Array _root_.Flock.Registered.Port}
    {reg : _root_.Flock.Registered.Port} (hown : r.ins.registered = some own) (hreg : reg ∈ own) {p : ℕ}
    (hp : (List.range r.sts[0]!.c.ports.size).find? (fun k => r.sts[0]!.c.ports[k]!.name == reg.name) = some p)
    {s : Slot} (hs : s.sized = false) (hs0 : s.units = 0) (hst : s.stride = 2) (hpos : s.pos = r.u)
    (harm : ∀ st rec draw e, decode r.x = some st → jsonMsg r.m₁ = some rec →
      secondDraw (held R B desc q c₀) r.c₀ r.c₁ = some draw →
      Spot.SecondStage.readOk (ownOf (held R B desc q c₀) st rec draw r.c₀) (Spot.SecondStage.byTemplate rec)
        (r.drawnPart (held R B desc q c₀)) reg.name e = true → ReadsAt.ReadFields e reg →
      ReadsAt.armOf (ownOf (held R B desc q c₀) st rec draw r.c₀) (r.drawnPart (held R B desc q c₀)) reg.name e =
        some (((held R B desc q c₀).reg.sample (src r.c₀)).size * (r.drawnPart (held R B desc q c₀)).n,
          fun u => some (u - (r.drawnPart (held R B desc q c₀)).lo)))
    {t : _root_.Flock.Tables} {positions : Array ℕ} {paths : Array ByteArray} (ht : r.sts[0]!.pub.tables = some t)
    (hent : _root_.Flock.Registered.entry r.sts[0]!.pub reg = .ok (positions, paths)) {en : Entry}
    (hl : en.leaves = reg.leaves) :
    leafAt (irootsOf r.m₁ r.ρ) s en =
      positions.getD (_root_.Flock.Registered.ref t r.sts[0]!.c.ports.size p (r.n (held R B desc q c₀))) 0 := by
  have hdp := drawnPart_ok hr
  have hρ := rho_ok hr
  have hu2 := u_lt_two hr
  have hst' : s.stride = (r.drawnPart (held R B desc q c₀)).n := by rw [hdp]; exact hst
  have hpos' : s.pos < (r.drawnPart (held R B desc q c₀)).n := by
    rw [hdp]
    show s.pos < 2
    omega
  have hu : (held R B desc q c₀).reg.blockOf r.ρ (templateOf (held R B desc q c₀).ms r.u) s.pos = r.u := by
    show r.ρ * 2 + s.pos = r.u
    omega
  exact CheckedRun.leafAt_own _ hr (CheckedRun.count_le_one hr) (counted_held _ _ _ _ _ hr)
    (firstSlots_held R B desc q c₀) hown hreg hp hs hs0 hst' hpos' hu harm ht hent hl

end SpotFlock.Instance
