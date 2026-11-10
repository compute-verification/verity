module

public import Proofs.Flock.SpotFlock.Instance
public import Proofs.Flock.SpotFlock.EntriesAt
public import Proofs.Flock.Claims.Sampling.Select.Second

/-!
# Where every accepted run of the two-stage instance stands

At `SpotFlock.Instance.held R B desc q c₀`, for every `R` and `B`, every accepted run (`CheckedRun.Ok`) has:
* the replay sample `#[0]` (`sample_ok`): round 2 sampled the second stage (`secondDraw`), which `Select.secondProblem`
  lets sample only over a nonempty, ascending sample, and the first stage samples unit slots (`firstSlots_held`) of a
  layout of one proof block (`sample_getD`);
* replay block `0` (`rho_ok`), the replay partition's one, and a unit below two (`u_lt_two`);
* its unit in template `0`, whose part is two units from unit `0` (`drawnPart_ok`);
and round 2 reads no wire at any port (`wireOf_held`), and the design gives the instance's rows one a replay block
at every port it gives rows (`per_held`).
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

/-- **The instance's rows are one a replay block**, at every port the design gives rows. -/
theorem per_held {st rec draw c} {t port : String} {ro : Spot.SecondStage.Rows}
    (hro : Spot.SecondStage.rowsOf (ownOf (held R B desc q c₀) st rec draw c) t port = some ro) : ro.per = 1 := by
  have hm : ro ∈ R.links.map (fun l => (⟨l.template, l.port, 1, #[⟨0, 2, 0, 0⟩]⟩ : Spot.SecondStage.Rows)) :=
    Array.mem_of_find?_eq_some hro
  obtain ⟨l, -, rfl⟩ := Array.mem_map.1 hm
  rfl

end SpotFlock.Instance
