import FlockSoundness.Discharge.PrivateCircuit.ReadsHidden
import Mathlib.Algebra.MvPolynomial.SchwartzZippel

/-!
# Route P: hidden wiring over a public table of gates

Route P hides a circuit whose proof units all differ, with no recursion. Every proof unit runs the same public unit circuit,
a table of gate rows (each row one AND and one XOR gate in a fixed layout) over `nc` local cells; what differs between
units, and is hidden, is each unit's **wiring** `σ_u : Fin nc → Fin nc`, whose cycles are the unit's nets (one source cell
and its consumers). The wiring is committed at registration in the unit's topology entry, with the store cells the unit
reads (`hiddenPop` at `routeCircuits`). Each audit checks, by a grand-product argument over the drawn units' cells, that
the committed cell values follow the wiring: cells on one cycle carry one value.

* **`WiringSound`** (`WireId`, `Wired`, `wirePoly`): over any field `F` and any finite set `ι` of cells, tagged by an
  injective `lab : ι → F`, the identity `∏ᵢ (wᵢ + β·lab i + γ) = ∏ᵢ (wᵢ + β·lab (σ i) + γ)` holds as a polynomial in
  `(β, γ)` iff `w ∘ σ = w` (`wirePoly_eq_zero_iff`, for a permutation; `wired_of_wirePoly_eq_zero` for any function,
  which is what a check of a committed `σ` sees), and `w ∘ σ = w` iff `w` is constant on `σ`'s cycles
  (`wired_iff_cycles`). `wiring_sound`: when `w ∘ σ ≠ w`, the identity holds at uniform `(β, γ)` with probability at most
  `|ι|/|F|` (Schwartz–Zippel, total degree `|ι|`). `wiring_escape` is the form the protocol uses: whatever the prover
  fixes before the verifier draws `(β, γ)`, accepting the check while its values do not follow its wiring has probability
  at most `|ι|/|F|`. `wired_diag`: the batched check over several units' cells, tagged by (unit, local cell), checks
  each unit's wiring.
* **`route_p_sound`**: protocol 2 with route P's units. It is `private_circuit_sound_reads` at `routeCircuits`, with
  the block conjunct of the class relation replaced by route P's unit check (`RouteRel`: the public gate rows hold on the
  unit's committed cells, which carry its reads and writes) and one more term, the wiring check's escape `εW` (`hW`,
  `wiring_escape` at the session's coins over the drawn units' cells). A wrong owed output means a drawn unit wrong in
  the committed values (`ε_ks + δ_link`), the draw missing the units wrong against the store (the law's escape), a drawn
  unit whose cells do not follow its wiring with the check passed (`εW`), or a SHA-512 collision on the store or the
  table. The draw is over units and reads only the verifier's coins and the public class: one size class, one public
  unit circuit, so the law has no strata by circuit (`s.Lm` is any law over the units).
* What `σ` computes is the unit circuit's reference semantics `runP σ`, at the registered wirings only (`Determines`):
  a cyclic wiring can leave the cells under-determined, so where the registration does not prove that each consumer reads
  an earlier source, the owed outputs are judged against `runP` at whatever `Determines` holds of.

Route P's privacy rests on the registered commitment's hiding over the openings one registration allows
(`Assumptions.RegHides`, budgeted per registration), stated and not proved here.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

namespace RouteP

open Game
open scoped ENNReal

section Wiring

variable {F : Type} [_root_.Field F] {ι : Type} [Fintype ι] [DecidableEq ι]

/-- **The grand-product identity** at the verifier's coins `β, γ`: the cells' values `w` tagged by their own labels and by
their wiring's labels give one product. -/
def WireId (lab : ι → F) (w : ι → F) (σ : ι → ι) (β γ : F) : Prop :=
  ∏ i, (w i + β * lab i + γ) = ∏ i, (w i + β * lab (σ i) + γ)

/-- **The values follow the wiring**: every cell carries the value of the cell its wiring sends it to. -/
def Wired {α : Type} (w : ι → α) (σ : ι → ι) : Prop :=
  ∀ i, w (σ i) = w i

/-- **The identity as a polynomial**, in `γ = X 0` and `β = X 1`: the two products' difference. -/
noncomputable def wirePoly (lab w : ι → F) (σ : ι → ι) : MvPolynomial (Fin 2) F :=
  ∏ i, (MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (lab i) * MvPolynomial.X 1) -
    ∏ i, (MvPolynomial.X 0 + MvPolynomial.C (w i) + MvPolynomial.C (lab (σ i)) * MvPolynomial.X 1)

/-- **Following a permutation's wiring is being constant on its cycles.** -/
theorem wired_iff_cycles {α : Type} (w : ι → α) (σ : Equiv.Perm ι) :
    Wired w σ ↔ ∀ (i : ι) (k : ℕ), w ((σ ^ k) i) = w i := by
  sorry

/-- The polynomial vanishes at `(γ, β)` exactly when the identity holds there. -/
theorem wirePoly_eval (lab w : ι → F) (σ : ι → ι) (β γ : F) :
    MvPolynomial.eval ![γ, β] (wirePoly lab w σ) = 0 ↔ WireId lab w σ β γ := by
  sorry

/-- **The identity as a polynomial forces the wiring**, for any committed `σ`, with injective labels. -/
theorem wired_of_wirePoly_eq_zero (lab w : ι → F) (σ : ι → ι) (hlab : Function.Injective lab)
    (h : wirePoly lab w σ = 0) : Wired w σ := by
  sorry

/-- **At a permutation, the identity holds as a polynomial iff the values follow the wiring.** -/
theorem wirePoly_eq_zero_iff (lab w : ι → F) (σ : Equiv.Perm ι) (hlab : Function.Injective lab) :
    wirePoly lab w σ = 0 ↔ Wired w σ := by
  sorry

/-- **Completeness**: values that follow a permutation's wiring pass at every coin. -/
theorem wireId_of_wired (lab w : ι → F) (σ : Equiv.Perm ι) (h : Wired w σ) (β γ : F) : WireId lab w σ β γ := by
  sorry

/-- **The batched check's wiring is each unit's**: on cells tagged (unit, local cell), the block-diagonal wiring
`(u, i) ↦ (u, σ u i)` is followed iff every unit's own wiring is. -/
theorem wired_diag {U κ α : Type} (c : U → κ → α) (σ : U → κ → κ) :
    Wired (fun p : U × κ => c p.1 p.2) (fun p => (p.1, σ p.1 p.2)) ↔ ∀ u, Wired (c u) (σ u) := by
  sorry

variable [Fintype F] [DecidableEq F]

/-- **`WiringSound`**: values that do not follow the wiring pass the identity at uniform coins `(β, γ)` with probability
at most `|ι|/|F|`, for any committed `σ` and injective labels. -/
theorem wiring_sound (lab w : ι → F) (σ : ι → ι) (hlab : Function.Injective lab) (h : ¬ Wired w σ) :
    prCoin (fun c : F × F => WireId lab w σ c.1 c.2) ≤ (Fintype.card ι : ℝ≥0∞) / Fintype.card F := by
  sorry

/-- **The wiring check's escape over the verifier's coins**: in any game `g` after which the verifier draws `(β, γ)`,
and for any prover, the outcome's values `w` and wiring `σ` (whatever `g` fixed) not following each other while the
identity holds at the drawn coins has probability at most `|ι|/|F|`. -/
theorem wiring_escape {α : Type} (lab : ι → F) (hlab : Function.Injective lab) (w : α → ι → F) (σ : α → ι → ι)
    (g : Game α) (s : Strategy (g.bind fun a => Game.coin (F × F) fun c => .ret (a, c))) :
    prob (fun x : α × (F × F) => ¬ Wired (w x.1) (σ x.1) ∧ WireId lab (w x.1) (σ x.1) x.2.1 x.2.2)
        (g.bind fun a => Game.coin (F × F) fun c => .ret (a, c)) s
      ≤ (Fintype.card ι : ℝ≥0∞) / Fintype.card F := by
  sorry

end Wiring

end RouteP

namespace Reads

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit RouteP

/-- **Route P's unit circuits**: a unit circuit is a wiring of `nc` local cells over the public gate table, run by its
reference semantics `runP`. -/
abbrev routeCircuits (F : Type) (nc nIn nOut : ℕ) (runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F) :
    UnitCircuits F nIn nOut where
  UC := Fin nc → Fin nc
  gates _ := nc
  run := runP

/-- **A wiring determines its outputs**: any cells on which the public gate rows hold (`gok`) and which follow the wiring
`σ` carry `runP σ` of their input cells at their output cells. -/
def Determines {F : Type} {nc nIn nOut : ℕ} (gok : (Fin nc → F) → Prop) (inC : Fin nIn → Fin nc) (outC : Fin nOut → Fin nc)
    (runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F) (σ : Fin nc → Fin nc) : Prop :=
  ∀ c : Fin nc → F, gok c → Wired c σ → ∀ l, c (outC l) = runP σ (fun k => c (inC k)) l

/-- **Route P's class relation at unit `u`**: `ClassRel`'s reads (the entry read opens the table root at `u` and is acyclic,
each input read opens the store root at the cell the entry names, each output write at `u`'s output row), with the block
conjunct replaced by the unit check: the public gate rows hold on `u`'s committed cells `cl u` (`gok`), whose input cells
carry its input reads and whose output cells its output writes. The wiring is not checked here: the batched check over the
drawn units' cells does that (`wiring_escape`). -/
def RouteRel {F : Type} {nc nIn nOut N G M : ℕ} {runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F}
    (L : ReadLayout (hiddenPop (routeCircuits F nc nIn nOut runP) N) G)
    (lfT : ExplicitLeaf (Entry (hiddenPop (routeCircuits F nc nIn nOut runP) N))) (lfS : ExplicitLeaf F)
    (idx : Cell → Fin M) (gok : (Fin nc → F) → Prop) (inC : Fin nIn → Fin nc) (outC : Fin nOut → Fin nc)
    (cl : Fin N → Fin nc → Reads.Field G F) (rtT rtS : Flock.Registered.Port) (v : Fin G → Bool) (u : Fin N) : Prop :=
  ReadOk lfT rtT u ((L.ent u).read v) ∧ (L.tabOf v u).Acyclic u ∧
    (∀ k, ReadOk lfS rtS (idx ((L.tabOf v u).src k)) ((L.inp u k).read v)) ∧
    (∀ l : Fin nOut, ReadOk lfS rtS (idx (.out u l)) ((L.out u l).read v)) ∧
    gok (fun i => (cl u i).read v) ∧ (∀ k, (cl u (inC k)).read v = L.xOf v u k) ∧
    ∀ l, (cl u (outC l)).read v = L.yOf v u l

end Reads

section Sound

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit Reads
  RouteP
open scoped ENNReal

variable {F : Type} [Inhabited F] {nc nIn nOut N : ℕ} {runP : (Fin nc → Fin nc) → (Fin nIn → F) → Fin nOut → F}
  {C : Audit.Circuit} {Pt : Audit.Partition C N} {Reg : Type} {M : ℕ}

open Classical in
/-- **Route P is sound at the drawn units.** Protocol 2 at `hiddenPop (routeCircuits F nc nIn nOut runP) N`: each unit's
topology entry holds its wiring `σ_u` and the store cells it reads, committed under the table root; the class net computes
route P's unit relation at each unit's check gate (`hcomp`: reads, public gate rows on the unit's committed cells `cl u`,
and those cells carrying its reads and writes); every registered wiring determines its outputs (`hdet`); and the wiring
check over the drawn units' cells escapes with at most `εW` (`hW`: accepting while a drawn unit not wrong in the committed
values has cells not following its entry's wiring). The rest is `private_circuit_sound_reads`'s. Then, with `T₀`, `w` the
registered table and store: (1) a delivered value off the committed one is at most `q_S²/2^513`; (2) if the registered
circuit's owed outputs are wrong, accepting is at most `escape(bad T₀ w) + ε_ks + δ_link + εW + q_S²/2^513 +
q_T²/2^513`; (3) a delivered public input off the committed cell or a delivered output off the registered circuit's end
to end on the committed cells is at most `q_S²/2^513` plus, when the owed outputs are wrong,
`escape(bad T₀ w) + ε_ks + δ_link + εW + q_T²/2^513`. -/
theorem route_p_sound
    (L : ReadLayout (hiddenPop (routeCircuits F nc nIn nOut runP) N) C.N)
    (lfT : ExplicitLeaf (Entry (hiddenPop (routeCircuits F nc nIn nOut runP) N))) (lfS : ExplicitLeaf F)
    (idx : Cell → Fin M) (gok : (Fin nc → F) → Prop) (inC : Fin nIn → Fin nc) (outC : Fin nOut → Fin nc)
    (cl : Fin N → Fin nc → Reads.Field C.N F)
    (rootT rootS : Reg → Flock.Registered.Port) (good : Reg → (Fin C.N → Bool) → Prop) (ok : Fin N → Fin C.N)
    (hcomp : ClassComputes Pt good (fun R v u => RouteRel L lfT lfS idx gok inC outC cl (rootT R) (rootS R) v u) ok)
    (rtT₀ rtS₀ : Flock.Registered.Port)
    (tr : Fin N → ReadVal (Entry (hiddenPop (routeCircuits F nc nIn nOut runP) N))) (wo : Fin M → ReadVal F)
    (htr : ∀ j : Fin N, ReadOk lfT rtT₀ j (tr j)) (hwo : ∀ i : Fin M, ReadOk lfS rtS₀ i (wo i))
    (hdet : ∀ j : Fin N, Determines gok inC outC runP (tr j).val.dims)
    (s : UnitSession Pt Reg) (drawn : s.Out → Finset (Fin N)) (hD : ZkSessionDrawn s drawn)
    (hE : DrawEscapes s drawn) (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (τ : Strategy s.G) (hCR : s.CR τ) (hgood : ∀ o, s.Accepts o → good (s.regOf o) (s.vals τ o))
    (εW : ℝ≥0∞)
    (hW : prob (fun o => s.Accepts o ∧ ∃ u ∈ drawn o, u ∉ Pt.wrong (s.vals τ o) ∧
        ¬ Wired (fun i => (cl u i).read (s.vals τ o)) (L.tabOf (s.vals τ o) u).dims) s.G τ ≤ εW)
    (owed : Finset (Fin N × Fin nOut)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin N × Fin nOut → ReadVal F) (xD : s.Out → ℕ → ReadVal F)
    (hdel : ∀ o, s.Accepts o →
      (∀ p ∈ owed, ReadOk lfS (rootS (s.regOf o)) (idx (.out p.1 p.2)) (yD o p)) ∧
        ∀ m ∈ pubIn, ReadOk lfS (rootS (s.regOf o)) (idx (.reg m)) (xD o m))
    (costS costT : s.Out → ℝ) (qS qT : ℝ) (hcS : ∀ o, costS o ≤ qS) (hcT : ∀ o, costT o ≤ qT)
    (hS : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some (readPick lfS rtS₀ Fin.val wo (storeReads L idx s τ drawn owed pubIn yD xD) o)) costS qS)
    (hT : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some ((regCommit lfT N).ext (rtT₀, tabFill L s τ drawn tr o) (rtT₀, tr))) costT qT) :
    let w : Cell → F := fun c => (wo (idx c)).val
    let T₀ : Table (hiddenPop (routeCircuits F nc nIn nOut runP) N) := fun j => (tr j).val
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) ∧
    (OwedWrongR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) + (s.εks τ + s.δlink τ) + εW +
          ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))) ∧
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨
          ∃ p ∈ owed, (yD o p).val ≠ evalR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ (fun m => w (.reg m))
            p.1 p.2)) s.G τ
      ≤ (if OwedWrongR (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w owed then
          s.Lm.escape (bad (hiddenPop (routeCircuits F nc nIn nOut runP) N) T₀ w) + (s.εks τ + s.δlink τ) + εW +
            ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))
          else 0) +
        ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end Sound

end FlockSoundness.Discharge.PrivateCircuit
