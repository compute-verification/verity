import Properties.NetworkScan
import Proofs.Properties.NetworkScan.Accept
import Proofs.Properties.NetworkScan.Lowered
import Proofs.Properties.NetworkScan.Query
import Proofs.Properties.NetworkScan.OfProg
import Proofs.Flock.SamplingScope

/-!
# Toward an accepted scan's fit, at Bits' positions

`ScanFits` and `ScanScopeAcc` are stated at Bits' positions (`Properties.NetworkScan`): the statement's circuit is the
scan's audit's members' joint program (`jointOf`), its layout read through `SampledProofs.Bits.inBit` and `outBit`,
with no wires. What `scanOf`'s acceptance gives of the Program is `Fits.scanOf_accepted` (`Accept`), and what the joint
program holds at each position, its inputs, outputs and reads, is `Lowered`'s.

The layout's causality and wiring (`NetTiming.BoundaryGates.Fits`'s last two parts):

* `not_depends_instPos`: a placed instance's gate depends on no gate of an instance at another template node, since
  dependence never leaves a unit and the code locates a unit at one node.
* `egress_unitPos`, `ingress_unitPos`: the layout's egress gates are positions of the egress template's instance's
  unit, and a later step's ingress gates of the later ingress template's.
* `scan_causal`: where the egress gates are returned, no egress gate depends on a later step's ingress gate: a later
  step exists only with three templates, and its ingress is the middle one's.
* `depends_of_dependsWired`, `wiresFit_none`: with no wires, dependence is through reads and the wiring fits.

The facts both statements' witnesses read:

* `not_fits_of_no_outs`: a circuit that returns nothing fits no layout over a horizon of at least one step.
* `jointOf_nil`, `not_fits_of_no_members`: an audit of no members has the empty joint program, so no layout fits it.
* `oneSlot`, `canon_null`, `bytes_of_data_toList`: a schedule of one slot and one step, C-Flock's canonical `null`, and
  a byte array through its list of bytes.
* `nullStmt`, `acceptAllDeploy`: the statement whose descriptor is `null`'s bytes and whose circuit is empty, and a
  deployment whose scope check takes everything and which parses every message to it.
-/

namespace Properties.NetworkScan

open NetworkCheck.Period (Schedule templateNodes)
open SampledProofs.Bits (Members)

/-! ## Causality and wiring -/

/-- **A placed instance's gate depends on no gate of another node's instance**: where template `i`'s instance `t` is
placed inside the members' program at a unit position (`pos`), and template `i'`'s instance `t'` at a unit position or
past the program, at another root node, the first's gate doesn't depend on the second's. -/
theorem not_depends_instPos {P : Flock.Program.Prog} {rs : Array Flock.TemplateQuery.TRange} {ms : Members}
    {i t i' t' : ℕ} {pos pos' : Flock.Circuit → ℕ → ℕ → Option ℕ}
    (hpos : ∀ c n u x, pos c n u = some x → Lowered.UnitPos c n u x)
    (hpos' : ∀ c n u x, pos' c n u = some x → Lowered.UnitPos c n u x)
    (hk : (templateNodes P).getD i 0 ≠ (templateNodes P).getD i' 0)
    (hg : instPos P rs ms i t pos < (jointOf ms).gates.length) :
    ¬ NetTiming.Depends (jointOf ms) (instPos P rs ms i' t' pos') (instPos P rs ms i t pos) := by
  intro hd
  obtain ⟨m, hm, u, -, hl, x, hx, he⟩ := Query.instPos_lt (Nat.ne_of_lt hg)
  have hgu : Lowered.JointUnit ms m u (instPos P rs ms i t pos) := ⟨hm, x, he, hpos _ _ _ _ hx⟩
  by_cases hi : instPos P rs ms i' t' pos' = (jointOf ms).gates.length
  · rw [hi] at hd
    have := Lowered.eq_of_depends_of_le hd (Nat.le_refl _)
    omega
  · obtain ⟨m', hm', u', -, hl', x', hx', he'⟩ := Query.instPos_lt hi
    have hiu : Lowered.JointUnit ms m' u' (instPos P rs ms i' t' pos') := ⟨hm', x', he', hpos' _ _ _ _ hx'⟩
    obtain ⟨rfl, rfl⟩ := Lowered.jointUnit_unique (Lowered.depends_unit hd hiu) hgu
    exact hk (Query.locate_node_unique hl hl')

/-- **An egress gate is a position of the egress template's instance `t`'s unit.** -/
theorem egress_unitPos {P : Flock.Program.Prog} {s : Schedule} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Members} {t : ℕ} {ℓ : Fin 1} {σ : Fin s.allocation} {g : ℕ}
    (h : g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ)) :
    ∃ pos : Flock.Circuit → ℕ → ℕ → Option ℕ, (∀ c n u x, pos c n u = some x → Lowered.UnitPos c n u x) ∧
      g = instPos P rs ms ((templateNodes P).length - 1) t pos := by
  simp only [scanLayout, NetTiming.slotGates, List.mem_cons, List.mem_map, List.mem_range] at h
  rcases h with rfl | ⟨b, -, rfl⟩
  all_goals
    refine ⟨_, ?_, rfl⟩
    intro c n u x hx
    exact Lowered.outBit_unitPos hx

/-- **Step `t`'s ingress gate, past step 0, is a position of the later ingress template's instance `t − 1`'s unit.** -/
theorem ingress_unitPos {P : Flock.Program.Prog} {s : Schedule} {rs : Array Flock.TemplateQuery.TRange}
    {ms : Members} {t i : ℕ} (ht : t ≠ 0) (h : i ∈ (scanLayout P s rs ms).ingress t) :
    ∃ pos : Flock.Circuit → ℕ → ℕ → Option ℕ, (∀ c n u x, pos c n u = some x → Lowered.UnitPos c n u x) ∧
      i = instPos P rs ms 1 (t - 1) pos := by
  simp only [scanLayout, ht, ↓reduceIte, List.mem_map, List.mem_range] at h
  obtain ⟨ℓ, -, rfl⟩ := h
  refine ⟨_, ?_, rfl⟩
  intro c n u x hx
  exact Lowered.inBit_unitPos hx

theorem templateNodes_nodup (P : Flock.Program.Prog) : (templateNodes P).Nodup := by
  unfold templateNodes
  exact List.Pairwise.filter _ List.nodup_range

/-- With no wires, dependence through reads and wires is dependence through reads. -/
theorem depends_of_dependsWired {C : Claims.Sampling.Prog} {i g : ℕ}
    (h : NetTiming.DependsWired C (fun _ => none) i g) : NetTiming.Depends C i g := by
  unfold NetTiming.DependsWired at h
  induction h with
  | refl => exact Relation.ReflTransGen.refl
  | tail _ hbc ih => exact Relation.ReflTransGen.tail ih (Or.resolve_right hbc (by simp))

/-- With no wires, the wiring fits. -/
theorem wiresFit_none {L S : ℕ} {lay : NetTiming.BoundaryGates L S} {C : Claims.Sampling.Prog} {k : ℕ}
    (hw : lay.wires = fun _ => none) : lay.WiresFit C k := by
  refine ⟨fun g w h => ?_, ⟨fun _ => 0, fun g w g' h => ?_⟩⟩ <;> simp [hw] at h

/-- **The scan's layout is causal**: of an accepted scan, where its egress gates are returned, no egress gate of a step
depends on an ingress gate of a later step. A later step exists only where the Program has three templates; its
ingress is the middle template's instance, and every egress gate the last template's. -/
theorem scan_causal {P : Flock.Program.Prog} {s : Schedule} {rs : Array Flock.TemplateQuery.TRange} {ms : Members}
    (hacc : Fits.Accepted P s)
    (hout : ∀ t < s.window, ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ),
      g ∈ (jointOf ms).outs) :
    ∀ t < s.window, ∀ t' < s.window, t < t' → ∀ ℓ σ, ∀ g ∈ NetTiming.slotGates ((scanLayout P s rs ms).egress t ℓ σ),
      ∀ i ∈ (scanLayout P s rs ms).ingress t',
        ¬ NetTiming.DependsWired (jointOf ms) (scanLayout P s rs ms).wires i g := by
  intro t ht t' ht' htt' ℓ σ g hg i hi hd
  have hgl := Lowered.lt_length_of_mem_outs (hout t ht ℓ σ g hg)
  obtain ⟨pos, hpos, rfl⟩ := egress_unitPos hg
  obtain ⟨pos', hpos', rfl⟩ := ingress_unitPos (by omega) hi
  rcases hacc.shape with ⟨a, e, -, hT⟩ | ⟨a, b, e, hts⟩
  · omega
  · have hbe : b ≠ e := by
      intro hbe
      subst hbe
      have hnd := templateNodes_nodup P
      rw [hts] at hnd
      simp [List.nodup_cons] at hnd
    have hk : (templateNodes P).getD ((templateNodes P).length - 1) 0 ≠ (templateNodes P).getD 1 0 := by
      rw [hts]
      simpa using hbe.symm
    exact not_depends_instPos hpos hpos' hk hgl (depends_of_dependsWired hd)

/-- One slot of one lane and a window of one step. -/
def oneSlot : NetworkCheck.Period.Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1⟩

/-- A circuit that returns nothing fits no layout over a horizon of at least one step: step 0's egress presence gate
isn't returned. -/
theorem not_fits_of_no_outs {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (gs : List Claims.Sampling.BGate) {k : ℕ}
    (hk : 0 < k) (ℓ : Fin L) (σ : Fin S) : ¬ lay.Fits ⟨gs, []⟩ k := by
  rintro ⟨h, -, -⟩
  have := h 0 hk ℓ σ _ (List.mem_cons.2 (Or.inl rfl))
  simp at this

/-- An audit of no members has the empty joint program. -/
theorem jointOf_nil : jointOf [] = ⟨[], []⟩ := rfl

/-- An audit of no members fits no layout over a horizon of at least one step. -/
theorem not_fits_of_no_members {L S : ℕ} (lay : NetTiming.BoundaryGates L S) {k : ℕ} (hk : 0 < k) (ℓ : Fin L)
    (σ : Fin S) : ¬ lay.Fits (jointOf []) k := by
  rw [jointOf_nil]
  exact not_fits_of_no_outs lay [] hk ℓ σ

theorem canon_null : Flock.canon .null = .ok "null" := by
  simp only [Flock.canon]
  rfl

theorem bytes_of_data_toList (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := rfl

/-- The statement whose descriptor is the bytes of `null` and whose circuit is empty. -/
def nullStmt : Claims.Sampling.AuditStatement :=
  { Claims.Sampling.ScopeWitness.toyStmt with desc := "null".toUTF8.data.toList, prog := ⟨[], []⟩ }

/-- A deployment whose scope check takes everything and which parses every message to `nullStmt`. -/
def acceptAllDeploy : Claims.Sampling.Deploy Bool ℕ Unit :=
  { Claims.Sampling.ScopeWitness.toyD with parse := fun _ => some nullStmt }

end Properties.NetworkScan
