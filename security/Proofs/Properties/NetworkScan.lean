module

public import Properties.NetworkScan
public import Proofs.Properties.NetworkScan.Fits
public import Proofs.Properties.NetworkScan.Causal
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
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
import Mathlib.Tactic.ENatToNat
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
import Mathlib.Topology.Sheaves.Init
import Std.Tactic.BVDecide.Normalize.Prop

/-! An accepted scan's statements are in the sampled audit's scope, from its layout fitting; `scanFits_nontrivial` holds
at a circuit that returns nothing, and `scanScopeAcc_nontrivial` at one that returns only an input. -/

@[expose] public section

namespace Properties.NetworkScan

theorem scanScopeAcc_of_scanFits (h : ScanFits) : ScanScopeAcc := by
  intro j program s sc c hscan hc Coin _ _ _ A D x hflat hS hdesc
  refine ⟨hS, ?_⟩
  obtain ⟨st, _, hx, _⟩ := hS
  obtain ⟨P, hP, hF⟩ := hflat x st hx
  refine ⟨st, hx, ?_⟩
  unfold layOf
  rw [hP]
  rw [hdesc st hx] at hP
  exact h j program s sc c P st.prog hscan hc hP hF

/-- **An accepted scan's layout fits its Program flattened.** -/
theorem scanFits : ScanFits := by
  intro j program s sc c P C h hc hP hF
  obtain ⟨he, hi⟩ := Fits.scanFits_egress_ingress h hc hP hF
  exact ⟨he, hi, CausalProof.scanCausal j program s sc c P C h hc hP (Fits.scanOf_accepted h hc hP).topo hF⟩

/-- **An accepted scan's statements are in the sampled audit's scope.** -/
theorem scanScopeAcc : ScanScopeAcc := scanScopeAcc_of_scanFits scanFits

def oneSlot : NetworkCheck.Period.Schedule := ⟨0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 2, none, none, none, false⟩

theorem not_fits_of_no_outs {L S : ℕ} (lay : NetTiming.BoundaryGates L S) (gs : List Claims.Sampling.BGate) {k : ℕ}
    (hk : 0 < k) (ℓ : Fin L) (σ : Fin S) : ¬ lay.Fits ⟨gs, []⟩ k := by
  rintro ⟨h, -, -⟩
  have := h 0 hk ℓ σ _ (List.mem_cons.2 (Or.inl rfl))
  simp at this

theorem not_fits_of_outs_inputs {L S : ℕ} (lay : NetTiming.BoundaryGates L S) {C : Claims.Sampling.Prog} {k : ℕ}
    (hC : ∀ g ∈ C.outs, C.gates[g]? = some .input) (hk : 0 < k) (ℓ : Fin L) (σ : Fin S) : ¬ lay.Fits C k := by
  rintro ⟨h, -, -⟩
  obtain ⟨ho, hn⟩ := h 0 hk ℓ σ _ (List.mem_cons.2 (Or.inl rfl))
  exact hn (hC _ ho)

theorem empty_flattens : Flattens default ⟨[], []⟩ :=
  ⟨rfl, fun g hg => absurd hg (Nat.not_lt_zero g),
    fun g => ⟨fun h => by simp at h, fun ⟨j, hj, _⟩ => absurd hj (Nat.not_lt_zero j)⟩⟩

theorem canon_null : Flock.canon .null = .ok "null" := by
  simp only [Flock.canon]
  rfl

/-- **`ScanFits`' and `ScanScopeAcc`'s honest witness** (the two-lemma rule's `_honest`): of what the two take,
Flock's canonical form accepts a scan (`Flock.canon` at `null`), a Program has a flattening (`Flattens`, at the
Program with no definitions), and a compiled sampled verifier has a statement in its sampling scope
(`Claims.SamplingGeneric.Compiled.Scope`). The statement names no verifier; the proof stands at the toy verifier
`Claims.Witness.SamplingGeneric.twoL`, which samples both of its proof blocks and accepts only two empty messages.
The compiled sampled verifier's premise is a named assumption (`Properties.NetworkScan.Assumptions`).
`scanOf`'s and `Flock.Program.decode`'s acceptances are not exhibited: no descriptor is evaluated here. -/
theorem scan_honest : Flock.canon .null = .ok "null" ∧ Flattens default ⟨[], []⟩ ∧
    ∃ (A : Claims.SamplingGeneric.Arg Bool) (D : Claims.SamplingGeneric.Compiled Bool A) (x : Claims.Msg),
      D.Scope x :=
  ⟨canon_null, empty_flattens, _, Claims.Witness.SamplingGeneric.twoL Bool, [],
    Claims.Witness.SamplingGeneric.twoL_scope []⟩

/-- **`ScanFits`' conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a Program and a flattening of
it, built by hand, which a scan check accepting everything would pass, the layout doesn't fit. The Program has no
definitions, so its flattening returns nothing (`empty_flattens`, `not_fits_of_no_outs`). The instance is the model's:
it is built by hand, and no file decoding to it is exhibited. -/
theorem scanFits_nontrivial :
    ∃ (P : Flock.Program.Prog) (C : Claims.Sampling.Prog) (s : NetworkCheck.Period.Schedule) (_ : Flattens P C),
      ¬ (scanLayout P s).Fits C s.window :=
  ⟨default, ⟨[], []⟩, oneSlot, empty_flattens, not_fits_of_no_outs _ [] (by decide) 0 ⟨0, by decide⟩⟩

theorem bytes_of_data_toList (b : ByteArray) : (⟨b.data.toList.toArray⟩ : ByteArray) = b := rfl

def nullStmt : Claims.Sampling.AuditStatement :=
  { Claims.Witness.Sampling.toyStmt with desc := "null".toUTF8.data.toList }

theorem inScope_null : Claims.Sampling.InScope Claims.Witness.Sampling.toyPart nullStmt :=
  Claims.Witness.Sampling.inScope_toy

theorem null_outs_inputs : ∀ g ∈ nullStmt.prog.outs, nullStmt.prog.gates[g]? = some .input := by
  intro g hg
  obtain rfl : g = 0 := List.mem_singleton.1 hg
  rfl

noncomputable def acceptAllCompiled :
    Claims.SamplingGeneric.Compiled Bool (Claims.Witness.SamplingGeneric.toyA Bool) :=
  { Claims.Witness.SamplingGeneric.sksL Bool with parse := fun _ => some nullStmt }

theorem acceptAllCompiled_scope (x : Claims.Msg) : acceptAllCompiled.Scope x :=
  ⟨nullStmt, Claims.Witness.Sampling.toyPart, rfl, rfl, inScope_null⟩

/-- **`ScanScopeAcc`'s conclusion is not trivial** (the two-lemma rule's `_nontrivial`): at a scan Flock
canonicalizes as `null` and a compiled sampled verifier that accepts everything and whose statements have that text as
their descriptor, a statement in its sampling scope is not in `ScopeAcc` at a schedule (`oneSlot`): the only gate its
circuit returns is an input (`not_fits_of_outs_inputs`). -/
theorem scanScopeAcc_nontrivial :
    ∃ (s : NetworkCheck.Period.Schedule) (j : Lean.Json) (c : String) (_ : Flock.canon j = .ok c)
      (A : Claims.SamplingGeneric.Arg Bool) (D : Claims.SamplingGeneric.Compiled Bool A) (x : Claims.Msg)
      (_ : D.Scope x)
      (_ : ∀ st, D.parse x = some st → (⟨st.desc.toArray⟩ : ByteArray) = c.toUTF8),
      ¬ NetTiming.ScopeAcc D (layOf s) (fun _ => s.window) x := by
  refine ⟨oneSlot, .null, "null", canon_null, _, acceptAllCompiled, [], acceptAllCompiled_scope [],
    fun st hx => ?_, ?_⟩
  · cases (Option.some.inj hx : nullStmt = st)
    exact bytes_of_data_toList "null".toUTF8
  · rintro ⟨-, st, hx, hF⟩
    cases (Option.some.inj hx : nullStmt = st)
    exact not_fits_of_outs_inputs _ null_outs_inputs (by decide) 0 ⟨0, by decide⟩ hF

/-- **`ScanScopeAcc`'s conclusion is not trivial**: at any compiled sampled verifier, a statement that parses to a
descriptor decoding to the empty Program, with the empty circuit, is not in `ScopeAcc` at that Program's schedule. -/
theorem scanScopeAcc_nontrivial_of {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]
    {A : Claims.SamplingGeneric.Arg Coin} {D : Claims.SamplingGeneric.Compiled Coin A} {x : Claims.Msg}
    {st : Claims.Sampling.AuditStatement} (hx : D.parse x = some st)
    (hP : Flock.Program.decode ⟨st.desc.toArray⟩ = .ok default) (hC : st.prog = ⟨[], []⟩) :
    ¬ NetTiming.ScopeAcc D (layOf oneSlot) (fun _ => oneSlot.window) x := by
  rintro ⟨-, st', hx', hF⟩
  obtain rfl := Option.some.inj (hx.symm.trans hx')
  unfold layOf at hF
  rw [hP, hC] at hF
  exact not_fits_of_no_outs _ [] (show 0 < oneSlot.window by decide) 0 ⟨0, by decide⟩ hF

end Properties.NetworkScan
