module

public import Definitions.Claims.SamplingServed
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.Claims.Deploy
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
import Mathlib.Topology.Sheaves.Init

/-!
# The two-stage compiler's statement says something

`CompilerKSTwo` and `CompilerExfilTwo` are guarantees that the lock's `claims.generic` names, each held to two
witnesses: an honest one, whose conclusion is its premises holding together at one instance with two sampled sessions
(both read `compilerKSTwo_honest`: they have the same premises), and a nontrivial one, at which its conclusion fails.
Each is at coins `Bool` and the hash SHA-512 (`FlockDeploy.sha512`), on the toys of `Witnesses.SamplingGeneric`; the
claims `CompiledTwo.KS` and `CompiledTwo.Exfil` they read have their `fails` witnesses here too.

* `sksLTwo`: `sksL` (accepts after one round, samples nothing, opens nothing) with the two-stage design `toyTwo` (its
  replay partition the proof-block partition `{0}`, its first stage `all`, `k = 1`). Its one proof block is incorrect at
  any interior values, so its footprint is at least `1` at every interior, the cell's escape `0` and `P̄_a = 0`: no witness is in the
  relation at `δ = 0`, and `CompiledTwo.KS` fails (`compiledTwoKS_fails`). Its influence set is then empty, so no
  witness is in the influence relation either, and `CompiledTwo.Exfil` fails (`compiledTwoExfil_fails`).
* `twoLTwo`: `twoL` (both proof blocks of `twoStmt` sampled, block `u`'s session at round `u`) with the design `twoTwo`
  (replay blocks the proof blocks `{0}`, `{1}`, first stage `all`, `k = 1`), sampling both replay blocks. It meets
  `CompiledTwo.Sessions` at every hash (`twoSessionsTwo_meets`): the first stage misses a set only when it is empty, and
  the second stage misses `X` only when `X` holds no proof block, where each cell escapes.
* `CapTwo`'s premises hold at `toyTwo`, both rates `1` (`capTwo_honest`), and its conclusion fails at `zeroTwo`, the toy
  with a second stage that samples nothing (`capTwo_nontrivial`).
-/

@[expose] public section

namespace Claims.Witness.SamplingGeneric

open Claims.SamplingGeneric
open Claims.Sampling (forkOn InScopeTwo cell escCell footprint wrongIn)
open Claims.Witness.Sampling (rejCm rejCm_val toyStmt toyPart toyPart_blocks inScope_toy toy_Pa)

/-! ## A two-stage verifier that accepts with an empty relation -/

/-- The toy's two-stage design: its replay partition the proof-block partition, its first stage `all`, `k = 1`. -/
def toyTwo : Claims.Sampling.Design.Two := ⟨toyPart, toyStmt.design, [1]⟩

/-- `sksL` with the design `toyTwo`, its first stage sampling no replay block. -/
noncomputable def sksLTwo (Coin : Type) : CompiledTwo Coin (toyA Coin) where
  toCompiled := sksL Coin
  two _ := some toyTwo
  replaySample _ _ := []

theorem inScopeTwo_toy : InScopeTwo toyPart toyTwo toyStmt :=
  ⟨inScope_toy.1, inScope_toy.2.1, by decide, by decide, ⟨inScope_toy.2.2, by decide, by decide⟩⟩

theorem sksLTwo_scope {Coin : Type} (x : Msg) : (sksLTwo Coin).Scope x :=
  ⟨toyStmt, toyPart, toyTwo, rfl, rfl, rfl, inScopeTwo_toy⟩

/-- The proof block is incorrect under any witness at any interior values: `rejCm` opens nothing. -/
theorem toy_wrongIn (H : Msg → Msg) (w : Claims.Sampling.Wit) (ι : ℕ → Bool) :
    0 ∈ wrongIn H rejCm toyStmt toyPart toyPart w 0 0 ι := by
  classical
  unfold wrongIn
  refine Finset.mem_filter.2 ⟨by decide, ?_⟩
  rintro ⟨u, r, hu, hr, hc⟩
  rw [toyPart_blocks] at hu hr
  simp only [List.getElem?_cons_zero, Option.some.injEq] at hu hr
  subst hu hr
  obtain ⟨b, hb⟩ := toy_correct_val _ ι hc
  rw [rejCm_val] at hb
  cases hb

theorem toy_footprint (H : Msg → Msg) (w : Claims.Sampling.Wit) (ι : ℕ → ℕ → Bool) :
    1 ≤ footprint H rejCm toyStmt toyPart toyPart w ι 0 0 :=
  Finset.card_pos.2 ⟨0, toy_wrongIn H w (ι 0)⟩

theorem escCell_one_one {m : ℕ} (hm : 1 ≤ m) : escCell 1 1 m = 0 := by
  simp [escCell, Nat.sub_eq_zero_of_le hm]

theorem toyTwo_esc {fp : ℕ → ℕ → ℕ} (h : 1 ≤ fp 0 0) : toyTwo.esc toyPart fp 0 = 0 := by
  unfold Claims.Sampling.Design.Two.esc
  refine Finset.prod_eq_zero (i := 0) (by decide) ?_
  have hc : (cell toyPart toyTwo.part 0 0).card = 1 := by decide
  show escCell (cell toyPart toyTwo.part 0 0).card (toyTwo.k.getD 0 0) (fp 0 0) = 0
  rw [hc]
  exact escCell_one_one h

/-- At a footprint with an incorrect proof block, `P̄_a = 0`: the first stage samples the one replay block, whose
escape is `0`. -/
theorem toyTwo_Pa {fp : ℕ → ℕ → ℕ} (h : 1 ≤ fp 0 0) : toyTwo.Pa toyPart fp = 0 := by
  unfold Claims.Sampling.Design.Two.Pa
  refine Finset.sum_eq_zero fun T _ => ?_
  by_cases h0 : 0 ∈ T
  · have hw : Claims.Sampling.window 0 1 T ≠ ∅ := by
      intro hw
      have h0w : 0 ∈ Claims.Sampling.window 0 1 T := by simp [Claims.Sampling.window, h0]
      rw [hw] at h0w
      simp at h0w
    have hPa : toyTwo.replay.Pa toyTwo.part T = 0 := by
      show toyStmt.design.Pa toyPart T = 0
      rw [toy_Pa]
      simp [hw]
    rw [hPa, zero_mul]
  · have he : ∏ ρ ∈ Finset.range toyTwo.part.blocks.length \ T, toyTwo.esc toyPart fp ρ = 0 :=
      Finset.prod_eq_zero (Finset.mem_sdiff.2 ⟨by decide, h0⟩) (toyTwo_esc h)
    rw [he, mul_zero, mul_zero]

/-- No witness is in `sksLTwo`'s relation at `δ = 0`. -/
theorem sksLTwo_noR {Coin : Type} (H : Msg → Msg) (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ Claims.Sampling.RTwo H (sksLTwo Coin).commit (sksLTwo Coin).parse (sksLTwo Coin).partition (sksLTwo Coin).two 0
      x w := by
  rintro ⟨s, Pt, d, hs, hPt, hd, ι, hM⟩
  obtain rfl : s = toyStmt := (Option.some.inj hs).symm
  obtain rfl : Pt = toyPart := (Option.some.inj hPt).symm
  obtain rfl : d = toyTwo := (Option.some.inj hd).symm
  have hlt : (0 : ℝ) < toyTwo.Pa toyPart (footprint H rejCm toyStmt toyPart toyPart w ι) := hM
  rw [toyTwo_Pa (toy_footprint H w ι)] at hlt
  exact lt_irrefl _ hlt

/-- **`CompiledTwo.KS` fails** at `sksLTwo`, at `δ = 0`. -/
theorem compiledTwoKS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).KS H := by
  intro h
  have h1 := h 0 le_rfl [] (sksLTwo_scope []) Unit silentP
  have hb := sksL_bound_fails (Coin := Coin) H
    (Claims.Sampling.RTwo H (sksLTwo Coin).commit (sksLTwo Coin).parse (sksLTwo Coin).partition (sksLTwo Coin).two 0)
    ((sksL Coin).κ 0 []) [] (sksLTwo_noR H []) (by linarith [sksL_κ (Coin := Coin) []])
  exact hb h1

/-- No witness is in `sksLTwo`'s influence relation at `δ = 0`: a served output in the influence set is some boundary
trace's in `RTwo`, and there is none. -/
theorem sksLTwo_noRInfl {Coin : Type} (H : Msg → Msg) (x : Msg) (w : Claims.Sampling.Wit) :
    ¬ Claims.Sampling.RInflTwo H (sksLTwo Coin).commit (sksLTwo Coin).parse (sksLTwo Coin).partition
      (sksLTwo Coin).two 0 x w := by
  classical
  rintro ⟨s, Pt, d, hs, hPt, hd, hy⟩
  unfold Claims.Sampling.influenceTwo at hy
  obtain ⟨w', _, ⟨ι, hM⟩, _⟩ := (Finset.mem_filter.1 hy).2
  exact sksLTwo_noR H x w' ⟨s, Pt, d, hs, hPt, hd, ι, hM⟩

/-- **`CompiledTwo.Exfil` fails** at `sksLTwo`, at `δ = 0`: its knowledge-soundness half does. -/
theorem compiledTwoExfil_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ¬ (sksLTwo Coin).Exfil H := by
  intro h
  have h1 := (h 0 le_rfl).1 [] (sksLTwo_scope []) Unit silentP
  have hb := sksL_bound_fails (Coin := Coin) H
    (Claims.Sampling.RInflTwo H (sksLTwo Coin).commit (sksLTwo Coin).parse (sksLTwo Coin).partition
      (sksLTwo Coin).two 0)
    ((sksL Coin).κ 0 []) [] (sksLTwo_noRInfl H []) (by linarith [sksL_κ (Coin := Coin) []])
  exact hb h1

/-! ## Two sampled sessions meet the two-stage contract -/

/-- The two-session toy's two-stage design: replay blocks the proof blocks `{0}` and `{1}`, its first stage `all`,
`k = 1`. -/
def twoTwo : Claims.Sampling.Design.Two := ⟨twoPart, twoStmt.design, [1]⟩

/-- `twoL` with the design `twoTwo`, its first stage sampling both replay blocks. -/
def twoLTwo (Coin : Type) : CompiledTwo Coin (twoA Coin) where
  toCompiled := twoL Coin
  two _ := some twoTwo
  replaySample _ _ := [0, 1]

theorem inScopeTwo_two : InScopeTwo twoPart twoTwo twoStmt :=
  ⟨inScope_two.1, inScope_two.2.1, by decide, by decide, ⟨inScope_two.2.2, by decide, by decide⟩⟩

theorem twoLTwo_scope {Coin : Type} (x : Msg) : (twoLTwo Coin).Scope x :=
  ⟨twoStmt, twoPart, twoTwo, rfl, rfl, rfl, inScopeTwo_two⟩

/-- `twoLTwo` meets the two-stage contract at every hash. -/
theorem twoLTwo_sessions {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    (twoLTwo Coin).Sessions H where
  replayDraws := by
    intro x s Pt d hs hP hd _ T hT
    obtain rfl : s = twoStmt := (Option.some.inj hs).symm
    obtain rfl : Pt = twoPart := (Option.some.inj hP).symm
    obtain rfl : d = twoTwo := (Option.some.inj hd).symm
    by_cases h0 : T = ∅
    · subst h0
      have h1 : twoStmt.design.Pa twoPart ∅ = 1 := by rw [two_Pa]; simp [Claims.Sampling.window]
      exact (Pr_le_one _).trans_eq h1.symm
    · obtain ⟨v, hv⟩ := Finset.nonempty_iff_ne_empty.2 h0
      have hT' : T ⊆ Finset.range twoPart.blocks.length := hT
      rw [twoPart_blocks] at hT'
      have hv2 : v < 2 := Finset.mem_range.1 (hT' hv)
      have hn : ∀ r : Fin ((twoLTwo Coin).V.k x) → Coin, ¬ ∀ ρ ∈ (twoLTwo Coin).replaySample x r, ρ ∉ T := by
        intro r hall
        refine hall v ?_ hv
        show v ∈ [0, 1]
        interval_cases v <;> simp
      refine (Pr_congr fun r => iff_false_intro (hn r)).trans_le ?_
      rw [Pr_false]
      exact two_Pa_nonneg T
  secondDraws := by
    intro x s Pt d hs hP hd _ r X
    obtain rfl : s = twoStmt := (Option.some.inj hs).symm
    obtain rfl : Pt = twoPart := (Option.some.inj hP).symm
    obtain rfl : d = twoTwo := (Option.some.inj hd).symm
    by_cases hX : ∃ v ∈ X, v < 2
    · obtain ⟨v, hvX, hv2⟩ := hX
      have hn : ∀ e : Fin ((twoLTwo Coin).V.k x) → Coin,
          ¬ ∀ u ∈ (twoLTwo Coin).checked x (forkOn ((twoLTwo Coin).stage2 x) r e), u ∉ X := by
        intro e hall
        refine hall v ?_ hvX
        show v ∈ [0, 1]
        interval_cases v <;> simp
      refine (Pr_congr fun e => iff_false_intro (hn e)).trans_le ?_
      rw [Pr_false]
      exact Finset.prod_nonneg fun ρ _ => Finset.prod_nonneg fun t _ => Claims.Sampling.escCell_nonneg _ _ _
    · have hc : ∀ ρ t, X ∩ cell twoPart twoTwo.part ρ t = ∅ := by
        intro ρ t
        refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
        obtain ⟨hvX, hvc⟩ := Finset.mem_inter.1 hv
        have hvlt : v < twoPart.blocks.length := Finset.mem_range.1 (Claims.Sampling.cell_subset _ _ _ _ hvc)
        rw [twoPart_blocks] at hvlt
        exact hX ⟨v, hvX, hvlt⟩
      refine (Pr_le_one _).trans (le_of_eq (Finset.prod_eq_one fun ρ _ => Finset.prod_eq_one fun t _ => ?_).symm)
      rw [hc ρ t, Finset.card_empty, Claims.Sampling.escCell_zero]
  slots _ _ := le_rfl
  unions x s Pt d hs hP _ _ :=
    (twoLTwo Coin).unions_of_nPU_le (fun _ => le_rfl) (fun _ => le_rfl) x s Pt d hs hP
  refuses x _ _ _ _ _ := twoLTwo_scope x
  interior _ _ _ _ _ _ _ := rfl
  fixed := (twoL_sessions (Coin := Coin) H).fixed
  runs := (twoL_sessions (Coin := Coin) H).runs
  opens := by
    intro x s Pt d Ω Pv ω r u xS vS w hs hP hd _ _ hu _ _
    obtain rfl : s = twoStmt := (Option.some.inj hs).symm
    obtain rfl : Pt = twoPart := (Option.some.inj hP).symm
    obtain rfl : d = twoTwo := (Option.some.inj hd).symm
    have hu2 : u < 2 := by
      change u ∈ [0, 1] at hu
      simp at hu
      omega
    have hb : twoPart.blocks[u]? = some [u] := by rw [twoPart_blocks]; interval_cases u <;> rfl
    refine ⟨[u], u, [u], hb, by interval_cases u <;> decide, hb, fun val ι hq => ?_⟩
    have hv := (hq (twoStmt.roots, ⟨u, true, []⟩) (List.mem_singleton_self _) rfl).1 rfl
    exact two_correct hu2 val ι hv

/-- **Two sampled sessions meet the two-stage contract** (`twoLTwo`, at every hash): over a knowledge-sound proof
system (`twoA`) with a binding commitment, a compiled two-stage verifier whose first stage samples both replay blocks of
a statement in scope and whose second stage samples both proof blocks, each block's session at its own round, meets
`CompiledTwo.Sessions`, and the silent prover is accepted. So `CompilerKSTwo` reads antecedents that hold with more than
one sampled session. -/
theorem twoSessionsTwo_meets {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    (twoA Coin).KS H ∧ (twoLTwo Coin).commit.Binds H ∧ (twoLTwo Coin).Sessions H ∧ (twoLTwo Coin).Scope [] ∧
      ∀ r, (twoLTwo Coin).replaySample [] r = [0, 1] ∧ (twoLTwo Coin).checked [] r = [0, 1] ∧
        (run H silentP (twoLTwo Coin).V [] () r).out = .acc :=
  ⟨twoA_KS H, trueCm_binds H, twoLTwo_sessions H, twoLTwo_scope [], fun r => ⟨rfl, rfl, twoL_silent H r⟩⟩

end Claims.Witness.SamplingGeneric

/-! ## Stated over the spec -/

namespace Claims.SamplingGeneric

open Claims.Witness (silentP)
open Claims.Witness.SamplingGeneric

/-- **`CompiledTwo.KS` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.KS_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.KS H :=
  ⟨_, _, compiledTwoKS_fails H⟩

/-- **`CompilerKSTwo`'s premises hold together at two sessions**: `twoA` is knowledge sound, `twoLTwo`'s commitment
binds, `twoLTwo` meets the two-stage contract, the empty statement is in scope, and each run samples both replay blocks
and both proof blocks and is accepted. -/
theorem compilerKSTwo_honest :
    (twoA Bool).KS FlockDeploy.sha512 ∧ (twoLTwo Bool).commit.Binds FlockDeploy.sha512 ∧
      (twoLTwo Bool).Sessions FlockDeploy.sha512 ∧ (twoLTwo Bool).Scope [] ∧
      ∀ r, (twoLTwo Bool).replaySample [] r = [0, 1] ∧ (twoLTwo Bool).checked [] r = [0, 1] ∧
        (run FlockDeploy.sha512 silentP (twoLTwo Bool).V [] () r).out = .acc :=
  twoSessionsTwo_meets _

/-- **`CompilerKSTwo`'s conclusion fails** at `sksLTwo`, which accepts with an empty relation. -/
theorem compilerKSTwo_nontrivial : ¬ (sksLTwo Bool).KS FlockDeploy.sha512 := compiledTwoKS_fails _

/-- **`CompiledTwo.Exfil` fails** at some compiled two-stage verifier. -/
theorem CompiledTwo.Exfil_fails {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] (H : Msg → Msg) :
    ∃ (A : Arg Coin) (L : CompiledTwo Coin A), ¬ L.Exfil H :=
  ⟨_, _, compiledTwoExfil_fails H⟩

/-- **`CompilerExfilTwo`'s conclusion fails** at `sksLTwo`, which accepts with an empty influence relation. -/
theorem compilerExfilTwo_nontrivial : ¬ (sksLTwo Bool).Exfil FlockDeploy.sha512 := compiledTwoExfil_fails _

end Claims.SamplingGeneric

/-! ## The patterns' form says something -/

namespace Claims.Witness.SamplingGeneric

open Claims.Sampling

/-- **`InfluenceTwoPatterns`'s premise holds** at the two-session toy: it is in the two-stage scope, the scope its
compiled two-stage verifier checks. -/
theorem influenceTwoPatterns_honest : InScopeTwo twoPart twoTwo twoStmt ∧ (twoLTwo Bool).Scope [] :=
  ⟨inScopeTwo_two, twoLTwo_scope []⟩

/-- A commitment scheme under which every opening opens. -/
def openCm : CommitmentScheme Bool := ⟨fun _ _ => .ret true, fun _ _ _ => []⟩

/-- A statement whose one gate, an empty `and`, is returned; no gate is an input. -/
def openStmt : AuditStatement := ⟨[], ⟨[.and []], [0]⟩, ⟨"", 0, []⟩, ⟨[]⟩, []⟩

/-- The partition with no proof blocks: it partitions no gate of `openStmt`, which is outside the two-stage scope. -/
def noPart : Partition := ⟨[]⟩

/-- The two-stage design over no replay blocks, with no first-stage rule and no stratum. -/
def noTwo : Design.Two := ⟨noPart, ⟨[]⟩, []⟩

/-- The witness that opens the returned gate at `b`. -/
def openW (b : Bool) : Wit := ⟨[⟨0, b, []⟩]⟩

/-- Over no replay blocks the first stage samples nothing and the second stage misses nothing: `P̄_a = 1`. -/
theorem noTwo_Pa (fp : ℕ → ℕ → ℕ) : noTwo.Pa noPart fp = 1 := by
  have hl : noTwo.part.blocks.length = 0 := rfl
  unfold Design.Two.Pa
  rw [hl, Finset.range_zero, Finset.powerset_empty, Finset.sum_singleton]
  simp only [Finset.prod_empty, Finset.sdiff_self, mul_one]
  rfl

theorem admissibleTwo_noTwo : admissibleTwo noPart noTwo 0 = {∅} := by
  ext T
  have hl : noPart.blocks.length = 0 := rfl
  simp only [admissibleTwo, Finset.mem_filter, Finset.mem_powerset, Finset.mem_singleton, hl, Finset.range_zero,
    Finset.subset_empty]
  refine ⟨fun h => h.1, fun h => ⟨h, ?_⟩⟩
  show (0 : ℝ) < noTwo.Pa noPart _
  rw [noTwo_Pa]
  exact one_pos

theorem inputsOf_open (b : Bool) : inputsOf id openCm openStmt (openW b) = fun _ => none := by
  funext g
  have h : openStmt.prog.gates[g]? ≠ some .input := by
    match g with
    | 0 => simp [openStmt]
    | n + 1 => simp [openStmt]
  simp [inputsOf, h]

theorem served_open (b : Bool) : served id openCm openStmt (openW b) = fun _ => b := by
  funext ⟨k, hk⟩
  match k, hk with
  | 0, _ => cases b <;> rfl
  | n + 1, hk => exact absurd hk (by simp [openStmt])

/-- Two witnesses serve `true` and `false` on no inputs, each at a footprint in `M̄_0`. -/
theorem two_le_card_influence_open : 2 ≤ (influenceTwo id openCm openStmt noPart noTwo 0 fun _ => none).card := by
  classical
  have hmem : ∀ b, (fun _ => b) ∈ influenceTwo id openCm openStmt noPart noTwo 0 (fun _ => none) := by
    intro b
    unfold influenceTwo
    refine Finset.mem_filter.2 ⟨Finset.mem_univ _, openW b, inputsOf_open b, ⟨fun _ _ => false, ?_⟩, served_open b⟩
    show (0 : ℝ) < noTwo.Pa noPart _
    rw [noTwo_Pa]
    exact one_pos
  have hne : (fun _ => true : Fin openStmt.prog.outs.length → Bool) ≠ fun _ => false := by
    intro h
    have := congrFun h ⟨0, Nat.zero_lt_one⟩
    simp at this
  calc 2 = ({fun _ => true, fun _ => false} : Finset (Fin openStmt.prog.outs.length → Bool)).card :=
        (Finset.card_pair hne).symm
    _ ≤ _ := Finset.card_le_card fun y hy => by
        rcases Finset.mem_insert.1 hy with rfl | hy
        · exact hmem true
        · rw [Finset.mem_singleton.1 hy]
          exact hmem false

/-- The admissible sets are `{∅}`, whose one pattern touches no request: the patterns' sum is `2^0`. -/
theorem patterns_noTwo :
    ∑ p ∈ (admissibleTwo noPart noTwo 0).image
        (fun T r => lightOf openStmt noPart (servedBits (fun _ : Fin openStmt.prog.outs.length => ()) r)
          (blocksTo openStmt.prog noPart (fun _ => ()) T r)),
      2 ^ ∑ r, lightBits openStmt noPart (servedBits (fun _ : Fin openStmt.prog.outs.length => ()) r) (p r) = 1 := by
  rw [admissibleTwo_noTwo, Finset.image_singleton, Finset.sum_singleton]
  simp [blocksTo, lightOf, lightBits]

/-- **`InfluenceTwoPatterns`'s conclusion fails** outside the two-stage scope: at `openStmt` over the partition with no
proof blocks, two witnesses serve different outputs at footprints in `M̄_0`, so the influence set has two members, and
the patterns' sum is `1`. -/
theorem influenceTwoPatterns_nontrivial :
    ¬ ((influenceTwo id openCm openStmt noPart noTwo 0 fun _ => none).card ≤
      ∑ p ∈ (admissibleTwo noPart noTwo 0).image
          (fun T r => lightOf openStmt noPart (servedBits (fun _ : Fin openStmt.prog.outs.length => ()) r)
            (blocksTo openStmt.prog noPart (fun _ => ()) T r)),
        2 ^ ∑ r, lightBits openStmt noPart (servedBits (fun _ : Fin openStmt.prog.outs.length => ()) r) (p r)) := by
  rw [patterns_noTwo]
  have := two_le_card_influence_open
  omega

end Claims.Witness.SamplingGeneric

/-! ## The per-window cap says something -/

namespace Claims.Witness.SamplingGeneric

open Claims.Sampling
open Claims.Witness.Sampling (rejCm toyStmt toyPart toy_Pa)

/-- The toy's first stage `all` catches its one replay block at rate `1`. -/
theorem toyTwo_firstRate : toyTwo.FirstRate 1 := by
  intro T hT
  show toyStmt.design.Pa toyPart T ≤ (1 - 1) ^ T.card
  rw [toy_Pa, sub_self]
  split_ifs with hw
  · have hT0 : T = ∅ := by
      refine Finset.eq_empty_of_forall_notMem fun v hv => ?_
      have hlen : toyTwo.part.blocks.length = 1 := rfl
      have hv0 : v = 0 := by
        have := Finset.mem_range.1 (hT hv)
        omega
      subst hv0
      have h0w : 0 ∈ Claims.Sampling.window 0 1 T := by simp [Claims.Sampling.window, hv]
      rw [hw] at h0w
      simp at h0w
    subst hT0
    simp
  · exact pow_nonneg le_rfl _

/-- The toy's second stage samples its one cell, of one proof block, whole: rate `1`. -/
theorem toyTwo_secondRate : toyTwo.SecondRate toyPart 1 := by
  intro ρ hρ t ht _
  have hlen : toyTwo.part.blocks.length = 1 := rfl
  have hs : toyPart.strata.length = 1 := rfl
  obtain rfl : ρ = 0 := by omega
  obtain rfl : t = 0 := by omega
  have hc : (cell toyPart toyTwo.part 0 0).card = 1 := by decide
  rw [hc]
  norm_num [toyTwo]

/-- **`CapTwo`'s premises hold** at the toy: its first stage catches its replay block at rate `1` and its second stage
samples its cell at rate `1`. -/
theorem capTwo_honest : toyTwo.FirstRate 1 ∧ toyTwo.SecondRate toyPart 1 := ⟨toyTwo_firstRate, toyTwo_secondRate⟩

/-- The toy with a second stage that samples nothing (`k = 0`). -/
def zeroTwo : Design.Two := ⟨toyPart, toyStmt.design, [0]⟩

theorem escCell_k0 (n m : ℕ) : escCell n 0 m = 1 := by simp [escCell]

theorem zeroTwo_esc (fp : ℕ → ℕ → ℕ) (ρ : ℕ) : zeroTwo.esc toyPart fp ρ = 1 := by
  unfold Design.Two.esc
  refine Finset.prod_eq_one fun t _ => ?_
  have hk : zeroTwo.k.getD t 0 = 0 := by cases t <;> simp [zeroTwo]
  rw [hk, escCell_k0]

/-- A second stage that samples nothing never catches: `P̄_a = 1` at every footprint. -/
theorem zeroTwo_Pa (fp : ℕ → ℕ → ℕ) : zeroTwo.Pa toyPart fp = 1 := by
  have hlen : zeroTwo.part.blocks.length = 1 := rfl
  have h1 : zeroTwo.replay.Pa zeroTwo.part ∅ = 1 := by
    show toyStmt.design.Pa toyPart ∅ = 1
    rw [toy_Pa]
    simp [Claims.Sampling.window]
  have hps : ({0} : Finset ℕ).powerset = {∅, {0}} := by
    ext t
    simp [Finset.subset_singleton_iff]
  unfold Design.Two.Pa
  rw [hlen, Finset.range_one, hps, Finset.sum_pair (Finset.singleton_ne_empty 0).symm]
  simp [zeroTwo_esc, h1]

theorem zeroTwo_wrongBlocks {fp : ℕ → ℕ → ℕ} (h : 1 ≤ fp 0 0) : (zeroTwo.wrongBlocks toyPart fp).card = 1 := by
  have hlen : zeroTwo.part.blocks.length = 1 := rfl
  have hs : toyPart.strata.length = 1 := rfl
  have hp : ∃ t < toyPart.strata.length, 0 < fp 0 t := ⟨0, by rw [hs]; exact Nat.one_pos, h⟩
  unfold Design.Two.wrongBlocks
  rw [hlen, Finset.range_one, Finset.filter_singleton, if_pos hp, Finset.card_singleton]

/-- **`CapTwo`'s conclusion fails** at a second stage that samples nothing: the toy's one proof block is incorrect at
every interior, so the footprint marks its one replay block and the bound at rates `1` and `1` is `0`, while
`P̄_a = 1`. -/
theorem capTwo_nontrivial :
    ¬ (zeroTwo.Pa toyPart (footprint id rejCm toyStmt toyPart zeroTwo.part ⟨[]⟩ fun _ _ => false) ≤
      (1 - 1 * 1) ^ (zeroTwo.wrongBlocks toyPart
        (footprint id rejCm toyStmt toyPart zeroTwo.part ⟨[]⟩ fun _ _ => false)).card) := by
  rw [zeroTwo_Pa, zeroTwo_wrongBlocks (fp := footprint id rejCm toyStmt toyPart zeroTwo.part ⟨[]⟩ fun _ _ => false)
    (toy_footprint id ⟨[]⟩ _)]
  norm_num

end Claims.Witness.SamplingGeneric
