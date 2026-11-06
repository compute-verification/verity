module

public import Definitions.Claims.SamplingTwo
public import Proofs.Flock.Claims.Witnesses.SamplingGeneric
public import Proofs.Flock.SamplingGeneric.Two
public import Proofs.Flock.Claims.Deploy

/-!
# The two-stage compiler's statement says something

`CompilerKSTwo` is a guarantee that the lock's `claims.generic` names, held to two witnesses: an honest one, whose
conclusion is its premises holding together at one instance with two sampled sessions, and a nontrivial one, at which
its conclusion fails. Each is at coins `Bool` and the hash SHA-512 (`FlockDeploy.sha512`), on the toys of
`Witnesses.SamplingGeneric`; the claim `CompiledTwo.KS` they read has its `fails` witness here too.

* `sksLTwo`: `sksL` (accepts after one round, samples nothing, opens nothing) with the two-stage design `toyTwo` (its
  replay partition the proof-block partition `{0}`, its first stage `all`, `k = 1`). Its one proof block is incorrect at
  any interior values, so its footprint is at least `1` at every interior, the cell's escape `0` and `P̄_a = 0`: no witness is in the
  relation at `δ = 0`, and `CompiledTwo.KS` fails (`compiledTwoKS_fails`).
* `twoLTwo`: `twoL` (both proof blocks of `twoStmt` sampled, block `u`'s session at round `u`) with the design `twoTwo`
  (replay blocks the proof blocks `{0}`, `{1}`, first stage `all`, `k = 1`), sampling both replay blocks. It meets
  `CompiledTwo.Sessions` at every hash (`twoSessionsTwo_meets`): the first stage misses a set only when it is empty, and
  the second stage misses `X` only when `X` holds no proof block, where each cell escapes.
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

end Claims.SamplingGeneric
