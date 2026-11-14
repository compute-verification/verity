module

public import Proofs.Flock.SamplingGeneric.SessionFinds
public import Proofs.Flock.SamplingGeneric.TwoReg
import Proofs.Flock.Claims.Aggregate
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.Finder
import Proofs.Flock.Claims.Prob
import Proofs.Flock.Claims.Sampling.Binding
import Proofs.Flock.Claims.Sampling.Extract
import Proofs.Flock.SamplingGeneric.Assemble
import Proofs.Flock.Soundness.Rewinding

/-!
# The two-stage event against the one-wait extraction's binding terms

`TwoReg.lean` bounds the event of the committed form, an accepted run that samples a proof block incorrect at a
boundary trace and its replay block's committed interior, by the sessions' bound plus three binding terms of the run's
extractions: they disagree with themselves, open a gate off the boundary trace, or open one off a committed interior
(`regTwo_split`). Each binding term is paid for by a finder that reruns the extraction, so its cost is the
extraction's, and the extraction of a proof system whose extractor waits for `k` accepting runs costs `k` waits.

Here the binding terms are those of the compiled verifier over the one-wait proof system (`CompiledTwo.oneWait`),
whose extraction is one wait, at twice their probability:

* `Compiled.pr_found_le_at`: `SessionFinds` averaged over the tape, at a selection and a property that read the run's
  coins off the second stage only (`pr_unfound_le_at`'s form).
* `Compiled.sessKept`, `Compiled.OffAt`: what a session's witness reveals at the session's proof blocks that passes
  its opening check, and the witness being off a boundary trace and interiors there: two of those openings disagree,
  or one opens a gate of the program off the trace, or off its replay block's interior.
* `regTwo_found_split`: **the deterministic half at the session's own witness**. At an accepted run that samples a
  proof block incorrect at a boundary trace and interiors, the block's session is not found at its first slot's
  coins, or is found with a witness that is off them (`regTwo_split`'s proof, the values read off the one witness).
* `off_of_foundWith`: a session found at its first slot's coins with a witness that is off a boundary trace and
  interiors puts one of the three binding terms in the run's extractions, which keep what the witness reveals.
* `eventTwo_oneWait_le_of_off`: **the event's bound from bounds on the one-wait extraction's binding terms**,
  `3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(b_I + b_B + b_N)`: the sessions' bound where no witness is found, and
  `SessionFinds` where one is found off the traces, the property a function of the run's coins and the witness.
  The interiors are the one-wait extraction's committed ones, which read no second-stage coin.
-/

@[expose] public section

namespace Claims.SamplingGeneric

open Claims Finset
open Claims.Sampling (AuditStatement Opening Partition InScopeTwo cell replayOf forkOn wrongInAt assemble
  gateBlock gateBlock_of_mem correct_refIn_congr val_assemble ιOf ιOf_eq wrongAllAt wrongAllAt_subset)

namespace Compiled

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The sessions' bound on what is found, averaged over the tape -/

/-- **The sessions' bound on what the extractions find, at a selection and a property that read the first stage's
coins**, averaged over the tape: the run accepts and the proof block at the picked slot is found, at the slot's
coins, with a witness with the property with probability at most twice that at the one-wait proof system, plus
`2·κ_s + √(k(x)·γ_s·Pr[B])`, for a selection that reads the run's coins off the second stage only
(`pr_unfound_le_at`'s form). -/
theorem pr_found_le_at (H : Msg → Msg) (L : Compiled Coin A) {x : Msg} {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (P : Prover Ω) (β : Ω → (Fin (L.V.k x) → Coin) → ℝ) (hβ0 : ∀ ω r, 0 ≤ β ω r)
    (hβ : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => β p.1 p.2) ≤ PrB H L.V P x)
    (sel : Ω → (Fin (L.V.k x) → Coin) → List ℕ → Option (Fin (L.nPU x)))
    (hsel : ∀ ω r e, sel ω (forkOn (L.stage2 x) r e) = sel ω r)
    (G : Ω → (Fin (L.V.k x) → Coin) → Fin (L.nPU x) → Msg → A.W → Prop)
    (hκ : ∀ ω r (sl : List ℕ → Option (Fin (L.nPU x))), Pr (fun p : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
        ∃ i, sl (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
            L.FoundWith H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 i u)
              (G ω (forkOn (L.stage2 x) r p.1) i)) ≤
        2 * Pr (fun p : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
          (run H P L.V x ω (forkOn (L.stage2 x) r p.1)).out = .acc ∧
            ∃ i, sl (L.checked x (forkOn (L.stage2 x) r p.1)) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r p.1))[i.val]? = some u.val ∧
                L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.stage2 x) r p.1) (p.2 i u)
                  (G ω (forkOn (L.stage2 x) r p.1) i)) +
          (2 * L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r))) :
    Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω rc.1 (L.checked x rc.1) = some i ∧
        ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
          L.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) (G ω rc.1 i)) ≤
      2 * Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω rc.1 (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            L.oneWait.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) (G ω rc.1 i)) +
        (2 * L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x)) := by
  have hγ0 : 0 ≤ (L.V.k x : ℝ) * L.γs x := mul_nonneg (Nat.cast_nonneg _) (L.γs_nonneg x)
  have hω : ∀ ω : Ω, Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω rc.1 (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            L.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) (G ω rc.1 i)) ≤
      2 * Pr (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧ ∃ i, sel ω rc.1 (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            L.oneWait.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) (G ω rc.1 i)) +
        (2 * L.κs x + Ex fun r => Real.sqrt (L.V.k x * L.γs x * β ω r)) := by
    intro ω
    rw [Pr_forkOn_prod (L.stage2 x) (fun (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) =>
        (run H P L.V x ω r).out = .acc ∧ ∃ i, sel ω r (L.checked x r) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x r)[i.val]? = some u.val ∧
            L.FoundWith H x (run H P L.V x ω) u r (c i u) (G ω r i)),
      Pr_forkOn_prod (L.stage2 x) (fun (r : Fin (L.V.k x) → Coin) (c : L.oneWait.SlotEC x) =>
        (run H P L.V x ω r).out = .acc ∧ ∃ i, sel ω r (L.checked x r) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x r)[i.val]? = some u.val ∧
            L.oneWait.FoundWith H x (run H P L.V x ω) u r (c i u) (G ω r i))]
    have hr : ∀ r : Fin (L.V.k x) → Coin, Pr (fun q : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
          (run H P L.V x ω (forkOn (L.stage2 x) r q.1)).out = .acc ∧
            ∃ i, sel ω (forkOn (L.stage2 x) r q.1) (L.checked x (forkOn (L.stage2 x) r q.1)) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r q.1))[i.val]? = some u.val ∧
                L.FoundWith H x (run H P L.V x ω) u (forkOn (L.stage2 x) r q.1) (q.2 i u)
                  (G ω (forkOn (L.stage2 x) r q.1) i)) ≤
        2 * Pr (fun q : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
          (run H P L.V x ω (forkOn (L.stage2 x) r q.1)).out = .acc ∧
            ∃ i, sel ω (forkOn (L.stage2 x) r q.1) (L.checked x (forkOn (L.stage2 x) r q.1)) = some i ∧
              ∃ u : Fin (L.nPU x), (L.checked x (forkOn (L.stage2 x) r q.1))[i.val]? = some u.val ∧
                L.oneWait.FoundWith H x (run H P L.V x ω) u (forkOn (L.stage2 x) r q.1) (q.2 i u)
                  (G ω (forkOn (L.stage2 x) r q.1) i)) +
          (2 * L.κs x + Real.sqrt (L.V.k x * L.γs x * β ω r)) := fun r => by
      simp only [hsel]
      exact hκ ω r (sel ω r)
    refine (Ex_mono hr).trans (le_of_eq ?_)
    rw [Ex_add, Ex_mul_left, Ex_add, Ex_const]
  refine (Ex_mono hω).trans ?_
  rw [Ex_add, Ex_mul_left, Ex_add, Ex_const,
    ← Ex_prod (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2))]
  have hsq : Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => Real.sqrt (L.V.k x * L.γs x * β p.1 p.2)) ≤
      Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) := by
    refine (Ex_sqrt_le _ fun p => mul_nonneg hγ0 (hβ0 _ _)).trans (Real.sqrt_le_sqrt ?_)
    rw [Ex_mul_left]
    exact mul_le_mul_of_nonneg_left hβ hγ0
  linarith

/-! ## A session's witness, off a boundary trace and interiors -/

/-- **What a session's witness keeps**: the openings the witness `w` reveals, at the proof blocks of slot `i₀`'s
session in the sample `l`, that pass their opening check. A session's extraction that finds `w` returns them
(`sessExt`). -/
def sessKept (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (l : List ℕ) (i₀ : ℕ) (v : Seen)
    (w : A.W) : List (List Msg × Opening Bool) :=
  (L.sessBlocks x l i₀).flatMap fun u' => (L.reveal s u' v w).filter fun q => L.commit.Opens H q.1 q.2

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
theorem mem_sessKept {H : Msg → Msg} {L : Compiled Coin A} {x : Msg} {s : AuditStatement} {l : List ℕ} {i₀ : ℕ}
    {v : Seen} {w : A.W} {q : List Msg × Opening Bool} :
    q ∈ L.sessKept H x s l i₀ v w ↔
      ∃ u' ∈ L.sessBlocks x l i₀, q ∈ L.reveal s u' v w ∧ L.commit.Opens H q.1 q.2 = true := by
  simp only [sessKept, List.mem_flatMap, List.mem_filter]

/-- **A session's witness is off a boundary trace `tr` and interiors `itr`**, `itr ρ` replay block `ρ`'s: among what
the witness keeps (`sessKept`), two openings under one roots at one position differ, or one under the statement's
roots opens a gate of the program off `tr`, or one under the interior roots of the replay block of `Pρ` that holds
its gate, a gate of the program, opens it off that block's `itr`. -/
def OffAt (H : Msg → Msg) (L : Compiled Coin A) (x : Msg) (s : AuditStatement) (Pρ : Partition)
    (tr : ℕ → Option Bool) (itr : ℕ → ℕ → Bool) (l : List ℕ) (i₀ : ℕ) (v : Seen) (w : A.W) : Prop :=
  (∃ q ∈ L.sessKept H x s l i₀ v w, ∃ q' ∈ L.sessKept H x s l i₀ v w,
      q.1 = q'.1 ∧ q.2.pos = q'.2.pos ∧ q.2.val ≠ q'.2.val) ∨
    (∃ q ∈ L.sessKept H x s l i₀ v w, q.1 = s.roots ∧ q.2.pos < L.nG x ∧ tr q.2.pos ≠ some q.2.val) ∨
    ∃ q ∈ L.sessKept H x s l i₀ v w, ∃ ρ, gateBlock Pρ q.2.pos = some ρ ∧ q.1 = L.iroots x v ρ ∧
      q.2.pos < L.nG x ∧ itr ρ q.2.pos ≠ q.2.val

end Compiled

namespace CompiledTwo

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] {A : Arg Coin}

/-! ## The deterministic half, at the session's own witness -/

set_option synthInstance.maxSize 1024 in
/-- **The deterministic half at the session's own witness**: at an accepted run whose sample holds, at slot `i`, a
proof block `u` incorrect at the boundary trace `tr` and the interiors `itr` (`wrongAllAt`), the session of `i` is not
found at its first slot `i₀`'s coins, or is found there with a witness that is off `tr` and `itr` (`Compiled.OffAt`).
Otherwise the boundary values that are `tr` on the program's gates and what the witness keeps elsewhere, and the
interior values that are `u`'s replay block's `itr` on its gates and what the witness keeps elsewhere, agree with
every opening the witness reveals at `u` that passes, so they make `u` correct (`Sessions.opens`); and `tr` and `itr`
agree with them where `u`'s correctness reads (`correct_refIn_congr`), so `u` is correct at them, which
`u ∈ wrongAllAt` denies. It is `regTwo_split` with the values read off the session's one witness in place of the
run's extractions, so the interiors are the reader's too. -/
theorem regTwo_found_split {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x) (hacc : (run H P L.V x ω r).out = .acc)
    (i u : Fin (L.nPU x)) (hiu : (L.checked x r)[i.val]? = some u.val) (tr : ℕ → Option Bool)
    (itr : ℕ → ℕ → Bool) (hF : u.val ∈ wrongAllAt s Pt d.part tr itr) (i₀ u₀ : Fin (L.nPU x))
    (hlead : L.leadOf x (L.checked x r) i = some i₀) (hu₀ : (L.checked x r)[i₀.val]? = some u₀.val) :
    ¬ L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀) ∨
      L.FoundWith H x (run H P L.V x ω) u₀ r (c i₀ u₀) fun _ w =>
        L.OffAt H x s d.part tr itr (L.checked x r) i₀.val (run H P L.V x ω r).seen w := by
  have hu := List.mem_of_getElem? hiu
  by_cases hfd : L.Found H x (run H P L.V x ω) u₀ r (c i₀ u₀)
  swap
  · exact Or.inl hfd
  refine Or.inr ?_
  obtain ⟨xS, vS, hsess, hsh, w, hw, hR⟩ := hfd
  refine ⟨xS, vS, hsess, hsh, w, hw, ?_⟩
  by_contra hoff
  have hagree : ∀ q ∈ L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w,
      ∀ q' ∈ L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w,
        q.1 = q'.1 → q.2.pos = q'.2.pos → q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => Classical.byContradiction fun hne => hoff (Or.inl ⟨q, hq, q', hq', h1, h2, hne⟩)
  have hreg : ∀ q ∈ L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w, q.1 = s.roots →
      q.2.pos < L.nG x → tr q.2.pos = some q.2.val :=
    fun q hq h1 hg => Classical.byContradiction fun hne => hoff (Or.inr (Or.inl ⟨q, hq, h1, hg, hne⟩))
  have hireg : ∀ q ∈ L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w, ∀ ρ,
      gateBlock d.part q.2.pos = some ρ → q.1 = L.iroots x (run H P L.V x ω r).seen ρ → q.2.pos < L.nG x →
        itr ρ q.2.pos = q.2.val :=
    fun q hq ρ hρ h1 hg => Classical.byContradiction fun hne =>
      hoff (Or.inr (Or.inr ⟨q, hq, ρ, hρ, h1, hg, hne⟩))
  have hnG : L.nG x = s.prog.gates.length := by simp only [Compiled.nG, hs]
  obtain ⟨hk, hc, -⟩ := Compiled.of_leadOf hlead
  have hsu : L.session x u (run H P L.V x ω r).seen = some (xS, vS) := by
    rw [← hS.shared x Ω P ω r _ _ _ _ hacc hu₀ hiu hk]
    exact hsess
  have hub : u.val ∈ L.sessBlocks x (L.checked x r) i₀ := by
    rw [Compiled.sessBlocks, hc]
    exact List.mem_filterMap.2 ⟨i.val, Compiled.self_mem_sessSlots hiu u.isLt, hiu⟩
  obtain ⟨b, ρ, rb, hb, hρ, hrb, hcor⟩ := hS.opens x s Pt d Ω P ω r u xS vS w hs hPt hd hin hacc hu hsu hR
  simp only [wrongAllAt, mem_biUnion, mem_product, mem_range] at hF
  obtain ⟨⟨ρ', t⟩, -, hwr⟩ := hF
  simp only [wrongInAt, mem_filter] at hwr
  obtain ⟨hcell, hnot⟩ := hwr
  have hρ' : ρ' = ρ := by
    simp only [cell, mem_filter] at hcell
    exact Option.some.inj (hcell.2.1.symm.trans hρ)
  rw [hρ'] at hnot
  obtain ⟨V, hV⟩ : ∃ V : ℕ → Option Bool, ∀ g, V g =
      if g < s.prog.gates.length then tr g
      else (assemble s (L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w)).val H L.commit s g :=
    ⟨_, fun _ => rfl⟩
  obtain ⟨I, hI⟩ : ∃ I : ℕ → Bool, ∀ g, I g =
      if g < s.prog.gates.length ∧ g ∈ rb then itr ρ g
      else ιOf (L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w)
        (L.iroots x (run H P L.V x ω r).seen ρ) g := ⟨_, fun _ => rfl⟩
  have hc' : s.prog.Correct (s.prog.refIn rb V I) b := hcor V I fun q hq hok => by
    have hqk : q ∈ L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w :=
      Compiled.mem_sessKept.2 ⟨u.val, hub, hq, hok⟩
    refine ⟨fun hr => ?_, fun hi => ?_⟩
    · rw [hV]
      by_cases hg : q.2.pos < s.prog.gates.length
      · rw [ite_eq_left hg]
        exact hreg q hqk hr (by rw [hnG]; exact hg)
      · rw [ite_eq_right hg]
        rw [hr] at hok
        exact val_assemble hagree hqk hr hok
    · rw [hI]
      by_cases hg : q.2.pos < s.prog.gates.length ∧ q.2.pos ∈ rb
      · rw [ite_eq_left hg]
        exact hireg q hqk ρ (gateBlock_of_mem hin.2.2.1 hrb hg.2) hi (by rw [hnG]; exact hg.1)
      · rw [ite_eq_right hg, ← hi]
        exact ιOf_eq hagree hqk
  exact hnot ⟨b, rb, hb, hrb, (correct_refIn_congr hin.1 hin.2.1 hb rb (fun g hg => by rw [hV, ite_eq_left hg])
    fun g hg hgr => by rw [hI, ite_eq_left ⟨hg, hgr⟩]).1 hc'⟩

/-- **A witness off the traces puts a binding term in the run's extractions**: at a run whose slot `i₀`, its
session's first, holds proof block `u₀`, found at the slot's coins with a witness that is off a boundary trace `tr`
and interiors `itr`, the run's extractions disagree with themselves, or open a gate of the program off `tr`, or open
one, under the interior roots of the replay block that holds it, off that block's `itr`. The run's extractions keep
what the witness keeps (`sessExt_mem`, `mem_runExt`), and where they agree with themselves they open what each kept
opening does (`val_assemble`, `iopensAt_of_mem`). -/
theorem off_of_foundWith {H : Msg → Msg} {L : CompiledTwo Coin A} {x : Msg} {s : AuditStatement}
    (hs : L.parse x = some s) {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin (L.V.k x) → Coin) (c : L.SlotEC x)
    (Pρ : Partition) (tr : ℕ → Option Bool) (itr : ℕ → ℕ → Bool) {i₀ u₀ : Fin (L.nPU x)}
    (hu₀ : (L.checked x r)[i₀.val]? = some u₀.val)
    (hfirst : (L.sessSlots x (L.checked x r) i₀).head? = some i₀.val)
    (h : L.FoundWith H x (run H P L.V x ω) u₀ r (c i₀ u₀) fun _ w =>
      L.OffAt H x s Pρ tr itr (L.checked x r) i₀.val (run H P L.V x ω r).seen w) :
    L.InConf H P x s ω r c ∨ (∃ g < L.nG x, ∃ b, L.opensAt H P x ω r c g = some b ∧ tr g ≠ some b) ∨
      ∃ g < L.nG x, ∃ ρ, gateBlock Pρ g = some ρ ∧ ∃ b, L.iopensAt H P x ω r c ρ g = some b ∧ itr ρ g ≠ b := by
  obtain ⟨xS, vS, hsess, hsh, w, hw, hoff⟩ := h
  have hsub : ∀ q ∈ L.sessKept H x s (L.checked x r) i₀.val (run H P L.V x ω r).seen w,
      q ∈ L.kept H P x s ω r c ∧ L.commit.Opens H q.1 q.2 = true := by
    intro q hq
    obtain ⟨u', hu', hq', hok⟩ := Compiled.mem_sessKept.1 hq
    exact ⟨L.mem_runExt H x (run H P L.V x ω) s r _ c hu₀ u₀.isLt hfirst
      (L.sessExt_mem H x (run H P L.V x ω) s _ u₀ r (c i₀ u₀) hsess hsh hw hu' hq' hok), hok⟩
  by_cases hconf : L.InConf H P x s ω r c
  · exact Or.inl hconf
  have hagree : ∀ q ∈ L.kept H P x s ω r c, ∀ q' ∈ L.kept H P x s ω r c, q.1 = q'.1 → q.2.pos = q'.2.pos →
      q.2.val = q'.2.val :=
    fun q hq q' hq' h1 h2 => Classical.byContradiction fun hne => hconf ⟨q, hq, q', hq', h1, h2, hne⟩
  rcases hoff with ⟨q, hq, q', hq', h1, h2, hne⟩ | ⟨q, hq, h1, hg, hne⟩ | ⟨q, hq, ρ, hρ, h1, hg, hne⟩
  · exact absurd (hagree q (hsub q hq).1 q' (hsub q' hq').1 h1 h2) hne
  · refine Or.inr (Or.inl ⟨q.2.pos, hg, q.2.val, ?_, hne⟩)
    have hok := (hsub q hq).2
    rw [h1] at hok
    rw [L.opensAt_eq H P x ω hs]
    exact val_assemble hagree (hsub q hq).1 h1 hok
  · exact Or.inr (Or.inr ⟨q.2.pos, hg, ρ, hρ, q.2.val,
      iopensAt_of_mem hs hagree (hsub q hq).1 h1 (hsub q hq).2, hne⟩)

/-! ## The event -/

set_option synthInstance.maxSize 1024 in
/-- **The event at a tape, at a boundary trace and interiors**: an accepted run that samples a proof block incorrect
at the boundary trace `tr` and the interiors `itr` has the first such block's session, at its first slot (at
`leadSel (selOf …)`), unfound, or found with a witness that is off `tr` and `itr` (`regTwo_found_split`). -/
theorem eventTwo_found_split {H : Msg → Msg} {L : CompiledTwo Coin A} (hS : L.Sessions H) {x : Msg}
    {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two} (hs : L.parse x = some s)
    (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s) {Ω : Type} (P : Prover Ω)
    (ω : Ω) (tr : ℕ → Option Bool) (itr : (Fin (L.V.k x) → Coin) → ℕ → ℕ → Bool) :
    Pr (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
      ∃ u ∈ L.checked x r, u ∈ wrongAllAt s Pt d.part tr (itr r)) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part tr (itr rc.1))) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) +
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part tr (itr rc.1))) (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            L.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) fun _ w =>
              L.OffAt H x s d.part tr (itr rc.1) (L.checked x rc.1) i.val (run H P L.V x ω rc.1).seen w) := by
  refine (Pr_fst (β := L.SlotEC x) (fun r : Fin (L.V.k x) → Coin => (run H P L.V x ω r).out = .acc ∧
    ∃ u ∈ L.checked x r, u ∈ wrongAllAt s Pt d.part tr (itr r))).symm.trans_le
      (Sampling.Pr_or_le (fun rc hrc => ?_) le_rfl le_rfl)
  obtain ⟨hacc, v, hvl, hvF⟩ := hrc
  obtain ⟨i, hi, u, hiu, huF⟩ := L.selOf_some x (hS.slots x rc.1)
    (fun w hw => by rw [Compiled.nPU_eq hs hPt]; exact mem_range.1 (wrongAllAt_subset s Pt d.part _ _ hw)) hvl hvF
  obtain ⟨i₀, u₀, hlead, hu₀⟩ := Compiled.leadOf_some hiu u.isLt
  have hsel : L.leadSel x (L.selOf x (wrongAllAt s Pt d.part tr (itr rc.1))) (L.checked x rc.1) = some i₀ := by
    rw [Compiled.leadSel, hi, Option.bind_some, hlead]
  rcases regTwo_found_split hS hs hPt hd hin P ω rc.1 rc.2 hacc i u hiu tr (itr rc.1) huF i₀ u₀ hlead hu₀
    with h | h
  · exact Or.inl ⟨hacc, i₀, hsel, u₀, hu₀, h⟩
  · exact Or.inr ⟨hacc, i₀, hsel, u₀, hu₀, h⟩

set_option synthInstance.maxSize 1024 in
/-- **The event's bound from bounds on the one-wait extraction's three binding terms**, at a boundary trace that may
read the tape: an accepted run samples a proof block incorrect at the boundary trace and its replay block's interior
committed by the one-wait extraction with probability at most
`3·κ_s + 2·√(k·γ_s·Pr[B]) + 2·(b_I + b_B + b_N)`, where, at the compiled verifier over the one-wait proof system,
`b_I` bounds one run's own disagreement, `b_B` the accepted runs that open a gate off the boundary trace and `b_N`
the accepted runs that open a gate off a committed interior.

* Where the first incorrect block's session is unfound at its first slot: the sessions' bound, `κ_s + √…`
  (`sessionKS_of`, from the proof system's knowledge soundness).
* Where it is found with a witness that is off the traces: at most twice the same at the one-wait proof system, plus
  `2·κ_s + √…` (`sessionFinds_of`, from `Arg.OneWaitFinds`), and there the one-wait run's extractions hold one of the
  three binding terms (`off_of_foundWith`).

The selection and the property read the run's coins off the second stage only, as the committed interior does
(`iregVal_forkOn`). The contract's clauses it reads are those `eventTwo_le_of_off` reads. -/
theorem eventTwo_oneWait_le_of_off {H : Msg → Msg} {L : CompiledTwo Coin A} (hF : A.OneWaitFinds H)
    (hA : A.KS H) (hS : L.Sessions H) {x : Msg} {s : AuditStatement} {Pt : Partition} {d : Sampling.Design.Two}
    (hs : L.parse x = some s) (hPt : L.partition s = some Pt) (hd : L.two s = some d) (hin : InScopeTwo Pt d s)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (P : Prover Ω) (tr : Ω → ℕ → Option Bool) {bI bB bN : ℝ}
    (hI : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
      L.oneWait.InConf H P x s ω rc.1 rc.2) ≤ bI)
    (hB : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.oneWait.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ tr q.1 g ≠ some b) ≤ bB)
    (hN : Pr (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffIreg H P x q.1 d.part q.2.1 q.2.2) ≤ bN) :
    Pr (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
        ∃ u ∈ L.checked x p.2, u ∈ wrongAllAt s Pt d.part (tr p.1) (L.oneWait.iregVal H P x p.1 p.2)) ≤
      3 * L.κs x + 2 * Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) + 2 * (bI + bB + bN) := by
  obtain ⟨β, hβ0, hβ, hκ⟩ := Compiled.sessionKS_of hA (L := L.toCompiled) ⟨hS.fixed, hS.runs⟩ x Ω P
  obtain ⟨β', hβ0', hβ', hκ'⟩ := Compiled.sessionFinds_of hF (L := L.toCompiled) ⟨hS.fixed, hS.runs⟩ x Ω P
  have hsel : ∀ (ω : Ω) (r e : Fin (L.V.k x) → Coin),
      L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω)
          (L.oneWait.iregVal H P x ω (forkOn (L.stage2 x) r e)))) =
        L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω r))) :=
    fun ω r e => congrArg (fun I => L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) I)))
      (iregVal_forkOn H L.oneWait P x ω r e)
  have hT : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            ¬ L.Found H x (run H P L.V x ω) u rc.1 (rc.2 i u)) ≤
      L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x) :=
    Compiled.pr_unfound_le_at H L.toCompiled P β hβ0 hβ
      (fun ω r => L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω r))))
      hsel fun ω r sl => hκ ω r sl
  have hG : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1)))
            (L.checked x rc.1) = some i ∧
          ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
            L.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) fun _ w =>
              L.OffAt H x s d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1) (L.checked x rc.1) i.val
                (run H P L.V x ω rc.1).seen w) ≤
      2 * Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧
          ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1)))
              (L.checked x rc.1) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
              L.oneWait.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) fun _ w =>
                L.OffAt H x s d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1) (L.checked x rc.1) i.val
                  (run H P L.V x ω rc.1).seen w) +
        (2 * L.κs x + Real.sqrt (L.V.k x * L.γs x * PrB H L.V P x)) :=
    Compiled.pr_found_le_at H L.toCompiled P β' hβ0' hβ'
      (fun ω r => L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω r))))
      hsel
      (fun ω r i _ w => L.OffAt H x s d.part (tr ω) (L.oneWait.iregVal H P x ω r) (L.checked x r) i.val
        (run H P L.V x ω r).seen w)
      fun ω r sl => hκ' ω r sl fun r' i _ w =>
        L.OffAt H x s d.part (tr ω) (L.oneWait.iregVal H P x ω r') (L.checked x r') i.val
          (run H P L.V x ω r').seen w
  have hC : ∀ ω : Ω, Pr (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
        (run H P L.V x ω rc.1).out = .acc ∧
          ∃ i, L.leadSel x (L.selOf x (wrongAllAt s Pt d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1)))
              (L.checked x rc.1) = some i ∧
            ∃ u : Fin (L.nPU x), (L.checked x rc.1)[i.val]? = some u.val ∧
              L.oneWait.FoundWith H x (run H P L.V x ω) u rc.1 (rc.2 i u) fun _ w =>
                L.OffAt H x s d.part (tr ω) (L.oneWait.iregVal H P x ω rc.1) (L.checked x rc.1) i.val
                  (run H P L.V x ω rc.1).seen w) ≤
      Pr (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x => L.oneWait.InConf H P x s ω rc.1 rc.2) +
        (Pr (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
            ∃ g < L.nG x, ∃ b, L.oneWait.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr ω g ≠ some b) +
          Pr (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
            L.oneWait.OffIreg H P x ω d.part rc.1 rc.2)) := by
    intro ω
    refine (Pr_mono fun rc hrc => ?_).trans ((Sampling.Pr_union_le
      (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x => L.oneWait.InConf H P x s ω rc.1 rc.2)
      (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
        ((run H P L.V x ω rc.1).out = .acc ∧
            ∃ g < L.nG x, ∃ b, L.oneWait.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr ω g ≠ some b) ∨
          ((run H P L.V x ω rc.1).out = .acc ∧ L.oneWait.OffIreg H P x ω d.part rc.1 rc.2))).trans
        (add_le_add le_rfl (Sampling.Pr_union_le
          (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
            ∃ g < L.nG x, ∃ b, L.oneWait.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr ω g ≠ some b)
          (fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x => (run H P L.V x ω rc.1).out = .acc ∧
            L.oneWait.OffIreg H P x ω d.part rc.1 rc.2))))
    obtain ⟨hacc, i₀, hlsel, u₀, hu₀, hfw⟩ := hrc
    obtain ⟨i, -, hlead⟩ := Option.bind_eq_some_iff.1 hlsel
    obtain ⟨-, -, hfirst⟩ := Compiled.of_leadOf hlead
    rcases off_of_foundWith (L := L.oneWait) hs P ω rc.1 rc.2 d.part (tr ω)
      (L.oneWait.iregVal H P x ω rc.1) hu₀ hfirst hfw with h | h | h
    · exact Or.inl h
    · exact Or.inr (Or.inl ⟨hacc, h⟩)
    · exact Or.inr (Or.inr ⟨hacc, h⟩)
  have hB' : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.oneWait.opensAt H P x ω rc.1 rc.2 g = some b ∧ tr ω g ≠ some b) ≤ bB :=
    (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧
        ∃ g < L.nG x, ∃ b, L.oneWait.opensAt H P x q.1 q.2.1 q.2.2 g = some b ∧ tr q.1 g ≠ some b)).symm.trans_le hB
  have hN' : Ex (fun ω : Ω => Pr fun rc : (Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x =>
      (run H P L.V x ω rc.1).out = .acc ∧ L.oneWait.OffIreg H P x ω d.part rc.1 rc.2) ≤ bN :=
    (Sampling.Pr_prod_eq_Ex (fun q : Ω × ((Fin (L.V.k x) → Coin) × L.oneWait.SlotEC x) =>
      (run H P L.V x q.1 q.2.1).out = .acc ∧ L.oneWait.OffIreg H P x q.1 d.part q.2.1 q.2.2)).symm.trans_le hN
  have hC' := Ex_mono hC
  rw [Ex_add, Ex_add] at hC'
  have hE := (Sampling.Pr_prod_eq_Ex (fun p : Ω × (Fin (L.V.k x) → Coin) => (run H P L.V x p.1 p.2).out = .acc ∧
      ∃ u ∈ L.checked x p.2, u ∈ wrongAllAt s Pt d.part (tr p.1) (L.oneWait.iregVal H P x p.1 p.2))).trans_le
    (Ex_mono fun ω => eventTwo_found_split hS hs hPt hd hin P ω (tr ω) fun r => L.oneWait.iregVal H P x ω r)
  rw [Ex_add] at hE
  linarith

end CompiledTwo

end Claims.SamplingGeneric
