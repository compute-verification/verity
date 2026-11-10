module

public import Proofs.Flock.SampledGeneric.HidingSessions

/-!
# Witnesses of the hiding statements

Honest: the premises hold where the statements are used.

* `hm96TraceHiding_honest`: `Hm96TraceHiding`'s premise, a leaf hiding at `δ₁`, holds at the executable's pinned key
  with `δ₁` the hiding gap there.
* `spotFlock_premises`: `GenericSampledZKAt`'s two premises hold together at Flock's masked session and the
  executable's commit strings.

Nontrivial: each statement is false with a premise dropped or its bound halved.

* `hm96TraceHiding_not_card`: `Hm96TraceHiding` at `N·δ₁` in place of `2·N·δ₁` is false
  (`hm96_commit_two_traces_not_card`'s instance).
* `genericSampledZKAt_needs_hiding`: `GenericSampledZKAt` without its hiding premise is false, at a commitment that
  publishes the trace (`openHider`) under a proof system that says nothing (`muteZk`).
* `genericSampledZKAt_needs_zk`: `GenericSampledZKAt` without its zero-knowledge premise is false, at a commitment that
  publishes nothing (`constHider`) under a proof system whose honest view its simulator never outputs (`leakZk`).
* `genericSessionsZKAt_needs_each`: `GenericSessionsZKAt` with its sum stopped after the first session is false, at a
  proof system that leaks at one statement only (`tellZk`), proved second.
-/

@[expose] public section

namespace FlockSoundness.SampledGeneric

open Game
open scoped ENNReal

/-! ## The audit's two experiments, as averages over the commitment's coins -/

/-- An experiment that flips a coin and then relabels a coin's experiment by the coin: its probability is the average
over the coin (`Close.coinAvg`'s first step). -/
theorem Exp.coinMap_pr {Rnd V W : Type} [Fintype Rnd] [Nonempty Rnd] (c : Rnd → Exp V) (f : Rnd → V → W)
    (E : W → Prop) :
    Exp.pr ⟨.coin Rnd fun r => (c r).g.map (f r), fun r => Strategy.map _ _ (c r).s⟩ E =
      avg fun r => (c r).pr fun v => E (f r v) :=
  congrArg avg (funext fun r => prob_map_strat E (f r) (c r).g (c r).s)

section Audit

variable {n : ℕ} {Tr Rnd Com Wit VS View : Type} [Fintype Rnd] [Nonempty Rnd]

theorem auditView_pr (H : Hider Tr Rnd Com) (Zs : Zk (Com × Finset (Fin n)) Wit VS View)
    (samp : VS → Com → Finset (Fin n)) (wit : Tr → Rnd → Finset (Fin n) → Wit) (Xt : Tr) (v : VS)
    (E : Com × Finset (Fin n) × View → Prop) :
    (auditView H Zs samp wit Xt v).pr E = avg fun r =>
      (Zs.real (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v).pr
        fun a => E (H.commit Xt r, samp v (H.commit Xt r), a) :=
  Exp.coinMap_pr (fun r => Zs.real (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v)
    (fun r a => (H.commit Xt r, samp v (H.commit Xt r), a)) E

theorem auditSim_pr (H : Hider Tr Rnd Com) (Zs : Zk (Com × Finset (Fin n)) Wit VS View)
    (samp : VS → Com → Finset (Fin n)) (X₀ : Tr) (v : VS) (E : Com × Finset (Fin n) × View → Prop) :
    (auditSim H Zs samp X₀ v).pr E = avg fun r =>
      (Zs.sim (H.commit X₀ r, samp v (H.commit X₀ r)) v).pr
        fun a => E (H.commit X₀ r, samp v (H.commit X₀ r), a) :=
  Exp.coinMap_pr (fun r => Zs.sim (H.commit X₀ r, samp v (H.commit X₀ r)) v)
    (fun r a => (H.commit X₀ r, samp v (H.commit X₀ r), a)) E

end Audit

/-! ## Honest -/

open Classical in
/-- **`Hm96TraceHiding`'s premise holds at the executable**: a leaf's pair at the pinned key, `(M(κ₀)·y, c(y))` with
`c` the salt's SHA-512, hides at the hiding gap there (`CROnly.hm96Hiding_gap`). -/
theorem hm96TraceHiding_honest :
    Claims.Hm96Hiding Discharge.ZkView.My Discharge.ZkView.saltHash hm96Gap :=
  CROnly.hm96Hiding_gap Discharge.ZkView.My Discharge.ZkView.saltHash

/-- **`GenericSampledZKAt`'s two premises hold together at Flock**: the masked session is zero knowledge at `flockErr`
on the masked record's relation (`flockSession_zkAt`, from `MaskedZK`), and the executable's commit strings hide at
`2·N·δ₁` (`hm96Exe_hiding`). -/
theorem spotFlock_premises (nr : Claims.Msg → ℕ → ℕ) (t : ℕ) (ht : 0 < t) (Θ ι : Type) [Fintype ι] [DecidableEq ι] :
    ((flockSession nr t Θ).ZKAt (Claims.Flock.R (flockZk nr t).parse) fun x w v => flockErr nr t x w v) ∧
      (hm96Exe ι).Hiding (2 * Fintype.card ι * hm96Gap) :=
  ⟨flockSession_zkAt nr t ht Θ, hm96Exe_hiding ι⟩

/-! ## Nontrivial -/

/-- **hm96's trace commitment need not hide at `N·δ₁`**: `Hm96TraceHiding` at `N·δ₁` in place of `2·N·δ₁` is false.
One leaf with a constant mask satisfies `Hm96Hiding` at `δ₁ = 1/2` (`hm96Hiding_const_half`), but its commitment is the
trace itself, so the event "the leaf is `0`" has probability `1` at the trace `0` and `0` at the trace `1`, more than
`N·δ₁ = 1/2` apart. -/
theorem hm96TraceHiding_not_card :
    ¬ ∀ (ι Sa Bs C D : Type) [Fintype ι] [DecidableEq ι] [Fintype Sa] [Nonempty Sa] [AddCommGroup Bs] [Fintype Bs]
      (My : Sa → Bs) (c : Sa → C) (δ₁ : ℝ≥0∞) (Hf : Bs × C → D),
      Claims.Hm96Hiding My c δ₁ → (hm96Hider ι My c Hf).Hiding (Fintype.card ι * δ₁) := by
  intro h
  have h10 : ¬ (1 : ZMod 2) = 0 := by decide
  have h0 := (h Unit Unit (ZMod 2) Unit (ZMod 2 × Unit) (fun _ => 0) (fun _ => ()) (1 / 2) id hm96Hiding_const_half
    (ZMod 2 × Unit) (fun _ => 0) (fun _ => 1) (fun d => ⟨.ret (d ()), PUnit.unit⟩) fun p => p.1 = 0).1
  rw [Hider.exp_andThen_pr, Hider.exp_andThen_pr] at h0
  have h1 : avg (fun _ : Unit → Unit => @ite ℝ≥0∞ ((0 : ZMod 2) + 0 = 0) (Classical.propDecidable _) 1 0) ≤
      avg (fun _ : Unit → Unit => @ite ℝ≥0∞ ((1 : ZMod 2) + 0 = 0) (Classical.propDecidable _) 1 0) +
        (Fintype.card Unit : ℝ≥0∞) * (1 / 2) := h0
  rw [avg_const, avg_const, add_zero, add_zero, ite_eq_left rfl, ite_eq_right h10, zero_add, Fintype.card_unit,
    Nat.cast_one, one_mul] at h1
  exact absurd h1 (not_le.2 (ENNReal.half_lt_self one_ne_zero ENNReal.one_ne_top))

/-- The commitment that publishes the trace. -/
def openHider : Hider Bool Unit Bool := ⟨fun b _ => b⟩

/-- The commitment that publishes nothing. -/
def constHider : Hider Bool Unit Unit := ⟨fun _ _ => ()⟩

/-- The proof system that says nothing: its honest view and its simulator's are the one empty view. -/
def muteZk (Stmt : Type) : Zk Stmt Unit Unit Unit :=
  ⟨fun _ _ _ => ⟨.ret (), PUnit.unit⟩, fun _ _ => ⟨.ret (), PUnit.unit⟩⟩

/-- The proof system whose honest view is `true` and whose simulator's is `false`. -/
def leakZk (Stmt : Type) : Zk Stmt Unit Unit Bool :=
  ⟨fun _ _ _ => ⟨.ret true, PUnit.unit⟩, fun _ _ => ⟨.ret false, PUnit.unit⟩⟩

/-- A proof system that says nothing is zero knowledge at no error, on every relation. -/
theorem muteZk_zkAt (Stmt : Type) (Rel : Stmt → Unit → Prop) : (muteZk Stmt).ZKAt Rel fun _ _ _ => 0 :=
  fun _ _ _ _ => Close.refl _

/-- A commitment that publishes nothing hides at no error. -/
theorem constHider_hiding : constHider.Hiding 0 :=
  fun _ _ _ _ => Close.refl _

/-- **`GenericSampledZKAt` needs its hiding premise**: with the premise dropped, at `ε_hide = 0`, it is false. Under a
proof system that says nothing and is zero knowledge at no error (`muteZk_zkAt`), a commitment that publishes the trace
puts `true` in the view at the trace `true` and `false` in the simulator's at the trace `false`. -/
theorem genericSampledZKAt_needs_hiding :
    ¬ ∀ (n : ℕ) (Tr Rnd Com Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
      (Zs : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop)
      (ε : Com × Finset (Fin n) → Wit → VS → ℝ≥0∞) (samp : VS → Com → Finset (Fin n))
      (wit : Tr → Rnd → Finset (Fin n) → Wit),
      Zs.ZKAt Rel ε → ∀ Xt : Tr, (∀ r S, Rel (H.commit Xt r, S) (wit Xt r S)) →
      ∀ (X₀ : Tr) (v : VS), Close (auditView H Zs samp wit Xt v) (auditSim H Zs samp X₀ v)
        (avg fun r => ε (H.commit Xt r, samp v (H.commit Xt r)) (wit Xt r (samp v (H.commit Xt r))) v) := by
  intro h
  have h0 := (h 0 Bool Unit Bool Unit Unit Unit openHider (muteZk _) (fun _ _ => True) (fun _ _ _ => 0)
    (fun _ _ => ∅) (fun _ _ _ => ()) (muteZk_zkAt _ _) true (fun _ _ => trivial) false () fun p => p.1 = true).1
  rw [auditView_pr, auditSim_pr] at h0
  have h1 : avg (fun _ : Unit => @ite ℝ≥0∞ (true = true) (Classical.propDecidable _) 1 0) ≤
      avg (fun _ : Unit => @ite ℝ≥0∞ (false = true) (Classical.propDecidable _) 1 0) +
        avg (fun _ : Unit => (0 : ℝ≥0∞)) := h0
  rw [avg_const, avg_const, avg_const, ite_eq_left rfl, ite_eq_right (by decide : ¬ false = true), add_zero] at h1
  exact absurd h1 (not_le.2 zero_lt_one)

/-- **`GenericSampledZKAt` needs its zero-knowledge premise**: with the premise dropped, at no session error, it is
false. Under a commitment that publishes nothing and hides at no error (`constHider_hiding`), a proof system whose
honest view is `true` and whose simulator's is `false` puts `true` in the audit's view and `false` in the
simulator's. -/
theorem genericSampledZKAt_needs_zk :
    ¬ ∀ (n : ℕ) (Tr Rnd Com Wit VS View : Type) [Fintype Rnd] [Nonempty Rnd] (H : Hider Tr Rnd Com)
      (Zs : Zk (Com × Finset (Fin n)) Wit VS View) (Rel : Com × Finset (Fin n) → Wit → Prop) (εh : ℝ≥0∞)
      (samp : VS → Com → Finset (Fin n)) (wit : Tr → Rnd → Finset (Fin n) → Wit),
      H.Hiding εh → ∀ Xt : Tr, (∀ r S, Rel (H.commit Xt r, S) (wit Xt r S)) →
      ∀ (X₀ : Tr) (v : VS), Close (auditView H Zs samp wit Xt v) (auditSim H Zs samp X₀ v) εh := by
  intro h
  have h0 := (h 0 Bool Unit Unit Unit Unit Bool constHider (leakZk _) (fun _ _ => True) 0 (fun _ _ => ∅)
    (fun _ _ _ => ()) constHider_hiding true (fun _ _ => trivial) false () fun p => p.2.2 = true).1
  rw [auditView_pr, auditSim_pr] at h0
  have h1 : avg (fun _ : Unit => @ite ℝ≥0∞ (true = true) (Classical.propDecidable _) 1 0) ≤
      avg (fun _ : Unit => @ite ℝ≥0∞ (false = true) (Classical.propDecidable _) 1 0) + 0 := h0
  rw [avg_const, avg_const, ite_eq_left rfl, ite_eq_right (by decide : ¬ false = true), add_zero] at h1
  exact absurd h1 (not_le.2 zero_lt_one)

/-! ## Sessions in turn: every session's error counts -/

/-- Any two experiments are `1`-close. -/
theorem Close.one {V : Type} (a b : Exp V) : Close a b 1 :=
  fun E => ⟨(prob_le_one E a.g a.s).trans le_add_self, (prob_le_one E b.g b.s).trans le_add_self⟩

/-- The proof system whose honest view is its statement and whose simulator's is `false`: it leaks at the statement
`true` and nowhere else. -/
def tellZk : Zk Bool Unit Unit Bool :=
  ⟨fun x _ _ => ⟨.ret x, PUnit.unit⟩, fun _ _ => ⟨.ret false, PUnit.unit⟩⟩

/-- `tellZk` is zero knowledge at no error at the statement `false`, and at error `1` at `true`. -/
theorem tellZk_zkAt (Rel : Bool → Unit → Prop) : tellZk.ZKAt Rel fun x _ _ => cond x 1 0 := by
  intro x w _ v
  cases x
  · exact Close.refl _
  · exact Close.one _ _

/-- **`GenericSessionsZKAt` needs every session's error**: with the sum stopped after the first session it is false.
`tellZk` is zero knowledge at its error (`tellZk_zkAt`), which is `0` at the first of the statements `false`, `true`;
over the two sessions the honest view is `[false, true]` and the simulator's is `[false, false]`. -/
theorem genericSessionsZKAt_needs_each :
    ¬ ∀ (Stmt Wit VS View : Type) (Zs : Zk Stmt Wit VS View) (Rel : Stmt → Wit → Prop) (ε : Stmt → Wit → VS → ℝ≥0∞),
      Zs.ZKAt Rel ε →
      Zs.seq.ZKAt (List.Forall₂ Rel) fun xs ws v => Zs.seqErr ε ((xs.zip ws).take 1) (v xs) := by
  intro h
  have h0 := (h Bool Unit Unit Bool tellZk (fun _ _ => True) (fun x _ _ => cond x 1 0) (tellZk_zkAt _)
    [false, true] [(), ()] (List.Forall₂.cons trivial (List.Forall₂.cons trivial List.Forall₂.nil)) (fun _ _ => ())
    fun l => l = [false, true]).1
  have h1 : @ite ℝ≥0∞ ([false, true] = [false, true]) (Classical.propDecidable _) 1 0 ≤
      @ite ℝ≥0∞ ([false, false] = [false, true]) (Classical.propDecidable _) 1 0 + (0 + 0) := h0
  rw [ite_eq_left rfl, ite_eq_right (by decide : ¬ [false, false] = [false, true]), add_zero, add_zero] at h1
  exact absurd h1 (not_le.2 zero_lt_one)

end FlockSoundness.SampledGeneric
