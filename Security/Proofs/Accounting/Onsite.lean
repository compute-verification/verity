import Definitions.Accounting.Guarantees.Onsite
import Proofs.Accounting.JointLive

/-!
# Proofs of the onsite bridges

`CertifierLinksDeclared` is the device model's links read on each run. `ClaimsImplyOnsiteClass` is the bridges read
pointwise: on a run where every claim holds and every wipe survives, each hypothesis of `Accounting.OnsiteClass` is a
premise or a bridge's conclusion (`onsiteClass_of`), so the event is empty. `AcceptedOnsiteClass` adds Layer 2: an
accepted run outside the class, with no break, has some subprotocol accepted with its claim failed or some wipe
accepted with its survivor bound failed, and the union bound over the one game sums their errors
(`Accounting.SecurityProofs.Live.prob_exists_univ_le`, `Verity.Game.prob_or_le`), as `JointPeriodLive`'s proof does.

The honest and nontrivial lemmas are on one constructed instance, a period of the empty circuit over the one-run game
(`Verity.Game.ret ()`) with an empty actual circuit (`emptyRun`): every hypothesis holds there
(`AcceptedOnsiteClass_honest`, `ClaimsImplyOnsiteClass_honest`, `CertifierLinksDeclared_honest`); with every verdict
accepting and a wipe whose survivor bound fails, `AcceptedOnsiteClass`'s bound fails with every bridge and device
premise in place (`AcceptedOnsiteClass_nontrivial`); without `WipeState`, `ClaimsImplyOnsiteClass`'s event has
probability 1 (`ClaimsImplyOnsiteClass_nontrivial`); and without the capacity premise, `CertifierLinks` fails
(`CertifierLinksDeclared_nontrivial`).
-/

namespace Accounting.SecurityProofs.Onsite

open Accounting Verity Verity.Game Finset

/-- **The links half from the links as declared** (`Accounting.Guarantees.Onsite.CertifierLinksDeclared`). -/
theorem CertifierLinksDeclared : Accounting.Guarantees.Onsite.CertifierLinksDeclared := by
  intro O L E _ S ends β link n X α F hD hcap o
  exact ⟨hD.2 o, hcap⟩

/-- On a run where every claim holds and every wipe survives, the device model and the bridges give the onsite
class. -/
theorem onsiteClass_of {ι Acc : Type} [Fintype Acc] {d : Device ι} {C : Verity.Circuit} {Cm O : Type}
    {r : PeriodLive d Acc C Cm O} {G : Verity.Game O} {c : Certificates} {p : Price} {m : ℕ}
    {w : Fin m → WipeLive d O} {f : ℝ} {L E : Type} [Fintype E] [DecidableEq L] {S : O → Actual.Run L}
    {ℓ : L → L → ℕ} {Φ : Finset L → ℕ} {ends : E → L × L} {β : E → ℕ}
    {link : ∀ o, Fin (S o).A.N → Fin (S o).A.N → E} {n : ℕ} {X : Fin n → Finset L}
    {θ : ∀ o, Finset (Fin (S o).A.N)} {hθ : ∀ o, ∀ a ∈ θ o, (S o).A.isInput a = true} {t ω : ℕ} {R F α : ℝ}
    {ref : Declaration} {s : Verity.Game.Strategy G}
    (hc : ∀ o, (S o).A.Causal ℓ) (hΦ : ∀ o, (S o).A.ComputeRate (Set.range X) (fun _ => 1) Φ)
    (hl : CertifierLinks S ends β link n X α F) (hw : Assumptions.WipeState S w f X t ω R)
    (hi : Assumptions.IntegrityContains r G s c p S X t θ hθ ref) {o : O}
    (hH : ∀ a, r.Holds c p G s a o) (hS : ∀ k, (w k).Survives f o) :
    OnsiteClass (S o).A ℓ Φ ends β (link o) n X (θ o) (hθ o) t ω R F α ref :=
  ⟨hc o, hΦ o, (hl o).1, (hl o).2, hw o hS, hi o (hH .sampled) (hH .pouw)⟩

/-- **The claims imply the onsite class** (`Accounting.Guarantees.Onsite.ClaimsImplyOnsiteClass`). -/
theorem ClaimsImplyOnsiteClass : Accounting.Guarantees.Onsite.ClaimsImplyOnsiteClass := by
  intro ι Acc _ d C Cm O r G c p m w f L E _ _ S ℓ Φ ends β link n X θ hθ t ω R F α ref s hc hΦ hl hw hi
  refine le_antisymm ?_ zero_le
  calc _ ≤ prob (fun _ => False) G s :=
        prob_mono (fun _ h => h.2 (onsiteClass_of hc hΦ hl hw hi h.1.1 h.1.2)) G s
    _ = 0 := prob_false G s

/-- **The accepted runs are in the onsite class, but for Layer 2's errors**
(`Accounting.Guarantees.Onsite.AcceptedOnsiteClass`). -/
theorem AcceptedOnsiteClass : Accounting.Guarantees.Onsite.AcceptedOnsiteClass := by
  intro ι Acc _ d C Cm O r G c p δ m w f εWipe L E _ _ S ℓ Φ ends β link n X θ hθ t ω R F α ref hδ hε s
    hc hΦ hl hw hi
  have hsub : ∀ o, (((∀ a, r.accepts a o) ∧ ∀ k, (w k).accepts o) ∧
      ¬ OnsiteClass (S o).A ℓ Φ ends β (link o) n X (θ o) (hθ o) t ω R F α ref ∧ ¬ r.Break G s o) →
      (∃ a, r.accepts a o ∧ ¬ r.Holds c p G s a o ∧ ¬ r.Break G s o) ∨
        ∃ k, (w k).accepts o ∧ ¬ (w k).Survives f o := by
    rintro o ⟨⟨hA, hW⟩, hn, hb⟩
    by_cases hH : ∀ a, r.Holds c p G s a o
    · by_cases hS : ∀ k, (w k).Survives f o
      · exact absurd (onsiteClass_of hc hΦ hl hw hi hH hS) hn
      · obtain ⟨k, hk⟩ := not_forall.1 hS
        exact Or.inr ⟨k, hW k, hk⟩
    · obtain ⟨a, ha⟩ := not_forall.1 hH
      exact Or.inl ⟨a, hA a, ha, hb⟩
  calc _ ≤ _ := prob_mono hsub G s
    _ ≤ _ := prob_or_le _ _ G s
    _ ≤ ∑ a, ENNReal.ofReal (δ a) + ∑ k, ENNReal.ofReal (εWipe k) :=
        add_le_add ((Accounting.SecurityProofs.Live.prob_exists_univ_le _ G s).trans
            (Finset.sum_le_sum fun a _ => hδ a s))
          ((Accounting.SecurityProofs.Live.prob_exists_univ_le _ G s).trans
            (Finset.sum_le_sum fun k _ => hε k s))

/-! ## The constructed instance -/

/-- The empty actual circuit, over the leaves `L`. -/
def emptyPlaced (L : Type) : Placed L where
  N := 0
  w := fun g => g.elim0
  isInput := fun g => g.elim0
  reads := fun g => g.elim0
  reads_lt := fun g => g.elim0
  input_reads := fun g => g.elim0
  op := fun g => g.elim0
  op_congr := fun g => g.elim0
  outputs := ∅
  x := fun g => g.elim0
  t := fun g => g.elim0
  cell := fun g => g.elim0
  cell_inj := fun g => g.elim0

/-- Its run: no gates, nothing held. -/
def emptyRun (L : Type) : Actual.Run L where
  A := emptyPlaced L
  val := fun g => g.elim0
  faithful := fun g => g.elim0
  held := fun _ _ => ∅
  holding := ⟨fun _ _ a ha => absurd ha (Finset.notMem_empty a), fun a => a.elim0, fun g => g.elim0⟩

/-- A device with no resources. -/
def device0 : Device Unit := ⟨0, 0, fun _ => 0, 0, fun _ => 0⟩

/-- The step that charges nothing. -/
def step0 : Step Unit Unit := ⟨fun _ _ => 0, fun _ => 0, fun _ _ => 0, fun _ => 0, fun _ => 0⟩

theorem step0_fits : step0.Fits device0 := by
  refine ⟨fun _ => ?_, ?_, fun _ => ?_, ?_, ?_⟩ <;> simp [step0, device0]

/-- A period of the empty circuit over the one-run game, every verdict `acc`, the network claim `True`. -/
def period (acc : Prop) : PeriodLive device0 Unit (emptyPlaced Unit).toCircuit Unit Unit where
  units := 0
  parts := ⟨fun g => g.elim0, fun g => g.elim0⟩
  acct := fun _ => ()
  win := ⟨0, 1⟩
  commit := fun _ => ()
  shows := fun _ _ _ => False
  schedule := fun _ _ => step0
  fits := fun _ _ => step0_fits
  commWithin := fun _ _ => True
  accepts := fun _ _ => acc

/-- The one-run game. -/
def game : Verity.Game Unit := .ret ()

/-- Certificates of nothing. -/
def certs0 : Certificates := ⟨0, 0, 0, 0, 0⟩

/-- One wipe of no fill, verdict `acc`, keeping `kept` bits. -/
def wipes (acc : Prop) (kept : ℕ) : Fin 1 → WipeLive device0 Unit :=
  fun _ => ⟨0, le_refl 0, fun _ => acc, fun _ => kept⟩

/-- Every run's actual circuit is the empty one, at one leaf. -/
def runs : Unit → Actual.Run Unit := fun _ => emptyRun Unit

/-- The reference circuit: the empty circuit, with no baseline. -/
def ref0 : Declaration := ⟨(emptyPlaced Unit).toCircuit, ∅, fun g hg => absurd hg (Finset.notMem_empty g)⟩

/-- The empty baseline of every run. -/
def θ0 : ∀ o, Finset (Fin (runs o).A.N) := fun _ => ∅

theorem hθ0 : ∀ o, ∀ a ∈ θ0 o, (runs o).A.isInput a = true := fun _ a ha => absurd ha (Finset.notMem_empty a)

theorem period_holds (acc : Prop) (s : Verity.Game.Strategy game) (a : Audit) (o : Unit) :
    (period acc).Holds certs0 (fun _ => 0) game s a o := by
  cases a
  · intro X _
    have : (unbound (period acc).parts ((period acc).opens game s o) X).card = 0 :=
      Finset.card_eq_zero.2 (Finset.eq_empty_of_forall_notMem fun u => u.elim0)
    simp [certs0, this]
  · exact Nat.cast_nonneg _
  · show (0 : ℝ) * _ ≤ _
    rw [zero_mul]
    exact Nat.cast_nonneg _
  · exact Nat.cast_nonneg _
  · trivial

theorem period_noBreak (acc : Prop) (s : Verity.Game.Strategy game) (o : Unit) : ¬ (period acc).Break game s o :=
  fun ⟨g, _⟩ => g.elim0

theorem runs_causal (ℓ : Unit → Unit → ℕ) (o : Unit) : (runs o).A.Causal ℓ := fun g => g.elim0

theorem runs_rate (n : ℕ) (X : Fin n → Finset Unit) (Φ : Finset Unit → ℕ) (o : Unit) :
    (runs o).A.ComputeRate (Set.range X) (fun _ => 1) Φ := by
  intro Y _ a b
  rw [Placed.work, Finset.sum_eq_zero fun g _ => g.elim0]
  exact Nat.zero_le _

theorem emptyRun_links {L E : Type} (ends : E → L × L) (β : E → ℕ) (link : Fin (emptyRun L).A.N → Fin (emptyRun L).A.N → E) :
    (emptyRun L).A.Links ends β link := by
  refine ⟨fun g => g.elim0, fun e y a b => ?_⟩
  rw [Circuit.width, Finset.sum_eq_zero fun g _ => g.elim0]
  exact Nat.zero_le _

theorem runs_state (o : Unit) (Y : Finset Unit) (u : ℕ) : (runs o).A.state Y u = 0 := by
  rw [Placed.state, Circuit.width, Finset.sum_eq_zero fun g _ => g.elim0]

theorem runs_links :
    CertifierLinks runs (fun _ : Unit => ((), ())) (fun _ => 0) (fun _ _ _ => ()) 1 (fun _ => {()}) 0 0 :=
  fun o => ⟨emptyRun_links _ _ _, fun _ => by simp [linkCapacity]⟩

theorem runs_contains (n : ℕ) (X : Fin n → Finset Unit) (t : ℕ) (o : Unit) :
    ref0.Contains ((runs o).A.declaration (univ.biUnion X) t (θ0 o) (hθ0 o)) :=
  ⟨⟨fun g => g.elim0, fun g => g.elim0, fun g => g.elim0, fun g => g.elim0, fun g => g.elim0, fun g => g.elim0,
    fun g => g.elim0⟩⟩

theorem runs_integrity (acc : Prop) (s : Verity.Game.Strategy game) :
    Assumptions.IntegrityContains (period acc) game s certs0 (fun _ => 0) runs (fun _ : Fin 1 => {()}) 1 θ0 hθ0
      ref0 :=
  fun o _ _ => runs_contains _ _ _ o

theorem runs_wipeState (kept : ℕ) : Assumptions.WipeState runs (wipes True kept) 0 (fun _ : Fin 1 => {()}) 1 1 0 :=
  fun o _ i k _ => by rw [runs_state]; simp

theorem wipes_survive (acc : Prop) (o : Unit) (k : Fin 1) : (wipes acc 0 k).Survives 0 o := by
  simp only [WipeLive.Survives, wipes, device0]
  norm_num

theorem wipes_fail (acc : Prop) (o : Unit) (k : Fin 1) : ¬ (wipes acc 1000 k).Survives 0 o := by
  simp only [WipeLive.Survives, wipes, device0]
  norm_num

/-- **`CertifierLinksDeclared` is honest**: at the empty actual circuit on one leaf, one run, one link of no capacity
and `α F = 0`, its premises hold: the links are as declared and the boundary capacity is at most `α F`. -/
theorem CertifierLinksDeclared_honest :
    DeclaredLinks runs (fun _ : Unit => ((), ())) (fun _ => 0) (fun _ _ _ => ()) ∧
      ∀ i : Fin 1, (linkCapacity (fun _ : Unit => ((), ())) (fun _ => 0) ((fun _ : Fin 1 => ({()} : Finset Unit)) i) :
        ℝ) ≤ 0 * 0 := by
  refine ⟨⟨fun e y u => ?_, fun o => emptyRun_links _ _ _⟩, fun _ => by simp [linkCapacity]⟩
  rw [Set.range_unique, Set.encard_singleton]
  simp

/-- **`CertifierLinksDeclared` is nontrivial**: without its capacity premise, its conclusion fails. Two leaves, the
location `{true}` and one link of capacity 1 bit a tick joining it to `false`: the links are as declared on the empty
actual circuit, and the location's boundary capacity, 1, is above `α F = 0`. -/
theorem CertifierLinksDeclared_nontrivial :
    DeclaredLinks (fun _ : Unit => emptyRun Bool) (fun _ : Unit => (true, false)) (fun _ => 1) (fun _ _ _ => ()) ∧
      ¬ CertifierLinks (fun _ : Unit => emptyRun Bool) (fun _ : Unit => (true, false)) (fun _ => 1)
        (fun _ _ _ => ()) 1 (fun _ => {true}) 0 0 := by
  refine ⟨⟨fun e y u => ?_, fun o => emptyRun_links _ _ _⟩, fun h => ?_⟩
  · rw [Set.range_unique, Set.encard_singleton]
    simp
  · have := (h ()).2 0
    simp [linkCapacity, boundary] at this
    exact absurd this (by norm_num)

/-- **`AcceptedOnsiteClass` is honest**: at the constructed instance (a period of the empty circuit over the one-run
game, every verdict accepting, one wipe of no fill keeping nothing, the empty actual circuit at one leaf, the reference
circuit empty), every hypothesis holds: the five subprotocols' per-period guarantees at `δ = 0`, the wipe's at
`ε_wipe = 0`, and at every strategy the device model and the three bridges. -/
theorem AcceptedOnsiteClass_honest :
    (∀ a, (period True).Sound game certs0 (fun _ => 0) a 0) ∧
    (∀ (k : Fin 1) (s : Verity.Game.Strategy game),
      prob (fun o => (wipes True 0 k).accepts o ∧ ¬ (wipes True 0 k).Survives 0 o) game s ≤ ENNReal.ofReal 0) ∧
    ∀ s : Verity.Game.Strategy game,
      (∀ o, (runs o).A.Causal fun _ _ => 0) ∧
      (∀ o, (runs o).A.ComputeRate (Set.range fun _ : Fin 1 => ({()} : Finset Unit)) (fun _ => 1) fun _ => 0) ∧
      CertifierLinks runs (fun _ : Unit => ((), ())) (fun _ => 0) (fun _ _ _ => ()) 1 (fun _ => {()}) 0 0 ∧
      Assumptions.WipeState runs (wipes True 0) 0 (fun _ : Fin 1 => {()}) 1 1 0 ∧
      Assumptions.IntegrityContains (period True) game s certs0 (fun _ => 0) runs (fun _ : Fin 1 => {()}) 1 θ0
        hθ0 ref0 := by
  refine ⟨fun a s => ?_, fun k s => ?_, fun s => ⟨runs_causal _, runs_rate _ _ _, runs_links, runs_wipeState 0,
    runs_integrity True s⟩⟩
  · show prob _ (.ret ()) s ≤ _
    simp only [prob]
    split_ifs with h
    · exact absurd (period_holds True s a ()) h.2.1
    · exact zero_le
  · show prob _ (.ret ()) s ≤ _
    simp only [prob]
    split_ifs with h
    · exact absurd (wipes_survive True () k) h.2
    · exact zero_le

/-- **`ClaimsImplyOnsiteClass` is honest**: at `AcceptedOnsiteClass_honest`'s instance, every hypothesis holds, and
every claim and the wipe's survivor bound hold on its run, so its event is the class failing alone. -/
theorem ClaimsImplyOnsiteClass_honest :
    ∀ s : Verity.Game.Strategy game,
      (∀ o, (runs o).A.Causal fun _ _ => 0) ∧
      (∀ o, (runs o).A.ComputeRate (Set.range fun _ : Fin 1 => ({()} : Finset Unit)) (fun _ => 1) fun _ => 0) ∧
      CertifierLinks runs (fun _ : Unit => ((), ())) (fun _ => 0) (fun _ _ _ => ()) 1 (fun _ => {()}) 0 0 ∧
      Assumptions.WipeState runs (wipes True 0) 0 (fun _ : Fin 1 => {()}) 1 1 0 ∧
      Assumptions.IntegrityContains (period True) game s certs0 (fun _ => 0) runs (fun _ : Fin 1 => {()}) 1 θ0
        hθ0 ref0 ∧
      ∀ o, (∀ a, (period True).Holds certs0 (fun _ => 0) game s a o) ∧ ∀ k, (wipes True 0 k).Survives 0 o :=
  fun s => ⟨runs_causal _, runs_rate _ _ _, runs_links, runs_wipeState 0, runs_integrity True s,
    fun o => ⟨fun a => period_holds True s a o, fun k => wipes_survive True o k⟩⟩

/-- **`AcceptedOnsiteClass` is nontrivial**: with every verdict accepting (the five subprotocols' and the wipe's) and a
wipe that keeps 1000 bits, above its survivor bound of 256, the device model and the three bridges hold at `R = −1`
(`WipeState` because the wipe's bound fails), the run's actual circuit is outside the class (its state, 0, is above
`R`), and the bound at `δ = 0` and `ε_wipe = 0` fails: the event has probability 1. -/
theorem AcceptedOnsiteClass_nontrivial :
    ∀ s : Verity.Game.Strategy game,
      (∀ o, (runs o).A.Causal fun _ _ => 0) ∧
      (∀ o, (runs o).A.ComputeRate (Set.range fun _ : Fin 1 => ({()} : Finset Unit)) (fun _ => 1) fun _ => 0) ∧
      CertifierLinks runs (fun _ : Unit => ((), ())) (fun _ => 0) (fun _ _ _ => ()) 1 (fun _ => {()}) 0 0 ∧
      Assumptions.WipeState runs (wipes True 1000) 0 (fun _ : Fin 1 => {()}) 1 1 (-1) ∧
      Assumptions.IntegrityContains (period True) game s certs0 (fun _ => 0) runs (fun _ : Fin 1 => {()}) 1 θ0
        hθ0 ref0 ∧
      ¬ prob (fun o => ((∀ a, (period True).accepts a o) ∧ ∀ k, (wipes True 1000 k).accepts o) ∧
          ¬ OnsiteClass (runs o).A (fun _ _ => 0) (fun _ => 0) (fun _ : Unit => ((), ())) (fun _ => 0)
            (fun _ _ => ()) 1 (fun _ => {()}) (θ0 o) (hθ0 o) 1 1 (-1) 0 0 ref0 ∧
          ¬ (period True).Break game s o) game s ≤
        ∑ _a : Audit, ENNReal.ofReal 0 + ∑ _k : Fin 1, ENNReal.ofReal 0 := by
  intro s
  refine ⟨runs_causal _, runs_rate _ _ _, runs_links, fun o h => absurd (h 0) (wipes_fail True o 0),
    runs_integrity True s, ?_⟩
  have hn : ¬ OnsiteClass (runs ()).A (fun _ _ => 0) (fun _ => 0) (fun _ : Unit => ((), ())) (fun _ => 0)
      (fun _ _ => ()) 1 (fun _ => {()}) (θ0 ()) (hθ0 ()) 1 1 (-1) 0 0 ref0 := fun h => by
    have := h.2.2.2.2.1 0 0 (by norm_num)
    rw [runs_state] at this
    norm_num at this
  show ¬ prob _ (.ret ()) s ≤ _
  simp only [prob, ENNReal.ofReal_zero, Finset.sum_const_zero, add_zero]
  split_ifs with h
  · simp
  · exact absurd ⟨⟨fun _ => trivial, fun _ => trivial⟩, hn, period_noBreak True s ()⟩ h

/-- **`ClaimsImplyOnsiteClass` is nontrivial**: without `WipeState`, its conclusion fails. At `R = −1`, with the wipe
keeping nothing, every other hypothesis holds and every claim and the survivor bound hold on the run, `WipeState`
fails, and the event, claims holding with the actual circuit outside the class, has probability 1. -/
theorem ClaimsImplyOnsiteClass_nontrivial :
    ∀ s : Verity.Game.Strategy game,
      (∀ o, (runs o).A.Causal fun _ _ => 0) ∧
      (∀ o, (runs o).A.ComputeRate (Set.range fun _ : Fin 1 => ({()} : Finset Unit)) (fun _ => 1) fun _ => 0) ∧
      CertifierLinks runs (fun _ : Unit => ((), ())) (fun _ => 0) (fun _ _ _ => ()) 1 (fun _ => {()}) 0 0 ∧
      ¬ Assumptions.WipeState runs (wipes True 0) 0 (fun _ : Fin 1 => {()}) 1 1 (-1) ∧
      Assumptions.IntegrityContains (period True) game s certs0 (fun _ => 0) runs (fun _ : Fin 1 => {()}) 1 θ0
        hθ0 ref0 ∧
      prob (fun o => ((∀ a, (period True).Holds certs0 (fun _ => 0) game s a o) ∧ ∀ k, (wipes True 0 k).Survives 0 o) ∧
          ¬ OnsiteClass (runs o).A (fun _ _ => 0) (fun _ => 0) (fun _ : Unit => ((), ())) (fun _ => 0)
            (fun _ _ => ()) 1 (fun _ => {()}) (θ0 o) (hθ0 o) 1 1 (-1) 0 0 ref0) game s = 1 := by
  intro s
  have hst : ¬ ((runs ()).A.state ((fun _ : Fin 1 => ({()} : Finset Unit)) 0) (0 * 1) : ℝ) ≤ -1 := by
    rw [runs_state]
    norm_num
  refine ⟨runs_causal _, runs_rate _ _ _, runs_links, fun h => hst (h () (fun k => wipes_survive True () k) 0 0
    (by norm_num)), runs_integrity True s, ?_⟩
  show prob _ (.ret ()) s = 1
  simp only [prob]
  split_ifs with h
  · rfl
  · exact absurd ⟨⟨fun a => period_holds True s a (), fun k => wipes_survive True () k⟩,
      fun h => hst (h.2.2.2.2.1 0 0 (by norm_num))⟩ h

end Accounting.SecurityProofs.Onsite
