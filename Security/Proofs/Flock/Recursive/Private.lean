import Proofs.Flock.Recursive.Guarantees
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Hides
import Proofs.Flock.Verifier.Registered

/-!
# Proofs of the recursive audit's circuit privacy, and that it ran the registered private circuit

* `recursiveCircuitPrivate_le` is `circuit_private_of_zk`'s argument with `RecursiveZK` as the step: at each
  registration salt `r`, `RecursiveZK` at an hm96 hiding bound `δ` (`recursiveZK_le`; and, for the other way, the same
  on the event's complement) moves
  the firewall's commitments and V*'s sessions to the ideal leaves and V*'s simulator, which read `r` only through the
  registration's view `reg P r`; then the registration's hiding, as a test with values in `[0, 1]` (`Hides.avg_le`),
  moves `reg P r` to `reg p₀ r`. `Flock.SecurityProofs.RecursiveCircuitPrivate` is it at the pinned key's hiding gap
  (`CROnly.hm96Hiding_gap`).
* `RecursiveCircuitIndist` is the two ways of `recursiveCircuitPrivate_le` through the simulator at `p₀ = P`, at the gap.
* `regLeaves_hides`: a registration by hm96 leaves hides at `2·n_r·δ`, `ideal_leaves_swap` from each private circuit to the
  ideal leaves and back (on the complement); `recursiveCircuitPrivateLeaves_le` is `recursiveCircuitPrivate_le` with it,
  `RecursiveCircuitPrivateLeaves` that at the gap, and `RecursiveAudit`'s (ii) the same at the firewall, at the gap
  (`Proofs.Flock.Recursive.FirewallZK`).
* `RanRegisteredRun`: an accepted outcome off the registered program is, by the binding, one whose opening and the
  registrant's collide; with no such outcome reached (`prob_mono_outcome`), accepting off `p` adds nothing to `B`.
* `RecursiveRanRegistered`: `RecursiveSound`'s three cases; in its bound's, the same argument at each row the run read
  (its opening and the developer's) gives that the registered private circuit has the rows the run read, so the hidden statement
  is `p`'s (`_hx`).
* `regRowCommit_binding`: `RecursiveRanRegistered`'s `_hbind` for the executable's registered rows. An opening of row
  `i` is the verifier's check of a registered read (`regRowCommit_opens`: `Flock.Registered.opens` of the row's commit
  string), and two of one port at `i` to different rows collide, in hm96's leaf (`ExplicitLeaf.hm96`) or along the
  registered path (`Flock.Registered.opensLeaf_binding`).
-/

namespace Flock.SecurityProofs

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion FlockSoundness.ZK
open FlockSoundness.Discharge.PrivateCircuit
open Game
open scoped ENNReal

section Private

theorem prCoin_add_compl {C : Type} [Fintype C] [Nonempty C] (P : C → Prop) :
    prCoin P + prCoin (fun c => ¬ P c) = 1 := by
  classical
  unfold prCoin
  rw [← avg_add]
  calc avg _ = avg (fun _ : C => (1 : ℝ≥0∞)) := by
        congr 1
        funext c
        by_cases hc : P c <;> simp [hc]
    _ = 1 := avg_const _

/-- **A bound on every complement is a bound the other way**: `Pr_A[¬f] ≤ Pr_B[¬g] + δ` gives `Pr_B[g] ≤ Pr_A[f] + δ`. -/
theorem prCoin_flip {A B : Type} [Fintype A] [Nonempty A] [Fintype B] [Nonempty B] (f : A → Prop) (g : B → Prop)
    {δ : ℝ≥0∞} (h : prCoin (fun x => ¬ f x) ≤ prCoin (fun y => ¬ g y) + δ) : prCoin g ≤ prCoin f + δ := by
  have a1 := prCoin_add_compl f
  have a2 := prCoin_add_compl g
  have hfin : prCoin (fun x => ¬ f x) ≠ ⊤ :=
    ne_top_of_le_ne_top ENNReal.one_ne_top (le_of_le_of_eq le_add_self a1)
  refine (ENNReal.add_le_add_iff_right hfin).1 ?_
  calc prCoin g + prCoin (fun x => ¬ f x) ≤ prCoin g + (prCoin (fun y => ¬ g y) + δ) := add_le_add le_rfl h
    _ = (prCoin g + prCoin (fun y => ¬ g y)) + δ := by ring
    _ = (prCoin f + prCoin (fun x => ¬ f x)) + δ := by rw [a1, a2]
    _ = prCoin f + δ + prCoin (fun x => ¬ f x) := by ring

variable {Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V : Type} [Fintype Rr] [Nonempty Rr] [Fintype Ωa] [Nonempty Ωa]
  [Fintype Ωp] [Nonempty Ωp] [Fintype Ωo] [Nonempty Ωo] [Fintype Ωs] [Nonempty Ωs]

/-- The simulated view's probability of `E` at the registration's view `v`: ideal commitments and V*'s simulator. -/
noncomputable def idealAt (nh : ℕ) (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) (o : Out)
    (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) (v : Rv) : ℝ≥0∞ :=
  prCoin fun z : (Ωa × (Fin nh → ZkView.Bits × ZkView.Salt)) × Ωs =>
    E (v, z.1.1, (fun i => Flock.Guarantees.idealLeaf (z.1.2 i)),
      sim o v z.1.1 (fun i => Flock.Guarantees.idealLeaf (z.1.2 i)) z.2)

/-- The real view's probability of `E` at the registration's randomness `r`. -/
noncomputable def realAt (reg : Prog → Rr → Rv) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (P : Prog) (w : Wit) (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) (r : Rr) : ℝ≥0∞ :=
  prCoin fun z : ((Ωa × Ωp) × (Fin nh → ZkView.Salt)) × Ωo =>
    E (reg P r, z.1.1.1, (fun i => Flock.Guarantees.firewallLeaf (msgs P w r z.1.1.1 z.1.1.2 i) (z.1.2 i)),
      view P w r z.1.1.1 (fun i => Flock.Guarantees.firewallLeaf (msgs P w r z.1.1.1 z.1.1.2 i) (z.1.2 i))
        z.1.1.2 z.1.2 z.2)

omit [Fintype Rr] [Nonempty Rr] in
/-- **`RecursiveZK` at one registration salt, both ways**, at any hm96 hiding bound `δ` (`recursiveZK_le`). -/
theorem recZK_at (reg : Prog → Rr → Rv) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞} (o : Out) (P : Prog) (w : Wit) (r : Rr)
    (hOuter : ∀ a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs =>
            E (sim o (reg P r) a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) :
    realAt reg nh msgs view P w E r ≤ idealAt nh sim o E (reg P r) + (εo + nh * δ) ∧
      idealAt nh sim o E (reg P r) ≤ realAt reg nh msgs view P w E r + (εo + nh * δ) := by
  have zk := fun E' => recursiveZK_le nh (msgs P w r) (view P w r) (sim o (reg P r)) hOuter hT1 E'
  refine ⟨(zk fun t => E (reg P r, t.1, t.2.1, t.2.2)).trans (le_of_eq (add_assoc _ _ _)), ?_⟩
  refine prCoin_flip _ _ ((zk fun t => ¬ E (reg P r, t.1, t.2.1, t.2.2)).trans (le_of_eq (add_assoc _ _ _)))

/-- **`RecursiveCircuitPrivate` at any hm96 hiding bound**: `Flock.Guarantees.RecursiveCircuitPrivate` with
`Assumptions.Hm96Hiding` at `δ` taken as a hypothesis, and `nh·δ` in place of `nh·hidingGap My saltHash`. -/
theorem recursiveCircuitPrivate_le (reg : Prog → Rr → Rv) {ε : ℝ≥0∞} (hreg : Hides reg ε)
    (honest : Out → Prog → Wit → Prop) (nh : ℕ) (msgs : Prog → Wit → Rr → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → Rr → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs =>
            E (sim o (reg P r) a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit) (hw : honest o P w) (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) :
    Flock.Guarantees.recRealPr reg nh msgs view P w E ≤ Flock.Guarantees.recSimPr reg nh sim o p₀ E + (ε + εo + nh * δ) ∧
      Flock.Guarantees.recSimPr reg nh sim o p₀ E ≤ Flock.Guarantees.recRealPr reg nh msgs view P w E +
        (ε + εo + nh * δ) := by
  have hf1 : ∀ v, idealAt nh sim o E v ≤ 1 := fun v => prCoin_le_one' _
  have hstep := fun r => recZK_at reg nh msgs view sim o P w r (hOuter o P w hw r) hT1 E
  have hreal : Flock.Guarantees.recRealPr reg nh msgs view P w E = avg (realAt reg nh msgs view P w E) := rfl
  have hsim : Flock.Guarantees.recSimPr reg nh sim o p₀ E = avg fun r => idealAt nh sim o E (reg p₀ r) := rfl
  rw [hreal, hsim]
  constructor
  · calc avg (realAt reg nh msgs view P w E)
        ≤ avg (fun r => idealAt nh sim o E (reg P r) + (εo + nh * δ)) :=
          avg_mono fun r => (hstep r).1
      _ = avg (fun r => idealAt nh sim o E (reg P r)) + (εo + nh * δ) := by
          rw [avg_add, avg_const]
      _ ≤ (avg (fun r => idealAt nh sim o E (reg p₀ r)) + ε) + (εo + nh * δ) :=
          add_le_add (hreg.avg_le P p₀ (idealAt nh sim o E) hf1) le_rfl
      _ = avg (fun r => idealAt nh sim o E (reg p₀ r)) + (ε + εo + nh * δ) := by ring
  · calc avg (fun r => idealAt nh sim o E (reg p₀ r))
        ≤ avg (fun r => idealAt nh sim o E (reg P r)) + ε := hreg.avg_le p₀ P (idealAt nh sim o E) hf1
      _ ≤ avg (fun r => realAt reg nh msgs view P w E r + (εo + nh * δ)) + ε :=
          add_le_add (avg_mono fun r => (hstep r).2) le_rfl
      _ = avg (realAt reg nh msgs view P w E) + (ε + εo + nh * δ) := by
          rw [avg_add, avg_const]; ring

/-- **The recursive audit is circuit private.** -/
theorem RecursiveCircuitPrivate : Flock.Guarantees.RecursiveCircuitPrivate := by
  intro Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V _ _ _ _ _ _ _ _ _ _ reg ε hreg honest nh msgs view sim εo hOuter p₀ o P w hw E
  exact recursiveCircuitPrivate_le reg hreg honest nh msgs view sim hOuter
    (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) p₀ o P w hw E

/-- **Two private circuits of a class are indistinguishable in the recursive audit.** -/
theorem RecursiveCircuitIndist : Flock.Guarantees.RecursiveCircuitIndist := by
  intro Out Prog Wit Rr Rv Ωa Ωp Ωo Ωs V _ _ _ _ _ _ _ _ _ _ reg ε hreg honest nh msgs view sim εo hOuter o P P'
    w w' hw hw' E
  have hT1 := CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash
  have h := (recursiveCircuitPrivate_le reg hreg honest nh msgs view sim hOuter hT1 P o P w hw E).1
  have h' := (recursiveCircuitPrivate_le reg hreg honest nh msgs view sim hOuter hT1 P o P' w' hw' E).2
  calc Flock.Guarantees.recRealPr reg nh msgs view P w E
      ≤ Flock.Guarantees.recSimPr reg nh sim o P E +
          (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) := h
    _ ≤ (Flock.Guarantees.recRealPr reg nh msgs view P' w' E +
          (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash))) +
          (ε + εo + nh * ENNReal.ofReal (CROnly.hidingGap ZkView.My ZkView.saltHash)) := add_le_add h' le_rfl
    _ = _ := by ring

/-- The unit, dropped from an experiment's randomness. -/
def unitProd (T : Type) : Unit × T ≃ T where
  toFun p := p.2
  invFun x := ((), x)
  left_inv _ := rfl
  right_inv _ := rfl

/-- **T1 for a private circuit's rows**: the leaves of `Q`'s rows under fresh salts are within `n_r·δ` of ideal leaves, at
every event `G` of the leaves, at any hm96 hiding bound `δ`. -/
theorem rowLeaves_swap {nr : ℕ} (rows : Prog → Fin nr → ByteArray)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ) (Q : Prog)
    (G : (Fin nr → ByteArray) → Prop) :
    prCoin (fun r : Fin nr → ZkView.Salt => G fun i => Flock.Guarantees.firewallLeaf (rows Q i) (r i)) ≤
      prCoin (fun u : Fin nr → ZkView.Bits × ZkView.Salt => G fun i => Flock.Guarantees.idealLeaf (u i)) + nr * δ :=
  calc prCoin (fun r : Fin nr → ZkView.Salt => G fun i => Flock.Guarantees.firewallLeaf (rows Q i) (r i))
      = prCoin (fun p : Unit × (Fin nr → ZkView.Salt) =>
          G fun i => Flock.Guarantees.firewallLeaf (rows Q i) (p.2 i)) :=
        (prCoin_congr_equiv (unitProd _)
          (fun r : Fin nr → ZkView.Salt => G fun i => Flock.Guarantees.firewallLeaf (rows Q i) (r i))).symm
    _ ≤ prCoin (fun p : Unit × (Fin nr → ZkView.Bits × ZkView.Salt) =>
          G fun i => Flock.Guarantees.idealLeaf (p.2 i)) + nr * δ :=
        ideal_leaves_swap ZkView.My ZkView.saltHash hT1 ZkView.leafH nr (Ω := Unit)
          (fun _ i => ZkView.bitsOf (Sha512.hash (rows Q i))) (fun _ l => G l)
    _ = prCoin (fun u : Fin nr → ZkView.Bits × ZkView.Salt => G fun i => Flock.Guarantees.idealLeaf (u i)) + nr * δ :=
        congrArg (· + nr * δ) (prCoin_congr_equiv (unitProd _)
          (fun u : Fin nr → ZkView.Bits × ZkView.Salt => G fun i => Flock.Guarantees.idealLeaf (u i)))

/-- **A registration by hm96 leaves hides**, at `2·n_r·δ` at any hm96 hiding bound `δ`: each private circuit's leaves
are within `n_r·δ` of ideal ones (`rowLeaves_swap`), the second way on the complement. -/
theorem regLeaves_hides {nr : ℕ} (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ) :
    Hides (Flock.Guarantees.regLeaves rows pub) (2 * nr * δ) := by
  intro P P' E
  have h1 := rowLeaves_swap rows hT1 P fun l => E (pub l)
  have h2 := prCoin_flip _ _ (rowLeaves_swap rows hT1 P' fun l => ¬ E (pub l))
  calc prCoin (fun r => E (Flock.Guarantees.regLeaves rows pub P r))
      ≤ prCoin (fun u : Fin nr → ZkView.Bits × ZkView.Salt => E (pub fun i => Flock.Guarantees.idealLeaf (u i))) +
          nr * δ := h1
    _ ≤ (prCoin (fun r => E (Flock.Guarantees.regLeaves rows pub P' r)) + nr * δ) + nr * δ := add_le_add h2 le_rfl
    _ = prCoin (fun r => E (Flock.Guarantees.regLeaves rows pub P' r)) + 2 * nr * δ := by ring

/-- **`RecursiveCircuitPrivateLeaves` at any hm96 hiding bound**: `Flock.Guarantees.RecursiveCircuitPrivateLeaves` with
`Assumptions.Hm96Hiding` at `δ` taken as a hypothesis, and `δ` in place of `hidingGap My saltHash`. -/
theorem recursiveCircuitPrivateLeaves_le (nr : ℕ) (rows : Prog → Fin nr → ByteArray) (pub : (Fin nr → ByteArray) → Rv)
    (honest : Out → Prog → Wit → Prop)
    (nh : ℕ) (msgs : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → Ωp → Fin nh → ByteArray)
    (view : Prog → Wit → (Fin nr → ZkView.Salt) → Ωa → (Fin nh → ByteArray) → Ωp → (Fin nh → ZkView.Salt) → Ωo → V)
    (sim : Out → Rv → Ωa → (Fin nh → ByteArray) → Ωs → V) {εo : ℝ≥0∞}
    (hOuter : ∀ o P w, honest o P w → ∀ r a p (y : Fin nh → ZkView.Salt) (E : V → Prop),
      prCoin (fun q : Ωo => E (view P w r a (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)) p y q))
        ≤ prCoin (fun s : Ωs =>
            E (sim o (Flock.Guarantees.regLeaves rows pub P r) a
              (fun i => Flock.Guarantees.firewallLeaf (msgs P w r a p i) (y i)) s)) + εo)
    {δ : ℝ≥0∞} (hT1 : FlockSoundness.Assumptions.Hm96Hiding ZkView.My ZkView.saltHash δ)
    (p₀ : Prog) (o : Out) (P : Prog) (w : Wit) (hw : honest o P w) (E : Rv × Ωa × (Fin nh → ByteArray) × V → Prop) :
    Flock.Guarantees.recRealPr (Flock.Guarantees.regLeaves rows pub) nh msgs view P w E ≤
        Flock.Guarantees.recSimPr (Flock.Guarantees.regLeaves rows pub) nh sim o p₀ E + (2 * nr * δ + εo + nh * δ) ∧
      Flock.Guarantees.recSimPr (Flock.Guarantees.regLeaves rows pub) nh sim o p₀ E ≤
        Flock.Guarantees.recRealPr (Flock.Guarantees.regLeaves rows pub) nh msgs view P w E + (2 * nr * δ + εo + nh * δ) :=
  recursiveCircuitPrivate_le (Flock.Guarantees.regLeaves rows pub) (regLeaves_hides rows pub hT1) honest nh msgs view sim
    hOuter hT1 p₀ o P w hw E

/-- **The recursive audit is circuit private, the private circuit registered by hm96 leaves.** -/
theorem RecursiveCircuitPrivateLeaves : Flock.Guarantees.RecursiveCircuitPrivateLeaves := by
  intro Out Prog Wit Rv Ωa Ωp Ωo Ωs V _ _ _ _ _ _ _ _ nr rows pub honest nh msgs view sim εo hOuter p₀ o P w hw E
  exact recursiveCircuitPrivateLeaves_le nr rows pub honest nh msgs view sim hOuter
    (CROnly.hm96Hiding_gap ZkView.My ZkView.saltHash) p₀ o P w hw E

end Private

/-- **A bound at every outcome the strategy reaches is a bound**: if every reached outcome with `E` has `F`, then
`Pr[E] ≤ Pr[F]`. -/
theorem prob_mono_outcome {α : Type} {E F : α → Prop} :
    ∀ (g : Game α) (s : Strategy g), (∀ a, Game.Outcome g s a → E a → F a) → prob E g s ≤ prob F g s
  | .ret a, _, h => by
      classical
      simp only [prob]
      by_cases hE : E a
      · simp [hE, h a rfl hE]
      · simp [hE]
  | .send _ k, s, h => prob_mono_outcome (k s.1) s.2 h
  | @Game.coin _ _ _ _ k, s, h =>
      Finset.sum_le_sum fun c _ => mul_le_mul_right (prob_mono_outcome (k c) (s c) fun a ha => h a ⟨c, ha⟩) _

theorem RanRegisteredRun : Flock.Guarantees.RanRegisteredRun := by
  intro Root P Op com hbind rt p op₀ h₀ β G σ acc Q prog op hacc B hB
  by_cases hb : ∃ b, Game.Outcome G σ b ∧ acc b ∧ prog b ≠ p ∧ Binding.Collides (com.ext (op b) op₀)
  · exact Or.inl hb
  · refine Or.inr (le_trans (prob_mono_outcome (E := fun b => acc b ∧ (Q b ∨ prog b ≠ p))
      (F := fun b => acc b ∧ Q b) G σ ?_) hB)
    rintro b hr ⟨ha, hq⟩
    exact ⟨ha, hq.resolve_right fun hne => hb ⟨b, hr, ha, hne, hbind rt _ _ _ _ (hacc b ha) h₀ hne⟩⟩

theorem RecursiveRanRegistered : Flock.Guarantees.RecursiveRanRegistered := by
  intro n L Reg I Cm S Vs Reg' Reg₂ hA3 zs dg N k Rw M hk hRw hM ρ hρ hr σ x wmsg Sk cmt msg hbr εin hin
    Root Pos Row Op com hbind rt p op₀ hreg stmt reads row op hop hx
  rcases RecursiveSound hA3 zs dg N hk hRw hM hρ hr σ x wmsg Sk cmt msg hbr hin with hf | hf | hf | hb
  · exact Or.inl hf
  · exact Or.inr (Or.inl hf)
  · exact Or.inr (Or.inr (Or.inl hf))
  · by_cases hc : ∃ o, Game.Outcome (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ o ∧
        AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
        ∃ i, reads o i ∧ row o i ≠ p i ∧ Binding.Collides ((com i).ext (op o i) (op₀ i))
    · exact Or.inr (Or.inr (Or.inr (Or.inl hc)))
    · have hmono : prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (stmt p o.2.1))
            (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ ≤
          prob (fun o => AcceptsV (zs o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ¬ I.holds (x o.1 o.2.1))
            (recGame L Reg I Cm fun R ω t => outerV (zs R ω t) dg N) σ := by
        refine prob_mono_outcome _ σ fun o hro h => ⟨h.1, fun hh => ?_⟩
        have hagree : ∀ i, reads o i → p i = row o i := fun i hi => by
          by_contra hne
          exact hc ⟨o, hro, h.1, i, hi, fun he => hne he.symm,
            hbind i rt (row o i) (p i) (op o i) (op₀ i) (hop o h.1 i hi) (hreg i) fun he => hne he.symm⟩
        rw [hx o h.1 p hagree] at hh
        exact h.2 hh
      exact Or.inr (Or.inr (Or.inr (Or.inr (hmono.trans hb))))

section Rows

open FlockSoundness.Binding Refine

/-- A collision on byte arrays is one on their byte lists. -/
theorem collides_of_bytes {x : ByteArray × ByteArray} (h : Flock.Registered.Collides x) :
    Binding.Collides (x.1.data.toList, x.2.data.toList) := by
  refine ⟨fun e => h.1 ?_, Subtype.ext ?_⟩
  · exact (bytes_of_list x.1).symm.trans ((congrArg (fun l : List UInt8 => (⟨l.toArray⟩ : ByteArray)) e).trans
      (bytes_of_list x.2))
  · show Sha512.hash ⟨x.1.data.toList.toArray⟩ = Sha512.hash ⟨x.2.data.toList.toArray⟩
    rw [bytes_of_list, bytes_of_list]
    exact h.2

open Classical in
/-- **The registration's commitment at row position `i`**, as the verifier checks a registered read. An opening
`(reg, v, salt, path)` of the port `reg` (its root, domain and leaf count) to the row `v`, encoded injectively by `enc`,
is the verifier's `opensLeaf` of `v`'s hm96-sha512 leaf under `salt` at `i` through `path` (`regRowCommit_opens`). The
extractor returns hm96's pair when the two openings' leaves agree, and the registered path's (`openPair`) otherwise. -/
noncomputable def regRowCommit {Val : Type} (enc : Val → List UInt8) (hinj : Function.Injective enc) (i : ℕ) :
    Discharge.PrivateCircuit.Commit Flock.Registered.Port Val
      (Flock.Registered.Port × Val × ByteArray × ByteArray) where
  Opens reg v op := op.1 = reg ∧ op.2.1 = v ∧
    Flock.Registered.opensLeaf reg ((ExplicitLeaf.hm96 enc hinj).leaf v op.2.2.1.data.toList).1 i op.2.2.2 = true
  ext op op' :=
    if (ExplicitLeaf.hm96 enc hinj).leaf op.2.1 op.2.2.1.data.toList =
        (ExplicitLeaf.hm96 enc hinj).leaf op'.2.1 op'.2.2.1.data.toList then
      (ExplicitLeaf.hm96 enc hinj).pair op.2.1 op.2.2.1.data.toList op'.2.1 op'.2.2.1.data.toList
    else
      ((Flock.Registered.openPair op.1 i ((ExplicitLeaf.hm96 enc hinj).leaf op.2.1 op.2.2.1.data.toList).1 op.2.2.2
          ((ExplicitLeaf.hm96 enc hinj).leaf op'.2.1 op'.2.2.1.data.toList).1 op'.2.2.2).1.data.toList,
        (Flock.Registered.openPair op.1 i ((ExplicitLeaf.hm96 enc hinj).leaf op.2.1 op.2.2.1.data.toList).1 op.2.2.2
          ((ExplicitLeaf.hm96 enc hinj).leaf op'.2.1 op'.2.2.1.data.toList).1 op'.2.2.2).2.data.toList)

/-- **An opening of row `i` is the verifier's registered read**: `Flock.Registered.opens` of the row's commit string
`b ‖ c` (`Hm96.Default512.commitString` of the row's SHA-512 digest under the salt) at `i` through the path. -/
theorem regRowCommit_opens {Val : Type} (enc : Val → List UInt8) (hinj : Function.Injective enc) (i : ℕ)
    (reg : Flock.Registered.Port) (v : Val) (salt path : ByteArray) :
    (regRowCommit enc hinj i).Opens reg v (reg, v, salt, path) ↔
      Flock.Registered.opens reg (Hm96.Default512.commitString (Sha512.hash ⟨(enc v).toArray⟩) salt) i path = true := by
  show reg = reg ∧ v = v ∧ Flock.Registered.opensLeaf reg ((ExplicitLeaf.hm96 enc hinj).leaf v salt.data.toList).1 i
    path = true ↔ _
  rw [hm96_treeLeaf enc hinj v salt]
  simp only [Flock.Registered.opens, true_and]

/-- **The registration binds each row**, `RecursiveRanRegistered`'s `_hbind` for the executable's registered rows: two
openings of one port at `i` to different rows give a SHA-512 collision, hm96's (`ExplicitLeaf.hm96`) when their leaves
agree, else the registered path's (`Flock.Registered.opensLeaf_binding`). -/
theorem regRowCommit_binding {Val : Type} (enc : Val → List UInt8) (hinj : Function.Injective enc) (i : ℕ) :
    (regRowCommit enc hinj i).Binding := by
  rintro reg v v' ⟨r1, w1, y1, p1⟩ ⟨r2, w2, y2, p2⟩ ⟨h1r, h1v, h1⟩ ⟨h2r, h2v, h2⟩ hne
  simp only at h1r h1v h1 h2r h2v h2
  unfold regRowCommit
  simp only
  rw [h1r, h1v, h2v]
  split_ifs with hl
  · exact (ExplicitLeaf.hm96 enc hinj).pair_spec _ _ _ _ hne hl
  · exact collides_of_bytes (Flock.Registered.opensLeaf_binding reg i
      ((ExplicitLeaf.hm96 enc hinj).leaf v y1.data.toList).1 p1
      ((ExplicitLeaf.hm96 enc hinj).leaf v' y2.data.toList).1 p2 ((ExplicitLeaf.hm96 enc hinj).leaf v _).2
      ((ExplicitLeaf.hm96 enc hinj).leaf v' _).2 (fun e => hl (Subtype.ext e)) h1 h2)

end Rows

end Flock.SecurityProofs
