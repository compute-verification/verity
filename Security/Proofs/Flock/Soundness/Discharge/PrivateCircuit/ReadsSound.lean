import Proofs.Flock.Soundness.Discharge.PrivateCircuit.ReadsBind
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.ReadsCompose
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Soundness
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.Privacy
import Proofs.Flock.Soundness.Discharge.PrivateCircuit.InstanceHides

/-!
# Protocol 2 with hidden reads, end to end

`U cls` is the class circuit (a unit circuit plus its read gadget, `ClassRel`), and the program is the topology table,
each unit's entry its dimensions and the store cells it reads. The registrant registers the table under the root `rtT₀`
and the row store under `rtS₀`, and holds an opening of every index of each (`tr`, `wo`). The registered circuit is the
table `T₀ j = (tr j).val` and the store `w c = (wo (idx c)).val`; its registered cells `w (.reg m)` are the committed
public and secret inputs.

* `private_circuit_sound_reads`, soundness at the drawn units, for every live prover of a session whose registration
  carries the registrant's roots:
  1. the delivered values are the committed ones: a delivered public input other than the store's registered cell, or a
     delivered output other than the store's output row, is a read off the store, at most `q_S²/2^513`;
  2. if the registered circuit's owed outputs are wrong (`OwedWrongR`), accepting is at most the law's escape at the units
     wrong against the store (`bad T₀ w`, fixed before the draw) plus `ε_ks + δ_link` plus `q_S²/2^513` plus `q_T²/2^513`:
     a drawn unit wrong against the store is wrong in the committed values, or its gadget's reads are not the
     registered ones (a collision with the registrant's opening: the store's finder, `readPick`; the table's, through
     `ran_registered` at `regCommit`);
  3. together: accepting with a delivered public input other than the committed cell, or a delivered output other than
     the registered circuit's end to end on the committed cells, is at most `q_S²/2^513` when the registered circuit's
     owed outputs are right, and part 2's bound otherwise.
  The inputs condition is explicit: the circuit runs on the committed cells, and the delivered public inputs are bound
  to them (`pubIn`); every other registered cell (the secret inputs) is the registrant's.
* `private_circuit_private_reads`, `private_circuit_indist_reads`: privacy at a public class list `cl` (`ClassList`:
  template, dimension class and padded count per block). The program commitment is `readsScheme`, the topology table's
  `cl.units` entries registered as hm96 rows. The verifier's view at the registered table is within
  `2·cl.units·δ₁ + zkGap` of the simulator's, which reads the class list `cl` and the owed outputs `o` only; so two
  circuits of one class list, for the same owed outputs, give every event of the view probabilities within twice that.
  What is public is exactly the class list and the owed outputs (with the session's public parameters, the class
  circuit `U cl`): the view's distribution is a function of them, nothing more. The class list is not hidden: it can
  tell apart circuits whose class lists differ, and within one class list only the committed contents of the table
  (dimensions within a class, read positions) are hidden. At hidden unit circuits the class list is size-only
  (`ReadsHidden.lean`).

Protocol 1's drawn bound (`ZkSessionDrawn`, `zk_session_drawnR`'s form) is a hypothesis, as `ZkSessionSound` was. The
class circuit enters as a fact about its net (`ClassComputes`: each unit's check gate computes `ClassRel`), not as its
wrong units.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit Reads
open scoped ENNReal

section Sound

variable {V : Type} [Inhabited V] {P : Pop V} {C : Audit.Circuit} {Pt : Audit.Partition C P.N} {Reg : Type} {M : ℕ}

open Classical in
/-- **The store reads an outcome makes**, at index `i` with opening `x`: an input read or an output write of a drawn unit
that is not wrong in the committed values, at the index of the cell its entry read names or of its output row; a
delivered output, at its output row's index; or a delivered public input, at its cell's index. -/
def storeReads (L : ReadLayout P C.N) (idx : Cell → Fin M) (s : UnitSession Pt Reg) (τ : Strategy s.G)
    (drawn : s.Out → Finset (Fin P.N)) (owed : Finset (Fin P.N × Fin P.o)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin P.N × Fin P.o → ReadVal V) (xD : s.Out → ℕ → ReadVal V) (o : s.Out) (i : Fin M)
    (x : ReadVal V) : Prop :=
  (∃ u ∈ drawn o, u ∉ Pt.wrong (s.vals τ o) ∧
      ((∃ k, idx ((L.tabOf (s.vals τ o) u).src k) = i ∧ x = (L.inp u k).read (s.vals τ o)) ∨
        ∃ l : Fin P.o, idx (.out u l) = i ∧ x = (L.out u l).read (s.vals τ o))) ∨
    (∃ p ∈ owed, idx (.out p.1 p.2) = i ∧ x = yD o p) ∨ ∃ m ∈ pubIn, idx (.reg m) = i ∧ x = xD o m

open Classical in
/-- **The table opening an outcome shows**: at a drawn unit not wrong in the committed values, its gadget's entry read;
at every other unit, the registrant's opening. -/
noncomputable def tabFill (L : ReadLayout P C.N) (s : UnitSession Pt Reg) (τ : Strategy s.G)
    (drawn : s.Out → Finset (Fin P.N)) (tr : Fin P.N → ReadVal (Entry P)) (o : s.Out) (j : Fin P.N) :
    ReadVal (Entry P) :=
  if j ∈ drawn o ∧ j ∉ Pt.wrong (s.vals τ o) then (L.ent j).read (s.vals τ o) else tr j

open Classical in
/-- **Protocol 2 with hidden reads is sound at the drawn units.** The class net computes its relation at each unit's check
gate `ok u` (`hcomp`, at the good assignments, which the committed values of an accepted outcome are, `hgood`), so its
wrong units are the relation's (`classWrongOn_of_computes`); the registrant's
openings of the table and the store check against its roots (`htr`, `hwo`); the session meets protocol 1's drawn bound
(`hD`) and escapes sets fixed before the draw with at most the law's escape (`hE`); the verifier checks the delivered
outputs and public inputs against the registration's store root (`hdel`); and the store's and the table's finders meet
`cr/sha-512` at budgets `qS`, `qT` (`hS`, `hT`). Then, at sessions whose registration carries the registrant's roots, with
`T₀`, `w` the registered table and store: (1) a delivered value off the committed one is at most `q_S²/2^513`; (2) if the
registered circuit's owed outputs are wrong, accepting is at most `escape(bad T₀ w) + ε_ks + δ_link + q_S²/2^513 +
q_T²/2^513`; (3) a delivered public input off the committed cell or a delivered output off the registered circuit's end
to end on the committed cells is at most `q_S²/2^513` plus, when the registered circuit's owed outputs are wrong,
`escape(bad T₀ w) + ε_ks + δ_link + q_T²/2^513`. -/
theorem private_circuit_sound_reads
    (L : ReadLayout P C.N) (lfT : ExplicitLeaf (Entry P)) (lfS : ExplicitLeaf V) (idx : Cell → Fin M)
    (rootT rootS : Reg → Flock.Registered.Port) (good : Reg → (Fin C.N → Bool) → Prop)
    (ok : Fin P.N → Fin C.N)
    (hcomp : ClassComputes Pt good (fun R v u => ClassRel L lfT lfS idx (rootT R) (rootS R) v u) ok)
    (rtT₀ rtS₀ : Flock.Registered.Port) (tr : Fin P.N → ReadVal (Entry P)) (wo : Fin M → ReadVal V)
    (htr : ∀ j : Fin P.N, ReadOk lfT rtT₀ j (tr j)) (hwo : ∀ i : Fin M, ReadOk lfS rtS₀ i (wo i))
    (s : UnitSession Pt Reg) (drawn : s.Out → Finset (Fin P.N)) (hD : ZkSessionDrawn s drawn)
    (hE : DrawEscapes s drawn) (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (τ : Strategy s.G) (hCR : s.CR τ) (hgood : ∀ o, s.Accepts o → good (s.regOf o) (s.vals τ o))
    (owed : Finset (Fin P.N × Fin P.o)) (pubIn : Finset ℕ)
    (yD : s.Out → Fin P.N × Fin P.o → ReadVal V) (xD : s.Out → ℕ → ReadVal V)
    (hdel : ∀ o, s.Accepts o →
      (∀ p ∈ owed, ReadOk lfS (rootS (s.regOf o)) (idx (.out p.1 p.2)) (yD o p)) ∧
        ∀ m ∈ pubIn, ReadOk lfS (rootS (s.regOf o)) (idx (.reg m)) (xD o m))
    (costS costT : s.Out → ℝ) (qS qT : ℝ) (hcS : ∀ o, costS o ≤ qS) (hcT : ∀ o, costT o ≤ qT)
    (hS : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some (readPick lfS rtS₀ Fin.val wo (storeReads L idx s τ drawn owed pubIn yD xD) o)) costS qS)
    (hT : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some ((regCommit lfT P.N).ext (rtT₀, tabFill L s τ drawn tr o) (rtT₀, tr))) costT qT) :
    let w : Cell → V := fun c => (wo (idx c)).val
    let T₀ : Table P := fun j => (tr j).val
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) ∧
    (OwedWrongR P T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad P T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) +
          ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ))) ∧
    prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
        ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨
          ∃ p ∈ owed, (yD o p).val ≠ evalR P T₀ (fun m => w (.reg m)) p.1 p.2)) s.G τ
      ≤ (if OwedWrongR P T₀ w owed then
          s.Lm.escape (bad P T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) else 0) +
        ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
  intro w T₀
  classical
  have hwr := classWrongOn_of_computes hcomp
  have hRB := reads_bind_le s.G τ lfS rtS₀ Fin.val wo hwo (storeReads L idx s τ drawn owed pubIn yD xD) costS qS hcS hS
  have hrel : ∀ o, s.Accepts o → ∀ u, u ∉ Pt.wrong (s.vals τ o) →
      ClassRel L lfT lfS idx (rootT (s.regOf o)) (rootS (s.regOf o)) (s.vals τ o) u := fun o ha u hu => by
    by_contra hr
    exact hu ((hwr _ _ (hgood o ha) u).2 hr)
  have h1 : prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀ ∧
      ((∃ m ∈ pubIn, (xD o m).val ≠ w (.reg m)) ∨ ∃ p ∈ owed, (yD o p).val ≠ w (.out p.1 p.2))) s.G τ
      ≤ ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
    refine le_trans (prob_mono (fun o ⟨ha, _, hS', hq⟩ => ?_) _ _) hRB
    have hd := hdel o ha
    rw [hS'] at hd
    rcases hq with ⟨m, hm, hne⟩ | ⟨p, hp, hne⟩
    · exact ⟨idx (.reg m), xD o m, Or.inr (Or.inr ⟨m, hm, rfl, rfl⟩), hd.2 m hm, hne⟩
    · exact ⟨idx (.out p.1 p.2), yD o p, Or.inr (Or.inl ⟨p, hp, rfl, rfl⟩), hd.1 p hp, hne⟩
  have h2 : OwedWrongR P T₀ w owed →
      prob (fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀) s.G τ
        ≤ s.Lm.escape (bad P T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) +
          ENNReal.ofReal (qT ^ 2 / 2 ^ (513 : ℝ)) := by
    intro _
    let acc : s.Out → Prop := fun o => s.Accepts o ∧ rootT (s.regOf o) = rtT₀ ∧ rootS (s.regOf o) = rtS₀
    let Q : s.Out → Prop := fun o => (∃ u ∈ drawn o, u ∈ Pt.wrong (s.vals τ o)) ∨
      (∃ i x, storeReads L idx s τ drawn owed pubIn yD xD o i x ∧ ReadOk lfS rtS₀ (↑i) x ∧ x.val ≠ (wo i).val) ∨
        Disjoint (drawn o) (bad P T₀ w)
    have hQ : prob (fun o => acc o ∧ Q o) s.G τ
        ≤ s.Lm.escape (bad P T₀ w) + (s.εks τ + s.δlink τ) + ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) := by
      calc _ ≤ prob (fun o => (s.Accepts o ∧ ∃ u ∈ drawn o, u ∈ Pt.wrong (s.vals τ o)) ∨
            ((∃ i x, storeReads L idx s τ drawn owed pubIn yD xD o i x ∧ ReadOk lfS rtS₀ (↑i) x ∧
              x.val ≠ (wo i).val) ∨
              (s.Accepts o ∧ Disjoint (drawn o) ((fun _ : Reg => bad P T₀ w) (s.regOf o))))) s.G τ := by
            refine prob_mono (fun o ⟨⟨ha, _⟩, hq⟩ => ?_) _ _
            rcases hq with h | h | h
            · exact Or.inl ⟨ha, h⟩
            · exact Or.inr (Or.inl h)
            · exact Or.inr (Or.inr ⟨ha, h⟩)
        _ ≤ _ := prob_or_le _ _ _ _
        _ ≤ _ := add_le_add le_rfl (prob_or_le _ _ _ _)
        _ ≤ (s.εks τ + s.δlink τ) + (ENNReal.ofReal (qS ^ 2 / 2 ^ (513 : ℝ)) + s.Lm.escape (bad P T₀ w)) :=
            add_le_add (hD hA3 hA6 τ hCR) (add_le_add hRB ((hE τ _).trans (iSup_le fun _ => le_rfl)))
        _ = _ := by ring
    have hOpens : ∀ o, acc o → (regCommit lfT P.N).Opens rtT₀ (fun j => (tabFill L s τ drawn tr o j).val)
        (rtT₀, tabFill L s τ drawn tr o) := by
      rintro o ⟨ha, hT', -⟩
      refine show _ ∧ _ from ⟨rfl, fun j => ⟨rfl, ?_⟩⟩
      by_cases hj : j ∈ drawn o ∧ j ∉ Pt.wrong (s.vals τ o)
      · have := (hrel o ha j hj.2).1
        rw [hT'] at this
        simpa only [tabFill, ite_eq_left hj] using this
      · simpa only [tabFill, ite_eq_right hj] using htr j
    have hRR := ran_registered (regCommit lfT P.N) (regCommit_binding lfT P.N) rtT₀ T₀ (rtT₀, tr)
      (show _ ∧ _ from ⟨rfl, fun j => ⟨rfl, htr j⟩⟩) s.G τ acc Q
      (fun o j => (tabFill L s τ drawn tr o j).val) (fun o => (rtT₀, tabFill L s τ drawn tr o)) hOpens _ hQ
      costT qT hcT hT
    refine le_trans (prob_mono (fun o ha => ⟨ha, ?_⟩) _ _) hRR
    obtain ⟨ha, hT', hS'⟩ := ha
    by_cases hdis : Disjoint (drawn o) (bad P T₀ w)
    · exact Or.inl (Or.inr (Or.inr hdis))
    obtain ⟨u, hu, hbad⟩ := Finset.not_disjoint_iff.1 hdis
    by_cases hwu : u ∈ Pt.wrong (s.vals τ o)
    · exact Or.inl (Or.inl ⟨u, hu, hwu⟩)
    have hr := hrel o ha u hwu
    by_cases hT : L.tabOf (s.vals τ o) u = T₀ u
    · rcases classRel_bad L lfT lfS idx _ _ _ u hr T₀ w hT hbad with ⟨k, hk⟩ | ⟨l, hl⟩
      · refine Or.inl (Or.inr (Or.inl ⟨idx ((L.tabOf (s.vals τ o) u).src k), (L.inp u k).read (s.vals τ o),
          Or.inl ⟨u, hu, hwu, Or.inl ⟨k, rfl, rfl⟩⟩, ?_, ?_⟩))
        · have := hr.2.2.1 k
          rw [hS'] at this
          exact this
        · show L.xOf (s.vals τ o) u k ≠ (wo (idx ((L.tabOf (s.vals τ o) u).src k))).val
          rw [hT]
          exact hk
      · refine Or.inl (Or.inr (Or.inl ⟨idx (.out u l), (L.out u l).read (s.vals τ o),
          Or.inl ⟨u, hu, hwu, Or.inr ⟨l, rfl, rfl⟩⟩, ?_, hl⟩))
        have := hr.2.2.2.1 l
        rw [hS'] at this
        exact this
    · refine Or.inr fun heq => hT ?_
      have := congrFun heq u
      unfold tabFill at this
      rw [ite_eq_left ⟨hu, hwu⟩] at this
      exact this
  refine ⟨h1, h2, ?_⟩
  by_cases hOW : OwedWrongR P T₀ w owed
  · rw [ite_eq_left hOW]
    refine le_trans (prob_mono (fun o ⟨ha, hT', hS', _⟩ => ⟨ha, hT', hS'⟩) _ _) ((h2 hOW).trans (le_of_eq ?_))
    ring
  · rw [ite_eq_right hOW, zero_add]
    refine le_trans (prob_mono (fun o ⟨ha, hT', hS', hq⟩ => ⟨ha, hT', hS', ?_⟩) _ _) h1
    rcases hq with h | ⟨p, hp, hne⟩
    · exact Or.inl h
    · refine Or.inr ⟨p, hp, fun heq => hne (heq.trans ?_)⟩
      by_contra hne'
      exact hOW ⟨p, hp, hne'⟩

end Sound

/-- **The program commitment at a class list**: `hm96Scheme` with the class list as the class and the topology table of
`cl.units` entries as the program, each entry registered as its hm96 commit string (`encE`). -/
abbrev readsScheme {Circ Out Ent : Type} {K W Hi VL : Type} [Field K] (fits : Circ → ClassList → Prop)
    (tab : Circ → (cl : ClassList) → Fin cl.units → Ent) (encE : Ent → List UInt8) (J : ClassList → ℕ)
    (zs : (cl : ClassList) → Fin (J cl) → Sizes)
    (U : (cl : ClassList) → Out → (Fin cl.units → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs cl j)) :=
  hm96Scheme fits tab ClassList.units (fun _ T j => encE (T j)) J zs U

/-- **Protocol 2 with hidden reads is private at a public class list.** The public view is the class list `cl`
(template, dimension class and padded count per block) and the owed outputs `o`; the program is the topology table of
`cl.units` entries (`tab C cl`), each registered as its hm96 commit string (`encE`) under the registrant's uniform salts
(`readsScheme`), and the class circuit's `--zk` tables `U cl o v` read the class list, the owed outputs and that
registration `v` only. Under `hash-derived-key` at `δ₁` (`hKey`), where the tables at every registration of `tab C cl`
meet `ZkWireFixed`'s conditions (`hZk`) and the honest witness satisfies them (`hSat`): for every dummy table `p₀` and
dummy witness `w₀`, every encoding of the session's views and every event of the registration's view and that encoding,
the real view and the simulator's, which reads `cl` and `o` and nothing of `C`, are within `2·cl.units·δ₁ +
zkGap (zs cl)`, both ways. The class list is public; within it the table's contents are hidden.
`circuit_private` at `readsScheme`. -/
theorem private_circuit_private_reads
    {Circ Out Ent : Type} {K W Hi VL : Type} [Field K] [Fintype K] [DecidableEq K]
    [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (fits : Circ → ClassList → Prop)
    (tab : Circ → (cl : ClassList) → Fin cl.units → Ent) (encE : Ent → List UInt8) (J : ClassList → ℕ)
    (zs : (cl : ClassList) → Fin (J cl) → Sizes)
    (U : (cl : ClassList) → Out → (Fin cl.units → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs cl j))
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (cl : ClassList) (o : Out) (C : Circ) (w : (Fin cl.units → Salt) → Fin (J cl) → W)
    (hZk : ∀ r, ZkOk (U cl o (hm96Reg (fun T j => rowBits (encE (T j))) (tab C cl) r)))
    (hSat : ∀ r, Sat (U cl o (hm96Reg (fun T j => rowBits (encE (T j))) (tab C cl) r)) (w r))
    (p₀ : Fin cl.units → Ent) (w₀ : Fin (J cl) → W) {Obs : Type}
    (encV : ∀ v, (∀ j, (U cl o v j).tab.View) → Obs) (E : (Fin cl.units → Bits × ByteArray) → Obs → Prop) :
    realPr (readsScheme fits tab encE J zs U) cl o (tab C cl) w encV E
        ≤ simPr (readsScheme fits tab encE J zs U) cl o p₀ w₀ encV E + (2 * cl.units * δ₁ + zkGap (zs cl)) ∧
      simPr (readsScheme fits tab encE J zs U) cl o p₀ w₀ encV E
        ≤ realPr (readsScheme fits tab encE J zs U) cl o (tab C cl) w encV E + (2 * cl.units * δ₁ + zkGap (zs cl)) :=
  circuit_private (readsScheme fits tab encE J zs U) cl (2 * cl.units * δ₁) (hm96Reg_hides _ hKey) o C w hZk hSat p₀ w₀
    encV E

/-- **The view is a function of the public class list and the owed outputs**: any two circuits `C`, `C'` registered at
one class list `cl` (template, dimension class and padded count per block), with honest witnesses for the same owed
outputs `o`, give every event of the verifier's view probabilities within `2·(2·cl.units·δ₁ + zkGap (zs cl))`, under
`hash-derived-key` at `δ₁`. Nothing is claimed for circuits of different class lists: the class list is public.
`circuit_indist` at `readsScheme`. -/
theorem private_circuit_indist_reads
    {Circ Out Ent : Type} {K W Hi VL : Type} [Field K] [Fintype K] [DecidableEq K]
    [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] (fits : Circ → ClassList → Prop)
    (tab : Circ → (cl : ClassList) → Fin cl.units → Ent) (encE : Ent → List UInt8) (J : ClassList → ℕ)
    (zs : (cl : ClassList) → Fin (J cl) → Sizes)
    (U : (cl : ClassList) → Out → (Fin cl.units → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs cl j))
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (cl : ClassList) (o : Out) (C C' : Circ) (w w' : (Fin cl.units → Salt) → Fin (J cl) → W)
    (hZk : ∀ r, ZkOk (U cl o (hm96Reg (fun T j => rowBits (encE (T j))) (tab C cl) r)))
    (hZk' : ∀ r, ZkOk (U cl o (hm96Reg (fun T j => rowBits (encE (T j))) (tab C' cl) r)))
    (hSat : ∀ r, Sat (U cl o (hm96Reg (fun T j => rowBits (encE (T j))) (tab C cl) r)) (w r))
    (hSat' : ∀ r, Sat (U cl o (hm96Reg (fun T j => rowBits (encE (T j))) (tab C' cl) r)) (w' r)) {Obs : Type}
    (encV : ∀ v, (∀ j, (U cl o v j).tab.View) → Obs) (E : (Fin cl.units → Bits × ByteArray) → Obs → Prop) :
    realPr (readsScheme fits tab encE J zs U) cl o (tab C cl) w encV E
      ≤ realPr (readsScheme fits tab encE J zs U) cl o (tab C' cl) w' encV E +
        2 * (2 * cl.units * δ₁ + zkGap (zs cl)) :=
  circuit_indist (readsScheme fits tab encE J zs U) cl (2 * cl.units * δ₁) (hm96Reg_hides _ hKey) o C C' w w' hZk hZk'
    hSat hSat' encV E

end FlockSoundness.Discharge.PrivateCircuit
