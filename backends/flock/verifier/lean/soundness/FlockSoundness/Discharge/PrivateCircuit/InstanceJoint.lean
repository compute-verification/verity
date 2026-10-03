import FlockSoundness.Discharge.PrivateCircuit.InstanceTree
import FlockSoundness.Discharge.PrivateCircuit.InstanceSound
import FlockSoundness.Discharge.PrivateCircuit.InstancePrivate

/-!
# Protocol 2 at the universal unit: soundness and privacy over one registration

`private_circuit`: the registrant draws salts `r` once and registers `pad C c` as its commit strings
`v = regView (enc c) (pad C c) r`, the view `private_circuit_private` hides. The registered root the verifier holds is
`regPort pins v`, a function of `v` and the port's public pins, and the registrant's opening of it (`viewOpening`) opens
it (`opens_view`); the session's layout reads the program's rows at that root, with the registrant's rows and salts
`pad C c`'s and `r`'s (`hreg`, `hsalt`), and so the soundness theorem's registered leaves are the view's (`leafH (v i)`).
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Game FlockSoundness.Discharge.ZkView FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-- **Protocol 2 at the universal unit, over one registration.** The program of class `c` is `L c` rows, `enc c p i` the
bytes of row `i`, the layout's row encoding of its values `row p i` (`henc`). The registrant draws salts `r` and registers
`pad C c` as its commit strings `v = regView (enc c) (pad C c) r`; the verifier holds the root `regPort pins v` of their
leaves, and the session's layout `lay` reads the program's `L c` rows there (`ps`), the registrant's rows and salts
there `pad C c`'s rows and `r`'s bytes (`hreg`, `hsalt`). Under `private_circuit_sound`'s hypotheses at that layout (the
partition the interpreter's, `UnitWrong`; protocol 1's soundness headline at the universal unit's session `s`, A3, A6,
`cr/sha-512` for `τ`'s knowledge finders, the registered check `hchk`, the step to the program `LeafBinds` and
`SHA512CRStrict` for the binding finder `hBind`), and `private_circuit_private`'s (`hash-derived-key` at `δ₁`, protocol 1's
zero knowledge at the tables of every registration of `pad C c`):

* the registrant's opening opens the registered root at every program position (`progCommit`);
* accepting with at least `K₀` wrong units of `C` at the committed values' claimed inputs and outputs, or with a program
  leaf other than the view's `leafH (v i)`, has probability at most `miss(K₀) + ε_ks + δ_link + q²/2^513`;
* the real view and the simulator's (which reads `c`, `o`, `p₀`, `w₀` only) are within
  `2·L c·δ₁ + ∑ⱼ 2·(nU + 2·nPU)·δ₁`, both ways. -/
theorem private_circuit
    {Out : Type} {K W Hi VL : Type} [Field K] [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi]
    (L : Universal.Cls → ℕ) (enc : (c : Universal.Cls) → c.Prog → Fin (L c) → List UInt8) (J : Universal.Cls → ℕ)
    (zs : (c : Universal.Cls) → Fin (J c) → Sizes)
    (U : (c : Universal.Cls) → Out → (Fin (L c) → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs c j))
    {UC : Universal.Cls → Audit.Circuit} (UP : (c : Universal.Cls) → Audit.Partition (UC c) (Universal.units c))
    (c : Universal.Cls) (C : Universal.Circ) (hC : Universal.fits C c)
    (progOf : (Fin (UC c).N → Bool) → c.Prog) (xOf : (Fin (UC c).N → Bool) → ℕ → ℕ → Bool)
    (yOf : (Fin (UC c).N → Bool) → ℕ → List Bool) (hunit : UnitWrong (UP c) progOf xOf yOf)
    (pins : Flock.Registered.Port) (r : Fin (L c) → Salt)
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (Universal.units c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (Universal.units c))) → (R : Reg) → Fin (Universal.units c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UC c) (Reg := Reg) plan tab}
    {lay : Binding.Layout (UP c) plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    (ps : ProgSites rd (regPort pins (regView (enc c) (Universal.pad C c) r)) (L c))
    (row : c.Prog → Fin (L c) → lay.Val) (henc : ∀ p i, enc c p i = rs.enc (row p i))
    (hreg : ∀ i, rd.wv (ps.site i) = row (Universal.pad C c) i)
    (hsalt : ∀ i, rd.ws (ps.site i) = (saltBytes (r i)).data.toList)
    (s : UnitSession (UP c) Reg) (hS : ZkSessionSound s)
    (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (path : s.Out → lay.Pos → ByteArray) (hchk : ∀ b, s.Accepts b → rd.Checked (s.regOf b) (path b))
    (τ : Strategy s.G) (hCR : s.CR τ) (valPair : s.Out → List UInt8 × List UInt8)
    (hstep : LeafBinds ps s τ progOf (Universal.pad C c) valPair)
    (K₀ : ℕ) (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ b, cost b ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ (fun b => some (progPair ps s path valPair b)) cost q)
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (o : Out) (w : (Fin (L c) → Salt) → Fin (J c) → W)
    (hView : ∀ r', ZkSessionView (U c o (regView (enc c) (Universal.pad C c) r')) (w r') δ₁)
    (p₀ : c.Prog) (w₀ : Fin (J c) → W) {Obs : Type} (encV : ∀ v, (∀ j, (U c o v j).tab.View) → Obs)
    (E : (Fin (L c) → Bits × ByteArray) → Obs → Prop) :
    (progCommit (L c)).Opens (regPort pins (regView (enc c) (Universal.pad C c) r))
        (viewDig (regView (enc c) (Universal.pad C c) r)) (viewOpening pins (regView (enc c) (Universal.pad C c) r)) ∧
      prob (fun b => s.Accepts b ∧
          (K₀ ≤ wrong (Universal.eval C) (Universal.size C) (xOf (s.vals τ b)) (yOf (s.vals τ b)) ∨
            ∃ i, (lay.regLeaf (s.regOf b) (ps.site i)).1 ≠ leafH (regView (enc c) (Universal.pad C c) r i))) s.G τ
        ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
      realPr (hm96Scheme Universal.fits Universal.pad L enc J zs U) c o (Universal.pad C c) w encV E
        ≤ simPr (hm96Scheme Universal.fits Universal.pad L enc J zs U) c o p₀ w₀ encV E
          + (2 * L c * δ₁ + viewGap (zs c) δ₁) ∧
      simPr (hm96Scheme Universal.fits Universal.pad L enc J zs U) c o p₀ w₀ encV E
        ≤ realPr (hm96Scheme Universal.fits Universal.pad L enc J zs U) c o (Universal.pad C c) w encV E
          + (2 * L c * δ₁ + viewGap (zs c) δ₁) := by
  have hS' := private_circuit_sound UP c C hC progOf xOf yOf hunit rd ps row hreg s hS hA3 hA6 path hchk τ hCR valPair
    hstep K₀ cost q hcost hBind
  refine ⟨opens_view pins _, le_trans (prob_mono (fun b ⟨ha, hq⟩ => ⟨ha, hq.imp id fun ⟨i, hi⟩ => ⟨i, fun he => hi ?_⟩⟩)
    s.G τ) hS', private_circuit_private Universal.fits Universal.pad L enc J zs U δ₁ hKey c o C w hView p₀ w₀ encV E⟩
  calc (lay.regLeaf (s.regOf b) (ps.site i)).1
      = ((ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (row (Universal.pad C c) i) (saltBytes (r i)).data.toList).1 := by
        rw [he, hsalt]
    _ = leafH (rowBits (rs.enc (row (Universal.pad C c) i)) + My (r i), saltHash (r i)) := hm96_leaf_view _ _ _ _
    _ = leafH (regView (enc c) (Universal.pad C c) r i) := by rw [regView, hm96Reg, henc]

end FlockSoundness.Discharge.PrivateCircuit
