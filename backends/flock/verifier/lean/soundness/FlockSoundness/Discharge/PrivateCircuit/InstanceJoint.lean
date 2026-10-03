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
there `pad C c`'s rows and `r`'s bytes (`hreg`, `hsalt`). Under `private_circuit_sound`'s hypotheses at that layout (and
`PadCorrect`, protocol 1's soundness headline at the universal unit's session `s`, A3, A6, `cr/sha-512` for `τ`'s
knowledge finders, the registered check `hchk` and `SHA512CRStrict` for the binding finder `hBind`), and
`private_circuit_private`'s (`hash-derived-key` at `δ₁`, protocol 1's zero knowledge at the tables of every registration
of `pad C c`):

* the registrant's opening opens the registered root at every program position (`progCommit`);
* accepting with at least `K₀` wrong units of `UP c`, or with a program leaf other than the view's `leafH (v i)`, has
  probability at most `miss(K₀) + ε_ks + δ_link + q²/2^513`, and the universal unit running `pad C c` computes `C`;
* the real view and the simulator's (which reads `c`, `o`, `p₀`, `w₀` only) are within
  `2·L c·δ₁ + ∑ⱼ 2·(nU + 2·nPU)·δ₁`, both ways. -/
theorem private_circuit
    {Cls Circ In Res Out : Type} {Prog : Cls → Type} {K W Hi VL : Type} [Field K] [AddCommGroup Hi] [Fintype Hi]
    [DecidableEq Hi] (fits : Circ → Cls → Prop) (pad : Circ → (c : Cls) → Prog c)
    (run : (c : Cls) → Prog c → In → Res) (eval : Circ → In → Res) (hpad : PadCorrect fits pad run eval)
    (L : Cls → ℕ) (enc : (c : Cls) → Prog c → Fin (L c) → List UInt8) (J : Cls → ℕ)
    (zs : (c : Cls) → Fin (J c) → Sizes)
    (U : (c : Cls) → Out → (Fin (L c) → Bits × ByteArray) → ∀ j, Coined K W Hi VL (zs c j))
    {UC : Cls → Audit.Circuit} {N : Cls → ℕ} (UP : (c : Cls) → Audit.Partition (UC c) (N c))
    (c : Cls) (C : Circ) (hC : fits C c) (pins : Flock.Registered.Port) (r : Fin (L c) → Salt)
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (N c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (N c))) → (R : Reg) → Fin (N c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UC c) (Reg := Reg) plan tab}
    {lay : Binding.Layout (UP c) plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    (ps : ProgSites rd (regPort pins (regView (enc c) (pad C c) r)) (L c))
    (row : Prog c → Fin (L c) → lay.Val) (henc : ∀ p i, enc c p i = rs.enc (row p i))
    (hreg : ∀ i, rd.wv (ps.site i) = row (pad C c) i)
    (hsalt : ∀ i, rd.ws (ps.site i) = (saltBytes (r i)).data.toList)
    (s : UnitSession (UP c) Reg) (hS : ZkSessionSound s)
    (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (path : s.Out → lay.Pos → ByteArray) (hchk : ∀ b, s.Accepts b → rd.Checked (s.regOf b) (path b))
    (τ : Strategy s.G) (hCR : s.CR τ) (K₀ : ℕ) (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ b, cost b ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ
      (fun b => some ((progCommit (L c)).ext (ps.opening (s.regOf b) (path b)) ps.regOpening)) cost q)
    (δ₁ : ℝ≥0∞) (hKey : Assumptions.Hm96Hiding My saltHash δ₁)
    (o : Out) (w : (Fin (L c) → Salt) → Fin (J c) → W)
    (hView : ∀ r', ZkSessionView (U c o (regView (enc c) (pad C c) r')) (w r') δ₁)
    (p₀ : Prog c) (w₀ : Fin (J c) → W) {Obs : Type} (encV : ∀ v, (∀ j, (U c o v j).tab.View) → Obs)
    (E : (Fin (L c) → Bits × ByteArray) → Obs → Prop) :
    (progCommit (L c)).Opens (regPort pins (regView (enc c) (pad C c) r)) (viewDig (regView (enc c) (pad C c) r))
        (viewOpening pins (regView (enc c) (pad C c) r)) ∧
      prob (fun b => s.Accepts b ∧ (K₀ ≤ ((UP c).wrong (s.vals τ b)).card ∨
          ∃ i, (lay.regLeaf (s.regOf b) (ps.site i)).1 ≠ leafH (regView (enc c) (pad C c) r i))) s.G τ
        ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
      (∀ x, run c (pad C c) x = eval C x) ∧
      realPr (hm96Scheme fits pad L enc J zs U) c o (pad C c) w encV E
        ≤ simPr (hm96Scheme fits pad L enc J zs U) c o p₀ w₀ encV E + (2 * L c * δ₁ + viewGap (zs c) δ₁) ∧
      simPr (hm96Scheme fits pad L enc J zs U) c o p₀ w₀ encV E
        ≤ realPr (hm96Scheme fits pad L enc J zs U) c o (pad C c) w encV E + (2 * L c * δ₁ + viewGap (zs c) δ₁) := by
  sorry

end FlockSoundness.Discharge.PrivateCircuit
