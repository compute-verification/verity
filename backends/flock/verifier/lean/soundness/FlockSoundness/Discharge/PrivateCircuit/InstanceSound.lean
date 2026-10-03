import FlockSoundness.Discharge.PrivateCircuit.InstanceDefs
import FlockSoundness.Discharge.PrivateCircuit.Soundness
import FlockSoundness.Discharge.PrivateCircuit.InstanceCommit

/-!
# Protocol 2 is sound at the universal unit

`private_circuit_sound`: `ran_registered` at the program commitment of the registered root (`progCommit`), the
session's opening of it from the registered check (`Reads.Checked`), and protocol 1's headline at the universal unit
(`ZkSessionSound`) as `Q` and `B`. Accepting with at least `K₀` wrong units of the universal unit's partition, or with
program leaves other than the registered program `pad C c`'s, is at most `miss(K₀) + ε_ks + δ_link + q²/2^513`; and the
universal unit running `pad C c` computes `C` (`PadCorrect`), which is how the guarantee reads as one about `C`.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock FlockSoundness.Game FlockSoundness.Registered Refine Binding Audit
open scoped ENNReal

/-- **Protocol 2 is sound at the universal unit.** A circuit `C` fits the class `c`, and `pad C c` is registered: the
registrant's rows at the program's sites (`ps`, `L` registered reads of the root `rt` in the universal unit's layout
`lay` over its partition `UP c`, of `N c` units) are `pad C c`'s rows (`row`). The universal unit's session `s` meets
protocol 1's soundness headline (`ZkSessionSound`, under A3 and A6), and on every accepted outcome the registered check
holds at the outcome's registration (`hchk`, `Flock.Registered.check_ok` read at the model's leaves).

Then for every live prover `τ` whose knowledge finders meet `cr/sha-512` (`hCR`), accepting with at least `K₀` wrong
units of `UP c`, or with program leaves other than `pad C c`'s registered ones, has probability at most
`miss(K₀) + ε_ks + δ_link + q²/2^513`, the last under `SHA512CRStrict` (`cr/sha-512`) for the finder that plays the
session once and outputs the registered reads' extractor on the session's opening and the registrant's, at most `q`
SHA-512 evaluations on every outcome (`hBind`). `miss` is the law's over the padded `N c` units, dummies included. And
the universal unit running `pad C c` computes `C` (`PadCorrect`). -/
theorem private_circuit_sound
    {Cls Circ In Res : Type} {Prog : Cls → Type} (fits : Circ → Cls → Prop) (pad : Circ → (c : Cls) → Prog c)
    (run : (c : Cls) → Prog c → In → Res) (eval : Circ → In → Res) (hpad : PadCorrect fits pad run eval)
    {UC : Cls → Audit.Circuit} {N : Cls → ℕ} (UP : (c : Cls) → Audit.Partition (UC c) (N c))
    (c : Cls) (C : Circ) (hC : fits C c)
    {Reg : Type} {mPts : ℕ} {plan : Finset (Fin (N c)) → Reg → List (TabSpec mPts)}
    {tab : (S : Finset (Fin (N c))) → (R : Reg) → Fin (N c) → Fin (plan S R).length}
    {decode : Audit.Partition.DecoderB (C := UC c) (Reg := Reg) plan tab}
    {lay : Binding.Layout (UP c) plan tab decode} {rs : Binding.HmRows lay} (rd : Registered.Reads lay rs)
    {rt : Flock.Registered.Port} {L : ℕ} (ps : ProgSites rd rt L)
    (row : Prog c → Fin L → lay.Val) (hreg : ∀ i, rd.wv (ps.site i) = row (pad C c) i)
    (s : UnitSession (UP c) Reg) (hS : ZkSessionSound s)
    (hA3 : Assumptions.UniformRandomBytes s.Lb s.src) (hA6 : s.A6)
    (path : s.Out → lay.Pos → ByteArray) (hchk : ∀ o, s.Accepts o → rd.Checked (s.regOf o) (path o))
    (τ : Strategy s.G) (hCR : s.CR τ) (K₀ : ℕ) (cost : s.Out → ℝ) (q : ℝ) (hcost : ∀ o, cost o ≤ q)
    (hBind : Assumptions.SHA512CRStrict H512 s.G τ
      (fun o => some ((progCommit L).ext (ps.opening (s.regOf o) (path o)) ps.regOpening)) cost q) :
    prob (fun o => s.Accepts o ∧ (K₀ ≤ ((UP c).wrong (s.vals τ o)).card ∨
        ∃ i, lay.regLeaf (s.regOf o) (ps.site i) ≠
          (ExplicitLeaf.hm96 rs.enc rs.enc_inj).leaf (row (pad C c) i) (rd.ws (ps.site i)))) s.G τ
      ≤ s.Lm.miss K₀ + s.εks τ + s.δlink τ + ENNReal.ofReal (q ^ 2 / 2 ^ (513 : ℝ)) ∧
    ∀ x, run c (pad C c) x = eval C x := by
  refine ⟨?_, fun x => hpad C c hC x⟩
  have h := ran_registered (progCommit L) (progCommit_binding L) rt ps.regLeaves ps.regOpening ps.opens_reg s.G τ
    s.Accepts (fun o => K₀ ≤ ((UP c).wrong (s.vals τ o)).card) (fun o => ps.leaves (s.regOf o))
    (fun o => ps.opening (s.regOf o) (path o)) (fun o ho => ps.opens_of_checked (hchk o ho))
    _ (hS hA3 hA6 τ hCR K₀) cost q hcost hBind
  refine le_trans (prob_mono (fun o ⟨ha, hq⟩ => ⟨ha, hq.imp id fun ⟨i, hi⟩ hl => hi ?_⟩) s.G τ) h
  have hi' := congrFun hl i
  simp only [ProgSites.leaves, ProgSites.regLeaves, Reads.leaf, hreg] at hi'
  exact hi'

end FlockSoundness.Discharge.PrivateCircuit
