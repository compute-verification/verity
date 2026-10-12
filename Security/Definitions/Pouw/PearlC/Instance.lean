import Definitions.Pouw.PearlC.Game
import Definitions.Pouw.PearlC.Skip
import Definitions.Pouw.PearlC.PeelExact
import Definitions.Pouw.PearlC.SaltDead

/-!
# Pearl-C's named protocol and error (trusted; the cheap-binding lane's file)

`pearlCProtocol sem` is the protocol of the named assumption. Its reference functions are the shared definitions:
* the noisy quantize (`FormChain.quantA`, `quantB`) and the promoted ticket chain (`ctChain` at `Params.pi.G`), from
  `Defs`;
* the useful output U under the pinned order, `PeelExact.peelExactVal` (`pearl_c_u.py` at `76134352`, op for op): C̃,
  `A′·F_B`, `B̃·F_A` and the Gram matrix as chain words, `P_A`, `P_B` with its FMA chain, and U by four K16 steps into C̃;
* the debit's flags, `Skip.unitFlags` (P1's `jointFlags` under G4, per output word, on the noisy codes), P3's
  `needed` (`Skip.needed`), and P3's salt-dead elements (`SaltDead.saltDead` at the rows' own `alpha` and `beta`).

Its domain rules are `Defs`' liveness rule (`LiveRow` under `Stat.every8`), finite words, and the noise floor. Its costs
are `creditOf` minus the replayed debit (`unitDebit`), and `wrefOf`.

`sem` collects the pieces that are not pinned yet, each to be replaced by a shared definition when it lands:
* `noise`: the unit's noise factors from the oracle, as E4M3 codes: E_A per unit, keyed by (salt, A root, B root, unit
  index), and F_A, F_B and E_B per job, keyed by (salt, B root);
* `fc`: the forming chain before the cast (§5's atom-mix);
* `alpha`, `beta`, `floor`: a row's scale, noise scale and noise-floor quantity (§5: α from the model's quantizer,
  σ = δ·ρ with ρ over every 8th position, floor σ·α).
-/

namespace Pouw.PearlC

open Finset Pouw.Fp8Atom Matrix

/-- What the named protocol takes that is not pinned yet (see the module docstring). -/
structure PearlCSem (Q R S : Type) where
  noise : (Q → R) → S → (U : Workload) → (u : ℕ) → Codes →
    Noise (U.layout.shape u).m (U.layout.shape u).n (U.layout.shape u).k Params.pi.r
  fc : FormChain Params.pi.r
  alpha : (k : ℕ) → (Fin k → ℚ) → ℚ
  beta : (k : ℕ) → (Fin k → ℚ) → ℚ
  floor : (k : ℕ) → (Fin k → ℚ) → ℚ

variable {Q R S : Type}

/-- A matrix of FP32 words read as values (a non-finite word reads as `0`; the domain rules reject it). -/
def wordsVal (p q : ℕ) (W : Codes) : Matrix (Fin p) (Fin q) ℚ := fun i l => (wordQ (W i l)).getD 0

/-- A shaped matrix read at natural indices, `none` outside it. -/
def atNat {p q : ℕ} (M : Matrix (Fin p) (Fin q) (Option ℚ)) : ℕ → ℕ → Option ℚ :=
  fun i j => if h : i < p ∧ j < q then M ⟨i, h.1⟩ ⟨j, h.2⟩ else none

/-- The BF16 word of a value BF16 represents exactly (an E4M3 code's value): the top half of its FP32 word. -/
def bf16WordOf (v : ℚ) : ℕ := f32Word v >>> 16

/-- **An E4M3 code widened to BF16** (`F2fpBf16 ∘ E4m3ToF32`, `pearl_c._bf16_of_code`): the code's sign bit over the
BF16 word of its magnitude, which BF16 holds exactly. So `−0` (`0x80`) widens to `0x8000`. -/
def codeBf16 (c : Code) : ℕ := bf16WordOf |codeVal c| ||| (if c.sign then 0x8000 else 0)

/-- A committed operand of `p` rows and `q` columns passes the domain rules: every word finite, every row live
(`LiveRow` at δ under §5's statistic `Stat.every8`, the ρ of the noise scale) and above the noise floor. Under the
full row's RMS the rule would admit every nonzero row (`liveRow_of_meanSq_pos`). -/
def PearlCSem.OperandOK (sem : PearlCSem Q R S) (p q : ℕ) (W : Codes) : Prop :=
  (∀ i < p, ∀ l < q, (wordQ (W i l)).isSome) ∧
    ∀ i : Fin p, LiveRow Params.pi.δ (wordsVal p q W i) Stat.every8 ∧ 1 ≤ sem.floor q (wordsVal p q W i)

/-- Row `i` of a committed operand of `p` rows and `q` columns passes Π's rules: `PearlCSem.OperandOK`, one row. -/
def PearlCSem.RowOK (sem : PearlCSem Q R S) (p q : ℕ) (W : Codes) (i : ℕ) : Prop :=
  (∀ l < q, (wordQ (W i l)).isSome) ∧ ∀ h : i < p,
    LiveRow Params.pi.δ (wordsVal p q W ⟨i, h⟩) Stat.every8 ∧ 1 ≤ sem.floor q (wordsVal p q W ⟨i, h⟩)

/-- The unit's noisy operands and ticket chain. -/
noncomputable def PearlCSem.unitChain (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) :
    Matrix (Fin (U.layout.shape u).m) (Fin (U.layout.shape u).k) Code ×
      Matrix (Fin (U.layout.shape u).n) (Fin (U.layout.shape u).k) Code :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let W := wordsVal sh.n sh.k (U.weight (U.layout.wid u))
  let z := sem.noise H s U u act
  (sem.fc.quantA (fun i => sem.alpha sh.k (X i)) (fun i => sem.beta sh.k (X i)) X z,
   sem.fc.quantB (fun j => sem.alpha sh.k (W j)) (fun j => sem.beta sh.k (W j)) W z)

/-- **The unit's useful output U under the pinned order** (`PeelExact.peelExactVal`), from the unit's noisy codes
(`unitChain`), the bases' codes `F_A` and `F_B`, and the peel's noise rows under §5's forming: `bf16(E′)` (`codeBf16`)
with `E′ = lineCodes β e` per row. -/
noncomputable def PearlCSem.unitUseful (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) :
    Matrix (Fin (U.layout.shape u).m) (Fin (U.layout.shape u).n) (Option ℚ) :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let W := wordsVal sh.n sh.k (U.weight (U.layout.wid u))
  let z := sem.noise H s U u act
  let AB := sem.unitChain H s U u act
  let be := fun (β : ℚ) (e : Fin Params.pi.r → Code) (p : Fin 32) => codeBf16 (lineCodes β e p)
  PeelExact.peelExactVal AB.1 AB.2 (fun l p => z.FA l p) (fun l p => z.FB l p)
    (fun i => be (sem.beta sh.k (X i)) (z.EA i)) (fun j => be (sem.beta sh.k (W j)) (z.EB j))

open Classical in
/-- **The unit's replayed debit**, computed by the reference from the committed activations, the weights and the realized
noise (the same `unitChain` the ticket reads):
- `atoms`: over every output word `(i, j)`, the atoms `t < k/32` that `Skip.unitFlags` flags on the word's noisy rows
  `A′ i` and `B̃ j` (P1's `jointFlags` under G4);
- `elems`: the A′ elements `(i, l)` that are salt-dead (`SaltDead.saltDead`: the code's value is the same at every flat
  noise pair), or not `Skip.needed` (every word `(i, j)` has the atom
  containing `l` flagged), each element once. -/
noncomputable def PearlCSem.unitDebit (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes) :
    Debit :=
  let sh := U.layout.shape u
  let X := wordsVal sh.m sh.k act
  let AB := sem.unitChain H s U u act
  let fl := Skip.unitFlags AB.1 AB.2
  ⟨∑ i : Fin sh.m, ∑ j : Fin sh.n, ((range (sh.k / 32)).filter fun t => fl i j t = true).card,
   (univ.filter fun e : Fin sh.m × Fin sh.k => SaltDead.saltDead sem.alpha sem.beta sh.k (X e.1) e.2 ∨ ¬ Skip.needed fl e.1 e.2).card⟩

/-- **Pearl-C's protocol:** the shared reference functions and domain rules, and Pearl-C's accounting at `c` with the
debit cap `ρ`: the credit is `creditOf` minus the unit's replayed debit, and a unit whose debit exceeds `ρ·creditOf` is
rejected. -/
noncomputable def pearlCProtocol (sem : PearlCSem Q R S) (c : Costs) (ρ : ℚ) : Protocol Q R S where
  ticket := fun H s U u act =>
    atNat (ctChain Params.pi.G (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2)
  useful := fun H s U u act => atNat (sem.unitUseful H s U u act)
  actOK := fun sh act => sem.OperandOK sh.m sh.k act
  weightOK := fun sh w j => sem.RowOK sh.n sh.k w j
  credit := fun H s U u act =>
    ((creditOf Params.pi c (U.layout.shape u) - debitOf Params.pi c (sem.unitDebit H s U u act) : ℚ) : ℝ)
  capOK := fun H s U u act =>
    debitOf Params.pi c (sem.unitDebit H s U u act) ≤ ρ * creditOf Params.pi c (U.layout.shape u)
  Wref := fun L u => (wrefOf Params.pi c (L.shape u) : ℝ)

/-- **The named error** `ε(q, N) = (q + N)·2^−128`. Each of the `q` oracle calls, preprocessing included, yields at
most one fresh noise draw, and each of the `N` declared units has its own E_A draw (keyed by the unit's index; F_A, F_B
and E_B are per job, shared by the job's units). The conjecture is that a draw on which some program undercuts that
unit's credit by more than `γ₀` has probability at most `2^−128`, the seeds' entropy; the union bound over the grinding
draws and the units gives `ε`, and it needs no independence between them. With `q, N ≤ 2^64`, `ε ≤ 2^−63`.

**The noise-loss event has probability 0**, for F's lines at norm 16 (Π's), for one-row units as well.
- A row the noise floor admits (`mul.rn(σ, α) ≥ 1`) can't lose its noise, whatever the draw.
- Pearl's line rule scales each E line to L2 norm 256 in BF16 and rounds each entry to E4M3, so a line's largest entry
  is at least its RMS. Over every line its largest entry is at least 44. This is exact: the roundings are monotone, so
  the minimum is taken over the largest magnitude `M = 1..128`, with all 32 magnitudes equal to `M`.
- At the floor, `β_row ≥ f32(√32/(256·16))`, so the largest `E′` entry is at least `e4m3(f32(β_row·44)) = 0.0625`, a
  normal code. `E′ = 0` is therefore impossible, and the per-row probability is `0 ≤ 2^−128`.
- Every draw also keeps at least 85% of its nominal `‖E′‖ = σα·√32/16`, so no draw can be recognized from `E′` alone as
  weak. The `2^−128` above is thus a conjecture about draws that all carry their noise.
- The check is `internal/pouw/cheap-binding/pearlc-noise-loss.py` in the research store. At norm 256 the largest entry
  is at least `2^−8`, also nonzero; an earlier `2^−90` for norm 256 wrongly treated the 32 entries as independent. -/
noncomputable def εPearlC (q N : ℕ) : ℝ := ((q + N : ℕ) : ℝ) / 2 ^ 128

end Pouw.PearlC
