import Proofs.Flock.Soundness.Discharge.ZkView.Statements
import Proofs.Flock.Soundness.Binding.Leaf

/-!
# Protocol 2, the private circuit: the objects

The developer's circuit is hidden as well as its values. It is padded to a public size class and becomes a program, a
value that the public universal unit `U cls` interprets. The program is registered under a hiding commitment, and every
session is a protocol-1 `--zk` session of `U cls` with the program among its hidden values.

* `Scheme`: the size classes `Cls` and the hidden circuits `Circ`, with `fits C cls` (the circuit fits the class);
  `pad C cls : Prog cls`, the program a circuit becomes at a class, every program of a class of one shape; the
  registration `reg cls p r`, the verifier's view of program `p` registered at the registrant's salts `r`; and the
  session's `--zk` tables at fixed coins, `U cls o v`, a function of the class, the outputs `o` owed to the verifier and
  the registration's view `v` only. That `U` reads nothing else is what "the universal unit is public" means.
* `PadCorrect`: `U cls` running `pad C cls` computes `C`, at every class `C` fits.
* `Sat`: a witness the honest prover runs on, every table's registered values and inner constraint holding
  (`ZkWireFixed`'s `hreg` and `hinner`). `ZkOk`: a table family meets `ZkWireFixed`'s conditions on the tables.
* `zkGap`: `ZkWireFixed`'s bound at the sizes `zs`, `∑ⱼ 2·|Hidⱼ|·hidingGap(My, saltHash)`.
* `Hides reg ε`: the registration hides the program: for any two programs of the class, any event of the registration's
  view has probabilities within `ε`, over the registrant's uniform salts.
* `realPr`, `simPr`: the probability of an event `E` of the verifier's view (the registration's view, and an encoding
  `enc` of the session's views, the proof files) at program `p` with the honest witness `w` (a function of the
  registrant's salts); and at the simulator's, which registers a dummy program `p₀` and runs `S_shvzk` on a dummy witness
  `w₀`, and so reads the class and the owed outputs only.
* `Commit`, `Commit.Binding`: a program commitment, by its opening relation and the extractor's two SHA-512 inputs from
  two openings; it binds when two openings of one root to two programs give an explicit SHA-512 collision.
-/

namespace FlockSoundness.Discharge.PrivateCircuit

open Flock ZK FlockSoundness.Discharge.ZkView
open FlockSoundness.Game (prCoin avg)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- **A private-circuit scheme**: the size classes `Cls`, the hidden circuits `Circ` and the class each fits (`fits`);
the programs of class `cls` (`Prog cls`), and `pad C cls`, the program circuit `C` becomes at class `cls`; the
registration of a program `p` at the registrant's salts `r`, as the verifier sees it (`reg cls p r : Rv cls`); and the
universal unit's `--zk` session at class `cls`: `J cls` tables of sizes `zs cls`, at fixed coins, given the outputs owed
`o : Out` and the registration's view `v` (`U cls o v`). -/
structure Scheme (Cls Circ Out : Type) (Prog Rr Rv : Cls → Type) (K W Hi VL : Type) [Field K] where
  fits : Circ → Cls → Prop
  pad : Circ → (c : Cls) → Prog c
  reg : (c : Cls) → Prog c → Rr c → Rv c
  J : Cls → ℕ
  zs : (c : Cls) → Fin (J c) → Sizes
  U : (c : Cls) → Out → Rv c → ∀ j, Coined K W Hi VL (zs c j)

/-- **Padding computes the circuit**: at every class `c` that `C` fits, `U c` running `pad C c` (`run c`) computes what
`C` computes (`eval C`), on every input. -/
def PadCorrect {Cls Circ In Out : Type} {Prog : Cls → Type} (fits : Circ → Cls → Prop) (pad : Circ → (c : Cls) → Prog c)
    (run : (c : Cls) → Prog c → In → Out) (eval : Circ → In → Out) : Prop :=
  ∀ C c, fits C c → ∀ x, run c (pad C c) x = eval C x

section Tables

variable {K W Hi VL : Type} [Field K] {J : ℕ} {zs : Fin J → Sizes}

/-- **The honest prover's witness** `w` for the tables `T`: every table's registered values are the public ones and its
inner constraint holds (`ZkWireFixed`'s `hreg` and `hinner`). -/
def Sat (T : ∀ j, Coined K W Hi VL (zs j)) (w : Fin J → W) : Prop :=
  ∀ j, (T j).tab.sReg (w j) = (T j).tab.pubReg ∧ (T j).tab.InnerHolds (w j)

/-- **The tables meet `ZkWireFixed`'s conditions**: rank, the level-0 padding and the pads code onto, the extra lanes'
map bijective, and `β ≠ 0`, at every table. -/
def ZkOk (T : ∀ j, Coined K W Hi VL (zs j)) : Prop :=
  ∀ j, (T j).tab.Rank ∧ (T j).tab.PadOnto ∧ (∀ r, Function.Bijective fun p : 𝔽 × 𝔽 => (T j).tab.Wr r p.1 p.2) ∧
    (T j).tab.PadsOnto ∧ ∀ r, (T j).tab.β r ≠ 0

end Tables

/-- **`ZkWireFixed`'s bound at the sizes `zs`**: `∑ⱼ 2·|Hidⱼ|·hidingGap(My, saltHash)`, with `|Hidⱼ| = nU + 2·nPU` the
leaves that table `j`'s final message doesn't open. -/
noncomputable def zkGap {J : ℕ} (zs : Fin J → Sizes) : ℝ≥0∞ :=
  ∑ j, 2 * (((zs j).nU + 2 * (zs j).nPU : ℕ) : ℝ≥0∞) * ENNReal.ofReal (CROnly.hidingGap My saltHash)

/-- **The registration hides the program** up to `ε`: for any two programs `p`, `p'` and any event `E` of the
registration's view, over the registrant's uniform salts, `E` at `p` is at most `E` at `p'` plus `ε`. -/
def Hides {P R V : Type} [Fintype R] (reg : P → R → V) (ε : ℝ≥0∞) : Prop :=
  ∀ (p p' : P) (E : V → Prop), prCoin (fun r => E (reg p r)) ≤ prCoin (fun r => E (reg p' r)) + ε

section Views

variable {Cls Circ Out : Type} {Prog Rr Rv : Cls → Type} [∀ c, Fintype (Rr c)] {K W Hi VL : Type} [Field K]
  [AddCommGroup Hi] [Fintype Hi] [DecidableEq Hi] {Obs : Type}

/-- **The verifier's view, real**: the probability that the registration's view `v = reg c p r` and the encoding `enc v`
of the session's views satisfy `E`, over the registrant's uniform salts `r` and then the session's randomness and hidden
leaves' salts, for program `p` at class `c`, owed outputs `o`, and the honest witness `w r` (every hidden leaf the
executable's, `Coined.viewExe`). -/
noncomputable def realPr (S : Scheme Cls Circ Out Prog Rr Rv K W Hi VL) (c : Cls) (o : Out) (p : Prog c)
    (w : Rr c → Fin (S.J c) → W) (enc : ∀ v, (∀ j, (S.U c o v j).tab.View) → Obs) (E : Rv c → Obs → Prop) : ℝ≥0∞ :=
  avg fun r : Rr c =>
    prCoin fun q : ∀ j, (S.U c o (S.reg c p r) j).tab.Rand × ((S.U c o (S.reg c p r) j).tab.Hid → Salt) =>
      E (S.reg c p r) (enc (S.reg c p r) fun j => (S.U c o (S.reg c p r) j).viewExe (w r j) (q j).1 (q j).2)

/-- **The simulator's view**: it registers the dummy program `p₀` at uniform salts, then runs `S_shvzk` with the dummy
witness `w₀` on the universal unit's tables at the class, the owed outputs and that registration (`Coined.simExe`). It
reads the class `c` and the owed outputs `o`, never the circuit. -/
noncomputable def simPr (S : Scheme Cls Circ Out Prog Rr Rv K W Hi VL) (c : Cls) (o : Out) (p₀ : Prog c)
    (w₀ : Fin (S.J c) → W) (enc : ∀ v, (∀ j, (S.U c o v j).tab.View) → Obs) (E : Rv c → Obs → Prop) : ℝ≥0∞ :=
  avg fun r : Rr c =>
    prCoin fun q : ∀ j, (S.U c o (S.reg c p₀ r) j).tab.SimRand × ((S.U c o (S.reg c p₀ r) j).tab.Hid → Salt) =>
      E (S.reg c p₀ r) (enc (S.reg c p₀ r) fun j => (S.U c o (S.reg c p₀ r) j).simExe (w₀ j) (q j).1 (q j).2)

end Views

/-- **A program commitment**: `Opens rt p op`, the opening `op` shows that the root `rt` commits to the program `p`; and
`ext op op'`, the extractor's two SHA-512 inputs from two openings. -/
structure Commit (Root P Op : Type) where
  Opens : Root → P → Op → Prop
  ext : Op → Op → List UInt8 × List UInt8

/-- **The commitment binds under SHA-512**: two openings of one root to two different programs give an explicit
collision of SHA-512 (`Binding.Collides`), the extractor's pair. -/
def Commit.Binding {Root P Op : Type} (com : Commit Root P Op) : Prop :=
  ∀ rt p p' op op', com.Opens rt p op → com.Opens rt p' op' → p ≠ p' → Binding.Collides (com.ext op op')

end FlockSoundness.Discharge.PrivateCircuit
