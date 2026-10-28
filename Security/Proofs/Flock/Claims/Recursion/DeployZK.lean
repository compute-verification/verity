import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.DeployOC

/-!
# What the recursion's zero knowledge rests on at `deployRec`

`Claims.Recursion.ZK` at the masked record `deployRecZ IV ivScope b tapeOf SO` (`deployRec` with its reference prover's
tape at each statement drawn from a finite nonce and rest by `tapeOf`, and the outer simulators `SO`) and the outer
record `deployZkRec relO` is `DO.zk = true`, which holds (`deployZkRec` is the masked record), and the bound `ZKBound`: at
each attempt budget `t ≥ 1` and wherever hm96 leaves hide up to `δ₁` at the pinned key, `ZKmalRx` at `Sim` within
`εzk + γzk · Pr[Fork]` (`zk_deployRec`, `zk_deployRec_iff`). `ZKBound` is the hybrid over the three parts of a run, and
this file proves what `deployRec` fixes of each:

* **The inner session**: `Sim` sends the commit strings to the zero digest under its salts (`Deploy.zeroCommit`, the
  pinned key's, `zeroCommit_sha512`). The hybrid needs the prover's inner messages to be hm96 commit strings under
  fresh salts of its tape, each then within `2δ₁` (`Hm96Hiding`); that is the inner prover's (`RefParts.PI`), which
  `deployRec` leaves a parameter.
* **The staging round**: the reference prover sends the record's `stage` at the commit strings of its staged values'
  digests under the staging salts on its tape (`refP_stage`), and `Sim` the same `stage` at commit strings to the zero
  digest under its salts; at the zero digest the prover's commit strings are `Sim`'s (`commitThen_zero`). So the two
  differ only in the digests the commit strings hide, `2δ₁` per commit string (`Hm96Hiding`).
* **The V\* sessions**: `Sim` runs the outer simulator `SO t` at each staged statement, and the record's error at
  part `s` is Flock's proved `--zk` simulation error at the outer session's leaf counts (`nums_εzk`: `8·n_L·δ₁ +
  8t·n_B·δ₁ + 2·3^{−t}` at `Brings.zl`, `Flock.zkR_gkE_hm96_deployZk`'s at `nL`, `nB`). The hybrid needs Flock's ZK
  at `deployZkRec` at each staged statement and the witness the inner prover's output gives (`RefParts.wit`).

`ZK` reads the record's own reference prover (`ZkDeploy.prover`: `refP` on the tape `tapeOf x` draws). `ZKmalRx` reads
a finite tape at each statement, and the reference prover's tape (`Tape b`: a staging salt and a masked prover's tape
for every index) is not finite, so the record draws it at each statement `x` from a finite nonce and rest of `x`'s,
and `ZKBound` is stated at the tape map `tapeOf`.

`ZKPriv` at `deployRec` holds outright (`zkPriv_deployRec`): the shipped verifier has no private-circuit form, so
`parse` reads no circuit commitment (`parse_circ`) and no statement is in `RPriv`.
-/

namespace RecDeploy

open Flock
open FlockDeploy (Coin)
open scoped NNReal

variable {N Ω : Claims.Msg → Type} [∀ x, Fintype (N x)] [∀ x, Nonempty (N x)] [∀ x, Fintype (Ω x)]
  [∀ x, Nonempty (Ω x)]

/-! ## The masked record and the claim's bound -/

/-- **The masked record at `deployRec`**: the reference prover's tape at each statement `x` drawn from the finite nonce
and rest `N x × Ω x` by `tapeOf x`, and the outer simulators `SO`. -/
noncomputable def deployRecZ (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZkDeploy Coin :=
  { deployRec IV ivScope b with Nonce := N, Rest := Ω, tapeOf := tapeOf, SO := SO }

/-- **`ZK`'s bound at `deployRec`**: at each attempt budget `t ≥ 1`, wherever hm96 leaves hide up to `δ₁` at the
pinned key, the reference prover on the tape `tapeOf x` draws at each statement `x` is simulated by `Sim` over the outer
simulators `SO t` against every malicious verifier of the compiled verifier's rounds, within `εzk` plus `γzk` times
`Fork`'s collision probability. -/
def ZKBound (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b)
    (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) : Prop :=
  ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
    Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.ZKmalRx N Ω H (deployRec IV ivScope b).V.k (deployRecZ IV ivScope b tapeOf SO).prover
        (Claims.Recursion.R (deployRec IV ivScope b) I) (Claims.Recursion.Sim (deployRec IV ivScope b) I (SO t))
        (Claims.Recursion.εzk (deployRec IV ivScope b) I t δ₁) (Claims.Recursion.γzk (deployRec IV ivScope b) t)

/-- **The recursion's ZK at `deployRec`, from its bound**: the outer record is the masked one. -/
theorem zk_deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim)
    (h : ZKBound H IV ivScope b I tapeOf SO) :
    Claims.Recursion.ZK H (deployRecZ IV ivScope b tapeOf SO) I (deployZkRec relO) :=
  ⟨rfl, h⟩

/-- **The recursion's ZK at `deployRec` is its bound.** -/
theorem zk_deployRec_iff (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZK H (deployRecZ IV ivScope b tapeOf SO) I (deployZkRec relO) ↔
      ZKBound H IV ivScope b I tapeOf SO :=
  ⟨And.right, zk_deployRec H IV ivScope b I relO tapeOf SO⟩

/-! ## What `deployRec` fixes of the hybrid -/

/-- **The reference prover's staging round is the record's `stage`**, at the commit strings (`commitThen`) of its
staged values' digests under the staging salts on its tape. -/
theorem refP_stage (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (w : Claims.Flock.Wit)
    (x : Claims.Msg) (hist : List (Claims.Msg × Claims.Msg)) (ω : Tape b) (h₁ : IV.k (xIOf x) ≤ hist.length)
    (h₂ : hist.length < IV.k (xIOf x) + kS x) :
    (refP IV b w).next x hist ω =
      commitThen (fun cs => (deployRec IV ivScope b).stage x (asRounds (hist.take (IV.k (xIOf x)))) cs)
        ((List.range ((deployRec IV ivScope b).nC x)).map fun j =>
          ((b.ref.digests w ω.1 (hist.take (IV.k (xIOf x)))).getD j 0, ω.2.1 j)) [] := by
  simp only [refP, deployRec, Nat.not_lt.2 h₁, h₂, ↓reduceIte]

/-- **At the zero digest, the prover's commit strings are `Sim`'s**: `commitThen`'s commit string to the zero digest
under a salt is `Deploy.zeroCommit`'s at `deployRec`, the same query and the same string. -/
theorem commitThen_zero (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings)
    (k : List Claims.Msg → Claims.OC Claims.Msg) (y : Salt) (ps : List ((Fin 64 → UInt8) × Salt))
    (cs : List Claims.Msg) :
    commitThen k ((0, y) :: ps) cs =
      Claims.OC.bind ((deployRec IV ivScope b).zeroCommit y) fun c => commitThen k ps (cs ++ [c]) := by
  simp only [commitThen, Claims.Recursion.Deploy.zeroCommit, Claims.OC.bind, deployRec, maskOf, Pi.zero_apply,
    UInt8.zero_xor]

/-- **Part `s`'s simulation error is Flock's proved `--zk` error** at the outer session's leaf counts (`Brings.zl` at
the staged statement `nums` reads): `Flock.zkR_gkE_hm96_deployZk`'s at `nL`, `nB`. -/
theorem nums_εzk (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (x : Claims.Msg) (s : ℕ)
    (t : ℕ) (δ₁ : ℝ) :
    ((deployRec IV ivScope b).nums x s).εzk t δ₁ =
      8 * (b.zl (staged b.st x (pick b.st x s) s)).1 * δ₁ + 8 * t * (b.zl (staged b.st x (pick b.st x s) s)).2 * δ₁ +
        2 * (1 / 3) ^ t :=
  rfl

/-! ## The private variant -/

/-- **`parse` reads no circuit commitment** (`RecScope.toStmt`). -/
theorem parse_circ {x : Claims.Msg} {s : Claims.Recursion.CompiledStatement} (h : parse x = some s) : s.circ = none := by
  simp only [parse, RecScope.parseOf, Option.map_eq_some_iff] at h
  obtain ⟨r, -, rfl⟩ := h
  rfl

/-- **`ZKPriv` at `deployRec` holds**: no statement is in `RPriv`, as `parse` reads no circuit commitment. -/
theorem zkPriv_deployRec (H : Claims.Msg → Claims.Msg) (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool)
    (b : Brings) (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (tapeOf : (x : Claims.Msg) → N x × Ω x → Tape b) (SO : ℕ → Claims.Msg → ℕ → Claims.Recursion.OSim) :
    Claims.Recursion.ZKPriv H (deployRecZ IV ivScope b tapeOf SO) I (deployZkRec relO) := by
  refine ⟨rfl, fun t _ δ₁ _ x w hR => ?_⟩
  obtain ⟨s, hs, hc, -⟩ := hR
  rw [parse_circ (x := x) hs] at hc
  cases hc

end RecDeploy
