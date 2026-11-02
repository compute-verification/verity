import Proofs.Flock.Claims.GS.RadiusDeploy
import Proofs.Flock.Claims.ERepCost
import Proofs.Flock.Claims.DeployCode

/-!
# Flock's efficiency claim at both deployed records

`Flock.Efficient H D p` is four facts about the record's own decoder `D.dec`, a named, counted program: `E^rep`'s reruns
(`ERepCost` at `D.V`, `D.waits`, `D.k`, `D.cap` and `D.cols`), the decoder meeting the list-decoding spec at `D.len`,
`D.enc` and `D.θ` (`ListWithin`), its count at most `p` at the opened word's encoded length (`Flock.wordLen`), and its
lists that long at most, which bounds `assemble`'s checks. The first holds at every deployment (`Claims.ERep_cost`,
`Flock.Efficient_of_decoder`). Both records name the counted radius decoder (`GS.plainCountedRD`, `GS.zkCountedRD`),
whose count and lists are at most `14·(flockPolyR + X)` (`11·(zkPolyR + X)` masked) at its family's size; that size
is at most its value on no columns times one more than their number (`GS.plainSizeRD_le`), and their number is at most
the word's encoded length (`Flock.length_le_wordLen`). So, at every `H`:

* `efficient_plain`: the plain record the claims read, `deployPlainTrim rel`, at `plainStepPoly`, `14·(flockPolyR + X)`
  composed with `plainSizeRD [] · (X + 1)`;
* `efficient_zk`: the masked record `deployZkTrim rel` at `zkStepPoly`, `11·(zkPolyR + X)` composed with
  `zkSizeRD [] · (X + 1)`. The masked columns and code are the `2^k0` witness lanes of each opened row (`zkRepColsOf`,
  `Flock.zkWidth i = 2^k0`), so `zkDecodeRD` decodes those lanes only.

A bound too small fails (`not_efficient_plain`, `not_efficient_zk`): on no columns the word's encoded length is `0` and
the decoder still fills its words, so no `p` with `p(0) = 0`, `0` and `X` among them, bounds its count.

At the deployed numbers (`erepRuns_plain`, `erepRuns_zk`), `E^rep` makes `Numbers.waits = 210` waits of at most
`Numbers.cap = 2^256` draws, so it reruns the prover at most `1 + 210·2^256` times, and `1 + 210·k` times in
expectation, `k = ⌈4·N₀/e⌉` with `N₀` the tables' level-0 codeword lengths summed (`2^d` at one table).
-/

namespace Claims

open Polynomial

/-- **`Efficient` from a count at a size of the proof's choosing**: when the deployed decoder meets the spec, its count
and its lists are at most `q` at `size`, and `size` is at most `a` at the opened word's encoded length, `Efficient`
holds at `q ∘ a`. -/
theorem Flock.Efficient_of_size {Coin F T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype F]
    [DecidableEq F] [Zero (List F)] (H : Msg → Msg) (D : Flock.Deploy Coin (List F) T)
    (size : List (ℕ × List F) → ℕ) (q a : Polynomial ℕ) (hl : Flock.ListWithin D.len D.enc D.θ D.decode)
    (hs : ∀ os, (D.dec os).2 ≤ q.eval (size os)) (hlen : ∀ os, (D.decode os).length ≤ q.eval (size os))
    (ha : ∀ os, size os ≤ a.eval (Flock.wordLen os)) : Flock.Efficient H D (q.comp a) := by
  have hq : ∀ os, q.eval (size os) ≤ (q.comp a).eval (Flock.wordLen os) := fun os => by
    rw [eval_comp]
    exact GS.eval_mono_nat q (ha os)
  exact Flock.Efficient_of_decoder H D _ hl (fun os => (hs os).trans (hq os)) fun os => (hlen os).trans (hq os)

/-- **A size at most `c·(L + 1)`** at `L` opened columns is at most `c·(X + 1)` at the word's encoded length. -/
theorem Flock.le_eval_wordLen {F : Type} [Fintype F] (size : List (ℕ × List F) → ℕ) (c : ℕ)
    (h : ∀ os, size os ≤ c * (os.length + 1)) (os : List (ℕ × List F)) :
    size os ≤ (C c * (Polynomial.X + 1)).eval (Flock.wordLen os) := by
  simp only [eval_mul, eval_C, eval_add, eval_X, eval_one]
  exact (h os).trans (Nat.mul_le_mul_left c (Nat.add_le_add_right (Flock.length_le_wordLen os) 1))

/-- **A bound too small fails**: a record whose decoder counts a step on no columns is not `Efficient` at any `p`
with `p(0)` below that count. -/
theorem Flock.not_efficient_of_nil {Coin F T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [Fintype F]
    [DecidableEq F] [Zero (List F)] (H : Msg → Msg) (D : Flock.Deploy Coin (List F) T) (p : Polynomial ℕ)
    (hp : p.eval 0 < (D.dec []).2) : ¬ Flock.Efficient H D p := fun hE => by
  have h := hE.2.2.1 []
  simp only [Flock.wordLen, List.map_nil, List.sum_nil] at h
  omega

end Claims

namespace FlockDeploy

open Flock Polynomial

local notation "𝔽" => FlockLevel3.GF128

/-- The plain record's decoder is the plain radius decoder. -/
theorem decode_plain (rel : Readable.Parsed → Readable.Relation) :
    (deployPlainTrim rel).decode = Claims.GS.plainDecodeRD :=
  funext Claims.GS.plainCountedRD_fst

/-- The masked record's decoder is the masked radius decoder. -/
theorem decode_zk (rel : Readable.Parsed → Readable.Relation) : (deployZkTrim rel).decode = Claims.GS.zkDecodeRD :=
  funext Claims.GS.zkCountedRD_fst

/-- **The plain radius decoder meets `ListWithin`** at the plain instance's code, length and radius. -/
theorem listWithin_plain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployPlain rel).len (deployPlain rel).enc (deployPlain rel).θ Claims.GS.plainDecodeRD :=
  Claims.GS.listWithin_plainDecodeRD id fun _ _ => rfl

/-- **The masked radius decoder meets `ListWithin`** at the masked instance's code, length and radius. -/
theorem listWithin_zk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployZk rel).len (deployZk rel).enc (deployZk rel).θ Claims.GS.zkDecodeRD :=
  Claims.GS.listWithin_zkDecodeRD id fun _ _ => rfl

/-- **The plain radius decoder meets `ListWithin`** at the plain record the claims read, its code trimmed of trailing
zeros: the spec `Flock.KS` and `Flock.Bits` ask of a decoder there, and the record's own decoder (`decode_plain`). -/
theorem listWithin_plainTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployPlainTrim rel).len (deployPlainTrim rel).enc (deployPlainTrim rel).θ
      Claims.GS.plainDecodeRD :=
  Claims.GS.listWithin_plainDecodeRD trimCol trimCol_getD

/-- **The masked radius decoder meets `ListWithin`** at the masked record the claims read, its code trimmed. -/
theorem listWithin_zkTrim (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListWithin (deployZkTrim rel).len (deployZkTrim rel).enc (deployZkTrim rel).θ Claims.GS.zkDecodeRD :=
  Claims.GS.listWithin_zkDecodeRD trimCol trimCol_getD

/-- **The plain record's step polynomial**: `14·(flockPolyR + X)`, the plain family's bound at its size, composed with
`plainSizeRD [] · (X + 1)`, that size's bound at the word's encoded length. -/
noncomputable def plainStepPoly : Polynomial ℕ :=
  (C 14 * (Claims.GS.flockPolyR + Polynomial.X)).comp (C (Claims.GS.plainSizeRD []) * (Polynomial.X + 1))

/-- **The masked record's step polynomial**: `11·(zkPolyR + X)`, the masked family's bound at its size, composed with
`zkSizeRD [] · (X + 1)`. -/
noncomputable def zkStepPoly : Polynomial ℕ :=
  (C 11 * (Claims.GS.zkPolyR + Polynomial.X)).comp (C (Claims.GS.zkSizeRD []) * (Polynomial.X + 1))

/-- **Flock: efficiency at the deployed plain verifier**, at every `H`: `E^rep`'s reruns, and the record's decoder, the
counted plain radius decoder, meets `ListWithin` at the record's `len`, `enc` and `θ`, with its count and its lists at
most `plainStepPoly` at the opened word's encoded length. -/
theorem efficient_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Efficient H (deployPlainTrim rel) plainStepPoly := by
  refine Claims.Flock.Efficient_of_size H (deployPlainTrim rel) Claims.GS.plainSizeRD _ _ ?_
    (fun os => (Claims.GS.plainDecodeRD_complexity os).2) ?_
    (Claims.Flock.le_eval_wordLen _ _ Claims.GS.plainSizeRD_le)
  · rw [decode_plain]
    exact listWithin_plainTrim rel
  · rw [decode_plain]
    exact Claims.GS.plainDecodeRD_length_le

/-- **Flock: efficiency at the deployed masked (`--zk`) verifier**, at every `H`: `E^rep`'s reruns, and the record's
decoder, the counted masked radius decoder, meets `ListWithin` at the record's `len`, `enc` and `θ`, with its count and
its lists at most `zkStepPoly` at the opened word's encoded length. -/
theorem efficient_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.Efficient H (deployZkTrim rel) zkStepPoly := by
  refine Claims.Flock.Efficient_of_size H (deployZkTrim rel) Claims.GS.zkSizeRD _ _ ?_
    (fun os => (Claims.GS.zkDecodeRD_complexity os).2) ?_
    (Claims.Flock.le_eval_wordLen _ _ Claims.GS.zkSizeRD_le)
  · rw [decode_zk]
    exact listWithin_zkTrim rel
  · rw [decode_zk]
    exact Claims.GS.zkDecodeRD_length_le

/-- **A bound too small fails at the plain record**: its decoder counts a step on no columns, whose encoded length is
`0`, so it is not `Efficient` at any `p` with `p(0) = 0`. -/
theorem not_efficient_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (p : Polynomial ℕ) (hp : p.eval 0 = 0) : ¬ Claims.Flock.Efficient H (deployPlainTrim rel) p :=
  Claims.Flock.not_efficient_of_nil H _ p (by rw [hp]; exact Claims.GS.plainCountedRD_nil_pos)

/-- **A bound too small fails at the masked record**: not `Efficient` at any `p` with `p(0) = 0`. -/
theorem not_efficient_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (p : Polynomial ℕ) (hp : p.eval 0 = 0) : ¬ Claims.Flock.Efficient H (deployZkTrim rel) p :=
  Claims.Flock.not_efficient_of_nil H _ p (by rw [hp]; exact Claims.GS.zkCountedRD_nil_pos)

/-- **Neither record is `Efficient` at `0` or at `X`.** -/
theorem not_efficient_zero_X (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation) :
    ¬ Claims.Flock.Efficient H (deployPlainTrim rel) 0 ∧ ¬ Claims.Flock.Efficient H (deployPlainTrim rel) Polynomial.X ∧
      ¬ Claims.Flock.Efficient H (deployZkTrim rel) 0 ∧ ¬ Claims.Flock.Efficient H (deployZkTrim rel) Polynomial.X :=
  ⟨not_efficient_plain H rel 0 (eval_zero), not_efficient_plain H rel Polynomial.X (eval_X),
    not_efficient_zk H rel 0 (eval_zero), not_efficient_zk H rel Polynomial.X (eval_X)⟩

/-- **`E^rep`'s reruns at the deployed plain verifier**, at every decoder: at most `1 + 210·2^256`, and
`1 + 210·⌈4·N₀/e⌉` in expectation. -/
theorem erepRuns_plain (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.PlainTrace) :
    Claims.ExtractRunsLe H (Claims.Flock.E (deployPlainTrim rel) decode) (fun _ => 1 + 210 * 2 ^ 256) ∧
      Claims.ExtractRunsExpLe H (Claims.Flock.E (deployPlainTrim rel) decode)
        (fun x => 1 + 210 * (⌈4 * N₀ false x / Real.exp 1⌉₊ : ℝ)) :=
  Claims.ERep_cost H _ _ _ _ _ _ _

/-- **`E^rep`'s reruns at the deployed masked verifier**, at every decoder: at most `1 + 210·2^256`, and
`1 + 210·⌈4·N₀/e⌉` in expectation. -/
theorem erepRuns_zk (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (decode : List (ℕ × List 𝔽) → List Claims.Flock.ZkTrace) :
    Claims.ExtractRunsLe H (Claims.Flock.E (deployZkTrim rel) decode) (fun _ => 1 + 210 * 2 ^ 256) ∧
      Claims.ExtractRunsExpLe H (Claims.Flock.E (deployZkTrim rel) decode)
        (fun x => 1 + 210 * (⌈4 * N₀ true x / Real.exp 1⌉₊ : ℝ)) :=
  Claims.ERep_cost H _ _ _ _ _ _ _

end FlockDeploy
