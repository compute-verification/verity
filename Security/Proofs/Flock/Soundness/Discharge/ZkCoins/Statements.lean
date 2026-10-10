import Proofs.Flock.Soundness.Discharge.ZkCoins.SessionDefs
import Proofs.Flock.Soundness.CROnly.Hm96Defs

/-!
# The coin step's guarantees, as statements

* `CommittedLe`: on any interaction that opens each committed round at most once, a committed prover succeeds with
  probability at most the optimal prover's against live coins (`value p g.erase`) plus `δ₁` for each committed round,
  where `δ₁` is the leaf's hiding gap (`Assumptions.Hm96Hiding`).
* `CommittedLeHm96`: the same at hm96-sha512 under a fresh uniform key, with `δ₁ = 2^-257` and no hypothesis on the
  key: the key is drawn first and is public, and the event, the interaction, the digests, the leaf hash and the prover
  may all depend on it.
* `SessionCoinsSound`, `SessionCoinsHm96`: the batched `--zk` session on C-Flock ZK's coin tree (`sessionZKC`): table
  `j`'s `padLinkError + (padRepError + εInner)²` plus `δ₁` (resp. `2^-257`) for each of the `1 + 2·J·R` committed
  rounds.

The bound against live coins is `value`, the optimal adaptive prover's against independent live coins, so the table
bound's square (`value_interleave_le`: two reps on independent coins) carries over whole; the commitment adds a sum over
committed rounds, not a factor.
-/

namespace FlockSoundness.Discharge.ZkCoins

open FlockSoundness Game
open scoped ENNReal

/-- **Committed coins cost their hiding gap once a round.** If no path of `g` opens a committed round twice and the
leaf's hiding gap is `δ₁` (`Assumptions.Hm96Hiding My cs δ₁`), then a committed prover `τ`, which sees every round's leaf
before its first message and each round's block and salt at its opening, makes `g`'s output satisfy `p` with
probability at most `value p g.erase + |I|·δ₁`. `dig`, `H` and the event are arbitrary. -/
def CommittedLe : Prop :=
  ∀ {I Blk Salt B Cs D α : Type} [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt]
    [AddCommGroup B] [Fintype B] (p : α → Prop) (g : TGame I Blk α), g.Distinct →
    ∀ (dig : I → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D) (δ₁ : ℝ≥0∞),
      Assumptions.Hm96Hiding My cs δ₁ →
      ∀ τ : (I → D) → TGame.CStrat Salt g, probC p g dig My cs H τ ≤ value p g.erase + Fintype.card I * δ₁

/-- **Committed coins at hm96-sha512 under a fresh key.** As `CommittedLe`, with the leaf
`H κ (dig κ i b + M(κ)·y, cs y)` for a 1536-bit salt `y`, the Hankel matrix `M(κ)` of a uniform 2047-bit key `κ` drawn
first, and a salt hash `cs` with at most `2^512` values. The event `p κ`, the interaction `g κ`, the digests `dig κ`, the
leaf hash `H κ` (hm96-sha512's carries `H(κ)`) and the prover `τ κ` are chosen after the key: the probability, averaged
over the key, is at most the average of `value (p κ) (g κ).erase` plus `|I|·2^-257`. -/
def CommittedLeHm96 : Prop :=
  ∀ {E I Blk Cs D α : Type} [Field E] [Fintype E] [DecidableEq E], Fintype.card E = 2 →
    ∀ [Fintype I] [DecidableEq I] [Fintype Blk] [Nonempty Blk] [DecidableEq Cs]
      (p : (Fin (512 + 1536 - 1) → E) → α → Prop) (g : (Fin (512 + 1536 - 1) → E) → TGame I Blk α),
      (∀ κ, (g κ).Distinct) →
      ∀ (dig : (Fin (512 + 1536 - 1) → E) → I → Blk → (Fin 512 → E)) (cs : (Fin 1536 → E) → Cs),
      (Finset.univ.image cs).card ≤ 2 ^ 512 →
      ∀ (H : (Fin (512 + 1536 - 1) → E) → (Fin 512 → E) × Cs → D)
        (τ : (κ : Fin (512 + 1536 - 1) → E) → (I → D) → TGame.CStrat (Fin 1536 → E) (g κ)),
        avg (fun κ => probC (p κ) (g κ) (dig κ) (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) (τ κ)) ≤
          avg (fun κ => value (p κ) (g κ).erase) + Fintype.card I * ENNReal.ofReal (1 / 2 ^ 257)

/-- **Soundness of table `j` in a batched `--zk` session on committed coins.** For every committed prover, the session
on C-Flock ZK's coin tree (`sessionZKC`, `R` rounds a stream, any balanced decoders `dc`) accepts while table `j`'s
committed values do not satisfy its statement with probability at most table `j`'s
`padLinkError + (padRepError + εInner)²` plus `δ₁` for each of its `1 + 2·J·R` committed rounds. -/
def SessionCoinsSound : Prop :=
  ∀ {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {mPts : ℕ}
    (A : Model.Arith F K), A.Correct → ∀ (ts : List (ZkSession.TabZK F mPts)) (j : Fin ts.length)
    {Blk Salt B Cs D : Type} [Fintype Blk] [Nonempty Blk] [Fintype Salt] [Nonempty Salt] [AddCommGroup B] [Fintype B]
    (dc : Decoders Blk) (R : ℕ) (dig : SessIdx ts.length R → Blk → B) (My : Salt → B) (cs : Salt → Cs) (H : B × Cs → D)
    (δ₁ : ℝ≥0∞), Assumptions.Hm96Hiding My cs δ₁ →
    ∀ τ : (SessIdx ts.length R → D) → TGame.CStrat Salt (sessionZKC dc R A ts),
      probC (fun outs => ZkSession.acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j ZkSession.dfltZK).2)
          (sessionZKC dc R A ts) dig My cs H τ ≤
        (ts.get j).bound + ((1 + 2 * ts.length * R : ℕ) : ℝ≥0∞) * δ₁

/-- **Soundness of table `j` in a batched `--zk` session on C-Flock ZK's coin tree at hm96-sha512.** As
`SessionCoinsSound`, with the leaves under a fresh uniform 2047-bit key that the prover sees, and the digests `dig κ`
and the leaf hash `H κ` chosen after it: table `j`'s bound plus `2^-257` for each of the `1 + 2·J·R` committed rounds,
with no hypothesis on the key. -/
def SessionCoinsHm96 : Prop :=
  ∀ {F K E : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] [Field E] [Fintype E]
    [DecidableEq E] {mPts : ℕ} (A : Model.Arith F K), A.Correct → Fintype.card E = 2 →
    ∀ (ts : List (ZkSession.TabZK F mPts)) (j : Fin ts.length) {Blk Cs D : Type} [Fintype Blk] [Nonempty Blk] [DecidableEq Cs]
      (dc : Decoders Blk) (R : ℕ) (dig : (Fin (512 + 1536 - 1) → E) → SessIdx ts.length R → Blk → (Fin 512 → E))
      (cs : (Fin 1536 → E) → Cs), (Finset.univ.image cs).card ≤ 2 ^ 512 →
      ∀ (H : (Fin (512 + 1536 - 1) → E) → (Fin 512 → E) × Cs → D)
        (τ : (Fin (512 + 1536 - 1) → E) → (SessIdx ts.length R → D) → TGame.CStrat (Fin 1536 → E) (sessionZKC dc R A ts)),
        avg (fun κ => probC (fun outs => ZkSession.acceptedZK ts outs ∧ ¬ (ts.get j).Committed A (outs.getD j ZkSession.dfltZK).2)
            (sessionZKC dc R A ts) (dig κ) (CROnly.hankel (n := 512) (l := 1536) κ) cs (H κ) (τ κ)) ≤
          (ts.get j).bound + ((1 + 2 * ts.length * R : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 257)

end FlockSoundness.Discharge.ZkCoins
