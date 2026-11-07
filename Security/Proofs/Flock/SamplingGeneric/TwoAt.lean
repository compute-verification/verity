module

public import Proofs.Flock.SamplingGeneric.TwoCode
public import Proofs.Flock.SamplingGeneric.Session
public import Proofs.Flock.Soundness.Assumptions
public import Proofs.Core.Game.Basic

/-!
# The two-stage sampled verifier at its round order (`SampledKSTwo`)

`SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂` is `codeTwo` at the deployed verifier's round order. The statement's
round pair `j x` (`Order`) puts the coin the first stage reads (`first`) before the coin the second stage reads
(`second`), as `SampledProofs.Run.run` reads the second stage's coins after the interior registration's digest and
delivers the sessions after the second stage. So the second stage is every round after the first stage's coin
(`Code.atOrder`), and a session's rounds are those of `c`'s after the second stage's coin. Each stage's bytes are read
off its round's coin (`bytesFirst`, `bytesSecond`), by the challenger's byte maps `b₁ x` and `b₂ x`.

At this code, four of `ksTwo_code`'s hypotheses are facts of the order and the byte maps:

* `firstFixed_atOrder`, `bytesFixedTwo_atOrder`: `FirstFixed` and `BytesFixedTwo` by the order alone. The first stage's
  coin is no second-stage round, and neither stage's coin is a session's round.
* `bytesUniform_atOrder`, `secondUniform_atOrder`: `BytesUniform` and `SecondUniform` from `UniformRandomBytes` at each
  challenger call (`CallsUniform`): the bytes `b x` returns on its round's coin are uniform. A3 holds every call's
  bytes independent of every earlier call's, so the second stage's bytes are uniform whatever the first stage drew.
* `ksTwo_atOrder`: `CompiledTwo.KS` at Flock's masked record and SHA-512 from those two calls' A3 and six facts of
  the code (`RegMatchesTwo`, `RefusesTwo`, `InteriorFixed`, `SessionRuns`, `SessionsIn`, `SessionOpensTwo`). The
  verifier in public-coin form, its parser, partition, sessions and reveals are `c`'s.

`Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo` states it. Its witnesses:

* `sampledKSTwo_honest`: every premise at a toy code (`toyCode`) that parses no statement, accepts none, opens no
  session and reads no bytes;
* `compilerKSTwo_nontrivial`: the conclusion's `CompiledTwo.KS` fails at a verifier that accepts with an empty relation.
-/

@[expose] public section

namespace SamplingDeploy

open Claims Claims.Sampling Finset

/-! ## The round order -/

/-- **The deployed round order** of a statement's `k` rounds: the coin the first stage reads (`first`) comes before the
coin the second stage reads (`second`). -/
structure Order (k : ℕ) where
  first : Fin k
  second : Fin k
  lt : first < second

/-- **The code at its round order**: `c` with its second stage the rounds after the first stage's coin, and each
session's rounds those of `c`'s after the second stage's coin. -/
def Code.atOrder (c : Code) (j : (x : Msg) → Order (c.V.k x)) : Code :=
  { c with
    stage2 := fun x => univ.filter fun t => (j x).first < t
    rounds := fun x u => (c.rounds x u).filter fun t => (j x).second < t }

/-- **A challenger call's bytes are uniform** (`UniformRandomBytes`, A3): at every statement `x`, the bytes `b x` returns
on a uniform coin. -/
def CallsUniform (L : Msg → ℕ) (b : (x : Msg) → FlockDeploy.Coin → Fin (L x) → Fin 256) : Prop :=
  ∀ x, FlockSoundness.Assumptions.UniformRandomBytes (fun _ : Fin 1 => L x) fun c _ => b x c

variable (c : Code) (reg : Msg → Option RegTwo) (two : AuditStatement → Option Claims.Sampling.Design.Two)
  (L₁ L₂ : Msg → ℕ) (j : (x : Msg) → Order (c.V.k x))
  (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256) (b₂ : (x : Msg) → FlockDeploy.Coin → Fin (L₂ x) → Fin 256)

/-- The first stage's bytes: `b₁ x` on the first stage's coin. -/
def bytesFirst : (x : Msg) → (Fin ((c.atOrder j).V.k x) → FlockDeploy.Coin) → Fin (L₁ x) → Fin 256 :=
  fun x r => b₁ x (r (j x).first)

/-- The second stage's bytes: `b₂ x` on the second stage's coin. -/
def bytesSecond : (x : Msg) → (Fin ((c.atOrder j).V.k x) → FlockDeploy.Coin) → Fin (L₂ x) → Fin 256 :=
  fun x r => b₂ x (r (j x).second)

/-- **The two-stage sampled verifier's code at its round order.** -/
abbrev codeTwoAt : Code := codeTwo (c.atOrder j) reg L₁ L₂ (bytesFirst c L₁ j b₁) (bytesSecond c L₂ j b₂)

/-! ## The bytes -/

/-- **A uniform call's bytes have the uniform law**: A3 at a source of one stream. -/
theorem pr_call {Ω : Type} [Fintype Ω] {L : ℕ} {b : Ω → Fin L → Fin 256}
    (h : FlockSoundness.Assumptions.UniformRandomBytes (fun _ : Fin 1 => L) fun ω _ => b ω)
    (q : (Fin L → Fin 256) → Prop) : Pr (fun ω => q (b ω)) = Pr q := by
  have h1 := h fun f => q (f 0)
  have h2 := Claims.SamplingGeneric.Pr_pi_eval_dep (E := fun _ : Fin 1 => Fin L → Fin 256) 0 q
  rw [Pr_eq_toReal_prCoin, Pr_eq_toReal_prCoin] at h2
  rw [Pr_eq_toReal_prCoin, Pr_eq_toReal_prCoin, ← h2]
  exact congrArg ENNReal.toReal h1

theorem firstFixed_atOrder : FirstFixed (c.atOrder j) L₁ (bytesFirst c L₁ j b₁) := by
  intro x r e
  have h : (j x).first ∉ (univ.filter fun t => (j x).first < t : Finset (Fin (c.V.k x))) :=
    fun h => lt_irrefl _ (mem_filter.1 h).2
  change b₁ x (if (j x).first ∈ (univ.filter fun t => (j x).first < t : Finset (Fin (c.V.k x))) then e (j x).first
    else r (j x).first) = b₁ x (r (j x).first)
  simp only [h, ↓reduceIte]

theorem bytesFixedTwo_atOrder :
    BytesFixedTwo (c.atOrder j) L₁ L₂ (bytesFirst c L₁ j b₁) (bytesSecond c L₂ j b₂) := by
  intro x u
  have hlt := (j x).lt
  have h₁ : (j x).first ∉ ((c.rounds x u).filter fun t => (j x).second < t) :=
    fun h => absurd (mem_filter.1 h).2 (not_lt.2 hlt.le)
  have h₂ : (j x).second ∉ ((c.rounds x u).filter fun t => (j x).second < t) :=
    fun h => lt_irrefl _ (mem_filter.1 h).2
  refine ⟨fun t ht => mem_filter.2 ⟨mem_univ _, hlt.trans (mem_filter.1 ht).2⟩, fun r d => ⟨?_, ?_⟩⟩
  · change b₁ x (if (j x).first ∈ ((c.rounds x u).filter fun t => (j x).second < t) then d (j x).first
      else r (j x).first) = b₁ x (r (j x).first)
    simp only [h₁, ↓reduceIte]
  · change b₂ x (if (j x).second ∈ ((c.rounds x u).filter fun t => (j x).second < t) then d (j x).second
      else r (j x).second) = b₂ x (r (j x).second)
    simp only [h₂, ↓reduceIte]

theorem bytesUniform_atOrder (h : CallsUniform L₁ b₁) : BytesUniform (c.atOrder j) L₁ (bytesFirst c L₁ j b₁) := by
  intro x q
  change Pr (fun r : Fin (c.V.k x) → FlockDeploy.Coin => (fun a => q (b₁ x a)) (r (j x).first)) = Pr q
  rw [Claims.SamplingGeneric.Pr_pi_eval_dep (E := fun _ => FlockDeploy.Coin) (j x).first fun a => q (b₁ x a)]
  exact pr_call (h x) q

theorem secondUniform_atOrder (h : CallsUniform L₂ b₂) : SecondUniform (c.atOrder j) L₂ (bytesSecond c L₂ j b₂) := by
  intro x r q
  have hin : (j x).second ∈ (univ.filter fun t => (j x).first < t : Finset (Fin (c.V.k x))) :=
    mem_filter.2 ⟨mem_univ _, (j x).lt⟩
  change Pr (fun e : Fin (c.V.k x) → FlockDeploy.Coin =>
    q (b₂ x (if (j x).second ∈ (univ.filter fun t => (j x).first < t : Finset (Fin (c.V.k x))) then e (j x).second
      else r (j x).second))) = Pr q
  simp only [hin, ↓reduceIte]
  exact (Claims.SamplingGeneric.Pr_pi_eval_dep (E := fun _ => FlockDeploy.Coin) (j x).second
    fun a => q (b₂ x a)).trans (pr_call (h x) q)

/-! ## Knowledge soundness at the round order -/

/-- **The two-stage compiler at the deployed record, at its round order**: `CompiledTwo.KS` at Flock's masked record
and SHA-512 (`ksTwo_code`), with `BytesUniform` and `SecondUniform` from each call's A3 and `FirstFixed` and
`BytesFixedTwo` from the order. -/
theorem ksTwo_atOrder (iroots : Msg → Seen → ℕ → List Msg) (hA₁ : CallsUniform L₁ b₁) (hA₂ : CallsUniform L₂ b₂)
    (hreg : RegMatchesTwo (c.atOrder j) reg two) (hr : (c.atOrder j).RefusesTwo FlockDeploy.sha512 two)
    (hi : (c.atOrder j).InteriorFixed FlockDeploy.sha512 iroots)
    (hruns : Claims.Sampling.SessionRuns FlockDeploy.sha512
      (deploy _root_.Flock.Readable.relation (codeTwoAt c reg L₁ L₂ j b₁ b₂)))
    (hnum : SessionsIn (codeTwoAt c reg L₁ L₂ j b₁ b₂))
    (hopens : Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (deploy _root_.Flock.Readable.relation (codeTwoAt c reg L₁ L₂ j b₁ b₂)) two iroots) :
    (compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation) (codeTwoAt c reg L₁ L₂ j b₁ b₂) iroots two
      (replayTwo (c.atOrder j) reg L₁ (bytesFirst c L₁ j b₁))).KS FlockDeploy.sha512 :=
  ksTwo_code (c.atOrder j) reg two L₁ L₂ (bytesFirst c L₁ j b₁) (bytesSecond c L₂ j b₂) iroots hreg
    (bytesUniform_atOrder c L₁ j b₁ hA₁) (firstFixed_atOrder c L₁ j b₁) (secondUniform_atOrder c L₂ j b₂ hA₂)
    (bytesFixedTwo_atOrder c L₁ L₂ j b₁ b₂) hr hi hruns hnum hopens

end SamplingDeploy

namespace Flock.Guarantees

open Claims

/-- **The deployed two-stage sampled verifier is knowledge sound at SHA-512**, at its round order (`codeTwoAt`): on
the challenger's bytes uniform at each of its two calls (`CallsUniform`, `UniformRandomBytes`), and the code's facts
(the registration is the statement's two-stage design, the verifier refuses outside the two-stage scope, the interior
roots don't read the second stage's coins, each checked unit's session runs Flock on its rounds within its numbers and
opens the unit's proof block on its replay block), `CompiledTwo.KS` holds at Flock's masked record over the readable
relation and SHA-512: `KSLinOn` at the two-stage relation, the extractor's errors and the binding finder `Bext`. -/
def SampledKSTwo : Prop :=
  ∀ (c : SamplingDeploy.Code) (reg : Msg → Option SamplingDeploy.RegTwo)
    (two : Sampling.AuditStatement → Option Sampling.Design.Two) (L₁ L₂ : Msg → ℕ)
    (j : (x : Msg) → SamplingDeploy.Order (c.V.k x))
    (b₁ : (x : Msg) → FlockDeploy.Coin → Fin (L₁ x) → Fin 256)
    (b₂ : (x : Msg) → FlockDeploy.Coin → Fin (L₂ x) → Fin 256) (iroots : Msg → Seen → ℕ → List Msg),
    SamplingDeploy.CallsUniform L₁ b₁ → SamplingDeploy.CallsUniform L₂ b₂ →
    SamplingDeploy.RegMatchesTwo (c.atOrder j) reg two →
    (c.atOrder j).RefusesTwo FlockDeploy.sha512 two →
    (c.atOrder j).InteriorFixed FlockDeploy.sha512 iroots →
    Sampling.SessionRuns FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂)) →
    SamplingDeploy.SessionsIn (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂) →
    SamplingGeneric.SessionOpensTwo FlockDeploy.sha512
      (SamplingDeploy.deploy _root_.Flock.Readable.relation (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂)) two iroots →
    (SamplingDeploy.compiledTwo (FlockDeploy.deployZk _root_.Flock.Readable.relation)
      (SamplingDeploy.codeTwoAt c reg L₁ L₂ j b₁ b₂) iroots two
      (SamplingDeploy.replayTwo (c.atOrder j) reg L₁ (SamplingDeploy.bytesFirst c L₁ j b₁))).KS FlockDeploy.sha512

end Flock.Guarantees

theorem Flock.SecurityProofs.SampledKSTwo : Flock.Guarantees.SampledKSTwo :=
  fun c reg two L₁ L₂ j b₁ b₂ iroots hA₁ hA₂ hreg hr hi hruns hnum hopens =>
    SamplingDeploy.ksTwo_atOrder c reg two L₁ L₂ j b₁ b₂ iroots hA₁ hA₂ hreg hr hi hruns hnum hopens

/-! ## Witnesses -/

namespace SamplingDeploy

open Claims Claims.Sampling

/-- A verifier of two rounds that rejects every run. -/
def rejV2 : Verifier FlockDeploy.Coin := ⟨fun _ => 2, fun _ _ _ => .ret .rej⟩

theorem run_rejV2_out {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin (rejV2.k x) → FlockDeploy.Coin) : (run H P rejV2 x ω r).out = .rej := by
  simp [run, rejV2, stepWith, OC.eval]

/-- **The toy code**: it parses no statement, accepts no run and opens no session. -/
def toyCode : Code where
  V := rejV2
  parse _ := none
  scopeCheck _ := false
  partition _ := none
  stage2 _ := ∅
  checked _ _ := []
  rounds _ _ := ∅
  session _ _ _ := none
  reveal _ _ _ _ := []
  own := #[]
  at_ _ _ := none
  layer _ _ _ := []
  place _ := ⟨1, 0⟩
  bits _ _ := false
  salt _ := []
  π _ := []
  innerN₀ _ := 0
  tables _ := 0
  reveals _ := 0
  sessions _ := 0
  step _ _ := .say []
  reads _ _ := []
  wit _ _ _ := ⟨[]⟩

/-- The toy's round order: round `0`, then round `1`. -/
def toyOrder (x : Msg) : Order (toyCode.V.k x) := ⟨⟨0, Nat.zero_lt_two⟩, ⟨1, Nat.one_lt_two⟩, Nat.zero_lt_one⟩

/-- The toy's byte map, which reads no bytes. -/
def toyBytes (x : Msg) : FlockDeploy.Coin → Fin ((fun _ : Msg => 0) x) → Fin 256 := fun _ i => i.elim0

/-- **A3 holds at a source with one possible output.** -/
theorem uniformRandomBytes_of_subsingleton {Ω : Type} [Fintype Ω] [Nonempty Ω] {m : ℕ} {L : Fin m → ℕ}
    [Subsingleton ((i : Fin m) → Fin (L i) → Fin 256)] (t : Ω → (i : Fin m) → Fin (L i) → Fin 256) :
    FlockSoundness.Assumptions.UniformRandomBytes L t := by
  classical
  intro E
  rw [Verity.Game.prCoin_eq_card, Verity.Game.prCoin_eq_card]
  have : Nonempty ((i : Fin m) → Fin (L i) → Fin 256) := ⟨t (Classical.arbitrary Ω)⟩
  by_cases h : E (t (Classical.arbitrary Ω))
  · have h1 : Finset.univ.filter (fun ω => E (t ω)) = Finset.univ :=
      Finset.filter_true_of_mem fun ω _ => by rwa [Subsingleton.elim (t ω) (t (Classical.arbitrary Ω))]
    have h2 : Finset.univ.filter E = Finset.univ :=
      Finset.filter_true_of_mem fun f _ => by rwa [Subsingleton.elim f (t (Classical.arbitrary Ω))]
    rw [h1, h2, Finset.card_univ, Finset.card_univ,
      ENNReal.div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _),
      ENNReal.div_self (Nat.cast_ne_zero.2 Fintype.card_ne_zero) (ENNReal.natCast_ne_top _)]
  · have h1 : Finset.univ.filter (fun ω => E (t ω)) = ∅ :=
      Finset.filter_false_of_mem fun ω _ hω => h (by rwa [Subsingleton.elim (t ω) (t (Classical.arbitrary Ω))] at hω)
    have h2 : Finset.univ.filter E = ∅ :=
      Finset.filter_false_of_mem fun f _ hf => h (by rwa [Subsingleton.elim f (t (Classical.arbitrary Ω))] at hf)
    rw [h1, h2]
    simp

theorem callsUniform_toy : CallsUniform (fun _ => 0) toyBytes := fun _ =>
  haveI : Subsingleton ((i : Fin 1) → Fin 0 → Fin 256) := ⟨fun _ _ => funext fun _ => funext fun t => t.elim0⟩
  uniformRandomBytes_of_subsingleton _

/-- **`SampledKSTwo`'s premises hold at the toy code**, at its round order: both calls' A3 (they read no bytes), and
the code's facts, since it parses no statement, accepts no run and opens no session. -/
theorem sampledKSTwo_honest :
    CallsUniform (fun _ => 0) toyBytes ∧ CallsUniform (fun _ => 0) toyBytes ∧
      RegMatchesTwo (toyCode.atOrder toyOrder) (fun _ => none) (fun _ => none) ∧
      (toyCode.atOrder toyOrder).RefusesTwo FlockDeploy.sha512 (fun _ => none) ∧
      (toyCode.atOrder toyOrder).InteriorFixed FlockDeploy.sha512 (fun _ _ _ => []) ∧
      SessionRuns FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation
        (codeTwoAt toyCode (fun _ => none) (fun _ => 0) (fun _ => 0) toyOrder toyBytes toyBytes)) ∧
      SessionsIn (codeTwoAt toyCode (fun _ => none) (fun _ => 0) (fun _ => 0) toyOrder toyBytes toyBytes) ∧
      Claims.SamplingGeneric.SessionOpensTwo FlockDeploy.sha512 (deploy _root_.Flock.Readable.relation
        (codeTwoAt toyCode (fun _ => none) (fun _ => 0) (fun _ => 0) toyOrder toyBytes toyBytes))
        (fun _ => none) (fun _ _ _ => []) := by
  refine ⟨callsUniform_toy, callsUniform_toy, ?_, ?_, fun _ _ _ _ _ _ _ => rfl, ?_, ?_, ?_⟩
  · intro x s _ _ hx
    change (none : Option AuditStatement) = some s at hx
    simp at hx
  · intro x Ω P ω r hacc
    exact absurd ((run_rejV2_out FlockDeploy.sha512 P x ω r).symm.trans hacc) (by simp)
  · intro x Ω P ω r u ⟨p, hp⟩
    change ([] : List ℕ)[p.val]? = some u at hp
    simp at hp
  · intro x u v xS vS hs
    change (none : Option (Msg × Seen)) = some (xS, vS) at hs
    simp at hs
  · intro x s Pt d Ω P ω r u xS vS wF hx
    change (none : Option AuditStatement) = some s at hx
    simp at hx

end SamplingDeploy
