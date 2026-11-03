import Definitions.Pouw.PearlC.Sha512
import Proofs.Pouw.Lead.Bridge.Game
import Proofs.Pouw.Lead.Tail.G
import Proofs.Pouw.Lead.Witness.Deployed

/-!
# `GammaFromTail` and `EndToEndFromTail` say something

Each guarantee's honest and nontrivial witnesses (the lock's `witnesses`), at the tail's point, a `4096 × 4096` call
at T 512 × 64 (`pt512x64 4096 4096`), where `NontrivialTail` holds: the spec's SHA-512 (`Pouw.PearlC.Sha512.digest`),
zero data at every unit, the deployed commitment of the forward's one call (`l1wCommit`), its form classifier
(`l1wIsA`), the certifier's coin stream (`certStream`) and a game of one unit (`N₀ = 1`) over a one-point oracle. Each
statement reads only the spec's definitions; the strategies, commitments and adversaries the proofs build are theirs.

* `fixedGapBelow`: the fixed gap is below the honest work (`Game.FixedGapBelow`) at the three points, for every call of
  `n, K ≥ 4,096` at a commitment of the lead's chunks, from `NontrivialTail`.
* `GammaFromTail_honest`: every premise of `GammaFromTail` holds there, `G_tail` (`Tail.G_tail`) and the fixed gap
  among them, with the nominal W1 table and a set of strategies holding an honest one accepted at every coin value
  (`Compose.honestL1w`).
* `EndToEndFromTail_honest`: the same, and a workload of one unit in the game's domain with an adversary playing the
  honest strategy (`player`) within a budget, whose unit is correct at every salt, so the audit accepts it.
* `GammaFromTail_nontrivial`, `EndToEndFromTail_nontrivial`: at a commitment that hashes nothing (`Witness.blank`), an
  adversary playing a strategy that does nothing (`idleProg`), accepted at every coin value, costs nothing and
  hashes nothing, while the honest work exceeds the fixed gap: with budget 0 every run wins and the audit accepts
  every run, of probability 1, above `etaLead` and above `1/2 + etaLead` at `ε_s = 1/2` and one sample.
-/

namespace Pouw.SecurityProofs.Lead.Bridge

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Guarantees.Lead Pouw.Dimension.H100

/-- The spec's SHA-512 (`Pouw.PearlC.Sha512.digest`) on bytes, its digest's 64 bytes. -/
local notation "SHA512" => fun (x : Hash.Msg) (i : Fin 64) =>
  Nat.toUInt8 (List.getD (Pouw.PearlC.Sha512.digest (List.map UInt8.toNat x)) (Fin.val i) 0)

/-- The tail's point: a `4096 × 4096` call at T 512 × 64. -/
local notation "L₁" => pt512x64 4096 4096

local notation "κ₁" => (fun _ _ _ => 0 : Fin (Pouw.Lead.Layout.m (pt512x64 4096 4096)) →
  Fin (Pouw.Lead.Layout.S (pt512x64 4096 4096)) → Fin (Pouw.Lead.Layout.n (pt512x64 4096 4096)) → ℤ)

/-- Every unit's data: zero. -/
local notation "κU" => (fun _ _ _ _ _ => 0 : Pouw.Workload → ℕ → Fin (Pouw.Lead.Layout.m (pt512x64 4096 4096)) →
  Fin (Pouw.Lead.Layout.S (pt512x64 4096 4096)) → Fin (Pouw.Lead.Layout.n (pt512x64 4096 4096)) → ℤ)

/-- The deployed commitment of the forward's one call, at zero A, under SHA-512. -/
local notation "C₁" =>
  l1wCommit (pt512x64 4096 4096) ⟨[], [], 1, fun _ => [], 1⟩ ⟨0, [], []⟩ SHA512 (fun _ _ => 0)

local notation "isA₁" =>
  l1wIsA (pt512x64 4096 4096) ⟨[], [], 1, fun _ => [], 1⟩ ⟨0, [], []⟩ SHA512 (fun _ _ => 0)

/-- A workload of one unit, the `4096 × 4096` linear at 64 rows, with zero weights. -/
local notation "U₁" =>
  Pouw.Workload.mk (Pouw.Layout.mk 1 (fun _ => Pouw.Shape.mk 64 4096 4096) (fun _ => 0)) (fun _ _ _ => 0)

/-! ## The fixed gap at the three points -/

/-- **The fixed gap is below the honest work at the three points** (`Game.FixedGapBelow`), for every call of
`n, K ≥ 4,096` at a commitment of the lead's chunks: `NontrivialTail`'s first half. -/
theorem fixedGapBelow (n K : ℕ) (hn : 4096 ≤ n) (hK : 4096 ≤ K) (L : Pouw.Lead.Layout)
    (hL : L ∈ [pt512x64 n K, pt512x512 n K, pt8192 n K]) (C : Pouw.Lead.Commit L) (hC : ChunksOf L C) :
    Pouw.Lead.Game.FixedGapBelow L C :=
  (Tail.nontrivialTail_mem n K hn hK L hL).1 C hC

/-! ## The adversaries the witnesses build -/

/-- The program whose every unit plays `σ`. -/
def constProg {L : Pouw.Lead.Layout} (N₀ : ℕ) {can : Strategy L → Prop} (σ : Strategy L) (hσ : can σ) :
    Pouw.Lead.Game.LeadProg L N₀ can :=
  ⟨fun _ _ => σ, fun _ _ => hσ⟩

/-- The program whose every unit does nothing, accepted at the blank commitment at every coin value. -/
def idleProg (L : Pouw.Lead.Layout) (N₀ : ℕ) (hadm : L.Admits) (sha : Hash.Msg → Hash.Digest)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) :
    Pouw.Lead.Game.LeadProg L N₀ (fun σ => ∀ ω, Accepted L sha (Witness.blank L) κ σ ω) :=
  constProg N₀ (Witness.idle L) (Witness.idle_accepted L hadm sha κ)

/-- The game's adversary that makes no oracle call and plays `p`. -/
def player (Q R : Type) {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (sha : Hash.Msg → Hash.Digest)
    (C : Pouw.Lead.Commit L) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop}
    (p : Pouw.Lead.Game.LeadProg L N₀ can) : Adversary (Pouw.Lead.Game.costModel L N₀ Q R prc sha C κ can) :=
  ⟨QTree.done [], p, fun _ => 0⟩

/-- The adversary's run at a salt costs what its units cost, at least 0. -/
theorem cost_nonneg {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (C : Pouw.Lead.Commit L)
    {can : Strategy L → Prop} (p : Pouw.Lead.Game.LeadProg L N₀ can) (U : Pouw.Workload)
    (s : Pouw.Lead.Game.Salt L N₀) :
    0 ≤ ∑ u ∈ Finset.range U.layout.N, (Pouw.Lead.Game.unitCost prc C p s u : ℝ) :=
  Finset.sum_nonneg fun u _ => by exact_mod_cast unitCost_nonneg prc C p s u

/-- `player` is within any budget its every salt's cost is within, and makes no oracle call. -/
theorem player_bounded (Q R : Type) {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (sha : Hash.Msg → Hash.Digest)
    (C : Pouw.Lead.Commit L) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop}
    (p : Pouw.Lead.Game.LeadProg L N₀ can) (U : Pouw.Workload) (B : ℝ)
    (hB : ∀ s, ∑ u ∈ Finset.range U.layout.N, (Pouw.Lead.Game.unitCost prc C p s u : ℝ) ≤ B) :
    (player Q R prc sha C κ p).Bounded U B 0 :=
  fun _ s => ⟨hB s, Nat.le_zero.2 rfl⟩

/-- `player` is within the sum over salts of its cost. -/
theorem player_bounded_total (Q R : Type) {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop}
    (p : Pouw.Lead.Game.LeadProg L N₀ can) (U : Pouw.Workload) :
    (player Q R prc sha C κ p).Bounded U
      (∑ s, ∑ u ∈ Finset.range U.layout.N, (Pouw.Lead.Game.unitCost prc C p s u : ℝ)) 0 :=
  player_bounded Q R prc sha C κ p U _ fun s =>
    Finset.single_le_sum (f := fun s => ∑ u ∈ Finset.range U.layout.N, (Pouw.Lead.Game.unitCost prc C p s u : ℝ))
      (fun s _ => cost_nonneg prc C p U s) (Finset.mem_univ s)

/-- At an accepted call `player`'s unit is correct: its commitment is the ideal one. -/
theorem player_correct (Q R : Type) {L : Pouw.Lead.Layout} {N₀ : ℕ} (prc : Prices100) (sha : Hash.Msg → Hash.Digest)
    (C : Pouw.Lead.Commit L) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ) {can : Strategy L → Prop}
    (p : Pouw.Lead.Game.LeadProg L N₀ can) (U : Pouw.Workload) (H : Q → R) (s : Pouw.Lead.Game.Salt L N₀) (u : ℕ)
    (hu : u < U.layout.N) (h : u < N₀) (hacc : Accepted L sha C (κ U u) (p.play s ⟨u, h⟩) (s ⟨u, h⟩)) :
    u ∈ correctSet (Pouw.Lead.Game.protocol L N₀ Q R C) U H s ((player Q R prc sha C κ p).transcript U H s) := by
  classical
  show u ∈ correctSet (Pouw.Lead.Game.protocol L N₀ Q R C) U H s (Pouw.Lead.Game.unitCommit sha C κ p U s)
  unfold correctSet
  rw [Finset.mem_filter]
  refine ⟨Finset.mem_range.2 hu, ⟨0, fun _ => 0, mm (U.layout.shape u).k 0 (U.weight (U.layout.wid u))⟩, ?_, ?_⟩
  · rw [Pouw.Lead.Game.unitCommit, dif_pos h, if_pos hacc]
  · unfold Correct CheckedOK ActOK OutOK
    refine ⟨⟨fun i _ l _ => ?_, fun x hx => absurd hx (Finset.notMem_empty x)⟩, fun _ _ _ _ => rfl⟩
    unfold InInt7
    norm_num

/-- A unit that does nothing costs nothing. -/
theorem idle_unitCost (L : Pouw.Lead.Layout) (N₀ : ℕ) (hadm : L.Admits) (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (s : Pouw.Lead.Game.Salt L N₀) (u : ℕ) :
    Pouw.Lead.Game.unitCost prc (Witness.blank L) (idleProg L N₀ hadm sha κ) s u = 0 := by
  unfold Pouw.Lead.Game.unitCost
  split <;> simp [Pouw.Lead.Game.LeadProg.play, idleProg, constProg, Witness.idle_work, Witness.idle_egress]

/-- The idle program's adversary is within budget 0. -/
theorem idle_bounded (Q R : Type) (L : Pouw.Lead.Layout) (N₀ : ℕ) (hadm : L.Admits) (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ)
    (κ₀ : Fin L.m → Fin L.S → Fin L.n → ℤ) (U : Pouw.Workload) :
    (player Q R prc sha (Witness.blank L) κ (idleProg L N₀ hadm sha κ₀)).Bounded U 0 0 :=
  player_bounded Q R prc sha _ κ _ U 0 fun s => by simp [idle_unitCost]

/-- The idle program at the blank commitment hashes nothing, so no run breaks. -/
theorem idle_not_breaks (Q R : Type) (L : Pouw.Lead.Layout) (N₀ : ℕ) (hadm : L.Admits) (prc : Prices100)
    (sha : Hash.Msg → Hash.Digest) (κ : Pouw.Workload → ℕ → Fin L.m → Fin L.S → Fin L.n → ℤ)
    (κ₀ : Fin L.m → Fin L.S → Fin L.n → ℤ) (U : Pouw.Workload) (H : Q → R) (s : Pouw.Lead.Game.Salt L N₀) :
    ¬ (player Q R prc sha (Witness.blank L) κ (idleProg L N₀ hadm sha κ₀)).Breaks U H s := by
  rintro ⟨y, hy, -⟩
  simp [player, Adversary.hashed, Pouw.Lead.Game.costModel, Pouw.Lead.Game.unitHashed, Pouw.Lead.Game.LeadProg.play,
    idleProg, constProg, Witness.idle, Witness.loud, Witness.blank] at hy

/-! ## The deployed point's numbers -/

/-- The forward of the one call has its words leaves at most `2^64`. -/
theorem leaves_pt₁ : CallForward.leaves ⟨0, [], []⟩ 1 (pt512x64 4096 4096) ≤ 2 ^ 64 := by
  rw [CallForward.leaves, wordsLeaves, Compose.wordsTiles_length _ Nat.one_pos]
  decide

/-- The deployed commitment's chunks are the lead's. -/
theorem chunks_pt₁ (H : Hash.Msg → Hash.Digest)
    (A : Fin (pt512x64 4096 4096).rows → Fin (pt512x64 4096 4096).K → UInt8) :
    ChunksOf _ (l1wCommit (pt512x64 4096 4096) ⟨[], [], 1, fun _ => [], 1⟩ ⟨0, [], []⟩ H A) := by
  unfold ChunksOf l1wCommit
  dsimp only
  rw [wordsLeaves, Compose.wordsTiles_length _ Nat.one_pos]
  decide

theorem admits_pt₁ : (pt512x64 4096 4096).Admits :=
  Witness.admits_pt 4096 4096 (by norm_num) (by norm_num) (by norm_num)

/-- The honest work at the tail's point is at least one register, so `W_ref` is positive. -/
theorem wref_pos (C : Pouw.Lead.Commit (pt512x64 4096 4096)) : 0 < ⌊32 * honestRegs (pt512x64 4096 4096) C⌋₊ := by
  have hw : wordsRegs (pt512x64 4096 4096) ≤ honestRegs (pt512x64 4096 4096) C := Witness.wordsRegs_le_honestRegs _ C
  have hS : 1 ≤ (pt512x64 4096 4096).S := by decide
  have hw' : (1 : ℚ) ≤ wordsRegs (pt512x64 4096 4096) := by
    show (1 : ℚ) ≤ ((64 : ℕ) : ℚ) * ((4096 : ℕ) : ℚ) * (((pt512x64 4096 4096).S : ℕ) : ℚ)
    have : (1 : ℚ) ≤ (((pt512x64 4096 4096).S : ℕ) : ℚ) := by exact_mod_cast hS
    push_cast
    linarith
  exact Nat.floor_pos.2 (by linarith)

/-- The one unit's workload is in the game's domain of one unit. -/
theorem inDomain_U₁ : (U₁).InDomain (Pouw.Lead.Game.domain 1) :=
  ⟨by norm_num, le_rfl, fun _ _ _ _ _ _ => ⟨by norm_num, by norm_num⟩⟩

/-! ## The witnesses -/

/-- **`GammaFromTail`'s honest witness**: at the tail's point every premise of `GammaFromTail` holds, `G_tail` and the
fixed gap below the honest work among them, for a set of strategies holding one accepted at every coin value. -/
theorem GammaFromTail_honest :
    ∃ (real : Strategy L₁ → Prop) (σ : {σ : Strategy L₁ // real σ}),
      G_tail ∧ Pouw.Lead.Game.FixedGapBelow L₁ C₁ ∧ CommitL1w SHA512 C₁ isA₁ κ₁ ∧
      TileGrain SHA512 C₁ κ₁ (certStream L₁) real ∧ HonestHashRuled SHA512 C₁ κ₁ real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead L₁ C₁) SHA512 C₁ κ₁ real ∧ H32 h100Nominal ∧
      W1 h100Nominal real ∧ OneInstructionMinimal real ∧ CardWithinAccelW1 real real ∧ ShaWrites isA₁ real ∧
      ∀ ω, Accepted L₁ SHA512 C₁ κ₁ σ.1 ω := by
  obtain ⟨real, σ, hC, h32, hW1, hTG, hOne, hCard, hSha, hHash, hOut, hσ, hacc⟩ :=
    Compose.honestL1w 512 64 4096 4096 admits_pt₁ (by norm_num) SHA512 κ₁ ⟨[], [], 1, fun _ => [], 1⟩ Nat.one_pos
      ⟨0, [], []⟩ leaves_pt₁ (fun _ _ => 0)
  exact ⟨real, ⟨σ, hσ⟩, Tail.G_tail,
    fixedGapBelow 4096 4096 le_rfl le_rfl _ (by simp) _ (chunks_pt₁ _ _), hC, hTG, hHash, hOut, h32, hW1, hOne,
    hCard, hSha, hacc⟩

/-- **`EndToEndFromTail`'s honest witness**: every premise of `EndToEndFromTail` holds at the tail's point, for a
workload of one unit in the game's domain and an adversary playing the honest strategy within a budget, whose unit is
correct at every salt. -/
theorem EndToEndFromTail_honest :
    ∃ (real : Strategy L₁ → Prop) (σ : {σ : Strategy L₁ // real σ})
      (A : Adversary (Pouw.Lead.Game.costModel L₁ 1 Unit Unit h100Nominal SHA512 C₁ κU real)) (B : ℝ),
      G_tail ∧ Pouw.Lead.Game.FixedGapBelow L₁ C₁ ∧ CommitL1w SHA512 C₁ isA₁ κ₁ ∧
      TileGrain SHA512 C₁ κ₁ (certStream L₁) real ∧ HonestHashRuled SHA512 C₁ κ₁ real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead L₁ C₁) SHA512 C₁ κ₁ real ∧ H32 h100Nominal ∧
      W1 h100Nominal real ∧ OneInstructionMinimal real ∧ CardWithinAccelW1 real real ∧ ShaWrites isA₁ real ∧
      (U₁).InDomain (Pouw.Lead.Game.domain 1) ∧ 0 ≤ B ∧ A.Bounded U₁ B 0 ∧
      ∀ (H : Unit → Unit) (s : Pouw.Lead.Game.Salt L₁ 1) (ω : Coins L₁), Accepted L₁ SHA512 C₁ κ₁ σ.1 ω ∧
        0 ∈ correctSet (Pouw.Lead.Game.protocol L₁ 1 Unit Unit C₁) U₁ H s (A.transcript U₁ H s) := by
  obtain ⟨real, σ, hC, h32, hW1, hTG, hOne, hCard, hSha, hHash, hOut, hσ, hacc⟩ :=
    Compose.honestL1w 512 64 4096 4096 admits_pt₁ (by norm_num) SHA512 κ₁ ⟨[], [], 1, fun _ => [], 1⟩ Nat.one_pos
      ⟨0, [], []⟩ leaves_pt₁ (fun _ _ => 0)
  exact ⟨real, ⟨σ, hσ⟩, player Unit Unit h100Nominal SHA512 C₁ κU (constProg 1 σ hσ),
    ∑ s, ∑ u ∈ Finset.range (U₁).layout.N, (Pouw.Lead.Game.unitCost h100Nominal C₁ (constProg 1 σ hσ) s u : ℝ),
    Tail.G_tail,
    fixedGapBelow 4096 4096 le_rfl le_rfl _ (by simp) _ (chunks_pt₁ _ _), hC, hTG, hHash, hOut, h32, hW1, hOne,
    hCard, hSha, inDomain_U₁, Finset.sum_nonneg fun s _ => cost_nonneg h100Nominal C₁ (constProg 1 σ hσ) U₁ s,
    player_bounded_total Unit Unit h100Nominal SHA512 C₁ κU (constProg 1 σ hσ) U₁,
    fun H s ω => ⟨hacc ω, player_correct Unit Unit h100Nominal SHA512 C₁ κU (constProg 1 σ hσ) U₁ H s 0 Nat.one_pos
      Nat.one_pos (hacc _)⟩⟩

/-- **`GammaFromTail`'s nontrivial witness**: at a commitment of the lead's chunks, with the strategies accepted at
every coin value, `G_γ` fails: the adversary playing a strategy that does nothing wins every run within budget 0 and
hashes no collision, while `etaLead < 1`. -/
theorem GammaFromTail_nontrivial :
    ∃ C : {C : Pouw.Lead.Commit L₁ // ChunksOf L₁ C},
      ¬ Pouw.Gγ (Pouw.Lead.Game.costModel L₁ 1 Unit Unit h100Nominal SHA512 C.1 κU
          fun σ => ∀ ω, Accepted L₁ SHA512 C.1 κ₁ σ ω)
        (Pouw.Lead.Game.protocol L₁ 1 Unit Unit C.1) (Pouw.Lead.Game.domain 1) (Pouw.Lead.Game.gamma L₁ C.1)
        (Pouw.Lead.Game.eta L₁) := by
  obtain ⟨-, heta⟩ := Tail.nontrivialTail_mem 4096 4096 le_rfl le_rfl L₁ (by simp)
  have hη : etaLead L₁ < 1 := heta (by decide)
  refine ⟨⟨Witness.blank _, rfl⟩, fun hG => ?_⟩
  have h1 := hG U₁ inDomain_U₁ 0 le_rfl 0 (by norm_num) _
    (idle_bounded Unit Unit L₁ 1 admits_pt₁ h100Nominal SHA512 κU κ₁ U₁)
  have : Nonempty ((Unit → Unit) × Pouw.Lead.Game.Salt L₁ 1) := ⟨(fun _ => (), fun _ => Witness.zeroCoins _)⟩
  rw [Pouw.pr_eq_one] at h1
  · have : Pouw.Lead.Game.eta L₁ 0 (U₁).layout.N = etaLead L₁ := by
      show ((1 : ℕ) : ℝ) * etaLead L₁ = etaLead L₁
      rw [Nat.cast_one, one_mul]
    linarith
  · intro x
    refine ⟨?_, idle_not_breaks Unit Unit L₁ 1 admits_pt₁ h100Nominal SHA512 κU κ₁ U₁ x.1 x.2⟩
    have h0 := player_correct Unit Unit h100Nominal SHA512 (Witness.blank L₁) κU (idleProg L₁ 1 admits_pt₁ SHA512 κ₁)
      U₁ x.1 x.2 0 Nat.one_pos Nat.one_pos (Witness.idle_accepted L₁ admits_pt₁ SHA512 κ₁ _)
    unfold Wins
    rw [zero_div]
    exact lt_of_lt_of_le (by exact_mod_cast wref_pos (Witness.blank L₁))
      (Finset.single_le_sum (fun _ _ => Nat.cast_nonneg _) h0)

/-- **`EndToEndFromTail`'s nontrivial witness**: at a commitment of the lead's chunks, with the strategies accepted at
every coin value, the audit with one sample at `ε_s = 1/2` accepts every run of the adversary playing a strategy that
does nothing within budget 0, which hashes no collision: probability 1, past `1/2 + etaLead`. -/
theorem EndToEndFromTail_nontrivial :
    ∃ (C : {C : Pouw.Lead.Commit L₁ // ChunksOf L₁ C})
      (A : {A : Adversary (Pouw.Lead.Game.costModel L₁ 1 Unit Unit h100Nominal SHA512 C.1 κU
          fun σ => ∀ ω, Accepted L₁ SHA512 C.1 κ₁ σ ω) // A.Bounded U₁ 0 0}),
      ¬ Pouw.pr (fun x : (Unit → Unit) × Pouw.Lead.Game.Salt L₁ 1 ×
            (Fin 1 → Ticket ((Pouw.Lead.Game.protocol L₁ 1 Unit Unit C.1).Wref (U₁).layout) (U₁).layout.N) =>
          AuditAccepts (correctSet (Pouw.Lead.Game.protocol L₁ 1 Unit Unit C.1) U₁ x.1 x.2.1
            (A.1.transcript U₁ x.1 x.2.1)) x.2.2 ∧
          (0 : ℝ) < (1 - Pouw.Lead.Game.gamma L₁ C.1) * (1 - 1 / 2) *
            totalWref (Pouw.Lead.Game.protocol L₁ 1 Unit Unit C.1) (U₁).layout ∧
          ¬ A.1.Breaks U₁ x.1 x.2.1)
        ≤ (1 - 1 / 2) ^ 1 + (U₁).layout.N * etaLead L₁ := by
  obtain ⟨hgap, -⟩ := Tail.nontrivialTail_mem 4096 4096 le_rfl le_rfl L₁ (by simp)
  have hgap' : gapFixed L₁ (Witness.blank L₁) < honestRegs L₁ (Witness.blank L₁) := hgap (Witness.blank L₁) rfl
  have hη : etaLead L₁ ≤ 1 / 2 ^ 34 := Tail.tailAt_mem 1 4096 4096 le_rfl le_rfl L₁ (by simp) (by decide)
  have hW := wref_pos (Witness.blank L₁)
  refine ⟨⟨Witness.blank _, rfl⟩, ⟨_, idle_bounded Unit Unit L₁ 1 admits_pt₁ h100Nominal SHA512 κU κ₁ U₁⟩,
    fun h => ?_⟩
  have hne : Nonempty ((Unit → Unit) × Pouw.Lead.Game.Salt L₁ 1 ×
      (Fin 1 → Ticket ((Pouw.Lead.Game.protocol L₁ 1 Unit Unit (Witness.blank L₁)).Wref (U₁).layout) (U₁).layout.N)) :=
    ⟨(fun _ => (), fun _ => Witness.zeroCoins _, fun _ => ⟨⟨0, Nat.one_pos⟩, ⟨0, hW⟩⟩)⟩
  rw [Pouw.pr_eq_one] at h
  · have hN : (((U₁).layout.N : ℕ) : ℝ) = 1 := Nat.cast_one
    rw [hN, one_mul] at h
    have : (1 / 2 ^ 34 : ℝ) < 1 / 2 := by norm_num
    norm_num at h
    linarith
  · intro x
    have h0 := player_correct Unit Unit h100Nominal SHA512 (Witness.blank L₁) κU (idleProg L₁ 1 admits_pt₁ SHA512 κ₁)
      U₁ x.1 x.2.1 0 Nat.one_pos Nat.one_pos (Witness.idle_accepted L₁ admits_pt₁ SHA512 κ₁ _)
    refine ⟨?_, ?_, idle_not_breaks Unit Unit L₁ 1 admits_pt₁ h100Nominal SHA512 κU κ₁ U₁ x.1 x.2.1⟩
    · intro i
      rw [Nat.lt_one_iff.1 (x.2.2 i).1.isLt]
      exact h0
    · have hpos : (0 : ℝ) <
          ((32 * (honestRegs L₁ (Witness.blank L₁) - gapFixed L₁ (Witness.blank L₁)) : ℚ) : ℝ) := by
        have : (0 : ℚ) < 32 * (honestRegs L₁ (Witness.blank L₁) - gapFixed L₁ (Witness.blank L₁)) := by linarith
        exact_mod_cast this
      have hWr : (0 : ℝ) < ((⌊32 * honestRegs L₁ (Witness.blank L₁)⌋₊ : ℕ) : ℝ) := by exact_mod_cast hW
      have hγ : 0 < 1 - Pouw.Lead.Game.gamma L₁ (Witness.blank L₁) := by
        unfold Pouw.Lead.Game.gamma
        rw [sub_sub_cancel]
        exact div_pos hpos hWr
      have htot : totalWref (Pouw.Lead.Game.protocol L₁ 1 Unit Unit (Witness.blank L₁)) (U₁).layout =
          ((⌊32 * honestRegs L₁ (Witness.blank L₁)⌋₊ : ℕ) : ℝ) := by
        show ∑ u ∈ Finset.range 1, ((⌊32 * honestRegs L₁ (Witness.blank L₁)⌋₊ : ℕ) : ℝ) = _
        rw [Finset.sum_range_one]
      show (0 : ℝ) < (1 - Pouw.Lead.Game.gamma L₁ (Witness.blank L₁)) * (1 - 1 / 2) *
        totalWref (Pouw.Lead.Game.protocol L₁ 1 Unit Unit (Witness.blank L₁)) (U₁).layout
      rw [htot]
      exact mul_pos (mul_pos hγ (by norm_num)) hWr

end Pouw.SecurityProofs.Lead.Bridge
