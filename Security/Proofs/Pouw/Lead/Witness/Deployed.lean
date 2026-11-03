import Definitions.Pouw.PearlC.Sha512
import Proofs.Pouw.Prob
import Proofs.Pouw.Lead.Compose.Deployed
import Proofs.Pouw.Lead.Statement.Gap
import Proofs.Pouw.Lead.Tail.Nontrivial
import Proofs.Pouw.Lead.Network.Term

/-!
# `G`, `G_tail`, `NetworkFwd` and `NetworkLlama` say something

Each guarantee's honest and nontrivial witnesses (the lock's `witnesses`), at one deployed instance: the spec's SHA-512
(`Pouw.PearlC.Sha512.digest`), Llama-3.2-1B's `2048 × 2048` linear at T 512 × 64 (`pt512x64 2048 2048`), zero data,
the deployed commitment of the forward's one call (`l1wCommit`), its form classifier (`l1wIsA`), the certifier's coin
stream (`certStream`) and a certifier running `certCode` with the coins as its entropy. Each statement reads only the
spec's definitions; the strategies and commitments the proofs build are theirs alone.

* `G_honest`, the honest witness of `G` and of `G_tail` (they have the same premises): every premise of both holds
  there, with the nominal W1 table and a set of strategies holding an honest one accepted at every entropy value
  (`Compose.honestL1w`, `Compose.certifierCoins`).
* `G_nontrivial`: at a commitment that hashes nothing (`blank`), a strategy that does nothing (`idle`) is accepted at
  every coin value, does no work, exchanges nothing and hashes nothing, while the honest work exceeds the gap
  (`Statement.nontrivial`), so with `O` every value `G`'s conclusion is false there.
* `G_tail_nontrivial`: the same at `pt512x64 4096 4096`, where the honest work exceeds the fixed gap and
  `etaLead < 1` (`Tail.nontrivialTail`): every entropy value is in `G_tail`'s event, of probability 1.
* `networkLlama_honest`, the honest witness of `NetworkFwd` and `NetworkLlama`: the forward of the honest strategy's
  call at the deployed instance, which exchanges nothing, is a forward of Llama-3.2-1B's calls (`LlamaForward`), meets
  `SerialForwards` and is accepted with one release bit.
* `networkFwd_nontrivial`, `networkLlama_nontrivial`: a forward of that call at a strategy that exchanges `2^20` bits
  (`loud`), every one counted, meets `SerialForwards` and `LlamaForward`, and its network term is past both bounds:
  only the certifier's refusal of a forward past its bits (`Forward.Accepted`) excludes it.
-/

namespace Pouw.SecurityProofs.Lead.Witness

open Pouw.Lead Pouw.Assumptions.Lead Pouw.Guarantees.Lead Pouw.Dimension.H100

/-- The spec's SHA-512 (`Pouw.PearlC.Sha512.digest`) on bytes, its digest's 64 bytes. -/
local notation "SHA512" => fun (x : Hash.Msg) (i : Fin 64) =>
  Nat.toUInt8 (List.getD (Pouw.PearlC.Sha512.digest (List.map UInt8.toNat x)) (Fin.val i) 0)

/-- Llama-3.2-1B's `2048 × 2048` linear at T 512 × 64. -/
local notation "L₀" => pt512x64 2048 2048

local notation "κ₀" => (fun _ _ _ => 0 : Fin (Pouw.Lead.Layout.m (pt512x64 2048 2048)) →
  Fin (Pouw.Lead.Layout.S (pt512x64 2048 2048)) → Fin (Pouw.Lead.Layout.n (pt512x64 2048 2048)) → ℤ)

/-- The deployed commitment of the forward's one call, at zero A, under SHA-512. -/
local notation "C₀" =>
  l1wCommit (pt512x64 2048 2048) ⟨[], [], 1, fun _ => [], 1⟩ ⟨0, [], []⟩ SHA512 (fun _ _ => 0)

local notation "isA₀" =>
  l1wIsA (pt512x64 2048 2048) ⟨[], [], 1, fun _ => [], 1⟩ ⟨0, [], []⟩ SHA512 (fun _ _ => 0)

/-- The certifier's code, its entropy the coins themselves. -/
local notation "cert₀" => fun (_ : Unit) => certCode (Equiv.refl (Coins (pt512x64 2048 2048)))

/-- The tail's point: a `4096 × 4096` call at T 512 × 64. -/
local notation "L₁" => pt512x64 4096 4096

local notation "κ₁" => (fun _ _ _ => 0 : Fin (Pouw.Lead.Layout.m (pt512x64 4096 4096)) →
  Fin (Pouw.Lead.Layout.S (pt512x64 4096 4096)) → Fin (Pouw.Lead.Layout.n (pt512x64 4096 4096)) → ℤ)

local notation "cert₁" => fun (_ : Unit) => certCode (Equiv.refl (Coins (pt512x64 4096 4096)))

/-! ## The strategies and the commitment the proofs build -/

/-- A strategy that does nothing but exchange `E` bits: no tables, no instructions, nothing hashed, the zero roots. -/
def loud (L : Pouw.Lead.Layout) (E : ℕ) : Strategy L :=
  ⟨fun _ _ => 0, fun _ _ => rfl, fun _ => 0, fun _ => ⟨[], [], 0, [], E, fun _ => 0⟩⟩

/-- A strategy that does nothing. -/
def idle (L : Pouw.Lead.Layout) : Strategy L := loud L 0

/-- A commitment format that hashes nothing, its roots zero, at the lead's words chunks. -/
def blank (L : Pouw.Lead.Layout) : Pouw.Lead.Commit L :=
  ⟨fun _ _ => [], fun _ _ _ => 0, [], fun _ _ => 0, wordChunks L⟩

/-- The coins at zero. -/
def zeroCoins (L : Pouw.Lead.Layout) : Coins L :=
  (fun _ _ => 0, fun _ _ => 0, fun _ _ => 0, fun _ _ _ _ => ⟨0, by positivity⟩)

theorem idle_work (L : Pouw.Lead.Layout) (pr : Prices100) (ω : Coins L) : (idle L).work pr ω = 0 := rfl

theorem idle_egress (L : Pouw.Lead.Layout) (ω : Coins L) : ((idle L).exec ω).egress = 0 := rfl

theorem loud_egress (L : Pouw.Lead.Layout) (E : ℕ) (ω : Coins L) : ((loud L E).exec ω).egress = E := rfl

theorem honest_egress (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (C : Pouw.Lead.Commit L)
    (isA : Hash.Msg → Bool) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) :
    ((Compose.honestStrategy L H C isA κ).exec ω).egress = 0 := rfl

/-- At an admitted call the blank commitment accepts the strategy that does nothing at every coin value. -/
theorem idle_accepted (L : Pouw.Lead.Layout) (hadm : L.Admits) (H : Hash.Msg → Hash.Digest)
    (κ : Fin L.m → Fin L.S → Fin L.n → ℤ) (ω : Coins L) : Accepted L H (blank L) κ (idle L) ω :=
  ⟨hadm, rfl, rfl⟩

/-- Nothing hashed, no collision. -/
theorem not_break_idle (L : Pouw.Lead.Layout) (H : Hash.Msg → Hash.Digest) (κ : Fin L.m → Fin L.S → Fin L.n → ℤ)
    (ω : Coins L) : ¬ Break L H (blank L) κ (idle L) ω := by
  rintro ⟨x, hx, -⟩
  simp [idle, loud, blank] at hx

/-! ## The deployed point's numbers -/

theorem admits_pt (n K : ℕ) (hn : n ≤ 2 ^ 20) (hK : 1 ≤ K) (hK' : K ≤ 2 ^ 16) : (pt512x64 n K).Admits :=
  (Statement.admits_of_caps 512 64 n K (by norm_num) hn hK hK').1

/-- The forward of the one call has its `148` words leaves, at most `2^64`. -/
theorem leaves_pt : CallForward.leaves ⟨0, [], []⟩ 1 (pt512x64 2048 2048) ≤ 2 ^ 64 := by
  rw [CallForward.leaves, wordsLeaves, Compose.wordsTiles_length _ Nat.one_pos]
  decide

/-- The deployed commitment's chunks are the lead's. -/
theorem chunks_pt (H : Hash.Msg → Hash.Digest) (A : Fin (pt512x64 2048 2048).rows → Fin (pt512x64 2048 2048).K → UInt8) :
    ChunksOf _ (l1wCommit (pt512x64 2048 2048) ⟨[], [], 1, fun _ => [], 1⟩ ⟨0, [], []⟩ H A) := by
  unfold ChunksOf l1wCommit
  dsimp only
  rw [wordsLeaves, Compose.wordsTiles_length _ Nat.one_pos]
  decide

theorem callBits_pt : (pt512x64 2048 2048).callBits = 8589934592 / 973078528 := by
  show ((32 : ℕ) : ℚ) * (((64 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ)) / ((973078528 : ℕ) : ℚ) = _
  norm_num

theorem servedRegs_pt : servedRegs (pt512x64 2048 2048) = 8388608 := by
  show ((64 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ) / 32 = _
  norm_num

theorem cap_pt : (pt512x64 2048 2048).cap = 32 := rfl

theorem wordsRegs_le_honestRegs (L : Pouw.Lead.Layout) (C : Pouw.Lead.Commit L) : wordsRegs L ≤ honestRegs L C := by
  have : (0 : ℚ) ≤ shaWordsRegs * wordsBlocks L C := by unfold shaWordsRegs wordsBlocks wordsRegs; positivity
  have : (0 : ℚ) ≤ shaARegs * aBlocks L := by unfold shaARegs aBlocks; positivity
  unfold honestRegs
  linarith

/-- An exchanged bit replaces at least 98 registers at the deployed point: `betaLead = honestRegs / 98,304`, and the
honest work is at least the words' `64 · 2048 · 74`. -/
theorem betaLead_pt_ge (C : Pouw.Lead.Commit (pt512x64 2048 2048)) : 98 ≤ betaLead (pt512x64 2048 2048) C := by
  have hw : ((64 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ) * ((74 : ℕ) : ℚ) ≤ honestRegs (pt512x64 2048 2048) C :=
    wordsRegs_le_honestRegs _ C
  have e : betaLead (pt512x64 2048 2048) C = honestRegs (pt512x64 2048 2048) C / 98304 := by
    show 256 / 3 * (32 * honestRegs (pt512x64 2048 2048) C /
      (((64 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ) * ((2048 : ℕ) : ℚ))) = _
    push_cast
    ring
  rw [e, le_div_iff₀ (by norm_num)]
  push_cast at hw
  linarith

/-- A forward of one call at the deployed point, at a commitment of the lead's chunks, is a forward of
Llama-3.2-1B's calls. -/
theorem llama_pt (C : Pouw.Lead.Commit (pt512x64 2048 2048)) (σ : Strategy (pt512x64 2048 2048))
    (ω : Coins (pt512x64 2048 2048)) (hC : ChunksOf _ C) (count release : ℕ) :
    LlamaForward ⟨[⟨pt512x64 2048 2048, C, σ, ω⟩], count, release⟩ := by
  intro c hc
  obtain rfl := List.mem_singleton.1 hc
  exact ⟨⟨512, 64, (2048, 2048), by simp [llamaCalls], Or.inl rfl⟩, hC⟩

/-! ## The witnesses -/

/-- **`G`'s and `G_tail`'s honest witness**: at the deployed instance every premise of `G` and `G_tail` holds, for a
set of strategies holding one accepted at every entropy value. -/
theorem G_honest :
    ∃ (Real : Strategy L₀ → Prop) (σ : {σ : Strategy L₀ // Real σ}),
      CertifierCoins () (Equiv.refl (Coins L₀)) cert₀ ∧ CommitL1w SHA512 C₀ isA₀ κ₀ ∧ H32 h100Nominal ∧
      W1 h100Nominal Real ∧ TileGrain SHA512 C₀ κ₀ (certStream L₀) Real ∧ OneInstructionMinimal Real ∧
      CardWithinAccelW1 Real Real ∧ ShaWrites isA₀ Real ∧ HonestHashRuled SHA512 C₀ κ₀ Real ∧
      OutsourceWithinEgressLead h100Nominal (betaLead L₀ C₀) SHA512 C₀ κ₀ Real ∧
      ∀ e : Coins L₀, Accepted L₀ SHA512 C₀ κ₀ σ.1 (released () cert₀ σ.1 e) := by
  obtain ⟨Real, σ, hC, h32, hW1, hTG, hOne, hCard, hSha, hHash, hOut, hσ, hacc⟩ :=
    Compose.honestL1w 512 64 2048 2048 (admits_pt 2048 2048 (by norm_num) (by norm_num) (by norm_num))
      (by norm_num) SHA512 κ₀ ⟨[], [], 1, fun _ => [], 1⟩ Nat.one_pos ⟨0, [], []⟩ leaves_pt (fun _ _ => 0)
  exact ⟨Real, ⟨σ, hσ⟩, Compose.certifierCoins _, hC, h32, hW1, hTG, hOne, hCard, hSha, hHash, hOut,
    fun _ => hacc _⟩

/-- **`G`'s nontrivial witness**: at a commitment of the lead's chunks, a strategy accepted at every coin value has
`G`'s conclusion false with `O` every entropy value: below `|O| · honest − |Ω| · gap`, and no collision. -/
theorem G_nontrivial :
    ∃ (C : {C : Pouw.Lead.Commit L₀ // ChunksOf L₀ C}) (σ : {σ : Strategy L₀ // ∀ ω, Accepted L₀ SHA512 C.1 κ₀ σ ω}),
      ¬ (32 * (((Finset.univ : Finset (Coins L₀)).card : ℚ) * honestRegs L₀ C.1 -
            (Fintype.card (Coins L₀) : ℚ) * gapRegs L₀ C.1) ≤
          ∑ e : Coins L₀, ((σ.1.work h100Nominal (released () cert₀ σ.1 e) : ℚ) +
            betaLead L₀ C.1 * (σ.1.exec (released () cert₀ σ.1 e)).egress) ∨
        ∃ e ∈ (Finset.univ : Finset (Coins L₀)), Break L₀ SHA512 C.1 κ₀ σ.1 (released () cert₀ σ.1 e)) := by
  have hadm := admits_pt 2048 2048 (by norm_num) (by norm_num) (by norm_num)
  have hgap : gapRegs L₀ (blank L₀) < honestRegs L₀ (blank L₀) :=
    (Statement.nontrivial 2048 2048 (by norm_num) le_rfl).1 (blank _) rfl
  have hcard : (0 : ℚ) < Fintype.card (Coins L₀) := by exact_mod_cast Fintype.card_pos_iff.2 ⟨zeroCoins _⟩
  refine ⟨⟨blank _, rfl⟩, ⟨idle _, idle_accepted _ hadm _ _⟩, ?_⟩
  rintro (h | ⟨e, -, he⟩)
  · simp only [idle_work, idle_egress, Nat.cast_zero, mul_zero, add_zero, Finset.sum_const_zero,
      Finset.card_univ] at h
    nlinarith
  · exact not_break_idle _ _ _ _ he

/-- **`G_tail`'s nontrivial witness**: at a commitment of the lead's chunks, a strategy accepted at every coin value has
`G_tail`'s conclusion false: every entropy value is in its event, and `etaLead < 1`. -/
theorem G_tail_nontrivial :
    ∃ (C : {C : Pouw.Lead.Commit L₁ // ChunksOf L₁ C}) (σ : {σ : Strategy L₁ // ∀ ω, Accepted L₁ SHA512 C.1 κ₁ σ ω}),
      ¬ Pouw.pr (fun e : Coins L₁ =>
          Accepted L₁ SHA512 C.1 κ₁ σ.1 (released () cert₁ σ.1 e) ∧
            ¬ Break L₁ SHA512 C.1 κ₁ σ.1 (released () cert₁ σ.1 e) ∧
            (σ.1.work h100Nominal (released () cert₁ σ.1 e) : ℚ) +
                betaLead L₁ C.1 * (σ.1.exec (released () cert₁ σ.1 e)).egress <
              32 * (honestRegs L₁ C.1 - gapFixed L₁ C.1)) ≤
        etaLead L₁ := by
  have hadm := admits_pt 4096 4096 (by norm_num) (by norm_num) (by norm_num)
  obtain ⟨hgap, heta⟩ := Tail.nontrivialTail_mem 4096 4096 le_rfl le_rfl L₁ (by simp)
  have hη : etaLead L₁ < 1 := heta (by decide)
  have hlt := hgap (blank _) rfl
  have : Nonempty (Coins L₁) := ⟨zeroCoins _⟩
  refine ⟨⟨blank _, rfl⟩, ⟨idle _, idle_accepted _ hadm _ _⟩, fun h => absurd (h.trans_lt hη) ?_⟩
  rw [Pouw.pr_eq_one]
  · exact lt_irrefl 1
  · intro e
    refine ⟨idle_accepted _ hadm _ _ _, not_break_idle _ _ _ _, ?_⟩
    simp only [idle_work, idle_egress, Nat.cast_zero, mul_zero, add_zero]
    linarith

/-- **`NetworkFwd`'s and `NetworkLlama`'s honest witness**: the forward of the honest strategy's call at the deployed
instance, accepted at every coin value, is a forward of Llama-3.2-1B's calls that meets `SerialForwards` and that the
network certifier accepts, with one release bit. -/
theorem networkLlama_honest :
    ∃ (σ : {σ : Strategy L₀ // ∀ ω, Accepted L₀ SHA512 C₀ κ₀ σ ω}) (ω : Coins L₀),
      LlamaForward ⟨[⟨L₀, C₀, σ.1, ω⟩], 0, 1⟩ ∧ SerialForwards ⟨[⟨L₀, C₀, σ.1, ω⟩], 0, 1⟩ ∧
        Forward.Accepted ⟨[⟨L₀, C₀, σ.1, ω⟩], 0, 1⟩ := by
  obtain ⟨-, -, -, -, -, -, -, -, -, hacc⟩ := Compose.honestAt L₀ SHA512 κ₀ (certStream L₀) C₀ isA₀
    (admits_pt 2048 2048 (by norm_num) (by norm_num) (by norm_num))
  refine ⟨⟨_, hacc⟩, zeroCoins _, llama_pt C₀ _ _ (chunks_pt SHA512 (fun _ _ => 0)) 0 1, ?_, ?_⟩
  · simp [SerialForwards, FwdCall.egress, honest_egress]
  · norm_num [Forward.Accepted, Forward.bits, callBits_pt]

/-- **`NetworkFwd`'s nontrivial witness**: a forward under `SerialForwards` whose network term is past its calls'
largest `β` times its bits over 32. -/
theorem networkFwd_nontrivial :
    ∃ F : {F : Forward // SerialForwards F}, ¬ F.1.networkRegs ≤ F.1.betaMax * F.1.bits / 32 := by
  refine ⟨⟨⟨[⟨L₀, blank L₀, loud L₀ (2 ^ 20), zeroCoins L₀⟩], 2 ^ 20, 0⟩,
    by simp [SerialForwards, FwdCall.egress, loud_egress]⟩, ?_⟩
  have hβ := betaLead_pt_ge (blank L₀)
  simp only [Forward.networkRegs, Forward.betaMax, Forward.bits, List.map_cons, List.map_nil, List.sum_cons,
    List.sum_nil, List.foldr_cons, List.foldr_nil, add_zero, FwdCall.egress, loud_egress, callBits_pt,
    max_eq_left (Network.betaLead_nonneg (pt512x64 2048 2048) (blank _))]
  push_cast
  rw [not_le]
  linarith

/-- **`NetworkLlama`'s nontrivial witness**: a forward of Llama-3.2-1B's calls under `SerialForwards` whose network term
is past 0.0363% of its calls' served stock at cap 32. -/
theorem networkLlama_nontrivial :
    ∃ F : {F : Forward // SerialForwards F ∧ LlamaForward F},
      ¬ F.1.networkRegs ≤ 363 / 1000000 * (F.1.calls.map fun c => (c.L.cap / 32 : ℚ) * servedRegs c.L).sum := by
  refine ⟨⟨⟨[⟨L₀, blank L₀, loud L₀ (2 ^ 20), zeroCoins L₀⟩], 2 ^ 20, 0⟩,
    by simp [SerialForwards, FwdCall.egress, loud_egress],
    llama_pt (blank L₀) (loud L₀ (2 ^ 20)) (zeroCoins L₀) rfl _ _⟩, ?_⟩
  have hβ := betaLead_pt_ge (blank L₀)
  simp only [Forward.networkRegs, List.map_cons, List.map_nil, List.sum_cons, List.sum_nil, add_zero,
    FwdCall.egress, loud_egress, servedRegs_pt, cap_pt]
  push_cast
  rw [not_le]
  linarith

end Pouw.SecurityProofs.Lead.Witness
