import Proofs.Pouw.Lead.Compose.HashCompose
import Definitions.Pouw.DitherInstance

/-!
# The call model and the period's inventory with the hosts counted

* **The atomic call, T1's layer** (`AtomicCall`; `atomicCall_closes`): T1's adversary is a program on the card alone.
  `AtomicCall` makes the call one program from the call's free words, so `root(A)` is not free. With `ShaWrites` over
  every digest of the call, `root(A)`'s included, the card's cost is at least `c·(|T| + R_A + R_W)`: the pre-noise
  hashing of `A` is charged to the card, beside the targets. Without it the hash's credit is zero
  (`shaWrites_false_of_free`, under a free set holding `root(A)`).
* **The cards' inventory** (`Run`, `costOn`, `PeriodInventory`; `costOn_split`, `egress_charged`,
  `preNoiseHash_credited`): instructions on devices, cost counted on the cards `I`. The cards are charged the bound
  less what devices outside `I` ran (`egress_charged`); for devices outside the network certifier's boundary, bounding
  that is the egress layer's. `PeriodInventory`, every device of both runs in `I`, makes it zero. Without it,
  outsourced hashing goes uncounted on `I` (`costOn_outside`, `outsourced_uncredited`); a toy instance meets every
  hypothesis and attains the bound (`toy_attained`).
* **The hosts inside T1** (`PeriodInventoryT1`, `HostsWithinRow`, `hostsW1`; `hostsCounted_closes`,
  `hostsCounted_call`, `hostsCounted_capacity`, `hostRow_residual`, `routeH_toy`): inside the network certifier's
  boundary every device is a card or one of its hosts, and the hosts are counted at θ's rows (`HostRow`, as
  `HostsCounted` counts them), so a host's hashing is credited as hashing. The credited W1 is at most the cards' cost,
  the hosts' counted W1 and what came from outside the boundary, and with the other work on the cards it fits the
  period's capacity with the hosts at their rows (`hostsCounted_capacity`). A row below the hosts' best leaves
  `Σ_r Tp·n·w_r·(best_r − rate_r)⁺`, resource by resource. Both premises read the headline's own terms: the hosts'
  counted W1 is within `Period.cap` at the host resources of the `G` that `OnCard` describes (`hostsW1_le_hostsCounted`,
  `hostsW1_le_cap`), and `Period.Fits` at hosts counted at the row gives `HostsWithinRow` (`hostsWithinRow_of_fits`,
  `hostsCounted_closes_of_fits`; `hostsCounted_others`: the row's own hosts are counted).
-/

namespace Pouw.SecurityProofs.Lead.Compose

open Finset Pouw.Dimension Pouw.Dimension.H100

/-! ## 6. The call model: `AtomicCall` and the cards' `PeriodInventory` (T1's) -/

section Call

variable {G Dev : Type}

/-- **`AtomicCall pr Fcall L P`** (named premise): the card's call is one program. Its inputs `L` and its
operand words are the call's free words `Fcall` (the constants, the call's input words and its raw coin words), so no
function of the call's input computed before the noise (such as `root(A)`) enters free; the grid it is judged on
ranges over the call's input. -/
def AtomicCall (pr : Prices100) (Fcall : Set (G → ℚ)) (L : List (G → ℚ)) (P : List (Op100 G)) : Prop :=
  (∀ f ∈ L, f ∈ Fcall) ∧ ∀ o ∈ P, o.WFw pr Fcall

/-- **What the atomic call alone closes** (given `AtomicCall` and `ShaWrites`; T1's layer, the card alone): on
the call grid `S`, with `ShaWrites` over every digest of the call (`root(A)`'s `DigA` and the committed rows' `DigW`,
`R_A + R_W` registers jointly), the card's program, admissible, writing a register in `Rel` to each separated target
and outputting every digest, costs at least `c·(|T| + R_A + R_W)`. -/
theorem atomicCall_closes {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {L : List (G → ℚ)} {P : List (Op100 G)} (hatom : AtomicCall pr Fcall L P)
    (hadm : Adm P L) (hw : WritesBy Rel fpSem P L T) (hOD : OutputsOn fpSem S P L (DigA ∪ DigW)) :
    c * (T.card + RA + RW) ≤ cost100 pr P := by
  rw [Nat.add_assoc]
  exact hashCompose_by hpr hconst hFX hS hRX hsha hatom.1 hatom.2 hadm hsep hw hOD

/-- A run of the call model: its instructions, each on a device. -/
abbrev Run (Dev G : Type) := List (Dev × Op100 G)

/-- The run's instructions. -/
def ops (P : Run Dev G) : List (Op100 G) := P.map Prod.snd

/-- The devices the run uses. -/
def devs (P : Run Dev G) : Set Dev := {d | ∃ o, (d, o) ∈ P}

open Classical in
/-- The run's cost counted on the devices `I`. -/
noncomputable def costOn (pr : Prices100) (I : Set Dev) (P : Run Dev G) : ℕ :=
  cost100 pr (ops (P.filter fun p => p.1 ∈ I))

/-- **`PeriodInventory I Ph Pa`** (named premise; T1's), the call model's form of the Accel premise of that
name for the devices that run the card's instructions: the honest run `Ph` and the adversary's run `Pa` use only
devices of `I`, so what either computes in the period is counted on `I`. The cards' hosts run no card instruction and
are counted at their rows (`PeriodInventoryT1`, §9); a device outside the network certifier's boundary is the egress
layer's (`egress_charged`). -/
def PeriodInventory (I : Set Dev) (Ph Pa : Run Dev G) : Prop := devs Ph ⊆ I ∧ devs Pa ⊆ I

theorem costOn_eq (pr : Prices100) {I : Set Dev} {P : Run Dev G} (h : devs P ⊆ I) :
    costOn pr I P = cost100 pr (ops P) := by
  classical
  unfold costOn
  congr 2
  refine List.filter_eq_self.2 fun p hp => ?_
  simpa using h ⟨p.2, hp⟩

/-- **A device outside the inventory is not counted**: without `PeriodInventory`, a run on such a device costs
nothing on `I`, whatever it computes. -/
theorem costOn_outside (pr : Prices100) {I : Set Dev} {d : Dev} (hd : d ∉ I) (Q : List (Op100 G)) :
    costOn pr I (Q.map fun o => (d, o)) = 0 := by
  classical
  unfold costOn
  have : (Q.map fun o => (d, o)).filter (fun p => decide (p.1 ∈ I)) = [] := by
    refine List.filter_eq_nil_iff.2 fun p hp => ?_
    obtain ⟨o, _, rfl⟩ := List.mem_map.1 hp
    simpa using hd
  rw [this]
  rfl

/-- The run's cost is its cost on `I` plus its cost off `I`. -/
theorem costOn_split (pr : Prices100) (I : Set Dev) (P : Run Dev G) :
    costOn pr I P + costOn pr Iᶜ P = cost100 pr (ops P) := by
  classical
  unfold costOn
  induction P with
  | nil => rfl
  | cons p P ih =>
    by_cases h : p.1 ∈ I
    · have h' : p.1 ∉ Iᶜ := fun h' => h' h
      simp only [List.filter_cons, h, h', decide_true, decide_false, Bool.false_eq_true, ↓reduceIte, ops,
        List.map_cons, cost100, List.sum_cons] at ih ⊢
      omega
    · have h' : p.1 ∈ Iᶜ := h
      simp only [List.filter_cons, h, h', decide_true, decide_false, Bool.false_eq_true, ↓reduceIte, ops,
        List.map_cons, cost100, List.sum_cons] at ih ⊢
      omega

/-- **The card is charged the bound less what ran off it** (given `AtomicCall` over the joint run and
`ShaWrites`; the egress layer's interface): if the adversary's joint run, on the card's devices `I` and outside, meets
`atomicCall_closes`'s hypotheses, and the egress layer bounds the outside devices' cost by `E`, the card's cost is at
least `c·(|T| + R_A + R_W) − E`. -/
theorem egress_charged {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {I : Set Dev} {Pa : Run Dev G} {L : List (G → ℚ)} {E : ℕ}
    (hatom : AtomicCall pr Fcall L (ops Pa)) (hadm : Adm (ops Pa) L) (hw : WritesBy Rel fpSem (ops Pa) L T)
    (hOD : OutputsOn fpSem S (ops Pa) L (DigA ∪ DigW)) (hE : costOn pr Iᶜ Pa ≤ E) :
    c * (T.card + RA + RW) ≤ costOn pr I Pa + E := by
  have h := atomicCall_closes hpr hconst hFX hS hRX hsha hsep hatom hadm hw hOD
  rw [← costOn_split pr I Pa] at h
  omega

/-- **The pre-noise hashing of `A` is credited on `I`** (given `PeriodInventory`, `AtomicCall` and
`ShaWrites`; T1's): with every device of both runs in `I`, the adversary's cost on `I` is at least
`c·(|T| + R_A + R_W)`. -/
theorem preNoiseHash_credited {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {I : Set Dev} {Ph Pa : Run Dev G} {L : List (G → ℚ)}
    (hinv : PeriodInventory I Ph Pa) (hatom : AtomicCall pr Fcall L (ops Pa)) (hadm : Adm (ops Pa) L)
    (hw : WritesBy Rel fpSem (ops Pa) L T) (hOD : OutputsOn fpSem S (ops Pa) L (DigA ∪ DigW)) :
    c * (T.card + RA + RW) ≤ costOn pr I Pa := by
  rw [costOn_eq pr hinv.2]
  exact atomicCall_closes hpr hconst hFX hS hRX hsha hsep hatom hadm hw hOD

/-- **Both sides on one inventory** (T1's): the gap on `I` is at most honest's whole cost less the
adversary's bound. -/
theorem gap_on_inventory {c : ℕ} {pr : Prices100} {I : Set Dev} {Ph Pa : Run Dev G} {B : ℕ}
    (hinv : PeriodInventory I Ph Pa) (hB : c * B ≤ costOn pr I Pa) :
    (costOn pr I Ph : ℤ) - costOn pr I Pa ≤ (cost100 pr (ops Ph) : ℤ) - c * B := by
  rw [costOn_eq pr hinv.1]
  have : ((c * B : ℕ) : ℤ) ≤ (costOn pr I Pa : ℤ) := by exact_mod_cast hB
  push_cast at this
  linarith

/-- **The hashing term closes to `δ` of honest's hashing**: if `ShaWrites` counts at least `(1 − δ)·R_h` of the
`R_h` hash registers honest writes, and honest pays at most `c·(|T| + R_h) + E`, the gap is at most
`c·δ·R_h + E`. -/
theorem hashTerm_closes {c Th Rh R E costH costA : ℕ} {δ : ℚ} (hR : (1 - δ) * Rh ≤ R)
    (hH : costH ≤ c * (Th + Rh) + E) (hA : c * (Th + R) ≤ costA) :
    (costH : ℚ) - costA ≤ c * δ * Rh + E := by
  have h1 : (costH : ℚ) ≤ c * (Th + Rh) + E := by exact_mod_cast hH
  have h2 : (c : ℚ) * (Th + R) ≤ costA := by exact_mod_cast hA
  have hc : (0 : ℚ) ≤ c := by positivity
  have h3 : (c : ℚ) * ((1 - δ) * Rh) ≤ c * R := mul_le_mul_of_nonneg_left hR hc
  nlinarith

/-- **A toy instance meets every hypothesis of `preNoiseHash_credited` and attains its bound**: on two grid points,
one target `tw`, one digest word `td` (the toy's `root(A)`), the constants free and the target's carriers excluded,
with `Adm` the programs that write `td`; the honest run on one inventoried device costs exactly `32·(1 + 1)`. -/
theorem toy_attained :
    let S : Set Bool := Set.univ
    let X : Set (Bool → ℚ) := Consts Bool ∪ RelOf (Carries S) {tw}
    let Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop := fun P L => td ∈ regs100 (fun _ _ _ _ => 0) P L
    let Pa : Run Unit Bool := [((), Op100.int tw), ((), Op100.int td)]
    ShaWrites h100Floor Adm (fun _ _ _ _ => 0) S X ({td} ∪ ∅) (1 + 0) ∧ SepBy (Carries S) (Consts Bool) {tw} ∧
      PeriodInventory Set.univ Pa Pa ∧ AtomicCall h100Floor (Consts Bool) [] (ops Pa) ∧ Adm (ops Pa) [] ∧
      WritesBy (Carries S) (fun _ _ _ _ => 0) (ops Pa) [] {tw} ∧
      OutputsOn (fun _ _ _ _ => 0) S (ops Pa) [] ({td} ∪ ∅) ∧
      costOn h100Floor Set.univ Pa = 32 * (({tw} : Finset (Bool → ℚ)).card + 1 + 0) := by
  intro S X Adm Pa
  have htdX : ∀ x ∈ X, ¬ ∀ g ∈ S, td g = x g := by
    rintro x (⟨q, rfl⟩ | ⟨t, ht, q, hq⟩) h
    · have h1 := h true trivial
      have h0 := h false trivial
      simp [td] at h1 h0
      rw [← h1] at h0
      norm_num at h0
    · simp at ht
      subst ht
      have h1 := h true trivial
      have h0 := h false trivial
      rw [hq true trivial] at h1
      rw [hq false trivial] at h0
      simp [td, tw] at h1 h0
      rw [← h0] at h1
      norm_num at h1
  have hregs : regs100 (fun _ _ _ _ => 0) (ops Pa) [] = [tw, td] := by
    simp [regs100, ops, Pa, Op100.outs]
  refine ⟨?_, ?_, ?_, ?_, ?_, ?_, ?_, ?_⟩
  · intro L P _ _ hA _
    refine ⟨{td}, by simp, ⟨fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht'], fun t ht x hx => ?_⟩,
      fun t ht => ⟨td, hA, fun g _ => by simp at ht; rw [ht]⟩⟩
    simp at ht
    rw [ht]
    exact htdX x hx
  · refine sepBy_carries (fun t ht t' ht' _ => by simp at ht ht'; rw [ht, ht']) fun t ht f hf => ?_
    simp at ht
    subst ht
    obtain ⟨q', rfl⟩ := hf
    rintro ⟨q, hq⟩
    have h1 := hq true trivial
    have h0 := hq false trivial
    simp [tw] at h1 h0
    rw [h0] at h1
    norm_num at h1
  · exact ⟨fun _ _ => trivial, fun _ _ => trivial⟩
  · refine ⟨by simp, fun o ho => ?_⟩
    simp [ops, Pa] at ho
    rcases ho with rfl | rfl <;> trivial
  · show td ∈ regs100 (fun _ _ _ _ => 0) (ops Pa) []
    rw [hregs]
    simp
  · intro t ht
    simp at ht
    exact ⟨tw, by rw [hregs]; simp, ⟨0, fun g _ => by rw [ht]; ring⟩⟩
  · intro t ht
    simp at ht
    exact ⟨td, by rw [hregs]; simp, fun g _ => by rw [ht]⟩
  · rw [costOn_eq (I := Set.univ) h100Floor (fun _ _ => Set.mem_univ _), Finset.card_singleton]
    decide

/-- **Outsourced hashing goes uncounted on the card** without `PeriodInventory`: the toy's run with its digest
computed on a device outside the card (`false ∉ I = {true}`) meets `ShaWrites`, `AtomicCall` over the joint run,
admissibility and the digest's output, yet costs the card `32`, below `32·(1 + 1)`. -/
theorem outsourced_uncredited :
    let S : Set Bool := Set.univ
    let X : Set (Bool → ℚ) := Consts Bool ∪ RelOf (Carries S) {tw}
    let Adm : List (Op100 Bool) → List (Bool → ℚ) → Prop := fun P L => td ∈ regs100 (fun _ _ _ _ => 0) P L
    let Pa : Run Bool Bool := [(true, Op100.int tw), (false, Op100.int td)]
    ShaWrites h100Floor Adm (fun _ _ _ _ => 0) S X ({td} ∪ ∅) (1 + 0) ∧
      AtomicCall h100Floor (Consts Bool) [] (ops Pa) ∧ Adm (ops Pa) [] ∧
      OutputsOn (fun _ _ _ _ => 0) S (ops Pa) [] ({td} ∪ ∅) ∧ ¬ PeriodInventory {true} Pa Pa ∧
      costOn h100Floor {true} Pa < 32 * (({tw} : Finset (Bool → ℚ)).card + 1 + 0) := by
  intro S X Adm Pa
  obtain ⟨hsha, -, -, hatom, hadm, -, hOD, -⟩ := toy_attained
  refine ⟨hsha, hatom, hadm, hOD, fun h => ?_, ?_⟩
  · have hm : false ∈ devs Pa := ⟨Op100.int td, by simp [Pa]⟩
    simpa using h.1 hm
  · simp [costOn, Pa, ops]
    decide

end Call

/-! ## 9. The hosts inside T1: `PeriodInventoryT1` with the hosts counted at θ's rows -/

section Hosts

open Pouw.Dither (HostRes HostRow)

/-- A sum over the host resources. -/
theorem sum_hostRes {β : Type} [AddCommMonoid β] (f : HostRes → β) : ∑ r, f r = f .amx + f .sha := by
  show ∑ r ∈ ({HostRes.amx, HostRes.sha} : Finset HostRes), f r = _
  rw [Finset.sum_insert (by simp), Finset.sum_singleton]

/-- A host row's rate at a host resource: AMX in s8 MACs a second, SHA-512 in compressions a second. -/
def rateAt (hr : HostRow) : HostRes → ℝ
  | .amx => hr.amx
  | .sha => hr.sha

/-- **The hosts' counted W1 for a period** (`HostsCounted` in the call model): the `n` cards' hosts at `n` times the
row `hr`'s rate at each resource for the period's `Tp` seconds, a unit at its credited W1 `w r` (a compression at the
W1 the verifier credits for one, a MAC at a word's credit over its MACs). -/
noncomputable def hostsW1 (n : ℕ) (hr : HostRow) (w : HostRes → ℝ) (Tp : ℝ) : ℝ :=
  ∑ r, Tp * n * w r * rateAt hr r

theorem hostsW1_eq (n : ℕ) (hr : HostRow) (w : HostRes → ℝ) (Tp : ℝ) :
    hostsW1 n hr w Tp = Tp * n * w .amx * hr.amx + Tp * n * w .sha * hr.sha := by
  simp only [hostsW1, sum_hostRes, rateAt]

/-- **The headline's `HostsCounted` covers the hosts' counted W1**: when the period's devices count the `n` cards'
hosts at the row `hr` (`Pouw.Dither.HostsCounted`), some injection of the host resources into the other devices'
gives those devices, at the credited W1 a unit, at least `hostsW1` for the period. So the call model's count reads
the same rows as the headline's capacity. -/
theorem hostsW1_le_hostsCounted {c : Verity.Accel.Card} {D : Pouw.Dither.Devices c} {hr : HostRow}
    (h : Pouw.Dither.HostsCounted D hr) {w : HostRes → ℝ} {Tp : ℝ} (hTp : 0 ≤ Tp) (hw : ∀ r, 0 ≤ w r) :
    ∃ j : HostRes → D.oth.Res, Function.Injective j ∧ hostsW1 D.n hr w Tp ≤ ∑ r, Tp * w r * D.oth.rate (j r) := by
  obtain ⟨j, hj, ha, hs⟩ := h
  refine ⟨j, hj, ?_⟩
  have h1 : Tp * w .amx * (D.n * hr.amx) ≤ Tp * w .amx * D.oth.rate (j .amx) :=
    mul_le_mul_of_nonneg_left ha (mul_nonneg hTp (hw _))
  have h2 : Tp * w .sha * (D.n * hr.sha) ≤ Tp * w .sha * D.oth.rate (j .sha) :=
    mul_le_mul_of_nonneg_left hs (mul_nonneg hTp (hw _))
  rw [hostsW1_eq, sum_hostRes]
  linarith

/-- **`PeriodInventoryT1 Wc Wcard hc Wout`** (named premise; T1's): inside the network certifier's boundary
every device is a card of the inventory or one of those cards' hosts. So the period's credited W1 `Wc` is at most what
the cards computed (`Wcard`), what the hosts computed through each host resource (`hc r`, at its credited W1), and what
devices outside the boundary computed (`Wout`), which reaches the devices only across the certifier and which the
egress layer bounds. -/
def PeriodInventoryT1 (Wc Wcard : ℝ) (hc : HostRes → ℝ) (Wout : ℝ) : Prop := Wc ≤ Wcard + ∑ r, hc r + Wout

/-- **`HostsWithinRow n hr w Tp hc`** (named premise: θ's host row is at least the hosts' best rate, a device
premise for the row's cores a card and all-core clock, and algorithmic knowledge for SHA-512's rate a core): through each host resource `r`, the `n` cards'
hosts compute in `Tp` seconds at most `Tp·n·rate_r` units' credited W1. It is resource by resource, so AMX's idle
capacity doesn't absorb SHA-512 beyond its row. -/
def HostsWithinRow (n : ℕ) (hr : HostRow) (w : HostRes → ℝ) (Tp : ℝ) (hc : HostRes → ℝ) : Prop :=
  ∀ r, hc r ≤ Tp * n * w r * rateAt hr r

/-- **With the hosts counted, the host's hashing is hashing** (given `PeriodInventoryT1` and
`HostsWithinRow`): the period's credited W1 is at most the cards' share, the hosts' counted W1 and what came from
outside the boundary. -/
theorem hostsCounted_closes {Wc Wcard Wout Tp : ℝ} {n : ℕ} {hr : HostRow} {w hc : HostRes → ℝ}
    (hinv : PeriodInventoryT1 Wc Wcard hc Wout) (hrow : HostsWithinRow n hr w Tp hc) :
    Wc ≤ Wcard + hostsW1 n hr w Tp + Wout := by
  have : ∑ r, hc r ≤ hostsW1 n hr w Tp := Finset.sum_le_sum fun r _ => hrow r
  unfold PeriodInventoryT1 at hinv
  linarith

/-- **What a row below the hosts' best leaves** (given `PeriodInventoryT1`): if the hosts compute through each
resource at most at the rates `best`, the credited W1 exceeds the cards' share, the hosts' counted W1 at the row `hr`
and the outside's by at most `Σ_r Tp·n·w_r·(best_r − rate_r)⁺`, resource by resource. -/
theorem hostRow_residual {Wc Wcard Wout Tp : ℝ} {n : ℕ} {hr best : HostRow} {w hc : HostRes → ℝ}
    (hTp : 0 ≤ Tp) (hw : ∀ r, 0 ≤ w r) (hinv : PeriodInventoryT1 Wc Wcard hc Wout)
    (hbest : HostsWithinRow n best w Tp hc) :
    Wc ≤ Wcard + hostsW1 n hr w Tp + ∑ r, Tp * n * w r * max 0 (rateAt best r - rateAt hr r) + Wout := by
  have key : ∀ r, hc r ≤ Tp * n * w r * rateAt hr r + Tp * n * w r * max 0 (rateAt best r - rateAt hr r) := by
    intro r
    have hk : 0 ≤ Tp * n * w r := mul_nonneg (mul_nonneg hTp (Nat.cast_nonneg n)) (hw r)
    have hm : rateAt best r ≤ rateAt hr r + max 0 (rateAt best r - rateAt hr r) := by
      have := le_max_right 0 (rateAt best r - rateAt hr r)
      linarith
    calc hc r ≤ Tp * n * w r * rateAt best r := hbest r
      _ ≤ Tp * n * w r * (rateAt hr r + max 0 (rateAt best r - rateAt hr r)) := mul_le_mul_of_nonneg_left hm hk
      _ = _ := by ring
  have hs : ∑ r, hc r ≤ ∑ r, (Tp * n * w r * rateAt hr r + Tp * n * w r * max 0 (rateAt best r - rateAt hr r)) :=
    Finset.sum_le_sum fun r _ => key r
  rw [Finset.sum_add_distrib] at hs
  unfold PeriodInventoryT1 at hinv
  unfold hostsW1
  linarith

/-- **T1 with the hosts, from the card's premises** (given `ShaWrites`, `AtomicCall`, the cards'
`PeriodInventory`, `PeriodInventoryT1` and `HostsWithinRow`): the cards' share is the targets and digests their
program writes, `c·(|T| + R_A + R_W)`, which `preNoiseHash_credited` charges to the cards; the hosts' share is their
row's. So the period's credited W1 is at most the cards' cost on `I`, the hosts' counted W1 and the outside's. -/
theorem hostsCounted_call {G Dev : Type} {c : ℕ} {pr : Prices100} (hpr : Floor c pr)
    {Adm : List (Op100 G) → List (G → ℚ) → Prop} {fpSem : ℕ → ℚ → ℚ → ℚ → ℚ} {S : Set G}
    {Fcall X : Set (G → ℚ)} (hconst : ∀ q : ℚ, (fun _ : G => q) ∈ Fcall) (hFX : Fcall ⊆ X)
    {Rel : (G → ℚ) → (G → ℚ) → Prop} (hS : OnS S Rel) {T DigA DigW : Finset (G → ℚ)} {RA RW : ℕ}
    [DecidableEq (G → ℚ)] (hRX : RelOf Rel T ⊆ X) (hsha : ShaWrites pr Adm fpSem S X (DigA ∪ DigW) (RA + RW))
    (hsep : SepBy Rel Fcall T) {I : Set Dev} {Ph Pa : Run Dev G} {L : List (G → ℚ)}
    (hinvC : PeriodInventory I Ph Pa) (hatom : AtomicCall pr Fcall L (ops Pa)) (hadm : Adm (ops Pa) L)
    (hw : WritesBy Rel fpSem (ops Pa) L T) (hOD : OutputsOn fpSem S (ops Pa) L (DigA ∪ DigW))
    {Wc Wout Tp : ℝ} {n : ℕ} {hr : HostRow} {wh hc : HostRes → ℝ}
    (hinv : PeriodInventoryT1 Wc ((c * (T.card + RA + RW) : ℕ) : ℝ) hc Wout) (hrow : HostsWithinRow n hr wh Tp hc) :
    Wc ≤ (costOn pr I Pa : ℝ) + hostsW1 n hr wh Tp + Wout := by
  have hcard : ((c * (T.card + RA + RW) : ℕ) : ℝ) ≤ (costOn pr I Pa : ℝ) := by
    exact_mod_cast preNoiseHash_credited hpr hconst hFX hS hRX hsha hsep hinvC hatom hadm hw hOD
  have := hostsCounted_closes hinv hrow
  linarith

/-- **Route H gains no capacity** (given `PeriodInventoryT1` and `HostsWithinRow`): if the cards' share is
at most the adversary's card cost and that cost, with the other work `W_other` it runs on the cards, fits the cards'
capacity `C_card` for the period, then the credited W1 and the other work together fit the period's capacity with the
hosts counted at their rows, plus what came from outside the boundary. So hashing moved to the hosts frees no card
capacity the period doesn't count. -/
theorem hostsCounted_capacity {Wc Wcard Wout Tp costA Wother Ccard : ℝ} {n : ℕ} {hr : HostRow} {w hc : HostRes → ℝ}
    (hinv : PeriodInventoryT1 Wc Wcard hc Wout) (hrow : HostsWithinRow n hr w Tp hc) (hcard : Wcard ≤ costA)
    (hcap : costA + Wother ≤ Ccard) : Wc + Wother ≤ Ccard + hostsW1 n hr w Tp + Wout := by
  have := hostsCounted_closes hinv hrow
  linarith

/-- **The row's own hosts are counted**: devices whose other devices are the row's hosts for their `n` cards
(`HostRow.others`) meet `HostsCounted` at that row. -/
theorem hostsCounted_others {c : Verity.Accel.Card} (n : ℕ) (hn : 0 < n) (f mem : ℝ) (hr : HostRow) :
    Pouw.Dither.HostsCounted (c := c) ⟨n, hn, f, hr.others n mem⟩ hr :=
  ⟨fun r => r, fun _ _ h => h, le_rfl, le_rfl⟩

/-- **The hosts' counted W1 is within the headline's capacity** (given `OnCard`'s rates and `HostsCounted`):
at the `G` the headline is read at, the period's capacity (`Period.cap`) at the host resources, a unit at its credited
W1, is at least `hostsW1` for the period. So the term `hostsCounted_capacity` adds is capacity the headline's
`OPT_honest` and `C` already count. -/
theorem hostsW1_le_cap {GA : Pouw.Dither.Accel} {c : Verity.Accel.Card} {x} {D : Pouw.Dither.Devices c} {P} {L}
    {e : GA.Res ≃ Verity.Accel.Res ⊕ D.oth.Res} (hon : Pouw.Dither.OnCard GA c x D P L e) (Pd : Pouw.Dither.Period GA)
    {hr : HostRow} (h : Pouw.Dither.HostsCounted D hr) {w : HostRes → ℝ} (hT : 0 ≤ Pd.T) (hw : ∀ r, 0 ≤ w r) :
    ∃ j : HostRes → D.oth.Res, Function.Injective j ∧
      hostsW1 D.n hr w Pd.T ≤ ∑ r, w r * Pd.cap (e.symm (.inr (j r))) := by
  obtain ⟨j, hj, hle⟩ := hostsW1_le_hostsCounted h hT hw
  refine ⟨j, hj, hle.trans (le_of_eq (Finset.sum_congr rfl fun r _ => ?_))⟩
  rw [Pouw.Dither.Period.cap, hon.2.1]
  ring

/-- **`HostsWithinRow` is the headline's capacity at a row-counted host** (given `OnCard`'s rates): if a
strategy fits the period (`Period.Fits`, which `FreedSound` reads at every other work `z` and `TimedWorkFits` gives at
`z = 0` for an accepted period's sampling blocks' work) and the host resources `j` are counted at no more than the
row, then its use of them, a unit at its credited W1, meets `HostsWithinRow` at that row. So for the hosts' timed work
the premise is the headline's own (G2) rate at θ's row. -/
theorem hostsWithinRow_of_fits {GA : Pouw.Dither.Accel} {c : Verity.Accel.Card} {x} {D : Pouw.Dither.Devices c} {P}
    {L} {e : GA.Res ≃ Verity.Accel.Res ⊕ D.oth.Res} (hon : Pouw.Dither.OnCard GA c x D P L e)
    {Pd : Pouw.Dither.Period GA} {X : GA.Res → ℝ} {z : Pd.J → ℝ} (hfit : Pd.Fits X z)
    (hneeds : ∀ k r, 0 ≤ Pd.needs k r) (hT : 0 ≤ Pd.T) {hr : HostRow} {j : HostRes → D.oth.Res}
    (hj : ∀ r, D.oth.rate (j r) ≤ D.n * rateAt hr r) {w : HostRes → ℝ} (hw : ∀ r, 0 ≤ w r) :
    HostsWithinRow D.n hr w Pd.T (fun r => w r * X (e.symm (.inr (j r)))) := by
  intro r
  have hz : 0 ≤ ∑ k, z k * Pd.needs k (e.symm (.inr (j r))) :=
    Finset.sum_nonneg fun k _ => mul_nonneg (hfit.1 k) (hneeds k _)
  have h1 := hfit.2 (e.symm (.inr (j r)))
  have h2 : Pd.cap (e.symm (.inr (j r))) ≤ D.n * rateAt hr r * Pd.T := by
    rw [Pouw.Dither.Period.cap, hon.2.1]
    exact mul_le_mul_of_nonneg_right (hj r) hT
  have hX : X (e.symm (.inr (j r))) ≤ Pd.T * D.n * rateAt hr r := by linarith
  calc w r * X (e.symm (.inr (j r))) ≤ w r * (Pd.T * D.n * rateAt hr r) := mul_le_mul_of_nonneg_left hX (hw r)
    _ = Pd.T * D.n * w r * rateAt hr r := by ring

/-- **The hosts inside T1 from the headline's terms** (given `OnCard`'s rates and `PeriodInventoryT1`): when
the strategy fits the period and the host resources are counted at no more than the row, the credited W1 is at most
the cards' share, the hosts' counted W1 at the row and the outside's. `HostsWithinRow` is then no premise of its own. -/
theorem hostsCounted_closes_of_fits {GA : Pouw.Dither.Accel} {c : Verity.Accel.Card} {x} {D : Pouw.Dither.Devices c}
    {P} {L} {e : GA.Res ≃ Verity.Accel.Res ⊕ D.oth.Res} (hon : Pouw.Dither.OnCard GA c x D P L e)
    {Pd : Pouw.Dither.Period GA} {X : GA.Res → ℝ} {z : Pd.J → ℝ} (hfit : Pd.Fits X z)
    (hneeds : ∀ k r, 0 ≤ Pd.needs k r) (hT : 0 ≤ Pd.T) {hr : HostRow} {j : HostRes → D.oth.Res}
    (hj : ∀ r, D.oth.rate (j r) ≤ D.n * rateAt hr r) {w : HostRes → ℝ} (hw : ∀ r, 0 ≤ w r) {Wc Wcard Wout : ℝ}
    (hinv : PeriodInventoryT1 Wc Wcard (fun r => w r * X (e.symm (.inr (j r)))) Wout) :
    Wc ≤ Wcard + hostsW1 D.n hr w Pd.T + Wout :=
  hostsCounted_closes hinv (hostsWithinRow_of_fits hon hfit hneeds hT hj hw)

/-- **Route H on a toy, and the host row that closes it**: the card writes the word `tw` (32) and a host hashes the
digest, credited 32 through its SHA-512. The cards' `PeriodInventory` holds, and the credited 64 is over the card's
cost. Counted at a row of one compression a second (credited 32), the hosts meet `HostsWithinRow` and the bound of
`hostsCounted_closes` is attained; at a zero SHA-512 row, `HostsWithinRow` fails. -/
theorem routeH_toy :
    let Pa : Run Unit Bool := [((), Op100.int tw)]
    let hc : HostRes → ℝ := fun r => match r with
      | .amx => 0
      | .sha => 32
    PeriodInventory Set.univ Pa Pa ∧ PeriodInventoryT1 64 32 hc 0 ∧
      (costOn h100Floor Set.univ Pa : ℝ) + 0 < 64 ∧
      HostsWithinRow 1 ⟨0, 1⟩ (fun _ => 32) 1 hc ∧
      (64 : ℝ) = (costOn h100Floor Set.univ Pa : ℝ) + hostsW1 1 ⟨0, 1⟩ (fun _ => 32) 1 + 0 ∧
      ¬ HostsWithinRow 1 ⟨0, 0⟩ (fun _ => 32) 1 hc := by
  intro Pa hc
  have hcost : costOn h100Floor Set.univ Pa = 32 := by
    rw [costOn_eq (I := Set.univ) h100Floor (fun _ _ => Set.mem_univ _)]
    decide
  refine ⟨⟨fun _ _ => Set.mem_univ _, fun _ _ => Set.mem_univ _⟩, ?_, ?_, ?_, ?_, ?_⟩
  · show (64 : ℝ) ≤ 32 + ∑ r, hc r + 0
    rw [sum_hostRes]
    norm_num [hc]
  · rw [hcost]
    norm_num
  · intro r
    cases r <;> norm_num [hc, rateAt]
  · rw [hcost, hostsW1_eq]
    norm_num
  · intro h
    have := h .sha
    norm_num [hc, rateAt] at this

end Hosts

end Pouw.SecurityProofs.Lead.Compose
