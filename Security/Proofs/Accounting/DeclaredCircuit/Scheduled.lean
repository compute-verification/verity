import Proofs.Accounting.DeclaredCircuit.Requirement
import Properties.Accounting.DeclaredCircuit
import Proofs.Properties.RemoteAudit.Schedule

/-!
# A declared circuit's space and bandwidth at the committed steering cap, proved

`Properties.DeclaredCircuit.SecurityProofs.scheduledRequirement`, the theorem of
`Properties.DeclaredCircuit.ScheduledRequirement`: `Accounting.SecurityProofs.DeclaredCircuitRequirement`'s argument on
the coins the schedule check accepts. At each prefix, `Properties.RemoteAudit.steered_of_checked` over the coins the
check accepts there makes the layout the slot of the schedule's `cap + 1` padded bits (`steered_on`). The session priced
(`RateSketch.pr_session_le`) is `A` on summaries of `a + (cap + 1)` bits, the devices' `a` then the schedule's
(`SketchSummarySession.widen`), on the schedule's slot, which is the layout on the accepted coins, so there each pass's
events are `G`'s (`RateSketch.pass_congr`). A window's reads counted against the draws are `G`'s on a session the check
accepts at every pass, and its correct receipts on one it refuses, which no shortfall reaches.
-/

namespace Accounting.RateSketch.SketchSummarySession

variable {a n P N m ρ h S L q ks G d : ℕ}

/-- `A` on summaries of `b` more bits: a summary's session is `A`'s at its first `a` bits. -/
def widen (A : SketchSummarySession a n P N m ρ h S L q ks G d) (b : ℕ) :
    SketchSummarySession (a + b) n P N m ρ h S L q ks G d :=
  ⟨fun s => A.session fun i => s (Fin.castAdd b i)⟩

theorem widen_session (A : SketchSummarySession a n P N m ρ h S L q ks G d) {b : ℕ} (s : Pous.Bits a)
    (t : Pous.Bits b) : (A.widen b).session (Fin.append s t) = A.session s := by
  simp only [widen, Fin.append_left]

end Accounting.RateSketch.SketchSummarySession

namespace Accounting.RateSketch

variable {n P N ρ S L q G T M : ℕ} {Ω Ca Ch : Type}

/-- A pass's events read its device and its store: sessions that tell the pass one device, on one layout, give it the
same guessed tags and the same `k + 1` correct digests behind unobtained positions. -/
theorem pass_congr {a a' k : ℕ} {A : SketchSummarySession a n P N 128 ρ 512 S L q 16 G 2773}
    {A' : SketchSummarySession a' n P N 128 ρ 512 S L q 16 G 2773}
    {steer : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a}
    {steer' : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Pous.Bits a'}
    {enc : Ω → Fin T → Ch → (Fin T → Ca) → Fin M → (Fin 128 → Sector)}
    {choose : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω → (b : Fin T) → ({b' : Fin T // b' < b} → Ca) → Ch}
    {layout layout' : SketchPrefix n P N 128 ρ 512 16 G 2773 → Ω × (Fin T → Ca) → Fin N ↪ Fin T × Fin M}
    {c : SketchPrefix n P N 128 ρ 512 16 G 2773} {y : Ω × (Fin T → Ca)}
    (hs : A'.session (steer' c y) = A.session (steer c y)) (hl : layout' c y = layout c y)
    (pc : SketchPassCoins N ρ 512 16 G) :
    (PassGuess A' steer' enc choose layout' c y pc ↔ PassGuess A steer enc choose layout c y pc) ∧
      (PassBadK k A' steer' enc choose layout' c y pc ↔ PassBadK k A steer enc choose layout c y pc) := by
  have hst : steeredStore enc choose layout' c y = steeredStore enc choose layout c y := by
    unfold steeredStore
    rw [hl]
  simp only [PassGuess, PassBadK, passDev, hs, hst, iff_self, and_self]

end Accounting.RateSketch

namespace Properties.DeclaredCircuit.SecurityProofs

open Accounting Accounting.RateSketch
open Pous (pr)
open scoped ENNReal
open Bandwidth (pr_mono)

/-- **The layout steered on the accepted coins.** Under `ScheduleLayoutGap`, on coins `acc` at which the schedule check
at the committed cap `cap` accepts, the layout is the slot of the schedule's `cap + 1` padded bits:
`Properties.RemoteAudit.steered_of_checked` at each prefix, over the coins `acc` holds of there. -/
theorem steered_on {Pfx Y : Type} {N T M blocks cap : ℕ} {idx : Pfx → ℕ}
    {pos : Pfx → Array (ℕ × ℕ) → Array (ℕ × ℕ) → Fin N ↪ Fin T × Fin M} {appended sent : Pfx → Y → Array (ℕ × ℕ)}
    {layout : Pfx → Y → Fin N ↪ Fin T × Fin M} (gap : Properties.RemoteAudit.ScheduleLayoutGap pos appended sent layout)
    (acc : Pfx → Y → Prop)
    (hacc : ∀ c y, acc c y → ∃ sc passes b, RemoteAudit.Schedule.check sc blocks cap passes (appended c y) = .ok b ∧
      (passes[idx c]?).map Prod.snd = some (sent c y)) :
    ∃ (sched : Pfx → Y → Pous.Bits (cap + 1)) (slot : Pfx → Pous.Bits (cap + 1) → Fin N ↪ Fin T × Fin M),
      ∀ c y, acc c y → layout c y = slot c (sched c y) := by
  classical
  have h : ∀ c, ∃ (sched : {y // acc c y} → Pous.Bits (cap + 1)) (slot : Pous.Bits (cap + 1) → Fin N ↪ Fin T × Fin M),
      ∀ y : {y // acc c y}, layout c y.1 = slot (sched y) := by
    intro c
    obtain ⟨sched, slot, hs⟩ := Properties.RemoteAudit.steered_of_checked (Pfx := Unit) (Y := {y // acc c y})
      (blocks := blocks) (a := cap) (idx := fun _ => idx c) (pos := fun _ => pos c)
      (appended := fun _ y => appended c y.1) (sent := fun _ y => sent c y.1) (layout := fun _ y => layout c y.1)
      (fun (_ : Unit) (y : {y // acc c y}) => gap c y.1) (fun (_ : Unit) (y : {y // acc c y}) => hacc c y.1 y.2)
    exact ⟨sched (), slot (), fun y => hs () y⟩
  choose sched slot hs using h
  refine ⟨fun c y => if hy : acc c y then sched c ⟨y, hy⟩ else fun _ => false, slot, fun c y hy => ?_⟩
  simp only [hy, ↓reduceDIte]
  exact hs c ⟨y, hy⟩

/-- **A declared circuit's held and moved bits from an accepted stacked audit, at the committed steering cap**,
`Properties.DeclaredCircuit.ScheduledRequirement`. -/
theorem scheduledRequirement : ScheduledRequirement := by
  intro n P N ρ S q G Xr Lr qr τ k a L E es B εnum εden T M ε δ ιW Wc caps Aud Ω Ca Ch _ _ enc choose layout steer
    joined A blocks cap idx pos appended sent V Kg Dv _ _ W pl ℓ grp bind mm plan addr hn hP hN hE0 hrank hKB hSq hCs
    hEle hε hB hCC hgap hlife _hres hprop hreads hx RL C γ owned credit tiles store win accepted
  have hE : Gates.EncodeBound Gates.p1024 3 B E εnum εden :=
    Accounting.SecurityProofs.Gates.encodeBound_of hSq hCs hEle hε (by decide)
      Accounting.SecurityProofs.Gates.p1024_floor.1 Accounting.SecurityProofs.Gates.p1024_floor.2
  obtain ⟨sched, slot, hslot⟩ := steered_on (blocks := blocks) (cap := cap) (idx := idx) hgap accepted
    fun _ _ h => h
  have hB' : ∀ s w η j pre, Assumptions.SketchHashBudget
      ((((A.widen (cap + 1)).session s).window w η).device j pre) (qr * τ) :=
    fun s w η j pre => hB (fun i => s (Fin.castAdd (cap + 1) i)) w η j pre
  have hsess := pr_session_le (k := k) hrank hKB choose (fun c y => slot c (sched c y))
    (fun c y => Fin.append (steer c y) (sched c y))
    ⟨fun c s => slot c fun i => s (Fin.natAdd a i), fun c y =>
      congrArg (slot c) (funext fun i => (Fin.append_right (steer c y) (sched c y) i).symm)⟩
    (A.widen (cap + 1)) hB'
  have hpass : ∀ c y, accepted c y → ∀ pc : SketchPassCoins N ρ 512 16 G,
      (PassGuess (A.widen (cap + 1)) (fun c y => Fin.append (steer c y) (sched c y)) enc choose
          (fun c y => slot c (sched c y)) c y pc ↔ PassGuess A steer enc choose layout c y pc) ∧
        (PassBadK k (A.widen (cap + 1)) (fun c y => Fin.append (steer c y) (sched c y)) enc choose
          (fun c y => slot c (sched c y)) c y pc ↔ PassBadK k A steer enc choose layout c y pc) :=
    fun c y h pc => pass_congr (k := k) (A := A) (A' := A.widen (cap + 1)) (steer := steer)
      (steer' := fun c y => Fin.append (steer c y) (sched c y)) (enc := enc) (choose := choose) (layout := layout)
      (layout' := fun c y => slot c (sched c y)) (c := c) (y := y)
      (SketchSummarySession.widen_session A (steer c y) (sched c y)) (hslot c y h).symm pc
  have hX : ∀ (c : SketchPrefix n P N 128 ρ 512 16 G 2773) (y : Ω × (Fin T → Ca)) (g : Fin G)
      (R : RateSketch.SketchStream ρ) (O : RateSketch.Oracle N ρ 512 RateSketch.Sector (Fin 16 → ZMod (2 ^ 32))),
      ((passDev A steer c y g).prog c.signs (Data.ofParts (steeredStore enc choose layout c y) R)).rebuildsIn
        (appendedTiles layout c y) O (Data.ofParts (steeredStore enc choose layout c y) R) * 256 ≤ Xr * τ + C := by
    intro c y g R O
    obtain ⟨K, I, hreb, hI, hcard, hK⟩ := hCC hlife c g y
    have h1 := hreb R O
    dsimp only at h1
    exact h1.trans (Nat.add_le_add_left (Carrier.card_le_census hE hE0 hprop hreads K hI hcard hK) _)
  have hcredit : ∀ x w j, credit x (bind x w j) ≤ (tiles x w j).card * 32768 := by
    intro x w j
    have hm := (hx x).2.2 w j
    dsimp only at hm
    obtain ⟨hS, hPl, hO, hinj, hrange⟩ := hm
    exact Requirement.credit_le_tiles W (pl x).toExecution (ℓ x) (mm x) (plan x w j) (grp x) (bind x w j) (addr x)
      (layout (SketchPrefix.of x.2 w j) x.1) hS hPl hO hinj hrange
  have htiles : ∀ x w j, tiles x w j ⊆ appendedTiles layout (SketchPrefix.of x.2 w j) x.1 := fun x w j =>
    Requirement.tiles_subset_appended layout (SketchPrefix.of x.2 w j) x.1 (addr x)
      (Requirement.blocks (ℓ x) (Requirement.creditedReads W (pl x).toExecution (ℓ x) (bind x w j)))
  have hcs : ∀ x w, ∑ j, credit x (bind x w j) ≤ P * (N * 32768) := by
    intro x w
    calc ∑ j, credit x (bind x w j) ≤ ∑ _j : Fin P, N * 32768 := Finset.sum_le_sum fun j _ =>
          (hcredit x w j).trans (Nat.mul_le_mul_right _ ((Finset.card_le_univ _).trans_eq (Fintype.card_fin N)))
      _ = P * (N * 32768) := by simp
  have hcor : ∀ (o : Ω × (Fin T → Ca)) (ws ws' : Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) (w : Fin n),
      (∀ v, v < w → ws v = ws' v) → (ws w).1 = (ws' w).1 → (ws w).2.1 = (ws' w).2.1 →
      (win (o, ws) w).correctOn (store o ws w) (ws w).1 (ws w).2.1 =
        (win (o, ws') w).correctOn (store o ws' w) (ws' w).1 (ws' w).2.1 := by
    intro o ws ws' w hlt hσ hp
    have hη : (fun v : {v : Fin n // v < w} => ws v.1) = fun v => ws' v.1 := funext fun v => hlt v.1 v.2
    have hpre : ∀ j, SketchPrefix.of ws w j = SketchPrefix.of ws' w j := fun j => by
      simp only [SketchPrefix.of]
      rw [hη, hσ, hp]
    have hwin : win (o, ws) w = win (o, ws') w := by
      show A.win w ws (fun j => steer (SketchPrefix.of ws w j) o) =
        A.win w ws' (fun j => steer (SketchPrefix.of ws' w j) o)
      simp only [hpre, SketchSummarySession.win, SketchDeployedSession.win]
      rw [hη]
    have hstore : store o ws w = store o ws' w := funext fun j => by
      show steeredStore enc choose layout (SketchPrefix.of ws w j) o =
        steeredStore enc choose layout (SketchPrefix.of ws' w j) o
      rw [hpre j]
    rw [hwin, hstore, hσ, hp]
  classical
  let rd : Ω × (Fin T → Ca) → (Fin n → SketchWindowCoins P N 128 ρ 512 16 G 2773) → Fin n → ℕ := fun o ws w =>
    if ∀ v i, accepted (SketchPrefix.of ws v i) o then
      (win (o, ws) w).readsIn (store o ws w) (ws w).1 (ws w).2.1 (tiles (o, ws) w) +
        (P * (N * 32768) - ∑ j, credit (o, ws) (bind (o, ws) w j))
    else ((win (o, ws) w).correctOn (store o ws w) (ws w).1 (ws w).2.1).card * (128 * 256)
  have hu : pr (fun z : StackedCoins Ω Ca T n P N ρ G => ∃ w,
      rd z.1 z.2 w + P * (S + (Lr * τ - ρ * 256) + (Xr * τ + C + k * 32768)) <
      ((win z w).correctOn (store z.1 z.2 w) (z.2 w).1 (z.2 w).2.1).card * (128 * 256)) ≤
      (n : ℝ≥0∞) * ε + (n * P * G : ℝ≥0∞) * ((N : ℝ≥0∞) / 2 ^ 512) +
        (n * P : ℝ≥0∞) * (2 ^ (a + (cap + 1)) * (((qr * τ : ℕ) : ℝ≥0∞) / 2 ^ 96) ^ (k + 1) + δ) := by
    refine le_trans (pr_mono fun x hx' => ?_) hsess
    obtain ⟨w, hw⟩ := hx'
    by_cases hacc : ∀ v i, accepted (SketchPrefix.of x.2 v i) x.1
    swap
    · have h0 : rd x.1 x.2 w =
          ((win x w).correctOn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1).card * (128 * 256) := ite_eq_right hacc
      rw [h0] at hw
      exact absurd hw (Nat.not_lt.mpr (Nat.le_add_right _ _))
    have h1 : rd x.1 x.2 w = (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w) +
        (P * (N * 32768) - ∑ j, credit x (bind x w j)) := ite_eq_left hacc
    rw [h1] at hw
    by_cases hr : RankGood 12 (x.2 w).1
    swap
    · exact ⟨w, Or.inl hr⟩
    obtain ⟨j, hj⟩ := SketchDeployedWindow.exists_unpaid_tiles (win x w) (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1
      (tiles x w) (fun j => credit x (bind x w j)) (S + (Lr * τ - ρ * 256) + (Xr * τ + C + k * 32768))
      (hcredit x w) hw
    have hXp : (((win x w).dev j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1).prog (x.2 w).1
        (Data.ofParts (store x.1 x.2 w j) ((x.2 w).2.1 j).1)).rebuildsIn (tiles x w j)
        ((win x w).oracle j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
        (Data.ofParts (store x.1 x.2 w j) ((x.2 w).2.1 j).1) * 256 ≤ Xr * τ + C :=
      le_trans (Nat.mul_le_mul_right 256 (RateSketch.Chip.rebuildsIn_mono (htiles x w j) _ _ _))
        (hX (SketchPrefix.of x.2 w j) x.1 ((win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
          ((x.2 w).2.1 j).1 (((x.2 w).2.1 j).2 ((win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)))
    have hp := hpass (SketchPrefix.of x.2 w j) x.1 (hacc w j) ((x.2 w).2.1 j)
    rcases RateSketch.Device.unpaid_tiles_split ((win x w).dev j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
        RateSketch.sketch (x.2 w).1 ((win x w).oracle j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1)
        (Data.ofParts (store x.1 x.2 w j) ((x.2 w).2.1 j).1) (tiles x w j) (Xr * τ + C) k (by norm_num) hXp hj
      with ⟨t, ht⟩ | ⟨ι, hb⟩
    · exact ⟨w, Or.inr ⟨j, Or.inl
        (hp.1.mpr ⟨(win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1, t, ht⟩)⟩⟩
    · exact ⟨w, Or.inr ⟨j, Or.inr
        (hp.2.mpr ⟨hr, (win x w).kept j (store x.1 x.2 w j) (x.2 w).1 (x.2 w).2.1, ι, hb⟩)⟩⟩
  refine le_trans (pr_mono fun x hx' => ?_) ((SketchDeployedSession.pr_deployed_of_unpaid_causal
    (fun o ws w => (win (o, ws) w).correctOn (store o ws w) (ws w).1 (ws w).2.1) rd
    hcor (Xr * τ + C + k * 32768) hn hP hN _ hu).trans (le_of_eq ?_))
  · obtain ⟨hacc, hall, hT | ⟨w, hw⟩⟩ := hx'
    · refine ⟨hall, Or.inl ?_⟩
      exact Requirement.sum_lt_of_credit (fun w => rd x.1 x.2 w)
        (fun w => (win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w))
        (fun w => ∑ j, credit x (bind x w j)) (P * (N * 32768)) (0.0015 + γ) (n * P * N * 32768)
        (fun _ => ite_eq_left hacc) (hcs x) (by push_cast; ring) hT
    · refine ⟨hall, Or.inr ⟨w, ?_⟩⟩
      exact Requirement.lt_of_credit (rd x.1 x.2 w)
        ((win x w).readsIn (store x.1 x.2 w) (x.2 w).1 (x.2 w).2.1 (tiles x w))
        (∑ j, credit x (bind x w j)) (P * (N * 32768)) (0.01 + γ) (P * N * 32768) (ite_eq_left hacc) (hcs x w)
        (by push_cast; ring) hw
  · ring

end Properties.DeclaredCircuit.SecurityProofs
