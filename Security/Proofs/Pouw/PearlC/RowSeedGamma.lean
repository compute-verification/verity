import Definitions.Pouw.Assumptions.PearlC.TTOutRowSeed
import Proofs.Pouw.PearlC.DeviceRev1Gamma
import Proofs.Pouw.PearlC.DeviceSm120Gamma

/-!
# Row additivity: rev1's TT_OUT gives the row-seeded one (staging, nothing pinned)

The per-row seed is not a new assumption: rev1's TT_OUT on the split workload, where every row is a unit of its own,
gives `TTOutRowSeed`, and the γ theorems carry over with the error at the row count.

* **Transport:** at a row-local record (C0, `devAt_rowLocal`, by `rfl` at every captured step) and under C1, row `i`'s
  ticket and useful words (`split_ticket`, `split_useful`) and its debit (`split_debit`, `split_tileDebit`) are those of
  the split unit `rowIdx L u i`.
* **Per unit:** a unit's credit is the sum of its rows' credits as split units (`split_credit`); a correct row under
  C2 is a correct split unit (`split_correct`), and the per-row cap implies rev1's (`rowCap_agg`). So the rows' credit
  is at most the split transcript's (`split_credit_le`), and with C3 `ttOutRowSeed_of_ttOut` follows.
* **Per audit tile:** the same through the split layout's 64×64 tiles (`split_tileCredit_le`),
  `ttOutTileRowSeed_of_ttOutTile`.
* **One-row units are admitted:** `pearlCDomainDev_split`, `devSm120v1_domain_row`, `devSm120v2_domain_row`.
* **γ at any record,** on the one-shape domain with at most `2^64` rows, at `γ = 1 − (399/400)/ω` and error
  `εPearlC q (N·m)` (`ttOut_of_ttOutRowSeed`, `ttOutTile_of_ttOutTileRowSeed`): `pearlCGammaDevRev1RowSeedAt` and
  `pearlCSampledDevRev1RowSeedAt`.
* **γ at the sm_120 records**, from rev1's TT_OUT (per unit or per tile), C1 and C3 alone:
  - v1 (`devSm120v1 Prices.sm120`), cap 1/400: `16223/3177500` at 8192³ and `2963/586700` at 16384³;
  - v2 (`devSm120v2 Prices.sm120`), cap 1/1,000: `216793/59950000` at 8192³ and `2950127/829250000` at 16384³;
  per unit and per audit tile, the rev1 values at both records.
-/

namespace Pouw.PearlC

open Finset Pouw.PearlC.Assumptions Pouw.Fp8Atom

/-! ## The split layout's indices -/

theorem rowOffset_succ (L : Layout) (u : ℕ) : rowOffset L (u + 1) = rowOffset L u + (L.shape u).m := by
  unfold rowOffset; exact sum_range_succ _ _

theorem rowOffset_mono (L : Layout) {u w : ℕ} (h : u ≤ w) : rowOffset L u ≤ rowOffset L w := by
  unfold rowOffset; exact sum_le_sum_of_subset (range_subset_range.mpr h)

theorem unitOfRow_rowIdx {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    unitOfRow L (rowIdx L u i) = u := by
  unfold unitOfRow
  rw [Nat.findGreatest_eq_iff]
  refine ⟨by omega, fun _ => ?_, fun w hw _ => ?_⟩
  · show rowOffset L u ≤ rowIdx L u i
    unfold rowIdx; omega
  · have h1 := rowOffset_mono L (Nat.succ_le_of_lt hw)
    rw [rowOffset_succ] at h1
    show ¬ rowOffset L w ≤ rowIdx L u i
    unfold rowIdx; omega

theorem rowOfRow_rowIdx {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    rowOfRow L (rowIdx L u i) = i := by
  unfold rowOfRow; rw [unitOfRow_rowIdx hu hi]; unfold rowIdx; omega

theorem rowIdx_lt_rowsOf {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    rowIdx L u i < rowsOf L := by
  have h1 := rowOffset_mono L (Nat.succ_le_of_lt hu)
  rw [rowOffset_succ] at h1
  unfold rowIdx rowsOf; omega

theorem rowIdx_inj {L : Layout} {u i u' i' : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) (hu' : u' < L.N)
    (hi' : i' < (L.shape u').m) (h : rowIdx L u i = rowIdx L u' i') : u = u' ∧ i = i' := by
  have h1 := unitOfRow_rowIdx hu hi
  have h2 := unitOfRow_rowIdx hu' hi'
  rw [h] at h1
  have e : u = u' := h1.symm.trans h2
  subst e
  refine ⟨rfl, ?_⟩
  unfold rowIdx at h; omega

theorem unitOfRow_lt {L : Layout} {v : ℕ} (hv : v < rowsOf L) : unitOfRow L v < L.N := by
  have hN : L.N ≠ 0 := by
    intro h
    unfold rowsOf rowOffset at hv
    rw [h] at hv
    simp at hv
  have := Nat.findGreatest_le (P := fun u => rowOffset L u ≤ v) (L.N - 1)
  unfold unitOfRow; omega

/-- On a one-shape layout the declared rows are `N·m`. -/
theorem rowsOf_const {L : Layout} {s : Shape} (h : ∀ u < L.N, L.shape u = s) : rowsOf L = L.N * s.m := by
  unfold rowsOf rowOffset
  rw [sum_congr rfl fun u hu => by rw [h u (mem_range.mp hu)], sum_const, card_range, smul_eq_mul]

theorem splitShape_eq {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    (splitLayout L).shape (rowIdx L u i) = ⟨1, (L.shape u).n, (L.shape u).k⟩ := by
  show (⟨1, (L.shape (unitOfRow L (rowIdx L u i))).n, (L.shape (unitOfRow L (rowIdx L u i))).k⟩ : Shape) = _
  rw [unitOfRow_rowIdx hu hi]

theorem splitShape_n {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    ((splitLayout L).shape (rowIdx L u i)).n = (L.shape u).n := by
  rw [splitShape_eq hu hi]

theorem splitShape_k {L : Layout} {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    ((splitLayout L).shape (rowIdx L u i)).k = (L.shape u).k := by
  rw [splitShape_eq hu hi]

theorem splitShapeW_eq {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) :
    (splitWorkload U).layout.shape (rowIdx U.layout u i) = ⟨1, (U.layout.shape u).n, (U.layout.shape u).k⟩ :=
  splitShape_eq hu hi

theorem splitShapeW_n {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) :
    ((splitWorkload U).layout.shape (rowIdx U.layout u i)).n = (U.layout.shape u).n :=
  splitShape_n hu hi

theorem splitShapeW_k {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) :
    ((splitWorkload U).layout.shape (rowIdx U.layout u i)).k = (U.layout.shape u).k :=
  splitShape_k hu hi

/-! ## One-row units are in the domain -/

/-- **The split layout of an in-domain layout is in the domain**: one-row units are admitted. -/
theorem pearlCDomainDev_split {d : PearlCDevice} {L : Layout} (h : pearlCDomainDev d L) :
    pearlCDomainDev d (splitLayout L) := fun _ hv =>
  have hw := h _ (unitOfRow_lt hv)
  ⟨Nat.one_pos, hw.2.1, hw.2.2.1, hw.2.2.2.1, hw.2.2.2.2.1, hw.2.2.2.2.2⟩

/-- An in-domain unit has positive `m` and `n`. -/
theorem pearlCDomainDev_pos {d : PearlCDevice} {L : Layout} (h : pearlCDomainDev d L) {u : ℕ} (hu : u < L.N) :
    0 < (L.shape u).m ∧ 0 < (L.shape u).n :=
  ⟨(h u hu).1, lt_of_lt_of_le (by norm_num) (h u hu).2.1.1⟩

/-- A one-row unit at either headline depth is in v1's domain. -/
theorem devSm120v1_domain_row (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDev (devSm120v1 Prices.sm120) ⟨1, fun _ => ⟨1, s.n, s.k⟩, fun _ => 0⟩ := by
  intro u _
  rcases hs with rfl | rfl <;> dsimp only <;>
    exact ⟨by decide, by decide, by decide, by decide, Or.inr (by decide), by decide⟩

/-- A one-row unit at either headline depth is in v2's domain. -/
theorem devSm120v2_domain_row (s : Shape) (hs : s = sh8192 ∨ s = sh16384) :
    pearlCDomainDev (devSm120v2 Prices.sm120) ⟨1, fun _ => ⟨1, s.n, s.k⟩, fun _ => 0⟩ := by
  intro u _
  rcases hs with rfl | rfl <;> dsimp only <;> exact ⟨by decide, by decide, by decide, by decide, Or.inl rfl, by decide⟩

/-! ## Row-local records -/

/-- **(C0) holds at every captured step's record**, by `rfl`: its ticket, U and flags read `A′`'s row `i` only. -/
theorem devAt_rowLocal (p : Pipeline) (G : ℕ) (pr : Prices) : (devAt p G pr).RowLocal :=
  ⟨fun _ _ _ _ => ⟨rfl, fun _ => rfl⟩, fun _ _ _ _ _ _ _ _ => rfl⟩

/-! ## Transport across propositionally equal dimensions -/

/-- Two vectors of propositionally equal lengths with the same entries at every natural index. -/
def VecEq {α : Type} {q q' : ℕ} (v : Fin q → α) (v' : Fin q' → α) : Prop :=
  q = q' ∧ ∀ (l : ℕ) (hl : l < q) (hl' : l < q'), v ⟨l, hl⟩ = v' ⟨l, hl'⟩

/-- Two matrices of propositionally equal dimensions with the same entries at every natural index. -/
def MatEq {α : Type} {p q p' q' : ℕ} (M : Matrix (Fin p) (Fin q) α) (M' : Matrix (Fin p') (Fin q') α) : Prop :=
  p = p' ∧ q = q' ∧
    ∀ (a l : ℕ) (ha : a < p) (hl : l < q) (ha' : a < p') (hl' : l < q'), M ⟨a, ha⟩ ⟨l, hl⟩ = M' ⟨a, ha'⟩ ⟨l, hl'⟩

theorem VecEq.eq {α : Type} {q : ℕ} {v v' : Fin q → α} (h : VecEq v v') : v = v' :=
  funext fun l => h.2 l l.2 l.2

theorem VecEq.of_eq {α : Type} {q : ℕ} {v v' : Fin q → α} (h : v = v') : VecEq v v' :=
  ⟨rfl, fun _ _ _ => h ▸ rfl⟩

theorem MatEq.eq {α : Type} {p q : ℕ} {M M' : Matrix (Fin p) (Fin q) α} (h : MatEq M M') : M = M' :=
  Matrix.ext fun a l => h.2.2 a l a.2 l.2 a.2 l.2

theorem MatEq.refl {α : Type} {p q : ℕ} (M : Matrix (Fin p) (Fin q) α) : MatEq M M :=
  ⟨rfl, rfl, fun _ _ _ _ _ _ => rfl⟩

theorem matEq_row {α : Type} {q q' : ℕ} {v : Fin q → α} {v' : Fin q' → α} (h : VecEq v v') :
    MatEq (fun _ : Fin 1 => v) (fun _ : Fin 1 => v') :=
  ⟨rfl, h.1, fun _ l _ hl _ hl' => h.2 l hl hl'⟩

theorem finFun_congr {β : Sort*} {F F' : (k : ℕ) → (Fin k → ℚ) → β} (hF : F = F') {k k' : ℕ} (hk : k = k')
    {f : Fin k → ℚ} {f' : Fin k' → ℚ} (hf : ∀ (l : ℕ) (hl : l < k) (hl' : l < k'), f ⟨l, hl⟩ = f' ⟨l, hl'⟩) :
    F k f = F' k' f' := by
  subst hF hk
  congr 1
  funext l
  exact hf l l.2 l.2

theorem MatEq.atNat_eq {p q p' q' : ℕ} {M : Matrix (Fin p) (Fin q) (Option ℚ)}
    {M' : Matrix (Fin p') (Fin q') (Option ℚ)} (h : MatEq M M') : atNat M = atNat M' := by
  have hp := h.1
  have hq := h.2.1
  subst hp hq
  rw [h.eq]

theorem MatEq.ticket (d : PearlCDevice) {m n k m' n' k' : ℕ} {A : Matrix (Fin m) (Fin k) Code}
    {A' : Matrix (Fin m') (Fin k') Code} {B : Matrix (Fin n) (Fin k) Code} {B' : Matrix (Fin n') (Fin k') Code}
    (hA : MatEq A A') (hB : MatEq B B') : MatEq (d.ticket A B) (d.ticket A' B') := by
  have hm := hA.1
  have hk := hA.2.1
  have hn := hB.1
  subst hm hk hn
  rw [hA.eq, hB.eq]
  exact MatEq.refl _

theorem MatEq.useful (d : PearlCDevice) {m n k m' n' k' : ℕ} {A : Matrix (Fin m) (Fin k) Code}
    {A' : Matrix (Fin m') (Fin k') Code} {B : Matrix (Fin n) (Fin k) Code} {B' : Matrix (Fin n') (Fin k') Code}
    {FA FB : Matrix (Fin k) (Fin 32) Code} {FA' FB' : Matrix (Fin k') (Fin 32) Code}
    {beA : Matrix (Fin m) (Fin 32) ℕ} {beA' : Matrix (Fin m') (Fin 32) ℕ}
    {beB : Matrix (Fin n) (Fin 32) ℕ} {beB' : Matrix (Fin n') (Fin 32) ℕ}
    (hA : MatEq A A') (hB : MatEq B B') (hFA : MatEq FA FA') (hFB : MatEq FB FB') (hbA : MatEq beA beA')
    (hbB : MatEq beB beB') : MatEq (d.useful A B FA FB beA beB) (d.useful A' B' FA' FB' beA' beB') := by
  have hm := hA.1
  have hk := hA.2.1
  have hn := hB.1
  subst hm hk hn
  rw [hA.eq, hB.eq, hFA.eq, hFB.eq, hbA.eq, hbB.eq]
  exact MatEq.refl _

theorem FormChain.elem_congr {r : ℕ} {fc fc' : FormChain r} (hfc : fc = fc') {a a' x x' b b' : ℚ}
    {e e' f f' : Fin r → Code} (ha : a = a') (hx : x = x') (hb : b = b') (he : e = e') (hf : f = f') :
    fc.elem a x b e f = fc'.elem a' x' b' e' f' := by
  subst hfc ha hx hb he hf; rfl

/-! ## Row views of the ticket and U -/

section Split

variable {Q R S : Type}

theorem unitChain_fst_apply (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes)
    (a : Fin (U.layout.shape u).m) (l : Fin (U.layout.shape u).k) :
    (sem.unitChain H s U u act).1 a l =
      sem.fc.elem (sem.alpha (U.layout.shape u).k (wordsVal (U.layout.shape u).m (U.layout.shape u).k act a))
        (wordsVal (U.layout.shape u).m (U.layout.shape u).k act a l)
        (sem.beta (U.layout.shape u).k (wordsVal (U.layout.shape u).m (U.layout.shape u).k act a))
        ((sem.noise H s U u act).EA a) ((sem.noise H s U u act).FA l) := rfl

theorem unitChain_snd_apply (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload) (u : ℕ) (act : Codes)
    (b : Fin (U.layout.shape u).n) (l : Fin (U.layout.shape u).k) :
    (sem.unitChain H s U u act).2 b l =
      sem.fc.elem
        (sem.alpha (U.layout.shape u).k
          (wordsVal (U.layout.shape u).n (U.layout.shape u).k (U.weight (U.layout.wid u)) b))
        (wordsVal (U.layout.shape u).n (U.layout.shape u).k (U.weight (U.layout.wid u)) b l)
        (sem.beta (U.layout.shape u).k
          (wordsVal (U.layout.shape u).n (U.layout.shape u).k (U.weight (U.layout.wid u)) b))
        ((sem.noise H s U u act).EB b) ((sem.noise H s U u act).FB l) := rfl

/-- The peel's BF16 noise row of one operand row, `bf16(E′)` with `E′ = lineCodes β e`, as `unitUsefulDev` reads it. -/
def beRow (β : ℚ) (e : Fin Params.pi.r → Code) : Fin 32 → ℕ := fun p => codeBf16 (lineCodes β e p)

theorem atNat_ticket_row {d : PearlCDevice} (hd : d.RowLocal) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) {i : ℕ} (hi : i < m) (j : ℕ) :
    atNat (d.ticket A B) i j = atNat (d.ticket (fun _ : Fin 1 => A ⟨i, hi⟩) B) 0 j := by
  unfold atNat
  by_cases hj : j < n
  · rw [dite_eq_left (And.intro hi hj), dite_eq_left (And.intro Nat.one_pos hj)]
    exact (hd.1 A B ⟨i, hi⟩ ⟨j, hj⟩).1
  · rw [dite_eq_right (fun h => hj h.2), dite_eq_right (fun h => hj h.2)]

theorem atNat_useful_row {d : PearlCDevice} (hd : d.RowLocal) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (FA FB : Matrix (Fin k) (Fin 32) Code) (beA : Matrix (Fin m) (Fin 32) ℕ)
    (beB : Matrix (Fin n) (Fin 32) ℕ) {i : ℕ} (hi : i < m) (j : ℕ) :
    atNat (d.useful A B FA FB beA beB) i j =
      atNat (d.useful (fun _ : Fin 1 => A ⟨i, hi⟩) B FA FB (fun _ => beA ⟨i, hi⟩) beB) 0 j := by
  unfold atNat
  by_cases hj : j < n
  · rw [dite_eq_left (And.intro hi hj), dite_eq_left (And.intro Nat.one_pos hj)]
    exact hd.2 A B FA FB beA beB ⟨i, hi⟩ ⟨j, hj⟩
  · rw [dite_eq_right (fun h => hj h.2), dite_eq_right (fun h => hj h.2)]

theorem useful_row {d : PearlCDevice} (hd : d.RowLocal) (sem : PearlCSem Q R S) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (act : Codes) {i : ℕ} (hi : i < (U.layout.shape u).m) (j : ℕ) :
    atNat (sem.unitUsefulDev d H s U u act) i j =
      atNat (d.useful (fun _ : Fin 1 => (sem.unitChain H s U u act).1 ⟨i, hi⟩) (sem.unitChain H s U u act).2
        (fun l p => (sem.noise H s U u act).FA l p) (fun l p => (sem.noise H s U u act).FB l p)
        (fun _ : Fin 1 => beRow (sem.beta (U.layout.shape u).k
          (wordsVal (U.layout.shape u).m (U.layout.shape u).k act ⟨i, hi⟩)) ((sem.noise H s U u act).EA ⟨i, hi⟩))
        (fun b => beRow (sem.beta (U.layout.shape u).k
          (wordsVal (U.layout.shape u).n (U.layout.shape u).k (U.weight (U.layout.wid u)) b))
            ((sem.noise H s U u act).EB b))) 0 j :=
  atNat_useful_row hd _ _ _ _ _ _ hi j

theorem splitWeight_row {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (b l : ℕ)
    (hb : b < (U.layout.shape u).n) (hl : l < (U.layout.shape u).k)
    (hb' : b < ((splitWorkload U).layout.shape (rowIdx U.layout u i)).n)
    (hl' : l < ((splitWorkload U).layout.shape (rowIdx U.layout u i)).k) :
    wordsVal (U.layout.shape u).n (U.layout.shape u).k (U.weight (U.layout.wid u)) ⟨b, hb⟩ ⟨l, hl⟩ =
      wordsVal ((splitWorkload U).layout.shape (rowIdx U.layout u i)).n
        ((splitWorkload U).layout.shape (rowIdx U.layout u i)).k
        ((splitWorkload U).weight ((splitWorkload U).layout.wid (rowIdx U.layout u i))) ⟨b, hb'⟩ ⟨l, hl'⟩ := by
  show (wordQ (U.weight (U.layout.wid u) b l)).getD 0 =
    (wordQ (U.weight (U.layout.wid (unitOfRow U.layout (rowIdx U.layout u i))) b l)).getD 0
  rw [unitOfRow_rowIdx hu hi]

variable {sem sem' : PearlCSem Q R S}

/-- Row `i` of unit `u`'s `A′` under the row seeds is the split unit's one row. -/
theorem splitRow_fst (hN : RowSeedNoise sem sem') (H : Q → R) (s : S) {U : Workload} {u i : ℕ}
    (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (act : Codes) :
    VecEq ((sem'.unitChain H s U u act).1 ⟨i, hi⟩)
      ((sem.unitChain H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i)).1 ⟨0, Nat.one_pos⟩) := by
  refine ⟨(splitShape_k hu hi).symm, fun l hl hl' => ?_⟩
  rw [unitChain_fst_apply, unitChain_fst_apply]
  have hz := hN.2.2.2.2 H s U u act hu i hi
  exact FormChain.elem_congr hN.1 (finFun_congr hN.2.1 (splitShape_k hu hi).symm fun _ _ _ => rfl) rfl
    (finFun_congr hN.2.2.1 (splitShape_k hu hi).symm fun _ _ _ => rfl) (funext hz.1)
    (funext fun t => (hz.2.1 l hl hl' t).1)

/-- Unit `u`'s `B̃` under the row seeds is the split unit's. -/
theorem splitRow_snd (hN : RowSeedNoise sem sem') (H : Q → R) (s : S) {U : Workload} {u i : ℕ}
    (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (act : Codes) :
    MatEq (sem'.unitChain H s U u act).2
      (sem.unitChain H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i)).2 := by
  refine ⟨(splitShape_n hu hi).symm, (splitShape_k hu hi).symm, fun b l hb hl hb' hl' => ?_⟩
  rw [unitChain_snd_apply, unitChain_snd_apply]
  have hz := hN.2.2.2.2 H s U u act hu i hi
  exact FormChain.elem_congr hN.1
    (finFun_congr hN.2.1 (splitShape_k hu hi).symm fun l hl hl' => splitWeight_row hu hi b l hb hl hb' hl')
    (splitWeight_row hu hi b l hb hl hb' hl')
    (finFun_congr hN.2.2.1 (splitShape_k hu hi).symm fun l hl hl' => splitWeight_row hu hi b l hb hl hb' hl')
    (funext (hz.2.2 b hb hb')) (funext fun t => (hz.2.1 l hl hl' t).2)

/-- **Row `i`'s ticket words** under the row seeds are the split unit's. -/
theorem split_ticket {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') (ρ : ℚ) (H : Q → R)
    (s : S) {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (act : Codes)
    (j : ℕ) :
    (pearlCProtocolDevRev1RowCap d sem' ρ).ticket H s U u act i j =
      (pearlCProtocolDevRev1 d sem ρ).ticket H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) 0 j := by
  refine (atNat_ticket_row hd _ _ hi j).trans ?_
  refine Eq.trans ?_ (atNat_ticket_row hd _ _ (i := 0) Nat.one_pos j).symm
  exact congrFun (congrFun (MatEq.atNat_eq
    (MatEq.ticket d (matEq_row (splitRow_fst hN H s hu hi act)) (splitRow_snd hN H s hu hi act))) 0) j

/-- **Row `i`'s useful words** under the row seeds are the split unit's. -/
theorem split_useful {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') (ρ : ℚ) (H : Q → R)
    (s : S) {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (act : Codes)
    (j : ℕ) :
    (pearlCProtocolDevRev1RowCap d sem' ρ).useful H s U u act i j =
      (pearlCProtocolDevRev1 d sem ρ).useful H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) 0 j := by
  refine (useful_row hd sem' H s U u act hi j).trans ?_
  refine Eq.trans ?_
    (useful_row hd sem H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) Nat.one_pos j).symm
  have hz := hN.2.2.2.2 H s U u act hu i hi
  have hk := (splitShape_k hu hi).symm
  refine congrFun (congrFun (MatEq.atNat_eq (MatEq.useful d (matEq_row (splitRow_fst hN H s hu hi act))
    (splitRow_snd hN H s hu hi act) ?_ ?_ ?_ ?_)) 0) j
  · exact ⟨hk, rfl, fun l p hl hp hl' _ => (hz.2.1 l hl hl' ⟨p, hp⟩).1⟩
  · exact ⟨hk, rfl, fun l p hl hp hl' _ => (hz.2.1 l hl hl' ⟨p, hp⟩).2⟩
  · exact matEq_row (VecEq.of_eq (congrArg₂ beRow (finFun_congr hN.2.2.1 hk fun _ _ _ => rfl) (funext hz.1)))
  · refine ⟨(splitShape_n hu hi).symm, rfl, fun b p hb hp hb' _ => ?_⟩
    exact congrFun (congrArg₂ beRow
      (finFun_congr hN.2.2.1 hk fun l hl hl' => splitWeight_row hu hi b l hb hl hb' hl')
      (funext (hz.2.2 b hb hb'))) ⟨p, hp⟩

/-! ## The debit, row by row -/

open Classical in
/-- rev1's replayed debit of row `i` of a unit's words in the columns `cols`, from the unit's noisy codes `A`, `B` and
the row's activations `x`: `tileDebitRev1`'s terms on the one-row tile `{i} × cols`. -/
noncomputable def rowDebit (d : PearlCDevice) (α β : (k : ℕ) → (Fin k → ℚ) → ℚ) {m n k : ℕ}
    (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (cols : Finset ℕ)
    (x : Fin k → ℚ) : ℚ :=
  debitDev d ⟨∑ j ∈ univ.filter (fun j : Fin n => (j : ℕ) ∈ cols),
      ((range (k / 32)).filter fun t => d.flags A B i j t = true).card, 0⟩ +
    (d.prices.costs.fs : ℚ) * (cols.card / n) * (univ.filter fun l : Fin k => d.saltDead α β k x l ∨
      ∀ j : Fin n, (j : ℕ) ∈ cols → d.flags A B i j ((l : ℕ) / 32) = true).card +
    d.prices.add * ∑ j ∈ univ.filter (fun j : Fin n => (j : ℕ) ∈ cols), (laterZeroPromos d A B i j : ℚ)

theorem sum_filter_row {M : Type} [AddCommMonoid M] {m n : ℕ} (g : Fin m × Fin n → M) {i : ℕ} (hi : i < m)
    (P : Fin m → Fin n → Prop) {_ : DecidablePred fun e : Fin m × Fin n => (e.1 : ℕ) ∈ ({i} : Finset ℕ) ∧ P e.1 e.2}
    [DecidablePred (P ⟨i, hi⟩)] :
    ∑ e ∈ univ.filter (fun e : Fin m × Fin n => (e.1 : ℕ) ∈ ({i} : Finset ℕ) ∧ P e.1 e.2), g e =
      ∑ j ∈ univ.filter (P ⟨i, hi⟩), g (⟨i, hi⟩, j) := by
  rw [sum_filter, sum_filter, Fintype.sum_prod_type, sum_eq_single ⟨i, hi⟩]
  · exact sum_congr rfl fun j _ => if_congr (by simp) rfl rfl
  · intro b _ hb
    exact sum_eq_zero fun j _ => ite_eq_right fun h => hb (Fin.ext (mem_singleton.mp h.1))
  · intro h; exact absurd (mem_univ _) h

theorem card_filter_row {m k : ℕ} {i : ℕ} (hi : i < m) (P : Fin m → Fin k → Prop)
    {_ : DecidablePred fun e : Fin m × Fin k => (e.1 : ℕ) ∈ ({i} : Finset ℕ) ∧ P e.1 e.2}
    [DecidablePred (P ⟨i, hi⟩)] :
    (univ.filter fun e : Fin m × Fin k => (e.1 : ℕ) ∈ ({i} : Finset ℕ) ∧ P e.1 e.2).card =
      (univ.filter (P ⟨i, hi⟩)).card := by
  rw [card_eq_sum_ones, card_eq_sum_ones]
  exact sum_filter_row (fun _ => 1) hi P

theorem card_filter_prod {m k : ℕ} (p : Fin m × Fin k → Prop) (q : Fin m → Fin k → Prop) {_ : DecidablePred p}
    [∀ i, DecidablePred (q i)] (h : ∀ i l, p (i, l) ↔ q i l) :
    (univ.filter p).card = ∑ i, (univ.filter (q i)).card := by
  rw [card_filter, Fintype.sum_prod_type]
  refine sum_congr rfl fun i _ => ?_
  rw [card_filter]
  exact sum_congr rfl fun l _ => if_congr (h i l) rfl rfl

/-- **A tile's debit on one row** is the row's debit. -/
theorem tileDebitRev1_row (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) {i : ℕ} (hi : i < (U.layout.shape u).m) (cols : Finset ℕ) :
    sem.tileDebitRev1 d H s U u {i} cols act =
      rowDebit d sem.alpha sem.beta (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 ⟨i, hi⟩ cols
        (wordsVal (U.layout.shape u).m (U.layout.shape u).k act ⟨i, hi⟩) := by
  classical
  simp only [PearlCSem.tileDebitRev1, PearlCSem.tileDebitDev, rowDebit]
  rw [sum_filter_row _ hi (fun _ b => (b : ℕ) ∈ cols), sum_filter_row _ hi (fun _ b => (b : ℕ) ∈ cols),
    card_filter_row hi (fun a l => d.saltDead sem.alpha sem.beta (U.layout.shape u).k
      (wordsVal (U.layout.shape u).m (U.layout.shape u).k act a) l ∨ ∀ j : Fin (U.layout.shape u).n,
        (j : ℕ) ∈ cols → d.flags (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 a j ((l : ℕ) / 32) = true)]

/-- **A unit's debit is the sum of its rows' debits** over all its columns. -/
theorem unitDebitRev1_eq_sum_rows (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (act : Codes) (hn : 0 < (U.layout.shape u).n) :
    sem.unitDebitRev1 d H s U u act =
      ∑ i, rowDebit d sem.alpha sem.beta (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 i
        (range (U.layout.shape u).n) (wordsVal (U.layout.shape u).m (U.layout.shape u).k act i) := by
  classical
  have hf : (univ.filter fun j : Fin (U.layout.shape u).n => (j : ℕ) ∈ range (U.layout.shape u).n) = univ :=
    filter_true_of_mem fun j _ => mem_range.mpr j.2
  have hc : ((range (U.layout.shape u).n).card : ℚ) / (U.layout.shape u).n = 1 := by
    rw [card_range]; exact div_self (by exact_mod_cast hn.ne')
  simp only [PearlCSem.unitDebitRev1, PearlCSem.unitDebitDev, rowDebit, debitDev, hf, hc]
  rw [card_filter_prod _ (fun a l => d.saltDead sem.alpha sem.beta (U.layout.shape u).k
      (wordsVal (U.layout.shape u).m (U.layout.shape u).k act a) l ∨ ∀ j : Fin (U.layout.shape u).n,
        (j : ℕ) ∈ range (U.layout.shape u).n →
          d.flags (sem.unitChain H s U u act).1 (sem.unitChain H s U u act).2 a j ((l : ℕ) / 32) = true)
    (fun a l => by simp [Skip.needed])]
  push_cast
  simp only [sum_add_distrib, ← mul_sum, mul_one, mul_zero, add_zero]

theorem laterZeroPromos_row {d : PearlCDevice} (hd : d.RowLocal) {m n k : ℕ} (A : Matrix (Fin m) (Fin k) Code)
    (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (j : Fin n) :
    laterZeroPromos d A B i j = laterZeroPromos d (fun _ : Fin 1 => A i) B 0 j := by
  unfold laterZeroPromos
  split_ifs
  · rfl
  · exact congrArg card (filter_congr fun g _ => exists_congr fun h => by
      rw [(hd.1 (prefixCols A _ h) (prefixCols B _ h) i j).1]; rfl)

open Classical in
/-- A row's debit reads that row of `A` only, at a row-local record. -/
theorem rowDebit_row {d : PearlCDevice} (hd : d.RowLocal) (α β : (k : ℕ) → (Fin k → ℚ) → ℚ) {m n k : ℕ}
    (A : Matrix (Fin m) (Fin k) Code) (B : Matrix (Fin n) (Fin k) Code) (i : Fin m) (cols : Finset ℕ)
    (x : Fin k → ℚ) :
    rowDebit d α β A B i cols x = rowDebit d α β (fun _ : Fin 1 => A i) B 0 cols x := by
  have hf : ∀ j t, d.flags A B i j t = d.flags (fun _ : Fin 1 => A i) B 0 j t := fun j t => (hd.1 A B i j).2 t
  simp only [rowDebit, hf, laterZeroPromos_row hd A B i]

theorem rowDebit_congr (d : PearlCDevice) (α β : (k : ℕ) → (Fin k → ℚ) → ℚ) {n k n' k' : ℕ}
    {A : Matrix (Fin 1) (Fin k) Code} {A' : Matrix (Fin 1) (Fin k') Code} {B : Matrix (Fin n) (Fin k) Code}
    {B' : Matrix (Fin n') (Fin k') Code} (cols : Finset ℕ) {x : Fin k → ℚ} {x' : Fin k' → ℚ} (hA : MatEq A A')
    (hB : MatEq B B') (hx : VecEq x x') : rowDebit d α β A B 0 cols x = rowDebit d α β A' B' 0 cols x' := by
  have hk := hA.2.1
  have hn := hB.1
  subst hk hn
  rw [hA.eq, hB.eq, hx.eq]

/-- **Row `i`'s debit** under the row seeds, on any columns, is the split unit's. -/
theorem split_tileDebit {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') (H : Q → R) (s : S)
    {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (cols : Finset ℕ)
    (act : Codes) :
    sem.tileDebitRev1 d H s (splitWorkload U) (rowIdx U.layout u i) {0} cols (rowCodes act i) =
      sem'.tileDebitRev1 d H s U u {i} cols act := by
  rw [tileDebitRev1_row sem d H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) (i := 0) Nat.one_pos,
    tileDebitRev1_row sem' d H s U u act hi,
    rowDebit_row hd sem.alpha sem.beta (sem.unitChain H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i)).1,
    rowDebit_row hd sem'.alpha sem'.beta (sem'.unitChain H s U u act).1, hN.2.1, hN.2.2.1]
  exact (rowDebit_congr d _ _ cols (matEq_row (splitRow_fst hN H s hu hi act)) (splitRow_snd hN H s hu hi act)
    ⟨(splitShape_k hu hi).symm, fun _ _ _ => rfl⟩).symm

/-- A unit's debit is the sum of its one-row tiles' debits. -/
theorem unitDebitRev1_eq_sum_tiles (sem : PearlCSem Q R S) (d : PearlCDevice) (H : Q → R) (s : S) (U : Workload)
    (u : ℕ) (act : Codes) (hn : 0 < (U.layout.shape u).n) :
    sem.unitDebitRev1 d H s U u act =
      ∑ i ∈ range (U.layout.shape u).m, sem.tileDebitRev1 d H s U u {i} (range (U.layout.shape u).n) act := by
  rw [unitDebitRev1_eq_sum_rows sem d H s U u act hn, ← Fin.sum_univ_eq_sum_range]
  exact sum_congr rfl fun i _ => (tileDebitRev1_row sem d H s U u act i.2 _).symm

/-- **The split unit's debit is row `i`'s debit** under the row seeds. -/
theorem split_debit {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') (H : Q → R) (s : S)
    {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (hn : 0 < (U.layout.shape u).n)
    (act : Codes) :
    sem.unitDebitRev1 d H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) =
      sem'.tileDebitRev1 d H s U u {i} (range (U.layout.shape u).n) act := by
  rw [unitDebitRev1_eq_sum_tiles sem d H s _ _ _ ((splitShapeW_n hu hi).symm ▸ hn)]
  refine (sum_range_one _).trans ?_
  rw [splitShapeW_n hu hi]
  exact split_tileDebit hd hN H s hu hi _ act

end Split

/-! ## The credit, row by row -/

theorem creditDevRev1_rows (d : PearlCDevice) (sh : Shape) :
    creditDevRev1 d sh = sh.m * creditDevRev1 d ⟨1, sh.n, sh.k⟩ := by
  simp only [creditDevRev1, creditDev, firstAddDev]
  split_ifs <;> push_cast <;> ring

/-- A row's share of the credit is the one-row unit's credit. -/
theorem rowShare_credit (d : PearlCDevice) (ρ : ℚ) (sh : Shape) (i : ℕ) (hm : 0 < sh.m) (hn : 0 < sh.n) :
    ρ * creditDevRev1 d sh * tileShare sh {i} (range sh.n) = ρ * creditDevRev1 d ⟨1, sh.n, sh.k⟩ := by
  rw [creditDevRev1_rows d sh, tileShare, card_singleton, card_range, Nat.cast_one]
  have hm' : (sh.m : ℚ) ≠ 0 := by exact_mod_cast hm.ne'
  have hn' : (sh.n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  field_simp

/-! ## Correct rows -/

section Reduction

variable {Q R S : Type} {sem sem' : PearlCSem Q R S}

theorem rowOK_of_floor (h : sem'.floor = sem.floor) {p q i : ℕ} {W : Codes} (hW : sem'.RowOK p q W i) :
    sem.RowOK p q W i :=
  ⟨hW.1, fun hi => ⟨(hW.2 hi).1, h ▸ (hW.2 hi).2⟩⟩

/-- A row of admissible activations is admissible as its own unit. -/
theorem split_actOK (hN : RowSeedNoise sem sem') {U : Workload} {u i : ℕ} (hu : u < U.layout.N)
    (hi : i < (U.layout.shape u).m) {act : Codes} (h : sem'.OperandOK (U.layout.shape u).m (U.layout.shape u).k act) :
    sem.OperandOK ((splitWorkload U).layout.shape (rowIdx U.layout u i)).m
      ((splitWorkload U).layout.shape (rowIdx U.layout u i)).k (rowCodes act i) := by
  have hk := splitShapeW_k hu hi
  refine ⟨fun a ha l hl => ?_, fun a => ?_⟩
  · have ha1 : a < 1 := ha
    have ha0 : a = 0 := by omega
    subst ha0
    exact h.1 i hi l (by rw [← hk]; exact hl)
  · exact Eq.mp (finFun_congr (F := fun k x => LiveRow Params.pi.δ x Stat.every8 ∧ 1 ≤ sem'.floor k x)
      (F' := fun k x => LiveRow Params.pi.δ x Stat.every8 ∧ 1 ≤ sem.floor k x) (by rw [hN.2.2.2.1]) hk.symm
      (f := wordsVal (U.layout.shape u).m (U.layout.shape u).k act ⟨i, hi⟩)
      (f' := wordsVal _ _ (rowCodes act i) a) fun _ _ _ => rfl) (h.2 ⟨i, hi⟩)

/-- **The per-row cap implies rev1's cap.** -/
theorem rowCap_agg (sem : PearlCSem Q R S) (d : PearlCDevice) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (u : ℕ)
    (act : Codes) (hm : 0 < (U.layout.shape u).m) (hn : 0 < (U.layout.shape u).n)
    (h : (pearlCProtocolDevRev1RowCap d sem ρ).capOK H s U u act) :
    (pearlCProtocolDevRev1 d sem ρ).capOK H s U u act := by
  show sem.unitDebitRev1 d H s U u act ≤ ρ * creditDevRev1 d (U.layout.shape u)
  rw [unitDebitRev1_eq_sum_tiles sem d H s U u act hn]
  calc ∑ i ∈ range (U.layout.shape u).m, sem.tileDebitRev1 d H s U u {i} (range (U.layout.shape u).n) act
      ≤ ∑ _i ∈ range (U.layout.shape u).m,
          ρ * creditDevRev1 d ⟨1, (U.layout.shape u).n, (U.layout.shape u).k⟩ :=
        sum_le_sum fun i hi => (h i (mem_range.mp hi)).trans_eq (rowShare_credit d ρ _ i hm hn)
    _ = ρ * creditDevRev1 d (U.layout.shape u) := by
        rw [sum_const, card_range, nsmul_eq_mul, creditDevRev1_rows d (U.layout.shape u)]; ring

/-- A weight row of a unit passes as a weight row of each of its rows' split units. -/
theorem split_weightOK (hN : RowSeedNoise sem sem') {d : PearlCDevice} (ρ : ℚ) {U : Workload} {u i j : ℕ}
    (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m)
    (h : (pearlCProtocolDevRev1RowCap d sem' ρ).weightOK (U.layout.shape u) (U.weight (U.layout.wid u)) j) :
    (pearlCProtocolDevRev1 d sem ρ).weightOK ((splitWorkload U).layout.shape (rowIdx U.layout u i))
      ((splitWorkload U).weight ((splitWorkload U).layout.wid (rowIdx U.layout u i))) j := by
  show sem.RowOK (U.layout.shape (unitOfRow U.layout (rowIdx U.layout u i))).n
    (U.layout.shape (unitOfRow U.layout (rowIdx U.layout u i))).k
    (U.weight (U.layout.wid (unitOfRow U.layout (rowIdx U.layout u i)))) j
  rw [unitOfRow_rowIdx hu hi]
  exact rowOK_of_floor hN.2.2.2.1 h

/-- **A row of a correct unit is a correct unit of the split workload.** -/
theorem split_correct {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') (ρ : ℚ) (H : Q → R) (s : S)
    {U : Workload} {u i : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m) (hn : 0 < (U.layout.shape u).n)
    {c : Commit} (hc : Correct (pearlCProtocolDevRev1RowCap d sem' ρ) U H s u c) :
    Correct (pearlCProtocolDevRev1 d sem ρ) (splitWorkload U) H s (rowIdx U.layout u i) (rowCommit c i) := by
  refine ⟨split_actOK hN hu hi hc.1, ?_,
    fun j hj => split_weightOK hN ρ hu hi (hc.2.2.1 j (by rw [← splitShapeW_n hu hi]; exact hj)), fun a ha j hj => ?_⟩
  · show sem.unitDebitRev1 d H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes c.act i) ≤
      ρ * creditDevRev1 d ((splitWorkload U).layout.shape (rowIdx U.layout u i))
    rw [split_debit hd hN H s hu hi hn, splitShapeW_eq hu hi, ← rowShare_credit d ρ _ i (by omega) hn]
    exact hc.2.1 i hi
  · have ha1 : a < 1 := ha
    have ha0 : a = 0 := by omega
    subst ha0
    have hj' : j < (U.layout.shape u).n := by rw [← splitShapeW_n hu hi]; exact hj
    exact ⟨(hc.2.2.2 i hi j hj').1.trans (split_ticket hd hN ρ H s hu hi c.act j),
      (hc.2.2.2 i hi j hj').2.trans (split_useful hd hN ρ H s hu hi c.act j)⟩

theorem splitTranscript_rowIdx {L : Layout} (τ : Transcript) {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    splitTranscript L τ (rowIdx L u i) = (τ u).map fun c => rowCommit c i := by
  unfold splitTranscript
  rw [ite_eq_left (rowIdx_lt_rowsOf hu hi), unitOfRow_rowIdx hu hi, rowOfRow_rowIdx hu hi]

theorem actOf_split {L : Layout} (τ : Transcript) {u i : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) :
    actOf (splitTranscript L τ) (rowIdx L u i) = rowCodes (actOf τ u) i := by
  unfold actOf
  rw [splitTranscript_rowIdx τ hu hi]
  rcases τ u with _ | c <;> rfl

/-! ## The credit, row by row, and the reduction -/

/-- **A unit's credit is the sum of its rows' credits as split units.** -/
theorem split_credit {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') (ρ : ℚ) (H : Q → R) (s : S)
    {U : Workload} {u : ℕ} (hu : u < U.layout.N) (hn : 0 < (U.layout.shape u).n) (act : Codes) :
    (pearlCProtocolDevRev1RowCap d sem' ρ).credit H s U u act =
      ∑ i ∈ range (U.layout.shape u).m,
        (pearlCProtocolDevRev1 d sem ρ).credit H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) := by
  calc (pearlCProtocolDevRev1RowCap d sem' ρ).credit H s U u act
      = ((((U.layout.shape u).m : ℚ) * creditDevRev1 d ⟨1, (U.layout.shape u).n, (U.layout.shape u).k⟩ -
          ∑ i ∈ range (U.layout.shape u).m, sem'.tileDebitRev1 d H s U u {i} (range (U.layout.shape u).n) act :
            ℚ) : ℝ) := by
        show ((creditDevRev1 d (U.layout.shape u) - sem'.unitDebitRev1 d H s U u act : ℚ) : ℝ) = _
        rw [creditDevRev1_rows d (U.layout.shape u), unitDebitRev1_eq_sum_tiles sem' d H s U u act hn]
    _ = ∑ i ∈ range (U.layout.shape u).m, ((creditDevRev1 d ⟨1, (U.layout.shape u).n, (U.layout.shape u).k⟩ -
          sem'.tileDebitRev1 d H s U u {i} (range (U.layout.shape u).n) act : ℚ) : ℝ) := by
        rw [← Rat.cast_sum, sum_sub_distrib, sum_const, card_range, nsmul_eq_mul]
    _ = _ := sum_congr rfl fun i hi => by
        show _ = ((creditDevRev1 d ((splitWorkload U).layout.shape (rowIdx U.layout u i)) -
          sem.unitDebitRev1 d H s (splitWorkload U) (rowIdx U.layout u i) (rowCodes act i) : ℚ) : ℝ)
        rw [splitShapeW_eq hu (mem_range.mp hi), split_debit hd hN H s hu (mem_range.mp hi) hn act]

open Classical in
/-- **The row-seeded credit is at most the split game's**, on every transcript. -/
theorem split_credit_le {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') {ρ : ℚ} (hρ : ρ ≤ 1)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) {U : Workload} (hU : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Transcript) :
    ∑ u ∈ correctSet (pearlCProtocolDevRev1RowCap d sem' ρ) U H s τ,
        (pearlCProtocolDevRev1RowCap d sem' ρ).credit H s U u (actOf τ u) ≤
      ∑ v ∈ correctSet (pearlCProtocolDevRev1 d sem ρ) (splitWorkload U) H s (splitTranscript U.layout τ),
        (pearlCProtocolDevRev1 d sem ρ).credit H s (splitWorkload U) v (actOf (splitTranscript U.layout τ) v) := by
  have hmem : ∀ u ∈ correctSet (pearlCProtocolDevRev1RowCap d sem' ρ) U H s τ, u < U.layout.N ∧
      ∃ c, τ u = some c ∧ Correct (pearlCProtocolDevRev1RowCap d sem' ρ) U H s u c := fun u hu => by
    have h := mem_filter.mp hu
    exact ⟨mem_range.mp h.1, h.2⟩
  have h1 : ∀ u ∈ correctSet (pearlCProtocolDevRev1RowCap d sem' ρ) U H s τ,
      (pearlCProtocolDevRev1RowCap d sem' ρ).credit H s U u (actOf τ u) =
        ∑ i ∈ range (U.layout.shape u).m, (pearlCProtocolDevRev1 d sem ρ).credit H s (splitWorkload U)
          (rowIdx U.layout u i) (actOf (splitTranscript U.layout τ) (rowIdx U.layout u i)) := by
    intro u hu
    have hu' := (hmem u hu).1
    rw [split_credit hd hN ρ H s hu' (pearlCDomainDev_pos hU hu').2 (actOf τ u)]
    exact sum_congr rfl fun i hi => by rw [actOf_split τ hu' (mem_range.mp hi)]
  have hinj : Set.InjOn (fun p : (_ : ℕ) × ℕ => rowIdx U.layout p.1 p.2)
      ((correctSet (pearlCProtocolDevRev1RowCap d sem' ρ) U H s τ).sigma fun u => range (U.layout.shape u).m) := by
    rintro ⟨u, i⟩ hp ⟨u', i'⟩ hp' h
    obtain ⟨hu, hi⟩ := mem_sigma.mp hp
    obtain ⟨hu', hi'⟩ := mem_sigma.mp hp'
    obtain ⟨rfl, rfl⟩ := rowIdx_inj (hmem u hu).1 (mem_range.mp hi) (hmem u' hu').1 (mem_range.mp hi') h
    rfl
  rw [sum_congr rfl h1, sum_sigma']
  refine (sum_image (f := fun v => (pearlCProtocolDevRev1 d sem ρ).credit H s (splitWorkload U) v
    (actOf (splitTranscript U.layout τ) v)) hinj).symm.trans_le (sum_le_sum_of_subset_of_nonneg ?_ ?_)
  · intro v hv
    obtain ⟨⟨u, i⟩, hp, rfl⟩ := mem_image.mp hv
    obtain ⟨hu, hi⟩ := mem_sigma.mp hp
    obtain ⟨huN, c, hc, hcor⟩ := hmem u hu
    have hi' := mem_range.mp hi
    refine mem_filter.mpr ⟨mem_range.mpr (rowIdx_lt_rowsOf huN hi'), rowCommit c i, ?_,
      split_correct hd hN ρ H s huN hi' (pearlCDomainDev_pos hU huN).2 hcor⟩
    rw [splitTranscript_rowIdx τ huN hi', hc]
    rfl
  · intro v hv _
    obtain ⟨hvr, c, hc, _, hcap, _⟩ := mem_filter.mp hv
    have hsp := pearlCDomainDev_split hU v (mem_range.mp hvr)
    have hcr := creditDevRev1_nonneg d hadd hbf ((splitWorkload U).layout.shape v) hsp.2.2.1 hsp.2.2.2.2.1
    have hact : actOf (splitTranscript U.layout τ) v = c.act := by simp [actOf, hc]
    rw [hact]
    have h2 : sem.unitDebitRev1 d H s (splitWorkload U) v c.act ≤
        ρ * creditDevRev1 d ((splitWorkload U).layout.shape v) := hcap
    show (0 : ℝ) ≤ ((creditDevRev1 d ((splitWorkload U).layout.shape v) -
      sem.unitDebitRev1 d H s (splitWorkload U) v c.act : ℚ) : ℝ)
    have h3 : (0 : ℚ) ≤ creditDevRev1 d ((splitWorkload U).layout.shape v) -
        sem.unitDebitRev1 d H s (splitWorkload U) v c.act :=
      sub_nonneg.mpr (h2.trans (mul_le_of_le_one_left hcr hρ))
    exact_mod_cast h3

/-- **Row additivity for P2 (a theorem).** rev1's TT_OUT at `sem` bounds the row-seeded game at `sem'`, with the cap
checked per row: the row-seeded adversary is the split game's, relabelled row by row (C3), each row of a correct unit is
a correct one-row unit whose credit is the row's (C0, C1), and the error is the split layout's, at `R = Σ_u m_u` units.
One-row units are in the domain (`pearlCDomainDev_split`). -/
theorem ttOutRowSeed_of_ttOut [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {d : PearlCDevice} {ρ : ℚ} (hTT : TTOutPearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) :
    TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCDomainDev d) (1 / 400) := by
  refine ⟨hTT.1, fun U hU hR T hT q hq A hA => ?_⟩
  obtain ⟨prog', act', hrun⟩ := hCM A.online U A.act
  have hA' : (⟨A.pre, prog', act'⟩ : Adversary CM).Bounded (splitWorkload U) T q := fun H s =>
    ⟨(hrun H (A.pre.eval H) s).2.1.trans (hA H s).1,
      le_trans (Nat.add_le_add_left (hrun H (A.pre.eval H) s).2.2 _) (hA H s).2⟩
  have hU' : (splitWorkload U).InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) :=
    ⟨hR, pearlCDomainDev_split hU.2⟩
  refine le_trans (Pouw.pr_mono fun x hx => lt_of_lt_of_le hx ?_)
    (hTT.2 (splitWorkload U) hU' T hT q hq ⟨A.pre, prog', act'⟩ hA')
  have htr : (⟨A.pre, prog', act'⟩ : Adversary CM).transcript (splitWorkload U) x.1 x.2 =
      splitTranscript U.layout (A.transcript U x.1 x.2) := (hrun x.1 (A.pre.eval x.1) x.2).1
  rw [htr]
  exact split_credit_le hd hN hρ hadd hbf hU.2 x.1 x.2 (A.transcript U x.1 x.2)

end Reduction

/-! ## The audit's grid on the split layout -/

theorem mem_gridTiles {tm tn : ℕ} {L : Layout} {u bi bj : ℕ} :
    (u, bi, bj) ∈ gridTiles tm tn L ↔
      u < L.N ∧ bi < ((L.shape u).m + tm - 1) / tm ∧ bj < ((L.shape u).n + tn - 1) / tn := by
  unfold gridTiles
  simp only [List.mem_flatMap, List.mem_range, List.mem_map, Prod.mk.injEq]
  constructor
  · rintro ⟨u', hu, bi', hbi, bj', hbj, rfl, rfl, rfl⟩
    exact ⟨hu, hbi, hbj⟩
  · rintro ⟨hu, hbi, hbj⟩
    exact ⟨u, hu, bi, hbi, bj, hbj, rfl, rfl, rfl⟩

theorem gridTiles_nodup (tm tn : ℕ) (L : Layout) : (gridTiles tm tn L).Nodup := by
  unfold gridTiles
  refine List.nodup_flatMap.mpr ⟨fun u _ => List.nodup_flatMap.mpr ⟨fun bi _ => List.nodup_range.map ?_,
    List.pairwise_lt_range.imp fun {a b} hab => ?_⟩, List.pairwise_lt_range.imp fun {a b} hab => ?_⟩
  · intro x y h
    simpa using h
  · intro x hx hy
    simp only [List.mem_map, List.mem_range] at hx hy
    obtain ⟨_, _, rfl⟩ := hx
    obtain ⟨_, _, h⟩ := hy
    simp only [Prod.mk.injEq] at h
    omega
  · intro x hx hy
    simp only [List.mem_flatMap, List.mem_map, List.mem_range] at hx hy
    obtain ⟨_, _, _, _, rfl⟩ := hx
    obtain ⟨_, _, _, _, h⟩ := hy
    simp only [Prod.mk.injEq] at h
    omega

theorem gridTiles_getD {tm tn : ℕ} {L : Layout} {g : ℕ} (hg : g < (auditTiling tm tn L).NT) :
    (gridTiles tm tn L).getD g (0, 0, 0) = (gridTiles tm tn L)[g]'hg := by
  rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hg, Option.getD_some]

theorem gridTiles_getD_mem {tm tn : ℕ} {L : Layout} {g : ℕ} (hg : g < (auditTiling tm tn L).NT) :
    (gridTiles tm tn L).getD g (0, 0, 0) ∈ gridTiles tm tn L := by
  rw [gridTiles_getD hg]
  exact List.getElem_mem hg

theorem gridTiles_getD_inj {tm tn : ℕ} {L : Layout} {g g' : ℕ} (hg : g < (auditTiling tm tn L).NT)
    (hg' : g' < (auditTiling tm tn L).NT)
    (h : (gridTiles tm tn L).getD g (0, 0, 0) = (gridTiles tm tn L).getD g' (0, 0, 0)) : g = g' := by
  rw [gridTiles_getD hg, gridTiles_getD hg'] at h
  exact (gridTiles_nodup tm tn L).getElem_inj_iff.mp h

/-- The split tile of row `i` of block column `bj` of unit `u`: the index of the one-row block `(rowIdx u i, 0, bj)` in
the split layout's grid. -/
def splitTile (L : Layout) (u i bj : ℕ) : ℕ := (gridTiles 64 64 (splitLayout L)).idxOf (rowIdx L u i, 0, bj)

section SplitTile

variable {L : Layout} {u i bj : ℕ} (hu : u < L.N) (hi : i < (L.shape u).m) (hbj : bj < ((L.shape u).n + 64 - 1) / 64)
include hu hi hbj

theorem splitTile_mem : (rowIdx L u i, 0, bj) ∈ gridTiles 64 64 (splitLayout L) :=
  mem_gridTiles.mpr ⟨rowIdx_lt_rowsOf hu hi, by show 0 < (1 + 64 - 1) / 64; norm_num,
    by rw [splitShape_n hu hi]; exact hbj⟩

theorem splitTile_lt : splitTile L u i bj < (auditTiling 64 64 (splitLayout L)).NT :=
  List.idxOf_lt_length_of_mem (splitTile_mem hu hi hbj)

theorem splitTile_getD :
    (gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0) = (rowIdx L u i, 0, bj) := by
  rw [gridTiles_getD (splitTile_lt hu hi hbj)]
  exact List.getElem_idxOf (splitTile_lt hu hi hbj)

theorem splitTile_unit : (auditTiling 64 64 (splitLayout L)).unit (splitTile L u i bj) = rowIdx L u i := by
  show ((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).1 = _
  rw [splitTile_getD hu hi hbj]

theorem splitTile_rows : (auditTiling 64 64 (splitLayout L)).rows (splitTile L u i bj) = {0} := by
  show Ico (((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).2.1 * 64)
    (min ((((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).2.1 + 1) * 64)
      ((splitLayout L).shape ((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).1).m) = _
  rw [splitTile_getD hu hi hbj]
  show Ico (0 * 64) (min ((0 + 1) * 64) 1) = {0}
  decide

theorem splitTile_cols : (auditTiling 64 64 (splitLayout L)).cols (splitTile L u i bj) =
    Ico (bj * 64) (min ((bj + 1) * 64) (L.shape u).n) := by
  show Ico (((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).2.2 * 64)
    (min ((((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).2.2 + 1) * 64)
      ((splitLayout L).shape ((gridTiles 64 64 (splitLayout L)).getD (splitTile L u i bj) (0, 0, 0)).1).n) = _
  rw [splitTile_getD hu hi hbj, splitShape_n hu hi]

end SplitTile

section TileReduction

variable {Q R S : Type} {sem sem' : PearlCSem Q R S}

/-- A row that passes Π's rules passes them as its own unit. -/
theorem split_rowOK (hN : RowSeedNoise sem sem') {U : Workload} {u i : ℕ} (hu : u < U.layout.N)
    (hi : i < (U.layout.shape u).m) {act : Codes} (h : sem'.RowOK (U.layout.shape u).m (U.layout.shape u).k act i) :
    sem.RowOK ((splitWorkload U).layout.shape (rowIdx U.layout u i)).m
      ((splitWorkload U).layout.shape (rowIdx U.layout u i)).k (rowCodes act i) 0 := by
  have hk := splitShapeW_k hu hi
  refine ⟨fun l hl => h.1 l (by rw [← hk]; exact hl), fun h0 => ?_⟩
  exact Eq.mp (finFun_congr (F := fun k x => LiveRow Params.pi.δ x Stat.every8 ∧ 1 ≤ sem'.floor k x)
    (F' := fun k x => LiveRow Params.pi.δ x Stat.every8 ∧ 1 ≤ sem.floor k x) (by rw [hN.2.2.2.1]) hk.symm
    (f := wordsVal (U.layout.shape u).m (U.layout.shape u).k act ⟨i, hi⟩)
    (f' := wordsVal _ _ (rowCodes act i) ⟨0, h0⟩) fun _ _ _ => rfl) (h.2 hi)

theorem split_passRows (hN : RowSeedNoise sem sem') {U : Workload} {u i : ℕ} (hu : u < U.layout.N)
    (hi : i < (U.layout.shape u).m) {act : Codes} (h : sem'.RowOK (U.layout.shape u).m (U.layout.shape u).k act i) :
    sem.passRows ((splitWorkload U).layout.shape (rowIdx U.layout u i)) (rowCodes act i) {0} = {0} := by
  classical
  unfold PearlCSem.passRows
  rw [filter_singleton, ite_eq_left (split_rowOK hN hu hi h)]

/-- A tile's credit is its credited rows' one-row credits. -/
theorem tileCredit_rows (d : PearlCDevice) (ρ : ℚ) (sh : Shape) (hm : 0 < sh.m) (hn : 0 < sh.n)
    (rows cols : Finset ℕ) :
    (1 - ρ) * creditDevRev1 d sh * tileShare sh rows cols =
      ∑ _i ∈ rows, (1 - ρ) * creditDevRev1 d ⟨1, sh.n, sh.k⟩ * tileShare ⟨1, sh.n, sh.k⟩ {0} cols := by
  rw [sum_const, nsmul_eq_mul, creditDevRev1_rows d sh]
  unfold tileShare
  dsimp only
  rw [card_singleton, Nat.cast_one]
  have hm' : (sh.m : ℚ) ≠ 0 := by exact_mod_cast hm.ne'
  have hn' : (sh.n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  field_simp

/-- A row's share of a tile's cap is the one-row tile's. -/
theorem rowTile_cap (d : PearlCDevice) (ρ : ℚ) (sh : Shape) (i : ℕ) (hm : 0 < sh.m) (hn : 0 < sh.n)
    (cols : Finset ℕ) :
    ρ * creditDevRev1 d sh * tileShare sh {i} cols =
      ρ * creditDevRev1 d ⟨1, sh.n, sh.k⟩ * tileShare ⟨1, sh.n, sh.k⟩ {0} cols := by
  rw [creditDevRev1_rows d sh]
  unfold tileShare
  dsimp only
  rw [card_singleton, card_singleton, Nat.cast_one]
  have hm' : (sh.m : ℚ) ≠ 0 := by exact_mod_cast hm.ne'
  have hn' : (sh.n : ℚ) ≠ 0 := by exact_mod_cast hn.ne'
  field_simp

theorem tile_facts {L : Layout} {g : ℕ} (hg : g < (auditTiling 64 64 L).NT) : ∃ u bi bj : ℕ,
    (gridTiles 64 64 L).getD g (0, 0, 0) = (u, bi, bj) ∧ u < L.N ∧ bj < ((L.shape u).n + 64 - 1) / 64 := by
  have hmem := gridTiles_getD_mem hg
  rcases hb : (gridTiles 64 64 L).getD g (0, 0, 0) with ⟨u, bi, bj⟩
  rw [hb] at hmem
  obtain ⟨hu, _, hbj⟩ := mem_gridTiles.mp hmem
  exact ⟨u, bi, bj, rfl, hu, hbj⟩

section Block

variable {L : Layout} {g u bi bj : ℕ} (hb : (gridTiles 64 64 L).getD g (0, 0, 0) = (u, bi, bj))
include hb

theorem tile_unit : (auditTiling 64 64 L).unit g = u := by
  show ((gridTiles 64 64 L).getD g (0, 0, 0)).1 = u
  rw [hb]

theorem tile_rows : (auditTiling 64 64 L).rows g = Ico (bi * 64) (min ((bi + 1) * 64) (L.shape u).m) := by
  show Ico (((gridTiles 64 64 L).getD g (0, 0, 0)).2.1 * 64)
    (min ((((gridTiles 64 64 L).getD g (0, 0, 0)).2.1 + 1) * 64)
      (L.shape ((gridTiles 64 64 L).getD g (0, 0, 0)).1).m) = _
  rw [hb]

theorem tile_cols : (auditTiling 64 64 L).cols g = Ico (bj * 64) (min ((bj + 1) * 64) (L.shape u).n) := by
  show Ico (((gridTiles 64 64 L).getD g (0, 0, 0)).2.2 * 64)
    (min ((((gridTiles 64 64 L).getD g (0, 0, 0)).2.2 + 1) * 64)
      (L.shape ((gridTiles 64 64 L).getD g (0, 0, 0)).1).n) = _
  rw [hb]

end Block

theorem tilesCredit_eq (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (g : ℕ)
    (act : Codes) :
    (pearlCTilesDevRev1 d sem ρ).credit H s U g act =
      (((1 - ρ) * creditDevRev1 d (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) *
        tileShare (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
          (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) act
            ((auditTiling 64 64 U.layout).rows g))
          ((auditTiling 64 64 U.layout).cols g) : ℚ) : ℝ) := rfl

theorem tilesCap_eq (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ) (H : Q → R) (s : S) (U : Workload) (g : ℕ)
    (act : Codes) :
    (pearlCTilesDevRev1 d sem ρ).capOK H s U g act ↔
      sem.tileDebitRev1 d H s U ((auditTiling 64 64 U.layout).unit g)
          (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) act
            ((auditTiling 64 64 U.layout).rows g))
          ((auditTiling 64 64 U.layout).cols g) act ≤
        ρ * creditDevRev1 d (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) *
          tileShare (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
            (sem.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g)) act
              ((auditTiling 64 64 U.layout).rows g))
            ((auditTiling 64 64 U.layout).cols g) := Iff.rfl

section SplitTileW

variable {U : Workload} {u i bj : ℕ} (hu : u < U.layout.N) (hi : i < (U.layout.shape u).m)
  (hbj : bj < ((U.layout.shape u).n + 64 - 1) / 64)
include hu hi hbj

theorem splitTileW_lt : splitTile U.layout u i bj < (auditTiling 64 64 (splitWorkload U).layout).NT :=
  splitTile_lt hu hi hbj

theorem splitTileW_unit :
    (auditTiling 64 64 (splitWorkload U).layout).unit (splitTile U.layout u i bj) = rowIdx U.layout u i :=
  splitTile_unit hu hi hbj

theorem splitTileW_rows : (auditTiling 64 64 (splitWorkload U).layout).rows (splitTile U.layout u i bj) = {0} :=
  splitTile_rows hu hi hbj

theorem splitTileW_cols : (auditTiling 64 64 (splitWorkload U).layout).cols (splitTile U.layout u i bj) =
    Ico (bj * 64) (min ((bj + 1) * 64) (U.layout.shape u).n) :=
  splitTile_cols hu hi hbj

/-- The split tile's credit, for a row that passes Π's rules. -/
theorem splitTile_credit (hN : RowSeedNoise sem sem') (d : PearlCDevice) (ρ : ℚ) (H : Q → R) (s : S) {act : Codes}
    (hrow : sem'.RowOK (U.layout.shape u).m (U.layout.shape u).k act i) :
    (pearlCTilesDevRev1 d sem ρ).credit H s (splitWorkload U) (splitTile U.layout u i bj) (rowCodes act i) =
      (((1 - ρ) * creditDevRev1 d ⟨1, (U.layout.shape u).n, (U.layout.shape u).k⟩ *
        tileShare ⟨1, (U.layout.shape u).n, (U.layout.shape u).k⟩ {0}
          (Ico (bj * 64) (min ((bj + 1) * 64) (U.layout.shape u).n)) : ℚ) : ℝ) := by
  rw [tilesCredit_eq, splitTileW_unit hu hi hbj, splitTileW_rows hu hi hbj, splitTileW_cols hu hi hbj,
    split_passRows hN hu hi hrow, splitShapeW_eq hu hi]

end SplitTileW

open Classical in
/-- **The row-seeded tile credit is at most the split game's**, on every transcript: each credited row of a good tile
is a good one-row tile of the split layout with the row's share of the credit. -/
theorem split_tileCredit_le {d : PearlCDevice} (hd : d.RowLocal) (hN : RowSeedNoise sem sem') {ρ : ℚ} (hρ : ρ ≤ 1)
    (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) {U : Workload} (hU : pearlCDomainDev d U.layout)
    (H : Q → R) (s : S) (τ : Transcript) :
    ∑ g ∈ goodTiles (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCTilesDevRev1RowCap d sem' ρ) U H s τ,
        (pearlCTilesDevRev1RowCap d sem' ρ).credit H s U g (actOf τ ((auditTiling 64 64 U.layout).unit g)) ≤
      ∑ g ∈ goodTiles (pearlCProtocolDevRev1 d sem ρ) (pearlCTilesDevRev1 d sem ρ) (splitWorkload U) H s
          (splitTranscript U.layout τ),
        (pearlCTilesDevRev1 d sem ρ).credit H s (splitWorkload U) g
          (actOf (splitTranscript U.layout τ) ((auditTiling 64 64 (splitWorkload U).layout).unit g)) := by
  have h1 : ∀ g ∈ goodTiles (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCTilesDevRev1RowCap d sem' ρ) U H s τ,
      (pearlCTilesDevRev1RowCap d sem' ρ).credit H s U g (actOf τ ((auditTiling 64 64 U.layout).unit g)) =
        ∑ i ∈ sem'.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
            (actOf τ ((auditTiling 64 64 U.layout).unit g)) ((auditTiling 64 64 U.layout).rows g),
          (pearlCTilesDevRev1 d sem ρ).credit H s (splitWorkload U)
            (splitTile U.layout ((auditTiling 64 64 U.layout).unit g) i
              ((gridTiles 64 64 U.layout).getD g (0, 0, 0)).2.2)
            (actOf (splitTranscript U.layout τ) ((auditTiling 64 64 (splitWorkload U).layout).unit
              (splitTile U.layout ((auditTiling 64 64 U.layout).unit g) i
                ((gridTiles 64 64 U.layout).getD g (0, 0, 0)).2.2))) := by
    intro g hg
    obtain ⟨u, bi, bj, hb, hu, hbj⟩ := tile_facts (mem_range.mp (mem_filter.mp hg).1)
    have hdom := pearlCDomainDev_pos hU hu
    show (pearlCTilesDevRev1 d sem' ρ).credit H s U g _ = _
    rw [tilesCredit_eq, hb, tile_unit hb, tile_rows hb, tile_cols hb, tileCredit_rows d ρ _ hdom.1 hdom.2,
      Rat.cast_sum]
    refine sum_congr rfl fun i hi => ?_
    obtain ⟨hiI, hrow⟩ := mem_filter.mp hi
    have hi' : i < (U.layout.shape u).m := by
      have := mem_Ico.mp hiI
      omega
    rw [splitTileW_unit hu hi' hbj, actOf_split τ hu hi', splitTile_credit hu hi' hbj hN d ρ H s hrow]
  have hinj : Set.InjOn (fun p : (_ : ℕ) × ℕ => splitTile U.layout ((auditTiling 64 64 U.layout).unit p.1) p.2
        ((gridTiles 64 64 U.layout).getD p.1 (0, 0, 0)).2.2)
      ((goodTiles (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCTilesDevRev1RowCap d sem' ρ) U H s τ).sigma fun g =>
        sem'.passRows (U.layout.shape ((auditTiling 64 64 U.layout).unit g))
          (actOf τ ((auditTiling 64 64 U.layout).unit g)) ((auditTiling 64 64 U.layout).rows g)) := by
    rintro ⟨g, i⟩ hp ⟨g', i'⟩ hp' h
    obtain ⟨hg, hi⟩ := mem_sigma.mp hp
    obtain ⟨hg', hi'⟩ := mem_sigma.mp hp'
    dsimp only at hg hi hg' hi' h
    have hgl := mem_range.mp (mem_filter.mp hg).1
    have hgl' := mem_range.mp (mem_filter.mp hg').1
    obtain ⟨u, bi, bj, hb, hu, hbj⟩ := tile_facts hgl
    obtain ⟨u', bi', bj', hb', hu', hbj'⟩ := tile_facts hgl'
    have hiI := (mem_filter.mp hi).1
    have hiI' := (mem_filter.mp hi').1
    rw [tile_rows hb] at hiI
    rw [tile_rows hb'] at hiI'
    have hiu : i < (U.layout.shape u).m := by have := mem_Ico.mp hiI; omega
    have hiu' : i' < (U.layout.shape u').m := by have := mem_Ico.mp hiI'; omega
    have h2 : splitTile U.layout u i bj = splitTile U.layout u' i' bj' := by
      rw [tile_unit hb, tile_unit hb', hb, hb'] at h
      exact h
    have h3 := congrArg (fun v => (gridTiles 64 64 (splitLayout U.layout)).getD v (0, 0, 0)) h2
    simp only [splitTile_getD hu hiu hbj, splitTile_getD hu' hiu' hbj', Prod.mk.injEq] at h3
    obtain ⟨rfl, rfl⟩ := rowIdx_inj hu hiu hu' hiu' h3.1
    have hbi : bi = bi' := by
      have := mem_Ico.mp hiI
      have := mem_Ico.mp hiI'
      omega
    subst hbi
    have hgg : g = g' := gridTiles_getD_inj hgl hgl' (by rw [hb, hb', h3.2.2])
    subst hgg
    rfl
  rw [sum_congr rfl h1, sum_sigma']
  refine (sum_image (f := fun v => (pearlCTilesDevRev1 d sem ρ).credit H s (splitWorkload U) v
    (actOf (splitTranscript U.layout τ) ((auditTiling 64 64 (splitWorkload U).layout).unit v))) hinj).symm.trans_le
    (sum_le_sum_of_subset_of_nonneg ?_ ?_)
  · intro v hv
    obtain ⟨⟨g, i⟩, hp, rfl⟩ := mem_image.mp hv
    obtain ⟨hg, hi⟩ := mem_sigma.mp hp
    dsimp only at hg hi ⊢
    obtain ⟨hgr, c, hc, hcap, hwts, hwords⟩ := mem_filter.mp hg
    obtain ⟨u, bi, bj, hb, hu, hbj⟩ := tile_facts (mem_range.mp hgr)
    have hdom := pearlCDomainDev_pos hU hu
    have hc' : τ u = some c := by
      have h' : τ ((auditTiling 64 64 U.layout).unit g) = some c := hc
      rwa [tile_unit hb] at h'
    have hact : actOf τ u = c.act := by simp [actOf, hc']
    obtain ⟨hiI, hrow⟩ := mem_filter.mp hi
    rw [tile_rows hb] at hiI
    rw [tile_unit hb, hact] at hrow
    have hiu : i < (U.layout.shape u).m := by have := mem_Ico.mp hiI; omega
    simp only [tile_unit hb, hb]
    refine mem_filter.mpr ⟨mem_range.mpr (splitTileW_lt hu hiu hbj), rowCommit c i, ?_, ?_, ?_, ?_⟩
    · show splitTranscript U.layout τ
        ((auditTiling 64 64 (splitWorkload U).layout).unit (splitTile U.layout u i bj)) = _
      rw [splitTileW_unit hu hiu hbj, splitTranscript_rowIdx τ hu hiu, hc']
      rfl
    · show (pearlCTilesDevRev1 d sem ρ).capOK H s (splitWorkload U) (splitTile U.layout u i bj) (rowCodes c.act i)
      rw [tilesCap_eq, splitTileW_unit hu hiu hbj, splitTileW_rows hu hiu hbj, splitTileW_cols hu hiu hbj,
        split_passRows hN hu hiu hrow, split_tileDebit hd hN H s hu hiu _ c.act, splitShapeW_eq hu hiu,
        ← rowTile_cap d ρ _ i hdom.1 hdom.2]
      have hpass : i ∈ sem'.passRows (U.layout.shape u) c.act ((auditTiling 64 64 U.layout).rows g) := by
        rw [tile_rows hb]
        exact mem_filter.mpr ⟨hiI, hrow⟩
      have := hcap i (by rw [tile_unit hb]; exact hpass)
      rw [tile_unit hb, tile_cols hb] at this
      exact this
    · intro j hj'
      have hj : j ∈ (auditTiling 64 64 (splitWorkload U).layout).cols (splitTile U.layout u i bj) := hj'
      rw [splitTileW_cols hu hiu hbj] at hj
      have hw := hwts j (show j ∈ (auditTiling 64 64 U.layout).cols g by rw [tile_cols hb]; exact hj)
      rw [show ((pearlCTilesDevRev1RowCap d sem' ρ).tiling U.layout).unit g = u from tile_unit hb] at hw
      show (pearlCProtocolDevRev1 d sem ρ).weightOK
        ((splitWorkload U).layout.shape ((auditTiling 64 64 (splitWorkload U).layout).unit (splitTile U.layout u i bj)))
        ((splitWorkload U).weight ((splitWorkload U).layout.wid
          ((auditTiling 64 64 (splitWorkload U).layout).unit (splitTile U.layout u i bj)))) j
      rw [splitTileW_unit hu hiu hbj]
      exact split_weightOK hN ρ hu hiu hw
    · intro i' hi'' _ j hj'
      have hi0 : i' ∈ (auditTiling 64 64 (splitWorkload U).layout).rows (splitTile U.layout u i bj) := hi''
      have hj : j ∈ (auditTiling 64 64 (splitWorkload U).layout).cols (splitTile U.layout u i bj) := hj'
      rw [splitTileW_rows hu hiu hbj, mem_singleton] at hi0
      subst hi0
      rw [splitTileW_cols hu hiu hbj] at hj
      have hw : wordQ (c.ticket i j) = (pearlCProtocolDevRev1RowCap d sem' ρ).ticket H s U u c.act i j ∧
          wordQ (c.useful i j) = (pearlCProtocolDevRev1RowCap d sem' ρ).useful H s U u c.act i j := by
        have h' := hwords i (show i ∈ (auditTiling 64 64 U.layout).rows g by rw [tile_rows hb]; exact hiI)
          (show sem'.RowOK (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).m
            (U.layout.shape ((auditTiling 64 64 U.layout).unit g)).k c.act i by rw [tile_unit hb]; exact hrow) j
          (show j ∈ (auditTiling 64 64 U.layout).cols g by rw [tile_cols hb]; exact hj)
        rwa [show ((pearlCTilesDevRev1RowCap d sem' ρ).tiling U.layout).unit g = u from tile_unit hb] at h'
      show wordQ (c.ticket i j) = (pearlCProtocolDevRev1 d sem ρ).ticket H s (splitWorkload U)
          ((auditTiling 64 64 (splitWorkload U).layout).unit (splitTile U.layout u i bj)) (rowCodes c.act i) 0 j ∧
        wordQ (c.useful i j) = (pearlCProtocolDevRev1 d sem ρ).useful H s (splitWorkload U)
          ((auditTiling 64 64 (splitWorkload U).layout).unit (splitTile U.layout u i bj)) (rowCodes c.act i) 0 j
      rw [splitTileW_unit hu hiu hbj]
      exact ⟨hw.1.trans (split_ticket hd hN ρ H s hu hiu c.act j), hw.2.trans (split_useful hd hN ρ H s hu hiu c.act j)⟩
  · intro v hv _
    have hvl := mem_range.mp (mem_filter.mp hv).1
    have hu' := auditTiling_unit_lt 64 64 _ hvl
    have hsp := pearlCDomainDev_split hU _ hu'
    have hcr := creditDevRev1_nonneg d hadd hbf _ hsp.2.2.1 hsp.2.2.2.2.1
    rw [tilesCredit_eq]
    exact_mod_cast mul_nonneg (mul_nonneg (sub_nonneg.mpr hρ) hcr) (tileShare_nonneg' _ _ _)

/-- **Row additivity for P2 per audit tile (a theorem).** rev1's TT_OUT per tile at `sem` bounds the row-seeded game
per tile at `sem'`, with the cap checked per credited row: each credited row of a good tile is a good one-row tile of
the split layout carrying that row's credit, and the error is the split layout's, at `R = Σ_u m_u` units. -/
theorem ttOutTileRowSeed_of_ttOutTile [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S] {CM : CostModel Q R S}
    {d : PearlCDevice} {ρ : ℚ} (hTT : TTOutTilePearlCDevRev1 CM d sem ρ) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) (hd : d.RowLocal) (hρ : ρ ≤ 1) (hadd : 0 ≤ d.prices.add) (hbf : 0 ≤ d.prices.bf16) :
    TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem' ρ) (pearlCTilesDevRev1RowCap d sem' ρ)
      (pearlCDomainDev d) (1 / 400) := by
  refine ⟨hTT.1, fun U hU hR T hT q hq A hA => ?_⟩
  obtain ⟨prog', act', hrun⟩ := hCM A.online U A.act
  have hA' : (⟨A.pre, prog', act'⟩ : Adversary CM).Bounded (splitWorkload U) T q := fun H s =>
    ⟨(hrun H (A.pre.eval H) s).2.1.trans (hA H s).1,
      le_trans (Nat.add_le_add_left (hrun H (A.pre.eval H) s).2.2 _) (hA H s).2⟩
  have hU' : (splitWorkload U).InDomain (pearlCProtocolDevRev1 d sem ρ) (pearlCDomainDev d) :=
    ⟨hR, pearlCDomainDev_split hU.2⟩
  refine le_trans (Pouw.pr_mono fun x hx => lt_of_lt_of_le hx ?_)
    (hTT.2 (splitWorkload U) hU' T hT q hq ⟨A.pre, prog', act'⟩ hA')
  have htr : (⟨A.pre, prog', act'⟩ : Adversary CM).transcript (splitWorkload U) x.1 x.2 =
      splitTranscript U.layout (A.transcript U x.1 x.2) := (hrun x.1 (A.pre.eval x.1) x.2).1
  rw [htr]
  exact split_tileCredit_le hd hN hρ hadd hbf hU.2 x.1 x.2 (A.transcript U x.1 x.2)

end TileReduction

/-! ## The γ theorems, with the error at the row count -/

section Gamma

variable {Q R S : Type} [Fintype Q] [DecidableEq Q] [Fintype R] [Fintype S]

/-- **The row-seeded TT_OUT is a TT_OUT at the row count** on a one-shape domain of at most `2^64` rows: its error is
`εPearlC q (N·m)`, `(q + R)/2^128` at the `R = N·m` declared rows. -/
theorem ttOut_of_ttOutRowSeed {CM : CostModel Q R S} {P : Protocol Q R S} {d : PearlCDevice} {s : Shape} {γ₀ : ℝ}
    (h : TTOutRowSeed CM P (pearlCDomainDev d) γ₀) :
    TTOut CM P (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) γ₀ (fun q N => εPearlC q (N * s.m)) :=
  ⟨h.1, fun U hU T hT q hq A hA => by
    have h' := h.2 U ⟨hU.1, hU.2.1.1⟩ hU.2.2 T hT q hq A hA
    rwa [rowsOf_const hU.2.1.2] at h'⟩

/-- **The row-seeded TT_OUT per audit tile is a TT_OUT per tile at the row count**, as `ttOut_of_ttOutRowSeed`. -/
theorem ttOutTile_of_ttOutTileRowSeed {CM : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S}
    {d : PearlCDevice} {s : Shape} {γ₀ : ℝ} (h : TTOutTileRowSeed CM P TR (pearlCDomainDev d) γ₀) :
    TTOutTile CM P TR (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) γ₀ (fun q N => εPearlC q (N * s.m)) :=
  ⟨h.1, fun U hU T hT q hq A hA => by
    have h' := h.2 U ⟨hU.1, hU.2.1.1⟩ hU.2.2 T hT q hq A hA
    rwa [rowsOf_const hU.2.1.2] at h'⟩

/-- The row-seeded TT_OUT does not read `W_ref`. -/
theorem ttOutRowSeed_wref {CM : CostModel Q R S} {P : Protocol Q R S} {D : Layout → Prop} {γ₀ : ℝ}
    (W : Layout → ℕ → ℝ) (h : TTOutRowSeed CM P D γ₀) : TTOutRowSeed CM { P with Wref := W } D γ₀ :=
  h

/-- The row-seeded TT_OUT per tile reads neither `W_ref` nor the credited rows' `W_ref`. -/
theorem ttOutTileRowSeed_wref {CM : CostModel Q R S} {P : Protocol Q R S} {TR : TileRules Q R S}
    {D : Layout → Prop} {γ₀ : ℝ} (W W' : Layout → ℕ → ℝ) (Wc : Layout → ℕ → Codes → ℝ)
    (h : TTOutTileRowSeed CM P TR D γ₀) :
    TTOutTileRowSeed CM { P with Wref := W } { TR with Wref := W', Wcred := Wc } D γ₀ :=
  h

/-- **The row-seeded TT_OUT gives `G_γ` at any per-shape `W_ref`** `W`: rev1 with the per-row cap and `W_ref` at `W`, on
the one-shape domain at `γ = 1 − (399/400)/ω`, where `W s = ω·(1 − ρ)·creditDevRev1`, with the error at the row count
(`gammaFromTTOut`). The reduction doesn't read `W_ref` (`ttOutRowSeed_wref`); `W` enters only through the per-unit bound
`W_ref ≤ ω·credit` on the original layout. rev1 (`wrefDevRev1`) and the honest kernel's `wrefDevRev1K` instantiate it. -/
theorem pearlCGammaDevRev1RowSeedWAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCDomainDev d) (1 / 400)) :
    Gγ CM { pearlCProtocolDevRev1RowCap d sem ρ with Wref := fun L u => (W (L.shape u) : ℝ) }
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (γ : ℝ) (fun q N => εPearlC q (N * s.m)) := by
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  refine gammaFromTTOut CM _ _ (1 / 400) (ω : ℝ) _ (by exact_mod_cast hω0) ?_
    (ttOut_of_ttOutRowSeed (ttOutRowSeed_wref (fun L u => (W (L.shape u) : ℝ)) hTT))
  intro U hU H s' u hu act _ hcap
  have hsh : U.layout.shape u = s := hU.2.1.2 u hu
  have hdom := pearlCDomainDev_pos hU.2.1.1 hu
  have hc : sem.unitDebitRev1 d H s' U u act ≤ ρ * creditDevRev1 d (U.layout.shape u) :=
    rowCap_agg sem d ρ H s' U u act hdom.1 hdom.2 hcap
  show (W (U.layout.shape u) : ℝ) ≤
    (ω : ℝ) * ((creditDevRev1 d (U.layout.shape u) - sem.unitDebitRev1 d H s' U u act : ℚ) : ℝ)
  rw [hsh] at hc ⊢
  rw [hω]
  have : ω * ((1 - ρ) * creditDevRev1 d s) ≤ ω * (creditDevRev1 d s - sem.unitDebitRev1 d H s' U u act) :=
    mul_le_mul_of_nonneg_left (by linarith) hω0.le
  exact_mod_cast this

/-- **The row-seeded TT_OUT per audit tile gives `GγSampled` at any per-shape `W_ref`** `W`: rev1's tiles with the
per-row cap and `W_ref` and the credited rows' `W_ref` at `W`, on the one-shape domain at `γ = 1 − (399/400)/ω`, where
`W s = ω·(1 − ρ)·creditDevRev1 ≥ 0`, with the error at the row count (`gammaSampled`), as
`pearlCGammaDevRev1RowSeedWAt`. -/
theorem pearlCSampledDevRev1RowSeedWAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ)
    (W : Shape → ℚ) (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ W s)
    (hω : W s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCTilesDevRev1RowCap d sem ρ)
      (pearlCDomainDev d) (1 / 400)) :
    GγSampled CM { pearlCProtocolDevRev1RowCap d sem ρ with Wref := fun L u => (W (L.shape u) : ℝ) }
      { pearlCTilesDevRev1RowCap d sem ρ with
        Wref := fun L g =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ)
        Wcred := fun L g act =>
          let tl := auditTiling 64 64 L
          let sh := L.shape (tl.unit g)
          ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ) }
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (γ : ℝ) (fun q N => εPearlC q (N * s.m)) := by
  have e : (γ : ℝ) = 1 - (1 - 1 / 400) / (ω : ℝ) := by rw [hγ]; push_cast; ring
  rw [e]
  refine gammaSampled CM _ _ _ (1 / 400) (ω : ℝ) _ (by norm_num) (by exact_mod_cast hω0) ?_ ?_
    (ttOutTile_of_ttOutTileRowSeed (ttOutTileRowSeed_wref (fun L u => (W (L.shape u) : ℝ))
      (fun L g =>
        let tl := auditTiling 64 64 L
        let sh := L.shape (tl.unit g)
        ((W sh * tileShare sh (tl.rows g) (tl.cols g) : ℚ) : ℝ))
      (fun L g act =>
        let tl := auditTiling 64 64 L
        let sh := L.shape (tl.unit g)
        ((W sh * tileShare sh (sem.passRows sh act (tl.rows g)) (tl.cols g) : ℚ) : ℝ)) hTT))
  · intro U hU H s' g hg act _
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    simp only [pearlCTilesDevRev1RowCap, pearlCTilesDevRev1]
    rw [hsh, hω]
    push_cast
    exact le_of_eq (by ring)
  · intro U hU g hg act
    have hu := auditTiling_unit_lt 64 64 U.layout hg
    have hsh : U.layout.shape ((auditTiling 64 64 U.layout).unit g) = s := hU.2.1.2 _ hu
    simp only [pearlCTilesDevRev1RowCap, pearlCTilesDevRev1]
    rw [hsh]
    exact_mod_cast mul_le_mul_of_nonneg_left (tileShare_mono _ (passRows_subset sem _ _ _) _) hw0

/-- The row-seeded TT_OUT gives `G_γ` for rev1 with the per-row cap on the one-shape domain at
`γ = 1 − (399/400)/ω`, where `wrefDevRev1 = ω·(1 − ρ)·creditDevRev1`, with the error at the row count
(`gammaFromTTOut`). -/
theorem pearlCGammaDevRev1RowSeedAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ)
    (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s))
    (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCDomainDev d) (1 / 400)) :
    Gγ CM (pearlCProtocolDevRev1RowCap d sem ρ) (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (γ : ℝ)
      (fun q N => εPearlC q (N * s.m)) :=
  pearlCGammaDevRev1RowSeedWAt CM d sem ρ (wrefDevRev1 d) s ω γ hω0 hω hγ hTT

/-- The row-seeded TT_OUT per audit tile gives `GγSampled` for rev1's tiles with the per-row cap on the one-shape
domain at `γ = 1 − (399/400)/ω`, where `wrefDevRev1 = ω·(1 − ρ)·creditDevRev1 ≥ 0`, with the error at the row count
(`gammaSampled`). -/
theorem pearlCSampledDevRev1RowSeedAt (CM : CostModel Q R S) (d : PearlCDevice) (sem : PearlCSem Q R S) (ρ : ℚ)
    (s : Shape) (ω γ : ℚ) (hω0 : 0 < ω) (hw0 : 0 ≤ wrefDevRev1 d s)
    (hω : wrefDevRev1 d s = ω * ((1 - ρ) * creditDevRev1 d s)) (hγ : γ = 1 - (1 - 1 / 400) / ω)
    (hTT : TTOutTileRowSeed CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCTilesDevRev1RowCap d sem ρ)
      (pearlCDomainDev d) (1 / 400)) :
    GγSampled CM (pearlCProtocolDevRev1RowCap d sem ρ) (pearlCTilesDevRev1RowCap d sem ρ)
      (fun L => pearlCDomainDevAt d s L ∧ rowsOf L ≤ 2 ^ 64) (γ : ℝ) (fun q N => εPearlC q (N * s.m)) :=
  pearlCSampledDevRev1RowSeedWAt CM d sem ρ (wrefDevRev1 d) s ω γ hω0 hw0 hω hγ hTT

/-! ## The sm_120 corollaries -/

variable (CM : CostModel Q R S) {sem sem' : PearlCSem Q R S}

theorem devSm120_add (G : ℕ) : 0 ≤ (devAt sm120E4m3 G Prices.sm120).prices.add := by
  show (0 : ℚ) ≤ Prices.sm120.add; norm_num [Prices.sm120]

theorem devSm120_bf16 (G : ℕ) : 0 ≤ (devAt sm120E4m3 G Prices.sm120).prices.bf16 := by
  show (0 : ℚ) ≤ Prices.sm120.bf16; norm_num [Prices.sm120]

/-- **v1 with per-row seeds at 8192³**, from rev1's TT_OUT: `γ = 16223/3177500`, rev1's, with the error at the
`8192·N` rows. -/
theorem pearlCGammaSm120v1RowSeed_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    Gγ CM (pearlCProtocolDevRev1RowCap (devSm120v1 Prices.sm120) sem' (1 / 400))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((16223 / 3177500 : ℚ) : ℝ) (fun q N => εPearlC q (N * 8192)) :=
  pearlCGammaDevRev1RowSeedAt CM _ sem' _ sh8192 (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num)
    (ttOutRowSeed_of_ttOut hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 4) (devSm120_bf16 4))

/-- **v1 with per-row seeds at 16384³**, from rev1's TT_OUT: `γ = 2963/586700`, rev1's, with the error at the
`16384·N` rows. -/
theorem pearlCGammaSm120v1RowSeed_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    Gγ CM (pearlCProtocolDevRev1RowCap (devSm120v1 Prices.sm120) sem' (1 / 400))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384 L ∧ rowsOf L ≤ 2 ^ 64)
      ((2963 / 586700 : ℚ) : ℝ) (fun q N => εPearlC q (N * 16384)) :=
  pearlCGammaDevRev1RowSeedAt CM _ sem' _ sh16384 (5867 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num)
    (ttOutRowSeed_of_ttOut hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 4) (devSm120_bf16 4))

/-- **v2 with per-row seeds at the cap 1/1,000 at 8192³**, from rev1's TT_OUT (rev1 is v2's row at `G = 0`):
`γ = 216793/59950000`, rev1's, with the error at the `8192·N` rows. -/
theorem pearlCGammaSm120v2RowSeedCap1000_8192
    (hTT : TTOutPearlCDevRev1 CM (devSm120v2 Prices.sm120) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    Gγ CM (pearlCProtocolDevRev1RowCap (devSm120v2 Prices.sm120) sem' (1 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((216793 / 59950000 : ℚ) : ℝ) (fun q N => εPearlC q (N * 8192)) :=
  pearlCGammaDevRev1RowSeedAt CM _ sem' _ sh8192 (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v2_G, devSm120v2_prices]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num)
    (ttOutRowSeed_of_ttOut hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 0) (devSm120_bf16 0))

/-- **v2 with per-row seeds at the cap 1/1,000 at 16384³**, from rev1's TT_OUT: `γ = 2950127/829250000`, rev1's,
with the error at the `16384·N` rows. -/
theorem pearlCGammaSm120v2RowSeedCap1000_16384
    (hTT : TTOutPearlCDevRev1 CM (devSm120v2 Prices.sm120) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    Gγ CM (pearlCProtocolDevRev1RowCap (devSm120v2 Prices.sm120) sem' (1 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384 L ∧ rowsOf L ≤ 2 ^ 64)
      ((2950127 / 829250000 : ℚ) : ℝ) (fun q N => εPearlC q (N * 16384)) :=
  pearlCGammaDevRev1RowSeedAt CM _ sem' _ sh16384 (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v2_G, devSm120v2_prices]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num)
    (ttOutRowSeed_of_ttOut hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 0) (devSm120_bf16 0))

/-- **v1 with per-row seeds per audit tile at 8192³**, from rev1's TT_OUT per tile: `γ = 16223/3177500`. -/
theorem pearlCSampledSm120v1RowSeed_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GγSampled CM (pearlCProtocolDevRev1RowCap (devSm120v1 Prices.sm120) sem' (1 / 400))
      (pearlCTilesDevRev1RowCap (devSm120v1 Prices.sm120) sem' (1 / 400))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((16223 / 3177500 : ℚ) : ℝ) (fun q N => εPearlC q (N * 8192)) :=
  pearlCSampledDevRev1RowSeedAt CM _ sem' _ sh8192 (31775 / 31692) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 4)
      (devSm120_bf16 4))

/-- **v1 with per-row seeds per audit tile at 16384³**, from rev1's TT_OUT per tile: `γ = 2963/586700`. -/
theorem pearlCSampledSm120v1RowSeed_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v1 Prices.sm120) sem (1 / 400)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GγSampled CM (pearlCProtocolDevRev1RowCap (devSm120v1 Prices.sm120) sem' (1 / 400))
      (pearlCTilesDevRev1RowCap (devSm120v1 Prices.sm120) sem' (1 / 400))
      (fun L => pearlCDomainDevAt (devSm120v1 Prices.sm120) sh16384 L ∧ rowsOf L ≤ 2 ^ 64)
      ((2963 / 586700 : ℚ) : ℝ) (fun q N => εPearlC q (N * 16384)) :=
  pearlCSampledDevRev1RowSeedAt CM _ sem' _ sh16384 (5867 / 5852) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v1_G, devSm120v1_prices]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 4)
      (devSm120_bf16 4))

/-- **v2 with per-row seeds at the cap 1/1,000 per audit tile at 8192³**, from rev1's TT_OUT per tile:
`γ = 216793/59950000`. -/
theorem pearlCSampledSm120v2RowSeedCap1000_8192
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v2 Prices.sm120) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GγSampled CM (pearlCProtocolDevRev1RowCap (devSm120v2 Prices.sm120) sem' (1 / 1000))
      (pearlCTilesDevRev1RowCap (devSm120v2 Prices.sm120) sem' (1 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v2 Prices.sm120) sh8192 L ∧ rowsOf L ≤ 2 ^ 64)
      ((216793 / 59950000 : ℚ) : ℝ) (fun q N => εPearlC q (N * 8192)) :=
  pearlCSampledDevRev1RowSeedAt CM _ sem' _ sh8192 (1049125 / 1047951) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v2_G, devSm120v2_prices]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v2_G, devSm120v2_prices]
        norm_num [Prices.sm120, Params.pi, sh8192])
    (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 0)
      (devSm120_bf16 0))

/-- **v2 with per-row seeds at the cap 1/1,000 per audit tile at 16384³**, from rev1's TT_OUT per tile:
`γ = 2950127/829250000`. -/
theorem pearlCSampledSm120v2RowSeedCap1000_16384
    (hTT : TTOutTilePearlCDevRev1 CM (devSm120v2 Prices.sm120) sem (1 / 1000)) (hN : RowSeedNoise sem sem')
    (hCM : CM.SplitClosed) :
    GγSampled CM (pearlCProtocolDevRev1RowCap (devSm120v2 Prices.sm120) sem' (1 / 1000))
      (pearlCTilesDevRev1RowCap (devSm120v2 Prices.sm120) sem' (1 / 1000))
      (fun L => pearlCDomainDevAt (devSm120v2 Prices.sm120) sh16384 L ∧ rowsOf L ≤ 2 ^ 64)
      ((2950127 / 829250000 : ℚ) : ℝ) (fun q N => εPearlC q (N * 16384)) :=
  pearlCSampledDevRev1RowSeedAt CM _ sem' _ sh16384 (2073125 / 2070927) _ (by norm_num)
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v2_G, devSm120v2_prices]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by rw [wrefDevRev1, creditDevRev1, firstAddDev, creditDev, devSm120v2_G, devSm120v2_prices]
        norm_num [Prices.sm120, Params.pi, sh16384])
    (by norm_num)
    (ttOutTileRowSeed_of_ttOutTile hTT hN hCM (devAt_rowLocal _ _ _) (by norm_num) (devSm120_add 0)
      (devSm120_bf16 0))

end Gamma

end Pouw.PearlC
