import Proofs.Flock.Claims.Recursion.SeqSession
import Proofs.Flock.Soundness.ZK.HidingAdaptive
import Proofs.Flock.Soundness.ZK.RealLeaves
import Proofs.Flock.Claims.ZK.Rewind

/-!
# The masked recursion's first part, from `Hm96Hiding`

`zkBoundM_of` takes the first part's `ZKmalAt` (`h₀`): the inner session and the staging round, against `Sim`'s first
two parts, which send commit strings to the zero digest under fresh salts. Where the inner prover's messages are hm96
commit strings of digests under fresh salts of its tape (`RefParts.PI`'s commit form) and the staged values' digests
read no staging salt, the two are within `2δ₁` per commit string:

* `XorBytes`: 64 bytes under XOR, the group the hm96 leaves' digests and masks add in.
* `leaves_close`: `n` leaves of digests `x(ω)` are within `2n·δ₁` of `n` leaves of the zero digest (`ideal_leaves_swap`
  twice: each side is within `n·δ₁` of ideal leaves).
* `interact_leaves_close`: a run whose message `i` is a leaf whose digest reads the history before it is within
  `2n·δ₁` of the run whose messages are leaves of the zero digest (`ideal_leaves_adaptive`: the run's histories are
  functions of the leaves before them, `interact_adaptive`).
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Game FlockSoundness.ZK
open scoped ENNReal NNReal

/-! ## 64 bytes under XOR -/

/-- **64 bytes under XOR**, the group of hm96's digests and salt masks. -/
def XorBytes : Type := Fin 64 → UInt8

instance : Fintype XorBytes := inferInstanceAs (Fintype (Fin 64 → UInt8))

instance : Nonempty XorBytes := ⟨fun _ => 0⟩

/-- XOR, bytewise. -/
def xorBytes (a b : Fin 64 → UInt8) : Fin 64 → UInt8 := fun i => a i ^^^ b i

instance : Zero XorBytes := ⟨fun _ => 0⟩

instance : Add XorBytes := ⟨xorBytes⟩

instance : Neg XorBytes := ⟨id⟩

instance : AddCommGroup XorBytes where
  add_assoc a b c := funext fun i => UInt8.xor_assoc (a i) (b i) (c i)
  zero_add a := funext fun i => UInt8.zero_xor (a := a i)
  add_zero a := funext fun i => UInt8.xor_zero (a := a i)
  neg_add_cancel a := funext fun i => UInt8.xor_self (a := a i)
  add_comm a b := funext fun i => UInt8.xor_comm (a i) (b i)
  nsmul := nsmulRec
  zsmul := zsmulRec

theorem xorBytes_add (a b : XorBytes) : a + b = xorBytes a b := rfl

/-! ## Probabilities, both ways -/

theorem Pr_le_of_prCoin {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] {p : α → Prop} {q : β → Prop}
    {e : ℝ≥0∞} (he : e ≠ ⊤) (h : prCoin p ≤ prCoin q + e) : Pr p ≤ Pr q + e.toReal := by
  rw [Pr_eq_prCoin, Pr_eq_prCoin, ← ENNReal.toReal_add (prCoin_ne_top q) he]
  exact ENNReal.toReal_mono (ENNReal.add_ne_top.2 ⟨prCoin_ne_top q, he⟩) h

/-- **A one-sided bound on every event is a two-sided one.** -/
theorem abs_Pr_le {α β : Type} [Fintype α] [Nonempty α] [Fintype β] [Nonempty β] (p : α → Prop) (q : β → Prop) {e : ℝ}
    (h1 : Pr p ≤ Pr q + e) (h2 : Pr (fun a => ¬ p a) ≤ Pr (fun b => ¬ q b) + e) : |Pr p - Pr q| ≤ e := by
  rw [Pr_compl, Pr_compl] at h2
  rw [abs_le]
  constructor <;> linarith

theorem toReal_nat_mul (n : ℕ) (δ₁ : ℝ≥0) : ((n : ℝ≥0∞) * (δ₁ : ℝ≥0∞)).toReal = n * (δ₁ : ℝ) := by
  rw [ENNReal.toReal_mul, ENNReal.toReal_natCast, ENNReal.coe_toReal]

theorem nat_mul_ne_top (n : ℕ) (δ₁ : ℝ≥0) : (n : ℝ≥0∞) * (δ₁ : ℝ≥0∞) ≠ ⊤ :=
  ENNReal.mul_ne_top (ENNReal.natCast_ne_top n) ENNReal.coe_ne_top

/-! ## Leaves of a digest against leaves of the zero digest -/

variable {Sa Gp Cm : Type} [Fintype Sa] [Nonempty Sa] [AddCommGroup Gp] [Fintype Gp]

/-- **`n` leaves are within `n·δ₁` of ideal ones, both ways** (`ideal_leaves_swap` on the event and on its negation). -/
theorem leaves_ideal_close (My : Sa → Gp) (c : Sa → Cm) {δ₁ : ℝ≥0} (hT1 : Claims.Hm96Hiding My c δ₁) {D : Type}
    (Hl : Gp × Cm → D) (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → Fin n → Gp) (G : Ω → (Fin n → D) → Prop) :
    |Pr (fun p : Ω × (Fin n → Sa) => G p.1 fun i => Hl (x p.1 i + My (p.2 i), c (p.2 i))) -
      Pr (fun p : Ω × (Fin n → Gp × Sa) => G p.1 fun i => Hl ((p.2 i).1, c (p.2 i).2))| ≤ n * δ₁ := by
  have h1 := Pr_le_of_prCoin (nat_mul_ne_top n δ₁) (ideal_leaves_swap My c hT1 Hl n x G)
  have h2 := Pr_le_of_prCoin (nat_mul_ne_top n δ₁) (ideal_leaves_swap My c hT1 Hl n x fun ω l => ¬ G ω l)
  rw [toReal_nat_mul] at h1 h2
  exact abs_Pr_le _ _ h1 h2

/-- **`n` leaves of digests `x(ω)` are within `2n·δ₁` of `n` leaves of the zero digest.** -/
theorem leaves_close (My : Sa → Gp) (c : Sa → Cm) {δ₁ : ℝ≥0} (hT1 : Claims.Hm96Hiding My c δ₁) {D : Type}
    (Hl : Gp × Cm → D) (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω] (x : Ω → Fin n → Gp) (G : Ω → (Fin n → D) → Prop) :
    |Pr (fun p : Ω × (Fin n → Sa) => G p.1 fun i => Hl (x p.1 i + My (p.2 i), c (p.2 i))) -
      Pr (fun p : Ω × (Fin n → Sa) => G p.1 fun i => Hl (My (p.2 i), c (p.2 i)))| ≤ 2 * n * δ₁ := by
  have h1 := leaves_ideal_close My c hT1 Hl n x G
  have h0 := leaves_ideal_close My c hT1 Hl n (fun _ _ => 0) G
  simp only [zero_add] at h0
  calc _ ≤ _ := abs_sub_le _ (Pr fun p : Ω × (Fin n → Gp × Sa) => G p.1 fun i => Hl ((p.2 i).1, c (p.2 i).2)) _
    _ ≤ n * δ₁ + n * δ₁ := add_le_add h1 (by rwa [abs_sub_comm])
    _ = 2 * n * δ₁ := by ring

/-! ## A run of adaptive leaves -/

/-- The prover whose message `i` is `ℓ i`, `[]` past them. -/
def lpm {m : ℕ} (ℓ : Fin m → Msg) (h : List (Msg × Msg)) : Msg := if hh : h.length < m then ℓ ⟨h.length, hh⟩ else []

/-- **A run's message `i` is a leaf whose digest reads the history before it**: at history `h` of length `i < n`,
`Hl(d(h) + M·y_i, c(y_i))`. -/
def leafPm (Hl : Gp × Cm → Msg) (n : ℕ) (d : List (Msg × Msg) → Gp) (My : Sa → Gp) (c : Sa → Cm) (y : Fin n → Sa)
    (h : List (Msg × Msg)) : Msg :=
  if hh : h.length < n then Hl (d h + My (y ⟨h.length, hh⟩), c (y ⟨h.length, hh⟩)) else []

theorem histOf_take_length (τ : List (Msg × Out)) {i : ℕ} (hi : i < τ.length) : (histOf (τ.take i)).length = i := by
  simp [histOf, List.length_take, Nat.min_eq_left hi.le]

/-- **`i` rounds of `n`** are the first `i` rounds of the run. -/
theorem interact_take (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (hist : List (Msg × Msg)) {i n : ℕ}
    (h : i ≤ n) : (interact ask pm n hist).take i = interact ask pm i hist := by
  obtain ⟨r, rfl⟩ := Nat.exists_eq_add_of_le h
  rw [interact_add ask pm i r hist]
  have hl := interact_length_le ask pm i hist
  unfold cont
  split
  · rename_i hg
    rw [List.take_append_of_le_length (le_of_eq (goesOn_length hg).symm)]
    exact List.take_of_length_le (le_of_eq (goesOn_length hg))
  · exact List.take_of_length_le hl

/-- **The run of leaves `ℓ` for `i` rounds reads only the first `i`.** -/
theorem interact_lpm_cast (ask : List Msg → Out) {n : ℕ} (ℓ : Fin n → Msg) (i : ℕ) (hi : i ≤ n) :
    interact ask (lpm fun j : Fin i => ℓ (Fin.castLE hi j)) i [] = interact ask (lpm ℓ) i [] := by
  refine interact_congr ask (lpm ℓ) _ i [] fun j hj => ?_
  have hl := interact_length_le ask (lpm ℓ) i []
  have hlen := histOf_take_length (interact ask (lpm ℓ) i []) hj
  have hji : j < i := by omega
  have hjn : j < n := by omega
  simp only [List.nil_append, lpm, hlen, hji, hjn, ↓reduceDIte]
  rfl

omit [Fintype Sa] [Nonempty Sa] [Fintype Gp] in
/-- **A run of adaptive leaves is the run of the leaves `adaptiveLeaves` gives**: leaf `i`'s digest is `d` at the
history the leaves before it make. -/
theorem interact_adaptive (Hl : Gp × Cm → Msg) (My : Sa → Gp) (c : Sa → Cm) (ask : List Msg → Out) (n : ℕ)
    (d : List (Msg × Msg) → Gp) (y : Fin n → Sa) :
    interact ask (leafPm Hl n d My c y) n [] =
      interact ask (lpm (adaptiveLeaves Hl (fun i ℓ => d (histOf (interact ask (lpm ℓ) i [])))
        fun i => (My (y i), c (y i)))) n [] := by
  set L := adaptiveLeaves Hl (fun (i : Fin n) ℓ => d (histOf (interact ask (lpm ℓ) i [])))
    fun i => (My (y i), c (y i)) with hL
  refine interact_congr ask (lpm L) _ n [] fun i hi => ?_
  have hl := interact_length_le ask (lpm L) n []
  have hin : i < n := by omega
  have hlen := histOf_take_length (interact ask (lpm L) n []) hi
  have hLi : L ⟨i, hin⟩ = Hl (d (histOf (interact ask (lpm fun j : Fin i => L (Fin.castLE hin.le j)) i [])) +
      My (y ⟨i, hin⟩), c (y ⟨i, hin⟩)) := adaptiveLeaves_eq Hl _ _ ⟨i, hin⟩
  simp only [List.nil_append, leafPm, lpm, hlen, hin, ↓reduceDIte]
  rw [hLi, interact_lpm_cast ask L i hin.le, ← interact_take ask (lpm L) [] hin.le]

/-- **A run of adaptive leaves is within `2n·δ₁` of the run of leaves of the zero digest**: message `i` the leaf of the
digest `d` reads at the history before it, against the leaf of the zero digest, under the same salts; any event may
also read the other randomness `ω`. -/
theorem interact_leaves_close (My : Sa → Gp) (c : Sa → Cm) {δ₁ : ℝ≥0} (hT1 : Claims.Hm96Hiding My c δ₁)
    (Hl : Gp × Cm → Msg) (ask : List Msg → Out) (n : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (d : Ω → List (Msg × Msg) → Gp) (E : Ω → List (Msg × Out) → Prop) :
    |Pr (fun p : Ω × (Fin n → Sa) => E p.1 (interact ask (leafPm Hl n (d p.1) My c p.2) n [])) -
      Pr (fun p : Ω × (Fin n → Sa) => E p.1 (interact ask (leafPm Hl n (fun _ => 0) My c p.2) n []))| ≤
      2 * n * δ₁ := by
  let G : Ω → (Fin n → Msg) → Prop := fun ω ℓ => E ω (interact ask (lpm ℓ) n [])
  let xA : Ω → (i : Fin n) → (Fin i → Msg) → Gp := fun ω i ℓ => d ω (histOf (interact ask (lpm ℓ) i []))
  have hR : (fun p : Ω × (Fin n → Sa) => E p.1 (interact ask (leafPm Hl n (d p.1) My c p.2) n [])) =
      fun p => G p.1 (adaptiveLeaves Hl (xA p.1) fun i => (My (p.2 i), c (p.2 i))) :=
    funext fun p => by rw [interact_adaptive Hl My c ask n (d p.1) p.2]
  have hZ : (fun p : Ω × (Fin n → Sa) => E p.1 (interact ask (leafPm Hl n (fun _ => 0) My c p.2) n [])) =
      fun p => G p.1 fun i => Hl (My (p.2 i), c (p.2 i)) := by
    funext p
    show E p.1 _ = E p.1 _
    congr 2
    funext h
    simp only [leafPm, lpm, zero_add]
  have h1 := Pr_le_of_prCoin (nat_mul_ne_top n δ₁) (ideal_leaves_adaptive My c hT1 Hl n xA G)
  have h2 := Pr_le_of_prCoin (nat_mul_ne_top n δ₁)
    (ideal_leaves_adaptive My c hT1 Hl n xA fun ω ℓ => ¬ G ω ℓ)
  rw [toReal_nat_mul] at h1 h2
  have hA := abs_Pr_le _ _ h1 h2
  have h0 := leaves_ideal_close My c hT1 Hl n (fun _ _ => 0) G
  simp only [zero_add] at h0
  rw [hR, hZ]
  calc _ ≤ _ := abs_sub_le _ (Pr fun p : Ω × (Fin n → Gp × Sa) => G p.1 fun i => Hl ((p.2 i).1, c (p.2 i).2)) _
    _ ≤ n * δ₁ + n * δ₁ := add_le_add hA (by rwa [abs_sub_comm])
    _ = 2 * n * δ₁ := by ring

/-! ## Leaves, then a round of leaves -/

/-- **A first part of leaves**: `kI` rounds of leaves of the digests `dI` reads under the salts `yI`, then, where they
go on, one round whose message `stg` makes of the history and `nC` leaves of the digests `dS` reads under the salts
`yS`. -/
def prefixRun (Hl : Gp × Cm → Msg) (My : Sa → Gp) (c : Sa → Cm) (ask : List Msg → Out) (kI nC : ℕ)
    (dI : List (Msg × Msg) → Gp) (dS : List (Msg × Msg) → Fin nC → Gp)
    (stg : List (Msg × Msg) → (Fin nC → Msg) → Msg) (yI : Fin kI → Sa) (yS : Fin nC → Sa) : List (Msg × Out) :=
  cont kI (interact ask (leafPm Hl kI dI My c yI) kI [])
    (interact ask (fun h => stg h fun j => Hl (dS h j + My (yS j), c (yS j))) 1
      (histOf (interact ask (leafPm Hl kI dI My c yI) kI [])))

/-- One round reads the prover only at the history it starts from. -/
theorem interact_one (ask : List Msg → Out) (pm : List (Msg × Msg) → Msg) (hist : List (Msg × Msg)) :
    interact ask pm 1 hist = interact ask (fun _ => pm hist) 1 hist := by
  simp only [interact]

/-- **A first part of leaves is within `2(kI + nC)·δ₁` of the one of leaves of the zero digest**: the staging round's
`nC` leaves first (`leaves_close`, the salts before them fixed), then the `kI` adaptive leaves before it
(`interact_leaves_close`, the staging round's salts fixed). The other randomness `R` the digests read drops out. -/
theorem prefix_close (My : Sa → Gp) (c : Sa → Cm) {δ₁ : ℝ≥0} (hT1 : Claims.Hm96Hiding My c δ₁)
    (Hl : Gp × Cm → Msg) (ask : List Msg → Out) (kI nC : ℕ) {R : Type} [Fintype R] [Nonempty R]
    (dI : R → List (Msg × Msg) → Gp) (dS : R × (Fin kI → Sa) → List (Msg × Msg) → Fin nC → Gp)
    (stg : List (Msg × Msg) → (Fin nC → Msg) → Msg) (Ev : List (Msg × Out) → Prop) :
    |Pr (fun p : (R × (Fin kI → Sa)) × (Fin nC → Sa) =>
        Ev (prefixRun Hl My c ask kI nC (dI p.1.1) (dS p.1) stg p.1.2 p.2)) -
      Pr (fun q : (Fin kI → Sa) × (Fin nC → Sa) =>
        Ev (prefixRun Hl My c ask kI nC (fun _ => 0) (fun _ _ => 0) stg q.1 q.2))| ≤ 2 * (kI + nC) * δ₁ := by
  let τI : R → (Fin kI → Sa) → List (Msg × Out) := fun ρ yI => interact ask (leafPm Hl kI (dI ρ) My c yI) kI []
  let G : R × (Fin kI → Sa) → (Fin nC → Msg) → Prop := fun ω ℓ =>
    Ev (cont kI (τI ω.1 ω.2) (interact ask (fun h => stg h ℓ) 1 (histOf (τI ω.1 ω.2))))
  have h1 := leaves_close My c hT1 Hl nC (fun ω j => dS ω (histOf (τI ω.1 ω.2)) j) G
  have hR : (fun p : (R × (Fin kI → Sa)) × (Fin nC → Sa) =>
      Ev (prefixRun Hl My c ask kI nC (dI p.1.1) (dS p.1) stg p.1.2 p.2)) =
      fun p => G p.1 fun j => Hl (dS p.1 (histOf (τI p.1.1 p.1.2)) j + My (p.2 j), c (p.2 j)) := by
    funext p
    simp only [prefixRun, G, τI]
    rw [interact_one]
    conv_rhs => rw [interact_one]
  let E : R × (Fin nC → Sa) → List (Msg × Out) → Prop := fun ω τ =>
    Ev (cont kI τ (interact ask (fun h => stg h fun j => Hl (My (ω.2 j), c (ω.2 j))) 1 (histOf τ)))
  have h2 := interact_leaves_close My c hT1 Hl ask kI (fun ω => dI ω.1) E
  let e : (R × (Fin nC → Sa)) × (Fin kI → Sa) ≃ (R × (Fin kI → Sa)) × (Fin nC → Sa) :=
    { toFun := fun q => ((q.1.1, q.2), q.1.2), invFun := fun p => ((p.1.1, p.2), p.1.2),
      left_inv := fun _ => rfl, right_inv := fun _ => rfl }
  have hH : Pr (fun p : (R × (Fin kI → Sa)) × (Fin nC → Sa) => G p.1 fun j => Hl (My (p.2 j), c (p.2 j))) =
      Pr (fun q : (R × (Fin nC → Sa)) × (Fin kI → Sa) =>
        E q.1 (interact ask (leafPm Hl kI (dI q.1.1) My c q.2) kI [])) :=
    (Pr_equiv e _).symm
  let e₂ : ((Fin kI → Sa) × (Fin nC → Sa)) × R ≃ (R × (Fin nC → Sa)) × (Fin kI → Sa) :=
    { toFun := fun a => ((a.2, a.1.2), a.1.1), invFun := fun q => ((q.2, q.1.2), q.1.1),
      left_inv := fun _ => rfl, right_inv := fun _ => rfl }
  have hZ : Pr (fun q : (R × (Fin nC → Sa)) × (Fin kI → Sa) =>
        E q.1 (interact ask (leafPm Hl kI (fun _ => 0) My c q.2) kI [])) =
      Pr (fun q : (Fin kI → Sa) × (Fin nC → Sa) =>
        Ev (prefixRun Hl My c ask kI nC (fun _ => 0) (fun _ _ => 0) stg q.1 q.2)) := by
    rw [← Pr_equiv e₂]
    refine (Eq.trans ?_ (Pr_fst (β := R) _))
    congr 1
    funext a
    simp only [e₂, E, prefixRun, zero_add]
    rfl
  rw [hR]
  rw [hH] at h1
  rw [hZ] at h2
  calc _ ≤ _ := abs_sub_le _ (Pr fun q : (R × (Fin nC → Sa)) × (Fin kI → Sa) =>
        E q.1 (interact ask (leafPm Hl kI (dI q.1.1) My c q.2) kI [])) _
    _ ≤ 2 * nC * δ₁ + 2 * kI * δ₁ := add_le_add h1 h2
    _ = 2 * (kI + nC) * δ₁ := by ring

end Claims.ZKMal
