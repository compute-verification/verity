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
* `prefix_close`: a first part of `kI` adaptive leaves, then a staging round of `nC` leaves (`prefixRun`), is within
  `2(kI + nC)·δ₁` of the one of leaves of the zero digest.

At `deployRec`, both sides are a `prefixRun`: `Sim`'s first part at the zero digest (`simPre_eval`), and the first
part's prover `preP` where its inner messages are commit strings (`preP_eval`). So the first part is `ZKmalAt` within
`2(k_I + k_S·n_C)·δ₁` (`zkAt_pre`), `zkBoundM_of`'s `h₀` holds where the first part commits by hm96 under fresh salts at
every statement (`PreCommits`, `h0_of_commits`), and `zkBoundM_of_commits` is `zkBoundM_of_tables` without `h₀`.

`PreCommits` holds at the reference prover whose inner prover commits (`RefParts.commit`: each inner message hm96's
commit string of a digest of the witness, its other randomness and the history, under that round's salt), on each
statement's own first-part tape (`preTape`: the other randomness, one salt per inner round, one per commit string),
which reads as the inner prover's tape and the staging salts with the zero salt past them (`preOf`;
`preCommits_commit`). On one tape for every statement, it would bound the inner rounds and the commit strings.
-/

namespace Claims.ZKMal

open Claims
open FlockSoundness.Game FlockSoundness.ZK
open scoped ENNReal NNReal

/-! ## 64 bytes under XOR -/

/-- **64 bytes under XOR**, the group of hm96's digests and salt masks. -/
def XorBytes : Type := Fin 64 → UInt8

instance instFintypeXorBytes : Fintype XorBytes := inferInstanceAs (Fintype (Fin 64 → UInt8))

instance instNonemptyXorBytes : Nonempty XorBytes := ⟨fun _ => 0⟩

/-- XOR, bytewise. -/
def xorBytes (a b : Fin 64 → UInt8) : Fin 64 → UInt8 := fun i => a i ^^^ b i

instance instZeroXorBytes : Zero XorBytes := ⟨fun _ => 0⟩

instance instAddXorBytes : Add XorBytes := ⟨xorBytes⟩

instance instNegXorBytes : Neg XorBytes := ⟨id⟩

instance instAddCommGroupXorBytes : AddCommGroup XorBytes where
  add_assoc a b c := funext fun i => UInt8.xor_assoc (a i) (b i) (c i)
  zero_add a := funext fun i => UInt8.zero_xor (a := a i)
  add_zero a := funext fun i => UInt8.xor_zero (a := a i)
  neg_add_cancel a := funext fun i => UInt8.xor_self (a := a i)
  add_comm a b := funext fun i => UInt8.xor_comm (a i) (b i)
  nsmul := nsmulRec
  zsmul := zsmulRec

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

namespace RecDeploy

open _root_.Flock
open FlockDeploy (Coin)
open Claims.ZKMal (ZKmalAt SeqTape XorBytes leafPm prefixRun Proto)
open FlockSoundness.Discharge
open scoped NNReal

variable (IV : Claims.Verifier Coin) (b : Brings)

/-! ## The first part's leaves at the pinned key -/

/-- hm96's commit string `b ‖ c` as a leaf. -/
def hmLeaf (p : XorBytes × Claims.Msg) : Claims.Msg := List.ofFn p.1 ++ p.2

/-- The pinned key's salt hash, `c = H(p_salt ‖ y)` (`Deploy.saltHash` at `deployRec`). -/
def saltH (H : Claims.Msg → Claims.Msg) (y : Salt) : Claims.Msg :=
  H (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn y)

/-- The pinned key's salt mask as 64 bytes under XOR (`Deploy.mask` at `deployRec`). -/
def maskX (y : Salt) : XorBytes := maskOf y

/-- **The staging message** at the history's first `n` rounds and the commit strings `ℓ`. -/
noncomputable def stgOf (x : Claims.Msg) (n : ℕ) (h : List (Claims.Msg × Claims.Msg))
    (ℓ : Fin (b.nC x) → Claims.Msg) : Claims.Msg :=
  stageMsg b.hs x (asRounds (h.take n)) (List.ofFn ℓ)

theorem zero_leaf (H : Claims.Msg → Claims.Msg) (y : Salt) :
    hmLeaf (0 + maskX y, saltH H y) = List.ofFn (maskOf y) ++ saltH H y := by
  rw [zero_add]
  rfl

/-! ## `Sim`'s first part is a first part of leaves of the zero digest -/

theorem simRounds_stop (f : List (Claims.Msg × Claims.Out) → Claims.SP Claims.Msg) :
    ∀ (n : ℕ) (ms : List (Claims.Msg × Claims.Out)), Claims.Recursion.goesOn ms = false →
      Claims.Recursion.simRounds f n ms = .ret ms
  | 0, _, _ => rfl
  | _ + 1, _, h => by simp only [Claims.Recursion.simRounds, h, Bool.false_eq_true, ↓reduceIte]

theorem recGoesOn_of_msgs {ms : List (Claims.Msg × Claims.Out)} (h : ∀ p ∈ ms, Claims.ZKMal.isMsg p.2 = true) :
    Claims.Recursion.goesOn ms = true := by
  unfold Claims.Recursion.goesOn
  cases hl : ms.getLast? with
  | none => rfl
  | some p =>
    obtain ⟨a, o⟩ := p
    have := h _ (List.mem_of_getLast? hl)
    cases o <;> simp_all [Claims.ZKMal.isMsg]

/-- **`simRounds` after a transcript of messages is a run**: of the prover whose message at a history is `f`'s at
the transcript it reads as. -/
theorem simRounds_interact (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (f : List (Claims.Msg × Claims.Out) → Claims.SP Claims.Msg) :
    ∀ (n : ℕ) (ms : List (Claims.Msg × Claims.Out)), (∀ p ∈ ms, Claims.ZKMal.isMsg p.2 = true) →
      (Claims.SP.eval H ask (Claims.Recursion.simRounds f n ms)).1 =
        ms ++ Claims.ZKMal.interact ask (fun h => (Claims.SP.eval H ask (f (asRounds h))).1) n
          (Claims.ZKMal.histOf ms)
  | 0, ms, _ => by simp [Claims.Recursion.simRounds, Claims.SP.eval, Claims.ZKMal.interact]
  | n + 1, ms, hm => by
      rw [Claims.Recursion.simRounds, ite_eq_left_iff.2 fun h => absurd (recGoesOn_of_msgs hm) h,
        Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval]
      have hf : (Claims.ZKMal.histOf ms).map Prod.fst = ms.map Prod.fst := by simp [Claims.ZKMal.histOf]
      simp only [Claims.SP.eval, Claims.ZKMal.interact, asRounds_histOf hm, hf]
      cases hk : ask (ms.map Prod.fst ++ [(Claims.SP.eval H ask (f ms)).1]) with
      | msg m =>
        simp only
        rw [simRounds_interact H ask f n _ (fun p hp => ?_)]
        · simp [Claims.ZKMal.histOf]
        · rcases List.mem_append.1 hp with hp | hp
          · exact hm p hp
          · simp only [List.mem_singleton] at hp
            subst hp
            rfl
      | acc =>
        simp only
        rw [simRounds_stop f n _ (by simp [Claims.Recursion.goesOn])]
        simp [Claims.SP.eval]
      | rej =>
        simp only
        rw [simRounds_stop f n _ (by simp [Claims.Recursion.goesOn])]
        simp [Claims.SP.eval]

theorem zeroCommit_eval (ivScope : Claims.Msg → Bool) (H : Claims.Msg → Claims.Msg)
    (ask : List Claims.Msg → Claims.Out) (y : Salt) :
    (Claims.SP.eval H ask (Claims.Recursion.spOf ((deployRec IV ivScope b).zeroCommit y))).1 =
      List.ofFn (maskOf y) ++ saltH H y := by
  simp only [Claims.Recursion.Deploy.zeroCommit, Claims.Recursion.spOf, Claims.SP.eval, deployRec, saltH]
  rfl

theorem zeroCommits_eval (ivScope : Claims.Msg → Bool) (H : Claims.Msg → Claims.Msg)
    (ask : List Claims.Msg → Claims.Out) :
    ∀ ys : List (Fin (deployRec IV ivScope b).saltLen → UInt8),
      (Claims.SP.eval H ask ((deployRec IV ivScope b).zeroCommits ys)).1 =
        ys.map fun y => List.ofFn (maskOf y) ++ saltH H y
  | [] => rfl
  | y :: ys => by
    simp only [Claims.Recursion.Deploy.zeroCommits, Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval,
      zeroCommits_eval ivScope H ask ys, Claims.SP.eval]
    exact congrArg (fun a => a :: _) (zeroCommit_eval IV b ivScope H ask y)

theorem stage_eval (ivScope : Claims.Msg → Bool) (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (x : Claims.Msg) (ms : List (Claims.Msg × Claims.Out)) (cs : List Claims.Msg) :
    (Claims.SP.eval H ask (Claims.Recursion.spOf ((deployRec IV ivScope b).stage x ms cs))).1 =
      stageMsg b.hs x ms cs :=
  rfl

/-- **`Sim`'s inner rounds are leaves of the zero digest** under its inner salts. -/
theorem simInner_eval (ivScope : Claims.Msg → Bool) (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (n : ℕ) (salts : Fin n → Salt) :
    (Claims.SP.eval H ask (Claims.Recursion.simInner (deployRec IV ivScope b) n salts)).1 =
      Claims.ZKMal.interact ask (leafPm hmLeaf n (fun _ => 0) maskX (saltH H) salts) n [] := by
  unfold Claims.Recursion.simInner
  rw [simRounds_interact H ask _ n [] (by simp)]
  simp only [List.nil_append, Claims.ZKMal.histOf, List.map_nil]
  congr 1
  funext h
  have hl : (asRounds h).length = h.length := List.length_map _
  by_cases hh : h.length < n
  · simp only [hl, hh, ↓reduceDIte, leafPm, zero_leaf, zeroCommit_eval]
  · simp only [hl, hh, ↓reduceDIte, leafPm, Claims.SP.eval]

/-- **`Sim`'s staging round after `n` rounds of messages** is the staging message at commit strings to the zero
digest under its staging salts. -/
theorem simStage_eval (ivScope : Claims.Msg → Bool) (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out)
    (x : Claims.Msg) (n : ℕ) (salts : Fin (kS x) → Fin (b.nC x) → Salt) (τ : List (Claims.Msg × Claims.Out))
    (hm : ∀ p ∈ τ, Claims.ZKMal.isMsg p.2 = true) (hl : τ.length = n) :
    (Claims.SP.eval H ask (Claims.Recursion.simStage (deployRec IV ivScope b) x n salts τ)).1 =
      τ ++ Claims.ZKMal.interact ask (fun h => stgOf b x n h fun j =>
        hmLeaf (0 + maskX (salts ⟨0, Nat.zero_lt_one⟩ j), saltH H (salts ⟨0, Nat.zero_lt_one⟩ j))) 1
        (Claims.ZKMal.histOf τ) := by
  unfold Claims.Recursion.simStage
  rw [simRounds_interact H ask _ _ τ hm]
  congr 1
  change Claims.ZKMal.interact ask _ 1 _ = _
  rw [Claims.ZKMal.interact_one]
  conv_rhs => rw [Claims.ZKMal.interact_one]
  congr 2
  funext _
  have hτ : (asRounds (Claims.ZKMal.histOf τ)).length - n = 0 := by
    rw [asRounds_histOf hm, hl, Nat.sub_self]
  have h0 : (asRounds (Claims.ZKMal.histOf τ)).length - n < (deployRec IV ivScope b).kS x := by
    rw [hτ]
    exact Nat.zero_lt_one
  have ht : (Claims.ZKMal.histOf τ).take n = Claims.ZKMal.histOf τ :=
    List.take_of_length_le (by rw [Claims.ZKMal.histOf_length, hl])
  have hs : (⟨(asRounds (Claims.ZKMal.histOf τ)).length - n, h0⟩ : Fin ((deployRec IV ivScope b).kS x)) =
      ⟨0, Nat.zero_lt_one⟩ := Fin.ext hτ
  simp only [h0, ↓reduceDIte, hs]
  simp only [Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval, zeroCommits_eval, stgOf, ht, stage_eval]
  congr 1
  rw [List.ofFn_eq_map]
  erw [List.map_map]
  congr 1
  funext j
  exact (zero_leaf H _).symm

/-- **`Sim`'s first part is a first part of leaves of the zero digest**: its inner rounds' commit strings under its
inner salts, then its staging round's under its staging salts. -/
theorem simPre_eval {W : Type} (ivScope : Claims.Msg → Bool) (I : Claims.Recursion.Inner Coin W)
    (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) (x : Claims.Msg)
    (c : (simPre (deployRec IV ivScope b) I).C x) :
    (Claims.SP.eval H ask ((simPre (deployRec IV ivScope b) I).prog x c)).1 =
      prefixRun hmLeaf maskX (saltH H) ask (I.V.k (xIOf x)) (b.nC x) (fun _ => 0) (fun _ _ => 0)
        (stgOf b x (I.V.k (xIOf x))) c.1 (c.2 ⟨0, Nat.zero_lt_one⟩) := by
  simp only [simPre, Claims.ZKMal.recSpBind_eq, Claims.ZKMal.spBind_eval]
  rw [simInner_eval IV b ivScope H ask _ c.1]
  unfold prefixRun Claims.ZKMal.cont
  split
  · rename_i hg
    have hl := Claims.ZKMal.goesOn_length hg
    have hm : ∀ p ∈ Claims.ZKMal.interact ask (leafPm hmLeaf (I.V.k (xIOf x)) (fun _ => 0) maskX (saltH H) c.1)
        (I.V.k (xIOf x)) [], Claims.ZKMal.isMsg p.2 = true := by
      simp only [Claims.ZKMal.goesOn, Bool.and_eq_true, List.all_eq_true] at hg
      exact hg.2
    exact simStage_eval IV b ivScope H ask x _ c.2 _ hm hl
  · rename_i hg
    unfold Claims.Recursion.simStage
    rw [simRounds_stop _ _ _ ?_]
    · rfl
    rcases Nat.eq_zero_or_pos (I.V.k (xIOf x)) with h0 | h0
    · exact absurd (by simp [h0, Claims.ZKMal.interact, Claims.ZKMal.goesOn]) hg
    · exact (Claims.ZKMal.recGoesOn_interact ask _ (I.V.k (xIOf x)) [] h0).trans (Bool.eq_false_iff.2 hg)

/-! ## The first part's prover is a first part of leaves -/

/-- 64 bytes as a digest under XOR. -/
def asX (d : Fin 64 → UInt8) : XorBytes := d

/-- **Commit strings, then `k`**: `k` at the commit strings' hm96 leaves. -/
theorem commitThen_eval (H : Claims.Msg → Claims.Msg) (k : List Claims.Msg → Claims.OC Claims.Msg) :
    ∀ (ps : List ((Fin 64 → UInt8) × Salt)) (cs : List Claims.Msg),
      ((commitThen k ps cs).eval H).1 =
        ((k (cs ++ ps.map fun p => hmLeaf (asX p.1 + maskX p.2, saltH H p.2))).eval H).1
  | [], cs => by simp [commitThen]
  | (d, y) :: ps, cs => by
      simp only [commitThen, Claims.OC.eval]
      rw [commitThen_eval H k ps]
      simp only [List.map_cons, List.append_assoc, List.singleton_append]
      rfl

/-- **The first part's prover is a first part of leaves**: where its inner messages are hm96 commit strings of the
digests `dI` reads under the inner salts `yI` (`hPI`), its staged values' digests are `dS`'s (`hdig`) and its staging
salts are `yS` (`hsalt`). -/
theorem preP_eval (H : Claims.Msg → Claims.Msg) (ask : List Claims.Msg → Claims.Out) {T : Type}
    (pre : T → b.ref.TI × (ℕ → Salt)) (w : Claims.Flock.Wit) (x : Claims.Msg) (t : T)
    (yI : Fin (IV.k (xIOf x)) → Salt) (yS : Fin (b.nC x) → Salt) (dI : List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dS : List (Claims.Msg × Claims.Msg) → Fin (b.nC x) → Fin 64 → UInt8)
    (hPI : ∀ h (hh : h.length < IV.k (xIOf x)), (((b.ref.PI w).next (xIOf x) h (pre t).1).eval H).1 =
      List.ofFn (fun i => dI h i ^^^ maskOf (yI ⟨h.length, hh⟩) i) ++ saltH H (yI ⟨h.length, hh⟩))
    (hdig : ∀ h j (hj : j < b.nC x), (b.ref.digests w (pre t).1 h).getD j 0 = dS h ⟨j, hj⟩)
    (hsalt : ∀ j (hj : j < b.nC x), (pre t).2 j = yS ⟨j, hj⟩) :
    Claims.ZKMal.interact ask (fun h => (((preP IV b pre w).next x h t).eval H).1) (kPre IV x) [] =
      prefixRun hmLeaf maskX (saltH H) ask (IV.k (xIOf x)) (b.nC x) (fun h => asX (dI h))
        (fun h j => asX (dS (h.take (IV.k (xIOf x))) j)) (stgOf b x (IV.k (xIOf x))) yI yS := by
  have hin : Claims.ZKMal.interact ask (fun h => (((preP IV b pre w).next x h t).eval H).1) (IV.k (xIOf x)) [] =
      Claims.ZKMal.interact ask (leafPm hmLeaf (IV.k (xIOf x)) (fun h => asX (dI h)) maskX (saltH H) yI)
        (IV.k (xIOf x)) [] := by
    refine Claims.ZKMal.interact_congr_lt ask _ _ (IV.k (xIOf x)) (fun h hh => ?_) _ [] (by simp)
    have e : (((preP IV b pre w).next x h t).eval H).1 = (((b.ref.PI w).next (xIOf x) h (pre t).1).eval H).1 := by
      simp only [preP, hh, ↓reduceIte]
    rw [e, hPI h hh]
    simp only [leafPm, hh, ↓reduceDIte]
    rfl
  unfold kPre kS
  rw [Claims.ZKMal.interact_add ask _ (IV.k (xIOf x)) 1 [], hin, List.nil_append]
  unfold prefixRun Claims.ZKMal.cont
  split
  · rename_i hg
    congr 1
    rw [Claims.ZKMal.interact_one]
    conv_rhs => rw [Claims.ZKMal.interact_one]
    congr 2
    funext _
    have hl : (Claims.ZKMal.histOf (Claims.ZKMal.interact ask
        (leafPm hmLeaf (IV.k (xIOf x)) (fun h => asX (dI h)) maskX (saltH H) yI) (IV.k (xIOf x)) [])).length =
        IV.k (xIOf x) := by
      rw [Claims.ZKMal.histOf_length, Claims.ZKMal.goesOn_length hg]
    have e : ∀ h : List (Claims.Msg × Claims.Msg), h.length = IV.k (xIOf x) →
        (((preP IV b pre w).next x h t).eval H).1 = ((commitThen (fun cs => .ret (stageMsg b.hs x
          (asRounds (h.take (IV.k (xIOf x)))) cs)) ((List.range (b.nC x)).map fun j =>
            ((b.ref.digests w (pre t).1 (h.take (IV.k (xIOf x)))).getD j 0, (pre t).2 j)) []).eval H).1 := by
      intro h hh
      simp only [preP, hh, lt_irrefl, ↓reduceIte]
    rw [e _ hl, commitThen_eval]
    simp only [Claims.OC.eval, stgOf, List.nil_append]
    congr 1
    refine List.ext_getElem (by simp) fun j h1 h2 => ?_
    simp only [List.getElem_map, List.getElem_range, List.getElem_ofFn, hdig _ j (by simpa using h1),
      hsalt j (by simpa using h1)]
  · rfl

/-! ## The first part's zero knowledge -/

/-- **The first part is `ZKmalAt` at `Sim`'s first two parts within `2δ₁` per commit string**, at `x`: where the
first part's tape splits evenly (`split`, every fiber of one size) into other randomness `R`, the inner salts and the
staging salts; the inner prover's message at each inner round is hm96's commit string of a digest `dI` reads (the
witness, the other randomness and the history) under that round's inner salt (`hPI`, `RefParts.PI`'s commit form);
the staged values' digests read no staging salt (`hdig`); and the staging salts are the tape's (`hsalt`). -/
theorem zkAt_pre (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (x : Claims.Msg) (hI : I.V.k (xIOf x) = IV.k (xIOf x)) {R : Type} [Fintype R] [Nonempty R]
    (split : T₀.N × T₀.Ω → (R × (Fin (IV.k (xIOf x)) → Salt)) × (Fin (b.nC x) → Salt)) {m : ℕ}
    (hsplit : ∀ p, Nat.card {t // split t = p} = m)
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dS : Claims.Flock.Wit → R × (Fin (IV.k (xIOf x)) → Salt) → List (Claims.Msg × Claims.Msg) →
      Fin (b.nC x) → Fin 64 → UInt8)
    (hPI : ∀ w t h (hh : h.length < IV.k (xIOf x)), (((b.ref.PI w).next (xIOf x) h (pre t).1).eval H).1 =
      List.ofFn (fun i => dI w (split t).1.1 h i ^^^ maskOf ((split t).1.2 ⟨h.length, hh⟩) i) ++
        saltH H ((split t).1.2 ⟨h.length, hh⟩))
    (hdig : ∀ w t h j (hj : j < b.nC x), (b.ref.digests w (pre t).1 h).getD j 0 = dS w (split t).1 h ⟨j, hj⟩)
    (hsalt : ∀ t j (hj : j < b.nC x), (pre t).2 j = (split t).2 ⟨j, hj⟩) {δ₁ : ℝ≥0}
    (hδ : Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁) :
    ZKmalAt H (kPre IV x) (preP IV b pre) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
      ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
      (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0 := by
  intro w _ Θ Vs θ hV Ev
  have hT1 : Claims.Hm96Hiding maskX (saltH H) δ₁ := hδ
  have hreal : Claims.Pr (fun ω => Ev (Claims.runStar H (preP IV b pre w) Vs x ω θ).msgs) =
      Claims.Pr (fun p : (R × (Fin (IV.k (xIOf x)) → Salt)) × (Fin (b.nC x) → Salt) =>
        Ev (prefixRun hmLeaf maskX (saltH H) (fun as => ((Vs.next x as θ).eval H).1) (IV.k (xIOf x)) (b.nC x)
          (fun h => asX (dI w p.1.1 h)) (fun h j => asX (dS w p.1 (h.take (IV.k (xIOf x))) j))
          (stgOf b x (IV.k (xIOf x))) p.1.2 p.2)) := by
    rw [← Claims.ZKMal.Pr_comp_fiber split hsplit]
    congr 1
    funext ω
    rw [Claims.ZKMal.runStar_msgs, hV, preP_eval IV b H _ pre w x ω _ _ _ _ (hPI w ω) (hdig w ω) (hsalt ω)]
  let e : (simPre (deployRec IV ivScope b) I).C x ≃ (Fin (I.V.k (xIOf x)) → Salt) × (Fin (b.nC x) → Salt) :=
    Equiv.prodCongr (Equiv.refl _) (Equiv.funUnique (Fin 1) (Fin (b.nC x) → Salt))
  have hn : ∀ n, n = IV.k (xIOf x) →
      Claims.Pr (fun q : (Fin n → Salt) × (Fin (b.nC x) → Salt) =>
        Ev (prefixRun hmLeaf maskX (saltH H) (fun as => ((Vs.next x as θ).eval H).1) n (b.nC x) (fun _ => 0)
          (fun _ _ => 0) (stgOf b x n) q.1 q.2)) =
      Claims.Pr (fun q : (Fin (IV.k (xIOf x)) → Salt) × (Fin (b.nC x) → Salt) =>
        Ev (prefixRun hmLeaf maskX (saltH H) (fun as => ((Vs.next x as θ).eval H).1) (IV.k (xIOf x)) (b.nC x)
          (fun _ => 0) (fun _ _ => 0) (stgOf b x (IV.k (xIOf x))) q.1 q.2)) := by
    rintro n rfl
    rfl
  have hsim : Claims.Pr (fun c => Ev (Claims.ZKMal.spOut H Vs θ x ((simPre (deployRec IV ivScope b) I).prog x c))) =
      Claims.Pr (fun q : (Fin (IV.k (xIOf x)) → Salt) × (Fin (b.nC x) → Salt) =>
        Ev (prefixRun hmLeaf maskX (saltH H) (fun as => ((Vs.next x as θ).eval H).1) (IV.k (xIOf x)) (b.nC x)
          (fun _ => 0) (fun _ _ => 0) (stgOf b x (IV.k (xIOf x))) q.1 q.2)) := by
    rw [← hn _ hI, ← Claims.ZKMal.Pr_equiv e]
    congr 1
    funext c
    unfold Claims.ZKMal.spOut
    rw [simPre_eval IV b ivScope I H _ x c]
    rfl
  have h := Claims.ZKMal.prefix_close maskX (saltH H) hT1 hmLeaf (fun as => ((Vs.next x as θ).eval H).1)
    (IV.k (xIOf x)) (b.nC x) (fun ρ h => asX (dI w ρ h)) (fun ω₀ h j => asX (dS w ω₀ (h.take (IV.k (xIOf x))) j))
    (stgOf b x (IV.k (xIOf x))) Ev
  rw [hreal, hsim]
  refine h.trans (le_of_eq ?_)
  rw [hI]
  simp only [kS, Nat.cast_one, one_mul, zero_mul, add_zero]

/-- **The first part commits by hm96 under fresh salts** at `x` (`zkAt_pre`'s premises): its tape splits evenly into
other randomness, the inner salts and the staging salts; each inner message is hm96's commit string of a digest the
witness, the other randomness and the history give, under that round's inner salt (`RefParts.PI`'s commit form); the
staged values' digests read no staging salt; and the staging salts are the tape's. -/
def PreCommits (H : Claims.Msg → Claims.Msg) (T₀ : SeqTape) (pre : T₀.N × T₀.Ω → b.ref.TI × (ℕ → Salt))
    (x : Claims.Msg) : Prop :=
  ∃ (R : Type) (_ : Fintype R) (_ : Nonempty R)
    (split : T₀.N × T₀.Ω → (R × (Fin (IV.k (xIOf x)) → Salt)) × (Fin (b.nC x) → Salt)) (m : ℕ)
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dS : Claims.Flock.Wit → R × (Fin (IV.k (xIOf x)) → Salt) → List (Claims.Msg × Claims.Msg) →
      Fin (b.nC x) → Fin 64 → UInt8),
    (∀ p, Nat.card {t // split t = p} = m) ∧
    (∀ w t h (hh : h.length < IV.k (xIOf x)), (((b.ref.PI w).next (xIOf x) h (pre t).1).eval H).1 =
      List.ofFn (fun i => dI w (split t).1.1 h i ^^^ maskOf ((split t).1.2 ⟨h.length, hh⟩) i) ++
        saltH H ((split t).1.2 ⟨h.length, hh⟩)) ∧
    (∀ w t h j (hj : j < b.nC x), (b.ref.digests w (pre t).1 h).getD j 0 = dS w (split t).1 h ⟨j, hj⟩) ∧
    (∀ t j (hj : j < b.nC x), (pre t).2 j = (split t).2 ⟨j, hj⟩)

/-- **`zkBoundM_of`'s first part (`h₀`)**, where the inner protocol's rounds are the inner verifier's and the first
part commits by hm96 under fresh salts at every statement. -/
theorem h0_of_commits (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (T₀ : Claims.Msg → SeqTape)
    (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x)) (hpre : ∀ x, PreCommits IV b H (T₀ x) (pre x) x) :
    ∀ t, 0 < t → ∀ δ₁ : ℝ≥0,
      Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ → ∀ x,
        ZKmalAt H (kPre IV x) (preP IV b (pre x)) (Claims.Recursion.R (deployRec IV ivScope b) I x) x
          ((simPre (deployRec IV ivScope b) I).C x) ((simPre (deployRec IV ivScope b) I).prog x)
          (2 * (I.V.k (xIOf x) + kS x * b.nC x) * δ₁) 0 := by
  intro _ _ δ₁ hδ x
  obtain ⟨R, _, _, split, m, dI, dS, hs, hPI, hdig, hsalt⟩ := hpre x
  exact zkAt_pre IV b H ivScope I (T₀ x) (pre x) x (hI x) split hs dI dS hPI hdig hsalt hδ

/-- **The recursion's zero knowledge at the masked record, from the masked session's models and the first part's
commit strings**: `zkBoundM_of_tables` with its first part (`h₀`) from `PreCommits` at every statement. -/
theorem zkBoundM_of_commits (H : Claims.Msg → Claims.Msg) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (T₀ : Claims.Msg → SeqTape) (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x)) (hpre : ∀ x, PreCommits IV b H (T₀ x) (pre x) x)
    (hkey : ∀ δ₁ : ℝ≥0, Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash H) δ₁ →
      Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁)
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink H (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding H)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope b) I x w → ∀ j < nS x, ∀ (t₀ : (T₀ x).N × (T₀ x).Ω)
      (u : List (Claims.Msg × Claims.Msg)), u.length = (parts IV b ((T₀ x).N × (T₀ x).Ω)).start x j →
        Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u) (wit w (pre x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV b H ivScope I T₀ G.Nonce G.Rand pre wit G.prover
      fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) :=
  zkBoundM_of_tables IV b H ivScope I relO T₀ pre wit G Mp Mc nL nB CB hI
    (h0_of_commits IV b H ivScope I T₀ pre hI hpre) hkey hK hnL hnB hL hOK hC hzl hCB hW

/-! ## A reference prover whose inner prover commits -/

/-- **The inner prover that commits**: at each inner round, hm96's commit string of the digest `dI` reads (the
witness, the other randomness `R` and the history) under that round's salt, on a tape of `R` and one salt per round;
the staged values' digests (`dg`) and the outer witnesses (`wo`) read that tape. -/
def RefParts.commit (R : Type) [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit) : RefParts where
  TI := R × (ℕ → Salt)
  PI w := ⟨fun _ h ρ => .query (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn (ρ.2 h.length)) fun c =>
    .ret (List.ofFn (fun i => dI w ρ.1 h i ^^^ maskOf (ρ.2 h.length) i) ++ c)⟩
  digests := dg
  wit := wo

/-- What the deployment brings, its inner prover the one that commits. -/
abbrev Brings.commit (b : Brings) (R : Type) [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit) : Brings :=
  { b with ref := RefParts.commit R dI dg wo }

/-- **The first part's tape at `x`** for an inner prover that commits: the other randomness and one salt per inner
round, then one staging salt per commit string. -/
def preTape (R : Type) [Fintype R] [Nonempty R] (x : Claims.Msg) : SeqTape where
  N := R × (Fin (IV.k (xIOf x)) → Salt)
  Ω := Fin (b.nC x) → Salt

/-- Salts on `Fin n` read on `ℕ`, the zero salt past `n`. -/
def saltsOn {n : ℕ} (y : Fin n → Salt) (i : ℕ) : Salt := if h : i < n then y ⟨i, h⟩ else fun _ => 0

/-- **The first part's tape at `x` read as the inner prover's tape and the staging salts.** -/
def preOf (R : Type) [Fintype R] [Nonempty R] (x : Claims.Msg) :
    (preTape IV b R x).N × (preTape IV b R x).Ω → (R × (ℕ → Salt)) × (ℕ → Salt) :=
  fun t => ((t.1.1, saltsOn t.1.2), saltsOn t.2)

/-- **The first part commits by hm96 under fresh salts** at the inner prover that commits, on `x`'s first-part tape:
the tape is the split itself, so each fiber is one point. -/
theorem preCommits_commit (H : Claims.Msg → Claims.Msg) (R : Type) [Fintype R] [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (x : Claims.Msg) : PreCommits IV (b.commit R dI dg wo) H (preTape IV b R x) (preOf IV b R x) x := by
  refine ⟨R, inferInstance, inferInstance, id, 1, dI, fun w p h j => (dg w (p.1, saltsOn p.2) h).getD j 0,
    fun p => ?_, fun w t h hh => ?_, fun _ _ _ _ _ => rfl, fun t j hj => dite_eq_left hj⟩
  · exact Nat.card_eq_one_iff_unique.2 ⟨⟨fun a a' => Subtype.ext (a.2.trans a'.2.symm)⟩, ⟨⟨p, rfl⟩⟩⟩
  · have e : saltsOn t.1.2 h.length = t.1.2 ⟨h.length, hh⟩ := dite_eq_left hh
    show List.ofFn (fun i => dI w t.1.1 h i ^^^ maskOf (saltsOn t.1.2 h.length) i) ++
      H (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn (saltsOn t.1.2 h.length)) = _
    rw [e]
    rfl

end RecDeploy
