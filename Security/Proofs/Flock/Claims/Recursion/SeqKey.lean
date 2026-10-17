import Proofs.Flock.Claims.Recursion.SeqCap
import Proofs.Flock.Claims.Recursion.SeqPrefix
import Proofs.Flock.Soundness.Discharge.ZkView.Leaf

/-!
# The deployed key's hiding is the session analysis's, at SHA-512

The recursion's `ZK` reads `Hm96Hiding` at the record's key on bytes (`Deploy.mask`, `Deploy.saltHash H`), at the
claim's `H`; the `--zk` sessions' analysis reads it at the same key on bits (`ZkView.My`, `ZkView.saltHash`), at
SHA-512 itself. At `H = FlockDeploy.sha512` the first gives the second (`RecDeploy.hiding_sha512`), so
`zkBoundM_of_commits`' `hkey` holds there:

* `Claims.Hm96Hiding.of_equiv`: hiding carries over along a bijection of salts, one of masks and a function of the salt
  hashes.
* `RecDeploy.bytesFn`: `8·k` bits as `k` bytes, in `ZkView.bitsOf`'s order, a bijection (`bytesFn_bijective`); the
  record's mask of a salt's bytes is the bytes of `ZkView.My` of its bits (`bytesFn_My`, from `Hm.bitL_mask` and
  `ZkView.My_apply`), and its salt hash at SHA-512 is `ZkView.saltHash` (`saltHash_sha512`).

At another `H` the two assumptions are about different functions, so `hkey` stays a premise there.

`zkBoundM_of_commit_sha512` is `zkBoundM_of_commits_sha512` at the reference prover whose inner prover commits
(`Brings.commit`), on each statement's first-part tape (`preTape`), where `hpre` holds too (`preCommits_commit`).
-/

open scoped ENNReal NNReal

namespace Claims

open Classical in
theorem PrENNReal_equiv {α β : Type} [Fintype α] [Fintype β] (e : α ≃ β) (p : β → Prop) :
    PrENNReal (fun a => p (e a)) = PrENNReal p := by
  unfold PrENNReal
  rw [Fintype.card_congr e]
  exact Equiv.sum_comp e fun b => (Fintype.card β : ℝ≥0∞)⁻¹ * if p b then 1 else 0

/-- **`Hm96Hiding` along bijections**: of the salts (`eY`) and of the masks (`eB`), with the salt hashes read through
`g`. -/
theorem Hm96Hiding.of_equiv {Sa Gp Cm Sa' Gp' Cm' : Type} [Fintype Sa] [Fintype Gp] [Fintype Sa'] [Fintype Gp']
    {My : Sa → Gp} {c : Sa → Cm} {My' : Sa' → Gp'} {c' : Sa' → Cm'} {δ₁ : ℝ≥0∞} (h : Hm96Hiding My c δ₁)
    (eY : Sa' ≃ Sa) (eB : Gp' ≃ Gp) (g : Cm → Cm') (hM : ∀ y, eB (My' y) = My (eY y))
    (hc : ∀ y, c' y = g (c (eY y))) :
    Hm96Hiding My' c' δ₁ := by
  intro E
  have h1 := h fun p => E (eB.symm p.1, g p.2)
  have e1 : PrENNReal (fun y : Sa' => E (My' y, c' y)) =
      PrENNReal (fun y : Sa => E (eB.symm (My y), g (c y))) := by
    rw [← PrENNReal_equiv eY fun y : Sa => E (eB.symm (My y), g (c y))]
    congr 1
    funext y
    rw [← hM y, Equiv.symm_apply_apply, hc y]
  have e2 : PrENNReal (fun p : Gp' × Sa' => E (p.1, c' p.2)) =
      PrENNReal (fun p : Gp × Sa => E (eB.symm p.1, g (c p.2))) := by
    rw [← PrENNReal_equiv (eB.prodCongr eY) fun p : Gp × Sa => E (eB.symm p.1, g (c p.2))]
    congr 1
    funext p
    simp only [Equiv.prodCongr_apply, Prod.map_fst, Prod.map_snd, Equiv.symm_apply_apply, hc p.2]
  rw [e1, e2]
  exact h1

end Claims

namespace RecDeploy

open _root_.Flock
open FlockSoundness.Discharge
open FlockDeploy (Coin)

/-- **`8·k` bits as `k` bytes**, in `ZkView.bitsOf`'s order (`ZkView.bytesOf`'s bytes). -/
def bytesFn {k : ℕ} (v : Fin (8 * k) → ZMod 2) : Fin k → UInt8 := fun i =>
  ZkView.byteOf fun b => v ⟨8 * i.val + b.val, by have := i.isLt; have := b.isLt; omega⟩

theorem ofFn_bytesFn {k : ℕ} (v : Fin (8 * k) → ZMod 2) :
    List.ofFn (bytesFn v) = (ZkView.bytesOf v).data.toList := by
  rw [ZkView.bytesOf, Array.toList_ofFn]
  rfl

theorem bitL_ofFn_bytesFn {k : ℕ} (v : Fin (8 * k) → ZMod 2) {t : ℕ} (ht : t < 8 * k) :
    Hm.bitL (List.ofFn (bytesFn v)) t = decide (v ⟨t, ht⟩ = 1) := by
  rw [ofFn_bytesFn]
  exact ZkView.bitL_bytesOf v ht

theorem bytesFn_injective {k : ℕ} : Function.Injective (bytesFn (k := k)) := by
  intro v v' h
  funext t
  have e : decide (v t = 1) = decide (v' t = 1) := by
    rw [← bitL_ofFn_bytesFn v t.isLt, ← bitL_ofFn_bytesFn v' t.isLt, h]
  rw [ZkView.zmod2_ind (v t), ZkView.zmod2_ind (v' t), e]

theorem card_UInt8 : Fintype.card UInt8 = 256 := by
  rw [FinEnum.card_eq_fintypeCard.symm]
  rfl

theorem bytesFn_bijective {k : ℕ} : Function.Bijective (bytesFn (k := k)) := by
  refine (Fintype.bijective_iff_injective_and_card _).2 ⟨bytesFn_injective, ?_⟩
  rw [Fintype.card_fun, Fintype.card_fun, ZMod.card, Fintype.card_fin, Fintype.card_fin, card_UInt8, pow_mul]
  norm_num

/-- A salt's bits as its bytes. -/
noncomputable def eSalt : ZkView.Salt ≃ Salt := Equiv.ofBijective (bytesFn (k := 192)) bytesFn_bijective

/-- A mask's bits as its bytes. -/
noncomputable def eMask : ZkView.Bits ≃ (Fin 64 → UInt8) := Equiv.ofBijective (bytesFn (k := 64)) bytesFn_bijective

theorem saltBytes_toList (y : ZkView.Salt) : (ZkView.saltBytes y).data.toList = List.ofFn (eSalt y) :=
  (ofFn_bytesFn (k := 192) y).symm

theorem bitL_ofFn_maskOf (z : Salt) {t : ℕ} (ht : t < 512) :
    Hm.bitL (List.ofFn (maskOf z)) t = Hm.bitL (Hm96.Default512.mask ⟨Array.ofFn z⟩).data.toList t := by
  have ht8 : t / 8 < 64 := by omega
  simp only [Hm.bitL, List.getD_eq_getElem?_getD, List.getElem?_ofFn, ht8, maskOf]
  rw [Hm.get!_eq_getD, List.getD_eq_getElem?_getD]
  rfl

/-- **The record's mask of a salt's bytes is the bytes of `ZkView.My` of its bits.** -/
theorem bytesFn_My (y : ZkView.Salt) : eMask (ZkView.My y) = maskOf (eSalt y) := by
  refine List.ofFn_injective (Hm.eq_of_bitL (by rw [List.length_ofFn, List.length_ofFn]) fun t ht => ?_)
  rw [List.length_ofFn] at ht
  change Hm.bitL (List.ofFn (bytesFn (k := 64) (ZkView.My y))) t = _
  rw [bitL_ofFn_bytesFn (k := 64) _ ht, bitL_ofFn_maskOf _ ht, ← List.toArray_ofFn, Hm.bitL_mask _ ht,
    ZkView.My_apply y ⟨t, ht⟩, ← saltBytes_toList]
  cases Hm.maskV Hm96.Default512.key (Hm.bitL (ZkView.saltBytes y).data.toList) t <;> decide

/-- **The record's salt hash at SHA-512 is `ZkView.saltHash`**, as bytes. -/
theorem saltHash_sha512 (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (y : ZkView.Salt) :
    ZkView.saltHash y = Claims.ofMsg ((deployRec IV ivScope b).saltHash FlockDeploy.sha512 (eSalt y)) := by
  show _ = Claims.ofMsg (Claims.toMsg (Sha512.hash (Claims.ofMsg
    (Claims.toMsg Hm96.Default512.saltPrefix ++ List.ofFn (eSalt y)))))
  rw [Claims.ofMsg_toMsg, ZkView.saltHash, ← saltBytes_toList]
  congr 1

/-- **`zkBoundM_of_commits`' `hkey` at SHA-512**: hiding at the record's key on bytes, at `H = FlockDeploy.sha512`,
is hiding at the session analysis's key on bits. -/
theorem hiding_sha512 (IV : Claims.Verifier Coin) (ivScope : Claims.Msg → Bool) (b : Brings) (δ₁ : ℝ≥0)
    (h : Claims.Hm96Hiding (deployRec IV ivScope b).mask ((deployRec IV ivScope b).saltHash FlockDeploy.sha512) δ₁) :
    Claims.Hm96Hiding ZkView.My ZkView.saltHash δ₁ :=
  h.of_equiv eSalt eMask Claims.ofMsg bytesFn_My (saltHash_sha512 IV ivScope b)

open Claims.ZKMal (SeqTape Proto) in
/-- **`zkBoundM_of_commits` at SHA-512**, with its `hkey` discharged (`hiding_sha512`). -/
theorem zkBoundM_of_commits_sha512 (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (T₀ : Claims.Msg → SeqTape) (pre : (x : Claims.Msg) → (T₀ x).N × (T₀ x).Ω → b.ref.TI × (ℕ → Salt))
    (wit : Claims.Flock.Wit → b.ref.TI × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x)) (hpre : ∀ x, PreCommits IV b FlockDeploy.sha512 (T₀ x) (pre x) x)
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512)
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
    ZKBoundM IV b FlockDeploy.sha512 ivScope I T₀ G.Nonce G.Rand pre wit G.prover
      fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) :=
  zkBoundM_of_commits IV b FlockDeploy.sha512 ivScope I relO T₀ pre wit G Mp Mc nL nB CB hI hpre
    (hiding_sha512 IV ivScope b) hK hnL hnB hL hOK hC hzl hCB hW

open Claims.ZKMal (Proto) in
/-- **`zkBoundM_of_commits_sha512` at the inner prover that commits**, on each statement's first-part tape, with its
`hpre` discharged (`preCommits_commit`). -/
theorem zkBoundM_of_commit_sha512 (IV : Claims.Verifier Coin) (b : Brings) (ivScope : Claims.Msg → Bool)
    (I : Claims.Recursion.Inner Coin Claims.Flock.Wit) (relO : Readable.Parsed → Readable.Relation)
    (R : Type) [Fintype R] [Nonempty R]
    (dI : Claims.Flock.Wit → R → List (Claims.Msg × Claims.Msg) → Fin 64 → UInt8)
    (dg : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → List (Fin 64 → UInt8))
    (wo : Claims.Flock.Wit → R × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (wit : Claims.Flock.Wit → (R × (ℕ → Salt)) × (ℕ → Salt) → List (Claims.Msg × Claims.Msg) → ℕ → Claims.Flock.Wit)
    (G : Proto Claims.Flock.Wit) [Fintype G.Coin] (Mp : ∀ x, G.Nonce → G.TPreModel x)
    (Mc : ∀ x, G.Nonce → (Fin (G.K x) → G.Coin) → G.TCoinModel x) (nL nB : Claims.Msg → ℕ)
    (CB : ℕ → Claims.Msg → ℕ → Type) [∀ t x j, Fintype (CB t x j)] [∀ t x j, Nonempty (CB t x j)]
    (hI : ∀ x, I.V.k (xIOf x) = IV.k (xIOf x))
    (hK : ∀ x, G.K x = VO.k x - 1) (hnL : ∀ x ν, (Mp x ν).size ≤ nL x) (hnB : ∀ x ν c, (Mc x ν c).size ≤ nB x)
    (hL : G.TMsgLink FlockDeploy.sha512 (Claims.Flock.R (deployZkRec relO).parse) Mp Mc) (hOK : G.TablesOK Mp Mc)
    (hC : G.Binding FlockDeploy.sha512)
    (hzl : ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
      nL (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).1 ∧
        nB (xsOf IV b x j u) ≤ (b.zl (staged b.st x (pick b.st x j) j)).2)
    (hCB : ∀ t, 0 < t → ∀ x, ∀ j < nS x, ∀ u : List (Claims.Msg × Claims.Msg),
      u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
        Nonempty (CB t x j ≃ ((G.withSim Mc).simulatorE t (t * 3 ^ t)).C (xsOf IV b x j u)))
    (hW : ∀ x w, Claims.Recursion.R (deployRec IV ivScope (b.commit R dI dg wo)) I x w → ∀ j < nS x,
      ∀ (t₀ : (preTape IV b R x).N × (preTape IV b R x).Ω) (u : List (Claims.Msg × Claims.Msg)),
        u.length = (parts IV b ((preTape IV b R x).N × (preTape IV b R x).Ω)).start x j →
          Claims.Flock.R (deployZkRec relO).parse (xsOf IV b x j u)
            (wit w (preOf IV b R x t₀) (u.take (IV.k (xIOf x))) j)) :
    ZKBoundM IV (b.commit R dI dg wo) FlockDeploy.sha512 ivScope I (preTape IV b R) G.Nonce G.Rand (preOf IV b R) wit
      G.prover fun t => soOf ((G.withSim Mc).simulatorE t (t * 3 ^ t)) (CB t) :=
  zkBoundM_of_commits_sha512 IV (b.commit R dI dg wo) ivScope I relO (preTape IV b R) (preOf IV b R) wit G Mp Mc nL
    nB CB hI (fun x => preCommits_commit IV b FlockDeploy.sha512 R dI dg wo x) hK hnL hnB hL hOK hC hzl hCB hW

end RecDeploy
