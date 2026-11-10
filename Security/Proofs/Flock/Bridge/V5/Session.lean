module

public import Proofs.Flock.Bridge.V5.Transports
public import Proofs.Flock.Bridge.V5.OpeningsCanon
public import Proofs.Flock.Bridge.V5.Assembly
public import Proofs.Flock.Bridge.FaultyRightC

/-!
# A session's satisfying message opens its proof block correct at public inputs

What `SessionOpens` asks of one Flock session, at V5's objects (`keyProgP`, `rsZHJP`, `constsOfP` and the public file's
leaf map): a message that satisfies drawn proof block `u`'s table reveals openings, each of which opens under the
deployed commitment, such that any committed values carrying those that open make `u`'s proof block correct, whatever
its interior values.

* `revealOf`: the openings at values `Xu` and data `dat`: at each committed wire of `u` that `κ` doesn't fix, its
  position with `Xu`'s bit and `dat`'s datum, under `roots`. `correct_of_revealOf`: where `Xu` makes `u` correct and
  agrees with `κ`, and each opening opens, committed values carrying those that open make the proof block correct
  (`correct_of_local`). `revealOf_length_le`: at most `|P.io u|` openings.
* `revealV5`: `revealOf` at the bits of the openings a message gives (`vbP`'s `opening`), with each wire's registered
  row datum (`dataOfRow`, `dataAt`). `correct_of_revealV5`: they make `u` correct (`opening_spec` and
  `loweringSoundZK_of_place`), carry the constants (`openingsCanonV5_const`) and open under the deployed commitment
  (`openingsCanonV5`, `transports_shapedP_at`).
* `packOf`, `packZ`: the message packing a witness of a `--zk` table's statement, which gives it back
  (`witnessOf_packOf`, `wit_packZ`).
* `tabOfZHJP_one`: at one table and a constant draw, a drawn proof block's table is any set-up table's at that draw
  (`DrawSetupZHJP.table_congr`).
-/

@[expose] public section

namespace Flock.Bridge

open FlockSoundness.Audit Claims.Sampling

/-! ## The openings at a unit's values -/

section Reveal

variable {C : Verity.Boolean.Circuit} {n : ℕ} (P : FlockSoundness.Audit.Partition C n)
  (κ : Fin C.N → Option Bool) (roots : List Claims.Msg) (u : Fin n) (Xu : Fin C.N → Bool)
  (dat : Fin C.N → Claims.Msg)

/-- **The openings at values `Xu`**: at each committed wire `g` of unit `u` that `κ` doesn't fix, in wire order, `g`'s
position with `Xu`'s bit there and the datum `dat g`, under `roots`. -/
noncomputable def revealOf : List (List Claims.Msg × Opening Bool) :=
  ((List.finRange C.N).filter fun g => decide (g ∈ P.io u ∧ κ g = none)).map fun g =>
    (roots, ⟨posOf C g, Xu g, dat g⟩)

theorem mem_revealOf {q : List Claims.Msg × Opening Bool} :
    q ∈ revealOf P κ roots u Xu dat ↔ ∃ g ∈ P.io u, κ g = none ∧ q = (roots, ⟨posOf C g, Xu g, dat g⟩) := by
  unfold revealOf
  rw [List.mem_map]
  constructor
  · rintro ⟨g, hg, rfl⟩
    obtain ⟨hio, hk⟩ := of_decide_eq_true (List.mem_filter.1 hg).2
    exact ⟨g, hio, hk, rfl⟩
  · rintro ⟨g, hg, hk, rfl⟩
    exact ⟨g, List.mem_filter.2 ⟨List.mem_finRange g, decide_eq_true ⟨hg, hk⟩⟩, rfl⟩

/-- **At most one opening per committed wire.** -/
theorem revealOf_length_le : (revealOf P κ roots u Xu dat).length ≤ (P.io u).card := by
  unfold revealOf
  rw [List.length_map]
  have hnd := (List.nodup_finRange C.N).filter fun g => decide (g ∈ P.io u ∧ κ g = none)
  rw [← List.toFinset_card_of_nodup hnd]
  refine Finset.card_le_card fun g hg => ?_
  rw [List.mem_toFinset, List.mem_filter] at hg
  exact (of_decide_eq_true hg.2).1

/-- **Committed values carrying the openings that open make the proof block correct**: where `Xu` makes `u` correct
(`Partition.Correct`) and is `κ`'s constant wherever `κ` fixes one of `u`'s committed wires, and each opening of
`revealOf` opens under `Cm`, committed values `val` that carry each opening that opens under `roots` make slot `u`'s
proof block correct, whatever its interior values `ι` (`correct_of_local`). -/
theorem correct_of_revealOf (H : Claims.Msg → Claims.Msg) (Cm : CommitmentScheme Bool)
    (hκin : ∀ g b, κ g = some b → (C.op g).isInput = true) (hX : P.Correct Xu u)
    (hκ : ∀ g ∈ P.io u, ∀ b, κ g = some b → Xu g = b)
    (hopen : ∀ g ∈ P.io u, κ g = none → Cm.Opens H roots ⟨posOf C g, Xu g, dat g⟩ = true)
    (val : ℕ → Option Bool) (ι : ℕ → Bool)
    (hval : ∀ q ∈ revealOf P κ roots u Xu dat, Cm.Opens H q.1 q.2 = true → q.1 = roots →
      val q.2.pos = some q.2.val) :
    (progOf C κ).Correct ((progOf C κ).refIn (blockOf P κ u) val ι) (blockOf P κ u) :=
  correct_of_local P u hκin Xu hX hκ val
    (fun g hg hk => hval _ ((mem_revealOf P κ roots u Xu dat).2 ⟨g, hg, hk, rfl⟩) (hopen g hg hk) rfl) ι

end Reveal

/-! ## A registered row's datum -/

/-- **The datum of wire `g`'s registered row** at an opening's row bits `v` and salt `y` (`dataAt`), at a leaf map `L`;
empty at a wire with no row. -/
def dataOfRow (c : Flock.Circuit) (L : LeafMap) {N : ℕ} (rows : Fin N → Option RowAt) (g : Fin N) {W : ℕ}
    (v : FlockSoundness.Discharge.Layout.BitVal W) (y : List UInt8) : Claims.Msg :=
  match rows g with
  | some r => dataAt c L r v y
  | none => []

theorem dataOfRow_some {c : Flock.Circuit} {L : LeafMap} {N : ℕ} {rows : Fin N → Option RowAt} {g : Fin N}
    {r : RowAt} (h : rows g = some r) {W : ℕ} (v : FlockSoundness.Discharge.Layout.BitVal W) (y : List UInt8) :
    dataOfRow c L rows g v y = dataAt c L r v y := by
  simp only [dataOfRow, h]

theorem opens_dataOfRow {H : Claims.Msg → Claims.Msg} {Cm : CommitmentScheme Bool} {roots : List Claims.Msg}
    {c : Flock.Circuit} {L : LeafMap} {N : ℕ} {rows : Fin N → Option RowAt} {g : Fin N} {r : RowAt}
    (hr : rows g = some r) {W : ℕ} {v : FlockSoundness.Discharge.Layout.BitVal W} {y : List UInt8} {pos : ℕ}
    {b : Bool} (h : Cm.Opens H roots ⟨pos, b, dataAt c L r v y⟩ = true) :
    Cm.Opens H roots ⟨pos, b, dataOfRow c L rows g v y⟩ = true := by
  rw [dataOfRow_some hr]
  exact h

/-! ## Satisfying messages of a `--zk` table -/

section Messages

open FlockSoundness FlockSoundness.Model FlockSoundness.Discharge.ZkSession FlockSoundness.Discharge.ZkBind

theorem two_pow_split {m k L : ℕ} (hm : m = 7 + k + L) : 2 ^ m = 2 ^ L * 2 ^ k * 128 := by
  rw [show (128 : ℕ) = 2 ^ 7 by norm_num, ← pow_add, ← pow_add]; congr 1; omega

theorem packOf_lt {m k L : ℕ} (hm : m = 7 + k + L) (ℓ : Fin (2 ^ k)) (c : Fin (2 ^ L)) (v : Fin 128) :
    (c.val * 2 ^ k + ℓ.val) * 128 + v.val < 2 ^ m := by
  have := c.isLt; have := ℓ.isLt; have := v.isLt
  rw [two_pow_split hm]
  calc (c.val * 2 ^ k + ℓ.val) * 128 + v.val < (c.val * 2 ^ k + ℓ.val + 1) * 128 := by nlinarith
    _ ≤ (c.val * 2 ^ k + 2 ^ k) * 128 := Nat.mul_le_mul_right 128 (by omega)
    _ = (c.val + 1) * 2 ^ k * 128 := by ring
    _ ≤ 2 ^ L * 2 ^ k * 128 := Nat.mul_le_mul_right 128 (Nat.mul_le_mul_right _ (by omega))

/-- **The message packing a witness** at `m = 7 + k + L`: lane `ℓ`'s column `c` is the 128-bit field element of the
witness's bits from `(c · 2^k + ℓ) · 128` on (`execArith`'s `pack`). -/
noncomputable def packOf {m k L : ℕ} (hm : m = 7 + k + L) (z : Witness m) : Message FlockLevel3.GF128 k L :=
  fun ℓ c => execArith.pack fun v => z ⟨(c.val * 2 ^ k + ℓ.val) * 128 + v.val, packOf_lt hm ℓ c v⟩

/-- **A witness is the one its packing gives** (`witnessOf` inverts `packOf`). -/
theorem witnessOf_packOf {m k L : ℕ} (hm : m = 7 + k + L) (z : Witness m) :
    witnessOf execArith m k L (packOf hm z) = z := by
  funext a
  have ha : a.val < 2 ^ L * 2 ^ k * 128 := two_pow_split hm ▸ a.isLt
  have hc : a.val / 128 / 2 ^ k < 2 ^ L := by
    rw [Nat.div_lt_iff_lt_mul (Nat.two_pow_pos _), Nat.div_lt_iff_lt_mul (by norm_num)]
    linarith
  simp only [witnessOf, packOf]
  rw [dite_eq_left_of_eq_true (eq_true hc), execArith_correct.unpack_pack]
  congr 1
  apply Fin.ext
  simp only
  rw [Nat.div_add_mod' (a.val / 128) (2 ^ k), Nat.div_add_mod']

/-- A `--zk` table's schedule splits its `m`: `m = 7 + k₀ + cols₀` (`m - 7 = k₀ + cols₀`, `m = S.m ≥ 13`). -/
theorem tabZK_m {mPts : ℕ} (T : TabZK FlockLevel3.GF128 mPts) :
    T.S.m = 7 + T.sch.k0 + T.sch.level₀.logCols := by
  have h1 := T.hshape.2.2.1
  have h2 := T.hmsch
  have h3 := T.hm
  show T.S.m = 7 + T.sch.k0 + (T.sch.levels.headD default).logCols
  omega

/-- **The message of a `--zk` table packing a witness** of its statement (`packOf` at the table's schedule). -/
noncomputable def packZ {mPts : ℕ} (T : TabZK FlockLevel3.GF128 mPts) (z : Witness T.S.m) :
    Message FlockLevel3.GF128 T.sch.k0 T.sch.level₀.logCols :=
  packOf (tabZK_m T) z

theorem wit_packZ {mPts : ℕ} (T : TabZK FlockLevel3.GF128 mPts) (z : Witness T.S.m) :
    wit execArith T.S T.sch (packZ T z) = z :=
  witnessOf_packOf (tabZK_m T) z

end Messages

/-! ## One table at a constant draw -/

section OneTable

open FlockSoundness FlockSoundness.Model FlockSoundness.Discharge FlockSoundness.Discharge.Exec
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkSession FlockSoundness.Discharge.ZkHidden
  FlockSoundness.Discharge.ZkBind

/-- **A table set up at one draw is one table**, along equal inputs, drawn units and draws (`DrawSetupZHJP.table_eq`). -/
theorem _root_.FlockSoundness.Discharge.ZkHidden.DrawSetupZHJP.table_congr {n mPts : ℕ} {I₁ I₂ : Inputs}
    {own : Option ByteArray} {S₁ S₂ : Finset (Fin n)} {d₁ d₂ : Lean.Json} (y₁ : DrawSetupZHJP I₁ own mPts S₁ d₁)
    (y₂ : DrawSetupZHJP I₂ own mPts S₂ d₂) (hI : I₁ = I₂) (hS : S₁ = S₂) (hd : d₁ = d₂) : y₁.table = y₂.table := by
  subst hI hS hd
  exact DrawSetupZHJP.table_eq y₁ y₂

variable {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : Option ByteArray} {S₀ : Finset (Fin n)} {d₀ : Lean.Json}

theorem partOf_one (h1 : I.count ≤ 1) (d : Lean.Json) (v : ℕ) : (Composed.partOf I d v).val = 0 := by
  have h2 := (Composed.partOf I d v).isLt
  have h3 : nTab I = 1 := by simp [nTab, h1]
  omega

/-- **At one table and a constant draw, a drawn proof block's table is the table set up at that draw** (`tableZHJP`'s
first branch at table `0`, which proves every drawn unit: `partS` is `S`). -/
theorem tabOfZHJP_one (h1 : I.count ≤ 1) {d : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀)
    {S : Finset (Fin n)} (y : DrawSetupZHJP (inputsJ I 0) own mPts S d) (R : Reg) (u : Fin n) :
    tabOfZ (planZHJP (fun _ => d) y₀) (tabZHJP (fun _ => d) y₀) S R u = y.table := by
  rw [tabOfZHJP_eq]
  have hj := partOf_one h1 d u.val
  have hS : partS I d S (Composed.partOf I d u.val) = S :=
    Finset.filter_true_of_mem fun v _ => Fin.ext ((partOf_one h1 d v.val).trans hj.symm)
  have hd : drawJ I d (Composed.partOf I d u.val).val = d := by
    unfold drawJ
    rw [ite_eq_left h1]
  have hne : Nonempty (DrawSetupZHJP (inputsJ I (Composed.partOf I d u.val)) own mPts
      (partS I d S (Composed.partOf I d u.val)) (drawJ I d (Composed.partOf I d u.val))) := by
    rw [hS, hd, hj]
    exact ⟨y⟩
  have key : tableZHJP (fun _ => d) y₀ S (Composed.partOf I d u.val) = (Classical.choice hne).table := by
    unfold tableZHJP
    exact dite_eq_left hne
  rw [key]
  exact DrawSetupZHJP.table_congr _ _ (congrArg (inputsJ I) hj) hS hd

end OneTable

/-! ## The openings a session's message reveals, at public inputs -/

section V5

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge.Exec FlockSoundness.Refine
  FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.Placed
  FlockSoundness.Discharge.ZkHidden FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge
  FlockSoundness.Discharge.Composed FlockSoundness.Discharge.ZkSession

/-- **The openings a message of drawn proof block `u`'s table reveals**, at public inputs: `revealOf` under the roots of
the public file's leaf map, at each wire's bit in the opening the message gives of its commit string (`vbP`'s
`opening`), with the datum of the wire's registered row at that opening's row bits and salt. -/
noncomputable def revealV5 {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : Option ByteArray}
    (ht : I.tags.typed = false) {D : Flock.Desc} (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (hU : Built D.unroll.unit) {pi : Flock.HmIn.Ports} (hpi : Flock.HmIn.parse D.unroll = .ok pi)
    {pub : Flock.Public} (hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (M : Model.Message FlockLevel3.GF128 (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch.k0
      (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch.level₀.logCols) :
    List (List Claims.Msg × Opening Bool) :=
  let vb := vbP dj y₀ ht (Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) hscope hU hpi hpub
    (Reg := Reg)
  revealOf ((keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi))
    (fun g => constsOfP D.unroll hU n pi pub g) (rootsOfL D.unroll (LeafMap.ofPublic D.unroll pub n)) u
    (fun g => vb.bit (vb.val (vb.opening S R u M (vb.pos g))) g)
    fun g => dataOfRow D.unroll (LeafMap.ofPublic D.unroll pub n)
      (rowsOfLP D.unroll hU n pi (LeafMap.ofPublic D.unroll pub n)) g (vb.opening S R u M (vb.pos g)).2.1
      (vb.opening S R u M (vb.pos g)).2.2

theorem revealV5_length_le {n mPts : ℕ} {Reg : Type} {I : Inputs} {own : Option ByteArray}
    (ht : I.tags.typed = false) {D : Flock.Desc} (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D)
    (hU : Built D.unroll.unit) {pi : Flock.HmIn.Ports} (hpi : Flock.HmIn.parse D.unroll = .ok pi)
    {pub : Flock.Public} (hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    {d₀ : Lean.Json} (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) (S : Finset (Fin n)) (R : Reg) (u : Fin n)
    (M : Model.Message FlockLevel3.GF128 (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch.k0
      (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch.level₀.logCols) :
    (revealV5 ht hD hU hpi hpub hscope dj y₀ S R u M).length ≤
      (((keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)).io u).card :=
  revealOf_length_le _ _ _ _ _ _

/-- **A drawn proof block's satisfying message reveals openings that make it correct**, at public inputs: on its table's
satisfying message `M`, committed values that carry each opening of `revealV5` that opens under the deployed commitment
at the public file's leaf map make proof block `u` correct, whatever its interior values. Its values are the openings'
bits, the decoder's on `u`'s committed wires (`opening_spec`), on which `u` is correct (`loweringSoundZK_of_place`) and
which carry the constants (`openingsCanonV5_const`), and each opening is canonical (`openingsCanonV5`), so its row's
datum opens it (`transports_shapedP_at`, under `dupName` and `RootFixesDomainL`). -/
theorem correct_of_revealV5
    (n : ℕ) (Reg : Type)
    (I : Inputs) (own : Option ByteArray) (ht : I.tags.typed = false)
    (D : Flock.Desc) (hD : Flock.Desc.parse (zkTags I) I.circuitFile I.tables = .ok D) (hU : Built D.unroll.unit)
    (pi : Flock.HmIn.Ports) (hpi : Flock.HmIn.parse D.unroll = .ok pi)
    (pub : Flock.Public) (hpub : Flock.HmIn.loadPublic (zkTags I) D.unroll pi I.publicFile = .ok pub)
    (hscope : Discharge.Layout.scopeOkH D.unroll = true)
    (hdup : Flock.Registered.dupName D.unroll = none)
    (hroots : RootFixesDomainL D.unroll (LeafMap.ofPublic D.unroll pub n))
    (kd : ℕ) (dj : Finset (Fin n) → Lean.Json) (S₀ : Finset (Fin n)) (d₀ : Lean.Json)
    (y₀ : DrawSetupZHJP (inputsJ I 0) own (mPtsOf D.unroll.g (kd / nTab I)) S₀ d₀) :
    let P := (keyProgP D.unroll pi hU n).partition (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Cc := (keyProgP D.unroll pi hU n).circuit (Discharge.Layout.outsOfP D.unroll hU n pi)
    let Lm := LeafMap.ofPublic D.unroll pub n
    let κ : Fin Cc.N → Option Bool := fun g => constsOfP D.unroll hU n pi pub g
    let roots := rootsOfL D.unroll Lm
    let Cd := commitDepL Cc (rowsOfLP D.unroll hU n pi Lm) D.unroll Lm roots
    ∀ (S : Finset (Fin n)) (R : Reg) (u : Fin n), u ∈ S → ∀ M,
      (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S.Satisfies
        (wit execArith (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).S
          (tabOfZ (planZHJP dj y₀) (tabZHJP dj y₀) S R u).sch M) →
      ∀ (val : ℕ → Option Bool) (ι : ℕ → Bool),
        (∀ q ∈ revealV5 ht hD hU hpi hpub hscope dj y₀ S R u M, Cd.Opens sha512Msg q.1 q.2 = true →
          q.1 = roots → val q.2.pos = some q.2.val) →
        (progOf Cc κ).Correct ((progOf Cc κ).refIn (blockOf P κ u) val ι) (blockOf P κ u) := by
  intro P Cc Lm κ roots Cd S R u hu M hM val ι hval
  let vb := vbP dj y₀ ht (Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) hscope hU hpi hpub
    (Reg := Reg)
  have hlow := loweringSoundZK_of_place execArith P (planZHJP dj y₀) (tabZHJP dj y₀)
    (placeZHJP dj y₀ ht (Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) hscope hU hpi) S R u hu M hM
  refine correct_of_revealOf P κ roots u (fun g => vb.bit (vb.val (vb.opening S R u M (vb.pos g))) g)
    (fun g => dataOfRow D.unroll Lm (rowsOfLP D.unroll hU n pi Lm) g (vb.opening S R u M (vb.pos g)).2.1
      (vb.opening S R u M (vb.pos g)).2.2) sha512Msg Cd
    (fun g b h => constsOfP_isInput (Discharge.Layout.outsOfP D.unroll hU n pi) h) ?_ ?_ ?_ val ι hval
  · exact P.correct_congr (fun g hg => (vb.opening_spec S R u hu M hM g hg).2) hlow
  · exact fun g hg b hb =>
      openingsCanonV5_const n Reg I own ht D (Flock.Desc.parseStmt_of_parse hD) hU pi hpi pub hpub hscope kd dj S₀ d₀
        y₀ S R u hu M hM g hg b hb
  · intro g hg hk
    obtain ⟨hcom, -⟩ := vb.opening_spec S R u hu M hM g hg
    obtain ⟨⟨r, hr, hv, hvw⟩, hy⟩ :=
      (openingsCanonV5 n Reg I own ht D (Flock.Desc.parseStmt_of_parse hD) hU pi hpi pub hpub hscope kd dj S₀ d₀ y₀
        S R u hu M hM g hg).2 hk
    have hr' : rowsOfLP D.unroll hU n pi Lm g = some r := hr
    have hv' : FlockSoundness.Discharge.Layout.RowShaped (vb.opening S R u M (vb.pos g)).2.1 := hv
    have hvw' : (vb.opening S R u M (vb.pos g)).2.1.1.val = r.width := hvw
    have hy' : (vb.opening S R u M (vb.pos g)).2.2.length = saltLen := hy
    have hcom' : vb.com ((vb.opening S R u M (vb.pos g)).1, (vb.opening S R u M (vb.pos g)).2.1,
        (vb.opening S R u M (vb.pos g)).2.2) = vb.cs R (vb.pos g) := hcom
    exact opens_dataOfRow hr' (transports_shapedP_at (dj := dj) (y₀ := y₀) (ht := ht)
      (hc := Flock.Desc.parseStmt_stmt (Flock.Desc.parseStmt_of_parse hD)) (hscope := hscope) (hU := hU) (hpi := hpi)
      (hpub := hpub) R
      (rowsOk_of_scopeOkH hscope) (rootsDistinctL_of_fixes hdup hroots)
      (fun u q => ref_lt_cntP hpub u.isLt q.isLt) hr' (vb.opening S R u M (vb.pos g)).1
      (vb.opening S R u M (vb.pos g)).2.1 (vb.opening S R u M (vb.pos g)).2.2 hv' hvw' hy' hcom')

end V5

end Flock.Bridge
