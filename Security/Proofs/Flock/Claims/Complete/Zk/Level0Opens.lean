import Proofs.Flock.Claims.Complete.Zk.Level0Honest
import Proofs.Flock.Claims.Complete.Replay.LigeritoMerkle

/-!
# The honest masked Ligerito's openings pass the masked verifier's Merkle checks

`merkleOkZ_of_opens`: plain's `Lg.merkleOk_of_opens` for the masked verifier (`MerkleOkZ`). Level 0's commitment
(`commitPZk`) is a Merkle tree over rows `2^lanes + EXTRA_LANES` wide (`TreeW`, `commitRowsP_tree`), whose openings pass
the check at that width (`merkle_openW`). The masked prover's Ligerito opens every level honestly (`Lg.Opens`) at the
query coins of the rounds two after plain's, the pair's two coins coming between (`ligeritoPZk_opens`).
-/

namespace FlockComplete.L0

open Flock Transcript FlockComplete Replay Replay.Lg

/-! ## Level 0's commitment, as its Merkle tree -/

/-- **A commitment as its Merkle tree** over rows `W` wide: `Lg.Tree` at any row width. -/
structure TreeW (ops : MerkleOps V) (ms : MerkleScheme) (lv : Level) (W w : ℕ) (cm : Commit) : Prop where
  rows : cm.rows.size = 2 ^ lv.d
  rowW : ∀ q < 2 ^ lv.d, cm.rows[q]!.size = W
  saltW : ∀ q < 2 ^ lv.d, cm.salts[q]!.size = ms.saltBytes
  saltE : ms.saltBytes = 0 → ∀ q < 2 ^ lv.d, cm.salts[q]! = ByteArray.empty
  layer : ∀ h ≤ lv.d - lv.cap, cm.layers[h]!.size = 2 ^ (lv.d - h)
  width : ∀ h ≤ lv.d - lv.cap, ∀ x ∈ cm.layers[h]!, x.size = w
  leaf : ∀ q < 2 ^ lv.d, ops.leaf cm.rows[q]! cm.salts[q]! = .ok (cm.layers[0]!)[q]!
  node : ∀ h < lv.d - lv.cap, ∀ i < 2 ^ (lv.d - h - 1),
    ops.node (cm.layers[h]!)[2 * i]! (cm.layers[h]!)[2 * i + 1]! = .ok (cm.layers[h + 1]!)[i]!
  cap : cm.cap = cm.layers[lv.d - lv.cap]!

theorem commitRowsP_rows {hs : Hashes V} {ms : MerkleScheme} {lv : Level} {rows : Array (Array F128)}
    {salt : ℕ → ByteArray} {cm : Commit} (h : commitRowsP hs ms lv rows salt = .ok cm) : cm.rows = rows := by
  unfold commitRowsP at h
  obtain ⟨leaves, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, -, h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  rfl

/-- **A commitment of given rows is their Merkle tree**, through `Hashes` whose node and leaf digests are `w` bytes:
`Lg.commitP_tree` for `commitRowsP`, on `2^d` rows of `W` elements. -/
theorem commitRowsP_tree {hs : Hashes V} {ms : MerkleScheme} {lv : Level} {rows : Array (Array F128)} {W : ℕ}
    {salt : ℕ → ByteArray} {cm : Commit} {w : ℕ} (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = w)
    (hlf : ∀ row s v, (ms.opsWith hs).leaf row s = .ok v → v.size = w) (hcap : lv.cap ≤ lv.d)
    (hr : rows.size = 2 ^ lv.d) (hw : ∀ q < 2 ^ lv.d, rows[q]!.size = W)
    (h : commitRowsP hs ms lv rows salt = .ok cm) : TreeW (ms.opsWith hs) ms lv W w cm := by
  unfold commitRowsP at h
  obtain ⟨leaves, hlv, h⟩ := FlockSoundness.Refine.ebind_ok h
  obtain ⟨r, hr', h⟩ := FlockSoundness.Refine.ebind_ok h
  cases h
  obtain ⟨hls, hleaf⟩ := mapM_get hlv
  simp only [Array.size_range] at hls
  have hI := forIn_inv (lv.d - lv.cap) (#[leaves], leaves)
    (fun i (p : Array (Array ByteArray) × Array ByteArray) => p.1.size = i + 1 ∧ p.2 = p.1[i]! ∧
      p.1[0]! = leaves ∧ (∀ j ≤ i, p.1[j]!.size = 2 ^ (lv.d - j) ∧ ∀ x ∈ p.1[j]!, x.size = w) ∧
      ∀ j < i, ∀ k < 2 ^ (lv.d - j - 1),
        (ms.opsWith hs).node (p.1[j]!)[2 * k]! (p.1[j]!)[2 * k + 1]! = .ok (p.1[j + 1]!)[k]!)
    ⟨rfl, rfl, rfl, fun j hj => by
      obtain rfl : j = 0 := by omega
      refine ⟨by simpa using hls, fun x hx => ?_⟩
      have hx' : x ∈ leaves := by simpa using hx
      obtain ⟨q, hq, rfl⟩ := Array.mem_iff_getElem.1 hx'
      have := hleaf q (by simp; omega)
      rw [getElem!_pos leaves q hq] at this
      exact hlf _ _ _ this, fun j hj => absurd hj (Nat.not_lt_zero _)⟩
    (fun i hi p r' hp e => by
      obtain ⟨cur', hc, e⟩ := FlockSoundness.Refine.ebind_ok e
      cases e
      obtain ⟨h1, h2, h3, h4, h5⟩ := hp
      obtain ⟨hcs, hcn⟩ := mapM_get hc
      simp only [Array.size_range] at hcs
      have hp2 : p.2.size = 2 ^ (lv.d - i) := by rw [h2]; exact (h4 i le_rfl).1
      have hd : lv.d - i = lv.d - (i + 1) + 1 := by omega
      have hsz : cur'.size = 2 ^ (lv.d - (i + 1)) := by
        rw [hcs, hp2, hd, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]
      have hget : ∀ j ≤ i, (p.1.push cur')[j]! = p.1[j]! := fun j hj => push_get_lt _ (by omega)
      have hlast : (p.1.push cur')[i + 1]! = cur' := by rw [← h1]; exact push_get_eq _ _
      refine ⟨_, rfl, by simp [h1], hlast.symm, by rw [hget 0 (Nat.zero_le _)]; exact h3, fun j hj => ?_,
        fun j hj k hk => ?_⟩
      · rcases Nat.lt_or_eq_of_le hj with hj | rfl
        · rw [hget j (by omega)]
          exact h4 j (by omega)
        · rw [hlast]
          refine ⟨hsz, fun x hx => ?_⟩
          obtain ⟨k, hk, rfl⟩ := Array.mem_iff_getElem.1 hx
          have := hcn k (by rw [Array.size_range, ← hcs]; exact hk)
          rw [getElem!_pos cur' k hk] at this
          exact hn _ _ _ this
      · rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.1 hj) with hj | rfl
        · rw [hget j (by omega), hget (j + 1) (by omega)]
          exact h5 j hj k hk
        · rw [hget j le_rfl, hlast, ← h2]
          have := hcn k (by
            rw [Array.size_range, hp2, hd, Nat.pow_succ, Nat.mul_div_cancel _ (by norm_num)]
            rwa [Nat.sub_sub] at hk)
          simpa using this) hr'
  obtain ⟨-, hcur, h0, hL, hN⟩ := hI
  refine ⟨hr, hw, fun q hq => ?_, fun h0 q hq => ?_, fun j hj => (hL j hj).1,
    fun j hj => (hL j hj).2, fun q hq => ?_, hN, hcur⟩
  · rw [getElem!_pos _ q (by simp; omega)]
    simp only [Array.getElem_map, Array.getElem_range]
    exact fitBytes_size' _ _
  · rw [getElem!_pos _ q (by simp; omega)]
    simp only [Array.getElem_map, Array.getElem_range]
    exact bytes_empty (by rw [fitBytes_size', h0])
  · have := hleaf q (by simp; omega)
    simp only [Array.getElem_range] at this
    rw [h0]
    exact this

/-- **Merkle completeness at any row width**: `Lg.merkle_open` for a tree over rows `W` wide, checked at width `W`. -/
theorem merkle_openW {ops : MerkleOps V} {ms : MerkleScheme} {lv : Level} {W : ℕ} {cm : Commit}
    (ht : TreeW ops ms lv W ms.digestLen cm) (hcap : lv.cap ≤ lv.d) {pos : Array ℕ} (hp : ∀ p ∈ pos, p < 2 ^ lv.d) :
    merkleCheckWith ops ms cm.cap lv.cap lv.d pos (openP ms cm lv pos).rows (openP ms cm lv pos).siblings
      (openP ms cm lv pos).salts W = .ok () := by
  have hD : lv.d - lv.cap ≤ lv.d := Nat.sub_le _ _
  have hcs : cm.cap.size = 2 ^ lv.cap := by
    rw [ht.cap, ht.layer _ le_rfl]
    congr 1
    omega
  have hcw : ∀ x ∈ cm.cap, x.size = ms.digestLen := by
    rw [ht.cap]
    exact ht.width _ le_rfl
  let blk : ℕ → Array ByteArray := fun p => (Array.range (lv.d - lv.cap)).map fun h => (cm.layers[h]!)[(p >>> h) ^^^ 1]!
  have hblk : ∀ p, (blk p).toList.length = lv.d - lv.cap := by simp [blk]
  have hsib : (openP ms cm lv pos).siblings.toList = pos.toList.flatMap fun p => (blk p).toList := by
    show (pos.foldl (fun acc p => acc ++ blk p) #[]).toList = _
    rw [← Array.foldl_toList, foldl_append_toList]
    simp
  have hsz : (openP ms cm lv pos).siblings.size = pos.size * (lv.d - lv.cap) := by
    rw [← Array.length_toList, hsib, List.length_flatMap]
    simp only [hblk, List.map_const', List.sum_replicate, smul_eq_mul, Array.length_toList]
  have hsw : ∀ x ∈ (openP ms cm lv pos).siblings, x.size = ms.digestLen := by
    intro x hx
    rw [← Array.mem_toList_iff, hsib, List.mem_flatMap] at hx
    obtain ⟨p, hpm, hx⟩ := hx
    simp only [blk, Array.toList_map, List.mem_map, Array.mem_toList_iff, Array.mem_range] at hx
    obtain ⟨h, hh, rfl⟩ := hx
    have hp' := hp p (Array.mem_toList_iff.1 hpm)
    refine ht.width h (by omega) _ (mem_of_get ?_)
    rw [ht.layer h (by omega)]
    have hd : 1 ≤ lv.d - h := by omega
    exact Nat.xor_lt_two_pow (shiftRight_lt hp' (by omega)) (Nat.one_lt_two_pow (by omega))
  have hsibs : ∀ j < pos.size, sibsOf (openP ms cm lv pos).siblings (lv.d - lv.cap) j = (blk pos[j]!).toList := by
    intro j hj
    unfold sibsOf
    rw [Array.toList_extract, hsib, List.extract_eq_take_drop, Nat.add_sub_cancel_left,
      flatMap_block (fun p => (blk p).toList) hblk pos.toList j (by simpa using hj)]
    simp [getElem!_pos pos j hj]
  have hopen : ∀ j < pos.size, opensWith ops ms cm.cap lv.cap lv.d W pos[j]!
      (openP ms cm lv pos).rows[j]! ((openP ms cm lv pos).salts.getD j ByteArray.empty)
      (sibsOf (openP ms cm lv pos).siblings (lv.d - lv.cap) j) = .ok true := by
    intro j hj
    have hpj : pos[j]! < 2 ^ lv.d := hp _ (mem_of_get hj)
    have hrow : (openP ms cm lv pos).rows[j]! = cm.rows[pos[j]!]! := by
      show (pos.map fun p => cm.rows[p]!)[j]! = _
      rw [getElem!_pos _ j (by simpa using hj), Array.getElem_map, getElem!_pos pos j hj]
    have hsalt : (openP ms cm lv pos).salts.getD j ByteArray.empty = cm.salts[pos[j]!]! := by
      by_cases h0 : ms.saltBytes = 0
      · rw [ht.saltE h0 _ hpj]
        simp [openP, h0]
      · simp [openP, h0, Array.getD, hj]
    rw [hrow, hsalt, hsibs j hj]
    unfold opensWith
    have hc : (decide (cm.rows[pos[j]!]!.size = W) && decide ((blk pos[j]!).toList.length =
        lv.d - lv.cap) && decide (cm.salts[pos[j]!]!.size = ms.saltBytes)) = true := by
      simp [ht.rowW _ hpj, ht.saltW _ hpj, blk]
    rw [ite_eq_left hc, ht.leaf _ hpj]
    have hcl := climb_ok hD ht.node hpj (lv.d - lv.cap) 0 (Nat.zero_add _)
    rw [Nat.shiftRight_zero] at hcl
    have hbl : (blk pos[j]!).toList = (List.range' 0 (lv.d - lv.cap)).map fun h =>
        (cm.layers[h]!)[(pos[j]! >>> h) ^^^ 1]! := by
      simp [blk, List.range_eq_range']
    have hidx : pos[j]! >>> (lv.d - lv.cap) < cm.cap.size := by
      rw [hcs]
      have := shiftRight_lt hpj hD
      rwa [show lv.d - (lv.d - lv.cap) = lv.cap by omega] at this
    rw [← ht.cap] at hcl
    show (climbWith ops ((cm.layers[0]!)[pos[j]!]!) pos[j]! (blk pos[j]!).toList >>= fun top =>
      pure (decide (top = cm.cap.getD (pos[j]! >>> (lv.d - lv.cap)) ByteArray.empty))) = _
    rw [hbl, hcl]
    show Except.ok (decide (_ = _)) = _
    rw [Array.getD, dite_eq_left hidx, getElem!_pos cm.cap _ hidx]
    simp
  unfold merkleCheckWith
  have h1 : (cm.cap.size != 2 ^ lv.cap || (openP ms cm lv pos).rows.size != pos.size ||
      (openP ms cm lv pos).siblings.size != pos.size * (lv.d - lv.cap) ||
      (openP ms cm lv pos).salts.size != saltCount ms pos.size) = false := by
    have hrs : (openP ms cm lv pos).rows.size = pos.size := by simp [openP]
    have hss : (openP ms cm lv pos).salts.size = saltCount ms pos.size := by
      by_cases h0 : ms.saltBytes = 0 <;> simp [openP, saltCount, h0]
    rw [hcs, hrs, hsz, hss]
    simp
  have h2 : (cm.cap.any (·.size != ms.digestLen) || (openP ms cm lv pos).siblings.any (·.size != ms.digestLen)) =
      false := by
    simp only [Bool.or_eq_false_iff, Array.any_eq_false]
    exact ⟨fun i hi => by simp [hcw _ (Array.getElem_mem hi)], fun i hi => by simp [hsw _ (Array.getElem_mem hi)]⟩
  rw [h1, h2]
  simp only [Bool.false_eq_true, ↓reduceIte]
  rw [findM_none (List.range pos.size) fun j hj => by
    rw [List.mem_range] at hj
    show (opensWith ops ms cm.cap lv.cap lv.d W pos[j]! _ _ _ >>= fun o => pure !o) = _
    rw [hopen j hj]
    rfl]
  rfl

/-- **Level 0's masked commitment is its Merkle tree** over rows `2^lanes + EXTRA_LANES` wide. -/
theorem commitPZk_tree {hs : Hashes V} {ms : MerkleScheme} {lv : Level} {tPad : ℕ} {g : Array F128}
    {extra lanePad : ℕ → ℕ → F128} {salt : ℕ → ByteArray} {cm : Commit} {w : ℕ}
    (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = w)
    (hlf : ∀ row s v, (ms.opsWith hs).leaf row s = .ok v → v.size = w) (hcap : lv.cap ≤ lv.d)
    (h : commitPZk hs ms lv tPad g extra lanePad salt = .ok cm) :
    TreeW (ms.opsWith hs) ms lv (2 ^ lv.lanes + Zk.EXTRA_LANES) w cm := by
  obtain ⟨hr, hw⟩ := encodeRowsZk_shape lv tPad g extra lanePad
  refine commitRowsP_tree hn hlf hcap hr (fun q hq => hw _ ?_) h
  rw [getElem!_pos _ q (by omega)]
  exact Array.getElem_mem _

/-! ## The masked prover's openings -/

/-- A refusal followed by anything never returns. -/
theorem KRuns.throwBind {L : ℕ} {α β : Type} (e : String) (f : α → LP PS β) (k k' : ℕ) (Q : β → Prop) :
    KRuns L (throw e >>= f) k k' Q := by
  intro st t l b st' t' l' _ _ h
  obtain ⟨_, _, _, _, h0, _⟩ := lp_bind_ok h
  cases h0

/-- **The honest masked prover's Ligerito opens honestly**: run from round `k` with `L + k` bodies sent, it opens
level `j`'s commitment at the positions of round `k + 2 + qOff sch j`'s coins, the replies' coins (`cfOf resp L`):
plain's `ligeritoP_opens`, two rounds later, after the pair's coins. -/
theorem ligeritoPZk_opens (resp : ℕ → Claims.Msg) (h : ByteArray → ByteArray) (ms : MerkleScheme) (sch : Schedule)
    (T : F128) (cm0 : Commit) (tb b : Array F128) (lay : Zk.PadLayout) (hv : Array F128) (rep tPad : ℕ)
    (extra lanePad : ℕ → ℕ → F128) (salt : ℕ → ℕ → ByteArray) (nonce : ℕ → ℕ → UInt64) (L : ℕ)
    (hn : sch.n ≠ 0) (hk0 : 1 ≤ sch.k0) (hL : 2 ≤ sch.levels.size)
    (hq : ∀ j < sch.levels.size, (sch.levels.getD j default).q ≠ 0) {t : Transcript} {l : Array Claims.Msg}
    (hl : l.size = L + t.k) {P : LigeritoProof} {zl : Zk.Level0Msgs} {rc : Zk.L0Rec} {t' : Transcript}
    {l' : Array Claims.Msg}
    (hrun : ((ligeritoPZk (asksS resp) (Hashes.ofSha512 h) ms sch T cm0 tb b lay hv rep tPad extra lanePad salt
      nonce).run t).go l = (.ok ((P, zl, rc), t'), l')) :
    Opens (Hashes.ofSha512 h : Hashes V) ms sch cm0 P (cfOf resp L) (t.k + 2) := by
  unfold ligeritoPZk at hrun
  rw [bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l), bind_go (liftTM_ok (Flock.RecAlgebra.absorbF_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.absorbBytes_eq _ _) l),
    bind_go (liftTM_ok (Flock.RecAlgebra.label_eq _ _) l)] at hrun
  dsimp only at hrun
  rw [StateT.run_bind] at hrun
  obtain ⟨⟨⟨⟨caps, rproofs, fin, op0, zl', initCh, rho, b0', y0⟩, stF⟩, t₁⟩, l₁, h1, h2⟩ := JsonRT.go_bind_ok hrun
  obtain ⟨h21, rfl⟩ := Prod.mk.inj h2
  obtain ⟨hP3, rfl⟩ := Prod.mk.inj (Except.ok.inj h21)
  obtain ⟨hP, -⟩ := Prod.mk.inj hP3
  subst hP
  set R := sch.levels.size - 1 with hR
  set k := t.k
  set hsV : Hashes V := Hashes.ofSha512 h
  have hR1 : 1 ≤ R := by omega
  have key := (?prog : KRuns (β := Array (Array ByteArray) × Array LevelProof × FinalProof × LevelProof ×
      Zk.Level0Msgs × Array F256 × (F256 × F256) × F128 × F128) L _ k
    (k + 2 + sch.k0 + 9 + 11 * (R - 1) + 7) fun r =>
      r.2.2.2.1 = openP ms cm0 (sch.levels.getD 0 default) (posAt sch (cfOf resp L) (k + 2) 0) ∧
        LoopOk hsV ms sch (cfOf resp L) (k + 2) R R (0, default, r.1, r.2.1, r.2.2.1)) _ _ l _ _ _ _ (by rfl) hl h1
  case prog =>
    refine (KRuns.sampleVec resp hn _).bind fun z _ => KRuns.oodPush _ ?_
    refine (KRuns.pow nonce 2 _).bind fun _ _ => (KRuns.sample resp _).bind fun _ _ =>
      (KRuns.modify _ _).bind fun _ _ => (KRuns.get _).bind fun _ _ => (KRuns.sendMsg _ _).bind fun _ _ => ?_
    refine KRuns.bind ((KRuns.loop (σ := Array F256 × (F256 × F256) × Zk.Level0Msgs) _
      (fun j => k + 1 + 1 + j + if j = sch.k0 then 2 else 0) (fun _ _ => True) sch.k0 ?lane _ trivial).cast
      (by rw [ite_eq_right (by omega)]) (by rw [ite_eq_left rfl])) fun _ _ => ?_
    case lane =>
      intro j hj s _
      refine (KRuns.sampleK resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => ?_
      by_cases hj1 : j + 1 = sch.k0
      · refine KRuns.ite (fun _ => ?_) (fun hne => absurd (by simp [hj1]) hne)
        refine (KRuns.get _).bind fun _ _ => ?_
        refine (KRuns.absorb _ (pKv _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ =>
          (KRuns.absorb _ (pKv _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ => ?_
        refine (KRuns.sampleK resp _).bind fun _ _ => (KRuns.sampleK resp _).bind fun _ _ => ?_
        refine KRuns.ite (fun _ => KRuns.throwBind _ _ _ _ _) (fun _ => ?_)
        refine (KRuns.absorb _ (pLabel _) (fun t => Flock.RecAlgebra.label_eq _ t) _).bind fun _ _ =>
          (KRuns.absorb _ (pKv _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ =>
          (KRuns.absorb _ (pSlice _) (fun t => Flock.RecAlgebra.absorbSlice_eq _ t) _).bind fun _ _ => ?_
        exact (KRuns.modify _ _).bind fun _ _ => (KRuns.get _).bind fun _ _ => (KRuns.sendMsg _ _).bind fun _ _ =>
          KRuns.pure ⟨_, rfl, trivial⟩ (by split_ifs <;> omega)
      · refine KRuns.ite (fun hc => absurd (by simpa using hc) hj1) fun _ => ?_
        exact (KRuns.get _).bind fun _ _ => (KRuns.sendMsg _ _).bind fun _ _ =>
          KRuns.pure ⟨_, rfl, trivial⟩ (by split_ifs <;> omega)
    refine (KRuns.get _).bind fun d _ => (KRuns.commit h ms _ _ _ _).bind fun cm1 hcm1 => ?_
    refine (KRuns.absorb _ (pBytes _) (fun t => Flock.RecAlgebra.absorbBytes_eq _ t) _).bind fun _ _ => ?_
    refine (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ =>
      (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ => ?_
    refine (KRuns.query resp nonce ms _ cm0 (hq 0 (by omega)) _).bind fun x hx => ?_
    refine (KRuns.sendMsg _ _).bind fun _ _ => (KRuns.pow nonce 2 _).bind fun _ _ =>
      (KRuns.sample resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => ?_
    have hx0 : x.1 = openP ms cm0 (sch.levels.getD 0 default) (posAt sch (cfOf resp L) (k + 2) 0) := by
      rw [hx]
      unfold posAt cfOf qOff
      rw [ite_eq_left (by omega)]
      congr 5
    refine KRuns.bind (KRuns.cast (KRuns.loop (σ := ℕ × Commit × Array (Array ByteArray) × Array LevelProof × FinalProof)
      _ (fun j => if j < R then k + 2 + sch.k0 + 9 + 11 * j else k + 2 + sch.k0 + 9 + 11 * (R - 1) + 7)
      (LoopOk hsV ms sch (cfOf resp L) (k + 2) R) R ?body _ ⟨fun _ => ⟨⟨_, _, hcm1⟩, rfl, rfl, rfl⟩,
        fun i _ hi => absurd hi (Nat.not_lt_zero _), fun h0 => absurd h0 (by omega)⟩)
      (by rw [ite_eq_left (by omega)]) (by rw [ite_eq_right (by omega)])) fun σ hσ => KRuns.pure ⟨hx0, ?fin⟩ rfl
    case fin =>
      obtain ⟨-, h2, h3⟩ := hσ
      exact ⟨fun h => absurd h (lt_irrefl _), h2, h3⟩
    case body =>
      intro j hj s hs
      obtain ⟨hcur, hpre, -⟩ := hs
      obtain ⟨⟨g₀, salt₀, hcm⟩, hcs, hcl, hrs⟩ := hcur hj
      rw [ite_eq_left hj]
      by_cases hj1 : j + 1 < R
      · rw [ite_eq_left hj1]
        refine (levelChalsK resp _).bind fun f _ => ?_
        refine KRuns.ite (fun _ => ?_) fun hne => absurd (by simp; omega) hne
        refine (KRuns.commit h ms _ _ _ _).bind fun cmN hcmN => ?_
        refine (KRuns.absorb _ (pBytes _) (fun t => Flock.RecAlgebra.absorbBytes_eq _ t) _).bind fun _ _ => ?_
        refine (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ =>
          (KRuns.oodIntro resp nonce _ (Nat.succ_ne_zero _) _).bind fun _ _ => ?_
        refine (KRuns.query resp nonce ms _ s.2.1 (hq (j + 1) (by omega)) _).bind fun q hq' => ?_
        refine (KRuns.sendMsg _ _).bind fun _ _ => (KRuns.pow nonce 2 _).bind fun _ _ =>
          (KRuns.sample resp _).bind fun _ _ => (KRuns.modify _ _).bind fun _ _ => KRuns.pure ?_ (by omega)
        have hop : LvOpen hsV ms sch (cfOf resp L) (k + 2) (j + 1) (s.2.2.1.push cmN.cap)[j]!
            ((s.2.2.2.1.push q.1)[j]!) := by
          rw [push_get_lt _ (by omega), push_get_eq' _ hrs, hcl]
          refine ⟨s.2.1, g₀, salt₀, hcm, rfl, ?_⟩
          rw [hq']
          unfold posAt cfOf qOff
          rw [ite_eq_left (by omega)]
          congr 5
          omega
        refine ⟨_, rfl, fun _ => ⟨⟨_, _, hcmN⟩, by simp [hcs], ?_, by simp [hrs]⟩, fun i hi1 hi => ?_,
          fun h => absurd h (by omega)⟩
        · rw [← hcs]
          exact push_get_eq _ _
        · rcases Nat.lt_or_eq_of_le (Nat.lt_succ_iff.1 hi) with hi | rfl
          · rw [push_get_lt _ (by omega), push_get_lt _ (by omega)]
            exact hpre i hi1 hi
          · exact hop
      · rw [ite_eq_right hj1]
        have hjR : j + 1 = R := by omega
        refine (levelChalsK resp _).bind fun f _ => ?_
        refine KRuns.ite (fun hc => absurd hc (by simp [hjR])) fun _ => ?_
        refine (KRuns.absorbAll _ _).bind fun _ _ => ?_
        refine (KRuns.query resp nonce ms _ s.2.1 (hq _ (by omega)) _).bind fun q hq' => ?_
        refine (KRuns.pow nonce 2 _).bind fun _ _ => (KRuns.sample resp _).bind fun _ _ => KRuns.pure ?_ (by omega)
        refine ⟨_, rfl, fun h => absurd h (by omega), fun i hi1 hi => hpre i hi1 (by omega), fun _ => ?_⟩
        dsimp only
        refine ⟨by omega, by omega, ?_⟩
        rw [show R - 1 = j by omega, hcl]
        refine ⟨s.2.1, g₀, salt₀, by rw [← hjR]; exact hcm, rfl, ?_⟩
        rw [show (⟨q.1.rows, q.1.siblings, q.1.salts⟩ : LevelProof) = q.1 from rfl, hq']
        unfold posAt cfOf qOff
        rw [ite_eq_right (by omega)]
        congr 5
        omega
  obtain ⟨-, -, hop0, -, hloop, hfin⟩ := key
  obtain ⟨hcs, hrs, hlast⟩ := hfin rfl
  exact ⟨hop0, hcs, hrs, fun i hi => hloop i hi (by omega), hlast⟩

/-! ## Every level's check passes at the honest openings -/

/-- **Every level's Merkle check passes at the masked prover's openings** (`MerkleOkZ`): level `0`'s of a tree over
rows `2^lanes + EXTRA_LANES` wide, every later level's of its own (`Lg.Opens`), through a scheme whose node and leaf
digests are its width, at levels that cap at most their depth and query below `2^d`. -/
theorem merkleOkZ_of_opens {hs : Hashes V} {ms : MerkleScheme} {sch : Schedule} {cm0 : Commit} {p : LigeritoProof}
    {cf : ℕ → ℕ → F128} {k : ℕ} (hn : ∀ l r v, (ms.opsWith hs).node l r = .ok v → v.size = ms.digestLen)
    (hlf : ∀ row s v, (ms.opsWith hs).leaf row s = .ok v → v.size = ms.digestLen) (hL : 2 ≤ sch.levels.size)
    (hcap : ∀ j < sch.levels.size, (sch.levels.getD j default).cap ≤ (sch.levels.getD j default).d)
    (hpos : ∀ j < sch.levels.size, ∀ coins, ∀ x ∈ positions (sch.levels.getD j default) coins,
      x < 2 ^ (sch.levels.getD j default).d)
    (h0 : TreeW (ms.opsWith hs) ms (sch.levels.getD 0 default)
      (2 ^ (sch.levels.getD 0 default).lanes + Zk.EXTRA_LANES) ms.digestLen cm0)
    (ho : Opens hs ms sch cm0 p cf k) : MerkleOkZ hs ms sch cm0.cap p cf k := by
  obtain ⟨h0p, -, -, hin, hfin⟩ := ho
  refine ⟨?_, fun j hj hj0 => ?_⟩
  · rw [h0p]
    exact merkle_openW h0 (hcap 0 (by omega)) (hpos 0 (by omega) _)
  · have hj0' : j ≠ 0 := by omega
    by_cases hjR : j < sch.levels.size - 1
    · obtain ⟨cm, g', salt', hc, hcc, hop⟩ := hin (j - 1) (by omega)
      rw [Nat.sub_add_cancel (show 1 ≤ j by omega)] at hc hop
      have hoj : opening p cm0.cap (sch.levels.size - 1) j = (p.recursiveCaps[j - 1]!,
          p.recursiveProofs[j - 1]!.rows, p.recursiveProofs[j - 1]!.siblings, p.recursiveProofs[j - 1]!.salts) := by
        unfold opening
        rw [ite_eq_right hj0', ite_eq_left hjR]
      rw [hoj]
      dsimp only
      rw [hop, ← hcc]
      exact merkle_open (commitP_tree hn hlf (hcap j hj) hc) (hcap j hj) (hpos j hj _)
    · obtain rfl : j = sch.levels.size - 1 := by omega
      obtain ⟨cm, g', salt', hc, hcc, hop⟩ := hfin
      have hoj : opening p cm0.cap (sch.levels.size - 1) (sch.levels.size - 1) =
          (p.recursiveCaps[sch.levels.size - 1 - 1]!, p.finalProof.rows, p.finalProof.siblings,
            p.finalProof.salts) := by
        unfold opening
        rw [ite_eq_right hj0', ite_eq_right (lt_irrefl _)]
      rw [hoj]
      dsimp only
      have e1 := congrArg LevelProof.rows hop
      have e2 := congrArg LevelProof.siblings hop
      have e3 := congrArg LevelProof.salts hop
      dsimp only at e1 e2 e3
      rw [e1, e2, e3, ← hcc]
      exact merkle_open (commitP_tree hn hlf (hcap _ hj) hc) (hcap _ hj) (hpos _ hj _)

end FlockComplete.L0
