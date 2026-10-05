import Proofs.Flock.Soundness.Refine.Regions

/-!
# The salted scheme `hm96-sha512/v1` refines the model's salted leaves (R7 and R8 for hm96; PROTOCOL.md §14)

`MerkleScheme.hm96Sha512` hashes an opened row's leaf as `SHA-512(leaf_prefix ‖ b ‖ c)`, with `x = SHA-512(row)`,
`b = x ⊕ M·y` and `c = SHA-512(salt_prefix ‖ y)` for the opening's 192-byte salt `y` (`Hm96.Default512`). Its nodes are
SHA-512 of the pair, as the unsalted scheme's are. The compiled model's `Merkle.Leaf.hm96 K` is that leaf for a key `K`.
`K512` is the executable's own key: its prefixes and its mask are `Hm96.Default512`'s, so the two leaves are equal
(`hm96_leaf`).

* **R7:** a passed hm96 Merkle check is the model's salted path check (`merkleCheck_verifiesL`), and a rep whose requests
  passed it is `OpensOKL` at the proof's rows and salts (`opensOKL_of`).
* **R8:** `verify_refines_hm96` is `verify_refines` for the salted scheme, with `OpensOKL` at the proofs' salts.
* **R9:** `verify_refines_ofCircuit_hm96`, for the verifier's own statements. For `Stmt.setupH`'s, `setupH_wf` gives its
  layout and region hypotheses.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript Model Accounting

local notation "𝔽" => FlockLevel3.GF128

/-! ## The executable's key -/

theorem foldl_size {α : Type} (g : ByteArray → α → ByteArray) (c : ℕ) (hg : ∀ r x, (g r x).size = r.size + c) :
    ∀ (l : List α) (r : ByteArray), (l.foldl g r).size = r.size + c * l.length
  | [], r => by simp
  | x :: l, r => by
    rw [List.foldl_cons, foldl_size g c hg l, hg, List.length_cons, Nat.mul_succ]; omega

theorem mask_out_size (acc : Array UInt64) :
    (Id.run do
      let mut out := ByteArray.emptyWithCapacity 64
      for k in [0:8] do
        for j in [0:8] do out := out.push ((acc[k]! >>> (8 * j).toUInt64) &&& 0xff).toUInt8
      return out).size = 64 := by
  simp only [Id.run, Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl, Std.Legacy.Range.size,
    pure_bind, bind_pure]
  show ByteArray.size (List.foldl _ _ _) = 64
  rw [foldl_size _ 8 (fun r k => by rw [foldl_size _ 1 (fun r j => ByteArray.size_push ..)]; simp)]
  rfl

/-- The mask `M·y` is 64 bytes, whatever the salt. -/
theorem mask_size (salt : ByteArray) : (Hm96.Default512.mask salt).size = 64 :=
  mask_out_size _

theorem length_data (b : ByteArray) : b.data.toList.length = b.size := by
  rw [Array.length_toList, ByteArray.size_data]

/-- **The executable's hm96 key** (`Hm96.Default512`): its leaf and salt prefixes, and its mask `y ↦ M·y`, 64 bytes. -/
def K512 : Merkle.Hm96Key where
  leafPrefix := Hm96.Default512.leafPrefix.data.toList
  saltPrefix := Hm96.Default512.saltPrefix.data.toList
  mask y := (Hm96.Default512.mask ⟨y.toArray⟩).data.toList
  L := 64
  mask_len y := by rw [length_data, mask_size]

theorem hL512 (d : Dig) : (enc512.dig d).length = K512.L := by
  show d.1.data.toList.length = 64
  rw [length_data, d.2]

/-- hm96's leaf scheme at the executable's key. -/
abbrev leaf512 : Merkle.Leaf (List 𝔽) (List UInt8) Dig H512 := Merkle.Leaf.hm96 H512 enc512 K512 hL512

theorem data_append (a b : ByteArray) : (a ++ b).data.toList = a.data.toList ++ b.data.toList := by
  rw [ByteArray.data_append, Array.toList_append]

theorem get!_data (a : ByteArray) (i : ℕ) (hi : i < a.size) :
    a.get! i = a.data.toList[i]'(by rw [Array.length_toList, ByteArray.size_data]; exact hi) := by
  rw [Array.getElem_toList]
  simp [ByteArray.get!, getElem!_pos a.data i (by rw [ByteArray.size_data]; exact hi)]

/-- The executable's bytewise xors of two `n`-byte strings are the model's `xorBytes`. -/
theorem xor_range (a b : ByteArray) (n : ℕ) (ha : a.size = n) (hb : b.size = n) :
    (ByteArray.mk ((Array.range n).map fun i => a.get! i ^^^ b.get! i)).data.toList =
      Merkle.xorBytes a.data.toList b.data.toList := by
  show ((Array.range n).map fun i => a.get! i ^^^ b.get! i).toList = _
  rw [Array.toList_map, Array.toList_range]
  apply List.ext_getElem
  · simp only [List.length_map, List.length_range, Merkle.xorBytes, List.length_zipWith, Array.length_toList,
      ByteArray.size_data, ha, hb, Nat.min_self]
  · intro i h1 h2
    have hi : i < n := by simpa using h1
    simp only [List.getElem_map, List.getElem_range, Merkle.xorBytes, List.getElem_zipWith]
    rw [get!_data a i (by omega), get!_data b i (by omega)]

theorem dig_eq (d : Dig) : enc512.dig d = d.1.data.toList := rfl

theorem col_eq (c : List F128) : enc512.col c = colBytes c := rfl

theorem H512_val (l : List UInt8) : (H512 l).1 = Sha512.hash ⟨l.toArray⟩ := rfl

theorem leaf512_leaf (c : List F128) (y : List UInt8) :
    leaf512.leaf c y = H512 (K512.leafPrefix ++ (Merkle.xorBytes (enc512.dig (H512 (enc512.col c))) (K512.mask y) ++
      enc512.dig (H512 (K512.saltPrefix ++ y)))) := rfl

theorem K512_mask (y : List UInt8) : K512.mask y = (Hm96.Default512.mask ⟨y.toArray⟩).data.toList := rfl

/-- **The executable's salted leaf is the model's** at the executable's key: for an opened row and its salt,
`hm96Sha512.leaf` is `Leaf.hm96 K512`'s leaf of the row's words under the salt's bytes. -/
theorem hm96_leaf (row : Array F128) (salt : ByteArray) :
    (leaf512.leaf row.toList salt.data.toList).1 = MerkleScheme.hm96Sha512.leaf row salt := by
  rw [leaf512_leaf, H512_val, dig_eq, dig_eq, H512_val, H512_val, col_eq, ← rowBytes_data, bytes_of_list, K512_mask,
    bytes_of_list, ← xor_range _ _ 64 (sha512_size _) (mask_size salt)]
  show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ (_ ++
      (Sha512.hash ⟨(Hm96.Default512.saltPrefix.data.toList ++ salt.data.toList).toArray⟩).data.toList)).toArray⟩ =
    Sha512.hash (Hm96.Default512.leafPrefix ++ Hm96.Default512.commitString (Sha512.hash (MerkleScheme.rowBytes row)) salt)
  rw [← data_append Hm96.Default512.saltPrefix salt, bytes_of_list, ← data_append, ← data_append, bytes_of_list]
  rfl

/-! ## A passed hm96 check is the model's salted path check -/

/-- hm96's tree climbs as the unsalted SHA-512 tree does: the node hashes are equal. -/
theorem climb_hm96 (msId : String) : ∀ (sibs : List ByteArray) (idx : ℕ) (acc : ByteArray),
    climb MerkleScheme.hm96Sha512 acc idx sibs = climb (MerkleScheme.sha512 msId) acc idx sibs
  | [], _, _ => rfl
  | s :: rest, idx, acc => by
      simp only [climb]
      exact climb_hm96 msId rest _ _

/-- **A passed Merkle check is the model's salted path check** (§14), for `hm96-sha512/v1`: every query's row has the lane
count, and its path verifies from the row's leaf under its salt against the cap. -/
theorem merkleCheck_verifiesL {cap : Array ByteArray} {c d : ℕ} {pos : Array ℕ}
    {rows : Array (Array F128)} {sibs salts : Array ByteArray} {lanes : ℕ}
    (h : merkleCheck MerkleScheme.hm96Sha512 cap c d pos rows sibs salts lanes = .ok ()) (j : ℕ) (hj : j < pos.size) :
    rows[j]!.size = lanes ∧
      Merkle.VerifiesL H512 enc512 leaf512 (fun i => toDig (cap.getD i ByteArray.empty)) (d - c) pos[j]!
        rows[j]!.toList (salts.getD j ByteArray.empty).data.toList ((sibsOf sibs (d - c) j).map toDig) := by
  unfold merkleCheck at h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hw
  simp only [Bool.or_eq_true, not_or, Array.any_eq_true, not_exists, bne_iff_ne, ne_eq, not_not] at hw
  split at h
  · cases h
  rename_i hf
  have hop := List.find?_eq_none.1 hf j (List.mem_range.2 hj)
  have h2 : opens MerkleScheme.hm96Sha512 cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty)
      (sibsOf sibs (d - c) j) = true := by
    cases h3 : opens MerkleScheme.hm96Sha512 cap c d lanes pos[j]! rows[j]! (salts.getD j ByteArray.empty)
      (sibsOf sibs (d - c) j)
    · rw [h3] at hop; exact absurd rfl hop
    · rfl
  simp only [opens, Bool.and_eq_true, decide_eq_true_eq] at h2
  obtain ⟨⟨⟨hsz, hlen⟩, -⟩, hclimb⟩ := h2
  have hsib : ∀ b ∈ sibsOf sibs (d - c) j, b.size = 64 := by
    intro b hb
    unfold sibsOf at hb
    rw [Array.mem_toList_iff] at hb
    obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hb
    rw [Array.getElem_extract]
    exact hw.2 _ (by simp only [Array.size_extract] at hi; omega)
  have hmap : ((sibsOf sibs (d - c) j).map toDig).map (·.1) = sibsOf sibs (d - c) j := by
    rw [List.map_map]
    exact (List.map_congr_left fun b hb => toDig_val (hsib b hb)).trans (List.map_id _)
  refine ⟨hsz, by rw [List.length_map, hlen], ?_⟩
  apply Subtype.ext
  unfold Merkle.reachL
  rw [walk_val "", hmap, hm96_leaf, ← climb_hm96 "", hclimb, ← Nat.shiftRight_eq_div_pow]
  have h64 : (cap.getD (pos[j]! >>> (d - c)) ByteArray.empty).size = 64 := by
    rw [← hclimb, climb_hm96 ""]
    exact climb_size "" _ _ _ (sha512_size _)
  exact (toDig_val h64).symm

/-! ## The proof's openings, with their salts, pass the model's salted path checks -/

/-- The proof's salt at level `ℓ`, query `j`, as bytes. -/
def saltRow (p : LigeritoProof) (sch : Flock.Schedule) (ℓ j : ℕ) : List UInt8 :=
  ((openAt p (sch.levels.size - 1) ℓ).2.2.getD j ByteArray.empty).data.toList

/-- **The proof's openings pass the model's salted path checks** (§14): for `hm96-sha512/v1`, a rep whose requests passed
the executable's Merkle checks (`ReqOK`) and whose caps are the proof's is `OpensOKL` at openings and salts that are the
proof's rows, siblings and salts. -/
theorem opensOKL_of (sch : Flock.Schedule) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (cap : Array ByteArray) (p : LigeritoProof) (r : ℕ) (pend : PendL 𝔽 Dig) (o : Opens 𝔽 Dig)
    (sl : Salts (List UInt8))
    (hcaps : pend.caps = (List.range (sch.levels.size - 1)).map (fun j => capDig p.recursiveCaps[j]!))
    (hreqs : ∀ req ∈ pend.reqs, ReqOK MerkleScheme.hm96Sha512 sch cap p req)
    (ho : ∀ ℓ j, o r ℓ j = openRow p sch ℓ j) (hs : ∀ ℓ j, sl r ℓ j = saltRow p sch ℓ j) :
    OpensOKL H512 enc512 leaf512 (capDig cap) r pend o sl := by
  intro req hreq
  obtain ⟨hlev, hlanes, hdepth, hmc⟩ := hreqs req hreq
  refine ⟨capDig (capAt cap p req.level), ?_, fun j hj => ?_⟩
  · unfold capOf capAt
    split_ifs with h0
    · rfl
    · rw [hcaps, List.getElem?_map, List.getElem?_range (by omega)]
      rfl
  · have hj' : j < req.positions.toArray.size := by simpa using hj
    obtain ⟨hsz, hver⟩ := merkleCheck_verifiesL hmc j hj'
    have hd := (hlv req.level hlev).2.2.1
    rw [ho, hs]
    have hpos : req.positions.getD j 0 = req.positions.toArray[j]! := by
      rw [List.getD_eq_getElem _ _ hj, getElem!_pos _ j hj', List.getElem_toArray]
    have hrow : (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf) =
        ((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList :=
      (List.map_congr_left fun _ _ => rfl).trans (List.map_id _)
    refine ⟨?_, ?_⟩
    · show (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf).length = _
      rw [List.length_map, Array.length_toList, hsz]
    · show Merkle.VerifiesL H512 enc512 leaf512 _ _ _
        (((openAt p (sch.levels.size - 1) req.level).1)[j]!.toList.map fOf) _ _
      rw [hrow, hdepth, hpos, ← hd]
      exact hver

/-- The proof files' salts (§14): rep `r`'s opened rows' salts, level by level. -/
def saltsOf (p₀ p₁ : Proof) (sch : Flock.Schedule) : Salts (List UInt8) :=
  fun r ℓ j => if r = 0 then saltRow p₀.ligerito sch ℓ j else saltRow p₁.ligerito sch ℓ j

/-! ## The verifier refines the model's salted table -/

/-- **The verifier refines the model's table, for `hm96-sha512/v1`** (§3, §5, §9–§14). `verify_refines` for the salted
scheme: when `Flock.verify` accepts a one-table session record and its two proof files under `hm96Sha512`, on a statement
whose `m`, `k_log`, circuit fold and extra claims are the model's `S` and `extra`, the record and both proofs decode.
Then at the proofs' openings, each rep's compiled game, played with its proof's messages and every coin of its stream,
accepts, and both reps' paths verify from their rows' salted leaves, under the proofs' salts, against the one level-0
cap. -/
theorem verify_refines_hm96 (spec : SessionSpec) (st : Setup) (S : Model.Statement) (hm : 13 ≤ S.m)
    (hmS : st.m = S.m) (hkS : st.kLog = S.kLog) (hfold : FoldRealizes st.fold S)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (extra : Session → List (Model.Claim 𝔽 S.m))
    (record : String) (proofs : Array ByteArray)
    (hext : ∀ sess, Record.decode spec record = .ok sess → ExtraRealizes st S sess.points (extra sess))
    (h : Flock.verify spec #[st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf st sch (extra sess).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 S (schedOf sch) hm (extra sess))
        (repMsgs st sch (2 + (extra sess).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) := by
  obtain ⟨sess, sch, c, hsess, hf, h0, h1⟩ := verify_one_ok h
  have h15 : 15 ≤ S.m := by have := exec_fast100_ge hf; omega
  have hwf := decode_roundsWF hsess
  obtain ⟨hdim, hsm⟩ := fast100_dims st.m sch hf
  have hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default) := fun i hi =>
    fast100_level st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hα1 : ∀ i < sch.levels.size, ceilLog2 (sch.levels.getD i default).q ≠ 1 := fun i hi =>
    fast100_alpha st.m sch hf _ (by unfold Array.getD; rw [dite_eq_left hi]; exact Array.getElem_mem hi)
  have hschm : sch.m = S.m := hsm.trans hmS
  obtain ⟨s₀, p₀, hs₀, hp₀, hc₀, -, hsz₀, hw₀, hrest₀⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold (extra sess)
    sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 0 _ _ h0
  obtain ⟨s₁, p₁, hs₁, hp₁, hc₁, -, hsz₁, hw₁, hrest₁⟩ := rep_refines spec st S hm h15 sch hmS hkS hfold (extra sess)
    sess (hext sess hsess) hdim hlv hα1 hschm hwf 0 1 _ _ h1
  obtain ⟨pend₀, hpl₀, hcaps₀, hreqs₀, hver₀⟩ := hrest₀ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  obtain ⟨pend₁, hpl₁, hcaps₁, hreqs₁, hver₁⟩ := hrest₁ capDig (opensOf p₀ p₁ sch) fun ℓ _ j _ => rfl
  rw [hms] at hreqs₀ hreqs₁ hw₀ hw₁
  have hok₀ := opensOKL_of sch hlv p₀.cap p₀.ligerito 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) hcaps₀ hreqs₀
    (fun ℓ j => rfl) (fun ℓ j => rfl)
  have hok₁ := opensOKL_of sch hlv p₁.cap p₁.ligerito 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) hcaps₁ hreqs₁
    (fun ℓ j => rfl) (fun ℓ j => rfl)
  have hcapeq : p₁.cap = p₀.cap := by
    apply Array.ext'
    refine entries_eq_of_flat _ _ (fun x hx => hw₁ x hx) (fun x hx => hw₀ x hx) (by simp [hsz₀, hsz₁]) ?_
    have e := congrArg (fun b : ByteArray => b.data.toList) (hc₁.symm.trans hc₀)
    simp only [← Array.foldl_toList, flat_data] at e
    simpa using e
  rw [hcapeq] at hok₁
  exact ⟨sess, sch, p₀, p₁, s₀, s₁, pend₀, pend₁, hsess, hf, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, hver₀, hver₁, hok₀,
    hok₁⟩

open Game Model in
/-- **The verifier refines the model's reps, for the statements it is set up with, under `hm96-sha512/v1`** (§3, §5, §10,
§14, §16). `verify_refines_ofCircuit` for the salted scheme: when `Flock.verify` accepts a one-table session under
`hm96Sha512`, with the setup `Setup.ofCircuit st` of a statement whose layout and regions are well formed, the conclusion
of `verify_refines_hm96` holds for the model statement `stmtOf st` read off the layout, with the extra claims the model's
`extraClaims` at the record's link points. -/
theorem verify_refines_ofCircuit_hm96 (st : Flock.Stmt) (hwf : StmtWF st) (hr : RegionsWF st) (hm : 13 ≤ st.m)
    (hms : st.merkle = MerkleScheme.hm96Sha512) (record : String) (proofs : Array ByteArray)
    (h : Flock.verify st.spec #[Setup.ofCircuit st] record proofs = .ok ()) :
    ∃ (sess : Session) (sch : Flock.Schedule) (p₀ p₁ : Proof) (s₀ s₁ : StreamRec) (pend₀ pend₁ : PendL 𝔽 Dig),
      Record.decode st.spec record = .ok sess ∧ Flock.fast100 st.m = some sch ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[0]! = .ok p₀ ∧
      decodeProof (shapeOf (Setup.ofCircuit st) sch (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
        (ptsOf st.mPts sess.points)).length) proofs[1]! = .ok p₁ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.tables[0]! ++ toString "/rep" ++ toString 0) = some s₀ ∧
      sess.streams.find? (fun x => x.name == toString st.spec.tables[0]! ++ toString "/rep" ++ toString 1) = some s₁ ∧
      Plays fOf (Model.repC execArith 0 (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
          (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
        (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
          (ptsOf st.mPts sess.points)).length) p₀ capDig) (s₀.rounds.toList.map (·.coins)) pend₀ ∧
      Plays fOf (Model.repC execArith 1 (stmtOf st hwf (regionsOf st)) (schedOf sch) hm
          (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points)))
        (repMsgs (Setup.ofCircuit st) sch (2 + (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts
          (ptsOf st.mPts sess.points)).length) p₁ capDig) (s₁.rounds.toList.map (·.coins)) pend₁ ∧
      pend₀.verdict (opensOf p₀ p₁ sch) = true ∧ pend₁.verdict (opensOf p₀ p₁ sch) = true ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) ∧
      OpensOKL H512 enc512 leaf512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) (saltsOf p₀ p₁ sch) :=
  verify_refines_hm96 st.spec (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) hm rfl rfl (ofCircuit_fold st hwf _)
    hms (fun sess => Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points))
    record proofs
    (fun sess hsess => ofCircuit_extra st hwf hr sess.points (decode_points hsess).1 (decode_points hsess).2) h

end FlockSoundness.Refine
