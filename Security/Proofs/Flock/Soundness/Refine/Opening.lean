import Proofs.Flock.Soundness.Refine.Lincheck

/-!
# Ring switching and batching refine the model's opening (PROTOCOL.md §12)

`opening_refines`: if the executable's `ringSwitch` succeeds on claims that are the model's (`ClaimRel`), the model's
`opening`, played with the proof's ring switches (`opMsgs`) and the coins of the rounds consumed, returns `some op`,
and `op` is the executable's batched value and batched basis (`OpenRel`).

The executable transposes each switched claim's bits (`tOf`: bit `v` of `t[u]` is bit `u` of `s[v]`) with `F128.bit` and
monomials; the model reads them through `unpack` and `pack` (`recombine`). `unpack_fOf` and `pack_eq_sum` are the seam.
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3 Game Transcript

local notation "𝔽" => FlockLevel3.GF128

/-- `Game` is a lawful monad: its `bind` is associative and `ret` is its unit. -/
instance instLawfulMonadGame : LawfulMonad Game := LawfulMonad.mk'
  (id_map := fun x => Game.bind_ret_right x)
  (pure_bind := fun _ _ => rfl)
  (bind_assoc := fun x f g => Game.bind_assoc x f g)

/-! ## Bits -/

theorem uint_bit (x : UInt64) (k : ℕ) (hk : k < 64) : ((x >>> k.toUInt64) &&& 1 == 1) = x.toNat.testBit k := by
  have h1 : ((x >>> k.toUInt64) &&& 1 = 1) ↔ x.toNat.testBit k = true := by
    rw [and_one_eq_one, UInt64.toNat_shiftRight, toNat_toUInt64 k hk, Nat.mod_eq_of_lt hk, Nat.testBit_shiftRight,
      Nat.add_zero]
  by_cases h : x.toNat.testBit k = true
  · rw [h]; exact beq_iff_eq.mpr (h1.mpr h)
  · rw [Bool.not_eq_true] at h
    rw [h, Bool.eq_false_iff]
    intro h'
    have := h1.mp (beq_iff_eq.mp h')
    rw [h] at this
    cases this

/-- Bit `u` of an `F128` is bit `u` of its 128-bit word. -/
theorem bit_testBit (a : F128) (u : ℕ) (hu : u < 128) : a.bit u = (w128 a.lo a.hi).testBit u := by
  rw [h_bit]
  unfold F128.bit
  split_ifs with h
  · exact uint_bit a.lo u h
  · exact uint_bit a.hi (u - 64) (by omega)

/-- **Bits are the packing basis's coordinates** (§12.1): `unpack` reads bit `u` of the word. -/
theorem unpack_fOf (a : F128) (u : Fin 128) : unpack (fOf a) u = if a.bit u.val then 1 else 0 := by
  show (F128.poly a).coeff u = _
  rw [F128.poly, coeff_natPoly, ← bit_testBit a u u.isLt]
  rfl

theorem sum_unpack (f : Fin 128 → 𝔽) (i : Fin 128) : unpack (∑ v, f v) i = ∑ v, unpack (f v) i := by
  have := map_sum FlockLevel3.unpackHom f Finset.univ
  exact congrFun this i |>.trans (Finset.sum_apply i Finset.univ fun v => FlockLevel3.unpackHom (f v))

/-- **`pack` is the sum of the monomials `x^v` of the set bits** (§12.1). -/
theorem pack_eq_sum (b : Fin 128 → ZMod 2) :
    pack b = ∑ v : Fin 128, if b v = 1 then fOf (F128.monomial v) else 0 := by
  have h : unpack (∑ v : Fin 128, if b v = 1 then fOf (F128.monomial v) else 0) = b := by
    funext i
    rw [sum_unpack]
    simp only [unpack_ite, unpack_monomial]
    rw [Finset.sum_eq_single i (fun v _ hv => by simp [Ne.symm hv]) (by simp)]
    simp only [ite_true]
    rcases (show ∀ z : ZMod 2, z = 0 ∨ z = 1 by decide) (b i) with h0 | h1
    · rw [h0]; simp
    · rw [h1]; simp
  have e := pack_unpack (∑ v : Fin 128, if b v = 1 then fOf (F128.monomial v) else 0)
  rw [h] at e
  exact e

/-- The executable's bit transpose (§12.3): bit `v` of `t[u]` is bit `u` of `s[v]`. -/
def tOf (s : Array F128) : Array F128 :=
  (Array.range 128).map fun u =>
    (Array.range 128).foldl (fun acc v => if s[v]!.bit u = true then acc + F128.monomial v else acc) 0

theorem tOf_size (s : Array F128) : (tOf s).size = 128 := by simp [tOf]

theorem fOf_foldl_bits (s : Array F128) (u : ℕ) : ∀ (k : ℕ) (acc : F128),
    fOf ((List.range k).foldl (fun acc v => if s[v]!.bit u = true then acc + F128.monomial v else acc) acc) =
      fOf acc + ∑ v ∈ Finset.range k, if s[v]!.bit u = true then fOf (F128.monomial v) else 0 := by
  intro k
  induction k with
  | zero => intro acc; simp
  | succ k ih =>
      intro acc
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ]
      split_ifs with hb
      · rw [fOf_add, ih, add_assoc]
      · rw [ih, add_zero]

/-- **The bit transpose is the model's recombination's inner term:** `t[u] = pack (bit u of s[·])`. -/
theorem fOf_tOf (s : Array F128) (u : Fin 128) :
    fOf (tOf s)[u.val]! = pack fun v => unpack (vecOf 128 s v) u := by
  rw [pack_eq_sum]
  simp only [tOf]
  rw [getElem!_range_map _ 128 u u.isLt, ← Array.foldl_toList, Array.toList_range, fOf_foldl_bits, fOf_zero,
    zero_add, ← Fin.sum_univ_eq_sum_range (fun v => if s[v]!.bit u = true then fOf (F128.monomial v) else 0)]
  refine Finset.sum_congr rfl fun v _ => ?_
  simp only [vecOf, unpack_fOf]
  split_ifs <;> simp_all

/-- The executable's equality tensor at `b` is the model's `eqAt` on the point's coordinates. -/
theorem eqTable_vecOf (r : Array F128) (n : ℕ) (hn : r.size = n) (b : ℕ) (hb : b < 2 ^ n) :
    fOf (eqTable r)[b]! = Model.eqAt (vecOf n r) b := by
  subst hn
  show FlockLevel3.GF128.of (eqTable r)[b]! = _
  rw [eqTable_eqAt r b hb]
  rfl

/-- **The recombined value** (§12.3): `β = Σ_u eq(r'', u)·t[u]` is the model's `recombine`. -/
theorem recombine_eq (s rpp : Array F128) (hrpp : rpp.size = 7) :
    fOf (dot (eqTable rpp) (tOf s)) = Model.recombine execArith (vecOf 128 s) (vecOf 7 rpp) := by
  rw [fOf_dot _ _ 128 (by rw [eqTable_size, hrpp]; rfl)]
  unfold Model.recombine
  refine Finset.sum_congr rfl fun u _ => ?_
  rw [eqTable_vecOf rpp 7 hrpp u.val (by simp), fOf_tOf]
  rfl

/-! ## The per-claim check (O4) -/

/-- The executable's weights of a claim on the 128 packed positions (§12.1). -/
def wOf (c : Flock.Claim) : Array F128 :=
  Array.map (fun v => c.skip[v % 64]! * if (v / 64 == 1) = true then c.x[0]! else (1 : F128) + c.x[0]!)
    (Array.range 128)

theorem packedWeight_eq {m : ℕ} (hm : 7 ≤ m) (c : Flock.Claim) (c' : Model.Claim 𝔽 m) (hc : ClaimRel m c c')
    (s : Array F128) : fOf (dot (wOf c) s) = ∑ v : Fin 128, Model.packedWeight c' hm v * vecOf 128 s v := by
  rw [fOf_dot _ _ 128 (by simp [wOf])]
  refine Finset.sum_congr rfl fun v _ => ?_
  simp only [wOf, Model.packedWeight]
  rw [getElem!_range_map _ 128 v v.isLt, fOf_mul, hc.skip, hc.x]
  congr 1
  simp only [beq_iff_eq]
  split_ifs <;> simp only [fOf_add, fOf_one]

/-! ## The loops -/

/-- `forIn_rounds`, and every step's check `Q i` passed. -/
theorem forIn_roundsQ {σ : Type} (body : ℕ → σ → TM (ForInStep σ)) (step : ℕ → σ → Array F128 → σ) (N : ℕ)
    (Q : ℕ → Prop)
    (hbody : ∀ i s t b t', body i s t = .ok (b, t') →
      (t.rounds[t.k]!).n = N ∧ t'.rounds = t.rounds ∧ t'.k = t.k + 1 ∧
        b = .yield (step i s (t.rounds[t.k]!).coins) ∧ Q i) :
    ∀ (n a base : ℕ) (s : σ) (t t' : Transcript) (out : σ), t.k = base + a →
      forIn (List.range' a n) s body t = .ok (out, t') →
      t'.rounds = t.rounds ∧ t'.k = t.k + n ∧ (∀ i ∈ List.range' a n, (t.rounds[base + i]!).n = N ∧ Q i) ∧
      out = (List.range' a n).foldl (fun s i => step i s (t.rounds[base + i]!).coins) s := by
  intro n
  induction n with
  | zero =>
      intro a base s t t' out _ h
      simp only [List.range'_zero, List.forIn_nil] at h
      obtain ⟨rfl, rfl⟩ := pure_ok h
      simp
  | succ n ih =>
      intro a base s t t' out hk h
      rw [forIn_range'_cons] at h
      obtain ⟨b, t₁, h₁, h₂⟩ := bind_ok h
      obtain ⟨hN, hr₁, hk₁, rfl, hQ⟩ := hbody a s t b t₁ h₁
      obtain ⟨hr, hk', hNs, hout⟩ := ih (a + 1) base _ t₁ t' out (by omega) h₂
      refine ⟨hr.trans hr₁, by omega, ?_, ?_⟩
      · intro i hi
        rw [List.range'_succ, List.mem_cons] at hi
        rcases hi with rfl | hi
        · rw [← hk]; exact ⟨hN, hQ⟩
        · rw [← hr₁]; exact hNs i hi
      · rw [hout, hr₁, List.range'_succ, List.foldl_cons, hk]

/-- One ring switch (O3–O5) as the executable records it: the recombined value, and the coins with the suffix point. -/
def oStep (claims : Array Flock.Claim) (rs : Array (Array F128)) (k : ℕ)
    (s : Array F128 × Array (Array F128 × Array F128)) (c : Array F128) :
    Array F128 × Array (Array F128 × Array F128) :=
  (s.1.push (dot (eqTable c) (tOf rs[k]!)), s.2.push (c, claims[k]!.x.extract 1))

theorem oStep_fold (claims : Array Flock.Claim) (rs : Array (Array F128)) (C : ℕ → Array F128) :
    ∀ (l : List ℕ) (s : Array F128 × Array (Array F128 × Array F128)),
      (l.foldl (fun s i => oStep claims rs i s (C i)) s).1 =
          s.1 ++ (l.map fun k => dot (eqTable (C k)) (tOf rs[k]!)).toArray ∧
        (l.foldl (fun s i => oStep claims rs i s (C i)) s).2 =
          s.2 ++ (l.map fun k => (C k, claims[k]!.x.extract 1)).toArray := by
  intro l
  induction l with
  | nil => intro s; simp
  | cons i l ih =>
      intro s
      obtain ⟨h1, h2⟩ := ih (oStep claims rs i s (C i))
      rw [List.foldl_cons, h1, h2]
      simp [oStep]

/-- **The model's ring switches** (O3, O5): claim by claim, receive `ŝ` and draw `r''`. -/
theorem mapM_plays {m : ℕ} (rs : Array (Array F128)) (C : ℕ → Array F128) :
    ∀ (cs' : List (Model.Claim 𝔽 m)) (a : ℕ), (∀ i ∈ List.range' a cs'.length, (C i).size = 7) →
      Plays fOf (cs'.mapM fun c => do
          let s ← recv (Fin 128 → 𝔽)
          let r'' ← draw (Fin 7 → 𝔽)
          pure (c, s, r''))
        ((List.range' a cs'.length).map fun k => ⟨Fin 128 → 𝔽, vecOf 128 rs[k]!⟩)
        ((List.range' a cs'.length).map C)
        ((cs'.zipIdx a).map fun ck => (ck.1, vecOf 128 rs[ck.2]!, vecOf 7 (C ck.2)))
  | [], a, _ => by simpa using Plays.pure (ι := fOf) ([] : List (Model.Claim 𝔽 m × (Fin 128 → 𝔽) × (Fin 7 → 𝔽)))
  | c :: cs', a, hC => by
      rw [List.mapM_cons]
      have hCa : (C a).size = 7 := hC a (by simp [List.range'_succ])
      have ih := mapM_plays rs C cs' (a + 1) (fun i hi => hC i (by
        rw [List.length_cons, List.range'_succ]; exact List.mem_cons_of_mem _ hi))
      have e : ((c :: cs').zipIdx a).map (fun ck => (ck.1, vecOf 128 rs[ck.2]!, vecOf 7 (C ck.2))) =
          (c, vecOf 128 rs[a]!, vecOf 7 (C a)) ::
            (cs'.zipIdx (a + 1)).map (fun ck => (ck.1, vecOf 128 rs[ck.2]!, vecOf 7 (C ck.2))) := by
        simp [List.zipIdx_cons]
      rw [e]
      have h1 : Plays fOf (do
            let s ← recv (Fin 128 → 𝔽)
            let r'' ← draw (Fin 7 → 𝔽)
            pure (c, s, r''))
          [⟨Fin 128 → 𝔽, vecOf 128 rs[a]!⟩] [C a] (c, vecOf 128 rs[a]!, vecOf 7 (C a)) :=
        Plays.bind' (ms₂ := []) (cs₂ := [C a]) (Plays.recv _ (vecOf 128 rs[a]!))
          (Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec (C a) hCa (by omega)) (Plays.pure _) (by simp) (by simp))
          (by simp) (by simp)
      refine Plays.bind' (ms₂ := (List.range' (a + 1) cs'.length).map fun k => ⟨Fin 128 → 𝔽, vecOf 128 rs[k]!⟩)
        (cs₂ := (List.range' (a + 1) cs'.length).map C) h1 ?_ ?_ ?_
      · exact Plays.bind' (ms₂ := []) (cs₂ := []) ih (Plays.pure _) (by simp) (by simp)
      · simp [List.range'_succ]
      · simp [List.range'_succ]

/-- The check of one ring switch (O4): `Σ_v w[v]·ŝ[v] = value`. -/
def RsCheck (claims : Array Flock.Claim) (rs : Array (Array F128)) (k : ℕ) : Prop :=
  (dot (wOf claims[k]!) rs[k]! == claims[k]!.value) = true

theorem getD_zipIdx_map {α β : Type} (l : List α) (f : α × ℕ → β) (k : ℕ) (hk : k < l.length) (d : β) :
    ((l.zipIdx 0).map f).getD k d = f (l[k], k) := by
  rw [List.getD_eq_getElem _ _ (by simpa using hk), List.getElem_map, List.getElem_zipIdx, Nat.zero_add]

/-- **Ring switching and batching refine the model's opening** (PROTOCOL.md §12). If the executable's `ringSwitch`
succeeds from `t` to `t'` on claims that are the model's `cs'`, the model's opening of `cs'`, played with the proof's
ring switches and the coins of the rounds consumed, returns the executable's batched value and basis. -/
theorem opening_refines (m : ℕ) (hm : 7 ≤ m) (claims : Array Flock.Claim) (cs' : List (Model.Claim 𝔽 m))
    (hlen : claims.size = cs'.length) (hcs : ∀ (k : ℕ) (hk : k < cs'.length), ClaimRel m claims[k]! cs'[k])
    (hn : 0 < claims.size) (hn1 : claims.size ≠ 1) (rs : Array (Array F128)) (t t' : Transcript) (T : F128)
    (sws : Array Switched) (hwf : RoundsWF t.rounds) (h : ringSwitch m claims rs t = .ok ((T, sws), t')) :
    t'.rounds = t.rounds ∧ t.k ≤ t'.k ∧
      ∃ op, Plays fOf (Model.opening execArith hm cs') (opMsgs claims.size rs) (consumed t.rounds t.k t'.k)
          (some op) ∧ OpenRel m claims.size T sws op := by
  unfold ringSwitch at h
  obtain ⟨t₁, k₁, h⟩ := label_bind h
  obtain ⟨s, t₂, hloop, h⟩ := bind_ok h
  obtain ⟨lt₂, n₂, t₃, r₃, kk₃, h⟩ := sampleVec_bind (by omega) h
  obtain ⟨hTs, rfl⟩ := pure_ok h
  rw [forIn_range_eq] at hloop
  obtain ⟨hr₂, hk₂, hQ, hs⟩ := forIn_roundsQ _ (oStep claims rs) 7 (RsCheck claims rs) (by
      intro k st tt b tt' hb
      obtain ⟨tt₁, kk₁, hb⟩ := label_bind hb
      obtain ⟨tt₂, kk₂, hb⟩ := absorbSlice_bind hb
      obtain ⟨hval, hb⟩ := check_bind hb
      obtain ⟨-, nn, tt₃, rr₃, kk₃, hb⟩ := sampleVec_bind (by omega) hb
      obtain ⟨rfl, rfl⟩ := pure_ok hb
      have e1 : tt₂.rounds = tt.rounds := by rw [kk₂.2, kk₁.2]
      have e2 : tt₂.k = tt.k := by rw [kk₂.1, kk₁.1]
      refine ⟨by rw [← e1, ← e2]; exact nn, by rw [rr₃, e1], by rw [kk₃, e2], ?_, hval⟩
      rw [← e1, ← e2]
      rfl) claims.size 0 t₁.k _ t₁ t₂ s (by omega) hloop
  obtain ⟨hT, hsws⟩ := Prod.mk.inj hTs
  subst hT hsws
  set C : ℕ → Array F128 := fun i => (t₁.rounds[t₁.k + i]!).coins with hC
  set cγ := (t₂.rounds[t₂.k]!).coins with hcγ
  have hwf₁ : RoundsWF t₁.rounds := k₁.2 ▸ hwf
  have hwf₂ : RoundsWF t₂.rounds := (hr₂.trans k₁.2) ▸ hwf
  have hC7 : ∀ i ∈ List.range' 0 cs'.length, (C i).size = 7 := fun i hi => by
    rw [← hlen] at hi
    rw [hC]; simp only; rw [RoundsWF.get hwf₁, (hQ i hi).1]
  have hγ : cγ.size = cs'.length := by rw [hcγ, RoundsWF.get hwf₂, n₂, hlen]
  have hcoins : consumed t.rounds t.k t'.k = (List.range' 0 cs'.length).map C ++ [cγ] := by
    rw [hcγ, hC, kk₃, hk₂, hr₂, k₁.1, k₁.2,
      consumed_append t.rounds (show t.k ≤ t.k + claims.size by omega) (by omega), consumed_succ, consumed_eq_map,
      hlen]
  obtain ⟨hs1, hs2⟩ := oStep_fold claims rs C (List.range' 0 claims.size) (#[], #[])
  rw [← hs] at hs1 hs2
  have hs1k : ∀ k < cs'.length, s.1[k]! = dot (eqTable (C k)) (tOf rs[k]!) := fun k hk => by
    rw [hs1, getElem!_pos _ k (by simp [hlen, hk])]; simp
  have hs2k : ∀ k < cs'.length, s.2[k]! = (C k, claims[k]!.x.extract 1) := fun k hk => by
    rw [hs2, getElem!_pos _ k (by simp [hlen, hk])]; simp
  refine ⟨by rw [r₃, hr₂, k₁.2], by have := k₁.1; omega, ?op, ?plays, ?rel⟩
  case plays =>
    rw [hcoins]
    unfold Model.opening
    refine Plays.bind' (ms₂ := []) (cs₂ := [cγ]) (mapM_plays rs C cs' 0 hC7) ?_
      (by simp [opMsgs, hlen, List.range_eq_range']) (by simp)
    refine Plays.bind' (ms₂ := []) (cs₂ := []) (Plays.drawVec cγ hγ (by omega)) ?_ (by simp) (by simp)
    dsimp only
    split
    · exact Plays.pure (ι := fOf) _
    · rename_i hne
      exfalso
      apply hne
      rw [List.all_eq_true]
      intro x hx
      obtain ⟨ck, hck, rfl⟩ := List.mem_map.1 hx
      obtain ⟨k, hk, rfl⟩ := List.mem_iff_getElem.1 hck
      rw [List.getElem_zipIdx, Nat.zero_add]
      rw [List.length_zipIdx] at hk
      simp only [decide_eq_true_eq]
      have hval := (hQ k (by simp; omega)).2
      rw [← packedWeight_eq hm claims[k]! cs'[k] (hcs k hk) rs[k]!, eq_of_beq hval, (hcs k hk).value]
  case rel =>
    have hL : ∀ (k : ℕ) (hk : k < cs'.length) d,
        ((cs'.zipIdx).map (fun ck => (ck.1, vecOf 128 rs[ck.2]!, vecOf 7 (C ck.2)))).getD k d =
          (cs'[k], vecOf 128 rs[k]!, vecOf 7 (C k)) := fun k hk d => getD_zipIdx_map cs' _ k hk d
    have hsw : ∀ k < claims.size,
        (Array.map (fun k => ({ gamma := cγ[k]!, rpp := s.2[k]!.1, z := s.2[k]!.2 } : Switched))
          (Array.range claims.size))[k]! = { gamma := cγ[k]!, rpp := C k, z := claims[k]!.x.extract 1 } := by
      intro k hk
      rw [getElem!_pos _ k (by simp [hk]), Array.getElem_map, Array.getElem_range, hs2k k (by omega)]
    refine ⟨?_, by simp, fun k hk => ?_, fun k hk => ?_, fun y => ?_⟩
    · show ∑ k : Fin cs'.length, _ = fOf (dot cγ s.1)
      rw [fOf_dot cγ s.1 cs'.length hγ]
      refine Finset.sum_congr rfl fun k _ => ?_
      rw [hL k.val k.isLt, hs1k k.val k.isLt, recombine_eq _ _ (hC7 k.val (by simp))]
    · rw [hsw k hk]; exact hC7 k (by simp; omega)
    · rw [hsw k hk]
      have := (hcs k (by omega)).x_size
      simp; omega
    · show ∑ k : Fin cs'.length, _ = ∑ k : Fin claims.size, _
      refine Fintype.sum_equiv (finCongr hlen.symm) _ _ fun k => ?_
      simp only [finCongr_apply, Fin.val_cast]
      rw [hsw k.val (by omega), hL k.val k.isLt]
      simp only
      have hx : (fun j : Fin (m - 7) => cs'[k.val].x ⟨j.val + 1, by omega⟩) =
          vecOf (m - 7) (claims[k.val]!.x.extract 1) := by
        funext j
        rw [(hcs k.val k.isLt).x]
        simp only [vecOf]
        rw [extract_get! _ 1 _ j (by have := (hcs k.val k.isLt).x_size; omega) le_rfl, Nat.add_comm]
      rw [hx]

end FlockSoundness.Refine
