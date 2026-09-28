import Proofs.Flock.Soundness.Refine.StmtOf
import Proofs.Flock.Soundness.Refine.Table

/-!
# The statement decoding, part 2: the regions are the model's (PROTOCOL.md §11, §16.2)

`regionOf` reads a region of the executable's statement as the model's: its free bits, the fixed bits the executable's
`regionPoint` sets, and bit `k` of block `b`'s bytes. `ofCircuit_extra`: at two link points of `m_pts` coordinates, the
extra claims `Setup.ofCircuit st` hands the verifier are the model's `extraClaims` of `stmtOf st` with those regions
(`ExtraRealizes`). It needs `RegionsWF`: `m = k_log + nbl`, `m_pts = PT_LOCAL + nbl`, and each region's free bits are
distinct in-block bits, at most `PT_LOCAL` of them.

`verify_refines_ofCircuit` composes this and `ofCircuit_fold` with `verify_refines`: for a verifier set up with
`Setup.ofCircuit st`, its conclusion holds for `stmtOf st`, with nothing assumed of the fold or the extra claims. The
record's link points have the configured shape because `Record.decode` checks it (S9, `decode_points`).
-/

namespace FlockSoundness.Refine

open Flock FlockLevel3

local notation "𝔽" => FlockLevel3.GF128

theorem forIn_id_foldl_pure {σ : Type} (k : ℕ) (s : σ) (g : ℕ → σ → σ) :
    (forIn [:k] s fun i r => pure (ForInStep.yield (g i r)) : Id σ) = (List.range k).foldl (fun r i => g i r) s := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range', List.forIn_pure_yield_eq_foldl]
  simp [Std.Legacy.Range.size, List.range_eq_range']
  rfl

theorem forIn_id_foldl {σ : Type} (k : ℕ) (s : σ) (g : ℕ → σ → σ) :
    (forIn (m := Id) [:k] s fun i r => ForInStep.yield (g i r)) = (List.range k).foldl (fun r i => g i r) s :=
  forIn_id_foldl_pure k s g

theorem regionPoint_eq (st : Stmt) (reg : Region) (r : Array F128) :
    st.regionPoint reg r =
      (List.range st.nbl).foldl (fun p i => p.set! (st.c.kLog + i) r[PT_LOCAL + i]!)
        ((List.range reg.free.size).foldl (fun p t => p.set! reg.free[t]! r[t]!)
          ((Array.range st.m).map fun j =>
            if j < st.c.kLog && (reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1 then 1 else 0)) := by
  unfold Stmt.regionPoint
  simp only [Id.run, bind, pure]
  rw [forIn_id_foldl, forIn_id_foldl]

/-! ### `regionValue` -/

/-- A loop whose steps either throw or yield a pure update: when it returns, no step threw. -/
theorem list_forIn_guard {α σ : Type} (c : α → Bool) (e : String) (g : α → σ → σ) :
    ∀ (L : List α) (s v : σ), (forIn L s fun a r =>
        if c a = true then (throw e : V (ForInStep σ)) else pure (ForInStep.yield (g a r))) = .ok v →
      (∀ a ∈ L, c a = false) ∧ v = L.foldl (fun r a => g a r) s
  | [], s, v, h => by simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h; exact ⟨by simp, h.symm⟩
  | a :: L, s, v, h => by
      rw [List.forIn_cons] at h
      cases hc : c a
      · simp only [hc, Bool.false_eq_true, ite_false, pure, Except.pure, bind, Except.bind] at h
        obtain ⟨h1, h2⟩ := list_forIn_guard c e g L _ v h
        exact ⟨fun b hb => (List.mem_cons.1 hb).elim (fun h => h ▸ hc) (h1 b), by rw [List.foldl_cons]; exact h2⟩
      · simp only [hc, ite_true] at h; cases h

theorem forIn_guard {σ : Type} (k : ℕ) (c : ℕ → Bool) (e : String) (g : ℕ → σ → σ) (s v : σ)
    (h : (forIn [:k] s fun a r =>
        if c a = true then (throw e : V (ForInStep σ)) else pure (ForInStep.yield (g a r))) = .ok v) :
    (∀ a < k, c a = false) ∧ v = (List.range k).foldl (fun r a => g a r) s := by
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h
  have := list_forIn_guard c e g _ s v h
  simp only [Std.Legacy.Range.size, Nat.add_sub_cancel, Nat.div_one, ← List.range_eq_range'] at this
  exact ⟨fun a ha => this.1 a (List.mem_range.2 ha), this.2⟩

/-- The bits of a block's bytes, weighted: `Σ_{i, t : bit t of byte i} wt[8i + t]`. -/
def bitSum (bytes : ByteArray) (wt : Array F128) : F128 :=
  (List.range bytes.size).foldl (fun s i => (List.range 8).foldl (fun s t =>
    if ((bytes.get! i >>> t.toUInt8) % 2 == 1) = true then s + wt[8 * i + t]! else s) s) 0

theorem bits_loop (bytes : ByteArray) (wt : Array F128) :
    (forIn [:bytes.size] (0 : F128) fun i s => do
        let s ← forIn [:8] s fun t s =>
          if ((bytes.get! i >>> t.toUInt8) % 2 == 1) = true then pure (ForInStep.yield (s + wt[8 * i + t]!))
          else pure (ForInStep.yield s)
        pure (ForInStep.yield s) : V F128) = .ok (bitSum bytes wt) := by
  have hin : ∀ i s, (forIn [:8] s fun t s =>
      if ((bytes.get! i >>> t.toUInt8) % 2 == 1) = true then pure (ForInStep.yield (s + wt[8 * i + t]!))
      else pure (ForInStep.yield s) : V F128) = .ok ((List.range 8).foldl (fun s t =>
        if ((bytes.get! i >>> t.toUInt8) % 2 == 1) = true then s + wt[8 * i + t]! else s) s) := by
    intro i s
    rw [← forIn_range_foldl]
    congr 1; funext t s; split_ifs <;> rfl
  simp only [hin]
  exact forIn_range_foldl _ _ _

theorem regionValue_ok (st : Stmt) (reg : Region) (r : Array F128) (v : F128) (h : st.regionValue reg r = .ok v) :
    (∀ b < 2 ^ st.nbl, (reg.bytes b).size * 8 = 2 ^ reg.free.size) ∧
      v = (List.range (2 ^ st.nbl)).foldl (fun acc b => acc +
        (eqTable (r.extract PT_LOCAL (PT_LOCAL + st.nbl)))[b]! *
          bitSum (reg.bytes b) (eqTable (r.extract 0 reg.free.size))) 0 := by
  unfold Stmt.regionValue at h
  dsimp only at h
  obtain ⟨v', hv', h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  simp only [bits_loop] at hv'
  obtain ⟨hc, hv⟩ := forIn_guard (2 ^ st.nbl) (fun b => (reg.bytes b).size * 8 != 2 ^ reg.free.size)
    (toString "region " ++ toString reg.name ++ toString ": its bytes are not its free bits")
    (fun b acc => acc + (eqTable (r.extract PT_LOCAL (PT_LOCAL + st.nbl)))[b]! *
      bitSum (reg.bytes b) (eqTable (r.extract 0 reg.free.size))) 0 v' hv'
  exact ⟨fun b hb => by simpa using hc b hb, h ▸ hv⟩

/-! ### The regions, as the model's -/

/-- A region of the executable's statement, as the model's (§16.2): its free bits, the fixed bits the executable's
`regionPoint` sets, and bit `k` of block `b`'s bytes. -/
def regionOf (kl nbl : ℕ) (reg : Region) : Model.Region kl nbl where
  free := reg.free.toList.filterMap fun b => if h : b < kl then some ⟨b, h⟩ else none
  fixedBits j := (reg.mask >>> j.val) % 2 == 1 && (reg.fixed >>> j.val) % 2 == 1
  bits b k := if (((reg.bytes b.val).get! (k / 8) >>> (k % 8).toUInt8) % 2 == 1) = true then 1 else 0

/-- The regions of an executable statement, as the model's. -/
def regionsOf (st : Stmt) : List (Model.Region st.c.kLog (st.m - st.c.kLog)) :=
  st.regions.toList.map (regionOf st.c.kLog (st.m - st.c.kLog))

/-- A region's free bits are distinct in-block bits, at most `PT_LOCAL` of them. -/
structure RegionWF (kl : ℕ) (reg : Region) : Prop where
  free_lt : ∀ b ∈ reg.free.toList, b < kl
  nodup : reg.free.toList.Nodup
  free_size : reg.free.size ≤ PT_LOCAL

theorem filterMap_fin_length (kl : ℕ) : ∀ (L : List ℕ), (∀ b ∈ L, b < kl) →
    (L.filterMap fun b => if h : b < kl then some (⟨b, h⟩ : Fin kl) else none).length = L.length
  | [], _ => rfl
  | b :: L, h => by
      rw [List.filterMap_cons, dite_eq_left (h b List.mem_cons_self)]
      simp only [List.length_cons]
      rw [filterMap_fin_length kl L fun x hx => h x (List.mem_cons_of_mem _ hx)]

theorem of_foldl_range_add (F : ℕ → F128) : ∀ (N : ℕ) (s : F128),
    GF128.of ((List.range N).foldl (fun acc b => acc + F b) s) = GF128.of s + ∑ b ∈ Finset.range N, GF128.of (F b)
  | 0, s => by simp
  | N + 1, s => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, of_add, of_foldl_range_add F N s,
        Finset.sum_range_succ, add_assoc]

theorem of_foldl_range_gen (G : ℕ → F128 → F128) (D : ℕ → 𝔽) (hG : ∀ i s, GF128.of (G i s) = GF128.of s + D i) :
    ∀ (N : ℕ) (s : F128), GF128.of ((List.range N).foldl (fun s i => G i s) s) = GF128.of s + ∑ i ∈ Finset.range N, D i
  | 0, s => by simp
  | N + 1, s => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, hG, of_foldl_range_gen G D hG N s,
        Finset.sum_range_succ, add_assoc]

theorem sum_bytes {M : Type} [AddCommMonoid M] (f : ℕ → M) : ∀ n : ℕ,
    ∑ i ∈ Finset.range n, ∑ t ∈ Finset.range 8, f (8 * i + t) = ∑ k ∈ Finset.range (8 * n), f k
  | 0 => by simp
  | n + 1 => by
      rw [Finset.sum_range_succ, sum_bytes f n, show 8 * (n + 1) = 8 * n + 8 by ring, Finset.sum_range_add]

theorem of_bitSum (bytes : ByteArray) (wt : Array F128) :
    GF128.of (bitSum bytes wt) = ∑ k ∈ Finset.range (8 * bytes.size),
      (if (((bytes.get! (k / 8) >>> (k % 8).toUInt8) % 2 == 1) = true) then GF128.of wt[k]! else 0) := by
  unfold bitSum
  rw [of_foldl_range_gen _ (fun i => ∑ t ∈ Finset.range 8,
      (if (((bytes.get! i >>> t.toUInt8) % 2 == 1) = true) then GF128.of wt[8 * i + t]! else 0)) ?_,
    of_zero', zero_add, ← sum_bytes]
  · refine Finset.sum_congr rfl fun i _ => Finset.sum_congr rfl fun t ht => ?_
    have ht8 : t < 8 := Finset.mem_range.1 ht
    rw [show (8 * i + t) / 8 = i by omega, show (8 * i + t) % 8 = t by omega]
  · intro i s
    have key : ∀ (N : ℕ) (s : F128), GF128.of ((List.range N).foldl (fun s t =>
        if ((bytes.get! i >>> t.toUInt8) % 2 == 1) = true then s + wt[8 * i + t]! else s) s) =
        GF128.of s + ∑ t ∈ Finset.range N,
          (if (((bytes.get! i >>> t.toUInt8) % 2 == 1) = true) then GF128.of wt[8 * i + t]! else 0) := by
      intro N
      induction N with
      | zero => intro s; simp
      | succ N ih =>
          intro s
          rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil, Finset.sum_range_succ]
          split_ifs with hb
          · rw [of_add, ih, add_assoc]
          · rw [ih, add_zero]
    exact key 8 s

theorem eqTable_extract (r : Array F128) (a n : ℕ) (hr : a + n ≤ r.size) (b : ℕ) (hb : b < 2 ^ n) :
    GF128.of (eqTable (r.extract a (a + n)))[b]! =
      ∏ i ∈ Finset.range n, (if b.testBit i then GF128.of r[a + i]! else 1 + GF128.of r[a + i]!) := by
  have hsz : (r.extract a (a + n)).size = n := by simp [Array.size_extract]; omega
  rw [eqTable_eqAt _ b (by rw [hsz]; exact hb)]
  unfold FlockLevel3.eqAt
  rw [Finset.prod_range]
  refine Fintype.prod_equiv (finCongr hsz) _ _ fun i => ?_
  have hi : i.val < n := by have := i.isLt; omega
  simp only [finCongr_apply, Fin.val_cast]
  rw [getElem!_pos _ i.val i.isLt, Array.getElem_extract, getElem!_pos r (a + i.val) (by omega)]

theorem regionValue_model (st : Stmt) (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (reg : Region) (hw : RegionWF st.c.kLog reg) (hm : st.m = st.c.kLog + st.nbl) (r : Array F128)
    (hr : r.size = PT_LOCAL + st.nbl) (v : F128) (hv : st.regionValue reg r = .ok v) :
    Model.regionValue (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL (PT_LOCAL + st.nbl)
      (fun t => fOf r[t.val]!) = fOf v := by
  obtain ⟨hsz, hv⟩ := regionValue_ok st reg r v hv
  simp only [fOf]
  rw [hv, of_foldl_range_add, of_zero', zero_add]
  unfold Model.regionValue
  rw [Finset.sum_fin_eq_sum_range]
  have hN : (stmtOf st hwf regs).nbl = st.nbl := by show st.m - st.c.kLog = st.nbl; omega
  have hlen0 : (regionOf st.c.kLog (st.m - st.c.kLog) reg).free.length = reg.free.size := by
    show (reg.free.toList.filterMap _).length = _
    rw [filterMap_fin_length _ _ hw.free_lt, Array.length_toList]
  have hlen : (@Model.Region.free (stmtOf st hwf regs).kLog (stmtOf st hwf regs).nbl
      (regionOf st.c.kLog (st.m - st.c.kLog) reg)).length = reg.free.size := hlen0
  have hnf : reg.free.size ≤ PT_LOCAL := hw.free_size
  refine Finset.sum_congr (by rw [hN]) fun b hb => ?_
  have hb' : b < 2 ^ st.nbl := Finset.mem_range.1 hb
  rw [dite_eq_left (by rw [hN]; exact hb'), of_mul]
  refine congrArg₂ (· * ·) ?_ ?_
  · rw [eqTable_extract r PT_LOCAL st.nbl (by omega) b hb', Finset.prod_fin_eq_prod_range]
    refine Finset.prod_congr (by rw [hN]) fun i hi => ?_
    have hi' : i < st.nbl := Finset.mem_range.1 hi
    rw [dite_eq_left (by rw [hN]; exact hi')]
    simp only [dite_eq_left (show PT_LOCAL + i < PT_LOCAL + st.nbl by omega)]
  · rw [of_bitSum, show 8 * (reg.bytes b).size = 2 ^ reg.free.size by have := hsz b hb'; omega]
    refine Finset.sum_congr (by rw [hlen]) fun k hk => ?_
    have hk' : k < 2 ^ reg.free.size := Finset.mem_range.1 hk
    have hW := eqTable_extract r 0 reg.free.size (by omega) k hk'
    simp only [zero_add] at hW
    rw [hW, Finset.prod_fin_eq_prod_range, Finset.prod_congr (by rw [hlen])
      (g := fun t => if k.testBit t then GF128.of r[t]! else 1 + GF128.of r[t]!) fun t ht => by
        have ht' : t < reg.free.size := Finset.mem_range.1 ht
        rw [dite_eq_left (by rw [hlen]; exact ht')]
        simp only [dite_eq_left (show t < PT_LOCAL + st.nbl by omega)]]
    simp only [regionOf]
    split_ifs <;> simp

/-! ### `regionPoint` -/

theorem foldl_set!_size (f : ℕ → ℕ) (val : ℕ → F128) : ∀ (L : List ℕ) (p : Array F128),
    (L.foldl (fun p t => p.set! (f t) (val t)) p).size = p.size
  | [], _ => rfl
  | a :: L, p => by rw [List.foldl_cons, foldl_set!_size f val L]; simp [Array.set!]

/-- A write at a distinct index is the last one there. -/
theorem foldl_set_hit (f : ℕ → ℕ) (val : ℕ → F128) : ∀ (n : ℕ), (∀ a < n, ∀ b < n, f a = f b → a = b) →
    ∀ (p : Array F128) (t : ℕ), t < n → f t < p.size →
      ((List.range n).foldl (fun p t => p.set! (f t) (val t)) p)[f t]! = val t
  | 0, _, _, t, ht, _ => absurd ht (Nat.not_lt_zero _)
  | n + 1, hinj, p, t, ht, hf => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        set!_get _ _ _ _ (by rw [foldl_set!_size]; exact hf)]
      by_cases htn : t = n
      · subst htn; simp
      · rw [ite_eq_right (fun h => htn (hinj n (by omega) t ht h).symm)]
        exact foldl_set_hit f val n (fun a ha b hb => hinj a (by omega) b (by omega)) p t (by omega) hf

/-- An index no write reaches keeps its entry. -/
theorem foldl_set_miss (f : ℕ → ℕ) (val : ℕ → F128) : ∀ (n : ℕ) (p : Array F128) (j : ℕ), j < p.size →
    (∀ t < n, f t ≠ j) → ((List.range n).foldl (fun p t => p.set! (f t) (val t)) p)[j]! = p[j]!
  | 0, _, _, _, _ => rfl
  | n + 1, p, j, hj, hne => by
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil,
        set!_get _ _ _ _ (by rw [foldl_set!_size]; exact hj), ite_eq_right (hne n (by omega))]
      exact foldl_set_miss f val n p j hj (fun t ht => hne t (by omega))

theorem idxOf?_filterMap_fin (kl j : ℕ) (hj : j < kl) : ∀ (L : List ℕ), (∀ b ∈ L, b < kl) →
    (L.filterMap fun b => if h : b < kl then some (⟨b, h⟩ : Fin kl) else none).idxOf? ⟨j, hj⟩ = L.idxOf? j
  | [], _ => rfl
  | b :: L, h => by
      rw [List.filterMap_cons, dite_eq_left (h b List.mem_cons_self)]
      simp only [List.idxOf?_cons]
      rw [idxOf?_filterMap_fin kl j hj L fun x hx => h x (List.mem_cons_of_mem _ hx)]
      simp only [beq_iff_eq, Fin.mk.injEq]

/-- **The region's point, entry by entry** (§16.2): a free bit's link-point coordinate, a fixed bit, or a block bit. -/
theorem regionPoint_get (st : Stmt) (reg : Region) (r : Array F128) (hm : st.m = st.c.kLog + st.nbl)
    (hw : RegionWF st.c.kLog reg) (j : ℕ) (hj : j < st.m) :
    (st.regionPoint reg r)[j]! =
      if j < st.c.kLog then
        (match reg.free.toList.idxOf? j with
         | some t => r[t]!
         | none => if ((reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1) = true then 1 else 0)
      else r[PT_LOCAL + (j - st.c.kLog)]! := by
  rw [regionPoint_eq]
  have hsz0 : ((Array.range st.m).map fun j =>
      if (j < st.c.kLog && (reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1) = true then (1 : F128) else 0).size
      = st.m := by simp
  have hfree : ∀ t < reg.free.size, reg.free[t]! < st.c.kLog := fun t ht => by
    rw [getElem!_pos _ t ht]; exact hw.free_lt _ (Array.getElem_mem_toList ht)
  have hinj : ∀ a < reg.free.size, ∀ b < reg.free.size, reg.free[a]! = reg.free[b]! → a = b := by
    intro a ha b hb hab
    rw [getElem!_pos _ a ha, getElem!_pos _ b hb] at hab
    have := hw.nodup
    rw [← Array.getElem_toList ha, ← Array.getElem_toList hb] at hab
    exact (List.Nodup.getElem_inj_iff this).1 hab
  by_cases hjk : j < st.c.kLog
  · rw [ite_eq_left hjk, foldl_set_miss (fun i => st.c.kLog + i) _ st.nbl _ j
      (by rw [foldl_set!_size, hsz0]; exact hj) (fun t _ => by omega)]
    cases hidx : reg.free.toList.idxOf? j with
    | some t =>
        obtain ⟨ht, hft, -⟩ := List.idxOf?_eq_some_iff.1 hidx
        rw [Array.length_toList] at ht
        rw [Array.getElem_toList] at hft
        have hft' : reg.free[t]! = j := by rw [getElem!_pos _ t ht]; exact hft
        dsimp only
        rw [← hft', foldl_set_hit (fun t => reg.free[t]!) (fun t => r[t]!) _ hinj _ t ht
          (by rw [hsz0]; have := hfree t ht; omega)]
    | none =>
        have hmiss : ∀ t < reg.free.size, reg.free[t]! ≠ j := by
          intro t ht hft
          rw [getElem!_pos _ t ht] at hft
          have : j ∈ reg.free.toList := hft ▸ Array.getElem_mem_toList ht
          exact (List.idxOf?_eq_none_iff.1 hidx) this
        dsimp only
        rw [foldl_set_miss (fun t => reg.free[t]!) _ _ _ j (by rw [hsz0]; exact hj) hmiss,
          getElem!_pos _ j (by rw [hsz0]; exact hj), Array.getElem_map, Array.getElem_range]
        simp [hjk]
  · rw [ite_eq_right hjk]
    have hj' : j = st.c.kLog + (j - st.c.kLog) := by omega
    conv_lhs => rw [hj']
    rw [foldl_set_hit (fun i => st.c.kLog + i) (fun i => r[PT_LOCAL + i]!) st.nbl
      (fun a _ b _ h => by omega) _ (j - st.c.kLog) (by omega) (by rw [foldl_set!_size, hsz0]; omega)]

theorem model_regionPoint_lt (S : Model.Statement) (R : Model.Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (rr : Fin mPts → 𝔽) (j : ℕ) (hj : j < S.m) (hjk : j < S.kLog) :
    Model.regionPoint S R ptLocal mPts rr ⟨j, hj⟩ =
      match R.free.idxOf? ⟨j, hjk⟩ with
      | some t => if ht : t < mPts then rr ⟨t, ht⟩ else 0
      | none => if R.fixedBits ⟨j, hjk⟩ then 1 else 0 := by
  unfold Model.regionPoint; rw [dite_eq_left hjk]; rfl

theorem model_regionPoint_ge (S : Model.Statement) (R : Model.Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (rr : Fin mPts → 𝔽) (j : ℕ) (hj : j < S.m) (hjk : ¬ j < S.kLog) :
    Model.regionPoint S R ptLocal mPts rr ⟨j, hj⟩ =
      if hi : ptLocal + (j - S.kLog) < mPts then rr ⟨ptLocal + (j - S.kLog), hi⟩ else 0 := by
  unfold Model.regionPoint; rw [dite_eq_right hjk]

/-- **The region's point is the model's**, coordinate by coordinate (§16.2). -/
theorem regionPoint_model (st : Stmt) (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (reg : Region) (hw : RegionWF st.c.kLog reg) (hm : st.m = st.c.kLog + st.nbl) (r : Array F128)
    (j : ℕ) (hj : j < st.m) :
    Model.regionPoint (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL (PT_LOCAL + st.nbl)
      (fun t => fOf r[t.val]!) ⟨j, hj⟩ = fOf (st.regionPoint reg r)[j]! := by
  rw [regionPoint_get st reg r hm hw j hj]
  by_cases hjk : j < st.c.kLog
  · refine (model_regionPoint_lt (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL
      (PT_LOCAL + st.nbl) (fun t => fOf r[t.val]!) j hj hjk).trans ?_
    rw [ite_eq_left hjk]
    split
    · rename_i t heq
      have hidx : reg.free.toList.idxOf? j = some t :=
        (idxOf?_filterMap_fin st.c.kLog j hjk _ hw.free_lt).symm.trans heq
      obtain ⟨ht, -, -⟩ := List.idxOf?_eq_some_iff.1 hidx
      rw [Array.length_toList] at ht
      rw [hidx]
      simp only [dite_eq_left (show t < PT_LOCAL + st.nbl by have := hw.free_size; omega)]
    · rename_i heq
      have hidx : reg.free.toList.idxOf? j = none :=
        (idxOf?_filterMap_fin st.c.kLog j hjk _ hw.free_lt).symm.trans heq
      rw [hidx]
      simp only [regionOf]
      by_cases hc : ((reg.mask >>> j) % 2 == 1 && (reg.fixed >>> j) % 2 == 1) = true
      · rw [ite_eq_left hc]; exact (ite_eq_left hc).trans GF128.one_eq
      · rw [ite_eq_right hc]; exact (ite_eq_right hc).trans of_zero'.symm
  · refine (model_regionPoint_ge (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL
      (PT_LOCAL + st.nbl) (fun t => fOf r[t.val]!) j hj hjk).trans ?_
    rw [ite_eq_right hjk, dite_eq_left (show PT_LOCAL + (j - (stmtOf st hwf regs).kLog) < PT_LOCAL + st.nbl from by
      show PT_LOCAL + (j - st.c.kLog) < PT_LOCAL + st.nbl; omega)]
    rfl

theorem regionPoint_size (st : Stmt) (reg : Region) (r : Array F128) : (st.regionPoint reg r).size = st.m := by
  rw [regionPoint_eq, foldl_set!_size, foldl_set!_size]; simp

/-- The executable's claim of region `reg` at link point `r`, with value `v` (`Stmt.extraClaims`). -/
def regionClaimOf (st : Stmt) (reg : Region) (r : Array F128) (v : F128) : Flock.Claim :=
  ⟨v, eqTable ((st.regionPoint reg r).extract 0 6), (st.regionPoint reg r).extract 6 (st.regionPoint reg r).size⟩

theorem model_regionClaim_skip (S : Model.Statement) (R : Model.Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (rr : Fin mPts → 𝔽) (s : Fin 64) :
    (Model.regionClaim S R ptLocal mPts rr).skip s = ∏ j : Fin 6,
      (if s.val.testBit j then (if h : j.val < S.m then Model.regionPoint S R ptLocal mPts rr ⟨j.val, h⟩ else 0)
        else 1 + (if h : j.val < S.m then Model.regionPoint S R ptLocal mPts rr ⟨j.val, h⟩ else 0)) := rfl

theorem model_regionClaim_x (S : Model.Statement) (R : Model.Region S.kLog S.nbl) (ptLocal mPts : ℕ)
    (rr : Fin mPts → 𝔽) (i : Fin (S.m - 6)) :
    (Model.regionClaim S R ptLocal mPts rr).x i =
      if h : i.val + 6 < S.m then Model.regionPoint S R ptLocal mPts rr ⟨i.val + 6, h⟩ else 0 := rfl

/-- **One region claim is the model's** (§16.2): at a link point of `m_pts` coordinates, the executable's claim of a
well-formed region is the model's `regionClaim`. -/
theorem regionClaim_rel (st : Stmt) (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (reg : Region) (hw : RegionWF st.c.kLog reg) (hm : st.m = st.c.kLog + st.nbl) (r : Array F128)
    (hr : r.size = PT_LOCAL + st.nbl) (v : F128) (hv : st.regionValue reg r = .ok v) :
    ClaimRel st.m (regionClaimOf st reg r v)
      (Model.regionClaim (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL
        (PT_LOCAL + st.nbl) (fun t => fOf r[t.val]!)) := by
  have hpsz := regionPoint_size st reg r
  have h6 : 6 ≤ st.m := by have := hwf.hk6; have := hwf.hk; omega
  refine ⟨regionValue_model st hwf regs reg hw hm r hr v hv, ?_, ?_, ?_, ?_⟩
  · show (eqTable _).size = 64
    rw [eqTable_size, Array.size_extract, hpsz, show min 6 st.m - 0 = 6 by omega]; rfl
  · intro s
    have hW := eqTable_extract (st.regionPoint reg r) 0 6 (by omega) s.val (by simp)
    simp only [zero_add] at hW
    refine (model_regionClaim_skip (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL
      (PT_LOCAL + st.nbl) (fun t => fOf r[t.val]!) s).trans ?_
    show _ = GF128.of (eqTable ((st.regionPoint reg r).extract 0 6))[s.val]!
    rw [hW, Finset.prod_range (fun x => if s.val.testBit x = true then GF128.of (st.regionPoint reg r)[x]!
      else 1 + GF128.of (st.regionPoint reg r)[x]!)]
    refine Finset.prod_congr rfl fun j _ => ?_
    have hj : j.val < st.m := by omega
    rw [dite_eq_left (show j.val < (stmtOf st hwf regs).m from hj),
      regionPoint_model st hwf regs reg hw hm r j.val hj]
  · show (Array.extract _ 6 _).size = st.m - 6
    rw [Array.size_extract, hpsz]; omega
  · intro i
    have hi : i.val + 6 < st.m := by omega
    refine (model_regionClaim_x (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL
      (PT_LOCAL + st.nbl) (fun t => fOf r[t.val]!) i).trans ?_
    rw [dite_eq_left (show i.val + 6 < (stmtOf st hwf regs).m from hi),
      regionPoint_model st hwf regs reg hw hm r (i.val + 6) hi]
    show _ = fOf ((st.regionPoint reg r).extract 6 (st.regionPoint reg r).size)[i.val]!
    rw [getElem!_pos _ i.val (by rw [Array.size_extract, hpsz]; omega), Array.getElem_extract,
      show i.val + 6 = 6 + i.val from Nat.add_comm _ _, getElem!_pos _ (6 + i.val) (by omega)]

/-! ### `extraClaims`: region-major, then point by point -/

theorem list_forIn_push {α β γ : Type} (g : α → V γ) (mk : α → γ → β) :
    ∀ (L : List α) (out out' : Array β),
      (forIn L out fun a out => do let v ← g a; pure (ForInStep.yield (out.push (mk a v)))) = .ok out' →
      ∃ es, List.Forall₂ (fun a e => ∃ v, g a = .ok v ∧ e = mk a v) L es ∧ out'.toList = out.toList ++ es
  | [], out, out', h => by
      simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨[], .nil, by simp [h]⟩
  | a :: L, out, out', h => by
      rw [List.forIn_cons] at h
      obtain ⟨y, hy, h⟩ := ebind_ok h
      obtain ⟨v, hv, hy⟩ := ebind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      obtain ⟨es, hes, heq⟩ := list_forIn_push g mk L _ out' h
      exact ⟨mk a v :: es, .cons ⟨v, hv, rfl⟩ hes, by simp [heq]⟩

theorem list_forIn_outer {α β : Type} (F : α → Array β → V (Array β)) (P : α → List β → Prop)
    (hF : ∀ a out out', F a out = .ok out' → ∃ es, P a es ∧ out'.toList = out.toList ++ es) :
    ∀ (L : List α) (out out' : Array β),
      (forIn L out fun a out => do let out ← F a out; pure (ForInStep.yield out)) = .ok out' →
      ∃ ess, List.Forall₂ P L ess ∧ out'.toList = out.toList ++ ess.flatten
  | [], out, out', h => by
      simp only [List.forIn_nil, pure, Except.pure, Except.ok.injEq] at h
      exact ⟨[], .nil, by simp [h]⟩
  | a :: L, out, out', h => by
      rw [List.forIn_cons] at h
      obtain ⟨y, hy, h⟩ := ebind_ok h
      obtain ⟨o, ho, hy⟩ := ebind_ok hy
      simp only [pure, Except.pure, Except.ok.injEq] at hy
      subst hy
      obtain ⟨es, hes, heq⟩ := hF a out o ho
      obtain ⟨ess, hess, heq'⟩ := list_forIn_outer F P hF L _ out' h
      exact ⟨es :: ess, .cons hes hess, by simp [heq', heq]⟩

/-- When `extraClaims` succeeds, its claims are each region's, point by point, with the values `regionValue` gave. -/
theorem extraClaims_ok (st : Stmt) (pts : Array (Array F128)) (extras : Array Flock.Claim)
    (h : st.extraClaims pts = .ok extras) :
    ∃ ess : List (List Flock.Claim),
      List.Forall₂ (fun reg es => List.Forall₂ (fun r e => ∃ v, st.regionValue reg r = .ok v ∧
        e = regionClaimOf st reg r v) pts.toList es) st.regions.toList ess ∧ extras.toList = ess.flatten := by
  unfold Stmt.extraClaims at h
  dsimp only at h
  obtain ⟨o, ho, h⟩ := ebind_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  rw [← Array.forIn_toList] at ho
  obtain ⟨ess, hess, heq⟩ := list_forIn_outer _ _ (fun reg out out' hr => by
    rw [← Array.forIn_toList] at hr
    exact list_forIn_push (st.regionValue reg) (regionClaimOf st reg) _ out out' hr) _ #[] o ho
  exact ⟨ess, hess, by simpa using heq⟩

/-! ### The extra claims are the model's -/

/-- The link points as the model reads them: coordinate `t` of point `k` (§16.2). -/
def ptsOf (mPts : ℕ) (points : Array (Array F128)) : Fin 2 → Fin mPts → 𝔽 := fun k t => fOf (points[k.val]!)[t.val]!

/-- The facts about an executable statement's regions that reading its claims as the model's needs. -/
structure RegionsWF (st : Stmt) : Prop where
  hm : st.m = st.c.kLog + st.nbl
  mPts : st.mPts = PT_LOCAL + st.nbl
  regions : ∀ reg ∈ st.regions.toList, RegionWF st.c.kLog reg

theorem regionClaim_rel' (st : Stmt) (hwf : StmtWF st) (regs : List (Model.Region st.c.kLog (st.m - st.c.kLog)))
    (reg : Region) (hw : RegionWF st.c.kLog reg) (hm : st.m = st.c.kLog + st.nbl) (mPts : ℕ)
    (hmP : mPts = PT_LOCAL + st.nbl) (r : Array F128) (hr : r.size = mPts) (rr : Fin mPts → 𝔽)
    (hrr : ∀ t, rr t = fOf r[t.val]!) (v : F128) (hv : st.regionValue reg r = .ok v) :
    ClaimRel st.m (regionClaimOf st reg r v)
      (Model.regionClaim (stmtOf st hwf regs) (regionOf st.c.kLog (st.m - st.c.kLog) reg) PT_LOCAL mPts rr) := by
  subst hmP
  obtain rfl : rr = fun t => fOf r[t.val]! := funext hrr
  exact regionClaim_rel st hwf regs reg hw hm r hr v hv

/-- **The statement's extra claims are the model's** (§11, §16.2). At two link points of `m_pts` coordinates, the extra
claims `Setup.ofCircuit st` hands the verifier are `extraClaims` of `stmtOf st` with the regions read off `st.regions`. -/
theorem ofCircuit_extra (st : Stmt) (hwf : StmtWF st) (hr : RegionsWF st) (points : Array (Array F128))
    (hpts : points.size = 2) (hsz : ∀ p ∈ points.toList, p.size = st.mPts) :
    ExtraRealizes (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) points
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts points)) := by
  intro extras hex
  obtain ⟨ess, hess, heq⟩ := extraClaims_ok st points extras hex
  have h0 : 0 < points.size := by omega
  have h1 : 1 < points.size := by omega
  have hlist : points.toList = [points[0]!, points[1]!] := by
    rw [getElem!_pos points 0 h0, getElem!_pos points 1 h1]
    apply List.ext_getElem (by simp [hpts])
    intro n hn _
    have : n < 2 := by simpa [hpts] using hn
    interval_cases n <;> simp
  have hs0 : points[0]!.size = st.mPts := hsz _ (by rw [hlist]; simp)
  have hs1 : points[1]!.size = st.mPts := hsz _ (by rw [hlist]; simp)
  have key : ∀ (L : List Region) (ess : List (List Flock.Claim)), (∀ reg ∈ L, RegionWF st.c.kLog reg) →
      List.Forall₂ (fun reg es => List.Forall₂ (fun r e => ∃ v, st.regionValue reg r = .ok v ∧
        e = regionClaimOf st reg r v) points.toList es) L ess →
      List.Forall₂ (ClaimRel st.m) ess.flatten ((L.map (regionOf st.c.kLog (st.m - st.c.kLog))).flatMap fun R =>
        [Model.regionClaim (stmtOf st hwf (regionsOf st)) R PT_LOCAL st.mPts (ptsOf st.mPts points 0),
         Model.regionClaim (stmtOf st hwf (regionsOf st)) R PT_LOCAL st.mPts (ptsOf st.mPts points 1)]) := by
    intro L ess hL h
    induction h with
    | nil => exact .nil
    | @cons reg es L' ess' hes _ ih =>
        rw [hlist] at hes
        rcases hes with _ | ⟨⟨v0, hv0, rfl⟩, hrest⟩
        rcases hrest with _ | ⟨⟨v1, hv1, rfl⟩, hnil⟩
        rcases hnil
        have hw := hL reg List.mem_cons_self
        simp only [List.map_cons, List.flatten_cons, List.cons_append, List.nil_append]
        exact .cons (regionClaim_rel' st hwf _ reg hw hr.hm st.mPts hr.mPts _ hs0 _ (fun t => rfl) v0 hv0)
          (.cons (regionClaim_rel' st hwf _ reg hw hr.hm st.mPts hr.mPts _ hs1 _ (fun t => rfl) v1 hv1)
            (ih fun x hx => hL x (List.mem_cons_of_mem _ hx)))
  have hF : List.Forall₂ (ClaimRel st.m) extras.toList
      (Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts points)) := by
    rw [heq]; exact key st.regions.toList ess hr.regions hess
  refine ⟨by rw [← Array.length_toList]; exact hF.length_eq, fun j hj => ?_⟩
  have hjx : j < extras.toList.length := by rw [hF.length_eq]; exact hj
  rw [getElem!_pos extras j (by simpa using hjx), ← Array.getElem_toList]
  exact hF.get hjx hj

/-! ### The record's link points, and the composition -/

theorem check_pts {pts : Array (Array F128)} {n k : ℕ} {why : String} {u : Unit}
    (h : Record.check (pts.size == n && pts.all fun x => x.size == k) why = .ok u) :
    pts.size = n ∧ ∀ p ∈ pts.toList, p.size = k := by
  have := check_ok_iff.1 h
  simp only [Bool.and_eq_true, beq_iff_eq, Array.all_eq_true] at this
  refine ⟨this.1, fun p hp => ?_⟩
  obtain ⟨i, hi, rfl⟩ := List.getElem_of_mem hp
  simpa using this.2 i (by simpa using hi)

set_option maxHeartbeats 1000000 in
/-- The record's link points have the configured shape: `Record.decode` checks it (S9). -/
theorem decode_points {spec : SessionSpec} {text : String} {sess : Session}
    (h : Record.decode spec text = .ok sess) :
    sess.points.size = spec.points ∧ ∀ p ∈ sess.points.toList, p.size = spec.mPts := by
  unfold Record.decode at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    simp only [pure, Except.pure, Except.ok.injEq] at h
    subst h
    exact check_pts (by assumption)

open Game Model in
/-- **The verifier refines the model's reps, for the statements it is set up with** (§3, §5, §10, §16). When
`Flock.verify` accepts a one-table session under the unsalted SHA-512 scheme, with the setup `Setup.ofCircuit st` of a
statement whose layout and regions are well formed, the conclusion of `verify_refines` holds for the model statement
`stmtOf st` read off the layout, with the extra claims the model's `extraClaims` at the record's link points. -/
theorem verify_refines_ofCircuit (st : Stmt) (hwf : StmtWF st) (hr : RegionsWF st) (hm : 13 ≤ st.m) (msId : String)
    (hms : st.merkle = MerkleScheme.sha512 msId) (record : String) (proofs : Array ByteArray)
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
      OpensOK H512 enc512 (capDig p₀.cap) 0 pend₀ (opensOf p₀ p₁ sch) ∧
      OpensOK H512 enc512 (capDig p₀.cap) 1 pend₁ (opensOf p₀ p₁ sch) :=
  verify_refines st.spec (Setup.ofCircuit st) (stmtOf st hwf (regionsOf st)) hm rfl rfl (ofCircuit_fold st hwf _) msId hms
    (fun sess => Model.extraClaims (stmtOf st hwf (regionsOf st)) PT_LOCAL st.mPts (ptsOf st.mPts sess.points))
    record proofs
    (fun sess hsess => ofCircuit_extra st hwf hr sess.points (decode_points hsess).1 (decode_points hsess).2) h

end FlockSoundness.Refine
