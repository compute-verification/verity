module

public import Proofs.Properties.Verity.Sampled.Record.Facts
public import Proofs.Flock.Claims.Sampling.Select.Size
public import Proofs.Flock.Claims.Sampling.Session
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.Count
public import Proofs.Flock.Bridge.Scope
public import Proofs.Flock.Bridge.PopMatches
public import Proofs.Flock.Claims.ZK.Prefix
public import Proofs.Flock.Claims.ERepCost
public import Proofs.Flock.Level3.ArithFacts
public import Proofs.Flock.Soundness.Refine.Paths
public import Proofs.Flock.Soundness.Discharge.ZkView.Decode

/-!
# The draw law at one-stage v2's session

* `drawLaw`: the record's draw meets `DrawLaw`. The draw coins are all first-stage (`stage2` is the session's rounds,
  `coinsAt_congr`), so a set `B`'s weight is its miss indicator at the first-stage coins. Its mean is the probability that
  v2's challenger draw (`Spot.Draw.drawUnits`) on the draw coins' bytes misses `B`; a draw checks at most the
  population's `n` units (`kd_le_of_lawOn`), so every position it holds is checked. The bytes of `DRAW_ROUNDS` uniform
  coins are uniform (`pr_drawBytes`), so that probability is at most the law's `P_a(B)` on the registration's strata
  (`Flock.Bridge.drawUnits_miss_le`), which is the statement's design's on its partition (`Flock.Bridge.pa_partOf`). A
  failed draw draws every unit and misses no nonempty `B`.
* `ScopeOk`, `refuses_of_scopeOk`: the record refuses what is out of scope (`Refuses`) where its statement is in scope,
  and of that scope `Live` gives the program's order; `ScopeOk` is the rest.
* `pr_drawBytes`: the draw coins' bytes, as v2's challenger reads them (`drawBytes`), are a bijective image of the draw
  rounds' coins (`srcOf`: every coin's elements' 16 bytes, in order), so an event of them has its probability on uniform
  bytes.
-/

@[expose] public section

namespace Verity.Sampled.Record

open Claims Claims.Sampling

/-! ## Lists of pieces of one length -/

theorem length_flatMap_const {α β : Type} {f : α → List β} {m : ℕ} (hf : ∀ a, (f a).length = m) :
    ∀ l : List α, (l.flatMap f).length = l.length * m
  | [] => by simp
  | a :: l => by
    rw [List.flatMap_cons, List.length_append, hf, length_flatMap_const hf l, List.length_cons, Nat.add_mul,
      Nat.one_mul]
    exact Nat.add_comm _ _

theorem flatMap_inj_const {α β : Type} {f : α → List β} {m : ℕ} (hf : ∀ a, (f a).length = m)
    (hinj : ∀ a b, f a = f b → a = b) :
    ∀ {l l' : List α}, l.length = l'.length → l.flatMap f = l'.flatMap f → l = l'
  | [], [], _, _ => rfl
  | [], _ :: _, hl, _ => by simp at hl
  | _ :: _, [], hl, _ => by simp at hl
  | a :: l, b :: l', hl, h => by
    rw [List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨h1, h2⟩ := List.append_inj h (by rw [hf, hf])
    rw [hinj a b h1, flatMap_inj_const hf hinj (by simpa using hl) h2]

theorem eq_of_flatMap_eq {α β : Type} {f g : α → List β} {m : ℕ} (hf : ∀ a, (f a).length = m)
    (hg : ∀ a, (g a).length = m) : ∀ l : List α, l.flatMap f = l.flatMap g → ∀ a ∈ l, f a = g a
  | [], _ => by simp
  | a :: l, h => by
    rw [List.flatMap_cons, List.flatMap_cons] at h
    obtain ⟨h1, h2⟩ := List.append_inj h (by rw [hf, hg])
    intro b hb
    rcases List.mem_cons.1 hb with rfl | hb
    · exact h1
    · exact eq_of_flatMap_eq hf hg l h2 b hb

/-! ## The draw coins' bytes -/

/-- The bytes the draw rounds' coins hold. -/
abbrev drawLen : ℕ := DRAW_ROUNDS * (65536 * 16)

theorem f128_bytes_length (a : _root_.Flock.F128) : a.toBytes.data.toList.length = 16 := by
  rw [Array.length_toList, ByteArray.size_data, FlockSoundness.Discharge.ZkView.toBytes_size]

theorem coin_bytes_length (c : FlockDeploy.Coin) : c.bytes.length = 65536 * 16 := by
  unfold FlockDeploy.Coin.bytes
  rw [length_flatMap_const (f := fun i : Fin 65536 => (c i).toBytes.data.toList) (m := 16)
    (fun i => f128_bytes_length (c i)), List.length_finRange]

theorem coin_bytes_inj {c c' : FlockDeploy.Coin} (h : c.bytes = c'.bytes) : c = c' := by
  have hp := eq_of_flatMap_eq (f := fun i : Fin 65536 => (c i).toBytes.data.toList)
    (g := fun i : Fin 65536 => (c' i).toBytes.data.toList) (fun i => f128_bytes_length (c i))
    (fun i => f128_bytes_length (c' i)) (List.finRange 65536) h
  exact funext fun i =>
    FlockSoundness.Refine.toBytes_inj (ByteArray.ext (Array.ext' (hp i (List.mem_finRange i))))

/-- **The draw rounds' coins' bytes**, in order. -/
def drawList (c : Fin DRAW_ROUNDS → FlockDeploy.Coin) : List UInt8 := (List.ofFn c).flatMap FlockDeploy.Coin.bytes

theorem drawList_length (c : Fin DRAW_ROUNDS → FlockDeploy.Coin) : (drawList c).length = drawLen := by
  unfold drawList
  rw [length_flatMap_const coin_bytes_length, List.length_ofFn]

/-- **The draw rounds' coins as bytes.** -/
def srcOf (c : Fin DRAW_ROUNDS → FlockDeploy.Coin) (j : Fin drawLen) : Fin 256 :=
  ((drawList c)[j.val]'(by rw [drawList_length]; exact j.isLt)).toFin

theorem ofFin_toFin_u8 (x : UInt8) : UInt8.ofFin x.toFin = x := rfl

theorem bytesOf_srcOf (c : Fin DRAW_ROUNDS → FlockDeploy.Coin) :
    _root_.Properties.Spot.bytesOf (srcOf c) = drawBytes (List.ofFn c) := by
  have hlen : ((List.ofFn c).flatMap FlockDeploy.Coin.bytes).length = drawLen := drawList_length c
  have h : (Array.ofFn fun i => UInt8.ofFin (srcOf c i)) =
      ((List.ofFn c).flatMap FlockDeploy.Coin.bytes).toArray := by
    apply Array.ext
    · rw [Array.size_ofFn, List.size_toArray, hlen]
    · intro i _ _
      rw [Array.getElem_ofFn, List.getElem_toArray]
      exact ofFin_toFin_u8 _
  unfold _root_.Properties.Spot.bytesOf drawBytes
  rw [h]

theorem ofFn_inj_of {α : Type} {n : ℕ} {f g : Fin n → α} (h : List.ofFn f = List.ofFn g) : f = g := by
  funext i
  have := congrArg (fun l => l[i.val]?) h
  simpa using this

theorem srcOf_inj : Function.Injective srcOf := by
  intro c c' h
  have h1 : drawBytes (List.ofFn c) = drawBytes (List.ofFn c') := by rw [← bytesOf_srcOf, ← bytesOf_srcOf, h]
  have h2 : drawList c = drawList c' := List.toArray_inj (congrArg ByteArray.data h1)
  exact ofFn_inj_of (flatMap_inj_const coin_bytes_length (fun _ _ => coin_bytes_inj) (by simp) h2)

theorem card_draw : Fintype.card (Fin DRAW_ROUNDS → FlockDeploy.Coin) = Fintype.card (Fin drawLen → Fin 256) := by
  have hC : Fintype.card FlockDeploy.Coin = (256 ^ 16) ^ 65536 := by
    show Fintype.card (Fin 65536 → FlockLevel3.GF128) = _
    rw [Fintype.card_fun, FlockLevel3.card_F, Fintype.card_fin, show (2 : ℕ) ^ 128 = 256 ^ 16 by norm_num]
  rw [Fintype.card_fun, Fintype.card_fun, hC, Fintype.card_fin, Fintype.card_fin, Fintype.card_fin, ← pow_mul,
    ← pow_mul, show 16 * (65536 * DRAW_ROUNDS) = drawLen by unfold drawLen; ring]

theorem fiber_srcOf (b : Fin drawLen → Fin 256) : Nat.card {c // srcOf c = b} = 1 := by
  obtain ⟨c, hc⟩ := ((Fintype.bijective_iff_injective_and_card srcOf).2 ⟨srcOf_inj, card_draw⟩).2 b
  exact Nat.card_eq_one_iff_unique.2 ⟨⟨fun x y => Subtype.ext (srcOf_inj (x.2.trans y.2.symm))⟩, ⟨⟨c, hc⟩⟩⟩

theorem coinsAt_eq {k : ℕ} (hk : DRAW_ROUNDS ≤ k) (r : Fin k → FlockDeploy.Coin) :
    coinsAt r = List.ofFn fun i : Fin DRAW_ROUNDS => r (Fin.castLE hk i) := by
  apply List.ext_getElem
  · simp [coinsAt]
  · intro i h1 _
    have hi : i < DRAW_ROUNDS := by simpa [coinsAt] using h1
    simp only [coinsAt, List.getElem_map, List.getElem_range, List.getElem_ofFn]
    unfold coinSeq
    exact dite_eq_left (Nat.lt_of_lt_of_le hi hk)

/-- **The draw coins' bytes are uniform**: an event of the bytes the draw rounds' coins hold has its probability on
uniform bytes. -/
theorem pr_drawBytes {k : ℕ} (hk : DRAW_ROUNDS ≤ k) (Q : ByteArray → Prop) :
    Pr (fun r : Fin k → FlockDeploy.Coin => Q (drawBytes (coinsAt r))) =
      Pr (fun src : Fin drawLen → Fin 256 => Q (_root_.Properties.Spot.bytesOf src)) := by
  classical
  have h1 : Pr (fun r : Fin k → FlockDeploy.Coin => Q (drawBytes (coinsAt r))) =
      Pr (fun c : Fin DRAW_ROUNDS → FlockDeploy.Coin => Q (drawBytes (List.ofFn c))) := by
    simp only [coinsAt_eq hk]
    exact Claims.Sampling.Pr_comp_inj (Fin.castLE hk) (Fin.castLE_injective hk) fun c => Q (drawBytes (List.ofFn c))
  have h2 : Pr (fun c : Fin DRAW_ROUNDS → FlockDeploy.Coin => Q (drawBytes (List.ofFn c))) =
      Pr (fun c : Fin DRAW_ROUNDS → FlockDeploy.Coin => Q (_root_.Properties.Spot.bytesOf (srcOf c))) :=
    Pr_congr fun c => by rw [bytesOf_srcOf]
  rw [h1, h2]
  exact Claims.ZKMal.Pr_comp_fiber srcOf fiber_srcOf fun src => Q (_root_.Properties.Spot.bytesOf src)

/-! ## The law -/

/-- **A law on the strata draws at most the population**: each stratum's `k` is at most its size. -/
theorem kd_le_of_lawOn {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ} (h : Flock.Bridge.LawOn law ns kd) :
    kd ≤ ns.sum := by
  cases law with
  | subset k =>
    obtain ⟨-, hk⟩ := h
    exact hk
  | bernoulli a b => exact h.elim
  | stratified K St =>
    obtain ⟨⟨-, hsz⟩, hK, hn, hk, rfl⟩ := h
    have hle : (St.toList.map (·.k)).sum ≤ (St.toList.map (·.size)).sum := List.sum_le_sum fun s hs => by
      show s.k ≤ s.size
      have hm : s.size ∈ ns := hsz ▸ List.mem_map_of_mem hs
      rw [hk s hs]
      exact Flock.Bridge.stratumK_le hK (hn _ hm) (List.le_sum_of_mem hm)
    rwa [hsz] at hle
  | work K ws =>
    obtain ⟨⟨-, hsz⟩, hk, rfl⟩ := h
    have hle : (ws.toList.map (·.k)).sum ≤ ((ws.toList.map (·.toStratum)).map (·.size)).sum := by
      rw [List.map_map]
      refine List.sum_le_sum fun s hs => ?_
      show s.k ≤ s.size
      rw [hk s hs]
      exact min_le_left _ _
    rwa [hsz] at hle

variable (A : Live) (o : Own) (e : Ext)

theorem record_checked_xOf (r : Fin ((record A o e).V.k (xOf A o)) → FlockDeploy.Coin) :
    (record A o e).checked (xOf A o) r = drawn A (drawBytes (coinsAt r)) :=
  (record_checked A o e (xOf A o) r).trans (ite_eq_left rfl)

theorem nPU_record : nPU (record A o e) (xOf A o) = A.n := by
  have hlen : (partA A).blocks.length = A.n :=
    Flock.Bridge.blocks_length_partOf A.P (Flock.Bridge.nsOf_sum A.hpop)
  unfold nPU
  rw [record_parse]
  exact hlen

theorem drawn_of_ok {src : ByteArray} {u : Array ℕ} {rd : ℕ}
    (h : _root_.Spot.Draw.drawUnits A.design A.n src = .ok (u, rd)) : drawn A src = u.toList := by
  simp only [drawn, h]

theorem drawn_of_error {src : ByteArray} {err : String}
    (h : _root_.Spot.Draw.drawUnits A.design A.n src = .error err) : drawn A src = List.range A.n := by
  simp only [drawn, h]

/-- **The record's draw law.** -/
theorem drawLaw : DrawLaw (record A o e) := by
  classical
  intro x s Pt hx hP _ B hB
  have hxo : x = xOf A o := by
    by_contra h
    have hn : (record A o e).parse x = none := (record_parse_eq A o e x).trans (ite_eq_right h)
    rw [hn] at hx
    cases hx
  subst hxo
  rw [record_parse] at hx
  cases hx
  change some (partA A) = some Pt at hP
  cases hP
  have hns := Flock.Bridge.nsOf_sum A.hpop
  have hnPU := nPU_record A o e
  have hlen : (partA A).blocks.length = A.n := Flock.Bridge.blocks_length_partOf A.P hns
  rw [hlen] at hB
  refine ⟨fun r => if (∀ v ∈ B, ∀ p : Fin (nPU (record A o e) (xOf A o)),
      ((record A o e).checked (xOf A o) r)[p.val]? ≠ some v) then 1 else 0,
    fun _ => ite_nonneg zero_le_one le_rfl, fun r => ?_, ?_⟩
  · have hfork : ∀ d, (record A o e).checked (xOf A o) (forkOn ((record A o e).stage2 (xOf A o)) r d) =
        (record A o e).checked (xOf A o) r := fun d => by
      rw [record_checked_xOf, record_checked_xOf,
        coinsAt_congr (r := forkOn ((record A o e).stage2 (xOf A o)) r d) (r' := r) fun t ht => ite_eq_right ht]
    simp only [hfork]
    split_ifs with hE
    · exact Pr_le_one _
    · exact (Pr_congr fun _ => iff_false_intro hE).trans_le Pr_false.le
  · refine (Ex_ind _).trans_le ?_
    have hPa : (stmtA A).design.Pa (partA A) B =
        (_root_.Properties.Spot.oneStageOf A.design).Pa (Flock.Bridge.nsOf A.desc A.qr) B :=
      Flock.Bridge.pa_partOf A.P hns _ B hB
    rw [hPa]
    rcases B.eq_empty_or_nonempty with rfl | ⟨v0, hv0⟩
    · rw [Flock.Bridge.pa_empty_of_lawOn A.hlaw]
      exact Pr_le_one _
    have hk : DRAW_ROUNDS ≤ (record A o e).V.k (xOf A o) := by
      show DRAW_ROUNDS ≤ (V A o).k (xOf A o)
      rw [V_k_xOf]
      omega
    refine (Pr_mono fun r hr => ?_).trans ((pr_drawBytes hk fun b => ∃ units read,
      _root_.Spot.Draw.drawUnits A.design A.n b = .ok (units, read) ∧ ∀ u ∈ B, u ∉ units).le.trans
      (Flock.Bridge.drawUnits_miss_le A.hlaw hns B hB drawLen))
    have hr' : ∀ v ∈ B, ∀ p : Fin A.n, (drawn A (drawBytes (coinsAt r)))[p.val]? ≠ some v := fun v hv p => by
      have h := hr v hv ⟨p.val, by rw [hnPU]; exact p.isLt⟩
      rwa [record_checked_xOf] at h
    show ∃ units read, _root_.Spot.Draw.drawUnits A.design A.n (drawBytes (coinsAt r)) = .ok (units, read) ∧
      ∀ u ∈ B, u ∉ units
    have hv0n : v0 < A.n := Finset.mem_range.1 (hB hv0)
    cases hd : _root_.Spot.Draw.drawUnits A.design A.n (drawBytes (coinsAt r)) with
    | error err =>
      have h := hr' v0 hv0 ⟨v0, hv0n⟩
      rw [drawn_of_error A hd] at h
      exact absurd (List.getElem?_range hv0n) h
    | ok res =>
      obtain ⟨units, read⟩ := res
      refine ⟨units, read, rfl, fun u hu hmem => ?_⟩
      obtain ⟨i, hi, rfl⟩ := Array.mem_iff_getElem.1 hmem
      have hsz : units.size = A.kd := _root_.Properties.Spot.SecurityProofs.drawUnits_size A.hlaw hd
      have hkd : A.kd ≤ A.n := (kd_le_of_lawOn A.hlaw).trans_eq hns
      have h := hr' _ hu ⟨i, by omega⟩
      rw [drawn_of_ok A hd] at h
      exact h (by simp [hi])

/-! ## The scope -/

/-- **What `Live` doesn't give of the record's scope** (`InScope`): V5's partition at the session's population
partitions the lowered program into nonempty proof blocks whose committed values fit their public size classes, and the
law's one-stage rule is well formed over the population's strata. The program's order is the lowering's own
(`Flock.Bridge.topological_progOf`). -/
def ScopeOk : Prop :=
  (partA A).Partitions (stmtA A).prog ∧
    (_root_.Properties.Spot.oneStageOf A.design).WellFormed (Flock.Bridge.nsOf A.desc A.qr)

theorem inScope_of_scopeOk (h : ScopeOk A) : InScope (partA A) (stmtA A) :=
  ⟨Flock.Bridge.topological_progOf, h.1,
    Flock.Bridge.designOf_wellFormed A.P (Flock.Bridge.nsOf_sum A.hpop) h.2⟩

/-- **The record's scope refusal**, where `ScopeOk` holds. -/
theorem refuses_of_scopeOk (H : Msg → Msg) (h : ScopeOk A) : Refuses H (record A o e) :=
  refuses_of_inScope A o e H (inScope_of_scopeOk A h)

end Verity.Sampled.Record
