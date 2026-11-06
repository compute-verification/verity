module

public import Proofs.Flock.Claims.Sampling.Select.Design
public import Proofs.Flock.Claims.Sampling.Count
public import Proofs.Flock.Soundness.ExecDraw
public import Definitions.Claims.SamplingTwo
public import Properties.SampledProofs.Draw
import all SampledProofs.Draw
import all Lean.Data.Json.Basic
import all Init.Data.Ord.String

/-!
# The second stage misses a set of proof blocks no more often than the replay blocks' escapes

What `Select.selectSecond`, a two-stage rule's second stage, misses on `L` uniform bytes (`bytesOf`), as the claims'
`CompiledTwo.Sessions` reads it (`secondDraws`). It is a lemma of the deploy instance (`SamplingDeploy.secondDraws_code`),
not a guarantee: a guarantee of the Properties lock would carry an honest witness that `selectSecond` accepts at a built
instance.

* `flockLaw_second`: `secondLaw`'s C-Flock law is stratified, its `i`-th stratum exactly `[i·n, (i+1)·n)` drawing `k`.
* `selectSecond_ok`: an accepted second stage has `1 ≤ k < n`, drew its units from the coins under that law, and proves
  each of them.
* `strataPa_replicate`: `strataPa` of `S` strata of `n` drawing `k` is the product of their windows' escapes.
* `secondMisses`: the second stage is accepted and proves no proof block of `X` with probability at most
  `∏_i escCell n k |X ∩ [i·n, (i+1)·n)|`. It takes no premise: the draw is a function of its inputs and its bytes, and a
  draw that would read past the `L` bytes is refused, so it misses nothing.
-/

@[expose] public section

open private SampledProofs.Draw.stratum? SampledProofs.Draw.natRanges from SampledProofs.Draw

open SampledProofs

namespace Properties.SampledProofs.SecurityProofs

open Lean (Json)
open Claims.Sampling Finset

/-- The proof blocks the second stage's draw proves (`Draw.proved`), none where it is refused. -/
def secondOf : Flock.V (Json × ℕ) → Array Int
  | .ok (draw, _) => Draw.proved draw
  | .error _ => #[]

/-- The `i`-th stratum object of `secondLaw`, replay block `r` of the sample. -/
def secondObj (n k i r : ℕ) : Json :=
  Json.mkObj [("template", Json.str (toString r)), ("units", Json.arr #[natsJson #[i * n, (i + 1) * n]]),
    ("k", Lean.toJson k)]

theorem getD_secondLaw_law (sample : Array ℕ) (n k : ℕ) :
    getD (Select.secondLaw sample n k) "law" = Json.str "stratified" := (rfl)

theorem getD_secondLaw_k (sample : Array ℕ) (n k : ℕ) :
    getD (Select.secondLaw sample n k) "k" = Lean.toJson (sample.size * k) := (rfl)

theorem getD_secondLaw_strata (sample : Array ℕ) (n k : ℕ) :
    getD (Select.secondLaw sample n k) "strata" = Json.arr (sample.mapIdx fun i r => secondObj n k i r) := (rfl)

theorem getD_secondObj_template (n k i r : ℕ) : getD (secondObj n k i r) "template" = Json.str (toString r) := (rfl)

theorem getD_secondObj_units (n k i r : ℕ) :
    getD (secondObj n k i r) "units" = Json.arr #[natsJson #[i * n, (i + 1) * n]] := (rfl)

theorem getD_secondObj_k (n k i r : ℕ) : getD (secondObj n k i r) "k" = Lean.toJson k := (rfl)

theorem int_toJson (a : ℕ) : int? (Lean.toJson a) = some (a : Int) := (rfl)

theorem ranges_pair {a b : ℕ} {rs : Array (Int × Int)} (h : Draw.ranges? (Json.arr #[natsJson #[a, b]]) = some rs) :
    rs = #[((a : Int), (b : Int))] := by
  unfold Draw.ranges? at h
  simp only [natsJson, Array.isEmpty, List.map_toArray, List.map_cons, List.map_nil, Array.mapM_eq_mapM_toList] at h
  by_cases hab : a < b <;> simp [hab, int_toJson] at h
  exact h.symm

private theorem natRanges_pair {a b : Int} {rs : Array (ℕ × ℕ)} (h : SampledProofs.Draw.natRanges #[(a, b)] = some rs) :
    rs = #[(a.toNat, b.toNat)] := by
  unfold SampledProofs.Draw.natRanges at h
  by_cases ha : 0 ≤ a <;> simp [ha, Array.mapM_eq_mapM_toList] at h
  exact h.symm

/-- A stratum object of `secondLaw` that parses is `[i·n, (i+1)·n)` drawing `k`. -/
private theorem stratum_second {n k i r : ℕ} {ws : Flock.Draw.WorkStratum}
    (h : SampledProofs.Draw.stratum? false (secondObj n k i r) = some ws) :
    ws.units = #[(i * n, (i + 1) * n)] ∧ ws.k = k := by
  unfold SampledProofs.Draw.stratum? at h
  simp only [getD_secondObj_template, getD_secondObj_units, getD_secondObj_k, nat_toJson, Bool.false_eq_true,
    ↓reduceIte] at h
  split at h
  · simp at h
  · simp only [Option.bind_eq_some_iff, pure, bind, Option.some.injEq] at h
    obtain ⟨rs0, hr, rs, hn, _, rfl, _, rfl, _, rfl, rfl⟩ := h
    obtain rfl := ranges_pair hr
    obtain rfl := natRanges_pair hn
    exact ⟨by simp only [Int.toNat_natCast], rfl⟩

theorem list_mapM_forall₂ {α β : Type} (f : α → Option β) :
    ∀ (xs : List α) (ys : List β), xs.mapM f = some ys → List.Forall₂ (fun a b => f a = some b) xs ys
  | [], ys, h => by
    simp only [List.mapM_nil, pure, Option.some.injEq] at h; subst h; exact .nil
  | x :: xs, ys, h => by
    simp only [List.mapM_cons, Option.bind_eq_some_iff, bind, pure, Option.some.injEq] at h
    obtain ⟨y, hy, ys', hys, rfl⟩ := h
    exact .cons hy (list_mapM_forall₂ f xs ys' hys)

theorem mapM_forall₂ {α β : Type} (f : α → Option β) (xs : Array α) (ys : Array β) (h : xs.mapM f = some ys) :
    List.Forall₂ (fun a b => f a = some b) xs.toList ys.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => simp [hl] at h
  | some ys' =>
    rw [hl] at h
    obtain rfl : ys = ys'.toArray := (Option.some.inj h).symm
    exact list_mapM_forall₂ f _ _ hl

/-- **`secondLaw`'s C-Flock law**: stratified, at `K = |sample|·k`, its `i`-th stratum `[i·n, (i+1)·n)` drawing `k`. -/
theorem flockLaw_second (sample : Array ℕ) (n k m : ℕ) (parsed : Flock.Draw.Law)
    (h : Draw.flockLaw (Select.secondLaw sample n k) m = .ok parsed) :
    ∃ St : Array Flock.Draw.Stratum, parsed = .stratified (sample.size * k) St ∧
      St.toList.map (fun s => (s.units, s.k)) = (List.range sample.size).map fun i => (#[(i * n, (i + 1) * n)], k) := by
  unfold Draw.flockLaw at h
  rw [getD_secondLaw_law] at h
  dsimp only at h
  split at h
  all_goals try (rename_i hs; exact absurd hs (by decide))
  · split at h
    · exact absurd h (throw_bind_ne _ _ _)
    rw [getD_secondLaw_k, getD_secondLaw_strata, nat_toJson] at h
    dsimp only at h
    split at h
    · exact absurd h (throw_bind_ne _ _ _)
    split at h
    · rename_i strata hss
      have hk : ("stratified" == "work") = false := by decide
      simp only [hk, Bool.false_eq_true, ↓reduceIte, pure, Except.pure, Except.ok.injEq] at h hss
      subst h
      refine ⟨_, rfl, ?_⟩
      have hF := mapM_forall₂ _ _ _ hss
      rw [Array.toList_mapIdx] at hF
      obtain ⟨hlen, hall⟩ := List.forall₂_iff_get.1 hF
      simp only [List.length_mapIdx, Array.length_toList] at hlen
      apply List.ext_getElem
      · simp [hlen]
      · intro j h1 h2
        have hj := hall j (by simpa [hlen] using h1) (by simpa using h1)
        simp only [List.get_eq_getElem, List.getElem_mapIdx] at hj
        obtain ⟨hu, hk⟩ := stratum_second hj
        simp only [Array.getElem_toList] at hu hk
        simp [hu, hk]
    · exact absurd h (throw_ne _ _)
  · exact absurd h (throw_ne _ _)

theorem getD_draw_units (sample : Array ℕ) (n k m : ℕ) (units : Array ℕ) :
    getD (Json.mkObj ((entries (Select.secondLaw sample n k)).toList ++
      [("population", Lean.toJson m), ("units", natsJson units)])) "units" = natsJson units := (rfl)

/-- A draw object whose `units` are `units` proves each of them. -/
theorem proved_units (draw : Json) (units : Array ℕ) (hd : getD draw "units" = natsJson units)
    (u : ℕ) (hu : u ∈ units) : (u : Int) ∈ Draw.proved draw := by
  unfold Draw.proved
  rw [FlockSoundness.Audit.DrawExec.mem_qsort]
  refine Array.mem_append_left _ ?_
  dsimp only
  rw [hd]
  simp only [natsJson, Array.mem_filterMap, Array.mem_map]
  exact ⟨_, ⟨u, hu, rfl⟩, int_toJson u⟩

/-- **`Select.selectSecond`'s steps**: an accepted second stage has `2 ≤ n` and `1 ≤ k < n`, drew its units from the
coins under `secondLaw`'s C-Flock law over the `|sample|·n` proof blocks, and proves each of them. -/
theorem selectSecond_ok {sample : Array ℕ} {n k : ℕ} {src : ByteArray} {draw : Json} {read : ℕ}
    (h : Select.selectSecond sample n k src = .ok (draw, read)) :
    2 ≤ n ∧ 0 < k ∧ k < n ∧ ∃ parsed units, Draw.flockLaw (Select.secondLaw sample n k) (sample.size * n) = .ok parsed ∧
      Draw.drawUnits parsed (sample.size * n) src = .ok (units, read) ∧ ∀ u ∈ units, (u : Int) ∈ Draw.proved draw := by
  unfold Select.selectSecond at h
  dsimp only at h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  rename_i hn hk
  simp only [Bool.or_eq_true, beq_iff_eq, decide_eq_true_eq, not_or, not_le] at hk
  refine ⟨by omega, by omega, hk.2, ?_⟩
  obtain ⟨⟨draw', read'⟩, hd, h⟩ := bind_eq_ok.1 h
  split at h
  · exact absurd h (throw_bind_ne _ _ _)
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
  obtain ⟨rfl, rfl⟩ := h
  unfold Draw.drawFirstRead at hd
  obtain ⟨parsed, hp, hd⟩ := bind_eq_ok.1 hd
  obtain ⟨⟨units, read''⟩, hu, hd⟩ := bind_eq_ok.1 hd
  simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at hd
  obtain ⟨rfl, rfl⟩ := hd
  exact ⟨parsed, units, hp, hu, proved_units _ _ (getD_draw_units _ _ _ _ _)⟩

theorem flatMap_strata (S n : ℕ) : (List.range S).flatMap (fun i => List.range' (i * n) n) = List.range (S * n) := by
  induction S with
  | zero => simp
  | succ S ih =>
    rw [List.range_succ, List.flatMap_append, ih, List.flatMap_singleton, Nat.succ_mul, List.range_add,
      List.range'_eq_map_range]

/-- `secondLaw`'s strata, in order, are the units `[0, |sample|·n)`. -/
theorem positions_second {St : Array Flock.Draw.Stratum} {S n k : ℕ}
    (hmap : St.toList.map (fun s => (s.units, s.k)) = (List.range S).map fun i => (#[(i * n, (i + 1) * n)], k)) :
    St.toList.flatMap (fun s => expand s.units.toList) = List.range (S * n) := by
  have h : St.toList.flatMap (fun s => expand s.units.toList) =
      (St.toList.map (fun s => (s.units, s.k))).flatMap (fun p => expand p.1.toList) := by
    rw [List.flatMap_map]
  rw [h, hmap, List.flatMap_map, ← flatMap_strata]
  congr 1
  funext i
  simp [expand, Nat.succ_mul]

/-- `secondLaw`'s strata are `S` of `n` units, each drawing `k`. -/
theorem sizes_second {St : Array Flock.Draw.Stratum} {S n k : ℕ}
    (hmap : St.toList.map (fun s => (s.units, s.k)) = (List.range S).map fun i => (#[(i * n, (i + 1) * n)], k)) :
    St.toList.map (fun s => (s.size, s.k)) = List.replicate S (n, k) := by
  have h : St.toList.map (fun s => (s.size, s.k)) =
      (St.toList.map (fun s => (s.units, s.k))).map
        (fun p => (p.1.foldl (fun acc (r : ℕ × ℕ) => acc + (r.2 - r.1)) 0, p.2)) := by
    simp only [List.map_map, Function.comp_def]
    rfl
  rw [h, hmap, List.map_map, List.eq_replicate_iff]
  refine ⟨by simp, fun p hp => ?_⟩
  obtain ⟨i, -, rfl⟩ := List.mem_map.1 hp
  simp [Nat.succ_mul]

theorem strataPa_replicate (n k : ℕ) (W : Finset ℕ) : ∀ (S o : ℕ),
    strataPa (List.replicate S (n, k)) o W =
      ∏ i ∈ range S, (((n - (window (o + i * n) n W).card).choose k : ℝ) / (n.choose k : ℝ))
  | 0, o => by simp [strataPa]
  | S + 1, o => by
    rw [List.replicate_succ, strataPa, strataPa_replicate n k W S (o + n), prod_range_succ', mul_comm]
    congr 1
    · refine prod_congr rfl fun i _ => ?_
      rw [show o + n + i * n = o + (i + 1) * n by ring]
    · simp

/-- **The second stage misses `X` no more often than the replay blocks' escapes**: for every replay sample, replay block
size `n`, draw `k`, number `L` of uniform bytes and set `X` of proof blocks, the draw `Select.selectSecond` makes from the
bytes is accepted and proves no proof block of `X` with probability at most `∏_i C(n − |X ∩ [i·n, (i+1)·n)|, k)/C(n, k)`
(`escCell`), the `i`-th replay block of the sample holding `[i·n, (i+1)·n)`. -/
theorem secondMisses (sample : Array ℕ) (n k L : ℕ) (X : Finset ℕ) :
    Claims.Pr (fun src : Fin L → Fin 256 =>
      (Select.selectSecond sample n k (bytesOf src)).isOk = true ∧
        ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample n k (bytesOf src))) ≤
      ∏ i ∈ Finset.range sample.size,
        Claims.Sampling.escCell n k (X.filter fun j => i * n ≤ j ∧ j < (i + 1) * n).card := by
  have hev : ∀ src : Fin L → Fin 256, ((Select.selectSecond sample n k (bytesOf src)).isOk = true ∧
        ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample n k (bytesOf src))) →
      ∃ draw read, Select.selectSecond sample n k (bytesOf src) = .ok (draw, read) ∧
        ∀ j ∈ X, (j : Int) ∉ Draw.proved draw := by
    rintro src ⟨hok, hm⟩
    cases h : Select.selectSecond sample n k (bytesOf src) with
    | error e => rw [h] at hok; exact absurd hok (by simp [Except.isOk, Except.toBool])
    | ok x => rw [h] at hm; exact ⟨x.1, x.2, rfl, hm⟩
  by_cases hex : ∃ src : Fin L → Fin 256, (Select.selectSecond sample n k (bytesOf src)).isOk = true ∧
      ∀ j ∈ X, (j : Int) ∉ secondOf (Select.selectSecond sample n k (bytesOf src))
  swap
  · rw [Claims.Pr_congr fun src => iff_false_intro fun hs => hex ⟨src, hs⟩, Claims.Pr_false]
    exact prod_nonneg fun i _ => by unfold escCell; positivity
  obtain ⟨src0, hs0⟩ := hex
  obtain ⟨draw0, read0, h0, -⟩ := hev src0 hs0
  obtain ⟨-, -, hkn, parsed, -, hp, -, -⟩ := selectSecond_ok h0
  obtain ⟨St, rfl, hmap⟩ := flockLaw_second _ _ _ _ _ hp
  refine le_trans (Claims.Pr_mono fun src hs => ?_)
    (strata_bound (.stratified (sample.size * k) St) St (Or.inl ⟨_, rfl⟩) (sample.size * n)
      (X.filter (· < sample.size * n)) L) |>.trans_eq ?_
  · obtain ⟨draw, read, h, hm⟩ := hev src hs
    obtain ⟨-, -, -, parsed', units, hp', hu, hpr⟩ := selectSecond_ok h
    rw [hp] at hp'
    obtain rfl := Except.ok.inj hp'
    refine ⟨units, read, hu, fun v hv hvu => ?_⟩
    rw [positions_second hmap] at hvu
    have hv' := mem_filter.1 hv
    rw [List.getD_eq_getElem _ _ (by simpa using hv'.2), List.getElem_range] at hvu
    exact hm v hv'.1 (hpr v hvu)
  · rw [sizes_second hmap, strataPa_replicate]
    refine prod_congr rfl fun i hi => ?_
    have hle : i * n + n ≤ sample.size * n := by
      rw [← Nat.succ_mul]; exact Nat.mul_le_mul_right n (mem_range.1 hi)
    rw [Count.card_window, filter_filter]
    unfold escCell
    rw [min_eq_left hkn.le]
    congr 5
    refine filter_congr fun v _ => ?_
    rw [Nat.succ_mul]
    omega

end Properties.SampledProofs.SecurityProofs
