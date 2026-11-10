module

public import Proofs.Properties.Spot
public import Proofs.Properties.Spot.Units
public import Proofs.Core.QSort
public import Proofs.Properties.Pous.MemoryChallenger.Sort
public import Proofs.Properties.Accounting.Remote.Sort
public import Proofs.Flock.Bridge.V5.Accepted
public import Proofs.Flock.Claims.Complete.Json.Stmt

/-!
# v2 credits its one session with the coins' draw

Where one-stage v2 (`Spot.OneStageZk.checkBlobsAt`) accepts, at the coins `ξ`, evidence it parsed
(`Spot.OneStageZk.evidenceOf`), the units its one session proves (`Spot.OneStageZk.provedUnits` on the
session's record) are the units `ξ` draws under the verifier's law as derived, over the registered population
(`Spot.OneStage.drawnFrom`), ascending (`credited_of_checkBlobsAt`); at the public file's load and the record's
draw, `Units.unitsAt` reads them (`unitsAt_of_checkBlobsAt`).

* **v1 at the coins** (`provedUnits_of_checkAt`): v1's `checkAt` refuses a draw other than the coins' (`coinsWhy`), and
  its `check` holds the evidence's `proved_units` to the draw's units and closure (`acceptMeans`); a draw without a
  closure proves its units, which are nonnegative integers, ascending;
* **parsed evidence has no closure** (`closure_of_evidenceOf`): v1's `evidenceOf`, which v2's runs, refuses one;
* **v2's one session** proves units the run holds to the evidence's `proved_units` (`provedHeld`); `provedUnits` sorts
  them already (`provedUnits_sorted`), and sorting sorted naturals leaves them (`qsort_qsort`).

The population `n` is the public file's (`population_of_checkBlobsAt`): v2 holds the registration to the public file
(`publicHeld`), whose last check is the registration's `population` against the header's `instances`
(`population_of_matchesPublic`), and the file's load keeps `instances` as its count of instances (`instances_of_loadPublic`).
Its units are that population's in order (`units_of_checkBlobsAt`): v2 holds the `units.indices` of the header the load
keeps to `0, …, n − 1` (`publicHeld`'s `indicesHeld`).
-/

@[expose] public section

open Lean (Json)
open Spot

namespace Properties.Spot.Credited

private theorem bind_ok {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {b : β} (h : (x >>= f) = .ok b) :
    ∃ a, x = .ok a ∧ f a = .ok b := by
  cases x with
  | error e => simp [bind, Except.bind] at h
  | ok a => exact ⟨a, rfl, h⟩

private theorem throw_bind {ε α β : Type} (e : ε) (f : α → Except ε β) :
    (throw e >>= f : Except ε β) = .error e := rfl

/-! ## Sorting naturals -/

/-- Sorting naturals twice is sorting them once. -/
theorem qsort_qsort (as : Array Nat) : (as.qsort (· < ·)).qsort (· < ·) = as.qsort (· < ·) := by
  have hs : ∀ bs : Array Nat, (bs.qsort (· < ·)).toList.Pairwise (· ≤ ·) := fun bs =>
    Verity.QSort.ByKey.qsort_sorted id bs
  exact Array.toList_inj.1 (List.Perm.eq_of_pairwise (fun _ _ _ _ h1 h2 => Nat.le_antisymm h1 h2) (hs _) (hs _)
    (Properties.RemoteAudit.qsort_toList_perm _ _))

/-- Sorting naturals as integers is sorting them. -/
theorem qsort_cast (as : Array Nat) :
    (as.map fun n : Nat => (n : Int)).qsort (· < ·) = (as.qsort (· < ·)).map fun n : Nat => (n : Int) := by
  rw [MemoryChallenger.Properties.SecurityProofs.Sort.qsort_map]
  congr 2
  funext _ _
  simp

/-! ## A draw's units -/

theorem filterMap_int : ∀ {l : List Json} {u : List Int}, l.mapM int? = some u → l.filterMap int? = u
  | [], u, h => by
    simp only [List.mapM_nil] at h
    exact Option.some.inj h
  | a :: l, u, h => by
    simp only [List.mapM_cons] at h
    cases ha : int? a with
    | none => simp [ha] at h
    | some x =>
      cases hl : l.mapM int? with
      | none => simp [ha, hl] at h
      | some xs =>
        simp only [ha, hl, Option.bind_eq_bind, Option.bind_some] at h
        obtain rfl := Option.some.inj h
        simp only [List.filterMap_cons, ha, filterMap_int hl]

theorem filterMap_nat : ∀ {l : List Json} {u : List Int}, l.mapM int? = some u → (∀ i ∈ u, 0 ≤ i) →
    (l.filterMap nat?).map (fun n : Nat => (n : Int)) = u
  | [], u, h, _ => by
    simp only [List.mapM_nil] at h
    simp only [List.filterMap_nil, List.map_nil]
    exact Option.some.inj h
  | a :: l, u, h, h0 => by
    simp only [List.mapM_cons] at h
    cases ha : int? a with
    | none => simp [ha] at h
    | some x =>
      cases hl : l.mapM int? with
      | none => simp [ha, hl] at h
      | some xs =>
        simp only [ha, hl, Option.bind_eq_bind, Option.bind_some] at h
        obtain rfl := Option.some.inj h
        have hx : 0 ≤ x := h0 x (List.mem_cons_self ..)
        have hn : nat? a = some x.toNat := by simp only [nat?, ha, hx, ↓reduceIte]
        simp only [List.filterMap_cons, hn, List.map_cons, Int.toNat_of_nonneg hx,
          filterMap_nat hl (fun i hi => h0 i (List.mem_cons_of_mem _ hi))]

theorem toList_of_mapM {us : Array Json} {xs : Array Int} (h : us.mapM int? = some xs) :
    us.toList.mapM int? = some xs.toList := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : us.toList.mapM int? with
  | none => rw [hl] at h; cases h
  | some l =>
    rw [hl] at h
    obtain rfl := Option.some.inj h
    rfl

/-- Integers read with `int?` throughout are read by `filterMap int?`. -/
theorem filterMap_int_arr {us : Array Json} {xs : Array Int} (h : us.mapM int? = some xs) :
    us.filterMap int? = xs :=
  Array.toList_inj.1 (by rw [Array.toList_filterMap]; exact filterMap_int (toList_of_mapM h))

/-- Nonnegative integers read with `int?` throughout are read, as naturals, by `filterMap nat?`. -/
theorem filterMap_nat_arr {us : Array Json} {xs : Array Int} (h : us.mapM int? = some xs)
    (h0 : ∀ i ∈ xs.toList, 0 ≤ i) : (us.filterMap nat?).map (fun n : Nat => (n : Int)) = xs :=
  Array.toList_inj.1 (by rw [Array.toList_map, Array.toList_filterMap]; exact filterMap_nat (toList_of_mapM h) h0)

/-- Strictly ascending integers from a nonnegative first are nonnegative. -/
theorem nonneg_of_ascending {xs : Array Int} (ha : Ascending xs) (h0 : ∀ (h : 0 < xs.size), 0 ≤ xs[0]) :
    ∀ i ∈ xs.toList, 0 ≤ i := by
  have key : ∀ k (hk : k < xs.size), 0 ≤ xs[k] := by
    intro k
    induction k with
    | zero => exact h0
    | succ k ih => intro hk; exact Int.le_of_lt (Int.lt_of_le_of_lt (ih (by omega)) (ha k hk))
  intro i hi
  obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hi
  simp only [Array.getElem_toList]
  exact key k (by simpa using hk)

/-! ## v1 at the coins -/

/-- **v1 at the coins `ξ`, on a draw without a closure.** Where `checkAt` accepts, the verifier's law derives (`law`), the
record names a population `n`, the coins draw `units` under `law` over `n` (`drawnFrom`), and the evidence's
`proved_units` are those units, ascending. -/
theorem provedUnits_of_checkAt {p : OneStage.Params} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : OneStage.Evidence} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : OneStage.checkAt p c (some ξ) e = .accept pub prof) (hcl : field? e.draw "closure" = none) :
    ∃ law n units read,
      Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable = .ok law ∧
      nat? (getD c.record "population") = some n ∧ OneStage.drawnFrom p.verifier law n ξ = .ok (units, read) ∧
      e.provedUnits = units.qsort (· < ·) := by
  cases hw : OneStage.coinsWhy p c e ξ with
  | some why =>
    simp only [OneStage.checkAt, Option.bind_some, hw] at h
    cases h
  | none =>
    simp only [OneStage.checkAt, Option.bind_some, hw] at h
    obtain ⟨law, -, n, hreg, hdrawn, hunits, -, -, hn, -, -⟩ := acceptMeans p c e pub prof h
    have hder : Draw.derive p.verifier.program p.verifier.query p.verifier.law p.verifier.workTable = .ok law :=
      hreg.1.2.2.2.1
    obtain ⟨-, -, -, -, -, us, xs, hus, hxs, hasc, h0, -⟩ := hdrawn
    simp only [OneStage.coinsWhy, hder] at hw
    unfold OneStage.coinsReason at hw
    rw [hn] at hw
    dsimp only at hw
    cases hd : OneStage.drawnFrom p.verifier law n ξ with
    | error why => rw [hd] at hw; cases hw
    | ok r =>
      obtain ⟨units, read⟩ := r
      rw [hd] at hw
      dsimp only at hw
      rw [hus] at hw
      dsimp only at hw
      split at hw
      · cases hw
      · rename_i hne
        have hu : units = us.filterMap nat? := by simpa only [bne_iff_ne, ne_eq, not_not] using hne
        refine ⟨law, n, units, read, hder, hn, hd, ?_⟩
        have hp : Draw.proved e.draw = xs.qsort (· < ·) := by
          have hgc : getD e.draw "closure" = .null := by simp only [getD, hcl, Option.getD_none]
          unfold Draw.proved
          dsimp only
          rw [hus, hgc]
          dsimp only
          rw [Array.append_empty, filterMap_int_arr hxs]
        have hcast := hunits.2.1
        rw [hp, ← filterMap_nat_arr hxs (nonneg_of_ascending hasc fun h => (h0 h).1), ← hu, qsort_cast] at hcast
        exact (Array.map_inj_right fun _ _ h => Int.ofNat_inj.1 h).1 hcast

/-! ## v2 -/

/-- **Parsed evidence has no closure**: v2's `evidenceOf` runs v1's, which refuses a draw that carries one. -/
theorem closure_of_evidenceOf {b : ByteArray} {e : OneStageZk.Evidence} (h : OneStageZk.evidenceOf b = .ok e) :
    field? e.base.draw "closure" = none := by
  unfold OneStageZk.evidenceOf at h
  obtain ⟨j, -, h⟩ := bind_ok h
  simp only [throw_bind] at h
  cases j <;> simp only [Bool.not_true, Bool.not_false, Bool.false_eq_true, ↓reduceIte] at h
  all_goals try cases h
  split at h
  · cases h
  · split at h
    · obtain ⟨_, -, h⟩ := bind_ok h
      obtain ⟨base, hb, h⟩ := bind_ok h
      have hbe : e.base = base := by
        try (split at h <;> try (cases h; done))
        obtain ⟨_, -, h⟩ := bind_ok h
        split at h
        · cases h
        · obtain rfl := Except.ok.inj h
          rfl
      rw [hbe]
      unfold OneStage.evidenceOf at hb
      obtain ⟨e', -, hb⟩ := bind_ok hb
      cases hc : field? e'.draw "closure" with
      | some _ =>
        simp only [hc, Option.isSome_some, ↓reduceIte, throw_bind] at hb
        cases hb
      | none =>
        simp only [hc, Option.isSome_none, Bool.false_eq_true, ↓reduceIte] at hb
        obtain rfl := Except.ok.inj hb
        exact hc
    · cases h

/-- `provedUnits` sorts the units it reads. -/
theorem provedUnits_sorted {pubB : ByteArray} {record : String} {us : Array Nat}
    (h : OneStageZk.provedUnits pubB record = .ok us) : ∃ xs : Array Nat, us = xs.qsort (· < ·) := by
  unfold OneStageZk.provedUnits at h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  obtain ⟨_, -, h⟩ := bind_ok h
  split at h
  · obtain ⟨xs, -, h⟩ := bind_ok h
    exact ⟨xs, (Except.ok.inj h).symm⟩
  · cases h

/-- **v2 credits its one session with the coins' draw.** Where v2 accepts evidence it parsed at the coins `ξ`: the
evidence names one session `s`, whose record is on the blobs as UTF-8 (`record`), and the parameters' public file is on
the blobs (`pubB`); the verifier's law derives (`law`) and the record names a population `n`; and the units the session
proves (`provedUnits`) are the units `ξ` draws under `law` over `n` (`drawnFrom`), ascending. -/
theorem credited_of_checkBlobsAt {b : ByteArray} {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : OneStageZk.Evidence} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (he : OneStageZk.evidenceOf b = .ok e) (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ s pubB B record law n units read, e.sessions = #[s] ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      bs.get s.record s!"session {s.name}'s record" = .ok B ∧ String.fromUTF8? B = some record ∧
      Draw.derive p.base.verifier.program p.base.verifier.query p.base.verifier.law p.base.verifier.workTable =
        .ok law ∧
      nat? (getD c.record "population") = some n ∧ OneStage.drawnFrom p.base.verifier law n ξ = .ok (units, read) ∧
      OneStageZk.provedUnits pubB record = .ok (units.qsort (· < ·)) := by
  unfold OneStageZk.checkBlobsAt at h
  dsimp only at h
  split at h
  case h_1 => cases h
  rename_i R hR
  obtain ⟨pb, pf, hca, hall⟩ : ∃ pb pf,
      OneStage.checkAt p.base c (some ξ) { e.base with verdicts := R.out } = .accept pb pf ∧ R.all = true := by
    cases hca : OneStage.checkAt p.base c (some ξ) { e.base with verdicts := R.out } with
    | accept pb pf =>
      simp only [hca] at h
      cases hb : R.all
      · simp only [hb, Bool.false_eq_true, ↓reduceIte] at h
        cases h
      · exact ⟨pb, pf, rfl, rfl⟩
    | _ =>
      simp only [hca] at h
      cases h
  obtain ⟨law, n, units, read, hder, hn, hd, hpu⟩ := provedUnits_of_checkAt hca (closure_of_evidenceOf he)
  have hpu' : e.base.provedUnits = units.qsort (· < ·) := hpu
  obtain ⟨_, h1, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  have hsz : e.sessions.size = 1 := by
    unfold OneStageZk.oneSession at h1
    by_contra hne
    have hb : ¬ (e.sessions.size == 1) = true := by simpa using hne
    rw [ite_eq_right hb] at h1
    cases h1
  obtain ⟨s, hs⟩ : ∃ s, e.sessions = #[s] := by
    rcases hl : e.sessions with ⟨_ | ⟨s, _ | ⟨_, _⟩⟩⟩
    all_goals rw [hl] at hsz
    all_goals first | (simp at hsz; done) | exact ⟨s, rfl⟩
  unfold OneStageZk.run at hR
  cases ht : Flock.Tags.find? OneStageZk.STATEMENT with
  | none => (simp only [ht] at hR; cases hR)
  | some tags0 =>
  simp only [ht] at hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨cB, -, hR⟩ := bind_ok hR
  obtain ⟨pB, hpB, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨rd, -, hR⟩ := bind_ok hR
  obtain ⟨R', hfold, hR⟩ := bind_ok hR
  have hheld : OneStageZk.provedHeld R'.proved e.base.provedUnits = .ok () ∧ R' = R := by
    split at hR
    · obtain ⟨_, hu, hR⟩ := bind_ok hR
      exact ⟨hu, Except.ok.inj hR⟩
    · rename_i hr
      obtain rfl := Except.ok.inj hR
      exact absurd hall hr
  obtain ⟨hheld, rfl⟩ := hheld
  rw [hs, ← Array.foldlM_toList] at hfold
  simp only [List.foldlM_cons, List.foldlM_nil] at hfold
  obtain ⟨R'', hstep, hpure⟩ := bind_ok hfold
  obtain rfl : R'' = R' := Except.ok.inj hpure
  unfold OneStageZk.step at hstep
  obtain ⟨B, hB, hstep⟩ := bind_ok hstep
  cases hu : String.fromUTF8? B with
  | none => rw [hu] at hstep; cases hstep
  | some record =>
    rw [hu] at hstep
    dsimp only at hstep
    obtain ⟨_, -, hstep⟩ := bind_ok hstep
    try dsimp only at hstep
    unfold OneStageZk.push at hstep
    dsimp only at hstep
    split at hstep
    · obtain rfl := Except.ok.inj hstep
      cases hall
    · split at hstep
      · cases hstep
      · obtain ⟨us, hus, hstep⟩ := bind_ok hstep
        obtain rfl := Except.ok.inj hstep
        have hus' : OneStageZk.provedUnits pB record = .ok us := hus
        have hq : (#[] ++ us).qsort (· < ·) = e.base.provedUnits := by
          unfold OneStageZk.provedHeld at hheld
          dsimp only at hheld
          split at hheld
          · cases hheld
          · rename_i hne
            simpa only [bne_iff_ne, ne_eq, not_not] using hne
        refine ⟨s, pB, B, record, law, n, units, read, hs, hpB, hB, hu, hder, hn, hd, ?_⟩
        obtain ⟨xs, rfl⟩ := provedUnits_sorted hus'
        rw [hus', ← hpu', ← hq, Array.empty_append, qsort_qsort]

/-- **v2's credit, as `Units.unitsAt` reads it**: where v2 accepts evidence it parsed at the coins `ξ`, at any load of
the parameters' public file and the draw the session's record reads (`Flock.Witness.recordDraw`), `unitsAt` is the units
`ξ` draws under the verifier's law as derived over the registered population, ascending. -/
theorem unitsAt_of_checkBlobsAt {b : ByteArray} {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : OneStageZk.Evidence} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (he : OneStageZk.evidenceOf b = .ok e) (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ s pubB B record law n units read, e.sessions = #[s] ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      bs.get s.record s!"session {s.name}'s record" = .ok B ∧ String.fromUTF8? B = some record ∧
      Draw.derive p.base.verifier.program p.base.verifier.query p.base.verifier.law p.base.verifier.workTable =
        .ok law ∧
      nat? (getD c.record "population") = some n ∧ OneStage.drawnFrom p.base.verifier law n ξ = .ok (units, read) ∧
      ∀ (tags : Flock.Tags) (cc : Flock.Circuit) (pi : Flock.HmIn.Ports) (P : Flock.Public) (D : Json),
        Flock.HmIn.loadPublic tags cc pi pubB = .ok P → Flock.Witness.recordDraw record = some D →
          Units.unitsAt P.header D = some (units.qsort (· < ·)) := by
  obtain ⟨s, pubB, B, record, law, n, units, read, hs, hp, hB, hr, hder, hn, hd, hc⟩ :=
    credited_of_checkBlobsAt he h
  exact ⟨s, pubB, B, record, law, n, units, read, hs, hp, hB, hr, hder, hn, hd,
    fun _ _ _ _ _ hl hD => Units.unitsAt_of_provedUnits hl hD hc⟩

/-! ## The registered population -/

theorem bool_of_not_not {b : Bool} (h : ¬ (!b) = true) : b = true := by
  cases b <;> simp_all

/-- The registration's checks against a public file pass only at the file's population: the last is the registration's
`population` against the one the file names. -/
theorem population_of_matchesPublic {rec s r d pop : Json}
    (h : (Record.matchesPublic rec s r d pop).codes = #[]) : jeq pop (getD rec "population") = true := by
  unfold Record.matchesPublic at h
  try dsimp only at h
  repeat' split at h
  all_goals first
    | (simp only [Id.run_pure, Properties.Spot.fail_ne_nil] at h; done)
    | exact bool_of_not_not (by assumption)

/-- The load of a public file keeps its header's `instances` as its count of instances. -/
theorem instances_of_loadPublic {tags : Flock.Tags} {c : Flock.Circuit} {pi : Flock.HmIn.Ports} {f : ByteArray}
    {P : Flock.Public} (hp : Flock.HmIn.loadPublic tags c pi f = .ok P) :
    P.header.getObjVal? "instances" = .ok (.num ⟨Int.ofNat P.n, 0⟩) := by
  rw [← Flock.With.inLoadPublicWith_std] at hp
  have hr := RecDeploy.SetupValues.hmInLoad_row _ tags c pi f P hp
  obtain ⟨_, hd, _, -, he, hn, -⟩ := Flock.Bridge.loadPublic_core _ tags _ f P hr
  obtain ⟨hd', he', -, hh⟩ := RecDeploy.SetupValues.loadPublic_counts _ tags _ f P hr
  rw [he] at he'
  obtain rfl := Option.some.inj he'
  rw [hh]
  exact FlockComplete.JsonRT.nat_ok hn

theorem canon_ofNat (a : ℕ) : Flock.canon (.num ⟨Int.ofNat a, 0⟩) = .ok (toString (Int.ofNat a)) := by
  rw [Flock.canon]
  rfl

theorem toString_ofNat (n : ℕ) : toString (Int.ofNat n) = toString n := by
  rw [Int.toString_eq_repr, Int.repr_eq_ite, Nat.toString_eq_repr]
  simp

theorem toString_inj {n n' : ℕ} (h : toString n = toString n') : n = n' := by
  have e := congrArg String.toList h
  rw [Nat.toString_eq_repr, Nat.toString_eq_repr, Nat.toList_repr, Nat.toList_repr] at e
  rw [← Nat.ofDigitChars_ten_toDigits (n := n), e, Nat.ofDigitChars_ten_toDigits]

theorem eq_of_nat? {y : Json} {b : ℕ} (h : nat? y = some b) : y = .num ⟨Int.ofNat b, 0⟩ := by
  cases y with
  | num n =>
    obtain ⟨m, e⟩ := n
    cases e with
    | succ e => simp [nat?, int?] at h
    | zero =>
      cases m with
      | ofNat k =>
        have hk : k = b := by simpa [nat?, int?] using h
        subst hk
        rfl
      | negSucc k =>
        have hn : ¬ (0 : Int) ≤ Int.negSucc k := Int.not_le.2 (Int.negSucc_lt_zero k)
        simp [nat?, int?, hn] at h
  | _ => simp [nat?, int?] at h

theorem eq_of_jeq {a b : ℕ} (h : jeq (.num ⟨Int.ofNat a, 0⟩) (.num ⟨Int.ofNat b, 0⟩) = true) : a = b := by
  simp only [jeq, canon_ofNat, beq_iff_eq] at h
  rw [toString_ofNat, toString_ofNat] at h
  exact toString_inj h

/-- **v2 holds the record's population to the public file's.** Where v2 accepts at the coins `ξ`, at the parameters'
public file on the blobs (`pubB`) and any load of it (`P`), the record names the population `P.n`, the instances the
file's header counts. -/
theorem population_of_checkBlobsAt {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : OneStageZk.Evidence} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) {pubB : ByteArray}
    (hpB : bs.get p.own.pub "public file" = .ok pubB) {tags : Flock.Tags} {cc : Flock.Circuit}
    {pi : Flock.HmIn.Ports} {P : Flock.Public} (hl : Flock.HmIn.loadPublic tags cc pi pubB = .ok P) :
    nat? (getD c.record "population") = some P.n := by
  unfold OneStageZk.checkBlobsAt at h
  dsimp only at h
  split at h
  case h_1 => cases h
  rename_i R hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, hpop, hR⟩ := bind_ok hR
  obtain ⟨pB, hpB', hR⟩ := bind_ok hR
  obtain ⟨_, hph, -⟩ := bind_ok hR
  obtain rfl : pB = pubB := Except.ok.inj (hpB'.symm.trans hpB)
  obtain ⟨n, hn⟩ : ∃ n, nat? (getD c.record "population") = some n := by
    cases hn : nat? (getD c.record "population") with
    | none =>
      unfold OneStageZk.popHeld at hpop
      rw [hn] at hpop
      cases hpop
    | some n => exact ⟨n, rfl⟩
  have hj : jeq (getD P.header "instances") (getD c.record "population") = true := by
    unfold OneStageZk.publicHeld at hph
    obtain ⟨hd, hhd, hph⟩ := bind_ok hph
    rw [Units.headerOf_loadPublic hl] at hhd
    obtain rfl := Except.ok.inj hhd
    try dsimp only at hph
    split at hph
    · first
        | (cases hph; done)
        | simp [bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at hph
    · rename_i hne
      exact population_of_matchesPublic (Array.isEmpty_iff.1 (bool_of_not_not hne))
  have hi : getD P.header "instances" = .num ⟨Int.ofNat P.n, 0⟩ := by
    unfold getD field?
    rw [instances_of_loadPublic hl]
    try rfl
  rw [hi, eq_of_nat? hn] at hj
  rw [hn, eq_of_jeq hj]

theorem mapM_range {ε : Type} {f : Json → Except ε ℕ} (hf : ∀ j u, nat? j = some u → f j = .ok u) :
    ∀ (l : List Json) (k : ℕ), (∀ i (hi : i < l.length), nat? l[i] = some (k + i)) →
      l.mapM f = .ok (List.range' k l.length)
  | [], _, _ => rfl
  | a :: l, k, h => by
    have ha : f a = .ok k := hf a k (h 0 (by simp))
    have hl := mapM_range hf l (k + 1) fun i hi => by
      have := h (i + 1) (by simpa using hi)
      simpa [Nat.add_assoc, Nat.add_comm 1 i] using this
    simp only [List.mapM_cons, ha, hl, List.length_cons, List.range'_succ]
    rfl

/-- **`indicesHeld` is `FilePart.indicesOf` at the population's units in order**: where the header `h`'s
`units.indices` are the record `rec`'s population units in order, the record names a population `n` and
`FilePart.indicesOf` reads `0, …, n − 1`. -/
theorem indicesOf_of_indicesHeld {rec h : Json} (hi : OneStageZk.indicesHeld rec h = true) :
    ∃ n, nat? (getD rec "population") = some n ∧
      _root_.Spot.FilePart.indicesOf h = .ok (Array.range n) := by
  unfold OneStageZk.indicesHeld at hi
  split at hi
  · rename_i n is hn his
    refine ⟨n, hn, ?_⟩
    simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hi
    obtain ⟨hsz, hall⟩ := hi
    unfold _root_.Spot.FilePart.indicesOf
    simp only [his]
    rw [Array.mapM_eq_mapM_toList, mapM_range (fun j u hj => by simp only [hj]; rfl) is.toList 0 fun i hi => by
      have := hall i (by simpa [← hsz] using hi)
      simpa [Array.getElem?_eq_getElem (by simpa using hi)] using this]
    simp only [Array.length_toList, hsz, ← List.range_eq_range']
    show Except.ok (List.range n).toArray = Except.ok (Array.range n)
    rw [← Array.toList_range]
    all_goals rfl
  · cases hi

/-- **v2 holds the public file's units to the record's population, in order.** Where v2 accepts at the coins `ξ`, at
the parameters' public file on the blobs (`pubB`) and any load of it (`P`), the record names a population `n` and the
`units.indices` of the header the load keeps are `0, …, n − 1` (`FilePart.indicesOf`, through `publicHeld`'s
`indicesHeld`). -/
theorem units_of_checkBlobsAt {p : OneStageZk.Params} {c : OneStage.Commitment} {ξ : ByteArray}
    {e : OneStageZk.Evidence} {bs : Core.Blobs} {pub : OneStage.Public} {prof : Core.Integrity}
    (h : OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) {pubB : ByteArray}
    (hpB : bs.get p.own.pub "public file" = .ok pubB) {tags : Flock.Tags} {cc : Flock.Circuit}
    {pi : Flock.HmIn.Ports} {P : Flock.Public} (hl : Flock.HmIn.loadPublic tags cc pi pubB = .ok P) :
    ∃ n, nat? (getD c.record "population") = some n ∧
      _root_.Spot.FilePart.indicesOf P.header = .ok (Array.range n) := by
  unfold OneStageZk.checkBlobsAt at h
  dsimp only at h
  split at h
  case h_1 => cases h
  rename_i R hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨_, -, hR⟩ := bind_ok hR
  obtain ⟨pB, hpB', hR⟩ := bind_ok hR
  obtain ⟨⟨⟩, hu, -⟩ := bind_ok hR
  obtain rfl : pB = pubB := Except.ok.inj (hpB'.symm.trans hpB)
  obtain ⟨hd, hhd, hi⟩ := OneStageZk.indicesHeld_of_publicHeld hu
  rw [Units.headerOf_loadPublic hl] at hhd
  obtain rfl := Except.ok.inj hhd
  exact indicesOf_of_indicesHeld hi

end Properties.Spot.Credited
