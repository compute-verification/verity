module

public import Proofs.Properties.Verity.SampledTwo.FormDraw
public import Proofs.Properties.Verity.Recursive.BaseSetup
public import Proofs.Flock.SpotFlock.ParamsWf
public import Proofs.Flock.Soundness.ExecStrataOf

/-!
# The member each statement names, from the stage coins

`MemberAtFixed` (`FormStmts.lean`) says two accepted plays with one parameters and one log before the statements'
round name, template by template, one member of the parameters. Each play's evidence names its members itself, so
no name is read off the log. What the two plays share is the units: the model's check of the statements' round holds
each statement's public file's header to the template's part of the second stage's population at the stage coins
(`Spot.SecondStage.populationOk`), and the form's check holds every member's public file to exactly the member's
units, in order (`Spot.OneStageZk.Member.population`). Here `MemberAtFixed` is a theorem where the two plays'
statements pass the model's check at one pair of stage coins (`StageCoinsOne`): no two members of the parameters
have the same units (`MembersApart`, a theorem at any case).

* **The model's check holds the header to the part** (`SpotFlock.population_of_stmtOk`): where round `stage + 1`'s
  check passes, template `t`'s statement decodes, its public file has a header, and the header's `units.indices` are
  the slots of template `t`'s part at the replay sample's size.
* **The form's check holds every member's public file** (`SessionCoins.population_at`): on an accepted play, each
  member of the parameters, sampled or not, has its public file on the blobs, and its population check passes on it
  (`Spot.OneStageZk.runMembers_files`).
* **A member's population check holds the file's units to the member's** (`population_units`), at the header as the
  form reads it (`Spot.FilePart.header`), which is `Flock.norm` of the header the model reads (`header_norm`).
* **The two readings give the member's units** (`units_of_population`): a public file whose header passes the
  model's population check at a part, and whose header as the form reads it holds the units `us`, has `us` the
  part's slots, in order. The header and its `norm` are equivalent (`FlockDeploy.jsonEquiv_norm`), so their
  `units.indices` read the same numbers.
* **`MemberAtFixed`** (`memberAtFixed_of_coins`), from the four, `StageCoinsOne` and `MembersApart`: the two plays'
  members at template `t` have the units of one part, so they are one member.
* **`MembersApart`** (`membersApart`), with no hypothesis: the parameters' reading (`Spot.OneStageZk.membersOf`)
  refuses a member with no range, a range with no unit and a range that starts before another member's ends, so two
  members it returns have no unit in common and a member has one (`FormMember.membersOf_apart`). The reader is
  walked join point by join point (`SpotFlock.DoWalk`).

**Not established here**: `StageCoinsOne`, a hypothesis, which `stageCoinsOne_of_play` (`FormDraw.lean`) gives from
`PlayCoinsOne`.
-/

@[expose] public section

namespace SpotFlock

open Lean (Json)

variable {h : Held}

/-- **Round 2's statement check holds a statement's header to its part's slots** (`second-population`). -/
theorem population_of_codesOk {own : Spot.SecondStage.Own} {t : ℕ} {S : Spot.SecondStage.Stmt}
    (hc : codesOk h own t S = true) :
    Spot.SecondStage.populationOk own.sample.size (partAt h own.sample.size t) S.header = true := by
  unfold codesOk at hc
  obtain ⟨-, -, -, hp, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  exact hp

/-- **Round `stage + 1`'s check holds template `t`'s statement's public file to the template's part**: where it
passes, the statement decodes, its public file has a header, and the header's `units.indices` are the slots of
template `t`'s part of the second stage's population at the replay sample's size, in order
(`Spot.SecondStage.populationOk`). -/
theorem population_of_stmtOk {x m₁ m₂ : Claims.Msg} {c₀ : FlockDeploy.Coin} {c₁ : Stage h}
    (hok : stmtOk h x m₁ m₂ c₀ c₁ = true) {t : ℕ} (ht : t < h.ms.length) :
    ∃ w hd, FlockDeploy.decode (stmtAt h m₂ t) = some w ∧ headerOf w.publicFile = some hd ∧
      Spot.SecondStage.populationOk (h.reg.sample (src c₀)).size (partAt h (h.reg.sample (src c₀)).size t) hd =
        true := by
  obtain ⟨st, rec, draw, w, S, -, -, -, hw, hS, hc, -⟩ := reads_of_stmtOk hok ht
  obtain ⟨-, hH, -, -⟩ := stmtJsonOf_fields h hS
  exact ⟨w, S.header, hw, hH, population_of_codesOk hc⟩

end SpotFlock

namespace Verity.SampledTwo

open Claims
open Lean (Json)

private theorem throw_ne {α : Type} (w : String) (a : α) : (throw w : Flock.V α) ≠ .ok a := fun h => by cases h

/-- **The form reads `norm` of the header the model reads**: where a public file has a header as the model's check
reads it (`SpotFlock.headerOf`, its first line parsed), the form's reading (`Spot.FilePart.header`) is `Flock.norm`
of it. -/
theorem header_norm {f : ByteArray} {hd : Json} (h : SpotFlock.headerOf f = some hd) :
    Spot.FilePart.header f = .ok (Flock.norm hd) := by
  refine Verity.Sampled.header_of_headerOf ?_
  unfold SpotFlock.headerOf at h
  unfold RecDeploy.SetupValues.headerOf
  cases hnl : Flock.findByte f 10 with
  | none => simp [hnl] at h
  | some nl =>
    cases hs : String.fromUTF8? (f.extract 0 nl) with
    | none => simp [hnl, hs] at h
    | some s =>
      cases hp : Flock.JsonParse.parse s with
      | error e => simp [hnl, hs, hp, Except.toOption] at h
      | ok j =>
        simp [hnl, hs, hp, Except.toOption] at h
        subst h
        simp [hs, hp, Except.toOption]

/-- **A member's population check holds its public file's units to the member's**: where it passes on the file `f`,
the form reads `f`'s header and the header's `units.indices` are exactly the member's units, in order
(`Spot.OneStageZk.unitsHeld` at `Member.indices`). -/
theorem population_units {m : Spot.OneStageZk.Member} {slice : Json} {f : ByteArray}
    (h : m.population slice f = .ok ()) :
    ∃ hdr, Spot.FilePart.header f = .ok hdr ∧ Spot.OneStageZk.unitsHeld m.indices hdr = true := by
  unfold Spot.OneStageZk.Member.population at h
  obtain ⟨prog, -, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨strata, -, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨hdr, hh, h⟩ := Verity.Recursive.BaseSetup.bind_ok h
  obtain ⟨⟨⟩, hheld, -⟩ := Verity.Recursive.BaseSetup.bind_ok h
  refine ⟨hdr, hh, ?_⟩
  unfold Spot.OneStageZk.Member.heldAt at hheld
  split at hheld
  · exact absurd hheld (throw_ne _ _)
  · split at hheld
    · assumption
    · exact absurd hheld (throw_ne _ _)

/-- **The units a public file holds, where both checks read it**: a public file whose header `hd` passes the
model's population check at the part `P` of a replay sample of `q` blocks (`Spot.SecondStage.populationOk`), and
whose header as the form reads it (`Flock.norm hd`) holds exactly the units `us` (`Spot.OneStageZk.unitsHeld`), has
`us` the part's `q · n` slots from `lo`, in order. The header is well formed (`SpotFlock.HeaderRoots.wf_of_headerOf`),
so it and its `norm` are equivalent (`FlockDeploy.jsonEquiv_norm`) and their `units.indices` read the same numbers. -/
theorem units_of_population {f : ByteArray} {hd : Json} (hH : SpotFlock.headerOf f = some hd) {q : ℕ}
    {P : Spot.SecondStage.Part} (hpo : Spot.SecondStage.populationOk q P hd = true) {us : Array ℕ}
    (hu : Spot.OneStageZk.unitsHeld us (Flock.norm hd) = true) :
    us.toList = (List.range (q * P.n)).map (P.lo + ·) := by
  have hW := SpotFlock.HeaderRoots.wf_of_headerOf hH
  have hsn := Properties.Challenger.CanonRead.sortedKeys_of_wf _ (FlockDeploy.wf_norm hd)
  have hsr := Properties.Challenger.CanonRead.sortedKeys_of_wf _ hW
  have hEi := Properties.Challenger.CanonRead.getD_equiv
    (Properties.Challenger.CanonRead.sorted_getD hsn "units") (Properties.Challenger.CanonRead.sorted_getD hsr "units")
    (Properties.Challenger.CanonRead.getD_equiv hsn hsr (FlockDeploy.jsonEquiv_norm hd hW) "units") "indices"
  cases hgi : Spot.getD (Spot.getD hd "units") "indices" <;>
    cases hgm : Spot.nat? (Spot.getD hd "instances") <;>
    simp only [Spot.SecondStage.populationOk, hgi, hgm, Bool.and_eq_true, beq_iff_eq, decide_eq_true_eq,
      Bool.false_eq_true] at hpo
  rename_i is m
  obtain ⟨-, hlist⟩ := hpo
  rw [hgi] at hEi
  obtain ⟨a, ha⟩ := FlockDeploy.jsonEquiv_arr hEi.symm
  rw [ha] at hEi
  obtain ⟨hsz, hel⟩ := (FlockDeploy.jsonEquiv_arr_iff a is).1 hEi
  unfold Spot.OneStageZk.unitsHeld at hu
  rw [ha] at hu
  simp only [Bool.and_eq_true, beq_iff_eq, List.all_eq_true, List.mem_range] at hu
  obtain ⟨hus, hall⟩ := hu
  have hlen : is.size = q * P.n := by simpa using congrArg List.length hlist
  refine List.ext_getElem (by simp; omega) fun i h₁ h₂ => ?_
  have hi : i < us.size := by simpa using h₁
  have hia : i < a.size := by omega
  have hii : i < is.size := by omega
  have hiq : i < q * P.n := by omega
  have h3 := hall i hi
  have h4 : Spot.nat? is[i] = some (P.lo + i) := by
    have := congrArg (fun l => l[i]?) hlist
    simpa [hii, hiq] using this
  rw [Array.getElem?_eq_getElem hia, Array.getElem?_eq_getElem hi] at h3
  have h5 : Spot.nat? a[i] = some us[i] := h3
  rw [Properties.Challenger.CanonLine.nat?_equiv (hel i hia hii), h4] at h5
  simp [← Option.some.inj h5]

namespace SessionCoins

/-- **The form's check holds every member's public file, sampled or not**: on a play the form accepts, each member
of the parameters has its public file on the blobs, and its population check (`Spot.OneStageZk.Member.population`)
passes on it at some slice of the registration. The per-member form's run reads every member's files before any
session (`Spot.OneStageZk.runMembers_files`), and a member's statement is `statementOf`'s at its files
(`MemberChecks.statement_ok`). -/
theorem population_at {C : CaseOf} {o : Verity.Exchange} (hacc : FormAccepts C o) {k : Case} (hk : C o = some k)
    {p : VStar.Recursive.SampledTwo.Params} {e : VStar.Recursive.SampledTwo.Evidence}
    (hf : SessionMsgs.formOf k = some (p, e)) {m : Spot.OneStageZk.Member} (hm : m ∈ p.members) :
    ∃ pubB slice, k.blobs.get m.own.pub "public file" = .ok pubB ∧ m.population slice pubB = .ok () := by
  obtain ⟨k', hk', pp, c, e', pub, prof, hdec, hck⟩ := hacc
  obtain rfl : k = k' := Option.some.inj (hk.symm.trans hk')
  have hpe : (pp, e') = (p, e) := Option.some.inj ((SessionMsgs.formOf_of_decoded hdec).symm.trans hf)
  obtain rfl : p = pp := (congrArg Prod.fst hpe).symm
  obtain rfl : e = e' := (congrArg Prod.snd hpe).symm
  obtain ⟨-, c₂, ξ₂, r, i, run, hin, -⟩ := VStar.Recursive.SampledTwo.accepted_steps hck
  obtain ⟨-, -, hrun, -, -⟩ := VStar.Recursive.SampledTwo.inputAt_ok hin
  obtain ⟨held, ts, L, roots, -, -, -, -, -, hrm⟩ := membersRun_ran hrun
  obtain ⟨a, ha, -, -, -, hpop⟩ := Spot.OneStageZk.runMembers_files hrm m hm
  obtain ⟨circuitB, pubB, rd, -, -, hp, -, rfl⟩ := MemberChecks.statement_ok ha
  have hpop' : m.population (m.slice c₂.record roots) pubB = .ok () := hpop
  exact ⟨pubB, _, hp, hpop'⟩

end SessionCoins

/-- **No two members of the parameters have the same units**: two members of the parameters the case's form reads
whose units, each range's in order (`Spot.OneStageZk.Member.indices`), are the same are one member. -/
def MembersApart (C : CaseOf) : Prop :=
  ∀ o k p e, C o = some k → SessionMsgs.formOf k = some (p, e) →
    ∀ m ∈ p.members, ∀ m' ∈ p.members, m.indices = m'.indices → m = m'

/-- **`MemberAtFixed` from the stage coins** (`StageCoinsOne`) and `MembersApart`, where the reading's statements
are the case's: each play's `t`-th statement is its member's at the member's public file on the blobs
(`stmtAt_raw`), the model's check holds both files' headers to template `t`'s part at one replay sample's size
(`SpotFlock.population_of_stmtOk`), and the form's check holds each file's units to its member's
(`SessionCoins.population_at`, `population_units`), so the two members' units are the part's slots
(`units_of_population`) and the two are one member. -/
theorem memberAtFixed_of_coins {A : Live₂} {C : CaseOf} {S : SessionReading} {Good : Verity.Exchange → Prop}
    {L : CoinMap.Locate} (hS : ∀ o, S.stmts o = (C o).bind SessionMsgs.stmtsMsg) (hst : StmtsRead A C S)
    (hcount : StmtsCount A C Good) (hfit : SessionCoins.StmtsFit C Good) (hone : StageCoinsOne A C S Good L)
    (hapart : MembersApart C) : MemberAtFixed A C Good L := by
  intro o o' k k' p e e' hA hG hA' hG' hk hk' hlog hf hf' t ht
  obtain ⟨x₀, hx₀⟩ := stmtAtCase_of_read hS hst hcount hA hG hk hf ht
  obtain ⟨x₀', hx₀'⟩ := stmtAtCase_of_read hS hst hcount hA' hG' hk' hf' ht
  obtain ⟨p₁, e₁, _, fs, record, _, _, hf₁, -, hfs, hrec, -, -, -⟩ := SessionMsgs.stmtAtCase_some hx₀
  obtain ⟨rfl, rfl⟩ : p = p₁ ∧ e = e₁ := Prod.mk.inj (Option.some.inj (hf.symm.trans hf₁))
  obtain ⟨p₁', e₁', _, fs', record', _, _, hf₁', -, hfs', hrec', -, -, -⟩ := SessionMsgs.stmtAtCase_some hx₀'
  obtain ⟨rfl, rfl⟩ : p = p₁' ∧ e' = e₁' := Prod.mk.inj (Option.some.inj (hf'.symm.trans hf₁'))
  obtain ⟨m₂, m, circuitB, pubB, hm₂, hm, hp, hdec, -⟩ :=
    stmtAt_raw hS hst hcount hfit hA hG hk hf ht hfs hrec
  obtain ⟨m₂', m', circuitB', pubB', hm₂', hm', hp', hdec', -⟩ :=
    stmtAt_raw hS hst hcount hfit hA' hG' hk' hf' ht hfs' hrec'
  obtain ⟨x, x', m₁, m₁', c₀, c₁, hok, hok'⟩ :=
    hone o o' k k' p e e' m₂ m₂' hA hG hA' hG' hk hk' hlog hf hf' hm₂ hm₂'
  have key : ∀ {o : Verity.Exchange} {k : Case} {e : VStar.Recursive.SampledTwo.Evidence} {x m₁ m₂ : Msg}
      {m : Spot.OneStageZk.Member} {circuitB pubB : ByteArray} {record : String}, FormAccepts C o → C o = some k →
      SessionMsgs.formOf k = some (p, e) → SessionMsgs.memberAt p e t = some m →
      k.blobs.get m.own.pub "public file" = .ok pubB →
      FlockDeploy.decode (SpotFlock.stmtAt A.h m₂ t) =
        some (SessionMsgs.stmtRaw m circuitB pubB (Spot.OneStageZk.tablesOf k.blobs) (SessionMsgs.drawBytes record)
          (m.own.registered.map fun j => (Flock.RecordText.printT j).toUTF8)) →
      SpotFlock.stmtOk A.h x m₁ m₂ c₀ c₁ = true →
      m.indices.toList =
        (List.range ((A.h.reg.sample (SpotFlock.src c₀)).size *
          (SpotFlock.partAt A.h (A.h.reg.sample (SpotFlock.src c₀)).size t).n)).map
          ((SpotFlock.partAt A.h (A.h.reg.sample (SpotFlock.src c₀)).size t).lo + ·) := by
    intro o k e x m₁ m₂ m circuitB pubB record hA hk hf hm hp hdec hok
    obtain ⟨w, hd, hw, hH, hpo⟩ := SpotFlock.population_of_stmtOk hok ht
    obtain rfl := Option.some.inj (hw.symm.trans hdec)
    have hH₁ : SpotFlock.headerOf pubB = some hd := hH
    obtain ⟨pubB₂, slice, hp₂, hpop⟩ := SessionCoins.population_at hA hk hf (memberAt_mem hm)
    obtain rfl : pubB₂ = pubB := Except.ok.inj (hp₂.symm.trans hp)
    obtain ⟨hdr, hh, hu⟩ := population_units hpop
    obtain rfl : hdr = Flock.norm hd := Except.ok.inj (hh.symm.trans (header_norm hH₁))
    exact units_of_population hH₁ hpo hu
  have hi := key hA hk hf hm hp hdec hok
  have hi' := key hA' hk' hf' hm' hp' hdec' hok'
  rw [hm, hm']
  exact congrArg some
    (hapart o k p e hk hf m (memberAt_mem hm) m' (memberAt_mem hm') (Array.toList_inj.1 (hi.trans hi'.symm)))

/-! ## No two members of the parameters have the same units -/

namespace FormMember

open Lean (Json)

/-- Two ranges `[lo, hi)` with no unit in both: one ends at or before the other's start. -/
def Apart (x y : ℕ × ℕ) : Prop := x.2 ≤ y.1 ∨ y.2 ≤ x.1

/-- Two different members of a list pairwise in a symmetric relation are in it. -/
theorem pairwise_ne {α : Type} {R : α → α → Prop} (hs : ∀ a b, R a b → R b a) :
    ∀ {l : List α}, l.Pairwise R → ∀ a ∈ l, ∀ b ∈ l, a ≠ b → R a b
  | [], _, a, ha, _, _, _ => by cases ha
  | x :: l, h, a, ha, b, hb, hne => by
    rw [List.pairwise_cons] at h
    rcases List.mem_cons.1 ha with rfl | ha' <;> rcases List.mem_cons.1 hb with rfl | hb'
    · exact absurd rfl hne
    · exact h.1 b hb'
    · exact hs _ _ (h.1 a ha')
    · exact pairwise_ne hs h.2 a ha' b hb' hne

/-- The members' ranges, appended member by member, are each member's ranges in the members' order. -/
theorem units_toList (l : List Spot.OneStageZk.Member) (init : Array (ℕ × ℕ)) :
    (l.foldl (fun acc m => acc ++ m.units) init).toList = init.toList ++ l.flatMap (·.units.toList) := by
  induction l generalizing init with
  | nil => simp
  | cons m l ih => rw [List.foldl_cons, ih]; simp [List.flatMap_cons, List.append_assoc]

/-- Each of ascending ranges holds a unit. -/
theorem ascending_lt : ∀ (at_ : ℕ) (rs : List (ℕ × ℕ)), Spot.OneStageZk.ascending at_ rs = true →
    ∀ r ∈ rs, r.1 < r.2
  | _, [], _, r, hr => by cases hr
  | at_, (lo, hi) :: rs, h, r, hr => by
    simp only [Spot.OneStageZk.ascending, Bool.and_eq_true, decide_eq_true_eq] at h
    rcases List.mem_cons.1 hr with rfl | hr'
    · exact h.1.2
    · exact ascending_lt hi rs h.2 r hr'

/-- **Ranges that each hold a unit and each start at or after the end of the one before are pairwise apart.** -/
theorem apart_of_adjacent {rs : Array (ℕ × ℕ)} (hlt : ∀ r ∈ rs, r.1 < r.2)
    (hadj : ∀ i, 1 ≤ i → i < rs.size → rs[i - 1]!.2 ≤ rs[i]!.1) : rs.toList.Pairwise Apart := by
  rw [List.pairwise_iff_getElem]
  intro i j hi hj hij
  have hj' : j < rs.size := by simpa using hj
  have key : ∀ d, ∀ (h : i + 1 + d < rs.size), rs[i].2 ≤ rs[i + 1 + d].1 := by
    intro d
    induction d with
    | zero =>
      intro h
      have h1 := hadj (i + 1) (by omega) h
      rw [getElem!_pos rs (i + 1) h, getElem!_pos rs (i + 1 - 1) (by omega)] at h1
      exact h1
    | succ d ih =>
      intro h
      have h1 := ih (by omega)
      have h2 := hadj (i + 1 + (d + 1)) (by omega) h
      rw [getElem!_pos rs (i + 1 + (d + 1)) h, getElem!_pos rs (i + 1 + (d + 1) - 1) (by omega)] at h2
      have h3 := hlt rs[i + 1 + d] (Array.getElem_mem _)
      have e : rs[i + 1 + (d + 1) - 1]'(by omega) = rs[i + 1 + d]'(by omega) := by congr 1
      rw [e] at h2
      omega
  left
  have h := key (j - (i + 1)) (by omega)
  have e : rs[i + 1 + (j - (i + 1))]'(by omega) = rs[j] := by congr 1; omega
  rw [e] at h
  simpa using h

/-- **A member's units are the units of its ranges.** -/
theorem mem_indices {m : Spot.OneStageZk.Member} {u : ℕ} :
    u ∈ m.indices ↔ ∃ r ∈ m.units, r.1 ≤ u ∧ u < r.2 := by
  unfold Spot.OneStageZk.Member.indices
  simp only [Array.mem_flatMap, Array.mem_map, Array.mem_range]
  constructor
  · rintro ⟨r, hr, i, hi, rfl⟩
    exact ⟨r, hr, by omega, by omega⟩
  · rintro ⟨r, hr, h1, h2⟩
    exact ⟨r, hr, u - r.1, by omega, by omega⟩

theorem list_mapM_some_length {α β : Type} {f : α → Option β} :
    ∀ {l : List α} {r : List β}, l.mapM f = some r → r.length = l.length
  | [], r, h => by
    simp at h
    subst h
    rfl
  | a :: l, r, h => by
    rw [List.mapM_cons] at h
    cases hfa : f a with
    | none => simp [hfa] at h
    | some b =>
      cases hl : l.mapM f with
      | none => simp [hfa, hl] at h
      | some bs =>
        simp [hfa, hl] at h
        subst h
        simp [list_mapM_some_length hl]

/-- An array mapped in `Option` keeps its size. -/
theorem array_mapM_some_size {α β : Type} {f : α → Option β} {xs : Array α} {r : Array β}
    (h : xs.mapM f = some r) : r.size = xs.size := by
  rw [Array.mapM_eq_mapM_toList] at h
  cases hl : xs.toList.mapM f with
  | none => simp [hl] at h
  | some l =>
    simp [hl] at h
    subst h
    simpa using list_mapM_some_length hl

/-- The ranges `Spot.Draw.ranges?` reads are not empty. -/
theorem ranges_size {j : Json} {rs : Array (Int × Int)} (h : Spot.Draw.ranges? j = some rs) : 0 < rs.size := by
  unfold Spot.Draw.ranges? at h
  split at h
  · split at h
    · cases h
    · rename_i a hne
      have := array_mapM_some_size h
      have : 0 < a.size := by
        rcases Nat.eq_zero_or_pos a.size with h0 | h0
        · exact absurd (by simpa using h0) hne
        · exact h0
      omega
  · cases h

/-- **A member `membersOf` returns has a range, and each of its ranges holds a unit**: its units read as a nonempty
list of ranges (`Spot.Draw.ranges?`), ascending (`Spot.OneStageZk.ascending`). -/
theorem membersOf_units {shared : Spot.OneStageZk.Own} {j : Json} {ms : Array Spot.OneStageZk.Member}
    (h : Spot.OneStageZk.membersOf shared j = .ok ms) :
    ∀ m ∈ ms, 0 < m.units.size ∧ ∀ r ∈ m.units, r.1 < r.2 := by
  unfold Spot.OneStageZk.membersOf at h
  do_walk h to ms
  all_goals
    cases h
    intro m hm
    obtain ⟨⟨name, mj⟩, hx, hf⟩ := SpotFlock.ParamsWf.array_mapM_mem (by assumption) m hm
    dsimp -zeta only at hf
    jp_step hf to m with (fun _ => True) by trivial
    have h' := Properties.Spot.SecurityProofs.bind_eq_ok.1 hf
    clear hf
    obtain ⟨_, _, hf⟩ := h'
    have h' := Properties.Spot.SecurityProofs.bind_eq_ok.1 hf
    clear hf
    obtain ⟨_, _, hf⟩ := h'
    split at hf
    · rename_i units hu
      jp_step hf to m with (fun _ => Spot.OneStageZk.ascending 0 units.toList = true) by
        first
          | (rename_i hasc; simp only [Bool.not_eq_true', Bool.not_eq_false] at hasc; exact hasc)
          | (rename_i hthrow; cases hthrow)
      have hasc : Spot.OneStageZk.ascending 0 units.toList = true := by assumption
      do_walk hf to m
      all_goals
        cases hf
        obtain ⟨rs, hrs, hmap⟩ := Option.bind_eq_some_iff.1 hu
        refine ⟨?_, fun r hr => ascending_lt 0 _ hasc r (Array.mem_toList_iff.2 hr)⟩
        have h1 := ranges_size hrs
        have h2 := array_mapM_some_size hmap
        show 0 < units.size
        omega
    · cases hf

/-- **No two members `membersOf` returns have the same units**: two with the same units, each range's in order
(`Spot.OneStageZk.Member.indices`), are one member. Each member has a range and each range holds a unit
(`membersOf_units`). The reading sorts every member's ranges by their starts and refuses one that starts before
the one before it ends, so the sorted ranges are pairwise apart (`apart_of_adjacent`), and so are the members'
ranges in the members' order, a permutation of them (`FlockSoundness.Audit.StrataExec.qsort_perm`): two different
members have no unit in common, and a member has one. -/
theorem membersOf_apart {shared : Spot.OneStageZk.Own} {j : Json} {ms : Array Spot.OneStageZk.Member}
    (h : Spot.OneStageZk.membersOf shared j = .ok ms) :
    ∀ m ∈ ms, ∀ m' ∈ ms, m.indices = m'.indices → m = m' := by
  have hun := membersOf_units h
  unfold Spot.OneStageZk.membersOf at h
  do_walk h to ms
  all_goals
    cases h
    have hloop := ‹forIn _ _ _ = Except.ok _›
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hloop
    have hstep := Verity.Recursive.BaseSetup.forIn_list_mem (fun a s hs => by
      split at hs
      · cases hs
      · cases hs; rfl) hloop
    generalize hU : Array.foldl (fun acc m => acc ++ m.units) #[] ms = U at hstep
    have hperm := FlockSoundness.Audit.StrataExec.qsort_perm (fun x1 x2 : ℕ × ℕ => decide (x1.fst < x2.fst)) U
    have hUl : U.toList = ms.toList.flatMap (·.units.toList) := by
      rw [← hU, ← Array.foldl_toList, units_toList]
      rfl
    have hlt : ∀ r ∈ U.qsort (fun x1 x2 => decide (x1.fst < x2.fst)), r.1 < r.2 := by
      intro r hr
      have hr' : r ∈ ms.toList.flatMap (·.units.toList) := by
        rw [← hUl]
        exact hperm.mem_iff.1 (Array.mem_toList_iff.2 hr)
      obtain ⟨m, hm, hrm⟩ := List.mem_flatMap.1 hr'
      exact (hun m (Array.mem_toList_iff.1 hm)).2 r (Array.mem_toList_iff.1 hrm)
    have hadj : ∀ i, 1 ≤ i → i < (U.qsort (fun x1 x2 => decide (x1.fst < x2.fst))).size →
        (U.qsort (fun x1 x2 => decide (x1.fst < x2.fst)))[i - 1]!.2 ≤
          (U.qsort (fun x1 x2 => decide (x1.fst < x2.fst)))[i]!.1 := by
      intro i h1 h2
      have hi := hstep i (List.mem_range'_1.2 ⟨h1, by simp only [Std.Legacy.Range.size]; omega⟩)
      split at hi
      · cases hi
      · omega
    have hap : (ms.toList.flatMap (·.units.toList)).Pairwise Apart := by
      rw [← hUl]
      exact (List.Perm.pairwise_iff (fun {x y} h => Or.symm h) hperm).1 (apart_of_adjacent hlt hadj)
    have hms := (List.pairwise_flatMap.1 hap).2
    intro m hm m' hm' hi
    refine Classical.byContradiction fun hne => ?_
    have hq := pairwise_ne (fun a b hab x hx y hy => Or.symm (hab y hy x hx)) hms m (Array.mem_toList_iff.2 hm) m'
      (Array.mem_toList_iff.2 hm') hne
    obtain ⟨h0, hl⟩ := hun m hm
    have hx := hl m.units[0] (Array.getElem_mem _)
    have hu : m.units[0].1 ∈ m'.indices := by
      rw [← hi]
      exact mem_indices.2 ⟨_, Array.getElem_mem _, Nat.le_refl _, hx⟩
    obtain ⟨y, hy, hy1, hy2⟩ := mem_indices.1 hu
    rcases hq m.units[0] (Array.mem_toList_iff.2 (Array.getElem_mem _)) y (Array.mem_toList_iff.2 hy) with h | h
    · omega
    · omega

/-- No two members of the per-member form's parameters have the same units (`membersOf_apart`). -/
theorem membersParams_apart {J : Json} {a : Spot.OneStageZk.Params}
    (h : Spot.OneStageZk.membersParamsOfJson J = .ok a) :
    ∀ ms, a.members = some ms → ∀ m ∈ ms, ∀ m' ∈ ms, m.indices = m'.indices → m = m' := by
  unfold Spot.OneStageZk.membersParamsOfJson at h
  do_walk h to a
  all_goals
    cases h
    intro ms hms
    obtain rfl := Option.some.inj hms
    exact membersOf_apart (by assumption)

/-- No two members of the form's parameters have the same units (`membersParams_apart`). -/
theorem members_apart {j : Json} {p : VStar.Recursive.SampledTwo.Params}
    (h : VStar.Recursive.SampledTwo.paramsOfJson j = .ok p) :
    ∀ m ∈ p.members, ∀ m' ∈ p.members, m.indices = m'.indices → m = m' := by
  replace h := (VStar.Recursive.SampledTwo.paramsOfJson_ok h).1
  unfold VStar.Recursive.SampledTwo.paramsParsed at h
  do_walk h to p
  all_goals
    cases h
    exact membersParams_apart (by assumption) _ (by assumption)

end FormMember

/-- **`MembersApart`**, at any case: the form's parameters of a case are read from the tree `Core.canonicalJson`
returns of its parameters' bytes (`FormPrint.formOf_params`), and the reading returns no two members with the same
units (`FormMember.members_apart`). -/
theorem membersApart (C : CaseOf) : MembersApart C := by
  intro o k p e _ hf
  obtain ⟨j, -, hp⟩ := FormPrint.formOf_params hf
  exact FormMember.members_apart hp

end Verity.SampledTwo
