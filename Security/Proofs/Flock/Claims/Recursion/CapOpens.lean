import Proofs.Flock.Claims.Recursion.OpensReads
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.C3.Compose.RsFacts
import Proofs.Flock.C3.Reps.RoundLinks
import Proofs.Flock.VBridge.Pinned
import Proofs.Flock.VBridge.Sparse.Unit

/-!
# RecCap's part holds round 0 as the coin server recorded it

At `layoutBTops`, RecCap's part holds round 0 whole: port `top`'s hm96 slot at each of its `N` instances, in order
(`capHold`). `PartOpensB` at `cmtTr recRead` asks those bytes to be the record's round 0, the coin server's `root_B`
(`Record.msgs`, `link.roots`). This is that half (`cap_cmt`).

* The setup runs `RecCoins.check` at RecCap, which refuses a `rec-cap/…` unit not named as `VStar.capName l N` gives
  (`capParams_of_check`), and so runs `checkCap` at its `(l, N)` (`capCheck_of_check`): the verifier's reads have a port
  `top` reading `rec-top-L<l>` at node `n` in instance `n`, and the table labels its instances `0` to `n - 1` in order
  (`checkCap_ok`).
* `RecVStar.checkTopRows` holds each table row of `top` to `topRow`'s row at its registered position
  (`checkTopRows_rows`), and `RecCoins.checkLink` holds `root_B` to the cap's `N` nodes at level 0
  (`root_of_checkLink`).
* So instance `n`'s slot at `top` reads `topRow lk l n` (`capRow_bytes`), node `n` of `root_B` at level 0, and the `N`
  slots in order are `root_B`'s nodes, the record's round 0 (`staged_reads_link`, `FlockC3.VStarList.checkReads_ok`).
-/

namespace RecDeploy

open Flock
open Claims (Msg)
open FlockDeploy (Coin)
open Lean (Json)

/-! ## RecCap's name -/

theorem capName_of_params {name : String} {l N : ℕ} (h : VStar.capParams name = some (l, N)) :
    VStar.capName l N = name := by
  unfold VStar.capParams at h
  repeat' split at h
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl⟩ := h; exact beq_iff_eq.1 ‹_›)
    | cases h

theorem capName_toList (l N : ℕ) :
    (VStar.capName l N).toList = "rec-cap/n".toList ++ (Nat.toDigits 10 N ++ match l with
      | 0 => []
      | l + 1 => '-' :: 'l' :: Nat.toDigits 10 (l + 1)) := by
  cases l <;> simp [VStar.capName, VStar.capLevelTag, FlockVBridge.toString_str, Nat.toList_repr]

theorem capName_eq (l N : ℕ) : VStar.capName l N = "rec-cap/" ++ s!"n{N}{VStar.capLevelTag l}" := by
  simp [VStar.capName, ← String.toList_inj, FlockVBridge.toString_str]

theorem capName_startsWith (l N : ℕ) : (VStar.capName l N).startsWith "rec-cap/" = true := by
  rw [capName_eq]
  simp [String.startsWith_string_iff]

/-- **The name `rec-cap/n<N>` or `rec-cap/n<N>-l<l>` determines `l` and `N`.** -/
theorem capName_inj {l N l' N' : ℕ} (h : VStar.capName l N = VStar.capName l' N') : l = l' ∧ N = N' := by
  have h := congrArg String.toList h
  rw [capName_toList, capName_toList] at h
  have h := List.append_cancel_left h
  have dig : ∀ n, ∀ x ∈ Nat.toDigits 10 n, x.isDigit := fun n x hx =>
    Nat.isDigit_of_mem_toDigits (by decide) (by decide) hx
  cases l with
  | zero =>
    cases l' with
    | zero => exact ⟨rfl, FlockVBridge.eq_of_toDigits (by simpa using h)⟩
    | succ l' =>
      exfalso
      have hm : '-' ∈ Nat.toDigits 10 N := by simp only [List.append_nil] at h; rw [h]; simp
      exact absurd (dig N _ hm) (by decide)
  | succ l =>
    cases l' with
    | zero =>
      exfalso
      have hm : '-' ∈ Nat.toDigits 10 N' := by simp only [List.append_nil] at h; rw [← h]; simp
      exact absurd (dig N' _ hm) (by decide)
    | succ l' =>
      obtain ⟨h1, h2⟩ := FlockVBridge.digits_dash (dig _) (dig _) h
      simp only [List.cons.injEq, true_and] at h2
      exact ⟨FlockVBridge.eq_of_toDigits h2, FlockVBridge.eq_of_toDigits h1⟩

theorem residualsName_of_params {name : String} {m k c j : ℕ}
    (h : RecResiduals.params name = some (m, k, c, j)) : RecResiduals.unitName m k c j = name := by
  unfold RecResiduals.params at h
  repeat' (first | split at h | dsimp only at h)
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl, rfl⟩ := h; assumption)
    | cases h

theorem capName_ne_open (l N L H : ℕ) : RecOpen.unitName L H ≠ VStar.capName l N := by
  intro h
  have h := congrArg (fun s : String => s.toList.take 5) h
  simp only [RecOpen.unitName, VStar.capName, String.toList_append, List.append_assoc, FlockVBridge.toString_str] at h
  rw [FlockC3.PubRows.take_five (by decide), FlockC3.PubRows.take_five (by decide)] at h
  exact absurd h (by decide)

theorem stackedParams_cap (l N : ℕ) : RecCoins.stackedParams (VStar.capName l N) = none := by
  have hs : (VStar.capName l N).startsWith "rec-open-stacked/" = false := by
    cases hs : (VStar.capName l N).startsWith "rec-open-stacked/" with
    | false => rfl
    | true =>
      obtain ⟨t, ht⟩ := String.startsWith_string_iff.1 hs
      have h := congrArg (List.take 5) ht
      rw [capName_toList, FlockC3.PubRows.take_five (by decide), FlockC3.PubRows.take_five (by decide)] at h
      exact absurd h (by decide)
  unfold RecCoins.stackedParams
  rw [hs]
  rfl

/-- **RecCap's name is no other V\* unit's**: none of `RecCoins.check`'s other readers reads it. -/
theorem params_cap (l N : ℕ) :
    RecOpen.params (VStar.capName l N) = none ∧ RecResiduals.params (VStar.capName l N) = none ∧
      RecSparse.params (VStar.capName l N) = none ∧ RecCoins.stackedParams (VStar.capName l N) = none ∧
      RecOpening.params (VStar.capName l N) = none := by
  refine ⟨?_, ?_, ?_, stackedParams_cap l N, ?_⟩
  · cases hp : RecOpen.params (VStar.capName l N) with
    | none => rfl
    | some p =>
      obtain ⟨L, H⟩ := p
      exact absurd (FlockVBridge.unitName_params hp) (capName_ne_open l N L H)
  · cases hp : RecResiduals.params (VStar.capName l N) with
    | none => rfl
    | some p =>
      obtain ⟨m, k, c, j⟩ := p
      exact absurd (residualsName_of_params hp) ((FlockC3.Reps.resName_ne m k c j).2.2.2 l N)
  · cases hp : RecSparse.params (VStar.capName l N) with
    | none => rfl
    | some sh =>
      exact absurd (FlockVBridge.Sparse.unitName_params hp) ((FlockC3.PubRows.sparse_ne_names sh).2.2.2.2 l N)
  · cases hp : RecOpening.params (VStar.capName l N) with
    | none => rfl
    | some p =>
      obtain ⟨m, c, j⟩ := p
      exact absurd (Flock.Readable.openingName_of_params hp) ((FlockC3.Compose.openingName_ne m c j).2.2.2.2 l N)

/-! ## What `RecCoins.check` and `checkLink` pin at RecCap -/

theorem capCheck_of_check {st : Stmt} {own : Option (Array Registered.Port)} {f : Option RecCoins.Source}
    {c3 : Option RecCoins.C3} (h : RecCoins.check st own f c3 = .ok ()) {l N : ℕ}
    (hc : VStar.capParams st.c.unit.name = some (l, N))
    (ho : RecOpen.params st.c.unit.name = none) (hres : RecResiduals.params st.c.unit.name = none)
    (hsp : RecSparse.params st.c.unit.name = none) (hso : RecCoins.stackedParams st.c.unit.name = none)
    (hop : RecOpening.params st.c.unit.name = none) :
    ∃ own', own = some own' ∧ RecCoins.checkCap st.c st.pub own' l N = .ok () := by
  unfold RecCoins.check at h
  simp only [ho, hres, hsp, hso, hop, hc, Option.isNone_some, Option.isNone_none, Bool.and_false, Bool.false_and,
    Bool.false_eq_true, ↓reduceIte] at h
  cases own with
  | none => cases h
  | some own =>
    cases f with
    | none => cases h
    | some f =>
      refine ⟨own, rfl, ?_⟩
      simp only [bind, Except.bind] at h
      split at h
      · cases h
      · exact h

theorem capNamed_of_check {st : Stmt} {own : Option (Array Registered.Port)} {f : Option RecCoins.Source}
    {c3 : Option RecCoins.C3} (h : RecCoins.check st own f c3 = .ok ())
    (hs : st.c.unit.name.startsWith "rec-cap/" = true)
    (ho : RecOpen.params st.c.unit.name = none) (hres : RecResiduals.params st.c.unit.name = none)
    (hsp : RecSparse.params st.c.unit.name = none) (hso : RecCoins.stackedParams st.c.unit.name = none)
    (hop : RecOpening.params st.c.unit.name = none) : (VStar.capParams st.c.unit.name).isSome := by
  cases hc : VStar.capParams st.c.unit.name with
  | some _ => rfl
  | none =>
    exfalso
    unfold RecCoins.check at h
    simp only [ho, hres, hsp, hso, hop, hc, hs, Option.isNone_none, Bool.not_true, Bool.and_false,
      Bool.false_eq_true, ↓reduceIte] at h
    cases own with
    | none => cases h
    | some own =>
      cases f with
      | none => cases h
      | some f =>
        simp only [bind, Except.bind] at h
        split at h
        · cases h
        · cases h

/-- **A unit `RecCoins.check` accepted under the name `capName l N` gives back `(l, N)`**: the check refuses a
`rec-cap/…` unit `capParams` reads nothing of, and any pair it reads has the same name. -/
theorem capParams_of_check {st : Stmt} {own : Option (Array Registered.Port)} {f : Option RecCoins.Source}
    {c3 : Option RecCoins.C3} (h : RecCoins.check st own f c3 = .ok ()) {l N : ℕ}
    (hn : st.c.unit.name = VStar.capName l N) : VStar.capParams st.c.unit.name = some (l, N) := by
  obtain ⟨ho, hres, hsp, hso, hop⟩ := params_cap l N
  rw [← hn] at ho hres hsp hso hop
  obtain ⟨⟨l', N'⟩, hc⟩ := Option.isSome_iff_exists.1
    (capNamed_of_check h (by rw [hn]; exact capName_startsWith l N) ho hres hsp hso hop)
  obtain ⟨rfl, rfl⟩ := capName_inj ((capName_of_params hc).trans hn)
  exact hc

/-- **`checkCap` accepting**: the verifier's reads have a port `top` reading `rec-top-L<l>` at node `i` in instance
`i`, and the table labels its instances in order. -/
theorem checkCap_ok {c : Circuit} {pub : Public} {own : Array Registered.Port} {l N : ℕ}
    (h : RecCoins.checkCap c pub own l N = .ok ()) :
    ∃ reg ∈ own, reg.name = "top" ∧ reg.value = s!"rec-top-L{l}" ∧ reg.reads = (Array.range N).map some ∧
      Registered.unitIndices pub = .ok (Array.range pub.n) := by
  unfold RecCoins.checkCap at h
  simp only at h
  split at h
  · rename_i reg hreg
    split at h
    · exact absurd h (Registered.throw_ne_ok _ _)
    rename_i hv
    split at h
    · exact absurd h (Registered.throw_ne_ok _ _)
    split at h
    · exact absurd h (Registered.throw_ne_ok _ _)
    rename_i hr
    obtain ⟨idx, hidx, h⟩ := Registered.except_bind_ok h
    split at h
    · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok h
      exact absurd h1 (Registered.throw_ne_ok _ _)
    rename_i hi
    have hname := List.find?_some (l := own.toList) (by simpa using hreg)
    refine ⟨reg, Array.mem_of_find?_eq_some hreg, by simpa using hname, by simpa using hv, by simpa using hr, ?_⟩
    rw [hidx]
    simpa using hi
  · exact absurd h (Registered.throw_ne_ok _ _)

/-- At level 0, `checkLink` holds the coin server's `root_B` to RecCap's `N` nodes. -/
theorem root_of_checkLink {st : Stmt} {own : Option (Array Registered.Port)} {f : Option RecCoins.Source}
    {lk : RecVStar.InnerLink} (h : RecCoins.checkLink st own f (some lk) = .ok ()) {N : ℕ}
    (hc : VStar.capParams st.c.unit.name = some (0, N)) : lk.root.size = N := by
  unfold RecCoins.checkLink at h
  simp only [hc, Option.isNone_some, Bool.and_false, Bool.false_eq_true, ↓reduceIte] at h
  by_cases hne : lk.root.size = N
  · exact hne
  · simp [hne] at h

/-! ## `checkTopRows` -/

theorem topLevel_top (l : ℕ) : RecVStar.topLevel s!"rec-top-L{l}" = some l :=
  FlockC3.Reps.index_pre "rec-top-L" l rfl

/-- **`checkTopRowsAt` accepting**: every row `r` of the port's table is `topRow`'s row at its position. -/
theorem checkTopRowsAt_rows {unit name value : String} {lk : RecVStar.InnerLink} {l : ℕ} {positions : Array ℕ}
    {row : ℕ → ByteArray} (h : RecVStar.checkTopRowsAt unit name value lk l positions row = .ok ()) :
    ∀ r < positions.size, ∃ wh c, RecVStar.topRow lk l (positions.getD r 0) = .ok (wh, c) ∧ row r = c := by
  intro r hr
  unfold RecVStar.checkTopRowsAt at h
  obtain ⟨_, h, -⟩ := Registered.except_bind_ok h
  rw [Std.Legacy.Range.forIn_eq_forIn_range', FlockC3.PubRows.rows_range_size] at h
  refine FlockC3.PubRows.rows_forIn_each
    (fun i => ∃ wh c, RecVStar.topRow lk l (positions.getD i 0) = .ok (wh, c) ∧ row i = c)
    (fun i s hi => ?_) _ h r (List.mem_range'_1.2 ⟨Nat.zero_le _, by simpa using hr⟩)
  have hget : positions[i]! = positions.getD i 0 := by
    simp only [getElem!_def, Array.getD_eq_getD_getElem?]
    cases positions[i]? <;> rfl
  split at hi
  · rename_i wh c hc
    split at hi
    · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hi
      exact absurd h1 (Registered.throw_ne_ok _ _)
    · rename_i hne
      have he : (row i == c) = true := by simpa [bne] using hne
      exact ⟨⟨wh, c, by rw [← hget]; exact hc, ByteArray.ext (beq_iff_eq.1 he)⟩, (Except.ok.inj hi).symm⟩
  · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hi
    exact absurd h1 (Registered.throw_ne_ok _ _)

/-- **`checkTopRows` accepting**: at each port of the reads registering `rec-top-L<l>`, the public file is a table, the
port is the circuit's first of its name, and each table row is `topRow`'s row at the position the port's entry gives
it. -/
theorem checkTopRows_rows {st : Stmt} {own : Array Registered.Port} {lk : RecVStar.InnerLink}
    (h : RecVStar.checkTopRows st (some own) (some lk) = .ok ()) {reg : Registered.Port} (hmem : reg ∈ own)
    {l : ℕ} (hl : RecVStar.topLevel reg.value = some l) :
    ∃ t, st.pub.tables = some t ∧ ∃ p positions paths,
      (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == reg.name) = some p ∧
      Registered.entry st.pub reg = .ok (positions, paths) ∧
      ∀ r < positions.size, ∃ wh c, RecVStar.topRow lk l (positions.getD r 0) = .ok (wh, c) ∧
        Registered.tableRow st.pub t p r = c := by
  have hreg : reg ∈ own.filter (fun reg => (RecVStar.topLevel reg.value).isSome) :=
    Array.mem_filter.2 ⟨hmem, by simp [hl]⟩
  unfold RecVStar.checkTopRows at h
  simp only [Option.getD_some] at h
  split at h
  · split at h
    · rename_i t ht
      obtain ⟨_, h, -⟩ := Registered.except_bind_ok h
      rw [← Array.forIn_toList] at h
      refine ⟨t, ht, FlockC3.PubRows.rows_forIn_each (fun a : Registered.Port => ∀ l,
          RecVStar.topLevel a.value = some l → ∃ p positions paths,
          (List.range st.c.ports.size).find? (fun p => st.c.ports[p]!.name == a.name) = some p ∧
          Registered.entry st.pub a = .ok (positions, paths) ∧
          ∀ r < positions.size, ∃ wh c, RecVStar.topRow lk l (positions.getD r 0) = .ok (wh, c) ∧
            Registered.tableRow st.pub t p r = c)
        (fun a r hb => ?_) _ h reg (Array.mem_toList_iff.2 hreg) l hl⟩
      split at hb
      · rename_i l' hl'
        split at hb
        · rename_i p hp
          obtain ⟨⟨positions, paths⟩, hent, hb⟩ := Registered.except_bind_ok hb
          obtain ⟨_, hrows, hb⟩ := Registered.except_bind_ok hb
          refine ⟨fun l hl => ?_, (Except.ok.inj hb).symm⟩
          rw [hl] at hl'
          cases hl'
          exact ⟨p, positions, paths, hp, hent, checkTopRowsAt_rows hrows⟩
        · obtain ⟨_, h1, -⟩ := Registered.except_bind_ok hb
          exact absurd h1 (Registered.throw_ne_ok _ _)
      · rename_i hnone
        exact ⟨fun l hl => (hnone l hl).elim, (Except.ok.inj hb).symm⟩
    · exact absurd h (Registered.throw_ne_ok _ _)
  · rename_i hnone
    exact absurd (Array.getElem?_eq_getElem (Array.size_pos_of_mem hreg)) (hnone _)

theorem topRow_zero {lk : RecVStar.InnerLink} {n : ℕ} {wh : String} {c : ByteArray}
    (h : RecVStar.topRow lk 0 n = .ok (wh, c)) : lk.root.getD n default = c := by
  unfold RecVStar.topRow at h
  simp only [beq_self_eq_true, ↓reduceIte] at h
  split at h
  · rename_i c' hc'
    cases h
    rw [Array.getD_eq_getD_getElem?, hc', Option.getD_some]
  · cases h

/-! ## RecCap's slots are the coin server's tops -/

theorem findIdx_of_find_range {α : Type} [Inhabited α] {a : Array α} {f : α → Bool} {p : ℕ}
    (hp : (List.range a.size).find? (fun j => f a[j]!) = some p) : a.findIdx? f = some p := by
  rw [List.find?_range_eq_some] at hp
  obtain ⟨hfp, hpm, hlt⟩ := hp
  have hps : p < a.size := List.mem_range.1 hpm
  rw [Array.findIdx?_eq_some_iff_getElem]
  refine ⟨hps, by rwa [getElem!_pos a p hps] at hfp, fun j hj => ?_⟩
  have := hlt j hj
  rw [getElem!_pos a j (by omega)] at this
  simpa using this

/-- **RecCap's slot at instance `n` is row `n` of its level's tops**: at a staged RecCap part `.cap l N` the witness
checks against, port `top`'s hm96 slot at instance `n < N`, read off the witness (`hmCommit`), is `topRow lk l n`, the
coin server's node `n` of `root_B` at level 0 (where `root_B` has the cap's `N` nodes) or of the reps' level-`l` tops.
The slot is its instance's public row (`slotPub_table`); `RecCoins.check` ran `checkCap` at the unit's `(l, N)`
(`capParams_of_check`), so `top` reads `rec-top-L<l>` at node `n` in instance `n`, whose label is `n` (`checkCap_ok`),
and `checkTopRows` pins that row to `topRow lk l n` (`checkTopRows_rows`). -/
theorem capRow_bytes {st : Stage} {s : ℕ} {x xO : Msg} {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {r : XR} {c : Circuit}
    (hr : readX x = some r) (hpc : r.w.parts[s]?.bind partCircuit = some c) {l N : ℕ}
    (hsv : (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap l N)) {i : RecIns}
    {lk : RecVStar.InnerLink} (hi : insO xO = some i) (hlk : i.link = some lk) :
    ∃ q, capPort c = some q ∧ (l = 0 → lk.root.size = N) ∧ ∀ n < N, ∃ wh cc, RecVStar.topRow lk l n = .ok (wh, cc) ∧
      Claims.Recursion.VStarB.readForms w (hmCommit c q n) = cc.toList := by
  obtain ⟨p, r', sv, hx, ht, hpi, hr', ⟨ss, hss, hsv'⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have esv : sv = .cap l N := by
    rw [hss] at hsv
    exact Option.some.inj (hsv'.symm.trans hsv)
  subst esv
  have hS' := hS
  obtain ⟨ms, i', spec, sts, rfl, hsu, -⟩ := hS'
  obtain ⟨hins, hset⟩ := setupO_some hsu
  obtain rfl : i' = i := Option.some.inj (hins.symm.trans hi)
  have hp := Option.some.inj (hx.symm.trans (parse_of_setupO hsu))
  obtain ⟨hsz1, -⟩ := oneFull_staged hS hsu
  have h0' : 0 < sts.size := by omega
  have e0 : p.sts[0] = sts[0] := by subst hp; rfl
  have hsz : p.sts.size = 1 := by rw [hp]; exact hsz1
  have hc0 := partCircuit_staged hS hx ht hpi hr hpc h0
  obtain ⟨L, hz⟩ := layout_of_check_regs hck hx ht hpi h0
  have hz1 : (L.model (FlockSoundness.Refine.regionsOf p.sts[0])).Satisfies (Flock.Readable.zOf 1 w 0 p.sts[0].m) := by
    have e := congrArg (fun J => Flock.Readable.zOf J w 0 p.sts[0].m) hsz
    rw [← e]
    exact hz
  obtain ⟨-, hts⟩ := tablesOf_of_parse hx
  obtain ⟨_, hset0⟩ := Flock.Readable.setup_of_tablesOf hts
  unfold Flock.Readable.Read.setup at hset0
  rw [hpi] at hset0
  obtain ⟨-, hall⟩ := FlockSoundness.Discharge.HiddenExec.setupTables_setupOne hset0
  obtain ⟨d, hd, -⟩ := hall 0 h0
  have ht' : (FlockSoundness.Discharge.Exec.tagsAt p.tags p.I.count 0).typed = false :=
    (FlockSoundness.Discharge.Exec.tagsAt_typed _ _ _).trans ht
  obtain ⟨hscope, -⟩ := staged_scope hS hx h0
  have hv1 := FlockSoundness.Discharge.Layout.scopeOk_ports
    ((FlockSoundness.Discharge.FailClosed.scopeOk_eq _).symm.trans hscope)
  obtain ⟨_, -, hupv, -, -, -⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok (htab 0 h0).2.2
  obtain ⟨-, hg0, hnbl, -⟩ := table_rows ht' hd (by omega) hv1 L _ hz1
  subst hc0
  have hn' : p.sts[0].c.unit.name = VStar.capName l N := (htab 0 h0).1
  have hcoin := Flock.Readable.array_forM_ok (RecIns.setup_coins hset) 0 h0'
  have htop := Flock.Readable.array_forM_ok (RecIns.setup_topRows hset) 0 h0'
  have hlnk := Flock.Readable.array_forM_ok (RecIns.setup_checkLink hset) 0 h0'
  have hreg := RecIns.setup_registered hset sts[0] (Array.getElem_mem h0')
  rw [← e0] at hcoin htop hlnk hreg
  have hcp := capParams_of_check hcoin hn'
  obtain ⟨ho, hres, hsp, hso, hop⟩ := params_cap l N
  rw [← hn'] at ho hres hsp hso hop
  obtain ⟨own, hown, hcap⟩ := capCheck_of_check hcoin hcp ho hres hsp hso hop
  rw [hown] at htop hreg
  rw [hlk] at htop hlnk
  obtain ⟨reg, hmem, hname, hval, hreads, hidx⟩ := checkCap_ok hcap
  have hl : RecVStar.topLevel reg.value = some l := by rw [hval]; exact topLevel_top l
  obtain ⟨t, htt, pp, positions, paths, hpp, hent, hrows⟩ := checkTopRows_rows htop hmem hl
  obtain ⟨t', idx, ht'', hidx', hports⟩ :=
    FlockSoundness.Discharge.ZkReg.check_tables (FlockSoundness.Discharge.ZkReg.checkRegistered_ok hreg)
  rw [htt] at ht''
  cases ht''
  obtain rfl : Array.range p.sts[0].pub.n = idx := Except.ok.inj (hidx.symm.trans hidx')
  obtain ⟨p', positions', paths', hp', hent', hall'⟩ := FlockSoundness.Discharge.ZkReg.checkPort_at (hports _ hmem)
  obtain ⟨p'', positions'', paths'', hp'', hent'', hsize⟩ := FlockC3.PubRows.checkPort_size (hports _ hmem)
  rw [hpp] at hp' hp''
  cases hp'
  cases hp''
  rw [hent] at hent' hent''
  cases hent'
  cases hent''
  rw [hname] at hpp
  have hq2 : p.sts[0].c.ports.findIdx? (·.name == "top") = some pp :=
    findIdx_of_find_range (f := fun (x : Port) => x.name == "top") hpp
  refine ⟨pp, hq2, fun hl0 => ?_, fun n hn => ?_⟩
  · subst hl0
    exact root_of_checkLink hlnk hcp
  have hNn := capFull_staged hS hsu hr hsv h0'
  rw [← e0] at hNn
  have hnn : n < p.sts[0].n := by omega
  have hnp : n < p.sts[0].pub.n := by rw [← setupOne_n hd]; exact hnn
  obtain ⟨hq', -⟩ := Array.findIdx?_eq_some_iff_getElem.1 hq2
  have hsl := slotPub_table ht' hd (by omega) hv1 L hz1 hq' hnp (div_lt_blocks hg0 hnn hnbl)
  rw [digest_tables htt] at hsl
  obtain ⟨hlt, hrd, -⟩ := hall' n hnp
  have hrn : ((Array.range N).map some)[n]? = some (some n) := by simp [hn]
  rw [hreads, range_getD hnp, hrn] at hrd
  have hpos : positions.getD (Registered.ref t p.sts[0].c.ports.size pp n) 0 = n :=
    (Option.some.inj (Option.some.inj hrd)).symm
  obtain ⟨wh, cc, htr, hrow⟩ := hrows _ (hsize ▸ hlt)
  rw [hpos] at htr
  refine ⟨wh, cc, htr, ?_⟩
  rw [hsl, hrow, byteArray_toList_eq]

/-! ## Round 0 is `root_B` -/

theorem link_of_input {wr : RecRaw} {src : Option RecCoins.Source} {t : ByteArray} {tt record : String} {tr : Json}
    {lk : RecVStar.InnerLink} {o : Option RecVStar.InnerLink} (ht : wr.transcript = some t)
    (hrec : src.map (·.record) = some record) (htx : String.fromUTF8? t = some tt) (hp : Json.parse tt = .ok tr)
    (hlk : RecVStar.innerLink tr record = .ok lk) (h : linkInputOf wr src = .ok o) : o = some lk := by
  have htt : textOf "rec inner transcript" t = .ok tt := by rw [textOf, htx]
  have e : linkInputOf wr src = .ok (some lk) := by
    unfold linkInputOf
    rw [ht, hrec]
    dsimp only
    simp only [htt, hp, hlk, bind, Except.bind]
    all_goals rfl
  exact Except.ok.inj (h.symm.trans e)

/-- **A staged part's transcript and link, as the verifier read them**: `staged_reads`, with the link the setup holds
(`i.link`, `linkInputOf`) the one `checkReads` read off the same transcript and record (`RecVStar.innerLink`). -/
theorem staged_reads_link {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) :
    ∃ (i : RecIns) (sh : RecVStar.Inner) (cs : RecVStar.InnerCommits) (tt : String) (tr : Json) (record : String)
      (r : RecRecord.Record) (lk : RecVStar.InnerLink),
      insO xO = some i ∧ i.inner = some sh ∧ i.commits = some cs ∧ i.link = some lk ∧
      RecSetup.checkReads sh tt tr record cs = .ok () ∧ RecVStar.innerLink tr record = .ok lk ∧
      RecRecord.readFull tt.toUTF8.toList = some r ∧
      shownOf recRead x xO = some (r.msgs.map fun mc => (mc.1, mc.2.toArray)) := by
  obtain ⟨ms, i, _spec, _sts, rfl, hsu, -⟩ := hS
  obtain ⟨hins, -⟩ := setupO_some hsu
  obtain ⟨xr, -, hinner, -⟩ := insO_staged hins
  obtain ⟨wr, hd, htr, -, hlink⟩ := insO_staged_link hins
  obtain ⟨wr', hd', hw⟩ := ins_of_insO hins
  rw [hd] at hd'
  cases hd'
  have hc := RecRaw.ins_commits hw
  rw [hinner] at hc
  obtain ⟨cs, tt, tr, record, hcs, htx, hpj, hrec, hR⟩ := commitsOf_reads htr hc
  obtain ⟨lk, r, _want, hlk, hr, -⟩ := FlockC3.VStarList.checkReads_ok hR
  refine ⟨i, xr.shape, cs, tt, tr, record, r, lk, hins, hinner, hcs,
    link_of_input htr hrec htx hpj hlk hlink, hR, hlk, hr, ?_⟩
  have hb : (st.transcript x ms).data.toList = tt.toUTF8.toList := by
    rw [byteArray_toList_eq, toUTF8_of_fromUTF8? htx]
  simp only [shownOf, hd, htr, Option.bind_some, recRead, hb, hr, Option.map_some]

theorem capAt_stmt {r : XR} {si N s : ℕ} (h : capAt r = some (si, N)) (hs : s = si) :
    (RecVStar.statements r.shape).toOption.bind (·[s]?) = some (.cap 0 N) := by
  subst hs
  unfold capAt at h
  obtain ⟨ss, hss, h⟩ := Option.bind_eq_some_iff.1 h
  rw [hss, Option.bind_some]
  split at h
  · rename_i N' hb
    simp only [Option.some.injEq, Prod.mk.injEq] at h
    obtain ⟨rfl, rfl⟩ := h
    simpa [Array.back?] using hb
  · cases h

theorem flatMap_range_nodes (a : Array ByteArray) (g : ℕ → List UInt8)
    (hg : ∀ n < a.size, g n = (a.getD n default).toList) :
    (List.range a.size).flatMap g = (a.toList.map (·.toList)).flatten := by
  rw [List.flatMap.eq_1]
  congr 1
  apply List.ext_getElem (by simp)
  intro n h1 h2
  simp only [List.getElem_map, List.getElem_range, Array.getElem_toList]
  have hn : n < a.size := by simpa using h1
  rw [hg n hn]
  simp [Array.getD, hn]

theorem msgs_zero (rc : RecRecord.Record) (t : List UInt8) (nodes : List (List UInt8))
    (h : rc.roots = [(t, nodes)]) : (rc.msgs.map Prod.fst).getD 0 [] = nodes.flatten := by
  simp [RecRecord.Record.msgs, h]

/-- **RecCap's part holds round 0 as the coin server recorded it**: at a staged part the witness checks against, what
part `s` holds whole at round 0 at `layoutBTops` is RecCap's `N` slots at port `top` in order (`capHold`), each node `n`
of the coin server's `root_B` (`capRow_bytes`), which has `N` nodes, so the record's round 0 (`Record.msgs`,
`FlockC3.VStarList.checkReads_ok`), which is round 0 as `cmtTr recRead` reads it (`staged_reads_link`). -/
theorem cap_cmt (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) {st : Stage} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) {m : Msg}
    (hm : (layoutBTops fin).held x s w 0 = some m) : m = cmtTr recRead x 0 xO := by
  obtain ⟨i, sh, cs, tt, tr, record, rc, lk, hi, -, -, hil, hR, hlk, hrd, hshown⟩ := staged_reads_link hS
  obtain ⟨lk', rc', want, hlk', hrd', -, hroots, -⟩ := FlockC3.VStarList.checkReads_ok hR
  rw [hlk] at hlk'
  obtain rfl : lk = lk' := Except.ok.inj hlk'
  rw [hrd] at hrd'
  obtain rfl : rc = rc' := Option.some.inj hrd'
  rw [cmtTr_record hshown, msgs_zero rc _ _ hroots]
  unfold Claims.Recursion.VStarB.LayoutB.held at hm
  split at hm
  · cases hm
  obtain ⟨segs, hseg, rfl⟩ := Option.map_eq_some_iff.1 hm
  rw [layoutBTops_msgAt_zero] at hseg
  obtain ⟨r, hr, hseg⟩ := Option.bind_eq_some_iff.1 hseg
  unfold capHold at hseg
  obtain ⟨⟨si, N⟩, hcap, hseg⟩ := Option.bind_eq_some_iff.1 hseg
  split at hseg
  · rename_i hsi
    obtain ⟨c, hpc, hseg⟩ := Option.bind_eq_some_iff.1 hseg
    obtain ⟨q, hq, rfl⟩ := Option.map_eq_some_iff.1 hseg
    obtain ⟨q', hq', hroot, hbytes⟩ := capRow_bytes hS hck hr hpc (capAt_stmt hcap hsi) hi hil
    obtain rfl : q = q' := Option.some.inj (hq.symm.trans hq')
    have hN := hroot rfl
    unfold Claims.Recursion.VStarB.readSegs
    rw [List.flatMap_map, ← hN]
    refine flatMap_range_nodes lk.root _ fun n hn => ?_
    obtain ⟨wh, cc, htop, hb⟩ := hbytes n (hN ▸ hn)
    simp only [List.nil_append]
    rw [hb, topRow_zero htop]
  · cases hseg

/-! ## `PartOpensB` at `layoutBTops`, but the tops rounds' pieces -/

/-- **Every round a staged part holds whole is the record's message there**: round 0 at RecCap's part (`cap_cmt`), a
later round at its held row's part (`held_cmt`). -/
theorem heldB_cmt (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) {st : Stage} {s : ℕ} {x xO : Msg}
    {w : Claims.Flock.Wit} (hS : StagedPart st s x xO)
    (hck : Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true) (i : ℕ) {m : Msg}
    (hm : (layoutBTops fin).held x s w i = some m) : m = cmtTr recRead x i xO := by
  cases i with
  | zero => exact cap_cmt fin hS hck hm
  | succ j => exact held_cmt fin hS hck hm

/-- **`PartOpensB` at `layoutBTops` reduces to the tops rounds' pieces**: at `showsPart recRead` every round a part
holds whole is the record's (`heldB_cmt`), and before the last round only a stream round that commits tops has pieces
(`stepPieces`, `topsPieces`). So `PartOpensB` holds wherever each such piece of a staged part reads the record's
message at its offset. -/
theorem partOpensB_layoutBTops (fin : Msg → List (ℕ × List (Msg × List HmNets.Lin))) (b : Brings)
    {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop} {s : ℕ}
    (hP : ∀ (x : Msg) st, D.parse x = some st → shape st.form → ∀ (xO : Msg) (w : Claims.Flock.Wit),
      Claims.Flock.check (deployZkRec Flock.Readable.relation).parse xO w = true → StagedPart b.st s x xO →
      ∀ j, j + 2 < layoutOf.rounds x → ∀ pre p post, topsPieces x j = pre ++ p :: post → p.1 = s →
        ((cmtTr recRead x (j + 1) xO).drop (pre.map fun q => Claims.Recursion.VStarB.pieceLen q.2).sum).take
          (Claims.Recursion.VStarB.pieceLen p.2) = Claims.Recursion.VStarB.readSegs w p.2) :
    Claims.Recursion.VStarB.PartOpensB H D (deployZkRec Flock.Readable.relation) (layoutBTops fin) shape
      (showsPart recRead b) (cmtTr recRead) s := by
  intro x st hx hf xO w hck _
  refine Or.inl fun c hc i hi => ⟨fun m hm => heldB_cmt fin hc.2 hck i hm, fun pre p post hpp hps => ?_⟩
  have hr : (layoutBTops fin).rounds x = layoutOf.rounds x := rfl
  rw [hr] at hi
  have hst : (layoutBTops fin).pieceAt x i = stepPieces fin x i := rfl
  rw [hst, stepPieces, ite_eq_right (show ¬(i + 1 = layoutOf.rounds x) by omega)] at hpp
  cases i with
  | zero => simp at hpp
  | succ j =>
    rw [ite_eq_right (by omega : j + 1 ≠ 0), Nat.add_sub_cancel] at hpp
    exact hP x st hx hf xO w hck hc.2 j (by omega) pre p post hpp hps

end RecDeploy
