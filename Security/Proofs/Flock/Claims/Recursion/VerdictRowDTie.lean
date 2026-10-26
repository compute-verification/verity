import Proofs.Flock.Claims.Recursion.VerdictRowDCoin
import Proofs.Flock.Claims.Recursion.OpenLowerDeployed

/-!
# Row D's ties at the deployed layout: the side's schedule and check 5's positions

`OpenLowerDeployed` lowers a `rec-open` piece at the deployed layout from facts about the side and position readers it
is handed. This file gives two of them at row D's readers `RecDeploy.Rows.sideAt` and `RecDeploy.Rows.posAt`:

* `sideSchR_sideAt`: `SideSchR` at `Rows.sideAt`, outright: its schedule is `schOf`, which is `schAt`, whatever the
  messages.
* `PosTieR`: check 5's `dirs` rows at a staged `rec-open` part (`openRows`, at its source's inner session) are the low
  `HH` bits of the positions `posAt` reads off the coins. `posTieR_of` proves it from `InnerCoinsR`: check 5's inner
  session is the inner statement's table at the coins (its `m`, `k_log`, its extra claims at the first coin's link
  points, and each rep's rounds `canonRounds` of the coins after the link). The rest is the verifier's own algebra:
  `openRows`' `dirs` row `r q + j` is rep `r`'s query `j` at the one level of `LANES` lanes and `HH` siblings
  (`openRows_dirs`, `openDirs_get`), that level is the piece's (`levelMatch`, `innerLevels_unique`), and `verifierRows`
  reads the same positions at check 5's shape (`RecResiduals.shape?`) as at the reader's (`shapeOf`), which differ only
  in the Merkle widths it doesn't read (`verifierRows_congr`), and at any comb (`verifierRows_comb`).
-/

namespace RecDeploy.OpenLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf tabsOf pointsOf)
open Claims.Recursion.VStarB
open FlockSoundness.Refine (shapeOf)
open FlockSoundness.Discharge.Recursion (openRows openDirs innerLevels)
open RecDeploy.Rows (canonRounds posAt sideAt schOf verifierRows_comb)

/-! ## The side's schedule -/

theorem schOf_eq (x : Msg) (t : ℕ) : schOf x t = schAt x t := by
  unfold schOf schAt
  cases setupOf false x <;> rfl

section Side

variable {H : Msg → Msg} {D : Claims.Recursion.Deploy Coin} {shape : Claims.Recursion.Form → Prop}
  {untr : TrReader} {b : Brings} {cmt : Msg → ℕ → Msg → Msg} {L : LayoutB}
  {Bd : Claims.Recursion.Bound Coin (List FlockLevel3.GF128) Claims.Flock.PlainTrace}

/-- **`SideSchR` at row D's side reader**: `Rows.sideAt`'s schedule is the table's, whatever the messages. -/
theorem sideSchR_sideAt (rd : Msg → Option FinalP) (pieces : Msg → List OpenPiece) :
    SideSchR H D shape untr b cmt L Bd (sideAt rd) pieces := by
  unfold SideSchR AtPieces AtOpened
  intros
  intros
  exact schOf_eq _ _

end Side

/-! ## The verifier rows read only the algebra's fields of the shape -/

/-- **`verifierRows` doesn't read the Merkle widths**: at shapes of the same `m`, `k_log`, claims, schedule and
`holo`, it computes the same rows. -/
theorem verifierRows_congr {sh sh' : Shape} (hm : sh.m = sh'.m) (hk : sh.kLog = sh'.kLog)
    (hc : sh.claims = sh'.claims) (hs : sh.sched = sh'.sched) (hh : sh.holo = sh'.holo)
    (rounds : Array (Array F128)) (extras : Array Claim) (comb : Array F128) :
    RecAlgebra.verifierRows sh rounds extras comb = RecAlgebra.verifierRows sh' rounds extras comb := by
  obtain ⟨m, k, c, s, h₁, s₁, o⟩ := sh
  obtain ⟨m', k', c', s', h₂, s₂, o'⟩ := sh'
  dsimp only at hm hk hc hs hh
  subst hm hk hc hs hh
  rfl

/-! ## Check 5's `dirs` rows -/

/-- **Check 5's `rec-open` rows, opened**: where `openRows` gives rows, the inner session's algebra, the one level of
`LANES` lanes and `H` siblings, both reps, and the `dirs` rows `openDirs` builds of them. -/
theorem openRows_dirs {inn : RecCoins.Inner} {LANES H : ℕ} {cr dr : Array (Array ℕ)}
    (h : openRows inn LANES H = some (cr, dr)) :
    ∃ sh extras l r0 r1, inn.algebra = .ok (sh, extras) ∧ innerLevels sh LANES H = [l] ∧
      inn.rep sh extras 0 = .ok r0 ∧ inn.rep sh extras 1 = .ok r1 ∧ dr = openDirs sh l H r0 r1 := by
  unfold openRows at h
  cases ha : inn.algebra with
  | error e => rw [ha] at h; cases h
  | ok p =>
    obtain ⟨sh, extras⟩ := p
    rw [ha] at h
    dsimp only at h
    cases hl : innerLevels sh LANES H with
    | nil => rw [hl] at h; simp at h
    | cons l t =>
      cases t with
      | cons _ _ => rw [hl] at h; simp at h
      | nil =>
        rw [hl] at h
        dsimp only at h
        cases h0 : inn.rep sh extras 0 <;> cases h1 : inn.rep sh extras 1 <;> rw [h0, h1] at h <;>
          simp only [reduceCtorEq] at h
        rename_i r0 r1
        simp only [Option.some.injEq, Prod.mk.injEq] at h
        exact ⟨sh, extras, l, r0, r1, rfl, rfl, rfl, rfl, h.2.symm⟩

/-- **`dirs` row `r q + j` is rep `r`'s query `j`**: at level `l` of `q` queries, instance `r q + j` (`r < 2`,
`j < q`) holds the low `H` bits of rep `r`'s `j`-th position there. -/
theorem openDirs_get (sh : Shape) (l H : ℕ) (r0 r1 : RecCoins.Rep) {r j : ℕ} (hr : r < 2)
    (hj : j < (sh.sched.levels[l]!).q) :
    (openDirs sh l H r0 r1)[r * (sh.sched.levels[l]!).q + j]! =
      RecCoins.dirsRow (((if r = 0 then r0 else r1).info.lg.conss[l]!).pos[j]!) H := by
  have hlt : r * (sh.sched.levels[l]!).q + j < (openDirs sh l H r0 r1).size := by
    unfold openDirs
    rw [Array.size_map, Array.size_range]
    have : r * (sh.sched.levels[l]!).q ≤ 1 * (sh.sched.levels[l]!).q := Nat.mul_le_mul_right _ (by omega)
    omega
  rw [getElem!_pos _ _ hlt]
  unfold openDirs
  simp only [Array.getElem_map, Array.getElem_range]
  obtain rfl | rfl : r = 0 ∨ r = 1 := by omega
  · simp [hj, Nat.mod_eq_of_lt hj]
  · simp only [Nat.one_mul]
    have h1 : ¬ (sh.sched.levels[l]!).q + j < (sh.sched.levels[l]!).q := by omega
    have h2 : ((sh.sched.levels[l]!).q + j) % (sh.sched.levels[l]!).q = j := by
      rw [Nat.add_mod_left, Nat.mod_eq_of_lt hj]
    simp [h1, h2, Nat.mod_eq_of_lt hj]

/-- **The one level of `LANES` lanes and `H` siblings is any level of them.** -/
theorem innerLevels_unique {sh : Shape} {LANES H l l' : ℕ} (h : innerLevels sh LANES H = [l])
    (hl' : l' < sh.sched.levels.size) (hL : 2 ^ (sh.sched.levels[l']!).lanes = LANES)
    (hH : (sh.sched.levels[l']!).d - (sh.sched.levels[l']!).cap = H) : l' = l := by
  have hm : l' ∈ innerLevels sh LANES H := by
    unfold innerLevels
    simp [List.mem_filter, hl', hL, hH]
  rw [h] at hm
  simpa using hm

theorem algebra_ok {inn : RecCoins.Inner} {sh : Shape} {extras : Array Claim}
    (h : inn.algebra = .ok (sh, extras)) :
    inn.extras = .ok extras ∧ RecResiduals.shape? inn.m inn.kLog (2 + extras.size) = some sh := by
  unfold RecCoins.Inner.algebra at h
  cases he : inn.extras with
  | error e => simp [he, bind, Except.bind] at h
  | ok ex =>
    cases hs : RecResiduals.shape? inn.m inn.kLog (2 + ex.size) with
    | none => simp [he, hs, bind, Except.bind, throw, throwThe, MonadExceptOf.throw] at h
    | some s =>
      simp [he, hs, bind, Except.bind, pure, Except.pure] at h
      obtain ⟨rfl, rfl⟩ := h
      exact ⟨he, hs⟩

theorem shape?_ok {m k c : ℕ} {sh : Shape} (h : RecResiduals.shape? m k c = some sh) :
    ∃ sch, fast100 m = some sch ∧ sh.m = m ∧ sh.kLog = k ∧ sh.claims = c ∧ sh.sched = sch ∧ sh.holo = false := by
  unfold RecResiduals.shape? at h
  cases hf : fast100 m with
  | none => simp [hf] at h
  | some s =>
    refine ⟨s, rfl, ?_⟩
    by_cases hc : (k < 6 || c < 2) = true
    · simp [hf, hc] at h
    · simp [hf, hc] at h
      subst h
      exact ⟨rfl, rfl, rfl, rfl, rfl⟩

/-! ## The position tie -/

/-- **Check 5's inner session is the inner statement at the coins** (`InnerCoinsR`): at a V\* statement `xO` part `s`
stages at `x` and shows the coins `c` at, wherever check 5 loads a table's inner session from its source, the session
is the inner statement's table 0 (its `m` and `k_log`), its extra claims are the table's at the first coin's link points,
and each rep's rounds are `canonRounds` of the coins after the link, as row D's readers take them. This is the coin
side of check 5 at every value it registers (`staged_rep_drawn` is its round-by-round form). -/
def InnerCoinsR (untr : TrReader) (b : Brings) (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ t (_ : t < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[t].c.unit.name = .ok inn →
      ∃ v, setupOf false (xIOf x) = some v ∧ inn.m = (tabsOf v.2.2)[0]!.m ∧ inn.kLog = (tabsOf v.2.2)[0]!.kLog ∧
        inn.extras = (tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default)) ∧
        ∀ r < 2, inn.rounds r = .ok (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast r)

/-- **The position tie** (`PosTieR`, the coin side of `CopyDirsTieR`): at a V\* statement `xO` part `s` stages at `x`
and shows the coins `c` at, wherever check 5 loads a table's inner session from its source and builds `rec-open`'s
rows at the level part's `LANES`, `HH` (`openRows`), the `dirs` row of a piece `ownQR` gives part `s`, instance
`rep · q + query` (check 5's order, `copyAtR`'s), is the low `HH` bits of the position `posAt` reads off the coins. -/
def PosTieR (untr : TrReader) (b : Brings) (pieces : Msg → List OpenPiece)
    (posAt : Msg → List Coin → OpenPiece → ℕ) (Bk : Msg → ℕ) (s : ℕ) (x : Msg) (c : ℕ → Coin) : Prop :=
  ∀ (xO : Msg), showsPart untr b s x xO c → ∀ (i : RecIns) (spec : SessionSpec) (sts : Array Stmt),
    setupO xO = some (i, spec, sts) → ∀ t (_ : t < sts.size) (f : RecCoins.Source) (inn : RecCoins.Inner),
      RecCoins.Source.atC3 i.src i.inner i.commits = some f → f.load sts[t].c.unit.name = .ok inn →
      ∀ (r : XR) (ss : Array VStar.Stmt) (lv LANES HH n : ℕ), readX x = some r →
        _root_.Flock.RecVStar.statements r.shape = .ok ss → ss[s]? = some (.recOpen lv LANES HH n) →
      ∀ (cr dr : Array (Array ℕ)), openRows inn LANES HH = some (cr, dr) →
      ∀ q ∈ pieces (xIOf x), ownQR x q = some s →
        dr[q.rep * qAt (xIOf x) q.level + q.query]! =
          RecCoins.dirsRow (posAt (xIOf x) ((List.range (Bk (xIOf x))).map c) q) HH

/-- **`PosTieR` at row D's position reader**, from `InnerCoinsR`, at a statement the reader reads at the dense form
and pieces in range. -/
theorem posTieR_of {untr : TrReader} {b : Brings} {pieces : Msg → List OpenPiece} {Bk : Msg → ℕ} {s : ℕ}
    {x : Msg} {c : ℕ → Coin} {st : Claims.Recursion.Stmt} (hst : parse x = some st)
    (hd : Claims.Recursion.Form.Dense st.form) (hR : PiecesInRangeR pieces)
    (hI : InnerCoinsR untr b Bk s x c) : PosTieR untr b pieces posAt Bk s x c := by
  intro xO hsh i spec sts hsu t ht f inn hf hload r ss lv LANES HH n hr hss hsv cr dr hop q hq ho
  obtain ⟨v, hv, hm, hk, hex, hrd⟩ := hI xO hsh i spec sts hsu t ht f inn hf hload
  obtain ⟨sh, extras, l, r0, r1, halg, hl, h0, h1, rfl⟩ := openRows_dirs hop
  obtain ⟨hext, hsh'⟩ := algebra_ok halg
  obtain ⟨sch, hsch, hshm, hshk, hshc, hshs, hshh⟩ := shape?_ok hsh'
  -- the piece's level is the statement's
  obtain ⟨hq0, r', ss', hr', hss', L', H', n', hsv'⟩ := ownQR_some ho
  rw [hr] at hr'
  cases hr'
  rw [hss] at hss'
  cases hss'
  rw [hsv] at hsv'
  simp only [Option.some.injEq, VStar.Stmt.recOpen.injEq] at hsv'
  obtain ⟨hlvq, -, -, -⟩ := hsv'
  obtain ⟨hlvs, hLANES, hHH, -⟩ := levelMatch hst hd hr hss hsv
  have hS : schAt (xIOf x) 0 = sch := by
    unfold schAt
    rw [hv]
    dsimp only
    rw [← hm, hsch, Option.getD_some]
  obtain ⟨hrep, hquery⟩ := hR (xIOf x) q hq
  rw [hq0, hS] at hquery
  have hlev : q.level < sch.levels.size := by
    by_contra hge
    rw [getElem!_neg _ _ hge] at hquery
    exact Nat.not_lt_zero _ hquery
  rw [hS, ← hlvs, hlvq] at hLANES hHH
  have hql : q.level = l := innerLevels_unique hl (by rw [hshs]; exact hlev) (by rw [hshs]; exact hLANES.symm)
    (by rw [hshs]; exact hHH.symm)
  -- check 5's row
  have hqa : qAt (xIOf x) q.level = (sh.sched.levels[l]!).q := by
    unfold qAt
    rw [hS, hshs, hql]
  rw [hqa, openDirs_get sh l HH r0 r1 hrep (by rw [hshs, ← hql]; exact hquery)]
  refine congrArg (fun p => RecCoins.dirsRow p HH) ?_
  -- rep `q.rep`'s rounds and rows
  have hrq : inn.rep sh extras q.rep = .ok (if q.rep = 0 then r0 else r1) := by
    obtain h | h : q.rep = 0 ∨ q.rep = 1 := by omega
    · rw [h]; simpa using h0
    · rw [h]; simpa using h1
  obtain ⟨hrounds, hvr⟩ := rep_ok hrq
  rw [hrd q.rep hrep] at hrounds
  have hR' := (Except.ok.inj hrounds).symm
  rw [hR', verifierRows_comb] at hvr
  -- the reader's position
  have hexs : ((tabsOf v.2.2)[0]!.extra (pointsOf v.2.1 (((List.range (Bk (xIOf x))).map c).headD default))) =
      .ok extras := by rw [← hex, hext]
  unfold posAt
  rw [hv]
  dsimp only
  rw [hq0, hexs]
  simp only [Except.toOption, Option.getD_some, Nat.mul_zero, Nat.zero_add]
  rw [verifierRows_congr (sh := shapeOf (tabsOf v.2.2)[0]! ((fast100 (tabsOf v.2.2)[0]!.m).getD default) extras.size)
    (sh' := sh) (by rw [hshm, hm]; rfl) (by rw [hshk, hk]; rfl) (by rw [hshc]; rfl) (by rw [hshs, ← hm, hsch]; rfl)
    (by rw [hshh]; rfl)]
  cases h : RecAlgebra.verifierRows sh (canonRounds v.2.1 v.2.2 ((List.range (Bk (xIOf x))).map c).tail.dropLast q.rep)
      extras #[] with
  | error e => rw [h] at hvr; cases hvr
  | ok w =>
    rw [h] at hvr
    obtain ⟨w₁, w₂, w₃⟩ := w
    simp only [Functor.map, Except.map, Except.ok.injEq, Prod.mk.injEq] at hvr
    obtain ⟨-, hi, -⟩ := hvr
    rw [← hi, hql]

end RecDeploy.OpenLower
