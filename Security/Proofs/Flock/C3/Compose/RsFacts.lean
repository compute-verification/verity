import Proofs.Flock.C3.Compose.SparseEnds
import Proofs.Flock.C3.Reps.IoCols
import Proofs.Flock.C3.Opening.Session
import Proofs.Flock.C3.PubRows.Shape
import Proofs.Flock.Recursive.Loads

/-!
# RsRead's facts

`rsRead_of_layer` takes four named facts. Here they come from facts closer to V*'s files, circuits and reads file:

* `OneCopy` from each accepted session being V*'s statement of its index (`SessStmt`, from the verifier's statement
  check, `sessStmt_of_checks`) and the verifier's `rec-opening` check of one copy (`OpenUpv`, `openUpv`),
  `oneCopy_of`: one session per part (`openSess_of`: the list has one statement named as the part's unit) of one unit
  (`nv_of_copy`: that statement is one of R's opening parts, of one instance).
* `OpenAll` from V*'s sessions (`SessionAt`'s (a)), `openAll_of_session`.
* `RsSk` from the reads file at the ports keying the ring switches' columns (`RsSel`, S's `SelPort`), `rsSk_of_sel`:
  the column's key (`OpeningZeroAtL`'s `RegBit`) puts its wire at the port's commit string (`posK_of_keyAt`).
* `RsIo` from the unit reading its ring switches' columns (`RsReadCol`, S's `ReadCol`, which the opening's check gives,
  `rsReadCol`), `rsIo_of_readCol`: copy `j` reads a column its computed rows read (`mem_io_of_readCol`,
  `ioColAt_of_readCol` at copy `j`).

`rsRead_of_facts` composes them. `RsSel` is at the round commitments (`rsSel_of_rounds`): ring switch `i`'s columns
are keyed by port `m<i>` (`RsKeys`, the opening's check, `rsKeys`), whose reads-file entry is `rec-opening-c<i>`, read
at the unit where stage `c₀ + i` reads it (`RoundReads`; `WRoundReads` is it at the residuals' parts; both at the
verifier's own ports, `own := ownAt ss`, are `C3/Reps/RoundReads.lean`'s). `SessStmt` is at any family whose checks
pass (`sessStmt_of_accepted`, from `ChecksAccepted`), and at the checked family from the guarantee's `_hF`
(`sessStmt_checked`: the checked sessions carry the prover's inputs, `tagsFound_checked`); `rsRead_of_accepted`
composes them, and `rsRead_checked` is it at the checked family.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge (keyInst)
open FlockVBridge.Family (skOf mem_skOf_inl)
open FlockVBridge.OfVStar (rowCol partRep readCol_rsAt part_eqOpening wIdx_size ReadCol)
open FlockVBridge.PartS (wIdx)
open FlockC3.Reps (SelPort)

/-- **A column of copy `j` keyed by bit `k` of row port `q` is at `q`'s commit string** (`posK_of_key` at copy
`j`). -/
theorem posK_of_keyAt {Q : VStmt} (u : Fin Q.nv) {j : ℕ} (hj : j < Q.c.upv) {c : ℕ}
    (hc : c < (Rows.ofNet Q.hU).nIn) {q : Fin Q.c.ports.size} {k : ℕ} (hk : k < Layout.portRowW Q.c q)
    (hkey : Aliased.srcKey Q.c (j * (Rows.ofNet Q.hU).nIn + c) = .msg q.val (k / 8) (k % 8))
    (hcw : c < (Rows.ofNet Q.hU).w) :
    posK Q.c Q.hU Q.nv ((keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).wire
      ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨c, hcw⟩)) = .inl (u, q) := by
  have hx : ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨c, hcw⟩).val = j * (Rows.ofNet Q.hU).nIn + c := by
    rw [Rows.stackCol, dite_eq_left hc]
  have hr := HiddenExec.rowOfH_of_key (Layout.outsOf Q.c Q.hU Q.nv)
    (FlockVBridge.Decode.keyRowOf_keyInst (Layout.outsOf Q.c Q.hU Q.nv) u
      ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨c, hcw⟩) (by rw [hx]; exact Rows.mul_add_lt hj hc) hk
      (by rw [hx]; exact hkey))
  simp only [posK, hr]

/-- **Copy `j` reads a column its computed rows read** (`ioColAt_of_readCol` at copy `j`): copy `j`'s computed row `i`
of the stack reads the copy's columns of row `i` (`stack_a`, `stack_b`), and an input column is no gate of the unit
(`outside`), so its wire is one of the unit's inputs (`mem_io_of_read`). -/
theorem mem_io_of_readCol {Q : VStmt} (u : Fin Q.nv) {j : ℕ} (hj : j < Q.c.upv) {col : ℕ}
    (hc : col < (Rows.ofNet Q.hU).nIn) (h : ReadCol (Rows.ofNet Q.hU) col) (hcw : col < (Rows.ofNet Q.hU).w) :
    (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).wire ((Rows.ofNet Q.hU).stackCol Q.c.upv hj ⟨col, hcw⟩) ∈
      ((Placed.keyProg Q.c Q.hU Q.nv).partition (Layout.outsOf Q.c Q.hU Q.nv)).io u := by
  obtain ⟨i, x, hx, rfl⟩ := h
  refine (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).mem_io_of_read
    (i := ⟨j * ((Rows.ofNet Q.hU).nComp + 1) + (1 + i.val), Rows.mul_add_lt hj (by omega)⟩) ?_ ?_
  · rw [FlockVBridge.stack_a _ _ hj i, FlockVBridge.stack_b _ _ hj i]
    rcases List.mem_append.1 hx with hx | hx
    · exact List.mem_append_left _ (List.mem_map_of_mem hx)
    · exact List.mem_append_right _ (List.mem_map_of_mem hx)
  · refine (keyInst Q.hU Q.nv (Layout.outsOf Q.c Q.hU Q.nv) u).outside _ fun i' h' => ?_
    have he := congrArg Fin.val h'
    have hlt := Rows.mul_add_lt hj hc
    rw [Rows.stackCol, dite_eq_left hc] at he
    simp only [Rows.comp, Rows.stack] at he
    omega

/-- **`RecOpening.check` accepts a `rec-opening` unit of one copy only** (`openUpv`): a unit named
`rec-opening/m<m>-c<claims>-p<j>` that `check` accepts has `upv = 1`. -/
def OpenUpv : Prop :=
  ∀ c : Flock.Circuit, RecOpening.check c = .ok () → ∀ m cl p, RecOpening.params c.unit.name = some (m, cl, p) →
    c.upv = 1

/-- **`OpenUpv` from the verifier's `rec-opening` check**, which refuses any other `upv` (`upv_of_check`). -/
theorem openUpv : OpenUpv := fun _ h _ _ _ hp => FlockVBridge.OfVStar.upv_of_check h hp

/-! ## V*'s list at a `rec-opening` name -/

/-- `params` gives `(m, c, j)` only for the name `unitName m c j`. -/
theorem openingName_params {name : String} {m c j : ℕ} (h : RecOpening.params name = some (m, c, j)) :
    RecOpening.unitName m c j = name := by
  unfold RecOpening.params at h
  repeat' split at h
  all_goals first
    | (simp only [Option.some.injEq, Prod.mk.injEq] at h; obtain ⟨rfl, rfl, rfl⟩ := h; assumption)
    | cases h

theorem openShape_m {m c : ℕ} {sh : Flock.Shape} (h : RecOpening.shape? m c = some sh) : sh.m = m := by
  unfold RecOpening.shape? at h
  cases hf : Flock.fast100 m with
  | none => simp [hf] at h
  | some s =>
    by_cases hc : c < 1
    · simp [hf, hc] at h
    · simp [hf, hc] at h
      rw [← h]

theorem take_nine {p : String} {t : List Char} (hp : 9 ≤ p.toList.length) :
    (p.toList ++ t).take 9 = p.toList.take 9 := List.take_append_of_le_length hp

/-- **No other statement's name is a `rec-opening` unit's**: V*'s levels and parts, R's levels and RecSparse are named
`rec-open/`, `rec-resid…`, `rec-open-…` or `rec-spars…`, R's opening `rec-openi…`. -/
theorem openingName_ne (m c j : ℕ) :
    (∀ L H : ℕ, RecOpening.unitName m c j ≠ s!"rec-open/l{L}-h{H}") ∧
    (∀ m' k c' j' : ℕ, RecOpening.unitName m c j ≠ s!"rec-residuals/m{m'}-k{k}-c{c'}-p{j'}") ∧
    (∀ L H, RecOpening.unitName m c j ≠ VStar.recOpenStackedName L H) ∧
    (∀ sh, RecOpening.unitName m c j ≠ RecSparse.unitName sh) := by
  refine ⟨?_, ?_, ?_, ?_⟩ <;> intros <;> intro h <;>
    have h := congrArg (fun s : String => s.toList.take 9) h <;>
    simp only [RecSparse.unitName, VStar.recOpenStackedName, RecOpening.unitName, String.toList_append,
      List.append_assoc, FlockVBridge.toString_str] at h <;>
    rw [take_nine (by decide), take_nine (by decide)] at h <;>
    exact absurd h (by decide)

/-- **The list's statement named `unitName m c j` is one of R's opening parts, of one instance** (`.opening`): every
other statement of V*'s and candidate 3's is named otherwise (`openingName_ne`). -/
theorem instances_of_openingName {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    {s : VStar.Stmt} (hs : s ∈ ss) {m c j : ℕ} (hn : RecOpening.unitName m c j = s.name) : s.instances = 1 := by
  obtain ⟨n1, n2, n3, n4⟩ := openingName_ne m c j
  obtain ⟨vsh, A, C, hA, hC, rfl, -⟩ := FlockC3.VStarList.statements_ok h
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨_, _, _, -, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
    simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_range] at hs
    rcases hs with ⟨l, -, rfl⟩ | ⟨j', -, rfl⟩
    · exact absurd hn (n1 _ _)
    · exact absurd hn (n2 _ _ _ _)
  · cases hsp : i.sparse with
    | none =>
      rw [(FlockC3.VStarList.candidate3_ok hC).1 hsp] at hs
      simp at hs
    | some sp =>
      obtain ⟨rest, rfl, hrest⟩ := FlockC3.PubRows.candidate3_rest hC hsp
      rcases Array.mem_append.1 hs with hs | hs
      · simp only [List.mem_toArray, List.mem_singleton] at hs
        subst hs
        exact absurd hn (n4 _)
      · rcases hrest s hs with ⟨l, L, H, n', rfl⟩ | ⟨m', c', k, S', ports, nv, rfl⟩
        · exact absurd hn (n3 _ _)
        · rfl

theorem name_of_filter {l : List VStar.Stmt} {name : String} {x : VStar.Stmt}
    (h : l.filter (·.name == name) = [x]) : x ∈ l ∧ x.name = name := by
  have hm : x ∈ l.filter (·.name == name) := by rw [h]; exact List.mem_singleton_self x
  rw [List.mem_filter] at hm
  exact ⟨hm.1, by simpa using hm.2⟩

/-- **One statement so named, at one index**: where the list's statements named `name` are one, two indices holding a
statement so named are one. -/
theorem idx_eq_of_filter {name : String} {x : VStar.Stmt} :
    ∀ (l : List VStar.Stmt), l.filter (·.name == name) = [x] → ∀ (i i' : ℕ) (a a' : VStar.Stmt),
      l[i]? = some a → l[i']? = some a' → a.name = name → a'.name = name → i = i'
  | [], _, _, _, _, _, h, _, _, _ => by simp at h
  | b :: l, hf, i, i', a, a', hi, hi', ha, ha' => by
    cases hb : (b.name == name) with
    | false =>
      rw [List.filter_cons_of_neg (by simp [hb])] at hf
      cases i with
      | zero =>
        simp only [List.getElem?_cons_zero, Option.some.injEq] at hi
        subst hi
        simp [ha] at hb
      | succ i =>
        cases i' with
        | zero =>
          simp only [List.getElem?_cons_zero, Option.some.injEq] at hi'
          subst hi'
          simp [ha'] at hb
        | succ i' =>
          simp only [List.getElem?_cons_succ] at hi hi'
          rw [idx_eq_of_filter l hf i i' a a' hi hi' ha ha']
    | true =>
      rw [List.filter_cons_of_pos (by simp [hb]), List.cons.injEq] at hf
      have hnil : ∀ {i : ℕ} {a : VStar.Stmt}, l[i]? = some a → a.name = name → False := by
        intro i a hi ha
        have hm : a ∈ l.filter (·.name == name) := List.mem_filter.2 ⟨List.mem_of_getElem? hi, by simp [ha]⟩
        rw [hf.2] at hm
        simp at hm
      cases i with
      | zero =>
        cases i' with
        | zero => rfl
        | succ i' =>
          simp only [List.getElem?_cons_succ] at hi'
          exact (hnil hi' ha').elim
      | succ i =>
        simp only [List.getElem?_cons_succ] at hi
        exact (hnil hi ha).elim

theorem toList_getElem_of {xs : Array VStar.Stmt} {i : ℕ} {a : VStar.Stmt} (h : xs[i]? = some a) :
    xs.toList[i]? = some a := by
  rw [Array.getElem?_toList]
  exact h

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Each accepted session is V*'s statement of its index** (named; `sessStmt_of_checks` discharges it from the
verifier's statement check, `StmtsChecked`, and the inputs' tags, `TagsFound`): at an accepted outer proof, session
`s`'s unit names exactly one statement of the verifier's list at `ish`, statement `s`, and its circuit is the parse of
V*'s file for it under `bind s`, at V*'s tags and no lookup tables. -/
def SessStmt (ish : RecVStar.Inner) (bind : Fin S → VStar.Bindings)
    (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) : Prop :=
  ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o → ∀ s : Fin S,
    ∃ (ss : Array VStar.Stmt) (sv : VStar.Stmt) (cf : ByteArray), RecVStar.statements ish = .ok ss ∧
      ss[(s : ℕ)]? = some sv ∧ ss.toList.filter (·.name == (Vs s).c.unit.name) = [sv] ∧
      RecVStar.circuitFile sv (bind s) ish.holo = .ok cf ∧
      Flock.HmRow.parse (Flock.Zk.tags (Flock.Tags.circuit.withCoins
        (.tree (Flock.Zk.coinSpec #[Flock.Tags.circuit.table]))) (Flock.Zk.maskLogOf cf) 1) cf [] = .ok (Vs s).c

/-- **`SessStmt` from the verifier's statement check**: where a session accepts, some transcript it accepts passes
`checkStmt` as statement `s` under `bind s` (`StmtsChecked`, the checked family's `ChecksAccepted`), and the inputs
carry the tags their name names (`TagsFound`): the first table's statement has the session's circuit
(`setup_zero_c`), the check finds its unit's one statement (`checkStmt_ok`) and V*'s file for it is
the inputs', whose parse is the session's circuit at V*'s tags (`parse_zkTags_vstar`, as `loads_of_verifiesV`). -/
theorem sessStmt_of_checks {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {rows : Option (Array Flock.Registered.Port)}
    (hck : ∀ R ω t R₂ (s : Fin S), (∃ tr, AcceptsZKR (zs R ω t R₂ s).I (zs R ω t R₂ s).own tr) →
      ∃ tr, FlockC3.PubRows.StmtsChecked ish s (bind s) rows (zs R ω t R₂ s).I (zs R ω t R₂ s).own tr)
    (hF : ∀ R ω t R₂ s, TagsFound (zs R ω t R₂ s).I) : SessStmt ish bind zs dg N k Rw := by
  intro R ω t R₂ _ _ hA s
  obtain ⟨hacc, -⟩ := hA
  obtain ⟨_, _, sts, hs, hfm⟩ := hck R ω t R₂ s ⟨_, (hacc s).1⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c (zs R ω t R₂ s).ht (zs R ω t R₂ s).hTags.1 (zs R ω t R₂ s).hcH hs
  obtain ⟨htag, htab, ss, sv, hss, hj, -, -, hcf, -, hfl⟩ :=
    FlockSoundness.Discharge.Recursion.checkStmt_ok (array_forM_ok hfm 0 h0)
  rw [hc0] at hfl
  have hT : (zs R ω t R₂ s).I.tags = Flock.Tags.circuit := by
    have h : Flock.Tags.find? (zs R ω t R₂ s).I.tags.name = some (zs R ω t R₂ s).I.tags := hF R ω t R₂ s
    rw [htag, find?_circuit] at h
    exact (Option.some.inj h).symm
  have hZ := (zs R ω t R₂ s).hc
  rw [parse_zkTags_vstar hT, htab] at hZ
  exact ⟨ss, sv, _, hss, hj, hfl, hcf, hZ⟩

/-- **A `rec-opening` part's session has one unit** (with `OpenUpv`, one copy): the session's circuit is V*'s file's
parse for its statement, so its units are the statement's instances (`VStarParsesScoped`), and the statement is named
as its `rec-opening` unit, so it is one of R's opening parts, of one instance (`instances_of_openingName`). -/
theorem nv_of_copy {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    (hst : SessStmt ish bind zs dg N k Rw) {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {s : Fin S}
    {un : Fin (Vs s).nv} {j p : ℕ} (hO : OpenCopy zs dg N k Rw sh w gs s un j p) : (Vs s).nv = 1 := by
  obtain ⟨R, ω, t, R₂, τ, o, _, hA, _, _, -, -, -, m, cl, hpm, -⟩ := hO
  obtain ⟨ss, sv, cf, hss, hj, hf, hcf, hc⟩ := hst R ω t R₂ τ o hA s
  rw [(hob s ss sv cf _ hss hj hcf hc (Vs s).hscope).2]
  exact instances_of_openingName hss (Array.mem_of_getElem? hj)
    ((openingName_params hpm).trans (name_of_filter hf).2.symm)

/-- **One session per part** (named, V*'s list): the copies registering w at accepted outer proofs as part `p` are in
one session. -/
def OpenSess (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ gs (s s' : Fin S) (un : Fin (Vs s).nv) (un' : Fin (Vs s').nv) (j j' p : ℕ),
    OpenCopy zs dg N k Rw sh w gs s un j p → OpenCopy zs dg N k Rw sh w gs s' un' j' p → s = s'

/-- **`OpenSess` from V*'s list**: copies at part `p` of the opening at `sh` are units named `unitName sh.m
sh.claims p` (`openingName_params`, `shape?`'s `m` and claims), and the list has one statement so named, at one
index (`idx_eq_of_filter`), each session's. -/
theorem openSess_of {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    (hst : SessStmt ish bind zs dg N k Rw) : OpenSess zs dg N k Rw sh w := by
  intro gs s s' un un' j j' p hc hc'
  obtain ⟨R, ω, t, R₂, τ, o, _, hA, _, _, -, -, -, m, cl, hpm, hsh, -⟩ := hc
  obtain ⟨R', ω', t', R₂', τ', o', _, hA', _, _, -, -, -, m', cl', hpm', hsh', -⟩ := hc'
  obtain ⟨ss, sv, -, hss, hj, hf, -⟩ := hst R ω t R₂ τ o hA s
  obtain ⟨ss', sv', -, hss', hj', hf', -⟩ := hst R' ω' t' R₂' τ' o' hA' s'
  obtain rfl : ss = ss' := Except.ok.inj (hss.symm.trans hss')
  obtain rfl : m = m' := (openShape_m hsh).symm.trans (openShape_m hsh')
  obtain rfl : cl = cl' := (FlockC3.Opening.shape_claims hsh).symm.trans (FlockC3.Opening.shape_claims hsh')
  rw [← (openingName_params hpm).symm.trans (openingName_params hpm')] at hf'
  exact Fin.ext (idx_eq_of_filter ss.toList hf s s' sv sv' (toList_getElem_of hj) (toList_getElem_of hj')
    (name_of_filter hf).2 (name_of_filter hf').2)

/-- **`OneCopy` from V*'s list and the verifier's `rec-opening` check**: copies at part `p` are in one session
(`openSess_of`) of one unit (`nv_of_copy`) of one copy (`OpenUpv`). -/
theorem oneCopy_of {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs)
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    (hst : SessStmt ish bind zs dg N k Rw) : OneCopy zs dg N k Rw sh w := by
  intro gs s s' un un' j j' p hc hc'
  obtain rfl := openSess_of (sh := sh) (w := w) hst gs s s' un un' j j' p hc hc'
  have hn := nv_of_copy hob hst hc
  obtain ⟨_, _, _, _, _, _, _, -, _, hj, -, hchk, -, m, cl, hpm, -⟩ := hc
  obtain ⟨_, _, _, _, _, _, _, -, _, hj', -⟩ := hc'
  have h1 := openUpv _ hchk m cl p hpm
  refine ⟨rfl, ?_, ?_⟩
  · rw [Nat.lt_one_iff.mp (lt_of_lt_of_eq un.isLt hn), Nat.lt_one_iff.mp (lt_of_lt_of_eq un'.isLt hn)]
  · rw [Nat.lt_one_iff.mp (lt_of_lt_of_eq hj h1), Nat.lt_one_iff.mp (lt_of_lt_of_eq hj' h1)]

/-- **`OpenAll` from V*'s sessions**: `SessionAt`'s (a) has a copy registering w at each part of the groups of the
opening's structure, which are the ones `structureOf` and `groups` give. -/
theorem openAll_of_session {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128} {unit : String}
    {c3 : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Flock.RecCoins.C3}
    {ops : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → FlockVBridge.AccChain.Openings}
    (hsess : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ) zs dg N k Rw R ω t R₂ τ o →
      FlockC3.Opening.SessionAt (zs R ω t R₂) (layers zs dg N k Rw R ω t R₂ τ) sh unit
        (c3 R ω t (layers zs dg N k Rw R ω t R₂ τ)) (w R ω t (layers zs dg N k Rw R ω t R₂ τ))
        (ops R ω t (layers zs dg N k Rw R ω t R₂ τ))) :
    OpenAll zs dg N k Rw sh w := by
  intro R ω t R₂ τ o hA S0 gs hS hg p hp
  obtain ⟨_, -, -, -, ⟨S', gs', sess, un, cp, hS', hg', hparts⟩, -⟩ := hsess R ω t R₂ τ o hA
  obtain rfl : S0 = S' := Except.ok.inj (hS.symm.trans hS')
  obtain rfl : gs = gs' := Except.ok.inj (hg.symm.trans hg')
  exact ⟨sess p, un p, cp p, _, hparts p hp⟩

/-- **A copy's ring-switch column is a keyed input** (`OpeningZeroAtL`'s `RegBit`, read by `readCol_rsAt`): at a copy
registering w as part `p`, bit `b` of the part's row-port element `q` holding ring switch `i`'s slot `t` is a column
below the unit's inputs' end, keyed at copy `j` by a bit of a row port. -/
theorem openCopy_key {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {s : Fin S} {un : Fin (Vs s).nv} {j p : ℕ}
    (hO : OpenCopy zs dg N k Rw sh w gs s un j p) {S0 : RecAlgebra.GfS} (hS : RecOpening.structureOf sh = .ok S0)
    (hg : Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs) (hpg : p < gs.size) {S₁ : RecOpen.GfS}
    {ns : List ℕ} {nv : ℕ} (hpart : RecOpening.part sh p = .ok (S₁, ns, nv)) {q e : ℕ}
    (he : (wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e) {i t : ℕ}
    (hwi : RecOpening.wi sh (.rs i t) = some e) {b : ℕ} (hb : b < 128) :
    128 * q + b < 128 * ns.sum ∧ rowCol (partRep ns nv).ports 0 (128 * q + b) < (Rows.ofNet (Vs s).hU).nIn ∧
      ∃ (q' : Fin (Vs s).c.ports.size) (kk : ℕ) (_ : kk < Layout.portRowW (Vs s).c q'),
        Aliased.srcKey (Vs s).c (j * (Rows.ofNet (Vs s).hU).nIn + rowCol (partRep ns nv).ports 0 (128 * q + b)) =
          .msg q'.val (kk / 8) (kk % 8) := by
  obtain ⟨_, _, _, _, _, _, _, -, hz⟩ := hO
  have hrd := readCol_rsAt _ hz hS hg hpg hpart he hwi hb
  obtain ⟨_, _, _, _, _, _, _, _, _, S₂, ns₂, nv₂, hpart₂, hw, -⟩ := hz
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  obtain ⟨-, hns, -⟩ := part_eqOpening hS hg hpg hpart
  obtain ⟨hk, -⟩ := Array.getElem?_eq_some_iff.mp he
  have hq : q < ns.sum := by rw [hns, ← wIdx_size]; exact hk
  obtain ⟨hcol, q', kk, hkk, hkey, -⟩ := hw (128 * q + b) (by omega) hrd
  exact ⟨by omega, hcol, q', kk, hkk, hkey⟩

/-- **Hypothesis on the reads file: stage `c₀ + i` reads the ports keying ring switch `i`'s columns** (`SelPort`, as
S's `RowSel`): at a copy of the `rec-opening` unit registering w as part `p`, a row port `q'` whose bit keys copy `j`'s
column of a bit of the part's row-port element holding ring switch `i`'s slot `t` is read by stage `c₀ + i` at the
copy's unit. -/
def RsSel (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (c₀ : ℕ) : Prop :=
  ∀ i < 2, ∀ gs (s : Fin S) (un : Fin (Vs s).nv) (j p : ℕ), OpenCopy zs dg N k Rw sh w gs s un j p →
    ∀ (S0 : RecAlgebra.GfS), RecOpening.structureOf sh = .ok S0 →
      Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs →
    ∀ (hpg : p < gs.size) (S₁ : RecOpen.GfS) (ns : List ℕ) (nv : ℕ), RecOpening.part sh p = .ok (S₁, ns, nv) →
      ∀ (q e : ℕ) (t : Fin 128), (wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e →
        RecOpening.wi sh (.rs i t.val) = some e → ∀ b < 128, ∀ (q' : Fin (Vs s).c.ports.size) (kk : ℕ),
          kk < Layout.portRowW (Vs s).c q' →
          Aliased.srcKey (Vs s).c (j * (Rows.ofNet (Vs s).hU).nIn + rowCol (partRep ns nv).ports 0 (128 * q + b)) =
            .msg q'.val (kk / 8) (kk % 8) → SelPort Vs own sel (c₀ + i) s un q'

/-- **`RsSk` from the reads file**: a ring switch's wire is at the commit string of the port keying its column
(`openCopy_key`, `posK_of_keyAt`), which stage `c₀ + i` reads (`RsSel`, `mem_skOf_inl`). -/
theorem rsSk_of_sel {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hsel : RsSel zs dg N k Rw sh w own sel c₀) : RsSk zs dg N k Rw sh w own sel c₀ := by
  intro i hi s un g hW
  obtain ⟨gs, j, p, hj, hpg, S0, S₁, ns, nv, q, t, e, b, hc, hO, hS, hg, hpart, he, hwi, hb, rfl⟩ := hW
  obtain ⟨-, hcol, q', kk, hkk, hkey⟩ := openCopy_key hO hS hg hpg hpart he hwi hb
  rw [posK_of_keyAt un hj hcol hkk hkey hc]
  exact mem_skOf_inl.2 (hsel i hi gs s un j p hO S0 hS hg hpg S₁ ns nv hpart q e t he hwi b hb q' kk hkk hkey)

/-- **The `rec-opening` unit reads its ring switches' columns** (S's `ReadCol`, `rsReadCol`): at a copy registering w as
part `p`, a column of bit `b` of the part's row-port element `q` holding ring switch `i`'s slot `t`, an input column
keyed at copy `j` by a bit of a row port `q'` (`openCopy_key`'s), is one a computed row of the unit reads. A part's
other row-port columns need not be: touched ports bring whole ports into a part. -/
def RsReadCol (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ i < 2, ∀ gs (s : Fin S) (un : Fin (Vs s).nv) (j p : ℕ), OpenCopy zs dg N k Rw sh w gs s un j p →
    ∀ (S0 : RecAlgebra.GfS), RecOpening.structureOf sh = .ok S0 →
      Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs → ∀ (hpg : p < gs.size) (S₁ : RecOpen.GfS)
        (ns : List ℕ) (nv : ℕ), RecOpening.part sh p = .ok (S₁, ns, nv) →
      ∀ (q e : ℕ) (t : Fin 128), (wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e →
        RecOpening.wi sh (.rs i t.val) = some e → ∀ b < 128, 128 * q + b < 128 * ns.sum →
          rowCol (partRep ns nv).ports 0 (128 * q + b) < (Rows.ofNet (Vs s).hU).nIn →
          ∀ (q' : Fin (Vs s).c.ports.size) (kk : ℕ), kk < Layout.portRowW (Vs s).c q' →
          Aliased.srcKey (Vs s).c (j * (Rows.ofNet (Vs s).hU).nIn + rowCol (partRep ns nv).ports 0 (128 * q + b)) =
            .msg q'.val (kk / 8) (kk % 8) →
          ReadCol (Rows.ofNet (Vs s).hU) (rowCol (partRep ns nv).ports 0 (128 * q + b))

/-- **The `rec-opening` unit reads its ring switches' columns** (`RsReadCol`): `OpenCopy` carries the opening's check,
which refuses a unit reading an element of a claim's ring switch in fewer than 128 bits (`readCol_rsAt`). -/
theorem rsReadCol {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128} :
    RsReadCol zs dg N k Rw sh w := by
  intro i _ gs s un j p hO S0 hS hg hpg S₁ ns nv hpart q e t he hwi b hb _ _ _ _ _ _
  obtain ⟨_, _, _, _, _, _, _, -, hz⟩ := hO
  exact readCol_rsAt _ hz hS hg hpg hpart he hwi hb

/-- **`RsIo` from the unit reading its ring switches' columns** (`openCopy_key`, `rsReadCol`, `mem_io_of_readCol`). -/
theorem rsIo_of_readCol {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128} : RsIo zs dg N k Rw sh w := by
  intro i hi s un g hW
  obtain ⟨gs, j, p, hj, hpg, S0, S₁, ns, nv, q, t, e, b, hc, hO, hS, hg, hpart, he, hwi, hb, rfl⟩ := hW
  obtain ⟨hq, hcol, q', kk, hkk, hkey⟩ := openCopy_key hO hS hg hpg hpart he hwi hb
  exact mem_io_of_readCol un hj hcol
    (rsReadCol i hi gs s un j p hO S0 hS hg hpg S₁ ns nv hpart q e t he hwi b hb hq hcol q' kk hkk hkey) hc

/-- **R's ring switches are read at their rounds** (`RsRead`) from V*'s statements and sessions, the verifier's
`rec-opening` check of one copy, V*'s sessions' copies at every part and the reads file at the ring switches' ports. -/
theorem rsRead_of_facts {ish : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs) (hst : SessStmt ish bind zs dg N k Rw)
    (hall : OpenAll zs dg N k Rw sh w) (hsel : RsSel zs dg N k Rw sh w own sel c₀) :
    RsRead zs dg N k Rw sh w own sel c₀ :=
  rsRead_of_layer (oneCopy_of hob hst) hall (rsSk_of_sel hsel) rsIo_of_readCol

/-! ## `RsSel` at the round commitments -/

/-- Discharged by the verifier's `rec-opening` check (`RecResiduals.misnamed` at `RecOpening.rsName`, `rsKeys`). **Ring
switch `i`'s columns are keyed by port `m<i>`**: at a
copy of the `rec-opening` unit registering w as part `p`, a row port `q'` whose bit keys copy `j`'s column of a bit of
the part's row-port element holding ring switch `i`'s slot `t` is named `m<i>`. -/
def RsKeys (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ i < 2, ∀ gs (s : Fin S) (un : Fin (Vs s).nv) (j p : ℕ), OpenCopy zs dg N k Rw sh w gs s un j p →
    ∀ (S0 : RecAlgebra.GfS), RecOpening.structureOf sh = .ok S0 →
      Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs →
    ∀ (hpg : p < gs.size) (S₁ : RecOpen.GfS) (ns : List ℕ) (nv : ℕ), RecOpening.part sh p = .ok (S₁, ns, nv) →
      ∀ (q e : ℕ) (t : Fin 128), (wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e →
        RecOpening.wi sh (.rs i t.val) = some e → ∀ b < 128, ∀ (q' : Fin (Vs s).c.ports.size) (kk : ℕ),
          kk < Layout.portRowW (Vs s).c q' →
          Aliased.srcKey (Vs s).c (j * (Rows.ofNet (Vs s).hU).nIn + rowCol (partRep ns nv).ports 0 (128 * q + b)) =
            .msg q'.val (kk / 8) (kk % 8) → (Vs s).c.ports[q'].name = s!"m{i}"

/-- **Hypothesis on the reads file and the stage selector: a copy's unit reads round `i`'s commitment at port `m<i>`,
where stage `c₀ + i` reads it**: at the session and unit of a copy registering w, where the session's circuit has a
port named `m<i>` (`RsKeys`' conclusion), the reads file has an entry named `m<i>` of value `rec-opening-c<i>` (round
`i`'s commitment, `rec_vstage.opening_messages`) whose read of the unit is at a position `x` that stage `c₀ + i` reads
of that value. -/
def RoundReads (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128)
    (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (c₀ : ℕ) : Prop :=
  ∀ i < 2, ∀ gs (s : Fin S) (un : Fin (Vs s).nv) (j p : ℕ), OpenCopy zs dg N k Rw sh w gs s un j p →
    ∀ q : Fin (Vs s).c.ports.size, (Vs s).c.ports[q].name = s!"m{i}" →
    ∃ pt ∈ own s, pt.name = s!"m{i}" ∧ pt.value = s!"rec-opening-c{i}" ∧
      ∃ x, pt.reads[un.val]? = some (some x) ∧ sel (c₀ + i) pt.value x

/-- **Hypothesis on the reads file and the stage selector: a part's unit reads round `m`'s commitment at port `m<m>`,
where rep `ρ`'s stage of round `m` reads it** (`RoundReads` at the residuals' parts): at the session of part `p`, where
its circuit has a port named `m<m>`, the reads file has an entry named `m<m>` of value `rec-c<m>` (round `m`'s
commitment) whose read of rep `ρ`'s unit is at a position `x` that `stage ρ m` reads of that value. -/
def WRoundReads (sh : Flock.Shape) (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (stage : Fin 2 → ℕ → ℕ) (sess : ℕ → Fin S) (un : (p : ℕ) → Fin 2 → Fin (Vs (sess p)).nv) : Prop :=
  ∀ p < (FlockVBridge.Family.partsOf sh).size, ∀ (m : ℕ) (ρ : Fin 2) (q : Fin (Vs (sess p)).c.ports.size),
    (Vs (sess p)).c.ports[q].name = s!"m{m}" →
    ∃ pt ∈ own (sess p), pt.name = s!"m{m}" ∧ pt.value = s!"rec-c{m}" ∧
      ∃ x, pt.reads[(un p ρ).val]? = some (some x) ∧ sel (stage ρ m) pt.value x

/-- **`RsSel` at the round commitments**: the port keying a ring switch's column is its round's (`RsKeys`), whose entry
in the reads file is that round's commitment, read at the unit where the round's stage reads it (`RoundReads`). -/
theorem rsSel_of_rounds {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hk : RsKeys zs dg N k Rw sh w) (hr : RoundReads zs dg N k Rw sh w own sel c₀) :
    RsSel zs dg N k Rw sh w own sel c₀ := by
  intro i hi gs s un j p hO S0 hS hg hpg S₁ ns nv hpart q e t he hwi b hb q' kk hkk hkey
  have hq := hk i hi gs s un j p hO S0 hS hg hpg S₁ ns nv hpart q e t he hwi b hb q' kk hkk hkey
  obtain ⟨pt, hpt, hn, -, x, hx, hsx⟩ := hr i hi gs s un j p hO q' hq
  exact ⟨pt, hpt, hn.trans hq.symm, x, hx, hsx⟩

/-! ## `SessStmt` from the verifier's checks -/

/-- **The checked sessions' inputs carry the tags their name names** where the prover's do (the guarantee's `_hF`): a
checked session's inputs are those of a session of its index as the prover supplies it (`zsOf` builds it, or another,
refused; `keepOwn` and `refuse` keep the inputs). -/
theorem tagsFound_checked {ish : RecVStar.Inner} {cs : List (Cm × (ℐ).Coin) → Option Flock.RecVStar.InnerCommits}
    {coins : List (Cm × (ℐ).Coin) → Option Flock.RecCoins.Source} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg'}
    {hC : Flock.Premises.RecordCustody zr} {h : Typed zr} (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (j : Fin S) :
    TagsFound (Flock.Guarantees.checkedC3 ish cs coins bind rows zr hC h R ω t R₂ j).I := by
  show TagsFound (Flock.Guarantees.custodied zr hC h R ω t R₂ j).I
  dsimp only [Flock.Guarantees.custodied]
  by_cases hx : (zr R ω t R₂ j).Typable
  · rw [zsOf_pos zr _ h R ω t R₂ j hx]
    exact hF R ω t R₂ j
  · rw [zsOf_neg zr _ h R ω t R₂ j hx]
    exact hF (h j).choose.1 (h j).choose.2.1 (h j).choose.2.2.1 (h j).choose.2.2.2 j

/-- **`SessStmt` where the verifier's checks pass** (`ChecksAccepted`'s statement check, `sessStmt_of_checks`) and the
sessions' inputs carry the tags their name names. -/
theorem sessStmt_of_accepted {xR : HoloReg GF128 Dig} {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) (hF : ∀ R ω t R₂ j, TagsFound (F.zs R ω t R₂ j).I)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} : SessStmt F.ish F.bind F.zs dg N k Rw :=
  sessStmt_of_checks (rows := rows) (fun R ω t R₂ s h => by
    obtain ⟨tr, -, -, hs⟩ := hca R ω t R₂ s h
    exact ⟨tr, hs⟩) hF

/-- **`SessStmt` at the checked family** (`C3Family.checked`) from the prover's sessions' tags (the guarantee's `_hF`):
where a checked session accepts, the verifier's checks pass (`checksAccepted_checked`), and the checked sessions carry
the prover's inputs (`tagsFound_checked`). -/
theorem sessStmt_checked {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray} {N k Rw : ℕ} :
    SessStmt ish bind (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw :=
  sessStmt_of_accepted
    (F := C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
    FlockC3.PubRows.checksAccepted_checked
    (tagsFound_checked (ish := ish) (cs := Flock.Guarantees.innerHeld wr st reg) (coins := fun t => coins (wr t).2)
      (bind := bind) (rows := rows) (hC := hCust) (h := h) hF)

/-- **R's ring switches are read at their rounds at a family whose checks pass** (`rsRead_of_facts`): `SessStmt` from
the verifier's checks and the inputs' tags (`sessStmt_of_accepted`), `RsSel` at the round commitments
(`rsSel_of_rounds`). -/
theorem rsRead_of_accepted {xR : HoloReg GF128 Dig} {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hca : FlockC3.PubRows.ChecksAccepted F rows wr reg) (hF : ∀ R ω t R₂ j, TagsFound (F.zs R ω t R₂ j).I)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Array F128}
    {own : Fin (vstarCount F.ish) → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hall : OpenAll F.zs dg N k Rw sh w) (hk : RsKeys F.zs dg N k Rw sh w)
    (hr : RoundReads F.zs dg N k Rw sh w own sel c₀) : RsRead F.zs dg N k Rw sh w own sel c₀ :=
  rsRead_of_facts F.hob (sessStmt_of_accepted hca hF) hall (rsSel_of_rounds hk hr)

/-- **R's ring switches are read at their rounds at the checked family** (`C3Family.checked`, `rsRead_of_accepted`):
the checked sessions pass the verifier's checks (`checksAccepted_checked`) and carry the prover's inputs
(`tagsFound_checked`, from the guarantee's `_hF`). -/
theorem rsRead_checked {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128} {c₀ : ℕ}
    (hall : OpenAll (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w)
    (hk : RsKeys (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w)
    (hr : RoundReads (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w own sel c₀) :
    RsRead (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w own sel c₀ :=
  rsRead_of_accepted
    (F := C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg)
    FlockC3.PubRows.checksAccepted_checked
    (tagsFound_checked (ish := ish) (cs := Flock.Guarantees.innerHeld wr st reg) (coins := fun t => coins (wr t).2)
      (bind := bind) (rows := rows) (hC := hCust) (h := h) hF)
    hall hk hr

/-- **R's opening's committed rounds at the checked family** (`OpenMsgsRead`, `openMsgsRead_of_layer`): the sparse
phases' part ends at `c₀` coins (`PartEnds`), R's opening is tied to V*'s layers (`OpenTieL`), and its ring switches are
read at their rounds (`rsRead_checked`). -/
theorem openMsgsRead_checked {xR : HoloReg GF128 Dig} {ish : RecVStar.Inner} {ss : Array VStar.Stmt}
    {hss : RecVStar.statements ish = .ok ss} {bind : Fin (vstarCount ish) → VStar.Bindings}
    {Vs : Fin (vstarCount ish) → VStmt} {hob : VStarParsesScoped Flock.Tags.circuit ish bind Vs}
    {zr : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin (vstarCount ish)) → (Vs j).OuterR Reg'}
    {hCust : Flock.Premises.RecordCustody zr} {h : Typed zr} {coins : String → Option Flock.RecCoins.Source}
    {rows : Option (Array Flock.Registered.Port)} {own : Fin (vstarCount ish) → Array Flock.Registered.Port}
    {sel : ℕ → String → ℕ → Prop}
    {bc : (j : Fin (vstarCount ish)) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8}
    {dec : ℕ → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → (ℐ).Msg}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I) {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → Array F128} {c₀ : ℕ}
    {src : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount ish)) → (Vs j).Bits) → C3Run (holoAt st hp)}
    (he : PartEnds sparsePart
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs dg N k Rw src c₀)
    (htie : ∀ R ω t R₂ τ o, FlockVBridge.Compiled.Accepted (I := ℐ)
      (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs dg N k Rw
        R ω t R₂ τ o →
      FlockC3.Opening.OpenTieL
        ((C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs R ω t R₂)
        (layers (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
          dg N k Rw R ω t R₂ τ) sh
        (w R ω t (layers
          (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
          dg N k Rw R ω t R₂ τ))
        (src R ω t (layers
          (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
          dg N k Rw R ω t R₂ τ)))
    (hall : OpenAll (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w)
    (hk : RsKeys (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w)
    (hr : RoundReads (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw sh w own sel c₀) :
    OpenMsgsRead (C3Family.checked (xR := xR) ish hss bind hob zr hCust h coins rows own sel bc dec wr reg).zs
      dg N k Rw src own sel :=
  openMsgsRead_of_layer he htie (rsRead_checked hF hall hk hr)

end

end FlockC3.Compose
