import Proofs.Flock.C3.Reps.AtList
import Proofs.Flock.C3.Reps.ReadWhole
import Proofs.Flock.C3.Reps.RoundLinks
import Proofs.Flock.Recursive.HPortTie
import Proofs.Flock.VBridge.Family.PartKeys
import Proofs.Flock.VBridge.Algebra.RingSwitch

/-!
# The algebra unit reads rep `i`'s `h` at `m<hr>` (`hAlgRows_of_res`)

`HAlgRows` (`HPortTie`), proved from the verifier's checks of V*'s holo list and each algebra part's `ResZeroAtLH`, the
decoder tie `AlgebraDecodes` is proved from (`RepMaps.algebraDecodes`):

* The list (`hRound_held`, `sparse_stmt_of_name`): at a holo shape `RecVStar.statements` refuses a list none of whose
  groups holds `h`'s round `hRound m kLog` as a row port, and its RecSparse statement is `.sparse _ (hRound m kLog)`.
* The layout (`rounds_holo`, `wi_h`, `portSizes_h`, `starts_h`, `portNames_h`): at a holo shape `h`'s round is the
  64-element round `m<hRound>` after the zerocheck's and lincheck's (`zlRounds`), `h l` its `l`-th element of `w`.
* The part (`readCol_h`, `rowBit_at`, `hRow_of_part`): `RecResiduals.check` refuses a holo part that reads an element of
  `h` in fewer than 128 bits (`hUnread`), so each bit of the part's `m<hRound>` row is a computed row's input, which
  `ResZeroAtLH`'s `hw` registers at the bit of `w` the part's `wIdx` puts there, `h l`'s.
* `hAlgRows_of_res`: at every accepted outcome, rep `i`'s unit (`RepUnitOfAlg`) has `U.d` the decoder's rep, whose `h`
  (`hOfW`) is `w`'s at `wi`, so the part's row at unit `i` holds it; `hAlgRows_withReads` at the reads' decoder
  (`RepMaps.withReads`, `resZeroAtLH_wRead`).
-/

namespace FlockC3.Reps

namespace HAlg

open Flock FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

/-! ## The list -/

/-- **A holo list holds `h`'s round in a group**: `RecVStar.statements` refuses a holo shape none of whose groups holds
`hRound m kLog` as a row port. -/
theorem hRound_held {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    (hh : i.holo = true) :
    ∃ sh0, RecResiduals.shape? i.m i.kLog i.claims = some sh0 ∧
      ∃ GS gs, RecAlgebra.structureOf { sh0 with holo := i.holo } = .ok GS ∧
        RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := i.holo }) = .ok gs ∧
        ∃ p, ∃ _ : p < gs.size, RecVStar.hRound i.m i.kLog ∈ gs[p].2.1 := by
  unfold RecVStar.statements at h
  dsimp only at h
  split at h
  rotate_left
  · cases h
  rename_i sh0 hsh
  split at h
  · cases h
  obtain ⟨ps, -, h⟩ := FlockC3.VStarList.eok_bind h
  obtain ⟨gs, hgs, h⟩ := FlockC3.VStarList.eok_bind h
  split at h
  · cases h
  split at h
  · cases h
  rename_i hany
  obtain ⟨GS, hGS, hgs⟩ := FlockC3.VStarList.eok_bind (ok_of_mapError hgs)
  rw [hh] at hany
  simp only [Bool.true_and, Bool.not_eq_true', Bool.not_eq_false] at hany
  obtain ⟨p, hp, hc⟩ : ∃ p, ∃ _ : p < gs.size, RecVStar.hRound i.m i.kLog ∈ gs[p].2.1 := by simpa using hany
  exact ⟨sh0, hsh, GS, gs, hGS, hgs, p, hp, hc⟩

/-- **`checkStmt`'s statement and its parse and part checks**: the session's unit is the list's statement `j`, by name,
and `checkParsed` and `checkPart` pass on it. -/
theorem parsed_of_checkStmt {i : RecVStar.Inner} {tags : Flock.Tags} {tables : List (String × ByteArray)}
    {st : Flock.Stmt} {circuitB : ByteArray} {partition : Option Lean.Json} {program : Option ByteArray}
    {own : Array Flock.Registered.Port} {j : ℕ} {f : VStar.Bindings} {rows : Option (Array Flock.Registered.Port)}
    (h : RecVStar.checkStmt (some i) tags tables st circuitB partition program (some own) (some j) (some f)
      rows = .ok ()) :
    ∃ ss s, RecVStar.statements i = .ok ss ∧ ss[j]? = some s ∧ s.name = st.c.unit.name ∧
      RecVStar.checkParsed s st.c = .ok () ∧ RecVStar.checkPart s st.c = .ok () := by
  unfold RecVStar.checkStmt at h
  dsimp only at h
  split at h
  · obtain ⟨_, h1, -⟩ := Refine.ebind_ok h
    cases h1
  · split at h
    · obtain ⟨_, h1, -⟩ := Refine.ebind_ok h
      cases h1
    · split at h
      · split at h
        · cases h
        obtain ⟨_, -, h⟩ := Refine.ebind_ok h
        obtain ⟨s, hs, -⟩ := Refine.ebind_ok h
        obtain ⟨hck, -, hpa, hpt⟩ := checkCircuit_ok hs
        obtain ⟨ss, hss, hj, hname, -⟩ := check_ok hck
        exact ⟨ss, s, hss, hj, hname, hpa, hpt⟩
      · cases h

/-- **The list's statement named as a RecSparse unit is its RecSparse statement**, at `h`'s round (`sparse_of_name`). -/
theorem sparse_stmt_of_name {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss)
    {s : VStar.Stmt} (hs : s ∈ ss) {sh : RecSparse.Shape} (hn : RecSparse.unitName sh = s.name) :
    ∃ sp, s = .sparse sp (RecVStar.hRound i.m i.kLog) := by
  obtain ⟨n1, n2, n3, n4, n5⟩ := FlockC3.PubRows.sparse_ne_names sh
  obtain ⟨vsh, A, C, K, L, N, hA, hC, hK, -, hL, -, rfl, -⟩ := FlockC3.VStarList.statements_ok h
  rcases Array.mem_append.1 hs with hs | hs
  rotate_left
  · rcases FlockC3.VStarList.tail_cases hK hL N _ hs with ⟨_, _, rfl⟩ | ⟨_, _, _, rfl⟩
    · exact absurd hn (n5 _ _)
    · exact absurd hn (FlockC3.PubRows.sparse_ne_linkName _ _ _)
  rcases Array.mem_append.1 hs with hs | hs
  · obtain ⟨_, _, _, -, -, -, rfl⟩ := FlockVBridge.Family.statementsOf_eq hA
    simp only [Array.mem_append, List.mem_toArray, List.mem_map, List.mem_range] at hs
    rcases hs with ⟨l, -, rfl⟩ | ⟨j, -, rfl⟩
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
        exact ⟨sp, hs⟩
      · rcases hrest s hs with ⟨l, L, H, n, rfl⟩ | ⟨m, c, k, S, ports, nv, rfl⟩
        · exact absurd hn (n3 _ _)
        · exact absurd hn (n4 _ _ _)

/-! ## The layout of `h`'s round -/

/-- The rounds before `h`'s at a holo shape: `RecAlgebra.preRounds` without its `h` round. -/
def zlRounds (sh : Flock.Shape) : Array (Array RecAlgebra.Key) :=
  #[(Array.range 64).map .ab ++ (Array.range 64).map .c] ++
    (Array.range (sh.m - 6)).map (fun i => #[.g1 i, .ginf i]) ++ #[#[.fa, .fb]] ++
    (Array.range (sh.kLog - 6)).map (fun j => #[.q1 j, .qinf j]) ++ #[(Array.range 64).map .zp]

theorem rounds_holo {sh : Flock.Shape} (hh : sh.holo = true) :
    RecAlgebra.rounds sh = zlRounds sh ++ #[(Array.range 64).map .h] ++
      ((Array.range sh.claims).map (fun k => (Array.range 128).map (.rs k)) ++ RecAlgebra.ligRounds sh.sched) := by
  rw [RecAlgebra.rounds_split]
  simp only [RecAlgebra.preRounds, RecAlgebra.hRound, hh, ite_true, zlRounds, Array.append_assoc]

/-- `h`'s round is round `hRound m kLog`. -/
theorem zlRounds_size (sh : Flock.Shape) (hm : 6 ≤ sh.m) (hk : 6 ≤ sh.kLog) :
    (zlRounds sh).size = RecVStar.hRound sh.m sh.kLog := by
  simp only [zlRounds, RecVStar.hRound, Array.size_append, Array.size_map, Array.size_range]
  simp
  omega

theorem h_not_zl (sh : Flock.Shape) (l : ℕ) : RecAlgebra.Key.h l ∉ (zlRounds sh).flatten := by
  simp [zlRounds, Array.mem_flatten]

theorem idxOf?_mid {xs hs ys : Array RecAlgebra.Key} {a : RecAlgebra.Key} {i : ℕ} (hx : a ∉ xs)
    (hh : hs[i]? = some a) (hlt : ∀ j < i, hs[j]? ≠ some a) : (xs ++ hs ++ ys).idxOf? a = some (xs.size + i) := by
  have hi := (Array.getElem?_eq_some_iff.1 hh).1
  apply RecAlgebra.idxOf?_of
  · rw [Array.getElem?_append_left (by simp only [Array.size_append]; omega),
      Array.getElem?_append_right (by omega), Nat.add_sub_cancel_left]
    exact hh
  · intro j hj he
    rw [Array.getElem?_append_left (by simp only [Array.size_append]; omega)] at he
    by_cases hjx : j < xs.size
    · rw [Array.getElem?_append_left hjx] at he
      exact hx (Array.mem_of_getElem? he)
    · rw [Array.getElem?_append_right (by omega)] at he
      exact hlt (j - xs.size) (by omega) he

/-- **`h l` is `w`'s element `l` of `h`'s round**. -/
theorem wi_h {sh : Flock.Shape} (hh : sh.holo = true) {l : ℕ} (hl : l < 64) :
    RecAlgebra.wi sh (.h l) = some ((zlRounds sh).flatten.size + l) := by
  unfold RecAlgebra.wi RecAlgebra.wNames
  rw [rounds_holo hh]
  simp only [Array.flatten_append, Array.append_assoc]
  rw [← Array.append_assoc]
  refine idxOf?_mid (h_not_zl sh l) ?_ fun j hj he => ?_
  · simp [hl]
  · simp only [Array.flatten_singleton, Array.getElem?_map, Array.getElem?_range] at he
    split at he <;> simp at he
    omega

theorem portSizes_h {sh : Flock.Shape} (hh : sh.holo = true) :
    (RecResiduals.portSizes sh)[(zlRounds sh).size]! = 64 := by
  unfold RecResiduals.portSizes
  rw [rounds_holo hh]
  simp [getElem!_pos, Array.getElem_push]

theorem starts_h {sh : Flock.Shape} (hh : sh.holo = true) :
    (RecResiduals.starts (RecResiduals.portSizes sh))[(zlRounds sh).size]! = (zlRounds sh).flatten.size := by
  rw [FlockVBridge.Family.starts_get _ (by unfold RecResiduals.portSizes; rw [rounds_holo hh]; simp; omega)]
  unfold RecResiduals.portSizes
  rw [rounds_holo hh, Array.size_flatten]
  simp [← Array.sum_toList]

theorem portNames_h {sh : Flock.Shape} (hh : sh.holo = true) :
    (RecVStar.portNames sh)[(zlRounds sh).size]! = s!"m{(zlRounds sh).size}" := by
  unfold RecVStar.portNames
  rw [rounds_holo hh, getElem!_pos _ _ (by simp; omega), Array.getElem_append_left (by simp; omega)]
  simp

/-! ## The part's row of `h` -/

theorem readBits_of_hUnread {wn : Array RecAlgebra.Key} {W : Array ℕ} {rd : Array Bool} {ns : List ℕ}
    (h : RecResiduals.hUnread wn W rd ns = none) {q : ℕ} (hq : q < W.size) {l : ℕ}
    (hk : wn[W[q]!]? = some (.h l)) : RecResiduals.readBits rd ns q = 128 := by
  unfold RecResiduals.hUnread at h
  have := List.filterMap_eq_nil_iff.1 (List.head?_eq_none_iff.1 h) q (List.mem_range.2 hq)
  simp only [hk] at this
  by_cases h128 : RecResiduals.readBits rd ns q = 128
  · exact h128
  · simp only [bne_iff_ne, ne_eq, h128, not_false_eq_true, ↓reduceIte, reduceCtorEq] at this

open FlockVBridge.OfVStar (ReadCol rowCol partRep) in
/-- **A holo part reads each bit of its elements of `h`** (`readCol_rs`'s at `h`): `RecResiduals.check`'s `hUnread`
refuses an element of `h` read in fewer than 128 bits. -/
theorem readCol_h {c : Flock.Circuit} (hU : Built c.unit) (h : RecResiduals.check c = .ok ())
    (hu : c.unit.unitLog ≤ 32) {m k cl p : ℕ} (hp : RecResiduals.params c.unit.name = some (m, k, cl, p))
    {sh : Flock.Shape} (hsh : RecResiduals.shape? m k cl = some sh) (hh : RecResiduals.holoOf c = .ok true)
    {S0 : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecAlgebra.structureOf { sh with holo := true } = .ok S0)
    (hg : RecResiduals.groups S0 (RecResiduals.portSizes { sh with holo := true }) = .ok gs) (hpg : p < gs.size)
    {S₁ : RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part { sh with holo := true } p = .ok (S₁, ns, nv)) {q e : ℕ}
    (he : (FlockVBridge.PartS.wIdx (RecResiduals.portSizes { sh with holo := true }) gs[p].2.1)[q]? = some e)
    {l : ℕ} (hwi : RecAlgebra.wi { sh with holo := true } (.h l) = some e) {b : ℕ} (hb : b < 128) :
    ReadCol (Rows.ofNet hU) (rowCol (partRep ns nv).ports 0 (128 * q + b)) := by
  obtain ⟨S₂, ns₂, nv₂, hpart₂, -, -, W, hW, hun, -⟩ := ReadWhole.check_whole h hp hsh hh
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  obtain rfl := ReadWhole.partW_eqR hS hg hpg hW
  obtain ⟨hq, hqe⟩ := Array.getElem?_eq_some_iff.1 he
  have hk : (RecAlgebra.wNames { sh with holo := true })[(FlockVBridge.PartS.wIdx
      (RecResiduals.portSizes { sh with holo := true }) gs[p].2.1)[q]!]? = some (.h l) := by
    rw [getElem!_pos _ q hq, hqe]
    exact ReadWhole.getElem?_of_wiA hwi
  obtain ⟨-, r, hr, hr', hc⟩ := (ReadWhole.getD_readIns c.unit _).1
    (ReadWhole.getD_of_readBits (readBits_of_hUnread hun hq hk) hb)
  rw [ReadWhole.rowCol_eq] at hc
  exact ReadWhole.readCol_of_row hU hu hr hr' hc

/-- **A part's rows are its statement's row ports, in order** (`srcKey_laid`), from its parse and `checkParsed`. -/
theorem repKeys_of {tags : Flock.Tags} {c : Flock.Circuit} (hU : Built c.unit) (P : ParseFacts tags c)
    {p : ℕ} {nm : String} {pt : VStar.Part} (hpa : RecVStar.checkParsed (.inner p nm pt) c = .ok ())
    {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ} {name : String}
    (hn : Flock.HmNets.netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit) :
    FlockVBridge.OfVStar.RepKeys c pt.S := by
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨u, hu, hupv, -, hlv, hports⟩ := checkParsed_ok hpa
  obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok (show VStar.innerUnit pt.S nm = .ok u from hu)
  unfold RecVStar.vstarRows at hports
  rw [VStar.UnitC.inWords, hin] at hlv
  rw [hin] at hports
  intro q hq
  exact FlockVBridge.OfVStar.srcKey_laid hU hn (check_facts P.typed hls).leaves hupv hlv hports
    (by
      show (pt.S.ports.map (·.1) ++ ["v"]).length = (FlockVBridge.OfVStar.repPorts pt.S).length
      rw [FlockVBridge.OfVStar.repPorts_length, List.length_append, List.length_map, List.length_singleton]) hq

/-- **The row key of bit `e` of element `l` of row port `ih`** is the circuit's port `ih`, named as the port, at bit
`128 l + e` (`srcKey_row`, `take_sum_inj`). -/
theorem rowBit_at {c : Flock.Circuit} {S : VStar.RepS} (h : FlockVBridge.OfVStar.RepKeys c S) (n : ℕ)
    {sizes P : Array ℕ} {names : Array String} (hns : S.ports.map (·.2) = P.toList.map (sizes[·]!))
    (hnm : S.ports.map (·.1) = P.toList.map (names[·]!)) {ih l e : ℕ} (hih : ih < P.size)
    (hl : l < sizes[P[ih]]!) (he : e < 128) {q : Fin c.ports.size} {b : ℕ}
    (hk : Aliased.srcKey c (0 * n + FlockVBridge.OfVStar.rowCol S.ports 0
      (128 * (((P.toList.map (sizes[·]!)).take ih).sum + l) + e)) = .msg q.val (b / 8) (b % 8)) :
    q.val = ih ∧ b = 128 * l + e ∧ c.ports[q].name = names[P[ih]]! := by
  have hih' : ih < (P.toList.map (sizes[·]!)).length := by simpa using hih
  have hnsi : (P.toList.map (sizes[·]!))[ih] = sizes[P[ih]]! := by simp
  have hq : 128 * (((P.toList.map (sizes[·]!)).take ih).sum + l) + e < 128 * (S.ports.map (·.2)).sum := by
    rw [hns]
    have := FlockVBridge.OfVStar.take_sum_add_le (P.toList.map (sizes[·]!)) hih'
    omega
  obtain ⟨i, hi, e0, he0, hqe, hname, -, hk0⟩ := FlockVBridge.OfVStar.srcKey_row h n hq
  rw [hk0] at hk
  simp only [Aliased.Src.msg.injEq] at hk
  obtain ⟨hiq, h1, h2⟩ := hk
  have hb : b = e0 := by omega
  have hlen : S.ports.length = P.size := by simpa using congrArg List.length hns
  have hi' : i.val < (P.toList.map (sizes[·]!)).length := by simp; omega
  have hpi : S.ports[i.val].2 = (P.toList.map (sizes[·]!))[i.val] := by
    have h' := List.getElem_of_eq hns (by rw [List.length_map]; exact hi)
    rw [List.getElem_map] at h'
    exact h'
  rw [hpi] at he0
  rw [hns] at hqe
  obtain ⟨hii, hle⟩ := FlockVBridge.OfVStar.take_sum_inj (P.toList.map (sizes[·]!)) hih' hi'
    (by rw [hnsi]; exact hl) (show e0 / 128 < _ by omega) (show _ = _ by omega)
  have hqi : q = i := Fin.ext hiq.symm
  subst hqi
  refine ⟨hii.symm, by omega, ?_⟩
  have hni : S.ports[q.val].1 = names[P[ih]]! := by
    have h' := List.getElem_of_eq hnm (by rw [List.length_map]; exact hi)
    rw [List.getElem_map, List.getElem_map, Array.getElem_toList] at h'
    rw [h']
    congr 2
    exact hii.symm
  exact hname.trans hni

open FlockVBridge (wbit bitOf) in
open FlockVBridge.Decode (regVal) in
open FlockVBridge.OfVStar (ResZeroAtLH) in
/-- **A part holding `h`'s round registers it at unit `u`**: where `ResZeroAtLH` holds at a holo part whose row ports
hold `h`'s round at index `ih`, the circuit's port `m<hRound>`, a row port of its statement, is read at unit `u`, and its
bit `128 l + e` is bit `e` of `w`'s element `h l` (`readCol_h`, then `ResZeroAtLH`'s `hw` at that bit, `rowBit_at`). -/
theorem hRow_of_part {V : VStmt} {Reg' : Type} (zo : V.Outer Reg') {p' : ℕ} {nm : String} {pt : VStar.Part}
    (hpa : RecVStar.checkParsed (.inner p' nm pt) V.c = .ok ())
    (hpt : RecVStar.checkPart (.inner p' nm pt) V.c = .ok ()) {bits : V.Bits} {u : Fin V.nv} {sh : Flock.Shape}
    (hh : sh.holo = true) {GS : RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hS : RecAlgebra.structureOf sh = .ok GS) (hg : RecResiduals.groups GS (RecResiduals.portSizes sh) = .ok gs)
    {p : ℕ} (hpg : p < gs.size) {wf : ℕ → F128} {rounds : Array (Array F128)} {extras : Array Flock.Claim}
    {comb : Array F128} (hR : ResZeroAtLH zo bits u 0 sh gs p wf rounds extras comb) {ih : ℕ}
    (hih : ih < gs[p].2.1.size) (hx : gs[p].2.1[ih] = (zlRounds sh).size) :
    ∃ q' : Fin V.c.ports.size, V.c.ports[q'].name = s!"m{(zlRounds sh).size}" ∧
      s!"m{(zlRounds sh).size}" ∈ RecVStar.rowPorts (.inner p' nm pt) ∧ zo.rd.isRead (.inl (u, q')) ∧
      ∀ (l : Fin 64) (e : ℕ), e < 128 → ∃ hb : 128 * l.val + e < Layout.portRowW V.c q',
        (regVal zo (.inl (u, q'))).2
            ⟨128 * l.val + e, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
          bitOf (wf ((zlRounds sh).flatten.size + l.val)) e := by
  obtain ⟨hpg', -, -, hc, hho, hu, m, k, cl, hpar, sh0, hsh0, rfl, S, ns, nv, hpart, hw, -, -⟩ := hR
  obtain ⟨-, hnsE, -⟩ := FlockVBridge.OfVStar.part_eq hS hg hpg hpart
  obtain ⟨hns, -, hnm⟩ := checkPart_ok hpt rfl hpar hsh0 hho hpart
  have hnm' := hnm _ (FlockVBridge.OfVStar.partPorts_eq hS hg hpg)
  have hns' := hns.trans hnsE
  obtain ⟨S', ns', nv', -, hn⟩ := FlockVBridge.OfVStar.check_part_holo hc hpar hsh0 hho
  have hK := repKeys_of V.hU (parse_facts zo.hc) hpa hn
  have hsz : (RecResiduals.portSizes { sh0 with holo := true })[gs[p].2.1[ih]]! = 64 := by
    rw [hx]; exact portSizes_h hh
  have hih' : ih < (gs[p].2.1.toList.map ((RecResiduals.portSizes { sh0 with holo := true })[·]!)).length := by
    simpa using hih
  have hle := FlockVBridge.OfVStar.take_sum_add_le
    (gs[p].2.1.toList.map ((RecResiduals.portSizes { sh0 with holo := true })[·]!)) hih'
  have hnsi : (gs[p].2.1.toList.map ((RecResiduals.portSizes { sh0 with holo := true })[·]!))[ih] = 64 := by
    simpa using hsz
  have hWs := FlockVBridge.OfVStar.wIdx_size (RecResiduals.portSizes { sh0 with holo := true }) gs[p].2.1
  have key : ∀ (l : Fin 64) (e : ℕ), e < 128 → ∃ (q₁ : Fin V.c.ports.size)
      (hk : 128 * l.val + e < Layout.portRowW V.c q₁), q₁.val = ih ∧
      V.c.ports[q₁].name = (RecVStar.portNames { sh0 with holo := true })[gs[p].2.1[ih]]! ∧
      zo.rd.isRead (.inl (u, q₁)) ∧
      (regVal zo (.inl (u, q₁))).2
          ⟨128 * l.val + e, lt_of_lt_of_le hk ((Finset.le_sup (Finset.mem_univ q₁)).trans (Nat.le_succ _))⟩ =
        bitOf (wf ((zlRounds { sh0 with holo := true }).flatten.size + l.val)) e := by
    intro l e he
    have ha : ((gs[p].2.1.toList.map ((RecResiduals.portSizes { sh0 with holo := true })[·]!)).take ih).sum + l.val <
        (FlockVBridge.PartS.wIdx (RecResiduals.portSizes { sh0 with holo := true }) gs[p].2.1).size := by
      have := l.isLt
      omega
    have hwg := FlockVBridge.OfVStar.wIdx_get (RecResiduals.portSizes { sh0 with holo := true }) gs[p].2.1 hih
      (by rw [hsz]; exact l.isLt) ha rfl
    rw [hx, starts_h hh] at hwg
    have he' := (Array.getElem?_eq_getElem ha).trans (congrArg some hwg)
    have hrc := readCol_h V.hU hc hu hpar hsh0 hho hS hg hpg hpart he' (wi_h hh l.isLt) he
    have hq : 128 * (((gs[p].2.1.toList.map ((RecResiduals.portSizes { sh0 with holo := true })[·]!)).take ih).sum +
        l.val) + e < 128 * ns.sum := by
      rw [hnsE]; omega
    obtain ⟨-, q₁, k₁, hk₁, hkey, hrd, hval⟩ := hw _ hq hrc
    rw [FlockVBridge.OfVStar.rowCol_snd (FlockVBridge.OfVStar.partRep ns nv).ports pt.S.ports
      ((FlockVBridge.OfVStar.partRep_snd ns nv).trans hns.symm)] at hkey
    obtain ⟨hq₁, rfl, hname⟩ := rowBit_at hK _ hns' hnm' hih (by rw [hsz]; exact l.isLt) he hkey
    refine ⟨q₁, hk₁, hq₁, hname, hrd, hval.trans ?_⟩
    rw [FlockVBridge.wbit_elem _ _ he, FlockVBridge.Decode.sub_lt _ _ _ ha, hwg]
  obtain ⟨q₀, -, hq₀, hname₀, hrd₀, -⟩ := key 0 0 (by omega)
  refine ⟨q₀, by rw [hname₀, hx, portNames_h hh], ?_, hrd₀, fun l e he => ?_⟩
  · show _ ∈ pt.S.ports.map (·.1)
    rw [hnm']
    exact List.mem_map.2 ⟨_, Array.getElem_mem_toList hih, by rw [hx]; exact portNames_h hh⟩
  · obtain ⟨q₁, hk₁, hq₁, -, -, hv⟩ := key l e he
    obtain rfl : q₁ = q₀ := Fin.ext (hq₁.trans hq₀.symm)
    exact ⟨hk₁, hv⟩

end HAlg

section family

open Flock FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg FlockLevel3 Accounting ZkBind Integrate Placed
open StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.PubRows (ChecksAccepted)
open FlockC3.Compose
open HAlg

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **Each accepted session's statement passes the parse and part checks**: `ChecksAccepted`'s statement check at the
session's circuit (`setup_zero_c`, `parsed_of_checkStmt`). -/
theorem stmtAt_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (s : Fin (vstarCount F.ish)) :
    ∃ sv, F.stmts[(s : ℕ)]? = some sv ∧ sv.name = (F.Vs s).c.unit.name ∧
      RecVStar.checkParsed sv (F.Vs s).c = .ok () ∧ RecVStar.checkPart sv (F.Vs s).c = .ok () := by
  obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ s ⟨_, (h.1 s).1⟩
  obtain ⟨h0, hc0⟩ := setup_zero_c (F.zs R ω t R₂ s).ht (F.zs R ω t R₂ s).hTags.1 (F.zs R ω t R₂ s).hcH hset
  obtain ⟨ss, sv, hss, hj, hname, hpa, hpt⟩ := parsed_of_checkStmt (array_forM_ok hfm 0 h0)
  rw [hc0] at hname hpa hpt
  obtain rfl : ss = F.stmts := Except.ok.inj (hss.symm.trans F.hstmts)
  exact ⟨sv, hj, hname, hpa, hpt⟩

open FlockVBridge.OfVStar (ResZeroAtLH) in
/-- **`HAlgRows` from the verifier's checks and the parts' decoder tie**: at a holo inner shape, where each algebra
part holds `ResZeroAtLH` at the decoder's rep (`hres`, the premise `RepMaps.algebraDecodes` proves `AlgebraDecodes`
from) and rep `i`'s unit is the decoder's rep (`RepUnitOfAlg`), a RecSparse copy's session is `.sparse _ (hRound m kLog)`
(`sparse_stmt_of_name`), some group holds `h`'s round (`hRound_held`), and that group's part reads it at unit `i` with
bit `128 l + e` bit `e` of `hOfW U.sh U.d.w`'s element `l` (`hRow_of_part`, `wi_h`, `hOf_holo`), the part's session at
two units (`VStarParsesScoped`). -/
theorem hAlgRows_of_res {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    (hF : ∀ R ω t R₂ j, TagsFound (F.zs R ω t R₂ j).I) (hholo : F.ish.holo = true) {sh0 : Flock.Shape}
    (hsh : RecResiduals.shape? F.ish.m F.ish.kLog F.ish.claims = some sh0) {stmt : Fin (vstarCount F.ish) → VStar.Stmt}
    (M : RepMaps stmt F.Vs { sh0 with holo := F.ish.holo } Reg L.Ω (Cm × (ℐ).Coin)) {dg : ByteArray → ByteArray}
    {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → C3Run (holoAt st hp)}
    (halg : AlgebraDecodes (I := ℐ) F.zs dg N k Rw { sh0 with holo := F.ish.holo } M.rep)
    (hacc : AccChains (I := ℐ) F.zs dg N k Rw { sh0 with holo := F.ish.holo } M.rep)
    (hU : RepUnitOfAlg F.zs dg N k Rw (tiedSrc stp) { sh0 with holo := F.ish.holo } M.rep)
    (hres : ∀ GS gs, RecAlgebra.structureOf { sh0 with holo := F.ish.holo } = .ok GS →
      RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := F.ish.holo }) = .ok gs →
      ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ p < gs.size,
        ResZeroAtLH (F.zs R ω t R₂ (M.sess p)) (layers F.zs dg N k Rw R ω t R₂ τ (M.sess p)) (M.un p r)
          (FlockVBridge.Family.cp p r) { sh0 with holo := F.ish.holo } gs p
          (RecAlgebra.el (M.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w)
          (M.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).rounds
          (M.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).extras
          (M.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 →
      Array Flock.F128) :
    FlockC3.HPort.HAlgRows F dg N k Rw stp w := by
  intro R ω t R₂ τ o hA i
  obtain ⟨U, hUsh, hUd⟩ := hU R ω t R₂ τ o hA i (halg R ω t R₂ τ o hA i) (hacc R ω t R₂ τ o hA i)
  refine ⟨U, fun s un j hj c hcp _ => ?_⟩
  obtain ⟨sv, hsv, hname, -, -⟩ := stmtAt_checked hchk hA s
  obtain ⟨sp, rfl⟩ := sparse_stmt_of_name F.hstmts (Array.mem_of_getElem? hsv)
    ((FlockVBridge.Sparse.unitName_params hcp.2.1).trans hname.symm)
  refine ⟨sp, _, hsv, ?_⟩
  obtain ⟨sh1, hsh1, GS, gs, hS, hg, p, hpg, hmem⟩ := hRound_held F.hstmts hholo
  obtain rfl : sh0 = sh1 := Option.some.inj (hsh.symm.trans hsh1)
  obtain ⟨ih, hih, hx⟩ := Array.getElem_of_mem hmem
  obtain ⟨sv', hsv', hname', hpa, hpt⟩ := stmtAt_checked hchk hA (M.sess p)
  obtain ⟨-, -, -, -, -, -, m, kk, cl, hpar, -⟩ := hres GS gs hS hg R ω t R₂ τ o hA i p hpg
  obtain ⟨p', nm, pt, rfl⟩ := inner_of_resName F.hstmts (Array.mem_of_getElem? hsv')
    ((FlockVBridge.Family.resUnitName_params hpar).trans hname'.symm)
  obtain ⟨ss, sv'', cf, hss, hj'', -, hcf, hc⟩ := sessStmt_of_accepted hchk hF R ω t R₂ τ o hA (M.sess p)
  obtain rfl : ss = F.stmts := Except.ok.inj (hss.symm.trans F.hstmts)
  obtain rfl : sv'' = .inner p' nm pt := Option.some.inj (hj''.symm.trans hsv')
  have hnv : (F.Vs (M.sess p)).nv = 2 := (F.hob _ _ _ cf _ hss hj'' hcf hc (F.Vs (M.sess p)).hscope).2
  have hh : ({ sh0 with holo := F.ish.holo } : Flock.Shape).holo = true := hholo
  obtain ⟨hf, hm, hk, -, h6, -⟩ := FlockVBridge.Family.shape?_eq hsh
  have h22 := (FlockVBridge.Family.fast100_range hf).1
  have hzl : RecVStar.hRound F.ish.m F.ish.kLog = (zlRounds { sh0 with holo := F.ish.holo }).size := by
    rw [zlRounds_size _ (by show 6 ≤ sh0.m; omega) (by show 6 ≤ sh0.kLog; omega)]
    show _ = RecVStar.hRound sh0.m sh0.kLog
    rw [hm, hk]
  obtain ⟨q', hq'n, hrp, hrd, hbits⟩ := hRow_of_part (F.zs R ω t R₂ (M.sess p)) hpa hpt hh hS hg hpg
    (hres GS gs hS hg R ω t R₂ τ o hA i p hpg) hih (hx.trans hzl)
  rw [← hzl] at hq'n hrp
  refine ⟨M.sess p, M.un p i, q', _, M.hun p i (by rw [hnv]; exact i.isLt), hsv', hq'n, hrp, hrd,
    fun l e he => ?_⟩
  obtain ⟨hb, hv⟩ := hbits l e he
  refine ⟨hb, hv.trans ?_⟩
  rw [hUsh, hUd, RecAlgebra.hOfW_get _ _ l.isLt, RecAlgebra.hOf_holo hh _ (wi_h hh l.isLt), RecAlgebra.getElem!_el]

open FlockVBridge.OfVStar (ResZeroAtLH) in
/-- **`HAlgRows` at the reads' decoder** (`RepMaps.withReads`, the decoder `withReads_algebraDecodes_atList` proves
`AlgebraDecodes` at): `hAlgRows_of_res`, each part's `ResZeroAtLH` at `wRead`'s `w` from one `w` per rep at which every
part holds it (`resZeroAtLH_wRead`). -/
theorem hAlgRows_withReads {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂}
    {rows : Option (Array Flock.Registered.Port)} {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hchk : ChecksAccepted F rows wr reg) (hF : ∀ R ω t R₂ j, TagsFound (F.zs R ω t R₂ j).I)
    (hholo : F.ish.holo = true) {sh0 : Flock.Shape}
    (hsh : RecResiduals.shape? F.ish.m F.ish.kLog F.ish.claims = some sh0) {stmt : Fin (vstarCount F.ish) → VStar.Stmt}
    (M : RepMaps stmt F.Vs { sh0 with holo := F.ish.holo } Reg L.Ω (Cm × (ℐ).Coin))
    {wh : List (Cm × (ℐ).Coin) → Lean.Json × String} {held : Lean.Json × String → Except String RecVStar.InnerCommits}
    {s : Fin 2 → String} {comb : ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 → Fin 64 → F128}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    {stp : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → C3Run (holoAt st hp)}
    (halg : AlgebraDecodes (I := ℐ) F.zs dg N k Rw { sh0 with holo := F.ish.holo } (M.withReads wh held s comb).rep)
    (hacc : AccChains (I := ℐ) F.zs dg N k Rw { sh0 with holo := F.ish.holo } (M.withReads wh held s comb).rep)
    (hU : RepUnitOfAlg F.zs dg N k Rw (tiedSrc stp) { sh0 with holo := F.ish.holo } (M.withReads wh held s comb).rep)
    (hrc : ResReadCol F.Vs { sh0 with holo := F.ish.holo } M.sess)
    (hw : ∀ GS gs, RecAlgebra.structureOf { sh0 with holo := F.ish.holo } = .ok GS →
      RecResiduals.groups GS (RecResiduals.portSizes { sh0 with holo := F.ish.holo }) = .ok gs →
      ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ ρ : Fin 2,
        ∃ w : ℕ → F128, ∀ p < gs.size, ResZeroAtLH (F.zs R ω t R₂ (M.sess p))
          (layers F.zs dg N k Rw R ω t R₂ τ (M.sess p)) (M.un p ρ) 0 { sh0 with holo := F.ish.holo } gs p w
          (heldRounds (held (wh t)) (s ρ)) (M.extras R ω t ρ) (Array.ofFn (comb (layers F.zs dg N k Rw R ω t R₂ τ) ρ)))
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Fin 2 →
      Array Flock.F128) :
    FlockC3.HPort.HAlgRows F dg N k Rw stp w :=
  hAlgRows_of_res hchk hF hholo hsh (M.withReads wh held s comb) halg hacc hU
    (fun GS gs hS hg R ω t R₂ τ o hA ρ _ hp => by
      obtain ⟨w, hw⟩ := hw GS gs hS hg R ω t R₂ τ o hA ρ
      exact resZeroAtLH_wRead hrc (fun j => (hA.1 j).2.2) hS hg ρ hw hp) w

end family

end FlockC3.Reps
