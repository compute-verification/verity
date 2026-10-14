import Proofs.Flock.VBridge.Family.OutCols
import Proofs.Flock.C3.RegLevels.OpensRegs

/-!
# A part's input keys, from V*'s META

`RecVStar.checkParsed` (`VStarFamily.hparsed`) holds a part circuit's input leaves and row ports to the ones V* wrote,
so the Δ key (`Aliased.srcKey`, copy 0) of each of the unit's input bits is the bit of V*'s port it lies in:

* `vstarRows_get`: the ports parsed as V*'s input rows have their words and bits, and start where V*'s do;
* `srcKey_laid`: at the verifier's `unitLaid`, the key of input bit `16 · (words before q) + k` is bit `k` of port `q`
  (`srcKey_inOrder`);
* `repKeys_part`: part `p`'s circuit has those keys at V*'s part `pt.S`'s ports (`RepKeys`);
* `srcKey_v`: so the `v` column `vBase + q` is bit `q` of the port V* names `v`.

`PartVCols` reads the keys at the verifier's part `RecResiduals.part F.sh p` (`partRep`), not at V*'s `pt.S`: that the
two have the same ports' words (`nsOf F.sh p`, `nvOf F.sh p`) is not proved here, nor are the row columns' keys
(`rowCol`, which `PartRowCols` and `PartRowBits` read).
-/

namespace FlockVBridge.OfVStar

open Flock.HmNets hiding and
open Flock FlockSoundness.Discharge.Hm
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge

/-- **V*'s input rows, port by port**: ports whose (name, words, bits) begin with V*'s input rows of words `ws` have
their names, words and 16 bits a word, and start past the words of the rows before them. -/
theorem vstarRows_get {ports : Array Flock.Port} {ins : List String} {ws : List ℕ} {rest : List (String × ℕ × ℕ)}
    (h : ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      (ins.zip ws).map (fun (nm, w) => (nm, w, 16 * w)) ++ rest)
    (hlen : ins.length = ws.length) {q : ℕ} (hq : q < ws.length) :
    ∃ hq' : q < ports.size, ports[q].name = ins[q]! ∧ ports[q].words = ws[q] ∧ ports[q].bits = 16 * ws[q] ∧
      ((ports.toList.take q).map (·.words)).sum = (ws.take q).sum := by
  have hz : (ins.zip ws).length = ws.length := by rw [List.length_zip, hlen, Nat.min_self]
  have hsz : ws.length ≤ ports.size := by
    have := congrArg List.length h
    rw [List.length_map, List.length_append, List.length_map, hz, Array.length_toList] at this
    omega
  refine ⟨by omega, ?_⟩
  have hget : ∀ i (hi : i < ws.length), (ports[i].name, ports[i].words, ports[i].bits) =
      (ins[i]!, ws[i], 16 * ws[i]) := by
    intro i hi
    have e := congrArg (·[i]?) h
    simp only at e
    have hzi : i < (ins.zip ws).length := by rw [hz]; exact hi
    have hpi : i < ports.toList.length := by rw [Array.length_toList]; omega
    rw [List.getElem?_map, List.getElem?_append_left (by rw [List.length_map, hz]; exact hi), List.getElem?_map,
      List.getElem?_eq_getElem hzi, List.getElem?_eq_getElem hpi, List.getElem_zip, Array.getElem_toList,
      Option.map_some, Option.map_some, Option.some.injEq] at e
    rw [e, getElem!_pos ins i (by omega)]
  have hh := hget q hq
  simp only [Prod.mk.injEq] at hh
  refine ⟨hh.1, hh.2.1, hh.2.2, congrArg List.sum (List.ext_getElem
    (by simp only [List.length_map, List.length_take, Array.length_toList]; omega) fun i h1 h2 => ?_)⟩
  simp only [List.length_map, List.length_take] at h1
  have hi := hget i (by omega)
  simp only [Prod.mk.injEq] at hi
  simp only [List.getElem_map, List.getElem_take, Array.getElem_toList]
  exact hi.2.1

theorem take_sum_add_le (l : List ℕ) {q : ℕ} (hq : q < l.length) : (l.take q).sum + l[q] ≤ l.sum := by
  have e : l.sum = (l.take (q + 1)).sum + (l.drop (q + 1)).sum := by rw [← List.sum_append, List.take_append_drop]
  rw [e, List.take_add_one, List.sum_append, List.getElem?_eq_getElem hq]
  simp only [Option.toList_some, List.sum_cons, List.sum_nil]
  omega

/-- **The verifier's `rec-residuals` unit keys its inputs by the rows V* wrote**: at `unitLaid` (one group of 16-bit
ports from column 0), with one unit per VU whose leaves are its input words in order and ports V*'s input rows of
words `ws`, the key of input bit `16 · (ws before q) + k` is bit `k` of port `q` (`srcKey_inOrder`). -/
theorem srcKey_laid {c : Flock.Circuit} (hU : Built c.unit) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    {name : String} (hn : netOf name (RecResiduals.unitLaid S ns nv) = .ok c.unit)
    (hleaves : c.leavesIn.size = c.upv ∧
      ∀ u < c.upv, (c.leavesIn[u]!).size = (c.unit.inGroups.getD 0 default).bits.size) (hupv : c.upv = 1)
    {ins : List String} {ws : List ℕ} {rest : List (String × ℕ × ℕ)}
    (hlv : c.leavesIn.getD 0 #[] = (Array.range (ws.foldl (· + ·) 0)).map Int.ofNat)
    (hports : c.ports.toList.map (fun p => (p.name, p.words, p.bits)) =
      (ins.zip ws).map (fun (nm, w) => (nm, w, 16 * w)) ++ rest)
    (hlen : ins.length = ws.length) {q : ℕ} (hq : q < ws.length) :
    ∃ hq' : q < c.ports.size, c.ports[q].name = ins[q]! ∧ c.ports[q].words = ws[q] ∧
      ∀ k < 16 * ws[q], Aliased.srcKey c (16 * (ws.take q).sum + k) = .msg q (k / 8) (k % 8) := by
  obtain ⟨hq', hnm, hw, hb, hsum⟩ := vstarRows_get hports hlen hq
  refine ⟨hq', hnm, hw, fun k hk => ?_⟩
  have hg : c.unit.inGroups = #[⟨0, groupWords ((partIn ns nv).foldl (· + ·) 0), partIn ns nv⟩] :=
    (Flock.Net.ofRows_ok hn).2.2.2.2.2.1.trans (layout_inGroups _ _ _ _)
  have h16 : ∀ b ∈ (partIn ns nv).toList, b = 16 := fun b hb => by
    unfold partIn at hb
    rw [Array.toList_replicate] at hb
    exact List.eq_of_mem_replicate hb
  have h0 : 0 < c.leavesIn.size := by rw [hleaves.1, hupv]; exact Nat.one_pos
  have hl0 := hleaves.2 0 (by rw [hupv]; exact Nat.one_pos)
  rw [hg, getElem!_pos c.leavesIn 0 h0] at hl0
  rw [Array.getD_eq_getD_getElem?, getElem?_pos c.leavesIn 0 h0, Option.getD_some] at hlv
  have hsz : ws.foldl (· + ·) 0 = (partIn ns nv).size := by
    rw [hlv, Array.size_map, Array.size_range] at hl0
    exact hl0
  have hN : 16 * (partIn ns nv).size ≤ c.unit.inWords * Flock.WORD := by
    rw [inWords_unitLaid hn, inRowsOf]
    unfold partIn
    rw [foldl_replicate, Array.size_replicate, Nat.mul_comm 16]
    exact le_groupWords _
  have hst : (FlockSoundness.Typed.starts c)[q]! = (ws.take q).sum := by
    rw [FlockSoundness.Discharge.Layout.starts_get c hq', hsum]
  have hfit : (FlockSoundness.Typed.starts c)[q]! + c.ports[q].words ≤ (partIn ns nv).size := by
    rw [hst, hw, ← hsz, foldl_add, Nat.zero_add]
    exact take_sum_add_le ws hq
  have key := FlockC3.RegLevels.srcKey_inOrder hg rfl h16 (Aliased.portBits_nodup hU) (j := 0)
    (by rw [Array.getD_eq_getD_getElem?, getElem?_pos c.leavesIn 0 h0, Option.getD_some, hlv, hsz])
    hq' (by rw [hw]; exact hk) (by rw [hb]; exact hk) hfit hN
  rw [hst, Nat.zero_mul, Nat.zero_add] at key
  exact key

/-- **A circuit keys V*'s part `S`'s input ports** (`InnerRepCheck{S}`'s, `repPorts`): port `q` has V*'s name and words,
and the key of input bit `16 · (words before q) + k` is its bit `k`. -/
def RepKeys (c : Flock.Circuit) (S : VStar.RepS) : Prop :=
  ∀ q (hq : q < (repPorts S).length), ∃ hq' : q < c.ports.size,
    c.ports[q].name = (S.ports.map (·.1) ++ ["v"])[q]! ∧ c.ports[q].words = (repPorts S)[q] ∧
      ∀ k < 16 * (repPorts S)[q], Aliased.srcKey c (16 * ((repPorts S).take q).sum + k) = .msg q (k / 8) (k % 8)

theorem repPorts_length (S : VStar.RepS) : (repPorts S).length = S.ports.length + 1 := by
  rw [repPorts, List.length_append, List.length_map, List.length_singleton]

theorem repPorts_take (S : VStar.RepS) {i : ℕ} (hi : i ≤ S.ports.length) :
    (repPorts S).take i = (S.ports.map fun p => VStar.portWordsOf p.2).take i := by
  rw [repPorts, List.take_append_of_le_length (by rw [List.length_map]; exact hi)]

theorem repPorts_v (S : VStar.RepS) (h : S.ports.length < (repPorts S).length) :
    (repPorts S)[S.ports.length] = VStar.portWordsOf S.gf.nv := by
  simp only [repPorts, List.getElem_append, List.length_map, Nat.lt_irrefl, dite_false, Nat.sub_self,
    List.getElem_singleton]

theorem names_v (S : VStar.RepS) : (S.ports.map (·.1) ++ ["v"])[S.ports.length]! = "v" := by
  rw [getElem!_pos _ _ (by rw [List.length_append, List.length_map, List.length_singleton]; omega),
    List.getElem_append_right (by simp only [List.length_map, Nat.le_refl])]
  simp only [List.length_map, Nat.sub_self, List.getElem_singleton]

/-- **`v`'s columns at V*'s part**: port `S.ports.length` is `v`, and the key of column `vBase S + q`, for `q` within
`v`'s `128 nv` bits, is its bit `q`, within its row. -/
theorem srcKey_v {c : Flock.Circuit} {S : VStar.RepS} (h : RepKeys c S) (n : ℕ) :
    ∃ qv : Fin c.ports.size, c.ports[qv].name = "v" ∧ ∀ q < 128 * S.gf.nv, ∃ _ : q < Layout.portRowW c qv,
      Aliased.srcKey c (0 * n + (vBase S + q)) = .msg qv.val (q / 8) (q % 8) := by
  have hl : S.ports.length < (repPorts S).length := by rw [repPorts_length]; omega
  obtain ⟨hq', hnm, hw, hkey⟩ := h _ hl
  have hv : 16 * ((repPorts S).take S.ports.length).sum = vBase S := by
    rw [repPorts_take S (Nat.le_refl _), List.take_of_length_le (by simp only [List.length_map, Nat.le_refl]), vBase]
  have hb : ∀ q < 128 * S.gf.nv, q < 16 * (repPorts S)[S.ports.length] := fun q hq => by
    rw [repPorts_v S hl]
    exact Nat.lt_of_lt_of_le hq (le_portWords _)
  refine ⟨⟨_, hq'⟩, hnm.trans (names_v S), fun q hq => ⟨?_, ?_⟩⟩
  · show q < 16 * c.ports[S.ports.length].words
    rw [hw]
    exact hb q hq
  · rw [Nat.zero_mul, Nat.zero_add, ← hv]
    exact hkey q (hb q hq)

end FlockVBridge.OfVStar

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **Part `p`'s circuit keys V*'s part `pt.S`'s ports** (`srcKey_laid`): its unit is the verifier's `unitLaid`
(`check_part`, at `HUnitName`), and its leaves and row ports are the ones V* wrote (`hparsed`, `checkParsed_ok`), whose
input rows are `pt.S`'s (`innerUnit_ok`). -/
theorem repKeys_part (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) : OfVStar.RepKeys (F.Vs j).c pt.S := by
  have hname := F.hUnitName_of hne
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts _ hj)
  have P := parse_facts (F.zs R ω [] R₂ j).hc
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := P.recResiduals
  obtain ⟨S, ns, nv, -, hn⟩ :=
    FlockVBridge.OfVStar.check_part hchk (resParams_of_check hchk hnm) F.hsh (F.hplain j)
  obtain ⟨ls, hls⟩ := P.check
  obtain ⟨u, hu, hupv, -, hlv, hports⟩ := checkParsed_ok (F.hparsed j)
  rw [hj] at hu hports
  obtain ⟨-, -, hin, -⟩ := FlockVBridge.OfVStar.innerUnit_ok (show VStar.innerUnit pt.S nm = .ok u from hu)
  unfold RecVStar.vstarRows at hports
  rw [VStar.UnitC.inWords, hin] at hlv
  rw [hin] at hports
  intro q hq
  exact OfVStar.srcKey_laid (F.Vs j).hU hn (check_facts P.typed hls).leaves hupv hlv hports
    (by
      show (pt.S.ports.map (·.1) ++ ["v"]).length = (OfVStar.repPorts pt.S).length
      rw [OfVStar.repPorts_length, List.length_append, List.length_map, List.length_singleton]) hq

/-- **V*'s part is the verifier's part**: part `p`'s statement's row ports' elements and v elements are those of the
verifier's part `p` at its shape (`hpart`, `checkPart_ok`, at the unit name and plain META `check_part` reads). -/
theorem part_counts (hne : Nonempty (Reg × L.Ω × Reg₂)) {p : ℕ} {j : Fin F.stmts.size} {nm : String}
    {pt : VStar.Part} (hj : F.stmts[j] = .inner p nm pt) {S : Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (hpart : RecResiduals.part F.sh p = .ok (S, ns, nv)) : pt.S.ports.map (·.2) = ns ∧ pt.S.gf.nv = nv := by
  have hname := F.hUnitName_of hne
  obtain ⟨⟨R, ω, R₂⟩⟩ := hne
  have hnm : (F.Vs j).c.unit.name = RecResiduals.unitName F.vsh.m F.vsh.kLog F.vsh.claims p :=
    (hname j p nm pt hj).trans (statements_name F.hstmts _ hj)
  have hchk : RecResiduals.check (F.Vs j).c = .ok () := (parse_facts (F.zs R ω [] R₂ j).hc).recResiduals
  have e : ∀ s : Flock.Shape, s.holo = false → { s with holo := false } = s := by
    intro s h0
    cases s
    simp only at h0
    subst h0
    rfl
  exact checkPart_ok (F.hpart j) hj (resParams_of_check hchk hnm) F.hsh (F.hplain j)
    (by rw [e F.sh (OfVStar.shape?_holo F.hsh)]; exact hpart)

end VStarFamily

end family

end FlockVBridge.Family
