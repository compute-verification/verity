import Proofs.Flock.C3.Sparse.SrcLayer
import Proofs.Flock.C3.Sparse.SessionReg
import Proofs.Flock.C3.Compose.RowsReg
import Proofs.Flock.C3.RegLevels.OpensRegs
import Proofs.Flock.C3.PubRows.Shape

/-!
# `sparseSessionAt_of_regs`' message keys, from the circuit, the reads file and the registration

`hkeys` (each message column a row of the sparse unit reads is a registered bit of copy 0 of unit `u`) is three facts,
each with its own producer, composed as `hRowKeys_of` composes `RowColKeys` and `RowsRegistered` at the parts:

* `SparseColKeys`: the unit's Δ key at each message column a row reads is a bit of the row of a port named as a message
  port, within the row. A fact of V*'s `rec-sparse` circuit file (`rec_sparse.ports`, the rows from column 0 in
  `messagePorts` order), `RowColKeys`' circuit half; `sparseColKeys_of_check` produces it from `RecSparse.check`'s
  `rowsOk` and `leavesOk` refusals, through `srcKey_inOrder` at copy 0 (`hKey_of_check` at every message port);
* `OwnNamesMsgs`: the session's reads file names every message port, `RowColKeys`' reads-file half;
  `sparseOwnNames_of_checkOwn` produces it from `RecVStar.checkOwn (.sparse sh hr) own` (`checkOwn_ok`'s two loops:
  `h` and `claim` through `ownPorts`, the rest through `sparseMsgs`);
* `hread`: the session registers each row of a port its reads file names at unit `u`: `RowsRegisteredC3` at one
  session and unit (`rowsRegisteredC3_of_covered`, `rowsCovered_checked`), the form `regReads_of_check`,
  `haout_of_laid` and `sparseSessionAt_of_zero` take.

`hkeys_of` composes them at one session, as `regCols_of_check` does for an R level; the column is an input of the unit
by `rowCol_lt_nInS`. `hkeys_of_rows` takes `hread` from `RowsRegisteredC3` at an accepted outer proof, in
`sparseSessionAt_of_regs`' form.
-/

namespace FlockC3.Sparse

open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge.Decode (RegBit)
open FlockVBridge.OfVStar (ReadCol rowCol)

/-- **The sparse unit keys its message columns by message rows**: at each message column a row reads, the Δ key of copy
0's input is bit `k'` of the row of port `q'`, within the row, and `q'` is named as a message port. -/
def SparseColKeys (c : Flock.Circuit) (hU : Built c.unit) (sh : Flock.RecSparse.Shape) : Prop :=
  ∀ q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum,
    ReadCol (Rows.ofNet hU) (rowCol (Flock.RecSparse.messagePorts sh) 0 q) →
      ∃ (q' : Fin c.ports.size) (k' : ℕ) (_ : k' < Layout.portRowW c q'),
        Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + rowCol (Flock.RecSparse.messagePorts sh) 0 q) =
          .msg q'.val (k' / 8) (k' % 8) ∧
        c.ports[q'].name ∈ (Flock.RecSparse.messagePorts sh).map (·.1)

/-- **The reads file names every message port.** -/
def OwnNamesMsgs (own : Array Flock.Registered.Port) (sh : Flock.RecSparse.Shape) : Prop :=
  ∀ nm ∈ (Flock.RecSparse.messagePorts sh).map (·.1), ∃ pt ∈ own, pt.name = nm

/-! ## The circuit half, from `RecSparse.check` -/

/-- **`rowsOk` at a message port**: the statement's row `i` is the port, by name, its words a whole row and its bits as
many as the words hold. -/
theorem rowsOk_msg_get {c : Flock.Circuit} {sh : Flock.RecSparse.Shape}
    (h : Flock.RecSparse.rowsOk c sh = true) {i : ℕ}
    (hi : i < (Flock.RecSparse.messagePorts sh).length) :
    ∃ hi' : i < c.ports.size,
      c.ports[i].name = (Flock.RecSparse.messagePorts sh)[i].1 ∧
        c.ports[i].words = VStar.portWordsOf (Flock.RecSparse.messagePorts sh)[i].2 ∧
          c.ports[i].bits = 16 * VStar.portWordsOf (Flock.RecSparse.messagePorts sh)[i].2 := by
  have hiA : i < ((Flock.RecSparse.messagePorts sh).map fun p => (p.1, VStar.portWordsOf p.2)).length := by
    simpa using hi
  have hiL : i < (Flock.RecSparse.ports sh).length := by
    rw [FlockVBridge.Sparse.ports_eq]
    simp only [List.length_append, List.length_map]
    omega
  unfold Flock.RecSparse.rowsOk at h
  have e := of_decide_eq_true h
  have hs := congrArg Array.size e
  simp only [Array.size_map, Array.size_extract, List.size_toArray] at hs
  have hi' : i < c.ports.size := by omega
  refine ⟨hi', ?_⟩
  have e2 := congrArg (fun a => a[i]!) e
  rw [getElem!_pos _ i (by simp only [Array.size_map, Array.size_extract]; omega),
    getElem!_pos _ i (by simp only [Array.size_map, List.size_toArray]; exact hiL)] at e2
  simp only [Array.getElem_map, Array.getElem_extract, Nat.zero_add, List.getElem_toArray,
    FlockVBridge.Sparse.ports_eq, List.getElem_append_left hiA, List.getElem_map,
    Prod.mk.injEq] at e2
  exact ⟨e2.1, by omega, e2.2.2⟩

/-- The statement's first rows' words are the message ports', a whole row each (`rowsOk`, port by port). -/
theorem starts_msg {c : Flock.Circuit} {sh : Flock.RecSparse.Shape}
    (h : Flock.RecSparse.rowsOk c sh = true) :
    ∀ i, i ≤ (Flock.RecSparse.messagePorts sh).length →
      ((c.ports.toList.take i).map (·.words)).sum =
        (((Flock.RecSparse.messagePorts sh).take i).map fun p => VStar.portWordsOf p.2).sum
  | 0, _ => by simp
  | i + 1, hi => by
    have hip : i < (Flock.RecSparse.messagePorts sh).length := by omega
    obtain ⟨hcs, -, hw, -⟩ := rowsOk_msg_get h hip
    rw [List.take_add_one, List.map_append, List.sum_append,
      List.getElem?_eq_getElem (by simpa using hcs), List.take_add_one, List.map_append,
      List.sum_append, List.getElem?_eq_getElem (by simpa using hip), starts_msg h i (by omega)]
    simp [hw]

/-- A message column sits in some port's row: bit `k'` of port `i`, after the earlier ports' rows. -/
theorem rowCol_locate : ∀ (ps : List (String × ℕ)) (k q : ℕ), q < 128 * (ps.map (·.2)).sum →
    ∃ (i : ℕ) (hi : i < ps.length) (k' : ℕ), k' < 128 * ps[i].2 ∧
      rowCol ps k q = k + 16 * ((ps.take i).map fun p => VStar.portWordsOf p.2).sum + k'
  | [], _, _, hq => by simp at hq
  | p :: ps, k, q, hq => by
    simp only [List.map_cons, List.sum_cons] at hq
    by_cases hcase : q < 128 * p.2
    · exact ⟨0, by simp, q, hcase, by simp [rowCol, hcase]⟩
    · obtain ⟨i, hi, k', hk', he⟩ :=
        rowCol_locate ps (k + 16 * VStar.portWordsOf p.2) (q - 128 * p.2) (by omega)
      refine ⟨i + 1, by simpa using hi, k', hk', ?_⟩
      simp only [rowCol, hcase, ↓reduceIte, List.take_succ_cons, List.map_cons, List.sum_cons]
      rw [he]
      omega

/-- **`SparseColKeys` from the verifier's check** (`hKey_of_check` at every message port): at a unit `RecSparse.check`
accepted with a copy, the statement's rows are `ports sh` (`rowsOk`) and copy 0 reads their words in order
(`leavesOk`), so each message column's key is a bit of its own port's row (`srcKey_inOrder`), named as a message
port. -/
theorem sparseColKeys_of_check {c : Flock.Circuit} (hU : Built c.unit)
    (hck : Flock.RecSparse.check c = .ok ()) {sh : Flock.RecSparse.Shape}
    (hp : Flock.RecSparse.params c.unit.name = some sh) (h0 : 0 < c.upv) : SparseColKeys c hU sh := by
  intro q hq _
  obtain ⟨-, -, -, hrows, hlv, hn⟩ := FlockVBridge.Sparse.check_sparse_all hck hp
  have hn' : Flock.HmNets.netOf (Flock.RecSparse.unitName sh) (Flock.RecSparse.unitLaid sh) =
      .ok c.unit := hn
  obtain ⟨i, hi, k', hk', he⟩ := rowCol_locate (Flock.RecSparse.messagePorts sh) 0 q hq
  obtain ⟨hi', hname, hw, hb⟩ := rowsOk_msg_get hrows hi
  have hpw := FlockVBridge.OfVStar.le_portWords (Flock.RecSparse.messagePorts sh)[i].2
  have hsz : (Array.replicate (Flock.RecSparse.inWords sh) 16).size = Flock.RecSparse.inWords sh :=
    Array.size_replicate
  have h16 : ∀ x ∈ (Array.replicate (Flock.RecSparse.inWords sh) 16).toList, x = 16 := by
    intro x hx
    rw [Array.toList_replicate] at hx
    exact List.eq_of_mem_replicate hx
  have hN : 16 * (Array.replicate (Flock.RecSparse.inWords sh) 16).size ≤
      c.unit.inWords * Flock.WORD := by
    rw [FlockVBridge.Sparse.inWords_sparse hn', hsz]
    show 16 * _ ≤
      Flock.HmNets.groupWords ((Array.replicate (Flock.RecSparse.inWords sh) 16).foldl (· + ·) 0) *
        Flock.WORD
    rw [← Array.foldl_toList, Array.toList_replicate, FlockVBridge.foldl_replicate16, Nat.zero_add]
    exact Flock.HmNets.le_groupWords _
  have hmsg_le : ((Flock.RecSparse.messagePorts sh).map fun p => VStar.portWordsOf p.2).sum ≤
      Flock.RecSparse.inWords sh := by
    have he' := FlockVBridge.Sparse.inWords_eq sh
    rw [FlockVBridge.Sparse.ports_snd, FlockVBridge.OfVStar.foldl_add, Nat.zero_add,
      List.sum_append] at he'
    have hm : FlockVBridge.Sparse.msgWords sh =
        (Flock.RecSparse.messagePorts sh).map fun p => VStar.portWordsOf p.2 := rfl
    rw [hm] at he'
    omega
  have hst := Layout.starts_get c hi'
  have hs0 := starts_msg hrows i (by omega)
  have hs1 := starts_msg hrows (i + 1) (by omega)
  have htake : ((c.ports.toList.take (i + 1)).map (·.words)).sum =
      ((c.ports.toList.take i).map (·.words)).sum + c.ports[i].words := by
    rw [List.take_add_one, List.map_append, List.sum_append,
      List.getElem?_eq_getElem (by simpa using hi')]
    simp
  have htk : (((Flock.RecSparse.messagePorts sh).take (i + 1)).map
      fun p => VStar.portWordsOf p.2).sum ≤
      ((Flock.RecSparse.messagePorts sh).map fun p => VStar.portWordsOf p.2).sum := by
    conv_rhs => rw [← List.take_append_drop (i + 1) (Flock.RecSparse.messagePorts sh)]
    rw [List.map_append, List.sum_append]
    exact Nat.le_add_right _ _
  have key := FlockC3.RegLevels.srcKey_inOrder (FlockVBridge.Sparse.inGroups_sparse hn') rfl h16
    (Aliased.portBits_nodup hU) (j := 0) (q := i) (k := k')
    (by rw [hsz]; exact FlockVBridge.Sparse.leavesOk_get hlv h0) hi'
    (by rw [hw]; omega) (by rw [hb]; omega) (by rw [hsz]; omega) hN
  rw [Nat.zero_mul, Nat.zero_add, hst, hs0] at key
  refine ⟨⟨i, hi'⟩, k', ?_, ?_, ?_⟩
  · show k' < 16 * c.ports[i].words
    rw [hw]
    omega
  · show Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + rowCol (Flock.RecSparse.messagePorts sh) 0 q) =
      .msg i (k' / 8) (k' % 8)
    rw [Nat.zero_mul, Nat.zero_add, he, Nat.zero_add]
    exact key
  · show c.ports[i].name ∈ (Flock.RecSparse.messagePorts sh).map (·.1)
    rw [hname]
    exact List.mem_map.2 ⟨_, List.getElem_mem hi, rfl⟩

/-! ## The reads-file half, from `RecVStar.checkOwn` -/

/-- **`OwnNamesMsgs` from the verifier's reads-file check** (`checkOwn_ok`): `h` and `claim` are `ownPorts`', in the
file by the second loop, and every other message port is `sparseMsgs`', in the file by the third. -/
theorem sparseOwnNames_of_checkOwn {sh : Flock.RecSparse.Shape} {hr : ℕ}
    {own : Array Flock.Registered.Port}
    (h : Flock.RecVStar.checkOwn (.sparse sh hr) own = .ok ()) : OwnNamesMsgs own sh := by
  obtain ⟨hown, hmsgs⟩ := FlockC3.PubRows.checkOwn_ok h
  obtain ⟨M, hmp⟩ : ∃ M, Flock.RecSparse.messagePorts sh = ("h", 64) :: (M ++ [("claim", 1)]) := by
    refine ⟨(List.range Flock.RecSparse.PRE_ROUNDS).map (fun j => (s!"pre{j}", 2)) ++ [("v2", 1)] ++
      (sh.degrees.zipIdx.map fun (deg, j) => (s!"sc{j}", deg)) ++ [("u", sh.PP)] ++
      (if sh.stacked then [("vt", sh.T)] else []), ?_⟩
    simp [Flock.RecSparse.messagePorts, Flock.RecSparse.SKIP]
  have hdrop : ((Flock.RecSparse.messagePorts sh).drop 1).dropLast = M := by
    rw [hmp, List.drop_one, List.tail_cons, List.dropLast_concat]
  intro nm hnm
  rw [hmp] at hnm
  simp only [List.map_cons, List.map_append, List.mem_cons, List.mem_append,
    List.map_nil] at hnm
  rcases hnm with rfl | hnm | rfl | hnm
  · have hq := hown ("h", s!"rec-c{hr}", #[some 0, some 1]) (by simp [Flock.RecVStar.ownPorts])
    simp only [Flock.RecVStar.msgPorts, List.mem_map] at hq
    obtain ⟨p, hp, hpe⟩ := hq
    exact ⟨p, Array.mem_toList_iff.1 (List.mem_of_mem_filter hp), congrArg (·.1) hpe⟩
  · obtain ⟨p, hpM, rfl⟩ := List.mem_map.1 hnm
    have hin : (p.1, s!"rec-sparse-{p.1}", #[some 0, some 1]) ∈
        Flock.RecVStar.sparseMsgs (.sparse sh hr) := by
      simp only [Flock.RecVStar.sparseMsgs, hdrop, List.mem_map]
      exact ⟨p, hpM, rfl⟩
    have hq := hmsgs _ hin
    simp only [List.mem_map] at hq
    obtain ⟨p', hp', hpe⟩ := hq
    exact ⟨p', Array.mem_toList_iff.1 hp', congrArg (·.1) hpe⟩
  · have hq := hown ("claim", Flock.RecVStar.CLAIM_VALUE, #[some 0, some 1])
      (by simp [Flock.RecVStar.ownPorts])
    simp only [Flock.RecVStar.msgPorts, List.mem_map] at hq
    obtain ⟨p, hp, hpe⟩ := hq
    exact ⟨p, Array.mem_toList_iff.1 (List.mem_of_mem_filter hp), congrArg (·.1) hpe⟩
  · cases hnm

/-! ## The composition -/

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
  {hscope : Layout.scopeOk c = true} {Lv : Law nv}

/-- **`hkeys` from the circuit's keys, the reads file and the registration** (`hRowKeys_of`'s twin): the column's key
names a message row, which the reads file names, so a registered read whose bit is the registrant's. -/
theorem hkeys_of (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) {sh : Flock.RecSparse.Shape} {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok c.unit) {u : Fin nv}
    (hcol : SparseColKeys c hU sh) (hown : OwnNamesMsgs zo.own sh)
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) →
      zo.rd.isRead (.inl (u, q))) :
    ∀ q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum,
      ReadCol (Rows.ofNet hU) (rowCol (Flock.RecSparse.messagePorts sh) 0 q) →
        ∃ b, RegBit zo u 0 (rowCol (Flock.RecSparse.messagePorts sh) 0 q) b := by
  intro q hq hrd
  obtain ⟨q', k', hk, hkey, hnm⟩ := hcol q hq hrd
  obtain ⟨pt, hpt, hpn⟩ := hown _ hnm
  exact ⟨_, FlockVBridge.Sparse.rowCol_lt_nInS hU hn hq, q', k', hk, hkey, hread q' ⟨pt, hpt, hpn⟩, rfl⟩

section Family

open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

/-- **`sparseSessionAt_of_regs`' `hkeys` at an accepted outer proof** (`hkeys_of`), with `hread` from
`RowsRegisteredC3`. -/
theorem hkeys_of_rows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) {dg : ByteArray → ByteArray} {N k Rw : ℕ}
    (hrr : F.RowsRegisteredC3 dg N k Rw) {R : Reg} {ω : L.Ω} {t : List (Cm × (holoInnerAt T CT st hp).Coin)} {R₂ : Reg₂}
    {τ : (j : Fin (vstarCount F.ish)) → Strategy ((F.zs R ω t R₂ j).game dg N)}
    {o : (j : Fin (vstarCount F.ish)) → Reg' × Finset (Fin (F.Vs j).nv) × Exec.Transcript}
    (hA : Accepted (I := holoInnerAt T CT st hp) F.zs dg N k Rw R ω t R₂ τ o) {s : Fin (vstarCount F.ish)}
    {sh : Flock.RecSparse.Shape} {name : String}
    (hn : Flock.HmNets.netOf name (Flock.RecSparse.unitLaid sh) = .ok (F.Vs s).c.unit)
    (hcol : SparseColKeys (F.Vs s).c (F.Vs s).hU sh) (hown : OwnNamesMsgs (F.zs R ω t R₂ s).own sh)
    (u : Fin (F.Vs s).nv) :
    ∀ q < 128 * ((Flock.RecSparse.messagePorts sh).map (·.2)).sum,
      ReadCol (Rows.ofNet (F.Vs s).hU) (rowCol (Flock.RecSparse.messagePorts sh) 0 q) →
        ∃ b, RegBit (F.zs R ω t R₂ s) u 0 (rowCol (Flock.RecSparse.messagePorts sh) 0 q) b :=
  hkeys_of (F.zs R ω t R₂ s) hn hcol hown (hrr R ω t R₂ τ o hA s u)

end Family

end FlockC3.Sparse
