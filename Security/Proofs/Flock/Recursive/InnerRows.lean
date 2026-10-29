import Proofs.Flock.Claims.Scope.StmtOf
import Proofs.Flock.Claims.Scope.Sizes
import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
import Proofs.Flock.Soundness.Discharge.HiddenExec.SiteH
import Proofs.Flock.Soundness.Discharge.HiddenExec.Flat
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Upv
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Block
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Layout.KeyProg
import Proofs.Flock.Soundness.Discharge.Integrate.Accepted

set_option Elab.async false

/-!
# A unit's rows, read off the inner witness

A recursive session's link proof block opens a unit's row as a digest and a salt, and checks their commit string
`b ‖ c` against the rows tree the verifier builds from the commit strings the inner statement publishes
(`RecLinkTree.rowsRoot`). `rows_commit` ties those published strings to the inner relation's witness: in a witness of
the deployed relation (`Readable.relation`) at a statement the verifier parsed and kept in scope, with hidden outputs,
the commit string of row `q` of unit `i` of table `j` (the bytes the witness's row hash reads, `rowBytes`, under the salt
its `hm96` slot reads, `salt`) is the one the table's public file publishes for the unit's row (`Public.digest`). It is
the circuit side of the row commitment (`HmRowComputes`) at the inner statement's rows, proved with no premise: the row
hash and the `hm96` slot compute the commit string (`bc_site_commitH`), and the regions pin its 128 bytes to the public
file's (`bc_pinned'`).

* `bitAt`: table `j`'s share of the hidden values, block by block (`Readable.zOf`);
* `rowBytes`, `salt`: unit `i`'s row `q`, at template instance `i % g` of block `i / g` (`Circuit.instOf`);
* `commitString_data`: `Binding.commitString` is the executable's `Hm96.Default512.commitString` of the row's SHA-512;
* `pub_digest_size`: every unit's `b ‖ c` in a hidden-output statement's public file is 128 bytes, drawn or not;
* `unit_commit`: the same at one accepted statement, for any satisfying witness of its block R1CS.
-/

namespace Flock.InnerRows

open FlockSoundness FlockSoundness.Discharge Exec Readable

/-- **Bit `y` of block `o`** of table `j`'s share of an inner witness's hidden values (`Readable.zOf`: of `J` tables,
table `j`'s bit `a` is hidden value `a·J + j`). -/
def bitAt (p : Parsed) (w : Claims.Flock.Wit) (j o y : ℕ) : Bool :=
  w.hid.getD ((o * 2 ^ p.sts[j]!.c.kLog + y) * p.sts.size + j) false

/-- **Row `q` of unit `i` of table `j`, as its row hash reads it**: the bytes at its message columns (`Typed.msgCol`),
behind its prefix or row-segment encoding (`Hidden.Any.rowMsg`), at template instance `i % g` of block `i / g`. -/
def rowBytes (p : Parsed) (w : Claims.Flock.Wit) (j i q : ℕ) : List UInt8 :=
  Hidden.Any.rowMsg (p.sts[j]!.c.ports[q]!)
    (Hm.bytesOf (bitAt p w j (i / p.sts[j]!.c.g)) (2 * (p.sts[j]!.c.ports[q]!).words)
      (Typed.msgCol p.sts[j]!.c (i % p.sts[j]!.c.g) q))

/-- **Row `q` of unit `i` of table `j`'s salt**: the 192 bytes at its `hm96` slot's salt columns (`Hm.saltCol`). -/
def salt (p : Parsed) (w : Claims.Flock.Wit) (j i q : ℕ) : ByteArray :=
  ⟨(Hm.bytesOf (bitAt p w j (i / p.sts[j]!.c.g)) 192 (Hm.saltCol p.sts[j]!.c (i % p.sts[j]!.c.g) q)).toArray⟩

theorem bytesOf_congr {f f' : ℕ → Bool} {N K : ℕ} {col : ℕ → ℕ → ℕ} (hc : ∀ j < N, ∀ t < 8, col j t < K)
    (h : ∀ y < K, f y = f' y) : Hm.bytesOf f N col = Hm.bytesOf f' N col := by
  unfold Hm.bytesOf
  congr 1
  funext j
  congr 1
  funext t
  exact h _ (hc j j.isLt t t.isLt)

theorem bit_toZ : ∀ b : Bool, FlockSoundness.bit (toZ b) = b := by decide

theorem eq_of_data {a b : ByteArray} (h : a.data.toList = b.data.toList) : a = b := by
  cases a
  cases b
  simp only [ByteArray.mk.injEq]
  exact Array.ext' h

/-- **`Binding.commitString` is the executable's**: of a row's bytes `r` under a salt, the commit string `b ‖ c` is
`Hm96.Default512.commitString` of the row's SHA-512 under the salt. -/
theorem commitString_data (r : List UInt8) (salt : ByteArray) :
    Binding.commitString r salt.data.toList =
      (Hm96.Default512.commitString (Sha512.hash ⟨r.toArray⟩) salt).data.toList := by
  show Merkle.xorBytes (Refine.enc512.dig (Refine.H512 r)) (Refine.K512.mask salt.data.toList) ++
      Refine.enc512.dig (Refine.H512 (Refine.K512.saltPrefix ++ salt.data.toList)) = _
  rw [Refine.dig_eq, Refine.dig_eq, Refine.H512_val, Refine.H512_val, Refine.K512_mask, Refine.bytes_of_list,
    ← Refine.xor_range _ _ 64 (Refine.sha512_size _) (Refine.mask_size salt)]
  show _ ++ (Sha512.hash ⟨(Hm96.Default512.saltPrefix.data.toList ++ salt.data.toList).toArray⟩).data.toList = _
  rw [← Refine.data_append Hm96.Default512.saltPrefix salt, Refine.bytes_of_list]
  exact (Refine.data_append _ _).symm

/-- **Every unit's `b ‖ c` in a hidden-output statement's public file is 128 bytes**, drawn or not, with or without
shared rows. -/
theorem pub_digest_size {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt}
    (h : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) {i q : ℕ}
    (hi : i < st.pub.n) (hq : q < st.c.ports.size) : (st.pub.digest i q st.c.ports.size).size = 128 := by
  obtain ⟨pop, hpop, hnone, hsome⟩ := (Hidden.setupHidden_spec' h).pub
  have hload := Hidden.loadPublic_nil hpop
  cases draw with
  | none =>
    rw [hnone rfl] at hi ⊢
    exact Hidden.pop_digest_sizeW hload hi hq
  | some d =>
    have hdrawn : HmRow.drawn { st.c with outPorts := #[] } pop d = .ok st.pub := hsome d rfl
    obtain ⟨D, hD, hpn, hn, -⟩ := Layout.drawn_spec hdrawn
    have hlt := Layout.proved_lt_population hD
    have hr : i < D.proved.size := hn ▸ hi
    exact (congrArg ByteArray.size (Hidden.drawn_digestW hload hdrawn hD hlt hr hq)).trans
      (Hidden.pop_digest_sizeW hload (by rw [← hpn]; exact hlt _ (Array.getElem_mem hr)) hq)

/-- **A site's row commits its table's `b ‖ c`**: in a satisfying witness of an accepted statement whose regions
contain the executable's (`hregs`) and whose rows are each fit or `sha512/row-seg/v1` (`hrf`), at block `o`, template
instance `g` and row `q`, the commit string of the bytes the row hashes under the salt at `saltCol` is the 128 bytes the
statement's public file pins there (`Layout.bcChunk`), once each chunk up to `g` is 128 bytes long. -/
theorem site_commit {T : Model.Statement} (a : HiddenExec.AcceptedH T)
    (hregs : ∀ r ∈ a.st.regions.toList, Refine.regionOf a.st.c.kLog (a.st.m - a.st.c.kLog) r ∈ a.regs)
    (hrf : Hidden.Any.RowsFitS a.st.c) {z : Model.Witness T.m} (hz : T.Satisfies z) (o : Fin (2 ^ T.nbl))
    {g q : ℕ} (hg : g < a.st.c.g) (hq : q < a.st.c.ports.size)
    (hsz : ∀ g' ≤ g, (Layout.bcChunk a.st.c a.st.pub o.val g' q).size = 128) :
    Binding.commitString (Hidden.Any.rowMsg a.st.c.ports[q]!
        (Hm.bytesOf (Hm.blockBit T z o) (2 * a.st.c.ports[q]!.words) (Typed.msgCol a.st.c g q)))
      (Hm.bytesOf (Hm.blockBit T z o) 192 (Hm.saltCol a.st.c g q)) =
      (Layout.bcChunk a.st.c a.st.pub o.val g q).data.toList := by
  have hcom := HiddenExec.bc_site_commitH a (Hidden.setupHidden_upv_pos a.ht a.h) hrf hz o hg hq
  rw [HiddenExec.ofFn_siteH a z o _ _ (fun b _ t ht => Layout.msgCol_lt_of_parse a.parse hg hq b t ht),
    HiddenExec.bytesAt_siteH a z o 192 _ (fun b hb t ht => Hm.saltCol_lt a.rowRanges hg hq hb ht)] at hcom
  rw [← hcom]
  obtain ⟨tags, cf, pf, coins, tables, draw, partition, program, st, ht, h, L, hR, regs, hT⟩ := a
  subst hT
  dsimp only at hregs hrf hg hq hsz ⊢
  have s := Hidden.setupHidden_spec' h
  have P := parse_facts (Hidden.parse_untyped ht s.parse).1
  obtain ⟨ls, hls⟩ := P.check
  have hg2 := Layout.vu_lt_pow2_of (check_facts P.typed hls) hg
  obtain ⟨hrsz, hreg⟩ := Hidden.regions_get s.regions
  have hpr : q < st.regions.size := hrsz ▸ hq
  exact Hidden.bc_pinned' L regs hregs hz o hg hg2 (Array.getElem_mem_toList hpr) (hreg _ hpr) hsz

/-- **Unit `i`'s row `q` commits its `b ‖ c`**: at an accepted untyped hidden-output statement whose rows are each fit or
`sha512/row-seg/v1` (`hrf`), in a satisfying witness `z` of its block R1CS whose bits in block `i / g` are `f`'s, the
commit string of row `q` at template instance `i % g` is the unit's `b ‖ c` in the statement's public file. -/
theorem unit_commit {tags : Tags} {cf pf : ByteArray} {coins : Coins} {tables : List (String × ByteArray)}
    {draw partition : Option Lean.Json} {program : Option ByteArray} {st : Stmt} (ht : tags.typed = false)
    (hd : Stmt.setupHidden tags cf pf coins tables draw partition program = .ok st) (hrf : Hidden.Any.RowsFitS st.c)
    (L : FlockSoundness.Layout st) {z : Model.Witness st.m} (hz : (L.model (Refine.regionsOf st)).Satisfies z)
    {i q : ℕ} (hi : i < st.pub.n) (hq : q < st.c.ports.size) {f : ℕ → Bool}
    (hf : ∀ (h : i / st.c.g < 2 ^ (L.model (Refine.regionsOf st)).nbl) (y : ℕ), y < 2 ^ st.c.kLog →
      Hm.blockBit (L.model (Refine.regionsOf st)) z ⟨i / st.c.g, h⟩ y = f y) :
    Binding.commitString (Hidden.Any.rowMsg st.c.ports[q]!
        (Hm.bytesOf f (2 * st.c.ports[q]!.words) (Typed.msgCol st.c (i % st.c.g) q)))
      (Hm.bytesOf f 192 (Hm.saltCol st.c (i % st.c.g) q)) = (st.pub.digest i q st.c.ports.size).data.toList := by
  have s := Hidden.setupHidden_spec' hd
  have hp := (Hidden.parse_untyped ht s.parse).1
  have P := parse_facts hp
  obtain ⟨ls, hls⟩ := P.check
  have hgpos : 0 < st.c.g := Hidden.check_g_pos P.typed hls
  have hob : i / st.c.g < 2 ^ (L.model (Refine.regionsOf st)).nbl := by
    show i / st.c.g < 2 ^ (st.m - st.c.kLog)
    rw [s.m, Nat.add_sub_cancel_left, s.nbl, s.blocksReal, s.n]
    exact (Nat.div_lt_iff_lt_mul hgpos).2
      (lt_of_lt_of_le hi ((Integrate.fits_blocks hgpos).trans_eq (Nat.mul_comm _ _)))
  have hg : i % st.c.g < st.c.g := Nat.mod_lt _ hgpos
  have hdm : i / st.c.g * st.c.g + i % st.c.g = i := Nat.div_add_mod' i st.c.g
  have hin : ∀ g' ≤ i % st.c.g, i / st.c.g * st.c.g + g' < st.pub.n := fun g' hg' =>
    lt_of_le_of_lt ((Nat.add_le_add_left hg' _).trans_eq hdm) hi
  have hchunk : ∀ g' ≤ i % st.c.g,
      Layout.bcChunk st.c st.pub (i / st.c.g) g' q = st.pub.digest (i / st.c.g * st.c.g + g') q st.c.ports.size :=
    fun g' hg' => by
      unfold Layout.bcChunk Circuit.instOf
      simp only [hin g' hg', ↓reduceIte]
  have key : Binding.commitString (Hidden.Any.rowMsg st.c.ports[q]!
        (Hm.bytesOf (Hm.blockBit (L.model (Refine.regionsOf st)) z ⟨i / st.c.g, hob⟩) (2 * st.c.ports[q]!.words)
          (Typed.msgCol st.c (i % st.c.g) q)))
      (Hm.bytesOf (Hm.blockBit (L.model (Refine.regionsOf st)) z ⟨i / st.c.g, hob⟩) 192
        (Hm.saltCol st.c (i % st.c.g) q)) =
      (Layout.bcChunk st.c st.pub (i / st.c.g) (i % st.c.g) q).data.toList :=
    site_commit ⟨_, _, _, _, _, _, _, _, st, ht, hd, L, HiddenExec.unitRowsFlat ht hd hrf.1, Refine.regionsOf st, rfl⟩
      (fun r hr => List.mem_map_of_mem hr) hrf hz ⟨i / st.c.g, hob⟩ hg hq fun g' hg' => by
        show (Layout.bcChunk st.c st.pub (i / st.c.g) g' q).size = 128
        rw [hchunk g' hg']
        exact pub_digest_size hd (hin g' hg') hq
  have hR : Hm.RowRanges st.c := HiddenExec.AcceptedH.rowRanges
    ⟨_, _, _, _, _, _, _, _, st, ht, hd, L, HiddenExec.unitRowsFlat ht hd hrf.1, Refine.regionsOf st, rfl⟩
  rw [hchunk _ le_rfl, hdm,
    bytesOf_congr (N := 2 * st.c.ports[q]!.words) (fun b _ t h8 => Layout.msgCol_lt_of_parse hp hg hq b t h8)
      (hf hob),
    bytesOf_congr (N := 192) (fun b hb t h8 => Hm.saltCol_lt hR hg hq hb h8) (hf hob)] at key
  exact key

/-- **A unit's rows, read off the inner witness**: at a statement the verifier parsed and kept in scope, with hidden
outputs, every witness of the deployed relation commits row `q` of unit `i` of table `j` as the public file publishes it:
the executable's commit string (`Hm96.Default512.commitString`) of the SHA-512 of the bytes the witness's row hash reads,
under the witness's salt, is the table's `b ‖ c` for the unit's row. -/
theorem rows_commit {p : Parsed} {w : Claims.Flock.Wit} (hP : Parses p) (hsc : p.scope = .ok ())
    (hh : p.tags.hiddenOutputs = true) (hR : Claims.Flock.RelS (toReadable relation p) w)
    {j i q : ℕ} (hj : j < p.sts.size) (hi : i < p.sts[j].pub.n) (hq : q < p.sts[j].c.ports.size) :
    Hm96.Default512.commitString (Sha512.hash ⟨(rowBytes p w j i q).toArray⟩) (salt p w j i q) =
      p.sts[j].pub.digest i q p.sts[j].c.ports.size := by
  obtain ⟨hc, -⟩ := (scope_ok_iff p).1 hsc
  obtain ⟨hRu, c0, -, hc0, -, -, -, hok⟩ := (check_ok_iff _ _ _ _ _ _ _ _).1 hc
  have hpi : p.publicInputs = none := Option.not_isSome_iff_eq_none.1 (by simp [hRu.2.2.2.2.2])
  have ht : p.tags.typed = false := hRu.2.1
  obtain ⟨spec, hset⟩ := setup_of_tablesOf (tablesOf_of_parses hP)
  unfold Read.setup at hset
  rw [hpi] at hset
  obtain ⟨-, hall⟩ := HiddenExec.setupTables_setupOne hset
  obtain ⟨d, hd, -⟩ := hall j hj
  have ht' : (tagsAt p.tags p.I.count j).typed = false := (tagsAt_typed _ _ _).trans ht
  unfold HiddenExec.setupOne at hd
  rw [tagsAt_hiddenOutputs, hh] at hd
  simp only [↓reduceIte] at hd
  obtain ⟨hf, hsat⟩ := (relS_relation p w).1 hR j hj
  have e! : p.sts[j]! = p.sts[j] := getElem!_pos p.sts j hj
  revert hf hsat
  unfold rowBytes salt
  rw [e!]
  generalize hst : p.sts[j] = st at hd hi hq ⊢
  intro hf hsat
  have s := Hidden.setupHidden_spec' hd
  have hcst : HmOut.parse p.tags p.I.circuitFile p.I.tables = .ok st.c :=
    (parseH_tagsAt ht _ _ _ _).symm.trans s.parse
  have hc0' : c0 = st.c := by
    unfold ProvedScope.circuitOf at hc0
    rw [hh] at hc0
    simp only [↓reduceIte] at hc0
    exact Except.ok.inj (hc0.symm.trans hcst)
  rw [hh] at hok
  simp only [↓reduceIte] at hok
  rw [hc0', FailClosed.scopeOkI_eq] at hok
  have hrows : (Layout.v1Ports st.c || Layout.bitRows st.c) = true := by
    unfold Layout.scopeOkI at hok
    simp only [Bool.and_eq_true] at hok
    exact hok.2
  have hpu := Hidden.parse_untyped ht' s.parse
  have hrf : Hidden.Any.RowsFitS st.c := HiddenExec.rowsFit_of_rows hpu.1 hpu.2 hrows
  have L : FlockSoundness.Layout st := Hidden.hidden_layout ht' hd (Hidden.setupHidden_unpinned ht' hd hrf.1.1)
  refine eq_of_data ((commitString_data _ _).symm.trans ?_)
  have hk : p.sts[j]!.c.kLog = st.c.kLog := by rw [e!, hst]
  refine unit_commit ht' hd hrf L hsat hi hq fun h y hy => ?_
  refine (Hm.blockBit_of_lt (L.model (Refine.regionsOf st)) (zOf p.sts.size w j st.m) ⟨i / st.c.g, h⟩ hy).trans ?_
  show FlockSoundness.bit (toZ (w.hid.getD ((i / st.c.g * 2 ^ st.c.kLog + y) * p.sts.size + j) false)) =
    w.hid.getD ((i / st.c.g * 2 ^ p.sts[j]!.c.kLog + y) * p.sts.size + j) false
  rw [bit_toZ, hk]

end Flock.InnerRows
