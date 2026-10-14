import Proofs.Flock.C3.RegLevels.ValueTie
import Proofs.Flock.C3.RegLevels.OutRegs

/-!
# `opensTopAt_of_regs`' `hT` at R's levels `ℓ ≥ 1`, from a root over the cap's rows

`opensTopAt_of_regs` takes `htop` and `hT`: the unit's `out` columns are the cap entry's 512 bits, registered. A level's
cap entry `top` is 64 bytes (`capAt_size`, from `CapsAt`), its row is 64 words (`capRow`), and bit `b < 512` of that row
is bit `b` of `top` (`capRow_bit`). So where `out`'s row is a registered read of a root that is the verifier's tree over
rows whose read row is `top`'s (`tie_root`'s `hroot`), `hT_of_root`: each `out` column is registered as `top`'s bit
(`regOuts_of_laid` at output 0, `bits_of_rowInV`; `hT_of_enc`), or `valueColl` outputs two different SHA-512 inputs
with one digest.

By the read's leaf instead (`hT_of_leaf`): at an accepted session the registrant's leaf at a row read is the table's,
`treeLeaf` of the public file's digest there (`regLeaf_inl`), or the read's two openings conflict (`leaf_of_accepts`,
`LeafColl`'s event); where that digest is a node's commit string under the held salts (`TopOf`), `salt_tie` gives the
node's row, or `saltFinder`'s pair collides.
-/

namespace FlockC3.RegLevels

open Flock
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open FlockVBridge FlockVBridge.AccChain
open FlockSoundness.Discharge.Hm (bitL)

theorem testBit_lohi (lo hi : ℕ) (hlo : lo < 256) (k : ℕ) :
    (lo + 256 * hi).testBit k = if k < 8 then lo.testBit k else hi.testBit (k - 8) := by
  have := Nat.testBit_two_pow_mul_add (a := hi) (b := lo) (i := 8) (b_lt := by simpa using hlo) (j := k)
  rw [show (256 : ℕ) = 2 ^ 8 by rfl, Nat.add_comm, this]

theorem getD_get! {top : ByteArray} {k : ℕ} : top.get! k = top.data.toList.getD k 0 := by
  simp only [ByteArray.get!, getElem!_def, List.getD_eq_getElem?_getD, Array.getElem?_toList]
  cases top.data[k]? <;> rfl

/-- **Bit `b` of a 64-byte node's row is the node's bit `b`**: word `b / 16` is bytes `b / 8` (low) and `b / 8 + 1`
(high) of the node, little-endian (`Firewall.topRows`). -/
theorem capRow_bit {top : ByteArray} (htop : top.size = 64) {b : ℕ} (hb : b < 512) :
    ((capRow top)[b / 16]!).testBit (b % 16) = bitL top.data.toList b := by
  rw [capRow, topRows_of_size htop, List.headD_cons, getElem!_pos _ (b / 16)
    (by simp only [Array.size_append, Array.size_map, Array.size_range, Array.size_replicate]; omega),
    Array.getElem_append_left (by simp only [Array.size_map, Array.size_range]; omega), Array.getElem_map,
    Array.getElem_range, testBit_lohi _ _ (UInt8.toNat_lt _), getD_get!, getD_get!]
  unfold bitL
  split
  · rw [show 64 * 0 + 2 * (b / 16) = b / 8 by omega, show b % 16 = b % 8 by omega]
  · rw [show 64 * 0 + 2 * (b / 16) + 1 = b / 8 by omega, show b % 16 - 8 = b % 8 by omega]

/-- **A level's cap entry is 64 bytes** (`CapsAt`), at each index below the level's `2^cap`. -/
theorem capAt_size {sch : Flock.Schedule} {cap : Array ByteArray} {lig : Flock.LigeritoProof}
    (h : FlockVBridge.Family.CapsAt sch cap lig) {ℓ : ℕ} (hℓ : ℓ < sch.levels.size) {i : ℕ}
    (hi : i < 2 ^ (sch.levels[ℓ]!).cap) :
    ((FlockSoundness.Refine.capAt cap lig ℓ).getD i ByteArray.empty).size = 64 := by
  obtain ⟨hs, hw⟩ := h ℓ hℓ
  rw [Array.getD_eq_getD_getElem?, Array.getElem?_eq_getElem (by omega), Option.getD_some]
  exact hw _ (Array.getElem_mem _)

section Session

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **The registered `out` row's bits are `top`'s**: `hT_of_root`'s conclusion where `out`'s row bytes are `top`'s
row's (`rowInV (capRow top)`). -/
theorem hT_of_enc {u : Fin nv} {j : ℕ} (hj : j < c.upv) {LANES H : ℕ} {p : Ports} {outBits : Array ℕ}
    {name : String} (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hob : 2048 ≤ outBits.foldl (· + ·) 0) (hlaid : OutsLaid c j) (hq : HmOut.nIn c < c.ports.size)
    (hp : zo.rd.isRead (.inl (u, ⟨_, hq⟩))) (hseg : c.ports[HmOut.nIn c]!.segs = none) {top : ByteArray}
    (htop : top.size = 64)
    (he : FlockSoundness.Discharge.HiddenExec.encPortH c.ports[HmOut.nIn c]!
      (FlockVBridge.Decode.regVal zo (.inl (u, ⟨_, hq⟩))) = rowInV (capRow top)) :
    ∀ t < 512, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + t)
      (bitL top.data.toList t) := by
  intro t ht
  have he' : Layout.rowEncP (FlockSoundness.Discharge.Hidden.rowPre c.ports[HmOut.nIn c]!.v2)
      (FlockVBridge.Decode.regVal zo (.inl (u, ⟨_, hq⟩))) = rowInV (capRow top) :=
    (congrFun (FlockVBridge.Family.encPortH_of_none hseg) _).symm.trans he
  obtain ⟨hq', hk, hreg⟩ := regOuts_of_laid zo (u := u) hj hl hob hlaid (jo := 0) (by decide) (fun _ => hp)
    (b := t) (by omega)
  have e := (bits_of_rowInV (FlockVBridge.Family.rowPre_length _) he' t
    (lt_of_lt_of_le hk ((Finset.le_sup (f := Layout.portRowW c)
      (Finset.mem_univ (⟨HmOut.nIn c + 0, hq'⟩ : Fin c.ports.size))).trans (Nat.le_succ _)))).trans
    (capRow_bit htop ht)
  convert hreg using 1 <;> first | omega | exact e.symm

/-- **`opensTopAt_of_regs`' `hT`, or SHA-512 collides**: at copy `j` of unit `u` built from `recOpenLaid` with V*'s
output layout (`OutsLaid`), where `out`'s row (port `nIn`, without segments) is a registered read of a root that is the
verifier's tree over `rows` under `salt` (`hroot`, `hn`) and the read row is the 64-byte `top`'s (`hrow`), each of the
first 512 output columns is registered as `top`'s bit, or `valueColl` collides. -/
theorem hT_of_root {u : Fin nv} {j : ℕ} (hj : j < c.upv) {LANES H : ℕ} {p : Ports} {outBits : Array ℕ}
    {name : String} (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hob : 2048 ≤ outBits.foldl (· + ·) 0) (hlaid : OutsLaid c j) (hq : HmOut.nIn c < c.ports.size)
    (hp : zo.rd.isRead (.inl (u, ⟨_, hq⟩))) (hseg : c.ports[HmOut.nIn c]!.segs = none)
    {rows : Array (Array ℕ)} {salt : ℕ → ByteArray}
    (hroot : FV3.root (zo.rd.port (.inl (u, ⟨_, hq⟩))).domain
      (leavesV (zo.rd.port (.inl (u, ⟨_, hq⟩))).domain rows salt) Sha512.hash =
        (zo.rd.port (.inl (u, ⟨_, hq⟩))).root)
    (hn : (zo.rd.port (.inl (u, ⟨_, hq⟩))).leaves = rows.size) {top : ByteArray} (htop : top.size = 64)
    (hrow : rows[zo.rd.readAt (.inl (u, ⟨_, hq⟩))]! = capRow top) :
    (∀ t < 512, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + t)
        (bitL top.data.toList t)) ∨
      ∃ x y, valueColl (zo.rd.port (.inl (u, ⟨_, hq⟩))) rows salt (zo.rd.readAt (.inl (u, ⟨_, hq⟩)))
          (FlockSoundness.Discharge.HiddenExec.encPortH c.ports[HmOut.nIn c]!
            (FlockVBridge.Decode.regVal zo (.inl (u, ⟨_, hq⟩))))
          (zo.rd.ws (.inl (u, ⟨_, hq⟩))) (zo.rd.wpath (.inl (u, ⟨_, hq⟩))) = some (x, y) ∧ x ≠ y ∧
        FlockSoundness.Refine.H512 x = FlockSoundness.Refine.H512 y := by
  obtain ⟨-, hor⟩ := tie_root zo.rd hp hroot hn
  exact hor.imp (fun he => hT_of_enc zo hj hl hob hlaid hq hp hseg htop (he.trans (congrArg rowInV hrow))) id

/-- **`opensTopAt_of_regs`' `hT` at R's level 0, by the read's leaf**: `hT_of_root` with the read's leaf in place of
the root. Where `bcs` are `cap`'s commit strings under the held salts `salt` (`TopOf`), the read's leaf is `treeLeaf`
of `bcs`'s string at its position (`hleaf`: `leaf_of_accepts` where the public file's digest there is that string) and
the node's row is `top`'s (`hrow`), each of the first 512 output columns is registered as `top`'s bit, or the read's
hm96 pair against the node under `salt` collides (`ReadsX.SaltClash`, `saltFinder`'s event at level 0). -/
theorem hT_of_leaf {u : Fin nv} {j : ℕ} (hj : j < c.upv) {LANES H : ℕ} {p : Ports} {outBits : Array ℕ}
    {name : String} (hl : Flock.HmNets.netOf name (recOpenLaid LANES H p outBits) = .ok c.unit)
    (hob : 2048 ≤ outBits.foldl (· + ·) 0) (hlaid : OutsLaid c j) (hq : HmOut.nIn c < c.ports.size)
    (hp : zo.rd.isRead (.inl (u, ⟨_, hq⟩))) (hseg : c.ports[HmOut.nIn c]!.segs = none)
    {lv : FlockSoundness.Accounting.Level} {cap : FlockSoundness.Model.Cap FlockSoundness.Refine.Dig}
    {salt : ℕ → ByteArray} {bcs : Array ByteArray} (htc : TopOf lv cap salt bcs)
    (hi : zo.rd.readAt (.inl (u, ⟨_, hq⟩)) < bcs.size)
    (hleaf : (zo.rd.leaf (.inl (u, ⟨_, hq⟩))).1 = Hm96.Default512.treeLeaf bcs[zo.rd.readAt (.inl (u, ⟨_, hq⟩))])
    {top : ByteArray} (htop : top.size = 64)
    (hrow : (capRows lv cap)[zo.rd.readAt (.inl (u, ⟨_, hq⟩))]! = capRow top) :
    (∀ t < 512, RegOut zo u hj (Hm.outStart ((recOpenNet LANES H p).run Flock.HmNets.St.empty).2 p.bits outBits + t)
        (bitL top.data.toList t)) ∨ zo.rd.SaltClash (capRows lv cap) salt :=
  (salt_tie zo.rd hp htc hi hleaf).imp
    (fun he => hT_of_enc zo hj hl hob hlaid hq hp hseg htop (he.trans (congrArg rowInV hrow))) id

/-- **The table's leaf at a row read** is `treeLeaf` of the public file's digest there, at every registration. -/
theorem regLeaf_inl (R : Reg') (u : Fin nv) (q : Fin c.ports.size) :
    (zo.lay.regLeaf R (.inl (u, q))).1 = Hm96.Default512.treeLeaf (zo.pub.digest u.val q.val c.ports.size) := by
  show (FlockSoundness.Refine.H512
    (FlockSoundness.Refine.K512.leafPrefix ++ (zo.pub.digest u.val q.val c.ports.size).data.toList)).1 = _
  rw [FlockSoundness.Refine.H512_val]
  show Sha512.hash
      ⟨(Hm96.Default512.leafPrefix.data.toList ++ (zo.pub.digest u.val q.val c.ports.size).data.toList).toArray⟩ =
    Sha512.hash (Hm96.Default512.leafPrefix ++ zo.pub.digest u.val q.val c.ports.size)
  rw [← FlockSoundness.Refine.data_append, FlockSoundness.Refine.bytes_of_list]

/-- **A registered row read's leaf is the table's, or its two openings conflict**: at an outcome `o` the session
accepts, where the reads `A` keeps are the verifier's own (`hown`), the registrant's leaf at a read `(u, q)` that `A`
keeps is `treeLeaf` of the public file's digest there (`regLeaf_inl`), or its opening under `o`'s registration through
the public file header's path (`regPathsHJ`) and the registrant's conflict (`LeafColl`'s event). -/
theorem leaf_of_accepts (o : Reg' × Finset (Fin nv) × FlockSoundness.Discharge.Exec.Transcript)
    (hacc : ZkReg.LiveAcceptsZKR zo.I zo.own o)
    {A : zo.lay.Pos → Prop}
    (hown : ZkReg.OwnReadsHJ zo.I zo.own c zo.pub o.2.2 (zo.rd.restrict A).isRead zo.rd.port zo.rd.readAt)
    {u : Fin nv} {q : Fin c.ports.size} (hp : zo.rd.isRead (.inl (u, q))) (hA : A (.inl (u, q))) :
    (zo.rd.leaf (.inl (u, q))).1 = Hm96.Default512.treeLeaf (zo.pub.digest u.val q.val c.ports.size) ∨
      (zo.rd.openings o.1 (ZkReg.regPathsHJ zo.I c zo.pub o.2.2 zo.rd.port) (.inl (u, q))).Conflict := by
  have hc := ZkReg.checked_of_acceptsHJ zo.I zo.ht zo.own zo.dj zo.y₀ zo.hcH hU zo.hpub
    (Layout.scopeOkH_of_scopeOk hscope) zo.hTags o hacc (zo.rd.restrict A) hown
  by_cases h : zo.lay.regLeaf o.1 (.inl (u, q)) = zo.rd.leaf (.inl (u, q))
  · rw [← h]
    exact Or.inl (regLeaf_inl zo o.1 u q)
  · exact Or.inr ⟨h, hc _ ⟨hp, hA⟩, zo.rd.committed _ hp⟩

end Session

end FlockC3.RegLevels

#print axioms FlockC3.RegLevels.hT_of_root

#print axioms FlockC3.RegLevels.hT_of_leaf

#print axioms FlockC3.RegLevels.leaf_of_accepts
