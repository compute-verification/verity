import Proofs.Flock.C3.RegLevels.TopTie
import Proofs.Flock.C3.RegLevels.TopRead
import Proofs.Flock.C3.RegLevels.AccReads
import Proofs.Flock.C3.PubRows.Check
import Proofs.Flock.C3.Reps.RoundLinks
import Proofs.Flock.Recursive.Registrants

/-!
# `hT_of_leaf`'s `hleaf` at R's levels' `out` reads, and two sessions' reads of one row

`hT_of_leaf` takes the read's leaf as `treeLeaf` of the node's commit string. At an accepted session:

* `fixedRow_top`, `heldRow_top`: `fixedRow` at `rec-top-stacked-L<l>` and row `x` is node `x` of R's level-`l` tops in
  `stacked/rep0` (`stackedTopLevel_top`; neither `cIndex` nor `openingIndex` reads the name);
* `port_of_topRead`, `read_of_topRead`: where the reads file names each port once (`hnd`), its `out` entry is
  `TopRead`'s, of value `rec-top-stacked-L<l>`, and unit `u`'s read there is at node `pos[u] >>> H`;
* `digest_top`: the public file's row at such a read is the node's commit string (`HPubRows` at the verifier's hold,
  `hrow`);
* `leaf_top`: so the read's leaf is `treeLeaf` of that string, or its two openings conflict (`leaf_of_accepts` at the
  outcome, with `ReadsOwnC3`'s port); `leaf_drawn`: `leaf_of_accepts` at a drawn outcome, whose reads the session's
  `hown` gives.

Two registrant records' reads at one root, leaf count and domain, at one position, have one row, or the records clash
(`enc_eq_of_not_clash`, from `TwoOpenings.collides` and `hm96Pair_spec`). At an accepted history no two of V*'s
sessions clash (`Accepted`'s `¬ RegColl`), so two sessions' reads of one registered value at one position have one row
where the two sessions' entries for that value share their root, leaf count and domain (`SharedOwnRoot`, a hypothesis:
nothing proved yet ties two sessions' reads files to each other; `row_agree`).
-/

namespace FlockSoundness.Registered

open Game Refine Binding Audit

section Agree

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : Binding.Layout P plan tab decode} {rs : HmRowsX lay}
  {C' : Circuit} {n' : ℕ} {P' : Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : Binding.Layout P' plan' tab' decode'} {rs' : HmRowsX lay'}

/-- **Two registered reads at one root, leaf count, domain and position have one row**, or the records clash: their
leaves differ and both open the root there (`TwoOpenings.collides`), or the leaves agree over two rows
(`hm96Pair_spec`). -/
theorem enc_eq_of_not_clash {rd : ReadsX lay rs} {rd' : ReadsX lay' rs'} (hc : ¬ rd.Clash rd') {p : lay.Pos}
    {p' : lay'.Pos} (hp : rd.isRead p) (hp' : rd'.isRead p') (hroot : (rd.port p).root = (rd'.port p').root)
    (hn : (rd.port p).leaves = (rd'.port p').leaves) (hdom : (rd.port p).domain = (rd'.port p').domain)
    (hpos : rd.readAt p = rd'.readAt p') : rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p') := by
  have hleaf : rd.leaf p = rd'.leaf p' := Classical.byContradiction fun hne =>
    hc ⟨p, p', hp, hp', Or.inl (TwoOpenings.collides
      ⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ ⟨hne, rd.committed p hp, by
      show Flock.Registered.opensLeaf (rd.port p) (rd'.leaf p').1 (rd.readAt p) (rd'.wpath p') = true
      unfold Flock.Registered.opensLeaf
      rw [hroot, hn, hdom, hpos]
      exact rd'.committed p' hp'⟩)⟩
  exact Classical.byContradiction fun hne => hc ⟨p, p', hp, hp', Or.inr
    (hm96Pair_spec (rs.enc p (rd.wv p)) (rd.ws p) (rs'.enc p' (rd'.wv p')) (rd'.ws p') hne hleaf)⟩

end Agree

end FlockSoundness.Registered

namespace FlockC3.RegLevels

open Flock
open Flock.RecVStar (cIndex openingIndex stackedTopLevel fixedRow InnerCommits InnerStream STACKED)
open FlockSoundness FlockSoundness.Audit FlockSoundness.Discharge FlockSoundness.Discharge.Recursion

/-! ## `fixedRow` at R's level tops -/

theorem stackedTopLevel_top (l : ℕ) : stackedTopLevel s!"rec-top-stacked-L{l}" = some l :=
  FlockC3.Reps.index_pre "rec-top-stacked-L" l rfl

theorem top_not_startsWith (l : ℕ) {p : String} (h5 : 5 ≤ p.toList.length) (hne : p.toList.take 5 ≠ "rec-t".toList) :
    ¬ (s!"rec-top-stacked-L{l}".startsWith p = true) := fun h => by
  obtain ⟨t, ht⟩ := String.startsWith_string_iff.1 h
  have := congrArg (List.take 5) ht
  simp only [String.toList_append] at this
  rw [List.take_append_of_le_length h5, List.take_append_of_le_length (by decide)] at this
  exact hne (this.trans (by decide))

theorem cIndex_top (l : ℕ) : cIndex s!"rec-top-stacked-L{l}" = none := by
  unfold cIndex
  simp only [top_not_startsWith l (p := "rec-c") (by decide) (by decide), Bool.false_eq_true, ↓reduceIte]

theorem openingIndex_top (l : ℕ) : openingIndex s!"rec-top-stacked-L{l}" = none := by
  unfold openingIndex
  simp only [top_not_startsWith l (p := "rec-opening-c") (by decide) (by decide), Bool.false_eq_true, ↓reduceIte]

/-- **`fixedRow` at `rec-top-stacked-L<l>`, row `x`**: node `x` of R's level-`l` tops. -/
theorem fixedRow_top {cs : InnerCommits} {l x : ℕ} {sparse : Option RecSparse.Shape} {s : InnerStream}
    {bcs : Array ByteArray} (hs : cs.stream? s!"{STACKED}/rep0" = some s) (ht : s.tops[l]? = some bcs)
    (hx : x < bcs.size) :
    fixedRow cs s!"rec-top-stacked-L{l}" x sparse = some (.ok (s!"node {x} of R's level {l} top", bcs[x])) := by
  unfold fixedRow
  rw [cIndex_top, openingIndex_top, stackedTopLevel_top]
  simp only [hs, ht, Option.bind_some, Array.getElem?_eq_getElem hx]

theorem heldRow_top {cs : InnerCommits} {sparse : Option RecSparse.Shape} {l x : ℕ} {s : InnerStream}
    {bcs : Array ByteArray} (hs : cs.stream? s!"{STACKED}/rep0" = some s) (ht : s.tops[l]? = some bcs)
    (hx : x < bcs.size) :
    FlockSoundness.Assumptions.heldRow (.ok cs) sparse s!"rec-top-stacked-L{l}" x = some bcs[x].toList := by
  simp only [FlockSoundness.Assumptions.heldRow, fixedRow_top hs ht hx]

theorem bytes_of_toList (b : ByteArray) : (⟨b.toList.toArray⟩ : ByteArray) = b := by
  rw [FlockC3.PubRows.bytes_toList_eq] <;> cases b <;> simp

/-! ## `TopRead`'s entry -/

/-- **The reads file's `out` entry is `TopRead`'s**, where it names each port once. -/
theorem port_of_topRead {own : Array Flock.Registered.Port} {l qq H : ℕ} {pos : Array ℕ}
    (ht : TopRead own l qq H pos) (hnd : ∀ p ∈ own, ∀ p' ∈ own, p.name = p'.name → p = p')
    {p : Flock.Registered.Port} (hp : p ∈ own) (hn : p.name = "out") :
    p.value = s!"rec-top-stacked-L{l}" ∧ p.reads = (Array.range qq).map fun j => some (pos[j]! >>> H) := by
  obtain ⟨pt, hf, hv, hr⟩ := ht
  obtain rfl : pt = p := hnd pt (Array.mem_of_find?_eq_some hf) p hp
    ((by simpa using Array.find?_some hf : pt.name = "out").trans hn)
  exact ⟨hv, hr⟩

/-- **Unit `u`'s read of `TopRead`'s entry is at node `pos[u] >>> H`**, one of the `qq` openings. -/
theorem read_of_topRead {qq H u x : ℕ} {pos : Array ℕ} {reads : Array (Option ℕ)}
    (hr : reads = (Array.range qq).map fun j => some (pos[j]! >>> H)) (hx : reads[u]? = some (some x)) :
    u < qq ∧ x = pos[u]! >>> H := by
  subst hr
  by_cases hu : u < qq
  · simp only [Array.getElem?_map, Array.getElem?_range, hu, ↓reduceIte, Option.map_some, Option.some.injEq] at hx
    exact ⟨hu, hx.symm⟩
  · simp [hu] at hx

/-! ## At a drawn outcome -/

section Session

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **`leaf_of_accepts` at a drawn outcome**: the session's `hown` gives the drawn units' reads. -/
theorem leaf_drawn {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : ZkReg.LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv}
    (hu : u ∈ Lv.draw ω) {q : Fin c.ports.size} (hp : zo.rd.isRead (.inl (u, q))) :
    (zo.rd.leaf (.inl (u, q))).1 = Hm96.Default512.treeLeaf (zo.pub.digest u.val q.val c.ports.size) ∨
      (zo.rd.openings R' (ZkReg.regPathsHJ zo.I c zo.pub (zo.wr (Lv.draw ω) R' o) zo.rd.port)
        (.inl (u, q))).Conflict :=
  leaf_of_accepts zo _ hacc (zo.hown ω R' o hacc) hp hu

end Session

/-! ## At the family's accepted sessions -/

section Family

open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled FlockC3.Compose

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The public file's row at a read of R's level-`l` tops is the node's commit string**: `HPubRows` at the verifier's
hold of the inner transcript (`hrow`), where it holds `stacked/rep0`'s level-`l` tops `bcs`. -/
theorem digest_top {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hpr : FlockC3.PubRows.HPubRows F)
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hrow : ∀ v x t, F.row v x t =
      FlockSoundness.Assumptions.heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) F.ish.sparse v x)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {j : Fin (vstarCount F.ish)}
    (hacc : ∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) {pt : Flock.Registered.Port}
    (hpt : pt ∈ (F.zs R ω t R₂ j).own) {l : ℕ} (hv : pt.value = s!"rec-top-stacked-L{l}")
    {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size} (hq : (F.Vs j).c.ports[q].name = pt.name) {x : ℕ}
    (hx : pt.reads[u.val]? = some (some x)) {cs : InnerCommits} {s : InnerStream} {bcs : Array ByteArray}
    (hcs : innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs) (hs : cs.stream? s!"{STACKED}/rep0" = some s)
    (ht : s.tops[l]? = some bcs) (hi : x < bcs.size) :
    (F.zs R ω t R₂ j).pub.digest u.val q.val (F.Vs j).c.ports.size = bcs[x] :=
  (hpr R ω t R₂ j hacc pt hpt u q x hq hx bcs[x].toList (by rw [hrow, hcs, hv]; exact heldRow_top hs ht hi)).trans
    (bytes_of_toList _)

/-- **`hT_of_leaf`'s `hleaf` at an accepted session**: where a registered read's port is `own`'s entry of its name
read at the unit's position (`ReadsOwnC3`), registering R's level-`l` tops, and the outcome's reads `A` keeps are the
verifier's own (`hown`), the read's leaf is `treeLeaf` of node `readAt`'s commit string in the verifier's hold, or its
two openings conflict (`LeafColl`'s event). -/
theorem leaf_top {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hpr : FlockC3.PubRows.HPubRows F)
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json}
    (hrow : ∀ v x t, F.row v x t =
      FlockSoundness.Assumptions.heldRow (innerCommitsAt (wr t).1 (wr t).2 st reg) F.ish.sparse v x)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnC3 F dg N k Rw)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j : Fin (vstarCount F.ish))
    {A : (F.zs R ω t R₂ j).lay.Pos → Prop}
    (hown : ZkReg.OwnReadsHJ (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own (F.Vs j).c (F.zs R ω t R₂ j).pub (o j).2.2
      ((F.zs R ω t R₂ j).rd.restrict A).isRead (F.zs R ω t R₂ j).rd.port (F.zs R ω t R₂ j).rd.readAt)
    {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size} (hp : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    (hA : A (.inl (u, q))) {l : ℕ}
    (hv : ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).value = s!"rec-top-stacked-L{l}")
    {cs : InnerCommits} {s : InnerStream} {bcs : Array ByteArray}
    (hcs : innerCommitsAt (wr t).1 (wr t).2 st reg = .ok cs) (hs : cs.stream? s!"{STACKED}/rep0" = some s)
    (ht : s.tops[l]? = some bcs) (hi : (F.zs R ω t R₂ j).rd.readAt (.inl (u, q)) < bcs.size) :
    ((F.zs R ω t R₂ j).rd.leaf (.inl (u, q))).1 =
        Hm96.Default512.treeLeaf bcs[(F.zs R ω t R₂ j).rd.readAt (.inl (u, q))] ∨
      ((F.zs R ω t R₂ j).rd.openings (o j).1 (ZkReg.regPathsHJ (F.zs R ω t R₂ j).I (F.Vs j).c
        (F.zs R ω t R₂ j).pub (o j).2.2 (F.zs R ω t R₂ j).rd.port) (.inl (u, q))).Conflict := by
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o h j u q hp
  rw [← digest_top hpr hrow ⟨_, (h.1 j).1⟩ hmem hv hname.symm hpos hcs hs ht hi]
  exact leaf_of_accepts (F.zs R ω t R₂ j) (o j) (h.1 j).1 hown hp hA

/-- Two of V*'s accepted sessions at a history register one value under one root, leaf count and domain. -/
def SharedOwnRoot (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) : Prop :=
  ∀ R ω t R₂ j j', (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) →
    (∃ tr, AcceptsZKR (F.zs R ω t R₂ j').I (F.zs R ω t R₂ j').own tr) →
    ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j').own, p.value = p'.value →
      p.root = p'.root ∧ p.leaves = p'.leaves ∧ p.domain = p'.domain

/-- **Two sessions' reads of one registered value at one position have one row** at an accepted history (no two
sessions clash, `Accepted`'s `¬ RegColl`), where the sessions' entries for the value share their root
(`SharedOwnRoot`) and each read's port is its reads file's entry (`ReadsOwnC3`). -/
theorem row_agree {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} (hsh : SharedOwnRoot F)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnC3 F dg N k Rw)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (j j' : Fin (vstarCount F.ish))
    {u : Fin (F.Vs j).nv} {q : Fin (F.Vs j).c.ports.size} (hp : (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)))
    {u' : Fin (F.Vs j').nv} {q' : Fin (F.Vs j').c.ports.size} (hp' : (F.zs R ω t R₂ j').rd.isRead (.inl (u', q')))
    (hv : ((F.zs R ω t R₂ j).rd.port (.inl (u, q))).value = ((F.zs R ω t R₂ j').rd.port (.inl (u', q'))).value)
    (hpos : (F.zs R ω t R₂ j).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ j').rd.readAt (.inl (u', q'))) :
    HiddenExec.encPortH (F.Vs j).c.ports[q.val]! (FlockVBridge.Decode.regVal (F.zs R ω t R₂ j) (.inl (u, q))) =
      HiddenExec.encPortH (F.Vs j').c.ports[q'.val]!
        (FlockVBridge.Decode.regVal (F.zs R ω t R₂ j') (.inl (u', q'))) := by
  obtain ⟨hroot, hn, hdom⟩ := hsh R ω t R₂ j j' ⟨_, (h.1 j).1⟩ ⟨_, (h.1 j').1⟩ _ (hro R ω t R₂ τ o h j u q hp).1 _
    (hro R ω t R₂ τ o h j' u' q' hp').1 hv
  exact FlockSoundness.Registered.enc_eq_of_not_clash (fun hc => h.2 ⟨j, j', hc⟩) hp hp' hroot hn hdom hpos

end Family

end FlockC3.RegLevels

#print axioms FlockSoundness.Registered.enc_eq_of_not_clash

#print axioms FlockC3.RegLevels.fixedRow_top

#print axioms FlockC3.RegLevels.heldRow_top

#print axioms FlockC3.RegLevels.bytes_of_toList

#print axioms FlockC3.RegLevels.port_of_topRead

#print axioms FlockC3.RegLevels.read_of_topRead

#print axioms FlockC3.RegLevels.leaf_drawn

#print axioms FlockC3.RegLevels.digest_top

#print axioms FlockC3.RegLevels.leaf_top

#print axioms FlockC3.RegLevels.row_agree
