import Proofs.Flock.Recursive.HPortTie

/-!
# Each rep's claim word is R's `rec-claim` row, across two units' reads of one registered row

RecSparse's `claim` port reads `rec-claim` at row `un` (`RecVStar.ownPorts (.sparse ..)`, which `checkOwn` enforces), and
R's opening's `claim<i>` reads it at row `i`. Where `--rec-rows` registers `rec-claim` (`SharedValue rows CLAIM_VALUE`),
`checkShared` holds both reads to its entry, so, as at `h` (`HPortTie.lean`), the two reads at one registration and
position carry one row's bits or two records clash, which `Accepted` excludes.

* `ClaimPortReads` (named), `claim_eq_of_reads`: at a RecSparse copy at w, w's element `wc` is `x` where each of its
  columns' keys is a registered row bit that some unit's registered read, at the same registration and position, holds
  as `x`'s bit.
* `claimIndex_claim`, `mIndex_claim`, `mem_ownPorts_sparseClaim`, `mem_ownPorts_opening`, `readAt_of_reads01`,
  `readAt_of_reads1`: what `checkOwn` pins for the two ports.
* `claimSite_of_checked`: the two reads' registration and position (`checkOwn`, `checkShared`).
* `HClaimKeys`, `HOpenClaimRows` (named), `claim_word_of`, `claimWords_of`: each rep's claim word is R's `rec-claim` row
  (W's `ClaimWords`, unfolded).
-/

namespace FlockC3.HPort

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockC3.Compose
open FlockVBridge (wbit wbit_elem bitOf)
open FlockVBridge.Decode (RegBit regVal)
open FlockVBridge.OfVStar (ReadCol rowCol)
open FlockVBridge.AccChain (eq_of_bitOf)

section point

variable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type}

/-- Obligation: the cross-unit read of w's element `wc`, at copy `j` of unit `un` of session `s` (a RecSparse copy) and
R's claim word `x`. Each column of the element's bit `e` (message column `128 wc + e`) is an input column a row reads,
and where its key is bit `b` of a registered row `q` of the copy's unit, that row is unsegmented and some session `s'`'s
unit `u'` reads a registered unsegmented row `q'` holding bit `b`, at the same registration and position, whose
registrant bit `b` is bit `e` of `x`. `claim_word_of` gives it from the verifier's checks and two named pieces. -/
def ClaimPortReads (zs : (j : Fin S) → (Vs j).Outer Reg') (mp : List (String × ℕ)) (s : Fin S) (un : Fin (Vs s).nv)
    (j wc : ℕ) (x : Flock.F128) : Prop :=
  (∀ e < 128, ReadCol (FlockSoundness.Rows.ofNet (Vs s).hU) (rowCol mp 0 (128 * wc + e))) ∧
    ∀ e < 128, ∀ (q : Fin (Vs s).c.ports.size) (b : ℕ), b < Layout.portRowW (Vs s).c q →
      Aliased.srcKey (Vs s).c (j * (FlockSoundness.Rows.ofNet (Vs s).hU).nIn + rowCol mp 0 (128 * wc + e)) =
          .msg q.val (b / 8) (b % 8) →
      (zs s).rd.isRead (.inl (un, q)) →
      (Vs s).c.ports[q.val]!.segs = none ∧
        ∃ (s' : Fin S) (u' : Fin (Vs s').nv) (q' : Fin (Vs s').c.ports.size) (hb' : b < Layout.portRowW (Vs s').c q'),
          (zs s').rd.isRead (.inl (u', q')) ∧ (Vs s').c.ports[q'.val]!.segs = none ∧
          FlockVBridge.Family.site ((zs s).rd.port (.inl (un, q))) =
            FlockVBridge.Family.site ((zs s').rd.port (.inl (u', q'))) ∧
          (zs s).rd.readAt (.inl (un, q)) = (zs s').rd.readAt (.inl (u', q')) ∧
          (regVal (zs s') (.inl (u', q'))).2
              ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
            bitOf x e

/-- **w's element `wc` is `x`** at a RecSparse copy at w (`SparseCopyAt`'s registered message bits), under the
cross-unit read (`ClaimPortReads`) and no clash (`regVal_bit_eq`), as `w_eq_of_reads` at `h`. -/
theorem claim_eq_of_reads {zs : (j : Fin S) → (Vs j).Outer Reg'} {Lb : (j : Fin S) → (Vs j).Bits} {e : ExecHolo}
    {w : Array Flock.F128} {c : FlockVBridge.Sparse.SCoins} {s : Fin S} {un : Fin (Vs s).nv} {j : ℕ}
    {hj : j < (Vs s).c.upv} {wc : ℕ} {x : Flock.F128} (hnc : ¬ ∃ j j', (zs j).rd.Clash (zs j').rd)
    (hcp : FlockC3.Sparse.SparseCopyAt zs Lb e w c s un j hj)
    (hwc : wc < ((Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape e)).map (·.2)).sum)
    (hx : ClaimPortReads zs (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape e)) s un j wc x) :
    w[wc]! = x := by
  refine eq_of_bitOf fun t ht => ?_
  have hq : 128 * wc + t < 128 * ((Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape e)).map (·.2)).sum := by
    omega
  obtain ⟨-, q, b, hb, hkey, hread, hval⟩ := hcp.2.2.2.2.1 _ hq (hx.1 t ht)
  obtain ⟨hseg, s', u', q', hb', hread', hseg', hsite, hat, hv⟩ := hx.2 t ht q b hb hkey hread
  rw [← hv, ← regVal_bit_eq zs hnc hread hread' hsite hat hseg hseg' b hb hb']
  exact (wbit_elem (fun k => w[k]!) wc ht).symm.trans hval.symm

end point

section checks

/-- `claim<n>`'s `claimIndex` is `n`. -/
theorem claimIndex_claim (n : ℕ) : Flock.RecVStar.claimIndex s!"claim{n}" = some n := by
  have hm : (s!"claim{n}").toList = 'c' :: 'l' :: 'a' :: 'i' :: 'm' :: (Nat.repr n).toList := by
    rw [String.toList_append]; rfl
  have hd : ((s!"claim{n}").drop 5).toString = Nat.repr n := by
    apply String.toList_inj.mp
    rw [String.Slice.toString_eq, String.toList_copy_drop, hm]; rfl
  have hs : (s!"claim{n}").startsWith "claim" = true := by
    rw [String.startsWith_string_iff, hm]; exact ⟨_, rfl⟩
  unfold Flock.RecVStar.claimIndex
  rw [ite_eq_left hs, hd, Nat.toNat?_repr]
  simp

/-- `claim<n>` is no message round's port. -/
theorem mIndex_claim (n : ℕ) : Flock.RecVStar.mIndex s!"claim{n}" = none := by
  have hm : (s!"claim{n}").toList = 'c' :: 'l' :: 'a' :: 'i' :: 'm' :: (Nat.repr n).toList := by
    rw [String.toList_append]; rfl
  have hs : ¬ (s!"claim{n}").startsWith "m" = true := by
    rw [String.startsWith_string_iff, hm]
    rintro ⟨_, h⟩
    cases h
  unfold Flock.RecVStar.mIndex
  rw [ite_eq_right hs]

/-- **RecSparse's `claim` reads `rec-claim` at both reps' rows** (`ownPorts`). -/
theorem mem_ownPorts_sparseClaim {sh : Flock.RecSparse.Shape} {hr : ℕ} :
    ("claim", Flock.RecVStar.CLAIM_VALUE, #[some 0, some 1]) ∈ Flock.RecVStar.ownPorts (.sparse sh hr) := by
  unfold Flock.RecVStar.ownPorts
  exact List.mem_cons_of_mem _ (List.mem_cons_self ..)

/-- **R's opening's `claim<k>` reads `rec-claim` at row `k`** (`ownPorts`). -/
theorem mem_ownPorts_opening {m cl jp nv k x : ℕ} {S : Flock.RecOpen.GfS} {ports : List (String × ℕ)}
    (hx : (s!"claim{k}", x) ∈ ports) :
    (s!"claim{k}", Flock.RecVStar.CLAIM_VALUE, #[some k]) ∈
      Flock.RecVStar.ownPorts (.opening m cl jp S ports nv) := by
  unfold Flock.RecVStar.ownPorts
  exact List.mem_filterMap.2 ⟨_, hx, by dsimp only; simp only [mIndex_claim, claimIndex_claim]⟩

theorem readAt_of_reads01 {u r : ℕ} (h : (#[some 0, some 1] : Array (Option ℕ))[u]? = some (some r)) : r = u := by
  rcases u with _ | _ | u <;> simp at h <;> omega

theorem readAt_of_reads1 {k u r : ℕ} (h : (#[some k] : Array (Option ℕ))[u]? = some (some r)) : r = k := by
  rcases u with _ | u <;> simp at h
  omega

end checks

section family

open FlockC3.PubRows (ChecksAccepted)
open FlockC3.RegLevels (SharedValue site_of_checkShared site_of_portDiff)

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {xR : HoloReg GF128 Dig} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **The sparse unit's `claim` and R's opening's `claim<i>` reads have one registration and position**: `checkOwn` puts
both reads' entries (`ReadsOwnAt`, the reads file's names unique) under `rec-claim`, the first at both reps' rows and the
second at row `i`, and where `--rec-rows` registers `rec-claim` (`hsv`), `checkShared` holds each to its first entry of
that value in every field but reads (`site_of_checkShared`), so the copy at unit `i` and R's read one row. -/
theorem claimSite_of_checked {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnAt F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    (hsv : SharedValue rows Flock.RecVStar.CLAIM_VALUE)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {s s' : Fin (vstarCount F.ish)} {sh : Flock.RecSparse.Shape}
    {hr : ℕ} (hss : F.stmts[(s : ℕ)]? = some (.sparse sh hr)) {m cl jp nv : ℕ} {S0 : Flock.RecOpen.GfS}
    {ports : List (String × ℕ)} (hss' : F.stmts[(s' : ℕ)]? = some (.opening m cl jp S0 ports nv)) {i x : ℕ}
    (hx : (s!"claim{i}", x) ∈ ports) {un : Fin (F.Vs s).nv} {u' : Fin (F.Vs s').nv} {q : Fin (F.Vs s).c.ports.size}
    {q' : Fin (F.Vs s').c.ports.size} (hq : (F.Vs s).c.ports[q].name = "claim")
    (hq' : (F.Vs s').c.ports[q'].name = s!"claim{i}") (hu : un.val = i)
    (hrd : (F.zs R ω t R₂ s).rd.isRead (.inl (un, q))) (hrd' : (F.zs R ω t R₂ s').rd.isRead (.inl (u', q'))) :
    FlockVBridge.Family.site ((F.zs R ω t R₂ s).rd.port (.inl (un, q))) =
        FlockVBridge.Family.site ((F.zs R ω t R₂ s').rd.port (.inl (u', q'))) ∧
      (F.zs R ω t R₂ s).rd.readAt (.inl (un, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) := by
  have chk : ∀ j : Fin (vstarCount F.ish), ∃ sj, F.stmts[(j : ℕ)]? = some sj ∧
      Flock.RecVStar.checkOwn sj (F.zs R ω t R₂ j).own = .ok () ∧
      Flock.RecVStar.checkShared sj (F.zs R ω t R₂ j).own rows = .ok () := fun j => by
    obtain ⟨_, -, -, _, _, hset, hfm⟩ := hchk R ω t R₂ j ⟨_, (h.1 j).1⟩
    obtain ⟨h0, -⟩ := setup_zero_c (F.zs R ω t R₂ j).ht (F.zs R ω t R₂ j).hTags.1 (F.zs R ω t R₂ j).hcH hset
    obtain ⟨ss, sj, hss₁, hj₁, ho, -, hsh⟩ := regChecks_of_checkStmt (array_forM_ok hfm 0 h0)
    obtain rfl : ss = F.stmts := Except.ok.inj (hss₁.symm.trans F.hstmts)
    exact ⟨sj, hj₁, ho, hsh⟩
  obtain ⟨hmem, hname, hpos⟩ := hro R ω t R₂ τ o h s un q hrd
  obtain ⟨hmem', hname', hpos'⟩ := hro R ω t R₂ τ o h s' u' q' hrd'
  obtain ⟨sa, hsa, hoa, hsha⟩ := chk s
  obtain rfl : sa = .sparse sh hr := Option.some.inj (hsa.symm.trans hss)
  obtain ⟨sb, hsb, hob, hshb⟩ := chk s'
  obtain rfl : sb = .opening m cl jp S0 ports nv := Option.some.inj (hsb.symm.trans hss')
  obtain ⟨hva, hra⟩ := own_of_checkOwn hoa mem_ownPorts_sparseClaim hmem (hname.trans hq) (hnd R ω t R₂ s)
  obtain ⟨hvb, hrb⟩ := own_of_checkOwn hob (mem_ownPorts_opening hx) hmem' (hname'.trans hq') (hnd R ω t R₂ s')
  obtain ⟨w₀, hw₀, hd₀⟩ := site_of_checkShared hsha hmem (by rw [hva]; exact hsv)
  obtain ⟨w₁, hw₁, hd₁⟩ := site_of_checkShared hshb hmem' (by rw [hvb]; exact hsv)
  rw [hva] at hw₀
  rw [hvb, hw₀] at hw₁
  obtain rfl := Option.some.inj hw₁
  refine ⟨(site_of_portDiff hd₀).trans (site_of_portDiff hd₁).symm, ?_⟩
  rw [hra] at hpos
  rw [hrb] at hpos'
  exact (readAt_of_reads01 hpos).trans (hu.trans (readAt_of_reads1 hpos').symm)

/-- Obligation (the RecSparse circuit's parse): each column of w's element `wc`, bit `e`, at copy `j` is an input column a
row reads, and where its key is bit `b` of row `q`, `q` is the port named `claim` and `b` is `e` (its one element). -/
def HClaimKeys (V : VStmt) (mp : List (String × ℕ)) (j wc : ℕ) : Prop :=
  (∀ e < 128, ReadCol (FlockSoundness.Rows.ofNet V.hU) (rowCol mp 0 (128 * wc + e))) ∧
    ∀ e < 128, ∀ (q : Fin V.c.ports.size) (b : ℕ), b < Layout.portRowW V.c q →
      Aliased.srcKey V.c (j * (FlockSoundness.Rows.ofNet V.hU).nIn + rowCol mp 0 (128 * wc + e)) =
          .msg q.val (b / 8) (b % 8) →
      V.c.ports[q].name = "claim" ∧ b = e

/-- Obligation (R's opening's decode of its claims): at every accepted outer proof and each `i`, some unit of R's
opening's session (`.opening`, with a port `claim<i>`) reads its row port `claim<i>`, whose bit `e` is bit `e` of R's
word at `claimW sh i` (`wR`, as `SessionAt` holds it). -/
def HOpenClaimRows (F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
    (sh : Flock.Shape)
    (wR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Array Flock.F128) :
    Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ i : Fin 2,
    ∃ (s' : Fin (vstarCount F.ish)) (u' : Fin (F.Vs s').nv) (q' : Fin (F.Vs s').c.ports.size) (m cl jp nv x : ℕ)
      (S0 : Flock.RecOpen.GfS) (ports : List (String × ℕ)),
      F.stmts[(s' : ℕ)]? = some (.opening m cl jp S0 ports nv) ∧ (s!"claim{i.val}", x) ∈ ports ∧
      (F.Vs s').c.ports[q'].name = s!"claim{i.val}" ∧ (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) ∧
      ∀ e < 128, ∃ hb : e < Layout.portRowW (F.Vs s').c q',
        (regVal (F.zs R ω t R₂ s') (.inl (u', q'))).2
            ⟨e, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩ =
          bitOf (wR R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[Flock.RecOpening.claimW sh i.val]! e

/-- **Rep `i`'s claim word is R's `claim<i>` row** at a RecSparse copy at unit `i` of a `.sparse` session
(`claim_eq_of_reads`): the copy's `claim` keys (`HClaimKeys`) and R's opening's row (`HOpenClaimRows`) are two reads at
one registration and position (`claimSite_of_checked`), unsegmented (`readsUnsegC3`). -/
theorem claim_word_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnAt F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    (hsv : SharedValue rows Flock.RecVStar.CLAIM_VALUE)
    (hkeys : ∀ (s : Fin (vstarCount F.ish)) (j : ℕ), Flock.RecSparse.check (F.Vs s).c = .ok () →
      Flock.RecSparse.params (F.Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      HClaimKeys (F.Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))) j
        (FlockVBridge.Sparse.wClaim (holoAt st hp).cls.shape))
    {sh : Flock.Shape}
    {wR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Array Flock.F128}
    (hop : HOpenClaimRows F dg N k Rw sh wR) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hA : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {i : Fin 2} {w : Array Flock.F128}
    {c : FlockVBridge.Sparse.SCoins} {s : Fin (vstarCount F.ish)} {un : Fin (F.Vs s).nv} {j : ℕ}
    {hj : j < (F.Vs s).c.upv}
    (hcp : FlockC3.Sparse.SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp) w c s un j hj)
    (hun : un.val = i.val) (hss : ∃ ssh hr, F.stmts[(s : ℕ)]? = some (.sparse ssh hr)) :
    w[FlockVBridge.Sparse.wClaim (holoAt st hp).cls.shape]! =
      (wR R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[Flock.RecOpening.claimW sh i.val]! := by
  obtain ⟨ssh, hr, hss⟩ := hss
  obtain ⟨s', u', q', m, cl, jp, nv, x, S0, ports, hss', hx, hq', hrd', hbits⟩ := hop R ω t R₂ τ o hA i
  have hk := hkeys s j hcp.1 hcp.2.1
  have hwc : FlockVBridge.Sparse.wClaim (holoAt st hp).cls.shape <
      ((Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))).map (·.2)).sum :=
    Nat.lt_of_lt_of_le (Nat.lt_succ_self _) (FlockVBridge.Sparse.nw_le_msum (FlockC3.Sparse.unitShape (holoAt st hp)))
  refine claim_eq_of_reads hA.2.1 hcp hwc ⟨hk.1, fun e he q b hb hkey hread => ?_⟩
  obtain ⟨hq, rfl⟩ := hk.2 e he q b hb hkey
  obtain ⟨hb', hv⟩ := hbits _ he
  obtain ⟨hst, hat⟩ := claimSite_of_checked hchk hro hnd hsv hA hss hss' hx hq hq' hun hread hrd'
  have hseg := readsUnsegC3 F dg N k Rw R ω t R₂ τ o hA
  exact ⟨hseg s un q hread, s', u', q', hb', hrd', hseg s' u' q' hrd', hst, hat, hv⟩

/-- **W's `ClaimWords` at an accepted outer proof** (`claim_word_of` at each rep): where each rep `i`'s words `w i` are
held by a RecSparse copy at unit `i` of a `.sparse` session (`SparseSessionAt`'s copy, at the unit it does not yet
state), each rep's claim word is R's `rec-claim` row. -/
theorem claimWords_of {F : C3Family T CT st hp xR L Reg Cm Reg' Reg₂} {rows : Option (Array Flock.Registered.Port)}
    {wr : List (Cm × (ℐ).Coin) → Lean.Json × String} {reg : Lean.Json} (hchk : ChecksAccepted F rows wr reg)
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} (hro : ReadsOwnAt F dg N k Rw)
    (hnd : ∀ R ω t R₂ j, ∀ p ∈ (F.zs R ω t R₂ j).own, ∀ p' ∈ (F.zs R ω t R₂ j).own, p.name = p'.name → p = p')
    (hsv : SharedValue rows Flock.RecVStar.CLAIM_VALUE)
    (hkeys : ∀ (s : Fin (vstarCount F.ish)) (j : ℕ), Flock.RecSparse.check (F.Vs s).c = .ok () →
      Flock.RecSparse.params (F.Vs s).c.unit.name = some (FlockC3.Sparse.unitShape (holoAt st hp)) →
      HClaimKeys (F.Vs s) (Flock.RecSparse.messagePorts (FlockC3.Sparse.unitShape (holoAt st hp))) j
        (FlockVBridge.Sparse.wClaim (holoAt st hp).cls.shape))
    {sh : Flock.Shape}
    {wR : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin (vstarCount F.ish)) → (F.Vs j).Bits) → Array Flock.F128}
    (hop : HOpenClaimRows F dg N k Rw sh wR) {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (hA : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) {w : Fin 2 → Array Flock.F128}
    {c : Fin 2 → FlockVBridge.Sparse.SCoins}
    (hs : ∀ i : Fin 2, ∃ (s : Fin (vstarCount F.ish)) (un : Fin (F.Vs s).nv) (j : ℕ) (hj : j < (F.Vs s).c.upv),
      un.val = i.val ∧ (∃ ssh hr, F.stmts[(s : ℕ)]? = some (.sparse ssh hr)) ∧
      FlockC3.Sparse.SparseCopyAt (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) (holoAt st hp) (w i) (c i) s un j
        hj) :
    ∀ i : Fin 2, fOf (w i)[FlockVBridge.Sparse.wClaim (holoAt st hp).cls.shape]! =
      fOf (wR R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[Flock.RecOpening.claimW sh i.val]! := fun i => by
  obtain ⟨s, un, j, hj, hun, hss, hcp⟩ := hs i
  exact congrArg fOf (claim_word_of hchk hro hnd hsv hkeys hop hA hcp hun hss)

end family

end

end FlockC3.HPort
