import Proofs.Flock.VBridge.Family.Ends
import Proofs.Flock.VBridge.Family.Opened
import Proofs.Flock.VBridge.Family.Registrants

/-!
# VBridge G at V*'s family: `rec-acc`'s rows keyed by their registration, and the openings' tops

`Ends.lean` keys every read of `rec-acc`'s rows by one registered port `pa` (`EndKeys`, `OpenKeys`), the whole
`Registered.Port`, name and positions included. V*'s reads files name `rec-acc` once per row port, each entry with its
own name and positions (rec_outer.py:235-238: `acc` and `acc_out`, each `chain.reg.entry()` with the level's positions;
rec_vstage.py:357: `acc0`, `acc1`), and at a drawn unit `OwnReads` ties a registered read's port to the circuit's port
of its name. So:

* `eq_of_port_eq`: two registered reads of one drawn unit at one registered port are reads of one port of the circuit;
  `not_colKeyed_pair`, `not_endKeys`: `EndKeys` fails at a drawn algebra unit whose `acc0` and `acc1` columns are keyed
  by two ports (rec_residuals.py:101), as V*'s are;
* `colKeyed_drawn`: a keyed column of a drawn unit names its port in `own` and its position in the port's reads, so
  `OpenKeys` puts one port, with one name and one list of positions, in every level's reads file (`openKeys_drawn`),
  which V*'s files are not (each level's `acc` reads that level's positions).

The restatement keys a read by its registration (`site`: value, root, leaf count, row width, domain; not its name or
positions): `ColKeyedS`, `SiteRowsShared` (implies `RowsShared`, `rowsShared_of_site`; from `SharedRowsS`, and that
from `NoRegistrantCollisionS`, since the verifier's leaf check reads only a port's registration, `opensLeaf_site`;
all three hold where no read is segmented, `siteRowsShared` at `ReadsUnsegmented`, since `Accepted` excludes `RegColl`),
`AccRowsAreS` at `accRegS` (`accRowsAreS_of_shared`), `EndKeysS`, `OpenKeysS`, and `hEnds_of_partS`,
`hAccRows_of_keysS` as in `Ends.lean`.

`hTopsL_of_out` gives `HTopsL` from `TopsOutL`, the openings' `out` columns registered bits of the cap's entry, as
`AccOutL` gives `acc_out`.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open Flock.RecAlgebra (structureOf verifierRows el valOf wi)
open FlockVBridge.OfVStar (ReadCol rowCol partRep)
open FlockVBridge.Decode (RegBit)
open FlockVBridge.AccChain (levelLanes)
open FlockSoundness.Discharge.Hm (bitL)
open FlockSoundness.Discharge.ZkReg (LiveAcceptsZKR)

/-- The registration a registered port reads: its value, root, leaf count, row width and domain, all its fields but
its name and its read positions (`Registered.Port`; rec_outer.py:235, `read(chain.reg.entry(), …)`). -/
abbrev Site := String × ByteArray × ℕ × Option ℕ × ℕ × ByteArray

/-- A port's registration (`Site`). -/
def site (p : Flock.Registered.Port) : Site := (p.value, p.root, p.leaves, p.words, p.wordBits, p.domain)

/-! ## A drawn unit's registered reads -/

section Drawn

variable {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit} {hscope : Layout.scopeOk c = true}
  {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv)

/-- **Two registered reads of a drawn unit at one registered port read one port of the circuit** (`hown`: each read's
port is the circuit's port of its name, `OwnReads`). -/
theorem eq_of_port_eq {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv} (hu : u ∈ Lv.draw ω)
    {q q' : Fin c.ports.size} (hr : zo.rd.isRead (.inl (u, q))) (hr' : zo.rd.isRead (.inl (u, q')))
    (hp : zo.rd.port (.inl (u, q)) = zo.rd.port (.inl (u, q'))) : q = q' := by
  obtain ⟨u₁, q₁, -, -, -, -, h₁, -, hf₁, -⟩ := zo.hown ω R' o hacc (.inl (u, q)) ⟨hr, hu⟩
  obtain ⟨u₂, q₂, -, -, -, -, h₂, -, hf₂, -⟩ := zo.hown ω R' o hacc (.inl (u, q')) ⟨hr', hu⟩
  simp only [Sum.inl.injEq, Prod.mk.injEq] at h₁ h₂
  obtain ⟨rfl, rfl⟩ := h₁
  obtain ⟨rfl, rfl⟩ := h₂
  rw [hp] at hf₁
  exact Fin.ext (Option.some.inj (hf₁.symm.trans hf₂))

/-- **Two columns of a drawn unit keyed by two ports of the circuit are not keyed at one registered port**: `ColKeyed`
at one `pa` makes the ports' registered reads one port's (`eq_of_port_eq`). -/
theorem not_colKeyed_pair {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv} (hu : u ∈ Lv.draw ω)
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q)))
    {col col' k k' pos pos' q₀ q₁ b₀ e₀ b₁ e₁ : ℕ} {pa : Flock.Registered.Port} (hne : q₀ ≠ q₁)
    (hk₀ : Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + col) = .msg q₀ b₀ e₀)
    (hk₁ : Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + col') = .msg q₁ b₁ e₁)
    (h : ColKeyed zo u col k pa pos) (h' : ColKeyed zo u col' k' pa pos') : False := by
  obtain ⟨-, q, -, hkey, hown, hport, -⟩ := h
  obtain ⟨-, q', -, hkey', hown', hport', -⟩ := h'
  have e := eq_of_port_eq zo hacc hu (hread q hown) (hread q' hown') (hport.trans hport'.symm)
  rw [hk₀] at hkey
  rw [hk₁] at hkey'
  injection hkey with h₁
  injection hkey' with h₂
  subst e
  exact hne (h₁.trans h₂.symm)

/-- **A keyed column of a drawn unit names its registered port in `own`, by the name of its key's port, and its
position among the port's reads** (`hown`, `OwnReads`). -/
theorem colKeyed_drawn {ω : Lv.Ω} {R' : Reg'} {o}
    (hacc : LiveAcceptsZKR zo.I zo.own (R', Lv.draw ω, zo.wr (Lv.draw ω) R' o)) {u : Fin nv} (hu : u ∈ Lv.draw ω)
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q)))
    {col k pos : ℕ} {pa : Flock.Registered.Port} (h : ColKeyed zo u col k pa pos) :
    pa ∈ zo.own ∧ (∃ q : Fin c.ports.size, Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + col) = .msg q.val (k / 8) (k % 8) ∧
      (List.range c.ports.size).find? (fun j => c.ports[j]!.name == pa.name) = some q.val) ∧
      ∃ i : ℕ, pa.reads[i]? = some (some pos) := by
  obtain ⟨-, q, -, hkey, hown, hport, hat⟩ := h
  obtain ⟨u₁, q₁, d, D, dp, idx, h₁, hmem, hf, -, -, -, -, -, hpos⟩ :=
    zo.hown ω R' o hacc (.inl (u, q)) ⟨hread q hown, hu⟩
  simp only [Sum.inl.injEq, Prod.mk.injEq] at h₁
  obtain ⟨rfl, rfl⟩ := h₁
  rw [hport] at hmem hf hpos
  rw [hat] at hpos
  exact ⟨hmem, ⟨q, hkey, hf⟩, _, hpos⟩

/-- **A column keyed at the registration `s`** (`ColKeyed`, the read's port's `site` `s` in place of the port): its Δ
key is bit `k` of row port `q`, the session's reads file names `q`, and unit `u`'s row `q` is read at a port of
registration `s`, position `pos`. -/
def ColKeyedS (u : Fin nv) (col k : ℕ) (s : Site) (pos : ℕ) : Prop :=
  ∃ (_ : col < (Rows.ofNet hU).nIn) (q : Fin c.ports.size) (_ : k < Layout.portRowW c q),
    Aliased.srcKey c (0 * (Rows.ofNet hU).nIn + col) = .msg q.val (k / 8) (k % 8) ∧
      (∃ pt ∈ zo.own, pt.name = c.ports[q].name) ∧ site (zo.rd.port (.inl (u, q))) = s ∧
      zo.rd.readAt (.inl (u, q)) = pos

theorem colKeyedS_of {u : Fin nv} {col k pos : ℕ} {pa : Flock.Registered.Port} (h : ColKeyed zo u col k pa pos) :
    ColKeyedS zo u col k (site pa) pos := by
  obtain ⟨hc, q, hk, hkey, hown, hport, hat⟩ := h
  exact ⟨hc, q, hk, hkey, hown, by rw [hport], hat⟩

/-- **A column keyed at a registration is a registered bit of the registrant's row** (`regBit_of_colKeyed`). -/
theorem regBit_of_colKeyedS {u : Fin nv} {col k : ℕ} {s : Site} {pos : ℕ} {K : ℕ} (hkK : k < K)
    (hread : ∀ q : Fin c.ports.size, (∃ pt ∈ zo.own, pt.name = c.ports[q].name) → zo.rd.isRead (.inl (u, q)))
    (row : ℕ → ℕ → Bool)
    (hval : ∀ q : Fin c.ports.size, zo.rd.isRead (.inl (u, q)) → site (zo.rd.port (.inl (u, q))) = s →
      ∀ b < K, ∀ (hb : b < Layout.portRowW c q),
        (Decode.regVal zo (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          row (zo.rd.readAt (.inl (u, q))) b)
    (h : ColKeyedS zo u col k s pos) : RegBit zo u 0 col (row pos k) := by
  obtain ⟨hc, q, hk, hkey, hown, hport, hat⟩ := h
  have hr := hread q hown
  refine ⟨hc, q, k, hk, hkey, hr, ?_⟩
  rw [hval q hr hport k hkK hk, hat]

end Drawn

/-! ## Two registrants' records at one registration and position -/

section Two

open FlockSoundness.Registered (ReadsX TwoOpenings)
open FlockSoundness.Binding (HmRowsX Collides hm96Pair hm96Pair_spec)

variable {C : FlockSoundness.Audit.Circuit} {n : ℕ} {P : FlockSoundness.Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (FlockSoundness.TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : FlockSoundness.Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}
  {lay : FlockSoundness.Binding.Layout P plan tab decode} {rs : HmRowsX lay}
  {C' : FlockSoundness.Audit.Circuit} {n' : ℕ} {P' : FlockSoundness.Audit.Partition C' n'} {Reg₁ : Type} {mPts' : ℕ}
  {plan' : Finset (Fin n') → Reg₁ → List (FlockSoundness.TabSpec mPts')}
  {tab' : (S : Finset (Fin n')) → (R : Reg₁) → Fin n' → Fin (plan' S R).length}
  {decode' : FlockSoundness.Audit.Partition.DecoderB (C := C') (Reg := Reg₁) plan' tab'}
  {lay' : FlockSoundness.Binding.Layout P' plan' tab' decode'} {rs' : HmRowsX lay'}

/-- The verifier's leaf check reads a port's registration only (`opensLeaf`: its leaf count, domain and root). -/
theorem opensLeaf_site {p p' : Flock.Registered.Port} (h : site p = site p') :
    Flock.Registered.opensLeaf p = Flock.Registered.opensLeaf p' := by
  simp only [site, Prod.mk.injEq] at h
  obtain ⟨-, hr, hl, -, -, hd⟩ := h
  funext a b c
  simp only [Flock.Registered.opensLeaf, hr, hl, hd]

/-- **`registrants_eq_or_collides` at one registration**: two registered reads of one registration (`site`) at one
position carry one row's bytes, or their openings collide, or hm96's leaves do (`opensLeaf_site`). -/
theorem registrants_eq_or_collides_site (rd : ReadsX lay rs) (rd' : ReadsX lay' rs') {p : lay.Pos} {p' : lay'.Pos}
    (hp : rd.isRead p) (hp' : rd'.isRead p') (hsite : site (rd.port p) = site (rd'.port p'))
    (hat : rd.readAt p = rd'.readAt p') :
    rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p') ∨
      Collides (⟨rd.port p, rd.readAt p, rd.leaf p, rd.wpath p, rd'.leaf p', rd'.wpath p'⟩ : TwoOpenings).pair ∨
      Collides (hm96Pair (rs.enc p (rd.wv p)) (rd.ws p) (rs'.enc p' (rd'.wv p')) (rd'.ws p')) := by
  by_cases hl : rd.leaf p = rd'.leaf p'
  · by_cases he : rs.enc p (rd.wv p) = rs'.enc p' (rd'.wv p')
    · exact Or.inl he
    · exact Or.inr (Or.inr (hm96Pair_spec _ _ _ _ he hl))
  · refine Or.inr (Or.inl (TwoOpenings.collides _ ⟨hl, rd.committed p hp, ?_⟩))
    show Flock.Registered.opensLeaf (rd.port p) (rd'.leaf p').1 (rd.readAt p) (rd'.wpath p') = true
    rw [opensLeaf_site hsite, hat]
    exact rd'.committed p' hp'

end Two

/-! ## The openings' tops -/

section Tops

variable {Q : VStmt} {Reg' : Type} (zo : Q.Outer Reg')

/-- **An opening's `out`, from its registered output columns** (`copyVal_regOut`, as `accOut_of_regOut`): in a `Good`
layer, when the first `8 · |B|` output columns of copy 0 are registered bits of `B`, the bytes there are `B`. -/
theorem bytesAt_of_regOut {bits : Q.Bits} (hG : zo.Good bits) {u : Fin Q.nv} (hj : 0 < Q.c.upv) {LANES H : ℕ}
    (hp : Flock.RecOpen.params Q.c.unit.name = some (LANES, H)) {B : ByteArray}
    (h : ∀ t < 8 * B.size, AccChain.RegOut zo u hj (outCol LANES H + t) (bitL B.data.toList t)) :
    bytesAt (colOf Q bits u) (outCol LANES H) B.size = B := by
  have hnIn : (Rows.ofNet Q.hU).nIn ≤ outCol LANES H := by
    rw [Layout.ofNet_nIn_eq, inWords_unit (Flock.RecOpen.check_ok (parse_facts zo.hc).recNet hp).2]
    exact Decode.inRows_le_outStart _ _ _
  refine bytesAt_of_cols fun t ht => ?_
  rw [colOf_eq Q bits u hj]
  exact AccChain.copyVal_regOut zo hG.2 (keyProg_correct Q.hU Q.nv hG.1 u) hj (by omega) (h t ht)

end Tops

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)

/-! ## `EndKeys` and `OpenKeys` at one registered port -/

/-- **`EndKeys` fails at a drawn algebra unit whose `acc0` and `acc1` columns are keyed by two ports**: at an accepted
outer proof whose algebra session `sA r` accepts at a draw holding `uA r` (`hlive`, `hu`), where bit 0 of `acc0 0` and
of `acc1 0` are keyed by ports `q₀ ≠ q₁` (V*'s `acc0`, `acc1`, rec_residuals.py:101), `EndKeys` reads both at one
port, so one port of the circuit (`not_colKeyed_pair`). -/
theorem not_endKeys {pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port}
    {sA : Fin 2 → Fin F.stmts.size} {uA : (r : Fin 2) → Fin (F.Vs (sA r)).nv} {col0 col1 : Fin 2 → ℕ → ℕ → ℕ}
    (hk : F.EndKeys dg N k Rw pa sA uA col0 col1) (hreg : F.RowsRegistered dg N k Rw) {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o} (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (r : Fin 2)
    {ω' : (F.Vs (sA r)).law.live.Ω} {R' : Reg'} {o'}
    (hlive : LiveAcceptsZKR (F.zs R ω t R₂ (sA r)).I (F.zs R ω t R₂ (sA r)).own
      (R', (F.Vs (sA r)).law.live.draw ω', (F.zs R ω t R₂ (sA r)).wr ((F.Vs (sA r)).law.live.draw ω') R' o'))
    (hu : uA r ∈ (F.Vs (sA r)).law.live.draw ω') {q₀ q₁ b₀ e₀ b₁ e₁ : ℕ} (hne : q₀ ≠ q₁)
    (hk₀ : Aliased.srcKey (F.Vs (sA r)).c (0 * (Rows.ofNet (F.Vs (sA r)).hU).nIn + col0 r 0 0) = .msg q₀ b₀ e₀)
    (hk₁ : Aliased.srcKey (F.Vs (sA r)).c (0 * (Rows.ofNet (F.Vs (sA r)).hU).nIn + col1 r 0 0) = .msg q₁ b₁ e₁) :
    False :=
  not_colKeyed_pair _ hlive hu (hreg R ω t R₂ τ o h (sA r) (uA r)) hne hk₀ hk₁
    (hk R ω t R₂ τ o h r 0 (by omega) 0 (by omega)).1 (hk R ω t R₂ τ o h r 0 (by omega) 0 (by omega)).2

/-- **`OpenKeys` puts its port in every level's reads file, with every opening's position among its reads**: at an
accepted outer proof whose level-`ℓ` session accepts at a draw holding opening `j`'s unit, `pa` is one of the
session's `own`, by the name of the `acc` column's port, and reads the opening's chain position
(`colKeyed_drawn`). V*'s level files name `rec-acc` as `acc` with that level's positions only (rec_outer.py:236), so no
one port is in two levels' files with both levels' positions. -/
theorem openKeys_drawn {pa : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Flock.Registered.Port}
    (hk : F.OpenKeys dg N k Rw pa) (hreg : F.RowsRegistered dg N k Rw) {R : Reg} {ω : L.Ω}
    {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o} (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (r : Fin 2)
    {ℓ : ℕ} (hℓ : ℓ < F.sh.sched.levels.size) {j : ℕ} (hj : j < F.q ℓ)
    {ω' : (F.Vs (F.sessO ℓ)).law.live.Ω} {R' : Reg'} {o'}
    (hlive : LiveAcceptsZKR (F.zs R ω t R₂ (F.sessO ℓ)).I (F.zs R ω t R₂ (F.sessO ℓ)).own
      (R', (F.Vs (F.sessO ℓ)).law.live.draw ω',
        (F.zs R ω t R₂ (F.sessO ℓ)).wr ((F.Vs (F.sessO ℓ)).law.live.draw ω') R' o'))
    (hu : F.unO ℓ r j ∈ (F.Vs (F.sessO ℓ)).law.live.draw ω') :
    pa R ω t R₂ ∈ (F.zs R ω t R₂ (F.sessO ℓ)).own ∧
      ∃ i : ℕ, (pa R ω t R₂).reads[i]? = some (some (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j)) := by
  obtain ⟨hmem, -, hpos⟩ := colKeyed_drawn _ hlive hu (hreg R ω t R₂ τ o h _ _)
    (hk R ω t R₂ τ o h r ℓ hℓ j hj 0 (by omega)).1
  exact ⟨hmem, hpos⟩

/-! ## `rec-acc`'s rows keyed by their registration -/

/-- Obligation: discharged by `SharedRowsS` (`siteRowsShared_of_sharedRowsS`, `siteRowsShared`), as `RowsShared` by
`SharedRows`; the transfer's `cr/sha-512` term does not price it.
**One registrant row per registration and position** (`RowsShared` at `site`): at every accepted outer proof, two
sessions' registered reads of one registration (`site`: value, root, leaf count, row width, domain) at one position
carry the same registrant bits, whatever the ports' names and read positions. -/
def SiteRowsShared : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    site ((F.zs R ω t R₂ s).rd.port (.inl (u, q))) = site ((F.zs R ω t R₂ s').rd.port (.inl (u', q'))) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ∀ (b : ℕ) (hb : b < Layout.portRowW (F.Vs s).c q) (hb' : b < Layout.portRowW (F.Vs s').c q'),
      (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))).2
          ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
        (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))).2
          ⟨b, lt_of_lt_of_le hb' ((Finset.le_sup (Finset.mem_univ q')).trans (Nat.le_succ _))⟩

/-- **`RowsShared` from one registrant row per registration and position**: one port is one registration. -/
theorem rowsShared_of_site (h : F.SiteRowsShared dg N k Rw) : F.RowsShared dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hport hat b hb hb'
  exact h R ω t R₂ τ o ha s s' u u' q q' hr hr' (congrArg site hport) hat b hb hb'

section Shared

open FlockSoundness.Registered (TwoOpenings)
open FlockSoundness.Binding (Collides hm96Pair)
open FlockSoundness.Discharge.HiddenExec (encPortH)

/-- Obligation: discharged, as `NoRegistrantCollision`, by `Accepted`'s `¬ RegColl` (`noRegistrantCollisionS`); no
finder behind the transfer's `cr/sha-512` term prices it.
**No two sessions' registrant records collide at one registration and position** (`NoRegistrantCollision` at
`site`). -/
def NoRegistrantCollisionS : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    site ((F.zs R ω t R₂ s).rd.port (.inl (u, q))) = site ((F.zs R ω t R₂ s').rd.port (.inl (u', q'))) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    ¬ Collides (⟨(F.zs R ω t R₂ s).rd.port (.inl (u, q)), (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)),
        (F.zs R ω t R₂ s).rd.leaf (.inl (u, q)), (F.zs R ω t R₂ s).rd.wpath (.inl (u, q)),
        (F.zs R ω t R₂ s').rd.leaf (.inl (u', q')), (F.zs R ω t R₂ s').rd.wpath (.inl (u', q'))⟩ : TwoOpenings).pair ∧
      ¬ Collides (hm96Pair
        (encPortH (F.Vs s).c.ports[q.val]! (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))))
        ((F.zs R ω t R₂ s).rd.ws (.inl (u, q)))
        (encPortH (F.Vs s').c.ports[q'.val]! (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q'))))
        ((F.zs R ω t R₂ s').rd.ws (.inl (u', q'))))

/-- Obligation: discharged by `NoRegistrantCollisionS` (`sharedRowsS`).
**Two sessions' registrant records at one registration and position hold one row's bytes** (`SharedRows` at
`site`). -/
def SharedRowsS : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ (s s' : Fin F.stmts.size)
    (u : Fin (F.Vs s).nv) (u' : Fin (F.Vs s').nv) (q : Fin (F.Vs s).c.ports.size) (q' : Fin (F.Vs s').c.ports.size),
    (F.zs R ω t R₂ s).rd.isRead (.inl (u, q)) → (F.zs R ω t R₂ s').rd.isRead (.inl (u', q')) →
    site ((F.zs R ω t R₂ s).rd.port (.inl (u, q))) = site ((F.zs R ω t R₂ s').rd.port (.inl (u', q'))) →
    (F.zs R ω t R₂ s).rd.readAt (.inl (u, q)) = (F.zs R ω t R₂ s').rd.readAt (.inl (u', q')) →
    encPortH (F.Vs s).c.ports[q.val]! (Decode.regVal (F.zs R ω t R₂ s) (.inl (u, q))) =
      encPortH (F.Vs s').c.ports[q'.val]! (Decode.regVal (F.zs R ω t R₂ s') (.inl (u', q')))

/-- **`SharedRowsS` where no two registrant records collide** (`registrants_eq_or_collides_site`). -/
theorem sharedRowsS_of (h : F.NoRegistrantCollisionS dg N k Rw) : F.SharedRowsS dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hsite hat
  obtain ⟨h1, h2⟩ := h R ω t R₂ τ o ha s s' u u' q q' hr hr' hsite hat
  rcases registrants_eq_or_collides_site (F.zs R ω t R₂ s).rd (F.zs R ω t R₂ s').rd hr hr' hsite hat with he | hc | hc
  · exact he
  · exact absurd hc h1
  · exact absurd hc h2

/-- **`SiteRowsShared` from one row's bytes at unsegmented reads** (`SharedRowsS`, `ReadsUnsegmented`,
`bits_of_encPortH`). -/
theorem siteRowsShared_of_sharedRowsS (h : F.SharedRowsS dg N k Rw) (hu : F.ReadsUnsegmented dg N k Rw) :
    F.SiteRowsShared dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' hsite hat b hb hb'
  exact bits_of_encPortH (hu R ω t R₂ τ o ha s u q hr) (hu R ω t R₂ τ o ha s' u' q' hr')
    (h R ω t R₂ τ o ha s s' u u' q q' hr hr' hsite hat) b _ _

/-- **No two sessions' registrant records collide at one registration and position** (`NoRegistrantCollisionS`): both
collisions are `RegColl`'s, which `Accepted` excludes. -/
theorem noRegistrantCollisionS : F.NoRegistrantCollisionS dg N k Rw := by
  intro R ω t R₂ τ o ha s s' u u' q q' hr hr' _ _
  exact ⟨fun hc => ha.2.1 ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inl hc⟩,
    fun hc => ha.2.1 ⟨s, s', .inl (u, q), .inl (u', q'), hr, hr', Or.inr hc⟩⟩

/-- **`SharedRowsS`** (`noRegistrantCollisionS`). -/
theorem sharedRowsS : F.SharedRowsS dg N k Rw := F.sharedRowsS_of dg N k Rw (F.noRegistrantCollisionS dg N k Rw)

/-- **`SiteRowsShared` where no read is segmented** (`sharedRowsS`). -/
theorem siteRowsShared (hu : F.ReadsUnsegmented dg N k Rw) : F.SiteRowsShared dg N k Rw :=
  F.siteRowsShared_of_sharedRowsS dg N k Rw (F.sharedRowsS dg N k Rw) hu

end Shared

/-- **The registrant's rows of registration `s` are `V`'s** (`AccRowsAre` at `site`). It holds at `V := accRegS s`
from `SiteRowsShared` (`accRowsAreS_of_shared`). -/
def AccRowsAreS (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ j (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), (F.zs R ω t R₂ j).rd.isRead (.inl (u, q)) →
      site ((F.zs R ω t R₂ j).rd.port (.inl (u, q))) = s R ω t R₂ → ∀ b < 256,
        ∀ (hb : b < Layout.portRowW (F.Vs j).c q),
        (Decode.regVal (F.zs R ω t R₂ j) (.inl (u, q))).2
            ⟨b, lt_of_lt_of_le hb ((Finset.le_sup (Finset.mem_univ q)).trans (Nat.le_succ _))⟩ =
          bitL (MerkleScheme.rowBytes (V R ω t R₂ ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))))).data.toList b

/-- A registered read of registration `s` at position `p`, in some session, whose row has bit `b`. -/
def AccReadS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p b : ℕ)
    (x : Σ j : Fin F.stmts.size, Fin (F.Vs j).nv × Fin (F.Vs j).c.ports.size) : Prop :=
  (F.zs R ω t R₂ x.1).rd.isRead (.inl (x.2.1, x.2.2)) ∧
    site ((F.zs R ω t R₂ x.1).rd.port (.inl (x.2.1, x.2.2))) = s R ω t R₂ ∧
    (F.zs R ω t R₂ x.1).rd.readAt (.inl (x.2.1, x.2.2)) = p ∧ b < Layout.portRowW (F.Vs x.1).c x.2.2

open Classical in
/-- Bit `b` of the registrant's row of registration `s` at position `p`: its bit at some registered read there
(`AccReadS`), `false` if there is none. -/
noncomputable def accBitS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p b : ℕ) : Bool :=
  if h : ∃ x, F.AccReadS s R ω t R₂ p b x then
    (Decode.regVal (F.zs R ω t R₂ (Classical.choose h).1)
      (.inl ((Classical.choose h).2.1, (Classical.choose h).2.2))).2
        ⟨b, lt_of_lt_of_le (Classical.choose_spec h).2.2.2 ((Finset.le_sup (Finset.mem_univ _)).trans (Nat.le_succ _))⟩
  else false

/-- **`rec-acc`'s rows as registered at registration `s`**: row `p`'s first two elements are the registrant's
(`accBitS`). -/
noncomputable def accRegS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p : ℕ) : Array F128 :=
  #[ofBits fun e => F.accBitS s R ω t R₂ p e, ofBits fun e => F.accBitS s R ω t R₂ p (128 + e)]

theorem bitL_accRegS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (R : Reg) (ω : L.Ω)
    (t : List (Cm × (ℐ).Coin)) (R₂ : Reg₂) (p : ℕ) {b : ℕ} (hb : b < 256) :
    bitL (MerkleScheme.rowBytes (F.accRegS s R ω t R₂ p)).data.toList b = F.accBitS s R ω t R₂ p b := by
  have hl : (F.accRegS s R ω t R₂ p).toList =
      [ofBits fun e => F.accBitS s R ω t R₂ p e, ofBits fun e => F.accBitS s R ω t R₂ p (128 + e)] := rfl
  have hm : b % 128 < 128 := Nat.mod_lt _ (by norm_num)
  rw [bitL_rowBytes, hl]
  simp only [FlockVBridge.wbit]
  rcases (show b / 128 = 0 ∨ b / 128 = 1 by omega) with h0 | h0
  · simp only [h0, List.getD_cons_zero]
    rw [bitOf_ofBits _ hm, show b % 128 = b by omega]
  · simp only [h0, List.getD_cons_succ, List.getD_cons_zero]
    rw [bitOf_ofBits _ hm, show 128 + b % 128 = b by omega]

/-- **`AccRowsAreS` at `rec-acc`'s rows as registered** (`accRegS`), from one registrant row per registration and
position (`SiteRowsShared`). -/
theorem accRowsAreS_of_shared (hshared : F.SiteRowsShared dg N k Rw)
    (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) :
    F.AccRowsAreS dg N k Rw (F.accRegS s) s := by
  intro R ω t R₂ τ o h j u q hread hsite b hb256 hb
  rw [F.bitL_accRegS s R ω t R₂ _ hb256]
  have hex : ∃ x, F.AccReadS s R ω t R₂ ((F.zs R ω t R₂ j).rd.readAt (.inl (u, q))) b x :=
    ⟨⟨j, (u, q)⟩, hread, hsite, rfl, hb⟩
  unfold accBitS
  rw [dite_eq_left hex]
  have hx := Classical.choose_spec hex
  exact (hshared R ω t R₂ τ o h _ j _ u _ q hx.1 hread (hx.2.1.trans hsite.symm) hx.2.2.1 b hx.2.2.2 hb).symm

/-- **`AccRowsAreS` at `rec-acc`'s rows as registered** (`accRegS`; `siteRowsShared`), where no read is segmented. -/
theorem accRowsAreS (hu : F.ReadsUnsegmented dg N k Rw) (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) :
    F.AccRowsAreS dg N k Rw (F.accRegS s) s :=
  F.accRowsAreS_of_shared dg N k Rw (F.siteRowsShared dg N k Rw hu) s

/-- Obligation: discharged by the parse of the part's circuit (its keys), the session's reads file (`acc0`, `acc1`
read `rec-acc` at the rep's ends, rec_vstage.py:355-357) and a verifier check that every session's `rec-acc` entry is
one registration.
**The algebra unit's end columns are keyed by `rec-acc`'s rows at the rep's ends, at registration `s`** (`EndKeys`
at `site`): bit `b` of `acc0 i` and of `acc1 i` at rep `r`'s algebra unit is bit `128 i + b` of a row `own` names,
read at a port of registration `s` at the rep's first row `r · nOpen` and its last `r · nOpen + nOpen`. -/
def EndKeysS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (sA : Fin 2 → Fin F.stmts.size)
    (uA : (r : Fin 2) → Fin (F.Vs (sA r)).nv) (col0 col1 : Fin 2 → ℕ → ℕ → ℕ) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ i < 2, ∀ b < 128,
    ColKeyedS (F.zs R ω t R₂ (sA r)) (uA r) (col0 r i b) (128 * i + b) (s R ω t R₂) (r.val * F.nOpen) ∧
      ColKeyedS (F.zs R ω t R₂ (sA r)) (uA r) (col1 r i b) (128 * i + b) (s R ω t R₂) (r.val * F.nOpen + F.nOpen)

/-- **`HEnds` from the registrant's rows at a registration** (`hEnds_of_regs` at `regBit_of_colKeyedS`). -/
theorem hEnds_of_regsS (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (sA : Fin 2 → Fin F.stmts.size)
    (uA : (r : Fin 2) → Fin (F.Vs (sA r)).nv) (col0 col1 : Fin 2 → ℕ → ℕ → ℕ)
    (hd0 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col0 r i b)
        (bitOf (valOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).cons (.acc0 i)) b))
    (hd1 : ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ i < 2, ∀ b < 128,
      Decode.RegBit (F.zs R ω t R₂ (sA r)) (uA r) 0 (col1 r i b)
        (bitOf (valOf F.sh (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).w
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).comb
          (F.rep R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r).cons (.acc1 i)) b))
    (hreg : F.RowsRegistered dg N k Rw) (hacc : F.AccRowsAreS dg N k Rw V s)
    (hkeys : F.EndKeysS dg N k Rw s sA uA col0 col1) : F.HEnds dg N k Rw V :=
  F.hEnds_of dg N k Rw V sA uA col0 col1 hd0
    (fun R ω t R₂ τ o h r i hi b hb => regBit_of_colKeyedS _ (K := 256) (by omega) (hreg R ω t R₂ τ o h (sA r) (uA r))
      (fun p b => bitL (MerkleScheme.rowBytes (V R ω t R₂ p)).data.toList b) (hacc R ω t R₂ τ o h (sA r) (uA r))
      (hkeys R ω t R₂ τ o h r i hi b hb).1)
    hd1
    (fun R ω t R₂ τ o h r i hi b hb => regBit_of_colKeyedS _ (K := 256) (by omega) (hreg R ω t R₂ τ o h (sA r) (uA r))
      (fun p b => bitL (MerkleScheme.rowBytes (V R ω t R₂ p)).data.toList b) (hacc R ω t R₂ τ o h (sA r) (uA r))
      (hkeys R ω t R₂ τ o h r i hi b hb).2)

/-- **`HEnds` from `HResL` and the registrant's rows at a registration** (`hEnds_of_part` at `EndKeysS`). -/
theorem hEnds_of_partS {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)}
    (hst : F.HStruct GS gs) (hres : F.HResL dg N k Rw gs) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) {pA : ℕ} {a0 a1 : ℕ → ℕ}
    (hin : F.EndsInPart pA a0 a1) (hreg : F.RowsRegistered dg N k Rw) (hacc : F.AccRowsAreS dg N k Rw V s)
    (hkeys : F.EndKeysS dg N k Rw s (fun _ => F.sess pA) (F.un pA) (fun _ i b => rowColOf F.sh pA (128 * a0 i + b))
      (fun _ i b => rowColOf F.sh pA (128 * a1 i + b))) :
    F.HEnds dg N k Rw V :=
  F.hEnds_of_regsS dg N k Rw V s (fun _ => F.sess pA) (F.un pA) (fun _ i b => rowColOf F.sh pA (128 * a0 i + b))
    (fun _ i b => rowColOf F.sh pA (128 * a1 i + b))
    (fun R ω t R₂ τ o h r i hi b hb => by
      obtain ⟨⟨ha, hwi⟩, -, hrd⟩ := hin.2 i hi
      rw [valOf_acc0 hwi]
      exact F.regBit_of_hResL dg N k Rw hst hres h r hin.1 ha hb (hrd b hb).1)
    (fun R ω t R₂ τ o h r i hi b hb => by
      obtain ⟨-, ⟨ha, hwi⟩, hrd⟩ := hin.2 i hi
      rw [valOf_acc1 hwi]
      exact F.regBit_of_hResL dg N k Rw hst hres h r hin.1 ha hb (hrd b hb).2)
    hreg hacc hkeys

/-- Obligation: discharged by the parse of the open circuit (its keys, `Flock.RecOpen`), the level's reads file (`acc`
reads `rec-acc` at the level's chain positions, rec_outer.py:236) and a verifier check that every session's `rec-acc`
entry is one registration.
**Each opening's `acc` columns are keyed by `rec-acc`'s row at its chain position, at registration `s`** (`OpenKeys`
at `site`). -/
def OpenKeysS (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r : Fin 2, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ, ∀ b < 256,
      ColKeyedS (F.zs R ω t R₂ (F.sessO ℓ)) (F.unO ℓ r j) (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b) b (s R ω t R₂)
          (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j) ∧
        ReadCol (Rows.ofNet (F.Vs (F.sessO ℓ)).hU) (accCol (levelLanes F.sh ℓ) (F.depth ℓ) + b)

/-- **`HAccRows` from the registrant's rows at a registration and the openings' outputs** (`hAccRows_of_keys` at
`OpenKeysS`, `AccRowsAreS`). -/
theorem hAccRows_of_keysS (hU : F.HOpenUnits) (V : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → ℕ → Array F128)
    (s : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → Site) (hreg : F.RowsRegistered dg N k Rw)
    (hacc : F.AccRowsAreS dg N k Rw V s) (hk : F.OpenKeysS dg N k Rw s) (hout : F.AccOutL dg N k Rw V) :
    F.HAccRows dg N k Rw V := by
  intro R ω t R₂ τ o h r ℓ hℓ j hj i hi
  obtain ⟨hup, -⟩ := hU ℓ hℓ
  refine ⟨accIn_of_regBit (F.zs R ω t R₂ (F.sessO ℓ)) (h.1 (F.sessO ℓ)).2.2 (u := F.unO ℓ r j) hup
    (base := accCol (levelLanes F.sh ℓ) (F.depth ℓ))
    (A := V R ω t R₂ (r.val * F.nOpen + ((List.range ℓ).map F.q).sum + j)) (fun b hb => ?_) (fun b hb hc => ?_) i hi,
    hout R ω t R₂ τ o h r ℓ hℓ j hj i hi⟩
  · exact regBit_of_colKeyedS _ (K := 256) hb (hreg R ω t R₂ τ o h _ _)
      (fun p b => bitL (MerkleScheme.rowBytes (V R ω t R₂ p)).data.toList b) (hacc R ω t R₂ τ o h _ _)
      (hk R ω t R₂ τ o h r ℓ hℓ j hj b hb).1
  · obtain ⟨⟨hc', -⟩, hrd⟩ := hk R ω t R₂ τ o h r ℓ hℓ j hj b hb
    exact ioColAt_of_readCol (F.unO ℓ r j) hc' hrd hup hc

/-! ## The openings' tops -/

variable (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

/-- Obligation: to be discharged by acceptance, now that the layout keys a returned output gate to its output row
(`HiddenExec.rowOfH`), with each level's `out` a read of the
tops registration at the opening's cap index whose rows are the cap message's entries (rec_outer.py:226-228, 236: the
prover checks `out_rows` against `R.expected(q.cap_node)`; the cap message, the transcript lane's `MsgsRead`).
**The openings' `out` columns are registered bits of the cap's entry**: at every accepted outer proof, the cap entry
`HTopsL` names is 64 bytes, and the first 512 output columns of rep `r`'s opening `j` of level `ℓ` are registered bits
of it (`RegOut`). -/
def TopsOutL : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → ∀ r, ∀ ℓ < F.sh.sched.levels.size,
    ∀ j < F.q ℓ,
      ((capAt (cap (layers F.zs dg N k Rw R ω t R₂ τ)) (lig (layers F.zs dg N k Rw R ω t R₂ τ) r) ℓ).getD
          ((F.posOf R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ)[j]! >>> F.depth ℓ) ByteArray.empty).size = 64 ∧
        ∀ (hj : 0 < (F.Vs (F.sessO ℓ)).c.upv), ∀ t' < 512,
          AccChain.RegOut (F.zs R ω t R₂ (F.sessO ℓ)) (F.unO ℓ r j) hj
            (outCol (levelLanes F.sh ℓ) (F.depth ℓ) + t')
            (bitL ((capAt (cap (layers F.zs dg N k Rw R ω t R₂ τ)) (lig (layers F.zs dg N k Rw R ω t R₂ τ) r) ℓ).getD
              ((F.posOf R ω t (layers F.zs dg N k Rw R ω t R₂ τ) r ℓ)[j]! >>> F.depth ℓ)
                ByteArray.empty).data.toList t')

/-- **`HTopsL` from the openings' registered `out` columns** (`TopsOutL`, `bytesAt_of_regOut`). -/
theorem hTopsL_of_out (hU : F.HOpenUnits) (hout : F.TopsOutL dg N k Rw cap lig) : F.HTopsL dg N k Rw cap lig := by
  intro R ω t R₂ τ o h r ℓ hℓ j hj
  obtain ⟨hup, -, H, hp⟩ := hU ℓ hℓ
  obtain ⟨hsz, hb⟩ := hout R ω t R₂ τ o h r ℓ hℓ j hj
  have hp' : Flock.RecOpen.params (F.Vs (F.sessO ℓ)).c.unit.name = some (levelLanes F.sh ℓ, F.depth ℓ) := by
    rw [show F.depth ℓ = H by rw [depth, hp]; rfl]
    exact hp
  have e := bytesAt_of_regOut (F.zs R ω t R₂ (F.sessO ℓ)) (h.1 (F.sessO ℓ)).2 (u := F.unO ℓ r j) hup
    hp' fun t' ht' => hb hup t' (by rw [hsz] at ht'; omega)
  rw [hsz] at e
  exact e

end VStarFamily

end family

end FlockVBridge.Family
