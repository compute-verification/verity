import Proofs.Flock.C3.Compose.SparseEnds
import Proofs.Flock.C3.Reps.IoCols
import Proofs.Flock.C3.Opening.Session

/-!
# RsRead's facts

`rsRead_of_layer` takes four named facts. Here they come from facts closer to V*'s files, circuits and reads file:

* `OneCopy` from one session per part (`OpenSess`) and the `rec-opening` statements' one unit of one copy
  (`OpenOne`: `nv = 1`, `upv = 1`), `oneCopy_of`.
* `OpenAll` from V*'s sessions (`SessionAt`'s (a)), `openAll_of_session`.
* `RsSk` from the reads file at the ports keying the ring switches' columns (`RsSel`, S's `SelPort`), `rsSk_of_sel`:
  the column's key (`OpeningZeroAt`'s `RegBit`) puts its wire at the port's commit string (`posK_of_keyAt`).
* `RsIo` from the unit reading its row ports' columns (`RsReadCol`, S's `ReadCol`), `rsIo_of_readCol`: copy `j` reads
  a column its computed rows read (`mem_io_of_readCol`, `ioColAt_of_readCol` at copy `j`).

`rsRead_of_facts` composes them.
-/

namespace FlockC3.Compose

noncomputable section

open FlockSoundness FlockSoundness.Discharge FlockSoundness.Discharge.Recursion
open Game Audit Exec Refine Model ZkSession ZkLink ZkExec Composed ZkReg
open Flock FlockLevel3 Accounting ZkBind Integrate Placed StrictCR DecodeZL FlockVBridge.Compiled
open FlockVBridge (keyInst)
open FlockVBridge.Family (skOf mem_skOf_inl)
open FlockVBridge.OfVStar (OpeningZeroAt rowCol partRep part_eqOpening wIdx_size ReadCol)
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

/-- **The `rec-opening` statements are one copy of one unit** (named, V*'s files): a session whose unit is a
`rec-opening` part's has one unit (`.opening`'s one instance, `VStarParsesScoped`'s `nv`) of one copy (`upv = 1`,
the `units_per_vu` of `metaHead`'s META line as `HmRow.parse` reads it). -/
def OpenOne {S : ℕ} (Vs : Fin S → VStmt) : Prop :=
  ∀ (s : Fin S) (m cl p : ℕ), RecOpening.params (Vs s).c.unit.name = some (m, cl, p) →
    (Vs s).nv = 1 ∧ (Vs s).c.upv = 1

variable {nT r : ℕ} {T : Fin nT → Type} {CT : Fin r → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
  {st : Flock.Stmt} {hp : HoloParams} {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type} {S : ℕ} {Vs : Fin S → VStmt}

local notation "ℐ" => holoInnerAt T CT st hp

/-- **One session per part** (named, V*'s list): the copies registering w at accepted outer proofs as part `p` are in
one session. -/
def OpenSess (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ gs (s s' : Fin S) (un : Fin (Vs s).nv) (un' : Fin (Vs s').nv) (j j' p : ℕ),
    OpenCopy zs dg N k Rw sh w gs s un j p → OpenCopy zs dg N k Rw sh w gs s' un' j' p → s = s'

/-- **`OneCopy` from one session per part and its one unit of one copy.** -/
theorem oneCopy_of {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    (hs : OpenSess zs dg N k Rw sh w) (h1 : OpenOne Vs) : OneCopy zs dg N k Rw sh w := by
  intro gs s s' un un' j j' p hc hc'
  obtain rfl := hs gs s s' un un' j j' p hc hc'
  obtain ⟨_, _, _, _, _, _, _, -, _, hj, -, -, -, m, cl, hpm, -⟩ := hc
  obtain ⟨_, _, _, _, _, _, _, -, _, hj', -⟩ := hc'
  obtain ⟨hn, hu⟩ := h1 s m cl p hpm
  refine ⟨rfl, ?_, ?_⟩
  · rw [Nat.lt_one_iff.mp (lt_of_lt_of_eq un.isLt hn), Nat.lt_one_iff.mp (lt_of_lt_of_eq un'.isLt hn)]
  · rw [Nat.lt_one_iff.mp (lt_of_lt_of_eq hj hu), Nat.lt_one_iff.mp (lt_of_lt_of_eq hj' hu)]

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

/-- **A copy's row-port column is a keyed input** (`OpeningZeroAt`'s `RegBit`): at a copy registering w as part `p`,
bit `b` of the part's row-port element `q` is a column below the unit's inputs' end, keyed at copy `j` by a bit of a
row port. -/
theorem openCopy_key {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {s : Fin S} {un : Fin (Vs s).nv} {j p : ℕ}
    (hO : OpenCopy zs dg N k Rw sh w gs s un j p) {S0 : RecAlgebra.GfS} (hS : RecOpening.structureOf sh = .ok S0)
    (hg : Flock.RecResiduals.groups S0 (RecOpening.portSizes sh) = .ok gs) (hpg : p < gs.size) {S₁ : RecOpen.GfS}
    {ns : List ℕ} {nv : ℕ} (hpart : RecOpening.part sh p = .ok (S₁, ns, nv)) {q e : ℕ}
    (he : (wIdx (RecOpening.portSizes sh) gs[p].2.1)[q]? = some e) {b : ℕ} (hb : b < 128) :
    128 * q + b < 128 * ns.sum ∧ rowCol (partRep ns nv).ports 0 (128 * q + b) < (Rows.ofNet (Vs s).hU).nIn ∧
      ∃ (q' : Fin (Vs s).c.ports.size) (kk : ℕ) (_ : kk < Layout.portRowW (Vs s).c q'),
        Aliased.srcKey (Vs s).c (j * (Rows.ofNet (Vs s).hU).nIn + rowCol (partRep ns nv).ports 0 (128 * q + b)) =
          .msg q'.val (kk / 8) (kk % 8) := by
  obtain ⟨_, _, _, _, _, _, _, -, hz⟩ := hO
  obtain ⟨_, _, _, _, _, _, _, _, _, S₂, ns₂, nv₂, hpart₂, hw, -⟩ := hz
  obtain ⟨rfl, rfl, rfl⟩ : S₁ = S₂ ∧ ns = ns₂ ∧ nv = nv₂ := by
    simpa only [Prod.mk.injEq] using Except.ok.inj (hpart.symm.trans hpart₂)
  obtain ⟨-, hns, -⟩ := part_eqOpening hS hg hpg hpart
  obtain ⟨hk, -⟩ := Array.getElem?_eq_some_iff.mp he
  have hq : q < ns.sum := by rw [hns, ← wIdx_size]; exact hk
  obtain ⟨hcol, q', kk, hkk, hkey, -⟩ := hw (128 * q + b) (by omega)
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
  obtain ⟨-, hcol, q', kk, hkk, hkey⟩ := openCopy_key hO hS hg hpg hpart he hb
  rw [posK_of_keyAt un hj hcol hkk hkey hc]
  exact mem_skOf_inl.2 (hsel i hi gs s un j p hO hpg S₁ ns nv hpart q e t he hwi b hb q' kk hkk hkey)

/-- **The `rec-opening` unit reads its row ports' columns** (named, S's `ReadCol`): at a copy registering w as part
`p`, each column of the part's row ports is one a computed row of the unit reads. -/
def RsReadCol (zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg')
    (dg : ByteArray → ByteArray) (N k Rw : ℕ) (sh : Flock.Shape)
    (w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128) : Prop :=
  ∀ gs (s : Fin S) (un : Fin (Vs s).nv) (j p : ℕ), OpenCopy zs dg N k Rw sh w gs s un j p →
    ∀ (S₁ : RecOpen.GfS) (ns : List ℕ) (nv : ℕ), RecOpening.part sh p = .ok (S₁, ns, nv) →
      ∀ c < 128 * ns.sum, ReadCol (Rows.ofNet (Vs s).hU) (rowCol (partRep ns nv).ports 0 c)

/-- **`RsIo` from the unit reading its row ports' columns** (`openCopy_key`, `mem_io_of_readCol`). -/
theorem rsIo_of_readCol {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    (hr : RsReadCol zs dg N k Rw sh w) : RsIo zs dg N k Rw sh w := by
  intro _ _ s un g hW
  obtain ⟨gs, j, p, hj, hpg, _, S₁, ns, nv, _, _, _, _, hc, hO, hS, hg, hpart, he, -, hb, rfl⟩ := hW
  obtain ⟨hq, hcol, -⟩ := openCopy_key hO hS hg hpg hpart he hb
  exact mem_io_of_readCol un hj hcol (hr gs s un j p hO S₁ ns nv hpart _ hq) hc

/-- **R's ring switches are read at their rounds** (`RsRead`) from one session per part, its one unit of one copy,
V*'s sessions, the reads file at the ring switches' ports and the unit reading its row ports' columns. -/
theorem rsRead_of_facts {zs : Reg → L.Ω → List (Cm × (ℐ).Coin) → Reg₂ → (j : Fin S) → (Vs j).Outer Reg'}
    {dg : ByteArray → ByteArray} {N k Rw : ℕ} {sh : Flock.Shape}
    {w : Reg → L.Ω → List (Cm × (ℐ).Coin) → ((j : Fin S) → (Vs j).Bits) → Array F128}
    {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {c₀ : ℕ}
    (hs : OpenSess zs dg N k Rw sh w) (h1 : OpenOne Vs) (hall : OpenAll zs dg N k Rw sh w)
    (hsel : RsSel zs dg N k Rw sh w own sel c₀) (hr : RsReadCol zs dg N k Rw sh w) :
    RsRead zs dg N k Rw sh w own sel c₀ :=
  rsRead_of_layer (oneCopy_of hs h1) hall (rsSk_of_sel hsel) (rsIo_of_readCol hr)

end

end FlockC3.Compose
