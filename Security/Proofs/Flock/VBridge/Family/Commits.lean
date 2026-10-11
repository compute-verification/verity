import Proofs.Flock.VBridge.Family.Rep
import Proofs.Flock.VBridge.Family.Openings
import Proofs.Flock.VBridge.Family.ReadTransfer
import Proofs.Flock.Soundness.Audit.FlockPublic

/-!
# The family's commit strings, digests and messages

V*'s stage commit strings, the digests its public files register at them, and the messages they decode to, defined from
V*'s registered reads, and the family's obligations on them (`HCommits`, `HDecodes`, `HFull`):

* `skOf`: stage `i`'s commit strings in session `j` are the rows `(u, q)` of the ports V*'s reads file registers
  (`registered.json`, `rec_residuals.write`, `rec_residuals.py:441-445`) under a value whose row unit `u` reads holds
  round `i`'s message (`sel`): `rec-c<k>`, the round's elements (`rec_vstage.messages`, `rec_vstage.py:156-168`), or
  `rec-top-L<l>`, a level's tops;
* `cmtOf`: the digest at a row is its leaf `SHA-512(leaf_prefix ‖ b ‖ c)` (`Readers.regLeaf`) of the `b ‖ c` the round's
  commitment gives (`bc`);
* `wmsgOf`, `msgOf`: round `i`'s message is a decoder's (`dec`) at the layer's bits on stage `i`'s commit strings, every
  other wire read as 0, so a layer and a value agreeing there decode alike.

`vb_cs_inl` is the digest a session's value binding registers at a row: the public file's `b ‖ c` there. `HCommits`
follows where the public files hold the stages' `b ‖ c` (`PubCommits`), `HDecodes` by the masking, and `HFull` where
every wire of a stage row is read (`RowsRead`). `RowsRead` fails at V*'s padded message rows; `VFullRead` and
`VDecodesRead` restate `VFull` and `VDecodes` at the wires some unit reads, and hold without it (`vFullRead_of`, at
V*'s full draw, and `vDecodesRead_of`).
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

/-! ## A session's digests and units at a row -/

/-- **The digest a session's value binding registers at a row** is the leaf of the public file's `b ‖ c` of unit `u`'s
row `q` (`keyReaders`' `regBc`, `Readers.regLeaf` at a row), at every registration. -/
theorem vb_cs_inl {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) (R' : Reg')
    (u : Fin nv) (q : Fin c.ports.size) :
    zo.vb.cs R' (Sum.inl (u, q) : PosK c hU nv) =
      ((Sum.inl (u, q) : PosK c hU nv),
        Refine.H512 (Refine.K512.leafPrefix ++ (zo.pub.digest u.val q.val c.ports.size).data.toList)) :=
  rfl

/-- **A wire of unit `u`'s row is a wire of unit `u` alone**: a unit whose committed wires hold it reads it as a column
of its own instance, an input row's or a returned output row's (`rowOfH_wire_unit`). -/
theorem unit_of_io {c : Flock.Circuit} {hU : Built c.unit} {nv : ℕ}
    {g : Fin ((Placed.keyProg c hU nv).circuit (Layout.outsOf c hU nv)).N} {u u' : Fin nv} {q : Fin c.ports.size}
    (hg : posK c hU nv g = .inl (u, q))
    (hio : g ∈ ((Placed.keyProg c hU nv).partition (Layout.outsOf c hU nv)).io u') : u' = u := by
  unfold posK at hg
  rcases h : HiddenExec.rowOfH c hU nv (Layout.outsOf c hU nv) g with _ | ⟨u₀, q₀, k⟩ <;> rw [h] at hg
  · simp at hg
  · simp only [Sum.inl.injEq, Prod.mk.injEq] at hg
    obtain ⟨rfl, -⟩ := hg
    obtain ⟨x, rfl⟩ :=
      (Const.Prog.instC (Placed.keyProg c hU nv) (Layout.outsOf c hU nv) u').exists_wire_of_mem_io hio
    exact (HiddenExec.rowOfH_wire_unit (Layout.outsOf c hU nv) x h).symm

/-! ## The definitions -/

section defs

variable {S : ℕ} (Vs : Fin S → VStmt)

open Classical in
/-- **Stage `i`'s commit strings in session `j`**: the rows `(u, q)` of port `q` where `own j`, the session's reads file,
registers a port of `q`'s name under a value whose row unit `u` reads holds round `i`'s message (`sel i value row`). -/
noncomputable def skOf (own : Fin S → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop) (i : ℕ) (j : Fin S) :
    Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv) :=
  Finset.univ.filter fun
    | .inl (u, q) => ∃ p ∈ own j, p.name = (Vs j).c.ports[q].name ∧
        ∃ x, p.reads[u.val]? = some (some x) ∧ sel i p.value x
    | .inr _ => False

/-- **The digest at each commit string**: the leaf of the `b ‖ c` the commitment gives there (`bc`). -/
noncomputable def cmtOf {Cm : Type} (bc : (j : Fin S) → Cm → PosK (Vs j).c (Vs j).hU (Vs j).nv → List UInt8)
    (j : Fin S) (e : Cm) (pos : PosK (Vs j).c (Vs j).hU (Vs j).nv) : (obK (Vs j).c (Vs j).hU (Vs j).nv).Dg :=
  (pos, Refine.H512 (Refine.K512.leafPrefix ++ bc j e pos))

variable (Sk : ℕ → (j : Fin S) → Finset (PosK (Vs j).c (Vs j).hU (Vs j).nv))

open Classical in
/-- A layer's bits on the read wires of stage `i`'s commit strings, 0 elsewhere. -/
noncomputable def maskOf (i : ℕ) (Xs : (j : Fin S) → (Vs j).Bits) : (j : Fin S) → (Vs j).Bits :=
  fun j g => if posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j ∧ ReadW Vs j g then Xs j g else false

open Classical in
/-- The bits values `v` give each read wire of stage `i`'s commit strings, 0 elsewhere. -/
noncomputable def layerOf (i : ℕ) (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) :
    (j : Fin S) → (Vs j).Bits :=
  fun j g => if posK (Vs j).c (Vs j).hU (Vs j).nv g ∈ Sk i j ∧ ReadW Vs j g then
    bitK (Vs j).c (Vs j).hU (Vs j).nv (v j (posK (Vs j).c (Vs j).hU (Vs j).nv g)) g else false

/-- **The messages a layer holds**: round `i`'s, of `r` rounds, `dec i` at the layer's bits on the read wires of stage
`i`'s commit strings (V*'s nets read only those: `InnerRepCheck` the first `128 n` bits of a row port of `n` elements,
`InnerRepCheck.lean:39-44`). -/
noncomputable def wmsgOf {Msg : Type} (r : ℕ) (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → Msg)
    (Xs : (j : Fin S) → (Vs j).Bits) : List Msg :=
  (List.range r).map fun i => dec i (maskOf Vs Sk i Xs)

/-- **The message stage `i`'s commit strings decode to** at values `v`: `dec i` at their read wires' bits. -/
noncomputable def msgOf {Msg : Type} (dec : ℕ → ((j : Fin S) → (Vs j).Bits) → Msg) (i : ℕ)
    (v : (j : Fin S) → PosK (Vs j).c (Vs j).hU (Vs j).nv → ValK (Vs j).c) : Msg :=
  dec i (layerOf Vs Sk i v)

variable {Vs}

theorem mem_skOf_inl {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {i : ℕ} {j : Fin S}
    {u : Fin (Vs j).nv} {q : Fin (Vs j).c.ports.size} :
    Sum.inl (u, q) ∈ skOf Vs own sel i j ↔
      ∃ p ∈ own j, p.name = (Vs j).c.ports[q].name ∧ ∃ x, p.reads[u.val]? = some (some x) ∧ sel i p.value x := by
  simp only [skOf, Finset.mem_filter, Finset.mem_univ, true_and]

theorem inr_not_mem_skOf {own : Fin S → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop} {i : ℕ}
    {j : Fin S} (g : Fin ((Placed.keyProg (Vs j).c (Vs j).hU (Vs j).nv).circuit
      (Layout.outsOf (Vs j).c (Vs j).hU (Vs j).nv)).N) :
    Sum.inr g ∉ skOf Vs own sel i j := by
  simp only [skOf, Finset.mem_filter, Finset.mem_univ, true_and, not_false_eq_true]

end defs

/-! ## The obligations at the definitions -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **Each session's public file holds the stages' `b ‖ c`**: at every history `t` whose stage-`i` commitment is `e.1`
and every registration, session `j`'s public file (`pub.bin`) holds at each row of `Sk i j` the `b ‖ c` that `e.1` gives
there (`bc`). V* stages the verifier's `pub.bin` with the inner session's `c_k` at its message rows (`rec_vstage.stage`,
`rec_vstage.py:353-366`; `rec_residuals.write`, `rec_residuals.py:441-445,464`). The verifier's `--registered` check
(`Flock.Registered.check`) ties those rows to the entry `rec_residuals.commits_entry` rebuilds from the `c_k`
(`rec_residuals.py:398-406`) only on an accepted run, which is where `VCommits` reads it. -/
def PubCommits (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8) : Prop :=
  ∀ i R ω (t : List (Cm × (ℐ).Coin)) (e : Cm × (ℐ).Coin), t[i]? = some e → ∀ R₂ j,
    (∃ tr, ZkReg.AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) → ∀ (u : Fin (F.Vs j).nv)
    (q : Fin (F.Vs j).c.ports.size), Sum.inl (u, q) ∈ F.Sk i j →
      ((F.zs R ω t R₂ j).pub.digest u.val q.val (F.Vs j).c.ports.size).data.toList = bc j e.1 (.inl (u, q))

/-- **`HCommits` at V*'s rows**: a family whose commit strings are `skOf`'s and whose digests are `cmtOf bc`'s
registers the stages' commitments where its public files hold their `b ‖ c` (`PubCommits`). -/
theorem hCommits_of (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)
    (bc : (j : Fin F.stmts.size) → Cm → PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv → List UInt8)
    (hSk : F.Sk = skOf F.Vs own sel) (hcmt : F.cmt = cmtOf F.Vs bc) (hpub : F.PubCommits bc) : F.HCommits := by
  intro i R ω t e he R₂ j hacc pos hpos R'
  rcases pos with ⟨u, q⟩ | g
  · rw [hcmt]
    exact (vb_cs_inl (F.zs R ω t R₂ j) R' u q).trans
      (congrArg (fun l => ((Sum.inl (u, q) : PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv),
        Refine.H512 (Refine.K512.leafPrefix ++ l))) (hpub i R ω t e he R₂ j hacc u q hpos))
  · rw [hSk] at hpos
    exact absurd hpos (inr_not_mem_skOf g)

/-- **`VDecodes` at the read wires**, the restatement of `VDecodes` that V*'s padded rows need (`RowsRead`): `VDecodesR`
at the family, layers whose bits on the read wires of each `Sk i j` are those of the values `v j` holding a round-`i`
message the inner verifier reads as `msg i v`. -/
def VDecodesRead : Prop := VDecodesR F.wmsg (fun _ _ a b => a = b) F.Sk F.msg

/-- **`VDecodesRead` at a decoder**: a family whose messages are `wmsgOf`'s and `msgOf`'s at one decoder `dec` decodes
each stage's message from the read wires of its commit strings, whatever its commit strings: both read only their
bits. -/
theorem vDecodesRead_of (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec) : F.VDecodesRead := by
  intro i hi _ Xt v hX
  rw [hw, hm]
  simp only [wmsgOf, List.getElem?_map, List.getElem?_range hi, Option.map_some]
  refine Option.Rel.some ?_
  show dec i (maskOf F.Vs F.Sk i Xt) = dec i (layerOf F.Vs F.Sk i v)
  refine congrArg (dec i) (funext fun j => funext fun g => ?_)
  simp only [maskOf, layerOf]
  split_ifs with h
  · exact hX j g h.1 h.2
  · rfl

/-- **`HDecodes` at a decoder**: `vDecodesRead_of`, whose layers agree with the values on every wire of the commit
strings. -/
theorem hDecodes_of (dec : ℕ → ((j : Fin F.stmts.size) → (F.Vs j).Bits) → (ℐ).Msg)
    (hw : F.wmsg = wmsgOf F.Vs F.Sk (ℐ).r dec) (hm : F.msg = msgOf F.Vs F.Sk dec) : F.HDecodes :=
  F.vDecodesRead_of dec hw hm

/-- **The stage selector names round commitments**: every value whose row holds a round's message is a round's
commitment (`Flock.Registered.roundCommitment`, `Registered.lean:182-193`: V*'s `rec-c<k>` and `rec-top-L<l>`, and R's
and the sparse phase's), the values U4 checks. -/
def SelRounds (sel : ℕ → String → ℕ → Prop) : Prop :=
  ∀ i v x, sel i v x → Flock.Registered.roundCommitment v = true

/-- **Every wire of a stage's commit strings is read**: a committed wire of some unit. A circuit fact no check makes.
It fails at V*'s message rows: `InnerRepCheck`'s port of `n` elements is `64 ⌈8 n / 64⌉` words
(`portWordsOf`, `InnerRepCheck.lean:24`), its net reads only the first `128 n` bits (`InnerRepCheck.lean:39-44`), and the
rest are row wires (`portRowW` is the port's `16 · words` bits) no unit reads, in no unit's `io`. -/
def RowsRead : Prop := ∀ i < (ℐ).r, ∀ j g, posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g ∈ F.Sk i j → ReadW F.Vs j g

/-- **A read wire of a stage's row is read by a unit that reads a round's commitment**: at `skOf`'s commit strings, the
unit reading it is the row's own (`unit_of_io`), which reads the row's port of `own` by `skOf`, a round's commitment by
`SelRounds`. -/
theorem reads_of_skOf {own : Fin F.stmts.size → Array Flock.Registered.Port} {sel : ℕ → String → ℕ → Prop}
    (hSk : F.Sk = skOf F.Vs own sel) (hsel : SelRounds sel) {i : ℕ} {j : Fin F.stmts.size}
    {g : Fin ((Placed.keyProg (F.Vs j).c (F.Vs j).hU (F.Vs j).nv).circuit
      (Layout.outsOf (F.Vs j).c (F.Vs j).hU (F.Vs j).nv)).N} {u : Fin (F.Vs j).nv}
    (hg : posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g ∈ F.Sk i j)
    (hio : g ∈ ((Placed.keyProg (F.Vs j).c (F.Vs j).hU (F.Vs j).nv).partition
      (Layout.outsOf (F.Vs j).c (F.Vs j).hU (F.Vs j).nv)).io u) :
    ∃ p ∈ own j, Flock.Registered.roundCommitment p.value = true ∧ ∃ x, p.reads[u.val]? = some (some x) := by
  rw [hSk] at hg
  rcases hpos : posK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv g with ⟨u₀, q⟩ | g'
  · rw [hpos, mem_skOf_inl] at hg
    obtain ⟨p, hp, -, x, hx, hs⟩ := hg
    obtain rfl := unit_of_io hpos hio
    exact ⟨p, hp, hsel i p.value x hs, x, hx⟩
  · rw [hpos] at hg
    exact absurd hg (inr_not_mem_skOf g')

/-- **`HFull`** where its stage wires are read. -/
theorem hFull_of (hrd : F.RowsRead) : F.HFull := hrd

/-- **`VFull` at the read wires**, the restatement of `VFull` that V*'s padded rows need (`RowsRead`): `VFullR` at the
family, each wire of `Sk i j` (`i < r`) some unit reads being a wire of a unit session `j`'s law draws. -/
def VFullRead : Prop := VFullR (ℐ).r F.Sk

/-- **`VFullRead` without `RowsRead`**: V*'s laws are the full draw, which draws every unit (`vFullR_of_read`). -/
theorem vFullRead_of : F.VFullRead := vFullR_of_read _ _

end VStarFamily

/-- **`v-bridge` at the read wires, at V*'s statement family**: `VBridgeRead` at the family, from the obligations
`vbridge_of_family` takes for its first conjunct (the parts' and openings' outputs read in the layers: `HResL`, through
`algebraHolds_of_decodeL`, and `HUL`, through `unitSums_of_opensAtL`), `HCommits`, and `VDecodesRead` and `VFullRead`,
which `vDecodesRead_of` and `vFullRead_of` give. -/
theorem vbridgeRead_of_family (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray)
    (N k Rw : ℕ) {GS : Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} (hst : F.HStruct GS gs)
    (hsh : F.HShape dg N k Rw GS) (hres : F.HResL dg N k Rw gs) (hu : F.HUL dg N k Rw) (hrd : F.HReads dg N k Rw)
    (hsch : F.HSch) (hfiles : F.HFiles dg N k Rw) (hruns : F.HRuns dg N k Rw) (hcm : F.HCommits)
    (hdec : F.VDecodesRead) (hfull : F.VFullRead) :
    VBridgeRead (I := ℐ) F.zs dg N k Rw F.x F.wmsg F.Sk F.cmt F.msg :=
  ⟨fun R ω t R₂ hc τ o h₀ =>
    have h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o := ⟨h₀, hc⟩
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).accepts regs (F.x R ω) T CT sch (hsch R ω t _) _
      (hfiles R ω t R₂ τ o h) (hruns R ω t R₂ τ o h fun r => by
        obtain ⟨⟨v, info, kappa, hv, hvs⟩, hw, hc⟩ := hsh R ω t R₂ τ o h r
        exact ⟨FlockVBridge.OfVStar.algebraHolds_of_decodeL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ)
            (fun s => (h.1 s).2) hst.1 (Flock.RecAlgebra.structureOf_bitsOk hst.1) hv hw hc hvs hst.2 F.sess
            (fun p => F.un p r) (fun p => cp p r) (hres R ω t R₂ τ o h r),
          FlockVBridge.AccChain.accHolds_of_units
            (unitSums_of_opensAtL (F.zs R ω t R₂) (layers F.zs dg N k Rw R ω t R₂ τ) F.sh _ _ F.sessO
              (fun ℓ j => F.unO ℓ r j) (fun ℓ j => cpO ℓ r j) (hu R ω t R₂ τ o h r))
            (F.coefsAre R ω t _ r) (F.consAre R ω t _ r) (hrd R ω t R₂ τ o h r)⟩),
    hcm, hdec, hfull⟩

end family

end FlockVBridge.Family
