import Proofs.Flock.Soundness.Discharge.ZkPubIn.Defs
import Proofs.Flock.Soundness.Discharge.ZkHidden.Defs

/-!
# The `--zk` tables of a hidden-output session with the verifier's public inputs

HiddenExec's `DrawSetupH`, `TableH` and ZkHidden's `DrawSetupZKH`, `tableAtZKH`, `planAtZKH`, `tabAtZKH`,
`DrawOkZKH` (`HiddenExec/Tables.lean`, `ZkHidden/Defs.lean`) with `Stmt.setupHidden … (some own)` for
`Stmt.setupHidden …` at `publicInputs := none`, field for field:

* `DrawSetupHP I own mPts S d`, `TableHP I own mPts S`, `DrawSetupHP.table`, `.refused`;
* `DrawSetupZKHP I own mPts S d`, `.table`, `.refused`, `tableAtZKHP`, `tablesAtZKHP`, `planAtZKHP`, `tabAtZKHP`;
* `DrawOkZKHP`: the coin server's draw at `S` proves `S`, and setupHidden with `own` sets up at it with `mPts` link
  points.
-/

namespace FlockSoundness.Discharge.ZkPubIn

open Flock FlockLevel3 Game Model Refine Exec DecodeZL ZkSession ZkExec ZkLink HiddenExec ZkHidden
open FlockSoundness.Discharge.Integrate (DrawSetup)
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

variable {n : ℕ} (I : Inputs) (own : ByteArray) (mPts : ℕ)

/-- **A draw flock-verify sets up under hidden outputs with the verifier's public inputs**: `DrawSetupH` with
`Stmt.setupHidden … (some own)`. -/
structure DrawSetupHP (S : Finset (Fin n)) (d : Lean.Json) where
  st : Stmt
  h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program (some own) =
    .ok st
  D : Draw.UnitDraw
  hD : Draw.UnitDraw.ofJson d = .ok D
  hS : ∀ u ∈ S, u.val ∈ D.proved
  hg : 0 < st.c.g
  hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl
  m : ℕ
  k0 : ℕ
  l₀ : Accounting.Level
  l₁ : Accounting.Level
  rest : List Accounting.Level
  ext : ℕ
  hsch : Accounting.fast100 st.m = some ⟨m, k0, l₀ :: l₁ :: rest, ext⟩
  hm : 13 ≤ st.m
  hmPts : st.mPts = mPts
  wf : StmtWF st ∧ RegionsWF st

/-- **A hidden-output draw's table with its sites, with the verifier's public inputs**: `TableH` with
`Stmt.setupHidden … (some own)`. -/
structure TableHP (S : Finset (Fin n)) where
  T : TabSpec mPts
  st : Stmt
  d : Lean.Json
  h : Stmt.setupHidden I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program (some own) =
    .ok st
  regs : List (Model.Region st.c.kLog (st.m - st.c.kLog))
  hwf : StmtWF st
  hT : T.S = stmtOf st hwf regs
  vu : ∀ u ∈ S, ℕ
  hvu : ∀ u (h : u ∈ S), vu u h < st.c.g
  o : ∀ u ∈ S, Fin (2 ^ T.S.nbl)

variable {I own mPts}

namespace DrawSetupHP

variable {S : Finset (Fin n)} {d : Lean.Json} (x : DrawSetupHP I own mPts S d)

/-- The table at statement regions `regs` (`DrawSetupH.specAt`). -/
noncomputable def specAt (regs : List (Model.Region x.st.c.kLog (x.st.m - x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabSpec mPts where
  S := stmtOf x.st x.wf.1 regs
  m := x.m
  k0 := x.k0
  l₀ := x.l₀
  l₁ := x.l₁
  rest := x.rest
  ext := x.ext
  hsch := x.hsch
  hm := x.hm
  ptLocal := PT_LOCAL
  hlay := ⟨hregs, by
    obtain ⟨-, hr⟩ := x.wf
    show PT_LOCAL + (x.st.m - x.st.c.kLog) ≤ mPts
    have h1 := hr.mPts
    have h2 := hr.hm
    have h3 := x.hmPts
    omega⟩

/-- A drawn unit's rank among the draw's proved units. -/
def rank (u : Fin n) : ℕ := x.D.proved.toList.idxOf u.val

theorem rank_lt {u : Fin n} (h : u ∈ S) : x.rank u < x.D.proved.size := by
  have := List.idxOf_lt_length_of_mem (Array.mem_toList_iff.2 (x.hS u h))
  simpa [rank] using this

theorem rank_div_lt {u : Fin n} (h : u ∈ S) : x.rank u / x.st.c.g < 2 ^ (x.st.m - x.st.c.kLog) := by
  obtain ⟨-, hr⟩ := x.wf
  rw [Nat.div_lt_iff_lt_mul x.hg, hr.hm, Nat.add_sub_cancel_left, Nat.mul_comm]
  exact lt_of_lt_of_le (x.rank_lt h) x.hblk

/-- **The draw's table** (`DrawSetupH.table`): `stmtOf` of the statement at its regions, each drawn unit at its rank
`r`: block `r / c.g`, VU `r % c.g`. -/
noncomputable def table : TableHP I own mPts S where
  T := x.specAt (regionsOf x.st) (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1
  st := x.st
  d := d
  h := x.h
  regs := regionsOf x.st
  hwf := x.wf.1
  hT := rfl
  vu u _ := x.rank u % x.st.c.g
  hvu _ _ := Nat.mod_lt _ x.hg
  o u h := ⟨x.rank u / x.st.c.g, x.rank_div_lt h⟩

theorem refused_wf : ∀ R ∈ regionsOf x.st ++ DrawSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog),
    R.free.Nodup ∧ R.free.length ≤ PT_LOCAL := by
  intro R hR
  rcases List.mem_append.1 hR with hR | hR
  · exact (stmtOf_linkLayout x.st x.wf.1 x.wf.2).1 R hR
  · exact DrawSetup.contra_wf R hR

/-- **The table of a refused draw `S'`** (`DrawSetupH.refused`): `x`'s statement with `contra`'s regions, which no
witness satisfies, every unit at block 0, VU 0. -/
noncomputable def refused (S' : Finset (Fin n)) : TableHP I own mPts S' where
  T := x.specAt (regionsOf x.st ++ DrawSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog)) x.refused_wf
  st := x.st
  d := d
  h := x.h
  regs := regionsOf x.st ++ DrawSetup.contra x.st.c.kLog (x.st.m - x.st.c.kLog)
  hwf := x.wf.1
  hT := rfl
  vu _ _ := 0
  hvu _ _ := x.hg
  o _ _ := ⟨0, Nat.two_pow_pos _⟩

theorem refused_unsat (S' : Finset (Fin n)) (z : Witness (x.refused S').T.S.m) :
    ¬ (x.refused S').T.S.Satisfies z :=
  DrawSetup.contra_unsat (S := (x.refused S').T.S) (fun _ hR => List.mem_append_right _ hR) z

end DrawSetupHP

variable (I own mPts)

/-- **A draw `verify --zk --public-inputs F` sets up under hidden outputs**: `DrawSetupZKH` with a `DrawSetupHP` at
`zkInputs I` and `own`. -/
structure DrawSetupZKHP (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetupHP (zkInputs I) own mPts S d
  sch : Flock.Schedule
  tPad : ℕ
  hz : Zk.schedule x.st.m = .ok (sch, tPad)
  hshape : 2 ≤ (schedOf sch).levels.length ∧ ((schedOf sch).levels.headD default).lanes = (schedOf sch).k0 ∧
    (schedOf sch).m - 7 = (schedOf sch).k0 + ((schedOf sch).levels.headD default).logCols ∧
    levelsChained (schedOf sch).levels (schedOf sch).ext = true
  hmsch : (schedOf sch).m = x.st.m
  hpad : Accounting.PadOK (schedOf sch).level₀ tPad
  hN : innerN (layOf x.st sch) < 2 ^ 64
  hL : (layOf x.st sch).len + Zk.K_PAD ≤ innerN (layOf x.st sch)
  hpos : ∀ D : Finset (Fin (innerN (layOf x.st sch))),
    prCoin (fun w => innerPos _ (lt_of_lt_of_le (by simp [Zk.K_PAD]) hL) w ∈ D) ≤
      (D.card : ℝ≥0∞) / innerN (layOf x.st sch)

variable {I own mPts}

namespace DrawSetupZKHP

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZKHP I own mPts S d)

theorem innerN_pos : 0 < innerN (layOf y.x.st y.sch) := lt_of_lt_of_le (by simp [Zk.K_PAD]) y.hL

/-- `codeAt`'s points, distinct below `2^64` (`DrawSetupZKH.innerDom`). -/
def innerDom : Fin (innerN (layOf y.x.st y.sch)) ↪ 𝔽 :=
  ⟨fun j => innerPt j.val, fun a b h => by
    have hlt : ∀ j : Fin (innerN (layOf y.x.st y.sch)), j.val + 1 < 2 ^ 64 := fun j => by
      have := y.hN
      have := j.isLt
      omega
    have h' : ((a.val + 1).toUInt64 : UInt64) = (b.val + 1).toUInt64 := congrArg F128.lo h
    have := congrArg UInt64.toNat h'
    simp only [Nat.toUInt64_eq, UInt64.toNat_ofNat', Nat.mod_eq_of_lt (hlt a), Nat.mod_eq_of_lt (hlt b)] at this
    exact Fin.ext (by omega)⟩

/-- **The `--zk` table of a draw with the verifier's public inputs**, at statement regions `regs`
(`DrawSetupZKH.tabAt`). -/
noncomputable def tabAt (regs : List (Model.Region y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)))
    (hregs : ∀ R ∈ regs, R.free.Nodup ∧ R.free.length ≤ PT_LOCAL) : TabZK 𝔽 mPts where
  S := stmtOf y.x.st y.x.wf.1 regs
  sch := schedOf y.sch
  hshape := y.hshape
  hmsch := y.hmsch
  hm := y.x.hm
  t := y.tPad
  hpad := y.hpad
  c₀ := fOf Zk.PAD_C
  e := 2
  ptLocal := PT_LOCAL
  hlay := ⟨hregs, by
    have h := (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).2
    rw [y.x.hmPts] at h
    exact h⟩
  np := (layOf y.x.st y.sch).len
  pads := padsOfLay (layOf y.x.st y.sch)
  padsA := padsAOfLay (layOf y.x.st y.sch)
  tr := tripleOfLay (layOf y.x.st y.sch)
  N := innerN (layOf y.x.st y.sch)
  dom := y.innerDom
  nP := Zk.K_PAD
  q := Zk.Q
  hnP := ⟨by simp [Zk.K_PAD]⟩
  pos := innerPos _ y.innerN_pos
  hpos := y.hpos
  δ := Code.relativeUniqueDecodingRadius
    (ReedSolomon.code y.innerDom ((layOf y.x.st y.sch).len + Zk.K_PAD) : Set (Fin (innerN (layOf y.x.st y.sch)) → 𝔽))
  hL := y.hL
  hδ := le_rfl

/-- The draw's `--zk` table: the statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused draw's `--zk` table: the statement with `contra`'s regions, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (regionsOf y.x.st ++ Integrate.DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) y.x.refused_wf

end DrawSetupZKHP

variable {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZKHP I own mPts S₀ (dj S₀))

open Classical in
/-- **The `--zk` table at draw `S` with the verifier's public inputs** (`tableAtZKH`): the draw's table when
`verify --zk --public-inputs F` sets up the draw's JSON `dj S`, and otherwise `y₀`'s refused table. -/
noncomputable def tableAtZKHP (S : Finset (Fin n)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZKHP I own mPts S (dj S)) then (Classical.choice h).table else y₀.refused

/-- `tableAtZKHP` at every registration. -/
noncomputable abbrev tablesAtZKHP : Finset (Fin n) → Reg → TabZK 𝔽 mPts := fun S _ => tableAtZKHP dj y₀ S

/-- The one-table plan: at draw `S`, the table `tableAtZKHP … S`, for every registration. -/
noncomputable abbrev planAtZKHP : Finset (Fin n) → Reg → List (TabZK 𝔽 mPts) := fun S _ => [tableAtZKHP dj y₀ S]

/-- Every drawn unit is in the plan's one table. -/
abbrev tabAtZKHP : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (planAtZKHP (Reg := Reg) dj y₀ S R).length :=
  fun _ _ _ => ⟨0, Nat.zero_lt_one⟩

/-- **The coin server's draw at `S`, with the verifier's public inputs** (`DrawOkZKH` with `own`): it proves `S`, and
setupHidden with `own` sets up at it under the `--zk` tags and coin tree with `mPts` link points. -/
def DrawOkZKHP (I : Inputs) (own : ByteArray) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) :
    Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupHidden (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition
      I.program (some own) = .ok st → st.mPts = mPts

end FlockSoundness.Discharge.ZkPubIn
