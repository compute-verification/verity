module

public import Proofs.Flock.SpotFlock.Placement
public import Proofs.Flock.Soundness.Discharge.Hidden.Commit
public import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Block

@[expose] public section

/-!
# A session layout at a rank, and `Embeds` there

`layoutAt rank` puts unit `u` of a session statement `xS` in the statement's first table, at template instance
`rank % g` of block `(rank / g) % 2^nbl`, where `rank` is the unit's rank in its session (`rank x m₁ xS u`) and `g` the
table's template instances per block. Each wire the unit computes or reads goes to the column at which the accepted
statement places one of the unit's columns on that wire (`colAt`, as `Layout.col` places it; `wireIn`).

`embeds_layoutAt`: at every `rank` the layout holds each checked unit's gates (`Embeds`). The layout reads only the
session statement's first table and the unit's template, never the run's values, and every accepted statement of that
table places the unit's rows at every template instance of every block. Which `rank` puts the decoded values at the
unit's committed ones is `FrameIs`'s.
-/

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy FlockSoundness FlockSoundness.Discharge Exec Flock Flock.Readable

/-- **A session statement's first table**, as the verifier parses it. -/
def firstStmt (xS : Msg) : Option _root_.Flock.Stmt := (parseBytes (FlockDeploy.readOf true) xS).bind (·.sts[0]?)

/-- **Column `x` of template instance `gi`'s stacked rows in statement `st`**, as the accepted statement's layout places
it (`Layout.col`): a copy's column at its unit slot plus its position in the unit (`stackCopy`, `stackLocal`, `upos`),
and the shared constant at the pin. -/
noncomputable def colAt (st : _root_.Flock.Stmt) (hb : Built (lunit st.c)) (gi x : ℕ) : ℕ :=
  if x < (unitRows st hb).stackW st.c.upv - 1 then
    st.c.unitRange.slot (gi * st.c.upv + stackCopy (unitRows st hb) st.c.upv x) +
      upos st.c (stackLocal (unitRows st hb) st.c.upv x)
  else st.pin

/-- **The accepted statement's column `x` at template instance `gi` is `colAt`'s.** -/
theorem col_val {T : Model.Statement} (a : HiddenExec.AcceptedH T) {gi : ℕ} (hg : gi < a.st.c.g) (x : Fin a.rows.w) :
    (a.col hg x).val = colAt a.st a.L.unit_built gi x.val := by
  show (stackPlaceAt (a.L.model a.regs) (unitRows a.st a.L.unit_built) a.st.c.upv
      (fun u => a.st.c.unitRange.slot (gi * a.st.c.upv + u)) (fun c => upos a.st.c c.val) (a.L.hbM a.regs hg) x).val = _
  unfold stackPlaceAt colAt
  by_cases hx : x.val < (unitRows a.st a.L.unit_built).stackW a.st.c.upv - 1
  · rw [dif_pos hx, if_pos hx]
    rfl
  · rw [dif_neg hx, if_neg hx]
    rfl

variable (h : Held)

open Classical in
/-- **Where member `t`'s unit `k` sends position `G` of the members' program**, at template instance `gi` of statement
`st`: `colAt` at one of the unit's columns on the wire at `G` (`Bits.base`, then Bridge's `posOf`), chosen once; `0`
where no wire of the unit is at `G`. -/
noncomputable def wireIn (st : _root_.Flock.Stmt) (gi t k G : ℕ) : ℕ :=
  if ht : t < h.ms.length then
    if hU : Built h.ms[t].1.unit then
      if hk : k < h.ms[t].2.size then
        if hy : ∃ y, Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y = G then
          if hc : ∃ cc, (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
              (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨k, hk⟩).wire cc = hy.choose then
            if hb : Built (lunit st.c) then colAt st hb gi hc.choose.val else 0
          else 0
        else 0
      else 0
    else 0

/-- **The session layout at rank `rank`** (the module's docstring): unit `u` of session statement `xS` at template
instance `rank % g` of block `(rank / g) % 2^nbl` of the statement's first table, and each of its wires at `wireIn`'s
column there. -/
noncomputable def layoutAt (rank : Msg → Msg → Msg → ℕ → ℕ) : SpotFlock.Layout where
  blockOf x m₁ xS u := (firstStmt xS).map fun st => rank x m₁ xS u / st.c.g % 2 ^ (st.m - st.c.kLog)
  wire x m₁ xS u G := ((firstStmt xS).map fun st => wireIn h st (rank x m₁ xS u % st.c.g) (templateOf h.ms u)
    (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) G).getD 0

/-- **Each wire a unit computes or reads is at one of its columns on that wire** (`wireIn`): at the wire's position in
the members' program, `wireIn` gives `colAt` at a column of member `t`'s unit `k` on the wire (`exists_wire`), since
the position fixes the wire (`posOf_injective`). -/
theorem wireIn_eq (st : _root_.Flock.Stmt) (hb : Built (lunit st.c)) (gi t : ℕ) (ht : t < h.ms.length)
    (hU : Built h.ms[t].1.unit) (k : Fin h.ms[t].2.size) (y : Fin (keyC h t ht hU).N)
    (hy : (keyP h t ht hU).unit y = some k ∨
      ∃ g', (keyP h t ht hU).unit g' = some k ∧ y ∈ ((keyC h t ht hU).op g').args) :
    ∃ cc : Fin ((Placed.keyProg h.ms[t].1 hU h.ms[t].2.size).rows k).w,
      (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
        (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) k).wire cc = y ∧
      wireIn h st gi t k (Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y) =
        colAt st hb gi cc.val := by
  classical
  have hex : ∃ y', Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y' =
      Spot.Bits.base h.ms t + _root_.Flock.Bridge.posOf (keyC h t ht hU) y := ⟨y, rfl⟩
  have hyc : hex.choose = y := _root_.Flock.Bridge.posOf_injective (Nat.add_left_cancel hex.choose_spec)
  have hc : ∃ cc, (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨k.val, k.isLt⟩).wire cc = hex.choose := by
    rw [hyc]
    exact exists_wire (Const.Prog.instC (Placed.keyProg h.ms[t].1 hU h.ms[t].2.size)
      (Discharge.Layout.outsOf h.ms[t].1 hU h.ms[t].2.size) ⟨k.val, k.isLt⟩) hy
  refine ⟨hc.choose, hc.choose_spec.trans hyc, ?_⟩
  unfold wireIn
  rw [dif_pos ht, dif_pos hU, dif_pos k.isLt, dif_pos hex, dif_pos hc, dif_pos hb]

/-- **`layoutAt` places every unit of a session statement that has a table** (`PlacesAt`), in its first table: at the
accepted statement's template instance `rank % g`, in range since `HmRow.check` gives `0 < g` (`check_g_pos`), of block
`(rank / g) % 2^nbl`, and each wire the unit computes or reads at one of its columns on that wire (`wireIn_eq`,
`col_val`). -/
theorem placesAt_layoutAt (rank : Msg → Msg → Msg → ℕ → ℕ) {x m₁ xS : Msg} {u : ℕ}
    (hpos : ∀ p : Parsed, parseBytes (FlockDeploy.readOf true) xS = some p → 0 < p.sts.size) :
    PlacesAt h (layoutAt h rank) x m₁ xS u := by
  intro p hp t ht hU k hu
  have h0 := hpos p hp
  refine ⟨0, h0, fun T a hparse hst => ?_⟩
  have hfs : firstStmt xS = some a.st := by
    rw [hst]
    show (parseBytes (FlockDeploy.readOf true) xS).bind (·.sts[0]?) = _
    rw [hp]
    exact getElem?_pos p.sts 0 h0
  obtain ⟨hty, ⟨ls, hc⟩, -⟩ := Hidden.Committed.check a.ht a.committed
  have hg0 : 0 < a.st.c.g := Hidden.check_g_pos hty hc
  have hnb : 2 ^ (a.st.m - a.st.c.kLog) = 2 ^ T.nbl := a.nbl_eq
  have htu : templateOf h.ms u = t := by
    rw [hu]
    exact templateOf_unitBase ht k.isLt
  have hku : u - Properties.Spot.unitBase h.ms t = k := by omega
  have hgi : rank x m₁ xS u % a.st.c.g < a.st.c.g := Nat.mod_lt _ hg0
  refine ⟨_, hgi, ⟨rank x m₁ xS u / a.st.c.g % 2 ^ T.nbl, Nat.mod_lt _ (Nat.two_pow_pos _)⟩, ?_, fun y hy => ?_⟩
  · show (firstStmt xS).map (fun st => rank x m₁ xS u / st.c.g % 2 ^ (st.m - st.c.kLog)) = _
    rw [hfs]
    exact congrArg (fun n => some (rank x m₁ xS u / a.st.c.g % n)) hnb
  · obtain ⟨cc, hw, he⟩ := wireIn_eq h a.st a.L.unit_built (rank x m₁ xS u % a.st.c.g) t ht hU k y hy
    refine ⟨cc, hw, ?_⟩
    have hL : ∀ G, (layoutAt h rank).wire x m₁ xS u G = wireIn h a.st (rank x m₁ xS u % a.st.c.g) t k G := by
      intro G
      show ((firstStmt xS).map fun st => wireIn h st (rank x m₁ xS u % st.c.g) (templateOf h.ms u)
        (u - Properties.Spot.unitBase h.ms (templateOf h.ms u)) G).getD 0 = _
      rw [hfs, htu, hku]
      rfl
    rw [hL, he, Nat.zero_mul, Nat.zero_add]
    exact (col_val a hgi (Fin.cast (congrArg FlockSoundness.Rows.w (rows_keyProg hU a hparse k)) cc)).symm

/-- **A session statement that passes round 2 has a table** (`zkOk_sound`). -/
theorem sts_pos_of_stmtOkAt {x m₁ xS : Msg} {c₀ c₁ : FlockDeploy.Coin} {t : ℕ} {p : Parsed}
    (hok : stmtOkAt h x m₁ xS c₀ c₁ t = true) (hp : parseBytes (FlockDeploy.readOf true) xS = some p) :
    0 < p.sts.size := by
  have hs := sessionOk_of_stmtOkAt h hok
  simp only [sessionOk, Bool.and_eq_true] at hs
  obtain ⟨ins, spec, sts, hset, hpos, -⟩ := zkOk_sound hs.1.1.1
  obtain ⟨rfl, -⟩ := parsed_setupOf hset hp
  exact hpos

/-- **`Embeds` at `layoutAt rank`, for every `rank`**: each checked unit's session statement passes round 2, so it has a
table (`sts_pos_of_stmtOkAt`), and `layoutAt` places the unit there (`placesAt_layoutAt`, `embeds_of_placesAt`). -/
theorem embeds_layoutAt (rank : Msg → Msg → Msg → ℕ → ℕ) : Embeds h (layoutAt h rank) :=
  embeds_of_placesAt h fun _ _ _ _ _ _ _ hx hok hu =>
    placesAt_layoutAt h rank fun _ hp => sts_pos_of_stmtOkAt h (stmtOkAt_unit h hx hok hu) hp

end SpotFlock
