import Proofs.Flock.VBridge.Family.Rounds
import Proofs.Flock.Soundness.Discharge.ZkReg.Checked
import Proofs.Flock.Verifier.Registered

/-!
# The family's public files at accepting sessions: `PubCommits` up to a SHA-512 collision

The prover writes each session's `pub.bin` (`rec_residuals.write`, `rec_residuals.py:468`); the verifier builds the
reads file's `rec-c<k>` entries from the `c_k` alone (`rec_vstage.messages`, `rec_vstage.py:156-168`;
`rec_residuals.commits_entry`, `rec_residuals.py:402-410`): two leaves, the frame-v3 tree over the `c_k`'s `b ‖ c`.
At an accepting session `Registered.check` opens each drawn instance's table row against that root, so the row's leaf is
the `c_k`'s, or the verifier's opening and the honest one collide:

* `commitsRoot`, `commitsSib`: the verifier's entry's root over two `b ‖ c`, and the honest opening's sibling;
  `opens_commitsRoot`: the honest opening is accepted; `treeLeaf_eq_or_collides`: an accepted opening at the same
  position has the `c_k`'s leaf, or `openPair` collides (`Flock.Registered.opensLeaf_binding`);
* `opens_of_acceptsZKR`: at a transcript the session accepts, a drawn unit's row of a port of `own` opens the port's root
  at the position `own` gives (`checked_of_acceptsHJ`, without a registrant), at the table the unit is drawn into;
* `HOwnRoots`, `RowsDrawn`, `PortNamesUnique`: what the family's sessions must be for that: the verifier's reads files'
  roots are `commitsRoot` over the history's `c_k`, an accepted transcript draws every unit that reads a round's
  commitment and labels it itself, and a circuit's port names are distinct;
* `pubCommits_or_collides`: at every history and accepting session, each row of `Sk i j` registers `cmtOf (bc cOf)`'s
  digest of `t`'s stage-`i` commitment (`VCommits` at the row), or `openPair` collides;
* `HPubRows`, `pubCommits_of_pubRows`: where the verifier compares the public file's `rec-c<k>` rows with the `c_k`
  itself, `PubCommits` holds exactly, at every history.
-/

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Refine FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages
open FlockSoundness.Discharge.ZkReg FlockSoundness.Discharge.ZkExec FlockSoundness.Discharge.ZkLink
open FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.Integrate FlockSoundness.Discharge.DecodeZL
open FlockSoundness.Discharge.Composed (drawJ partOf)

/-! ## The verifier's entry of two `c_k` -/

/-- **The verifier's root over two rows' `b ‖ c`** (`rec_residuals.commits_entry`): the frame-v3 node over the two
`hm96-sha512/row/v1` leaves of their tree leaves, the root `Registered.climb` reaches over two leaves. -/
def commitsRoot (dom c0 c1 : ByteArray) : ByteArray :=
  Registered.node dom 0 0 (Registered.rowLeaf dom 0 (Hm96.Default512.treeLeaf c0))
    (Registered.rowLeaf dom 1 (Hm96.Default512.treeLeaf c1))

/-- Row `x`'s `b ‖ c` of two. -/
def commitAt (c0 c1 : ByteArray) (x : ℕ) : ByteArray := if x = 0 then c0 else c1

/-- The honest opening of row `x` under `commitsRoot`: the other row's leaf. -/
def commitsSib (dom c0 c1 : ByteArray) (x : ℕ) : ByteArray :=
  if x = 0 then Registered.rowLeaf dom 1 (Hm96.Default512.treeLeaf c1)
  else Registered.rowLeaf dom 0 (Hm96.Default512.treeLeaf c0)

theorem treeLeaf_size (b : ByteArray) : (Hm96.Default512.treeLeaf b).size = 64 := Sha512.hash_size _

/-- **The honest opening is accepted**: row `x < 2`'s `b ‖ c` opens `commitsRoot` at `x` through `commitsSib`, at a
port of `hm96-sha512/row/v1` rows. -/
theorem opens_commitsRoot {reg : Registered.Port} {c0 c1 : ByteArray} (hl : reg.leaves = 2)
    (hs : reg.schema = HmRow.ROW_SCHEMA) (hr : reg.root = commitsRoot reg.domain c0 c1) {x : ℕ} (hx : x < 2) :
    Registered.opens reg (commitAt c0 c1 x) x (commitsSib reg.domain c0 c1 x) = true := by
  have hz : (commitsSib reg.domain c0 c1 x).size = 64 := by
    unfold commitsSib
    split <;> exact Sha512.hash_size _
  have hsib : Registered.sibs (commitsSib reg.domain c0 c1 x) 1 = [commitsSib reg.domain c0 c1 x] := by
    have he : (commitsSib reg.domain c0 c1 x).extract 0 64 = commitsSib reg.domain c0 c1 x := by
      conv_lhs => rw [show (64 : ℕ) = (commitsSib reg.domain c0 c1 x).size from hz.symm]
      exact ByteArray.extract_zero_size
    simp [Registered.sibs, he]
  have hlog : log2ceil reg.leaves = 1 := by rw [hl]; rfl
  unfold Registered.opens
  rw [Registered.opensLeaf_row hs]
  unfold Registered.climb
  rw [hlog, hsib, hr, hz, hl]
  simp only [Bool.and_eq_true, decide_eq_true_eq]
  refine ⟨⟨hx, trivial⟩, ?_⟩
  interval_cases x <;> simp [Registered.climbL, commitAt, commitsSib, commitsRoot]

/-- **An accepted opening of row `x` under `commitsRoot` has row `x`'s leaf**, or it and the honest opening collide
(`opensLeaf_binding`, `openPair` the extractor). -/
theorem treeLeaf_eq_or_collides {reg : Registered.Port} {c0 c1 : ByteArray} (hl : reg.leaves = 2)
    (hs : reg.schema = HmRow.ROW_SCHEMA) (hr : reg.root = commitsRoot reg.domain c0 c1) {x : ℕ} (hx : x < 2)
    {bc path : ByteArray}
    (h : Registered.opens reg bc x path = true) :
    Hm96.Default512.treeLeaf bc = Hm96.Default512.treeLeaf (commitAt c0 c1 x) ∨
      Registered.Collides (Registered.openPair reg x (Hm96.Default512.treeLeaf bc) path
        (Hm96.Default512.treeLeaf (commitAt c0 c1 x)) (commitsSib reg.domain c0 c1 x)) := by
  by_cases he : Hm96.Default512.treeLeaf bc = Hm96.Default512.treeLeaf (commitAt c0 c1 x)
  · exact Or.inl he
  · exact Or.inr (Registered.opensLeaf_binding reg x _ path _ _ (treeLeaf_size _) (treeLeaf_size _) he h
      (opens_commitsRoot hl hs hr hx))

/-! ## A session's accepted rows -/

/-- **An accepted transcript opens a drawn unit's registered row** (`checked_of_acceptsHJ`, at a port of `own` rather
than a registrant's read): where the transcript's draw `d` draws unit `u` into its table (`partOf`), that table's drawn
statement (`HmOut.drawn` at `drawJ`) labels it `idx`, and `reg ∈ own` is the circuit's port `q` by name and gives
position `x` for `u`'s label, the public file's `b ‖ c` of `u`'s row `q` opens `reg`'s root at `x` through the header's
path. -/
theorem opens_of_acceptsZKR {mPts nv J : ℕ} {Reg' : Type} {c : Flock.Circuit} {hU : Built c.unit}
    {hscope : Layout.scopeOk c = true} {Lv : Law nv} (zo : ZkOuter mPts nv J Reg' c hU hscope Lv) {tr : Exec.Transcript}
    (hacc : AcceptsZKR zo.I zo.own tr) {reg : Registered.Port} (hmem : reg ∈ zo.own) {u : Fin nv}
    {q : Fin c.ports.size} (hfind : (List.range c.ports.size).find? (fun p => c.ports[p]!.name == reg.name) = some q.val)
    {d : Lean.Json} {D : Draw.UnitDraw} {dp : Public} {idx : Array ℕ} (hd : recordDraw tr.record = some d)
    (hD : Draw.UnitDraw.ofJson (drawJ zo.I d (partOf zo.I d u.val)) = .ok D)
    (hdp : HmOut.drawn c zo.pub (drawJ zo.I d (partOf zo.I d u.val)) = .ok dp)
    (hidx : Registered.unitIndices dp = .ok idx) (hu : u.val ∈ D.proved) {x : ℕ}
    (hread : reg.reads[idx.getD (D.proved.idxOf u.val) 0]? = some (some x)) :
    ∃ path, Registered.opens reg (zo.pub.digest u.val q.val c.ports.size) x path = true := by
  obtain ⟨spec, sts, hs, hreg, -, -⟩ := hacc
  obtain ⟨hsz, hstJ⟩ := ZkHidden.setupOf_tableJH zo.hTags.1 hd hs
  have hjs : (partOf zo.I d u.val).val < sts.size := by rw [hsz]; exact (partOf zo.I d u.val).isLt
  have s := Hidden.setupHidden_spec' (hstJ _ hjs)
  have hcst : sts[(partOf zo.I d u.val).val].c = c := by
    have h1 := s.parseF ((FlockSoundness.Discharge.Exec.tagsAt_isTmpl _ _ _).trans (HmOut.parse_isTmpl zo.hcH))
    rw [ZkHidden.parseH_tagsAt ((zkTags_typed zo.I).trans zo.ht), zo.hcH] at h1
    exact (Except.ok.inj h1).symm
  obtain ⟨pop', hpop', -, hdr⟩ := s.pub
  rw [hcst, ZkHidden.loadPublicH_tagsAt, zo.hpub] at hpop'
  cases hpop'
  have hdr' := hdr _ rfl
  rw [hcst, hdp] at hdr'
  have hpd : sts[(partOf zo.I d u.val).val].pub = dp := (Except.ok.inj hdr').symm
  have hdpR : HmRow.drawn { c with outPorts := #[] } zo.pub (drawJ zo.I d (partOf zo.I d u.val)) = .ok dp := hdp
  have hck := checkRegistered_ok (array_forM_ok hreg _ hjs)
  rw [hcst, hpd] at hck
  obtain ⟨tb, idx', htb, hidx', hports⟩ := check_tables hck
  rw [hidx] at hidx'
  cases hidx'
  obtain ⟨p', positions, paths, hp', hent, hall⟩ := checkPort_at (hports _ hmem)
  rw [hfind] at hp'
  cases hp'
  obtain ⟨D', hD', -, hn, -, -⟩ := Layout.drawn_spec hdpR
  rw [hD] at hD'
  cases hD'
  have hi : D.proved.idxOf u.val < D.proved.size := Array.idxOf_lt_length_of_mem hu
  obtain ⟨-, hreads, hopens⟩ := hall _ (by rw [hn]; exact hi)
  rw [hread] at hreads
  have hx : x = positions.getD (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) 0 := by
    simpa using hreads
  have hget : D.proved[D.proved.idxOf u.val]'hi = u.val := by
    have := Array.findIdx_getElem (xs := D.proved) (p := (· == u.val)) (w := hi)
    simp only [beq_iff_eq] at this
    exact this
  have hdig : zo.pub.digest u.val q.val c.ports.size =
      Registered.tableRow dp tb q.val (Registered.ref tb c.ports.size q.val (D.proved.idxOf u.val)) := by
    have h1 := Hidden.drawn_digestW (Hidden.loadPublic_nil zo.hpub) hdpR hD (Layout.proved_lt_population hD) hi q.isLt
    rw [hget] at h1
    rw [← h1]
    simp only [Public.digest, htb, Registered.tableRow, Registered.ref]
  exact ⟨_, by rw [hx, hdig]; exact hopens⟩

/-- **A row's digest is fixed by its leaf**: `vb_cs_inl`'s digest at the public file's `b ‖ c` is `cmtOf`'s at a `b ‖ c`
of the same tree leaf. -/
theorem h512_of_treeLeaf {b : ByteArray} {l : List UInt8}
    (h : Hm96.Default512.treeLeaf b = Hm96.Default512.treeLeaf ⟨l.toArray⟩) :
    Refine.H512 (Refine.K512.leafPrefix ++ b.data.toList) = Refine.H512 (Refine.K512.leafPrefix ++ l) := by
  apply Subtype.ext
  rw [H512_val, H512_val]
  show Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ b.data.toList).toArray⟩ =
    Sha512.hash ⟨(Hm96.Default512.leafPrefix.data.toList ++ l).toArray⟩
  have hl : l = (⟨l.toArray⟩ : ByteArray).data.toList := by simp
  rw [hl, ← data_append, ← data_append, bytes_of_list, bytes_of_list]
  exact h

/-! ## At the family -/

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : Law n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂)

/-- **The verifier's `rec-c<k>` roots are its own, over the history's `c_k`**: at every history `t`, each port of session
`j`'s reads file registering `rec-c<k>` has two `hm96-sha512/row/v1` leaves and the root `commitsRoot` over two `b ‖ c`,
row `x`'s the `b ‖ c` (`cOf`) of the commitment of `t`'s round `x · n₀ + kc k` (rep `x`'s message round `k`, `sel`'s
row `x`) where `t` has that round. V* builds them from the `c_k` alone (`rec_vstage.messages`, `rec_vstage.py:164`; `rec_residuals.commits_entry`,
`rec_residuals.py:402-410`). The check that establishes it: the verifier computes its reads file's `rec-c<k>` entries
itself from the round commitments it holds, by `commitsRoot` (no file the prover sends carries them). -/
def HOwnRoots (cOf : Cm → List UInt8) : Prop :=
  ∀ R ω t R₂ j, ∀ reg ∈ (F.zs R ω t R₂ j).own, ∀ k, cIndex reg.value = some k → reg.leaves = 2 ∧
    reg.schema = HmRow.ROW_SCHEMA ∧ ∃ c₀ c₁, reg.root = commitsRoot reg.domain c₀ c₁ ∧ ∀ x < 2, ∀ e : Cm × (ℐ).Coin, t[x * F.n₀ + F.kc k]? = some e →
      commitAt c₀ c₁ x = ⟨(cOf e.1).toArray⟩

/-- **An accepted transcript draws the units that read a round's commitment, and labels them itself**: at every
history, each transcript session `j` accepts has a draw that draws every unit some round-commitment port of `own`
reads into its table (`partOf`, at the table's draw `drawJ`), and that table's drawn statement (`HmOut.drawn`) labels
the unit with the unit. U4 (`checkFull` at the record's draw, which `flock-verify --registered --draw` runs) draws those
units in full; the labels are the verifier's own where its partition derives the units (`Registered.derived`, which
`Registered.check` requires). -/
def RowsDrawn : Prop :=
  ∀ R ω t R₂ j tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr → ∀ reg ∈ (F.zs R ω t R₂ j).own,
    Registered.roundCommitment reg.value = true → ∀ (u : Fin (F.Vs j).nv) x, reg.reads[u.val]? = some (some x) →
      ∃ d D dp idx, recordDraw tr.record = some d ∧
        Draw.UnitDraw.ofJson (drawJ (F.zs R ω t R₂ j).I d (partOf (F.zs R ω t R₂ j).I d u.val)) = .ok D ∧
        HmOut.drawn (F.Vs j).c (F.zs R ω t R₂ j).pub (drawJ (F.zs R ω t R₂ j).I d (partOf (F.zs R ω t R₂ j).I d u.val)) =
          .ok dp ∧ Registered.unitIndices dp = .ok idx ∧
        u.val ∈ D.proved ∧ idx.getD (D.proved.idxOf u.val) 0 = u.val

/-- **A session's circuit names each port once**: a fact of V*'s circuits (`rec_residuals.statement`'s ports are
`S["ports"]`'s names, distinct). -/
def PortNamesUnique : Prop :=
  ∀ j (q q' : Fin (F.Vs j).c.ports.size), (F.Vs j).c.ports[q].name = (F.Vs j).c.ports[q'].name → q = q'

theorem find_of_unique (hu : F.PortNamesUnique) (j : Fin F.stmts.size) (q : Fin (F.Vs j).c.ports.size) :
    (List.range (F.Vs j).c.ports.size).find? (fun p => (F.Vs j).c.ports[p]!.name == (F.Vs j).c.ports[q].name) =
      some q.val := by
  rw [List.find?_eq_some_iff_getElem]
  refine ⟨by rw [getElem!_pos (F.Vs j).c.ports q.val q.isLt]; simp, q.val, by simp, by simp, fun k hk => ?_⟩
  have hk' : k < (F.Vs j).c.ports.size := by omega
  simp only [List.getElem_range]
  rw [getElem!_pos (F.Vs j).c.ports k hk']
  have hne : (F.Vs j).c.ports[k].name ≠ (F.Vs j).c.ports[q].name := fun he =>
    Nat.ne_of_lt hk (congrArg Fin.val (hu j ⟨k, hk'⟩ q he))
  simpa using hne

/-- **`PubCommits` at accepting sessions, up to a SHA-512 collision.** At every history `t` whose stage-`i` commitment is
`e.1` and every registration, at each session that accepts some transcript, each row `(u, q)` of `skOf`'s commit
strings registers `cmtOf (bc cOf)`'s digest of `e.1` (`VCommits` at the row), or the verifier's opening of the public
file's row and the honest opening of the `c_k` collide (`openPair`). -/
theorem pubCommits_or_collides (cOf : Cm → List UInt8) (hSk : F.Sk = skOf F.Vs F.own F.sel) (hown : F.HOwn)
    (hroots : F.HOwnRoots cOf) (hdrawn : F.RowsDrawn) (hnames : F.PortNamesUnique) :
    ∀ i R ω (t : List (Cm × (ℐ).Coin)) (e : Cm × (ℐ).Coin), t[i]? = some e → ∀ R₂ j,
      (∃ tr, AcceptsZKR (F.zs R ω t R₂ j).I (F.zs R ω t R₂ j).own tr) → ∀ (u : Fin (F.Vs j).nv)
        (q : Fin (F.Vs j).c.ports.size), Sum.inl (u, q) ∈ F.Sk i j → ∀ R' : Reg',
          (F.zs R ω t R₂ j).vb.cs R' (.inl (u, q)) = cmtOf F.Vs (F.bc cOf) j e.1 (.inl (u, q)) ∨
            ∃ reg x path y path', Registered.Collides (Registered.openPair reg x y path
              (Hm96.Default512.treeLeaf ⟨(cOf e.1).toArray⟩) path') := by
  intro i R ω t e he R₂ j ⟨tr, hacc⟩ u q hpos R'
  rw [hSk, mem_skOf_inl] at hpos
  obtain ⟨p, hp, hname, x, hx, hsel⟩ := hpos
  obtain ⟨hrc, k, hk, hx2, rfl⟩ := hsel
  have hq : (p.name, p.value, p.reads) ∈ msgPorts (F.zs R ω t R₂ j).own :=
    (hown R ω t R₂ j _).2 (mem_msgPorts.2 ⟨p, hp, isMsg_ownOf hp, rfl⟩)
  obtain ⟨reg, hreg, -, hre⟩ := mem_msgPorts.1 hq
  simp only [Prod.mk.injEq] at hre
  obtain ⟨hn, hv, hr⟩ := hre
  have hxr : reg.reads[u.val]? = some (some x) := by rw [hr]; exact hx
  obtain ⟨d, D, dp, idx, hd, hD, hdp, hidx, hu, hlab⟩ :=
    hdrawn R ω t R₂ j tr hacc reg hreg (by rw [hv]; exact hrc) u x hxr
  have hfind := F.find_of_unique hnames j q
  rw [← hname, ← hn] at hfind
  obtain ⟨path, hopen⟩ := opens_of_acceptsZKR (F.zs R ω t R₂ j) hacc hreg hfind hd hD hdp hidx hu
    (by rw [hlab]; exact hxr)
  obtain ⟨hl, hsch, c₀, c₁, hroot, hc⟩ := hroots R ω t R₂ j reg hreg k (by rw [hv]; exact hk)
  have hat := hc x hx2 e he
  rcases treeLeaf_eq_or_collides hl hsch hroot hx2 hopen with h | h
  · left
    rw [hat] at h
    rw [vb_cs_inl]
    exact congrArg (fun z => ((Sum.inl (u, q) : PosK (F.Vs j).c (F.Vs j).hU (F.Vs j).nv), z))
      (h512_of_treeLeaf h)
  · right
    rw [hat] at h
    exact ⟨reg, x, path, _, _, h⟩

/-- Obligation: discharged by a verifier check that `pub.bin`'s rows at its `rec-c<k>` reads are the `c_k` it holds (not
run today).
**The public file's round-commitment rows are the `c_k`'s**: at every history `t`, where session `j`'s circuit port
`q` is a port of its reads file registering `rec-c<k>` and unit `u` reads it at row `x`, the public file's `b ‖ c` of
`u`'s row `q` is the `b ‖ c` (`cOf`) of the commitment of `t`'s round `x · n₀ + kc k`. The check that establishes it:
the verifier compares `pub.bin`'s rows at its `rec-c<k>` reads with the `c_k` it holds (`rec_vstage.py:164`), rather
than opening them against `commits_entry`'s root (`Flock.Registered.check`), which binds them only up to a SHA-512
collision (`pubCommits_or_collides`). -/
def HPubRows (cOf : Cm → List UInt8) : Prop :=
  ∀ R ω t R₂ j, ∀ reg ∈ (F.zs R ω t R₂ j).own, ∀ k, cIndex reg.value = some k →
    ∀ (u : Fin (F.Vs j).nv) (q : Fin (F.Vs j).c.ports.size) x, (F.Vs j).c.ports[q].name = reg.name →
      reg.reads[u.val]? = some (some x) → ∀ e : Cm × (ℐ).Coin, t[x * F.n₀ + F.kc k]? = some e →
        (F.zs R ω t R₂ j).pub.digest u.val q.val (F.Vs j).c.ports.size = ⟨(cOf e.1).toArray⟩

/-- **`PubCommits` from the verifier's checks** (`HOwn`, `HPubRows`), at every accepting session (and at every other). -/
theorem pubCommits_of_pubRows (cOf : Cm → List UInt8) (hSk : F.Sk = skOf F.Vs F.own F.sel) (hown : F.HOwn)
    (hrows : F.HPubRows cOf) : F.PubCommits (F.bc cOf) := by
  intro i R ω t e he R₂ j _ u q hpos
  rw [hSk, mem_skOf_inl] at hpos
  obtain ⟨p, hp, hname, x, hx, hsel⟩ := hpos
  obtain ⟨-, k, hk, -, rfl⟩ := hsel
  have hq : (p.name, p.value, p.reads) ∈ msgPorts (F.zs R ω t R₂ j).own :=
    (hown R ω t R₂ j _).2 (mem_msgPorts.2 ⟨p, hp, isMsg_ownOf hp, rfl⟩)
  obtain ⟨reg, hreg, -, hre⟩ := mem_msgPorts.1 hq
  simp only [Prod.mk.injEq] at hre
  obtain ⟨hn, hv, hr⟩ := hre
  have h := hrows R ω t R₂ j reg hreg k (by rw [hv]; exact hk) u q x (hname.symm.trans hn.symm)
    (by rw [hr]; exact hx) e he
  show _ = cOf e.1
  simp [h]

end VStarFamily

end family

end FlockVBridge.Family
