import Proofs.Flock.Soundness.Discharge.Integrate.Bridge
import Proofs.Flock.Soundness.Discharge.DecodeZL.Statements
import Proofs.Flock.Soundness.Discharge.ZkExec.Statements
import Proofs.Flock.Soundness.Discharge.ZkSession.Session
import Proofs.Flock.Soundness.Discharge.ZkExec.Setup
import Proofs.Flock.Soundness.Discharge.ZkExec.Verify
import Proofs.Flock.Soundness.Discharge.Hidden.Wide.Block
import Proofs.Flock.Soundness.Discharge.ZkLink.Sched

/-!
# The `--zk` table at each draw, from flock-verify's setup (statements)

#828's `tablesAt` is the plain headline's table at draw `S`: `stmtOf` of the statement setupH sets up at the coin
server's draw `dj S`, wherever flock-verify accepts. Here is its `--zk` counterpart, `tablesAtZK`, a `TabZK` built from
what `verify --zk` sets up at the draw (`zkInputs I`):

* the statement `stmtOf st` at setupH's statement `st` and its regions, link layout at `PT_LOCAL`;
* `Zk.schedule st.m`'s schedule and `t_pad`, the padding constant `PAD_C`, two extra lanes;
* the pads at `PadLayout.new st.m st.kLog k0`'s positions (`padsOfLay`, `padsAOfLay`, `tripleOfLay`);
* the inner code `verifyInner` reads: `innerN` columns, column `j` at the point `j + 1` (`codeAt`), `K_PAD` padding
  symbols, `Q` queries at `w.lo mod innerN`, and the unique-decoding radius.

`DrawSetupZK` is a draw `verify --zk` sets up (`DrawSetup` at `zkInputs I`) with its schedule and the facts the table
needs of it; at a draw with none the table is a reference setup's statement with `contra`'s two regions alone, which no
witness satisfies (as `tableAt`'s `refusedC`).

* `RecordsLiveZK`, `DrawOkZK`: #862's `RecordsLive` and `DrawOk` at `--zk` (`AcceptsZK`, `zkInputs I`).
* `TablesAtZKLive`: wherever flock-verify accepts what `wr` writes, the draw's `--zk` table is the executable's: its
  statement is `stmtOf` of setupH's, its dimensions and inner code are `Zk.verify`'s, and its extra claims at the
  record's link points are the executable's (`ExtraRealizes`, as `ofCircuit_extra`).
* `tablesAtZK_live`: its proof, with `0 < g` from setupH (`setupH_g_pos`, D9), the schedule's shape and padding from
  the m check (`schedule_zk`), the inner code's length (`innerN_pow`) and its query positions' uniformity
  (`prCoin_innerPos_le`), and the table the same at every `DrawSetupZK` of a draw (`tableZK_eq`).
-/

namespace FlockSoundness.Discharge.ZkLink

open Flock FlockLevel3 Game Model Refine Exec Integrate DecodeZL ZkSession ZkExec
open scoped ENNReal NNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The pads and the inner code of a pads layout -/

/-- Pad `i` of a layout, as an index into its `len` pads. -/
def padIx (l : Zk.PadLayout) (i : ℕ) : Fin l.len := ⟨i % l.len, Nat.mod_lt _ (Nat.succ_pos _)⟩

/-- Level 0's pads at a layout's positions: the OOD value, the lane rounds' messages and the pair's sums. -/
def padsOfLay (l : Zk.PadLayout) : Pads0 l.len 2 :=
  ⟨padIx l (l.lgOod 0), fun j t => padIx l (l.lgMsg j t), fun i c => padIx l (l.lgE i c)⟩

/-- The algebraic phase's pads at a layout's positions. -/
def padsAOfLay (l : Zk.PadLayout) : PadsA l.len :=
  ⟨fun i => padIx l (l.r1ab i), fun i => padIx l (l.r1c i), fun j t => padIx l (l.ml j t), padIx l l.fc,
    fun j t => padIx l (l.lc j t), fun i => padIx l (l.zp i)⟩

/-- The sacrifice's pads at a layout's positions: `f_a, f_b, h_ab` and the triple `x, y, w`. -/
def tripleOfLay (l : Zk.PadLayout) : Triple l.len :=
  ⟨padIx l l.fa, padIx l l.fb, padIx l l.hab, padIx l l.tx, padIx l l.ty, padIx l l.tw⟩

/-- The `--zk` pads layout of a statement at a schedule. -/
def layOf (st : Stmt) (sch : Flock.Schedule) : Zk.PadLayout := Zk.PadLayout.new st.m st.c.kLog sch.k0

/-- `codeAt`'s evaluation point of column `j`: `j + 1` as a word. -/
def innerPt (j : ℕ) : 𝔽 := fOf ⟨(j + 1).toUInt64, 0⟩

/-- `verifyInner`'s query position of a coin word: `w.lo mod N`. -/
def innerPos (N : ℕ) (hN : 0 < N) (w : 𝔽) : Fin N := ⟨(GF128.val w).lo.toNat % N, Nat.mod_lt _ hN⟩

/-! ## The inner code's facts -/

/-- Residues mod `N` of `Fin (K · N)`: each class has `K` members. -/
theorem card_mod_mem (K N : ℕ) (hN : 0 < N) (D : Finset (Fin N)) :
    (Finset.univ.filter fun a : Fin (K * N) => (⟨a.val % N, Nat.mod_lt _ hN⟩ : Fin N) ∈ D).card = K * D.card := by
  classical
  have e : (Finset.univ.filter fun a : Fin (K * N) => (⟨a.val % N, Nat.mod_lt _ hN⟩ : Fin N) ∈ D) =
      (Finset.univ ×ˢ D).map finProdFinEquiv.toEmbedding := by
    ext a
    simp only [Finset.mem_filter, Finset.mem_univ, true_and, Finset.mem_map_equiv, Finset.mem_product]
    have : (⟨a.val % N, Nat.mod_lt _ hN⟩ : Fin N) = (finProdFinEquiv.symm a).2 := by
      ext; simp [finProdFinEquiv_symm_apply, Fin.modNat]
    rw [this]
  rw [e, Finset.card_map, Finset.card_product, Finset.card_univ, Fintype.card_fin]

set_option linter.constructorNameAsVariable false in
/-- **`verifyInner`'s query position is uniform** when `N` divides `2^64`: `w.lo mod N` lands in `D` with probability
`|D| / N`. -/
theorem prCoin_innerPos_le (N : ℕ) (hN : 0 < N) (hd : N ∣ 2 ^ 64) (D : Finset (Fin N)) :
    prCoin (fun w : 𝔽 => innerPos N hN w ∈ D) ≤ (D.card : ℝ≥0∞) / N := by
  classical
  obtain ⟨K, hK⟩ := hd
  have hK' : 2 ^ 64 = K * N := by rw [hK, Nat.mul_comm]
  let E : 𝔽 ≃ Fin (K * N) × Fin (2 ^ 64) :=
    FlockLevel3.f128Equiv.trans (Equiv.prodCongr (finCongr hK') (Equiv.refl _))
  have hE : ∀ w : 𝔽, (innerPos N hN w).val = (E w).1.val % N := fun w => rfl
  rw [prCoin_eq_card]
  have hc : (Finset.univ.filter fun w : 𝔽 => innerPos N hN w ∈ D).card = (2 ^ 64 * K) * D.card := by
    have hb : (Finset.univ.filter fun w : 𝔽 => innerPos N hN w ∈ D).card =
        ((Finset.univ.filter fun a : Fin (K * N) => (⟨a.val % N, Nat.mod_lt _ hN⟩ : Fin N) ∈ D) ×ˢ
          (Finset.univ : Finset (Fin (2 ^ 64)))).card := by
      refine Finset.card_bij (fun w _ => ((E w).1, (E w).2)) ?_ ?_ ?_
      · intro w hw
        simp only [Finset.mem_filter, Finset.mem_univ, true_and] at hw
        simp only [Finset.mem_product, Finset.mem_filter, Finset.mem_univ, true_and, and_true]
        convert hw using 1
        exact Fin.ext (hE w).symm
      · intro a _ b _ h
        exact E.injective (Prod.ext (Prod.ext_iff.1 h).1 (Prod.ext_iff.1 h).2)
      · intro x hx
        refine ⟨E.symm x, ?_, by simp⟩
        simp only [Finset.mem_product, Finset.mem_filter, Finset.mem_univ, true_and, and_true] at hx
        simp only [Finset.mem_filter, Finset.mem_univ, true_and]
        convert hx using 1
        exact Fin.ext (by rw [hE]; simp)
    rw [hb, Finset.card_product, card_mod_mem K N hN D, Finset.card_univ, Fintype.card_fin]
    ring
  have hF : Fintype.card 𝔽 = (2 ^ 64 * K) * N := by
    rw [Fintype.card_congr E, Fintype.card_prod, Fintype.card_fin, Fintype.card_fin]
    ring
  have hK0 : 2 ^ 64 * K ≠ 0 := by
    rintro h
    rcases Nat.mul_eq_zero.1 h with h | h
    · exact absurd h (by positivity)
    · rw [h, Nat.zero_mul] at hK'; exact absurd hK' (by positivity)
  rw [hc, hF]
  generalize 2 ^ 64 * K = c at hK0
  rw [Nat.cast_mul, Nat.cast_mul, ENNReal.mul_div_mul_left _ _ (by exact_mod_cast hK0) (ENNReal.natCast_ne_top _)]


/-- The pads layout's length at `m ≤ 35`, `k_log ≤ 27` and `k0 ≤ 35`. -/
theorem layLen_le {m kLog k0 : ℕ} (hm : m ≤ 35) (hk : kLog ≤ 27) (h0 : k0 ≤ 35) :
    (Zk.PadLayout.new m kLog k0).len ≤ 444 := by
  simp only [Zk.PadLayout.len, Zk.PadLayout.tw, Zk.PadLayout.hab, Zk.PadLayout.zp, Zk.PadLayout.lc, Zk.PadLayout.fc,
    Zk.PadLayout.fa, Zk.PadLayout.new]
  omega

/-- The inner code's length holds the pads message. -/
theorem innerN_ge (lay : Zk.PadLayout) : lay.len + Zk.K_PAD ≤ ZkExec.innerN lay :=
  (Nat.le_nextPowerOfTwo _).trans (Nat.le_mul_of_pos_right _ (by positivity))

/-- The inner code's length is a power of two, at most `2^13` for a short layout. -/
theorem innerN_pow (lay : Zk.PadLayout) (h : lay.len ≤ 444) :
    ∃ k, ZkExec.innerN lay = 2 ^ k ∧ k ≤ 13 := by
  obtain ⟨k, hk⟩ := Nat.isPowerOfTwo_nextPowerOfTwo (lay.len + Zk.K_PAD)
  have hle : (lay.len + Zk.K_PAD).nextPowerOfTwo ≤ 2 ^ 10 :=
    Nat.nextPowerOfTwo_le (by simp only [Zk.K_PAD]; omega) ⟨10, rfl⟩
  rw [hk] at hle
  refine ⟨k + 3, by rw [ZkExec.innerN, hk, Zk.INNER_LOG_INV_RATE, pow_add], ?_⟩
  have := (Nat.pow_le_pow_iff_right (by norm_num : 1 < 2)).1 hle
  omega

/-! ## A draw `verify --zk` sets up -/

variable {n : ℕ} (I : Inputs) (mPts : ℕ)

/-- **A draw `verify --zk` sets up**: a `DrawSetup` at `zkInputs I` (setupH accepts the draw under the `--zk` tags and
coin tree), `Zk.schedule`'s schedule and `t_pad` at its `m`, and the facts of them the `--zk` table needs: the schedule's
shape and `m`, `t_pad`'s padding, the inner code's length (below `2^64`, so `codeAt`'s points are distinct, and at least
the pads message's), and its query positions' uniformity; and the statement passes the `--zk` shape check, as
`verify --zk` checks at setup (`Zk.shapeOk`). -/
structure DrawSetupZK (S : Finset (Fin n)) (d : Lean.Json) where
  x : DrawSetup (zkInputs I) mPts S d
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
  hzk : Zk.shapeOk x.st = .ok ()

variable {I mPts}

namespace DrawSetupZK

variable {S : Finset (Fin n)} {d : Lean.Json} (y : DrawSetupZK I mPts S d)

theorem innerN_pos : 0 < innerN (layOf y.x.st y.sch) := lt_of_lt_of_le (by simp [Zk.K_PAD]) y.hL

/-- `codeAt`'s points, distinct below `2^64`. -/
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

/-- **The `--zk` table of a draw `verify --zk` sets up**, at statement regions `regs`. -/
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

/-- The draw's `--zk` table: setupH's statement at its regions. -/
noncomputable def table : TabZK 𝔽 mPts := y.tabAt (regionsOf y.x.st) (stmtOf_linkLayout y.x.st y.x.wf.1 y.x.wf.2).1

/-- A refused draw's `--zk` table: the statement with `contra`'s two regions alone, which no witness satisfies. -/
noncomputable def refused : TabZK 𝔽 mPts :=
  y.tabAt (DrawSetup.contra y.x.st.c.kLog (y.x.st.m - y.x.st.c.kLog)) DrawSetup.contra_wf

end DrawSetupZK

section

variable {S : Finset (Fin n)} {d : Lean.Json}

/-- `drawSetup_of_setupH` without the unit's returning the outputs: `0 < st.c.g` from setupH (`setupH_g_pos`). -/
theorem drawSetup_of_setupH_g (ht : I.tags.typed = false) {st : Flock.Stmt}
    (h : Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some d) I.partition I.program = .ok st)
    {sch : Flock.Schedule} (hf : Flock.fast100 st.m = some sch) {D : Draw.UnitDraw}
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hS : ∀ u ∈ S, u.val ∈ D.proved) (hmPts : st.mPts = mPts) :
    ∃ x : Integrate.DrawSetup I mPts S d, x.st = st := by
  have hg : 0 < st.c.g := Hidden.setupH_g_pos ht h
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn h
  obtain ⟨D', hD', -, hn, -⟩ := Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  obtain ⟨hn', hnbl⟩ := Integrate.setupH_n h
  have hblk : D.proved.size ≤ st.c.g * 2 ^ st.nbl := by
    rw [hnbl, ← hn, ← hn']
    exact Integrate.fits_blocks hg
  obtain ⟨k0, l₀, l₁, rest, ext, hsch⟩ := Integrate.model_sched hf
  exact ⟨⟨st, h, D, hD, hS, hg, hblk, st.m, k0, l₀, l₁, rest, ext, hsch,
    by have := exec_fast100_ge hf; omega, hmPts⟩, rfl⟩

/-- A `DrawSetup` at `zkInputs I` whose statement passes `Zk.shapeOk`, with `Zk.schedule`'s schedule at its `m`, is a
`DrawSetupZK`. -/
theorem drawSetupZK_of (ht : I.tags.typed = false) (x : Integrate.DrawSetup (ZkExec.zkInputs I) mPts S d)
    (hzk : Zk.shapeOk x.st = .ok ()) {sch : Flock.Schedule} {tPad : ℕ} (hz : Zk.schedule x.st.m = .ok (sch, tPad)) :
    ∃ y : DrawSetupZK I mPts S d, y.x = x ∧ y.sch = sch ∧ y.tPad = tPad := by
  obtain ⟨hshape, hmsch, hpad⟩ := schedule_zk hz
  obtain ⟨-, hm35, -⟩ := schedule_spec hz
  have ht' : (ZkExec.zkInputs I).tags.typed = false := (ZkExec.zkTags_typed I).trans ht
  have hk := (setupH_spec ht' x.h).2.2.2.2.2.1
  have h0 : sch.k0 ≤ 35 := by
    have := hshape.2.2.1
    simp only [schedOf] at this hmsch
    omega
  have hlen := layLen_le hm35 hk h0
  obtain ⟨k, hk', hk13⟩ := innerN_pow _ hlen
  have hN : ZkExec.innerN (layOf x.st sch) < 2 ^ 64 := by
    rw [layOf, hk']
    exact Nat.pow_lt_pow_right (by norm_num) (by omega)
  have hd : ZkExec.innerN (layOf x.st sch) ∣ 2 ^ 64 := by
    rw [layOf, hk']
    exact Nat.pow_dvd_pow 2 (by omega)
  exact ⟨⟨x, sch, tPad, hz, hshape, hmsch, hpad, hN, innerN_ge _,
    fun D => prCoin_innerPos_le _ _ hd D, hzk⟩, rfl, rfl, rfl⟩

/-- A draw's `--zk` table is the same at every `DrawSetupZK` of it: setupH and `Zk.schedule` are functions. -/
theorem tableZK_eq (y₁ y₂ : DrawSetupZK I mPts S d) : y₁.table = y₂.table := by
  obtain ⟨⟨st₁, h₁, D₁, hD₁, hS₁, hg₁, hb₁, m₁, k₁, a₁, b₁, r₁, e₁, hs₁, hm₁, hp₁⟩, sch₁, t₁, hz₁, _, _, _, _, _, _⟩ := y₁
  obtain ⟨⟨st₂, h₂, D₂, hD₂, hS₂, hg₂, hb₂, m₂, k₂, a₂, b₂, r₂, e₂, hs₂, hm₂, hp₂⟩, sch₂, t₂, hz₂, _, _, _, _, _, _⟩ := y₂
  obtain rfl : st₁ = st₂ := Except.ok.inj (h₁.symm.trans h₂)
  obtain ⟨rfl, rfl⟩ := Prod.ext_iff.1 (Except.ok.inj (hz₁.symm.trans hz₂))
  rfl

end

variable (ht : I.tags.typed = false) {Reg : Type} (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
  (y₀ : DrawSetupZK I mPts S₀ (dj S₀))

open Classical in
/-- **The `--zk` table at draw `S`** (`tableAt` at `--zk`): the draw's table when `verify --zk` sets up the draw's JSON
`dj S`, and otherwise the reference setup `y₀`'s refused table. -/
noncomputable def tableAtZK (S : Finset (Fin n)) : TabZK 𝔽 mPts :=
  if h : Nonempty (DrawSetupZK I mPts S (dj S)) then (Classical.choice h).table else y₀.refused

/-- **The `--zk` tables** (`tablesAt` at `--zk`): `tableAtZK` at every registration. -/
noncomputable abbrev tablesAtZK : Finset (Fin n) → Reg → TabZK 𝔽 mPts := fun S _ => tableAtZK dj y₀ S

/-! ## The live session's record -/

/-- **`wr` is the challenger's record of the live `--zk` session** (#862's `RecordsLive` at `--zk`): whenever
flock-verify accepts what `wr` writes for the outcome `o` at draw `S`, the record carries the draw `dj S`, the proof
files are `o`'s, and the record decodes to a session whose table's `root_B`, link points and rep streams' rounds are
`o`'s. -/
def RecordsLiveZK (I : Inputs) (dj : Finset (Fin n) → Lean.Json)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ S R o, AcceptsZK I (wr S R o) →
    recordDraw (wr S R o).record = some (dj S) ∧ (wr S R o).proofs = o.proofs ∧
    ∀ spec sts sess, setupOf (zkInputs I) (wr S R o).record = .ok (spec, sts) →
      Record.decode spec (wr S R o).record = .ok sess →
      sess.rootB[0]! = o.rootB ∧ ptsOf mPts sess.points = o.pts ∧
      ∀ r : Fin 2, ∃ s : StreamRec,
        sess.streams.find? (fun x => x.name == toString spec.tables[0]! ++ toString "/rep" ++ toString r.val) =
          some s ∧ s.rounds.toList = pick (o.rep0, o.rep1) r

/-- **The challenger's draw at `S`, at `--zk`** (#862's `DrawOk` at `zkInputs I`): it proves `S`, and setupH sets up
at it under the `--zk` tags and coin tree with `mPts` link points. -/
def DrawOkZK (I : Inputs) (dj : Finset (Fin n) → Lean.Json) (mPts : ℕ) (S : Finset (Fin n)) : Prop :=
  (∀ D, Draw.UnitDraw.ofJson (dj S) = .ok D → ∀ u ∈ S, u.val ∈ D.proved) ∧
    ∀ st, Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition I.program =
      .ok st → st.mPts = mPts

/-! ## The statement -/

/-- **Wherever flock-verify accepts, the `--zk` table is the executable's.** On public-output one-table inputs whose
`--statement` tags are untyped, with `wr` the challenger's record (`RecordsLiveZK`): whenever the draw `S` is `DrawOkZK`
(pointwise: link points depend on `|S|`, so no `mPts` holds at every draw) and `verify --zk` accepts what `wr` writes
for `o` at it, setupH's statement `st` at `dj S` under the `--zk` tags is
well formed (`StmtWF`, `RegionsWF`), `Zk.schedule st.m` is `(sch, t_pad)`, the draw has a `DrawSetupZK` at `st`, `sch`
and `t_pad` whose table (`stmtOf st` at `st`'s regions, `Zk.verify`'s dimensions, pads and inner code) is
`tablesAtZK … S R`, and at the record's link points `Setup.ofCircuit st`'s extra claims are that table's
(`ExtraRealizes`, as `ofCircuit_extra`). The reference setup `y₀` (a refused draw's table) is data, so no hypothesis is
taken at its draw `S₀`. -/
def TablesAtZKLive : Prop :=
  ∀ (I : Inputs) (_ht : I.tags.typed = false) (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)}
    (y₀ : DrawSetupZK I mPts S₀ (dj S₀)) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript),
    I.tags.hiddenOutputs = false → I.count ≤ 1 → RecordsLiveZK I dj wr →
    ∀ S R o, DrawOkZK I dj mPts S → AcceptsZK I (wr S R o) →
      ∃ (st : Stmt) (sch : Flock.Schedule) (tPad : ℕ), StmtWF st ∧ RegionsWF st ∧
        Stmt.setupH (zkTags I) I.circuitFile I.publicFile (zkCoins I) I.tables (some (dj S)) I.partition I.program =
          .ok st ∧ Zk.schedule st.m = .ok (sch, tPad) ∧
        ∃ y : DrawSetupZK I mPts S (dj S), y.x.st = st ∧ y.sch = sch ∧ y.tPad = tPad ∧
          tablesAtZK (Reg := Reg) dj y₀ S R = y.table ∧
          ∀ points : Array (Array F128), points.size = 2 → (∀ p ∈ points.toList, p.size = mPts) →
            ExtraRealizes (Setup.ofCircuit st) y.table.S points
              (Model.extraClaims y.table.S y.table.ptLocal mPts (ptsOf mPts points))

theorem tablesAtZK_live : TablesAtZKLive (n := n) (mPts := mPts) (Reg := Reg) := by
  intro I ht dj S₀ y₀ wr hh hc1 hrec S R o hdraw hacc
  obtain ⟨st, hst, hsh, hv⟩ := ZkExec.acceptsZK_one I _ hh hc1 hacc
  rw [(hrec S R o hacc).1] at hst
  obtain ⟨-, sess, schs, -, -, -, -, -, hall⟩ := ZkExec.zk_verify_ok _ _ _ _ hv
  obtain ⟨hs, c, hz, -⟩ := hall 0 (by simp)
  have hz' : Zk.schedule st.m = .ok (schs[0].1, schs[0].2) := hz
  generalize schs[0].1 = sch at hz'
  generalize schs[0].2 = tPad at hz'
  obtain ⟨-, -, sch₀, hf⟩ := schedule_spec hz'
  have ht' : (ZkExec.zkInputs I).tags.typed = false := (ZkExec.zkTags_typed I).trans ht
  obtain ⟨pop, -, hdr⟩ := Layout.setupH_drawn hst
  obtain ⟨D, hD, -⟩ := Layout.drawn_spec hdr
  obtain ⟨x, rfl⟩ := drawSetup_of_setupH_g (I := ZkExec.zkInputs I) (S := S) ht' hst hf hD (hdraw.1 D hD)
    (hdraw.2 _ hst)
  obtain ⟨y, rfl, rfl, rfl⟩ := drawSetupZK_of ht x hsh hz'
  have hwf := setupH_wf hst
  refine ⟨y.x.st, y.sch, y.tPad, hwf.1, hwf.2, hst, hz', y, rfl, rfl, rfl, ?_, ?_⟩
  · show tableAtZK dj y₀ S = y.table
    unfold tableAtZK
    split
    · exact tableZK_eq _ _
    · rename_i h
      exact absurd ⟨y⟩ h
  · intro points hp hsz
    have hm := y.x.hmPts
    have := ofCircuit_extra y.x.st hwf.1 hwf.2 points hp (by rw [hm]; exact hsz)
    rw [hm] at this
    exact this

end FlockSoundness.Discharge.ZkLink
