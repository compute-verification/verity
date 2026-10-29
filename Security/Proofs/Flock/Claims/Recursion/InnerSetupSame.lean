import Proofs.Flock.Claims.Recursion.InnerForm
import Proofs.Flock.Soundness.Discharge.Layout.SharedRows
import Proofs.Flock.Soundness.Discharge.Hidden.Public

/-!
# A full draw sets the inner statement up as its files do, region for region

`innerSetup_files` gives Flock's plain verifier's inner statement (set up at the inner statement's draw) the files'
`m`, `k_log` and region count. `innerSetup_same` gives it the files' statement whole, as the verifier's algebra reads
it: the extra claims at any points (`Stmt.extraClaims`) and the lincheck's circuit fold (`Stmt.circuitFold`).

* `drawn_same`: a hidden-output statement set up at a draw that proves every one of the undrawn statement's units, in
  order, has its circuit, leaf, `nbl`, `m`, pin, Δ and regions. The regions read the public file only at its instances'
  `b ‖ c` (`regions_same`), which the drawn file keeps at each proved unit (`drawn_digestW`).
* `extraClaims_congr`, `circuitFold_congr`: the extra claims read a statement's `k_log`, `m`, `nbl` and regions, and the
  circuit fold its circuit, leaf, pin and Δ.
-/

namespace RecDeploy

open Flock FlockSoundness.Refine
open Claims (Msg)

namespace SetupSame

/-! ## The statement's algebra reads few of its fields -/

theorem listMapM_congr {β : Type} :
    ∀ {l : List ℕ} {f g : ℕ → Except String β}, (∀ p ∈ l, f p = g p) → l.mapM f = l.mapM g
  | [], _, _, _ => rfl
  | a :: l, f, g, h => by
    rw [List.mapM_cons, List.mapM_cons, h a List.mem_cons_self,
      listMapM_congr fun p hp => h p (List.mem_cons_of_mem _ hp)]

theorem mapM_range_congr {β : Type} {n : ℕ} {f g : ℕ → Except String β} (h : ∀ p < n, f p = g p) :
    (Array.range n).mapM f = (Array.range n).mapM g := by
  rw [Array.mapM_eq_mapM_toList, Array.mapM_eq_mapM_toList, Array.toList_range,
    listMapM_congr fun p hp => h p (List.mem_range.1 hp)]

/-- **The extra claims read the statement's `k_log`, `m`, `nbl` and regions.** -/
theorem extraClaims_congr {st st' : Stmt} (hk : st.c.kLog = st'.c.kLog) (hm : st.m = st'.m) (hn : st.nbl = st'.nbl)
    (hr : st.regions = st'.regions) : st.extraClaims = st'.extraClaims := by
  have hp : st.regionPoint = st'.regionPoint := by
    funext reg r
    unfold Stmt.regionPoint
    rw [hk, hm, hn]
  have hv : st.regionValue = st'.regionValue := by
    funext reg r
    unfold Stmt.regionValue
    rw [hn]
  funext pts
  unfold Stmt.extraClaims
  rw [hr, hp, hv]

section Fold

variable {st st' : Stmt}

theorem slotBase_congr (hc : st.c = st'.c) (hl : st.leaf = st'.leaf) (a : F128) (lw rho : Array F128)
    (key : Kind × ℕ) : st.slotBase a lw rho key = st'.slotBase a lw rho key := by
  unfold Stmt.slotBase
  rw [hc, hl]

theorem foldRanges_congr (hc : st.c = st'.c) (hl : st.leaf = st'.leaf) (a : F128) (lw rho : Array F128) :
    ∀ rs bases out, st.foldRanges a lw rho rs bases out = st'.foldRanges a lw rho rs bases out
  | [], _, _ => rfl
  | r :: rs, bases, out => by
    rw [Stmt.foldRanges, Stmt.foldRanges]
    simp only [slotBase_congr hc hl, foldRanges_congr hc hl a lw rho rs]

theorem unitShare_congr (hc : st.c = st'.c) (a : F128) (lw rho rr : Array F128) (r : Range) :
    st.unitShare a lw rho rr r = st'.unitShare a lw rho rr r := by
  cases st
  cases st'
  dsimp only at hc
  subst hc
  rfl

theorem partialRangesT_congr (hc : st.c = st'.c) (hl : st.leaf = st'.leaf) (a : F128) (lw rho rr : Array F128) :
    ∀ rs bases out, st.partialRangesT a lw rho rr rs bases out = st'.partialRangesT a lw rho rr rs bases out
  | [], _, _ => rfl
  | r :: rs, bases, out => by
    rw [Stmt.partialRangesT, Stmt.partialRangesT]
    simp only [slotBase_congr hc hl, unitShare_congr hc, partialRangesT_congr hc hl a lw rho rr rs]

/-- **The circuit fold reads the statement's circuit, leaf, pin and Δ.** -/
theorem circuitFold_congr (hc : st.c = st'.c) (hl : st.leaf = st'.leaf) (hpin : st.pin = st'.pin)
    (hda : st.da = st'.da) (hdb : st.db = st'.db) : st.circuitFold = st'.circuitFold := by
  have hfold : st.fold = st'.fold := by
    funext a lw rho
    unfold Stmt.fold
    simp only [foldRanges_congr hc hl, hc, hda, hdb]
  have hstr : st.structured = st'.structured := by
    unfold Stmt.structured
    rw [hc]
  have hpart : st.foldedPartial = st'.foldedPartial := by
    funext a b lw rho ts
    unfold Stmt.foldedPartial
    simp only [partialRangesT_congr hc hl, hc, hda, hdb, hpin]
  have hfolded : st.folded = st'.folded := by
    funext a b lw rho ts
    unfold Stmt.folded
    rw [hstr, hpart, hfold, hpin, hc]
  unfold Stmt.circuitFold
  rw [hc, hpin, hfold, hfolded]

end Fold

/-! ## A full draw keeps the regions -/

/-- **`HmOut.regions` reads the public file only at its instances' `b ‖ c`.** -/
theorem regions_same {c : Circuit} {pub pop : Public} (hn : pub.n = pop.n)
    (hd : ∀ r < pop.n, ∀ p < c.ports.size, pub.digest r p c.ports.size = pop.digest r p c.ports.size) :
    HmOut.regions c pub = HmOut.regions c pop := by
  have hb : ∀ p < c.ports.size, ∀ b g, FlockSoundness.Discharge.Layout.bcChunk c pub b g p =
      FlockSoundness.Discharge.Layout.bcChunk c pop b g p := by
    intro p hp b g
    unfold FlockSoundness.Discharge.Layout.bcChunk Circuit.instOf
    rw [hn]
    by_cases hi : b * c.g + g < pop.n
    · simp only [hi, ↓reduceIte]
      exact hd _ hi p hp
    · simp only [hi, ↓reduceIte]
  unfold HmOut.regions
  simp only []
  apply mapM_range_congr
  intro p hp
  have e : (fun b => (Array.range c.g).foldl
        (fun acc g => acc ++ FlockSoundness.Discharge.Layout.bcChunk c pub b g p) ByteArray.empty) =
      fun b => (Array.range c.g).foldl
        (fun acc g => acc ++ FlockSoundness.Discharge.Layout.bcChunk c pop b g p) ByteArray.empty := by
    funext b
    simp only [hb p hp b]
  exact congrArg (mkRegion _ _ _ _) e

theorem getElem_range_of {A : Array ℕ} {N r : ℕ} (h : A = Array.range N) (hr : r < A.size) : A[r] = r := by
  subst h
  simp

/-- **A full draw keeps the statement's algebra**: a hidden-output statement set up at a draw that proves every one
of the undrawn statement's units, in order, has its circuit, leaf, `nbl`, `m`, pin, Δ and regions. -/
theorem drawn_same {T : Tags} {cf pf : ByteArray} {co co' : Coins} {tb : List (String × ByteArray)}
    {pa pa' : Option Lean.Json} {pr pr' : Option ByteArray} {d : Lean.Json} {D : Draw.UnitDraw} {st st' : Stmt}
    (h' : Stmt.setupHidden T cf pf co tb none pa pr = .ok st')
    (h : Stmt.setupHidden T cf pf co' tb (some d) pa' pr' = .ok st)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hp : D.proved = Array.range st'.n) :
    st.c = st'.c ∧ st.leaf = st'.leaf ∧ st.nbl = st'.nbl ∧ st.m = st'.m ∧ st.pin = st'.pin ∧ st.da = st'.da ∧
      st.db = st'.db ∧ st.regions = st'.regions := by
  have s := FlockSoundness.Discharge.Hidden.setupHidden_spec' h
  have s' := FlockSoundness.Discharge.Hidden.setupHidden_spec' h'
  have hc : st.c = st'.c := Except.ok.inj (s.parse.symm.trans s'.parse)
  obtain ⟨pop, hpop, -, hd⟩ := s.pub
  obtain ⟨pop', hpop', hu', -⟩ := s'.pub
  have hpub' : st'.pub = pop' := hu' rfl
  rw [hc] at hpop
  have hpp : pop' = pop := Except.ok.inj (hpop'.symm.trans hpop)
  rw [hpp] at hpub'
  have hdr : HmRow.drawn { st'.c with outPorts := #[] } pop d = .ok st.pub := by
    have := hd d rfl
    rw [hc] at this
    exact this
  obtain ⟨D', hD', hpn, hn, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
  rw [hD] at hD'
  cases hD'
  have hpopn : pop.n = st'.n := by rw [s'.n, hpub']
  have hN : st.n = st'.n := by rw [s.n, hn, hp, Array.size_range]
  have hnbl : st.nbl = st'.nbl := by rw [s.nbl, s'.nbl, s.blocksReal, s'.blocksReal, hN, hc]
  have hpin : st.pin = st'.pin := by
    have := s.pin
    rw [hc, s'.pin] at this
    exact (Except.ok.inj this).symm
  have hreg : st.regions = st'.regions := by
    have e : HmOut.regions st'.c st.pub = HmOut.regions st'.c pop := by
      apply regions_same
      · rw [← s.n, hN, hpopn]
      · intro r hr p hq
        have hlt : ∀ u ∈ D.proved, u < D.population := by
          intro u hu
          rw [hp, Array.mem_range] at hu
          rw [hpn, hpopn]
          exact hu
        have hr' : r < D.proved.size := by rw [hp, Array.size_range, ← hpopn]; exact hr
        have := FlockSoundness.Discharge.Hidden.drawn_digestW
          (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop) hdr hD hlt hr' hq
        rw [getElem_range_of hp hr'] at this
        exact this
    have := s.regions
    rw [hc, e, ← hpub', s'.regions] at this
    exact (Except.ok.inj this).symm
  refine ⟨hc, s.leaf.trans s'.leaf.symm, hnbl, by rw [s.m, s'.m, hnbl, hc], hpin, ?_, ?_, hreg⟩
  · rw [s.da, s'.da, hc, hpin]
  · rw [s.db, s'.db, hc, hpin]

/-- **A full draw keeps the statement's extra claims, circuit fold and link points' length.** -/
theorem drawn_setup {T : Tags} {cf pf : ByteArray} {co co' : Coins} {tb : List (String × ByteArray)}
    {pa pa' : Option Lean.Json} {pr pr' : Option ByteArray} {d : Lean.Json} {D : Draw.UnitDraw} {st st' : Stmt}
    (h' : Stmt.setupHidden T cf pf co tb none pa pr = .ok st')
    (h : Stmt.setupHidden T cf pf co' tb (some d) pa' pr' = .ok st)
    (hD : Draw.UnitDraw.ofJson d = .ok D) (hp : D.proved = Array.range st'.n) :
    st.m = st'.m ∧ st.c.kLog = st'.c.kLog ∧ st.extraClaims = st'.extraClaims ∧ st.circuitFold = st'.circuitFold ∧
      st.mPts = st'.mPts := by
  obtain ⟨hc, hl, hn, hm, hpin, hda, hdb, hr⟩ := drawn_same h' h hD hp
  refine ⟨hm, by rw [hc], extraClaims_congr (by rw [hc]) hm hn hr, circuitFold_congr hc hl hpin hda hdb, ?_⟩
  rw [(FlockSoundness.Discharge.Hidden.setupHidden_spec' h).mPts,
    (FlockSoundness.Discharge.Hidden.setupHidden_spec' h').mPts, hn]

end SetupSame

open SetupSame

/-- **Flock's plain verifier sets the inner statement up as its files do, as the algebra reads it**: where the
setup's checks accept what the verifier reads of a compiled statement (`RecScope.accepts`), Flock's plain verifier sets
the inner statement up as one table whose `m`, `k_log`, extra claims (at any points) and circuit fold are the files'
statement's (`RecCoins.Files.stmt`), in a session of two link points of the files' `m_pts`. As `innerSetup_files`,
with `drawn_setup` for `drawn_full`. -/
theorem innerSetup_same {x : Msg} {r : XR} (hr : readX x = some r) (ha : RecScope.accepts (readOf r) = true) :
    ∃ f spec' st' s spec st, filesOf r = some (.files f) ∧ f.stmt "the inner session" = .ok (spec', st') ∧
      FlockDeploy.setupOf false r.w.inner = some (s, spec, #[st]) ∧
      st.m = st'.m ∧ st.c.kLog = st'.c.kLog ∧ st.extraClaims = st'.extraClaims ∧
      st.circuitFold = st'.circuitFold ∧ spec.points = 2 ∧ spec.mPts = st'.mPts := by
  unfold RecScope.accepts at ha
  simp only [Bool.and_eq_true, RecScope.isOk_iff] at ha
  obtain ⟨⟨-, ⟨u, hsh⟩⟩, hheld⟩ := ha
  have hsh : RecSetup.checkShape (some r.shape) (filesOf r) = .ok u := hsh
  have hheld : (filesOf r).isSome = true := hheld
  obtain ⟨p, hp⟩ : ∃ p, r.partition = some p := by
    cases hq : r.partition with
    | none => simp [filesOf, hq] at hheld
    | some p => exact ⟨p, rfl⟩
  have hfs : filesOf r = some (.files ⟨r.innerRaw.circuit, r.innerRaw.publicFile, p, "", none,
      some (innerFormOf r.innerRaw)⟩) := by rw [filesOf, hp]; rfl
  rw [hfs] at hsh
  obtain ⟨spec', st', hstm, -, -, -, -, hform⟩ := checkShape_files hsh
  obtain ⟨hst', hspec'⟩ := files_stmt hstm
  obtain ⟨hs1, hs2, hs3, b, d, D, hb, hjd, hD, hpr⟩ := checkForm_ok hform
  have hs1 : r.innerRaw.statement = Tags.circuitPlainLeaves.name := hs1
  have hs2 : r.innerRaw.osCoins = true := hs2
  have hs3 : r.innerRaw.tables = [] := hs3
  have hb : r.innerRaw.draw = some b := hb
  obtain ⟨hdec, -, hok⟩ := readX_inner hr
  obtain ⟨s, spec, sts, hset, hchk⟩ := innerOk_ok hok
  obtain ⟨hins, hsu⟩ := FlockDeploy.setupOf_some hset
  have hw : r.innerRaw.zk = false ∧ r.innerRaw.ins = .ok s := by
    unfold FlockDeploy.insOf at hins
    obtain ⟨w, hw0, hw⟩ := Option.bind_eq_some_iff.1 hins
    rw [hdec] at hw0
    cases hw0
    split_ifs at hw with hz
    · cases hw
    change r.innerRaw.ins.toOption = some s at hw
    exact ⟨not_not.1 hz, toOption_eq_some hw⟩
  obtain ⟨z1, z2, z3, z4, z5, -, z7, z8, z9, -⟩ := ins_plainLeaves hw.2 hw.1 hs1 hs2
  obtain ⟨hcount, hpi⟩ := check_plain hchk z1
  have hpart : s.read.I.partition = some p := by
    have hpa := (readX_spec hr).2.1
    rw [z8] at hpa
    simp only [Except.toOption, Option.some.injEq] at hpa
    rw [hpa, hp]
  have hdraw : s.read.draw = some (Flock.norm d) := by
    obtain ⟨dr, hz9, he⟩ := z9
    rw [hb] at hz9
    simp only [Option.mapM_some, hjd, Functor.map, Except.map, Except.ok.injEq] at hz9
    rw [he, ← hz9]
    rfl
  have hsetup := FlockDeploy.Ins.setup_read hsu
  unfold Readable.Read.setup at hsetup
  rw [z2, z3, z4, z5, z7, hs3, hdraw, hpart, hpi] at hsetup
  have hh : RecCoins.innerTags.hiddenOutputs = true := rfl
  rcases FlockSoundness.Discharge.HiddenExec.setupTables_okH hh hsetup with ⟨-, st, hst, rfl, rfl⟩ | ⟨hc, -⟩
  · obtain ⟨e1, e2, e3, e4, e5⟩ := drawn_setup hst' hst hD hpr
    exact ⟨_, spec', st', s, _, st, hfs, hstm, hset, e1, e2, e3, e4, rfl, e5⟩
  · omega

end RecDeploy
