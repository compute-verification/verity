import Proofs.Flock.Soundness.Discharge.ZkExec.VerdictReplay

/-!
# A `--zk` rep's inner check holds on its inner openings

`ZkRepAccepts`'s second conclusion. The compiled rep's seven level-0 rows are the seven constraints `repProg` hands
`Zk.verifyInner`, in their order: the replay's three (`zk_replay_rows`), the two ring switches' (the masked opening's
values are the claim checks, `openingZK_plays'`) and level 0's two lane rows (`ligeritoPadZC_row`). So
`zk_inner_accepts` applies to the compiled rep's inner proof.

* `zk_decode_paths`: a decoded proof's path entries are 64 bytes.
* `zk_inner_rows`: the run of the compiled rep that `zk_rep_plays` builds has an inner check that holds on the proof's
  inner openings.
* `zk_inner_of_rep`: at `ZkRepAccepts`'s hypotheses, every result of the compiled rep does (`plays_det`).
-/

namespace FlockSoundness.Discharge.ZkExec

open Flock Refine Exec Game Model DecodeZL Transcript Reader
open FlockSoundness.Refine (ebind_ok)

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

/-! ## The decoded paths -/

theorem yields_vecAny {α : Type} {minSize : ℕ} {elem : Reader.M α} {w : String} {a : Array α}
    (h : Yields (vecAny minSize elem w) a) : ∀ e ∈ a, Yields elem e := by
  unfold vecAny at h
  obtain ⟨n, -, h⟩ := yields_bind h
  obtain ⟨r, -, h⟩ := yields_bind h
  dsimp only at h
  by_cases hc : n.toNat * minSize > r.data.size - r.pos
  · rw [ite_eq_left hc] at h
    obtain ⟨_, h, -⟩ := yields_bind h
    exact (yields_throw h).elim
  · rw [ite_eq_right hc] at h
    obtain ⟨out, hout, h⟩ := yields_bind h
    rw [yields_pure h]
    obtain ⟨hs, hk⟩ := yields_forIn_range (fun _ => elem) n.toNat _ out hout
    have h0 : (Array.mkEmpty n.toNat : Array α).size = 0 := rfl
    intro e he
    obtain ⟨k, hk', rfl⟩ := Array.mem_iff_getElem.1 he
    rw [hs, h0] at hk'
    obtain ⟨_, hy⟩ := hk k (by omega)
    simpa [h0] using hy

theorem post_inner_paths :
    Post Zk.innerReader (fun i => ∀ k < i.paths.size, ∀ b ∈ (i.paths[k]!).toList, b.size = 64) := by
  unfold Zk.innerReader
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
  refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun paths hp => post_pure ?_
  intro k hk b hb
  rw [getElem!_pos paths k hk] at hb
  exact yields_bytes (yields_vecAny (yields_vecAny hp _ (Array.getElem_mem hk)) b (Array.mem_toList_iff.1 hb))

/-- **A decoded proof's path entries are 64 bytes.** -/
theorem zk_decode_paths {sh : Shape} {bytes : ByteArray} {zp : Zk.ZkProof} (h : Zk.decode sh bytes = .ok zp) :
    ∀ k < zp.inner.paths.size, ∀ b ∈ (zp.inner.paths[k]!).toList, b.size = 64 := by
  unfold Zk.decode at h
  obtain ⟨⟨p', r'⟩, hrun, hp⟩ := ebind_ok h
  have hp' : p' = zp := Except.ok.inj hp
  rw [← hp']
  refine (?_ : Post (Zk.reader sh)
    (fun z => ∀ k < z.inner.paths.size, ∀ b ∈ (z.inner.paths[k]!).toList, b.size = 64)) p' ⟨_, _, hrun⟩
  unfold Zk.reader
  dsimp only
  refine post_bind_vecBytes fun cap hcap hcapE => ?_
  refine post_bind fun m _ => post_bind fun lir _ => post_bind fun lbs _ => post_bind fun profile _ => ?_
  refine post_bind fun tag _ => ?_
  split
  rotate_left 2
  · exact post_bind_throw
  all_goals
    refine post_bind fun numLanes _ => post_bind fun mh _ => ?_
    refine post_bind_vec fun r1ab hr1ab => post_bind_vec fun r1c hr1c => post_bind_vec fun rounds hrounds => ?_
    refine post_bind fun fa _ => post_bind fun fb _ => post_bind fun fc _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun lr hlr => post_bind_vec fun zp hzp => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_switches fun rs hrs hrsE => post_bind fun _ _ => ?_
    refine post_bind_vecBytes fun icap _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun n _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind_caps fun rcaps hrc hrcE => ?_
    refine post_bind fun np _ => post_ite (fun _ => post_bind_throw) (fun _ => ?_)
    refine post_bind fun rproofs _ => post_bind_vec fun yr hyr => ?_
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind_vec fun msgs hmsgs => post_bind_vec fun qn hqn => post_bind_vec fun ood hood => ?_
    refine post_bind fun _ _ => post_bind_vec fun cbn hcbn => post_bind_vec fun con hcon => ?_
    refine post_bind fun _ _ => ?_
    split
    rotate_left 1
    · exact post_bind_throw
    · exact post_bind_throw
    refine post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => post_bind fun _ _ => ?_
    refine post_bind fun _ _ => post_bind fun inner hinner => ?_
    exact post_pure (post_inner_paths inner hinner)

/-! ## The layout's positions -/

theorem lay_lt (l : Zk.PadLayout) :
    (∀ i < 64, l.r1ab i < l.len) ∧ (∀ i < 64, l.r1c i < l.len) ∧ (∀ j < l.nMlv, ∀ t < 2, l.ml j t < l.len) ∧
      l.fc < l.len ∧ (∀ j < l.nLc, ∀ t < 2, l.lc j t < l.len) ∧ (∀ i < 64, l.zp i < l.len) ∧
      l.lgOod 0 < l.len ∧ (∀ j < l.k0, ∀ t < 4, l.lgMsg j t < l.len) ∧ (∀ i < 2, ∀ c < 2, l.lgE i c < l.len) ∧
      l.fa ≠ l.fb ∧ l.fa ≠ l.hab ∧ l.fb ≠ l.hab := by
  unfold Zk.PadLayout.len Zk.PadLayout.tw Zk.PadLayout.hab Zk.PadLayout.lgOod Zk.PadLayout.lgMsg Zk.PadLayout.lgE
    Zk.PadLayout.zp Zk.PadLayout.lc Zk.PadLayout.fc Zk.PadLayout.fb Zk.PadLayout.fa Zk.PadLayout.r1ab Zk.PadLayout.r1c
    Zk.PadLayout.ml
  refine ⟨fun _ _ => ?_, fun _ _ => ?_, fun _ _ _ _ => ?_, ?_, fun _ _ _ _ => ?_, fun _ _ => ?_, ?_,
    fun _ _ _ _ => ?_, fun _ _ _ _ => ?_, ?_, ?_, ?_⟩ <;> omega

/-! ## The masked opening's values -/

/-- `openingZK_plays` with the values the masked opening gives the two claims: the claim checks `va`, `vc`. -/
theorem openingZK_plays' {m : ℕ} (hm : 7 ≤ m) (va vc : 𝔽) (a c : Model.Claim 𝔽 m) (extra : List (Model.Claim 𝔽 m))
    {ms : List Msg} {cs : List (Array F128)} {op : OpenOut 𝔽}
    (h : Plays fOf (Model.opening execArith hm (⟨va, a.skip, a.x⟩ :: ⟨vc, c.skip, c.x⟩ :: extra)) ms cs (some op)) :
    ∃ x, Plays fOf (ZkSession.openingZK execArith hm a c extra) ms cs x ∧ x.1 = some op ∧ x.2 = (va, vc) := by
  rw [opening_eq] at h
  obtain ⟨ms₁, ms₂, cs₁, cs₂, sw, rfl, rfl, h₁, h⟩ := plays_bind_inv _ h
  obtain ⟨ms₃, ms₄, cs₃, cs₄, γ, rfl, rfl, h₃, h⟩ := plays_bind_inv _ h
  obtain ⟨rfl, rfl, hop⟩ := plays_ret_inv h
  simp only [List.mapM_cons, bind_eq, pure_eq, bind_assoc, bind_ret] at h₁
  obtain ⟨msx, ms₅, csx, cs₅, x, rfl, rfl, hx, h₁⟩ := plays_bind_inv _ h₁
  obtain ⟨msy, ms₆, csy, cs₆, y, rfl, rfl, hy, h₁⟩ := plays_bind_inv _ h₁
  obtain ⟨msE, ms₇, csE, cs₇, swE, rfl, rfl, hE, h₁⟩ := plays_bind_inv _ h₁
  obtain ⟨rfl, rfl, rfl⟩ := plays_ret_inv h₁
  obtain ⟨x2, rfl, hxa⟩ := opStep_inv hx
  obtain ⟨y2, rfl, hyc⟩ := opStep_inv hy
  rw [ZkSession.opResult_two execArith hm va vc a c extra x2 y2 swE γ] at hop
  split_ifs at hop with hck
  · exact ⟨_, by
      unfold ZkSession.openingZK
      exact Plays.bind' (hxa a) (Plays.bind' (hyc c) (Plays.bind' hE (Plays.bind' h₃ (Plays.pure _) rfl rfl) rfl rfl)
        rfl rfl) (by simp) (by simp), by exact hop.symm, by rw [← hck.1, ← hck.2]⟩

/-! ## Level 0's lane rows, at the rounds `Zk.ligerito` consumed -/

theorem consumed_lig (rs : Array Round) (k K k' : ℕ) (h : k + 4 + K ≤ k') :
    consumed rs k k' = (rs[k]!).coins :: (rs[k + 1]!).coins :: ((List.range' 0 K).map (fun i => (rs[k + 2 + i]!).coins) ++
      (rs[k + 2 + K]!).coins :: (rs[k + 3 + K]!).coins :: consumed rs (k + 4 + K) k') := by
  rw [consumed_append rs (Nat.le_succ k) (by omega : k + 1 ≤ k'), consumed_succ,
    consumed_append rs (Nat.le_succ (k + 1)) (by omega : k + 1 + 1 ≤ k'), consumed_succ,
    consumed_append rs (show k + 1 + 1 ≤ k + 2 + K by omega) (by omega : k + 2 + K ≤ k'),
    show k + 2 + K = (k + 2) + K by rfl, consumed_eq_map,
    consumed_append rs (Nat.le_succ (k + 2 + K)) (by omega : k + 2 + K + 1 ≤ k'), consumed_succ,
    consumed_append rs (Nat.le_succ (k + 2 + K + 1)) (by omega : k + 2 + K + 1 + 1 ≤ k'), consumed_succ,
    show k + 2 + K + 1 = k + 3 + K by omega]
  have e1 : (k + 3 + K).succ = k + 4 + K := by omega
  have e2 : ∀ i, k + 1 + 1 + i = k + 2 + i := fun i => by omega
  rw [e1]
  simp [e2]

theorem getElem!_map_range' {β : Type} [Inhabited β] (f : ℕ → β) (K : ℕ) {i : ℕ} (hi : i < K) :
    ((List.range' 0 K).map f)[i]! = f i := by
  rw [getElem!_pos ((List.range' 0 K).map f) i (by simpa using hi)]
  simp

/-- **Level 0's lane row, at a run of `Zk.ligerito`**: a run of the compiled masked level 0 on the rounds the run
consumed, at pads the layout places and the target the opening's `T` is, returns the lane phase's row (`laneLhs`) at the
run's record. -/
theorem zk_lig_row (ms : MerkleScheme) (sch : Flock.Schedule) (T : F128) (cap : Array ByteArray)
    (p : LigeritoProof) (basisAt : Array F256 → ℕ → Array F256) (zl : Zk.Level0) (t t' : Flock.Transcript)
    (rec : Zk.L0Rec) (hdim : DimSeam sch) (hlv : ∀ i < sch.levels.size, LevelSeam (sch.levels.getD i default))
    (hα1 : ∀ i < sch.levels.size, Accounting.ceilLog2 (sch.levels.getD i default).q ≠ 1) (hk0 : 1 ≤ sch.k0)
    (hwf : RoundsWF t.rounds) (h : Zk.ligerito ms sch T cap p basisAt zl t = .ok (rec, t'))
    (lay : Zk.PadLayout) (hk' : sch.k0 = lay.k0) {np e : ℕ} (P : ZkSession.Pads0 np e) (hnp : np = lay.len)
    (he : e = 2) (hood : P.ood.val = lay.lgOod 0)
    (hmsg : ∀ j < lay.k0, ∀ t : Fin 4, (P.msg j t).val = lay.lgMsg j t.val)
    (hev : ∀ (i : Fin e) (c : Fin 2), (P.ev i c).val = lay.lgE i.val c.val) (c₀ : 𝔽) (r : ℕ) (Tm : 𝔽) (b : ℕ → 𝔽)
    (hT : Tm = fOf T) :
    ∃ out : ZkSession.LinK 𝕂 np × PendL 𝔽 Dig,
      Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith P (schedOf sch) zl.tPad c₀ r Tm b)
        (ligMsgsZ p capDig sch zl.tPad zl.msgs) (consumed t.rounds t.k t'.k) out ∧
      Rel2 out.1 (laneLhs lay rec p zl.msgs) := by
  subst hnp he
  obtain ⟨out, hpl, -, -⟩ := zk_ligerito_reqs ms sch T cap p basisAt zl t t' rec hdim hlv hα1 hk0 hwf h P c₀ r Tm b
  obtain ⟨hTr, -, hβ, -, hch, hρ, -, hkk, hnz, hnb, hnl, hnρ₀, hnρ₁⟩ :=
    zk_ligerito_rec ms sch T cap p basisAt zl t t' rec hdim hlv hk0 h
  refine ⟨out, hpl, ?_⟩
  obtain ⟨hn7, h2, -, hc0, -, hcols, -⟩ := hdim
  have hc01 := hcols 0 (by omega)
  rw [consumed_lig t.rounds t.k sch.k0 t'.k hkk] at hpl
  obtain ⟨lv₀, rest, hls⟩ : ∃ lv₀ rest, sch.levels.toList = lv₀ :: rest :=
    List.exists_cons_of_ne_nil (by
      intro h0; have := congrArg List.length h0; rw [Array.length_toList, List.length_nil] at this; omega)
  have hlvs : (schedOf sch).levels = levelOf lv₀ :: rest.map levelOf := by simp [schedOf, hls]
  refine ligeritoPadZC_row lay P (schedOf sch) sch zl.tPad c₀ r Tm b p zl.msgs rec (levelOf lv₀) (rest.map levelOf)
    hlvs hk' hk' (by omega) hood hmsg hev (by rw [hT, hTr]) _ _ _ _ _ _ ?_ ?_ ?_ hβ ?_ ?_ ?_ ?_ ?_ hpl
  · show _ = sch.m - 7
    rw [RoundsWF.get hwf, hnz, hn7]
  · show sch.m - 7 ≠ 1
    omega
  · rw [RoundsWF.get hwf, hnb]
  · simp [hk']
  · intro i hi
    rw [getElem!_map_range' _ _ (by omega), RoundsWF.get hwf, hnl i (by omega), hch i (by omega)]
    exact ⟨rfl, rfl⟩
  · rw [RoundsWF.get hwf, hnρ₀]
  · rw [RoundsWF.get hwf, hnρ₁]
  · exact hρ

/-! ## The seven rows -/

theorem cons7_get {α : Type} [Inhabited α] (xs lg : Array α) (a b : α) (hX : xs.size = 3) (hlg : lg.size = 2) :
    (xs ++ #[a, b] ++ lg)[0]! = xs[0]! ∧ (xs ++ #[a, b] ++ lg)[1]! = xs[1]! ∧ (xs ++ #[a, b] ++ lg)[2]! = xs[2]! ∧
      (xs ++ #[a, b] ++ lg)[3]! = a ∧ (xs ++ #[a, b] ++ lg)[4]! = b ∧ (xs ++ #[a, b] ++ lg)[5]! = lg[0]! ∧
      (xs ++ #[a, b] ++ lg)[6]! = lg[1]! := by
  rcases xs with ⟨_ | ⟨x₀, _ | ⟨x₁, _ | ⟨x₂, _ | ⟨_, _⟩⟩⟩⟩⟩ <;> simp at hX
  rcases lg with ⟨_ | ⟨l₀, _ | ⟨l₁, _ | ⟨_, _⟩⟩⟩⟩ <;> simp at hlg
  simp

theorem rows7 {np : ℕ} {β : Type} (row row' A B C D E : ZkSession.Lin 𝔽 np) (p : β)
    (cons : Array (String × Zk.Lin)) (h0 : LinRel B cons[0]!.2) (h1 : LinRel row cons[1]!.2)
    (h2 : LinRel row' cons[2]!.2) (h3 : LinRel A cons[3]!.2) (h4 : LinRel C cons[4]!.2) (h5 : LinRel D cons[5]!.2)
    (h6 : LinRel E cons[6]!.2) :
    ∀ i : Fin 7, LinRel (((ZkLink.consRowC row (ZkLink.consRowC row' ((![A, B, C, D, E] : Fin 5 → _), p))).1 ∘
      ![3, 0, 1, 2, 4, 5, 6]) i) cons[i.val]!.2 := by
  intro i
  fin_cases i
  exacts [h0, h1, h2, h3, h4, h5, h6]

/-! ## The rep's inner check -/

set_option maxHeartbeats 1000000 in
/-- **`repProg`'s run has an inner check that holds** (`zk_rep_plays` with its inner check): the run of the compiled
rep that `zk_rep_plays` builds, at a table whose pads and inner code are the executable's and whose statement's fold
the setup's circuit fold realizes, on a decoded proof, has an inner check that holds on the proof's inner openings. -/
theorem zk_inner_rows {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2) (st : Setup)
    (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript)
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (hseam : ZkTabSeam T st sch tPad) (htab : ZkTabLay T st sch)
    (hfold : FoldRealizes st.fold T.S)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hdec : Zk.decode (shapeZK st sch extras) bytes = .ok zp) (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) :
    ∃ pend : ZkLink.PendZ 𝔽 Dig (List UInt8),
      Plays fOf (ZkLink.repZCAt execArith hs512 T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
        (rounds.toList.map (·.coins)) pend ∧
      pend.inner (innerOpensOf zp.inner) := by
  obtain ⟨h25, -, -, hk0, -, hdim, hlv, hα1, -⟩ := zk_schedule_seams _ _ _ hsch
  have hTm : T.S.m = st.m := hseam.hm
  have h7 : 7 ≤ T.S.m := by omega
  obtain ⟨b1, b2, b3, b4, b5, b6, b7, b8, b9, d1, d2, d3⟩ := lay_lt (layZ st sch)
  obtain ⟨efa, efb, eab, -, -, -⟩ := tr_pos htab
  unfold repProg at hrun
  rw [StateT.run] at hrun
  try dsimp only at hrun
  obtain ⟨rp, t₁, h₁, hrun⟩ := bind_ok hrun
  try dsimp only at hrun
  obtain ⟨⟨Tb, sws⟩, t₂, h₂, hrun⟩ := bind_ok hrun
  obtain ⟨rc, t₃, h₃, hrun⟩ := bind_ok hrun
  cases hv : Zk.ligeritoConstraints (Zk.PadLayout.new st.m st.kLog sch.k0) rc zp.proof.ligerito zp.level0 with
  | error e =>
    rw [hv] at hrun
    try dsimp only at hrun
    obtain ⟨_, _, h₄, _⟩ := bind_ok hrun
    cases h₄
  | ok lg =>
  rw [hv] at hrun
  try dsimp only at hrun
  obtain ⟨lg', t₄, h₄, h₅⟩ := bind_ok hrun
  clear hrun
  obtain ⟨hlg, ht₄⟩ := pure_ok h₄
  rw [hlg, ht₄] at h₅
  have hwf₀ : RoundsWF (Flock.Transcript.init rounds dom).rounds := hwf
  obtain ⟨hr₁, hk₁, k₁, zc, row, abM, row', hk₁a, hk₁b, hpz, hpl, hc0, hc1, hcs3, hR0, hR1, hR2, hAB, hCV⟩ :=
    zk_replay_rows st (layZ st sch) zp.proof zp.inner.root _ t₁ rp T.S hTm.symm hseam.hk.symm (by omega) hwf₀ hfold
      h₁ T.padsA T.tr hseam.hnp.symm
      (fun i => by rw [htab.hr1ab, Nat.mod_eq_of_lt (b1 _ i.isLt)])
      (fun i => by rw [htab.hr1c, Nat.mod_eq_of_lt (b2 _ i.isLt)])
      (fun j hj c => by rw [htab.hml, Nat.mod_eq_of_lt (b3 _ (by show j < st.m - 6; omega) _ c.isLt)])
      (by rw [htab.hfc, Nat.mod_eq_of_lt b4])
      (fun j hj c => by
        rw [htab.hlc, Nat.mod_eq_of_lt (b5 _ (by show j < st.kLog - 6; rw [← hseam.hk]; omega) _ c.isLt)])
      (fun i => by rw [htab.hzp, Nat.mod_eq_of_lt (b6 _ i.isLt)]) efa efb eab d1 d2 d3
  rw [← hTm] at h₂
  obtain ⟨g₀, g₁, g₂⟩ := two_append_get! ({ rp.claims[0]! with value := Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]! } : Flock.Claim)
    ({ rp.claims[1]! with value := Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]! } : Flock.Claim) extras
  obtain ⟨hr₂, hk₂, op, hop, hopR⟩ := opening_refines T.S.m h7 _
    (⟨fOf (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!), abM.skip, abM.x⟩ ::
      ⟨fOf (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!), execArith.lagS zc.z, zc.r⟩ ::
        Model.extraClaims T.S T.ptLocal mPts pts)
    (by rw [size_two_append]; simp; omega)
    (fun k hk => by
      rcases k with _ | _ | k
      · rw [g₀]; exact hc0 _
      · rw [g₁]; exact hc1 _
      · have hk' : k < (Model.extraClaims T.S T.ptLocal mPts pts).length := by simp at hk; omega
        rw [g₂ k (by omega)]
        exact hext k hk')
    (by rw [size_two_append]; omega) (by rw [size_two_append]; omega) _ t₁ t₂ Tb sws (hr₁ ▸ hwf₀) h₂
  rw [size_two_append] at hop
  obtain ⟨xo, hxo, hxo1, hxo2⟩ := openingZK_plays' h7 _ _ abM.pub ⟨0, execArith.lagS zc.z, zc.r⟩ _ hop
  obtain ⟨xo1, xo2⟩ := xo
  obtain rfl : xo1 = some op := hxo1
  obtain rfl : xo2 = _ := hxo2
  have hwf₂ : RoundsWF t₂.rounds := hr₂ ▸ hr₁ ▸ hwf₀
  obtain ⟨hr₃, hk₃, -⟩ := zk_ligerito_plays st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩
    t₂ t₃ rc hdim hlv hα1 hk0 hwf₂ h₃
  have hlig : ∃ out : ZkSession.LinK 𝕂 T.np × PendL 𝔽 Dig,
      Plays fOf (ZkLink.ligeritoPadZC (D := Dig) execArith T.pads T.sch T.t T.c₀ r.val op.T op.b)
        (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0) (consumed t₂.rounds t₂.k t₃.k) out ∧
      Rel2 out.1 (laneLhs (layZ st sch) rc zp.proof.ligerito zp.level0) := by
    rw [hseam.hsch, hseam.ht]
    exact zk_lig_row st.merkle sch Tb zp.proof.cap zp.proof.ligerito _ ⟨tPad, r.val, zp.level0⟩ t₂ t₃ rc hdim hlv hα1
      hk0 hwf₂ h₃ (layZ st sch) rfl T.pads hseam.hnp hseam.he (by rw [htab.hood, Nat.mod_eq_of_lt b7])
      (fun j hj t => by rw [htab.hmsg, Nat.mod_eq_of_lt (b8 _ hj _ t.isLt)])
      (fun i c => by
        rw [htab.hev, Nat.mod_eq_of_lt (b9 _ (by have := i.isLt; have := hseam.he; omega) _ c.isLt)])
      T.c₀ r.val op.T op.b hopR.T
  have hwf₃ : RoundsWF t₃.rounds := hr₃ ▸ hwf₂
  have hlg2 := ligeritoConstraints_size hv
  have hcons7 : (rp.cons ++ #[("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!)),
      ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!))] ++ lg).size = 7 := by
    rw [Array.size_append, Array.size_append, hcs3, hlg2]; rfl
  obtain ⟨hr₅, hk₅, hIP⟩ := zk_inner_plays _ zp.inner zp.proof.zerocheck.finalC _ t₃ tr hcons7 hwf₃ h₅
  obtain ⟨-, hl0, hl1⟩ := ligeritoConstraints_eq hv rfl
  obtain ⟨c0, c1, c2, c3, c4, c5, c6⟩ := cons7_get rp.cons lg
    ("ring switch ab", rp.abValue.addConst (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!))
    ("ring switch c", rp.cValue.addConst (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!)) hcs3 hlg2
  have hM : ∃ o, Plays fOf (ZkLink.repMaskedZC (D := Dig) execArith T.pads T.padsA T.tr T.S T.sch T.hm T.t T.c₀ r.val
        (Model.extraClaims T.S T.ptLocal mPts pts))
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC]))))
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) o ∧
      ∀ i : Fin 7, LinRel (o.1 i) (rp.cons ++ #[("ring switch ab", rp.abValue.addConst
          (Zk.claimCheck rp.claims[0]! zp.proof.ringSwitches[0]!)), ("ring switch c", rp.cValue.addConst
          (Zk.claimCheck rp.claims[1]! zp.proof.ringSwitches[1]!))] ++ lg)[i.val]!.2 := by
    unfold ZkLink.repMaskedZC
    refine pq_map ?_
    refine pq_bind hpz ?_ rfl rfl
    refine pq_map ?_
    refine pq_bind hpl ?_ rfl rfl
    refine pq_map ?_
    unfold ZkLink.restZC
    refine pq_bind hxo ?_ rfl rfl
    obtain ⟨out, hout, hrow⟩ := hlig
    refine pq_bind hout ?_ rfl (List.append_nil _).symm
    refine ⟨_, Plays.bind' (Plays.recv _ (fOf zp.proof.zerocheck.finalC)) (Plays.pure _) rfl rfl, ?_⟩
    obtain ⟨hlr0, hlr1⟩ := limbRows_rel hrow
    refine rows7 _ _ _ _ _ _ _ _ _ ?_ ?_ ?_ ?_ ?_ ?_ ?_
    · rw [c0]; exact hR0
    · rw [c1]; exact hR1
    · rw [c2]; exact hR2
    · rw [c3]; exact rel_addConst hAB _
    · rw [c4, hCV]
      refine rel_addConst ?_ _
      rw [show (layZ st sch).len = T.np from hseam.hnp.symm]
      exact rel_masked _ T.padsA.fc (by rw [htab.hfc, Nat.mod_eq_of_lt b4])
    · rw [c5, hl0]; exact hlr0
    · rw [c6, hl1]; exact hlr1
  have hmsgs : restMsgsZ st sch (2 + extras.size) tPad zp =
      (zcMsgs T.S.m zp.proof.zerocheck ++ (lcMsgs T.S.kLog zp.proof.lincheck ++
        (opMsgs (2 + extras.size) zp.proof.ringSwitches ++
          (ligMsgsZ zp.proof.ligerito capDig sch tPad zp.level0 ++ [fMsg zp.proof.zerocheck.finalC])))) ++
      [⟨𝔽 × 𝔽, (fOf zp.inner.rho, fOf zp.inner.sigma)⟩, ⟨Dig, toDig zp.inner.tauCommit⟩,
        ⟨Fin (T.np + T.nP) → 𝔽, vecOf (T.np + T.nP) zp.inner.y⟩] := by
    rw [hTm, hseam.hk, hseam.hnp, hseam.hnP]
    simp [restMsgsZ, innerMsgs]
  have hcoins : rounds.toList.map (·.coins) =
      (consumed (Flock.Transcript.init rounds dom).rounds (Flock.Transcript.init rounds dom).k k₁ ++
        (consumed (Flock.Transcript.init rounds dom).rounds k₁ t₁.k ++
          (consumed t₁.rounds t₁.k t₂.k ++ consumed t₂.rounds t₂.k t₃.k))) ++ consumed t₃.rounds t₃.k tr.k := by
    rw [hr₃, hr₂, hr₁]
    simp only [List.append_assoc]
    rw [← consumed_append _ hk₃ hk₅, ← consumed_append _ hk₂ (hk₃.trans hk₅),
      ← consumed_append _ hk₁b (hk₂.trans (hk₃.trans hk₅)),
      ← consumed_append _ hk₁a (hk₁b.trans (hk₂.trans (hk₃.trans hk₅))), hk]
    exact (consumed_all _).symm
  unfold ZkLink.repZCAt ZkLink.repZC
  rw [hseam.hq]
  refine exQ_recv (toDig zp.inner.root) ?_
  refine pq_bindQ hM (fun o ho => ?_) hmsgs hcoins
  obtain ⟨v, hv⟩ := hIP hs512 T.dom T.np T.nP T.pos o.1 T.tr (toDig zp.inner.root)
  exact ⟨⟨o.2, v⟩, Plays.bind' hv (Plays.pure _) (List.append_nil _).symm (List.append_nil _).symm,
    zk_inner_accepts _ zp.inner _ _ t₃ tr T st sch tPad o.1 rfl hseam htab hcons7 ho (zk_decode_paths hdec) hwf₃ h₅
      v (by rw [hseam.hq]; exact hv)⟩

/-- **A `--zk` rep's inner check holds on its inner openings** (`ZkRepAccepts`'s second conclusion, at its
hypotheses): every result of the compiled rep, played with the rep's messages and the rounds' coins, has an inner check
that holds at openings whose rep-`r` inner openings are the proof's. -/
theorem zk_inner_of_rep {mPts : ℕ} (T : ZkSession.TabZK 𝔽 mPts) (pts : Fin 2 → Fin mPts → 𝔽) (r : Fin 2)
    (st : Setup) (sch : Flock.Schedule) (tPad : ℕ) (extras : Array Flock.Claim) (bytes : ByteArray) (zp : Zk.ZkProof)
    (rounds : Array Round) (dom : ByteArray) (tr : Flock.Transcript) (cap₀ : Cap Dig)
    (os : ZkLink.OpensZ 𝔽 Dig (List UInt8))
    (hsch : Zk.schedule st.m = .ok (sch, tPad)) (_hms : st.merkle = MerkleScheme.hm96Sha512)
    (hseam : ZkTabSeam T st sch tPad) (htab : ZkTabLay T st sch) (hfold : FoldRealizes st.fold T.S)
    (hlen : extras.size = (Model.extraClaims T.S T.ptLocal mPts pts).length)
    (hext : ∀ (j : ℕ) (hj : j < (Model.extraClaims T.S T.ptLocal mPts pts).length),
      ClaimRel T.S.m extras[j]! (Model.extraClaims T.S T.ptLocal mPts pts)[j])
    (hdec : Zk.decode (shapeZK st sch extras) bytes = .ok zp) (hwf : RoundsWF rounds)
    (hrun : (repProg st sch tPad r.val extras zp).run (Flock.Transcript.init rounds dom) = .ok ((), tr))
    (hk : tr.k = rounds.size) (_hcap : cap₀ = capDig zp.proof.cap)
    (_ho : ∀ ℓ j, os.rows r.val ℓ j = openRow zp.proof.ligerito sch ℓ j)
    (_hs : ∀ ℓ j, os.salts r.val ℓ j = saltRow zp.proof.ligerito sch ℓ j) (hinner : os.inner r.val = innerOpensOf zp.inner)
    (pend : ZkLink.PendZ 𝔽 Dig (List UInt8))
    (hp : Plays fOf (ZkLink.repZCAt execArith hs512 T pts r) (repMsgsZ st sch (2 + extras.size) tPad zp)
      (rounds.toList.map (·.coins)) pend) :
    pend.inner (os.inner r.val) := by
  obtain ⟨pend', hp', hin⟩ :=
    zk_inner_rows T pts r st sch tPad extras bytes zp rounds dom tr hsch hseam htab hfold hlen hext hdec hwf hrun hk
  obtain rfl := plays_det hp hp'
  rw [hinner]
  exact hin

end FlockSoundness.Discharge.ZkExec
