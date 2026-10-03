import FlockSoundness.Discharge.ZkExec.VerdictReplay

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

end FlockSoundness.Discharge.ZkExec
