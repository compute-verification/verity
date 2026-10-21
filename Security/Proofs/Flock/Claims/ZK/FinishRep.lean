import Proofs.Flock.Claims.ZK.FinishLig
import Proofs.Flock.Claims.ZK.FinishInner

/-!
# The firewall's rep in the analysis's terms

`Flock.Firewall.Finish.repFw` is the analysis's phases one for one, its copies held to them in `FinishCommit`,
`FinishPiop`, `FinishLig` and `FinishInner`. Here they are put together:

* `repFwC`: the firewall's rep in `FlockComplete`'s terms. `piopCut`'s zerocheck, lincheck and ring switch at the masked
  values `vals + h` and the switched vectors of the packed table, with no rank check (the frame's,
  `LeanRelease.FrameAdmits`); `FlockComplete.ligeritoPZk` at SHA-512 and `hm96-sha512/v1`; the session's replay and
  constraints; the stop where one fails on the pads; then `FlockComplete.innerP`.
* **`repFw_eq`**: at the firewall's commitment (`commitFw`) and a scheme of `hm96-sha512/v1`, at pads with
  `h_ab = h_a·h_b` and `w = x·y` on the layout's length, in a monad where a throw ends the run, `repFw` is `repFwC`.
  Where the stop fires, both stop with the same message; elsewhere no constraint fails on the pads, so the batched
  constraint holds at every coin (`batched_of_noFail`) and the firewall's inner proof is `FlockComplete.innerP`'s
  (`innerP_eq`).
* `evalL_combo_linAt`, `batched_of_noFail`: `Lin.combo` on the pads' length is the weighted sum of `failingOn`'s values
  of its forms (`linAt`), whatever their lengths, so where none fails on the pads the batched constraint holds, with no
  premise on the constraints' lengths.
* `tmw_throw_bind`, `throw_bind_sv`: a throw ends a run in the transcript monad over a monad where it does, as it does
  in `flock-firewall finish`'s own (`StateT Nat V`).
-/

namespace Flock.Firewall.Finish

open FlockLevel3 FlockSoundness.Discharge.Composed.ZerocheckMasked
open FlockSoundness.Discharge.Composed.Zerocheck (eq_of_of of_foldl_dot mul_sum')
open Transcript

/-! ## The batched constraint where no constraint fails -/

theorem outer_fold_linAt (ws : Array F128) (ls : Array Zk.Lin) (h : Array F128) :
    ∀ n, GF128.of ((List.range n).foldl (outerStep ws ls h.size) (0, Array.replicate h.size 0)).1 +
        ∑ k ∈ Finset.range h.size,
          GF128.of ((List.range n).foldl (outerStep ws ls h.size) (0, Array.replicate h.size 0)).2[k]! *
            GF128.of h[k]! =
      ∑ i ∈ Finset.range n, GF128.of ws[i]! * GF128.of (linAt ls[i]! h)
  | 0 => by
      simp only [List.range_zero, List.foldl_nil, Finset.range_zero, Finset.sum_empty]
      rw [← GF128.zero_eq, zero_add]
      refine Finset.sum_eq_zero fun k hk => ?_
      rw [get_replicate _ _ (Finset.mem_range.1 hk), ← GF128.zero_eq, zero_mul]
  | n + 1 => by
      have ih := outer_fold_linAt ws ls h n
      have h1 := size_outer ws ls h.size (List.range n) (0, Array.replicate h.size 0) (by simp)
      rw [List.range_succ, List.foldl_append, List.foldl_cons, List.foldl_nil]
      generalize hst : (List.range n).foldl (outerStep ws ls h.size) (0, Array.replicate h.size 0) = st at ih h1
      simp only [outerStep]
      have hent : ∀ k ∈ Finset.range h.size,
          GF128.of ((List.range (min h.size ls[n]!.a.size)).foldl (innerStep ws[n]! ls[n]!.a) st.2)[k]! *
              GF128.of h[k]! =
            GF128.of st.2[k]! * GF128.of h[k]! + GF128.of ws[n]! *
              (if k < min h.size ls[n]!.a.size then GF128.of ls[n]!.a[k]! * GF128.of h[k]! else 0) :=
        fun k hk => by
          rw [inner_fold _ _ _ h.size h1 _ (min_le_left _ _) k (Finset.mem_range.1 hk)]
          split_ifs
          · rw [of_add, of_mul]
            ring
          · ring
      have hlin : GF128.of (linAt ls[n]! h) = GF128.of ls[n]!.c +
          ∑ k ∈ Finset.range h.size,
            (if k < min h.size ls[n]!.a.size then GF128.of ls[n]!.a[k]! * GF128.of h[k]! else 0) := by
        rw [linAt, ← Array.foldl_toList, Array.toList_range, of_foldl_dot, ← Finset.sum_filter]
        congr 1
        refine Finset.sum_congr ?_ fun _ _ => rfl
        ext k
        simp only [Finset.mem_range, Finset.mem_filter, lt_min_iff]
        omega
      have hms : GF128.of ws[n]! * ∑ k ∈ Finset.range h.size,
            (if k < min h.size ls[n]!.a.size then GF128.of ls[n]!.a[k]! * GF128.of h[k]! else 0) =
          ∑ k ∈ Finset.range h.size, GF128.of ws[n]! *
            (if k < min h.size ls[n]!.a.size then GF128.of ls[n]!.a[k]! * GF128.of h[k]! else 0) :=
        mul_sum' _ _ _
      rw [Finset.sum_congr rfl hent, Finset.sum_add_distrib, ← hms, Finset.sum_range_succ, ← ih, hlin, of_add, of_mul]
      ring

/-- **`Lin.combo` on the pads' length** evaluates to the weighted sum of `failingOn`'s values of its forms, whatever
their lengths. -/
theorem evalL_combo_linAt (ws : Array F128) (ls : Array Zk.Lin) (h : Array F128) :
    GF128.of (evalL (Zk.Lin.combo ws ls h.size) h) =
      ∑ i ∈ Finset.range (min ws.size ls.size), GF128.of ws[i]! * GF128.of (linAt ls[i]! h) := by
  rw [of_evalL_range _ h h.size (size_combo ws ls h.size), combo_eq]
  dsimp only
  exact outer_fold_linAt ws ls h _

/-- **The batched constraint holds where no constraint fails on the pads**, at pads with `h_ab = h_a·h_b` and
`w = x·y` on the layout's length, at every sacrifice and batching coin, whatever the constraints' lengths. -/
theorem batched_of_noFail (lay : Zk.PadLayout) {h : Array F128} {cons : Array (String × Zk.Lin)}
    (hs : h.size = lay.len) (hab : h[lay.hab]! = h[lay.fa]! * h[lay.fb]!) (htw : h[lay.tw]! = h[lay.tx]! * h[lay.ty]!)
    (hf : failingOn cons h = #[]) (eps : F128) (gammas : Array F128) :
    Batched lay h cons eps gammas = true := by
  have hz : ∀ q ∈ cons ++ Zk.triple lay eps (eps * h[lay.fa]! + h[lay.tx]!) (h[lay.fb]! + h[lay.ty]!),
      linAt q.2 h = 0 := by
    intro q hq
    rcases Array.mem_append.1 hq with hq | hq
    · unfold failingOn at hf
      rw [Array.map_eq_empty_iff, Array.filter_eq_empty_iff] at hf
      simpa [bne] using hf q hq
    · rw [linAt_eq]
      exact (FlockComplete.triple_holds lay h hs hab htw eps q hq).2
  have e : GF128.of (evalL (Zk.Lin.combo gammas
      ((cons ++ Zk.triple lay eps (eps * h[lay.fa]! + h[lay.tx]!) (h[lay.fb]! + h[lay.ty]!)).map (·.2)) h.size) h) =
      GF128.of 0 := by
    rw [evalL_combo_linAt, ← GF128.zero_eq]
    refine Finset.sum_eq_zero fun i hi => ?_
    have hi' := (lt_min_iff.1 (Finset.mem_range.1 hi)).2
    obtain ⟨q, hq, hq2⟩ := Array.mem_map.1 (Array.getElem_mem hi')
    rw [getElem!_pos ((cons ++ Zk.triple lay eps (eps * h[lay.fa]! + h[lay.tx]!) (h[lay.fb]! + h[lay.ty]!)).map (·.2))
      i hi', ← hq2, hz q hq, ← GF128.zero_eq, mul_zero]
  have e' := eq_of_of e
  rw [hs] at e'
  unfold evalL at e'
  simp only [Batched]
  rw [e']
  rfl

/-! ## A throw ends the run -/

section

variable {M : Type → Type} [Monad M] [MonadExceptOf String M]

/-- **A throw ends a run of the transcript monad** over a monad where a throw ends the run. -/
theorem tmw_throw_bind (hthrow : ∀ {α β : Type} (e : String) (f : α → M β), (throw e : M α) >>= f = throw e)
    {α β : Type} (e : String) (f : α → TMW M β) : (throw e : TMW M α) >>= f = throw e := by
  apply StateT.ext
  intro s
  rw [StateT.run_bind]
  have h1 : StateT.run (throw e : TMW M α) s = ((throw e : M α) >>= fun a => pure (a, s)) := rfl
  have h2 : StateT.run (throw e : TMW M β) s = ((throw e : M β) >>= fun a => pure (a, s)) := rfl
  simp only [h1, h2, hthrow]

end

/-- **A throw ends a run of `flock-firewall finish`'s own monad** (`proofFile`'s `StateT Nat V`). -/
theorem throw_bind_sv {α β : Type} (e : String) (f : α → StateT ℕ V β) :
    (throw e : StateT ℕ V α) >>= f = throw e := rfl

/-! ## The rep -/

section

variable {M : Type → Type} [Monad M] [LawfulMonad M] [MonadExceptOf String M] [MonadLiftT V M]

/-- **The firewall's rep in the analysis's terms**: `piopCut`'s zerocheck, lincheck and ring switch at the masked values
`vals + h` and the switched vectors of `g`, with no rank check; `FlockComplete.ligeritoPZk` at SHA-512 and
`hm96-sha512/v1`; the session's replay and constraints; the stop where one fails on the pads; then
`FlockComplete.innerP`. -/
def repFwC (io : FlockComplete.Asks M) (hs : Hashes M) (st : Setup) (sch : Flock.Schedule) (tPad rep : ℕ)
    (dom : ByteArray) (g : Array F128) (cm0 : FlockComplete.Commit) (extras : Array Claim)
    (pads : FlockComplete.PadsZk) (ic : FlockComplete.InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray) : TMW M Zk.ZkProof := do
  let lay := Zk.PadLayout.new st.m st.kLog sch.k0
  let mk : ℕ → F128 := fun i => vals[i]! + pads.h[i]!
  let (zp, zc) ← FlockComplete.zerocheckCut io st.m st.digest (FlockComplete.flat cm0.cap) ic.root lay mk
  let (lp, lc) ← FlockComplete.lincheckCut io st.fold st.m lay mk zc
  let claims := abcClaims st.m st.kLog zc lc ++ extras
  let (ss, T, sws) ← FlockComplete.ringSwitchCut io (fun k => switchVec g claims[k]!) claims
  let (lg, zl, rec) ← FlockComplete.ligeritoPZk io (Hashes.ofSha512 Sha512.hash) MerkleScheme.hm96Sha512 sch T cm0 g
    (FlockComplete.basisVals sws (st.m - 7)) lay pads.h rep tPad extra lanePad salt (fun _ _ => 0)
  let p : Proof := { cap := cm0.cap, params := pcsParams st.merkle sch, zerocheck := zp, lincheck := lp,
                     ringSwitches := ss, ligerito := lg }
  let tr' ← get
  let (rp, _) ← StateT.lift (monadLift ((Zk.replay st lay p ic.root).run (Transcript.init tr'.rounds dom)) :
    M (Zk.Replayed × Transcript))
  let ccAb := Zk.claimCheck rp.claims[0]! ss[0]!
  let ccC := Zk.claimCheck rp.claims[1]! ss[1]!
  let lgc ← StateT.lift (monadLift (Zk.ligeritoConstraints lay rec lg zl) : M (Array (String × Zk.Lin)))
  let cons := rp.cons ++ #[("ring switch ab", rp.abValue.addConst ccAb), ("ring switch c", rp.cValue.addConst ccC)] ++ lgc
  let fails := failingOn cons pads.h
  if !fails.isEmpty then throw s!"{Frame.STOP}the constraints {", ".intercalate fails.toList} fail on the rep's pads"
  let inner ← FlockComplete.innerP io hs lay pads ic zp.finalC cons
  return { proof := p, level0 := zl, inner }

theorem PadsZk.toC_h (p : PadsZk) : p.toC.h = p.h := rfl

theorem flat_eq (cap : Array ByteArray) : flat cap = FlockComplete.flat cap := rfl

theorem InnerCommit.toC_root (ic : InnerCommit) : ic.toC.root = ic.root := rfl

/-- **The firewall's rep is `repFwC`** at its own commitment and `hm96-sha512/v1`, at pads with `h_ab = h_a·h_b` and
`w = x·y` on the layout's length, in a monad where a throw ends the run. -/
theorem repFw_eq (io : Asks M) (hs : Hashes M) (st : Setup) (hm : st.merkle = MerkleScheme.hm96Sha512)
    (sch : Flock.Schedule) (tPad rep : ℕ) (dom : ByteArray) (g : Array F128) (cm0 : Commit) (extras : Array Claim)
    (pads : PadsZk) (ic : InnerCommit) (vals : Array F128) (extra lanePad : ℕ → ℕ → F128)
    (salt : ℕ → ℕ → ByteArray)
    (hps : pads.h.size = (Zk.PadLayout.new st.m st.kLog sch.k0).len)
    (hab : pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).hab]! =
      pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fa]! * pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).fb]!)
    (htw : pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).tw]! =
      pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).tx]! * pads.h[(Zk.PadLayout.new st.m st.kLog sch.k0).ty]!)
    (hthrow : ∀ {α β : Type} (e : String) (f : α → M β), (throw e : M α) >>= f = throw e) :
    repFw io hs (fun lv g s => pure (commitFw lv g s)) st sch tPad rep dom g cm0 extras pads ic vals extra lanePad salt =
      repFwC io.toC hs st sch tPad rep dom g cm0.toC extras pads.toC ic.toC vals extra lanePad salt := by
  unfold repFw repCons repPiop repOpening repFwC
  simp only [hm, zerocheckFw_eq, lincheckFw_eq, ringSwitchP_eq, ligeritoPZk_fw, basisVals_eq, PadsZk.toC_h,
    InnerCommit.toC_root, Commit.toC_cap, flat_eq, bind_assoc]
  refine bind_congr fun ⟨zp, zc⟩ => ?_
  try simp only [bind_assoc]
  refine bind_congr fun ⟨lp, lc⟩ => ?_
  try simp only [pure_bind, bind_assoc]
  refine bind_congr fun ⟨ss, T, sws⟩ => ?_
  try simp only [bind_assoc]
  refine bind_congr fun ⟨lg, zl, rec⟩ => ?_
  try simp only [pure_bind, bind_assoc]
  refine bind_congr fun tr' => ?_
  refine bind_congr fun ⟨rp, _⟩ => ?_
  try simp only [bind_assoc]
  refine bind_congr fun lgc => ?_
  try simp only [pure_bind]
  try dsimp only
  split
  · rw [tmw_throw_bind hthrow, tmw_throw_bind hthrow]
  · rename_i hf
    rw [innerP_eq io hs _ pads ic _ _ (batched_of_noFail _ hps hab htw (by simpa using hf))]

end

end Flock.Firewall.Finish
